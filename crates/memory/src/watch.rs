//! Write tracking for the GPU caches: the [`WriteWatch`] seam between the GPU's back thread and
//! the kernel, and the [`PageRange`] it speaks in.
//!
//! The design is the execution model's "Write tracking for the GPU caches" and gpu-protocol's
//! "Learning of CPU writes" (approved by the owner, 2026-11-12, decision log "Approved specs
//! amended for the graphics designs"). The GPU crate sits above the kernel in the dependency
//! order, so neither calls the other: both hold an `Arc<WriteWatch>`.
//!
//! - The back thread **requests** protection for a range ([`WriteWatch::request`]) and gets a
//!   [`WatchTicket`]. It cannot apply the protection itself: a stage-1 change needs a TLB flush
//!   from inside the guest.
//! - The kernel, at its next entry on any core, sees [`WriteWatch::has_requests`] (one atomic
//!   load), takes the waiting requests ([`WriteWatch::take_requests`]), write-protects their
//!   pages and flushes, then reports them applied ([`WriteWatch::arm`]). From then on the pages
//!   are *watched*.
//! - On a write fault, the kernel asks [`WriteWatch::record_fault`]. If the page is watched it
//!   stops being watched, is added to the dirty log, and the answer is `true`: the kernel
//!   restores write access and resumes. A page faults once per tracking period; tracking it again
//!   takes a new request.
//! - The back thread asks [`WriteWatch::armed`] whether a request has taken effect (until then it
//!   treats the range as untracked and reads it again at each use), and drains the dirty log
//!   with [`WriteWatch::take_dirty`] before each draw, dispatch and copy. With nothing dirty the
//!   drain is one atomic load and takes no lock.
//!
//! The kernel half (applying protections, the TLB flush, the fault path) is a kernel slice
//! (`docs/specs/kernel.md`, "the kernel half of `WriteWatch`"). With the `test-support` feature
//! this module also exports [`FakeWatchKernel`], which plays the kernel's part in other crates'
//! tests.
//!
//! What the specs leave open, decided here (**not** from a reference; there is none for this
//! seam, it is new):
//!
//! - [`WriteWatch::armed`] answers `true` once the request **and every request issued before
//!   it** are armed. A kernel that arms two batches out of order (two cores) makes the later
//!   tickets read as unarmed until the earlier batch is armed too: the back thread then reads
//!   those ranges again at each use for a little longer, which is safe. Pinned by
//!   `tests::armed_waits_for_every_earlier_request`.
//! - Watched pages and dirty pages are each kept as a set of disjoint, coalesced ranges, so the
//!   memory either takes grows with the number of requests and faults, not with the number of
//!   pages they cover. [`WriteWatch::take_dirty`] reports the dirty pages as coalesced ranges in
//!   ascending address order.

mod page_set;

#[cfg(any(test, feature = "test-support"))]
mod fake;

#[cfg(test)]
mod tests;

use std::collections::{BTreeMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard, PoisonError};

use crate::translation::{MapError, PAGE_SIZE, check_page_range};

#[cfg(any(test, feature = "test-support"))]
pub use fake::FakeWatchKernel;

use page_set::PageSet;

/// A page-aligned, non-empty range of guest virtual memory.
///
/// Its end is always representable: `start + size` never overflows, so the last page of the
/// 64-bit space cannot be part of a range.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PageRange {
    start: u64,
    size: u64,
}

impl PageRange {
    /// Describes `size` bytes of guest virtual memory starting at `start`.
    ///
    /// Checked in this order: `start` is a multiple of [`PAGE_SIZE`]
    /// ([`MapError::MisalignedAddress`]), `size` is ([`MapError::MisalignedSize`]), `size` is
    /// non-zero ([`MapError::Empty`]), and `start + size` does not overflow
    /// ([`MapError::VirtualRange`]).
    pub fn new(start: u64, size: u64) -> Result<Self, MapError> {
        check_page_range(start, size)?;
        if start.checked_add(size).is_none() {
            return Err(MapError::VirtualRange { va: start, size });
        }
        Ok(Self { start, size })
    }

    /// The smallest range of whole pages that holds the `len` bytes starting at `va`.
    ///
    /// `len` must be non-zero ([`MapError::Empty`]), and the end of the last page touched must be
    /// representable ([`MapError::VirtualRange`], naming `va` and `len`).
    pub fn covering(va: u64, len: u64) -> Result<Self, MapError> {
        if len == 0 {
            return Err(MapError::Empty);
        }
        let out_of_range = MapError::VirtualRange { va, size: len };
        let last_byte = va.checked_add(len - 1).ok_or(out_of_range)?;
        let start = va - va % PAGE_SIZE;
        let end = (last_byte - last_byte % PAGE_SIZE)
            .checked_add(PAGE_SIZE)
            .ok_or(out_of_range)?;
        Ok(Self {
            start,
            size: end - start,
        })
    }

    /// The first address of the range.
    pub const fn start(self) -> u64 {
        self.start
    }

    /// The length of the range in bytes: a non-zero multiple of [`PAGE_SIZE`].
    pub const fn size(self) -> u64 {
        self.size
    }

    /// One past the last address of the range. Never overflows: the constructors check it.
    pub const fn end(self) -> u64 {
        self.start.wrapping_add(self.size)
    }

    /// Whether `va` is inside the range.
    pub const fn contains(self, va: u64) -> bool {
        self.start <= va && va < self.end()
    }
}

/// A request for write tracking, as [`WriteWatch::request`] numbered it.
///
/// Tickets are numbered from zero in the order requests were made. A ticket only means something
/// to the watch that issued it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct WatchTicket(u64);

impl WatchTicket {
    /// The ticket's number.
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// A request the kernel has to apply: protect `range` read-only, then report it with
/// [`WriteWatch::arm`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WatchRequest {
    ticket: WatchTicket,
    range: PageRange,
}

impl WatchRequest {
    /// The ticket the back thread holds for this request.
    pub const fn ticket(self) -> WatchTicket {
        self.ticket
    }

    /// The pages to write-protect.
    pub const fn range(self) -> PageRange {
        self.range
    }
}

/// Why [`WriteWatch::arm`] refused a batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum WatchError {
    /// The request was not taken with [`WriteWatch::take_requests`], is already armed, or came
    /// from another watch.
    #[error("watch request {ticket} is not waiting to be armed")]
    NotTaken {
        /// The request's ticket number.
        ticket: u64,
    },
}

/// The write-tracking seam shared by the GPU's back thread and the kernel (see the module docs).
///
/// Shared as an `Arc<WriteWatch>`. Every method takes `&self`; the state sits behind one lock,
/// with atomic flags in front of it so the two hot checks ([`Self::has_requests`] at each kernel
/// entry and an empty [`Self::take_dirty`] before each draw) take no lock.
#[derive(Debug, Default)]
pub struct WriteWatch {
    /// Set while requests wait for the kernel to take them.
    requests_waiting: AtomicBool,
    /// Set while the dirty log holds pages.
    dirty_waiting: AtomicBool,
    /// Every ticket numbered below this is armed.
    armed_below: AtomicU64,
    state: Mutex<State>,
}

#[derive(Debug, Default)]
struct State {
    /// The number the next request gets.
    next: u64,
    /// Requests the kernel has not taken yet, in ticket order.
    waiting: VecDeque<WatchRequest>,
    /// Requests the kernel took and has not armed yet, by ticket number.
    taken: BTreeMap<u64, PageRange>,
    /// Armed pages that have not been written since.
    watched: PageSet,
    /// Watched pages the guest wrote, not yet drained.
    dirty: PageSet,
}

impl State {
    /// The lowest ticket number not yet armed.
    fn armed_below(&self) -> u64 {
        let waiting = self.waiting.front().map_or(self.next, |r| r.ticket.0);
        let taken = self.taken.keys().next().copied().unwrap_or(self.next);
        waiting.min(taken)
    }
}

impl WriteWatch {
    /// An empty watch: no requests, nothing watched, nothing dirty.
    pub fn new() -> Self {
        Self::default()
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Back thread: asks for `range` to be write-protected, so the first guest write to each of
    /// its pages is recorded as dirty. The protection takes effect when the kernel arms it
    /// ([`Self::armed`]).
    pub fn request(&self, range: PageRange) -> WatchTicket {
        let mut state = self.lock();
        let ticket = WatchTicket(state.next);
        // 2^64 requests at one a nanosecond take 584 years, so the count cannot wrap in practice;
        // wrapping keeps the method infallible.
        state.next = state.next.wrapping_add(1);
        state.waiting.push_back(WatchRequest { ticket, range });
        self.requests_waiting.store(true, Ordering::Release);
        self.armed_below
            .store(state.armed_below(), Ordering::Release);
        ticket
    }

    /// Back thread: whether the protection `ticket` asked for has taken effect, and so has the
    /// protection of every request issued before it (see the module docs).
    pub fn armed(&self, ticket: WatchTicket) -> bool {
        ticket.0 < self.armed_below.load(Ordering::Acquire)
    }

    /// Back thread: moves every dirty page into `out`, as coalesced ranges in ascending address
    /// order, and empties the dirty log. Appends to `out`; does not clear it.
    pub fn take_dirty(&self, out: &mut Vec<PageRange>) {
        if !self.dirty_waiting.load(Ordering::Acquire) {
            return;
        }
        let mut state = self.lock();
        state.dirty.drain_into(out);
        self.dirty_waiting.store(false, Ordering::Release);
    }

    /// Kernel: whether requests wait to be taken. One atomic load, for every kernel entry.
    pub fn has_requests(&self) -> bool {
        self.requests_waiting.load(Ordering::Acquire)
    }

    /// Kernel: moves every waiting request into `out`, in ticket order. The kernel protects
    /// their pages, flushes, and then calls [`Self::arm`] with them. Appends to `out`; does not
    /// clear it.
    pub fn take_requests(&self, out: &mut Vec<WatchRequest>) {
        let mut state = self.lock();
        let state = &mut *state;
        for request in state.waiting.drain(..) {
            state.taken.insert(request.ticket.0, request.range);
            out.push(request);
        }
        self.requests_waiting.store(false, Ordering::Release);
    }

    /// Kernel: reports that the pages of `applied` are write-protected and flushed. Their pages
    /// become watched.
    ///
    /// All or nothing: every request must have been taken with [`Self::take_requests`] and not
    /// armed yet, or nothing changes and the first one that was not is named
    /// ([`WatchError::NotTaken`]).
    pub fn arm(&self, applied: &[WatchRequest]) -> Result<(), WatchError> {
        let mut state = self.lock();
        if let Some(stray) = applied
            .iter()
            .find(|r| state.taken.get(&r.ticket.0) != Some(&r.range))
        {
            return Err(WatchError::NotTaken {
                ticket: stray.ticket.0,
            });
        }
        for request in applied {
            if state.taken.remove(&request.ticket.0).is_some() {
                state.watched.insert(request.range);
            }
        }
        self.armed_below
            .store(state.armed_below(), Ordering::Release);
        Ok(())
    }

    /// Kernel: a guest write faulted at `va`. If its page is watched, the page stops being
    /// watched, joins the dirty log, and the answer is `true`: restore write access and resume.
    /// `false` means the fault is not the watch's.
    pub fn record_fault(&self, va: u64) -> bool {
        let page = va - va % PAGE_SIZE;
        let mut state = self.lock();
        if !state.watched.remove_page(page) {
            return false;
        }
        // A watched page is inside a `PageRange`, whose end is representable, so this succeeds.
        if let Ok(range) = PageRange::new(page, PAGE_SIZE) {
            state.dirty.insert(range);
            self.dirty_waiting.store(true, Ordering::Release);
        }
        true
    }
}
