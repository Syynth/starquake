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
//!   are *watched*. The kernel must arm every request it takes: it protects the pages of the
//!   range that are mapped (an unmapped page never faults, so it needs nothing) and arms the
//!   request even when some or all of its pages are unmapped. There is no way to settle a taken
//!   request other than arming it, and one that is never armed holds back [`WriteWatch::armed`]
//!   for every later ticket (below).
//! - On a write fault, the kernel first checks the guest's own permission for the page. A write
//!   the guest's permission does not allow (a read-only mapping, a code page) is the guest's
//!   fault: the kernel delivers it as a permission fault and does not call the watch, which stays
//!   unchanged. Only for a write the guest's permission allows does the kernel ask
//!   [`WriteWatch::record_fault`]. If the page is watched, or belongs to a request taken but not
//!   yet armed (the kernel has protected it, so a store on another core can fault before
//!   [`WriteWatch::arm`] runs), it stops being watched, leaves every such pending request, is
//!   added to the dirty log, and the answer is `true`: the kernel restores the guest's own
//!   permission for the page (not plain write access) and resumes. A page faults once per
//!   tracking period; tracking it again takes a new request.
//! - Because a fault on a taken request's page takes the page out of that request, the kernel
//!   must not let the two cross: it takes requests and protects their pages under the same lock
//!   its fault path holds while it calls [`WriteWatch::record_fault`] and restores the
//!   permission. Otherwise a fault recorded between the take and the protect would leave the
//!   page protected but not watched, and its next write would read as not the watch's.
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
//!   it** are armed. A later ticket reads as unarmed (a false negative) until every earlier
//!   request has been taken and armed; it ends at that moment and not before. A kernel that arms
//!   two batches out of order (two cores) causes one until the earlier batch is armed; a taken
//!   request that is never armed would cause one for every later ticket for the rest of the
//!   session, which is why the kernel must arm every request it takes (above). During a false
//!   negative the back thread reads those ranges again at each use, which is safe. Pinned by
//!   `tests::armed_waits_for_every_earlier_request` and
//!   `tests::a_taken_request_never_armed_holds_back_every_later_ticket`.
//! - A fault on a page of a request taken but not yet armed is the watch's: the page is recorded
//!   dirty and leaves the pending request, so arming the request does not watch it (its
//!   permission was restored), while [`WriteWatch::armed`] still changes only on
//!   [`WriteWatch::arm`]. Pinned by `tests::a_write_between_protect_and_arm_is_the_watchs`.
//! - A ticket carries no identity of the watch that issued it. [`WriteWatch::arm`] checks only
//!   that each ticket is taken and not yet armed on this watch, with the same range, so a request
//!   taken from another watch that matches both is accepted. Pinned by
//!   `tests::arm_checks_the_ticket_and_range_not_which_watch_issued_them`.
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
    /// This watch has no request with the ticket's number that was taken with
    /// [`WriteWatch::take_requests`] and not yet armed, with the same range. That is the whole
    /// check: a ticket does not say which watch issued it.
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
    taken: BTreeMap<u64, Taken>,
    /// Armed pages that have not been written since.
    watched: PageSet,
    /// Watched pages the guest wrote, not yet drained.
    dirty: PageSet,
}

/// A request the kernel took and has not armed yet.
#[derive(Debug)]
struct Taken {
    /// The pages the request covers.
    range: PageRange,
    /// Pages of `range` the guest wrote since the kernel took the request. Their permission was
    /// restored, so arming the request does not watch them.
    written: PageSet,
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
    /// their mapped pages, flushes, and then calls [`Self::arm`] with every one of them, mapped
    /// or not (see the module docs: a taken request that is never armed holds back
    /// [`Self::armed`] for every later ticket). From the take on, a fault on their pages is the
    /// watch's ([`Self::record_fault`]). Appends to `out`; does not clear it.
    pub fn take_requests(&self, out: &mut Vec<WatchRequest>) {
        let mut state = self.lock();
        let state = &mut *state;
        for request in state.waiting.drain(..) {
            state.taken.insert(
                request.ticket.0,
                Taken {
                    range: request.range,
                    written: PageSet::default(),
                },
            );
            out.push(request);
        }
        self.requests_waiting.store(false, Ordering::Release);
    }

    /// Kernel: reports that the pages of `applied` are write-protected and flushed. Their pages
    /// become watched, except those the guest wrote since the take (their permission was
    /// restored, [`Self::record_fault`]).
    ///
    /// All or nothing: every request must name a ticket taken on this watch with
    /// [`Self::take_requests`] and not armed yet, with the same range, or nothing changes and the
    /// first one that does not is named ([`WatchError::NotTaken`]).
    pub fn arm(&self, applied: &[WatchRequest]) -> Result<(), WatchError> {
        let mut state = self.lock();
        if let Some(stray) = applied
            .iter()
            .find(|r| state.taken.get(&r.ticket.0).map(|t| t.range) != Some(r.range))
        {
            return Err(WatchError::NotTaken {
                ticket: stray.ticket.0,
            });
        }
        let state = &mut *state;
        for request in applied {
            // A request listed twice is removed and its pages watched the first time; the second
            // finds nothing left to arm.
            if let Some(taken) = state.taken.remove(&request.ticket.0) {
                state.watched.insert_except(taken.range, &taken.written);
            }
        }
        self.armed_below
            .store(state.armed_below(), Ordering::Release);
        Ok(())
    }

    /// Kernel: a guest write faulted at `va`, and the guest's own permission for the page allows
    /// the write. The kernel calls this only for such writes: a write the guest's permission does
    /// not allow is delivered to the guest as a permission fault and leaves the watch unchanged.
    ///
    /// The fault is the watch's when the page is watched, or is a page of a request taken but not
    /// yet armed that the guest has not written since the take. Then the page stops being
    /// watched, leaves every such pending request (arming one does not watch it), joins the dirty
    /// log, and the answer is `true`: restore the guest's own permission for the page and resume.
    /// `false` means the fault is not the watch's.
    ///
    /// Walks the requests taken and not yet armed, so a fault costs time in proportion to how
    /// many the kernel holds between [`Self::take_requests`] and [`Self::arm`].
    pub fn record_fault(&self, va: u64) -> bool {
        let page = va - va % PAGE_SIZE;
        // The last page of the 64-bit space is never part of a `PageRange`, so it is neither
        // watched nor in a taken request.
        let Ok(range) = PageRange::new(page, PAGE_SIZE) else {
            return false;
        };
        let mut state = self.lock();
        let state = &mut *state;
        let mut ours = state.watched.remove_page(page);
        for taken in state.taken.values_mut() {
            if taken.range.contains(page) && !taken.written.contains(page) {
                taken.written.insert(range);
                ours = true;
            }
        }
        if ours {
            state.dirty.insert(range);
            self.dirty_waiting.store(true, Ordering::Release);
        }
        ours
    }
}
