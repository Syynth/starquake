//! A [`Mapper`] for tests: it records what it is asked and can be told to refuse.
//!
//! Exported behind the `test-support` feature (test strategy spec, "Shared test support": each
//! crate exports its own fakes), for this crate's tests and for crates that run the kernel
//! without a translation behind it.
//!
//! It backs nothing, with one exception: shared memory. [`Mapper::allocate_shared`] hands out
//! pages of a small guest-physical memory of its own ([`SHARED_BASE`], [`SHARED_CAPACITY`]
//! bytes), and [`MapperLog::shared_view`] reads and writes guest addresses that
//! [`Mapper::map_shared`] mapped, through the same pages, so a test can see a host write
//! through a guest mapping and the reverse.
//!
//! # How it differs from the real translation
//!
//! `starquake-core`'s `SpaceMapper` is the translation the kernel runs on. This one keeps its
//! contract for shared memory with these differences:
//!
//! - **Running out**: more than the room left ([`SHARED_CAPACITY`] in all) is
//!   [`MapperError::OutOfMemory`], as the real one's is, but in words of its own.
//! - **Released pages are never handed out again** ([`Mapper::release_shared`]): the call is
//!   recorded ([`MapperCall::ReleaseShared`]) and checked, and the pages stay used. The real
//!   one hands them out again once every flush owed by their unmappings has run; this one has
//!   no flushes to wait for, and the tests that use it observe the call, not the reuse.
//!   A release is refused, with nothing recorded, unless the pages are exactly a run it
//!   handed out and not released yet, and no page of them is mapped anywhere, as the real one
//!   refuses. It tracks the mapped pages one by one, as the real one does: what
//!   [`Mapper::map_shared`] maps, what [`Mapper::map_alias`] maps of those pages again, and
//!   what [`Mapper::unmap`] takes away, a page at a time, so a mapping partly unmapped or an
//!   alias left after the original went still holds the pages (tested against the real one
//!   by `the_recording_mapper_gives_back_as_this_one_does_but_never_hands_out_again` in
//!   `crates/core/src/space/tests/shared.rs`). Those two checks come before the refusal
//!   switches ([`MapperLog::refuse`], [`MapperLog::refuse_calls`]), so a release they refuse
//!   is not counted as a call.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex, PoisonError};

use starquake_memory::{BackingRange, Fault, FlatMemory, GuestMemory, PAGE_SIZE, Perms};

use super::mapper::{Mapper, MapperError};
use super::shared::{SharedPages, span_within};
use super::space::DeviceShareToken;
use crate::id::ProcessId;

/// Where the guest-physical memory that backs shared memory starts.
pub const SHARED_BASE: u64 = 0x10_0000_0000;
/// How many bytes of shared memory a [`RecordingMapper`] can hand out in all: 64 pages.
pub const SHARED_CAPACITY: u64 = 64 * PAGE_SIZE;

/// [`PAGE_SIZE`] as a step for walking pages.
const PAGE_STEP: usize = 1 << PAGE_SIZE.trailing_zeros();

/// One call a [`RecordingMapper`] received.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MapperCall {
    /// [`Mapper::map`].
    Map {
        /// The first address.
        address: u64,
        /// The size in bytes.
        size: u64,
        /// What guest code may do there.
        perms: Perms,
    },
    /// [`Mapper::unmap`].
    Unmap {
        /// The first address.
        address: u64,
        /// The size in bytes.
        size: u64,
    },
    /// [`Mapper::protect`].
    Protect {
        /// The first address.
        address: u64,
        /// The size in bytes.
        size: u64,
        /// What guest code may do there from now on.
        perms: Perms,
    },
    /// [`Mapper::allocate_shared`].
    AllocateShared {
        /// The pages handed out.
        backing: BackingRange,
    },
    /// [`Mapper::release_shared`].
    ReleaseShared {
        /// The pages given back.
        backing: BackingRange,
    },
    /// [`Mapper::map_shared`].
    MapShared {
        /// The first address.
        address: u64,
        /// The pages mapped there.
        backing: BackingRange,
        /// What guest code may do there.
        perms: Perms,
    },
    /// [`Mapper::map_alias`].
    MapAlias {
        /// The first address of the alias.
        address: u64,
        /// The first address of the pages it maps again.
        source: u64,
        /// The size in bytes.
        size: u64,
        /// What guest code may do through the alias.
        perms: Perms,
    },
}

#[derive(Debug)]
struct State {
    calls: Vec<MapperCall>,
    refuse: bool,
    /// How many calls were made since [`MapperLog::refuse_calls`] was last told which to
    /// refuse, accepted or refused.
    counted: usize,
    /// Which of the counted calls, from zero, are refused.
    refuse_calls: BTreeSet<usize>,
    /// The guest-physical memory shared memory is backed from.
    physical: Arc<FlatMemory>,
    /// The first byte of it not handed out yet.
    next: u64,
    /// What `map_shared` mapped and nothing has unmapped since: first address to the pages.
    shared: BTreeMap<u64, BackingRange>,
    /// Every virtual page that maps a page of shared memory (through `map_shared` or a
    /// `map_alias` of one) and that nothing has unmapped since, to that guest-physical page.
    mapped_pages: BTreeMap<u64, u64>,
    /// The runs `allocate_shared` handed out and `release_shared` has not taken back: first
    /// address to one past the last.
    handed_out: BTreeMap<u64, u64>,
}

impl Default for State {
    fn default() -> Self {
        let capacity = usize::try_from(SHARED_CAPACITY).unwrap_or(usize::MAX);
        Self {
            calls: Vec::new(),
            refuse: false,
            counted: 0,
            refuse_calls: BTreeSet::new(),
            physical: Arc::new(FlatMemory::new(SHARED_BASE, capacity)),
            next: SHARED_BASE,
            shared: BTreeMap::new(),
            mapped_pages: BTreeMap::new(),
            handed_out: BTreeMap::new(),
        }
    }
}

/// The test's end of a [`RecordingMapper`]: what it was asked, and the switch that makes it
/// refuse.
#[derive(Debug, Clone, Default)]
pub struct MapperLog(Arc<Mutex<State>>);

impl MapperLog {
    fn with<R>(&self, f: impl FnOnce(&mut State) -> R) -> R {
        f(&mut self.0.lock().unwrap_or_else(PoisonError::into_inner))
    }

    /// The calls received since the last [`Self::take`], in order, and forgets them.
    #[must_use]
    pub fn take(&self) -> Vec<MapperCall> {
        self.with(|state| std::mem::take(&mut state.calls))
    }

    /// Makes every call from now on fail (`true`) or succeed (`false`). A refused call is not
    /// recorded.
    pub fn refuse(&self, refuse: bool) {
        self.with(|state| state.refuse = refuse);
    }

    /// Makes exactly the calls numbered `calls` fail, counting every call from now on from
    /// zero, accepted or refused: a fault in the middle of an operation. The calls not named
    /// are accepted unless [`Self::refuse`] says otherwise. A refused call is not recorded.
    pub fn refuse_calls(&self, calls: &[usize]) {
        self.with(|state| {
            state.counted = 0;
            state.refuse_calls = calls.iter().copied().collect();
        });
    }

    /// Guest memory as guest code sees the shared memory mapped through this log's mapper: an
    /// access that lies wholly inside one mapping reads and writes its pages, and any other is a
    /// [`Fault`] at its first address.
    #[must_use]
    pub fn shared_view(&self) -> SharedView {
        SharedView(self.clone())
    }
}

/// Shared memory pages of `size` bytes in host memory of their own, with no address space
/// behind them: what a fake kernel that has no [`Mapper`] hands a service. `None` unless `size`
/// is whole pages, not zero, and at most [`SHARED_CAPACITY`]: the pages are allocated at once,
/// and the size can be a guest's, so it is capped as a [`RecordingMapper`]'s room is.
#[must_use]
pub fn standalone_shared_pages(size: u64) -> Option<SharedPages> {
    if size > SHARED_CAPACITY {
        return None;
    }
    let backing = BackingRange::new(0, size).ok()?;
    let memory = FlatMemory::new(0, usize::try_from(size).ok()?);
    Some(SharedPages::new(backing, Arc::new(memory)))
}

/// A device-share token naming `process` and `size` bytes at `address`, with no lock behind
/// it: what a fake kernel that keeps its own locks hands a service, and what a test plays a
/// saved token with. It exists only for tests (the `test-support` feature), and it is the only
/// way to make a token with no check at all. Outside tests, the one way to get a token from
/// raw numbers is `crate::dispatch::Dispatcher::reclaim_device_share`, which is meant only for
/// a snapshot restore and makes one only for a range the kernel holds a live lock on. The
/// kernel's own unlock refuses a token whose lock it did not take
/// (`crate::dispatch::Dispatcher::unlock_device_share`).
#[must_use]
pub const fn device_share_token(process: ProcessId, address: u64, size: u64) -> DeviceShareToken {
    DeviceShareToken::new(process, address, size)
}

/// See [`MapperLog::shared_view`].
#[derive(Debug, Clone)]
pub struct SharedView(MapperLog);

impl SharedView {
    /// The guest-physical address of `len` bytes at `va`, if one mapping holds them all.
    fn resolve(&self, va: u64, len: usize) -> Result<(Arc<FlatMemory>, u64), Fault> {
        let fault = Fault { address: va };
        self.0.with(|state| {
            let (start, backing) = state.shared.range(..=va).next_back().ok_or(fault)?;
            // `start <= va`: the mapping is the last one that starts at or before `va`.
            let offset = va.checked_sub(*start).ok_or(fault)?;
            let at = span_within(backing.base(), backing.size(), offset, len).ok_or(fault)?;
            Ok((Arc::clone(&state.physical), at))
        })
    }
}

impl GuestMemory for SharedView {
    fn read(&self, va: u64, buf: &mut [u8]) -> Result<(), Fault> {
        let (physical, at) = self.resolve(va, buf.len())?;
        physical.read(at, buf).map_err(|_| Fault { address: va })
    }

    fn write(&self, va: u64, data: &[u8]) -> Result<(), Fault> {
        let (physical, at) = self.resolve(va, data.len())?;
        physical.write(at, data).map_err(|_| Fault { address: va })
    }
}

/// A [`Mapper`] that changes nothing and records every call it accepts in its [`MapperLog`].
#[derive(Debug, Default)]
pub struct RecordingMapper(MapperLog);

impl RecordingMapper {
    /// A mapper and the log it writes to.
    #[must_use]
    pub fn new() -> (Self, MapperLog) {
        let log = MapperLog::default();
        (Self(log.clone()), log)
    }

    fn record(&self, call: MapperCall) -> Result<(), MapperError> {
        self.0.with(|state| {
            let number = state.counted;
            state.counted = state.counted.saturating_add(1);
            if state.refuse || state.refuse_calls.contains(&number) {
                return Err(MapperError::Refused(
                    "the test told the mapper to refuse".into(),
                ));
            }
            state.calls.push(call);
            Ok(())
        })
    }
}

impl Mapper for RecordingMapper {
    fn map(&mut self, address: u64, size: u64, perms: Perms) -> Result<(), MapperError> {
        self.record(MapperCall::Map {
            address,
            size,
            perms,
        })
    }

    fn unmap(&mut self, address: u64, size: u64) -> Result<(), MapperError> {
        self.record(MapperCall::Unmap { address, size })?;
        let end = address.saturating_add(size);
        self.0.with(|state| {
            state
                .shared
                .retain(|start, _| !(address..end).contains(start));
            // Bounded by the pages mapped, not by `size`.
            let gone: Vec<u64> = state
                .mapped_pages
                .range(address..end)
                .map(|(page, _)| *page)
                .collect();
            for page in gone {
                state.mapped_pages.remove(&page);
            }
        });
        Ok(())
    }

    fn protect(&mut self, address: u64, size: u64, perms: Perms) -> Result<(), MapperError> {
        self.record(MapperCall::Protect {
            address,
            size,
            perms,
        })
    }

    fn allocate_shared(&mut self, size: u64) -> Result<SharedPages, MapperError> {
        let (backing, physical) = self.0.with(|state| {
            let backing = state
                .next
                .checked_add(size)
                .filter(|end| *end <= SHARED_BASE + SHARED_CAPACITY)
                .and_then(|_| BackingRange::new(state.next, size).ok())
                .ok_or_else(|| {
                    MapperError::OutOfMemory(format!(
                        "no room for {size:#x} bytes of shared memory"
                    ))
                })?;
            Ok::<_, MapperError>((backing, Arc::clone(&state.physical)))
        })?;
        self.record(MapperCall::AllocateShared { backing })?;
        self.0.with(|state| {
            state.next = backing.end();
            state.handed_out.insert(backing.base(), backing.end());
        });
        Ok(SharedPages::new(backing, physical))
    }

    fn release_shared(&mut self, backing: BackingRange) -> Result<(), MapperError> {
        self.0.with(|state| {
            let (base, size) = (backing.base(), backing.size());
            if state.handed_out.get(&base) != Some(&backing.end()) {
                return Err(MapperError::Refused(format!(
                    "the {size:#x} bytes at {base:#x} are not a shared memory this mapper \
                     handed out"
                )));
            }
            if let Some(address) = state
                .mapped_pages
                .iter()
                .find(|(_, pa)| (base..backing.end()).contains(*pa))
                .map(|(address, _)| *address)
            {
                return Err(MapperError::Refused(format!(
                    "the {size:#x} bytes at {base:#x} are still mapped at {address:#x}"
                )));
            }
            Ok(())
        })?;
        self.record(MapperCall::ReleaseShared { backing })?;
        self.0
            .with(|state| state.handed_out.remove(&backing.base()));
        Ok(())
    }

    fn map_shared(
        &mut self,
        address: u64,
        backing: BackingRange,
        perms: Perms,
    ) -> Result<(), MapperError> {
        self.record(MapperCall::MapShared {
            address,
            backing,
            perms,
        })?;
        self.0.with(|state| {
            state.shared.insert(address, backing);
            let pages = (backing.base()..backing.end()).step_by(PAGE_STEP);
            for (offset, pa) in (0..).step_by(PAGE_STEP).zip(pages) {
                if let Some(va) = address.checked_add(offset) {
                    state.mapped_pages.insert(va, pa);
                }
            }
        });
        Ok(())
    }

    fn map_alias(
        &mut self,
        address: u64,
        source: u64,
        size: u64,
        perms: Perms,
    ) -> Result<(), MapperError> {
        self.record(MapperCall::MapAlias {
            address,
            source,
            size,
            perms,
        })?;
        // The alias maps again whichever shared pages the source maps, bounded by those pages.
        self.0.with(|state| {
            let end = source.saturating_add(size);
            let aliased: Vec<(u64, u64)> = state
                .mapped_pages
                .range(source..end)
                .filter_map(|(page, pa)| {
                    let va = address.checked_add(page.checked_sub(source)?)?;
                    Some((va, *pa))
                })
                .collect();
            state.mapped_pages.extend(aliased);
        });
        Ok(())
    }
}
