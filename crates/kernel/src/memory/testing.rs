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

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, PoisonError};

use starquake_memory::{BackingRange, Fault, FlatMemory, GuestMemory, PAGE_SIZE, Perms};

use super::mapper::{Mapper, MapperError};
use super::shared::SharedPages;

/// Where the guest-physical memory that backs shared memory starts.
pub const SHARED_BASE: u64 = 0x10_0000_0000;
/// How many bytes of shared memory a [`RecordingMapper`] can hand out in all: 64 pages.
pub const SHARED_CAPACITY: u64 = 64 * PAGE_SIZE;

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
    /// [`Mapper::map_shared`].
    MapShared {
        /// The first address.
        address: u64,
        /// The pages mapped there.
        backing: BackingRange,
        /// What guest code may do there.
        perms: Perms,
    },
}

#[derive(Debug)]
struct State {
    calls: Vec<MapperCall>,
    refuse: bool,
    /// The guest-physical memory shared memory is backed from.
    physical: Arc<FlatMemory>,
    /// The first byte of it not handed out yet.
    next: u64,
    /// What `map_shared` mapped and nothing has unmapped since: first address to the pages.
    shared: BTreeMap<u64, BackingRange>,
}

impl Default for State {
    fn default() -> Self {
        let capacity = usize::try_from(SHARED_CAPACITY).unwrap_or(usize::MAX);
        Self {
            calls: Vec::new(),
            refuse: false,
            physical: Arc::new(FlatMemory::new(SHARED_BASE, capacity)),
            next: SHARED_BASE,
            shared: BTreeMap::new(),
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
/// is whole pages and not zero. The pages are allocated at once, so a test asks for few.
#[must_use]
pub fn standalone_shared_pages(size: u64) -> Option<SharedPages> {
    let backing = BackingRange::new(0, size).ok()?;
    let memory = FlatMemory::new(0, usize::try_from(size).ok()?);
    Some(SharedPages::new(backing, Arc::new(memory)))
}

/// See [`MapperLog::shared_view`].
#[derive(Debug, Clone)]
pub struct SharedView(MapperLog);

impl SharedView {
    /// The guest-physical address of `len` bytes at `va`, if one mapping holds them all.
    fn resolve(&self, va: u64, len: usize) -> Result<(Arc<FlatMemory>, u64), Fault> {
        let fault = Fault { address: va };
        let len = u64::try_from(len).map_err(|_| fault)?;
        let end = va.checked_add(len).ok_or(fault)?;
        self.0.with(|state| {
            let (start, backing) = state.shared.range(..=va).next_back().ok_or(fault)?;
            let offset = va - start;
            if end - start > backing.size() {
                return Err(fault);
            }
            Ok((state.physical.clone(), backing.base() + offset))
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
            if state.refuse {
                return Err(MapperError("the test told the mapper to refuse".into()));
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
        self.0
            .with(|state| state.shared.retain(|start, _| !(address..end).contains(start)));
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
                .ok_or_else(|| MapperError(format!("no room for {size:#x} bytes of shared memory")))?;
            Ok::<_, MapperError>((backing, state.physical.clone()))
        })?;
        self.record(MapperCall::AllocateShared { backing })?;
        self.0.with(|state| state.next = backing.end());
        Ok(SharedPages::new(backing, physical))
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
        self.0.with(|state| state.shared.insert(address, backing));
        Ok(())
    }
}
