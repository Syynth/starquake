//! A [`Mapper`] for tests: it records what it is asked and can be told to refuse.
//!
//! Exported behind the `test-support` feature (test strategy spec, "Shared test support": each
//! crate exports its own fakes), for this crate's tests and for crates that run the kernel
//! without a translation behind it.

use std::sync::{Arc, Mutex, PoisonError};

use starquake_memory::Perms;

use super::mapper::{Mapper, MapperError};

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
}

#[derive(Debug, Default)]
struct State {
    calls: Vec<MapperCall>,
    refuse: bool,
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
        self.record(MapperCall::Unmap { address, size })
    }

    fn protect(&mut self, address: u64, size: u64, perms: Perms) -> Result<(), MapperError> {
        self.record(MapperCall::Protect {
            address,
            size,
            perms,
        })
    }
}
