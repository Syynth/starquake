//! [`FakeWatchKernel`]: the kernel's half of a [`WriteWatch`], for tests of the GPU caches (the
//! buffer and texture caches' "fake watch", gpu-protocol slices G16 and G17).
//!
//! Compiled for this crate's own tests and for crates that enable the `test-support` feature
//! (`docs/specs/testing.md`, "Shared test support").

use std::sync::Arc;

use super::{PageRange, WatchError, WatchRequest, WriteWatch};
use crate::translation::{MapError, PAGE_SIZE};

/// Plays the kernel against a [`WriteWatch`]: [`Self::enter`] is a kernel entry that applies
/// every waiting request, and [`Self::cpu_write`] is a guest store that faults on each watched
/// page it touches, exactly as the real kernel's calls into the watch would.
#[derive(Debug)]
pub struct FakeWatchKernel {
    watch: Arc<WriteWatch>,
    taken: Vec<WatchRequest>,
}

impl FakeWatchKernel {
    /// A fake kernel for `watch`.
    pub fn new(watch: Arc<WriteWatch>) -> Self {
        Self {
            watch,
            taken: Vec::new(),
        }
    }

    /// The watch this fake drives.
    pub fn watch(&self) -> &Arc<WriteWatch> {
        &self.watch
    }

    /// A kernel entry: takes every waiting request and arms it, as the kernel does once it has
    /// protected the pages and flushed. Returns how many requests it armed.
    pub fn enter(&mut self) -> Result<usize, WatchError> {
        self.taken.clear();
        self.watch.take_requests(&mut self.taken);
        self.watch.arm(&self.taken)?;
        Ok(self.taken.len())
    }

    /// A guest store of `len` bytes at `va`: each page it touches that is watched faults once
    /// and is recorded dirty ([`WriteWatch::record_fault`]). Returns how many pages faulted.
    ///
    /// Walks every page of the store, so it is meant for test-sized stores. The range must be
    /// one [`PageRange::covering`] accepts.
    pub fn cpu_write(&self, va: u64, len: u64) -> Result<u64, MapError> {
        let range = PageRange::covering(va, len)?;
        let mut faulted = 0;
        let mut page = range.start();
        while page < range.end() {
            if self.watch.record_fault(page) {
                faulted += 1;
            }
            // The range's end is representable, so the step stops at it without overflowing.
            let Some(next) = page.checked_add(PAGE_SIZE) else {
                break;
            };
            page = next;
        }
        Ok(faulted)
    }
}
