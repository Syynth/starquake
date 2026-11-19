//! The completion wake (docs/specs/render-backend.md, "The completion wake"; decided by the
//! owner on 2026-11-18, render-backend decision 14; gpu-protocol, "Releases and completion").
//!
//! The GPU back thread sleeps on its ring's wake-up, and must also wake when a list it submitted
//! completes, because releases and syncpoint increments wait on completion and neither
//! [`Backend::completed`] nor [`Backend::wait`] can wake a thread that sleeps on something else.
//! So `core` hands each backend, at construction, a [`CompletionWake`]: a function to call,
//! nothing more, which the GPU crate makes from its ring's consumer side. It is not a trait
//! method, like the compile pool and the wait-while-compiling setting.
//!
//! What a backend does with it (the spec's three rules):
//!
//! - it calls [`CompletionWake::wake`] each time the value [`Backend::completed`] reports
//!   advances, *after* storing the new value, so a back thread it wakes reads that value or a
//!   later one;
//! - it may call it more often (a spurious wake costs the back thread one look), and calls it
//!   once for several lists that complete together;
//! - it never blocks in it, and never calls it while holding a lock the back thread can take.
//!
//! A backend that forgets is caught twice: at run time the back thread's 100 ms safety-net
//! recheck finds [`Backend::completed`] moved without a wake and counts it as a lost wake (the
//! GPU crate's `back::fence`), and here [`check_completion_wake`] is the contract check every
//! backend's tests run, with a [`WakeCounter`] as the wake, to see that a wake follows every
//! advance. The recording backend passes it (`tests::the_recording_backend_meets_the_contract`).
//!
//! # What the check cannot see
//!
//! [`check_completion_wake`] waits for each list to complete and then for the wake count to pass
//! what it was before that list was submitted. A backend that wakes more than once per advance
//! can therefore lend a late extra wake to the next list and hide one missing wake there; and
//! the check cannot see whether a wake came *after* the store, since the wake has no handle to
//! the backend. The recording backend pins both properties in its own tests
//! (`recording::tests::each_submit_wakes_once_after_the_new_value_is_stored`).

use crate::backend::{Backend, WaitOutcome};
use crate::command::CommandList;
use crate::error::RenderError;
use std::fmt;
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::time::Duration;
use thiserror::Error;

/// The function a backend calls after each advance of [`Backend::completed`] (module docs).
///
/// Cloneable, `Send` and `Sync`: a backend may call it from its render thread, or, like the
/// recording backend, from whichever thread submits.
#[derive(Clone)]
pub struct CompletionWake(Arc<dyn Fn() + Send + Sync>);

impl CompletionWake {
    /// A wake that calls `wake`. It must not block, and must not take a lock the backend holds
    /// while it calls the wake.
    pub fn new(wake: impl Fn() + Send + Sync + 'static) -> Self {
        Self(Arc::new(wake))
    }

    /// A wake that does nothing, for a backend no GPU back thread sleeps on (a test, or a
    /// replay of a host-level capture).
    pub fn none() -> Self {
        Self::new(|| {})
    }

    /// Calls the wake.
    pub fn wake(&self) {
        (self.0)();
    }
}

impl fmt::Debug for CompletionWake {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("CompletionWake")
    }
}

/// A wake that counts its calls, for [`check_completion_wake`]. Share it in an [`Arc`], build
/// the backend with [`WakeCounter::completion_wake`], and hand the same counter to the check.
#[derive(Debug, Default)]
pub struct WakeCounter {
    count: Mutex<u64>,
    rang: Condvar,
}

impl WakeCounter {
    /// A counter at zero.
    pub fn new() -> Self {
        Self::default()
    }

    fn lock(&self) -> MutexGuard<'_, u64> {
        // The guarded value is a plain count, so a poisoned lock holds a usable one.
        self.count.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// A [`CompletionWake`] that adds one to this counter.
    pub fn completion_wake(self: &Arc<Self>) -> CompletionWake {
        let counter = Arc::clone(self);
        CompletionWake::new(move || {
            let mut count = counter.lock();
            *count = count.saturating_add(1);
            drop(count);
            counter.rang.notify_all();
        })
    }

    /// How many wakes so far.
    pub fn count(&self) -> u64 {
        *self.lock()
    }

    /// Waits until more than `seen` wakes have been counted, or `timeout` passes; whether they
    /// were.
    pub fn wait_beyond(&self, seen: u64, timeout: Duration) -> bool {
        let count = self.lock();
        let (count, _) = self
            .rang
            .wait_timeout_while(count, timeout, |count| *count <= seen)
            .unwrap_or_else(PoisonError::into_inner);
        *count > seen
    }
}

/// How a backend broke the completion-wake contract, or why the check could not run.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum WakeContractError {
    /// The backend refused one of the check's empty lists.
    #[error("the backend refused list {list}: {error}")]
    Submit {
        /// Which list, from 0.
        list: u32,
        /// The backend's error.
        error: RenderError,
    },
    /// Waiting for a list failed.
    #[error("waiting for list {list} failed: {error}")]
    Wait {
        /// Which list, from 0.
        list: u32,
        /// The backend's error.
        error: RenderError,
    },
    /// A list did not complete within the timeout.
    #[error("list {list} (fence value {fence}) did not complete in time")]
    NotCompleted {
        /// Which list, from 0.
        list: u32,
        /// Its fence value.
        fence: u64,
    },
    /// The backend's completed value reached a list's fence and no wake followed in time.
    #[error("list {list} (fence value {fence}) completed and no wake followed")]
    NoWake {
        /// Which list, from 0.
        list: u32,
        /// Its fence value.
        fence: u64,
    },
}

/// Checks that a wake follows every advance of `backend`'s completed value: submits `lists`
/// empty lists one at a time, waits up to `timeout` for each to complete, then up to `timeout`
/// for `wakes` to count a wake made since that list was submitted. `backend` must have been
/// built with `wakes.completion_wake()` and have nothing else in flight.
///
/// The first failure is returned, in the order of the steps above; what it cannot see is in
/// the module docs.
pub fn check_completion_wake(
    backend: &dyn Backend,
    wakes: &WakeCounter,
    lists: u32,
    timeout: Duration,
) -> Result<(), WakeContractError> {
    for list in 0..lists {
        let seen = wakes.count();
        let fence = backend
            .submit(CommandList::new())
            .map_err(|error| WakeContractError::Submit { list, error })?;
        match backend.wait(fence, timeout) {
            Ok(WaitOutcome::Reached) => {}
            Ok(WaitOutcome::TimedOut) => {
                return Err(WakeContractError::NotCompleted {
                    list,
                    fence: fence.get(),
                });
            }
            Err(error) => return Err(WakeContractError::Wait { list, error }),
        }
        if !wakes.wait_beyond(seen, timeout) {
            return Err(WakeContractError::NoWake {
                list,
                fence: fence.get(),
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
