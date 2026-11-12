//! The kernel's tables. Each system call declares which of these it touches; nothing else can
//! reach kernel state. The ones not yet built are placeholders until their port slices land.

use crate::time::CounterRate;

/// Threads, per-core run queues, wait lists, events, sessions, the address arbiter and timers.
/// Threads, run queues and the object table are built ([`crate::sched`], [`crate::object`]);
/// the rest arrive with their slices.
pub use crate::sched::Scheduler;

/// One process's memory blocks, regions and heap.
#[derive(Debug, Default)]
pub struct AddressSpace {
    _private: (),
}

/// Per-process facts fixed at creation (capabilities, ids, the counter rate).
///
/// PROVISIONAL: there is one process, so this holds one process's facts and no rows. Kernel
/// slice 8 (process creation) makes it a table of processes, each with these facts, and the
/// accessors here take a process id.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Processes {
    counter_rate: CounterRate,
}

impl Processes {
    /// The facts of the one process: its counter runs at `counter_rate`
    /// ([`CounterRate::SWITCH_1`] or [`CounterRate::SWITCH_2`], by which console the program
    /// was built for).
    #[must_use]
    pub const fn new(counter_rate: CounterRate) -> Self {
        Self { counter_rate }
    }

    /// The rate of the process's counter: what the kernel converts the environment's clock with
    /// for it (kernel spec, "Guest time"), and what its patched counter reads must be rescaled
    /// to.
    #[must_use]
    pub const fn counter_rate(&self) -> CounterRate {
        self.counter_rate
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_process_keeps_the_counter_rate_it_was_created_with() {
        for rate in [CounterRate::SWITCH_1, CounterRate::SWITCH_2] {
            assert_eq!(Processes::new(rate).counter_rate(), rate);
        }
    }
}
