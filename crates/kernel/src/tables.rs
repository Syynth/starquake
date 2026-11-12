//! The kernel's tables. Each system call declares which of these it touches; nothing else can
//! reach kernel state.

use crate::time::CounterRate;

/// Threads, per-core run queues, wait lists, events, sessions, the address arbiter and timers.
/// Threads, run queues and the object table are built ([`crate::sched`], [`crate::object`]);
/// the rest arrive with their slices.
pub use crate::sched::Scheduler;

/// One process's memory blocks, regions, heap and memory accounting ([`crate::memory`]).
pub use crate::memory::AddressSpace;

/// Per-process facts fixed at creation (capabilities, ids, the counter rate).
///
/// PROVISIONAL: there is one process, so this holds one process's facts and no rows. Kernel
/// slice 8 (process creation) makes it a table of processes, each with these facts, and the
/// accessors here take a process id.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Processes {
    counter_rate: CounterRate,
}

/// The process id Horizon gives the first process it creates after its initial ones (the
/// system modules it starts itself): `KProcess::ProcessIdMin`, `InitialProcessIdMax + 1`
/// (`eden/src/core/hle/kernel/k_process.h` @ cf654052), where Eden's `next_user_process_id`
/// starts (`eden/src/core/hle/kernel/kernel.cpp`).
///
/// PROVISIONAL: the one process has this id. Kernel slice 8 (process creation) hands ids out.
pub const FIRST_USER_PROCESS_ID: u64 = 0x51;

impl Processes {
    /// The facts of the one process: its counter runs at `counter_rate`
    /// ([`CounterRate::SWITCH_1`] or [`CounterRate::SWITCH_2`], by which console the program
    /// was built for).
    #[must_use]
    pub const fn new(counter_rate: CounterRate) -> Self {
        Self { counter_rate }
    }

    /// The process's id as the guest knows it: what a request's process id slot is given
    /// ([`FIRST_USER_PROCESS_ID`] for the one process).
    #[must_use]
    pub const fn process_id(&self) -> u64 {
        FIRST_USER_PROCESS_ID
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
    fn the_one_process_has_the_first_user_process_id() {
        let processes = Processes::new(CounterRate::SWITCH_1);
        assert_eq!(processes.process_id(), FIRST_USER_PROCESS_ID);
    }

    #[test]
    fn a_process_keeps_the_counter_rate_it_was_created_with() {
        for rate in [CounterRate::SWITCH_1, CounterRate::SWITCH_2] {
            assert_eq!(Processes::new(rate).counter_rate(), rate);
        }
    }
}
