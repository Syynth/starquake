//! The kernel's tables. Each system call declares which of these it touches; nothing else can
//! reach kernel state.

use std::sync::Arc;

use crate::id::ProcessId;
use crate::process::{ProcessDescription, ProcessError, ProcessInfo};
use crate::slab::{Slab, SlabError};
use crate::time::CounterRate;

/// Threads, per-core run queues, wait lists, events, sessions, the address arbiter and timers.
/// Threads, run queues and the object table are built ([`crate::sched`], [`crate::object`]), and
/// so are events, wait lists and timers ([`crate::sync`]) and sessions ([`crate::ipc`]); the
/// address arbiter arrives with its slice.
pub use crate::sched::Scheduler;

/// One process's memory blocks, regions, heap and memory accounting ([`crate::memory`]).
pub use crate::memory::AddressSpace;

/// The process id Horizon gives the first process it creates after its initial ones (the
/// system modules it starts itself): `KProcess::ProcessIdMin`, `InitialProcessIdMax + 1`
/// (`eden/src/core/hle/kernel/k_process.h` @ cf654052), where Eden's `next_user_process_id`
/// starts (`eden/src/core/hle/kernel/kernel.cpp`). [`Processes`] hands ids out from here.
pub const FIRST_USER_PROCESS_ID: u64 = 0x51;

/// Per-process facts fixed at creation (kernel spec, "Tables behind a dispatcher": the
/// `Processes` table): one [`ProcessInfo`] per process, and which process the other tables
/// beside it (the handle table and the address space) belong to.
///
/// The facts never change once a process exists, so the table needs no lock of its own and no
/// call declares it: the dispatcher reads it for the guest time and hands a copy to the calls
/// that check a process's capabilities. The rows are shared ([`Arc`]), so that copy is cheap;
/// creating a process copies them once.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Processes {
    rows: Arc<Slab<ProcessId, ProcessInfo>>,
    /// The process whose handle table and address space sit beside this table.
    own: ProcessId,
    /// Its guest id and counter rate, the two facts the dispatcher reads on every dispatch,
    /// kept out of the rows so that reading them cannot fail.
    own_process_id: u64,
    own_rate: CounterRate,
    /// The guest id the next process gets (`next_user_process_id`).
    next_process_id: u64,
}

impl Processes {
    /// A table holding one process, created from `description`: the process the handle table
    /// and the address space beside it belong to. It gets [`FIRST_USER_PROCESS_ID`].
    pub fn new(description: &ProcessDescription) -> Result<Self, ProcessError> {
        let mut rows = Slab::new();
        let info = ProcessInfo::new(description, FIRST_USER_PROCESS_ID)?;
        let own_rate = info.counter_rate();
        let own = rows.insert(info).map_err(|_| ProcessError::TableFull)?;
        Ok(Self {
            rows: Arc::new(rows),
            own,
            own_process_id: FIRST_USER_PROCESS_ID,
            own_rate,
            next_process_id: FIRST_USER_PROCESS_ID
                .checked_add(1)
                .ok_or(ProcessError::OutOfIds)?,
        })
    }

    /// Adds a process created from `description`, with the next guest id
    /// (`KernelCore::CreateNewUserProcessID`), and returns its id. A description the checks
    /// refuse takes no id.
    pub fn create(&mut self, description: &ProcessDescription) -> Result<ProcessId, ProcessError> {
        let info = ProcessInfo::new(description, self.next_process_id)?;
        let next = self
            .next_process_id
            .checked_add(1)
            .ok_or(ProcessError::OutOfIds)?;
        let id = Arc::make_mut(&mut self.rows)
            .insert(info)
            .map_err(|_| ProcessError::TableFull)?;
        self.next_process_id = next;
        Ok(id)
    }

    /// The facts of the process `id` names.
    pub fn get(&self, id: ProcessId) -> Result<&ProcessInfo, SlabError> {
        self.rows.get(id)
    }

    /// The process the handle table and the address space beside this table belong to.
    #[must_use]
    pub const fn own(&self) -> ProcessId {
        self.own
    }

    /// The facts of [`Processes::own`].
    pub fn own_info(&self) -> Result<&ProcessInfo, SlabError> {
        self.rows.get(self.own)
    }

    /// The own process's id as the guest knows it: what a request's process id slot is given.
    #[must_use]
    pub const fn process_id(&self) -> u64 {
        self.own_process_id
    }

    /// The rate of the own process's counter: what the kernel converts the environment's clock
    /// with for it (kernel spec, "Guest time"), and what its patched counter reads must be
    /// rescaled to. Decided by the console the process was built for
    /// ([`crate::process::Console::counter_rate`]).
    #[must_use]
    pub const fn counter_rate(&self) -> CounterRate {
        self.own_rate
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::process::Console;
    use crate::result::KernelError;

    #[test]
    fn the_first_process_has_the_first_user_process_id() {
        let processes = Processes::new(&ProcessDescription::homebrew(Console::Switch1)).unwrap();
        assert_eq!(processes.process_id(), FIRST_USER_PROCESS_ID);
        assert_eq!(
            processes.get(processes.own()).map(ProcessInfo::process_id),
            Ok(FIRST_USER_PROCESS_ID)
        );
    }

    #[test]
    fn a_process_counts_at_the_rate_of_its_console() {
        for console in [Console::Switch1, Console::Switch2] {
            let processes = Processes::new(&ProcessDescription::homebrew(console)).unwrap();
            assert_eq!(processes.counter_rate(), console.counter_rate());
        }
    }

    #[test]
    fn later_processes_take_the_next_ids_and_a_refused_one_takes_none() {
        let mut processes =
            Processes::new(&ProcessDescription::homebrew(Console::Switch2)).unwrap();
        let mut refused = ProcessDescription::homebrew(Console::Switch2);
        refused.cores.first = refused.cores.last + 1;
        assert_eq!(
            processes.create(&refused),
            Err(ProcessError::Capability(KernelError::InvalidCombination))
        );
        let second = processes
            .create(&ProcessDescription::homebrew(Console::Switch1))
            .unwrap();
        let info = processes.get(second).unwrap();
        assert_eq!(info.process_id(), FIRST_USER_PROCESS_ID + 1);
        assert_eq!(info.console(), Console::Switch1);
        // Each later process takes the next id again, not the same one.
        let third = processes
            .create(&ProcessDescription::homebrew(Console::Switch2))
            .unwrap();
        assert_eq!(
            processes.get(third).unwrap().process_id(),
            FIRST_USER_PROCESS_ID + 2
        );
        // The own process is still the first, and its rate is its own console's.
        assert_eq!(processes.process_id(), FIRST_USER_PROCESS_ID);
        assert_eq!(processes.counter_rate(), Console::Switch2.counter_rate());
    }

    #[test]
    fn a_copy_of_the_table_does_not_see_later_processes() {
        let mut processes =
            Processes::new(&ProcessDescription::homebrew(Console::Switch1)).unwrap();
        let copy = processes.clone();
        let second = processes
            .create(&ProcessDescription::homebrew(Console::Switch1))
            .unwrap();
        assert!(processes.get(second).is_ok());
        assert!(copy.get(second).is_err());
    }
}
