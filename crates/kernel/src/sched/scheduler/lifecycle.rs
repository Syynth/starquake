//! Threads and processes from creation to destruction, as the scheduler table keeps them (kernel
//! slice 8).
//!
//! Ported from Eden @ cf654052 (GPL-3.0-or-later), paths relative to the reference folder:
//!
//! - `eden/src/core/hle/kernel/k_thread.cpp`: `Initialize` for a user thread (the virtual core
//!   translated to a physical one through `VirtualToPhysicalCoreMap`, the thread-local region
//!   from the owner's `CreateThreadLocalRegion`, the context from `ResetThreadContext64`, a new
//!   thread id, `RegisterThread`), `Finalize` (the region given back, `UnregisterThread`),
//!   `Exit`/`StartTermination`/`FinishTermination` (state `Terminated`, the running count, the
//!   thread's reference to itself closed), and the virtual half of `SetCoreMask`.
//! - `eden/src/core/hle/kernel/k_process.cpp`: `RegisterThread`, `UnregisterThread`,
//!   `Increment`/`DecrementRunningThreadCount` (the last running thread to end terminates the
//!   process), `TerminateChildren`.
//! - `eden/src/core/hle/kernel/k_auto_object.h`: an object is destroyed with its last reference.
//!
//! A thread's object row is referenced by the handles to it and, while it runs, by the thread
//! itself. When the last reference goes the thread is destroyed ([`Scheduler::tear_down`]): its
//! thread-local region goes back to its process and its row is removed. A thread that ended
//! while a handle to it is open keeps its row, so `GetThreadId` and friends still answer, as
//! they do on a console.
//!
//! # Divergences
//!
//! Each is pinned by the named test in `sched/scheduler/lifecycle/tests.rs`, unless another
//! file is named.
//!
//! - **`x18` starts at zero.** Eden's `ResetThreadContext64` seeds it with a random odd number
//!   (`Common::Random::Random64(0) | 1`). Creation has no route to the environment's random
//!   numbers yet. Test: `a_user_thread_starts_where_its_creator_says`.
//! - **A destroyed thread is taken off the cores at once.** Eden's `FinishTermination` waits
//!   until no core runs the thread before it closes the thread's reference. Here the reference
//!   is closed when the thread ends, and a destroyed thread's id is cleared from any core that
//!   still names it, which is then marked as needing scheduling. Test:
//!   `a_thread_destroyed_while_a_core_names_it_leaves_that_core`.
//! - **Termination is immediate.** Eden asks every other thread to terminate
//!   (`RequestTerminate`) and waits for each. Here every thread is ended in the call, in thread
//!   order: there is no other core running one of them yet (slice 10). Test (in
//!   `svc/process/tests.rs`): `exiting_the_process_ends_every_thread_and_closes_every_handle`.
//! - **A thread that ends while it waits leaves its wait at once** (see the module docs of
//!   [`crate::sync`]). Test: `a_thread_that_terminates_while_waiting_leaves_every_wait_list`.

use starquake_cpu::GuestRegs;

use super::super::thread::ThreadState;
use super::super::{Priority, SchedError};
use super::Scheduler;
use crate::id::{ObjectId, ProcessId, ThreadId};
use crate::memory::AddressSpace;
use crate::object::Object;
use crate::process::{
    IDEAL_CORE_DONT_CARE, IDEAL_CORE_NO_UPDATE, ProcessRecord, ProcessState, physical_core,
    physical_mask,
};
use crate::result::{CallResult, KernelError, SvcError};

/// What a user thread is created with (`KThread::InitializeUserThread`'s arguments).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UserThread {
    /// Where it starts (`func`).
    pub entry: u64,
    /// What it finds in `x0` (`arg`).
    pub argument: u64,
    /// Its stack pointer (`user_stack_top`).
    pub stack_top: u64,
    /// Its priority.
    pub priority: Priority,
    /// Its virtual core, already checked against the process (below
    /// [`crate::process::VIRTUAL_CORE_COUNT`]).
    pub virtual_core: u8,
}

/// What ending a thread left of its process.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThreadExit {
    /// The process has other running threads, or the thread had no process, or it was never
    /// started.
    Others,
    /// It was the process's last running thread: the process must be terminated
    /// (`DecrementRunningThreadCount` reaching zero calls `KProcess::Terminate`).
    LastOf(ProcessId),
}

/// A failure that only broken bookkeeping explains.
fn broken(what: impl core::fmt::Display) -> SvcError {
    SvcError::Internal(what.to_string())
}

impl Scheduler {
    /// Gives the scheduler a row for a new process, in [`ProcessState::Created`]. A process it
    /// already has is damaged bookkeeping ([`SchedError::Corrupt`]).
    pub fn add_process(&mut self, process: ProcessId) -> Result<(), SchedError> {
        if self.processes.contains_key(&process) {
            return Err(SchedError::Corrupt);
        }
        self.processes.insert(process, ProcessRecord::new());
        Ok(())
    }

    /// What changes about a process while it runs.
    pub fn process(&self, process: ProcessId) -> Result<&ProcessRecord, SchedError> {
        self.record(process)
    }

    pub(super) fn record(&self, process: ProcessId) -> Result<&ProcessRecord, SchedError> {
        self.processes
            .get(&process)
            .ok_or(SchedError::UnknownProcess(process))
    }

    pub(super) fn record_mut(
        &mut self,
        process: ProcessId,
    ) -> Result<&mut ProcessRecord, SchedError> {
        self.processes
            .get_mut(&process)
            .ok_or(SchedError::UnknownProcess(process))
    }

    /// Marks a created process as running (`KProcess::Run`'s `ChangeState`): anything but
    /// [`ProcessState::Created`] is [`KernelError::InvalidState`], Eden's result.
    pub fn mark_running(&mut self, process: ProcessId) -> Result<(), SchedError> {
        let record = self.record_mut(process)?;
        if record.state != ProcessState::Created {
            return Err(KernelError::InvalidState.into());
        }
        record.state = ProcessState::Running;
        Ok(())
    }

    /// Puts a running process back to [`ProcessState::Created`]: `KProcess::Run`'s
    /// `ON_RESULT_FAILURE_2 { ChangeState(state) }`, for a start that failed after the state
    /// changed ([`crate::process::run`]). Anything but [`ProcessState::Running`] is
    /// [`KernelError::InvalidState`].
    pub fn unmark_running(&mut self, process: ProcessId) -> Result<(), SchedError> {
        let record = self.record_mut(process)?;
        if record.state != ProcessState::Running {
            return Err(KernelError::InvalidState.into());
        }
        record.state = ProcessState::Created;
        Ok(())
    }

    /// Marks a process as ended, once its threads are and its handles are closed.
    pub fn mark_terminated(&mut self, process: ProcessId) -> Result<(), SchedError> {
        self.record_mut(process)?.state = ProcessState::Terminated;
        Ok(())
    }

    /// Creates a thread of `process` (`KThread::Initialize` for `ThreadType::User`): on the
    /// physical core its virtual core maps to, with a thread-local region reserved from the
    /// process's pages in `space`, its context reset to start at `spec.entry` with `x0` the
    /// argument, `sp` the stack top and `TPIDRRO_EL0` its region, and an object row holding
    /// one reference, its creator's. The creator adds a handle and closes that reference
    /// ([`Scheduler::close_object`]), as `CreateThread` and `KProcess::Run` do.
    ///
    /// A virtual core with no physical core in this scheduler is
    /// [`KernelError::InvalidCoreId`]; a region that cannot be mapped fails with the address
    /// space's error; a full object or thread table with [`KernelError::OutOfResource`] or an
    /// internal failure. On any failure nothing is left behind: the region goes back and the
    /// thread row, if it was made, is removed.
    pub fn create_user_thread(
        &mut self,
        space: &mut AddressSpace,
        process: ProcessId,
        spec: &UserThread,
    ) -> CallResult<(ThreadId, ObjectId)> {
        let core = physical_core(spec.virtual_core)
            .filter(|core| self.core_mask().contains(*core))
            .ok_or(KernelError::InvalidCoreId)?;
        let tls = self.record_mut(process)?.tls.reserve(space)?;
        let thread = match self.insert_thread(spec.priority, core, Some(spec.virtual_core)) {
            Ok(thread) => thread,
            Err(error) => {
                self.record_mut(process)?.tls.release(tls).map_err(broken)?;
                return Err(error.into());
            }
        };
        let row = self.thread_mut(thread)?;
        row.process = Some(process);
        row.tls_address = tls;
        let mut context = GuestRegs {
            pc: spec.entry,
            sp: spec.stack_top,
            tpidrro: tls,
            ..GuestRegs::default()
        };
        context.x[0] = spec.argument;
        row.context = context;
        self.record_mut(process)?.threads.insert(thread);
        let object = match self.objects.create(Object::Thread(thread)) {
            Ok(object) => object,
            Err(error) => {
                self.destroy_thread(thread)?;
                return Err(error.into());
            }
        };
        self.thread_mut(thread)?.object = Some(object);
        Ok((thread, object))
    }

    /// Drops one reference to `object`; when it was the last, tears down what it named
    /// ([`Scheduler::tear_down`]).
    pub fn close_object(&mut self, object: ObjectId) -> CallResult<()> {
        if let Some(kind) = self.objects.release(object)? {
            self.tear_down(object, kind)?;
        }
        Ok(())
    }

    /// Tears down what an object named once its last reference has gone: the one place that
    /// decides what destroying each kind means. A session's client end goes and its server is
    /// told (`KClientSession::Destroy`); a thread is destroyed (`KThread::Finalize`); a process
    /// object names a row that outlives it; an event's end is its row, which the object table
    /// has removed already.
    pub fn tear_down(&mut self, object: ObjectId, kind: Object) -> CallResult<()> {
        match kind {
            Object::Session => self.ipc.forget(object)?,
            Object::Thread(thread) => self.destroy_thread(thread)?,
            Object::Process(_) | Object::Event(_) | Object::ReadableEvent(_) => {}
        }
        Ok(())
    }

    /// `KThread::Finalize`: the thread's region goes back to its process, the process forgets
    /// it, any core that still names it lets it go, and its row is removed. Only a thread that
    /// was never started or has ended can lose its last reference; anything else is broken
    /// bookkeeping.
    fn destroy_thread(&mut self, id: ThreadId) -> CallResult<()> {
        let thread = self.thread(id)?;
        if !matches!(
            thread.state,
            ThreadState::Initialized | ThreadState::Terminated
        ) || thread.holds_itself
        {
            return Err(broken(format!(
                "thread {id:?} lost its last reference while it runs"
            )));
        }
        let (process, tls) = (thread.process, thread.tls_address);
        if let Some(process) = process {
            let record = self.record_mut(process)?;
            if tls != 0 {
                record.tls.release(tls).map_err(broken)?;
            }
            record.threads.remove(&id);
        }
        for core in &mut self.cores {
            if core.highest == Some(id) {
                core.highest = None;
                core.needs_scheduling = true;
            }
            if core.current == Some(id) {
                core.current = None;
                core.needs_scheduling = true;
            }
        }
        self.threads.remove(id).map_err(SchedError::from)?;
        Ok(())
    }

    /// Ends a thread (`KThread::Exit` for the calling thread, `RequestTerminate` for another):
    /// a wait it is in ends without a result and its pending cancel goes
    /// (`Scheduler::forget_waits`), it becomes [`ThreadState::Terminated`] and leaves the
    /// queues, and the threads waiting on it wake ([`Scheduler::notify_available`]). A thread
    /// that was started stops counting as one of its process's running threads, and closes its
    /// reference to itself, which destroys it if no handle or wait names it. A thread that has
    /// ended already is left as it is.
    pub fn terminate_thread(&mut self, id: ThreadId) -> CallResult<ThreadExit> {
        let thread = self.thread(id)?;
        if thread.state == ThreadState::Terminated {
            return Ok(ThreadExit::Others);
        }
        let started = thread.state != ThreadState::Initialized;
        let (process, object, holds_itself) = (thread.process, thread.object, thread.holds_itself);
        // A terminated thread must not stay on a wait list, or a later signal would make it
        // runnable again.
        let forgotten = self.forget_waits(id);
        self.set_state(id, ThreadState::Terminated)?;
        // WAIT SEAM (kernel slice 4): the thread is signalled now (`is_signalled` reads its
        // state), and the threads waiting on it (a join) wake, as `KThread::FinishTermination`
        // sets `m_signaled` and calls `NotifyAvailable` (`eden/src/core/hle/kernel/k_thread.cpp`
        // @ cf654052, line 444), before the thread's reference to itself is closed below.
        let notified = match object {
            Some(object) => self.notify_available(object),
            None => Ok(()),
        };
        let mut exit = ThreadExit::Others;
        if started && let Some(process) = process {
            let record = self.record_mut(process)?;
            record.running_threads = record
                .running_threads
                .checked_sub(1)
                .ok_or(SchedError::Corrupt)?;
            if record.running_threads == 0 {
                exit = ThreadExit::LastOf(process);
            }
        }
        if holds_itself && let Some(object) = object {
            self.thread_mut(id)?.holds_itself = false;
            self.close_object(object)?;
        }
        forgotten?;
        notified?;
        Ok(exit)
    }

    /// `ExitThread`: ends the thread the calling core runs ([`Scheduler::terminate_thread`]).
    /// A core running nothing is [`SchedError::NoCurrentThread`].
    pub fn exit_thread(&mut self, core: super::super::CoreId) -> CallResult<ThreadExit> {
        let current = self
            .current(core)?
            .ok_or(SchedError::NoCurrentThread { core: core.index() })?;
        self.terminate_thread(current)
    }

    /// Ends every thread of `process` that has not ended (`TerminateChildren`, without the
    /// waiting; see the module docs), in thread order.
    pub fn terminate_threads(&mut self, process: ProcessId) -> CallResult<()> {
        let threads: Vec<ThreadId> = self.record(process)?.threads().collect();
        for thread in threads {
            // The thread may be gone already: ending an earlier one cannot destroy it, but a
            // stale id here would be broken bookkeeping, which `terminate_thread` reports.
            self.terminate_thread(thread)?;
        }
        Ok(())
    }

    /// The virtual half of `KThread::SetCoreMask`: with `core_id` [`IDEAL_CORE_NO_UPDATE`] the
    /// thread keeps its virtual ideal core, which must be in `virtual_mask`
    /// ([`KernelError::InvalidCombination`] otherwise; for a "don't care" ideal core Eden's
    /// `1ULL << -1` is taken as bit 63, the guess the physical path makes too); then the ideal
    /// core and the mask are kept as the guest named them and translated to physical cores
    /// ([`crate::process::VIRTUAL_TO_PHYSICAL_CORE`]) for the scheduler.
    ///
    /// The system call has checked the arguments against the process. A physical core this
    /// scheduler lacks is [`KernelError::InvalidCoreId`], as is an ideal core below
    /// [`IDEAL_CORE_DONT_CARE`] other than [`IDEAL_CORE_NO_UPDATE`].
    pub fn set_virtual_core_mask(
        &mut self,
        id: ThreadId,
        core_id: i32,
        virtual_mask: u64,
    ) -> Result<(), SchedError> {
        let thread = self.thread(id)?;
        let core_id = if core_id == IDEAL_CORE_NO_UPDATE {
            let kept = thread.virtual_ideal_core;
            let bit =
                u32::try_from(kept).map_or(1 << 63, |shift| 1_u64.checked_shl(shift).unwrap_or(0));
            if virtual_mask & bit == 0 {
                return Err(KernelError::InvalidCombination.into());
            }
            kept
        } else {
            core_id
        };
        let physical_ideal = if core_id == IDEAL_CORE_DONT_CARE {
            None
        } else {
            let virtual_core = u8::try_from(core_id).map_err(|_| KernelError::InvalidCoreId)?;
            Some(physical_core(virtual_core).ok_or(KernelError::InvalidCoreId)?)
        };
        let mask = physical_mask(virtual_mask);
        if mask.bits() & !self.core_mask().bits() != 0
            || physical_ideal.is_some_and(|core| !self.core_mask().contains(core))
        {
            return Err(KernelError::InvalidCoreId.into());
        }
        let row = self.thread_mut(id)?;
        row.virtual_ideal_core = core_id;
        row.virtual_affinity = virtual_mask;
        self.apply_core_mask(id, physical_ideal, mask)
    }
}

#[cfg(test)]
mod tests;
