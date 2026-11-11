//! The system-call dispatcher: the only code that takes kernel locks. It acquires the tables a call
//! declares, in one fixed order, and records the event before releasing them.
//!
//! Kernel spec, "Tables behind a dispatcher, one lock to start" and "System calls":
//!
//! - **One lock.** To start, a single mutex guards every table ([`Tables`]). Acquiring the tables a
//!   call declares is taking that lock. When the lock is split (the spec's steps 1-3), the
//!   dispatcher takes the per-table locks in the fixed order handle table, address space,
//!   scheduler, and nothing outside this module changes.
//! - **One `match`** from system-call number to a typed function. Arguments are decoded with the
//!   call's [`Convention`], and outputs written back with it.
//! - **Unknown and unimplemented calls** (an unlisted number, or a call whose [`Status`] isn't
//!   [`Status::Implemented`]) are logged and return [`KernelError::NotImplemented`] in `x0`. Any
//!   system call number and any register values dispatch without a panic, given an [`Env`],
//!   [`GuestMemory`] and [`DebugSink`] that don't panic themselves (tests: every unlisted number
//!   in `0..=u16::MAX`, and a property test over any number, core and `x0`-`x7`).
//! - **One recording point** (observability spec, "One recording point per boundary"): every
//!   dispatch that returns to the guest ends in one `EventKind::Svc` event, recorded with the
//!   caller's [`Recorder`] while the lock is still held. The recorder assigns the event's `seq`
//!   there, under the kernel lock, so sorting by `seq` gives the kernel's order. The event's
//!   `result` is the call's `Result` output (the word in `x0` when the call's convention begins
//!   with one, or [`KernelError::NotImplemented`]); a call with no `Result` output
//!   (`GetSystemTick`, `ExitProcess`) records success, and its other outputs are not in the
//!   event, which has no field for them. A dispatch that ends in [`Outcome::EnvFailed`] records
//!   nothing: the call never returned, so it has no result, and the caller reports the failed
//!   replay itself.
//! - **`ExitProcess`** ends the dispatch in [`Outcome::ProcessExited`] and the caller stops
//!   running the guest. PROVISIONAL (walking skeleton, #74): there is no process or thread
//!   table to update yet, so the call changes no kernel state. Kernel slice 8 (process creation,
//!   thread lifecycle) replaces the dispatch arm with a call that terminates the process's
//!   threads, as Eden's does (`eden/src/core/hle/kernel/svc/svc_process.cpp` @ cf654052,
//!   `ExitProcess`).
//!
//! Intentional divergences from Eden (`eden/src/core/hle/kernel/svc.cpp` @ cf654052, `Call64`),
//! pinned by tests in `dispatch/tests.rs`:
//! - Eden's `Call64` reaches `UNREACHABLE_MSG` for a number it doesn't list. Here an unlisted
//!   number returns [`KernelError::NotImplemented`] (kernel spec, "System calls").
//! - Eden's generated wrapper for a call writes every output its convention lists (a void call
//!   writes none). Here a call that isn't implemented writes [`KernelError::NotImplemented`] to
//!   `x0` and leaves `x1`-`x7` as they were, whatever its convention lists, void calls included.
//! - When the environment fails (a replay that doesn't match its recording), the call writes
//!   nothing and the dispatch ends in [`Outcome::EnvFailed`]: there is no honest value to give the
//!   guest, and the run has already diverged. Eden reads its clock directly and can't fail here.
//!
//! # A call can't reach a table it didn't declare
//!
//! A system call is a function whose parameters are what it touches. The tables live inside the
//! [`Dispatcher`], behind its lock, and nothing public hands them out while it can still dispatch.
//! A call that declares only the environment compiles, even when it is also handed the dispatcher:
//!
//! ```
//! use starquake_kernel::dispatch::Dispatcher;
//! use starquake_trace::{Env, EnvError, GuestTicks};
//!
//! fn get_system_tick(env: &dyn Env, _dispatcher: &Dispatcher) -> Result<u64, EnvError> {
//!     env.now().map(|GuestTicks(ticks)| ticks)
//! }
//! ```
//!
//! The same call can't reach the scheduler through the dispatcher's lock, which is private
//! (`E0616`, a private field):
//!
//! ```compile_fail,E0616
//! use starquake_kernel::dispatch::Dispatcher;
//! use starquake_trace::{Env, EnvError, GuestTicks};
//!
//! fn get_system_tick(env: &dyn Env, _dispatcher: &Dispatcher) -> Result<u64, EnvError> {
//!     let _scheduler = &_dispatcher.tables;
//!     env.now().map(|GuestTicks(ticks)| ticks)
//! }
//! ```
//!
//! nor by taking the tables out, which consumes the dispatcher, and a call never owns it (`E0507`,
//! a move out of a borrow):
//!
//! ```compile_fail,E0507
//! use starquake_kernel::dispatch::Dispatcher;
//! use starquake_trace::{Env, EnvError, GuestTicks};
//!
//! fn get_system_tick(env: &dyn Env, _dispatcher: &Dispatcher) -> Result<u64, EnvError> {
//!     let _scheduler = _dispatcher.into_tables().scheduler;
//!     env.now().map(|GuestTicks(ticks)| ticks)
//! }
//! ```
//!
//! Each failing example differs from the compiling one by its `let _scheduler` line.

use std::sync::{Mutex, PoisonError};

use starquake_cpu::GuestRegs;
use starquake_memory::GuestMemory;
use starquake_trace::{Env, EnvError, EventKind, GuestTicks, Recorder, SVC_ARG_COUNT, ThreadId};

use crate::handle::HandleTable;
use crate::result::{KernelError, ResultCode, result_code};
use crate::svc::{
    Convention, ConventionError, DebugSink, Kind, Status, SvcId, debug, processor, tick,
};
use crate::tables::{AddressSpace, Processes, Scheduler};

#[cfg(test)]
mod tests;

/// The register that carries a call's result (Eden's wrappers: `SetArg64(args, 0, ret)`).
const RESULT_REGISTER: usize = 0;

/// Every kernel table, as the dispatcher's lock guards them. Fields are listed in lock order.
///
/// One of each for now: a second process brings its own handle table and address space (kernel
/// spec, rule 2), and this struct grows per-process maps.
#[derive(Debug)]
pub struct Tables {
    /// The process's handle table; first in lock order.
    pub handles: HandleTable,
    /// The process's address space; second in lock order.
    pub address_space: AddressSpace,
    /// Threads, run queues and wait lists; third in lock order.
    pub scheduler: Scheduler,
    /// Per-process facts fixed at creation. Needs no lock of its own.
    pub processes: Processes,
}

/// What a dispatch needs from the calling core besides the tables: which core and guest thread
/// it is, when the call was made, the environment, the caller's memory, where debug strings go,
/// and the event stream's recorder.
pub struct Caller<'a> {
    /// The index of the core making the call.
    pub core: u8,
    /// The guest thread making the call, as the event stream names it.
    ///
    /// PROVISIONAL (walking skeleton, #74): the core loop names the thread. Once the scheduler
    /// tracks which thread each core is running (kernel slice 10), the dispatcher reads it from
    /// there and this field goes away.
    pub thread: ThreadId,
    /// When the call was made, for the event's `time`. The core loop supplies it; the dispatcher
    /// does not read [`Env::now`] for it, so recording an event never adds a clock reading to the
    /// replay log. The recorder does not interpret the unit.
    pub time: GuestTicks,
    /// The injectable environment (observability spec, rule 1).
    pub env: &'a dyn Env,
    /// The calling process's memory.
    pub memory: &'a dyn GuestMemory,
    /// Where `OutputDebugString` sends its text.
    pub debug: &'a mut dyn DebugSink,
    /// The event stream's recorder: every call that returns to the guest is recorded here once.
    pub recorder: &'a Recorder,
}

/// How a dispatch ended.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Outcome {
    /// An implemented call ran and wrote its outputs (which may include an error result).
    Completed,
    /// The number is unlisted or the call isn't implemented: `x0` holds
    /// [`KernelError::NotImplemented`].
    NotImplemented,
    /// The environment failed, so the call wrote nothing. Only a replay that has diverged from its
    /// recording does this; the caller should stop the run and report it.
    EnvFailed(EnvError),
    /// `ExitProcess` ran: the calling process is over, and the caller must not resume the guest
    /// thread. No register was written.
    ProcessExited,
}

/// One dispatched system call, as the recording point sees it: the fields of the event stream's
/// `EventKind::Svc` that only the dispatcher knows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Record {
    /// The calling core.
    pub core: u8,
    /// The `svc` immediate.
    pub number: u16,
    /// The call, if the number is listed.
    pub id: Option<SvcId>,
    /// `x0`-`x7` as the guest passed them, before the call.
    pub args: [u64; SVC_ARG_COUNT],
    /// How it ended.
    pub outcome: Outcome,
}

/// Runs system calls against the kernel's tables, one at a time.
#[derive(Debug)]
pub struct Dispatcher {
    tables: Mutex<Tables>,
}

impl Dispatcher {
    /// A dispatcher over these tables.
    #[must_use]
    pub fn new(tables: Tables) -> Self {
        Self {
            tables: Mutex::new(tables),
        }
    }

    /// Handles `svc #number` from `caller`: decodes the arguments from `regs`, runs the call, and
    /// writes its outputs back to `regs`.
    pub fn dispatch(&self, number: u16, regs: &mut GuestRegs, caller: &mut Caller<'_>) -> Record {
        // A poisoned lock means a panic elsewhere; the tables are still plain data, so carry on.
        let guard = self.tables.lock().unwrap_or_else(PoisonError::into_inner);

        let mut args = [0; SVC_ARG_COUNT];
        for (arg, register) in args.iter_mut().zip(&regs.x) {
            *arg = *register;
        }
        let id = SvcId::from_number(number);
        let outcome = if let Some(id) = id {
            run(id, regs, caller)
        } else {
            tracing::warn!(number, "unknown system call");
            not_implemented(regs)
        };
        let record = Record {
            core: caller.core,
            number,
            id,
            args,
            outcome,
        };
        record_point(&record, regs, caller);
        drop(guard);
        record
    }

    /// Takes the tables back, for a snapshot or a test's inspection.
    #[must_use]
    pub fn into_tables(self) -> Tables {
        self.tables
            .into_inner()
            .unwrap_or_else(PoisonError::into_inner)
    }
}

/// RECORDING POINT for the event stream: the dispatcher's one call into the recorder
/// (observability spec, "The event stream"). It runs after the call with the kernel lock still
/// held, so the recorder's `seq` follows the kernel's order. See the module docs for what the
/// event's `result` holds and why a failed environment records nothing.
fn record_point(record: &Record, regs: &GuestRegs, caller: &Caller<'_>) {
    let Some(result) = recorded_result(record, regs) else {
        return;
    };
    caller.recorder.record(
        caller.time,
        caller.core,
        caller.thread,
        EventKind::Svc {
            id: u32::from(record.number),
            args: record.args,
            // The one place a kernel result becomes the event stream's result word.
            result: starquake_trace::ResultCode(result.raw()),
        },
    );
}

/// The result word a finished dispatch is recorded with, or `None` if the call never returned to
/// the guest.
fn recorded_result(record: &Record, regs: &GuestRegs) -> Option<ResultCode> {
    match &record.outcome {
        Outcome::EnvFailed(_) => None,
        Outcome::NotImplemented => Some(KernelError::NotImplemented.code()),
        Outcome::ProcessExited => Some(ResultCode::SUCCESS),
        Outcome::Completed => {
            let result = record
                .id
                .and_then(SvcId::convention)
                .and_then(|convention| convention.outputs.first())
                .filter(|output| output.kind == Kind::Result)
                .and_then(|output| {
                    let register = regs.x.get(usize::from(output.reg))?;
                    u32::try_from(output.kind.truncate(*register)).ok()
                });
            Some(result.map_or(ResultCode::SUCCESS, ResultCode::from_raw))
        }
    }
}

/// Runs one listed call. Calls that touch tables will get them here, borrowed from the locked
/// [`Tables`] in lock order; the calls implemented so far touch none.
fn run(id: SvcId, regs: &mut GuestRegs, caller: &mut Caller<'_>) -> Outcome {
    let Some(convention) = id.convention() else {
        return unimplemented(id, regs);
    };
    match id {
        SvcId::ExitProcess => {
            // PROVISIONAL (walking skeleton, #74): see the module docs.
            tracing::info!("the process called ExitProcess");
            Outcome::ProcessExited
        }
        SvcId::GetCurrentProcessorNumber => {
            let core = processor::get_current_processor_number(caller.core);
            complete(id, convention, regs, &[u64::from(core.cast_unsigned())])
        }
        SvcId::GetSystemTick => match tick::get_system_tick(caller.env) {
            Ok(tick) => complete(id, convention, regs, &[tick]),
            Err(error) => {
                tracing::error!(name = id.name(), %error, "environment failed");
                Outcome::EnvFailed(error)
            }
        },
        SvcId::OutputDebugString => match convention.read(regs) {
            Ok([address, len]) => {
                let result =
                    debug::output_debug_string(caller.memory, &mut *caller.debug, address, len);
                complete(id, convention, regs, &[result_word(result_code(&result))])
            }
            Err(error) => mismatch(id, error, regs),
        },
        _ => unimplemented(id, regs),
    }
}

/// Writes an implemented call's outputs.
fn complete(id: SvcId, convention: &Convention, regs: &mut GuestRegs, outputs: &[u64]) -> Outcome {
    match convention.write(regs, outputs) {
        Ok(()) => Outcome::Completed,
        Err(error) => mismatch(id, error, regs),
    }
}

/// An implemented call disagrees with its convention: a dispatcher bug. The guest sees "not
/// implemented" rather than garbage.
fn mismatch(id: SvcId, error: ConventionError, regs: &mut GuestRegs) -> Outcome {
    tracing::error!(name = id.name(), %error, "system call does not match its convention");
    not_implemented(regs)
}

fn unimplemented(id: SvcId, regs: &mut GuestRegs) -> Outcome {
    let status = id.status();
    if status == Status::Implemented {
        // The status table and this dispatcher disagree; a test keeps them in step.
        tracing::error!(
            name = id.name(),
            "implemented system call has no dispatch arm"
        );
    } else {
        tracing::warn!(
            name = id.name(),
            number = id.number(),
            ?status,
            "system call not implemented"
        );
    }
    not_implemented(regs)
}

fn not_implemented(regs: &mut GuestRegs) -> Outcome {
    regs.x[RESULT_REGISTER] = result_word(KernelError::NotImplemented.code());
    Outcome::NotImplemented
}

/// A result as the register holds it: the 32-bit word, zero-extended.
fn result_word(code: ResultCode) -> u64 {
    u64::from(code.raw())
}
