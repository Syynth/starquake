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
//!   [`Status::Implemented`]) are logged and return [`KernelError::NotImplemented`] in `x0`. No
//!   input makes the dispatcher panic.
//! - **One recording point** (observability spec, "One recording point per boundary"): every
//!   dispatch produces a [`Record`] with its sequence number, assigned and recorded while the lock
//!   is held, so sorting by `seq` gives the kernel's order.
//!
//! Intentional divergences from Eden (`eden/src/core/hle/kernel/svc.cpp` @ cf654052, `Call64`),
//! pinned by tests in `dispatch/tests.rs`:
//! - Eden's `Call64` reaches `UNREACHABLE_MSG` for a number it doesn't list. Here an unlisted
//!   number returns [`KernelError::NotImplemented`] (kernel spec, "System calls").
//! - Eden's generated wrapper for a call writes every output its convention lists (a void call
//!   writes none). Here a call that isn't implemented writes [`KernelError::NotImplemented`] to
//!   `x0` and leaves `x1`-`x7` as they were, whatever its convention lists, void calls included.
//!
//! # A call can't reach a table it didn't declare
//!
//! A system call is a function whose parameters are what it touches. The tables live inside the
//! [`Dispatcher`], behind its lock, and nothing public hands them out while it can still dispatch.
//! A call that declares only the environment compiles, even when it is also handed the dispatcher:
//!
//! ```
//! use starquake_kernel::dispatch::Dispatcher;
//! use starquake_kernel::svc::Env;
//!
//! fn get_system_tick(env: &dyn Env, _dispatcher: &Dispatcher) -> u64 {
//!     env.system_tick()
//! }
//! ```
//!
//! The same call can't reach the scheduler through the dispatcher's lock, which is private
//! (`E0616`, a private field):
//!
//! ```compile_fail,E0616
//! use starquake_kernel::dispatch::Dispatcher;
//! use starquake_kernel::svc::Env;
//!
//! fn get_system_tick(env: &dyn Env, _dispatcher: &Dispatcher) -> u64 {
//!     let _scheduler = &_dispatcher.state;
//!     env.system_tick()
//! }
//! ```
//!
//! nor by taking the tables out, which consumes the dispatcher, and a call never owns it (`E0507`,
//! a move out of a borrow):
//!
//! ```compile_fail,E0507
//! use starquake_kernel::dispatch::Dispatcher;
//! use starquake_kernel::svc::Env;
//!
//! fn get_system_tick(env: &dyn Env, _dispatcher: &Dispatcher) -> u64 {
//!     let _scheduler = _dispatcher.into_tables().scheduler;
//!     env.system_tick()
//! }
//! ```
//!
//! Each failing example differs from the compiling one by its `let _scheduler` line.

use std::sync::{Mutex, PoisonError};

use starquake_cpu::GuestRegs;
use starquake_memory::GuestMemory;

use crate::handle::HandleTable;
use crate::result::{KernelError, ResultCode, result_code};
use crate::svc::{
    Convention, ConventionError, DebugSink, Env, Status, SvcId, debug, processor, tick,
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
#[derive(Debug, Default)]
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

/// What a dispatch needs from the calling core besides the tables: which core it is, the
/// environment, the caller's memory, and where debug strings go.
pub struct Caller<'a> {
    /// The index of the core making the call.
    pub core: u8,
    /// The injectable environment (clock).
    pub env: &'a dyn Env,
    /// The calling process's memory.
    pub memory: &'a dyn GuestMemory,
    /// Where `OutputDebugString` sends its text.
    pub debug: &'a mut dyn DebugSink,
}

/// How a dispatch ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Outcome {
    /// An implemented call ran and wrote its outputs (which may include an error result).
    Completed,
    /// The number is unlisted or the call isn't implemented: `x0` holds
    /// [`KernelError::NotImplemented`].
    NotImplemented,
}

/// One dispatched system call, as the recording point sees it.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Record {
    /// Position in the kernel's global order, assigned under the lock; starts at 0.
    pub seq: u64,
    /// The calling core.
    pub core: u8,
    /// The `svc` immediate.
    pub number: u16,
    /// The call, if the number is listed.
    pub id: Option<SvcId>,
    /// How it ended.
    pub outcome: Outcome,
}

/// The lock's contents: the tables and the next sequence number.
#[derive(Debug)]
struct State {
    tables: Tables,
    next_seq: u64,
}

/// Runs system calls against the kernel's tables, one at a time.
#[derive(Debug)]
pub struct Dispatcher {
    state: Mutex<State>,
}

impl Dispatcher {
    /// A dispatcher over these tables. The first dispatch gets sequence number 0.
    #[must_use]
    pub fn new(tables: Tables) -> Self {
        Self {
            state: Mutex::new(State {
                tables,
                next_seq: 0,
            }),
        }
    }

    /// Handles `svc #number` from `caller`: decodes the arguments from `regs`, runs the call, and
    /// writes its outputs back to `regs`.
    pub fn dispatch(&self, number: u16, regs: &mut GuestRegs, caller: &mut Caller<'_>) -> Record {
        // A poisoned lock means a panic elsewhere; the tables are still plain data, so carry on.
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        let seq = state.next_seq;
        // Not guest-controlled: one per call, so it can't wrap in practice.
        state.next_seq = seq.wrapping_add(1);

        let id = SvcId::from_number(number);
        let outcome = if let Some(id) = id {
            run(id, regs, caller)
        } else {
            tracing::warn!(number, "unknown system call");
            not_implemented(regs)
        };
        let record = Record {
            seq,
            core: caller.core,
            number,
            id,
            outcome,
        };
        record_point(&record);
        drop(state);
        record
    }

    /// Takes the tables back, for a snapshot or a test's inspection.
    #[must_use]
    pub fn into_tables(self) -> Tables {
        self.state
            .into_inner()
            .unwrap_or_else(PoisonError::into_inner)
            .tables
    }
}

/// RECORDING POINT for the event stream: the dispatcher's one call into the recorder
/// (observability spec, "The event stream"). A no-op until the recorder lands. It runs with the
/// kernel lock held, after the call, so events are recorded in `seq` order.
fn record_point(_record: &Record) {}

/// Runs one listed call. Calls that touch tables will get them here, borrowed from the locked
/// [`Tables`] in lock order; the calls implemented so far touch none.
fn run(id: SvcId, regs: &mut GuestRegs, caller: &mut Caller<'_>) -> Outcome {
    let Some(convention) = id.convention() else {
        return unimplemented(id, regs);
    };
    match id {
        SvcId::GetCurrentProcessorNumber => {
            let core = processor::get_current_processor_number(caller.core);
            complete(id, convention, regs, &[u64::from(core.cast_unsigned())])
        }
        SvcId::GetSystemTick => {
            let tick = tick::get_system_tick(caller.env);
            complete(id, convention, regs, &[tick])
        }
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
