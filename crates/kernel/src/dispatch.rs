//! The system-call dispatcher: the only code that takes kernel locks. It acquires the tables a call
//! declares, in one fixed order, hands the call exactly those, and records the event before
//! releasing them.
//!
//! Kernel spec, "Tables behind a dispatcher, one lock to start", "System calls" and "Guest time":
//!
//! - **Declared tables.** A system call is a function whose parameters are the tables it touches
//!   (`fn close_handle(handles: &mut HandleTable, scheduler: &mut Scheduler, ..)`). Its dispatch
//!   arm names the same set as a type, a tuple of `&Table` and `&mut Table` in lock order
//!   ([`TableSet`]), and receives those borrows and nothing else. See "A call can't reach a
//!   table it didn't declare" below.
//! - **One lock.** To start, a single mutex guards every table ([`Tables`]). Acquiring the tables
//!   a call declares is taking that lock and splitting the borrow. When the lock is split (the
//!   spec's steps 1-3), the dispatcher takes the per-table locks in the fixed order handle
//!   table, address space, scheduler, shared or exclusive as [`TableSet::DECLARED`] says, and
//!   nothing outside this module changes.
//! - **Lock waits are measured** ([`Dispatcher::lock_waits`]), per table, so the decision to
//!   split is made from numbers. See [`LockWaits`] for what is charged to a table.
//! - **One clock reading per dispatch.** After taking the lock, the dispatcher reads the
//!   environment's clock once and converts it to guest ticks with the process's counter rate
//!   ([`crate::time`]). That one reading is the event's `time` and what `GetSystemTick`
//!   returns, and is handed to any call that needs the time. System calls never read the clock
//!   themselves, so they stay pure functions of their tables and arguments, and a replay feeds
//!   back exactly one reading per dispatch, in the kernel's order.
//! - **One `match`** from system-call number to a typed function. Arguments are decoded with the
//!   call's [`Convention`], and outputs written back with it.
//! - **Unknown and unimplemented calls** (an unlisted number, or a call whose [`Status`] isn't
//!   [`Status::Implemented`]) are logged with their status and return
//!   [`KernelError::NotImplemented`] in `x0`. Any system call number and any register values
//!   dispatch without a panic, given an [`Env`], [`GuestMemory`] and [`DebugSink`] that don't
//!   panic themselves (tests: every unlisted number in `0..=u16::MAX`, and a property test over
//!   any number, core and `x0`-`x7`).
//! - **One recording point** (observability spec, "One recording point per boundary"): every
//!   dispatch that returns to the guest ends in one `EventKind::Svc` event, recorded with the
//!   caller's [`Recorder`] while the lock is still held. The recorder assigns the event's `seq`
//!   there, under the kernel lock, so sorting by `seq` gives the kernel's order. The event's
//!   `args` are the call's decoded arguments (observability spec: "system call, with decoded
//!   arguments"): each input of the call's [`Convention`], truncated to its kind, in the slot of
//!   the register that carried it, and zero everywhere else. What an earlier call left in a
//!   register the call does not take never reaches the event, so a trace does not depend on it.
//!   A number with no convention (an unlisted number, or a listed call whose row is not in the
//!   table yet) keeps all eight registers as passed: nothing says which are arguments, and the
//!   trace is the to-do list for such calls. The event's
//!   `result` is the call's `Result` output (the word in `x0` when the call's convention begins
//!   with one, or [`KernelError::NotImplemented`]); a call with no `Result` output
//!   (`GetSystemTick`, `ExitProcess`) records success, and its other outputs are not in the
//!   event, which has no field for them. A dispatch that ends in [`Outcome::EnvFailed`] records
//!   nothing: the call never returned, so it has no result, and the caller reports the failed
//!   replay itself.
//! - **IPC** (`ConnectToNamedPort`, `SendSyncRequest`) has arms of its own, in `dispatch/ipc.rs`:
//!   `SendSyncRequest` takes the lock more than once and calls a server's handler without it,
//!   and a request that cannot be answered at once ends the dispatch in [`Outcome::Waiting`],
//!   recording nothing until the request ends. Its event, and the IPC events its handler made
//!   before it, are recorded under the lock then, with the dispatch's one clock reading.
//! - **Synchronisation** (events, `WaitSynchronization`, `CancelSynchronization`) has arms of
//!   its own, in `dispatch/sync.rs`. A `WaitSynchronization` that waits ends in
//!   [`Outcome::Waiting`] the way `SendSyncRequest` does, and its event is recorded when the
//!   wait ends, by whichever dispatch (or [`Dispatcher::signal_event`],
//!   [`Dispatcher::fire_timers`]) records the wake. `dispatch/sync.rs` says how.
//! - **`ExitProcess`** ends the dispatch in [`Outcome::ProcessExited`] and the caller stops
//!   running the guest. PROVISIONAL (walking skeleton, #74): there is no process or thread
//!   table to update yet, so the call changes no kernel state. Kernel slice 8 (process creation,
//!   thread lifecycle) replaces the dispatch arm with a call that terminates the process's
//!   threads, as Eden's does (`eden/src/core/hle/kernel/svc/svc_process.cpp` @ cf654052,
//!   `ExitProcess`).
//!
//! Intentional divergences from Eden (`eden/src/core/hle/kernel/svc.cpp` @ cf654052, `Call64`),
//! pinned by the named tests in `dispatch/tests.rs`:
//! - Eden's `Call64` reaches `UNREACHABLE_MSG` for a number it doesn't list. Here an unlisted
//!   number returns [`KernelError::NotImplemented`] (kernel spec, "System calls"). Test:
//!   `unknown_numbers_return_not_implemented_and_touch_only_x0`.
//! - Eden's generated wrapper for a call writes every output its convention lists (a void call
//!   writes none). Here a call that isn't implemented writes [`KernelError::NotImplemented`] to
//!   `x0` and leaves `x1`-`x7` as they were, whatever its convention lists, void calls included.
//!   Test: `a_planned_call_returns_not_implemented_and_leaves_its_outputs_alone`. (An
//!   implemented call that fails with a result the guest sees does write them, as zeros. That
//!   matches Eden's wrapper only for a call whose body assigns no output before it fails: the
//!   wrapper value-initialises the outputs and writes whatever the body left in them.
//!   `GetThreadPriority` is such a call. The docs of `answering`, the function that writes
//!   them, name a call that is not, `WaitSynchronization`, which writes its index with an error
//!   result. Test:
//!   `a_failing_call_writes_its_result_and_zeroes_its_other_outputs`.)
//! - When the environment's clock fails (a replay that doesn't match its recording), the call
//!   does not run and writes nothing, and the dispatch ends in [`Outcome::EnvFailed`]: there is
//!   no honest value to give the guest, and the run has already diverged. Eden reads its clock
//!   directly and can't fail here. Test: `a_failing_environment_ends_the_call_without_writing_registers`.
//! - When a table operation fails in a way no guest input explains ([`SvcError::Internal`]: the
//!   kernel's own bookkeeping is inconsistent, or the core loop has not told the scheduler what
//!   it runs), the failure is logged as an error and the guest sees
//!   [`KernelError::NotImplemented`], with `x1`-`x7` untouched, as for a call that isn't
//!   implemented. Eden asserts. The run goes on, and its summary lists the call as stubbed. Test:
//!   `an_internal_failure_is_not_implemented_to_the_guest`.
//!
//! # A call can't reach a table it didn't declare
//!
//! The tables live inside the [`Dispatcher`], behind its lock. The only way to them is to name a
//! [`TableSet`]: [`Dispatcher::with`] (and every dispatch arm, which goes through the same
//! private step) locks, splits the borrow, and passes exactly the declared tables. A system call
//! that closes a handle declares the handle table and the scheduler, and gets them:
//!
//! ```
//! use starquake_kernel::dispatch::Dispatcher;
//! use starquake_kernel::handle::{Handle, HandleTable};
//! use starquake_kernel::result::CallResult;
//! use starquake_kernel::tables::Scheduler;
//!
//! fn close_handle(handles: &mut HandleTable, scheduler: &mut Scheduler, handle: Handle) -> CallResult<()> {
//!     handles.remove(scheduler.objects_mut(), handle)?;
//!     Ok(())
//! }
//!
//! fn run(dispatcher: &Dispatcher, handle: Handle) -> CallResult<()> {
//!     dispatcher.with::<(&mut HandleTable, &mut Scheduler), _>(|(handles, scheduler)| {
//!         close_handle(handles, scheduler, handle)
//!     })
//! }
//! ```
//!
//! Declaring only the handle table and still asking for the scheduler does not compile: the
//! call is handed a one-table tuple, and there is no second table in it (`E0308`, mismatched
//! types):
//!
//! ```compile_fail,E0308
//! use starquake_kernel::dispatch::Dispatcher;
//! use starquake_kernel::handle::{Handle, HandleTable};
//! use starquake_kernel::result::CallResult;
//! use starquake_kernel::tables::Scheduler;
//!
//! fn close_handle(handles: &mut HandleTable, scheduler: &mut Scheduler, handle: Handle) -> CallResult<()> {
//!     handles.remove(scheduler.objects_mut(), handle)?;
//!     Ok(())
//! }
//!
//! fn run(dispatcher: &Dispatcher, handle: Handle) -> CallResult<()> {
//!     dispatcher.with::<(&mut HandleTable,), _>(|(handles, scheduler)| {
//!         close_handle(handles, scheduler, handle)
//!     })
//! }
//! ```
//!
//! A table declared shared can't be changed (`E0596`, a mutable borrow of a `&` reference):
//!
//! ```compile_fail,E0596
//! use starquake_kernel::dispatch::Dispatcher;
//! use starquake_kernel::handle::{Handle, HandleTable};
//! use starquake_kernel::result::CallResult;
//! use starquake_kernel::tables::Scheduler;
//!
//! fn run(dispatcher: &Dispatcher, handle: Handle) -> CallResult<()> {
//!     dispatcher.with::<(&HandleTable, &Scheduler), _>(|(handles, scheduler)| {
//!         handles.remove(scheduler.objects_mut(), handle)?;
//!         Ok(())
//!     })
//! }
//! ```
//!
//! The tables are taken in the fixed order, so a set that names them out of order, or one table
//! twice, is not a [`TableSet`] (`E0277`, an unsatisfied trait bound):
//!
//! ```compile_fail,E0277
//! use starquake_kernel::dispatch::Dispatcher;
//! use starquake_kernel::handle::HandleTable;
//! use starquake_kernel::tables::Scheduler;
//!
//! fn run(dispatcher: &Dispatcher) {
//!     dispatcher.with::<(&mut Scheduler, &mut HandleTable), _>(|(_scheduler, _handles)| ());
//! }
//! ```
//!
//! ```compile_fail,E0277
//! use starquake_kernel::dispatch::Dispatcher;
//! use starquake_kernel::tables::Scheduler;
//!
//! fn run(dispatcher: &Dispatcher) {
//!     dispatcher.with::<(&mut Scheduler, &Scheduler), _>(|(_first, _second)| ());
//! }
//! ```
//!
//! And nothing reaches around the declaration: the lock is private (`E0616`, a private field).
//!
//! ```compile_fail,E0616
//! use starquake_kernel::dispatch::Dispatcher;
//!
//! fn run(dispatcher: &Dispatcher) {
//!     let _everything = &dispatcher.state;
//! }
//! ```
//!
//! Each failing example differs from the compiling one only in the lines the text above it
//! names.

use std::collections::BTreeMap;
use std::fmt;
use std::marker::PhantomData;
use std::sync::{Mutex, MutexGuard, PoisonError, TryLockError};
use std::time::Duration;

use starquake_cpu::GuestRegs;
use starquake_memory::GuestMemory;
use starquake_trace::{
    Env, EnvError, EventKind, GuestTicks, HostNanos, Recorder, SVC_ARG_COUNT, ThreadId,
};

use crate::handle::{Handle, HandleTable};
use crate::id::{ObjectId, ServerId};
use crate::ipc::SessionHandler;
use crate::result::{CallResult, KernelError, ResultCode, SvcError};
use crate::sched::CoreId;
use crate::svc::{
    Convention, ConventionError, DebugSink, Kind, Status, SvcId, debug, handle, info, memory,
    processor, thread, tick,
};
use crate::tables::{AddressSpace, Processes, Scheduler};

mod ipc;
mod sync;
#[cfg(test)]
mod tests;

pub use ipc::{IpcContext, Woken};
pub use sync::WakeError;

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
    /// Threads, run queues, wait lists and the object table; third in lock order.
    pub scheduler: Scheduler,
    /// Per-process facts fixed at creation. Needs no lock of its own, so no call declares it:
    /// the dispatcher reads the counter rate from it to keep guest time.
    pub processes: Processes,
}

/// How a call uses a table it declares: `&Table` or `&mut Table`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Access {
    /// `&Table`: the call only reads it.
    Shared,
    /// `&mut Table`: the call may change it.
    Exclusive,
}

/// The tables a [`TableSet`] declares, and how. One field per lockable table, named as in
/// [`Tables`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Declared {
    /// The handle table.
    pub handles: Option<Access>,
    /// The address space.
    pub address_space: Option<Access>,
    /// The scheduler.
    pub scheduler: Option<Access>,
}

impl Declared {
    /// No table at all: what a call such as `GetSystemTick` declares.
    pub const NONE: Self = Self {
        handles: None,
        address_space: None,
        scheduler: None,
    };
}

mod sealed {
    /// Keeps [`TableSet`](super::TableSet) closed: the sets are exactly the ones this module
    /// lists.
    pub trait Sealed {}
}

/// The tables a system call declares: `()`, or a tuple of `&Table` and `&mut Table` naming each
/// table at most once, in lock order (handle table, address space, scheduler). For example
/// `(&mut HandleTable, &Scheduler)`; a one-table set is a one-element tuple, `(&Scheduler,)`.
///
/// Implemented for exactly those tuples, so declaring the tables out of order or twice does not
/// compile (see the module docs).
pub trait TableSet: sealed::Sealed {
    /// The borrows the call receives: the same tuple, borrowed for as long as the dispatcher
    /// holds the tables.
    type Borrowed<'a>;

    /// Which tables the set names, and how.
    const DECLARED: Declared;

    /// Splits the locked tables into this set's borrows. Called by the dispatcher; code outside
    /// it has no locked [`Tables`] to pass.
    fn split(tables: &mut Tables) -> Self::Borrowed<'_>;
}

impl sealed::Sealed for () {}

impl TableSet for () {
    type Borrowed<'a> = ();

    const DECLARED: Declared = Declared::NONE;

    fn split(_: &mut Tables) -> Self::Borrowed<'_> {}
}

/// Implements [`TableSet`] for every tuple over the listed tables, shared and exclusive. Each
/// row is one set of tables in lock order; the `@place` rules walk the row, placing each table
/// once as `&Table` and once as `&mut Table`.
macro_rules! table_sets {
    ($([$($field:ident: $table:ident),+])*) => {
        $( table_sets!(@place [] $($field: $table,)+); )*
    };
    (@place [$(($field:ident, $table:ident, $access:ident $(, $mutable:tt)?))*]) => {
        impl sealed::Sealed for ($(&$($mutable)? $table,)*) {}

        impl TableSet for ($(&$($mutable)? $table,)*) {
            type Borrowed<'a> = ($(&'a $($mutable)? $table,)*);

            const DECLARED: Declared = {
                let mut declared = Declared::NONE;
                $(declared.$field = Some(Access::$access);)*
                declared
            };

            fn split(tables: &mut Tables) -> Self::Borrowed<'_> {
                ($(&$($mutable)? tables.$field,)*)
            }
        }
    };
    (@place [$($placed:tt)*] $field:ident: $table:ident, $($rest:tt)*) => {
        table_sets!(@place [$($placed)* ($field, $table, Shared)] $($rest)*);
        table_sets!(@place [$($placed)* ($field, $table, Exclusive, mut)] $($rest)*);
    };
}

table_sets! {
    [handles: HandleTable]
    [address_space: AddressSpace]
    [scheduler: Scheduler]
    [handles: HandleTable, address_space: AddressSpace]
    [handles: HandleTable, scheduler: Scheduler]
    [address_space: AddressSpace, scheduler: Scheduler]
    [handles: HandleTable, address_space: AddressSpace, scheduler: Scheduler]
}

/// How the dispatcher times a wait for its lock: a monotonic host clock.
///
/// Not the [`Env`]: these readings measure the host, never reach kernel state or a guest, and
/// are not recorded for replay. The kernel does not read the host's clock itself (observability
/// spec, rule 1), so whoever builds the dispatcher supplies one
/// ([`Dispatcher::with_lock_timer`]). It is read only when the lock was found held, twice per
/// such acquisition; an acquisition that did not have to wait reads nothing.
pub trait LockTimer: Send + Sync {
    /// The time since some fixed point. Successive readings must not decrease.
    fn now(&self) -> Duration;
}

/// Counts for one lock, or for the calls that declared one table.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Waits {
    /// How many times the lock was taken.
    pub acquisitions: u64,
    /// How many of those declared the table exclusively (`&mut`). For [`LockWaits::lock`]: how
    /// many declared any table exclusively.
    pub exclusive: u64,
    /// How many of them found the lock held and had to wait.
    pub contended: u64,
    /// The total time those spent waiting. Zero unless the dispatcher has a [`LockTimer`].
    pub waited: Duration,
}

impl Waits {
    fn note(&mut self, access: Access, contended: Option<Duration>) {
        self.acquisitions = self.acquisitions.saturating_add(1);
        if access == Access::Exclusive {
            self.exclusive = self.exclusive.saturating_add(1);
        }
        if let Some(waited) = contended {
            self.contended = self.contended.saturating_add(1);
            self.waited = self.waited.saturating_add(waited);
        }
    }
}

/// How long calls waited for the dispatcher's lock, per table (kernel spec: "The dispatcher
/// reports how long calls wait for each lock in the run's summary, so the decision to split is
/// made from measurements").
///
/// While one mutex guards every table, a call waits for that mutex whatever it declared. Its
/// wait is counted once in [`LockWaits::lock`] and once more under each table it declared, so a
/// table's numbers are what the calls that need that table lost to the single lock: the most
/// that giving the table a lock of its own could save them. A call that declares no table is
/// counted in [`LockWaits::lock`] only. The `exclusive` counts say how many acquisitions a
/// reader-writer lock could not have shared.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LockWaits {
    /// Whether waits were timed: the dispatcher has a [`LockTimer`]. When `false`, contended
    /// acquisitions are still counted and every `waited` is zero.
    pub timed: bool,
    /// Every acquisition of the one lock: each dispatch, and each [`Dispatcher::with`].
    pub lock: Waits,
    /// Acquisitions that declared the handle table.
    pub handles: Waits,
    /// Acquisitions that declared the address space.
    pub address_space: Waits,
    /// Acquisitions that declared the scheduler.
    pub scheduler: Waits,
}

impl LockWaits {
    /// Counts one acquisition by a call that declared `declared`, which waited `contended` if
    /// the lock was held.
    fn note(&mut self, declared: Declared, contended: Option<Duration>) {
        let tables = [
            (&mut self.handles, declared.handles),
            (&mut self.address_space, declared.address_space),
            (&mut self.scheduler, declared.scheduler),
        ];
        let mut any = Access::Shared;
        for (waits, access) in tables {
            if let Some(access) = access {
                waits.note(access, contended);
                if access == Access::Exclusive {
                    any = Access::Exclusive;
                }
            }
        }
        self.lock.note(any, contended);
    }
}

/// What a dispatch needs from the calling core besides the tables: which core and guest thread
/// it is, the environment, the caller's memory, where debug strings go, and the event stream's
/// recorder.
pub struct Caller<'a> {
    /// The index of the core making the call.
    pub core: u8,
    /// The guest thread making the call, as the event stream names it.
    ///
    /// PROVISIONAL (walking skeleton, #74): the core loop names the thread. Once the scheduler
    /// tracks which thread each core is running (kernel slice 10), the dispatcher reads it from
    /// there and this field goes away.
    pub thread: ThreadId,
    /// The injectable environment (observability spec, rule 1). The dispatcher reads its clock
    /// once per dispatch.
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
    /// `x0` holds [`KernelError::NotImplemented`]: the number is unlisted, the call isn't
    /// implemented, or it failed inside the kernel (see the module docs).
    NotImplemented,
    /// The environment's clock failed, so the call did not run and wrote nothing. Only a replay
    /// that has diverged from its recording does this; the caller should stop the run and
    /// report it.
    EnvFailed(EnvError),
    /// `ExitProcess` ran: the calling process is over, and the caller must not resume the guest
    /// thread. No register was written.
    ProcessExited,
    /// The calling thread waits: a `SendSyncRequest` whose request waits behind another or was
    /// deferred, or a `WaitSynchronization` that nothing has ended yet. No register was written
    /// and no event recorded; both happen when the wait ends, and the thread is then reported
    /// by [`Dispatcher::take_woken`]. The caller must not resume the thread until then. After a
    /// `WaitSynchronization` the caller's registers are already saved as its thread's context,
    /// where the wake writes its result: the core loop must not save over them
    /// (`dispatch/sync.rs`).
    Waiting,
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
    /// `x0`-`x7` as the guest passed them, before the call: every register, whether or not the
    /// call takes it. (The recorded event carries the decoded arguments instead.)
    pub args: [u64; SVC_ARG_COUNT],
    /// How it ended.
    pub outcome: Outcome,
}

/// What the dispatcher's lock guards.
struct State {
    tables: Tables,
    waits: LockWaits,
    /// Each server's handler, in its slot while no request runs on it (`dispatch::ipc`).
    handlers: BTreeMap<ServerId, Option<Box<dyn SessionHandler>>>,
    /// Senders woken by other threads' dispatches, for [`Dispatcher::take_woken`].
    woken: Vec<Woken>,
    /// Requests started on a server while its handler was out of its slot without holding a
    /// worker (delivering deferred replies), oldest first: whoever puts the handler back runs
    /// them (`dispatch::ipc`).
    parked: BTreeMap<ServerId, Vec<ObjectId>>,
}

/// One reading of the environment's clock, as host nanoseconds and as the process's guest ticks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Now {
    /// The reading itself: what timeouts are measured on.
    host: HostNanos,
    /// The reading in guest ticks: what events are stamped with.
    guest: GuestTicks,
}

/// The locked tables, as a call that declared `B` may see them.
struct Locked<'d, B> {
    state: MutexGuard<'d, State>,
    declared: PhantomData<fn() -> B>,
}

impl<B: TableSet> Locked<'_, B> {
    /// The declared tables, and nothing else.
    fn tables(&mut self) -> B::Borrowed<'_> {
        B::split(&mut self.state.tables)
    }

    /// One reading of the environment's clock, and the guest time it is at the process's
    /// counter rate.
    fn clock(&self, env: &dyn Env) -> Result<Now, EnvError> {
        let host = env.now()?;
        let guest = self.state.tables.processes.counter_rate().ticks_at(host);
        Ok(Now { host, guest })
    }

    /// Guest time at a reading of the environment's clock, at the process's counter rate.
    fn guest_time(&self, env: &dyn Env) -> Result<GuestTicks, EnvError> {
        self.clock(env).map(|now| now.guest)
    }
}

/// One dispatch in progress: the call, the registers as passed, and where its outputs go.
struct Call<'c, 'k> {
    number: u16,
    id: Option<SvcId>,
    passed: [u64; SVC_ARG_COUNT],
    regs: &'c mut GuestRegs,
    caller: &'c mut Caller<'k>,
}

/// Runs system calls against the kernel's tables, one at a time.
pub struct Dispatcher {
    state: Mutex<State>,
    timer: Option<Box<dyn LockTimer>>,
}

impl fmt::Debug for Dispatcher {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Dispatcher")
            .field("timed", &self.timer.is_some())
            .finish_non_exhaustive()
    }
}

impl Dispatcher {
    /// A dispatcher over these tables. Its lock waits are counted but not timed until it is
    /// given a timer ([`Dispatcher::with_lock_timer`]).
    #[must_use]
    pub fn new(tables: Tables) -> Self {
        Self {
            state: Mutex::new(State {
                tables,
                waits: LockWaits::default(),
                handlers: BTreeMap::new(),
                woken: Vec::new(),
                parked: BTreeMap::new(),
            }),
            timer: None,
        }
    }

    /// Times this dispatcher's lock waits with `timer`.
    #[must_use]
    pub fn with_lock_timer(mut self, timer: impl LockTimer + 'static) -> Self {
        self.timer = Some(Box::new(timer));
        self
    }

    /// A reading of the lock timer, or zero when there is none, so that an untimed wait
    /// measures as zero.
    fn timer_now(&self) -> Duration {
        self.timer
            .as_ref()
            .map_or(Duration::ZERO, |timer| timer.now())
    }

    /// Takes the lock for a call that declares `B`, counting the acquisition and timing the
    /// wait if there was one.
    fn lock<B: TableSet>(&self) -> Locked<'_, B> {
        // A poisoned lock means a panic elsewhere; the tables are still plain data, so carry on.
        let (mut state, contended) = match self.state.try_lock() {
            Ok(state) => (state, None),
            Err(TryLockError::Poisoned(poisoned)) => (poisoned.into_inner(), None),
            Err(TryLockError::WouldBlock) => {
                let started = self.timer_now();
                let state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
                (state, Some(self.timer_now().saturating_sub(started)))
            }
        };
        state.waits.note(B::DECLARED, contended);
        Locked {
            state,
            declared: PhantomData,
        }
    }

    /// Runs `f` with exactly the tables `B` declares, under the lock: the step every dispatch
    /// arm goes through, offered so that the tables can be set up and inspected between
    /// dispatches (a test creating a thread, the core loop creating its process). Counted in
    /// [`Dispatcher::lock_waits`] like a dispatch. Records no event and reads no clock.
    ///
    /// The lock is held while `f` runs, and it is not re-entrant: `f` must not dispatch, call
    /// `with` again or read [`Dispatcher::lock_waits`] on this dispatcher.
    pub fn with<B: TableSet, R>(&self, f: impl for<'a> FnOnce(B::Borrowed<'a>) -> R) -> R {
        let mut locked = self.lock::<B>();
        f(locked.tables())
    }

    /// How long calls have waited for the lock so far, per table. Reading this takes the lock
    /// briefly and is not itself counted.
    #[must_use]
    pub fn lock_waits(&self) -> LockWaits {
        let state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        LockWaits {
            timed: self.timer.is_some(),
            ..state.waits
        }
    }

    /// Handles `svc #number` from `caller`: decodes the arguments from `regs`, runs the call, and
    /// writes its outputs back to `regs`.
    pub fn dispatch(&self, number: u16, regs: &mut GuestRegs, caller: &mut Caller<'_>) -> Record {
        let mut passed = [0; SVC_ARG_COUNT];
        for (arg, register) in passed.iter_mut().zip(&regs.x) {
            *arg = *register;
        }
        let id = SvcId::from_number(number);
        let core = caller.core;
        let outcome = self.run(&mut Call {
            number,
            id,
            passed,
            regs,
            caller,
        });
        Record {
            core,
            number,
            id,
            args: passed,
            outcome,
        }
    }

    /// Takes the tables back, for a snapshot or a test's inspection.
    #[must_use]
    pub fn into_tables(self) -> Tables {
        self.state
            .into_inner()
            .unwrap_or_else(PoisonError::into_inner)
            .tables
    }

    /// One dispatch, start to finish: take the tables `B` declares, read the clock, run `body`
    /// with those tables and the guest time, record the event, release.
    fn call<B: TableSet>(
        &self,
        call: &mut Call<'_, '_>,
        body: impl for<'a> FnOnce(B::Borrowed<'a>, &mut Call<'_, '_>, GuestTicks) -> Outcome,
    ) -> Outcome {
        self.call_at::<B>(call, |tables, call, now| body(tables, call, now.guest))
    }

    /// [`Dispatcher::call`], handing `body` the clock reading in both units. Waits that ended
    /// since the last dispatch are recorded first, and the waits this one ended last
    /// (`dispatch/sync.rs`).
    fn call_at<B: TableSet>(
        &self,
        call: &mut Call<'_, '_>,
        body: impl for<'a> FnOnce(B::Borrowed<'a>, &mut Call<'_, '_>, Now) -> Outcome,
    ) -> Outcome {
        let mut locked = self.lock::<B>();
        // Read under the lock, so the environment's recording is in the kernel's order.
        let now = match locked.clock(call.caller.env) {
            Ok(now) => now,
            Err(error) => {
                tracing::error!(number = call.number, %error, "environment failed");
                return Outcome::EnvFailed(error);
            }
        };
        sync::drain_wakes(&mut locked.state, call.caller.recorder, now.guest);
        let outcome = body(locked.tables(), call, now);
        record_point(call, now.guest, &outcome);
        sync::drain_wakes(&mut locked.state, call.caller.recorder, now.guest);
        drop(locked);
        outcome
    }

    /// The one `match` from call to typed function. Each arm names the tables its function
    /// declares.
    fn run(&self, call: &mut Call<'_, '_>) -> Outcome {
        let Some(id) = call.id else {
            return self.call::<()>(call, |(), call, _| {
                tracing::warn!(number = call.number, "unknown system call");
                not_implemented(call.regs)
            });
        };
        let Some(convention) = id.convention() else {
            return self.call::<()>(call, |(), call, _| unimplemented(id, call.regs));
        };
        match id {
            SvcId::SetHeapSize => self.call::<(&mut AddressSpace,)>(call, |(space,), call, _| {
                set_heap_size(space, id, convention, call.regs)
            }),
            SvcId::SetMemoryPermission => self
                .call::<(&mut AddressSpace,)>(call, |(space,), call, _| {
                    set_memory_permission(space, id, convention, call.regs)
                }),
            SvcId::SetMemoryAttribute => self
                .call::<(&mut AddressSpace,)>(call, |(space,), call, _| {
                    set_memory_attribute(space, id, convention, call.regs)
                }),
            SvcId::QueryMemory => self.call::<(&AddressSpace,)>(call, |(space,), call, _| {
                query_memory(space, call.caller.memory, id, convention, call.regs)
            }),
            SvcId::GetInfo => self.call::<(&HandleTable, &AddressSpace, &Scheduler)>(
                call,
                |(handles, space, scheduler), call, _| {
                    get_info((handles, space, scheduler), id, convention, call.regs)
                },
            ),
            SvcId::ExitProcess => self.call::<()>(call, |(), _, _| {
                // PROVISIONAL (walking skeleton, #74): see the module docs.
                tracing::info!("the process called ExitProcess");
                Outcome::ProcessExited
            }),
            SvcId::GetThreadPriority => {
                self.call::<(&HandleTable, &Scheduler)>(call, |(handles, scheduler), call, _| {
                    let core = call.caller.core;
                    returning(id, convention, call.regs, |[thread_handle]| {
                        let priority = thread::get_thread_priority(
                            scheduler,
                            handles,
                            calling_core(core)?,
                            Handle::from_register(thread_handle),
                        )?;
                        Ok([u64::from(priority.cast_unsigned())])
                    })
                })
            }
            SvcId::GetCurrentProcessorNumber => self.call::<()>(call, |(), call, _| {
                let core = processor::get_current_processor_number(call.caller.core);
                complete(
                    id,
                    convention,
                    call.regs,
                    &[u64::from(core.cast_unsigned())],
                )
            }),
            SvcId::CloseHandle => {
                let mut notices = false;
                let outcome = self.call::<(&mut HandleTable, &mut Scheduler)>(
                    call,
                    |(handles, scheduler), call, _| {
                        let outcome = returning(id, convention, call.regs, |[closing]| {
                            let closing = Handle::from_register(closing);
                            handle::close_handle(handles, scheduler, closing)?;
                            Ok([])
                        });
                        notices = scheduler.ipc().has_any_notices();
                        outcome
                    },
                );
                // A session whose last handle closed: its server forgets it now, if it is free.
                if notices {
                    self.settle_notices();
                }
                outcome
            }
            SvcId::SignalEvent
            | SvcId::ClearEvent
            | SvcId::ResetSignal
            | SvcId::WaitSynchronization
            | SvcId::CancelSynchronization
            | SvcId::CreateEvent => self.synchronization(call, id, convention),
            SvcId::ConnectToNamedPort => self.connect_to_named_port(call, id, convention),
            SvcId::SendSyncRequest => self.send_sync_request(call, id, convention),
            SvcId::GetSystemTick => self.call::<()>(call, |(), call, time| {
                complete(id, convention, call.regs, &[tick::get_system_tick(time)])
            }),
            SvcId::OutputDebugString => self.call::<()>(call, |(), call, time| {
                let memory = call.caller.memory;
                let sink = &mut *call.caller.debug;
                returning(id, convention, call.regs, |[address, len]| {
                    debug::output_debug_string(memory, sink, time, address, len)?;
                    Ok([])
                })
            }),
            _ => self.call::<()>(call, |(), call, _| unimplemented(id, call.regs)),
        }
    }
}

// The bodies of the dispatch arms for the calls that work on the address space (kernel slices 6
// and 7), kept out of `Dispatcher::run` only for its length. Each decodes its call's arguments
// and writes its outputs, as every arm does.

fn set_heap_size(
    space: &mut AddressSpace,
    id: SvcId,
    convention: &Convention,
    regs: &mut GuestRegs,
) -> Outcome {
    returning(id, convention, regs, |[size]| {
        Ok([memory::set_heap_size(space, size)?])
    })
}

fn set_memory_permission(
    space: &mut AddressSpace,
    id: SvcId,
    convention: &Convention,
    regs: &mut GuestRegs,
) -> Outcome {
    returning(id, convention, regs, |[address, size, permission]| {
        memory::set_memory_permission(space, address, size, word(permission)?)?;
        Ok([])
    })
}

fn set_memory_attribute(
    space: &mut AddressSpace,
    id: SvcId,
    convention: &Convention,
    regs: &mut GuestRegs,
) -> Outcome {
    returning(id, convention, regs, |[address, size, mask, attribute]| {
        let (mask, attribute) = (word(mask)?, word(attribute)?);
        memory::set_memory_attribute(space, address, size, mask, attribute)?;
        Ok([])
    })
}

fn query_memory(
    space: &AddressSpace,
    guest: &dyn GuestMemory,
    id: SvcId,
    convention: &Convention,
    regs: &mut GuestRegs,
) -> Outcome {
    returning(id, convention, regs, |[out_memory_info, address]| {
        let page_info = memory::query_memory(space, guest, out_memory_info, address)?;
        Ok([u64::from(page_info)])
    })
}

fn get_info(
    (handles, space, scheduler): (&HandleTable, &AddressSpace, &Scheduler),
    id: SvcId,
    convention: &Convention,
    regs: &mut GuestRegs,
) -> Outcome {
    returning(id, convention, regs, |[info_type, process, sub_id]| {
        let process = Handle::from_register(process);
        let info_type = word(info_type)?;
        Ok([info::get_info(
            handles, space, scheduler, info_type, process, sub_id,
        )?])
    })
}

/// RECORDING POINT for the event stream: the dispatcher's one call into the recorder
/// (observability spec, "The event stream"). It runs after the call with the kernel lock still
/// held, so the recorder's `seq` follows the kernel's order. `time` is the dispatch's one clock
/// reading, as guest ticks. See the module docs for what the event's `result` holds.
fn record_point(call: &Call<'_, '_>, time: GuestTicks, outcome: &Outcome) {
    let Some(result) = recorded_result(call.id, outcome, call.regs) else {
        return;
    };
    call.caller.recorder.record(
        time,
        call.caller.core,
        call.caller.thread,
        EventKind::Svc {
            id: u32::from(call.number),
            args: recorded_args(call.id, &call.passed),
            // The one place a kernel result becomes the event stream's result word.
            result: starquake_trace::ResultCode(result.raw()),
        },
    );
}

/// The arguments a dispatch is recorded with: the call's inputs, decoded with its convention,
/// or every register as passed when the number has no convention (see the module docs).
fn recorded_args(id: Option<SvcId>, passed: &[u64; SVC_ARG_COUNT]) -> [u64; SVC_ARG_COUNT] {
    let Some(convention) = id.and_then(SvcId::convention) else {
        return *passed;
    };
    let mut args = [0; SVC_ARG_COUNT];
    for input in convention.inputs {
        let index = usize::from(input.reg);
        if let (Some(slot), Some(passed)) = (args.get_mut(index), passed.get(index)) {
            *slot = input.kind.truncate(*passed);
        }
    }
    args
}

/// The result word a finished dispatch is recorded with, or `None` if the call never returned to
/// the guest.
fn recorded_result(id: Option<SvcId>, outcome: &Outcome, regs: &GuestRegs) -> Option<ResultCode> {
    match outcome {
        Outcome::EnvFailed(_) | Outcome::Waiting => None,
        Outcome::NotImplemented => Some(KernelError::NotImplemented.code()),
        Outcome::ProcessExited => Some(ResultCode::SUCCESS),
        Outcome::Completed => {
            let result = id
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

/// The calling core as the scheduler names it. A core index a [`CoreId`] can't hold is the core
/// loop's bug, not a guest's.
fn calling_core(core: u8) -> CallResult<CoreId> {
    CoreId::try_from(core).map_err(|error| SvcError::internal(&error))
}

/// A decoded 4-byte argument as the `u32` it is. The convention truncated it to four bytes, so
/// a value that does not fit is a disagreement between a dispatch arm and its convention: a
/// dispatcher bug, not a guest's.
fn word(argument: u64) -> CallResult<u32> {
    u32::try_from(argument).map_err(|error| SvcError::internal(&error))
}

/// Runs an implemented call that returns a `Result` first: decodes its `I` inputs, runs `body`,
/// and writes the result and its `O` other outputs.
///
/// A call that fails with a result the guest sees ([`SvcError::Kernel`]) writes that result and
/// zeroes its other outputs. Eden's wrapper value-initialises its outputs
/// (`int32_t out_priority{};`), runs the call, and writes whatever the call left in them
/// (`SvcWrap_GetThreadPriority64` in `eden/src/core/hle/kernel/svc.cpp` @ cf654052). The two
/// agree only for a call whose body assigns no output before it fails, as for
/// `GetThreadPriority`: its one failure, `ResultInvalidHandle`, comes before `*out_priority` is
/// assigned (`eden/src/core/hle/kernel/svc/svc_thread.cpp` @ cf654052, `GetThreadPriority`). A
/// call that assigns an output and then fails goes through [`answering`] instead.
fn returning<const I: usize, const O: usize>(
    id: SvcId,
    convention: &Convention,
    regs: &mut GuestRegs,
    body: impl FnOnce([u64; I]) -> CallResult<[u64; O]>,
) -> Outcome {
    answering(id, convention, regs, |inputs| {
        body(inputs).map(|outputs| Answer::Now {
            result: ResultCode::SUCCESS,
            outputs,
        })
    })
}

/// How a call run by [`answering`] ends.
enum Answer<const O: usize> {
    /// It returns now: `result` in `x0` and `outputs` after it. An error result may come with
    /// outputs: the "error with outputs" form.
    Now {
        /// The result.
        result: ResultCode,
        /// The other outputs, in the convention's order.
        outputs: [u64; O],
    },
    /// The calling thread waits; nothing is written ([`Outcome::Waiting`]).
    Later,
}

/// Runs an implemented call that returns a `Result` first and may return other outputs
/// together with an error: decodes its `I` inputs, runs `body`, and writes what its [`Answer`]
/// says. A [`SvcError::Kernel`] from `body` writes that result and zeroes the other outputs, as
/// [`returning`] does.
///
/// The call that needs it is `WaitSynchronization`. `KSynchronizationObject::Wait`
/// (`eden/src/core/hle/kernel/k_synchronization_object.cpp` @ cf654052), when the thread did
/// begin to wait, clears its synced index to `-1` first, and afterwards sets
/// `*out_index = thread->GetSyncedIndex()` and returns the wait's result. A wait ended by its
/// timer or by a cancel, not by an object, so returns `ResultTimedOut` or `ResultCancelled`
/// with `*out_index` at `-1`, and `SvcWrap_WaitSynchronization64` (`svc.cpp`) writes that index
/// to `x1`. (Its earlier exits, a zero timeout and a cancel already pending, return the same
/// two results without assigning `*out_index`, which the wrapper then writes as zero.)
fn answering<const I: usize, const O: usize>(
    id: SvcId,
    convention: &Convention,
    regs: &mut GuestRegs,
    body: impl FnOnce([u64; I]) -> CallResult<Answer<O>>,
) -> Outcome {
    let inputs = match convention.read::<I>(regs) {
        Ok(inputs) => inputs,
        Err(error) => return mismatch(id, error, regs),
    };
    let (code, outputs) = match body(inputs) {
        Ok(Answer::Now { result, outputs }) => (result, outputs),
        Ok(Answer::Later) => return Outcome::Waiting,
        Err(SvcError::Kernel(error)) => (error.code(), [0; O]),
        Err(SvcError::Internal(what)) => return fault(id, &what, regs),
    };
    // The result, then the other outputs. A convention has at most SVC_ARG_COUNT registers, so
    // a call with more outputs than fit disagrees with any convention.
    let mut values = [0; SVC_ARG_COUNT];
    values[RESULT_REGISTER] = result_word(code);
    for (slot, output) in values.iter_mut().skip(1).zip(outputs) {
        *slot = output;
    }
    let Some(values) = values.get(..=O) else {
        let error = ConventionError::Arity {
            expected: convention.outputs.len(),
            actual: O.saturating_add(1),
        };
        return mismatch(id, error, regs);
    };
    complete(id, convention, regs, values)
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

/// An implemented call failed inside the kernel ([`SvcError::Internal`]). The one place that
/// decides what that means (see the module docs): it is logged, and the guest sees "not
/// implemented".
fn fault(id: SvcId, what: &str, regs: &mut GuestRegs) -> Outcome {
    tracing::error!(
        name = id.name(),
        number = id.number(),
        what,
        "system call failed inside the kernel"
    );
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
