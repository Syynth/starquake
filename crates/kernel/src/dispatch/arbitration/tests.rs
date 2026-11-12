//! The arbitration arms of the dispatcher, driven with guest registers, flat guest memory, the
//! fixture's address space and threads running on their own cores: a call that waits records
//! nothing until its wait ends, and then its own event; a call that fails at once writes and
//! records its result; `SignalProcessWideKey` writes no register; a timer ends a wait.

use std::sync::Arc;

use starquake_cpu::GuestRegs;
use starquake_memory::{FlatMemory, GuestMemory, PAGE_SIZE};
use starquake_trace::{
    Event, EventKind, FixedClock, GuestTicks, LiveEnv, Recorder, SVC_ARG_COUNT, ThreadId, VecSink,
};

use super::*;
use crate::dispatch::{Caller, Record, Tables, Woken};
use crate::memory::fixture::{self, DATA, DATA_PAGES};
use crate::process::{Console, ProcessDescription};
use crate::result::KernelError;
use crate::svc::DebugSink;
use crate::sync::testing::World as Threads;
use crate::sync::{ArbitrationType, HANDLE_WAIT_MASK, SignalType};
use crate::tables::Processes;

const AT: u64 = DATA + 0x40;
const CV: u64 = DATA + 0x80;
const CLOCK_START_NS: u64 = 1_000_000;
const CLOCK_STEP_NS: u64 = 1_000;
/// What a register holds before a call, to see whether the call writes it.
const STALE: u64 = 0x5555_5555_5555_5555;

struct NoDebug;

impl DebugSink for NoDebug {
    fn output(&mut self, _: GuestTicks, _: &[u8]) {}
}

/// A dispatcher over the sync tests' threads (thread `i` on core `i`, as the event stream's
/// thread `0x100 + i`), the fixture's address space and flat memory over the data.
struct World {
    dispatcher: Dispatcher,
    memory: FlatMemory,
    env: LiveEnv,
    recorder: Recorder,
    events: Arc<VecSink>,
    threads: Vec<crate::id::ThreadId>,
    handles: Vec<Handle>,
}

impl World {
    fn new(priorities: &[u8]) -> Self {
        let threads = Threads::new(priorities);
        let tables = Tables {
            handles: threads.handles,
            address_space: fixture::loaded().0,
            scheduler: threads.scheduler,
            processes: Processes::new(&ProcessDescription::homebrew(Console::Switch2)).unwrap(),
        };
        let recorder = Recorder::new();
        let events = Arc::new(VecSink::new());
        recorder.add_sink(Arc::clone(&events));
        let size = usize::try_from(DATA_PAGES * PAGE_SIZE).unwrap();
        Self {
            dispatcher: Dispatcher::new(tables),
            memory: FlatMemory::new(DATA, size),
            env: LiveEnv::new(FixedClock::new(CLOCK_START_NS, CLOCK_STEP_NS), 0),
            recorder,
            events,
            threads: threads.threads,
            handles: threads.thread_handles,
        }
    }

    fn guest_thread(index: usize) -> ThreadId {
        crate::sync::testing::origin(index).thread
    }

    /// `id` from thread `index` with `x0`-`x3` as given and every other register stale.
    fn call(&self, index: usize, id: SvcId, x: [u64; 4]) -> (Record, GuestRegs) {
        let mut regs = GuestRegs {
            x: [STALE; 31],
            ..GuestRegs::default()
        };
        regs.x[..4].copy_from_slice(&x);
        let debug: &mut NoDebug = Box::leak(Box::new(NoDebug));
        let mut caller = Caller {
            core: u8::try_from(index).unwrap(),
            thread: Self::guest_thread(index),
            env: &self.env,
            memory: &self.memory,
            debug,
            recorder: &self.recorder,
        };
        let record = self
            .dispatcher
            .dispatch(id.number(), &mut regs, &mut caller);
        (record, regs)
    }

    fn set(&self, address: u64, value: u32) {
        self.memory.write(address, &value.to_le_bytes()).unwrap();
    }

    fn get(&self, address: u64) -> u32 {
        let mut bytes = [0; 4];
        self.memory.read(address, &mut bytes).unwrap();
        u32::from_le_bytes(bytes)
    }

    fn context(&self, index: usize) -> GuestRegs {
        let thread = self.threads[index];
        self.dispatcher.with::<(&Scheduler,), _>(|(scheduler,)| {
            scheduler.thread(thread).unwrap().context().clone()
        })
    }

    /// `(thread, call, args, result)` of each event; every event is a system call's.
    fn events(&self) -> Vec<(ThreadId, u32, [u64; SVC_ARG_COUNT], u32)> {
        let events = self.events.events();
        let calls: Vec<_> = events
            .iter()
            .filter_map(|event: &Event| match event.kind {
                EventKind::Svc { id, args, result } => Some((event.thread, id, args, result.0)),
                _ => None,
            })
            .collect();
        assert_eq!(calls.len(), events.len(), "only system calls: {events:?}");
        calls
    }
}

/// An event's args: `x` in `x0`-`x3`, zero after.
fn args(x: [u64; 4]) -> [u64; SVC_ARG_COUNT] {
    let mut args = [0; SVC_ARG_COUNT];
    args[..4].copy_from_slice(&x);
    args
}

fn number(id: SvcId) -> u32 {
    u32::from(id.number())
}

fn code(error: KernelError) -> u64 {
    u64::from(error.code().raw())
}

#[test]
fn an_address_wait_records_nothing_until_a_signal_ends_it_and_then_its_own_event() {
    let world = World::new(&[30, 30]);
    world.set(AT, 0);
    let wait = [
        AT,
        u64::from(ArbitrationType::WaitIfEqual.raw()),
        0,
        u64::MAX,
    ];
    let (record, regs) = world.call(0, SvcId::WaitForAddress, wait);
    assert_eq!(record.outcome, Outcome::Waiting);
    assert_eq!(regs.x[0], AT, "a call that waits writes nothing");
    assert_eq!(world.events(), []);

    let signal = [AT, u64::from(SignalType::Signal.raw()), 0, 1];
    let (record, regs) = world.call(1, SvcId::SignalToAddress, signal);
    assert_eq!(record.outcome, Outcome::Completed);
    assert_eq!(regs.x[0], 0);
    assert_eq!(
        world.events(),
        [
            (
                World::guest_thread(1),
                number(SvcId::SignalToAddress),
                args(signal),
                0
            ),
            (
                World::guest_thread(0),
                number(SvcId::WaitForAddress),
                args(wait),
                0
            ),
        ]
    );
    assert_eq!(
        world.dispatcher.take_woken(),
        [Woken {
            thread: World::guest_thread(0),
            core: 0,
            result: ResultCode::SUCCESS,
        }]
    );
    let context = world.context(0);
    assert_eq!(context.x[0], 0, "the wake wrote the result");
    assert_eq!(context.x[1..4], wait[1..], "and nothing else");
}

#[test]
fn a_call_that_fails_at_once_writes_and_records_its_result() {
    let world = World::new(&[30]);
    let cases = [
        (
            SvcId::WaitForAddress,
            [AT, 3, 0, 0],
            KernelError::InvalidEnumValue,
        ),
        (
            SvcId::SignalToAddress,
            [AT + 1, 0, 0, 0],
            KernelError::InvalidAddress,
        ),
        (
            SvcId::ArbitrateUnlock,
            [u64::MAX - 3, 0, 0, 0],
            KernelError::InvalidCurrentMemory,
        ),
        (
            SvcId::WaitProcessWideKeyAtomic,
            [AT, CV, 0, 0],
            KernelError::TimedOut,
        ),
    ];
    for (id, x, error) in cases {
        let (record, regs) = world.call(0, id, x);
        assert_eq!(record.outcome, Outcome::Completed, "{id:?}");
        assert_eq!(regs.x[0], code(error), "{id:?}");
        let last = world.events().pop().unwrap();
        assert_eq!(last.1, number(id));
        assert_eq!(u64::from(last.3), code(error), "{id:?}");
    }
}

#[test]
fn signal_process_wide_key_writes_no_register_and_records_success() {
    let world = World::new(&[30]);
    let (record, regs) = world.call(0, SvcId::SignalProcessWideKey, [CV, 1, 0, 0]);
    assert_eq!(record.outcome, Outcome::Completed);
    assert_eq!(regs.x[0], CV);
    let (_, id, _, result) = world.events().pop().unwrap();
    assert_eq!((id, result), (number(SvcId::SignalProcessWideKey), 0));
}

#[test]
fn a_lock_is_handed_on_through_the_dispatcher() {
    let world = World::new(&[30, 20]);
    let owner = world.handles[0].raw();
    let tag = world.handles[1].raw();
    // A lock word that does not name the owner: done at once.
    world.set(AT, 0);
    let lock = [u64::from(owner), AT, u64::from(tag), 0];
    let (record, regs) = world.call(1, SvcId::ArbitrateLock, lock);
    assert_eq!((record.outcome, regs.x[0]), (Outcome::Completed, 0));
    // A held lock: 1 waits until 0 lets go.
    world.set(AT, owner | HANDLE_WAIT_MASK);
    let (record, _) = world.call(1, SvcId::ArbitrateLock, lock);
    assert_eq!(record.outcome, Outcome::Waiting);
    let (record, regs) = world.call(0, SvcId::ArbitrateUnlock, [AT, 0, 0, 0]);
    assert_eq!((record.outcome, regs.x[0]), (Outcome::Completed, 0));
    assert_eq!(world.get(AT), tag);
    let events = world.events();
    assert_eq!(
        events[events.len() - 2..],
        [
            (
                World::guest_thread(0),
                number(SvcId::ArbitrateUnlock),
                args([AT, 0, 0, 0]),
                0
            ),
            (
                World::guest_thread(1),
                number(SvcId::ArbitrateLock),
                args(lock),
                0
            ),
        ]
    );
}

#[test]
fn a_timer_ends_a_condition_variable_wait_and_records_timed_out() {
    let world = World::new(&[30]);
    world.set(AT, world.handles[0].raw());
    let timeout = CLOCK_STEP_NS / 2;
    let wait = [AT, CV, u64::from(world.handles[0].raw()), timeout];
    let (record, _) = world.call(0, SvcId::WaitProcessWideKeyAtomic, wait);
    assert_eq!(record.outcome, Outcome::Waiting);
    assert_eq!(world.get(AT), 0, "the lock was let go");
    // The timer thread's reading is a clock step later: past the deadline.
    let next = world.dispatcher.fire_timers(&world.env, &world.recorder);
    assert_eq!(next, Ok(None));
    assert_eq!(
        world.events().pop().unwrap(),
        (
            World::guest_thread(0),
            number(SvcId::WaitProcessWideKeyAtomic),
            args(wait),
            KernelError::TimedOut.code().raw()
        )
    );
    assert_eq!(world.context(0).x[0], code(KernelError::TimedOut));
}
