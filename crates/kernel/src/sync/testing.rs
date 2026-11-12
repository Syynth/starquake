//! A world for the arbitration tests: one thread per core, each running on its core with a
//! handle to its thread object, and user memory in a map ([`MapWords`]).

use starquake_cpu::GuestRegs;
use starquake_trace::{GuestTicks, HostNanos, SVC_ARG_COUNT};

use super::words::testing::MapWords;
use super::{Arbitrated, ArbitrationType, SignalType, Timeout, Wake, WaitCall, WaitOrigin};
use crate::handle::{Handle, HandleTable, MAX_TABLE_SIZE};
use crate::id::{ObjectId, ThreadId};
use crate::object::Object;
use crate::result::{CallResult, ResultCode};
use crate::sched::{CoreId, CoreSignals, Priority, Scheduler};
use crate::svc::SvcId;

/// Signals nobody: the tests read the scheduler's choices directly.
pub(crate) struct NoSignals;

impl CoreSignals for NoSignals {
    fn interrupt(&mut self, _: CoreId) {}

    fn wake(&mut self, _: CoreId) {}
}

pub(crate) fn core(index: usize) -> CoreId {
    CoreId::new(u8::try_from(index).unwrap()).unwrap()
}

/// The origin thread `index`'s calls are recorded with: distinct per thread.
pub(crate) fn origin(index: usize) -> WaitOrigin {
    let n = u64::try_from(index).unwrap();
    WaitOrigin {
        thread: starquake_trace::ThreadId(0x100 + n),
        core: u8::try_from(index).unwrap(),
        args: [n; SVC_ARG_COUNT],
    }
}

/// The registers thread `index` makes its calls with: distinct per thread.
pub(crate) fn regs(index: usize) -> GuestRegs {
    let n = u64::try_from(index).unwrap();
    GuestRegs {
        x: [0x1000 * (n + 1); 31],
        ..GuestRegs::default()
    }
}

/// A result as the register holds it.
pub(crate) fn word(result: ResultCode) -> u64 {
    u64::from(result.raw())
}

/// The scheduler, a handle table, user memory, and one running thread per core.
pub(crate) struct World {
    pub(crate) scheduler: Scheduler,
    pub(crate) handles: HandleTable,
    pub(crate) words: MapWords,
    pub(crate) threads: Vec<ThreadId>,
    pub(crate) objects: Vec<ObjectId>,
    pub(crate) thread_handles: Vec<Handle>,
}

impl World {
    /// One thread per entry of `priorities`, thread `i` running on core `i`.
    pub(crate) fn new(priorities: &[u8]) -> Self {
        let mut scheduler = Scheduler::default();
        let mut handles = HandleTable::new(i32::from(MAX_TABLE_SIZE)).unwrap();
        let mut threads = Vec::new();
        let mut objects = Vec::new();
        let mut thread_handles = Vec::new();
        for (index, priority) in priorities.iter().enumerate() {
            let thread = scheduler
                .create_thread(Priority::new(*priority).unwrap(), core(index))
                .unwrap();
            scheduler.start(thread).unwrap();
            let object = scheduler
                .objects_mut()
                .create(Object::Thread(thread))
                .unwrap();
            let handle = handles.add(scheduler.objects_mut(), object).unwrap();
            threads.push(thread);
            objects.push(object);
            thread_handles.push(handle);
        }
        let mut world = Self {
            scheduler,
            handles,
            words: MapWords::default(),
            threads,
            objects,
            thread_handles,
        };
        world.update();
        for index in 0..priorities.len() {
            let on = core(index);
            assert_eq!(
                world.scheduler.schedule(on, &mut NoSignals),
                Ok(Some(world.threads[index]))
            );
        }
        world
    }

    /// Ends the batch of changes, as the dispatcher does before it releases the lock.
    pub(crate) fn update(&mut self) {
        self.scheduler
            .update(GuestTicks(0), None, &mut NoSignals)
            .unwrap();
    }

    /// Thread `index`'s effective priority.
    pub(crate) fn priority(&self, index: usize) -> u8 {
        self.scheduler
            .thread(self.threads[index])
            .unwrap()
            .priority()
            .get()
    }

    /// The handle word a lock owned by thread `index` holds.
    pub(crate) fn tag(&self, index: usize) -> u32 {
        self.thread_handles[index].raw()
    }

    /// Thread `index`'s saved `x0`.
    pub(crate) fn x0(&self, index: usize) -> u64 {
        self.scheduler.thread(self.threads[index]).unwrap().context().x[0]
    }

    /// Whether thread `index` is in a wait.
    pub(crate) fn waiting(&self, index: usize) -> bool {
        self.scheduler.sync().wait(self.threads[index]).is_some()
    }

    /// The wake of thread `index` for `call` with `result`.
    pub(crate) fn wake(&self, index: usize, call: SvcId, result: ResultCode) -> Wake {
        Wake {
            thread: self.threads[index],
            origin: origin(index),
            call,
            result,
            index: super::wait::NO_INDEX,
        }
    }

    /// `WaitForAddress` from thread `index`.
    pub(crate) fn wait_address(
        &mut self,
        index: usize,
        address: u64,
        kind: ArbitrationType,
        value: i32,
        timeout: Timeout,
    ) -> CallResult<Arbitrated> {
        let regs = regs(index);
        let call = WaitCall {
            core: core(index),
            regs: &regs,
            origin: origin(index),
        };
        self.scheduler
            .wait_for_address(&self.words, call, address, kind, value, timeout)
    }

    /// `SignalToAddress` from thread `index`.
    pub(crate) fn signal_address(
        &mut self,
        index: usize,
        address: u64,
        kind: SignalType,
        value: i32,
        count: i32,
    ) -> CallResult<()> {
        self.scheduler
            .signal_to_address(&self.words, core(index), address, kind, value, count)
    }

    /// `ArbitrateLock` from thread `index` on the lock at `address`, naming `owner`'s handle,
    /// with the lock word already saying that `owner` holds it and threads wait.
    pub(crate) fn lock(
        &mut self,
        index: usize,
        owner: usize,
        address: u64,
    ) -> CallResult<Arbitrated> {
        let owner_handle = self.thread_handles[owner];
        self.words
            .set(address, owner_handle.raw() | super::HANDLE_WAIT_MASK);
        self.lock_with(index, owner_handle, address)
    }

    /// `ArbitrateLock` from thread `index` naming `owner_handle`, with the lock word as it is.
    pub(crate) fn lock_with(
        &mut self,
        index: usize,
        owner_handle: Handle,
        address: u64,
    ) -> CallResult<Arbitrated> {
        let regs = regs(index);
        let call = WaitCall {
            core: core(index),
            regs: &regs,
            origin: origin(index),
        };
        let tag = self.tag(index);
        self.scheduler
            .arbitrate_lock(&self.handles, &self.words, call, owner_handle, address, tag)
    }

    /// `ArbitrateUnlock` from thread `index`.
    pub(crate) fn unlock(&mut self, index: usize, address: u64) -> CallResult<()> {
        self.scheduler
            .arbitrate_unlock(&self.words, core(index), address)
    }

    /// `WaitProcessWideKeyAtomic` from thread `index`, with its own handle as its tag.
    pub(crate) fn cv_wait(
        &mut self,
        index: usize,
        address: u64,
        cv_key: u64,
        timeout: Timeout,
    ) -> CallResult<Arbitrated> {
        let regs = regs(index);
        let call = WaitCall {
            core: core(index),
            regs: &regs,
            origin: origin(index),
        };
        let tag = self.tag(index);
        self.scheduler
            .wait_condition_variable(&self.words, call, address, cv_key, tag, timeout)
    }

    /// `SignalProcessWideKey` from thread `index`.
    pub(crate) fn cv_signal(&mut self, index: usize, cv_key: u64, count: i32) -> CallResult<()> {
        self.scheduler
            .signal_condition_variable(&self.handles, &self.words, core(index), cv_key, count)
    }

    /// Expires every timer due at `now`, and returns how many waits that ended.
    pub(crate) fn fire(&mut self, now: u64) -> usize {
        self.scheduler.fire_timers(HostNanos(now)).unwrap()
    }
}
