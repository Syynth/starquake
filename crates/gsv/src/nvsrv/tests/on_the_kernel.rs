//! nvmap's device-share locks and the process's end on the kernel's own dispatcher, not the
//! services framework's fake: the nvsrv server on a port, nvmap's ioctls sent through the
//! kernel as a title sends them, the process ended by `ExitProcess`, and the completion queue
//! drained through the kernel as the timer thread drains it (`nvmap` module docs, "The
//! device-share lock" and "When the process's last session closes"; `dispatch/device.rs` in the
//! kernel, "When the process ends").

use std::any::Any;

use starquake_cpu::GuestRegs;
use starquake_ipc::BufferDescriptor;
use starquake_kernel::dispatch::{Caller, DeviceShareError, Dispatcher, Outcome};
use starquake_kernel::handle::Handle;
use starquake_kernel::ipc::PortName;
use starquake_kernel::memory::testing::RecordingMapper;
use starquake_kernel::memory::{
    AddressSpace, DeviceShareToken, MemoryArrangement, ProcessMemory, RegionOffsets, SvcPermission,
};
use starquake_kernel::process::{self, Console, ProcessDescription};
use starquake_kernel::svc::{DebugSink, SvcId};
use starquake_memory::{FlatMemory, GuestMemory, PAGE_SIZE};
use starquake_trace::{EventKind, FixedClock, GuestTicks, LiveEnv, Recorder, ThreadId};

use super::*;
use crate::shared::TimerHost;

/// The process's memory, one page each: the message buffer, the port name, the device's path,
/// an ioctl's argument, its out buffer, and the memory nvmap allocates.
const BASE: u64 = 0x8000_0000;
const TLS: u64 = BASE;
const NAME: u64 = BASE + PAGE_SIZE;
const PATH: u64 = BASE + 2 * PAGE_SIZE;
const IN: u64 = BASE + 3 * PAGE_SIZE;
const OUT: u64 = BASE + 4 * PAGE_SIZE;
const MEMORY: u64 = BASE + 5 * PAGE_SIZE;
const END: u64 = BASE + 6 * PAGE_SIZE;

/// The size of the handle's memory: one page.
const SIZE: u32 = 0x1000;

struct NoDebug;

impl DebugSink for NoDebug {
    fn output(&mut self, _: GuestTicks, _: &[u8]) {}
}

/// A Switch 2 process whose memory is all read-write, with the nvsrv server on the port
/// `nvst:` and a session to it.
struct KernelRig {
    dispatcher: Dispatcher,
    memory: FlatMemory,
    env: LiveEnv,
    recorder: Recorder,
    shared: Arc<Shared>,
    session: Handle,
}

impl KernelRig {
    fn new() -> Self {
        let (mapper, _log) = RecordingMapper::new();
        let layout = ProcessMemory {
            code_address: BASE,
            code_size: END - BASE,
            arrangement: MemoryArrangement::FourGb,
            alias_region_extra: false,
            offsets: RegionOffsets::default(),
        };
        let mut space = AddressSpace::new(&layout, Box::new(mapper)).unwrap();
        space
            .set_process_memory_permission(BASE, END - BASE, SvcPermission::ReadWrite)
            .unwrap();
        let description = ProcessDescription::homebrew(Console::Switch2);
        let dispatcher = Dispatcher::new(process::create(&description, space).unwrap());
        let (shared, front) = wiring();
        let server = nvsrv_server(Arc::clone(&shared), as_front(&front), UnknownPolicy::Strict)
            .with_service_id(NVSRV);
        dispatcher
            .add_server(
                NVSRV_NAME,
                NVSRV,
                Some(PortName::new("nvst:").unwrap()),
                Box::new(server),
            )
            .unwrap();
        let mut rig = Self {
            dispatcher,
            memory: FlatMemory::new(BASE, usize::try_from(END - BASE).unwrap()),
            env: LiveEnv::new(FixedClock::new(0, 1), 0),
            recorder: Recorder::new(),
            shared,
            session: Handle::from_raw(0),
        };
        rig.memory.write(NAME, b"nvst:\0").unwrap();
        let regs = rig.call(SvcId::ConnectToNamedPort, [0, NAME]);
        assert_eq!(regs.x[0], 0);
        rig.session = Handle::from_register(regs.x[1]);
        rig
    }

    /// Makes system call `id` with `x0` and `x1`; the registers after it and how it ended.
    fn dispatch(&self, id: SvcId, x: [u64; 2]) -> (GuestRegs, Outcome) {
        let mut regs = GuestRegs {
            tpidrro: TLS,
            ..GuestRegs::default()
        };
        regs.x[..2].copy_from_slice(&x);
        let mut debug = NoDebug;
        let mut caller = Caller {
            core: 0,
            thread: ThreadId(1),
            env: &self.env,
            memory: &self.memory,
            debug: &mut debug,
            recorder: &self.recorder,
        };
        let record = self
            .dispatcher
            .dispatch(id.number(), &mut regs, &mut caller);
        (regs, record.outcome)
    }

    /// Makes system call `id`, which must complete; the registers after it.
    fn call(&self, id: SvcId, x: [u64; 2]) -> GuestRegs {
        let (regs, outcome) = self.dispatch(id, x);
        assert_eq!(outcome, Outcome::Completed, "{}", id.name());
        regs
    }

    /// Sends `request` on the session through the kernel; the reply's result and `words` data
    /// words.
    fn send(&self, request: &RequestBuilder, words: usize) -> Answer {
        let mut buffer = vec![0; BUFFER_SIZE];
        request.build(&mut buffer).unwrap();
        self.memory.write(TLS, &buffer).unwrap();
        let regs = self.call(SvcId::SendSyncRequest, [u64::from(self.session.raw()), 0]);
        assert_eq!(regs.x[0], 0);
        self.memory.read(TLS, &mut buffer).unwrap();
        answer(&buffer, words)
    }

    /// Opens `/dev/nvmap`; its fd.
    fn open_nvmap(&self) -> i32 {
        let path = Device::Nvmap.path().as_bytes();
        self.memory.write(PATH, path).unwrap();
        let request = cmif(OPEN, &[]).send_buffer(BufferDescriptor {
            address: PATH,
            size: u64::try_from(path.len()).unwrap(),
            flags: 0,
        });
        let answer = self.send(&request, 2);
        assert_eq!(answer.words[1], NvResult::Success.raw());
        i32::try_from(answer.words[0]).unwrap()
    }

    /// An nvmap ioctl, which must succeed; the struct it gives back.
    fn nvmap(&self, fd: i32, raw: u32, input: &[u8]) -> Vec<u8> {
        self.memory.write(IN, input).unwrap();
        let len = len16(input.len());
        let request = Auto::alias(IN, len).send(cmif(IOCTL, &fd_and(fd, raw)), 0);
        let request = Auto::alias(OUT, len).receive(request);
        assert_eq!(self.send(&request, 1), ok(&[NvResult::Success.raw()]));
        let mut output = vec![0; input.len()];
        self.memory.read(OUT, &mut output).unwrap();
        output
    }

    /// Opens nvmap and creates a one-page handle; the fd and the handle.
    fn created(&self) -> (i32, u32) {
        let fd = self.open_nvmap();
        let output = self.nvmap(fd, NVMAP_CREATE, &le(&[SIZE, 0]));
        let handle = u32::from_le_bytes(output[4..8].try_into().unwrap());
        (fd, handle)
    }

    /// [`created`](Self::created), and the handle allocated over [`MEMORY`], whose lock the
    /// kernel then holds; the fd and the handle.
    fn allocated(&self) -> (i32, u32) {
        let (fd, handle) = self.created();
        let address = MEMORY.to_le_bytes();
        let low = u32::from_le_bytes(address[..4].try_into().unwrap());
        let high = u32::from_le_bytes(address[4..].try_into().unwrap());
        self.nvmap(fd, NVMAP_ALLOC, &le(&[handle, 0, 0, SIZE, 0, 0, low, high]));
        assert_eq!(self.locks(), 1);
        assert_eq!(self.nvmap_lock(handle), Held::Lock(MEMORY, u64::from(SIZE)));
        (fd, handle)
    }

    /// The live device-share locks the kernel holds on [`MEMORY`]'s page.
    fn locks(&self) -> u32 {
        self.dispatcher
            .with::<(&AddressSpace,), _>(|(space,)| space.device_locks(MEMORY, PAGE_SIZE))
    }

    /// Whether the kernel's address space holds any device-share lock at all.
    fn any_lock(&self) -> bool {
        self.dispatcher
            .with::<(&AddressSpace,), _>(|(space,)| space.device_share_locks().next().is_some())
    }

    /// nvsrv's nvmap table, read through the kernel's handlers.
    fn read_nvmap<R>(&self, read: impl Fn(&NvMap) -> R) -> R {
        let mut seen = None;
        self.dispatcher.inspect_handlers(|service, handler| {
            if service == NVSRV {
                let handler: &dyn Any = handler;
                let server = handler.downcast_ref::<Server>().unwrap();
                seen = Some(read(state_of(server).unwrap().nvmap()));
            }
        });
        seen.unwrap()
    }

    /// What nvmap holds for `handle`.
    fn nvmap_lock(&self, handle: u32) -> Held {
        self.read_nvmap(|nvmap| match (nvmap.handle(handle), nvmap.lock(handle)) {
            (None, _) => Held::Gone,
            (Some(_), None) => Held::NoLock,
            (Some(_), Some(token)) => Held::Lock(token.address(), token.size()),
        })
    }

    /// Ends the process with `ExitProcess`.
    fn exit(&self) {
        assert_eq!(
            self.dispatch(SvcId::ExitProcess, [0, 0]).1,
            Outcome::ProcessExited
        );
    }

    /// Drains the completion queue through the kernel, as the timer thread does; what the
    /// kernel answered each release.
    fn drain(&self) -> Vec<Result<(), DeviceShareError>> {
        let mut host = OnTheKernel {
            dispatcher: &self.dispatcher,
            answers: Vec::new(),
        };
        self.shared.drain(&mut host);
        host.answers
    }
}

/// What nvmap holds for a handle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Held {
    /// The handle is not in the table.
    Gone,
    /// It is, holding no lock.
    NoLock,
    /// It is, holding a lock on this address and size.
    Lock(u64, u64),
}

/// The timer thread's host over the kernel's dispatcher: releases go to
/// `Dispatcher::unlock_device_share`, and each answer is kept.
struct OnTheKernel<'d> {
    dispatcher: &'d Dispatcher,
    answers: Vec<Result<(), DeviceShareError>>,
}

impl TimerHost for OnTheKernel<'_> {
    fn record(&mut self, _: EventKind) {}

    fn unlock_device_share(&mut self, token: DeviceShareToken) -> Result<(), DeviceShareError> {
        let answer = self.dispatcher.unlock_device_share(token);
        self.answers.push(answer.clone());
        answer
    }

    fn signal_event(&mut self, _: ObjectId) -> Result<(), KernelError> {
        Ok(())
    }
}

#[test]
fn alloc_takes_the_kernel_s_lock_and_the_last_free_releases_it() {
    let rig = KernelRig::new();
    let (fd, handle) = rig.allocated();
    rig.nvmap(fd, NVMAP_FREE, &le(&[handle, 0, 0, 0, 0, 0]));
    assert_eq!(rig.locks(), 0);
    assert_eq!(rig.nvmap_lock(handle), Held::Gone);
    // Nothing was queued: the free released the lock during its request.
    assert_eq!(rig.drain(), []);
}

#[test]
fn a_process_ending_with_a_locked_handle_releases_the_lock_once_and_frees_the_handle() {
    let rig = KernelRig::new();
    let (_, handle) = rig.allocated();
    rig.exit();
    // The kernel released the lock as the process ended, and the session's close freed the
    // handle, queueing its lock's release.
    assert_eq!(rig.locks(), 0);
    assert!(!rig.any_lock());
    assert!(rig.read_nvmap(NvMap::is_empty));
    assert_eq!(rig.nvmap_lock(handle), Held::Gone);
    // The drain's release is refused for the ended process and releases nothing again.
    assert_eq!(rig.drain(), [Err(DeviceShareError::ProcessEnded)]);
    assert_eq!(rig.locks(), 0);
}

#[test]
fn a_process_ending_with_an_unallocated_handle_frees_it_and_queues_nothing() {
    let rig = KernelRig::new();
    let (_, handle) = rig.created();
    assert_eq!(rig.nvmap_lock(handle), Held::NoLock);
    rig.exit();
    assert!(rig.read_nvmap(NvMap::is_empty));
    assert!(!rig.any_lock());
    assert_eq!(rig.drain(), []);
}

#[test]
fn a_release_queued_before_the_process_ends_is_refused_at_the_drain() {
    // The session closes with the process alive: the handle goes and its release is queued,
    // and the kernel still holds the lock. The process then ends before the drain.
    let rig = KernelRig::new();
    rig.allocated();
    rig.call(SvcId::CloseHandle, [u64::from(rig.session.raw()), 0]);
    assert!(rig.read_nvmap(NvMap::is_empty));
    assert_eq!(rig.locks(), 1);
    rig.exit();
    assert_eq!(rig.locks(), 0);
    assert_eq!(rig.drain(), [Err(DeviceShareError::ProcessEnded)]);
    assert_eq!(rig.locks(), 0);
    assert!(!rig.any_lock());
}

#[test]
fn a_release_drained_while_the_process_lives_unlocks_once() {
    // The same close, drained before the process ends: the release is taken, and the process's
    // end then finds no lock to release.
    let rig = KernelRig::new();
    rig.allocated();
    rig.call(SvcId::CloseHandle, [u64::from(rig.session.raw()), 0]);
    assert_eq!(rig.locks(), 1);
    assert_eq!(rig.drain(), [Ok(())]);
    assert_eq!(rig.locks(), 0);
    rig.exit();
    assert!(!rig.any_lock());
    assert_eq!(rig.drain(), []);
}
