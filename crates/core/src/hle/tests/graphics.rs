//! The graphics server as a run installs it ([`install_with_graphics`]), behind the real
//! dispatcher: the root hands out sessions on gsv's other servers, the timer thread's turn
//! ([`Graphics::turn`]) completes what waits on the GPU, and a recorded session replays to the
//! same state ([`Graphics::replay`]; the spec's offline test for port slice 16).
//!
//! Requests are laid out by hand from the layouts the references check, as gsv's own tests lay
//! them out: TIPC header words as PR compares them
//! (`eden-pr4588/src/core/hle/service/am/service/graphics_supervisor.cpp` @ e5fb3b97), nvsrv's
//! CMIF requests as libnx sends them (`libnx/nx/source/services/nv.c` @ b156b477), and ioctl
//! arguments from Eden's structs (`eden/src/core/hle/service/nvdrv/devices/nvhost_as_gpu.h` and
//! `nvhost_gpu.h` @ cf654052).

use std::sync::atomic::{AtomicUsize, Ordering};

use starquake_gpu::interface::SyncpointId;
use starquake_gpu::syncpoint::UNALLOCATED;
use starquake_gsv::gpu::command as gpu_command;
use starquake_gsv::note::Note;
use starquake_gsv::nvsrv::command as nv_command;
use starquake_gsv::nvsrv::gpu::FENCE_INCREMENT;
use starquake_gsv::nvsrv::{Device, NvResult};
use starquake_gsv::protocol::command as root_command;
use starquake_gsv::shared::TimerHost;
use starquake_gsv::shared::page::{self, READ_ONLY_SIZE, WRITABLE_SIZE};
use starquake_gsv::syncpt::command as syncpt_command;
use starquake_ipc::{Body, TIPC_COMMAND_OFFSET};
use starquake_kernel::dispatch::DeviceShareError;
use starquake_kernel::handle::HandleTable;
use starquake_kernel::id::{ObjectId, ProcessId, SlabId};
use starquake_kernel::memory::testing::device_share_token;
use starquake_trace::{EnvCall, EnvError, HostIoReply, HostIoRequest, HostNanos};

use super::*;
use crate::hle::graphics::KernelHost;

mod vsync;

/// Where nvsrv's ioctl arguments and their out buffers go: the third page, in and out halves.
const IOCTL_IN: u64 = FIRMWARE;
const IOCTL_OUT: u64 = FIRMWARE + PAGE_SIZE / 2;

/// A sparse region inside NVN's window (row 25: `0x8000000` to `0xC0_0000_0000`), eight pages.
const REGION: u64 = 0x81_0000_0000;
const REGION_SIZE: u64 = 8 * PAGE_SIZE;

/// The core and thread a turn's events are recorded on: the timer thread's.
const TIMER: (u8, ThreadId) = (0, ThreadId(0));

/// A rig whose services are installed with `graphics` as the graphics server.
fn rig_with(graphics: &Graphics) -> Rig {
    let dispatcher = dispatcher();
    let env = Arc::new(LiveEnv::new(FixedClock::new(0, 1), 0));
    let services: Arc<dyn Env> = Arc::<LiveEnv>::clone(&env);
    let ready = Arc::new(ReadyServers::default());
    let fsp_reads = reads_on(Arc::clone(&services), &ready);
    let guest_log: Arc<dyn lm::GuestLog> = Arc::new(NoGuestLog);
    install_with_graphics(&dispatcher, services, fsp_reads, guest_log, graphics).unwrap();
    let recorder = Recorder::new();
    let events = Arc::new(VecSink::new());
    recorder.add_sink(Arc::clone(&events));
    Rig {
        dispatcher,
        memory: FlatMemory::new(BASE, usize::try_from(PAGES * PAGE_SIZE).unwrap()),
        env,
        recorder,
        events,
        ready,
    }
}

/// A TIPC reply: its result, its data words and its copied handles.
struct Tipc {
    result: u32,
    data: Vec<u32>,
    copied: Vec<Handle>,
}

impl Rig {
    /// The graphics server's root, as a title gets it: `appletOE`'s `ISelfController` 2000.
    fn graphics_root(&self) -> Handle {
        let oe = self.service(am::OE_NAME);
        let root = self.convert(oe);
        let (_, buffer) = self.send(oe, &open_proxy(root, Handle::CURRENT_PROCESS));
        let (_, _, objects) = domain_reply(&buffer, 0);
        let proxy_interface = am::IApplicationProxy::default();
        let (_, buffer) = self.send(
            oe,
            &domain(objects[0], &proxy_interface, "get_self_controller"),
        );
        let (_, _, controller) = domain_reply(&buffer, 0);
        let ask = domain(
            controller[0],
            &am::ISelfController,
            "get_graphics_supervisor",
        );
        let (code, buffer) = self.send(oe, &ask);
        assert_eq!(code, ResultCode::SUCCESS);
        let message = Message::parse_reply_sized(&buffer, SessionKind::Domain, 0).unwrap();
        Handle::from_raw(message.copy_handles[0])
    }

    /// Sends the TIPC message `words` on `session`; its reply.
    fn tipc(&self, session: Handle, words: &[u32]) -> Tipc {
        let mut buffer = [0; BUFFER_SIZE];
        let bytes: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
        buffer[..bytes.len()].copy_from_slice(&bytes);
        self.memory.write(TLS, &buffer).unwrap();
        let regs = self.call(SvcId::SendSyncRequest, u64::from(session.raw()), 0);
        assert_eq!(result(&regs), ResultCode::SUCCESS, "{words:x?}");
        self.memory.read(TLS, &mut buffer).unwrap();
        let reply = Message::parse_reply(&buffer, SessionKind::Session).unwrap();
        let Body::TipcReply { result, data } = reply.body else {
            unreachable!("not a TIPC reply: {:?}", reply.body);
        };
        let data = data
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| u32::from_le_bytes(*word))
            .collect();
        Tipc {
            result,
            data,
            copied: reply
                .copy_handles
                .into_iter()
                .map(Handle::from_raw)
                .collect(),
        }
    }

    /// A session one of the root's family commands hands out (`command`, no data words).
    fn family(&self, root: Handle, command: u32) -> Handle {
        let reply = self.tipc(root, &[tipc_type(command), 0]);
        assert_eq!((reply.result, reply.copied.len()), (0, 1), "{command}");
        reply.copied[0]
    }

    /// nvsrv's `Open` of `device`: its fd.
    fn nv_open(&self, nv: Handle, device: Device) -> i32 {
        let path = device.path().as_bytes();
        self.memory.write(PATH, path).unwrap();
        let open = libnx(nv_command::OPEN).send_buffer(BufferDescriptor {
            address: PATH,
            size: u64::try_from(path.len()).unwrap(),
            flags: 0,
        });
        let (code, buffer) = self.send(nv, &open);
        assert_eq!(code, ResultCode::SUCCESS);
        let (word, data, _) = reply(&buffer, 8);
        assert_eq!(word, 0);
        let result = u32::from_le_bytes(data[4..].try_into().unwrap());
        assert_eq!(result, NvResult::Success.raw(), "{device:?}");
        i32::from_le_bytes(data[..4].try_into().unwrap())
    }

    /// nvsrv's `Ioctl` of `raw` on `fd` with `input`, as libnx sends it with map-alias buffers;
    /// the ioctl's result and the out buffer of `input`'s size afterwards.
    fn nv_ioctl(&self, nv: Handle, fd: i32, raw: u32, input: &[u8]) -> (u32, Vec<u8>) {
        self.memory.write(IOCTL_IN, input).unwrap();
        self.memory.write(IOCTL_OUT, &vec![0; input.len()]).unwrap();
        let mut data = fd.to_le_bytes().to_vec();
        data.extend(raw.to_le_bytes());
        let size = u16::try_from(input.len()).unwrap();
        let ioctl = libnx(nv_command::IOCTL)
            .data(&data)
            .send_buffer(BufferDescriptor {
                address: IOCTL_IN,
                size: u64::from(size),
                flags: 0,
            })
            .static_descriptor(StaticDescriptor {
                counter: 0,
                address: 0,
                size: 0,
            })
            .receive_buffer(BufferDescriptor {
                address: IOCTL_OUT,
                size: u64::from(size),
                flags: 0,
            })
            .out_pointer(ReceiveDescriptor {
                address: 0,
                size: 0,
            });
        let (code, buffer) = self.send(nv, &ioctl);
        assert_eq!(code, ResultCode::SUCCESS);
        let (word, data, _) = reply(&buffer, 4);
        assert_eq!(word, 0);
        let mut out = vec![0; input.len()];
        self.memory.read(IOCTL_OUT, &mut out).unwrap();
        (u32::from_le_bytes(data.try_into().unwrap()), out)
    }

    /// `ResetSignal` on `handle`.
    fn reset(&self, handle: Handle) {
        let regs = self.call(SvcId::ResetSignal, u64::from(handle.raw()), 0);
        assert_eq!(result(&regs), ResultCode::SUCCESS);
    }

    /// The timer thread's turn over this rig's kernel, at time 0.
    fn turn(&self, graphics: &Graphics) {
        self.turn_at(graphics, HostNanos(0));
    }

    /// The timer thread's turn over this rig's kernel, at `now`.
    fn turn_at(&self, graphics: &Graphics, now: HostNanos) {
        let mut host = KernelHost::new(
            &self.dispatcher,
            &*self.env,
            &self.recorder,
            &self.memory,
            TIMER,
        );
        graphics.turn(&mut host, now);
        host.finish().unwrap();
    }

    /// Every event recorded so far, as core, thread and kind.
    fn all_events(&self) -> Vec<(u8, ThreadId, EventKind)> {
        self.events
            .events()
            .into_iter()
            .map(|event| (event.core, event.thread, event.kind))
            .collect()
    }

    /// What each of gsv's four servers saves now, by number.
    fn gsv_servers(&self) -> Vec<(ServiceId, Vec<u8>)> {
        let mut saved = Vec::new();
        let out = self.dispatcher.inspect_handlers(|service, handler| {
            let gsv = [GSV_ROOT, GSV_SYNCPT, GSV_GPU, GSV_NVSRV];
            if gsv.contains(&service) {
                let handler: &dyn core::any::Any = handler;
                let server = handler
                    .downcast_ref::<starquake_services::framework::Server>()
                    .unwrap();
                saved.push((service, server.save().unwrap()));
            }
        });
        assert_eq!(out, 0);
        saved
    }
}

fn tipc_type(command: u32) -> u32 {
    u32::from(TIPC_COMMAND_OFFSET) + command
}

/// Two `u32` words of `value`, low first.
fn halves(value: u64) -> [u32; 2] {
    let bytes = value.to_le_bytes();
    [
        u32::from_le_bytes(bytes[..4].try_into().unwrap()),
        u32::from_le_bytes(bytes[4..].try_into().unwrap()),
    ]
}

fn le(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|word| word.to_le_bytes()).collect()
}

/// libnx `_NV_IOWR(group, nr, size)`: both direction bits.
const fn iowr(group: u32, nr: u32, size: u32) -> u32 {
    0xC000_0000 | (size << 16) | (group << 8) | nr
}

/// nvhost-as-gpu's `AllocAsEx`: `_NV_IOWR('A', 0x09, ..)`, 40 bytes (`IoctlAllocAsEx`).
const AS_ALLOC_AS_EX: u32 = iowr(0x41, 0x09, 40);
/// nvhost-as-gpu's `BindChannel`: `_NV_IOW('A', 0x01, ..)`, 4 bytes (`IoctlBindChannel`).
const AS_BIND_CHANNEL: u32 = 0x4004_4101;
/// nvhost-gpu's `AllocGPFIFOEx2`: `_NV_IOWR('H', 0x1A, ..)`, 32 bytes.
const ALLOC_GPFIFO_EX2: u32 = iowr(0x48, 0x1A, 32);
/// nvhost-gpu's `SubmitGPFIFO` with no entries after the 24-byte struct: `_NV_IOWR('H', 0x08,
/// ..)`.
const SUBMIT_GPFIFO: u32 = iowr(0x48, 0x08, 24);

/// What the session leaves to check: the syncpt slot event and the gpu job event, the channel's
/// syncpoint.
struct Session {
    slot_event: Handle,
    job_event: Handle,
    syncpoint: SyncpointId,
}

/// The recorded session: a channel's submission that an armed syncpt wait waits for, and a gpu
/// worker job that completes only once the GPU says the ordered update is done. `turn` is the
/// timer thread's turn, called where the session needs one; each call is numbered from 0.
fn session(rig: &Rig, mut turn: impl FnMut(usize)) -> Session {
    let root = rig.graphics_root();
    let syncpt = rig.family(root, root_command::GET_SYNCPOINT_SESSION);
    let nv = rig.family(root, root_command::GET_NVIDIA_DRIVER);

    // syncpt 21 (PR `SetEventCount`: `0x25, 1`, the count at word 2): one slot; then 22 (PR
    // `GetEvent`: `0x26, 1`, the slot at word 2): slot 0's event.
    let count = rig.tipc(syncpt, &[tipc_type(syncpt_command::SET_EVENT_COUNT), 1, 1]);
    assert_eq!(count.result, 0);
    let slot = rig.tipc(syncpt, &[tipc_type(syncpt_command::GET_EVENT), 1, 0]);
    assert_eq!((slot.result, slot.copied.len()), (0, 1));
    let slot_event = slot.copied[0];

    // A channel: an address space (`AllocAsEx`: flags, as_fd, big page size, reserved, start,
    // end, split; all zero, PR's native layout), bound to an nvhost-gpu fd, given a GPFIFO.
    let as_fd = rig.nv_open(nv, Device::NvhostAsGpu);
    let (result, _) = rig.nv_ioctl(nv, as_fd, AS_ALLOC_AS_EX, &[0; 40]);
    assert_eq!(result, NvResult::Success.raw());
    let gpu_fd = rig.nv_open(nv, Device::NvhostGpu);
    let (result, _) = rig.nv_ioctl(nv, as_fd, AS_BIND_CHANNEL, &gpu_fd.to_le_bytes());
    assert_eq!(result, NvResult::Success.raw());
    // `IoctlAllocGpfifoEx`: entries, jobs, flags, the fence out (id, value), three reserved.
    let alloc = le(&[0x400, 0, 0, 0, 0, 0, 0, 0]);
    let (result, out) = rig.nv_ioctl(nv, gpu_fd, ALLOC_GPFIFO_EX2, &alloc);
    assert_eq!(result, NvResult::Success.raw());
    let id = u32::from_le_bytes(out[12..16].try_into().unwrap());
    let syncpoint = SyncpointId::new(u8::try_from(id).unwrap()).unwrap();

    // syncpt 201 (PR `DecodeSyncpointWait`: `0xd9, 5`, slot, `SyncCounterId` (the NOT of the
    // id), threshold low and high, mode 0): slot 0 waits for the channel's syncpoint to reach 2.
    let arm = rig.tipc(
        syncpt,
        &[tipc_type(syncpt_command::ARM_WAIT), 5, 0, !id, 2, 0, 0],
    );
    assert_eq!(arm.result, 0);
    assert_eq!(poll(rig, slot_event), KernelError::TimedOut.code());

    // `IoctlSubmitGpfifo`: address (two words, not read), no entries, flag bit 1 (increment by
    // 2), no fence; the fence comes back at 16 and 20.
    let submit = le(&[0, 0, 0, FENCE_INCREMENT, 0, 0]);
    let (result, out) = rig.nv_ioctl(nv, gpu_fd, SUBMIT_GPFIFO, &submit);
    assert_eq!(result, NvResult::Success.raw());
    assert_eq!(out[16..24], le(&[id, 2])[..]);
    // Nothing is signalled until the timer thread's turn.
    assert_eq!(poll(rig, slot_event), KernelError::TimedOut.code());
    turn(0);
    assert_eq!(poll(rig, slot_event), ResultCode::SUCCESS);

    let job_event = gpu_job(rig, root, || turn(1));
    Session {
        slot_event,
        job_event,
        syncpoint,
    }
}

/// The session's second half: a gpu worker job (a sparse region's destroy) that completes only
/// once the GPU says the ordered update is done and `turn` drains that; the job's event.
fn gpu_job(rig: &Rig, root: Handle, turn: impl FnOnce()) -> Handle {
    // The gpu session (497, flags 4) and its worker: 12 (the last index's event), 21 and 22,
    // as `starquake_gsv::gpu`'s tests set it up; slot 0's event (12, index 0); a graphics device
    // (41: name, flags 0, label) and an address space (33).
    let gpu = rig.tipc(root, &[tipc_type(root_command::OPEN_GPU_SESSION), 2, 4, 0]);
    assert_eq!((gpu.result, gpu.copied.len()), (0, 1));
    let gpu = gpu.copied[0];
    let last = 63;
    rig.tipc(
        gpu,
        &[tipc_type(gpu_command::GET_NOTIFICATION_EVENT), 1, last],
    );
    rig.tipc(gpu, &[tipc_type(gpu_command::REQUEST_WORKER), 1, last]);
    let worker = rig.tipc(gpu, &[tipc_type(gpu_command::ACQUIRE_WORKER), 0]);
    assert_eq!((worker.result, worker.copied.len()), (0, 1));
    let worker = worker.copied[0];
    let job = rig.tipc(gpu, &[tipc_type(gpu_command::GET_NOTIFICATION_EVENT), 1, 0]);
    let job_event = job.copied[0];
    let device = rig.tipc(
        gpu,
        &[
            tipc_type(gpu_command::CREATE_GRAPHICS_DEVICE),
            6,
            0x11,
            0,
            0,
            0,
            0x22,
            0,
        ],
    );
    assert_eq!(device.result, 0);
    let space = rig.tipc(gpu, &[tipc_type(gpu_command::CREATE_ADDRESS_SPACE), 1, 0]);
    assert_eq!(space.result, 0);
    let (device, space) = (device.data[0], space.data[0]);

    // 214 (`0xe6, 10`: space, first, last, page shift 12, mode 0, device, sync 0, slot 0)
    // completes at once; 215 (`0xe7, 7`: space, first, last, sync 0, slot 0) only once the GPU
    // says it is done and the turn drains that.
    let [first_low, first_high] = halves(REGION);
    let [last_low, last_high] = halves(REGION + REGION_SIZE - 1);
    let create = rig.tipc(
        worker,
        &[
            tipc_type(gpu_command::CREATE_SPARSE_REGION),
            10,
            space,
            first_low,
            first_high,
            last_low,
            last_high,
            12,
            0,
            device,
            0,
            0,
        ],
    );
    assert_eq!(create.result, 0);
    assert_eq!(poll(rig, job_event), ResultCode::SUCCESS);
    rig.reset(job_event);
    let destroy = rig.tipc(
        worker,
        &[
            tipc_type(gpu_command::DESTROY_SPARSE_REGION),
            7,
            space,
            first_low,
            first_high,
            last_low,
            last_high,
            0,
            0,
        ],
    );
    assert_eq!(destroy.result, 0);
    assert_eq!(poll(rig, job_event), KernelError::TimedOut.code());
    turn();
    assert_eq!(poll(rig, job_event), ResultCode::SUCCESS);
    job_event
}

/// The gsv notes among `events`, in order, decoded.
fn notes(events: &[(u8, ThreadId, EventKind)]) -> Vec<Note> {
    events
        .iter()
        .filter_map(|(_, _, kind)| match *kind {
            EventKind::ServiceNote {
                service,
                code,
                value,
            } if service == GSV_ROOT => Some(Note::decode(code, value).unwrap()),
            _ => None,
        })
        .collect()
}

/// The state a replay is compared on: gsv's shared state, the shared page, its GPU memory, the
/// syncpoints, the queue, the four servers, and the GPU front's channels.
fn state_of(rig: &Rig, graphics: &Graphics) -> Vec<Vec<u8>> {
    let shared = graphics.shared();
    let mut state = vec![
        shared.save().unwrap(),
        shared.save_page().unwrap(),
        shared.lock().gpu_memory().save().unwrap(),
        shared.syncpoints().save().unwrap(),
        shared.queue().save().unwrap(),
        format!("{:?}", graphics.front().channels()).into_bytes(),
    ];
    for (service, bytes) in rig.gsv_servers() {
        state.push(service.0.to_le_bytes().to_vec());
        state.push(bytes);
    }
    state
}

/// The root of a run hands out the syncpt, nvsrv and gpu servers' sessions, and each answers its
/// own commands: nvsrv's 13, the command a title sends right after 1402, is nvsrv's
/// (`SetGraphicsFirmwareMemoryMarginEnabled`, an Eden stub that replies nothing), and recorded
/// under nvsrv's number, not the root's.
#[test]
fn the_root_hands_out_the_graphics_server_s_sessions() {
    let graphics = Graphics::new(FrontMode::Live, None);
    let rig = rig_with(&graphics);
    let root = rig.graphics_root();
    let nv = rig.family(root, root_command::GET_NVIDIA_DRIVER);
    let margin = libnx(nv_command::SET_GRAPHICS_FIRMWARE_MEMORY_MARGIN_ENABLED);
    let (code, buffer) = rig.send(nv, &margin);
    assert_eq!(code, ResultCode::SUCCESS);
    assert_eq!(reply(&buffer, 0), (0, Vec::new(), Vec::new()));

    let syncpt = rig.family(root, root_command::GET_SYNCPOINT_SESSION);
    // syncpt 0 (NX2 `Handshake`): the token back.
    let token = rig.tipc(
        syncpt,
        &[tipc_type(syncpt_command::HANDSHAKE), 2, 0x51, 0x52],
    );
    assert_eq!((token.result, token.data), (0, vec![0x51, 0x52]));

    let gpu = rig.tipc(root, &[tipc_type(root_command::OPEN_GPU_SESSION), 2, 4, 0]);
    let range = rig.tipc(
        gpu.copied[0],
        &[tipc_type(gpu_command::GET_NOTIFICATION_RANGE), 0],
    );
    assert_eq!((range.result, range.data), (0, vec![0, 63]));

    let ipc = rig.ipc();
    let tail: Vec<_> = ipc.iter().rev().take(6).rev().copied().collect();
    assert_eq!(
        tail,
        [
            (GSV_ROOT, root_command::GET_NVIDIA_DRIVER),
            (
                GSV_NVSRV,
                nv_command::SET_GRAPHICS_FIRMWARE_MEMORY_MARGIN_ENABLED
            ),
            (GSV_ROOT, root_command::GET_SYNCPOINT_SESSION),
            (GSV_SYNCPT, syncpt_command::HANDSHAKE),
            (GSV_ROOT, root_command::OPEN_GPU_SESSION),
            (GSV_GPU, gpu_command::GET_NOTIFICATION_RANGE),
        ]
    );
}

/// The root of a run answers the native memory and image commands from gsv's shared state, not
/// with its stubs: 1510 sizes an image (the stub answers zeros), 111 hands out id 1 (the stub's
/// id is 0), and a second 112 of it is `NotFound` (the stub answers success); each is recorded
/// under the root's number. Layouts are PR's (`CalculateImageRequirements`: `0x5f6, 17`, the
/// preference at words 2–17 as `native_image_layout.h` reads it, the count at 18;
/// `CreateNativeMemory`: `0x7f, 3`, mode, size; `DestroyNativeMemory`: `0x80, 1`, id).
#[test]
fn the_root_answers_native_memory_and_images() {
    let graphics = Graphics::new(FrontMode::Live, None);
    let rig = rig_with(&graphics);
    let root = rig.graphics_root();

    // 1280 by 720 RGBA8, kind 2, 128-row blocks, 80 GOBs a row; three images: 3 * 80 * 64 *
    // 768 bytes, aligned to a page.
    let mut words = vec![tipc_type(root_command::CALCULATE_IMAGE_REQUIREMENTS), 17];
    words.extend([0, 2, (1280 - 1) | ((720 - 1) << 16), 0, 7, 80, 1, 1]);
    words.extend([0; 8]);
    words.push(3);
    let sized = rig.tipc(root, &words);
    let mut expected = halves(3 * 3_932_160).to_vec();
    expected.extend(halves(0x1000));
    assert_eq!((sized.result, sized.data), (0, expected));

    let [low, high] = halves(PAGE_SIZE);
    let created = rig.tipc(
        root,
        &[
            tipc_type(root_command::CREATE_NATIVE_MEMORY),
            3,
            0,
            low,
            high,
        ],
    );
    assert_eq!((created.result, created.data), (0, vec![1]));
    let destroy = [tipc_type(root_command::DESTROY_NATIVE_MEMORY), 1, 1];
    assert_eq!(rig.tipc(root, &destroy).result, 0);
    assert_eq!(
        rig.tipc(root, &destroy).result,
        KernelError::NotFound.code().raw()
    );
    assert_eq!(
        graphics.shared().lock().native_objects().memories().count(),
        0
    );

    let ipc = rig.ipc();
    let tail: Vec<_> = ipc.iter().rev().take(4).rev().copied().collect();
    assert_eq!(
        tail,
        [
            (GSV_ROOT, root_command::CALCULATE_IMAGE_REQUIREMENTS),
            (GSV_ROOT, root_command::CREATE_NATIVE_MEMORY),
            (GSV_ROOT, root_command::DESTROY_NATIVE_MEMORY),
            (GSV_ROOT, root_command::DESTROY_NATIVE_MEMORY),
        ]
    );
}

/// A run's session: the null front performs the submission's increments and finishes the
/// ordered destroy at once, and each turn drains what that pushed, recording a note for each
/// and signalling what waited on it; nothing is signalled before its turn.
#[test]
fn a_run_s_turns_complete_what_waits_on_the_gpu() {
    let graphics = Graphics::new(FrontMode::Live, None);
    let rig = rig_with(&graphics);
    let mut turns = Vec::new();
    let done = session(&rig, |n| {
        let before = rig.all_events().len();
        rig.turn(&graphics);
        turns.push(notes(&rig.all_events()[before..]));
        assert_eq!(turns.len(), n + 1);
    });
    let ticket = match turns[1].as_slice() {
        [Note::GpuOrderedDone { ticket }] => *ticket,
        other => unreachable!("the destroy's turn recorded {other:?}"),
    };
    assert_eq!(
        turns,
        [
            vec![Note::SyncpointObserved {
                id: done.syncpoint,
                value: 2
            }],
            vec![Note::GpuOrderedDone { ticket }],
        ]
    );
    // Each note is recorded on the timer thread's core and thread.
    for (core, thread, kind) in rig.all_events() {
        if matches!(kind, EventKind::ServiceNote { .. }) {
            assert_eq!((core, thread), TIMER);
        }
    }
    assert_eq!(graphics.shared().syncpoints().read(done.syncpoint), 2);
    assert_ne!(done.slot_event, done.job_event);
}

/// The spec's offline test for the wiring: a recorded gsv session replays to the same state.
/// The session runs once with the null front completing the GPU's work, recording; then again
/// from scratch with a front that completes nothing, the recorded notes fed back
/// ([`Graphics::replay`]) where the run's turns drained them, each note then a turn. The two
/// event streams (every system call, IPC event, wake and note, by core and thread) are the
/// same, and so is the state: gsv's shared state, its page, its GPU memory, the syncpoints, the
/// queue, the front's channels and the four servers' snapshots.
#[test]
fn a_recorded_session_replays_to_the_same_state() {
    let run = Graphics::new(FrontMode::Live, None);
    let recorded = rig_with(&run);
    let mut drained: Vec<Vec<Note>> = Vec::new();
    session(&recorded, |_| {
        let before = recorded.all_events().len();
        recorded.turn(&run);
        drained.push(notes(&recorded.all_events()[before..]));
    });
    assert!(drained.iter().all(|notes| !notes.is_empty()));

    let replay = Graphics::new(FrontMode::Replay, None);
    let replayed = rig_with(&replay);
    session(&replayed, |n| {
        for note in &drained[n] {
            replay.replay(*note).unwrap();
            replayed.turn(&replay);
        }
    });

    assert_eq!(replayed.all_events(), recorded.all_events());
    assert_eq!(state_of(&replayed, &replay), state_of(&recorded, &run));
}

// ----- The shared page and the kernel host -----------------------------------------------------

/// The title's two transfer memories for 204: the writable page at [`NAME`] (free once the root
/// is had) and the read-only pages at [`FIRMWARE`] and [`PATH`].
const WRITABLE_AT: u64 = NAME;
const READ_ONLY_AT: u64 = FIRMWARE;

/// A syncpoint the page is asked about: group 0, bit 40.
const ASKED: u8 = 40;

impl Rig {
    /// Makes system call `id` with `x0` to `x3`; the registers after it.
    fn call4(&self, id: SvcId, x: [u64; 4]) -> GuestRegs {
        let mut regs = GuestRegs {
            tpidrro: TLS,
            ..GuestRegs::default()
        };
        regs.x[..4].copy_from_slice(&x);
        let mut debug = NoDebug;
        let mut caller = Caller {
            core: 0,
            thread: THREAD,
            env: &*self.env,
            memory: &self.memory,
            debug: &mut debug,
            recorder: &self.recorder,
        };
        let record = self
            .dispatcher
            .dispatch(id.number(), &mut regs, &mut caller);
        assert_eq!(record.outcome, Outcome::Completed, "{}", id.name());
        regs
    }

    /// The guest's `CreateTransferMemory` of `size` bytes at `address`, keeping `permission`.
    fn lend(&self, address: u64, size: u32, permission: SvcPermission) -> Handle {
        let bits = u64::from(permission.bits());
        let regs = self.call4(
            SvcId::CreateTransferMemory,
            [0, address, u64::from(size), bits],
        );
        assert_eq!(result(&regs), ResultCode::SUCCESS);
        Handle::from_register(regs.x[1])
    }

    /// The object a handle in the guest's table names.
    fn object(&self, handle: Handle) -> ObjectId {
        self.dispatcher
            .with::<(&HandleTable,), _>(|(handles,)| handles.get(handle))
            .unwrap()
    }

    fn read_u64(&self, address: u64) -> u64 {
        let mut bytes = [0; 8];
        self.memory.read(address, &mut bytes).unwrap();
        u64::from_le_bytes(bytes)
    }

    /// A host over this rig's kernel, as [`Rig::turn`] makes one.
    fn host(&self) -> KernelHost<'_> {
        KernelHost::new(
            &self.dispatcher,
            &*self.env,
            &self.recorder,
            &self.memory,
            TIMER,
        )
    }
}

/// The page bound by 204 through a run's root, with its two transfer memories; the request
/// event's writable end and the answer event, as the guest holds them, and the memories.
fn bound(rig: &Rig) -> (Handle, Handle, Handle, Handle) {
    let root = rig.graphics_root();
    let writable = rig.lend(WRITABLE_AT, WRITABLE_SIZE, SvcPermission::None);
    let read_only = rig.lend(READ_ONLY_AT, READ_ONLY_SIZE, SvcPermission::Read);
    // 204 as PR checks it (`0xdc, 0x80000006, 4`, two copied handles, then `1, 0x1ff, 0x1000,
    // 0, 0x2000, 0`).
    let bind = rig.tipc(
        root,
        &[
            tipc_type(root_command::BIND_SYNCPOINT_MEMORY),
            0x8000_0006,
            4,
            writable.raw(),
            read_only.raw(),
            1,
            0x1FF,
            WRITABLE_SIZE,
            0,
            READ_ONLY_SIZE,
            0,
        ],
    );
    assert_eq!((bind.result, bind.copied.len()), (0, 2));
    (bind.copied[0], bind.copied[1], writable, read_only)
}

/// 204 through a run's root binds the page in the title's own transfer memories, and a request
/// the title makes and signals is answered by the timer thread's turn through the kernel host:
/// the value, the group's acknowledge and the summary written into the read-only memory, the
/// two notes recorded, and the answer event signalled.
#[test]
fn the_shared_page_is_bound_and_answered_through_the_kernel() {
    let graphics = Graphics::new(FrontMode::Live, None);
    let rig = rig_with(&graphics);
    let (request, answer, _, _) = bound(&rig);
    let mut header = [0; page::HEADER_SIZE];
    rig.memory.read(READ_ONLY_AT, &mut header).unwrap();
    assert_eq!(header, page::header());
    let asked = SyncpointId::new(ASKED).unwrap();
    let value_at = READ_ONLY_AT + page::value_offset(asked);
    assert_eq!(rig.read_u64(value_at), UNALLOCATED);

    // The counter moves, the title asks about it, and signals.
    graphics.shared().syncpoints().increment_by(asked, 2);
    let summary = 1_u64 << 0;
    let slot = 1_u64 << ASKED;
    let request_at = WRITABLE_AT + u64::from(page::GROUP_REQUEST_OFFSET);
    rig.memory
        .write(request_at, &summary.to_le_bytes())
        .unwrap();
    let slot_at = WRITABLE_AT + u64::from(page::SLOT_REQUEST_OFFSET);
    rig.memory.write(slot_at, &slot.to_le_bytes()).unwrap();
    assert!(graphics.is_idle());
    let regs = rig.call(SvcId::SignalEvent, u64::from(request.raw()), 0);
    assert_eq!(result(&regs), ResultCode::SUCCESS);
    // The signal waits in the queue for the turn: the graphics server is not idle.
    assert!(!graphics.is_idle());
    assert_eq!(poll(&rig, answer), KernelError::TimedOut.code());

    let before = rig.all_events().len();
    // The drain alone takes the signal and marks the page: with nothing queued, the graphics
    // server is still not idle, since the page's turn has the request to answer.
    let mut host = rig.host();
    graphics.shared().drain(&mut host);
    assert!(graphics.shared().queue().is_empty());
    assert!(!graphics.is_idle());
    graphics.shared().tick_page(&mut host, HostNanos(0));
    host.finish().unwrap();
    assert_eq!(rig.read_u64(value_at), 2);
    let ack_at = READ_ONLY_AT + u64::from(page::SLOT_ACK_OFFSET);
    assert_eq!(rig.read_u64(ack_at), slot);
    let summary_at = READ_ONLY_AT + u64::from(page::GROUP_ACK_OFFSET);
    assert_eq!(rig.read_u64(summary_at), summary);
    assert_eq!(
        notes(&rig.all_events()[before..]),
        [
            Note::SyncptPageAnswered {
                id: asked,
                value: 2
            },
            Note::SyncptPageAcknowledged { groups: summary },
        ]
    );
    assert_eq!(poll(&rig, answer), ResultCode::SUCCESS);
    assert!(graphics.is_idle());

    // Asked again with the counter where the page shows it, the request is held: not idle,
    // with nothing queued, until the turn at the bound answers it.
    rig.reset(answer);
    let summary = summary ^ 1;
    let slot = slot ^ (1 << ASKED);
    rig.memory
        .write(request_at, &summary.to_le_bytes())
        .unwrap();
    rig.memory.write(slot_at, &slot.to_le_bytes()).unwrap();
    let regs = rig.call(SvcId::SignalEvent, u64::from(request.raw()), 0);
    assert_eq!(result(&regs), ResultCode::SUCCESS);
    rig.turn_at(&graphics, HostNanos(0));
    assert!(graphics.shared().queue().is_empty());
    assert!(!graphics.is_idle());
    assert_eq!(graphics.deadline(), Some(HostNanos(page::ANSWER_BOUND)));
    assert_eq!(poll(&rig, answer), KernelError::TimedOut.code());
    rig.turn_at(&graphics, HostNanos(page::ANSWER_BOUND));
    assert_eq!(rig.read_u64(summary_at), summary);
    assert_eq!(poll(&rig, answer), ResultCode::SUCCESS);
    assert!(graphics.is_idle());
    assert_eq!(graphics.deadline(), None);
}

/// The kernel host reads and writes a transfer memory up to its exact end and not a byte past
/// it (`InvalidCurrentMemory`, nothing copied), also where mapped memory follows the transfer
/// memory, so only its bounds can refuse; it refuses an object that is not a transfer memory
/// (`InvalidHandle`), and passes the kernel's own answer to a signal or an unlock through.
#[test]
fn the_kernel_host_reaches_what_the_kernel_holds_and_no_more() {
    let graphics = Graphics::new(FrontMode::Live, None);
    let rig = rig_with(&graphics);
    let (request, _, writable, read_only) = bound(&rig);
    let memory = rig.object(read_only);
    let size = u64::from(READ_ONLY_SIZE);
    let mut host = rig.host();

    rig.memory
        .write(READ_ONLY_AT + size - 4, &[1, 2, 3, 4])
        .unwrap();
    let mut out = [0; 4];
    assert_eq!(
        host.read_transfer_memory(memory, size - 4, &mut out),
        Ok(())
    );
    assert_eq!(out, [1, 2, 3, 4]);
    let mut out = [0xEE; 4];
    assert_eq!(
        host.read_transfer_memory(memory, size - 3, &mut out),
        Err(KernelError::InvalidCurrentMemory)
    );
    assert_eq!(out, [0xEE; 4]);
    assert_eq!(
        host.write_transfer_memory(memory, size - 4, &[5, 6, 7, 8]),
        Ok(())
    );
    let mut written = [0; 4];
    rig.memory
        .read(READ_ONLY_AT + size - 4, &mut written)
        .unwrap();
    assert_eq!(written, [5, 6, 7, 8]);
    assert_eq!(
        host.write_transfer_memory(memory, size - 3, &[9; 4]),
        Err(KernelError::InvalidCurrentMemory)
    );
    rig.memory
        .read(READ_ONLY_AT + size - 4, &mut written)
        .unwrap();
    assert_eq!(written, [5, 6, 7, 8]);
    assert_eq!(
        host.read_transfer_memory(memory, u64::MAX, &mut [0; 1]),
        Err(KernelError::InvalidCurrentMemory)
    );

    // The read-only memory ends where the rig's memory does, so a byte past it is refused by the
    // memory too. The writable one is followed by mapped memory (the read-only one's first
    // page): a byte past its end is refused by the transfer memory's bounds alone.
    let writable_memory = rig.object(writable);
    let size = u64::from(WRITABLE_SIZE);
    assert_eq!(WRITABLE_AT + size, READ_ONLY_AT);
    let mut past = [0; 4];
    rig.memory.read(WRITABLE_AT + size - 3, &mut past).unwrap();
    rig.memory
        .write(WRITABLE_AT + size - 4, &[0x11, 0x22, 0x33, 0x44])
        .unwrap();
    let mut out = [0; 4];
    assert_eq!(
        host.read_transfer_memory(writable_memory, size - 4, &mut out),
        Ok(())
    );
    assert_eq!(out, [0x11, 0x22, 0x33, 0x44]);
    let mut out = [0xEE; 4];
    assert_eq!(
        host.read_transfer_memory(writable_memory, size - 3, &mut out),
        Err(KernelError::InvalidCurrentMemory)
    );
    assert_eq!(out, [0xEE; 4]);
    assert_eq!(
        host.write_transfer_memory(writable_memory, size - 4, &[0x55, 0x66, 0x77, 0x88]),
        Ok(())
    );
    let before = rig.read_u64(WRITABLE_AT + size - 4);
    assert_eq!(
        host.write_transfer_memory(writable_memory, size - 3, &[0x99; 4]),
        Err(KernelError::InvalidCurrentMemory)
    );
    assert_eq!(rig.read_u64(WRITABLE_AT + size - 4), before);
    assert_eq!(
        before.to_le_bytes()[..4],
        [0x55, 0x66, 0x77, 0x88],
        "the write up to the end"
    );

    // An event is not a transfer memory; a transfer memory is not an event.
    let event = rig.object(request);
    assert_eq!(
        host.read_transfer_memory(event, 0, &mut [0; 1]),
        Err(KernelError::InvalidHandle)
    );
    assert_eq!(
        host.write_transfer_memory(event, 0, &[0]),
        Err(KernelError::InvalidHandle)
    );
    assert_eq!(host.signal_event(memory), Err(KernelError::InvalidHandle));
    assert_eq!(host.signal_event(event), Ok(()));

    // A token of a process this kernel does not hold.
    let token = device_share_token(ProcessId::from_parts(9, 9), 0x8000_0000, 0x1000);
    assert_eq!(
        host.unlock_device_share(token),
        Err(DeviceShareError::WrongProcess)
    );
    assert_eq!(host.finish(), Ok(()));
}

/// A note the host records goes to the run's stream on the core and thread the host was given;
/// an environment that fails while the host records or signals is kept, the first one, and
/// `finish` gives it back, while a signal it failed answers `NotImplemented`.
#[test]
fn the_kernel_host_records_on_its_core_and_keeps_the_environment_s_first_failure() {
    let graphics = Graphics::new(FrontMode::Live, None);
    let rig = rig_with(&graphics);
    let (request, ..) = bound(&rig);
    let event = rig.object(request);
    let mut host = KernelHost::new(
        &rig.dispatcher,
        &*rig.env,
        &rig.recorder,
        &rig.memory,
        (3, ThreadId(0x77)),
    );
    let note = Note::QueueRoom {
        channel: starquake_gpu::interface::ChannelId(5),
    }
    .event(GSV_ROOT);
    host.record(note);
    assert_eq!(host.finish(), Ok(()));
    assert_eq!(rig.all_events().last(), Some(&(3, ThreadId(0x77), note)));

    let failing = starquake_trace::ReplayEnv::new(Vec::new());
    let before = rig.all_events().len();
    // A failed record alone, a failed signal alone, and both.
    for (records, signals) in [(true, false), (false, true), (true, true)] {
        let mut host =
            KernelHost::new(&rig.dispatcher, &failing, &rig.recorder, &rig.memory, TIMER);
        if records {
            host.record(note);
        }
        if signals {
            assert_eq!(host.signal_event(event), Err(KernelError::NotImplemented));
        }
        assert!(
            matches!(host.finish(), Err(EnvError::Exhausted { index: 0, .. })),
            "{records} {signals}"
        );
    }
    assert_eq!(rig.all_events().len(), before);

    // Two failures told apart: each reading of this environment fails with the next index, so
    // the host must give back the one with index 0 whichever call met it, not a later one.
    for records_first in [true, false] {
        let failing = Failing::default();
        let mut host =
            KernelHost::new(&rig.dispatcher, &failing, &rig.recorder, &rig.memory, TIMER);
        for records in [records_first, !records_first] {
            if records {
                host.record(note);
            } else {
                assert_eq!(host.signal_event(event), Err(KernelError::NotImplemented));
            }
        }
        assert!(failing.readings() >= 2, "{records_first}");
        let first = host.finish();
        assert!(
            matches!(first, Err(EnvError::Exhausted { index: 0, .. })),
            "{records_first}: {first:?}"
        );
    }
    assert_eq!(rig.all_events().len(), before);
}

/// An environment every reading of which fails, each with the next index
/// (`EnvError::Exhausted { index }`), so two failures can be told apart.
#[derive(Default)]
struct Failing {
    readings: AtomicUsize,
}

impl Failing {
    fn fail(&self, requested: EnvCall) -> EnvError {
        let index = self.readings.fetch_add(1, Ordering::SeqCst);
        EnvError::Exhausted { index, requested }
    }

    fn readings(&self) -> usize {
        self.readings.load(Ordering::SeqCst)
    }
}

impl Env for Failing {
    fn now(&self) -> Result<HostNanos, EnvError> {
        Err(self.fail(EnvCall::Clock))
    }

    fn random_u64(&self) -> Result<u64, EnvError> {
        Err(self.fail(EnvCall::Random))
    }

    fn host_io(&self, _: &HostIoRequest) -> Result<HostIoReply, EnvError> {
        Err(self.fail(EnvCall::HostIo))
    }
}
