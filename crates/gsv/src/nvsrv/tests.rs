//! The nvsrv server driven through CMIF messages laid out as libnx lays them out
//! (`libnx/nx/source/services/nv.c` @ b156b477: `nvOpen`, `nvIoctl`, `nvIoctl2`, `nvIoctl3`,
//! `nvClose`, `nvQueryEvent`), over the services framework's fake kernel.
//!
//! Ioctl arguments are laid out by hand from Eden's structs
//! (`eden/src/core/hle/service/nvdrv/devices/nvmap.h` @ cf654052), and ioctl numbers are written
//! as libnx writes them. One request and its reply are laid out word by word as well
//! (`the_ioctl_message_fixture`); the rest are framed by `starquake-ipc`'s request builder, which
//! that test checks against the hand layout.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use starquake_gpu::interface::{Completions, Ticket, VmUpdate};
use starquake_gpu::syncpoint::Syncpoints;
use starquake_gpu::vm::GpuVm;
use starquake_ipc::{
    BUFFER_SIZE, Body, BufferDescriptor, CommandType, Message, ReceiveDescriptor, RequestBuilder,
    RequestLayout, SFCO_MAGIC, SessionKind, StaticDescriptor,
};
use starquake_kernel::id::{ObjectId, SlabId};
use starquake_kernel::result::{KernelError, ResultCode};
use starquake_services::framework::fake::FakeKernel;
use starquake_services::framework::{
    BufferKind, DeclaredBuffer, Direction, Family, Handled, Protocol, Session, SnapshotError,
    UNKNOWN_COMMAND_ID,
};
use starquake_trace::{EventKind, IpcOutcome, ServiceId};

use super::command::{INITIALIZE_DEVTOOLS, MAP_SHARED_MEM, SET_ARUID_FOR_TEST};
use super::nvmap::{HANDLE_ID_INCREMENT, NvMapStateError};
use super::*;
use crate::completion::CompletionQueue;
use crate::gpu::AsId;

/// The number the tests' server is known by.
const NVSRV: ServiceId = ServiceId(0x1402);
/// What fills the message buffer and the caller's memory beforehand.
pub(super) const STALE: u8 = 0xA5;
/// Where the hand-laid request keeps its argument and its out buffer: below 2^32, so a
/// descriptor's third word holds no address bits.
const IN_AT: u64 = 0x8000_0000;
const OUT_AT: u64 = 0x9000_0000;
/// Where [`Rig::place`] maps the first buffer, and how far apart it maps them: each buffer gets
/// fresh memory, since where two mappings overlap the fake kernel answers from the first.
const PLACE_FROM: u64 = 0x1000_0000;
const PLACE_STRIDE: u64 = 0x10_0000;
/// Memory nothing maps: inline buffers that must not be touched.
const INLINE_AT: u64 = 0xA000_0000;
const INLINE_OUT_AT: u64 = 0xB000_0000;

/// nvmap's `IocCreate` as libnx sends it: `_NV_IOWR(0x01, 0x01, data)`, 8 bytes.
const NVMAP_CREATE: u32 = 0xC008_0101;
/// nvmap's `IocAlloc`: `_NV_IOWR(0x01, 0x04, data)`, 32 bytes.
const NVMAP_ALLOC: u32 = 0xC020_0104;
/// nvmap's `IocFree`: `_NV_IOWR(0x01, 0x05, data)`, 24 bytes.
const NVMAP_FREE: u32 = 0xC018_0105;
/// nvmap's `IocParam`: `_NV_IOWR(0x01, 0x09, data)`, 12 bytes.
const NVMAP_PARAM: u32 = 0xC00C_0109;
/// nvhost-ctrl-gpu's `GetCharacteristics` (`nvhost_ctrl_gpu.cpp`: group 0x47, command 0x5),
/// both directions, 0xB0 bytes, on `Ioctl`: the Switch 1 form, which ctrl-gpu does not serve.
const CTRL_GPU_CHARACTERISTICS: u32 = 0xC0B0_4705;

// ----- The wiring ------------------------------------------------------------------------------

/// One call the front was given.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum FrontCall {
    Create(AsId),
    Update(AsId, VmUpdate, Option<Ticket>),
    Barrier(Ticket),
}

/// A GPU front that records its calls. nvsrv only ever makes address spaces through it.
#[derive(Default)]
pub(super) struct RecordingFront {
    calls: Mutex<Vec<FrontCall>>,
}

impl GpuVmFront for RecordingFront {
    fn create_address_space(&self, space: AsId, _vm: Arc<GpuVm>) {
        self.calls.lock().unwrap().push(FrontCall::Create(space));
    }

    fn update_vm(&self, space: AsId, update: VmUpdate, done: Option<Ticket>) {
        let call = FrontCall::Update(space, update, done);
        self.calls.lock().unwrap().push(call);
    }

    fn barrier(&self, done: Ticket) {
        self.calls.lock().unwrap().push(FrontCall::Barrier(done));
    }
}

impl RecordingFront {
    pub(super) fn calls(&self) -> Vec<FrontCall> {
        self.calls.lock().unwrap().clone()
    }
}

/// gsv's shared state, over a fresh completion queue and syncpoints, as the gpu tests build it.
pub(super) fn shared_state() -> Arc<Shared> {
    let queue = Arc::new(CompletionQueue::new());
    let completions: Arc<dyn Completions> = Arc::<CompletionQueue>::clone(&queue);
    let syncpoints = Arc::new(Syncpoints::new(completions));
    Arc::new(Shared::new(syncpoints, queue, NVSRV))
}

/// nvsrv's wiring: gsv's shared state and a recording front.
pub(super) fn wiring() -> (Arc<Shared>, Arc<RecordingFront>) {
    (shared_state(), Arc::new(RecordingFront::default()))
}

/// `front` as the trait object nvsrv takes.
pub(super) fn as_front(front: &Arc<RecordingFront>) -> Arc<dyn GpuVmFront> {
    Arc::<RecordingFront>::clone(front)
}

/// A server over fresh wiring, its state `state`.
fn server_with(state: NvsrvState) -> Server {
    let (shared, front) = wiring();
    let family = NvsrvFamily::new(Arc::clone(&shared), as_front(&front)).with_state(state);
    nvsrv_server(shared, as_front(&front), UnknownPolicy::Strict).with_family(family)
}

/// The state of `server`'s family.
fn state_of(server: &Server) -> Option<&NvsrvState> {
    server.family::<NvsrvFamily>().map(NvsrvFamily::state)
}

/// An nvsrv state driven directly, without the server, over a GPU memory and a recording front
/// of its own.
pub(super) struct Direct {
    pub(super) state: NvsrvState,
    pub(super) memory: GpuMemory,
    pub(super) front: RecordingFront,
}

impl Direct {
    pub(super) fn new() -> Self {
        Self {
            state: NvsrvState::new(),
            memory: GpuMemory::new(),
            front: RecordingFront::default(),
        }
    }

    /// Opens `device`, which must succeed, with made-up events if it is an nvhost-ctrl-gpu; its
    /// fd.
    pub(super) fn open(&mut self, device: Device) -> i32 {
        let (fd, result) = open_direct(&mut self.state, device);
        assert_eq!(result, NvResult::Success);
        fd
    }

    /// `raw` on `fd` through `variant` under `policy`, with `input` and out buffers of `out_len`
    /// and `inline_out_len` bytes.
    pub(super) fn call(
        &mut self,
        policy: UnknownPolicy,
        variant: IoctlVariant,
        fd: i32,
        raw: u32,
        input: &[u8],
        (out_len, inline_out_len): (usize, usize),
    ) -> IoctlAnswer {
        let call = IoctlCall {
            variant,
            fd,
            ioctl: Ioctl::from_raw(raw),
            input,
            inline_input: &[],
            out_len,
            inline_out_len,
        };
        let gpu = GpuAccess {
            memory: &mut self.memory,
            front: &self.front,
            channels: None,
        };
        self.state.ioctl(policy, &call, gpu)
    }

    /// Closes `fd` over this memory and front.
    pub(super) fn close(&mut self, fd: i32) -> NvResult {
        let gpu = GpuAccess {
            memory: &mut self.memory,
            front: &self.front,
            channels: None,
        };
        self.state.close(fd, gpu)
    }

    /// `raw` on `fd` through `Ioctl`, strict, out buffer 0 the input's size.
    pub(super) fn ioctl(&mut self, fd: i32, raw: u32, input: &[u8]) -> IoctlAnswer {
        let sizes = (input.len(), 0);
        self.call(
            UnknownPolicy::Strict,
            IoctlVariant::Ioctl,
            fd,
            raw,
            input,
            sizes,
        )
    }
}

/// Two events no kernel made, distinct for each fd: what a test that drives the state without a
/// kernel opens an nvhost-ctrl-gpu with.
pub(super) fn made_up_events(fd: u32) -> ctrl_gpu::CtrlGpuEvents {
    ctrl_gpu::CtrlGpuEvents {
        error_notifier: ObjectId::from_parts(0x1_0000 + 2 * fd, 0x77),
        unknown: ObjectId::from_parts(0x1_0001 + 2 * fd, 0x77),
    }
}

/// [`NvsrvState::open`] of `device`'s path, with [`made_up_events`] for the next fd if `device`
/// is an nvhost-ctrl-gpu.
pub(super) fn open_direct(state: &mut NvsrvState, device: Device) -> (i32, NvResult) {
    let events = (device == Device::NvhostCtrlGpu).then(|| made_up_events(state.next_fd));
    state.open(device.path().as_bytes(), events)
}

/// [`NvsrvState::ioctl`] of `raw` on `fd` with `input`, out buffers of `input`'s size, over a
/// fresh GPU memory.
pub(super) fn direct_ioctl(
    state: &mut NvsrvState,
    policy: UnknownPolicy,
    variant: IoctlVariant,
    fd: i32,
    raw: u32,
    input: &[u8],
) -> IoctlAnswer {
    let mut direct = Direct::new();
    std::mem::swap(&mut direct.state, state);
    let sizes = (input.len(), input.len());
    let answer = direct.call(policy, variant, fd, raw, input, sizes);
    std::mem::swap(&mut direct.state, state);
    answer
}

/// nvhost-as-gpu's `AllocAsEx` as NvRm sends it: `_NV_IOWR('A', 0x09, ..)`, a 40-byte struct
/// (`nvhost_as_gpu.h`: `IoctlAllocAsEx`).
pub(super) const AS_ALLOC_AS_EX: u32 = 0xC028_4109;
/// `GetVARegions`: `_NV_IOWR('A', 0x08, ..)`, 0x40 bytes (PR's gpu session sends `0xc0404108`).
pub(super) const AS_GET_VA_REGIONS: u32 = 0xC040_4108;
/// `GetSyncpointShimInfo`: `_NV_IOWR('A', 0x0B, ..)`, 16 bytes (`IoctlSyncpointShimInfo`).
pub(super) const AS_SHIM_INFO: u32 = 0xC010_410B;

/// Little-endian words as bytes.
pub(super) fn le(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|word| word.to_le_bytes()).collect()
}

/// A whole message buffer that starts with `words` and is zero after them.
fn buffer_of(words: &[u32]) -> Vec<u8> {
    let mut buffer = le(words);
    buffer.resize(BUFFER_SIZE, 0);
    buffer
}

/// Where an auto-select buffer goes: its map-alias (A or B) descriptor, or its pointer (X or C)
/// descriptor, the other one empty (libnx `cmifRequestInAutoBuffer`, `cmifRequestOutAutoBuffer`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Via {
    MapAlias,
    Pointer,
}

/// An auto-select buffer: where, at what address, how long.
#[derive(Debug, Clone, Copy)]
pub(super) struct Auto {
    via: Via,
    address: u64,
    len: u16,
}

impl Auto {
    pub(super) fn alias(address: u64, len: u16) -> Self {
        Self {
            via: Via::MapAlias,
            address,
            len,
        }
    }

    pub(super) fn pointer(address: u64, len: u16) -> Self {
        Self {
            via: Via::Pointer,
            address,
            len,
        }
    }

    /// Adds it as an in buffer: an A and an X descriptor.
    pub(super) fn send(self, request: RequestBuilder, index: u32) -> RequestBuilder {
        let (alias, pointer) = match self.via {
            Via::MapAlias => ((self.address, self.len), (0, 0)),
            Via::Pointer => ((0, 0), (self.address, self.len)),
        };
        request
            .send_buffer(BufferDescriptor {
                address: alias.0,
                size: u64::from(alias.1),
                flags: 0,
            })
            .static_descriptor(StaticDescriptor {
                counter: index,
                address: pointer.0,
                size: pointer.1,
            })
    }

    /// Adds it as an out buffer: a B and a C descriptor.
    pub(super) fn receive(self, request: RequestBuilder) -> RequestBuilder {
        let (alias, pointer) = match self.via {
            Via::MapAlias => ((self.address, self.len), (0, 0)),
            Via::Pointer => ((0, 0), (self.address, self.len)),
        };
        request
            .receive_buffer(BufferDescriptor {
                address: alias.0,
                size: u64::from(alias.1),
                flags: 0,
            })
            .out_pointer(ReceiveDescriptor {
                address: pointer.0,
                size: pointer.1,
            })
    }
}

/// A CMIF request to `command` with `data`, in libnx's layout.
pub(super) fn cmif(command: u32, data: &[u8]) -> RequestBuilder {
    RequestBuilder::cmif(CommandType::Request, command)
        .layout(RequestLayout::Libnx)
        .data(data)
}

/// An ioctl command's arguments: the fd and the ioctl number.
pub(super) fn fd_and(fd: i32, raw: u32) -> Vec<u8> {
    let mut data = fd.to_le_bytes().to_vec();
    data.extend(raw.to_le_bytes());
    data
}

/// What a reply said: its CMIF result and its data words.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Answer {
    pub(super) result: u32,
    pub(super) words: Vec<u32>,
}

/// Reads a reply, keeping `words` data words.
pub(super) fn answer(buffer: &[u8], words: usize) -> Answer {
    let message = Message::parse_reply(buffer, SessionKind::Session).unwrap();
    let Body::CmifReply(reply) = message.body else {
        unreachable!("not a CMIF reply: {message:?}");
    };
    let data: Vec<u32> = reply
        .data
        .as_chunks::<4>()
        .0
        .iter()
        .take(words)
        .map(|word| u32::from_le_bytes(*word))
        .collect();
    Answer {
        result: reply.header.result,
        words: data,
    }
}

/// A successful reply with these data words.
pub(super) fn ok(words: &[u32]) -> Answer {
    Answer {
        result: ResultCode::SUCCESS.raw(),
        words: words.to_vec(),
    }
}

/// A server over gsv's shared state and a recording front, a kernel, and one session on it.
pub(super) struct Rig {
    pub(super) kernel: FakeKernel,
    pub(super) server: Server,
    pub(super) shared: Arc<Shared>,
    pub(super) front: Arc<RecordingFront>,
    session: ObjectId,
    next_at: u64,
}

impl Rig {
    /// A rig over `shared` and `front`, its state `state`.
    pub(super) fn over(
        policy: UnknownPolicy,
        state: NvsrvState,
        shared: Arc<Shared>,
        front: Arc<RecordingFront>,
    ) -> Self {
        Self::over_with(policy, state, shared, front, None)
    }

    /// A rig over `shared` and `front`, its state `state`, and `channels` as its channel front
    /// if given.
    pub(super) fn over_with(
        policy: UnknownPolicy,
        state: NvsrvState,
        shared: Arc<Shared>,
        front: Arc<RecordingFront>,
        channels: Option<Arc<dyn gpu::ChannelFront>>,
    ) -> Self {
        let mut kernel = FakeKernel::new(0).unwrap();
        kernel.set_current_service(NVSRV);
        let session = kernel.new_object();
        let mut family = NvsrvFamily::new(Arc::clone(&shared), as_front(&front)).with_state(state);
        if let Some(channels) = channels {
            family = family.with_channel_front(channels);
        }
        let mut server = nvsrv_server(Arc::clone(&shared), as_front(&front), policy)
            .with_family(family)
            .with_service_id(NVSRV);
        server.accept(session).unwrap();
        Self {
            kernel,
            server,
            shared,
            front,
            session,
            next_at: PLACE_FROM,
        }
    }

    fn with_state(policy: UnknownPolicy, state: NvsrvState) -> Self {
        let (shared, front) = wiring();
        Self::over(policy, state, shared, front)
    }

    pub(super) fn with(policy: UnknownPolicy) -> Self {
        Self::with_state(policy, NvsrvState::new())
    }

    pub(super) fn new() -> Self {
        Self::with(UnknownPolicy::Strict)
    }

    pub(super) fn state(&self) -> &NvsrvState {
        state_of(&self.server).unwrap()
    }

    /// The session the rig opened.
    pub(super) const fn session(&self) -> ObjectId {
        self.session
    }

    /// Sends `request` on `session`; the message buffer afterwards.
    pub(super) fn send_on(&mut self, session: ObjectId, request: &RequestBuilder) -> Vec<u8> {
        let mut buffer = vec![STALE; BUFFER_SIZE];
        request.build(&mut buffer).unwrap();
        let handled = self.server.handle(&mut self.kernel, session, &mut buffer);
        assert_eq!(handled, Ok(Handled::Replied));
        buffer
    }

    pub(super) fn send(&mut self, request: &RequestBuilder) -> Vec<u8> {
        self.send_on(self.session, request)
    }

    /// Maps `bytes` in fresh memory; where.
    pub(super) fn place(&mut self, bytes: &[u8]) -> u64 {
        let at = self.next_at;
        assert!(u64::try_from(bytes.len()).unwrap() < PLACE_STRIDE);
        self.next_at += PLACE_STRIDE;
        self.kernel.map(at, bytes.to_vec());
        at
    }

    /// The caller's `len` bytes at `address`.
    pub(super) fn memory(&self, address: u64, len: usize) -> Vec<u8> {
        self.kernel.memory(address, len).unwrap().to_vec()
    }

    /// `Open` of `path` as libnx sends it (the path's bytes, no zero byte): the fd and result.
    fn open(&mut self, path: &[u8]) -> Answer {
        let at = self.place(path);
        let request = cmif(OPEN, &[]).send_buffer(BufferDescriptor {
            address: at,
            size: u64::try_from(path.len()).unwrap(),
            flags: 0,
        });
        answer(&self.send(&request), 2)
    }

    /// Opens `device`, which must succeed; its fd.
    pub(super) fn open_device(&mut self, device: Device) -> i32 {
        let answer = self.open(device.path().as_bytes());
        assert_eq!(answer.result, 0);
        assert_eq!(answer.words[1], NvResult::Success.raw());
        i32::try_from(answer.words[0]).unwrap()
    }

    /// `Ioctl` (1) with `input` in a map-alias in buffer and an out buffer of `out_len` stale
    /// bytes: the result word and the out buffer afterwards.
    pub(super) fn ioctl(
        &mut self,
        fd: i32,
        raw: u32,
        input: &[u8],
        out_len: usize,
    ) -> (Answer, Vec<u8>) {
        let input_at = self.place(input);
        let out_at = self.place(&vec![STALE; out_len]);
        let request = cmif(IOCTL, &fd_and(fd, raw));
        let request = Auto::alias(input_at, len16(input.len())).send(request, 0);
        let request = Auto::alias(out_at, len16(out_len)).receive(request);
        let answer = answer(&self.send(&request), 1);
        (answer, self.memory(out_at, out_len))
    }

    /// `Ioctl3` (12) with `input` in a map-alias in buffer, and out buffers 0 and 1 of `out_len`
    /// and `inline_len` stale bytes: the result word and both out buffers afterwards.
    pub(super) fn ioctl3(
        &mut self,
        fd: i32,
        raw: u32,
        input: &[u8],
        (out_len, inline_len): (usize, usize),
    ) -> (Answer, Vec<u8>, Vec<u8>) {
        let input_at = self.place(input);
        let out_at = self.place(&vec![STALE; out_len]);
        let inline_at = self.place(&vec![STALE; inline_len]);
        let request = cmif(IOCTL3, &fd_and(fd, raw));
        let request = Auto::alias(input_at, len16(input.len())).send(request, 0);
        let request = Auto::alias(out_at, len16(out_len)).receive(request);
        let request = Auto::alias(inline_at, len16(inline_len)).receive(request);
        let answer = answer(&self.send(&request), 1);
        let out = self.memory(out_at, out_len);
        (answer, out, self.memory(inline_at, inline_len))
    }

    /// Runs an nvmap ioctl that must succeed; the struct it gives back.
    fn nvmap(&mut self, fd: i32, raw: u32, input: &[u8]) -> Vec<u8> {
        let (answer, output) = self.ioctl(fd, raw, input, input.len());
        assert_eq!(answer, ok(&[NvResult::Success.raw()]), "{raw:#x}");
        output
    }
}

pub(super) fn len16(len: usize) -> u16 {
    u16::try_from(len).unwrap()
}

// ----- The command table -----------------------------------------------------------------------

/// The ids, names and buffers of Eden's table and libnx's calls; no TIPC command.
#[test]
fn the_command_table_is_eden_s() {
    let auto_in = DeclaredBuffer {
        direction: Direction::In,
        kind: BufferKind::AutoSelect,
    };
    let auto_out = DeclaredBuffer {
        direction: Direction::Out,
        kind: BufferKind::AutoSelect,
    };
    let alias_in = DeclaredBuffer {
        direction: Direction::In,
        kind: BufferKind::MapAlias,
    };
    let table: [(u32, &str, &[DeclaredBuffer]); 11] = [
        (0, "open", &[alias_in]),
        (1, "ioctl", &[auto_in, auto_out]),
        (2, "close", &[]),
        (3, "initialize", &[]),
        (4, "query_event", &[]),
        (6, "get_status", &[]),
        (8, "set_aruid", &[]),
        (9, "result_only", &[]),
        (11, "ioctl2", &[auto_in, auto_in, auto_out]),
        (12, "ioctl3", &[auto_in, auto_out, auto_out]),
        (13, "result_only", &[]),
    ];
    let commands = Nvsrv.commands();
    for (id, name, buffers) in table {
        assert_eq!(commands.name(Protocol::Cmif, id), Some(name), "{id}");
        let row = commands.find(Protocol::Cmif, id).unwrap();
        assert_eq!(row.buffers, buffers, "{id}");
    }
    for id in [MAP_SHARED_MEM, SET_ARUID_FOR_TEST, INITIALIZE_DEVTOOLS, 14] {
        assert_eq!(commands.name(Protocol::Cmif, id), None, "{id}");
    }
    for id in 0..=14 {
        assert_eq!(commands.name(Protocol::Tipc, id), None, "{id}");
    }
    let ids = [
        OPEN,
        IOCTL,
        CLOSE,
        INITIALIZE,
        QUERY_EVENT,
        MAP_SHARED_MEM,
        GET_STATUS,
        SET_ARUID_FOR_TEST,
        SET_ARUID,
        DUMP_GRAPHICS_MEMORY_INFO,
        INITIALIZE_DEVTOOLS,
        IOCTL2,
        IOCTL3,
        SET_GRAPHICS_FIRMWARE_MEMORY_MARGIN_ENABLED,
    ];
    assert_eq!(ids.to_vec(), (0..=13).collect::<Vec<u32>>());
    let rig = Rig::new();
    assert_eq!(rig.server.name(), NVSRV_NAME);
}

/// Each result's word, as `nvdata.h` numbers it.
#[test]
fn nv_results_are_eden_s() {
    let words = [
        (NvResult::Success, 0x0),
        (NvResult::NotImplemented, 0x1),
        (NvResult::NotSupported, 0x2),
        (NvResult::BadParameter, 0x4),
        (NvResult::InsufficientMemory, 0x6),
        (NvResult::InvalidState, 0x8),
        (NvResult::BadValue, 0xB),
        (NvResult::Busy, 0xE),
        (NvResult::FileOperationFailed, 0x3_0003),
        (NvResult::AccessDenied, 0x3_0010),
    ];
    for (result, word) in words {
        assert_eq!(result.raw(), word, "{result:?}");
    }
}

#[test]
fn a_tipc_request_goes_to_the_policy() {
    let mut rig = Rig::new();
    let buffer = rig.send(&RequestBuilder::tipc(u16::try_from(OPEN).unwrap()));
    let message = Message::parse_reply(&buffer, SessionKind::Session).unwrap();
    let Body::TipcReply { result, .. } = message.body else {
        unreachable!("not a TIPC reply: {message:?}");
    };
    assert_eq!(result, UNKNOWN_COMMAND_ID.raw());
    assert_eq!(rig.state().open_files(), 0);
}

#[test]
fn unserved_commands_go_to_the_policy() {
    for policy in [UnknownPolicy::Strict, UnknownPolicy::AutoStub] {
        let mut rig = Rig::with(policy);
        for id in [MAP_SHARED_MEM, SET_ARUID_FOR_TEST, INITIALIZE_DEVTOOLS, 14] {
            let reply = answer(&rig.send(&cmif(id, &[])), 0);
            let expected = match policy {
                UnknownPolicy::Strict => UNKNOWN_COMMAND_ID.raw(),
                UnknownPolicy::AutoStub => ResultCode::SUCCESS.raw(),
            };
            assert_eq!(reply.result, expected, "{id} {policy:?}");
        }
    }
}

/// `Initialize`, `GetStatus` and `SetAruid` reply `Success` after the result; 9 and 13 reply the
/// result alone.
#[test]
fn the_stubs_reply_eden_s_words() {
    let mut rig = Rig::new();
    let success = NvResult::Success.raw();
    // Initialize as libnx sends it: a transfer memory size, two copied handles. Nothing is read.
    let buffer = rig.send(&cmif(INITIALIZE, &0x30_0000_u32.to_le_bytes()));
    assert_eq!(answer(&buffer, 1), ok(&[success]));
    assert_eq!(answer(&rig.send(&cmif(GET_STATUS, &[])), 1), ok(&[success]));
    let aruid = 0x0000_0051_0000_00A1_u64.to_le_bytes();
    assert_eq!(
        answer(&rig.send(&cmif(SET_ARUID, &aruid)), 1),
        ok(&[success])
    );
    // Exactly the result: the data is empty.
    for id in [
        DUMP_GRAPHICS_MEMORY_INFO,
        SET_GRAPHICS_FIRMWARE_MEMORY_MARGIN_ENABLED,
    ] {
        let buffer = rig.send(&cmif(id, &[]));
        let words = [0, 6, 0, 0, SFCO_MAGIC, 0, 0, 0];
        assert_eq!(buffer, buffer_of(&words), "{id}");
    }
}

// ----- Open, Close, QueryEvent -----------------------------------------------------------------

/// Every device opens, with fds from 1 up, one each; the reply is the fd and `Success`.
#[test]
fn every_device_opens_with_fds_from_one() {
    let mut rig = Rig::new();
    // The first reply, word by word: header, raw size 8 (padding, `SFCO`, two data words),
    // padding, `SFCO`, version, result, token, fd 1, `Success`.
    let at = rig.place(b"/dev/nvmap");
    let request = cmif(OPEN, &[]).send_buffer(BufferDescriptor {
        address: at,
        size: 10,
        flags: 0,
    });
    let buffer = rig.send(&request);
    assert_eq!(buffer, buffer_of(&[0, 8, 0, 0, SFCO_MAGIC, 0, 0, 0, 1, 0]));
    for (index, device) in DEVICES.into_iter().enumerate().skip(1) {
        let fd = i32::try_from(index).unwrap() + 1;
        assert_eq!(rig.open_device(device), fd, "{device:?}");
        assert_eq!(rig.state().file(fd), Ok(device));
    }
    assert_eq!(rig.state().open_files(), DEVICES.len());
    // The same device again: a new fd.
    assert_eq!(rig.open_device(Device::Nvmap), 11);
}

/// The devices are Eden's and PR's paths, in the enum's order, each found by its own path.
#[test]
fn each_device_is_found_by_its_path() {
    let paths = [
        "/dev/nvmap",
        "/dev/nvhost-ctrl-gpu",
        "/dev/nvhost-as-gpu",
        "/dev/nvhost-tsg-gpu",
        "/dev/nvhost-gpu",
        "/dev/nvhost-ctrl",
        "/dev/nvdisp_disp0",
        "/dev/nvhost-nvdec",
        "/dev/nvhost-nvjpg",
        "/dev/nvhost-vic",
    ];
    for (device, path) in DEVICES.into_iter().zip(paths) {
        assert_eq!(device.path(), path);
        assert_eq!(Device::from_path(path.as_bytes()), Some(device));
    }
    assert_eq!(PROFILER_PATH, "/dev/nvhost-prof-gpu");
    assert_eq!(LONGEST_PATH, 20);
    assert_eq!(Device::from_path(b"/dev/nvmap\0"), None);
    assert_eq!(Device::from_path(b""), None);
}

/// The path ends at its first zero byte; without one, at the buffer's end.
#[test]
fn a_path_ends_at_its_first_zero_byte() {
    let success = NvResult::Success.raw();
    let failed = (
        INVALID_FD.cast_unsigned(),
        NvResult::FileOperationFailed.raw(),
    );
    let cases: [(&[u8], (u32, u32)); 6] = [
        (b"/dev/nvmap\0", (1, success)),
        (b"/dev/nvmap\0garbage", (2, success)),
        (
            b"/dev/nvmap\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0",
            (3, success),
        ),
        (b"/dev/nvmapX", failed),
        (b"/dev/nvma", failed),
        (b"", failed),
    ];
    let mut rig = Rig::new();
    for (path, (fd, result)) in cases {
        assert_eq!(rig.open(path), ok(&[fd, result]), "{path:?}");
    }
}

/// Only one byte past the longest path is read: a path buffer whose tail is not mapped still
/// opens, and the name inside it is found.
#[test]
fn a_path_is_read_up_to_one_byte_past_the_longest() {
    let mut rig = Rig::new();
    let mut path = Device::NvhostCtrlGpu.path().as_bytes().to_vec();
    assert_eq!(path.len(), LONGEST_PATH);
    path.push(0);
    let at = rig.place(&path);
    // The descriptor says 0x100 bytes; only `LONGEST_PATH + 1` are mapped.
    let request = cmif(OPEN, &[]).send_buffer(BufferDescriptor {
        address: at,
        size: 0x100,
        flags: 0,
    });
    assert_eq!(answer(&rig.send(&request), 2), ok(&[1, 0]));
    // A longest path and one more byte that is not zero: what is read has no zero byte, so the
    // path is all 21 bytes, which no device has.
    let mut long = Device::NvhostCtrlGpu.path().as_bytes().to_vec();
    long.push(b'X');
    let answer = rig.open(&long);
    assert_eq!(answer.words[1], NvResult::FileOperationFailed.raw());
}

#[test]
fn the_profiler_is_not_supported() {
    let mut rig = Rig::new();
    let not_supported = ok(&[0, NvResult::NotSupported.raw()]);
    assert_eq!(rig.open(PROFILER_PATH.as_bytes()), not_supported);
    assert_eq!(rig.state().open_files(), 0);
    // Checked before the table: refused as not supported even with the table full.
    let mut rig = Rig::with_state(UnknownPolicy::Strict, full_table(MAX_OPEN_FILES));
    assert_eq!(rig.open(PROFILER_PATH.as_bytes()), not_supported);
}

#[test]
fn an_unknown_path_is_refused_under_both_policies() {
    for policy in [UnknownPolicy::Strict, UnknownPolicy::AutoStub] {
        let mut rig = Rig::with(policy);
        for path in [
            &b"/dev/nvhost-dbg-gpu"[..],
            b"/dev/nvhost-sched-gpu",
            b"nvmap",
        ] {
            let refused = ok(&[0xFFFF_FFFF, NvResult::FileOperationFailed.raw()]);
            assert_eq!(rig.open(path), refused, "{path:?} {policy:?}");
        }
        assert_eq!(rig.state().open_files(), 0);
        // Nothing was used up: the next open gets fd 1.
        assert_eq!(rig.open_device(Device::Nvmap), 1);
    }
}

/// A state with `count` nvmap files open, as fds 1 to `count`.
fn full_table(count: usize) -> NvsrvState {
    let count = u32::try_from(count).unwrap();
    NvsrvState {
        files: (FIRST_FD..FIRST_FD + count)
            .map(|fd| (fd, Device::Nvmap))
            .collect(),
        next_fd: FIRST_FD + count,
        ..NvsrvState::new()
    }
}

#[test]
fn open_refuses_a_full_table_and_spent_fds() {
    let refused = ok(&[0xFFFF_FFFF, NvResult::FileOperationFailed.raw()]);
    // One below the limit: the last file opens.
    let mut rig = Rig::with_state(UnknownPolicy::Strict, full_table(MAX_OPEN_FILES - 1));
    let last = u32::try_from(MAX_OPEN_FILES).unwrap();
    assert_eq!(rig.open(b"/dev/nvmap"), ok(&[last, 0]));
    // At the limit: refused, and no fd used.
    assert_eq!(rig.open(b"/dev/nvmap"), refused);
    assert_eq!(rig.state().open_files(), MAX_OPEN_FILES);
    // Closing one makes room, and the next fd is the one not used.
    let reply = answer(&rig.send(&cmif(CLOSE, &1_i32.to_le_bytes())), 1);
    assert_eq!(reply, ok(&[0]));
    assert_eq!(rig.open(b"/dev/nvmap"), ok(&[last + 1, 0]));

    // The last fd is handed out; after it, fds are spent.
    let state = NvsrvState {
        next_fd: FD_END - 1,
        ..NvsrvState::new()
    };
    let mut rig = Rig::with_state(UnknownPolicy::Strict, state);
    let last = i32::MAX.cast_unsigned();
    assert_eq!(rig.open(b"/dev/nvmap"), ok(&[last, 0]));
    assert_eq!(rig.open(b"/dev/nvmap"), refused);
    assert_eq!(rig.state().open_files(), 1);
}

#[test]
fn close_checks_the_fd_then_closes_it() {
    let mut rig = Rig::new();
    let fd = rig.open_device(Device::Nvmap);
    let close = |rig: &mut Rig, fd: i32| answer(&rig.send(&cmif(CLOSE, &fd.to_le_bytes())), 1);
    assert_eq!(close(&mut rig, -1), ok(&[NvResult::InvalidState.raw()]));
    assert_eq!(
        close(&mut rig, i32::MIN),
        ok(&[NvResult::InvalidState.raw()])
    );
    assert_eq!(close(&mut rig, 0), ok(&[NvResult::NotImplemented.raw()]));
    assert_eq!(
        close(&mut rig, fd + 1),
        ok(&[NvResult::NotImplemented.raw()])
    );
    assert_eq!(rig.state().open_files(), 1);
    assert_eq!(close(&mut rig, fd), ok(&[NvResult::Success.raw()]));
    assert_eq!(rig.state().file(fd), Err(NvResult::NotImplemented));
    // Closed: a second close, and an ioctl, find nothing.
    assert_eq!(close(&mut rig, fd), ok(&[NvResult::NotImplemented.raw()]));
    let (reply, _) = rig.ioctl(fd, NVMAP_CREATE, &le(&[0x1000, 0]), 8);
    assert_eq!(reply, ok(&[NvResult::NotImplemented.raw()]));
    // The fd is not reused.
    assert_eq!(rig.open_device(Device::Nvmap), fd + 1);
}

/// Closing an fd leaves its neighbours open.
#[test]
fn close_closes_only_its_fd() {
    let mut rig = Rig::new();
    let first = rig.open_device(Device::Nvmap);
    let second = rig.open_device(Device::NvhostGpu);
    let third = rig.open_device(Device::NvhostAsGpu);
    let reply = answer(&rig.send(&cmif(CLOSE, &second.to_le_bytes())), 1);
    assert_eq!(reply, ok(&[0]));
    assert_eq!(rig.state().file(first), Ok(Device::Nvmap));
    assert_eq!(rig.state().file(second), Err(NvResult::NotImplemented));
    assert_eq!(rig.state().file(third), Ok(Device::NvhostAsGpu));
}

/// The fd checks come first; a device with no events (nvmap) answers `BadParameter` and no
/// handle for every id, 1 and 2 (ctrl-gpu's) included. ctrl-gpu's own are in `ctrl_gpu/tests.rs`
/// (`query_event_1_and_2_hand_out_the_events_made_at_open`).
#[test]
fn query_event_checks_the_fd_and_finds_no_event() {
    let mut rig = Rig::new();
    let fd = rig.open_device(Device::Nvmap);
    let query = |rig: &mut Rig, fd: i32, event: u32| {
        let mut data = fd.to_le_bytes().to_vec();
        data.extend(event.to_le_bytes());
        answer(&rig.send(&cmif(QUERY_EVENT, &data)), 1)
    };
    assert_eq!(query(&mut rig, -5, 1), ok(&[NvResult::InvalidState.raw()]));
    assert_eq!(
        query(&mut rig, fd + 1, 1),
        ok(&[NvResult::NotImplemented.raw()])
    );
    for event in [0, 1, 2, u32::MAX] {
        assert_eq!(
            query(&mut rig, fd, event),
            ok(&[NvResult::BadParameter.raw()])
        );
    }
    // No handle is given.
    let mut data = fd.to_le_bytes().to_vec();
    data.extend(1_u32.to_le_bytes());
    let buffer = rig.send(&cmif(QUERY_EVENT, &data));
    let message = Message::parse_reply(&buffer, SessionKind::Session).unwrap();
    assert_eq!(message.copy_handles, Vec::<u32>::new());
}

/// Every session shares the open files and the handles, as both references share the nvdrv
/// module: an fd opened on one session is used on another.
#[test]
fn every_session_shares_the_files_and_handles() {
    let mut rig = Rig::new();
    let other = rig.kernel.new_object();
    rig.server.accept(other).unwrap();
    let fd = rig.open_device(Device::Nvmap);
    let input_at = rig.place(&le(&[0x1000, 0]));
    let out_at = rig.place(&[STALE; 8]);
    let request = cmif(IOCTL, &fd_and(fd, NVMAP_CREATE));
    let request = Auto::alias(input_at, 8).send(request, 0);
    let request = Auto::alias(out_at, 8).receive(request);
    let reply = answer(&rig.send_on(other, &request), 1);
    assert_eq!(reply, ok(&[0]));
    assert_eq!(rig.memory(out_at, 8), le(&[0x1000, HANDLE_ID_INCREMENT]));
    assert_eq!(rig.state().nvmap().len(), 1);
}

// ----- Ioctls ----------------------------------------------------------------------------------

/// One `Ioctl` request laid out word by word as libnx's `nvIoctl` lays it out, and its reply:
/// nvmap's `IocCreate` of 0x1001 bytes, the argument in a map-alias buffer, the struct back in
/// one.
#[test]
fn the_ioctl_message_fixture() {
    let mut rig = Rig::new();
    let fd = rig.open_device(Device::Nvmap);
    assert_eq!(fd, 1);
    rig.kernel
        .map(IN_AT, vec![0x01, 0x10, 0, 0, 0x77, 0x77, 0x77, 0x77]);
    rig.kernel.map(OUT_AT, vec![STALE; 8]);
    let words: [u32; 23] = [
        // Request (4), one X, one A, one B descriptor.
        0x0111_0004,
        // 11 raw words (libnx: 16 + 16 + 8 bytes, one out-pointer size, rounded up), receive
        // list flag 3 (one C descriptor).
        0x0000_0C0B,
        // X: empty.
        0,
        0,
        // A: 8 bytes at IN_AT.
        8,
        0x8000_0000,
        0,
        // B: 8 bytes at OUT_AT.
        8,
        0x9000_0000,
        0,
        // Raw data: two words of padding to 16 bytes, `SFCI`, version 0, command 1, token 0.
        0,
        0,
        u32::from_le_bytes(*b"SFCI"),
        0,
        1,
        0,
        // fd 1, ioctl 0xC0080101.
        1,
        0xC008_0101,
        // libnx's reserve, then the out-pointer size table: one empty C descriptor.
        0,
        0,
        0,
        // C: empty.
        0,
        0,
    ];
    let mut buffer = buffer_of(&words);
    // The request builder lays it out the same way.
    let built = Auto::alias(OUT_AT, 8)
        .receive(Auto::alias(IN_AT, 8).send(cmif(IOCTL, &fd_and(fd, NVMAP_CREATE)), 0));
    let mut from_builder = vec![0; BUFFER_SIZE];
    built.build(&mut from_builder).unwrap();
    assert_eq!(from_builder, buffer);
    let handled = rig.server.handle(&mut rig.kernel, rig.session, &mut buffer);
    assert_eq!(handled, Ok(Handled::Replied));
    // Reply: raw size 7, padding, `SFCO`, version, result, token, `Success`.
    assert_eq!(buffer, buffer_of(&[0, 7, 0, 0, SFCO_MAGIC, 0, 0, 0, 0]));
    // The struct back: the size as sent, the first handle id.
    assert_eq!(rig.memory(OUT_AT, 8), [0x01, 0x10, 0, 0, 4, 0, 0, 0]);
    let handle = rig.state().nvmap().handle(4).unwrap();
    assert_eq!(handle.created_size(), 0x1001);
}

/// nvmap through the server, with every buffer in its pointer descriptor: create, allocate, ask
/// the kind, free.
#[test]
fn nvmap_through_pointer_buffers() {
    let mut rig = Rig::new();
    let fd = rig.open_device(Device::Nvmap);
    let call = |rig: &mut Rig, raw: u32, input: &[u8]| {
        let input_at = rig.place(input);
        let out_at = rig.place(&vec![STALE; input.len()]);
        let request = cmif(IOCTL, &fd_and(fd, raw));
        let request = Auto::pointer(input_at, len16(input.len())).send(request, 0);
        let request = Auto::pointer(out_at, len16(input.len())).receive(request);
        let reply = answer(&rig.send(&request), 1);
        (reply, rig.memory(out_at, input.len()))
    };
    let success = ok(&[0]);
    assert_eq!(
        call(&mut rig, NVMAP_CREATE, &le(&[0x2000, 0])),
        (success.clone(), le(&[0x2000, 4]))
    );
    let alloc = le(&[4, 0, 1, 0x1000, 0x2B, 0, 0x5000_0000, 0x51]);
    assert_eq!(
        call(&mut rig, NVMAP_ALLOC, &alloc),
        (success.clone(), alloc)
    );
    let kind = le(&[4, nvmap::param::KIND, 0]);
    assert_eq!(
        call(&mut rig, NVMAP_PARAM, &kind),
        (success.clone(), le(&[4, nvmap::param::KIND, 0x2B]))
    );
    let free = le(&[4, 0, 0, 0, 0, 0]);
    let freed = le(&[4, 0, 0x5000_0000, 0x51, 0x2000, nvmap::MAP_UNCACHED]);
    assert_eq!(call(&mut rig, NVMAP_FREE, &free), (success, freed));
    assert!(rig.state().nvmap().is_empty());
}

/// An ioctl the device does not know: `NotImplemented` and nothing written under strict;
/// `Success` and the declared size zeroed under auto-stub, clipped to the buffer, and nothing
/// written when the ioctl has no out direction.
#[test]
fn an_unknown_ioctl_under_each_policy() {
    let declared = usize::try_from(Ioctl::from_raw(CTRL_GPU_CHARACTERISTICS).size()).unwrap();
    assert_eq!(declared, 0xB0);
    let input = vec![0x3C; declared];

    let mut rig = Rig::with(UnknownPolicy::Strict);
    let fd = rig.open_device(Device::NvhostCtrlGpu);
    let (reply, output) = rig.ioctl(fd, CTRL_GPU_CHARACTERISTICS, &input, 0x100);
    assert_eq!(reply, ok(&[NvResult::NotImplemented.raw()]));
    assert_eq!(output, vec![STALE; 0x100]);

    let mut rig = Rig::with(UnknownPolicy::AutoStub);
    let fd = rig.open_device(Device::NvhostCtrlGpu);
    // A buffer larger than the declared size: zeros up to it, the rest untouched.
    let (reply, output) = rig.ioctl(fd, CTRL_GPU_CHARACTERISTICS, &input, 0x100);
    assert_eq!(reply, ok(&[NvResult::Success.raw()]));
    let mut expected = vec![0; declared];
    expected.resize(0x100, STALE);
    assert_eq!(output, expected);
    // A smaller one: zeros to its end.
    let (reply, output) = rig.ioctl(fd, CTRL_GPU_CHARACTERISTICS, &input, 0x10);
    assert_eq!(reply, ok(&[NvResult::Success.raw()]));
    assert_eq!(output, vec![0; 0x10]);
    // In only: nothing written.
    let in_only = CTRL_GPU_CHARACTERISTICS & !(1 << ioctl::OUT_BIT);
    let (reply, output) = rig.ioctl(fd, in_only, &input, 0x100);
    assert_eq!(reply, ok(&[NvResult::Success.raw()]));
    assert_eq!(output, vec![STALE; 0x100]);
}

/// nvmap's `IocCreate` number is known to nvmap's `Ioctl` alone: no other device has an ioctl in
/// nvmap's group, and nvmap has none on `Ioctl2` or `Ioctl3`.
#[test]
fn only_nvmap_s_ioctl_1_commands_are_known() {
    let mut state = NvsrvState::new();
    for device in DEVICES {
        open_direct(&mut state, device);
    }
    for (index, device) in DEVICES.into_iter().enumerate() {
        let fd = i32::try_from(index).unwrap() + 1;
        for variant in [
            IoctlVariant::Ioctl,
            IoctlVariant::Ioctl2,
            IoctlVariant::Ioctl3,
        ] {
            let answer = direct_ioctl(
                &mut state,
                UnknownPolicy::Strict,
                variant,
                fd,
                NVMAP_CREATE,
                &le(&[0x1000, 0]),
            );
            let known = device == Device::Nvmap && variant == IoctlVariant::Ioctl;
            let expected = if known {
                NvResult::Success
            } else {
                NvResult::NotImplemented
            };
            assert_eq!(answer.result, expected, "{device:?} {variant:?}");
        }
    }
    assert_eq!(state.nvmap().len(), 1);
    assert_eq!(
        [
            IoctlVariant::Ioctl,
            IoctlVariant::Ioctl2,
            IoctlVariant::Ioctl3
        ]
        .map(IoctlVariant::word),
        ["Ioctl", "Ioctl2", "Ioctl3"]
    );
}

/// A bad fd is answered before the device or the policy: under auto-stub too, and nothing is
/// written.
#[test]
fn fd_checks_come_before_the_policy() {
    for policy in [UnknownPolicy::Strict, UnknownPolicy::AutoStub] {
        let mut rig = Rig::with(policy);
        let fd = rig.open_device(Device::NvhostCtrlGpu);
        let input = vec![0x3C; 0xB0];
        let cases = [
            (-1, NvResult::InvalidState),
            (fd + 1, NvResult::NotImplemented),
        ];
        for (asked, result) in cases {
            let (reply, output) = rig.ioctl(asked, CTRL_GPU_CHARACTERISTICS, &input, 0x100);
            assert_eq!(reply, ok(&[result.raw()]), "{asked} {policy:?}");
            assert_eq!(output, vec![STALE; 0x100], "{asked} {policy:?}");
        }
    }
}

/// What a device gives back is written as far as it goes and no further, and clipped to a
/// smaller buffer; nothing when the ioctl has no out direction.
#[test]
fn an_out_buffer_gets_only_the_answer() {
    let mut rig = Rig::new();
    let fd = rig.open_device(Device::Nvmap);
    let (reply, output) = rig.ioctl(fd, NVMAP_CREATE, &le(&[0x1000, 0]), 0x20);
    assert_eq!(reply, ok(&[0]));
    let mut expected = le(&[0x1000, 4]);
    expected.resize(0x20, STALE);
    assert_eq!(output, expected);
    let (reply, output) = rig.ioctl(fd, NVMAP_CREATE, &le(&[0x1000, 0]), 6);
    assert_eq!(reply, ok(&[0]));
    assert_eq!(output, le(&[0x1000, 8])[..6]);
    // In only: the handle is made, but nothing comes back.
    let in_only = NVMAP_CREATE & !(1 << ioctl::OUT_BIT);
    let (reply, output) = rig.ioctl(fd, in_only, &le(&[0x1000, 0]), 8);
    assert_eq!(reply, ok(&[0]));
    assert_eq!(output, vec![STALE; 8]);
    assert_eq!(rig.state().nvmap().len(), 3);
    // An empty out buffer is fine.
    let (reply, output) = rig.ioctl(fd, NVMAP_CREATE, &le(&[0x1000, 0]), 0);
    assert_eq!((reply, output), (ok(&[0]), Vec::new()));
}

/// The device's struct is written back whatever the result (Eden `WrapGeneric` and
/// `NVDRV::Ioctl1` write the output after any device answer): `IocAlloc` on a handle that does
/// not exist answers `BadValue`, and out buffer 0 still gets the struct with its alignment raised
/// to a page, every other field as sent.
#[test]
fn a_failed_ioctl_still_writes_its_struct_back() {
    let mut rig = Rig::new();
    let fd = rig.open_device(Device::Nvmap);
    let missing = HANDLE_ID_INCREMENT;
    let page = u32::try_from(starquake_memory::PAGE_SIZE).unwrap();
    let sent = le(&[missing, 0x11, 0x22, 0, 0x33, 0x44, 0x6000, 0x55]);
    let (reply, output) = rig.ioctl(fd, NVMAP_ALLOC, &sent, sent.len());
    assert_eq!(reply, ok(&[NvResult::BadValue.raw()]));
    assert_eq!(
        output,
        le(&[missing, 0x11, 0x22, page, 0x33, 0x44, 0x6000, 0x55])
    );
    assert!(rig.state().nvmap().is_empty());
}

/// Only [`ioctl::MAX_ARGUMENT_BYTES`] bytes of an in buffer are read: a buffer whose tail is not
/// mapped still works, and the argument at its start is found.
///
/// This pins the upper edge only: reading one byte more fails here, but reading one byte fewer
/// would pass, since no device of this slice reads anywhere near that far into its argument
/// (nvmap's largest struct is 32 bytes). The lower edge gets a test when a device whose argument
/// can be that long arrives.
#[test]
fn an_in_buffer_is_read_up_to_the_largest_argument() {
    let mut rig = Rig::new();
    let fd = rig.open_device(Device::Nvmap);
    let limit = usize::try_from(ioctl::MAX_ARGUMENT_BYTES).unwrap();
    let mut input = le(&[0x3000, 0]);
    input.resize(limit, 0xEE);
    let input_at = rig.place(&input);
    let out_at = rig.place(&[STALE; 8]);
    let request = cmif(IOCTL, &fd_and(fd, NVMAP_CREATE));
    // The descriptor says a page more than is mapped.
    let request = Auto::alias(input_at, len16(limit) + 0x1000).send(request, 0);
    let request = Auto::alias(out_at, 8).receive(request);
    assert_eq!(answer(&rig.send(&request), 1), ok(&[0]));
    assert_eq!(rig.memory(out_at, 8), le(&[0x3000, 4]));
}

/// `Ioctl2` or `Ioctl3` (`command`) of nvmap's `IocCreate` on `fd`, with the inline buffer in
/// memory that is not mapped: the reply and out buffer 0 afterwards.
fn inline(rig: &mut Rig, command: u32, fd: i32) -> (Answer, Vec<u8>) {
    let input_at = rig.place(&le(&[0x1000, 0]));
    let out_at = rig.place(&[STALE; 8]);
    let request = cmif(command, &fd_and(fd, NVMAP_CREATE));
    let mut request = Auto::alias(input_at, 8).send(request, 0);
    if command == IOCTL2 {
        request = Auto::alias(INLINE_AT, 0x40).send(request, 1);
    }
    request = Auto::alias(out_at, 8).receive(request);
    if command == IOCTL3 {
        request = Auto::alias(INLINE_OUT_AT, 0x40).receive(request);
    }
    let reply = answer(&rig.send(&request), 1);
    (reply, rig.memory(out_at, 8))
}

/// `Ioctl2`'s inline input is not read and `Ioctl3`'s inline output not written: both are in
/// memory that is not mapped, and the ioctls still answer, by the policy, since nvmap has neither.
#[test]
fn inline_buffers_are_left_alone() {
    for command in [IOCTL2, IOCTL3] {
        let mut rig = Rig::with(UnknownPolicy::AutoStub);
        let fd = rig.open_device(Device::Nvmap);
        // Auto-stubbed: zeros in out buffer 0, and no handle made.
        assert_eq!(
            inline(&mut rig, command, fd),
            (ok(&[0]), vec![0; 8]),
            "{command}"
        );
        assert!(rig.state().nvmap().is_empty());
        let mut rig = Rig::new();
        let fd = rig.open_device(Device::Nvmap);
        let not_implemented = ok(&[NvResult::NotImplemented.raw()]);
        assert_eq!(
            inline(&mut rig, command, fd),
            (not_implemented, vec![STALE; 8]),
            "{command}"
        );
    }
}

/// nvmap's allocate and free do not lock or unlock the caller's memory for the device (Eden's
/// do; `nvmap` module docs, "Where this differs from Eden").
#[test]
fn alloc_and_free_take_no_device_lock() {
    let mut rig = Rig::new();
    let fd = rig.open_device(Device::Nvmap);
    rig.nvmap(fd, NVMAP_CREATE, &le(&[0x1000, 0]));
    let address = 0x5000_0000;
    rig.nvmap(fd, NVMAP_ALLOC, &le(&[4, 0, 0, 0x1000, 0, 0, address, 0]));
    assert_eq!(rig.kernel.device_locks(u64::from(address), 0x1000), 0);
    rig.nvmap(fd, NVMAP_FREE, &le(&[4, 0, 0, 0, 0, 0]));
    assert_eq!(rig.kernel.device_locks(u64::from(address), 0x1000), 0);
}

// ----- Recording and snapshots -----------------------------------------------------------------

/// Every request is one IPC event of the run's stream, with nvsrv's number, the command and its
/// result.
#[test]
fn each_request_is_recorded() {
    let mut rig = Rig::new();
    let fd = rig.open_device(Device::Nvmap);
    rig.ioctl(fd, NVMAP_CREATE, &le(&[0x1000, 0]), 8);
    rig.send(&cmif(CLOSE, &fd.to_le_bytes()));
    rig.send(&cmif(MAP_SHARED_MEM, &[]));
    let recorded: Vec<_> = rig
        .kernel
        .recorded()
        .into_iter()
        .filter_map(|event| match event.kind {
            EventKind::Ipc {
                service,
                command,
                result,
                outcome,
                ..
            } => Some((service, command, result.0, outcome)),
            _ => None,
        })
        .collect();
    let done = |command| (NVSRV, command, ResultCode::SUCCESS.raw(), IpcOutcome::Done);
    assert_eq!(
        recorded,
        [
            done(OPEN),
            done(IOCTL),
            done(CLOSE),
            (
                NVSRV,
                MAP_SHARED_MEM,
                UNKNOWN_COMMAND_ID.raw(),
                IpcOutcome::Done
            ),
        ]
    );
}

/// An ioctl auto-stubbed under [`UnknownPolicy::AutoStub`] is recorded as a done request, not a
/// stubbed one, and is not in the server's record of stubbed commands, however often it is sent
/// (module docs, "Where this differs from the references"). An unknown command on the same rig
/// is recorded as stubbed, so the rig can tell the two apart.
#[test]
fn an_auto_stubbed_ioctl_is_recorded_as_done() {
    let mut rig = Rig::with(UnknownPolicy::AutoStub);
    let fd = rig.open_device(Device::NvhostCtrlGpu);
    let input = vec![0x3C; 0xB0];
    for _ in 0..2 {
        let (reply, _) = rig.ioctl(fd, CTRL_GPU_CHARACTERISTICS, &input, 0x100);
        assert_eq!(reply, ok(&[NvResult::Success.raw()]));
    }
    assert_eq!(rig.server.stubbed().count(), 0);
    rig.send(&cmif(MAP_SHARED_MEM, &[]));
    let recorded: Vec<_> = rig
        .kernel
        .recorded()
        .into_iter()
        .filter_map(|event| match event.kind {
            EventKind::Ipc {
                command,
                result,
                outcome,
                ..
            } => Some((command, result.0, outcome)),
            _ => None,
        })
        .collect();
    let success = ResultCode::SUCCESS.raw();
    assert_eq!(
        recorded,
        [
            (OPEN, success, IpcOutcome::Done),
            (IOCTL, success, IpcOutcome::Done),
            (IOCTL, success, IpcOutcome::Done),
            (MAP_SHARED_MEM, success, IpcOutcome::Stubbed),
        ]
    );
    let stubbed: Vec<_> = rig.server.stubbed().map(|stub| stub.id).collect();
    assert_eq!(stubbed, [MAP_SHARED_MEM]);
}

/// A server saved with files open and a handle allocated restores to the same state, and the
/// restored server carries on from it: the handle answers through the saved fd, and the next fd
/// and handle id follow the saved ones.
#[test]
fn a_saved_server_restores_and_carries_on() {
    let mut rig = Rig::new();
    let nvmap = rig.open_device(Device::Nvmap);
    let gpu = rig.open_device(Device::NvhostGpu);
    rig.send(&cmif(CLOSE, &gpu.to_le_bytes()));
    rig.nvmap(nvmap, NVMAP_CREATE, &le(&[0x1800, 0]));
    rig.nvmap(
        nvmap,
        NVMAP_ALLOC,
        &le(&[4, 0, 0, 0x2000, 0x19, 0, 0x6000, 0]),
    );
    let state = rig.state().clone();
    let saved = rig.server.save().unwrap();

    let (shared, front) = wiring();
    let mut restored =
        nvsrv_server(shared, as_front(&front), UnknownPolicy::Strict).with_service_id(NVSRV);
    restored.restore(&saved).unwrap();
    assert_eq!(state_of(&restored), Some(&state));
    assert_eq!(restored.save().unwrap(), saved);
    rig.server = restored;

    let kind = rig.nvmap(nvmap, NVMAP_PARAM, &le(&[4, nvmap::param::KIND, 0]));
    assert_eq!(kind, le(&[4, nvmap::param::KIND, 0x19]));
    assert_eq!(rig.state().file(gpu), Err(NvResult::NotImplemented));
    assert_eq!(rig.open_device(Device::NvhostAsGpu), gpu + 1);
    let created = rig.nvmap(nvmap, NVMAP_CREATE, &le(&[0x1000, 0]));
    assert_eq!(created, le(&[0x1000, 8]));
}

/// The saved layout of the state, as `postcard` writes it: the file count, each fd and its
/// device's index, the next fd, the nvmap table (its own layout is pinned in `nvmap/tests.rs`),
/// then nvhost-as-gpu's views, the layouts of the address spaces it made and its bound channels,
/// then nvhost-tsg-gpu's groups, then nvhost-gpu's channels. Every device appears, each fd
/// distinct; the state is reached by ioctls, each field of a view, a layout and a group a
/// distinct non-zero value. The bindings and the channels are empty here (`Direct` has no channel
/// front, so nothing binds or makes a channel): a binding and a channel with values are pinned by
/// `gpu/tests.rs`, `a_bound_channel_s_saved_layout_is_pinned`, and a channel's classes and
/// events by `a_channel_s_saved_layout_is_pinned`.
#[test]
fn the_saved_state_s_layout_is_pinned() {
    let mut direct = Direct::new();
    for device in DEVICES {
        direct.open(device);
    }
    // One address space first, so the view's is 2.
    direct.memory.create_address_space(&direct.front).unwrap();
    // fd 3 is nvhost-as-gpu: start 0x2000, end 0x9000, split 0x5000, big pages 0x10000.
    let alloc = le(&[0, 0, 0x1_0000, 0, 0x2000, 0, 0x9000, 0, 0x5000, 0]);
    assert_eq!(
        direct.ioctl(3, AS_ALLOC_AS_EX, &alloc).result,
        NvResult::Success
    );
    // fd 4 is nvhost-tsg-gpu: a timeslice of 0x77, channel fd 5 (nvhost-gpu) at subcontext 0x21.
    let timeslice = le(&[0x77]);
    let set = direct.ioctl(4, tsg_gpu::SET_TIMESLICE, &timeslice);
    assert_eq!(set.result, NvResult::Success);
    let bind = le(&[5, 0x21, 0, 0, 0, 0]);
    let bound = direct.ioctl(4, tsg_gpu::BIND_CHANNEL_EX, &bind);
    assert_eq!(bound.result, NvResult::Success);
    let state = direct.state;

    let mut expected = vec![10];
    for (fd, index) in (1..=10).zip(0..10) {
        expected.extend([fd, index]);
    }
    // The next fd; then the nvmap table: no handles, next id 4.
    expected.extend([0x0B, 0, 4]);
    // A layout: start 0x2000, split 0x5000, end 0x9000, big page size 0x10000, as varints.
    let layout = [
        0x80, 0x40, 0x80, 0xA0, 0x01, 0x80, 0xA0, 0x02, 0x80, 0x80, 0x04,
    ];
    // One view: fd 3, address space 2, its layout.
    expected.extend([1, 3, 2]);
    expected.extend(layout);
    // One address space made: 2, its layout.
    expected.extend([1, 2]);
    expected.extend(layout);
    // No bound channel.
    expected.push(0);
    // One group: fd 4, timeslice 0x77, one channel: fd 5, subcontext 0x21.
    expected.extend([1, 4, 0x77, 1, 5, 0x21]);
    // No nvhost-gpu channel.
    expected.push(0);
    // One ctrl-gpu fd, 2, with its events: index 0x1_0004 (a varint), generation 0x77; index
    // 0x1_0005, generation 0x77 (`made_up_events`).
    expected.extend([1, 2, 0x84, 0x80, 0x04, 0x77, 0x85, 0x80, 0x04, 0x77]);
    let bytes = state.save_state().unwrap();
    assert_eq!(bytes, expected);
    let mut loaded = NvsrvState::new();
    loaded.load_state(&bytes).unwrap();
    assert_eq!(loaded, state);
}

/// Each saved state no sequence of commands reaches, and the error it is refused with.
fn unreachable_states() -> Vec<(NvsrvState, NvsrvStateError)> {
    let valid = full_table(3);
    let with = |change: &dyn Fn(&mut NvsrvState)| {
        let mut state = valid.clone();
        change(&mut state);
        state
    };
    vec![
        (
            with(&|state| state.next_fd = 0),
            NvsrvStateError::NextFd { next: 0 },
        ),
        (
            with(&|state| state.next_fd = FD_END + 1),
            NvsrvStateError::NextFd { next: FD_END + 1 },
        ),
        (
            full_table(MAX_OPEN_FILES + 1),
            NvsrvStateError::TooManyFiles {
                count: MAX_OPEN_FILES + 1,
            },
        ),
        (
            with(&|state| {
                state.files.insert(0, Device::Nvmap);
            }),
            NvsrvStateError::NotIssued { fd: 0 },
        ),
        (
            with(&|state| {
                state.files.insert(4, Device::Nvmap);
            }),
            NvsrvStateError::NotIssued { fd: 4 },
        ),
        (
            // Two faults: the next fd is reported before the files.
            with(&|state| {
                state.next_fd = 0;
                state.files.insert(9, Device::Nvmap);
            }),
            NvsrvStateError::NextFd { next: 0 },
        ),
        (
            // Two faults: the count is reported before a file.
            {
                let mut state = full_table(MAX_OPEN_FILES + 1);
                state.files.insert(0, Device::Nvmap);
                state
            },
            NvsrvStateError::TooManyFiles {
                count: MAX_OPEN_FILES + 2,
            },
        ),
        (
            // ctrl-gpu events on an open nvmap.
            with(&|state| state.ctrl_gpu.keep(2, made_up_events(2)).unwrap()),
            NvsrvStateError::NotACtrlGpu { fd: 2 },
        ),
        (
            // ctrl-gpu events on an fd never handed out.
            with(&|state| state.ctrl_gpu.keep(9, made_up_events(9)).unwrap()),
            NvsrvStateError::NotACtrlGpu { fd: 9 },
        ),
        (
            // An open ctrl-gpu without its events.
            with(&|state| {
                state.files.insert(3, Device::NvhostCtrlGpu);
            }),
            NvsrvStateError::CtrlGpuWithoutEvents { fd: 3 },
        ),
        (
            // Two faults: events on a file that is not a ctrl-gpu are reported before a ctrl-gpu
            // without events.
            with(&|state| {
                state.files.insert(3, Device::NvhostCtrlGpu);
                state.ctrl_gpu.keep(2, made_up_events(2)).unwrap();
            }),
            NvsrvStateError::NotACtrlGpu { fd: 2 },
        ),
        (
            // Two faults: the files are checked before the ctrl-gpu events.
            with(&|state| {
                state.files.insert(4, Device::Nvmap);
                state.ctrl_gpu.keep(2, made_up_events(2)).unwrap();
            }),
            NvsrvStateError::NotIssued { fd: 4 },
        ),
    ]
}

/// A ctrl-gpu fd with its events, the state `Open` leaves, restores; so does one whose fd was
/// closed (its events gone with it). The ctrl-gpu errors read as plain words.
#[test]
fn a_saved_ctrl_gpu_with_its_events_restores() {
    let mut state = full_table(3);
    state.files.insert(2, Device::NvhostCtrlGpu);
    state.ctrl_gpu.keep(2, made_up_events(2)).unwrap();
    assert_eq!(NvsrvState::try_from(saved(&state)), Ok(state.clone()));
    let bytes = state.save_state().unwrap();
    let mut loaded = NvsrvState::new();
    loaded.load_state(&bytes).unwrap();
    assert_eq!(loaded, state);
    // Closed: no file and no events.
    state.files.remove(&2);
    state.ctrl_gpu.close(2);
    assert_eq!(NvsrvState::try_from(saved(&state)), Ok(state));
    assert_eq!(
        NvsrvStateError::NotACtrlGpu { fd: 2 }.to_string(),
        "fd 2 has ctrl-gpu events but is not an open nvhost-ctrl-gpu"
    );
    assert_eq!(
        NvsrvStateError::CtrlGpuWithoutEvents { fd: 3 }.to_string(),
        "nvhost-ctrl-gpu fd 3 is open without its events"
    );
}

// ----- Opening ctrl-gpu with its events --------------------------------------------------------

/// The index the fake kernel gives the next object it mints, minting one to see it.
fn next_index(rig: &mut Rig) -> u32 {
    rig.kernel.new_object().index()
}

/// Through the server, `Open` of nvhost-ctrl-gpu makes two kernel events and keeps them as the
/// fd's; another device, and a path `Open` refuses, makes none.
#[test]
fn opening_ctrl_gpu_makes_its_two_events() {
    let mut rig = Rig::new();
    let before = next_index(&mut rig);
    let fd = rig.open_device(Device::NvhostCtrlGpu);
    let events = rig.state().ctrl_gpu().events(fd.cast_unsigned()).unwrap();
    assert_ne!(events.error_notifier, events.unknown);
    for event in [events.error_notifier, events.unknown] {
        // Events the kernel made (and nothing has signalled).
        assert_eq!(rig.kernel.signals(event), Some(0), "{event:?}");
    }
    // Exactly two objects were made.
    assert_eq!(next_index(&mut rig), before + 3);
    // nvmap, the profiler and a path no device has make none.
    let before = next_index(&mut rig);
    rig.open_device(Device::Nvmap);
    rig.open(PROFILER_PATH.as_bytes());
    rig.open(b"/dev/nvhost-ctrl-gpu-not");
    assert_eq!(next_index(&mut rig), before + 1);
    assert_eq!(
        rig.state().ctrl_gpu().fds().collect::<Vec<_>>(),
        [fd.cast_unsigned()]
    );
}

/// A full table refuses nvhost-ctrl-gpu before any event is made.
#[test]
fn a_refused_ctrl_gpu_open_makes_no_event() {
    let mut rig = Rig::with_state(UnknownPolicy::Strict, full_table(MAX_OPEN_FILES));
    let before = next_index(&mut rig);
    let refused = ok(&[0xFFFF_FFFF, NvResult::FileOperationFailed.raw()]);
    assert_eq!(rig.open(Device::NvhostCtrlGpu.path().as_bytes()), refused);
    assert_eq!(next_index(&mut rig), before + 1);
    assert_eq!(rig.state().ctrl_gpu().fds().count(), 0);
}

/// A kernel that cannot make an event fails `Open` of nvhost-ctrl-gpu with its error, and no fd
/// is used; nvmap still opens, and once events can be made ctrl-gpu opens with the next fd.
#[test]
fn a_kernel_that_cannot_make_an_event_opens_no_ctrl_gpu() {
    let mut rig = Rig::new();
    rig.kernel.refuse_objects(Some(KernelError::OutOfResource));
    let reply = rig.open(Device::NvhostCtrlGpu.path().as_bytes());
    assert_eq!(reply.result, KernelError::OutOfResource.code().raw());
    assert_eq!(rig.state().open_files(), 0);
    assert_eq!(rig.state().ctrl_gpu().fds().count(), 0);
    assert_eq!(rig.open_device(Device::Nvmap), 1);
    rig.kernel.refuse_objects(None);
    assert_eq!(rig.open_device(Device::NvhostCtrlGpu), 2);
    assert!(rig.state().ctrl_gpu().events(2).is_some());
}

/// A kernel that makes the first event and refuses the second fails `Open` of nvhost-ctrl-gpu
/// with its error and uses no fd; the first event stays in the kernel with no owner, as
/// `ctrl_gpu`'s "Where this differs from the references" lists. Once events can be made again,
/// ctrl-gpu opens with fd 1 and two fresh events.
#[test]
fn a_ctrl_gpu_open_whose_second_event_fails_leaves_the_first() {
    let mut rig = Rig::new();
    let before = rig.kernel.new_object();
    rig.kernel
        .refuse_objects_after(1, Some(KernelError::OutOfResource));
    let reply = rig.open(Device::NvhostCtrlGpu.path().as_bytes());
    assert_eq!(reply.result, KernelError::OutOfResource.code().raw());
    assert_eq!(rig.state().open_files(), 0);
    assert_eq!(rig.state().ctrl_gpu().fds().count(), 0);
    // One event was made: the index after `before`, a kernel event no fd holds.
    let leaked = ObjectId::from_parts(before.index() + 1, before.generation());
    assert_eq!(rig.kernel.signals(leaked), Some(0));
    assert_eq!(next_index(&mut rig), before.index() + 2);
    rig.kernel.refuse_objects(None);
    assert_eq!(rig.open_device(Device::NvhostCtrlGpu), 1);
    let events = rig.state().ctrl_gpu().events(1).unwrap();
    assert!(![events.error_notifier, events.unknown].contains(&leaked));
}

/// [`NvsrvState::open`] keeps events only for nvhost-ctrl-gpu: without them, or with a pair
/// [`ctrl_gpu::CtrlGpu::check`] refuses, it opens nothing and uses no fd; given for another
/// device they are not kept and the device opens.
#[test]
fn open_takes_events_only_for_ctrl_gpu() {
    let path = Device::NvhostCtrlGpu.path().as_bytes();
    let refused = (INVALID_FD, NvResult::FileOperationFailed);
    let mut state = NvsrvState::new();
    assert_eq!(state.open(path, None), refused);
    let first = made_up_events(1);
    let same = ctrl_gpu::CtrlGpuEvents {
        error_notifier: first.unknown,
        unknown: first.unknown,
    };
    assert_eq!(state.open(path, Some(same)), refused);
    assert_eq!((state.open_files(), state.next_fd), (0, FIRST_FD));
    assert_eq!(state.open(path, Some(first)), (1, NvResult::Success));
    // A pair that shares one of fd 1's events.
    let sharing = ctrl_gpu::CtrlGpuEvents {
        error_notifier: made_up_events(2).error_notifier,
        unknown: first.unknown,
    };
    assert_eq!(state.open(path, Some(sharing)), refused);
    assert_eq!((state.open_files(), state.next_fd), (1, 2));
    // nvmap given events: opened, the events not kept.
    let nvmap = Device::Nvmap.path().as_bytes();
    assert_eq!(
        state.open(nvmap, Some(made_up_events(2))),
        (2, NvResult::Success)
    );
    assert_eq!(state.ctrl_gpu().fds().collect::<Vec<_>>(), [1]);
    assert_eq!(state.ctrl_gpu().events(1), Some(first));
}

/// `opens_ctrl_gpu` is true exactly when `Open` would open an nvhost-ctrl-gpu: its path, with or
/// without bytes after a zero, and a table with room; not for another path, nor a full table.
#[test]
fn opens_ctrl_gpu_is_open_s_own_answer() {
    let state = NvsrvState::new();
    assert!(state.opens_ctrl_gpu(b"/dev/nvhost-ctrl-gpu"));
    assert!(state.opens_ctrl_gpu(b"/dev/nvhost-ctrl-gpu\0junk"));
    for path in [
        &b"/dev/nvhost-ctrl-gpuX"[..],
        b"/dev/nvhost-ctrl",
        b"/dev/nvmap",
        PROFILER_PATH.as_bytes(),
        b"",
    ] {
        assert!(!state.opens_ctrl_gpu(path), "{path:?}");
    }
    let full = full_table(MAX_OPEN_FILES);
    assert!(!full.opens_ctrl_gpu(b"/dev/nvhost-ctrl-gpu"));
    let room = full_table(MAX_OPEN_FILES - 1);
    assert!(room.opens_ctrl_gpu(b"/dev/nvhost-ctrl-gpu"));
    let spent = NvsrvState {
        next_fd: FD_END,
        ..NvsrvState::new()
    };
    assert!(!spent.opens_ctrl_gpu(b"/dev/nvhost-ctrl-gpu"));
}

/// The saved form of `state`, field for field.
pub(super) fn saved(state: &NvsrvState) -> SavedNvsrvState {
    SavedNvsrvState {
        files: state.files.clone(),
        next_fd: state.next_fd,
        nvmap: state.nvmap.clone(),
        as_gpu: state.as_gpu.clone(),
        tsg_gpu: state.tsg_gpu.clone(),
        gpu: state.gpu.clone(),
        ctrl_gpu: state.ctrl_gpu.clone(),
    }
}

#[test]
fn a_saved_state_no_command_reaches_is_refused() {
    for (state, error) in unreachable_states() {
        assert_eq!(NvsrvState::try_from(saved(&state)), Err(error), "{error}");
        // Through a whole server snapshot: refused, and the server is left as it was.
        let bytes = server_with(state).save().unwrap();
        let mut restored = server_with(NvsrvState::new());
        let refused = SnapshotError::Decode(postcard::Error::SerdeDeCustom);
        assert_eq!(restored.restore(&bytes), Err(refused), "{error}");
        assert_eq!(state_of(&restored), Some(&NvsrvState::new()));
    }
}

/// A state whose nvmap table no ioctl reaches is refused by the table's own check, through a
/// whole server snapshot.
#[test]
fn a_saved_state_with_a_bad_table_is_refused() {
    let mut state = NvsrvState::new();
    state.open(b"/dev/nvmap", None);
    direct_ioctl(
        &mut state,
        UnknownPolicy::Strict,
        IoctlVariant::Ioctl,
        1,
        NVMAP_CREATE,
        &le(&[0x1000, 0]),
    );
    let mut bytes = state.save_state().unwrap();
    // The table's last byte is its next id, 8, before six empty maps (as-gpu's views, made
    // spaces and bindings, tsg-gpu's groups, nvhost-gpu's channels, ctrl-gpu's events): make it
    // 2, which no table reaches.
    let at = bytes.len() - 7;
    assert_eq!(bytes[at..], [8, 0, 0, 0, 0, 0, 0]);
    bytes[at] = 2;
    let mut loaded = NvsrvState::new();
    assert_eq!(
        loaded.load_state(&bytes),
        Err(SnapshotError::Decode(postcard::Error::SerdeDeCustom))
    );
    assert_eq!(loaded, NvsrvState::new());
    assert_eq!(
        NvMapStateError::NextId { next: 2 }.to_string(),
        "the next nvmap handle id 0x2 is not one handle ids can reach"
    );
}

/// Each relation's edge, on the side that is reachable: accepted as it is.
#[test]
fn a_saved_state_at_each_edge_is_accepted() {
    let cases = [
        NvsrvState::new(),
        full_table(1),
        full_table(MAX_OPEN_FILES),
        NvsrvState {
            next_fd: FD_END,
            ..NvsrvState::new()
        },
        NvsrvState {
            files: BTreeMap::from([(FD_END - 1, Device::NvhostVic)]),
            next_fd: FD_END,
            ..NvsrvState::new()
        },
    ];
    for state in cases {
        assert_eq!(NvsrvState::try_from(saved(&state)).as_ref(), Ok(&state));
        let bytes = server_with(state.clone()).save().unwrap();
        let mut restored = server_with(NvsrvState::new());
        restored.restore(&bytes).unwrap();
        assert_eq!(state_of(&restored), Some(&state));
    }
}

#[test]
fn a_refused_state_is_said_in_plain_words() {
    let said = [
        (
            NvsrvStateError::NextFd { next: 0 },
            "the next fd 0 is not one fds can reach",
        ),
        (
            NvsrvStateError::TooManyFiles { count: 7 },
            "7 files are open, more than can be",
        ),
        (
            NvsrvStateError::NotIssued { fd: 4 },
            "fd 4 is open but was never handed out",
        ),
    ];
    for (error, words) in said {
        assert_eq!(error.to_string(), words);
    }
}

#[test]
fn a_new_state_is_empty() {
    let state = NvsrvState::default();
    assert_eq!(state.open_files(), 0);
    assert!(state.nvmap().is_empty());
    assert_eq!(state, NvsrvState::new());
}
