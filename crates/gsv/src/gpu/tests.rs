//! The gpu session and its worker driven through raw TIPC messages, over the services
//! framework's fake kernel, a fake GPU front, and gsv's shared state and drain.
//!
//! Requests are laid out by hand, word by word, from the layouts PR checks
//! (`eden-pr4588/src/core/hle/service/am/service/gpu_session.cpp` and
//! `gpu_memory_registration.h` @ e5fb3b97): each handler there compares the first two words with
//! literals (`NormalMapGpuMemory4K`: `packet[0] == 0xe3 && packet[1] == 16`), which the tests
//! write as they are, next to the command id they encode.

use std::collections::BTreeMap;
use std::sync::Mutex;

use starquake_gpu::interface::{Completions, GpuCompletion, RegionId, Target, Ticket, VmUpdate};
use starquake_gpu::syncpoint::Syncpoints;
use starquake_gpu::vm::{GpuVm, Mapping};
use starquake_ipc::{BUFFER_SIZE, Body, Message, SessionKind, TIPC_COMMAND_OFFSET};
use starquake_kernel::dispatch::DeviceShareError;
use starquake_kernel::id::SlabId;
use starquake_kernel::memory::DeviceShareToken;
use starquake_kernel::memory::testing::device_share_token;
use starquake_kernel::result::ResultCode;
use starquake_memory::{PAGE_SIZE, PageRange};
use starquake_services::framework::fake::FakeKernel;
use starquake_services::framework::result::INVALID_CMIF_REQUEST;
use starquake_services::framework::{Handled, Kernel};
use starquake_trace::{EventKind, ServiceId};

use super::command::*;
use super::kind::{GENERIC_16BX2, S8Z24, Z16};
use super::memory::{
    DEVICE_SLOTS, MAX_ADDRESS_SPACES, MAX_PENDING_JOBS, MAX_PIECES, SELF_MAP_KIND,
    SMALL_PAGE_SHIFT, SPARSE_KIND,
};
use super::*;
use crate::completion::{Completion, CompletionQueue};
use crate::shared::TimerHost;

/// The number the tests' gpu server is known by: not the default, so a hard-coded one fails.
const GPU: ServiceId = ServiceId(0x0749);
/// What fills a message buffer before a request is laid into it.
const STALE: u8 = 0xA5;
/// A registration's CPU address in the tests: inside NVN's window, so a 211 may map at it too.
const MEMORY: u64 = 0x1_0000_0000;
/// The registration's size: four pages.
const MEMORY_SIZE: u64 = 4 * PAGE_SIZE;
/// Where the tests' 211s map: another place in the window.
const GPU_VA: u64 = 0x2_0000_0000;

// ----- The fake GPU front ----------------------------------------------------------------------

/// One call the front was given.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FrontCall {
    Create(AsId),
    Update(AsId, VmUpdate, Option<Ticket>),
    Barrier(Ticket),
}

/// A GPU front that records its calls and, when a test says the GPU has caught up
/// ([`FakeFront::finish`]), applies the queued updates in order and reports the barriers.
#[derive(Default)]
struct FakeFront {
    calls: Mutex<Vec<FrontCall>>,
    queued: Mutex<Vec<FrontCall>>,
    tables: Mutex<BTreeMap<AsId, Arc<GpuVm>>>,
}

impl GpuVmFront for FakeFront {
    fn create_address_space(&self, space: AsId, vm: Arc<GpuVm>) {
        self.calls.lock().unwrap().push(FrontCall::Create(space));
        self.tables.lock().unwrap().insert(space, vm);
    }

    fn update_vm(&self, space: AsId, update: VmUpdate, done: Option<Ticket>) {
        let call = FrontCall::Update(space, update, done);
        self.calls.lock().unwrap().push(call);
        self.queued.lock().unwrap().push(call);
    }

    fn barrier(&self, done: Ticket) {
        self.calls.lock().unwrap().push(FrontCall::Barrier(done));
        self.queued.lock().unwrap().push(FrontCall::Barrier(done));
    }
}

impl FakeFront {
    fn calls(&self) -> Vec<FrontCall> {
        self.calls.lock().unwrap().clone()
    }

    /// Plays the GPU catching up: applies every queued update to its table, in order, and
    /// pushes every queued barrier's ticket onto `queue`. Returns the tickets.
    fn finish(&self, queue: &CompletionQueue) -> Vec<Ticket> {
        let queued = std::mem::take(&mut *self.queued.lock().unwrap());
        let mut tickets = Vec::new();
        for call in queued {
            match call {
                FrontCall::Update(space, update, _) => {
                    self.tables.lock().unwrap()[&space].apply(&update);
                }
                FrontCall::Barrier(ticket) => {
                    Completions::push(queue, GpuCompletion::GpuOrdered { ticket });
                    tickets.push(ticket);
                }
                FrontCall::Create(_) => {}
            }
        }
        tickets
    }
}

// ----- The timer thread ------------------------------------------------------------------------

/// What the drain asked of the timer host.
#[derive(Clone, Debug, PartialEq, Eq)]
enum TimerCall {
    Record(EventKind),
    Unlock(DeviceShareToken),
    Signal(ObjectId),
}

/// A timer host over the fake kernel: it unlocks and signals there, and keeps the calls.
struct Timer<'k> {
    kernel: &'k mut FakeKernel,
    calls: Vec<TimerCall>,
}

impl TimerHost for Timer<'_> {
    fn record(&mut self, kind: EventKind) {
        self.calls.push(TimerCall::Record(kind));
    }

    fn unlock_device_share(&mut self, token: DeviceShareToken) -> Result<(), DeviceShareError> {
        self.calls.push(TimerCall::Unlock(token));
        Kernel::unlock_device_share(self.kernel, token).map_err(|_| DeviceShareError::WrongProcess)
    }

    fn signal_event(&mut self, event: ObjectId) -> Result<(), KernelError> {
        self.calls.push(TimerCall::Signal(event));
        Kernel::signal_event(self.kernel, event)
    }
}

/// A timer host that unlocks and keeps the tokens, and refuses every signal.
struct Refusing(Vec<DeviceShareToken>);

impl TimerHost for Refusing {
    fn record(&mut self, _: EventKind) {}

    fn unlock_device_share(&mut self, token: DeviceShareToken) -> Result<(), DeviceShareError> {
        self.0.push(token);
        Ok(())
    }

    fn signal_event(&mut self, _event: ObjectId) -> Result<(), KernelError> {
        Err(KernelError::NotImplemented)
    }
}

// ----- The rig ---------------------------------------------------------------------------------

/// A gpu server over gsv's shared state, a fake front, a kernel, and one gpu session.
struct Rig {
    kernel: FakeKernel,
    server: Server,
    shared: Arc<Shared>,
    front: Arc<FakeFront>,
    session: ObjectId,
}

fn shared_state() -> Arc<Shared> {
    let queue = Arc::new(CompletionQueue::new());
    let completions: Arc<dyn Completions> = Arc::<CompletionQueue>::clone(&queue);
    let syncpoints = Arc::new(Syncpoints::new(completions));
    Arc::new(Shared::new(syncpoints, queue, GPU))
}

impl Rig {
    fn new() -> Self {
        let mut kernel = FakeKernel::new(0).unwrap();
        kernel.set_current_service(GPU);
        let shared = shared_state();
        let front = Arc::new(FakeFront::default());
        let as_front: Arc<dyn GpuVmFront> = Arc::<FakeFront>::clone(&front);
        let mut server =
            gpu_server(Arc::clone(&shared), as_front, UnknownPolicy::Strict).with_service_id(GPU);
        let session = kernel.new_object();
        server.accept(session).unwrap();
        Self {
            kernel,
            server,
            shared,
            front,
            session,
        }
    }

    fn send_on(&mut self, session: ObjectId, words: &[u32]) -> Vec<u8> {
        let mut buffer = vec![STALE; BUFFER_SIZE];
        let bytes = le(words);
        buffer[..bytes.len()].copy_from_slice(&bytes);
        let handled = self.server.handle(&mut self.kernel, session, &mut buffer);
        assert_eq!(handled, Ok(Handled::Replied));
        buffer
    }

    fn send(&mut self, words: &[u32]) -> Vec<u8> {
        self.send_on(self.session, words)
    }

    /// The notification event at `index`, made through gpu 12.
    fn event(&mut self, index: u32) -> ObjectId {
        self.send(&[tipc_type(GET_NOTIFICATION_EVENT), 1, index]);
        self.shared
            .lock()
            .gpu_memory()
            .notification_event(index)
            .unwrap()
    }

    /// A worker session: gpu 12 and 21 on the last index (so slot 0's event starts unsignalled),
    /// then 22.
    fn worker(&mut self) -> ObjectId {
        let last = NOTIFICATION_COUNT - 1;
        self.event(last);
        self.send(&[tipc_type(REQUEST_WORKER), 1, last]);
        self.send(&[tipc_type(ACQUIRE_WORKER), 0]);
        *self.kernel.sessions().last().unwrap()
    }

    /// An address space through gpu 33.
    fn address_space(&mut self) -> u32 {
        let reply = self.send(&[tipc_type(CREATE_ADDRESS_SPACE), 1, 0]);
        data(&reply)[0]
    }

    /// A graphics device through gpu 41.
    fn device(&mut self) -> u32 {
        let reply = self.send(&create_device_words(0x11, 0, 0x22));
        data(&reply)[0]
    }

    /// Maps `size` bytes at `address`, marks them GPU-accessible, and registers them through
    /// 101 with `device`; the handle.
    fn register(&mut self, worker: ObjectId, address: u64, size: u64, device: u32) -> u32 {
        self.kernel
            .map(address, vec![0; usize::try_from(size).unwrap()]);
        self.kernel.mark_gpu_accessible(address, size);
        let reply = self.send_on(
            worker,
            &register_words(address, address + size - 1, 0, device),
        );
        assert_eq!(result_of(&reply), ResultCode::SUCCESS, "101");
        data(&reply)[0]
    }

    fn gpu(&self) -> crate::shared::Locked<'_> {
        self.shared.lock()
    }

    fn mappings(&self, space: u32) -> Vec<Mapping> {
        self.gpu().gpu_memory().mappings(AsId(space))
    }

    /// Plays the GPU catching up and the timer thread draining; the drain's calls.
    fn finish(&mut self) -> Vec<TimerCall> {
        self.apply();
        self.drain()
    }

    /// Plays the GPU catching up only: its updates applied, its barriers queued, nothing
    /// drained yet.
    fn apply(&self) {
        self.front.finish(self.shared.queue());
    }

    /// Plays the timer thread draining; the drain's calls.
    fn drain(&mut self) -> Vec<TimerCall> {
        let mut timer = Timer {
            kernel: &mut self.kernel,
            calls: Vec::new(),
        };
        self.shared.drain(&mut timer);
        timer.calls
    }

    /// What the table of `space` shows at the page at `va` ([`shown_at`]).
    fn at(&self, space: u32, va: u64) -> Option<(Target, Kind)> {
        shown_at(&self.mappings(space), va)
    }
}

/// What `mappings` show at the page at `va`: the target moved to that page, and the kind; so
/// a test reads what a page translates to, however the table split its entries.
fn shown_at(mappings: &[Mapping], va: u64) -> Option<(Target, Kind)> {
    let mapping = mappings
        .iter()
        .find(|m| m.range.start() <= va && va < m.range.end())?;
    let delta = va - mapping.range.start();
    let target = match mapping.target {
        Target::Guest { cpu_va } => Target::Guest {
            cpu_va: cpu_va + delta,
        },
        Target::GpuOnly { region, offset } => Target::GpuOnly {
            region,
            offset: offset + delta,
        },
        Target::SyncpointShim => Target::SyncpointShim,
    };
    Some((target, mapping.kind))
}

/// A guest page at `cpu_va` with `kind`, as [`shown_at`] reads it.
fn page_at(cpu_va: u64, kind: Kind) -> Option<(Target, Kind)> {
    shown_at(
        &[guest(page_range(cpu_va, PAGE_SIZE), cpu_va, kind)],
        cpu_va,
    )
}

/// The usual set-up: a worker, a device, an address space, a four-page registration at
/// [`MEMORY`], and the slot-0 event.
struct Ready {
    rig: Rig,
    worker: ObjectId,
    device: u32,
    space: u32,
    handle: u32,
    event: ObjectId,
}

fn ready() -> Ready {
    let mut rig = Rig::new();
    let worker = rig.worker();
    let device = rig.device();
    let space = rig.address_space();
    let handle = rig.register(worker, MEMORY, MEMORY_SIZE, device);
    let event = rig.event(0);
    Ready {
        rig,
        worker,
        device,
        space,
        handle,
        event,
    }
}

// ----- Words -----------------------------------------------------------------------------------

fn le(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

fn tipc_type(id: u32) -> u32 {
    u32::from(TIPC_COMMAND_OFFSET) + id
}

fn halves(value: u64) -> [u32; 2] {
    let bytes = value.to_le_bytes();
    let word = |at: usize| u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
    [word(0), word(4)]
}

/// The reply's result word and its data words after it.
fn reply_of(buffer: &[u8]) -> (ResultCode, Vec<u32>) {
    let message = Message::parse_reply(buffer, SessionKind::Session).unwrap();
    let Body::TipcReply { result, data } = message.body else {
        unreachable!("not a TIPC reply: {:?}", message.body);
    };
    let (chunks, rest) = data.as_chunks::<4>();
    assert!(rest.is_empty(), "a reply's data is whole words");
    let words = chunks.iter().map(|c| u32::from_le_bytes(*c)).collect();
    (ResultCode::from_raw(result), words)
}

/// The reply's result word.
fn result_of(buffer: &[u8]) -> ResultCode {
    reply_of(buffer).0
}

/// The reply's data words after the result.
fn data(buffer: &[u8]) -> Vec<u32> {
    reply_of(buffer).1
}

fn code(error: KernelError) -> ResultCode {
    error.code()
}

/// 41 (`0x39, 6`): name 2–3, flags 4–5, label 6–7.
fn create_device_words(name: u64, flags: u64, label: u64) -> Vec<u32> {
    assert_eq!(tipc_type(CREATE_GRAPHICS_DEVICE), 0x39);
    let mut words = vec![0x39, 6];
    words.extend(halves(name));
    words.extend(halves(flags));
    words.extend(halves(label));
    words
}

/// 101 (`0x75, 6`, PR `DecodeGpuMemoryRegistration`): first 2–3, last 4–5, flags 6, device 7.
fn register_words(first: u64, last: u64, flags: u32, device: u32) -> Vec<u32> {
    assert_eq!(tipc_type(CREATE_GPU_ACCESSIBLE_MEMORY), 0x75);
    let mut words = vec![0x75, 6];
    words.extend(halves(first));
    words.extend(halves(last));
    words.extend([flags, device]);
    assert_eq!(words.len(), 8);
    words
}

/// 211's one-byte fields, words 10 to 16.
#[derive(Clone, Copy)]
struct MapBytes {
    permission: u32,
    cache: u32,
    kind: u32,
    fallback: u32,
    mode: u32,
    device: u32,
    sync: u32,
}

impl MapBytes {
    /// An accepted set: no permission, cached, generic kind 6 with fallback 6.
    fn of(device: u32) -> Self {
        Self {
            permission: 0,
            cache: 1,
            kind: 6,
            fallback: 6,
            mode: 0,
            device,
            sync: 1,
        }
    }
}

/// 211 (`0xe3, 16`): space 2, first 3–4, last 5–6, handle 7, offset 8–9, bytes 10–16, slot 17.
fn map_words(
    space: u32,
    first: u64,
    last: u64,
    handle: u32,
    offset: u64,
    b: MapBytes,
    slot: u32,
) -> Vec<u32> {
    assert_eq!(tipc_type(NORMAL_MAP_GPU_MEMORY_4K), 0xE3);
    let mut words = vec![0xE3, 16, space];
    words.extend(halves(first));
    words.extend(halves(last));
    words.push(handle);
    words.extend(halves(offset));
    words.extend([
        b.permission,
        b.cache,
        b.kind,
        b.fallback,
        b.mode,
        b.device,
        b.sync,
        slot,
    ]);
    assert_eq!(words.len(), 18);
    words
}

/// 212 (`0xe4, 9`): space 2, first 3–4, last 5–6, kind 7, fallback 8, sync 9, slot 10.
fn modify_words(
    space: u32,
    first: u64,
    last: u64,
    kind: u32,
    fallback: u32,
    sync: u32,
    slot: u32,
) -> Vec<u32> {
    assert_eq!(tipc_type(NORMAL_MODIFY_MEMORY_KIND_4K), 0xE4);
    let mut words = vec![0xE4, 9, space];
    words.extend(halves(first));
    words.extend(halves(last));
    words.extend([kind, fallback, sync, slot]);
    assert_eq!(words.len(), 11);
    words
}

/// 213 (`0xe5, 7`) or 215 (`0xe7, 7`): space 2, first 3–4, last 5–6, sync 7, slot 8.
fn range_words(command: u32, space: u32, first: u64, last: u64, sync: u32, slot: u32) -> Vec<u32> {
    let mut words = vec![tipc_type(command), 7, space];
    words.extend(halves(first));
    words.extend(halves(last));
    words.extend([sync, slot]);
    assert_eq!(words.len(), 9);
    words
}

/// 214 (`0xe6, 10`, PR `DecodeGpuSparseRegion`): space 2, first 3–4, last 5–6, page shift 7,
/// mode 8, device 9, sync 10, slot 11.
#[expect(
    clippy::too_many_arguments,
    reason = "one argument per request word, in the order the layout gives them"
)]
fn sparse_words(
    space: u32,
    first: u64,
    last: u64,
    shift: u32,
    mode: u32,
    device: u32,
    sync: u32,
    slot: u32,
) -> Vec<u32> {
    assert_eq!(tipc_type(CREATE_SPARSE_REGION), 0xE6);
    let mut words = vec![0xE6, 10, space];
    words.extend(halves(first));
    words.extend(halves(last));
    words.extend([shift, mode, device, sync, slot]);
    assert_eq!(words.len(), 12);
    words
}

fn page_range(start: u64, size: u64) -> PageRange {
    PageRange::new(start, size).unwrap()
}

fn guest(range: PageRange, cpu_va: u64, kind: Kind) -> Mapping {
    Mapping {
        range,
        target: Target::Guest { cpu_va },
        kind,
    }
}

impl Ready {
    /// A 211 of the whole registration at `va`; the reply.
    fn map_at(&mut self, va: u64) -> Vec<u8> {
        let words = map_words(
            self.space,
            va,
            va + MEMORY_SIZE - 1,
            self.handle,
            0,
            MapBytes::of(self.device),
            0,
        );
        self.rig.send_on(self.worker, &words)
    }
}

/// The tests' registration's own mapping.
fn own() -> Mapping {
    guest(page_range(MEMORY, MEMORY_SIZE), MEMORY, SELF_MAP_KIND)
}

use starquake_gpu::interface::Kind;

// ----- The gpu session -------------------------------------------------------------------------

#[test]
fn the_notification_range_is_prs() {
    let mut rig = Rig::new();
    let reply = rig.send(&[tipc_type(GET_NOTIFICATION_RANGE), 0]);
    assert_eq!(result_of(&reply), ResultCode::SUCCESS);
    assert_eq!(data(&reply), [0, NOTIFICATION_COUNT - 1]);
    assert_eq!(NOTIFICATION_COUNT - 1, 63, "the spec's choice of PR's 63");
}

/// gpu 12 makes an event the first time and copies the same one after; the last index is
/// accepted and the one past it refused.
#[test]
fn notification_events_are_made_once_and_copied() {
    let mut rig = Rig::new();
    let last = NOTIFICATION_COUNT - 1;
    let first = rig.send(&[tipc_type(GET_NOTIFICATION_EVENT), 1, last]);
    let message = Message::parse_reply(&first, SessionKind::Session).unwrap();
    assert_eq!(message.copy_handles.len(), 1);
    assert_eq!(message.move_handles, Vec::<u32>::new());
    let event = rig.gpu().gpu_memory().notification_event(last).unwrap();
    rig.send(&[tipc_type(GET_NOTIFICATION_EVENT), 1, last]);
    assert_eq!(rig.gpu().gpu_memory().notification_event(last), Some(event));
    let past = rig.send(&[tipc_type(GET_NOTIFICATION_EVENT), 1, NOTIFICATION_COUNT]);
    assert_eq!(result_of(&past), code(KernelError::InvalidCombination));
}

/// gpu 21 needs the index's event (else `InvalidState`) and signals it; 22 before any 21 is
/// `InvalidState`.
#[test]
fn the_worker_is_requested_then_acquired() {
    let mut rig = Rig::new();
    let early = rig.send(&[tipc_type(ACQUIRE_WORKER), 0]);
    assert_eq!(result_of(&early), code(KernelError::InvalidState));
    let no_event = rig.send(&[tipc_type(REQUEST_WORKER), 1, 3]);
    assert_eq!(result_of(&no_event), code(KernelError::InvalidState));
    let past = rig.send(&[tipc_type(REQUEST_WORKER), 1, NOTIFICATION_COUNT]);
    assert_eq!(result_of(&past), code(KernelError::InvalidCombination));
    let event = rig.event(3);
    let requested = rig.send(&[tipc_type(REQUEST_WORKER), 1, 3]);
    assert_eq!(result_of(&requested), ResultCode::SUCCESS);
    assert_eq!(rig.kernel.signals(event), Some(1));
    let acquired = rig.send(&[tipc_type(ACQUIRE_WORKER), 0]);
    assert_eq!(result_of(&acquired), ResultCode::SUCCESS);
    let message = Message::parse_reply(&acquired, SessionKind::Session).unwrap();
    assert_eq!(message.copy_handles.len(), 1);
}

/// Two gpu sessions, and two workers, see one state.
#[test]
fn two_sessions_share_the_state() {
    let mut rig = Rig::new();
    let first = rig.worker();
    rig.send(&[tipc_type(ACQUIRE_WORKER), 0]);
    let second = *rig.kernel.sessions().last().unwrap();
    assert_ne!(first, second);
    let other = rig.kernel.new_object();
    rig.server.accept(other).unwrap();
    let space = data(&rig.send_on(other, &[tipc_type(CREATE_ADDRESS_SPACE), 1, 0]))[0];
    let device = rig.device();
    let handle = rig.register(second, MEMORY, PAGE_SIZE, device);
    // The first worker unmaps nothing of it but sees the registration.
    let slot = NOTIFICATION_COUNT - 1;
    let words = map_words(
        space,
        GPU_VA,
        GPU_VA + PAGE_SIZE - 1,
        handle,
        0,
        MapBytes::of(device),
        slot,
    );
    assert_eq!(result_of(&rig.send_on(first, &words)), ResultCode::SUCCESS);
}

/// gpu 33: the word must be 0; ids count from 1; the front gets each new table; the ninth is
/// `OutOfResource`.
#[test]
fn address_spaces_are_counted_and_bounded() {
    let mut rig = Rig::new();
    let refused = rig.send(&[tipc_type(CREATE_ADDRESS_SPACE), 1, 1]);
    assert_eq!(result_of(&refused), code(KernelError::InvalidCombination));
    let ids: Vec<u32> = (0..MAX_ADDRESS_SPACES)
        .map(|_| rig.address_space())
        .collect();
    let expected: Vec<u32> = (1..=u32::try_from(MAX_ADDRESS_SPACES).unwrap()).collect();
    assert_eq!(ids, expected);
    let creates: Vec<FrontCall> = expected
        .iter()
        .map(|&id| FrontCall::Create(AsId(id)))
        .collect();
    assert_eq!(rig.front.calls(), creates);
    let ninth = rig.send(&[tipc_type(CREATE_ADDRESS_SPACE), 1, 0]);
    assert_eq!(result_of(&ninth), code(KernelError::OutOfResource));
}

/// gpu 41 and 42: flags must be 0; ids from 1 to 255; 42 names a device by its low byte.
#[test]
fn graphics_devices_are_prs() {
    let mut rig = Rig::new();
    let flagged = rig.send(&create_device_words(1, 1, 2));
    assert_eq!(result_of(&flagged), code(KernelError::NotImplemented));
    let ids: Vec<u32> = (1..DEVICE_SLOTS).map(|_| rig.device()).collect();
    assert_eq!(ids, (1..DEVICE_SLOTS).collect::<Vec<_>>());
    let full = rig.send(&create_device_words(1, 0, 2));
    assert_eq!(result_of(&full), code(KernelError::OutOfResource));
    // 0x105's low byte is 5.
    let destroyed = rig.send(&[tipc_type(DESTROY_GRAPHICS_DEVICE), 1, 0x105]);
    assert_eq!(result_of(&destroyed), ResultCode::SUCCESS);
    assert_eq!(rig.gpu().gpu_memory().device(5), None);
    let again = rig.send(&[tipc_type(DESTROY_GRAPHICS_DEVICE), 1, 5]);
    assert_eq!(result_of(&again), code(KernelError::NotFound));
    assert_eq!(rig.device(), 5, "the first free id again");
}

#[test]
fn partition_replies_are_nx2_s() {
    let mut rig = Rig::new();
    let size = halves(0xC0_0000_0000 - 0x800_0000);
    let start = halves(0x800_0000);
    let user = rig.send(&[tipc_type(GET_USER_PARTITION), 1, 9]);
    assert_eq!(data(&user), [start[0], start[1], size[0], size[1]]);
    let gsv = rig.send(&[tipc_type(GET_GSV_PARTITION), 1, 9]);
    assert_eq!(data(&gsv), [0, 0, 0, 0]);
    let range = rig.send(&[tipc_type(GET_VA_RANGE), 1, 9]);
    assert_eq!(data(&range), [1, 0, start[0], start[1], size[0], size[1]]);
}

/// 216 is served (`sparse_map`'s tests drive it): a request with none of its descriptors is the
/// framework's refusal, not the unknown-command policy's.
#[test]
fn sparse_map_requests_are_served() {
    let mut rig = Rig::new();
    let worker = rig.worker();
    let reply = rig.send_on(worker, &[tipc_type(EXECUTE_SPARSE_MAP_REQUESTS), 0]);
    assert_eq!(result_of(&reply), INVALID_CMIF_REQUEST);
}

#[test]
fn a_short_request_is_refused() {
    let mut rig = Rig::new();
    let worker = rig.worker();
    // 213 with its slot word missing.
    let mut words = range_words(NORMAL_UNMAP_GPU_MEMORY_4K, 1, GPU_VA, GPU_VA, 0, 0);
    words[1] = 6;
    words.pop();
    assert_eq!(
        result_of(&rig.send_on(worker, &words)),
        INVALID_CMIF_REQUEST
    );
}

// ----- 101 -------------------------------------------------------------------------------------

/// 101 locks the range, maps it at its own address in every address space (one made before,
/// one after), and replies with the handle.
#[test]
fn a_registration_is_locked_and_mapped_in_every_space() {
    let mut s = ready();
    assert_eq!(s.rig.kernel.device_locks(MEMORY, MEMORY_SIZE), 1);
    let later = s.rig.address_space();
    assert_eq!(s.rig.mappings(s.space), [own()]);
    assert_eq!(s.rig.mappings(later), [own()]);
    let registration = *s.rig.gpu().gpu_memory().registration(s.handle).unwrap();
    assert_eq!(registration.range, page_range(MEMORY, MEMORY_SIZE));
    assert_eq!(registration.device, u8::try_from(s.device).unwrap());
}

/// 101's checks in order, each at its edge, and with two faults at once the first answers.
#[test]
fn registration_checks_answer_in_prs_order() {
    let mut rig = Rig::new();
    let worker = rig.worker();
    let device = rig.device();
    let page = MEMORY;
    rig.kernel
        .map(page, vec![0; usize::try_from(2 * PAGE_SIZE).unwrap()]);
    rig.kernel.mark_gpu_accessible(page, PAGE_SIZE);
    let mut ask = |words: Vec<u32>| result_of(&rig.send_on(worker, &words));
    let last = page + PAGE_SIZE - 1;
    // Flags (and a bad device too): NotImplemented first.
    assert_eq!(
        ask(register_words(page, last, 1, 0)),
        code(KernelError::NotImplemented)
    );
    // A flags word whose low byte is 0 passes: one-byte field.
    assert_eq!(
        ask(register_words(page, last, 0x100, 0)),
        code(KernelError::NotFound)
    );
    // Device, then a range ending at the last address.
    assert_eq!(
        ask(register_words(page, u64::MAX, 0, 0)),
        code(KernelError::NotFound)
    );
    assert_eq!(
        ask(register_words(page, u64::MAX, 0, device)),
        code(KernelError::InvalidSize)
    );
    assert_eq!(
        ask(register_words(page, page - 1, 0, device)),
        code(KernelError::InvalidSize)
    );
    // First not aligned (and size not aligned): InvalidAddress first.
    assert_eq!(
        ask(register_words(page + 1, last + 2, 0, device)),
        code(KernelError::InvalidAddress)
    );
    assert_eq!(
        ask(register_words(page, last + 1, 0, device)),
        code(KernelError::InvalidSize)
    );
    // The second page is mapped but not GPU-accessible.
    let two = page + 2 * PAGE_SIZE - 1;
    assert_eq!(
        ask(register_words(page, two, 0, device)),
        code(KernelError::InvalidCurrentMemory)
    );
    assert_eq!(rig.kernel.device_locks(page, 2 * PAGE_SIZE), 0);
    assert_eq!(
        result_of(&rig.send_on(worker, &register_words(page, last, 0, device))),
        ResultCode::SUCCESS
    );
}

// ----- 211 -------------------------------------------------------------------------------------

/// 211 maps at once, replies with the selected kind, and signals its event.
#[test]
fn a_map_is_published_and_signalled_at_once() {
    let mut s = ready();
    let mut bytes = MapBytes::of(s.device);
    // PR: compressible S8Z24 (0xc) with fallback 3 selects 3, backend S8Z24.
    bytes.kind = 0xC;
    bytes.fallback = 3;
    let words = map_words(
        s.space,
        GPU_VA,
        GPU_VA + PAGE_SIZE - 1,
        s.handle,
        PAGE_SIZE,
        bytes,
        0,
    );
    let reply = s.rig.send_on(s.worker, &words);
    assert_eq!(result_of(&reply), ResultCode::SUCCESS);
    assert_eq!(data(&reply), [3]);
    assert_eq!(s.rig.kernel.signals(s.event), Some(1));
    let mapped = guest(page_range(GPU_VA, PAGE_SIZE), MEMORY + PAGE_SIZE, S8Z24);
    assert_eq!(s.rig.mappings(s.space), [own(), mapped]);
}

/// A refused 211 (after its slot) signals its event and replies with the result alone; one with
/// no event at its slot signals nothing.
#[test]
fn a_refused_map_signals_and_answers_the_result_only() {
    let mut s = ready();
    let words = map_words(
        s.space,
        GPU_VA,
        GPU_VA,
        s.handle,
        0,
        MapBytes::of(s.device),
        0,
    );
    let reply = s.rig.send_on(s.worker, &words);
    assert_eq!(result_of(&reply), code(KernelError::InvalidCurrentMemory));
    assert_eq!(data(&reply), Vec::<u32>::new());
    assert_eq!(s.rig.kernel.signals(s.event), Some(1));
    let no_slot = map_words(
        s.space,
        GPU_VA,
        GPU_VA,
        s.handle,
        0,
        MapBytes::of(s.device),
        5,
    );
    let reply = s.rig.send_on(s.worker, &no_slot);
    assert_eq!(result_of(&reply), code(KernelError::InvalidCombination));
    assert_eq!(s.rig.kernel.signals(s.event), Some(1));
}

/// 211's checks in PR's order: each pair of faults answers the first.
#[test]
fn map_checks_answer_in_prs_order() {
    let mut s = ready();
    let last = GPU_VA + PAGE_SIZE - 1;
    let ok = MapBytes::of(s.device);
    let (space, handle) = (s.space, s.handle);
    let mut ask = |space: u32, first: u64, last: u64, handle: u32, offset: u64, b: MapBytes| {
        result_of(&s.rig.send_on(
            s.worker,
            &map_words(space, first, last, handle, offset, b, 0),
        ))
    };
    let mut no_device = ok;
    no_device.device = 9;
    let mut permission = ok;
    permission.permission = 1;
    let mut cache = ok;
    cache.cache = 2;
    let mut mode = ok;
    mode.mode = 1;
    let mut sync = ok;
    sync.sync = 2;
    let mut kind = ok;
    kind.fallback = 7;
    // Address space before handle before device.
    assert_eq!(
        ask(9, GPU_VA, last, 9, 0, no_device),
        code(KernelError::NotFound)
    );
    assert_eq!(
        ask(space, GPU_VA, last, 9, 0, no_device),
        code(KernelError::NotFound)
    );
    assert_eq!(
        ask(space, GPU_VA, last, handle, 0, no_device),
        code(KernelError::NotFound)
    );
    // Each byte, before a bad range.
    for bytes in [permission, cache, mode, sync, kind] {
        assert_eq!(
            ask(space, GPU_VA, GPU_VA - 1, handle, 0, bytes),
            code(KernelError::NotImplemented)
        );
    }
    // A one-byte field takes the low byte only (PR masks words 10 to 16).
    let mut padded = ok;
    padded.permission = 0x100;
    padded.kind = 0x206;
    assert_eq!(
        ask(space, GPU_VA, last, handle, 0, padded),
        ResultCode::SUCCESS
    );
    // The range, before alignment.
    assert_eq!(
        ask(space, GPU_VA + 1, GPU_VA, handle, 0, ok),
        code(KernelError::InvalidSize)
    );
    assert_eq!(
        ask(space, GPU_VA, u64::MAX, handle, 0, ok),
        code(KernelError::InvalidSize)
    );
    let refused = code(KernelError::InvalidCurrentMemory);
    assert_eq!(ask(space, GPU_VA + 1, last, handle, 0, ok), refused);
    assert_eq!(ask(space, GPU_VA, last + 1, handle, 0, ok), refused);
    assert_eq!(ask(space, GPU_VA, last, handle, 1, ok), refused);
}

/// The offset and size inside the registration, and NVN's window, each at both sides of its
/// edge.
#[test]
fn map_bounds_are_checked_at_their_edges() {
    let mut s = ready();
    let ok = MapBytes::of(s.device);
    let (space, handle) = (s.space, s.handle);
    let mut ask = |first: u64, size: u64, offset: u64| {
        let words = map_words(space, first, first + size - 1, handle, offset, ok, 0);
        result_of(&s.rig.send_on(s.worker, &words))
    };
    let refused = code(KernelError::InvalidCurrentMemory);
    // The last page of the registration, and one page past it.
    assert_eq!(
        ask(GPU_VA, PAGE_SIZE, MEMORY_SIZE - PAGE_SIZE),
        ResultCode::SUCCESS
    );
    assert_eq!(ask(GPU_VA, PAGE_SIZE, MEMORY_SIZE), refused);
    assert_eq!(ask(GPU_VA, MEMORY_SIZE, 0), ResultCode::SUCCESS);
    assert_eq!(ask(GPU_VA, MEMORY_SIZE + PAGE_SIZE, 0), refused);
    // An offset so large that offset + size wraps.
    assert_eq!(ask(GPU_VA, PAGE_SIZE, u64::MAX - PAGE_SIZE + 1), refused);
    // The window's first page and the page before it; its last page and the page after.
    assert_eq!(ask(VA_WINDOW_START, PAGE_SIZE, 0), ResultCode::SUCCESS);
    assert_eq!(ask(VA_WINDOW_START - PAGE_SIZE, PAGE_SIZE, 0), refused);
    assert_eq!(
        ask(VA_WINDOW_END - PAGE_SIZE, PAGE_SIZE, 0),
        ResultCode::SUCCESS
    );
    assert_eq!(ask(VA_WINDOW_END, PAGE_SIZE, 0), refused);
    assert_eq!(ask(VA_WINDOW_END - PAGE_SIZE, 2 * PAGE_SIZE, 0), refused);
}

/// conflict (NX2): a 211 over another cuts it to its parts outside, which keep their place in
/// the registration; the neighbour that starts exactly at the end is left whole.
#[test]
fn a_map_over_a_mapping_cuts_it() {
    let mut s = ready();
    s.map_at(GPU_VA);
    let neighbour = GPU_VA + MEMORY_SIZE;
    s.map_at(neighbour);
    // One page over the second page of the first mapping, from the registration's last page.
    let words = map_words(
        s.space,
        GPU_VA + PAGE_SIZE,
        GPU_VA + 2 * PAGE_SIZE - 1,
        s.handle,
        3 * PAGE_SIZE,
        MapBytes::of(s.device),
        0,
    );
    assert_eq!(
        result_of(&s.rig.send_on(s.worker, &words)),
        ResultCode::SUCCESS
    );
    let kind = GENERIC_16BX2;
    assert_eq!(
        s.rig.mappings(s.space),
        [
            own(),
            guest(page_range(GPU_VA, PAGE_SIZE), MEMORY, kind),
            guest(
                page_range(GPU_VA + PAGE_SIZE, PAGE_SIZE),
                MEMORY + 3 * PAGE_SIZE,
                kind
            ),
            guest(
                page_range(GPU_VA + 2 * PAGE_SIZE, 2 * PAGE_SIZE),
                MEMORY + 2 * PAGE_SIZE,
                kind
            ),
            guest(page_range(neighbour, MEMORY_SIZE), MEMORY, kind),
        ]
    );
    // The cut mapping is three now: its old range no longer matches exactly.
    let words = range_words(
        NORMAL_UNMAP_GPU_MEMORY_4K,
        s.space,
        GPU_VA,
        GPU_VA + MEMORY_SIZE - 1,
        0,
        0,
    );
    assert_eq!(
        result_of(&s.rig.send_on(s.worker, &words)),
        code(KernelError::NotFound)
    );
}

// ----- 213: ordered completion -----------------------------------------------------------------

/// The spec's offline test: an unmap's event fires only after the fake front reports the
/// barrier, and the table loses the mapping only when the GPU applies the update.
#[test]
fn an_unmap_s_event_fires_only_after_the_barrier() {
    let mut s = ready();
    s.map_at(GPU_VA);
    let before = s.rig.kernel.signals(s.event);
    let words = range_words(
        NORMAL_UNMAP_GPU_MEMORY_4K,
        s.space,
        GPU_VA,
        GPU_VA + MEMORY_SIZE - 1,
        1,
        0,
    );
    let reply = s.rig.send_on(s.worker, &words);
    assert_eq!(result_of(&reply), ResultCode::SUCCESS);
    // PR's reply: the bytes unmapped, as a u64.
    assert_eq!(data(&reply), halves(MEMORY_SIZE));
    let range = page_range(GPU_VA, MEMORY_SIZE);
    let calls = s.rig.front.calls();
    assert_eq!(
        calls[calls.len() - 2..],
        [
            FrontCall::Update(AsId(s.space), VmUpdate::Unmap { range }, None),
            FrontCall::Barrier(Ticket(1)),
        ]
    );
    // Not yet: the GPU has not caught up.
    assert_eq!(s.rig.kernel.signals(s.event), before);
    assert_eq!(s.rig.mappings(s.space).len(), 2);
    let empty = Timer {
        kernel: &mut s.rig.kernel,
        calls: Vec::new(),
    };
    let mut timer = empty;
    s.rig.shared.drain(&mut timer);
    assert!(timer.calls.is_empty(), "nothing queued, nothing done");
    let calls = s.rig.finish();
    assert_eq!(calls.last(), Some(&TimerCall::Signal(s.event)));
    assert_eq!(s.rig.kernel.signals(s.event), before.map(|n| n + 1));
    assert_eq!(s.rig.mappings(s.space), [own()]);
}

/// 213 matches exactly one 211 mapping: a page shorter or longer, or a sparse region, is
/// `NotFound`, signalled at once; a sync byte over 1 or no slot event is `InvalidCombination`,
/// signalled not at all.
#[test]
fn unmap_matches_a_mapping_exactly() {
    let mut s = ready();
    s.map_at(GPU_VA);
    let last = GPU_VA + MEMORY_SIZE - 1;
    let space = s.space;
    let mut ask = |first: u64, last: u64, sync: u32, slot: u32| {
        let words = range_words(NORMAL_UNMAP_GPU_MEMORY_4K, space, first, last, sync, slot);
        result_of(&s.rig.send_on(s.worker, &words))
    };
    let signals = |s: &Ready| s.rig.kernel.signals(s.event).unwrap();
    assert_eq!(
        ask(GPU_VA, last, 2, 0),
        code(KernelError::InvalidCombination)
    );
    assert_eq!(
        ask(GPU_VA, last, 0, 7),
        code(KernelError::InvalidCombination)
    );
    assert_eq!(
        ask(GPU_VA, last - PAGE_SIZE, 0, 0),
        code(KernelError::NotFound)
    );
    assert_eq!(
        ask(GPU_VA, last + PAGE_SIZE, 0, 0),
        code(KernelError::NotFound)
    );
    assert_eq!(
        ask(GPU_VA + PAGE_SIZE, last, 0, 0),
        code(KernelError::NotFound)
    );
    let fired = signals(&s);
    // The two InvalidCombination refusals signalled nothing; the three NotFound did, at once
    // (the 211 signalled once before them).
    assert_eq!(fired, 4);
    let region = sparse_words(
        s.space,
        0x81_0000_0000,
        0x81_0000_0000 + PAGE_SIZE - 1,
        12,
        0,
        s.device,
        0,
        0,
    );
    s.rig.send_on(s.worker, &region);
    let words = range_words(
        NORMAL_UNMAP_GPU_MEMORY_4K,
        s.space,
        0x81_0000_0000,
        0x81_0000_0000 + PAGE_SIZE - 1,
        0,
        0,
    );
    assert_eq!(
        result_of(&s.rig.send_on(s.worker, &words)),
        code(KernelError::NotFound)
    );
}

// ----- 102 -------------------------------------------------------------------------------------

/// The spec's offline test: a destroy releases the lock only once the GPU reports the barrier,
/// and releases it before it signals the event.
#[test]
fn a_destroy_releases_the_lock_only_when_the_gpu_is_done() {
    let mut s = ready();
    let token = s
        .rig
        .gpu()
        .gpu_memory()
        .registration(s.handle)
        .unwrap()
        .token;
    let reply = s.rig.send_on(s.worker, &[0x76, 2, s.handle, 0]);
    assert_eq!(result_of(&reply), ResultCode::SUCCESS);
    assert_eq!(s.rig.kernel.device_locks(MEMORY, MEMORY_SIZE), 1);
    assert_eq!(s.rig.kernel.signals(s.event), Some(0));
    assert_eq!(s.rig.mappings(s.space), [own()]);
    let calls = s.rig.finish();
    let acted: Vec<TimerCall> = calls
        .into_iter()
        .filter(|call| !matches!(call, TimerCall::Record(_)))
        .collect();
    assert_eq!(
        acted,
        [TimerCall::Unlock(token), TimerCall::Signal(s.event)]
    );
    assert_eq!(s.rig.kernel.device_locks(MEMORY, MEMORY_SIZE), 0);
    assert_eq!(s.rig.kernel.signals(s.event), Some(1));
    assert_eq!(s.rig.mappings(s.space), Vec::<Mapping>::new());
}

/// conflict (the larger): a 102 takes the registration's 211 mappings out with it, in one job.
#[test]
fn a_destroy_unmaps_the_registration_s_maps() {
    let mut s = ready();
    s.map_at(GPU_VA);
    let reply = s.rig.send_on(s.worker, &[0x76, 2, s.handle, 0]);
    assert_eq!(result_of(&reply), ResultCode::SUCCESS);
    let own = page_range(MEMORY, MEMORY_SIZE);
    let mapped = page_range(GPU_VA, MEMORY_SIZE);
    let space = AsId(s.space);
    let calls = s.rig.front.calls();
    assert_eq!(
        calls[calls.len() - 3..],
        [
            FrontCall::Update(space, VmUpdate::DestroyMemory { range: own }, None),
            FrontCall::Update(space, VmUpdate::Unmap { range: mapped }, None),
            FrontCall::Barrier(Ticket(1)),
        ]
    );
    s.rig.finish();
    assert_eq!(s.rig.mappings(s.space), Vec::<Mapping>::new());
}

#[test]
fn a_second_destroy_is_not_found() {
    let mut s = ready();
    assert_eq!(
        result_of(&s.rig.send_on(s.worker, &[0x76, 2, s.handle, 0])),
        ResultCode::SUCCESS
    );
    let again = s.rig.send_on(s.worker, &[0x76, 2, s.handle, 0]);
    assert_eq!(result_of(&again), code(KernelError::NotFound));
    // No slot: InvalidCombination before NotFound, and nothing signalled.
    let no_slot = s.rig.send_on(s.worker, &[0x76, 2, 99, 9]);
    assert_eq!(result_of(&no_slot), code(KernelError::InvalidCombination));
    assert_eq!(s.rig.kernel.signals(s.event), Some(0));
}

// ----- 212 -------------------------------------------------------------------------------------

/// 212 changes kinds inside one 211 mapping only, replies with the selected kind, and completes
/// in order; a 213 of the whole mapping still matches after it.
#[test]
fn modify_kind_stays_inside_one_mapping() {
    let mut s = ready();
    s.map_at(GPU_VA);
    s.map_at(GPU_VA + MEMORY_SIZE);
    let space = s.space;
    let mut ask = |first: u64, last: u64, kind: u32, fallback: u32| {
        let words = modify_words(space, first, last, kind, fallback, 0, 0);
        let reply = s.rig.send_on(s.worker, &words);
        (result_of(&reply), data(&reply))
    };
    let not_found = (code(KernelError::NotFound), vec![]);
    // Across two mappings, past the end, and a refused kind pair.
    assert_eq!(
        ask(GPU_VA, GPU_VA + MEMORY_SIZE + PAGE_SIZE - 1, 6, 6),
        not_found
    );
    assert_eq!(
        ask(GPU_VA - PAGE_SIZE, GPU_VA + PAGE_SIZE - 1, 6, 6),
        not_found
    );
    assert_eq!(ask(GPU_VA, GPU_VA + PAGE_SIZE - 1, 6, 7), not_found);
    assert_eq!(ask(GPU_VA + 1, GPU_VA + PAGE_SIZE, 6, 6), not_found);
    // The second page: compressible Z16 (0xb) with fallback 1 selects 1.
    let page = GPU_VA + PAGE_SIZE;
    assert_eq!(
        ask(page, page + PAGE_SIZE - 1, 0xB, 1),
        (ResultCode::SUCCESS, vec![1])
    );
    let range = page_range(page, PAGE_SIZE);
    assert_eq!(
        s.rig.front.calls().last(),
        Some(&FrontCall::Barrier(Ticket(1)))
    );
    s.rig.finish();
    let kind = GENERIC_16BX2;
    assert_eq!(
        s.rig.mappings(s.space)[1..4],
        [
            guest(page_range(GPU_VA, PAGE_SIZE), MEMORY, kind),
            guest(range, MEMORY + PAGE_SIZE, Z16),
            guest(
                page_range(page + PAGE_SIZE, 2 * PAGE_SIZE),
                MEMORY + 2 * PAGE_SIZE,
                kind
            ),
        ]
    );
    let words = range_words(
        NORMAL_UNMAP_GPU_MEMORY_4K,
        s.space,
        GPU_VA,
        GPU_VA + MEMORY_SIZE - 1,
        0,
        0,
    );
    assert_eq!(
        result_of(&s.rig.send_on(s.worker, &words)),
        ResultCode::SUCCESS
    );
}

// ----- 214 and 215 -----------------------------------------------------------------------------

const REGION: u64 = 0x81_0000_0000;

/// 214's checks in PR's order, at their edges.
#[test]
fn sparse_region_checks_answer_in_prs_order() {
    let mut s = ready();
    let (space, device) = (s.space, s.device);
    let mut ask =
        |space: u32, first: u64, last: u64, shift: u32, mode: u32, device: u32, sync: u32| {
            result_of(&s.rig.send_on(
                s.worker,
                &sparse_words(space, first, last, shift, mode, device, sync, 0),
            ))
        };
    let last = REGION + PAGE_SIZE - 1;
    assert_eq!(
        ask(9, REGION, last, 13, 0, 9, 0),
        code(KernelError::NotFound)
    );
    assert_eq!(
        ask(space, REGION, last, 13, 0, 9, 0),
        code(KernelError::NotFound)
    );
    assert_eq!(
        ask(space, REGION, last, 13, 2, device, 0),
        code(KernelError::InvalidSize)
    );
    assert_eq!(
        ask(space, REGION, REGION - 1, 12, 2, device, 0),
        code(KernelError::InvalidSize)
    );
    assert_eq!(
        ask(space, REGION, u64::MAX, 12, 2, device, 0),
        code(KernelError::InvalidSize)
    );
    assert_eq!(
        ask(
            space,
            REGION + PAGE_SIZE,
            REGION + 0x1_0000 + PAGE_SIZE - 1,
            16,
            0,
            device,
            0
        ),
        code(KernelError::InvalidSize)
    );
    // 64 KiB pages decode but are not implemented; so are a mode or sync over 1.
    assert_eq!(
        ask(space, REGION, REGION + 0x1_0000 - 1, 16, 0, device, 0),
        code(KernelError::NotImplemented)
    );
    assert_eq!(
        ask(space, REGION, last, 12, 2, device, 0),
        code(KernelError::NotImplemented)
    );
    assert_eq!(
        ask(space, REGION, last, 12, 1, device, 2),
        code(KernelError::NotImplemented)
    );
}

/// 214's window and overlap checks, each at both sides of its edge.
#[test]
fn sparse_region_bounds_are_checked_at_their_edges() {
    let mut s = ready();
    let (space, device) = (s.space, s.device);
    let mut ask = |space: u32, first: u64, last: u64| {
        result_of(&s.rig.send_on(
            s.worker,
            &sparse_words(space, first, last, 12, 0, device, 0, 0),
        ))
    };
    let last = REGION + PAGE_SIZE - 1;
    let refused = code(KernelError::InvalidCurrentMemory);
    assert_eq!(
        ask(space, VA_WINDOW_START - PAGE_SIZE, VA_WINDOW_START - 1),
        refused
    );
    assert_eq!(
        ask(space, VA_WINDOW_END, VA_WINDOW_END + PAGE_SIZE - 1),
        refused
    );
    assert_eq!(
        ask(space, VA_WINDOW_END - PAGE_SIZE, VA_WINDOW_END - 1),
        ResultCode::SUCCESS
    );
    assert_eq!(ask(space, REGION, last), ResultCode::SUCCESS);
    // Overlapping it by one page, and the neighbour exactly at its end.
    assert_eq!(ask(space, REGION, last + PAGE_SIZE), refused);
    assert_eq!(
        ask(space, REGION + PAGE_SIZE, last + PAGE_SIZE),
        ResultCode::SUCCESS
    );
}

/// A sparse region shows GPU-only pages; a 211 inside it takes its pages over, and once its
/// 213 is done the region's pages show again (not a hole).
#[test]
fn an_unmap_inside_a_sparse_region_shows_the_region_again() {
    let mut s = ready();
    let size = 8 * PAGE_SIZE;
    let words = sparse_words(s.space, REGION, REGION + size - 1, 12, 0, s.device, 0, 0);
    assert_eq!(
        result_of(&s.rig.send_on(s.worker, &words)),
        ResultCode::SUCCESS
    );
    let sparse = |start: u64, length: u64| Mapping {
        range: page_range(start, length),
        target: Target::GpuOnly {
            region: RegionId(1),
            offset: start - REGION,
        },
        kind: SPARSE_KIND,
    };
    assert_eq!(s.rig.mappings(s.space), [own(), sparse(REGION, size)]);
    let inside = REGION + 2 * PAGE_SIZE;
    s.map_at(inside);
    let mapped = guest(page_range(inside, MEMORY_SIZE), MEMORY, GENERIC_16BX2);
    assert_eq!(
        s.rig.mappings(s.space),
        [
            own(),
            sparse(REGION, 2 * PAGE_SIZE),
            mapped,
            sparse(inside + MEMORY_SIZE, 2 * PAGE_SIZE)
        ]
    );
    let words = range_words(
        NORMAL_UNMAP_GPU_MEMORY_4K,
        s.space,
        inside,
        inside + MEMORY_SIZE - 1,
        0,
        0,
    );
    assert_eq!(
        result_of(&s.rig.send_on(s.worker, &words)),
        ResultCode::SUCCESS
    );
    s.rig.finish();
    assert_eq!(
        s.rig.mappings(s.space),
        [
            own(),
            sparse(REGION, 2 * PAGE_SIZE),
            sparse(inside, MEMORY_SIZE),
            sparse(inside + MEMORY_SIZE, 2 * PAGE_SIZE)
        ]
    );
}

/// A sparse region over a registration's own range lies under it, and shows once the
/// registration is destroyed.
#[test]
fn a_sparse_region_lies_under_a_registration() {
    let mut s = ready();
    let words = sparse_words(
        s.space,
        MEMORY,
        MEMORY + MEMORY_SIZE - 1,
        12,
        0,
        s.device,
        0,
        0,
    );
    assert_eq!(
        result_of(&s.rig.send_on(s.worker, &words)),
        ResultCode::SUCCESS
    );
    assert_eq!(s.rig.mappings(s.space), [own()]);
    s.rig.send_on(s.worker, &[0x76, 2, s.handle, 0]);
    s.rig.finish();
    let region = Mapping {
        range: page_range(MEMORY, MEMORY_SIZE),
        target: Target::GpuOnly {
            region: RegionId(1),
            offset: 0,
        },
        kind: SPARSE_KIND,
    };
    assert_eq!(s.rig.mappings(s.space), [region]);
}

/// 215 matches a region exactly, completes in order, and leaves the 211 mappings inside in the
/// bookkeeping (a divergence from NX2, which trims them). Its destroy covers the whole region,
/// so between the GPU applying it and the drain the mapping inside is gone from the table too;
/// the drain shows it again.
#[test]
fn a_sparse_region_is_destroyed_in_order() {
    let mut s = ready();
    let size = 8 * PAGE_SIZE;
    s.rig.send_on(
        s.worker,
        &sparse_words(s.space, REGION, REGION + size - 1, 12, 0, s.device, 0, 0),
    );
    s.map_at(REGION);
    let short = range_words(
        DESTROY_SPARSE_REGION,
        s.space,
        REGION,
        REGION + size - PAGE_SIZE - 1,
        0,
        0,
    );
    assert_eq!(
        result_of(&s.rig.send_on(s.worker, &short)),
        code(KernelError::NotFound)
    );
    let signals = s.rig.kernel.signals(s.event).unwrap();
    let words = range_words(
        DESTROY_SPARSE_REGION,
        s.space,
        REGION,
        REGION + size - 1,
        0,
        0,
    );
    assert_eq!(
        result_of(&s.rig.send_on(s.worker, &words)),
        ResultCode::SUCCESS
    );
    let range = page_range(REGION, size);
    assert_eq!(
        s.rig.front.calls()[s.rig.front.calls().len() - 2],
        FrontCall::Update(
            AsId(s.space),
            VmUpdate::DestroySparseRegion {
                range,
                region: RegionId(1)
            },
            None
        )
    );
    assert_eq!(s.rig.kernel.signals(s.event), Some(signals));
    let mapped = guest(page_range(REGION, MEMORY_SIZE), MEMORY, GENERIC_16BX2);
    // Before the GPU applies it, the table is as it was.
    let rest = Mapping {
        range: page_range(REGION + MEMORY_SIZE, size - MEMORY_SIZE),
        target: Target::GpuOnly {
            region: RegionId(1),
            offset: MEMORY_SIZE,
        },
        kind: SPARSE_KIND,
    };
    assert_eq!(s.rig.mappings(s.space), [own(), mapped, rest]);
    // Applied, not drained: every mapping in the region's range is gone, the 211 one included.
    s.rig.apply();
    assert_eq!(s.rig.mappings(s.space), [own()]);
    s.rig.drain();
    assert_eq!(s.rig.kernel.signals(s.event), Some(signals + 1));
    assert_eq!(s.rig.mappings(s.space), [own(), mapped]);
    assert!(
        s.rig.gpu().gpu_memory().spaces[&AsId(s.space)]
            .records
            .contains_key(&REGION),
        "the bookkeeping keeps the 211 mapping inside the destroyed region"
    );
}

/// The region a sparse mapping at `start` leads to, in `space`'s table.
fn region_at(s: &Ready, start: u64) -> Option<RegionId> {
    s.rig
        .mappings(s.space)
        .iter()
        .find(|mapping| mapping.range.start() == start)
        .and_then(|mapping| match mapping.target {
            Target::GpuOnly { region, .. } => Some(region),
            _ => None,
        })
}

/// A destroyed region's id is not handed out again before the GPU reports the destroy done
/// (the wiring note on issue #332: the GPU frees a destroyed region's pages when it applies the
/// destroy, so an id reused before then would have the new region's pages freed too). Ids are
/// never reused at all: one made after the report takes the next number too.
#[test]
fn a_destroyed_regions_id_is_not_reused_before_or_after_the_destroy_is_done() {
    let mut s = ready();
    let size = 8 * PAGE_SIZE;
    let second = REGION + 2 * size;
    let third = REGION + 4 * size;
    s.rig.send_on(
        s.worker,
        &sparse_words(s.space, REGION, REGION + size - 1, 12, 0, s.device, 0, 0),
    );
    assert_eq!(region_at(&s, REGION), Some(RegionId(1)));
    let destroy = range_words(
        DESTROY_SPARSE_REGION,
        s.space,
        REGION,
        REGION + size - 1,
        0,
        0,
    );
    assert_eq!(
        result_of(&s.rig.send_on(s.worker, &destroy)),
        ResultCode::SUCCESS
    );
    // The destroy is queued for the GPU and not applied.
    s.rig.send_on(
        s.worker,
        &sparse_words(s.space, second, second + size - 1, 12, 0, s.device, 0, 0),
    );
    assert_eq!(region_at(&s, second), Some(RegionId(2)));
    s.rig.apply();
    s.rig.drain();
    s.rig.send_on(
        s.worker,
        &sparse_words(s.space, third, third + size - 1, 12, 0, s.device, 0, 0),
    );
    assert_eq!(region_at(&s, third), Some(RegionId(3)));
}

// ----- Bounds and the shim ---------------------------------------------------------------------

/// A memory with no kernel behind it, for the bounds: registrations are given tokens directly.
fn bare() -> (GpuMemory, FakeFront) {
    let mut gpu = GpuMemory::new();
    gpu.create_device(Device { name: 1, label: 2 }).unwrap();
    (gpu, FakeFront::default())
}

fn token_for(range: PageRange) -> DeviceShareToken {
    device_share_token(
        crate::completion::testing::process(),
        range.start(),
        range.size(),
    )
}

/// Registers `range` with device 1 straight on the bookkeeping, its token made up; the handle.
fn register_bare(gpu: &mut GpuMemory, range: PageRange) -> u32 {
    let request = RegisterRequest {
        first: range.start(),
        last: range.end() - 1,
        flags: 0,
        device: 1,
    };
    gpu.check_registration(&request)
        .unwrap()
        .register(token_for(range))
}

/// An address space at exactly [`MAX_PIECES`] pieces refuses one more piece, and one below
/// takes it: registrations and sparse regions count together.
#[test]
fn pieces_are_bounded_at_the_limit() {
    let (mut gpu, front) = bare();
    let space = gpu.create_address_space(&front).unwrap();
    let range = page_range(MEMORY, PAGE_SIZE);
    register_bare(&mut gpu, range);
    let sparse = |n: u64| SparseRequest {
        space,
        first: REGION + n * PAGE_SIZE,
        last: REGION + (n + 1) * PAGE_SIZE - 1,
        page_shift: 12,
        mode: 0,
        device: 1,
        sync: 0,
    };
    let regions = u64::try_from(MAX_PIECES).unwrap() - 2;
    for n in 0..regions {
        gpu.create_sparse_region(&sparse(n)).unwrap();
    }
    assert_eq!(gpu.pieces(space), MAX_PIECES - 1);
    gpu.create_sparse_region(&sparse(regions)).unwrap();
    assert_eq!(gpu.pieces(space), MAX_PIECES);
    assert_eq!(
        gpu.create_sparse_region(&sparse(regions + 1)),
        Err(KernelError::OutOfResource)
    );
    let request = RegisterRequest {
        first: 0x9_0000_0000,
        last: 0x9_0000_0000 + PAGE_SIZE - 1,
        flags: 0,
        device: 1,
    };
    assert_eq!(
        gpu.check_registration(&request).err(),
        Some(KernelError::OutOfResource)
    );
    assert!(
        gpu.registration(2).is_none(),
        "the refused registration is not kept"
    );
}

/// [`MAX_PENDING_JOBS`] jobs may wait; one more is `Busy`, and changes nothing.
#[test]
fn pending_jobs_are_bounded_at_the_limit() {
    let (mut gpu, front) = bare();
    let space = gpu.create_address_space(&front).unwrap();
    let range = page_range(MEMORY, PAGE_SIZE);
    let handle = register_bare(&mut gpu, range);
    let event = ObjectId::from_parts(4, 5);
    let map = MapRequest {
        space,
        first: GPU_VA,
        last: GPU_VA + PAGE_SIZE - 1,
        handle,
        offset: 0,
        permission: 0,
        cache: 0,
        kind: 6,
        fallback: 6,
        mode: 0,
        device: 1,
        sync: 0,
    };
    gpu.map(&map).unwrap();
    let modify = ModifyRequest {
        space,
        first: GPU_VA,
        last: GPU_VA + PAGE_SIZE - 1,
        kind: 6,
        fallback: 6,
    };
    for _ in 1..MAX_PENDING_JOBS {
        gpu.modify_kind(&front, &modify, event).unwrap();
    }
    assert_eq!(gpu.pending_jobs(), MAX_PENDING_JOBS - 1);
    gpu.modify_kind(&front, &modify, event).unwrap();
    assert_eq!(gpu.pending_jobs(), MAX_PENDING_JOBS);
    let unmap = RangeRequest {
        space,
        first: GPU_VA,
        last: GPU_VA + PAGE_SIZE - 1,
    };
    assert_eq!(gpu.unmap(&front, &unmap, event), Err(KernelError::Busy));
    assert_eq!(
        gpu.mappings(space).len(),
        2,
        "the refused unmap changed nothing"
    );
    assert_eq!(
        gpu.destroy_memory(&front, handle, event),
        Err(KernelError::Busy)
    );
    assert!(gpu.registration(handle).is_some());
    // gpu.rs: 102's "no such handle" comes before "no room for another job".
    assert_eq!(
        gpu.destroy_memory(&front, handle + 1, event),
        Err(KernelError::NotFound)
    );
}

/// One syncpoint shim per address space.
#[test]
fn one_syncpoint_shim_per_address_space() {
    let (mut gpu, front) = bare();
    let space = gpu.create_address_space(&front).unwrap();
    let other = gpu.create_address_space(&front).unwrap();
    let page = VA_WINDOW_END;
    assert_eq!(
        gpu.map_syncpoint_shim(space, page + 1),
        Err(KernelError::InvalidAddress)
    );
    assert_eq!(
        gpu.map_syncpoint_shim(AsId(9), page),
        Err(KernelError::NotFound)
    );
    gpu.map_syncpoint_shim(space, page).unwrap();
    assert_eq!(
        gpu.map_syncpoint_shim(space, page + PAGE_SIZE),
        Err(KernelError::InvalidState)
    );
    gpu.map_syncpoint_shim(other, page).unwrap();
    let shim = Mapping {
        range: page_range(page, PAGE_SIZE),
        target: Target::SyncpointShim,
        kind: SELF_MAP_KIND,
    };
    assert_eq!(gpu.mappings(space), [shim]);
}

/// A ticket no job waits on does nothing but its note; a job's event the kernel refuses is
/// logged and the drain goes on to the next.
#[test]
fn the_drain_skips_unknown_tickets_and_goes_on_after_a_refused_signal() {
    let mut s = ready();
    s.map_at(GPU_VA);
    s.rig.send_on(
        s.worker,
        &range_words(
            NORMAL_UNMAP_GPU_MEMORY_4K,
            s.space,
            GPU_VA,
            GPU_VA + MEMORY_SIZE - 1,
            0,
            0,
        ),
    );
    s.rig.send_on(s.worker, &[0x76, 2, s.handle, 0]);
    let queue = Arc::clone(s.rig.shared.queue());
    queue.push(Completion::Gpu(GpuCompletion::GpuOrdered {
        ticket: Ticket(77),
    }));
    s.rig.front.finish(&queue);
    // A host that refuses every signal.
    let mut host = Refusing(Vec::new());
    let taken = s.rig.shared.drain(&mut host);
    assert_eq!(taken.len(), 3);
    // The destroy after the refused unmap signal still released its lock.
    assert_eq!(host.0.len(), 1);
    assert_eq!(s.rig.gpu().gpu_memory().pending_jobs(), 0);
}

// ----- What the table shows while a job waits ----------------------------------------------------

/// The sparse region request for the `n`th page from [`REGION`] in `space`, with `device`.
fn sparse_at(space: AsId, n: u64, device: u8) -> SparseRequest {
    SparseRequest {
        space,
        first: REGION + n * PAGE_SIZE,
        last: REGION + (n + 1) * PAGE_SIZE - 1,
        page_shift: SMALL_PAGE_SHIFT,
        mode: 0,
        device,
        sync: 0,
    }
}

/// The 213 of the tests' whole-registration mapping at `va`.
fn unmap_words(space: u32, va: u64) -> Vec<u32> {
    range_words(
        NORMAL_UNMAP_GPU_MEMORY_4K,
        space,
        va,
        va + MEMORY_SIZE - 1,
        0,
        0,
    )
}

/// A 214 over the pages a waiting 213 removed is not written into the table until the job is
/// done: until the GPU applies the unmap, the table still shows the mapping (as the work queued
/// before the 213 expects), then a hole, then, once drained, the region.
#[test]
fn a_sparse_region_over_a_pending_unmap_waits_for_the_drain() {
    let mut s = ready();
    s.map_at(GPU_VA);
    let mapped = guest(page_range(GPU_VA, MEMORY_SIZE), MEMORY, GENERIC_16BX2);
    let unmap = unmap_words(s.space, GPU_VA);
    assert_eq!(
        result_of(&s.rig.send_on(s.worker, &unmap)),
        ResultCode::SUCCESS
    );
    let words = sparse_words(
        s.space,
        GPU_VA,
        GPU_VA + MEMORY_SIZE - 1,
        u32::from(SMALL_PAGE_SHIFT),
        0,
        s.device,
        0,
        0,
    );
    assert_eq!(
        result_of(&s.rig.send_on(s.worker, &words)),
        ResultCode::SUCCESS
    );
    assert_eq!(s.rig.mappings(s.space), [own(), mapped]);
    s.rig.apply();
    assert_eq!(s.rig.mappings(s.space), [own()]);
    s.rig.drain();
    let region = Mapping {
        range: page_range(GPU_VA, MEMORY_SIZE),
        target: Target::GpuOnly {
            region: RegionId(1),
            offset: 0,
        },
        kind: SPARSE_KIND,
    };
    assert_eq!(s.rig.mappings(s.space), [own(), region]);
}

/// A 101 at the CPU address a waiting 213 removed at the GPU address is not mapped there until
/// the job is done.
#[test]
fn a_registration_over_a_pending_unmap_waits_for_the_drain() {
    let mut s = ready();
    s.map_at(GPU_VA);
    let mapped = guest(page_range(GPU_VA, MEMORY_SIZE), MEMORY, GENERIC_16BX2);
    let unmap = unmap_words(s.space, GPU_VA);
    assert_eq!(
        result_of(&s.rig.send_on(s.worker, &unmap)),
        ResultCode::SUCCESS
    );
    s.rig.register(s.worker, GPU_VA, MEMORY_SIZE, s.device);
    assert_eq!(s.rig.mappings(s.space), [own(), mapped]);
    s.rig.apply();
    assert_eq!(s.rig.mappings(s.space), [own()]);
    s.rig.drain();
    let self_mapped = guest(page_range(GPU_VA, MEMORY_SIZE), GPU_VA, SELF_MAP_KIND);
    assert_eq!(s.rig.mappings(s.space), [own(), self_mapped]);
}

/// A 213 sends its unmap only where its mapping shows: the shim inside it stays in the table
/// while the GPU applies the unmap, and after the drain.
#[test]
fn an_unmap_keeps_the_shim() {
    let mut s = ready();
    s.map_at(GPU_VA);
    let shim = GPU_VA + PAGE_SIZE;
    s.rig
        .gpu()
        .gpu()
        .map_syncpoint_shim(AsId(s.space), shim)
        .unwrap();
    let shim_mapping = Mapping {
        range: page_range(shim, PAGE_SIZE),
        target: Target::SyncpointShim,
        kind: SELF_MAP_KIND,
    };
    assert_eq!(
        s.rig.at(s.space, shim),
        Some((Target::SyncpointShim, SELF_MAP_KIND))
    );
    let unmap = unmap_words(s.space, GPU_VA);
    assert_eq!(
        result_of(&s.rig.send_on(s.worker, &unmap)),
        ResultCode::SUCCESS
    );
    let space = AsId(s.space);
    let calls = s.rig.front.calls();
    assert_eq!(
        calls[calls.len() - 3..],
        [
            FrontCall::Update(
                space,
                VmUpdate::Unmap {
                    range: page_range(GPU_VA, PAGE_SIZE)
                },
                None
            ),
            FrontCall::Update(
                space,
                VmUpdate::Unmap {
                    range: page_range(GPU_VA + 2 * PAGE_SIZE, 2 * PAGE_SIZE)
                },
                None
            ),
            FrontCall::Barrier(Ticket(1)),
        ]
    );
    s.rig.apply();
    assert_eq!(s.rig.mappings(s.space), [own(), shim_mapping]);
    s.rig.drain();
    assert_eq!(s.rig.mappings(s.space), [own(), shim_mapping]);
}

/// A 212 sends its kind change only where its mapping shows: the shim inside it keeps its kind
/// while the GPU applies the change, and after the drain.
#[test]
fn a_kind_change_keeps_the_shim() {
    let mut s = ready();
    s.map_at(GPU_VA);
    let shim = GPU_VA + PAGE_SIZE;
    s.rig
        .gpu()
        .gpu()
        .map_syncpoint_shim(AsId(s.space), shim)
        .unwrap();
    let words = modify_words(s.space, GPU_VA, GPU_VA + MEMORY_SIZE - 1, 0xB, 1, 0, 0);
    let reply = s.rig.send_on(s.worker, &words);
    assert_eq!(result_of(&reply), ResultCode::SUCCESS);
    assert_eq!(data(&reply), [1]);
    let space = AsId(s.space);
    let calls = s.rig.front.calls();
    assert_eq!(
        calls[calls.len() - 3..],
        [
            FrontCall::Update(
                space,
                VmUpdate::ModifyKind {
                    range: page_range(GPU_VA, PAGE_SIZE),
                    kind: Z16
                },
                None
            ),
            FrontCall::Update(
                space,
                VmUpdate::ModifyKind {
                    range: page_range(GPU_VA + 2 * PAGE_SIZE, 2 * PAGE_SIZE),
                    kind: Z16
                },
                None
            ),
            FrontCall::Barrier(Ticket(1)),
        ]
    );
    let shim_page = Some((Target::SyncpointShim, SELF_MAP_KIND));
    s.rig.apply();
    assert_eq!(s.rig.at(s.space, shim), shim_page);
    assert_eq!(s.rig.at(s.space, GPU_VA), page_at(MEMORY, Z16));
    s.rig.drain();
    assert_eq!(s.rig.at(s.space, shim), shim_page);
    assert_eq!(
        s.rig.at(s.space, GPU_VA + 3 * PAGE_SIZE),
        page_at(MEMORY + 3 * PAGE_SIZE, Z16)
    );
    // A 212 of the shim's page alone sends no kind change, only its barrier.
    let words = modify_words(s.space, shim, shim + PAGE_SIZE - 1, 6, 6, 0, 0);
    assert_eq!(
        result_of(&s.rig.send_on(s.worker, &words)),
        ResultCode::SUCCESS
    );
    let calls = s.rig.front.calls();
    assert_eq!(calls.last(), Some(&FrontCall::Barrier(Ticket(2))));
    assert!(matches!(calls[calls.len() - 2], FrontCall::Barrier(_)));
}

/// A 102 sends its destroy only where its registration shows: another registration's own
/// mapping, another handle's 211 mapping and the shim on its range stay in the table while the
/// GPU applies it, and after the drain.
#[test]
fn a_destroy_keeps_what_shows_above_it() {
    let mut s = ready();
    // Another registration over the last page of the first's range, a 211 of it at the first's
    // first page, and the shim at its second: only the third page shows the first registration.
    let other = s
        .rig
        .register(s.worker, MEMORY + 3 * PAGE_SIZE, PAGE_SIZE, s.device);
    let words = map_words(
        s.space,
        MEMORY,
        MEMORY + PAGE_SIZE - 1,
        other,
        0,
        MapBytes::of(s.device),
        0,
    );
    assert_eq!(
        result_of(&s.rig.send_on(s.worker, &words)),
        ResultCode::SUCCESS
    );
    s.rig
        .gpu()
        .gpu()
        .map_syncpoint_shim(AsId(s.space), MEMORY + PAGE_SIZE)
        .unwrap();
    let pages = |s: &Ready| -> Vec<Option<(Target, Kind)>> {
        (0..4)
            .map(|n| s.rig.at(s.space, MEMORY + n * PAGE_SIZE))
            .collect()
    };
    let mapped = page_at(MEMORY + 3 * PAGE_SIZE, GENERIC_16BX2);
    let shim = Some((Target::SyncpointShim, SELF_MAP_KIND));
    let third = page_at(MEMORY + 2 * PAGE_SIZE, SELF_MAP_KIND);
    let fourth = page_at(MEMORY + 3 * PAGE_SIZE, SELF_MAP_KIND);
    assert_eq!(pages(&s), [mapped, shim, third, fourth]);
    assert_eq!(
        result_of(&s.rig.send_on(s.worker, &[0x76, 2, s.handle, 0])),
        ResultCode::SUCCESS
    );
    let space = AsId(s.space);
    assert_eq!(
        s.rig.front.calls()[1..],
        [
            FrontCall::Update(
                space,
                VmUpdate::DestroyMemory {
                    range: page_range(MEMORY + 2 * PAGE_SIZE, PAGE_SIZE)
                },
                None
            ),
            FrontCall::Barrier(Ticket(1)),
        ]
    );
    assert_eq!(pages(&s), [mapped, shim, third, fourth]);
    s.rig.apply();
    assert_eq!(pages(&s), [mapped, shim, None, fourth]);
    s.rig.drain();
    assert_eq!(pages(&s), [mapped, shim, None, fourth]);
}

/// A 102 takes its 211 mappings out with it, and the drain publishes their ranges again: a
/// sparse region under one shows there once more.
#[test]
fn a_destroy_shows_a_region_under_its_mappings_again() {
    let mut s = ready();
    let size = 8 * PAGE_SIZE;
    let words = sparse_words(
        s.space,
        REGION,
        REGION + size - 1,
        u32::from(SMALL_PAGE_SHIFT),
        0,
        s.device,
        0,
        0,
    );
    assert_eq!(
        result_of(&s.rig.send_on(s.worker, &words)),
        ResultCode::SUCCESS
    );
    s.map_at(REGION + 2 * PAGE_SIZE);
    assert_eq!(
        s.rig.at(s.space, REGION + 2 * PAGE_SIZE),
        page_at(MEMORY, GENERIC_16BX2)
    );
    assert_eq!(
        result_of(&s.rig.send_on(s.worker, &[0x76, 2, s.handle, 0])),
        ResultCode::SUCCESS
    );
    s.rig.finish();
    for n in 0..8 {
        let offset = n * PAGE_SIZE;
        let region = Target::GpuOnly {
            region: RegionId(1),
            offset,
        };
        assert_eq!(
            s.rig.at(s.space, REGION + offset),
            Some((region, SPARSE_KIND)),
            "page {n}"
        );
    }
    assert_eq!(s.rig.at(s.space, MEMORY), None);
}

/// A 211 that cuts a mapping leaves its tail at its own place in the registration: published
/// again from the bookkeeping (a 212 of the tail, drained), the tail's last page is the
/// registration's last.
#[test]
fn a_cut_off_tail_keeps_its_place_in_the_registration() {
    let mut s = ready();
    s.map_at(GPU_VA);
    let words = map_words(
        s.space,
        GPU_VA + PAGE_SIZE,
        GPU_VA + 2 * PAGE_SIZE - 1,
        s.handle,
        0,
        MapBytes::of(s.device),
        0,
    );
    assert_eq!(
        result_of(&s.rig.send_on(s.worker, &words)),
        ResultCode::SUCCESS
    );
    let tail = GPU_VA + 2 * PAGE_SIZE;
    // Compressible Z16 (0xb) with fallback 1: backend Z16.
    let modify = modify_words(s.space, tail, tail + 2 * PAGE_SIZE - 1, 0xB, 1, 0, 0);
    assert_eq!(
        result_of(&s.rig.send_on(s.worker, &modify)),
        ResultCode::SUCCESS
    );
    s.rig.finish();
    assert_eq!(
        s.rig.at(s.space, tail),
        page_at(MEMORY + 2 * PAGE_SIZE, Z16)
    );
    assert_eq!(
        s.rig.at(s.space, tail + PAGE_SIZE),
        page_at(MEMORY + MEMORY_SIZE - PAGE_SIZE, Z16)
    );
}

/// A 212 of a middle page keeps the runs before and after it, so the runs still cover the
/// mapping and the bookkeeping saves.
#[test]
fn a_middle_kind_change_keeps_the_runs_after_it() {
    let mut s = ready();
    s.map_at(GPU_VA);
    let page = GPU_VA + PAGE_SIZE;
    let modify = modify_words(s.space, page, page + PAGE_SIZE - 1, 0xB, 1, 0, 0);
    assert_eq!(
        result_of(&s.rig.send_on(s.worker, &modify)),
        ResultCode::SUCCESS
    );
    let locked = s.rig.gpu();
    let memory = locked.gpu_memory();
    assert_eq!(
        memory.spaces[&AsId(s.space)].records[&GPU_VA].spans,
        [
            (GPU_VA, page, GENERIC_16BX2),
            (page, page + PAGE_SIZE, Z16),
            (page + PAGE_SIZE, GPU_VA + MEMORY_SIZE, GENERIC_16BX2),
        ]
    );
    assert!(memory.save().is_ok());
}

/// A 214 may not overlap a 211 mapping, by one page; neighbours ending exactly at its start and
/// starting exactly at its end are allowed.
#[test]
fn a_sparse_region_may_not_overlap_a_mapping() {
    let mut s = ready();
    s.map_at(GPU_VA);
    let (space, device) = (s.space, s.device);
    let mut ask = |first: u64| {
        let words = sparse_words(
            space,
            first,
            first + PAGE_SIZE - 1,
            u32::from(SMALL_PAGE_SHIFT),
            0,
            device,
            0,
            0,
        );
        result_of(&s.rig.send_on(s.worker, &words))
    };
    assert_eq!(
        ask(GPU_VA + MEMORY_SIZE - PAGE_SIZE),
        code(KernelError::InvalidCurrentMemory)
    );
    assert_eq!(ask(GPU_VA), code(KernelError::InvalidCurrentMemory));
    assert_eq!(ask(GPU_VA + MEMORY_SIZE), ResultCode::SUCCESS);
    assert_eq!(ask(GPU_VA - PAGE_SIZE), ResultCode::SUCCESS);
}

/// `regions` one-page sparse regions from [`REGION`] on, made in `space` of `gpu`.
fn fill_with_regions(gpu: &mut GpuMemory, space: AsId, regions: usize, device: u8) {
    for n in 0..u64::try_from(regions).unwrap() {
        gpu.create_sparse_region(&sparse_at(space, n, device))
            .unwrap();
    }
}

/// A fresh 211 at [`MAX_PIECES`] pieces is `OutOfResource`, one below it fits, and one that
/// exactly replaces a mapping adds nothing and fits at the limit.
#[test]
fn a_map_is_bounded_at_the_limit() {
    let (mut gpu, front) = bare();
    let space = gpu.create_address_space(&front).unwrap();
    let handle = register_bare(&mut gpu, page_range(MEMORY, MEMORY_SIZE));
    let map = |first: u64| MapRequest {
        space,
        first,
        last: first + PAGE_SIZE - 1,
        handle,
        offset: 0,
        permission: 0,
        cache: 0,
        kind: 6,
        fallback: 6,
        mode: 0,
        device: 1,
        sync: 0,
    };
    gpu.map(&map(GPU_VA)).unwrap();
    // The registration and the mapping, then regions up to one below the limit.
    fill_with_regions(&mut gpu, space, MAX_PIECES - 3, 1);
    assert_eq!(gpu.pieces(space), MAX_PIECES - 1);
    gpu.map(&map(GPU_VA + PAGE_SIZE)).unwrap();
    assert_eq!(gpu.pieces(space), MAX_PIECES);
    assert_eq!(
        gpu.map(&map(GPU_VA + 2 * PAGE_SIZE)),
        Err(KernelError::OutOfResource)
    );
    assert!(gpu.map(&map(GPU_VA)).is_ok());
    assert_eq!(gpu.pieces(space), MAX_PIECES);
}

/// A 212 of a middle page adds two runs: at [`MAX_PIECES`] - 1 pieces it is `OutOfResource`
/// and queues nothing; at [`MAX_PIECES`] - 2 it fits exactly.
#[test]
fn a_kind_change_is_bounded_at_the_limit() {
    let (mut gpu, front) = bare();
    let space = gpu.create_address_space(&front).unwrap();
    let handle = register_bare(&mut gpu, page_range(MEMORY, MEMORY_SIZE));
    let map = MapRequest {
        space,
        first: GPU_VA,
        last: GPU_VA + 3 * PAGE_SIZE - 1,
        handle,
        offset: 0,
        permission: 0,
        cache: 0,
        kind: 6,
        fallback: 6,
        mode: 0,
        device: 1,
        sync: 0,
    };
    gpu.map(&map).unwrap();
    fill_with_regions(&mut gpu, space, MAX_PIECES - 3, 1);
    assert_eq!(gpu.pieces(space), MAX_PIECES - 1);
    let middle = ModifyRequest {
        space,
        first: GPU_VA + PAGE_SIZE,
        last: GPU_VA + 2 * PAGE_SIZE - 1,
        kind: 0xB,
        fallback: 1,
    };
    let event = ObjectId::from_parts(4, 5);
    assert_eq!(
        gpu.modify_kind(&front, &middle, event),
        Err(KernelError::OutOfResource)
    );
    assert_eq!(gpu.pending_jobs(), 0);
    let first_region = RangeRequest {
        space,
        first: REGION,
        last: REGION + PAGE_SIZE - 1,
    };
    gpu.destroy_sparse_region(&front, &first_region, event)
        .unwrap();
    assert_eq!(gpu.pieces(space), MAX_PIECES - 2);
    assert!(gpu.modify_kind(&front, &middle, event).is_ok());
    assert_eq!(gpu.pieces(space), MAX_PIECES);
}

/// gpu.rs: 101's room check comes before the GPU-accessible attribute, so a full address space
/// and a page without the attribute answer `OutOfResource`, and no lock is taken; with room,
/// the same request answers the attribute.
#[test]
fn registration_room_is_checked_before_the_attribute() {
    let mut s = ready();
    let space = AsId(s.space);
    let device = u8::try_from(s.device).unwrap();
    {
        let mut locked = s.rig.gpu();
        // The tests' registration is one piece; the regions fill the rest.
        fill_with_regions(locked.gpu(), space, MAX_PIECES - 1, device);
        assert_eq!(locked.gpu_memory().pieces(space), MAX_PIECES);
    }
    let page = 0x9_0000_0000;
    s.rig
        .kernel
        .map(page, vec![0; usize::try_from(PAGE_SIZE).unwrap()]);
    let words = register_words(page, page + PAGE_SIZE - 1, 0, s.device);
    assert_eq!(
        result_of(&s.rig.send_on(s.worker, &words)),
        code(KernelError::OutOfResource)
    );
    assert_eq!(s.rig.kernel.device_locks(page, PAGE_SIZE), 0);
    let first_region = RangeRequest {
        space,
        first: REGION,
        last: REGION + PAGE_SIZE - 1,
    };
    s.rig
        .gpu()
        .gpu()
        .destroy_sparse_region(&*s.rig.front, &first_region, s.event)
        .unwrap();
    assert_eq!(
        result_of(&s.rig.send_on(s.worker, &words)),
        code(KernelError::InvalidCurrentMemory)
    );
    assert_eq!(s.rig.kernel.device_locks(page, PAGE_SIZE), 0);
}

/// A 102 unmaps the 211 mappings it takes out only where they show: the shim inside one stays
/// in the table while the GPU applies the destroy, and after the drain.
#[test]
fn a_destroy_keeps_the_shim_inside_its_mappings() {
    let mut s = ready();
    s.map_at(GPU_VA);
    let shim = GPU_VA + PAGE_SIZE;
    s.rig
        .gpu()
        .gpu()
        .map_syncpoint_shim(AsId(s.space), shim)
        .unwrap();
    assert_eq!(
        result_of(&s.rig.send_on(s.worker, &[0x76, 2, s.handle, 0])),
        ResultCode::SUCCESS
    );
    let space = AsId(s.space);
    let unmap = |start: u64, size: u64| {
        FrontCall::Update(
            space,
            VmUpdate::Unmap {
                range: page_range(start, size),
            },
            None,
        )
    };
    let destroy = FrontCall::Update(
        space,
        VmUpdate::DestroyMemory {
            range: page_range(MEMORY, MEMORY_SIZE),
        },
        None,
    );
    assert_eq!(
        s.rig.front.calls()[1..],
        [
            destroy,
            unmap(GPU_VA, PAGE_SIZE),
            unmap(GPU_VA + 2 * PAGE_SIZE, 2 * PAGE_SIZE),
            FrontCall::Barrier(Ticket(1)),
        ]
    );
    let shim_mapping = Mapping {
        range: page_range(shim, PAGE_SIZE),
        target: Target::SyncpointShim,
        kind: SELF_MAP_KIND,
    };
    s.rig.apply();
    assert_eq!(s.rig.mappings(s.space), [shim_mapping]);
    s.rig.drain();
    assert_eq!(s.rig.mappings(s.space), [shim_mapping]);
}

// ----- Destroying an address space -----------------------------------------------------------

/// A 211 of `pages` pages of registration `handle` at `va` in `space`, device 1, generic kind.
fn map_bare(gpu: &mut GpuMemory, space: AsId, va: u64, handle: u32, pages: u64) {
    let request = MapRequest {
        space,
        first: va,
        last: va + pages * PAGE_SIZE - 1,
        handle,
        offset: 0,
        permission: 0,
        cache: 0,
        kind: 6,
        fallback: 6,
        mode: 0,
        device: 1,
        sync: 0,
    };
    gpu.map(&request).unwrap();
}

/// What a destroy of `space` sends: `update`, with no ticket.
fn destroy_call(space: AsId, update: VmUpdate) -> FrontCall {
    FrontCall::Update(space, update, None)
}

/// A destroy sends, in address order and with no ticket, a `DestroySparseRegion` over each
/// sparse region and an `Unmap` over each run the table maps outside them (touching runs merged;
/// a 211 inside a region covered by the region's destroy; the shim on the page right after the
/// region an `Unmap` of its own), which empty the GPU's table; the other address space, which has
/// pieces at the same addresses, is untouched.
#[test]
fn destroying_an_address_space_empties_its_table_in_order() {
    let (mut gpu, front) = bare();
    let space = gpu.create_address_space(&front).unwrap();
    let other = gpu.create_address_space(&front).unwrap();
    let handle = register_bare(&mut gpu, page_range(MEMORY, MEMORY_SIZE));
    // A 211 that starts exactly where the registration's own mapping ends.
    map_bare(&mut gpu, space, MEMORY + MEMORY_SIZE, handle, 2);
    gpu.create_sparse_region(&SparseRequest {
        last: REGION + 4 * PAGE_SIZE - 1,
        ..sparse_at(space, 0, 1)
    })
    .unwrap();
    map_bare(&mut gpu, space, REGION + PAGE_SIZE, handle, 1);
    let shim = REGION + 4 * PAGE_SIZE;
    gpu.map_syncpoint_shim(space, shim).unwrap();
    map_bare(&mut gpu, other, MEMORY + MEMORY_SIZE, handle, 2);
    let other_before = gpu.mappings(other);
    let before = front.calls().len();

    assert_eq!(gpu.destroy_address_space(&front, space), Ok(()));

    assert_eq!(
        front.calls()[before..],
        [
            destroy_call(
                space,
                VmUpdate::Unmap {
                    range: page_range(MEMORY, MEMORY_SIZE + 2 * PAGE_SIZE)
                }
            ),
            destroy_call(
                space,
                VmUpdate::DestroySparseRegion {
                    range: page_range(REGION, 4 * PAGE_SIZE),
                    region: RegionId(1),
                }
            ),
            destroy_call(
                space,
                VmUpdate::Unmap {
                    range: page_range(shim, PAGE_SIZE)
                }
            ),
        ]
    );
    assert_eq!(gpu.address_spaces().collect::<Vec<_>>(), [other]);
    assert!(gpu.table(space).is_none());
    assert_eq!(gpu.pieces(space), 0);
    assert_eq!(gpu.mappings(other), other_before);
    // The GPU applies the removals: its table of the destroyed space is empty.
    let queue = CompletionQueue::new();
    front.finish(&queue);
    let table = Arc::clone(&front.tables.lock().unwrap()[&space]);
    assert_eq!(table.mappings(), []);
    assert_eq!(gpu.mappings(other), other_before);
    // The registration stays.
    assert!(gpu.registration(handle).is_some());
}

/// After a destroy every request naming the space (211 to 215 over what the space held, the
/// shim's placing, and a second destroy) is `NotFound` and sends nothing; the slot is free at
/// once, and the next space takes a new id.
#[test]
fn a_destroyed_space_is_gone_and_its_id_is_never_reused() {
    let (mut gpu, front) = bare();
    let spaces: Vec<AsId> = (0..MAX_ADDRESS_SPACES)
        .map(|_| gpu.create_address_space(&front).unwrap())
        .collect();
    assert_eq!(
        gpu.create_address_space(&front),
        Err(KernelError::OutOfResource)
    );
    let handle = register_bare(&mut gpu, page_range(MEMORY, MEMORY_SIZE));
    let space = spaces[1];
    map_bare(&mut gpu, space, GPU_VA, handle, 1);
    gpu.create_sparse_region(&sparse_at(space, 0, 1)).unwrap();
    gpu.destroy_address_space(&front, space).unwrap();
    let calls = front.calls().len();
    let event = ObjectId::from_parts(4, 5);
    let range = RangeRequest {
        space,
        first: GPU_VA,
        last: GPU_VA + PAGE_SIZE - 1,
    };
    assert_eq!(
        gpu.destroy_address_space(&front, space),
        Err(KernelError::NotFound)
    );
    let map = MapRequest {
        space,
        first: GPU_VA,
        last: GPU_VA + PAGE_SIZE - 1,
        handle,
        offset: 0,
        permission: 0,
        cache: 0,
        kind: 6,
        fallback: 6,
        mode: 0,
        device: 1,
        sync: 0,
    };
    assert_eq!(gpu.map(&map), Err(KernelError::NotFound));
    // 212 over the mapping the space had.
    let modify = ModifyRequest {
        space,
        first: GPU_VA,
        last: GPU_VA + PAGE_SIZE - 1,
        kind: 6,
        fallback: 6,
    };
    assert_eq!(
        gpu.modify_kind(&front, &modify, event),
        Err(KernelError::NotFound)
    );
    assert_eq!(gpu.unmap(&front, &range, event), Err(KernelError::NotFound));
    assert_eq!(
        gpu.create_sparse_region(&sparse_at(space, 1, 1)),
        Err(KernelError::NotFound)
    );
    // 215 over the sparse region the space had.
    let region = RangeRequest {
        space,
        first: REGION,
        last: REGION + PAGE_SIZE - 1,
    };
    assert_eq!(
        gpu.destroy_sparse_region(&front, &region, event),
        Err(KernelError::NotFound)
    );
    assert_eq!(
        gpu.map_syncpoint_shim(space, VA_WINDOW_END),
        Err(KernelError::NotFound)
    );
    assert_eq!(front.calls().len(), calls, "a refusal sends nothing");
    assert_eq!(gpu.pending_jobs(), 0);
    // One slot free: the next space is made, with the id after the last one handed out.
    let last = spaces[MAX_ADDRESS_SPACES - 1];
    assert_eq!(gpu.create_address_space(&front), Ok(AsId(last.0 + 1)));
    assert_eq!(
        gpu.create_address_space(&front),
        Err(KernelError::OutOfResource)
    );
    // An id gsv never handed out is not found either.
    assert_eq!(
        gpu.destroy_address_space(&front, AsId(last.0 + 2)),
        Err(KernelError::NotFound)
    );
}

/// A destroy while ordered jobs wait: each keeps its lock and event and still completes, but no
/// longer holds or publishes the destroyed space's pages; the other space's pages they hold are
/// published when they are done.
#[test]
fn a_destroy_drops_a_waiting_job_s_pages_of_the_space_and_keeps_its_event() {
    let (mut gpu, front) = bare();
    let space = gpu.create_address_space(&front).unwrap();
    let other = gpu.create_address_space(&front).unwrap();
    let handle = register_bare(&mut gpu, page_range(MEMORY, MEMORY_SIZE));
    let kept = register_bare(&mut gpu, page_range(GPU_VA, PAGE_SIZE));
    map_bare(&mut gpu, space, GPU_VA + PAGE_SIZE, kept, 1);
    map_bare(&mut gpu, other, GPU_VA + PAGE_SIZE, kept, 1);
    let event = ObjectId::from_parts(4, 5);
    // A 213 in each space, then a 102: three jobs, each holding pages.
    let unmap = |space| RangeRequest {
        space,
        first: GPU_VA + PAGE_SIZE,
        last: GPU_VA + 2 * PAGE_SIZE - 1,
    };
    gpu.unmap(&front, &unmap(space), event).unwrap();
    gpu.unmap(&front, &unmap(other), event).unwrap();
    let ticket = gpu.destroy_memory(&front, handle, event).unwrap();
    assert!(gpu.waiting.contains_key(&space));
    assert!(
        gpu.pending
            .values()
            .any(|job| job.reveal.iter().any(|&(id, _)| id == space))
    );

    gpu.destroy_address_space(&front, space).unwrap();

    assert!(!gpu.waiting.contains_key(&space));
    assert!(gpu.waiting.contains_key(&other));
    for job in gpu.pending.values() {
        assert!(job.reveal.iter().all(|&(id, _)| id == other), "{job:?}");
    }
    assert_eq!(gpu.pending_jobs(), 3);
    // The GPU applies everything queued, then each job is done.
    front.finish(&CompletionQueue::new());
    let finished = gpu.ordered_done(ticket).unwrap();
    assert_eq!(finished.event, event);
    assert_eq!(
        finished.releases,
        [token_for(page_range(MEMORY, MEMORY_SIZE))]
    );
    gpu.ordered_done(Ticket(ticket.0 - 1)).unwrap();
    gpu.ordered_done(Ticket(ticket.0 - 2)).unwrap();
    assert!(gpu.waiting.is_empty());
    assert!(gpu.table(space).is_none());
    // Every job done, the other space shows what its bookkeeping holds: the kept registration's
    // own mapping.
    assert_eq!(
        gpu.mappings(other),
        [guest(page_range(GPU_VA, PAGE_SIZE), GPU_VA, SELF_MAP_KIND)]
    );
}

/// The ordering that protects guest memory: a 102 after a destroy sends its removals and its
/// barrier after the destroy's removals, and none for the destroyed space, so the GPU has
/// stopped reading through the destroyed table before the drain releases the lock.
#[test]
fn a_later_102_is_ordered_after_the_destroy_s_removals() {
    let (mut gpu, front) = bare();
    let space = gpu.create_address_space(&front).unwrap();
    let other = gpu.create_address_space(&front).unwrap();
    let handle = register_bare(&mut gpu, page_range(MEMORY, MEMORY_SIZE));
    let before = front.calls().len();
    gpu.destroy_address_space(&front, space).unwrap();
    let event = ObjectId::from_parts(4, 5);
    let ticket = gpu.destroy_memory(&front, handle, event).unwrap();
    let own = page_range(MEMORY, MEMORY_SIZE);
    assert_eq!(
        front.calls()[before..],
        [
            destroy_call(space, VmUpdate::Unmap { range: own }),
            FrontCall::Update(other, VmUpdate::DestroyMemory { range: own }, None),
            FrontCall::Barrier(ticket),
        ]
    );
}

mod mirror;
mod sparse_map;
