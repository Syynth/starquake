//! nvhost-gpu driven directly over an nvsrv state ([`Bench`]) and through the nvsrv server
//! ([`Live`]), over a fake channel front that models a channel's bounded queue.
//!
//! Arguments are laid out by hand from Eden's structs (`nvhost_gpu.h` @ cf654052:
//! `IoctlAllocGpfifoEx`, `IoctlAllocObjCtx`, `IoctlSubmitGpfifo`; `nvhost_as_gpu.h`:
//! `IoctlBindChannel`), and ioctl numbers are written as libnx writes them
//! (`libnx/nx/source/nvidia/ioctl/nvchannel.c`, `nvhost-as-gpu.c` @ b156b477: `_NV_IOWR` sets
//! both direction bits, `_NV_IOW` only the "in" bit 30, and the size field holds the struct's
//! size).

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use starquake_gpu::interface::{Completions, GpuCompletion, SYNCPOINT_COUNT};
use starquake_ipc::{BUFFER_SIZE, Message, SessionKind};
use starquake_kernel::dispatch::DeviceShareError;
use starquake_kernel::id::SlabId;
use starquake_kernel::memory::DeviceShareToken;
use starquake_kernel::result::{ResultCode, SvcResult};
use starquake_services::framework::{Family, Handled, TokenId, UnknownPolicy};
use starquake_trace::EventKind;

use super::super::command::{CLOSE, IOCTL2, QUERY_EVENT};
use super::super::tests::{
    AS_ALLOC_AS_EX, Answer, Auto, RecordingFront, Rig, STALE, answer, as_front, cmif, fd_and, le,
    len16, ok, open_direct, saved, shared_state,
};
use super::super::{
    Device, GpuAccess, IoctlCall, NvsrvFamily, NvsrvState, NvsrvStateError, nvsrv_server,
};
use super::submits::{
    MAX_WAITING_ENTRIES, OverCap, SNAPSHOT_MAGIC, SNAPSHOT_VERSION, SavedSubmit, SnapshotError,
    charge,
};
use super::*;
use crate::gpu::GpuMemory;
use crate::shared::TimerHost;

// ----- Ioctl numbers, as libnx writes them -----------------------------------------------------

/// libnx `_NV_IOWR(group, nr, size)`: both direction bits.
const fn iowr(group: u32, nr: u32, size: u32) -> u32 {
    0xC000_0000 | (size << 16) | (group << 8) | nr
}

/// `_NV_IOWR(0x48, 0x1A, ..)`, 32 bytes.
const ALLOC_GPFIFO_EX2: u32 = iowr(0x48, 0x1A, 32);
/// `_NV_IOWR(0x48, 0x09, ..)`, 16 bytes.
const ALLOC_OBJ_CTX: u32 = iowr(0x48, 0x09, 16);
/// `_NV_IOWR(0x48, 0x1B, ..)`, 24 bytes (libnx `nvioctlChannel_KickoffPb`, on `Ioctl2`).
const KICKOFF: u32 = iowr(0x48, 0x1B, 24);
/// nvhost-as-gpu's `BindChannel`: `_NV_IOW(0x41, 0x01, ..)`, 4 bytes.
const AS_BIND_CHANNEL: u32 = 0x4004_4101;

/// `_NV_IOWR(0x48, 0x08, ..)` for a submission of `count` entries after the 24-byte struct.
fn submit_number(count: usize) -> u32 {
    iowr(0x48, 0x08, u32::try_from(24 + 8 * count).unwrap())
}

#[test]
fn the_ioctl_numbers_decode_to_the_device_s_commands() {
    use crate::nvsrv::ioctl::Ioctl;
    let cases = [
        (ALLOC_GPFIFO_EX2, command::ALLOC_GPFIFO_EX2, 32),
        (ALLOC_OBJ_CTX, command::ALLOC_OBJ_CTX, 16),
        (KICKOFF, command::SUBMIT_GPFIFO_INLINE, 24),
        (submit_number(2), command::SUBMIT_GPFIFO, 40),
    ];
    for (raw, cmd, size) in cases {
        let ioctl = Ioctl::from_raw(raw);
        assert_eq!(
            (ioctl.group(), ioctl.command(), ioctl.size()),
            (GROUP, cmd, size)
        );
    }
    let bind = Ioctl::from_raw(AS_BIND_CHANNEL);
    assert!(crate::nvsrv::as_gpu::is_bind_channel(
        IoctlVariant::Ioctl,
        bind
    ));
    assert!(!crate::nvsrv::as_gpu::is_bind_channel(
        IoctlVariant::Ioctl2,
        bind
    ));
}

// ----- The fake front --------------------------------------------------------------------------

/// The first channel id the fake hands out: non-zero, so a zero cannot pass for one.
const FIRST_CHANNEL: u32 = 0x30;

/// One call the fake channel front took.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Call {
    Create(ChannelConfig),
    Bind(ChannelId, u32),
    Queued(ChannelId, Submission),
    Full(ChannelId),
    Destroy(ChannelId),
}

/// A channel front that records its calls and keeps, per channel, how many entries its queue
/// holds: a submission that does not fit in [`CHANNEL_QUEUE_ENTRIES`] is `Full` and queues
/// nothing, as gpu-protocol's front does ("A full queue").
#[derive(Default)]
struct FakeChannels {
    calls: Mutex<Vec<Call>>,
    held: Mutex<BTreeMap<ChannelId, usize>>,
    next: Mutex<u32>,
    refuse_channels: Mutex<bool>,
    refuse_classes: Mutex<bool>,
}

impl ChannelFront for FakeChannels {
    fn create_channel(&self, config: ChannelConfig) -> Result<ChannelId, GpuError> {
        if *self.refuse_channels.lock().unwrap() {
            return Err(GpuError);
        }
        let mut next = self.next.lock().unwrap();
        let id = ChannelId(FIRST_CHANNEL + *next);
        *next += 1;
        self.calls.lock().unwrap().push(Call::Create(config));
        Ok(id)
    }

    fn bind_class(&self, channel: ChannelId, class: u32) -> Result<(), GpuError> {
        self.calls.lock().unwrap().push(Call::Bind(channel, class));
        if *self.refuse_classes.lock().unwrap() {
            Err(GpuError)
        } else {
            Ok(())
        }
    }

    fn submit(&self, channel: ChannelId, submission: Submission) -> Submitted {
        let mut held = self.held.lock().unwrap();
        let count = held.entry(channel).or_default();
        if *count + submission.entries.len() > CHANNEL_QUEUE_ENTRIES {
            self.calls.lock().unwrap().push(Call::Full(channel));
            return Submitted::Full;
        }
        *count += submission.entries.len();
        let call = Call::Queued(channel, submission);
        self.calls.lock().unwrap().push(call);
        Submitted::Queued
    }

    fn destroy_channel(&self, channel: ChannelId) {
        self.calls.lock().unwrap().push(Call::Destroy(channel));
    }
}

impl FakeChannels {
    fn calls(&self) -> Vec<Call> {
        self.calls.lock().unwrap().clone()
    }

    fn clear(&self) {
        self.calls.lock().unwrap().clear();
    }

    /// The submissions queued, in order.
    fn queued(&self) -> Vec<(ChannelId, Submission)> {
        self.calls()
            .into_iter()
            .filter_map(|call| match call {
                Call::Queued(channel, submission) => Some((channel, submission)),
                _ => None,
            })
            .collect()
    }

    /// Sets how many entries `channel`'s queue holds.
    fn hold(&self, channel: ChannelId, entries: usize) {
        self.held.lock().unwrap().insert(channel, entries);
    }

    fn refuse_channels(&self, refuse: bool) {
        *self.refuse_channels.lock().unwrap() = refuse;
    }

    fn refuse_classes(&self, refuse: bool) {
        *self.refuse_classes.lock().unwrap() = refuse;
    }
}

fn channel_front(fake: &Arc<FakeChannels>) -> Arc<dyn ChannelFront> {
    Arc::<FakeChannels>::clone(fake)
}

/// A drain host that records what is recorded and refuses nothing.
#[derive(Default)]
struct Host(Vec<EventKind>);

impl TimerHost for Host {
    fn record(&mut self, kind: EventKind) {
        self.0.push(kind);
    }

    fn unlock_device_share(&mut self, _: DeviceShareToken) -> Result<(), DeviceShareError> {
        Ok(())
    }

    fn signal_event(&mut self, _: ObjectId) -> SvcResult<()> {
        Ok(())
    }
}

// ----- Arguments, laid out by hand -------------------------------------------------------------

/// `IoctlAllocGpfifoEx` with distinct values: entries 0x400, jobs 0x11, flags 0x22, fence out
/// 0x33 and 0x44 (overwritten on success), reserved 0x55, 0x66, 0x77.
fn alloc_request() -> Vec<u8> {
    le(&[0x400, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77])
}

/// What `alloc_request` comes back as with fence (`id`, `value`).
fn alloc_reply(id: u32, value: u32) -> Vec<u8> {
    le(&[0x400, 0x11, 0x22, id, value, 0x55, 0x66, 0x77])
}

/// `IoctlAllocObjCtx`: `class`, `flags`, object id 0xDEADBEEF (libnx's).
fn obj_request(class: u32, flags: u32) -> Vec<u8> {
    le(&[class, flags, 0xDEAD_BEEF, 0])
}

/// `IoctlSubmitGpfifo`: address `0x1234_5678_9ABC` (not read), `count`, `flags`, fence (`id`,
/// `value`).
fn submit_struct(count: u32, flags: u32, id: u32, value: u32) -> Vec<u8> {
    le(&[0x5678_9ABC, 0x1234, count, flags, id, value])
}

/// `entries` as GP entries' bytes.
fn entry_bytes(entries: &[u64]) -> Vec<u8> {
    entries
        .iter()
        .flat_map(|entry| entry.to_le_bytes())
        .collect()
}

/// Distinct entries, from `first`: each differs in both its words.
fn entries_from(first: u64, count: usize) -> Vec<u64> {
    (0..u64::try_from(count).unwrap())
        .map(|n| (first + n) | ((first + n) << 40))
        .collect()
}

/// `AllocAsEx` with no parent: big pages 0x10000, start 0x2000, end 0x9000, split 0x5000.
fn as_request() -> Vec<u8> {
    le(&[0, 0, 0x1_0000, 0, 0x2000, 0, 0x9000, 0, 0x5000, 0])
}

// ----- Driving the state directly --------------------------------------------------------------

/// An nvsrv state driven without the server, over a GPU memory, a recording VM front, a waiting
/// table, syncpoints and a fake channel front of its own.
struct Bench {
    state: NvsrvState,
    memory: GpuMemory,
    vm: RecordingFront,
    submits: Submits,
    syncpoints: Arc<Syncpoints>,
    fake: Arc<FakeChannels>,
    front: Arc<dyn ChannelFront>,
}

impl Bench {
    fn new() -> Self {
        let queue = Arc::new(crate::completion::CompletionQueue::new());
        let completions: Arc<dyn Completions> = queue;
        let fake = Arc::new(FakeChannels::default());
        Self {
            state: NvsrvState::new(),
            memory: GpuMemory::new(),
            vm: RecordingFront::default(),
            submits: Submits::new(),
            syncpoints: Arc::new(Syncpoints::new(completions)),
            front: channel_front(&fake),
            fake,
        }
    }

    fn open(&mut self, device: Device) -> i32 {
        let (fd, result) = open_direct(&mut self.state, device);
        assert_eq!(result, NvResult::Success);
        fd
    }

    /// `raw` on `fd` through `variant`, with `input` and `inline`: the answer and the deferral.
    fn call(
        &mut self,
        variant: IoctlVariant,
        fd: i32,
        raw: u32,
        input: &[u8],
        inline: &[u8],
    ) -> (IoctlAnswer, Option<Deferral>) {
        let call = IoctlCall {
            variant,
            fd,
            ioctl: Ioctl::from_raw(raw),
            input,
            inline_input: inline,
            out_len: input.len(),
            inline_out_len: 0,
        };
        let mut deferred = None;
        let gpu = GpuAccess {
            memory: &mut self.memory,
            front: &self.vm,
            channels: Some(ChannelAccess {
                submits: &mut self.submits,
                syncpoints: &self.syncpoints,
                front: &self.front,
                deferred: &mut deferred,
            }),
        };
        let answer = self.state.ioctl(UnknownPolicy::Strict, &call, gpu);
        (answer, deferred)
    }

    /// `raw` on `fd` through `Ioctl`, which must not defer.
    fn ioctl(&mut self, fd: i32, raw: u32, input: &[u8]) -> IoctlAnswer {
        let (answer, deferral) = self.call(IoctlVariant::Ioctl, fd, raw, input, &[]);
        assert_eq!(deferral, None);
        answer
    }

    fn close(&mut self, fd: i32) -> NvResult {
        let mut deferred = None;
        let gpu = GpuAccess {
            memory: &mut self.memory,
            front: &self.vm,
            channels: Some(ChannelAccess {
                submits: &mut self.submits,
                syncpoints: &self.syncpoints,
                front: &self.front,
                deferred: &mut deferred,
            }),
        };
        self.state.close(fd, gpu)
    }

    /// An as-gpu fd with a view of a new address space.
    fn space(&mut self) -> (i32, AsId) {
        let fd = self.open(Device::NvhostAsGpu);
        assert_eq!(
            self.ioctl(fd, AS_ALLOC_AS_EX, &as_request()).result,
            NvResult::Success
        );
        let space = self.state.as_gpu().view(fd.cast_unsigned()).unwrap().space;
        (fd, space)
    }

    /// Binds channel fd `channel` through as-gpu fd `fd`.
    fn bind(&mut self, fd: i32, channel: i32) -> NvResult {
        self.ioctl(fd, AS_BIND_CHANNEL, &channel.to_le_bytes())
            .result
    }

    /// `AllocGPFIFOEx2` on `fd`, which must succeed: the fence out.
    fn alloc(&mut self, fd: i32) -> (u32, u32) {
        let answer = self.ioctl(fd, ALLOC_GPFIFO_EX2, &alloc_request());
        assert_eq!(answer.result, NvResult::Success);
        let word = |at: usize| u32::from_le_bytes(answer.output[at..at + 4].try_into().unwrap());
        (
            word(layout::ALLOC_FENCE_ID),
            word(layout::ALLOC_FENCE_VALUE),
        )
    }

    /// A channel bound to a new address space, with a syncpoint and a GPU channel: its fd, the
    /// as-gpu fd, its syncpoint and its GPU channel.
    fn channel(&mut self) -> (i32, i32, SyncpointId, ChannelId) {
        let (as_fd, _) = self.space();
        let fd = self.open(Device::NvhostGpu);
        assert_eq!(self.bind(as_fd, fd), NvResult::Success);
        let (id, _) = self.alloc(fd);
        let channel = self.state.gpu().channel(fd.cast_unsigned()).unwrap();
        let made = ChannelId(channel.made.unwrap().channel);
        let syncpoint = SyncpointId::new(u8::try_from(id).unwrap()).unwrap();
        (fd, as_fd, syncpoint, made)
    }

    /// `SubmitGPFIFO` 0x8 of `entries` with `flags` and fence (`id`, `value`).
    fn submit(
        &mut self,
        fd: i32,
        entries: &[u64],
        flags: u32,
        (id, value): (u32, u32),
    ) -> (IoctlAnswer, Option<Deferral>) {
        let count = u32::try_from(entries.len()).unwrap();
        let mut input = submit_struct(count, flags, id, value);
        input.extend(entry_bytes(entries));
        let raw = submit_number(entries.len());
        self.call(IoctlVariant::Ioctl, fd, raw, &input, &[])
    }

    /// `SubmitGPFIFO` 0x1B on `Ioctl2` with `count` and the entries `inline`.
    fn kickoff(
        &mut self,
        fd: i32,
        count: u32,
        inline: &[u8],
        flags: u32,
    ) -> (IoctlAnswer, Option<Deferral>) {
        let input = submit_struct(count, flags, 0, 0);
        self.call(IoctlVariant::Ioctl2, fd, KICKOFF, &input, inline)
    }
}

/// The fence a submission's answer gives back.
fn fence_of(answer: &IoctlAnswer) -> (u32, u32) {
    let word = |at: usize| u32::from_le_bytes(answer.output[at..at + 4].try_into().unwrap());
    (
        word(layout::SUBMIT_FENCE_ID),
        word(layout::SUBMIT_FENCE_VALUE),
    )
}

// ----- AllocGPFIFOEx2 --------------------------------------------------------------------------

/// Opening allocates nothing; `AllocGPFIFOEx2` allocates the lowest free syncpoint and gives back
/// it and its maximum, the rest of the struct as sent; closing frees it.
#[test]
fn the_syncpoint_comes_with_alloc_gpfifo_ex2() {
    let mut bench = Bench::new();
    let first = SyncpointId::new(FIRST_ALLOCATABLE).unwrap();
    let fd = bench.open(Device::NvhostGpu);
    assert!(!bench.syncpoints.is_allocated(first));
    // A maximum already raised (a syncpoint keeps its values when freed).
    bench.syncpoints.allocate().unwrap();
    bench.syncpoints.increment_max(first, 0x99).unwrap();
    bench.syncpoints.free(first).unwrap();
    let answer = bench.ioctl(fd, ALLOC_GPFIFO_EX2, &alloc_request());
    assert_eq!(
        (answer.result, answer.output),
        (
            NvResult::Success,
            alloc_reply(u32::from(FIRST_ALLOCATABLE), 0x99)
        )
    );
    assert!(bench.syncpoints.is_allocated(first));
    let channel = bench.state.gpu().channel(fd.cast_unsigned()).unwrap();
    assert_eq!(channel.syncpoint, Some(FIRST_ALLOCATABLE));
    // Unbound: no GPU channel yet.
    assert_eq!(channel.made, None);
    assert_eq!(bench.fake.calls(), []);
    assert_eq!(bench.close(fd), NvResult::Success);
    assert!(!bench.syncpoints.is_allocated(first));
    assert_eq!(bench.state.gpu().channel(fd.cast_unsigned()), None);
}

#[test]
fn a_second_alloc_gpfifo_ex2_is_already_allocated() {
    let mut bench = Bench::new();
    let fd = bench.open(Device::NvhostGpu);
    bench.alloc(fd);
    let before = bench.state.clone();
    let answer = bench.ioctl(fd, ALLOC_GPFIFO_EX2, &alloc_request());
    // Refused, the struct given back as sent.
    assert_eq!(
        (answer.result, answer.output),
        (NvResult::AlreadyAllocated, alloc_request())
    );
    assert_eq!(bench.state, before);
    // One syncpoint, not two: the next allocatable one is still free.
    let next = SyncpointId::new(FIRST_ALLOCATABLE + 1).unwrap();
    assert!(!bench.syncpoints.is_allocated(next));
}

#[test]
fn alloc_gpfifo_ex2_with_no_free_syncpoint_is_insufficient_memory() {
    let mut bench = Bench::new();
    while bench.syncpoints.allocate().is_ok() {}
    let fd = bench.open(Device::NvhostGpu);
    let answer = bench.ioctl(fd, ALLOC_GPFIFO_EX2, &alloc_request());
    assert_eq!(
        (answer.result, answer.output),
        (NvResult::InsufficientMemory, alloc_request())
    );
    assert_eq!(bench.state.gpu().channel(fd.cast_unsigned()), None);
}

/// Bound first, `AllocGPFIFOEx2` makes the GPU channel in the bound space with the syncpoint.
#[test]
fn alloc_gpfifo_ex2_makes_the_channel_when_bound() {
    let mut bench = Bench::new();
    let (as_fd, space) = bench.space();
    let fd = bench.open(Device::NvhostGpu);
    assert_eq!(bench.bind(as_fd, fd), NvResult::Success);
    assert_eq!(bench.fake.calls(), []);
    let (id, _) = bench.alloc(fd);
    let syncpoint = SyncpointId::new(u8::try_from(id).unwrap()).unwrap();
    assert_eq!(
        bench.fake.calls(),
        [Call::Create(ChannelConfig { space, syncpoint })]
    );
    let made = bench.state.gpu().channel(fd.cast_unsigned()).unwrap().made;
    assert_eq!(
        made,
        Some(Made {
            channel: FIRST_CHANNEL,
            space
        })
    );
}

/// A GPU that refuses the channel: `InsufficientMemory`, nothing recorded, and the syncpoint is
/// free again (the next allocation gets it).
#[test]
fn a_refused_channel_frees_its_syncpoint() {
    let mut bench = Bench::new();
    let (as_fd, _) = bench.space();
    let fd = bench.open(Device::NvhostGpu);
    bench.bind(as_fd, fd);
    bench.fake.refuse_channels(true);
    let answer = bench.ioctl(fd, ALLOC_GPFIFO_EX2, &alloc_request());
    assert_eq!(
        (answer.result, answer.output),
        (NvResult::InsufficientMemory, alloc_request())
    );
    assert_eq!(bench.state.gpu().channel(fd.cast_unsigned()), None);
    let first = SyncpointId::new(FIRST_ALLOCATABLE).unwrap();
    assert!(!bench.syncpoints.is_allocated(first));
    // The next attempt succeeds, with the same syncpoint.
    bench.fake.refuse_channels(false);
    assert_eq!(bench.alloc(fd).0, u32::from(FIRST_ALLOCATABLE));
}

// ----- BindChannel -----------------------------------------------------------------------------

#[test]
fn bind_channel_checks_the_channel_then_the_view_then_the_space() {
    let mut bench = Bench::new();
    let (as_fd, space) = bench.space();
    let empty_as = bench.open(Device::NvhostAsGpu);
    let nvmap = bench.open(Device::Nvmap);
    let fd = bench.open(Device::NvhostGpu);
    // The channel: negative, not open, not an nvhost-gpu.
    for channel in [-1, fd + 1, nvmap, as_fd] {
        let answer = bench.ioctl(as_fd, AS_BIND_CHANNEL, &channel.to_le_bytes());
        assert_eq!(
            (answer.result, answer.output),
            (NvResult::BadParameter, channel.to_le_bytes().to_vec())
        );
    }
    // A short argument reads as fd 0, which is never open.
    assert_eq!(
        bench.ioctl(as_fd, AS_BIND_CHANNEL, &[]).result,
        NvResult::BadParameter
    );
    // Both a bad channel and no view: the channel is reported.
    assert_eq!(bench.bind(empty_as, nvmap), NvResult::BadParameter);
    // No view.
    assert_eq!(bench.bind(empty_as, fd), NvResult::InvalidState);
    assert_eq!(bench.state.as_gpu().bound(fd.cast_unsigned()), None);
    // Bound.
    assert_eq!(bench.bind(as_fd, fd), NvResult::Success);
    assert_eq!(bench.state.as_gpu().bound(fd.cast_unsigned()), Some(space));
    // Made in this space: binding here again is fine, another space is refused.
    bench.alloc(fd);
    assert_eq!(bench.bind(as_fd, fd), NvResult::Success);
    let (other_as, _) = bench.space();
    assert_eq!(bench.bind(other_as, fd), NvResult::InvalidState);
    assert_eq!(bench.state.as_gpu().bound(fd.cast_unsigned()), Some(space));
}

/// The issue's case (#189): the last as-gpu fd of a space closes while a channel is bound to it,
/// and the space lives until the channel goes.
#[test]
fn a_bound_channel_keeps_its_space_after_the_last_view_closes() {
    let mut bench = Bench::new();
    let (fd, as_fd, _, _) = bench.channel();
    let space = bench.state.as_gpu().bound(fd.cast_unsigned()).unwrap();
    assert_eq!(bench.close(as_fd), NvResult::Success);
    assert!(bench.memory.table(space).is_some());
    assert!(bench.state.as_gpu().made(space).is_some());
    // A submission still works.
    let (answer, _) = bench.submit(fd, &entries_from(1, 1), 0, (0, 0));
    assert_eq!(answer.result, NvResult::Success);
    assert_eq!(bench.close(fd), NvResult::Success);
    assert!(bench.memory.table(space).is_none());
    assert_eq!(bench.state.as_gpu().made(space), None);
    // The channel was destroyed (that it goes before the space is pinned by
    // `closing_a_channel_destroys_it_before_the_space_it_held`).
    assert!(matches!(bench.fake.calls().last(), Some(Call::Destroy(_))));
}

/// The other order: the channel goes first, and the view keeps the space; the view's close then
/// destroys it.
#[test]
fn a_view_keeps_its_space_after_its_channel_closes() {
    let mut bench = Bench::new();
    let (fd, as_fd, _, _) = bench.channel();
    let space = bench.state.as_gpu().bound(fd.cast_unsigned()).unwrap();
    bench.close(fd);
    assert!(bench.memory.table(space).is_some());
    assert_eq!(bench.state.as_gpu().bound(fd.cast_unsigned()), None);
    bench.close(as_fd);
    assert!(bench.memory.table(space).is_none());
}

/// Binding elsewhere (before the GPU channel is made) releases the old space: a made space with
/// no view left goes then, and one a view holds stays.
#[test]
fn rebinding_releases_the_old_space() {
    let mut bench = Bench::new();
    let (first_as, first) = bench.space();
    let (second_as, second) = bench.space();
    let fd = bench.open(Device::NvhostGpu);
    bench.bind(first_as, fd);
    bench.close(first_as);
    assert!(bench.memory.table(first).is_some());
    assert_eq!(bench.bind(second_as, fd), NvResult::Success);
    assert!(bench.memory.table(first).is_none());
    // Back and forth with a view held: nothing goes.
    let (third_as, third) = bench.space();
    assert_eq!(bench.bind(third_as, fd), NvResult::Success);
    assert!(bench.memory.table(second).is_some());
    assert!(bench.memory.table(third).is_some());
}

#[test]
fn a_binding_s_space_is_checked_against_the_memory() {
    let mut bench = Bench::new();
    let (fd, as_fd, _, _) = bench.channel();
    bench.close(as_fd);
    let as_gpu = bench.state.as_gpu().clone();
    assert_eq!(as_gpu.check_spaces(&bench.memory), Ok(()));
    let space = as_gpu.bound(fd.cast_unsigned()).unwrap();
    // Against a memory that holds nothing: the made space is reported before the binding.
    assert_eq!(
        as_gpu.check_spaces(&GpuMemory::new()),
        Err(crate::nvsrv::as_gpu::AsGpuStateError::MadeSpaceNotHeld { space: space.0 })
    );
    // A binding alone (to a space gpu 33 made, which as-gpu does not record as made).
    let mut gpu33 = Bench::new();
    let space = gpu33.memory.create_address_space(&gpu33.vm).unwrap();
    let parent = i32::try_from(space.0).unwrap();
    let as_fd = gpu33.open(Device::NvhostAsGpu);
    let view = le(&[
        5,
        parent.cast_unsigned(),
        0x1000,
        0,
        0x800_0000,
        0,
        0x900_0000,
        0,
        0,
        0,
    ]);
    assert_eq!(
        gpu33.ioctl(as_fd, AS_ALLOC_AS_EX, &view).result,
        NvResult::Success
    );
    let channel = gpu33.open(Device::NvhostGpu);
    assert_eq!(gpu33.bind(as_fd, channel), NvResult::Success);
    gpu33.close(as_fd);
    let as_gpu = gpu33.state.as_gpu();
    assert_eq!(
        as_gpu.check_spaces(&GpuMemory::new()),
        Err(crate::nvsrv::as_gpu::AsGpuStateError::BoundSpaceNotHeld {
            fd: channel.cast_unsigned(),
            space: space.0
        })
    );
    assert_eq!(as_gpu.check_spaces(&gpu33.memory), Ok(()));
}

// ----- AllocObjCtx -----------------------------------------------------------------------------

/// NX2 gives back the class it mapped onto, with the flags 0 and the object id as sent; the GPU
/// is given the class as sent.
#[test]
fn alloc_obj_ctx_gives_back_the_mapped_class() {
    let mut bench = Bench::new();
    let (fd, _, _, channel) = bench.channel();
    bench.fake.clear();
    let answer = bench.ioctl(fd, ALLOC_OBJ_CTX, &obj_request(clc797::CLASS, 0x5));
    assert_eq!(
        (answer.result, answer.output),
        (NvResult::Success, obj_request(class::CTX_3D, 0))
    );
    assert_eq!(bench.fake.calls(), [Call::Bind(channel, clc797::CLASS)]);
    // A slot taken: refused, the mapped class and zero flags still given back.
    let answer = bench.ioctl(fd, ALLOC_OBJ_CTX, &obj_request(0xC697, 0x7));
    assert_eq!(
        (answer.result, answer.output),
        (NvResult::AlreadyAllocated, obj_request(class::CTX_3D, 0))
    );
    assert_eq!(bench.fake.calls().len(), 1);
}

/// Every class NX2 maps, on a channel of its own, takes its slot; others are refused.
#[test]
fn every_mapped_class_is_taken_and_others_refused() {
    let mapped: Vec<(u32, u32)> = [
        (class::ON_3D.as_slice(), class::CTX_3D),
        (class::ON_COMPUTE.as_slice(), class::CTX_COMPUTE),
        (class::ON_DMA.as_slice(), class::CTX_DMA),
        (class::ON_GPFIFO.as_slice(), class::CTX_CHANNEL_GPFIFO),
        (class::SLOTS.as_slice(), 0),
    ]
    .into_iter()
    .flat_map(|(classes, onto)| {
        classes
            .iter()
            .map(move |&class| (class, if onto == 0 { class } else { onto }))
    })
    .collect();
    for (requested, onto) in mapped {
        let mut bench = Bench::new();
        let fd = bench.open(Device::NvhostGpu);
        let answer = bench.ioctl(fd, ALLOC_OBJ_CTX, &obj_request(requested, 0));
        assert_eq!(
            (answer.result, answer.output),
            (NvResult::Success, obj_request(onto, 0)),
            "{requested:#x}"
        );
        assert_eq!(
            bench
                .state
                .gpu()
                .channel(fd.cast_unsigned())
                .unwrap()
                .classes,
            [requested]
        );
    }
    for requested in [1, 0xC797 + 1, 0xB196, 0xC897, u32::MAX] {
        let mut bench = Bench::new();
        let fd = bench.open(Device::NvhostGpu);
        let answer = bench.ioctl(fd, ALLOC_OBJ_CTX, &obj_request(requested, 0));
        assert_eq!(
            (answer.result, answer.output),
            (NvResult::BadParameter, obj_request(requested, 0)),
            "{requested:#x}"
        );
        assert_eq!(bench.state.gpu().channel(fd.cast_unsigned()), None);
    }
}

/// Class 0 fills 3D, compute, copy, 2D and inline-to-memory, in that order, then is refused; with
/// one slot taken it skips that one; the GPU, once the channel is made, is given the classes in
/// the order bound.
#[test]
fn class_0_fills_the_free_slots_in_nx2_s_order() {
    let mut bench = Bench::new();
    let fd = bench.open(Device::NvhostGpu);
    for expected in class::ZERO_CANDIDATES {
        let answer = bench.ioctl(fd, ALLOC_OBJ_CTX, &obj_request(0, 0));
        assert_eq!(answer.output, obj_request(expected, 0));
    }
    assert_eq!(
        [
            class::CTX_3D,
            class::CTX_COMPUTE,
            class::CTX_DMA,
            class::CTX_2D,
            class::CTX_KEPLER
        ],
        class::ZERO_CANDIDATES
    );

    let mut bench = Bench::new();
    let fd = bench.open(Device::NvhostGpu);
    // Compute taken first, so class 0 skips it.
    bench.ioctl(fd, ALLOC_OBJ_CTX, &obj_request(clc7c0::CLASS, 0));
    let mut given = vec![clc7c0::CLASS];
    for expected in [
        class::CTX_3D,
        class::CTX_DMA,
        class::CTX_2D,
        class::CTX_KEPLER,
    ] {
        let answer = bench.ioctl(fd, ALLOC_OBJ_CTX, &obj_request(0, 0));
        assert_eq!(
            (answer.result, answer.output),
            (NvResult::Success, obj_request(expected, 0))
        );
        given.push(expected);
    }
    let answer = bench.ioctl(fd, ALLOC_OBJ_CTX, &obj_request(0, 0));
    assert_eq!(
        (answer.result, answer.output),
        (NvResult::BadParameter, obj_request(0, 0))
    );
    // The GPFIFO slot is still free for its own class.
    let answer = bench.ioctl(fd, ALLOC_OBJ_CTX, &obj_request(clc76f::CLASS, 0));
    assert_eq!(answer.result, NvResult::Success);
    given.push(clc76f::CLASS);
    assert_eq!(
        bench
            .state
            .gpu()
            .channel(fd.cast_unsigned())
            .unwrap()
            .classes,
        given
    );
    // Making the channel binds them all, in that order.
    let (as_fd, space) = bench.space();
    bench.bind(as_fd, fd);
    bench.alloc(fd);
    let channel = ChannelId(FIRST_CHANNEL);
    let mut expected = vec![Call::Create(ChannelConfig {
        space,
        syncpoint: SyncpointId::new(FIRST_ALLOCATABLE).unwrap(),
    })];
    expected.extend(given.iter().map(|&class| Call::Bind(channel, class)));
    assert_eq!(bench.fake.calls(), expected);
}

/// A GPU that refuses a class on a made channel: `BadParameter`, and nothing recorded, so the
/// slot is free for the next attempt.
#[test]
fn a_refused_class_is_not_recorded() {
    let mut bench = Bench::new();
    let (fd, _, _, _) = bench.channel();
    bench.fake.refuse_classes(true);
    let answer = bench.ioctl(fd, ALLOC_OBJ_CTX, &obj_request(clc7b5::CLASS, 0));
    assert_eq!(
        (answer.result, answer.output),
        (NvResult::BadParameter, obj_request(class::CTX_DMA, 0))
    );
    assert_eq!(
        bench
            .state
            .gpu()
            .channel(fd.cast_unsigned())
            .unwrap()
            .classes,
        Vec::<u32>::new()
    );
    bench.fake.refuse_classes(false);
    let answer = bench.ioctl(fd, ALLOC_OBJ_CTX, &obj_request(clc7b5::CLASS, 0));
    assert_eq!(answer.result, NvResult::Success);
}

// ----- SubmitGPFIFO ----------------------------------------------------------------------------

/// The plan for every combination of the five flag bits the reference names (0, 1, 2, 4, 8),
/// each with a fence value of 0, a small one and the largest that does not overflow, against the
/// reference's arithmetic written out here; and the same through the device: the fence given
/// back, the maximum raised, and the submission the front gets.
#[test]
fn fence_arithmetic_for_every_flag_combination() {
    let bits = [
        FENCE_WAIT,
        FENCE_INCREMENT,
        NEW_HW_FORMAT,
        SUPPRESS_WFI,
        INCREMENT_VALUE,
    ];
    for combination in 0..1_u32 << bits.len() {
        let flags = bits
            .iter()
            .enumerate()
            .filter(|(at, _)| combination & (1 << at) != 0)
            .fold(0, |flags, (_, &bit)| flags | bit);
        for value in [0, 0x2A, u32::MAX - FENCE_INCREMENT_AMOUNT] {
            let wait = flags & FENCE_WAIT != 0;
            let by_value = flags & INCREMENT_VALUE != 0;
            let expected = if wait && by_value {
                Err(NvResult::BadParameter)
            } else {
                let fixed = if flags & FENCE_INCREMENT != 0 {
                    FENCE_INCREMENT_AMOUNT
                } else {
                    0
                };
                Ok(Plan {
                    wait,
                    increments: fixed + if by_value { value } else { 0 },
                    wfi: flags & SUPPRESS_WFI == 0,
                })
            };
            assert_eq!(plan(flags, value), expected, "{flags:#x} {value:#x}");

            let mut bench = Bench::new();
            let (fd, _, syncpoint, channel) = bench.channel();
            let before = bench.syncpoints.fence(syncpoint).unwrap().value;
            // The wait fence names syncpoint 0x21 (not the channel's), value `value`.
            let entries = entries_from(0x100, 3);
            let (answer, deferral) = bench.submit(fd, &entries, flags, (0x21, value));
            assert_eq!(deferral, None);
            let Ok(plan) = expected else {
                assert_eq!(answer.result, NvResult::BadParameter);
                let mut sent = submit_struct(3, flags, 0x21, value);
                sent.extend(entry_bytes(&entries));
                assert_eq!(answer.output, sent);
                assert_eq!(bench.fake.queued(), []);
                assert_eq!(bench.syncpoints.fence(syncpoint).unwrap().value, before);
                continue;
            };
            let raised = before.wrapping_add(plan.increments);
            assert_eq!(answer.result, NvResult::Success, "{flags:#x} {value:#x}");
            assert_eq!(fence_of(&answer), (u32::from(syncpoint.get()), raised));
            assert_eq!(bench.syncpoints.fence(syncpoint).unwrap().value, raised);
            // The flags word comes back 0, the rest as sent, the entries after it.
            let mut back = submit_struct(3, 0, u32::from(syncpoint.get()), raised);
            back.extend(entry_bytes(&entries));
            assert_eq!(answer.output, back);
            let wait_on = Fence {
                id: SyncpointId::new(0x21).unwrap(),
                value,
            };
            let submission = Submission {
                entries: entries.iter().copied().map(GpfifoEntry).collect(),
                wait: plan.wait.then_some(wait_on),
                increments: plan.increments,
                wfi: plan.wfi,
            };
            assert_eq!(bench.fake.queued(), [(channel, submission)]);
        }
    }
}

/// Bits 1 and 8 with a value that takes the increment exactly to `u32::MAX` is taken; one more
/// overflows and is refused, nothing raised.
#[test]
fn an_overflowing_increment_is_refused() {
    let flags = FENCE_INCREMENT | INCREMENT_VALUE;
    let edge = u32::MAX - FENCE_INCREMENT_AMOUNT;
    assert_eq!(plan(flags, edge).map(|plan| plan.increments), Ok(u32::MAX));
    assert_eq!(plan(flags, edge + 1), Err(NvResult::BadParameter));
    // Bit 8 alone takes `u32::MAX`.
    assert_eq!(
        plan(INCREMENT_VALUE, u32::MAX).map(|plan| plan.increments),
        Ok(u32::MAX)
    );
    let mut bench = Bench::new();
    let (fd, _, syncpoint, _) = bench.channel();
    let (answer, _) = bench.submit(fd, &[], flags, (0, edge + 1));
    assert_eq!(answer.result, NvResult::BadParameter);
    assert_eq!(bench.syncpoints.fence(syncpoint).unwrap().value, 0);
    // Both faults, bits 0 and 8 and an overflow: the pair is what is reported (it is checked
    // first, and both answer `BadParameter`, so the plan's own order pins it).
    assert_eq!(
        plan(flags | FENCE_WAIT, edge + 1),
        Err(NvResult::BadParameter)
    );
}

/// PR refuses a bit-8 value above 64; NX2, the larger, takes it.
#[test]
fn a_large_increment_value_is_taken() {
    let mut bench = Bench::new();
    let (fd, _, syncpoint, _) = bench.channel();
    let (answer, _) = bench.submit(fd, &entries_from(1, 1), INCREMENT_VALUE, (0, 65));
    assert_eq!(answer.result, NvResult::Success);
    assert_eq!(fence_of(&answer), (u32::from(syncpoint.get()), 65));
    assert_eq!(bench.fake.queued()[0].1.increments, 65);
}

/// A wait fence's id must be a syncpoint: the last id is taken, the count and a negative id are
/// refused; without bit 0 the id is not read.
#[test]
fn a_wait_on_no_syncpoint_is_refused() {
    let last = SYNCPOINT_COUNT - 1;
    for (id, result) in [
        (last, NvResult::Success),
        (SYNCPOINT_COUNT, NvResult::BadParameter),
        (u32::MAX, NvResult::BadParameter),
        (0x100 + 1, NvResult::BadParameter),
    ] {
        let mut bench = Bench::new();
        let (fd, _, _, _) = bench.channel();
        let (answer, _) = bench.submit(fd, &[], FENCE_WAIT, (id, 7));
        assert_eq!(answer.result, result, "{id:#x}");
        let (answer, _) = bench.submit(fd, &[], 0, (id, 7));
        assert_eq!(answer.result, NvResult::Success, "{id:#x}");
    }
}

/// A submission needs the channel's syncpoint and an address space: before `AllocGPFIFOEx2`, and
/// after it unbound, `InvalidState`; bound, the GPU channel is made then (Eden's lazy init).
#[test]
fn a_submission_needs_a_syncpoint_and_a_space() {
    let mut bench = Bench::new();
    let fd = bench.open(Device::NvhostGpu);
    let (answer, _) = bench.submit(fd, &entries_from(1, 1), 0, (0, 0));
    assert_eq!(answer.result, NvResult::InvalidState);
    // A channel with only an event: the same.
    bench
        .state
        .set_event(fd, event::SM_EXCEPTION_INT, ObjectId::from_parts(4, 5));
    let (answer, _) = bench.submit(fd, &entries_from(1, 1), 0, (0, 0));
    assert_eq!(answer.result, NvResult::InvalidState);
    let (id, _) = bench.alloc(fd);
    let (answer, _) = bench.submit(fd, &entries_from(1, 1), FENCE_INCREMENT, (0, 0));
    assert_eq!(answer.result, NvResult::InvalidState);
    let syncpoint = SyncpointId::new(u8::try_from(id).unwrap()).unwrap();
    assert_eq!(bench.syncpoints.fence(syncpoint).unwrap().value, 0);
    let (as_fd, space) = bench.space();
    bench.bind(as_fd, fd);
    assert_eq!(bench.fake.calls(), []);
    let (answer, _) = bench.submit(fd, &entries_from(1, 1), FENCE_INCREMENT, (0, 0));
    assert_eq!(answer.result, NvResult::Success);
    let calls = bench.fake.calls();
    assert_eq!(calls[0], Call::Create(ChannelConfig { space, syncpoint }));
    assert!(matches!(calls[1], Call::Queued(..)));
    // Both an unbound channel and bits 0 and 8: the bits are reported.
    let other = bench.open(Device::NvhostGpu);
    bench.alloc(other);
    let (answer, _) = bench.submit(other, &[], FENCE_WAIT | INCREMENT_VALUE, (0, 0));
    assert_eq!(answer.result, NvResult::BadParameter);
    // A GPU that refuses the channel made at submission: `InsufficientMemory`, nothing raised.
    bench.fake.refuse_channels(true);
    bench.bind(as_fd, other);
    let (answer, _) = bench.submit(other, &[], FENCE_INCREMENT, (0, 0));
    assert_eq!(answer.result, NvResult::InsufficientMemory);
    let other_syncpoint = SyncpointId::new(FIRST_ALLOCATABLE + 1).unwrap();
    assert_eq!(bench.syncpoints.fence(other_syncpoint).unwrap().value, 0);
}

/// The count against the entries the request carries: equal is taken, one more refused, a
/// trailing partial entry not counted; on `Ioctl` and `Ioctl2` alike. Both faults (too many and
/// bits 0 and 8): the count is reported.
#[test]
fn the_count_is_checked_against_the_entries_sent() {
    let mut bench = Bench::new();
    let (fd, _, _, _) = bench.channel();
    let entries = entries_from(0x40, 2);
    let mut partial = entry_bytes(&entries);
    partial.extend([0xEE; GP_ENTRY_BYTES - 1]);
    for (count, result) in [(2, NvResult::Success), (3, NvResult::InvalidSize)] {
        let mut input = submit_struct(count, 0, 0, 0);
        input.extend(&partial);
        let raw = submit_number(2);
        let (answer, _) = bench.call(IoctlVariant::Ioctl, fd, raw, &input, &[]);
        assert_eq!(answer.result, result, "{count}");
        // Given back: the struct and the whole entries after it.
        if result != NvResult::Success {
            let mut back = submit_struct(count, 0, 0, 0);
            back.extend(entry_bytes(&entries));
            assert_eq!(answer.output, back);
        }
        let (answer, _) = bench.kickoff(fd, count, &partial, 0);
        assert_eq!(answer.result, result, "{count}");
        // `Ioctl2` gives back the struct alone.
        if result != NvResult::Success {
            assert_eq!(answer.output, submit_struct(count, 0, 0, 0));
        }
    }
    let both = FENCE_WAIT | INCREMENT_VALUE;
    let (answer, _) = bench.kickoff(fd, 3, &partial, both);
    assert_eq!(answer.result, NvResult::InvalidSize);
    // The entries queued came from the inline input and the bytes after the struct, in order.
    let queued = bench.fake.queued();
    let sent: Vec<GpfifoEntry> = entries.iter().copied().map(GpfifoEntry).collect();
    assert_eq!(queued.len(), 2);
    assert!(
        queued
            .iter()
            .all(|(_, submission)| submission.entries == sent)
    );
}

/// One submission of exactly a queue's entries is taken; one more entry is refused, nothing
/// raised.
#[test]
fn a_submission_larger_than_a_queue_is_refused() {
    let mut bench = Bench::new();
    let (fd, _, syncpoint, channel) = bench.channel();
    let entries = entries_from(1, CHANNEL_QUEUE_ENTRIES + 1);
    let bytes = entry_bytes(&entries);
    let over = u32::try_from(CHANNEL_QUEUE_ENTRIES + 1).unwrap();
    let (answer, _) = bench.kickoff(fd, over, &bytes, FENCE_INCREMENT);
    assert_eq!(answer.result, NvResult::InvalidSize);
    assert_eq!(bench.syncpoints.fence(syncpoint).unwrap().value, 0);
    let exact = u32::try_from(CHANNEL_QUEUE_ENTRIES).unwrap();
    let (answer, deferral) = bench.kickoff(fd, exact, &bytes, FENCE_INCREMENT);
    assert_eq!((answer.result, deferral), (NvResult::Success, None));
    let queued = bench.fake.queued();
    assert_eq!(queued[0].0, channel);
    assert_eq!(queued[0].1.entries.len(), CHANNEL_QUEUE_ENTRIES);
}

/// Entries reach the front in the order the submissions came, each submission's in its own
/// order, across two channels.
#[test]
fn entries_reach_the_front_in_order() {
    let mut bench = Bench::new();
    let (a, _, _, first) = bench.channel();
    let (b, _, _, second) = bench.channel();
    bench.fake.clear();
    let sent = [
        (a, entries_from(0x10, 3)),
        (b, entries_from(0x20, 1)),
        (a, entries_from(0x30, 2)),
        (a, entries_from(0x40, 4)),
        (b, entries_from(0x50, 2)),
    ];
    for (fd, entries) in &sent {
        let (answer, _) = bench.submit(*fd, entries, 0, (0, 0));
        assert_eq!(answer.result, NvResult::Success);
    }
    let expected: Vec<(ChannelId, Vec<GpfifoEntry>)> = sent
        .iter()
        .map(|(fd, entries)| {
            let channel = if *fd == a { first } else { second };
            (channel, entries.iter().copied().map(GpfifoEntry).collect())
        })
        .collect();
    let queued: Vec<_> = bench
        .fake
        .queued()
        .into_iter()
        .map(|(channel, submission)| (channel, submission.entries))
        .collect();
    assert_eq!(queued, expected);
}

/// A queue holding one entry fewer than its bound takes a one-entry submission; a queue at
/// exactly its bound answers `Full`, and the submission waits.
#[test]
fn a_queue_at_its_bound_defers_and_one_below_does_not() {
    let mut bench = Bench::new();
    let (fd, _, syncpoint, channel) = bench.channel();
    bench.fake.hold(channel, CHANNEL_QUEUE_ENTRIES - 1);
    let (answer, deferral) = bench.submit(fd, &entries_from(1, 1), FENCE_INCREMENT, (0, 0));
    assert_eq!((answer.result, deferral), (NvResult::Success, None));
    // Now exactly at the bound.
    let entries = entries_from(2, 1);
    let (answer, deferral) = bench.submit(fd, &entries, FENCE_INCREMENT, (0, 0));
    // The fence is given back at once, the maximum raised.
    assert_eq!(answer.result, NvResult::Success);
    let raised = 2 * FENCE_INCREMENT_AMOUNT;
    assert_eq!(fence_of(&answer), (u32::from(syncpoint.get()), raised));
    let submission = Submission {
        entries: entries.iter().copied().map(GpfifoEntry).collect(),
        wait: None,
        increments: FENCE_INCREMENT_AMOUNT,
        wfi: true,
    };
    assert_eq!(
        deferral,
        Some(Deferral {
            channel,
            submission
        })
    );
    assert_eq!(bench.fake.calls().last(), Some(&Call::Full(channel)));
}

/// `Ioctl` 0x1B (entries read from the guest's memory at the struct's address), `Ioctl3`, and
/// `Ioctl2` 0x8 go to the policy.
#[test]
fn submissions_from_memory_go_to_the_policy() {
    let mut bench = Bench::new();
    let (fd, _, _, _) = bench.channel();
    let mut input = submit_struct(1, 0, 0, 0);
    input.extend(entry_bytes(&entries_from(1, 1)));
    let inline = entry_bytes(&entries_from(1, 1));
    for (variant, raw) in [
        (IoctlVariant::Ioctl, KICKOFF),
        (IoctlVariant::Ioctl3, KICKOFF),
        (IoctlVariant::Ioctl3, submit_number(1)),
        (IoctlVariant::Ioctl2, submit_number(1)),
        (IoctlVariant::Ioctl2, ALLOC_GPFIFO_EX2),
    ] {
        let (answer, _) = bench.call(variant, fd, raw, &input, &inline);
        assert_eq!(
            answer.result,
            NvResult::NotImplemented,
            "{variant:?} {raw:#x}"
        );
    }
    assert_eq!(bench.fake.queued(), []);
}

/// A server with no channel front leaves nvhost-gpu's ioctls and `BindChannel` to the policy,
/// as before this slice.
#[test]
fn without_a_channel_front_the_ioctls_go_to_the_policy() {
    let mut rig = Rig::new();
    let fd = rig.open_device(Device::NvhostGpu);
    let as_fd = rig.open_device(Device::NvhostAsGpu);
    rig.ioctl(as_fd, AS_ALLOC_AS_EX, &as_request(), 40);
    let not_implemented = ok(&[NvResult::NotImplemented.raw()]);
    let (reply, out) = rig.ioctl(fd, ALLOC_GPFIFO_EX2, &alloc_request(), 32);
    assert_eq!((reply, out), (not_implemented.clone(), vec![STALE; 32]));
    let (reply, _) = rig.ioctl(as_fd, AS_BIND_CHANNEL, &fd.to_le_bytes(), 4);
    assert_eq!(reply, not_implemented);
    assert_eq!(rig.state().as_gpu().bound(fd.cast_unsigned()), None);
    assert_eq!(rig.state().gpu().channel(fd.cast_unsigned()), None);
}

// ----- Through the server ----------------------------------------------------------------------

/// The nvsrv server with a fake channel front, over gsv's shared state.
struct Live {
    rig: Rig,
    fake: Arc<FakeChannels>,
}

impl Live {
    fn new() -> Self {
        let fake = Arc::new(FakeChannels::default());
        let shared = shared_state();
        let front = Arc::new(RecordingFront::default());
        let rig = Rig::over_with(
            UnknownPolicy::Strict,
            NvsrvState::new(),
            shared,
            front,
            Some(channel_front(&fake)),
        );
        Self { rig, fake }
    }

    /// A channel bound to a new address space with its GPU channel: its fd.
    fn channel(&mut self) -> i32 {
        let as_fd = self.rig.open_device(Device::NvhostAsGpu);
        let (reply, _) = self.rig.ioctl(as_fd, AS_ALLOC_AS_EX, &as_request(), 40);
        assert_eq!(reply, ok(&[0]));
        let fd = self.rig.open_device(Device::NvhostGpu);
        let (reply, _) = self.rig.ioctl(as_fd, AS_BIND_CHANNEL, &fd.to_le_bytes(), 4);
        assert_eq!(reply, ok(&[0]));
        let (reply, _) = self.rig.ioctl(fd, ALLOC_GPFIFO_EX2, &alloc_request(), 32);
        assert_eq!(reply, ok(&[0]));
        fd
    }

    /// The GPU channel of fd `fd`.
    fn made(&self, fd: i32) -> ChannelId {
        let channel = self.rig.state().gpu().channel(fd.cast_unsigned()).unwrap();
        ChannelId(channel.made.unwrap().channel)
    }

    /// A new session on the server.
    fn session(&mut self) -> ObjectId {
        let session = self.rig.kernel.new_object();
        self.rig.server.accept(session).unwrap();
        session
    }

    /// `SubmitGPFIFO` 0x1B through `Ioctl2` on `session`, with `entries` inline and an out
    /// buffer of 24 stale bytes: how it was handled, the message buffer, and where the out
    /// buffer is.
    fn submit_on(
        &mut self,
        session: ObjectId,
        fd: i32,
        entries: &[u64],
        flags: u32,
    ) -> (Handled, Vec<u8>, u64) {
        self.submit_with(session, fd, entries, flags, (0, 0))
    }

    /// As [`Live::submit_on`], with fence (`id`, `value`).
    fn submit_with(
        &mut self,
        session: ObjectId,
        fd: i32,
        entries: &[u64],
        flags: u32,
        (id, value): (u32, u32),
    ) -> (Handled, Vec<u8>, u64) {
        let out_at = self.rig.place(&[STALE; 24]);
        self.submit_out(session, fd, entries, flags, (id, value), out_at)
    }

    /// As [`Live::submit_with`], with the out buffer at `out_at`, which need not be mapped.
    fn submit_out(
        &mut self,
        session: ObjectId,
        fd: i32,
        entries: &[u64],
        flags: u32,
        (id, value): (u32, u32),
        out_at: u64,
    ) -> (Handled, Vec<u8>, u64) {
        let count = u32::try_from(entries.len()).unwrap();
        let input = submit_struct(count, flags, id, value);
        let inline = entry_bytes(entries);
        let inline_at = self.rig.place(&inline);
        let (handled, buffer) =
            self.kickoff_at(session, fd, &input, (inline_at, inline.len()), out_at);
        (handled, buffer, out_at)
    }

    /// `SubmitGPFIFO` 0x1B through `Ioctl2` on `session` and `fd` with `input`, in buffer 1 of
    /// `inline_len` bytes at `inline_at`, and a 24-byte out buffer at `out_at`; neither need be
    /// mapped. How it was handled, and the message buffer.
    fn kickoff_at(
        &mut self,
        session: ObjectId,
        fd: i32,
        input: &[u8],
        (inline_at, inline_len): (u64, usize),
        out_at: u64,
    ) -> (Handled, Vec<u8>) {
        let input_at = self.rig.place(input);
        let request = cmif(IOCTL2, &fd_and(fd, KICKOFF));
        let request = Auto::alias(input_at, len16(input.len())).send(request, 0);
        // In buffer 1 as `Auto::alias(..).send(request, 1)` writes it, with a length that may
        // pass 16 bits (a submission of up to a whole queue's entries).
        let request = request
            .send_buffer(starquake_ipc::BufferDescriptor {
                address: inline_at,
                size: u64::try_from(inline_len).unwrap(),
                flags: 0,
            })
            .static_descriptor(starquake_ipc::StaticDescriptor {
                counter: 1,
                address: 0,
                size: 0,
            });
        let request = Auto::alias(out_at, 24).receive(request);
        let mut buffer = vec![STALE; BUFFER_SIZE];
        request.build(&mut buffer).unwrap();
        let handled = self
            .rig
            .server
            .handle(&mut self.rig.kernel, session, &mut buffer)
            .unwrap();
        (handled, buffer)
    }

    /// Pushes room for `channel` and runs the timer thread's drain; what it recorded.
    fn room(&mut self, channel: ChannelId) -> Vec<EventKind> {
        let room = GpuCompletion::GpuQueueRoom { channel };
        Completions::push(&**self.rig.shared.queue(), room);
        let mut host = Host::default();
        self.rig.shared.drain(&mut host);
        host.0
    }

    /// Delivers every completed reply: each one's session and its answer.
    fn deliver(&mut self) -> Vec<(ObjectId, Answer)> {
        let mut delivered = Vec::new();
        while let Some(completed) = self.rig.server.take_completed() {
            let session = completed.session();
            let mut buffer = vec![STALE; BUFFER_SIZE];
            self.rig
                .server
                .deliver(&mut self.rig.kernel, completed, &mut buffer)
                .unwrap();
            delivered.push((session, answer(&buffer, 1)));
        }
        delivered
    }

    fn waiting(&self, channel: ChannelId) -> usize {
        let shared = &self.rig.shared;
        let mut locked = shared.lock();
        locked.submits().waiting(channel)
    }

    fn waiting_entries(&self, channel: ChannelId) -> usize {
        let shared = &self.rig.shared;
        let mut locked = shared.lock();
        locked.submits().waiting_entries(channel)
    }
}

/// A full queue: the request is deferred, its fence written at once, and the room report the
/// drain takes queues it and completes its reply with `Success`.
#[test]
fn a_full_queue_defers_and_room_completes() {
    let mut live = Live::new();
    let fd = live.channel();
    let channel = live.made(fd);
    live.fake.hold(channel, CHANNEL_QUEUE_ENTRIES);
    let session = live.rig.session();
    let entries = entries_from(0x70, 2);
    let (handled, _, out_at) = live.submit_on(session, fd, &entries, FENCE_INCREMENT);
    assert!(matches!(handled, Handled::Deferred(_)));
    // Nothing completed yet.
    assert_eq!(live.deliver(), []);
    assert_eq!(live.waiting(channel), 1);
    // Room for nothing: still waiting.
    live.room(channel);
    assert_eq!(live.deliver(), []);
    assert_eq!(live.fake.calls().last(), Some(&Call::Full(channel)));
    // Room: queued, and the reply completes.
    live.fake.hold(channel, CHANNEL_QUEUE_ENTRIES - 2);
    let recorded = live.room(channel);
    assert_eq!(recorded.len(), 1);
    assert_eq!(live.deliver(), [(session, ok(&[NvResult::Success.raw()]))]);
    assert_eq!(live.waiting(channel), 0);
    let queued = live.fake.queued();
    let sent: Vec<GpfifoEntry> = entries.iter().copied().map(GpfifoEntry).collect();
    assert_eq!(queued.last().map(|(_, s)| &s.entries), Some(&sent));
    // The fence was written when the request came.
    let syncpoint = u32::from(FIRST_ALLOCATABLE);
    assert_eq!(
        live.rig.memory(out_at, 24),
        submit_struct(2, 0, syncpoint, FENCE_INCREMENT_AMOUNT)
    );
}

/// The out buffer of a submission that waits holds its fence before the reply is delivered.
#[test]
fn a_waiting_submission_writes_its_fence_at_once() {
    let mut live = Live::new();
    let fd = live.channel();
    live.fake.hold(live.made(fd), CHANNEL_QUEUE_ENTRIES);
    let session = live.rig.session();
    let flags = INCREMENT_VALUE;
    let (handled, _, out_at) = live.submit_on(session, fd, &entries_from(1, 1), flags);
    assert!(matches!(handled, Handled::Deferred(_)));
    let syncpoint = u32::from(FIRST_ALLOCATABLE);
    // Bit 8 with a fence value of 0 (the request's): no increment, the fence at 0.
    assert_eq!(
        live.rig.memory(out_at, 24),
        submit_struct(1, 0, syncpoint, 0)
    );
}

/// A submission behind a waiting one waits too, though the front has room, and the two reach
/// the front in the order they came once room is reported; room for another channel leaves them
/// waiting.
#[test]
fn a_submission_behind_a_waiting_one_waits_too() {
    let mut live = Live::new();
    let fd = live.channel();
    let other_fd = live.channel();
    let channel = live.made(fd);
    let other = live.made(other_fd);
    live.fake.hold(channel, CHANNEL_QUEUE_ENTRIES);
    let first_session = live.rig.session();
    let second_session = live.session();
    let first = entries_from(0x10, 1);
    let second = entries_from(0x20, 1);
    let (handled, _, _) = live.submit_on(first_session, fd, &first, 0);
    assert!(matches!(handled, Handled::Deferred(_)));
    // Room, but the first still waits in gsv: the second must not pass it.
    live.fake.hold(channel, 0);
    let (handled, _, _) = live.submit_on(second_session, fd, &second, 0);
    assert!(matches!(handled, Handled::Deferred(_)));
    // The push tried the queue once more: both went, in order, and both replies complete.
    let queued: Vec<_> = live
        .fake
        .queued()
        .into_iter()
        .map(|(_, s)| s.entries)
        .collect();
    let tail = queued[queued.len() - 2..].to_vec();
    let as_entries = |entries: &[u64]| -> Vec<GpfifoEntry> {
        entries.iter().copied().map(GpfifoEntry).collect()
    };
    assert_eq!(tail, vec![as_entries(&first), as_entries(&second)]);
    let delivered = live.deliver();
    assert_eq!(
        delivered,
        [(first_session, ok(&[0])), (second_session, ok(&[0]))]
    );
    // Now with the queue full for both: room for the other channel changes nothing.
    live.fake.hold(channel, CHANNEL_QUEUE_ENTRIES);
    live.submit_on(first_session, fd, &first, 0);
    live.submit_on(second_session, fd, &second, 0);
    live.room(other);
    assert_eq!(live.deliver(), []);
    assert_eq!(live.waiting(channel), 2);
    // Room for one: the oldest goes, the other waits for the next report.
    live.fake.hold(channel, CHANNEL_QUEUE_ENTRIES - 1);
    live.room(channel);
    assert_eq!(live.deliver(), [(first_session, ok(&[0]))]);
    assert_eq!(live.waiting(channel), 1);
    live.fake.hold(channel, 0);
    live.room(channel);
    assert_eq!(live.deliver(), [(second_session, ok(&[0]))]);
}

/// Closing a channel answers each waiting reply `NotImplemented`, oldest first, gives the front
/// nothing, then destroys the GPU channel and frees the syncpoint.
#[test]
fn closing_a_channel_answers_its_waiting_submissions() {
    let mut live = Live::new();
    let fd = live.channel();
    let channel = live.made(fd);
    live.fake.hold(channel, CHANNEL_QUEUE_ENTRIES);
    let first = live.rig.session();
    let second = live.session();
    live.submit_on(first, fd, &entries_from(1, 1), 0);
    let (_, _, out_at) = live.submit_on(second, fd, &entries_from(2, 1), 0);
    live.fake.clear();
    // Both sessions wait; a third closes the fd.
    let closer = live.session();
    let reply = answer(
        &live.rig.send_on(closer, &cmif(CLOSE, &fd.to_le_bytes())),
        1,
    );
    assert_eq!(reply, ok(&[0]));
    let not_implemented = ok(&[NvResult::NotImplemented.raw()]);
    assert_eq!(
        live.deliver(),
        [(first, not_implemented.clone()), (second, not_implemented)]
    );
    assert_eq!(live.fake.calls(), [Call::Destroy(channel)]);
    assert_eq!(live.waiting(channel), 0);
    let syncpoint = SyncpointId::new(FIRST_ALLOCATABLE).unwrap();
    assert!(!live.rig.shared.syncpoints().is_allocated(syncpoint));
    // The out buffer keeps the fence it was given.
    let id = u32::from(FIRST_ALLOCATABLE);
    assert_eq!(live.rig.memory(out_at, 24), submit_struct(1, 0, id, 0));
    // A room report after the close finds nothing.
    live.room(channel);
    assert_eq!(live.deliver(), []);
}

// ----- Events ----------------------------------------------------------------------------------

/// Events 1, 2, 3 and 7 are answered with a copied handle and `Success`, 7 being 3's, each made
/// once; any other id is `BadParameter` and no handle.
#[test]
fn events_1_2_3_and_7_are_answered() {
    let mut live = Live::new();
    let fd = live.rig.open_device(Device::NvhostGpu);
    let query = |live: &mut Live, id: u32| {
        let mut data = fd.to_le_bytes().to_vec();
        data.extend(id.to_le_bytes());
        let buffer = live.rig.send(&cmif(QUERY_EVENT, &data));
        let handles = Message::parse_reply(&buffer, SessionKind::Session)
            .unwrap()
            .copy_handles;
        (answer(&buffer, 1), handles)
    };
    let success = ok(&[NvResult::Success.raw()]);
    let mut handles = Vec::new();
    for id in [
        event::SM_EXCEPTION_INT,
        event::SM_EXCEPTION_PAUSE,
        event::ERROR_NOTIFIER,
    ] {
        let (reply, copied) = query(&mut live, id);
        assert_eq!(reply, success);
        assert_eq!(copied.len(), 1);
        handles.push(copied[0]);
    }
    let events: Vec<ObjectId> = live
        .rig
        .state()
        .gpu()
        .channel(fd.cast_unsigned())
        .unwrap()
        .events
        .iter()
        .map(|event| event.unwrap())
        .collect();
    let distinct: std::collections::BTreeSet<_> = events.iter().collect();
    assert_eq!(distinct.len(), EVENT_SLOTS);
    // 7 is 3's event, and asking again makes nothing new.
    let (reply, _) = query(&mut live, event::NATIVE_ERROR);
    assert_eq!(reply, success);
    let (reply, _) = query(&mut live, event::SM_EXCEPTION_INT);
    assert_eq!(reply, success);
    let after: Vec<ObjectId> = live
        .rig
        .state()
        .gpu()
        .channel(fd.cast_unsigned())
        .unwrap()
        .events
        .iter()
        .map(|event| event.unwrap())
        .collect();
    assert_eq!(after, events);
    assert_eq!(
        event_slot(event::NATIVE_ERROR),
        event_slot(event::ERROR_NOTIFIER)
    );
    for id in [0, 4, 5, 6, 8, u32::MAX] {
        let (reply, copied) = query(&mut live, id);
        assert_eq!(reply, ok(&[NvResult::BadParameter.raw()]), "{id}");
        assert_eq!(copied, Vec::<u32>::new());
    }
    // The fd checks come first.
    let mut data = (-1_i32).to_le_bytes().to_vec();
    data.extend(1_u32.to_le_bytes());
    let reply = answer(&live.rig.send(&cmif(QUERY_EVENT, &data)), 1);
    assert_eq!(reply, ok(&[NvResult::InvalidState.raw()]));
}

// ----- Saved state -----------------------------------------------------------------------------

/// The saved layout of nvhost-gpu's channels, as `postcard` writes it: the channel count, each
/// fd, its syncpoint, its GPU channel and space, its classes, its three events (index and
/// generation each), and its settings (client data, error notifier, preemption modes, zcull
/// bind). Each value distinct and non-zero.
#[test]
fn a_channel_s_saved_layout_is_pinned() {
    let channel = Channel {
        syncpoint: Some(0x21),
        made: Some(Made {
            channel: 0x31,
            space: AsId(0x41),
        }),
        classes: vec![clc797::CLASS, class::CTX_DMA],
        events: [
            Some(ObjectId::from_parts(0x51, 0x52)),
            None,
            Some(ObjectId::from_parts(0x53, 0x54)),
        ],
        client_data: 0x6162,
        error_notifier: Some(ErrorNotifier {
            offset: 0x63,
            size: 0x64,
            mem: 0x65,
        }),
        preemption: Preemption {
            graphics: preemption::GRAPHICS_MAX,
            compute: preemption::COMPUTE_MAX,
        },
        zcull: Some(ZcullBind {
            gpu_va: 0x66,
            mode: 0x67,
        }),
    };
    let gpu = NvhostGpu::try_from(SavedNvhostGpu {
        channels: BTreeMap::from([(0x0B, channel)]),
    })
    .unwrap();
    let bytes = postcard::to_allocvec(&gpu).unwrap();
    let mut expected = vec![1, 0x0B];
    // Syncpoint: some, 0x21. GPU channel: some, 0x31, space 0x41.
    expected.extend([1, 0x21, 1, 0x31, 0x41]);
    // Two classes: 0xC797 and 0xB0B5 as varints.
    expected.extend([2, 0x97, 0x8F, 0x03, 0xB5, 0xE1, 0x02]);
    // Events: some (0x51, 0x52), none, some (0x53, 0x54).
    expected.extend([1, 0x51, 0x52, 0, 1, 0x53, 0x54]);
    // Client data 0x6162 as a varint; error notifier: some, offset 0x63, size 0x64, mem 0x65;
    // preemption modes 1 and 2; zcull bind: some, address 0x66, mode 0x67.
    expected.extend([0xE2, 0xC2, 0x01, 1, 0x63, 0x64, 0x65, 1, 2, 1, 0x66, 0x67]);
    assert_eq!(bytes, expected);
    let back: NvhostGpu = postcard::from_bytes(&bytes).unwrap();
    assert_eq!(back, gpu);
}

/// The saved layout of a bound channel, as `postcard` writes as-gpu's part and nvhost-gpu's part
/// of the nvsrv state: the view (fd 3, space 6, its layout), the space made (6, its layout), the
/// binding (fd 4 to space 6), then the channel (fd 4, syncpoint 1, GPU channel 0x30 made in space
/// 6, no classes, no events). Five spaces are made first so the space's id is none of the fds;
/// the space's id repeats because the binding and the GPU channel are of that one space. Each
/// value is non-zero.
#[test]
fn a_bound_channel_s_saved_layout_is_pinned() {
    let mut bench = Bench::new();
    for _ in 0..5 {
        bench.memory.create_address_space(&bench.vm).unwrap();
    }
    bench.open(Device::Nvmap);
    bench.open(Device::NvhostCtrlGpu);
    let (as_fd, space) = bench.space();
    let fd = bench.open(Device::NvhostGpu);
    assert_eq!((as_fd, space, fd), (3, AsId(6), 4));
    assert_eq!(bench.bind(as_fd, fd), NvResult::Success);
    assert_eq!(bench.alloc(fd), (u32::from(FIRST_ALLOCATABLE), 0));
    // A layout: start 0x2000, split 0x5000, end 0x9000, big page size 0x10000, as varints.
    let layout = [
        0x80, 0x40, 0x80, 0xA0, 0x01, 0x80, 0xA0, 0x02, 0x80, 0x80, 0x04,
    ];
    // One view: fd 3, space 6, its layout; one space made: 6, its layout; one binding: fd 4,
    // space 6.
    let mut expected = vec![1, 3, 6];
    expected.extend(layout);
    expected.extend([1, 6]);
    expected.extend(layout);
    expected.extend([1, 4, 6]);
    assert_eq!(
        postcard::to_allocvec(bench.state.as_gpu()).unwrap(),
        expected
    );
    // One channel: fd 4, syncpoint some 1, GPU channel some 0x30 in space 6, no classes, no
    // events, client data 0, no error notifier, both preemption modes the default, no zcull
    // bind.
    let modes = u8::try_from(preemption::DEFAULT).unwrap();
    let expected = [
        1,
        4,
        1,
        FIRST_ALLOCATABLE,
        1,
        0x30,
        6,
        0,
        0,
        0,
        0,
        0,
        0,
        modes,
        modes,
        0,
    ];
    assert_eq!(postcard::to_allocvec(bench.state.gpu()).unwrap(), expected);
    // The whole state reads back as it was.
    let bytes = bench.state.save_state().unwrap();
    let mut loaded = NvsrvState::new();
    loaded.load_state(&bytes).unwrap();
    assert_eq!(loaded, bench.state);
}

/// A channel with each of the fields set, from a bench, as saved.
fn saved_channel(channel: Channel) -> SavedNvhostGpu {
    SavedNvhostGpu {
        channels: BTreeMap::from([(7, channel)]),
    }
}

/// Each saved channel table the ioctls cannot reach, and its error, first fault first; and the
/// edges that are reachable.
#[test]
fn a_saved_channel_table_no_ioctl_reaches_is_refused() {
    let made = Some(Made {
        channel: 3,
        space: AsId(1),
    });
    let with = |syncpoint: Option<u8>, made: Option<Made>, classes: Vec<u32>| Channel {
        syncpoint,
        made,
        classes,
        ..Channel::default()
    };
    let reserved = RESERVED_SYNCPOINTS[0];
    let count = u8::try_from(SYNCPOINT_COUNT).unwrap();
    let refused = [
        (
            with(Some(0), None, vec![]),
            GpuStateError::Syncpoint { fd: 7, id: 0 },
        ),
        (
            with(Some(count), None, vec![]),
            GpuStateError::Syncpoint { fd: 7, id: count },
        ),
        (
            with(Some(reserved), None, vec![]),
            GpuStateError::Syncpoint {
                fd: 7,
                id: reserved,
            },
        ),
        (
            with(None, made, vec![]),
            GpuStateError::MadeWithoutSyncpoint { fd: 7 },
        ),
        (
            with(Some(1), None, vec![0]),
            GpuStateError::Class { fd: 7, class: 0 },
        ),
        (
            with(Some(1), None, vec![0x1234]),
            GpuStateError::Class {
                fd: 7,
                class: 0x1234,
            },
        ),
        (
            with(Some(1), None, vec![clc797::CLASS, class::CTX_3D]),
            GpuStateError::Class {
                fd: 7,
                class: class::CTX_3D,
            },
        ),
        // Two faults: the syncpoint is reported before the GPU channel and the class.
        (
            with(Some(0), None, vec![0]),
            GpuStateError::Syncpoint { fd: 7, id: 0 },
        ),
        (
            with(None, made, vec![0]),
            GpuStateError::MadeWithoutSyncpoint { fd: 7 },
        ),
    ];
    for (channel, error) in refused {
        assert_eq!(
            NvhostGpu::try_from(saved_channel(channel)),
            Err(error),
            "{error}"
        );
    }
    // The edges: the first and last allocatable syncpoints; every slot filled.
    let last = (FIRST_ALLOCATABLE..count)
        .rev()
        .find(|id| !RESERVED_SYNCPOINTS.contains(id))
        .unwrap();
    let full = vec![
        cl902d::CLASS,
        clc797::CLASS,
        clc7c0::CLASS,
        cla140::CLASS,
        clc7b5::CLASS,
        clc76f::CLASS,
    ];
    for channel in [
        with(Some(FIRST_ALLOCATABLE), made, vec![]),
        with(Some(last), None, full),
    ] {
        assert!(NvhostGpu::try_from(saved_channel(channel)).is_ok());
    }
}

/// Two saved channels sharing a syncpoint, then a GPU channel, are refused; both shared, the
/// syncpoint is reported.
#[test]
fn two_saved_channels_share_neither_a_syncpoint_nor_a_gpu_channel() {
    let made = Some(Made {
        channel: 3,
        space: AsId(1),
    });
    let with = |syncpoint: Option<u8>, made: Option<Made>, classes: Vec<u32>| Channel {
        syncpoint,
        made,
        classes,
        ..Channel::default()
    };
    // Two channels sharing a syncpoint, then a GPU channel.
    let pair = |a: Channel, b: Channel| SavedNvhostGpu {
        channels: BTreeMap::from([(4, a), (9, b)]),
    };
    assert_eq!(
        NvhostGpu::try_from(pair(
            with(Some(5), None, vec![]),
            with(Some(5), None, vec![])
        )),
        Err(GpuStateError::SharedSyncpoint {
            first: 4,
            second: 9,
            id: 5
        })
    );
    assert_eq!(
        NvhostGpu::try_from(pair(
            with(Some(5), made, vec![]),
            with(Some(6), made, vec![])
        )),
        Err(GpuStateError::SharedChannel {
            first: 4,
            second: 9,
            channel: 3
        })
    );
    // Both shared: the syncpoint is reported.
    assert_eq!(
        NvhostGpu::try_from(pair(
            with(Some(5), made, vec![]),
            with(Some(5), made, vec![])
        )),
        Err(GpuStateError::SharedSyncpoint {
            first: 4,
            second: 9,
            id: 5
        })
    );
    // Through postcard, a refused table is refused (the struct's one field is the map).
    let bytes =
        postcard::to_allocvec(&BTreeMap::from([(7_u32, with(Some(0), None, vec![]))])).unwrap();
    assert!(postcard::from_bytes::<NvhostGpu>(&bytes).is_err());
}

#[test]
fn a_refused_channel_table_is_said_in_plain_words() {
    let said = [
        (
            GpuStateError::Syncpoint { fd: 7, id: 0 },
            "channel fd 7 has syncpoint 0, which a channel cannot be given",
        ),
        (
            GpuStateError::MadeWithoutSyncpoint { fd: 7 },
            "channel fd 7 has a GPU channel but no syncpoint",
        ),
        (
            GpuStateError::Class {
                fd: 7,
                class: 0x1234,
            },
            "channel fd 7 has class 0x1234, which it cannot have",
        ),
        (
            GpuStateError::SharedSyncpoint {
                first: 4,
                second: 9,
                id: 5,
            },
            "channel fds 4 and 9 share syncpoint 5",
        ),
        (
            GpuStateError::SharedChannel {
                first: 4,
                second: 9,
                channel: 3,
            },
            "channel fds 4 and 9 share GPU channel 3",
        ),
        (
            GpuStateError::SyncpointNotAllocated { fd: 7, id: 5 },
            "channel fd 7 has syncpoint 5, which is not allocated",
        ),
    ];
    for (error, words) in said {
        assert_eq!(error.to_string(), words);
    }
    assert_eq!(GpuError.to_string(), "the GPU front refused the request");
}

/// The relations between the channels and the open files and bindings, each refused with its
/// error, and the real states live operations leave accepted: a channel whose space's last view
/// closed, a channel unbound with only events.
#[test]
fn the_channels_relations_to_the_files_are_checked() {
    let check = |state: &NvsrvState| NvsrvState::try_from(saved(state));
    let mut bench = Bench::new();
    let (fd, as_fd, _, _) = bench.channel();
    let events_only = bench.open(Device::NvhostGpu);
    bench.state.set_event(
        events_only,
        event::SM_EXCEPTION_INT,
        ObjectId::from_parts(8, 9),
    );
    bench.close(as_fd);
    // The real state: a channel whose space's last view closed, and one with only events.
    let state = bench.state.clone();
    assert_eq!(check(&state).as_ref(), Ok(&state));
    let fd = fd.cast_unsigned();
    let events_only = events_only.cast_unsigned();

    // A channel's fd closed, or open as another device.
    let mut closed = state.clone();
    closed.files.remove(&events_only);
    assert_eq!(
        check(&closed),
        Err(NvsrvStateError::NotAGpuChannel { fd: events_only })
    );
    let mut other = state.clone();
    other.files.insert(fd, Device::Nvmap);
    assert_eq!(check(&other), Err(NvsrvStateError::NotAGpuChannel { fd }));
    // A binding on an fd that is not an nvhost-gpu (and has no channel, which would be
    // reported first).
    let mut unbound = state.clone();
    unbound.gpu.channels.remove(&fd);
    unbound.files.insert(fd, Device::NvhostCtrl);
    assert_eq!(
        check(&unbound),
        Err(NvsrvStateError::NotABoundChannel { fd })
    );
    // A GPU channel made in a space its fd is not bound to.
    let mut moved = state.clone();
    if let Some(made) = moved.gpu.channels.get_mut(&fd).unwrap().made.as_mut() {
        made.space = AsId(made.space.0 + 1);
    }
    assert_eq!(check(&moved), Err(NvsrvStateError::MadeUnbound { fd }));
    // Both a closed channel fd and a made channel unbound: the file is reported.
    let mut both = moved;
    both.files.remove(&fd);
    assert_eq!(check(&both), Err(NvsrvStateError::NotAGpuChannel { fd }));
    let said = [
        (
            NvsrvStateError::NotAGpuChannel { fd: 3 },
            "fd 3 has a GPU channel but is not an open nvhost-gpu",
        ),
        (
            NvsrvStateError::NotABoundChannel { fd: 3 },
            "fd 3 is bound to an address space but is not an open nvhost-gpu",
        ),
        (
            NvsrvStateError::MadeUnbound { fd: 3 },
            "channel fd 3's GPU channel was made in an address space it is not bound to",
        ),
    ];
    for (error, words) in said {
        assert_eq!(error.to_string(), words);
    }
}

#[test]
fn check_syncpoints_reports_the_first_unallocated_one() {
    let mut bench = Bench::new();
    let (a, _, first, _) = bench.channel();
    let (b, _, second, _) = bench.channel();
    let gpu = bench.state.gpu().clone();
    assert_eq!(gpu.check_syncpoints(&bench.syncpoints), Ok(()));
    bench.syncpoints.free(second).unwrap();
    assert_eq!(
        gpu.check_syncpoints(&bench.syncpoints),
        Err(GpuStateError::SyncpointNotAllocated {
            fd: b.cast_unsigned(),
            id: second.get()
        })
    );
    bench.syncpoints.free(first).unwrap();
    assert_eq!(
        gpu.check_syncpoints(&bench.syncpoints),
        Err(GpuStateError::SyncpointNotAllocated {
            fd: a.cast_unsigned(),
            id: first.get()
        })
    );
}

/// A server saved with a bound channel, its classes and events restores and carries on: a
/// submission after the restore goes to the same GPU channel with the next fence.
#[test]
fn a_restored_server_carries_on_with_a_channel() {
    let mut live = Live::new();
    let fd = live.channel();
    let channel = live.made(fd);
    let (reply, _) = live
        .rig
        .ioctl(fd, ALLOC_OBJ_CTX, &obj_request(clc797::CLASS, 0), 16);
    assert_eq!(reply, ok(&[0]));
    let session = live.rig.session();
    live.submit_on(session, fd, &entries_from(1, 1), FENCE_INCREMENT);
    let state = live.rig.state().clone();
    let saved = live.rig.server.save().unwrap();

    let restored = nvsrv_server(
        Arc::clone(&live.rig.shared),
        as_front(&live.rig.front),
        UnknownPolicy::Strict,
    );
    let family = NvsrvFamily::new(Arc::clone(&live.rig.shared), as_front(&live.rig.front))
        .with_channel_front(channel_front(&live.fake));
    live.rig.server = restored
        .with_family(family)
        .with_service_id(live.rig.server.service_id());
    live.rig.server.restore(&saved).unwrap();
    assert_eq!(live.rig.state(), &state);
    live.fake.clear();
    let (handled, _, out_at) = live.submit_on(session, fd, &entries_from(2, 1), FENCE_INCREMENT);
    assert_eq!(handled, Handled::Replied);
    let id = u32::from(FIRST_ALLOCATABLE);
    let fence = 2 * FENCE_INCREMENT_AMOUNT;
    assert_eq!(live.rig.memory(out_at, 24), submit_struct(1, 0, id, fence));
    assert_eq!(live.fake.queued()[0].0, channel);
    // The class slot is still taken after the restore.
    let (reply, _) = live
        .rig
        .ioctl(fd, ALLOC_OBJ_CTX, &obj_request(clc797::CLASS, 0), 16);
    assert_eq!(reply, ok(&[NvResult::AlreadyAllocated.raw()]));
}

/// `Ioctl3` on a channel goes to the policy through the server, and nothing is written.
#[test]
fn ioctl3_on_a_channel_goes_to_the_policy() {
    let mut live = Live::new();
    let fd = live.channel();
    let (reply, out, inline) = live
        .rig
        .ioctl3(fd, ALLOC_GPFIFO_EX2, &alloc_request(), (32, 8));
    assert_eq!(reply, ok(&[NvResult::NotImplemented.raw()]));
    assert_eq!((out, inline), (vec![STALE; 32], vec![STALE; 8]));
}

// ----- The waiting submissions' snapshot -------------------------------------------------------

/// A token id, as a snapshot holds it.
fn token(id: u8) -> TokenId {
    postcard::from_bytes(&[id]).unwrap()
}

/// A waiting-submissions snapshot of `list`, its header laid out by hand: the magic, then the
/// version as a varint.
fn snapshot_of(list: &[SavedSubmit]) -> Vec<u8> {
    let mut bytes = SNAPSHOT_MAGIC.to_vec();
    bytes.push(u8::try_from(SNAPSHOT_VERSION).unwrap());
    bytes.extend(postcard::to_allocvec(list).unwrap());
    bytes
}

/// Two channels, three waiting submissions: on the first channel tokens 1 and 3, on the second
/// token 2 (token 0 is spent first on a submission that completes), each with distinct values.
/// The live and its two GPU channels.
fn three_waiting() -> (Live, ChannelId, ChannelId) {
    let mut live = Live::new();
    let a = live.channel();
    let b = live.channel();
    let (first, second) = (live.made(a), live.made(b));
    // Token 0: deferred and completed.
    let spend = live.rig.session();
    live.fake.hold(first, CHANNEL_QUEUE_ENTRIES);
    live.submit_on(spend, a, &[0x10], 0);
    live.fake.hold(first, 0);
    live.room(first);
    assert_eq!(live.deliver().len(), 1);
    live.fake.hold(first, CHANNEL_QUEUE_ENTRIES);
    live.fake.hold(second, CHANNEL_QUEUE_ENTRIES);
    let sessions = [live.rig.session(), live.session(), live.session()];
    let wait = FENCE_WAIT | FENCE_INCREMENT;
    live.submit_with(sessions[0], a, &[0x11], wait, (0x21, 0x22));
    let by_value = INCREMENT_VALUE | SUPPRESS_WFI;
    live.submit_with(sessions[1], b, &[0x12, 0x13], by_value, (0, 0x23));
    live.submit_with(sessions[2], a, &[0x14], 0, (0, 0));
    (live, first, second)
}

/// The waiting table's saved layout: the header, the count, then each submission in channel
/// order, oldest first: channel, token id (an option: none once its request's session closed),
/// entries, wait fence, increments, wait-for-idle.
#[test]
fn the_snapshot_layout_is_pinned() {
    let (mut live, first, second) = three_waiting();
    assert_eq!((first, second), (ChannelId(0x30), ChannelId(0x31)));
    // Token 3's session closes: its submission is saved without a token.
    let (_, closed) = live
        .rig
        .server
        .pending()
        .find(|&(id, _)| id == token(3))
        .unwrap();
    live.rig.server.close(closed).unwrap();
    let bytes = live
        .rig
        .shared
        .lock()
        .submits()
        .save(&live.rig.server)
        .unwrap();
    let mut expected = b"SQSB".to_vec();
    expected.extend([1, 3]);
    // Channel 0x30, token some 1, one entry 0x11, waits on (0x21, 0x22), 2 increments, wait for
    // idle.
    expected.extend([0x30, 1, 1, 1, 0x11, 1, 0x21, 0x22, 2, 1]);
    // Channel 0x30, no token, one entry 0x14, no wait, no increments, wait for idle.
    expected.extend([0x30, 0, 1, 0x14, 0, 0, 1]);
    // Channel 0x31, token some 2, entries 0x12 and 0x13, no wait, 0x23 increments, no wait for
    // idle.
    expected.extend([0x31, 1, 2, 2, 0x12, 0x13, 0, 0x23, 0]);
    assert_eq!(bytes, expected);
}

/// A server and its waiting table saved together restore, and the restored table, once its
/// channels are tried, queues the submissions in order and completes their replies through the
/// restored server.
#[test]
fn a_saved_waiting_table_restores_and_carries_on() {
    let (mut live, first, second) = three_waiting();
    let table = live
        .rig
        .shared
        .lock()
        .submits()
        .save(&live.rig.server)
        .unwrap();
    let server = live.rig.server.save().unwrap();
    let mut pending: Vec<(TokenId, ObjectId)> = live.rig.server.pending().collect();
    pending.sort();
    let sessions: Vec<ObjectId> = pending.iter().map(|&(_, session)| session).collect();
    assert_eq!(sessions.len(), 3);

    let family = NvsrvFamily::new(Arc::clone(&live.rig.shared), as_front(&live.rig.front))
        .with_channel_front(channel_front(&live.fake));
    let mut restored = nvsrv_server(
        Arc::clone(&live.rig.shared),
        as_front(&live.rig.front),
        UnknownPolicy::Strict,
    )
    .with_family(family)
    .with_service_id(live.rig.server.service_id());
    restored.restore(&server).unwrap();
    let front = channel_front(&live.fake);
    let submits = Submits::restore(&table, &restored, &front).unwrap();
    assert_eq!(&submits, live.rig.shared.lock().submits());
    assert_eq!(submits.save(&restored).unwrap(), table);
    live.rig.server = restored;
    *live.rig.shared.lock().submits() = submits;
    // Nothing went to the front on the restore.
    live.fake.clear();
    assert_eq!(live.deliver(), []);
    live.fake.hold(first, 0);
    live.fake.hold(second, 0);
    live.rig.shared.lock().submits().retry_all();
    let queued: Vec<(ChannelId, Vec<GpfifoEntry>)> = live
        .fake
        .queued()
        .into_iter()
        .map(|(channel, submission)| (channel, submission.entries))
        .collect();
    let entries = |raw: &[u64]| raw.iter().copied().map(GpfifoEntry).collect::<Vec<_>>();
    assert_eq!(
        queued,
        [
            (first, entries(&[0x11])),
            (first, entries(&[0x14])),
            (second, entries(&[0x12, 0x13])),
        ]
    );
    // Tokens 1, 2, 3 were taken by the first, second and third sessions; channel order
    // completes 1, 3, then 2.
    let delivered: Vec<ObjectId> = live.deliver().into_iter().map(|(s, _)| s).collect();
    assert_eq!(delivered, [sessions[0], sessions[2], sessions[1]]);
    assert!(live.rig.shared.lock().submits().is_empty());
}

/// A waiting submission on `channel` for request `id`, one entry, no wait.
fn waiting_entry(channel: u32, id: u8) -> SavedSubmit {
    SavedSubmit {
        channel,
        token: Some(token(id)),
        entries: vec![1],
        wait: None,
        increments: 0,
        wfi: true,
    }
}

/// Each snapshot `Submits::restore` refuses while requests 1, 2 and 3 are pending, and its
/// error, first fault first.
fn refused_tables() -> Vec<(Vec<u8>, SnapshotError)> {
    let entry = waiting_entry;
    let valid = snapshot_of(&[entry(0x30, 1), entry(0x30, 3), entry(0x31, 2)]);
    let mut bad_magic = valid.clone();
    bad_magic[0] = b'X';
    let mut version = valid.clone();
    version[4] = 2;
    let mut trailing = valid.clone();
    trailing.push(0);
    let truncated = valid[..valid.len() - 1].to_vec();
    let count = u8::try_from(SYNCPOINT_COUNT).unwrap();
    let many = SavedSubmit {
        entries: vec![1; CHANNEL_QUEUE_ENTRIES + 1],
        ..entry(0x30, 1)
    };
    let waits_on_count = SavedSubmit {
        wait: Some((count, 9)),
        ..entry(0x30, 1)
    };
    vec![
        (bad_magic, SnapshotError::BadMagic),
        (version, SnapshotError::UnsupportedVersion { found: 2 }),
        (trailing, SnapshotError::TrailingBytes { count: 1 }),
        (
            truncated,
            SnapshotError::Decode(postcard::Error::DeserializeUnexpectedEnd),
        ),
        (
            snapshot_of(std::slice::from_ref(&waits_on_count)),
            SnapshotError::WaitSyncpoint {
                index: 0,
                id: count,
            },
        ),
        (
            snapshot_of(&[entry(0x30, 1), many.clone()]),
            SnapshotError::TooManyEntries {
                index: 1,
                count: CHANNEL_QUEUE_ENTRIES + 1,
            },
        ),
        (
            snapshot_of(&[entry(0x30, 1), entry(0x31, 1)]),
            SnapshotError::DuplicateToken { index: 1, token: 1 },
        ),
        // Tokens out of order within a channel, and channels out of order.
        (
            snapshot_of(&[entry(0x30, 3), entry(0x30, 1)]),
            SnapshotError::Misordered { index: 1 },
        ),
        (
            snapshot_of(&[entry(0x31, 2), entry(0x30, 1)]),
            SnapshotError::Misordered { index: 1 },
        ),
        (
            snapshot_of(&[entry(0x30, 1), entry(0x30, 5)]),
            SnapshotError::NotPending { index: 1, token: 5 },
        ),
        // Two faults in one entry: the syncpoint before the count, the duplicate before the
        // order, the order before the pending check.
        (
            snapshot_of(&[SavedSubmit {
                entries: many.entries,
                ..waits_on_count
            }]),
            SnapshotError::WaitSyncpoint {
                index: 0,
                id: count,
            },
        ),
        (
            snapshot_of(&[entry(0x31, 3), entry(0x30, 3)]),
            SnapshotError::DuplicateToken { index: 1, token: 3 },
        ),
        (
            snapshot_of(&[entry(0x31, 2), entry(0x30, 5)]),
            SnapshotError::Misordered { index: 1 },
        ),
        // An entry without a token: its channel is still ordered, and it does not reset the
        // last token id of its channel.
        (
            snapshot_of(&[entry(0x31, 2), tokenless(0x30)]),
            SnapshotError::Misordered { index: 1 },
        ),
        (
            snapshot_of(&[entry(0x30, 3), tokenless(0x30), entry(0x30, 1)]),
            SnapshotError::Misordered { index: 2 },
        ),
    ]
}

/// Each snapshot `Submits::restore` refuses for the cap on a channel's waiting entries
/// while requests 1, 2 and 3 are pending, and its error, first fault first.
fn refused_by_the_cap() -> Vec<(Vec<u8>, SnapshotError)> {
    let entry = waiting_entry;
    let below_cap = SavedSubmit {
        entries: vec![1; MAX_WAITING_ENTRIES - 1],
        ..entry(0x30, 1)
    };
    let at_cap = SavedSubmit {
        entries: vec![1; MAX_WAITING_ENTRIES],
        ..entry(0x30, 1)
    };
    let two = SavedSubmit {
        entries: vec![1, 2],
        ..entry(0x30, 3)
    };
    let empty = SavedSubmit {
        entries: Vec::new(),
        ..tokenless(0x30)
    };
    vec![
        // A channel's waiting entries one past the cap, with entries and with none (charged
        // one).
        (
            snapshot_of(&[below_cap.clone(), two.clone()]),
            SnapshotError::OverCap {
                index: 1,
                channel: 0x30,
            },
        ),
        (
            snapshot_of(&[at_cap.clone(), empty]),
            SnapshotError::OverCap {
                index: 1,
                channel: 0x30,
            },
        ),
        // The count before the cap, the cap before the duplicate.
        (
            snapshot_of(&[
                below_cap,
                SavedSubmit {
                    entries: vec![1; CHANNEL_QUEUE_ENTRIES + 1],
                    ..entry(0x30, 3)
                },
            ]),
            SnapshotError::TooManyEntries {
                index: 1,
                count: CHANNEL_QUEUE_ENTRIES + 1,
            },
        ),
        (
            snapshot_of(&[at_cap, entry(0x30, 1)]),
            SnapshotError::OverCap {
                index: 1,
                channel: 0x30,
            },
        ),
    ]
}

/// A waiting submission on `channel` without a token, one entry, no wait.
fn tokenless(channel: u32) -> SavedSubmit {
    SavedSubmit {
        token: None,
        ..waiting_entry(channel, 0)
    }
}

/// Each snapshot `Submits::restore` refuses, with its error; a refused restore leaves the
/// server's pending requests as they were, and posts nothing. Then the edges, which are taken.
#[test]
fn a_waiting_table_no_run_reaches_is_refused() {
    let (mut live, _, _) = three_waiting();
    let front = channel_front(&live.fake);
    for (bytes, error) in refused_tables().into_iter().chain(refused_by_the_cap()) {
        assert_eq!(
            Submits::restore(&bytes, &live.rig.server, &front).err(),
            Some(error.clone()),
            "{error}"
        );
    }
    // Nothing was completed or dropped: the three requests are still pending, and nothing was
    // posted.
    assert_eq!(live.rig.server.pending().count(), 3);
    assert_eq!(live.deliver(), []);
    // The edges, taken: the table as it is, the last syncpoint, exactly a queue's entries. (Each
    // restored table is dropped, which answers its requests with the framework's error.)
    let last = u8::try_from(SYNCPOINT_COUNT - 1).unwrap();
    let edges = [
        vec![
            waiting_entry(0x30, 1),
            waiting_entry(0x30, 3),
            waiting_entry(0x31, 2),
        ],
        vec![SavedSubmit {
            wait: Some((last, 9)),
            ..waiting_entry(0x30, 1)
        }],
        vec![SavedSubmit {
            entries: vec![1; CHANNEL_QUEUE_ENTRIES],
            ..waiting_entry(0x30, 1)
        }],
        // A channel's waiting entries exactly at the cap, in two entries, and in each of two
        // channels at once.
        vec![
            SavedSubmit {
                entries: vec![1; MAX_WAITING_ENTRIES - 1],
                ..waiting_entry(0x30, 1)
            },
            waiting_entry(0x30, 3),
        ],
        vec![
            SavedSubmit {
                entries: vec![1; MAX_WAITING_ENTRIES - 1],
                ..waiting_entry(0x30, 1)
            },
            SavedSubmit {
                entries: Vec::new(),
                ..tokenless(0x30)
            },
            SavedSubmit {
                entries: vec![1; MAX_WAITING_ENTRIES],
                ..waiting_entry(0x31, 2)
            },
        ],
        // Entries without a token before, between and after the others, several in a channel.
        vec![
            tokenless(0x30),
            waiting_entry(0x30, 1),
            tokenless(0x30),
            waiting_entry(0x30, 3),
            tokenless(0x30),
            tokenless(0x31),
            waiting_entry(0x31, 2),
        ],
    ];
    for edge in edges {
        let restored = Submits::restore(&snapshot_of(&edge), &live.rig.server, &front);
        assert_eq!(restored.map(|table| table.len()).ok(), Some(edge.len()));
    }
    // A restored table counts its entries against the cap, and refuses a push past it.
    let at_cap = [
        SavedSubmit {
            entries: vec![1; MAX_WAITING_ENTRIES - 1],
            ..waiting_entry(0x30, 1)
        },
        SavedSubmit {
            entries: Vec::new(),
            ..tokenless(0x30)
        },
    ];
    let mut restored = Submits::restore(&snapshot_of(&at_cap), &live.rig.server, &front).unwrap();
    assert_eq!(
        restored.waiting_entries(ChannelId(0x30)),
        MAX_WAITING_ENTRIES
    );
    assert!(!restored.admits(ChannelId(0x30), 0));
    let submission = Submission {
        entries: Vec::new(),
        wait: None,
        increments: 0,
        wfi: true,
    };
    let refused = restored.push(ChannelId(0x30), None, submission, Arc::clone(&front));
    assert!(refused.is_err());
}

/// What a channel's waiting submissions count against the cap falls by each one's charge as
/// room takes it, oldest first, so a partly drained channel takes again what was taken out.
#[test]
fn room_gives_back_what_it_takes_from_the_cap() {
    let mut bench = Bench::new();
    let (fd, _, _, channel) = bench.channel();
    bench.fake.hold(channel, CHANNEL_QUEUE_ENTRIES);
    wait_with(&mut bench, fd, 3);
    wait_with(&mut bench, fd, 0);
    wait_with(&mut bench, fd, MAX_WAITING_ENTRIES - 3 - 1);
    assert_eq!(bench.submits.waiting_entries(channel), MAX_WAITING_ENTRIES);
    // Room for the oldest two: the three entries and the empty one, charged one, go.
    bench.fake.hold(channel, CHANNEL_QUEUE_ENTRIES - 3);
    bench.submits.room(channel);
    assert_eq!(bench.submits.waiting(channel), 1);
    let left = MAX_WAITING_ENTRIES - 3 - 1;
    assert_eq!(bench.submits.waiting_entries(channel), left);
    // Exactly what went fits again, and one entry more does not.
    assert!(bench.submits.admits(channel, 3 + 1));
    assert!(!bench.submits.admits(channel, 3 + 2));
    wait_with(&mut bench, fd, 3 + 1);
    let (answer, deferral) = bench.submit(fd, &[], 0, (0, 0));
    assert_eq!((answer.result, deferral), (NvResult::Busy, None));
}

#[test]
fn a_refused_waiting_table_is_said_in_plain_words() {
    let said = [
        (
            SnapshotError::BadMagic,
            "not a waiting submissions snapshot (bad magic)",
        ),
        (
            SnapshotError::UnsupportedVersion { found: 2 },
            "waiting submissions snapshot layout version 2 is not supported (this build reads 1)",
        ),
        (
            SnapshotError::TrailingBytes { count: 3 },
            "3 unexpected bytes after the waiting submissions snapshot",
        ),
        (
            SnapshotError::WaitSyncpoint { index: 1, id: 200 },
            "waiting submission 1 waits on syncpoint 200, which does not exist",
        ),
        (
            SnapshotError::TooManyEntries { index: 1, count: 9 },
            "waiting submission 1 has 9 entries, more than a queue holds",
        ),
        (
            SnapshotError::DuplicateToken { index: 1, token: 4 },
            "waiting submission 1 answers request 4, which an earlier one answers",
        ),
        (
            SnapshotError::Misordered { index: 1 },
            "waiting submission 1 is out of order",
        ),
        (
            SnapshotError::NotPending { index: 1, token: 4 },
            "waiting submission 1 answers request 4, which is not pending",
        ),
    ];
    for (error, words) in said {
        assert_eq!(error.to_string(), words);
    }
    let over = SnapshotError::OverCap {
        index: 1,
        channel: 0x30,
    };
    assert_eq!(
        over.to_string(),
        format!(
            "waiting submission 1 takes GPU channel 48's waiting entries past the cap of \
             {MAX_WAITING_ENTRIES}"
        )
    );
}

/// The drain records `QueueRoom` before it acts on it, and records it for a channel with
/// nothing waiting too.
#[test]
fn room_is_recorded_whether_or_not_anything_waits() {
    let mut live = Live::new();
    let fd = live.channel();
    let channel = live.made(fd);
    let recorded = live.room(channel);
    let note = crate::note::Note::QueueRoom { channel };
    assert_eq!(recorded, [note.event(live.rig.server.service_id())]);
    assert_eq!(live.deliver(), []);
}

// ----- Submissions without a reply token ---------------------------------------------------------

/// Where a test puts an out buffer that is not mapped: below the first address the rig places
/// anything at.
const UNMAPPED_OUT: u64 = 0x0F00_0000;

/// The increments the fake front was given for `channel`, in all.
fn increments_queued(live: &Live, channel: ChannelId) -> u32 {
    live.fake
        .queued()
        .into_iter()
        .filter(|&(queued, _)| queued == channel)
        .map(|(_, submission)| submission.increments)
        .sum()
}

/// A submission that must wait, whose out buffer cannot be written: the request answers the
/// fault, but the syncpoint's maximum was raised, so the submission waits all the same, without
/// a token. A later submission on the channel waits behind it, and once room comes the front is
/// given both, so the later fence is reached.
#[test]
fn a_waiting_submission_with_an_unwritable_out_buffer_still_counts() {
    let mut live = Live::new();
    let fd = live.channel();
    let channel = live.made(fd);
    live.fake.hold(channel, CHANNEL_QUEUE_ENTRIES);
    let first = live.rig.session();
    let (handled, buffer, _) = live.submit_out(
        first,
        fd,
        &entries_from(1, 1),
        FENCE_INCREMENT,
        (0, 0),
        UNMAPPED_OUT,
    );
    assert_eq!(handled, Handled::Replied);
    assert_ne!(answer(&buffer, 0).result, ResultCode::SUCCESS.raw());
    assert_eq!(live.waiting(channel), 1);
    assert_eq!(live.rig.server.pending().count(), 0);
    // A later submission waits behind it, with the fence that counts both.
    let second = live.session();
    let (handled, _, out_at) =
        live.submit_with(second, fd, &entries_from(2, 1), FENCE_INCREMENT, (0, 0));
    assert!(matches!(handled, Handled::Deferred(_)));
    let id = u32::from(FIRST_ALLOCATABLE);
    let fence = 2 * FENCE_INCREMENT_AMOUNT;
    assert_eq!(live.rig.memory(out_at, 24), submit_struct(1, 0, id, fence));
    // Room: both are queued, the later one's reply alone completes, and the increments the front
    // was given reach the later fence.
    live.fake.hold(channel, 0);
    live.room(channel);
    assert_eq!(live.deliver(), [(second, ok(&[NvResult::Success.raw()]))]);
    assert_eq!(live.waiting(channel), 0);
    assert_eq!(increments_queued(&live, channel), fence);
}

/// A waiting submission whose session closes is saved without a token, and the table restores
/// with the server: the save passes the checks the restore makes. The restored submission is
/// still given to the front, and no reply is completed for it.
#[test]
fn a_closed_session_s_waiting_submission_saves_and_restores() {
    let mut live = Live::new();
    let fd = live.channel();
    let channel = live.made(fd);
    live.fake.hold(channel, CHANNEL_QUEUE_ENTRIES);
    let session = live.session();
    let (handled, _, _) = live.submit_on(session, fd, &entries_from(1, 1), FENCE_INCREMENT);
    assert!(matches!(handled, Handled::Deferred(_)));
    live.rig.server.close(session).unwrap();
    assert_eq!(live.rig.server.pending().count(), 0);
    let table = live
        .rig
        .shared
        .lock()
        .submits()
        .save(&live.rig.server)
        .unwrap();
    let server = live.rig.server.save().unwrap();

    let family = NvsrvFamily::new(Arc::clone(&live.rig.shared), as_front(&live.rig.front))
        .with_channel_front(channel_front(&live.fake));
    let mut restored = nvsrv_server(
        Arc::clone(&live.rig.shared),
        as_front(&live.rig.front),
        UnknownPolicy::Strict,
    )
    .with_family(family)
    .with_service_id(live.rig.server.service_id());
    restored.restore(&server).unwrap();
    let front = channel_front(&live.fake);
    let submits = Submits::restore(&table, &restored, &front).unwrap();
    assert_eq!(submits.waiting(channel), 1);
    assert_eq!(submits.tokens(channel).count(), 0);
    live.rig.server = restored;
    *live.rig.shared.lock().submits() = submits;
    live.fake.clear();
    live.fake.hold(channel, 0);
    live.rig.shared.lock().submits().retry_all();
    assert_eq!(increments_queued(&live, channel), FENCE_INCREMENT_AMOUNT);
    assert_eq!(live.deliver(), []);
    assert!(live.rig.shared.lock().submits().is_empty());
}

// ----- The cap on a channel's waiting entries --------------------------------------------------

/// What `submits`' module docs state ("The cap"): while the queue stays full, sessions that
/// submit and close leave submissions waiting with no request pending, but only up to
/// [`MAX_WAITING_ENTRIES`] entries. The submission that reaches the cap exactly waits; the next
/// one answers `Busy` with the argument as sent, its syncpoint's maximum not raised and nothing
/// given to the front. Once room drains the table, the channel takes submissions again.
#[test]
fn abandoned_submissions_are_bounded_by_the_cap() {
    let mut live = Live::new();
    let fd = live.channel();
    let channel = live.made(fd);
    let syncpoint = SyncpointId::new(FIRST_ALLOCATABLE).unwrap();
    live.fake.hold(channel, CHANNEL_QUEUE_ENTRIES);
    // All but one entry of the cap, abandoned.
    let big = live.session();
    let entries = entries_from(1, MAX_WAITING_ENTRIES - 1);
    let (handled, _, _) = live.submit_on(big, fd, &entries, FENCE_INCREMENT);
    assert!(matches!(handled, Handled::Deferred(_)));
    live.rig.server.close(big).unwrap();
    // One entry more reaches the cap exactly: it waits, and is abandoned too.
    let last = live.session();
    let (handled, _, _) = live.submit_on(last, fd, &entries_from(0x10, 1), FENCE_INCREMENT);
    assert!(matches!(handled, Handled::Deferred(_)));
    live.rig.server.close(last).unwrap();
    assert_eq!(live.waiting_entries(channel), MAX_WAITING_ENTRIES);
    assert_eq!(live.waiting(channel), 2);
    assert_eq!(live.rig.server.pending().count(), 0);
    let raised = 2 * FENCE_INCREMENT_AMOUNT;
    let fence = |live: &Live| live.rig.shared.syncpoints().fence(syncpoint).unwrap().value;
    assert_eq!(fence(&live), raised);
    // One past the cap: `Busy`.
    live.fake.clear();
    let refused = live.session();
    let (handled, buffer, out_at) =
        live.submit_on(refused, fd, &entries_from(0x20, 1), FENCE_INCREMENT);
    assert_eq!(handled, Handled::Replied);
    assert_eq!(answer(&buffer, 1), ok(&[NvResult::Busy.raw()]));
    assert_eq!(
        live.rig.memory(out_at, 24),
        submit_struct(1, FENCE_INCREMENT, 0, 0)
    );
    assert_eq!(fence(&live), raised);
    assert_eq!(live.fake.calls(), []);
    assert_eq!(live.waiting_entries(channel), MAX_WAITING_ENTRIES);
    assert_eq!(live.rig.server.pending().count(), 0);
    // Room drains both, oldest first; then the channel takes a submission again.
    live.fake.hold(channel, 0);
    live.room(channel);
    assert_eq!(live.waiting(channel), 0);
    assert_eq!(increments_queued(&live, channel), raised);
    // The GPU has run them.
    live.fake.hold(channel, 0);
    let again = live.session();
    let (handled, buffer, _) = live.submit_on(again, fd, &entries_from(0x30, 1), FENCE_INCREMENT);
    assert_eq!(handled, Handled::Replied);
    assert_eq!(answer(&buffer, 1), ok(&[NvResult::Success.raw()]));
    assert_eq!(fence(&live), raised + FENCE_INCREMENT_AMOUNT);
}

/// Defers a submission of `count` entries on `fd`, whose channel's queue the caller keeps full,
/// and puts it in the bench's waiting table, as the server does.
fn wait_with(bench: &mut Bench, fd: i32, count: usize) {
    let (answer, deferral) = bench.submit(fd, &entries_from(1, count), FENCE_INCREMENT, (0, 0));
    assert_eq!(answer.result, NvResult::Success);
    let Deferral {
        channel,
        submission,
    } = deferral.unwrap();
    let front = Arc::clone(&bench.front);
    bench
        .submits
        .push(channel, None, submission, front)
        .unwrap();
}

/// The cap's edge, each pair differing only there: a submission that reaches the cap waits, one
/// a single entry past it is `Busy` before its fence is raised; a submission with no entries is
/// charged one entry (a listed departure). A whole queue's entries wait when nothing else does.
#[test]
fn the_cap_takes_a_submission_that_reaches_it_and_refuses_one_past() {
    let mut bench = Bench::new();
    let (fd, _, syncpoint, channel) = bench.channel();
    bench.fake.hold(channel, CHANNEL_QUEUE_ENTRIES);
    let max = |bench: &Bench| bench.syncpoints.fence(syncpoint).unwrap().value;
    // Nothing waits: a whole queue's entries wait, though they fill the cap.
    wait_with(&mut bench, fd, CHANNEL_QUEUE_ENTRIES);
    assert_eq!(bench.submits.waiting_entries(channel), MAX_WAITING_ENTRIES);
    bench.submits.cancel(channel);
    // One entry below the cap.
    wait_with(&mut bench, fd, MAX_WAITING_ENTRIES - 1);
    let before = max(&bench);
    // Two entries pass it by one: `Busy`, the argument as sent, the maximum as it was.
    let flags = FENCE_INCREMENT;
    let (answer, deferral) = bench.submit(fd, &entries_from(5, 2), flags, (0, 0));
    assert_eq!((answer.result, deferral), (NvResult::Busy, None));
    let mut sent = submit_struct(2, flags, 0, 0);
    sent.extend(entry_bytes(&entries_from(5, 2)));
    assert_eq!(answer.output, sent);
    assert_eq!(max(&bench), before);
    // No entries, charged one, reach it: taken, with the fence the refused one did not raise.
    let (answer, deferral) = bench.submit(fd, &[], flags, (0, 0));
    assert_eq!(answer.result, NvResult::Success);
    let expected = before + FENCE_INCREMENT_AMOUNT;
    assert_eq!(fence_of(&answer), (u32::from(syncpoint.get()), expected));
    let Deferral {
        channel: waits_on,
        submission,
    } = deferral.unwrap();
    bench
        .submits
        .push(waits_on, None, submission, Arc::clone(&bench.front))
        .unwrap();
    assert_eq!(bench.submits.waiting_entries(channel), MAX_WAITING_ENTRIES);
    // At the cap: no entries, and one entry, are both `Busy`.
    for count in [0, 1] {
        let (answer, deferral) = bench.submit(fd, &entries_from(9, count), flags, (0, 0));
        assert_eq!((answer.result, deferral), (NvResult::Busy, None), "{count}");
    }
    assert_eq!(max(&bench), expected);
    assert_eq!(bench.submits.waiting(channel), 2);
}

/// The cap is each channel's: another channel, with nothing waiting, takes a submission while
/// the first is at the cap.
#[test]
fn the_cap_is_each_channel_s() {
    let mut bench = Bench::new();
    let (fd, _, _, channel) = bench.channel();
    let (other_fd, _, _, other) = bench.channel();
    bench.fake.hold(channel, CHANNEL_QUEUE_ENTRIES);
    wait_with(&mut bench, fd, MAX_WAITING_ENTRIES);
    let (answer, deferral) = bench.submit(other_fd, &entries_from(1, 1), 0, (0, 0));
    assert_eq!((answer.result, deferral), (NvResult::Success, None));
    assert_eq!(bench.fake.queued().last().map(|(c, _)| *c), Some(other));
    // And with the other's queue full too, it waits rather than being refused.
    bench.fake.hold(other, CHANNEL_QUEUE_ENTRIES);
    wait_with(&mut bench, other_fd, 1);
    assert_eq!(bench.submits.waiting_entries(other), 1);
}

/// Each check before the cap answers first when both faults are present: with the channel at
/// the cap, a count above the entries sent (check 1) and a count above a queue's entries
/// (check 2) are `InvalidSize`; bits 0 and 8, an overflowing increment, and a wait on no
/// syncpoint (checks 3 to 5) are `BadParameter`; and nothing is raised.
#[test]
fn the_checks_before_the_cap_answer_first() {
    let mut bench = Bench::new();
    let (fd, _, syncpoint, channel) = bench.channel();
    bench.fake.hold(channel, CHANNEL_QUEUE_ENTRIES);
    wait_with(&mut bench, fd, MAX_WAITING_ENTRIES);
    let before = bench.syncpoints.fence(syncpoint).unwrap();
    // Check 1: count 2 with one entry sent.
    let mut input = submit_struct(2, 0, 0, 0);
    input.extend(entry_bytes(&entries_from(1, 1)));
    let (answer, _) = bench.call(IoctlVariant::Ioctl, fd, submit_number(1), &input, &[]);
    assert_eq!(answer.result, NvResult::InvalidSize);
    // Check 2: one entry more than a queue holds, all of them sent.
    let over = entries_from(1, CHANNEL_QUEUE_ENTRIES + 1);
    let count = u32::try_from(over.len()).unwrap();
    let (answer, deferral) = bench.kickoff(fd, count, &entry_bytes(&over), 0);
    assert_eq!((answer.result, deferral), (NvResult::InvalidSize, None));
    assert_eq!(bench.submits.waiting_entries(channel), MAX_WAITING_ENTRIES);
    let no_syncpoint = SYNCPOINT_COUNT;
    let faults = [
        (FENCE_WAIT | INCREMENT_VALUE, (0, 0)),
        (FENCE_INCREMENT | INCREMENT_VALUE, (0, u32::MAX)),
        (FENCE_WAIT, (no_syncpoint, 0)),
    ];
    for (flags, fence) in faults {
        let (answer, deferral) = bench.submit(fd, &entries_from(1, 1), flags, fence);
        assert_eq!((answer.result, deferral), (NvResult::BadParameter, None));
    }
    assert_eq!(bench.syncpoints.fence(syncpoint).unwrap(), before);
}

/// A submission with no entries is charged one, and one with entries what it holds.
#[test]
fn a_waiting_submission_is_charged_its_entries_and_at_least_one() {
    assert_eq!(charge(0), 1);
    assert_eq!(charge(1), 1);
    assert_eq!(charge(2), 2);
    assert_eq!(charge(MAX_WAITING_ENTRIES), MAX_WAITING_ENTRIES);
}

/// `Submits::push` refuses a submission that would pass the cap, whoever calls it: its token,
/// if it has one, answers `Busy`, and the table is unchanged. One that reaches the cap exactly
/// is taken.
#[test]
fn a_push_past_the_cap_is_refused() {
    let mut live = Live::new();
    let fd = live.channel();
    let channel = live.made(fd);
    live.fake.hold(channel, CHANNEL_QUEUE_ENTRIES);
    let session = live.session();
    let entries = entries_from(1, MAX_WAITING_ENTRIES - 1);
    let (handled, _, _) = live.submit_on(session, fd, &entries, 0);
    assert!(matches!(handled, Handled::Deferred(_)));
    let (pending, _) = live.rig.server.pending().next().unwrap();
    let front = channel_front(&live.fake);
    let submission = |count: usize| Submission {
        entries: entries_from(0x40, count)
            .into_iter()
            .map(GpfifoEntry)
            .collect(),
        wait: None,
        increments: 0,
        wfi: true,
    };
    // Two entries pass the cap by one: refused, and the reissued token answers `Busy`.
    let token = live.rig.server.reissue(pending);
    let refused =
        live.rig
            .shared
            .lock()
            .submits()
            .push(channel, token, submission(2), Arc::clone(&front));
    let over = OverCap {
        channel: channel.0,
        charged: MAX_WAITING_ENTRIES - 1,
        entries: 2,
        charge: 2,
    };
    assert_eq!(refused, Err(over));
    assert_eq!(live.deliver(), [(session, ok(&[NvResult::Busy.raw()]))]);
    assert_eq!(live.waiting(channel), 1);
    assert_eq!(live.waiting_entries(channel), MAX_WAITING_ENTRIES - 1);
    // One entry reaches it: taken.
    let accepted =
        live.rig
            .shared
            .lock()
            .submits()
            .push(channel, None, submission(1), Arc::clone(&front));
    assert_eq!(accepted, Ok(()));
    assert_eq!(live.waiting_entries(channel), MAX_WAITING_ENTRIES);
    // At the cap, a submission with no entries is refused too, without a token.
    let refused = live
        .rig
        .shared
        .lock()
        .submits()
        .push(channel, None, submission(0), front);
    let over = OverCap {
        channel: channel.0,
        charged: MAX_WAITING_ENTRIES,
        entries: 0,
        charge: charge(0),
    };
    assert_eq!(refused, Err(over));
    assert_eq!(live.waiting(channel), 2);
}

/// The refusal names what the channel's waiting submissions count, and the refused
/// submission's entries and what they count, also for a submission with no entries.
#[test]
fn a_push_past_the_cap_is_said_in_plain_words() {
    let over = OverCap {
        channel: 0x30,
        charged: 7,
        entries: 2,
        charge: 3,
    };
    assert_eq!(
        over.to_string(),
        format!(
            "GPU channel 48's waiting submissions count 7 entries against the cap of \
             {MAX_WAITING_ENTRIES}, and a submission of 2 entries, counted as 3, would pass it"
        )
    );
    let empty = OverCap {
        channel: 0x31,
        charged: MAX_WAITING_ENTRIES,
        entries: 0,
        charge: charge(0),
    };
    let one = charge(0);
    assert_eq!(
        empty.to_string(),
        format!(
            "GPU channel 49's waiting submissions count {MAX_WAITING_ENTRIES} entries against \
             the cap of {MAX_WAITING_ENTRIES}, and a submission of 0 entries, counted as {one}, \
             would pass it"
        )
    );
}

// ----- The order of a channel fd's close -------------------------------------------------------

/// One call to either front, in one log.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Logged {
    DestroyChannel(ChannelId),
    Update(AsId),
}

/// A VM front and a channel front that log into one list.
#[derive(Clone, Default)]
struct OneLog(Arc<Mutex<Vec<Logged>>>);

impl crate::gpu::GpuVmFront for OneLog {
    fn create_address_space(&self, _: AsId, _: Arc<starquake_gpu::vm::GpuVm>) {}

    fn update_vm(
        &self,
        space: AsId,
        _: starquake_gpu::interface::VmUpdate,
        _: Option<starquake_gpu::interface::Ticket>,
    ) {
        self.0.lock().unwrap().push(Logged::Update(space));
    }

    fn barrier(&self, _: starquake_gpu::interface::Ticket) {}
}

impl ChannelFront for OneLog {
    fn create_channel(&self, _: ChannelConfig) -> Result<ChannelId, GpuError> {
        Err(GpuError)
    }

    fn bind_class(&self, _: ChannelId, _: u32) -> Result<(), GpuError> {
        Err(GpuError)
    }

    fn submit(&self, _: ChannelId, _: Submission) -> Submitted {
        Submitted::Full
    }

    fn destroy_channel(&self, channel: ChannelId) {
        self.0.lock().unwrap().push(Logged::DestroyChannel(channel));
    }
}

/// Closing a channel fd whose binding is the last holder of its space destroys the GPU channel
/// before the space (`NvsrvState::close`): the space has a sparse region, so its destroy sends
/// the VM front an update, and both fronts log into one list.
#[test]
fn closing_a_channel_destroys_it_before_the_space_it_held() {
    let mut bench = Bench::new();
    let (fd, as_fd, _, made) = bench.channel();
    let space = bench.state.as_gpu().bound(fd.cast_unsigned()).unwrap();
    let device = bench
        .memory
        .create_device(crate::gpu::memory::Device { name: 1, label: 2 })
        .unwrap();
    let start = crate::gpu::memory::VA_WINDOW_START;
    let region = crate::gpu::memory::SparseRequest {
        space,
        first: start,
        last: start + starquake_memory::PAGE_SIZE - 1,
        page_shift: crate::gpu::memory::SMALL_PAGE_SHIFT,
        mode: 0,
        device,
        sync: 0,
    };
    bench.memory.create_sparse_region(&region).unwrap();
    assert_eq!(bench.close(as_fd), NvResult::Success);
    assert!(bench.memory.table(space).is_some());

    let log = OneLog::default();
    let channels: Arc<dyn ChannelFront> = Arc::new(log.clone());
    let mut deferred = None;
    let gpu = GpuAccess {
        memory: &mut bench.memory,
        front: &log,
        channels: Some(ChannelAccess {
            submits: &mut bench.submits,
            syncpoints: &bench.syncpoints,
            front: &channels,
            deferred: &mut deferred,
        }),
    };
    assert_eq!(bench.state.close(fd, gpu), NvResult::Success);
    assert!(bench.memory.table(space).is_none());
    let logged = log.0.lock().unwrap().clone();
    assert_eq!(logged.first(), Some(&Logged::DestroyChannel(made)));
    assert!(logged.len() > 1, "{logged:?}");
    assert!(
        logged[1..]
            .iter()
            .all(|&call| call == Logged::Update(space)),
        "{logged:?}"
    );
}

// ----- Whose inline input is read ---------------------------------------------------------------

/// With a channel front present, `Ioctl2` 0x1B reads its inline input only on an nvhost-gpu fd.
/// On an nvmap fd and on a closed fd, in buffer 1 lies where nothing is mapped and is not read:
/// the request answers what the fd gives (the strict policy's `NotImplemented` for nvmap,
/// `NvsrvState::file`'s `NotImplemented` for the closed fd) and its out buffer is untouched. On
/// the channel fd the same buffer is read, and the request answers the kernel's fault.
#[test]
fn inline_input_is_read_only_on_a_channel_fd() {
    let mut live = Live::new();
    let channel_fd = live.channel();
    let nvmap = live.rig.open_device(Device::Nvmap);
    let closed = live.rig.open_device(Device::NvhostGpu);
    let reply = answer(&live.rig.send(&cmif(CLOSE, &closed.to_le_bytes())), 1);
    assert_eq!(reply, ok(&[NvResult::Success.raw()]));
    let session = live.session();
    let input = submit_struct(1, 0, 0, 0);
    // The address the out-buffer tests use: nothing is mapped there.
    let inline = (UNMAPPED_OUT, entry_bytes(&entries_from(1, 1)).len());
    for fd in [nvmap, closed] {
        let out_at = live.rig.place(&[STALE; 24]);
        let (handled, buffer) = live.kickoff_at(session, fd, &input, inline, out_at);
        assert_eq!(handled, Handled::Replied, "fd {fd}");
        assert_eq!(
            answer(&buffer, 1),
            ok(&[NvResult::NotImplemented.raw()]),
            "fd {fd}"
        );
        assert_eq!(live.rig.memory(out_at, 24), vec![STALE; 24], "fd {fd}");
    }
    assert_eq!(live.fake.queued(), []);
    let out_at = live.rig.place(&[STALE; 24]);
    let (handled, buffer) = live.kickoff_at(session, channel_fd, &input, inline, out_at);
    assert_eq!(handled, Handled::Replied);
    assert_ne!(answer(&buffer, 0).result, ResultCode::SUCCESS.raw());
    assert_eq!(live.fake.queued(), []);
}

// ----- What a whole queue's inline submission reads --------------------------------------------

/// `SubmitGPFIFO` 0x1B with exactly [`CHANNEL_QUEUE_ENTRIES`] entries inline, through the
/// server: the inline input is read up to [`MAX_ENTRY_BYTES`], so every entry reaches the front.
/// The descriptors are written by hand, as the buffer is longer than a 16-bit length.
#[test]
fn an_inline_submission_of_a_whole_queue_is_read_whole() {
    let mut live = Live::new();
    let fd = live.channel();
    let channel = live.made(fd);
    let entries = entries_from(1, CHANNEL_QUEUE_ENTRIES);
    let count = u32::try_from(CHANNEL_QUEUE_ENTRIES).unwrap();
    let input = submit_struct(count, 0, 0, 0);
    let inline = entry_bytes(&entries);
    assert_eq!(inline.len(), MAX_ENTRY_BYTES);
    let input_at = live.rig.place(&input);
    let inline_at = live.rig.place(&inline);
    let out_at = live.rig.place(&[STALE; 24]);
    let request = cmif(IOCTL2, &fd_and(fd, KICKOFF));
    let request = Auto::alias(input_at, len16(input.len())).send(request, 0);
    let request = request
        .send_buffer(starquake_ipc::BufferDescriptor {
            address: inline_at,
            size: u64::try_from(inline.len()).unwrap(),
            flags: 0,
        })
        .static_descriptor(starquake_ipc::StaticDescriptor {
            counter: 1,
            address: 0,
            size: 0,
        });
    let request = Auto::alias(out_at, 24).receive(request);
    let reply = answer(&live.rig.send(&request), 1);
    assert_eq!(reply, ok(&[NvResult::Success.raw()]));
    let queued = live.fake.queued();
    assert_eq!(queued.len(), 1);
    assert_eq!(queued[0].0, channel);
    let sent: Vec<GpfifoEntry> = entries.into_iter().map(GpfifoEntry).collect();
    assert_eq!(queued[0].1.entries, sent);
}

// ----- A binding alone holds its space through a restore ---------------------------------------

/// A channel's binding is the only holder of its space once the space's last view closes; the
/// saved server restores (as-gpu's restore counts a binding as a holder), and the space is still
/// made.
#[test]
fn a_space_held_only_by_a_binding_restores() {
    let mut live = Live::new();
    let as_fd = live.rig.open_device(Device::NvhostAsGpu);
    let (reply, _) = live.rig.ioctl(as_fd, AS_ALLOC_AS_EX, &as_request(), 40);
    assert_eq!(reply, ok(&[0]));
    let fd = live.rig.open_device(Device::NvhostGpu);
    let (reply, _) = live.rig.ioctl(as_fd, AS_BIND_CHANNEL, &fd.to_le_bytes(), 4);
    assert_eq!(reply, ok(&[0]));
    let space = live.rig.state().as_gpu().bound(fd.cast_unsigned()).unwrap();
    let reply = answer(&live.rig.send(&cmif(CLOSE, &as_fd.to_le_bytes())), 1);
    assert_eq!(reply, ok(&[0]));
    assert!(live.rig.state().as_gpu().made(space).is_some());
    let state = live.rig.state().clone();
    let saved = live.rig.server.save().unwrap();

    let family = NvsrvFamily::new(Arc::clone(&live.rig.shared), as_front(&live.rig.front))
        .with_channel_front(channel_front(&live.fake));
    let mut restored = nvsrv_server(
        Arc::clone(&live.rig.shared),
        as_front(&live.rig.front),
        UnknownPolicy::Strict,
    )
    .with_family(family)
    .with_service_id(live.rig.server.service_id());
    restored.restore(&saved).unwrap();
    live.rig.server = restored;
    assert_eq!(live.rig.state(), &state);
    assert!(live.rig.state().as_gpu().made(space).is_some());
}

// ----- A channel's events across closes --------------------------------------------------------

/// The divergence in the module docs: closing a channel fd drops its event ids but frees no
/// kernel event, so two cycles of open, `QueryEvent` 1, 2 and 3, and close make six distinct
/// events, each still held by the kernel.
#[test]
fn a_closed_channel_s_events_are_not_freed() {
    let mut live = Live::new();
    let mut made = Vec::new();
    for _ in 0..2 {
        let fd = live.rig.open_device(Device::NvhostGpu);
        for id in [
            event::SM_EXCEPTION_INT,
            event::SM_EXCEPTION_PAUSE,
            event::ERROR_NOTIFIER,
        ] {
            let mut data = fd.to_le_bytes().to_vec();
            data.extend(id.to_le_bytes());
            let reply = answer(&live.rig.send(&cmif(QUERY_EVENT, &data)), 1);
            assert_eq!(reply, ok(&[NvResult::Success.raw()]));
        }
        let channel = live.rig.state().gpu().channel(fd.cast_unsigned()).unwrap();
        made.extend(channel.events.iter().map(|event| event.unwrap()));
        let reply = answer(&live.rig.send(&cmif(CLOSE, &fd.to_le_bytes())), 1);
        assert_eq!(reply, ok(&[0]));
        assert_eq!(live.rig.state().gpu().channel(fd.cast_unsigned()), None);
    }
    let distinct: std::collections::BTreeSet<_> = made.iter().collect();
    assert_eq!(distinct.len(), 2 * EVENT_SLOTS);
    for event in made {
        assert_eq!(live.rig.kernel.signals(event), Some(0), "{event:?}");
    }
}

mod settings;

mod zcull;
