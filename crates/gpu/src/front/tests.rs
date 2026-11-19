//! Tests for the front loop: scripted multi-channel scenarios. Streams are built with
//! `fifo::PushBuffer` and read through a `TestGpuMemory`; the back is the real fence path
//! (`back::fence::Fences` over the recording backend) behind a small executor that turns ring
//! entries into fenced actions, so releases latch the real waiter table, increments reach the
//! real `Syncpoints`, and markers apply their ordered updates and report back through
//! `FrontWake`; the clock is a fake.
//!
//! Nothing is recorded on the back's open list, so every fenced action is performed as soon as
//! the back pops it (the empty-fence rule); a test that wants the back to lag simply does not
//! drain it.

mod ordered;
mod queues;
mod scheduling;
mod thread;
mod units;
mod waits;

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use starquake_memory::{FlatMemory, PAGE_SIZE, PageRange};
use starquake_render::caps::{
    Caps, DynamicState, Features, FormatFeatures, FormatTable, Limits, SampleCounts, StageSupport,
};
use starquake_render::recording::RecordingBackend;

use super::*;
use crate::back::fence::{
    CloseReason, FenceClock, FenceEnv, FencedAction, Fences, Mode,
};
use crate::classes::{clc76f, clc797};
use crate::fifo::{GpEntry, MethodMode, PushBuffer, Sync};
use crate::interface::{
    ChannelConfig, GpfifoEntry, Kind, Submission, Submitted, SyncpointId, Target,
};
use crate::memory::TestGpuMemory;
use crate::regs::EntrySink;
use crate::ring::{ClassId, Consumer, RingEntry, ring};
use crate::sema::{self, AcquireOp, HostSemaphore, PayloadSize, ReleaseForm};

// ---------------------------------------------------------------------------------------------
// Rig
// ---------------------------------------------------------------------------------------------

/// The CPU memory behind the mapping.
const CPU_BASE: u64 = 0x8000_0000;
/// Pages mapped from `GPU_BASE`.
const PAGES: u64 = 64;
/// Above 32 bits, so the semaphore registers' upper halves matter.
const GPU_BASE: u64 = 0x12_3400_0000;
/// Command buffers, two pages apart (a long stream at `CMD_B` may run over `CMD_C`; no test
/// uses both).
const CMD_A: u64 = GPU_BASE;
const CMD_B: u64 = GPU_BASE + 2 * PAGE_SIZE;
const CMD_C: u64 = GPU_BASE + 4 * PAGE_SIZE;
/// Semaphores: not page aligned.
const SEM: u64 = GPU_BASE + 48 * PAGE_SIZE + 0x10;
const SEM2: u64 = SEM + 0x40;
/// The address space the rig's channels use: not 0, so a hard-coded space fails.
const SPACE: AsId = AsId(0x21);
/// A second address space.
const OTHER_SPACE: AsId = AsId(0x22);
/// The first channel id the rig's front hands out: not 0, so a hard-coded id fails.
const FIRST_CHANNEL: u64 = 0x40;
/// Words in the test ring: far more than any scenario pushes before the back drains.
const RING_WORDS_FOR_TESTS: usize = 1 << 16;
/// Turns `settle` makes before it gives up.
const SETTLE_TURNS: usize = 10_000;

fn syncpoint(id: u8) -> SyncpointId {
    SyncpointId::new(id).unwrap()
}

/// The class bound on subchannel 0 by the streams below.
fn class_3d() -> ClassId {
    ClassId(clc797::CLASS)
}

/// A method number as a header takes it.
fn m(method: u32) -> u16 {
    u16::try_from(method).unwrap()
}

/// The fake clock: the front's and the back's.
#[derive(Default)]
struct Clock {
    nanos: AtomicU64,
    reads: AtomicU64,
}

impl Clock {
    fn advance(&self, by: Duration) {
        self.nanos
            .fetch_add(u64::try_from(by.as_nanos()).unwrap(), Ordering::SeqCst);
    }

    fn reads(&self) -> u64 {
        self.reads.load(Ordering::SeqCst)
    }
}

impl FrontClock for Clock {
    fn now(&self) -> Duration {
        self.reads.fetch_add(1, Ordering::SeqCst);
        Duration::from_nanos(self.nanos.load(Ordering::SeqCst))
    }
}

impl FenceClock for Clock {
    fn elapsed(&self) -> Duration {
        Duration::from_nanos(self.nanos.load(Ordering::SeqCst))
    }

    fn gpu_timestamp(&self) -> u64 {
        self.nanos.load(Ordering::SeqCst)
    }
}

/// gsv's completion queue.
#[derive(Default)]
struct Pushed(Mutex<Vec<GpuCompletion>>);

impl Pushed {
    fn all(&self) -> Vec<GpuCompletion> {
        self.0.lock().unwrap().clone()
    }
}

impl Completions for Pushed {
    fn push(&self, completion: GpuCompletion) {
        self.0.lock().unwrap().push(completion);
    }
}

fn caps() -> Caps {
    Caps {
        limits: Limits {
            max_buffer_size: 1 << 16,
            max_staging_size: 1 << 12,
            max_texture_dimension_1d: 256,
            max_texture_dimension_2d: 256,
            max_texture_dimension_3d: 64,
            max_texture_array_layers: 8,
            max_sampler_anisotropy: 16,
            max_vertex_attributes: 16,
            max_vertex_buffers: 8,
            max_vertex_buffer_stride: 2048,
            max_color_attachments: 8,
            max_viewports: 1,
            max_bind_groups: 4,
            max_bindings_per_bind_group: 16,
            max_compute_workgroups_per_dimension: 65_535,
            max_uniform_buffers_per_stage: 12,
            max_storage_buffers_per_stage: 8,
            max_sampled_textures_per_stage: 16,
            max_storage_textures_per_stage: 4,
            max_samplers_per_stage: 16,
        },
        features: Features::ALL,
        stages: StageSupport::ALL,
        dynamic_state: DynamicState::ALL,
        sample_counts: SampleCounts::X1,
        view_swizzle: true,
        formats: FormatTable::uniform(FormatFeatures::ALL),
    }
}

type Memory = TestGpuMemory<FlatMemory>;

/// The back: pops ring entries and hands them to the real fence path as the executor (slice G19)
/// will (module docs).
struct Back {
    consumer: Consumer,
    fences: Fences,
    ordered: Arc<OrderedTable>,
    memory: Arc<Memory>,
    log: Vec<RingEntry<'static>>,
}

impl Back {
    /// Pops and performs every published entry; how many.
    fn drain(&mut self) -> usize {
        let mut count = 0;
        while let Some(entry) = self.consumer.try_pop().unwrap() {
            count += 1;
            self.handle(entry);
        }
        count
    }

    /// Performs one entry and logs it.
    fn handle(&mut self, entry: RingEntry<'static>) {
        {
            match &entry {
                RingEntry::Release(release) => {
                    let action = FencedAction::Release {
                        memory: Arc::<Memory>::clone(&self.memory),
                        space: SPACE,
                        release: *release,
                    };
                    self.fences.attach(action).unwrap();
                }
                RingEntry::Syncpoint { id } => {
                    self.fences.attach(FencedAction::Increment(*id)).unwrap();
                }
                RingEntry::Marker { token } => {
                    let action = FencedAction::Marker {
                        token: *token,
                        ordered: self.ordered.take(*token),
                    };
                    self.fences.attach(action).unwrap();
                }
                RingEntry::WaitForIdle { .. } => {
                    self.fences.close(CloseReason::WaitForIdle).unwrap();
                }
                RingEntry::SetReference { .. } => {
                    self.fences.close(CloseReason::SetReference).unwrap();
                }
                _ => {}
            }
            self.log.push(entry);
        }
    }
}

struct Rig {
    front: Front,
    shared: Arc<FrontShared>,
    back: Back,
    mem: Arc<Memory>,
    vm: Arc<GpuVm>,
    syncpoints: Arc<Syncpoints>,
    waiters: Arc<WaiterTable>,
    ordered: Arc<OrderedTable>,
    pushed: Arc<Pushed>,
    clock: Arc<Clock>,
}

/// A rig over a memory that `wrap` may hook, with a ring of `ring_words`.
fn rig_with(ring_words: usize, wrap: impl FnOnce(Arc<Memory>) -> Arc<dyn GpuMemory>) -> Rig {
    let size = usize::try_from(PAGES * PAGE_SIZE).unwrap();
    let mem = Arc::new(TestGpuMemory::new(FlatMemory::new(CPU_BASE, size)));
    mem.map(PageRange::new(GPU_BASE, PAGES * PAGE_SIZE).unwrap(), CPU_BASE)
        .unwrap();
    let vm = Arc::new(GpuVm::new());
    vm.map(
        PageRange::new(GPU_BASE, PAGES * PAGE_SIZE).unwrap(),
        Target::Guest { cpu_va: CPU_BASE },
        Kind(0),
    )
    .unwrap();
    let pushed = Arc::new(Pushed::default());
    let syncpoints = Arc::new(Syncpoints::new(Arc::<Pushed>::clone(&pushed)));
    let waiters = Arc::new(WaiterTable::new());
    let ordered = Arc::new(OrderedTable::new());
    let clock = Arc::new(Clock::default());
    let shared = Arc::new(FrontShared::new());
    shared.set_next_channel(FIRST_CHANNEL);
    let (producer, consumer) = ring(ring_words).unwrap();
    let env = FenceEnv {
        syncpoints: Arc::clone(&syncpoints),
        waiters: Arc::clone(&waiters),
        completions: Arc::<Pushed>::clone(&pushed),
        clock: Arc::<Clock>::clone(&clock),
        front: Arc::<FrontShared>::clone(&shared),
    };
    let backend = Arc::new(RecordingBackend::new(caps(), 1 << 30));
    let fences = Fences::new(backend, env, Mode::Fenced);
    let front = Front::new(FrontDeps {
        shared: Arc::clone(&shared),
        ring: producer,
        syncpoints: Arc::clone(&syncpoints),
        completions: Arc::<Pushed>::clone(&pushed),
        waiters: Arc::clone(&waiters),
        ordered: Arc::clone(&ordered),
        clock: Arc::<Clock>::clone(&clock),
    });
    shared.create_address_space(SPACE, Arc::clone(&vm), wrap(Arc::clone(&mem)));
    Rig {
        front,
        shared,
        back: Back {
            consumer,
            fences,
            ordered: Arc::clone(&ordered),
            memory: Arc::clone(&mem),
            log: Vec::new(),
        },
        mem,
        vm,
        syncpoints,
        waiters,
        ordered,
        pushed,
        clock,
    }
}

fn rig() -> Rig {
    rig_with(RING_WORDS_FOR_TESTS, |mem| mem)
}

impl Rig {
    /// A channel in `SPACE` incrementing syncpoint `id`.
    fn channel(&self, id: u8) -> ChannelId {
        self.shared
            .create_channel(ChannelConfig {
                space: SPACE,
                syncpoint: syncpoint(id),
            })
            .unwrap()
    }

    fn write(&self, va: u64, words: &[u32]) {
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        self.mem.write(va, &bytes).unwrap();
    }

    fn read_word(&self, va: u64) -> u32 {
        let mut bytes = [0; 4];
        self.mem.read(va, &mut bytes).unwrap();
        u32::from_le_bytes(bytes)
    }

    /// Writes `words` at `va` and returns a GP entry over them.
    fn entry(&self, va: u64, words: &[u32]) -> GpfifoEntry {
        self.write(va, words);
        raw_entry(va, words.len(), Sync::Proceed)
    }

    /// Submits `entries` on `channel` with no fence, increments or wait for idle.
    fn submit(&self, channel: ChannelId, entries: Vec<GpfifoEntry>) -> Submitted {
        self.shared.submit(channel, plain(entries))
    }

    /// Turns the front and drains the back until neither has anything left; the last turn.
    fn settle(&mut self) -> Turn {
        let mut last = Turn::Worked;
        for _ in 0..SETTLE_TURNS {
            last = self.front.turn().unwrap();
            let drained = self.back.drain();
            if matches!(last, Turn::Idle { .. }) && drained == 0 {
                break;
            }
        }
        last
    }

    /// One turn of the front, then the back drains what was published.
    fn turn(&mut self) -> Turn {
        let turn = self.front.turn().unwrap();
        self.back.drain();
        turn
    }

    /// The text of every entry the back took, and forgets them.
    fn take(&mut self) -> Vec<String> {
        std::mem::take(&mut self.back.log)
            .iter()
            .map(ToString::to_string)
            .collect()
    }
}

/// A GP entry of `length` words at `va`.
fn raw_entry(va: u64, length: usize, sync: Sync) -> GpfifoEntry {
    let entry = GpEntry {
        address: va,
        length: u32::try_from(length).unwrap(),
        sync,
        ..GpEntry::default()
    };
    GpfifoEntry(entry.to_raw().unwrap())
}

fn plain(entries: Vec<GpfifoEntry>) -> Submission {
    Submission {
        entries,
        wait: None,
        increments: 0,
        wfi: false,
    }
}

/// A stream that binds the 3D class on subchannel 0.
fn bind_3d(pb: &mut PushBuffer) {
    pb.methods(
        MethodMode::Incrementing,
        0,
        m(clc76f::SET_OBJECT.method()),
        &[clc797::CLASS],
    )
    .unwrap();
}

/// The 3D register the streams write: `SET_MME_SHADOW_SCRATCH(i)`, plain state, so a write is
/// sent only when it changes the register (the streams write distinct non-zero values).
fn scratch(index: u32) -> u16 {
    m(clc797::SET_MME_SHADOW_SCRATCH.method_at(&[index]).unwrap())
}

/// The text a write of `value` to `SET_MME_SHADOW_SCRATCH(index)` on `channel` takes.
fn scratch_text(channel: ChannelId, index: u32, value: u32) -> String {
    RingEntry::Write {
        channel,
        class: class_3d(),
        method: scratch(index),
        value,
    }
    .to_string()
}

/// The four `SEMAPHOREA`–`D` words for `address`, `payload` and the operation word `op`.
fn semaphored(pb: &mut PushBuffer, address: u64, payload: u32, op: u32) {
    let upper = u32::try_from(address >> 32).unwrap();
    let lower = u32::try_from(address & 0xffff_ffff).unwrap();
    pb.methods(
        MethodMode::Incrementing,
        0,
        m(clc76f::SEMAPHOREA.method()),
        &[upper, lower, payload, op],
    )
    .unwrap();
}

fn acquire(pb: &mut PushBuffer, address: u64, op: AcquireOp, payload: u32) {
    let word = sema::encode_semaphored(&HostSemaphore::Acquire {
        op,
        size: PayloadSize::Four,
        switch: false,
    })
    .unwrap();
    semaphored(pb, address, payload, word);
}

fn release(pb: &mut PushBuffer, address: u64, payload: u32) {
    let word = sema::encode_semaphored(&HostSemaphore::Release {
        form: ReleaseForm {
            size: PayloadSize::Four,
            timestamp: false,
        },
        wfi: false,
    })
    .unwrap();
    semaphored(pb, address, payload, word);
}

/// A stream of `bind_3d` then one write of `value` to scratch `index`.
fn write_scratch(index: u32, value: u32) -> Vec<u32> {
    let mut pb = PushBuffer::new();
    bind_3d(&mut pb);
    pb.methods(MethodMode::Incrementing, 0, scratch(index), &[value])
        .unwrap();
    pb.into_words()
}

/// Whether `texts` holds `text`.
fn has(texts: &[String], text: &str) -> bool {
    texts.iter().any(|t| t == text)
}

/// Where `text` is in `texts`.
fn position(texts: &[String], text: &str) -> usize {
    texts
        .iter()
        .position(|t| t == text)
        .unwrap_or_else(|| unreachable_index(texts, text))
}

/// The index a missing entry is given: past the end, after asserting it is there.
fn unreachable_index(texts: &[String], text: &str) -> usize {
    assert!(texts.iter().any(|t| t == text), "{text} not in {texts:#?}");
    texts.len()
}
