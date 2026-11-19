//! Tests for the wiring: the `Gpu` driven as gsv drives it, through its threads, over the
//! production memory (`VmMemory`) and the recording backend, with the host clock on the front
//! (the production arrangement) and a clock held still on the back, so no list is closed for
//! the time it recorded. Every wait is bounded by [`PATIENCE`].
//!
//! `back` drives the back thread's loop one pass at a time, with no thread, where an order
//! between a control and an entry has to be pinned exactly; `recorder` tests the flight
//! recorder's text; `idle` drives idleness with the acquire poll, some of it on a front clock
//! held still.

mod back;
mod counters;
mod idle;
mod recorder;
mod threads;

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use starquake_memory::{FlatMemory, GuestMemory, PAGE_SIZE, PageRange, WriteWatch};
use starquake_render::caps::{
    Caps, DynamicState, Features, FormatFeatures, FormatTable, Limits, SampleCounts, StageSupport,
};
use starquake_render::command::Rect;
use starquake_render::recording::RecordingBackend;
use starquake_shader::lookup::{EngineState, ProgramMemory, ProgramRequest, VariantKey};

use super::*;
use crate::back::buffer::InvalidationSink;
use crate::back::draw::shaders::{SetLookup, ShaderLookup};
use crate::back::exec::PRESENT_FORMAT_RGBA8;
use crate::back::fence::tests::{Clock, LateBackend};
use crate::classes::{clc76f, clc797};
use crate::fifo::{GpEntry, MethodMode, PushBuffer, Sync};
use crate::interface::{DisplayId, GpfifoEntry, ImageDesc, Kind, LayerId, SyncpointId, Target};
use crate::ring::RingEntry;
use crate::sema::{self, AcquireOp, HostSemaphore, PayloadSize, ReleaseForm};

/// How long any wait for the threads may take before the test fails.
const PATIENCE: Duration = Duration::from_secs(20);

/// The CPU memory behind the mapping.
const CPU_BASE: u64 = 0x8000_0000;
/// Pages mapped from `GPU_BASE`.
const PAGES: u64 = 64;
/// The GPU address of `CPU_BASE`: not equal to it, so a missed translation reads elsewhere.
const GPU_BASE: u64 = 0x12_3400_0000;
/// Command buffers.
const CMD: u64 = GPU_BASE;
const CMD2: u64 = GPU_BASE + 2 * PAGE_SIZE;
/// Semaphores: not page aligned.
const SEM: u64 = GPU_BASE + 48 * PAGE_SIZE + 0x10;
const SEM2: u64 = SEM + 0x40;
/// An 8x8 RGBA8 frame.
const FRAME: u64 = GPU_BASE + 32 * PAGE_SIZE;
const FRAME_SIZE: u32 = 8;
/// The address space the rig's channels use: not 0, so a hard-coded space fails.
const SPACE: AsId = AsId(0x21);
/// A second address space.
const OTHER_SPACE: AsId = AsId(0x22);
/// The syncpoint the rig's channels increment.
const SYNCPOINT: u8 = 5;

fn syncpoint() -> SyncpointId {
    SyncpointId::new(SYNCPOINT).unwrap()
}

fn caps() -> Caps {
    Caps {
        limits: Limits {
            max_buffer_size: PAGES * PAGE_SIZE,
            max_staging_size: PAGES * PAGE_SIZE,
            max_texture_dimension_1d: 4096,
            max_texture_dimension_2d: 4096,
            max_texture_dimension_3d: 256,
            max_texture_array_layers: 64,
            max_sampler_anisotropy: 16,
            max_vertex_attributes: 16,
            max_vertex_buffers: 8,
            max_vertex_buffer_stride: 2048,
            max_color_attachments: 8,
            max_viewports: 1,
            max_bind_groups: 4,
            max_bindings_per_bind_group: 16,
            max_compute_workgroups_per_dimension: 64,
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

/// A shader cache with no shader: every lookup is pending and never finishes, so a draw or
/// dispatch is skipped. No test here draws.
struct NoShaders;

impl ShaderLookup for NoShaders {
    fn lookup(
        &mut self,
        _requests: &[ProgramRequest],
        _variant: &VariantKey,
        _memory: &dyn ProgramMemory,
        _state: &dyn EngineState,
    ) -> SetLookup {
        SetLookup::Pending(starquake_shader::lookup::Ticket::new(0))
    }

    fn wait(&mut self, ticket: starquake_shader::lookup::Ticket) -> SetLookup {
        SetLookup::Pending(ticket)
    }
}

impl InvalidationSink for NoShaders {
    fn invalidate(&mut self, _range: PageRange) {}
}

/// gsv's completion queue: every push, with the word at [`SEM2`]'s CPU address as it was when
/// it was pushed (to see what a release wrote before an ordered update was reported).
#[derive(Default)]
struct Pushed {
    all: Mutex<Vec<(GpuCompletion, u32)>>,
    guest: Mutex<Option<Arc<FlatMemory>>>,
}

impl Pushed {
    fn all(&self) -> Vec<GpuCompletion> {
        self.all.lock().unwrap().iter().map(|&(c, _)| c).collect()
    }

    /// The probe word when `completion` was pushed.
    fn probe_at(&self, completion: GpuCompletion) -> Option<u32> {
        self.all
            .lock()
            .unwrap()
            .iter()
            .find(|&&(c, _)| c == completion)
            .map(|&(_, word)| word)
    }
}

impl Completions for Pushed {
    fn push(&self, completion: GpuCompletion) {
        let word = self
            .guest
            .lock()
            .unwrap()
            .as_ref()
            .map_or(0, |guest| cpu_word(guest, SEM2));
        self.all.lock().unwrap().push((completion, word));
    }
}

/// The CPU address of GPU address `va` in the rig's mapping.
fn cpu(va: u64) -> u64 {
    va - GPU_BASE + CPU_BASE
}

/// The word at GPU address `va`, read through the CPU's side.
fn cpu_word(guest: &FlatMemory, va: u64) -> u32 {
    let mut bytes = [0; 4];
    guest.read(cpu(va), &mut bytes).unwrap();
    u32::from_le_bytes(bytes)
}

/// Waits until `done` holds; fails the test after [`PATIENCE`].
fn wait_until(what: &str, mut done: impl FnMut() -> bool) {
    let deadline = Instant::now() + PATIENCE;
    while !done() {
        assert!(Instant::now() < deadline, "timed out waiting until {what}");
        std::thread::sleep(Duration::from_millis(1));
    }
}

/// Which backend the rig records through.
enum Recording {
    Now(Arc<RecordingBackend>),
    Late(Arc<LateBackend>),
}

struct Rig {
    gpu: Gpu,
    guest: Arc<FlatMemory>,
    vm: Arc<GpuVm>,
    pushed: Arc<Pushed>,
    syncpoints: Arc<Syncpoints>,
    backend: Recording,
}

/// The table every rig space starts with: all of `GPU_BASE`'s pages to `CPU_BASE`.
fn table() -> Arc<GpuVm> {
    let vm = Arc::new(GpuVm::new());
    vm.map(
        PageRange::new(GPU_BASE, PAGES * PAGE_SIZE).unwrap(),
        Target::Guest { cpu_va: CPU_BASE },
        Kind(0),
    )
    .unwrap();
    vm
}

fn rig_on(late: bool) -> Rig {
    rig_with(late, Arc::new(HostClock::new()))
}

/// [`rig_on`] with `front_clock` on the front thread.
fn rig_with(late: bool, front_clock: Arc<dyn FrontClock>) -> Rig {
    let size = usize::try_from(PAGES * PAGE_SIZE).unwrap();
    let guest = Arc::new(FlatMemory::new(CPU_BASE, size));
    let pushed = Arc::new(Pushed::default());
    *pushed.guest.lock().unwrap() = Some(Arc::clone(&guest));
    let syncpoints = Arc::new(Syncpoints::new(Arc::<Pushed>::clone(&pushed)));
    let ring = GpuRing::new(1 << 16).unwrap();
    let (backend, recording): (Arc<dyn Backend>, Recording) = if late {
        let late = Arc::new(LateBackend::with_caps(
            caps(),
            1 << 30,
            ring.completion_wake(),
        ));
        (Arc::<LateBackend>::clone(&late), Recording::Late(late))
    } else {
        let now = Arc::new(RecordingBackend::with_completion_wake(
            caps(),
            1 << 30,
            ring.completion_wake(),
        ));
        (Arc::<RecordingBackend>::clone(&now), Recording::Now(now))
    };
    let gpu = Gpu::new(
        GpuDeps {
            ring,
            syncpoints: Arc::clone(&syncpoints),
            completions: Arc::<Pushed>::clone(&pushed),
            backend,
            shaders: Box::new(NoShaders),
            guest: Arc::<FlatMemory>::clone(&guest),
            watch: Arc::new(WriteWatch::new()),
            front_clock,
            fence_clock: Arc::new(Clock::default()),
        },
        GpuConfig::default(),
    )
    .unwrap();
    Rig {
        gpu,
        guest,
        vm: table(),
        pushed,
        syncpoints,
        backend: recording,
    }
}

fn rig() -> Rig {
    rig_on(false)
}

impl Rig {
    /// `SPACE`, made with the rig's table.
    fn space(&self) {
        self.gpu.create_address_space(SPACE, Arc::clone(&self.vm));
    }

    /// A channel in `SPACE` incrementing [`SYNCPOINT`].
    fn channel(&self) -> ChannelId {
        self.gpu
            .create_channel(ChannelConfig {
                space: SPACE,
                syncpoint: syncpoint(),
            })
            .unwrap()
    }

    fn write(&self, va: u64, words: &[u32]) {
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        self.guest.write(cpu(va), &bytes).unwrap();
    }

    fn word(&self, va: u64) -> u32 {
        cpu_word(&self.guest, va)
    }

    /// Writes `words` at `va` and returns a GP entry over them.
    fn entry(&self, va: u64, words: &[u32]) -> GpfifoEntry {
        self.write(va, words);
        raw_entry(va, words.len())
    }

    fn recording(&self) -> &RecordingBackend {
        match &self.backend {
            Recording::Now(now) => now,
            Recording::Late(late) => &late.inner,
        }
    }

    fn late(&self) -> &LateBackend {
        match &self.backend {
            Recording::Late(late) => late,
            Recording::Now(_) => unreachable!("not a late rig"),
        }
    }

    fn wait_for(&self, completion: GpuCompletion) {
        wait_until(&format!("{completion:?} is pushed"), || {
            self.pushed.all().contains(&completion)
        });
    }
}

/// A GP entry of `length` words at `va`.
fn raw_entry(va: u64, length: usize) -> GpfifoEntry {
    let entry = GpEntry {
        address: va,
        length: u32::try_from(length).unwrap(),
        sync: Sync::Proceed,
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

/// A method number as a header takes it.
fn m(method: u32) -> u16 {
    u16::try_from(method).unwrap()
}

/// Binds the 3D class on subchannel 0.
fn bind_3d(pb: &mut PushBuffer) {
    pb.methods(
        MethodMode::Incrementing,
        0,
        m(clc76f::SET_OBJECT.method()),
        &[clc797::CLASS],
    )
    .unwrap();
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

fn acquire(pb: &mut PushBuffer, address: u64, payload: u32) {
    let word = sema::encode_semaphored(&HostSemaphore::Acquire {
        op: AcquireOp::Equal,
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

/// A stream that waits for `address` to hold `payload`, then releases `then` at `release_at`.
fn acquires_then_releases(address: u64, payload: u32, release_at: u64, then: u32) -> Vec<u32> {
    let mut pb = PushBuffer::new();
    acquire(&mut pb, address, payload);
    release(&mut pb, release_at, then);
    pb.into_words()
}

/// A stream that releases `payload` at `address`.
fn releases(address: u64, payload: u32) -> Vec<u32> {
    let mut pb = PushBuffer::new();
    release(&mut pb, address, payload);
    pb.into_words()
}

/// A present of the 8x8 RGBA8 frame at [`FRAME`], all of it shown.
fn present_request() -> PresentRequest {
    PresentRequest {
        image: ImageDesc {
            address: FRAME,
            width: FRAME_SIZE,
            height: FRAME_SIZE,
            pitch: 1,
            block_height_log2: 0,
            format: PRESENT_FORMAT_RGBA8,
        },
        shown: Rect {
            x: 0,
            y: 0,
            width: FRAME_SIZE,
            height: FRAME_SIZE,
        },
        layer: LayerId(1),
        display: DisplayId(0),
    }
}
