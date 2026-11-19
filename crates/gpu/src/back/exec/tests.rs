//! Tests of the executor: ring text in, command lists out (`insta`, through the recording
//! backend's text), with a fake shader cache that answers fixed sets and records the
//! invalidations it is sent, and the late backend of the fence tests where a test must see a
//! list before it completes.

use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use starquake_memory::{FlatMemory, GuestMemory, PAGE_SIZE, PageRange, WriteWatch};
use starquake_render::caps::{DynamicState, StageSupport};
use starquake_render::shader::ResourceKind;
use starquake_render::{
    Backend, Caps, Command, CompletionWake, Features, FenceValue, FormatFeatures, FormatTable,
    Limits, RecordingBackend, SampleCounts,
};
use starquake_shader::lookup::{
    EngineState, GuestStage, ProgramMemory, ProgramRequest, ShaderSet, Ticket, VariantKey,
};

use super::*;
use crate::back::buffer::InvalidationSink;
use crate::back::draw::shaders::{
    ReadySet, ResourceSource, ResourceUse, SetLookup, ShaderLookup, Slot, StageInfo,
};
use crate::back::draw::tests::host_shader;
use crate::back::fence::tests::{Clock, Front, LateBackend, Pushed};
use crate::back::fence::{FenceEnv, Mode, OrderedUpdate, VmChange, WaiterTable};
use crate::back::pipeline::ShaderStamp;
use crate::back::qmd::tests::{SlotSpec, Spec, encode};
use crate::back::qmd::{CONSTANT_BUFFER_SIZE_SHIFT, ConstantBuffer, QMD_ADDRESS_SHIFT, QMD_WORDS};
use crate::classes::{Method, cl902d, cla140, clc7b5, clc7c0, clc797};
use crate::field::Field;
use crate::interface::{DisplayId, ImageDesc, LayerId, Rect, VmUpdate};
use crate::memory::TestGpuMemory;
use crate::ring::{MarkerToken, parse_text};
use crate::syncpoint::Syncpoints;
use crate::test_log::{at, captured};
use crate::vm::GpuVm;

// ---------------------------------------------------------------------------------------------
// The rig.

/// Where test memory starts, at the same GPU and CPU address (row 30).
const BASE: u64 = 0x8000_0000;
/// Pages of test memory.
const PAGES: u64 = 64;
/// Not channel 0, so a hard-coded channel fails.
const CHANNEL: ChannelId = ChannelId(3);
/// A second channel, in the same address space.
const OTHER: ChannelId = ChannelId(9);
/// Not space 0.
const SPACE: AsId = AsId(0x5a);

const VERTEX_PROGRAM: u64 = BASE + 0x1000;
const PIXEL_PROGRAM: u64 = BASE + 0x2000;
const VERTICES: u64 = BASE + 0x3000;
const UNIFORMS: u64 = BASE + 0x5000;
const COMPUTE_PROGRAM: u64 = BASE + 0x6000;
const QMD: u64 = BASE + 0x7000;
const INLINE_QMD: u64 = BASE + 0x7400;
const SEM: u64 = BASE + 0x9000;
const SEM2: u64 = BASE + 0x9010;
const REPORT: u64 = BASE + 0x9100;
const SRC: u64 = BASE + 0xa000;
const DST: u64 = BASE + 0xb000;
const TARGET: u64 = BASE + 0x10000;
const FRAME: u64 = BASE + 0x20000;
/// The target's and frame's size in pixels.
const SIZE: u32 = 8;

const THREE_D: u32 = clc797::CLASS;
const COMPUTE: u32 = clc7c0::CLASS;

type Memory = TestGpuMemory<FlatMemory>;

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

fn memory() -> Arc<Memory> {
    let size = usize::try_from(PAGES * PAGE_SIZE).unwrap();
    let mem = TestGpuMemory::new(FlatMemory::new(BASE, size));
    mem.map(PageRange::new(BASE, PAGES * PAGE_SIZE).unwrap(), BASE)
        .unwrap();
    Arc::new(mem)
}

/// What the fake shader cache saw.
#[derive(Default)]
struct ShaderLog {
    lookups: Vec<(Vec<ProgramRequest>, VariantKey)>,
    /// The answer to the compute constant buffer question it asks: bank 0, offset 0.
    compute_word: Vec<Option<u32>>,
    invalidated: Vec<PageRange>,
}

/// The fake shader cache: a vertex and pixel set for draws, a one-shader set for compute.
struct FakeShaders {
    log: Arc<Mutex<ShaderLog>>,
    draw: SetLookup,
    compute: SetLookup,
}

impl ShaderLookup for FakeShaders {
    fn lookup(
        &mut self,
        requests: &[ProgramRequest],
        variant: &VariantKey,
        _memory: &dyn ProgramMemory,
        state: &dyn EngineState,
    ) -> SetLookup {
        let mut log = self.log.lock().unwrap_or_else(PoisonError::into_inner);
        log.lookups.push((requests.to_vec(), variant.clone()));
        if requests.first().map(|r| r.stage) == Some(GuestStage::Compute) {
            log.compute_word
                .push(state.cbuf_word(GuestStage::Compute, 0, 0));
            self.compute.clone()
        } else {
            self.draw.clone()
        }
    }

    fn wait(&mut self, _ticket: Ticket) -> SetLookup {
        self.draw.clone()
    }
}

impl InvalidationSink for FakeShaders {
    fn invalidate(&mut self, range: PageRange) {
        self.log
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .invalidated
            .push(range);
    }
}

const VERTEX_STAMP: ShaderStamp = ShaderStamp::new(0x11);
const PIXEL_STAMP: ShaderStamp = ShaderStamp::new(0x22);
const COMPUTE_STAMP: ShaderStamp = ShaderStamp::new(0x33);

fn draw_set() -> SetLookup {
    SetLookup::Ready(ReadySet {
        shaders: Arc::new(ShaderSet {
            stages: vec![
                host_shader(GuestStage::Vertex),
                host_shader(GuestStage::Pixel),
            ],
        }),
        stages: vec![
            StageInfo {
                stamp: VERTEX_STAMP,
                resources: vec![],
            },
            StageInfo {
                stamp: PIXEL_STAMP,
                resources: vec![],
            },
        ],
    })
}

/// A compute set: the shader crate translates no compute program yet, so the set's one shader
/// is a translated vertex program, which the executor makes a compute shader (module docs of
/// `compute`). It reads constant buffer 0.
fn compute_set(resources: Vec<ResourceUse>) -> SetLookup {
    SetLookup::Ready(ReadySet {
        shaders: Arc::new(ShaderSet {
            stages: vec![host_shader(GuestStage::Vertex)],
        }),
        stages: vec![StageInfo {
            stamp: COMPUTE_STAMP,
            resources,
        }],
    })
}

fn uniform_bank(bank: u8, size: u32) -> ResourceUse {
    ResourceUse {
        slot: Slot {
            group: 0,
            binding: 0,
        },
        kind: ResourceKind::UniformBuffer,
        source: ResourceSource::ConstantBuffer { bank, size },
    }
}

struct Rig {
    clock: Arc<Clock>,
    recording: Option<Arc<RecordingBackend>>,
    late: Option<Arc<LateBackend>>,
    exec: Executor,
    mem: Arc<Memory>,
    watch: Arc<WriteWatch>,
    shaders: Arc<Mutex<ShaderLog>>,
    ordered: Arc<OrderedTable>,
}

impl Rig {
    fn new() -> Self {
        let recording = Arc::new(RecordingBackend::new(caps(), 1 << 40));
        let mut rig = Self::on(
            Arc::<RecordingBackend>::clone(&recording),
            compute_set(vec![]),
        );
        rig.recording = Some(recording);
        rig
    }

    fn late() -> Self {
        let late = Arc::new(LateBackend::with_caps(
            caps(),
            1 << 40,
            CompletionWake::none(),
        ));
        let mut rig = Self::on(Arc::<LateBackend>::clone(&late), compute_set(vec![]));
        rig.late = Some(late);
        rig
    }

    fn on(backend: Arc<dyn Backend>, compute: SetLookup) -> Self {
        Self::with(backend, compute, ExecConfig::default())
    }

    fn with(backend: Arc<dyn Backend>, compute: SetLookup, config: ExecConfig) -> Self {
        let clock = Arc::new(Clock::default());
        let pushed = Arc::new(Pushed::default());
        let waiters = Arc::new(WaiterTable::new());
        let env = FenceEnv {
            syncpoints: Arc::new(Syncpoints::new(Arc::<Pushed>::clone(&pushed))),
            waiters,
            completions: pushed,
            clock: Arc::<Clock>::clone(&clock),
            front: Arc::new(Front::default()),
        };
        let fences = Fences::new(Arc::clone(&backend), env, Mode::Fenced);
        let shaders = Arc::new(Mutex::new(ShaderLog::default()));
        let fake = FakeShaders {
            log: Arc::clone(&shaders),
            draw: draw_set(),
            compute,
        };
        let ordered = Arc::new(OrderedTable::new());
        let mut exec = Executor::new(
            backend,
            fences,
            Box::new(fake),
            Arc::clone(&ordered),
            config,
        );
        let mem = memory();
        let watch = Arc::new(WriteWatch::new());
        exec.add_space(SPACE, Arc::<Memory>::clone(&mem), Arc::clone(&watch))
            .unwrap();
        exec.add_channel(CHANNEL, SPACE).unwrap();
        exec.add_channel(OTHER, SPACE).unwrap();
        Self {
            clock,
            recording: None,
            late: None,
            exec,
            mem,
            watch,
            shaders,
            ordered,
        }
    }

    /// Applies every line of ring text `text`, each of which must be accepted.
    fn run(&mut self, text: &str) {
        for entry in parse_text(text).unwrap() {
            let applied = self.exec.apply(&entry);
            assert_eq!(applied, Ok(()), "{entry}");
        }
    }

    /// Closes the open list as an empty ring does, and returns the recording backend's text.
    fn submit(&mut self) -> String {
        self.exec.close(CloseReason::RingEmpty).unwrap();
        self.recording.as_ref().unwrap().snapshot()
    }

    fn write_words(&self, address: u64, words: &[u32]) {
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        self.mem.guest().write(address, &bytes).unwrap();
    }

    fn word(&self, address: u64) -> u32 {
        let mut bytes = [0; 4];
        self.mem.read(address, &mut bytes).unwrap();
        u32::from_le_bytes(bytes)
    }

    fn log(&self) -> std::sync::MutexGuard<'_, ShaderLog> {
        self.shaders.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn commands(&self) -> Vec<Command> {
        self.recording
            .as_ref()
            .unwrap()
            .lists()
            .into_iter()
            .flat_map(|(_, list)| list.commands)
            .collect()
    }
}

/// A word with `fields` set.
fn fields(fields: &[(Field, u32)]) -> u32 {
    fields.iter().fold(0, |word, &(field, value)| {
        field.insert(word, value).unwrap()
    })
}

fn method_number(method: &Method, indices: &[u32]) -> u16 {
    u16::try_from(method.method_at(indices).unwrap()).unwrap()
}

/// The ring text of a write of `value` to `method` at `indices`, of `class`, on `channel`.
fn write_on(
    channel: ChannelId,
    class: u32,
    method: &Method,
    indices: &[u32],
    value: u32,
) -> String {
    let entry = RingEntry::Write {
        channel,
        class: ClassId(class),
        method: method_number(method, indices),
        value,
    };
    format!("{entry}\n")
}

fn write(class: u32, method: &Method, indices: &[u32], value: u32) -> String {
    write_on(CHANNEL, class, method, indices, value)
}

/// A 64-bit address into a pair of methods.
fn address(class: u32, upper: &Method, lower: &Method, indices: &[u32], address: u64) -> String {
    write(class, upper, indices, u32::try_from(address >> 32).unwrap())
        + &write(
            class,
            lower,
            indices,
            u32::try_from(address & 0xffff_ffff).unwrap(),
        )
}

/// The 3D state every draw test draws with: one float attribute from stream 0, a vertex and a
/// pixel program, one colour target and a viewport (the draw path's own test state).
fn draw_state() -> String {
    let mut text = String::from("bind channel=3 subchannel=0 class=0xc797\n");
    let c = THREE_D;
    text += &write(
        c,
        &clc797::OGL_SET_FRONT_FACE,
        &[],
        clc797::OGL_SET_FRONT_FACE_V_CCW,
    );
    text += &write(
        c,
        &clc797::SET_FRONT_POLYGON_MODE,
        &[],
        clc797::SET_FRONT_POLYGON_MODE_V_FILL,
    );
    text += &write(
        c,
        &clc797::SET_RASTER_ENABLE,
        &[],
        clc797::SET_RASTER_ENABLE_V_TRUE,
    );
    text += &write(
        c,
        &clc797::SET_VERTEX_ATTRIBUTE_A,
        &[0],
        fields(&[
            (clc797::SET_VERTEX_ATTRIBUTE_A_STREAM, 0),
            (clc797::SET_VERTEX_ATTRIBUTE_A_OFFSET, 4),
            (
                clc797::SET_VERTEX_ATTRIBUTE_A_COMPONENT_BIT_WIDTHS,
                clc797::SET_VERTEX_ATTRIBUTE_A_COMPONENT_BIT_WIDTHS_R32_G32_B32_A32,
            ),
            (
                clc797::SET_VERTEX_ATTRIBUTE_A_NUMERICAL_TYPE,
                clc797::SET_VERTEX_ATTRIBUTE_A_NUMERICAL_TYPE_NUM_FLOAT,
            ),
        ]),
    );
    text += &write(
        c,
        &clc797::SET_VERTEX_STREAM_A_FORMAT,
        &[0],
        fields(&[
            (clc797::SET_VERTEX_STREAM_A_FORMAT_STRIDE, 20),
            (clc797::SET_VERTEX_STREAM_A_FORMAT_ENABLE, 1),
        ]),
    );
    text += &address(
        c,
        &clc797::SET_VERTEX_STREAM_A_LOCATION_A,
        &clc797::SET_VERTEX_STREAM_A_LOCATION_B,
        &[0],
        VERTICES,
    );
    text += &address(
        c,
        &clc797::SET_VERTEX_STREAM_SIZE_A,
        &clc797::SET_VERTEX_STREAM_SIZE_B,
        &[0],
        0x100,
    );
    text += &address(
        c,
        &clc797::SET_PIPELINE_PROGRAM_ADDRESS_A,
        &clc797::SET_PIPELINE_PROGRAM_ADDRESS_B,
        &[1],
        VERTEX_PROGRAM,
    );
    text += &write(
        c,
        &clc797::SET_PIPELINE_SHADER,
        &[5],
        fields(&[(clc797::SET_PIPELINE_SHADER_ENABLE, 1)]),
    );
    text += &address(
        c,
        &clc797::SET_PIPELINE_PROGRAM_ADDRESS_A,
        &clc797::SET_PIPELINE_PROGRAM_ADDRESS_B,
        &[5],
        PIXEL_PROGRAM,
    );
    text + &target_state()
}

/// The colour target, surface clip and viewport of [`draw_state`].
fn target_state() -> String {
    let c = THREE_D;
    let mut text = write(
        c,
        &clc797::SET_CT_SELECT,
        &[],
        fields(&[
            (clc797::SET_CT_SELECT_TARGET_COUNT, 1),
            (clc797::SET_CT_SELECT_TARGET0, 0),
        ]),
    );
    text += &address(
        c,
        &clc797::SET_COLOR_TARGET_A,
        &clc797::SET_COLOR_TARGET_B,
        &[0],
        TARGET,
    );
    text += &write(c, &clc797::SET_COLOR_TARGET_WIDTH, &[0], SIZE);
    text += &write(c, &clc797::SET_COLOR_TARGET_HEIGHT, &[0], SIZE);
    text += &write(
        c,
        &clc797::SET_COLOR_TARGET_FORMAT,
        &[0],
        clc797::SET_COLOR_TARGET_FORMAT_V_A8B8G8R8,
    );
    text += &write(c, &clc797::SET_COLOR_TARGET_THIRD_DIMENSION, &[0], 1);
    text += &write(
        c,
        &clc797::SET_SURFACE_CLIP_HORIZONTAL,
        &[],
        fields(&[(clc797::SET_SURFACE_CLIP_HORIZONTAL_WIDTH, SIZE)]),
    );
    text += &write(
        c,
        &clc797::SET_SURFACE_CLIP_VERTICAL,
        &[],
        fields(&[(clc797::SET_SURFACE_CLIP_VERTICAL_HEIGHT, SIZE)]),
    );
    text += &write(
        c,
        &clc797::SET_VIEWPORT_SCALE_OFFSET,
        &[],
        clc797::SET_VIEWPORT_SCALE_OFFSET_ENABLE_TRUE,
    );
    for (method, value) in [
        (&clc797::SET_VIEWPORT_SCALE_X, 4.0f32),
        (&clc797::SET_VIEWPORT_SCALE_Y, 4.0),
        (&clc797::SET_VIEWPORT_OFFSET_X, 4.0),
        (&clc797::SET_VIEWPORT_OFFSET_Y, 4.0),
        (&clc797::SET_VIEWPORT_SCALE_Z, 0.5),
        (&clc797::SET_VIEWPORT_OFFSET_Z, 0.5),
    ] {
        text += &write(c, method, &[0], value.to_bits());
    }
    text
}

/// A draw of three vertices from the vertex array.
fn draw() -> String {
    let c = THREE_D;
    write(c, &clc797::SET_VERTEX_ARRAY_START, &[], 0)
        + &write(c, &clc797::DRAW_VERTEX_ARRAY, &[], 3)
        + &write(c, &clc797::BEGIN, &[], clc797::BEGIN_OP_TRIANGLES)
        + &write(c, &clc797::END, &[], 0)
}

/// How many `Draw` commands `commands` holds.
fn draws(commands: &[Command]) -> usize {
    commands
        .iter()
        .filter(|c| matches!(c, Command::Draw(_)))
        .count()
}

// ---------------------------------------------------------------------------------------------
// Golden lists: ring text in, command list text out.

#[test]
fn golden_draw() {
    let mut rig = Rig::new();
    rig.run(&(draw_state() + &draw()));
    assert_eq!(rig.exec.draws(CHANNEL).unwrap().counters().drawn, 1);
    insta::assert_snapshot!(rig.submit());
}

/// The size [`compute_qmd`] binds constant buffer 0 with.
const COMPUTE_BANK_SIZE: u32 = 0x100;

/// A QMD at [`QMD`] dispatching [`COMPUTE_PROGRAM`] over a 2x3x1 grid of 8x4x1 blocks, binding
/// [`COMPUTE_BANK_SIZE`] bytes of constant buffer 0 at [`UNIFORMS`], with releases in slots 0
/// and 2.
fn compute_qmd() -> [u32; QMD_WORDS] {
    let mut spec = Spec::new(COMPUTE_PROGRAM)
        .slot(0, SlotSpec::write(SEM, 0x11))
        .slot(2, SlotSpec::write(SEM2, 0x22));
    spec.dispatch.grid = [2, 3, 1];
    spec.dispatch.block = [8, 4, 1];
    spec.dispatch.shared_memory = 0x400;
    spec.dispatch.constant_buffers[0] = ConstantBuffer {
        valid: true,
        address: UNIFORMS,
        size: COMPUTE_BANK_SIZE,
    };
    encode(&spec)
}

/// Binds the compute class and signals the QMD at [`QMD`] to schedule
/// (`SEND_PCAS_A`, then `SEND_SIGNALING_PCAS2_B` with action `INVALIDATE_COPY_SCHEDULE`).
fn compute_signal() -> String {
    let c = COMPUTE;
    let mut text = String::from("bind channel=3 subchannel=1 class=0xc7c0\n");
    text += &write(
        c,
        &clc7c0::SEND_PCAS_A,
        &[],
        u32::try_from(QMD >> crate::back::qmd::QMD_ADDRESS_SHIFT).unwrap(),
    );
    text += &write(
        c,
        &clc7c0::SEND_SIGNALING_PCAS2_B,
        &[],
        fields(&[(
            clc7c0::SEND_SIGNALING_PCAS2_B_PCAS_ACTION,
            clc7c0::SEND_SIGNALING_PCAS2_B_PCAS_ACTION_INVALIDATE_COPY_SCHEDULE,
        )]),
    );
    text
}

#[test]
fn golden_dispatch_with_qmd_releases() {
    let backend = Arc::new(RecordingBackend::new(caps(), 1 << 40));
    let mut rig = Rig::on(
        Arc::<RecordingBackend>::clone(&backend),
        compute_set(vec![uniform_bank(0, 0x40)]),
    );
    rig.recording = Some(backend);
    rig.write_words(QMD, &compute_qmd());
    rig.write_words(UNIFORMS, &[0xcafe_f00d]);
    rig.run(&compute_signal());
    assert_eq!(rig.exec.counters().dispatches, 1);
    // The program, the workgroup size and the shared memory reached the lookup, and the
    // constant buffer question was answered from the QMD's binding.
    {
        let log = rig.log();
        let (requests, key) = log.lookups.last().unwrap();
        assert_eq!(
            requests,
            &vec![ProgramRequest {
                stage: GuestStage::Compute,
                address: COMPUTE_PROGRAM,
            }]
        );
        assert_eq!(key.workgroup_size, [8, 4, 1]);
        assert_eq!(key.shared_memory_size, 0x400);
        assert_eq!(log.compute_word, vec![Some(0xcafe_f00d)]);
    }
    // The releases wait on the list holding the dispatch.
    assert_eq!(rig.word(SEM), 0);
    let snapshot = rig.submit();
    assert_eq!((rig.word(SEM), rig.word(SEM2)), (0, 0));
    rig.exec.complete();
    assert_eq!((rig.word(SEM), rig.word(SEM2)), (0x11, 0x22));
    insta::assert_snapshot!(snapshot);
}

/// A one-line pitch copy of 16 bytes from [`SRC`] to [`DST`] releasing `payload` at [`SEM`].
fn dma_copy_with_semaphore(payload: u32) -> String {
    let c = clc7b5::CLASS;
    let mut text = String::from("bind channel=3 subchannel=4 class=0xc7b5\n");
    text += &address(
        c,
        &clc7b5::OFFSET_IN_UPPER,
        &clc7b5::OFFSET_IN_LOWER,
        &[],
        SRC,
    );
    text += &address(
        c,
        &clc7b5::OFFSET_OUT_UPPER,
        &clc7b5::OFFSET_OUT_LOWER,
        &[],
        DST,
    );
    text += &write(c, &clc7b5::LINE_LENGTH_IN, &[], 16);
    text += &write(c, &clc7b5::LINE_COUNT, &[], 1);
    text += &address(
        c,
        &clc7b5::SET_SEMAPHORE_A,
        &clc7b5::SET_SEMAPHORE_B,
        &[],
        SEM,
    );
    text += &write(c, &clc7b5::SET_SEMAPHORE_PAYLOAD, &[], payload);
    text += &write(
        c,
        &clc7b5::LAUNCH_DMA,
        &[],
        fields(&[
            (
                clc7b5::LAUNCH_DMA_DATA_TRANSFER_TYPE,
                clc7b5::LAUNCH_DMA_DATA_TRANSFER_TYPE_NON_PIPELINED,
            ),
            (
                clc7b5::LAUNCH_DMA_SRC_MEMORY_LAYOUT,
                clc7b5::LAUNCH_DMA_SRC_MEMORY_LAYOUT_PITCH,
            ),
            (
                clc7b5::LAUNCH_DMA_DST_MEMORY_LAYOUT,
                clc7b5::LAUNCH_DMA_DST_MEMORY_LAYOUT_PITCH,
            ),
            (
                clc7b5::LAUNCH_DMA_SEMAPHORE_TYPE,
                clc7b5::LAUNCH_DMA_SEMAPHORE_TYPE_RELEASE_ONE_WORD_SEMAPHORE,
            ),
        ]),
    );
    text
}

#[test]
fn golden_copy_with_a_semaphore() {
    let mut rig = Rig::late();
    rig.write_words(SRC, &[1, 2, 3, 4]);
    // A draw first, so the copy's release has work before it in the open list.
    rig.run(&(draw_state() + &draw() + &dma_copy_with_semaphore(0x77)));
    let late = rig.late.clone().unwrap();
    // The copy ran on the CPU at once; its release waits for the list.
    assert_eq!(
        (0..4).map(|i| rig.word(DST + i * 4)).collect::<Vec<_>>(),
        vec![1, 2, 3, 4]
    );
    rig.exec.close(CloseReason::RingEmpty).unwrap();
    rig.exec.complete();
    assert_eq!(rig.word(SEM), 0);
    late.finish(FenceValue::new(1));
    rig.exec.complete();
    assert_eq!(rig.word(SEM), 0x77);
    insta::assert_snapshot!(late.inner.snapshot());
}

/// A present of the 8x8 RGBA8 frame at [`FRAME`] showing all of it.
fn present() -> String {
    let request = PresentRequest {
        image: ImageDesc {
            address: FRAME,
            width: SIZE,
            height: SIZE,
            pitch: 1,
            block_height_log2: 0,
            format: present::PRESENT_FORMAT_RGBA8,
        },
        shown: Rect {
            x: 0,
            y: 0,
            width: SIZE,
            height: SIZE,
        },
        layer: LayerId(1),
        display: DisplayId(0),
    };
    format!("{}\n", RingEntry::Present(request))
}

#[test]
fn golden_present_ordered_after_earlier_work() {
    let mut rig = Rig::new();
    rig.exec.set_present_space(SPACE).unwrap();
    rig.run(&(draw_state() + &draw() + &present() + &draw() + &present()));
    let snapshot = rig.submit();
    assert_eq!(rig.exec.fences().counters().closed.present, 2);
    assert_eq!(rig.exec.counters().presents, 2);
    insta::assert_snapshot!(snapshot);
    // Each list ends with its present: the first holds the first draw, the second the second.
    let lists = rig.recording.as_ref().unwrap().lists();
    assert_eq!(lists.len(), 2);
    for (index, (_, list)) in lists.iter().enumerate() {
        assert_eq!(draws(&list.commands), 1);
        let frame = u64::try_from(index).unwrap();
        assert!(
            matches!(
                list.commands.last(),
                Some(Command::Present(desc)) if desc.frame == frame && desc.src_rect.width == SIZE
            ),
            "list {index} does not end with its present"
        );
    }
}

#[test]
fn golden_wait_for_idle_closes_the_list() {
    let mut rig = Rig::new();
    rig.run(&(draw_state() + &draw() + "wait-for-idle channel=3\n" + &draw()));
    let snapshot = rig.submit();
    let counters = rig.exec.fences().counters().closed;
    assert_eq!((counters.wait_for_idle, counters.ring_empty), (1, 1));
    insta::assert_snapshot!(snapshot);
}

#[test]
fn set_reference_closes_the_list_and_attaches_nothing() {
    let mut rig = Rig::late();
    rig.run(&(draw_state() + &draw() + "set-reference channel=3 value=0x5\n"));
    assert_eq!(rig.exec.fences().counters().closed.set_reference, 1);
    assert_eq!(rig.exec.fences().counters().fenced, 0);
    assert_eq!(
        rig.late.as_ref().unwrap().submitted(),
        vec![FenceValue::new(1)]
    );
}

// ---------------------------------------------------------------------------------------------
// Releases

#[test]
fn a_release_is_attached_to_the_list_that_recorded_the_work_before_it() {
    let mut rig = Rig::late();
    let release = |payload: u32| {
        format!(
            "release channel=3 address={SEM:#x} payload={payload:#x} size=4 timestamp=no op=write\n"
        )
    };
    rig.run(&(draw_state() + &draw() + &release(1) + "wait-for-idle channel=3\n"));
    rig.run(&(draw() + &release(2)));
    rig.exec.close(CloseReason::RingEmpty).unwrap();
    let late = rig.late.clone().unwrap();
    assert_eq!(
        late.submitted(),
        vec![FenceValue::new(1), FenceValue::new(2)]
    );
    rig.exec.complete();
    assert_eq!(rig.word(SEM), 0, "no list has completed");
    late.finish(FenceValue::new(1));
    rig.exec.complete();
    assert_eq!(rig.word(SEM), 1, "the first list's release only");
    late.finish(FenceValue::new(2));
    rig.exec.complete();
    assert_eq!(rig.word(SEM), 2);
}

#[test]
fn a_release_is_written_through_its_channels_address_space() {
    // Channel 5 lives in a second space whose memory maps SEM elsewhere: the release must land
    // there, not in the first space's memory.
    let mut rig = Rig::new();
    let second = AsId(SPACE.0 + 1);
    let other_mem = memory();
    rig.exec
        .add_space(
            second,
            Arc::<Memory>::clone(&other_mem),
            Arc::new(WriteWatch::new()),
        )
        .unwrap();
    rig.exec.add_channel(ChannelId(5), second).unwrap();
    rig.run(&format!(
        "release channel=5 address={SEM:#x} payload=0x9 size=4 timestamp=no op=write\n"
    ));
    let mut bytes = [0; 4];
    other_mem.read(SEM, &mut bytes).unwrap();
    assert_eq!(u32::from_le_bytes(bytes), 9);
    assert_eq!(rig.word(SEM), 0);
}

#[test]
fn a_marker_carries_the_ordered_update_filed_under_its_token() {
    let mut rig = Rig::new();
    let vm = Arc::new(GpuVm::new());
    let range = PageRange::new(UNIFORMS, PAGE_SIZE).unwrap();
    rig.ordered.insert(
        MarkerToken(7),
        OrderedUpdate {
            change: Some(VmChange {
                space: SPACE,
                vm,
                update: VmUpdate::Unmap { range },
            }),
            ticket: None,
        },
    );
    rig.run("marker token=0x7\n");
    assert!(rig.ordered.is_empty());
    // Nothing in flight: performed at once, and the range reached the shader cache.
    assert_eq!(rig.log().invalidated, vec![range]);
}

// ---------------------------------------------------------------------------------------------
// Binding

#[test]
fn an_entry_for_an_unbound_class_is_the_exact_error() {
    let mut rig = Rig::new();
    let entry = parse_text("write channel=3 class=0xc797 method=0x45 value=0x1\n").unwrap();
    let error = rig.exec.apply(&entry[0]).unwrap_err();
    assert_eq!(
        error,
        ExecError::Unbound {
            channel: CHANNEL,
            class: ClassId(THREE_D),
        }
    );
    assert_eq!(
        error.to_string(),
        "channel 3 wrote to class 0xc797, which none of its subchannels has bound"
    );
    assert_eq!(rig.exec.counters().failed, 1);
    assert!(rig.exec.mirror(CHANNEL, ClassId(THREE_D)).is_none());
}

#[test]
fn an_entry_for_an_unknown_channel_is_the_exact_error() {
    let mut rig = Rig::new();
    let entry = parse_text("wait-for-idle channel=44\n").unwrap();
    let error = rig.exec.apply(&entry[0]).unwrap_err();
    assert_eq!(
        error,
        ExecError::UnknownChannel {
            channel: ChannelId(44)
        }
    );
    assert_eq!(error.to_string(), "channel 44 is not known to the back");
    assert_eq!(rig.exec.fences().counters().closed.wait_for_idle, 0);
}

#[test]
fn an_unbound_class_after_an_unbind_is_refused() {
    let mut rig = Rig::new();
    rig.run("bind channel=3 subchannel=0 class=0xc797\nbind channel=3 subchannel=0 class=0\n");
    // Class 0 leaves the subchannel unbound, not bound to class 0.
    assert_eq!(rig.exec.bound(CHANNEL, Subchannel::new(0).unwrap()), None);
    let entry = parse_text("write channel=3 class=0xc797 method=0x45 value=0x1\n").unwrap();
    assert_eq!(
        rig.exec.apply(&entry[0]),
        Err(ExecError::Unbound {
            channel: CHANNEL,
            class: ClassId(THREE_D),
        })
    );
}

#[test]
fn a_rebind_keeps_the_classs_registers() {
    let mut rig = Rig::new();
    let method = clc797::SET_COLOR_TARGET_WIDTH.method_at(&[0]).unwrap();
    rig.run(
        &("bind channel=3 subchannel=0 class=0xc797\n".to_owned()
            + &write(THREE_D, &clc797::SET_COLOR_TARGET_WIDTH, &[0], 0x123)
            + "bind channel=3 subchannel=0 class=0xc7c0\n"
            + "bind channel=3 subchannel=0 class=0xc797\n"),
    );
    let mirror = rig.exec.mirror(CHANNEL, ClassId(THREE_D)).unwrap();
    assert_eq!(mirror.value(method), Some(0x123));
    // The other channel's 3D engine is its own.
    rig.run("bind channel=9 subchannel=0 class=0xc797\n");
    assert_eq!(
        rig.exec
            .mirror(OTHER, ClassId(THREE_D))
            .unwrap()
            .value(method),
        Some(0)
    );
}

#[test]
fn subchannel_2_starts_bound_to_inline_to_memory() {
    let mut rig = Rig::new();
    let two = Subchannel::new(2).unwrap();
    assert_eq!(rig.exec.bound(CHANNEL, two), Some(ClassId(cla140::CLASS)));
    for index in [0, 1, 3, 4, 5, 6, 7] {
        assert_eq!(
            rig.exec.bound(CHANNEL, Subchannel::new(index).unwrap()),
            None
        );
    }
    // An inline upload of one word to DST with no bind.
    let c = cla140::CLASS;
    let mut text = address(c, &cla140::OFFSET_OUT_UPPER, &cla140::OFFSET_OUT, &[], DST);
    text += &write(c, &cla140::LINE_LENGTH_IN, &[], 4);
    text += &write(c, &cla140::LINE_COUNT, &[], 1);
    text += &write(
        c,
        &cla140::LAUNCH_DMA,
        &[],
        fields(&[(
            cla140::LAUNCH_DMA_DST_MEMORY_LAYOUT,
            cla140::LAUNCH_DMA_DST_MEMORY_LAYOUT_PITCH,
        )]),
    );
    text += &write(c, &cla140::LOAD_INLINE_DATA, &[], 0xabcd_0123);
    rig.run(&text);
    assert_eq!(rig.word(DST), 0xabcd_0123);
}

#[test]
fn words_to_a_class_with_no_engine_are_counted_and_dropped() {
    let mut rig = Rig::new();
    rig.run(
        "bind channel=3 subchannel=6 class=0xb197\n\
         write channel=3 class=0xb197 method=0x45 value=0x1\n\
         run channel=3 class=0xb197 method=0x46 words=0x1,0x2\n",
    );
    assert_eq!(rig.exec.counters().unhandled_words, 3);
    assert!(rig.exec.mirror(CHANNEL, ClassId(0xb197)).is_none());
}

#[test]
fn capture_control_is_counted() {
    let mut rig = Rig::new();
    rig.run("capture command=start\ncapture command=stop\n");
    assert_eq!(rig.exec.counters().captures, 2);
}

#[test]
fn spaces_and_channels_are_checked_when_added() {
    let mut rig = Rig::new();
    assert_eq!(
        rig.exec.add_space(
            SPACE,
            Arc::<Memory>::clone(&rig.mem),
            Arc::clone(&rig.watch)
        ),
        Err(ExecError::DuplicateSpace { space: SPACE })
    );
    assert_eq!(
        rig.exec.add_channel(CHANNEL, SPACE),
        Err(ExecError::DuplicateChannel { channel: CHANNEL })
    );
    // Both faults at once: the duplicate channel is reported first.
    assert_eq!(
        rig.exec.add_channel(CHANNEL, AsId(1)),
        Err(ExecError::DuplicateChannel { channel: CHANNEL })
    );
    assert_eq!(
        rig.exec.add_channel(ChannelId(1), AsId(1)),
        Err(ExecError::UnknownSpace { space: AsId(1) })
    );
    assert_eq!(rig.exec.remove_channel(OTHER), Ok(true));
    assert_eq!(rig.exec.remove_channel(OTHER), Ok(false));
}

// ---------------------------------------------------------------------------------------------
// Presents

#[test]
fn a_present_needs_a_present_space_and_a_known_format() {
    let mut rig = Rig::new();
    let entry = parse_text(&present()).unwrap();
    let error = rig.exec.apply(&entry[0]).unwrap_err();
    assert_eq!(error, ExecError::NoPresentSpace);
    assert_eq!(
        error.to_string(),
        "a present arrived, but no address space is set for presents"
    );
    rig.exec.set_present_space(SPACE).unwrap();
    let bad = present().replace("format=1", "format=2");
    let entry = parse_text(&bad).unwrap();
    let error = rig.exec.apply(&entry[0]).unwrap_err();
    assert_eq!(error, ExecError::PresentFormat { format: 2 });
    assert_eq!(
        error.to_string(),
        "present format 2 is not supported (1 is RGBA8, 4 is RGB10A2)"
    );
    assert_eq!(
        rig.exec.set_present_space(AsId(1)),
        Err(ExecError::UnknownSpace { space: AsId(1) })
    );
}

#[test]
fn present_formats_are_the_graphics_servers_words() {
    // graphics-server, "What vi2 needs from the image": "the format (1 RGBA8, 4 RGB10A2)".
    assert_eq!(present::PRESENT_FORMAT_RGBA8, 1);
    assert_eq!(present::PRESENT_FORMAT_RGB10A2, 4);
    assert_eq!(
        present::present_format(1),
        Ok(starquake_render::PixelFormat::A8B8G8R8Unorm)
    );
    assert_eq!(
        present::present_format(4),
        Ok(starquake_render::PixelFormat::A2B10G10R10Unorm)
    );
    for format in [0, 2, 3, 5, u32::MAX] {
        assert_eq!(
            present::present_format(format),
            Err(ExecError::PresentFormat { format })
        );
    }
}

#[test]
fn a_presented_frame_the_gpu_drew_is_its_render_target() {
    // The frame is the colour target the draw rendered into: one texture, no upload over it.
    let mut rig = Rig::new();
    rig.exec.set_present_space(SPACE).unwrap();
    rig.run(
        &(draw_state()
            + &draw()
            + &present().replace(&format!("{FRAME:#x}"), &format!("{TARGET:#x}"))),
    );
    let commands = rig.commands();
    let presented = commands
        .iter()
        .find_map(|c| match c {
            Command::Present(desc) => Some(desc.view),
            _ => None,
        })
        .unwrap();
    let target = commands
        .iter()
        .find_map(|c| match c {
            Command::SetRenderTargets(targets) => targets.colors[0],
            _ => None,
        })
        .unwrap();
    let snapshot = rig.recording.as_ref().unwrap().snapshot();
    // The present's view is a view of the target's texture.
    let texture_of = |view: starquake_render::ViewId| {
        let line = snapshot
            .lines()
            .find(|l| l.starts_with(&format!("create v{}:", view.index())))
            .unwrap()
            .to_owned();
        line.split(" of ")
            .nth(1)
            .unwrap()
            .split(' ')
            .next()
            .unwrap()
            .to_owned()
    };
    assert_eq!(texture_of(presented), texture_of(target));
}

/// The `dump` flag of every present follows [`ExecConfig::dumps`] (module docs, "Frame
/// dumps"): two presents under each setting, both flagged or neither, and the recording backend
/// reads back exactly the flagged ones.
#[test]
fn a_present_is_dumped_as_the_setting_says() {
    // The default dumps nothing.
    assert_eq!(ExecConfig::default().dumps, FrameDumps::Off);
    assert_eq!(FrameDumps::default(), FrameDumps::Off);
    for (dumps, flagged) in [(FrameDumps::Off, false), (FrameDumps::Every, true)] {
        let recording = Arc::new(RecordingBackend::new(caps(), 1 << 40));
        let config = ExecConfig {
            dumps,
            ..ExecConfig::default()
        };
        let mut rig = Rig::with(
            Arc::<RecordingBackend>::clone(&recording),
            compute_set(vec![]),
            config,
        );
        rig.recording = Some(Arc::clone(&recording));
        rig.exec.set_present_space(SPACE).unwrap();
        rig.run(&(present() + &present()));
        let presents: Vec<(u64, bool)> = rig
            .commands()
            .iter()
            .filter_map(|command| match command {
                Command::Present(desc) => Some((desc.frame, desc.dump)),
                _ => None,
            })
            .collect();
        assert_eq!(presents, [(0, flagged), (1, flagged)], "{dumps:?}");
        let dumped: Vec<u64> = recording
            .take_frame_dumps()
            .iter()
            .map(|dump| dump.frame)
            .collect();
        let expected: &[u64] = if flagged { &[0, 1] } else { &[] };
        assert_eq!(dumped, expected, "{dumps:?}");
    }
}

// ---------------------------------------------------------------------------------------------
// The drain, invalidations and deferred draws

/// Makes the guest write the page at `va` as the kernel would report it: requested, armed, then
/// faulted.
fn cpu_write(watch: &WriteWatch, va: u64) {
    watch.request(PageRange::new(va - va % PAGE_SIZE, PAGE_SIZE).unwrap());
    let mut requests = Vec::new();
    watch.take_requests(&mut requests);
    watch.arm(&requests).unwrap();
    assert!(watch.record_fault(va));
}

#[test]
fn an_invalidate_entry_drains_cpu_writes_to_the_shader_cache() {
    let mut rig = Rig::new();
    cpu_write(&rig.watch, UNIFORMS + 8);
    rig.run("invalidate channel=3 what=l2-sysmem-invalidate\n");
    assert_eq!(
        rig.log().invalidated,
        vec![PageRange::new(UNIFORMS, PAGE_SIZE).unwrap()]
    );
}

#[test]
fn a_draw_drains_cpu_writes_first_and_other_words_do_not() {
    let mut rig = Rig::new();
    rig.run(&draw_state());
    cpu_write(&rig.watch, UNIFORMS);
    // A state word starts nothing: no drain.
    rig.run(&write(THREE_D, &clc797::SET_COLOR_TARGET_WIDTH, &[0], SIZE));
    assert_eq!(rig.log().invalidated, vec![]);
    rig.run(&draw());
    assert_eq!(
        rig.log().invalidated,
        vec![PageRange::new(UNIFORMS, PAGE_SIZE).unwrap()]
    );
}

#[test]
fn a_copy_invalidates_its_destination_in_the_shader_cache() {
    let mut rig = Rig::new();
    rig.run(&dma_copy_with_semaphore(1));
    assert_eq!(
        rig.log().invalidated,
        vec![PageRange::new(DST, PAGE_SIZE).unwrap()]
    );
}

#[test]
fn the_inline_qmd_is_invalidated_in_the_caches() {
    let mut rig = Rig::new();
    let c = COMPUTE;
    let mut text = String::from("bind channel=3 subchannel=1 class=0xc7c0\n");
    text += &address(
        c,
        &clc7c0::SET_INLINE_QMD_ADDRESS_A,
        &clc7c0::SET_INLINE_QMD_ADDRESS_B,
        &[],
        INLINE_QMD >> crate::back::qmd::QMD_ADDRESS_SHIFT,
    );
    // A QMD that is not version 3.0 and needs a scheduling action: written, not scheduled.
    let mut words = [0u32; QMD_WORDS];
    words[4] = 1 << 12;
    // The front sends each word to its own element (an array of registers, not a trigger).
    for (index, word) in (0u32..).zip(words) {
        text += &write(c, &clc7c0::LOAD_INLINE_QMD_DATA, &[index], word);
    }
    rig.run(&text);
    let mut bytes = [0; 4];
    rig.mem.read(INLINE_QMD + 16, &mut bytes).unwrap();
    assert_eq!(u32::from_le_bytes(bytes), 1 << 12);
    assert_eq!(
        rig.log().invalidated,
        vec![PageRange::new(INLINE_QMD - INLINE_QMD % PAGE_SIZE, PAGE_SIZE).unwrap()]
    );
}

#[test]
fn deferred_instances_are_drawn_before_the_channels_next_release() {
    let mut rig = Rig::late();
    let c = THREE_D;
    // BEGIN FIRST ... END, then BEGIN SUBSEQUENT ... END: the instances wait.
    let instanced = write(c, &clc797::SET_VERTEX_ARRAY_START, &[], 0)
        + &write(c, &clc797::DRAW_VERTEX_ARRAY, &[], 3)
        + &write(c, &clc797::BEGIN, &[], clc797::BEGIN_OP_TRIANGLES)
        + &write(c, &clc797::END, &[], 0)
        + &write(
            c,
            &clc797::BEGIN,
            &[],
            fields(&[
                (clc797::BEGIN_OP, clc797::BEGIN_OP_TRIANGLES),
                (
                    clc797::BEGIN_INSTANCE_ID,
                    clc797::BEGIN_INSTANCE_ID_SUBSEQUENT,
                ),
            ]),
        )
        + &write(c, &clc797::END, &[], 0);
    rig.run(&(draw_state() + &instanced));
    assert!(rig.exec.draws(CHANNEL).unwrap().has_deferred());
    rig.run(&format!(
        "release channel=3 address={SEM:#x} payload=0x1 size=4 timestamp=no op=write\n"
    ));
    assert!(!rig.exec.draws(CHANNEL).unwrap().has_deferred());
    rig.exec.close(CloseReason::RingEmpty).unwrap();
    let lists = rig.late.as_ref().unwrap().inner.lists();
    let (_, list) = lists.last().unwrap();
    // The deferred draw, of both instances, is the list's last command.
    assert!(
        matches!(
            list.commands.last(),
            Some(Command::Draw(desc)) if matches!(
                desc.params,
                starquake_render::command::DrawParams::Direct {
                    instance_count: 2,
                    ..
                }
            )
        ),
        "{list:?}"
    );
}

#[test]
fn two_channels_drawing_into_one_list_each_set_their_targets() {
    let mut rig = Rig::new();
    let other_state = draw_state().replace("channel=3 ", "channel=9 ");
    rig.run(&(draw_state() + &other_state + &draw()));
    rig.run(&draw().replace("channel=3 ", "channel=9 "));
    rig.run(&draw());
    rig.submit();
    let commands = rig.commands();
    let targets = commands
        .iter()
        .filter(|c| matches!(c, Command::SetRenderTargets(_)))
        .count();
    assert_eq!(draws(&commands), 3);
    assert_eq!(targets, 3, "{commands:?}");
}

// ---------------------------------------------------------------------------------------------
// Conditional rendering

fn conditional(mode: u32) -> String {
    let c = THREE_D;
    write(
        c,
        &clc797::SET_RENDER_ENABLE_OVERRIDE,
        &[],
        clc797::SET_RENDER_ENABLE_OVERRIDE_MODE_USE_RENDER_ENABLE,
    ) + &address(
        c,
        &clc797::SET_RENDER_ENABLE_A,
        &clc797::SET_RENDER_ENABLE_B,
        &[],
        REPORT,
    ) + &write(c, &clc797::SET_RENDER_ENABLE_C, &[], mode)
}

#[test]
fn conditional_rendering_catches_up_with_a_pending_release_of_its_report() {
    // The report says "equal" only once the release of its current sequence is performed.
    for (pending, drawn) in [(true, 2), (false, 1)] {
        let mut rig = Rig::new();
        rig.write_words(REPORT, &[5, 1, 0, 0, 4, 1]);
        rig.run(&(draw_state() + &draw()));
        if pending {
            rig.run(&format!(
                "release channel=3 address={:#x} payload=0x5 size=4 timestamp=no op=write\n",
                REPORT + 16
            ));
        }
        rig.run(&(conditional(clc797::SET_RENDER_ENABLE_C_MODE_RENDER_IF_EQUAL) + &draw()));
        assert_eq!(
            rig.exec.draws(CHANNEL).unwrap().counters().drawn,
            drawn,
            "pending {pending}"
        );
        assert_eq!(rig.exec.counters().catch_ups, u64::from(pending));
    }
}

#[test]
fn a_conditional_mode_that_reads_no_report_catches_up_with_nothing() {
    let mut rig = Rig::new();
    rig.run(&(draw_state() + &draw()));
    rig.run(&format!(
        "release channel=3 address={:#x} payload=0x5 size=4 timestamp=no op=write\n",
        REPORT + 16
    ));
    rig.run(&conditional(clc797::SET_RENDER_ENABLE_C_MODE_TRUE));
    assert_eq!(rig.exec.counters().catch_ups, 0);
    assert_eq!(rig.exec.fences().counters().closed.catch_up, 0);
    assert_eq!(rig.word(REPORT + 16), 0);
}

// ---------------------------------------------------------------------------------------------
// Compute

#[test]
fn a_rejected_compute_program_still_completes_and_releases() {
    let backend = Arc::new(RecordingBackend::new(caps(), 1 << 40));
    let mut rig = Rig::on(
        Arc::<RecordingBackend>::clone(&backend),
        SetLookup::Rejected(Arc::new(starquake_shader::lookup::Rejection::Translate(
            starquake_shader::translate::TranslateError::Stage(
                starquake_shader::translate::Unsupported::Stage(GuestStage::Compute),
            ),
        ))),
    );
    rig.recording = Some(backend);
    rig.write_words(QMD, &compute_qmd());
    rig.run(&compute_signal());
    assert_eq!(rig.exec.counters().dispatches, 0);
    assert_eq!(rig.exec.counters().dispatches_skipped, 1);
    // Nothing recorded, nothing in flight: the releases are performed at once.
    assert_eq!((rig.word(SEM), rig.word(SEM2)), (0x11, 0x22));
}

#[test]
fn a_grid_past_the_host_limit_is_refused_and_still_releases() {
    let backend = Arc::new(RecordingBackend::new(caps(), 1 << 40));
    let mut rig = Rig::on(
        Arc::<RecordingBackend>::clone(&backend),
        compute_set(vec![uniform_bank(0, 0x40)]),
    );
    rig.recording = Some(backend);
    let limit = caps().limits.max_compute_workgroups_per_dimension;
    for (grid_x, dispatched) in [(limit, 1), (limit + 1, 0)] {
        let mut spec = Spec::new(COMPUTE_PROGRAM).slot(0, SlotSpec::write(SEM, grid_x));
        spec.dispatch.grid = [grid_x, 1, 1];
        spec.dispatch.constant_buffers[0] = ConstantBuffer {
            valid: true,
            address: UNIFORMS,
            size: 1 << CONSTANT_BUFFER_SIZE_SHIFT,
        };
        rig.write_words(QMD, &encode(&spec));
        let before = rig.exec.counters().dispatches;
        let ((), events) = captured(|| rig.run(&compute_signal()));
        assert_eq!(
            rig.exec.counters().dispatches - before,
            dispatched,
            "{grid_x}"
        );
        if dispatched == 0 {
            assert!(
                at(&events, tracing::Level::WARN)
                    .iter()
                    .any(|e| e.contains("passes the host's 64 workgroups")),
                "{events:?}"
            );
        }
        rig.submit();
        rig.exec.complete();
        assert_eq!(rig.word(SEM), grid_x);
    }
}

#[test]
fn a_compute_shader_binding_a_bank_the_qmd_does_not_bind_is_refused() {
    let backend = Arc::new(RecordingBackend::new(caps(), 1 << 40));
    let mut rig = Rig::on(
        Arc::<RecordingBackend>::clone(&backend),
        compute_set(vec![uniform_bank(1, 0x40)]),
    );
    rig.recording = Some(backend);
    rig.write_words(QMD, &compute_qmd());
    let ((), events) = captured(|| rig.run(&compute_signal()));
    assert_eq!(rig.exec.counters().dispatches_skipped, 1);
    assert!(
        at(&events, tracing::Level::WARN).iter().any(|e| e
            .contains("the compute shader reads constant buffer 1, which the QMD does not bind")),
        "{events:?}"
    );
    assert_eq!(rig.word(SEM), 0x11);
}

// ---------------------------------------------------------------------------------------------
// The production path: entries through the ring

#[test]
fn entries_through_the_ring_are_applied_and_the_list_closes_when_it_empties() {
    let mut rig = Rig::new();
    let (mut producer, mut consumer) = crate::ring::ring(1 << 12).unwrap();
    for entry in parse_text(&(draw_state() + &draw())).unwrap() {
        producer.push(&entry).unwrap();
    }
    producer.publish();
    let mut applied = 0;
    loop {
        let step = rig
            .exec
            .step(&mut consumer, Duration::from_millis(10))
            .unwrap();
        assert!(!matches!(step, Step::Refused(_)), "{step:?}");
        match step {
            Step::Applied => applied += 1,
            Step::Refused(_) | Step::Completed => {}
            Step::TimedOut => break,
        }
    }
    assert_eq!(
        applied,
        parse_text(&(draw_state() + &draw())).unwrap().len()
    );
    assert_eq!(rig.exec.fences().counters().closed.ring_empty, 1);
    assert_eq!(draws(&rig.commands()), 1);
}

#[test]
fn a_refused_entry_through_the_ring_does_not_stop_the_thread() {
    let mut rig = Rig::new();
    let (mut producer, mut consumer) = crate::ring::ring(1 << 12).unwrap();
    for entry in parse_text("wait-for-idle channel=44\ncapture command=start\n").unwrap() {
        producer.push(&entry).unwrap();
    }
    producer.publish();
    let first = rig
        .exec
        .step(&mut consumer, Duration::from_millis(10))
        .unwrap();
    assert_eq!(
        first,
        Step::Refused(ExecError::UnknownChannel {
            channel: ChannelId(44)
        })
    );
    let second = rig
        .exec
        .step(&mut consumer, Duration::from_millis(10))
        .unwrap();
    assert_eq!(second, Step::Applied);
    assert_eq!(rig.exec.counters().captures, 1);
}

#[test]
fn errors_read_as_plain_words() {
    let cases: [(ExecError, &str); 4] = [
        (
            ExecError::UnknownSpace { space: AsId(7) },
            "address space 7 is not known to the back",
        ),
        (
            ExecError::DuplicateSpace { space: AsId(7) },
            "address space 7 was already added",
        ),
        (
            ExecError::DuplicateChannel {
                channel: ChannelId(8),
            },
            "channel 8 was already added",
        ),
        (
            ExecError::Copy(CopyError::BlitClip),
            "copy: 2D blits with clipping enabled are not supported",
        ),
    ];
    for (error, text) in cases {
        assert_eq!(error.to_string(), text);
    }
    assert_eq!(
        ComputeError::BadSet {
            shaders: 0,
            infos: 1
        }
        .to_string(),
        "the compute shader set holds 0 shaders and 1 stage records, not one"
    );
    assert_eq!(
        ComputeError::Resource("a texture").to_string(),
        "the compute shader binds a texture, which compute dispatches do not supply yet"
    );
}

#[test]
fn the_default_config_waits_and_takes_the_catch_up_guess() {
    let config = ExecConfig::default();
    assert_eq!(config.setting, WaitWhileCompiling::Wait);
    assert_eq!(config.catch_up, CATCH_UP_TIMEOUT);
    assert_eq!(CATCH_UP_TIMEOUT, Duration::from_secs(5));
}

/// The 2D class's words reach the blit engine: a launch with clipping enabled is the 2D
/// engine's exact error.
#[test]
fn a_2d_launch_reaches_the_blit_engine() {
    let mut rig = Rig::new();
    let c = cl902d::CLASS;
    let text = "bind channel=3 subchannel=3 class=0x902d\n".to_owned()
        + &write(
            c,
            &cl902d::SET_CLIP_ENABLE,
            &[],
            fields(&[(cl902d::SET_CLIP_ENABLE_V, 1)]),
        )
        + &write(
            c,
            &cl902d::SET_OPERATION,
            &[],
            cl902d::SET_OPERATION_V_SRCCOPY,
        );
    rig.run(&text);
    let launch = parse_text(&write(c, &cl902d::PIXELS_FROM_MEMORY_SRC_Y0_INT, &[], 0)).unwrap();
    assert_eq!(
        rig.exec.apply(&launch[0]),
        Err(ExecError::Copy(CopyError::BlitClip))
    );
}

// ---------------------------------------------------------------------------------------------
// Lists closed mid-entry, write-backs, catch-up failures and the wait setting for compute

#[test]
fn a_list_closed_between_two_draws_of_one_run_sets_targets_again() {
    let mut rig = Rig::new();
    rig.run(&(draw_state() + &draw()));
    // The open list has recorded since 0; past the close interval, the next draw's commands
    // close it, and the draw after it starts a new list, which must set its targets again.
    rig.clock.set(crate::back::fence::CLOSE_AFTER);
    let run = RingEntry::Run {
        channel: CHANNEL,
        class: ClassId(THREE_D),
        method: method_number(&clc797::DRAW_VERTEX_ARRAY_BEGIN_END_INSTANCE_FIRST, &[]),
        words: std::borrow::Cow::Owned(vec![
            fields(&[
                (clc797::DRAW_VERTEX_ARRAY_BEGIN_END_INSTANCE_FIRST_COUNT, 3),
                (
                    clc797::DRAW_VERTEX_ARRAY_BEGIN_END_INSTANCE_FIRST_TOPOLOGY,
                    clc797::BEGIN_OP_TRIANGLES,
                ),
            ]);
            2
        ]),
    };
    rig.run(&format!("{run}\n"));
    rig.submit();
    let lists = rig.recording.as_ref().unwrap().lists();
    assert_eq!(lists.len(), 2);
    let (_, second) = &lists[1];
    assert_eq!(draws(&second.commands), 1);
    assert!(
        second
            .commands
            .iter()
            .any(|c| matches!(c, Command::SetRenderTargets(_))),
        "{second:?}"
    );
}

#[test]
fn a_dispatch_between_two_draws_makes_the_second_set_its_targets() {
    let backend = Arc::new(RecordingBackend::new(caps(), 1 << 40));
    let mut rig = Rig::on(
        Arc::<RecordingBackend>::clone(&backend),
        compute_set(vec![uniform_bank(0, 0x40)]),
    );
    rig.recording = Some(backend);
    rig.write_words(QMD, &compute_qmd());
    rig.run(&(draw_state() + &draw() + &compute_signal() + &draw()));
    rig.submit();
    let commands = rig.commands();
    let targets = commands
        .iter()
        .filter(|c| matches!(c, Command::SetRenderTargets(_)))
        .count();
    assert_eq!((draws(&commands), targets), (2, 2), "{commands:?}");
}

#[test]
fn a_copy_reading_a_gpu_drawn_target_writes_it_back_first() {
    let mut rig = Rig::new();
    let c = clc7b5::CLASS;
    let mut copy = String::from("bind channel=3 subchannel=4 class=0xc7b5\n");
    copy += &address(
        c,
        &clc7b5::OFFSET_IN_UPPER,
        &clc7b5::OFFSET_IN_LOWER,
        &[],
        TARGET,
    );
    copy += &address(
        c,
        &clc7b5::OFFSET_OUT_UPPER,
        &clc7b5::OFFSET_OUT_LOWER,
        &[],
        DST,
    );
    copy += &write(c, &clc7b5::LINE_LENGTH_IN, &[], 16);
    copy += &write(c, &clc7b5::LINE_COUNT, &[], 1);
    copy += &write(
        c,
        &clc7b5::LAUNCH_DMA,
        &[],
        fields(&[
            (
                clc7b5::LAUNCH_DMA_DATA_TRANSFER_TYPE,
                clc7b5::LAUNCH_DMA_DATA_TRANSFER_TYPE_NON_PIPELINED,
            ),
            (
                clc7b5::LAUNCH_DMA_SRC_MEMORY_LAYOUT,
                clc7b5::LAUNCH_DMA_SRC_MEMORY_LAYOUT_PITCH,
            ),
            (
                clc7b5::LAUNCH_DMA_DST_MEMORY_LAYOUT,
                clc7b5::LAUNCH_DMA_DST_MEMORY_LAYOUT_PITCH,
            ),
        ]),
    );
    rig.run(&(draw_state() + &draw()));
    let ((), events) = captured(|| rig.run(&copy));
    let counters = rig.exec.counters();
    assert_eq!(counters.write_backs, 1);
    // The recording backend cannot know the bytes a draw rendered: the write-back fails, is
    // logged, and the copy runs on guest memory as it is.
    assert_eq!(counters.write_back_failures, 1);
    assert!(
        at(&events, tracing::Level::WARN)
            .iter()
            .any(|e| e.contains("a write-back before a copy failed")),
        "{events:?}"
    );
    assert_eq!(rig.exec.fences().counters().closed.catch_up, 1);
    let readbacks = rig
        .commands()
        .iter()
        .filter(|c| matches!(c, Command::Readback { .. }))
        .count();
    assert_eq!(readbacks, 1);
}

#[test]
fn a_copy_of_memory_no_drawn_image_holds_writes_nothing_back() {
    let mut rig = Rig::new();
    rig.run(&(draw_state() + &draw() + &dma_copy_with_semaphore(1)));
    assert_eq!(rig.exec.counters().write_backs, 0);
    assert_eq!(rig.exec.fences().counters().closed.catch_up, 0);
}

#[test]
fn a_catch_up_the_host_never_finishes_is_counted_and_the_report_read_as_it_is() {
    let late = Arc::new(LateBackend::with_caps(
        caps(),
        1 << 40,
        CompletionWake::none(),
    ));
    let config = ExecConfig {
        catch_up: Duration::from_millis(1),
        ..ExecConfig::default()
    };
    let mut rig = Rig::with(
        Arc::<LateBackend>::clone(&late),
        compute_set(vec![]),
        config,
    );
    rig.late = Some(late);
    rig.write_words(REPORT, &[5, 1, 0, 0, 4, 1]);
    rig.run(&(draw_state() + &draw()));
    rig.run(&format!(
        "release channel=3 address={:#x} payload=0x5 size=4 timestamp=no op=write\n",
        REPORT + 16
    ));
    let ((), events) = captured(|| {
        rig.run(&(conditional(clc797::SET_RENDER_ENABLE_C_MODE_RENDER_IF_EQUAL) + &draw()));
    });
    let counters = rig.exec.counters();
    assert_eq!((counters.catch_ups, counters.catch_up_failures), (0, 1));
    assert!(
        at(&events, tracing::Level::WARN)
            .iter()
            .any(|e| e.contains("conditional-render catch-up failed")),
        "{events:?}"
    );
    // The report still says "not equal": the second draw was skipped.
    assert_eq!(rig.exec.draws(CHANNEL).unwrap().counters().drawn, 1);
}

#[test]
fn a_pending_compute_program_follows_the_wait_setting() {
    for (setting, waited) in [
        (WaitWhileCompiling::Skip, false),
        (WaitWhileCompiling::Wait, true),
    ] {
        let backend = Arc::new(RecordingBackend::new(caps(), 1 << 40));
        let config = ExecConfig {
            setting,
            ..ExecConfig::default()
        };
        let mut rig = Rig::with(
            Arc::<RecordingBackend>::clone(&backend),
            SetLookup::Pending(Ticket::new(4)),
            config,
        );
        rig.recording = Some(backend);
        rig.write_words(QMD, &compute_qmd());
        let ((), events) = captured(|| rig.run(&compute_signal()));
        // The fake's wait answers the draw set, two shaders: refused as a bad compute set.
        let refused = at(&events, tracing::Level::WARN)
            .iter()
            .any(|e| e.contains("the compute shader set holds 2 shaders"));
        assert_eq!(refused, waited, "{setting:?}: {events:?}");
        assert_eq!(rig.exec.counters().dispatches_skipped, 1);
        assert_eq!(rig.word(SEM), 0x11, "{setting:?}");
    }
}

#[test]
fn an_unmapped_buffer_is_destroyed_after_the_list_that_used_it() {
    let mut rig = Rig::new();
    let vm = Arc::new(GpuVm::new());
    let range = PageRange::new(VERTICES, PAGE_SIZE).unwrap();
    rig.ordered.insert(
        MarkerToken(3),
        OrderedUpdate {
            change: Some(VmChange {
                space: SPACE,
                vm,
                update: VmUpdate::Unmap { range },
            }),
            ticket: None,
        },
    );
    // The vertex buffer (b0) is used by list 1; the unmap drops it after that list.
    rig.run(&(draw_state() + &draw() + "wait-for-idle channel=3\nmarker token=0x3\n"));
    // The marker waits on list 1, which the recording backend has finished.
    rig.exec.complete();
    let snapshot = rig.recording.as_ref().unwrap().snapshot();
    assert!(
        snapshot.contains("destroy b0 after fence 1: now"),
        "{snapshot}"
    );
}

// ---------------------------------------------------------------------------------------------
// Review findings: deferred draws at every boundary, failed deferred draws, removed channels,
// the inline QMD, the drain before a copy, write-backs and edges.

/// A general draw (`BEGIN` `FIRST` ... `END`), then an instanced one (`BEGIN` `SUBSEQUENT` with
/// topology `op` ... `END`) whose two instances wait for a flush.
fn instanced(op: u32) -> String {
    let c = THREE_D;
    write(c, &clc797::SET_VERTEX_ARRAY_START, &[], 0)
        + &write(c, &clc797::DRAW_VERTEX_ARRAY, &[], 3)
        + &write(c, &clc797::BEGIN, &[], clc797::BEGIN_OP_TRIANGLES)
        + &write(c, &clc797::END, &[], 0)
        + &write(
            c,
            &clc797::BEGIN,
            &[],
            fields(&[
                (clc797::BEGIN_OP, op),
                (
                    clc797::BEGIN_INSTANCE_ID,
                    clc797::BEGIN_INSTANCE_ID_SUBSEQUENT,
                ),
            ]),
        )
        + &write(c, &clc797::END, &[], 0)
}

/// A `BEGIN_OP` no topology has: the draw path refuses it only when the instances are drawn.
const UNDEFINED_OP: u32 = clc797::BEGIN_OP_PATCH + 1;

/// The DMA copy of [`dma_copy_with_semaphore`] split into its setup and its `LAUNCH_DMA` word.
fn dma_setup_and_launch(payload: u32) -> (String, String) {
    let text = dma_copy_with_semaphore(payload);
    let (setup, launch) = text.trim_end().rsplit_once('\n').unwrap();
    (format!("{setup}\n"), format!("{launch}\n"))
}

/// Whether the last `Draw` of `commands` draws `instances` instances.
fn last_draw_has_instances(commands: &[Command], instances: u32) -> bool {
    commands.iter().rev().find_map(|c| match c {
        Command::Draw(desc) => Some(desc.params),
        _ => None,
    }) == Some(starquake_render::command::DrawParams::Direct {
        vertex_count: 3,
        instance_count: instances,
        first_vertex: 0,
        first_instance: 0,
    })
}

#[test]
fn deferred_instances_are_drawn_before_each_boundary() {
    // A marker and a syncpoint flush every channel's instances; a word to the channel's DMA or
    // inline-to-memory class flushes that channel's.
    let (setup, launch) = dma_setup_and_launch(1);
    let inline_word = write(cla140::CLASS, &cla140::LINE_COUNT, &[], 1);
    let boundaries: [(&str, String, String); 4] = [
        ("marker", String::new(), "marker token=0x7\n".to_owned()),
        ("syncpoint", String::new(), "syncpoint id=5\n".to_owned()),
        ("dma", setup, launch),
        ("inline", String::new(), inline_word),
    ];
    for (name, before, boundary) in boundaries {
        let mut rig = Rig::new();
        rig.write_words(SRC, &[1, 2, 3, 4]);
        rig.run(&(draw_state() + &before + &instanced(clc797::BEGIN_OP_TRIANGLES)));
        assert!(rig.exec.draws(CHANNEL).unwrap().has_deferred(), "{name}");
        rig.run(&boundary);
        assert!(!rig.exec.draws(CHANNEL).unwrap().has_deferred(), "{name}");
        rig.submit();
        let commands = rig.commands();
        assert_eq!(draws(&commands), 2, "{name}: {commands:?}");
        assert!(
            last_draw_has_instances(&commands, 2),
            "{name}: {commands:?}"
        );
    }
}

#[test]
fn a_failed_deferred_draw_does_not_stop_the_entry_after_it() {
    let refused = || {
        ExecError::Draw(DrawError::Topology {
            value: UNDEFINED_OP,
        })
    };
    let release =
        format!("release channel=3 address={SEM:#x} payload=0x1 size=4 timestamp=no op=write\n");
    let (setup, launch) = dma_setup_and_launch(0x5);
    let entries: [(&str, String, String); 5] = [
        ("syncpoint", String::new(), "syncpoint id=5\n".to_owned()),
        ("marker", String::new(), "marker token=0x7\n".to_owned()),
        ("release", String::new(), release),
        (
            "bind",
            String::new(),
            "bind channel=3 subchannel=4 class=0xc7b5\n".to_owned(),
        ),
        ("dma", setup, launch),
    ];
    for (name, before, entry) in entries {
        let mut rig = Rig::new();
        rig.write_words(SRC, &[1, 2, 3, 4]);
        rig.run(&(draw_state() + &before + &instanced(UNDEFINED_OP)));
        assert!(rig.exec.draws(CHANNEL).unwrap().has_deferred(), "{name}");
        let fenced = rig.exec.fences().counters().fenced;
        let parsed = parse_text(&entry).unwrap();
        assert_eq!(rig.exec.apply(&parsed[0]), Err(refused()), "{name}");
        assert!(!rig.exec.draws(CHANNEL).unwrap().has_deferred(), "{name}");
        let attached = rig.exec.fences().counters().fenced - fenced;
        match name {
            "syncpoint" | "marker" => assert_eq!(attached, 1, "{name}"),
            "release" => {
                assert_eq!(attached, 1);
                rig.submit();
                rig.exec.complete();
                assert_eq!(rig.word(SEM), 1);
            }
            "bind" => {
                assert_eq!(attached, 0);
                assert_eq!(
                    rig.exec.bound(CHANNEL, Subchannel::new(4).unwrap()),
                    Some(ClassId(clc7b5::CLASS))
                );
            }
            _ => {
                // The copy ran and its semaphore was attached.
                assert_eq!(attached, 1);
                assert_eq!(
                    (0..4).map(|i| rig.word(DST + i * 4)).collect::<Vec<_>>(),
                    vec![1, 2, 3, 4]
                );
            }
        }
    }
}

#[test]
fn a_failed_deferred_draw_and_a_failed_entry_return_the_entrys_error_and_count_both() {
    let mut rig = Rig::new();
    rig.run(&(draw_state() + &instanced(UNDEFINED_OP)));
    let failed = rig.exec.counters().failed;
    // The entry's own work fails too: a present with no present space.
    let entry = parse_text(&present()).unwrap();
    let ((), events) = captured(|| {
        assert_eq!(rig.exec.apply(&entry[0]), Err(ExecError::NoPresentSpace));
    });
    assert!(!rig.exec.draws(CHANNEL).unwrap().has_deferred());
    // One for the entry, one for the deferred draw logged beside it.
    assert_eq!(rig.exec.counters().failed - failed, 2);
    assert!(
        at(&events, tracing::Level::DEBUG).iter().any(|e| e
            .contains("could not draw a channel's deferred instances")
            && e.contains("topology")),
        "{events:?}"
    );
}

#[test]
fn a_removed_channels_draw_path_is_destroyed_after_the_list_that_used_it() {
    let mut rig = Rig::new();
    rig.run(&(draw_state() + &instanced(clc797::BEGIN_OP_TRIANGLES)));
    assert_eq!(rig.exec.remove_channel(CHANNEL), Ok(true));
    assert_eq!(rig.exec.fences().counters().closed.channel_removed, 1);
    // Nothing is left to close: the removal closed the list holding the draws.
    assert_eq!(rig.exec.close(CloseReason::RingEmpty), Ok(None));
    let lists = rig.recording.as_ref().unwrap().lists();
    assert_eq!(lists.len(), 1);
    let (fence, list) = &lists[0];
    // The deferred instances were drawn into it, not lost.
    assert!(last_draw_has_instances(&list.commands, 2), "{list:?}");
    let snapshot = rig.recording.as_ref().unwrap().snapshot();
    let destroyed: Vec<u64> = snapshot
        .lines()
        .filter(|l| l.starts_with("destroy "))
        .map(|l| {
            let after = l.split("after fence ").nth(1).unwrap();
            after.split(':').next().unwrap().parse().unwrap()
        })
        .collect();
    // The draw path's two host shaders, at least.
    assert!(destroyed.len() >= 2, "{snapshot}");
    assert!(
        destroyed.iter().all(|&after| after >= fence.get()),
        "{snapshot}"
    );
}

#[test]
fn removing_a_channel_with_no_3d_engine_closes_nothing() {
    let mut rig = Rig::new();
    rig.run(&(draw_state() + &draw()));
    assert_eq!(rig.exec.remove_channel(OTHER), Ok(true));
    assert_eq!(rig.exec.fences().counters().closed.channel_removed, 0);
    assert_eq!(rig.exec.fences().open_commands(), 4);
}

/// Binds the compute class and writes `words` as the inline QMD at `address`, every word but
/// the last; returns the ring text of the last word, which writes it.
fn inline_qmd(rig: &mut Rig, address: u64, words: &[u32; QMD_WORDS]) -> String {
    let c = COMPUTE;
    let mut text = String::from("bind channel=3 subchannel=1 class=0xc7c0\n");
    text += &address_text(
        c,
        &clc7c0::SET_INLINE_QMD_ADDRESS_A,
        &clc7c0::SET_INLINE_QMD_ADDRESS_B,
        address >> crate::back::qmd::QMD_ADDRESS_SHIFT,
    );
    let (last, rest) = words.split_last().unwrap();
    for (index, word) in (0u32..).zip(rest) {
        text += &write(c, &clc7c0::LOAD_INLINE_QMD_DATA, &[index], *word);
    }
    rig.run(&text);
    let index = u32::try_from(QMD_WORDS - 1).unwrap();
    write(c, &clc7c0::LOAD_INLINE_QMD_DATA, &[index], *last)
}

/// [`address`] for a class with no indices.
fn address_text(class: u32, upper: &Method, lower: &Method, value: u64) -> String {
    address(class, upper, lower, &[], value)
}

#[test]
fn a_refused_inline_qmd_is_still_invalidated_in_the_caches() {
    // Every word zero: version 0, and no scheduling action needed, so the scheduler loads the
    // QMD it has just written and refuses its version, with no sink callback.
    let mut rig = Rig::new();
    let last = parse_text(&inline_qmd(&mut rig, INLINE_QMD, &[0; QMD_WORDS])).unwrap();
    assert_eq!(
        rig.exec.apply(&last[0]),
        Err(ExecError::Qmd(QmdError::Version {
            address: INLINE_QMD,
            major: 0,
            minor: 0,
        }))
    );
    assert_eq!(
        rig.log().invalidated,
        vec![PageRange::new(INLINE_QMD - INLINE_QMD % PAGE_SIZE, PAGE_SIZE).unwrap()]
    );
}

/// Arms every page the caches asked the write watch to watch, as the kernel would: a page that
/// is watched and not written stays clean, so the buffer cache does not upload it again.
fn arm_requests(watch: &WriteWatch) {
    let mut requests = Vec::new();
    watch.take_requests(&mut requests);
    watch.arm(&requests).unwrap();
}

/// How many uploads to a buffer `snapshot` holds.
fn buffer_uploads(snapshot: &str) -> usize {
    snapshot
        .lines()
        .filter(|l| l.trim_start().starts_with("upload ") && l.contains(" to b"))
        .count()
}

#[test]
fn the_inline_qmd_is_dropped_from_the_buffer_cache() {
    // The inline QMD written over the vertex buffer a draw uploaded: the next draw uploads it
    // again. With no inline QMD in between, it does not. The first draw uploads the buffer, and
    // the first after the watch is armed reads it once more (`back::buffer`, "The first use
    // after arming reads the pages once more"): two uploads before the case begins.
    for (write_qmd, uploads) in [(false, 2), (true, 3)] {
        let mut rig = Rig::new();
        rig.run(&(draw_state() + &draw()));
        arm_requests(&rig.watch);
        rig.run(&draw());
        if write_qmd {
            let mut words = [0u32; QMD_WORDS];
            // Needs a scheduling action: written, not scheduled.
            words[4] = 1 << 12;
            let last = inline_qmd(&mut rig, VERTICES, &words);
            rig.run(&last);
        }
        rig.run(&draw());
        let snapshot = rig.submit();
        assert_eq!(
            buffer_uploads(&snapshot),
            uploads,
            "{write_qmd}: {snapshot}"
        );
    }
}

#[test]
fn starts_draw_is_exactly_the_draw_path_tables() {
    for method in 0..=u32::from(u16::MAX) {
        let listed = DRAW_METHODS
            .iter()
            .chain(UNHANDLED_METHODS.iter())
            .chain(CLEAR_METHODS.iter())
            .any(|m| m.method() == method);
        assert_eq!(starts_draw(method), listed, "{method:#x}");
    }
    assert!(!starts_draw(u32::MAX));
    // Each table's first and last entry, written out (`draw::call`'s tables).
    for method in [
        &clc797::BEGIN,
        &clc797::DRAW_VERTEX_ARRAY_BEGIN_END_B,
        &clc797::DRAW_INLINE_INDEX,
        &clc797::DRAW_AUTO,
        &clc797::CLEAR_SURFACE,
    ] {
        assert!(starts_draw(method.method()), "{}", method.name);
    }
    assert!(!starts_draw(clc797::SET_COLOR_TARGET_WIDTH.method()));
}

#[test]
fn a_copy_launch_drains_cpu_writes_first() {
    let page = |va: u64| PageRange::new(va - va % PAGE_SIZE, PAGE_SIZE).unwrap();
    // A DMA launch: the drain's range, then the copy's destination.
    let mut rig = Rig::new();
    let (setup, launch) = dma_setup_and_launch(1);
    rig.run(&setup);
    cpu_write(&rig.watch, UNIFORMS);
    assert_eq!(rig.log().invalidated, vec![]);
    rig.run(&launch);
    assert_eq!(rig.log().invalidated, vec![page(UNIFORMS), page(DST)]);
    // A 2D launch drains before the blit engine runs, even when it refuses the blit.
    let mut rig = Rig::new();
    let c = cl902d::CLASS;
    rig.run(
        &("bind channel=3 subchannel=3 class=0x902d\n".to_owned()
            + &write(
                c,
                &cl902d::SET_CLIP_ENABLE,
                &[],
                fields(&[(cl902d::SET_CLIP_ENABLE_V, 1)]),
            )
            + &write(
                c,
                &cl902d::SET_OPERATION,
                &[],
                cl902d::SET_OPERATION_V_SRCCOPY,
            )),
    );
    cpu_write(&rig.watch, UNIFORMS);
    let launch = parse_text(&write(c, &cl902d::PIXELS_FROM_MEMORY_SRC_Y0_INT, &[], 0)).unwrap();
    assert_eq!(
        rig.exec.apply(&launch[0]),
        Err(ExecError::Copy(CopyError::BlitClip))
    );
    assert_eq!(rig.log().invalidated, vec![page(UNIFORMS)]);
}

/// The byte a [`Painter`] answers each unknown readback byte with.
const PAINT: u8 = 0x5c;

/// The recording backend, answering a readback whose bytes it cannot know with [`PAINT`] bytes
/// of the [`SIZE`]-square RGBA8 target's level, as a host GPU would answer with what it drew.
struct Painter(RecordingBackend);

impl Backend for Painter {
    fn caps(&self) -> &Caps {
        self.0.caps()
    }
    fn budget(&self) -> starquake_render::MemoryBudget {
        self.0.budget()
    }
    fn create_buffer(
        &self,
        desc: &starquake_render::BufferDesc,
    ) -> Result<starquake_render::BufferId, starquake_render::RenderError> {
        self.0.create_buffer(desc)
    }
    fn create_texture(
        &self,
        desc: &starquake_render::TextureDesc,
    ) -> Result<starquake_render::TextureId, starquake_render::RenderError> {
        self.0.create_texture(desc)
    }
    fn create_view(
        &self,
        texture: starquake_render::TextureId,
        desc: &starquake_render::ViewDesc,
    ) -> Result<starquake_render::ViewId, starquake_render::RenderError> {
        self.0.create_view(texture, desc)
    }
    fn create_sampler(
        &self,
        desc: &starquake_render::SamplerDesc,
    ) -> Result<starquake_render::SamplerId, starquake_render::RenderError> {
        self.0.create_sampler(desc)
    }
    fn create_shader(
        &self,
        desc: starquake_render::ShaderDesc,
    ) -> Result<starquake_render::ShaderId, starquake_render::RenderError> {
        self.0.create_shader(desc)
    }
    fn destroy(&self, resource: starquake_render::ResourceId, after: FenceValue) {
        self.0.destroy(resource, after);
    }
    fn create_graphics_pipeline(
        &self,
        desc: starquake_render::GraphicsPipelineDesc,
    ) -> Result<starquake_render::PipelineId, starquake_render::RenderError> {
        self.0.create_graphics_pipeline(desc)
    }
    fn create_compute_pipeline(
        &self,
        desc: starquake_render::ComputePipelineDesc,
    ) -> Result<starquake_render::PipelineId, starquake_render::RenderError> {
        self.0.create_compute_pipeline(desc)
    }
    fn pipeline_status(
        &self,
        id: starquake_render::PipelineId,
    ) -> starquake_render::PipelineStatus {
        self.0.pipeline_status(id)
    }
    fn staging(
        &self,
        size: u64,
    ) -> Result<starquake_render::StagingSlice, starquake_render::RenderError> {
        self.0.staging(size)
    }
    fn submit(
        &self,
        list: starquake_render::CommandList,
    ) -> Result<FenceValue, starquake_render::RenderError> {
        self.0.submit(list)
    }
    fn completed(&self) -> FenceValue {
        self.0.completed()
    }
    fn wait(
        &self,
        value: FenceValue,
        timeout: Duration,
    ) -> Result<starquake_render::WaitOutcome, starquake_render::RenderError> {
        self.0.wait(value, timeout)
    }
    fn take_readback(
        &self,
        id: starquake_render::ReadbackId,
    ) -> Option<starquake_render::Readback> {
        match self.0.take_readback(id)? {
            starquake_render::Readback::Unknown => {
                let pixel = starquake_render::PixelFormat::A8B8G8R8Unorm.bytes_per_block();
                let len = usize::try_from(SIZE * SIZE * pixel).unwrap();
                Some(starquake_render::Readback::Bytes(vec![PAINT; len]))
            }
            known @ starquake_render::Readback::Bytes(_) => Some(known),
        }
    }
    fn set_present_target(
        &self,
        target: starquake_render::PresentTarget,
    ) -> Result<(), starquake_render::RenderError> {
        self.0.set_present_target(target)
    }
    fn take_present_feedback(&self) -> Vec<starquake_render::PresentFeedback> {
        self.0.take_present_feedback()
    }
}

#[test]
fn a_write_back_writes_the_drawn_bytes_before_the_copy_reads_them() {
    let backend = Arc::new(Painter(RecordingBackend::new(caps(), 1 << 40)));
    let mut rig = Rig::on(backend, compute_set(vec![]));
    let (_, launch) = dma_setup_and_launch(0);
    let c = clc7b5::CLASS;
    let mut copy = String::from("bind channel=3 subchannel=4 class=0xc7b5\n");
    copy += &address_text(
        c,
        &clc7b5::OFFSET_IN_UPPER,
        &clc7b5::OFFSET_IN_LOWER,
        TARGET,
    );
    copy += &address_text(c, &clc7b5::OFFSET_OUT_UPPER, &clc7b5::OFFSET_OUT_LOWER, DST);
    copy += &write(c, &clc7b5::LINE_LENGTH_IN, &[], 16);
    copy += &write(c, &clc7b5::LINE_COUNT, &[], 1);
    // The launch of `dma_copy_with_semaphore` also releases a semaphore at SEM; keep it.
    copy += &address_text(c, &clc7b5::SET_SEMAPHORE_A, &clc7b5::SET_SEMAPHORE_B, SEM2);
    copy += &launch;
    // A release after the draw, in the list the write-back closes.
    let release =
        format!("release channel=3 address={SEM:#x} payload=0x31 size=4 timestamp=no op=write\n");
    rig.run(&(draw_state() + &draw() + &release));
    assert_eq!(rig.word(SEM), 0);
    let ((), events) = captured(|| rig.run(&copy));
    let counters = rig.exec.counters();
    assert_eq!(
        (counters.write_backs, counters.write_back_failures),
        (1, 0),
        "{events:?}"
    );
    // The copy read the bytes the write-back wrote.
    let painted = u32::from_le_bytes([PAINT; 4]);
    assert_eq!(
        (0..4).map(|i| rig.word(DST + i * 4)).collect::<Vec<_>>(),
        vec![painted; 4]
    );
    // The write-back ran completion after its wait: the release in its list is performed.
    assert_eq!(rig.word(SEM), 0x31);
    assert_eq!(rig.exec.fences().lists_in_flight(), 0);
}

#[test]
fn a_compute_constant_buffer_is_bound_at_most_at_the_qmds_size() {
    let bound = COMPUTE_BANK_SIZE;
    // Larger than the QMD binds: capped. Zero: the whole binding. Smaller: as declared.
    for (declared, expected) in [
        (bound * 2, u64::from(bound)),
        (bound + 4, u64::from(bound)),
        (0, u64::from(bound)),
        (bound - 4, u64::from(bound - 4)),
    ] {
        let backend = Arc::new(RecordingBackend::new(caps(), 1 << 40));
        let mut rig = Rig::on(
            Arc::<RecordingBackend>::clone(&backend),
            compute_set(vec![uniform_bank(0, declared)]),
        );
        rig.recording = Some(backend);
        rig.write_words(QMD, &compute_qmd());
        rig.run(&compute_signal());
        assert_eq!(rig.exec.counters().dispatches, 1, "{declared:#x}");
        rig.submit();
        let sizes: Vec<u64> = rig
            .commands()
            .iter()
            .filter_map(|c| match c {
                Command::Dispatch(desc) => Some(desc),
                _ => None,
            })
            .flat_map(|desc| desc.bindings.iter())
            .filter_map(|b| match b.resource {
                starquake_render::command::BindingResource::Buffer(slice) => Some(slice.size),
                _ => None,
            })
            .collect();
        assert_eq!(sizes, vec![expected], "{declared:#x}");
    }
}

#[test]
fn compute_answers_read_words_up_to_the_exact_end_of_the_binding() {
    let rig = Rig::new();
    let size = 0x100;
    rig.write_words(UNIFORMS + u64::from(size - 8), &[0x1111_2222, 0x3333_4444]);
    let bound = [ConstantBuffer {
        valid: true,
        address: UNIFORMS,
        size,
    }];
    let answers = compute::ComputeAnswers {
        memory: &*rig.mem,
        constant_buffers: &bound,
    };
    let word = |offset: u32| answers.cbuf_word(GuestStage::Compute, 0, offset);
    // The last whole word ends exactly at the size; one byte later it would pass it.
    assert_eq!(word(size - 4), Some(0x3333_4444));
    assert_eq!(word(size - 8), Some(0x1111_2222));
    assert_eq!(word(size - 3), None);
    assert_eq!(word(size), None);
    assert_eq!(word(u32::MAX - 2), None);
    // A bank the QMD does not bind, or binds invalid, answers nothing.
    assert_eq!(answers.cbuf_word(GuestStage::Compute, 1, 0), None);
    let invalid = [ConstantBuffer {
        valid: false,
        ..bound[0]
    }];
    let answers = compute::ComputeAnswers {
        memory: &*rig.mem,
        constant_buffers: &invalid,
    };
    assert_eq!(answers.cbuf_word(GuestStage::Compute, 0, size - 4), None);
}

#[test]
fn the_inline_qmd_range_is_refused_exactly_past_the_shift() {
    let mut registers =
        vec![0u32; usize::try_from(crate::back::qmd::LAST_INLINE_QMD_METHOD).unwrap() + 1];
    let set = |registers: &mut Vec<u32>, upper: u32, lower: u32| {
        let a = usize::try_from(clc7c0::SET_INLINE_QMD_ADDRESS_A.method()).unwrap();
        let b = usize::try_from(clc7c0::SET_INLINE_QMD_ADDRESS_B.method()).unwrap();
        registers[a] = upper;
        registers[b] = lower;
    };
    let qmd_bytes = u64::try_from(QMD_BYTES).unwrap();
    // The highest bit the shift keeps: leading zeros exactly QMD_ADDRESS_SHIFT.
    let upper = 1 << (u32::BITS - QMD_ADDRESS_SHIFT - 1);
    set(&mut registers, upper, 0);
    let start = 1u64 << (u64::BITS - 1);
    assert_eq!(
        (u64::from(upper) << u32::BITS).leading_zeros(),
        QMD_ADDRESS_SHIFT
    );
    assert_eq!(inline_qmd_range(&registers), Some(start..start + qmd_bytes));
    // One bit higher: leading zeros QMD_ADDRESS_SHIFT - 1, which the shift would drop.
    set(&mut registers, upper << 1, 0);
    assert_eq!(
        (u64::from(upper << 1) << u32::BITS).leading_zeros(),
        QMD_ADDRESS_SHIFT - 1
    );
    assert_eq!(inline_qmd_range(&registers), None);
    // The last QMD of the address space: its end would pass 2^64.
    set(&mut registers, u32::MAX >> QMD_ADDRESS_SHIFT, u32::MAX);
    assert_eq!(inline_qmd_range(&registers), None);
    // The lower word counts: the same upper word with a lower word one less fits.
    set(&mut registers, u32::MAX >> QMD_ADDRESS_SHIFT, u32::MAX - 1);
    let start = u64::MAX - (2 * qmd_bytes - 1);
    assert_eq!(inline_qmd_range(&registers), Some(start..start + qmd_bytes));
}

#[test]
fn a_present_with_no_space_and_a_bad_format_is_no_present_space() {
    // Both faults at once: the missing present space is reported first.
    let mut rig = Rig::new();
    let bad = present().replace("format=1", "format=2");
    let entry = parse_text(&bad).unwrap();
    assert_eq!(rig.exec.apply(&entry[0]), Err(ExecError::NoPresentSpace));
}

// Clears through the executor (slice G24).
mod clear;
