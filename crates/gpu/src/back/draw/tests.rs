//! Tests of the draw path: register scripts written to a real 3D mirror, run through
//! [`Draws`] with the real buffer, texture and pipeline-key caches over the recording backend,
//! and a fake shader cache that returns fixed sets, pending tickets and rejections. The golden
//! draw descriptions are the recording backend's text of the submitted list (`insta`).

use std::collections::VecDeque;
use std::sync::Arc;

use starquake_memory::{FlatMemory, GuestMemory, PAGE_SIZE, PageRange, WriteWatch};
use starquake_render::caps::{DynamicState, StageSupport};
use starquake_render::command::{BindingResource, DrawParams, IndexFormat};
use starquake_render::pipeline::Topology;
use starquake_render::resource::ViewDimension;
use starquake_render::resource::{
    AddressMode, BorderColor, CompareFunction, FilterMode, ReductionMode,
};
use starquake_render::shader::{ResourceKind, SampleKind};
use starquake_render::{
    Backend, Caps, Command, CommandList, EmulationFlags, Features, FormatFeatures, FormatTable,
    Limits, PixelFormat, RecordingBackend, SampleCounts,
};
use starquake_shader::asm::assemble_program;
use starquake_shader::decode::{Instr, Op, OpExit, PredSrc};
use starquake_shader::exchange::{
    Answers, Question, Questioner, SampledType, TextureFormat, TextureKind, TextureUsage, exchange,
};
use starquake_shader::host::{
    GuestSlot, HostBuiltIn, HostShader, InterfaceSlot, SlotBinding, SlotType,
};
use starquake_shader::lookup::{
    EngineState, GuestStage, ProgramMemory, ProgramRequest, Rejection, ShaderSet, Ticket, Unmapped,
    VariantKey,
};
use starquake_shader::program::translate_program;
use starquake_shader::translate::{ColorOutputs, Stage, TranslateError, Unsupported};
use starquake_shader::variant::{
    AttributeMask, AttributeType, ColorType, CompareFunction as KeyCompare, EmulationKey,
    FollowingStage, PreviousStores, Swizzle, SwizzleSource,
};

use super::call::{DrawCalls, Source, Started};
use super::fixed::{self, Part};
use super::root::{FIRST_ROOT_BANK, ROOT_TABLE_BYTES, RootTables};
use super::sampler::{self, SAMPLER_WORDS};
use super::shaders::{
    self, GuestPrograms, HandleSource, ReadySet, ResourceSource, ResourceUse, SetLookup,
    ShaderLookup, Slot, StageInfo, StateAnswers,
};
use super::targets;
use super::*;
use crate::back::buffer::BufferCache;
use crate::back::pipeline::{PipelineCache, ShaderStamp};
use crate::back::state::{GroupKind, Mirror};
use crate::back::texture::{HEADER_WORDS, TextureCache};
use crate::classes::{Method, MwField, clc797, clc797tex as tex};
use crate::field::Field;
use crate::interface::ChannelId;
use crate::memory::{GpuFault, TestGpuMemory};
use crate::ring::ClassId;
use crate::test_log::{at, captured};

// ---------------------------------------------------------------------------------------------
// The rig.

/// Where test memory starts, at the same GPU and CPU address (row 30).
const BASE: u64 = 0x8000_0000;
/// Pages of test memory.
const PAGES: u64 = 64;
/// Not channel 0, so a hard-coded channel fails.
const CHANNEL: ChannelId = ChannelId(3);

/// Addresses of what the scripts place in memory.
const VERTEX_PROGRAM: u64 = BASE + 0x1000;
const PIXEL_PROGRAM: u64 = BASE + 0x2000;
const VERTICES: u64 = BASE + 0x3000;
const VERTICES_SIZE: u64 = 0x100;
const INDICES: u64 = BASE + 0x4000;
const INDICES_SIZE: u64 = 0x40;
const UNIFORMS: u64 = BASE + 0x5000;
const UNIFORMS_SIZE: u32 = 0x100;
const HEADERS: u64 = BASE + 0x6000;
const SAMPLERS: u64 = BASE + 0x7000;
const TEXTURE: u64 = BASE + 0x8000;
const TARGET: u64 = BASE + 0x10000;
const REPORT: u64 = BASE + 0x9000;
/// The test texture's and target's size.
const SIZE: u32 = 8;

/// The bank the bound texture handles are read from, and the handle's offset in it.
const HANDLE_BANK: u8 = 2;
const HANDLE_OFFSET: u32 = 0x10;
/// The texture header and sampler indices the handle names (distinct, so a swap fails).
const HEADER_INDEX: u32 = 3;
const SAMPLER_INDEX: u32 = 5;

type Memory = TestGpuMemory<FlatMemory>;

fn caps(view_swizzle: bool, features: Features, max_vertex_attributes: u32) -> Caps {
    Caps {
        limits: Limits {
            max_buffer_size: PAGES * PAGE_SIZE,
            max_staging_size: PAGES * PAGE_SIZE,
            max_texture_dimension_1d: 4096,
            max_texture_dimension_2d: 4096,
            max_texture_dimension_3d: 256,
            max_texture_array_layers: 64,
            max_sampler_anisotropy: 16,
            max_vertex_attributes,
            max_vertex_buffers: 8,
            max_vertex_buffer_stride: 2048,
            max_color_attachments: 8,
            max_viewports: 16,
            max_bind_groups: 4,
            max_bindings_per_bind_group: 16,
            max_compute_workgroups_per_dimension: 65_535,
            max_uniform_buffers_per_stage: 12,
            max_storage_buffers_per_stage: 8,
            max_sampled_textures_per_stage: 16,
            max_storage_textures_per_stage: 4,
            max_samplers_per_stage: 16,
        },
        features,
        stages: StageSupport::ALL,
        dynamic_state: DynamicState::ALL,
        sample_counts: SampleCounts::X1,
        view_swizzle,
        formats: FormatTable::uniform(FormatFeatures::ALL),
    }
}

fn memory() -> Memory {
    let size = usize::try_from(PAGES * PAGE_SIZE).unwrap();
    let mem = TestGpuMemory::new(FlatMemory::new(BASE, size));
    mem.map(PageRange::new(BASE, PAGES * PAGE_SIZE).unwrap(), BASE)
        .unwrap();
    mem
}

/// The exit-only program, translated for `stage`: a real host shader for the fake cache's sets.
pub(crate) fn host_shader(stage: GuestStage) -> HostShader {
    let code = assemble_program(&[Instr::new(Op::Exit(OpExit {
        keep_ref_count: false,
        no_at_exit: false,
        pred: PredSrc::TRUE,
    }))])
    .unwrap();
    translate_program(
        Stage {
            stage,
            color_outputs: ColorOutputs {
                targets: [0; starquake_shader::translate::COLOR_TARGETS],
            },
        },
        &code,
    )
    .unwrap()
}

/// Stamps of the two shaders (distinct, non-zero).
const VERTEX_STAMP: ShaderStamp = ShaderStamp::new(0x11);
const PIXEL_STAMP: ShaderStamp = ShaderStamp::new(0x22);

/// A ready vertex and pixel set: the vertex shader binds `vertex` and the pixel shader `pixel`.
fn ready(vertex: Vec<ResourceUse>, pixel: Vec<ResourceUse>) -> SetLookup {
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
                resources: vertex,
            },
            StageInfo {
                stamp: PIXEL_STAMP,
                resources: pixel,
            },
        ],
    })
}

fn uniform(group: u32, binding: u32, bank: u8, size: u32) -> ResourceUse {
    ResourceUse {
        slot: Slot { group, binding },
        kind: ResourceKind::UniformBuffer,
        count: None,
        source: ResourceSource::ConstantBuffer { bank, size },
    }
}

fn texture_use(group: u32, binding: u32) -> ResourceUse {
    ResourceUse {
        slot: Slot { group, binding },
        kind: ResourceKind::SampledTexture {
            dimension: ViewDimension::D2,
            sample: SampleKind::Float { filterable: true },
            multisampled: false,
        },
        count: None,
        source: ResourceSource::Texture(HandleSource {
            bank: HANDLE_BANK,
            offset: HANDLE_OFFSET,
        }),
    }
}

fn sampler_use(group: u32, binding: u32) -> ResourceUse {
    ResourceUse {
        slot: Slot { group, binding },
        kind: ResourceKind::Sampler { comparison: false },
        count: None,
        source: ResourceSource::Sampler(HandleSource {
            bank: HANDLE_BANK,
            offset: HANDLE_OFFSET,
        }),
    }
}

fn driver_use(group: u32, binding: u32) -> ResourceUse {
    ResourceUse {
        slot: Slot { group, binding },
        kind: ResourceKind::UniformBuffer,
        count: None,
        source: ResourceSource::DriverConstants,
    }
}

/// What the fake saw in one lookup.
#[derive(Clone, Debug)]
struct Seen {
    requests: Vec<ProgramRequest>,
    key: VariantKey,
    /// The first four bytes at the first request's address, read through the memory given.
    program_word: Result<[u8; 4], Unmapped>,
    /// The answer to a constant buffer question the fake asks: the handle word for the pixel
    /// stage.
    handle_word: Option<u32>,
}

/// The fake shader cache (module docs): scripted answers, then a default.
struct FakeShaders {
    answers: VecDeque<SetLookup>,
    default: SetLookup,
    waits: VecDeque<SetLookup>,
    seen: Vec<Seen>,
    waited: Vec<Ticket>,
}

impl FakeShaders {
    fn new(default: SetLookup) -> Self {
        Self {
            answers: VecDeque::new(),
            default,
            waits: VecDeque::new(),
            seen: Vec::new(),
            waited: Vec::new(),
        }
    }
}

impl ShaderLookup for FakeShaders {
    fn lookup(
        &mut self,
        requests: &[ProgramRequest],
        variant: &VariantKey,
        memory: &dyn ProgramMemory,
        state: &dyn EngineState,
    ) -> SetLookup {
        let mut word = [0u8; 4];
        let program_word = requests.first().map_or(Ok(word), |r| {
            memory.read(r.address, &mut word).map(|()| word)
        });
        self.seen.push(Seen {
            requests: requests.to_vec(),
            key: variant.clone(),
            program_word,
            handle_word: state.cbuf_word(GuestStage::Pixel, HANDLE_BANK, HANDLE_OFFSET),
        });
        self.answers
            .pop_front()
            .unwrap_or_else(|| self.default.clone())
    }

    fn wait(&mut self, ticket: Ticket) -> SetLookup {
        self.waited.push(ticket);
        self.waits
            .pop_front()
            .unwrap_or_else(|| self.default.clone())
    }
}

struct Rig {
    backend: Arc<RecordingBackend>,
    mem: Memory,
    buffers: BufferCache,
    textures: TextureCache,
    pipelines: PipelineCache,
    shaders: FakeShaders,
    draws: Draws,
    mirror: Mirror,
    list: CommandList,
}

impl Rig {
    fn new(answer: SetLookup) -> Self {
        Self::with(
            caps(true, Features::ALL, 16),
            WaitWhileCompiling::Wait,
            answer,
        )
    }

    fn with(caps: Caps, setting: WaitWhileCompiling, answer: SetLookup) -> Self {
        let backend = Arc::new(RecordingBackend::new(caps, 1 << 40));
        let dyn_backend: Arc<dyn Backend> = Arc::<RecordingBackend>::clone(&backend);
        let watch = Arc::new(WriteWatch::new());
        Self {
            buffers: BufferCache::new(Arc::clone(&dyn_backend), Arc::clone(&watch)),
            textures: TextureCache::new(Arc::clone(&dyn_backend), watch),
            pipelines: PipelineCache::new(Arc::clone(&dyn_backend)),
            draws: Draws::new(dyn_backend, setting),
            shaders: FakeShaders::new(answer),
            mem: memory(),
            mirror: Mirror::new(CHANNEL, ClassId(clc797::CLASS)),
            list: CommandList::new(),
            backend,
        }
    }

    /// Writes `value` to `method` at `indices`, as the executor would: into the mirror, then to
    /// the draw path.
    fn put(
        &mut self,
        method: &Method,
        indices: &[u32],
        value: u32,
    ) -> Result<Option<Outcome>, DrawError> {
        let number = method.method_at(indices).unwrap();
        self.mirror.write(number, value).unwrap();
        let mut ctx = DrawContext {
            memory: &self.mem,
            buffers: &mut self.buffers,
            textures: &mut self.textures,
            pipelines: &mut self.pipelines,
            shaders: &mut self.shaders,
            list: &mut self.list,
        };
        self.draws.method(&mut ctx, &mut self.mirror, number, value)
    }

    fn set(&mut self, method: &Method, indices: &[u32], value: u32) {
        assert_eq!(
            self.put(method, indices, value),
            Ok(None),
            "{}",
            method.name
        );
    }

    fn flush(&mut self) -> Result<Option<Outcome>, DrawError> {
        let mut ctx = DrawContext {
            memory: &self.mem,
            buffers: &mut self.buffers,
            textures: &mut self.textures,
            pipelines: &mut self.pipelines,
            shaders: &mut self.shaders,
            list: &mut self.list,
        };
        self.draws.flush(&mut ctx, &mut self.mirror)
    }

    fn write_words(&self, address: u64, words: &[u32]) {
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        self.mem.guest().write(address, &bytes).unwrap();
    }

    /// Binds constant buffer `bank` of bind group `group` at `address`, `size` bytes.
    fn bind(&mut self, group: u32, bank: u8, address: u64, size: u32) {
        self.set(&clc797::SET_CONSTANT_BUFFER_SELECTOR_A, &[], size);
        self.set(
            &clc797::SET_CONSTANT_BUFFER_SELECTOR_B,
            &[],
            u32::try_from(address >> 32).unwrap(),
        );
        self.set(
            &clc797::SET_CONSTANT_BUFFER_SELECTOR_C,
            &[],
            u32::try_from(address & 0xffff_ffff).unwrap(),
        );
        let bind = fields(&[
            (clc797::BIND_GROUP_CONSTANT_BUFFER_VALID, 1),
            (
                clc797::BIND_GROUP_CONSTANT_BUFFER_SHADER_SLOT,
                u32::from(bank),
            ),
        ]);
        self.set(&clc797::BIND_GROUP_CONSTANT_BUFFER, &[group], bind);
    }

    /// A 64-bit address into a pair of methods.
    fn address(&mut self, upper: &Method, lower: &Method, indices: &[u32], address: u64) {
        self.set(upper, indices, u32::try_from(address >> 32).unwrap());
        self.set(
            lower,
            indices,
            u32::try_from(address & 0xffff_ffff).unwrap(),
        );
    }

    /// The state every test draws with (module docs): one float attribute from stream 0, a
    /// vertex and a pixel program, one colour target, a viewport, the pools, a bound texture
    /// handle and a uniform buffer for the pixel stage.
    fn basic(&mut self) {
        self.set(
            &clc797::OGL_SET_FRONT_FACE,
            &[],
            clc797::OGL_SET_FRONT_FACE_V_CCW,
        );
        self.set(
            &clc797::SET_FRONT_POLYGON_MODE,
            &[],
            clc797::SET_FRONT_POLYGON_MODE_V_FILL,
        );
        self.set(
            &clc797::SET_RASTER_ENABLE,
            &[],
            clc797::SET_RASTER_ENABLE_V_TRUE,
        );
        let attribute = fields(&[
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
        ]);
        self.set(&clc797::SET_VERTEX_ATTRIBUTE_A, &[0], attribute);
        let format = fields(&[
            (clc797::SET_VERTEX_STREAM_A_FORMAT_STRIDE, 20),
            (clc797::SET_VERTEX_STREAM_A_FORMAT_ENABLE, 1),
        ]);
        self.set(&clc797::SET_VERTEX_STREAM_A_FORMAT, &[0], format);
        self.address(
            &clc797::SET_VERTEX_STREAM_A_LOCATION_A,
            &clc797::SET_VERTEX_STREAM_A_LOCATION_B,
            &[0],
            VERTICES,
        );
        self.address(
            &clc797::SET_VERTEX_STREAM_SIZE_A,
            &clc797::SET_VERTEX_STREAM_SIZE_B,
            &[0],
            VERTICES_SIZE,
        );
        self.address(
            &clc797::SET_PIPELINE_PROGRAM_ADDRESS_A,
            &clc797::SET_PIPELINE_PROGRAM_ADDRESS_B,
            &[1],
            VERTEX_PROGRAM,
        );
        self.set(
            &clc797::SET_PIPELINE_SHADER,
            &[5],
            fields(&[(clc797::SET_PIPELINE_SHADER_ENABLE, 1)]),
        );
        self.address(
            &clc797::SET_PIPELINE_PROGRAM_ADDRESS_A,
            &clc797::SET_PIPELINE_PROGRAM_ADDRESS_B,
            &[5],
            PIXEL_PROGRAM,
        );
        self.set(
            &clc797::SET_CT_SELECT,
            &[],
            fields(&[
                (clc797::SET_CT_SELECT_TARGET_COUNT, 1),
                (clc797::SET_CT_SELECT_TARGET0, 0),
            ]),
        );
        self.basic_target();
    }

    /// The colour target, surface clip and viewport of [`Rig::basic`].
    fn basic_target(&mut self) {
        self.address(
            &clc797::SET_COLOR_TARGET_A,
            &clc797::SET_COLOR_TARGET_B,
            &[0],
            TARGET,
        );
        self.set(&clc797::SET_COLOR_TARGET_WIDTH, &[0], SIZE);
        self.set(&clc797::SET_COLOR_TARGET_HEIGHT, &[0], SIZE);
        self.set(
            &clc797::SET_COLOR_TARGET_FORMAT,
            &[0],
            clc797::SET_COLOR_TARGET_FORMAT_V_A8B8G8R8,
        );
        self.set(&clc797::SET_COLOR_TARGET_THIRD_DIMENSION, &[0], 1);
        self.set(
            &clc797::SET_SURFACE_CLIP_HORIZONTAL,
            &[],
            fields(&[(clc797::SET_SURFACE_CLIP_HORIZONTAL_WIDTH, SIZE)]),
        );
        self.set(
            &clc797::SET_SURFACE_CLIP_VERTICAL,
            &[],
            fields(&[(clc797::SET_SURFACE_CLIP_VERTICAL_HEIGHT, SIZE)]),
        );
        // Viewport 0's scale and offset apply only while SET_VIEWPORT_SCALE_OFFSET is on; its
        // reset value is off (`without_scale_offset_viewports_and_scissors_are_the_surface_clip`).
        self.set(
            &clc797::SET_VIEWPORT_SCALE_OFFSET,
            &[],
            clc797::SET_VIEWPORT_SCALE_OFFSET_ENABLE_TRUE,
        );
        self.set(&clc797::SET_VIEWPORT_SCALE_X, &[0], 4.0f32.to_bits());
        self.set(&clc797::SET_VIEWPORT_SCALE_Y, &[0], 4.0f32.to_bits());
        self.set(&clc797::SET_VIEWPORT_OFFSET_X, &[0], 4.0f32.to_bits());
        self.set(&clc797::SET_VIEWPORT_OFFSET_Y, &[0], 4.0f32.to_bits());
        self.set(&clc797::SET_VIEWPORT_SCALE_Z, &[0], 0.5f32.to_bits());
        self.set(&clc797::SET_VIEWPORT_OFFSET_Z, &[0], 0.5f32.to_bits());
        self.basic_resources();
    }

    /// The pools, the bound texture handle and uniforms, the texture header and the sampler of
    /// [`Rig::basic`].
    fn basic_resources(&mut self) {
        self.address(
            &clc797::SET_TEX_HEADER_POOL_A,
            &clc797::SET_TEX_HEADER_POOL_B,
            &[],
            HEADERS,
        );
        self.set(&clc797::SET_TEX_HEADER_POOL_C, &[], 7);
        self.address(
            &clc797::SET_TEX_SAMPLER_POOL_A,
            &clc797::SET_TEX_SAMPLER_POOL_B,
            &[],
            SAMPLERS,
        );
        self.set(&clc797::SET_TEX_SAMPLER_POOL_C, &[], 7);
        // The pixel stage (bind group 4) binds the handle bank and a uniform bank.
        self.bind(4, HANDLE_BANK, UNIFORMS, UNIFORMS_SIZE);
        self.bind(4, 1, UNIFORMS, UNIFORMS_SIZE);
        self.write_words(
            UNIFORMS + u64::from(HANDLE_OFFSET),
            &[HEADER_INDEX | (SAMPLER_INDEX << shaders::SAMPLER_INDEX_SHIFT)],
        );
        self.write_words(VERTEX_PROGRAM, &[0x0bad_cafe]);
        let header = texture_header(TEXTURE, swizzle_sources(tex::TEXHEAD_BL_X_SOURCE_IN_R));
        self.write_words(
            HEADERS + u64::from(HEADER_INDEX) * shaders::HEADER_BYTES,
            &header,
        );
        self.write_words(
            SAMPLERS + u64::from(SAMPLER_INDEX) * sampler::SAMPLER_BYTES,
            &linear_sampler(),
        );
    }

    /// `BEGIN` with `op` and `INSTANCE_ID` `FIRST`, the vertex array, and `END`.
    fn draw_arrays(
        &mut self,
        op: u32,
        first: u32,
        count: u32,
    ) -> Result<Option<Outcome>, DrawError> {
        self.set(&clc797::SET_VERTEX_ARRAY_START, &[], first);
        self.set(&clc797::DRAW_VERTEX_ARRAY, &[], count);
        assert_eq!(self.put(&clc797::BEGIN, &[], op), Ok(None));
        self.put(&clc797::END, &[], 0)
    }

    fn draw(&mut self) -> Outcome {
        self.draw_arrays(clc797::BEGIN_OP_TRIANGLES, 0, 3)
            .unwrap()
            .unwrap()
    }

    /// Submits the open list and returns the recording backend's text of everything so far.
    fn submit(&mut self) -> String {
        let list = core::mem::take(&mut self.list);
        let fence = self.backend.submit(list).unwrap();
        self.buffers.submitted(fence);
        self.textures.submitted(fence);
        self.draws.submitted(fence);
        self.backend.snapshot()
    }

    fn draw_commands(&self) -> Vec<DrawDesc> {
        self.list
            .commands
            .iter()
            .filter_map(|c| match c {
                Command::Draw(d) => Some(d.clone()),
                _ => None,
            })
            .collect()
    }

    fn last_draw(&self) -> DrawDesc {
        self.draw_commands().pop().unwrap()
    }
}

/// A word with `fields` set.
fn fields(fields: &[(Field, u32)]) -> u32 {
    fields.iter().fold(0, |word, &(field, value)| {
        field.insert(word, value).unwrap()
    })
}

/// The four swizzle sources all from `first` up: R, G, B, A for `IN_R`.
fn swizzle_sources(first: u32) -> [u32; 4] {
    [first, first + 1, first + 2, first + 3]
}

/// A block-linear 2D A8B8G8R8 header of `SIZE` by `SIZE` at `address`, one level, with
/// `sources` for X to W.
fn texture_header(address: u64, sources: [u32; 4]) -> [u32; HEADER_WORDS] {
    let mut words = [0; HEADER_WORDS];
    let mut set = |field: MwField, value: u32| field.insert(&mut words, value).unwrap();
    set(
        tex::TEXHEAD_BL_HEADER_VERSION,
        tex::TEXHEAD_BL_HEADER_VERSION_SELECT_BLOCKLINEAR,
    );
    set(
        tex::TEXHEAD_BL_COMPONENTS,
        tex::TEXHEAD_BL_COMPONENTS_SIZES_A8B8G8R8,
    );
    for field in [
        tex::TEXHEAD_BL_R_DATA_TYPE,
        tex::TEXHEAD_BL_G_DATA_TYPE,
        tex::TEXHEAD_BL_B_DATA_TYPE,
        tex::TEXHEAD_BL_A_DATA_TYPE,
    ] {
        set(field, tex::TEXHEAD_BL_R_DATA_TYPE_NUM_UNORM);
    }
    for (field, source) in [
        tex::TEXHEAD_BL_X_SOURCE,
        tex::TEXHEAD_BL_Y_SOURCE,
        tex::TEXHEAD_BL_Z_SOURCE,
        tex::TEXHEAD_BL_W_SOURCE,
    ]
    .into_iter()
    .zip(sources)
    {
        set(field, source);
    }
    set(
        tex::TEXHEAD_BL_ADDRESS_BITS31TO9,
        u32::try_from((address >> 9) & 0x7f_ffff).unwrap(),
    );
    set(
        tex::TEXHEAD_BL_ADDRESS_BITS48TO32,
        u32::try_from(address >> 32).unwrap(),
    );
    set(tex::TEXHEAD_BL_WIDTH_MINUS_ONE, SIZE - 1);
    set(tex::TEXHEAD_BL_HEIGHT_MINUS_ONE, SIZE - 1);
    set(
        tex::TEXHEAD_BL_TEXTURE_TYPE,
        tex::TEXHEAD_BL_TEXTURE_TYPE_TWO_D,
    );
    words
}

/// A sampler entry: linear filters, point mip filter, clamp to edge, levels 0 to 1.
fn linear_sampler() -> [u32; SAMPLER_WORDS] {
    let w0 = fields(&[
        (Field::new(2, 0), tex::TEXSAMP0_ADDRESS_U_CLAMP_TO_EDGE),
        (Field::new(5, 3), tex::TEXSAMP0_ADDRESS_V_CLAMP_TO_EDGE),
        (Field::new(8, 6), tex::TEXSAMP0_ADDRESS_P_WRAP),
    ]);
    let w1 = fields(&[
        (
            tex::TEXSAMP1_MAG_FILTER,
            tex::TEXSAMP1_MAG_FILTER_MAG_LINEAR,
        ),
        (
            tex::TEXSAMP1_MIN_FILTER,
            tex::TEXSAMP1_MIN_FILTER_MIN_LINEAR,
        ),
        (tex::TEXSAMP1_MIP_FILTER, tex::TEXSAMP1_MIP_FILTER_MIP_POINT),
    ]);
    let w2 = fields(&[(tex::TEXSAMP2_MAX_LOD_CLAMP, 256)]);
    [w0, w1, w2, 0, 0, 0, 0, 1.0f32.to_bits()]
}

/// The pixel stage's resources of the golden draw: a uniform bank, the texture and sampler of
/// the handle, root table bank 24 and the driver constants.
fn pixel_resources() -> Vec<ResourceUse> {
    vec![
        uniform(0, 0, 1, UNIFORMS_SIZE),
        texture_use(0, 1),
        sampler_use(0, 2),
        uniform(
            0,
            3,
            FIRST_ROOT_BANK,
            u32::try_from(ROOT_TABLE_BYTES).unwrap(),
        ),
        driver_use(0, 4),
    ]
}

fn rejection() -> Arc<Rejection> {
    Arc::new(Rejection::Translate(TranslateError::Stage(
        Unsupported::Instruction("BRX"),
    )))
}

// ---------------------------------------------------------------------------------------------
// Golden draw descriptions.

#[test]
fn golden_non_indexed_draw() {
    let mut rig = Rig::new(ready(vec![], pixel_resources()));
    rig.basic();
    assert_eq!(rig.draw(), Outcome::Drawn);
    insta::assert_snapshot!(rig.submit());
}

#[test]
fn golden_small_index_draw_with_base_vertex_and_instance() {
    let mut rig = Rig::new(ready(vec![], vec![]));
    rig.basic();
    rig.address(
        &clc797::SET_INDEX_BUFFER_A,
        &clc797::SET_INDEX_BUFFER_B,
        &[],
        INDICES,
    );
    rig.address(
        &clc797::SET_INDEX_BUFFER_SIZE_A,
        &clc797::SET_INDEX_BUFFER_SIZE_B,
        &[],
        INDICES_SIZE,
    );
    // A negative base vertex (its two's complement) and a non-zero base instance.
    rig.set(
        &clc797::SET_GLOBAL_BASE_VERTEX_INDEX,
        &[],
        (-2i32).cast_unsigned(),
    );
    rig.set(&clc797::SET_GLOBAL_BASE_INSTANCE_INDEX, &[], 7);
    let word = fields(&[
        (
            clc797::DRAW_INDEX_BUFFER16_BEGIN_END_INSTANCE_FIRST_FIRST,
            2,
        ),
        (
            clc797::DRAW_INDEX_BUFFER16_BEGIN_END_INSTANCE_FIRST_COUNT,
            6,
        ),
        (
            clc797::DRAW_INDEX_BUFFER16_BEGIN_END_INSTANCE_FIRST_TOPOLOGY,
            clc797::BEGIN_OP_TRIANGLE_STRIP,
        ),
    ]);
    let outcome = rig.put(
        &clc797::DRAW_INDEX_BUFFER16_BEGIN_END_INSTANCE_FIRST,
        &[],
        word,
    );
    assert_eq!(outcome, Ok(Some(Outcome::Drawn)));
    let draw = rig.last_draw();
    assert_eq!(
        draw.params,
        DrawParams::Indexed {
            index_count: 6,
            instance_count: 1,
            first_index: 2,
            base_vertex: -2,
            first_instance: 7,
        }
    );
    assert_eq!(draw.index_buffer.unwrap().format, IndexFormat::U16);
    insta::assert_snapshot!(rig.submit());
}

// ---------------------------------------------------------------------------------------------
// Draw methods.

#[test]
fn begin_end_draws_the_vertex_array_with_its_registers() {
    let mut rig = Rig::new(ready(vec![], vec![]));
    rig.basic();
    rig.set(&clc797::SET_GLOBAL_BASE_INSTANCE_INDEX, &[], 9);
    assert_eq!(
        rig.draw_arrays(clc797::BEGIN_OP_TRIANGLES, 5, 12),
        Ok(Some(Outcome::Drawn))
    );
    assert_eq!(
        rig.last_draw().params,
        DrawParams::Direct {
            vertex_count: 12,
            instance_count: 1,
            first_vertex: 5,
            first_instance: 9,
        }
    );
}

#[test]
fn draw_index_buffer_makes_the_next_end_indexed_once() {
    let mut rig = Rig::new(ready(vec![], vec![]));
    rig.basic();
    rig.address(
        &clc797::SET_INDEX_BUFFER_A,
        &clc797::SET_INDEX_BUFFER_B,
        &[],
        INDICES,
    );
    rig.address(
        &clc797::SET_INDEX_BUFFER_SIZE_A,
        &clc797::SET_INDEX_BUFFER_SIZE_B,
        &[],
        INDICES_SIZE,
    );
    rig.set(
        &clc797::SET_INDEX_BUFFER_E,
        &[],
        fields(&[(
            clc797::SET_INDEX_BUFFER_E_INDEX_SIZE,
            clc797::SET_INDEX_BUFFER_E_INDEX_SIZE_ONE_BYTE,
        )]),
    );
    rig.set(&clc797::SET_INDEX_BUFFER_F, &[], 3);
    rig.set(&clc797::DRAW_INDEX_BUFFER, &[], 9);
    assert_eq!(
        rig.put(&clc797::BEGIN, &[], clc797::BEGIN_OP_TRIANGLES),
        Ok(None)
    );
    assert_eq!(rig.put(&clc797::END, &[], 0), Ok(Some(Outcome::Drawn)));
    let draw = rig.last_draw();
    assert!(matches!(
        draw.params,
        DrawParams::Indexed {
            index_count: 9,
            first_index: 3,
            ..
        }
    ));
    assert_eq!(draw.index_buffer.unwrap().format, IndexFormat::U8);
    // The next begin/end draws the vertex array again.
    assert_eq!(
        rig.put(&clc797::BEGIN, &[], clc797::BEGIN_OP_TRIANGLES),
        Ok(None)
    );
    assert_eq!(rig.put(&clc797::END, &[], 0), Ok(Some(Outcome::Drawn)));
    assert!(matches!(rig.last_draw().params, DrawParams::Direct { .. }));
}

#[test]
fn the_small_index_forms_take_their_size_from_the_method() {
    // SET_INDEX_BUFFER_E says one byte; each form's size wins (a listed divergence from Eden).
    for (method, size) in [
        (
            &clc797::DRAW_INDEX_BUFFER32_BEGIN_END_INSTANCE_FIRST,
            IndexFormat::U32,
        ),
        (
            &clc797::DRAW_INDEX_BUFFER16_BEGIN_END_INSTANCE_FIRST,
            IndexFormat::U16,
        ),
        (
            &clc797::DRAW_INDEX_BUFFER8_BEGIN_END_INSTANCE_FIRST,
            IndexFormat::U8,
        ),
    ] {
        let mut rig = Rig::new(ready(vec![], vec![]));
        rig.basic();
        rig.address(
            &clc797::SET_INDEX_BUFFER_A,
            &clc797::SET_INDEX_BUFFER_B,
            &[],
            INDICES,
        );
        rig.address(
            &clc797::SET_INDEX_BUFFER_SIZE_A,
            &clc797::SET_INDEX_BUFFER_SIZE_B,
            &[],
            INDICES_SIZE,
        );
        let other = if size == IndexFormat::U8 {
            clc797::SET_INDEX_BUFFER_E_INDEX_SIZE_FOUR_BYTES
        } else {
            clc797::SET_INDEX_BUFFER_E_INDEX_SIZE_ONE_BYTE
        };
        rig.set(
            &clc797::SET_INDEX_BUFFER_E,
            &[],
            fields(&[(clc797::SET_INDEX_BUFFER_E_INDEX_SIZE, other)]),
        );
        assert_eq!(
            rig.put(method, &[], 3),
            Ok(Some(Outcome::Drawn)),
            "{}",
            method.name
        );
        assert_eq!(
            rig.last_draw().index_buffer.unwrap().format,
            size,
            "{}",
            method.name
        );
    }
}

#[test]
fn the_small_index_forms_count_instances() {
    let mut rig = Rig::new(ready(vec![], vec![]));
    rig.basic();
    rig.address(
        &clc797::SET_INDEX_BUFFER_A,
        &clc797::SET_INDEX_BUFFER_B,
        &[],
        INDICES,
    );
    rig.address(
        &clc797::SET_INDEX_BUFFER_SIZE_A,
        &clc797::SET_INDEX_BUFFER_SIZE_B,
        &[],
        INDICES_SIZE,
    );
    rig.set(&clc797::SET_GLOBAL_BASE_INSTANCE_INDEX, &[], 10);
    let mut firsts = Vec::new();
    for method in [
        &clc797::DRAW_INDEX_BUFFER32_BEGIN_END_INSTANCE_FIRST,
        &clc797::DRAW_INDEX_BUFFER32_BEGIN_END_INSTANCE_SUBSEQUENT,
        &clc797::DRAW_INDEX_BUFFER32_BEGIN_END_INSTANCE_SUBSEQUENT,
        &clc797::DRAW_INDEX_BUFFER32_BEGIN_END_INSTANCE_FIRST,
    ] {
        rig.put(method, &[], 3).unwrap();
        let DrawParams::Indexed {
            first_instance,
            instance_count,
            ..
        } = rig.last_draw().params
        else {
            unreachable!("indexed");
        };
        assert_eq!(instance_count, 1);
        firsts.push(first_instance);
    }
    assert_eq!(firsts, [10, 11, 12, 10]);
}

#[test]
fn the_vertex_array_instanced_forms_count_from_zero() {
    let mut rig = Rig::new(ready(vec![], vec![]));
    rig.basic();
    let word = |start, count| {
        fields(&[
            (
                clc797::DRAW_VERTEX_ARRAY_BEGIN_END_INSTANCE_FIRST_START_INDEX,
                start,
            ),
            (
                clc797::DRAW_VERTEX_ARRAY_BEGIN_END_INSTANCE_FIRST_COUNT,
                count,
            ),
            (
                clc797::DRAW_VERTEX_ARRAY_BEGIN_END_INSTANCE_FIRST_TOPOLOGY,
                clc797::BEGIN_OP_POINTS,
            ),
        ])
    };
    rig.put(
        &clc797::DRAW_VERTEX_ARRAY_BEGIN_END_INSTANCE_FIRST,
        &[],
        word(4, 6),
    )
    .unwrap();
    assert_eq!(
        rig.last_draw().params,
        DrawParams::Direct {
            vertex_count: 6,
            instance_count: 1,
            first_vertex: 4,
            first_instance: 0
        }
    );
    rig.put(
        &clc797::DRAW_VERTEX_ARRAY_BEGIN_END_INSTANCE_SUBSEQUENT,
        &[],
        word(1, 2),
    )
    .unwrap();
    assert_eq!(
        rig.last_draw().params,
        DrawParams::Direct {
            vertex_count: 2,
            instance_count: 1,
            first_vertex: 1,
            first_instance: 1
        }
    );
}

#[test]
fn instanced_begin_end_defers_until_the_next_first_or_a_flush() {
    let mut rig = Rig::new(ready(vec![], vec![]));
    rig.basic();
    rig.set(&clc797::SET_VERTEX_ARRAY_START, &[], 0);
    rig.set(&clc797::DRAW_VERTEX_ARRAY, &[], 3);
    let begin = |id| {
        fields(&[
            (clc797::BEGIN_OP, clc797::BEGIN_OP_TRIANGLES),
            (clc797::BEGIN_INSTANCE_ID, id),
        ])
    };
    // FIRST then END draws one instance at once (Eden's general mode).
    assert_eq!(
        rig.put(&clc797::BEGIN, &[], begin(clc797::BEGIN_INSTANCE_ID_FIRST)),
        Ok(None)
    );
    assert_eq!(rig.put(&clc797::END, &[], 0), Ok(Some(Outcome::Drawn)));
    // Two SUBSEQUENT pairs defer.
    for _ in 0..2 {
        assert_eq!(
            rig.put(
                &clc797::BEGIN,
                &[],
                begin(clc797::BEGIN_INSTANCE_ID_SUBSEQUENT)
            ),
            Ok(None)
        );
        assert_eq!(rig.put(&clc797::END, &[], 0), Ok(None));
    }
    assert_eq!(rig.draw_commands().len(), 1);
    // A flush draws them: Eden's DrawDeferred, instance count + 1.
    assert_eq!(rig.flush(), Ok(Some(Outcome::Drawn)));
    assert!(matches!(
        rig.last_draw().params,
        DrawParams::Direct {
            instance_count: 3,
            ..
        }
    ));
    // Nothing is left to flush.
    assert_eq!(rig.flush(), Ok(None));
    // The same through a BEGIN with FIRST.
    assert_eq!(
        rig.put(
            &clc797::BEGIN,
            &[],
            begin(clc797::BEGIN_INSTANCE_ID_SUBSEQUENT)
        ),
        Ok(None)
    );
    assert_eq!(rig.put(&clc797::END, &[], 0), Ok(None));
    assert_eq!(
        rig.put(&clc797::BEGIN, &[], begin(clc797::BEGIN_INSTANCE_ID_FIRST)),
        Ok(Some(Outcome::Drawn))
    );
    assert!(matches!(
        rig.last_draw().params,
        DrawParams::Direct {
            instance_count: 2,
            ..
        }
    ));
}

#[test]
fn an_instanced_form_does_not_defer_a_begin_end_draw() {
    let mut rig = Rig::new(ready(vec![], vec![]));
    rig.basic();
    rig.put(&clc797::DRAW_VERTEX_ARRAY_BEGIN_END_INSTANCE_FIRST, &[], 3)
        .unwrap();
    rig.put(
        &clc797::DRAW_VERTEX_ARRAY_BEGIN_END_INSTANCE_SUBSEQUENT,
        &[],
        3,
    )
    .unwrap();
    // Eden's DrawArrayInstanced would leave instanced mode set and a flush would draw again.
    assert_eq!(rig.flush(), Ok(None));
}

#[test]
fn the_begin_end_b_forms_take_draw_control() {
    let mut rig = Rig::new(ready(vec![], vec![]));
    rig.basic();
    rig.set(&clc797::SET_GLOBAL_BASE_VERTEX_INDEX, &[], 100);
    rig.set(&clc797::SET_GLOBAL_BASE_INSTANCE_INDEX, &[], 200);
    rig.set(&clc797::SET_DRAW_CONTROL_B, &[], 4);
    rig.set(
        &clc797::SET_DRAW_CONTROL_A,
        &[],
        fields(&[
            (
                clc797::SET_DRAW_CONTROL_A_TOPOLOGY,
                clc797::SET_DRAW_CONTROL_A_TOPOLOGY_LINES,
            ),
            (
                clc797::SET_DRAW_CONTROL_A_IGNORE_GLOBAL_BASE_INSTANCE_INDEX,
                1,
            ),
        ]),
    );
    rig.set(&clc797::DRAW_VERTEX_ARRAY_BEGIN_END_A, &[], 6);
    assert_eq!(
        rig.put(&clc797::DRAW_VERTEX_ARRAY_BEGIN_END_B, &[], 10),
        Ok(Some(Outcome::Drawn))
    );
    assert_eq!(
        rig.last_draw().params,
        DrawParams::Direct {
            vertex_count: 10,
            instance_count: 4,
            first_vertex: 6,
            first_instance: 0
        }
    );
    rig.address(
        &clc797::SET_INDEX_BUFFER_A,
        &clc797::SET_INDEX_BUFFER_B,
        &[],
        INDICES,
    );
    rig.address(
        &clc797::SET_INDEX_BUFFER_SIZE_A,
        &clc797::SET_INDEX_BUFFER_SIZE_B,
        &[],
        INDICES_SIZE,
    );
    rig.set(
        &clc797::SET_DRAW_CONTROL_A,
        &[],
        fields(&[
            (
                clc797::SET_DRAW_CONTROL_A_TOPOLOGY,
                clc797::SET_DRAW_CONTROL_A_TOPOLOGY_LINES,
            ),
            (
                clc797::SET_DRAW_CONTROL_A_IGNORE_GLOBAL_BASE_VERTEX_INDEX,
                1,
            ),
        ]),
    );
    rig.set(&clc797::DRAW_INDEX_BUFFER_BEGIN_END_A, &[], 2);
    assert_eq!(
        rig.put(&clc797::DRAW_INDEX_BUFFER_BEGIN_END_B, &[], 8),
        Ok(Some(Outcome::Drawn))
    );
    assert_eq!(
        rig.last_draw().params,
        DrawParams::Indexed {
            index_count: 8,
            instance_count: 4,
            first_index: 2,
            base_vertex: 0,
            first_instance: 200,
        }
    );
}

#[test]
fn only_the_listed_methods_are_seen() {
    let mirror = Mirror::new(CHANNEL, ClassId(clc797::CLASS));
    let listed: Vec<u32> = call::DRAW_METHODS
        .iter()
        .chain(call::UNHANDLED_METHODS.iter())
        .map(|m| m.method())
        .collect();
    let mut seen = Vec::new();
    for method in 0..mirror.registers().len() {
        let method = u32::try_from(method).unwrap();
        let mut calls = DrawCalls::new();
        // A value whose DRAW_INDEX_BUFFER does nothing alone: a method that starts nothing and
        // changes no state leaves the next END drawing the vertex array.
        let started = calls.method(&mirror, method, 0).unwrap();
        let after = calls.method(&mirror, clc797::END.method(), 0).unwrap();
        let indexed =
            matches!(after, Started::Draw(c) if matches!(c.source, Source::Indexed { .. }));
        if started != Started::Nothing || indexed {
            seen.push(method);
        }
    }
    // BEGIN with value 0 is FIRST and starts nothing; DRAW_INDEX_BUFFER is seen through END.
    let mut expected: Vec<u32> = listed
        .iter()
        .copied()
        .filter(|&m| m != clc797::BEGIN.method())
        .collect();
    expected.sort_unstable();
    seen.sort_unstable();
    assert_eq!(seen, expected);
}

#[test]
fn unhandled_draw_methods_are_skipped_and_counted() {
    let mut rig = Rig::new(ready(vec![], vec![]));
    rig.basic();
    for method in call::UNHANDLED_METHODS {
        assert_eq!(
            rig.put(method, &[], 1),
            Ok(Some(Outcome::Skipped(Skip::Unhandled(method.name))))
        );
    }
    assert_eq!(rig.draws.counters().unhandled, 5);
    assert_eq!(rig.draw_commands(), []);
}

#[test]
fn draw_topologies_are_the_headers_literal_values() {
    // BEGIN_OP in clc797.h, written out.
    let cases = [
        (0x0, Topology::Points),
        (0x1, Topology::Lines),
        (0x2, Topology::LineLoop),
        (0x3, Topology::LineStrip),
        (0x4, Topology::Triangles),
        (0x5, Topology::TriangleStrip),
        (0x6, Topology::TriangleFan),
        (0x7, Topology::Quads),
        (0x8, Topology::QuadStrip),
        (0x9, Topology::Polygon),
        (0xa, Topology::LinesAdjacency),
        (0xb, Topology::LineStripAdjacency),
        (0xc, Topology::TrianglesAdjacency),
        (0xd, Topology::TriangleStripAdjacency),
        (0xe, Topology::Patches),
    ];
    for (value, topology) in cases {
        assert_eq!(fixed::topology(value), Ok(topology), "{value:#x}");
    }
    assert_eq!(
        fixed::topology(0xf),
        Err(DrawError::Topology { value: 0xf })
    );
    // Every draw method's topology field shares BEGIN_OP's values.
    assert_eq!(
        clc797::DRAW_VERTEX_ARRAY_BEGIN_END_INSTANCE_FIRST_TOPOLOGY_PATCH,
        0xe
    );
    assert_eq!(
        clc797::DRAW_INDEX_BUFFER32_BEGIN_END_INSTANCE_FIRST_TOPOLOGY_QUADS,
        0x7
    );
    assert_eq!(clc797::SET_DRAW_CONTROL_A_TOPOLOGY_TRIANGLE_FAN, 0x6);
}

#[test]
fn the_topology_override_maps_its_own_values_and_refuses_legacy_ones() {
    let mut rig = Rig::new(ready(vec![], vec![]));
    rig.basic();
    rig.set(
        &clc797::SET_PRIMITIVE_TOPOLOGY_CONTROL,
        &[],
        clc797::SET_PRIMITIVE_TOPOLOGY_CONTROL_OVERRIDE_USE_SEPARATE_TOPOLOGY_STATE,
    );
    // SET_PRIMITIVE_TOPOLOGY in clc797.h: LINELIST 2 is lines (BEGIN_OP's 2 is a line loop).
    for (value, topology) in [
        (0x1, Topology::Points),
        (0x2, Topology::Lines),
        (0x3, Topology::LineStrip),
        (0x4, Topology::Triangles),
        (0xe, Topology::Patches),
    ] {
        rig.set(&clc797::SET_PRIMITIVE_TOPOLOGY, &[], value);
        assert_eq!(
            fixed::effective_topology(&rig.mirror, clc797::BEGIN_OP_QUADS),
            Ok(topology)
        );
    }
    rig.set(
        &clc797::SET_PRIMITIVE_TOPOLOGY,
        &[],
        clc797::SET_PRIMITIVE_TOPOLOGY_V_LEGACY_POINTS,
    );
    assert_eq!(
        rig.draw_arrays(clc797::BEGIN_OP_TRIANGLES, 0, 3),
        Err(DrawError::Topology { value: 0x1001 })
    );
    // Without the override, BEGIN's own topology is used.
    rig.set(&clc797::SET_PRIMITIVE_TOPOLOGY_CONTROL, &[], 0);
    assert_eq!(
        fixed::effective_topology(&rig.mirror, clc797::BEGIN_OP_QUADS),
        Ok(Topology::Quads)
    );
}

// ---------------------------------------------------------------------------------------------
// Dirty groups.

#[test]
fn a_clean_group_is_not_re_derived() {
    let mut rig = Rig::new(ready(vec![], vec![]));
    rig.basic();
    rig.draw();
    let first = rig.draws.derivations();
    assert_eq!(first, [1; Part::ALL.len()]);
    assert!(rig.mirror.dirty().is_empty());
    // A second draw with nothing changed derives nothing.
    rig.draw();
    assert_eq!(rig.draws.derivations(), first);
    // Each part is derived again exactly when one of its groups changes, and only it.
    let writes: [(Part, &Method, &[u32], u32); 6] = [
        (
            Part::Viewports,
            &clc797::SET_VIEWPORT_SCALE_X,
            &[0],
            2.0f32.to_bits(),
        ),
        (Part::Scissors, &clc797::SET_SCISSOR_ENABLE, &[3], 0),
        (
            Part::Blend,
            &clc797::SET_BLEND_CONST_RED,
            &[],
            0.5f32.to_bits(),
        ),
        (
            Part::Raster,
            &clc797::SET_LINE_WIDTH_FLOAT,
            &[],
            2.0f32.to_bits(),
        ),
        (Part::DepthStencil, &clc797::SET_STENCIL_FUNC_REF, &[], 9),
        (
            Part::Vertex,
            &clc797::SET_VERTEX_STREAM_A_FORMAT,
            &[0],
            fields(&[
                (clc797::SET_VERTEX_STREAM_A_FORMAT_STRIDE, 24),
                (clc797::SET_VERTEX_STREAM_A_FORMAT_ENABLE, 1),
            ]),
        ),
    ];
    for (part, method, indices, value) in writes {
        let before = rig.draws.derivations();
        rig.set(method, indices, value);
        let dirty = rig.mirror.dirty();
        rig.draw();
        let after = rig.draws.derivations();
        for (i, p) in Part::ALL.iter().enumerate() {
            // The written part, and any other part that reads the same group (the scissors
            // read the viewport group, for the window origin and the surface clip).
            let reads = !fixed::part_groups(*p)
                .difference(fixed::part_groups(*p).difference(dirty))
                .is_empty();
            assert!(
                *p != part || reads,
                "{part:?} does not read what its write marked"
            );
            assert_eq!(
                after[i],
                before[i] + u64::from(reads),
                "{part:?} write, {p:?} count"
            );
        }
    }
    // A write to a group no part reads (a bind group) derives nothing.
    let before = rig.draws.derivations();
    rig.bind(4, 9, UNIFORMS, 0x10);
    rig.draw();
    assert_eq!(rig.draws.derivations(), before);
}

#[test]
fn every_group_a_part_reads_re_derives_it() {
    for part in Part::ALL {
        let groups = fixed::part_groups(part);
        assert!(!groups.is_empty());
        for group in groups.iter() {
            let mut parts = fixed::Parts::new();
            let mut rig = Rig::new(ready(vec![], vec![]));
            rig.basic();
            parts.refresh(&rig.mirror, rig.mirror.dirty()).unwrap();
            let before = parts.derivations();
            let only = crate::back::state::GroupSet::EMPTY.with(group);
            parts.refresh(&rig.mirror, only).unwrap();
            let after = parts.derivations();
            for (i, p) in Part::ALL.iter().enumerate() {
                let reads = fixed::part_groups(*p).contains(group);
                assert_eq!(
                    after[i],
                    before[i] + u64::from(reads),
                    "{group:?} for {p:?}"
                );
            }
        }
    }
}

#[test]
fn a_failed_part_is_derived_again_and_the_dirty_groups_are_kept() {
    let mut rig = Rig::new(ready(vec![], vec![]));
    rig.basic();
    // A front face the header does not define fails the raster part.
    rig.set(&clc797::OGL_SET_FRONT_FACE, &[], 0);
    let err = rig
        .draw_arrays(clc797::BEGIN_OP_TRIANGLES, 0, 3)
        .unwrap_err();
    assert!(
        matches!(
            err,
            DrawError::State(StateError::UndefinedValue {
                field: "V",
                value: 0,
                ..
            })
        ),
        "{err:?}"
    );
    assert!(
        rig.mirror
            .is_dirty(crate::back::state::Group::one(GroupKind::Rasterizer))
    );
    assert!(
        rig.mirror
            .is_dirty(crate::back::state::Group::one(GroupKind::DepthStencil))
    );
    assert_eq!(rig.draws.counters().failed, 1);
    rig.set(
        &clc797::OGL_SET_FRONT_FACE,
        &[],
        clc797::OGL_SET_FRONT_FACE_V_CW,
    );
    assert_eq!(rig.draw(), Outcome::Drawn);
    assert!(rig.mirror.dirty().is_empty());
}

// ---------------------------------------------------------------------------------------------
// Programs and lookups.

#[test]
fn requests_are_the_enabled_slots_with_their_addresses() {
    let mut rig = Rig::new(ready(vec![], vec![]));
    rig.basic();
    // Slot 2 enabled at a distinct address; slot 0 left disabled.
    rig.set(
        &clc797::SET_PIPELINE_SHADER,
        &[2],
        fields(&[(clc797::SET_PIPELINE_SHADER_ENABLE, 1)]),
    );
    rig.address(
        &clc797::SET_PIPELINE_PROGRAM_ADDRESS_A,
        &clc797::SET_PIPELINE_PROGRAM_ADDRESS_B,
        &[2],
        0x12_3456_7800,
    );
    rig.draw();
    let seen = &rig.shaders.seen[0];
    assert_eq!(
        seen.requests,
        [
            ProgramRequest {
                stage: GuestStage::Vertex,
                address: VERTEX_PROGRAM
            },
            ProgramRequest {
                stage: GuestStage::TessellationInit,
                address: 0x12_3456_7800
            },
            ProgramRequest {
                stage: GuestStage::Pixel,
                address: PIXEL_PROGRAM
            },
        ]
    );
    // The fake read the program through the draw path's memory.
    assert_eq!(seen.program_word, Ok(0x0bad_cafe_u32.to_le_bytes()));
}

#[test]
fn the_pixel_slot_is_skipped_while_rasterisation_is_off() {
    let mut rig = Rig::new(ready(vec![], vec![]));
    rig.basic();
    rig.set(
        &clc797::SET_RASTER_ENABLE,
        &[],
        clc797::SET_RASTER_ENABLE_V_FALSE,
    );
    rig.draw();
    assert_eq!(
        rig.shaders.seen[0].requests,
        [ProgramRequest {
            stage: GuestStage::Vertex,
            address: VERTEX_PROGRAM
        }]
    );
}

#[test]
fn a_rejected_set_skips_its_draws_and_logs_once() {
    let mut rig = Rig::new(SetLookup::Rejected(rejection()));
    rig.basic();
    let (outcomes, events) = captured(|| (0..3).map(|_| rig.draw()).collect::<Vec<_>>());
    assert_eq!(
        outcomes,
        [
            Outcome::Skipped(Skip::ShaderRejected { newly: true }),
            Outcome::Skipped(Skip::ShaderRejected { newly: false }),
            Outcome::Skipped(Skip::ShaderRejected { newly: false }),
        ]
    );
    let warnings = at(&events, tracing::Level::WARN);
    assert_eq!(warnings.len(), 1, "{events:?}");
    assert!(warnings[0].contains("BRX"), "{warnings:?}");
    let counters = rig.draws.counters();
    assert_eq!((counters.shader_rejected, counters.rejected_sets), (3, 1));
    assert_eq!(rig.draw_commands(), []);
    // Another set of requests is a new rejection, logged again.
    rig.address(
        &clc797::SET_PIPELINE_PROGRAM_ADDRESS_A,
        &clc797::SET_PIPELINE_PROGRAM_ADDRESS_B,
        &[5],
        PIXEL_PROGRAM + 0x100,
    );
    let (outcome, events) = captured(|| rig.draw());
    assert_eq!(
        outcome,
        Outcome::Skipped(Skip::ShaderRejected { newly: true })
    );
    assert_eq!(at(&events, tracing::Level::WARN).len(), 1);
    assert_eq!(rig.draws.counters().rejected_sets, 2);
}

#[test]
fn rejected_sets_past_the_cap_are_counted_not_remembered() {
    let mut rig = Rig::new(SetLookup::Rejected(rejection()));
    rig.basic();
    rig.draws.rejected_limit = 1;
    assert_eq!(
        rig.draw(),
        Outcome::Skipped(Skip::ShaderRejected { newly: true })
    );
    rig.address(
        &clc797::SET_PIPELINE_PROGRAM_ADDRESS_A,
        &clc797::SET_PIPELINE_PROGRAM_ADDRESS_B,
        &[5],
        PIXEL_PROGRAM + 0x100,
    );
    let (outcome, events) = captured(|| rig.draw());
    assert_eq!(
        outcome,
        Outcome::Skipped(Skip::ShaderRejected { newly: false })
    );
    assert_eq!(at(&events, tracing::Level::WARN), Vec::<String>::new());
    assert_eq!(at(&events, tracing::Level::DEBUG).len(), 1);
    let counters = rig.draws.counters();
    assert_eq!(
        (counters.rejected_sets, counters.rejected_unremembered),
        (1, 1)
    );
}

#[test]
fn a_pending_set_waits_under_the_wait_setting() {
    let mut rig = Rig::new(SetLookup::Pending(Ticket::new(4)));
    rig.shaders.waits.push_back(ready(vec![], vec![]));
    rig.basic();
    assert_eq!(rig.draw(), Outcome::Drawn);
    assert_eq!(rig.shaders.waited, [Ticket::new(4)]);
    // A wait that gives a rejection skips as a rejection.
    rig.shaders
        .waits
        .push_back(SetLookup::Rejected(rejection()));
    assert_eq!(
        rig.draw(),
        Outcome::Skipped(Skip::ShaderRejected { newly: true })
    );
    // A wait that is still pending skips.
    rig.shaders
        .waits
        .push_back(SetLookup::Pending(Ticket::new(5)));
    assert_eq!(rig.draw(), Outcome::Skipped(Skip::ShaderPending));
    assert_eq!(
        rig.shaders.waited,
        [Ticket::new(4), Ticket::new(4), Ticket::new(4)]
    );
}

#[test]
fn a_pending_set_is_skipped_under_the_skip_setting() {
    let mut rig = Rig::with(
        caps(true, Features::ALL, 16),
        WaitWhileCompiling::Skip,
        SetLookup::Pending(Ticket::new(4)),
    );
    rig.basic();
    assert_eq!(rig.draw(), Outcome::Skipped(Skip::ShaderPending));
    assert_eq!(rig.shaders.waited, []);
    assert_eq!(rig.draws.counters().shader_pending, 1);
    assert_eq!(rig.draws.setting(), WaitWhileCompiling::Skip);
}

#[test]
fn the_wait_setting_defaults_to_wait() {
    // Owner's decision 9: waiting is the default.
    assert_eq!(WaitWhileCompiling::default(), WaitWhileCompiling::Wait);
}

#[test]
fn a_set_without_a_vertex_shader_or_with_mismatched_stages_is_an_error() {
    let pixel_only = SetLookup::Ready(ReadySet {
        shaders: Arc::new(ShaderSet {
            stages: vec![host_shader(GuestStage::Pixel)],
        }),
        stages: vec![StageInfo {
            stamp: PIXEL_STAMP,
            resources: vec![],
        }],
    });
    let mut rig = Rig::new(pixel_only);
    rig.basic();
    assert_eq!(
        rig.draw_arrays(clc797::BEGIN_OP_TRIANGLES, 0, 3),
        Err(DrawError::BadSet("has no vertex shader"))
    );
    let mismatched = SetLookup::Ready(ReadySet {
        shaders: Arc::new(ShaderSet {
            stages: vec![host_shader(GuestStage::Vertex)],
        }),
        stages: vec![],
    });
    let mut rig = Rig::new(mismatched);
    rig.basic();
    assert_eq!(
        rig.draw_arrays(clc797::BEGIN_OP_TRIANGLES, 0, 3),
        Err(DrawError::BadSet(
            "has stage information for a different number of shaders"
        ))
    );
}

#[test]
fn a_failed_pipeline_skips_its_draws_newly_once() {
    let mut rig = Rig::new(ready(vec![], vec![]));
    rig.basic();
    // The pipeline is new: compiling on its first lookup, so this draw is drawn (the render
    // thread applies the wait setting to it).
    assert_eq!(rig.draw(), Outcome::Drawn);
    let pipeline = rig.last_draw().pipeline;
    rig.backend
        .script_pipeline(pipeline, vec![starquake_render::PipelineStatus::Failed])
        .unwrap();
    let (outcomes, events) = captured(|| (0..2).map(|_| rig.draw()).collect::<Vec<_>>());
    assert_eq!(
        outcomes,
        [
            Outcome::Skipped(Skip::PipelineFailed { newly: true }),
            Outcome::Skipped(Skip::PipelineFailed { newly: false }),
        ]
    );
    assert_eq!(at(&events, tracing::Level::WARN).len(), 1);
    assert_eq!(rig.draws.counters().pipeline_failed, 2);
}

// ---------------------------------------------------------------------------------------------
// The variant key.

#[test]
fn every_emulation_flag_reaches_the_key_field_it_mirrors() {
    assert_eq!(EmulationKey::MIRRORED.len(), EmulationFlags::NAMED.len());
    let named = |name: &str| {
        EmulationFlags::NAMED
            .iter()
            .find_map(|&(n, flag)| (n == name).then_some(flag))
            .unwrap()
    };
    let count = EmulationFlags::NAMED.len();
    for subset in 0u32..1 << count {
        let flags = EmulationFlags::NAMED
            .iter()
            .enumerate()
            .filter(|(i, _)| subset >> i & 1 != 0)
            .fold(EmulationFlags::EMPTY, |all, (_, &(_, flag))| all | flag);
        let key = shaders::emulation_key(flags, Some(CompareFunction::Less));
        for &(name, read) in EmulationKey::MIRRORED {
            assert_eq!(
                read(&key),
                flags.contains(named(name)),
                "{name} in {flags:?}"
            );
        }
    }
    // The alpha function is the guest's, and absent when the flag is clear even with a function.
    let key = shaders::emulation_key(
        EmulationFlags::ALPHA_TEST,
        Some(CompareFunction::GreaterEqual),
    );
    assert_eq!(key.alpha_test, Some(KeyCompare::GreaterThanEqual));
    assert_eq!(
        shaders::emulation_key(EmulationFlags::EMPTY, Some(CompareFunction::Less)).alpha_test,
        None
    );
}

#[test]
fn every_emulation_flag_reaches_the_key_through_a_draw() {
    // A device lacking every feature the flags stand in for, with 1 vertex attribute.
    let lacking = Features::ALL
        .difference(Features::DEPTH_RANGE_MINUS_ONE_TO_ONE)
        .difference(Features::PRIMITIVE_RESTART_OFF)
        .difference(Features::UNTYPED_FRAGMENT_OUTPUTS);
    let mut rig = Rig::with(
        caps(true, lacking, 1),
        WaitWhileCompiling::Wait,
        ready(vec![], vec![]),
    );
    rig.basic();
    // A register never written makes the clip range -1..1 (Eden's `DepthMode::MinusOneToOne` is
    // 0); start from 0..1 so one flag changes at a time.
    rig.set(
        &clc797::SET_VIEWPORT_Z_CLIP,
        &[],
        clc797::SET_VIEWPORT_Z_CLIP_RANGE_ZERO_TO_POSITIVE_W,
    );
    rig.draw();
    // Primitive restart is off, so it is emulated; nothing else is needed yet.
    let key = rig.shaders.seen.last().unwrap().key.emulation;
    assert_eq!(
        key,
        EmulationKey {
            primitive_restart_rewrite: true,
            ..EmulationKey::default()
        }
    );
    // Alpha test, -1..1 depth, a second attribute and an integer target.
    rig.set(&clc797::SET_ALPHA_TEST, &[], 1);
    rig.set(
        &clc797::SET_ALPHA_FUNC,
        &[],
        clc797::SET_ALPHA_FUNC_V_D3D_LESS,
    );
    rig.set(
        &clc797::SET_VIEWPORT_Z_CLIP,
        &[],
        clc797::SET_VIEWPORT_Z_CLIP_RANGE_NEGATIVE_W_TO_POSITIVE_W,
    );
    let second = fields(&[
        (
            clc797::SET_VERTEX_ATTRIBUTE_A_COMPONENT_BIT_WIDTHS,
            clc797::SET_VERTEX_ATTRIBUTE_A_COMPONENT_BIT_WIDTHS_R32,
        ),
        (
            clc797::SET_VERTEX_ATTRIBUTE_A_NUMERICAL_TYPE,
            clc797::SET_VERTEX_ATTRIBUTE_A_NUMERICAL_TYPE_NUM_UINT,
        ),
    ]);
    rig.set(&clc797::SET_VERTEX_ATTRIBUTE_A, &[1], second);
    rig.set(
        &clc797::SET_COLOR_TARGET_FORMAT,
        &[0],
        clc797::SET_COLOR_TARGET_FORMAT_V_AU8BU8GU8RU8,
    );
    rig.draw();
    let seen = rig.shaders.seen.last().unwrap();
    assert_eq!(
        seen.key.emulation,
        EmulationKey {
            alpha_test: Some(KeyCompare::Less),
            depth_range_rewrite: true,
            primitive_restart_rewrite: true,
            attribute_fetch: true,
            typed_outputs: true,
        }
    );
    assert_eq!(seen.key.vertex_attributes[0], AttributeType::Float);
    assert_eq!(seen.key.vertex_attributes[1], AttributeType::UnsignedInt);
    assert_eq!(seen.key.vertex_attributes[2], AttributeType::Disabled);
    assert_eq!(seen.key.color_targets[0], ColorType::UnsignedInt);
    assert_eq!(seen.key.color_targets[1], ColorType::Float);
    // The pipeline gets the same flags. With the second attribute past the device's one, the
    // backend refuses the pipeline (the fetch-from-buffer path is not built: every attribute is
    // still in the vertex layout), so that draw is skipped; without it the pipeline is made with
    // the other four flags.
    assert_eq!(
        rig.draw(),
        Outcome::Skipped(Skip::PipelineFailed { newly: false })
    );
    rig.set(&clc797::SET_VERTEX_ATTRIBUTE_A, &[1], 0);
    assert_eq!(rig.draw(), Outcome::Drawn);
    let snapshot = rig.submit();
    assert!(
        snapshot.contains(
            "emulation ALPHA_TEST|DEPTH_RANGE_REWRITE|PRIMITIVE_RESTART_REWRITE|TYPED_OUTPUTS"
        ),
        "{snapshot}"
    );
}

#[test]
fn colour_types_follow_the_target_formats() {
    assert_eq!(
        shaders::color_type(Some(PixelFormat::A8B8G8R8Sint)),
        ColorType::SignedInt
    );
    assert_eq!(
        shaders::color_type(Some(PixelFormat::R32Uint)),
        ColorType::UnsignedInt
    );
    assert_eq!(
        shaders::color_type(Some(PixelFormat::R16G16B16A16Float)),
        ColorType::Float
    );
    assert_eq!(shaders::color_type(None), ColorType::Float);
}

#[test]
fn dual_source_blending_reaches_the_key() {
    let mut rig = Rig::new(ready(vec![], vec![]));
    rig.basic();
    rig.set(&clc797::SET_BLEND, &[0], 1);
    for (method, value) in [
        (
            &clc797::SET_BLEND_COLOR_OP,
            clc797::SET_BLEND_COLOR_OP_V_D3D_ADD,
        ),
        (
            &clc797::SET_BLEND_COLOR_SOURCE_COEFF,
            clc797::SET_BLEND_COLOR_SOURCE_COEFF_V_D3D_ONE,
        ),
        (
            &clc797::SET_BLEND_COLOR_DEST_COEFF,
            clc797::SET_BLEND_COLOR_DEST_COEFF_V_OGL_SRC1COLOR,
        ),
        (
            &clc797::SET_BLEND_ALPHA_OP,
            clc797::SET_BLEND_ALPHA_OP_V_D3D_ADD,
        ),
        (
            &clc797::SET_BLEND_ALPHA_SOURCE_COEFF,
            clc797::SET_BLEND_ALPHA_SOURCE_COEFF_V_D3D_ONE,
        ),
        (
            &clc797::SET_BLEND_ALPHA_DEST_COEFF,
            clc797::SET_BLEND_ALPHA_DEST_COEFF_V_D3D_ZERO,
        ),
    ] {
        rig.set(method, &[], value);
    }
    rig.draw();
    assert!(rig.shaders.seen.last().unwrap().key.dual_source_blend);
    // A disabled target 0 is not read for it.
    rig.set(&clc797::SET_BLEND, &[0], 0);
    rig.draw();
    assert!(!rig.shaders.seen.last().unwrap().key.dual_source_blend);
}

#[test]
fn swizzles_reach_the_key_in_binding_order_only_without_view_swizzle() {
    let resources = vec![texture_use(0, 1), sampler_use(0, 2)];
    let mut rig = Rig::with(
        caps(false, Features::ALL, 16),
        WaitWhileCompiling::Wait,
        ready(vec![], resources.clone()),
    );
    rig.basic();
    // A swizzled header: B, G, R, one.
    let sources = [
        tex::TEXHEAD_BL_X_SOURCE_IN_B,
        tex::TEXHEAD_BL_X_SOURCE_IN_G,
        tex::TEXHEAD_BL_X_SOURCE_IN_R,
        tex::TEXHEAD_BL_X_SOURCE_IN_ONE_FLOAT,
    ];
    rig.write_words(
        HEADERS + u64::from(HEADER_INDEX) * shaders::HEADER_BYTES,
        &texture_header(TEXTURE, sources),
    );
    let swizzled = Swizzle {
        x: SwizzleSource::B,
        y: SwizzleSource::G,
        z: SwizzleSource::R,
        w: SwizzleSource::OneFloat,
    };
    // First draw: the set is unknown, so the first lookup has no swizzles, the second has them.
    assert_eq!(rig.draw(), Outcome::Drawn);
    let keys: Vec<_> = rig
        .shaders
        .seen
        .iter()
        .map(|s| s.key.texture_swizzles.as_slice().to_vec())
        .collect();
    assert_eq!(keys, [vec![], vec![swizzled]]);
    // Second draw: one lookup, with the swizzles remembered.
    rig.draw();
    assert_eq!(rig.shaders.seen.len(), 3);
    assert_eq!(
        rig.shaders.seen[2].key.texture_swizzles.as_slice(),
        [swizzled]
    );
    assert_eq!(rig.draws.counters().lookups, 3);

    // With view swizzle the key never carries them.
    let mut rig = Rig::new(ready(vec![], resources));
    rig.basic();
    rig.write_words(
        HEADERS + u64::from(HEADER_INDEX) * shaders::HEADER_BYTES,
        &texture_header(TEXTURE, sources),
    );
    rig.draw();
    rig.draw();
    // Three lookups: the first draw looks the set up again for the vertex stage's stores
    // (`the_previous_stages_stores_reach_the_key`), not for swizzles.
    assert_eq!(rig.shaders.seen.len(), 3);
    assert!(
        rig.shaders
            .seen
            .iter()
            .all(|s| s.key.texture_swizzles.as_slice().is_empty())
    );
}

#[test]
fn header_swizzle_codes_are_the_headers() {
    // X_SOURCE codes in clc797tex.h: IN_ZERO 0, IN_R 2 .. IN_A 5, IN_ONE_INT 6, IN_ONE_FLOAT 7.
    let words = texture_header(TEXTURE, [0, 6, 7, 5]);
    assert_eq!(
        shaders::header_swizzle(&words),
        Some(Swizzle {
            x: SwizzleSource::Zero,
            y: SwizzleSource::OneInt,
            z: SwizzleSource::OneFloat,
            w: SwizzleSource::A
        })
    );
    // Code 1 is no source.
    assert_eq!(
        shaders::header_swizzle(&texture_header(TEXTURE, [2, 1, 4, 5])),
        None
    );
}

// ---------------------------------------------------------------------------------------------
// Answering: ProgramMemory and EngineState.

#[test]
fn program_memory_reports_each_fault_at_its_address() {
    let mem = memory();
    let programs = GuestPrograms(&mem);
    let mut out = [0u8; 8];
    assert_eq!(programs.read(BASE, &mut out), Ok(()));
    let past = BASE + PAGES * PAGE_SIZE;
    assert_eq!(
        programs.read(past, &mut out),
        Err(Unmapped { address: past })
    );
    // A read running off the end of the address space: reported at its start.
    assert_eq!(
        programs.read(u64::MAX - 3, &mut out),
        Err(Unmapped {
            address: u64::MAX - 3
        })
    );
    // A read starting inside and ending outside: reported where it leaves the mapping.
    assert_eq!(
        programs.read(past - 4, &mut out),
        Err(Unmapped { address: past })
    );
}

#[test]
fn a_stage_reads_the_bind_group_of_its_slot_not_the_pipeline_binding() {
    let mut rig = Rig::new(ready(vec![], vec![]));
    rig.basic();
    rig.write_words(UNIFORMS, &[0x1111_1111]);
    rig.bind(0, 7, UNIFORMS, 0x10);
    // The pixel pipeline names bind group 0; Eden reads group 4 for the pixel stage.
    rig.set(&clc797::SET_PIPELINE_BINDING, &[5], 0);
    let answers = StateAnswers {
        mirror: &rig.mirror,
        roots: &RootTables::new(),
        memory: &rig.mem,
    };
    assert_eq!(
        answers.cbuf_word(GuestStage::Vertex, 7, 0),
        Some(0x1111_1111)
    );
    assert_eq!(
        answers.cbuf_word(GuestStage::VertexCullBeforeFetch, 7, 0),
        Some(0x1111_1111)
    );
    assert_eq!(answers.cbuf_word(GuestStage::Pixel, 7, 0), None);
    // Each stage's group: 0, 1, 2, 3, 4.
    for (stage, group) in [
        (GuestStage::Vertex, 0),
        (GuestStage::TessellationInit, 1),
        (GuestStage::Tessellation, 2),
        (GuestStage::Geometry, 3),
        (GuestStage::Pixel, 4),
    ] {
        assert_eq!(shaders::bind_group_of(stage), Some(group));
    }
    assert_eq!(shaders::bind_group_of(GuestStage::Compute), None);
}

#[test]
fn a_constant_buffer_word_must_fit_inside_the_bound_size() {
    let mut rig = Rig::new(ready(vec![], vec![]));
    rig.basic();
    rig.write_words(UNIFORMS + 0xc, &[0xabcd_0123]);
    rig.bind(4, 6, UNIFORMS, 0x10);
    let answers = StateAnswers {
        mirror: &rig.mirror,
        roots: &RootTables::new(),
        memory: &rig.mem,
    };
    // Bytes 0xc..0x10 end exactly at the size.
    assert_eq!(
        answers.cbuf_word(GuestStage::Pixel, 6, 0xc),
        Some(0xabcd_0123)
    );
    // 0xd..0x11 does not fit (Eden's offset < size would read it).
    assert_eq!(answers.cbuf_word(GuestStage::Pixel, 6, 0xd), None);
    assert_eq!(answers.cbuf_word(GuestStage::Pixel, 6, 0x10), None);
    assert_eq!(answers.cbuf_word(GuestStage::Pixel, 6, u32::MAX), None);
}

#[test]
fn unanswerable_questions_answer_none() {
    let mut rig = Rig::new(ready(vec![], vec![]));
    rig.basic();
    let roots = RootTables::new();
    let answers = StateAnswers {
        mirror: &rig.mirror,
        roots: &roots,
        memory: &rig.mem,
    };
    // An unbound bank.
    assert_eq!(answers.cbuf_word(GuestStage::Pixel, 9, 0), None);
    // Compute has no bind group.
    assert_eq!(answers.cbuf_word(GuestStage::Compute, HANDLE_BANK, 0), None);
    // A header index one past the pool's maximum (7), and the maximum itself.
    assert_eq!(answers.texture_kind(GuestStage::Pixel, 8), None);
    assert!(
        answers
            .texture_kind(GuestStage::Pixel, HEADER_INDEX)
            .is_some()
    );
    // An unmapped word.
    rig.bind(4, 9, BASE + PAGES * PAGE_SIZE, 0x10);
    let answers = StateAnswers {
        mirror: &rig.mirror,
        roots: &roots,
        memory: &rig.mem,
    };
    assert_eq!(answers.cbuf_word(GuestStage::Pixel, 9, 0), None);
}

#[test]
fn texture_kinds_describe_the_header() {
    let words = texture_header(TEXTURE, swizzle_sources(tex::TEXHEAD_BL_X_SOURCE_IN_R));
    let kind = shaders::texture_kind(&words).unwrap();
    assert_eq!(kind.sampled, SampledType::Float);
    assert!(!kind.depth && !kind.multisampled && !kind.buffer);
    assert_eq!(kind.format, TextureFormat::from_texhead_word0(words[0]));
    // An unsigned integer format (A8B8G8R8 UINT).
    let mut uint = words;
    for field in [
        tex::TEXHEAD_BL_R_DATA_TYPE,
        tex::TEXHEAD_BL_G_DATA_TYPE,
        tex::TEXHEAD_BL_B_DATA_TYPE,
        tex::TEXHEAD_BL_A_DATA_TYPE,
    ] {
        field
            .insert(&mut uint, tex::TEXHEAD_BL_R_DATA_TYPE_NUM_UINT)
            .unwrap();
    }
    assert_eq!(
        shaders::texture_kind(&uint).unwrap().sampled,
        SampledType::UnsignedInt
    );
    // A buffer header.
    let mut buffer = words;
    tex::TEXHEAD_BL_HEADER_VERSION
        .insert(
            &mut buffer,
            tex::TEXHEAD_BL_HEADER_VERSION_SELECT_ONE_D_BUFFER,
        )
        .unwrap();
    assert!(shaders::texture_kind(&buffer).unwrap().buffer);
    // A multisampled one.
    let mut msaa = words;
    tex::TEXHEAD_BL_MULTI_SAMPLE_COUNT
        .insert(&mut msaa, tex::TEXHEAD_BL_MULTI_SAMPLE_COUNT_MODE_2X2)
        .unwrap();
    assert!(shaders::texture_kind(&msaa).unwrap().multisampled);
    // An unknown format answers none.
    let mut unknown = words;
    tex::TEXHEAD_BL_COMPONENTS.insert(&mut unknown, 0).unwrap();
    assert_eq!(shaders::texture_kind(&unknown), None);
}

#[test]
fn the_handle_names_a_header_and_a_sampler_or_one_index_for_both() {
    let mut rig = Rig::new(ready(vec![], vec![]));
    rig.basic();
    let answers = StateAnswers {
        mirror: &rig.mirror,
        roots: &RootTables::new(),
        memory: &rig.mem,
    };
    assert_eq!(answers.handle_pair(0xfff0_0003), (3, 0xfff));
    assert_eq!(answers.handle_pair(0x000f_ffff), (0xf_ffff, 0));
    rig.set(
        &clc797::SET_SAMPLER_BINDING,
        &[],
        clc797::SET_SAMPLER_BINDING_V_VIA_HEADER_BINDING,
    );
    let answers = StateAnswers {
        mirror: &rig.mirror,
        roots: &RootTables::new(),
        memory: &rig.mem,
    };
    assert_eq!(answers.handle_pair(0xfff0_0003), (0xfff0_0003, 0xfff0_0003));
    // The fake read the pixel stage's handle word through the draw path's EngineState.
    rig.draw();
    assert_eq!(
        rig.shaders.seen[0].handle_word,
        Some(HEADER_INDEX | (SAMPLER_INDEX << shaders::SAMPLER_INDEX_SHIFT))
    );
}

// ---------------------------------------------------------------------------------------------
// Resources.

#[test]
fn bindings_follow_the_resource_table() {
    let mut rig = Rig::new(ready(vec![uniform(1, 0, 3, 0x20)], pixel_resources()));
    rig.basic();
    rig.bind(0, 3, UNIFORMS + 0x40, 0x80);
    assert_eq!(rig.draw(), Outcome::Drawn);
    let draw = rig.last_draw();
    let slots: Vec<_> = draw.bindings.iter().map(|b| (b.group, b.binding)).collect();
    assert_eq!(slots, [(1, 0), (0, 0), (0, 1), (0, 2), (0, 3), (0, 4)]);
    // The vertex uniform binds the shader's 0x20 bytes of the 0x80 bound.
    let BindingResource::Buffer(vertex_uniform) = draw.bindings[0].resource else {
        unreachable!()
    };
    assert_eq!(vertex_uniform.size, 0x20);
    // Bank 24 and the driver constants share the constants buffer, at their offsets.
    let (BindingResource::Buffer(root), BindingResource::Buffer(driver)) =
        (draw.bindings[4].resource, draw.bindings[5].resource)
    else {
        unreachable!()
    };
    assert_eq!(root.buffer, driver.buffer);
    assert_eq!(
        (root.offset, root.size),
        (0, u64::try_from(ROOT_TABLE_BYTES).unwrap())
    );
    assert_eq!(driver.offset, DRIVER_CONSTANTS_OFFSET);
    assert!(matches!(
        draw.bindings[2].resource,
        BindingResource::View(_)
    ));
    assert!(matches!(
        draw.bindings[3].resource,
        BindingResource::Sampler(_)
    ));
}

#[test]
fn an_unbound_constant_buffer_is_an_error() {
    let mut rig = Rig::new(ready(vec![], vec![uniform(0, 0, 11, 0x10)]));
    rig.basic();
    assert_eq!(
        rig.draw_arrays(clc797::BEGIN_OP_TRIANGLES, 0, 3),
        Err(DrawError::UnboundConstantBuffer {
            stage: GuestStage::Pixel,
            bank: 11
        })
    );
}

#[test]
fn a_handle_past_a_pool_maximum_is_an_error() {
    let mut rig = Rig::new(ready(vec![], vec![texture_use(0, 1)]));
    rig.basic();
    rig.set(&clc797::SET_TEX_HEADER_POOL_C, &[], HEADER_INDEX - 1);
    assert_eq!(
        rig.draw_arrays(clc797::BEGIN_OP_TRIANGLES, 0, 3),
        Err(DrawError::PoolIndex {
            pool: "texture header",
            index: HEADER_INDEX,
            maximum: HEADER_INDEX - 1
        })
    );
    rig.set(&clc797::SET_TEX_HEADER_POOL_C, &[], HEADER_INDEX);
    assert_eq!(rig.draw(), Outcome::Drawn);
    let mut rig = Rig::new(ready(vec![], vec![sampler_use(0, 2)]));
    rig.basic();
    rig.set(&clc797::SET_TEX_SAMPLER_POOL_C, &[], SAMPLER_INDEX - 1);
    assert_eq!(
        rig.draw_arrays(clc797::BEGIN_OP_TRIANGLES, 0, 3),
        Err(DrawError::PoolIndex {
            pool: "sampler",
            index: SAMPLER_INDEX,
            maximum: SAMPLER_INDEX - 1
        })
    );
    rig.set(&clc797::SET_TEX_SAMPLER_POOL_C, &[], SAMPLER_INDEX);
    assert_eq!(rig.draw(), Outcome::Drawn);
}

#[test]
fn a_failed_draw_leaves_only_cache_uploads() {
    // The uniform binds (an upload), then the unbound bank fails: no Draw, no target change.
    let mut rig = Rig::new(ready(
        vec![],
        vec![uniform(0, 0, 1, 0x10), uniform(0, 1, 12, 0x10)],
    ));
    rig.basic();
    assert!(rig.draw_arrays(clc797::BEGIN_OP_TRIANGLES, 0, 3).is_err());
    assert_ne!(rig.list.commands.len(), 0);
    assert!(
        rig.list
            .commands
            .iter()
            .all(|c| matches!(c, Command::Upload { .. })),
        "{:?}",
        rig.list.commands
    );
}

#[test]
fn render_targets_are_set_once_until_they_change() {
    let mut rig = Rig::new(ready(vec![], vec![]));
    rig.basic();
    rig.draw();
    rig.draw();
    let sets = |rig: &Rig| {
        rig.list
            .commands
            .iter()
            .filter(|c| matches!(c, Command::SetRenderTargets(_)))
            .count()
    };
    assert_eq!(sets(&rig), 1);
    rig.address(
        &clc797::SET_COLOR_TARGET_A,
        &clc797::SET_COLOR_TARGET_B,
        &[0],
        TARGET + 0x1000,
    );
    rig.draw();
    assert_eq!(sets(&rig), 2);
    // A new list sets them again.
    rig.submit();
    rig.draw();
    assert_eq!(sets(&rig), 1);
}

#[test]
fn the_constants_buffer_is_uploaded_when_its_bytes_change() {
    let mut rig = Rig::new(ready(vec![], vec![driver_use(0, 0)]));
    rig.basic();
    let uploads_to = |rig: &Rig, buffer| {
        rig.list
            .commands
            .iter()
            .filter(|c| matches!(c, Command::Upload { to: starquake_render::command::CopyRegion::Buffer(s), .. } if s.buffer == buffer))
            .count()
    };
    rig.draw();
    let BindingResource::Buffer(slot) = rig.last_draw().bindings[0].resource else {
        unreachable!()
    };
    assert_eq!(uploads_to(&rig, slot.buffer), 1);
    rig.draw();
    assert_eq!(uploads_to(&rig, slot.buffer), 1);
    // A different base instance changes the block.
    rig.set(&clc797::SET_GLOBAL_BASE_INSTANCE_INDEX, &[], 3);
    rig.draw();
    assert_eq!(uploads_to(&rig, slot.buffer), 2);
}

#[test]
fn driver_constants_take_their_sources() {
    let mut rig = Rig::new(ready(vec![], vec![]));
    rig.basic();
    rig.set(&clc797::SET_ALPHA_REF, &[], 0.25f32.to_bits());
    rig.set(
        &clc797::SET_WINDOW_ORIGIN,
        &[],
        fields(&[(
            clc797::SET_WINDOW_ORIGIN_MODE,
            clc797::SET_WINDOW_ORIGIN_MODE_LOWER_LEFT,
        )]),
    );
    let call = call::DrawCall {
        topology: 0,
        source: Source::Vertices { first: 0, count: 3 },
        instance_count: 1,
        base_instance: 5,
        base_vertex: 0xffff_fffe,
    };
    let constants = driver_constants(&rig.mirror, 0.25, &call).unwrap();
    assert_eq!(constants.alpha_reference, 0.25);
    assert!(constants.y_negate);
    assert_eq!(
        (constants.render_area_width, constants.render_area_height),
        (8.0, 8.0)
    );
    assert_eq!(
        (constants.base_vertex, constants.base_instance),
        (0xffff_fffe, 5)
    );
    // With no surface clip, colour target 0's size (Eden-NX2).
    rig.set(&clc797::SET_SURFACE_CLIP_HORIZONTAL, &[], 0);
    rig.set(&clc797::SET_COLOR_TARGET_WIDTH, &[0], 32);
    rig.set(&clc797::SET_COLOR_TARGET_HEIGHT, &[0], 16);
    let constants = driver_constants(&rig.mirror, 0.25, &call).unwrap();
    assert_eq!(
        (constants.render_area_width, constants.render_area_height),
        (32.0, 16.0)
    );
}

#[test]
fn every_active_attribute_is_in_the_layout() {
    let mut rig = Rig::new(ready(vec![], vec![]));
    rig.basic();
    // Attribute 3 reads stream 2; stream 1 is enabled but read by none.
    let attr = fields(&[
        (clc797::SET_VERTEX_ATTRIBUTE_A_STREAM, 2),
        (
            clc797::SET_VERTEX_ATTRIBUTE_A_COMPONENT_BIT_WIDTHS,
            clc797::SET_VERTEX_ATTRIBUTE_A_COMPONENT_BIT_WIDTHS_R16_G16,
        ),
        (
            clc797::SET_VERTEX_ATTRIBUTE_A_NUMERICAL_TYPE,
            clc797::SET_VERTEX_ATTRIBUTE_A_NUMERICAL_TYPE_NUM_SNORM,
        ),
    ]);
    rig.set(&clc797::SET_VERTEX_ATTRIBUTE_A, &[3], attr);
    for stream in [1, 2] {
        rig.set(
            &clc797::SET_VERTEX_STREAM_A_FORMAT,
            &[stream],
            fields(&[
                (clc797::SET_VERTEX_STREAM_A_FORMAT_STRIDE, 4),
                (clc797::SET_VERTEX_STREAM_A_FORMAT_ENABLE, 1),
            ]),
        );
        rig.address(
            &clc797::SET_VERTEX_STREAM_A_LOCATION_A,
            &clc797::SET_VERTEX_STREAM_A_LOCATION_B,
            &[stream],
            VERTICES,
        );
        rig.address(
            &clc797::SET_VERTEX_STREAM_SIZE_A,
            &clc797::SET_VERTEX_STREAM_SIZE_B,
            &[stream],
            0x40,
        );
    }
    rig.draw();
    let snapshot = rig.submit();
    let draw = rig
        .backend
        .lists()
        .last()
        .unwrap()
        .1
        .commands
        .iter()
        .find_map(|c| match c {
            Command::Draw(d) => Some(d.clone()),
            _ => None,
        })
        .unwrap();
    assert_eq!(draw.vertex_buffers.len(), 2, "{snapshot}");
    // An active attribute on a disabled stream is refused.
    rig.set(&clc797::SET_VERTEX_STREAM_A_FORMAT, &[2], 0);
    assert_eq!(
        rig.draw_arrays(clc797::BEGIN_OP_TRIANGLES, 0, 3),
        Err(DrawError::AttributeStream {
            attribute: 3,
            stream: 2
        })
    );
}

// ---------------------------------------------------------------------------------------------
// Conditional render.

#[test]
fn conditional_render_is_latched_when_the_mode_is_written() {
    let mut rig = Rig::new(ready(vec![], vec![]));
    rig.basic();
    // Before any mode write, draws render although the register holds FALSE.
    assert_eq!(rig.draw(), Outcome::Drawn);
    rig.set(
        &clc797::SET_RENDER_ENABLE_C,
        &[],
        clc797::SET_RENDER_ENABLE_C_MODE_FALSE,
    );
    assert_eq!(rig.draw(), Outcome::Skipped(Skip::RenderDisabled));
    assert_eq!(rig.draws.counters().render_disabled, 1);
    rig.set(
        &clc797::SET_RENDER_ENABLE_C,
        &[],
        clc797::SET_RENDER_ENABLE_C_MODE_TRUE,
    );
    assert_eq!(rig.draw(), Outcome::Drawn);
    // A conditional mode reads the report when written; a later report change is not seen.
    rig.address(
        &clc797::SET_RENDER_ENABLE_A,
        &clc797::SET_RENDER_ENABLE_B,
        &[],
        REPORT,
    );
    rig.write_words(REPORT, &[1, 1, 0, 0, 1, 1]);
    rig.set(
        &clc797::SET_RENDER_ENABLE_C,
        &[],
        clc797::SET_RENDER_ENABLE_C_MODE_RENDER_IF_NOT_EQUAL,
    );
    rig.write_words(REPORT, &[1, 1, 0, 0, 2, 1]);
    assert_eq!(rig.draw(), Outcome::Skipped(Skip::RenderDisabled));
}

#[test]
fn each_render_enable_mode_and_override_decides_as_eden() {
    use clc797::*;
    // (override, mode, report, renders)
    let report_equal = [1, 2, 9, 9, 1, 2];
    let report_differs_in_mode = [1, 2, 9, 9, 1, 3];
    let report_differs_in_sequence = [1, 2, 9, 9, 4, 2];
    let zero_mode = [1, 0, 0, 0, 1, 0];
    let cases = [
        (
            SET_RENDER_ENABLE_OVERRIDE_MODE_ALWAYS_RENDER,
            SET_RENDER_ENABLE_C_MODE_FALSE,
            report_equal,
            true,
        ),
        (
            SET_RENDER_ENABLE_OVERRIDE_MODE_NEVER_RENDER,
            SET_RENDER_ENABLE_C_MODE_TRUE,
            report_equal,
            false,
        ),
        (
            SET_RENDER_ENABLE_OVERRIDE_MODE_USE_RENDER_ENABLE,
            SET_RENDER_ENABLE_C_MODE_TRUE,
            report_equal,
            true,
        ),
        (
            SET_RENDER_ENABLE_OVERRIDE_MODE_USE_RENDER_ENABLE,
            SET_RENDER_ENABLE_C_MODE_FALSE,
            report_equal,
            false,
        ),
        (
            SET_RENDER_ENABLE_OVERRIDE_MODE_USE_RENDER_ENABLE,
            SET_RENDER_ENABLE_C_MODE_CONDITIONAL,
            report_equal,
            true,
        ),
        (
            SET_RENDER_ENABLE_OVERRIDE_MODE_USE_RENDER_ENABLE,
            SET_RENDER_ENABLE_C_MODE_CONDITIONAL,
            zero_mode,
            false,
        ),
        (
            SET_RENDER_ENABLE_OVERRIDE_MODE_USE_RENDER_ENABLE,
            SET_RENDER_ENABLE_C_MODE_CONDITIONAL,
            [0, 2, 0, 0, 0, 0],
            false,
        ),
        (
            SET_RENDER_ENABLE_OVERRIDE_MODE_USE_RENDER_ENABLE,
            SET_RENDER_ENABLE_C_MODE_RENDER_IF_EQUAL,
            report_equal,
            true,
        ),
        (
            SET_RENDER_ENABLE_OVERRIDE_MODE_USE_RENDER_ENABLE,
            SET_RENDER_ENABLE_C_MODE_RENDER_IF_EQUAL,
            report_differs_in_mode,
            false,
        ),
        (
            SET_RENDER_ENABLE_OVERRIDE_MODE_USE_RENDER_ENABLE,
            SET_RENDER_ENABLE_C_MODE_RENDER_IF_EQUAL,
            report_differs_in_sequence,
            false,
        ),
        (
            SET_RENDER_ENABLE_OVERRIDE_MODE_USE_RENDER_ENABLE,
            SET_RENDER_ENABLE_C_MODE_RENDER_IF_NOT_EQUAL,
            report_equal,
            false,
        ),
        (
            SET_RENDER_ENABLE_OVERRIDE_MODE_USE_RENDER_ENABLE,
            SET_RENDER_ENABLE_C_MODE_RENDER_IF_NOT_EQUAL,
            report_differs_in_mode,
            true,
        ),
        (
            SET_RENDER_ENABLE_OVERRIDE_MODE_USE_RENDER_ENABLE,
            SET_RENDER_ENABLE_C_MODE_RENDER_IF_NOT_EQUAL,
            report_differs_in_sequence,
            true,
        ),
    ];
    for (override_mode, mode, report, renders) in cases {
        let mut rig = Rig::new(ready(vec![], vec![]));
        rig.basic();
        rig.address(&SET_RENDER_ENABLE_A, &SET_RENDER_ENABLE_B, &[], REPORT);
        rig.write_words(REPORT, &report);
        rig.set(&SET_RENDER_ENABLE_OVERRIDE, &[], override_mode);
        rig.set(&SET_RENDER_ENABLE_C, &[], mode);
        let expected = if renders {
            Outcome::Drawn
        } else {
            Outcome::Skipped(Skip::RenderDisabled)
        };
        assert_eq!(rig.draw(), expected, "{override_mode} {mode} {report:?}");
    }
}

#[test]
fn an_undefined_render_enable_mode_is_an_error() {
    let mut rig = Rig::new(ready(vec![], vec![]));
    rig.basic();
    rig.set(
        &clc797::SET_RENDER_ENABLE_C,
        &[],
        clc797::SET_RENDER_ENABLE_C_MODE_FALSE,
    );
    let err = rig.put(&clc797::SET_RENDER_ENABLE_C, &[], 5).unwrap_err();
    assert_eq!(
        err.to_string(),
        "reading 3D state: SET_RENDER_ENABLE_C MODE holds 0x5, which the class header does not define"
    );
    // The kept answer is unchanged: still off.
    assert_eq!(rig.draw(), Outcome::Skipped(Skip::RenderDisabled));
    // An unreadable report is an error too, and keeps the answer.
    rig.address(
        &clc797::SET_RENDER_ENABLE_A,
        &clc797::SET_RENDER_ENABLE_B,
        &[],
        BASE + PAGES * PAGE_SIZE,
    );
    assert_eq!(
        rig.put(
            &clc797::SET_RENDER_ENABLE_C,
            &[],
            clc797::SET_RENDER_ENABLE_C_MODE_CONDITIONAL
        ),
        Err(DrawError::Fault(GpuFault::Unmapped {
            va: BASE + PAGES * PAGE_SIZE
        }))
    );
    assert_eq!(rig.draw(), Outcome::Skipped(Skip::RenderDisabled));
}

// ---------------------------------------------------------------------------------------------
// Root tables.

#[test]
fn root_table_loads_land_at_the_selected_offsets() {
    let mut rig = Rig::new(ready(vec![], vec![]));
    let selector = |table, offset| {
        fields(&[
            (clc797::SET_ROOT_TABLE_SELECTOR_ROOT_TABLE, table),
            (clc797::SET_ROOT_TABLE_SELECTOR_OFFSET, offset),
        ])
    };
    rig.set(&clc797::SET_ROOT_TABLE_SELECTOR, &[], selector(3, 0x08));
    let load = clc797::LOAD_ROOT_TABLE.method();
    rig.mirror.run(load, &[0xa, 0xb, 0xc]).unwrap();
    rig.draws
        .load_root_table(&rig.mirror, &[0xa, 0xb, 0xc])
        .unwrap();
    let roots = rig.draws.root_tables();
    let bank = FIRST_ROOT_BANK + 3;
    assert_eq!(
        [
            roots.word(bank, 0x8),
            roots.word(bank, 0xc),
            roots.word(bank, 0x10)
        ],
        [Some(0xa), Some(0xb), Some(0xc)]
    );
    assert_eq!(roots.word(bank, 0x4), Some(0));
    assert_eq!(roots.word(bank - 1, 0x8), Some(0));
    assert_eq!(roots.generation(), 3);
    // A word at an offset not a multiple of four is dropped (PR #4588's Load).
    rig.set(&clc797::SET_ROOT_TABLE_SELECTOR, &[], selector(0, 0x02));
    rig.mirror.run(load, &[0x77]).unwrap();
    rig.draws.load_root_table(&rig.mirror, &[0x77]).unwrap();
    assert_eq!(rig.draws.root_tables().dropped(), 1);
    assert_eq!(rig.draws.root_tables().generation(), 3);
}

#[test]
fn root_table_words_are_bounded_by_bank_and_offset() {
    let mut roots = RootTables::new();
    let mirror = {
        let mut m = Mirror::new(CHANNEL, ClassId(clc797::CLASS));
        m.write(
            clc797::SET_ROOT_TABLE_SELECTOR.method(),
            fields(&[
                (clc797::SET_ROOT_TABLE_SELECTOR_ROOT_TABLE, 7),
                (clc797::SET_ROOT_TABLE_SELECTOR_OFFSET, 0xfc),
            ]),
        )
        .unwrap();
        m
    };
    roots.load(&mirror, &[0x55]).unwrap();
    let last = FIRST_ROOT_BANK + 7;
    let end = u32::try_from(ROOT_TABLE_BYTES).unwrap();
    assert_eq!(roots.word(last, end - 4), Some(0x55));
    assert_eq!(roots.word(last, end), None);
    assert_eq!(roots.word(last + 1, 0), None);
    assert_eq!(roots.word(FIRST_ROOT_BANK - 1, 0), None);
    assert_eq!(roots.word(last, 2), None);
    // Banks 24 to 31 answer translation from the tables.
    let rig = Rig::new(ready(vec![], vec![]));
    let answers = StateAnswers {
        mirror: &rig.mirror,
        roots: &roots,
        memory: &rig.mem,
    };
    assert_eq!(
        answers.cbuf_word(GuestStage::Pixel, last, end - 4),
        Some(0x55)
    );
    // The constants buffer holds them at the table's offset.
    let bytes = constants_bytes(
        &roots,
        &starquake_shader::driver_constants::DriverConstants::default().encode(),
    );
    let at = 7 * ROOT_TABLE_BYTES + ROOT_TABLE_BYTES - 4;
    assert_eq!(bytes[at..at + 4], 0x55u32.to_le_bytes());
    assert_eq!(bytes.len(), usize::try_from(CONSTANTS_BUFFER_SIZE).unwrap());
}

#[test]
fn root_table_constants_are_pinned() {
    // PR #4588's NativeRootTables: FirstBank 24, Count 8, Size 256.
    assert_eq!(FIRST_ROOT_BANK, 24);
    assert_eq!(super::root::ROOT_TABLES, 8);
    assert_eq!(ROOT_TABLE_BYTES, 256);
    // The constants buffer: eight tables, then a 256-byte block.
    assert_eq!(DRIVER_CONSTANTS_OFFSET, 2048);
    assert_eq!(CONSTANTS_BUFFER_SIZE, 2304);
}

// ---------------------------------------------------------------------------------------------
// Targets.

#[test]
fn target_formats_are_edens_mapping() {
    // (header value, Eden's PixelFormat) from surface.cpp, written out.
    let cases = [
        (0xc0, PixelFormat::R32G32B32A32Float),
        (0xc3, PixelFormat::R32G32B32A32Float),
        (0xc2, PixelFormat::R32G32B32A32Uint),
        (0xca, PixelFormat::R16G16B16A16Float),
        (0xcf, PixelFormat::B8G8R8A8Unorm),
        (0xe6, PixelFormat::B8G8R8A8Unorm),
        (0xd0, PixelFormat::B8G8R8A8Srgb),
        (0xd5, PixelFormat::A8B8G8R8Unorm),
        (0xf9, PixelFormat::A8B8G8R8Unorm),
        (0xd6, PixelFormat::A8B8G8R8Srgb),
        (0xfa, PixelFormat::A8B8G8R8Srgb),
        (0xd9, PixelFormat::A8B8G8R8Uint),
        (0xe0, PixelFormat::B10G11R11Float),
        (0xe8, PixelFormat::R5G6B5Unorm),
        (0xf8, PixelFormat::A1R5G5B5Unorm),
        (0xea, PixelFormat::R8G8Unorm),
        (0xf3, PixelFormat::R8Unorm),
        (0xf6, PixelFormat::R8Uint),
    ];
    for (value, format) in cases {
        assert_eq!(targets::color_format(value), Ok(Some(format)), "{value:#x}");
    }
    assert_eq!(targets::color_format(0), Ok(None));
    let depth = [
        (0x14, PixelFormat::S8UintD24Unorm),
        (0x16, PixelFormat::D24UnormS8Uint),
        (0x0a, PixelFormat::D32Float),
        (0x13, PixelFormat::D16Unorm),
        (0x17, PixelFormat::S8Uint),
        (0x19, PixelFormat::D32FloatS8Uint),
        (0x15, PixelFormat::X8D24Unorm),
    ];
    for (value, format) in depth {
        assert_eq!(targets::depth_format(value), Ok(format), "{value:#x}");
    }
}

#[test]
fn target_formats_eden_does_not_map_are_refused() {
    // Z1R5G5B5, R32 (the header's names), and A8: Eden's default arm.
    for value in [0xfb, 0xff, 0xf7, 0x47] {
        assert_eq!(
            targets::color_format(value),
            Err(DrawError::TargetFormat { value })
        );
    }
    // V8Z24 and the X16V8 forms.
    for value in [0x18, 0x1d, 0x1e, 0x1f] {
        assert_eq!(
            targets::depth_format(value),
            Err(DrawError::TargetFormat { value })
        );
    }
}

#[test]
fn anti_alias_modes_share_the_texture_header_values() {
    use starquake_layout::SampleLayout;
    // SET_ANTI_ALIAS_SAMPLES and TEXHEAD_BL_MULTI_SAMPLE_COUNT name the same modes alike.
    for (anti_alias, header) in [
        (
            clc797::SET_ANTI_ALIAS_SAMPLES_MODE_1X1,
            tex::TEXHEAD_BL_MULTI_SAMPLE_COUNT_MODE_1X1,
        ),
        (
            clc797::SET_ANTI_ALIAS_SAMPLES_MODE_2X1,
            tex::TEXHEAD_BL_MULTI_SAMPLE_COUNT_MODE_2X1,
        ),
        (
            clc797::SET_ANTI_ALIAS_SAMPLES_MODE_2X2,
            tex::TEXHEAD_BL_MULTI_SAMPLE_COUNT_MODE_2X2,
        ),
        (
            clc797::SET_ANTI_ALIAS_SAMPLES_MODE_4X2,
            tex::TEXHEAD_BL_MULTI_SAMPLE_COUNT_MODE_4X2,
        ),
        (
            clc797::SET_ANTI_ALIAS_SAMPLES_MODE_4X2_D3D,
            tex::TEXHEAD_BL_MULTI_SAMPLE_COUNT_MODE_4X2_D3D,
        ),
        (
            clc797::SET_ANTI_ALIAS_SAMPLES_MODE_2X1_D3D,
            tex::TEXHEAD_BL_MULTI_SAMPLE_COUNT_MODE_2X1_D3D,
        ),
        (
            clc797::SET_ANTI_ALIAS_SAMPLES_MODE_4X4,
            tex::TEXHEAD_BL_MULTI_SAMPLE_COUNT_MODE_4X4,
        ),
    ] {
        assert_eq!(anti_alias, header);
    }
    assert_eq!(targets::sample_layout(0x0), Ok(SampleLayout::S1x1));
    assert_eq!(targets::sample_layout(0x5), Ok(SampleLayout::S2x1D3d));
    assert_eq!(targets::sample_layout(0x9), Ok(SampleLayout::S2x2));
    assert_eq!(targets::sample_layout(0xb), Ok(SampleLayout::S4x2));
    assert_eq!(
        targets::sample_layout(0x7),
        Err(DrawError::SampleMode { mode: 7 })
    );
}

#[test]
fn a_zero_layer_target_is_one_layer() {
    let mut rig = Rig::new(ready(vec![], vec![]));
    rig.basic();
    rig.set(&clc797::SET_COLOR_TARGET_THIRD_DIMENSION, &[0], 0);
    let t = targets::targets(&rig.mirror).unwrap();
    assert_eq!(t.colors[0].unwrap().layers, 1);
    rig.set(&clc797::SET_COLOR_TARGET_THIRD_DIMENSION, &[0], 4);
    assert_eq!(
        targets::targets(&rig.mirror).unwrap().colors[0]
            .unwrap()
            .layers,
        4
    );
}

#[test]
fn targets_follow_the_select_map_and_the_depth_select() {
    let mut rig = Rig::new(ready(vec![], vec![]));
    rig.basic();
    // Output 0 renders colour target 2; output 1 colour target 0.
    rig.set(
        &clc797::SET_COLOR_TARGET_FORMAT,
        &[2],
        clc797::SET_COLOR_TARGET_FORMAT_V_RF32,
    );
    rig.set(&clc797::SET_COLOR_TARGET_WIDTH, &[2], SIZE);
    rig.set(&clc797::SET_COLOR_TARGET_HEIGHT, &[2], SIZE);
    rig.set(
        &clc797::SET_CT_SELECT,
        &[],
        fields(&[
            (clc797::SET_CT_SELECT_TARGET_COUNT, 2),
            (clc797::SET_CT_SELECT_TARGET0, 2),
            (clc797::SET_CT_SELECT_TARGET1, 0),
        ]),
    );
    rig.set(&clc797::SET_ZT_FORMAT, &[], clc797::SET_ZT_FORMAT_V_ZF32);
    rig.set(&clc797::SET_ZT_SIZE_A, &[], SIZE);
    rig.set(&clc797::SET_ZT_SIZE_B, &[], SIZE);
    rig.set(&clc797::SET_ZT_SELECT, &[], 1);
    let t = targets::targets(&rig.mirror).unwrap();
    assert_eq!(
        t.formats.colors[..3],
        [
            Some(PixelFormat::R32Float),
            Some(PixelFormat::A8B8G8R8Unorm),
            None
        ]
    );
    assert_eq!(t.formats.depth_stencil, Some(PixelFormat::D32Float));
    assert_eq!(t.formats.sample_count, 1);
    // A pitch target: its width register is the pitch in bytes.
    rig.set(
        &clc797::SET_COLOR_TARGET_MEMORY,
        &[0],
        fields(&[(
            clc797::SET_COLOR_TARGET_MEMORY_LAYOUT,
            clc797::SET_COLOR_TARGET_MEMORY_LAYOUT_PITCH,
        )]),
    );
    rig.set(&clc797::SET_COLOR_TARGET_WIDTH, &[0], 64);
    let info = targets::targets(&rig.mirror).unwrap().colors[1].unwrap();
    assert_eq!(info.width, 16);
    assert_eq!(
        info.layout,
        crate::back::texture::MemoryLayout::Pitch { pitch_b: 64 }
    );
}

// ---------------------------------------------------------------------------------------------
// Samplers.

#[test]
fn a_sampler_decodes_each_field() {
    let w0 = fields(&[
        (Field::new(2, 0), tex::TEXSAMP0_ADDRESS_U_MIRROR),
        (Field::new(5, 3), tex::TEXSAMP0_ADDRESS_V_BORDER),
        (Field::new(8, 6), tex::TEXSAMP0_ADDRESS_P_MIRROR_ONCE_BORDER),
        (tex::TEXSAMP0_DEPTH_COMPARE, 1),
        (
            tex::TEXSAMP0_DEPTH_COMPARE_FUNC,
            tex::TEXSAMP0_DEPTH_COMPARE_FUNC_ZC_GEQUAL,
        ),
        (
            tex::TEXSAMP0_MAX_ANISOTROPY,
            tex::TEXSAMP0_MAX_ANISOTROPY_ANISO_4_TO_1,
        ),
    ]);
    let w1 = fields(&[
        (tex::TEXSAMP1_MAG_FILTER, tex::TEXSAMP1_MAG_FILTER_MAG_POINT),
        (
            tex::TEXSAMP1_MIN_FILTER,
            tex::TEXSAMP1_MIN_FILTER_MIN_LINEAR,
        ),
        (
            tex::TEXSAMP1_MIP_FILTER,
            tex::TEXSAMP1_MIP_FILTER_MIP_LINEAR,
        ),
        (
            tex::TEXSAMP1_REDUCTION_FILTER,
            tex::TEXSAMP1_REDUCTION_FILTER_RED_MAXIMUM,
        ),
        // -1.5 in signed 5.8: 0x2000 - 0x180.
        (tex::TEXSAMP1_MIP_LOD_BIAS, 0x2000 - 0x180),
    ]);
    let w2 = fields(&[
        (tex::TEXSAMP2_MIN_LOD_CLAMP, 0x80),
        (tex::TEXSAMP2_MAX_LOD_CLAMP, 0x300),
    ]);
    let words = [
        w0,
        w1,
        w2,
        0,
        0.25f32.to_bits(),
        0.5f32.to_bits(),
        0.75f32.to_bits(),
        1.0f32.to_bits(),
    ];
    let desc = sampler::decode(&words, 16).unwrap();
    assert_eq!(desc.mag_filter, FilterMode::Nearest);
    assert_eq!(desc.min_filter, FilterMode::Linear);
    assert_eq!(desc.mip_filter, FilterMode::Linear);
    assert_eq!(
        desc.address,
        [
            AddressMode::MirrorRepeat,
            AddressMode::ClampToBorder,
            AddressMode::MirrorClampToEdge
        ]
    );
    assert_eq!(desc.compare, Some(CompareFunction::GreaterEqual));
    assert_eq!(desc.max_anisotropy, 4);
    assert_eq!(desc.reduction, ReductionMode::Max);
    assert_eq!(desc.lod_bias, -1.5);
    assert_eq!((desc.min_lod, desc.max_lod), (0.5, 3.0));
    assert_eq!(
        desc.border_color,
        BorderColor::Custom([0.25, 0.5, 0.75, 1.0])
    );
}

#[test]
fn clamp_ogl_follows_the_magnification_filter_and_mip_none_clamps_the_levels() {
    let w0 = fields(&[(Field::new(2, 0), tex::TEXSAMP0_ADDRESS_U_CLAMP_OGL)]);
    let w1 = |mag| {
        fields(&[
            (tex::TEXSAMP1_MAG_FILTER, mag),
            (tex::TEXSAMP1_MIN_FILTER, tex::TEXSAMP1_MIN_FILTER_MIN_POINT),
            (tex::TEXSAMP1_MIP_FILTER, tex::TEXSAMP1_MIP_FILTER_MIP_NONE),
        ])
    };
    let w2 = fields(&[
        (tex::TEXSAMP2_MIN_LOD_CLAMP, 0x100),
        (tex::TEXSAMP2_MAX_LOD_CLAMP, 0x500),
    ]);
    let linear = sampler::decode(
        &[
            w0,
            w1(tex::TEXSAMP1_MAG_FILTER_MAG_LINEAR),
            w2,
            0,
            0,
            0,
            0,
            0,
        ],
        16,
    )
    .unwrap();
    assert_eq!(linear.address[0], AddressMode::ClampToBorder);
    let point = sampler::decode(
        &[
            w0,
            w1(tex::TEXSAMP1_MAG_FILTER_MAG_POINT),
            w2,
            0,
            0,
            0,
            0,
            0,
        ],
        16,
    )
    .unwrap();
    assert_eq!(point.address[0], AddressMode::ClampToEdge);
    // MIP_NONE: nearest, levels 0 to 0.25 whatever the clamps say (Eden).
    assert_eq!(point.mip_filter, FilterMode::Nearest);
    assert_eq!((point.min_lod, point.max_lod), (0.0, 0.25));
}

#[test]
fn anisotropy_follows_the_header_ratios() {
    let ratios = [1, 2, 4, 6, 8, 10, 12, 16];
    for (value, ratio) in (0u32..8).zip(ratios) {
        let mut words = linear_sampler();
        words[0] = tex::TEXSAMP0_MAX_ANISOTROPY
            .insert(words[0], value)
            .unwrap();
        assert_eq!(sampler::decode(&words, 16).unwrap().max_anisotropy, ratio);
        // Clamped to the device's limit, exactly at it and below.
        assert_eq!(
            sampler::decode(&words, 8).unwrap().max_anisotropy,
            ratio.min(8)
        );
    }
}

#[test]
fn undefined_sampler_fields_are_refused() {
    let mut words = linear_sampler();
    words[1] = tex::TEXSAMP1_MAG_FILTER.insert(words[1], 0).unwrap();
    assert_eq!(
        sampler::decode(&words, 16),
        Err(DrawError::Sampler {
            field: "MAG_FILTER",
            value: 0
        })
    );
    words[1] = tex::TEXSAMP1_MAG_FILTER
        .insert(words[1], tex::TEXSAMP1_MAG_FILTER_VCAA_4_TAP)
        .unwrap();
    assert_eq!(
        sampler::decode(&words, 16),
        Err(DrawError::Sampler {
            field: "MAG_FILTER",
            value: 3
        })
    );
    let mut words = linear_sampler();
    words[1] = tex::TEXSAMP1_MIN_FILTER.insert(words[1], 0).unwrap();
    assert_eq!(
        sampler::decode(&words, 16),
        Err(DrawError::Sampler {
            field: "MIN_FILTER",
            value: 0
        })
    );
    let mut words = linear_sampler();
    words[1] = tex::TEXSAMP1_MIN_FILTER
        .insert(words[1], tex::TEXSAMP1_MIN_FILTER_MIN_ANISO)
        .unwrap();
    assert_eq!(
        sampler::decode(&words, 16).unwrap().min_filter,
        FilterMode::Linear
    );
    let mut words = linear_sampler();
    words[1] = tex::TEXSAMP1_MIP_FILTER.insert(words[1], 0).unwrap();
    assert_eq!(
        sampler::decode(&words, 16),
        Err(DrawError::Sampler {
            field: "MIP_FILTER",
            value: 0
        })
    );
    let mut words = linear_sampler();
    words[1] = tex::TEXSAMP1_REDUCTION_FILTER.insert(words[1], 3).unwrap();
    assert_eq!(
        sampler::decode(&words, 16),
        Err(DrawError::Sampler {
            field: "REDUCTION_FILTER",
            value: 3
        })
    );
}

#[test]
fn samplers_are_made_once_per_entry_up_to_the_cap() {
    let backend: Arc<dyn Backend> = Arc::new(RecordingBackend::new(
        caps(true, Features::ALL, 16),
        1 << 40,
    ));
    let mut samplers = sampler::Samplers::with_limit(backend, 2);
    let a = linear_sampler();
    let mut b = a;
    b[7] = 0.5f32.to_bits();
    let mut c = a;
    c[6] = 0.5f32.to_bits();
    let id = samplers.get(&a).unwrap();
    assert_eq!(samplers.get(&a), Ok(id));
    assert_ne!(samplers.get(&b).unwrap(), id);
    assert_eq!(samplers.len(), 2);
    // The cap refuses a third, and an entry already held still answers.
    assert_eq!(samplers.get(&c), Err(DrawError::SamplersFull { limit: 2 }));
    assert_eq!(samplers.get(&a), Ok(id));
    assert_eq!(samplers.len(), 2);
}

#[test]
fn host_shaders_are_made_once_per_stamp_up_to_the_cap() {
    let backend: Arc<dyn Backend> = Arc::new(RecordingBackend::new(
        caps(true, Features::ALL, 16),
        1 << 40,
    ));
    let mut shaders = shaders::HostShaders::with_limit(backend, 1);
    let vertex = host_shader(GuestStage::Vertex);
    let info = StageInfo {
        stamp: VERTEX_STAMP,
        resources: vec![],
    };
    let id = shaders.get(&vertex, &info).unwrap();
    assert_eq!(shaders.get(&vertex, &info), Ok(id));
    let other = StageInfo {
        stamp: PIXEL_STAMP,
        resources: vec![],
    };
    assert_eq!(
        shaders.get(&vertex, &other),
        Err(DrawError::ShadersFull { limit: 1 })
    );
    assert_eq!(shaders.len(), 1);
}

#[test]
fn the_shader_description_carries_the_module_slots_and_resources() {
    let vertex = host_shader(GuestStage::Vertex);
    let desc = shaders::shader_desc(&vertex, &[uniform(2, 3, 1, 0x10)]);
    assert_eq!(desc.stage, starquake_render::ShaderStage::Vertex);
    assert_eq!(desc.entry, vertex.entry);
    assert!(matches!(
        desc.source,
        starquake_render::ShaderSource::Naga { .. }
    ));
    assert_eq!(
        desc.interface.resources,
        [starquake_render::ResourceBinding {
            group: 2,
            binding: 3,
            kind: ResourceKind::UniformBuffer,
            count: None,
        }]
    );
    // Built-ins are not interface variables: an exit-only vertex shader writes only position.
    let locations = vertex
        .outputs
        .iter()
        .filter(|s| matches!(s.binding, starquake_shader::host::SlotBinding::Location(_)))
        .count();
    assert_eq!(desc.interface.outputs.len(), locations);
}

/// A shader with a binding array and a subgroup requirement keeps both through the conversion
/// (render-backend, "Shader emission": the interface "carries the shader's requirements and each
/// binding's array count"; ruling 20 of 2026-11-19), and one without either keeps neither: each
/// row's count is its own, in table order.
#[test]
fn the_conversion_keeps_binding_array_counts_and_requirements() {
    let four = std::num::NonZeroU32::new(4);
    let resources = [
        ResourceUse {
            count: four,
            ..texture_use(1, 5)
        },
        uniform(2, 3, 1, 0x10),
    ];
    for subgroup in [true, false] {
        let mut shader = host_shader(GuestStage::Pixel);
        shader.requirements = starquake_shader::host::Requirements {
            subgroup,
            ..starquake_shader::host::Requirements::default()
        };
        let desc = shaders::shader_desc(&shader, &resources);
        let counts: Vec<_> = desc
            .interface
            .resources
            .iter()
            .map(|r| (r.group, r.binding, r.count))
            .collect();
        assert_eq!(counts, [(1, 5, four), (2, 3, None)], "subgroup {subgroup}");
        assert_eq!(
            desc.interface.requirements,
            starquake_render::ShaderRequirements {
                subgroup,
                ..starquake_render::ShaderRequirements::default()
            },
        );
    }
}

/// The requirements map field by field: each field set alone reaches the field it mirrors and
/// no other, so two swapped fields fail; none set and all set map to the same.
#[test]
fn requirements_map_field_by_field() {
    use starquake_render::ShaderRequirements;
    use starquake_shader::host::Requirements;
    let none = Requirements::default();
    let cases = [
        (
            Requirements {
                subgroup: true,
                ..none
            },
            ShaderRequirements {
                subgroup: true,
                ..ShaderRequirements::default()
            },
        ),
        (
            Requirements { f16: true, ..none },
            ShaderRequirements {
                f16: true,
                ..ShaderRequirements::default()
            },
        ),
        (
            Requirements { f64: true, ..none },
            ShaderRequirements {
                f64: true,
                ..ShaderRequirements::default()
            },
        ),
        (
            Requirements { i64: true, ..none },
            ShaderRequirements {
                i64: true,
                ..ShaderRequirements::default()
            },
        ),
        (
            Requirements {
                i64_atomics: true,
                ..none
            },
            ShaderRequirements {
                i64_atomics: true,
                ..ShaderRequirements::default()
            },
        ),
        (none, ShaderRequirements::default()),
        (
            Requirements {
                subgroup: true,
                f16: true,
                f64: true,
                i64: true,
                i64_atomics: true,
            },
            ShaderRequirements {
                subgroup: true,
                f16: true,
                f64: true,
                i64: true,
                i64_atomics: true,
            },
        ),
    ];
    for (from, to) in cases {
        assert_eq!(shaders::requirements(from), to, "{from:?}");
    }
}

#[test]
fn draws_destroy_the_constants_buffer_after_the_last_fence() {
    let mut rig = Rig::new(ready(
        vec![],
        vec![driver_use(0, 0), texture_use(0, 1), sampler_use(0, 2)],
    ));
    rig.basic();
    rig.draw();
    rig.submit();
    let Rig { draws, backend, .. } = rig;
    let before = backend.snapshot();
    drop(draws);
    let snapshot = backend.snapshot();
    let destroys: Vec<&str> = snapshot
        .strip_prefix(before.as_str())
        .unwrap()
        .lines()
        .filter(|line| line.contains("destroy"))
        .collect();
    // The constants buffer, the one sampler and the two shaders, each once.
    assert_eq!(destroys.len(), 4, "{destroys:?}");
    assert_eq!(backend.refused_destroys(), 0);
}

// ---------------------------------------------------------------------------------------------
// Errors.

#[test]
fn error_messages_are_plain_words() {
    let cases: [(DrawError, &str); 8] = [
        (
            DrawError::RootBank {
                stage: GuestStage::Pixel,
                bank: 32,
            },
            "the pixel stage reads constant buffer 32, which no root table backs (banks 24 to 31)",
        ),
        (
            DrawError::NotFinite { what: "viewport" },
            "the draw's viewport is not finite",
        ),
        (
            DrawError::Topology { value: 0x1001 },
            "primitive topology 0x1001 is not one the class header defines for a draw",
        ),
        (
            DrawError::AttributeStream {
                attribute: 3,
                stream: 2,
            },
            "vertex attribute 3 reads vertex stream 2, which is not enabled",
        ),
        (
            DrawError::UnboundConstantBuffer {
                stage: GuestStage::Pixel,
                bank: 11,
            },
            "the pixel stage reads constant buffer 11, which is not bound",
        ),
        (
            DrawError::PoolIndex {
                pool: "sampler",
                index: 9,
                maximum: 8,
            },
            "sampler index 9 is past the pool's maximum, 8",
        ),
        (
            DrawError::SamplersFull { limit: 2 },
            "the sampler table is full (2 samplers)",
        ),
        (
            DrawError::BadSet("has no vertex shader"),
            "the shader set has no vertex shader",
        ),
    ];
    for (error, text) in cases {
        assert_eq!(error.to_string(), text);
    }
}

// ---------------------------------------------------------------------------------------------
// Fixed-function values.

#[test]
fn viewports_and_scissors_flip_with_the_window_origin() {
    let mut rig = Rig::new(ready(vec![], vec![]));
    rig.basic();
    rig.set(&clc797::SET_SCISSOR_ENABLE, &[0], 1);
    rig.set(
        &clc797::SET_SCISSOR_HORIZONTAL,
        &[0],
        fields(&[
            (clc797::SET_SCISSOR_HORIZONTAL_XMIN, 1),
            (clc797::SET_SCISSOR_HORIZONTAL_XMAX, 5),
        ]),
    );
    rig.set(
        &clc797::SET_SCISSOR_VERTICAL,
        &[0],
        fields(&[
            (clc797::SET_SCISSOR_VERTICAL_YMIN, 2),
            (clc797::SET_SCISSOR_VERTICAL_YMAX, 3),
        ]),
    );
    rig.draw();
    let draw = rig.last_draw();
    let viewport = draw.dynamic.viewports[0];
    // Offset 4, scale 4: x and y 0, width and height 8; depth 0.5 - 0.5 to 0.5 + 0.5 under
    // the -1..1 clip range a register never written gives.
    assert_eq!(
        (viewport.x, viewport.y, viewport.width, viewport.height),
        (0.0, 0.0, 8.0, 8.0)
    );
    assert_eq!((viewport.min_depth, viewport.max_depth), (0.0, 1.0));
    let scissor = draw.dynamic.scissors[0];
    assert_eq!(
        (scissor.x, scissor.y, scissor.width, scissor.height),
        (1, 2, 4, 1)
    );
    // A disabled scissor is the whole target, the largest i32 in each extent (Eden).
    let whole = draw.dynamic.scissors[1];
    assert_eq!((whole.width, whole.height), (0x7fff_ffff, 0x7fff_ffff));
    // Only the window origin changes: both parts follow it. The clip height is 8.
    rig.set(
        &clc797::SET_WINDOW_ORIGIN,
        &[],
        fields(&[(
            clc797::SET_WINDOW_ORIGIN_MODE,
            clc797::SET_WINDOW_ORIGIN_MODE_LOWER_LEFT,
        )]),
    );
    rig.draw();
    let draw = rig.last_draw();
    let viewport = draw.dynamic.viewports[0];
    assert_eq!((viewport.y, viewport.height), (8.0, -8.0));
    let scissor = draw.dynamic.scissors[0];
    assert_eq!((scissor.y, scissor.height), (5, 1));
    // A negative y swizzle then flips by the viewport's own height.
    rig.set(
        &clc797::SET_VIEWPORT_COORDINATE_SWIZZLE,
        &[0],
        fields(&[(
            clc797::SET_VIEWPORT_COORDINATE_SWIZZLE_Y,
            clc797::SET_VIEWPORT_COORDINATE_SWIZZLE_Y_NEG_Y,
        )]),
    );
    rig.draw();
    let viewport = rig.last_draw().dynamic.viewports[0];
    assert_eq!((viewport.y, viewport.height), (0.0, 8.0));
}

#[test]
fn depth_bias_applies_only_with_the_topologys_offset_enable() {
    let mut rig = Rig::new(ready(vec![], vec![]));
    rig.basic();
    rig.set(&clc797::SET_DEPTH_BIAS, &[], 2.0f32.to_bits());
    rig.set(&clc797::SET_SLOPE_SCALE_DEPTH_BIAS, &[], 3.0f32.to_bits());
    rig.set(&clc797::SET_POLY_OFFSET_LINE, &[], 1);
    // Triangles: the fill enable is off, so no bias.
    rig.draw();
    assert_eq!(rig.last_draw().dynamic.depth_bias.constant, 0.0);
    // Lines: the line enable is on. The constant is SET_DEPTH_BIAS / 2, as Eden's
    // UpdateDepthBias passes `regs.depth_bias / 2.0f`; the slope scale is unchanged.
    rig.draw_arrays(clc797::BEGIN_OP_LINES, 0, 2).unwrap();
    let bias = rig.last_draw().dynamic.depth_bias;
    assert_eq!((bias.constant, bias.slope_scale), (1.0, 3.0));
    assert_eq!(fixed::DEPTH_BIAS_UNITS_PER_HOST_UNIT, 2.0);
    // Points: off again.
    rig.draw_arrays(clc797::BEGIN_OP_POINTS, 0, 1).unwrap();
    assert_eq!(rig.last_draw().dynamic.depth_bias.constant, 0.0);
}

#[test]
fn a_zero_layer_depth_target_is_one_layer() {
    let mut rig = Rig::new(ready(vec![], vec![]));
    rig.basic();
    rig.set(&clc797::SET_ZT_FORMAT, &[], clc797::SET_ZT_FORMAT_V_Z16);
    rig.set(&clc797::SET_ZT_SELECT, &[], 1);
    rig.set(&clc797::SET_ZT_SIZE_C, &[], 0);
    let depth = targets::targets(&rig.mirror).unwrap().depth.unwrap();
    assert_eq!(depth.layers, 1);
    rig.set(
        &clc797::SET_ZT_SIZE_C,
        &[],
        fields(&[(clc797::SET_ZT_SIZE_C_THIRD_DIMENSION, 3)]),
    );
    assert_eq!(
        targets::targets(&rig.mirror).unwrap().depth.unwrap().layers,
        3
    );
    // ARRAY_SIZE_IS_ONE wins over the third dimension.
    rig.set(
        &clc797::SET_ZT_SIZE_C,
        &[],
        fields(&[
            (clc797::SET_ZT_SIZE_C_THIRD_DIMENSION, 3),
            (
                clc797::SET_ZT_SIZE_C_CONTROL,
                clc797::SET_ZT_SIZE_C_CONTROL_ARRAY_SIZE_IS_ONE,
            ),
        ]),
    );
    assert_eq!(
        targets::targets(&rig.mirror).unwrap().depth.unwrap().layers,
        1
    );
}

#[test]
fn bank_24_answers_from_the_first_root_table() {
    let mut rig = Rig::new(ready(vec![], vec![]));
    rig.basic();
    // A constant buffer bound to slot 24 too: the root table wins (PR #4588's `IsBank`).
    rig.write_words(UNIFORMS, &[0xdead_beef]);
    rig.bind(4, FIRST_ROOT_BANK, UNIFORMS, 0x10);
    rig.set(&clc797::SET_ROOT_TABLE_SELECTOR, &[], 0);
    rig.mirror
        .run(clc797::LOAD_ROOT_TABLE.method(), &[0x2424])
        .unwrap();
    rig.draws.load_root_table(&rig.mirror, &[0x2424]).unwrap();
    let answers = StateAnswers {
        mirror: &rig.mirror,
        roots: rig.draws.root_tables(),
        memory: &rig.mem,
    };
    assert_eq!(
        answers.cbuf_word(GuestStage::Pixel, FIRST_ROOT_BANK, 0),
        Some(0x2424)
    );
    // Bank 23 is a bind group's.
    rig.bind(4, FIRST_ROOT_BANK - 1, UNIFORMS, 0x10);
    let answers = StateAnswers {
        mirror: &rig.mirror,
        roots: rig.draws.root_tables(),
        memory: &rig.mem,
    };
    assert_eq!(
        answers.cbuf_word(GuestStage::Pixel, FIRST_ROOT_BANK - 1, 0),
        Some(0xdead_beef)
    );
}

// ---------------------------------------------------------------------------------------------
// Review fixes: inline draws, root banks, the surface-clip viewports, the rasterizer values,
// finite values and the host's viewport count.

#[test]
fn an_inline_index_draw_draws_nothing_until_the_next_begin() {
    for inline in call::INLINE_METHODS {
        let mut rig = Rig::new(ready(vec![], vec![]));
        rig.basic();
        // Stale vertex-array registers an END in the general mode would draw.
        rig.set(&clc797::SET_VERTEX_ARRAY_START, &[], 1);
        rig.set(&clc797::DRAW_VERTEX_ARRAY, &[], 3);
        let unhandled = Ok(Some(Outcome::Skipped(Skip::Unhandled(inline.name))));
        assert_eq!(
            rig.put(&clc797::BEGIN, &[], clc797::BEGIN_OP_TRIANGLES),
            Ok(None)
        );
        assert_eq!(rig.put(inline, &[], 0x0001_0000), unhandled);
        assert_eq!(rig.put(&clc797::END, &[], 0), unhandled, "{}", inline.name);
        // A flush draws nothing in the inline mode (Eden's DrawDeferred returns), and a second
        // END is still the inline draw.
        assert_eq!(rig.flush(), Ok(None));
        assert_eq!(rig.put(&clc797::END, &[], 0), unhandled);
        assert_eq!(rig.draw_commands(), [], "{}", inline.name);
        // BEGIN with INSTANCE_ID UNCHANGED keeps the mode.
        let unchanged = fields(&[
            (clc797::BEGIN_OP, clc797::BEGIN_OP_TRIANGLES),
            (
                clc797::BEGIN_INSTANCE_ID,
                clc797::BEGIN_INSTANCE_ID_UNCHANGED,
            ),
        ]);
        assert_eq!(rig.put(&clc797::BEGIN, &[], unchanged), Ok(None));
        assert_eq!(rig.put(&clc797::END, &[], 0), unhandled);
        // BEGIN with INSTANCE_ID FIRST leaves it: the next END draws the vertex array.
        assert_eq!(rig.draw(), Outcome::Drawn);
        assert_eq!(rig.draw_commands().len(), 1);
        assert_eq!(rig.draws.counters().unhandled, 4);
    }
    // BEGIN with INSTANCE_ID SUBSEQUENT leaves it too, for the instance mode.
    let mut rig = Rig::new(ready(vec![], vec![]));
    rig.basic();
    rig.put(&clc797::DRAW_INLINE_INDEX, &[], 0).unwrap();
    let subsequent = fields(&[
        (clc797::BEGIN_OP, clc797::BEGIN_OP_TRIANGLES),
        (
            clc797::BEGIN_INSTANCE_ID,
            clc797::BEGIN_INSTANCE_ID_SUBSEQUENT,
        ),
    ]);
    assert_eq!(rig.put(&clc797::BEGIN, &[], subsequent), Ok(None));
    assert_eq!(rig.put(&clc797::END, &[], 0), Ok(None));
    assert_eq!(rig.flush(), Ok(Some(Outcome::Drawn)));
    // DRAW_ZERO_INDEX and DRAW_AUTO do not enter the inline mode.
    for method in [&clc797::DRAW_ZERO_INDEX, &clc797::DRAW_AUTO] {
        let mut rig = Rig::new(ready(vec![], vec![]));
        rig.basic();
        rig.set(&clc797::SET_VERTEX_ARRAY_START, &[], 0);
        rig.set(&clc797::DRAW_VERTEX_ARRAY, &[], 3);
        assert_eq!(
            rig.put(&clc797::BEGIN, &[], clc797::BEGIN_OP_TRIANGLES),
            Ok(None)
        );
        rig.put(method, &[], 0).unwrap();
        assert_eq!(
            rig.put(&clc797::END, &[], 0),
            Ok(Some(Outcome::Drawn)),
            "{}",
            method.name
        );
    }
}

#[test]
fn root_banks_past_the_last_table_are_refused() {
    let last = super::root::LAST_ROOT_BANK;
    // PR #4588's IsBank: banks 24 to 31, written out.
    assert_eq!(last, 31);
    let mut rig = Rig::new(ready(vec![], vec![uniform(0, 0, last, 0)]));
    rig.basic();
    assert_eq!(rig.draw(), Outcome::Drawn);
    let BindingResource::Buffer(slice) = rig.last_draw().bindings[0].resource else {
        unreachable!("a buffer");
    };
    let table = u64::from(last - FIRST_ROOT_BANK);
    let bytes = u64::try_from(ROOT_TABLE_BYTES).unwrap();
    assert_eq!((slice.offset, slice.size), (table * bytes, bytes));
    assert!(slice.offset + slice.size <= DRIVER_CONSTANTS_OFFSET);
    for bank in [last + 1, 40, u8::MAX] {
        let mut rig = Rig::new(ready(vec![], vec![uniform(0, 0, bank, 0)]));
        rig.basic();
        assert_eq!(
            rig.draw_arrays(clc797::BEGIN_OP_TRIANGLES, 0, 3),
            Err(DrawError::RootBank {
                stage: GuestStage::Pixel,
                bank
            })
        );
        assert_eq!(rig.draw_commands(), []);
    }
}

#[test]
fn a_bound_constant_buffer_of_size_zero_is_unbound() {
    let mut rig = Rig::new(ready(vec![], vec![uniform(0, 0, 11, 0x10)]));
    rig.basic();
    // VALID set, size 0.
    rig.bind(4, 11, UNIFORMS, 0);
    assert_eq!(
        rig.draw_arrays(clc797::BEGIN_OP_TRIANGLES, 0, 3),
        Err(DrawError::UnboundConstantBuffer {
            stage: GuestStage::Pixel,
            bank: 11
        })
    );
    // One byte is bound.
    rig.bind(4, 11, UNIFORMS, 1);
    assert_eq!(rig.draw(), Outcome::Drawn);
}

/// Writes the surface clip `(x, y, width, height)`.
fn surface_clip(rig: &mut Rig, x: u32, y: u32, width: u32, height: u32) {
    rig.set(
        &clc797::SET_SURFACE_CLIP_HORIZONTAL,
        &[],
        fields(&[
            (clc797::SET_SURFACE_CLIP_HORIZONTAL_X, x),
            (clc797::SET_SURFACE_CLIP_HORIZONTAL_WIDTH, width),
        ]),
    );
    rig.set(
        &clc797::SET_SURFACE_CLIP_VERTICAL,
        &[],
        fields(&[
            (clc797::SET_SURFACE_CLIP_VERTICAL_Y, y),
            (clc797::SET_SURFACE_CLIP_VERTICAL_HEIGHT, height),
        ]),
    );
}

fn lower_left(rig: &mut Rig) {
    rig.set(
        &clc797::SET_WINDOW_ORIGIN,
        &[],
        fields(&[(
            clc797::SET_WINDOW_ORIGIN_MODE,
            clc797::SET_WINDOW_ORIGIN_MODE_LOWER_LEFT,
        )]),
    );
}

#[test]
fn without_scale_offset_viewports_and_scissors_are_the_surface_clip() {
    use starquake_render::command::{Rect, Viewport};
    let mut rig = Rig::new(ready(vec![], vec![]));
    rig.basic();
    // The reset value is off (Eden's viewport_scale_offset_enabled is 0 at reset).
    let fresh = Mirror::new(CHANNEL, ClassId(clc797::CLASS));
    assert_eq!(
        fresh
            .reg(&clc797::SET_VIEWPORT_SCALE_OFFSET, &[])
            .unwrap()
            .field(clc797::SET_VIEWPORT_SCALE_OFFSET_ENABLE),
        Ok(clc797::SET_VIEWPORT_SCALE_OFFSET_ENABLE_FALSE)
    );
    rig.set(
        &clc797::SET_VIEWPORT_SCALE_OFFSET,
        &[],
        clc797::SET_VIEWPORT_SCALE_OFFSET_ENABLE_FALSE,
    );
    // A scissor that is on is ignored too.
    rig.set(&clc797::SET_SCISSOR_ENABLE, &[0], 1);
    surface_clip(&mut rig, 2, 3, 10, 6);
    rig.draw();
    let draw = rig.last_draw();
    let viewport = Viewport {
        x: 2.0,
        y: 3.0,
        width: 10.0,
        height: 6.0,
        min_depth: 0.0,
        max_depth: 1.0,
    };
    let rect = Rect {
        x: 2,
        y: 3,
        width: 10,
        height: 6,
    };
    let slots = usize::try_from(crate::back::state::VIEWPORTS).unwrap();
    assert_eq!(draw.dynamic.viewports, vec![viewport; slots]);
    assert_eq!(draw.dynamic.scissors, vec![rect; slots]);
    // Lower left: the viewport flips by its height; the scissor's y is 6 - (3 + 6), clamped
    // at 0 (Eden's u32 wraps: a listed divergence).
    lower_left(&mut rig);
    rig.draw();
    let draw = rig.last_draw();
    assert_eq!(
        draw.dynamic.viewports[0],
        Viewport {
            y: 9.0,
            height: -6.0,
            ..viewport
        }
    );
    assert_eq!(draw.dynamic.scissors[0], Rect { y: 0, ..rect });
    // A zero clip is 1 by 1.
    surface_clip(&mut rig, 0, 0, 0, 0);
    rig.draw();
    let draw = rig.last_draw();
    assert_eq!(
        draw.dynamic.viewports[0],
        Viewport {
            x: 0.0,
            y: 1.0,
            width: 1.0,
            height: -1.0,
            min_depth: 0.0,
            max_depth: 1.0,
        }
    );
    assert_eq!(
        draw.dynamic.scissors[0],
        Rect {
            x: 0,
            y: 0,
            width: 1,
            height: 1
        }
    );
    // Turning the scale and offset on again reads viewport 0's registers.
    rig.set(
        &clc797::SET_VIEWPORT_SCALE_OFFSET,
        &[],
        clc797::SET_VIEWPORT_SCALE_OFFSET_ENABLE_TRUE,
    );
    surface_clip(&mut rig, 0, 0, SIZE, SIZE);
    rig.draw();
    let viewport = rig.last_draw().dynamic.viewports[0];
    assert_eq!((viewport.width, viewport.height), (8.0, -8.0));
}

#[test]
fn the_lower_left_flip_of_a_negative_scale_turns_at_the_clip_height() {
    // Eden's GetViewportState: `src.scale_y < 0.0f && y >= clip_height` takes y = clip - y.
    let mut rig = Rig::new(ready(vec![], vec![]));
    rig.basic();
    lower_left(&mut rig);
    let clip = f32::from(u16::try_from(SIZE).unwrap());
    rig.set(&clc797::SET_VIEWPORT_SCALE_Y, &[0], (-2.0f32).to_bits());
    // y = offset - scale = clip exactly: flipped to clip - y = 0, height -(-4) = 4.
    rig.set(&clc797::SET_VIEWPORT_OFFSET_Y, &[0], (clip - 2.0).to_bits());
    rig.draw();
    let viewport = rig.last_draw().dynamic.viewports[0];
    assert_eq!((viewport.y, viewport.height), (0.0, 4.0));
    // Half a pixel below: y + clip.
    rig.set(&clc797::SET_VIEWPORT_OFFSET_Y, &[0], (clip - 2.5).to_bits());
    rig.draw();
    let viewport = rig.last_draw().dynamic.viewports[0];
    assert_eq!((viewport.y, viewport.height), (clip - 0.5 + clip, 4.0));
    // A positive scale with y past the clip is not flipped.
    rig.set(&clc797::SET_VIEWPORT_SCALE_Y, &[0], 2.0f32.to_bits());
    rig.set(&clc797::SET_VIEWPORT_OFFSET_Y, &[0], (clip + 2.0).to_bits());
    rig.draw();
    let viewport = rig.last_draw().dynamic.viewports[0];
    assert_eq!((viewport.y, viewport.height), (clip + clip, -4.0));
}

/// The rasterizer part derived from the rig's state.
fn raster_part(rig: &Rig) -> fixed::RasterPart {
    let mut parts = fixed::Parts::new();
    *parts
        .refresh(&rig.mirror, rig.mirror.dirty())
        .unwrap()
        .raster
}

#[test]
fn the_front_face_flips_with_flip_y() {
    use starquake_render::pipeline::FrontFace;
    let mut rig = Rig::new(ready(vec![], vec![]));
    rig.basic();
    let origin = |flip| {
        fields(&[
            (
                clc797::SET_WINDOW_ORIGIN_MODE,
                clc797::SET_WINDOW_ORIGIN_MODE_UPPER_LEFT,
            ),
            (clc797::SET_WINDOW_ORIGIN_FLIP_Y, flip),
        ])
    };
    for (face, flip, expected) in [
        (
            clc797::OGL_SET_FRONT_FACE_V_CCW,
            clc797::SET_WINDOW_ORIGIN_FLIP_Y_FALSE,
            FrontFace::Ccw,
        ),
        (
            clc797::OGL_SET_FRONT_FACE_V_CCW,
            clc797::SET_WINDOW_ORIGIN_FLIP_Y_TRUE,
            FrontFace::Cw,
        ),
        (
            clc797::OGL_SET_FRONT_FACE_V_CW,
            clc797::SET_WINDOW_ORIGIN_FLIP_Y_FALSE,
            FrontFace::Cw,
        ),
        (
            clc797::OGL_SET_FRONT_FACE_V_CW,
            clc797::SET_WINDOW_ORIGIN_FLIP_Y_TRUE,
            FrontFace::Ccw,
        ),
    ] {
        rig.set(&clc797::OGL_SET_FRONT_FACE, &[], face);
        rig.set(&clc797::SET_WINDOW_ORIGIN, &[], origin(flip));
        assert_eq!(raster_part(&rig).front_face, expected, "{face:#x} {flip}");
        // Through a draw: SET_WINDOW_ORIGIN re-derives the rasterizer part.
        assert_eq!(rig.draw(), Outcome::Drawn);
    }
}

#[test]
fn the_line_width_follows_the_anti_aliased_line_enable() {
    let mut rig = Rig::new(ready(vec![], vec![]));
    rig.basic();
    rig.set(&clc797::SET_LINE_WIDTH_FLOAT, &[], 3.0f32.to_bits());
    rig.set(&clc797::SET_ALIASED_LINE_WIDTH_FLOAT, &[], 5.0f32.to_bits());
    // Off (the reset value): the aliased width.
    rig.draw();
    assert_eq!(rig.last_draw().dynamic.line_width, 5.0);
    rig.set(
        &clc797::SET_ANTI_ALIASED_LINE,
        &[],
        clc797::SET_ANTI_ALIASED_LINE_ENABLE_TRUE,
    );
    rig.draw();
    assert_eq!(rig.last_draw().dynamic.line_width, 3.0);
    assert_eq!(raster_part(&rig).line_width, 3.0);
    rig.set(
        &clc797::SET_ANTI_ALIASED_LINE,
        &[],
        clc797::SET_ANTI_ALIASED_LINE_ENABLE_FALSE,
    );
    rig.draw();
    assert_eq!(rig.last_draw().dynamic.line_width, 5.0);
}

#[test]
fn the_topology_override_none_keeps_the_begin_topology() {
    // Eden's PrimitiveTopologyOverride::None, written out.
    assert_eq!(fixed::TOPOLOGY_OVERRIDE_NONE, 0);
    let mut rig = Rig::new(ready(vec![], vec![]));
    rig.basic();
    rig.set(
        &clc797::SET_PRIMITIVE_TOPOLOGY_CONTROL,
        &[],
        clc797::SET_PRIMITIVE_TOPOLOGY_CONTROL_OVERRIDE_USE_SEPARATE_TOPOLOGY_STATE,
    );
    rig.set(&clc797::SET_PRIMITIVE_TOPOLOGY, &[], 0);
    for (begin, topology) in [
        (clc797::BEGIN_OP_QUADS, Topology::Quads),
        (clc797::BEGIN_OP_LINE_LOOP, Topology::LineLoop),
    ] {
        assert_eq!(fixed::effective_topology(&rig.mirror, begin), Ok(topology));
    }
    assert_eq!(
        rig.draw_arrays(clc797::BEGIN_OP_TRIANGLES, 0, 3),
        Ok(Some(Outcome::Drawn))
    );
    // One past it is points.
    rig.set(&clc797::SET_PRIMITIVE_TOPOLOGY, &[], 1);
    assert_eq!(
        fixed::effective_topology(&rig.mirror, clc797::BEGIN_OP_QUADS),
        Ok(Topology::Points)
    );
}

#[test]
fn a_draw_with_a_value_that_is_not_finite_is_refused_and_the_list_still_submits() {
    let cases: [(&Method, &[u32], u32, &str); 6] = [
        (
            &clc797::SET_VIEWPORT_SCALE_X,
            &[0],
            f32::NAN.to_bits(),
            "viewport",
        ),
        // f32::MAX * 2 is an infinite height.
        (
            &clc797::SET_VIEWPORT_SCALE_Y,
            &[0],
            f32::MAX.to_bits(),
            "viewport",
        ),
        (
            &clc797::SET_BLEND_CONST_RED,
            &[],
            f32::INFINITY.to_bits(),
            "blend constant",
        ),
        (
            &clc797::SET_DEPTH_BIAS,
            &[],
            f32::NAN.to_bits(),
            "depth bias",
        ),
        (
            &clc797::SET_ALIASED_LINE_WIDTH_FLOAT,
            &[],
            f32::NEG_INFINITY.to_bits(),
            "line width",
        ),
        (
            &clc797::SET_SLOPE_SCALE_DEPTH_BIAS,
            &[],
            f32::NAN.to_bits(),
            "depth bias",
        ),
    ];
    for (method, indices, value, what) in cases {
        let mut rig = Rig::new(ready(vec![], vec![]));
        rig.basic();
        // The fill offset makes the bias reach a triangle draw.
        rig.set(&clc797::SET_POLY_OFFSET_FILL, &[], 1);
        assert_eq!(rig.draw(), Outcome::Drawn);
        let before = rig.list.commands.len();
        rig.set(method, indices, value);
        assert_eq!(
            rig.draw_arrays(clc797::BEGIN_OP_TRIANGLES, 0, 3),
            Err(DrawError::NotFinite { what }),
            "{}",
            method.name
        );
        // Refused before anything was appended.
        assert_eq!(rig.list.commands.len(), before, "{}", method.name);
        assert_eq!(rig.draws.counters().failed, 1);
        // The draw before it still reaches the backend.
        let list = core::mem::take(&mut rig.list);
        assert!(rig.backend.submit(list).is_ok(), "{}", method.name);
    }
    // A bias that a topology does not apply is not looked at.
    let mut rig = Rig::new(ready(vec![], vec![]));
    rig.basic();
    rig.set(&clc797::SET_DEPTH_BIAS, &[], f32::NAN.to_bits());
    assert_eq!(rig.draw(), Outcome::Drawn);
}

#[test]
fn a_host_with_one_viewport_gets_one_viewport_and_scissor() {
    let mut one = caps(true, Features::ALL, 16);
    one.limits.max_viewports = 1;
    let mut rig = Rig::with(one, WaitWhileCompiling::Wait, ready(vec![], vec![]));
    rig.basic();
    assert_eq!(rig.draw(), Outcome::Drawn);
    let draw = rig.last_draw();
    assert_eq!(
        (draw.dynamic.viewports.len(), draw.dynamic.scissors.len()),
        (1, 1)
    );
    let list = core::mem::take(&mut rig.list);
    assert!(rig.backend.submit(list).is_ok());
    // A host with more than the guest's 16 still gets 16.
    let mut many = caps(true, Features::ALL, 16);
    many.limits.max_viewports = 32;
    let mut rig = Rig::with(many, WaitWhileCompiling::Wait, ready(vec![], vec![]));
    rig.basic();
    rig.draw();
    let slots = usize::try_from(crate::back::state::VIEWPORTS).unwrap();
    assert_eq!(rig.last_draw().dynamic.viewports.len(), slots);
}

#[test]
fn loading_a_root_table_uploads_the_constants_buffer_again() {
    let mut rig = Rig::new(ready(vec![], vec![uniform(0, 0, FIRST_ROOT_BANK, 0)]));
    rig.basic();
    let uploads = |rig: &Rig| {
        rig.list
            .commands
            .iter()
            .filter(|c| matches!(c, Command::Upload { from, .. } if u64::try_from(from.len()) == Ok(CONSTANTS_BUFFER_SIZE)))
            .count()
    };
    rig.draw();
    assert_eq!(uploads(&rig), 1);
    rig.draw();
    assert_eq!(uploads(&rig), 1);
    rig.set(
        &clc797::SET_ROOT_TABLE_SELECTOR,
        &[],
        fields(&[(clc797::SET_ROOT_TABLE_SELECTOR_OFFSET, 4)]),
    );
    rig.mirror
        .run(clc797::LOAD_ROOT_TABLE.method(), &[0xabcd])
        .unwrap();
    rig.draws.load_root_table(&rig.mirror, &[0xabcd]).unwrap();
    rig.draw();
    assert_eq!(uploads(&rig), 2);
    let Some(Command::Upload { from, .. }) = rig
        .list
        .commands
        .iter()
        .rev()
        .find(|c| matches!(c, Command::Upload { .. }))
    else {
        unreachable!("an upload");
    };
    assert_eq!(from.bytes()[4..8], 0xabcdu32.to_le_bytes());
}

#[test]
fn a_target_at_address_zero_is_made_like_any_other() {
    // Eden's FindColorBuffer and FindDepthBuffer return no view for GPU address 0; here the
    // texture cache makes the target (a listed divergence).
    let mut rig = Rig::new(ready(vec![], vec![]));
    rig.basic();
    rig.address(
        &clc797::SET_COLOR_TARGET_A,
        &clc797::SET_COLOR_TARGET_B,
        &[0],
        0,
    );
    rig.set(&clc797::SET_ZT_FORMAT, &[], clc797::SET_ZT_FORMAT_V_Z16);
    rig.set(&clc797::SET_ZT_SELECT, &[], 1);
    rig.set(&clc797::SET_ZT_SIZE_A, &[], SIZE);
    rig.set(&clc797::SET_ZT_SIZE_B, &[], SIZE);
    let targets = targets::targets(&rig.mirror).unwrap();
    assert_eq!(targets.colors[0].map(|info| info.address), Some(0));
    assert_eq!(targets.depth.map(|info| info.address), Some(0));
    assert_eq!(rig.draw(), Outcome::Drawn);
    let set = rig
        .list
        .commands
        .iter()
        .find_map(|c| match c {
            Command::SetRenderTargets(targets) => Some(*targets),
            _ => None,
        })
        .unwrap();
    assert!(set.colors[0].is_some());
    assert!(set.depth_stencil.is_some());
}

// ---------------------------------------------------------------------------------------------
// The previous stage's stores (owner's ruling 8 of 2026-11-19).

/// A generic output slot `generic`, as the lowering declares it.
fn generic_slot(generic: u8) -> InterfaceSlot {
    InterfaceSlot {
        binding: SlotBinding::Location(u32::from(generic)),
        ty: SlotType::F32x4,
        guest: GuestSlot::Generic(generic),
    }
}

/// The exit-only vertex shader with the generic outputs `generics` added after its position.
fn vertex_storing(generics: &[u8]) -> HostShader {
    let mut shader = host_shader(GuestStage::Vertex);
    shader
        .outputs
        .extend(generics.iter().copied().map(generic_slot));
    shader
}

/// A ready set of `vertex` and the exit-only pixel shader, binding nothing.
fn ready_with(vertex: HostShader, pixel: HostShader) -> SetLookup {
    SetLookup::Ready(ReadySet {
        shaders: Arc::new(ShaderSet {
            stages: vec![vertex, pixel],
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

/// The mask of the words `words`.
fn words(words: &[u8]) -> AttributeMask {
    words
        .iter()
        .copied()
        .fold(AttributeMask::EMPTY, AttributeMask::with)
}

/// The previous stages' stores each lookup's key carried, for the pixel stage.
fn pixel_stores(rig: &Rig) -> Vec<AttributeMask> {
    rig.shaders
        .seen
        .iter()
        .map(|s| s.key.previous_stage_stores.get(FollowingStage::Pixel))
        .collect()
}

#[test]
fn stored_words_are_whole_slots() {
    let outputs = [
        InterfaceSlot {
            binding: SlotBinding::BuiltIn(HostBuiltIn::Position),
            ty: SlotType::F32x4,
            guest: GuestSlot::Position,
        },
        generic_slot(0),
        generic_slot(31),
        // Past the last generic: no words.
        generic_slot(32),
        generic_slot(u8::MAX),
        // Not attribute words.
        InterfaceSlot {
            binding: SlotBinding::Location(1),
            ty: SlotType::F32x4,
            guest: GuestSlot::RenderTarget(1),
        },
        InterfaceSlot {
            binding: SlotBinding::BuiltIn(HostBuiltIn::SubgroupInvocationId),
            ty: SlotType::U32,
            guest: GuestSlot::LaneId,
        },
    ];
    // Eden's `attribute.h`, written out: PositionX..W = 28..31, Generic0X..W = 32..35,
    // Generic31X..W = 156..159.
    assert_eq!(
        shaders::stored_words(&outputs),
        words(&[28, 29, 30, 31, 32, 33, 34, 35, 156, 157, 158, 159])
    );
    // Each slot alone, so a slot that adds the wrong words fails on its own.
    assert_eq!(
        shaders::stored_words(&outputs[..1]),
        words(&[28, 29, 30, 31])
    );
    assert_eq!(
        shaders::stored_words(&[generic_slot(1)]),
        words(&[36, 37, 38, 39])
    );
    for slot in &outputs[3..] {
        assert_eq!(
            shaders::stored_words(core::slice::from_ref(slot)),
            AttributeMask::EMPTY,
            "{slot:?}"
        );
    }
    assert_eq!(shaders::stored_words(&[]), AttributeMask::EMPTY);
    // The exit-only vertex shader stores its position only, and the pixel shader no attribute.
    assert_eq!(
        shaders::stored_words(&host_shader(GuestStage::Vertex).outputs),
        words(&[28, 29, 30, 31])
    );
}

#[test]
fn previous_stage_stores_follow_the_request_order() {
    let request = |stage| ProgramRequest {
        stage,
        address: VERTEX_PROGRAM,
    };
    // Distinct stores for every stage, and none known for geometry.
    let stores_of = |stage| match stage {
        GuestStage::VertexCullBeforeFetch => Some(words(&[1])),
        GuestStage::Vertex => Some(words(&[2])),
        GuestStage::TessellationInit => Some(words(&[3])),
        GuestStage::Tessellation => Some(words(&[4])),
        GuestStage::Geometry => None,
        GuestStage::Pixel => Some(words(&[6])),
        GuestStage::Compute => Some(words(&[7])),
    };
    let all = [
        GuestStage::VertexCullBeforeFetch,
        GuestStage::Vertex,
        GuestStage::TessellationInit,
        GuestStage::Tessellation,
        GuestStage::Geometry,
        GuestStage::Pixel,
    ]
    .map(request);
    let stores = shaders::previous_stage_stores(&all, stores_of);
    assert_eq!(
        FollowingStage::ALL.map(|stage| stores.get(stage)),
        [words(&[2]), words(&[3]), words(&[4]), AttributeMask::EMPTY]
    );
    // Vertex and pixel: the pixel stage gets the vertex stage's, the rest stay empty.
    let pair = [GuestStage::Vertex, GuestStage::Pixel].map(request);
    assert_eq!(
        shaders::previous_stage_stores(&pair, stores_of),
        PreviousStores::EMPTY.with(FollowingStage::Pixel, words(&[2]))
    );
    // With the vertex-cull program first, the vertex program's stores (not the cull
    // program's) reach the pixel stage.
    let culled = [
        GuestStage::VertexCullBeforeFetch,
        GuestStage::Vertex,
        GuestStage::Pixel,
    ]
    .map(request);
    assert_eq!(
        shaders::previous_stage_stores(&culled, stores_of),
        PreviousStores::EMPTY.with(FollowingStage::Pixel, words(&[2]))
    );
    // Geometry then pixel: geometry's stores are unknown, so the pixel stage's are empty, and
    // geometry gets the vertex stage's.
    let geometry = [GuestStage::Vertex, GuestStage::Geometry, GuestStage::Pixel].map(request);
    assert_eq!(
        shaders::previous_stage_stores(&geometry, stores_of),
        PreviousStores::EMPTY.with(FollowingStage::Geometry, words(&[2]))
    );
    // A vertex stage alone, or nothing: no stores.
    assert_eq!(
        shaders::previous_stage_stores(&[request(GuestStage::Vertex)], stores_of),
        PreviousStores::EMPTY
    );
    assert_eq!(
        shaders::previous_stage_stores(&[], stores_of),
        PreviousStores::EMPTY
    );
}

/// The 32 bytes of the pixel stage's stores in an encoded key: after the version, 32 attribute
/// types, 8 colour types, the dual-source switch and the three earlier stages' 32 bytes each.
fn pixel_stores_bytes(key: &VariantKey) -> Vec<u8> {
    let at = 1 + 32 + 8 + 1 + 3 * 32;
    key.encode()[at..at + 32].to_vec()
}

#[test]
fn the_previous_stages_stores_reach_the_key() {
    // The vertex shader stores its position and generic 2.
    let mut rig = Rig::new(ready_with(
        vertex_storing(&[2]),
        host_shader(GuestStage::Pixel),
    ));
    rig.basic();
    // First draw: the set is unknown, so the first lookup has no stores, the second has them.
    assert_eq!(rig.draw(), Outcome::Drawn);
    let stored = words(&[28, 29, 30, 31, 40, 41, 42, 43]);
    assert_eq!(pixel_stores(&rig), [AttributeMask::EMPTY, stored]);
    // Golden: words 28 to 31 are lane 0 bits 28 to 31 (byte 3, high nibble), words 40 to 43
    // lane 0 bits 40 to 43 (byte 5, low nibble).
    let mut golden = [0u8; 32];
    golden[3] = 0xf0;
    golden[5] = 0x0f;
    assert_eq!(pixel_stores_bytes(&rig.shaders.seen[1].key), golden);
    assert_eq!(pixel_stores_bytes(&rig.shaders.seen[0].key), [0u8; 32]);
    // Only the pixel stage is specialised: the other stages are not in the requests.
    let key = &rig.shaders.seen[1].key;
    for stage in [
        FollowingStage::TessellationInit,
        FollowingStage::Tessellation,
        FollowingStage::Geometry,
    ] {
        assert_eq!(key.previous_stage_stores.get(stage), AttributeMask::EMPTY);
    }
    // The two keys differ in the stores only.
    let mut first = rig.shaders.seen[0].key.clone();
    first.previous_stage_stores = key.previous_stage_stores;
    assert_eq!(&first, key);

    // Second draw: one lookup, with the stores remembered.
    assert_eq!(rig.draw(), Outcome::Drawn);
    assert_eq!(pixel_stores(&rig)[2..], [stored]);
    assert_eq!(rig.draws.counters().lookups, 3);

    // The pixel shader's outputs change (not a stage anything follows): the same key, one
    // lookup.
    let mut pixel = host_shader(GuestStage::Pixel);
    pixel.outputs.push(generic_slot(5));
    rig.shaders.default = ready_with(vertex_storing(&[2]), pixel);
    assert_eq!(rig.draw(), Outcome::Drawn);
    assert_eq!(pixel_stores(&rig)[3..], [stored]);
    assert_eq!(rig.shaders.seen[3].key, rig.shaders.seen[2].key);

    // The vertex shader's stores change (its program was rewritten): the remembered stores
    // give the first lookup, the new ones the second, and the key changes.
    rig.shaders.default = ready_with(vertex_storing(&[0]), host_shader(GuestStage::Pixel));
    assert_eq!(rig.draw(), Outcome::Drawn);
    let restored = words(&[28, 29, 30, 31, 32, 33, 34, 35]);
    assert_eq!(pixel_stores(&rig)[4..], [stored, restored]);
    let mut golden = [0u8; 32];
    golden[3] = 0xf0;
    golden[4] = 0x0f;
    assert_eq!(pixel_stores_bytes(&rig.shaders.seen[5].key), golden);
    assert_ne!(rig.shaders.seen[5].key, rig.shaders.seen[2].key);
    assert_eq!(rig.draws.counters().lookups, 6);
}

#[test]
fn stores_are_not_learned_from_a_waited_set() {
    let mut rig = Rig::new(SetLookup::Pending(Ticket::new(4)));
    rig.shaders.waits.push_back(ready_with(
        vertex_storing(&[2]),
        host_shader(GuestStage::Pixel),
    ));
    rig.basic();
    assert_eq!(rig.draw(), Outcome::Drawn);
    assert_eq!(rig.shaders.waited, [Ticket::new(4)]);
    // Drawn with the set the wait gave, looked up once with no stores.
    assert_eq!(pixel_stores(&rig), [AttributeMask::EMPTY]);
    // The next draw still starts from no stores.
    rig.shaders.default = ready_with(vertex_storing(&[2]), host_shader(GuestStage::Pixel));
    assert_eq!(rig.draw(), Outcome::Drawn);
    assert_eq!(
        pixel_stores(&rig)[1..],
        [
            AttributeMask::EMPTY,
            words(&[28, 29, 30, 31, 40, 41, 42, 43])
        ]
    );
}

#[test]
fn remembered_stores_are_capped() {
    let mut rig = Rig::new(ready_with(
        vertex_storing(&[2]),
        host_shader(GuestStage::Pixel),
    ));
    rig.basic();
    rig.draws.previous_stores_limit = 1;
    let stored = words(&[28, 29, 30, 31, 40, 41, 42, 43]);
    // The first set fills the table to its cap.
    rig.draw();
    rig.draw();
    assert_eq!(rig.draws.counters().lookups, 3);
    // Another set past the cap is not remembered: each draw looks it up twice.
    rig.address(
        &clc797::SET_PIPELINE_PROGRAM_ADDRESS_A,
        &clc797::SET_PIPELINE_PROGRAM_ADDRESS_B,
        &[5],
        PIXEL_PROGRAM + 0x100,
    );
    rig.draw();
    rig.draw();
    assert_eq!(rig.draws.counters().lookups, 7);
    assert_eq!(
        pixel_stores(&rig)[3..],
        [AttributeMask::EMPTY, stored, AttributeMask::EMPTY, stored]
    );
    // The set already remembered is still updated at the cap.
    rig.address(
        &clc797::SET_PIPELINE_PROGRAM_ADDRESS_A,
        &clc797::SET_PIPELINE_PROGRAM_ADDRESS_B,
        &[5],
        PIXEL_PROGRAM,
    );
    rig.shaders.default = ready_with(vertex_storing(&[0]), host_shader(GuestStage::Pixel));
    rig.draw();
    rig.draw();
    let restored = words(&[28, 29, 30, 31, 32, 33, 34, 35]);
    assert_eq!(pixel_stores(&rig)[7..], [stored, restored, restored]);
    assert_eq!(rig.draws.counters().lookups, 10);
    assert_eq!(MAX_STORE_SETS, 4096);
}

// ---------------------------------------------------------------------------------------------
// The texture kind's usage (owner's ruling 7 of 2026-11-19), through the production engine
// state.

/// A program that asks about the bound handle once, with `usage`.
struct AsksHandle(TextureUsage);

impl Questioner for AsksHandle {
    fn questions(&self, _answers: &Answers) -> Vec<Question> {
        vec![Question::TextureKind {
            handle: HEADER_INDEX,
            usage: self.0,
        }]
    }
}

#[test]
fn a_format_change_keeps_a_sampled_answer_through_engine_state() {
    let mut rig = Rig::new(ready(vec![], vec![]));
    rig.basic();
    let header_at = HEADERS + u64::from(HEADER_INDEX) * shaders::HEADER_BYTES;
    let unorm = texture_header(TEXTURE, swizzle_sources(tex::TEXHEAD_BL_X_SOURCE_IN_R));
    // The same layout with signed normalised components: another format, the same kind
    // (sampled as float, not depth, not multisampled, not a buffer).
    let mut snorm = unorm;
    for field in [
        tex::TEXHEAD_BL_R_DATA_TYPE,
        tex::TEXHEAD_BL_G_DATA_TYPE,
        tex::TEXHEAD_BL_B_DATA_TYPE,
        tex::TEXHEAD_BL_A_DATA_TYPE,
    ] {
        field
            .insert(&mut snorm, tex::TEXHEAD_BL_R_DATA_TYPE_NUM_SNORM)
            .unwrap();
    }
    assert_ne!(unorm[0], snorm[0]);
    let roots = RootTables::new();
    let exchanged = |rig: &Rig, words: &[u32; HEADER_WORDS], usage| {
        rig.write_words(header_at, words);
        let answers = StateAnswers {
            mirror: &rig.mirror,
            roots: &roots,
            memory: &rig.mem,
        };
        exchange(GuestStage::Pixel, &AsksHandle(usage), &answers)
            .unwrap()
            .answers
    };
    let sampled = [
        exchanged(&rig, &unorm, TextureUsage::Sampled),
        exchanged(&rig, &snorm, TextureUsage::Sampled),
    ];
    assert_eq!(sampled[0], sampled[1]);
    assert_eq!(
        sampled[0].texture_kind(HEADER_INDEX, TextureUsage::Sampled),
        Some(Some(TextureKind {
            sampled: SampledType::Float,
            depth: false,
            multisampled: false,
            buffer: false,
            format: TextureFormat::UNREAD,
        }))
    );
    let image = [
        exchanged(&rig, &unorm, TextureUsage::Image),
        exchanged(&rig, &snorm, TextureUsage::Image),
    ];
    assert_ne!(image[0], image[1]);
    assert_eq!(
        image[1]
            .texture_kind(HEADER_INDEX, TextureUsage::Image)
            .flatten()
            .map(|kind| kind.format),
        Some(TextureFormat::from_texhead_word0(snorm[0]))
    );
}
