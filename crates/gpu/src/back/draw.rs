//! Draws: turning 3D engine state into a [`Command::Draw`] (gpu-protocol spec, "3D: from state to
//! a draw", "What the back hands the render backend" and "Shader pipeline (#13)"; slice G18).
//!
//! [`Draws`] serves one 3D mirror (one channel's 3D class). The executor (slice G19) applies each
//! ring entry to the mirror and then hands every word written to a 3D method to
//! [`Draws::method`]; a word that starts a draw ([`call`]'s methods) derives it and appends its
//! commands to the open list. Eden draws deferred instances (`DrawDeferred`) after each macro;
//! the ring marks no macro's end, so the executor calls [`Draws::flush`] at the boundaries its
//! module docs list ("Deferred instanced draws"). It hands each run of `LOAD_ROOT_TABLE` words
//! to [`Draws::load_root_table`] before the selector write that follows the run. A word written
//! to `CLEAR_SURFACE` is a clear ([`clear`], slice G24), which shares the draw path's render
//! targets, conditional render answer and host-object lifetime.
//!
//! # A draw, step by step
//!
//! 1. **Conditional render** (`SET_RENDER_ENABLE_*`): a draw the guest disabled is skipped
//!    ([`Skip::RenderDisabled`]) before anything is derived, as Eden checks `ShouldExecute`
//!    before `rasterizer->Draw`. The answer is worked out when `SET_RENDER_ENABLE_C` is written
//!    and kept, as Eden keeps `execute_on` ([`conditional`]).
//! 2. **Parts** ([`fixed`]): each part is derived again only when a group it reads is dirty, and
//!    the dirty groups are cleared once every part is derived. The render targets, the program
//!    addresses, the alpha test, the topology override and the sample mode are read every draw.
//! 3. **Programs**: one [`ProgramRequest`] per enabled pipeline slot `j`, its stage the slot's
//!    (Eden's `ShaderType(index)`, the names `SET_PIPELINE_SHADER_TYPE` gives the same values)
//!    and its address `SET_PIPELINE_PROGRAM_ADDRESS_A/B(j)` (decided by the owner, 2026-11-18).
//!    Slot 1 (vertex) is always enabled and the pixel slot is skipped while rasterisation is off,
//!    as Eden's `RefreshStages` has it (`eden-nx2/src/video_core/shader_cache.cpp` @ 50fa86bb).
//!    `SET_PIPELINE_SHADER_TYPE` is not read: Eden takes the stage from the slot.
//! 4. **The variant key** ([`shaders::variant_key`]), with the backend's [`EmulationFlags`](starquake_render::EmulationFlags)
//!    from [`starquake_render::emulation`], the previous stages' stores
//!    ([`shaders::previous_stage_stores`]) and, without view swizzle, the texture swizzles.
//! 5. **The lookup** ([`ShaderLookup::lookup`]): `Ready` gives the set; `Pending` follows the
//!    one wait-while-compiling setting ([`WaitWhileCompiling`], owner's decision 9 and the
//!    first of the graphics decisions of 2026-11-12): wait for the ticket, or skip
//!    ([`Skip::ShaderPending`]); `Rejected` skips the draw ([`Skip::ShaderRejected`]), logged and
//!    counted once per distinct set of requests (rows 90, 112; the rejection does not say which
//!    program of the set failed, so the set is the unit). **This departs from the approved
//!    spec**, which says (gpu-protocol, "3D: from state to a draw"): "A draw whose shader was
//!    rejected is skipped and counted, once per program in the log (negative cache, rows 90,
//!    112)." Eden likewise keys `IsRejected` by program address. Here a program rejected in two
//!    sets (one vertex program with two pixel programs) is logged and counted twice, and a set
//!    is remembered whole. Keying by program needs the shader cache's rejection to name the
//!    program, which the seam's [`starquake_shader::lookup::Rejection`] does not; an **owner
//!    question**, pinned by `tests::a_rejected_set_skips_its_draws_and_logs_once`.
//! 6. **Resources**: each shader's constant buffers (banks 24 to 31 from the root tables,
//!    [`root`]; a bank past 31 is refused, [`DrawError::RootBank`], as PR #4588's `IsBank`
//!    requires `bank - 24 < 8`, `tests::root_banks_past_the_last_table_are_refused`; a bank
//!    below 24 needs `VALID` and a size above 0, `tests::a_bound_constant_buffer_of_size_zero_is_unbound`),
//!    textures and samplers (by handle, through the texture cache and [`sampler`]),
//!    the vertex streams and the index buffer through the buffer cache, the render targets
//!    through the texture cache ([`targets`]), and the pipeline through the pipeline-key cache.
//!    A pipeline that failed is skipped ([`Skip::PipelineFailed`]); one still compiling is drawn,
//!    and the render thread applies the wait setting to it (render-backend, "Compiling without
//!    stalling a frame": "pipeline waits block the render thread").
//! 7. **Driver constants** ([`starquake_shader::driver_constants`]): the alpha reference
//!    (`SET_ALPHA_REF`'s bits as a float, as Eden's `bit_cast`), the viewport flip (Eden's
//!    `y_negate`: the window origin is the lower left, `fixed_pipeline_state.cpp`), the render
//!    area (the surface clip, or colour target 0's size when the clip is not set, as Eden-NX2's
//!    `vk_graphics_pipeline.cpp`), and the draw's base vertex and instance.
//! 8. **The draw**: a [`Command::SetRenderTargets`] when the targets' views changed since the
//!    last one, then the [`Command::Draw`], with the first `min(Caps::limits.max_viewports, 16)`
//!    viewports and scissors, as Eden sends `min(device.GetMaxViewports(), NumViewports)`
//!    (`UpdateViewportsState`, `eden/src/video_core/renderer_vulkan/vk_rasterizer.cpp`;
//!    `tests::a_host_with_one_viewport_gets_one_viewport_and_scissor`).
//!
//! Between steps 2 and 3 the dynamic values the draw hands the backend (every viewport, the blend
//! constant, the depth bias the topology applies and the line width) must be finite, or the draw
//! is refused ([`DrawError::NotFinite`]) before any cache or the list is touched: the backend
//! would refuse the value at submit and lose every draw in the list. Eden hands such a value to
//! the host API as it is; a listed divergence,
//! `tests::a_draw_with_a_value_that_is_not_finite_is_refused_and_the_list_still_submits`.
//!
//! # Errors and what they leave
//!
//! A draw whose state cannot be turned into a valid description is an `Err` ([`DrawError`]),
//! which the executor logs and counts, and the draw is not drawn. The dirty groups are cleared
//! only when every part derives; a part whose derivation failed is derived again next draw.
//! Uploads the caches appended for resources resolved before the failure stay in the list: they
//! are the caches' own consistent work (each cache's state says the bytes were uploaded), and no
//! `Draw` command follows them. Pinned by `tests::a_failed_draw_leaves_only_cache_uploads`.
//!
//! # The constants buffer
//!
//! The root tables and the driver constants reach the shaders through one host uniform buffer
//! the draw path owns: the eight tables at offsets `256 * i`, then the driver constants at
//! [`DRIVER_CONSTANTS_OFFSET`]. That offsets are multiples of [`CONSTANTS_ALIGN`] (256, WebGPU's
//! default uniform-buffer offset alignment) is a **guess** at what every host accepts; `Caps`
//! names no such limit. The buffer is uploaded before a draw that binds it whenever the root
//! tables were written ([`RootTables::generation`] moved) or the encoded driver constants changed
//! since the last upload; the command model orders each upload before the draws after it ("Every
//! command says which resources it reads and writes"). The comparison allocates nothing; the
//! bytes are built in a stack array only for an upload
//! (`tests::the_constants_buffer_is_uploaded_when_its_bytes_change`,
//! `tests::loading_a_root_table_uploads_the_constants_buffer_again`).
//!
//! # Divergences from Eden
//!
//! Each fails a draw (or makes an object) where Eden still draws (or binds nothing); each is
//! pinned by the test named.
//!
//! - **An active attribute on a disabled stream fails the draw** ([`DrawError::AttributeStream`]).
//!   Eden binds all 32 streams and leaves out the attributes the vertex program does not load
//!   (`stage_infos[0].loads.Generic`, `vk_graphics_pipeline.cpp`); the seam's sets carry no
//!   such list (`fixed`'s vertex docs). `tests::every_active_attribute_is_in_the_layout`.
//! - **A texture or sampler index past its pool's maximum fails the draw**
//!   ([`DrawError::PoolIndex`]). Eden binds `NULL_IMAGE_VIEW_ID` or `NULL_SAMPLER_ID`
//!   (`VisitImageView` and `GetSamplerId`, `eden/src/video_core/texture_cache/texture_cache.h`).
//!   `tests::a_handle_past_a_pool_maximum_is_an_error`.
//! - **A colour or depth target at GPU address 0 is made like any other.** Eden's
//!   `FindColorBuffer` and `FindDepthBuffer` (same file) return no view for address 0; here the
//!   texture cache makes the target. `tests::a_target_at_address_zero_is_made_like_any_other`.
//! - **A non-finite dynamic value fails the draw** (above).
//!
//! # Not handled here
//!
//! The inline index draws, `DRAW_ZERO_INDEX` and `DRAW_AUTO` ([`Skip::Unhandled`]), indirect
//! draws (they reach the 3D class through macros), storage buffers and images (the seam's
//! resource table has no such kind yet), the catch-up before reading a conditional-render value
//! that a pending release will write (the executor's, `back::exec`), tessellation and geometry
//! shaders in the pipeline (the
//! shader crate's `HostStage` has vertex and fragment only), and the storage-buffer fetch of
//! attributes past the host's limit: with `ATTRIBUTE_FETCH` set every attribute is still in the
//! vertex layout, which the backend refuses, so such a draw is skipped as a failed pipeline
//! (`tests::every_emulation_flag_reaches_the_key_through_a_draw`).

pub mod call;
pub mod clear;
pub mod conditional;
pub mod fixed;
pub mod root;
pub mod sampler;
pub mod shaders;
pub mod targets;

#[cfg(test)]
pub(crate) mod tests;

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use starquake_render::command::{
    Binding, BindingResource, BufferSlice, CopyRegion, DrawDesc, DrawParams, DynamicValues,
    IndexBufferBinding, IndexFormat, RenderTargets,
};
use starquake_render::pipeline::{GraphicsPipelineDesc, RasterState, Stages};
use starquake_render::validate::viewport_limit;
use starquake_render::{
    Backend, BufferDesc, BufferId, BufferUsage, Command, CommandList, DepthRange, EmulationState,
    FenceValue, RenderError, ResourceId, emulation,
};
use starquake_shader::driver_constants::{DRIVER_CONSTANTS_SIZE, DriverConstants};
use starquake_shader::lookup::{GuestStage, ProgramRequest};
use starquake_shader::variant::{KeyError, PreviousStores, Swizzle, TextureSwizzles};
use thiserror::Error;

use crate::back::buffer::{BufferCache, BufferError};
use crate::back::pipeline::{
    Lookup as PipelineLookup, PipelineCache, PipelineCacheError, ShaderStamp, StageStamps,
};
use crate::back::state::{Mirror, PIPELINES, StateError};
use crate::back::texture::{HeaderError, TextureCache, TextureError, TextureHeader};
use crate::classes::clc797;
use crate::memory::{GpuFault, GpuMemory};
use call::{DrawCall, DrawCalls, IndexSize, Source, Started};
use fixed::Parts;
use root::{ROOT_TABLE_BYTES, ROOT_TABLES, RootTables};
use sampler::{SAMPLER_BYTES, SAMPLER_WORDS, Samplers};
use shaders::{
    GuestPrograms, HandleSource, HostShaders, ReadySet, ResourceSource, SetLookup, ShaderLookup,
    StateAnswers, header_swizzle,
};

/// What a draw does while its shaders are being translated (owner's decision 9: one setting,
/// owned by this crate, covering translation and pipeline builds; render-backend, "Compiling
/// without stalling a frame").
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum WaitWhileCompiling {
    /// Wait for it: correct frames and deterministic dumps (the default, decision 9).
    #[default]
    Wait,
    /// Skip the draw and count it (opt-in, once the persistent caches exist).
    Skip,
}

/// Why a draw was not drawn, though nothing was wrong with its state.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Skip {
    /// Conditional rendering turned it off.
    RenderDisabled,
    /// A draw method this slice does not handle, by its name in the class header.
    Unhandled(&'static str),
    /// Its shaders are being translated and the setting is [`WaitWhileCompiling::Skip`] (or a
    /// wait gave a ticket still pending).
    ShaderPending,
    /// A program of its set was rejected; `newly` on the first draw of that set.
    ShaderRejected {
        /// Whether this is the first draw of the set to be skipped for it.
        newly: bool,
    },
    /// Its pipeline cannot be built; `newly` on the first lookup to learn of it.
    PipelineFailed {
        /// The pipeline-key cache's [`crate::back::pipeline::Failure::newly`].
        newly: bool,
    },
}

/// What one draw method did.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Outcome {
    /// A `Draw` command was appended.
    Drawn,
    /// The draw was skipped.
    Skipped(Skip),
    /// A `CLEAR_SURFACE` word was handled ([`clear`]): what became of each part.
    Cleared(clear::Cleared),
}

impl Outcome {
    /// Whether a draw was appended, which sets render targets: a draw, or a clear drawn
    /// ([`clear::Cleared::drew`]).
    pub fn drew(&self) -> bool {
        match self {
            Self::Drawn => true,
            Self::Skipped(_) => false,
            Self::Cleared(cleared) => cleared.drew(),
        }
    }
}

/// Counts of what the draw path did, for counters and tests.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DrawCounters {
    /// Draws appended.
    pub drawn: u64,
    /// Draws skipped by conditional rendering.
    pub render_disabled: u64,
    /// Words to draw methods this slice does not handle, and `END`s in the inline index mode
    /// ([`call`]'s module docs).
    pub unhandled: u64,
    /// Draws skipped while their shaders were translated.
    pub shader_pending: u64,
    /// Draws skipped for a rejected set.
    pub shader_rejected: u64,
    /// Distinct request sets found rejected (each logged once).
    pub rejected_sets: u64,
    /// Rejected sets past [`MAX_REJECTED_SETS`], counted but not remembered or logged.
    pub rejected_unremembered: u64,
    /// Draws skipped for a failed pipeline.
    pub pipeline_failed: u64,
    /// Shader lookups made.
    pub lookups: u64,
    /// Draws that failed with a [`DrawError`].
    pub failed: u64,
}

/// The most rejected request sets remembered for logging once (a **guess** bounding the memory
/// a hostile guest can make the draw path hold; past it a rejected set is counted in
/// [`DrawCounters::rejected_unremembered`] and logged at debug level).
pub const MAX_REJECTED_SETS: usize = 4096;

/// The most request sets whose texture handles are remembered for the swizzle key (the same
/// kind of guess).
pub const MAX_SWIZZLE_SETS: usize = 4096;

/// The most request sets whose previous stages' stores are remembered for the key (the same
/// kind of guess as [`MAX_SWIZZLE_SETS`]; past it a set is looked up with the empty stores
/// first, then with its own).
pub const MAX_STORE_SETS: usize = 4096;

/// [`CONSTANTS_ALIGN`] as a length.
const ALIGN_BYTES: usize = 256;

/// [`DRIVER_CONSTANTS_OFFSET`] as an index.
const DRIVER_CONSTANTS_AT: usize = ROOT_TABLES * ROOT_TABLE_BYTES;

/// [`CONSTANTS_BUFFER_SIZE`] as a length: the stack array the buffer's bytes are built in.
const CONSTANTS_BUFFER_LEN: usize = DRIVER_CONSTANTS_AT + ALIGN_BYTES;

/// The alignment of each block in the constants buffer (module docs; a guess).
pub const CONSTANTS_ALIGN: u64 = ALIGN_BYTES as u64;

/// Where the driver constants start in the constants buffer: after the root tables.
pub const DRIVER_CONSTANTS_OFFSET: u64 = DRIVER_CONSTANTS_AT as u64;

/// The constants buffer's size: the root tables, then the driver constants padded to
/// [`CONSTANTS_ALIGN`].
pub const CONSTANTS_BUFFER_SIZE: u64 = CONSTANTS_BUFFER_LEN as u64;

const _: () = assert!(DRIVER_CONSTANTS_OFFSET.is_multiple_of(CONSTANTS_ALIGN));
const _: () = assert!((ROOT_TABLE_BYTES as u64).is_multiple_of(CONSTANTS_ALIGN));
const _: () = assert!(DRIVER_CONSTANTS_SIZE as u64 <= CONSTANTS_ALIGN);
const _: () = assert!(CONSTANTS_BUFFER_SIZE == DRIVER_CONSTANTS_OFFSET + CONSTANTS_ALIGN);

/// Why a draw could not be derived. The draw is not drawn; the executor logs and counts it.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum DrawError {
    /// A register read failed (another class's mirror, or a value the header does not define).
    #[error("reading 3D state: {0}")]
    State(#[from] StateError),
    /// A topology value no draw method defines.
    #[error("primitive topology {value:#x} is not one the class header defines for a draw")]
    Topology {
        /// The value.
        value: u32,
    },
    /// A value the class header defines but that has no host equivalent.
    #[error("{what} {name} has no host equivalent")]
    Unmapped {
        /// What the value is.
        what: &'static str,
        /// Its name in the class header.
        name: &'static str,
    },
    /// An active vertex attribute reads a stream that is not enabled.
    #[error("vertex attribute {attribute} reads vertex stream {stream}, which is not enabled")]
    AttributeStream {
        /// The attribute.
        attribute: u32,
        /// The stream.
        stream: u32,
    },
    /// A colour or depth target format that is not mapped.
    #[error("render target format {value:#x} is not supported")]
    TargetFormat {
        /// The format value.
        value: u32,
    },
    /// An anti-alias mode the class header does not define.
    #[error("anti-alias sample mode {mode:#x} is not defined")]
    SampleMode {
        /// The mode.
        mode: u32,
    },
    /// A sampler pool entry field holds a value that has no meaning.
    #[error("sampler field {field} holds {value:#x}, which is not supported")]
    Sampler {
        /// The field's name in `clc797tex.h`.
        field: &'static str,
        /// The value.
        value: u32,
    },
    /// The sampler table is full.
    #[error("the sampler table is full ({limit} samplers)")]
    SamplersFull {
        /// Its cap.
        limit: usize,
    },
    /// The host shader table is full.
    #[error("the host shader table is full ({limit} shaders)")]
    ShadersFull {
        /// Its cap.
        limit: usize,
    },
    /// A shader binds a constant buffer the stage has not bound.
    #[error("the {} stage reads constant buffer {bank}, which is not bound", .stage.name())]
    UnboundConstantBuffer {
        /// The stage.
        stage: GuestStage,
        /// The bank.
        bank: u8,
    },
    /// A shader binds a constant buffer bank at or past [`root::FIRST_ROOT_BANK`] that no root
    /// table backs (PR #4588's `IsBank`: `bank - 24 < 8`).
    #[error("the {} stage reads constant buffer {bank}, which no root table backs (banks {} to {})", .stage.name(), root::FIRST_ROOT_BANK, root::LAST_ROOT_BANK)]
    RootBank {
        /// The stage.
        stage: GuestStage,
        /// The bank.
        bank: u8,
    },
    /// A value the draw hands the render backend is not finite (a NaN or an infinity the guest
    /// wrote, or one its arithmetic made); the backend refuses such a value at submit, which
    /// would throw away the whole list.
    #[error("the draw's {what} is not finite")]
    NotFinite {
        /// Which value: `viewport`, `blend constant`, `depth bias` or `line width`.
        what: &'static str,
    },
    /// A texture or sampler handle could not be read from its constant buffer.
    #[error("the {} stage's handle at constant buffer {bank} offset {offset:#x} cannot be read", .stage.name())]
    Handle {
        /// The stage.
        stage: GuestStage,
        /// The bank.
        bank: u8,
        /// The offset.
        offset: u32,
    },
    /// A texture header or sampler index is past its pool's maximum.
    #[error("{pool} index {index} is past the pool's maximum, {maximum}")]
    PoolIndex {
        /// `texture header` or `sampler`.
        pool: &'static str,
        /// The index.
        index: u32,
        /// The pool's maximum index.
        maximum: u32,
    },
    /// A texture header does not decode.
    #[error("texture header {index}: {error}")]
    Header {
        /// The header's index.
        index: u32,
        /// Why.
        error: HeaderError,
    },
    /// A set's shaders and its stage information disagree, or it has no vertex shader.
    #[error("the shader set {0}")]
    BadSet(&'static str),
    /// The variant key refused the swizzles.
    #[error("variant key: {0}")]
    Key(#[from] KeyError),
    /// Reading guest memory failed.
    #[error("reading guest memory: {0}")]
    Fault(#[from] GpuFault),
    /// The buffer cache refused.
    #[error("buffer cache: {0}")]
    Buffer(#[from] BufferError),
    /// The texture cache refused.
    #[error("texture cache: {0}")]
    Texture(#[from] TextureError),
    /// The pipeline-key cache refused.
    #[error("pipeline cache: {0}")]
    Pipeline(#[from] PipelineCacheError),
    /// The render backend refused.
    #[error("render backend: {0}")]
    Render(RenderError),
    /// A state the code rules out was reached (a bug, reported rather than a panic).
    #[error("internal error: {0}")]
    Internal(&'static str),
    /// A masked clear needs a new pipeline and the draw-based clear holds its cap
    /// ([`clear::MAX_CLEAR_PIPELINES`]).
    #[error("the clear pipeline table is full ({limit} pipelines)")]
    ClearPipelinesFull {
        /// Its cap.
        limit: usize,
    },
}

/// What a draw needs besides its own state: guest memory, the caches, the shader cache and the
/// open command list.
pub struct DrawContext<'a, M: GpuMemory + ?Sized> {
    /// Guest memory of the channel's address space.
    pub memory: &'a M,
    /// The buffer cache.
    pub buffers: &'a mut BufferCache,
    /// The texture cache.
    pub textures: &'a mut TextureCache,
    /// The pipeline-key cache.
    pub pipelines: &'a mut PipelineCache,
    /// The shader cache.
    pub shaders: &'a mut dyn ShaderLookup,
    /// The open list.
    pub list: &'a mut CommandList,
}

/// The draw path of one 3D mirror (module docs).
pub struct Draws {
    backend: Arc<dyn Backend>,
    setting: WaitWhileCompiling,
    calls: DrawCalls,
    parts: Parts,
    roots: RootTables,
    host_shaders: HostShaders,
    samplers: Samplers,
    rejected: HashSet<Vec<ProgramRequest>>,
    rejected_limit: usize,
    swizzle_handles: HashMap<Vec<ProgramRequest>, Vec<(GuestStage, HandleSource)>>,
    /// The previous stages' stores each request set's last ready answer gave.
    previous_stores: HashMap<Vec<ProgramRequest>, PreviousStores>,
    previous_stores_limit: usize,
    constants: Option<BufferId>,
    /// What the last upload held: the root tables' generation and the encoded driver constants.
    uploaded: Option<(u64, [u8; DRIVER_CONSTANTS_SIZE])>,
    last_targets: Option<RenderTargets>,
    last_fence: FenceValue,
    /// The clear path ([`clear`]).
    clears: clear::Clears,
    /// Whether draws render (Eden's `execute_on`, true until `SET_RENDER_ENABLE_C` says
    /// otherwise).
    execute: bool,
    counters: DrawCounters,
}

impl core::fmt::Debug for Draws {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Draws")
            .field("setting", &self.setting)
            .field("counters", &self.counters)
            .finish_non_exhaustive()
    }
}

/// One stage's place in a ready set: its request's stage and its shader's index in the set.
struct SetStage {
    stage: GuestStage,
    index: usize,
}

impl Draws {
    /// The draw path for one 3D mirror, making host objects through `backend`.
    pub fn new(backend: Arc<dyn Backend>, setting: WaitWhileCompiling) -> Self {
        Self {
            host_shaders: HostShaders::new(Arc::clone(&backend)),
            samplers: Samplers::new(Arc::clone(&backend)),
            clears: clear::Clears::new(Arc::clone(&backend)),
            backend,
            setting,
            calls: DrawCalls::new(),
            parts: Parts::new(),
            roots: RootTables::new(),
            rejected: HashSet::new(),
            rejected_limit: MAX_REJECTED_SETS,
            swizzle_handles: HashMap::new(),
            previous_stores: HashMap::new(),
            previous_stores_limit: MAX_STORE_SETS,
            constants: None,
            uploaded: None,
            last_targets: None,
            last_fence: FenceValue::default(),
            execute: true,
            counters: DrawCounters::default(),
        }
    }

    /// The wait-while-compiling setting.
    pub fn setting(&self) -> WaitWhileCompiling {
        self.setting
    }

    /// What the draw path has done so far.
    pub fn counters(&self) -> DrawCounters {
        self.counters
    }

    /// What clears have done so far ([`clear`]).
    pub fn clear_counters(&self) -> clear::ClearCounters {
        self.clears.counters()
    }

    /// The pipelines the draw-based clear holds ([`clear`]).
    pub fn clear_pipelines(&self) -> usize {
        self.clears.pipelines()
    }

    /// How many times each fixed-function part was derived, in [`fixed::Part::ALL`] order.
    pub fn derivations(&self) -> [u64; fixed::Part::ALL.len()] {
        self.parts.derivations()
    }

    /// The root tables.
    pub fn root_tables(&self) -> &RootTables {
        &self.roots
    }

    /// Stores a run of `LOAD_ROOT_TABLE` words the mirror has applied ([`RootTables::load`]).
    ///
    /// # Errors
    ///
    /// As [`RootTables::load`].
    pub fn load_root_table(&mut self, mirror: &Mirror, words: &[u32]) -> Result<(), StateError> {
        self.roots.load(mirror, words)
    }

    /// The executor's report that the open list was submitted with fence `fence`: the constants
    /// buffer is destroyed after the last such fence when the draw path is dropped.
    pub fn submitted(&mut self, fence: FenceValue) {
        self.last_fence = self.last_fence.max(fence);
        // A new list starts with no targets set.
        self.last_targets = None;
    }

    /// One word `value` written to 3D method `method`, after `mirror` stored it: the draw it
    /// starts, if any (module docs).
    ///
    /// # Errors
    ///
    /// [`DrawError`] for a draw whose state cannot be turned into a description.
    pub fn method<M: GpuMemory + ?Sized>(
        &mut self,
        ctx: &mut DrawContext<'_, M>,
        mirror: &mut Mirror,
        method: u32,
        value: u32,
    ) -> Result<Option<Outcome>, DrawError> {
        if method == clc797::SET_RENDER_ENABLE_C.method() {
            self.execute = conditional::render_enabled(mirror, ctx.memory)?;
            return Ok(None);
        }
        if method == clc797::CLEAR_SURFACE.method() {
            return self
                .clears
                .clear(ctx, mirror, value, self.execute, &mut self.last_targets)
                .map(|cleared| Some(Outcome::Cleared(cleared)));
        }
        match self.calls.method(mirror, method, value)? {
            Started::Nothing => Ok(None),
            Started::Unhandled(name) => {
                self.counters.unhandled = self.counters.unhandled.saturating_add(1);
                tracing::debug!(method = name, "draw method not handled");
                Ok(Some(Outcome::Skipped(Skip::Unhandled(name))))
            }
            Started::Draw(call) => self.counted(ctx, mirror, &call).map(Some),
        }
    }

    /// Whether a begin/end has deferred instances for [`Draws::flush`] to draw.
    pub fn has_deferred(&self) -> bool {
        self.calls.has_deferred()
    }

    /// Draws the instances a begin/end deferred, if any (Eden's `DrawDeferred` after a macro).
    ///
    /// # Errors
    ///
    /// As [`Draws::method`].
    pub fn flush<M: GpuMemory + ?Sized>(
        &mut self,
        ctx: &mut DrawContext<'_, M>,
        mirror: &mut Mirror,
    ) -> Result<Option<Outcome>, DrawError> {
        match self.calls.flush(mirror)? {
            Some(call) => self.counted(ctx, mirror, &call).map(Some),
            None => Ok(None),
        }
    }

    /// [`Draws::draw`], counting the outcome.
    fn counted<M: GpuMemory + ?Sized>(
        &mut self,
        ctx: &mut DrawContext<'_, M>,
        mirror: &mut Mirror,
        call: &DrawCall,
    ) -> Result<Outcome, DrawError> {
        let result = self.draw(ctx, mirror, call);
        let c = &mut self.counters;
        let bump = |n: &mut u64| *n = n.saturating_add(1);
        match result {
            Ok(Outcome::Drawn) => bump(&mut c.drawn),
            // `draw` never clears.
            Ok(Outcome::Cleared(_)) => {}
            Ok(Outcome::Skipped(skip)) => match skip {
                Skip::RenderDisabled => bump(&mut c.render_disabled),
                Skip::Unhandled(_) => bump(&mut c.unhandled),
                Skip::ShaderPending => bump(&mut c.shader_pending),
                Skip::ShaderRejected { .. } => bump(&mut c.shader_rejected),
                Skip::PipelineFailed { .. } => bump(&mut c.pipeline_failed),
            },
            Err(_) => bump(&mut c.failed),
        }
        result
    }

    /// The requests of the enabled pipeline slots (module docs, step 3).
    fn requests(mirror: &Mirror, raster_enable: bool) -> Result<Vec<ProgramRequest>, DrawError> {
        const STAGES: [GuestStage; PIPELINES as usize] = [
            GuestStage::VertexCullBeforeFetch,
            GuestStage::Vertex,
            GuestStage::TessellationInit,
            GuestStage::Tessellation,
            GuestStage::Geometry,
            GuestStage::Pixel,
        ];
        let three_d = mirror.three_d()?;
        let mut requests = Vec::new();
        for (index, stage) in (0..PIPELINES).zip(STAGES) {
            let pipeline = three_d.pipeline(index)?;
            let enabled = stage == GuestStage::Vertex
                || pipeline.enable()?.value == clc797::SET_PIPELINE_SHADER_ENABLE_TRUE;
            if !enabled || (stage == GuestStage::Pixel && !raster_enable) {
                continue;
            }
            requests.push(ProgramRequest {
                stage,
                address: pipeline.program_address()?,
            });
        }
        Ok(requests)
    }

    /// The swizzles of `handles` read from current state, in their order.
    fn swizzles<M: GpuMemory + ?Sized>(
        answers: &StateAnswers<'_, M>,
        handles: &[(GuestStage, HandleSource)],
    ) -> Vec<Swizzle> {
        use starquake_shader::lookup::EngineState;
        handles
            .iter()
            .map(|&(stage, source)| {
                answers
                    .cbuf_word(stage, source.bank, source.offset)
                    .and_then(|raw| answers.header_words(answers.handle_pair(raw).0))
                    .and_then(|words| header_swizzle(&words))
                    .unwrap_or(Swizzle::IDENTITY)
            })
            .collect()
    }

    /// The texture handles a ready set binds, in binding order: each shader's texture uses in
    /// set order, then resource order.
    fn texture_handles(
        requests: &[ProgramRequest],
        set: &ReadySet,
    ) -> Vec<(GuestStage, HandleSource)> {
        Self::stages(requests, set)
            .iter()
            .flat_map(|s| {
                set.stages
                    .get(s.index)
                    .into_iter()
                    .flat_map(|info| info.resources.iter())
                    .filter_map(move |r| match r.source {
                        ResourceSource::Texture(handle) => Some((s.stage, handle)),
                        _ => None,
                    })
            })
            .collect()
    }

    /// The previous stages' stores of `requests` as the ready `set` gives them (module docs of
    /// [`shaders`], "The previous stage's stores").
    fn set_stores(requests: &[ProgramRequest], set: &ReadySet) -> PreviousStores {
        let stages = Self::stages(requests, set);
        shaders::previous_stage_stores(requests, |stage| {
            let index = stages.iter().find(|s| s.stage == stage)?.index;
            Some(shaders::stored_words(
                &set.shaders.stages.get(index)?.outputs,
            ))
        })
    }

    /// Each host shader's guest stage: the vertex shader is the vertex request's (the last
    /// vertex-like request), the fragment shader the pixel request's.
    fn stages(requests: &[ProgramRequest], set: &ReadySet) -> Vec<SetStage> {
        use starquake_shader::host::HostStage;
        set.shaders
            .stages
            .iter()
            .enumerate()
            .filter_map(|(index, shader)| {
                let stage = match shader.stage {
                    HostStage::Vertex => requests
                        .iter()
                        .rev()
                        .find(|r| {
                            matches!(
                                r.stage,
                                GuestStage::Vertex | GuestStage::VertexCullBeforeFetch
                            )
                        })
                        .map(|r| r.stage)?,
                    HostStage::Fragment => GuestStage::Pixel,
                };
                Some(SetStage { stage, index })
            })
            .collect()
    }

    /// One draw (module docs, "A draw, step by step").
    #[expect(
        clippy::too_many_lines,
        reason = "the spec's steps in order, each a few lines; splitting would hide the order"
    )]
    fn draw<M: GpuMemory + ?Sized>(
        &mut self,
        ctx: &mut DrawContext<'_, M>,
        mirror: &mut Mirror,
        call: &DrawCall,
    ) -> Result<Outcome, DrawError> {
        // Step 1.
        if !self.execute {
            return Ok(Outcome::Skipped(Skip::RenderDisabled));
        }
        // Step 2.
        let dirty = mirror.dirty();
        let (blend, raster, depth_stencil, vertex, mut viewports, mut scissors) = {
            let parts = self.parts.refresh(mirror, dirty)?;
            (
                parts.blend.clone(),
                *parts.raster,
                *parts.depth_stencil,
                parts.vertex.clone(),
                parts.viewports.to_vec(),
                parts.scissors.to_vec(),
            )
        };
        mirror.clear(dirty);
        let targets = targets::targets(mirror)?;
        let topology = fixed::effective_topology(mirror, call.topology)?;
        let depth_bias = raster.bias_for(topology);
        check_finite(&viewports, &blend.constant, depth_bias, raster.line_width)?;
        // Eden sends `min(device.GetMaxViewports(), NumViewports)` viewports and scissors
        // (`UpdateViewportsState` and `UpdateScissorsState` in `vk_rasterizer.cpp`).
        let used = usize::try_from(viewport_limit(self.backend.caps())).unwrap_or(usize::MAX);
        viewports.truncate(used);
        scissors.truncate(used);
        let alpha_on = mirror
            .reg(&clc797::SET_ALPHA_TEST, &[])?
            .field(clc797::SET_ALPHA_TEST_ENABLE)?
            != 0;
        // The function is read only while the test is on: its reset value is 0,
        // which the header does not name.
        let alpha_func = if alpha_on {
            Some(fixed::compare(
                mirror
                    .reg(&clc797::SET_ALPHA_FUNC, &[])?
                    .named(clc797::SET_ALPHA_FUNC_V)?,
            )?)
        } else {
            None
        };
        let alpha_ref = mirror.reg(&clc797::SET_ALPHA_REF, &[])?.float();
        let minus_one_to_one = mirror
            .reg(&clc797::SET_VIEWPORT_Z_CLIP, &[])?
            .field(clc797::SET_VIEWPORT_Z_CLIP_RANGE)?
            == clc797::SET_VIEWPORT_Z_CLIP_RANGE_NEGATIVE_W_TO_POSITIVE_W;
        // Step 3.
        let requests = Self::requests(mirror, raster.raster_enable)?;
        // Step 4.
        let caps = self.backend.caps().clone();
        let flags = emulation(
            &caps,
            &EmulationState {
                alpha_test: alpha_func,
                depth_range: if minus_one_to_one {
                    DepthRange::MinusOneToOne
                } else {
                    DepthRange::ZeroToOne
                },
                primitive_restart: raster.primitive_restart,
                vertex_attributes: u32::try_from(vertex.layout.attributes.len())
                    .unwrap_or(u32::MAX),
                color_targets: targets.formats.colors,
            },
        );
        let emulation_key = shaders::emulation_key(flags, alpha_func);
        let answers = StateAnswers {
            mirror,
            roots: &self.roots,
            memory: ctx.memory,
        };
        let programs = GuestPrograms(ctx.memory);
        let key_for = |swizzles: Vec<Swizzle>, stores: PreviousStores| -> Result<_, DrawError> {
            Ok(shaders::variant_key(
                &vertex.types,
                &targets.formats.colors,
                blend.dual_source,
                emulation_key,
                stores,
                TextureSwizzles::new(swizzles)?,
            ))
        };
        // Step 5.
        let first_swizzles = if caps.view_swizzle {
            Vec::new()
        } else {
            self.swizzle_handles
                .get(&requests)
                .map_or_else(Vec::new, |handles| Self::swizzles(&answers, handles))
        };
        let first_stores = self
            .previous_stores
            .get(&requests)
            .copied()
            .unwrap_or(PreviousStores::EMPTY);
        let mut lookup = ctx.shaders.lookup(
            &requests,
            &key_for(first_swizzles.clone(), first_stores)?,
            &programs,
            &answers,
        );
        self.counters.lookups = self.counters.lookups.saturating_add(1);
        if let SetLookup::Ready(set) = &lookup {
            let mut swizzles = first_swizzles.clone();
            if !caps.view_swizzle {
                let handles = Self::texture_handles(&requests, set);
                swizzles = Self::swizzles(&answers, &handles);
                if self.swizzle_handles.len() < MAX_SWIZZLE_SETS
                    || self.swizzle_handles.contains_key(&requests)
                {
                    self.swizzle_handles.insert(requests.clone(), handles);
                }
            }
            let stores = Self::set_stores(&requests, set);
            if self.previous_stores.len() < self.previous_stores_limit
                || self.previous_stores.contains_key(&requests)
            {
                self.previous_stores.insert(requests.clone(), stores);
            }
            if swizzles != first_swizzles || stores != first_stores {
                lookup =
                    ctx.shaders
                        .lookup(&requests, &key_for(swizzles, stores)?, &programs, &answers);
                self.counters.lookups = self.counters.lookups.saturating_add(1);
            }
        }
        if let SetLookup::Pending(ticket) = lookup {
            match self.setting {
                WaitWhileCompiling::Skip => return Ok(Outcome::Skipped(Skip::ShaderPending)),
                WaitWhileCompiling::Wait => lookup = ctx.shaders.wait(ticket),
            }
        }
        let set = match lookup {
            SetLookup::Ready(set) => set,
            SetLookup::Pending(_) => return Ok(Outcome::Skipped(Skip::ShaderPending)),
            SetLookup::Rejected(reason) => {
                let newly = reject_set(
                    &mut self.rejected,
                    self.rejected_limit,
                    &mut self.counters,
                    &requests,
                    &reason,
                );
                return Ok(Outcome::Skipped(Skip::ShaderRejected { newly }));
            }
        };
        if set.stages.len() != set.shaders.stages.len() {
            return Err(DrawError::BadSet(
                "has stage information for a different number of shaders",
            ));
        }
        // Step 6: shaders, then resources.
        let set_stages = Self::stages(&requests, &set);
        let mut vertex_shader = None;
        let mut fragment_shader = None;
        let mut bindings = Vec::new();
        let mut binds_constants = false;
        for s in &set_stages {
            let (Some(shader), Some(info)) =
                (set.shaders.stages.get(s.index), set.stages.get(s.index))
            else {
                return Err(DrawError::Internal("a set stage index is out of range"));
            };
            let id = self.host_shaders.get(shader, info)?;
            let slot = match s.stage {
                GuestStage::Pixel => &mut fragment_shader,
                _ => &mut vertex_shader,
            };
            *slot = Some((id, info.stamp));
            for resource in &info.resources {
                let bound = match resource.source {
                    ResourceSource::ConstantBuffer { bank, size } => {
                        if bank >= root::FIRST_ROOT_BANK {
                            if self.roots.table(bank).is_none() {
                                return Err(DrawError::RootBank {
                                    stage: s.stage,
                                    bank,
                                });
                            }
                            binds_constants = true;
                            let table = u64::from(bank - root::FIRST_ROOT_BANK);
                            BindingResource::Buffer(BufferSlice {
                                buffer: constants_buffer(&mut self.constants, &*self.backend)?,
                                offset: table * ROOT_TABLE_BYTES as u64,
                                size: ROOT_TABLE_BYTES as u64,
                            })
                        } else {
                            let (address, bound) = answers
                                .constant_buffer(s.stage, bank)
                                .filter(|&(_, bound)| bound > 0)
                                .ok_or(DrawError::UnboundConstantBuffer {
                                    stage: s.stage,
                                    bank,
                                })?;
                            let len = if size == 0 { bound } else { size.min(bound) };
                            BindingResource::Buffer(ctx.buffers.bind(
                                ctx.memory,
                                address,
                                u64::from(len),
                                ctx.list,
                            )?)
                        }
                    }
                    ResourceSource::Texture(source) => {
                        let (index, _) = answers.handle_pair(handle(&answers, s.stage, source)?);
                        let words = header_words(&answers, index)?;
                        let header = TextureHeader::decode(&words)
                            .map_err(|error| DrawError::Header { index, error })?;
                        BindingResource::View(
                            ctx.textures.texture(ctx.memory, &header, ctx.list)?.view,
                        )
                    }
                    ResourceSource::Sampler(source) => {
                        let (_, index) = answers.handle_pair(handle(&answers, s.stage, source)?);
                        let words = sampler_words(&answers, index)?;
                        BindingResource::Sampler(self.samplers.get(&words)?)
                    }
                    ResourceSource::DriverConstants => {
                        binds_constants = true;
                        BindingResource::Buffer(BufferSlice {
                            buffer: constants_buffer(&mut self.constants, &*self.backend)?,
                            offset: DRIVER_CONSTANTS_OFFSET,
                            size: DRIVER_CONSTANTS_SIZE as u64,
                        })
                    }
                };
                bindings.push(Binding {
                    group: resource.slot.group,
                    binding: resource.slot.binding,
                    resource: bound,
                });
            }
        }
        let (vertex_id, vertex_stamp) =
            vertex_shader.ok_or(DrawError::BadSet("has no vertex shader"))?;
        let mut vertex_buffers = Vec::with_capacity(vertex.streams.len());
        for stream in &vertex.streams {
            vertex_buffers.push(ctx.buffers.bind(
                ctx.memory,
                stream.address,
                stream.size,
                ctx.list,
            )?);
        }
        let (index_buffer, params) = Self::params(ctx, mirror, call)?;
        let mut render_targets = RenderTargets::default();
        for (view, info) in render_targets.colors.iter_mut().zip(&targets.colors) {
            if let Some(info) = info {
                *view = Some(ctx.textures.render_target(ctx.memory, info, ctx.list)?.view);
            }
        }
        if let Some(info) = &targets.depth {
            render_targets.depth_stencil =
                Some(ctx.textures.render_target(ctx.memory, info, ctx.list)?.view);
        }
        let desc = GraphicsPipelineDesc {
            stages: Stages {
                vertex: vertex_id,
                tess_control: None,
                tess_eval: None,
                geometry: None,
                fragment: fragment_shader.map(|(id, _)| id),
            },
            vertex_input: vertex.layout.clone(),
            topology,
            primitive_restart: raster.primitive_restart,
            raster: RasterState {
                cull: raster.cull,
                front_face: raster.front_face,
                polygon_mode: raster.polygon_mode,
                depth_clamp: false,
                depth_bias,
                line_width: raster.line_width,
            },
            depth_stencil: depth_stencil.state,
            blend: blend.targets,
            logic_op: blend.logic_op,
            targets: targets.formats,
            emulation: flags,
        };
        let stamps = StageStamps {
            vertex: vertex_stamp,
            tess_control: None,
            tess_eval: None,
            geometry: None,
            fragment: fragment_shader.map(|(_, stamp): (_, ShaderStamp)| stamp),
        };
        let pipeline = match ctx.pipelines.lookup_graphics(&desc, &stamps)? {
            PipelineLookup::Ready(id) | PipelineLookup::Compiling(id) => id,
            PipelineLookup::Failed(failure) => {
                if failure.newly {
                    tracing::warn!(cause = ?failure.cause, "a draw's pipeline failed to build; its draws are skipped");
                }
                return Ok(Outcome::Skipped(Skip::PipelineFailed {
                    newly: failure.newly,
                }));
            }
        };
        // Step 7.
        if binds_constants {
            let constants = driver_constants(mirror, alpha_ref, call)?;
            upload_constants(
                ctx,
                &self.roots,
                &mut self.uploaded,
                &mut self.constants,
                &*self.backend,
                &constants,
            )?;
        }
        // Step 8.
        if self.last_targets != Some(render_targets) {
            ctx.list.push(Command::SetRenderTargets(render_targets));
            self.last_targets = Some(render_targets);
        }
        ctx.list.push(Command::Draw(DrawDesc {
            pipeline,
            bindings,
            vertex_buffers,
            index_buffer,
            dynamic: DynamicValues {
                viewports,
                scissors,
                blend_constant: blend.constant,
                stencil_reference: depth_stencil.reference,
                stencil_read_mask: depth_stencil.read_mask,
                stencil_write_mask: depth_stencil.write_mask,
                depth_bias,
                line_width: raster.line_width,
            },
            params,
        }));
        Ok(Outcome::Drawn)
    }

    /// The index buffer and counts of `call`.
    fn params<M: GpuMemory + ?Sized>(
        ctx: &mut DrawContext<'_, M>,
        mirror: &Mirror,
        call: &DrawCall,
    ) -> Result<(Option<IndexBufferBinding>, DrawParams), DrawError> {
        match call.source {
            Source::Vertices { first, count } => Ok((
                None,
                DrawParams::Direct {
                    vertex_count: count,
                    instance_count: call.instance_count,
                    first_vertex: first,
                    first_instance: call.base_instance,
                },
            )),
            Source::Indexed { first, count, size } => {
                let ib = mirror.three_d()?.index_buffer();
                let size = match size {
                    Some(size) => size,
                    None => match ib.index_size()?.value {
                        clc797::SET_INDEX_BUFFER_E_INDEX_SIZE_ONE_BYTE => IndexSize::U8,
                        clc797::SET_INDEX_BUFFER_E_INDEX_SIZE_TWO_BYTES => IndexSize::U16,
                        _ => IndexSize::U32,
                    },
                };
                let slice = ctx
                    .buffers
                    .bind(ctx.memory, ib.address()?, ib.size()?, ctx.list)?;
                Ok((
                    Some(IndexBufferBinding {
                        slice,
                        format: match size {
                            IndexSize::U8 => IndexFormat::U8,
                            IndexSize::U16 => IndexFormat::U16,
                            IndexSize::U32 => IndexFormat::U32,
                        },
                    }),
                    DrawParams::Indexed {
                        index_count: count,
                        instance_count: call.instance_count,
                        first_index: first,
                        base_vertex: i32::from_ne_bytes(call.base_vertex.to_ne_bytes()),
                        first_instance: call.base_instance,
                    },
                ))
            }
        }
    }
}

impl Drop for Draws {
    /// Destroys every host object the draw path made (the constants buffer, the host samplers
    /// and the host shaders) after the last submitted fence, so opening and closing channels does
    /// not grow the backend.
    fn drop(&mut self) {
        if let Some(buffer) = self.constants.take() {
            self.backend
                .destroy(ResourceId::Buffer(buffer), self.last_fence);
        }
        self.samplers.destroy_all(self.last_fence);
        self.host_shaders.destroy_all(self.last_fence);
        self.clears.destroy_all(self.last_fence);
    }
}

/// Refuses a draw whose dynamic values are not all finite ([`DrawError::NotFinite`]), before
/// any cache or the list is touched: the render backend refuses such a value at submit
/// (`validate_draw`, "draw state values must be finite"), which would lose every draw in the
/// list.
fn check_finite(
    viewports: &[starquake_render::command::Viewport],
    blend_constant: &[f32; 4],
    depth_bias: starquake_render::pipeline::DepthBias,
    line_width: f32,
) -> Result<(), DrawError> {
    let viewports_finite = viewports.iter().all(|v| {
        [v.x, v.y, v.width, v.height, v.min_depth, v.max_depth]
            .iter()
            .all(|x| x.is_finite())
    });
    let bias = [
        depth_bias.constant,
        depth_bias.slope_scale,
        depth_bias.clamp,
    ];
    let checks = [
        ("viewport", viewports_finite),
        (
            "blend constant",
            blend_constant.iter().all(|x| x.is_finite()),
        ),
        ("depth bias", bias.iter().all(|x| x.is_finite())),
        ("line width", line_width.is_finite()),
    ];
    match checks.into_iter().find(|&(_, finite)| !finite) {
        Some((what, _)) => Err(DrawError::NotFinite { what }),
        None => Ok(()),
    }
}

/// Remembers a rejected set and logs it the first time (module docs, step 5); whether this draw
/// is the first of the set to be skipped for it.
fn reject_set(
    rejected: &mut HashSet<Vec<ProgramRequest>>,
    limit: usize,
    counters: &mut DrawCounters,
    requests: &[ProgramRequest],
    reason: &starquake_shader::lookup::Rejection,
) -> bool {
    if rejected.contains(requests) {
        return false;
    }
    if rejected.len() >= limit {
        counters.rejected_unremembered = counters.rejected_unremembered.saturating_add(1);
        tracing::debug!(%reason, "a draw's shader set was rejected (not remembered: the table is full)");
        return false;
    }
    rejected.insert(requests.to_vec());
    counters.rejected_sets = counters.rejected_sets.saturating_add(1);
    tracing::warn!(%reason, programs = requests.len(), "a draw's shader set was rejected; its draws are skipped");
    true
}

/// The constants buffer, made through `backend` on first use.
fn constants_buffer(
    slot: &mut Option<BufferId>,
    backend: &dyn Backend,
) -> Result<BufferId, DrawError> {
    if let Some(id) = *slot {
        return Ok(id);
    }
    let id = backend
        .create_buffer(&BufferDesc {
            size: CONSTANTS_BUFFER_SIZE,
            usage: BufferUsage::UNIFORM | BufferUsage::COPY_DST,
        })
        .map_err(DrawError::Render)?;
    *slot = Some(id);
    Ok(id)
}

/// The constants buffer's bytes: the root tables, then `encoded` (the driver constants) at
/// [`DRIVER_CONSTANTS_OFFSET`], the rest zero. Built on the stack, only when uploading.
fn constants_bytes(
    roots: &RootTables,
    encoded: &[u8; DRIVER_CONSTANTS_SIZE],
) -> [u8; CONSTANTS_BUFFER_LEN] {
    let mut bytes = [0; CONSTANTS_BUFFER_LEN];
    if let Some(tables) = bytes.get_mut(..DRIVER_CONSTANTS_AT) {
        roots.write_le(tables);
    }
    if let Some(slot) =
        bytes.get_mut(DRIVER_CONSTANTS_AT..DRIVER_CONSTANTS_AT + DRIVER_CONSTANTS_SIZE)
    {
        slot.copy_from_slice(encoded);
    }
    bytes
}

/// Uploads the constants buffer when the root tables were written or the driver constants
/// changed since the last upload (module docs, "The constants buffer"): it compares the tables'
/// generation and the encoded driver constants, so a draw that uploads nothing allocates and
/// copies nothing.
fn upload_constants<M: GpuMemory + ?Sized>(
    ctx: &mut DrawContext<'_, M>,
    roots: &RootTables,
    uploaded: &mut Option<(u64, [u8; DRIVER_CONSTANTS_SIZE])>,
    buffer: &mut Option<BufferId>,
    backend: &dyn Backend,
    constants: &DriverConstants,
) -> Result<(), DrawError> {
    let encoded = constants.encode();
    let state = (roots.generation(), encoded);
    if *uploaded == Some(state) {
        return Ok(());
    }
    let bytes = constants_bytes(roots, &encoded);
    let id = constants_buffer(buffer, backend)?;
    let mut staging = backend
        .staging(CONSTANTS_BUFFER_SIZE)
        .map_err(DrawError::Render)?;
    let Some(target) = staging.bytes_mut().get_mut(..bytes.len()) else {
        return Err(DrawError::Internal(
            "a staging slice is shorter than asked for",
        ));
    };
    target.copy_from_slice(&bytes);
    ctx.list.push(Command::Upload {
        from: staging,
        to: CopyRegion::Buffer(BufferSlice {
            buffer: id,
            offset: 0,
            size: CONSTANTS_BUFFER_SIZE,
        }),
    });
    *uploaded = Some(state);
    Ok(())
}

/// The handle word at `source` for `stage`.
fn handle<M: GpuMemory + ?Sized>(
    answers: &StateAnswers<'_, M>,
    stage: GuestStage,
    source: HandleSource,
) -> Result<u32, DrawError> {
    use starquake_shader::lookup::EngineState;
    answers
        .cbuf_word(stage, source.bank, source.offset)
        .ok_or(DrawError::Handle {
            stage,
            bank: source.bank,
            offset: source.offset,
        })
}

/// The words of texture header `index`.
fn header_words<M: GpuMemory + ?Sized>(
    answers: &StateAnswers<'_, M>,
    index: u32,
) -> Result<[u32; crate::back::texture::HEADER_WORDS], DrawError> {
    let pools = answers.mirror.three_d()?.tex_pools();
    let maximum = pools.header_maximum_index()?;
    if index > maximum {
        return Err(DrawError::PoolIndex {
            pool: "texture header",
            index,
            maximum,
        });
    }
    let address = pools.header_address()?;
    let at = u64::from(index)
        .checked_mul(shaders::HEADER_BYTES)
        .and_then(|offset| address.checked_add(offset))
        .ok_or(GpuFault::OutOfRange {
            va: address,
            len: shaders::HEADER_BYTES,
        })?;
    Ok(shaders::read_words(answers.memory, at)?)
}

/// The words of sampler `index`.
fn sampler_words<M: GpuMemory + ?Sized>(
    answers: &StateAnswers<'_, M>,
    index: u32,
) -> Result<[u32; SAMPLER_WORDS], DrawError> {
    let pools = answers.mirror.three_d()?.tex_pools();
    let maximum = pools.sampler_maximum_index()?;
    if index > maximum {
        return Err(DrawError::PoolIndex {
            pool: "sampler",
            index,
            maximum,
        });
    }
    let address = pools.sampler_address()?;
    let at = u64::from(index)
        .checked_mul(SAMPLER_BYTES)
        .and_then(|offset| address.checked_add(offset))
        .ok_or(GpuFault::OutOfRange {
            va: address,
            len: SAMPLER_BYTES,
        })?;
    Ok(shaders::read_words(answers.memory, at)?)
}

/// The driver constants of a draw (module docs, step 7).
fn driver_constants(
    mirror: &Mirror,
    alpha_reference: f32,
    call: &DrawCall,
) -> Result<DriverConstants, DrawError> {
    let three_d = mirror.three_d()?;
    let select = three_d.ct_select();
    let (clip_width, clip_height) = (select.surface_clip_width()?, select.surface_clip_height()?);
    let (width, height) = if clip_width != 0 && clip_height != 0 {
        (clip_width, clip_height)
    } else {
        let ct0 = three_d.color_target(0)?;
        (ct0.width()?, ct0.height()?)
    };
    let lower_left = three_d.rasterizer().window_origin_mode()?.value
        != clc797::SET_WINDOW_ORIGIN_MODE_UPPER_LEFT;
    #[expect(
        clippy::cast_precision_loss,
        reason = "Eden-NX2 converts the render area's u32s to floats the same way"
    )]
    Ok(DriverConstants {
        alpha_reference,
        y_negate: lower_left,
        render_area_width: width as f32,
        render_area_height: height as f32,
        base_vertex: call.base_vertex,
        base_instance: call.base_instance,
    })
}
