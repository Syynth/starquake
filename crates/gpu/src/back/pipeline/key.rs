//! Pipeline keys: what of a pipeline description the cache keys on, and the checks a key passes
//! before it is saved or after it is loaded (slice G23; render-backend, "Commands" and "What a
//! pipeline is made of").
//!
//! A graphics key is a [`GraphicsPipelineDesc`] with two changes:
//!
//! - **Its shaders are [`ShaderStamp`]s, not [`ShaderId`]s.** An id is a host handle, handed out
//!   afresh on every boot, so a key list written with ids could not be used on the next one. The
//!   in-memory cache still tells pipelines apart by their [`Stages`] of ids (the parent module);
//!   only the key list uses the stamps.
//! - **Its dynamic fields are left out.** For each state in [`KEYED_DYNAMIC_STATE`] that the
//!   device sets per draw ([`starquake_render::Caps::dynamic_state`]), the description's baked
//!   value is dropped from the key, and the pipeline is created with [`CANONICAL_DEPTH_BIAS`],
//!   [`CANONICAL_LINE_WIDTH`] or [`CANONICAL_STENCIL_MASK`] in its place, so a pipeline never
//!   depends on which draw made it first. Each key carries the mask it was built under, as
//!   Eden's `FixedPipelineState` carries its dynamic-feature bits
//!   (`eden/src/video_core/renderer_vulkan/vk_pipeline_cache.cpp` @ cf654052,
//!   `LoadDiskResources`, which skips a key whose bits differ from the device's).
//!
//! Floating-point fields (depth bias, line width) are keyed by their bits, so `0.0` and `-0.0`
//! key two pipelines; that costs a pipeline, never a wrong one. Nothing else is normalised: two
//! descriptions that differ only in the factors of a blend that is off key two pipelines.
//!
//! # The dynamic states with no field in a description
//!
//! [`starquake_render::caps::DynamicState`] also names the viewport, the scissor, the blend
//! constant and the stencil reference. [`GraphicsPipelineDesc`] has no field for any of them:
//! they travel only in each draw's `DynamicValues`. So the key cannot include them, whatever
//! the device's `Caps` say, and a `Caps` without them keys exactly as one with them (pinned by
//! `tests::flags_with_no_field_in_the_description_do_not_change_the_key`). render-backend says
//! the gpu side keys "only on what is not dynamic"; for these four a device that cannot set them
//! per draw cannot be served by the description at all. That is a gap in the render types, not
//! something this cache can close, and is raised as an owner question in the pull request.
//!
//! # Checks
//!
//! [`check_graphics`] is what a key must pass to be saved and to be loaded, in this order:
//!
//! 1. the mask names only states in [`KEYED_DYNAMIC_STATE`]
//!    ([`KeyFault::UnkeyedDynamicState`]);
//! 2. each dynamic field is stored exactly when the mask says it is not dynamic, depth bias,
//!    then line width, then the stencil masks (the last only when the key has a stencil test)
//!    ([`KeyFault::DynamicFieldStored`], [`KeyFault::FixedFieldMissing`]);
//! 3. the description the key stands for passes
//!    [`starquake_render::validate::validate_graphics_pipeline`] against a device that has
//!    every feature, stage, sample count and format, with every device limit at its largest
//!    ([`KeyFault::Invalid`]). What is left are the checks that do not depend on a device: the
//!    guest's own limits (32 attributes, 32 buffers, 8 colour targets), tessellation in pairs and
//!    with patches, attribute buffers and duplicate locations, finite raster values, the sample
//!    counts any device could have, and colour and depth formats in the right slots. Whether the
//!    device of the boot that loads the key can build it is the backend's to say when the
//!    pipeline is created.
//!
//! A compute key is a stamp and has nothing to check.

use serde::{Deserialize, Serialize};
use starquake_render::caps::{
    Caps, DynamicState, Features, FormatFeatures, FormatTable, GUEST_COLOR_TARGETS, Limits,
    SampleCounts, StageSupport,
};
use starquake_render::pipeline::{
    BlendState, CullMode, DepthBias, DepthStencilState, EmulationFlags, FrontFace,
    GraphicsPipelineDesc, LogicOp, PolygonMode, RasterState, Stages, StencilFace, StencilState,
    TargetFormats, Topology, VertexLayout,
};
use starquake_render::resource::CompareFunction;
use starquake_render::validate::validate_graphics_pipeline;
use starquake_render::{RenderError, ShaderId, ShaderStage};
use thiserror::Error;

/// The dynamic states a [`GraphicsPipelineDesc`] has a baked field for, and so the only ones a
/// key can leave out: depth bias ([`RasterState::depth_bias`]), line width
/// ([`RasterState::line_width`]) and the stencil compare and write masks
/// ([`StencilState::read_mask`], [`StencilState::write_mask`]). Module docs, "The dynamic states
/// with no field in a description", says why the others are not here.
pub const KEYED_DYNAMIC_STATE: DynamicState = DynamicState::DEPTH_BIAS
    .union(DynamicState::LINE_WIDTH)
    .union(DynamicState::STENCIL_MASKS);

/// The depth bias a pipeline is created with when the device sets depth bias per draw. The
/// device ignores it; this module's choice, not a reference's.
pub const CANONICAL_DEPTH_BIAS: DepthBias = DepthBias {
    constant: 0.0,
    slope_scale: 0.0,
    clamp: 0.0,
};

/// The line width a pipeline is created with when the device sets line width per draw. The
/// device ignores it; 1.0 is Vulkan's and OpenGL's initial line width, this module's choice.
pub const CANONICAL_LINE_WIDTH: f32 = 1.0;

/// The stencil compare and write masks a pipeline is created with when the device sets them per
/// draw. The device ignores them; this module's choice.
pub const CANONICAL_STENCIL_MASK: u32 = 0;

/// A shader's identity across boots, which the caller gives with each [`ShaderId`] it looks a
/// pipeline up with.
///
/// The cache only compares stamps and writes them into the key list; on the next boot the
/// caller maps each stamp back to the shader it has made for it ([`super::PipelineCache::precompile`]).
/// What goes in it is the caller's: it must be the same for the same host shader on every boot
/// and differ between host shaders. The natural value is a digest of the shader cache's
/// shader-entry key (content hash, answers, variant key, host target and translator version;
/// shader-pipeline, "Identity, caching and invalidation", layer 3), which shader slice 21
/// defines. That the width is 128 bits, the width of that spec's content hash, is a **guess**
/// at what the caller will have; the stamp is otherwise opaque here.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ShaderStamp(u128);

impl ShaderStamp {
    /// The stamp `value`.
    pub const fn new(value: u128) -> Self {
        Self(value)
    }

    /// The stamp's value.
    pub const fn get(self) -> u128 {
        self.0
    }
}

/// The stamps of a graphics pipeline's shaders, one per stage present in its [`Stages`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct StageStamps {
    /// The vertex shader's stamp.
    pub vertex: ShaderStamp,
    /// The tessellation control shader's stamp, present exactly when that shader is.
    pub tess_control: Option<ShaderStamp>,
    /// The tessellation evaluation shader's stamp, present exactly when that shader is.
    pub tess_eval: Option<ShaderStamp>,
    /// The geometry shader's stamp, present exactly when that shader is.
    pub geometry: Option<ShaderStamp>,
    /// The fragment shader's stamp, present exactly when that shader is.
    pub fragment: Option<ShaderStamp>,
}

impl StageStamps {
    /// The optional stages, in the order they are checked: each with its stamp.
    fn optional(&self) -> [(ShaderStage, Option<ShaderStamp>); 4] {
        [
            (ShaderStage::TessControl, self.tess_control),
            (ShaderStage::TessEval, self.tess_eval),
            (ShaderStage::Geometry, self.geometry),
            (ShaderStage::Fragment, self.fragment),
        ]
    }
}

/// The optional stages of `stages`, in the order [`StageStamps::optional`] has them.
fn optional_ids(stages: &Stages) -> [Option<ShaderId>; 4] {
    [
        stages.tess_control,
        stages.tess_eval,
        stages.geometry,
        stages.fragment,
    ]
}

/// The first optional stage, in the order tessellation control, tessellation evaluation,
/// geometry, fragment, that has a shader but no stamp or a stamp but no shader.
pub(super) fn mismatched_stage(stages: &Stages, stamps: &StageStamps) -> Option<ShaderStage> {
    stamps
        .optional()
        .into_iter()
        .zip(optional_ids(stages))
        .find(|((_, stamp), id)| stamp.is_some() != id.is_some())
        .map(|((stage, _), _)| stage)
}

/// The shaders `stamps` stand for, as `shader` maps each stamp, or `None` when it maps one to
/// nothing. Stamps are mapped in stage order and the first miss stops the walk.
pub(super) fn resolve(
    stamps: &StageStamps,
    shader: &mut dyn FnMut(ShaderStamp) -> Option<ShaderId>,
) -> Option<Stages> {
    let vertex = shader(stamps.vertex)?;
    let mut one = |stamp: Option<ShaderStamp>| match stamp {
        Some(stamp) => shader(stamp).map(Some),
        None => Some(None),
    };
    Some(Stages {
        vertex,
        tess_control: one(stamps.tess_control)?,
        tess_eval: one(stamps.tess_eval)?,
        geometry: one(stamps.geometry)?,
        fragment: one(stamps.fragment)?,
    })
}

/// A depth bias, keyed by the bits of its three numbers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub(super) struct BiasBits {
    pub(super) constant: u32,
    pub(super) slope_scale: u32,
    pub(super) clamp: u32,
}

impl BiasBits {
    fn of(bias: DepthBias) -> Self {
        Self {
            constant: bias.constant.to_bits(),
            slope_scale: bias.slope_scale.to_bits(),
            clamp: bias.clamp.to_bits(),
        }
    }

    fn bias(self) -> DepthBias {
        DepthBias {
            constant: f32::from_bits(self.constant),
            slope_scale: f32::from_bits(self.slope_scale),
            clamp: f32::from_bits(self.clamp),
        }
    }
}

/// The stencil compare and write masks.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub(super) struct StencilMasks {
    pub(super) read: u32,
    pub(super) write: u32,
}

/// A stencil test, with its masks when they are not dynamic.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub(super) struct StencilKey {
    pub(super) front: StencilFace,
    pub(super) back: StencilFace,
    pub(super) masks: Option<StencilMasks>,
}

/// Everything of a [`GraphicsPipelineDesc`] but its shaders, with the dynamic fields left out
/// (module docs). The field order is the key list's layout ([`super::list`]).
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub(super) struct FixedState {
    /// The dynamic states the key was built under, within [`KEYED_DYNAMIC_STATE`].
    pub(super) dynamic: DynamicState,
    pub(super) vertex_input: VertexLayout,
    pub(super) topology: Topology,
    pub(super) primitive_restart: bool,
    pub(super) cull: CullMode,
    pub(super) front_face: FrontFace,
    pub(super) polygon_mode: PolygonMode,
    pub(super) depth_clamp: bool,
    /// `None` when depth bias is dynamic.
    pub(super) depth_bias: Option<BiasBits>,
    /// The line width's bits; `None` when line width is dynamic.
    pub(super) line_width: Option<u32>,
    pub(super) depth_test: Option<CompareFunction>,
    pub(super) depth_write: bool,
    pub(super) stencil: Option<StencilKey>,
    pub(super) blend: [BlendState; GUEST_COLOR_TARGETS],
    pub(super) logic_op: Option<LogicOp>,
    pub(super) targets: TargetFormats,
    pub(super) emulation: EmulationFlags,
}

impl FixedState {
    /// The key of `desc` on a device that sets `dynamic` per draw. States outside
    /// [`KEYED_DYNAMIC_STATE`] are ignored (module docs).
    pub(super) fn of(desc: &GraphicsPipelineDesc, dynamic: DynamicState) -> Self {
        let dynamic = Self::keyed_dynamic(dynamic);
        let fixed = |state: DynamicState| !dynamic.contains(state);
        let raster = &desc.raster;
        let ds = &desc.depth_stencil;
        Self {
            dynamic,
            vertex_input: desc.vertex_input.clone(),
            topology: desc.topology,
            primitive_restart: desc.primitive_restart,
            cull: raster.cull,
            front_face: raster.front_face,
            polygon_mode: raster.polygon_mode,
            depth_clamp: raster.depth_clamp,
            depth_bias: fixed(DynamicState::DEPTH_BIAS).then(|| BiasBits::of(raster.depth_bias)),
            line_width: fixed(DynamicState::LINE_WIDTH).then(|| raster.line_width.to_bits()),
            depth_test: ds.depth_test,
            depth_write: ds.depth_write,
            stencil: ds.stencil.map(|s| StencilKey {
                front: s.front,
                back: s.back,
                masks: fixed(DynamicState::STENCIL_MASKS).then_some(StencilMasks {
                    read: s.read_mask,
                    write: s.write_mask,
                }),
            }),
            blend: desc.blend,
            logic_op: desc.logic_op,
            targets: desc.targets,
            emulation: desc.emulation,
        }
    }

    /// The states of `dynamic` a key can leave out: those in [`KEYED_DYNAMIC_STATE`].
    pub(super) fn keyed_dynamic(dynamic: DynamicState) -> DynamicState {
        dynamic.difference(DynamicState::ALL.difference(KEYED_DYNAMIC_STATE))
    }

    /// The description the key stands for with `stages` as its shaders: the canonical baked
    /// values in place of the fields left out.
    pub(super) fn desc(&self, stages: Stages) -> GraphicsPipelineDesc {
        let masks = |s: &StencilKey| {
            s.masks.unwrap_or(StencilMasks {
                read: CANONICAL_STENCIL_MASK,
                write: CANONICAL_STENCIL_MASK,
            })
        };
        GraphicsPipelineDesc {
            stages,
            vertex_input: self.vertex_input.clone(),
            topology: self.topology,
            primitive_restart: self.primitive_restart,
            raster: RasterState {
                cull: self.cull,
                front_face: self.front_face,
                polygon_mode: self.polygon_mode,
                depth_clamp: self.depth_clamp,
                depth_bias: self.depth_bias.map_or(CANONICAL_DEPTH_BIAS, BiasBits::bias),
                line_width: self.line_width.map_or(CANONICAL_LINE_WIDTH, f32::from_bits),
            },
            depth_stencil: DepthStencilState {
                depth_test: self.depth_test,
                depth_write: self.depth_write,
                stencil: self.stencil.map(|s| StencilState {
                    front: s.front,
                    back: s.back,
                    read_mask: masks(&s).read,
                    write_mask: masks(&s).write,
                }),
            },
            blend: self.blend,
            logic_op: self.logic_op,
            targets: self.targets,
            emulation: self.emulation,
        }
    }
}

/// A graphics pipeline as the key list holds it: shaders by stamp, state without its dynamic
/// fields.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub(super) struct GraphicsKey {
    pub(super) stamps: StageStamps,
    pub(super) fixed: FixedState,
}

/// One key of the key list. The variant order is the list's layout.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub(super) enum PipelineKey {
    /// A graphics pipeline.
    Graphics(Box<GraphicsKey>),
    /// A compute pipeline: its shader's stamp.
    Compute(ShaderStamp),
}

/// A dynamic field of a graphics key.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum KeyField {
    /// The depth bias.
    DepthBias,
    /// The line width.
    LineWidth,
    /// The stencil compare and write masks.
    StencilMasks,
}

impl KeyField {
    /// The field's name in plain words, as errors print it.
    pub const fn name(self) -> &'static str {
        match self {
            Self::DepthBias => "depth bias",
            Self::LineWidth => "line width",
            Self::StencilMasks => "stencil masks",
        }
    }
}

/// Why a key cannot be saved or loaded (module docs, "Checks").
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum KeyFault {
    /// The key's dynamic-state mask names states a description has no field for.
    #[error("its dynamic-state mask has bits {bits:#x} for state a pipeline has no field for")]
    UnkeyedDynamicState {
        /// The bits outside [`KEYED_DYNAMIC_STATE`].
        bits: u32,
    },
    /// The key stores a field its mask says is dynamic.
    #[error("it stores a {} although its mask says that is dynamic", .0.name())]
    DynamicFieldStored(KeyField),
    /// The key lacks a field its mask says is not dynamic.
    #[error("it lacks a {} although its mask says that is not dynamic", .0.name())]
    FixedFieldMissing(KeyField),
    /// The description the key stands for breaks a rule no device lifts.
    #[error("its pipeline breaks a rule: {0}")]
    Invalid(RenderError),
}

/// Checks a graphics key, as saving and loading both do (module docs, "Checks").
pub(super) fn check_graphics(key: &GraphicsKey) -> Result<(), KeyFault> {
    let fixed = &key.fixed;
    let unkeyed = fixed.dynamic.bits() & !KEYED_DYNAMIC_STATE.bits();
    if unkeyed != 0 {
        return Err(KeyFault::UnkeyedDynamicState { bits: unkeyed });
    }
    let stencil_masks = fixed.stencil.map(|s| s.masks.is_some());
    for (field, state, stored) in [
        (
            KeyField::DepthBias,
            DynamicState::DEPTH_BIAS,
            Some(fixed.depth_bias.is_some()),
        ),
        (
            KeyField::LineWidth,
            DynamicState::LINE_WIDTH,
            Some(fixed.line_width.is_some()),
        ),
        (
            KeyField::StencilMasks,
            DynamicState::STENCIL_MASKS,
            stencil_masks,
        ),
    ] {
        let Some(stored) = stored else { continue };
        match (fixed.dynamic.contains(state), stored) {
            (true, true) => return Err(KeyFault::DynamicFieldStored(field)),
            (false, false) => return Err(KeyFault::FixedFieldMissing(field)),
            _ => {}
        }
    }
    // The checks look at which stages are present, not at which shaders they are.
    let placeholder = ShaderId::new(0, 0);
    let stamps = &key.stamps;
    let id = |stamp: Option<ShaderStamp>| stamp.map(|_| placeholder);
    let stages = Stages {
        vertex: placeholder,
        tess_control: id(stamps.tess_control),
        tess_eval: id(stamps.tess_eval),
        geometry: id(stamps.geometry),
        fragment: id(stamps.fragment),
    };
    validate_graphics_pipeline(&ANY_DEVICE, &fixed.desc(stages)).map_err(KeyFault::Invalid)
}

/// A device that can do everything: what [`check_graphics`] validates against, so that only the
/// checks no device lifts remain.
static ANY_DEVICE: Caps = Caps {
    limits: Limits {
        max_buffer_size: u64::MAX,
        max_staging_size: u64::MAX,
        max_texture_dimension_1d: u32::MAX,
        max_texture_dimension_2d: u32::MAX,
        max_texture_dimension_3d: u32::MAX,
        max_texture_array_layers: u32::MAX,
        max_sampler_anisotropy: u32::MAX,
        max_vertex_attributes: u32::MAX,
        max_vertex_buffers: u32::MAX,
        max_vertex_buffer_stride: u32::MAX,
        max_color_attachments: u32::MAX,
        max_viewports: u32::MAX,
        max_bind_groups: u32::MAX,
        max_bindings_per_bind_group: u32::MAX,
        max_compute_workgroups_per_dimension: u32::MAX,
        max_uniform_buffers_per_stage: u32::MAX,
        max_storage_buffers_per_stage: u32::MAX,
        max_sampled_textures_per_stage: u32::MAX,
        max_storage_textures_per_stage: u32::MAX,
        max_samplers_per_stage: u32::MAX,
    },
    features: Features::ALL,
    stages: StageSupport::ALL,
    dynamic_state: DynamicState::ALL,
    sample_counts: SampleCounts::ALL,
    view_swizzle: true,
    formats: FormatTable::uniform(FormatFeatures::ALL),
};
