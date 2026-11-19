//! The fixed-function parts of a draw, each derived from its state groups and kept until one of
//! them is dirty again (gpu-protocol spec, "3D: from state to a draw" step 1).
//!
//! [`Parts`] holds one part per group set: the viewports ([`GroupKind::Viewport`]), the
//! scissors ([`GroupKind::Scissor`] and, for the window origin flip, `Viewport`), the blend
//! state ([`GroupKind::Blend`] and every [`GroupKind::BlendPerTarget`]), the rasterizer
//! ([`GroupKind::Rasterizer`]), depth and stencil ([`GroupKind::DepthStencil`]) and the vertex
//! input ([`GroupKind::VertexAttribute`] and every [`GroupKind::VertexStream`]). A part is
//! derived again only when a group it reads is in the dirty set handed to [`Parts::refresh`];
//! [`Parts::derivations`] counts each part's derivations, so a test can see that a clean group
//! was not re-derived.
//!
//! Every value is read through the typed views of [`crate::back::state`]; the enums are mapped
//! by the names the class header gives them (`classes/clc797.h`), which every field sharing an
//! enum (`SET_BLEND_*` and `SET_BLEND_PER_TARGET_*`; the depth, stencil and alpha functions)
//! gives alike.
//!
//! # Sources
//!
//! Paths are relative to the reference folder (`tools/refs-path.sh`), at the commits pinned in
//! `docs/references.md`; Eden is GPL-3.0-or-later.
//!
//! - Viewports and scissors: Eden's `GetViewportState` and `GetScissorState`
//!   (`eden/src/video_core/renderer_vulkan/vk_rasterizer.cpp` @ cf654052), at resolution scale
//!   1: x and y from the offset minus the scale, width and height twice the scale, the lower-left
//!   window origin flipped by the surface clip height, a negative y swizzle flipped by the
//!   viewport height (the path Eden takes without `NV_viewport_swizzle`, which no wgpu host has),
//!   a zero width or height made 1, depth from the offset and scale with `reduce_z` from the clip
//!   range, clamped to 0..1 (the path Eden takes without `EXT_depth_range_unrestricted`, which
//!   wgpu does not expose). A disabled scissor is the whole target.
//! - Blending: Eden's `BlendingAttachment::Refresh` (`fixed_pipeline_state.cpp`): the write mask
//!   from `SET_CT_WRITE(0)` for every target when `SET_SINGLE_CT_WRITE_CONTROL` is on, factors
//!   only for a target whose `SET_BLEND(j)` is on, from the per-target fields when
//!   `SET_BLEND_STATE_PER_TARGET` is on, else from the shared ones. Dual-source blending is on
//!   when a factor of target 0's equation reads the second source (Eden's
//!   `uses_dual_source`, same file).
//! - Rasterizer: culling, front face, polygon mode (the front mode, as Eden packs
//!   `polygon_mode_front`), primitive restart, and depth bias applied only when the polygon offset
//!   enable of the topology's class is on (Eden's `POLYGON_OFFSET_ENABLE_LUT`).
//! - Depth and stencil: Eden's `FixedPipelineState::DynamicState` fields, with the back face
//!   taking the front face's state when `SET_TWO_SIDED_STENCIL_TEST` is off.
//! - Vertex input: one buffer per enabled stream and one attribute per active attribute, its
//!   location the attribute's index (Eden's `VertexAttribute` loop in `vk_graphics_pipeline.cpp`).
//!
//! # Not read
//!
//! `SET_VIEWPORT_CLIP_CONTROL` (depth clamp is off), the viewport x, z and w swizzles, the back
//! polygon mode, the line stipple, conservative raster, anti-aliased lines, `SET_ITERATED_BLEND`
//! (Eden's squashed-blend workaround is a setting), alpha to coverage and to one, and the
//! global colour key.

use starquake_render::caps::GUEST_COLOR_TARGETS;
use starquake_render::command::{Rect, Viewport};
use starquake_render::pipeline::{
    BlendComponent, BlendFactor, BlendOperation, BlendState, ColorWrites, CullMode, DepthBias,
    DepthStencilState, FrontFace, LogicOp, PolygonMode, StencilFace, StencilOperation,
    StencilState, Topology, VertexAttribute as HostAttribute, VertexBufferLayout, VertexComponents,
    VertexKind, VertexLayout, VertexStepMode,
};
use starquake_render::resource::CompareFunction;
use starquake_shader::variant::AttributeType;

use super::DrawError;
use crate::back::state::{
    BLEND_TARGETS, Blend, BlendPerTarget, Group, GroupKind, GroupSet, Mirror, SCISSORS, StateError,
    ThreeD, VERTEX_ATTRIBUTES, VERTEX_STREAMS, VIEWPORTS,
};
use crate::classes::{Value, clc797};

/// The parts [`Parts`] keeps, in [`Parts::derivations`] order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Part {
    /// The viewports.
    Viewports,
    /// The scissors.
    Scissors,
    /// The blend state.
    Blend,
    /// The rasterizer state.
    Raster,
    /// Depth and stencil.
    DepthStencil,
    /// The vertex input.
    Vertex,
}

impl Part {
    /// Every part, in counter order.
    pub const ALL: [Self; 6] = [
        Self::Viewports,
        Self::Scissors,
        Self::Blend,
        Self::Raster,
        Self::DepthStencil,
        Self::Vertex,
    ];

    /// Whether `dirty` holds a group this part reads.
    fn is_dirty(self, dirty: GroupSet) -> bool {
        dirty.iter().any(|group| self.reads(group.kind()))
    }

    /// Whether this part reads groups of `kind`.
    const fn reads(self, kind: GroupKind) -> bool {
        matches!(
            (self, kind),
            (Self::Viewports, GroupKind::Viewport)
                | (Self::Scissors, GroupKind::Scissor | GroupKind::Viewport)
                | (Self::Blend, GroupKind::Blend | GroupKind::BlendPerTarget)
                | (Self::Raster, GroupKind::Rasterizer)
                | (Self::DepthStencil, GroupKind::DepthStencil)
                | (
                    Self::Vertex,
                    GroupKind::VertexAttribute | GroupKind::VertexStream
                )
        )
    }

    const fn index(self) -> usize {
        self as usize
    }
}

/// The blend part.
#[derive(Clone, Debug, PartialEq)]
pub struct BlendPart {
    /// Blending per colour target.
    pub targets: [BlendState; GUEST_COLOR_TARGETS],
    /// The logic operation, when on.
    pub logic_op: Option<LogicOp>,
    /// The blend constant.
    pub constant: [f32; 4],
    /// Whether target 0's equation reads the second source.
    pub dual_source: bool,
}

/// The rasterizer part.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RasterPart {
    /// Culling.
    pub cull: CullMode,
    /// Front face.
    pub front_face: FrontFace,
    /// The front polygon mode.
    pub polygon_mode: PolygonMode,
    /// Primitive restart.
    pub primitive_restart: bool,
    /// The bias registers, before the topology's polygon offset enable is applied.
    pub depth_bias: DepthBias,
    /// Polygon offset enables: points, lines, fill.
    pub offset_enables: [bool; 3],
    /// Line width.
    pub line_width: f32,
    /// `SET_RASTER_ENABLE`.
    pub raster_enable: bool,
}

/// The depth and stencil part.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DepthStencilPart {
    /// The pipeline's depth and stencil state.
    pub state: DepthStencilState,
    /// The stencil reference.
    pub reference: u32,
    /// The stencil compare mask.
    pub read_mask: u32,
    /// The stencil write mask.
    pub write_mask: u32,
}

/// One enabled vertex stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stream {
    /// The stream's index.
    pub index: u32,
    /// Its GPU address.
    pub address: u64,
    /// Its size in bytes.
    pub size: u64,
}

/// The vertex input part.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VertexPart {
    /// The pipeline's vertex input; buffer `i` is `streams[i]`.
    pub layout: VertexLayout,
    /// The enabled streams, in index order.
    pub streams: Vec<Stream>,
    /// Each attribute's component type for the variant key, `Disabled` when inactive.
    pub types: [AttributeType; VERTEX_ATTRIBUTES as usize],
}

/// The parts, each kept until a group it reads is dirty (module docs).
#[derive(Clone, Debug, Default)]
pub struct Parts {
    viewports: Option<Vec<Viewport>>,
    scissors: Option<Vec<Rect>>,
    blend: Option<BlendPart>,
    raster: Option<RasterPart>,
    depth_stencil: Option<DepthStencilPart>,
    vertex: Option<VertexPart>,
    derivations: [u64; Part::ALL.len()],
}

/// The parts of one draw, borrowed from [`Parts`].
#[derive(Clone, Copy, Debug)]
pub struct Current<'a> {
    /// The viewports.
    pub viewports: &'a [Viewport],
    /// The scissors.
    pub scissors: &'a [Rect],
    /// The blend state.
    pub blend: &'a BlendPart,
    /// The rasterizer state.
    pub raster: &'a RasterPart,
    /// Depth and stencil.
    pub depth_stencil: &'a DepthStencilPart,
    /// The vertex input.
    pub vertex: &'a VertexPart,
}

impl Parts {
    /// No part derived yet.
    pub fn new() -> Self {
        Self::default()
    }

    /// How many times each part was derived, in [`Part::ALL`] order.
    pub fn derivations(&self) -> [u64; Part::ALL.len()] {
        self.derivations
    }

    /// Derives every part that is missing or reads a group in `dirty`, and returns them all.
    /// On an error, every part derived before it is kept and the failing part stays missing, so
    /// the next refresh derives it again.
    pub fn refresh(&mut self, mirror: &Mirror, dirty: GroupSet) -> Result<Current<'_>, DrawError> {
        let three_d = mirror.three_d()?;
        let stale = |part: Part, have: bool| !have || part.is_dirty(dirty);
        if stale(Part::Viewports, self.viewports.is_some()) {
            self.viewports = None;
            self.viewports = Some(viewports(mirror, three_d)?);
            self.count(Part::Viewports);
        }
        if stale(Part::Scissors, self.scissors.is_some()) {
            self.scissors = None;
            self.scissors = Some(scissors(three_d)?);
            self.count(Part::Scissors);
        }
        if stale(Part::Blend, self.blend.is_some()) {
            self.blend = None;
            self.blend = Some(blend(three_d)?);
            self.count(Part::Blend);
        }
        if stale(Part::Raster, self.raster.is_some()) {
            self.raster = None;
            self.raster = Some(raster(three_d)?);
            self.count(Part::Raster);
        }
        if stale(Part::DepthStencil, self.depth_stencil.is_some()) {
            self.depth_stencil = None;
            self.depth_stencil = Some(depth_stencil(three_d)?);
            self.count(Part::DepthStencil);
        }
        if stale(Part::Vertex, self.vertex.is_some()) {
            self.vertex = None;
            self.vertex = Some(vertex(three_d)?);
            self.count(Part::Vertex);
        }
        match (
            &self.viewports,
            &self.scissors,
            &self.blend,
            &self.raster,
            &self.depth_stencil,
            &self.vertex,
        ) {
            (
                Some(viewports),
                Some(scissors),
                Some(blend),
                Some(raster),
                Some(ds),
                Some(vertex),
            ) => Ok(Current {
                viewports,
                scissors,
                blend,
                raster,
                depth_stencil: ds,
                vertex,
            }),
            // Each part was set above or kept from before; none can be missing here.
            _ => Err(DrawError::Internal("a derived part is missing")),
        }
    }

    fn count(&mut self, part: Part) {
        if let Some(slot) = self.derivations.get_mut(part.index()) {
            *slot = slot.saturating_add(1);
        }
    }
}

/// The groups the parts read, for tests: every group of every kind a [`Part`] reads.
pub fn part_groups(part: Part) -> GroupSet {
    GroupKind::ALL
        .iter()
        .filter(|&&kind| part.reads(kind))
        .flat_map(|&kind| (0..kind.count()).filter_map(move |i| Group::new(kind, i)))
        .collect()
}

// ---------------------------------------------------------------------------------------------
// Enum mappings, by the header's value names.

/// The error for a value name no arm maps: the getter accepted it, so the class header defines
/// it, but this mapping has no host equivalent.
fn unmapped(value: &'static Value, what: &'static str) -> DrawError {
    DrawError::Unmapped {
        what,
        name: value.name,
    }
}

/// A topology from `BEGIN_OP`'s values (shared by every draw method's topology field).
pub fn topology(value: u32) -> Result<Topology, DrawError> {
    Ok(match value {
        clc797::BEGIN_OP_POINTS => Topology::Points,
        clc797::BEGIN_OP_LINES => Topology::Lines,
        clc797::BEGIN_OP_LINE_LOOP => Topology::LineLoop,
        clc797::BEGIN_OP_LINE_STRIP => Topology::LineStrip,
        clc797::BEGIN_OP_TRIANGLES => Topology::Triangles,
        clc797::BEGIN_OP_TRIANGLE_STRIP => Topology::TriangleStrip,
        clc797::BEGIN_OP_TRIANGLE_FAN => Topology::TriangleFan,
        clc797::BEGIN_OP_QUADS => Topology::Quads,
        clc797::BEGIN_OP_QUAD_STRIP => Topology::QuadStrip,
        clc797::BEGIN_OP_POLYGON => Topology::Polygon,
        clc797::BEGIN_OP_LINELIST_ADJCY => Topology::LinesAdjacency,
        clc797::BEGIN_OP_LINESTRIP_ADJCY => Topology::LineStripAdjacency,
        clc797::BEGIN_OP_TRIANGLELIST_ADJCY => Topology::TrianglesAdjacency,
        clc797::BEGIN_OP_TRIANGLESTRIP_ADJCY => Topology::TriangleStripAdjacency,
        clc797::BEGIN_OP_PATCH => Topology::Patches,
        other => return Err(DrawError::Topology { value: other }),
    })
}

/// The draw's topology after `SET_PRIMITIVE_TOPOLOGY_CONTROL` (Eden's `UpdateTopology`,
/// `eden-nx2/src/video_core/engines/draw_manager.cpp`): with the separate state chosen,
/// `SET_PRIMITIVE_TOPOLOGY`'s point, line and line strip values map to their topologies and its
/// other non-legacy values are `BEGIN_OP`'s; a legacy value is refused (module docs of `call`).
pub fn effective_topology(mirror: &Mirror, begin: u32) -> Result<Topology, DrawError> {
    let control = mirror.reg(&clc797::SET_PRIMITIVE_TOPOLOGY_CONTROL, &[])?;
    if control.field(clc797::SET_PRIMITIVE_TOPOLOGY_CONTROL_OVERRIDE)?
        != clc797::SET_PRIMITIVE_TOPOLOGY_CONTROL_OVERRIDE_USE_SEPARATE_TOPOLOGY_STATE
    {
        return topology(begin);
    }
    let value = mirror
        .reg(&clc797::SET_PRIMITIVE_TOPOLOGY, &[])?
        .field(clc797::SET_PRIMITIVE_TOPOLOGY_V)?;
    match value {
        clc797::SET_PRIMITIVE_TOPOLOGY_V_POINTLIST => Ok(Topology::Points),
        clc797::SET_PRIMITIVE_TOPOLOGY_V_LINELIST => Ok(Topology::Lines),
        clc797::SET_PRIMITIVE_TOPOLOGY_V_LINESTRIP => Ok(Topology::LineStrip),
        clc797::SET_PRIMITIVE_TOPOLOGY_V_TRIANGLELIST
        | clc797::SET_PRIMITIVE_TOPOLOGY_V_TRIANGLESTRIP
        | clc797::SET_PRIMITIVE_TOPOLOGY_V_LINELIST_ADJCY
        | clc797::SET_PRIMITIVE_TOPOLOGY_V_LINESTRIP_ADJCY
        | clc797::SET_PRIMITIVE_TOPOLOGY_V_TRIANGLELIST_ADJCY
        | clc797::SET_PRIMITIVE_TOPOLOGY_V_TRIANGLESTRIP_ADJCY
        | clc797::SET_PRIMITIVE_TOPOLOGY_V_PATCHLIST => topology(value),
        other => Err(DrawError::Topology { value: other }),
    }
}

/// A comparison, from the names `SET_DEPTH_FUNC`, `SET_STENCIL_FUNC` and `SET_ALPHA_FUNC` share.
pub fn compare(value: &'static Value) -> Result<CompareFunction, DrawError> {
    Ok(match value.name {
        "OGL_NEVER" | "D3D_NEVER" => CompareFunction::Never,
        "OGL_LESS" | "D3D_LESS" => CompareFunction::Less,
        "OGL_EQUAL" | "D3D_EQUAL" => CompareFunction::Equal,
        "OGL_LEQUAL" | "D3D_LESSEQUAL" => CompareFunction::LessEqual,
        "OGL_GREATER" | "D3D_GREATER" => CompareFunction::Greater,
        "OGL_NOTEQUAL" | "D3D_NOTEQUAL" => CompareFunction::NotEqual,
        "OGL_GEQUAL" | "D3D_GREATEREQUAL" => CompareFunction::GreaterEqual,
        "OGL_ALWAYS" | "D3D_ALWAYS" => CompareFunction::Always,
        _ => return Err(unmapped(value, "comparison")),
    })
}

/// A blend factor, from the names every `*_COEFF` field shares. Direct3D's `BOTHSRCALPHA` and
/// `BOTHINVSRCALPHA` have no host equivalent and are refused.
pub fn blend_factor(value: &'static Value) -> Result<BlendFactor, DrawError> {
    Ok(match value.name {
        "OGL_ZERO" | "D3D_ZERO" => BlendFactor::Zero,
        "OGL_ONE" | "D3D_ONE" => BlendFactor::One,
        "OGL_SRC_COLOR" | "D3D_SRCCOLOR" => BlendFactor::Src,
        "OGL_ONE_MINUS_SRC_COLOR" | "D3D_INVSRCCOLOR" => BlendFactor::OneMinusSrc,
        "OGL_SRC_ALPHA" | "D3D_SRCALPHA" => BlendFactor::SrcAlpha,
        "OGL_ONE_MINUS_SRC_ALPHA" | "D3D_INVSRCALPHA" => BlendFactor::OneMinusSrcAlpha,
        "OGL_DST_ALPHA" | "D3D_DESTALPHA" => BlendFactor::DstAlpha,
        "OGL_ONE_MINUS_DST_ALPHA" | "D3D_INVDESTALPHA" => BlendFactor::OneMinusDstAlpha,
        "OGL_DST_COLOR" | "D3D_DESTCOLOR" => BlendFactor::Dst,
        "OGL_ONE_MINUS_DST_COLOR" | "D3D_INVDESTCOLOR" => BlendFactor::OneMinusDst,
        "OGL_SRC_ALPHA_SATURATE" | "D3D_SRCALPHASAT" => BlendFactor::SrcAlphaSaturated,
        "OGL_CONSTANT_COLOR" | "OGL_CONSTANT_ALPHA" | "D3D_BLENDFACTOR" => BlendFactor::Constant,
        "OGL_ONE_MINUS_CONSTANT_COLOR" | "OGL_ONE_MINUS_CONSTANT_ALPHA" | "D3D_INVBLENDFACTOR" => {
            BlendFactor::OneMinusConstant
        }
        "OGL_SRC1COLOR" | "D3D_SRC1COLOR" => BlendFactor::Src1,
        "OGL_INVSRC1COLOR" | "D3D_INVSRC1COLOR" => BlendFactor::OneMinusSrc1,
        "OGL_SRC1ALPHA" | "D3D_SRC1ALPHA" => BlendFactor::Src1Alpha,
        "OGL_INVSRC1ALPHA" | "D3D_INVSRC1ALPHA" => BlendFactor::OneMinusSrc1Alpha,
        _ => return Err(unmapped(value, "blend factor")),
    })
}

/// A blend equation, from the names every `*_OP` field shares.
pub fn blend_op(value: &'static Value) -> Result<BlendOperation, DrawError> {
    Ok(match value.name {
        "OGL_FUNC_ADD" | "D3D_ADD" => BlendOperation::Add,
        "OGL_FUNC_SUBTRACT" | "D3D_SUBTRACT" => BlendOperation::Subtract,
        "OGL_FUNC_REVERSE_SUBTRACT" | "D3D_REVSUBTRACT" => BlendOperation::ReverseSubtract,
        "OGL_MIN" | "D3D_MIN" => BlendOperation::Min,
        "OGL_MAX" | "D3D_MAX" => BlendOperation::Max,
        _ => return Err(unmapped(value, "blend equation")),
    })
}

/// A stencil operation, from the names every `*_STENCIL_OP_*` field shares.
pub fn stencil_op(value: &'static Value) -> Result<StencilOperation, DrawError> {
    Ok(match value.name {
        "OGL_KEEP" | "D3D_KEEP" => StencilOperation::Keep,
        "OGL_ZERO" | "D3D_ZERO" => StencilOperation::Zero,
        "OGL_REPLACE" | "D3D_REPLACE" => StencilOperation::Replace,
        "OGL_INCRSAT" | "D3D_INCRSAT" => StencilOperation::IncrementClamp,
        "OGL_DECRSAT" | "D3D_DECRSAT" => StencilOperation::DecrementClamp,
        "OGL_INVERT" | "D3D_INVERT" => StencilOperation::Invert,
        "OGL_INCR" | "D3D_INCR" => StencilOperation::IncrementWrap,
        "OGL_DECR" | "D3D_DECR" => StencilOperation::DecrementWrap,
        _ => return Err(unmapped(value, "stencil operation")),
    })
}

/// A logic operation, from `SET_LOGIC_OP_FUNC`'s names.
pub fn logic_op(value: &'static Value) -> Result<LogicOp, DrawError> {
    Ok(match value.name {
        "CLEAR" => LogicOp::Clear,
        "AND" => LogicOp::And,
        "AND_REVERSE" => LogicOp::AndReverse,
        "COPY" => LogicOp::Copy,
        "AND_INVERTED" => LogicOp::AndInverted,
        "NOOP" => LogicOp::NoOp,
        "XOR" => LogicOp::Xor,
        "OR" => LogicOp::Or,
        "NOR" => LogicOp::Nor,
        "EQUIV" => LogicOp::Equivalent,
        "INVERT" => LogicOp::Invert,
        "OR_REVERSE" => LogicOp::OrReverse,
        "COPY_INVERTED" => LogicOp::CopyInverted,
        "OR_INVERTED" => LogicOp::OrInverted,
        "NAND" => LogicOp::Nand,
        "SET" => LogicOp::Set,
        _ => return Err(unmapped(value, "logic operation")),
    })
}

/// Whether a `TRUE`/`FALSE` (or `ENABLE`) field is on.
fn on(value: &'static Value) -> bool {
    value.name == "TRUE"
}

// ---------------------------------------------------------------------------------------------
// Parts.

/// The surface clip height, as Eden's viewport and scissor flips read it.
fn clip_height(three_d: ThreeD<'_>) -> Result<u32, StateError> {
    three_d.ct_select().surface_clip_height()
}

/// Whether the window origin is the lower left.
fn lower_left(three_d: ThreeD<'_>) -> Result<bool, StateError> {
    Ok(three_d.rasterizer().window_origin_mode()?.value
        != clc797::SET_WINDOW_ORIGIN_MODE_UPPER_LEFT)
}

/// Every viewport (module docs, "Sources").
fn viewports(mirror: &Mirror, three_d: ThreeD<'_>) -> Result<Vec<Viewport>, DrawError> {
    let lower_left = lower_left(three_d)?;
    #[expect(
        clippy::cast_precision_loss,
        reason = "Eden converts the 16-bit surface clip height to a float the same way"
    )]
    let clip = clip_height(three_d)? as f32;
    let z_clip = mirror.reg(&clc797::SET_VIEWPORT_Z_CLIP, &[])?;
    let reduce_z = if z_clip.field(clc797::SET_VIEWPORT_Z_CLIP_RANGE)?
        == clc797::SET_VIEWPORT_Z_CLIP_RANGE_NEGATIVE_W_TO_POSITIVE_W
    {
        1.0
    } else {
        0.0
    };
    (0..VIEWPORTS)
        .map(|index| {
            let v = three_d.viewport(index)?;
            let (scale_x, scale_y, scale_z) = (v.scale_x()?, v.scale_y()?, v.scale_z()?);
            let (offset_x, offset_y, offset_z) = (v.offset_x()?, v.offset_y()?, v.offset_z()?);
            let x = offset_x - scale_x;
            let width = scale_x * 2.0;
            let mut y = offset_y - scale_y;
            let mut height = scale_y * 2.0;
            if lower_left {
                if scale_y < 0.0 && y >= clip {
                    y = clip - y;
                } else {
                    y += clip;
                }
                height = -height;
            }
            if v.swizzle_y()?.value == clc797::SET_VIEWPORT_COORDINATE_SWIZZLE_Y_NEG_Y {
                y += height;
                height = -height;
            }
            Ok(Viewport {
                x,
                y,
                width: if width == 0.0 { 1.0 } else { width },
                height: if height == 0.0 { 1.0 } else { height },
                min_depth: (offset_z - scale_z * reduce_z).clamp(0.0, 1.0),
                max_depth: (offset_z + scale_z).clamp(0.0, 1.0),
            })
        })
        .collect()
}

/// Every scissor (module docs, "Sources"): a disabled one is the whole target, the largest
/// `i32` in each extent as Eden has it.
fn scissors(three_d: ThreeD<'_>) -> Result<Vec<Rect>, DrawError> {
    const WHOLE: u32 = i32::MAX.unsigned_abs();
    let lower_left = lower_left(three_d)?;
    let clip = i64::from(clip_height(three_d)?);
    (0..SCISSORS)
        .map(|index| {
            let s = three_d.scissor(index)?;
            if !on(s.enable()?) {
                return Ok(Rect {
                    x: 0,
                    y: 0,
                    width: WHOLE,
                    height: WHOLE,
                });
            }
            let (min_x, max_x) = (i64::from(s.xmin()?), i64::from(s.xmax()?));
            let (min_y, max_y) = (i64::from(s.ymin()?), i64::from(s.ymax()?));
            let (min_y, max_y) = if lower_left {
                (clip - max_y, clip - min_y)
            } else {
                (min_y, max_y)
            };
            let (min_y, max_y) = (min_y.max(0), max_y.max(0));
            let extent = |span: i64| u32::try_from(span.max(0)).unwrap_or(0);
            Ok(Rect {
                x: extent(min_x),
                y: extent(min_y),
                width: extent(max_x - min_x),
                height: extent(max_y - min_y),
            })
        })
        .collect()
}

/// One target's equations from the shared or per-target fields.
fn equations(
    shared: Blend<'_>,
    per_target: &BlendPerTarget<'_>,
    use_per_target: bool,
) -> Result<(BlendComponent, BlendComponent), DrawError> {
    let component = |src, dst, op| -> Result<BlendComponent, DrawError> {
        Ok(BlendComponent {
            src: blend_factor(src)?,
            dst: blend_factor(dst)?,
            op: blend_op(op)?,
        })
    };
    if use_per_target {
        Ok((
            component(
                per_target.color_source_coeff()?,
                per_target.color_dest_coeff()?,
                per_target.color_op()?,
            )?,
            component(
                per_target.alpha_source_coeff()?,
                per_target.alpha_dest_coeff()?,
                per_target.alpha_op()?,
            )?,
        ))
    } else {
        Ok((
            component(
                shared.color_source_coeff()?,
                shared.color_dest_coeff()?,
                shared.color_op()?,
            )?,
            component(
                shared.alpha_source_coeff()?,
                shared.alpha_dest_coeff()?,
                shared.alpha_op()?,
            )?,
        ))
    }
}

/// The blend part (module docs, "Sources").
fn blend(three_d: ThreeD<'_>) -> Result<BlendPart, DrawError> {
    let shared = three_d.blend();
    let per_target = on(shared.state_per_target()?);
    let common_mask = on(shared.single_ct_write_control()?);
    let disabled = BlendComponent {
        src: BlendFactor::One,
        dst: BlendFactor::Zero,
        op: BlendOperation::Add,
    };
    let mut targets = [BlendState {
        enabled: false,
        color: disabled,
        alpha: disabled,
        write_mask: ColorWrites::EMPTY,
    }; GUEST_COLOR_TARGETS];
    let mut dual_source = false;
    for (index, slot) in (0..BLEND_TARGETS).zip(targets.iter_mut()) {
        let target = three_d.blend_per_target(index)?;
        let mask_target = three_d.blend_per_target(if common_mask { 0 } else { index })?;
        let mut write_mask = ColorWrites::EMPTY;
        for (enabled, flag) in [
            (mask_target.write_red()?, ColorWrites::RED),
            (mask_target.write_green()?, ColorWrites::GREEN),
            (mask_target.write_blue()?, ColorWrites::BLUE),
            (mask_target.write_alpha()?, ColorWrites::ALPHA),
        ] {
            if on(enabled) {
                write_mask = write_mask | flag;
            }
        }
        slot.write_mask = write_mask;
        if on(target.enable()?) {
            let (color, alpha) = equations(shared, &target, per_target)?;
            slot.enabled = true;
            if index == 0 {
                dual_source = [color.src, color.dst, alpha.src, alpha.dst]
                    .iter()
                    .any(|factor| factor.is_dual_source());
            }
            slot.color = color;
            slot.alpha = alpha;
        }
    }
    let logic = if on(shared.logic_op_enable()?) {
        Some(logic_op(shared.logic_op_func()?)?)
    } else {
        None
    };
    Ok(BlendPart {
        targets,
        logic_op: logic,
        constant: [
            shared.const_red()?,
            shared.const_green()?,
            shared.const_blue()?,
            shared.const_alpha()?,
        ],
        dual_source,
    })
}

/// The rasterizer part (module docs, "Sources").
fn raster(three_d: ThreeD<'_>) -> Result<RasterPart, DrawError> {
    let r = three_d.rasterizer();
    let cull = if on(r.cull_enable()?) {
        let face = r.cull_face()?;
        match face.name {
            "FRONT" => CullMode::Front,
            "BACK" => CullMode::Back,
            "FRONT_AND_BACK" => CullMode::FrontAndBack,
            _ => return Err(unmapped(face, "cull face")),
        }
    } else {
        CullMode::None
    };
    let front = r.front_face()?;
    let front_face = match front.name {
        "CW" => FrontFace::Cw,
        "CCW" => FrontFace::Ccw,
        _ => return Err(unmapped(front, "front face")),
    };
    let mode = r.front_polygon_mode()?;
    let polygon_mode = match mode.name {
        "POINT" => PolygonMode::Point,
        "LINE" => PolygonMode::Line,
        "FILL" => PolygonMode::Fill,
        _ => return Err(unmapped(mode, "polygon mode")),
    };
    Ok(RasterPart {
        cull,
        front_face,
        polygon_mode,
        primitive_restart: on(r.primitive_restart()?),
        depth_bias: DepthBias {
            constant: r.depth_bias()?,
            slope_scale: r.slope_scale_depth_bias()?,
            clamp: r.depth_bias_clamp()?,
        },
        offset_enables: [
            on(r.poly_offset_point()?),
            on(r.poly_offset_line()?),
            on(r.poly_offset_fill()?),
        ],
        line_width: r.line_width()?,
        raster_enable: on(r.raster_enable()?),
    })
}

impl RasterPart {
    /// The depth bias a draw of `topology` applies: the registers when the polygon offset enable
    /// of the topology's class (points, lines, or fill for the rest) is on, else none (Eden's
    /// `POLYGON_OFFSET_ENABLE_LUT`).
    pub fn bias_for(&self, topology: Topology) -> DepthBias {
        let class = match topology {
            Topology::Points => 0,
            Topology::Lines
            | Topology::LineLoop
            | Topology::LineStrip
            | Topology::LinesAdjacency
            | Topology::LineStripAdjacency => 1,
            _ => 2,
        };
        if self.offset_enables.get(class).copied().unwrap_or(false) {
            self.depth_bias
        } else {
            DepthBias::default()
        }
    }
}

/// The depth and stencil part (module docs, "Sources").
fn depth_stencil(three_d: ThreeD<'_>) -> Result<DepthStencilPart, DrawError> {
    let d = three_d.depth_stencil();
    let depth_test = if on(d.depth_test()?) {
        Some(compare(d.depth_func()?)?)
    } else {
        None
    };
    let read_mask = d.stencil_func_mask()?;
    let write_mask = d.stencil_mask()?;
    let stencil = if on(d.stencil_test()?) {
        let front = StencilFace {
            compare: compare(d.stencil_func()?)?,
            fail: stencil_op(d.stencil_op_fail()?)?,
            depth_fail: stencil_op(d.stencil_op_zfail()?)?,
            pass: stencil_op(d.stencil_op_zpass()?)?,
        };
        let back = if on(d.two_sided_stencil_test()?) {
            StencilFace {
                compare: compare(d.back_stencil_func()?)?,
                fail: stencil_op(d.back_stencil_op_fail()?)?,
                depth_fail: stencil_op(d.back_stencil_op_zfail()?)?,
                pass: stencil_op(d.back_stencil_op_zpass()?)?,
            }
        } else {
            front
        };
        Some(StencilState {
            front,
            back,
            read_mask,
            write_mask,
        })
    } else {
        None
    };
    Ok(DepthStencilPart {
        state: DepthStencilState {
            depth_test,
            depth_write: on(d.depth_write()?),
            stencil,
        },
        reference: d.stencil_func_ref()?,
        read_mask,
        write_mask,
    })
}

/// A vertex attribute's component layout, from `COMPONENT_BIT_WIDTHS`' names.
pub fn components(value: &'static Value) -> Result<VertexComponents, DrawError> {
    Ok(match value.name {
        "R32_G32_B32_A32" => VertexComponents::R32G32B32A32,
        "R32_G32_B32" => VertexComponents::R32G32B32,
        "R16_G16_B16_A16" => VertexComponents::R16G16B16A16,
        "R32_G32" => VertexComponents::R32G32,
        "R16_G16_B16" => VertexComponents::R16G16B16,
        "A8B8G8R8" | "R8_G8_B8_A8" => VertexComponents::R8G8B8A8,
        "X8B8G8R8" => VertexComponents::X8B8G8R8,
        "A2B10G10R10" => VertexComponents::A2B10G10R10,
        "B10G11R11" => VertexComponents::B10G11R11,
        "R16_G16" => VertexComponents::R16G16,
        "R32" => VertexComponents::R32,
        "R8_G8_B8" => VertexComponents::R8G8B8,
        "G8R8" => VertexComponents::G8R8,
        "R8_G8" => VertexComponents::R8G8,
        "R16" => VertexComponents::R16,
        "R8" => VertexComponents::R8,
        "A8" => VertexComponents::A8,
        _ => return Err(unmapped(value, "vertex component layout")),
    })
}

/// A vertex attribute's numeric type and its variant-key type, from `NUMERICAL_TYPE`'s names
/// (Eden's `AttributeType` from `VertexAttribute::Type`: `vk_pipeline_cache.cpp`'s
/// `CastAttributeType`).
pub fn vertex_kind(value: &'static Value) -> Result<(VertexKind, AttributeType), DrawError> {
    Ok(match value.name {
        "NUM_SNORM" => (VertexKind::Snorm, AttributeType::Float),
        "NUM_UNORM" => (VertexKind::Unorm, AttributeType::Float),
        "NUM_SINT" => (VertexKind::Sint, AttributeType::SignedInt),
        "NUM_UINT" => (VertexKind::Uint, AttributeType::UnsignedInt),
        "NUM_USCALED" => (VertexKind::Uscaled, AttributeType::UnsignedScaled),
        "NUM_SSCALED" => (VertexKind::Sscaled, AttributeType::SignedScaled),
        "NUM_FLOAT" => (VertexKind::Float, AttributeType::Float),
        _ => return Err(unmapped(value, "vertex numeric type")),
    })
}

/// The name `NUMERICAL_TYPE` gives 0, which Eden's `CastAttributeType` reads as disabled.
const UNUSED_NUMERICAL_TYPE: &str = "UNUSED_ENUM_DO_NOT_USE_BECAUSE_IT_WILL_GO_AWAY";

/// The vertex input part (module docs, "Sources").
///
/// An attribute is in the layout when it is `ACTIVE` and its numerical type is not the unused
/// value (Eden's `CastAttributeType` reads that value as disabled, and a register file that was
/// never written holds it, with `ACTIVE` as 0). Each stream an attribute reads becomes one
/// buffer, in stream order; a stream that no attribute reads is not bound. An attribute reading
/// a stream that is not enabled is refused ([`DrawError::AttributeStream`]).
///
/// Eden's pipeline also leaves out attributes the vertex program does not load
/// (`stage_infos[0].loads.Generic(index)` in `vk_graphics_pipeline.cpp`); the seam's sets carry
/// no such list yet, so every active attribute is described. A listed divergence:
/// `tests::every_active_attribute_is_in_the_layout`.
fn vertex(three_d: ThreeD<'_>) -> Result<VertexPart, DrawError> {
    struct Active {
        index: u32,
        stream: u32,
        offset: u32,
        components: VertexComponents,
        kind: VertexKind,
    }
    let mut types = [AttributeType::Disabled; VERTEX_ATTRIBUTES as usize];
    let mut active = Vec::new();
    for (index, ty) in (0..VERTEX_ATTRIBUTES).zip(types.iter_mut()) {
        let attribute = three_d.vertex_attribute(index)?;
        let numerical = attribute.numerical_type()?;
        if attribute.source()?.name != "ACTIVE" || numerical.name == UNUSED_NUMERICAL_TYPE {
            continue;
        }
        let (kind, key) = vertex_kind(numerical)?;
        *ty = key;
        active.push(Active {
            index,
            stream: attribute.stream()?,
            offset: attribute.offset()?,
            components: components(attribute.component_bit_widths()?)?,
            kind,
        });
    }
    let mut layout = VertexLayout::default();
    let mut streams = Vec::new();
    let mut buffer_of = [None; VERTEX_STREAMS as usize];
    for (index, slot) in (0..VERTEX_STREAMS).zip(buffer_of.iter_mut()) {
        if !active.iter().any(|a| a.stream == index) {
            continue;
        }
        let stream = three_d.vertex_stream(index)?;
        if !on(stream.enable()?) {
            continue;
        }
        *slot = Some(u32::try_from(layout.buffers.len()).unwrap_or(u32::MAX));
        layout.buffers.push(VertexBufferLayout {
            stride: stream.stride()?,
            step: if on(stream.instanced()?) {
                VertexStepMode::Instance
            } else {
                VertexStepMode::Vertex
            },
        });
        streams.push(Stream {
            index,
            address: stream.address()?,
            size: stream.size()?,
        });
    }
    for a in active {
        let buffer = buffer_of
            .get(usize::try_from(a.stream).unwrap_or(usize::MAX))
            .copied()
            .flatten()
            .ok_or(DrawError::AttributeStream {
                attribute: a.index,
                stream: a.stream,
            })?;
        layout.attributes.push(HostAttribute {
            location: a.index,
            buffer,
            offset: a.offset,
            components: a.components,
            kind: a.kind,
        });
    }
    Ok(VertexPart {
        layout,
        streams,
        types,
    })
}
