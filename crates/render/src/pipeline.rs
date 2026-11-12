//! Graphics and compute pipeline descriptions (docs/specs/render-backend.md, "Pipelines").
//!
//! A pipeline is shaders plus the fixed-function state that is not dynamic on the host. The GPU
//! crate keys pipelines on everything here that [`crate::Caps::dynamic_state`] leaves out; the
//! state that may be dynamic (depth bias, line width, stencil masks) is here as the baked value
//! and in [`crate::command::DynamicValues`] as the per-draw value, and the backend uses whichever
//! its device takes. The guest's topologies, including quads and line loops, are named as the
//! guest names them (`PrimitiveTopology`, `eden/src/video_core/engines/maxwell_3d.h` @ cf654052);
//! converting the ones the host lacks is the backend's job.

use crate::caps::GUEST_COLOR_TARGETS;
use crate::flags::flag_set;
use crate::format::PixelFormat;
use crate::id::ShaderId;
use crate::resource::CompareFunction;
use serde::{Deserialize, Serialize};

/// The shaders of a graphics pipeline.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Stages {
    /// Vertex shader.
    pub vertex: ShaderId,
    /// Tessellation control shader; present exactly when `tess_eval` is.
    pub tess_control: Option<ShaderId>,
    /// Tessellation evaluation shader; present exactly when `tess_control` is.
    pub tess_eval: Option<ShaderId>,
    /// Geometry shader.
    pub geometry: Option<ShaderId>,
    /// Fragment shader; none for depth-only passes.
    pub fragment: Option<ShaderId>,
}

/// Whether a vertex buffer advances per vertex or per instance.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VertexStepMode {
    /// Per vertex.
    Vertex,
    /// Per instance.
    Instance,
}

/// One vertex buffer's layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct VertexBufferLayout {
    /// Bytes between consecutive elements.
    pub stride: u32,
    /// How it advances.
    pub step: VertexStepMode,
}

/// The component layout of a vertex attribute (the guest's `VertexAttribute::Size`,
/// `maxwell_3d.h` @ cf654052).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VertexComponents {
    /// Four 32-bit components.
    R32G32B32A32,
    /// Three 32-bit components.
    R32G32B32,
    /// Four 16-bit components.
    R16G16B16A16,
    /// Two 32-bit components.
    R32G32,
    /// Three 16-bit components.
    R16G16B16,
    /// Four 8-bit components.
    R8G8B8A8,
    /// Two 16-bit components.
    R16G16,
    /// One 32-bit component.
    R32,
    /// Three 8-bit components.
    R8G8B8,
    /// Two 8-bit components.
    R8G8,
    /// One 16-bit component.
    R16,
    /// One 8-bit component.
    R8,
    /// Packed 2-10-10-10.
    A2B10G10R10,
    /// Packed 10-11-11 floats.
    B10G11R11,
    /// Two 8-bit components, green first.
    G8R8,
    /// Three 8-bit components and padding, blue first.
    X8B8G8R8,
    /// One 8-bit alpha component.
    A8,
}

/// How a vertex attribute's components are read (the guest's `VertexAttribute::Type`,
/// `maxwell_3d.h` @ cf654052).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VertexKind {
    /// Signed normalised.
    Snorm,
    /// Unsigned normalised.
    Unorm,
    /// Signed integer.
    Sint,
    /// Unsigned integer.
    Uint,
    /// Unsigned integer converted to float.
    Uscaled,
    /// Signed integer converted to float.
    Sscaled,
    /// Float.
    Float,
}

/// One vertex attribute.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct VertexAttribute {
    /// The shader input location.
    pub location: u32,
    /// Index into [`VertexLayout::buffers`].
    pub buffer: u32,
    /// Byte offset within an element.
    pub offset: u32,
    /// Component layout.
    pub components: VertexComponents,
    /// Component type.
    pub kind: VertexKind,
}

/// The vertex input state: up to the guest's 32 attributes and 32 buffers.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct VertexLayout {
    /// Buffer layouts, indexed by [`VertexAttribute::buffer`].
    pub buffers: Vec<VertexBufferLayout>,
    /// Attributes, each at a distinct location.
    pub attributes: Vec<VertexAttribute>,
}

/// The guest's primitive topologies (`PrimitiveTopology`, `maxwell_3d.h` @ cf654052).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Topology {
    /// Points.
    Points,
    /// Separate lines.
    Lines,
    /// A closed line strip; converted by index rewrite.
    LineLoop,
    /// A line strip.
    LineStrip,
    /// Separate triangles.
    Triangles,
    /// A triangle strip.
    TriangleStrip,
    /// A triangle fan.
    TriangleFan,
    /// Separate quads; converted by index rewrite.
    Quads,
    /// A quad strip; converted by index rewrite.
    QuadStrip,
    /// A polygon; converted by index rewrite.
    Polygon,
    /// Lines with adjacency.
    LinesAdjacency,
    /// A line strip with adjacency.
    LineStripAdjacency,
    /// Triangles with adjacency.
    TrianglesAdjacency,
    /// A triangle strip with adjacency.
    TriangleStripAdjacency,
    /// Patches for tessellation; needs tessellation shaders.
    Patches,
}

/// Which faces are culled.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CullMode {
    /// None.
    None,
    /// Front faces.
    Front,
    /// Back faces.
    Back,
    /// Both.
    FrontAndBack,
}

/// Which winding is the front face.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FrontFace {
    /// Counter-clockwise.
    Ccw,
    /// Clockwise.
    Cw,
}

/// How polygons are drawn.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PolygonMode {
    /// Filled.
    Fill,
    /// Edges only; needs [`crate::caps::Features::POLYGON_MODE_LINE`].
    Line,
    /// Vertices only; needs [`crate::caps::Features::POLYGON_MODE_POINT`].
    Point,
}

/// Depth bias, as baked into a pipeline or set per draw.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct DepthBias {
    /// Constant term.
    pub constant: f32,
    /// Slope-scaled term.
    pub slope_scale: f32,
    /// Largest bias applied (0 for none).
    pub clamp: f32,
}

/// Rasterisation state.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct RasterState {
    /// Culling.
    pub cull: CullMode,
    /// Front face winding.
    pub front_face: FrontFace,
    /// Fill, line or point.
    pub polygon_mode: PolygonMode,
    /// Clamp depth instead of clipping; needs [`crate::caps::Features::DEPTH_CLAMP`].
    pub depth_clamp: bool,
    /// Depth bias, when the device cannot set it per draw.
    pub depth_bias: DepthBias,
    /// Line width, when the device cannot set it per draw.
    pub line_width: f32,
}

/// What a stencil test does to the stored value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StencilOperation {
    /// Keep it.
    Keep,
    /// Set it to 0.
    Zero,
    /// Set it to the reference.
    Replace,
    /// Increment, clamping at the maximum.
    IncrementClamp,
    /// Decrement, clamping at 0.
    DecrementClamp,
    /// Invert its bits.
    Invert,
    /// Increment, wrapping.
    IncrementWrap,
    /// Decrement, wrapping.
    DecrementWrap,
}

/// One face's stencil test.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct StencilFace {
    /// The comparison.
    pub compare: CompareFunction,
    /// On a failed stencil test.
    pub fail: StencilOperation,
    /// On a passed stencil test and a failed depth test.
    pub depth_fail: StencilOperation,
    /// On both passing.
    pub pass: StencilOperation,
}

/// Stencil state.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct StencilState {
    /// Front faces.
    pub front: StencilFace,
    /// Back faces.
    pub back: StencilFace,
    /// Compare mask, when the device cannot set it per draw.
    pub read_mask: u32,
    /// Write mask, when the device cannot set it per draw.
    pub write_mask: u32,
}

/// Depth and stencil state.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DepthStencilState {
    /// The depth test, if enabled.
    pub depth_test: Option<CompareFunction>,
    /// Whether depth is written.
    pub depth_write: bool,
    /// The stencil test, if enabled.
    pub stencil: Option<StencilState>,
}

/// A blend factor.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BlendFactor {
    /// 0.
    Zero,
    /// 1.
    One,
    /// Source colour.
    Src,
    /// 1 - source colour.
    OneMinusSrc,
    /// Source alpha.
    SrcAlpha,
    /// 1 - source alpha.
    OneMinusSrcAlpha,
    /// Destination colour.
    Dst,
    /// 1 - destination colour.
    OneMinusDst,
    /// Destination alpha.
    DstAlpha,
    /// 1 - destination alpha.
    OneMinusDstAlpha,
    /// min(source alpha, 1 - destination alpha).
    SrcAlphaSaturated,
    /// The blend constant.
    Constant,
    /// 1 - the blend constant.
    OneMinusConstant,
    /// The second source colour; needs [`crate::caps::Features::DUAL_SOURCE_BLEND`].
    Src1,
    /// 1 - the second source colour; needs `DUAL_SOURCE_BLEND`.
    OneMinusSrc1,
    /// The second source alpha; needs `DUAL_SOURCE_BLEND`.
    Src1Alpha,
    /// 1 - the second source alpha; needs `DUAL_SOURCE_BLEND`.
    OneMinusSrc1Alpha,
}

impl BlendFactor {
    /// Whether the factor reads the second fragment output.
    pub const fn is_dual_source(self) -> bool {
        matches!(
            self,
            Self::Src1 | Self::OneMinusSrc1 | Self::Src1Alpha | Self::OneMinusSrc1Alpha
        )
    }
}

/// A blend equation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BlendOperation {
    /// src * sf + dst * df.
    Add,
    /// src * sf - dst * df.
    Subtract,
    /// dst * df - src * sf.
    ReverseSubtract,
    /// min(src, dst).
    Min,
    /// max(src, dst).
    Max,
}

/// One blend equation and its factors.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BlendComponent {
    /// Source factor.
    pub src: BlendFactor,
    /// Destination factor.
    pub dst: BlendFactor,
    /// Equation.
    pub op: BlendOperation,
}

flag_set! {
    /// Which colour channels a target writes.
    ColorWrites {
        /// Red.
        RED = 0,
        /// Green.
        GREEN = 1,
        /// Blue.
        BLUE = 2,
        /// Alpha.
        ALPHA = 3,
    }
}

/// One colour target's blending.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BlendState {
    /// Whether blending is on; the factors are ignored when it is off.
    pub enabled: bool,
    /// Colour channels.
    pub color: BlendComponent,
    /// Alpha channel.
    pub alpha: BlendComponent,
    /// Channels written.
    pub write_mask: ColorWrites,
}

/// A colour logic operation (needs [`crate::caps::Features::LOGIC_OP`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LogicOp {
    /// 0.
    Clear,
    /// s & d.
    And,
    /// s & !d.
    AndReverse,
    /// s.
    Copy,
    /// !s & d.
    AndInverted,
    /// d.
    NoOp,
    /// s ^ d.
    Xor,
    /// s | d.
    Or,
    /// !(s | d).
    Nor,
    /// !(s ^ d).
    Equivalent,
    /// !d.
    Invert,
    /// s | !d.
    OrReverse,
    /// !s.
    CopyInverted,
    /// !s | d.
    OrInverted,
    /// !(s & d).
    Nand,
    /// All ones.
    Set,
}

/// The formats a pipeline renders to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TargetFormats {
    /// Colour target formats, by target index.
    pub colors: [Option<PixelFormat>; GUEST_COLOR_TARGETS],
    /// Depth/stencil format.
    pub depth_stencil: Option<PixelFormat>,
    /// Samples per pixel.
    pub sample_count: u32,
}

flag_set! {
    /// What the pipeline's shaders were specialised to emulate because the host lacks it
    /// (render-backend, "Fixed-function state the host lacks"). Decided by the render crate's
    /// `emulation()` (slice R2) and copied by the GPU crate into the shader variant key; here it
    /// is part of the pipeline key.
    EmulationFlags {
        /// Alpha test as a fragment-shader discard.
        ALPHA_TEST = 0,
        /// The vertex shader rewrites z from the -1..1 depth range.
        DEPTH_RANGE_REWRITE = 1,
        /// Indices equal to the restart value are rewritten because restart cannot be turned
        /// off on the host.
        PRIMITIVE_RESTART_REWRITE = 2,
        /// Attributes past the host's limit are fetched from a storage buffer.
        ATTRIBUTE_FETCH = 3,
        /// The fragment shader's outputs are typed to match integer render targets.
        TYPED_OUTPUTS = 4,
    }
}

/// A graphics pipeline (render-backend, "What a pipeline is made of").
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GraphicsPipelineDesc {
    /// The shaders.
    pub stages: Stages,
    /// Vertex input.
    pub vertex_input: VertexLayout,
    /// Primitive topology.
    pub topology: Topology,
    /// Whether an index equal to the strip's restart value starts a new primitive.
    pub primitive_restart: bool,
    /// Rasterisation.
    pub raster: RasterState,
    /// Depth and stencil.
    pub depth_stencil: DepthStencilState,
    /// Blending per colour target.
    pub blend: [BlendState; GUEST_COLOR_TARGETS],
    /// Logic operation, replacing blending when set.
    pub logic_op: Option<LogicOp>,
    /// Target formats and sample count.
    pub targets: TargetFormats,
    /// What the shaders were specialised for.
    pub emulation: EmulationFlags,
}

/// A compute pipeline.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ComputePipelineDesc {
    /// The compute shader.
    pub shader: ShaderId,
}
