//! A graphics pipeline's fixed-function state in wgpu's terms (docs/specs/render-backend.md,
//! "What a pipeline is made of"). A pure conversion, made on the calling thread when the
//! pipeline is created, so a description wgpu cannot express is refused there and then; the
//! compile worker builds the pipeline from the result.
//!
//! For policy only, the choices follow Eden's Vulkan back end, which turns the same guest state
//! into Vulkan state (`eden/src/video_core/renderer_vulkan/maxwell_to_vk.cpp` and
//! `vk_graphics_pipeline.cpp` @ cf654052); the code is new.
//!
//! # Vertex input
//!
//! Each attribute's components and type pick a wgpu vertex format as Eden's `VertexFormat`
//! picks a Vulkan one, where wgpu has that format:
//!
//! - `A8` reads as `R8`, `G8R8` as `R8G8` and `X8B8G8R8` as `R8G8B8A8`, as Eden's table has
//!   them;
//! - the scaled types read as the integer type of the same signedness, as Eden does on a device
//!   that `MustEmulateScaledFormats`, which wgpu is (it has no scaled vertex formats); the
//!   vertex shader converts the integers to floats, which is the shader side's to do;
//! - `A2B10G10R10` unorm is wgpu's `Unorm10_10_10_2`, which wgpu-hal 30.0.2 gives Vulkan's
//!   `A2B10G10R10_UNORM_PACK32` (`src/vulkan/conv.rs`), the format Eden picks.
//!
//! Refused, with [`VERTEX_FORMAT`], because wgpu has no vertex format for them: three 8-bit or
//! three 16-bit components (Eden's `R8G8B8` and `R16G16B16` formats), `A2B10G10R10` of any type
//! but unorm, `B10G11R11`, 8-bit floats, and the normalised and scaled 32-bit types (none in
//! Eden's table either). These attributes would need the attribute-fetch emulation.
//!
//! Attributes are grouped by the buffer they read, in the order the description lists them.
//! wgpu's own rules on strides and offsets (alignment to 4 bytes, an attribute inside its
//! stride, the device's stride limit beyond `Caps`) are not repeated here: a pipeline that
//! breaks one fails to build, and its status says so.
//!
//! # Topology
//!
//! The guest's topologies map to wgpu's directly where wgpu has them. The ones it lacks are drawn
//! from rewritten indices (render-backend, "Fixed-function state the host lacks": quads and line
//! loops by an index rewrite; the rewrite itself is slice R7's), and the pipeline gets the
//! topology the rewrite produces: quads and quad strips as triangle lists, as Eden's
//! `PrimitiveTopology` maps them; triangle fans and polygons as triangle lists too, since wgpu
//! has no fans (Eden keeps fans native, and maps polygons to fans); line loops as line strips
//! closed by repeating the first index (Eden maps them to triangle lists, which draws no line;
//! guess: R7 settles it). The topologies with adjacency are refused ([`ADJACENCY`]): wgpu has
//! none, and without a geometry shader nothing reads the extra vertices. Patches never reach
//! here: the shared checks need tessellation shaders for them, which a wgpu device cannot run.
//! `strip_index_format` is left unset: the pipeline does not know the index format, and wgpu then
//! restarts strips at the index format's restart value (render-backend's table).
//!
//! # Rasterisation, depth and stencil
//!
//! Culling both faces is refused ([`CULL_BOTH`]): wgpu has no such mode. Depth clamping is wgpu's
//! `unclipped_depth`. The baked depth bias is used as given, its constant term rounded to the
//! nearest integer (wgpu takes an `i32`, Vulkan a float; guess: the unit is the same). wgpu has
//! no line width: lines are one pixel wide whatever the description says (a listed divergence,
//! since `Caps` cannot say so). Without a depth test the depth state compares `Always` and does
//! not write, as Vulkan does with the test off.
//!
//! # Colour targets and blending
//!
//! Each target's format must have a wgpu form ([`TARGET_FORMAT`]). wgpu refuses a min or max
//! blend equation whose factors are not one (`InvalidMinMaxBlendFactor`, wgpu-core 30.0.2
//! `src/device/resource.rs`); the factors play no part in those equations, so they are passed as
//! one. A description with colour targets and no fragment shader is refused
//! ([`TARGETS_WITHOUT_FRAGMENT`]): wgpu takes the targets from the fragment state.

use crate::convert::compare;
use crate::format::texture_format;
use starquake_render::RenderError;
use starquake_render::pipeline::{
    BlendComponent, BlendFactor, BlendOperation, BlendState, ColorWrites, CullMode,
    DepthStencilState, FrontFace, GraphicsPipelineDesc, PolygonMode, StencilFace, StencilOperation,
    Topology, VertexAttribute, VertexComponents, VertexKind, VertexStepMode,
};

/// A vertex attribute wgpu has no format for.
pub const VERTEX_FORMAT: &str =
    "wgpu has no vertex format for this attribute's components and type";
/// A topology with adjacency.
pub const ADJACENCY: &str = "wgpu has no topologies with adjacency or patches";
/// Culling both faces.
pub const CULL_BOTH: &str = "wgpu cannot cull both faces";
/// A render target format with no wgpu form.
pub const TARGET_FORMAT: &str = "a render target's format has no wgpu form";
/// Colour targets without a fragment shader.
pub const TARGETS_WITHOUT_FRAGMENT: &str =
    "a pipeline with colour targets needs a fragment shader under wgpu";

/// One vertex buffer's layout, owned.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct BufferPlan {
    /// Bytes between elements.
    pub(crate) stride: u64,
    /// How it advances.
    pub(crate) step: wgpu::VertexStepMode,
    /// Its attributes, in the description's order.
    pub(crate) attributes: Vec<wgpu::VertexAttribute>,
}

/// A graphics pipeline's fixed-function state, owned, for the compile worker.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct GraphicsPlan {
    /// Vertex buffers, by slot.
    pub(crate) buffers: Vec<BufferPlan>,
    /// Topology, faces and rasterisation.
    pub(crate) primitive: wgpu::PrimitiveState,
    /// Depth and stencil, when the pipeline has a depth/stencil target.
    pub(crate) depth_stencil: Option<wgpu::DepthStencilState>,
    /// Samples.
    pub(crate) multisample: wgpu::MultisampleState,
    /// Colour targets, by index, up to the last one present.
    pub(crate) targets: Vec<Option<wgpu::ColorTargetState>>,
}

/// The wgpu vertex format for `components` read as `kind` (module docs), or `None`.
pub(crate) const fn vertex_format(
    components: VertexComponents,
    kind: VertexKind,
) -> Option<wgpu::VertexFormat> {
    use VertexComponents as C;
    use VertexKind as K;
    use wgpu::VertexFormat as F;
    // Eden's `MustEmulateScaledFormats` path: scaled types read as integers.
    let kind = match kind {
        K::Uscaled => K::Uint,
        K::Sscaled => K::Sint,
        other => other,
    };
    Some(match (components, kind) {
        (C::R8 | C::A8, K::Unorm) => F::Unorm8,
        (C::R8 | C::A8, K::Snorm) => F::Snorm8,
        (C::R8 | C::A8, K::Uint) => F::Uint8,
        (C::R8 | C::A8, K::Sint) => F::Sint8,
        (C::R8G8 | C::G8R8, K::Unorm) => F::Unorm8x2,
        (C::R8G8 | C::G8R8, K::Snorm) => F::Snorm8x2,
        (C::R8G8 | C::G8R8, K::Uint) => F::Uint8x2,
        (C::R8G8 | C::G8R8, K::Sint) => F::Sint8x2,
        (C::R8G8B8A8 | C::X8B8G8R8, K::Unorm) => F::Unorm8x4,
        (C::R8G8B8A8 | C::X8B8G8R8, K::Snorm) => F::Snorm8x4,
        (C::R8G8B8A8 | C::X8B8G8R8, K::Uint) => F::Uint8x4,
        (C::R8G8B8A8 | C::X8B8G8R8, K::Sint) => F::Sint8x4,
        (C::R16, K::Unorm) => F::Unorm16,
        (C::R16, K::Snorm) => F::Snorm16,
        (C::R16, K::Uint) => F::Uint16,
        (C::R16, K::Sint) => F::Sint16,
        (C::R16, K::Float) => F::Float16,
        (C::R16G16, K::Unorm) => F::Unorm16x2,
        (C::R16G16, K::Snorm) => F::Snorm16x2,
        (C::R16G16, K::Uint) => F::Uint16x2,
        (C::R16G16, K::Sint) => F::Sint16x2,
        (C::R16G16, K::Float) => F::Float16x2,
        (C::R16G16B16A16, K::Unorm) => F::Unorm16x4,
        (C::R16G16B16A16, K::Snorm) => F::Snorm16x4,
        (C::R16G16B16A16, K::Uint) => F::Uint16x4,
        (C::R16G16B16A16, K::Sint) => F::Sint16x4,
        (C::R16G16B16A16, K::Float) => F::Float16x4,
        (C::R32, K::Uint) => F::Uint32,
        (C::R32, K::Sint) => F::Sint32,
        (C::R32, K::Float) => F::Float32,
        (C::R32G32, K::Uint) => F::Uint32x2,
        (C::R32G32, K::Sint) => F::Sint32x2,
        (C::R32G32, K::Float) => F::Float32x2,
        (C::R32G32B32, K::Uint) => F::Uint32x3,
        (C::R32G32B32, K::Sint) => F::Sint32x3,
        (C::R32G32B32, K::Float) => F::Float32x3,
        (C::R32G32B32A32, K::Uint) => F::Uint32x4,
        (C::R32G32B32A32, K::Sint) => F::Sint32x4,
        (C::R32G32B32A32, K::Float) => F::Float32x4,
        (C::A2B10G10R10, K::Unorm) => F::Unorm10_10_10_2,
        _ => return None,
    })
}

/// wgpu's topology for a guest topology (module docs), or `None` for one with adjacency or
/// patches.
pub(crate) const fn topology(topology: Topology) -> Option<wgpu::PrimitiveTopology> {
    use wgpu::PrimitiveTopology as P;
    Some(match topology {
        Topology::Points => P::PointList,
        Topology::Lines => P::LineList,
        Topology::LineStrip | Topology::LineLoop => P::LineStrip,
        Topology::Triangles
        | Topology::TriangleFan
        | Topology::Quads
        | Topology::QuadStrip
        | Topology::Polygon => P::TriangleList,
        Topology::TriangleStrip => P::TriangleStrip,
        Topology::LinesAdjacency
        | Topology::LineStripAdjacency
        | Topology::TrianglesAdjacency
        | Topology::TriangleStripAdjacency
        | Topology::Patches => return None,
    })
}

const fn blend_factor(factor: BlendFactor) -> wgpu::BlendFactor {
    use wgpu::BlendFactor as F;
    match factor {
        BlendFactor::Zero => F::Zero,
        BlendFactor::One => F::One,
        BlendFactor::Src => F::Src,
        BlendFactor::OneMinusSrc => F::OneMinusSrc,
        BlendFactor::SrcAlpha => F::SrcAlpha,
        BlendFactor::OneMinusSrcAlpha => F::OneMinusSrcAlpha,
        BlendFactor::Dst => F::Dst,
        BlendFactor::OneMinusDst => F::OneMinusDst,
        BlendFactor::DstAlpha => F::DstAlpha,
        BlendFactor::OneMinusDstAlpha => F::OneMinusDstAlpha,
        BlendFactor::SrcAlphaSaturated => F::SrcAlphaSaturated,
        BlendFactor::Constant => F::Constant,
        BlendFactor::OneMinusConstant => F::OneMinusConstant,
        BlendFactor::Src1 => F::Src1,
        BlendFactor::OneMinusSrc1 => F::OneMinusSrc1,
        BlendFactor::Src1Alpha => F::Src1Alpha,
        BlendFactor::OneMinusSrc1Alpha => F::OneMinusSrc1Alpha,
    }
}

/// One blend equation; a min or max equation gets factors of one (module docs).
const fn blend_component(c: BlendComponent) -> wgpu::BlendComponent {
    let operation = match c.op {
        BlendOperation::Add => wgpu::BlendOperation::Add,
        BlendOperation::Subtract => wgpu::BlendOperation::Subtract,
        BlendOperation::ReverseSubtract => wgpu::BlendOperation::ReverseSubtract,
        BlendOperation::Min => wgpu::BlendOperation::Min,
        BlendOperation::Max => wgpu::BlendOperation::Max,
    };
    let (src, dst) = match c.op {
        BlendOperation::Min | BlendOperation::Max => {
            (wgpu::BlendFactor::One, wgpu::BlendFactor::One)
        }
        _ => (blend_factor(c.src), blend_factor(c.dst)),
    };
    wgpu::BlendComponent {
        src_factor: src,
        dst_factor: dst,
        operation,
    }
}

fn write_mask(mask: ColorWrites) -> wgpu::ColorWrites {
    let mut out = wgpu::ColorWrites::empty();
    for (ours, theirs) in [
        (ColorWrites::RED, wgpu::ColorWrites::RED),
        (ColorWrites::GREEN, wgpu::ColorWrites::GREEN),
        (ColorWrites::BLUE, wgpu::ColorWrites::BLUE),
        (ColorWrites::ALPHA, wgpu::ColorWrites::ALPHA),
    ] {
        if mask.contains(ours) {
            out |= theirs;
        }
    }
    out
}

fn color_target(
    format: starquake_render::PixelFormat,
    blend: &BlendState,
) -> Result<wgpu::ColorTargetState, RenderError> {
    Ok(wgpu::ColorTargetState {
        format: texture_format(format).ok_or(RenderError::Unsupported(TARGET_FORMAT))?,
        blend: blend.enabled.then(|| wgpu::BlendState {
            color: blend_component(blend.color),
            alpha: blend_component(blend.alpha),
        }),
        write_mask: write_mask(blend.write_mask),
    })
}

const fn stencil_operation(op: StencilOperation) -> wgpu::StencilOperation {
    use wgpu::StencilOperation as O;
    match op {
        StencilOperation::Keep => O::Keep,
        StencilOperation::Zero => O::Zero,
        StencilOperation::Replace => O::Replace,
        StencilOperation::IncrementClamp => O::IncrementClamp,
        StencilOperation::DecrementClamp => O::DecrementClamp,
        StencilOperation::Invert => O::Invert,
        StencilOperation::IncrementWrap => O::IncrementWrap,
        StencilOperation::DecrementWrap => O::DecrementWrap,
    }
}

const fn stencil_face(face: StencilFace) -> wgpu::StencilFaceState {
    wgpu::StencilFaceState {
        compare: compare(face.compare),
        fail_op: stencil_operation(face.fail),
        depth_fail_op: stencil_operation(face.depth_fail),
        pass_op: stencil_operation(face.pass),
    }
}

/// The depth bias constant as wgpu's integer: rounded, saturating (module docs; the shared
/// checks have made it finite).
#[expect(
    clippy::cast_possible_truncation,
    reason = "a float-to-integer `as` saturates at the integer's bounds, which is the clamp wanted"
)]
fn bias_constant(constant: f32) -> i32 {
    constant.round() as i32
}

fn depth_stencil(
    format: starquake_render::PixelFormat,
    state: &DepthStencilState,
    bias: starquake_render::pipeline::DepthBias,
) -> Result<wgpu::DepthStencilState, RenderError> {
    let stencil = state
        .stencil
        .map_or_else(wgpu::StencilState::default, |s| wgpu::StencilState {
            front: stencil_face(s.front),
            back: stencil_face(s.back),
            read_mask: s.read_mask,
            write_mask: s.write_mask,
        });
    Ok(wgpu::DepthStencilState {
        format: texture_format(format).ok_or(RenderError::Unsupported(TARGET_FORMAT))?,
        depth_write_enabled: Some(state.depth_test.is_some() && state.depth_write),
        depth_compare: Some(
            state
                .depth_test
                .map_or(wgpu::CompareFunction::Always, compare),
        ),
        stencil,
        bias: wgpu::DepthBiasState {
            constant: bias_constant(bias.constant),
            slope_scale: bias.slope_scale,
            clamp: bias.clamp,
        },
    })
}

fn vertex_buffers(desc: &GraphicsPipelineDesc) -> Result<Vec<BufferPlan>, RenderError> {
    let input = &desc.vertex_input;
    let mut buffers: Vec<BufferPlan> = input
        .buffers
        .iter()
        .map(|b| BufferPlan {
            stride: b.stride.into(),
            step: match b.step {
                VertexStepMode::Vertex => wgpu::VertexStepMode::Vertex,
                VertexStepMode::Instance => wgpu::VertexStepMode::Instance,
            },
            attributes: Vec::new(),
        })
        .collect();
    for &VertexAttribute {
        location,
        buffer,
        offset,
        components,
        kind,
    } in &input.attributes
    {
        let format =
            vertex_format(components, kind).ok_or(RenderError::Unsupported(VERTEX_FORMAT))?;
        // The shared checks have put `buffer` below the buffer count.
        let Some(plan) = usize::try_from(buffer)
            .ok()
            .and_then(|b| buffers.get_mut(b))
        else {
            return Err(RenderError::NotBelow {
                what: "vertex attribute buffer",
                index: buffer.into(),
                count: u64::try_from(input.buffers.len()).unwrap_or(u64::MAX),
            });
        };
        plan.attributes.push(wgpu::VertexAttribute {
            format,
            offset: offset.into(),
            shader_location: location,
        });
    }
    Ok(buffers)
}

/// The fixed-function state of a description the shared checks accepted, or the refusal
/// (module docs). Checked in the order: vertex formats, topology, culling, colour targets
/// (a fragment shader, then each target's format in index order), the depth/stencil format.
pub(crate) fn graphics(desc: &GraphicsPipelineDesc) -> Result<GraphicsPlan, RenderError> {
    let buffers = vertex_buffers(desc)?;
    let topology = topology(desc.topology).ok_or(RenderError::Unsupported(ADJACENCY))?;
    let raster = &desc.raster;
    let cull_mode = match raster.cull {
        CullMode::None => None,
        CullMode::Front => Some(wgpu::Face::Front),
        CullMode::Back => Some(wgpu::Face::Back),
        CullMode::FrontAndBack => return Err(RenderError::Unsupported(CULL_BOTH)),
    };
    let primitive = wgpu::PrimitiveState {
        topology,
        strip_index_format: None,
        front_face: match raster.front_face {
            FrontFace::Ccw => wgpu::FrontFace::Ccw,
            FrontFace::Cw => wgpu::FrontFace::Cw,
        },
        cull_mode,
        unclipped_depth: raster.depth_clamp,
        polygon_mode: match raster.polygon_mode {
            PolygonMode::Fill => wgpu::PolygonMode::Fill,
            PolygonMode::Line => wgpu::PolygonMode::Line,
            PolygonMode::Point => wgpu::PolygonMode::Point,
        },
        conservative: false,
    };
    let targets = &desc.targets;
    let used = targets
        .colors
        .iter()
        .rposition(Option::is_some)
        .map_or(0, |i| i.saturating_add(1));
    if used > 0 && desc.stages.fragment.is_none() {
        return Err(RenderError::Unsupported(TARGETS_WITHOUT_FRAGMENT));
    }
    let colors = targets
        .colors
        .iter()
        .zip(&desc.blend)
        .take(used)
        .map(|(format, blend)| format.map(|f| color_target(f, blend)).transpose())
        .collect::<Result<Vec<_>, _>>()?;
    let depth_stencil = targets
        .depth_stencil
        .map(|format| depth_stencil(format, &desc.depth_stencil, raster.depth_bias))
        .transpose()?;
    Ok(GraphicsPlan {
        buffers,
        primitive,
        depth_stencil,
        multisample: wgpu::MultisampleState {
            count: targets.sample_count,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        targets: colors,
    })
}

#[cfg(test)]
mod tests;
