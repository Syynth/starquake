//! Draws and dispatches on the wgpu backend (docs/specs/render-backend.md, slice R7: "Commands",
//! "Pipelines", "Fixed-function state the host lacks").
//!
//! A `SetRenderTargets`, `Draw` or `Dispatch` is checked on the submitting thread like every
//! other command (`crate::plan`) and becomes a step ([`DrawStep`], [`DispatchStep`],
//! [`Targets`]) that the render thread encodes (`crate::thread`). The policy follows Eden's Vulkan
//! rasterizer and its render pass cache and descriptor updates
//! (`eden/src/video_core/renderer_vulkan/vk_rasterizer.cpp`, `vk_render_pass_cache.cpp`,
//! `vk_update_descriptor.cpp` @ cf654052), for policy only; the code is new.
//!
//! # Render passes
//!
//! The guest has no render passes. A host pass is opened at the first draw after a
//! `SetRenderTargets`, with load and store actions that load and store (render-backend,
//! "Commands": "Load and store actions start conservative"), and closed when:
//!
//! - a `SetRenderTargets` changes the targets (one that sets the same targets keeps the pass);
//! - any other command is encoded: an upload, copy, clear, readback, present or dispatch. The
//!   spec closes a pass when "a copy or readback touches an attachment"; wgpu cannot encode a
//!   copy, a clear pass, a compute pass or a readback while a render pass is open on the encoder
//!   (it locks the encoder), so under wgpu every such command closes it, whether it touches an
//!   attachment or not. Markers do not;
//! - a draw needs its indices rewritten first (below): the rewrite is a compute pass;
//! - a draw uses a resource in a way that conflicts, under wgpu's usage rule, with how an
//!   earlier draw of the pass used it (a buffer one draw writes as storage and the next reads
//!   as vertices): one render pass is one usage scope in wgpu ([`rules::Uses`]);
//! - the list ends.
//!
//! An attachment is the view's base mip level at its base layer (a 3D texture: slice 0), drawn
//! through a view the render thread makes for the pass, since a wgpu attachment is one level of
//! one layer. A layered target (a view of several layers that a shader picks from) draws into
//! its base layer only (a divergence: wgpu has no layered rendering without multiview).
//!
//! # A draw
//!
//! Each draw sets its pipeline, one bind group per group its pipeline's layout uses (made from
//! the draw's bindings for that draw), its vertex buffers in slot order, its index buffer, the
//! viewport, the scissor, the blend constant and the stencil reference, then draws. Those four
//! are the state wgpu sets per draw (decision 1's table); the rest of `DynamicValues` (depth
//! bias, stencil masks, line width) comes from the pipeline, as baked into its description. The
//! pipeline is the one `WgpuBackend::draw_pipeline` gives the render thread under the
//! wait-while-compiling setting: a draw whose pipeline is compiling waits or is skipped and
//! counted, and one whose pipeline failed is skipped (`crate::pipeline`, "Draws: wait or skip").
//!
//! A scissor that reaches past the target is cut to it (the same pixels are drawn; wgpu refuses
//! the rectangle as given); no scissor is the whole target.
//!
//! An indirect draw of several draws is one wgpu call per draw, `stride` bytes apart, since
//! wgpu's multi-draw takes no stride; at most [`MAX_INDIRECT_DRAWS`] draws per command.
//!
//! # The index rewrite
//!
//! wgpu has no quads, quad strips, triangle fans, polygons or line loops and no 8-bit indices,
//! a strip pipeline here takes 32-bit indices only (`crate::pipeline::state`, "Topology"), and
//! the host restarts every indexed strip at the restart value (render-backend, "Quads, line
//! loops" and "Primitive restart off"). A draw that needs one of those has its indices rewritten
//! by a compute pass ([`rewrite`], after Eden's `vulkan_quad_indexed.comp`) into 32-bit indices
//! of the topology its pipeline was built with (`crate::pipeline::state`): quads and quad strips
//! as triangle lists in Eden's orders, fans and polygons as triangle lists, line loops as line
//! strips that end on their first vertex, and small indices widened. A non-indexed draw reads
//! the vertex numbers it would have drawn. The draw then draws the rewritten indices, keeping
//! its own base vertex and instances. The rewritten indices live in one buffer per list, with
//! the parameter blocks in another; both are made when the list is encoded and freed with it,
//! host memory the budget does not count, and the list is refused when its rewrites need more
//! than the largest buffer ([`SCRATCH`]).
//!
//! # Typed outputs
//!
//! wgpu checks that a fragment shader's output type fits each colour target: a float output
//! for an integer target fails the pipeline (wgpu-core 30.0.2: `create_render_pipeline` in
//! `src/device/resource.rs` checks each fragment output with `check_texture_format` from
//! `src/validation.rs`, `ColorStateError::IncompatibleFormat`). So the
//! typed-output emulation is needed under wgpu, as `emulation()` already says
//! (`Features::UNTYPED_FRAGMENT_OUTPUTS` is never set, `crate::caps`); pinned by
//! `backend::tests::draws::an_integer_target_needs_a_typed_output`.
//!
//! # Flipped viewports
//!
//! wgpu-core 30.0.2's `set_viewport` (`src/command/render.rs`) refuses a negative width or
//! height and a depth range that is reversed or outside 0..=1, and guest y-flips are common (a
//! lower-left window origin flips every viewport). For a pipeline made with
//! `EmulationFlags::FLIPPED_VIEWPORT`, whose vertex shader flips y and mirrors depth from the
//! driver constants (shader-pipeline slice 17 lowers it), the check sets
//! `starquake_render::emulation::host_viewport` of the draw's first viewport: a negative height
//! made positive over the same rows and a reversed range swapped (decided by the owner,
//! 2026-11-19; the render crate's `emulation` module docs give the arithmetic). What is still
//! refused, with [`VIEWPORT`]: a flipped viewport for a pipeline without the flag (its shader
//! does not flip, so drawing would land on the wrong rows), a negative width, a depth outside
//! 0..=1 (whether the shader should remap one instead, as the spec's row says, is an open owner
//! question: the render crate's `emulation` module docs, "A depth outside 0..=1"), and a size or
//! position past the device's range after the flip. Pinned by
//! `draw::tests::a_flipped_viewport_is_set_positive_only_for_a_flipped_pipeline` and, on a real
//! GPU, `backend::tests::draws::a_flipped_viewport_draws_the_upright_pixels`; that the check
//! reads the pipeline's own flag is pinned with no GPU by
//! `draw::tests::graphics_info_reads_the_flipped_viewport_flag`.
//!
//! # The checks
//!
//! A draw is checked in this order, and the first refusal is the `Err` of `submit` (the module
//! constants name each one):
//!
//! 1. the shared checks (`validate::validate_draw`);
//! 2. the pipeline: live (`validate::NO_PIPELINE`), and a graphics one ([`NEEDS_GRAPHICS`]);
//! 3. every id names a live resource of its kind, and every buffer and texture has the usage
//!    its role needs, as the recording backend checks them (`crate::plan` names the errors);
//! 4. the targets: set, with at least one attachment ([`NO_TARGETS`]), of one size and one
//!    sample count ([`TARGETS_DIFFER`]), no two colour attachments in one mip level and layer
//!    of a texture ([`ATTACHMENT_OVERLAP`]), and the formats and sample count the pipeline was
//!    made for ([`TARGET_MISMATCH`]);
//! 5. the bindings against the pipeline's layout ([`check`]'s order);
//! 6. the vertex buffers: every one the pipeline reads is bound ([`MISSING_VERTEX_BUFFER`]),
//!    each from a 4-byte boundary ([`VERTEX_OFFSET`]); an indexed draw's index buffer from a
//!    multiple of its index size, whether it is bound directly or rewritten ([`INDEX_OFFSET`]);
//! 7. the viewport ([`VIEWPORT`]), after a pipeline with `EmulationFlags::FLIPPED_VIEWPORT` has
//!    had its first viewport made positive (below, "Flipped viewports");
//! 8. the counts: an indirect draw's parameters ([`INDIRECT_REWRITE`], [`INDIRECT_DRAWS`],
//!    [`INDIRECT_STRIDE`], [`INDIRECT_RANGE`]); a direct draw's vertices, instances and indices
//!    within their buffers ([`VERTEX_LIMIT`], [`INDEX_LIMIT`]); the rewrite's sizes
//!    ([`REWRITE_SIZE`], [`SCRATCH`]);
//! 9. the draw's own uses of its resources ([`USAGE_CONFLICT`]).
//!
//! A dispatch is checked by steps 1 to 3 (with [`NEEDS_COMPUTE`]), 5, its indirect parameters
//! ([`INDIRECT_RANGE`]), and 9. Each wgpu rule is mirrored from wgpu-core 30.0.2 in [`rules`],
//! which cites the check. Anything wgpu still reports on the render thread is logged and counted
//! (`WgpuBackend::host_errors`): a bind group wgpu refuses skips its draw.
//!
//! # Divergences
//!
//! Where this backend refuses or draws differently from the recording backend or the guest,
//! each with the test that pins it:
//!
//! - every refusal of steps 4 to 9, which the recording backend does not make (steps 1 to 3
//!   are its own checks): `draw::tests::checks_refuse_in_order` (one fault per step, and every
//!   pair of steps at once), `draw::tests::every_draw_refusal_names_its_rule` and
//!   `draw::tests::every_count_and_use_refusal_names_its_rule`, with a few end to end in
//!   `backend::tests::draws::what_draws_refuse` and
//!   `backend::tests::draws::a_draw_that_samples_its_own_target_is_refused`;
//! - a binding array cannot be filled, since a draw's binding names one resource
//!   ([`BINDING_ARRAY`]): `draw::tests::every_draw_refusal_names_its_rule`;
//! - a sampled float texture whose format only an adapter-specific feature makes filterable is
//!   refused for a filterable binding ([`rules::sample_type_fits`]):
//!   `draw::tests::sample_types_fit_as_wgpu_matches_them`;
//! - a filtering sampler bound where the pipeline's build found the shader samples an
//!   unfilterable texture through it skips the draw at encode time and counts it
//!   (`WgpuBackend::encode_skips`), since that is known only once the build has run
//!   (`crate::pipeline::layout`, "Filtering samplers"): `backend::tests::draws::a_filtering_sampler_on_an_unfilterable_texture_skips_the_draw`;
//! - the rewrite does not honour primitive restart inside quads, a quad strip, a fan or a
//!   polygon, nor inside a line loop from 8-bit or 16-bit indices: the restart value is drawn as
//!   a vertex (Eden's quad rewrite ignores restart too):
//!   `draw::rewrite::tests::the_rewrite_plan_for_every_topology_and_source`. The exception is a
//!   line loop from 32-bit indices: its rewrite copies `0xFFFFFFFF` into a line strip whose
//!   pipeline names 32-bit indices, so the host restarts there whatever the pipeline's restart
//!   setting, as for the 32-bit strips below:
//!   `draw::rewrite::tests::only_a_line_loop_from_32_bit_indices_restarts_after_the_rewrite`;
//! - a 32-bit index equal to `0xFFFFFFFF` in a strip whose pipeline does not restart still
//!   restarts: a strip pipeline names 32-bit indices, which turns restart on under Vulkan, and
//!   Metal always restarts (`crate::pipeline::state`, "Topology"); no wider index exists to
//!   rewrite it into: `draw::rewrite::tests::the_rewrite_plan_for_every_topology_and_source`;
//! - line width is ignored (wgpu draws one-pixel lines) and the dynamic depth bias and stencil
//!   masks are the pipeline's (slice R6);
//! - a layered target draws into its base layer, a 3D target into slice 0 (above).

pub(crate) mod check;
pub mod rewrite;
pub mod rules;

use crate::pipeline::PipelineCell;
use starquake_render::command::{BufferSlice, Rect, Viewport};
use starquake_render::pipeline::Topology;
use starquake_render::{BufferId, PipelineId, SamplerId, TextureId, ViewId};
use std::ops::Range;
use std::sync::Arc;

/// A draw whose pipeline is a compute pipeline (the recording backend's words).
pub const NEEDS_GRAPHICS: &str = "a draw needs a graphics pipeline";
/// A dispatch whose pipeline is a graphics pipeline (the recording backend's words).
pub const NEEDS_COMPUTE: &str = "a dispatch needs a compute pipeline";
/// A draw with no render target.
pub const NO_TARGETS: &str = "a draw needs a render target: a wgpu render pass needs one";
/// Render targets of different sizes or sample counts.
pub const TARGETS_DIFFER: &str =
    "a draw's render targets must share one size and one sample count under wgpu";
/// Two colour targets that draw into one mip level of one layer of a texture (two views of
/// one 3D texture always do: each draws into slice 0).
pub const ATTACHMENT_OVERLAP: &str = "two of a draw's colour targets draw into one mip level and layer, which a wgpu render pass \
     refuses";
/// A pipeline made for other targets.
pub const TARGET_MISMATCH: &str =
    "a draw's pipeline was made for other target formats or another sample count";
/// A binding of the pipeline's layout that the draw does not bind.
pub const MISSING_BINDING: &str = "a binding of the pipeline's layout is not bound";
/// A binding the pipeline's layout does not have.
pub const EXTRA_BINDING: &str = "a group and binding the pipeline's layout does not have is bound";
/// A resource of another kind than the layout's binding.
pub const BINDING_KIND: &str = "a bound resource is not of the kind its binding takes";
/// A binding array.
pub const BINDING_ARRAY: &str = "the wgpu backend cannot fill a binding array from bindings";
/// A bound buffer range off the device's offset alignment.
pub const BUFFER_OFFSET: &str =
    "a bound buffer range does not start at the device's offset alignment";
/// A bound buffer range larger than the device binds.
pub const BUFFER_RANGE: &str = "a bound buffer range is larger than the device binds";
/// A storage buffer range that is not whole words.
pub const STORAGE_SIZE: &str = "a storage buffer binding's size is not a multiple of 4 bytes";
/// A bound buffer range smaller than the shader's declaration.
pub const BINDING_TOO_SMALL: &str =
    "a bound buffer range is smaller than the shader's declaration of it";
/// A view that does not fit its binding's dimension, samples, sample type or format.
pub const VIEW_KIND: &str =
    "a bound view's dimension, samples, sample type or format does not fit its binding";
/// A view of both depth and stencil, bound to a shader.
pub const DEPTH_STENCIL_VIEW: &str = "wgpu cannot bind a view of both depth and stencil";
/// A comparison sampler where a plain one is bound, or the other way round.
pub const SAMPLER_KIND: &str = "a comparison sampler and a plain one are not interchangeable";
/// A resource written by one use and used otherwise in the same draw or dispatch.
pub const USAGE_CONFLICT: &str =
    "a draw or dispatch writes a resource that it also uses another way";
/// A vertex buffer the pipeline reads that the draw does not bind.
pub const MISSING_VERTEX_BUFFER: &str = "a vertex buffer the pipeline reads is not bound";
/// A vertex buffer range off a 4-byte boundary.
pub const VERTEX_OFFSET: &str = "a vertex buffer range does not start on a 4-byte boundary";
/// An index buffer range off a multiple of its index size.
pub const INDEX_OFFSET: &str =
    "an index buffer range does not start on a multiple of its index size";
/// Vertices or instances past the bound vertex buffers.
pub const VERTEX_LIMIT: &str = "a draw reads vertices or instances past its vertex buffers";
/// Indices past the bound index buffer.
pub const INDEX_LIMIT: &str = "a draw reads indices past its index buffer";
/// An indirect draw or dispatch whose parameters are off a 4-byte boundary or past the buffer.
pub const INDIRECT_RANGE: &str =
    "indirect parameters must start on a 4-byte boundary and end inside their buffer";
/// An indirect multi-draw whose stride is too small or not whole words.
pub const INDIRECT_STRIDE: &str =
    "an indirect multi-draw's stride must be at least its parameters' size and a multiple of 4";
/// More indirect draws in one command than [`MAX_INDIRECT_DRAWS`].
pub const INDIRECT_DRAWS: &str = "an indirect draw command draws more than the wgpu backend's cap";
/// An indirect draw that needs its indices rewritten.
pub const INDIRECT_REWRITE: &str = "the wgpu backend cannot rewrite the indices of an indirect draw (quads, fans, line loops, \
     8-bit indices or restart off)";
/// A rewrite whose input or output is larger than the device binds, or whose output count does
/// not fit a `u32`.
pub const REWRITE_SIZE: &str = "a draw's index rewrite is larger than the device can bind";
/// A list whose rewrites need more scratch than the largest buffer.
pub const SCRATCH: &str = "a list's index rewrites need more than the largest buffer";
/// A viewport wgpu cannot set.
pub const VIEWPORT: &str = "wgpu cannot set this viewport (a negative or too large size, a \
                            position out of range, or depths outside 0..=1 or reversed)";

/// The most draws one indirect draw command may make: each is a wgpu call on the render thread
/// (wgpu-hal's Metal backend loops on the CPU for a multi-draw too), so the count bounds the
/// time a hostile command costs. The backend's choice (a guess at what titles need; the guest's
/// own limit is not known), to revisit when a title needs more.
pub const MAX_INDIRECT_DRAWS: u32 = 4096;

/// What the device allows a draw, beyond `Caps`: the alignments and binding sizes wgpu checks
/// (`wgpu::Limits`) and the features its sample types depend on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DrawLimits {
    /// The bound buffer rules ([`rules::check_buffer_binding`]).
    pub binding: rules::BindingLimits,
    /// `max_texture_dimension_2d`, which bounds a viewport ([`rules::check_viewport`]).
    pub max_texture_dimension_2d: u32,
    /// `max_compute_workgroups_per_dimension`, which bounds a rewrite's dispatch.
    pub max_workgroups: u32,
    /// The largest buffer, which bounds a list's rewrite scratch (`Caps::limits`).
    pub max_buffer_size: u64,
    /// The device's features, for texture sample types.
    pub features: wgpu::Features,
}

impl DrawLimits {
    /// The limits of a device with `limits` and `features`, and `max_buffer_size` from `Caps`.
    pub fn new(limits: &wgpu::Limits, features: wgpu::Features, max_buffer_size: u64) -> Self {
        Self {
            binding: rules::BindingLimits {
                uniform_alignment: limits.min_uniform_buffer_offset_alignment,
                storage_alignment: limits.min_storage_buffer_offset_alignment,
                max_uniform: limits.max_uniform_buffer_binding_size,
                max_storage: limits.max_storage_buffer_binding_size,
            },
            max_texture_dimension_2d: limits.max_texture_dimension_2d,
            max_workgroups: limits.max_compute_workgroups_per_dimension,
            max_buffer_size,
            features,
        }
    }
}

/// wgpu's usages for a buffer, beyond the description's (`convert::buffer_usages`): an index
/// buffer is also a storage buffer, so the index rewrite can read it.
pub fn extra_buffer_usages(usage: starquake_render::BufferUsage) -> wgpu::BufferUsages {
    if usage.contains(starquake_render::BufferUsage::INDEX) {
        wgpu::BufferUsages::STORAGE
    } else {
        wgpu::BufferUsages::empty()
    }
}

/// One layout entry of a pipeline, as the check reads it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct EntryInfo {
    /// The binding number.
    pub(crate) binding: u32,
    /// Its type, before the build turns samplers non-filtering.
    pub(crate) ty: wgpu::BindingType,
    /// Whether it is a binding array.
    pub(crate) array: bool,
    /// For a buffer, the size of the largest declaration of it among the stages (wgpu's late
    /// size check); 0 otherwise.
    pub(crate) min_size: u64,
}

/// A pipeline's layout entries: groups 0 to the highest used, each in binding order.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct LayoutInfo {
    pub(crate) groups: Vec<Vec<EntryInfo>>,
}

/// A graphics pipeline's facts the check needs.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct GraphicsInfo {
    /// The guest's topology.
    pub(crate) topology: Topology,
    /// Whether the guest restarts strips.
    pub(crate) primitive_restart: bool,
    /// Colour target formats, up to the last one present.
    pub(crate) colors: Vec<Option<wgpu::TextureFormat>>,
    /// The depth/stencil format.
    pub(crate) depth_stencil: Option<wgpu::TextureFormat>,
    /// Samples per pixel.
    pub(crate) samples: u32,
    /// The vertex buffers, by slot.
    pub(crate) vertex: Vec<rules::VertexStep>,
    /// The layout.
    pub(crate) layout: LayoutInfo,
    /// Whether the pipeline has `EmulationFlags::FLIPPED_VIEWPORT` (module docs, "Flipped
    /// viewports").
    pub(crate) flipped_viewport: bool,
}

/// What the check knows of a pipeline, fixed at creation.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum PipelineInfo {
    /// A graphics pipeline.
    Graphics(Box<GraphicsInfo>),
    /// A compute pipeline and its layout.
    Compute(LayoutInfo),
}

/// A render target attachment: one mip level of one layer (or the first slice of a 3D level).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Attachment {
    /// The texture.
    pub texture: TextureId,
    /// The view's format, in wgpu's terms.
    pub format: wgpu::TextureFormat,
    /// The mip level.
    pub mip: u32,
    /// The layer (0 for a 3D texture).
    pub layer: u32,
    /// Whether the texture is 3D (the attachment is its slice 0).
    pub three_d: bool,
    /// The level's width and height.
    pub size: (u32, u32),
    /// The texture's samples per pixel.
    pub samples: u32,
}

/// The attachments draws render into until the targets change.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Targets {
    /// Colour attachments by target index, up to the last one present.
    pub colors: Vec<Option<Attachment>>,
    /// The depth/stencil attachment.
    pub depth_stencil: Option<Attachment>,
}

/// One resource of a bind group.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bound {
    /// A buffer range.
    Buffer(BufferSlice),
    /// A view.
    View(ViewId),
    /// A sampler, and whether it filters ([`rules::filtering`]).
    Sampler(SamplerId, bool),
}

/// One bind group: its number and its entries in binding order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BoundGroup {
    /// The group number.
    pub group: u32,
    /// The entries, `(binding, resource)`.
    pub entries: Vec<(u32, Bound)>,
}

/// A draw's index rewrite, placed in its list's scratch buffers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RewriteStep {
    /// The parameter block's offset in the list's parameter buffer.
    pub params_offset: u64,
    /// The input binding (buffer, offset, size), or `None` for a non-indexed draw.
    pub input: Option<(BufferId, u64, u64)>,
    /// The output's offset in the list's output buffer.
    pub output_offset: u64,
    /// The indices written.
    pub count: u32,
}

/// What a draw draws.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DrawCall {
    /// Non-indexed.
    Vertices {
        /// Vertex numbers.
        vertices: Range<u32>,
        /// Instance numbers.
        instances: Range<u32>,
    },
    /// Indexed, from the bound index buffer or the rewrite's output.
    Indices {
        /// Index positions.
        indices: Range<u32>,
        /// Added to each index.
        base_vertex: i32,
        /// Instance numbers.
        instances: Range<u32>,
    },
    /// Parameters read from a buffer, one call per draw.
    Indirect {
        /// The buffer.
        buffer: BufferId,
        /// The first draw's parameters.
        offset: u64,
        /// Bytes between draws.
        stride: u64,
        /// Draws.
        count: u32,
        /// Whether the draws are indexed.
        indexed: bool,
    },
}

/// A checked draw (module docs, "A draw"). Two steps are equal when every field but the
/// pipeline's status cell is: the cell is the one `pipeline` named when the list was checked.
#[derive(Clone, Debug)]
pub struct DrawStep {
    /// The pipeline's id.
    pub pipeline: PipelineId,
    /// The pipeline's status, held so the render thread finds it however the id fares.
    pub(crate) cell: Arc<PipelineCell>,
    /// Whether the draw starts a new render pass.
    pub fresh_pass: bool,
    /// Its bind groups.
    pub groups: Vec<BoundGroup>,
    /// Its vertex buffers, by slot.
    pub vertex_buffers: Vec<BufferSlice>,
    /// The guest's index buffer and format, when the draw reads it directly.
    pub index: Option<(BufferSlice, wgpu::IndexFormat)>,
    /// The rewrite, whose output is then the index buffer.
    pub rewrite: Option<RewriteStep>,
    /// The viewport.
    pub viewport: Viewport,
    /// The scissor, within the targets.
    pub scissor: Rect,
    /// The blend constant.
    pub blend_constant: wgpu::Color,
    /// The stencil reference.
    pub stencil_reference: u32,
    /// What it draws.
    pub call: DrawCall,
}

impl PartialEq for DrawStep {
    fn eq(&self, other: &Self) -> bool {
        self.pipeline == other.pipeline
            && self.fresh_pass == other.fresh_pass
            && self.groups == other.groups
            && self.vertex_buffers == other.vertex_buffers
            && self.index == other.index
            && self.rewrite == other.rewrite
            && self.viewport == other.viewport
            && self.scissor == other.scissor
            && self.blend_constant == other.blend_constant
            && self.stencil_reference == other.stencil_reference
            && self.call == other.call
    }
}

/// What a dispatch runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DispatchCall {
    /// Workgroup counts.
    Direct([u32; 3]),
    /// Counts read from a buffer at an offset.
    Indirect(BufferId, u64),
}

/// A checked dispatch; equal as [`DrawStep`]s are.
#[derive(Clone, Debug)]
pub struct DispatchStep {
    /// The pipeline's id.
    pub pipeline: PipelineId,
    /// The pipeline's status.
    pub(crate) cell: Arc<PipelineCell>,
    /// Its bind groups.
    pub groups: Vec<BoundGroup>,
    /// The counts.
    pub call: DispatchCall,
}

impl PartialEq for DispatchStep {
    fn eq(&self, other: &Self) -> bool {
        self.pipeline == other.pipeline && self.groups == other.groups && self.call == other.call
    }
}

/// The scratch one list's rewrites use: their parameter blocks, and the bytes of their output.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RewriteScratch {
    /// The parameter blocks, each at the device's uniform offset alignment.
    pub params: Vec<u8>,
    /// Bytes of the output buffer.
    pub output: u64,
}

#[cfg(test)]
mod tests;
