//! Command lists: the plain data the GPU back thread fills and the render thread encodes
//! (docs/specs/render-backend.md, "Command lists are data" and "Commands").
//!
//! A list holds handles and owned bytes, no references and no host objects, so it can cross
//! threads, be kept by the recording backend, and be serialised for a host-level capture
//! (`serde`; the capture file format, with its magic and version header, is slice R11's). The
//! serialised layout is pinned by `command::tests::golden_bytes`.
//!
//! Render passes, barriers and image layouts are the backend's business: every command says what
//! it reads and writes, and the backend opens host passes and inserts barriers from that.
//!
//! The fence a list signals is not stored in the list: `Backend::submit` assigns it and returns
//! it, so a caller cannot name a wrong one. (The spec describes a list as "a `Vec<Command>` plus
//! the fence value it signals"; the value is the same, held by the backend.)
//!
//! Texture data in a staging slice, a buffer copy or a readback is tightly packed: rows of whole
//! blocks with no padding, row after row, then slice or layer after slice
//! ([`crate::validate::packed_size`]).

use crate::caps::GUEST_COLOR_TARGETS;
use crate::id::{BufferId, PipelineId, ReadbackId, SamplerId, TextureId, ViewId};
use crate::pipeline::DepthBias;
use crate::resource::{Extent3d, FilterMode, Origin3d};
use serde::{Deserialize, Serialize};

/// Host-visible memory the caller fills and then names in an [`Command::Upload`]. Handed out by
/// `Backend::staging`; owned by the list once submitted.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StagingSlice {
    bytes: Vec<u8>,
}

impl StagingSlice {
    /// A slice holding `bytes`. Backends build slices; the GPU crate gets them from
    /// `Backend::staging`.
    pub const fn from_bytes(bytes: Vec<u8>) -> Self {
        Self { bytes }
    }

    /// The contents.
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// The contents, to fill.
    pub fn bytes_mut(&mut self) -> &mut [u8] {
        &mut self.bytes
    }

    /// Length in bytes.
    pub const fn len(&self) -> usize {
        self.bytes.len()
    }

    /// Whether the slice is empty.
    pub const fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }
}

/// A byte range of a buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BufferSlice {
    /// The buffer.
    pub buffer: BufferId,
    /// First byte.
    pub offset: u64,
    /// Length in bytes, at least 1.
    pub size: u64,
}

/// A box of texels in one mip level of a texture.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TextureRegion {
    /// The texture.
    pub texture: TextureId,
    /// Mip level.
    pub mip: u32,
    /// First texel (z: slice of a 3D texture, or first layer).
    pub origin: Origin3d,
    /// Size in texels (depth: slices of a 3D texture, or layers).
    pub extent: Extent3d,
}

/// One side of a copy: a buffer range or a texture region.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CopyRegion {
    /// Bytes of a buffer; texture data in it is tightly packed.
    Buffer(BufferSlice),
    /// Texels of a texture.
    Texture(TextureRegion),
}

/// The source of a copy or readback.
pub type CopySrc = CopyRegion;

/// The destination of a copy or upload.
pub type CopyDst = CopyRegion;

/// A copy between buffers and textures, in any combination.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CopyDesc {
    /// Read from.
    pub src: CopySrc,
    /// Written to.
    pub dst: CopyDst,
}

/// A rectangle in texels.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Rect {
    /// Left column.
    pub x: u32,
    /// Top row.
    pub y: u32,
    /// Width.
    pub width: u32,
    /// Height.
    pub height: u32,
}

/// A scaled, filtered copy between views (the 2D engine, the present blit).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BlitDesc {
    /// Source view.
    pub src: ViewId,
    /// Source rectangle.
    pub src_rect: Rect,
    /// Destination view.
    pub dst: ViewId,
    /// Destination rectangle.
    pub dst_rect: Rect,
    /// Filter used when scaling.
    pub filter: FilterMode,
}

/// The value a clear writes.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum ClearValue {
    /// A colour for a float or normalised target.
    Float([f32; 4]),
    /// A colour for a signed integer target.
    Sint([i32; 4]),
    /// A colour for an unsigned integer target.
    Uint([u32; 4]),
    /// Depth only.
    Depth(f32),
    /// Stencil only.
    Stencil(u32),
    /// Depth and stencil.
    DepthStencil {
        /// Depth value.
        depth: f32,
        /// Stencil value.
        stencil: u32,
    },
}

/// A clear of a view, whole or within a rectangle.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ClearDesc {
    /// The view cleared.
    pub target: ViewId,
    /// What it is cleared to.
    pub value: ClearValue,
    /// The area, or the whole view.
    pub rect: Option<Rect>,
}

/// The views draws render into, until the next [`Command::SetRenderTargets`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RenderTargets {
    /// Colour views, by the guest's target index.
    pub colors: [Option<ViewId>; GUEST_COLOR_TARGETS],
    /// Depth/stencil view.
    pub depth_stencil: Option<ViewId>,
}

/// A viewport.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Viewport {
    /// Left edge.
    pub x: f32,
    /// Top edge.
    pub y: f32,
    /// Width (may be negative where the guest flips).
    pub width: f32,
    /// Height (may be negative where the guest flips).
    pub height: f32,
    /// Depth mapped from the near plane.
    pub min_depth: f32,
    /// Depth mapped from the far plane.
    pub max_depth: f32,
}

/// Draw state that may be dynamic on the host (render-backend, "Commands"). The backend applies
/// per draw what [`crate::Caps::dynamic_state`] says it can; the rest it took from the pipeline.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DynamicValues {
    /// Viewports, at least one.
    pub viewports: Vec<Viewport>,
    /// Scissor rectangles, one per viewport or none.
    pub scissors: Vec<Rect>,
    /// Blend constant colour.
    pub blend_constant: [f32; 4],
    /// Stencil reference.
    pub stencil_reference: u32,
    /// Stencil compare mask.
    pub stencil_read_mask: u32,
    /// Stencil write mask.
    pub stencil_write_mask: u32,
    /// Depth bias.
    pub depth_bias: DepthBias,
    /// Line width.
    pub line_width: f32,
}

/// What a binding points at.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BindingResource {
    /// A buffer range (uniform or storage).
    Buffer(BufferSlice),
    /// A texture view (sampled or storage).
    View(ViewId),
    /// A sampler.
    Sampler(SamplerId),
}

/// One resource bound at a group and binding of the pipeline's shaders.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Binding {
    /// Bind group number.
    pub group: u32,
    /// Binding number.
    pub binding: u32,
    /// The resource.
    pub resource: BindingResource,
}

/// The size of an index.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum IndexFormat {
    /// 8-bit indices (the guest has them; most hosts do not).
    U8,
    /// 16-bit indices.
    U16,
    /// 32-bit indices.
    U32,
}

/// The index buffer of an indexed draw.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct IndexBufferBinding {
    /// The indices.
    pub slice: BufferSlice,
    /// Index size.
    pub format: IndexFormat,
}

/// A draw's counts, or where to read them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DrawParams {
    /// Non-indexed.
    Direct {
        /// Vertices per instance.
        vertex_count: u32,
        /// Instances.
        instance_count: u32,
        /// First vertex.
        first_vertex: u32,
        /// First instance.
        first_instance: u32,
    },
    /// Indexed.
    Indexed {
        /// Indices per instance.
        index_count: u32,
        /// Instances.
        instance_count: u32,
        /// First index.
        first_index: u32,
        /// Added to each index.
        base_vertex: i32,
        /// First instance.
        first_instance: u32,
    },
    /// Parameters read from a buffer.
    Indirect {
        /// The buffer holding them.
        buffer: BufferId,
        /// Byte offset of the first draw's parameters.
        offset: u64,
        /// Draws, at least 1; more than 1 needs
        /// [`crate::caps::Features::MULTI_DRAW_INDIRECT`].
        draw_count: u32,
        /// Bytes between consecutive draws' parameters.
        stride: u32,
        /// Whether the draws are indexed.
        indexed: bool,
    },
}

/// A draw (render-backend, "Commands").
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DrawDesc {
    /// The graphics pipeline.
    pub pipeline: PipelineId,
    /// Resources bound for the pipeline's shaders.
    pub bindings: Vec<Binding>,
    /// Vertex buffers, by the pipeline's vertex buffer index.
    pub vertex_buffers: Vec<BufferSlice>,
    /// Index buffer, for indexed draws.
    pub index_buffer: Option<IndexBufferBinding>,
    /// State that may be set per draw.
    pub dynamic: DynamicValues,
    /// Counts.
    pub params: DrawParams,
}

/// A dispatch's group counts, or where to read them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DispatchParams {
    /// Workgroups in x, y and z.
    Direct {
        /// Workgroups in x.
        x: u32,
        /// Workgroups in y.
        y: u32,
        /// Workgroups in z.
        z: u32,
    },
    /// Counts read from a buffer.
    Indirect {
        /// The buffer.
        buffer: BufferId,
        /// Byte offset.
        offset: u64,
    },
}

/// A compute dispatch.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DispatchDesc {
    /// The compute pipeline.
    pub pipeline: PipelineId,
    /// Resources bound for its shader.
    pub bindings: Vec<Binding>,
    /// Counts.
    pub params: DispatchParams,
}

/// How a presented image is turned on its way to the screen.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PresentTransform {
    /// As is.
    Identity,
    /// Mirrored left to right.
    FlipHorizontal,
    /// Mirrored top to bottom.
    FlipVertical,
    /// Turned 90 degrees clockwise.
    Rotate90,
    /// Turned 180 degrees.
    Rotate180,
    /// Turned 270 degrees clockwise.
    Rotate270,
}

/// A present of a view (render-backend, "Presenting a guest framebuffer").
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PresentDesc {
    /// The view shown.
    pub view: ViewId,
    /// The part of it shown.
    pub src_rect: Rect,
    /// How it is turned.
    pub transform: PresentTransform,
    /// The frame number, for feedback and dumps.
    pub frame: u64,
    /// Whether the frame is read back for a frame dump.
    pub dump: bool,
}

/// Guest submission and draw ids, for captures and logs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Marker {
    /// The guest submission.
    pub submit: u64,
    /// The draw within it.
    pub draw: u64,
}

/// One command (render-backend, "Commands").
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Command {
    /// Copies a staging slice into a buffer range or texture region.
    Upload {
        /// The bytes.
        from: StagingSlice,
        /// Where they go.
        to: CopyDst,
    },
    /// Copies between buffers and textures.
    Copy(CopyDesc),
    /// Scaled, filtered copy between views.
    Blit(BlitDesc),
    /// Clears a view.
    Clear(ClearDesc),
    /// Sets the views later draws render into.
    SetRenderTargets(RenderTargets),
    /// Draws.
    Draw(DrawDesc),
    /// Dispatches compute work.
    Dispatch(DispatchDesc),
    /// Copies a buffer range or texture region out, collected with `Backend::take_readback`
    /// once the list's fence has passed.
    Readback {
        /// What is read.
        from: CopySrc,
        /// The id to collect it by.
        id: ReadbackId,
    },
    /// Presents a view.
    Present(PresentDesc),
    /// Marks a point in the guest's work.
    Marker(Marker),
}

/// A list of commands, handed to `Backend::submit` as a whole.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct CommandList {
    /// The commands, in order.
    pub commands: Vec<Command>,
}

impl CommandList {
    /// An empty list.
    pub const fn new() -> Self {
        Self {
            commands: Vec::new(),
        }
    }

    /// Appends a command.
    pub fn push(&mut self, command: Command) {
        self.commands.push(command);
    }
}

#[cfg(test)]
pub(crate) mod tests;
