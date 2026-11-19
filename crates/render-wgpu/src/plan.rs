//! The check pass of `submit`: a list resolved against the live resources, checked, and turned
//! into the [`Step`]s the render thread encodes (docs/specs/render-backend.md, "Handles,
//! lifetimes and fences": a stale or foreign id is an error when the list is encoded).
//!
//! The check runs on the submitting thread, under the backend's lock, before the list is queued,
//! so every refusal is the `Err` of `submit` and nothing is queued, fenced or reserved (a
//! render capture sees every refusal where the trait reports it, render-backend "What R5 and R6
//! must do now"). Commands are checked in list order, and the first error is returned. Within a
//! command the checks run in this order:
//!
//! 1. its ids name live resources of their kind (a view's texture too);
//! 2. the shared checks of `starquake_render::validate` (bounds, usages, formats, overlaps);
//! 3. the rules wgpu adds ([`crate::rules`]);
//! 4. these, which the shared checks are not given what they need for (the recording backend
//!    applies the same ones): a readback id is not already waiting to be taken or in flight,
//!    and is not used twice in the list; colour target views store colour and depth/stencil
//!    target views do not, both from textures with the render-target usage.
//!
//! One exception: `SetRenderTargets` checks its target count (2) before its ids (1), as the
//! recording backend does. Errors wgpu would report later on the render thread are what rule 3
//! exists to prevent; the backend counts any that still occur (`WgpuBackend::host_errors`).
//!
//! Blits, draws and dispatches are refused with [`BLITS_NOT_YET`] and [`DRAWS_NOT_YET`]: the
//! wgpu backend has no pipelines until slices R6 and R7 (the present blit is R8's).
//!
//! The steps carry wgpu's numbers: texture boxes in whole blocks (a region that ends at a
//! compressed level's edge is rounded up to the block, which wgpu calls the physical size),
//! and the tightly packed row layout of every buffer side (`starquake_render::validate::packed_size`).

use crate::format::texture_format;
use crate::rules::{self, HostTexture};
use starquake_render::command::{
    ClearDesc, ClearValue, CopyRegion, PresentDesc, Rect, RenderTargets, TextureRegion,
};
use starquake_render::format::FormatAspect;
use starquake_render::resource::{Extent3d, Origin3d, TextureDimension};
use starquake_render::validate::{self, Resolved, ResolvedView};
use starquake_render::{
    BufferDesc, BufferId, Caps, Command, CommandList, PixelFormat, ReadbackId, RenderError, Slots,
    StagingSlice, TextureDesc, TextureId, TextureUsage, ViewDesc, ViewId,
};
use std::collections::BTreeSet;

/// A list with a blit.
pub const BLITS_NOT_YET: &str = "the wgpu backend cannot blit yet (pipelines arrive with slice R6)";
/// A list with a draw or a dispatch.
pub const DRAWS_NOT_YET: &str =
    "the wgpu backend cannot draw or dispatch yet (draws arrive with slice R7)";
/// An id that names no live buffer.
pub const NO_BUFFER: &str = "no live buffer has this id";
/// An id that names no live texture.
pub const NO_TEXTURE: &str = "no live texture has this id";
/// An id that names no live view.
pub const NO_VIEW: &str = "no live view has this id";
/// A readback id already waiting, in flight, or used earlier in the list.
pub const READBACK_TAKEN: &str = "a readback id is already waiting to be taken";
/// A colour target view that does not store colour.
pub const COLOR_TARGET: &str = "a colour target view must store colour";
/// A depth/stencil target view that stores colour.
pub const DEPTH_TARGET: &str = "a depth/stencil target view must store depth or stencil";

/// A live texture: its description, the shape wgpu was given, and the bytes it takes.
#[derive(Clone, Copy, Debug)]
pub struct TextureRec {
    /// As created.
    pub desc: TextureDesc,
    /// What wgpu makes.
    pub host: HostTexture,
    /// Every level, every sample, for the budget.
    pub bytes: u64,
}

/// A live view: its texture and description.
#[derive(Clone, Copy, Debug)]
pub struct ViewRec {
    /// The texture viewed.
    pub texture: TextureId,
    /// As created.
    pub desc: ViewDesc,
}

/// The live resources, as the submitting side sees them.
#[derive(Debug, Default)]
pub struct Tables {
    /// Buffers.
    pub buffers: Slots<BufferDesc>,
    /// Textures.
    pub textures: Slots<TextureRec>,
    /// Views.
    pub views: Slots<ViewRec>,
    /// Samplers.
    pub samplers: Slots<starquake_render::SamplerDesc>,
}

impl Tables {
    fn buffer(&self, id: BufferId) -> Result<&BufferDesc, RenderError> {
        self.buffers
            .get(id.index(), id.generation())
            .ok_or(RenderError::Invalid(NO_BUFFER))
    }

    fn texture(&self, id: TextureId) -> Result<&TextureRec, RenderError> {
        self.textures
            .get(id.index(), id.generation())
            .ok_or(RenderError::Invalid(NO_TEXTURE))
    }

    /// A live view and its live texture.
    pub fn view(&self, id: ViewId) -> Result<(&ViewRec, &TextureRec), RenderError> {
        let view = self
            .views
            .get(id.index(), id.generation())
            .ok_or(RenderError::Invalid(NO_VIEW))?;
        Ok((view, self.texture(view.texture)?))
    }
}

/// A box of whole blocks in one mip level of a texture, in wgpu's terms.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TexelBox {
    /// The texture.
    pub texture: TextureId,
    /// The mip level.
    pub mip: u32,
    /// First texel; z is the slice of a 3D texture or the first layer.
    pub origin: Origin3d,
    /// Size in texels, rounded up to whole blocks; depth is slices or layers.
    pub extent: Extent3d,
}

/// Tightly packed rows of blocks outside a texture.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rows {
    /// Bytes of one row of blocks.
    pub row_bytes: u32,
    /// Rows of blocks in one slice or layer.
    pub rows: u32,
    /// Slices or layers.
    pub images: u32,
}

impl Rows {
    /// Bytes of all rows (saturating; a checked region's rows fit, `validate::packed_size`).
    pub fn total(self) -> u64 {
        u64::from(self.row_bytes)
            .saturating_mul(self.rows.into())
            .saturating_mul(self.images.into())
    }
}

/// The layers or slices a clear writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layers {
    /// Layers `first..first + count` of a 1D, 2D or cube texture.
    Array {
        /// First layer.
        first: u32,
        /// How many.
        count: u32,
    },
    /// Every slice of a 3D texture's level, `0..count`.
    Slices {
        /// How many.
        count: u32,
    },
}

/// What a clear writes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ClearOps {
    /// A colour, as wgpu takes it (integers exactly: every `u32` and `i32` is an `f64`).
    Color(wgpu::Color),
    /// Depth, stencil or both; `None` leaves that aspect as it is.
    DepthStencil {
        /// The depth value.
        depth: Option<f32>,
        /// The stencil value.
        stencil: Option<u32>,
    },
}

/// A clear of a view.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClearStep {
    /// The texture.
    pub texture: TextureId,
    /// The view's format, in wgpu's terms.
    pub format: wgpu::TextureFormat,
    /// The view's base mip level.
    pub mip: u32,
    /// The layers or slices.
    pub layers: Layers,
    /// The rectangle, or `None` for the whole level.
    pub rect: Option<Rect>,
    /// The value.
    pub ops: ClearOps,
}

/// Where a readback reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReadFrom {
    /// A byte range of a buffer.
    Buffer {
        /// The buffer.
        buffer: BufferId,
        /// First byte.
        offset: u64,
        /// Length.
        size: u64,
    },
    /// A texture box, packed into rows.
    Texture {
        /// The box.
        at: TexelBox,
        /// Its rows.
        rows: Rows,
    },
}

/// One checked command, ready to encode.
#[derive(Clone, Debug, PartialEq)]
pub enum Step {
    /// `Queue::write_buffer`.
    WriteBuffer {
        /// The buffer.
        buffer: BufferId,
        /// First byte.
        offset: u64,
        /// The bytes.
        bytes: Vec<u8>,
    },
    /// `Queue::write_texture`.
    WriteTexture {
        /// Where.
        at: TexelBox,
        /// The bytes' rows.
        rows: Rows,
        /// The bytes.
        bytes: Vec<u8>,
    },
    /// A buffer-to-buffer copy.
    CopyBuffer {
        /// Source buffer.
        src: BufferId,
        /// Source offset.
        src_offset: u64,
        /// Destination buffer.
        dst: BufferId,
        /// Destination offset.
        dst_offset: u64,
        /// Bytes.
        size: u64,
    },
    /// A buffer-to-texture copy.
    BufferToTexture {
        /// Source buffer.
        buffer: BufferId,
        /// Its first byte.
        offset: u64,
        /// Its rows.
        rows: Rows,
        /// Destination.
        at: TexelBox,
    },
    /// A texture-to-buffer copy.
    TextureToBuffer {
        /// Source.
        at: TexelBox,
        /// Destination buffer.
        buffer: BufferId,
        /// Its first byte.
        offset: u64,
        /// Its rows.
        rows: Rows,
    },
    /// A texture-to-texture copy of equal extents.
    TextureToTexture {
        /// Source.
        src: TexelBox,
        /// Destination.
        dst: TexelBox,
    },
    /// A clear.
    Clear(ClearStep),
    /// A readback.
    Readback {
        /// The id to collect it by.
        id: ReadbackId,
        /// What is read.
        from: ReadFrom,
    },
    /// A present to the offscreen target.
    Present {
        /// The command.
        desc: PresentDesc,
        /// The view's format.
        format: PixelFormat,
        /// What a dumped frame reads, or `None` when the frame is not dumped or its format is
        /// compressed (then a dump's pixels are unknown).
        read: Option<(TexelBox, Rows)>,
    },
}

/// A checked list: its steps and the readback ids it reserves.
#[derive(Debug, Default)]
pub struct Checked {
    /// The steps, in order.
    pub steps: Vec<Step>,
    /// The readback ids the list uses, in order.
    pub readbacks: Vec<ReadbackId>,
}

/// Checks `list` and turns it into steps (module docs). `taken` holds the readback ids waiting
/// to be taken or in flight.
pub fn check(
    tables: &Tables,
    caps: &Caps,
    taken: &BTreeSet<ReadbackId>,
    list: CommandList,
) -> Result<Checked, RenderError> {
    let mut out = Checked::default();
    let mut used = BTreeSet::new();
    for command in list.commands {
        let step = match command {
            Command::Upload { from, to } => upload_step(tables, from, &to)?,
            Command::Copy(copy) => {
                let src = resolve(tables, &copy.src)?;
                let dst = resolve(tables, &copy.dst)?;
                validate::validate_copy(src, dst)?;
                copy_step(src, dst)?
            }
            Command::Blit(_) => return Err(RenderError::Unsupported(BLITS_NOT_YET)),
            Command::Draw(_) | Command::Dispatch(_) => {
                return Err(RenderError::Unsupported(DRAWS_NOT_YET));
            }
            Command::Clear(clear) => clear_step(tables, &clear)?,
            Command::SetRenderTargets(targets) => {
                render_targets(tables, caps, &targets)?;
                continue;
            }
            Command::Readback { from, id } => {
                let resolved = resolve(tables, &from)?;
                validate::validate_readback(resolved)?;
                let from = match resolved {
                    Resolved::Buffer { slice, .. } => ReadFrom::Buffer {
                        buffer: slice.buffer,
                        offset: slice.offset,
                        size: slice.size,
                    },
                    Resolved::Texture { desc, region } => {
                        rules::check_buffer_texture(desc.format, 0)?;
                        let (at, rows) = texel_box(desc.format, region)?;
                        ReadFrom::Texture { at, rows }
                    }
                };
                if taken.contains(&id) || !used.insert(id) {
                    return Err(RenderError::Invalid(READBACK_TAKEN));
                }
                out.readbacks.push(id);
                Step::Readback { id, from }
            }
            Command::Present(present) => present_step(tables, present)?,
            Command::Marker(_) => continue,
        };
        out.steps.push(step);
    }
    Ok(out)
}

/// The step for an upload of `from` into `to`.
fn upload_step(tables: &Tables, from: StagingSlice, to: &CopyRegion) -> Result<Step, RenderError> {
    let len = u64::try_from(from.len()).unwrap_or(u64::MAX);
    let resolved = resolve(tables, to)?;
    validate::validate_upload(len, resolved)?;
    Ok(match resolved {
        Resolved::Buffer { slice, .. } => {
            rules::check_buffer_range(slice.offset, slice.size)?;
            Step::WriteBuffer {
                buffer: slice.buffer,
                offset: slice.offset,
                bytes: from.into_bytes(),
            }
        }
        Resolved::Texture { desc, region } => {
            rules::check_buffer_texture(desc.format, 0)?;
            let (at, rows) = texel_box(desc.format, region)?;
            Step::WriteTexture {
                at,
                rows,
                bytes: from.into_bytes(),
            }
        }
    })
}

/// The step for a clear. A rectangle that covers the whole level becomes `None`, so the render
/// thread clears it as a whole level.
fn clear_step(tables: &Tables, clear: &ClearDesc) -> Result<Step, RenderError> {
    let (view, tex) = tables.view(clear.target)?;
    let target = ResolvedView {
        texture: &tex.desc,
        view: &view.desc,
    };
    validate::validate_clear(target, clear)?;
    let extent = target.base_extent()?;
    let (width, height) = (extent.width, extent.height);
    rules::check_clear_area(
        &tex.desc,
        view.desc.format,
        clear.rect.as_ref(),
        width,
        height,
    )?;
    let layers = if tex.desc.dimension == TextureDimension::D3 {
        Layers::Slices {
            count: extent.depth_or_layers,
        }
    } else {
        Layers::Array {
            first: view.desc.base_layer,
            count: view.desc.layer_count,
        }
    };
    let whole = |r: &Rect| r.x == 0 && r.y == 0 && r.width == width && r.height == height;
    Ok(Step::Clear(ClearStep {
        texture: view.texture,
        format: host_format(view.desc.format)?,
        mip: view.desc.base_mip,
        layers,
        rect: clear.rect.filter(|r| !whole(r)),
        ops: clear_ops(clear.value),
    }))
}

/// The step for a present: what a dumped frame reads, from the view's base level at its base
/// layer (slice 0 of a 3D texture), as the recording backend reads it.
fn present_step(tables: &Tables, present: PresentDesc) -> Result<Step, RenderError> {
    let (view, tex) = tables.view(present.view)?;
    validate::validate_present(
        ResolvedView {
            texture: &tex.desc,
            view: &view.desc,
        },
        &present,
    )?;
    let format = view.desc.format;
    let read = if present.dump && !format.is_compressed() {
        let z = if tex.desc.dimension == TextureDimension::D3 {
            0
        } else {
            view.desc.base_layer
        };
        let r = present.src_rect;
        let region = TextureRegion {
            texture: view.texture,
            mip: view.desc.base_mip,
            origin: Origin3d { x: r.x, y: r.y, z },
            extent: Extent3d {
                width: r.width,
                height: r.height,
                depth_or_layers: 1,
            },
        };
        Some(texel_box(format, &region)?)
    } else {
        None
    };
    Ok(Step::Present {
        desc: present,
        format,
        read,
    })
}

/// The wgpu form of a format a live texture or view has; every such format has one (a texture
/// needs format support, and `rules::check_view` refuses a view format without a form).
fn host_format(format: PixelFormat) -> Result<wgpu::TextureFormat, RenderError> {
    texture_format(format).ok_or(RenderError::Unsupported(rules::VIEW_FORMAT))
}

/// The description a copy region names.
fn resolve<'t>(tables: &'t Tables, region: &'t CopyRegion) -> Result<Resolved<'t>, RenderError> {
    Ok(match region {
        CopyRegion::Buffer(slice) => Resolved::Buffer {
            desc: tables.buffer(slice.buffer)?,
            slice,
        },
        CopyRegion::Texture(region) => Resolved::Texture {
            desc: &tables.texture(region.texture)?.desc,
            region,
        },
    })
}

/// A validated region of a texture of `format` as whole blocks, and the rows it packs into.
pub fn texel_box(
    format: PixelFormat,
    region: &TextureRegion,
) -> Result<(TexelBox, Rows), RenderError> {
    let (bw, bh) = (format.block_width(), format.block_height());
    let wide = region.extent.width.div_ceil(bw);
    let high = region.extent.height.div_ceil(bh);
    let overflow = RenderError::Overflow {
        what: "texture box size",
    };
    let width = wide.checked_mul(bw).ok_or(overflow.clone())?;
    let height = high.checked_mul(bh).ok_or(overflow.clone())?;
    let row_bytes = wide.checked_mul(format.bytes_per_block()).ok_or(overflow)?;
    Ok((
        TexelBox {
            texture: region.texture,
            mip: region.mip,
            origin: region.origin,
            extent: Extent3d {
                width,
                height,
                depth_or_layers: region.extent.depth_or_layers,
            },
        },
        Rows {
            row_bytes,
            rows: high,
            images: region.extent.depth_or_layers,
        },
    ))
}

/// The step for a copy that has passed `validate_copy`, after wgpu's rules.
fn copy_step(src: Resolved<'_>, dst: Resolved<'_>) -> Result<Step, RenderError> {
    Ok(match (src, dst) {
        (Resolved::Buffer { slice: s, .. }, Resolved::Buffer { slice: d, .. }) => {
            rules::check_buffer_range(s.offset, s.size)?;
            rules::check_buffer_range(d.offset, d.size)?;
            rules::check_buffer_copy(s.buffer, d.buffer)?;
            Step::CopyBuffer {
                src: s.buffer,
                src_offset: s.offset,
                dst: d.buffer,
                dst_offset: d.offset,
                size: s.size,
            }
        }
        (Resolved::Buffer { slice, .. }, Resolved::Texture { desc, region }) => {
            rules::check_buffer_texture(desc.format, slice.offset)?;
            let (at, rows) = texel_box(desc.format, region)?;
            Step::BufferToTexture {
                buffer: slice.buffer,
                offset: slice.offset,
                rows,
                at,
            }
        }
        (Resolved::Texture { desc, region }, Resolved::Buffer { slice, .. }) => {
            rules::check_buffer_texture(desc.format, slice.offset)?;
            let (at, rows) = texel_box(desc.format, region)?;
            Step::TextureToBuffer {
                at,
                buffer: slice.buffer,
                offset: slice.offset,
                rows,
            }
        }
        (
            Resolved::Texture {
                desc: sd,
                region: sr,
            },
            Resolved::Texture {
                desc: dd,
                region: dr,
            },
        ) => {
            rules::check_texture_copy(sd, sr, dd, dr)?;
            Step::TextureToTexture {
                src: texel_box(sd.format, sr)?.0,
                dst: texel_box(dd.format, dr)?.0,
            }
        }
    })
}

/// What a clear value writes, in wgpu's terms.
pub fn clear_ops(value: ClearValue) -> ClearOps {
    let color = |c: [f64; 4]| {
        ClearOps::Color(wgpu::Color {
            r: c[0],
            g: c[1],
            b: c[2],
            a: c[3],
        })
    };
    match value {
        ClearValue::Float(c) => color(c.map(f64::from)),
        ClearValue::Sint(c) => color(c.map(f64::from)),
        ClearValue::Uint(c) => color(c.map(f64::from)),
        ClearValue::Depth(depth) => ClearOps::DepthStencil {
            depth: Some(depth),
            stencil: None,
        },
        ClearValue::Stencil(stencil) => ClearOps::DepthStencil {
            depth: None,
            stencil: Some(stencil),
        },
        ClearValue::DepthStencil { depth, stencil } => ClearOps::DepthStencil {
            depth: Some(depth),
            stencil: Some(stencil),
        },
    }
}

/// Checks a `SetRenderTargets` (module docs, rule 4). The wgpu backend opens no pass until
/// draws exist (slice R7), so the command changes nothing.
fn render_targets(
    tables: &Tables,
    caps: &Caps,
    targets: &RenderTargets,
) -> Result<(), RenderError> {
    validate::validate_render_targets(caps, targets)?;
    let colors = targets.colors.iter().flatten().map(|v| (*v, true));
    for (id, color) in colors.chain(targets.depth_stencil.map(|v| (v, false))) {
        let (view, tex) = tables.view(id)?;
        if !tex.desc.usage.contains(TextureUsage::RENDER_TARGET) {
            return Err(RenderError::MissingUsage {
                resource: "render target texture",
                usage: "render target",
            });
        }
        let is_color = view.desc.format.aspect() == FormatAspect::Color;
        match (color, is_color) {
            (true, false) => return Err(RenderError::Invalid(COLOR_TARGET)),
            (false, true) => return Err(RenderError::Invalid(DEPTH_TARGET)),
            _ => {}
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
