//! The check pass of `submit`: every command of a list resolved against the live resources and
//! validated, before anything changes, into the [`Op`]s the apply pass carries out.
//!
//! A command's ids must name live resources of their kind; its descriptions must pass
//! [`crate::validate`]; and these rules, which the validation functions cannot see because they
//! need ids, apply too:
//!
//! - a copy or blit must not read and write overlapping bytes of one buffer or one texture's mip
//!   level (Vulkan leaves overlapping copies undefined; wgpu refuses copies within one buffer;
//!   guess: both from memory of their documentation, to be confirmed by slice R5);
//! - a readback id must not already be waiting to be taken, nor be used twice in the list;
//! - colour target views store colour and depth/stencil target views do not, and both come from
//!   textures with the render-target usage;
//! - a draw needs a graphics pipeline and a dispatch a compute pipeline; buffers bound to a draw
//!   or dispatch have the uniform or storage usage, bound views come from textures with the
//!   sampled or storage usage, vertex buffers have the vertex usage, the index buffer the index
//!   usage, and an indirect parameter buffer the indirect usage. (How far an indirect draw reads
//!   is not checked: the backend that reads it, slice R7, owns that bound.)

use super::store::BlockBox;
use super::{
    NO_PIPELINE, NO_SAMPLER, PipelineKind, State, TextureRec, ViewRec, gone, level_shape, texel,
};
use crate::caps::Caps;
use crate::command::{
    Binding, BindingResource, BlitDesc, BufferSlice, ClearDesc, Command, CommandList, CopyRegion,
    DispatchParams, DrawDesc, DrawParams, PresentDesc, Rect, RenderTargets, TextureRegion,
};
use crate::error::RenderError;
use crate::format::{FormatAspect, PixelFormat};
use crate::id::{BufferId, PipelineId, ReadbackId, TextureId, ViewId};
use crate::resource::{BufferUsage, TextureDimension, TextureUsage};
use crate::validate::{self, Resolved, ResolvedView};
use std::collections::BTreeSet;

/// Bytes of one resource a command reads or writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Place {
    /// A byte range of a buffer.
    Buffer {
        /// The buffer.
        id: BufferId,
        /// First byte.
        offset: u64,
        /// Length.
        size: u64,
    },
    /// A box of blocks in one mip level of a texture.
    Texture {
        /// The texture.
        id: TextureId,
        /// The mip level.
        mip: u32,
        /// The blocks.
        area: BlockBox,
    },
}

/// What one checked command does to the model.
#[derive(Debug)]
pub(super) enum Op<'a> {
    /// Nothing (render target changes, markers).
    Nothing,
    /// Known bytes written.
    Upload { bytes: &'a [u8], to: Place },
    /// Bytes and their known flags moved.
    Copy { from: Place, to: Place },
    /// One texel written over a box.
    Fill { to: Place, texel: Vec<u8> },
    /// Bytes made unknown.
    Forget(Vec<Place>),
    /// Bytes read out under an id.
    Readback { from: Place, id: ReadbackId },
    /// A present, with the place its pixels come from when they can be read.
    Present {
        desc: &'a PresentDesc,
        format: PixelFormat,
        from: Option<Place>,
    },
}

/// State that lasts for one list.
#[derive(Default)]
struct ListState {
    targets: RenderTargets,
    readbacks: BTreeSet<ReadbackId>,
}

/// Checks every command of `list` and returns what each does, or the first error.
pub(super) fn check<'a>(
    state: &State,
    caps: &Caps,
    list: &'a CommandList,
) -> Result<Vec<Op<'a>>, RenderError> {
    let mut list_state = ListState::default();
    list.commands
        .iter()
        .map(|c| command(state, caps, &mut list_state, c))
        .collect()
}

fn command<'a>(
    state: &State,
    caps: &Caps,
    list: &mut ListState,
    command: &'a Command,
) -> Result<Op<'a>, RenderError> {
    match command {
        Command::Upload { from, to } => {
            let (resolved, place) = resolve(state, to)?;
            validate::validate_upload(len_u64(from.len()), resolved)?;
            Ok(Op::Upload {
                bytes: from.bytes(),
                to: place,
            })
        }
        Command::Copy(copy) => {
            let (src, from) = resolve(state, &copy.src)?;
            let (dst, to) = resolve(state, &copy.dst)?;
            validate::validate_copy(src, dst)?;
            if overlaps(&from, &to) {
                return Err(RenderError::Invalid(
                    "a copy's source and destination overlap",
                ));
            }
            Ok(Op::Copy { from, to })
        }
        Command::Blit(blit) => blit_op(state, blit),
        Command::Clear(clear) => clear_op(state, clear),
        Command::SetRenderTargets(targets) => {
            validate::validate_render_targets(caps, targets)?;
            for (id, color) in targets
                .colors
                .iter()
                .flatten()
                .map(|v| (*v, true))
                .chain(targets.depth_stencil.map(|v| (v, false)))
            {
                render_target(state, id, color)?;
            }
            list.targets = *targets;
            Ok(Op::Nothing)
        }
        Command::Draw(draw) => draw_op(state, caps, list, draw),
        Command::Dispatch(dispatch) => {
            validate::validate_dispatch(caps, dispatch)?;
            pipeline(state, dispatch.pipeline, false)?;
            let mut forget = Vec::new();
            bindings(state, &dispatch.bindings, &mut forget)?;
            if let DispatchParams::Indirect { buffer, .. } = dispatch.params {
                indirect(state, buffer)?;
            }
            Ok(Op::Forget(forget))
        }
        Command::Readback { from, id } => {
            let (resolved, place) = resolve(state, from)?;
            validate::validate_readback(resolved)?;
            if state.readbacks.contains_key(id) || !list.readbacks.insert(*id) {
                return Err(RenderError::Invalid(
                    "a readback id is already waiting to be taken",
                ));
            }
            Ok(Op::Readback {
                from: place,
                id: *id,
            })
        }
        Command::Present(present) => {
            let (view, tex) = state.view(present.view)?;
            validate::validate_present(resolved(view, tex), present)?;
            let format = view.desc.format;
            let from = if format.is_compressed() {
                None
            } else {
                Some(rect_place(view, tex, &present.src_rect)?)
            };
            Ok(Op::Present {
                desc: present,
                format,
                from,
            })
        }
        Command::Marker(_) => Ok(Op::Nothing),
    }
}

fn blit_op<'a>(state: &State, blit: &BlitDesc) -> Result<Op<'a>, RenderError> {
    let (sv, st) = state.view(blit.src)?;
    let (dv, dt) = state.view(blit.dst)?;
    let (src, dst) = (resolved(sv, st), resolved(dv, dt));
    validate::validate_blit(src, dst, blit)?;
    let from = rect_place(sv, st, &blit.src_rect)?;
    let to = rect_place(dv, dt, &blit.dst_rect)?;
    if overlaps(&from, &to) {
        return Err(RenderError::Invalid(
            "a blit's source and destination overlap",
        ));
    }
    let format = sv.desc.format;
    let unscaled =
        blit.src_rect.width == blit.dst_rect.width && blit.src_rect.height == blit.dst_rect.height;
    if unscaled && format == dv.desc.format && !format.is_compressed() {
        Ok(Op::Copy { from, to })
    } else {
        Ok(Op::Forget(vec![to]))
    }
}

fn clear_op<'a>(state: &State, clear: &ClearDesc) -> Result<Op<'a>, RenderError> {
    let (view, tex) = state.view(clear.target)?;
    validate::validate_clear(resolved(view, tex), clear)?;
    let target = resolved(view, tex);
    let extent = target.base_extent()?;
    let shape = level_shape(&tex.desc, view.desc.base_mip)?;
    let rect = clear.rect.unwrap_or(Rect {
        x: 0,
        y: 0,
        width: extent.width,
        height: extent.height,
    });
    let (z, depth) = layers(view, tex, shape.deep);
    let to = Place::Texture {
        id: view.texture,
        mip: view.desc.base_mip,
        area: rect_blocks(view.desc.format, &rect, z, depth)?,
    };
    Ok(match texel::encode_clear(view.desc.format, clear.value) {
        Some(texel) => Op::Fill { to, texel },
        None => Op::Forget(vec![to]),
    })
}

fn draw_op<'a>(
    state: &State,
    caps: &Caps,
    list: &ListState,
    draw: &DrawDesc,
) -> Result<Op<'a>, RenderError> {
    validate::validate_draw(caps, draw)?;
    pipeline(state, draw.pipeline, true)?;
    let mut forget = Vec::new();
    bindings(state, &draw.bindings, &mut forget)?;
    for slice in &draw.vertex_buffers {
        buffer_slice(state, slice, BufferUsage::VERTEX, "vertex buffer", "vertex")?;
    }
    if let Some(index) = &draw.index_buffer {
        buffer_slice(
            state,
            &index.slice,
            BufferUsage::INDEX,
            "index buffer",
            "index",
        )?;
    }
    if let DrawParams::Indirect { buffer, .. } = draw.params {
        indirect(state, buffer)?;
    }
    for id in list
        .targets
        .colors
        .iter()
        .flatten()
        .chain(list.targets.depth_stencil.iter())
    {
        let (view, tex) = state.view(*id)?;
        forget.extend(whole_view(view, tex)?);
    }
    Ok(Op::Forget(forget))
}

fn len_u64(len: usize) -> u64 {
    u64::try_from(len).unwrap_or(u64::MAX)
}

fn resolved<'s>(view: &'s ViewRec, tex: &'s TextureRec) -> ResolvedView<'s> {
    ResolvedView {
        texture: &tex.desc,
        view: &view.desc,
    }
}

/// The description a copy region names, and the place it covers.
fn resolve<'s>(
    state: &'s State,
    region: &'s CopyRegion,
) -> Result<(Resolved<'s>, Place), RenderError> {
    match region {
        CopyRegion::Buffer(slice) => {
            let rec = state.buffer(slice.buffer)?;
            Ok((
                Resolved::Buffer {
                    desc: &rec.desc,
                    slice,
                },
                Place::Buffer {
                    id: slice.buffer,
                    offset: slice.offset,
                    size: slice.size,
                },
            ))
        }
        CopyRegion::Texture(region) => {
            let rec = state.texture(region.texture)?;
            Ok((
                Resolved::Texture {
                    desc: &rec.desc,
                    region,
                },
                region_place(rec.desc.format, region)?,
            ))
        }
    }
}

/// The blocks a texture region covers. The validation has checked that it starts on a block
/// boundary and covers whole blocks or reaches the level's edge.
fn region_place(format: PixelFormat, region: &TextureRegion) -> Result<Place, RenderError> {
    let rect = Rect {
        x: region.origin.x,
        y: region.origin.y,
        width: region.extent.width,
        height: region.extent.height,
    };
    Ok(Place::Texture {
        id: region.texture,
        mip: region.mip,
        area: rect_blocks(
            format,
            &rect,
            region.origin.z,
            region.extent.depth_or_layers,
        )?,
    })
}

/// The smallest box of whole blocks covering `rect`, over `depth` slices or layers from `z`.
fn rect_blocks(
    format: PixelFormat,
    rect: &Rect,
    z: u32,
    depth: u32,
) -> Result<BlockBox, RenderError> {
    let (bw, bh) = (format.block_width(), format.block_height());
    let span = |start: u32, len: u32, block: u32| -> Result<(u32, u32), RenderError> {
        let first = start / block;
        let end = (u64::from(start) + u64::from(len)).div_ceil(u64::from(block));
        let end = u32::try_from(end).map_err(|_| RenderError::Overflow {
            what: "rectangle end",
        })?;
        Ok((first, end - first))
    };
    let (x, width) = span(rect.x, rect.width, bw)?;
    let (y, height) = span(rect.y, rect.height, bh)?;
    Ok(BlockBox {
        x,
        y,
        z,
        width,
        height,
        depth,
    })
}

/// The first layer and layer count a view covers in its texture's mip level of `deep` slices:
/// every slice of a 3D level, the view's layers otherwise.
fn layers(view: &ViewRec, tex: &TextureRec, deep: u32) -> (u32, u32) {
    if tex.desc.dimension == TextureDimension::D3 {
        (0, deep)
    } else {
        (view.desc.base_layer, view.desc.layer_count)
    }
}

/// `rect` in a view's base mip level at its base layer (slice 0 of a 3D texture).
fn rect_place(view: &ViewRec, tex: &TextureRec, rect: &Rect) -> Result<Place, RenderError> {
    let z = if tex.desc.dimension == TextureDimension::D3 {
        0
    } else {
        view.desc.base_layer
    };
    Ok(Place::Texture {
        id: view.texture,
        mip: view.desc.base_mip,
        area: rect_blocks(view.desc.format, rect, z, 1)?,
    })
}

/// Every mip level and layer a view covers.
fn whole_view(view: &ViewRec, tex: &TextureRec) -> Result<Vec<Place>, RenderError> {
    let first = view.desc.base_mip;
    let end = first
        .checked_add(view.desc.mip_count)
        .ok_or(RenderError::Overflow {
            what: "view mip range end",
        })?;
    (first..end)
        .map(|mip| {
            let shape = level_shape(&tex.desc, mip)?;
            let (z, depth) = layers(view, tex, shape.deep);
            Ok(Place::Texture {
                id: view.texture,
                mip,
                area: BlockBox {
                    x: 0,
                    y: 0,
                    z,
                    width: shape.wide,
                    height: shape.high,
                    depth,
                },
            })
        })
        .collect()
}

/// Whether two places share a byte.
fn overlaps(a: &Place, b: &Place) -> bool {
    let apart = |s1: u64, l1: u64, s2: u64, l2: u64| {
        s1.saturating_add(l1) <= s2 || s2.saturating_add(l2) <= s1
    };
    match (a, b) {
        (
            Place::Buffer {
                id: i1,
                offset: o1,
                size: s1,
            },
            Place::Buffer {
                id: i2,
                offset: o2,
                size: s2,
            },
        ) => i1 == i2 && !apart(*o1, *s1, *o2, *s2),
        (
            Place::Texture {
                id: i1,
                mip: m1,
                area: a1,
            },
            Place::Texture {
                id: i2,
                mip: m2,
                area: a2,
            },
        ) => {
            let axis = |s1: u32, l1: u32, s2: u32, l2: u32| {
                !apart(s1.into(), l1.into(), s2.into(), l2.into())
            };
            i1 == i2
                && m1 == m2
                && axis(a1.x, a1.width, a2.x, a2.width)
                && axis(a1.y, a1.height, a2.y, a2.height)
                && axis(a1.z, a1.depth, a2.z, a2.depth)
        }
        _ => false,
    }
}

/// A view bound as a render target: colour for a colour target, depth or stencil otherwise, and
/// from a texture with the render-target usage.
fn render_target(state: &State, id: ViewId, color: bool) -> Result<(), RenderError> {
    let (view, tex) = state.view(id)?;
    if !tex.desc.usage.contains(TextureUsage::RENDER_TARGET) {
        return Err(RenderError::MissingUsage {
            resource: "render target texture",
            usage: "render target",
        });
    }
    let is_color = view.desc.format.aspect() == FormatAspect::Color;
    match (color, is_color) {
        (true, false) => Err(RenderError::Invalid(
            "a colour target view must store colour",
        )),
        (false, true) => Err(RenderError::Invalid(
            "a depth/stencil target view must store depth or stencil",
        )),
        _ => Ok(()),
    }
}

/// A live pipeline of the kind a draw (`graphics`) or dispatch needs.
fn pipeline(state: &State, id: PipelineId, graphics: bool) -> Result<(), RenderError> {
    let rec = state
        .pipelines
        .get(id.index(), id.generation())
        .ok_or(gone(NO_PIPELINE))?;
    match (&rec.kind, graphics) {
        (PipelineKind::Graphics, true) | (PipelineKind::Compute, false) => Ok(()),
        (PipelineKind::Compute, true) => {
            Err(RenderError::Invalid("a draw needs a graphics pipeline"))
        }
        (PipelineKind::Graphics, false) => {
            Err(RenderError::Invalid("a dispatch needs a compute pipeline"))
        }
    }
}

/// A live buffer range inside its buffer, from a buffer with `needed`.
fn buffer_slice(
    state: &State,
    slice: &BufferSlice,
    needed: BufferUsage,
    resource: &'static str,
    usage: &'static str,
) -> Result<(), RenderError> {
    let rec = state.buffer(slice.buffer)?;
    validate::validate_buffer_slice(&rec.desc, slice)?;
    if rec.desc.usage.intersects(needed) {
        Ok(())
    } else {
        Err(RenderError::MissingUsage { resource, usage })
    }
}

fn indirect(state: &State, buffer: BufferId) -> Result<(), RenderError> {
    let rec = state.buffer(buffer)?;
    if rec.desc.usage.contains(BufferUsage::INDIRECT) {
        Ok(())
    } else {
        Err(RenderError::MissingUsage {
            resource: "indirect buffer",
            usage: "indirect",
        })
    }
}

/// Checks a draw's or dispatch's bindings and adds what a shader may write to `forget`.
fn bindings(
    state: &State,
    bindings: &[Binding],
    forget: &mut Vec<Place>,
) -> Result<(), RenderError> {
    for b in bindings {
        match &b.resource {
            BindingResource::Buffer(slice) => {
                buffer_slice(
                    state,
                    slice,
                    BufferUsage::UNIFORM | BufferUsage::STORAGE,
                    "bound buffer",
                    "uniform or storage",
                )?;
                if state
                    .buffer(slice.buffer)?
                    .desc
                    .usage
                    .contains(BufferUsage::STORAGE)
                {
                    forget.push(Place::Buffer {
                        id: slice.buffer,
                        offset: slice.offset,
                        size: slice.size,
                    });
                }
            }
            BindingResource::View(id) => {
                let (view, tex) = state.view(*id)?;
                let usage = tex.desc.usage;
                if !usage.intersects(TextureUsage::SAMPLED | TextureUsage::STORAGE) {
                    return Err(RenderError::MissingUsage {
                        resource: "bound texture",
                        usage: "sampled or storage",
                    });
                }
                if usage.contains(TextureUsage::STORAGE) {
                    forget.extend(whole_view(view, tex)?);
                }
            }
            BindingResource::Sampler(id) => {
                state
                    .samplers
                    .get(id.index(), id.generation())
                    .ok_or(gone(NO_SAMPLER))?;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{BlockBox, Place, overlaps, rect_blocks};
    use crate::command::Rect;
    use crate::format::PixelFormat;
    use crate::id::{BufferId, TextureId};

    fn buf(id: u32, offset: u64, size: u64) -> Place {
        Place::Buffer {
            id: BufferId::new(id, 0),
            offset,
            size,
        }
    }

    fn tex(mip: u32, x: u32, z: u32) -> Place {
        Place::Texture {
            id: TextureId::new(0, 0),
            mip,
            area: BlockBox {
                x,
                y: 0,
                z,
                width: 4,
                height: 4,
                depth: 1,
            },
        }
    }

    #[test]
    fn buffer_ranges_touching_at_the_end_do_not_overlap() {
        assert!(!overlaps(&buf(0, 0, 8), &buf(0, 8, 8)));
        assert!(!overlaps(&buf(0, 8, 8), &buf(0, 0, 8)));
        assert!(overlaps(&buf(0, 0, 9), &buf(0, 8, 8)));
        assert!(overlaps(&buf(0, 8, 8), &buf(0, 0, 9)));
        assert!(!overlaps(&buf(0, 0, 8), &buf(1, 0, 8)), "other buffer");
    }

    #[test]
    fn texture_boxes_overlap_only_in_one_level_on_every_axis() {
        assert!(overlaps(&tex(0, 0, 0), &tex(0, 3, 0)));
        assert!(!overlaps(&tex(0, 0, 0), &tex(0, 4, 0)), "touching in x");
        assert!(!overlaps(&tex(0, 0, 0), &tex(0, 0, 1)), "other layer");
        assert!(!overlaps(&tex(0, 0, 0), &tex(1, 0, 0)), "other mip");
        assert!(!overlaps(&tex(0, 0, 0), &buf(0, 0, 8)), "other kind");
        let other = Place::Texture {
            id: TextureId::new(1, 0),
            mip: 0,
            area: BlockBox {
                x: 0,
                y: 0,
                z: 0,
                width: 4,
                height: 4,
                depth: 1,
            },
        };
        assert!(!overlaps(&tex(0, 0, 0), &other), "other texture");
        let below = Place::Texture {
            id: TextureId::new(0, 0),
            mip: 0,
            area: BlockBox {
                x: 0,
                y: 4,
                z: 0,
                width: 4,
                height: 4,
                depth: 1,
            },
        };
        assert!(!overlaps(&tex(0, 0, 0), &below), "touching in y");
    }

    #[test]
    fn rect_blocks_cover_partial_blocks() {
        // 4x4 blocks: texels 2..7 across span blocks 0 and 1.
        let r = Rect {
            x: 2,
            y: 4,
            width: 5,
            height: 4,
        };
        assert_eq!(
            rect_blocks(PixelFormat::Bc1RgbaUnorm, &r, 3, 2),
            Ok(BlockBox {
                x: 0,
                y: 1,
                z: 3,
                width: 2,
                height: 1,
                depth: 2
            })
        );
        assert_eq!(
            rect_blocks(PixelFormat::R8Unorm, &r, 0, 1),
            Ok(BlockBox {
                x: 2,
                y: 4,
                z: 0,
                width: 5,
                height: 4,
                depth: 1
            })
        );
    }
}
