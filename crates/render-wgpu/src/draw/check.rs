//! The check pass's part for render targets, draws and dispatches (`crate::draw`, "The checks";
//! `crate::plan` calls it in list order).
//!
//! A draw's bindings are checked against its pipeline's layout in this order: every bound group
//! and binding is one the layout has ([`EXTRA_BINDING`], in the draw's order); then each layout
//! entry, in group and binding order, is not a binding array ([`BINDING_ARRAY`]), is bound
//! ([`MISSING_BINDING`]) by a resource of its kind ([`BINDING_KIND`]), and fits it: a buffer has
//! the uniform or storage usage and keeps [`rules::check_buffer_binding`]; a view has the
//! sampled or storage usage, is not of both depth and stencil ([`DEPTH_STENCIL_VIEW`]) and has
//! the binding's dimension, samples, sample type, format and (for storage) one mip level
//! ([`VIEW_KIND`]); a sampler compares exactly when the binding does ([`SAMPLER_KIND`]).

use super::rules::{self, BufferKind, Subresources, Uses, VertexStep};
use super::rewrite::{self, Rewrite, Source};
use super::{
    Attachment, BINDING_ARRAY, BINDING_KIND, Bound, BoundGroup, DEPTH_STENCIL_VIEW, DispatchCall,
    DispatchStep, DrawCall, DrawLimits, DrawStep, EXTRA_BINDING, EntryInfo, GraphicsInfo,
    INDEX_LIMIT, INDEX_OFFSET, INDIRECT_DRAWS, INDIRECT_REWRITE, LayoutInfo, MAX_INDIRECT_DRAWS,
    MISSING_BINDING, MISSING_VERTEX_BUFFER, NEEDS_COMPUTE, NEEDS_GRAPHICS, NO_TARGETS,
    PipelineInfo, REWRITE_SIZE, RewriteScratch, RewriteStep, SAMPLER_KIND, SCRATCH,
    TARGET_MISMATCH, TARGETS_DIFFER, Targets, USAGE_CONFLICT, VERTEX_LIMIT, VERTEX_OFFSET,
    VIEW_KIND,
};
use crate::convert::view_dimension;
use crate::format::texture_format;
use crate::pipeline::layout::LayoutPlan;
use crate::pipeline::state::GraphicsPlan;
use crate::pipeline::{PipelineEntry, Registry, ShaderRec};
use crate::plan::{NO_SAMPLER, Tables, TextureRec, ViewRec};
use starquake_render::command::{
    Binding, BindingResource, BufferSlice, DispatchDesc, DispatchParams, DrawDesc, DrawParams,
    IndexFormat, RenderTargets,
};
use starquake_render::format::FormatAspect;
use starquake_render::pipeline::GraphicsPipelineDesc;
use starquake_render::resource::TextureDimension;
use starquake_render::{
    BufferId, BufferUsage, Caps, PipelineId, RenderError, TextureUsage, ViewId, validate,
};
use std::collections::BTreeMap;
use std::sync::Arc;

/// A refusal of `submit` for a rule this module names.
const fn invalid(why: &'static str) -> RenderError {
    RenderError::Invalid(why)
}

/// The draw state of one list's check.
#[derive(Debug, Default)]
pub(crate) struct ListDraws {
    /// The targets set by the list's last `SetRenderTargets`, if any.
    targets: Option<Targets>,
    /// The uses of the open render pass, or `None` when no pass is open.
    pass: Option<Uses>,
    /// The scratch the list's rewrites use so far.
    pub(crate) scratch: RewriteScratch,
}

impl ListDraws {
    /// A command other than a draw closes the open pass (`draw` module docs, "Render passes").
    pub(crate) fn close_pass(&mut self) {
        self.pass = None;
    }

    /// Resolves the targets a `SetRenderTargets` binds (which `crate::plan` has checked), and
    /// returns them when they differ from the current ones, closing the pass; `None` when they
    /// are the same, which keeps it.
    pub(crate) fn set_targets(
        &mut self,
        tables: &Tables,
        targets: &RenderTargets,
    ) -> Result<Option<Targets>, RenderError> {
        let attach = |id: ViewId| -> Result<Attachment, RenderError> {
            let (view, tex) = tables.view(id)?;
            attachment(view, tex)
        };
        let mut colors = targets
            .colors
            .iter()
            .map(|v| v.map(attach).transpose())
            .collect::<Result<Vec<_>, _>>()?;
        while colors.last().is_some_and(Option::is_none) {
            colors.pop();
        }
        let depth_stencil = targets.depth_stencil.map(attach).transpose()?;
        let resolved = Targets {
            colors,
            depth_stencil,
        };
        if self.targets.as_ref() == Some(&resolved) {
            return Ok(None);
        }
        self.close_pass();
        self.targets = Some(resolved.clone());
        Ok(Some(resolved))
    }

    /// Checks a draw (`draw` module docs, "The checks") and returns its step, or `None` for a
    /// draw that draws nothing because its rewrite writes no index.
    pub(crate) fn draw(
        &mut self,
        tables: &Tables,
        pipelines: &Registry,
        caps: &Caps,
        limits: &DrawLimits,
        draw: &DrawDesc,
    ) -> Result<Option<DrawStep>, RenderError> {
        validate::validate_draw(caps, draw)?;
        let entry = pipeline(pipelines, draw.pipeline)?;
        let PipelineInfo::Graphics(info) = &*entry.info else {
            return Err(invalid(NEEDS_GRAPHICS));
        };
        let mut uses = Uses::new();
        let resolved = resolve_bindings(tables, &draw.bindings)?;
        for slice in &draw.vertex_buffers {
            buffer_slice(tables, slice, BufferUsage::VERTEX, "vertex buffer", "vertex")?;
        }
        if let Some(index) = &draw.index_buffer {
            buffer_slice(tables, &index.slice, BufferUsage::INDEX, "index buffer", "index")?;
        }
        if let DrawParams::Indirect { buffer, .. } = draw.params {
            indirect_buffer(tables, buffer)?;
        }
        let targets = self.targets.as_ref().ok_or(invalid(NO_TARGETS))?;
        let size = target_size(targets, info)?;
        attachment_uses(targets, &mut uses);
        let groups = bind(&info.layout, &resolved, limits, &mut uses)?;
        if draw.vertex_buffers.len() < info.vertex.len() {
            return Err(invalid(MISSING_VERTEX_BUFFER));
        }
        for slice in &draw.vertex_buffers {
            if !slice.offset.is_multiple_of(wgpu::VERTEX_ALIGNMENT) {
                return Err(invalid(VERTEX_OFFSET));
            }
            uses.buffer(slice.buffer, wgpu::BufferUses::VERTEX);
        }
        let index_format = match draw.params {
            DrawParams::Indexed { .. } | DrawParams::Indirect { indexed: true, .. } => {
                draw.index_buffer.map(|i| i.format)
            }
            DrawParams::Direct { .. } | DrawParams::Indirect { indexed: false, .. } => None,
        };
        let rewrite = rewrite::plan(info.topology, info.primitive_restart, index_format);
        let index = match (rewrite, index_format, draw.index_buffer) {
            (None, Some(format), Some(binding)) => {
                let host = host_index_format(format);
                if !binding.slice.offset.is_multiple_of(index_bytes(format)) {
                    return Err(invalid(INDEX_OFFSET));
                }
                uses.buffer(binding.slice.buffer, wgpu::BufferUses::INDEX);
                host.map(|h| (binding.slice, h))
            }
            _ => None,
        };
        let viewport = draw.dynamic.viewports.first().copied().ok_or(RenderError::OutOfRange {
            what: "viewport count",
            value: 0,
            min: 1,
            max: 1,
        })?;
        rules::check_viewport(&viewport, limits.max_texture_dimension_2d)?;
        let scissor = rules::clamp_scissor(draw.dynamic.scissors.first(), size.0, size.1);
        let sizes: Vec<Option<u64>> = draw.vertex_buffers.iter().map(|s| Some(s.size)).collect();
        let bounds = rules::vertex_limits(&info.vertex, &sizes);
        let counted = self.counts(tables, draw, rewrite, bounds, limits, &mut uses)?;
        if !uses.is_valid() {
            return Err(invalid(USAGE_CONFLICT));
        }
        let Some((call, rewrite)) = counted else {
            return Ok(None);
        };
        let fresh_pass = match &mut self.pass {
            Some(pass) if rewrite.is_none() && pass.fits(&uses) => {
                pass.merge(&uses);
                false
            }
            _ => {
                self.pass = Some(uses);
                true
            }
        };
        let d = &draw.dynamic;
        Ok(Some(DrawStep {
            pipeline: draw.pipeline,
            cell: Arc::clone(&entry.cell),
            fresh_pass,
            groups,
            vertex_buffers: draw.vertex_buffers.clone(),
            index,
            rewrite,
            viewport,
            scissor,
            blend_constant: wgpu::Color {
                r: d.blend_constant[0].into(),
                g: d.blend_constant[1].into(),
                b: d.blend_constant[2].into(),
                a: d.blend_constant[3].into(),
            },
            stencil_reference: d.stencil_reference,
            call,
        }))
    }

    /// Step 8 (`draw` module docs, "The checks"): what the draw draws, with `rewrite` its
    /// planned rewrite and `(vertex_limit, instance_limit)` what its vertex buffers hold, and
    /// the rewrite placed in the list's scratch; `None` when the rewrite writes no index.
    fn counts(
        &mut self,
        tables: &Tables,
        draw: &DrawDesc,
        rewrite: Option<Rewrite>,
        (vertex_limit, instance_limit): (u64, u64),
        limits: &DrawLimits,
        uses: &mut Uses,
    ) -> Result<Option<(DrawCall, Option<RewriteStep>)>, RenderError> {
        Ok(Some(match draw.params {
            DrawParams::Indirect {
                buffer,
                offset,
                draw_count,
                stride,
                indexed,
            } => {
                if rewrite.is_some() {
                    return Err(RenderError::Unsupported(INDIRECT_REWRITE));
                }
                if draw_count > MAX_INDIRECT_DRAWS {
                    return Err(RenderError::Unsupported(INDIRECT_DRAWS));
                }
                let size = tables.buffer(buffer)?.size;
                let args = rules::indirect_draw_size(indexed);
                rules::check_indirect(offset, stride.into(), draw_count, args, size)?;
                uses.buffer(buffer, wgpu::BufferUses::INDIRECT);
                let call = DrawCall::Indirect {
                    buffer,
                    offset,
                    stride: stride.into(),
                    count: draw_count,
                    indexed,
                };
                (call, None)
            }
            DrawParams::Direct {
                vertex_count,
                instance_count,
                first_vertex,
                first_instance,
            } => {
                let vertices = span(first_vertex, vertex_count, vertex_limit, VERTEX_LIMIT)?;
                let instances = span(first_instance, instance_count, instance_limit, VERTEX_LIMIT)?;
                match rewrite {
                    None => (
                        DrawCall::Vertices {
                            vertices,
                            instances,
                        },
                        None,
                    ),
                    Some(rw) => {
                        let Some(placed) =
                            self.place(rw, None, first_vertex, vertex_count, limits)?
                        else {
                            return Ok(None);
                        };
                        let call = DrawCall::Indices {
                            indices: 0..placed.count,
                            base_vertex: 0,
                            instances,
                        };
                        (call, Some(placed))
                    }
                }
            }
            DrawParams::Indexed {
                index_count,
                instance_count,
                first_index,
                base_vertex,
                first_instance,
            } => {
                let instances = span(first_instance, instance_count, instance_limit, VERTEX_LIMIT)?;
                // `validate_draw` has made an indexed draw's index buffer present.
                let binding = draw.index_buffer.ok_or(invalid(INDEX_LIMIT))?;
                let bytes = index_bytes(binding.format);
                let limit = binding.slice.size / bytes;
                let indices = span(first_index, index_count, limit, INDEX_LIMIT)?;
                match rewrite {
                    None => (
                        DrawCall::Indices {
                            indices,
                            base_vertex,
                            instances,
                        },
                        None,
                    ),
                    Some(rw) => {
                        let start = u64::from(first_index)
                            .checked_mul(bytes)
                            .and_then(|o| o.checked_add(binding.slice.offset))
                            .ok_or(invalid(INDEX_LIMIT))?;
                        let input = (binding.slice.buffer, start);
                        let Some(placed) =
                            self.place(rw, Some(input), 0, index_count, limits)?
                        else {
                            return Ok(None);
                        };
                        let call = DrawCall::Indices {
                            indices: 0..placed.count,
                            base_vertex,
                            instances,
                        };
                        (call, Some(placed))
                    }
                }
            }
        }))
    }

    /// Places a rewrite of `n` inputs (from `input`, a buffer and the first index's byte
    /// offset, or the sequential source from vertex `first`) in the list's scratch, after its
    /// size checks; `None` when it writes no index.
    fn place(
        &mut self,
        rw: Rewrite,
        input: Option<(BufferId, u64)>,
        first: u32,
        n: u32,
        limits: &DrawLimits,
    ) -> Result<Option<RewriteStep>, RenderError> {
        let too_big = RenderError::Unsupported(REWRITE_SIZE);
        let count = rewrite::output_count(rw.mode, n).ok_or(too_big.clone())?;
        if count == 0 {
            return Ok(None);
        }
        let b = &limits.binding;
        let (binding, first) = match input {
            // The vertex span check has kept `first + n` within a `u32`, so the last vertex
            // number, `first + n - 1`, is below the restart value.
            None => (None, first),
            Some((buffer, start)) => {
                let alignment = u64::from(b.storage_alignment.max(1));
                let skip = start % alignment;
                let read = u64::from(n) * u64::from(rw.source.index_bytes());
                let span = skip
                    .checked_add(read)
                    .and_then(|s| s.checked_next_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT))
                    .ok_or(too_big.clone())?;
                if span > b.max_storage {
                    return Err(too_big);
                }
                let skip32 = u32::try_from(skip).map_err(|_| too_big.clone())?;
                (Some((buffer, start - skip, span)), skip32)
            }
        };
        let bytes = u64::from(count) * 4;
        if bytes > b.max_storage {
            return Err(too_big);
        }
        let scratch_full = RenderError::Unsupported(SCRATCH);
        let params_at = u64::try_from(self.scratch.params.len())
            .ok()
            .and_then(|l| l.checked_next_multiple_of(u64::from(b.uniform_alignment.max(1))))
            .ok_or(scratch_full.clone())?;
        let output_at = self
            .scratch
            .output
            .checked_next_multiple_of(u64::from(b.storage_alignment.max(1)))
            .ok_or(scratch_full.clone())?;
        let output_end = output_at.checked_add(bytes).ok_or(scratch_full.clone())?;
        let params_end = params_at
            .checked_add(rewrite::PARAMS_SIZE)
            .ok_or(scratch_full.clone())?;
        if output_end > limits.max_buffer_size || params_end > limits.max_buffer_size {
            return Err(scratch_full);
        }
        let params_len = usize::try_from(params_at).map_err(|_| scratch_full)?;
        self.scratch.params.resize(params_len, 0);
        self.scratch
            .params
            .extend_from_slice(&rewrite::params(rw, count, first, n));
        self.scratch.output = output_end;
        Ok(Some(RewriteStep {
            params_offset: params_at,
            input: binding,
            output_offset: output_at,
            count,
        }))
    }

    /// Checks a dispatch (`draw` module docs, "The checks") and returns its step. A dispatch
    /// is its own usage scope.
    pub(crate) fn dispatch(
        &mut self,
        tables: &Tables,
        pipelines: &Registry,
        caps: &Caps,
        limits: &DrawLimits,
        dispatch: &DispatchDesc,
    ) -> Result<DispatchStep, RenderError> {
        validate::validate_dispatch(caps, dispatch)?;
        let entry = pipeline(pipelines, dispatch.pipeline)?;
        let PipelineInfo::Compute(layout) = &*entry.info else {
            return Err(invalid(NEEDS_COMPUTE));
        };
        let resolved = resolve_bindings(tables, &dispatch.bindings)?;
        if let DispatchParams::Indirect { buffer, .. } = dispatch.params {
            indirect_buffer(tables, buffer)?;
        }
        let mut uses = Uses::new();
        let groups = bind(layout, &resolved, limits, &mut uses)?;
        let call = match dispatch.params {
            DispatchParams::Direct { x, y, z } => DispatchCall::Direct([x, y, z]),
            DispatchParams::Indirect { buffer, offset } => {
                let size = tables.buffer(buffer)?.size;
                rules::check_indirect(offset, 0, 1, rules::INDIRECT_DISPATCH_SIZE, size)?;
                uses.buffer(buffer, wgpu::BufferUses::INDIRECT);
                DispatchCall::Indirect(buffer, offset)
            }
        };
        if !uses.is_valid() {
            return Err(invalid(USAGE_CONFLICT));
        }
        self.close_pass();
        Ok(DispatchStep {
            pipeline: dispatch.pipeline,
            cell: Arc::clone(&entry.cell),
            groups,
            call,
        })
    }
}

/// `first..first + count`, when it stays within `limit` and a `u32`; else `why`.
fn span(
    first: u32,
    count: u32,
    limit: u64,
    why: &'static str,
) -> Result<std::ops::Range<u32>, RenderError> {
    match first.checked_add(count) {
        Some(end) if rules::within(first, count, limit) => Ok(first..end),
        _ => Err(invalid(why)),
    }
}

/// Bytes per index.
const fn index_bytes(format: IndexFormat) -> u64 {
    Source::of(Some(format)).index_bytes() as u64
}

/// wgpu's index format, or `None` for 8-bit indices (always rewritten).
const fn host_index_format(format: IndexFormat) -> Option<wgpu::IndexFormat> {
    match format {
        IndexFormat::U8 => None,
        IndexFormat::U16 => Some(wgpu::IndexFormat::Uint16),
        IndexFormat::U32 => Some(wgpu::IndexFormat::Uint32),
    }
}

/// The live pipeline `id` names.
fn pipeline(pipelines: &Registry, id: PipelineId) -> Result<&PipelineEntry, RenderError> {
    pipelines
        .pipelines
        .get(id.index(), id.generation())
        .ok_or(invalid(validate::NO_PIPELINE))
}

/// The attachment a target view draws into.
fn attachment(view: &ViewRec, tex: &TextureRec) -> Result<Attachment, RenderError> {
    let format = texture_format(view.desc.format)
        .ok_or(RenderError::Unsupported(crate::rules::VIEW_FORMAT))?;
    let mip = view.desc.base_mip;
    let extent = tex.desc.mip_extent(mip).ok_or(RenderError::NotBelow {
        what: "mip level",
        index: mip.into(),
        count: tex.desc.mip_levels.into(),
    })?;
    let three_d = tex.desc.dimension == TextureDimension::D3;
    Ok(Attachment {
        texture: view.texture,
        format,
        mip,
        layer: if three_d { 0 } else { view.desc.base_layer },
        three_d,
        size: (extent.width, extent.height),
        samples: tex.desc.sample_count,
    })
}

/// The attachments' one size, after checking that they have one size and one sample count and
/// the formats and samples `info` was made for (`draw` module docs, step 4).
fn target_size(targets: &Targets, info: &GraphicsInfo) -> Result<(u32, u32), RenderError> {
    let mut all = targets.colors.iter().flatten().chain(&targets.depth_stencil);
    let first = all.next().ok_or(invalid(NO_TARGETS))?;
    if all.any(|a| a.size != first.size || a.samples != first.samples) {
        return Err(RenderError::Unsupported(TARGETS_DIFFER));
    }
    let colors: Vec<Option<wgpu::TextureFormat>> = targets
        .colors
        .iter()
        .map(|a| a.map(|a| a.format))
        .collect();
    let depth = targets.depth_stencil.map(|a| a.format);
    if colors != info.colors || depth != info.depth_stencil || first.samples != info.samples {
        return Err(invalid(TARGET_MISMATCH));
    }
    Ok(first.size)
}

/// The attachments' own uses: each written.
fn attachment_uses(targets: &Targets, uses: &mut Uses) {
    let colors = targets
        .colors
        .iter()
        .flatten()
        .map(|a| (a, wgpu::TextureUses::COLOR_TARGET));
    let depth = targets
        .depth_stencil
        .iter()
        .map(|a| (a, wgpu::TextureUses::DEPTH_STENCIL_WRITE));
    for (a, usage) in colors.chain(depth) {
        let part = Subresources {
            mips: a.mip..a.mip.saturating_add(1),
            layers: a.layer..a.layer.saturating_add(1),
        };
        uses.texture(a.texture, &part, usage);
    }
}

/// A binding with its resource found live.
#[derive(Clone, Copy)]
enum Live<'a> {
    Buffer(BufferSlice, &'a starquake_render::BufferDesc),
    View(ViewId, &'a ViewRec, &'a TextureRec),
    Sampler(starquake_render::SamplerId, &'a starquake_render::SamplerDesc),
}

/// Each binding's resource, found live and with the usage a binding needs (the recording
/// backend's rule, step 3), by group and binding.
fn resolve_bindings<'t>(
    tables: &'t Tables,
    bindings: &[Binding],
) -> Result<BTreeMap<(u32, u32), Live<'t>>, RenderError> {
    let mut out = BTreeMap::new();
    for b in bindings {
        let live = match b.resource {
            BindingResource::Buffer(slice) => {
                let desc = buffer_slice(
                    tables,
                    &slice,
                    BufferUsage::UNIFORM.union(BufferUsage::STORAGE),
                    "bound buffer",
                    "uniform or storage",
                )?;
                Live::Buffer(slice, desc)
            }
            BindingResource::View(id) => {
                let (view, tex) = tables.view(id)?;
                if !tex
                    .desc
                    .usage
                    .intersects(TextureUsage::SAMPLED.union(TextureUsage::STORAGE))
                {
                    return Err(RenderError::MissingUsage {
                        resource: "bound texture",
                        usage: "sampled or storage",
                    });
                }
                Live::View(id, view, tex)
            }
            BindingResource::Sampler(id) => {
                let desc = tables
                    .samplers
                    .get(id.index(), id.generation())
                    .ok_or(invalid(NO_SAMPLER))?;
                Live::Sampler(id, desc)
            }
        };
        out.insert((b.group, b.binding), live);
    }
    Ok(out)
}

/// A live buffer range inside its buffer, from a buffer with one of `needed` (the recording
/// backend's rule).
fn buffer_slice<'t>(
    tables: &'t Tables,
    slice: &BufferSlice,
    needed: BufferUsage,
    resource: &'static str,
    usage: &'static str,
) -> Result<&'t starquake_render::BufferDesc, RenderError> {
    let desc = tables.buffer(slice.buffer)?;
    validate::validate_buffer_slice(desc, slice)?;
    if desc.usage.intersects(needed) {
        Ok(desc)
    } else {
        Err(RenderError::MissingUsage { resource, usage })
    }
}

/// A live buffer with the indirect usage.
fn indirect_buffer(tables: &Tables, buffer: BufferId) -> Result<(), RenderError> {
    if tables.buffer(buffer)?.usage.contains(BufferUsage::INDIRECT) {
        Ok(())
    } else {
        Err(RenderError::MissingUsage {
            resource: "indirect buffer",
            usage: "indirect",
        })
    }
}

/// Checks `resolved` against `layout` (module docs' order), adds their uses to `uses`, and
/// returns the bind groups.
fn bind(
    layout: &LayoutInfo,
    resolved: &BTreeMap<(u32, u32), Live<'_>>,
    limits: &DrawLimits,
    uses: &mut Uses,
) -> Result<Vec<BoundGroup>, RenderError> {
    let has = |group: u32, binding: u32| {
        usize::try_from(group)
            .ok()
            .and_then(|g| layout.groups.get(g))
            .is_some_and(|entries| entries.binary_search_by_key(&binding, |e| e.binding).is_ok())
    };
    if resolved.keys().any(|&(g, b)| !has(g, b)) {
        return Err(invalid(EXTRA_BINDING));
    }
    let mut groups = Vec::new();
    for (g, entries) in (0_u32..).zip(&layout.groups) {
        if entries.is_empty() {
            continue;
        }
        let mut bound = Vec::with_capacity(entries.len());
        for entry in entries {
            if entry.array {
                return Err(RenderError::Unsupported(BINDING_ARRAY));
            }
            let live = resolved
                .get(&(g, entry.binding))
                .ok_or(invalid(MISSING_BINDING))?;
            bound.push((entry.binding, fit(entry, *live, limits, uses)?));
        }
        groups.push(BoundGroup {
            group: g,
            entries: bound,
        });
    }
    Ok(groups)
}

/// Checks a buffer range bound as `ty`, whose shader declares `min_size` bytes, and adds its
/// use.
fn fit_buffer(
    ty: wgpu::BufferBindingType,
    min_size: u64,
    slice: BufferSlice,
    desc: &starquake_render::BufferDesc,
    limits: &DrawLimits,
    uses: &mut Uses,
) -> Result<Bound, RenderError> {
    let (kind, usage, needed, name) = match ty {
        wgpu::BufferBindingType::Uniform => (
            BufferKind::Uniform,
            wgpu::BufferUses::UNIFORM,
            BufferUsage::UNIFORM,
            "uniform",
        ),
        wgpu::BufferBindingType::Storage { read_only } => (
            BufferKind::Storage,
            if read_only {
                wgpu::BufferUses::STORAGE_READ_ONLY
            } else {
                wgpu::BufferUses::STORAGE_READ_WRITE
            },
            BufferUsage::STORAGE,
            "storage",
        ),
    };
    if !desc.usage.contains(needed) {
        return Err(RenderError::MissingUsage {
            resource: "bound resource",
            usage: name,
        });
    }
    rules::check_buffer_binding(kind, slice.offset, slice.size, min_size, &limits.binding)?;
    uses.buffer(slice.buffer, usage);
    Ok(Bound::Buffer(slice))
}

/// Checks one resource against its layout entry (module docs) and adds its use.
fn fit(
    entry: &EntryInfo,
    live: Live<'_>,
    limits: &DrawLimits,
    uses: &mut Uses,
) -> Result<Bound, RenderError> {
    let missing = |usage| RenderError::MissingUsage {
        resource: "bound resource",
        usage,
    };
    match (entry.ty, live) {
        (wgpu::BindingType::Buffer { ty, .. }, Live::Buffer(slice, desc)) => {
            fit_buffer(ty, entry.min_size, slice, desc, limits, uses)
        }
        (
            wgpu::BindingType::Texture {
                sample_type,
                view_dimension: dimension,
                multisampled,
            },
            Live::View(id, view, tex),
        ) => {
            if !tex.desc.usage.contains(TextureUsage::SAMPLED) {
                return Err(missing("sampled"));
            }
            if view.desc.format.aspect() == FormatAspect::DepthStencil {
                return Err(RenderError::Unsupported(DEPTH_STENCIL_VIEW));
            }
            let host = texture_format(view.desc.format);
            let fits = view_dimension(view.desc.dimension) == Some(dimension)
                && multisampled == (tex.desc.sample_count > 1)
                && host
                    .and_then(|f| {
                        f.sample_type(Some(wgpu::TextureAspect::All), Some(limits.features))
                    })
                    .is_some_and(|s| rules::sample_type_fits(sample_type, s));
            if !fits {
                return Err(invalid(VIEW_KIND));
            }
            uses.texture(view.texture, &subresources(view, tex), wgpu::TextureUses::RESOURCE);
            Ok(Bound::View(id))
        }
        (
            wgpu::BindingType::StorageTexture {
                access,
                format,
                view_dimension: dimension,
            },
            Live::View(id, view, tex),
        ) => {
            if !tex.desc.usage.contains(TextureUsage::STORAGE) {
                return Err(missing("storage"));
            }
            let fits = view_dimension(view.desc.dimension) == Some(dimension)
                && texture_format(view.desc.format) == Some(format)
                && view.desc.mip_count == 1;
            if !fits {
                return Err(invalid(VIEW_KIND));
            }
            let usage = match access {
                wgpu::StorageTextureAccess::ReadOnly => wgpu::TextureUses::STORAGE_READ_ONLY,
                wgpu::StorageTextureAccess::WriteOnly => wgpu::TextureUses::STORAGE_WRITE_ONLY,
                wgpu::StorageTextureAccess::ReadWrite => wgpu::TextureUses::STORAGE_READ_WRITE,
                wgpu::StorageTextureAccess::Atomic => wgpu::TextureUses::STORAGE_ATOMIC,
            };
            uses.texture(view.texture, &subresources(view, tex), usage);
            Ok(Bound::View(id))
        }
        (wgpu::BindingType::Sampler(ty), Live::Sampler(id, desc)) => {
            let comparison = ty == wgpu::SamplerBindingType::Comparison;
            if comparison != desc.compare.is_some() {
                return Err(invalid(SAMPLER_KIND));
            }
            let filters = rules::filtering(desc.mag_filter, desc.min_filter, desc.mip_filter);
            Ok(Bound::Sampler(id, filters))
        }
        _ => Err(invalid(BINDING_KIND)),
    }
}

/// The mip levels and layers a bound view covers (a 3D texture's are layer 0).
fn subresources(view: &ViewRec, tex: &TextureRec) -> Subresources {
    let d = &view.desc;
    let layers = if tex.desc.dimension == TextureDimension::D3 {
        0..1
    } else {
        d.base_layer..d.base_layer.saturating_add(d.layer_count)
    };
    Subresources {
        mips: d.base_mip..d.base_mip.saturating_add(d.mip_count),
        layers,
    }
}

/// The check's view of a pipeline's layout: `plan`'s entries, with each buffer's late size, the
/// largest of the stages' declarations of it (`pipeline::scan`).
pub(crate) fn layout_info(plan: &LayoutPlan, shaders: &[&ShaderRec]) -> LayoutInfo {
    let groups = (0_u32..)
        .zip(&plan.groups)
        .map(|(g, entries)| {
            entries
                .iter()
                .map(|e| EntryInfo {
                    binding: e.binding,
                    ty: e.ty,
                    array: e.count.is_some(),
                    min_size: shaders
                        .iter()
                        .filter_map(|s| s.buffer_size(g, e.binding))
                        .max()
                        .unwrap_or(0),
                })
                .collect()
        })
        .collect();
    LayoutInfo { groups }
}

/// What the check knows of a graphics pipeline, from its description and the plans made for
/// it.
pub(crate) fn graphics_info(
    desc: &GraphicsPipelineDesc,
    plan: &GraphicsPlan,
    layout: LayoutInfo,
) -> GraphicsInfo {
    GraphicsInfo {
        topology: desc.topology,
        primitive_restart: desc.primitive_restart,
        colors: plan
            .targets
            .iter()
            .map(|t| t.as_ref().map(|c| c.format))
            .collect(),
        depth_stencil: plan.depth_stencil.as_ref().map(|d| d.format),
        samples: plan.multisample.count,
        vertex: plan
            .buffers
            .iter()
            .map(|b| VertexStep {
                stride: b.stride,
                last_stride: b
                    .attributes
                    .iter()
                    .map(|a| a.offset.saturating_add(a.format.size()))
                    .max()
                    .unwrap_or(0),
                instance: b.step == wgpu::VertexStepMode::Instance,
            })
            .collect(),
        layout,
    }
}
