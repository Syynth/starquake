//! Encoding draws and dispatches on the render thread (`crate::draw`): the render pass a list
//! keeps open across its draws, each draw's bind groups, and the index rewrite.
//!
//! Everything that can fail for a draw is looked up first: its pipeline under the
//! wait-while-compiling setting, its host objects, the samplers its built layout made
//! non-filtering, its bind groups (made inside wgpu error scopes, so one wgpu refuses is caught
//! here rather than breaking the list's encoder) and the rewrite's pipeline. A draw that fails
//! any of them is skipped, and counted where `crate::draw` says; only then is the open pass
//! closed for a rewrite, a pass opened, and the draw encoded.

use super::Render;
use crate::draw::rewrite::{IndexRewrite, Run};
use crate::draw::{
    Attachment, Bound, BoundGroup, DispatchCall, DispatchStep, DrawCall, DrawStep, RewriteScratch,
    Targets,
};
use crate::pipeline::{self, DrawPipeline, ReadyPipeline, scoped};
use std::sync::atomic::Ordering;
use wgpu::util::DeviceExt;

/// What a list's encoding keeps for its draws: the open render pass, the targets the next pass
/// opens on, and the list's rewrite buffers.
pub(super) struct Recorder {
    pass: Option<wgpu::RenderPass<'static>>,
    targets: Option<Targets>,
    /// The parameter blocks and the output, when the list rewrites indices.
    rewrite: Option<(wgpu::Buffer, wgpu::Buffer)>,
}

impl Recorder {
    /// The state for a list whose rewrites need `scratch`.
    pub(super) fn new(device: &wgpu::Device, scratch: &RewriteScratch) -> Self {
        let rewrite = (scratch.output > 0).then(|| {
            let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("starquake index rewrite parameters"),
                contents: &scratch.params,
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let output = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("starquake index rewrite output"),
                size: scratch.output,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDEX,
                mapped_at_creation: false,
            });
            (params, output)
        });
        Self {
            pass: None,
            targets: None,
            rewrite,
        }
    }

    /// Ends the open render pass, if any, so the encoder takes other commands.
    pub(super) fn close(&mut self) {
        self.pass = None;
    }

    /// Ends the open pass and draws later draws into `targets`.
    pub(super) fn set_targets(&mut self, targets: Targets) {
        self.close();
        self.targets = Some(targets);
    }
}

/// The conservative load and store actions every attachment gets (render-backend, "Commands").
const fn load_store<V>() -> wgpu::Operations<V> {
    wgpu::Operations {
        load: wgpu::LoadOp::Load,
        store: wgpu::StoreOp::Store,
    }
}

/// One bind group, made, at its group number.
type Made = Vec<(u32, wgpu::BindGroup)>;

/// An index buffer as a draw binds it: the buffer, the range's offset and size, the format.
type IndexBinding = (wgpu::Buffer, u64, u64, wgpu::IndexFormat);

/// A draw the render thread skips, already counted where its cause is.
#[derive(Clone, Copy, Debug)]
struct Skipped;

impl Render {
    /// The pipeline a draw or dispatch gets under the setting (`crate::pipeline`, "Draws: wait
    /// or skip"), or `None` when it is skipped (and counted there).
    fn ready(&self, cell: &pipeline::PipelineCell) -> Option<ReadyPipeline> {
        match pipeline::acquire(cell, self.shared.setting, &self.shared.counters) {
            DrawPipeline::Ready(ready) => Some(ready),
            DrawPipeline::Skipped(_) => None,
        }
    }

    /// Makes `groups` for `pipeline`, or `None` when the draw is to be skipped: a filtering
    /// sampler at a binding the build made non-filtering (counted in `encode_skips`), a missing
    /// host object, or a bind group wgpu refuses (each a host error).
    fn bind_groups(&self, pipeline: &ReadyPipeline, groups: &[BoundGroup]) -> Option<Made> {
        let non_filtering = &pipeline.layout().non_filtering;
        let refused = groups.iter().any(|g| {
            g.entries.iter().any(|&(binding, bound)| {
                matches!(bound, Bound::Sampler(_, true))
                    && non_filtering.contains(&(g.group, binding))
            })
        });
        if refused {
            self.shared.encode_skips.fetch_add(1, Ordering::Relaxed);
            return None;
        }
        let mut made = Vec::with_capacity(groups.len());
        for g in groups {
            let Some(layout) = pipeline.bind_group_layout(g.group) else {
                self.shared
                    .host_error(&"a checked bind group has no layout in its pipeline");
                return None;
            };
            let mut entries = Vec::with_capacity(g.entries.len());
            for &(binding, bound) in &g.entries {
                let resource = match bound {
                    Bound::Buffer(slice) => {
                        let buffer = self.buffer(slice.buffer)?;
                        wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer,
                            offset: slice.offset,
                            size: wgpu::BufferSize::new(slice.size),
                        })
                    }
                    Bound::View(id) => {
                        let Some(view) = self.views.get(id.index(), id.generation()) else {
                            self.shared.host_error(&"a checked view had no host object");
                            return None;
                        };
                        wgpu::BindingResource::TextureView(view)
                    }
                    Bound::Sampler(id, _) => {
                        let Some(sampler) = self.samplers.get(id.index(), id.generation()) else {
                            self.shared
                                .host_error(&"a checked sampler had no host object");
                            return None;
                        };
                        wgpu::BindingResource::Sampler(sampler)
                    }
                };
                entries.push(wgpu::BindGroupEntry { binding, resource });
            }
            let group = scoped(&self.device, || {
                self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout,
                    entries: &entries,
                })
            });
            match group {
                Ok(group) => made.push((g.group, group)),
                Err(error) => {
                    self.shared.host_error(&error);
                    return None;
                }
            }
        }
        Some(made)
    }

    /// The rewrite's pipeline, made on the first rewrite.
    fn index_rewrite(&mut self) -> Option<&IndexRewrite> {
        if self.rewrite.is_none() {
            match IndexRewrite::new(&self.device) {
                Ok(made) => self.rewrite = Some(made),
                Err(error) => {
                    self.shared.host_error(&error);
                    return None;
                }
            }
        }
        self.rewrite.as_ref()
    }

    /// Opens a render pass on `targets` (`crate::draw`, "Render passes"), or `None` (a host
    /// error) when an attachment's texture has no host object.
    fn open_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        targets: &Targets,
    ) -> Option<wgpu::RenderPass<'static>> {
        let view = |a: &Attachment| {
            let texture = self.texture(a.texture)?;
            let (dimension, layers) = if a.three_d {
                (wgpu::TextureViewDimension::D3, 0)
            } else {
                (wgpu::TextureViewDimension::D2, a.layer)
            };
            Some(texture.create_view(&wgpu::TextureViewDescriptor {
                label: None,
                format: Some(a.format),
                dimension: Some(dimension),
                usage: Some(wgpu::TextureUsages::RENDER_ATTACHMENT),
                aspect: wgpu::TextureAspect::All,
                base_mip_level: a.mip,
                mip_level_count: Some(1),
                base_array_layer: layers,
                array_layer_count: Some(1),
            }))
        };
        let colors: Vec<Option<(Attachment, wgpu::TextureView)>> = targets
            .colors
            .iter()
            .map(|a| {
                a.map(|a| view(&a).map(|v| (a, v)))
                    .map_or(Some(None), |v| v.map(Some))
            })
            .collect::<Option<_>>()?;
        let depth = match &targets.depth_stencil {
            Some(a) => Some((*a, view(a)?)),
            None => None,
        };
        let color_attachments: Vec<Option<wgpu::RenderPassColorAttachment<'_>>> = colors
            .iter()
            .map(|c| {
                c.as_ref().map(|(a, v)| wgpu::RenderPassColorAttachment {
                    view: v,
                    depth_slice: a.three_d.then_some(0),
                    resolve_target: None,
                    ops: load_store(),
                })
            })
            .collect();
        let depth_stencil_attachment =
            depth
                .as_ref()
                .map(|(a, v)| wgpu::RenderPassDepthStencilAttachment {
                    view: v,
                    depth_ops: a.format.has_depth_aspect().then(load_store),
                    stencil_ops: a.format.has_stencil_aspect().then(load_store),
                });
        let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("starquake draws"),
            color_attachments: &color_attachments,
            depth_stencil_attachment,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        Some(pass.forget_lifetime())
    }

    /// Encodes a draw (module docs): every lookup first, then the rewrite, the pass and the
    /// draw.
    pub(super) fn draw(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        rec: &mut Recorder,
        step: DrawStep,
    ) {
        let Some(ready) = self.ready(&step.cell) else {
            return;
        };
        let Some(render) = ready.render().cloned() else {
            self.shared
                .host_error(&"a checked draw's pipeline is not a graphics pipeline");
            return;
        };
        let Some(groups) = self.bind_groups(&ready, &step.groups) else {
            return;
        };
        let Some(vertex) = step
            .vertex_buffers
            .iter()
            .map(|s| self.buffer(s.buffer).cloned().map(|b| (b, *s)))
            .collect::<Option<Vec<_>>>()
        else {
            return;
        };
        let Ok(index) = self.draw_index(encoder, rec, &step) else {
            return;
        };
        if step.fresh_pass {
            rec.close();
        }
        if rec.pass.is_none() {
            let Some(targets) = rec.targets.clone() else {
                self.shared.host_error(&"a checked draw has no targets");
                return;
            };
            rec.pass = self.open_pass(encoder, &targets);
        }
        let indirect = match &step.call {
            DrawCall::Indirect { buffer, .. } => match self.buffer(*buffer) {
                Some(b) => Some(b.clone()),
                None => return,
            },
            DrawCall::Vertices { .. } | DrawCall::Indices { .. } => None,
        };
        let Some(pass) = rec.pass.as_mut() else {
            return;
        };
        pass.set_pipeline(&render);
        for (group, bind) in &groups {
            pass.set_bind_group(*group, bind, &[]);
        }
        for (slot, (buffer, slice)) in (0_u32..).zip(&vertex) {
            let end = slice.offset.saturating_add(slice.size);
            pass.set_vertex_buffer(slot, buffer.slice(slice.offset..end));
        }
        if let Some((buffer, offset, size, format)) = &index {
            pass.set_index_buffer(buffer.slice(*offset..offset.saturating_add(*size)), *format);
        }
        let v = step.viewport;
        pass.set_viewport(v.x, v.y, v.width, v.height, v.min_depth, v.max_depth);
        let s = step.scissor;
        pass.set_scissor_rect(s.x, s.y, s.width, s.height);
        pass.set_blend_constant(step.blend_constant);
        pass.set_stencil_reference(step.stencil_reference);
        match step.call {
            DrawCall::Vertices {
                vertices,
                instances,
            } => pass.draw(vertices, instances),
            DrawCall::Indices {
                indices,
                base_vertex,
                instances,
            } => pass.draw_indexed(indices, base_vertex, instances),
            DrawCall::Indirect {
                offset,
                stride,
                count,
                indexed,
                ..
            } => {
                let Some(args) = indirect else {
                    return;
                };
                for i in 0..u64::from(count) {
                    let at = offset.saturating_add(i.saturating_mul(stride));
                    if indexed {
                        pass.draw_indexed_indirect(&args, at);
                    } else {
                        pass.draw_indirect(&args, at);
                    }
                }
            }
        }
    }

    /// The index buffer a draw binds: the guest's, or the rewrite's output after the rewrite is
    /// encoded (closing the open pass); `Ok(None)` for a draw without indices, `Err` when the
    /// draw is to be skipped (already counted).
    fn draw_index(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        rec: &mut Recorder,
        step: &DrawStep,
    ) -> Result<Option<IndexBinding>, Skipped> {
        let Some(rw) = step.rewrite else {
            return match step.index {
                Some((slice, format)) => {
                    let buffer = self.buffer(slice.buffer).ok_or(Skipped)?.clone();
                    Ok(Some((buffer, slice.offset, slice.size, format)))
                }
                None => Ok(None),
            };
        };
        let Some((params, output)) = rec.rewrite.clone() else {
            self.shared
                .host_error(&"a checked rewrite has no scratch buffers");
            return Err(Skipped);
        };
        let input = match rw.input {
            Some((id, offset, size)) => {
                Some((self.buffer(id).ok_or(Skipped)?.clone(), offset, size))
            }
            None => None,
        };
        let max_groups = self.limits.max_workgroups;
        let device = self.device.clone();
        let rewriter = self.index_rewrite().ok_or(Skipped)?;
        rec.close();
        rewriter.encode(
            &device,
            encoder,
            Run {
                params: (&params, rw.params_offset),
                input: input.as_ref().map(|(b, o, s)| (b, *o, *s)),
                output: (&output, rw.output_offset, rw.count),
                max_groups,
            },
        );
        let size = u64::from(rw.count).saturating_mul(4);
        Ok(Some((
            output,
            rw.output_offset,
            size,
            wgpu::IndexFormat::Uint32,
        )))
    }

    /// Encodes a dispatch in a compute pass of its own.
    pub(super) fn dispatch(&mut self, encoder: &mut wgpu::CommandEncoder, step: &DispatchStep) {
        let Some(ready) = self.ready(&step.cell) else {
            return;
        };
        let Some(compute) = ready.compute().cloned() else {
            self.shared
                .host_error(&"a checked dispatch's pipeline is not a compute pipeline");
            return;
        };
        let Some(groups) = self.bind_groups(&ready, &step.groups) else {
            return;
        };
        let indirect = match step.call {
            DispatchCall::Indirect(id, offset) => match self.buffer(id) {
                Some(b) => Some((b.clone(), offset)),
                None => return,
            },
            DispatchCall::Direct(_) => None,
        };
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("starquake dispatch"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&compute);
        for (group, bind) in &groups {
            pass.set_bind_group(*group, bind, &[]);
        }
        match (step.call, indirect) {
            (DispatchCall::Direct([x, y, z]), _) => pass.dispatch_workgroups(x, y, z),
            (DispatchCall::Indirect(..), Some((buffer, offset))) => {
                pass.dispatch_workgroups_indirect(&buffer, offset);
            }
            (DispatchCall::Indirect(..), None) => {}
        }
    }
}
