//! Clears wgpu cannot perform as clears, drawn with the render crate's clear shader
//! (docs/specs/render-backend.md, "Commands": "Clears carry write masks"; decided by the owner,
//! 2026-11-19, ruling 6 of "Owner rulings on the wave 44 and 45 questions").
//!
//! A [`ClearDesc`] carries write masks (a colour clear's channels, a stencil clear's bits) and
//! a rectangle. wgpu clears only through a render pass's load operation, which writes every
//! channel and every bit of a whole attachment. So the check pass decides how each clear is
//! done ([`clear_how`], a pure function), and the render thread does it:
//!
//! - [`ClearHow::Nothing`]: the clear writes nothing (an empty colour mask, or a value whose
//!   only aspect is stencil through an empty stencil mask);
//! - [`ClearHow::Pass`]: a render pass clears each whole layer by its load operation, with
//!   every channel and every stencil bit written (or the stencil kept, when the mask is empty);
//! - [`ClearHow::Scratch`]: an unmasked rectangle of a single-sampled colour view: a scratch
//!   texture cleared once, its rectangle copied into each layer (`crate::thread`);
//! - [`ClearHow::Draw`]: everything else: a partial colour mask, a partial stencil mask, or a
//!   rectangle over a depth, stencil or multisampled view, which wgpu's copies cannot reach
//!   (`crate::rules`, `WHOLE_COPY`). The clear shader ([`HostShader::Clear`]) is drawn into each
//!   layer as one triangle covering the level, with the rectangle (or the whole level) as the
//!   scissor, the value in its parameter block, and a pipeline of the view's format and sample
//!   count that writes the colour channels the mask names, or depth (test `ALWAYS`) when the
//!   value has depth and stencil (`ALWAYS`, `REPLACE`, the value as the reference) through the
//!   stencil mask when the value has stencil. The pixels are the same as a clear's.
//!
//! The draw follows the shader's interface in `starquake_render::host_shaders::clear`: the
//! fragment entry point by the colour's kind (float, signed or unsigned integer), `fs_depth`
//! when the format has depth, and no fragment stage for a stencil-only format (wgpu refuses a
//! depth write without a depth aspect; the module's `FRAGMENT_DEPTH` docs). The shader's value
//! patterns are written without conversion, so an integer clear writes its integers exactly.
//! The draw-based clear of the GPU crate's slice G24 did the same with its own pipelines; this
//! module replaces it (ported from the Eden shaders that `blit_image` draws with, through
//! `clear.wgsl`: `eden-nx2/src/video_core/renderer_vulkan/blit_image.cpp` @ 50fa86bb,
//! `ClearColor` and `ClearDepthStencil`, GPL-3.0-or-later).
//!
//! # Values and ordering
//!
//! A drawn clear's value reaches the shader through one 16-byte uniform buffer the render
//! thread writes with `Queue::write_buffer`, whose bytes land before the next submission's
//! command buffers. So the render thread submits what it has encoded before it writes the value
//! of a drawn clear, as it does before an upload (`crate::thread`, "Encoding a list"), and each
//! drawn clear sees its own value (`backend::tests::clears::two_drawn_clears_in_one_list_each_write_their_value`).
//!
//! # Host objects
//!
//! The shader module, its bind group layout and pipeline layout, the parameter buffer and its
//! bind group are made on the first drawn clear and kept; pipelines are made per
//! [`PipelineKey`] and kept up to [`MAX_CLEAR_PIPELINES`] (a **guess**: the keys a guest can
//! reach are bounded by the formats, sample counts, four colour bits and 256 stencil masks, but
//! that product is large). Past it the kept pipelines are dropped all at once and made again as
//! needed; wgpu keeps a pipeline alive while recorded commands use it, so dropping them is safe.
//! They are host memory the budget does not count, like the rectangle-clear scratch. A clear
//! shader that does not parse (the render crate's tests rule it out) leaves every drawn clear
//! undone, each counted as a host error.

use starquake_render::command::{ALL_STENCIL_BITS, ClearDesc, ClearValue, Rect};
use starquake_render::format::{FormatAspect, NumericType};
use starquake_render::host_shaders::clear as shader;
use starquake_render::pipeline::ColorWrites;
use starquake_render::{HostShader, PixelFormat};
use std::collections::HashMap;

use crate::plan::ClearOps;

/// The most clear pipelines the render thread keeps (module docs; a **guess**).
pub const MAX_CLEAR_PIPELINES: usize = 256;

/// Bytes of the clear shader's parameter block.
const PARAMS_LEN: usize = shader::PARAMS_SIZE as usize;

/// How the render thread performs one clear (module docs).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ClearHow {
    /// Nothing is written.
    Nothing,
    /// A render pass's load operation over each whole layer.
    Pass,
    /// A scratch texture's rectangle copied into each layer.
    Scratch,
    /// The clear shader drawn into each layer.
    Draw(DrawnClear),
}

/// The colour kind of a target, which picks the clear shader's fragment entry point.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ColorKind {
    /// Float or normalised: `fs_color_float`.
    Float,
    /// Signed integer: `fs_color_sint`.
    Sint,
    /// Unsigned integer: `fs_color_uint`.
    Uint,
}

impl ColorKind {
    /// The kind of `format`'s colour.
    pub fn of(format: PixelFormat) -> Self {
        match format.numeric() {
            NumericType::Sint => Self::Sint,
            NumericType::Uint => Self::Uint,
            _ => Self::Float,
        }
    }

    /// The fragment entry point that writes this kind.
    pub const fn entry_point(self) -> &'static str {
        match self {
            Self::Float => shader::FRAGMENT_COLOR_FLOAT,
            Self::Sint => shader::FRAGMENT_COLOR_SINT,
            Self::Uint => shader::FRAGMENT_COLOR_UINT,
        }
    }
}

/// What a drawn clear's pipeline writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DrawTarget {
    /// A colour attachment, through `mask`.
    Color {
        /// The fragment entry point's kind.
        kind: ColorKind,
        /// The channels written.
        mask: wgpu::ColorWrites,
    },
    /// A depth and stencil attachment.
    DepthStencil {
        /// Depth is written (the value has depth).
        depth: bool,
        /// Stencil is written through this mask (the value has stencil and the mask is not
        /// empty).
        stencil: Option<u8>,
    },
}

/// The key of a drawn clear's pipeline: everything the pipeline depends on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PipelineKey {
    /// The view's host format.
    pub format: wgpu::TextureFormat,
    /// The texture's samples per pixel.
    pub samples: u32,
    /// What it writes.
    pub target: DrawTarget,
}

/// A clear drawn with the clear shader.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DrawnClear {
    /// Its pipeline.
    pub key: PipelineKey,
    /// The parameter block: the value as four 32-bit patterns, little-endian
    /// (`host_shaders::clear::VALUE_OFFSET`).
    pub params: [u8; PARAMS_LEN],
    /// The stencil reference: the stencil value, or 0.
    pub stencil_reference: u32,
}

/// The four 32-bit patterns of `value`, as the clear shader takes them: the colour's channels,
/// or the depth first.
pub fn value_patterns(value: ClearValue) -> [u32; 4] {
    match value {
        ClearValue::Float(c) => c.map(f32::to_bits),
        ClearValue::Sint(c) => c.map(i32::cast_unsigned),
        ClearValue::Uint(c) => c,
        ClearValue::Depth(depth) | ClearValue::DepthStencil { depth, .. } => {
            [depth.to_bits(), 0, 0, 0]
        }
        ClearValue::Stencil(_) => [0; 4],
    }
}

/// `value`'s parameter block.
fn params(value: ClearValue) -> [u8; PARAMS_LEN] {
    let mut bytes = [0; PARAMS_LEN];
    let offset = shader::VALUE_OFFSET as usize;
    let words = value_patterns(value);
    if let Some(slot) = bytes.get_mut(offset..) {
        for (chunk, word) in slot.as_chunks_mut::<4>().0.iter_mut().zip(words) {
            chunk.copy_from_slice(&word.to_le_bytes());
        }
    }
    bytes
}

/// wgpu's colour write mask for `mask`.
pub fn host_color_writes(mask: ColorWrites) -> wgpu::ColorWrites {
    [
        (ColorWrites::RED, wgpu::ColorWrites::RED),
        (ColorWrites::GREEN, wgpu::ColorWrites::GREEN),
        (ColorWrites::BLUE, wgpu::ColorWrites::BLUE),
        (ColorWrites::ALPHA, wgpu::ColorWrites::ALPHA),
    ]
    .into_iter()
    .filter(|(bit, _)| mask.contains(*bit))
    .fold(wgpu::ColorWrites::empty(), |all, (_, host)| all | host)
}

/// How a clear of a view whose format is `format` (wgpu's `host`), of a texture of `samples`
/// samples, is done (module docs); `whole` when its rectangle covers the view's level. The
/// shared checks have passed, so the value suits the format.
pub fn clear_how(
    format: PixelFormat,
    host: wgpu::TextureFormat,
    samples: u32,
    whole: bool,
    clear: &ClearDesc,
) -> ClearHow {
    let draw = |target| {
        ClearHow::Draw(DrawnClear {
            key: PipelineKey {
                format: host,
                samples,
                target,
            },
            params: params(clear.value),
            stencil_reference: stencil_value(clear.value).unwrap_or(0),
        })
    };
    if format.aspect() == FormatAspect::Color {
        let mask = clear.color_mask;
        return if mask.is_empty() {
            ClearHow::Nothing
        } else if mask != ColorWrites::ALL {
            draw(DrawTarget::Color {
                kind: ColorKind::of(format),
                mask: host_color_writes(mask),
            })
        } else if whole {
            ClearHow::Pass
        } else if samples == 1 {
            ClearHow::Scratch
        } else {
            draw(DrawTarget::Color {
                kind: ColorKind::of(format),
                mask: wgpu::ColorWrites::ALL,
            })
        };
    }
    let depth = format.has_depth() && has_depth(clear.value);
    let stencil = (format.has_stencil() && clear.stencil_mask != 0)
        .then_some(stencil_value(clear.value))
        .flatten()
        .map(|_| clear.stencil_mask);
    if !depth && stencil.is_none() {
        return ClearHow::Nothing;
    }
    let partial_stencil = stencil.is_some_and(|mask| mask != ALL_STENCIL_BITS);
    if whole && !partial_stencil {
        ClearHow::Pass
    } else {
        draw(DrawTarget::DepthStencil { depth, stencil })
    }
}

/// What a render pass's load operation writes for `clear` ([`ClearHow::Pass`]): the value's
/// aspects, with the stencil kept when the stencil mask is empty.
pub fn pass_ops(clear: &ClearDesc) -> ClearOps {
    match crate::plan::clear_ops(clear.value) {
        ClearOps::DepthStencil { depth, stencil } => ClearOps::DepthStencil {
            depth,
            stencil: stencil.filter(|_| clear.stencil_mask != 0),
        },
        color @ ClearOps::Color(_) => color,
    }
}

fn has_depth(value: ClearValue) -> bool {
    matches!(
        value,
        ClearValue::Depth(_) | ClearValue::DepthStencil { .. }
    )
}

fn stencil_value(value: ClearValue) -> Option<u32> {
    match value {
        ClearValue::Stencil(s) | ClearValue::DepthStencil { stencil: s, .. } => Some(s),
        _ => None,
    }
}

/// The render thread's objects for drawn clears (module docs), made on first use.
pub(crate) struct Drawer {
    module: wgpu::ShaderModule,
    layout: wgpu::PipelineLayout,
    params: wgpu::Buffer,
    group: wgpu::BindGroup,
    pipelines: HashMap<PipelineKey, wgpu::RenderPipeline>,
}

impl Drawer {
    /// The shader, layouts, buffer and bind group; `None` (logged) when the clear shader does
    /// not parse.
    pub(crate) fn new(device: &wgpu::Device) -> Option<Self> {
        let module = match naga::front::wgsl::parse_str(HostShader::Clear.source()) {
            Ok(module) => module,
            Err(e) => {
                tracing::error!("the clear shader did not parse: {e}");
                return None;
            }
        };
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("starquake clear"),
            source: wgpu::ShaderSource::Naga(std::borrow::Cow::Owned(module)),
        });
        let size = u64::from(shader::PARAMS_SIZE);
        let group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("starquake clear"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: shader::PARAMS_BINDING,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(size),
                },
                count: None,
            }],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("starquake clear"),
            bind_group_layouts: &[Some(&group_layout)],
            immediate_size: 0,
        });
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("starquake clear parameters"),
            size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("starquake clear"),
            layout: &group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: shader::PARAMS_BINDING,
                resource: params.as_entire_binding(),
            }],
        });
        Some(Self {
            module,
            layout,
            params,
            group,
            pipelines: HashMap::new(),
        })
    }

    /// The parameter buffer a drawn clear's value is written to.
    pub(crate) fn params(&self) -> &wgpu::Buffer {
        &self.params
    }

    /// Encodes `clear` into `view` (one layer, or one slice when `depth_slice` is set) of a
    /// level `area.size` wide and high, within `area.rect`. The value must already be in
    /// [`Drawer::params`].
    pub(crate) fn draw(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        (view, depth_slice): (&wgpu::TextureView, Option<u32>),
        clear: &DrawnClear,
        area: Area,
    ) {
        let kept = self.pipelines.contains_key(&clear.key);
        if must_evict(self.pipelines.len(), MAX_CLEAR_PIPELINES, kept) {
            tracing::debug!("the clear pipelines reached their cap and are made again as needed");
            self.pipelines.clear();
        }
        let (module, layout) = (&self.module, &self.layout);
        let pipeline = self
            .pipelines
            .entry(clear.key)
            .or_insert_with(|| make_pipeline(device, module, layout, &clear.key));
        let color = matches!(clear.key.target, DrawTarget::Color { .. }).then_some(
            wgpu::RenderPassColorAttachment {
                view,
                depth_slice,
                resolve_target: None,
                ops: keep(),
            },
        );
        let format = clear.key.format;
        let depth_stencil =
            matches!(clear.key.target, DrawTarget::DepthStencil { .. }).then(|| {
                wgpu::RenderPassDepthStencilAttachment {
                    view,
                    depth_ops: format.has_depth_aspect().then_some(keep()),
                    stencil_ops: format.has_stencil_aspect().then_some(keep()),
                }
            });
        let slot = [color];
        let colors: &[Option<wgpu::RenderPassColorAttachment<'_>>] =
            if slot[0].is_some() { &slot } else { &[] };
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("starquake clear"),
            // A depth and stencil pass has no colour slot at all: a pipeline with no colour
            // targets does not match one empty slot.
            color_attachments: colors,
            depth_stencil_attachment: depth_stencil,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        let (width, height) = area.size;
        let scissor = area.rect.unwrap_or(Rect {
            x: 0,
            y: 0,
            width,
            height,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(shader::GROUP, &self.group, &[]);
        pass.set_viewport(0.0, 0.0, extent(width), extent(height), 0.0, 1.0);
        pass.set_scissor_rect(scissor.x, scissor.y, scissor.width, scissor.height);
        pass.set_stencil_reference(clear.stencil_reference);
        pass.draw(0..shader::VERTEX_COUNT, 0..1);
    }
}

/// Whether the kept pipelines are dropped before a drawn clear (module docs): its key is not
/// kept and `kept` pipelines already reach `cap`.
pub(crate) const fn must_evict(kept: usize, cap: usize, has_key: bool) -> bool {
    !has_key && kept >= cap
}

/// Where a drawn clear writes in one layer.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Area {
    /// The level's width and height: the viewport.
    pub(crate) size: (u32, u32),
    /// The rectangle, or `None` for the whole level: the scissor.
    pub(crate) rect: Option<Rect>,
}

/// An attachment's operations that load what is there and store what is drawn.
fn keep<V>() -> wgpu::Operations<V> {
    wgpu::Operations {
        load: wgpu::LoadOp::Load,
        store: wgpu::StoreOp::Store,
    }
}

/// A level's extent as a viewport's float.
#[expect(
    clippy::cast_precision_loss,
    reason = "a texture extent is at most wgpu's largest dimension, far below 2^24"
)]
fn extent(v: u32) -> f32 {
    v as f32
}

/// The pipeline of `key` (module docs).
fn make_pipeline(
    device: &wgpu::Device,
    module: &wgpu::ShaderModule,
    layout: &wgpu::PipelineLayout,
    key: &PipelineKey,
) -> wgpu::RenderPipeline {
    let color_targets;
    let (fragment, depth_stencil) = match key.target {
        DrawTarget::Color { kind, mask } => {
            color_targets = [Some(wgpu::ColorTargetState {
                format: key.format,
                blend: None,
                write_mask: mask,
            })];
            (Some((kind.entry_point(), &color_targets[..])), None)
        }
        DrawTarget::DepthStencil { depth, stencil } => {
            let face = wgpu::StencilFaceState {
                compare: wgpu::CompareFunction::Always,
                fail_op: wgpu::StencilOperation::Keep,
                depth_fail_op: wgpu::StencilOperation::Keep,
                pass_op: wgpu::StencilOperation::Replace,
            };
            let state = wgpu::DepthStencilState {
                format: key.format,
                depth_write_enabled: Some(depth),
                depth_compare: Some(wgpu::CompareFunction::Always),
                stencil: stencil.map_or_else(wgpu::StencilState::default, |mask| {
                    wgpu::StencilState {
                        front: face,
                        back: face,
                        read_mask: u32::from(ALL_STENCIL_BITS),
                        write_mask: u32::from(mask),
                    }
                }),
                bias: wgpu::DepthBiasState::default(),
            };
            let fragment = key
                .format
                .has_depth_aspect()
                .then_some((shader::FRAGMENT_DEPTH, &[][..]));
            (fragment, Some(state))
        }
    };
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("starquake clear"),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module,
            entry_point: Some(shader::VERTEX),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: &[],
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..wgpu::PrimitiveState::default()
        },
        depth_stencil,
        multisample: wgpu::MultisampleState {
            count: key.samples,
            ..wgpu::MultisampleState::default()
        },
        fragment: fragment.map(|(entry, targets)| wgpu::FragmentState {
            module,
            entry_point: Some(entry),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            targets,
        }),
        multiview_mask: None,
        cache: None,
    })
}

#[cfg(test)]
mod tests;
