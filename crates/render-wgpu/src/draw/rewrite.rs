//! The index rewrite (`rewrite.wgsl`): which draws need it ([`plan`]), how many indices it writes
//! ([`output_count`]), the parameter block the shader reads ([`params`]), and the compute
//! pipeline the render thread runs it with ([`IndexRewrite`]). The crate's `draw` module docs,
//! "The index rewrite", say what it is for and where it differs from the guest.
//!
//! The shader is WGSL (render-backend, decision 12), validated with naga by
//! `draw::rewrite::tests`, with no GPU, against the names and layout here; a GPU test runs it on
//! every mode and source and compares its output with an independent model of the same rules.
//! It lives in this crate rather than beside R4's host shaders in `starquake-render`: only the
//! wgpu backend draws through it today, and a later direct backend can move it there.

use starquake_render::command::IndexFormat;
use starquake_render::pipeline::Topology;
use std::borrow::Cow;
use wgpu::naga;

/// The WGSL source.
pub const SOURCE: &str = include_str!("rewrite.wgsl");
/// The compute entry point.
pub const ENTRY: &str = "main";
/// The bind group every resource is in.
pub const GROUP: u32 = 0;
/// Binding of the parameter block, a uniform buffer of [`PARAMS_SIZE`] bytes.
pub const PARAMS_BINDING: u32 = 0;
/// Binding of the input indices, a read-only storage buffer of 32-bit words.
pub const INPUT_BINDING: u32 = 1;
/// Binding of the output indices, a read-write storage buffer of 32-bit words.
pub const OUTPUT_BINDING: u32 = 2;
/// Invocations per workgroup (the shader's `WORKGROUP_SIZE`).
pub const WORKGROUP_SIZE: u32 = 64;
/// The 32-bit restart value (the shader's `RESTART`), which an input equal to its own format's
/// restart value becomes when [`Rewrite::restart`] is set.
pub const RESTART: u32 = u32::MAX;

/// Bytes of the parameter block as the backend writes it: the shader's `Params` (six 32-bit
/// words, 24 bytes) padded to 32, so each block in a list's parameter buffer starts 16-byte
/// aligned whatever the device's uniform offset alignment is.
pub const PARAMS_LEN: usize = 32;
/// [`PARAMS_LEN`] as a buffer size.
pub const PARAMS_SIZE: u64 = PARAMS_LEN as u64;
/// Byte offset of `Params::count`: the output indices.
pub const COUNT_OFFSET: usize = 0;
/// Byte offset of `Params::mode`: a [`Mode`].
pub const MODE_OFFSET: usize = 4;
/// Byte offset of `Params::source`: a [`Source`].
pub const SOURCE_OFFSET: usize = 8;
/// Byte offset of `Params::first`: the first vertex, or the first index's byte offset.
pub const FIRST_OFFSET: usize = 12;
/// Byte offset of `Params::input_count`: the draw's vertex or index count.
pub const INPUT_COUNT_OFFSET: usize = 16;
/// Byte offset of `Params::restart`: 1 when [`Rewrite::restart`] is set.
pub const RESTART_OFFSET: usize = 20;

/// How output index k picks an input position (the shader's `MODE_` values).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Mode {
    /// Position k: indices widened, the topology kept.
    List = 0,
    /// Quads as two triangles each: positions `4q + (0, 1, 2, 0, 2, 3)` (Eden's order).
    Quads = 1,
    /// A quad strip as two triangles per quad: positions `2q + (0, 3, 1, 0, 2, 3)` (Eden's
    /// order).
    QuadStrip = 2,
    /// A fan (or polygon) as triangles `(t + 1, t + 2, 0)`, Vulkan's fan order.
    Fan = 3,
    /// A line loop as a line strip that repeats position 0 at its end.
    Loop = 4,
}

/// Where the value at an input position comes from (the shader's `SOURCE_` values).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Source {
    /// A non-indexed draw: the first vertex plus the position.
    Sequential = 0,
    /// 8-bit indices.
    U8 = 1,
    /// 16-bit indices.
    U16 = 2,
    /// 32-bit indices.
    U32 = 3,
}

impl Source {
    /// The source of a draw's indices, or of a non-indexed draw.
    pub const fn of(index: Option<IndexFormat>) -> Self {
        match index {
            None => Self::Sequential,
            Some(IndexFormat::U8) => Self::U8,
            Some(IndexFormat::U16) => Self::U16,
            Some(IndexFormat::U32) => Self::U32,
        }
    }

    /// Bytes per index, or 0 for the sequential source.
    pub const fn index_bytes(self) -> u32 {
        match self {
            Self::Sequential => 0,
            Self::U8 => 1,
            Self::U16 => 2,
            Self::U32 => 4,
        }
    }
}

/// What a draw's rewrite does.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Rewrite {
    /// How positions are picked.
    pub mode: Mode,
    /// Where values come from.
    pub source: Source,
    /// Whether an input equal to its format's restart value becomes [`RESTART`].
    pub restart: bool,
}

/// Whether wgpu restarts primitives of `topology` at the restart value: the strips (wgpu-hal
/// 30.0.2's Metal backend restarts every indexed strip; render-backend, decision 1's table).
const fn is_strip(topology: Topology) -> bool {
    matches!(topology, Topology::LineStrip | Topology::TriangleStrip)
}

/// The rewrite a draw of `topology` needs, with `index` its index format (`None` for a
/// non-indexed draw) and `primitive_restart` the pipeline's setting; `None` when wgpu draws it
/// as it is. In this order:
///
/// 1. quads, quad strips, fans, polygons and line loops are rewritten from any source, with
///    restart off (the restart value is drawn as a vertex, but for a line loop from 32-bit
///    indices, whose `0xFFFFFFFF` the host still restarts at: `draw` module docs,
///    "Divergences");
/// 2. 8-bit indices (wgpu has none) are widened, with restart kept on a strip whose pipeline
///    restarts;
/// 3. 16-bit indices of a strip are widened, since a strip pipeline takes 32-bit indices only
///    (`crate::pipeline::state::STRIP_INDEX_FORMAT`): `0xFFFF` becomes the 32-bit restart value
///    when the pipeline restarts, and stays a vertex when it does not (render-backend,
///    "Primitive restart off"). 32-bit strip indices are drawn as they are, and cannot be
///    widened when restart is off (`draw` module docs, "Divergences").
pub const fn plan(
    topology: Topology,
    primitive_restart: bool,
    index: Option<IndexFormat>,
) -> Option<Rewrite> {
    let source = Source::of(index);
    let mode = match topology {
        Topology::Quads => Mode::Quads,
        Topology::QuadStrip => Mode::QuadStrip,
        Topology::TriangleFan | Topology::Polygon => Mode::Fan,
        Topology::LineLoop => Mode::Loop,
        _ => Mode::List,
    };
    if !matches!(mode, Mode::List) {
        return Some(Rewrite {
            mode,
            source,
            restart: false,
        });
    }
    match index {
        Some(IndexFormat::U8) => Some(Rewrite {
            mode,
            source,
            restart: primitive_restart && is_strip(topology),
        }),
        Some(IndexFormat::U16) if is_strip(topology) => Some(Rewrite {
            mode,
            source,
            restart: primitive_restart,
        }),
        _ => None,
    }
}

/// How many indices a rewrite in `mode` writes for `n` input positions, or `None` when that
/// does not fit a `u32`.
pub fn output_count(mode: Mode, n: u32) -> Option<u32> {
    let n = u64::from(n);
    let count = match mode {
        Mode::List => n,
        Mode::Quads => (n / 4) * 6,
        Mode::QuadStrip if n >= 4 => (n - 2) / 2 * 6,
        Mode::Fan if n >= 3 => (n - 2) * 3,
        Mode::Loop if n >= 2 => n + 1,
        Mode::QuadStrip | Mode::Fan | Mode::Loop => 0,
    };
    u32::try_from(count).ok()
}

/// The parameter block of a rewrite of `input_count` positions into `count` indices, reading
/// from `first` (the first vertex, or the first index's byte offset in the bound input).
pub fn params(rewrite: Rewrite, count: u32, first: u32, input_count: u32) -> [u8; PARAMS_LEN] {
    let mut block = [0; PARAMS_LEN];
    for (offset, value) in [
        (COUNT_OFFSET, count),
        (MODE_OFFSET, rewrite.mode as u32),
        (SOURCE_OFFSET, rewrite.source as u32),
        (FIRST_OFFSET, first),
        (INPUT_COUNT_OFFSET, input_count),
        (RESTART_OFFSET, u32::from(rewrite.restart)),
    ] {
        if let Some(word) = block.get_mut(offset..offset + 4) {
            word.copy_from_slice(&value.to_le_bytes());
        }
    }
    block
}

/// Workgroups for a rewrite of `count` indices: enough for one index per invocation, at most
/// `max` (the device's per-dimension limit); the shader loops over the rest.
pub fn workgroups(count: u32, max: u32) -> u32 {
    count.div_ceil(WORKGROUP_SIZE).min(max).max(1)
}

/// The rewrite's compute pipeline and what every run binds, made once on the render thread.
#[derive(Debug)]
pub struct IndexRewrite {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    /// Bound as the input of a sequential rewrite, which reads none.
    empty: wgpu::Buffer,
}

/// One run of the rewrite.
#[derive(Clone, Copy, Debug)]
pub struct Run<'a> {
    /// The list's parameter buffer and this run's block in it.
    pub params: (&'a wgpu::Buffer, u64),
    /// The bound input (buffer, offset, size), or `None` for the sequential source.
    pub input: Option<(&'a wgpu::Buffer, u64, u64)>,
    /// The list's output buffer, this run's offset in it and the indices it writes.
    pub output: (&'a wgpu::Buffer, u64, u32),
    /// The device's per-dimension workgroup limit.
    pub max_groups: u32,
}

impl IndexRewrite {
    /// Makes the pipeline from [`SOURCE`], inside wgpu error scopes. `Err` carries why; the
    /// shader's own tests rule out a parse or validation failure.
    pub fn new(device: &wgpu::Device) -> Result<Self, String> {
        let module = naga::front::wgsl::parse_str(SOURCE).map_err(|e| e.to_string())?;
        crate::pipeline::scoped(device, || {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("starquake index rewrite"),
                source: wgpu::ShaderSource::Naga(Cow::Owned(module)),
            });
            let storage = |read_only| wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            };
            let entry = |binding, ty| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty,
                count: None,
            };
            let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("starquake index rewrite"),
                entries: &[
                    entry(
                        PARAMS_BINDING,
                        wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(PARAMS_SIZE),
                        },
                    ),
                    entry(INPUT_BINDING, storage(true)),
                    entry(OUTPUT_BINDING, storage(false)),
                ],
            });
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("starquake index rewrite"),
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("starquake index rewrite"),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some(ENTRY),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
            let empty = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("starquake index rewrite input"),
                size: wgpu::COPY_BUFFER_ALIGNMENT,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            });
            Self {
                pipeline,
                layout,
                empty,
            }
        })
    }

    /// Encodes `run` as a compute pass of its own.
    pub fn encode(&self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder, run: Run<'_>) {
        let (params, params_offset) = run.params;
        let (output, output_offset, count) = run.output;
        let input = match run.input {
            Some((buffer, offset, size)) => wgpu::BufferBinding {
                buffer,
                offset,
                size: wgpu::BufferSize::new(size),
            },
            None => self.empty.as_entire_buffer_binding(),
        };
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("starquake index rewrite"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: PARAMS_BINDING,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: params,
                        offset: params_offset,
                        size: wgpu::BufferSize::new(PARAMS_SIZE),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: INPUT_BINDING,
                    resource: wgpu::BindingResource::Buffer(input),
                },
                wgpu::BindGroupEntry {
                    binding: OUTPUT_BINDING,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: output,
                        offset: output_offset,
                        size: wgpu::BufferSize::new(u64::from(count).saturating_mul(4)),
                    }),
                },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("starquake index rewrite"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(GROUP, &group, &[]);
        pass.dispatch_workgroups(workgroups(count, run.max_groups), 1, 1);
    }
}

#[cfg(test)]
mod tests;
