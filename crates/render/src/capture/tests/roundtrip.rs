//! A generated build-then-read round trip: random sequences of every record kind, made valid
//! for the order checks, written through the writer and read back equal. Every field takes
//! random values, so two reads swapped would give a different record.

use super::super::{
    CapturedTarget, Create, CreateShader, CreateView, Created, Destroy, FrameResult, Limit,
    PipelineOutcome, PipelineResult, ReadbackResult, Record, ShaderCode, SourceRecord, Submit,
    TargetRecord,
};
use super::{read, writer};
use crate::backend::Readback;
use crate::command::tests::every_command_from;
use crate::command::{Command, CommandList, Marker};
use crate::format::PixelFormat;
use crate::id::{
    BufferId, FenceValue, PipelineId, ReadbackId, ResourceId, SamplerId, ShaderId, TextureId,
    ViewId,
};
use crate::pipeline::ComputePipelineDesc;
use crate::resource::{
    AddressMode, BorderColor, BufferDesc, BufferUsage, Extent3d, FilterMode, ReductionMode,
    SamplerDesc, Swizzle, TextureDesc, TextureDimension, TextureUsage, ViewDesc, ViewDimension,
};
use crate::shader::{
    InterfaceVariable, Interpolation, ResourceBinding, ResourceKind, ScalarKind, ShaderInterface,
    ShaderRequirements, ShaderStage,
};
use std::num::NonZeroU32;
use proptest::prelude::*;
use proptest::test_runner::{Config, RngSeed};

/// One generated step: a kind and the numbers its record is made from.
#[derive(Clone, Debug)]
struct Step {
    kind: u8,
    a: u64,
    b: u32,
    c: u32,
    made: bool,
    bytes: Vec<u8>,
}

fn step() -> impl Strategy<Value = Step> {
    (
        0..14u8,
        any::<u64>(),
        any::<u32>(),
        any::<u32>(),
        any::<bool>(),
        proptest::collection::vec(any::<u8>(), 0..6),
    )
        .prop_map(|(kind, a, b, c, made, bytes)| Step {
            kind,
            a,
            b,
            c,
            made,
            bytes,
        })
}

/// What the order checks need kept from step to step.
#[derive(Default)]
struct Kept {
    slot: u32,
    fence: u64,
    sources: u32,
    live: Vec<ResourceId>,
}

/// Turns steps into records the order checks accept: fresh ids, rising fences, sources in
/// order, shaders naming given sources.
fn records(steps: &[Step], limit: bool) -> Vec<Record> {
    let mut made = Kept::default();
    let mut out = Vec::new();
    for s in steps {
        made.slot += 1;
        let record = match s.kind {
            4 | 5 | 7 | 8 => kept(s, &mut made),
            _ => plain(s, made.slot),
        };
        if let Some(id) = record.facts().created {
            made.live.push(id);
        }
        out.push(record);
    }
    if limit {
        out.push(Record::Limit(Limit {
            fence: FenceValue::new(made.fence),
        }));
    }
    out.push(Record::End);
    out
}

/// A refusal's text.
fn text(s: &Step) -> String {
    format!("r{}", s.b)
}

/// The records that need nothing kept from step to step.
fn plain(s: &Step, slot: u32) -> Record {
    match s.kind {
        0 => Record::CreateBuffer(Create {
            desc: BufferDesc {
                size: s.a,
                usage: BufferUsage::from_bits(s.b & BufferUsage::ALL.bits()).expect("flags"),
            },
            result: if s.made {
                Created::Made(BufferId::new(slot, s.c))
            } else {
                Created::Refused(text(s))
            },
        }),
        1 => Record::CreateTexture(Create {
            desc: TextureDesc {
                dimension: TextureDimension::Cube,
                format: PixelFormat::ALL[usize::try_from(s.c % 112).expect("small")],
                size: Extent3d {
                    width: s.b,
                    height: s.c,
                    depth_or_layers: s.b ^ s.c,
                },
                mip_levels: s.b.rotate_left(3),
                sample_count: s.c.rotate_left(5),
                usage: TextureUsage::from_bits(s.c & TextureUsage::ALL.bits()).expect("flags"),
            },
            result: if s.made {
                Created::Made(TextureId::new(slot, s.b))
            } else {
                Created::Refused(text(s))
            },
        }),
        2 => Record::CreateView(CreateView {
            texture: TextureId::new(s.b, s.c),
            desc: ViewDesc {
                format: PixelFormat::ALL[usize::try_from(s.b % 112).expect("small")],
                dimension: ViewDimension::D1Array,
                base_mip: s.b,
                mip_count: s.c,
                base_layer: s.b.wrapping_add(1),
                layer_count: s.c.wrapping_add(2),
                swizzle: Swizzle::IDENTITY,
            },
            result: if s.made {
                Created::Made(ViewId::new(slot, s.c))
            } else {
                Created::Refused(text(s))
            },
        }),
        3 => Record::CreateSampler(Create {
            desc: SamplerDesc {
                mag_filter: FilterMode::Nearest,
                min_filter: FilterMode::Linear,
                mip_filter: FilterMode::Nearest,
                address: [
                    AddressMode::Repeat,
                    AddressMode::MirrorClampToEdge,
                    AddressMode::ClampToEdge,
                ],
                border_color: BorderColor::OpaqueWhite,
                lod_bias: f32::from(u16::try_from(s.b >> 16).expect("small")),
                min_lod: f32::from(u16::try_from(s.c >> 16).expect("small")),
                max_lod: 1.25,
                max_anisotropy: s.b,
                compare: None,
                reduction: ReductionMode::Min,
            },
            result: Created::Made(SamplerId::new(slot, s.b)),
        }),
        6 => Record::CreateComputePipeline(Create {
            desc: ComputePipelineDesc {
                shader: ShaderId::new(s.b, s.c),
            },
            result: Created::Made(PipelineId::new(slot, s.c)),
        }),
        _ => results(s),
    }
}

/// The result records, and a refused pipeline.
fn results(s: &Step) -> Record {
    match s.kind {
        9 => Record::PresentTarget(TargetRecord {
            target: if s.made {
                CapturedTarget::Window
            } else {
                CapturedTarget::Offscreen
            },
            result: if s.a.is_multiple_of(2) {
                Created::Made(())
            } else {
                Created::Refused(text(s))
            },
        }),
        10 => Record::ReadbackResult(ReadbackResult {
            id: ReadbackId::new(s.b, s.c),
            value: match s.a % 3 {
                0 => None,
                1 => Some(Readback::Unknown),
                _ => Some(Readback::Bytes(s.bytes.clone())),
            },
        }),
        11 => Record::PipelineOutcome(PipelineOutcome {
            pipeline: PipelineId::new(s.b, s.c),
            outcome: if s.made {
                PipelineResult::Ready
            } else {
                PipelineResult::Failed
            },
        }),
        12 => Record::FrameResult(FrameResult {
            frame: s.a,
            width: s.b,
            height: s.c,
            sha256: s.made.then(|| {
                std::array::from_fn(|i| {
                    u8::try_from(i).expect("small") ^ s.bytes.first().copied().unwrap_or(0)
                })
            }),
        }),
        _ => Record::CreateComputePipeline(Create {
            desc: ComputePipelineDesc {
                shader: ShaderId::new(s.c, s.b),
            },
            result: Created::Refused(text(s)),
        }),
    }
}

/// The records that need what was kept: sources, shaders naming them, destroys of live ids
/// and lists with rising fences.
fn kept(s: &Step, made: &mut Kept) -> Record {
    let slot = made.slot;
    match s.kind {
        4 => {
            let index = made.sources;
            made.sources += 1;
            Record::ShaderSource(SourceRecord {
                index,
                bytes: s.bytes.clone(),
            })
        }
        5 => Record::CreateShader(CreateShader {
            stage: ShaderStage::TessEval,
            entry: text(s),
            interface: ShaderInterface {
                resources: vec![ResourceBinding {
                    group: s.c,
                    binding: s.b,
                    kind: ResourceKind::Sampler {
                        comparison: s.made,
                    },
                    // `None` when `a % 64` is zero; the golden file holds both forms.
                    count: NonZeroU32::new(u32::try_from(s.a % 64).expect("small")),
                }],
                inputs: vec![InterfaceVariable {
                    location: s.b,
                    scalar: ScalarKind::Float,
                    components: s.c,
                    interpolation: Interpolation::Perspective,
                }],
                outputs: Vec::new(),
                requirements: ShaderRequirements {
                    subgroup: !s.made,
                },
            },
            code: match (made.sources, s.a % 4) {
                (n, 0) if n > 0 => ShaderCode::Origin {
                    source: s.b % n,
                    stage_index: s.c,
                },
                (_, 1) => ShaderCode::SpirV(vec![s.b, s.c]),
                (_, 2) => ShaderCode::Msl(text(s)),
                _ => ShaderCode::Missing,
            },
            result: if s.made {
                Created::Made(ShaderId::new(slot, s.b))
            } else {
                Created::Refused(text(s))
            },
        }),
        7 => {
            // Destroy a made.live id when there is one, else a made-up one.
            let id = if made.live.is_empty() {
                ResourceId::Sampler(SamplerId::new(s.b, s.c))
            } else {
                made.live
                    .remove(usize::try_from(s.a).unwrap_or(0) % made.live.len())
            };
            Record::Destroy(Destroy {
                id,
                after: FenceValue::new(s.a),
            })
        }
        _ => {
            let list = if s.made {
                every_command_from(s.b % 5000)
            } else {
                CommandList {
                    commands: vec![Command::Marker(Marker {
                        submit: s.a,
                        draw: u64::from(s.c),
                    })],
                }
            };
            let result = if s.a.is_multiple_of(3) {
                Created::Refused(text(s))
            } else {
                made.fence += 1 + u64::from(s.b % 1000);
                Created::Made(FenceValue::new(made.fence))
            };
            Record::Submit(Submit { list, result })
        }
    }
}

proptest! {
    #![proptest_config(Config {
        cases: 64,
        rng_seed: RngSeed::Fixed(0x5afe_c0de),
        ..Config::default()
    })]

    #[test]
    fn records_read_back_as_written(
        steps in proptest::collection::vec(step(), 0..24),
        limit in any::<bool>(),
    ) {
        let expected = records(&steps, limit);
        let (mut w, buffer) = writer();
        for record in &expected {
            w.write(record).expect("a valid record");
        }
        let read_back = read(&buffer.bytes()).expect("reads");
        prop_assert_eq!(&read_back[1..], &expected[..]);
    }
}
