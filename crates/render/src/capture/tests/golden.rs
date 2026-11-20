//! Golden bytes: the header and every record kind, each field with a distinct non-zero value
//! where its type allows one, written out by hand from the layout in the module docs. The file
//! is also the hand-built accepting case for every record kind: it is read back as well as
//! written.

use super::super::{
    CaptureBuffer, CaptureSink, CaptureWriter, CapturedTarget, Create, CreateShader, CreateView,
    Created, Destroy, FrameResult, Limit, PipelineOutcome, PipelineResult, ReadbackResult, Record,
    RecordKind, ShaderCode, SourceRecord, Start, Submit, TargetRecord, WaitSetting,
};
use super::read;
use crate::backend::Readback;
use crate::caps::{
    Caps, DynamicState, Features, FormatFeatures, FormatTable, Limits, SampleCounts, StageSupport,
};
use crate::command::{Command, CommandList, Marker};
use crate::format::PixelFormat;
use crate::id::{
    BufferId, FenceValue, PipelineId, ReadbackId, ResourceId, SamplerId, ShaderId, TextureId,
    ViewId,
};
use crate::pipeline::{
    BlendComponent, BlendFactor, BlendOperation, BlendState, ColorWrites, ComputePipelineDesc,
    CullMode, DepthBias, DepthStencilState, EmulationFlags, FrontFace, GraphicsPipelineDesc,
    LogicOp, PolygonMode, RasterState, Stages, StencilFace, StencilOperation, StencilState,
    TargetFormats, Topology, VertexAttribute, VertexBufferLayout, VertexComponents, VertexKind,
    VertexLayout, VertexStepMode,
};
use crate::resource::{
    AddressMode, BorderColor, BufferDesc, BufferUsage, CompareFunction, Extent3d, FilterMode,
    ReductionMode, SamplerDesc, Swizzle, SwizzleSource, TextureDesc, TextureDimension,
    TextureUsage, ViewDesc, ViewDimension,
};
use crate::shader::{
    InterfaceVariable, Interpolation, ResourceBinding, ResourceKind, SampleKind, ScalarKind,
    ShaderInterface, ShaderRequirements, ShaderStage,
};
use std::num::NonZeroU32;
use std::sync::Arc;

#[test]
fn tags() {
    // The tags, written out: a wrong one in `RecordKind::tag` fails here, not only a swap.
    let expected = [
        (RecordKind::End, 0),
        (RecordKind::Start, 1),
        (RecordKind::CreateBuffer, 2),
        (RecordKind::CreateTexture, 3),
        (RecordKind::CreateView, 4),
        (RecordKind::CreateSampler, 5),
        (RecordKind::ShaderSource, 6),
        (RecordKind::CreateShader, 7),
        (RecordKind::CreateGraphicsPipeline, 8),
        (RecordKind::CreateComputePipeline, 9),
        (RecordKind::Destroy, 10),
        (RecordKind::Submit, 11),
        (RecordKind::PresentTarget, 12),
        (RecordKind::ReadbackResult, 13),
        (RecordKind::PipelineOutcome, 14),
        (RecordKind::FrameResult, 15),
        (RecordKind::Limit, 16),
    ];
    assert_eq!(RecordKind::ALL.len(), expected.len());
    for (kind, tag) in expected {
        assert_eq!(kind.tag(), tag, "{kind}");
        assert_eq!(RecordKind::from_tag(tag), Some(kind));
    }
    assert_eq!(RecordKind::from_tag(17), None);
    assert_eq!(RecordKind::from_tag(u8::MAX), None);
}

#[test]
fn header() {
    let mut expected = b"SQRC".to_vec();
    expected.push(0x03);
    assert_eq!(super::super::MAGIC, *b"SQRC");
    assert_eq!(super::super::VERSION, 3);
    assert!(super::prefix().starts_with(&expected));
}

/// The golden capture's device: every limit distinct, the two 64-bit ones past 32 bits.
fn golden_caps() -> Caps {
    let mut formats = FormatTable::uniform(FormatFeatures::ALL);
    formats.set(
        PixelFormat::A8B8G8R8Unorm,
        FormatFeatures::from_bits(0x21).expect("flags"),
    );
    formats.set(
        PixelFormat::D32FloatS8Uint,
        FormatFeatures::from_bits(0x22).expect("flags"),
    );
    Caps {
        limits: Limits {
            max_buffer_size: 0x1_0000_0001,
            max_staging_size: 0x2_0000_0002,
            max_texture_dimension_1d: 3,
            max_texture_dimension_2d: 4,
            max_texture_dimension_3d: 5,
            max_texture_array_layers: 6,
            max_sampler_anisotropy: 7,
            max_vertex_attributes: 8,
            max_vertex_buffers: 9,
            max_vertex_buffer_stride: 10,
            max_color_attachments: 11,
            max_viewports: 12,
            max_bind_groups: 13,
            max_bindings_per_bind_group: 14,
            max_compute_workgroups_per_dimension: 15,
            max_uniform_buffers_per_stage: 16,
            max_storage_buffers_per_stage: 17,
            max_sampled_textures_per_stage: 18,
            max_storage_textures_per_stage: 19,
            max_samplers_per_stage: 20,
        },
        features: Features::from_bits(0x17).expect("flags"),
        stages: StageSupport::from_bits(0x02).expect("flags"),
        dynamic_state: DynamicState::from_bits(0x18).expect("flags"),
        sample_counts: SampleCounts::from_bits(0x19).expect("flags"),
        view_swizzle: true,
        formats,
    }
}

fn golden_start() -> Start {
    Start {
        build: "b1".to_owned(),
        backend: "rec".to_owned(),
        caps: golden_caps(),
        wait: WaitSetting::Skip,
        memory_limit: 0x3_0000_0003,
    }
}

/// An enabled blend state with every factor and operation distinct, for the first and last
/// colour target, so that the array's order shows.
fn blend(first: bool) -> BlendState {
    if first {
        BlendState {
            enabled: true,
            color: BlendComponent {
                src: BlendFactor::SrcAlpha,
                dst: BlendFactor::OneMinusSrcAlpha,
                op: BlendOperation::Subtract,
            },
            alpha: BlendComponent {
                src: BlendFactor::One,
                dst: BlendFactor::Dst,
                op: BlendOperation::Max,
            },
            write_mask: ColorWrites::from_bits(0x0B).expect("flags"),
        }
    } else {
        BlendState {
            enabled: true,
            color: BlendComponent {
                src: BlendFactor::Constant,
                dst: BlendFactor::OneMinusConstant,
                op: BlendOperation::ReverseSubtract,
            },
            alpha: BlendComponent {
                src: BlendFactor::Src1Alpha,
                dst: BlendFactor::OneMinusSrc1,
                op: BlendOperation::Min,
            },
            write_mask: ColorWrites::from_bits(0x06).expect("flags"),
        }
    }
}

fn blend_off() -> BlendState {
    let c = BlendComponent {
        src: BlendFactor::One,
        dst: BlendFactor::Zero,
        op: BlendOperation::Add,
    };
    BlendState {
        enabled: false,
        color: c,
        alpha: c,
        write_mask: ColorWrites::ALL,
    }
}

pub(super) fn golden_pipeline() -> GraphicsPipelineDesc {
    let mut blends = [blend_off(); 8];
    blends[0] = blend(true);
    blends[7] = blend(false);
    let mut colors = [None; 8];
    colors[0] = Some(PixelFormat::A8B8G8R8Uint);
    colors[7] = Some(PixelFormat::R5G6B5Unorm);
    GraphicsPipelineDesc {
        stages: Stages {
            vertex: ShaderId::new(50, 51),
            tess_control: Some(ShaderId::new(52, 53)),
            tess_eval: None,
            geometry: Some(ShaderId::new(54, 55)),
            fragment: Some(ShaderId::new(56, 57)),
        },
        vertex_input: VertexLayout {
            buffers: vec![VertexBufferLayout {
                stride: 58,
                step: VertexStepMode::Instance,
            }],
            attributes: vec![VertexAttribute {
                location: 59,
                buffer: 60,
                offset: 61,
                components: VertexComponents::R16G16B16A16,
                kind: VertexKind::Uscaled,
            }],
        },
        topology: Topology::Quads,
        primitive_restart: true,
        raster: RasterState {
            cull: CullMode::Back,
            front_face: FrontFace::Cw,
            polygon_mode: PolygonMode::Line,
            depth_clamp: true,
            depth_bias: DepthBias {
                constant: 8.5,
                slope_scale: 9.5,
                clamp: 10.5,
            },
            line_width: 11.5,
        },
        depth_stencil: DepthStencilState {
            depth_test: Some(CompareFunction::LessEqual),
            depth_write: true,
            stencil: Some(StencilState {
                front: StencilFace {
                    compare: CompareFunction::NotEqual,
                    fail: StencilOperation::Replace,
                    depth_fail: StencilOperation::Invert,
                    pass: StencilOperation::IncrementWrap,
                },
                back: StencilFace {
                    compare: CompareFunction::GreaterEqual,
                    fail: StencilOperation::Zero,
                    depth_fail: StencilOperation::DecrementClamp,
                    pass: StencilOperation::DecrementWrap,
                },
                read_mask: 62,
                write_mask: 63,
            }),
        },
        blend: blends,
        logic_op: Some(LogicOp::Xor),
        targets: TargetFormats {
            colors,
            depth_stencil: Some(PixelFormat::D32Float),
            sample_count: 64,
        },
        emulation: EmulationFlags::from_bits(0x15).expect("flags"),
    }
}

/// One record of every kind, and every variant of the record-level enums, in an order the
/// reading checks accept.
fn golden_records() -> Vec<Record> {
    let mut records = golden_creates();
    records.extend(golden_shaders());
    records.extend(golden_results());
    records
}

/// The buffer, texture, view and sampler records.
fn golden_creates() -> Vec<Record> {
    vec![
        Record::CreateBuffer(Create {
            desc: BufferDesc {
                size: 0x4_0000_0004,
                usage: BufferUsage::from_bits(0x45).expect("flags"),
            },
            result: Created::Made(BufferId::new(5, 6)),
        }),
        Record::CreateTexture(Create {
            desc: TextureDesc {
                dimension: TextureDimension::D3,
                format: PixelFormat::A2B10G10R10Uint,
                size: Extent3d {
                    width: 9,
                    height: 10,
                    depth_or_layers: 11,
                },
                mip_levels: 12,
                sample_count: 13,
                usage: TextureUsage::from_bits(0x15).expect("flags"),
            },
            result: Created::Refused("no".to_owned()),
        }),
        Record::CreateView(CreateView {
            texture: TextureId::new(14, 15),
            desc: ViewDesc {
                format: PixelFormat::A1B5G5R5Unorm,
                dimension: ViewDimension::CubeArray,
                base_mip: 16,
                mip_count: 17,
                base_layer: 18,
                layer_count: 19,
                swizzle: Swizzle {
                    r: SwizzleSource::B,
                    g: SwizzleSource::A,
                    b: SwizzleSource::One,
                    a: SwizzleSource::G,
                },
            },
            result: Created::Made(ViewId::new(20, 21)),
        }),
        Record::CreateSampler(Create {
            desc: SamplerDesc {
                mag_filter: FilterMode::Linear,
                min_filter: FilterMode::Linear,
                mip_filter: FilterMode::Linear,
                address: [
                    AddressMode::MirrorRepeat,
                    AddressMode::ClampToEdge,
                    AddressMode::ClampToBorder,
                ],
                border_color: BorderColor::Custom([1.5, 2.5, 3.5, 4.5]),
                lod_bias: 5.5,
                min_lod: 6.5,
                max_lod: 7.5,
                max_anisotropy: 22,
                compare: Some(CompareFunction::Greater),
                reduction: ReductionMode::Max,
            },
            result: Created::Made(SamplerId::new(23, 24)),
        }),
    ]
}

/// The shader sources, shaders and pipelines.
fn golden_shaders() -> Vec<Record> {
    let empty = ShaderInterface::default;
    vec![
        Record::ShaderSource(SourceRecord {
            index: 0,
            bytes: vec![0xAA],
        }),
        Record::ShaderSource(SourceRecord {
            index: 1,
            bytes: vec![0xBB, 0xCC],
        }),
        Record::CreateShader(CreateShader {
            stage: ShaderStage::Geometry,
            entry: "vs".to_owned(),
            interface: ShaderInterface {
                resources: vec![
                    ResourceBinding {
                        group: 25,
                        binding: 26,
                        kind: ResourceKind::SampledTexture {
                            dimension: ViewDimension::D2Array,
                            sample: SampleKind::Float { filterable: true },
                            multisampled: true,
                        },
                        count: NonZeroU32::new(0x4E),
                    },
                    ResourceBinding {
                        group: 0x4F,
                        binding: 0x50,
                        kind: ResourceKind::Sampler { comparison: true },
                        count: None,
                    },
                ],
                inputs: vec![InterfaceVariable {
                    location: 27,
                    scalar: ScalarKind::Uint,
                    components: 3,
                    interpolation: Interpolation::Flat,
                }],
                outputs: vec![InterfaceVariable {
                    location: 28,
                    scalar: ScalarKind::Sint,
                    components: 4,
                    interpolation: Interpolation::Linear,
                }],
                requirements: ShaderRequirements {
                    subgroup: true,
                    f16: true,
                    f64: true,
                    i64: true,
                    i64_atomics: true,
                },
            },
            code: ShaderCode::Origin {
                source: 1,
                stage_index: 29,
            },
            result: Created::Made(ShaderId::new(30, 31)),
        }),
        Record::CreateShader(CreateShader {
            stage: ShaderStage::Vertex,
            entry: "a".to_owned(),
            interface: empty(),
            code: ShaderCode::SpirV(vec![0x0723_0203, 5]),
            result: Created::Made(ShaderId::new(67, 68)),
        }),
        Record::CreateShader(CreateShader {
            stage: ShaderStage::Fragment,
            entry: "b".to_owned(),
            interface: empty(),
            code: ShaderCode::Msl("k".to_owned()),
            result: Created::Refused("r".to_owned()),
        }),
        Record::CreateShader(CreateShader {
            stage: ShaderStage::Compute,
            entry: "c".to_owned(),
            interface: empty(),
            code: ShaderCode::Missing,
            result: Created::Made(ShaderId::new(69, 70)),
        }),
        Record::CreateGraphicsPipeline(Create {
            desc: golden_pipeline(),
            result: Created::Made(PipelineId::new(65, 66)),
        }),
        Record::CreateComputePipeline(Create {
            desc: ComputePipelineDesc {
                shader: ShaderId::new(32, 33),
            },
            result: Created::Refused("x".to_owned()),
        }),
    ]
}

/// The destroy, lists, target, results, limit and end.
fn golden_results() -> Vec<Record> {
    vec![
        Record::Destroy(Destroy {
            id: ResourceId::Texture(TextureId::new(34, 35)),
            after: FenceValue::new(36),
        }),
        Record::Submit(Submit {
            list: CommandList {
                commands: vec![Command::Marker(Marker {
                    submit: 37,
                    draw: 38,
                })],
            },
            result: Created::Made(FenceValue::new(0x5_0000_0005)),
        }),
        Record::Submit(Submit {
            list: CommandList::new(),
            result: Created::Refused("q".to_owned()),
        }),
        Record::PresentTarget(TargetRecord {
            target: CapturedTarget::Window,
            result: Created::Made(()),
        }),
        Record::ReadbackResult(ReadbackResult {
            id: ReadbackId::new(39, 40),
            value: Some(Readback::Bytes(vec![0xDE, 0xAD])),
        }),
        Record::ReadbackResult(ReadbackResult {
            id: ReadbackId::new(71, 72),
            value: Some(Readback::Unknown),
        }),
        Record::ReadbackResult(ReadbackResult {
            id: ReadbackId::new(73, 74),
            value: None,
        }),
        Record::PipelineOutcome(PipelineOutcome {
            pipeline: PipelineId::new(41, 42),
            outcome: PipelineResult::Failed,
        }),
        Record::FrameResult(FrameResult {
            frame: 43,
            width: 44,
            height: 45,
            sha256: Some(std::array::from_fn(|i| {
                46 + u8::try_from(i).expect("small")
            })),
        }),
        Record::FrameResult(FrameResult {
            frame: 75,
            width: 76,
            height: 77,
            sha256: None,
        }),
        Record::Limit(Limit {
            fence: FenceValue::new(0x6_0000_0006),
        }),
        Record::End,
    ]
}

/// The file of [`golden_start`] and [`golden_records`], by hand.
fn golden_file() -> Vec<u8> {
    let mut f: Vec<u8> = b"SQRC".to_vec();
    f.push(0x03); // version
    // Start: tag 1, 159 bytes.
    f.extend([0x01, 0x9F, 0x01]);
    f.extend([0x02, b'b', b'1', 0x03, b'r', b'e', b'c']);
    f.extend([0x81, 0x80, 0x80, 0x80, 0x10]); // max_buffer_size 2^32 + 1
    f.extend([0x82, 0x80, 0x80, 0x80, 0x20]); // max_staging_size 2^33 + 2
    f.extend(0x03..=0x14); // the eighteen u32 limits, 3 to 20
    f.extend([0x17, 0x02, 0x18, 0x19, 0x01]); // features, stages, dynamic, samples, swizzle
    f.push(0x70); // 112 formats
    f.push(0x21);
    f.extend(std::iter::repeat_n(0x3F, 110));
    f.push(0x22);
    f.push(0x01); // wait: Skip
    f.extend([0x83, 0x80, 0x80, 0x80, 0x30]); // memory limit
    // CreateBuffer.
    f.extend([
        0x02, 0x09, 0x84, 0x80, 0x80, 0x80, 0x40, 0x45, 0x00, 0x05, 0x06,
    ]);
    // CreateTexture, refused "no".
    f.extend([0x03, 0x0C, 0x02, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x15]);
    f.extend([0x01, 0x02, b'n', b'o']);
    // CreateView.
    f.extend([0x04, 0x0F, 0x0E, 0x0F, 0x0A, 0x05, 0x10, 0x11, 0x12, 0x13]);
    f.extend([0x04, 0x05, 0x01, 0x03, 0x00, 0x14, 0x15]);
    // CreateSampler.
    f.extend([0x05, 0x2A, 0x01, 0x01, 0x01, 0x01, 0x02, 0x03, 0x03]);
    f.extend([0x00, 0x00, 0xC0, 0x3F, 0x00, 0x00, 0x20, 0x40]); // 1.5, 2.5
    f.extend([0x00, 0x00, 0x60, 0x40, 0x00, 0x00, 0x90, 0x40]); // 3.5, 4.5
    f.extend([0x00, 0x00, 0xB0, 0x40, 0x00, 0x00, 0xD0, 0x40]); // 5.5, 6.5
    f.extend([0x00, 0x00, 0xF0, 0x40]); // 7.5
    f.extend([0x16, 0x01, 0x04, 0x02, 0x00, 0x17, 0x18]);
    // ShaderSource 0 and 1.
    f.extend([0x06, 0x03, 0x00, 0x01, 0xAA]);
    f.extend([0x06, 0x04, 0x01, 0x02, 0xBB, 0xCC]);
    // CreateShader from source 1: 40 bytes.
    f.extend([0x07, 0x28, 0x03, 0x02, b'v', b's']);
    // Two resources: a sampled texture array of 0x4E, then one comparison sampler.
    f.extend([0x02, 0x19, 0x1A, 0x02, 0x03, 0x00, 0x01, 0x01, 0x01, 0x4E]);
    f.extend([0x4F, 0x50, 0x04, 0x01, 0x00]);
    f.extend([0x01, 0x1B, 0x02, 0x03, 0x02, 0x01, 0x1C, 0x01, 0x04, 0x01]);
    f.extend([0x01; 5]); // requirements: subgroup, f16, f64, i64, i64_atomics
    f.extend([0x00, 0x01, 0x1D, 0x00, 0x1E, 0x1F]);
    // CreateShader with SPIR-V; the empty interface is three empty tables and no requirements.
    f.extend([
        0x07, 0x15, 0x00, 0x01, b'a', 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    ]);
    f.extend([0x01, 0x02, 0x83, 0x84, 0x8C, 0x39, 0x05, 0x00, 0x43, 0x44]);
    // CreateShader with MSL, refused "r".
    f.extend([
        0x07, 0x11, 0x04, 0x01, b'b', 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    ]);
    f.extend([0x02, 0x01, b'k', 0x01, 0x01, b'r']);
    // CreateShader with no origin.
    f.extend([
        0x07, 0x0F, 0x05, 0x01, b'c', 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0x00,
        0x45, 0x46,
    ]);
    // CreateGraphicsPipeline: 140 bytes.
    f.extend([0x08, 0x8C, 0x01]);
    f.extend([
        0x32, 0x33, 0x01, 0x34, 0x35, 0x00, 0x01, 0x36, 0x37, 0x01, 0x38, 0x39,
    ]);
    f.extend([0x01, 0x3A, 0x01, 0x01, 0x3B, 0x3C, 0x3D, 0x02, 0x04]); // vertex input
    f.extend([0x07, 0x01]); // quads, primitive restart
    f.extend([0x02, 0x01, 0x01, 0x01]); // raster
    f.extend([0x00, 0x00, 0x08, 0x41, 0x00, 0x00, 0x18, 0x41]); // 8.5, 9.5
    f.extend([0x00, 0x00, 0x28, 0x41, 0x00, 0x00, 0x38, 0x41]); // 10.5, 11.5
    f.extend([
        0x01, 0x03, 0x01, 0x01, 0x05, 0x02, 0x05, 0x06, 0x06, 0x01, 0x04, 0x07,
    ]);
    f.extend([0x3E, 0x3F]);
    f.extend([0x01, 0x04, 0x05, 0x01, 0x01, 0x06, 0x04, 0x0B]); // blend 0
    for _ in 1..7 {
        f.extend([0x00, 0x01, 0x00, 0x00, 0x01, 0x00, 0x00, 0x0F]);
    }
    f.extend([0x01, 0x0B, 0x0C, 0x02, 0x0F, 0x0E, 0x03, 0x06]); // blend 7
    f.extend([0x01, 0x06]); // logic op
    f.extend([0x01, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x04]); // colour targets
    f.extend([0x01, 0x69, 0x40, 0x15]); // depth target, samples, emulation
    f.extend([0x00, 0x41, 0x42]);
    // CreateComputePipeline, refused "x".
    f.extend([0x09, 0x05, 0x20, 0x21, 0x01, 0x01, b'x']);
    // Destroy.
    f.extend([0x0A, 0x04, 0x01, 0x22, 0x23, 0x24]);
    // Submit: one marker, fence 2^34 + 5.
    f.extend([
        0x0B, 0x0A, 0x01, 0x09, 0x25, 0x26, 0x00, 0x85, 0x80, 0x80, 0x80, 0x50,
    ]);
    // Submit, empty and refused "q".
    f.extend([0x0B, 0x04, 0x00, 0x01, 0x01, b'q']);
    // PresentTarget: a window.
    f.extend([0x0C, 0x02, 0x01, 0x00]);
    // ReadbackResult: bytes, unknown, nothing.
    f.extend([0x0D, 0x07, 0x27, 0x28, 0x01, 0x00, 0x02, 0xDE, 0xAD]);
    f.extend([0x0D, 0x04, 0x47, 0x48, 0x01, 0x01]);
    f.extend([0x0D, 0x03, 0x49, 0x4A, 0x00]);
    // PipelineOutcome.
    f.extend([0x0E, 0x03, 0x29, 0x2A, 0x01]);
    // FrameResult with a hash, then without.
    f.extend([0x0F, 0x24, 0x2B, 0x2C, 0x2D, 0x01]);
    f.extend(46..78);
    f.extend([0x0F, 0x04, 0x4B, 0x4C, 0x4D, 0x00]);
    // Limit, then End.
    f.extend([0x10, 0x05, 0x86, 0x80, 0x80, 0x80, 0x60]);
    f.extend([0x00, 0x00]);
    f
}

#[test]
fn every_record_kind_writes_its_golden_bytes() {
    let buffer = Arc::new(CaptureBuffer::new());
    let sink: Arc<dyn CaptureSink> = Arc::<CaptureBuffer>::clone(&buffer);
    let mut writer = CaptureWriter::new(sink, golden_start(), u64::MAX).expect("start");
    for record in golden_records() {
        writer.write(&record).expect("a golden record");
    }
    let expected = golden_file();
    assert_eq!(buffer.bytes(), expected);
}

#[test]
fn the_golden_bytes_read_back_as_every_record_kind() {
    let mut expected = vec![Record::Start(golden_start())];
    expected.extend(golden_records());
    let records = read(&golden_file()).expect("the golden file");
    assert_eq!(records, expected);
    let mut kinds: Vec<RecordKind> = records.iter().map(Record::kind).collect();
    kinds.sort_by_key(|kind| kind.tag());
    kinds.dedup();
    assert_eq!(kinds, RecordKind::ALL, "every kind is in the golden file");
}

/// The variants the main golden file cannot hold beside their siblings: a second `Start` with
/// the other wait setting, an offscreen target and a ready pipeline. Each differs from the main
/// file's bytes in its variant byte only.
#[test]
fn the_other_variants_write_and_read_their_golden_bytes() {
    let start = Start {
        wait: WaitSetting::Wait,
        ..golden_start()
    };
    let records = [
        Record::PresentTarget(TargetRecord {
            target: CapturedTarget::Offscreen,
            result: Created::Made(()),
        }),
        Record::PipelineOutcome(PipelineOutcome {
            pipeline: PipelineId::new(41, 42),
            outcome: PipelineResult::Ready,
        }),
        Record::End,
    ];
    // The main file's header and `Start` (tag, two length bytes, 159 bytes of body), with the
    // wait setting, the byte before the five of the memory limit, now 0 (Wait).
    let mut expected = golden_file();
    let start_end = 5 + 3 + 159;
    expected.truncate(start_end);
    assert_eq!(expected[start_end - 6], 0x01, "Skip in the main file");
    expected[start_end - 6] = 0x00;
    expected.extend([0x0C, 0x02, 0x00, 0x00]); // PresentTarget: offscreen, made
    expected.extend([0x0E, 0x03, 0x29, 0x2A, 0x00]); // PipelineOutcome: ready
    expected.extend([0x00, 0x00]); // End

    let buffer = Arc::new(CaptureBuffer::new());
    let sink: Arc<dyn CaptureSink> = Arc::<CaptureBuffer>::clone(&buffer);
    let mut writer = CaptureWriter::new(sink, start.clone(), u64::MAX).expect("start");
    for record in &records {
        writer.write(record).expect("a golden record");
    }
    assert_eq!(buffer.bytes(), expected);
    let mut all = vec![Record::Start(start)];
    all.extend(records);
    assert_eq!(read(&expected).expect("the golden file"), all);
}
