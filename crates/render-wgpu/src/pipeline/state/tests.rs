use super::*;
use crate::fixtures::{graphics_desc, no_blend};
use starquake_render::pipeline::{
    DepthBias, DepthStencilState, StencilState, VertexBufferLayout, VertexLayout,
};
use starquake_render::resource::CompareFunction;
use starquake_render::{PixelFormat, ShaderId};

/// A change that puts one fault into a description.
type Fault<'a> = &'a dyn Fn(&mut GraphicsPipelineDesc);

fn desc() -> GraphicsPipelineDesc {
    graphics_desc(ShaderId::new(0, 0), Some(ShaderId::new(1, 0)))
}

/// Every combination of components and type, against a table written out by hand from Eden's
/// `VertexFormat` (`maxwell_to_vk.cpp` @ cf654052) and wgpu's format list: `None` where Eden has
/// no format or wgpu has none for Eden's.
#[test]
#[expect(clippy::too_many_lines, reason = "one table written out by hand")]
fn vertex_formats_follow_eden_where_wgpu_has_them() {
    use VertexComponents as C;
    use VertexKind as K;
    use wgpu::VertexFormat as F;
    let kinds = [
        K::Unorm,
        K::Snorm,
        K::Uint,
        K::Sint,
        K::Uscaled,
        K::Sscaled,
        K::Float,
    ];
    // Per row: the format for unorm, snorm, uint, sint, uscaled, sscaled, float.
    let table: [(C, [Option<F>; 7]); 17] = [
        (
            C::R8,
            [
                Some(F::Unorm8),
                Some(F::Snorm8),
                Some(F::Uint8),
                Some(F::Sint8),
                Some(F::Uint8),
                Some(F::Sint8),
                None,
            ],
        ),
        (
            C::A8,
            [
                Some(F::Unorm8),
                Some(F::Snorm8),
                Some(F::Uint8),
                Some(F::Sint8),
                Some(F::Uint8),
                Some(F::Sint8),
                None,
            ],
        ),
        (
            C::R8G8,
            [
                Some(F::Unorm8x2),
                Some(F::Snorm8x2),
                Some(F::Uint8x2),
                Some(F::Sint8x2),
                Some(F::Uint8x2),
                Some(F::Sint8x2),
                None,
            ],
        ),
        (
            C::G8R8,
            [
                Some(F::Unorm8x2),
                Some(F::Snorm8x2),
                Some(F::Uint8x2),
                Some(F::Sint8x2),
                Some(F::Uint8x2),
                Some(F::Sint8x2),
                None,
            ],
        ),
        (C::R8G8B8, [None; 7]),
        (
            C::R8G8B8A8,
            [
                Some(F::Unorm8x4),
                Some(F::Snorm8x4),
                Some(F::Uint8x4),
                Some(F::Sint8x4),
                Some(F::Uint8x4),
                Some(F::Sint8x4),
                None,
            ],
        ),
        (
            C::X8B8G8R8,
            [
                Some(F::Unorm8x4),
                Some(F::Snorm8x4),
                Some(F::Uint8x4),
                Some(F::Sint8x4),
                Some(F::Uint8x4),
                Some(F::Sint8x4),
                None,
            ],
        ),
        (
            C::R16,
            [
                Some(F::Unorm16),
                Some(F::Snorm16),
                Some(F::Uint16),
                Some(F::Sint16),
                Some(F::Uint16),
                Some(F::Sint16),
                Some(F::Float16),
            ],
        ),
        (
            C::R16G16,
            [
                Some(F::Unorm16x2),
                Some(F::Snorm16x2),
                Some(F::Uint16x2),
                Some(F::Sint16x2),
                Some(F::Uint16x2),
                Some(F::Sint16x2),
                Some(F::Float16x2),
            ],
        ),
        (C::R16G16B16, [None; 7]),
        (
            C::R16G16B16A16,
            [
                Some(F::Unorm16x4),
                Some(F::Snorm16x4),
                Some(F::Uint16x4),
                Some(F::Sint16x4),
                Some(F::Uint16x4),
                Some(F::Sint16x4),
                Some(F::Float16x4),
            ],
        ),
        (
            C::R32,
            [
                None,
                None,
                Some(F::Uint32),
                Some(F::Sint32),
                Some(F::Uint32),
                Some(F::Sint32),
                Some(F::Float32),
            ],
        ),
        (
            C::R32G32,
            [
                None,
                None,
                Some(F::Uint32x2),
                Some(F::Sint32x2),
                Some(F::Uint32x2),
                Some(F::Sint32x2),
                Some(F::Float32x2),
            ],
        ),
        (
            C::R32G32B32,
            [
                None,
                None,
                Some(F::Uint32x3),
                Some(F::Sint32x3),
                Some(F::Uint32x3),
                Some(F::Sint32x3),
                Some(F::Float32x3),
            ],
        ),
        (
            C::R32G32B32A32,
            [
                None,
                None,
                Some(F::Uint32x4),
                Some(F::Sint32x4),
                Some(F::Uint32x4),
                Some(F::Sint32x4),
                Some(F::Float32x4),
            ],
        ),
        (
            C::A2B10G10R10,
            [Some(F::Unorm10_10_10_2), None, None, None, None, None, None],
        ),
        (C::B10G11R11, [None; 7]),
    ];
    for (components, row) in table {
        for (kind, expected) in kinds.into_iter().zip(row) {
            assert_eq!(
                vertex_format(components, kind),
                expected,
                "{components:?} {kind:?}"
            );
        }
    }
}

#[test]
fn topologies_map_to_what_the_rewrite_draws() {
    use wgpu::PrimitiveTopology as P;
    let cases = [
        (Topology::Points, Some(P::PointList)),
        (Topology::Lines, Some(P::LineList)),
        (Topology::LineLoop, Some(P::LineStrip)),
        (Topology::LineStrip, Some(P::LineStrip)),
        (Topology::Triangles, Some(P::TriangleList)),
        (Topology::TriangleStrip, Some(P::TriangleStrip)),
        (Topology::TriangleFan, Some(P::TriangleList)),
        (Topology::Quads, Some(P::TriangleList)),
        (Topology::QuadStrip, Some(P::TriangleList)),
        (Topology::Polygon, Some(P::TriangleList)),
        (Topology::LinesAdjacency, None),
        (Topology::LineStripAdjacency, None),
        (Topology::TrianglesAdjacency, None),
        (Topology::TriangleStripAdjacency, None),
        (Topology::Patches, None),
    ];
    for (guest, host) in cases {
        assert_eq!(topology(guest), host, "{guest:?}");
    }
}

/// A strip pipeline names 32-bit indices, and every other pipeline none (module docs,
/// "Topology"): wgpu-core refuses an indexed strip draw whose index format the pipeline does
/// not name, and a strip format on another topology.
#[test]
fn only_strip_pipelines_name_their_index_format() {
    use wgpu::PrimitiveTopology as P;
    let cases = [
        (Topology::Points, P::PointList, None),
        (Topology::Lines, P::LineList, None),
        (Topology::LineLoop, P::LineStrip, Some(wgpu::IndexFormat::Uint32)),
        (Topology::LineStrip, P::LineStrip, Some(wgpu::IndexFormat::Uint32)),
        (Topology::Triangles, P::TriangleList, None),
        (Topology::TriangleStrip, P::TriangleStrip, Some(wgpu::IndexFormat::Uint32)),
        (Topology::TriangleFan, P::TriangleList, None),
        (Topology::Quads, P::TriangleList, None),
        (Topology::QuadStrip, P::TriangleList, None),
        (Topology::Polygon, P::TriangleList, None),
    ];
    for (guest, host, format) in cases {
        let mut d = desc();
        d.topology = guest;
        let primitive = graphics(&d).expect("accepted").primitive;
        assert_eq!(
            (primitive.topology, primitive.strip_index_format),
            (host, format),
            "{guest:?}"
        );
    }
}

#[test]
fn the_default_description_converts() {
    let plan = graphics(&desc()).expect("accepted");
    assert_eq!(
        plan,
        GraphicsPlan {
            buffers: vec![],
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            targets: vec![Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba8Unorm,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }
    );
}

#[test]
fn raster_state_converts_field_by_field() {
    let mut d = desc();
    d.raster.cull = CullMode::Back;
    d.raster.front_face = FrontFace::Cw;
    d.raster.polygon_mode = PolygonMode::Line;
    d.raster.depth_clamp = true;
    d.targets.sample_count = 4;
    let p = graphics(&d).expect("accepted");
    assert_eq!(p.primitive.cull_mode, Some(wgpu::Face::Back));
    assert_eq!(p.primitive.front_face, wgpu::FrontFace::Cw);
    assert_eq!(p.primitive.polygon_mode, wgpu::PolygonMode::Line);
    assert!(p.primitive.unclipped_depth);
    assert_eq!(p.multisample.count, 4);
    d.raster.cull = CullMode::Front;
    d.raster.polygon_mode = PolygonMode::Point;
    let p = graphics(&d).expect("accepted");
    assert_eq!(p.primitive.cull_mode, Some(wgpu::Face::Front));
    assert_eq!(p.primitive.polygon_mode, wgpu::PolygonMode::Point);
}

#[test]
fn attributes_are_grouped_by_buffer_in_order() {
    let mut d = desc();
    d.vertex_input = VertexLayout {
        buffers: vec![
            VertexBufferLayout {
                stride: 16,
                step: VertexStepMode::Vertex,
            },
            VertexBufferLayout {
                stride: 8,
                step: VertexStepMode::Instance,
            },
        ],
        attributes: vec![
            VertexAttribute {
                location: 3,
                buffer: 1,
                offset: 4,
                components: VertexComponents::R16G16,
                kind: VertexKind::Unorm,
            },
            VertexAttribute {
                location: 0,
                buffer: 0,
                offset: 0,
                components: VertexComponents::R32G32B32,
                kind: VertexKind::Float,
            },
            VertexAttribute {
                location: 1,
                buffer: 1,
                offset: 0,
                components: VertexComponents::R8G8B8A8,
                kind: VertexKind::Uint,
            },
        ],
    };
    let p = graphics(&d).expect("accepted");
    assert_eq!(
        p.buffers,
        vec![
            BufferPlan {
                stride: 16,
                step: wgpu::VertexStepMode::Vertex,
                attributes: vec![wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: 0,
                    shader_location: 0,
                }],
            },
            BufferPlan {
                stride: 8,
                step: wgpu::VertexStepMode::Instance,
                attributes: vec![
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Unorm16x2,
                        offset: 4,
                        shader_location: 3,
                    },
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Uint8x4,
                        offset: 0,
                        shader_location: 1,
                    },
                ],
            },
        ]
    );
}

#[test]
fn an_attribute_past_the_buffers_is_refused() {
    // The shared checks refuse it first; this is the backend's own guard.
    let mut d = desc();
    d.vertex_input.attributes.push(VertexAttribute {
        location: 0,
        buffer: 0,
        offset: 0,
        components: VertexComponents::R32,
        kind: VertexKind::Float,
    });
    assert_eq!(
        graphics(&d),
        Err(RenderError::NotBelow {
            what: "vertex attribute buffer",
            index: 0,
            count: 0
        })
    );
}

#[test]
fn blending_converts_and_min_max_take_factors_of_one() {
    let mut d = desc();
    let component = |src, dst, op| BlendComponent { src, dst, op };
    d.blend[0] = BlendState {
        enabled: true,
        color: component(
            BlendFactor::SrcAlpha,
            BlendFactor::OneMinusSrcAlpha,
            BlendOperation::Add,
        ),
        alpha: component(BlendFactor::Constant, BlendFactor::Dst, BlendOperation::Max),
        write_mask: ColorWrites::RED.union(ColorWrites::ALPHA),
    };
    let target = graphics(&d).expect("accepted").targets[0]
        .clone()
        .expect("target 0");
    assert_eq!(
        target.blend,
        Some(wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::SrcAlpha,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Max,
            },
        })
    );
    assert_eq!(
        target.write_mask,
        wgpu::ColorWrites::RED | wgpu::ColorWrites::ALPHA
    );
}

#[test]
fn every_blend_factor_and_operation_maps_by_name() {
    use wgpu::BlendFactor as W;
    let factors = [
        (BlendFactor::Zero, W::Zero),
        (BlendFactor::One, W::One),
        (BlendFactor::Src, W::Src),
        (BlendFactor::OneMinusSrc, W::OneMinusSrc),
        (BlendFactor::SrcAlpha, W::SrcAlpha),
        (BlendFactor::OneMinusSrcAlpha, W::OneMinusSrcAlpha),
        (BlendFactor::Dst, W::Dst),
        (BlendFactor::OneMinusDst, W::OneMinusDst),
        (BlendFactor::DstAlpha, W::DstAlpha),
        (BlendFactor::OneMinusDstAlpha, W::OneMinusDstAlpha),
        (BlendFactor::SrcAlphaSaturated, W::SrcAlphaSaturated),
        (BlendFactor::Constant, W::Constant),
        (BlendFactor::OneMinusConstant, W::OneMinusConstant),
        (BlendFactor::Src1, W::Src1),
        (BlendFactor::OneMinusSrc1, W::OneMinusSrc1),
        (BlendFactor::Src1Alpha, W::Src1Alpha),
        (BlendFactor::OneMinusSrc1Alpha, W::OneMinusSrc1Alpha),
    ];
    for (ours, theirs) in factors {
        // As the source factor, with the destination factor distinct.
        let c = blend_component(BlendComponent {
            src: ours,
            dst: BlendFactor::OneMinusConstant,
            op: BlendOperation::ReverseSubtract,
        });
        assert_eq!(c.src_factor, theirs, "{ours:?} as source");
        assert_eq!(c.dst_factor, W::OneMinusConstant);
        assert_eq!(c.operation, wgpu::BlendOperation::ReverseSubtract);
        let c = blend_component(BlendComponent {
            src: BlendFactor::Constant,
            dst: ours,
            op: BlendOperation::Subtract,
        });
        assert_eq!(c.dst_factor, theirs, "{ours:?} as destination");
        assert_eq!(c.operation, wgpu::BlendOperation::Subtract);
    }
    for op in [BlendOperation::Min, BlendOperation::Max] {
        let c = blend_component(BlendComponent {
            src: BlendFactor::Zero,
            dst: BlendFactor::SrcAlpha,
            op,
        });
        assert_eq!((c.src_factor, c.dst_factor), (W::One, W::One), "{op:?}");
    }
    assert_eq!(
        blend_component(BlendComponent {
            src: BlendFactor::One,
            dst: BlendFactor::One,
            op: BlendOperation::Min
        })
        .operation,
        wgpu::BlendOperation::Min
    );
}

#[test]
fn depth_and_stencil_convert() {
    let mut d = desc();
    // A format with both aspects, so neither part is dropped (`depth_and_stencil_state_follow_
    // the_formats_aspects` covers the one-aspect formats).
    d.targets.depth_stencil = Some(PixelFormat::D32FloatS8Uint);
    // No depth test: always passes, never writes, even with the write flag set.
    d.depth_stencil.depth_write = true;
    let ds = graphics(&d)
        .expect("accepted")
        .depth_stencil
        .expect("depth state");
    assert_eq!(ds.format, wgpu::TextureFormat::Depth32FloatStencil8);
    assert_eq!(ds.depth_compare, Some(wgpu::CompareFunction::Always));
    assert_eq!(ds.depth_write_enabled, Some(false));
    assert_eq!(ds.stencil, wgpu::StencilState::default());

    d.depth_stencil.depth_test = Some(CompareFunction::GreaterEqual);
    d.depth_stencil.stencil = Some(StencilState {
        front: StencilFace {
            compare: CompareFunction::Equal,
            fail: StencilOperation::Zero,
            depth_fail: StencilOperation::IncrementWrap,
            pass: StencilOperation::Replace,
        },
        back: StencilFace {
            compare: CompareFunction::NotEqual,
            fail: StencilOperation::Invert,
            depth_fail: StencilOperation::DecrementClamp,
            pass: StencilOperation::IncrementClamp,
        },
        read_mask: 0x0f,
        write_mask: 0xf0,
    });
    d.raster.depth_bias = DepthBias {
        constant: 2.6,
        slope_scale: 1.5,
        clamp: 0.25,
    };
    let ds = graphics(&d)
        .expect("accepted")
        .depth_stencil
        .expect("depth state");
    assert_eq!(ds.depth_compare, Some(wgpu::CompareFunction::GreaterEqual));
    assert_eq!(ds.depth_write_enabled, Some(true));
    assert_eq!(
        ds.stencil,
        wgpu::StencilState {
            front: wgpu::StencilFaceState {
                compare: wgpu::CompareFunction::Equal,
                fail_op: wgpu::StencilOperation::Zero,
                depth_fail_op: wgpu::StencilOperation::IncrementWrap,
                pass_op: wgpu::StencilOperation::Replace,
            },
            back: wgpu::StencilFaceState {
                compare: wgpu::CompareFunction::NotEqual,
                fail_op: wgpu::StencilOperation::Invert,
                depth_fail_op: wgpu::StencilOperation::DecrementClamp,
                pass_op: wgpu::StencilOperation::IncrementClamp,
            },
            read_mask: 0x0f,
            write_mask: 0xf0,
        }
    );
    assert_eq!(
        ds.bias,
        wgpu::DepthBiasState {
            constant: 3,
            slope_scale: 1.5,
            clamp: 0.25
        }
    );
    // The remaining stencil operations.
    assert_eq!(
        stencil_operation(StencilOperation::Keep),
        wgpu::StencilOperation::Keep
    );
    assert_eq!(
        stencil_operation(StencilOperation::DecrementWrap),
        wgpu::StencilOperation::DecrementWrap
    );
}

/// A stencil state whose every field differs from wgpu's default, in our terms and wgpu's.
fn stencil_pair() -> (StencilState, wgpu::StencilState) {
    let ours = StencilState {
        front: StencilFace {
            compare: CompareFunction::Equal,
            fail: StencilOperation::Zero,
            depth_fail: StencilOperation::IncrementWrap,
            pass: StencilOperation::Replace,
        },
        back: StencilFace {
            compare: CompareFunction::NotEqual,
            fail: StencilOperation::Invert,
            depth_fail: StencilOperation::DecrementClamp,
            pass: StencilOperation::IncrementClamp,
        },
        read_mask: 0x0f,
        write_mask: 0xf0,
    };
    let theirs = wgpu::StencilState {
        front: wgpu::StencilFaceState {
            compare: wgpu::CompareFunction::Equal,
            fail_op: wgpu::StencilOperation::Zero,
            depth_fail_op: wgpu::StencilOperation::IncrementWrap,
            pass_op: wgpu::StencilOperation::Replace,
        },
        back: wgpu::StencilFaceState {
            compare: wgpu::CompareFunction::NotEqual,
            fail_op: wgpu::StencilOperation::Invert,
            depth_fail_op: wgpu::StencilOperation::DecrementClamp,
            pass_op: wgpu::StencilOperation::IncrementClamp,
        },
        read_mask: 0x0f,
        write_mask: 0xf0,
    };
    (ours, theirs)
}

/// A depth test, a depth write and a stencil test on each depth/stencil format wgpu has: the
/// state of an aspect the format lacks is dropped, so wgpu-core's `FormatNotDepth` and
/// `FormatNotStencil` checks (`DepthStencilState::is_depth_enabled`,
/// `StencilState::is_enabled`) pass. Which format has which aspect is written out by hand.
#[test]
fn depth_and_stencil_state_follow_the_formats_aspects() {
    let (stencil, host_stencil) = stencil_pair();
    // Format, whether it has depth, whether it has stencil.
    let cases = [
        (PixelFormat::D16Unorm, true, false),
        (PixelFormat::D32Float, true, false),
        (PixelFormat::S8Uint, false, true),
        (PixelFormat::D32FloatS8Uint, true, true),
    ];
    for (format, depth, has_stencil) in cases {
        let mut d = desc();
        d.targets.depth_stencil = Some(format);
        d.depth_stencil = DepthStencilState {
            depth_test: Some(CompareFunction::Less),
            depth_write: true,
            stencil: Some(stencil),
        };
        let ds = graphics(&d)
            .expect("accepted")
            .depth_stencil
            .expect("depth state");
        let (compare, write) = if depth {
            (wgpu::CompareFunction::Less, true)
        } else {
            (wgpu::CompareFunction::Always, false)
        };
        assert_eq!(ds.depth_compare, Some(compare), "{format:?}");
        assert_eq!(ds.depth_write_enabled, Some(write), "{format:?}");
        assert_eq!(ds.is_depth_enabled(), depth, "{format:?}");
        let expected = if has_stencil {
            host_stencil.clone()
        } else {
            wgpu::StencilState::default()
        };
        assert_eq!(ds.stencil, expected, "{format:?}");
        assert_eq!(ds.stencil.is_enabled(), has_stencil, "{format:?}");
    }
}

/// Each bias term alone, and all three, on every topology wgpu draws: kept on triangles and
/// dropped elsewhere, so wgpu-core's `DepthBiasWithIncompatibleTopology` check passes. Which
/// guest topology is drawn as triangles is written out by hand.
#[test]
fn depth_bias_is_dropped_off_triangles() {
    let biases = [
        DepthBias {
            constant: 2.0,
            slope_scale: 0.0,
            clamp: 0.0,
        },
        DepthBias {
            constant: 0.0,
            slope_scale: 1.5,
            clamp: 0.0,
        },
        DepthBias {
            constant: 0.0,
            slope_scale: 0.0,
            clamp: 0.25,
        },
        DepthBias {
            constant: 2.0,
            slope_scale: 1.5,
            clamp: 0.25,
        },
    ];
    let cases = [
        (Topology::Points, false),
        (Topology::Lines, false),
        (Topology::LineLoop, false),
        (Topology::LineStrip, false),
        (Topology::Triangles, true),
        (Topology::TriangleStrip, true),
        (Topology::TriangleFan, true),
        (Topology::Quads, true),
        (Topology::QuadStrip, true),
        (Topology::Polygon, true),
    ];
    for bias in biases {
        for (guest, triangles) in cases {
            let mut d = desc();
            d.targets.depth_stencil = Some(PixelFormat::D32Float);
            d.topology = guest;
            d.raster.depth_bias = bias;
            let ds = graphics(&d)
                .expect("accepted")
                .depth_stencil
                .expect("depth state");
            let expected = if triangles {
                wgpu::DepthBiasState {
                    constant: bias_constant(bias.constant),
                    slope_scale: bias.slope_scale,
                    clamp: bias.clamp,
                }
            } else {
                wgpu::DepthBiasState::default()
            };
            assert_eq!(ds.bias, expected, "{guest:?} {bias:?}");
        }
    }
}

#[test]
fn depth_bias_constants_round_and_saturate() {
    assert_eq!(bias_constant(0.0), 0);
    assert_eq!(bias_constant(2.5), 3);
    assert_eq!(bias_constant(-2.5), -3);
    assert_eq!(bias_constant(-0.4), 0);
    assert_eq!(bias_constant(1e20), i32::MAX);
    assert_eq!(bias_constant(-1e20), i32::MIN);
}

#[test]
fn targets_stop_at_the_last_one_present() {
    let mut d = desc();
    d.targets.colors[0] = None;
    d.targets.colors[2] = Some(PixelFormat::R32Float);
    d.blend[2] = no_blend();
    let targets = graphics(&d).expect("accepted").targets;
    assert_eq!(targets.len(), 3);
    assert_eq!(targets[0], None);
    assert_eq!(targets[1], None);
    assert_eq!(
        targets[2].as_ref().map(|t| t.format),
        Some(wgpu::TextureFormat::R32Float)
    );
    // No colour targets and no fragment shader: a depth-only pipeline.
    let mut depth_only = desc();
    depth_only.stages.fragment = None;
    depth_only.targets.colors = [None; starquake_render::caps::GUEST_COLOR_TARGETS];
    assert_eq!(graphics(&depth_only).map(|p| p.targets), Ok(vec![]));
}

/// One refusal per check, each alone, then the documented order with two faults at once.
#[test]
fn refusals_are_named_and_found_in_order() {
    let bad_attribute = VertexAttribute {
        location: 0,
        buffer: 0,
        offset: 0,
        components: VertexComponents::R8G8B8,
        kind: VertexKind::Unorm,
    };
    let with_attribute = |d: &mut GraphicsPipelineDesc| {
        d.vertex_input = VertexLayout {
            buffers: vec![VertexBufferLayout {
                stride: 4,
                step: VertexStepMode::Vertex,
            }],
            attributes: vec![bad_attribute],
        };
    };
    let adjacency = |d: &mut GraphicsPipelineDesc| d.topology = Topology::TrianglesAdjacency;
    let cull = |d: &mut GraphicsPipelineDesc| d.raster.cull = CullMode::FrontAndBack;
    let no_fragment = |d: &mut GraphicsPipelineDesc| d.stages.fragment = None;
    let target = |d: &mut GraphicsPipelineDesc| {
        d.targets.colors[1] = Some(PixelFormat::R5G6B5Unorm);
    };
    let depth = |d: &mut GraphicsPipelineDesc| {
        d.targets.depth_stencil = Some(PixelFormat::S8UintD24Unorm);
    };
    let faults: [(Fault<'_>, &str); 6] = [
        (&with_attribute, VERTEX_FORMAT),
        (&adjacency, ADJACENCY),
        (&cull, CULL_BOTH),
        (&no_fragment, TARGETS_WITHOUT_FRAGMENT),
        (&target, TARGET_FORMAT),
        (&depth, TARGET_FORMAT),
    ];
    for (i, (fault, message)) in faults.iter().enumerate() {
        let mut d = desc();
        fault(&mut d);
        assert_eq!(
            graphics(&d),
            Err(RenderError::Unsupported(message)),
            "fault {i} alone"
        );
        // With each later fault present too, this one is still the one reported.
        for (later, _) in &faults[i + 1..] {
            let mut d = desc();
            later(&mut d);
            fault(&mut d);
            assert_eq!(
                graphics(&d),
                Err(RenderError::Unsupported(message)),
                "fault {i} first"
            );
        }
    }
}
