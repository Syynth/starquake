//! Tests of the key list's bytes: the golden layout, every refusal reading and writing make,
//! the order of their checks, and generated round trips.

use super::super::key::{
    BiasBits, FixedState, GraphicsKey, KeyFault, KeyField, PipelineKey, StencilKey, StencilMasks,
};
use super::super::list::{KeyList, KeyListError, MAGIC, VERSION};
use super::super::{
    CANONICAL_DEPTH_BIAS, CANONICAL_LINE_WIDTH, CANONICAL_STENCIL_MASK, KEYED_DYNAMIC_STATE,
    MAX_PIPELINES, ShaderStamp, StageStamps,
};
use proptest::prelude::*;
use proptest::test_runner::{Config, RngSeed};
use starquake_render::caps::{DynamicState, GUEST_COLOR_TARGETS};
use starquake_render::pipeline::{
    BlendComponent, BlendFactor, BlendOperation, BlendState, ColorWrites, CullMode, DepthBias,
    DepthStencilState, EmulationFlags, FrontFace, GraphicsPipelineDesc, LogicOp, PolygonMode,
    RasterState, Stages, StencilFace, StencilOperation, StencilState, TargetFormats, Topology,
    VertexAttribute, VertexBufferLayout, VertexComponents, VertexKind, VertexLayout,
    VertexStepMode,
};
use starquake_render::resource::CompareFunction;
use starquake_render::{PixelFormat, RenderError, ShaderId};

/// Every blend factor, in declaration order (`starquake-render`'s `pipeline.rs`).
const FACTORS: [BlendFactor; 17] = [
    BlendFactor::Zero,
    BlendFactor::One,
    BlendFactor::Src,
    BlendFactor::OneMinusSrc,
    BlendFactor::SrcAlpha,
    BlendFactor::OneMinusSrcAlpha,
    BlendFactor::Dst,
    BlendFactor::OneMinusDst,
    BlendFactor::DstAlpha,
    BlendFactor::OneMinusDstAlpha,
    BlendFactor::SrcAlphaSaturated,
    BlendFactor::Constant,
    BlendFactor::OneMinusConstant,
    BlendFactor::Src1,
    BlendFactor::OneMinusSrc1,
    BlendFactor::Src1Alpha,
    BlendFactor::OneMinusSrc1Alpha,
];

/// A stamp whose every byte is distinct and non-zero.
const BIG_STAMP: u128 = 0x0102_0304_0506_0708_090A_0B0C_0D0E_0F10;

/// A compute stamp above 64 bits.
const COMPUTE_STAMP: u128 = (1 << 100) | 0x55;

/// A graphics key with every field present and distinct, built under no dynamic state.
fn full_key() -> GraphicsKey {
    let blend: [BlendState; GUEST_COLOR_TARGETS] = std::array::from_fn(|i| BlendState {
        enabled: true,
        color: BlendComponent {
            src: FACTORS[i + 1],
            dst: FACTORS[i + 2],
            op: BlendOperation::Subtract,
        },
        alpha: BlendComponent {
            src: FACTORS[i + 3],
            dst: FACTORS[i + 9],
            op: BlendOperation::Max,
        },
        write_mask: ColorWrites::from_bits(u32::try_from(i).unwrap() + 1).unwrap(),
    });
    let mut colors = [None; GUEST_COLOR_TARGETS];
    colors[0] = Some(PixelFormat::R8Unorm);
    colors[2] = Some(PixelFormat::R32Float);
    GraphicsKey {
        stamps: StageStamps {
            vertex: ShaderStamp::new(BIG_STAMP),
            tess_control: Some(ShaderStamp::new(0x21)),
            tess_eval: Some(ShaderStamp::new(0x22)),
            geometry: Some(ShaderStamp::new(0x23)),
            fragment: Some(ShaderStamp::new(0x24)),
        },
        fixed: FixedState {
            dynamic: DynamicState::EMPTY,
            vertex_input: VertexLayout {
                buffers: vec![
                    VertexBufferLayout {
                        stride: 0x31,
                        step: VertexStepMode::Instance,
                    },
                    VertexBufferLayout {
                        stride: 300,
                        step: VertexStepMode::Vertex,
                    },
                ],
                attributes: vec![VertexAttribute {
                    location: 3,
                    buffer: 1,
                    offset: 5,
                    components: VertexComponents::R16G16B16A16,
                    kind: VertexKind::Uint,
                }],
            },
            topology: Topology::Patches,
            primitive_restart: true,
            cull: CullMode::Back,
            front_face: FrontFace::Cw,
            polygon_mode: PolygonMode::Line,
            depth_clamp: true,
            depth_bias: Some(BiasBits {
                constant: 2.0f32.to_bits(),
                slope_scale: 0.5f32.to_bits(),
                clamp: 4.0f32.to_bits(),
            }),
            line_width: Some(1.5f32.to_bits()),
            depth_test: Some(CompareFunction::Greater),
            depth_write: true,
            stencil: Some(StencilKey {
                front: StencilFace {
                    compare: CompareFunction::Equal,
                    fail: StencilOperation::Zero,
                    depth_fail: StencilOperation::Replace,
                    pass: StencilOperation::IncrementClamp,
                },
                back: StencilFace {
                    compare: CompareFunction::NotEqual,
                    fail: StencilOperation::Invert,
                    depth_fail: StencilOperation::IncrementWrap,
                    pass: StencilOperation::DecrementWrap,
                },
                masks: Some(StencilMasks {
                    read: 0x0F,
                    write: 0xF0,
                }),
            }),
            blend,
            logic_op: Some(LogicOp::Xor),
            targets: TargetFormats {
                colors,
                depth_stencil: Some(PixelFormat::D24UnormS8Uint),
                sample_count: 4,
            },
            emulation: EmulationFlags::ALL,
        },
    }
}

/// A graphics key with as little as a key can have, built with every keyed state dynamic.
fn bare_key() -> GraphicsKey {
    let off = BlendComponent {
        src: BlendFactor::One,
        dst: BlendFactor::Zero,
        op: BlendOperation::Add,
    };
    GraphicsKey {
        stamps: StageStamps {
            vertex: ShaderStamp::new(0x41),
            tess_control: None,
            tess_eval: None,
            geometry: None,
            fragment: None,
        },
        fixed: FixedState {
            dynamic: KEYED_DYNAMIC_STATE,
            vertex_input: VertexLayout::default(),
            topology: Topology::Triangles,
            primitive_restart: false,
            cull: CullMode::None,
            front_face: FrontFace::Ccw,
            polygon_mode: PolygonMode::Fill,
            depth_clamp: false,
            depth_bias: None,
            line_width: None,
            depth_test: None,
            depth_write: false,
            stencil: Some(StencilKey {
                front: StencilFace {
                    compare: CompareFunction::Always,
                    fail: StencilOperation::DecrementWrap,
                    depth_fail: StencilOperation::DecrementWrap,
                    pass: StencilOperation::DecrementWrap,
                },
                back: StencilFace {
                    compare: CompareFunction::Never,
                    fail: StencilOperation::Keep,
                    depth_fail: StencilOperation::Keep,
                    pass: StencilOperation::Keep,
                },
                masks: None,
            }),
            blend: [BlendState {
                enabled: false,
                color: off,
                alpha: off,
                write_mask: ColorWrites::ALL,
            }; GUEST_COLOR_TARGETS],
            logic_op: None,
            targets: TargetFormats {
                colors: [None; GUEST_COLOR_TARGETS],
                depth_stencil: None,
                sample_count: 1,
            },
            emulation: EmulationFlags::EMPTY,
        },
    }
}

fn graphics(key: GraphicsKey) -> PipelineKey {
    PipelineKey::Graphics(Box::new(key))
}

fn compute(stamp: u128) -> PipelineKey {
    PipelineKey::Compute(ShaderStamp::new(stamp))
}

fn list(keys: Vec<PipelineKey>) -> KeyList {
    KeyList { keys }
}

/// The bytes of [`full_key`], [`compute`] of [`COMPUTE_STAMP`] and [`bare_key`], in that order,
/// written out by hand from the layout in `list.rs`'s docs.
fn golden() -> Vec<u8> {
    let mut b = vec![
        b'S', b'Q', b'P', b'K', // magic
        0x01, // version 1
        0x03, // three keys
        // Key 0: graphics.
        0x00, // variant: graphics
        // Stamps: the vertex stamp as a 128-bit varint, then four optional stamps.
        0x90, 0x9E, 0xB8, 0xE8, 0xC0, 0xE1, 0x82, 0x85, 0x89, 0x90, 0x9C, 0xB0, 0xD0, 0x80, 0xC1,
        0x81, 0x82, 0x02, //
        0x01, 0x21, 0x01, 0x22, 0x01, 0x23, 0x01, 0x24, //
        0x00, // dynamic-state mask: none
        // Vertex input: two buffers (stride, step), one attribute.
        0x02, 0x31, 0x01, 0xAC, 0x02, 0x00, //
        0x01, 0x03, 0x01, 0x05, 0x02, 0x03, //
        0x0E, // topology: patches
        0x01, // primitive restart
        0x02, // cull: back
        0x01, // front face: clockwise
        0x01, // polygon mode: line
        0x01, // depth clamp
        // Depth bias: 2.0, 0.5, 4.0 as bit patterns.
        0x01, 0x80, 0x80, 0x80, 0x80, 0x04, 0x80, 0x80, 0x80, 0xF8, 0x03, 0x80, 0x80, 0x80, 0x84,
        0x04, //
        0x01, 0x80, 0x80, 0x80, 0xFE, 0x03, // line width 1.5
        0x01, 0x04, // depth test: greater
        0x01, // depth write
        // Stencil: front (equal, zero, replace, increment-clamp), back (not-equal, invert,
        // increment-wrap, decrement-wrap), masks 0x0F and 0xF0.
        0x01, 0x02, 0x01, 0x02, 0x03, 0x05, 0x05, 0x06, 0x07, 0x01, 0x0F, 0xF0, 0x01,
    ];
    // Eight blend states: enabled, colour (src, dst, subtract), alpha (src, dst, max), mask.
    for i in 0..8u8 {
        b.extend([0x01, i + 1, i + 2, 0x01, i + 3, i + 9, 0x04, i + 1]);
    }
    b.extend([
        0x01, 0x06, // logic op: xor
        // Colour formats: R8Unorm (12), none, R32Float (39), five none.
        0x01, 0x0C, 0x00, 0x01, 0x27, 0x00, 0x00, 0x00, 0x00, 0x00, //
        0x01, 0x6D, // depth/stencil: D24UnormS8Uint (109)
        0x04, // four samples
        0x3F, // every emulation flag
        // Key 1: compute, its stamp as a 128-bit varint.
        0x01, 0xD5, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        0x04, //
        // Key 2: graphics.
        0x00, 0x41, 0x00, 0x00, 0x00, 0x00, // vertex stamp only
        0x70, // dynamic-state mask: stencil masks, depth bias, line width
        0x00, 0x00, // no buffers, no attributes
        0x04, // topology: triangles
        0x00, 0x00, 0x00, 0x00, 0x00, // restart, cull, front face, polygon mode, clamp
        0x00, 0x00, // no depth bias, no line width (dynamic)
        0x00, 0x00, // no depth test, no depth write
        // Stencil: front (always, decrement-wrap x3), back (never, keep x3), masks dynamic.
        0x01, 0x07, 0x07, 0x07, 0x07, 0x00, 0x00, 0x00, 0x00, 0x00,
    ]);
    for _ in 0..8 {
        b.extend([0x00, 0x01, 0x00, 0x00, 0x01, 0x00, 0x00, 0x0F]);
    }
    b.extend([
        0x00, // no logic op
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // no colour targets
        0x00, // no depth target
        0x01, // one sample
        0x00, // no emulation
    ]);
    b
}

fn golden_list() -> KeyList {
    list(vec![
        graphics(full_key()),
        compute(COMPUTE_STAMP),
        graphics(bare_key()),
    ])
}

#[test]
fn golden_bytes_of_every_key_kind() {
    assert_eq!(golden_list().to_bytes().unwrap(), golden());
    assert_eq!(KeyList::from_bytes(&golden()).unwrap(), golden_list());
    // The format indices the bytes use are the render crate's.
    assert_eq!(
        [
            PixelFormat::R8Unorm.index(),
            PixelFormat::R32Float.index(),
            PixelFormat::D24UnormS8Uint.index()
        ],
        [12, 39, 109]
    );
}

#[test]
fn the_header_constants_are_the_documented_ones() {
    assert_eq!(&MAGIC, b"SQPK");
    assert_eq!(VERSION, 1);
}

/// The magic, a version and a key count, each varint one byte.
fn header(version: u8, count: &[u8]) -> Vec<u8> {
    let mut b = MAGIC.to_vec();
    b.push(version);
    b.extend(count);
    b
}

/// [`MAX_PIPELINES`] (65,536) and one more, as varints.
const MAX_VARINT: [u8; 3] = [0x80, 0x80, 0x04];
const PAST_MAX_VARINT: [u8; 3] = [0x81, 0x80, 0x04];

#[test]
fn numbers_written_longer_than_needed_are_read() {
    // Version 1 and a count of 1 in two bytes each, then a compute key with stamp 2.
    let mut b = MAGIC.to_vec();
    b.extend([0x81, 0x00, 0x81, 0x00, 0x01, 0x02]);
    assert_eq!(KeyList::from_bytes(&b), Ok(list(vec![compute(2)])));
}

#[test]
fn reading_refuses_another_kind_of_file() {
    assert_eq!(
        KeyList::from_bytes(b"SQPX\x01\x00"),
        Err(KeyListError::BadMagic)
    );
    assert_eq!(KeyList::from_bytes(b"XQ"), Err(KeyListError::BadMagic));
    // The start of the magic is a list cut short.
    assert_eq!(KeyList::from_bytes(b"SQ"), Err(KeyListError::Truncated));
    assert_eq!(KeyList::from_bytes(b""), Err(KeyListError::Truncated));
    // The magic is checked before the version.
    assert_eq!(
        KeyList::from_bytes(b"SQPX\x07\x00"),
        Err(KeyListError::BadMagic)
    );
    assert_eq!(
        KeyListError::BadMagic.to_string(),
        "not a pipeline key list (bad magic)"
    );
    // A good list with one byte of its magic changed.
    let mut b = golden();
    b[3] = b'C';
    assert_eq!(KeyList::from_bytes(&b), Err(KeyListError::BadMagic));
}

#[test]
fn reading_refuses_another_version_before_the_body() {
    let stale = VERSION - 1;
    let newer = VERSION + 1;
    for version in [stale, newer] {
        let byte = u8::try_from(version).unwrap();
        // Garbage after the version, and a key count past the limit: the version is reported.
        let mut b = header(byte, &PAST_MAX_VARINT);
        b.extend([0xFF; 8]);
        assert_eq!(
            KeyList::from_bytes(&b),
            Err(KeyListError::UnsupportedVersion { found: version })
        );
    }
    assert_eq!(
        KeyListError::UnsupportedVersion { found: stale }.to_string(),
        "pipeline key list layout version 0 is not supported (this build reads 1)"
    );
    // The golden list with its version changed.
    let mut b = golden();
    b[4] = 0x02;
    assert_eq!(
        KeyList::from_bytes(&b),
        Err(KeyListError::UnsupportedVersion { found: 2 })
    );
    assert_eq!(KeyList::from_bytes(&MAGIC), Err(KeyListError::Truncated));
}

#[test]
fn the_key_count_is_bounded_before_any_key_is_read() {
    let past = u64::try_from(MAX_PIPELINES).unwrap() + 1;
    assert_eq!(
        KeyList::from_bytes(&header(1, &PAST_MAX_VARINT)),
        Err(KeyListError::TooManyKeys { count: past })
    );
    // At the limit the count passes and the missing keys are what is wrong.
    assert_eq!(
        KeyList::from_bytes(&header(1, &MAX_VARINT)),
        Err(KeyListError::Truncated)
    );
    assert_eq!(
        KeyListError::TooManyKeys { count: past }.to_string(),
        "the key list has 65537 keys, more than the 65536 allowed"
    );
    assert_eq!(
        KeyList::from_bytes(&header(1, &[])),
        Err(KeyListError::Truncated)
    );
}

#[test]
fn a_list_of_exactly_the_limit_is_written_and_read_and_one_more_is_not() {
    let stamps = 0..u128::try_from(MAX_PIPELINES).unwrap();
    let mut keys: Vec<PipelineKey> = stamps.map(compute).collect();
    let at_limit = list(keys.clone());
    let bytes = at_limit.to_bytes().unwrap();
    assert_eq!(bytes[5..8], MAX_VARINT);
    assert_eq!(KeyList::from_bytes(&bytes).unwrap().len(), MAX_PIPELINES);

    keys.push(compute(u128::MAX));
    let past = u64::try_from(MAX_PIPELINES).unwrap() + 1;
    assert_eq!(
        list(keys).to_bytes(),
        Err(KeyListError::TooManyKeys { count: past })
    );
}

#[test]
fn a_list_cut_short_or_with_bytes_after_it_is_refused() {
    let b = golden();
    for len in [6, 7, 30, b.len() - 1] {
        assert_eq!(
            KeyList::from_bytes(&b[..len]),
            Err(KeyListError::Truncated),
            "{len}"
        );
    }
    let mut long = b.clone();
    long.extend([0x00, 0x00]);
    assert_eq!(
        KeyList::from_bytes(&long),
        Err(KeyListError::TrailingBytes { count: 2 })
    );
    assert_eq!(
        KeyListError::TrailingBytes { count: 2 }.to_string(),
        "2 trailing bytes after the key list"
    );
    // A variant index that names no key kind.
    let mut b = header(1, &[1]);
    b.push(0x02);
    assert!(matches!(
        KeyList::from_bytes(&b),
        Err(KeyListError::Decode(_))
    ));
}

#[test]
fn a_repeated_key_is_refused_reading_and_writing() {
    let mut b = header(1, &[4]);
    b.extend([0x01, 0x07, 0x01, 0x08, 0x01, 0x09, 0x01, 0x08]);
    assert_eq!(
        KeyList::from_bytes(&b),
        Err(KeyListError::DuplicateKey { index: 3 })
    );
    let repeated = list(vec![compute(7), compute(8), compute(9), compute(8)]);
    assert_eq!(
        repeated.to_bytes(),
        Err(KeyListError::DuplicateKey { index: 3 })
    );
    assert_eq!(
        KeyListError::DuplicateKey { index: 3 }.to_string(),
        "key 3 of the list repeats an earlier key"
    );
    // Two graphics keys that differ only in their dynamic-state mask and stencil masks are two
    // keys.
    let mut a = full_key();
    a.fixed.topology = Topology::Patches;
    let mut b = a.clone();
    b.fixed.dynamic = DynamicState::STENCIL_MASKS;
    if let Some(s) = b.fixed.stencil.as_mut() {
        s.masks = None;
    }
    let two = list(vec![graphics(a), graphics(b)]);
    assert_eq!(KeyList::from_bytes(&two.to_bytes().unwrap()), Ok(two));
}

/// A key broken in one way, and the fault it is refused with.
fn faults() -> Vec<(&'static str, GraphicsKey, KeyFault)> {
    let mut cases = Vec::new();
    let mut k = full_key();
    k.fixed.dynamic = DynamicState::VIEWPORT;
    cases.push((
        "unkeyed state",
        k,
        KeyFault::UnkeyedDynamicState {
            bits: DynamicState::VIEWPORT.bits(),
        },
    ));
    let mut k = full_key();
    k.fixed.dynamic = DynamicState::DEPTH_BIAS;
    cases.push((
        "bias stored",
        k,
        KeyFault::DynamicFieldStored(KeyField::DepthBias),
    ));
    let mut k = full_key();
    k.fixed.depth_bias = None;
    cases.push((
        "bias missing",
        k,
        KeyFault::FixedFieldMissing(KeyField::DepthBias),
    ));
    let mut k = full_key();
    k.fixed.dynamic = DynamicState::LINE_WIDTH;
    cases.push((
        "width stored",
        k,
        KeyFault::DynamicFieldStored(KeyField::LineWidth),
    ));
    let mut k = full_key();
    k.fixed.line_width = None;
    cases.push((
        "width missing",
        k,
        KeyFault::FixedFieldMissing(KeyField::LineWidth),
    ));
    let mut k = full_key();
    k.fixed.dynamic = DynamicState::STENCIL_MASKS;
    cases.push((
        "masks stored",
        k,
        KeyFault::DynamicFieldStored(KeyField::StencilMasks),
    ));
    let mut k = full_key();
    if let Some(s) = k.fixed.stencil.as_mut() {
        s.masks = None;
    }
    cases.push((
        "masks missing",
        k,
        KeyFault::FixedFieldMissing(KeyField::StencilMasks),
    ));
    let mut k = full_key();
    k.fixed.topology = Topology::Triangles;
    cases.push((
        "tessellation without patches",
        k,
        KeyFault::Invalid(RenderError::Invalid(
            "tessellation shaders need the patch topology",
        )),
    ));
    let mut k = full_key();
    k.fixed.line_width = Some(f32::NAN.to_bits());
    cases.push((
        "line width not finite",
        k,
        KeyFault::Invalid(RenderError::Invalid("rasteriser values must be finite")),
    ));
    let mut k = full_key();
    k.fixed.targets.sample_count = 3;
    cases.push((
        "sample count",
        k,
        KeyFault::Invalid(RenderError::SampleCount(3)),
    ));
    let mut k = full_key();
    k.fixed.targets.colors[1] = Some(PixelFormat::D32Float);
    cases.push((
        "depth format as colour",
        k,
        KeyFault::Invalid(RenderError::Invalid(
            "a colour target needs a colour format",
        )),
    ));
    let mut k = full_key();
    k.fixed.vertex_input.attributes[0].location = 32;
    cases.push((
        "attribute location past the guest's",
        k,
        KeyFault::Invalid(RenderError::NotBelow {
            what: "vertex attribute location",
            index: 32,
            count: 32,
        }),
    ));
    cases
}

#[test]
fn a_key_that_fails_its_checks_is_refused_on_save_as_on_load() {
    for (what, key, fault) in faults() {
        let bad = list(vec![compute(1), graphics(key)]);
        let want = KeyListError::InvalidKey { index: 1, fault };
        assert_eq!(bad.to_bytes(), Err(want.clone()), "save: {what}");
        // The same bytes as a good writer would lay them out.
        let mut b = header(1, &[2]);
        b.extend(postcard::to_allocvec(&bad.keys[0]).unwrap());
        b.extend(postcard::to_allocvec(&bad.keys[1]).unwrap());
        assert_eq!(KeyList::from_bytes(&b), Err(want), "load: {what}");
    }
    // At the edge: the guest's last attribute location passes.
    let mut k = full_key();
    k.fixed.vertex_input.attributes[0].location = 31;
    assert!(list(vec![graphics(k)]).to_bytes().is_ok());
}

#[test]
fn a_key_without_a_stencil_test_passes_under_either_stencil_mask_state() {
    for dynamic in [DynamicState::EMPTY, DynamicState::STENCIL_MASKS] {
        let mut k = full_key();
        k.fixed.dynamic = dynamic;
        k.fixed.stencil = None;
        let one = list(vec![graphics(k)]);
        assert_eq!(KeyList::from_bytes(&one.to_bytes().unwrap()), Ok(one));
    }
}

#[test]
fn fault_messages_are_plain_words() {
    let message = |fault| KeyListError::InvalidKey { index: 4, fault }.to_string();
    assert_eq!(
        message(KeyFault::UnkeyedDynamicState { bits: 0x3 }),
        "key 4 of the list is refused: its dynamic-state mask has bits 0x3 for state a pipeline \
         has no field for"
    );
    assert_eq!(
        message(KeyFault::DynamicFieldStored(KeyField::DepthBias)),
        "key 4 of the list is refused: it stores a depth bias although its mask says that is \
         dynamic"
    );
    assert_eq!(
        message(KeyFault::FixedFieldMissing(KeyField::LineWidth)),
        "key 4 of the list is refused: it lacks a line width although its mask says that is not \
         dynamic"
    );
    assert_eq!(
        message(KeyFault::FixedFieldMissing(KeyField::StencilMasks)),
        "key 4 of the list is refused: it lacks a stencil masks although its mask says that is \
         not dynamic"
    );
    assert_eq!(
        message(KeyFault::Invalid(RenderError::SampleCount(3))),
        "key 4 of the list is refused: its pipeline breaks a rule: sample count 3 is not \
         supported by this device"
    );
}

#[test]
fn checks_run_in_the_documented_order() {
    // Within a key: the mask's own bits, then the fields in order, then the description.
    let mut k = full_key();
    k.fixed.dynamic = DynamicState::VIEWPORT | DynamicState::DEPTH_BIAS;
    k.fixed.targets.sample_count = 3;
    assert_eq!(
        super::super::key::check_graphics(&k),
        Err(KeyFault::UnkeyedDynamicState {
            bits: DynamicState::VIEWPORT.bits()
        })
    );
    let mut k = full_key();
    k.fixed.dynamic = DynamicState::LINE_WIDTH | DynamicState::STENCIL_MASKS;
    k.fixed.depth_bias = None;
    k.fixed.targets.sample_count = 3;
    assert_eq!(
        super::super::key::check_graphics(&k),
        Err(KeyFault::FixedFieldMissing(KeyField::DepthBias))
    );
    let mut k = full_key();
    k.fixed.dynamic = DynamicState::STENCIL_MASKS;
    k.fixed.line_width = None;
    assert_eq!(
        super::super::key::check_graphics(&k),
        Err(KeyFault::FixedFieldMissing(KeyField::LineWidth))
    );
    let mut k = full_key();
    k.fixed.dynamic = DynamicState::STENCIL_MASKS;
    k.fixed.targets.sample_count = 3;
    assert_eq!(
        super::super::key::check_graphics(&k),
        Err(KeyFault::DynamicFieldStored(KeyField::StencilMasks))
    );

    // Across a list: the first failing key, before a later duplicate; reading and writing
    // agree.
    let mut bad = full_key();
    bad.fixed.targets.sample_count = 3;
    let mut worse = full_key();
    worse.fixed.targets.sample_count = 5;
    let keys = list(vec![compute(1), graphics(bad), compute(1), graphics(worse)]);
    let want = KeyListError::InvalidKey {
        index: 1,
        fault: KeyFault::Invalid(RenderError::SampleCount(3)),
    };
    assert_eq!(keys.to_bytes(), Err(want.clone()));
    let mut b = header(1, &[4]);
    for key in &keys.keys {
        b.extend(postcard::to_allocvec(key).unwrap());
    }
    assert_eq!(KeyList::from_bytes(&b), Err(want.clone()));
    // A failing key before trailing bytes is reported as the key.
    b.push(0x00);
    assert_eq!(KeyList::from_bytes(&b), Err(want));

    // Trailing bytes come before duplicates.
    let mut b = header(1, &[2]);
    b.extend([0x01, 0x07, 0x01, 0x07, 0x00]);
    assert_eq!(
        KeyList::from_bytes(&b),
        Err(KeyListError::TrailingBytes { count: 1 })
    );
    // On writing, too many keys come before a failing key.
    let mut keys: Vec<PipelineKey> = (0..u128::try_from(MAX_PIPELINES).unwrap())
        .map(compute)
        .collect();
    let mut bad = full_key();
    bad.fixed.targets.sample_count = 3;
    keys[0] = graphics(bad);
    keys.push(compute(u128::MAX));
    assert!(matches!(
        list(keys).to_bytes(),
        Err(KeyListError::TooManyKeys { .. })
    ));
}

/// Every value of the small enums a generated description picks from.
const TOPOLOGIES: [Topology; 14] = [
    Topology::Points,
    Topology::Lines,
    Topology::LineLoop,
    Topology::LineStrip,
    Topology::Triangles,
    Topology::TriangleStrip,
    Topology::TriangleFan,
    Topology::Quads,
    Topology::QuadStrip,
    Topology::Polygon,
    Topology::LinesAdjacency,
    Topology::LineStripAdjacency,
    Topology::TrianglesAdjacency,
    Topology::TriangleStripAdjacency,
];
const COMPARES: [CompareFunction; 8] = [
    CompareFunction::Never,
    CompareFunction::Less,
    CompareFunction::Equal,
    CompareFunction::LessEqual,
    CompareFunction::Greater,
    CompareFunction::NotEqual,
    CompareFunction::GreaterEqual,
    CompareFunction::Always,
];
const OPS: [StencilOperation; 8] = [
    StencilOperation::Keep,
    StencilOperation::Zero,
    StencilOperation::Replace,
    StencilOperation::IncrementClamp,
    StencilOperation::DecrementClamp,
    StencilOperation::Invert,
    StencilOperation::IncrementWrap,
    StencilOperation::DecrementWrap,
];

fn face() -> impl Strategy<Value = StencilFace> {
    (
        prop::sample::select(&COMPARES[..]),
        prop::sample::select(&OPS[..]),
        prop::sample::select(&OPS[..]),
        prop::sample::select(&OPS[..]),
    )
        .prop_map(|(compare, fail, depth_fail, pass)| StencilFace {
            compare,
            fail,
            depth_fail,
            pass,
        })
}

fn blend_state() -> impl Strategy<Value = BlendState> {
    let component = (
        prop::sample::select(&FACTORS[..]),
        prop::sample::select(&FACTORS[..]),
        prop::sample::select(vec![
            BlendOperation::Add,
            BlendOperation::Subtract,
            BlendOperation::ReverseSubtract,
            BlendOperation::Min,
            BlendOperation::Max,
        ]),
    )
        .prop_map(|(src, dst, op)| BlendComponent { src, dst, op });
    (any::<bool>(), component.clone(), component, 0u32..16).prop_map(
        |(enabled, color, alpha, mask)| BlendState {
            enabled,
            color,
            alpha,
            write_mask: ColorWrites::from_bits(mask).unwrap(),
        },
    )
}

fn vertex_layout() -> impl Strategy<Value = VertexLayout> {
    let buffer = (
        any::<u32>(),
        prop::sample::select(vec![VertexStepMode::Vertex, VertexStepMode::Instance]),
    )
        .prop_map(|(stride, step)| VertexBufferLayout { stride, step });
    prop::collection::vec(buffer, 0..4).prop_flat_map(|buffers| {
        let n = u32::try_from(buffers.len()).unwrap();
        let attribute = (
            0..n.max(1),
            any::<u32>(),
            prop::sample::select(vec![
                VertexComponents::R32G32B32A32,
                VertexComponents::R16G16,
                VertexComponents::A8,
            ]),
            prop::sample::select(vec![VertexKind::Snorm, VertexKind::Float, VertexKind::Uint]),
        );
        let count = if buffers.is_empty() { 0..1 } else { 0..5 };
        (Just(buffers), prop::collection::vec(attribute, count)).prop_map(|(buffers, attrs)| {
            VertexLayout {
                buffers,
                attributes: attrs
                    .into_iter()
                    .enumerate()
                    .map(|(i, (buffer, offset, components, kind))| VertexAttribute {
                        // Distinct, below the guest's 32.
                        location: u32::try_from(i * 7).unwrap(),
                        buffer,
                        offset,
                        components,
                        kind,
                    })
                    .collect(),
            }
        })
    })
}

/// Cull mode, clockwise, polygon mode, depth clamp, and the depth bias and line width.
fn raster() -> impl Strategy<Value = (CullMode, bool, PolygonMode, bool, (f32, f32, f32, f32))> {
    let finite = -1.0e6f32..1.0e6f32;
    (
        prop::sample::select(vec![
            CullMode::None,
            CullMode::Front,
            CullMode::Back,
            CullMode::FrontAndBack,
        ]),
        any::<bool>(),
        prop::sample::select(vec![
            PolygonMode::Fill,
            PolygonMode::Line,
            PolygonMode::Point,
        ]),
        any::<bool>(),
        (finite.clone(), finite.clone(), finite.clone(), finite),
    )
}

/// Colour formats, a depth format and a sample count.
fn targets() -> impl Strategy<Value = ([Option<PixelFormat>; 8], Option<PixelFormat>, u32)> {
    (
        prop::array::uniform8(prop::option::of(prop::sample::select(vec![
            PixelFormat::R8Unorm,
            PixelFormat::R32Float,
            PixelFormat::A8B8G8R8Unorm,
        ]))),
        prop::option::of(prop::sample::select(vec![
            PixelFormat::D32Float,
            PixelFormat::D16Unorm,
            PixelFormat::D24UnormS8Uint,
        ])),
        prop::sample::select(vec![1u32, 2, 4, 8, 16]),
    )
}

/// A valid description of any shape, its stamps, and a set of dynamic states.
fn case() -> impl Strategy<Value = (GraphicsPipelineDesc, StageStamps, DynamicState)> {
    let stages = (
        any::<bool>(),
        any::<bool>(),
        any::<bool>(),
        any::<[u128; 5]>(),
    );
    let raster = raster();
    let depth = (
        prop::option::of(prop::sample::select(&COMPARES[..])),
        any::<bool>(),
        prop::option::of((face(), face(), any::<u32>(), any::<u32>())),
    );
    let targets = targets();
    let rest = (
        prop::array::uniform8(blend_state()),
        prop::option::of(prop::sample::select(vec![
            LogicOp::Clear,
            LogicOp::Xor,
            LogicOp::Set,
        ])),
        0..=EmulationFlags::ALL.bits(),
        0u32..8,
        any::<bool>(),
        prop::sample::select(&TOPOLOGIES[..]),
    );
    (stages, vertex_layout(), raster, depth, targets, rest).prop_map(
        |(
            (tess, geometry, fragment, stamps),
            vertex_input,
            (cull, cw, polygon_mode, depth_clamp, (c, s, cl, w)),
            (depth_test, depth_write, stencil),
            (colors, depth_stencil, sample_count),
            (blend, logic_op, emulation, dynamic, primitive_restart, topology),
        )| {
            // Ids and stamps are given distinct values per stage.
            let id = |n| ShaderId::new(n, n + 10);
            let desc = GraphicsPipelineDesc {
                stages: Stages {
                    vertex: id(1),
                    tess_control: tess.then(|| id(2)),
                    tess_eval: tess.then(|| id(3)),
                    geometry: geometry.then(|| id(4)),
                    fragment: fragment.then(|| id(5)),
                },
                vertex_input,
                topology: if tess { Topology::Patches } else { topology },
                primitive_restart,
                raster: RasterState {
                    cull,
                    front_face: if cw { FrontFace::Cw } else { FrontFace::Ccw },
                    polygon_mode,
                    depth_clamp,
                    depth_bias: DepthBias {
                        constant: c,
                        slope_scale: s,
                        clamp: cl,
                    },
                    line_width: w,
                },
                depth_stencil: DepthStencilState {
                    depth_test,
                    depth_write,
                    stencil: stencil.map(|(front, back, read_mask, write_mask)| StencilState {
                        front,
                        back,
                        read_mask,
                        write_mask,
                    }),
                },
                blend,
                logic_op,
                targets: TargetFormats {
                    colors,
                    depth_stencil,
                    sample_count,
                },
                emulation: EmulationFlags::from_bits(emulation).unwrap(),
            };
            let stamp = |i: usize| ShaderStamp::new(stamps[i]);
            let stamps = StageStamps {
                vertex: stamp(0),
                tess_control: tess.then(|| stamp(1)),
                tess_eval: tess.then(|| stamp(2)),
                geometry: geometry.then(|| stamp(3)),
                fragment: fragment.then(|| stamp(4)),
            };
            // Bit n of the three keyed states, in their bit order.
            let dynamic = DynamicState::from_bits(dynamic << 4).unwrap();
            (desc, stamps, dynamic)
        },
    )
}

/// What `desc` should become under `dynamic`: each dynamic field replaced by its canonical
/// value, written out field by field rather than through the key.
fn canonical(desc: &GraphicsPipelineDesc, dynamic: DynamicState) -> GraphicsPipelineDesc {
    let mut d = desc.clone();
    if dynamic.contains(DynamicState::DEPTH_BIAS) {
        d.raster.depth_bias = CANONICAL_DEPTH_BIAS;
    }
    if dynamic.contains(DynamicState::LINE_WIDTH) {
        d.raster.line_width = CANONICAL_LINE_WIDTH;
    }
    if dynamic.contains(DynamicState::STENCIL_MASKS)
        && let Some(s) = d.depth_stencil.stencil.as_mut()
    {
        s.read_mask = CANONICAL_STENCIL_MASK;
        s.write_mask = CANONICAL_STENCIL_MASK;
    }
    d
}

proptest! {
    #![proptest_config(Config {
        cases: 256,
        rng_seed: RngSeed::Fixed(0x6723),
        ..Config::default()
    })]

    #[test]
    fn a_description_survives_the_key_and_the_list(case in case()) {
        let (desc, stamps, dynamic) = case;
        let fixed = FixedState::of(&desc, dynamic);
        prop_assert_eq!(fixed.dynamic, dynamic);
        // The key gives back the description, canonical in its dynamic fields.
        prop_assert_eq!(fixed.desc(desc.stages), canonical(&desc, dynamic));
        prop_assert_eq!(FixedState::of(&fixed.desc(desc.stages), dynamic), fixed.clone());

        // And the list gives back the key.
        let keys = list(vec![
            graphics(GraphicsKey { stamps, fixed }),
            compute(stamps.vertex.get()),
        ]);
        let bytes = keys.to_bytes().unwrap();
        prop_assert_eq!(KeyList::from_bytes(&bytes).unwrap(), keys);
    }
}
