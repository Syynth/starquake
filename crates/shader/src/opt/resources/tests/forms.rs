//! The three handle forms the pass takes, and Eden's opcode table.

use super::{
    BOUND_BANK, Engine, FLOAT, buffer, parse, pass, sample_of, scan_error, sources, untraced,
    value,
};
use crate::ir::{Inst, Opcode};
use crate::opt::resources::opcodes::{Form, classify, is_storage_image};
use crate::opt::resources::{ResourceError, Untraced, scan};
use crate::translate::texture::ABSENT;

/// The operation at `%60` of `function`.
fn op60(function: &crate::ir::Function) -> crate::ir::Operation {
    match function.inst(value(60)) {
        Some(Inst::Op(op)) => op.clone(),
        other => panic!("no operation: {other:?}"),
    }
}

#[test]
fn an_immediate_handle_is_an_offset_in_the_descriptor_buffer() {
    let text = sample_of("", "u32 0x40");
    assert_eq!(sources(&text), "%60: cbuf 2 at 0x40\n");
    let engine = Engine::new().word(BOUND_BANK, 0x40, 0x1234).kind(0x1234, FLOAT);
    let (function, table) = pass(&text, &engine);
    assert_eq!(table.to_string(), "texture 0: 2d, float, from cbuf 2 at 0x40\n");
    let op = op60(&function);
    assert_eq!(op.opcode, Opcode::ImageSampleImplicitLod);
    assert_eq!(op.args.first(), Some(&ABSENT));
}

#[test]
fn bound_and_bindless_forms_become_the_unbound_form() {
    let bound = "block0:
    %50: f32x2 = CompositeConstructF32x2 f32 0x0, f32 0x0
    %60: f32x4 = BoundImageSampleImplicitLod u32 0x48, %50, u32 0x0, u32 0x0 {type=2d}
    return
";
    assert_eq!(sources(bound), "%60: cbuf 2 at 0x48\n");
    let bindless = "block0:
    %0: u32 = GetCbufU32 u32 3, u32 0x10
    %50: f32x2 = CompositeConstructF32x2 f32 0x0, f32 0x0
    %60: f32x4 = BindlessImageSampleImplicitLod %0, %50, u32 0x0, u32 0x0 {type=2d}
    return
";
    assert_eq!(sources(bindless), "%60: cbuf 3 at 0x10\n");
    for (text, bank, offset) in [(bound, BOUND_BANK, 0x48), (bindless, 3, 0x10)] {
        let engine = Engine::new().word(bank, offset, 0x77).kind(0x77, FLOAT);
        let (function, _) = pass(text, &engine);
        let op = op60(&function);
        assert_eq!(op.opcode, Opcode::ImageSampleImplicitLod, "{function}");
        assert_eq!(op.args.first(), Some(&ABSENT));
    }
}

#[test]
fn a_value_handle_of_the_unbound_form_is_traced() {
    let text = sample_of("    %0: u32 = GetCbufU32 u32 5, u32 0x20\n", "%0");
    assert_eq!(sources(&text), "%60: cbuf 5 at 0x20\n");
}

#[test]
fn a_bound_form_with_a_value_is_malformed() {
    let text = "block0:
    %0: u32 = GetCbufU32 u32 3, u32 0x10
    %50: f32x2 = CompositeConstructF32x2 f32 0x0, f32 0x0
    %60: f32x4 = BoundImageSampleImplicitLod %0, %50, u32 0x0, u32 0x0 {type=2d}
    return
";
    let error = scan_error(text);
    assert_eq!(error, ResourceError::Malformed(value(60)));
    assert_eq!(
        error.to_string(),
        "%60: a texture operation with an argument or its flags missing or of the wrong kind"
    );
}

#[test]
fn a_bindless_immediate_is_refused() {
    let text = "block0:
    %50: f32x2 = CompositeConstructF32x2 f32 0x0, f32 0x0
    %60: f32x4 = BindlessImageSampleImplicitLod u32 0x40, %50, u32 0x0, u32 0x0 {type=2d}
    return
";
    assert_eq!(untraced(text), Untraced::Immediate);
}

/// The listed divergence (module docs, "Handle forms"): after SSA a register handle that holds
/// a constant is an immediate, which the unbound form reads as a bound offset.
#[test]
fn a_folded_register_handle_reads_as_bound() {
    let text = sample_of("", "u32 0x0");
    assert_eq!(sources(&text), "%60: cbuf 2 at 0x0\n");
}

#[test]
fn a_bound_handle_needs_a_descriptor_buffer() {
    let text = sample_of("", "u32 0x40");
    let error = scan(&parse(&text), crate::translate::texture::DescriptorBuffer::new())
        .expect_err("refused");
    assert_eq!(error, ResourceError::NoDescriptorBuffer(value(60)));
    assert_eq!(
        error.to_string(),
        "%60: a bound texture handle, but the program has no texture descriptor buffer"
    );
    // A bindless handle needs none.
    let bindless = sample_of("    %0: u32 = GetCbufU32 u32 3, u32 0x10\n", "%0");
    assert!(
        scan(
            &parse(&bindless),
            crate::translate::texture::DescriptorBuffer::new()
        )
        .is_ok()
    );
    let _ = buffer(BOUND_BANK);
}

/// Eden's `IndexedInstruction` rows, written out from
/// `eden/src/shader_recompiler/ir_opt/texture_pass.cpp` @ cf654052: bound, bindless, indexed.
const EDEN_ROWS: [(&str, &str, &str); 23] = [
    ("BoundImageSampleImplicitLod", "BindlessImageSampleImplicitLod", "ImageSampleImplicitLod"),
    ("BoundImageSampleExplicitLod", "BindlessImageSampleExplicitLod", "ImageSampleExplicitLod"),
    ("BoundImageSampleDrefImplicitLod", "BindlessImageSampleDrefImplicitLod", "ImageSampleDrefImplicitLod"),
    ("BoundImageSampleDrefExplicitLod", "BindlessImageSampleDrefExplicitLod", "ImageSampleDrefExplicitLod"),
    ("BoundImageGather", "BindlessImageGather", "ImageGather"),
    ("BoundImageGatherDref", "BindlessImageGatherDref", "ImageGatherDref"),
    ("BoundImageFetch", "BindlessImageFetch", "ImageFetch"),
    ("BoundImageQueryDimensions", "BindlessImageQueryDimensions", "ImageQueryDimensions"),
    ("BoundImageQueryLod", "BindlessImageQueryLod", "ImageQueryLod"),
    ("BoundImageGradient", "BindlessImageGradient", "ImageGradient"),
    ("BoundImageRead", "BindlessImageRead", "ImageRead"),
    ("BoundImageWrite", "BindlessImageWrite", "ImageWrite"),
    ("BoundImageAtomicIAdd32", "BindlessImageAtomicIAdd32", "ImageAtomicIAdd32"),
    ("BoundImageAtomicSMin32", "BindlessImageAtomicSMin32", "ImageAtomicSMin32"),
    ("BoundImageAtomicUMin32", "BindlessImageAtomicUMin32", "ImageAtomicUMin32"),
    ("BoundImageAtomicSMax32", "BindlessImageAtomicSMax32", "ImageAtomicSMax32"),
    ("BoundImageAtomicUMax32", "BindlessImageAtomicUMax32", "ImageAtomicUMax32"),
    ("BoundImageAtomicInc32", "BindlessImageAtomicInc32", "ImageAtomicInc32"),
    ("BoundImageAtomicDec32", "BindlessImageAtomicDec32", "ImageAtomicDec32"),
    ("BoundImageAtomicAnd32", "BindlessImageAtomicAnd32", "ImageAtomicAnd32"),
    ("BoundImageAtomicOr32", "BindlessImageAtomicOr32", "ImageAtomicOr32"),
    ("BoundImageAtomicXor32", "BindlessImageAtomicXor32", "ImageAtomicXor32"),
    ("BoundImageAtomicExchange32", "BindlessImageAtomicExchange32", "ImageAtomicExchange32"),
];

/// Eden's `IsStorageImageOpcode` list, written out.
const EDEN_STORAGE: [&str; 13] = [
    "ImageRead",
    "ImageAtomicIAdd32",
    "ImageAtomicSMin32",
    "ImageAtomicUMin32",
    "ImageAtomicSMax32",
    "ImageAtomicUMax32",
    "ImageAtomicInc32",
    "ImageAtomicDec32",
    "ImageAtomicAnd32",
    "ImageAtomicOr32",
    "ImageAtomicXor32",
    "ImageAtomicExchange32",
    "ImageWrite",
];

fn named(name: &str) -> Opcode {
    Opcode::from_name(name).unwrap_or_else(|| panic!("no opcode {name}"))
}

#[test]
fn every_row_matches_edens_table() {
    for (bound, bindless, indexed) in EDEN_ROWS {
        let indexed = named(indexed);
        assert_eq!(classify(named(bound)), Some((Form::Bound, indexed)), "{bound}");
        assert_eq!(
            classify(named(bindless)),
            Some((Form::Bindless, indexed)),
            "{bindless}"
        );
        assert_eq!(classify(indexed), Some((Form::Unbound, indexed)), "{indexed}");
    }
    let listed: Vec<Opcode> = EDEN_ROWS
        .iter()
        .flat_map(|&(a, b, c)| [named(a), named(b), named(c)])
        .collect();
    for &opcode in Opcode::ALL {
        if !listed.contains(&opcode) {
            assert_eq!(classify(opcode), None, "{opcode}");
        }
        let storage = EDEN_STORAGE.iter().any(|&name| named(name) == opcode);
        assert_eq!(is_storage_image(opcode), storage, "{opcode}");
    }
}
