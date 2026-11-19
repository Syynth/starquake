//! Texture and image operations against the scripted resource model.

use super::run;
use crate::interp::{ImageCall, InterpError, Machine, ScriptedImage, Val};
use crate::ir::{Opcode, TextureInfo, TextureType};

/// A sample at (0.5, 0.25) with bias 0 and offset 0, then a write of (1, 2, 3, 4) there.
const PROGRAM: &str = "block0:
    %0 = CompositeConstructF32x2 f32 0.5, f32 0.25
    %1 = BoundImageSampleImplicitLod u32 0x3, %0, f32 0.0, u32 0x0 {descriptor=2, type=2d}
    %2 = CompositeConstructU32x4 u32 0x1, u32 0x2, u32 0x3, u32 0x4
    %3 = BoundImageWrite u32 0x3, %0, %2 {type=2d}
    return
";

fn sample_info() -> TextureInfo {
    TextureInfo {
        descriptor_index: 2,
        ty: TextureType::Color2D,
        ..TextureInfo::default()
    }
}

fn coords() -> Val {
    Val::F32x2([0.5_f32.to_bits(), 0.25_f32.to_bits()])
}

fn sample() -> ScriptedImage {
    ScriptedImage {
        call: ImageCall {
            opcode: Opcode::BoundImageSampleImplicitLod,
            info: sample_info(),
            args: vec![Val::U32(3), coords(), Val::F32(0), Val::U32(0)],
        },
        reply: Some(Val::F32x4([1, 2, 3, 4])),
    }
}

fn write() -> ScriptedImage {
    ScriptedImage {
        call: ImageCall {
            opcode: Opcode::BoundImageWrite,
            info: TextureInfo {
                ty: TextureType::Color2D,
                ..TextureInfo::default()
            },
            args: vec![Val::U32(3), coords(), Val::U32x4([1, 2, 3, 4])],
        },
        reply: None,
    }
}

fn machine_with(script: Vec<ScriptedImage>) -> Machine {
    let mut machine = Machine::new();
    machine.images = script.into();
    machine
}

fn error(script: Vec<ScriptedImage>) -> (String, usize) {
    let mut machine = machine_with(script);
    let words = run(PROGRAM, &mut machine)
        .err()
        .map_or_else(String::new, |e| e.to_string());
    (words, machine.images.len())
}

#[test]
fn scripted_calls_reply_in_order() {
    let mut machine = machine_with(vec![sample(), write()]);
    let finished = run(PROGRAM, &mut machine).expect("runs");
    assert_eq!(finished.value(1), Some(Val::F32x4([1, 2, 3, 4])));
    assert_eq!(finished.value(3), None);
    assert!(machine.images.is_empty());
}

#[test]
fn a_call_must_match_its_script_entry() {
    let mismatch = "%1: the BoundImageSampleImplicitLod call differs from scripted call 0";
    // Each part of the call: the arguments, the texture information, the opcode.
    let mut other_args = sample();
    other_args.call.args[1] = Val::F32x2([0.5_f32.to_bits(), 0.5_f32.to_bits()]);
    let mut other_info = sample();
    other_info.call.info.descriptor_index = 1;
    let mut other_opcode = sample();
    other_opcode.call.opcode = Opcode::BindlessImageSampleImplicitLod;
    for script in [other_args, other_info, other_opcode] {
        // A refused call leaves its entry in the script.
        assert_eq!(error(vec![script, write()]), (mismatch.to_owned(), 2));
    }
    // The second call is counted from the run's start.
    let mut late = write();
    late.call.args[2] = Val::U32x4([1, 2, 3, 5]);
    assert_eq!(
        error(vec![sample(), late]),
        (
            "%3: the BoundImageWrite call differs from scripted call 1".to_owned(),
            1
        )
    );
}

#[test]
fn the_script_runs_out() {
    assert_eq!(
        error(Vec::new()),
        (
            "%1: BoundImageSampleImplicitLod has no scripted call left".to_owned(),
            0
        )
    );
    assert_eq!(
        error(vec![sample()]),
        (
            "%3: BoundImageWrite has no scripted call left".to_owned(),
            0
        )
    );
}

#[test]
fn a_reply_must_have_the_result_type() {
    let mut wrong = sample();
    wrong.reply = Some(Val::U32(1));
    assert_eq!(
        error(vec![wrong]),
        (
            "%1: scripted call 0 replies with the wrong type for BoundImageSampleImplicitLod"
                .to_owned(),
            1
        )
    );
    let mut missing = sample();
    missing.reply = None;
    assert_eq!(
        error(vec![missing]).0,
        "%1: scripted call 0 replies with the wrong type for BoundImageSampleImplicitLod"
    );
    let mut extra = write();
    extra.reply = Some(Val::U32(1));
    assert_eq!(
        error(vec![sample(), extra]).0,
        "%3: scripted call 1 replies with the wrong type for BoundImageWrite"
    );
    // Both faults: the call is checked before the reply.
    let mut both = sample();
    both.call.info.descriptor_index = 1;
    both.reply = Some(Val::U32(1));
    assert_eq!(
        run(PROGRAM, &mut machine_with(vec![both])),
        Err(InterpError::ImageMismatch {
            at: crate::ir::Function::parse(PROGRAM)
                .expect("parses")
                .listed()
                .nth(1)
                .map(|(value, _)| value)
                .expect("an instruction"),
            opcode: Opcode::BoundImageSampleImplicitLod,
            index: 0,
        })
    );
}
