//! Interpreter tests: hand-written IR in the text form, with results worked out by hand, one
//! file per operation family. Expected values are written out as literals (bits for floats),
//! so a wrong semantic cannot hide behind a shared helper.

mod compare;
mod composite;
mod control;
mod convert;
mod float;
mod image;
mod integer;
mod memory;

use super::{Finished, InterpError, Machine, Val};
use crate::ir::Function;

/// Parses `text`, which must be well-formed.
fn parse(text: &str) -> Function {
    Function::parse(text).expect("parses")
}

/// Runs `text` on `machine`.
fn run(text: &str, machine: &mut Machine) -> Result<Finished, InterpError> {
    machine.run(&parse(text))
}

/// The value of `%id` after running the straight-line `body` (instruction lines of one block,
/// which this adds a `return` to) on a fresh machine.
fn value_of(body: &str, id: u32) -> Val {
    let text = format!("block0:\n{body}\n    return\n");
    let finished = run(&text, &mut Machine::new()).expect("runs");
    finished.value(id).expect("a value")
}

/// The value of the single operation `op`, as `%0`.
fn eval(op: &str) -> Val {
    value_of(&format!("    %0 = {op}"), 0)
}

/// The error, as words, of running the single operation `op` on `machine`.
fn eval_error_on(op: &str, machine: &mut Machine) -> String {
    let text = format!("block0:\n    %0 = {op}\n    return\n");
    run(&text, machine)
        .err()
        .map_or_else(String::new, |error| error.to_string())
}

/// The error, as words, of running the single operation `op` on a fresh machine.
fn eval_error(op: &str) -> String {
    eval_error_on(op, &mut Machine::new())
}

/// The bits of a 32-bit float value.
fn f32_bits(value: Val) -> u32 {
    match value {
        Val::F32(bits) => Some(bits),
        _ => None,
    }
    .expect("an f32")
}

/// The bits of a 16-bit float value.
fn f16_bits(value: Val) -> u16 {
    match value {
        Val::F16(bits) => Some(bits),
        _ => None,
    }
    .expect("an f16")
}

/// The bits of a 64-bit float value.
fn f64_bits(value: Val) -> u64 {
    match value {
        Val::F64(bits) => Some(bits),
        _ => None,
    }
    .expect("an f64")
}

/// A 32-bit integer value.
fn u32_of(value: Val) -> u32 {
    match value {
        Val::U32(bits) => Some(bits),
        _ => None,
    }
    .expect("a u32")
}

/// A 64-bit integer value.
fn u64_of(value: Val) -> u64 {
    match value {
        Val::U64(bits) => Some(bits),
        _ => None,
    }
    .expect("a u64")
}

/// A boolean value.
fn u1_of(value: Val) -> bool {
    match value {
        Val::U1(bit) => Some(bit),
        _ => None,
    }
    .expect("a u1")
}

/// An `f32` immediate with the bits `bits`.
fn f32_imm(bits: u32) -> String {
    format!("f32 {bits:#x}")
}
