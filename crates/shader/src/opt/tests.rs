//! The passes' tests: golden IR (hand-written IR in, `insta` snapshots out), the departures
//! from Eden each module lists, interpreter equivalence on hand-written and generated
//! functions, and the work bound.

mod constant;
mod dce;
mod equivalence;
mod identity;
mod linear;
mod pipeline;
mod ssa;
mod structured;

use super::{Pass, PassError, run};
use crate::interp::Machine;
use crate::ir::{Function, Value};

/// Parses `text`, which must be well-formed.
fn parse(text: &str) -> Function {
    Function::parse(text).expect("parses")
}

/// The text of `text` after `pass`, which must succeed (and verify: [`run`] checks in tests).
fn after(text: &str, pass: Pass) -> String {
    let mut function = parse(text);
    run(&mut function, pass).expect("the pass runs");
    function.to_string()
}

/// The text of `text` after each pass of `passes` in turn.
fn after_each(text: &str, passes: &[Pass]) -> String {
    let mut function = parse(text);
    for &pass in passes {
        run(&mut function, pass).expect("the pass runs");
    }
    function.to_string()
}

/// The error `pass` gives on `text`.
fn error(text: &str, pass: Pass) -> PassError {
    let mut function = parse(text);
    run(&mut function, pass).expect_err("the pass fails")
}

/// The value id `index` (ids have no public constructor; a parsed function hands one out).
fn value(index: u32) -> Value {
    let text = format!("block0:\n    %{index} = Void\n    return\n");
    parse(&text)
        .blocks()
        .next()
        .and_then(|(_, block)| block.insts.first().copied())
        .expect("one instruction")
}

/// The machine a run of `function` on `machine` leaves.
fn ran(function: &Function, machine: &Machine) -> Machine {
    let mut machine = machine.clone();
    machine.run(function).expect("runs");
    machine
}
