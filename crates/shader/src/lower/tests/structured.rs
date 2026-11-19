//! Structured control flow: the structurizer's output lowered, its statements' shapes, and its
//! results on the host's GPU against the interpreter's.

use naga::Statement;

use super::run::{gpu_registers, interp_registers};
use super::*;
use crate::opt::{Pass, run as run_pass};
use crate::structure::{MAX_STRUCTURE_WORK, structure};

/// A loop that sums 10 down to 1 into r0, then an `if` on the sum that picks r2, then a second
/// loop that leaves early through a branch out of its middle, counting in r3.
const LOOPS: &str = "block0:
    %0 = SetRegister r0, u32 0
    %1 = SetRegister r1, u32 10
    jump block1
block1:
    %2: u32 = GetRegister r0
    %3: u32 = GetRegister r1
    %4: u32 = IAdd32 %2, %3
    %5 = SetRegister r0, %4
    %6: u32 = ISub32 %3, u32 1
    %7 = SetRegister r1, %6
    %8: u1 = INotEqual %6, u32 0
    branch %8, block1, block2
block2:
    %9: u32 = GetRegister r0
    %10: u1 = UGreaterThan %9, u32 50
    branch %10, block3, block4
block3:
    %11 = SetRegister r2, u32 1
    jump block5
block4:
    %12 = SetRegister r2, u32 2
    jump block5
block5:
    %13 = SetRegister r3, u32 0
    jump block6
block6:
    %14: u32 = GetRegister r3
    %15: u32 = IAdd32 %14, u32 1
    %16 = SetRegister r3, %15
    %17: u1 = IEqual %15, u32 7
    branch %17, block8, block7
block7:
    %18: u32 = GetRegister r3
    %19: u1 = ULessThan %18, u32 100
    branch %19, block6, block8
block8:
    return
";

/// The registers [`LOOPS`] writes.
const LOOP_REGS: [u8; 4] = [0, 1, 2, 3];

fn structured(text: &str) -> Function {
    structure(ir(text), MAX_STRUCTURE_WORK)
        .expect("structures")
        .function
}

/// The structurizer's output for loops, an `if` and an early exit lowers, validates both ways,
/// and gives the interpreter's registers on the host's GPU.
#[test]
fn structured_loops_match_the_interpreter_on_the_gpu() {
    let function = structured(LOOPS);
    let shader = lowered(GuestStage::Compute, &function, &LowerInputs::DEFAULT);
    let expected = interp_registers(&function, &LOOP_REGS);
    assert_eq!(expected, [55, 0, 1, 7], "the interpreter's own result");
    if let Some(found) = gpu_registers("structured_loops", &shader, &LOOP_REGS) {
        assert_eq!(found, expected);
    }
}

/// Identity removal and constant propagation run on the structured function (the SSA rewrite
/// refuses one: `opt::ssa`'s module docs), and the result still lowers and computes the same
/// registers. Dead code elimination is left out: before SSA it removes the register writes
/// themselves, which only the registers' later reads (gone after SSA) keep alive.
#[test]
fn the_structured_passes_keep_it_lowerable() {
    let mut function = structured(LOOPS);
    for pass in [Pass::IdentityRemoval, Pass::ConstantPropagation] {
        run_pass(&mut function, pass).expect("the pass runs on a structured function");
    }
    let shader = lowered(GuestStage::Compute, &function, &LowerInputs::DEFAULT);
    let expected = interp_registers(&function, &LOOP_REGS);
    assert_eq!(expected, [55, 0, 1, 7]);
    if let Some(found) = gpu_registers("structured_passes", &shader, &LOOP_REGS) {
        assert_eq!(found, expected);
    }
}

/// The MSL naga writes for [`LOOPS`] structured (snapshot; a change is explained in its PR).
#[test]
fn structured_loops_snapshot() {
    let shader = lowered(
        GuestStage::Compute,
        &structured(LOOPS),
        &LowerInputs::DEFAULT,
    );
    insta::assert_snapshot!("structured_loops", msl(&shader));
}

/// The entry point's statements, with emits left out.
fn statements(shader: &HostShader) -> Vec<Statement> {
    naga_module(shader).entry_points[0]
        .function
        .body
        .iter()
        .filter(|statement| !matches!(statement, Statement::Emit(_)))
        .cloned()
        .collect()
}

/// The shapes of the nodes: an `if` has an empty `reject`; a loop ends its body with
/// `if repeat {} else { break }` and has no continuing block; a `break` is `if cond { break }`;
/// `return` is a return, and nothing is appended after it.
#[test]
fn nodes_lower_to_their_naga_shapes() {
    let function = ir("block0:
    %0: u32 = GetRegister r0
    %1: u1 = IEqual %0, u32 0
block1:
    %2: u32 = GetRegister r1
    %3: u1 = IEqual %2, u32 1
structure:
    block0
    if %1 {
        loop {
            block1
            break %3
        } repeat %1
    }
    return
");
    let shader = lowered(GuestStage::Compute, &function, &LowerInputs::DEFAULT);
    let top = statements(&shader);
    let (accept, reject) = match top.as_slice() {
        [
            Statement::If { accept, reject, .. },
            Statement::Return { value: None },
        ] => Some((accept, reject)),
        _ => None,
    }
    .expect("an if and a return");
    assert!(reject.is_empty());
    let accept: Vec<&Statement> = accept
        .iter()
        .filter(|s| !matches!(s, Statement::Emit(_)))
        .collect();
    let (body, continuing) = match accept.as_slice() {
        [
            Statement::Loop {
                body,
                continuing,
                break_if: None,
            },
        ] => Some((body, continuing)),
        _ => None,
    }
    .expect("one loop");
    assert!(continuing.is_empty());
    let looped: Vec<&Statement> = body
        .iter()
        .filter(|s| !matches!(s, Statement::Emit(_) | Statement::Store { .. }))
        .collect();
    let (break_accept, break_reject, repeat_accept, repeat_reject) = match looped.as_slice() {
        [
            Statement::If {
                accept: break_accept,
                reject: break_reject,
                ..
            },
            Statement::If {
                accept: repeat_accept,
                reject: repeat_reject,
                ..
            },
        ] => Some((break_accept, break_reject, repeat_accept, repeat_reject)),
        _ => None,
    }
    .expect("the break, then the repeat test");
    assert!(matches!(
        break_accept.iter().collect::<Vec<_>>().as_slice(),
        [Statement::Break]
    ));
    assert!(break_reject.is_empty());
    assert!(repeat_accept.is_empty());
    assert!(matches!(
        repeat_reject.iter().collect::<Vec<_>>().as_slice(),
        [Statement::Break]
    ));
}

/// `unreachable` lowers as a return (a divergence in the module docs), at the end of the tree
/// and inside an `if`; nothing is appended after a final one.
#[test]
fn unreachable_returns() {
    let end = lowered(
        GuestStage::Compute,
        &ir("block0:\nstructure:\n    block0\n    unreachable\n"),
        &LowerInputs::DEFAULT,
    );
    assert!(matches!(
        statements(&end).as_slice(),
        [Statement::Return { value: None }]
    ));
    // A pixel program returns its output struct from an `unreachable` too.
    let pixel = lowered(
        GuestStage::Pixel,
        &ir("block0:
    %0: u1 = UndefU1
    %1 = SetFragColor u32 0, u32 0, f32 1.0
structure:
    block0
    if %0 {
        unreachable
    }
    return
"),
        &LowerInputs::DEFAULT,
    );
    let top = statements(&pixel);
    let accept = match top.as_slice() {
        [
            Statement::Store { .. },
            Statement::If { accept, .. },
            Statement::Return { value: Some(_) },
        ] => Some(accept),
        _ => None,
    }
    .expect("a store, an if and a return");
    assert!(
        accept
            .iter()
            .any(|s| matches!(s, Statement::Return { value: Some(_) }))
    );
    // A tree that ends without a return gets one.
    let open = lowered(
        GuestStage::Compute,
        &ir("block0:\nstructure:\n    block0\n"),
        &LowerInputs::DEFAULT,
    );
    assert!(matches!(
        statements(&open).as_slice(),
        [Statement::Return { value: None }]
    ));
}

/// `DemoteToHelperInvocation` is naga's `Kill` in a pixel program (a divergence in the module
/// docs) and refused elsewhere.
#[test]
fn demote_is_kill() {
    let text = "block0:\n    %0 = DemoteToHelperInvocation\n    return\n";
    let pixel = lowered(GuestStage::Pixel, &ir(text), &LowerInputs::DEFAULT);
    assert!(matches!(
        statements(&pixel).as_slice(),
        [Statement::Kill, Statement::Return { value: None }]
    ));
    for stage in [GuestStage::Vertex, GuestStage::Compute] {
        let refused = lower(stage, &ir(text)).expect_err("only pixel programs discard");
        assert_eq!(
            refused,
            LowerError::NotInStage {
                at: value_at(&ir(text), 0),
                opcode: Opcode::DemoteToHelperInvocation,
                stage,
            }
        );
        assert_eq!(
            refused.to_string(),
            format!("%0: a {stage} program cannot use DemoteToHelperInvocation")
        );
    }
}
