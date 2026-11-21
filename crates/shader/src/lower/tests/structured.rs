//! Structured control flow: the structurizer's output lowered, its statements' shapes, and its
//! results on the host's GPU against the interpreter's.

use naga::Statement;

use super::run::{gpu_registers, interp_registers};
use super::*;
use crate::interp::{InterpError, Machine};
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

/// Identity removal and constant propagation run on the structured function before SSA, and
/// the result still lowers and computes the same registers. Dead code elimination is left out:
/// before SSA it removes the register writes themselves, which only the registers' later reads
/// (gone after SSA) keep alive.
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

/// [`LOOPS`] structured, then rewritten into SSA form, in Eden's order (slice 8b), with identity
/// removal and constant propagation after it: the values its registers carried between blocks
/// now cross in phis, which lower as local variables, and the GPU gives the interpreter's
/// registers. Dead code elimination is left out, so the register writes the harness reads stay.
#[test]
fn structured_phis_match_the_interpreter_on_the_gpu() {
    let mut function = structured(LOOPS);
    for pass in [
        Pass::SsaRewrite,
        Pass::IdentityRemoval,
        Pass::ConstantPropagation,
    ] {
        run_pass(&mut function, pass).expect("the pass runs on a structured function");
    }
    assert!(
        function.listed().any(|(_, inst)| inst.is_phi()),
        "{function}"
    );
    assert!(!function.to_string().contains("GetRegister"), "{function}");
    let shader = lowered(GuestStage::Compute, &function, &LowerInputs::DEFAULT);
    let expected = interp_registers(&function, &LOOP_REGS);
    assert_eq!(expected, [55, 0, 1, 7], "the interpreter's own result");
    if let Some(found) = gpu_registers("structured_phis", &shader, &LOOP_REGS) {
        assert_eq!(found, expected);
    }
    insta::assert_snapshot!("structured_phis", msl(&shader));
}

/// A phi in the first block run, which no block ran before (`lower` module docs, the divergence
/// table): the interpreter refuses it (`PhiAtStart`), and the lowering reads its local
/// variable's zero. The phi plus 5 goes to r1, so the GPU gives 5 only if the phi read 0; the
/// MSL (snapshot) zero-initialises the phi's local (`uint phi0_ = {};`).
#[test]
fn a_phi_in_the_first_block_run_reads_zero_on_the_gpu() {
    const ADDED: u32 = 5;
    let function = ir(&format!(
        "block0:
    %0: u32 = Phi
    %1: u32 = IAdd32 %0, u32 {ADDED}
    %2 = SetRegister r1, %1
structure:
    block0
    return
"
    ));
    assert_eq!(
        Machine::new().run(&function).err(),
        Some(InterpError::PhiAtStart(value_at(&function, 0))),
        "the interpreter refuses it"
    );
    let shader = lowered(GuestStage::Compute, &function, &LowerInputs::DEFAULT);
    let text = msl(&shader);
    assert!(text.contains("uint phi0_ = {};"), "{text}");
    if let Some(found) = gpu_registers("first_block_phi", &shader, &[1]) {
        assert_eq!(found, [ADDED]);
    }
    insta::assert_snapshot!("first_block_phi", text);
}

/// A function whose block3 has a phi taking `incoming`, with block1 inside an `if` defining %1.
fn phi_taking(incoming: &str) -> Function {
    ir(&format!(
        "block0:
    %0: u1 = IEqual u32 0, u32 0
block1:
    %1: u32 = IAdd32 u32 1, u32 2
block2:
    %2: u32 = IAdd32 u32 3, u32 4
block3:
    %3: u32 = Phi {incoming}
    %4 = SetRegister r1, %3
structure:
    block0
    if %0 {{
        block1
    }}
    block2
    block3
    return
"
    ))
}

/// The lowering stores a phi's incoming value at the end of the block it names, so the value
/// must be in scope there and the block must be one that can run last before the phi's; the
/// verifier refuses any other (`ir::verify`, step 5) before the lowering starts, so
/// [`LowerError::PhiIncoming`] is not reached through [`lower_with`].
#[test]
fn a_phi_value_out_of_scope_is_refused() {
    // block2 is the one block that can run last before block3; %2 is in scope at its end.
    for incoming in ["[block2: %2]", "[block2: u32 9]"] {
        lowered(
            GuestStage::Compute,
            &phi_taking(incoming),
            &LowerInputs::DEFAULT,
        );
    }
    let block = |function: &Function, index: usize| {
        function
            .blocks()
            .nth(index)
            .map(|(id, _)| id)
            .expect("a block")
    };
    // %1 is out of scope at the end of block2 (the `if` body that holds it has ended).
    let function = phi_taking("[block2: %1]");
    assert_eq!(
        lower_with(GuestStage::Compute, &function, &LowerInputs::DEFAULT).err(),
        Some(LowerError::Verify(VerifyError::PhiOutOfScope {
            at: value_at(&function, 3),
            operand: value_at(&function, 1),
            block: block(&function, 2),
        }))
    );
    // block1 (inside the `if`) and block0 (where %1 is not computed yet either) cannot run last
    // before block3.
    for (incoming, from) in [("[block1: %1]", 1), ("[block0: %1]", 0)] {
        let function = phi_taking(incoming);
        assert_eq!(
            lower_with(GuestStage::Compute, &function, &LowerInputs::DEFAULT).err(),
            Some(LowerError::Verify(VerifyError::PhiNotPredecessor {
                at: value_at(&function, 3),
                block: block(&function, from),
            })),
            "{incoming}"
        );
    }
}

#[test]
fn a_phi_of_a_type_with_no_naga_type_is_refused() {
    let function = ir("block0:
    %0: f64 = Phi
structure:
    block0
    return
");
    let error =
        lower_with(GuestStage::Compute, &function, &LowerInputs::DEFAULT).expect_err("refused");
    assert_eq!(
        error,
        LowerError::Phi {
            at: value_at(&function, 0),
            ty: Type::F64,
        }
    );
    assert_eq!(error.to_string(), "%0: a phi of type f64 is not lowered");
}

#[test]
fn a_loop_bodys_value_leaves_scope_at_its_end() {
    // %0 is in scope at the end of block1, in the loop body, and out of it at block2's; the
    // phi in block3 takes it from the block run right before block3.
    let text = |after_loop: &str, from: &str| {
        format!(
            "block0:
block1:
    %0: u32 = IAdd32 u32 1, u32 2
block2:
block3:
    %1: u32 = Phi [{from}: %0]
    %2 = SetRegister r1, %1
structure:
    block0
    loop {{
        block1
    }} repeat u1 false
{after_loop}    return
"
        )
    };
    lowered(
        GuestStage::Compute,
        &ir(&text("    block3\n    block2\n", "block1")),
        &LowerInputs::DEFAULT,
    );
    let function = ir(&text("    block2\n    block3\n", "block2"));
    let block2 = function.blocks().nth(2).map(|(id, _)| id).expect("block2");
    assert_eq!(
        lower_with(GuestStage::Compute, &function, &LowerInputs::DEFAULT).err(),
        Some(LowerError::Verify(VerifyError::PhiOutOfScope {
            at: value_at(&function, 1),
            operand: value_at(&function, 0),
            block: block2,
        }))
    );
}

#[test]
fn a_half_phi_needs_f16() {
    let function = ir("block0:
    %0: f16 = Phi
structure:
    block0
    return
");
    let without = LowerInputs {
        caps: crate::host::HostCaps {
            f16: false,
            ..crate::host::HostCaps::ALL
        },
        ..LowerInputs::DEFAULT
    };
    assert_eq!(
        lower_with(GuestStage::Compute, &function, &without).err(),
        Some(LowerError::Missing(crate::host::Feature::Float16))
    );
    let shader = lowered(GuestStage::Compute, &function, &LowerInputs::DEFAULT);
    assert!(shader.requirements.f16);
}

#[test]
fn a_u64_phi_needs_i64() {
    let function = ir("block0:
    %0: u64 = Phi
structure:
    block0
    return
");
    let without = LowerInputs {
        caps: crate::host::HostCaps {
            i64: false,
            ..crate::host::HostCaps::ALL
        },
        ..LowerInputs::DEFAULT
    };
    assert_eq!(
        lower_with(GuestStage::Compute, &function, &without).err(),
        Some(LowerError::Missing(crate::host::Feature::Int64))
    );
    let shader = lowered(GuestStage::Compute, &function, &LowerInputs::DEFAULT);
    assert!(shader.requirements.i64);
}
