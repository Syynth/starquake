//! Identity removal: chains rewired in operations, phis, terminators and the structured tree;
//! `Identity` and `Void` leave their blocks; malformed chains are refused.

use super::{after, after_each, error, parse, value};
use crate::ir::{Inst, Opcode};
use crate::opt::{Pass, PassError, run};

/// The SSA rewrite's output for a loop of a header and a body.
const LOOP: &str = "block0:
    %0 = SetRegister r0, u32 3
    %1 = SetRegister r1, u32 0
    jump block1
block1:
    %2: u32 = GetRegister r0
    %3: u1 = INotEqual %2, u32 0
    branch %3, block2, block3
block2:
    %4: u32 = GetRegister r1
    %5: u32 = IAdd32 %4, %2
    %6 = SetRegister r1, %5
    %7: u32 = ISub32 %2, u32 1
    %8 = SetRegister r0, %7
    jump block1
block3:
    %9: u32 = GetRegister r1
    %10 = SetRegister r2, %9
    return
";

#[test]
fn identities_leave_after_the_rewrite() {
    let text = after_each(LOOP, &[Pass::SsaRewrite, Pass::IdentityRemoval]);
    assert!(!text.contains("Identity"), "{text}");
    insta::assert_snapshot!("identity_after_ssa", text);
}

#[test]
fn chains_operands_and_phis_are_rewired() {
    let text = "block0:
    %0: u32 = GetCbufU32 u32 0, u32 0
    %1: opaque = Identity %0
    %2: opaque = Identity %1
    %3: opaque = Identity u32 9
    %4: u32 = IAdd32 %2, %3
    %5 = Void
    jump block1
block1:
    %6: u32 = Phi [block0: %2], [block1: %7]
    %7: u32 = IAdd32 %6, %1
    %8: u1 = IEqual %7, %3
    branch %8, block2, block1
block2:
    return
";
    assert_eq!(
        after(text, Pass::IdentityRemoval),
        "block0:
    %0: u32 = GetCbufU32 u32 0x0, u32 0x0
    %4: u32 = IAdd32 %0, u32 0x9
    jump block1
block1:
    %6: u32 = Phi [block0: %0], [block1: %7]
    %7: u32 = IAdd32 %6, %0
    %8: u1 = IEqual %7, u32 0x9
    branch %8, block2, block1
block2:
    return
"
    );
}

#[test]
fn removed_instructions_are_invalidated() {
    let mut function = parse(
        "block0:
    %0: opaque = Identity u32 1
    %1 = Void
    %2 = SetRegister r0, %0
    return
",
    );
    run(&mut function, Pass::IdentityRemoval).expect("runs");
    for index in [0, 1] {
        match function.inst(value(index)) {
            Some(Inst::Op(op)) => {
                assert_eq!((op.opcode, op.args.len()), (Opcode::Void, 0));
            }
            other => panic!("%{index} is {other:?}"),
        }
    }
    let listed: Vec<_> = function.listed().map(|(v, _)| v).collect();
    assert_eq!(listed, vec![value(2)]);
}

#[test]
fn conditions_are_rewired() {
    let graph = "block0:
    %0: u1 = GetPred p0
    %1: opaque = Identity %0
    branch %1, block1, block1
block1:
    return
";
    assert_eq!(
        after(graph, Pass::IdentityRemoval),
        "block0:
    %0: u1 = GetPred p0
    branch %0, block1, block1
block1:
    return
"
    );
    let tree = "block0:
    %0: u1 = GetPred p0
    %1: opaque = Identity %0
    %2: opaque = Identity u1 false
    %3: opaque = Identity u1 true
block1:
    %4 = SetRegister r0, u32 1
structure:
    block0
    if %1 {
        loop {
            block1
            break %3
        } repeat %2
    }
    return
";
    assert_eq!(
        after(tree, Pass::IdentityRemoval),
        "block0:
    %0: u1 = GetPred p0
block1:
    %4 = SetRegister r0, u32 0x1
structure:
    block0
    if %0 {
        loop {
            block1
            break u1 true
        } repeat u1 false
    }
    return
"
    );
}

#[test]
fn cycles_are_refused() {
    // %1's chain is the first met: %1 -> %0 -> %1.
    let looped = "block0:
    %0: opaque = Identity %1
    %1: opaque = Identity %0
    %2 = SetRegister r0, %0
    return
";
    assert_eq!(
        error(looped, Pass::IdentityRemoval),
        PassError::Identity(value(1))
    );
    assert_eq!(
        PassError::Identity(value(1)).to_string(),
        "%1 is a chain of Identity operations that loops or ends without a value"
    );
    // A chain that ends at an id past the arena, and one that ends at an Identity with no
    // argument.
    let missing = "block0:
    %0: opaque = Identity %9
    %1 = SetRegister r0, %0
    return
";
    assert_eq!(
        error(missing, Pass::IdentityRemoval),
        PassError::Identity(value(0))
    );
    let empty = "block0:
    %0: opaque = Identity
    %1: opaque = Identity %0
    %2 = SetRegister r0, %1
    return
";
    assert_eq!(
        error(empty, Pass::IdentityRemoval),
        PassError::Identity(value(0))
    );
}

#[test]
fn a_missing_listed_id_is_refused() {
    let mut function = parse("block0:\n    %0 = Void\n    return\n");
    if let Some(block) = function.block_mut(function.entry()) {
        block.insts.push(value(5));
    }
    assert_eq!(
        run(&mut function, Pass::IdentityRemoval),
        Err(PassError::Missing(value(5)))
    );
}
