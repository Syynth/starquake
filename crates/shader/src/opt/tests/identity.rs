//! Identity removal: chains rewired in operations, phis, terminators and the structured tree;
//! `Identity` and `Void` leave their blocks; malformed chains are refused.

use super::{after, after_each, error, parse, value};
use std::fmt::Write;

use crate::ir::{Inst, Opcode, Operand};
use crate::opt::identity::Resolver;
use crate::opt::{Pass, PassError, Work, identity_of, run};

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
        let inst = function.inst(value(index));
        assert!(
            matches!(inst, Some(Inst::Op(op)) if op.opcode == Opcode::Void && op.args.is_empty()),
            "%{index} is {inst:?}"
        );
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

#[test]
fn a_chain_as_long_as_the_arena_resolves() {
    // The arena holds three instructions, a chain of three Identities, so the resolver's walk
    // takes exactly as many steps as the arena has slots, the most it allows. block0's
    // condition reads the chain before block1's links are rewired.
    let text = "block0:
    branch %0, block1, block1
block1:
    %0: opaque = Identity %1
    %1: opaque = Identity %2
    %2: opaque = Identity u1 true
    return
";
    assert_eq!(
        after(text, Pass::IdentityRemoval),
        "block0:
    branch u1 true, block1, block1
block1:
    return
"
    );
}

#[test]
fn chain_ends_are_remembered_for_users_listed_first() {
    // %0, then `count` users of the links from the last to the first, then the links, last
    // first: link k is `Identity` of link k - 1, and link 1 of %0. Each user is rewired before
    // any link is, so only the remembered ends keep the walk linear.
    let count: u32 = 5_000;
    let link = |k: u32| count + k;
    let mut text = String::from("block0:\n    %0: u32 = GetCbufU32 u32 0, u32 0\n");
    for user in 1..=count {
        let k = count + 1 - user;
        let _ = writeln!(text, "    %{user}: u32 = IAdd32 %{}, u32 1", link(k));
    }
    for k in (1..=count).rev() {
        let previous = if k == 1 { 0 } else { link(k - 1) };
        let _ = writeln!(text, "    %{}: opaque = Identity %{previous}", link(k));
    }
    text.push_str("    return\n");
    let mut function = parse(&text);
    let work = run(&mut function, Pass::IdentityRemoval).expect("runs");
    let size = u64::from(2 * count + 1);
    assert!(work.steps() <= 4 * size, "{} steps", work.steps());
    let out = function.to_string();
    assert!(out.contains("    %1: u32 = IAdd32 %0, u32 0x1\n"), "{out}");
    assert!(!out.contains("Identity"), "{out}");
}

#[test]
fn the_resolver_follows_the_arena_as_it_grows() {
    // The SSA rewrite adds instructions after making its resolver: a chain of three added
    // links is longer than the one-instruction arena the resolver was made for, and its ends
    // are remembered.
    let mut function = parse("block0:\n    %0: u32 = GetCbufU32 u32 0, u32 0\n    return\n");
    let mut resolver = Resolver::new(&function);
    let entry = function.entry();
    let mut last = Operand::Value(value(0));
    for _ in 0..3 {
        last = Operand::Value(function.push(entry, identity_of(last)).expect("pushes"));
    }
    let mut first = Work::default();
    assert_eq!(
        resolver.resolve(&function, last, &mut first),
        Ok(Operand::Value(value(0)))
    );
    // Once remembered, the chain costs one step to its remembered end and one to see it ends.
    let mut again = Work::default();
    assert_eq!(
        resolver.resolve(&function, last, &mut again),
        Ok(Operand::Value(value(0)))
    );
    assert_eq!(again.steps(), 2);
}
