//! Verifier tests: each rejection with its exact error, on hand-written IR, both sides of each
//! rule, and the documented order of checks with two faults present at once.

use super::super::*;

fn ir(text: &str) -> Function {
    Function::parse(text).expect("parses")
}

fn verify(text: &str) -> Result<(), VerifyError> {
    ir(text).verify()
}

fn v(index: u32) -> Value {
    Value(index)
}

fn b(index: u32) -> BlockId {
    BlockId(index)
}

#[test]
fn a_function_needs_a_block() {
    assert_eq!(verify(""), Err(VerifyError::NoBlocks));
    assert_eq!(verify("block0:\n    return\n"), Ok(()));
}

#[test]
fn block_lists_name_each_instruction_once() {
    let mut f = ir("block0:\n    %0: u32 = LaneId\n    return\n");
    f.blocks[0].insts.push(v(1));
    assert_eq!(f.verify(), Err(VerifyError::Missing(v(1))));
    let mut f = ir("block0:\n    %0: u32 = LaneId\n    return\n");
    f.blocks[0].insts.push(v(0));
    assert_eq!(f.verify(), Err(VerifyError::Repeated(v(0))));
    let mut f = ir("block0:\n    %0: u32 = LaneId\n    jump block1\nblock1:\n    return\n");
    f.blocks[1].insts.push(v(0));
    assert_eq!(f.verify(), Err(VerifyError::Repeated(v(0))));
}

#[test]
fn control_flow_graph_blocks_need_terminators_with_real_targets() {
    assert_eq!(
        verify("block0:\n    jump block1\nblock1:\n"),
        Err(VerifyError::Unterminated(b(1)))
    );
    assert_eq!(
        verify("block0:\n    jump block1\n"),
        Err(VerifyError::NoSuchBlock {
            at: Site::Terminator(b(0)),
            block: b(1)
        })
    );
    // The last block that exists is a valid target; one past it is not.
    assert_eq!(
        verify("block0:\n    branch u1 true, block1, block2\nblock1:\n    return\n"),
        Err(VerifyError::NoSuchBlock {
            at: Site::Terminator(b(0)),
            block: b(2)
        })
    );
    assert_eq!(
        verify("block0:\n    branch u1 true, block1, block0\nblock1:\n    return\n"),
        Ok(())
    );
    assert_eq!(verify("block0:\n    unreachable\n"), Ok(()));
}

#[test]
fn structured_functions_place_every_block_once() {
    assert_eq!(
        verify("block0:\nblock1:\nstructure:\n    block0\n    block1\n    return\n"),
        Ok(())
    );
    assert_eq!(
        verify("block0:\nblock1:\nstructure:\n    block0\n    return\n"),
        Err(VerifyError::NotPlaced(b(1)))
    );
    assert_eq!(
        verify("block0:\nstructure:\n    block0\n    block0\n"),
        Err(VerifyError::PlacedTwice(b(0)))
    );
    assert_eq!(
        verify("block0:\nstructure:\n    block0\n    block1\n"),
        Err(VerifyError::NoSuchBlock {
            at: Site::Structure,
            block: b(1)
        })
    );
    assert_eq!(
        verify("block0:\n    return\nstructure:\n    block0\n"),
        Err(VerifyError::TerminatorInStructure(b(0)))
    );
}

#[test]
fn breaks_must_be_inside_a_loop() {
    assert_eq!(
        verify("block0:\nstructure:\n    block0\n    break u1 true\n"),
        Err(VerifyError::BreakOutsideLoop)
    );
    assert_eq!(
        verify("block0:\nstructure:\n    block0\n    if u1 true {\n        break u1 true\n    }\n"),
        Err(VerifyError::BreakOutsideLoop)
    );
    assert_eq!(
        verify(
            "block0:\nstructure:\n    block0\n    loop {\n        if u1 true {\n            \
             break u1 true\n        }\n    } repeat u1 false\n"
        ),
        Ok(())
    );
    // Leaving a loop's body leaves the loop.
    assert_eq!(
        verify(
            "block0:\nstructure:\n    block0\n    loop {\n    } repeat u1 false\n    \
             break u1 true\n"
        ),
        Err(VerifyError::BreakOutsideLoop)
    );
}

/// A tree whose innermost body is at nesting level `depth`, built from `if` (or `loop`) nodes.
fn nested(depth: usize, looped: bool) -> Function {
    let mut nodes = vec![Node::Block(b(0))];
    for _ in 0..depth {
        nodes = if looped {
            vec![Node::Loop {
                body: nodes,
                repeat: Operand::Imm(Imm::U1(false)),
            }]
        } else {
            vec![Node::If {
                cond: Operand::Imm(Imm::U1(true)),
                body: nodes,
            }]
        };
    }
    let mut f = Function::new();
    f.set_structure(Some(nodes));
    f
}

#[test]
fn nesting_is_bounded() {
    for looped in [false, true] {
        assert_eq!(nested(MAX_NESTING, looped).verify(), Ok(()), "{looped}");
        assert_eq!(
            nested(MAX_NESTING + 1, looped).verify(),
            Err(VerifyError::TooDeep),
            "{looped}"
        );
    }
}

/// Uses must be dominated by their definitions: block order alone is not enough.
#[test]
fn dominance_is_checked_not_block_order() {
    // A diamond: block1 and block2 both lead to block3.
    let diamond = |use_of: &str| {
        format!(
            "block0:\n\
             \x20   %0: u32 = LaneId\n\
             \x20   branch u1 true, block1, block2\n\
             block1:\n\
             \x20   %1: u32 = LaneId\n\
             \x20   jump block3\n\
             block2:\n\
             \x20   jump block3\n\
             block3:\n\
             \x20   %2: u32 = IAdd32 {use_of}, u32 1\n\
             \x20   return\n"
        )
    };
    // The entry dominates the merge; block1 comes earlier in the list but does not.
    assert_eq!(verify(&diamond("%0")), Ok(()));
    assert_eq!(
        verify(&diamond("%1")),
        Err(VerifyError::Undefined {
            at: Site::Inst(v(2)),
            operand: v(1)
        })
    );
    // Within a block, the definition must come first.
    assert_eq!(
        verify("block0:\n    %0: u32 = IAdd32 %1, %1\n    %1: u32 = LaneId\n    return\n"),
        Err(VerifyError::Undefined {
            at: Site::Inst(v(0)),
            operand: v(1)
        })
    );
    assert_eq!(
        verify("block0:\n    %0: u32 = IAdd32 %0, u32 1\n    return\n"),
        Err(VerifyError::Undefined {
            at: Site::Inst(v(0)),
            operand: v(0)
        })
    );
    // A later block in the list may dominate an earlier one.
    assert_eq!(
        verify(
            "block0:\n    jump block2\nblock1:\n    %1: u32 = IAdd32 %0, u32 1\n    return\n\
             block2:\n    %0: u32 = LaneId\n    jump block1\n"
        ),
        Ok(())
    );
    // A loop: the header dominates the body, the body does not dominate the header.
    let looped = |header_use: &str| {
        format!(
            "block0:\n\
             \x20   jump block1\n\
             block1:\n\
             \x20   %0: u32 = LaneId\n\
             \x20   %1: u32 = IAdd32 {header_use}, u32 1\n\
             \x20   branch u1 true, block2, block3\n\
             block2:\n\
             \x20   %2: u32 = IAdd32 %0, u32 1\n\
             \x20   jump block1\n\
             block3:\n\
             \x20   return\n"
        )
    };
    assert_eq!(verify(&looped("%0")), Ok(()));
    assert_eq!(
        verify(&looped("%2")),
        Err(VerifyError::Undefined {
            at: Site::Inst(v(1)),
            operand: v(2)
        })
    );
}

/// A use in a block the entry cannot reach is accepted when its value is listed somewhere; a
/// value defined only in such a block cannot be used in a reachable one.
#[test]
fn unreachable_blocks() {
    let text = |use_block: &str| {
        format!(
            "block0:\n    %0: u32 = LaneId\n    {use_block}\nblock1:\n    \
             %1: u32 = IAdd32 %0, u32 1\n    return\n"
        )
    };
    // block1 is unreachable from block0 here; the use of %0 is accepted.
    assert_eq!(verify(&text("return")), Ok(()));
    assert_eq!(verify(&text("jump block1")), Ok(()));
    let mut detached_use = ir(&text("return"));
    if let Some(Inst::Op(op)) = detached_use.inst_mut(v(1)) {
        op.args[0] = Operand::Value(v(5));
    }
    assert_eq!(
        detached_use.verify(),
        Err(VerifyError::Undefined {
            at: Site::Inst(v(1)),
            operand: v(5)
        })
    );
    assert_eq!(
        verify(
            "block0:\n    %0: u32 = IAdd32 %1, u32 1\n    return\nblock1:\n    \
             %1: u32 = LaneId\n    return\n"
        ),
        Err(VerifyError::Undefined {
            at: Site::Inst(v(0)),
            operand: v(1)
        })
    );
}

#[test]
fn branch_conditions_are_defined_u1_values() {
    let text = |cond: &str, def: &str| {
        format!("block0:\n    {def}\n    branch {cond}, block1, block1\nblock1:\n    return\n")
    };
    assert_eq!(verify(&text("%0", "%0: u1 = GetPred p0")), Ok(()));
    assert_eq!(
        verify(&text("%0", "%0: u32 = LaneId")),
        Err(VerifyError::Type {
            at: Site::Terminator(b(0)),
            index: 0,
            wanted: Type::U1,
            found: Type::U32
        })
    );
    assert_eq!(
        verify(&text("%1", "%0: u1 = GetPred p0")),
        Err(VerifyError::Undefined {
            at: Site::Terminator(b(0)),
            operand: v(1)
        })
    );
    assert_eq!(
        verify(&text("%0", "%0 = Barrier")),
        Err(VerifyError::NoResult {
            at: Site::Terminator(b(0)),
            operand: v(0)
        })
    );
}

/// A branch condition defined in a block that does not dominate the branch is undefined there.
#[test]
fn branch_conditions_need_a_dominating_definition() {
    let text = |cond: &str| {
        format!(
            "block0:\n    %0: u1 = GetPred p0\n    branch %0, block1, block2\n\
             block1:\n    %1: u1 = GetPred p1\n    jump block2\n\
             block2:\n    branch {cond}, block3, block3\nblock3:\n    return\n"
        )
    };
    assert_eq!(verify(&text("%0")), Ok(()));
    assert_eq!(
        verify(&text("%1")),
        Err(VerifyError::Undefined {
            at: Site::Terminator(b(2)),
            operand: v(1)
        })
    );
}

#[test]
fn operations_are_checked_against_the_table() {
    assert_eq!(
        verify("block0:\n    %0 = Barrier\n    %1: u32 = IAdd32 %0, u32 1\n    return\n"),
        Err(VerifyError::NoResult {
            at: Site::Inst(v(1)),
            operand: v(0)
        })
    );
    assert_eq!(
        verify("block0:\n    %0: u32 = IAdd32 u32 1\n    return\n"),
        Err(VerifyError::ArgCount {
            at: v(0),
            opcode: Opcode::IAdd32,
            wanted: 2,
            found: 1
        })
    );
    assert_eq!(
        verify("block0:\n    %0: u32 = IAdd32 u32 1, f32 1.0\n    return\n"),
        Err(VerifyError::Type {
            at: Site::Inst(v(0)),
            index: 1,
            wanted: Type::U32,
            found: Type::F32
        })
    );
    // An opaque slot takes anything, and `Identity` passes its argument's type on.
    assert_eq!(
        verify(
            "block0:\n    %0: opaque = Identity f32 1.0\n    %1: f32 = FPAbs32 %0\n    \
             %2: u32x2 = LoadGlobal64 %1\n    return\n"
        ),
        Ok(())
    );
    let mut flags = ir("block0:\n    %0: u32 = IAdd32 u32 1, u32 2\n    return\n");
    if let Some(Inst::Op(op)) = flags.inst_mut(v(0)) {
        op.flags = Flags::Fp(FpControl::default());
    }
    assert_eq!(
        flags.verify(),
        Err(VerifyError::FlagsKind {
            at: v(0),
            opcode: Opcode::IAdd32
        })
    );
    let mut range = ir("block0:\n    %0: u32x4 = ImageRead u32 0, u32 0\n    return\n");
    if let Some(Inst::Op(op)) = range.inst_mut(v(0)) {
        op.flags = Flags::Texture(TextureInfo {
            num_derivatives: TEXTURE_TWO_BIT_MAX + 1,
            ..TextureInfo::default()
        });
    }
    assert_eq!(range.verify(), Err(VerifyError::FlagsRange(v(0))));
}

#[test]
fn phi_checks() {
    // A merge with predecessors block1 and block2.
    let merge = |phi: &str| {
        format!(
            "block0:\n\
             \x20   %0: u32 = LaneId\n\
             \x20   branch u1 true, block1, block2\n\
             block1:\n\
             \x20   %1: u32 = LaneId\n\
             \x20   jump block3\n\
             block2:\n\
             \x20   %2: f32 = BitCastF32U32 u32 0\n\
             \x20   jump block3\n\
             block3:\n\
             \x20   %3: {phi}\n\
             \x20   return\n"
        )
    };
    let at = v(3);
    assert_eq!(
        verify(&merge("u32 = Phi [block1: %1], [block2: %0]")),
        Ok(())
    );
    assert_eq!(
        verify(&merge("u32 = Phi [block2: u32 7], [block1: %1]")),
        Ok(()),
        "any order of predecessors"
    );
    assert_eq!(
        verify(&merge("void = Phi [block1: %1], [block2: %0]")),
        Err(VerifyError::PhiType { at, ty: Type::Void })
    );
    assert_eq!(
        verify(&merge("u32 = Phi [block1: %1], [block9: %0]")),
        Err(VerifyError::NoSuchBlock {
            at: Site::Inst(at),
            block: b(9)
        })
    );
    assert_eq!(
        verify(&merge("u32 = Phi [block1: %1], [block1: %1]")),
        Err(VerifyError::PhiRepeated { at, block: b(1) })
    );
    assert_eq!(
        verify(&merge("u32 = Phi [block1: %1], [block0: %0]")),
        Err(VerifyError::PhiNotPredecessor { at, block: b(0) })
    );
    assert_eq!(
        verify(&merge("u32 = Phi [block1: %1]")),
        Err(VerifyError::PhiMissingPredecessor { at, block: b(2) })
    );
    assert_eq!(
        verify(&merge("u32 = Phi [block1: %1], [block2: %2]")),
        Err(VerifyError::Type {
            at: Site::Inst(at),
            index: 1,
            wanted: Type::U32,
            found: Type::F32
        })
    );
    // An incoming value must be available at the end of its predecessor: %1 is defined in
    // block1, which does not dominate block2.
    assert_eq!(
        verify(&merge("u32 = Phi [block1: %1], [block2: %1]")),
        Err(VerifyError::Undefined {
            at: Site::Inst(at),
            operand: v(1)
        })
    );
    // A loop header's phi may take a value the latch defines later.
    assert_eq!(
        verify(
            "block0:\n    jump block1\nblock1:\n    %0: u32 = Phi [block0: u32 0], [block2: %1]\n    \
             branch u1 true, block2, block3\nblock2:\n    %1: u32 = IAdd32 %0, u32 1\n    \
             jump block1\nblock3:\n    return\n"
        ),
        Ok(())
    );
    // Phis come first in their block.
    assert_eq!(
        verify(
            "block0:\n    jump block1\nblock1:\n    %0: u32 = LaneId\n    \
             %1: u32 = Phi [block0: u32 0]\n    return\n"
        ),
        Err(VerifyError::PhiAfterOperation(v(1)))
    );
    assert_eq!(
        verify(
            "block0:\n    jump block1\nblock1:\n    %1: u32 = Phi [block0: u32 0]\n    \
             %2: u32 = Phi [block0: u32 1]\n    %0: u32 = LaneId\n    return\n"
        ),
        Ok(())
    );
}

#[test]
fn structured_phis_need_listed_values() {
    let text = |phi: &str| {
        format!(
            "block0:\n    %0: u32 = LaneId\nblock1:\n    %1: {phi}\n\
             structure:\n    block0\n    block1\n"
        )
    };
    assert_eq!(verify(&text("u32 = Phi [block0: %0]")), Ok(()));
    assert_eq!(
        verify(&text("u32 = Phi [block0: %5]")),
        Err(VerifyError::Undefined {
            at: Site::Inst(v(1)),
            operand: v(5)
        })
    );
    // An instruction in the arena that no block lists is not a value a phi can take.
    let mut unlisted = ir(&text("u32 = Phi [block0: %0]"));
    unlisted.blocks[0].insts.clear();
    assert_eq!(
        unlisted.verify(),
        Err(VerifyError::Undefined {
            at: Site::Inst(v(1)),
            operand: v(0)
        })
    );
    assert_eq!(
        verify(&text("u32 = Phi [block0: %0], [block0: %0]")),
        Err(VerifyError::PhiRepeated {
            at: v(1),
            block: b(0)
        })
    );
    assert_eq!(
        verify(&text("u32 = Phi [block4: %0]")),
        Err(VerifyError::NoSuchBlock {
            at: Site::Inst(v(1)),
            block: b(4)
        })
    );
    assert_eq!(
        verify(&text("opaque = Phi [block0: %0]")),
        Err(VerifyError::PhiType {
            at: v(1),
            ty: Type::Opaque
        })
    );
    assert_eq!(
        verify(&text("f32 = Phi [block0: %0]")),
        Err(VerifyError::Type {
            at: Site::Inst(v(1)),
            index: 0,
            wanted: Type::F32,
            found: Type::U32
        })
    );
    assert_eq!(
        verify("block0:\n    %0: u32 = LaneId\n    %1: u32 = Phi\nstructure:\n    block0\n"),
        Err(VerifyError::PhiAfterOperation(v(1)))
    );
}

/// Values defined in an `if` or loop body go out of scope when the body ends; a loop's repeat
/// condition sees its body's values.
#[test]
fn structured_scopes() {
    let text = |tree: &str| {
        format!(
            "block0:\n    %0: u1 = GetPred p0\nblock1:\n    %1: u1 = GetPred p1\n\
             block2:\n    %2: u1 = LogicalNot %1\nstructure:\n{tree}"
        )
    };
    let undefined = |at, operand| {
        Err(VerifyError::Undefined {
            at,
            operand: v(operand),
        })
    };
    assert_eq!(
        verify(&text(
            "    block0\n    if %0 {\n        block1\n        if %1 {\n            block2\n        }\n    }\n"
        )),
        Ok(())
    );
    assert_eq!(
        verify(&text(
            "    block0\n    if %0 {\n        block1\n    }\n    block2\n"
        )),
        undefined(Site::Inst(v(2)), 1)
    );
    assert_eq!(
        verify(&text(
            "    block0\n    if %0 {\n        block1\n    }\n    if %1 {\n        block2\n    }\n"
        )),
        undefined(Site::Structure, 1)
    );
    assert_eq!(
        verify(&text(
            "    if %0 {\n        block0\n    }\n    block1\n    block2\n"
        )),
        undefined(Site::Structure, 0)
    );
    assert_eq!(
        verify(&text(
            "    block0\n    loop {\n        block1\n        break %1\n        block2\n    } repeat %2\n"
        )),
        Ok(())
    );
    assert_eq!(
        verify(&text(
            "    block0\n    loop {\n        block1\n    } repeat %1\n    block2\n"
        )),
        undefined(Site::Inst(v(2)), 1)
    );
    assert_eq!(
        verify(&text(
            "    block0\n    loop {\n        block1\n        break %2\n        block2\n    } repeat %0\n"
        )),
        undefined(Site::Structure, 2)
    );
    assert_eq!(
        verify(&text(
            "    block0\n    block1\n    block2\n    if u1 true {\n    }\n    break %0\n"
        )),
        Err(VerifyError::BreakOutsideLoop)
    );
    assert_eq!(
        verify(&text(
            "    block0\n    if %0 {\n    }\n    block1\n    block2\n    loop {\n    } repeat %2\n"
        )),
        Ok(())
    );
    // A condition of the wrong type.
    assert_eq!(
        verify("block0:\n    %0: u32 = LaneId\nstructure:\n    block0\n    if %0 {\n    }\n"),
        Err(VerifyError::Type {
            at: Site::Structure,
            index: 0,
            wanted: Type::U1,
            found: Type::U32
        })
    );
    assert_eq!(
        verify("block0:\n    %0 = Barrier\nstructure:\n    block0\n    if %0 {\n    }\n"),
        Err(VerifyError::NoResult {
            at: Site::Structure,
            operand: v(0)
        })
    );
}

/// The documented order, each case with two faults at once: the earlier check wins.
#[test]
fn checks_run_in_the_documented_order() {
    // Step 2 before step 3: a missing id and an unterminated block.
    let mut f = ir("block0:\n");
    f.blocks[0].insts.push(v(3));
    assert_eq!(f.verify(), Err(VerifyError::Missing(v(3))));
    // Step 3 before step 4: an unterminated block and a type error.
    assert_eq!(
        verify("block0:\n    %0: f32 = FPAbs32 u32 1\n"),
        Err(VerifyError::Unterminated(b(0)))
    );
    // Within step 3, the tree before the per-block checks.
    assert_eq!(
        verify("block0:\n    return\nstructure:\n    block0\n    block0\n"),
        Err(VerifyError::PlacedTwice(b(0)))
    );
    // Per block in arena order: block0's terminator before block1's placement.
    assert_eq!(
        verify("block0:\n    return\nblock1:\nstructure:\n    block0\n"),
        Err(VerifyError::TerminatorInStructure(b(0)))
    );
    // 4a before 4b: an undefined argument and a wrong count.
    assert_eq!(
        verify("block0:\n    %0: u32 = IAdd32 %5\n    return\n"),
        Err(VerifyError::Undefined {
            at: Site::Inst(v(0)),
            operand: v(5)
        })
    );
    // 4b before 4c: a wrong count and wrong flags.
    let mut f = ir("block0:\n    %0: u32 = IAdd32 u32 1\n    return\n");
    if let Some(Inst::Op(op)) = f.inst_mut(v(0)) {
        op.flags = Flags::Fp(FpControl::default());
    }
    assert_eq!(
        f.verify(),
        Err(VerifyError::ArgCount {
            at: v(0),
            opcode: Opcode::IAdd32,
            wanted: 2,
            found: 1
        })
    );
    // 4c before 4d: wrong flags and a wrong type.
    let mut f = ir("block0:\n    %0: u32 = IAdd32 u32 1, f32 1.0\n    return\n");
    if let Some(Inst::Op(op)) = f.inst_mut(v(0)) {
        op.flags = Flags::Fp(FpControl::default());
    }
    assert_eq!(
        f.verify(),
        Err(VerifyError::FlagsKind {
            at: v(0),
            opcode: Opcode::IAdd32
        })
    );
    // 4c: the kind before the ranges.
    let mut f = ir("block0:\n    %0: u32 = IAdd32 u32 1, u32 2\n    return\n");
    if let Some(Inst::Op(op)) = f.inst_mut(v(0)) {
        op.flags = Flags::Texture(TextureInfo {
            gather_component: TEXTURE_TWO_BIT_MAX + 1,
            ..TextureInfo::default()
        });
    }
    assert_eq!(
        f.verify(),
        Err(VerifyError::FlagsKind {
            at: v(0),
            opcode: Opcode::IAdd32
        })
    );
    // 4d before 4e: a wrong type and a reserved name.
    assert_eq!(
        verify("block0:\n    %0 = SetRegister rz, f32 1.0\n    return\n"),
        Err(VerifyError::Type {
            at: Site::Inst(v(0)),
            index: 1,
            wanted: Type::U32,
            found: Type::F32
        })
    );
    // 4d in argument order: two wrong types.
    assert_eq!(
        verify("block0:\n    %0: u32 = IAdd32 f32 1.0, u1 true\n    return\n"),
        Err(VerifyError::Type {
            at: Site::Inst(v(0)),
            index: 0,
            wanted: Type::U32,
            found: Type::F32
        })
    );
    // Instructions before the terminator's condition.
    assert_eq!(
        verify(
            "block0:\n    %0: f32 = FPAbs32 u32 1\n    branch u32 0, block1, block1\n\
             block1:\n    return\n"
        ),
        Err(VerifyError::Type {
            at: Site::Inst(v(0)),
            index: 0,
            wanted: Type::F32,
            found: Type::U32
        })
    );
}

/// The documented order for phis, each case with two faults at once.
#[test]
fn phi_checks_run_in_the_documented_order() {
    // A phi: 4a (placement) before 4b (type).
    assert_eq!(
        verify("block0:\n    %0: u32 = LaneId\n    %1: void = Phi\n    return\n"),
        Err(VerifyError::PhiAfterOperation(v(1)))
    );
    // A phi's incoming pairs in order, and repeated before not-a-predecessor.
    assert_eq!(
        verify(
            "block0:\n    jump block1\nblock1:\n    %0: u32 = Phi [block2: u32 0], [block9: u32 0]\n    \
             return\nblock2:\n    return\n"
        ),
        Err(VerifyError::PhiNotPredecessor {
            at: v(0),
            block: b(2)
        })
    );
    assert_eq!(
        verify(
            "block0:\n    jump block1\nblock1:\n    %0: u32 = Phi [block2: u32 0], [block2: u32 0]\n    \
             return\nblock2:\n    return\n"
        ),
        Err(VerifyError::PhiNotPredecessor {
            at: v(0),
            block: b(2)
        })
    );
    assert_eq!(
        verify(
            "block0:\n    jump block1\nblock1:\n    %0: u32 = Phi [block0: u32 0], [block0: u32 0]\n    \
             return\n"
        ),
        Err(VerifyError::PhiRepeated {
            at: v(0),
            block: b(0)
        })
    );
    // A phi's incoming pairs before the missing predecessor.
    assert_eq!(
        verify(
            "block0:\n    branch u1 true, block1, block2\nblock1:\n    jump block2\nblock2:\n    \
             %0: u32 = Phi [block0: f32 0.0]\n    return\n"
        ),
        Err(VerifyError::Type {
            at: Site::Inst(v(0)),
            index: 0,
            wanted: Type::U32,
            found: Type::F32
        })
    );
}
