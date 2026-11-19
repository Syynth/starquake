//! The input checks of the module docs' "What it takes", each with its exact error and
//! words, and their order with two faults present at once.

use super::{parse, structured};
use crate::ir::{BlockId, Function, Node, Value, VerifyError};
use crate::structure::{Invariant, MAX_STRUCTURE_WORK, StructureError, structure};

/// The error of structuring `text`.
fn refusal(text: &str) -> StructureError {
    structure(parse(text), MAX_STRUCTURE_WORK).expect_err("refused")
}

/// Block `n`.
fn block(n: u32) -> BlockId {
    parse(
        &(0..=n)
            .map(|i| format!("block{i}:\n    return\n"))
            .collect::<Vec<_>>()
            .concat(),
    )
    .blocks()
    .nth(usize::try_from(n).expect("small"))
    .expect("the block")
    .0
}

/// Value `n`.
fn value(n: u32) -> Value {
    let text = (0..=n)
        .map(|i| format!("    %{i} = SetRegister r1, u32 {i}\n"))
        .collect::<Vec<_>>()
        .concat();
    parse(&format!("block0:\n{text}    return\n"))
        .listed()
        .nth(usize::try_from(n).expect("small"))
        .expect("the value")
        .0
}

#[test]
fn a_structured_function_is_refused_before_it_is_verified() {
    let mut function = structured("block0:\n    return\n").function;
    assert_eq!(
        structure(function.clone(), MAX_STRUCTURE_WORK),
        Err(StructureError::AlreadyStructured)
    );
    // Also malformed (a block placed twice): still refused as structured first.
    function.set_structure(Some(vec![Node::Block(block(0)), Node::Block(block(0))]));
    assert_eq!(function.verify(), Err(VerifyError::PlacedTwice(block(0))));
    assert_eq!(
        structure(function, MAX_STRUCTURE_WORK),
        Err(StructureError::AlreadyStructured)
    );
}

#[test]
fn a_malformed_function_is_refused() {
    // Unterminated, and with a phi: verification comes first.
    let mut function = Function::new();
    assert_eq!(
        structure(function.clone(), MAX_STRUCTURE_WORK),
        Err(StructureError::Invalid(VerifyError::Unterminated(block(0))))
    );
    let text = "block0:\n    jump block1\nblock1:\n    %0: u32 = Phi [block0: u32 0x1]\n";
    function = parse(text);
    assert_eq!(
        structure(function, MAX_STRUCTURE_WORK),
        Err(StructureError::Invalid(VerifyError::Unterminated(block(1))))
    );
}

#[test]
fn phis_are_refused() {
    let text = "block0:
    jump block1
block1:
    %0: u32 = Phi [block0: u32 0x1]
    return
";
    assert_eq!(refusal(text), StructureError::Phi(value(0)));
}

#[test]
fn cross_block_values_are_refused() {
    // An argument from another block.
    let argument = "block0:
    %0 = GetRegister r1
    jump block1
block1:
    %1 = SetRegister r2, %0
    return
";
    assert_eq!(
        refusal(argument),
        StructureError::CrossBlock {
            block: block(1),
            value: value(0)
        }
    );
    // A branch condition from another block.
    let condition = "block0:
    %0 = GetPred p0
    jump block1
block1:
    branch %0, block2, block2
block2:
    return
";
    assert_eq!(
        refusal(condition),
        StructureError::CrossBlock {
            block: block(1),
            value: value(0)
        }
    );
}

#[test]
fn values_of_the_same_block_are_accepted() {
    let text = "block0:
    %0 = GetRegister r1
    %1 = SetRegister r2, %0
    %2 = GetPred p0
    branch %2, block1, block1
block1:
    return
";
    assert!(structure(parse(text), MAX_STRUCTURE_WORK).is_ok());
}

#[test]
fn the_checks_run_block_by_block_in_order() {
    // block1 has a phi, block2 a cross-block use: block1's fault comes first.
    let phi_first = "block0:
    %0 = GetRegister r1
    jump block1
block1:
    %1: u32 = Phi [block0: u32 0x1]
    jump block2
block2:
    %2 = SetRegister r2, %0
    return
";
    assert_eq!(refusal(phi_first), StructureError::Phi(value(1)));
    // The other way round: block1's cross-block use comes before block2's phi.
    let cross_first = "block0:
    %0 = GetRegister r1
    jump block1
block1:
    %1 = SetRegister r2, %0
    jump block2
block2:
    %2: u32 = Phi [block1: u32 0x1]
    return
";
    assert_eq!(
        refusal(cross_first),
        StructureError::CrossBlock {
            block: block(1),
            value: value(0)
        }
    );
    // Within a block, instructions come before the terminator: the argument from block0
    // (%0) is reported, not the condition (%1).
    let instruction_first = "block0:
    %0 = GetRegister r1
    %1 = GetPred p0
    jump block1
block1:
    %2 = SetRegister r2, %0
    branch %1, block2, block2
block2:
    return
";
    assert_eq!(
        refusal(instruction_first),
        StructureError::CrossBlock {
            block: block(1),
            value: value(0)
        }
    );
}

#[test]
fn errors_read_as_plain_words() {
    let words = [
        (
            StructureError::AlreadyStructured,
            "the function is structured already",
        ),
        (
            StructureError::Invalid(VerifyError::Unterminated(block(0))),
            "the function is malformed: block0 has no terminator",
        ),
        (
            StructureError::Phi(value(3)),
            "%3 is a phi, and the structurizer takes a function before SSA",
        ),
        (
            StructureError::CrossBlock {
                block: block(2),
                value: value(1),
            },
            "block2 uses %1, which another block defines",
        ),
        (
            StructureError::LiftOverBreak,
            "a goto would be lifted over a break, which the structurizer does not support",
        ),
        (
            StructureError::Ir(crate::ir::BuildError::Full),
            "the IR cannot hold the structured function: the function has more than 4294967295 \
             instructions or blocks",
        ),
        (
            StructureError::Internal(Invariant::Parents),
            "the structurizer broke an invariant of its own: a statement's parents do not lead \
             to the root",
        ),
        (
            StructureError::Internal(Invariant::Siblings),
            "the structurizer broke an invariant of its own: a range of statements is not a run \
             of siblings",
        ),
        (
            StructureError::Internal(Invariant::Movement),
            "the structurizer broke an invariant of its own: a goto cannot move from where it is",
        ),
        (
            StructureError::Internal(Invariant::Kind),
            "the structurizer broke an invariant of its own: a statement is not of the kind \
             expected",
        ),
        (
            StructureError::Internal(Invariant::Unterminated(block(1))),
            "the structurizer broke an invariant of its own: block1 has no terminator",
        ),
        (
            StructureError::Internal(Invariant::Output(VerifyError::NotPlaced(block(1)))),
            "the structurizer broke an invariant of its own: the structured function does not \
             verify: block1 is not placed in the structure",
        ),
    ];
    for (error, text) in words {
        assert_eq!(error.to_string(), text);
    }
}
