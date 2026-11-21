//! Step 5's tests: the blocks that can run last for every kind of node, each found by trying
//! every block of the function as a phi's one incoming block; the scope of incoming values;
//! the documented order with two faults at once; and what step 5 does not check.

use std::fmt::Write as _;

use super::super::super::*;
use crate::interp::Machine;

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

/// The blocks that may be the incoming block of a phi in block `phi`, in a structured function
/// of `count` empty blocks with the tree `tree`: each block in turn is the phi's one incoming
/// block (of an immediate), and the verifier must accept it or refuse it with
/// `PhiNotPredecessor` naming it.
fn can_run_last(count: u32, tree: &str, phi: u32) -> Vec<u32> {
    let mut found = Vec::new();
    for from in 0..count {
        let mut text = String::new();
        for index in 0..count {
            writeln!(text, "block{index}:").expect("writes");
            if index == phi {
                writeln!(text, "    %0: u32 = Phi [block{from}: u32 0]").expect("writes");
            }
        }
        text.push_str("structure:\n");
        text.push_str(tree);
        let result = ir(&text).verify();
        if result.is_ok() {
            found.push(from);
        } else {
            assert_eq!(
                result,
                Err(VerifyError::PhiNotPredecessor {
                    at: v(0),
                    block: b(from)
                }),
                "{text}"
            );
        }
    }
    found
}

/// One case of [`CASES`]: what it shows, the block count, the tree, and each block named with
/// the blocks that can run last before it.
type Case = (
    &'static str,
    u32,
    &'static str,
    &'static [(u32, &'static [u32])],
);

/// Each kind of node, with the blocks that can run last before each block named, written out
/// from the rule in the module docs (and the interpreter's: the block run last).
#[test]
fn the_blocks_that_can_run_last() {
    for &(name, count, tree, expected) in CASES {
        for &(phi, blocks) in expected {
            assert_eq!(can_run_last(count, tree, phi), blocks, "{name}: block{phi}");
        }
    }
}

/// The cases of [`the_blocks_that_can_run_last`].
const CASES: &[Case] = &[
    (
        "a sequence; the first block follows nothing",
        3,
        "    block0\n    block1\n    block2\n",
        &[(0, &[]), (1, &[0]), (2, &[1])],
    ),
    (
        "an if: the block after it follows the body's end and the block before it",
        3,
        "    block0\n    if u1 true {\n        block1\n    }\n    block2\n",
        &[(1, &[0]), (2, &[0, 1])],
    ),
    (
        "an immediate false condition removes no edge",
        3,
        "    block0\n    if u1 false {\n        block1\n    }\n    block2\n",
        &[(1, &[0]), (2, &[0, 1])],
    ),
    (
        "an empty if body",
        2,
        "    block0\n    if u1 true {\n    }\n    block1\n",
        &[(1, &[0])],
    ),
    (
        "nested ifs that end together",
        4,
        "    block0\n    if u1 true {\n        block1\n        if u1 true {\n            \
             block2\n        }\n    }\n    block3\n",
        &[(1, &[0]), (2, &[1]), (3, &[0, 1, 2])],
    ),
    (
        "a loop: its first block also follows the body's end, and an immediate false \
             repeat removes no edge",
        4,
        "    block0\n    loop {\n        block1\n        block2\n    } repeat u1 false\n    \
             block3\n",
        &[(1, &[0, 2]), (2, &[1]), (3, &[2])],
    ),
    (
        "a loop of one block follows itself",
        3,
        "    block0\n    loop {\n        block1\n    } repeat u1 true\n    block2\n",
        &[(1, &[0, 1]), (2, &[1])],
    ),
    (
        "an empty loop body",
        2,
        "    block0\n    loop {\n    } repeat u1 true\n    block1\n",
        &[(1, &[0])],
    ),
    (
        "a break in an if: its block reaches the loop's exit and, the condition being \
             possibly false, the rest of the body",
        5,
        "    block0\n    loop {\n        block1\n        if u1 true {\n            block2\n            \
             break u1 true\n        }\n        block3\n    } repeat u1 true\n    block4\n",
        &[(1, &[0, 3]), (2, &[1]), (3, &[1, 2]), (4, &[2, 3])],
    ),
    (
        "a break first in a loop body passes on what ran before the body",
        3,
        "    block0\n    loop {\n        break u1 true\n        block1\n    } repeat u1 true\n    \
             block2\n",
        &[(1, &[0, 1]), (2, &[0, 1])],
    ),
    (
        "a break leaves the innermost loop only",
        6,
        "    block0\n    loop {\n        block1\n        loop {\n            block2\n            \
             break u1 true\n            block3\n        } repeat u1 true\n        block4\n    } \
             repeat u1 true\n    block5\n",
        &[
            (1, &[0, 4]),
            (2, &[1, 3]),
            (3, &[2]),
            (4, &[2, 3]),
            (5, &[4]),
        ],
    ),
    (
        "nothing runs last after a return",
        3,
        "    block0\n    if u1 true {\n        block1\n        return\n    }\n    block2\n",
        &[(1, &[0]), (2, &[0])],
    ),
    (
        "nothing runs last after an unreachable",
        2,
        "    block0\n    unreachable\n    block1\n",
        &[(1, &[])],
    ),
    (
        "a loop body that starts with an if: the body's end reaches the if's body and \
             the block after it",
        4,
        "    block0\n    loop {\n        if u1 true {\n            block1\n        }\n        \
             block2\n    } repeat u1 true\n    block3\n",
        &[(1, &[0, 2]), (2, &[0, 1, 2]), (3, &[2])],
    ),
];

/// The interpreter runs a loop body that starts with an `if` as the module docs say: on the
/// second pass, block2 ran last before block2 (the `if` is skipped), which the rewrite's walk
/// would not list (`opt::ssa::tree`; it never meets such a tree).
#[test]
fn a_loop_body_that_starts_with_an_if_takes_the_back_edge() {
    let function = ir("block0:
    %0 = SetRegister r0, u32 0
block1:
block2:
    %1: u32 = Phi [block0: u32 10], [block1: u32 20], [block2: u32 30]
    %2: u32 = GetRegister r0
    %3: u32 = IAdd32 %2, u32 1
    %4 = SetRegister r0, %3
    %5 = SetRegister r1, %1
    %6: u1 = ULessThan %3, u32 2
structure:
    block0
    loop {
        if u1 false {
            block1
        }
        block2
    } repeat %6
    return
");
    assert_eq!(function.verify(), Ok(()));
    let mut machine = Machine::new();
    machine.run(&function).expect("runs");
    assert_eq!((machine.regs[0], machine.regs[1]), (2, 30));
}

/// A structured function whose block3 holds a phi taking `incoming`, with block1 in an `if`.
fn after_an_if(incoming: &str) -> String {
    format!(
        "block0:
    %0: u32 = IAdd32 u32 1, u32 2
block1:
    %1: u32 = IAdd32 u32 3, u32 4
block2:
    %2: u32 = IAdd32 u32 5, u32 6
block3:
    %3: u32 = Phi {incoming}
structure:
    block0
    if u1 true {{
        block1
    }}
    block2
    block3
"
    )
}

/// A structured function with a header phi in block1 taking `back` from the end of the loop
/// (block3), a loop inside the body holding block2, and a phi in block4, after the loop,
/// taking `exit` from block3. block3 defines %3.
fn around_a_loop(first: &str, back: &str, exit: &str) -> String {
    format!(
        "block0:
    %0: u32 = IAdd32 u32 1, u32 2
block1:
    %1: u32 = Phi [block0: {first}], [block3: {back}]
block2:
    %2: u32 = IAdd32 %1, u32 1
block3:
    %3: u32 = IAdd32 u32 7, u32 8
block4:
    %4: u32 = Phi [block3: {exit}]
structure:
    block0
    loop {{
        block1
        loop {{
            block2
        }} repeat u1 false
        block3
    }} repeat u1 true
    block4
"
    )
}

#[test]
fn a_structured_phis_value_is_in_scope_at_the_end_of_its_block() {
    // A block's own value, a value of a body that encloses it, and an immediate.
    for incoming in ["[block2: %2]", "[block2: %0]", "[block2: u32 9]"] {
        assert_eq!(verify(&after_an_if(incoming)), Ok(()), "{incoming}");
    }
    let out_of_scope = |at, operand, block| {
        Err(VerifyError::PhiOutOfScope {
            at: v(at),
            operand: v(operand),
            block: b(block),
        })
    };
    // %1's `if` body has ended at block2's end.
    assert_eq!(verify(&after_an_if("[block2: %1]")), out_of_scope(3, 1, 2));
    // %3 is the phi itself, which comes later in the tree than block2.
    assert_eq!(verify(&after_an_if("[block2: %3]")), out_of_scope(3, 3, 2));

    for (first, back, exit) in [
        ("%0", "%1", "%1"),
        ("%0", "%0", "%3"),
        // block3's own value, taken from block3 along the back edge.
        ("u32 0", "%3", "%0"),
    ] {
        assert_eq!(
            verify(&around_a_loop(first, back, exit)),
            Ok(()),
            "{first} {back} {exit}"
        );
    }
    // %2's inner loop body has ended at block3's end, along the back edge and out of the loop.
    assert_eq!(
        verify(&around_a_loop("%0", "%2", "%1")),
        out_of_scope(1, 2, 3)
    );
    assert_eq!(
        verify(&around_a_loop("%0", "%1", "%2")),
        out_of_scope(4, 2, 3)
    );
    // %3 is not yet computed at block0's end.
    assert_eq!(
        verify(&around_a_loop("%3", "%1", "%1")),
        out_of_scope(1, 3, 0)
    );
    // Sibling bodies: %1's `if` body opens before block3's and closes before it opens, so %1
    // is out of scope at block3's end though block1 comes earlier in the tree.
    let siblings = |value: &str| {
        format!(
            "block0:
    %0: u32 = IAdd32 u32 1, u32 2
block1:
    %1: u32 = IAdd32 u32 3, u32 4
block2:
block3:
block4:
    %2: u32 = Phi [block2: u32 0], [block3: {value}]
structure:
    block0
    if u1 true {{
        block1
    }}
    block2
    if u1 true {{
        block3
    }}
    block4
"
        )
    };
    assert_eq!(verify(&siblings("%0")), Ok(()));
    assert_eq!(verify(&siblings("%1")), out_of_scope(2, 1, 3));
}

/// The two step-5 errors read in plain words.
#[test]
fn step_5_errors_read_plainly() {
    assert_eq!(
        verify(&after_an_if("[block1: %1]"))
            .err()
            .map(|e| e.to_string()),
        Some("%3 has an incoming value from block1, which is not a predecessor".to_owned())
    );
    assert_eq!(
        verify(&after_an_if("[block2: %1]"))
            .err()
            .map(|e| e.to_string()),
        Some("%3 takes %1 from block2, where it is not in scope".to_owned())
    );
}

/// The documented order, each case with two faults at once: the earlier check wins.
#[test]
fn step_5_runs_in_the_documented_order() {
    let not_predecessor = |at, block| {
        Err(VerifyError::PhiNotPredecessor {
            at: v(at),
            block: b(block),
        })
    };
    let out_of_scope = |at, operand, block| {
        Err(VerifyError::PhiOutOfScope {
            at: v(at),
            operand: v(operand),
            block: b(block),
        })
    };
    // Step 4 before step 5, in one phi: its type, and a block that cannot run last.
    assert_eq!(
        verify(&after_an_if("[block1: %1]").replace("%3: u32 = Phi", "%3: f32 = Phi")),
        Err(VerifyError::Type {
            at: Site::Inst(v(3)),
            index: 0,
            wanted: Type::F32,
            found: Type::U32
        })
    );
    // Step 4, over the whole tree, before step 5: block3's phi cannot take from block1, and
    // block4, later in the tree, has an operation of the wrong type.
    assert_eq!(
        verify(
            "block0:
block1:
block2:
block3:
    %0: u32 = Phi [block1: u32 0]
block4:
    %1: f32 = FPAbs32 u32 1
structure:
    block0
    if u1 true {
        block1
    }
    block2
    block3
    block4
"
        ),
        Err(VerifyError::Type {
            at: Site::Inst(v(1)),
            index: 0,
            wanted: Type::F32,
            found: Type::U32
        })
    );
    // In one pair, (a) before (b): block0 cannot run last before block3, and %1 is not yet
    // computed at its end.
    assert_eq!(verify(&after_an_if("[block0: %1]")), not_predecessor(3, 0));
    // A phi's pairs in order.
    assert_eq!(
        verify(&after_an_if("[block2: %1], [block0: u32 0]")),
        out_of_scope(3, 1, 2)
    );
    assert_eq!(
        verify(&after_an_if("[block0: u32 0], [block2: %1]")),
        not_predecessor(3, 0)
    );
    // A later pair is checked too when the first is sound (a scope fault in a later pair is
    // `around_a_loop("%0", "%2", "%1")` above).
    assert_eq!(
        verify(&after_an_if("[block2: %2], [block0: u32 0]")),
        not_predecessor(3, 0)
    );
    // A block's phis in order.
    let two_phis = |first: &str, second: &str| {
        after_an_if(first).replace(
            "structure:",
            &format!("    %4: u32 = Phi {second}\nstructure:"),
        )
    };
    // The second phi goes after block3's first, so it is listed there.
    let text = two_phis("[block2: %1]", "[block0: u32 0]");
    assert!(
        text.contains("%3: u32 = Phi [block2: %1]\n    %4: u32 = Phi"),
        "{text}"
    );
    assert_eq!(verify(&text), out_of_scope(3, 1, 2));
    assert_eq!(
        verify(&two_phis("[block0: u32 0]", "[block2: %1]")),
        not_predecessor(3, 0)
    );
    // A later phi is checked too when the first is sound.
    assert_eq!(
        verify(&two_phis("[block2: %2]", "[block0: u32 0]")),
        not_predecessor(4, 0)
    );
    assert_eq!(
        verify(&two_phis("[block2: %2]", "[block2: %1]")),
        out_of_scope(4, 1, 2)
    );
    // Blocks in tree order, not arena order: the tree places block2 before block1, and each
    // holds a phi naming a block that cannot run last before it.
    let tree_order = "block0:
block1:
    %0: u32 = Phi [block0: u32 0]
block2:
    %1: u32 = Phi [block1: u32 0]
structure:
    block0
    block2
    block1
";
    assert_eq!(verify(tree_order), not_predecessor(1, 1));
}

/// Step 5 does not check that every block that can run last has a pair: a phi in block2 with
/// no pair from block1 (or none at all) is accepted, and the interpreter refuses it when block1
/// does run last (`PhiNoIncoming`).
#[test]
fn a_missing_predecessor_is_not_checked() {
    let text = |incoming: &str| {
        format!(
            "block0:
block1:
block2:
    %0: u32 = Phi{incoming}
    %1 = SetRegister r0, %0
structure:
    block0
    if u1 true {{
        block1
    }}
    block2
"
        )
    };
    for incoming in [" [block0: u32 1]", ""] {
        let function = ir(&text(incoming));
        assert_eq!(function.verify(), Ok(()), "{incoming:?}");
        assert_eq!(
            Machine::new().run(&function).err().map(|e| e.to_string()),
            Some("%0 has no incoming value from block1".to_owned()),
            "{incoming:?}"
        );
    }
}

/// Each phi block's walk back starts from fresh marks: block1's walk marks block0, and block3,
/// later in the tree, names block0, which only that earlier walk reached. The marks are
/// stamped per block, so block3 sees block0 unmarked and refuses it; one stamp shared by every
/// block would leave block0 marked and accept the plain sequence.
#[test]
fn a_later_phi_block_does_not_see_an_earlier_blocks_marks() {
    assert_eq!(
        verify(
            "block0:
block1:
    %0: u32 = Phi [block0: u32 0]
block2:
block3:
    %1: u32 = Phi [block0: u32 0]
structure:
    block0
    block1
    block2
    block3
"
        ),
        Err(VerifyError::PhiNotPredecessor {
            at: v(1),
            block: b(0)
        })
    );
}
