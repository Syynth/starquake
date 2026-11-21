//! The SSA rewrite on structured functions (slice 8b): predecessors taken from the tree, the
//! blocks it adds where paths meet, values carried out of a body by a phi, the structurizer's
//! goto and condition variables turned into values, and the refusal of a structured phi.

use super::{check_phi_scope, error, parse, ran, value};
use crate::interp::Machine;
use crate::ir::{Function, VerifyError};
use crate::opt::{Pass, PassError, optimize, run};
use crate::structure::{MAX_STRUCTURE_WORK, structure};

/// The machine a test starts from: constant buffer 0 holds `words`, little-endian.
fn machine(words: &[u32]) -> Machine {
    let mut machine = Machine::new();
    let bytes = words.iter().flat_map(|word| word.to_le_bytes()).collect();
    machine.cbufs.insert(0, bytes);
    machine
}

/// Checks that the rewrite of `text` keeps each phi's incoming values in scope
/// ([`check_phi_scope`]) and leaves the same machine as `text` itself from each of `starts`,
/// and returns the rewritten function.
fn rewrite_keeps_state(text: &str, starts: &[Machine]) -> Function {
    let before = parse(text);
    before.verify().expect("the input verifies");
    let mut rewritten = before.clone();
    run(&mut rewritten, Pass::SsaRewrite).expect("rewrites");
    check_phi_scope(&rewritten).expect("phi values in scope");
    for start in starts {
        assert_eq!(ran(&rewritten, start), ran(&before, start), "{rewritten}");
    }
    rewritten
}

/// An `if` that writes r0 on one side: the block after it merges the two values.
const DIAMOND: &str = "block0:
    %0: u32 = GetCbufU32 u32 0, u32 0
    %1: u1 = INotEqual %0, u32 0
    %2 = SetRegister r0, u32 2
block1:
    %3 = SetRegister r0, u32 1
block2:
    %4: u32 = GetRegister r0
    %5 = SetRegister r1, %4
structure:
    block0
    if %1 {
        block1
    }
    block2
    return
";

#[test]
fn a_structured_if_merges_with_a_phi() {
    let out = rewrite_keeps_state(DIAMOND, &[machine(&[0]), machine(&[1])]);
    for (word, r1) in [(0, 2), (1, 1)] {
        assert_eq!(ran(&out, &machine(&[word])).regs[1], r1);
    }
    insta::assert_snapshot!("structured_diamond", out.to_string());
}

/// A loop that counts r0 up to a constant buffer word, then copies it to r1 after the loop.
const COUNT: &str = "block0:
    %0 = SetRegister r0, u32 0
    %1: u32 = GetCbufU32 u32 0, u32 0
block1:
    %2: u32 = GetRegister r0
    %3: u32 = IAdd32 %2, u32 1
    %4 = SetRegister r0, %3
    %5: u1 = ULessThan %3, %1
block2:
    %6: u32 = GetRegister r0
    %7 = SetRegister r1, %6
structure:
    block0
    loop {
        block1
    } repeat %5
    block2
    return
";

#[test]
fn a_loop_carries_its_value_in_a_header_phi_and_out_in_a_phi() {
    let starts = [machine(&[0]), machine(&[1]), machine(&[5])];
    let out = rewrite_keeps_state(COUNT, &starts);
    for (word, r1) in [(0, 1), (1, 1), (5, 5)] {
        assert_eq!(ran(&out, &machine(&[word])).regs[1], r1);
    }
    let text = out.to_string();
    // The header merges the entry's 0 and the back edge's %3.
    assert!(
        text.contains("Phi [block0: u32 0x0], [block1: %3]"),
        "{text}"
    );
    // block2's one predecessor is block1, inside the loop body: %3 is out of scope after
    // the loop, so a phi of one operand carries it out.
    assert!(
        text.contains("block2:\n    %9: u32 = Phi [block1: %3]\n"),
        "{text}"
    );
    insta::assert_snapshot!("structured_count", text);
}

/// A loop left by a `break` after r0 reaches the word, and by its repeat condition at r2's.
const BREAK: &str = "block0:
    %0 = SetRegister r0, u32 0
    %1: u32 = GetCbufU32 u32 0, u32 0
    %2: u32 = GetCbufU32 u32 0, u32 4
block1:
    %3: u32 = GetRegister r0
    %4: u32 = IAdd32 %3, u32 1
    %5 = SetRegister r0, %4
    %6: u1 = UGreaterThanEqual %4, %1
block2:
    %7 = SetRegister r2, %4
    %8: u1 = ULessThan %4, %2
block3:
    %9: u32 = GetRegister r0
    %10 = SetRegister r1, %9
structure:
    block0
    loop {
        block1
        break %6
        block2
    } repeat %8
    block3
    return
";

#[test]
fn a_break_and_the_repeat_both_reach_the_block_after_the_loop() {
    let starts = [
        machine(&[3, 10]),
        machine(&[10, 3]),
        machine(&[0, 0]),
        machine(&[2, 2]),
    ];
    let out = rewrite_keeps_state(BREAK, &starts);
    for (words, r1) in [([3, 10], 3), ([10, 3], 3), ([0, 0], 1), ([2, 2], 2)] {
        assert_eq!(ran(&out, &machine(&words)).regs[1], r1);
    }
    // Both ways out carry %4, which is out of scope after the loop: the phi is kept.
    let text = out.to_string();
    assert!(text.contains("Phi [block1: %4], [block2: %4]"), "{text}");
}

#[test]
fn a_value_from_before_the_body_needs_no_phi_after_it() {
    // r1 is written before the `if` and only read inside and after it: the merge's phi of r1
    // is trivial and stands for block0's value, which is in scope there.
    let text = "block0:
    %0 = SetRegister r1, u32 9
    %6: u32 = GetCbufU32 u32 0, u32 0
    %1: u1 = INotEqual %6, u32 0
block1:
    %2: u32 = GetRegister r1
    %3 = SetRegister r2, %2
block2:
    %4: u32 = GetRegister r1
    %5 = SetRegister r3, %4
structure:
    block0
    if %1 {
        block1
    }
    block2
    return
";
    let out = rewrite_keeps_state(text, &[machine(&[0]), machine(&[1])]).to_string();
    assert!(!out.contains("Phi"), "{out}");
    assert!(out.contains("%4: opaque = Identity u32 0x9"), "{out}");
}

/// A tree with no block where paths meet: it starts with an `if`, a loop follows an `if`, a
/// loop body starts with an `if`, a `break` follows an `if`, an `if` follows a loop, and an
/// `if` ends the list.
const BARE: &str = "block0:
    %0 = SetRegister r0, u32 1
block1:
    %1: u32 = GetRegister r0
    %2: u32 = IAdd32 %1, u32 1
    %3 = SetRegister r0, %2
block2:
    %4: u32 = GetRegister r0
    %5 = SetRegister r1, %4
structure:
    if u1 true {
        block0
    }
    loop {
        if u1 true {
            block1
        }
        break u1 true
    } repeat u1 false
    if u1 true {
        block2
    }
";

#[test]
fn blocks_are_added_where_paths_meet() {
    let out = rewrite_keeps_state(BARE, &[Machine::new()]);
    assert_eq!(ran(&out, &Machine::new()).regs[1], 2);
    let text = out.to_string();
    let tree = text.split("structure:\n").nth(1).expect("a tree");
    assert_eq!(
        tree,
        "    block3
    if u1 true {
        block0
    }
    block4
    loop {
        block5
        if u1 true {
            block1
        }
        block6
        break u1 true
    } repeat u1 false
    block7
    if u1 true {
        block2
    }
    block8
"
    );
    insta::assert_snapshot!("structured_bare", text);
}

#[test]
fn a_return_or_unreachable_after_an_if_adds_no_block() {
    let text = "block0:
    %0: u1 = GetPred p0
block1:
    %1 = SetRegister r1, u32 1
structure:
    block0
    if %0 {
        block1
    }
    return
";
    let out = rewrite_keeps_state(text, &[Machine::new()]);
    assert_eq!(out.block_count(), 2, "{out}");
    let unreachable = text.replace("    return\n", "    unreachable\n");
    let out = rewrite_keeps_state(&unreachable, &[]);
    assert_eq!(out.block_count(), 2, "{out}");
}

#[test]
fn an_empty_loop_body_gets_a_header() {
    let text = "block0:
    %0 = SetRegister r1, u32 1
structure:
    block0
    loop {
    } repeat u1 false
    return
";
    let out = rewrite_keeps_state(text, &[Machine::new()]);
    let printed = out.to_string();
    assert!(
        printed.ends_with("    loop {\n        block1\n    } repeat u1 false\n    return\n"),
        "{printed}"
    );
}

#[test]
fn blocks_after_a_return_read_undef() {
    // block1 follows the `return`: nothing runs before it, so its read is an Undef of its own,
    // never block0's value.
    let text = "block0:
    %0 = SetRegister r0, u32 1
block1:
    %1: u32 = GetRegister r0
    %2 = SetRegister r1, %1
structure:
    block0
    return
    block1
";
    let out = rewrite_keeps_state(text, &[Machine::new()]).to_string();
    assert!(
        out.contains("block1:\n    %3: u32 = UndefU32\n    %1: opaque = Identity %3\n"),
        "{out}"
    );
}

#[test]
fn the_first_block_reads_undef() {
    let text = "block0:
    %0: u32 = GetRegister r0
    %1 = SetRegister r1, %0
structure:
    block0
    return
";
    let out = rewrite_keeps_state(text, &[Machine::new()]).to_string();
    assert!(out.contains("%3: u32 = UndefU32\n"), "{out}");
    assert!(!out.contains("= Phi"), "{out}");
}

#[test]
fn structured_phis_are_refused_before_anything_changes() {
    let text = |phi: &str| {
        format!(
            "block0:\n    %0 = SetRegister r0, u32 1\nblock1:\n{phi}    %2: u32 = GetRegister r0\n\
             structure:\n    block0\n    if u1 true {{\n        block1\n    }}\n"
        )
    };
    let mut function = parse(&text("    %1: u32 = Phi [block0: u32 1]\n"));
    let unchanged = function.clone();
    assert_eq!(
        run(&mut function, Pass::SsaRewrite),
        Err(PassError::StructuredPhi(value(1)))
    );
    assert_eq!(function, unchanged);
    assert_eq!(
        PassError::StructuredPhi(value(1)).to_string(),
        "%1 is a phi in a structured function, which the SSA rewrite takes before SSA"
    );
    // The same function without the phi rewrites.
    let mut mended = parse(&text(""));
    run(&mut mended, Pass::SsaRewrite).expect("rewrites");
    // The same phi in a control-flow graph is the rewrite's input, as before.
    let graph = "block0:\n    %0 = SetRegister r0, u32 1\n    jump block1\nblock1:\n    \
                 %1: u32 = Phi [block0: u32 1]\n    %2: u32 = GetRegister r0\n    return\n";
    let mut graph = parse(graph);
    run(&mut graph, Pass::SsaRewrite).expect("rewrites");
}

#[test]
fn the_first_fault_in_block_order_is_refused_in_a_structured_function() {
    // A goto index given by a value (%2) and a phi (%4), each in either block: the one in the
    // earlier block is reported.
    let goto = "    %1: u32 = GetCbufU32 u32 0, u32 0\n    %2 = SetGotoVariable %1, u1 true\n";
    let phi = "    %4: u32 = Phi\n";
    let text = |first: &str, second: &str| {
        format!("block0:\n{first}block1:\n{second}structure:\n    block0\n    block1\n    return\n")
    };
    for (first, second, expected) in [
        (goto, phi, PassError::GotoIndex(value(2))),
        (phi, goto, PassError::StructuredPhi(value(4))),
    ] {
        let mut function = parse(&text(first, second));
        let unchanged = function.clone();
        assert_eq!(run(&mut function, Pass::SsaRewrite), Err(expected));
        assert_eq!(function, unchanged);
    }
    assert_eq!(
        error(&text(phi, ""), Pass::SsaRewrite),
        PassError::StructuredPhi(value(4))
    );
}

/// A diamond as a control-flow graph that stores the value it merges to an output.
const GRAPH: &str = "block0:
    %6: u32 = GetCbufU32 u32 0, u32 0
    %0: u1 = INotEqual %6, u32 0
    branch %0, block1, block2
block1:
    %1 = SetRegister r1, u32 1
    jump block3
block2:
    %2 = SetRegister r1, u32 2
    jump block3
block3:
    %3: u32 = GetRegister r1
    %4: f32 = BitCastF32U32 %3
    %5 = SetAttribute a0x80, %4, u32 0
    return
";

/// Whether `text` still names a goto variable.
fn has_goto_variable(text: &str) -> bool {
    text.contains("GotoVariable")
}

#[test]
fn the_structurizers_variables_are_gone_after_ssa() {
    let structured = structure(parse(GRAPH), MAX_STRUCTURE_WORK)
        .expect("structures")
        .function;
    assert!(has_goto_variable(&structured.to_string()));
    let mut rewritten = structured.clone();
    run(&mut rewritten, Pass::SsaRewrite).expect("rewrites");
    // Every read became a value; the writes stay for dead code elimination.
    let text = rewritten.to_string();
    assert!(!text.contains("GetGotoVariable"), "{text}");
    insta::assert_snapshot!("structured_goto_after_ssa", text);
    let mut optimized = structured.clone();
    optimize(&mut optimized).expect("optimizes");
    let text = optimized.to_string();
    assert!(!has_goto_variable(&text), "{text}");
    insta::assert_snapshot!("structured_goto_optimized", text);
    for (word, r1) in [(0, 2_u32), (1, 1)] {
        let start = machine(&[word]);
        let before = ran(&parse(GRAPH), &start);
        let after = ran(&optimized, &start);
        assert_eq!(after.outputs, before.outputs);
        assert_eq!(before.outputs.values().copied().collect::<Vec<_>>(), [r1]);
    }
}

#[test]
fn the_rewrites_work_on_the_structured_diamond() {
    // The exact work of one fixed input (the parent module's "Cost"), counted from this code: a
    // change is explained in its pull request. The shortcut to the entry keeps it here; walking
    // every read back through the blocks, as Eden does, takes more.
    let mut structured = structure(parse(GRAPH), MAX_STRUCTURE_WORK)
        .expect("structures")
        .function;
    let work = run(&mut structured, Pass::SsaRewrite).expect("rewrites");
    assert_eq!(work.steps(), 256);
}

/// A loop whose first block reads r0 and whose second block writes it: the write reaches the
/// read along the back edge, though it comes later in the tree than the read.
const LATER_WRITE: &str = "block0:
    %0: u32 = GetCbufU32 u32 0, u32 0
block1:
    %1: u32 = GetRegister r0
    %2: u32 = IAdd32 %1, u32 2
    %3: u1 = ULessThan %2, %0
block2:
    %4 = SetRegister r0, %2
block3:
    %5: u32 = GetRegister r0
    %6 = SetRegister r1, %5
structure:
    block0
    loop {
        block1
        block2
    } repeat %3
    block3
    return
";

#[test]
fn a_write_later_in_a_loop_reaches_its_first_block() {
    // The read in block1 is not past every write of r0 (block2's, later in the tree, reaches it
    // round the loop), so it is not taken from the entry.
    let starts = [machine(&[0]), machine(&[6]), machine(&[7])];
    let out = rewrite_keeps_state(LATER_WRITE, &starts);
    for (word, r1) in [(0, 2), (6, 6), (7, 8)] {
        assert_eq!(ran(&out, &machine(&[word])).regs[1], r1);
    }
    let text = out.to_string();
    assert!(text.contains("Phi [block0: %"), "{text}");
}

/// An inner loop reads r0, which the outer loop writes after the inner one: the write reaches
/// the read round the outer loop, past the inner loop's end.
const NESTED_LATER_WRITE: &str = "block0:
    %0: u32 = GetCbufU32 u32 0, u32 0
block1:
    %1: u32 = GetRegister r0
    %2: u32 = IAdd32 %1, u32 1
    %7 = SetRegister r2, %2
block2:
    %8: u32 = GetRegister r2
    %3 = SetRegister r0, %8
    %4: u1 = ULessThan %8, %0
block3:
    %5: u32 = GetRegister r0
    %6 = SetRegister r1, %5
structure:
    block0
    loop {
        loop {
            block1
        } repeat u1 false
        block2
    } repeat %4
    block3
    return
";

#[test]
fn a_read_in_an_inner_loop_reaches_to_the_outermost_loops_end() {
    let starts = [machine(&[0]), machine(&[3])];
    let out = rewrite_keeps_state(NESTED_LATER_WRITE, &starts);
    for (word, r1) in [(0, 1), (3, 3)] {
        assert_eq!(ran(&out, &machine(&[word])).regs[1], r1);
    }
}

/// A function whose block3 has a phi taking `incoming`, with block1 inside an `if` defining %1,
/// and `extra` (blocks placed nowhere in the tree, which the verifier refuses) after block3.
fn phi_taking(incoming: &str, extra: &str) -> Function {
    parse(&format!(
        "block0:
    %0: u1 = IEqual u32 0, u32 0
block1:
    %1: u32 = IAdd32 u32 1, u32 2
block2:
    %2: u32 = IAdd32 u32 3, u32 4
block3:
    %3: u32 = Phi {incoming}
    %4 = SetRegister r1, %3
{extra}structure:
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

/// The scope checker the equivalence proptests, the rewrites here and the flow tests run
/// ([`check_phi_scope`]) is the verifier: it accepts a value in scope at the end of block2 (the
/// one block that can run last before block3) and an immediate, and refuses a value whose `if`
/// body has ended there, a block that cannot run last before block3 (block1, inside the `if`,
/// and block0, where %1 is not computed yet either), and a block the tree does not place, each
/// with the verifier's own error. The verifier's own tests are `ir::verify::phis::tests`.
#[test]
fn the_scope_checker_refuses_a_phi_value_out_of_scope() {
    for incoming in ["[block2: %2]", "[block2: u32 9]"] {
        assert_eq!(
            check_phi_scope(&phi_taking(incoming, "")),
            Ok(()),
            "{incoming}"
        );
    }
    let function = phi_taking("[block2: %1]", "");
    let block = |index: usize| {
        function
            .blocks()
            .nth(index)
            .map(|(id, _)| id)
            .expect("a block")
    };
    assert_eq!(
        check_phi_scope(&function),
        Err(VerifyError::PhiOutOfScope {
            at: value(3),
            operand: value(1),
            block: block(2),
        })
    );
    for (incoming, from) in [("[block1: %1]", 1), ("[block0: %1]", 0)] {
        assert_eq!(
            check_phi_scope(&phi_taking(incoming, "")),
            Err(VerifyError::PhiNotPredecessor {
                at: value(3),
                block: block(from),
            }),
            "{incoming}"
        );
    }
    let function = phi_taking("[block4: %1]", "block4:\n");
    let block4 = function.blocks().nth(4).map(|(id, _)| id).expect("block4");
    assert_eq!(
        check_phi_scope(&function),
        Err(VerifyError::NotPlaced(block4))
    );
    // A control-flow graph's phi takes a value that dominates the end of its predecessor.
    assert_eq!(
        check_phi_scope(&parse(
            "block0:\n    %1: u32 = IAdd32 u32 1, u32 2\n    jump block1\nblock1:\n    \
             %0: u32 = Phi [block0: %1]\n    return\n"
        )),
        Ok(())
    );
}
