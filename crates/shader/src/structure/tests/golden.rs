//! Hand-built graphs against golden IR text (`golden/*.txt`), each also run in the
//! interpreter on inputs that take every path. A golden file is the structured function's
//! text form; each change to one is explained in its pull request.

use super::{Ending, parse, same_run, structured};
use crate::interp::Machine;

/// Structures `graph`, checks the text against `golden` and returns the structured function.
fn check(graph: &str, golden: &str) -> crate::ir::Function {
    let tree = structured(graph).function;
    let text = tree.to_string();
    if std::env::var_os("STARQUAKE_BLESS").is_some() {
        println!("---- golden ----\n{text}---- end ----");
    }
    assert_eq!(
        text, golden,
        "the structured text differs from its golden file"
    );
    tree
}

/// One block that returns.
const SINGLE: &str = "block0:
    %0 = SetRegister r1, u32 7
    return
";

#[test]
fn a_single_block_returns() {
    let tree = check(SINGLE, include_str!("golden/single.txt"));
    let (ending, machine) = same_run(&parse(SINGLE), &tree, &Machine::new());
    assert_eq!(ending, Ending::Returned);
    assert_eq!(machine.regs[1], 7);
}

#[test]
fn control_that_can_fall_off_the_end_reaches_unreachable() {
    // block1 (which nothing reaches) jumps back to block0: the goto is lifted into a loop
    // around both blocks, and control after the loop falls off the end of the program, where
    // Eden emits `Unreachable`. When the last statement is a `return`, nothing follows it.
    let looping = structured("block0:\n    return\nblock1:\n    jump block0\n").function;
    let nodes = looping.structure().expect("structured");
    assert!(matches!(nodes.last(), Some(crate::ir::Node::Unreachable)));
    assert!(matches!(
        nodes.get(nodes.len() - 2),
        Some(crate::ir::Node::Loop { .. })
    ));
    let single = structured(SINGLE).function;
    assert!(matches!(
        single.structure().expect("structured").last(),
        Some(crate::ir::Node::Return)
    ));
}

/// A diamond: block0 tests p0, block1 and block2 are the arms, block3 the join.
const DIAMOND: &str = "block0:
    %0 = GetPred p0
    branch %0, block1, block2
block1:
    %1 = SetRegister r1, u32 1
    jump block3
block2:
    %2 = SetRegister r1, u32 2
    jump block3
block3:
    return
";

#[test]
fn a_diamond_is_an_if() {
    let tree = check(DIAMOND, include_str!("golden/diamond.txt"));
    let graph = parse(DIAMOND);
    for (p0, r1) in [(false, 2), (true, 1)] {
        let mut machine = Machine::new();
        machine.preds[0] = p0;
        let (ending, after) = same_run(&graph, &tree, &machine);
        assert_eq!(ending, Ending::Returned);
        assert_eq!(after.regs[1], r1);
    }
}

/// An irreducible loop: the entry goes into block1 or block2 on p0, and the two jump into each
/// other, each counting its visits in its own register, until one of them has run three times.
/// Neither block dominates the other, so the loop has two entries.
const IRREDUCIBLE: &str = "block0:
    %0 = GetPred p0
    branch %0, block1, block2
block1:
    %1 = GetRegister r1
    %2 = IAdd32 %1, u32 1
    %3 = SetRegister r1, %2
    %4 = ULessThan %2, u32 3
    branch %4, block2, block3
block2:
    %5 = GetRegister r2
    %6 = IAdd32 %5, u32 1
    %7 = SetRegister r2, %6
    %8 = ULessThan %6, u32 3
    branch %8, block1, block3
block3:
    return
";

#[test]
fn an_irreducible_loop() {
    let tree = check(IRREDUCIBLE, include_str!("golden/irreducible.txt"));
    let graph = parse(IRREDUCIBLE);
    // From block1: 1, 2, 2, 3 (block1 reaches 3 first); from block2: block2 reaches 3 first.
    for (p0, r1, r2) in [(true, 3, 2), (false, 2, 3)] {
        let mut machine = Machine::new();
        machine.preds[0] = p0;
        let (ending, after) = same_run(&graph, &tree, &machine);
        assert_eq!(ending, Ending::Returned);
        assert_eq!((after.regs[1], after.regs[2]), (r1, r2));
    }
}

/// Nested loops with a break out of both: the inner loop (block2, block3) counts r2 and leaves
/// to the outer loop's tail every fourth count; block2 leaves both loops when r2 reaches the
/// value in r3; the outer loop (block1 to block4) counts r1 up to 3.
const NESTED_BREAK: &str = "block0:
    jump block1
block1:
    %0 = GetRegister r1
    %1 = IAdd32 %0, u32 1
    %2 = SetRegister r1, %1
    jump block2
block2:
    %3 = GetRegister r2
    %4 = IAdd32 %3, u32 1
    %5 = SetRegister r2, %4
    %6 = GetRegister r3
    %7 = IEqual %4, %6
    branch %7, block5, block3
block3:
    %8 = GetRegister r2
    %9 = BitwiseAnd32 %8, u32 3
    %10 = INotEqual %9, u32 0
    branch %10, block2, block4
block4:
    %11 = GetRegister r1
    %12 = ULessThan %11, u32 3
    branch %12, block1, block5
block5:
    return
";

#[test]
fn a_break_out_of_nested_loops() {
    let tree = check(NESTED_BREAK, include_str!("golden/nested_break.txt"));
    let graph = parse(NESTED_BREAK);
    // r3 = 6 breaks out of both loops in the second outer pass; r3 = 100 never matches, so the
    // outer loop runs out after three passes of four inner counts.
    for (r3, r1, r2) in [(6, 2, 6), (100, 3, 12)] {
        let mut machine = Machine::new();
        machine.regs[3] = r3;
        let (ending, after) = same_run(&graph, &tree, &machine);
        assert_eq!(ending, Ending::Returned);
        assert_eq!((after.regs[1], after.regs[2]), (r1, r2));
    }
}

/// A loop with an early return: block1 counts r1 and returns from block3 (marking r2) when it
/// reaches r3; otherwise the loop ends after ten counts and returns from block4.
const EARLY_RETURN: &str = "block0:
    jump block1
block1:
    %0 = GetRegister r1
    %1 = IAdd32 %0, u32 1
    %2 = SetRegister r1, %1
    %3 = GetRegister r3
    %4 = IEqual %1, %3
    branch %4, block3, block2
block2:
    %5 = GetRegister r1
    %6 = ULessThan %5, u32 10
    branch %6, block1, block4
block3:
    %7 = SetRegister r2, u32 1
    return
block4:
    return
";

#[test]
fn an_early_return_inside_a_loop() {
    let tree = check(EARLY_RETURN, include_str!("golden/early_return.txt"));
    let graph = parse(EARLY_RETURN);
    for (r3, r1, r2) in [(4, 4, 1), (50, 10, 0)] {
        let mut machine = Machine::new();
        machine.regs[3] = r3;
        let (ending, after) = same_run(&graph, &tree, &machine);
        assert_eq!(ending, Ending::Returned);
        assert_eq!((after.regs[1], after.regs[2]), (r1, r2));
    }
}

/// A jump table in flow analysis's form (`flow::ir_build`): r0 goes into the indirect branch
/// variable, and each comparison block reads it again and compares it with one target's
/// address; no match reaches `unreachable`.
const JUMP_TABLE: &str = "block0:
    %0 = GetRegister r0
    %1 = SetIndirectBranchVariable %0
    %2: u32 = GetIndirectBranchVariable
    %3 = IEqual %2, u32 0x10
    branch %3, block3, block1
block1:
    %4: u32 = GetIndirectBranchVariable
    %5 = IEqual %4, u32 0x20
    branch %5, block4, block2
block2:
    %6: u32 = GetIndirectBranchVariable
    %7 = IEqual %6, u32 0x30
    branch %7, block5, block6
block3:
    %8 = SetRegister r1, u32 1
    jump block7
block4:
    %9 = SetRegister r1, u32 2
    jump block7
block5:
    %10 = SetRegister r1, u32 3
    jump block7
block6:
    unreachable
block7:
    return
";

#[test]
fn a_switch_like_jump_table() {
    let tree = check(JUMP_TABLE, include_str!("golden/jump_table.txt"));
    let graph = parse(JUMP_TABLE);
    for (r0, ending, r1) in [
        (0x10, Ending::Returned, 1),
        (0x20, Ending::Returned, 2),
        (0x30, Ending::Returned, 3),
        (0x40, Ending::Unreachable, 0),
    ] {
        let mut machine = Machine::new();
        machine.regs[0] = r0;
        let (ended, after) = same_run(&graph, &tree, &machine);
        assert_eq!(ended, ending);
        assert_eq!(after.regs[1], r1);
    }
}

/// A goto lifted into a loop: block1, which nothing reaches, jumps back to block0 after it.
const LIFTED: &str = "block0:
    %0 = SetRegister r1, u32 1
    return
block1:
    jump block0
";

#[test]
fn the_work_of_each_golden_graph() {
    // The exact work of each graph, as literals: `tests::limits::the_work_limit_at_its_edge`
    // measures the work before it tests the limit, so it cannot see a loop that stops charging
    // (the work would only shrink). Between them these graphs run every charged loop of the
    // pass (the module docs, "Limits"): `level`, `directly_related`, `sibling_from_nephew`,
    // `are_ordered` and `splice`, the `SanitizeNoBreaks` walk of a lift (LIFTED), the
    // emission's statements and its condition walk. Each count was taken from this code; a
    // change to one is explained in its pull request.
    for (name, graph, work) in [
        ("single", SINGLE, 14),
        ("diamond", DIAMOND, 159),
        ("irreducible", IRREDUCIBLE, 253),
        ("nested_break", NESTED_BREAK, 322),
        ("early_return", EARLY_RETURN, 220),
        ("jump_table", JUMP_TABLE, 570),
        ("lifted", LIFTED, 55),
    ] {
        assert_eq!(structured(graph).work, work, "the work of {name}");
    }
}

/// A goto lifted into a loop: block1, which nothing reaches, jumps back to block0 after it.
const LIFTED: &str = "block0:
    %0 = SetRegister r1, u32 1
    return
block1:
    jump block0
";

#[test]
fn the_work_of_each_golden_graph() {
    // The exact work of each graph, as literals: `tests::limits::the_work_limit_at_its_edge`
    // measures the work before it tests the limit, so it cannot see a loop that stops charging
    // (the work would only shrink). Between them these graphs run every charged loop of the
    // pass (the module docs, "Limits"): `level`, `directly_related`, `sibling_from_nephew`,
    // `are_ordered` and `splice`, the `SanitizeNoBreaks` walk of a lift (LIFTED), the
    // emission's statements and its condition walk. Each count was taken from this code; a
    // change to one is explained in its pull request.
    for (name, graph, work) in [
        ("single", SINGLE, 14),
        ("diamond", DIAMOND, 159),
        ("irreducible", IRREDUCIBLE, 253),
        ("nested_break", NESTED_BREAK, 322),
        ("early_return", EARLY_RETURN, 220),
        ("jump_table", JUMP_TABLE, 570),
        ("lifted", LIFTED, 55),
    ] {
        assert_eq!(structured(graph).work, work, "the work of {name}");
    }
}

/// Two overlapping loops, so a backward goto is eliminated as a loop after moving out of the
/// loop a later goto made around it: block1 counts r1 and block2 counts r2, which jumps back
/// to block1 while r2 is below r4; block3 counts r3 and jumps back to block2 while r3 is below
/// r5; block4 returns.
const OVERLAPPING_LOOPS: &str = "block0:
    jump block1
block1:
    %0 = GetRegister r1
    %1 = IAdd32 %0, u32 1
    %2 = SetRegister r1, %1
    jump block2
block2:
    %3 = GetRegister r2
    %4 = IAdd32 %3, u32 1
    %5 = SetRegister r2, %4
    %6 = GetRegister r4
    %7 = ULessThan %4, %6
    branch %7, block1, block3
block3:
    %8 = GetRegister r3
    %9 = IAdd32 %8, u32 1
    %10 = SetRegister r3, %9
    %11 = GetRegister r5
    %12 = ULessThan %9, %11
    branch %12, block2, block4
block4:
    return
";

#[test]
fn overlapping_loops_repeat() {
    let tree = check(
        OVERLAPPING_LOOPS,
        include_str!("golden/overlapping_loops.txt"),
    );
    let graph = parse(OVERLAPPING_LOOPS);
    // (r4, r5) and the counts (r1, r2, r3) they lead to: each back edge taken never, once,
    // and several times. block2 repeats block1 until r2 reaches r4; each pass through block3
    // below r5 runs block2 again, which counts r2 past r4 at once.
    for (r4, r5, counts) in [
        (0, 0, (1, 1, 1)),
        (2, 0, (2, 2, 1)),
        (0, 2, (1, 2, 2)),
        (3, 3, (3, 5, 3)),
    ] {
        let mut machine = Machine::new();
        machine.regs[4] = r4;
        machine.regs[5] = r5;
        let (ending, after) = same_run(&graph, &tree, &machine);
        assert_eq!(ending, Ending::Returned);
        assert_eq!((after.regs[1], after.regs[2], after.regs[3]), counts);
    }
}
