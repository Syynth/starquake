//! The SSA rewrite: golden IR for straight-line code, a value used across blocks, a diamond, loops
//! that need phis, every variable class, unreachable blocks and an entry loop; the departures
//! from Eden; and the refusals.

use super::{after, error, parse, ran, value};
use crate::interp::Machine;
use crate::opt::{Pass, PassError, run};

/// The machine a test starts from: constant buffer 0 holds `words`, little-endian.
fn machine(words: &[u32]) -> Machine {
    let mut machine = Machine::new();
    let bytes = words.iter().flat_map(|word| word.to_le_bytes()).collect();
    machine.cbufs.insert(0, bytes);
    machine
}

/// Checks that the rewrite of `text` leaves the same machine as `text` itself, from `start`,
/// and returns the rewritten text.
fn rewrite_keeps_state(text: &str, start: &Machine) -> String {
    let before = parse(text);
    let mut rewritten = before.clone();
    run(&mut rewritten, Pass::SsaRewrite).expect("rewrites");
    assert_eq!(ran(&rewritten, start), ran(&before, start), "{rewritten}");
    rewritten.to_string()
}

const STRAIGHT: &str = "block0:
    %0 = SetRegister r1, u32 7
    %1: u32 = GetRegister r1
    %2: u32 = GetRegister r2
    %3: u32 = IAdd32 %1, %2
    %4 = SetRegister r1, %3
    %5: u32 = GetRegister r1
    %6 = SetRegister r0, %5
    return
";

#[test]
fn straight_line_reads_take_the_last_write_or_undef() {
    let text = rewrite_keeps_state(STRAIGHT, &Machine::new());
    insta::assert_snapshot!("ssa_straight", text);
}

/// r0 is written in block0 and read two blocks later, through a block that does not touch it.
const ACROSS: &str = "block0:
    %0: u32 = GetCbufU32 u32 0, u32 0
    %1 = SetRegister r0, %0
    jump block1
block1:
    %2 = SetRegister r1, u32 3
    jump block2
block2:
    %3: u32 = GetRegister r0
    %4: u32 = GetRegister r1
    %5: u32 = IMul32 %3, %4
    %6 = SetRegister r2, %5
    return
";

#[test]
fn a_value_used_across_blocks_needs_no_phi() {
    let text = rewrite_keeps_state(ACROSS, &machine(&[5]));
    insta::assert_snapshot!("ssa_across", text);
}

const DIAMOND: &str = "block0:
    %0: u32 = GetCbufU32 u32 0, u32 0
    %1: u1 = INotEqual %0, u32 0
    branch %1, block1, block2
block1:
    %2 = SetRegister r0, u32 1
    jump block3
block2:
    %3 = SetRegister r0, u32 2
    jump block3
block3:
    %4: u32 = GetRegister r0
    %5 = SetRegister r1, %4
    return
";

#[test]
fn a_diamond_merges_with_a_phi() {
    for word in [0, 1] {
        rewrite_keeps_state(DIAMOND, &machine(&[word]));
    }
    insta::assert_snapshot!("ssa_diamond", after(DIAMOND, Pass::SsaRewrite));
}

/// Sums 5 + 4 + 3 + 2 + 1 into r0 in a loop of one block, which writes the registers it carries
/// round the loop and branches back to itself.
const SELF_LOOP: &str = "block0:
    %0 = SetRegister r0, u32 0
    %1: u32 = GetCbufU32 u32 0, u32 0
    %2 = SetRegister r1, %1
    jump block1
block1:
    %3: u32 = GetRegister r0
    %4: u32 = GetRegister r1
    %5: u32 = IAdd32 %3, %4
    %6 = SetRegister r0, %5
    %7: u32 = ISub32 %4, u32 1
    %8 = SetRegister r1, %7
    %9: u1 = INotEqual %7, u32 0
    branch %9, block1, block2
block2:
    %10: u32 = GetRegister r0
    %11 = SetRegister r2, %10
    return
";

#[test]
fn a_self_loop_keeps_its_loop_carried_value() {
    // Eden's sealing (each block right after its visit) gives the loop header's phi the value
    // at the start of the back-edge block, which here is the header itself: r2 would be 5.
    let start = machine(&[5]);
    let text = rewrite_keeps_state(SELF_LOOP, &start);
    let function = parse(&text);
    function.verify().expect("verified");
    let end = ran(&function, &start);
    assert_eq!(end.regs[2], 15);
    insta::assert_snapshot!("ssa_self_loop", text);
}

/// A loop of a header and a body: the header reads the counter, the body writes it and
/// branches back.
const HEADER_AND_BODY: &str = "block0:
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
fn a_loop_with_a_header_carries_values_in_phis() {
    let start = Machine::new();
    let text = rewrite_keeps_state(HEADER_AND_BODY, &start);
    assert_eq!(ran(&parse(&text), &start).regs[2], 6);
    insta::assert_snapshot!("ssa_header_and_body", text);
}

/// Every variable class: predicates, uniform registers and predicates, the four flags, a goto
/// variable and the indirect branch variable, each written in block0 and read in block1.
const CLASSES: &str = "block0:
    %0 = SetPred p1, u1 true
    %1 = SetUniformRegister ur2, u32 9
    %2 = SetUniformPred up3, u1 true
    %3 = SetZFlag u1 true
    %4 = SetSFlag u1 true
    %5 = SetCFlag u1 true
    %6 = SetOFlag u1 true
    %7 = SetGotoVariable u32 4, u1 true
    %8 = SetIndirectBranchVariable u32 11
    jump block1
block1:
    %9: u1 = GetPred p1
    %10: u32 = GetUniformRegister ur2
    %11: u1 = GetUniformPred up3
    %12: u1 = GetZFlag
    %13: u1 = GetSFlag
    %14: u1 = GetCFlag
    %15: u1 = GetOFlag
    %16: u1 = GetGotoVariable u32 4
    %17: u32 = GetIndirectBranchVariable
    %18: u1 = GetPred p2
    %19 = SetPred p0, %9
    %20 = SetRegister r0, %10
    %21 = SetPred p4, %11
    %22 = SetPred p5, %12
    %23 = SetPred p6, %13
    %24 = SetUniformPred up0, %14
    %25 = SetUniformPred up1, %15
    %26 = SetUniformPred up2, %16
    %27 = SetRegister r1, %17
    %28 = SetUniformPred up4, %18
    return
";

#[test]
fn every_variable_class_is_rewritten() {
    let text = rewrite_keeps_state(CLASSES, &Machine::new());
    // Every read became an Identity of the written immediate, or of an Undef for p2.
    for opcode in [
        "GetPred",
        "GetUniformRegister",
        "GetUniformPred",
        "GetZFlag",
        "GetSFlag",
        "GetCFlag",
        "GetOFlag",
        "GetGotoVariable",
        "GetIndirectBranchVariable",
    ] {
        assert!(!text.contains(opcode), "{opcode} left in {text}");
    }
    insta::assert_snapshot!("ssa_classes", text);
}

/// block2 is unreachable and jumps into block1; block3 is unreachable and reads r0 with no
/// predecessor.
const UNREACHABLE: &str = "block0:
    %0 = SetRegister r0, u32 1
    jump block1
block1:
    %1: u32 = GetRegister r0
    %2 = SetRegister r1, %1
    return
block2:
    %3 = SetRegister r0, u32 2
    jump block1
block3:
    %4: u32 = GetRegister r0
    %5 = SetRegister r2, %4
    return
";

#[test]
fn unreachable_blocks_are_rewritten() {
    let text = rewrite_keeps_state(UNREACHABLE, &Machine::new());
    // block1 merges block0's and the unreachable block2's values; block3's read is undefined.
    assert!(text.contains("Phi [block0: u32 0x1], [block2: u32 0x2]"), "{text}");
    insta::assert_snapshot!("ssa_unreachable", text);
}

/// block0 is a loop header: it jumps back to itself.
const ENTRY_LOOP: &str = "block0:
    %0: u32 = GetRegister r0
    %1: u32 = IAdd32 %0, u32 2
    %2 = SetRegister r0, %1
    %3: u1 = ULessThan %1, u32 6
    branch %3, block0, block1
block1:
    return
";

#[test]
fn an_entry_loop_moves_to_its_own_block() {
    let start = Machine::new();
    let text = rewrite_keeps_state(ENTRY_LOOP, &start);
    assert_eq!(ran(&parse(&text), &start).regs[0], 6);
    // block0 keeps only the jump (and the undefined value block2's phi takes from it).
    assert!(text.contains("    jump block2\nblock1:\n    return\nblock2:\n"), "{text}");
    assert!(text.contains("Phi [block0: %6], [block2: %1]"), "{text}");
    insta::assert_snapshot!("ssa_entry_loop", text);
}

#[test]
fn an_entry_loop_retargets_phis_that_named_block0() {
    // block1's phi names block0; after the split its value arrives from block2.
    let text = "block0:
    %0: u1 = GetPred p0
    branch %0, block0, block1
block1:
    %1: u32 = Phi [block0: u32 7]
    %2 = SetRegister r0, %1
    return
";
    let out = after(text, Pass::SsaRewrite);
    assert!(out.contains("%1: u32 = Phi [block2: u32 0x7]"), "{out}");
}

#[test]
fn entry_phis_are_refused() {
    let text = "block0:
    %0: u32 = Phi [block1: u32 1]
    jump block1
block1:
    jump block0
";
    let mut function = parse(text);
    let unchanged = function.clone();
    assert_eq!(
        run(&mut function, Pass::SsaRewrite),
        Err(PassError::EntryPhi(value(0)))
    );
    assert_eq!(function, unchanged);
    assert_eq!(
        PassError::EntryPhi(value(0)).to_string(),
        "%0 is a phi in the entry block, which other blocks jump to"
    );
}

#[test]
fn an_entry_phi_with_no_predecessors_is_left_alone() {
    // A phi in block0 refuses only the split; with no predecessors there is nothing to split,
    // and an unwritten read in the entry is undefined: Eden's operandless phi, removed as
    // trivial, then the Undef.
    let text = "block0:
    %0: u32 = Phi
    %1: u32 = GetRegister r0
    %2 = SetRegister r1, %1
    return
";
    let mut function = parse(text);
    super::super::ssa::run(&mut function).expect("rewrites");
    assert_eq!(
        function.to_string(),
        "block0:
    %0: u32 = Phi
    %4: u32 = UndefU32
    %3: opaque = Identity %4
    %1: opaque = Identity %4
    %2 = SetRegister r1, %1
    return
"
    );
}

#[test]
fn structured_functions_are_refused() {
    let text = "block0:
    %0: u32 = GetRegister r0
structure:
    block0
    return
";
    assert_eq!(error(text, Pass::SsaRewrite), PassError::Structured);
    assert_eq!(
        PassError::Structured.to_string(),
        "the SSA rewrite needs a control-flow graph, not a structured function"
    );
}

#[test]
fn goto_indices_must_be_immediates() {
    for op in ["GetGotoVariable %0", "SetGotoVariable %0, u1 true"] {
        let text = format!(
            "block0:\n    %0: u32 = GetCbufU32 u32 0, u32 0\n    %1 = {op}\n    return\n"
        );
        assert_eq!(
            error(&text, Pass::SsaRewrite),
            PassError::GotoIndex(value(1))
        );
    }
    assert_eq!(
        PassError::GotoIndex(value(1)).to_string(),
        "%1 names its goto variable with a value, not an immediate"
    );
}

#[test]
fn reserved_names_are_left_for_constant_propagation() {
    // Not verifiable, so the rewrite runs alone, unverified: RZ and PT reads stay, and writes
    // to them change no definition.
    let text = "block0:
    %0 = SetRegister rz, u32 5
    %1: u32 = GetRegister rz
    %2 = SetPred pt, u1 false
    %3: u1 = GetPred pt
    %4 = SetUniformRegister urz, u32 5
    %5: u32 = GetUniformRegister urz
    %6 = SetUniformPred upt, u1 false
    %7: u1 = GetUniformPred upt
    return
";
    let mut function = parse(text);
    super::super::ssa::run(&mut function).expect("rewrites");
    assert_eq!(function.to_string(), parse(text).to_string());
}

#[test]
fn a_trivial_join_takes_the_resolved_value() {
    // Both sides copy r0 into r1, so block3's phi of r1 is trivial: it stands for %0, not for
    // block1's %3, which does not dominate block3.
    let text = "block0:
    %0: u32 = GetCbufU32 u32 0, u32 0
    %1 = SetRegister r0, %0
    %2: u1 = GetPred p0
    branch %2, block1, block2
block1:
    %3: u32 = GetRegister r0
    %4 = SetRegister r1, %3
    jump block3
block2:
    %5: u32 = GetRegister r0
    %6 = SetRegister r1, %5
    jump block3
block3:
    %7: u32 = GetRegister r1
    %8 = SetRegister r2, %7
    return
";
    let out = rewrite_keeps_state(text, &machine(&[9]));
    assert!(out.contains("    %7: opaque = Identity %0\n"), "{out}");
}

#[test]
fn a_read_in_a_branch_condition_becomes_its_value() {
    let text = "block0:
    %0 = SetPred p0, u1 true
    jump block1
block1:
    %1: u1 = GetPred p0
    branch %1, block2, block2
block2:
    return
";
    let out = after(text, Pass::SsaRewrite);
    assert!(out.contains("%1: opaque = Identity u1 true"), "{out}");
}
