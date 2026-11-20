//! The control and warp translator's tests: each instruction and modifier run on the
//! interpreter's one-lane warp with expected values worked out from the instruction's
//! documented behaviour, golden IR for what needs a whole warp, PR #4588's `native_shader.cpp`
//! cases for this group (their one-lane cases on the interpreter, the rest as golden IR or as
//! listed divergences), each refusal with its exact error, and generated instructions.
//!
//! PR #4588's cases ported (`eden-pr4588/src/tests/video_core/native_shader.cpp` @ e5fb3b97),
//! by the test that ports each:
//!
//! - "Native register ballots retain the active predicate mask": `pr_register_ballot`.
//! - "Native warp uniform transfers use the first selected active lane and preserve
//!   predicates" (its `VOTEU` words): `pr_uniform_vote`.
//! - "Native full warp butterfly reads the XOR partner in every lane": `pr_full_warp_butterfly`
//!   (golden IR: every lane reads another).
//! - "Native indexed shuffle retains its predicate at the last valid lane":
//!   `pr_indexed_shuffle`.
//! - "Native upward and immediate indexed shuffles preserve boundary predicates":
//!   `pr_immediate_indexed_shuffle`, and `register_clamp_shuffles_are_refused` for its upward
//!   word (a divergence).
//! - "Native downward shuffle retains warp bounds and aliased source values":
//!   `pr_downward_shuffle`.
//! - "Native warp sum OR and unsigned extrema reduce only active lanes": `pr_warp_reduction`.
//! - "Native histogram warp match compares both address words and preserves predicates":
//!   `pr_warp_match`, its words rebuilt from the decoded form (the name suggests a game's
//!   shader).
//! - "Native deferred-blocking barrier remains a full workgroup synchronization":
//!   `pr_deferred_blocking_barrier`.
//! - "Native dependency waits retain the full six-bit pending-operation count":
//!   `pr_dependency_wait_counts`; "Native multi-scoreboard dependency waits retain surrounding
//!   dataflow": `depbar_needs_le_sb5` (a divergence).
//! - "Native full warp synchronization retains ordering after reconvergence" (its `WARPSYNC`,
//!   `BSSY` and `BSYNC` words): `pr_warp_sync_words` (no IR: a divergence).
//! - "Native quad butterfly and swizzled add retain horizontal and vertical differences" (its
//!   `SHFL` words only): `pr_quad_butterfly`, as golden IR (`ShuffleButterfly` outside a pixel
//!   program, `QuadSwap` in one), since every lane reads another. Its `FSWZADD` word belongs
//!   to another group's translator and is not ported here.
//!
//! These cases check behaviour, not the dispatch of the visitor this slice ports: the PR's
//! `TranslateScalarNative` (`native_shader.cpp`) lowers the native words with `LowerToMaxwell`
//! and runs Eden's Maxwell `TranslatorVisitor` on the result, not the native `Visitor::Run`.

use proptest::prelude::*;

use super::*;
use crate::asm::assemble;
use crate::decode::{
    BMovDir, Barrier, BraCond, BranchOffset, BrxSrc, DecodeError, GROUP_HOOKS, GROUPS, Instruction,
    OpBMov, OpBSSy, OpBSync, OpBra, OpBreak, OpBrx, OpCall, OpDepBar, OpExit, OpKill, OpMov,
    OpNanosleep, OpNop, OpRet, VOTEU_OPCODE, decode,
};
use crate::flow::{BranchTables, flow_graph};
use crate::interp::{INACTIVE_LANE_VALUE, Machine};
use crate::ir::{Terminator, Value};
use crate::testing::{any_pred_src, config};

const PIXEL: GuestStage = GuestStage::Pixel;
const VERTEX: GuestStage = GuestStage::Vertex;
const COMPUTE: GuestStage = GuestStage::Compute;

fn r(index: u8) -> Gpr {
    Gpr::new(index)
}

fn p(index: u8) -> Pred {
    Pred::new(index).expect("a predicate")
}

fn ur(index: u8) -> UniformReg {
    UniformReg::new(index).expect("a uniform register")
}

fn up(index: u8) -> UniformPred {
    UniformPred::new(index).expect("a uniform predicate")
}

fn src(index: u8) -> PredSrc {
    PredSrc {
        reg: p(index),
        negated: false,
    }
}

fn not(index: u8) -> PredSrc {
    PredSrc {
        reg: p(index),
        negated: true,
    }
}

/// `!PT`: never runs.
const NEVER: PredSrc = PredSrc {
    reg: Pred::PT,
    negated: true,
};

fn guarded(guard: PredSrc, op: Op) -> Instr {
    Instr { guard, op }
}

/// The word of PR #4588's four 32-bit words.
fn word(words: [u32; 4]) -> Instruction {
    let bits = words
        .iter()
        .rev()
        .fold(0_u128, |bits, &w| (bits << 32) | u128::from(w));
    Instruction::from_bits(bits)
}

/// PR #4588's words, decoded.
fn decoded(words: [u32; 4]) -> Instr {
    decode(word(words)).expect("decodes")
}

/// A fresh function with `instr` translated into its entry, ended with a return.
fn translated(
    stage: GuestStage,
    instr: Instr,
    guard_tested: bool,
) -> Result<(Function, Emitted), TranslateError> {
    let mut function = Function::new();
    let entry = function.entry();
    let emitted = translate_control(&mut function, entry, stage, 0, instr, guard_tested)?;
    Builder::new(&mut function, entry)
        .terminate(Terminator::Return)
        .expect("terminates");
    assert_eq!(function.verify(), Ok(()));
    Ok((function, emitted))
}

/// [`translated`] for an unguarded or tested instruction that must be accepted.
fn accepted(stage: GuestStage, instr: Instr) -> (Function, Emitted) {
    translated(stage, instr, false).expect("translates")
}

/// The refusal `what` of `instr`, translated as instruction 0.
fn refusal(instr: Instr, what: Unsupported) -> Result<(Function, Emitted), TranslateError> {
    Err(TranslateError::Unsupported {
        index: 0,
        instr,
        what,
    })
}

fn control(what: ControlUnsupported) -> Unsupported {
    Unsupported::Control(what)
}

/// A machine whose one active lane is `lane`.
fn lane(lane: u32) -> Machine {
    let mut machine = Machine::new();
    machine.system.lane_id = lane;
    machine
}

/// Runs `function` on `machine`.
fn run(function: &Function, machine: &mut Machine) {
    machine.run(function).expect("runs");
}

/// The predicates `(lane + p) & 1`, as PR #4588's warp tests set them.
fn pr_predicates(machine: &mut Machine, lane: u32) {
    for (index, slot) in (0_u32..).zip(machine.preds.iter_mut().take(usize::from(Pred::PT.index())))
    {
        *slot = (lane + index) & 1 != 0;
    }
}

fn vote(op: VoteOp, ballot: Gpr, vote: Pred, pred: PredSrc) -> Op {
    Op::Vote(OpVote {
        op,
        dst: VoteDst::Reg { ballot, vote },
        pred,
    })
}

fn voteu(op: VoteOp, ballot: UniformReg, vote: UniformPred, pred: PredSrc) -> Op {
    Op::Vote(OpVote {
        op,
        dst: VoteDst::Uniform { ballot, vote },
        pred,
    })
}

fn shfl(op: ShflOp, dst: Gpr, in_bounds: Pred, src: Gpr, lane: ShflSrc, c: ShflSrc) -> Op {
    Op::Shfl(OpShfl {
        op,
        dst,
        in_bounds,
        src,
        lane,
        c,
    })
}

fn bar_sync() -> OpBar {
    OpBar {
        mode: BarMode::Sync,
        red: BarRed::Popc,
        id: 0,
        defer_blocking: false,
        unknown: 0,
    }
}

fn warpsync(mask: AluSrc, cond: PredSrc) -> Op {
    Op::WarpSync(OpWarpSync { mask, cond })
}

fn depbar(scoreboard: u8, le: bool, count: u8) -> Op {
    Op::DepBar(OpDepBar {
        scoreboard,
        le,
        count,
    })
}

fn redux(op: ReduxOp, dst: UniformReg, src: Gpr) -> Op {
    Op::Redux(OpRedux { op, dst, src })
}

fn match_all(u64: bool, src: Gpr, pred: Pred) -> Op {
    Op::Match(OpMatch {
        op: MatchOp::All,
        u64,
        mask: Gpr::RZ,
        src,
        pred,
    })
}

/// One instruction of every form that becomes no IR, with its PR #4588 condition met.
fn no_ir_forms() -> Vec<Op> {
    let offset = BranchOffset::from_words(4);
    vec![
        Op::Nop(OpNop {}),
        Op::Break(OpBreak {
            bar: Barrier::new(1).expect("a barrier"),
            cond: src(3),
        }),
        Op::BSSy(OpBSSy {
            bar: Barrier::new(2).expect("a barrier"),
            offset,
            cond: PredSrc::TRUE,
        }),
        Op::BSync(OpBSync {
            bar: Barrier::new(2).expect("a barrier"),
            cond: PredSrc::TRUE,
        }),
        Op::Brx(OpBrx {
            src: BrxSrc::Reg(r(4)),
            offset,
            cond: PredSrc::TRUE,
        }),
        Op::Brx(OpBrx {
            src: BrxSrc::UniformReg(ur(4)),
            offset,
            cond: PredSrc::TRUE,
        }),
        Op::Call(OpCall {
            offset,
            no_inc: true,
            cond: PredSrc::TRUE,
        }),
        Op::Ret(OpRet {
            src: r(6),
            offset,
            no_dec: true,
            cond: PredSrc::TRUE,
        }),
        depbar(DEPBAR_REQUIRED_SCOREBOARD, true, 1),
        warpsync(AluSrc::Imm(FULL_WARP_MASK), PredSrc::TRUE),
        Op::Kill(OpKill {
            pred: PredSrc::TRUE,
        }),
    ]
}

// ---------------------------------------------------------------------------------------------
// Constants.

/// The constants against PR #4588's and Eden's literals, written out.
#[test]
fn constants_match_the_reference() {
    assert_eq!(FULL_WARP_MASK, 0xffff_ffff);
    assert_eq!(DEPBAR_REQUIRED_SCOREBOARD, 5);
    assert_eq!(QUAD_SHUFFLE_MASK, 0x1c03);
    assert_eq!(QUAD_LAST_LANE, 3);
    assert_eq!(SHFL_FIELD_MASK, 0x1f);
    assert_eq!(SHFL_SEGMENT_SHIFT, 8);
    assert_eq!(WHOLE_WARP_CLAMP, 0x1f);
    assert_eq!(WARP_SIZE, 32);
    let identities = [
        (ReduxOp::And, 0xffff_ffff),
        (ReduxOp::Or, 0),
        (ReduxOp::Xor, 0),
        (ReduxOp::Sum, 0),
        (ReduxOp::Min { signed: false }, 0xffff_ffff),
        (ReduxOp::Min { signed: true }, 0x7fff_ffff),
        (ReduxOp::Max { signed: false }, 0),
        (ReduxOp::Max { signed: true }, 0x8000_0000),
    ];
    for (op, identity) in identities {
        assert_eq!(redux_identity(op), identity, "{op:?}");
    }
}

#[test]
fn subgroup_operations_are_the_warp_group_but_lane_id_and_derivatives() {
    let subgroup: Vec<&str> = Opcode::ALL
        .iter()
        .filter(|opcode| is_subgroup_op(**opcode))
        .map(|opcode| opcode.name())
        .collect();
    assert_eq!(
        subgroup,
        [
            "VoteAll",
            "VoteAny",
            "VoteEqual",
            "SubgroupBallot",
            "SubgroupEqMask",
            "SubgroupLtMask",
            "SubgroupLeMask",
            "SubgroupGtMask",
            "SubgroupGeMask",
            "ShuffleIndex",
            "ShuffleUp",
            "ShuffleDown",
            "ShuffleButterfly",
            "QuadBroadcast",
            "QuadSwap",
        ]
    );
}

// ---------------------------------------------------------------------------------------------
// VOTE and VOTEU.

#[test]
fn votes_on_one_lane() {
    // One lane votes: all and any give its predicate, eq gives true; the ballot is its bit.
    for (op, expect) in [
        (VoteOp::All, [false, true]),
        (VoteOp::Any, [false, true]),
        (VoteOp::Eq, [true, true]),
    ] {
        for active in [0_u32, 31] {
            for (value, expected) in [false, true].into_iter().zip(expect) {
                // `!P1` is voted on, so P1 holds the opposite of `value`.
                let (ir, emitted) = accepted(COMPUTE, Instr::new(vote(op, r(5), p(2), not(1))));
                assert!(emitted.needs_subgroup_size_32);
                let mut machine = lane(active);
                machine.preds[1] = !value;
                machine.preds[2] = !expected;
                machine.regs[5] = 0xcafe_beef;
                run(&ir, &mut machine);
                let bit = if value { 1 << active } else { 0 };
                assert_eq!(machine.regs[5], bit, "{op:?} {value} lane {active}");
                assert_eq!(machine.preds[2], expected, "{op:?} {value} lane {active}");
                assert_eq!(machine.preds[1], !value, "the source is unchanged");
            }
        }
    }
}

/// Each vote names its IR operation (one lane cannot tell `ALL` from `ANY`).
#[test]
fn each_vote_names_its_operation() {
    for (op, name) in [
        (VoteOp::All, "VoteAll"),
        (VoteOp::Any, "VoteAny"),
        (VoteOp::Eq, "VoteEqual"),
    ] {
        for instr in [
            Instr::new(vote(op, Gpr::RZ, p(2), src(1))),
            Instr::new(voteu(op, UniformReg::URZ, up(2), src(1))),
        ] {
            let text = accepted(COMPUTE, instr).0.to_string();
            assert!(text.contains(&format!("%1: u1 = {name} %0\n")), "{text}");
        }
    }
}

#[test]
fn vote_golden_ir() {
    let (ir, _) = accepted(COMPUTE, Instr::new(vote(VoteOp::Any, r(5), p(2), not(1))));
    assert_eq!(
        ir.to_string(),
        "block0:\n\
         \x20   %0: u1 = GetPred p1\n\
         \x20   %1: u1 = LogicalNot %0\n\
         \x20   %2: u32 = SubgroupBallot %1\n\
         \x20   %3 = SetRegister r5, %2\n\
         \x20   %4: u1 = VoteAny %1\n\
         \x20   %5 = SetPred p2, %4\n\
         \x20   return\n"
    );
    let (ir, _) = accepted(
        COMPUTE,
        Instr::new(vote(VoteOp::Eq, r(5), p(2), PredSrc::TRUE)),
    );
    assert_eq!(
        ir.to_string(),
        "block0:\n\
         \x20   %0: u32 = SubgroupBallot u1 true\n\
         \x20   %1 = SetRegister r5, %0\n\
         \x20   %2: u1 = VoteEqual u1 true\n\
         \x20   %3 = SetPred p2, %2\n\
         \x20   return\n"
    );
}

/// `RZ` and `PT` write nothing: each destination alone, and neither.
#[test]
fn vote_destinations_rz_and_pt_write_nothing() {
    let cases = [
        (Gpr::RZ, Pred::PT, false, false),
        (r(5), Pred::PT, true, false),
        (Gpr::RZ, p(2), false, true),
    ];
    for (ballot, result, writes_ballot, writes_vote) in cases {
        let (ir, emitted) = accepted(
            COMPUTE,
            Instr::new(vote(VoteOp::All, ballot, result, src(1))),
        );
        let text = ir.to_string();
        assert_eq!(text.contains("SubgroupBallot"), writes_ballot, "{text}");
        assert_eq!(text.contains("SetRegister"), writes_ballot, "{text}");
        assert_eq!(text.contains("VoteAll"), writes_vote, "{text}");
        assert_eq!(text.contains("SetPred"), writes_vote, "{text}");
        assert_eq!(emitted.needs_subgroup_size_32, writes_ballot || writes_vote);
    }
}

#[test]
fn pr_register_ballot() {
    let instr = decoded([0x001b_7806, 0, 0x000e_0000, 0]);
    assert_eq!(
        instr,
        Instr::new(vote(VoteOp::All, r(27), Pred::PT, src(0)))
    );
    let (ir, _) = accepted(COMPUTE, instr);
    for active in [0_u32, 31] {
        for selected in [0_u32, 0xaaaa_aaaa, 0xffff_ffff] {
            let mut machine = lane(active);
            machine.preds[0] = (selected >> active) & 1 != 0;
            let saved = machine.preds;
            run(&ir, &mut machine);
            assert_eq!(machine.regs[27], (1 << active) & selected);
            assert_eq!(machine.preds, saved);
        }
    }
}

#[test]
fn uniform_votes_write_uniform_registers_and_predicates() {
    for (op, value, expected) in [
        (VoteOp::All, true, true),
        (VoteOp::All, false, false),
        (VoteOp::Any, true, true),
        (VoteOp::Any, false, false),
        (VoteOp::Eq, false, true),
    ] {
        let (ir, emitted) = accepted(COMPUTE, Instr::new(voteu(op, ur(9), up(3), src(4))));
        assert!(emitted.needs_subgroup_size_32);
        let mut machine = lane(31);
        machine.preds[4] = value;
        machine.uniform_preds[3] = !expected;
        machine.uniform_regs[9] = 0xcafe_beef;
        run(&ir, &mut machine);
        assert_eq!(machine.uniform_regs[9], if value { 1 << 31 } else { 0 });
        assert_eq!(machine.uniform_preds[3], expected, "{op:?} {value}");
    }
    // `URZ` and `UPT` write nothing.
    let (ir, emitted) = accepted(
        COMPUTE,
        Instr::new(voteu(
            VoteOp::All,
            UniformReg::URZ,
            UniformPred::UPT,
            src(4),
        )),
    );
    assert!(!emitted.needs_subgroup_size_32);
    assert_eq!(
        ir.to_string(),
        "block0:\n    %0: u1 = GetPred p4\n    return\n"
    );
}

#[test]
fn pr_uniform_vote() {
    for (w2, op) in [(0x0080_0000, VoteOp::All), (0x0080_0100, VoteOp::Any)] {
        let instr = decoded([0x003f_7886, 0, w2, 0]);
        assert_eq!(instr, Instr::new(voteu(op, UniformReg::URZ, up(0), src(1))));
        let (ir, _) = accepted(COMPUTE, instr);
        for active in [0_u32, 31] {
            for selected in [false, true] {
                let mut machine = lane(active);
                pr_predicates(&mut machine, active);
                machine.preds[1] = selected;
                machine.uniform_preds[0] = !selected;
                let preds = machine.preds;
                run(&ir, &mut machine);
                assert_eq!(machine.uniform_preds[0], selected);
                assert_eq!(machine.preds, preds);
            }
        }
    }
}

/// A guarded `VOTEU` keeps its guard, read as a uniform predicate, as a select of each
/// result: the old values stay where the guard is false.
#[test]
fn a_guarded_uniform_vote_selects_its_results() {
    let instr = guarded(not(2), voteu(VoteOp::Any, ur(9), up(3), src(4)));
    let (ir, emitted) = translated(COMPUTE, instr, false).expect("translates");
    assert!(emitted.needs_subgroup_size_32);
    assert_eq!(
        ir.to_string(),
        "block0:\n\
         \x20   %0: u1 = GetUniformPred up2\n\
         \x20   %1: u1 = LogicalNot %0\n\
         \x20   %2: u1 = GetPred p4\n\
         \x20   %3: u32 = SubgroupBallot %2\n\
         \x20   %4: u32 = GetUniformRegister ur9\n\
         \x20   %5: u32 = SelectU32 %1, %3, %4\n\
         \x20   %6 = SetUniformRegister ur9, %5\n\
         \x20   %7: u1 = VoteAny %2\n\
         \x20   %8: u1 = GetUniformPred up3\n\
         \x20   %9: u1 = SelectU1 %1, %7, %8\n\
         \x20   %10 = SetUniformPred up3, %9\n\
         \x20   return\n"
    );
    for runs in [false, true] {
        let mut machine = lane(5);
        // The guard is `!UP2`; the per-thread P2 is set the other way, so reading it fails.
        machine.uniform_preds[2] = !runs;
        machine.preds[2] = runs;
        machine.preds[4] = true;
        machine.uniform_regs[9] = 0x1111;
        machine.uniform_preds[3] = false;
        run(&ir, &mut machine);
        assert_eq!(machine.uniform_regs[9], if runs { 1 << 5 } else { 0x1111 });
        assert_eq!(machine.uniform_preds[3], runs);
    }
    // In a block whose guard flow analysis tested, the guard is ignored.
    let (tested, _) = translated(COMPUTE, instr, true).expect("translates");
    assert!(!tested.to_string().contains("Select"));
}

// ---------------------------------------------------------------------------------------------
// SHFL.

/// Lane `active` shuffles `0x1000 + lane` from `lane` (an immediate) with `c`; the result and
/// the in-bounds predicate P3.
fn shuffle_on_lane(stage: GuestStage, op: ShflOp, active: u32, from: u16, c: u16) -> (u32, bool) {
    let instr = Instr::new(shfl(
        op,
        r(7),
        p(3),
        r(6),
        ShflSrc::Imm(from),
        ShflSrc::Imm(c),
    ));
    let (ir, emitted) = accepted(stage, instr);
    assert!(emitted.needs_subgroup_size_32);
    let mut machine = lane(active);
    machine.regs[6] = 0x1000 + active;
    machine.preds[3] = false;
    run(&ir, &mut machine);
    (machine.regs[7], machine.preds[3])
}

#[test]
fn shuffles_on_one_lane() {
    let own = |active: u32| 0x1000 + active;
    // Index: the lane itself is in range; past the clamp, the own value and false.
    assert_eq!(
        shuffle_on_lane(COMPUTE, ShflOp::Idx, 9, 9, 0x1f),
        (own(9), true)
    );
    assert_eq!(
        shuffle_on_lane(COMPUTE, ShflOp::Idx, 9, 9, 9),
        (own(9), true)
    );
    assert_eq!(
        shuffle_on_lane(COMPUTE, ShflOp::Idx, 9, 10, 9),
        (own(9), false)
    );
    assert_eq!(
        shuffle_on_lane(COMPUTE, ShflOp::Idx, 9, 8, 0x1f),
        (INACTIVE_LANE_VALUE, true)
    );
    // The segment mask comes from bits 8..12 of the immediate: segment 0x18 on lane 13 reads
    // lanes 8..15, so index 5 is lane 13.
    let segmented = (0x18 << SHFL_SEGMENT_SHIFT) | 0x1f;
    assert_eq!(
        shuffle_on_lane(COMPUTE, ShflOp::Idx, 13, 5, segmented),
        (own(13), true)
    );
    // Bits 5..7 of the immediate are neither field.
    assert_eq!(
        shuffle_on_lane(COMPUTE, ShflOp::Idx, 9, 10, 0xe9),
        (own(9), false)
    );
    // Up by 0 and down by 0 read the lane itself; past the bounds, the own value.
    assert_eq!(
        shuffle_on_lane(COMPUTE, ShflOp::Up, 4, 0, 0),
        (own(4), true)
    );
    assert_eq!(
        shuffle_on_lane(COMPUTE, ShflOp::Up, 4, 5, 0),
        (own(4), false)
    );
    assert_eq!(
        shuffle_on_lane(COMPUTE, ShflOp::Down, 4, 0, 0x1f),
        (own(4), true)
    );
    assert_eq!(
        shuffle_on_lane(COMPUTE, ShflOp::Down, 4, 28, 0x1f),
        (own(4), false)
    );
    // Butterfly by 0 is the lane; by 8 past clamp 7, the own value.
    assert_eq!(
        shuffle_on_lane(COMPUTE, ShflOp::Bfly, 5, 0, 0x1f),
        (own(5), true)
    );
    assert_eq!(
        shuffle_on_lane(COMPUTE, ShflOp::Bfly, 5, 8, 7),
        (own(5), false)
    );
}

#[test]
fn shuffle_golden_ir() {
    // A register lane, the in-bounds predicate written.
    let instr = Instr::new(shfl(
        ShflOp::Down,
        r(7),
        p(3),
        r(6),
        ShflSrc::Reg(r(2)),
        ShflSrc::Imm((0x18 << 8) | 0xe5),
    ));
    let (ir, _) = accepted(PIXEL, instr);
    assert_eq!(
        ir.to_string(),
        "block0:\n\
         \x20   %0: u32 = GetRegister r6\n\
         \x20   %1: u32 = GetRegister r2\n\
         \x20   %2: u32 = ShuffleDown %0, %1, u32 0x5, u32 0x18\n\
         \x20   %3: u1 = GetInBoundsFromOp %2\n\
         \x20   %4 = SetPred p3, %3\n\
         \x20   %5 = SetRegister r7, %2\n\
         \x20   return\n"
    );
    // `PT` and `RZ`: no in-bounds read, `RZ` reads zero and nothing is written.
    let instr = Instr::new(shfl(
        ShflOp::Up,
        Gpr::RZ,
        Pred::PT,
        Gpr::RZ,
        ShflSrc::Imm(3),
        ShflSrc::Imm(0x1f),
    ));
    let (ir, _) = accepted(PIXEL, instr);
    assert_eq!(
        ir.to_string(),
        "block0:\n\
         \x20   %0: u32 = ShuffleUp u32 0x0, u32 0x3, u32 0x1f, u32 0x0\n\
         \x20   return\n"
    );
}

#[test]
fn pr_full_warp_butterfly() {
    for offset in [1_u32, 2, 4, 8, 16] {
        let instr = decoded([0x181b_7f89, 0x0c00_1f00 | (offset << 21), 0x000e_0000, 0]);
        let lane_imm = u16::try_from(offset).expect("small");
        assert_eq!(
            instr,
            Instr::new(shfl(
                ShflOp::Bfly,
                r(27),
                Pred::PT,
                r(24),
                ShflSrc::Imm(lane_imm),
                ShflSrc::Imm(0x1f)
            ))
        );
        let (ir, _) = accepted(COMPUTE, instr);
        assert_eq!(
            ir.to_string(),
            format!(
                "block0:\n\
                 \x20   %0: u32 = GetRegister r24\n\
                 \x20   %1: u32 = ShuffleButterfly %0, u32 {offset:#x}, u32 0x1f, u32 0x0\n\
                 \x20   %2 = SetRegister r27, %1\n\
                 \x20   return\n"
            )
        );
    }
}

/// PR #4588's "Native quad butterfly and swizzled add retain horizontal and vertical
/// differences": `SHFL.BFLY PT, R16, R5, 0x1 (or 0x2), 0x1c03`, the quad mask with butterfly
/// lane 1 (horizontal) or 2 (vertical).
#[test]
fn pr_quad_butterfly() {
    for (word1, from) in [(0x0c3c_0300_u32, 1_u16), (0x0c5c_0300, 2)] {
        let instr = decoded([0x0510_7f89, word1, 0x000e_0000, 0]);
        assert_eq!(
            instr,
            Instr::new(shfl(
                ShflOp::Bfly,
                r(16),
                Pred::PT,
                r(5),
                ShflSrc::Imm(from),
                ShflSrc::Imm(0x1c03)
            ))
        );
        let (ir, _) = accepted(COMPUTE, instr);
        assert_eq!(
            ir.to_string(),
            format!(
                "block0:\n\
                 \x20   %0: u32 = GetRegister r5\n\
                 \x20   %1: u32 = ShuffleButterfly %0, u32 {from:#x}, u32 0x3, u32 0x1c\n\
                 \x20   %2 = SetRegister r16, %1\n\
                 \x20   return\n"
            )
        );
        let (ir, _) = accepted(PIXEL, instr);
        assert_eq!(
            ir.to_string(),
            format!(
                "block0:\n\
                 \x20   %0: u32 = GetRegister r5\n\
                 \x20   %1: u32 = QuadSwap %0, u32 {:#x}\n\
                 \x20   %2 = SetRegister r16, %1\n\
                 \x20   return\n",
                from - 1
            )
        );
    }
}

#[test]
fn pr_indexed_shuffle() {
    let instr = decoded([0x181e_7589, 0x0000_1e1b, 0, 0]);
    assert_eq!(
        instr,
        Instr::new(shfl(
            ShflOp::Idx,
            r(30),
            p(0),
            r(24),
            ShflSrc::Reg(r(27)),
            ShflSrc::Imm(0x1e)
        ))
    );
    let (ir, _) = accepted(COMPUTE, instr);
    // The PR's targets 0, 15 and 30 run on the lane they name; 31 is past the clamp on any lane.
    for (active, target) in [(0, 0), (15, 15), (30, 30), (0, 31), (31, 31)] {
        let mut machine = lane(active);
        machine.regs[24] = 100 + active;
        machine.regs[27] = target;
        run(&ir, &mut machine);
        assert_eq!(
            machine.regs[30],
            100 + active,
            "lane {active} target {target}"
        );
        assert_eq!(
            machine.preds[0],
            target <= 30,
            "lane {active} target {target}"
        );
    }
}

#[test]
fn pr_immediate_indexed_shuffle() {
    for distance in [0_u32, 1, 16, 31] {
        let instr = decoded([0x0b02_7f89, (distance << 21) | 0x1f00, 0, 0]);
        let lane_imm = u16::try_from(distance).expect("small");
        assert_eq!(
            instr,
            Instr::new(shfl(
                ShflOp::Idx,
                r(2),
                p(0),
                r(11),
                ShflSrc::Imm(lane_imm),
                ShflSrc::Imm(0x1f)
            ))
        );
        let (ir, _) = accepted(COMPUTE, instr);
        let mut machine = lane(distance);
        machine.regs[11] = 100 + distance;
        run(&ir, &mut machine);
        assert_eq!(machine.regs[2], 100 + distance);
        assert!(machine.preds[0]);
    }
}

#[test]
fn pr_downward_shuffle() {
    for destination in [3_u32, 8] {
        for distance in [1_u32, 2, 16, 31] {
            let instr = decoded([
                (8 << 24) | (destination << 16) | 0x7f89,
                0x0800_1f00 | (distance << 21),
                0x000e_0000,
                0,
            ]);
            let (ir, _) = accepted(COMPUTE, instr);
            // On lane 31 every distance is out of range: the lane keeps its own value. On lane
            // 0, distance 31 reads lane 31, inactive here (the PR checks only active sources).
            let mut machine = lane(31);
            machine.regs[8] = 31 * 17 + 7;
            run(&ir, &mut machine);
            assert_eq!(machine.regs[destination as usize], 31 * 17 + 7);
            assert_eq!(machine.preds, [false; crate::interp::PRED_SLOTS]);
        }
    }
}

/// In a pixel program, the quad mask with an immediate lane is a quad operation, and the
/// predicate is set true; one step off each condition, it is a plain shuffle.
#[test]
fn quad_shuffles_in_pixel_programs() {
    let quad = |op: ShflOp, from: u16, c: u16, stage: GuestStage| {
        let instr = Instr::new(shfl(
            op,
            r(7),
            p(3),
            r(6),
            ShflSrc::Imm(from),
            ShflSrc::Imm(c),
        ));
        accepted(stage, instr).0.to_string()
    };
    let mask = u16::try_from(QUAD_SHUFFLE_MASK).expect("thirteen bits");
    assert_eq!(
        quad(ShflOp::Idx, QUAD_LAST_LANE, mask, PIXEL),
        "block0:\n\
         \x20   %0: u32 = GetRegister r6\n\
         \x20   %1: u32 = QuadBroadcast %0, u32 0x3\n\
         \x20   %2 = SetRegister r7, %1\n\
         \x20   %3 = SetPred p3, u1 true\n\
         \x20   return\n"
    );
    assert_eq!(
        quad(ShflOp::Bfly, 1, mask, PIXEL),
        "block0:\n\
         \x20   %0: u32 = GetRegister r6\n\
         \x20   %1: u32 = QuadSwap %0, u32 0x0\n\
         \x20   %2 = SetRegister r7, %1\n\
         \x20   %3 = SetPred p3, u1 true\n\
         \x20   return\n"
    );
    // Every butterfly lane of a quad, so `lane - 1` cannot be swapped for `lane ^ 1` (they
    // agree on lanes 1 and 3 only).
    for from in 1..=QUAD_LAST_LANE {
        let direction = u32::from(from) - 1;
        assert!(
            quad(ShflOp::Bfly, from, mask, PIXEL)
                .contains(&format!("QuadSwap %0, u32 {direction:#x}")),
            "butterfly lane {from}"
        );
    }
    assert!(quad(ShflOp::Bfly, 2, mask, PIXEL).contains("QuadSwap %0, u32 0x1"));
    assert!(quad(ShflOp::Idx, 0, mask, PIXEL).contains("QuadBroadcast %0, u32 0x0"));
    // One step off: lane 4, butterfly lane 0, another stage, another mask, up or down.
    let plain = [
        quad(ShflOp::Idx, QUAD_LAST_LANE + 1, mask, PIXEL),
        quad(ShflOp::Bfly, 0, mask, PIXEL),
        quad(ShflOp::Bfly, QUAD_LAST_LANE + 1, mask, PIXEL),
        quad(ShflOp::Idx, 0, mask, VERTEX),
        quad(ShflOp::Idx, 0, mask - 1, PIXEL),
        quad(ShflOp::Idx, 0, mask + 1, PIXEL),
        quad(ShflOp::Up, 1, mask, PIXEL),
        quad(ShflOp::Down, 1, mask, PIXEL),
    ];
    for text in plain {
        assert!(!text.contains("Quad"), "{text}");
        assert!(text.contains("GetInBoundsFromOp"), "{text}");
    }
    // A register lane is never a quad operation.
    let instr = Instr::new(shfl(
        ShflOp::Idx,
        r(7),
        p(3),
        r(6),
        ShflSrc::Reg(r(1)),
        ShflSrc::Imm(mask),
    ));
    assert!(!accepted(PIXEL, instr).0.to_string().contains("Quad"));
    // On one lane, a broadcast of the lane's own quad index reads its value.
    let instr = Instr::new(shfl(
        ShflOp::Idx,
        r(7),
        p(3),
        r(6),
        ShflSrc::Imm(2),
        ShflSrc::Imm(mask),
    ));
    let (ir, _) = accepted(PIXEL, instr);
    let mut machine = lane(6);
    machine.regs[6] = 0x66;
    run(&ir, &mut machine);
    assert_eq!((machine.regs[7], machine.preds[3]), (0x66, true));
}

#[test]
fn register_clamp_shuffles_are_refused() {
    for (op, lane) in [
        (ShflOp::Up, ShflSrc::Imm(1)),
        (ShflOp::Idx, ShflSrc::Reg(r(1))),
    ] {
        let instr = Instr::new(shfl(op, r(7), p(3), r(6), lane, ShflSrc::Reg(r(2))));
        assert_eq!(
            translated(COMPUTE, instr, false),
            refusal(instr, control(ControlUnsupported::ShuffleClampRegister))
        );
    }
    // PR #4588's upward word (`0x989`), which its lowering path accepts.
    for distance in [0_u32, 1, 16, 31] {
        let instr = decoded([0x0b02_7989, 0x0400_0000 | (distance << 21), 0xff, 0]);
        assert_eq!(
            translated(COMPUTE, instr, false),
            refusal(instr, control(ControlUnsupported::ShuffleClampRegister))
        );
    }
}

// ---------------------------------------------------------------------------------------------
// REDUX.

/// Every operation on one lane: the identity combined with the lane's value, which is the
/// value; the other 31 lanes' shuffles read [`INACTIVE_LANE_VALUE`] and are selected away.
#[test]
fn reductions_on_one_lane() {
    let ops = [
        ReduxOp::And,
        ReduxOp::Or,
        ReduxOp::Xor,
        ReduxOp::Sum,
        ReduxOp::Min { signed: false },
        ReduxOp::Min { signed: true },
        ReduxOp::Max { signed: false },
        ReduxOp::Max { signed: true },
    ];
    for op in ops {
        let (ir, emitted) = accepted(COMPUTE, Instr::new(redux(op, ur(5), r(3))));
        assert!(emitted.needs_subgroup_size_32);
        for active in [0_u32, 31] {
            for value in [0_u32, 7, 0x8000_0000, 0xffff_fff0] {
                let mut machine = lane(active);
                machine.regs[3] = value;
                machine.uniform_regs[5] = 0xcafe_beef;
                run(&ir, &mut machine);
                assert_eq!(machine.uniform_regs[5], value, "{op:?} lane {active}");
            }
        }
    }
}

#[test]
fn pr_warp_reduction() {
    let words = [
        (0x0000_c000, ReduxOp::Sum),
        (0x0000_4000, ReduxOp::Or),
        (0x0001_0000, ReduxOp::Min { signed: false }),
        (0x0001_4000, ReduxOp::Max { signed: false }),
    ];
    for (w2, op) in words {
        let instr = decoded([0x0305_73c4, 0, w2, 0]);
        assert_eq!(instr, Instr::new(redux(op, ur(5), r(3))));
        let (ir, _) = accepted(COMPUTE, instr);
        for active in [0_u32, 31] {
            let mut machine = lane(active);
            let value = 0xffff_fff0_u32.wrapping_add(active * 17);
            machine.regs[3] = value;
            pr_predicates(&mut machine, active);
            let preds = machine.preds;
            run(&ir, &mut machine);
            assert_eq!(machine.uniform_regs[5], value, "{op:?}");
            assert_eq!(machine.preds, preds);
        }
    }
}

/// Each operation's first lane combines its identity with lane 0's value by the operation
/// `Visitor::REDUX` names (one lane cannot tell `SUM` from `OR`, so this pins the opcode).
#[test]
fn each_reduction_combines_with_its_operation() {
    let cases = [
        (ReduxOp::And, "BitwiseAnd32 u32 0xffffffff, %5"),
        (ReduxOp::Or, "BitwiseOr32 u32 0x0, %5"),
        (ReduxOp::Xor, "BitwiseXor32 u32 0x0, %5"),
        (ReduxOp::Sum, "IAdd32 u32 0x0, %5"),
        (ReduxOp::Min { signed: false }, "UMin32 u32 0xffffffff, %5"),
        (ReduxOp::Min { signed: true }, "SMin32 u32 0x7fffffff, %5"),
        (ReduxOp::Max { signed: false }, "UMax32 u32 0x0, %5"),
        (ReduxOp::Max { signed: true }, "SMax32 u32 0x80000000, %5"),
    ];
    for (op, first) in cases {
        let (ir, _) = accepted(COMPUTE, Instr::new(redux(op, ur(5), r(3))));
        let text = ir.to_string();
        assert!(
            text.contains(&format!("    %6: u32 = {first}\n")),
            "{op:?}: {text}"
        );
    }
}

/// The reduction walks all 32 lanes in order; its head and tail are pinned here, and the
/// count of each operation shows one iteration per lane.
#[test]
fn reduction_golden_ir() {
    let (ir, _) = accepted(
        COMPUTE,
        Instr::new(redux(ReduxOp::Min { signed: true }, ur(5), r(3))),
    );
    let text = ir.to_string();
    let head = "block0:\n\
                \x20   %0: u32 = GetRegister r3\n\
                \x20   %1: u32 = SubgroupBallot u1 true\n\
                \x20   %2: u32 = ShiftRightLogical32 %1, u32 0x0\n\
                \x20   %3: u32 = BitwiseAnd32 %2, u32 0x1\n\
                \x20   %4: u1 = INotEqual %3, u32 0x0\n\
                \x20   %5: u32 = ShuffleIndex %0, u32 0x0, u32 0x1f, u32 0x0\n\
                \x20   %6: u32 = SMin32 u32 0x7fffffff, %5\n\
                \x20   %7: u32 = SelectU32 %4, %6, u32 0x7fffffff\n\
                \x20   %8: u32 = ShiftRightLogical32 %1, u32 0x1\n";
    let tail = "    %188: u32 = ShiftRightLogical32 %1, u32 0x1f\n\
                \x20   %189: u32 = BitwiseAnd32 %188, u32 0x1\n\
                \x20   %190: u1 = INotEqual %189, u32 0x0\n\
                \x20   %191: u32 = ShuffleIndex %0, u32 0x1f, u32 0x1f, u32 0x0\n\
                \x20   %192: u32 = SMin32 %187, %191\n\
                \x20   %193: u32 = SelectU32 %190, %192, %187\n\
                \x20   %194 = SetUniformRegister ur5, %193\n\
                \x20   return\n";
    assert!(text.starts_with(head), "{text}");
    assert!(text.ends_with(tail), "{text}");
    for (name, count) in [
        ("ShuffleIndex", 32),
        ("SelectU32", 32),
        ("SMin32", 32),
        ("SubgroupBallot", 1),
    ] {
        assert_eq!(text.matches(name).count(), count, "{name}");
    }
}

#[test]
fn a_reduction_into_urz_is_refused() {
    let instr = Instr::new(redux(ReduxOp::Sum, UniformReg::URZ, r(3)));
    assert_eq!(
        translated(COMPUTE, instr, false),
        refusal(instr, control(ControlUnsupported::ReduxToUrz))
    );
    // The last uniform register before `URZ` is accepted.
    let last = ur(UniformReg::URZ.index() - 1);
    assert!(translated(COMPUTE, Instr::new(redux(ReduxOp::Sum, last, r(3))), false).is_ok());
}

// ---------------------------------------------------------------------------------------------
// MATCH.

#[test]
fn pr_warp_match() {
    for wide in [false, true] {
        for target in [0_u8, 1, 5] {
            let source = if wide { 0 } else { 3 };
            let instr = Instr::new(match_all(wide, r(source), p(target)));
            // The word round-trips through the decoder.
            assert_eq!(decode(assemble(&instr).expect("encodes")), Ok(instr));
            let (ir, emitted) = accepted(COMPUTE, instr);
            assert!(emitted.needs_subgroup_size_32);
            for active in [0_u32, 31] {
                let mut machine = lane(active);
                machine.regs[usize::from(source)] = 0xffff_fff0;
                machine.regs[usize::from(source) + 1] = 0x21;
                pr_predicates(&mut machine, active);
                let mut expected = machine.preds;
                expected[usize::from(target)] = true;
                run(&ir, &mut machine);
                assert_eq!(machine.preds, expected, "wide {wide} target {target}");
            }
        }
    }
}

#[test]
fn match_golden_ir() {
    let (ir, _) = accepted(COMPUTE, Instr::new(match_all(true, r(4), p(2))));
    assert_eq!(
        ir.to_string(),
        "block0:\n\
         \x20   %0: u32 = SubgroupBallot u1 true\n\
         \x20   %1: u32 = INeg32 %0\n\
         \x20   %2: u32 = BitwiseAnd32 %0, %1\n\
         \x20   %3: u32 = FindUMsb32 %2\n\
         \x20   %4: u32 = GetRegister r4\n\
         \x20   %5: u32 = ShuffleIndex %4, %3, u32 0x1f, u32 0x0\n\
         \x20   %6: u1 = IEqual %4, %5\n\
         \x20   %7: u32 = GetRegister r5\n\
         \x20   %8: u32 = ShuffleIndex %7, %3, u32 0x1f, u32 0x0\n\
         \x20   %9: u1 = IEqual %7, %8\n\
         \x20   %10: u1 = LogicalAnd %6, %9\n\
         \x20   %11: u1 = VoteAll %10\n\
         \x20   %12 = SetPred p2, %11\n\
         \x20   return\n"
    );
    let (narrow, _) = accepted(COMPUTE, Instr::new(match_all(false, r(4), Pred::PT)));
    let text = narrow.to_string();
    assert!(!text.contains("r5") && !text.contains("LogicalAnd") && !text.contains("SetPred"));
    assert!(text.contains("VoteAll %6"), "{text}");
}

#[test]
fn match_checks() {
    let refused = |op: Op, what: ControlUnsupported| {
        let instr = Instr::new(op);
        assert_eq!(
            translated(COMPUTE, instr, false),
            refusal(instr, control(what))
        );
    };
    // The mode: a mask register, `.ANY`.
    let mut with_mask = OpMatch {
        op: MatchOp::All,
        u64: false,
        mask: r(1),
        src: r(4),
        pred: p(2),
    };
    refused(Op::Match(with_mask), ControlUnsupported::MatchMode);
    with_mask.mask = Gpr::RZ;
    refused(
        Op::Match(OpMatch {
            op: MatchOp::Any,
            ..with_mask
        }),
        ControlUnsupported::MatchMode,
    );
    // The source: `RZ`; for 64 bits, an odd register or the pair from R254.
    refused(
        match_all(false, Gpr::RZ, p(2)),
        ControlUnsupported::MatchSource {
            src: Gpr::RZ,
            wide: false,
        },
    );
    for src in [r(5), r(254), Gpr::RZ] {
        refused(
            match_all(true, src, p(2)),
            ControlUnsupported::MatchSource { src, wide: true },
        );
    }
    // The edges accepted: R254 narrow, R252 (with R253) wide.
    assert!(translated(COMPUTE, Instr::new(match_all(false, r(254), p(2))), false).is_ok());
    assert!(translated(COMPUTE, Instr::new(match_all(true, r(252), p(2))), false).is_ok());
    // Both faults: the mode is reported first.
    refused(
        Op::Match(OpMatch {
            op: MatchOp::Any,
            u64: true,
            mask: r(1),
            src: r(5),
            pred: p(2),
        }),
        ControlUnsupported::MatchMode,
    );
}

// ---------------------------------------------------------------------------------------------
// BAR.

#[test]
fn bar_sync_is_a_barrier() {
    for defer_blocking in [false, true] {
        let instr = Instr::new(Op::Bar(OpBar {
            defer_blocking,
            ..bar_sync()
        }));
        let (ir, emitted) = accepted(COMPUTE, instr);
        assert!(!emitted.needs_subgroup_size_32);
        assert_eq!(ir.to_string(), "block0:\n    %0 = Barrier\n    return\n");
    }
}

#[test]
fn pr_deferred_blocking_barrier() {
    for (w2, defer_blocking) in [(0, false), (0x1_0000, true)] {
        let instr = decoded([0x7b1d, 0, w2, 0]);
        assert_eq!(
            instr,
            Instr::new(Op::Bar(OpBar {
                defer_blocking,
                ..bar_sync()
            }))
        );
        let (ir, _) = accepted(COMPUTE, instr);
        assert_eq!(ir.to_string().matches("Barrier").count(), 1);
        // The PR's invalid words: barrier 1 is refused here; bit 64 by the decoder.
        let id_one = decoded([0x7b1d, 1, w2, 0]);
        assert_eq!(
            translated(COMPUTE, id_one, false),
            refusal(id_one, control(ControlUnsupported::BarrierMode))
        );
        assert!(matches!(
            decode(word([0x7b1d, 0, w2 | 1, 0])),
            Err(DecodeError::UnknownBit { bit: 64, .. })
        ));
    }
}

/// Each field of `BAR` off its one accepted value, alone.
#[test]
fn other_barriers_are_refused() {
    let sync = bar_sync();
    let mut faults = vec![
        OpBar { id: 1, ..sync },
        OpBar {
            id: u8::MAX,
            ..sync
        },
        OpBar { unknown: 1, ..sync },
    ];
    faults.extend(
        BarMode::ALL
            .into_iter()
            .filter(|mode| *mode != BarMode::Sync)
            .map(|mode| OpBar { mode, ..sync }),
    );
    faults.extend(
        BarRed::ALL
            .into_iter()
            .filter(|red| *red != BarRed::Popc)
            .map(|red| OpBar { red, ..sync }),
    );
    for bar in faults {
        let instr = Instr::new(Op::Bar(bar));
        assert_eq!(
            translated(COMPUTE, instr, false),
            refusal(instr, control(ControlUnsupported::BarrierMode)),
            "{bar:?}"
        );
    }
}

// ---------------------------------------------------------------------------------------------
// Instructions that become no IR.

#[test]
fn instructions_with_no_ir() {
    for op in no_ir_forms() {
        for stage in [PIXEL, COMPUTE] {
            let (ir, emitted) = accepted(stage, Instr::new(op));
            assert_eq!(ir.to_string(), "block0:\n    return\n", "{op:?}");
            assert_eq!(emitted, Emitted::default(), "{op:?}");
        }
    }
}

/// A guard does not matter to an instruction that becomes no IR, tested or not.
#[test]
fn guards_on_instructions_with_no_ir_are_ignored() {
    for op in no_ir_forms() {
        for guard_tested in [false, true] {
            let (ir, _) = translated(COMPUTE, guarded(not(3), op), guard_tested).expect("accepted");
            assert_eq!(ir.to_string(), "block0:\n    return\n", "{op:?}");
        }
    }
}

#[test]
fn pr_dependency_wait_counts() {
    for count in [0_u32, 1, 15, 22, 63] {
        let instr = decoded([0x0000_791a, 0xd000 | (count << 6), 0, 0]);
        let count = u8::try_from(count).expect("six bits");
        assert_eq!(
            instr,
            Instr::new(depbar(DEPBAR_REQUIRED_SCOREBOARD, true, count))
        );
        assert_eq!(
            accepted(COMPUTE, instr).0.to_string(),
            "block0:\n    return\n"
        );
    }
    let unsupported = decoded([0x0000_791a, 0xc000, 0, 0]);
    assert_eq!(unsupported, Instr::new(depbar(4, true, 0)));
    assert_eq!(
        translated(COMPUTE, unsupported, false),
        refusal(unsupported, control(ControlUnsupported::DepBarMode))
    );
}

#[test]
fn depbar_needs_le_sb5() {
    // Every other scoreboard, and scoreboard 5 without `.LE`.
    let mut faults: Vec<Op> = (0..8)
        .filter(|scoreboard| *scoreboard != DEPBAR_REQUIRED_SCOREBOARD)
        .map(|scoreboard| depbar(scoreboard, true, 1))
        .collect();
    faults.push(depbar(DEPBAR_REQUIRED_SCOREBOARD, false, 1));
    for op in faults {
        let instr = Instr::new(op);
        assert_eq!(
            translated(COMPUTE, instr, false),
            refusal(instr, control(ControlUnsupported::DepBarMode)),
            "{op:?}"
        );
    }
    // The PR's lowering path accepts `0x8000` (scoreboard 0 with `.LE`) and `0x800e`; the
    // first is refused here, the second already by the decoder (bits 33..35).
    let sb0 = decoded([0x791a, 0x8000, 0, 0]);
    assert_eq!(
        translated(COMPUTE, sb0, false),
        refusal(sb0, control(ControlUnsupported::DepBarMode))
    );
    assert!(matches!(
        decode(word([0x791a, 0x800e, 0, 0])),
        Err(DecodeError::UnknownBit { bit: 33, .. })
    ));
}

#[test]
fn warpsync_checks() {
    let refused = |op: Op, what: ControlUnsupported| {
        let instr = Instr::new(op);
        assert_eq!(
            translated(COMPUTE, instr, false),
            refusal(instr, control(what))
        );
    };
    for mask in [
        AluSrc::reg(r(1)),
        AluSrc::UniformReg {
            reg: ur(1),
            abs: false,
            neg: false,
        },
        AluSrc::CBuf {
            cbuf: crate::decode::CBufRef {
                buf: crate::decode::CBuf::Binding(1),
                offset: 0x10,
            },
            abs: false,
            neg: false,
        },
    ] {
        refused(
            warpsync(mask, PredSrc::TRUE),
            ControlUnsupported::WarpSyncMask,
        );
    }
    for mask in [0, 1, FULL_WARP_MASK - 1, FULL_WARP_MASK >> 1] {
        refused(
            warpsync(AluSrc::Imm(mask), PredSrc::TRUE),
            ControlUnsupported::PartialWarpSync(mask),
        );
    }
    for cond in [src(0), not(0), NEVER] {
        refused(
            warpsync(AluSrc::Imm(FULL_WARP_MASK), cond),
            ControlUnsupported::PredicatedWarpSync,
        );
    }
    // In the PR's order: the form, the mask, the predicate.
    refused(
        warpsync(AluSrc::reg(r(1)), src(0)),
        ControlUnsupported::WarpSyncMask,
    );
    refused(
        warpsync(AluSrc::Imm(1), src(0)),
        ControlUnsupported::PartialWarpSync(1),
    );
}

#[test]
fn pr_warp_sync_words() {
    for token in [0_u32, 2, 7] {
        let bssy = decoded([0x7945 | (token << 16), 16, 0x0380_0000, 0]);
        let bsync = decoded([0x7941 | (token << 16), 0, 0x0380_0000, 0]);
        assert!(matches!(bssy.op, Op::BSSy(_)) && matches!(bsync.op, Op::BSync(_)));
        for instr in [bssy, bsync] {
            assert_eq!(
                accepted(COMPUTE, instr).0.to_string(),
                "block0:\n    return\n"
            );
        }
    }
    let warp_sync = decoded([0x7948, 0xffff_ffff, 0x0380_0000, 0]);
    assert_eq!(
        warp_sync,
        Instr::new(warpsync(AluSrc::Imm(FULL_WARP_MASK), PredSrc::TRUE))
    );
    assert_eq!(
        accepted(COMPUTE, warp_sync).0.to_string(),
        "block0:\n    return\n"
    );
    let broken = decoded([0x7948, 1, 0x0380_0000, 0]);
    assert_eq!(
        translated(COMPUTE, broken, false),
        refusal(broken, control(ControlUnsupported::PartialWarpSync(1)))
    );
    let nop = decoded([0x7918, 0, 0, 0]);
    assert_eq!(nop, Instr::new(Op::Nop(OpNop {})));
}

#[test]
fn branch_forms_with_their_checks() {
    let offset = BranchOffset::from_words(4);
    let cases = [
        (
            Op::Kill(OpKill { pred: src(1) }),
            ControlUnsupported::PredicatedKill,
        ),
        (
            Op::Kill(OpKill { pred: NEVER }),
            ControlUnsupported::PredicatedKill,
        ),
        (
            Op::Call(OpCall {
                offset,
                no_inc: false,
                cond: PredSrc::TRUE,
            }),
            ControlUnsupported::CallIncrement,
        ),
        (
            Op::Ret(OpRet {
                src: r(6),
                offset,
                no_dec: false,
                cond: PredSrc::TRUE,
            }),
            ControlUnsupported::ReturnDecrement,
        ),
        (
            Op::Bra(OpBra {
                cond: BraCond::Pred(PredSrc::TRUE),
                uniform: false,
                offset,
            }),
            ControlUnsupported::Branch,
        ),
    ];
    for (op, what) in cases {
        let instr = Instr::new(op);
        assert_eq!(
            translated(COMPUTE, instr, false),
            refusal(instr, control(what))
        );
    }
}

// ---------------------------------------------------------------------------------------------
// Guards.

/// `@!PT` never runs: nothing, whatever the instruction and its fields.
#[test]
fn never_guarded_instructions_become_nothing() {
    let faulty = [
        Op::Bar(OpBar {
            id: 1,
            ..bar_sync()
        }),
        warpsync(AluSrc::Imm(1), src(0)),
        depbar(0, false, 0),
        Op::Kill(OpKill { pred: src(1) }),
        vote(VoteOp::All, r(5), p(2), src(1)),
        redux(ReduxOp::Sum, UniformReg::URZ, r(3)),
        Op::Exit(OpExit {
            keep_ref_count: false,
            no_at_exit: false,
            pred: PredSrc::TRUE,
        }),
        Op::Nanosleep(OpNanosleep {
            time: AluSrc::Imm(1),
            clear: false,
            warp: false,
            rand: false,
            cond: PredSrc::TRUE,
        }),
    ];
    for op in faulty {
        for guard_tested in [false, true] {
            let (ir, emitted) =
                translated(COMPUTE, guarded(NEVER, op), guard_tested).expect("never runs");
            assert_eq!(ir.to_string(), "block0:\n    return\n", "{op:?}");
            assert_eq!(emitted, Emitted::default());
        }
    }
}

/// An instruction that writes IR under an untested guard is refused before its own checks;
/// tested, the guard is ignored; under `PT` it was never there.
#[test]
fn untested_guards_are_refused() {
    let writing = [
        Op::Bar(bar_sync()),
        vote(VoteOp::All, r(5), p(2), src(1)),
        shfl(
            ShflOp::Idx,
            r(7),
            p(3),
            r(6),
            ShflSrc::Imm(0),
            ShflSrc::Imm(0x1f),
        ),
        redux(ReduxOp::Sum, ur(5), r(3)),
        match_all(false, r(4), p(2)),
    ];
    for op in writing {
        for guard in [src(0), not(6)] {
            let instr = guarded(guard, op);
            assert_eq!(
                translated(COMPUTE, instr, false),
                refusal(instr, control(ControlUnsupported::UntestedGuard)),
                "{op:?}"
            );
            let tested = translated(COMPUTE, instr, true).expect("tested").0;
            let plain = accepted(COMPUTE, Instr::new(op)).0;
            assert_eq!(tested.to_string(), plain.to_string(), "{op:?}");
        }
    }
    // Both faults: the guard is reported before the instruction's own check.
    let both = guarded(
        src(0),
        Op::Bar(OpBar {
            id: 1,
            ..bar_sync()
        }),
    );
    assert_eq!(
        translated(COMPUTE, both, false),
        refusal(both, control(ControlUnsupported::UntestedGuard))
    );
    let both = guarded(src(0), redux(ReduxOp::Sum, UniformReg::URZ, r(3)));
    assert_eq!(
        translated(COMPUTE, both, false),
        refusal(both, control(ControlUnsupported::UntestedGuard))
    );
}

/// A tested guard is ignored by a `VOTE` that runs: the vote's own result does not depend on
/// the guard's predicate.
#[test]
fn a_tested_guard_is_ignored() {
    let instr = guarded(src(6), vote(VoteOp::All, r(5), p(2), src(1)));
    let (ir, _) = translated(COMPUTE, instr, true).expect("tested");
    for guard in [false, true] {
        let mut machine = lane(3);
        machine.preds[6] = guard;
        machine.preds[1] = true;
        run(&ir, &mut machine);
        assert_eq!(machine.regs[5], 1 << 3);
        assert!(machine.preds[2]);
    }
}

// ---------------------------------------------------------------------------------------------
// Other instructions, refusals in words, and the block left untouched.

#[test]
fn other_instructions_are_refused_by_name() {
    let shells_and_unsupported = [
        Op::BMov(OpBMov {
            dir: BMovDir::ToBarrier {
                bar: Barrier::new(0).expect("a barrier"),
                src: r(1),
            },
            clear: false,
        }),
        Op::Nanosleep(OpNanosleep {
            time: AluSrc::Imm(1),
            clear: false,
            warp: false,
            rand: false,
            cond: PredSrc::TRUE,
        }),
        Op::Jmx(crate::decode::OpJmx {}),
        Op::Bpt(crate::decode::OpBpt {}),
    ];
    for op in shells_and_unsupported {
        let instr = Instr::new(op);
        assert_eq!(
            translated(COMPUTE, instr, false),
            refusal(instr, Unsupported::Instruction(op.mnemonic()))
        );
    }
    let other_groups = [
        Op::Exit(OpExit {
            keep_ref_count: false,
            no_at_exit: false,
            pred: PredSrc::TRUE,
        }),
        Op::Mov(OpMov {
            dst: r(1),
            src: crate::decode::MovSrc::Imm(1),
            quad_lanes: OpMov::ALL_LANES,
        }),
    ];
    for op in other_groups {
        let instr = Instr::new(op);
        assert_eq!(
            translated(COMPUTE, instr, false),
            refusal(
                instr,
                control(ControlUnsupported::OtherGroup(op.mnemonic()))
            )
        );
    }
}

#[test]
fn refusals_read_as_words() {
    let cases = [
        (
            ControlUnsupported::UntestedGuard,
            "a guarded instruction outside its conditional block",
        ),
        (ControlUnsupported::Branch, "a bra inside a block"),
        (
            ControlUnsupported::PredicatedKill,
            "a kill with a predicate of its own",
        ),
        (ControlUnsupported::CallIncrement, "a call without .noinc"),
        (ControlUnsupported::ReturnDecrement, "a ret without .nodec"),
        (
            ControlUnsupported::WarpSyncMask,
            "a warpsync whose mask is not an immediate",
        ),
        (
            ControlUnsupported::PartialWarpSync(0),
            "a warpsync of lane mask 0x0",
        ),
        (
            ControlUnsupported::PartialWarpSync(0x7fff_ffff),
            "a warpsync of lane mask 0x7fffffff",
        ),
        (
            ControlUnsupported::PredicatedWarpSync,
            "a warpsync with a predicate of its own",
        ),
        (
            ControlUnsupported::DepBarMode,
            "a depbar other than depbar.le sb5",
        ),
        (
            ControlUnsupported::BarrierMode,
            "a bar other than bar.sync 0x0",
        ),
        (
            ControlUnsupported::ShuffleClampRegister,
            "a shfl with its clamp in a register",
        ),
        (ControlUnsupported::ReduxToUrz, "a redux into urz"),
        (
            ControlUnsupported::MatchMode,
            "a match with a mask register or .any",
        ),
        (
            ControlUnsupported::MatchSource {
                src: Gpr::RZ,
                wide: false,
            },
            "a match of rz",
        ),
        (
            ControlUnsupported::MatchSource {
                src: r(5),
                wide: true,
            },
            "a 64-bit match of r5",
        ),
        (
            ControlUnsupported::OtherGroup("exit"),
            "exit in the control and warp translator",
        ),
    ];
    for (what, words) in cases {
        assert_eq!(what.to_string(), words);
    }
    let instr = Instr::new(Op::Kill(OpKill { pred: src(1) }));
    let error = TranslateError::Unsupported {
        index: 7,
        instr,
        what: control(ControlUnsupported::PredicatedKill),
    };
    assert_eq!(
        error.to_string(),
        format!("instruction 7 ({instr}): a kill with a predicate of its own is not supported")
    );
}

/// A refused instruction leaves the block as it was, and the next instruction translates.
#[test]
fn refusals_write_nothing() {
    let refused = [
        (
            Op::Bar(OpBar {
                id: 1,
                ..bar_sync()
            }),
            ControlUnsupported::BarrierMode,
        ),
        (
            redux(ReduxOp::Sum, UniformReg::URZ, r(3)),
            ControlUnsupported::ReduxToUrz,
        ),
        (
            match_all(true, r(5), p(2)),
            ControlUnsupported::MatchSource {
                src: r(5),
                wide: true,
            },
        ),
        (
            shfl(
                ShflOp::Idx,
                r(7),
                p(3),
                r(6),
                ShflSrc::Imm(0),
                ShflSrc::Reg(r(1)),
            ),
            ControlUnsupported::ShuffleClampRegister,
        ),
        (
            warpsync(AluSrc::Imm(1), PredSrc::TRUE),
            ControlUnsupported::PartialWarpSync(1),
        ),
    ];
    for (op, what) in refused {
        let mut function = Function::new();
        let entry = function.entry();
        let first = translate_control(
            &mut function,
            entry,
            COMPUTE,
            0,
            Instr::new(vote(VoteOp::All, r(5), p(2), src(1))),
            false,
        )
        .expect("translates");
        assert!(first.needs_subgroup_size_32);
        let before = function.clone();
        assert_eq!(
            translate_control(&mut function, entry, COMPUTE, 1, Instr::new(op), false),
            Err(TranslateError::Unsupported {
                index: 1,
                instr: Instr::new(op),
                what: control(what),
            }),
            "{op:?}"
        );
        assert_eq!(function, before, "{op:?}");
        let next = translate_control(
            &mut function,
            entry,
            COMPUTE,
            2,
            Instr::new(Op::Bar(bar_sync())),
            false,
        )
        .expect("translates");
        assert!(!next.needs_subgroup_size_32);
    }
}

/// The index given is the one an error names.
#[test]
fn errors_name_the_index_given() {
    let mut function = Function::new();
    let entry = function.entry();
    let instr = Instr::new(Op::Kill(OpKill { pred: src(1) }));
    assert_eq!(
        translate_control(&mut function, entry, COMPUTE, 41, instr, true),
        Err(TranslateError::Unsupported {
            index: 41,
            instr,
            what: control(ControlUnsupported::PredicatedKill),
        })
    );
}

#[test]
fn a_block_that_does_not_exist_is_an_error() {
    let mut function = Function::new();
    let missing = {
        let mut other = Function::new();
        other.add_block().expect("a block")
    };
    assert_eq!(
        translate_control(
            &mut function,
            missing,
            COMPUTE,
            0,
            Instr::new(Op::Nop(OpNop {})),
            false
        ),
        Err(TranslateError::Build(BuildError::NoBlock(missing)))
    );
}

/// Subgroup operations written by an earlier instruction of the block do not count for a
/// later one.
#[test]
fn needs_subgroup_counts_only_this_instructions_ir() {
    let mut function = Function::new();
    let entry = function.entry();
    let vote_op = Instr::new(vote(VoteOp::All, r(5), p(2), src(1)));
    let first = translate_control(&mut function, entry, COMPUTE, 0, vote_op, false).expect("ok");
    let second = translate_control(
        &mut function,
        entry,
        COMPUTE,
        1,
        Instr::new(Op::Nop(OpNop {})),
        false,
    )
    .expect("ok");
    assert!(first.needs_subgroup_size_32);
    assert!(!second.needs_subgroup_size_32);
}

// ---------------------------------------------------------------------------------------------
// Through flow analysis.

/// Fills the IR blocks of `program` (assembled) with this translator, skipping the `EXIT`s
/// (slice 1's), and returns the function.
fn through_flow(program: &[Instr], stage: GuestStage) -> Function {
    let code: Vec<Instruction> = program
        .iter()
        .map(|instr| assemble(instr).expect("encodes"))
        .collect();
    let graph = flow_graph(&code, &BranchTables::new(0)).expect("flow");
    let mut ir = graph.build_ir().expect("IR blocks");
    for source in &ir.sources {
        let block = graph.block(source.flow).expect("a flow block");
        for index in block.code.clone() {
            let slot = usize::try_from(index).expect("an index");
            let instr = program[slot];
            if matches!(instr.op, Op::Exit(_)) {
                continue;
            }
            let _emitted = translate_control(
                &mut ir.function,
                source.block,
                stage,
                slot,
                instr,
                block.guard_tested,
            )
            .expect("translates");
        }
    }
    ir.function.verify().expect("verifies");
    ir.function
}

fn exit_instr() -> Instr {
    Instr::new(Op::Exit(OpExit {
        keep_ref_count: false,
        no_at_exit: false,
        pred: PredSrc::TRUE,
    }))
}

/// A guarded `KILL` gets its own block from flow analysis, which demotes; the `KILL` itself
/// becomes nothing.
#[test]
fn kill_demotes_through_flow_analysis() {
    let program = [
        guarded(
            src(0),
            Op::Kill(OpKill {
                pred: PredSrc::TRUE,
            }),
        ),
        exit_instr(),
    ];
    let function = through_flow(&program, PIXEL);
    assert_eq!(
        function
            .to_string()
            .matches("DemoteToHelperInvocation")
            .count(),
        1
    );
    for kill in [false, true] {
        let mut machine = Machine::new();
        machine.preds[0] = kill;
        run(&function, &mut machine);
        assert_eq!(machine.demoted, kill);
    }
}

/// A guarded `VOTE` gets its own conditional block whose guard is tested, so the translator
/// writes it unguarded and it runs only where the guard holds.
#[test]
fn a_guarded_vote_runs_in_its_conditional_block() {
    let program = [
        guarded(src(1), vote(VoteOp::All, r(5), p(2), src(0))),
        exit_instr(),
    ];
    let function = through_flow(&program, COMPUTE);
    for guard in [false, true] {
        let mut machine = lane(4);
        machine.preds[1] = guard;
        machine.preds[0] = true;
        machine.regs[5] = 0x55;
        run(&function, &mut machine);
        assert_eq!(machine.regs[5], if guard { 1 << 4 } else { 0x55 });
        assert_eq!(machine.preds[2], guard);
    }
}

/// A guarded `VOTEU` stays in its block, and its guard becomes a select.
#[test]
fn a_guarded_uniform_vote_stays_in_its_block() {
    let program = [
        guarded(src(2), voteu(VoteOp::Any, ur(1), up(1), src(0))),
        exit_instr(),
    ];
    let function = through_flow(&program, COMPUTE);
    assert!(function.to_string().contains("SelectU32"));
    for guard in [false, true] {
        let mut machine = lane(4);
        machine.uniform_preds[2] = guard;
        machine.preds[0] = true;
        machine.uniform_regs[1] = 0x55;
        run(&function, &mut machine);
        assert_eq!(machine.uniform_regs[1], if guard { 1 << 4 } else { 0x55 });
    }
}

// ---------------------------------------------------------------------------------------------
// Generated instructions.

/// Generators of the control group's decoded operations, found by its `VOTEU` form.
fn control_ops() -> Vec<proptest::strategy::BoxedStrategy<Op>> {
    let group = GROUPS
        .iter()
        .position(|forms| forms.iter().any(|form| form.opcode == VOTEU_OPCODE))
        .expect("the control group");
    (GROUP_HOOKS[group].any_ops)()
}

proptest! {
    #![proptest_config(config())]

    /// Arbitrary control and warp instructions never panic the translator; what it accepts
    /// verifies, and what it refuses leaves the block empty.
    #[test]
    fn generated_instructions_never_panic(
        guard in any_pred_src(),
        op in proptest::strategy::Union::new(control_ops()),
        guard_tested in any::<bool>(),
        stage_index in 0..3_usize,
    ) {
        let stage = [VERTEX, PIXEL, COMPUTE][stage_index];
        let instr = Instr { guard, op };
        let mut function = Function::new();
        let entry = function.entry();
        match translate_control(&mut function, entry, stage, 0, instr, guard_tested) {
            Ok(_) => {
                Builder::new(&mut function, entry)
                    .terminate(Terminator::Return)
                    .map_err(|error| TestCaseError::fail(error.to_string()))?;
                prop_assert_eq!(function.verify(), Ok(()));
            }
            Err(TranslateError::Unsupported { .. }) => {
                let empty: &[Value] = &[];
                prop_assert_eq!(
                    function.block(entry).map(|block| block.insts.as_slice()),
                    Some(empty)
                );
            }
            Err(other) => prop_assert!(false, "{}", other),
        }
    }
}

/// [`in_group`], the translation driver's test for this group, names exactly the operations
/// [`translate_control`] takes as its own: for every decoded operation (one instruction of each,
/// at least, from the driver's deterministic `decoded_samples`), `in_group` holds exactly when
/// the translator does not refuse it as another group's.
#[test]
fn in_group_matches_the_dispatch() {
    for instr in crate::translate::driver::tests::decoded_samples() {
        let mut function = Function::new();
        let entry = function.entry();
        // Unguarded: an instruction under `!PT` becomes nothing in every group.
        let instr = Instr {
            guard: PredSrc::TRUE,
            ..instr
        };
        let result = translate_control(&mut function, entry, VERTEX, 0, instr, true);
        let other_group = matches!(
            result,
            Err(TranslateError::Unsupported {
                what: Unsupported::Control(ControlUnsupported::OtherGroup(_)),
                ..
            })
        );
        assert_eq!(in_group(&instr.op), !other_group, "{instr}");
    }
}
