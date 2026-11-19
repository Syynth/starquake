//! Instruction translation for the control and warp group (spec, port slice 12h): the
//! instructions slice 6h decodes (`decode`'s `OpBra` to `OpNanosleep`) into the IR, one
//! instruction of a flow block at a time ([`translate_control`]).
//!
//! # Sources
//!
//! Ported by behaviour from PR #4588's native frontend
//! (`eden-pr4588/src/shader_recompiler/frontend/ampere/` @ e5fb3b97, GPL-3.0-or-later,
//! Copyright 2026 Eden Emulator Project): `Visitor::BAR`, `Visitor::VOTE`, `Visitor::VOTEU`,
//! `Visitor::SHFL` and `Visitor::REDUX` from `ops_memory.cpp`, `Visitor::MATCH` from
//! `ops_misc.cpp`, `Visitor::ShuffleLane` from `ops_half.cpp`, and the cases of
//! `Visitor::Run` in `visitor.cpp` for `NOP` (`0x918`), `BREAK` (`0x942`), `WARPSYNC`
//! (`0x948`), `DEPBAR` (`0x91a`), `KILL` (`0x95b`), `CALL` (`0x944`), `RET` (`0x950`), `BRX`
//! (`0x949`), `BRXU` (`0x958`), `BSSY` (`0x945`) and `BSYNC` (`0x941`), with the guard rules
//! of `Visitor::Run` (`guard.Never()`, `guard_select`) and the register and predicate writes
//! of `Visitor::SetPred`, `Visitor::UR` and `Visitor::SetUPred`. Cross-checked with Eden's
//! Maxwell translators (`eden/src/shader_recompiler/frontend/maxwell/translate/impl/` @
//! cf654052): `vote.cpp` (the same three votes and the ballot), `warp_shuffle.cpp` (the same
//! clamp and segment fields, `QUAD_MASK` and quad cases, and `GetInBoundsFromOp`) and
//! `barrier_operations.cpp` (only `BAR.SYNC` with zero operands becomes `Barrier`).
//!
//! # What each instruction becomes
//!
//! - **No IR**: `NOP`, `BREAK`, `BSSY`, `BSYNC`, `BRX` (and `BRXU`), `CALL` with `.NOINC`,
//!   `RET` with `.NODEC`, `DEPBAR.LE SB5` with any count, `WARPSYNC 0xffffffff` and `KILL`
//!   with no predicate of its own. Branches, calls, returns and convergence are edges of the
//!   flow graph (slice 10), and a `KILL`'s discard is the end of its conditional block,
//!   [`crate::flow::BlockEnd::Kill`], which [`crate::flow::FlowGraph::build_ir`] turns into
//!   `DemoteToHelperInvocation`, as the PR's visitor translates `KILL` to nothing ("the
//!   discard itself is a block end class"). No branch needs more than its block's end.
//! - **`BAR.SYNC 0x0`**: `Barrier`.
//! - **`VOTE`**: the predicate's `SubgroupBallot` into the register and its `VoteAll`,
//!   `VoteAny` or `VoteEqual` into the predicate (`RZ` and `PT` write nothing); **`VOTEU`**
//!   the same into a uniform register and a uniform predicate.
//! - **`SHFL`**: `ShuffleIndex`, `ShuffleUp`, `ShuffleDown` or `ShuffleButterfly` of the
//!   source register, with the clamp the low five bits of the immediate and the segment mask
//!   its bits 8 to 12 ([`SHFL_FIELD_MASK`], [`SHFL_SEGMENT_SHIFT`]), and the shuffle's
//!   `GetInBoundsFromOp` into the predicate. In a pixel program, an immediate lane with the
//!   quad mask ([`QUAD_SHUFFLE_MASK`]) is `QuadBroadcast` (index lanes 0 to 3) or `QuadSwap`
//!   (butterfly lanes 1 to 3, direction lane − 1), with the predicate set true.
//! - **`REDUX`**: the active lanes' ballot, then for each of the [`WARP_SIZE`] lanes in order
//!   a shuffle of the value from that lane combined into an accumulator that starts at the
//!   operation's identity ([`redux_identity`]) and is kept only where the lane is active; the
//!   result goes into the uniform register.
//! - **`MATCH.ALL`**: whether every active lane holds the first active lane's value (both
//!   words for `.U64`), `VoteAll` of the comparison, into the predicate.
//!
//! Every instruction that writes a subgroup operation into the IR records "needs subgroup
//! size 32" ([`Emitted::needs_subgroup_size_32`], spec "Compute shaders"); collecting it into
//! a program's requirements is the collect-info pass's job (spec, pass 7; slices 15 and 18).
//!
//! # Guards
//!
//! In this order (`Visitor::Run`): an instruction under `@!PT` never runs and becomes nothing,
//! whatever its fields. An instruction under `PT`, or the one instruction of a block whose
//! guard flow analysis has tested ([`crate::flow::FlowBlock::guard_tested`]), is translated
//! as if unguarded. Otherwise the instruction is under a guard flow analysis left in its
//! block: `VOTEU`, the only pure instruction of the group (`classify.h`), keeps it as a select
//! of each result against the old value, read as a uniform predicate, as the PR's
//! `guard_select` does for a uniform instruction; an instruction that becomes no IR ignores
//! it; and the others (`BAR`, `VOTE`, `SHFL`, `REDUX`, `MATCH`) are refused,
//! [`ControlUnsupported::UntestedGuard`], since flow analysis gives each such instruction a
//! conditional block of its own and the translator does not build blocks. Then the PR's
//! checks run, each instruction's in the PR's order, every one before any IR is written, so a
//! refused instruction leaves its block as it was.
//!
//! # Refusals
//!
//! Each of the PR's `Require` checks on these forms is kept as a named
//! [`ControlUnsupported`] variant: `KILL`'s "native predicated KILL", `CALL`'s "native call
//! increment mode", `RET`'s "native return decrement mode", `WARPSYNC`'s "native partial warp
//! sync" and "native predicated warp sync", `DEPBAR`'s "native dependency barrier mode",
//! `BAR`'s "native workgroup barrier mode", `REDUX`'s "native warp reduction mode" (its
//! operation bound is the decoder's), `MATCH`'s "native warp match mode" and "native warp
//! match source". `VOTE`'s and `VOTEU`'s "native warp vote mode" checks are the decoder's
//! (operation 3 is [`crate::decode::DecodeError::UnknownValue`]). The forms the PR's visitor
//! has no case for are refused by name: `WARPSYNC` with a mask that is not an immediate (the
//! PR dispatches only `0x948`), `SHFL` with its clamp in a register (only `0x589` and `0xf89`),
//! `BRA` (never handed to the visitor: it ends its block), and `BMOV`, `NANOSLEEP`, `JMX` and
//! `BPT` ([`Unsupported::Instruction`]). An instruction of another group, `EXIT` included
//! (slice 1's translator handles it), is [`ControlUnsupported::OtherGroup`].
//!
//! # Divergences from PR #4588
//!
//! | What | Here | Reference | Pinned by |
//! |---|---|---|---|
//! | `SHFL` with its clamp in a register (`0x389`, `0x989`) | Refused, [`ControlUnsupported::ShuffleClampRegister`] | The native visitor refuses both; the PR's lowering path (`lower_to_maxwell.h`, case `0x989`, which the spec does not port) accepts `0x989` in its up mode, and its test "Native upward and immediate indexed shuffles preserve boundary predicates" runs it | `tests::register_clamp_shuffles_are_refused` |
//! | `DEPBAR` other than `.LE SB5` | Refused, [`ControlUnsupported::DepBarMode`] | The native visitor refuses it; the lowering path accepts scoreboard 0 too (its test "Native multi-scoreboard dependency waits retain surrounding dataflow", `0x800e` and `0x8000`) | `tests::depbar_needs_le_sb5` |
//! | `WARPSYNC 0xffffffff` | No IR | The native visitor makes none; the lowering path writes a device memory barrier (its test "Native full warp synchronization retains ordering after reconvergence") | `tests::instructions_with_no_ir` |
//! | A guarded `BAR`, `VOTE`, `SHFL`, `REDUX` or `MATCH` outside its conditional block | Refused, [`ControlUnsupported::UntestedGuard`] | Cannot happen: the PR's control-flow builder always makes the block | `tests::untested_guards_are_refused` |
//! | `GetInBoundsFromOp` of a `SHFL` whose predicate is `PT` | Not written | Written and left unused (`PredOut` drops it) | `tests::shuffle_golden_ir` |
//! | `MATCH`'s source register | Read once per word | Read twice per word (`X(source)` for the comparison and the shuffle): the same value | `tests::match_golden_ir` |
//!
//! Where the spec and naga part from the IR: `KILL` is demote in the IR, but naga has only
//! `Kill`, which ends the invocation, so a lowered discard stops the lane's helper role in
//! derivatives (spec, "The backend through naga": "Use `Kill`; list the divergence"). The
//! lowering is slice 16's; `tests::kill_demotes_through_flow_analysis` pins the IR side.
//!
//! # What the tests can check
//!
//! The interpreter runs one invocation as the warp's only active lane ([`crate::interp`],
//! "Warp operations"), so the tests run every instruction on one lane, the active lane being
//! lane 0 or 31, as PR #4588's tests do in their one-lane cases (active masks `1` and
//! `0x80000000`). Results that need several active lanes (a butterfly partner's value, a
//! reduction or match over several lanes, a shuffle from another lane) are pinned as golden IR
//! only.

#[cfg(test)]
mod tests;

use thiserror::Error;

use super::{ColorOutputs, Failure, GuestStage, Stage, TranslateError, Translator, Unsupported};
use crate::decode::{
    AluSrc, BarMode, BarRed, Gpr, Instr, MatchOp, Op, OpBar, OpMatch, OpRedux, OpShfl, OpVote,
    OpWarpSync, Pred, PredSrc, ReduxOp, ShflOp, ShflSrc, UniformPred, UniformReg, VoteDst, VoteOp,
};
use crate::flow::read_pred;
use crate::ir::{BlockId, BuildError, Builder, Function, Imm, Inst, Opcode, Operand, WARP_SIZE};

/// The lane mask `WARPSYNC` must name: every lane of the warp (`visitor.cpp`, case `0x948`:
/// `Require(Take(32, 32) == 0xffffffffU, "native partial warp sync")`).
pub const FULL_WARP_MASK: u32 = u32::MAX;
/// The scoreboard `DEPBAR` must wait on, with `.LE` set (`visitor.cpp`, case `0x91a`:
/// `Require(Take(44, 4) == 0xd, "native dependency barrier mode")`, "DEPBAR.LE SB5, 1"; the
/// decoder reads bits 44..47 as the scoreboard and bit 47 as `.LE`, a guess it lists).
pub const DEPBAR_SCOREBOARD: u8 = 5;
/// The `SHFL` immediate that names a quad (`ops_memory.cpp`, `Visitor::SHFL`: `mask == ((28U
/// << 8) | 3U)`; Eden's `warp_shuffle.cpp`: `QUAD_MASK = (28u << 8) | 3u`).
pub const QUAD_SHUFFLE_MASK: u32 = (28 << 8) | 3;
/// The highest lane of a quad (`Visitor::SHFL`: `lane <= 3`, `lane >= 1 && lane <= 3`).
pub const QUAD_LAST_LANE: u16 = 3;
/// The width of `SHFL`'s clamp and segment-mask fields within its immediate
/// (`Visitor::SHFL`: `mask & 0x1f`, `(mask >> 8) & 0x1f`).
pub const SHFL_FIELD_MASK: u32 = 0x1f;
/// Where `SHFL`'s segment mask starts within its immediate (`(mask >> 8) & 0x1f`).
pub const SHFL_SEGMENT_SHIFT: u32 = 8;
/// The clamp of a shuffle that reads any lane of the warp: its last lane
/// (`ops_half.cpp`, `Visitor::ShuffleLane`: `ShuffleIndex(value, lane, Imm32(0x1fU),
/// Imm32(0U))`).
pub const WHOLE_WARP_CLAMP: u32 = WARP_SIZE - 1;

/// A control or warp form, or a field value, the translator does not handle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum ControlUnsupported {
    /// An instruction that writes IR under a guard that flow analysis did not test (see the
    /// module docs, "Guards").
    #[error("a guarded instruction outside its conditional block")]
    UntestedGuard,
    /// A `BRA` handed to the translator: flow analysis ends a block with it.
    #[error("a bra inside a block")]
    Branch,
    /// A `KILL` with a predicate of its own (`visitor.cpp`: "native predicated KILL").
    #[error("a kill with a predicate of its own")]
    PredicatedKill,
    /// A `CALL` without `.NOINC` (`visitor.cpp`: "native call increment mode").
    #[error("a call without .noinc")]
    CallIncrement,
    /// A `RET` without `.NODEC` (`visitor.cpp`: "native return decrement mode").
    #[error("a ret without .nodec")]
    ReturnDecrement,
    /// A `WARPSYNC` whose mask is a register, a constant-buffer word or a uniform register
    /// (the PR's visitor has a case only for the immediate form, `0x948`).
    #[error("a warpsync whose mask is not an immediate")]
    WarpSyncMask,
    /// A `WARPSYNC` of fewer than every lane (`visitor.cpp`: "native partial warp sync").
    #[error("a warpsync of lane mask {0:#x}")]
    PartialWarpSync(u32),
    /// A `WARPSYNC` with a predicate of its own (`visitor.cpp`: "native predicated warp
    /// sync").
    #[error("a warpsync with a predicate of its own")]
    PredicatedWarpSync,
    /// A `DEPBAR` other than `.LE` on scoreboard 5 (`visitor.cpp`: "native dependency barrier
    /// mode").
    #[error("a depbar other than depbar.le sb5")]
    DepBarMode,
    /// A `BAR` other than `bar.sync 0x0` (`ops_memory.cpp`: "native workgroup barrier mode").
    #[error("a bar other than bar.sync 0x0")]
    BarrierMode,
    /// A `SHFL` with its clamp and segment mask in a register (forms `0x389` and `0x989`,
    /// which the PR's visitor has no case for).
    #[error("a shfl with its clamp in a register")]
    ShuffleClampRegister,
    /// A `REDUX` into `URZ` (`ops_memory.cpp`: "native warp reduction mode", `udst < 63`).
    #[error("a redux into urz")]
    ReduxToUrz,
    /// A `MATCH` with a mask register or `.ANY` (`ops_misc.cpp`: "native warp match mode").
    #[error("a match with a mask register or .any")]
    MatchMode,
    /// A `MATCH` of `RZ`, or a 64-bit one of an odd register or of `R254`
    /// (`ops_misc.cpp`: "native warp match source").
    #[error("a {}match of {src}", if *wide { "64-bit " } else { "" })]
    MatchSource {
        /// The source register.
        src: Gpr,
        /// Whether the match is 64-bit.
        wide: bool,
    },
    /// An instruction of another group: another translator takes it.
    #[error("{0} in the control and warp translator")]
    OtherGroup(&'static str),
}

/// What translating one instruction recorded besides its IR.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[must_use]
pub struct Emitted {
    /// The IR it wrote holds a subgroup operation ([`is_subgroup_op`]), so the program needs
    /// a subgroup size of 32 (spec, "Compute shaders": warp instructions assume 32 lanes).
    pub needs_subgroup_size_32: bool,
}

/// Whether `opcode` is a subgroup operation whose result depends on the warp's lanes: the
/// votes, the ballot, the lane masks, the shuffles and the quad operations (the IR's "Warp
/// operations" group but `LaneId`, `FSwizzleAdd` and the derivatives).
pub const fn is_subgroup_op(opcode: Opcode) -> bool {
    matches!(
        opcode,
        Opcode::VoteAll
            | Opcode::VoteAny
            | Opcode::VoteEqual
            | Opcode::SubgroupBallot
            | Opcode::SubgroupEqMask
            | Opcode::SubgroupLtMask
            | Opcode::SubgroupLeMask
            | Opcode::SubgroupGtMask
            | Opcode::SubgroupGeMask
            | Opcode::ShuffleIndex
            | Opcode::ShuffleUp
            | Opcode::ShuffleDown
            | Opcode::ShuffleButterfly
            | Opcode::QuadBroadcast
            | Opcode::QuadSwap
    )
}

/// The value a `REDUX` accumulator starts from (`ops_memory.cpp`, `Visitor::REDUX`:
/// `0xffffffffU` for `AND`, `0x7fffffffU` or `0xffffffffU` for a signed or unsigned `MIN`,
/// `0x80000000U` or `0U` for `MAX`, `0U` otherwise).
pub const fn redux_identity(op: ReduxOp) -> u32 {
    match op {
        ReduxOp::And | ReduxOp::Min { signed: false } => u32::MAX,
        ReduxOp::Min { signed: true } => i32::MAX.cast_unsigned(),
        ReduxOp::Max { signed: true } => i32::MIN.cast_unsigned(),
        ReduxOp::Or | ReduxOp::Xor | ReduxOp::Sum | ReduxOp::Max { signed: false } => 0,
    }
}

/// The operation that combines a `REDUX` accumulator with a lane's value (`Visitor::REDUX`).
const fn redux_combine(op: ReduxOp) -> Opcode {
    match op {
        ReduxOp::And => Opcode::BitwiseAnd32,
        ReduxOp::Or => Opcode::BitwiseOr32,
        ReduxOp::Xor => Opcode::BitwiseXor32,
        ReduxOp::Sum => Opcode::IAdd32,
        ReduxOp::Min { signed: true } => Opcode::SMin32,
        ReduxOp::Min { signed: false } => Opcode::UMin32,
        ReduxOp::Max { signed: true } => Opcode::SMax32,
        ReduxOp::Max { signed: false } => Opcode::UMax32,
    }
}

/// Translates instruction `index` of a program, `instr`, into IR at the end of `block` of
/// `function`, for a `stage` program; `guard_tested` is its flow block's
/// [`crate::flow::FlowBlock::guard_tested`]. See the module docs for what each instruction
/// becomes and the order of the checks. A refused instruction writes nothing into the block.
/// A [`TranslateError::Build`] (an arena is full, or a malformed instruction, which would be
/// a bug here) can leave part of the instruction's IR in the block.
pub fn translate_control(
    function: &mut Function,
    block: BlockId,
    stage: GuestStage,
    index: usize,
    instr: Instr,
    guard_tested: bool,
) -> Result<Emitted, TranslateError> {
    let start = function
        .block(block)
        .map(|listed| listed.insts.len())
        .ok_or(TranslateError::Build(BuildError::NoBlock(block)))?;
    let mut t = Translator {
        b: Builder::new(function, block),
        stage: Stage {
            stage,
            color_outputs: ColorOutputs::default(),
        },
    };
    match t.control(instr, guard_tested) {
        Ok(()) => {}
        Err(Failure::Unsupported(what)) => {
            return Err(TranslateError::Unsupported { index, instr, what });
        }
        Err(Failure::Build(error)) => return Err(error.into()),
    }
    let written = function
        .block(block)
        .and_then(|listed| listed.insts.get(start..))
        .unwrap_or_default();
    let needs_subgroup_size_32 = written.iter().any(
        |&value| matches!(function.inst(value), Some(Inst::Op(op)) if is_subgroup_op(op.opcode)),
    );
    Ok(Emitted {
        needs_subgroup_size_32,
    })
}

/// A refusal of this group.
fn refuse(what: ControlUnsupported) -> Failure {
    Failure::Unsupported(Unsupported::Control(what))
}

/// `Ok` if `ok`, else a refusal of this group.
fn need(ok: bool, what: ControlUnsupported) -> Result<(), Failure> {
    if ok { Ok(()) } else { Err(refuse(what)) }
}

/// A 32-bit immediate.
const fn imm32(bits: u32) -> Operand {
    Operand::Imm(Imm::U32(bits))
}

/// A predicate immediate.
const fn imm1(bit: bool) -> Operand {
    Operand::Imm(Imm::U1(bit))
}

impl Translator<'_> {
    /// One instruction of the group (see the module docs).
    fn control(&mut self, instr: Instr, guard_tested: bool) -> Result<(), Failure> {
        let guard = instr.guard;
        if guard.reg.is_true() && guard.negated {
            return Ok(());
        }
        let untested = !guard_tested && !guard.is_true();
        let tested = |ok: bool| need(!ok, ControlUnsupported::UntestedGuard);
        match instr.op {
            Op::Kill(op) => need(op.pred.is_true(), ControlUnsupported::PredicatedKill),
            Op::Call(op) => need(op.no_inc, ControlUnsupported::CallIncrement),
            Op::Ret(op) => need(op.no_dec, ControlUnsupported::ReturnDecrement),
            Op::Brx(_) | Op::BSSy(_) | Op::BSync(_) | Op::Break(_) | Op::Nop(_) => Ok(()),
            Op::WarpSync(op) => warpsync(op),
            Op::DepBar(op) => need(
                op.scoreboard == DEPBAR_SCOREBOARD && op.le,
                ControlUnsupported::DepBarMode,
            ),
            Op::Bra(_) => Err(refuse(ControlUnsupported::Branch)),
            Op::Bar(op) => {
                tested(untested)?;
                self.bar(op)
            }
            Op::Vote(op) => match op.dst {
                VoteDst::Reg { ballot, vote } => {
                    tested(untested)?;
                    self.vote(op.op, op.pred, ballot, vote)
                }
                VoteDst::Uniform { ballot, vote } => {
                    let select = if untested { Some(guard) } else { None };
                    self.voteu(op, ballot, vote, select)
                }
            },
            Op::Shfl(op) => {
                tested(untested)?;
                self.shfl(op)
            }
            Op::Redux(op) => {
                tested(untested)?;
                self.redux(op)
            }
            Op::Match(op) => {
                tested(untested)?;
                self.match_all(op)
            }
            Op::BMov(_) | Op::Nanosleep(_) | Op::Jmx(_) | Op::Bpt(_) => Err(Failure::Unsupported(
                Unsupported::Instruction(instr.op.mnemonic()),
            )),
            other => Err(refuse(ControlUnsupported::OtherGroup(other.mnemonic()))),
        }
    }

    fn set_pred(&mut self, pred: Pred, value: Operand) -> Result<(), Failure> {
        if !pred.is_true() {
            self.op(Opcode::SetPred, &[Operand::Imm(Imm::Pred(pred)), value])?;
        }
        Ok(())
    }

    fn pred(&mut self, src: PredSrc) -> Result<Operand, Failure> {
        Ok(read_pred(&mut self.b, src)?)
    }

    /// `BAR` (`Visitor::BAR`): bits 32..63, 64..79 and 81..95 zero, so `bar.sync 0x0` with
    /// any `.DEFER_BLOCKING` (bit 80, which the PR ignores).
    fn bar(&mut self, op: OpBar) -> Result<(), Failure> {
        need(
            op.mode == BarMode::Sync && op.red == BarRed::Popc && op.id == 0 && op.unknown == 0,
            ControlUnsupported::BarrierMode,
        )?;
        self.op(Opcode::Barrier, &[])?;
        Ok(())
    }

    fn vote_op(&mut self, op: VoteOp, source: Operand) -> Result<Operand, Failure> {
        let opcode = match op {
            VoteOp::All => Opcode::VoteAll,
            VoteOp::Any => Opcode::VoteAny,
            VoteOp::Eq => Opcode::VoteEqual,
        };
        Ok(self.op(opcode, &[source])?.into())
    }

    /// `VOTE` (`Visitor::VOTE`): the ballot, then the vote.
    fn vote(&mut self, op: VoteOp, pred: PredSrc, ballot: Gpr, vote: Pred) -> Result<(), Failure> {
        let source = self.pred(pred)?;
        if !ballot.is_zero() {
            let bits = self.op(Opcode::SubgroupBallot, &[source])?;
            self.set(ballot, bits.into())?;
        }
        if !vote.is_true() {
            let result = self.vote_op(op, source)?;
            self.set_pred(vote, result)?;
        }
        Ok(())
    }

    /// `VOTEU` (`Visitor::VOTEU`), with the guard `select` kept as a select of each result
    /// (`Visitor::UR`, `Visitor::SetUPred`).
    fn voteu(
        &mut self,
        op: OpVote,
        ballot: UniformReg,
        vote: UniformPred,
        select: Option<PredSrc>,
    ) -> Result<(), Failure> {
        let guard = match select {
            Some(guard) => Some(self.uniform_guard(guard)?),
            None => None,
        };
        let source = self.pred(op.pred)?;
        if !ballot.is_zero() {
            let name = Operand::Imm(Imm::UniformReg(ballot));
            let mut bits: Operand = self.op(Opcode::SubgroupBallot, &[source])?.into();
            if let Some(guard) = guard {
                let old = self.op(Opcode::GetUniformRegister, &[name])?;
                bits = self
                    .op(Opcode::SelectU32, &[guard, bits, old.into()])?
                    .into();
            }
            self.op(Opcode::SetUniformRegister, &[name, bits])?;
        }
        if !vote.is_true() {
            let name = Operand::Imm(Imm::UniformPred(vote));
            let mut result = self.vote_op(op.op, source)?;
            if let Some(guard) = guard {
                let old = self.op(Opcode::GetUniformPred, &[name])?;
                result = self
                    .op(Opcode::SelectU1, &[guard, result, old.into()])?
                    .into();
            }
            self.op(Opcode::SetUniformPred, &[name, result])?;
        }
        Ok(())
    }

    /// A uniform instruction's guard, read as the uniform predicate of the same number
    /// (`Visitor::Run`: `UPred(guard.index, guard.negated)`).
    fn uniform_guard(&mut self, guard: PredSrc) -> Result<Operand, Failure> {
        let Some(reg) = UniformPred::new(guard.reg.index()).filter(|reg| !reg.is_true()) else {
            return Ok(imm1(!guard.negated));
        };
        let value = self.op(
            Opcode::GetUniformPred,
            &[Operand::Imm(Imm::UniformPred(reg))],
        )?;
        if guard.negated {
            Ok(self.op(Opcode::LogicalNot, &[value.into()])?.into())
        } else {
            Ok(value.into())
        }
    }

    /// `SHFL` (`Visitor::SHFL`; see the module docs).
    fn shfl(&mut self, op: OpShfl) -> Result<(), Failure> {
        let ShflSrc::Imm(mask) = op.c else {
            return Err(refuse(ControlUnsupported::ShuffleClampRegister));
        };
        let mask = u32::from(mask);
        let value = self.get(op.src)?;
        if let ShflSrc::Imm(lane) = op.lane
            && self.stage.stage == GuestStage::Pixel
            && mask == QUAD_SHUFFLE_MASK
        {
            let quad = match op.op {
                ShflOp::Idx if lane <= QUAD_LAST_LANE => {
                    Some((Opcode::QuadBroadcast, u32::from(lane)))
                }
                ShflOp::Bfly if (1..=QUAD_LAST_LANE).contains(&lane) => {
                    Some((Opcode::QuadSwap, u32::from(lane) - 1))
                }
                _ => None,
            };
            if let Some((opcode, arg)) = quad {
                let result = self.op(opcode, &[value, imm32(arg)])?;
                self.set(op.dst, result.into())?;
                return self.set_pred(op.in_bounds, imm1(true));
            }
        }
        let lane = match op.lane {
            ShflSrc::Reg(reg) => self.get(reg)?,
            ShflSrc::Imm(lane) => imm32(u32::from(lane)),
        };
        let clamp = imm32(mask & SHFL_FIELD_MASK);
        let segment = imm32((mask >> SHFL_SEGMENT_SHIFT) & SHFL_FIELD_MASK);
        let opcode = match op.op {
            ShflOp::Idx => Opcode::ShuffleIndex,
            ShflOp::Up => Opcode::ShuffleUp,
            ShflOp::Down => Opcode::ShuffleDown,
            ShflOp::Bfly => Opcode::ShuffleButterfly,
        };
        let result = self.op(opcode, &[value, lane, clamp, segment])?;
        if !op.in_bounds.is_true() {
            let in_bounds = self.op(Opcode::GetInBoundsFromOp, &[result.into()])?;
            self.set_pred(op.in_bounds, in_bounds.into())?;
        }
        self.set(op.dst, result.into())
    }

    /// A shuffle of `value` from `lane` over the whole warp (`Visitor::ShuffleLane`).
    fn shuffle_lane(&mut self, value: Operand, lane: Operand) -> Result<Operand, Failure> {
        Ok(self
            .op(
                Opcode::ShuffleIndex,
                &[value, lane, imm32(WHOLE_WARP_CLAMP), imm32(0)],
            )?
            .into())
    }

    /// The active lanes (`SubgroupBallot(Imm1(true))`).
    fn active_lanes(&mut self) -> Result<Operand, Failure> {
        Ok(self.op(Opcode::SubgroupBallot, &[imm1(true)])?.into())
    }

    /// `REDUX` (`Visitor::REDUX`; see the module docs).
    fn redux(&mut self, op: OpRedux) -> Result<(), Failure> {
        need(!op.dst.is_zero(), ControlUnsupported::ReduxToUrz)?;
        let value = self.get(op.src)?;
        let active = self.active_lanes()?;
        let combine = redux_combine(op.op);
        let mut accumulator = imm32(redux_identity(op.op));
        for lane in 0..WARP_SIZE {
            let shifted = self.op(Opcode::ShiftRightLogical32, &[active, imm32(lane)])?;
            let bit = self.op(Opcode::BitwiseAnd32, &[shifted.into(), imm32(1)])?;
            let present = self.op(Opcode::INotEqual, &[bit.into(), imm32(0)])?;
            let other = self.shuffle_lane(value, imm32(lane))?;
            let combined = self.op(combine, &[accumulator, other])?;
            accumulator = self
                .op(
                    Opcode::SelectU32,
                    &[present.into(), combined.into(), accumulator],
                )?
                .into();
        }
        self.op(
            Opcode::SetUniformRegister,
            &[Operand::Imm(Imm::UniformReg(op.dst)), accumulator],
        )?;
        Ok(())
    }

    /// `MATCH.ALL` (`Visitor::MATCH`; see the module docs).
    fn match_all(&mut self, op: OpMatch) -> Result<(), Failure> {
        need(
            op.mask.is_zero() && op.op == MatchOp::All,
            ControlUnsupported::MatchMode,
        )?;
        let src = op.src;
        let wide_ok = src.index().is_multiple_of(2) && super::register_run(src, 2).is_some();
        need(
            !src.is_zero() && (!op.u64 || wide_ok),
            ControlUnsupported::MatchSource { src, wide: op.u64 },
        )?;
        let active = self.active_lanes()?;
        let negated = self.op(Opcode::INeg32, &[active])?;
        let lowest = self.op(Opcode::BitwiseAnd32, &[active, negated.into()])?;
        let first = self.op(Opcode::FindUMsb32, &[lowest.into()])?;
        let mut equal = self.equal_to_lane(src, first.into())?;
        if op.u64 {
            let high = Gpr::new(src.index().saturating_add(1));
            let high_equal = self.equal_to_lane(high, first.into())?;
            equal = self.op(Opcode::LogicalAnd, &[equal, high_equal])?.into();
        }
        let all = self.op(Opcode::VoteAll, &[equal])?;
        self.set_pred(op.pred, all.into())
    }

    /// Whether `reg` equals its value in `lane`.
    fn equal_to_lane(&mut self, reg: Gpr, lane: Operand) -> Result<Operand, Failure> {
        let value = self.get(reg)?;
        let there = self.shuffle_lane(value, lane)?;
        Ok(self.op(Opcode::IEqual, &[value, there])?.into())
    }
}

/// `WARPSYNC` (`visitor.cpp`, case `0x948`): an immediate mask of every lane and no predicate
/// of its own; no IR.
fn warpsync(op: OpWarpSync) -> Result<(), Failure> {
    let AluSrc::Imm(mask) = op.mask else {
        return Err(refuse(ControlUnsupported::WarpSyncMask));
    };
    need(
        mask == FULL_WARP_MASK,
        ControlUnsupported::PartialWarpSync(mask),
    )?;
    need(op.cond.is_true(), ControlUnsupported::PredicatedWarpSync)
}
