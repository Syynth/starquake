//! Slice 12a: the integer ALU group's translator, decoded instructions of slice 6a to IR.
//!
//! [`translate_integer`] appends one instruction's IR at a [`Builder`]'s position, and
//! [`translate_integer_word`] decodes a word first, so an unknown opcode or bit is refused with
//! the decoder's exact error ([`IntegerError::Decode`]; flow analysis does not decode plain
//! instructions, `crate::flow`'s divergence table). [`is_integer`] says which operations are
//! this group's: `MOV`, `IADD3`, `IADD3.X`, `LEA`, `LEA.X`, `LOP3`, `IABS`, `BREV`, `POPC`,
//! `FLO`, `IMNMX`, `SEL`, `SGXT`, `BMSK`, `PRMT`, `SHF`, `IMAD` (with `IMAD.HI`), `IMAD.WIDE`,
//! `IDP`, `ISETP`, `PLOP3`, `P2R` and `R2P`; any other is [`IntegerError::OtherGroup`].
//!
//! # Sources
//!
//! Ported by behaviour from PR #4588's native frontend
//! (`eden-pr4588/src/shader_recompiler/frontend/ampere/` @ e5fb3b97, GPL-3.0-or-later, Copyright
//! 2026 Eden Emulator Project): `ops_int.cpp` (`LOP3`, `SHF`'s operand reads, `LEA`, `IMNMX`,
//! `PRMT`, `IMAD`, `MultiplyHigh`, `IMAD_HI`, `IMAD_WIDE`, `IABS`, `SGXT`, `FLO`, `POPC`, `IDP`),
//! `ops_alu.cpp` (`SEL`, `ISETP`, `IADD3`, `PLOP3`, `P2R`, `R2P`), `ops_move.cpp` (`MOV`),
//! `ops_memory.cpp` (`BREV`) and `visitor.cpp` (`Visitor::Cbuf`, `WideSlot`, `X`, `Pred`,
//! `Run`). Cross-checked with Eden's Maxwell translators
//! (`eden/src/shader_recompiler/frontend/maxwell/translate/impl/` @ cf654052,
//! GPL-2.0-or-later): `common_funcs.cpp` gives `IntegerCompare` and `PredicateCombine`, which
//! `ISETP` uses as the PR does, and `logic_operation_three_input.cpp`'s `ApplyLUT` the meaning of
//! a truth table (bit `a << 2 | b << 1 | c`, with `a` = `0xf0`, `b` = `0xcc`, `c` = `0xaa`).
//! `SHF`'s shift amount follows NAK's fold
//! (`mesa-nouveau/src/nouveau/compiler/nak/ir.rs` @ 52238762, MIT, `impl Foldable for OpShf`),
//! since the decoder reads `SHF`'s type field as NAK encodes it (see "Divergences").
//!
//! # Guards
//!
//! An instruction under a guard predicate other than `PT` becomes a conditional block (spec,
//! "Instruction translation"): the block the builder is in reads the guard and branches to a
//! new block holding the instruction's IR, which jumps to a second new block that takes over
//! the first block's terminator; the builder is left in that second block. The split is made
//! when the first IR operation is appended, after every check, so a refused instruction leaves
//! the function untouched (`tests::guards::refusals_leave_the_function_untouched`). An
//! instruction under `!PT` never runs and appends nothing, without its checks, as PR #4588's
//! `Visitor::Run` returns at once when `guard.Never()`; with [`GuardMode::Tested`] the guard is
//! ignored (flow analysis has tested it, [`crate::flow::FlowBlock::guard_tested`]).
//!
//! # Checks
//!
//! Each `Require` the PR makes on these forms is kept as a named [`IntegerUnsupported`] value
//! (its doc names the PR's message), checked in the PR's order, except where the decoder
//! refuses the word first: the absolute-value and negation bits of sources that have none,
//! `LOP3`'s `.PAND` and predicate input, the carry outputs of `IMAD`, `IMAD.HI`, `IMAD.WIDE` and
//! `FLO`, `ISETP`'s combining operation 3, `PRMT`'s bit 75 and the bits `P2R` and `R2P` require
//! clear (the decoder's module docs, "Divergences from PR #4588's table"). A constant-buffer
//! source is checked first, in every instruction, as the PR reads every operand before its
//! other checks (`Visitor::WideSlot`: "native bindless constant buffer").
//!
//! The PR's `Visitor::Cbuf` refuses a bank of 32 or more; a decoded [`CBuf::Binding`] is five
//! bits wide, so that check cannot fail here and is not repeated.
//!
//! # Carry chains
//!
//! A 64-bit addition is a chain of `IADD3` (or `LEA`) with a carry-out predicate and `IADD3.X`
//! (or `LEA.X`) reading it (scoping doc, "Missing pieces in the lowering"; rows 63, 83). Each
//! link is translated as the PR does, to a 64-bit IR sum (`PackUint2x32`, `IAdd64`,
//! `UnpackUint2x32`) whose high word gives the carry, so the chain's result is exact at the
//! 32-bit carry boundary (`tests::alu::a_two_instruction_carry_chain_adds_64_bits`).
//!
//! # Register pairs
//!
//! `IMAD.WIDE` writes a register pair and reads one as its addend. The PR's checks are kept
//! (`Visitor::IMAD_WIDE`: `dest % 2 == 0 && dest < limit` with `limit` 254 for the vector
//! datapath, and the same for a register or uniform-register addend other than `RZ`), so a pair
//! starts at an even register and ends at or below `R253`; a uniform-register addend must be an
//! even register too, which refuses `URZ` (63 is odd), as the PR does. An immediate addend is
//! refused (the PR's `default` case, "native wide multiply addend").
//!
//! # Divergences from PR #4588
//!
//! | What | Here | PR #4588 | Pinned by |
//! |---|---|---|---|
//! | A pure instruction under a guard | A conditional block (spec, "Instruction translation") | A select of each written value (`visitor.cpp`, `guard_select`) | `tests::guards::a_guard_makes_a_conditional_block` |
//! | `SHF`'s type field | NAK's encoding (0 `I64`, 1 `U64`, 2 `I32`, 3 `U32`) and NAK's shift amount: masked to the type's width less one with `.W`, else clamped to the width (32 or 64) | 1 `U32`, 2 `S64`, 3 `U64`, 0 refused; the amount masked to 31 or clamped to 32 for every type | `tests::alu::shf_follows_naks_types_and_amounts` |
//! | `IADD3.X` and `LEA.X` | Bit 74 in every form (the decoder, after NAK) | `IADD3.X` is source 2's absolute-value bit (bit 62 in forms 3 and 7, none in form 2); `LEA.X` with `b` in the register slot (forms 2, 3 and 7) refused (`Require(!b.abs)`) | `tests::alu::extended_adds_in_every_form` |
//! | `LEA`'s high word in forms 2, 3 and 7 | Source 2, the wide slot (NAK's operand order) | Bits 64..72, the same register as `b` | `tests::alu::lea_hi_reads_its_high_word_from_source_2` |
//! | `BREV` | Every source form (1, 4, 5, 6) | Form 1 only (`case 0x301`) | `tests::alu::brev_takes_every_source_form` |
//! | `PRMT`, forms 2, 3 and 7 | Refused, the selector is not an immediate | Refused the same way: the selector is the register at 64..72 | `tests::rejections::each_check_names_its_reason` |
//!
//! `IDP` is translated in its `.2A` mode only, the one the PR translates (`Visitor::IDP`,
//! "IDP.2A.LO.U16.U8", the PR's reading of the one encoding it met: a guess the PR marks as
//! such); `.4A`, which neither reference translates, is refused by name
//! ([`IntegerUnsupported::IdpFourWay`]). `BMSK` has no translator in either reference (none in
//! the PR's `ops_*.cpp`, Eden's Maxwell has no such instruction, and NAK's `OpBMsk` has no
//! fold) and is refused by name ([`IntegerUnsupported::Instruction`]).
//!
//! Where the PR's reading and NAK's differ on the meaning of a decoded field, this module
//! follows the decoder: `LEA.HI.X` with a negated `b` subtracts it as the PR's `.HI` path does
//! (`ir.INeg(addend)`), and `IMAD.X`'s negated addend is a bitwise not as in the PR.

#[cfg(test)]
mod tests;

use thiserror::Error;

use crate::decode::{
    AluSrc, CBuf, CBufRef, DecodeError, Gpr, IMadMode, Instr, Instruction, IntCmpOp, Op, OpBRev,
    OpFlo, OpIAbs, OpIAdd3, OpIAdd3X, OpIDp4, OpIMad, OpIMad64, OpIMnMx, OpISetP, OpLea, OpLeaX,
    OpLop3, OpMov, OpP2R, OpPLop3, OpPopC, OpPrmt, OpR2P, OpSel, OpSgxt, OpShf, PLop3Src, Pred,
    PredSetOp, PredSrc, PrmtMode, ShfType, UniformPredSrc, UniformReg, decode,
};
use crate::ir::{BlockId, BuildError, Builder, Imm, Opcode, Operand, Terminator};

/// How [`translate_integer`] treats an instruction's guard.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GuardMode {
    /// Honour it: a guard other than `PT` makes the instruction a conditional block, and `!PT`
    /// appends nothing (see the module docs, "Guards").
    Honour,
    /// Ignore it: flow analysis has tested it already ([`crate::flow::FlowBlock::guard_tested`]).
    Tested,
}

/// A form or field value of this group that is not translated: one of PR #4588's `Require`
/// checks, named by its message, or an instruction neither reference translates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum IntegerUnsupported {
    /// An instruction neither reference translates (`BMSK`).
    #[error("{0}, which neither reference translates,")]
    Instruction(&'static str),
    /// A constant-buffer source whose buffer is named by a uniform register
    /// (`Visitor::WideSlot`: "native bindless constant buffer").
    #[error("a bindless constant-buffer source")]
    BindlessConstantBuffer,
    /// A `MOV` lane mask other than `0xf` (`ops_move.cpp`: "native MOV lane mask").
    #[error("a mov with a partial lane mask")]
    MovLaneMask,
    /// `IADD3` with a carry input other than `!PT` (`Visitor::IADD3`: "native IADD3 carry
    /// predicate").
    #[error("an iadd3 carry input other than !pt")]
    Iadd3CarryPredicate,
    /// `LEA` with its shifted value negated (`Visitor::LEA`: "native LEA intermediate
    /// modifier").
    #[error("a lea with its shifted value negated")]
    LeaIntermediate,
    /// `LEA` with a carry input other than `!PT` (`Visitor::LEA`: "native LEA carry
    /// predicate").
    #[error("a lea carry input other than !pt")]
    LeaCarryPredicate,
    /// `LEA.HI` with a carry output (`Visitor::LEA`: "native LEA high word carry output").
    #[error("a lea.hi with a carry output")]
    LeaHighCarryOutput,
    /// `LEA.SX32` without `.HI` (`Visitor::LEA`: "native LEA sign extension without high
    /// word").
    #[error("a lea.sx32 without .hi")]
    LeaSignExtendWithoutHigh,
    /// `LEA.HI.SX32` with a high word other than `RZ` (`Visitor::LEA`: "native LEA sign
    /// extension with explicit high word").
    #[error("a lea.hi.sx32 with a high word other than rz")]
    LeaSignExtendWithHighWord,
    /// A `PRMT` mode other than the default (`Visitor::PRMT`: "native permute mode").
    #[error("a prmt mode other than the default")]
    PermuteMode,
    /// A `PRMT` selector that is not an immediate (`Visitor::PRMT`: "native register permute
    /// selector").
    #[error("a prmt selector that is not an immediate")]
    PermuteSelector,
    /// `IMAD` without `.X` with a carry input other than `!PT` (`Visitor::IMAD`: "native IMAD
    /// carry input").
    #[error("an imad carry input other than !pt")]
    ImadCarryInput,
    /// `IMAD.X` with a negated factor (`Visitor::IMAD`: "native IMAD carry-in with negated
    /// factor").
    #[error("an imad.x with a negated factor")]
    ImadCarryNegatedFactor,
    /// `IMAD.WIDE` with a carry input other than `!PT` (`Visitor::IMAD_WIDE`: "native wide
    /// multiply carry input").
    #[error("an imad.wide carry input other than !pt")]
    WideCarryInput,
    /// `IMAD.WIDE` whose destination pair does not start at an even register below
    /// [`WIDE_PAIR_LIMIT`] (`Visitor::IMAD_WIDE`: "native wide multiply destination").
    #[error("an imad.wide destination pair at {0}")]
    WideDestination(Gpr),
    /// `IMAD.WIDE` whose register addend pair, other than `RZ`, does not start at an even
    /// register below [`WIDE_PAIR_LIMIT`] (`Visitor::IMAD_WIDE`: "native wide multiply addend").
    #[error("an imad.wide addend pair at {0}")]
    WideAddendRegister(Gpr),
    /// `IMAD.WIDE` whose uniform-register addend pair does not start at an even register
    /// (`Visitor::IMAD_WIDE`: "native wide multiply addend"; `URZ` is odd).
    #[error("an imad.wide addend pair at {0}")]
    WideAddendUniform(UniformReg),
    /// `IMAD.WIDE` with an immediate addend (`Visitor::IMAD_WIDE`, the `default` case: "native
    /// wide multiply addend").
    #[error("an imad.wide immediate addend")]
    WideAddendImmediate,
    /// `SGXT.W`: the PR does not consume bit 75, so its `CheckConsumed` refuses the word.
    #[error("an sgxt.w")]
    SgxtWrap,
    /// `IDP.4A` (`Visitor::IDP`: "native packed dot-product mode", bit 76 clear).
    #[error("idp.4a, which neither reference translates,")]
    IdpFourWay,
    /// `IDP.2A` with signed bytes or a negated source (`Visitor::IDP`: "native packed
    /// dot-product mode", bits 72..76 set; source 1's negation bit, 63, is not consumed).
    #[error("an idp.2a with signed or negated sources")]
    IdpModifiers,
    /// `ISETP` with a low-word predicate other than `PT` (`Visitor::ISETP`: "native extended
    /// integer comparison predicate").
    #[error("an isetp low-word predicate other than pt")]
    IsetpLowPredicate,
    /// `ISETP.EX` (`Visitor::ISETP`: "native extended integer comparison").
    #[error("an isetp.ex")]
    IsetpExtended,
}

/// Why an integer ALU instruction was not translated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum IntegerError {
    /// The word does not decode.
    #[error(transparent)]
    Decode(#[from] DecodeError),
    /// The instruction belongs to another group.
    #[error("{0} is not an integer ALU instruction")]
    OtherGroup(&'static str),
    /// The form or a field value is not translated.
    #[error("{0} is not supported")]
    Unsupported(IntegerUnsupported),
    /// The IR could not be built: an arena is full, or this module built a malformed
    /// instruction (a bug). The function may hold part of the instruction's IR.
    #[error(transparent)]
    Build(#[from] BuildError),
}

impl From<IntegerUnsupported> for IntegerError {
    fn from(what: IntegerUnsupported) -> Self {
        Self::Unsupported(what)
    }
}

/// The register a pair of `IMAD.WIDE` must start below (`Visitor::IMAD_WIDE`: `limit =
/// uniform_op ? 62 : 254`, the vector datapath's).
pub const WIDE_PAIR_LIMIT: u8 = 254;

/// The predicates `P2R` and `R2P` move: `P0` to `P6`, one mask bit each (`Visitor::P2R`,
/// `Visitor::R2P`: `bit_index < 7`).
const MOVED_PREDICATES: u8 = Pred::PT.index();

/// The rows of a three-input truth table.
const LUT_ROWS: u8 = 8;

/// Whether `op` belongs to this group.
pub const fn is_integer(op: &Op) -> bool {
    matches!(
        op,
        Op::Mov(_)
            | Op::IAdd3(_)
            | Op::IAdd3X(_)
            | Op::Lea(_)
            | Op::LeaX(_)
            | Op::Lop3(_)
            | Op::IAbs(_)
            | Op::BRev(_)
            | Op::PopC(_)
            | Op::Flo(_)
            | Op::IMnMx(_)
            | Op::Sel(_)
            | Op::Sgxt(_)
            | Op::BMsk(_)
            | Op::Prmt(_)
            | Op::Shf(_)
            | Op::IMad(_)
            | Op::IMad64(_)
            | Op::IDp4(_)
            | Op::ISetP(_)
            | Op::PLop3(_)
            | Op::P2R(_)
            | Op::R2P(_)
    )
}

/// Decodes `word` and translates it ([`translate_integer`]).
pub fn translate_integer_word(
    builder: &mut Builder<'_>,
    word: Instruction,
    guard: GuardMode,
) -> Result<(), IntegerError> {
    let instr = decode(word)?;
    translate_integer(builder, &instr, guard)
}

/// Appends the IR of `instr`, an integer ALU instruction, at `builder`'s position (see the
/// module docs). The checks run in this order: the group, then the guard (`!PT` appends
/// nothing), then constant-buffer sources, then the instruction's own checks in PR #4588's
/// order.
pub fn translate_integer(
    builder: &mut Builder<'_>,
    instr: &Instr,
    guard: GuardMode,
) -> Result<(), IntegerError> {
    if !is_integer(&instr.op) {
        return Err(IntegerError::OtherGroup(instr.op.mnemonic()));
    }
    let pending = match guard {
        GuardMode::Tested => None,
        GuardMode::Honour if instr.guard.is_true() => None,
        GuardMode::Honour if instr.guard.reg.is_true() => return Ok(()),
        GuardMode::Honour => Some(instr.guard),
    };
    if sources(&instr.op).iter().flatten().any(|src| {
        matches!(
            src,
            AluSrc::CBuf {
                cbuf: CBufRef {
                    buf: CBuf::BindlessUniform(_),
                    ..
                },
                ..
            }
        )
    }) {
        return Err(IntegerUnsupported::BindlessConstantBuffer.into());
    }
    let mut e = Emitter {
        b: builder,
        pending,
        join: None,
    };
    e.instr(instr.op)?;
    e.finish()
}

/// The ALU sources of `op`, for the constant-buffer check every instruction makes first.
const fn sources(op: &Op) -> [Option<AluSrc>; 3] {
    match op {
        Op::Mov(op) => [Some(op.src.to_alu()), None, None],
        Op::IAdd3(OpIAdd3 {
            srcs: [a, b, c], ..
        })
        | Op::IAdd3X(OpIAdd3X {
            add: OpIAdd3 {
                srcs: [a, b, c], ..
            },
        })
        | Op::Lop3(OpLop3 {
            srcs: [a, b, c], ..
        })
        | Op::IMad(OpIMad {
            srcs: [a, b, c], ..
        })
        | Op::IMad64(OpIMad64 {
            srcs: [a, b, c], ..
        })
        | Op::IDp4(OpIDp4 {
            srcs: [a, b, c], ..
        }) => [Some(*a), Some(*b), Some(*c)],
        Op::Lea(lea) | Op::LeaX(OpLeaX { lea }) => [Some(lea.a), Some(lea.b), Some(lea.a_high)],
        Op::IAbs(OpIAbs { src, .. })
        | Op::BRev(OpBRev { src, .. })
        | Op::PopC(OpPopC { src, .. })
        | Op::Flo(OpFlo { src, .. }) => [Some(*src), None, None],
        Op::IMnMx(OpIMnMx { srcs: [a, b], .. })
        | Op::Sel(OpSel { srcs: [a, b], .. })
        | Op::ISetP(OpISetP { srcs: [a, b], .. }) => [Some(*a), Some(*b), None],
        Op::Sgxt(op) => [Some(op.a), Some(op.bits), None],
        Op::BMsk(op) => [Some(op.pos), Some(op.width), None],
        Op::Prmt(op) => [Some(op.srcs[0]), Some(op.sel), Some(op.srcs[1])],
        Op::Shf(op) => [Some(op.low), Some(op.shift), Some(op.high)],
        _ => [None, None, None],
    }
}

/// `!PT` as a predicate source: the carry input that adds nothing.
const NOT_PT: PredSrc = PredSrc {
    reg: Pred::PT,
    negated: true,
};

/// A 32-bit immediate operand.
const fn imm32(bits: u32) -> Operand {
    Operand::Imm(Imm::U32(bits))
}

/// A 64-bit immediate operand.
const fn imm64(bits: u64) -> Operand {
    Operand::Imm(Imm::U64(bits))
}

/// A boolean immediate operand.
const fn imm1(value: bool) -> Operand {
    Operand::Imm(Imm::U1(value))
}

const fn require(ok: bool, what: IntegerUnsupported) -> Result<(), IntegerUnsupported> {
    if ok { Ok(()) } else { Err(what) }
}

/// The opcodes a truth table is built with, for 32-bit values or for predicates.
struct LogicOps {
    and: Opcode,
    or: Opcode,
    not: Opcode,
    none: Operand,
    all: Operand,
}

const BITWISE: LogicOps = LogicOps {
    and: Opcode::BitwiseAnd32,
    or: Opcode::BitwiseOr32,
    not: Opcode::BitwiseNot32,
    none: imm32(0),
    all: imm32(u32::MAX),
};

const LOGICAL: LogicOps = LogicOps {
    and: Opcode::LogicalAnd,
    or: Opcode::LogicalOr,
    not: Opcode::LogicalNot,
    none: imm1(false),
    all: imm1(true),
};

/// Appends one instruction's IR, splitting the block for its guard at the first operation.
struct Emitter<'b, 'f> {
    b: &'b mut Builder<'f>,
    /// The guard to test before the first operation, until it is tested.
    pending: Option<PredSrc>,
    /// The block after the conditional block, once the guard is tested.
    join: Option<BlockId>,
}

impl Emitter<'_, '_> {
    /// Appends an operation, after the guard's split if it is still pending.
    fn op(&mut self, opcode: Opcode, args: &[Operand]) -> Result<Operand, BuildError> {
        if let Some(guard) = self.pending.take() {
            self.split(guard)?;
        }
        Ok(self.b.op(opcode, args)?.into())
    }

    /// Makes the conditional block: the current block tests `guard` and branches to a new
    /// block, which jumps to a second new block that takes the current block's terminator.
    fn split(&mut self, guard: PredSrc) -> Result<(), BuildError> {
        let here = self.b.block();
        let mut cond: Operand = self
            .b
            .op(Opcode::GetPred, &[Operand::Imm(Imm::Pred(guard.reg))])?
            .into();
        if guard.negated {
            cond = self.b.op(Opcode::LogicalNot, &[cond])?.into();
        }
        let old = self
            .b
            .function()
            .block(here)
            .map(|block| block.terminator)
            .ok_or(BuildError::NoBlock(here))?;
        let then = self.b.add_block()?;
        let join = self.b.add_block()?;
        self.b.terminate(Terminator::Branch {
            cond,
            then,
            otherwise: join,
        })?;
        self.b.switch_to(join)?;
        self.b.terminate(old)?;
        self.b.switch_to(then)?;
        self.b.terminate(Terminator::Jump(join))?;
        self.join = Some(join);
        Ok(())
    }

    /// Leaves the builder after the instruction: in the block after its conditional block, if
    /// it has one.
    fn finish(self) -> Result<(), IntegerError> {
        if let Some(join) = self.join {
            self.b.switch_to(join)?;
        }
        Ok(())
    }

    /// Reads a register's bits; `RZ` reads zero (`Visitor::X`).
    fn get(&mut self, reg: Gpr) -> Result<Operand, BuildError> {
        if reg.is_zero() {
            return Ok(imm32(0));
        }
        self.op(Opcode::GetRegister, &[Operand::Imm(Imm::Reg(reg))])
    }

    /// Writes a register; a write to `RZ` is dropped (`Visitor::X`).
    fn set(&mut self, reg: Gpr, value: Operand) -> Result<(), BuildError> {
        if !reg.is_zero() {
            self.op(Opcode::SetRegister, &[Operand::Imm(Imm::Reg(reg)), value])?;
        }
        Ok(())
    }

    /// Reads a predicate source; `PT` reads true (`Visitor::Pred`).
    fn pred(&mut self, src: PredSrc) -> Result<Operand, BuildError> {
        if src.reg.is_true() {
            return Ok(imm1(!src.negated));
        }
        let value = self.op(Opcode::GetPred, &[Operand::Imm(Imm::Pred(src.reg))])?;
        if src.negated {
            return self.op(Opcode::LogicalNot, &[value]);
        }
        Ok(value)
    }

    /// Reads a uniform predicate source; `UPT` reads true (`Visitor::UPred`).
    fn uniform_pred(&mut self, src: UniformPredSrc) -> Result<Operand, BuildError> {
        if src.reg.is_true() {
            return Ok(imm1(!src.negated));
        }
        let value = self.op(
            Opcode::GetUniformPred,
            &[Operand::Imm(Imm::UniformPred(src.reg))],
        )?;
        if src.negated {
            return self.op(Opcode::LogicalNot, &[value]);
        }
        Ok(value)
    }

    /// Writes a predicate; a write to `PT` is dropped (`Visitor::SetPred`).
    fn set_pred(&mut self, pred: Pred, value: Operand) -> Result<(), BuildError> {
        if !pred.is_true() {
            self.op(Opcode::SetPred, &[Operand::Imm(Imm::Pred(pred)), value])?;
        }
        Ok(())
    }

    /// A source's bits, without its modifier (the caller applies it). A bindless constant
    /// buffer is refused before any instruction is translated, so it reads zero here.
    fn src(&mut self, src: AluSrc) -> Result<Operand, BuildError> {
        match src {
            AluSrc::Reg { reg, .. } => self.get(reg),
            AluSrc::Imm(bits) => Ok(imm32(bits)),
            AluSrc::UniformReg { reg, .. } => self.uniform(reg),
            AluSrc::CBuf {
                cbuf:
                    CBufRef {
                        buf: CBuf::Binding(bank),
                        offset,
                    },
                ..
            } => self.cbuf(bank, u32::from(offset)),
            AluSrc::CBuf { .. } => Ok(imm32(0)),
        }
    }

    /// Reads a uniform register; `URZ` reads zero (`Visitor::UR`).
    fn uniform(&mut self, reg: UniformReg) -> Result<Operand, BuildError> {
        if reg.is_zero() {
            return Ok(imm32(0));
        }
        self.op(
            Opcode::GetUniformRegister,
            &[Operand::Imm(Imm::UniformReg(reg))],
        )
    }

    /// The 32 bits at byte `offset` of constant buffer `bank`: one aligned read, or the tail
    /// of one word and the head of the next for an unaligned offset (`Visitor::Cbuf`).
    fn cbuf(&mut self, bank: u8, offset: u32) -> Result<Operand, BuildError> {
        let bank = imm32(u32::from(bank));
        let skew = offset % 4;
        let aligned = offset - skew;
        if skew == 0 {
            return self.op(Opcode::GetCbufU32, &[bank, imm32(aligned)]);
        }
        let low = self.op(Opcode::GetCbufU32, &[bank, imm32(aligned)])?;
        // Cannot overflow: `offset` came from a 16-bit field.
        let high = self.op(
            Opcode::GetCbufU32,
            &[bank, imm32(aligned.saturating_add(4))],
        )?;
        let low = self.op(Opcode::ShiftRightLogical32, &[low, imm32(8 * skew)])?;
        let high = self.op(Opcode::ShiftLeftLogical32, &[high, imm32(32 - 8 * skew)])?;
        self.op(Opcode::BitwiseOr32, &[low, high])
    }

    fn select(&mut self, cond: Operand, a: Operand, b: Operand) -> Result<Operand, BuildError> {
        self.op(Opcode::SelectU32, &[cond, a, b])
    }

    /// 1 where `cond` is true, else 0.
    fn one_if(&mut self, cond: Operand) -> Result<Operand, BuildError> {
        self.select(cond, imm32(1), imm32(0))
    }

    /// `x` as the low word of a 64-bit value.
    fn widen(&mut self, x: Operand) -> Result<Operand, BuildError> {
        self.pack(x, imm32(0))
    }

    /// The 64-bit value `high:low`.
    fn pack(&mut self, low: Operand, high: Operand) -> Result<Operand, BuildError> {
        let pair = self.op(Opcode::CompositeConstructU32x2, &[low, high])?;
        self.op(Opcode::PackUint2x32, &[pair])
    }

    /// The low and high words of a 64-bit value.
    fn words(&mut self, x: Operand) -> Result<(Operand, Operand), BuildError> {
        let pair = self.op(Opcode::UnpackUint2x32, &[x])?;
        let low = self.op(Opcode::CompositeExtractU32x2, &[pair, imm32(0)])?;
        let high = self.op(Opcode::CompositeExtractU32x2, &[pair, imm32(1)])?;
        Ok((low, high))
    }

    /// The function a truth table gives of `inputs` (bit `in0 << 2 | in1 << 1 | in2`, Eden's
    /// `ApplyLUT` and PR #4588's `PLOP3`), as a sum of products: the PR's `evaluate` for
    /// predicates, used for bitwise values too.
    fn lut(
        &mut self,
        ops: &LogicOps,
        lut: u8,
        inputs: [Operand; 3],
    ) -> Result<Operand, BuildError> {
        if lut == 0 {
            return Ok(ops.none);
        }
        if lut == u8::MAX {
            return Ok(ops.all);
        }
        let mut inverted: [Option<Operand>; 3] = [None; 3];
        let mut result: Option<Operand> = None;
        for row in 0..LUT_ROWS {
            if lut & (1 << row) == 0 {
                continue;
            }
            let mut term: Option<Operand> = None;
            for (k, input) in (0_u8..).zip(inputs) {
                let wanted = row & (1 << (2 - k)) != 0;
                let literal = if wanted {
                    input
                } else if let Some(Some(not)) = inverted.get(usize::from(k)) {
                    *not
                } else {
                    let not = self.op(ops.not, &[input])?;
                    if let Some(slot) = inverted.get_mut(usize::from(k)) {
                        *slot = Some(not);
                    }
                    not
                };
                term = Some(match term {
                    None => literal,
                    Some(term) => self.op(ops.and, &[term, literal])?,
                });
            }
            let term = term.unwrap_or(ops.all);
            result = Some(match result {
                None => term,
                Some(result) => self.op(ops.or, &[result, term])?,
            });
        }
        Ok(result.unwrap_or(ops.none))
    }

    /// The high word of the 64-bit product, from 16-bit limbs (`Visitor::MultiplyHigh`).
    fn multiply_high(
        &mut self,
        a: Operand,
        b: Operand,
        signed: bool,
    ) -> Result<Operand, BuildError> {
        let mask = imm32(0xffff);
        let sixteen = imm32(16);
        let a_lo = self.op(Opcode::BitwiseAnd32, &[a, mask])?;
        let a_hi = self.op(Opcode::ShiftRightLogical32, &[a, sixteen])?;
        let b_lo = self.op(Opcode::BitwiseAnd32, &[b, mask])?;
        let b_hi = self.op(Opcode::ShiftRightLogical32, &[b, sixteen])?;
        // The PR's `t`, `u` and `v`: the partial sums of the limb products.
        let low_product = self.op(Opcode::IMul32, &[a_lo, b_lo])?;
        let cross_a = self.op(Opcode::IMul32, &[a_hi, b_lo])?;
        let low_carry = self.op(Opcode::ShiftRightLogical32, &[low_product, sixteen])?;
        let middle = self.op(Opcode::IAdd32, &[cross_a, low_carry])?;
        let cross_b = self.op(Opcode::IMul32, &[a_lo, b_hi])?;
        let middle_low = self.op(Opcode::BitwiseAnd32, &[middle, mask])?;
        let second_middle = self.op(Opcode::IAdd32, &[cross_b, middle_low])?;
        let high_product = self.op(Opcode::IMul32, &[a_hi, b_hi])?;
        let middle_high = self.op(Opcode::ShiftRightLogical32, &[middle, sixteen])?;
        let partial = self.op(Opcode::IAdd32, &[high_product, middle_high])?;
        let second_high = self.op(Opcode::ShiftRightLogical32, &[second_middle, sixteen])?;
        let mut hi = self.op(Opcode::IAdd32, &[partial, second_high])?;
        if signed {
            let zero = imm32(0);
            let a_negative = self.op(Opcode::SLessThan, &[a, zero])?;
            let fix_a = self.select(a_negative, b, zero)?;
            let b_negative = self.op(Opcode::SLessThan, &[b, zero])?;
            let fix_b = self.select(b_negative, a, zero)?;
            hi = self.op(Opcode::ISub32, &[hi, fix_a])?;
            hi = self.op(Opcode::ISub32, &[hi, fix_b])?;
        }
        Ok(hi)
    }

    fn instr(&mut self, op: Op) -> Result<(), IntegerError> {
        match op {
            Op::Mov(op) => self.mov(op),
            Op::IAdd3(op) => self.iadd3(op, false),
            Op::IAdd3X(op) => self.iadd3(op.add, true),
            Op::Lea(op) => self.lea(op, false),
            Op::LeaX(op) => self.lea(op.lea, true),
            Op::Lop3(op) => self.lop3(op),
            Op::IAbs(op) => self.unary(op.dst, op.src, Opcode::IAbs32),
            Op::BRev(op) => self.unary(op.dst, op.src, Opcode::BitReverse32),
            Op::PopC(op) => self.popc(op),
            Op::Flo(op) => self.flo(op),
            Op::IMnMx(op) => self.imnmx(op),
            Op::Sel(op) => self.sel(op),
            Op::Sgxt(op) => self.sgxt(op),
            Op::Prmt(op) => self.prmt(op),
            Op::Shf(op) => self.shf(op),
            Op::IMad(op) => self.imad(op),
            Op::IMad64(op) => self.imad_wide(op),
            Op::IDp4(op) => self.idp(op),
            Op::ISetP(op) => self.isetp(op),
            Op::PLop3(op) => self.plop3(op),
            Op::P2R(op) => self.p2r(op),
            Op::R2P(op) => self.r2p(op),
            Op::BMsk(_) => Err(IntegerUnsupported::Instruction("bmsk").into()),
            other => Err(IntegerError::OtherGroup(other.mnemonic())),
        }
    }

    /// `MOV` (`ops_move.cpp`).
    fn mov(&mut self, op: OpMov) -> Result<(), IntegerError> {
        require(
            op.quad_lanes == OpMov::ALL_LANES,
            IntegerUnsupported::MovLaneMask,
        )?;
        let value = self.src(op.src.to_alu())?;
        Ok(self.set(op.dst, value)?)
    }

    /// One source to one destination through `opcode` (`IABS`, `BREV`).
    fn unary(&mut self, dst: Gpr, src: AluSrc, opcode: Opcode) -> Result<(), IntegerError> {
        let value = self.src(src)?;
        let result = self.op(opcode, &[value])?;
        Ok(self.set(dst, result)?)
    }

    /// `IADD3` and `IADD3.X` (`Visitor::IADD3`): a 32-bit sum, or with a carry output or `.X` a
    /// 64-bit sum whose high word gives the carries.
    fn iadd3(&mut self, op: OpIAdd3, x: bool) -> Result<(), IntegerError> {
        let no_carry_in = op.carry.iter().all(|carry| *carry == NOT_PT);
        require(x || no_carry_in, IntegerUnsupported::Iadd3CarryPredicate)?;
        let mut values = [imm32(0); 3];
        for (value, src) in values.iter_mut().zip(op.srcs) {
            *value = self.src(src)?;
        }
        let negated = op.srcs.map(|src| src.modifiers().1);
        if !x && op.overflow.iter().all(|pred| pred.is_true()) {
            for (value, negated) in values.iter_mut().zip(negated) {
                if negated {
                    *value = self.op(Opcode::INeg32, &[*value])?;
                }
            }
            let [a, b, c] = values;
            let sum = self.op(Opcode::IAdd32, &[a, b])?;
            let sum = self.op(Opcode::IAdd32, &[sum, c])?;
            return Ok(self.set(op.dst, sum)?);
        }
        let mut terms = [imm64(0); 3];
        for ((term, value), negated) in terms.iter_mut().zip(values).zip(negated) {
            *term = if negated {
                // Integer negation is the two's complement `~x + 1` as a 64-bit term, so it
                // carries (`term` in the PR); with `.X` it is the bitwise not alone.
                let not = self.op(Opcode::BitwiseNot32, &[value])?;
                let wide = self.widen(not)?;
                if x {
                    wide
                } else {
                    self.op(Opcode::IAdd64, &[wide, imm64(1)])?
                }
            } else {
                self.widen(value)?
            };
        }
        let [a, b, c] = terms;
        let mut sum = self.op(Opcode::IAdd64, &[a, b])?;
        sum = self.op(Opcode::IAdd64, &[sum, c])?;
        if x {
            for carry in op.carry {
                let carry = self.pred(carry)?;
                let one = self.one_if(carry)?;
                let one = self.widen(one)?;
                sum = self.op(Opcode::IAdd64, &[sum, one])?;
            }
        }
        let (low, high) = self.words(sum)?;
        self.set(op.dst, low)?;
        let [first, second] = op.overflow;
        if !first.is_true() {
            let carried = self.op(Opcode::INotEqual, &[high, imm32(0)])?;
            self.set_pred(first, carried)?;
        }
        if !second.is_true() {
            let carried = self.op(Opcode::UGreaterThanEqual, &[high, imm32(2)])?;
            self.set_pred(second, carried)?;
        }
        Ok(())
    }

    /// `LEA` and `LEA.X` (`Visitor::LEA`), with the PR's checks in its order.
    fn lea(&mut self, op: OpLea, x: bool) -> Result<(), IntegerError> {
        use IntegerUnsupported as U;
        require(!op.intermediate, U::LeaIntermediate)?;
        require(x || op.carry == NOT_PT, U::LeaCarryPredicate)?;
        require(op.overflow.is_true() || !op.dst_high, U::LeaHighCarryOutput)?;
        require(op.dst_high || !op.sx32, U::LeaSignExtendWithoutHigh)?;
        require(
            !op.sx32 || op.a_high == AluSrc::reg(Gpr::RZ),
            U::LeaSignExtendWithHighWord,
        )?;
        let negated = op.b.modifiers().1;
        let shift = u32::from(op.shift);
        if !op.dst_high && (x || !op.overflow.is_true()) {
            // `(a << shift) + b (+ carry)` as a 64-bit sum: the low word is the result and the
            // high word the carry; with `.X` a negated `b` is a bitwise not.
            let mut addend = self.src(op.b)?;
            if negated {
                let opcode = if x {
                    Opcode::BitwiseNot32
                } else {
                    Opcode::INeg32
                };
                addend = self.op(opcode, &[addend])?;
            }
            let a = self.src(op.a)?;
            let shifted = if shift == 0 {
                a
            } else {
                self.op(Opcode::ShiftLeftLogical32, &[a, imm32(shift)])?
            };
            let shifted = self.widen(shifted)?;
            let addend = self.widen(addend)?;
            let mut sum = self.op(Opcode::IAdd64, &[shifted, addend])?;
            if x {
                let carry = self.pred(op.carry)?;
                let one = self.one_if(carry)?;
                let one = self.widen(one)?;
                sum = self.op(Opcode::IAdd64, &[sum, one])?;
            }
            let (low, high) = self.words(sum)?;
            self.set(op.dst, low)?;
            if !op.overflow.is_true() {
                let carried = self.op(Opcode::INotEqual, &[high, imm32(0)])?;
                self.set_pred(op.overflow, carried)?;
            }
            return Ok(());
        }
        let low = self.src(op.a)?;
        let mut addend = self.src(op.b)?;
        if negated {
            addend = self.op(Opcode::INeg32, &[addend])?;
        }
        if !op.dst_high {
            let shifted = if shift == 0 {
                low
            } else {
                self.op(Opcode::ShiftLeftLogical32, &[low, imm32(shift)])?
            };
            let sum = self.op(Opcode::IAdd32, &[shifted, addend])?;
            return Ok(self.set(op.dst, sum)?);
        }
        // `.HI`: `b` plus the high word of `(high:low) << shift`.
        let high = if op.sx32 {
            self.op(Opcode::ShiftRightArithmetic32, &[low, imm32(31)])?
        } else {
            self.src(op.a_high)?
        };
        let mut result = if shift == 0 {
            high
        } else {
            let up = self.op(Opcode::ShiftLeftLogical32, &[high, imm32(shift)])?;
            let down = self.op(Opcode::ShiftRightLogical32, &[low, imm32(32 - shift)])?;
            self.op(Opcode::BitwiseOr32, &[up, down])?
        };
        result = self.op(Opcode::IAdd32, &[result, addend])?;
        if x {
            let carry = self.pred(op.carry)?;
            let one = self.one_if(carry)?;
            result = self.op(Opcode::IAdd32, &[result, one])?;
        }
        Ok(self.set(op.dst, result)?)
    }

    /// `LOP3.LUT` (`Visitor::LOP3`): the table's function of three values, and whether it is
    /// not zero.
    fn lop3(&mut self, op: OpLop3) -> Result<(), IntegerError> {
        let [a, b, c] = op.srcs;
        let inputs = [self.src(a)?, self.src(b)?, self.src(c)?];
        let result = self.lut(&BITWISE, op.lut, inputs)?;
        self.set(op.dst, result)?;
        if !op.pred.is_true() {
            let nonzero = self.op(Opcode::INotEqual, &[result, imm32(0)])?;
            self.set_pred(op.pred, nonzero)?;
        }
        Ok(())
    }

    /// `POPC` (`Visitor::POPC`): the bits set, of the source or its bitwise not.
    fn popc(&mut self, op: OpPopC) -> Result<(), IntegerError> {
        let mut value = self.src(op.src)?;
        if op.src.modifiers().1 {
            value = self.op(Opcode::BitwiseNot32, &[value])?;
        }
        let count = self.op(Opcode::BitCount32, &[value])?;
        Ok(self.set(op.dst, count)?)
    }

    /// `FLO` (`Visitor::FLO`): the highest set bit (the highest bit unlike the sign, signed),
    /// `-1` for none, or with `.SH` its distance from bit 31.
    fn flo(&mut self, op: OpFlo) -> Result<(), IntegerError> {
        let mut value = self.src(op.src)?;
        if op.src.modifiers().1 {
            value = self.op(Opcode::BitwiseNot32, &[value])?;
        }
        let opcode = if op.signed {
            Opcode::FindSMsb32
        } else {
            Opcode::FindUMsb32
        };
        let mut msb = self.op(opcode, &[value])?;
        if op.return_shift_amount {
            msb = self.op(Opcode::ISub32, &[imm32(31), msb])?;
        }
        Ok(self.set(op.dst, msb)?)
    }

    /// `IMNMX` (`Visitor::IMNMX`): the minimum where the predicate is true, else the maximum.
    fn imnmx(&mut self, op: OpIMnMx) -> Result<(), IntegerError> {
        let [a, b] = op.srcs;
        let a = self.src(a)?;
        let b = self.src(b)?;
        let (min, max) = if op.signed {
            (Opcode::SMin32, Opcode::SMax32)
        } else {
            (Opcode::UMin32, Opcode::UMax32)
        };
        let min = self.op(min, &[a, b])?;
        let max = self.op(max, &[a, b])?;
        let cond = self.pred(op.min)?;
        let result = self.select(cond, min, max)?;
        Ok(self.set(op.dst, result)?)
    }

    /// `SEL` (`Visitor::SEL(false)`).
    fn sel(&mut self, op: OpSel) -> Result<(), IntegerError> {
        let [a, b] = op.srcs;
        let a = self.src(a)?;
        let b = self.src(b)?;
        let cond = self.pred(op.cond)?;
        let result = self.select(cond, a, b)?;
        Ok(self.set(op.dst, result)?)
    }

    /// `SGXT` (`Visitor::SGXT`, NAK's semantics): the low `bits` bits, sign- or
    /// zero-extended; 32 or more keeps the value, 0 gives 0.
    fn sgxt(&mut self, op: OpSgxt) -> Result<(), IntegerError> {
        require(!op.wrap, IntegerUnsupported::SgxtWrap)?;
        let value = self.src(op.a)?;
        let count = self.src(op.bits)?;
        let rest = self.op(Opcode::ISub32, &[imm32(32), count])?;
        let shift = self.op(Opcode::BitwiseAnd32, &[rest, imm32(31)])?;
        let shifted = self.op(Opcode::ShiftLeftLogical32, &[value, shift])?;
        let down = if op.signed {
            Opcode::ShiftRightArithmetic32
        } else {
            Opcode::ShiftRightLogical32
        };
        let extended = self.op(down, &[shifted, shift])?;
        let none = self.op(Opcode::IEqual, &[count, imm32(0)])?;
        let narrow = self.select(none, imm32(0), extended)?;
        let all = self.op(Opcode::UGreaterThanEqual, &[count, imm32(32)])?;
        let result = self.select(all, value, narrow)?;
        Ok(self.set(op.dst, result)?)
    }

    /// `PRMT` in its default mode with an immediate selector (`Visitor::PRMT`): each nibble of
    /// the selector picks a byte of `b:a` (bits 0..3), or with bit 3 that byte's sign repeated.
    fn prmt(&mut self, op: OpPrmt) -> Result<(), IntegerError> {
        require(op.mode == PrmtMode::Index, IntegerUnsupported::PermuteMode)?;
        let AluSrc::Imm(selector) = op.sel else {
            return Err(IntegerUnsupported::PermuteSelector.into());
        };
        let low = self.src(op.srcs[0])?;
        let high = self.src(op.srcs[1])?;
        let mut result = imm32(0);
        for byte in 0..4_u32 {
            let nibble = (selector >> (4 * byte)) & 0xf;
            let source = if nibble & 4 != 0 { high } else { low };
            let offset = (nibble & 3) * 8;
            let value = if nibble & 8 != 0 {
                let sign = self.op(
                    Opcode::BitFieldUExtract,
                    &[source, imm32(offset + 7), imm32(1)],
                )?;
                self.op(Opcode::IMul32, &[sign, imm32(0xff)])?
            } else {
                self.op(Opcode::BitFieldUExtract, &[source, imm32(offset), imm32(8)])?
            };
            result = self.op(
                Opcode::BitFieldInsert,
                &[result, value, imm32(byte * 8), imm32(8)],
            )?;
        }
        Ok(self.set(op.dst, result)?)
    }

    /// `SHF` (NAK's `OpShf` fold for SM 7.0 and later): `high:low` shifted by the amount,
    /// masked to the type's width less one with `.W`, else clamped to the width; a signed type
    /// shifts right arithmetically; `.HI` writes the high word.
    fn shf(&mut self, op: OpShf) -> Result<(), IntegerError> {
        let (width, signed) = match op.data_type {
            ShfType::I64 => (64, true),
            ShfType::U64 => (64, false),
            ShfType::I32 => (32, true),
            ShfType::U32 => (32, false),
        };
        let low = self.src(op.low)?;
        let high = self.src(op.high)?;
        let packed = self.pack(low, high)?;
        let amount = self.src(op.shift)?;
        let amount = if op.wrap {
            self.op(Opcode::BitwiseAnd32, &[amount, imm32(width - 1)])?
        } else {
            self.op(Opcode::UMin32, &[amount, imm32(width)])?
        };
        let opcode = if !op.right {
            Opcode::ShiftLeftLogical64
        } else if signed {
            Opcode::ShiftRightArithmetic64
        } else {
            Opcode::ShiftRightLogical64
        };
        let shifted = self.op(opcode, &[packed, amount])?;
        let pair = self.op(Opcode::UnpackUint2x32, &[shifted])?;
        let word = self.op(
            Opcode::CompositeExtractU32x2,
            &[pair, imm32(u32::from(op.dst_high))],
        )?;
        Ok(self.set(op.dst, word)?)
    }

    /// `IMAD` (`Visitor::IMAD`) and `IMAD.HI` (`Visitor::IMAD_HI`).
    fn imad(&mut self, op: OpIMad) -> Result<(), IntegerError> {
        let [a, b, c] = op.srcs;
        let IMadMode::Lo { x, carry } = op.mode else {
            let a = self.src(a)?;
            let b = self.src(b)?;
            let c = self.src(c)?;
            let hi = self.multiply_high(a, b, op.signed)?;
            let result = self.op(Opcode::IAdd32, &[hi, c])?;
            return Ok(self.set(op.dst, result)?);
        };
        let [neg_a, neg_b, neg_c] = op.srcs.map(|src| src.modifiers().1);
        require(x || carry == NOT_PT, IntegerUnsupported::ImadCarryInput)?;
        require(
            !x || (!neg_a && !neg_b),
            IntegerUnsupported::ImadCarryNegatedFactor,
        )?;
        let mut a = self.src(a)?;
        let mut b = self.src(b)?;
        let mut c = self.src(c)?;
        if neg_a {
            a = self.op(Opcode::INeg32, &[a])?;
        }
        if neg_b {
            b = self.op(Opcode::INeg32, &[b])?;
        }
        if neg_c {
            let opcode = if x {
                Opcode::BitwiseNot32
            } else {
                Opcode::INeg32
            };
            c = self.op(opcode, &[c])?;
        }
        let product = self.op(Opcode::IMul32, &[a, b])?;
        let mut sum = self.op(Opcode::IAdd32, &[product, c])?;
        if x {
            let carry = self.pred(carry)?;
            let one = self.one_if(carry)?;
            sum = self.op(Opcode::IAdd32, &[sum, one])?;
        }
        Ok(self.set(op.dst, sum)?)
    }

    /// `IMAD.WIDE` (`Visitor::IMAD_WIDE`): the 64-bit `a * b + c` into a register pair.
    fn imad_wide(&mut self, op: OpIMad64) -> Result<(), IntegerError> {
        use IntegerUnsupported as U;
        require(op.carry == NOT_PT, U::WideCarryInput)?;
        require(pair_start(op.dst.index()), U::WideDestination(op.dst))?;
        let [a, b, c] = op.srcs;
        match c {
            AluSrc::Reg { reg, .. } if !reg.is_zero() => {
                require(pair_start(reg.index()), U::WideAddendRegister(reg))?;
            }
            AluSrc::UniformReg { reg, .. } => {
                require(pair_start(reg.index()), U::WideAddendUniform(reg))?;
            }
            AluSrc::Imm(_) => return Err(U::WideAddendImmediate.into()),
            AluSrc::Reg { .. } | AluSrc::CBuf { .. } => {}
        }
        let c_low = self.src(c)?;
        let c_high = match c {
            AluSrc::Reg { reg, .. } if !reg.is_zero() => {
                self.get(Gpr::new(reg.index().saturating_add(1)))?
            }
            AluSrc::UniformReg { reg, .. } => {
                // Cannot fail: an even uniform register is at most 62.
                let next = reg.index().saturating_add(1);
                self.uniform(UniformReg::new(next).unwrap_or(UniformReg::URZ))?
            }
            AluSrc::CBuf {
                cbuf:
                    CBufRef {
                        buf: CBuf::Binding(bank),
                        offset,
                    },
                ..
            } => self.cbuf(bank, u32::from(offset) + 4)?,
            _ => imm32(0),
        };
        let a = self.src(a)?;
        let b = self.src(b)?;
        let low = self.op(Opcode::IMul32, &[a, b])?;
        let high = self.multiply_high(a, b, op.signed)?;
        let product = self.pack(low, high)?;
        let addend = self.pack(c_low, c_high)?;
        let sum = self.op(Opcode::IAdd64, &[product, addend])?;
        let (low, high) = self.words(sum)?;
        self.set(op.dst, low)?;
        Ok(self.set(Gpr::new(op.dst.index().saturating_add(1)), high)?)
    }

    /// `IDP.2A.LO.U16.U8` (`Visitor::IDP`): `c` plus the two 16-bit halves of `a` times the two
    /// low bytes of `b`, unsigned.
    fn idp(&mut self, op: OpIDp4) -> Result<(), IntegerError> {
        require(op.two_a, IntegerUnsupported::IdpFourWay)?;
        require(
            op.signed == [false, false] && op.srcs.iter().all(|src| !src.modifiers().1),
            IntegerUnsupported::IdpModifiers,
        )?;
        let [a, b, c] = op.srcs;
        let a = self.src(a)?;
        let b = self.src(b)?;
        let low_a = self.op(Opcode::BitwiseAnd32, &[a, imm32(0xffff)])?;
        let high_a = self.op(Opcode::ShiftRightLogical32, &[a, imm32(16)])?;
        let byte_0 = self.op(Opcode::BitwiseAnd32, &[b, imm32(0xff)])?;
        let b_down = self.op(Opcode::ShiftRightLogical32, &[b, imm32(8)])?;
        let byte_1 = self.op(Opcode::BitwiseAnd32, &[b_down, imm32(0xff)])?;
        let first = self.op(Opcode::IMul32, &[low_a, byte_0])?;
        let second = self.op(Opcode::IMul32, &[high_a, byte_1])?;
        let sum = self.op(Opcode::IAdd32, &[first, second])?;
        let c = self.src(c)?;
        let result = self.op(Opcode::IAdd32, &[sum, c])?;
        Ok(self.set(op.dst, result)?)
    }

    /// `ISETP` (`Visitor::ISETP`): the comparison combined with the accumulator into the first
    /// destination, and its inverse into the second.
    fn isetp(&mut self, op: OpISetP) -> Result<(), IntegerError> {
        require(op.low_cmp.is_true(), IntegerUnsupported::IsetpLowPredicate)?;
        require(!op.ex, IntegerUnsupported::IsetpExtended)?;
        let [a, b] = op.srcs;
        let a = self.src(a)?;
        let b = self.src(b)?;
        let compare = self.compare(op.cmp_op, op.signed, a, b)?;
        let accum = self.pred(op.accum)?;
        let first = self.combine(op.set_op, compare, accum)?;
        self.set_pred(op.dsts[0], first)?;
        let inverse = self.op(Opcode::LogicalNot, &[compare])?;
        let second = self.combine(op.set_op, inverse, accum)?;
        Ok(self.set_pred(op.dsts[1], second)?)
    }

    /// Eden's `IntegerCompare`.
    fn compare(
        &mut self,
        cmp: IntCmpOp,
        signed: bool,
        a: Operand,
        b: Operand,
    ) -> Result<Operand, BuildError> {
        let pick = |s, u| if signed { s } else { u };
        let opcode = match cmp {
            IntCmpOp::False => return Ok(imm1(false)),
            IntCmpOp::True => return Ok(imm1(true)),
            IntCmpOp::Lt => pick(Opcode::SLessThan, Opcode::ULessThan),
            IntCmpOp::Eq => Opcode::IEqual,
            IntCmpOp::Le => pick(Opcode::SLessThanEqual, Opcode::ULessThanEqual),
            IntCmpOp::Gt => pick(Opcode::SGreaterThan, Opcode::UGreaterThan),
            IntCmpOp::Ne => Opcode::INotEqual,
            IntCmpOp::Ge => pick(Opcode::SGreaterThanEqual, Opcode::UGreaterThanEqual),
        };
        self.op(opcode, &[a, b])
    }

    /// Eden's `PredicateCombine`.
    fn combine(&mut self, op: PredSetOp, a: Operand, b: Operand) -> Result<Operand, BuildError> {
        let opcode = match op {
            PredSetOp::And => Opcode::LogicalAnd,
            PredSetOp::Or => Opcode::LogicalOr,
            PredSetOp::Xor => Opcode::LogicalXor,
        };
        self.op(opcode, &[a, b])
    }

    /// `PLOP3.LUT` (`Visitor::PLOP3`): two truth tables of three predicates, the inputs read
    /// once before either destination is written.
    fn plop3(&mut self, op: OpPLop3) -> Result<(), IntegerError> {
        let [a, b] = op.srcs;
        let a = self.pred(a)?;
        let b = self.pred(b)?;
        let c = match op.c {
            PLop3Src::Pred(c) => self.pred(c)?,
            PLop3Src::Uniform(c) => self.uniform_pred(c)?,
        };
        for (dst, lut) in op.dsts.into_iter().zip(op.luts) {
            if !dst.is_true() {
                let value = self.lut(&LOGICAL, lut, [a, b, c])?;
                self.set_pred(dst, value)?;
            }
        }
        Ok(())
    }

    /// `P2R` (`Visitor::P2R`): the source with the bits the mask selects replaced by `P0` to
    /// `P6`.
    fn p2r(&mut self, op: OpP2R) -> Result<(), IntegerError> {
        let mask = u32::from(op.mask);
        let mut insert = imm32(0);
        for index in 0..MOVED_PREDICATES {
            let Some(pred) = Pred::new(index).filter(|_| mask & (1 << index) != 0) else {
                continue;
            };
            let value = self.pred(PredSrc {
                reg: pred,
                negated: false,
            })?;
            let bit = self.select(value, imm32(1 << index), imm32(0))?;
            insert = self.op(Opcode::BitwiseOr32, &[insert, bit])?;
        }
        let src = self.get(op.src)?;
        let kept = self.op(Opcode::BitwiseAnd32, &[src, imm32(!mask)])?;
        let result = self.op(Opcode::BitwiseOr32, &[kept, insert])?;
        Ok(self.set(op.dst, result)?)
    }

    /// `R2P` (`Visitor::R2P`): `P0` to `P6`, where the mask selects them, from the bits of one
    /// byte of the source.
    fn r2p(&mut self, op: OpR2P) -> Result<(), IntegerError> {
        let mask = u32::from(op.mask);
        let value = self.get(op.src)?;
        for index in 0..MOVED_PREDICATES {
            let Some(pred) = Pred::new(index).filter(|_| mask & (1 << index) != 0) else {
                continue;
            };
            let at = u32::from(op.byte) * 8 + u32::from(index);
            let bit = self.op(Opcode::BitFieldUExtract, &[value, imm32(at), imm32(1)])?;
            let set = self.op(Opcode::INotEqual, &[bit, imm32(0)])?;
            self.set_pred(pred, set)?;
        }
        Ok(())
    }
}

/// Whether a register pair of `IMAD.WIDE` may start at `index`: even and below
/// [`WIDE_PAIR_LIMIT`].
const fn pair_start(index: u8) -> bool {
    index.is_multiple_of(2) && index < WIDE_PAIR_LIMIT
}
