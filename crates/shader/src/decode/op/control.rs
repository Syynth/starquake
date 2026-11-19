//! Slice 6h, control and warp: branches, convergence barriers, calls, warp operations and
//! waits. Every instruction of the group is decoded except `JMX` and `BPT`, which stay shells:
//! neither NAK nor PR #4588's table has an encoding for them (`op`'s module docs; the
//! proposals' "Shader decoder follow-ups"), and no other reference in the reference folder
//! names their opcodes.
//!
//! # Sources
//!
//! Shapes follow NAK's `Op*` structs and encodings NAK's `impl SM70Op` blocks
//! (`mesa-nouveau/src/nouveau/compiler/nak/ir.rs` and `sm70_encode.rs` @ 52238762, MIT,
//! Copyright 2022 Collabora, Ltd.), as each constant says: `BRA`, `EXIT`, `KILL`, `BSSY`,
//! `BSYNC`, `BREAK`, `BMOV`, `WARPSYNC`, `BAR`, `VOTE`, `SHFL`, `MATCH`, `REDUX`, `NOP` and
//! `NANOSLEEP`. Where NAK writes a field as a constant (`EXIT`'s modes, `KILL`'s and
//! `WARPSYNC`'s predicates, `NANOSLEEP`'s modifier bits), the op carries it as a field so the
//! decoder can read any value of it.
//!
//! `BRX` (with its uniform form `BRXU`), `CALL`, `RET` and `DEPBAR` come from PR #4588 alone
//! (`eden-pr4588/src/shader_recompiler/frontend/ampere/ampere_isa_table.h` for the masks,
//! `visitor.cpp`, `Visitor::Run`, for the fields it takes, and `flow.cpp` and
//! `lower_to_maxwell.h` for what it requires, @ e5fb3b97). **Each of these four needs an
//! `nvdisasm` check** (spec, verification E2): the field split and the names below are read off
//! the PR's code, not off an encoder.
//!
//! # Branch offsets
//!
//! A relative branch target is a signed count of 32-bit words, from bit 34, added to the
//! address of the next instruction (PR #4588: `frontend/ampere/instruction.h`,
//! `RelativeOffset48` and `RelativeOffset30`, "SM70+ relative control-flow immediates start at
//! bit 34 and are expressed in 32-bit words"; `flow.cpp`, `BranchTarget`: `index * 16 + 16 +
//! relative`; NAK: `get_rel_offset` returns `target_ip - ip - 4`). `BSSY`'s offset is 30 bits
//! wide (`set_rel_offset(34..64)`, `RelativeOffset30`); `BRA`'s, `BRX`'s, `CALL`'s and `RET`'s
//! are 48 (`set_rel_offset(34..82)`, `RelativeOffset48`). [`BranchOffset`] carries the word
//! count; checking that a target lies inside the captured program is the job of capture and
//! flow analysis (slices 3 and 10), not the decoder's.
//!
//! # Decisions taken here
//!
//! - **`CALL.REL.NOINC`** (scoping doc row 67): bit 86 of `CALL` is read as `.NOINC` and of
//!   `RET` as `.NODEC`. A guess: PR #4588 requires the bit set in both (`visitor.cpp`: "native
//!   call increment mode", "native return decrement mode"), and names the calls it accepts
//!   `CALL.REL.NOINC` (`lower_to_maxwell.h`, case `0x944`).
//! - **`BCLEAR`** is owned by this slice and decodes as `BMOV`: NAK's `impl SM70Op for
//!   OpBClear` writes `BMOV`'s from-barrier form (`0x355`) with the destination `RZ` and bit 84
//!   (`.CLEAR`) set, the same bits as `OpBMov { dst: RZ, clear: true }`, so it is an
//!   [`OpBMov`] with [`BMovDir::FromBarrier`] and `clear`, printed `bmov.32.clear rz, bN`.
//!   PR #4588's row names the opcode `BClear/BMov`. Whether it should print as `bclear` is for
//!   the `nvdisasm` check.
//! - **`DEPBAR`'s fields** are PR #4588's `Take(38, 6)` (the count) and `Require(Take(44, 4) ==
//!   0xd)` with the comment `DEPBAR.LE SB5, 1`: bits 44..47 are read as the scoreboard (5) and
//!   bit 47 as `.LE` (1), a guess that splits the PR's four-bit field.
//! - **`BAR`'s fields** in form `0xb1d` (NAK's only live encoding) are the row's: bits 32..40,
//!   74..76, 77..79, 80 and 87..91. Their meanings come from NAK's commented-out encoding of form
//!   `0x31d` (a register at 32..40, the reduction at 74..76: `RED.POPC`, `.AND`, `.OR`; the mode
//!   at 77..79: `SYNC`, `ARV`, `RED`, `SCAN`; a predicate at 87..91), and bit 80, which PR #4588
//!   ignores (`ops_memory.cpp`, `Visitor::BAR`, `Ignore(80, 1)`), is named
//!   `defer_blocking` after `nvdisasm`'s usual spelling of a workgroup barrier: all guesses. NAK's
//!   live encoding writes zero to bits 87..91, which reads as `P0`.
//! - **The uniform `BRA` (`0x547`) and `BRXU` (`0x958`) require bit 91 set**, as the ALU
//!   uniform forms do (`decode`'s module docs): NAK writes it (`set_bit(91, true)`) and PR #4588
//!   requires it for `BRXU` (`flow.cpp`, "native computed branch mode"). The uniform `BRA` also
//!   requires bits 87..90 set (NAK: `set_field(87..90, 0x7)`), since its row has no bit 90 to
//!   negate that predicate with; what other values mean is not in the references.
//!
//! # Divergences from PR #4588's table
//!
//! | Rows | What this decoder does | Pinned by |
//! |---|---|---|
//! | `WARPSYNC` and `NANOSLEEP` forms 2, 3 and 7 (`0x548`, `0x748`, `0xf48`, `0x55d`, `0x75d`, `0xf5d`) | Unknown opcodes: NAK's `encode_alu(.., None, None, Some(src), None)` puts the one source in source 1, which only forms 1, 4, 5 and 6 hold, as for `MOV` | `decode::tests::control::control_rows_left_undecoded_are_unknown_opcodes` |
//! | Bits 62 and 63 of `WARPSYNC` and `NANOSLEEP` forms 1, 5 and 6 (`0x348`, `0xb48`, `0xd48`, `0x35d`, `0xb5d`, `0xd5d`) | Rejected: NAK never modifies these sources (a `u32` mask; `OpNanosleep::time` is unmodified), as for `MOV` | `control_sources_reject_modifier_bits` |
//! | `BAR` form `0x31d` | Unknown opcode: NAK's encoding of it is commented out and PR #4588's translator accepts only `0xb1d` (`visitor.cpp`, case `0xb1d`) | `control_rows_left_undecoded_are_unknown_opcodes` |
//! | Bits 87..90 and 91 of the uniform `BRA`, bit 91 of `BRXU` | Required set (see above); the table marks them known only | `control_fixed_bits_are_required` |
//!
//! Every other control form's mask is its row's exactly (`decode::tests::masks_match_the_reference_table`).
//!
//! # Values the decoder refuses
//!
//! A field whose value names nothing is [`DecodeError::UnknownValue`]: `VOTE`'s operation 3
//! (NAK: `All` 0, `Any` 1, `Eq` 2; PR #4588: `Require(vote <= 2)`), `REDUX`'s operations 6 and 7
//! (NAK: `And` 0 to `Max` 5; PR #4588: `Require(operation <= 5)`) and `BAR`'s reduction 3 (NAK's
//! comment lists 0 to 2). `REDUX`'s signed bit (73) is read only for `MIN` and `MAX` (NAK writes
//! it for those alone, and PR #4588 uses it for those alone), so with another operation it is an
//! unknown bit (`decode`'s step 5).

use std::fmt::{self, Display, Formatter};

use super::super::DecodeError;
use super::super::alu::{self, AluForm, Source, Sources};
use super::super::fields::{DST, mask_of};
use super::super::form::{Form, Reader};
use super::super::instruction::Field;
use super::super::operand::{AluSrc, Gpr, Pred, PredSrc, UniformPred, UniformPredSrc, UniformReg};
use super::super::print::{IntOperand, flag};
use super::{Named, Op, shells};
#[cfg(test)]
use {
    super::super::fields::{SLOT_ABS, SLOT_NEG},
    super::super::form::ReferenceRow,
    proptest::strategy::BoxedStrategy,
};

// ---------------------------------------------------------------------------------------------
// Fields several instructions share.

/// The predicate most control instructions carry besides their guard (NAK: `set_pred_src(87..90,
/// 90, ..)` in `OpBra`, `OpBSSy`, `OpBSync`, `OpBreak`, `OpWarpSync`, `OpKill`, `OpVote`,
/// `OpNanosleep` and `OpExit`).
pub(crate) const COND: Field = Field::new(87, Pred::BITS);
/// [`COND`]'s negation.
pub(crate) const COND_NOT: Field = Field::bit(90);
/// A predicate destination at bits 81..84 (NAK: `set_pred_dst(81..84, ..)` in `OpVote`,
/// `OpShfl` and `OpMatch`); `PT` writes nothing.
pub(crate) const PRED_DST: Field = Field::new(81, Pred::BITS);
/// A uniform register destination (NAK: `set_udst`, `set_ureg(16, ..)`, six bits below SM 10.0).
pub(crate) const UNIFORM_DST: Field = Field::new(16, UniformReg::BITS);
/// The 48-bit branch offset of `BRA`, `BRX`, `CALL` and `RET` (NAK `OpBra`:
/// `set_rel_offset(34..82)`; PR #4588 `visitor.cpp`: `Take(34, 48)`).
pub(crate) const BRANCH_OFFSET: Field = Field::new(34, 48);
/// `BSSY`'s 30-bit offset (NAK `OpBSSy`: `set_rel_offset(34..64)`; PR #4588:
/// `RelativeOffset30`).
pub(crate) const BSSY_OFFSET: Field = Field::new(34, 30);
/// The convergence barrier of `BSSY`, `BSYNC` and `BREAK` (NAK: `set_bar_dst(16..20, ..)`,
/// `set_bar_src(16..20, ..)`).
pub(crate) const CONV_BARRIER: Field = Field::new(16, Barrier::BITS);
/// The uniform-source bit of the uniform `BRA` and of `BRXU` (NAK `OpBra`: `set_bit(91, true)`;
/// PR #4588 `visitor.cpp`, `BRXU`: `Take(91, 1)`).
pub(crate) const UNIFORM_BIT: Field = Field::bit(91);

// ---------------------------------------------------------------------------------------------
// Operand types.

/// A convergence barrier register, `B0` to `B15` (NAK: `RegFile::Bar`, 16 registers on SM 7.0
/// and later, `sm70.rs`, `num_regs`; printed `b` and the number, `ir.rs`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Barrier(u8);

impl Barrier {
    /// The width of a barrier field (NAK's `set_bar_reg` asserts a 4-bit range).
    pub const BITS: u32 = 4;
    /// How many barrier registers there are.
    pub const COUNT: u8 = 16;

    /// Barrier `index`, if it exists.
    pub const fn new(index: u8) -> Option<Self> {
        if index < Self::COUNT {
            Some(Self(index))
        } else {
            None
        }
    }

    /// The barrier's number.
    pub const fn index(self) -> u8 {
        self.0
    }
}

impl Display for Barrier {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "b{}", self.0)
    }
}

/// A relative branch offset: a signed count of 32-bit words from the address of the next
/// instruction (see the module docs). The field it is read from bounds it (48 or 30 bits); the
/// type itself holds any `i64`, and the assembler refuses one its field cannot hold.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BranchOffset(i64);

impl BranchOffset {
    /// The size of the unit the offset counts, in bytes (PR #4588, `instruction.h`: "expressed
    /// in 32-bit words").
    pub const WORD_BYTES: i64 = 4;

    /// An offset of `words` 32-bit words.
    pub const fn from_words(words: i64) -> Self {
        Self(words)
    }

    /// The offset in 32-bit words.
    pub const fn words(self) -> i64 {
        self.0
    }

    /// The offset in bytes, or `None` when that overflows an `i64` (never for an offset the
    /// decoder read: 48 bits of words are at most 2^49 bytes).
    pub const fn bytes(self) -> Option<i64> {
        self.0.checked_mul(Self::WORD_BYTES)
    }
}

impl Display for BranchOffset {
    /// The displacement in bytes from the next instruction, signed, in hexadecimal: `+0x10`,
    /// `-0x20`. `nvdisasm` prints the absolute target instead, which the printer cannot know (it
    /// sees one instruction, not its address): a deliberate divergence.
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        // Cannot overflow: an i64 times four fits in an i128.
        let bytes = i128::from(self.0) * i128::from(Self::WORD_BYTES);
        let sign = if bytes < 0 { "-" } else { "+" };
        write!(f, "{sign}{:#x}", bytes.unsigned_abs())
    }
}

/// Writes `cond, ` when `cond` is not plain `PT`.
fn cond_prefix(f: &mut Formatter<'_>, cond: PredSrc) -> fmt::Result {
    if cond.is_true() {
        Ok(())
    } else {
        write!(f, "{cond}, ")
    }
}

// ---------------------------------------------------------------------------------------------
// BRA.

/// `BRA` with a predicate condition (NAK `impl SM70Op for OpBra`: `set_opcode(0x947)`).
pub(crate) const BRA_OPCODE: u16 = 0x947;
/// `BRA` with a uniform predicate condition (`set_opcode(0x547)`, SM 8.0 and later).
pub(crate) const BRA_UNIFORM_OPCODE: u16 = 0x547;
/// `BRA`'s `.U` bit (`set_bit(32, false)` in `0x947`, `set_bit(32, true)` in `0x547`).
pub(crate) const BRA_U: Field = Field::bit(32);
/// The uniform `BRA`'s condition (`set_upred_src(24..27, 27, ..)`).
pub(crate) const BRA_UPRED: Field = Field::new(24, UniformPred::BITS);
/// The uniform condition's negation.
pub(crate) const BRA_UPRED_NOT: Field = Field::bit(27);
/// The bits every uniform `BRA` has set: `set_field(87..90, 0x7)` and `set_bit(91, true)`.
pub(crate) const BRA_UNIFORM_FIXED: u128 = COND.mask() | UNIFORM_BIT.mask();

/// A `BRA`'s condition.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BraCond {
    /// A predicate (form `0x947`).
    Pred(PredSrc),
    /// A uniform predicate (form `0x547`).
    UniformPred(UniformPredSrc),
}

/// `BRA`: branches by `offset` where the guard and the condition are true.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OpBra {
    /// The condition.
    pub cond: BraCond,
    /// Bit 32 (NAK: `.U`, set in the uniform form).
    pub uniform: bool,
    /// The target.
    pub offset: BranchOffset,
}

impl Named for OpBra {
    const MNEMONIC: &'static str = "bra";
}

impl Display for OpBra {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str("bra")?;
        flag(f, self.uniform, ".u")?;
        f.write_str(" ")?;
        match self.cond {
            BraCond::Pred(cond) => cond_prefix(f, cond)?,
            // Always printed, so the two forms read differently.
            BraCond::UniformPred(cond) => write!(f, "{cond}, ")?,
        }
        write!(f, "{}", self.offset)
    }
}

fn decode_bra(reader: &mut Reader) -> Op {
    let cond = if reader.opcode() == u64::from(BRA_UNIFORM_OPCODE) {
        // Fixed bits of the form: the decoder has checked that they are set.
        reader.field(COND);
        reader.flag(UNIFORM_BIT);
        BraCond::UniformPred(reader.uniform_pred_src(BRA_UPRED, BRA_UPRED_NOT))
    } else {
        BraCond::Pred(reader.pred_src(COND, COND_NOT))
    };
    Op::Bra(OpBra {
        cond,
        uniform: reader.flag(BRA_U),
        offset: BranchOffset(reader.signed(BRANCH_OFFSET)),
    })
}

// ---------------------------------------------------------------------------------------------
// BRX.

/// `BRX` through a register (PR #4588 `ampere_isa_table.h`: row `0x949`, `BRX`). Needs an
/// `nvdisasm` check.
pub(crate) const BRX_OPCODE: u16 = 0x949;
/// `BRX` through a uniform register (row `0x958`, `BRXU`). Needs an `nvdisasm` check.
pub(crate) const BRXU_OPCODE: u16 = 0x958;
/// `BRX`'s register (`visitor.cpp`, `BRX`: `Take(24, 8)`).
pub(crate) const BRX_REG: Field = Field::new(24, Gpr::BITS);
/// `BRXU`'s uniform register (`BRXU`: `Take(24, 6)`).
pub(crate) const BRX_UNIFORM_REG: Field = Field::new(24, UniformReg::BITS);

/// Where a `BRX` reads its target from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BrxSrc {
    /// A register (form `0x949`).
    Reg(Gpr),
    /// A uniform register (form `0x958`, `BRXU`).
    UniformReg(UniformReg),
}

impl Display for BrxSrc {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Reg(reg) => write!(f, "{reg}"),
            Self::UniformReg(reg) => write!(f, "{reg}"),
        }
    }
}

/// `BRX`: an indirect branch through a register and an offset. PR #4588 accepts only an
/// offset that points at the program's start, so the register holds an address from there
/// (`flow.cpp`, "native computed branch mode").
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OpBrx {
    /// The register holding the target.
    pub src: BrxSrc,
    /// The offset added to it.
    pub offset: BranchOffset,
    /// The condition (bits 87..91).
    pub cond: PredSrc,
}

impl Named for OpBrx {
    const MNEMONIC: &'static str = "brx";
}

impl Display for OpBrx {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str("brx ")?;
        cond_prefix(f, self.cond)?;
        write!(f, "{} {}", self.src, self.offset)
    }
}

fn decode_brx(reader: &mut Reader) -> Op {
    let src = if reader.opcode() == u64::from(BRXU_OPCODE) {
        // A fixed bit of the form: the decoder has checked that it is set.
        reader.flag(UNIFORM_BIT);
        BrxSrc::UniformReg(reader.uniform_reg(BRX_UNIFORM_REG))
    } else {
        BrxSrc::Reg(reader.gpr(BRX_REG))
    };
    Op::Brx(OpBrx {
        src,
        offset: BranchOffset(reader.signed(BRANCH_OFFSET)),
        cond: reader.pred_src(COND, COND_NOT),
    })
}

// ---------------------------------------------------------------------------------------------
// CALL and RET.

/// `CALL.REL` (PR #4588 row `0x944`, `CALL`; `classify.h`: "call (0x944)"). Needs an
/// `nvdisasm` check.
pub(crate) const CALL_OPCODE: u16 = 0x944;
/// `CALL`'s `.NOINC` bit (a guess, see the module docs; `visitor.cpp`: `Require(Take(86, 1) ==
/// 1, "native call increment mode")`).
pub(crate) const CALL_NO_INC: Field = Field::bit(86);
/// `RET.REL` (row `0x950`, `RET`; `classify.h`: "return through a register (0x950)"). Needs an
/// `nvdisasm` check.
pub(crate) const RET_OPCODE: u16 = 0x950;
/// `RET`'s return-address register (`visitor.cpp`, case `0x950`: `Take(24, 8)`).
pub(crate) const RET_REG: Field = Field::new(24, Gpr::BITS);
/// `RET`'s `.NODEC` bit (a guess; `Require(Take(86, 1) == 1, "native return decrement
/// mode")`).
pub(crate) const RET_NO_DEC: Field = Field::bit(86);

/// `CALL.REL`: calls the subroutine at `offset`. With `.NOINC` and no return-address setup it is
/// a forward branch (scoping doc row 67; PR #4588 `flow.cpp`, `IsSkipCall`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OpCall {
    /// The target.
    pub offset: BranchOffset,
    /// Bit 86, `.NOINC` (a guess).
    pub no_inc: bool,
    /// The condition (bits 87..91).
    pub cond: PredSrc,
}

/// `RET.REL`: returns to the address in `src`, relative to `offset` (PR #4588 accepts only an
/// offset that points at the program's start: `flow.cpp`, "native return linkage").
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OpRet {
    /// The register holding the return address.
    pub src: Gpr,
    /// The offset the address is relative to.
    pub offset: BranchOffset,
    /// Bit 86, `.NODEC` (a guess).
    pub no_dec: bool,
    /// The condition (bits 87..91).
    pub cond: PredSrc,
}

impl Named for OpCall {
    const MNEMONIC: &'static str = "call";
}

impl Named for OpRet {
    const MNEMONIC: &'static str = "ret";
}

impl Display for OpCall {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str("call.rel")?;
        flag(f, self.no_inc, ".noinc")?;
        f.write_str(" ")?;
        cond_prefix(f, self.cond)?;
        write!(f, "{}", self.offset)
    }
}

impl Display for OpRet {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str("ret.rel")?;
        flag(f, self.no_dec, ".nodec")?;
        f.write_str(" ")?;
        cond_prefix(f, self.cond)?;
        write!(f, "{} {}", self.src, self.offset)
    }
}

fn decode_call(reader: &mut Reader) -> Op {
    Op::Call(OpCall {
        offset: BranchOffset(reader.signed(BRANCH_OFFSET)),
        no_inc: reader.flag(CALL_NO_INC),
        cond: reader.pred_src(COND, COND_NOT),
    })
}

fn decode_ret(reader: &mut Reader) -> Op {
    Op::Ret(OpRet {
        src: reader.gpr(RET_REG),
        offset: BranchOffset(reader.signed(BRANCH_OFFSET)),
        no_dec: reader.flag(RET_NO_DEC),
        cond: reader.pred_src(COND, COND_NOT),
    })
}

// ---------------------------------------------------------------------------------------------
// EXIT and KILL.

/// `EXIT` (`impl SM70Op for OpExit`: `set_opcode(0x94d)`).
pub(crate) const EXIT_OPCODE: u16 = 0x94d;
/// `EXIT`'s first mode bit; NAK's comment names it `./.KEEPREFCOUNT/.PREEMPTED/.INVALID3`
/// together with bit 85 (`set_field(84..85, false)`).
pub(crate) const EXIT_KEEP_REF_COUNT: Field = Field::bit(84);
/// `EXIT`'s `.NO_ATEXIT` bit (`set_field(85..86, false)`).
pub(crate) const EXIT_NO_AT_EXIT: Field = Field::bit(85);

/// `EXIT`: ends the thread.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OpExit {
    /// Bit 84 (NAK: `./.KEEPREFCOUNT/.PREEMPTED/.INVALID3` with bit 85).
    pub keep_ref_count: bool,
    /// Bit 85 (NAK: `.NO_ATEXIT`).
    pub no_at_exit: bool,
    /// The exit's own predicate (NAK always writes `PT`: `set_field(87..90, 0x7)`, "TODO:
    /// Predicate").
    pub pred: PredSrc,
}

impl Named for OpExit {
    const MNEMONIC: &'static str = "exit";
}

impl Display for OpExit {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str("exit")?;
        flag(f, self.keep_ref_count, ".keeprefcount")?;
        flag(f, self.no_at_exit, ".no_atexit")?;
        if !self.pred.is_true() {
            write!(f, " {}", self.pred)?;
        }
        Ok(())
    }
}

fn decode_exit(reader: &mut Reader) -> Op {
    Op::Exit(OpExit {
        keep_ref_count: reader.flag(EXIT_KEEP_REF_COUNT),
        no_at_exit: reader.flag(EXIT_NO_AT_EXIT),
        pred: reader.pred_src(COND, COND_NOT),
    })
}

/// `KILL` (`impl SM70Op for OpKill`: `set_opcode(0x95b)`).
pub(crate) const KILL_OPCODE: u16 = 0x95b;

/// `KILL`: discards the thread (pixel shaders).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OpKill {
    /// The kill's own predicate (NAK always writes `PT`: `set_pred_src(87..90, 90, True)`).
    pub pred: PredSrc,
}

impl Named for OpKill {
    const MNEMONIC: &'static str = "kill";
}

impl Display for OpKill {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str("kill")?;
        if !self.pred.is_true() {
            write!(f, " {}", self.pred)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------------------------
// Convergence barriers: BSSY, BSYNC, BREAK, BMOV.

/// `BSSY` (`impl SM70Op for OpBSSy`: `set_opcode(0x945)`).
pub(crate) const BSSY_OPCODE: u16 = 0x945;
/// `BSYNC` (`impl SM70Op for OpBSync`: `set_opcode(0x941)`).
pub(crate) const BSYNC_OPCODE: u16 = 0x941;
/// `BREAK` (`impl SM70Op for OpBreak`: `set_opcode(0x942)`).
pub(crate) const BREAK_OPCODE: u16 = 0x942;
/// `BMOV` from a register to a barrier (`impl SM70Op for OpBMov`: `set_opcode(0x356)`).
pub(crate) const BMOV_TO_BARRIER_OPCODE: u16 = 0x356;
/// `BMOV` from a barrier to a register, and `BCLEAR` (`set_opcode(0x355)` in `OpBMov` and
/// `OpBClear`).
pub(crate) const BMOV_FROM_BARRIER_OPCODE: u16 = 0x355;
/// `BMOV`'s barrier, in both directions (`set_bar_dst(24..28, ..)`, `set_bar_src(24..28, ..)`).
pub(crate) const BMOV_BARRIER: Field = Field::new(24, Barrier::BITS);
/// The register `BMOV` copies into a barrier (`set_reg_src(32..40, ..)`).
pub(crate) const BMOV_SRC: Field = Field::new(32, Gpr::BITS);
/// `BMOV`'s `.CLEAR` bit (`set_bit(84, self.clear)`; `OpBClear`: "`.CLEAR`").
pub(crate) const BMOV_CLEAR: Field = Field::bit(84);

/// `BSSY bar, offset`: sets convergence barrier `bar` up, to reconverge at `offset`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OpBSSy {
    /// The barrier.
    pub bar: Barrier,
    /// The reconvergence point (30 bits of words).
    pub offset: BranchOffset,
    /// The condition (bits 87..91).
    pub cond: PredSrc,
}

/// `BSYNC bar`: waits at convergence barrier `bar`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OpBSync {
    /// The barrier.
    pub bar: Barrier,
    /// The condition (bits 87..91).
    pub cond: PredSrc,
}

/// `BREAK bar`: removes the threads from convergence barrier `bar`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OpBreak {
    /// The barrier.
    pub bar: Barrier,
    /// The condition (bits 87..91).
    pub cond: PredSrc,
}

/// Which way a `BMOV` copies.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BMovDir {
    /// From a register into a barrier (form `0x356`).
    ToBarrier {
        /// The barrier written.
        bar: Barrier,
        /// The register read.
        src: Gpr,
    },
    /// From a barrier into a register (form `0x355`); `BCLEAR` is this with `RZ` and `clear`.
    FromBarrier {
        /// The register written.
        dst: Gpr,
        /// The barrier read.
        bar: Barrier,
    },
}

/// `BMOV.32`: copies a convergence barrier's state to or from a register.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OpBMov {
    /// The direction and operands.
    pub dir: BMovDir,
    /// `.CLEAR`: clears the barrier.
    pub clear: bool,
}

impl Named for OpBSSy {
    const MNEMONIC: &'static str = "bssy";
}

impl Named for OpBSync {
    const MNEMONIC: &'static str = "bsync";
}

impl Named for OpBreak {
    const MNEMONIC: &'static str = "break";
}

impl Named for OpBMov {
    const MNEMONIC: &'static str = "bmov";
}

impl Display for OpBSSy {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str("bssy ")?;
        cond_prefix(f, self.cond)?;
        write!(f, "{}, {}", self.bar, self.offset)
    }
}

impl Display for OpBSync {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str("bsync ")?;
        cond_prefix(f, self.cond)?;
        write!(f, "{}", self.bar)
    }
}

impl Display for OpBreak {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str("break ")?;
        cond_prefix(f, self.cond)?;
        write!(f, "{}", self.bar)
    }
}

impl Display for OpBMov {
    /// NAK's `bmov.32` and `.clear`, then the destination and the source.
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str("bmov.32")?;
        flag(f, self.clear, ".clear")?;
        match self.dir {
            BMovDir::ToBarrier { bar, src } => write!(f, " {bar}, {src}"),
            BMovDir::FromBarrier { dst, bar } => write!(f, " {dst}, {bar}"),
        }
    }
}

fn barrier(reader: &mut Reader, field: Field) -> Barrier {
    // Cannot fail: the fields are `Barrier::BITS` wide.
    Barrier::new(reader.byte(field)).unwrap_or(Barrier(0))
}

fn decode_bssy(reader: &mut Reader) -> Op {
    Op::BSSy(OpBSSy {
        bar: barrier(reader, CONV_BARRIER),
        offset: BranchOffset(reader.signed(BSSY_OFFSET)),
        cond: reader.pred_src(COND, COND_NOT),
    })
}

fn decode_bsync(reader: &mut Reader) -> Op {
    Op::BSync(OpBSync {
        bar: barrier(reader, CONV_BARRIER),
        cond: reader.pred_src(COND, COND_NOT),
    })
}

fn decode_break(reader: &mut Reader) -> Op {
    Op::Break(OpBreak {
        bar: barrier(reader, CONV_BARRIER),
        cond: reader.pred_src(COND, COND_NOT),
    })
}

fn decode_bmov(reader: &mut Reader) -> Op {
    let dir = if reader.opcode() == u64::from(BMOV_TO_BARRIER_OPCODE) {
        BMovDir::ToBarrier {
            bar: barrier(reader, BMOV_BARRIER),
            src: reader.gpr(BMOV_SRC),
        }
    } else {
        BMovDir::FromBarrier {
            dst: reader.gpr(DST),
            bar: barrier(reader, BMOV_BARRIER),
        }
    };
    Op::BMov(OpBMov {
        dir,
        clear: reader.flag(BMOV_CLEAR),
    })
}

// ---------------------------------------------------------------------------------------------
// WARPSYNC and NANOSLEEP: one ALU source.

/// `WARPSYNC`'s 9-bit opcode (`impl SM70Op for OpWarpSync`: `encode_alu(0x148, None, None,
/// Some(mask), None)`).
pub(crate) const WARPSYNC_OPCODE: u16 = 0x148;
/// `NANOSLEEP`'s 9-bit opcode (`impl SM70Op for OpNanosleep`: `encode_alu(0x15d, None, None,
/// Some(time), None)`).
pub(crate) const NANOSLEEP_OPCODE: u16 = 0x15d;
/// The sources of `WARPSYNC` and `NANOSLEEP`: source 1 only.
pub(crate) const SOURCE_1_ONLY: Sources = Sources {
    src0: false,
    src1: true,
    src2: false,
};
/// The operand forms of `WARPSYNC` and `NANOSLEEP`: those that put source 1 in the wide slot.
pub(crate) const SOURCE_1_FORMS: [AluForm; 4] = [
    AluForm::Reg,
    AluForm::ImmSrc1,
    AluForm::CBufSrc1,
    AluForm::UniformSrc1,
];
/// `NANOSLEEP`'s `.CLEAR` bit (`set_bit(83, false); // .CLEAR, SM80+`).
pub(crate) const NANOSLEEP_CLEAR: Field = Field::bit(83);
/// `NANOSLEEP`'s `.WARP` bit (`set_bit(85, false)`).
pub(crate) const NANOSLEEP_WARP: Field = Field::bit(85);
/// `NANOSLEEP`'s `.RAND` bit (`set_bit(86, false)`).
pub(crate) const NANOSLEEP_RAND: Field = Field::bit(86);

/// `WARPSYNC mask`: waits for the lanes of `mask`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OpWarpSync {
    /// The lane mask: a register, an immediate, a constant-buffer word or a uniform register,
    /// never modified (NAK's mask is a plain `u32`).
    pub mask: AluSrc,
    /// The condition (NAK always writes `PT`: `set_pred_src(87..90, 90, True)`).
    pub cond: PredSrc,
}

/// `NANOSLEEP time`: suspends the thread for about `time` nanoseconds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OpNanosleep {
    /// The time, never modified.
    pub time: AluSrc,
    /// Bit 83, `.CLEAR` (SM 8.0 and later).
    pub clear: bool,
    /// Bit 85, `.WARP`.
    pub warp: bool,
    /// Bit 86, `.RAND`.
    pub rand: bool,
    /// The condition (NAK always writes `PT`).
    pub cond: PredSrc,
}

impl Named for OpWarpSync {
    const MNEMONIC: &'static str = "warpsync";
}

impl Named for OpNanosleep {
    const MNEMONIC: &'static str = "nanosleep";
}

impl Display for OpWarpSync {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str("warpsync ")?;
        cond_prefix(f, self.cond)?;
        write!(f, "{}", IntOperand(self.mask))
    }
}

impl Display for OpNanosleep {
    /// `nanosleep` and the source, as NAK's `test_nanosleep` pins; the modifiers and the
    /// condition are guesses.
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str("nanosleep")?;
        flag(f, self.clear, ".clear")?;
        flag(f, self.warp, ".warp")?;
        flag(f, self.rand, ".rand")?;
        f.write_str(" ")?;
        cond_prefix(f, self.cond)?;
        write!(f, "{}", IntOperand(self.time))
    }
}

fn decode_warpsync(reader: &mut Reader) -> Result<Op, DecodeError> {
    let form = alu::form(reader)?;
    Ok(Op::WarpSync(OpWarpSync {
        mask: alu::src(reader, form, Source::One, false),
        cond: reader.pred_src(COND, COND_NOT),
    }))
}

fn decode_nanosleep(reader: &mut Reader) -> Result<Op, DecodeError> {
    let form = alu::form(reader)?;
    Ok(Op::Nanosleep(OpNanosleep {
        time: alu::src(reader, form, Source::One, false),
        clear: reader.flag(NANOSLEEP_CLEAR),
        warp: reader.flag(NANOSLEEP_WARP),
        rand: reader.flag(NANOSLEEP_RAND),
        cond: reader.pred_src(COND, COND_NOT),
    }))
}

/// `WARPSYNC` in `form`.
const fn warpsync(form: AluForm) -> Form {
    Form::new(
        form.opcode(WARPSYNC_OPCODE),
        mask_of(&[COND, COND_NOT]) | alu::known(form, SOURCE_1_ONLY, false),
        decode_warpsync,
    )
    .with_fixed(alu::fixed(form))
}

/// `NANOSLEEP` in `form`.
const fn nanosleep(form: AluForm) -> Form {
    Form::new(
        form.opcode(NANOSLEEP_OPCODE),
        mask_of(&[
            NANOSLEEP_CLEAR,
            NANOSLEEP_WARP,
            NANOSLEEP_RAND,
            COND,
            COND_NOT,
        ]) | alu::known(form, SOURCE_1_ONLY, false),
        decode_nanosleep,
    )
    .with_fixed(alu::fixed(form))
}

// ---------------------------------------------------------------------------------------------
// BAR.

/// `BAR` (`impl SM70Op for OpBar`: `set_opcode(0xb1d)`).
pub(crate) const BAR_OPCODE: u16 = 0xb1d;
/// `BAR`'s register (NAK's commented-out `0x31d` encoding: `set_reg_src(32..40, ..)`, "src0 ==
/// src1"; a guess for `0xb1d`).
pub(crate) const BAR_SRC: Field = Field::new(32, Gpr::BITS);
/// `BAR`'s reduction (the commented-out `set_field(74..76, ..)`: "00: RED.POPC, 01: RED.AND,
/// 02: RED.OR").
pub(crate) const BAR_RED: Field = Field::new(74, 2);
/// `BAR`'s mode (the commented-out `set_field(77..79, ..)`: "00: SYNC, 01: ARV, 02: RED, 03:
/// SCAN").
pub(crate) const BAR_MODE: Field = Field::new(77, 2);
/// Bit 80 of `BAR` (PR #4588's row; `Ignore(80, 1)`): named `defer_blocking`, a guess.
pub(crate) const BAR_DEFER_BLOCKING: Field = Field::bit(80);

/// What a `BAR` does (NAK's comment on bits 77..79).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BarMode {
    /// 0: wait for the workgroup.
    Sync,
    /// 1: arrive without waiting.
    Arrive,
    /// 2: wait and reduce.
    Reduce,
    /// 3: scan.
    Scan,
}

impl BarMode {
    /// Every mode, in encoding order.
    pub const ALL: [Self; 4] = [Self::Sync, Self::Arrive, Self::Reduce, Self::Scan];

    /// The two-bit encoding.
    pub const fn bits(self) -> u64 {
        match self {
            Self::Sync => 0,
            Self::Arrive => 1,
            Self::Reduce => 2,
            Self::Scan => 3,
        }
    }

    /// The mode an encoding names; only the low [`BAR_MODE`] bits are read.
    pub(crate) const fn from_bits(bits: u64) -> Self {
        Self::ALL[(bits & ((1 << BAR_MODE.width) - 1)) as usize]
    }
}

/// A `BAR`'s reduction (NAK's comment on bits 74..76).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BarRed {
    /// 0: population count.
    Popc,
    /// 1: and.
    And,
    /// 2: or.
    Or,
}

impl BarRed {
    /// Every reduction, in encoding order.
    pub const ALL: [Self; 3] = [Self::Popc, Self::And, Self::Or];

    /// The two-bit encoding.
    pub const fn bits(self) -> u64 {
        match self {
            Self::Popc => 0,
            Self::And => 1,
            Self::Or => 2,
        }
    }

    /// The reduction an encoding names; 3 names none.
    pub const fn from_bits(bits: u64) -> Option<Self> {
        match bits {
            0 => Some(Self::Popc),
            1 => Some(Self::And),
            2 => Some(Self::Or),
            _ => None,
        }
    }
}

/// `BAR`: a workgroup barrier. NAK encodes every field zero, which reads as `BAR.SYNC` with
/// register `R0` and predicate `P0`; the field meanings are guesses (module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OpBar {
    /// Bits 77..79.
    pub mode: BarMode,
    /// Bits 74..76.
    pub red: BarRed,
    /// Bits 32..40.
    pub src: Gpr,
    /// Bit 80.
    pub defer_blocking: bool,
    /// Bits 87..91.
    pub pred: PredSrc,
}

impl Named for OpBar {
    const MNEMONIC: &'static str = "bar";
}

impl Display for OpBar {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(match self.mode {
            BarMode::Sync => "bar.sync",
            BarMode::Arrive => "bar.arv",
            BarMode::Reduce => "bar.red",
            BarMode::Scan => "bar.scan",
        })?;
        f.write_str(match self.red {
            BarRed::Popc => "",
            BarRed::And => ".and",
            BarRed::Or => ".or",
        })?;
        flag(f, self.defer_blocking, ".defer_blocking")?;
        write!(f, " {}", self.src)?;
        if !self.pred.is_true() {
            write!(f, ", {}", self.pred)?;
        }
        Ok(())
    }
}

fn decode_bar(reader: &mut Reader) -> Result<Op, DecodeError> {
    let red_bits = reader.field(BAR_RED);
    let red = BarRed::from_bits(red_bits).ok_or_else(|| reader.unknown_value(BAR_RED, red_bits))?;
    Ok(Op::Bar(OpBar {
        mode: BarMode::from_bits(reader.field(BAR_MODE)),
        red,
        src: reader.gpr(BAR_SRC),
        defer_blocking: reader.flag(BAR_DEFER_BLOCKING),
        pred: reader.pred_src(COND, COND_NOT),
    }))
}

// ---------------------------------------------------------------------------------------------
// VOTE.

/// `VOTE` into a register (`impl SM70Op for OpVote`: `set_opcode(0x806)`).
pub(crate) const VOTE_OPCODE: u16 = 0x806;
/// `VOTE` into a uniform register and a uniform predicate (`set_opcode(0x886)`; PR #4588's
/// `VOTEU`, which writes a uniform predicate: `ops_memory.cpp`, `SetUPred`).
pub(crate) const VOTEU_OPCODE: u16 = 0x886;
/// `VOTE`'s operation (`set_field(72..74, ..)`).
pub(crate) const VOTE_OP: Field = Field::new(72, 2);

/// A vote's operation (NAK's `VoteOp` and its encoding).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum VoteOp {
    /// 0: true in every active lane.
    All,
    /// 1: true in some active lane.
    Any,
    /// 2: the same in every active lane.
    Eq,
}

impl VoteOp {
    /// Every operation, in encoding order.
    pub const ALL: [Self; 3] = [Self::All, Self::Any, Self::Eq];

    /// The two-bit encoding.
    pub const fn bits(self) -> u64 {
        match self {
            Self::All => 0,
            Self::Any => 1,
            Self::Eq => 2,
        }
    }

    /// The operation an encoding names; 3 names none.
    pub const fn from_bits(bits: u64) -> Option<Self> {
        match bits {
            0 => Some(Self::All),
            1 => Some(Self::Any),
            2 => Some(Self::Eq),
            _ => None,
        }
    }
}

impl Display for VoteOp {
    /// NAK's `VoteOp` printer.
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::All => "all",
            Self::Any => "any",
            Self::Eq => "eq",
        })
    }
}

/// Where a `VOTE` writes: the ballot and the vote's result.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum VoteDst {
    /// A register and a predicate (form `0x806`); `RZ` and `PT` write nothing.
    Reg {
        /// The ballot.
        ballot: Gpr,
        /// The result.
        vote: Pred,
    },
    /// A uniform register and a uniform predicate (form `0x886`); `URZ` and `UPT` write
    /// nothing.
    Uniform {
        /// The ballot.
        ballot: UniformReg,
        /// The result.
        vote: UniformPred,
    },
}

/// `VOTE`: a warp vote on a predicate.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OpVote {
    /// The operation.
    pub op: VoteOp,
    /// The destinations.
    pub dst: VoteDst,
    /// The predicate voted on (bits 87..91).
    pub pred: PredSrc,
}

impl Named for OpVote {
    const MNEMONIC: &'static str = "vote";
}

impl Display for OpVote {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self.dst {
            VoteDst::Reg { ballot, vote } => {
                write!(f, "vote.{} {ballot}, {vote}, {}", self.op, self.pred)
            }
            VoteDst::Uniform { ballot, vote } => {
                write!(f, "voteu.{} {ballot}, {vote}, {}", self.op, self.pred)
            }
        }
    }
}

fn decode_vote(reader: &mut Reader) -> Result<Op, DecodeError> {
    let op_bits = reader.field(VOTE_OP);
    let op = VoteOp::from_bits(op_bits).ok_or_else(|| reader.unknown_value(VOTE_OP, op_bits))?;
    let dst = if reader.opcode() == u64::from(VOTEU_OPCODE) {
        VoteDst::Uniform {
            ballot: reader.uniform_reg(UNIFORM_DST),
            // Cannot fail: the field is `UniformPred::BITS` wide.
            vote: UniformPred::new(reader.byte(PRED_DST)).unwrap_or(UniformPred::UPT),
        }
    } else {
        VoteDst::Reg {
            ballot: reader.gpr(DST),
            // Cannot fail: the field is `Pred::BITS` wide.
            vote: Pred::new(reader.byte(PRED_DST)).unwrap_or(Pred::PT),
        }
    };
    Ok(Op::Vote(OpVote {
        op,
        dst,
        pred: reader.pred_src(COND, COND_NOT),
    }))
}

// ---------------------------------------------------------------------------------------------
// SHFL.

/// `SHFL` with the lane and the clamp in registers (`impl SM70Op for OpShfl`:
/// `set_opcode(0x389)`).
pub(crate) const SHFL_OPCODE: u16 = 0x389;
/// `SHFL` with the lane in a register and the clamp immediate (`set_opcode(0x589)`).
pub(crate) const SHFL_IMM_C_OPCODE: u16 = 0x589;
/// `SHFL` with the lane immediate and the clamp in a register (`set_opcode(0x989)`).
pub(crate) const SHFL_IMM_LANE_OPCODE: u16 = 0x989;
/// `SHFL` with both immediate (`set_opcode(0xf89)`).
pub(crate) const SHFL_IMM_BOTH_OPCODE: u16 = 0xf89;
/// `SHFL`'s value register (`set_reg_src(24..32, src)`).
pub(crate) const SHFL_SRC: Field = Field::new(24, Gpr::BITS);
/// `SHFL`'s lane register (`set_reg_src(32..40, lane)`).
pub(crate) const SHFL_LANE_REG: Field = Field::new(32, Gpr::BITS);
/// `SHFL`'s immediate lane (`set_field(53..58, imm_lane)`).
pub(crate) const SHFL_LANE_IMM: Field = Field::new(53, 5);
/// `SHFL`'s clamp and segment register (`set_reg_src(64..72, c)`).
pub(crate) const SHFL_C_REG: Field = Field::new(64, Gpr::BITS);
/// `SHFL`'s immediate clamp and segment mask (`set_field(40..53, imm_c)`).
pub(crate) const SHFL_C_IMM: Field = Field::new(40, 13);
/// `SHFL`'s operation (`set_field(58..60, ..)`).
pub(crate) const SHFL_OP: Field = Field::new(58, 2);

/// A shuffle's operation (NAK's `ShflOp` and its encoding).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ShflOp {
    /// 0: from the given lane.
    Idx,
    /// 1: from a lower lane.
    Up,
    /// 2: from a higher lane.
    Down,
    /// 3: from the lane with the given bits flipped.
    Bfly,
}

impl ShflOp {
    /// Every operation, in encoding order.
    pub const ALL: [Self; 4] = [Self::Idx, Self::Up, Self::Down, Self::Bfly];

    /// The two-bit encoding.
    pub const fn bits(self) -> u64 {
        match self {
            Self::Idx => 0,
            Self::Up => 1,
            Self::Down => 2,
            Self::Bfly => 3,
        }
    }

    /// The operation an encoding names; only the low [`SHFL_OP`] bits are read.
    pub(crate) const fn from_bits(bits: u64) -> Self {
        Self::ALL[(bits & ((1 << SHFL_OP.width) - 1)) as usize]
    }
}

impl Display for ShflOp {
    /// NAK's `ShflOp` printer.
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Idx => "idx",
            Self::Up => "up",
            Self::Down => "down",
            Self::Bfly => "bfly",
        })
    }
}

/// A `SHFL` lane or clamp operand: a register or an immediate (five bits for the lane,
/// [`SHFL_LANE_IMM`]; thirteen for the clamp, [`SHFL_C_IMM`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ShflSrc {
    /// A register.
    Reg(Gpr),
    /// An immediate.
    Imm(u16),
}

impl Display for ShflSrc {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Reg(reg) => write!(f, "{reg}"),
            Self::Imm(imm) => write!(f, "{imm:#x}"),
        }
    }
}

/// `SHFL`: reads `src` from another lane of the warp.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OpShfl {
    /// The operation.
    pub op: ShflOp,
    /// The destination.
    pub dst: Gpr,
    /// Whether the source lane was in range (`PT` writes nothing).
    pub in_bounds: Pred,
    /// The value shuffled.
    pub src: Gpr,
    /// The lane, or the lane delta or mask.
    pub lane: ShflSrc,
    /// The clamp and segment mask.
    pub c: ShflSrc,
}

impl Named for OpShfl {
    const MNEMONIC: &'static str = "shfl";
}

impl Display for OpShfl {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "shfl.{} {}, {}, {}, {}, {}",
            self.op, self.in_bounds, self.dst, self.src, self.lane, self.c
        )
    }
}

fn decode_shfl(reader: &mut Reader) -> Op {
    let opcode = reader.opcode();
    let imm_lane =
        opcode == u64::from(SHFL_IMM_LANE_OPCODE) || opcode == u64::from(SHFL_IMM_BOTH_OPCODE);
    let imm_c = opcode == u64::from(SHFL_IMM_C_OPCODE) || opcode == u64::from(SHFL_IMM_BOTH_OPCODE);
    let lane = if imm_lane {
        ShflSrc::Imm(u16::from(reader.byte(SHFL_LANE_IMM)))
    } else {
        ShflSrc::Reg(reader.gpr(SHFL_LANE_REG))
    };
    let c = if imm_c {
        // Cannot truncate: the field is thirteen bits wide.
        ShflSrc::Imm(u16::try_from(reader.field(SHFL_C_IMM)).unwrap_or(u16::MAX))
    } else {
        ShflSrc::Reg(reader.gpr(SHFL_C_REG))
    };
    Op::Shfl(OpShfl {
        op: ShflOp::from_bits(reader.field(SHFL_OP)),
        dst: reader.gpr(DST),
        // Cannot fail: the field is `Pred::BITS` wide.
        in_bounds: Pred::new(reader.byte(PRED_DST)).unwrap_or(Pred::PT),
        src: reader.gpr(SHFL_SRC),
        lane,
        c,
    })
}

// ---------------------------------------------------------------------------------------------
// MATCH and REDUX.

/// `MATCH` (`impl SM70Op for OpMatch`: `set_opcode(0x3a1)`).
pub(crate) const MATCH_OPCODE: u16 = 0x3a1;
/// `MATCH`'s value register (`set_reg_src(24..32, ..)`).
pub(crate) const MATCH_SRC: Field = Field::new(24, Gpr::BITS);
/// `MATCH`'s 64-bit bit (`set_bit(73, self.u64)`).
pub(crate) const MATCH_U64: Field = Field::bit(73);
/// `MATCH`'s `.ANY` bit (`set_bit(79, ..)`: true for `MatchOp::Any`).
pub(crate) const MATCH_ANY: Field = Field::bit(79);

/// `REDUX` (`impl SM70Op for OpRedux`: `set_opcode(0x3c4)`).
pub(crate) const REDUX_OPCODE: u16 = 0x3c4;
/// `REDUX`'s value register (`set_reg_src(24..32, ..)`).
pub(crate) const REDUX_SRC: Field = Field::new(24, Gpr::BITS);
/// `REDUX`'s signed bit (`set_bit(73, ..)`: true for a signed `MIN` or `MAX`).
pub(crate) const REDUX_SIGNED: Field = Field::bit(73);
/// `REDUX`'s operation (`set_field(78..81, ..)`).
pub(crate) const REDUX_OP: Field = Field::new(78, 3);

/// A match's operation (NAK's `MatchOp`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MatchOp {
    /// Bit 79 clear: whether every active lane holds the same value.
    All,
    /// Bit 79 set: which lanes hold the same value as this one.
    Any,
}

/// `MATCH`: compares a value across the warp.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OpMatch {
    /// The operation.
    pub op: MatchOp,
    /// Whether the value is 64 bits (a register pair).
    pub u64: bool,
    /// The mask of matching lanes.
    pub mask: Gpr,
    /// The value.
    pub src: Gpr,
    /// The all-equal predicate (bits 81..84; `PT` writes nothing, and NAK writes `PT` for
    /// `.ANY`).
    pub pred: Pred,
}

/// A reduction's operation (NAK's `ReduxOp` and its encoding).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ReduxOp {
    /// 0.
    And,
    /// 1.
    Or,
    /// 2.
    Xor,
    /// 3.
    Sum,
    /// 4, signed or not.
    Min {
        /// Bit 73.
        signed: bool,
    },
    /// 5, signed or not.
    Max {
        /// Bit 73.
        signed: bool,
    },
}

impl ReduxOp {
    /// The three-bit encoding of the operation (the signed bit is apart).
    pub const fn bits(self) -> u64 {
        match self {
            Self::And => 0,
            Self::Or => 1,
            Self::Xor => 2,
            Self::Sum => 3,
            Self::Min { .. } => 4,
            Self::Max { .. } => 5,
        }
    }
}

/// `REDUX`: reduces a register across the warp into a uniform register.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OpRedux {
    /// The operation.
    pub op: ReduxOp,
    /// The destination.
    pub dst: UniformReg,
    /// The value.
    pub src: Gpr,
}

impl Named for OpMatch {
    const MNEMONIC: &'static str = "match";
}

impl Named for OpRedux {
    const MNEMONIC: &'static str = "redux";
}

impl Display for OpMatch {
    /// NAK's `test_match` text: `match.all p1, r3, r4;`, `match.any.u64 r3, r4;`.
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(match self.op {
            MatchOp::All => "match.all",
            MatchOp::Any => "match.any",
        })?;
        flag(f, self.u64, ".u64")?;
        f.write_str(" ")?;
        if !self.pred.is_true() {
            write!(f, "{}, ", self.pred)?;
        }
        write!(f, "{}, {}", self.mask, self.src)
    }
}

impl Display for OpRedux {
    /// NAK's `test_redux` text: `redux ur0, r1;` for `AND`, `.or`, `.xor`, `.sum`, `.min`,
    /// `.max`, `.min.s32` and `.max.s32` otherwise.
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str("redux")?;
        f.write_str(match self.op {
            ReduxOp::And => "",
            ReduxOp::Or => ".or",
            ReduxOp::Xor => ".xor",
            ReduxOp::Sum => ".sum",
            ReduxOp::Min { signed: false } => ".min",
            ReduxOp::Max { signed: false } => ".max",
            ReduxOp::Min { signed: true } => ".min.s32",
            ReduxOp::Max { signed: true } => ".max.s32",
        })?;
        write!(f, " {}, {}", self.dst, self.src)
    }
}

fn decode_match(reader: &mut Reader) -> Op {
    Op::Match(OpMatch {
        op: if reader.flag(MATCH_ANY) {
            MatchOp::Any
        } else {
            MatchOp::All
        },
        u64: reader.flag(MATCH_U64),
        mask: reader.gpr(DST),
        src: reader.gpr(MATCH_SRC),
        // Cannot fail: the field is `Pred::BITS` wide.
        pred: Pred::new(reader.byte(PRED_DST)).unwrap_or(Pred::PT),
    })
}

fn decode_redux(reader: &mut Reader) -> Result<Op, DecodeError> {
    let op_bits = reader.field(REDUX_OP);
    let op = match op_bits {
        0 => ReduxOp::And,
        1 => ReduxOp::Or,
        2 => ReduxOp::Xor,
        3 => ReduxOp::Sum,
        // The signed bit is read only here, so it is an unknown bit with another operation.
        4 => ReduxOp::Min {
            signed: reader.flag(REDUX_SIGNED),
        },
        5 => ReduxOp::Max {
            signed: reader.flag(REDUX_SIGNED),
        },
        _ => return Err(reader.unknown_value(REDUX_OP, op_bits)),
    };
    Ok(Op::Redux(OpRedux {
        op,
        dst: reader.uniform_reg(UNIFORM_DST),
        src: reader.gpr(REDUX_SRC),
    }))
}

// ---------------------------------------------------------------------------------------------
// NOP and DEPBAR.

/// `NOP` (`impl SM70Op for OpNop`: `set_opcode(0x918)`).
pub(crate) const NOP_OPCODE: u16 = 0x918;
/// `DEPBAR` (PR #4588 row `0x91a`, `DEPBAR`; `visitor.cpp`: "DEPBAR.LE SB5, 1"). Needs an
/// `nvdisasm` check.
pub(crate) const DEPBAR_OPCODE: u16 = 0x91a;
/// `DEPBAR`'s count (`visitor.cpp`, case `0x91a`: `Take(38, 6)`).
pub(crate) const DEPBAR_COUNT: Field = Field::new(38, 6);
/// `DEPBAR`'s scoreboard (bits 44..47 of the PR's `Take(44, 4)`; a guess, module docs).
pub(crate) const DEPBAR_SCOREBOARD: Field = Field::new(44, 3);
/// `DEPBAR`'s `.LE` (bit 47 of the PR's `Take(44, 4)`; a guess).
pub(crate) const DEPBAR_LE: Field = Field::bit(47);

/// `NOP`: does nothing.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct OpNop {}

/// `DEPBAR`: waits until a scoreboard's count of outstanding operations is at most `count`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OpDepBar {
    /// The scoreboard, 0 to 7.
    pub scoreboard: u8,
    /// `.LE`.
    pub le: bool,
    /// The count, 0 to 63.
    pub count: u8,
}

impl Named for OpNop {
    const MNEMONIC: &'static str = "nop";
}

impl Named for OpDepBar {
    const MNEMONIC: &'static str = "depbar";
}

impl Display for OpNop {
    /// NAK's `test_nop`: `nop;`.
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str("nop")
    }
}

impl Display for OpDepBar {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str("depbar")?;
        flag(f, self.le, ".le")?;
        write!(f, " sb{}, {:#x}", self.scoreboard, self.count)
    }
}

fn decode_depbar(reader: &mut Reader) -> Op {
    Op::DepBar(OpDepBar {
        scoreboard: reader.byte(DEPBAR_SCOREBOARD),
        le: reader.flag(DEPBAR_LE),
        count: reader.byte(DEPBAR_COUNT),
    })
}

// ---------------------------------------------------------------------------------------------
// The form table.

/// The control group's forms.
pub(crate) const FORMS: &[Form] = &[
    Form::new(
        BRA_OPCODE,
        mask_of(&[BRA_U, BRANCH_OFFSET, COND, COND_NOT]),
        |reader| Ok(decode_bra(reader)),
    ),
    Form::new(
        BRA_UNIFORM_OPCODE,
        mask_of(&[BRA_UPRED, BRA_UPRED_NOT, BRA_U, BRANCH_OFFSET]),
        |reader| Ok(decode_bra(reader)),
    )
    .with_fixed(BRA_UNIFORM_FIXED),
    Form::new(
        BRX_OPCODE,
        mask_of(&[BRX_REG, BRANCH_OFFSET, COND, COND_NOT]),
        |reader| Ok(decode_brx(reader)),
    ),
    Form::new(
        BRXU_OPCODE,
        mask_of(&[BRX_UNIFORM_REG, BRANCH_OFFSET, COND, COND_NOT]),
        |reader| Ok(decode_brx(reader)),
    )
    .with_fixed(UNIFORM_BIT.mask()),
    Form::new(
        CALL_OPCODE,
        mask_of(&[BRANCH_OFFSET, CALL_NO_INC, COND, COND_NOT]),
        |reader| Ok(decode_call(reader)),
    ),
    Form::new(
        RET_OPCODE,
        mask_of(&[RET_REG, BRANCH_OFFSET, RET_NO_DEC, COND, COND_NOT]),
        |reader| Ok(decode_ret(reader)),
    ),
    Form::new(
        EXIT_OPCODE,
        mask_of(&[EXIT_KEEP_REF_COUNT, EXIT_NO_AT_EXIT, COND, COND_NOT]),
        |reader| Ok(decode_exit(reader)),
    ),
    Form::new(KILL_OPCODE, mask_of(&[COND, COND_NOT]), |reader| {
        Ok(Op::Kill(OpKill {
            pred: reader.pred_src(COND, COND_NOT),
        }))
    }),
    Form::new(
        BSSY_OPCODE,
        mask_of(&[CONV_BARRIER, BSSY_OFFSET, COND, COND_NOT]),
        |reader| Ok(decode_bssy(reader)),
    ),
    Form::new(
        BSYNC_OPCODE,
        mask_of(&[CONV_BARRIER, COND, COND_NOT]),
        |reader| Ok(decode_bsync(reader)),
    ),
    Form::new(
        BREAK_OPCODE,
        mask_of(&[CONV_BARRIER, COND, COND_NOT]),
        |reader| Ok(decode_break(reader)),
    ),
    Form::new(
        BMOV_TO_BARRIER_OPCODE,
        mask_of(&[BMOV_BARRIER, BMOV_SRC, BMOV_CLEAR]),
        |reader| Ok(decode_bmov(reader)),
    ),
    Form::new(
        BMOV_FROM_BARRIER_OPCODE,
        mask_of(&[DST, BMOV_BARRIER, BMOV_CLEAR]),
        |reader| Ok(decode_bmov(reader)),
    ),
    warpsync(SOURCE_1_FORMS[0]),
    warpsync(SOURCE_1_FORMS[1]),
    warpsync(SOURCE_1_FORMS[2]),
    warpsync(SOURCE_1_FORMS[3]),
    Form::new(
        BAR_OPCODE,
        mask_of(&[
            BAR_SRC,
            BAR_RED,
            BAR_MODE,
            BAR_DEFER_BLOCKING,
            COND,
            COND_NOT,
        ]),
        decode_bar,
    ),
    Form::new(
        VOTE_OPCODE,
        mask_of(&[DST, VOTE_OP, PRED_DST, COND, COND_NOT]),
        decode_vote,
    ),
    Form::new(
        VOTEU_OPCODE,
        mask_of(&[UNIFORM_DST, VOTE_OP, PRED_DST, COND, COND_NOT]),
        decode_vote,
    ),
    Form::new(
        SHFL_OPCODE,
        mask_of(&[DST, SHFL_SRC, SHFL_LANE_REG, SHFL_C_REG, SHFL_OP, PRED_DST]),
        |reader| Ok(decode_shfl(reader)),
    ),
    Form::new(
        SHFL_IMM_C_OPCODE,
        mask_of(&[DST, SHFL_SRC, SHFL_LANE_REG, SHFL_C_IMM, SHFL_OP, PRED_DST]),
        |reader| Ok(decode_shfl(reader)),
    ),
    Form::new(
        SHFL_IMM_LANE_OPCODE,
        mask_of(&[DST, SHFL_SRC, SHFL_LANE_IMM, SHFL_C_REG, SHFL_OP, PRED_DST]),
        |reader| Ok(decode_shfl(reader)),
    ),
    Form::new(
        SHFL_IMM_BOTH_OPCODE,
        mask_of(&[DST, SHFL_SRC, SHFL_LANE_IMM, SHFL_C_IMM, SHFL_OP, PRED_DST]),
        |reader| Ok(decode_shfl(reader)),
    ),
    Form::new(
        MATCH_OPCODE,
        mask_of(&[DST, MATCH_SRC, MATCH_U64, MATCH_ANY, PRED_DST]),
        |reader| Ok(decode_match(reader)),
    ),
    Form::new(
        REDUX_OPCODE,
        mask_of(&[UNIFORM_DST, REDUX_SRC, REDUX_SIGNED, REDUX_OP]),
        decode_redux,
    ),
    Form::new(NOP_OPCODE, 0, |_| Ok(Op::Nop(OpNop {}))),
    Form::new(
        DEPBAR_OPCODE,
        mask_of(&[DEPBAR_COUNT, DEPBAR_SCOREBOARD, DEPBAR_LE]),
        |reader| Ok(decode_depbar(reader)),
    ),
    nanosleep(SOURCE_1_FORMS[0]),
    nanosleep(SOURCE_1_FORMS[1]),
    nanosleep(SOURCE_1_FORMS[2]),
    nanosleep(SOURCE_1_FORMS[3]),
];

/// The bits PR #4588's rows of the register, constant-buffer and uniform-register forms of
/// `WARPSYNC` and `NANOSLEEP` mark known that the forms reject: the wide slot's modifiers.
#[cfg(test)]
pub(crate) const SOURCE_DROPPED: u128 = mask_of(&[SLOT_ABS, SLOT_NEG]);

/// PR #4588's rows for the group's forms, in [`FORMS`] order
/// (`eden-pr4588/src/shader_recompiler/frontend/ampere/ampere_isa_table.h` @ e5fb3b97; each
/// row's four words, low word first), with the bits each form drops or requires (module docs).
#[cfg(test)]
pub(crate) const REFERENCE_ROWS: &[ReferenceRow] = &[
    ReferenceRow::new(0x947, [0x0000_ffff, 0xffff_fffd, 0x0783_ffff, 0xffff_fe00]),
    ReferenceRow::new(0x547, [0x0f00_ffff, 0xffff_fffd, 0x0b83_ffff, 0xffff_fe00])
        .fixing(BRA_UNIFORM_FIXED),
    ReferenceRow::new(0x949, [0xff00_ffff, 0xffff_fffc, 0x0783_ffff, 0xffff_fe00]),
    ReferenceRow::new(0x958, [0x3f00_ffff, 0xffff_fffc, 0x0f83_ffff, 0xffff_fe00])
        .fixing(UNIFORM_BIT.mask()),
    ReferenceRow::new(0x944, [0x0000_ffff, 0xffff_fffc, 0x07c3_ffff, 0xffff_fe00]),
    ReferenceRow::new(0x950, [0xff00_ffff, 0xffff_fffc, 0x07c3_ffff, 0xffff_fe00]),
    ReferenceRow::new(0x94d, [0x0000_ffff, 0x0000_0000, 0x07b0_0000, 0xffff_fe00]),
    ReferenceRow::new(0x95b, [0x0000_ffff, 0x0000_0000, 0x0780_0000, 0xffff_fe00]),
    ReferenceRow::new(0x945, [0x000f_ffff, 0xffff_fffc, 0x0780_0000, 0xffff_fe00]),
    ReferenceRow::new(0x941, [0x000f_ffff, 0x0000_0000, 0x0780_0000, 0xffff_fe00]),
    ReferenceRow::new(0x942, [0x000f_ffff, 0x0000_0000, 0x0780_0000, 0xffff_fe00]),
    ReferenceRow::new(0x356, [0x0f00_ffff, 0x0000_00ff, 0x0010_0000, 0xffff_fe00]),
    ReferenceRow::new(0x355, [0x0fff_ffff, 0x0000_0000, 0x0010_0000, 0xffff_fe00]),
    ReferenceRow::new(0x348, [0x0000_ffff, 0xc000_00ff, 0x0780_0000, 0xffff_fe00])
        .dropping(SOURCE_DROPPED),
    ReferenceRow::new(0x948, [0x0000_ffff, 0xffff_ffff, 0x0780_0000, 0xffff_fe00]),
    ReferenceRow::new(0xb48, [0x0000_ffff, 0xc7ff_ffff, 0x0f80_0000, 0xffff_fe00])
        .dropping(SOURCE_DROPPED),
    ReferenceRow::new(0xd48, [0x0000_ffff, 0xc000_003f, 0x0f80_0000, 0xffff_fe00])
        .dropping(SOURCE_DROPPED)
        .fixing(UNIFORM_BIT.mask()),
    ReferenceRow::new(0xb1d, [0x0000_ffff, 0x0000_00ff, 0x0781_6c00, 0xffff_fe00]),
    ReferenceRow::new(0x806, [0x00ff_ffff, 0x0000_0000, 0x078e_0300, 0xffff_fe00]),
    ReferenceRow::new(0x886, [0x003f_ffff, 0x0000_0000, 0x078e_0300, 0xffff_fe00]),
    ReferenceRow::new(0x389, [0xffff_ffff, 0x0c00_00ff, 0x000e_00ff, 0xffff_fe00]),
    ReferenceRow::new(0x589, [0xffff_ffff, 0x0c1f_ffff, 0x000e_0000, 0xffff_fe00]),
    ReferenceRow::new(0x989, [0xffff_ffff, 0x0fe0_0000, 0x000e_00ff, 0xffff_fe00]),
    ReferenceRow::new(0xf89, [0xffff_ffff, 0x0fff_ff00, 0x000e_0000, 0xffff_fe00]),
    ReferenceRow::new(0x3a1, [0xffff_ffff, 0x0000_0000, 0x000e_8200, 0xffff_fe00]),
    ReferenceRow::new(0x3c4, [0xff3f_ffff, 0x0000_0000, 0x0001_c200, 0xffff_fe00]),
    ReferenceRow::new(0x918, [0x0000_ffff, 0x0000_0000, 0x0000_0000, 0xffff_fe00]),
    ReferenceRow::new(0x91a, [0x0000_ffff, 0x0000_ffc0, 0x0000_0000, 0xffff_fe00]),
    ReferenceRow::new(0x35d, [0x0000_ffff, 0xc000_00ff, 0x07e8_0000, 0xffff_fe00])
        .dropping(SOURCE_DROPPED),
    ReferenceRow::new(0x95d, [0x0000_ffff, 0xffff_ffff, 0x07e8_0000, 0xffff_fe00]),
    ReferenceRow::new(0xb5d, [0x0000_ffff, 0xc7ff_ffff, 0x0fe8_0000, 0xffff_fe00])
        .dropping(SOURCE_DROPPED),
    ReferenceRow::new(0xd5d, [0x0000_ffff, 0xc000_003f, 0x0fe8_0000, 0xffff_fe00])
        .dropping(SOURCE_DROPPED)
        .fixing(UNIFORM_BIT.mask()),
];

/// Generators of the group's decoded operations, one per operation (`generate`).
#[cfg(test)]
pub(crate) fn any_ops() -> Vec<BoxedStrategy<Op>> {
    [generate::branches(), generate::barriers(), generate::warp()].concat()
}

#[cfg(test)]
mod generate;

shells! {
    Jmx(OpJmx) = "jmx";
    Bpt(OpBpt) = "bpt";
}
