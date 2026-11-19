//! Slice 12b: the 32-bit float and conversion instructions decoded by slice 6b (`FADD`, `FMUL`,
//! `FFMA`, `FMNMX`, `FSET`, `FSETP`, `FSEL`, `MUFU`, `FSWZADD`, `FRND`, `F2F`, `F2I`, `I2F`,
//! `I2FP`, `F2FP`) translated into IR.
//!
//! # Sources
//!
//! Ported by behaviour from PR #4588's native frontend
//! (`eden-pr4588/src/shader_recompiler/frontend/ampere/` @ e5fb3b97, GPL-3.0-or-later):
//! `ops_float.cpp` (`FMUL`, `MUFU`, `FSWZADD`), `ops_alu.cpp` (`FADD`, `FFMA`, `FMNMX`, `FSETP`,
//! `FSET`, and `FSEL` as `Visitor::SEL(true)`), `ops_convert.cpp` (`F2I`, `I2F` with `I2FP`,
//! `FRND`, `F2F`, `F2FP`) and `visitor.cpp` (sources: `Visitor::Source`, `WideSlot`, `Value`,
//! `FloatValue`, `Cbuf`; which opcodes reach which translator: `Visitor::Run`). Cross-checked
//! with Eden's Maxwell translators (`eden/src/shader_recompiler/frontend/maxwell/translate/impl/`
//! @ cf654052: `floating_point_*.cpp`, `floating_point_conversion_*.cpp`,
//! `integer_floating_point_conversion.cpp`, and `common_funcs.cpp` for the comparisons and
//! predicate combines the PR reuses).
//!
//! # What each instruction becomes
//!
//! Sources are read as the PR reads them: a register through `GetRegister` (`RZ` is the
//! immediate 0, as `Visitor::X`), a uniform register through `GetUniformRegister` (`URZ` is 0,
//! as `Visitor::UR`), an immediate as itself, and a constant-buffer word through `GetCbufU32`,
//! two aligned reads joined by shifts when the byte offset is not word aligned (`Visitor::Cbuf`).
//! A float source is bit cast, then takes its absolute value and then its negation
//! (`FloatValue`, Eden's `FPAbsNeg`). Each instruction then follows its PR translator, with
//! the controls the PR passes (`no_contraction`, the rounding field through `RoundingOf`, `FTZ`
//! or no flushing):
//!
//! - `FADD`, `FFMA`: `FPAdd32`, `FPFma32`, then `FPSaturate32` with `.SAT`.
//! - `FMUL`: the first operand times `2^(scale - 4)` (`ScaleFactor`) unless the scale is 4,
//!   then `FPMul32`, then `FPSaturate32` with `.SAT`.
//! - `FMNMX`: `FPMax32` and `FPMin32`, the predicate (read plain, the two swapped when it is
//!   negated) selecting the minimum; with `.NAN` an unordered pair gives `0x7fffffff`.
//! - `FSETP`, `FSET`: Eden's `FloatingPointCompare` (`.num` and `.nan` through `FPIsNan32`, `.f`
//!   and `.t` as constants) combined with the accumulator by `PredicateCombine`; `FSETP` writes
//!   the combination to its first predicate and the combination of the negated comparison to
//!   its second (a `PT` destination writes nothing), `FSET` writes `1.0` or `0`.
//! - `FSEL`: a select of the two modified sources' bits.
//! - `MUFU`: `FPCos`, `FPSin`, `FPExp2`, `FPLog2`, `FPRecip32`, `FPRecipSqrt32` or `FPSqrt`; the
//!   16-bit form widens the chosen half first and packs the narrowed result into the low half.
//!   `MUFU.SIN` and `MUFU.COS` take their argument in turns (row 95 of the scoping doc;
//!   NAK's `from_nir.rs` @ 52238762 maps `nir_op_fsin_normalized_2_pi`, "assert sm >= 70", to
//!   `MuFuOp::Sin`), so the argument is multiplied by `2 pi` ([`TURN`]) before `FPSin` or
//!   `FPCos`, which take radians as Eden defines them.
//! - `FSWZADD`: `FSwizzleAdd` of the two registers with the PR's swizzle immediate (the
//!   complement of the operation byte), whose arithmetic per lane Eden's SPIR-V emitter defines
//!   from the lane id (`emit_spirv_warp.cpp`, `EmitFSwizzleAdd`). The lane id is a subgroup
//!   value, so a program that uses it needs a 32-wide subgroup ("Compute shaders");
//!   [`needs_subgroup_size_32`] reports it from the translated IR.
//! - `FRND`: `FPRoundEven32`, `FPFloor32`, `FPCeil32` or `FPTrunc32` by the rounding field.
//! - `F2I`: rounded as `FRND` rounds, clamped to the destination's range in the source's
//!   format, converted with `ConvertS32*` or `ConvertU32*`, and a NaN gives 0 for a signed
//!   destination.
//! - `I2F`, `I2FP`: the selected byte or half extracted, sign- or zero-extended, then
//!   `ConvertF32S32` or `ConvertF32U32`.
//! - `F2F`: `ConvertF16F32` into the low half of the destination.
//! - `F2FP`: two `ConvertF16F32` packed, the first source (or with `.MERGE_C` the low half of
//!   the third) in the high half.
//!
//! # Rounding modes and flush-to-zero (owner's decision 10)
//!
//! The owner's decision 10 (2026-11-12, "(a), as recommended"): rounding modes and
//! flush-to-zero are carried on arithmetic, where the backend ignores them, and emulated on
//! float-to-integer and float-to-half conversions. Here:
//!
//! | Instruction | Rounding | Flush-to-zero | Pinned by |
//! |---|---|---|---|
//! | `FADD`, `FMUL`, `FFMA` | carried (`FpControl::rounding`) | carried (`FmzMode::Ftz`) | `tests::arith::arithmetic_carries_rounding_and_flushing` |
//! | `FMNMX`, `FSET`, `FSETP` | none | carried | `tests::arith::comparisons_carry_flushing` |
//! | `FRND` (float to integral float) | emulated by the operation the mode names | carried | `tests::convert::frnd_rounds_by_its_field` |
//! | `F2I` (float to integer) | emulated by the operation the mode names, as the PR does | emulated: a denormal 32-bit operand is replaced by a zero of its sign before rounding; also carried on the rounding operation | `tests::convert::f2i_flushes_denormals_before_rounding` |
//! | `F2F` (float to half) | emulated: converted to nearest, then moved one step when the wider value of the result lies on the wrong side of the operand; also carried on the conversion | emulated: the operand and a denormal half result are flushed to zeros of their sign; also carried on the conversion | `tests::convert::f2f_emulation_matches_the_modelled_conversion` |
//! | `I2F`, `I2FP` (integer to float) | carried on the conversion (decision 10 lists only float-to-integer and float-to-half conversions for emulation) | none | `tests::convert::i2f_carries_its_rounding` |
//! | `MUFU`, `FSWZADD`, `F2FP` | none: the PR accepts only to-nearest (`F2FP`, `FSWZADD`) or has no field (`MUFU`) | none | `tests::arith::fswzadd_refuses_its_modes`, `tests::convert::f2fp_refuses_the_selector_and_rounding` |
//!
//! **Owner question.** `F2I.FTZ` with a 16-bit operand is not emulated: the PR leaves the
//! denormal mode of a half operand unset (`ops_convert.cpp`, `F2I`: `denorm_cares =
//! !half_source`), and this module follows it, where decision 10 reads as emulating
//! flush-to-zero on every float-to-integer conversion. A half denormal rounded up gives 1 with
//! or without `.FTZ` (`tests::convert::f2i_flushes_denormals_before_rounding`).
//!
//! An emulated control is carried as well, so the IR means the same whether or not a backend
//! honours the flag: the emulation is idempotent (a correctly rounded or flushed value is left
//! as it is), which `tests::convert::f2f_emulation_matches_the_modelled_conversion` and the
//! fixed-seed `tests::convert::f2f_emulation_is_exact` check against the interpreter's model of
//! the flag on its own, with the translated function's flags honoured and ignored. That flushing a denormal half result is what
//! `F2F.FTZ` does on the hardware is a guess from the IR's definition of `FmzMode::Ftz` (Eden's
//! `modifiers.h`: "Flush denorms to zero"), to be confirmed by a hardware trace.
//!
//! # Guards
//!
//! An instruction under a guard predicate becomes a conditional block (spec, "Instruction
//! translation"): the guard is read, the block branches to a block holding the instruction or
//! past it, and translation continues in the block after. [`translate_instr`] skips this when
//! its block already tests the guard (flow analysis's `guard_tested`), and for `PT`.
//!
//! # Refusals
//!
//! Each `Require` of the PR's translators for these forms is a named [`FloatUnsupported`]
//! with its constant, checked in the PR's order (source slots first, then the instruction's own
//! checks, then the constant-buffer bank as the value is read). Every check runs before the
//! first instruction is appended, so a refusal leaves the function as it was
//! (`tests::refusals_append_nothing`). The order is pinned with two faults at once by
//! `tests::refusals_report_in_the_prs_order`,
//! `tests::conversion_refusals_report_in_the_prs_order`,
//! `tests::more_conversion_refusals_report_in_the_prs_order` and
//! `tests::the_bank_is_checked_last`. Two of the PR's checks cannot fail on a decoded
//! instruction and are not repeated: `Require(set_op <= 2)` of `FSET` and `FSETP`
//! ([`PredSetOp`] has three variants), and `F2I`'s `Ignore(77, 1)` is not a check.
//!
//! # Divergences from PR #4588
//!
//! | What | PR #4588 | Here | Pinned by |
//! |---|---|---|---|
//! | A guard on these instructions | a select of the result (`Visitor::Run`: `guard_select` for the pure ops `classify.h` lists) | a conditional block (spec, "Instruction translation") | `tests::guarded_instructions_become_conditional_blocks` |
//! | `@!PT` | the instruction is skipped unchecked (`guard.Never()`) | checked, and put in a block that never runs | `tests::never_guarded_instructions_are_checked` |
//! | `MUFU.SIN`, `MUFU.COS` | `FPSin`, `FPCos` of the argument as it is | of the argument times `2 pi` (row 95) | `tests::arith::mufu_sine_and_cosine_take_turns` |
//! | `F2I.FTZ` with a 32-bit operand, `F2F` rounding and `.FTZ` | carried as flags only | emulated as well (decision 10) | `tests::convert::f2i_flushes_denormals_before_rounding`, `tests::convert::f2f_emulation_matches_the_modelled_conversion` |
//! | `FSETP` with a `PT` destination | computes the combination, then `SetPred` discards it | computes nothing for it | `tests::arith::fsetp_writes_both_predicates` |
//!
//! Where the decoder reads more than the PR translates, this module refuses what the PR
//! refuses, as a listed choice: `FMUL`'s scale 0 ([`FloatUnsupported::FmulScale`]), the 8-bit
//! float size ([`FloatUnsupported::F2iSourceSize`] and its siblings), the 64-bit conversion
//! opcodes, which `Visitor::Run` sends to no translator ([`FloatUnsupported::WideConversion`]),
//! `MUFU`'s immediate form `0x908`, which it sends nowhere either
//! ([`FloatUnsupported::MufuImmediate`]), `MUFU.RCP64H`, `.RSQ64H` and `.TANH`, and the
//! `FSWZADD` operations other than the PR's two
//! (`tests::decoded_forms_the_pr_refuses_are_refused`,
//! `tests::wide_conversion_opcodes_are_refused`,
//! `tests::mufu_and_fswzadd_forms_the_pr_refuses_are_refused`).
//! The PR's `FSWZADD` works only for those two: for an operation byte it does not accept, the
//! complement it passes to Eden's `FSwizzleAdd` gives other arithmetic than the decoder's
//! documented guesses (for `add`, `-b`).

#[cfg(test)]
mod tests;

use std::fmt::{self, Display, Formatter};

use thiserror::Error;

use crate::decode::{
    AluSrc, CBuf, CBufRef, FMulScale, FSwzAddOp, FloatCmpOp, FloatType, Gpr, Instr, IntType,
    MuFuOp, Op, OpF2F, OpF2FP, OpF2I, OpFAdd, OpFFma, OpFMnMx, OpFMul, OpFRnd, OpFSel, OpFSet,
    OpFSetP, OpFSwzAdd, OpI2FP, OpMuFu, Pred, PredSetOp, PredSrc, RoundMode,
};
use crate::flow::read_pred;
use crate::ir::{
    BuildError, Builder, Flags, FmzMode, FpControl, FpRounding, Function, Imm, Inst, Opcode,
    Operand, Terminator,
};

/// The number of constant-buffer banks a source may name: the PR refuses `cbuf_index >= 32`
/// (`visitor.cpp`, `Visitor::Cbuf`: "native constant buffer address").
pub const CBUF_BANKS: u8 = 32;
/// The bytes of a constant-buffer word: a byte offset that is not a multiple of it is read as
/// the 32 bits starting there, from two aligned words (`Visitor::Cbuf`: `byte_offset % 4`).
pub const CBUF_WORD_BYTES: u16 = 4;
/// What `FSET` writes when its comparison holds: `1.0` (`ops_alu.cpp`, `FSET`:
/// `Imm32(0x3f800000U)`).
pub const FSET_TRUE: u32 = 0x3f80_0000;
/// What `FMNMX.NAN` writes for an unordered pair (`ops_alu.cpp`, `FMNMX`:
/// `Imm32(0x7fffffffU)`).
pub const FMNMX_NAN: u32 = 0x7fff_ffff;
/// The bits of `2 pi` as a 32-bit float, by which `MUFU.SIN` and `MUFU.COS` scale their
/// argument from turns to radians (module docs).
pub const TURN: u32 = std::f32::consts::TAU.to_bits();
/// The shift that moves a half into the high half of a word (`ops_convert.cpp`, `F2FP`:
/// `ShiftLeftLogical(high, Imm32(16U))`).
pub const HALF_SHIFT: u32 = 16;
/// The sign bit of a 32-bit float.
pub const F32_SIGN: u32 = 0x8000_0000;
/// The sign bit of a 16-bit float (IEEE 754 binary16).
pub const F16_SIGN: u32 = 0x8000;
/// The exponent field of a 16-bit float (IEEE 754 binary16): zero for a zero or a denormal.
pub const F16_EXPONENT: u32 = 0x7c00;
/// The operation bytes `FSWZADD` accepts, with the lanes (lane 0 first, NAK's order) each
/// encodes: PR #4588 accepts only these (`ops_float.cpp`, `FSWZADD`: `subop == 0x5a || subop ==
/// 0x66`). The table's pairing of lanes with bytes is checked against the test assembler by
/// `tests::swizzled_add_operations_match_the_assembler`.
pub const FSWZADD_OPERATIONS: [([FSwzAddOp; 4], u32); 2] = [
    (
        [
            FSwzAddOp::SubLeft,
            FSwzAddOp::SubLeft,
            FSwzAddOp::SubRight,
            FSwzAddOp::SubRight,
        ],
        0x5a,
    ),
    (
        [
            FSwzAddOp::SubLeft,
            FSwzAddOp::SubRight,
            FSwzAddOp::SubLeft,
            FSwzAddOp::SubRight,
        ],
        0x66,
    ),
];
/// The bits of an `FSWZADD` operation byte: the PR passes `~subop & 0xffU`.
pub const FSWZADD_BYTE: u32 = 0xff;

/// A form or field value of this group that is not translated: each a `Require` of PR #4588's
/// translators (named in each variant) or a form its `Visitor::Run` sends to no translator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum FloatUnsupported {
    /// An instruction of another group, which this module does not translate.
    #[error("{0}, which is not a float or conversion instruction,")]
    OtherGroup(&'static str),
    /// A constant-buffer source whose bank is in a uniform register (`visitor.cpp`,
    /// `WideSlot`: "native bindless constant buffer", `Take(91, 1) == 0`).
    #[error("a bindless constant-buffer source")]
    BindlessConstantBuffer,
    /// A constant-buffer bank past the last ([`CBUF_BANKS`]; `Visitor::Cbuf`: "native constant
    /// buffer address"). A decoded bank has five bits, so only a hand-made operation reaches it.
    #[error("constant buffer {0}")]
    ConstantBufferBank(u8),
    /// The 64-bit opcode of `FRND`, `F2F`, `F2I` or `I2F` (`0x110` to `0x113`), which
    /// `Visitor::Run` sends to no translator ("unsupported native opcode").
    #[error("the 64-bit form of {0}")]
    WideConversion(&'static str),
    /// `MUFU`'s immediate form (`0x908`), which `Visitor::Run` sends to no translator (it takes
    /// `0x308`, `0xb08` and `0xd08`).
    #[error("an immediate mufu operand")]
    MufuImmediate,
    /// `FFMA`'s multiply-by-zero mode (`ops_alu.cpp`: "native FFMA multiply-zero mode").
    #[error("ffma's multiply-by-zero mode")]
    FfmaMultiplyZero,
    /// `FMUL`'s multiply-by-zero mode (`ops_float.cpp`: "native FMUL multiply-zero mode").
    #[error("fmul's multiply-by-zero mode")]
    FmulMultiplyZero,
    /// An `FMUL` scale outside 1 to 7 (`ops_float.cpp`: "native FMUL scale", `scale >= 1 &&
    /// scale <= 7`): only scale 0.
    #[error("fmul scale {0}")]
    FmulScale(FMulScale),
    /// A scaled `FMUL` without `.FTZ` or with a rounding other than to nearest
    /// (`ops_float.cpp`: "native FMUL scale with non-FTZ or non-RN modifiers").
    #[error("a scaled fmul without ftz or with a rounding mode")]
    FmulScaleModifiers,
    /// A `MUFU` function the PR does not translate (`ops_float.cpp`: "native MUFU operation";
    /// it handles 0 to 5 and 8).
    #[error("mufu function {0}")]
    MufuFunction(MuFuOp),
    /// A 16-bit `MUFU` with an operand other than a register (`ops_float.cpp`: "native half
    /// special-function source").
    #[error("a 16-bit mufu operand that is not a register")]
    HalfMufuSource,
    /// `FSWZADD` with `.NDV`, a rounding mode or `.FTZ` (`ops_float.cpp`: "native swizzled add
    /// mode", `Take(77, 4) == 0`).
    #[error("a swizzled add with ndv, a rounding mode or ftz")]
    SwizzledAddMode,
    /// An `FSWZADD` operation byte other than the PR's two ([`FSWZADD_OPERATIONS`];
    /// `ops_float.cpp`: "native swizzled add operation").
    #[error("swizzled add operations {}", SwizzleOps(*.0))]
    SwizzledAddOperation([FSwzAddOp; 4]),
    /// An `F2I` destination wider than 32 bits (`ops_convert.cpp`: "native float-to-integer
    /// destination size", `dst_size <= 2`).
    #[error("an f2i destination of {}", Bits(.0.width()))]
    F2iDestinationSize(IntType),
    /// An `F2I` source other than 16 or 32 bits (`ops_convert.cpp`: "native float-to-integer
    /// source size", `src_size == 2 || src_size == 1`).
    #[error("an f2i source of {}", Bits(.0.width()))]
    F2iSourceSize(FloatType),
    /// A 16-bit `F2I` operand that is not a register (`ops_convert.cpp`: "native
    /// half-precision float-to-integer source").
    #[error("a 16-bit f2i operand that is not a register")]
    HalfF2iSource,
    /// An `I2F` destination other than 32 bits (`ops_convert.cpp`: "native integer-to-float
    /// destination size", `dst_size == 2`).
    #[error("an i2f destination of {}", Bits(.0.width()))]
    I2fDestinationSize(FloatType),
    /// An `I2F` source wider than 32 bits (`ops_convert.cpp`: "native integer-to-float source
    /// size", `src_size <= 2`).
    #[error("an i2f source of {}", Bits(.0.width()))]
    I2fSourceSize(IntType),
    /// An `I2F` byte or half select the source does not have (`ops_convert.cpp`: "native
    /// integer-to-float source select", `select < (32 >> (3 + src_size))` and zero for a full
    /// word). The decoder never produces one; a hand-made operation can.
    #[error("select {select} of an i2f source of {}", Bits(*.width))]
    I2fSelect {
        /// The select.
        select: u8,
        /// The source's width in bits.
        width: u32,
    },
    /// An `FRND` destination other than 32 bits (`ops_convert.cpp`: "native float round
    /// destination size").
    #[error("an frnd destination of {}", Bits(.0.width()))]
    FrndDestinationSize(FloatType),
    /// An `FRND` source other than 32 bits (`ops_convert.cpp`: "native float round source
    /// size").
    #[error("an frnd source of {}", Bits(.0.width()))]
    FrndSourceSize(FloatType),
    /// An `F2F` destination other than 16 bits (`ops_convert.cpp`: "native float conversion
    /// destination size").
    #[error("an f2f destination of {}", Bits(.0.width()))]
    F2fDestinationSize(FloatType),
    /// An `F2F` source other than 32 bits (`ops_convert.cpp`: "native float conversion source
    /// size").
    #[error("an f2f source of {}", Bits(.0.width()))]
    F2fSourceSize(FloatType),
    /// `F2FP.H1` (`ops_convert.cpp`: "native float packing selector", `Take(72, 1) == 0`).
    #[error("f2fp's high-half selector")]
    F2fpSelector,
    /// An `F2FP` rounding other than to nearest (`ops_convert.cpp`: "native float packing
    /// rounding", `Take(79, 2) == 0`).
    #[error("f2fp rounding {}", RoundName(*.0))]
    F2fpRounding(RoundMode),
}

/// A bit count in words, for error messages.
struct Bits(u32);

impl Display for Bits {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{} bits", self.0)
    }
}

/// A rounding mode in words, for error messages.
struct RoundName(RoundMode);

impl Display for RoundName {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(match self.0 {
            RoundMode::NearestEven => "to nearest",
            RoundMode::NegInf => "down",
            RoundMode::PosInf => "up",
            RoundMode::Zero => "toward zero",
        })
    }
}

/// `FSWZADD`'s lane operations, lane 0 first, for error messages.
struct SwizzleOps([FSwzAddOp; 4]);

impl Display for SwizzleOps {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let [lane0, lane1, lane2, lane3] = self.0;
        write!(f, "{lane0} {lane1} {lane2} {lane3}")
    }
}

/// Why [`translate_instr`] did not translate an instruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum FloatError {
    /// The instruction's form is not translated; nothing was appended.
    #[error("{0} is not supported")]
    Unsupported(FloatUnsupported),
    /// The IR could not be built: an arena is full, or this module built a malformed
    /// instruction (a bug). Instructions appended before the failure stay in the function.
    #[error(transparent)]
    Build(#[from] BuildError),
}

impl From<FloatUnsupported> for FloatError {
    fn from(what: FloatUnsupported) -> Self {
        Self::Unsupported(what)
    }
}

/// Whether `op` is one of this group's instructions.
pub const fn in_group(op: &Op) -> bool {
    matches!(
        op,
        Op::FAdd(_)
            | Op::FMul(_)
            | Op::FFma(_)
            | Op::FMnMx(_)
            | Op::FSet(_)
            | Op::FSetP(_)
            | Op::FSel(_)
            | Op::MuFu(_)
            | Op::FSwzAdd(_)
            | Op::FRnd(_)
            | Op::F2F(_)
            | Op::F2I(_)
            | Op::I2F(_)
            | Op::I2FP(_)
            | Op::F2FP(_)
    )
}

/// Translates `instr`, one of this group's instructions, at the end of `builder`'s block, and
/// leaves `builder` in the block where translation continues: the same block, or for a guarded
/// instruction the block after its conditional block (module docs). `guard_tested` says the
/// block already runs only where the guard holds (flow analysis's `guard_tested`), so the guard
/// is ignored. A refusal appends nothing.
pub fn translate_instr(
    builder: &mut Builder<'_>,
    instr: Instr,
    guard_tested: bool,
) -> Result<(), FloatError> {
    check(&instr.op)?;
    if guard_tested || instr.guard.is_true() {
        return Emit { b: builder }.op(&instr.op);
    }
    let cond = read_pred(builder, instr.guard)?;
    let then = builder.add_block()?;
    let after = builder.add_block()?;
    builder.terminate(Terminator::Branch {
        cond,
        then,
        otherwise: after,
    })?;
    builder.switch_to(then)?;
    Emit { b: builder }.op(&instr.op)?;
    builder.terminate(Terminator::Jump(after))?;
    builder.switch_to(after)?;
    Ok(())
}

/// Whether `function` uses a quad operation of this group (`FSwizzleAdd`, from `FSWZADD`), so
/// that the host must run it in 32-wide subgroups (spec, "Compute shaders"). It reads only this
/// group's opcode: other groups' warp operations are theirs to report.
pub fn needs_subgroup_size_32(function: &Function) -> bool {
    function
        .listed()
        .any(|(_, inst)| matches!(inst, Inst::Op(op) if op.opcode == Opcode::FSwizzleAdd))
}

// ---------------------------------------------------------------------------------------------
// Checks: every refusal, in the PR's order, before anything is appended.

fn require(ok: bool, what: FloatUnsupported) -> Result<(), FloatUnsupported> {
    if ok { Ok(()) } else { Err(what) }
}

/// The source-slot check (`WideSlot`): a bindless constant buffer is refused.
fn check_slot(src: AluSrc) -> Result<(), FloatUnsupported> {
    match src {
        AluSrc::CBuf {
            cbuf:
                CBufRef {
                    buf: CBuf::BindlessUniform(_),
                    ..
                },
            ..
        } => Err(FloatUnsupported::BindlessConstantBuffer),
        _ => Ok(()),
    }
}

/// The value check (`Visitor::Cbuf`): a bank past the last is refused.
fn check_bank(src: AluSrc) -> Result<(), FloatUnsupported> {
    match src {
        AluSrc::CBuf {
            cbuf:
                CBufRef {
                    buf: CBuf::Binding(bank),
                    ..
                },
            ..
        } if bank >= CBUF_BANKS => Err(FloatUnsupported::ConstantBufferBank(bank)),
        _ => Ok(()),
    }
}

/// Slot checks for every source, then bank checks for every source (the PR reads all
/// sources' slots before the instruction's own checks and their values after).
fn check_sources(srcs: &[AluSrc]) -> Result<(), FloatUnsupported> {
    srcs.iter().try_for_each(|&src| check_slot(src))
}

fn check_banks(srcs: &[AluSrc]) -> Result<(), FloatUnsupported> {
    srcs.iter().try_for_each(|&src| check_bank(src))
}

/// The function a `MUFU` operation computes, or its refusal.
const fn mufu_function(op: MuFuOp) -> Result<Opcode, FloatUnsupported> {
    Ok(match op {
        MuFuOp::Cos => Opcode::FPCos,
        MuFuOp::Sin => Opcode::FPSin,
        MuFuOp::Exp2 => Opcode::FPExp2,
        MuFuOp::Log2 => Opcode::FPLog2,
        MuFuOp::Rcp => Opcode::FPRecip32,
        MuFuOp::Rsq => Opcode::FPRecipSqrt32,
        MuFuOp::Sqrt => Opcode::FPSqrt,
        MuFuOp::Rcp64H | MuFuOp::Rsq64H | MuFuOp::Tanh => {
            return Err(FloatUnsupported::MufuFunction(op));
        }
    })
}

/// The Maxwell swizzle `FSWZADD`'s lanes are passed as, or its refusal.
fn swizzle_of(ops: [FSwzAddOp; 4]) -> Result<u32, FloatUnsupported> {
    FSWZADD_OPERATIONS
        .iter()
        .find(|(lanes, _)| *lanes == ops)
        .map(|&(_, byte)| !byte & FSWZADD_BYTE)
        .ok_or(FloatUnsupported::SwizzledAddOperation(ops))
}

/// The bits `I2F` extracts from its source: `None` for a full word.
fn i2f_field(src_type: IntType, select: u8) -> Result<Option<(u32, u32)>, FloatUnsupported> {
    let width = src_type.width();
    let refuse = FloatUnsupported::I2fSelect { select, width };
    if width >= u32::BITS {
        require(select == 0, refuse)?;
        return Ok(None);
    }
    let fields = u32::BITS / width;
    require(u32::from(select) < fields, refuse)?;
    Ok(Some((u32::from(select) * width, width)))
}

/// Every refusal of `op`, in the PR's order.
fn check(op: &Op) -> Result<(), FloatUnsupported> {
    match *op {
        Op::FAdd(op) => {
            check_sources(&op.srcs)?;
            check_banks(&op.srcs)
        }
        Op::FMul(op) => {
            check_sources(&op.srcs)?;
            require(!op.dnz, FloatUnsupported::FmulMultiplyZero)?;
            require(
                op.scale != FMulScale::D16,
                FloatUnsupported::FmulScale(op.scale),
            )?;
            require(
                op.scale == FMulScale::None || (op.ftz && op.rnd == RoundMode::NearestEven),
                FloatUnsupported::FmulScaleModifiers,
            )?;
            check_banks(&op.srcs)
        }
        Op::FFma(op) => {
            check_sources(&op.srcs)?;
            require(!op.dnz, FloatUnsupported::FfmaMultiplyZero)?;
            check_banks(&op.srcs)
        }
        Op::FMnMx(OpFMnMx { srcs, .. })
        | Op::FSet(OpFSet { srcs, .. })
        | Op::FSetP(OpFSetP { srcs, .. })
        | Op::FSel(OpFSel { srcs, .. }) => {
            check_sources(&srcs)?;
            check_banks(&srcs)
        }
        Op::MuFu(op) => check_mufu(op),
        Op::FSwzAdd(op) => {
            require(
                !op.ndv && op.rnd == RoundMode::NearestEven && !op.ftz,
                FloatUnsupported::SwizzledAddMode,
            )?;
            swizzle_of(op.ops).map(|_| ())
        }
        Op::F2I(op) => check_f2i(op),
        Op::I2F(op) => check_i2f(op.wide, op.src, op.dst_type, op.src_type, op.select, "i2f"),
        Op::I2FP(op) => check_i2f(false, op.src, op.dst_type, op.src_type, 0, "i2fp"),
        Op::FRnd(op) => check_frnd(op),
        Op::F2F(op) => check_f2f(op),
        Op::F2FP(op) => {
            check_sources(&op.srcs)?;
            require(!op.h1, FloatUnsupported::F2fpSelector)?;
            require(
                op.rnd == RoundMode::NearestEven,
                FloatUnsupported::F2fpRounding(op.rnd),
            )?;
            check_banks(&op.srcs)
        }
        other => Err(FloatUnsupported::OtherGroup(other.mnemonic())),
    }
}

/// `MUFU`'s checks, in the PR's order: the form, the slot, the 16-bit operand, the bank (read
/// before the function is chosen), the function.
fn check_mufu(op: OpMuFu) -> Result<(), FloatUnsupported> {
    require(
        !matches!(op.src, AluSrc::Imm(_)),
        FloatUnsupported::MufuImmediate,
    )?;
    check_slot(op.src)?;
    if op.f16 {
        require(
            matches!(op.src, AluSrc::Reg { .. }),
            FloatUnsupported::HalfMufuSource,
        )?;
    } else {
        check_bank(op.src)?;
    }
    mufu_function(op.op).map(|_| ())
}

/// `F2I`'s checks, in the PR's order.
fn check_f2i(op: OpF2I) -> Result<(), FloatUnsupported> {
    require(!op.wide, FloatUnsupported::WideConversion("f2i"))?;
    check_slot(op.src)?;
    require(
        op.dst_type.width() <= u32::BITS,
        FloatUnsupported::F2iDestinationSize(op.dst_type),
    )?;
    require(
        matches!(op.src_type, FloatType::F16 | FloatType::F32),
        FloatUnsupported::F2iSourceSize(op.src_type),
    )?;
    if op.src_type == FloatType::F16 {
        require(
            matches!(op.src, AluSrc::Reg { .. }),
            FloatUnsupported::HalfF2iSource,
        )
    } else {
        check_bank(op.src)
    }
}

/// `FRND`'s checks, in the PR's order.
fn check_frnd(op: OpFRnd) -> Result<(), FloatUnsupported> {
    require(!op.wide, FloatUnsupported::WideConversion("frnd"))?;
    check_slot(op.src)?;
    require(
        op.dst_type == FloatType::F32,
        FloatUnsupported::FrndDestinationSize(op.dst_type),
    )?;
    require(
        op.src_type == FloatType::F32,
        FloatUnsupported::FrndSourceSize(op.src_type),
    )?;
    check_bank(op.src)
}

/// `F2F`'s checks, in the PR's order.
fn check_f2f(op: OpF2F) -> Result<(), FloatUnsupported> {
    require(!op.wide, FloatUnsupported::WideConversion("f2f"))?;
    check_slot(op.src)?;
    require(
        op.dst_type == FloatType::F16,
        FloatUnsupported::F2fDestinationSize(op.dst_type),
    )?;
    require(
        op.src_type == FloatType::F32,
        FloatUnsupported::F2fSourceSize(op.src_type),
    )?;
    check_bank(op.src)
}

/// `I2F` and `I2FP`'s checks, in the PR's order (`Visitor::I2F` serves both).
fn check_i2f(
    wide: bool,
    src: AluSrc,
    dst_type: FloatType,
    src_type: IntType,
    select: u8,
    name: &'static str,
) -> Result<(), FloatUnsupported> {
    require(!wide, FloatUnsupported::WideConversion(name))?;
    check_slot(src)?;
    require(
        dst_type == FloatType::F32,
        FloatUnsupported::I2fDestinationSize(dst_type),
    )?;
    require(
        src_type.width() <= u32::BITS,
        FloatUnsupported::I2fSourceSize(src_type),
    )?;
    i2f_field(src_type, select)?;
    check_bank(src)
}

// ---------------------------------------------------------------------------------------------
// Emission.

/// PR #4588's `RoundingOf` (`ops_alu.cpp`, `ops_float.cpp`, `ops_convert.cpp`): the 2-bit
/// rounding field as an IR rounding mode.
pub(crate) const fn rounding_of(mode: RoundMode) -> FpRounding {
    match mode {
        RoundMode::NearestEven => FpRounding::Rn,
        RoundMode::NegInf => FpRounding::Rm,
        RoundMode::PosInf => FpRounding::Rp,
        RoundMode::Zero => FpRounding::Rz,
    }
}

/// `FTZ` or no flushing, as the PR's controls say it.
const fn flushing(ftz: bool) -> FmzMode {
    if ftz { FmzMode::Ftz } else { FmzMode::None }
}

/// Floating-point controls.
const fn control(no_contraction: bool, rounding: FpRounding, fmz_mode: FmzMode) -> FpControl {
    FpControl {
        no_contraction,
        rounding,
        fmz_mode,
    }
}

/// The controls of an operation whose rounding and flushing are fixed: to nearest, no
/// flushing (the PR's controls for `MUFU`'s conversions, `FSWZADD` and `F2FP`).
const NEAREST: FpControl = control(false, FpRounding::Rn, FmzMode::None);

/// The operation `FRND` and `F2I` round with (`ops_convert.cpp`: `FPRoundEven`, `FPFloor`,
/// `FPCeil`, `FPTrunc` by the rounding field), in the 16- or 32-bit format.
const fn round_op(mode: RoundMode, half: bool) -> Opcode {
    match (mode, half) {
        (RoundMode::NearestEven, false) => Opcode::FPRoundEven32,
        (RoundMode::NegInf, false) => Opcode::FPFloor32,
        (RoundMode::PosInf, false) => Opcode::FPCeil32,
        (RoundMode::Zero, false) => Opcode::FPTrunc32,
        (RoundMode::NearestEven, true) => Opcode::FPRoundEven16,
        (RoundMode::NegInf, true) => Opcode::FPFloor16,
        (RoundMode::PosInf, true) => Opcode::FPCeil16,
        (RoundMode::Zero, true) => Opcode::FPTrunc16,
    }
}

/// A float comparison's opcode, ordered and unordered (Eden's `FloatingPointCompare`, the
/// `.u` forms unordered); `None` for the four that are not one comparison.
const fn compare_op(cmp: FloatCmpOp) -> Option<Opcode> {
    Some(match cmp {
        FloatCmpOp::Lt => Opcode::FPOrdLessThan32,
        FloatCmpOp::Eq => Opcode::FPOrdEqual32,
        FloatCmpOp::Le => Opcode::FPOrdLessThanEqual32,
        FloatCmpOp::Gt => Opcode::FPOrdGreaterThan32,
        FloatCmpOp::Ne => Opcode::FPOrdNotEqual32,
        FloatCmpOp::Ge => Opcode::FPOrdGreaterThanEqual32,
        FloatCmpOp::LtU => Opcode::FPUnordLessThan32,
        FloatCmpOp::EqU => Opcode::FPUnordEqual32,
        FloatCmpOp::LeU => Opcode::FPUnordLessThanEqual32,
        FloatCmpOp::GtU => Opcode::FPUnordGreaterThan32,
        FloatCmpOp::NeU => Opcode::FPUnordNotEqual32,
        FloatCmpOp::GeU => Opcode::FPUnordGreaterThanEqual32,
        FloatCmpOp::False | FloatCmpOp::True | FloatCmpOp::Num | FloatCmpOp::Nan => return None,
    })
}

/// Eden's `PredicateCombine`.
const fn combine_op(set_op: PredSetOp) -> Opcode {
    match set_op {
        PredSetOp::And => Opcode::LogicalAnd,
        PredSetOp::Or => Opcode::LogicalOr,
        PredSetOp::Xor => Opcode::LogicalXor,
    }
}

/// The range `F2I` clamps to before converting, as the PR computes it in `f64` and narrows to
/// `f32` (`static_cast<f32>`), as float bits: `(min, max)`.
fn f2i_bounds(dst_type: IntType) -> (u32, u32) {
    match (dst_type.width(), dst_type.is_signed()) {
        (8, false) => (0, f32::from(u8::MAX).to_bits()),
        (8, true) => (f32::from(i8::MIN).to_bits(), f32::from(i8::MAX).to_bits()),
        (16, false) => (0, f32::from(u16::MAX).to_bits()),
        (16, true) => (f32::from(i16::MIN).to_bits(), f32::from(i16::MAX).to_bits()),
        // 2^32 - 1 and 2^31 - 1 round to 2^32 and 2^31 in `f32`; -2^31 is exact.
        (_, false) => (0, U32_MAX_AS_F32),
        (_, true) => (I32_MIN_AS_F32, I32_MAX_AS_F32),
    }
}

/// `u32::MAX` narrowed to `f32`: `2^32`.
pub const U32_MAX_AS_F32: u32 = 0x4f80_0000;
/// `i32::MAX` narrowed to `f32`: `2^31`.
pub const I32_MAX_AS_F32: u32 = 0x4f00_0000;
/// `i32::MIN` as `f32`: `-2^31`.
pub const I32_MIN_AS_F32: u32 = 0xcf00_0000;

/// A 32-bit immediate.
const fn imm32(bits: u32) -> Operand {
    Operand::Imm(Imm::U32(bits))
}

/// A 32-bit float immediate, as its bits.
const fn f32_imm(bits: u32) -> Operand {
    Operand::Imm(Imm::F32(bits))
}

/// Appends this group's instructions to one builder.
struct Emit<'b, 'f> {
    b: &'b mut Builder<'f>,
}

impl Emit<'_, '_> {
    fn op(&mut self, op: &Op) -> Result<(), FloatError> {
        match *op {
            Op::FAdd(op) => self.fadd(op),
            Op::FMul(op) => self.fmul(op),
            Op::FFma(op) => self.ffma(op),
            Op::FMnMx(op) => self.fmnmx(op),
            Op::FSet(op) => self.fset(op),
            Op::FSetP(op) => self.fsetp(op),
            Op::FSel(op) => self.fsel(op),
            Op::MuFu(op) => self.mufu(op),
            Op::FSwzAdd(op) => self.fswzadd(op),
            Op::FRnd(op) => self.frnd(op),
            Op::F2F(op) => self.f2f(op),
            Op::F2I(op) => self.f2i(op),
            Op::I2F(op) => self.i2f(op.src, op.src_type, op.select, op.rnd, op.dst),
            Op::I2FP(op) => self.i2fp(op),
            Op::F2FP(op) => self.f2fp(op),
            other => Err(FloatUnsupported::OtherGroup(other.mnemonic()).into()),
        }
    }

    fn emit(&mut self, opcode: Opcode, args: &[Operand]) -> Result<Operand, FloatError> {
        Ok(self.b.op(opcode, args)?.into())
    }

    fn emit_fp(
        &mut self,
        opcode: Opcode,
        args: &[Operand],
        control: FpControl,
    ) -> Result<Operand, FloatError> {
        Ok(self.b.op_with(opcode, args, Flags::Fp(control))?.into())
    }

    /// A register's raw bits; `RZ` reads zero (`Visitor::X`).
    fn get(&mut self, reg: Gpr) -> Result<Operand, FloatError> {
        if reg.is_zero() {
            return Ok(imm32(0));
        }
        self.emit(Opcode::GetRegister, &[Operand::Imm(Imm::Reg(reg))])
    }

    /// Writes a register's raw bits; a write to `RZ` is dropped (`Visitor::X`).
    fn set(&mut self, reg: Gpr, value: Operand) -> Result<(), FloatError> {
        if !reg.is_zero() {
            self.emit(Opcode::SetRegister, &[Operand::Imm(Imm::Reg(reg)), value])?;
        }
        Ok(())
    }

    fn set_f32(&mut self, reg: Gpr, value: Operand) -> Result<(), FloatError> {
        let bits = self.emit(Opcode::BitCastU32F32, &[value])?;
        self.set(reg, bits)
    }

    fn get_f32(&mut self, reg: Gpr) -> Result<Operand, FloatError> {
        let bits = self.get(reg)?;
        self.emit(Opcode::BitCastF32U32, &[bits])
    }

    /// A constant-buffer word at a byte offset (`Visitor::Cbuf`).
    fn cbuf(&mut self, bank: u8, offset: u16) -> Result<Operand, FloatError> {
        let bank = imm32(u32::from(bank));
        let skew = offset % CBUF_WORD_BYTES;
        let aligned = u32::from(offset - skew);
        if skew == 0 {
            return self.emit(Opcode::GetCbufU32, &[bank, imm32(aligned)]);
        }
        let skew_bits = u32::from(skew) * u8::BITS;
        let next = aligned + u32::from(CBUF_WORD_BYTES);
        let low = self.emit(Opcode::GetCbufU32, &[bank, imm32(aligned)])?;
        let high = self.emit(Opcode::GetCbufU32, &[bank, imm32(next)])?;
        let low = self.emit(Opcode::ShiftRightLogical32, &[low, imm32(skew_bits)])?;
        let high = self.emit(
            Opcode::ShiftLeftLogical32,
            &[high, imm32(u32::BITS - skew_bits)],
        )?;
        self.emit(Opcode::BitwiseOr32, &[low, high])
    }

    /// A source's raw bits, without modifiers (`Visitor::Value`).
    fn value(&mut self, src: AluSrc) -> Result<Operand, FloatError> {
        match src {
            AluSrc::Reg { reg, .. } => self.get(reg),
            AluSrc::UniformReg { reg, .. } if reg.is_zero() => Ok(imm32(0)),
            AluSrc::UniformReg { reg, .. } => self.emit(
                Opcode::GetUniformRegister,
                &[Operand::Imm(Imm::UniformReg(reg))],
            ),
            AluSrc::Imm(bits) => Ok(imm32(bits)),
            AluSrc::CBuf { cbuf, .. } => match cbuf.buf {
                CBuf::Binding(bank) => self.cbuf(bank, cbuf.offset),
                CBuf::BindlessUniform(_) => Err(FloatUnsupported::BindlessConstantBuffer.into()),
            },
        }
    }

    /// Takes the absolute value, then negates (Eden's `FPAbsNeg`).
    fn abs_neg(
        &mut self,
        mut value: Operand,
        abs: bool,
        neg: bool,
        half: bool,
    ) -> Result<Operand, FloatError> {
        let (abs_op, neg_op) = if half {
            (Opcode::FPAbs16, Opcode::FPNeg16)
        } else {
            (Opcode::FPAbs32, Opcode::FPNeg32)
        };
        if abs {
            value = self.emit(abs_op, &[value])?;
        }
        if neg {
            value = self.emit(neg_op, &[value])?;
        }
        Ok(value)
    }

    /// A float source with its modifiers (`Visitor::FloatValue`).
    fn float_value(&mut self, src: AluSrc) -> Result<Operand, FloatError> {
        let bits = self.value(src)?;
        let value = self.emit(Opcode::BitCastF32U32, &[bits])?;
        let (abs, neg) = src.modifiers();
        self.abs_neg(value, abs, neg, false)
    }

    /// Half `high` (or the low half) of a register, with modifiers (`Maxwell::Extract` with
    /// `Swizzle::H1_H0`: component 0 is the low half).
    fn half_of(&mut self, src: AluSrc, high: bool) -> Result<Operand, FloatError> {
        let AluSrc::Reg { reg, abs, neg } = src else {
            // `check` refuses every other operand for the 16-bit forms.
            return Err(FloatUnsupported::HalfF2iSource.into());
        };
        let bits = self.get(reg)?;
        let pair = self.emit(Opcode::UnpackFloat2x16, &[bits])?;
        let half = self.emit(
            Opcode::CompositeExtractF16x2,
            &[pair, imm32(u32::from(high))],
        )?;
        self.abs_neg(half, abs, neg, true)
    }

    /// `FPSaturate32` when `saturate` is set.
    fn saturate(&mut self, value: Operand, saturate: bool) -> Result<Operand, FloatError> {
        if saturate {
            self.emit(Opcode::FPSaturate32, &[value])
        } else {
            Ok(value)
        }
    }

    /// A predicate's value (`Visitor::Pred`): `PT` is an immediate.
    fn pred(&mut self, src: PredSrc) -> Result<Operand, FloatError> {
        Ok(read_pred(self.b, src)?)
    }

    /// Writes a predicate; a write to `PT` is dropped (`Visitor::SetPred`).
    fn set_pred(&mut self, dst: Pred, value: Operand) -> Result<(), FloatError> {
        if !dst.is_true() {
            self.emit(Opcode::SetPred, &[Operand::Imm(Imm::Pred(dst)), value])?;
        }
        Ok(())
    }

    /// The zero half the PR packs into a word's high half (`FPConvert(16, Imm32(0.0f))`).
    fn zero_half(&mut self) -> Result<Operand, FloatError> {
        self.emit(Opcode::ConvertF16F32, &[f32_imm(0)])
    }

    /// A half in the low half of a word, zero above (`PackFloat2x16` of the half and a zero).
    fn pack_low(&mut self, half: Operand, zero: Operand) -> Result<Operand, FloatError> {
        let pair = self.emit(Opcode::CompositeConstructF16x2, &[half, zero])?;
        self.emit(Opcode::PackFloat2x16, &[pair])
    }

    /// Replaces a 32-bit denormal (and a zero) with a zero of its sign; a NaN and every normal
    /// value are kept. The emulation of flush-to-zero on an operand (module docs).
    fn flush32(&mut self, value: Operand) -> Result<Operand, FloatError> {
        let magnitude = self.emit(Opcode::FPAbs32, &[value])?;
        let tiny = self.emit_fp(
            Opcode::FPOrdLessThan32,
            &[magnitude, f32_imm(f32::MIN_POSITIVE.to_bits())],
            control(false, FpRounding::DontCare, FmzMode::None),
        )?;
        let bits = self.emit(Opcode::BitCastU32F32, &[value])?;
        let sign = self.emit(Opcode::BitwiseAnd32, &[bits, imm32(F32_SIGN)])?;
        let zero = self.emit(Opcode::BitCastF32U32, &[sign])?;
        self.emit(Opcode::SelectF32, &[tiny, zero, value])
    }

    /// `FADD` (`ops_alu.cpp`, `FADD`).
    fn fadd(&mut self, op: OpFAdd) -> Result<(), FloatError> {
        let [a, b] = op.srcs;
        let a = self.float_value(a)?;
        let b = self.float_value(b)?;
        let value = self.emit_fp(
            Opcode::FPAdd32,
            &[a, b],
            control(true, rounding_of(op.rnd), flushing(op.ftz)),
        )?;
        let value = self.saturate(value, op.saturate)?;
        self.set_f32(op.dst, value)
    }

    /// `FMUL` (`ops_float.cpp`, `FMUL`): the scale multiplies the first operand with Eden's
    /// default controls (`ir.FPMul(op_a, ir.Imm32(ScaleFactor(scale)))`).
    fn fmul(&mut self, op: OpFMul) -> Result<(), FloatError> {
        let [a, b] = op.srcs;
        let mut a = self.float_value(a)?;
        if op.scale != FMulScale::None {
            let factor = 2.0_f32.powi(i32::from(op.scale.exponent()));
            a = self.emit_fp(
                Opcode::FPMul32,
                &[a, f32_imm(factor.to_bits())],
                FpControl::default(),
            )?;
        }
        let b = self.float_value(b)?;
        let value = self.emit_fp(
            Opcode::FPMul32,
            &[a, b],
            control(true, rounding_of(op.rnd), flushing(op.ftz)),
        )?;
        let value = self.saturate(value, op.saturate)?;
        self.set_f32(op.dst, value)
    }

    /// `FFMA` (`ops_alu.cpp`, `FFMA`).
    fn ffma(&mut self, op: OpFFma) -> Result<(), FloatError> {
        let [a, b, c] = op.srcs;
        let a = self.float_value(a)?;
        let b = self.float_value(b)?;
        let c = self.float_value(c)?;
        let value = self.emit_fp(
            Opcode::FPFma32,
            &[a, b, c],
            control(true, rounding_of(op.rnd), flushing(op.ftz)),
        )?;
        let value = self.saturate(value, op.saturate)?;
        self.set_f32(op.dst, value)
    }

    /// `FMNMX` (`ops_alu.cpp`, `FMNMX`).
    fn fmnmx(&mut self, op: OpFMnMx) -> Result<(), FloatError> {
        let [a, b] = op.srcs;
        let a = self.float_value(a)?;
        let b = self.float_value(b)?;
        let pred = self.pred(PredSrc {
            reg: op.min.reg,
            negated: false,
        })?;
        let ctl = control(false, FpRounding::DontCare, flushing(op.ftz));
        let max = self.emit_fp(Opcode::FPMax32, &[a, b], ctl)?;
        let min = self.emit_fp(Opcode::FPMin32, &[a, b], ctl)?;
        let (on_true, on_false) = if op.min.negated {
            (max, min)
        } else {
            (min, max)
        };
        let result = self.emit(Opcode::SelectF32, &[pred, on_true, on_false])?;
        if !op.nan {
            return self.set_f32(op.dst, result);
        }
        let a_nan = self.emit(Opcode::FPIsNan32, &[a])?;
        let b_nan = self.emit(Opcode::FPIsNan32, &[b])?;
        let unordered = self.emit(Opcode::LogicalOr, &[a_nan, b_nan])?;
        let bits = self.emit(Opcode::BitCastU32F32, &[result])?;
        let value = self.emit(Opcode::SelectU32, &[unordered, imm32(FMNMX_NAN), bits])?;
        self.set(op.dst, value)
    }

    /// Eden's `FloatingPointCompare` on two 32-bit floats.
    fn compare(
        &mut self,
        cmp: FloatCmpOp,
        a: Operand,
        b: Operand,
        ftz: bool,
    ) -> Result<Operand, FloatError> {
        if let Some(opcode) = compare_op(cmp) {
            let ctl = control(false, FpRounding::DontCare, flushing(ftz));
            return self.emit_fp(opcode, &[a, b], ctl);
        }
        match cmp {
            FloatCmpOp::False => Ok(Operand::Imm(Imm::U1(false))),
            FloatCmpOp::True => Ok(Operand::Imm(Imm::U1(true))),
            FloatCmpOp::Num => {
                let a_nan = self.emit(Opcode::FPIsNan32, &[a])?;
                let b_nan = self.emit(Opcode::FPIsNan32, &[b])?;
                let a_num = self.emit(Opcode::LogicalNot, &[a_nan])?;
                let b_num = self.emit(Opcode::LogicalNot, &[b_nan])?;
                self.emit(Opcode::LogicalAnd, &[a_num, b_num])
            }
            _ => {
                let a_nan = self.emit(Opcode::FPIsNan32, &[a])?;
                let b_nan = self.emit(Opcode::FPIsNan32, &[b])?;
                self.emit(Opcode::LogicalOr, &[a_nan, b_nan])
            }
        }
    }

    /// `FSETP` (`ops_alu.cpp`, `FSETP`).
    fn fsetp(&mut self, op: OpFSetP) -> Result<(), FloatError> {
        let [a, b] = op.srcs;
        let a = self.float_value(a)?;
        let b = self.float_value(b)?;
        let comparison = self.compare(op.cmp, a, b, op.ftz)?;
        let accum = self.pred(op.accum)?;
        let combine = combine_op(op.set_op);
        if !op.dst.is_true() {
            let value = self.emit(combine, &[comparison, accum])?;
            self.set_pred(op.dst, value)?;
        }
        if !op.dst1.is_true() {
            let inverted = self.emit(Opcode::LogicalNot, &[comparison])?;
            let value = self.emit(combine, &[inverted, accum])?;
            self.set_pred(op.dst1, value)?;
        }
        Ok(())
    }

    /// `FSET` (`ops_alu.cpp`, `FSET`).
    fn fset(&mut self, op: OpFSet) -> Result<(), FloatError> {
        let [a, b] = op.srcs;
        let a = self.float_value(a)?;
        let b = self.float_value(b)?;
        let comparison = self.compare(op.cmp, a, b, op.ftz)?;
        let accum = self.pred(op.accum)?;
        let combined = self.emit(combine_op(op.set_op), &[comparison, accum])?;
        let value = self.emit(Opcode::SelectU32, &[combined, imm32(FSET_TRUE), imm32(0)])?;
        self.set(op.dst, value)
    }

    /// `FSEL` (`ops_alu.cpp`, `Visitor::SEL(true)`).
    fn fsel(&mut self, op: OpFSel) -> Result<(), FloatError> {
        let [a, b] = op.srcs;
        let a = self.float_value(a)?;
        let a = self.emit(Opcode::BitCastU32F32, &[a])?;
        let b = self.float_value(b)?;
        let b = self.emit(Opcode::BitCastU32F32, &[b])?;
        let cond = self.pred(op.cond)?;
        let value = self.emit(Opcode::SelectU32, &[cond, a, b])?;
        self.set(op.dst, value)
    }

    /// `MUFU` (`ops_float.cpp`, `MUFU`), with the turns of row 95 (module docs).
    fn mufu(&mut self, op: OpMuFu) -> Result<(), FloatError> {
        let function = mufu_function(op.op)?;
        let input = if op.f16 {
            let half = self.half_of(op.src, op.high)?;
            self.emit_fp(Opcode::ConvertF32F16, &[half], NEAREST)?
        } else {
            self.float_value(op.src)?
        };
        let input = if matches!(op.op, MuFuOp::Sin | MuFuOp::Cos) {
            self.emit_fp(
                Opcode::FPMul32,
                &[input, f32_imm(TURN)],
                control(true, FpRounding::Rn, FmzMode::None),
            )?
        } else {
            input
        };
        let value = self.emit(function, &[input])?;
        if !op.f16 {
            return self.set_f32(op.dst, value);
        }
        let narrowed = self.emit_fp(Opcode::ConvertF16F32, &[value], NEAREST)?;
        let zero = self.zero_half()?;
        let packed = self.pack_low(narrowed, zero)?;
        self.set(op.dst, packed)
    }

    /// `FSWZADD` (`ops_float.cpp`, `FSWZADD`).
    fn fswzadd(&mut self, op: OpFSwzAdd) -> Result<(), FloatError> {
        let swizzle = swizzle_of(op.ops)?;
        let [a, b] = op.srcs;
        let a = self.get_f32(a)?;
        let b = self.get_f32(b)?;
        let value = self.emit_fp(Opcode::FSwizzleAdd, &[a, b, imm32(swizzle)], NEAREST)?;
        self.set_f32(op.dst, value)
    }

    /// `FRND` (`ops_convert.cpp`, `FRND`).
    fn frnd(&mut self, op: OpFRnd) -> Result<(), FloatError> {
        let input = self.float_value(op.src)?;
        let value = self.emit_fp(
            round_op(op.rnd, false),
            &[input],
            control(false, FpRounding::DontCare, flushing(op.ftz)),
        )?;
        self.set_f32(op.dst, value)
    }

    /// `F2I` (`ops_convert.cpp`, `F2I`), with flush-to-zero emulated on a 32-bit operand.
    fn f2i(&mut self, op: OpF2I) -> Result<(), FloatError> {
        let half = op.src_type == FloatType::F16;
        let operand = if half {
            self.half_of(op.src, op.high)?
        } else {
            self.float_value(op.src)?
        };
        // The PR leaves the denormal mode unset for a half operand (`denorm_cares`).
        let fmz_mode = if half {
            FmzMode::DontCare
        } else {
            flushing(op.ftz)
        };
        let input = if !half && op.ftz {
            self.flush32(operand)?
        } else {
            operand
        };
        let rounded = self.emit_fp(
            round_op(op.rnd, half),
            &[input],
            control(true, FpRounding::DontCare, fmz_mode),
        )?;
        let (min, max) = f2i_bounds(op.dst_type);
        let signed = op.dst_type.is_signed();
        let (clamped, convert) = if half {
            let min = self.emit(Opcode::ConvertF16F32, &[f32_imm(min)])?;
            let max = self.emit(Opcode::ConvertF16F32, &[f32_imm(max)])?;
            let clamped = self.emit(Opcode::FPClamp16, &[rounded, min, max])?;
            let convert = if signed {
                Opcode::ConvertS32F16
            } else {
                Opcode::ConvertU32F16
            };
            (clamped, convert)
        } else {
            let clamped = self.emit(Opcode::FPClamp32, &[rounded, f32_imm(min), f32_imm(max)])?;
            let convert = if signed {
                Opcode::ConvertS32F32
            } else {
                Opcode::ConvertU32F32
            };
            (clamped, convert)
        };
        let mut result = self.emit(convert, &[clamped])?;
        if signed {
            let is_nan = if half {
                Opcode::FPIsNan16
            } else {
                Opcode::FPIsNan32
            };
            let nan = self.emit(is_nan, &[operand])?;
            result = self.emit(Opcode::SelectU32, &[nan, imm32(0), result])?;
        }
        self.set(op.dst, result)
    }

    /// `I2F` and `I2FP` (`ops_convert.cpp`, `Visitor::I2F`).
    fn i2f(
        &mut self,
        src: AluSrc,
        src_type: IntType,
        select: u8,
        rnd: RoundMode,
        dst: Gpr,
    ) -> Result<(), FloatError> {
        let field = i2f_field(src_type, select)?;
        let signed = src_type.is_signed();
        let mut value = self.value(src)?;
        if let Some((offset, width)) = field {
            let extract = if signed {
                Opcode::BitFieldSExtract
            } else {
                Opcode::BitFieldUExtract
            };
            value = self.emit(extract, &[value, imm32(offset), imm32(width)])?;
        }
        let convert = if signed {
            Opcode::ConvertF32S32
        } else {
            Opcode::ConvertF32U32
        };
        let result = self.emit_fp(
            convert,
            &[value],
            control(false, rounding_of(rnd), FmzMode::DontCare),
        )?;
        self.set_f32(dst, result)
    }

    fn i2fp(&mut self, op: OpI2FP) -> Result<(), FloatError> {
        self.i2f(op.src, op.src_type, 0, op.rnd, op.dst)
    }

    /// `F2F` from 32 to 16 bits (`ops_convert.cpp`, `F2F`), with its rounding mode and
    /// flush-to-zero emulated (module docs).
    fn f2f(&mut self, op: OpF2F) -> Result<(), FloatError> {
        let mut input = self.float_value(op.src)?;
        if op.ftz {
            input = self.flush32(input)?;
        }
        let half = self.emit_fp(
            Opcode::ConvertF16F32,
            &[input],
            control(false, rounding_of(op.rnd), flushing(op.ftz)),
        )?;
        let zero = self.zero_half()?;
        let mut bits = self.pack_low(half, zero)?;
        if op.rnd != RoundMode::NearestEven {
            bits = self.directed_half(op.rnd, input, half, bits)?;
        }
        if op.ftz {
            let exponent = self.emit(Opcode::BitwiseAnd32, &[bits, imm32(F16_EXPONENT)])?;
            let tiny = self.emit(Opcode::IEqual, &[exponent, imm32(0)])?;
            let sign = self.emit(Opcode::BitwiseAnd32, &[bits, imm32(F16_SIGN)])?;
            bits = self.emit(Opcode::SelectU32, &[tiny, sign, bits])?;
        }
        self.set(op.dst, bits)
    }

    /// Moves a half (`bits`, zero above) one step to the side `mode` rounds to when its wider
    /// value lies on the other side of `input`. A to-nearest half is within one step of the
    /// directed one, and the step is one unit of its bits: up in magnitude away from zero, down
    /// toward it, `0x7c00` (infinity) stepping down to the largest finite half.
    fn directed_half(
        &mut self,
        mode: RoundMode,
        input: Operand,
        half: Operand,
        bits: Operand,
    ) -> Result<Operand, FloatError> {
        let raw = control(false, FpRounding::DontCare, FmzMode::None);
        let wide = self.emit_fp(Opcode::ConvertF32F16, &[half], NEAREST)?;
        let (wrong, step) = match mode {
            RoundMode::Zero => {
                let down = self.emit(Opcode::ISub32, &[bits, imm32(1)])?;
                let wide_abs = self.emit(Opcode::FPAbs32, &[wide])?;
                let input_abs = self.emit(Opcode::FPAbs32, &[input])?;
                let wrong =
                    self.emit_fp(Opcode::FPOrdGreaterThan32, &[wide_abs, input_abs], raw)?;
                (wrong, down)
            }
            RoundMode::NegInf | RoundMode::PosInf => {
                let up = self.emit(Opcode::IAdd32, &[bits, imm32(1)])?;
                let down = self.emit(Opcode::ISub32, &[bits, imm32(1)])?;
                let (compare, positive_step, negative_step) = if mode == RoundMode::NegInf {
                    (Opcode::FPOrdGreaterThan32, down, up)
                } else {
                    (Opcode::FPOrdLessThan32, up, down)
                };
                let wrong = self.emit_fp(compare, &[wide, input], raw)?;
                let sign = self.emit(Opcode::BitwiseAnd32, &[bits, imm32(F16_SIGN)])?;
                let negative = self.emit(Opcode::INotEqual, &[sign, imm32(0)])?;
                let step =
                    self.emit(Opcode::SelectU32, &[negative, negative_step, positive_step])?;
                (wrong, step)
            }
            RoundMode::NearestEven => return Ok(bits),
        };
        self.emit(Opcode::SelectU32, &[wrong, step, bits])
    }

    /// `F2FP.PACK_AB` (`ops_convert.cpp`, `F2FP`).
    fn f2fp(&mut self, op: OpF2FP) -> Result<(), FloatError> {
        let zero = self.zero_half()?;
        let [a, b] = op.srcs;
        let high = if op.merge_c {
            self.get(op.c)?
        } else {
            let bits = self.value(a)?;
            let value = self.emit(Opcode::BitCastF32U32, &[bits])?;
            self.low_half_of(value, zero)?
        };
        let bits = self.value(b)?;
        let value = self.emit(Opcode::BitCastF32U32, &[bits])?;
        let low = self.low_half_of(value, zero)?;
        let shifted = self.emit(Opcode::ShiftLeftLogical32, &[high, imm32(HALF_SHIFT)])?;
        let result = self.emit(Opcode::IAdd32, &[shifted, low])?;
        self.set(op.dst, result)
    }

    /// A float converted to a half in the low half of a word (`F2FP`'s `to_half`).
    fn low_half_of(&mut self, value: Operand, zero: Operand) -> Result<Operand, FloatError> {
        let half = self.emit_fp(Opcode::ConvertF16F32, &[value], NEAREST)?;
        self.pack_low(half, zero)
    }
}
