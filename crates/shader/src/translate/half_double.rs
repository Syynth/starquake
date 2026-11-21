//! Slice 12c: the half- and double-precision instructions decoded by slice 6c (`HADD2`,
//! `HFMA2`, `HMUL2`, `HSET2`, `HSETP2`, `HMNMX2`, `DADD`, `DMUL`, `DFMA`, `DSETP`) translated
//! into IR.
//!
//! # Sources
//!
//! Ported by behaviour from PR #4588's native frontend
//! (`eden-pr4588/src/shader_recompiler/frontend/ampere/` @ e5fb3b97, GPL-3.0-or-later, Copyright
//! 2026 Eden Emulator Project): `ops_half.cpp` (`Visitor::HalfSource`, `HalfLanes`, `HADD2`,
//! `HMUL2`, `HFMA2`, `HMNMX2`, `HSETP2`) and `visitor.cpp` (`Visitor::Cbuf`, `X`, `Pred`,
//! `SetPred`). The PR has no translator for `HSET2` or for the double-precision instructions
//! (its `Visitor::Run` sends them to none), so those follow Eden's Maxwell translators
//! (`eden/src/shader_recompiler/frontend/maxwell/translate/impl/` @ cf654052,
//! GPL-2.0-or-later yuzu code: `half_floating_point_set.cpp` for `HSET2`, which PR #4588's
//! lowering forwards to, `lower_to_maxwell.h` case `0x233`; `double_add.cpp`,
//! `double_multiply.cpp`, `double_fused_multiply_add.cpp`, `double_set_predicate.cpp`; and
//! `impl.cpp`, `TranslatorVisitor::D`, for register pairs), with the SM86 operand layout NAK
//! encodes (`mesa-nouveau/src/nouveau/compiler/nak/opt_copy_prop.rs` @ 52238762, MIT: a 64-bit
//! immediate is its high word over a zero low word, a constant-buffer double the word at the
//! offset and the word four bytes on). The other half translators are cross-checked with
//! Eden's (`half_floating_point_add.cpp`, `half_floating_point_multiply.cpp`,
//! `half_floating_point_fused_multiply_add.cpp`, `half_floating_point_set_predicate.cpp`,
//! `half_floating_point_helper.cpp`), and the comparisons and predicate combines with
//! `common_funcs.cpp`, which both reuse.
//!
//! # What each instruction becomes
//!
//! A packed-half source is read as the PR reads it (`Visitor::HalfSource`): a register (`RZ`
//! reads zero), an immediate, or a constant-buffer word through the shared source reader; then
//! its two lanes are taken by its swizzle (`Maxwell::Extract`: `UnpackFloat2x16`, lane 0 the low
//! half; `H0_H0` and `H1_H1` give one half in both lanes), and each lane takes its absolute
//! value and then its negation (`FPAbsNeg`). Each instruction then follows its translator:
//!
//! - `HADD2`, `HMUL2`, `HFMA2`: `FPAdd16`, `FPMul16`, `FPFma16` per lane with contraction off,
//!   no rounding and `.FTZ` or no flushing, then `FPSaturate16` with `.SAT`, packed with the low
//!   lane in the low half (`PackFloat2x16`). `HFMA2`'s `.H0_NH1` negates `b`'s high lane once
//!   more after its modifiers. `HFMA2.RELU` clamps each lane in single precision: below zero
//!   gives `+0.0`, a NaN of either sign gives [`RELU_NAN`] narrowed to a half (a NaN result of
//!   a conversion, which the IR makes its canonical half NaN, `0x7fff` in the interpreter, the
//!   value PR #4588's own tests expect), and anything else, `-0.0` included, is kept.
//! - `HADD2.F32`: the PR's "-RZ + b.Hn widened to F32": `b`'s selected half with its modifiers,
//!   converted to a 32-bit float to nearest without flushing.
//! - `HMNMX2`: per lane, both halves widened to single precision, `FPMin32` and `FPMax32`, the
//!   predicate (read with its negation) selecting the minimum, narrowed back.
//! - `HSETP2`: per lane, Eden's `FloatingPointCompare` (`.num` and `.nan` through `FPIsNan16`,
//!   `.f` and `.t` as constants) with `.FTZ` or no flushing, combined with the accumulator by
//!   `PredicateCombine`; the low lane's result goes to the first predicate and the high lane's
//!   to the second, or with `.H_AND` both results' conjunction to the first and its negation to
//!   the second (a `PT` destination is not written).
//! - `HSET2` (Eden's `HSET2`): the same per-lane comparison and combine; each lane is
//!   [`HSET2_TRUE_HALF`] (`1.0`) with `.BF` or [`HSET2_TRUE_MASK`] without where the result is
//!   true, zero where it is false.
//! - `DADD`, `DMUL`, `DFMA`: each source a register pair (low word first, `RZ` a zero), a
//!   uniform-register pair (`URZ` a zero), a constant-buffer double or an immediate high word,
//!   with its absolute value and then its negation; `FPAdd64`, `FPMul64` or `FPFma64` with
//!   contraction off, the rounding field (`RoundingOf`, shared with the float group) and no
//!   flushing; the result written to the destination pair (the high word dropped at `R254`, both
//!   words at `RZ`).
//! - `DSETP` (Eden's `DSETP`): the comparison with Eden's default controls, combined with the
//!   accumulator into the first predicate and its negation combined likewise into the second.
//!
//! Every packed-half instruction uses 16-bit floats and every double-precision one 64-bit
//! floats, so collect info records the f16 and f64 requirements from the translated IR
//! (`crate::info`, `opcode_requirements`); lowering them on hosts that lack them is slice 18
//! (`tests::requirements_are_recorded`).
//!
//! # Guards
//!
//! An instruction under a guard predicate becomes a conditional block (spec, "Instruction
//! translation"), through the shared helper ([`super::conditional_block`]): skipped when the
//! block already tests the guard (flow analysis's `guard_tested`), and for `PT`.
//!
//! # Refusals
//!
//! Each `Require` of the PR's translators for these forms is a named [`HalfDoubleUnsupported`]
//! with its message, checked in the PR's order: the instruction's own form check (`HADD2`,
//! `HMUL2`), each source as `HalfSource` reads it (its form check, then a bindless constant
//! buffer, then the bank as `Visitor::Cbuf` reads the value), the instruction's mode checks, and
//! last each source's swizzle as `HalfLanes` reads it, in operand order. Every check runs
//! before the first instruction is appended, so a refusal leaves the function as it was
//! (`tests::refusals_append_nothing`). The order is pinned with two faults at once by
//! `tests::add_and_multiply_refusals_report_in_the_prs_order`,
//! `tests::fma_refusals_report_in_the_prs_order` and
//! `tests::compare_refusals_report_in_the_prs_order`.
//!
//! A uniform-register source (forms 6 and 7, which slice 6c decodes from NAK) is refused by the
//! check of the PR's that refuses its form: `HADD2`'s `Form() <= 3` ("native half add source
//! form"), `HMUL2`'s `Form() == 1 || 4 || 5` ("native half multiply source form"), or
//! `HalfSource`'s `form >= 1 && form <= 5` ("native half source form") for `HFMA2`, `HMNMX2` and
//! `HSETP2`; for `HSET2`, which only the lowering has, its missing case (`0xe33` is not among
//! `0x233`, `0x433`, `0x633`: "unsupported native opcode"). The double-precision forms with a
//! uniform register are translated: NAK encodes them and no reference refuses them.
//!
//! Some of the PR's checks cannot fail on a decoded instruction and are not repeated:
//! `HADD2`'s `Require(Take(76, 1) == 0 && Take(79, 1) == 0, ..)` and `HMUL2`'s bit 78 in its
//! `Take(78, 2)` (the decoder rejects those bits), `HMNMX2`'s `Require(Take(91, 1) == 0, ..)`
//! (bit 91 is the bindless bit of form 5, refused first as a bindless constant buffer; it is
//! fixed in form 6, which the form check refuses first; forms 1 and 4 do not know it, so the
//! decoder rejects it there), `HSETP2`'s `Require(Take(65, 1) == 0, ..)` (the decoder rejects bit 65) and
//! `Require(set_op <= 2, ..)` ([`PredSetOp`] has three variants), and `HFMA2`'s
//! `Take(87, 4) == 0` without `.RELU` (the decoder refuses another value). The bank check runs
//! on every constant-buffer source, though a decoded bank has five bits, so only a hand-made
//! operation reaches it (`tests::a_bank_past_the_last_is_refused`).
//!
//! **Register pairs** (`DADD`, `DMUL`, `DFMA`, `DSETP`). Eden refuses a pair that does not
//! start at an even register (`TranslatorVisitor::D`: `IsAligned(reg, 2)`, "Unaligned source
//! register", "Unaligned destination register"). That check is kept, for uniform registers
//! too, but `RZ` and `URZ`, which are odd, are taken: as a source they read a zero double, as a
//! destination nothing is written (NAK writes a zero double source as `RZ`, `Src::ZERO`, and
//! a dropped result as `RZ`). A pair at `R254` or `UR62` is even and reads its high word from
//! `RZ` or `URZ`, as Eden reads `X(reg + 1)`. The pairs are checked after every source's
//! constant-buffer checks, in operand order (`a`, `b`, `c`) and then the destination: Eden's
//! order depends on the order its compiler evaluates arguments in, so this order is
//! starquake's (`tests::pairs_report_in_operand_order`).
//!
//! # Divergences
//!
//! | What | Reference | Here | Pinned by |
//! |---|---|---|---|
//! | A guard on these instructions | PR #4588: a select of the result (`guard_select`) | a conditional block (spec, "Instruction translation") | `tests::guarded_instructions_become_conditional_blocks` |
//! | `HADD2.F32` whose `a` is not `-RZ` with neither absolute value nor swizzle | the native translator ignores `a` (its comment: "-RZ + b.Hn widened to F32") | refused ([`HalfDoubleUnsupported::HalfAddFloatOperand`]), the lowering's check (`lower_to_maxwell.h` case `0x230`: `a.reg != 255 \|\| !a.negate \|\| a.absolute \|\| a.swizzle != 0`), so `a` is never silently dropped | `tests::half::hadd2_f32_needs_negated_rz` |
//! | `HSET2` with swizzle 1 on a source | the lowering passes it to Eden's `HSET2`, which widens the other source to single precision (`Extract`'s `F32`) | refused ([`HalfDoubleUnsupported::HalfAsFloat`]), as `HalfLanes` refuses it in the PR's other half translators | `tests::half::swizzle_1_is_refused_everywhere` |
//! | `RZ` and `URZ` in a double source or destination | Eden refuses them, being odd | a zero double, or nothing written (NAK's `Src::ZERO`) | `tests::double::double_pairs_at_each_edge` |
//! | A uniform-register pair in a double source | none: Eden has no uniform registers | read as a pair, refused when odd, as a register pair | `tests::double::double_pairs_at_each_edge` |
//!
//! **Owner question**: where PR #4588's lowering and its native translator read these
//! instructions differently, this module follows the native translator, as the spec's
//! "Instruction translation" names it and the owner's ruling of 2026-11-19 (row 5) did for the
//! translators then built; slice 6c's decoder and slice 12e follow the lowering under
//! graphics-server decision 8 (owner's ruling, 2026-11-20). Each point is pinned by a test:
//! swizzle 1 in `HADD2`, `HMUL2` and `HFMA2` (the lowering widens to single precision, as in
//! `HSET2` above; `tests::half::swizzle_1_is_refused_everywhere`); `HMUL2.FMZ` and `HFMA2.FMZ` (the
//! lowering forwards Maxwell's `FMZ` precision, whose Eden translator then compares a half lane
//! with a single-precision zero, `IREmitter::FPEqual`'s "Mismatching types", unless `.SAT` or a
//! widened source; `tests::half::fmz_is_refused`); `HFMA2.RELU` with a predicate other than `PT` (the
//! lowering keeps the clamped result only where it holds;
//! `tests::half::relu_with_a_predicate_is_refused`);
//! and `HFMA2.RELU`'s clamp of `-0.0` (the lowering, and PR #4588's own test of it in
//! `native_shader.cpp`, the `ReLU` case "preserves fusion values above one and canonical
//! NaNs", give `+0.0` where the native translator keeps `-0.0`;
//! `tests::half::relu_mirrors_the_native_clamp`, `tests::half::pr_relu_cases`).
//!
//! **Equivalent in the interpreter.** `HFMA2.RELU`'s replacement of a NaN lane by
//! [`RELU_NAN`] changes no value the interpreter gives: it makes every NaN result of the
//! narrowing conversion its canonical half NaN whatever the payload. The replacement is kept,
//! as the PR writes it, for a backend whose conversion keeps payloads, and pinned by the IR it
//! builds (`tests::half::golden_relu_clamp`).

#[cfg(test)]
mod tests;

use std::fmt::{self, Display, Formatter};

use thiserror::Error;

use super::float::{HALF_SHIFT, combine_op, control, flushing, rounding_of};
use super::source;
use crate::decode::{
    AluSrc, CBuf, CBufRef, FloatCmpOp, Gpr, HalfSrc, HalfSwizzle, Instr, Op, OpDAdd, OpDFma,
    OpDMul, OpDSetP, OpHAdd2, OpHFma2, OpHMnMx2, OpHMul2, OpHSet2, OpHSetP2, Pred, PredSetOp,
    PredSrc, RoundMode, UniformReg,
};
use crate::ir::{BuildError, Builder, Flags, FmzMode, FpControl, FpRounding, Imm, Opcode, Operand};

/// The number of constant-buffer banks a source may name: `Visitor::Cbuf` refuses
/// `cbuf_index >= 32` ("native constant buffer address"); the float group's bound.
pub use super::float::CBUF_BANKS;
/// What `HFMA2.RELU` gives a NaN lane, as 32-bit float bits before narrowing (`ops_half.cpp`,
/// `HFMA2`: `std::bit_cast<f32>(0x7fc00000U)`, "NaN of either sign becomes the canonical quiet
/// NaN").
pub const RELU_NAN: u32 = 0x7fc0_0000;
/// What `HSET2.BF` writes in a lane whose result is true: `1.0` as a half (Eden's `HSET2`:
/// `bf ? 0x3c00 : 0xffff`).
pub const HSET2_TRUE_HALF: u32 = 0x3c00;
/// What `HSET2` without `.BF` writes in a lane whose result is true: every bit of the lane
/// (Eden's `HSET2`: `bf ? 0x3c00 : 0xffff`).
pub const HSET2_TRUE_MASK: u32 = 0xffff;
/// The lane a packed half's low half is in (`Maxwell::Extract`: `CompositeExtract(vector, 0)`).
pub const LOW_LANE: u32 = 0;
/// The lane a packed half's high half is in (`Maxwell::Extract`: `CompositeExtract(vector, 1)`).
pub const HIGH_LANE: u32 = 1;
/// The byte distance from a constant-buffer double's low word to its high word (NAK's
/// `prop_to_64_src`: `cb.offset(4)`).
pub const DOUBLE_HIGH_WORD: u32 = 4;
/// The shift that puts a 64-bit immediate's 32 encoded bits in its high word (NAK's
/// `prop_to_64_src`: an `Imm32` is the high word, a zero the low).
pub const DOUBLE_IMM_SHIFT: u32 = 32;

/// A form or field value of this group that is not translated: each a `Require` of PR #4588's
/// translators (named in each variant), a case its lowering lacks, or a check of Eden's.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum HalfDoubleUnsupported {
    /// An instruction of another group, which this module does not translate.
    #[error("an instruction that is not a half or double precision one")]
    OtherGroup,
    /// `HADD2` with a uniform-register source (`ops_half.cpp`, `HADD2`: "native half add source
    /// form", `Form() <= 3`).
    #[error("an hadd2 uniform-register operand")]
    HalfAddSourceForm,
    /// `HMUL2` with a uniform-register source (`ops_half.cpp`, `HMUL2`: "native half multiply
    /// source form", `Form() == 1 || Form() == 4 || Form() == 5`).
    #[error("an hmul2 uniform-register operand")]
    HalfMultiplySourceForm,
    /// `HFMA2`, `HMNMX2` or `HSETP2` with a uniform-register source (`Visitor::HalfSource`:
    /// "native half source form", `form >= 1 && form <= 5`).
    #[error("a uniform-register packed-half operand")]
    HalfSourceForm,
    /// `HSET2` with a uniform-register source: PR #4588's lowering has no case for its opcode
    /// `0xe33` ("unsupported native opcode") and its native translator has no `HSET2`.
    #[error("an hset2 uniform-register operand")]
    HSet2UniformForm,
    /// A constant-buffer source whose bank is in a uniform register (`Visitor::HalfSource`:
    /// "native bindless constant buffer"; the lowering's `insn.Bits(91, 1)` for `HSET2`; Eden
    /// has no bindless form for the doubles).
    #[error("a bindless constant-buffer operand")]
    BindlessConstantBuffer,
    /// A constant-buffer bank past the last ([`CBUF_BANKS`]; `Visitor::Cbuf`: "native constant
    /// buffer address"). A decoded bank has five bits, so only a hand-made operation reaches it.
    #[error("constant buffer {0}")]
    ConstantBufferBank(u8),
    /// A packed-half source with swizzle 1, a 32-bit float (`Visitor::HalfLanes`: "native half
    /// source read as float"); the source by its operand number, `a` being 0.
    #[error("packed-half operand {0} read as a 32-bit float")]
    HalfAsFloat(u8),
    /// `HADD2.F32` with `.FTZ` or `.SAT` (`ops_half.cpp`, `HADD2`: "native float-output half add
    /// flags").
    #[error("an hadd2.f32 with ftz or saturation")]
    HalfAddFloatFlags,
    /// `HADD2.F32` whose `b` does not select one half (`ops_half.cpp`, `HADD2`: "native
    /// float-output half add source", `b.swizzle >= 2`).
    #[error("an hadd2.f32 operand of swizzle {}", SwizzleName(*.0))]
    HalfAddFloatSource(HalfSwizzle),
    /// `HADD2.F32` whose `a` is not `-RZ` with neither absolute value nor swizzle: PR #4588's
    /// lowering's check (`lower_to_maxwell.h`, case `0x230`: "native float-output half add
    /// mode"), where its native translator ignores `a` (module docs, "Divergences").
    #[error("an hadd2.f32 whose first operand is not -rz")]
    HalfAddFloatOperand,
    /// `HMUL2.RELU` (`ops_half.cpp`, `HMUL2`: "native half multiply mode", `Take(78, 2) == 0`).
    #[error("hmul2's relu")]
    HalfMultiplyMode,
    /// `HMUL2.FMZ` (`ops_half.cpp`, `HMUL2`: "native half multiply-zero mode").
    #[error("hmul2's multiply-by-zero mode")]
    HalfMultiplyZero,
    /// `HFMA2.FMZ` (`ops_half.cpp`, `HFMA2`: "native half fused multiply-add multiply-zero
    /// mode").
    #[error("hfma2's multiply-by-zero mode")]
    HalfFmaMultiplyZero,
    /// `HFMA2.F32` (`ops_half.cpp`, `HFMA2`: "native half fused multiply-add float mode").
    #[error("hfma2's float output")]
    HalfFmaFloatMode,
    /// `HFMA2.RELU` with a predicate other than `PT` (`ops_half.cpp`, `HFMA2`: "native half
    /// fused multiply-add mode", `Take(87, 4) == (relu ? 7U : 0U)`).
    #[error("an hfma2.relu under predicate {0}")]
    HalfFmaReluPredicate(PredSrc),
    /// `HFMA2` with both `.RELU` and `.SAT` (`ops_half.cpp`, `HFMA2`: "native half fused
    /// multiply-add relu with saturation").
    #[error("an hfma2 with both relu and saturation")]
    HalfFmaReluSaturate,
    /// `HFMA2.BF16_V2` (`ops_half.cpp`, `HFMA2`: "native half fused multiply-add bf16").
    #[error("hfma2's bfloat16 halves")]
    HalfFmaBf16,
    /// `HMNMX2` with `.F32`, `.FTZ`, `.NAN`, `.XORSIGN` or `.BF16_V2` (`ops_half.cpp`,
    /// `HMNMX2`: "native half min/max mode", `Take(76, 11) == 0`).
    #[error("an hmnmx2 with f32, ftz, nan, xorsign or bf16_v2")]
    HalfMinMaxMode,
    /// A double source register pair that does not start at an even register, `RZ` aside
    /// (Eden's `TranslatorVisitor::D`: "Unaligned source register").
    #[error("a double operand in the register pair at {0}")]
    DoubleRegister(Gpr),
    /// A double source uniform-register pair that does not start at an even register, `URZ`
    /// aside (Eden's alignment check, applied to uniform registers here).
    #[error("a double operand in the uniform register pair at {0}")]
    DoubleUniformRegister(UniformReg),
    /// A double destination pair that does not start at an even register, `RZ` aside (Eden's
    /// `TranslatorVisitor::D`: "Unaligned destination register").
    #[error("a double destination in the register pair at {0}")]
    DoubleDestination(Gpr),
}

/// A packed-half swizzle by its encoding, for error messages.
struct SwizzleName(HalfSwizzle);

impl Display for SwizzleName {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0.bits())
    }
}

/// Why [`translate_instr`] did not translate an instruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum HalfDoubleError {
    /// The instruction's form is not translated; nothing was appended.
    #[error("{0} is not supported")]
    Unsupported(HalfDoubleUnsupported),
    /// The IR could not be built: an arena is full, or this module built a malformed
    /// instruction (a bug). Instructions appended before the failure stay in the function.
    #[error(transparent)]
    Build(#[from] BuildError),
}

impl From<HalfDoubleUnsupported> for HalfDoubleError {
    fn from(what: HalfDoubleUnsupported) -> Self {
        Self::Unsupported(what)
    }
}

/// Whether `op` is one of this group's instructions.
pub const fn in_group(op: &Op) -> bool {
    matches!(
        op,
        Op::HAdd2(_)
            | Op::HFma2(_)
            | Op::HMul2(_)
            | Op::HSet2(_)
            | Op::HSetP2(_)
            | Op::HMnMx2(_)
            | Op::DAdd(_)
            | Op::DMul(_)
            | Op::DFma(_)
            | Op::DSetP(_)
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
) -> Result<(), HalfDoubleError> {
    check(&instr.op)?;
    super::conditional_block(builder, instr.guard, guard_tested, |b| {
        Emit { b }.op(&instr.op)
    })
}

// ---------------------------------------------------------------------------------------------
// Checks: every refusal, in the PR's order, before anything is appended.

fn require(ok: bool, what: HalfDoubleUnsupported) -> Result<(), HalfDoubleUnsupported> {
    if ok { Ok(()) } else { Err(what) }
}

const fn is_uniform(src: AluSrc) -> bool {
    matches!(src, AluSrc::UniformReg { .. })
}

/// Whether any of `srcs` is a uniform register: the operand forms 6 and 7.
fn any_uniform(srcs: &[HalfSrc]) -> bool {
    srcs.iter().any(|src| is_uniform(src.src))
}

/// A source's constant-buffer checks as `HalfSource` and `Visitor::Cbuf` make them: a bindless
/// buffer, then the bank.
fn check_cbuf(src: AluSrc) -> Result<(), HalfDoubleUnsupported> {
    match src {
        AluSrc::CBuf {
            cbuf:
                CBufRef {
                    buf: CBuf::BindlessUniform(_),
                    ..
                },
            ..
        } => Err(HalfDoubleUnsupported::BindlessConstantBuffer),
        AluSrc::CBuf {
            cbuf:
                CBufRef {
                    buf: CBuf::Binding(bank),
                    ..
                },
            ..
        } if bank >= CBUF_BANKS => Err(HalfDoubleUnsupported::ConstantBufferBank(bank)),
        _ => Ok(()),
    }
}

/// Every packed-half source as `HalfSource` reads it, in operand order: its form (`form`, the
/// refusal of a uniform register, when given), then its constant-buffer checks.
fn check_half_sources(
    srcs: &[HalfSrc],
    form: Option<HalfDoubleUnsupported>,
) -> Result<(), HalfDoubleUnsupported> {
    if let Some(what) = form {
        require(!any_uniform(srcs), what)?;
    }
    srcs.iter().try_for_each(|src| check_cbuf(src.src))
}

/// Every packed-half source's swizzle as `HalfLanes` reads it, in operand order.
fn check_lanes(srcs: &[HalfSrc]) -> Result<(), HalfDoubleUnsupported> {
    for (index, src) in (0_u8..).zip(srcs) {
        require(
            src.swizzle != HalfSwizzle::F32,
            HalfDoubleUnsupported::HalfAsFloat(index),
        )?;
    }
    Ok(())
}

/// Whether `src` is `-RZ` with no absolute value and both halves as stored: the only `a` of
/// `HADD2.F32` the lowering takes.
const fn is_negated_rz(src: HalfSrc) -> bool {
    matches!(
        src,
        HalfSrc {
            src: AluSrc::Reg {
                reg,
                abs: false,
                neg: true,
            },
            swizzle: HalfSwizzle::H1H0,
        } if reg.is_zero()
    )
}

/// A double source's pair check (module docs, "Register pairs").
fn check_pair(src: AluSrc) -> Result<(), HalfDoubleUnsupported> {
    match src {
        AluSrc::Reg { reg, .. } if !reg.is_zero() && reg.index() % 2 != 0 => {
            Err(HalfDoubleUnsupported::DoubleRegister(reg))
        }
        AluSrc::UniformReg { reg, .. } if !reg.is_zero() && reg.index() % 2 != 0 => {
            Err(HalfDoubleUnsupported::DoubleUniformRegister(reg))
        }
        _ => Ok(()),
    }
}

/// A double instruction's checks: every source's constant-buffer checks, then every source's
/// pair, then the destination pair (`dst`, when it writes one).
fn check_double(srcs: &[AluSrc], dst: Option<Gpr>) -> Result<(), HalfDoubleUnsupported> {
    srcs.iter().try_for_each(|&src| check_cbuf(src))?;
    srcs.iter().try_for_each(|&src| check_pair(src))?;
    match dst {
        Some(dst) if !dst.is_zero() && dst.index() % 2 != 0 => {
            Err(HalfDoubleUnsupported::DoubleDestination(dst))
        }
        _ => Ok(()),
    }
}

/// Every refusal of `op`, in the PR's order.
fn check(op: &Op) -> Result<(), HalfDoubleUnsupported> {
    use HalfDoubleUnsupported as U;
    match *op {
        Op::HAdd2(op) => {
            check_half_sources(&op.srcs, Some(U::HalfAddSourceForm))?;
            if op.f32 {
                let [a, b] = op.srcs;
                require(!op.ftz && !op.saturate, U::HalfAddFloatFlags)?;
                require(
                    matches!(b.swizzle, HalfSwizzle::H0H0 | HalfSwizzle::H1H1),
                    U::HalfAddFloatSource(b.swizzle),
                )?;
                require(is_negated_rz(a), U::HalfAddFloatOperand)
            } else {
                check_lanes(&op.srcs)
            }
        }
        Op::HMul2(op) => {
            check_half_sources(&op.srcs, Some(U::HalfMultiplySourceForm))?;
            require(!op.relu, U::HalfMultiplyMode)?;
            require(!op.dnz, U::HalfMultiplyZero)?;
            check_lanes(&op.srcs)
        }
        Op::HFma2(op) => {
            check_half_sources(&op.srcs, Some(U::HalfSourceForm))?;
            require(!op.dnz, U::HalfFmaMultiplyZero)?;
            require(!op.f32, U::HalfFmaFloatMode)?;
            if let Some(pred) = op.relu {
                require(pred == PredSrc::TRUE, U::HalfFmaReluPredicate(pred))?;
                require(!op.saturate, U::HalfFmaReluSaturate)?;
            }
            require(!op.bf16, U::HalfFmaBf16)?;
            check_lanes(&op.srcs)
        }
        Op::HMnMx2(op) => {
            check_half_sources(&op.srcs, Some(U::HalfSourceForm))?;
            require(
                !op.f32 && !op.ftz && !op.nan && !op.xorsign && !op.bf16,
                U::HalfMinMaxMode,
            )?;
            check_lanes(&op.srcs)
        }
        Op::HSetP2(op) => {
            check_half_sources(&op.srcs, Some(U::HalfSourceForm))?;
            check_lanes(&op.srcs)
        }
        Op::HSet2(op) => {
            check_half_sources(&op.srcs, Some(U::HSet2UniformForm))?;
            check_lanes(&op.srcs)
        }
        Op::DAdd(OpDAdd { dst, srcs, .. }) | Op::DMul(OpDMul { dst, srcs, .. }) => {
            check_double(&srcs, Some(dst))
        }
        Op::DFma(op) => check_double(&op.srcs, Some(op.dst)),
        Op::DSetP(op) => check_double(&op.srcs, None),
        _ => Err(U::OtherGroup),
    }
}

// ---------------------------------------------------------------------------------------------
// Emission.

/// A 32-bit immediate.
const fn imm32(bits: u32) -> Operand {
    Operand::Imm(Imm::U32(bits))
}

/// A 32-bit float immediate, as its bits.
const fn f32_imm(bits: u32) -> Operand {
    Operand::Imm(Imm::F32(bits))
}

/// The width of a floating-point comparison.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Width {
    Half,
    Double,
}

/// A float comparison's opcode in `width`, ordered and unordered (Eden's `FloatingPointCompare`,
/// the `.u` forms unordered); `None` for the four that are not one comparison.
const fn compare_op(cmp: FloatCmpOp, width: Width) -> Option<Opcode> {
    use Opcode as O;
    let half = matches!(width, Width::Half);
    Some(match cmp {
        FloatCmpOp::Lt if half => O::FPOrdLessThan16,
        FloatCmpOp::Lt => O::FPOrdLessThan64,
        FloatCmpOp::Eq if half => O::FPOrdEqual16,
        FloatCmpOp::Eq => O::FPOrdEqual64,
        FloatCmpOp::Le if half => O::FPOrdLessThanEqual16,
        FloatCmpOp::Le => O::FPOrdLessThanEqual64,
        FloatCmpOp::Gt if half => O::FPOrdGreaterThan16,
        FloatCmpOp::Gt => O::FPOrdGreaterThan64,
        FloatCmpOp::Ne if half => O::FPOrdNotEqual16,
        FloatCmpOp::Ne => O::FPOrdNotEqual64,
        FloatCmpOp::Ge if half => O::FPOrdGreaterThanEqual16,
        FloatCmpOp::Ge => O::FPOrdGreaterThanEqual64,
        FloatCmpOp::LtU if half => O::FPUnordLessThan16,
        FloatCmpOp::LtU => O::FPUnordLessThan64,
        FloatCmpOp::EqU if half => O::FPUnordEqual16,
        FloatCmpOp::EqU => O::FPUnordEqual64,
        FloatCmpOp::LeU if half => O::FPUnordLessThanEqual16,
        FloatCmpOp::LeU => O::FPUnordLessThanEqual64,
        FloatCmpOp::GtU if half => O::FPUnordGreaterThan16,
        FloatCmpOp::GtU => O::FPUnordGreaterThan64,
        FloatCmpOp::NeU if half => O::FPUnordNotEqual16,
        FloatCmpOp::NeU => O::FPUnordNotEqual64,
        FloatCmpOp::GeU if half => O::FPUnordGreaterThanEqual16,
        FloatCmpOp::GeU => O::FPUnordGreaterThanEqual64,
        FloatCmpOp::False | FloatCmpOp::True | FloatCmpOp::Num | FloatCmpOp::Nan => return None,
    })
}

/// The two lanes of a packed-half source, low first.
#[derive(Clone, Copy)]
struct Lanes {
    low: Operand,
    high: Operand,
}

/// Appends this group's instructions to one builder.
struct Emit<'b, 'f> {
    b: &'b mut Builder<'f>,
}

impl source::Emit for Emit<'_, '_> {
    fn emit_op(&mut self, opcode: Opcode, args: &[Operand]) -> Result<Operand, BuildError> {
        Ok(self.b.op(opcode, args)?.into())
    }
}

impl Emit<'_, '_> {
    fn op(&mut self, op: &Op) -> Result<(), HalfDoubleError> {
        match *op {
            Op::HAdd2(op) => self.hadd2(op),
            Op::HMul2(op) => self.hmul2(op),
            Op::HFma2(op) => self.hfma2(op),
            Op::HMnMx2(op) => self.hmnmx2(op),
            Op::HSetP2(op) => self.hsetp2(op),
            Op::HSet2(op) => self.hset2(op),
            Op::DAdd(op) => self.dadd(op),
            Op::DMul(op) => self.dmul(op),
            Op::DFma(op) => self.dfma(op),
            Op::DSetP(op) => self.dsetp(op),
            _ => Err(HalfDoubleUnsupported::OtherGroup.into()),
        }
    }

    fn emit(&mut self, opcode: Opcode, args: &[Operand]) -> Result<Operand, HalfDoubleError> {
        Ok(self.b.op(opcode, args)?.into())
    }

    fn emit_fp(
        &mut self,
        opcode: Opcode,
        args: &[Operand],
        control: FpControl,
    ) -> Result<Operand, HalfDoubleError> {
        Ok(self.b.op_with(opcode, args, Flags::Fp(control))?.into())
    }

    /// Writes a register's raw bits; a write to `RZ` is dropped (`Visitor::X`).
    fn set(&mut self, reg: Gpr, value: Operand) -> Result<(), HalfDoubleError> {
        if !reg.is_zero() {
            self.emit(Opcode::SetRegister, &[Operand::Imm(Imm::Reg(reg)), value])?;
        }
        Ok(())
    }

    /// Writes a predicate; a write to `PT` is dropped (`Visitor::SetPred`).
    fn set_pred(&mut self, dst: Pred, value: Operand) -> Result<(), HalfDoubleError> {
        if !dst.is_true() {
            self.emit(Opcode::SetPred, &[Operand::Imm(Imm::Pred(dst)), value])?;
        }
        Ok(())
    }

    /// A predicate's value (`Visitor::Pred`): `PT` is an immediate.
    fn pred(&mut self, src: PredSrc) -> Result<Operand, HalfDoubleError> {
        Ok(source::pred(self, src)?)
    }

    /// A source's raw bits, without modifiers (`Visitor::Value`, the shared reader). [`check`]
    /// refuses a bindless constant buffer first, so the reader's `None` is that refusal again.
    fn value(&mut self, src: AluSrc) -> Result<Operand, HalfDoubleError> {
        source::alu_bits(self, src)?
            .ok_or_else(|| HalfDoubleUnsupported::BindlessConstantBuffer.into())
    }

    /// Takes the absolute value, then negates (Eden's `FPAbsNeg`), with `abs_op` and `neg_op`.
    fn abs_neg(
        &mut self,
        mut value: Operand,
        (abs, neg): (bool, bool),
        (abs_op, neg_op): (Opcode, Opcode),
    ) -> Result<Operand, HalfDoubleError> {
        if abs {
            value = self.emit(abs_op, &[value])?;
        }
        if neg {
            value = self.emit(neg_op, &[value])?;
        }
        Ok(value)
    }

    /// A packed-half source's lanes with its modifiers (`Visitor::HalfLanes`). [`check`] refuses
    /// swizzle 1 first; were it reached, it is read as both halves as stored.
    fn lanes(&mut self, src: HalfSrc) -> Result<Lanes, HalfDoubleError> {
        let bits = self.value(src.src)?;
        let pair = self.emit(Opcode::UnpackFloat2x16, &[bits])?;
        let (low, high) = match src.swizzle {
            HalfSwizzle::H1H0 | HalfSwizzle::F32 => {
                let low = self.emit(Opcode::CompositeExtractF16x2, &[pair, imm32(LOW_LANE)])?;
                let high = self.emit(Opcode::CompositeExtractF16x2, &[pair, imm32(HIGH_LANE)])?;
                (low, high)
            }
            HalfSwizzle::H0H0 => {
                let low = self.emit(Opcode::CompositeExtractF16x2, &[pair, imm32(LOW_LANE)])?;
                (low, low)
            }
            HalfSwizzle::H1H1 => {
                let high = self.emit(Opcode::CompositeExtractF16x2, &[pair, imm32(HIGH_LANE)])?;
                (high, high)
            }
        };
        let modifiers = src.src.modifiers();
        let ops = (Opcode::FPAbs16, Opcode::FPNeg16);
        Ok(Lanes {
            low: self.abs_neg(low, modifiers, ops)?,
            high: self.abs_neg(high, modifiers, ops)?,
        })
    }

    /// Two half lanes packed into a word, the low lane in the low half
    /// (`PackFloat2x16(CompositeConstruct(lhs, rhs))`).
    fn pack(&mut self, low: Operand, high: Operand) -> Result<Operand, HalfDoubleError> {
        let pair = self.emit(Opcode::CompositeConstructF16x2, &[low, high])?;
        self.emit(Opcode::PackFloat2x16, &[pair])
    }

    /// `FPSaturate16` of both lanes when `saturate` is set.
    fn saturate(
        &mut self,
        low: Operand,
        high: Operand,
        saturate: bool,
    ) -> Result<(Operand, Operand), HalfDoubleError> {
        if !saturate {
            return Ok((low, high));
        }
        let low = self.emit(Opcode::FPSaturate16, &[low])?;
        let high = self.emit(Opcode::FPSaturate16, &[high])?;
        Ok((low, high))
    }

    /// The controls of `HADD2`, `HMUL2` and `HFMA2`'s arithmetic: contraction off, no rounding,
    /// `.FTZ` or no flushing.
    const fn arith_control(ftz: bool) -> FpControl {
        control(true, FpRounding::DontCare, flushing(ftz))
    }

    /// `HADD2` (`ops_half.cpp`, `HADD2`).
    fn hadd2(&mut self, op: OpHAdd2) -> Result<(), HalfDoubleError> {
        let [a, b] = op.srcs;
        if op.f32 {
            return self.hadd2_f32(op.dst, b);
        }
        let a = self.lanes(a)?;
        let b = self.lanes(b)?;
        let ctl = Self::arith_control(op.ftz);
        let low = self.emit_fp(Opcode::FPAdd16, &[a.low, b.low], ctl)?;
        let high = self.emit_fp(Opcode::FPAdd16, &[a.high, b.high], ctl)?;
        let (low, high) = self.saturate(low, high, op.saturate)?;
        let packed = self.pack(low, high)?;
        self.set(op.dst, packed)
    }

    /// `HADD2.F32` (`ops_half.cpp`, `HADD2`, `to_float`): `b`'s selected half widened.
    fn hadd2_f32(&mut self, dst: Gpr, b: HalfSrc) -> Result<(), HalfDoubleError> {
        // [`check`] takes only `H0_H0` and `H1_H1` here, which put the one half they select in
        // both lanes, so the low lane is it (the PR's `b.swizzle == 3 ? rhs : lhs`).
        let selected = self.lanes(b)?.low;
        let widen = control(false, FpRounding::Rn, FmzMode::None);
        let wide = self.emit_fp(Opcode::ConvertF32F16, &[selected], widen)?;
        let bits = self.emit(Opcode::BitCastU32F32, &[wide])?;
        self.set(dst, bits)
    }

    /// `HMUL2` (`ops_half.cpp`, `HMUL2`).
    fn hmul2(&mut self, op: OpHMul2) -> Result<(), HalfDoubleError> {
        let [a, b] = op.srcs;
        let a = self.lanes(a)?;
        let b = self.lanes(b)?;
        let ctl = Self::arith_control(op.ftz);
        let low = self.emit_fp(Opcode::FPMul16, &[a.low, b.low], ctl)?;
        let high = self.emit_fp(Opcode::FPMul16, &[a.high, b.high], ctl)?;
        let (low, high) = self.saturate(low, high, op.saturate)?;
        let packed = self.pack(low, high)?;
        self.set(op.dst, packed)
    }

    /// `HFMA2` (`ops_half.cpp`, `HFMA2`).
    fn hfma2(&mut self, op: OpHFma2) -> Result<(), HalfDoubleError> {
        let [a, b, c] = op.srcs;
        let a = self.lanes(a)?;
        let mut b = self.lanes(b)?;
        if op.neg_high_b {
            b.high = self.emit(Opcode::FPNeg16, &[b.high])?;
        }
        let c = self.lanes(c)?;
        let ctl = Self::arith_control(op.ftz);
        let mut low = self.emit_fp(Opcode::FPFma16, &[a.low, b.low, c.low], ctl)?;
        let mut high = self.emit_fp(Opcode::FPFma16, &[a.high, b.high, c.high], ctl)?;
        if op.relu.is_some() {
            low = self.relu(low)?;
            high = self.relu(high)?;
        }
        let (low, high) = self.saturate(low, high, op.saturate)?;
        let packed = self.pack(low, high)?;
        self.set(op.dst, packed)
    }

    /// `HFMA2.RELU`'s clamp of one lane (`ops_half.cpp`, `HFMA2`, `clamp`), in single
    /// precision with Eden's default controls.
    fn relu(&mut self, lane: Operand) -> Result<Operand, HalfDoubleError> {
        let plain = FpControl::default();
        let wide = self.emit_fp(Opcode::ConvertF32F16, &[lane], plain)?;
        let zero = f32_imm(0);
        let below = self.emit_fp(Opcode::FPOrdLessThan32, &[wide, zero], plain)?;
        let clamped = self.emit(Opcode::SelectF32, &[below, zero, wide])?;
        let nan = self.emit(Opcode::FPIsNan32, &[wide])?;
        let picked = self.emit(Opcode::SelectF32, &[nan, f32_imm(RELU_NAN), clamped])?;
        self.emit_fp(Opcode::ConvertF16F32, &[picked], plain)
    }

    /// `HMNMX2` (`ops_half.cpp`, `HMNMX2`).
    fn hmnmx2(&mut self, op: OpHMnMx2) -> Result<(), HalfDoubleError> {
        let [a, b] = op.srcs;
        let a = self.lanes(a)?;
        let b = self.lanes(b)?;
        let select_min = self.pred(op.min)?;
        let low = self.min_max_lane(select_min, a.low, b.low)?;
        let high = self.min_max_lane(select_min, a.high, b.high)?;
        let packed = self.pack(low, high)?;
        self.set(op.dst, packed)
    }

    /// One lane of `HMNMX2` (`ops_half.cpp`, `HMNMX2`, `lane`).
    fn min_max_lane(
        &mut self,
        select_min: Operand,
        x: Operand,
        y: Operand,
    ) -> Result<Operand, HalfDoubleError> {
        let plain = FpControl::default();
        let x = self.emit_fp(Opcode::ConvertF32F16, &[x], plain)?;
        let y = self.emit_fp(Opcode::ConvertF32F16, &[y], plain)?;
        let low = self.emit_fp(Opcode::FPMin32, &[x, y], plain)?;
        let high = self.emit_fp(Opcode::FPMax32, &[x, y], plain)?;
        let picked = self.emit(Opcode::SelectF32, &[select_min, low, high])?;
        self.emit_fp(Opcode::ConvertF16F32, &[picked], plain)
    }

    /// Eden's `FloatingPointCompare` in `width`.
    fn compare(
        &mut self,
        cmp: FloatCmpOp,
        a: Operand,
        b: Operand,
        width: Width,
        ctl: FpControl,
    ) -> Result<Operand, HalfDoubleError> {
        if let Some(opcode) = compare_op(cmp, width) {
            return self.emit_fp(opcode, &[a, b], ctl);
        }
        let is_nan = match width {
            Width::Half => Opcode::FPIsNan16,
            Width::Double => Opcode::FPIsNan64,
        };
        match cmp {
            FloatCmpOp::False => Ok(Operand::Imm(Imm::U1(false))),
            FloatCmpOp::True => Ok(Operand::Imm(Imm::U1(true))),
            FloatCmpOp::Num => {
                let a_nan = self.emit(is_nan, &[a])?;
                let b_nan = self.emit(is_nan, &[b])?;
                let a_num = self.emit(Opcode::LogicalNot, &[a_nan])?;
                let b_num = self.emit(Opcode::LogicalNot, &[b_nan])?;
                self.emit(Opcode::LogicalAnd, &[a_num, b_num])
            }
            _ => {
                let a_nan = self.emit(is_nan, &[a])?;
                let b_nan = self.emit(is_nan, &[b])?;
                self.emit(Opcode::LogicalOr, &[a_nan, b_nan])
            }
        }
    }

    /// The per-lane results of `HSETP2` and `HSET2`: each lane's comparison combined with the
    /// accumulator, low lane first.
    fn half_compare(
        &mut self,
        cmp: FloatCmpOp,
        srcs: [HalfSrc; 2],
        set_op: PredSetOp,
        accum: PredSrc,
        ftz: bool,
    ) -> Result<(Operand, Operand), HalfDoubleError> {
        let [a, b] = srcs;
        let a = self.lanes(a)?;
        let b = self.lanes(b)?;
        let ctl = control(false, FpRounding::DontCare, flushing(ftz));
        let accum = self.pred(accum)?;
        let combine = combine_op(set_op);
        let low = self.compare(cmp, a.low, b.low, Width::Half, ctl)?;
        let low = self.emit(combine, &[low, accum])?;
        let high = self.compare(cmp, a.high, b.high, Width::Half, ctl)?;
        let high = self.emit(combine, &[high, accum])?;
        Ok((low, high))
    }

    /// `HSETP2` (`ops_half.cpp`, `HSETP2`).
    fn hsetp2(&mut self, op: OpHSetP2) -> Result<(), HalfDoubleError> {
        let (low, high) = self.half_compare(op.cmp, op.srcs, op.set_op, op.accum, op.ftz)?;
        if op.horizontal {
            let both = self.emit(Opcode::LogicalAnd, &[low, high])?;
            self.set_pred(op.dst, both)?;
            let neither = self.emit(Opcode::LogicalNot, &[both])?;
            self.set_pred(op.dst1, neither)
        } else {
            self.set_pred(op.dst, low)?;
            self.set_pred(op.dst1, high)
        }
    }

    /// `HSET2` (Eden's `half_floating_point_set.cpp`, `HSET2`).
    fn hset2(&mut self, op: OpHSet2) -> Result<(), HalfDoubleError> {
        let (low, high) = self.half_compare(op.cmp, op.srcs, op.set_op, op.accum, op.ftz)?;
        let truth = if op.bf {
            HSET2_TRUE_HALF
        } else {
            HSET2_TRUE_MASK
        };
        let low = self.emit(Opcode::SelectU32, &[low, imm32(truth), imm32(0)])?;
        let high = self.emit(
            Opcode::SelectU32,
            &[high, imm32(truth << HALF_SHIFT), imm32(0)],
        )?;
        let value = self.emit(Opcode::BitwiseOr32, &[low, high])?;
        self.set(op.dst, value)
    }

    /// A double source with its modifiers: a register or uniform-register pair, a
    /// constant-buffer double or an immediate high word (module docs).
    fn double(&mut self, src: AluSrc) -> Result<Operand, HalfDoubleError> {
        let value = match src {
            AluSrc::Imm(bits) => {
                return Ok(Operand::Imm(Imm::F64(u64::from(bits) << DOUBLE_IMM_SHIFT)));
            }
            AluSrc::Reg { reg, .. } if reg.is_zero() => Operand::Imm(Imm::F64(0)),
            AluSrc::UniformReg { reg, .. } if reg.is_zero() => Operand::Imm(Imm::F64(0)),
            AluSrc::Reg { reg, .. } => {
                let low = source::reg(self, reg)?;
                let high = source::reg(self, next_gpr(reg))?;
                self.pack_double(low, high)?
            }
            AluSrc::UniformReg { reg, .. } => {
                let low = source::uniform_reg(self, reg)?;
                let high = source::uniform_reg(self, next_uniform(reg))?;
                self.pack_double(low, high)?
            }
            AluSrc::CBuf {
                cbuf:
                    CBufRef {
                        buf: CBuf::Binding(bank),
                        offset,
                    },
                ..
            } => {
                let offset = u32::from(offset);
                let low = source::cbuf_word(self, bank, offset)?;
                let high = source::cbuf_word(self, bank, offset + DOUBLE_HIGH_WORD)?;
                self.pack_double(low, high)?
            }
            AluSrc::CBuf { .. } => {
                return Err(HalfDoubleUnsupported::BindlessConstantBuffer.into());
            }
        };
        self.abs_neg(value, src.modifiers(), (Opcode::FPAbs64, Opcode::FPNeg64))
    }

    /// Two words as a double, the low word first (`PackDouble2x32`).
    fn pack_double(&mut self, low: Operand, high: Operand) -> Result<Operand, HalfDoubleError> {
        let pair = self.emit(Opcode::CompositeConstructU32x2, &[low, high])?;
        self.emit(Opcode::PackDouble2x32, &[pair])
    }

    /// Writes a double to the pair at `dst`, the low word first (Eden's
    /// `TranslatorVisitor::D`); `RZ` writes nothing and a pair at `R254` drops its high word.
    fn set_double(&mut self, dst: Gpr, value: Operand) -> Result<(), HalfDoubleError> {
        if dst.is_zero() {
            return Ok(());
        }
        let words = self.emit(Opcode::UnpackDouble2x32, &[value])?;
        let low = self.emit(Opcode::CompositeExtractU32x2, &[words, imm32(0)])?;
        self.set(dst, low)?;
        let high = self.emit(Opcode::CompositeExtractU32x2, &[words, imm32(1)])?;
        self.set(next_gpr(dst), high)
    }

    /// The controls of `DADD`, `DMUL` and `DFMA` (Eden's: contraction off, the rounding field,
    /// no flushing).
    const fn double_control(rnd: RoundMode) -> FpControl {
        control(true, rounding_of(rnd), FmzMode::None)
    }

    /// `DADD` (Eden's `double_add.cpp`).
    fn dadd(&mut self, op: OpDAdd) -> Result<(), HalfDoubleError> {
        let [a, b] = op.srcs;
        let a = self.double(a)?;
        let b = self.double(b)?;
        let value = self.emit_fp(Opcode::FPAdd64, &[a, b], Self::double_control(op.rnd))?;
        self.set_double(op.dst, value)
    }

    /// `DMUL` (Eden's `double_multiply.cpp`).
    fn dmul(&mut self, op: OpDMul) -> Result<(), HalfDoubleError> {
        let [a, b] = op.srcs;
        let a = self.double(a)?;
        let b = self.double(b)?;
        let value = self.emit_fp(Opcode::FPMul64, &[a, b], Self::double_control(op.rnd))?;
        self.set_double(op.dst, value)
    }

    /// `DFMA` (Eden's `double_fused_multiply_add.cpp`).
    fn dfma(&mut self, op: OpDFma) -> Result<(), HalfDoubleError> {
        let [a, b, c] = op.srcs;
        let a = self.double(a)?;
        let b = self.double(b)?;
        let c = self.double(c)?;
        let value = self.emit_fp(Opcode::FPFma64, &[a, b, c], Self::double_control(op.rnd))?;
        self.set_double(op.dst, value)
    }

    /// `DSETP` (Eden's `double_set_predicate.cpp`).
    fn dsetp(&mut self, op: OpDSetP) -> Result<(), HalfDoubleError> {
        let [a, b] = op.srcs;
        let a = self.double(a)?;
        let b = self.double(b)?;
        let comparison = self.compare(op.cmp, a, b, Width::Double, FpControl::default())?;
        let accum = self.pred(op.accum)?;
        let combine = combine_op(op.set_op);
        let first = self.emit(combine, &[comparison, accum])?;
        self.set_pred(op.dst, first)?;
        let inverted = self.emit(Opcode::LogicalNot, &[comparison])?;
        let second = self.emit(combine, &[inverted, accum])?;
        self.set_pred(op.dst1, second)
    }
}

/// The register after `reg` in a pair; `RZ` after `R254` (and after `RZ`, which no caller
/// passes: a pair at `RZ` is read or written as a whole).
fn next_gpr(reg: Gpr) -> Gpr {
    reg.index().checked_add(1).map_or(Gpr::RZ, Gpr::new)
}

/// The uniform register after `reg` in a pair; `URZ` after `UR62`.
fn next_uniform(reg: UniformReg) -> UniformReg {
    reg.index()
        .checked_add(1)
        .and_then(UniformReg::new)
        .unwrap_or(UniformReg::URZ)
}
