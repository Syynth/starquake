//! Slice 12d: the uniform datapath's translator, decoded instructions of slice 6d to IR.
//!
//! [`translate_uniform`] appends one instruction's IR at a [`Builder`]'s position, and
//! [`translate_uniform_word`] decodes a word first, so an unknown opcode or bit is refused with
//! the decoder's exact error ([`UniformError::Decode`]). [`in_group`] says which operations are
//! this group's: `R2UR`, `UP2UR` (a shell no word decodes to), `UMOV`, `ULDC` (with `LDCG`),
//! `UPLOP3` and the twins (`UIADD3`, `UIADD3.X`, `ULEA`, `ULEA.X`, `ULOP3`, `UPRMT`, `USHF`,
//! `UIMAD`, `UIMAD.WIDE`, `UISETP`, `USEL`, `USGXT`, `UBMSK`, `UFLO`, `UBREV`, `UPOPC`); any
//! other is [`UniformError::OtherGroup`].
//!
//! Uniform registers and predicates keep their own register class (spec, "The intermediate
//! representation"): they are read and written with `GetUniformRegister`,
//! `SetUniformRegister`, `GetUniformPred` and `SetUniformPred`, never packed into the
//! per-thread file as PR #4588 packs them (`native.h`, `UniformLayout`).
//!
//! # Sources
//!
//! Ported by behaviour from PR #4588's native frontend
//! (`eden-pr4588/src/shader_recompiler/frontend/ampere/` @ e5fb3b97, GPL-3.0-or-later, Copyright
//! 2026 Eden Emulator Project): `ops_memory.cpp` (`R2UR`, `BREV`), `ops_misc.cpp` (`UMOV`,
//! `ULDC`), `ops_alu.cpp` (`PLOP3`, `ISETP`, `SEL`, `IADD3`), `ops_int.cpp` (the uniform paths of
//! `LOP3`, `SHF`, `LEA`, `IMAD`, `IMAD_WIDE`, `SGXT`, `FLO`, `POPC`), `visitor.cpp`
//! (`Visitor::Run`, `UR`, `UPred`, `SetUPred`, `Dst`, `WriteDst`, `PredIn`, `PredOut`) and
//! `classify.h`; and its lowering to Maxwell (`lower_to_maxwell.h` @ e5fb3b97, the same
//! licence), whose reading wins wherever the two differ (decision log, 2026-11-20, "Owner
//! rulings on the wave 47 questions", row 7; spec, "The lowering's reading wins in general").
//! `LDCG`, which neither has, follows NAK (`mesa-nouveau/src/nouveau/compiler/nak/ir.rs` @
//! 52238762, MIT, Copyright 2022 Collabora, Ltd.: `OpLdcg`, "On true the load returns 0"; and
//! `sm70_encode.rs`, `impl SM70Op for OpLdcg`).
//!
//! # The twins
//!
//! A twin is the vector integer operation on the uniform datapath (the decoder's
//! `op::uniform` module docs). It is translated by the integer group's own code
//! ([`super::integer`]'s `translate_in`), with its [`crate::decode::Gpr`] operands naming the
//! uniform registers of the same numbers (`RZ` names `URZ`), so each twin has its vector
//! instruction's semantics, checks and divergences from PR #4588 (the integer module's docs),
//! as PR #4588's translator runs the same visitor functions with `uniform_op` set. Two
//! limits change with the datapath, as the PR's: an `IMAD.WIDE` pair starts below
//! [`super::integer::UNIFORM_WIDE_PAIR_LIMIT`] (62, `Visitor::IMAD_WIDE`: `limit = uniform_op ?
//! 62 : 254`), and its refusals name the uniform register
//! ([`IntegerUnsupported::WideUniformDestination`], [`IntegerUnsupported::WideAddendUniform`]).
//!
//! Before the integer group's checks, two forms are refused by name: `UBMSK`, which neither
//! reference translates ([`UniformUnsupported::Instruction`], as the integer group refuses
//! `BMSK`), and a twin's form 2 (an immediate source 2), which `classify.h` does not classify,
//! so `Visitor::Run` reaches its `default` case ("unsupported native opcode"), and which the
//! lowering has no case for ([`UniformUnsupported::ImmediateSource2`]).
//!
//! **Predicate files.** PR #4588's native translator reads every predicate operand of a twin
//! as a uniform predicate (`PredIn`, `PredOut`), but for `ULOP3`'s predicate output, which it
//! writes to a per-thread predicate (`Visitor::LOP3`: `SetPred`). The lowering reads three
//! fields differently, and its reading is followed (each a **guess** until an `nvdisasm` or
//! hardware check); for one form the references conflict and neither backs the reading made
//! here:
//!
//! | Field | Here | The lowering | Native translator | Pinned by |
//! |---|---|---|---|---|
//! | `ULOP3` form 4 (`0x892`, an immediate source 1)'s predicate output (bits 81..84) | A uniform predicate, the lowering's | A uniform predicate (`0x892` is in `uniform_predicate_bank`, which maps bits 81..84 into the uniform bank) | A predicate (`SetPred`) | `tests::pr4588::uniform_bit_tests` |
//! | `ULOP3` form 1 (`0x292`, a register source 1)'s predicate output | A uniform predicate, as form 4's: a **conflict**, read as form 4 is (a **guess**: neither reference backs it) | Refused unless it is `PT` (`opcode == 0x292 && ... insn.Bits(81, 3) != 7`: "native uniform logic mode") | A predicate (`SetPred`) | `tests::files::ulop3_writes_a_uniform_predicate` |
//! | `ULEA`'s and `ULEA.X`'s carry input (87..91) and output (81..84) | Predicates, the lowering's | Predicates (`0x291` and `0x891` are rewritten to `LEA`, `0x211`/`0x811`, outside `uniform_predicate_bank`, and `LEA`'s cases read both fields as predicates) | Uniform predicates | `tests::files::ulea_carries_are_predicates` |
//! | `UIMAD.X`'s carry input (87..91) | A predicate, the lowering's | A predicate (`0x2a4`/`0x8a4` are rewritten to `IMAD`, `0x224`/`0x824`, whose `.X` case reads it as one) | A uniform predicate | `tests::files::uimad_x_carry_is_a_predicate` |
//!
//! Every other predicate operand of a twin is a uniform predicate in both
//! (`UIADD3`'s and `UIADD3.X`'s carries: `0x290`/`0x890` are in `uniform_predicate_bank`;
//! `UISETP`'s accumulator and results and `USEL`'s condition: their cases map them into the
//! uniform bank). `ULEA.HI` writes its uniform destination, as the lowering does (`base(..)`
//! with the mapped register), where the native translator's `.HI` path writes the per-thread
//! register (`X(dest, result)`): `tests::files::ulea_hi_writes_a_uniform_register`.
//!
//! `UPRMT` is translated as `PRMT` is (an immediate selector in the default mode; the integer
//! group's checks): the lowering translates `0x896` (one selector, `0x9910`), where the native
//! translator has no case (`classify.h` does not classify `0x096`), a listed divergence pinned
//! by `tests::alu::uprmt_permutes_uniform_bytes`.
//!
//! # The other instructions
//!
//! - **`R2UR`** (`Visitor::R2UR`, the lowering's `case 0x3c2`): the source register of the
//!   lowest lane whose guard holds (a ballot of the guard, its lowest set bit, a shuffle from
//!   that lane) into the uniform register, in every lane while the ballot is not zero. The
//!   guard is read as data, a predicate, as both references read it (`classify.h`: "reads the
//!   guard as data, so it is not split into a block"; the lowering's `case 0x3c2` ballots
//!   `native_predicate`), so a guarded `R2UR` is **not** a conditional block (see
//!   "Divergences"). When no active lane's
//!   guard holds (the ballot is zero), the uniform register keeps its old value, as the
//!   lowering keeps it: its `VOTE.ANY` (`0x50d8`) sets `P6` to whether any lane's guard holds,
//!   and its closing `SEL` (`0x5ca0` on `P6`) writes `P6 ? shuffled : destination`. The native
//!   translator writes the shuffle unconditionally, from lane `FindUMsb32(0)`, which is out of
//!   range (a value the warp does not define); decision 8 (the lowering wins) settles it, and
//!   PR #4588's own test agrees ("Native warp uniform transfers use the first selected active
//!   lane and preserve predicates" expects the old value for an empty ballot). Unguarded, the
//!   running lane is in the ballot, so no select is made. PR #4588's `Require(destination <
//!   63)` is kept ([`UniformUnsupported::R2urDestination`]); its predicate-output check cannot
//!   fail here (the decoder fixes the field).
//! - **`UMOV`** (`Visitor::UMOV`): an immediate or a uniform register into a uniform register.
//! - **`ULDC`** (`Visitor::ULDC`): a constant-buffer read of the decoded type into one, two or
//!   four uniform registers (`GetCbufU8` to `GetCbufU32x2`; `.128` is four `GetCbufU32` four
//!   bytes apart), at the offset, plus the index register for `0xabb`. PR #4588's checks, in
//!   its order: a bindless buffer ([`UniformUnsupported::BindlessConstantLoad`]), an offset
//!   that is not a multiple of four ([`UniformUnsupported::ConstantLoadOffset`]), and the
//!   destination of a 64- or 128-bit read ([`UniformUnsupported::ConstantLoadDestination`]:
//!   even and below 62, a multiple of four below 60). Its bank (`< 32`) and width (`<= 6`)
//!   checks cannot fail on a decoded field.
//! - **`LDCG`** (NAK `OpLdcg`): a load of the decoded type from global memory at the uniform
//!   register pair plus the signed offset, or zero where its uniform predicate holds (no load
//!   then: the load is in a conditional block on the predicate's inverse). NAK's
//!   `assert_ne!(self.mem_type, MemType::B128)` below SM 10.0 is kept
//!   ([`UniformUnsupported::GlobalLoad128`]); a 64-bit destination must be even and below 62,
//!   as `ULDC`'s (a **guess**: NAK allocates the pair aligned, and no reference checks it).
//! - **`UPLOP3`** (`Visitor::PLOP3` with `uniform_op` set): two truth tables of three uniform
//!   predicates into two uniform predicates, by the integer group's `PLOP3` code over the
//!   uniform predicate file; any truth table (row 88).
//! - **`UP2UR`** has no translator in either reference's native frontend and no decoder form
//!   (it is a shell): refused by name ([`UniformUnsupported::Instruction`]).
//!
//! # Guards
//!
//! The guard follows the translation driver's rule (`super::guard_step`; `super`'s module
//! docs, "Guards"): `!PT` appends nothing, without any check; with [`GuardMode::Tested`] any
//! other guard is ignored; with [`GuardMode::Honour`] a guard other than `PT` makes the
//! instruction a conditional block ([`super::split_for_guard`]), its guard read as a uniform
//! predicate of the same number for the forms `classify.h` marks uniform (`UMOV`, `ULDC` `0xab9`
//! and `0xabb`, `UPLOP3`, and the twelve twin bases in forms 1 and 4: `UIADD3`, `ULEA`,
//! `ULOP3`, `USHF`, `UIMAD`, `UIMAD.WIDE`, `UBREV`, `USGXT`, `UFLO`, `UPOPC`, `UISETP`, `USEL`)
//! and as a predicate for the others (`UPRMT`, `LDCG`), as `Visitor::Run` reads it
//! ([`reads_uniform_guard`], checked against flow analysis's classes by
//! `tests::guards::the_guard_file_is_classify_s`). `R2UR` reads its guard as data instead.
//! A conditional block in a block one of whose successors has a phi naming it is refused
//! before anything is appended ([`UniformError::SuccessorPhi`]), as the integer group does.
//!
//! The order of the checks: the group, the guard (`!PT` appends nothing), the form (`UBMSK`, a
//! twin's form 2, `UP2UR`), then for a twin the integer group's own order, and for the others
//! the instruction's checks in PR #4588's order, then a guard's successor phis
//! (`tests::rejections::the_checks_come_in_their_order`). A refused instruction appends nothing.
//!
//! # Divergences
//!
//! | What | Here | PR #4588 | Pinned by |
//! |---|---|---|---|
//! | A guarded pure instruction | A conditional block (spec, "Instruction translation") | A select of each written value (`guard_select`) | `tests::guards::a_guard_makes_a_conditional_block` |
//! | A guarded `R2UR` | The guard read as data, no conditional block: an **owner question** (below) | The same (`classify.h`, `Visitor::R2UR`) | `tests::guards::r2ur_reads_its_guard_as_data` |
//! | A guarded `R2UR` whose guard holds in no active lane | The uniform register keeps its value (the lowering's `VOTE.ANY` and `SEL`): a **conflict**, the lowering's reading chosen (decision 8) | The lowering keeps it; the native translator writes the shuffle from the out-of-range lane `FindUMsb32(0)` | `tests::guards::r2ur_with_no_lane_keeps_its_register` |
//! | `@!UPT` on a twin the lowering takes (`0x891`, "Native predicated uniform address arithmetic") | Nothing appended (the driver's `!PT` rule; `@!UPT` is the same guard field as `@!PT`) | The native translator appends nothing (`Visitor::Run`: `if (guard.Never()) return;`); the lowering refuses it ("native inverted uniform true predicate"): a **conflict**, the native reading kept, as the driver keeps it for every group | `tests::pr4588::predicated_uniform_address_arithmetic` |
//! | The predicate files above | The lowering's, but `ULOP3` form 1's predicate output, a conflict read as form 4's | The native translator's | `tests::files`, `tests::pr4588::uniform_bit_tests` |
//! | `ULDC` `0xabb`'s offset with a 32- or 64-bit type, and `0xab9`'s with a 64-bit type | Signed 16 bits, as the lowering reads it: its `case 0xab9: case 0xabb` emits Maxwell's `LDC` (`0xef90`), whose offset Eden reads as signed (`frontend/maxwell/translate/impl/load_constant.h` @ cf654052, `BitField<20, 16, s64> offset`) | Unsigned (`Take(38, 16)` added as is) | `tests::constant::the_lowering_s_offsets_are_signed` |
//! | `ULDC` `0xab9`'s offset with a 32-bit, 8- or 16-bit or 128-bit type, and `0xabb`'s with an 8- or 16-bit or 128-bit type | Unsigned, the native reading: the lowering reads `0xab9` `.32` through Maxwell's unsigned constant-buffer operand, and refuses the other types on both opcodes (`width != 4 && width != 5`: "native uniform constant load mode") | The same | `tests::constant::the_lowering_s_offsets_are_signed` |
//! | `UPRMT` | Translated (an immediate selector) | No native case; the lowering's one selector | `tests::alu::uprmt_permutes_uniform_bytes` |
//! | `LDCG` | Translated from NAK | No case in either | `tests::global::ldcg_loads_or_returns_zero` |
//!
//! **Owner question: a guarded `R2UR`.** The spec reads "Every guarded instruction that writes
//! IR becomes a conditional block, pure ones included" ("Instruction translation", "Guards").
//! A guarded `R2UR` writes IR, but both references read its guard as data (the ballot's
//! input), and both write the uniform register in every lane when at least one active lane's
//! guard holds. They differ when none holds: the lowering keeps the register (its `VOTE.ANY`
//! and `SEL`), while the native translator writes the shuffle from an out-of-range lane, an
//! undefined value. Option 1, as built: read the guard as data, no block, and keep the
//! register when the ballot is zero; this follows the lowering in every case (decision 8) and
//! the native translator only when some lane's guard holds. Option 2, the spec's sentence: a
//! conditional block, so lanes where the guard is false keep their old value, and the ballot
//! sees only the lanes that entered it; this follows neither reference where the guard differs
//! between lanes.
//!
//! # Not translated
//!
//! The float twins (`UFMNMX`, `UI2F`, `UF2I`, row 88) are not decoded (the decoder's
//! `op::uniform` docs, "Left as unknown opcodes"), so they never reach this module; nor does
//! `S2UR` or `VOTEU`, which are the control group's. No other group's translator is handed a
//! uniform datapath form: of the forms flow analysis classifies uniform, only `S2UR` and
//! `VOTEU` are outside this group (`driver::tests::the_guard_of_every_pure_form_is_read_by_its_kind`),
//! so the integer, memory, float and texture groups' own guard reads, which read a predicate,
//! need no uniform-predicate read.

#[cfg(test)]
mod tests;

use thiserror::Error;

use super::integer::{
    self, Files, GuardMode, IntegerError, IntegerUnsupported, PredFile, successor_phi,
};
use super::memory::{cbuf_load, global_load, words_of};
use super::{GuardStep, guard_step, source, split_for_guard};
use crate::decode::uniform::TwinOperands;
use crate::decode::{
    AluSrc, CBuf, CBufRef, DecodeError, Instr, Instruction, MemType, Op, OpPLop3, OpR2UR, OpULdc,
    OpUMov, OpUPLop3, PLop3Src, Pred, PredSrc, ULdcSrc, UMovSrc, Uniform, UniformPredSrc,
    UniformReg, decode,
};
use crate::ir::{BlockId, BuildError, Builder, Imm, Opcode, Operand};

/// A form or field value of this group that is not translated: one of PR #4588's `Require`
/// checks (or NAK's assertion, for `LDCG`), named by its message, or an instruction neither
/// reference translates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum UniformUnsupported {
    /// An instruction neither reference's native frontend translates: `UBMSK`, and `UP2UR`, a
    /// shell no word decodes to.
    #[error("{0}, which neither reference translates,")]
    Instruction(&'static str),
    /// A twin with an immediate source 2 (form 2), which `classify.h` does not classify, so
    /// `Visitor::Run` refuses it ("unsupported native opcode"), and the lowering has no case
    /// for.
    #[error("{0} with an immediate source 2")]
    ImmediateSource2(&'static str),
    /// `R2UR` into `URZ` (`Visitor::R2UR`: "native register to uniform destination").
    #[error("an r2ur into urz")]
    R2urDestination,
    /// A bindless `ULDC` (`Visitor::ULDC`: "native uniform constant load flag", bit 91 set on
    /// `0xab9`).
    #[error("a bindless uldc")]
    BindlessConstantLoad,
    /// A `ULDC` offset that is not a multiple of four (`Visitor::ULDC`: "native uniform
    /// constant buffer address").
    #[error("a uldc at offset {0:#x}, which is not a multiple of four")]
    ConstantLoadOffset(u16),
    /// A 64- or 128-bit `ULDC` (or 64-bit `LDCG`) whose destination is not an even uniform
    /// register below 62, or a multiple of four below 60 (`Visitor::ULDC`: "native uniform
    /// constant load destination").
    #[error("a {words}-word load into {dst}")]
    ConstantLoadDestination {
        /// The destination.
        dst: UniformReg,
        /// The registers the type moves.
        words: u8,
    },
    /// A 128-bit `LDCG` (NAK `OpLdcg`: `assert_ne!(self.mem_type, MemType::B128)` below SM
    /// 10.0).
    #[error("a 128-bit ldcg")]
    GlobalLoad128,
}

/// Why a uniform datapath instruction was not translated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum UniformError {
    /// The word does not decode.
    #[error(transparent)]
    Decode(#[from] DecodeError),
    /// The instruction belongs to another group.
    #[error("{0} is not a uniform datapath instruction")]
    OtherGroup(&'static str),
    /// The form or a field value is not translated.
    #[error("{0} is not supported")]
    Unsupported(UniformUnsupported),
    /// One of the integer group's checks refuses a twin's vector operation (the module docs,
    /// "The twins"); its message names the vector instruction.
    #[error("{0} is not supported")]
    Twin(IntegerUnsupported),
    /// A twin built by hand has a source modifier the integer group would not apply.
    #[error("{0} has a source modifier it does not take")]
    Modifier(&'static str),
    /// A guarded instruction would split `block`, whose successor `successor` has a phi naming
    /// it (the module docs, "Guards").
    #[error("a guarded instruction cannot split {block}: {successor} has a phi naming it")]
    SuccessorPhi {
        /// The block the builder is in.
        block: BlockId,
        /// Its successor with the phi.
        successor: BlockId,
    },
    /// The IR could not be built: an arena is full, or this module built a malformed
    /// instruction (a bug). The function may hold part of the instruction's IR.
    #[error(transparent)]
    Build(#[from] BuildError),
}

impl From<UniformUnsupported> for UniformError {
    fn from(what: UniformUnsupported) -> Self {
        Self::Unsupported(what)
    }
}

/// The register a 64-bit `ULDC` destination must start below (`Visitor::ULDC`: `dest < 62`).
pub const ULDC_PAIR_LIMIT: u8 = 62;

/// The register a 128-bit `ULDC` destination must start below (`Visitor::ULDC`: `dest < 60`).
pub const ULDC_QUAD_LIMIT: u8 = 60;

/// The bytes of one constant-buffer word: the distance between the words of a `.128` read.
const WORD_BYTES: u32 = source::CBUF_WORD_BYTES;

/// `ShuffleIndex`'s clamp for a whole warp (`Visitor::R2UR`: `ir.Imm32(31U)`).
const SHUFFLE_CLAMP: u32 = 31;

/// Whether `op` belongs to this group.
pub const fn in_group(op: &Op) -> bool {
    matches!(
        op,
        Op::R2UR(_)
            | Op::UP2UR(_)
            | Op::UMov(_)
            | Op::ULdc(_)
            | Op::UPLop3(_)
            | Op::UIAdd3(_)
            | Op::UIAdd3X(_)
            | Op::ULea(_)
            | Op::ULeaX(_)
            | Op::ULop3(_)
            | Op::UPrmt(_)
            | Op::UShf(_)
            | Op::UIMad(_)
            | Op::UIMad64(_)
            | Op::UISetP(_)
            | Op::USel(_)
            | Op::USgxt(_)
            | Op::UBMsk(_)
            | Op::UFlo(_)
            | Op::UBRev(_)
            | Op::UPopC(_)
    )
}

/// Decodes `word` and translates it ([`translate_uniform`]).
pub fn translate_uniform_word(
    builder: &mut Builder<'_>,
    word: Instruction,
    guard: GuardMode,
) -> Result<(), UniformError> {
    let instr = decode(word)?;
    translate_uniform(builder, &instr, guard)
}

/// Appends the IR of `instr`, a uniform datapath instruction, at `builder`'s position (see the
/// module docs for the order of the checks). With a conditional block the builder is left in
/// the block after it.
pub fn translate_uniform(
    builder: &mut Builder<'_>,
    instr: &Instr,
    guard: GuardMode,
) -> Result<(), UniformError> {
    let op = instr.op;
    let mnemonic = op.mnemonic();
    if !in_group(&op) {
        return Err(UniformError::OtherGroup(mnemonic));
    }
    let step = guard_step(instr.guard, guard == GuardMode::Tested);
    let pending = match step {
        GuardStep::Skip => return Ok(()),
        GuardStep::Unguarded => None,
        GuardStep::Conditional => Some(instr.guard),
    };
    let file = if reads_uniform_guard(&op) {
        PredFile::Uniform
    } else {
        PredFile::Vector
    };
    if let Some((vector, source_2)) = twin(&op) {
        if matches!(op, Op::UBMsk(_)) {
            return Err(UniformUnsupported::Instruction(mnemonic).into());
        }
        if matches!(source_2, Some(AluSrc::Imm(_))) {
            return Err(UniformUnsupported::ImmediateSource2(mnemonic).into());
        }
        let files = twin_files(&op, file);
        return integer::translate_in(builder, vector, pending, files)
            .map_err(|error| from_integer(error, mnemonic));
    }
    match op {
        Op::UP2UR(_) => Err(UniformUnsupported::Instruction(mnemonic).into()),
        Op::R2UR(r2ur) => translate_r2ur(builder, r2ur, pending),
        Op::UPLop3(plop3) => {
            let files = Files {
                uniform_regs: true,
                pred_in: PredFile::Uniform,
                pred_out: PredFile::Uniform,
                guard: file,
            };
            integer::translate_in(builder, Op::PLop3(plop3_of(plop3)), pending, files)
                .map_err(|error| from_integer(error, mnemonic))
        }
        Op::UMov(umov) => guarded(builder, pending, file, false, |b| translate_umov(b, umov)),
        Op::ULdc(uldc) => {
            check_uldc(uldc)?;
            let splits = matches!(uldc.src, ULdcSrc::Global { pred, .. } if !pred.reg.is_true());
            guarded(builder, pending, file, splits, |b| translate_uldc(b, uldc))
        }
        // Not reached: `in_group` and `twin` take every other operation.
        other => Err(UniformError::OtherGroup(other.mnemonic())),
    }
}

/// Whether `Visitor::Run` reads `op`'s guard as a uniform predicate: `classify.h`'s `uniform`
/// (the module docs, "Guards"). `R2UR` reads its guard as data, a predicate.
pub(crate) fn reads_uniform_guard(op: &Op) -> bool {
    match op {
        Op::UMov(_)
        | Op::UPLop3(_)
        | Op::ULdc(OpULdc {
            src: ULdcSrc::CBuf(_) | ULdcSrc::Indexed { .. },
            ..
        }) => true,
        Op::UPrmt(_) | Op::UBMsk(_) => false,
        _ => twin(op).is_some_and(|(_, source_2)| !matches!(source_2, Some(AluSrc::Imm(_)))),
    }
}

/// Source 2 of a twin, where its vector operation has one.
fn source_2<T: TwinOperands>(twin: &Uniform<T>) -> Option<AluSrc> {
    twin.op().operands().1[2]
}

/// A twin's vector operation and its source 2; `None` for an operation that is not a twin.
fn twin(op: &Op) -> Option<(Op, Option<AluSrc>)> {
    macro_rules! twins {
        ($($twin:ident => $vector:ident),* $(,)?) => {
            match op {
                $(Op::$twin(t) => Some((Op::$vector(*t.op()), source_2(t))),)*
                _ => None,
            }
        };
    }
    twins! {
        UIAdd3 => IAdd3,
        UIAdd3X => IAdd3X,
        ULea => Lea,
        ULeaX => LeaX,
        ULop3 => Lop3,
        UPrmt => Prmt,
        UShf => Shf,
        UIMad => IMad,
        UIMad64 => IMad64,
        UISetP => ISetP,
        USel => Sel,
        USgxt => Sgxt,
        UBMsk => BMsk,
        UFlo => Flo,
        UBRev => BRev,
        UPopC => PopC,
    }
}

/// The predicate files of a twin, with its guard read from `guard` (the module docs,
/// "Predicate files").
const fn twin_files(op: &Op, guard: PredFile) -> Files {
    let (pred_in, pred_out) = match op {
        Op::ULea(_) | Op::ULeaX(_) => (PredFile::Vector, PredFile::Vector),
        Op::UIMad(_) => (PredFile::Vector, PredFile::Uniform),
        _ => (PredFile::Uniform, PredFile::Uniform),
    };
    Files {
        uniform_regs: true,
        pred_in,
        pred_out,
        guard,
    }
}

/// `UPLOP3` as the `PLOP3` the integer group translates: the same truth tables, its
/// predicates by number (read from the uniform file by the files it is handed), `c` as a
/// uniform predicate.
fn plop3_of(op: OpUPLop3) -> OpPLop3 {
    let pred = |reg: crate::decode::UniformPred| Pred::new(reg.index()).unwrap_or(Pred::PT);
    let src = |src: UniformPredSrc| PredSrc {
        reg: pred(src.reg),
        negated: src.negated,
    };
    let [a, b, c] = op.srcs;
    OpPLop3 {
        dsts: op.dsts.map(pred),
        srcs: [src(a), src(b)],
        c: PLop3Src::Uniform(c),
        luts: op.luts,
    }
}

/// The integer group's refusal of a twin as this group's.
fn from_integer(error: IntegerError, mnemonic: &'static str) -> UniformError {
    match error {
        IntegerError::Unsupported(what) => UniformError::Twin(what),
        IntegerError::Modifier(_) => UniformError::Modifier(mnemonic),
        IntegerError::SuccessorPhi { block, successor } => {
            UniformError::SuccessorPhi { block, successor }
        }
        IntegerError::Build(error) => UniformError::Build(error),
        IntegerError::Decode(error) => UniformError::Decode(error),
        IntegerError::OtherGroup(_) => UniformError::OtherGroup(mnemonic),
    }
}

const fn require(ok: bool, what: UniformUnsupported) -> Result<(), UniformUnsupported> {
    if ok { Ok(()) } else { Err(what) }
}

/// A 32-bit immediate operand.
const fn imm32(bits: u32) -> Operand {
    Operand::Imm(Imm::U32(bits))
}

/// Writes `body`'s IR at `builder`'s position, as a conditional block on `pending`, read from
/// `file`, when it is set; the builder is left in the block after it. A conditional block, or
/// `splits` (the body makes one of its own), first checks that no successor of the block has a
/// phi naming it.
fn guarded(
    builder: &mut Builder<'_>,
    pending: Option<PredSrc>,
    file: PredFile,
    splits: bool,
    body: impl FnOnce(&mut Builder<'_>) -> Result<(), BuildError>,
) -> Result<(), UniformError> {
    if pending.is_some() || splits {
        let block = builder.block();
        if let Some(successor) = successor_phi(builder.function(), block) {
            return Err(UniformError::SuccessorPhi { block, successor });
        }
    }
    let Some(guard) = pending else {
        return Ok(body(builder)?);
    };
    let cond = match file {
        PredFile::Vector => source::pred(builder, guard)?,
        PredFile::Uniform => source::uniform_guard(builder, guard)?,
    };
    let join = split_for_guard(builder, cond)?;
    body(builder)?;
    builder.switch_to(join)?;
    Ok(())
}

/// Writes a uniform register; a write to `URZ` is dropped (`Visitor::UR`).
fn set(builder: &mut Builder<'_>, reg: UniformReg, value: Operand) -> Result<(), BuildError> {
    if !reg.is_zero() {
        builder.op(
            Opcode::SetUniformRegister,
            &[Operand::Imm(Imm::UniformReg(reg)), value],
        )?;
    }
    Ok(())
}

/// The uniform register `k` places after `reg`; the destination checks keep it below `URZ`.
fn next(reg: UniformReg, k: u8) -> UniformReg {
    UniformReg::new(reg.index().saturating_add(k)).unwrap_or(UniformReg::URZ)
}

/// Writes the words of a value of `words` registers from `dst` on: one word as it is, a pair
/// extracted (`CompositeExtractU32x2`).
fn set_words(
    builder: &mut Builder<'_>,
    dst: UniformReg,
    value: Operand,
    words: u8,
) -> Result<(), BuildError> {
    if words == 1 {
        return set(builder, dst, value);
    }
    for k in 0..words {
        let word = builder.op(Opcode::CompositeExtractU32x2, &[value, imm32(u32::from(k))])?;
        set(builder, next(dst, k), word.into())?;
    }
    Ok(())
}

/// `R2UR` (`Visitor::R2UR`, and the lowering's `case 0x3c2`): `pending`, the guard of an
/// untested guarded `R2UR`, is read as data (a predicate); with none, every lane takes part.
/// With a guard, the uniform register keeps its old value when the ballot is zero (no active
/// lane's guard holds), as the lowering's `SEL` on its `VOTE.ANY` result keeps it (the module
/// docs, "Divergences"); unguarded, the running lane is in the ballot, so no select is made.
fn translate_r2ur(
    builder: &mut Builder<'_>,
    op: OpR2UR,
    pending: Option<PredSrc>,
) -> Result<(), UniformError> {
    require(!op.dst.is_zero(), UniformUnsupported::R2urDestination)?;
    let active = match pending {
        Some(guard) => source::pred(builder, guard)?,
        None => Operand::Imm(Imm::U1(true)),
    };
    let ballot = builder.op(Opcode::SubgroupBallot, &[active])?;
    let negated = builder.op(Opcode::INeg32, &[ballot.into()])?;
    let lowest = builder.op(Opcode::BitwiseAnd32, &[ballot.into(), negated.into()])?;
    let lane = builder.op(Opcode::FindUMsb32, &[lowest.into()])?;
    let value = source::reg(builder, op.src)?;
    let shuffled = builder.op(
        Opcode::ShuffleIndex,
        &[value, lane.into(), imm32(SHUFFLE_CLAMP), imm32(0)],
    )?;
    let result = if pending.is_some() {
        let any = builder.op(Opcode::INotEqual, &[ballot.into(), imm32(0)])?;
        let old = source::uniform_reg(builder, op.dst)?;
        builder
            .op(Opcode::SelectU32, &[any.into(), shuffled.into(), old])?
            .into()
    } else {
        shuffled.into()
    };
    Ok(set(builder, op.dst, result)?)
}

/// `UMOV` (`Visitor::UMOV`).
fn translate_umov(builder: &mut Builder<'_>, op: OpUMov) -> Result<(), BuildError> {
    let value = match op.src {
        UMovSrc::Imm(bits) => imm32(bits),
        UMovSrc::Reg(reg) => source::uniform_reg(builder, reg)?,
    };
    set(builder, op.dst, value)
}

/// `ULDC`'s and `LDCG`'s checks, in PR #4588's order (the module docs).
fn check_uldc(op: OpULdc) -> Result<(), UniformUnsupported> {
    let words = words_of(op.mem_type);
    match op.src {
        ULdcSrc::CBuf(CBufRef { buf, offset }) => {
            require(
                matches!(buf, CBuf::Binding(_)),
                UniformUnsupported::BindlessConstantLoad,
            )?;
            require(
                offset.is_multiple_of(4),
                UniformUnsupported::ConstantLoadOffset(offset),
            )?;
        }
        ULdcSrc::Indexed { offset, .. } => {
            require(
                offset.is_multiple_of(4),
                UniformUnsupported::ConstantLoadOffset(offset),
            )?;
        }
        ULdcSrc::Global { .. } => {
            require(
                op.mem_type != MemType::B128,
                UniformUnsupported::GlobalLoad128,
            )?;
        }
    }
    let index = op.dst.index();
    let fits = match words {
        2 => index.is_multiple_of(2) && index < ULDC_PAIR_LIMIT,
        4 => index.is_multiple_of(4) && index < ULDC_QUAD_LIMIT,
        _ => true,
    };
    require(
        fits,
        UniformUnsupported::ConstantLoadDestination { dst: op.dst, words },
    )
}

/// A 16-bit offset field read as signed, as the lowering's Maxwell `LDC` reads it (the module
/// docs, "Divergences").
fn signed(offset: u16) -> u32 {
    i32::from(offset.cast_signed()).cast_unsigned()
}

/// `ULDC` and `LDCG`, checked.
fn translate_uldc(builder: &mut Builder<'_>, op: OpULdc) -> Result<(), BuildError> {
    match op.src {
        ULdcSrc::CBuf(CBufRef {
            buf: CBuf::Binding(bank),
            offset,
        }) => {
            let offset = if op.mem_type == MemType::B64 {
                signed(offset)
            } else {
                u32::from(offset)
            };
            read_cbuf(builder, op, bank, None, offset)
        }
        ULdcSrc::Indexed { index, reg, offset } => {
            // The lowering takes `0xabb` only with a 32- or 64-bit type (its Maxwell `LDC`,
            // a signed offset); the native, unsigned, reading is kept for the others.
            let offset = if matches!(op.mem_type, MemType::B32 | MemType::B64) {
                signed(offset)
            } else {
                u32::from(offset)
            };
            read_cbuf(builder, op, index, Some(reg), offset)
        }
        ULdcSrc::Global { addr, offset, pred } => translate_ldcg(builder, op, addr, offset, pred),
        // Refused by `check_uldc`.
        ULdcSrc::CBuf(_) => Ok(()),
    }
}

/// The constant-buffer read of a `ULDC` of buffer `bank` at `offset`, plus the uniform register
/// `index` for `0xabb` (`Visitor::ULDC`: `IAdd(UR(index_reg), Imm32(offset))`, `URZ` reading 0).
fn read_cbuf(
    builder: &mut Builder<'_>,
    op: OpULdc,
    bank: u8,
    index: Option<UniformReg>,
    offset: u32,
) -> Result<(), BuildError> {
    let bank = imm32(u32::from(bank));
    let base = match index {
        Some(reg) => Some(source::uniform_reg(builder, reg)?),
        None => None,
    };
    let at = |builder: &mut Builder<'_>, k: u8| -> Result<Operand, BuildError> {
        let displacement = imm32(offset.wrapping_add(u32::from(k) * WORD_BYTES));
        match base {
            Some(base) => Ok(builder.op(Opcode::IAdd32, &[base, displacement])?.into()),
            None => Ok(displacement),
        }
    };
    let words = words_of(op.mem_type);
    let Some(read) = cbuf_load(op.mem_type) else {
        for k in 0..words {
            let address = at(builder, k)?;
            let word = builder.op(Opcode::GetCbufU32, &[bank, address])?;
            set(builder, next(op.dst, k), word.into())?;
        }
        return Ok(());
    };
    let address = at(builder, 0)?;
    let value = builder.op(read, &[bank, address])?;
    set_words(builder, op.dst, value.into(), words)
}

/// `LDCG` (NAK `OpLdcg`): the 64-bit address `addr+1:addr` (`URZ` reading 0 in either word)
/// plus the signed offset is read first; then the destination gets zero where `pred` holds
/// and the loaded value where it does not, the load made only in the second case.
fn translate_ldcg(
    builder: &mut Builder<'_>,
    op: OpULdc,
    addr: UniformReg,
    offset: i32,
    pred: UniformPredSrc,
) -> Result<(), BuildError> {
    let words = words_of(op.mem_type);
    let low = source::uniform_reg(builder, addr)?;
    let high = source::uniform_reg(builder, next(addr, 1))?;
    let pair = builder.op(Opcode::CompositeConstructU32x2, &[low, high])?;
    let mut address: Operand = builder.op(Opcode::PackUint2x32, &[pair.into()])?.into();
    if offset != 0 {
        let offset = Operand::Imm(Imm::U64(i64::from(offset).cast_unsigned()));
        address = builder.op(Opcode::IAdd64, &[address, offset])?.into();
    }
    let join = if pred.reg.is_true() {
        if !pred.negated {
            for k in 0..words {
                set(builder, next(op.dst, k), imm32(0))?;
            }
            return Ok(());
        }
        None
    } else {
        for k in 0..words {
            set(builder, next(op.dst, k), imm32(0))?;
        }
        let holds = source::uniform_pred(builder, pred)?;
        let loads = builder.op(Opcode::LogicalNot, &[holds])?;
        Some(split_for_guard(builder, loads.into())?)
    };
    let value = builder.op(global_load(op.mem_type), &[address])?;
    set_words(builder, op.dst, value.into(), words)?;
    if let Some(join) = join {
        builder.switch_to(join)?;
    }
    Ok(())
}
