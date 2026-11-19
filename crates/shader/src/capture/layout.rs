//! The instruction layout facts capture reads before full decoding (spec, "Instruction layout
//! facts used before full decoding"): the opcode (bits 0..12), the guard predicate (bits 12..16)
//! and the relative branch offsets (from bit 34), plus the few control-instruction bits
//! `ProgramCaptureState::Observe` compares.
//!
//! Every position is named once here, or in the decoder where it already is ([`OPCODE`],
//! [`GUARD`], [`GUARD_NOT`] and `EXIT`'s fields), and shared by the scan and the tests' encoder.
//! The values are PR #4588's: `eden-pr4588/src/shader_recompiler/frontend/ampere/
//! program_capture.h` @ e5fb3b97 (`Observe`), `instruction.h` (`RelativeOffset48`) and the
//! opcode rows of `ampere_isa_table.h`. The decoder slice for control instructions (6h) decodes
//! these instructions fully; it should take its opcodes and offset fields from here, or this
//! module from it, so each stays named once.

use crate::decode::{
    EXIT_KEEP_REF_COUNT, EXIT_NO_AT_EXIT, EXIT_OPCODE, EXIT_PRED, EXIT_PRED_NOT, Field, GUARD,
    GUARD_NOT, Instruction, OPCODE, Pred,
};

/// `CALL` (`ampere_isa_table.h`: `{0x944, {"CALL", ..}}`; `Observe`: `opcode == 0x944`).
pub(crate) const CALL_OPCODE: u16 = 0x944;
/// `BSSY` (`ampere_isa_table.h`: `{0x945, {"BSSy", ..}}`; `Observe`: `opcode == 0x945`).
pub(crate) const BSSY_OPCODE: u16 = 0x945;
/// `BRA` (`ampere_isa_table.h`: `{0x947, {"Bra", ..}}`; `Observe`: `opcode == 0x947`).
pub(crate) const BRA_OPCODE: u16 = 0x947;
/// `RET` (`ampere_isa_table.h`: `{0x950, {"RET", ..}}`; `Observe`: `opcode == 0x950`).
pub(crate) const RET_OPCODE: u16 = 0x950;

/// The low two bits of a relative branch's byte displacement. `Observe` reads the displacement
/// of `CALL` and `BRA` as bits 32..82 whole (`words[1] | (words[2] & 0x3ffff) << 32`) and of
/// `BSSY` as bits 32..64 (`static_cast<s32>(words[1])`), so these bits are part of it; a
/// displacement with either set is not a whole instruction and is refused. `BRA.DIV` keeps two
/// mode bits here instead ([`BRA_DIV_LOW`]).
pub(crate) const BRANCH_LOW: Field = Field::new(32, 2);
/// The relative offset of `CALL` and `BRA`: a signed count of 32-bit words, 48 bits from bit 34
/// (`instruction.h`, `RelativeOffset48`: `words[1] & ~3U | (words[2] & 0x3ffffU) << 32`, sign
/// extended from bit 49 of that value).
pub(crate) const BRANCH_OFFSET: Field = Field::new(34, 48);
/// `BSSY`'s relative offset: a signed count of 32-bit words, the 30 bits from bit 34 to the end
/// of the second word (`Observe`: `static_cast<s32>(instruction.words[1])`).
pub(crate) const BSSY_OFFSET: Field = Field::new(34, 30);
/// The bits from the end of [`BRANCH_OFFSET`] to the end of the third word, which `Observe`
/// compares whole to recognise `BRA.DIV` (`(instruction.words[2] & ~0x3ffffU) == 0x0b800000U`).
pub(crate) const BRANCH_MODE: Field = Field::new(82, 14);
/// [`BRANCH_MODE`] of a `BRA.DIV`: the branch's own predicate `PT` (bits 87..90) and the
/// uniform-register bit 91 (`Observe`: "bit 91 and a uniform register"), as `0x0b800000` of the
/// third word shifted down to [`BRANCH_MODE`]'s start.
pub(crate) const BRA_DIV_MODE: u64 = 0x0b80_0000 >> (BRANCH_MODE.start - 64);
/// [`BRANCH_LOW`] of a `BRA.DIV`: both mode bits set (`(instruction.words[1] & 3) == 3`), which
/// `Observe` subtracts from the displacement (`- (divergence ? 3 : 0)`).
pub(crate) const BRA_DIV_LOW: u64 = 3;

/// Bytes per unit of a relative offset: offsets count 32-bit words (`instruction.h`: "expressed
/// in 32-bit words").
pub(crate) const OFFSET_UNIT: i64 = 4;

/// The instruction's 12-bit opcode.
pub(crate) fn opcode(word: Instruction) -> u16 {
    // Cannot fail: the field is 12 bits wide.
    u16::try_from(word.field(OPCODE)).unwrap_or(u16::MAX)
}

/// Whether the predicate in `pred` and `not` is `PT` and not negated: the four bits read 7, the
/// value `Observe` compares (`Bits(12, 4) == 7`, `Bits(87, 4) == 7`).
fn is_always(word: Instruction, pred: Field, not: Field) -> bool {
    word.field(pred) == u64::from(Pred::PT.index()) && !word.flag(not)
}

/// Whether the instruction's guard is `PT`, not negated: it always runs (`trap_guards.h`: 7
/// means always, bit 3 negates).
pub(crate) fn always_runs(word: Instruction) -> bool {
    is_always(word, GUARD, GUARD_NOT)
}

/// Whether the instruction is an `EXIT` that ends every thread reaching it: guard and own
/// predicate `PT`, and bits 84 and 85 clear (`Observe`: `opcode == 0x94d && Bits(12, 4) == 7 &&
/// Bits(87, 4) == 7 && !Bits(84, 2)`).
pub(crate) fn is_final_exit(word: Instruction) -> bool {
    opcode(word) == EXIT_OPCODE
        && always_runs(word)
        && is_always(word, EXIT_PRED, EXIT_PRED_NOT)
        && !word.flag(EXIT_KEEP_REF_COUNT)
        && !word.flag(EXIT_NO_AT_EXIT)
}

/// Whether the instruction is a `RET` whose guard is `PT` (`Observe`: `opcode == 0x950 &&
/// Bits(12, 4) == 7`). Its other bits, the register it returns through among them, are not
/// read (row 88: the link register may be restored through code in between).
pub(crate) fn is_final_ret(word: Instruction) -> bool {
    opcode(word) == RET_OPCODE && always_runs(word)
}

/// The byte displacement of a relative `CALL`, `BSSY` or `BRA`, from the end of the instruction,
/// or `None` for any other opcode (`Observe`'s `relative`).
pub(crate) fn branch_displacement(word: Instruction) -> Option<i64> {
    let opcode = opcode(word);
    let low = word.field(BRANCH_LOW);
    // Cannot fail: the field is two bits wide.
    let low = i64::try_from(low).unwrap_or(0);
    let words = match opcode {
        BSSY_OPCODE => word.signed(BSSY_OFFSET),
        CALL_OPCODE | BRA_OPCODE => word.signed(BRANCH_OFFSET),
        _ => return None,
    };
    let divergence = opcode == BRA_OPCODE
        && word.field(BRANCH_MODE) == BRA_DIV_MODE
        && word.field(BRANCH_LOW) == BRA_DIV_LOW;
    // Never saturates: the offset is at most 48 bits, so four times it plus three fits.
    let bytes = words.saturating_mul(OFFSET_UNIT);
    Some(if divergence {
        bytes
    } else {
        bytes.saturating_add(low)
    })
}
