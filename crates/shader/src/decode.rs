//! The SM86 decoder: one 128-bit instruction word to an [`Instr`].
//!
//! This is the decoder framework of the spec's port slice 4, which the instruction-group slices
//! (6a to 6i) fill in:
//!
//! - the operand types every group shares (`operand`): per-thread and uniform registers,
//!   predicates and uniform predicates, constant-buffer references, ALU sources with their
//!   modifier bits, immediates read as unsigned or two's-complement fields;
//! - the operand forms of ALU instructions (`alu`), the slot rules sources 1 and 2 follow;
//! - the [`Op`] enum with one variant per instruction of slices 6a to 6i (`op`), each group in
//!   its own module, the instructions not decoded yet as shells;
//! - the form table (`form`): per 12-bit opcode, the bits the form can explain, the bits it
//!   always has set, and the function that reads it;
//! - the printer (`print` and each operation's `Display`), and the test assembler (`asm`,
//!   feature `test-support`), which builds words from [`Instr`] values.
//!
//! The forms decoded so far are slice 1's six instructions, grown to every operand form NAK
//! encodes for them: `MOV` in forms 1, 4, 5 and 6 (register, immediate, constant buffer,
//! uniform register), `FFMA` in all seven, `ALD`, `AST`, `EXIT` and `S2R`. Every other opcode is
//! [`DecodeError::UnknownOpcode`].
//!
//! **Strict decoding** (owner's decision 4, 2026-11-12): a word with a bit set that its form
//! does not explain is rejected, never decoded with the bit ignored. The decoder checks, in this
//! order, and reports the first failure:
//!
//! 1. the opcode has a form, else [`DecodeError::UnknownOpcode`];
//! 2. no bit outside the form's known-bit mask is set, else [`DecodeError::UnknownBit`] naming
//!    the lowest such bit, as PR #4588's `FirstUnknownBit` does (`frontend/ampere/isa.h`);
//! 3. every fixed bit of the form is set (bit 91 of the uniform-register ALU forms), else
//!    [`DecodeError::MissingBit`] naming the lowest clear one;
//! 4. the form's function reads its fields;
//! 5. every set bit was read, else [`DecodeError::UnknownBit`] naming the lowest unread one. A
//!    form explains some bits only in some of its layouts: in the constant-buffer ALU forms a
//!    bound buffer reads its index (bits 54..59) and a bindless one its uniform register (bits
//!    32..38), so a word with the other layout's bits set fails here.
//!
//! Step 5 is what lets an accepted word re-assemble to itself, scheduling bits aside
//! (`tests::decode_then_assemble_round_trips` checks it on generated words of every form). PR #4588's table lets those layout bits through
//! (its rows are one mask per opcode); its translator rejects them, as it records the bits it
//! takes (`visitor.cpp`, `Visitor::Take`). The scheduling bits 105..128 are part of every mask
//! and are not decoded: they encode stalls and dependency barriers, which NAK computes for
//! timing only (`nak/calc_instr_deps.rs`). That they never change a result is a guess (spec,
//! "Instruction layout facts used before full decoding").
//!
//! Divergences from PR #4588's table: `MOV` rejects bits 62 and 63 (see `op::integer`); the
//! uniform-register forms require bit 91 (the table marks it known; the PR's translator
//! requires it set, `Visitor::WideSlot`: "native uniform source flag"); `MOV`'s forms 2, 3 and
//! 7, which the table has rows for, are unknown opcodes here (see `op::integer`). Pinned by
//! `tests::mov_rejects_source_modifier_bits`, `tests::framework::uniform_forms_require_bit_91`,
//! `tests::unknown_opcodes_name_the_opcode` and `tests::masks_match_the_reference_table`.

mod alu;
mod fields;
mod form;
mod instruction;
mod op;
mod operand;
mod print;
#[cfg(test)]
mod tests;

use thiserror::Error;

#[cfg(any(test, feature = "test-support"))]
pub(crate) use alu::{AluForm, Slot, Source};
pub(crate) use fields::*;
#[cfg(test)]
pub(crate) use form::GROUPS;
pub use instruction::{Field, Instruction, LengthError};
pub use op::*;
#[cfg(any(test, feature = "test-support"))]
pub(crate) use op::{Named, attribute, control, float, integer, op_list};
pub use operand::{
    AluSrc, CBuf, CBufRef, Gpr, Pred, PredSrc, RoundMode, UniformPred, UniformPredSrc, UniformReg,
};

use form::{Reader, lookup};

/// Why a word is not a known instruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum DecodeError {
    /// The 12-bit opcode is not one of the forms this decoder knows.
    #[error("unknown opcode {opcode:#05x}")]
    UnknownOpcode {
        /// Bits 0..12.
        opcode: u16,
    },
    /// A bit is set that the opcode's form does not explain.
    #[error("opcode {opcode:#05x} has bit {bit} set, which its form does not use")]
    UnknownBit {
        /// Bits 0..12.
        opcode: u16,
        /// The lowest unexplained set bit.
        bit: u32,
    },
    /// A bit that every word of the opcode's form has set is clear.
    #[error("opcode {opcode:#05x} has bit {bit} clear, which its form requires set")]
    MissingBit {
        /// Bits 0..12.
        opcode: u16,
        /// The lowest clear required bit.
        bit: u32,
    },
}

/// Decodes one instruction.
pub fn decode(word: Instruction) -> Result<Instr, DecodeError> {
    // Cannot fail: the field is 12 bits wide.
    let opcode = u16::try_from(word.field(OPCODE)).unwrap_or(u16::MAX);
    let form = lookup(opcode).ok_or(DecodeError::UnknownOpcode { opcode })?;
    let stray = word.bits() & !form.known;
    if stray != 0 {
        return Err(DecodeError::UnknownBit {
            opcode,
            bit: stray.trailing_zeros(),
        });
    }
    let missing = form.fixed & !word.bits();
    if missing != 0 {
        return Err(DecodeError::MissingBit {
            opcode,
            bit: missing.trailing_zeros(),
        });
    }
    let mut reader = Reader::new(word);
    let guard = reader.guard();
    let op = (form.decode)(&mut reader)?;
    let unread = reader.unread();
    if unread != 0 {
        return Err(DecodeError::UnknownBit {
            opcode,
            bit: unread.trailing_zeros(),
        });
    }
    Ok(Instr { guard, op })
}

/// The bits the form's function reads from `word`, when `word` has a form, no bit outside its
/// mask and every fixed bit, and the function accepts it: what step 5 checks against.
#[cfg(test)]
pub(crate) fn bits_read(word: Instruction) -> Option<u128> {
    let opcode = u16::try_from(word.field(OPCODE)).ok()?;
    let form = lookup(opcode)?;
    if word.bits() & !form.known != 0 || form.fixed & !word.bits() != 0 {
        return None;
    }
    let mut reader = Reader::new(word);
    reader.guard();
    (form.decode)(&mut reader).ok()?;
    Some(reader.read())
}
