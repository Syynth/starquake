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
//! uniform register), `FFMA` in all seven, `ALD`, `AST` and `S2R`; slice 6h's control and
//! warp group (`op::control`), every instruction of it but `JMX` and `BPT`; slice 6g's
//! attribute and system-value group (`op::attribute`), every instruction of it, `S2R` grown to
//! its uniform form `S2UR`; slice 6b's 32-bit float and conversion group (`op::float`), every
//! instruction of it in the vector datapath's forms; slice 6a's integer group
//! (`op::integer`), every instruction of it in the vector datapath's forms; slice 6e's memory
//! group (`op::memory`), every instruction of it in the vector datapath's forms; and slice 6f's
//! texture and surface group (`op::texture`), every instruction of it, with PR #4588's `.UR`
//! texture forms. Every other opcode is [`DecodeError::UnknownOpcode`].
//!
//! **Strict decoding** (owner's decision 4, 2026-11-12): a word with a bit set that its form
//! does not explain is rejected, never decoded with the bit ignored. The decoder checks, in this
//! order, and reports the first failure:
//!
//! 1. the opcode has a form, else [`DecodeError::UnknownOpcode`];
//! 2. no bit outside the form's known-bit mask is set, else [`DecodeError::UnknownBit`] naming
//!    the lowest such bit, as PR #4588's `FirstUnknownBit` does (`frontend/ampere/isa.h`);
//! 3. every fixed bit of the form is set (bit 91 of the uniform-register ALU forms; bits 87..89
//!    and 91 of the uniform `BRA` and bit 91 of `BRXU`, `op::control`; bit 91 of `LDTRAM`,
//!    `op::attribute`; the integer group's fixed predicate fields, `op::integer`; bit 91 of the
//!    bindless `LDC` and the bits PR #4588 requires set in `LDGSTS` and `ATOMS.POPC`,
//!    `op::memory`; bit 59 of the `.B` texture forms and bit 91 of the `.UR` ones,
//!    `op::texture`), else
//!    [`DecodeError::MissingBit`] naming the lowest clear one;
//! 4. the form's function reads its fields, else [`DecodeError::UnknownValue`] for a field whose
//!    value names nothing (a vote or reduction operation past the defined ones: `op::control`;
//!    a `MUFU` function or a comparison's combine operation past them, `FMUL`'s scale 0 and a
//!    conversion's float size 0: `op::float`; an
//!    interpolation mode, pixel value or stage-buffer type past NAK's: `op::attribute`;
//!    `ISETP`'s combining operation 3 and `PRMT`'s mode 7: `op::integer`; a memory type, order,
//!    eviction priority, atomic type or operation, cache operation or `MEMBAR` scope past the
//!    ones the references name: `op::memory`; the exact set the texture group refuses, which
//!    includes memory orders in the holes between NAK's values as well as past them, `TLD4`'s
//!    offset mode 3 and `TXQ`'s query 3, is listed in `op::texture`'s module docs under "Values
//!    the decoder refuses");
//! 5. every set bit was read, else [`DecodeError::UnknownBit`] naming the lowest unread one. A
//!    form explains some bits only in some of its layouts or operations: in the constant-buffer
//!    ALU forms a bound buffer reads its index (bits 54..59) and a bindless one its uniform
//!    register (bits 32..38), and `REDUX` reads its signed bit (73) only for `MIN` and `MAX`
//!    (`op::control`), and the high-half and byte selects and `F2FP`'s `.H1` are read only for
//!    the source widths and modes that have them (`op::float`), so a word with a bit the decoded
//!    layout or operation does not read fails here.
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
//! `tests::mov_rejects_source_modifier_bits`, `tests::framework::fixed_bits_are_required`,
//! `tests::unknown_opcodes_name_the_opcode` and `tests::masks_match_the_reference_table`. The
//! control group's divergences (the same two for `WARPSYNC` and `NANOSLEEP`, `BAR`'s form
//! `0x31d`, and the fixed bits of the uniform `BRA` and of `BRXU`) are listed in `op::control`'s
//! module docs and pinned in `tests::control`. The attribute group's (`OUT`'s forms 2, 3 and 7,
//! and its sources' modifier bits) are listed in `op::attribute`'s module docs and pinned in
//! `tests::attribute`; the float group's (the forms NAK never encodes, the modifier bits of
//! slots without modifiers, and the high-half bit 60 its forms add) in `op::float`'s and pinned
//! in `tests::float32`; the integer
//! group's (the absolute-value and modifier bits its sources do not have, forms 2, 3 and 7 of
//! the instructions without source 2, and the bits PR #4588's translator requires clear) in
//! `op::integer`'s, pinned in `tests::integer`; the memory group's (bit 91 of the forms
//! without a uniform register, `ATOMG` `0x3a8`'s uniform-register bits, and the bits PR #4588's
//! translator requires clear or set in `LDGSTS` and `ATOMS.POPC`) in `op::memory`'s, pinned in
//! `tests::memory`; the texture group's (bit 89 of `TLD4.UR`, the surface forms with a
//! constant-buffer handle left unknown, and the `.F16` and `.CL` bits its `TEX` and `TLD` forms
//! add) in `op::texture`'s, pinned in `tests::texture`.
//!
//! **Known bits and canonical words** (slice 4b; owner's ruling 33, 2026-11-19). A form may know
//! a bit its PR #4588 row leaves out when another reference names it (NAK's encoder, a comment
//! in NAK's encoder, or PR #4588's own translator), each such bit listed with its reference in
//! the group's rows and divergence table (`form`'s module docs;
//! `tests::framework::added_bits_are_the_listed_cases`). And a form whose bare word (the opcode
//! and fixed bits alone) holds a field value that names nothing gives a canonical word, so the
//! decoder refuses that value at step 4 where a reference refuses it
//! (`tests::framework::canonical_words_decode`).

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

// `AluForm` is used by the translator; `Slot` and `Source` only by tests and the assembler.
pub(crate) use alu::AluForm;
#[cfg(any(test, feature = "test-support"))]
pub(crate) use alu::{Slot, Source};
pub(crate) use fields::*;
#[cfg(test)]
pub(crate) use form::{Form, GROUP_HOOKS, GROUPS, ReferenceRow};
pub use instruction::{Field, Instruction, LengthError};
// The control-instruction opcodes and fields program capture reads before full decoding
// (`crate::capture`), named once in `op::control`.
pub(crate) use op::control::{
    BRA_OPCODE, BRANCH_OFFSET, BSSY_OFFSET, BSSY_OPCODE, CALL_OPCODE, COND, COND_NOT,
    EXIT_KEEP_REF_COUNT, EXIT_NO_AT_EXIT, EXIT_OPCODE, RET_OPCODE,
};
// The opcodes flow analysis classifies (`crate::flow`, slice 10), named once in their groups.
pub(crate) use op::attribute::{ALD_OPCODE, CS2R_OPCODE, IPA_OPCODE, S2R_OPCODE, S2UR_OPCODE};
pub(crate) use op::control::{
    BAR_OPCODE, BRA_UNIFORM_OPCODE, BREAK_OPCODE, BRX_OPCODE, BRXU_OPCODE, BSYNC_OPCODE,
    DEPBAR_OPCODE, KILL_OPCODE, NOP_OPCODE, VOTEU_OPCODE, WARPSYNC_OPCODE,
};
pub(crate) use op::float::{
    F2F_OPCODE, F2FP_OPCODE, F2I_OPCODE, FADD_OPCODE, FFMA_OPCODE, FMNMX_OPCODE, FMUL_OPCODE,
    FRND_OPCODE, FSEL_OPCODE, FSET_OPCODE, FSETP_OPCODE, FSWZADD_OPCODE, I2F_OPCODE, I2FP_OPCODE,
    MUFU_OPCODE,
};
pub(crate) use op::integer::{
    BREV_OPCODE, FLO_OPCODE, IABS_OPCODE, IADD3_OPCODE, IMAD_HI_OPCODE, IMAD_OPCODE,
    IMAD_WIDE_OPCODE, IMNMX_OPCODE, ISETP_OPCODE, LEA_OPCODE, LOP3_OPCODE, MOV_OPCODE, P2R_OPCODE,
    PLOP3_OPCODE, POPC_OPCODE, PRMT_OPCODE, R2P_OPCODE, SEL_OPCODE, SGXT_OPCODE, SHF_OPCODE,
};
#[cfg(any(test, feature = "test-support"))]
pub(crate) use op::texture;
pub use op::*;
#[cfg(any(test, feature = "test-support"))]
pub(crate) use op::{Named, attribute, control, float, integer, memory, op_list};
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
    /// A field holds a value that names nothing in the opcode's form.
    #[error(
        "opcode {opcode:#05x} has the value {value} in its {width}-bit field at bit {bit}, which \
         its form does not define"
    )]
    UnknownValue {
        /// Bits 0..12.
        opcode: u16,
        /// The field's first bit.
        bit: u32,
        /// The field's width.
        width: u32,
        /// The value read.
        value: u64,
    },
}

/// Decodes one instruction.
pub fn decode(word: Instruction) -> Result<Instr, DecodeError> {
    let (opcode, instr, reader) = read(word)?;
    let unread = reader.unread();
    if unread != 0 {
        return Err(DecodeError::UnknownBit {
            opcode,
            bit: unread.trailing_zeros(),
        });
    }
    Ok(instr)
}

/// Steps 1 to 4 of [`decode`]: the opcode, the instruction the form's function reads, and the
/// reader, which knows the bits read. [`decode`] adds step 5; the tests' `bits_read` runs these
/// same steps, so its oracle cannot drift from the decoder's checks.
fn read(word: Instruction) -> Result<(u16, Instr, Reader), DecodeError> {
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
    Ok((opcode, Instr { guard, op }, reader))
}

/// The bits the form's function reads from `word`, when steps 1 to 4 of [`decode`] accept it:
/// what step 5 checks against.
#[cfg(test)]
pub(crate) fn bits_read(word: Instruction) -> Option<u128> {
    read(word).ok().map(|(_, _, reader)| reader.read())
}
