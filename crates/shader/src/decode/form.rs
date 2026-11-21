//! The form table and the field reader every decode form uses.
//!
//! A [`Form`] is one 12-bit opcode: the bits it can explain (its known-bit mask, PR #4588's
//! `IsaRow::known`), the bits every valid word of it has set, and the function that reads its
//! fields. Each instruction group lists its forms in its module under `op` (`FORMS`), and
//! [`GROUPS`] joins the lists; [`lookup`] indexes them by opcode through a table built at
//! compile time, as PR #4588's `IsaTable` does
//! (`eden-pr4588/src/shader_recompiler/frontend/ampere/isa.h` @ e5fb3b97). No two forms share an
//! opcode (`decode::tests::framework::forms_have_distinct_opcodes_and_lookup_finds_each`).
//!
//! For tests, `GROUP_HOOKS` joins each group's reference rows (`REFERENCE_ROWS`) and operation
//! generators (`any_ops`) the same way, from the same list of modules, so a slice adds a form
//! and its tests' data in its own module and edits no shared list.
//!
//! **Known bits and canonical words** (slice 4b; owner's ruling 33, 2026-11-19). A form's mask
//! is its PR #4588 row's, less the bits the form rejects (`ReferenceRow::dropping`) and plus the
//! bits the row leaves out that another reference names (`ReferenceRow::adding`, with a
//! `Namer` for each): NAK's 16-bit source swizzle at bit 60, for example, which PR #4588's
//! rows mark in only some forms. And a form whose bare word, its opcode and fixed bits alone,
//! holds a field value that names nothing (`F2F`'s float size 0, `FMUL`'s scale 0) gives a
//! canonical word (`ReferenceRow::with_canonical`), from which tests build its valid words, so
//! that the decoder can refuse that zero with [`DecodeError::UnknownValue`] where a reference
//! refuses it.
//!
//! The [`Reader`] records every bit a decode function reads, the way PR #4588's translator
//! records the bits it takes (`visitor.cpp`, `Visitor::Take`): after the form's function has
//! run, a set bit nobody read is rejected, so bits a form explains only in some of its layouts
//! (a bound constant buffer's index and a bindless one's register share a slot) cannot be set
//! in the other.

use super::DecodeError;
use super::fields::{COMMON, GUARD, GUARD_NOT, OPCODE};
use super::instruction::{Field, Instruction};
use super::op::uniform;
use super::op::{Op, attribute, control, float, half_double, integer, memory, tensor, texture};
use super::operand::{Gpr, Pred, PredSrc, UniformPred, UniformPredSrc, UniformReg};

/// A form's field reader.
pub(crate) type DecodeFn = fn(&mut Reader) -> Result<Op, DecodeError>;

/// One decodable 12-bit opcode.
#[derive(Clone, Copy)]
pub(crate) struct Form {
    /// The opcode, bits 0..12.
    pub(crate) opcode: u16,
    /// Every bit the form can explain, including [`COMMON`].
    pub(crate) known: u128,
    /// The bits every valid word of the form has set.
    pub(crate) fixed: u128,
    /// Reads the operation.
    pub(crate) decode: DecodeFn,
}

impl Form {
    /// A form of `opcode` explaining `fields` (a mask) besides [`COMMON`], with no fixed bits.
    pub(crate) const fn new(opcode: u16, fields: u128, decode: DecodeFn) -> Self {
        Self {
            opcode,
            known: COMMON | fields,
            fixed: 0,
            decode,
        }
    }

    /// This form with `fixed` bits that every valid word has set; they are known too.
    pub(crate) const fn with_fixed(self, fixed: u128) -> Self {
        Self {
            known: self.known | fixed,
            fixed,
            ..self
        }
    }
}

/// A reference that names a bit PR #4588's row for a form leaves out: why a form knows a bit
/// beyond its row ([`ReferenceRow::adding`]). Each such bit is a divergence from the row, which
/// the group's module docs list with the test that pins it.
#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Namer {
    /// NAK's encoder writes the bit in this form (`sm70_encode.rs` @ 52238762).
    NakEncoder,
    /// NAK's encoder names the bit for this instruction only in a comment and writes nothing
    /// there (`sm70_encode.rs` @ 52238762): a guess until an `nvdisasm` check (verification E2).
    NakComment,
    /// PR #4588's translator reads the bit in this form though the form's row leaves it out
    /// (`frontend/ampere/ops_*.cpp` @ e5fb3b97).
    Pr4588Translator,
}

#[cfg(test)]
impl Namer {
    /// Every namer, in [`Self::index`] order.
    pub(crate) const ALL: [Self; 3] = [Self::NakEncoder, Self::NakComment, Self::Pr4588Translator];

    /// The namer's position in [`Self::ALL`] and in a [`ReferenceRow`]'s added bits.
    pub(crate) const fn index(self) -> usize {
        match self {
            Self::NakEncoder => 0,
            Self::NakComment => 1,
            Self::Pr4588Translator => 2,
        }
    }
}

/// One row of PR #4588's decode table for a decoded form (`ampere_isa_table.h` @ e5fb3b97), and
/// how the form departs from it: the bits it rejects, the bits it adds (with the reference that
/// names each), the bits it requires, and the fields of its canonical word. Each group lists the
/// rows of its own forms, in its `FORMS` order, as `REFERENCE_ROWS`;
/// `decode::tests::masks_match_the_reference_table` checks them.
#[cfg(test)]
pub(crate) struct ReferenceRow {
    /// The row's opcode.
    pub(crate) opcode: u16,
    /// The row's known-bit mask: its four words, low word first.
    pub(crate) words: [u32; 4],
    /// Bits the row marks known that the form rejects: a divergence the group's module docs
    /// list and a test of its own pins.
    pub(crate) dropped: u128,
    /// Bits the row leaves out that the form knows, indexed by [`Namer::index`] of the
    /// reference that names them: a divergence the group's module docs list, which
    /// `decode::tests::framework::added_bits_are_the_listed_cases` enumerates and
    /// `decode::tests::framework::added_bits_decode_print_and_round_trip` pins.
    added: [u128; Namer::ALL.len()],
    /// The form's fixed bits, which the row can only mark known: the bits every valid word of
    /// the form has set (`Form::fixed`).
    pub(crate) fixed: u128,
    /// The fields, besides the opcode and the fixed bits, that the form's canonical word sets:
    /// none where the bare word (the opcode and the fixed bits alone) decodes, and otherwise a
    /// value for each field whose zero names nothing. Tests build each form's base word from
    /// it (`testing::canonical_word`).
    pub(crate) canonical: u128,
}

#[cfg(test)]
impl ReferenceRow {
    /// The row of `opcode` with the mask `words`, which the form matches exactly.
    pub(crate) const fn new(opcode: u16, words: [u32; 4]) -> Self {
        Self {
            opcode,
            words,
            dropped: 0,
            added: [0; Namer::ALL.len()],
            fixed: 0,
            canonical: 0,
        }
    }

    /// This row, with `dropped` bits that the form rejects.
    pub(crate) const fn dropping(self, dropped: u128) -> Self {
        Self { dropped, ..self }
    }

    /// This row, with `bits` that the row leaves out and the form knows because `by` names
    /// them. The bits of several calls accumulate.
    pub(crate) const fn adding(self, bits: u128, by: Namer) -> Self {
        let mut added = self.added;
        added[by.index()] |= bits;
        Self { added, ..self }
    }

    /// This row, with `fixed` bits that the form requires set.
    pub(crate) const fn fixing(self, fixed: u128) -> Self {
        Self { fixed, ..self }
    }

    /// This row, with the fields `canonical` that the form's canonical word sets besides its
    /// opcode and fixed bits.
    pub(crate) const fn with_canonical(self, canonical: u128) -> Self {
        Self { canonical, ..self }
    }

    /// The row's known-bit mask.
    pub(crate) fn mask(&self) -> u128 {
        self.words
            .iter()
            .rev()
            .fold(0, |bits, &word| (bits << 32) | u128::from(word))
    }

    /// The bits the form adds to the row because `by` names them.
    pub(crate) const fn added_by(&self, by: Namer) -> u128 {
        self.added[by.index()]
    }

    /// Every bit the form adds to the row, whichever reference names it.
    pub(crate) fn added(&self) -> u128 {
        self.added.iter().fold(0, |bits, &more| bits | more)
    }
}

/// The bits of `value` placed in `field`, for a row's canonical fields in a constant: what
/// `Instruction::with_field` writes into a clear word, its bits above the field's width dropped
/// (`decode::tests::framework::field_values_match_with_field`).
#[cfg(test)]
pub(crate) const fn field_value(field: Field, value: u64) -> u128 {
    let mut bits = 0;
    let mut index = 0;
    while index < field.width && index < u64::BITS {
        if (value >> index) & 1 != 0 {
            bits |= Field::bit(field.start.saturating_add(index)).mask();
        }
        index += 1;
    }
    bits
}

/// One group's test hooks, so that a slice adds its tests' data in its own module.
#[cfg(test)]
pub(crate) struct GroupHooks {
    /// PR #4588's rows for the group's forms, in `FORMS` order.
    pub(crate) reference_rows: &'static [ReferenceRow],
    /// Generators of the group's decoded operations, one per operation; none for a group with
    /// no forms. `testing::any_instr` combines every group's.
    pub(crate) any_ops: fn() -> Vec<proptest::strategy::BoxedStrategy<Op>>,
}

/// Defines [`GROUPS`] and, for tests, `GROUP_HOOKS` from one list of the group modules.
macro_rules! groups {
    ($($group:ident),* $(,)?) => {
        /// Every instruction group's forms, in slice order (6a to 6i).
        pub(crate) const GROUPS: [&[Form]; 9] = [$($group::FORMS),*];

        /// Every instruction group's test hooks, in the order of [`GROUPS`].
        #[cfg(test)]
        pub(crate) const GROUP_HOOKS: [GroupHooks; 9] = [$(GroupHooks {
            reference_rows: $group::REFERENCE_ROWS,
            any_ops: $group::any_ops,
        }),*];
    };
}

groups!(
    integer,
    float,
    half_double,
    uniform,
    memory,
    texture,
    attribute,
    control,
    tensor,
);

/// How many 12-bit opcodes there are.
const OPCODES: usize = 1 << 12;
/// An index entry with no form.
const NO_FORM: u16 = u16::MAX;
/// The shift of the group number in an index entry; the form's position is below it.
const GROUP_SHIFT: u32 = 12;

/// The form of each opcode, as `group << GROUP_SHIFT | position`, or [`NO_FORM`]. A second form
/// with the same opcode, or a group longer than `1 << GROUP_SHIFT` forms, is left out (the tests
/// rule both out).
static INDEX: [u16; OPCODES] = build_index();

const fn build_index() -> [u16; OPCODES] {
    let mut index = [NO_FORM; OPCODES];
    // `u16` counters, widened to index: no narrowing cast is needed to pack an entry.
    let mut group: u16 = 0;
    while (group as usize) < GROUPS.len() {
        let forms = GROUPS[group as usize];
        let mut position: u16 = 0;
        while (position as usize) < forms.len() && position < (1 << GROUP_SHIFT) {
            let opcode = forms[position as usize].opcode as usize;
            if opcode < OPCODES && index[opcode] == NO_FORM {
                index[opcode] = (group << GROUP_SHIFT) | position;
            }
            position += 1;
        }
        group += 1;
    }
    index
}

/// The form of a 12-bit opcode, if any.
pub(crate) fn lookup(opcode: u16) -> Option<&'static Form> {
    let entry = *INDEX.get(usize::from(opcode))?;
    let group = GROUPS.get(usize::from(entry >> GROUP_SHIFT))?;
    group.get(usize::from(entry & ((1 << GROUP_SHIFT) - 1)))
}

/// Reads fields of one word and records which bits were read.
///
/// A register, uniform register or predicate field read through [`Reader::uniform_reg`],
/// [`Reader::pred_src`] or [`Reader::uniform_pred_src`] whose value names no register (only a
/// field wider than the register file's, which no form passes) is not turned into `URZ` or
/// `UPT`: the reader keeps the first such value as a [`DecodeError::UnknownValue`]
/// ([`Reader::fault`]), which `decode` reports after the form's function has run (slice 6d,
/// from the review of slice 4: "check the width").
pub(crate) struct Reader {
    word: Instruction,
    read: u128,
    /// Whether per-thread register fields name uniform registers ([`Reader::uniform_datapath`]).
    uniform: bool,
    /// The first register field whose value names no register.
    fault: Option<DecodeError>,
}

impl Reader {
    /// A reader of `word` that counts [`COMMON`] as read.
    pub(crate) const fn new(word: Instruction) -> Self {
        Self {
            word,
            read: COMMON,
            uniform: false,
            fault: None,
        }
    }

    /// Reads every later [`Reader::gpr`] field as the uniform datapath encodes it: the low
    /// [`UniformReg::BITS`] bits of the field, a uniform register's number, with `URZ` (63)
    /// read as [`Gpr::RZ`] (`op::uniform`, whose twins reuse the vector decoders).
    pub(crate) const fn uniform_datapath(&mut self) {
        self.uniform = true;
    }

    /// The first register field whose value named no register, as the error `decode` reports.
    pub(crate) const fn fault(&self) -> Option<DecodeError> {
        self.fault
    }

    /// Keeps `field` holding `value`, which names no register, as the fault unless one is kept
    /// already.
    fn refuse(&mut self, field: Field, value: u64) {
        if self.fault.is_none() {
            self.fault = Some(self.unknown_value(field, value));
        }
    }

    /// The 12-bit opcode (always read).
    pub(crate) fn opcode(&self) -> u64 {
        self.word.field(OPCODE)
    }

    /// Every bit read so far, [`COMMON`] included.
    #[cfg(test)]
    pub(crate) const fn read(&self) -> u128 {
        self.read
    }

    /// The set bits nothing has read.
    pub(crate) const fn unread(&self) -> u128 {
        self.word.bits() & !self.read
    }

    /// Reads `field`.
    pub(crate) fn field(&mut self, field: Field) -> u64 {
        self.read |= field.mask();
        self.word.field(field)
    }

    /// Reads a one-bit field.
    pub(crate) fn flag(&mut self, field: Field) -> bool {
        self.field(field) != 0
    }

    /// Reads a field of at most eight bits. Cannot truncate for such a field; a wider one
    /// saturates.
    pub(crate) fn byte(&mut self, field: Field) -> u8 {
        u8::try_from(self.field(field)).unwrap_or(u8::MAX)
    }

    /// Reads a field of at most 32 bits; a wider one saturates.
    pub(crate) fn word32(&mut self, field: Field) -> u32 {
        u32::try_from(self.field(field)).unwrap_or(u32::MAX)
    }

    /// Reads a field as a two's-complement number of its width (`width` at most 64).
    pub(crate) fn signed(&mut self, field: Field) -> i64 {
        self.field(field);
        self.word.signed(field)
    }

    /// Reads a register field ([`Gpr::BITS`] wide). After [`Reader::uniform_datapath`], reads
    /// the field's low [`UniformReg::BITS`] bits instead, `URZ` as [`Gpr::RZ`].
    pub(crate) fn gpr(&mut self, field: Field) -> Gpr {
        if self.uniform {
            let reg = self.uniform_reg(Field::new(field.start, UniformReg::BITS));
            return if reg.is_zero() {
                Gpr::RZ
            } else {
                Gpr::new(reg.index())
            };
        }
        Gpr::new(self.byte(field))
    }

    /// Reads a uniform register field ([`UniformReg::BITS`] wide, so every value names a
    /// register; a wider field's value past `URZ` is the reader's fault).
    pub(crate) fn uniform_reg(&mut self, field: Field) -> UniformReg {
        let value = self.field(field);
        let reg = u8::try_from(value).ok().and_then(UniformReg::new);
        reg.unwrap_or_else(|| {
            self.refuse(field, value);
            UniformReg::URZ
        })
    }

    /// Reads a predicate field ([`Pred::BITS`] wide; a wider field's value past `PT` is the
    /// reader's fault) and its negation bit.
    pub(crate) fn pred_src(&mut self, reg: Field, not: Field) -> PredSrc {
        let value = self.field(reg);
        let pred = u8::try_from(value).ok().and_then(Pred::new);
        PredSrc {
            reg: pred.unwrap_or_else(|| {
                self.refuse(reg, value);
                Pred::PT
            }),
            negated: self.flag(not),
        }
    }

    /// Reads a uniform predicate field ([`UniformPred::BITS`] wide; a wider field's value past
    /// `UPT` is the reader's fault).
    pub(crate) fn uniform_pred(&mut self, field: Field) -> UniformPred {
        let value = self.field(field);
        let pred = u8::try_from(value).ok().and_then(UniformPred::new);
        pred.unwrap_or_else(|| {
            self.refuse(field, value);
            UniformPred::UPT
        })
    }

    /// Reads a uniform predicate field ([`UniformPred::BITS`] wide; a wider field's value past
    /// `UPT` is the reader's fault) and its negation bit.
    pub(crate) fn uniform_pred_src(&mut self, reg: Field, not: Field) -> UniformPredSrc {
        UniformPredSrc {
            reg: self.uniform_pred(reg),
            negated: self.flag(not),
        }
    }

    /// Reads the guard predicate.
    pub(crate) fn guard(&mut self) -> PredSrc {
        self.pred_src(GUARD, GUARD_NOT)
    }

    /// The error for `field` holding `value`, which names nothing in this word's form.
    pub(crate) fn unknown_value(&self, field: Field, value: u64) -> DecodeError {
        DecodeError::UnknownValue {
            // Cannot fail: the field is twelve bits wide.
            opcode: u16::try_from(self.opcode()).unwrap_or(u16::MAX),
            bit: field.start,
            width: field.width,
            value,
        }
    }
}
