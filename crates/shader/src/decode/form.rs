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

/// One row of PR #4588's decode table for a decoded form (`ampere_isa_table.h` @ e5fb3b97), and
/// the bits of it the form rejects. Each group lists the rows of its own forms, in its `FORMS`
/// order, as `REFERENCE_ROWS`; `decode::tests::masks_match_the_reference_table` checks them.
#[cfg(test)]
pub(crate) struct ReferenceRow {
    /// The row's opcode.
    pub(crate) opcode: u16,
    /// The row's known-bit mask: its four words, low word first.
    pub(crate) words: [u32; 4],
    /// Bits the row marks known that the form rejects: a divergence the group's module docs
    /// list and a test of its own pins.
    pub(crate) dropped: u128,
}

#[cfg(test)]
impl ReferenceRow {
    /// The row of `opcode` with the mask `words`, which the form matches exactly.
    pub(crate) const fn new(opcode: u16, words: [u32; 4]) -> Self {
        Self {
            opcode,
            words,
            dropped: 0,
        }
    }

    /// This row, with `dropped` bits that the form rejects.
    pub(crate) const fn dropping(self, dropped: u128) -> Self {
        Self { dropped, ..self }
    }

    /// The row's known-bit mask.
    pub(crate) fn mask(&self) -> u128 {
        self.words
            .iter()
            .rev()
            .fold(0, |bits, &word| (bits << 32) | u128::from(word))
    }
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
pub(crate) struct Reader {
    word: Instruction,
    read: u128,
}

impl Reader {
    /// A reader of `word` that counts [`COMMON`] as read.
    pub(crate) const fn new(word: Instruction) -> Self {
        Self { word, read: COMMON }
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
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "the memory and control slices' offsets are its first users"
        )
    )]
    pub(crate) fn signed(&mut self, field: Field) -> i64 {
        let raw = self.field(field);
        let unused = 64_u32.saturating_sub(field.width);
        // Shift the sign bit to bit 63 and back, arithmetically; a width of 0 reads 0.
        let value = raw.checked_shl(unused).unwrap_or(0).cast_signed();
        value.checked_shr(unused).unwrap_or(0)
    }

    /// Reads a register field ([`Gpr::BITS`] wide).
    pub(crate) fn gpr(&mut self, field: Field) -> Gpr {
        Gpr::new(self.byte(field))
    }

    /// Reads a uniform register field ([`UniformReg::BITS`] wide, so every value names a
    /// register).
    pub(crate) fn uniform_reg(&mut self, field: Field) -> UniformReg {
        UniformReg::new(self.byte(field)).unwrap_or(UniformReg::URZ)
    }

    /// Reads a predicate field ([`Pred::BITS`] wide) and its negation bit.
    pub(crate) fn pred_src(&mut self, reg: Field, not: Field) -> PredSrc {
        PredSrc {
            // Cannot fail for a three-bit field.
            reg: Pred::new(self.byte(reg)).unwrap_or(Pred::PT),
            negated: self.flag(not),
        }
    }

    /// Reads a uniform predicate field ([`UniformPred::BITS`] wide) and its negation bit.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "the uniform datapath slice (6d) is its first user"
        )
    )]
    pub(crate) fn uniform_pred_src(&mut self, reg: Field, not: Field) -> UniformPredSrc {
        UniformPredSrc {
            // Cannot fail for a field of `UniformPred::BITS`.
            reg: UniformPred::new(self.byte(reg)).unwrap_or(UniformPred::UPT),
            negated: self.flag(not),
        }
    }

    /// Reads the guard predicate.
    pub(crate) fn guard(&mut self) -> PredSrc {
        self.pred_src(GUARD, GUARD_NOT)
    }
}
