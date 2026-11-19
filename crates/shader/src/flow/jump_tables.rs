//! Jump tables: finding the table of each `BRX`, asking for its words, and resolving its
//! targets (PR #4588's `TrackCaptureIndirectBranches` and the reads of
//! `ResolveCaptureIndirectBranches`).
//!
//! Ported from `eden-pr4588/src/shader_recompiler/frontend/ampere/program_capture.h` @ e5fb3b97
//! (GPL-3.0-or-later, Copyright 2026 Eden Emulator Project). The extension itself is slice 3's
//! ([`Capture::resolve_jump_tables`]); this module finds the tables that call takes, asks for
//! their words, and reads the tables again after an extension.
//!
//! # The selector pattern
//!
//! A `BRX Rb` that counts from the program's start (guard `PT`, own predicate `PT`, offset
//! back to instruction 0) is tracked when, searching back from it:
//!
//! 1. the nearest `LDC.32 Rb, c[bank][Rs + offset]` (unguarded, no other bits in its second and
//!    third words, the offset a multiple of four) loads the target;
//! 2. before that `LDC`, the nearest `SHF.L.U32 Rs, Rm, 0x2, RZ` (unguarded, form 4) scales the
//!    selector by four bytes;
//! 3. before that `SHF`, the nearest `IMNMX.U32 Rm, Rx, N, PT` (unguarded, form 4, a minimum
//!    with `N` from 1 to 15) clamps it, so the table has `N + 1` entries.
//!
//! Each step searches for its own pattern only: a later write of the same register by some
//! other instruction does not stop it, as in the reference. A `BRX` that does not match is not
//! tracked, and flow analysis then refuses it ([`super::FlowError::NoTarget`]).
//!
//! The reference searches back from each `BRX`, which is quadratic in the worst case (row 121).
//! [`find_jump_tables`] gets the same answers in one forward pass: it keeps, per register, the
//! last instruction of each pattern seen so far, and records at each `LDC` the `SHF` it would
//! find and at each `SHF` the `IMNMX`. Pinned against a direct port of the backward search by
//! `tests::jump_tables::forward_pass_matches_the_backward_search`.
//!
//! # Words and targets
//!
//! A table's words are the answers to [`Question::CbufWord`] for its entries. An answer of
//! "no such word" ([`crate::exchange::Answer::CbufWord`] of `None`) reads as 0, which names no
//! instruction; the reference reads whatever its environment returns for an unbound buffer.
//! The capture is extended through the furthest target; when it was, the tables are found
//! again in the extended code and their words read again, and the targets come from that
//! second read with the first read's base, as in the reference.

use std::collections::BTreeSet;

use super::analysis::{guard_of, opcode_of, second_of};
use super::classify::{FORM_IMM, LDC_OPCODE, with_form};
use super::{BranchTables, FlowError};
use crate::capture::{Capture, Extension, ProgramReader};
use crate::decode::{
    BrxSrc, CBUF_INDEX, CBUF_OFFSET, DST, Field, IMNMX_OPCODE, Instruction, Op, SHF_OPCODE,
    SLOT_IMM, SRC0, decode,
};
use crate::exchange::{Answers, Question, Questioner};

/// The second 32-bit word of an instruction, bits 32..64 (`words[1]`).
pub(crate) const WORD1: Field = Field::new(32, 32);
/// The third 32-bit word, bits 64..96 (`words[2]`).
pub(crate) const WORD2: Field = Field::new(64, 32);
/// The third word of the `LDC` the pattern reads (`words[2] == 0x00000800`): bit 75 alone,
/// NAK's memory type `B32` (4) in bits 73..76 (`sm70_encode.rs`, `OpLdc`: `set_mem_type(73..76,
/// ..)`), with the mode bits 78..80 zero (`LDC`, not `LDC.IL` and friends).
pub(crate) const LDC_WORD2: u64 = 0x0000_0800;
/// The scale `SHF` applies: two bits left, four bytes per entry (`words[1] == 2`).
pub(crate) const SHF_SCALE: u64 = 2;
/// The third word of the `SHF` the pattern reads (`words[2] == 0x000006ff`): source 2 `RZ` in
/// bits 64..72 and the type 3 (`U32`) in bits 73..75, everything else (a left shift, no `.W`,
/// no `.HI`) clear.
pub(crate) const SHF_WORD2: u64 = 0x0000_06ff;
/// The third word of the `IMNMX` the pattern reads (`words[2] == 0x03800000`): the select
/// predicate `PT` in bits 87..90 (a minimum) and everything else, the signed bit 73 among
/// them, clear.
pub(crate) const IMNMX_WORD2: u64 = 0x0380_0000;
/// The largest clamp the pattern accepts (`instruction.words[1] <= 15`), so a table has at
/// most this many entries plus one.
pub(crate) const MAX_CLAMP: u64 = 15;
/// The most entries a jump table can have: [`MAX_CLAMP`] plus one.
pub const MAX_TABLE_ENTRIES: u32 = 16;
/// The bytes of one table entry.
const ENTRY_BYTES: u32 = 4;
/// The bits of an `LDC`'s second word the pattern allows: the bank and the offset
/// (`!(words[1] & ~((0x1fU << 22) | (0xffffU << 6)))`).
const LDC_WORD1_KNOWN: u128 = CBUF_INDEX.mask() | CBUF_OFFSET.mask();
/// The low two bits of the `LDC`'s offset, which must be clear (`!((words[1] >> 6) & 3)`).
const LDC_OFFSET_LOW: Field = Field::new(CBUF_OFFSET.start, 2);
/// The register count, for the per-register memory of the forward pass.
const REGISTERS: usize = 256;

/// A tracked `BRX` and where its table lies (`CaptureIndirectBranch`). Its fields are private
/// to the flow module, where only [`find_jump_tables`] (and the module's tests) builds one, so
/// a table has 2 to [`MAX_TABLE_ENTRIES`] entries, which bounds [`JumpTable::questions`] and
/// [`JumpTable::words`] (`tests::jump_tables::a_table_has_at_most_max_entries`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct JumpTable {
    /// The `BRX`'s instruction index.
    pub(super) branch: u32,
    /// The constant-buffer bank of the table.
    pub(super) bank: u8,
    /// The byte offset of the table's first entry in the bank.
    pub(super) offset: u32,
    /// How many entries it has: 2 to [`MAX_TABLE_ENTRIES`].
    pub(super) entries: u32,
}

impl JumpTable {
    /// The `BRX`'s instruction index.
    pub const fn branch(&self) -> u32 {
        self.branch
    }

    /// The constant-buffer bank of the table.
    pub const fn bank(&self) -> u8 {
        self.bank
    }

    /// The byte offset of the table's first entry in the bank.
    pub const fn offset(&self) -> u32 {
        self.offset
    }

    /// How many entries it has: 2 to [`MAX_TABLE_ENTRIES`].
    pub const fn entries(&self) -> u32 {
        self.entries
    }

    /// The byte offset of entry `entry`.
    fn entry_offset(&self, entry: u32) -> Option<u32> {
        entry.checked_mul(ENTRY_BYTES)?.checked_add(self.offset)
    }

    /// The questions that read the table's words, in entry order.
    pub fn questions(&self) -> impl Iterator<Item = Question> + '_ {
        (0..self.entries).filter_map(|entry| {
            Some(Question::CbufWord {
                bank: self.bank,
                offset: self.entry_offset(entry)?,
            })
        })
    }

    /// The table's words from `answers`, in entry order; an answer of no word reads as 0.
    pub fn words(&self, answers: &Answers) -> Result<Vec<u32>, FlowError> {
        let mut words = Vec::with_capacity(usize::try_from(self.entries).unwrap_or(0));
        for entry in 0..self.entries {
            let offset = self.entry_offset(entry).ok_or(FlowError::MissingAnswer {
                bank: self.bank,
                offset: u32::MAX,
            })?;
            let word = answers
                .cbuf_word(self.bank, offset)
                .ok_or(FlowError::MissingAnswer {
                    bank: self.bank,
                    offset,
                })?;
            words.push(word.unwrap_or(0));
        }
        Ok(words)
    }
}

/// A register field.
fn reg(word: Instruction, field: Field) -> usize {
    // Cannot fail: register fields are eight bits wide.
    usize::try_from(word.field(field)).unwrap_or(0)
}

/// Whether `word` is the pattern's `LDC`.
pub(crate) fn is_table_load(word: Instruction) -> bool {
    opcode_of(word) == LDC_OPCODE
        && guard_of(word).is_true()
        && word.field(WORD2) == LDC_WORD2
        && word.bits() & WORD1.mask() & !LDC_WORD1_KNOWN == 0
        && word.field(LDC_OFFSET_LOW) == 0
}

/// Whether `word` is the pattern's `SHF`.
pub(crate) fn is_table_scale(word: Instruction) -> bool {
    opcode_of(word) == with_form(SHF_OPCODE, FORM_IMM)
        && guard_of(word).is_true()
        && word.field(SLOT_IMM) == SHF_SCALE
        && word.field(WORD2) == SHF_WORD2
}

/// The clamp of `word` if it is the pattern's `IMNMX`.
pub(crate) fn table_clamp(word: Instruction) -> Option<u64> {
    let clamp = word.field(SLOT_IMM);
    (opcode_of(word) == with_form(IMNMX_OPCODE, FORM_IMM)
        && guard_of(word).is_true()
        && (1..=MAX_CLAMP).contains(&clamp)
        && word.field(WORD2) == IMNMX_WORD2)
        .then_some(clamp)
}

/// The register a `BRX` at `index` branches through, if it is one the pattern tracks: a
/// register `BRX` that decodes, with guard and own predicate `PT`, counting from the program's
/// start.
pub(crate) fn tracked_brx(word: Instruction, index: u32) -> Option<usize> {
    let instr = decode(word).ok()?;
    let Op::Brx(brx) = instr.op else {
        return None;
    };
    let BrxSrc::Reg(src) = brx.src else {
        return None;
    };
    (instr.guard.is_true()
        && second_of(word).is_true()
        && super::analysis::target_byte(index, brx.offset) == Some(0))
    .then_some(usize::from(src.index()))
}

/// The tables of every tracked `BRX` in `code`, in instruction order
/// (`TrackCaptureIndirectBranches`, in one forward pass: see the module docs).
pub fn find_jump_tables(code: &[Instruction]) -> Vec<JumpTable> {
    let mut last_load = [None::<usize>; REGISTERS];
    let mut last_scale = [None::<usize>; REGISTERS];
    let mut last_clamp = [None::<usize>; REGISTERS];
    // Per instruction: at an `LDC`, the `SHF` before it for its selector; at an `SHF`, the
    // `IMNMX` before it for its bounded selector.
    let mut before: Vec<Option<usize>> = vec![None; code.len()];
    let mut tables = Vec::new();
    for (slot, &word) in code.iter().enumerate() {
        if let Some(src) = u32::try_from(slot)
            .ok()
            .and_then(|index| tracked_brx(word, index))
        {
            if let Some(table) = track(code, &before, last_load.get(src).copied().flatten(), slot) {
                tables.push(table);
            }
        } else if is_table_load(word) {
            if let Some(link) = before.get_mut(slot) {
                *link = last_scale.get(reg(word, SRC0)).copied().flatten();
            }
            if let Some(last) = last_load.get_mut(reg(word, DST)) {
                *last = Some(slot);
            }
        } else if is_table_scale(word) {
            if let Some(link) = before.get_mut(slot) {
                *link = last_clamp.get(reg(word, SRC0)).copied().flatten();
            }
            if let Some(last) = last_scale.get_mut(reg(word, DST)) {
                *last = Some(slot);
            }
        } else if table_clamp(word).is_some()
            && let Some(last) = last_clamp.get_mut(reg(word, DST))
        {
            *last = Some(slot);
        }
    }
    tables
}

/// The table of the `BRX` at `branch` whose nearest `LDC` is `load`.
fn track(
    code: &[Instruction],
    before: &[Option<usize>],
    load: Option<usize>,
    branch: usize,
) -> Option<JumpTable> {
    let load = load?;
    let scale = before.get(load).copied().flatten()?;
    let clamp = before.get(scale).copied().flatten()?;
    let load_word = *code.get(load)?;
    let clamp = table_clamp(*code.get(clamp)?)?;
    Some(JumpTable {
        branch: u32::try_from(branch).ok()?,
        bank: u8::try_from(load_word.field(CBUF_INDEX)).ok()?,
        offset: u32::try_from(load_word.field(CBUF_OFFSET)).ok()?,
        entries: u32::try_from(clamp).ok()?.checked_add(1)?,
    })
}

/// A table with its words and targets (`CaptureIndirectTable`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolvedTable {
    /// Where the table is.
    pub table: JumpTable,
    /// Its words, in entry order.
    pub words: Vec<u32>,
    /// The distinct instructions inside the capture its words name, ascending.
    pub targets: Vec<u32>,
}

/// Every table of a capture, resolved (`CaptureIndirectResult`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolvedTables {
    /// The byte offset a value naming instruction 0 holds ([`crate::capture::JumpTables::base`]).
    pub base: u32,
    /// What happened to the capture.
    pub extension: Extension,
    /// The tables, in instruction order of their `BRX`.
    pub tables: Vec<ResolvedTable>,
}

impl ResolvedTables {
    /// The targets of each `BRX`, as flow analysis takes them.
    pub fn branch_tables(&self) -> BranchTables {
        let mut tables = BranchTables::new(self.base);
        for table in &self.tables {
            tables.insert(table.table.branch, table.targets.clone());
        }
        tables
    }
}

/// The instructions of `capture`.
fn instructions(capture: &Capture) -> Vec<Instruction> {
    capture.instructions().collect()
}

/// Each table's words, or the first missing answer.
fn read_words(tables: &[JumpTable], answers: &Answers) -> Result<Vec<Vec<u32>>, FlowError> {
    tables.iter().map(|table| table.words(answers)).collect()
}

/// Resolves the jump tables of `capture` with the words in `answers`, extending it through the
/// furthest target read from `reader` (see the module docs). The capture changes only when
/// every word was answered: on an error it is left as it was.
pub fn resolve_jump_tables<R: ProgramReader + ?Sized>(
    capture: &mut Capture,
    reader: &R,
    answers: &Answers,
) -> Result<ResolvedTables, FlowError> {
    let first = find_jump_tables(&instructions(capture));
    let first_words = read_words(&first, answers)?;
    let mut extended = capture.clone();
    let views: Vec<&[u32]> = first_words.iter().map(Vec::as_slice).collect();
    let resolved = extended.resolve_jump_tables(reader, &views);
    let tables = if matches!(resolved.extension, Extension::Extended { .. }) {
        let second = find_jump_tables(&instructions(&extended));
        let second_words = read_words(&second, answers)?;
        second
            .into_iter()
            .zip(second_words)
            .map(|(table, words)| ResolvedTable {
                table,
                targets: extended.jump_targets(resolved.base, &words),
                words,
            })
            .collect()
    } else {
        first
            .into_iter()
            .zip(first_words)
            .zip(resolved.targets)
            .map(|((table, words), targets)| ResolvedTable {
                table,
                words,
                targets,
            })
            .collect()
    };
    *capture = extended;
    Ok(ResolvedTables {
        base: resolved.base,
        extension: resolved.extension,
        tables,
    })
}

/// The questions a capture's jump tables ask, for the exchange (slice 9's [`Questioner`]): the
/// words of the tables in the capture and, once those are answered and the capture would be
/// extended, the words of the tables in the extended code.
#[derive(Debug)]
pub struct JumpTableQuestions<'a, R: ?Sized> {
    capture: &'a Capture,
    reader: &'a R,
}

impl<'a, R: ProgramReader + ?Sized> JumpTableQuestions<'a, R> {
    /// The questions of `capture`, whose extension reads through `reader`.
    pub const fn new(capture: &'a Capture, reader: &'a R) -> Self {
        Self { capture, reader }
    }
}

impl<R: ProgramReader + ?Sized> Questioner for JumpTableQuestions<'_, R> {
    fn questions(&self, answers: &Answers) -> Vec<Question> {
        let first = find_jump_tables(&instructions(self.capture));
        let mut questions: BTreeSet<Question> =
            first.iter().flat_map(JumpTable::questions).collect();
        if let Ok(words) = read_words(&first, answers) {
            let mut extended = self.capture.clone();
            let views: Vec<&[u32]> = words.iter().map(Vec::as_slice).collect();
            let resolved = extended.resolve_jump_tables(self.reader, &views);
            if matches!(resolved.extension, Extension::Extended { .. }) {
                questions.extend(
                    find_jump_tables(&instructions(&extended))
                        .iter()
                        .flat_map(JumpTable::questions),
                );
            }
        }
        questions.into_iter().collect()
    }
}
