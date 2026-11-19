//! The root tables: constant buffer banks 24 to 31, 256 bytes each, loaded through
//! `SET_ROOT_TABLE_SELECTOR` and `LOAD_ROOT_TABLE` (gpu-protocol spec, "Protocol facts": "Root
//! tables: banks 24–31, 256 bytes each, loaded with `SET_ROOT_TABLE_SELECTOR` (0x141) and
//! `LOAD_ROOT_TABLE` (0x142)", from `classes/clc797.h` and
//! `eden-pr4588/src/video_core/native_root_table.h`).
//!
//! PR #4588's `NativeRootTables` (GPL-3.0-or-later) is the model: the selector's `ROOT_TABLE`
//! field (2:0) picks a table and its `OFFSET` field (15:8) a byte offset; each loaded word is
//! stored at the offset, which then advances by four. A word whose offset is not a multiple of
//! four is dropped, as `Load` returns false for it (PR #4588 logs it; this counts it,
//! [`RootTables::dropped`]). The offset field is eight bits, so it is always below 256 and the
//! PR's `offset >= Size` check cannot fail here.
//!
//! The mirror keeps only the last word written to `LOAD_ROOT_TABLE`, so the words are handed to
//! [`RootTables::load`] with the run, before the selector's side-effect write that follows the
//! run is applied: the offset of each word comes from the mirror's
//! [`Mirror::run_positions`], the front's own advance rule (whose wrap after 252 is a guess of
//! the `regs` module).
//!
//! **Bank 24.** Row 65 of the Eden-NX2 notes synthesizes a console-provided blit block in bank
//! 24; PR #4588 has NVN2 load banks 24 to 31 itself. This follows the spec's root-table model
//! and does not synthesize bank 24: that NVN2 loads it is a **guess** until verification item 6
//! (gpu-protocol, "What to verify before building") traces `LOAD_ROOT_TABLE` writes.

use crate::back::state::{Mirror, StateError};
use crate::classes::clc797;

/// The first bank a root table backs (PR #4588's `FirstBank`).
pub const FIRST_ROOT_BANK: u8 = 24;

/// How many root tables there are (`Count`).
pub const ROOT_TABLES: usize = 8;

/// Bytes in one root table (`Size`).
pub const ROOT_TABLE_BYTES: usize = 256;

/// Words in one root table.
const ROOT_TABLE_WORDS: usize = ROOT_TABLE_BYTES / 4;

/// The root tables of one 3D mirror (module docs).
#[derive(Clone, Debug)]
pub struct RootTables {
    tables: [[u32; ROOT_TABLE_WORDS]; ROOT_TABLES],
    /// Bumped by every stored word, so a caller can tell the tables changed.
    generation: u64,
    dropped: u64,
}

impl Default for RootTables {
    fn default() -> Self {
        Self {
            tables: [[0; ROOT_TABLE_WORDS]; ROOT_TABLES],
            generation: 0,
            dropped: 0,
        }
    }
}

impl RootTables {
    /// Tables of zeros (PR #4588's value-initialised `tables{}`).
    pub fn new() -> Self {
        Self::default()
    }

    /// The table backing `bank`, or `None` for a bank below [`FIRST_ROOT_BANK`] or past the
    /// last table.
    pub fn table(&self, bank: u8) -> Option<&[u32; ROOT_TABLE_WORDS]> {
        let index = usize::from(bank.checked_sub(FIRST_ROOT_BANK)?);
        self.tables.get(index)
    }

    /// The word at byte `offset` of `bank`'s table: `None` for a bank no table backs, an offset
    /// not a multiple of four, or one whose word ends past the table.
    pub fn word(&self, bank: u8, offset: u32) -> Option<u32> {
        if !offset.is_multiple_of(4) {
            return None;
        }
        let index = usize::try_from(offset / 4).ok()?;
        self.table(bank)?.get(index).copied()
    }

    /// The tables' bytes, table after table, little-endian.
    pub fn bytes(&self) -> Vec<u8> {
        self.tables
            .iter()
            .flatten()
            .flat_map(|word| word.to_le_bytes())
            .collect()
    }

    /// Bumped by every stored word.
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Words dropped for an offset that is not a multiple of four.
    pub fn dropped(&self) -> u64 {
        self.dropped
    }

    /// Stores `words`, a run of `LOAD_ROOT_TABLE` words the mirror has applied, at the offsets
    /// the selector gives each (module docs). Call it before the selector write that follows
    /// the run is applied.
    ///
    /// # Errors
    ///
    /// [`StateError`] when the mirror has no such registers (another class); nothing changes.
    pub fn load(&mut self, mirror: &Mirror, words: &[u32]) -> Result<(), StateError> {
        let positions = mirror.run_positions(
            clc797::LOAD_ROOT_TABLE.method(),
            clc797::SET_ROOT_TABLE_SELECTOR.method(),
            words.len(),
        )?;
        for (selector, &word) in positions.zip(words) {
            let table = usize::try_from(clc797::SET_ROOT_TABLE_SELECTOR_ROOT_TABLE.get(selector))
                .unwrap_or(usize::MAX);
            let offset = clc797::SET_ROOT_TABLE_SELECTOR_OFFSET.get(selector);
            let slot = offset
                .is_multiple_of(4)
                .then(|| usize::try_from(offset / 4).ok())
                .flatten()
                .and_then(|index| self.tables.get_mut(table)?.get_mut(index));
            match slot {
                Some(slot) => {
                    *slot = word;
                    self.generation = self.generation.wrapping_add(1);
                }
                None => self.dropped = self.dropped.saturating_add(1),
            }
        }
        Ok(())
    }
}
