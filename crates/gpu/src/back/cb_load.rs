//! Constant-buffer loads (gpu-protocol spec, "DMA copy, inline-to-memory and 2D",
//! "Constant-buffer loads", and the slice table's G25; decided by the owner, 2026-11-19): the
//! words of `LOAD_CONSTANT_BUFFER(i)` are written to guest memory at the selected constant
//! buffer's address plus the upload offset, as Eden-NX2's `Maxwell3D::ProcessCBMultiData` does
//! (`eden-nx2/src/video_core/engines/maxwell_3d.cpp`), and the range is invalidated in the
//! caches like an inline write.
//!
//! No code is ported: [`ConstantBufferLoad`] is written from the behaviour of
//! `ProcessCBMultiData` and the memory manager's `WriteBlockCached`
//! (`eden-nx2/src/video_core/memory_manager.cpp`), with the divergences listed below.
//!
//! # What a load does
//!
//! For a `Write` or `Run` of any of the sixteen `LOAD_CONSTANT_BUFFER(i)` methods
//! ([`CONSTANT_BUFFER_LOADS`]; Eden-NX2 hands all sixteen to the same upload) of the 3D class,
//! once the channel's [`Mirror`] has stored the words, [`ConstantBufferLoad::method`]:
//!
//! 1. Takes the buffer from the selector registers ([`Mirror::constant_buffer_selector`]): the
//!    address `SET_CONSTANT_BUFFER_SELECTOR_B.ADDRESS_UPPER` (7:0) above
//!    `SET_CONSTANT_BUFFER_SELECTOR_C.ADDRESS_LOWER`, and the size
//!    `SET_CONSTANT_BUFFER_SELECTOR_A.SIZE` (16:0), Eden-NX2's `regs.const_buffer.Address()` and
//!    `regs.const_buffer.size`.
//! 2. Takes each word's byte offset from the `V` field (15:0) of `LOAD_CONSTANT_BUFFER_OFFSET`
//!    as it stood before that word ([`Mirror::run_positions`], the front's own advance rule).
//!    The front advances the offset and sends it as a `Write` after the run (gpu-protocol,
//!    "Engine methods: the register file", step 2), so the mirror still holds the value before
//!    the first word.
//! 3. Checks every word before anything is written ([`check`]), in word order, and for each word
//!    first the size bound, then the address: a word whose end (offset plus 4) passes the size
//!    is [`ConstantBufferError::PastEnd`]; a word whose end passes the 64-bit address space is
//!    [`ConstantBufferError::AddressOverflow`]. The first word that fails is reported, and a
//!    refused load writes nothing and invalidates nothing.
//! 4. Writes the words in pieces: words whose offsets follow one another go to guest memory in
//!    one [`GpuMemory::write`], as Eden-NX2 writes a run with one `WriteBlockCached`; a jump in
//!    the offset (its wrap within the 16-bit field, below) starts a new piece. A run of `n`
//!    words with no wrap is one write of `4n` bytes.
//! 5. Adds each piece's range to the address space's [`PendingWrites`], as an inline write's:
//!    merged and handed to the buffer, texture and shader caches before the next draw, dispatch
//!    or copy (row 134). Eden-NX2's `WriteBlockCached` likewise adds the range to the memory
//!    manager's accumulator instead of invalidating at once. A piece whose write faulted is added
//!    too, since the pages before the fault may have been written; the pieces after it are not
//!    written, and the fault is returned ([`ConstantBufferError::Fault`]).
//!
//! Nothing is written back first: `WriteBlockCached` writes with `CacheType::None` and asks no
//! cache to flush, and here no [`crate::back::copy::CopySink::make_current`] is made either.
//!
//! # Divergences
//!
//! Each is pinned by the test named (in `tests`).
//!
//! - **The whole extent is bounded by the size, and a load past it writes nothing.** Eden-NX2
//!   checks only where the run starts, `ASSERT(regs.const_buffer.offset <=
//!   regs.const_buffer.size)`, and an `ASSERT` logs without stopping, so Eden-NX2 writes the run
//!   wherever it reaches, past the size or from an offset at the size. Here the bounds check
//!   against the size covers every word's end (issue #429), and one word past it refuses the
//!   whole entry, the words before it included (the house rule that every check runs before the
//!   first write). `the_size_edge_accepts_a_word_ending_at_it_and_refuses_one_past`,
//!   `a_run_whose_last_word_passes_the_size_writes_none_of_its_words`.
//! - **A buffer address of zero is not refused here.** Eden-NX2 asserts `buffer_address != 0` (a
//!   log) and writes anyway; here the write goes to the address space too, whose memory refuses
//!   an unmapped page. `a_buffer_at_address_zero_is_written_like_any_other`.
//! - **A piece with an unmapped page writes nothing.** Eden-NX2's `WriteBlockImpl` skips the
//!   unmapped pages of a write and writes the mapped ones; here the address space's memory
//!   ([`crate::memory::VmMemory`] and [`crate::memory::TestGpuMemory`] both translate every page
//!   before writing) refuses the whole piece, and the entry is
//!   [`ConstantBufferError::Fault`]. `a_piece_over_an_unmapped_page_is_the_exact_fault`.
//! - **The selector's fields are the header's.** Eden-NX2's `ConstantBuffer::Address()` shifts
//!   the whole `address_high` word and compares the whole `size` word; here the address takes
//!   `ADDRESS_UPPER` (7:0) and the size `SIZE` (16:0), and the bits above are ignored (decision
//!   10: the headers for layouts), as a bind reads them.
//!   `a_run_lands_at_the_selected_address_plus_the_offset_in_one_write`.
//! - **The offset is the 16-bit field.** Eden-NX2 adds to the whole register, so a run that
//!   passes offset 0xFFFC goes on to 0x10000; the front here advances the offset within its
//!   field (`regs` module docs, "The constant-buffer offset wraps within its field"), so the
//!   next word goes to offset 0 and starts a new piece.
//!   `a_run_that_wraps_the_offset_field_is_two_writes`.
//!
//! **Unreachable from the registers:** [`ConstantBufferError::AddressOverflow`]. The selector's
//! address has 40 bits and a word's end at most 17, so no selected buffer reaches the end of the
//! 64-bit space; [`check`] is a pure function so that the bound is still tested
//! (`check_refuses_a_word_ending_past_the_address_space`). **Unreachable from the executor:**
//! [`ConstantBufferError::State`], which [`Mirror::run_positions`] returns only for a register
//! outside the file, and the 3D file the executor hands over holds `LOAD_CONSTANT_BUFFER_OFFSET`.

#[cfg(test)]
mod tests;

use core::ops::Range;

use thiserror::Error;

use crate::back::copy::PendingWrites;
use crate::back::state::{Mirror, StateError};
use crate::classes::clc797;
use crate::memory::{GpuFault, GpuMemory};
use crate::regs::{CONSTANT_BUFFER_LOADS, WORD_BYTES};

/// Why a constant-buffer load was refused, or failed while writing.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum ConstantBufferError {
    /// A word's end, its offset plus 4, passes the selected buffer's size; nothing was written.
    #[error(
        "a constant-buffer word at byte offset {offset:#x} ends past the buffer's size of \
         {size:#x} bytes"
    )]
    PastEnd {
        /// The word's byte offset in the buffer.
        offset: u32,
        /// The buffer's size in bytes.
        size: u32,
    },
    /// A word's end, the buffer's address plus its offset plus 4, passes the end of the 64-bit
    /// address space; nothing was written.
    #[error(
        "a constant-buffer word at byte offset {offset:#x} of the buffer at {address:#x} runs \
         past the end of the address space"
    )]
    AddressOverflow {
        /// The buffer's GPU address.
        address: u64,
        /// The word's byte offset in the buffer.
        offset: u32,
    },
    /// Reading the upload offset out of the mirror was refused.
    #[error("the upload offset could not be read: {0}")]
    State(#[from] StateError),
    /// Writing a piece of the load faulted; the pieces before it were written.
    #[error("writing the constant-buffer words faulted: {0}")]
    Fault(GpuFault),
}

/// The bytes one word takes in guest memory.
const WORD_LEN: u64 = WORD_BYTES as u64;

/// Whether 3D method `method` is a constant-buffer load ([`CONSTANT_BUFFER_LOADS`]).
#[must_use]
pub fn is_constant_buffer_load(method: u32) -> bool {
    CONSTANT_BUFFER_LOADS.contains(method)
}

/// Checks the words of a load of the buffer at `address` of `size` bytes, whose byte offsets are
/// `offsets`, in order (module docs, step 3): for each word, the size bound, then the address.
///
/// # Errors
///
/// The first word's [`ConstantBufferError::PastEnd`] or [`ConstantBufferError::AddressOverflow`].
pub(crate) fn check(
    address: u64,
    size: u32,
    offsets: impl Iterator<Item = u32>,
) -> Result<(), ConstantBufferError> {
    for offset in offsets {
        let end = u64::from(offset) + WORD_LEN;
        if end > u64::from(size) {
            return Err(ConstantBufferError::PastEnd { offset, size });
        }
        if address.checked_add(end).is_none() {
            return Err(ConstantBufferError::AddressOverflow { address, offset });
        }
    }
    Ok(())
}

/// The constant-buffer upload of one 3D engine's channel, or of the executor (module docs): it
/// holds only the scratch bytes a piece is written from, reused from one load to the next.
#[derive(Clone, Debug, Default)]
pub struct ConstantBufferLoad {
    bytes: Vec<u8>,
}

impl ConstantBufferLoad {
    /// An upload with no scratch yet.
    pub fn new() -> Self {
        Self::default()
    }

    /// Handles `words` written to 3D method `method`, after `mirror` (the channel's 3D mirror)
    /// has stored them: for a constant-buffer load, writes them to `memory` and adds their
    /// ranges to `pending` (module docs); for any other method, or no words, does nothing.
    ///
    /// # Errors
    ///
    /// [`ConstantBufferError::PastEnd`] or [`ConstantBufferError::AddressOverflow`] before
    /// anything is written; [`ConstantBufferError::Fault`] for a piece whose write faulted, with
    /// the pieces before it written and every attempted piece's range in `pending`.
    pub fn method<M: GpuMemory + ?Sized>(
        &mut self,
        mirror: &Mirror,
        method: u32,
        words: &[u32],
        memory: &M,
        pending: &mut PendingWrites,
    ) -> Result<(), ConstantBufferError> {
        if words.is_empty() || !is_constant_buffer_load(method) {
            return Ok(());
        }
        let buffer = mirror.constant_buffer_selector();
        let positions = mirror.run_positions(
            method,
            clc797::LOAD_CONSTANT_BUFFER_OFFSET.method(),
            words.len(),
        )?;
        let offsets = positions.map(|position| clc797::LOAD_CONSTANT_BUFFER_OFFSET_V.get(position));
        check(buffer.address, buffer.size, offsets.clone())?;
        let mut piece: Option<(u32, Range<usize>)> = None;
        for (index, offset) in offsets.enumerate() {
            let next = index + 1;
            piece = match piece {
                Some((start, words_range))
                    if u64::from(start) + range_len(&words_range) == u64::from(offset) =>
                {
                    Some((start, words_range.start..next))
                }
                Some((start, words_range)) => {
                    self.write_piece(buffer.address, start, &words[words_range], memory, pending)?;
                    Some((offset, index..next))
                }
                None => Some((offset, index..next)),
            };
        }
        if let Some((start, words_range)) = piece {
            self.write_piece(buffer.address, start, &words[words_range], memory, pending)?;
        }
        Ok(())
    }

    /// Writes `words` at `address` plus `offset`, then adds the range to `pending` whatever the
    /// write's result (module docs, step 5). [`check`] has made the range's end representable.
    fn write_piece<M: GpuMemory + ?Sized>(
        &mut self,
        address: u64,
        offset: u32,
        words: &[u32],
        memory: &M,
        pending: &mut PendingWrites,
    ) -> Result<(), ConstantBufferError> {
        self.bytes.clear();
        self.bytes
            .extend(words.iter().flat_map(|word| word.to_le_bytes()));
        let start = address.saturating_add(u64::from(offset));
        let len = u64::try_from(self.bytes.len()).unwrap_or(u64::MAX);
        let written = memory.write(start, &self.bytes);
        pending.add(start..start.saturating_add(len));
        written.map_err(ConstantBufferError::Fault)
    }
}

/// The bytes the words of `words` take.
fn range_len(words: &Range<usize>) -> u64 {
    u64::try_from(words.len())
        .unwrap_or(u64::MAX)
        .saturating_mul(WORD_LEN)
}
