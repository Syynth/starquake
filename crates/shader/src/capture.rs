//! Program capture: where a guest program ends, the instruction cap, and the content hash
//! (`docs/specs/shader-pipeline.md`, "Where programs come from" and "Identity, caching and
//! invalidation"; port slice 3).
//!
//! Nothing records a program's length. [`capture`] reads the header, then instructions one at a
//! time through the [`ProgramReader`] seam, until one ends the program; flow analysis (slice
//! 10), the shader cache (slice 19) and the capture format (slice 20) take the [`Capture`] it
//! returns. Slice 19 adapts slice 9's [`ProgramMemory`](crate::lookup::ProgramMemory), which
//! reads at guest addresses, to the seam, which reads at offsets from the program's first byte.
//!
//! # Sources
//!
//! Ported from PR #4588: `eden-pr4588/src/shader_recompiler/frontend/ampere/program_capture.h`
//! @ e5fb3b97 (GPL-3.0-or-later, Copyright 2026 Eden Emulator Project): the extent rules
//! (`ProgramCaptureState::Observe`, in the private `scan` module), the cap
//! (`MaximumInstructions`) and the jump-table extension (`ResolveCaptureIndirectBranches`), and
//! its caller's loop, `GenericEnvironment::InitializeNativeCode`
//! (`eden-pr4588/src/video_core/shader_environment.cpp`). The capture reads only the layout
//! facts the spec lists for use before full decoding (the private `layout` module names each).
//!
//! # Where a program ends
//!
//! At the first instruction, at index `i`, that is either
//!
//! - an `EXIT` whose guard and own predicate are `PT` and whose bits 84 and 85 are clear, with
//!   `i` at or past every branch target seen so far and no unconditional `CALL` seen; or
//! - a `RET` whose guard is `PT`, after such an `EXIT` was seen while a call was pending, with `i`
//!   at or past every branch target and every call target seen so far.
//!
//! The branches are the relative `CALL`, `BSSY` and `BRA`. A `CALL` whose guard is exactly `PT`
//! is a call; every other `CALL` (the predicated `CALL.REL.NOINC` of row 67 among them), every
//! `BSSY` and every `BRA` is a forward branch. Library functions may `EXIT` (row 88): once a call
//! is pending, an `EXIT` does not end the program. A call to instruction 0 is not counted as a
//! call, as in the reference, where a call target of 0 reads as "no call". Every branch target
//! must be a whole instruction at or past instruction 0 and below the cap
//! ([`CaptureError::TargetOutside`], [`CaptureError::MisalignedTarget`]). When the capture ends,
//! every relative branch target in it lies inside it: forward targets and call targets at or
//! before the final instruction, backward ones at or after instruction 0.
//!
//! # The header
//!
//! A graphics program starts with [`HEADER_BYTES`](crate::sph::HEADER_BYTES) of header and a
//! compute program with none ([`header_bytes`]), PR #4588's values
//! (`InitializeNativeCode(u32 source_header_size = 128, ..)` for graphics,
//! `InitializeNativeCode(0, ..)` for compute). Both are cited, not measured: verification E3
//! (spec, "What to verify before building", item 3) confirms them. PR #4588's native compute
//! path does set `initial_offset` and `start_address` to `sizeof(sph)` after that call, but
//! those offsets are into the lowered Maxwell code buffer it builds with a copied header in
//! front (`code.resize(sizeof(sph) / sizeof(u64))`), not into guest memory: its capture reads
//! instruction 0 of a compute program at the program address itself.
//!
//! # The cap
//!
//! [`MAX_INSTRUCTIONS`], 65,536 (`MaximumInstructions`). The reference's loop reads indices
//! `0..MaximumInstructions` (`for (u32 index = 0; index < maximum_instructions; ++index)`), so
//! a program that ends on instruction 65,535, 65,536 instructions long, is captured; and it
//! refuses a branch target at or past `MaximumInstructions * 16` bytes, so instruction 65,535 is
//! the last target allowed.
//!
//! # Jump tables
//!
//! A `BRX` selects its target from a table in a constant buffer, and a target may lie past the
//! instruction that ends the capture (row 92). [`Capture::resolve_jump_tables`] takes the
//! tables' words, which slice 9's `CbufWord` answers give (slice 10 finds the tables and slice 19
//! supplies the words), and extends the capture through the furthest target, as
//! `ResolveCaptureIndirectBranches` does after `TrackCaptureIndirectBranches` has found them.
//! Finding the tables (the selector pattern before each `BRX`) is not done here: it reads
//! operands of `IMNMX`, `SHF` and `LDC`, which needs the decoder.
//!
//! # Divergences from PR #4588
//!
//! - **A program with no end within the cap is rejected** ([`CaptureError::NoEnd`]), as the
//!   spec says; `InitializeNativeCode`'s loop ends without an error and keeps the first 65,536
//!   instructions. Pinned by `tests::a_program_with_no_end_within_the_cap_is_rejected`.
//! - **The header is read, and an unreadable header is a rejection**
//!   ([`CaptureError::HeaderUnreadable`]), because the hash covers it; the reference's capture
//!   loop reads no header bytes (it reads them only for its diagnostic dumps). Pinned by
//!   `tests::an_unreadable_header_is_rejected`.
//! - **A refused branch target names its reason**: [`CaptureError::TargetOutside`] or
//!   [`CaptureError::MisalignedTarget`], checked in the reference's order (below zero, not a
//!   whole instruction, at or past the cap); the reference returns `Result::InvalidTarget` for
//!   all three. The same set of targets is refused. Pinned by
//!   `tests::target_checks_run_in_the_reference_order`.
//! - **The hash** is this crate's, not the reference's 64-bit `NativeProgramKey` over the
//!   instructions only (spec, "Hash"; the private `hash` module).

mod hash;
mod layout;
mod scan;
#[cfg(test)]
mod tests;

use thiserror::Error;

pub use hash::ContentHash;
use scan::{Scan, Step};

use crate::decode::Instruction;
use crate::sph::HEADER_BYTES;
use crate::translate::GuestStage;

/// The most instructions a program may have (`program_capture.h`, `MaximumInstructions`).
pub const MAX_INSTRUCTIONS: u32 = 65_536;

/// The bytes of header in front of a program of `stage`: [`HEADER_BYTES`] for a graphics
/// stage, none for compute (spec, "Where programs come from").
pub const fn header_bytes(stage: GuestStage) -> usize {
    match stage {
        GuestStage::Compute => 0,
        GuestStage::VertexCullBeforeFetch
        | GuestStage::Vertex
        | GuestStage::TessellationInit
        | GuestStage::Tessellation
        | GuestStage::Geometry
        | GuestStage::Pixel => HEADER_BYTES,
    }
}

/// A read through [`ProgramReader`] reached a byte it cannot read.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
#[error("program byte {offset:#x} cannot be read")]
pub struct Unreadable {
    /// The offset, from the program's first byte, of the first byte that cannot be read.
    pub offset: u64,
}

/// The byte-reader seam capture reads a program through: offsets count from the program's first
/// byte (the header's first byte for a graphics program).
pub trait ProgramReader {
    /// Fills `out` with the bytes at `offset` onwards, or reports the first that cannot be
    /// read.
    fn read(&self, offset: u64, out: &mut [u8]) -> Result<(), Unreadable>;
}

/// A program held in memory: every byte past the end is unreadable.
impl ProgramReader for [u8] {
    fn read(&self, offset: u64, out: &mut [u8]) -> Result<(), Unreadable> {
        let len = u64::try_from(self.len()).unwrap_or(u64::MAX);
        let first_unreadable = Unreadable {
            offset: offset.max(len),
        };
        let start = usize::try_from(offset).map_err(|_| first_unreadable)?;
        let end = start.checked_add(out.len()).ok_or(first_unreadable)?;
        let bytes = self.get(start..end).ok_or(first_unreadable)?;
        out.copy_from_slice(bytes);
        Ok(())
    }
}

/// Why a program cannot be captured.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum CaptureError {
    /// The header cannot be read.
    #[error("the program header cannot be read at byte {offset:#x}")]
    HeaderUnreadable {
        /// The first byte, from the program's start, that cannot be read.
        offset: u64,
    },
    /// An instruction cannot be read (`InitializeNativeCode`: "Unmapped native shader
    /// instruction").
    #[error("instruction {index} cannot be read at byte {offset:#x}")]
    InstructionUnreadable {
        /// The instruction's index.
        index: u32,
        /// The first byte, from the program's start, that cannot be read.
        offset: u64,
    },
    /// A relative branch reaches below instruction 0 or at or past the cap.
    #[error(
        "instruction {index} branches to code byte {target}, outside the first {max} instructions",
        max = MAX_INSTRUCTIONS
    )]
    TargetOutside {
        /// The branch's index.
        index: u32,
        /// The target, in bytes from instruction 0.
        target: i64,
    },
    /// A relative branch reaches a byte that does not start an instruction.
    #[error(
        "instruction {index} branches to code byte {target}, which does not start an instruction"
    )]
    MisalignedTarget {
        /// The branch's index.
        index: u32,
        /// The target, in bytes from instruction 0.
        target: i64,
    },
    /// No instruction within the cap ends the program.
    #[error(
        "no instruction within the first {max} ends the program",
        max = MAX_INSTRUCTIONS
    )]
    NoEnd,
}

/// A captured program: its header and every instruction up to the one that ends it.
///
/// Made only by [`capture`], so its bytes are always a header of [`header_bytes`] for its stage
/// followed by at least one whole instruction, the last of which ends the program.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Capture {
    stage: GuestStage,
    header_len: usize,
    /// The header followed by the instructions, as in guest memory.
    bytes: Vec<u8>,
    /// The scan's state after the last instruction, which an extension continues from.
    scan: Scan,
}

/// Reads the program of `stage` through `reader` until an instruction ends it.
pub fn capture<R: ProgramReader + ?Sized>(
    reader: &R,
    stage: GuestStage,
) -> Result<Capture, CaptureError> {
    let header_len = header_bytes(stage);
    let mut bytes = vec![0; header_len];
    reader
        .read(0, &mut bytes)
        .map_err(|Unreadable { offset }| CaptureError::HeaderUnreadable { offset })?;
    let mut scan = Scan::default();
    for index in 0..MAX_INSTRUCTIONS {
        let word = read_instruction(reader, header_len, index)?;
        bytes.extend_from_slice(&word.to_le_bytes());
        if scan.observe(word, index)? == Step::Complete {
            return Ok(Capture {
                stage,
                header_len,
                bytes,
                scan,
            });
        }
    }
    Err(CaptureError::NoEnd)
}

/// Reads instruction `index` of a program with `header_len` bytes of header.
fn read_instruction<R: ProgramReader + ?Sized>(
    reader: &R,
    header_len: usize,
    index: u32,
) -> Result<Instruction, CaptureError> {
    // Cannot fail: the index is below the cap and the header 0 or 128 bytes.
    let offset = u64::from(index)
        .checked_mul(INSTRUCTION_SIZE)
        .and_then(|code| code.checked_add(u64::try_from(header_len).ok()?))
        .unwrap_or(u64::MAX);
    let mut word = [0; Instruction::SIZE];
    reader
        .read(offset, &mut word)
        .map_err(|Unreadable { offset }| CaptureError::InstructionUnreadable { index, offset })?;
    Ok(Instruction::from_le_bytes(word))
}

/// [`Instruction::SIZE`] as a byte offset.
const INSTRUCTION_SIZE: u64 = Instruction::SIZE as u64;

/// What [`Capture::resolve_jump_tables`] did to the capture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Extension {
    /// No table was given, or every target already lies inside the capture.
    NotNeeded,
    /// The capture was extended through the furthest target; it had `from` instructions.
    Extended {
        /// The instruction count before the extension.
        from: u32,
    },
    /// The capture could not be extended to an instruction that ends the program past the
    /// furthest target, for this reason; it is unchanged (`ResolveCaptureIndirectBranches`
    /// restores `native` and `capture`).
    Abandoned(CaptureError),
}

/// What [`Capture::resolve_jump_tables`] found (`CaptureIndirectResult`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JumpTables {
    /// The byte offset in the program that a table value of the instruction at index 0 holds:
    /// the header's length, or 0 when some non-zero value is below it (`result.base`). It is 0
    /// when no table was given.
    pub base: u32,
    /// What happened to the capture.
    pub extension: Extension,
    /// Per table, in the order given: the distinct instruction indices its values name that lie
    /// inside the capture, ascending (`CaptureIndirectTable::targets`).
    pub targets: Vec<Vec<u32>>,
}

impl Capture {
    /// The stage the program was captured for.
    pub const fn stage(&self) -> GuestStage {
        self.stage
    }

    /// The header's bytes: [`header_bytes`] of them for the stage.
    pub fn header(&self) -> &[u8] {
        self.bytes.get(..self.header_len).unwrap_or_default()
    }

    /// The instructions' bytes.
    pub fn code(&self) -> &[u8] {
        self.bytes.get(self.header_len..).unwrap_or_default()
    }

    /// The header and the instructions, as they lie in guest memory.
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// How many instructions were captured: at least 1 and at most [`MAX_INSTRUCTIONS`].
    pub fn instruction_count(&self) -> u32 {
        // Cannot fail: the capture holds at most the cap's instructions.
        u32::try_from(self.code().len() / Instruction::SIZE).unwrap_or(MAX_INSTRUCTIONS)
    }

    /// The captured instructions, in order.
    pub fn instructions(&self) -> impl Iterator<Item = Instruction> + '_ {
        self.code()
            .as_chunks::<{ Instruction::SIZE }>()
            .0
            .iter()
            .map(|word| Instruction::from_le_bytes(*word))
    }

    /// The content hash of [`Capture::bytes`].
    #[must_use]
    pub fn hash(&self) -> ContentHash {
        ContentHash::of(self.header_len, &self.bytes)
    }

    /// Resolves jump tables, each given as its words in order, and extends the capture through
    /// the furthest target they name (`ResolveCaptureIndirectBranches`, after its first
    /// `read_tables`).
    ///
    /// Table values are byte offsets from the program's first byte, so they include the
    /// header before instruction 0; a non-zero value below the header's length can only count
    /// from instruction 0, and selects that convention for every table ([`JumpTables::base`]).
    /// A value names instruction `(value - base) / 16` when it is non-zero, at least `base`, a
    /// whole number of instructions past it and below the cap; other values name nothing. When
    /// the furthest instruction named lies at or past the capture's end, the capture reads on
    /// from its end until an instruction ends the program past that one. If it cannot (a byte
    /// cannot be read, a branch target is refused, or the cap is reached), the capture is left
    /// as it was ([`Extension::Abandoned`]). No table given leaves everything as it is.
    ///
    /// The reference then finds the tables again in the extended code and reads their words
    /// once more, keeping the first base; [`Capture::jump_targets`] serves that second read.
    #[must_use]
    pub fn resolve_jump_tables<R: ProgramReader + ?Sized>(
        &mut self,
        reader: &R,
        tables: &[&[u32]],
    ) -> JumpTables {
        if tables.is_empty() {
            return JumpTables {
                base: 0,
                extension: Extension::NotNeeded,
                targets: Vec::new(),
            };
        }
        let header = u32::try_from(self.header_len).unwrap_or(u32::MAX);
        let values = || tables.iter().flat_map(|table| table.iter().copied());
        let base = if values().any(|value| value != 0 && value < header) {
            0
        } else {
            header
        };
        let furthest = values()
            .filter_map(|value| table_entry(value, base))
            .max()
            .unwrap_or(0);
        let extension = self.extend_through(reader, furthest);
        let targets = tables
            .iter()
            .map(|table| self.jump_targets(base, table))
            .collect();
        JumpTables {
            base,
            extension,
            targets,
        }
    }

    /// The distinct instruction indices inside the capture that `values` name with `base`,
    /// ascending (the reference's last loop over `result.tables`).
    #[must_use]
    pub fn jump_targets(&self, base: u32, values: &[u32]) -> Vec<u32> {
        let count = self.instruction_count();
        let mut targets: Vec<u32> = values
            .iter()
            .filter_map(|&value| table_entry(value, base))
            .filter(|&index| index < count)
            .collect();
        targets.sort_unstable();
        targets.dedup();
        targets
    }

    /// Reads on from the capture's end until an instruction ends the program past instruction
    /// `furthest`, or leaves the capture as it was (`ResolveCaptureIndirectBranches`' extension
    /// loop).
    #[must_use]
    pub(crate) fn extend_through<R: ProgramReader + ?Sized>(
        &mut self,
        reader: &R,
        furthest: u32,
    ) -> Extension {
        let from = self.instruction_count();
        if furthest < from {
            return Extension::NotNeeded;
        }
        let saved_len = self.bytes.len();
        let saved_scan = self.scan;
        let abandon = |capture: &mut Self, error| {
            capture.bytes.truncate(saved_len);
            capture.scan = saved_scan;
            Extension::Abandoned(error)
        };
        for index in from..MAX_INSTRUCTIONS {
            let word = match read_instruction(reader, self.header_len, index) {
                Ok(word) => word,
                Err(error) => return abandon(self, error),
            };
            self.bytes.extend_from_slice(&word.to_le_bytes());
            match self.scan.observe(word, index) {
                Err(error) => return abandon(self, error),
                Ok(Step::Complete) if index >= furthest => return Extension::Extended { from },
                Ok(_) => {}
            }
        }
        abandon(self, CaptureError::NoEnd)
    }
}

/// The instruction index table value `value` names with `base`, if any (the reference's
/// `entry_index`).
pub(crate) fn table_entry(value: u32, base: u32) -> Option<u32> {
    if value == 0 {
        return None;
    }
    let past = value.checked_sub(base)?;
    let size = u32::try_from(Instruction::SIZE).ok()?;
    if past % size != 0 {
        return None;
    }
    let index = past / size;
    (index < MAX_INSTRUCTIONS).then_some(index)
}
