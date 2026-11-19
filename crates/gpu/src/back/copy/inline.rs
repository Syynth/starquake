//! Inline-to-memory: `A140` (`AMPERE_INLINE_TO_MEMORY_A`), and the same methods in the 3D
//! (`C797`) and compute (`C7C0`) classes, `LAUNCH_DMA` at byte 0x1B0 and `LOAD_INLINE_DATA` at
//! 0x1B4.
//!
//! # Sources
//!
//! Paths are relative to the reference folder (`tools/refs-path.sh`), at the commits pinned in
//! `docs/references.md`.
//!
//! - The methods and fields of each class are read through [`crate::classes::cla140`],
//!   [`crate::classes::clc797`] and [`crate::classes::clc7c0`], generated from `cla140.h`,
//!   `clc797.h` and `clc7c0.h` in `mesa-nouveau/src/nouveau/headers/nvidia/classes/` @ 52238762
//!   (Copyright (c) 2020, NVIDIA CORPORATION, MIT licence). The three classes put every method
//!   at the same byte address, but their upper address fields differ in width
//!   (`OFFSET_OUT_UPPER_VALUE` is 24:0 in `A140`, 7:0 in `C797`, 16:0 in `C7C0`), so each class
//!   has its own [`InlineClass`].
//! - The behaviour is ported from `Upload::State` (`ProcessExec`, `ProcessData`) in
//!   `eden-nx2/src/video_core/engines/engine_upload.{h,cpp}` @ 50fa86bb (Copyright 2025 Eden
//!   Emulator Project, GPL-3.0-or-later; Copyright 2019 yuzu Emulator Project, GPL-2.0-or-later)
//!   and `KeplerMemory::CallMethod` in `kepler_memory.cpp` (same commit and licence).
//!
//! # What an upload does
//!
//! `LAUNCH_DMA` starts a transfer of `LINE_COUNT` lines of `LINE_LENGTH_IN` bytes to the
//! destination, pitch-linear (line `l` at `OFFSET_OUT + l * PITCH_OUT`) or block-linear (the
//! region at `SET_DST_ORIGIN_BYTES_X`, `SET_DST_ORIGIN_SAMPLES_Y` of a surface `SET_DST_WIDTH`
//! bytes by `_HEIGHT` by `_DEPTH` with `SET_DST_BLOCK_SIZE`) by `DST_MEMORY_LAYOUT`. It checks, in
//! order: the register file ([`CopyError::Registers`]), the completion semaphore's reduction
//! ([`CopyError::Reduction`]), the destination surface and region (as for a DMA copy). A launch
//! abandons any transfer still in progress, whether it succeeds or not. Then it calls
//! [`super::CopySink::make_current`] for the destination's cover.
//!
//! Each `LOAD_INLINE_DATA` word then supplies the next four bytes, little-endian, written to
//! guest memory at once, in order (the spec: "The back writes the data to guest memory (pitch or
//! block-linear) in order"); the last word supplies only the bytes still expected. Each word's
//! pages are probed before it writes ([`CopyError::Fault`] on the destination), and the runs it
//! writes join [`super::PendingWrites`], to reach the caches at the next draw, dispatch or copy
//! (row 134). A word that fails ends the transfer: the words before it stay written, later
//! words are ignored and no semaphore is released. When the last byte is written, a
//! `COMPLETION_TYPE` of `RELEASE_SEMAPHORE` releases `SET_I2M_SEMAPHORE_C` at
//! `SET_I2M_SEMAPHORE_A/B`: four words with a timestamp (`SEMAPHORE_STRUCT_SIZE_FOUR_WORDS`) or
//! one, reduced by `REDUCTION_OP` and `REDUCTION_FORMAT` when `REDUCTION_ENABLE` is set. A
//! launch of zero bytes completes, and releases, at once.
//!
//! # Divergences from Eden-NX2
//!
//! Each is pinned by the test named, in `copy/tests/inline.rs`.
//!
//! - **Each word is written when it arrives.** Eden-NX2 collects the whole transfer and writes it
//!   on the method call's last word, except for a pitch-linear one-line transfer in its "native
//!   partial" mode, which it writes as it goes. Pinned by `each_word_is_written_as_it_arrives`.
//! - **The completion semaphore is released.** Eden-NX2 ignores `COMPLETION_TYPE` and the
//!   `SET_I2M_SEMAPHORE_*` methods. Pinned by `completion_semaphore_gives_the_exact_release`.
//! - **Upper address bits are the header's.** Eden-NX2 reads all 32 bits of `OFFSET_OUT_UPPER`
//!   (`GPUVAddr{address_high} << 32`); here each class's field width applies. Pinned by
//!   `divergence_upper_address_bits_follow_each_class`.
//! - **A region outside its surface, a block width other than one GOB, or a block height or
//!   depth past `THIRTYTWO_GOBS` is refused** (Eden-NX2 writes past the surface, and ignores the
//!   block width). Pinned by `divergence_bad_destinations_are_refused`.
//! - **Lines that continue into the next depth slice take the next stream bytes.** A
//!   block-linear region whose lines pass the surface's last row continues in the next slice
//!   ([`super::surface`]). Eden-NX2 passes `LINE_LENGTH_IN` as `SwizzleSubrectImpl`'s linear pitch
//!   (`engine_upload.cpp`, `ProcessData`), so with `SET_DST_ORIGIN_SAMPLES_Y` above 0 its stream
//!   offset for line `line` of slice `s` is `s * LINE_LENGTH_IN * height + line * LINE_LENGTH_IN`,
//!   skipping `origin_y * LINE_LENGTH_IN` stream bytes per slice and reading past the bytes the
//!   transfer supplied; here line `l` of the region always takes stream bytes
//!   `l * LINE_LENGTH_IN` onward. Pinned by `divergence_slices_take_the_next_stream_bytes`.
//! - **As in Eden-NX2,** words past the transfer's end, and words with no transfer launched, are
//!   ignored, and `SET_DST_LAYER`, `INTERRUPT_TYPE` and `SYSMEMBAR_DISABLE` are not read. Pinned
//!   by `extra_words_and_words_without_a_launch_are_ignored`.

use core::ops::Range;

use super::surface::{BlockFields, BlockLinearDesc, CopySide, Region, Surface};
use super::{CopyError, CopySink, PendingWrites, Registers, registers_needed};
use crate::classes::Method;
use crate::field::Field;
use crate::memory::GpuMemory;
use crate::ring::{Release, ReleaseOp};
use crate::sema::{self, PayloadSize, ReleaseForm};

/// The inline-to-memory methods and fields of one class, from its generated constants.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InlineClass {
    /// The word name a reduction decode error carries (`A140 LAUNCH_DMA`).
    pub(crate) launch_word: &'static str,
    pub(crate) line_length_in: Method,
    pub(crate) line_count: Method,
    pub(crate) offset_out_upper: Method,
    pub(crate) offset_out_upper_value: Field,
    pub(crate) offset_out: Method,
    pub(crate) pitch_out: Method,
    pub(crate) block_size: Method,
    pub(crate) block_width: Field,
    pub(crate) block_height: Field,
    pub(crate) block_depth: Field,
    pub(crate) width: Method,
    pub(crate) height: Method,
    pub(crate) depth: Method,
    pub(crate) origin_x: Method,
    pub(crate) origin_x_value: Field,
    pub(crate) origin_y: Method,
    pub(crate) origin_y_value: Field,
    pub(crate) launch: Method,
    pub(crate) layout: Field,
    pub(crate) layout_pitch: u32,
    pub(crate) completion: Field,
    pub(crate) completion_release: u32,
    pub(crate) struct_size: Field,
    pub(crate) struct_four_words: u32,
    pub(crate) reduction_enable: Field,
    pub(crate) reduction_enable_true: u32,
    pub(crate) reduction_op: Field,
    pub(crate) reduction_format: Field,
    pub(crate) data: Method,
    pub(crate) semaphore_a: Method,
    pub(crate) semaphore_a_upper: Field,
    pub(crate) semaphore_b: Method,
    pub(crate) semaphore_c: Method,
}

/// An [`InlineClass`] from a generated class module.
macro_rules! inline_class {
    ($module:ident, $word:literal) => {{
        use crate::classes::$module as m;
        InlineClass {
            launch_word: $word,
            line_length_in: m::LINE_LENGTH_IN,
            line_count: m::LINE_COUNT,
            offset_out_upper: m::OFFSET_OUT_UPPER,
            offset_out_upper_value: m::OFFSET_OUT_UPPER_VALUE,
            offset_out: m::OFFSET_OUT,
            pitch_out: m::PITCH_OUT,
            block_size: m::SET_DST_BLOCK_SIZE,
            block_width: m::SET_DST_BLOCK_SIZE_WIDTH,
            block_height: m::SET_DST_BLOCK_SIZE_HEIGHT,
            block_depth: m::SET_DST_BLOCK_SIZE_DEPTH,
            width: m::SET_DST_WIDTH,
            height: m::SET_DST_HEIGHT,
            depth: m::SET_DST_DEPTH,
            origin_x: m::SET_DST_ORIGIN_BYTES_X,
            origin_x_value: m::SET_DST_ORIGIN_BYTES_X_V,
            origin_y: m::SET_DST_ORIGIN_SAMPLES_Y,
            origin_y_value: m::SET_DST_ORIGIN_SAMPLES_Y_V,
            launch: m::LAUNCH_DMA,
            layout: m::LAUNCH_DMA_DST_MEMORY_LAYOUT,
            layout_pitch: m::LAUNCH_DMA_DST_MEMORY_LAYOUT_PITCH,
            completion: m::LAUNCH_DMA_COMPLETION_TYPE,
            completion_release: m::LAUNCH_DMA_COMPLETION_TYPE_RELEASE_SEMAPHORE,
            struct_size: m::LAUNCH_DMA_SEMAPHORE_STRUCT_SIZE,
            struct_four_words: m::LAUNCH_DMA_SEMAPHORE_STRUCT_SIZE_FOUR_WORDS,
            reduction_enable: m::LAUNCH_DMA_REDUCTION_ENABLE,
            reduction_enable_true: m::LAUNCH_DMA_REDUCTION_ENABLE_TRUE,
            reduction_op: m::LAUNCH_DMA_REDUCTION_OP,
            reduction_format: m::LAUNCH_DMA_REDUCTION_FORMAT,
            data: m::LOAD_INLINE_DATA,
            semaphore_a: m::SET_I2M_SEMAPHORE_A,
            semaphore_a_upper: m::SET_I2M_SEMAPHORE_A_OFFSET_UPPER,
            semaphore_b: m::SET_I2M_SEMAPHORE_B,
            semaphore_c: m::SET_I2M_SEMAPHORE_C,
        }
    }};
}

/// The inline-to-memory class, `A140`.
pub const INLINE_A140: InlineClass = inline_class!(cla140, "A140 LAUNCH_DMA");
/// The 3D class's inline upload, `C797`.
pub const INLINE_C797: InlineClass = inline_class!(clc797, "C797 LAUNCH_DMA");
/// The compute class's inline upload, `C7C0`.
pub const INLINE_C7C0: InlineClass = inline_class!(clc7c0, "C7C0 LAUNCH_DMA");

impl InlineClass {
    /// How many registers the engine reads in this class: one past its highest method.
    pub const fn registers_needed(&self) -> usize {
        registers_needed(&[
            self.line_length_in,
            self.line_count,
            self.offset_out_upper,
            self.offset_out,
            self.pitch_out,
            self.block_size,
            self.width,
            self.height,
            self.depth,
            self.origin_x,
            self.origin_y,
            self.launch,
            self.data,
            self.semaphore_a,
            self.semaphore_b,
            self.semaphore_c,
        ])
    }
}

/// A transfer in progress.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Transfer {
    destination: Surface,
    line_length: u64,
    total: u64,
    written: u64,
    release: Option<Release>,
}

/// One inline-to-memory engine: one per class instance of each channel (an `A140` object, and
/// the inline uploads of a channel's 3D and compute classes).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct InlineUpload {
    transfer: Option<Transfer>,
}

impl InlineUpload {
    /// An engine with no transfer.
    pub fn new() -> Self {
        Self::default()
    }

    /// The bytes the transfer in progress still expects; 0 when there is none.
    pub fn remaining(&self) -> u64 {
        self.transfer.map_or(0, |t| t.total - t.written)
    }

    /// Handles one method of `class`, called after the channel's register mirror for that class
    /// (`registers`, indexed by method number) has stored its value: `LAUNCH_DMA` starts a
    /// transfer and `LOAD_INLINE_DATA` writes the next word, as the module docs describe; every
    /// other method returns `Ok(())` at once.
    pub fn method<M, S>(
        &mut self,
        class: &InlineClass,
        method: u32,
        registers: &[u32],
        memory: &M,
        pending: &mut PendingWrites,
        sink: &mut S,
    ) -> Result<(), CopyError>
    where
        M: GpuMemory + ?Sized,
        S: CopySink + ?Sized,
    {
        if method == class.launch.method() {
            self.transfer = None;
            let regs = Registers::new(registers, class.registers_needed())?;
            return self.launch(class, regs, sink);
        }
        if method == class.data.method() {
            let regs = Registers::new(registers, class.registers_needed())?;
            let Some(mut transfer) = self.transfer.take() else {
                return Ok(());
            };
            transfer.word(memory, regs.get(class.data), pending)?;
            if transfer.written == transfer.total {
                if let Some(release) = transfer.release {
                    sink.release(release);
                }
            } else {
                self.transfer = Some(transfer);
            }
        }
        Ok(())
    }

    /// Starts a transfer.
    fn launch<S: CopySink + ?Sized>(
        &mut self,
        class: &InlineClass,
        regs: Registers<'_>,
        sink: &mut S,
    ) -> Result<(), CopyError> {
        let launch = regs.get(class.launch);
        let release = completion(class, regs, launch)?;
        let address = regs.address(
            class.offset_out_upper,
            class.offset_out_upper_value,
            class.offset_out,
        );
        let destination = if class.layout.get(launch) == class.layout_pitch {
            Surface::pitch(
                CopySide::Destination,
                address,
                i64::from(regs.get(class.pitch_out)),
            )
        } else {
            let block = regs.get(class.block_size);
            Surface::block_linear(
                CopySide::Destination,
                address,
                &BlockLinearDesc {
                    block: BlockFields {
                        width: class.block_width.get(block),
                        height: class.block_height.get(block),
                        depth: class.block_depth.get(block),
                    },
                    width_b: u64::from(regs.get(class.width)),
                    height: regs.get(class.height),
                    depth: regs.get(class.depth),
                    origin_x_b: u64::from(class.origin_x_value.get(regs.get(class.origin_x))),
                    origin_y: class.origin_y_value.get(regs.get(class.origin_y)),
                },
            )?
        };
        let line_length = u64::from(regs.get(class.line_length_in));
        let region = Region {
            width: line_length,
            lines: regs.get(class.line_count),
        };
        let cover = destination.cover(region)?;
        // Two 32-bit values: the product fits 64 bits.
        let total = line_length * u64::from(region.lines);
        if let Some(cover) = cover {
            sink.make_current(cover);
        }
        if total == 0 {
            if let Some(release) = release {
                sink.release(release);
            }
            return Ok(());
        }
        self.transfer = Some(Transfer {
            destination,
            line_length,
            total,
            written: 0,
            release,
        });
        Ok(())
    }
}

/// Decodes the completion semaphore: `None` unless `COMPLETION_TYPE` is `RELEASE_SEMAPHORE`.
fn completion(
    class: &InlineClass,
    regs: Registers<'_>,
    launch: u32,
) -> Result<Option<Release>, CopyError> {
    if class.completion.get(launch) != class.completion_release {
        return Ok(None);
    }
    let op = if class.reduction_enable.get(launch) == class.reduction_enable_true {
        ReleaseOp::Reduce(
            sema::red_reduction(
                class.launch_word,
                class.reduction_op.get(launch),
                class.reduction_format.get(launch),
            )
            .map_err(CopyError::Reduction)?,
        )
    } else {
        ReleaseOp::Write
    };
    Ok(Some(Release {
        address: regs.address(
            class.semaphore_a,
            class.semaphore_a_upper,
            class.semaphore_b,
        ),
        payload: u64::from(regs.get(class.semaphore_c)),
        form: ReleaseForm {
            size: PayloadSize::Four,
            timestamp: class.struct_size.get(launch) == class.struct_four_words,
        },
        op,
    }))
}

impl Transfer {
    /// The pieces of bytes `range` of the transfer, each inside one line: (line, x range within
    /// the line, range within the word).
    fn pieces(&self, range: Range<u64>) -> Vec<(u32, Range<u64>, Range<usize>)> {
        let mut pieces = Vec::new();
        let mut at = range.start;
        while at < range.end {
            let line = at / self.line_length;
            let x = at % self.line_length;
            let len = (self.line_length - x).min(range.end - at);
            // `at < total = line_length * lines`, so the line fits 32 bits; a word is 4 bytes.
            let (Ok(line), Ok(from), Ok(to)) = (
                u32::try_from(line),
                usize::try_from(at - range.start),
                usize::try_from(at - range.start + len),
            ) else {
                break;
            };
            pieces.push((line, x..x + len, from..to));
            at += len;
        }
        pieces
    }

    /// Writes the next bytes of `word`: probes each piece first, then writes them in order.
    fn word<M: GpuMemory + ?Sized>(
        &mut self,
        memory: &M,
        word: u32,
        pending: &mut PendingWrites,
    ) -> Result<(), CopyError> {
        let bytes = word.to_le_bytes();
        let count = (self.total - self.written).min(bytes.len() as u64);
        let pieces = self.pieces(self.written..self.written + count);
        for (line, x, _) in &pieces {
            self.destination
                .probe(memory, *line..*line + 1, x.clone())?;
        }
        for (line, x, within) in pieces {
            let data = bytes.get(within).unwrap_or_default();
            self.destination
                .write(memory, line, x, data, |run| pending.add(run))?;
        }
        self.written += count;
        Ok(())
    }
}
