//! DMA copies, inline-to-memory and 2D blits through guest memory, on the CPU (gpu-protocol
//! spec, "DMA copy, inline-to-memory and 2D" and "GPU writes and the CPU"; slice G11; Switch 2
//! scoping rows 69, 134 and 137).
//!
//! Three engines move bytes without drawing:
//!
//! - **DMA copy** (`C7B5`, [`dma_method`]): `LAUNCH_DMA` copies lines between pitch-linear and
//!   block-linear surfaces, optionally rearranging components (remap), and may release a
//!   semaphore once done ([`dma`] has the rules).
//! - **Inline-to-memory** (`A140`, and the same methods at bytes 0x180–0x1E4 of the 3D and
//!   compute classes; [`InlineUpload`]): `LAUNCH_DMA` names a destination and the following
//!   `LOAD_INLINE_DATA` words are written to it in order ([`inline`] has the rules).
//! - **2D** (`902D`, [`blit_method`]): surface-to-surface blits with scaling and filtering
//!   ([`twod`] has the rules). Which 2D class NVN2 binds is not in any reference: `902D` is a
//!   **guess**, as the spec says (Mesa has no Ampere 2D class; verification item 10, a bind
//!   trace, confirms it).
//!
//! These are the CPU paths. When the texture cache (slice G17) holds a surface, the spec turns a
//! copy or blit into a host command instead; that choice joins with the executor (G19). Every
//! block-linear byte offset comes from the layout crate ([`starquake_layout`]; decision 5), and
//! the shared walker ([`surface`]) is the one place that turns a region into GPU addresses.
//!
//! # What reaches the sink
//!
//! A [`CopySink`] receives, for each operation that moves bytes, in this order:
//!
//! 1. [`CopySink::invalidate`] for each range of [`PendingWrites`] (the merged inline writes),
//!    when a DMA copy or 2D blit is launched: "Cache invalidations from many small writes are
//!    merged until the next draw, dispatch or copy" (row 134). The executor flushes them before
//!    each draw and dispatch with [`PendingWrites::flush`].
//! 2. [`CopySink::make_current`] for the GPU range covering the source, then the destination:
//!    memory read, or partly overwritten, as plain bytes, which is where an image the caches
//!    hold must be written back first (decision 4: write-back only when something reads the
//!    memory as plain bytes, "which a copy does").
//! 3. [`CopySink::invalidate`] for the range covering the destination, after the bytes are
//!    written (an inline word's runs go to [`PendingWrites`] instead).
//! 4. [`CopySink::release`] for the semaphore, if the launch asks for one: a fenced release,
//!    performed by the fence path (slice G15) once the host GPU has finished the work before it
//!    (decision 2).
//!
//! Pinned by `tests::dma::the_sink_sees_pending_then_current_then_invalidate_then_release`.
//!
//! # Failures and what they leave
//!
//! Each operation does every check that does not write before its first write: decoding the
//! registers and the semaphore, the surfaces, the size bound, and a probe that reads one byte of
//! every page the source and then the destination reach. So a register error, an undefined
//! field, a region outside its surface, an unmapped page on either side or a copy past
//! [`MAX_COPY_BYTES`] writes nothing and releases nothing; the next launch is unaffected. Pinned
//! by `tests::dma::a_fault_found_by_the_probe_writes_nothing_and_the_next_launch_works`.
//!
//! Two failures can still come after writing has begun, and keep what was written before them:
//! a guest memory fault on a page the probe found mapped (a write-protected page, or a mapping
//! removed between the probe and the write), and, for an inline upload, a data word whose page
//! is unmapped (each word is probed and written on its own as it arrives). Pinned by
//! `tests::inline::a_word_that_faults_keeps_the_words_before_it`.

mod dma;
mod inline;
pub(crate) mod surface;
mod twod;

#[cfg(test)]
mod tests;

use core::ops::Range;
use std::collections::BTreeMap;

use starquake_layout::LayoutError;
use thiserror::Error;

use crate::classes::Method;
use crate::memory::GpuFault;
use crate::ring::Release;
use crate::sema::DecodeError;

pub use dma::{DMA_REGISTERS_NEEDED, dma_method};
pub use inline::{INLINE_A140, INLINE_C797, INLINE_C7C0, InlineClass, InlineUpload};
pub use surface::CopySide;
pub use twod::{BILINEAR_FORMATS, BLIT_REGISTERS_NEEDED, blit_method, bytes_per_pixel};

/// The most bytes one DMA copy or 2D blit may move: a copy whose larger side (source or
/// destination bytes) exceeds this is [`CopyError::TooLarge`], checked before anything is read.
///
/// A **guess** with no reference behind it: Eden-NX2 has no bound and sizes its buffers from the
/// registers (`maxwell_dma.cpp`, `pitch_in * line_count`), so a hostile launch could ask it for
/// any amount. 256 MiB holds a 4096 by 4096 surface of 16-byte pixels, the largest the 2D class
/// names (`RF32_GF32_BF32_AF32`); it bounds the time one launch can take and the memory a 2D row
/// can hold.
pub const MAX_COPY_BYTES: u64 = 256 << 20;

/// Whether a copy of `bytes` is within [`MAX_COPY_BYTES`].
pub(crate) fn check_size(bytes: u128) -> Result<(), CopyError> {
    if bytes > u128::from(MAX_COPY_BYTES) {
        return Err(CopyError::TooLarge {
            bytes,
            max: MAX_COPY_BYTES,
        });
    }
    Ok(())
}

/// Why a copy, inline upload or blit failed. Every variant is reported before the first byte
/// is written, except the two the module docs name.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Error)]
pub enum CopyError {
    /// The register file is shorter than the methods the engine reads.
    #[error("the register file holds {len} registers, the engine reads {needed}")]
    Registers {
        /// Its length.
        len: usize,
        /// How many the engine reads (one past the highest method it reads).
        needed: usize,
    },
    /// Reading or writing one side faulted.
    #[error("the copy's {side}: {fault}")]
    Fault {
        /// The side whose access faulted.
        side: CopySide,
        /// The fault, naming the first byte that faulted.
        fault: GpuFault,
    },
    /// An address of one side is outside the 64-bit GPU address space.
    #[error("the copy's {side} runs past the end of the GPU address space")]
    Extent {
        /// The side.
        side: CopySide,
    },
    /// A block-linear side's description or region does not fit its surface.
    #[error("the copy's {side} surface: {error}")]
    Layout {
        /// The side.
        side: CopySide,
        /// What the layout crate refused.
        error: LayoutError,
    },
    /// A block width other than one GOB, the only width the classes define.
    #[error("the copy's {side} block width field is {value}, but only one GOB (0) is defined")]
    BlockWidth {
        /// The side.
        side: CopySide,
        /// The field.
        value: u32,
    },
    /// A block height past `THIRTYTWO_GOBS`.
    #[error("the copy's {side} block height field {value} is not defined")]
    BlockHeight {
        /// The side.
        side: CopySide,
        /// The field.
        value: u32,
    },
    /// A block depth past `THIRTYTWO_GOBS`.
    #[error("the copy's {side} block depth field {value} is not defined")]
    BlockDepth {
        /// The side.
        side: CopySide,
        /// The field.
        value: u32,
    },
    /// A DMA side addressed physically (`SRC_TYPE`/`DST_TYPE` `PHYSICAL`).
    #[error("the copy's {side} uses a physical address, which is not supported")]
    Physical {
        /// The side.
        side: CopySide,
    },
    /// The copy moves more than [`MAX_COPY_BYTES`].
    #[error("the copy moves {bytes} bytes, more than the {max} allowed")]
    TooLarge {
        /// The bytes of its larger side.
        bytes: u128,
        /// [`MAX_COPY_BYTES`].
        max: u64,
    },
    /// A DMA `SEMAPHORE_TYPE` this module does not perform (`RELEASE_CONDITIONAL_INTR_SEMAPHORE`).
    #[error("LAUNCH_DMA semaphore type {0} is not supported")]
    SemaphoreType(u32),
    /// A DMA float reduction (`FADD`, `FMIN`, `FMAX`), which the semaphore path cannot perform.
    #[error("LAUNCH_DMA float reduction {0:#x} is not supported")]
    FloatReduction(u32),
    /// A semaphore reduction or format the header does not define.
    #[error("{0}")]
    Reduction(DecodeError),
    /// A remap destination component whose selector is not defined (7).
    #[error("remap component {component} selector {value} is not defined")]
    RemapSelector {
        /// 0 to 3 for `DST_X` to `DST_W`.
        component: u32,
        /// The selector.
        value: u32,
    },
    /// A remap destination component reads a source component past `NUM_SRC_COMPONENTS`.
    #[error("remap component {component} reads source component {read} of {sources}")]
    RemapSource {
        /// 0 to 3 for `DST_X` to `DST_W`.
        component: u32,
        /// The source component it reads, 0 to 3.
        read: u32,
        /// How many source components there are.
        sources: u32,
    },
    /// A 2D operation other than `SRCCOPY`.
    #[error("2D operation {0} is not supported, only SRCCOPY (3) is")]
    BlitOperation(u32),
    /// A 2D blit with clipping enabled.
    #[error("2D blits with clipping enabled are not supported")]
    BlitClip,
    /// A 2D surface format this module has no size for.
    #[error("the blit's {side} format {format:#x} is not supported")]
    BlitFormat {
        /// The side.
        side: CopySide,
        /// The `SET_SRC_FORMAT`/`SET_DST_FORMAT` value.
        format: u32,
    },
    /// A 2D blit between different formats, which needs a format conversion.
    #[error("a 2D blit from format {from:#x} to format {to:#x} needs a conversion, which is not supported")]
    BlitConversion {
        /// The source format.
        from: u32,
        /// The destination format.
        to: u32,
    },
    /// A scaled bilinear 2D blit of a format bilinear filtering is not implemented for.
    #[error("bilinear filtering of format {0:#x} is not supported")]
    BlitFilter(u32),
    /// A 2D rectangle with a negative start or an end before its start.
    #[error("the blit's {side} rectangle is negative or inverted")]
    BlitRect {
        /// The side.
        side: CopySide,
    },
}

/// Where the copy engines' side effects go. The back's executor (G19) implements it: it hands
/// [`CopySink::make_current`] and [`CopySink::invalidate`] to the caches (G16, G17) and
/// [`CopySink::release`] to the fence path (G15). The module docs give the order of calls.
pub trait CopySink {
    /// The engine is about to read, or partly overwrite, `range` as plain bytes: anything that
    /// holds a newer copy of it than guest memory (an image the texture cache rendered to) must
    /// write it back first.
    fn make_current(&mut self, range: Range<u64>);
    /// `range` of GPU memory has been written: caches holding any of it must drop it.
    fn invalidate(&mut self, range: Range<u64>);
    /// A semaphore release, to be performed once the work before it has finished on the host
    /// GPU.
    fn release(&mut self, release: Release);
}

/// The most disjoint ranges [`PendingWrites`] keeps: a **guess** (no reference bounds Eden-NX2's
/// list). One more is merged with all the others into the single range covering them, so a
/// stream of scattered writes costs bounded memory and over-invalidates instead.
pub const MAX_PENDING_RANGES: usize = 1024;

/// Inline writes whose cache invalidations are held back, merged, until the next draw, dispatch
/// or copy (row 134).
///
/// Ranges are kept sorted and disjoint: a range that overlaps or touches one already held is
/// merged with it. [`PendingWrites::flush`] hands them to the sink in increasing address order.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PendingWrites {
    /// End by start; disjoint and not touching.
    ranges: BTreeMap<u64, u64>,
}

impl PendingWrites {
    /// Nothing pending.
    pub fn new() -> Self {
        Self::default()
    }

    /// The pending ranges, in increasing address order.
    pub fn ranges(&self) -> Vec<Range<u64>> {
        self.ranges.iter().map(|(&start, &end)| start..end).collect()
    }

    /// Whether nothing is pending.
    pub fn is_empty(&self) -> bool {
        self.ranges.is_empty()
    }

    /// Adds `range`, merging it with every held range it overlaps or touches. An empty range is
    /// ignored. Past [`MAX_PENDING_RANGES`] disjoint ranges, everything becomes one range from
    /// the lowest start to the highest end.
    pub fn add(&mut self, range: Range<u64>) {
        if range.is_empty() {
            return;
        }
        let mut start = range.start;
        let mut end = range.end;
        // The held range starting at or before `start` may reach it.
        if let Some((&s, &e)) = self.ranges.range(..=start).next_back()
            && e >= start
        {
            start = s;
            end = end.max(e);
        }
        // Every held range starting inside `start..=end` merges.
        let inside: Vec<(u64, u64)> = self
            .ranges
            .range(start..=end)
            .map(|(&s, &e)| (s, e))
            .collect();
        for (s, e) in inside {
            self.ranges.remove(&s);
            end = end.max(e);
        }
        self.ranges.insert(start, end);
        if self.ranges.len() > MAX_PENDING_RANGES {
            let low = self.ranges.first_key_value().map_or(start, |(&s, _)| s);
            let high = self.ranges.values().copied().max().unwrap_or(end);
            self.ranges.clear();
            self.ranges.insert(low, high);
        }
    }

    /// Hands every pending range to `sink` ([`CopySink::invalidate`]), in increasing address
    /// order, and empties the list.
    pub fn flush<S: CopySink + ?Sized>(&mut self, sink: &mut S) {
        for (start, end) in core::mem::take(&mut self.ranges) {
            sink.invalidate(start..end);
        }
    }
}

/// A register file checked to reach the highest method an engine reads.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Registers<'a> {
    registers: &'a [u32],
}

impl<'a> Registers<'a> {
    /// `registers`, which must hold at least `needed` registers ([`CopyError::Registers`]).
    pub fn new(registers: &'a [u32], needed: usize) -> Result<Self, CopyError> {
        if registers.len() < needed {
            return Err(CopyError::Registers {
                len: registers.len(),
                needed,
            });
        }
        Ok(Self { registers })
    }

    /// The value of `method` (element 0 of an array method). A method past the file reads as
    /// zero; every caller's `needed` covers the methods it reads (pinned by each engine's
    /// `*_registers_needed_covers_every_method_read` test).
    pub fn get(self, method: Method) -> u32 {
        usize::try_from(method.method())
            .ok()
            .and_then(|index| self.registers.get(index))
            .copied()
            .unwrap_or(0)
    }

    /// A 64-bit address from an upper register's field and a lower register.
    pub fn address(self, upper: Method, upper_field: crate::field::Field, lower: Method) -> u64 {
        (u64::from(upper_field.get(self.get(upper))) << u32::BITS) | u64::from(self.get(lower))
    }
}

/// One past the highest method number among `methods`, as a register count.
pub(crate) const fn registers_needed(methods: &[Method]) -> usize {
    let mut highest = 0;
    let mut i = 0;
    while i < methods.len() {
        let method = methods[i].method();
        if method > highest {
            highest = method;
        }
        i += 1;
    }
    highest as usize + 1
}
