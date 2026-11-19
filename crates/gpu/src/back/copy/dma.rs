//! The DMA copy class, `C7B5` (`AMPERE_DMA_COPY_A`): `LAUNCH_DMA` (byte 0x300).
//!
//! # Sources
//!
//! Paths are relative to the reference folder (`tools/refs-path.sh`), at the commits pinned in
//! `docs/references.md`.
//!
//! - Every method and field is read through [`crate::classes::clc7b5`], generated from
//!   `mesa-nouveau/src/nouveau/headers/nvidia/classes/clc7b5.h` @ 52238762 (Copyright (c) 2020,
//!   NVIDIA CORPORATION, MIT licence).
//! - The behaviour is ported from `MaxwellDMA::Launch`, `CopyPitchToBlockLinear`,
//!   `CopyBlockLinearToPitch`, `CopyBlockLinearToBlockLinear` and `ReleaseSemaphore` in
//!   `eden-nx2/src/video_core/engines/maxwell_dma.cpp` @ 50fa86bb (Copyright 2026 Eden Emulator
//!   Project, GPL-3.0-or-later; Copyright 2018 yuzu Emulator Project, GPL-2.0-or-later), with the
//!   register layout of `maxwell_dma.h`.
//!
//! # What a launch does
//!
//! The copy is `LINE_COUNT` lines (one line when `MULTI_LINE_ENABLE` is clear) of
//! `LINE_LENGTH_IN` elements. Without remap an element is one byte; with `REMAP_ENABLE` a source
//! element is `NUM_SRC_COMPONENTS` components of `COMPONENT_SIZE` bytes and a destination
//! element `NUM_DST_COMPONENTS` of them, each destination component taken from a source
//! component, `SET_REMAP_CONST_A` or `_B` (their low `COMPONENT_SIZE` bytes, little-endian), or
//! left as it was (`NO_WRITE`). Each side is pitch-linear or block-linear by its
//! `*_MEMORY_LAYOUT` bit (7 for the source, 8 for the destination):
//!
//! - pitch: line `l` at `OFFSET_* + l * PITCH_*`, the pitch a signed 32-bit value as
//!   Eden-NX2 types it (`s32 pitch_in`);
//! - block-linear: the region at `SET_*_ORIGIN` of a surface `SET_*_WIDTH` by `_HEIGHT` by
//!   `_DEPTH` with `SET_*_BLOCK_SIZE`, the width and origin x counted in elements (in bytes
//!   without remap), as Eden-NX2's `bytes_per_pixel` makes them; see [`super::surface`] for how
//!   lines continue into the next depth slice.
//!
//! The checks run in this order, each before anything is written: the register file
//! ([`CopyError::Registers`]); the semaphore ([`CopyError::SemaphoreType`],
//! [`CopyError::FloatReduction`], [`CopyError::Reduction`]); then, unless
//! `DATA_TRANSFER_TYPE` is `NONE`, the source's then the destination's address type
//! ([`CopyError::Physical`]); the remap ([`CopyError::RemapSelector`],
//! [`CopyError::RemapSource`], in component order); the size ([`CopyError::TooLarge`]); the
//! source surface, then the destination surface ([`super::surface::Surface::block_linear`]); the
//! source region, then the destination region (`Surface::cover`); a probe of every page of the
//! source, then of the destination ([`CopyError::Fault`]). Then the bytes move, line by line in
//! increasing line order, each line in chunks of at most [`CHUNK_BYTES`] in increasing x, each
//! chunk read whole before it is written.
//!
//! # Divergences from Eden-NX2
//!
//! Each is pinned by the test named, in `copy/tests/dma.rs`.
//!
//! - **Upper address bits.** `OFFSET_IN_UPPER`, `OFFSET_OUT_UPPER` and `SET_SEMAPHORE_A` hold 17
//!   bits (16:0); Eden-NX2's `PackedGPUVAddr` keeps 8 (`upper & 0xff`). Pinned by
//!   `divergence_upper_address_bits_are_seventeen`.
//! - **`DATA_TRANSFER_TYPE_NONE` moves nothing** and only releases the semaphore; Eden-NX2
//!   ignores the field and copies. Pinned by `divergence_transfer_type_none_moves_nothing`.
//! - **The layout bits decide a one-line copy's layouts.** Without `MULTI_LINE_ENABLE`, Eden-NX2
//!   picks each side's layout from its page kind (`IsPitchKind`) and swizzles a block-linear side
//!   within each 512-byte GOB only (`convert_linear_2_blocklinear_addr`); here the layout bits
//!   and the full surface parameters apply to one line, as for a multi-line copy (this module's
//!   memory carries no page kinds). Pinned by `divergence_a_one_line_copy_follows_the_layout_bits`.
//! - **Remap rearranges components on every path.** Eden-NX2 remaps only a one-line copy whose
//!   `DST_X` is `CONST_A`, filling `LINE_LENGTH_IN * COMPONENT_SIZE` bytes with the repeated
//!   32-bit constant (so a 1- or 2-byte component gets the constant's bytes in turn, not its low
//!   bytes); a multi-line copy only uses `NUM_DST_COMPONENTS * COMPONENT_SIZE` as the element
//!   size of both sides and copies bytes unchanged, and a pitch-to-pitch copy ignores remap and
//!   copies `LINE_LENGTH_IN` bytes. Here every launch with remap applies the selectors as the
//!   header's field names describe, with source elements of `NUM_SRC_COMPONENTS` components.
//!   The selectors' meaning beyond Eden-NX2's constant fill is a **guess** from the header's
//!   names; a trace of a remapped copy settles it. Pinned by
//!   `divergence_remap_applies_the_selectors_on_every_path` and
//!   `divergence_one_byte_constant_fill_uses_the_low_byte`.
//! - **A block-linear region must lie inside its surface.** Eden-NX2 does not bound x, and drops
//!   lines past the last depth slice; here either is [`CopyError::Layout`] with
//!   [`starquake_layout::LayoutError::RegionOutOfBounds`]. Pinned by
//!   `divergence_a_region_past_the_surface_is_refused`.
//! - **Lines that continue into the next depth slice keep the pitch side's stride.** On a
//!   block-linear side with `origin_y` above 0, Eden-NX2's linear offset for slice `s` is
//!   `s * pitch * height + line * pitch`, skipping `origin_y` lines per slice; here line `l` of the
//!   pitch side is always `l * pitch`. Pinned by `divergence_slices_keep_the_pitch_stride`.
//! - **Signed pitch on every path.** Eden-NX2 uses `|PITCH_OUT|` for a block-linear-to-pitch
//!   copy and the pitch as an unsigned stride for pitch-to-block-linear; here both pitches are
//!   signed everywhere. Pinned by `divergence_a_negative_pitch_walks_down_on_every_path`.
//! - **The semaphore's payload size and reduction are honoured.** Eden-NX2 always releases a
//!   32-bit payload (`regs.semaphore.payload`) and never reduces; here `SEMAPHORE_PAYLOAD_SIZE`
//!   `TWO_WORD` releases 64 bits (`SET_SEMAPHORE_PAYLOAD_UPPER` above `SET_SEMAPHORE_PAYLOAD`) and
//!   `SEMAPHORE_REDUCTION_ENABLE` makes it a reduction (the host reductions `IMIN` to `DEC`,
//!   signed by `SEMAPHORE_REDUCTION_SIGN`). Pinned by `semaphore_fields_give_the_exact_release`.
//! - **Refused where Eden-NX2 goes on:** `RELEASE_CONDITIONAL_INTR_SEMAPHORE` (Eden-NX2 asserts),
//!   the float reductions `FADD`, `FMIN`, `FMAX` and the undefined `INVALID*` values, a physical
//!   source or destination (Eden-NX2 ignores `SRC_TYPE` and `DST_TYPE`), a block width other than
//!   one GOB (Eden-NX2's `UNIMPLEMENTED_IF`, which goes on), a block height or depth past
//!   `THIRTYTWO_GOBS` (Eden-NX2 shifts by any 4-bit value), and a copy past
//!   [`super::MAX_COPY_BYTES`]. Pinned by `refusals_are_exact`,
//!   `address_remap_and_block_refusals_are_exact` and `the_size_bound_is_checked_at_the_edge`.
//! - **Not read, as in Eden-NX2:** `INTERRUPT_TYPE` (Eden-NX2 asserts `NONE`; nothing here raises
//!   GPU interrupts, so any value copies), `SET_*_LAYER` (Eden-NX2 refuses a non-zero
//!   destination layer only on pitch-to-block-linear), the block size's `GOB_HEIGHT`, the 32-bit
//!   `SRC_ORIGIN_X/Y` and `DST_ORIGIN_X/Y` methods (Eden-NX2 reads the packed `SET_*_ORIGIN`;
//!   whether NVN2 ever writes the others is a **guess**), `FLUSH_*`, `SET_*_PHYS_MODE` and the
//!   render enable. Pinned by `fields_that_are_not_read_change_nothing`.

use core::ops::Range;

use super::surface::{BlockFields, BlockLinearDesc, CopySide, Region, Surface};
use super::{CopyError, CopySink, PendingWrites, Registers, check_size, registers_needed};
use crate::classes::clc7b5 as c;
use crate::memory::GpuMemory;
use crate::ring::{Release, ReleaseOp};
use crate::sema::{self, PayloadSize, ReleaseForm};

/// The word name semaphore decode errors carry.
pub(crate) const LAUNCH_DMA_WORD: &str = "LAUNCH_DMA";

/// The register file must reach every method a launch reads.
pub const DMA_REGISTERS_NEEDED: usize = registers_needed(&[
    c::LAUNCH_DMA,
    c::OFFSET_IN_UPPER,
    c::OFFSET_IN_LOWER,
    c::OFFSET_OUT_UPPER,
    c::OFFSET_OUT_LOWER,
    c::PITCH_IN,
    c::PITCH_OUT,
    c::LINE_LENGTH_IN,
    c::LINE_COUNT,
    c::SET_REMAP_CONST_A,
    c::SET_REMAP_CONST_B,
    c::SET_REMAP_COMPONENTS,
    c::SET_DST_BLOCK_SIZE,
    c::SET_DST_WIDTH,
    c::SET_DST_HEIGHT,
    c::SET_DST_DEPTH,
    c::SET_DST_ORIGIN,
    c::SET_SRC_BLOCK_SIZE,
    c::SET_SRC_WIDTH,
    c::SET_SRC_HEIGHT,
    c::SET_SRC_DEPTH,
    c::SET_SRC_ORIGIN,
    c::SET_SEMAPHORE_A,
    c::SET_SEMAPHORE_B,
    c::SET_SEMAPHORE_PAYLOAD,
    c::SET_SEMAPHORE_PAYLOAD_UPPER,
]);

/// The most bytes moved through one buffer at a time: each line is copied in chunks of at most
/// this many bytes of its larger side, so a launch holds at most two such buffers.
pub(crate) const CHUNK_BYTES: u64 = 64 << 10;

/// Handles one DMA copy method, called after the channel's `C7B5` register mirror
/// (`registers`, indexed by method number) has stored its value. Only `LAUNCH_DMA` does
/// anything: it first flushes `pending` to `sink`, then copies and releases as the module docs
/// describe. Every other method returns `Ok(())` at once.
pub fn dma_method<M, S>(
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
    if method != c::LAUNCH_DMA.method() {
        return Ok(());
    }
    pending.flush(sink);
    let regs = Registers::new(registers, DMA_REGISTERS_NEEDED)?;
    let launch = regs.get(c::LAUNCH_DMA);
    let release = semaphore(regs, launch)?;
    if c::LAUNCH_DMA_DATA_TRANSFER_TYPE.get(launch) != c::LAUNCH_DMA_DATA_TRANSFER_TYPE_NONE {
        Plan::new(regs, launch)?.run(memory, sink)?;
    }
    if let Some(release) = release {
        sink.release(release);
    }
    Ok(())
}

/// Decodes the launch's semaphore: `None` for `SEMAPHORE_TYPE_NONE`.
fn semaphore(regs: Registers<'_>, launch: u32) -> Result<Option<Release>, CopyError> {
    let timestamp = match c::LAUNCH_DMA_SEMAPHORE_TYPE.get(launch) {
        c::LAUNCH_DMA_SEMAPHORE_TYPE_NONE => return Ok(None),
        c::LAUNCH_DMA_SEMAPHORE_TYPE_RELEASE_ONE_WORD_SEMAPHORE => false,
        c::LAUNCH_DMA_SEMAPHORE_TYPE_RELEASE_FOUR_WORD_SEMAPHORE => true,
        other => return Err(CopyError::SemaphoreType(other)),
    };
    let low = u64::from(regs.get(c::SET_SEMAPHORE_PAYLOAD));
    let (size, payload) = if c::LAUNCH_DMA_SEMAPHORE_PAYLOAD_SIZE.get(launch)
        == c::LAUNCH_DMA_SEMAPHORE_PAYLOAD_SIZE_TWO_WORD
    {
        let high = u64::from(regs.get(c::SET_SEMAPHORE_PAYLOAD_UPPER));
        (PayloadSize::Eight, (high << u32::BITS) | low)
    } else {
        (PayloadSize::Four, low)
    };
    let op = if c::LAUNCH_DMA_SEMAPHORE_REDUCTION_ENABLE.get(launch)
        == c::LAUNCH_DMA_SEMAPHORE_REDUCTION_ENABLE_TRUE
    {
        let reduction = c::LAUNCH_DMA_SEMAPHORE_REDUCTION.get(launch);
        if matches!(
            reduction,
            c::LAUNCH_DMA_SEMAPHORE_REDUCTION_FADD
                | c::LAUNCH_DMA_SEMAPHORE_REDUCTION_FMIN
                | c::LAUNCH_DMA_SEMAPHORE_REDUCTION_FMAX
        ) {
            return Err(CopyError::FloatReduction(reduction));
        }
        ReleaseOp::Reduce(
            sema::host_reduction(
                LAUNCH_DMA_WORD,
                reduction,
                c::LAUNCH_DMA_SEMAPHORE_REDUCTION_SIGN.get(launch),
            )
            .map_err(CopyError::Reduction)?,
        )
    } else {
        ReleaseOp::Write
    };
    Ok(Some(Release {
        address: regs.address(
            c::SET_SEMAPHORE_A,
            c::SET_SEMAPHORE_A_UPPER,
            c::SET_SEMAPHORE_B,
        ),
        payload,
        form: ReleaseForm { size, timestamp },
        op,
    }))
}

/// What one destination component of a remapped element holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Select {
    /// Source component `n` (0 to 3).
    Source(u32),
    /// `SET_REMAP_CONST_A`'s low bytes.
    ConstA,
    /// `SET_REMAP_CONST_B`'s low bytes.
    ConstB,
    /// Left as it was.
    NoWrite,
}

/// The four destination selectors, `DST_X` to `DST_W`.
const SELECTORS: [crate::field::Field; 4] = [
    c::SET_REMAP_COMPONENTS_DST_X,
    c::SET_REMAP_COMPONENTS_DST_Y,
    c::SET_REMAP_COMPONENTS_DST_Z,
    c::SET_REMAP_COMPONENTS_DST_W,
];

/// A decoded `SET_REMAP_COMPONENTS` with its constants.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Remap {
    /// `COMPONENT_SIZE + 1` bytes.
    size: usize,
    /// `NUM_SRC_COMPONENTS + 1`.
    sources: usize,
    /// The first `NUM_DST_COMPONENTS + 1` selectors.
    selects: [Select; 4],
    /// `NUM_DST_COMPONENTS + 1`.
    destinations: usize,
    /// `SET_REMAP_CONST_A`.
    const_a: u32,
    /// `SET_REMAP_CONST_B`.
    const_b: u32,
}

impl Remap {
    /// Decodes the remap registers. Only the selectors of the `NUM_DST_COMPONENTS + 1`
    /// components written are checked, in component order.
    fn decode(regs: Registers<'_>) -> Result<Self, CopyError> {
        let word = regs.get(c::SET_REMAP_COMPONENTS);
        // Each count field is two bits, so these are 1 to 4.
        let count = |field: crate::field::Field| field.get(word) + 1;
        let size = count(c::SET_REMAP_COMPONENTS_COMPONENT_SIZE);
        let sources = count(c::SET_REMAP_COMPONENTS_NUM_SRC_COMPONENTS);
        let destinations = count(c::SET_REMAP_COMPONENTS_NUM_DST_COMPONENTS);
        let mut selects = [Select::NoWrite; 4];
        for (component, (select, field)) in
            (0u32..destinations).zip(selects.iter_mut().zip(SELECTORS))
        {
            // The DST_Y, _Z and _W values equal DST_X's
            // (`tests::dma::remap_selector_values_match_the_header_written_out`).
            *select = match field.get(word) {
                value @ (c::SET_REMAP_COMPONENTS_DST_X_SRC_X
                | c::SET_REMAP_COMPONENTS_DST_X_SRC_Y
                | c::SET_REMAP_COMPONENTS_DST_X_SRC_Z
                | c::SET_REMAP_COMPONENTS_DST_X_SRC_W) => {
                    let read = value - c::SET_REMAP_COMPONENTS_DST_X_SRC_X;
                    if read >= sources {
                        return Err(CopyError::RemapSource {
                            component,
                            read,
                            sources,
                        });
                    }
                    Select::Source(read)
                }
                c::SET_REMAP_COMPONENTS_DST_X_CONST_A => Select::ConstA,
                c::SET_REMAP_COMPONENTS_DST_X_CONST_B => Select::ConstB,
                c::SET_REMAP_COMPONENTS_DST_X_NO_WRITE => Select::NoWrite,
                value => return Err(CopyError::RemapSelector { component, value }),
            };
        }
        Ok(Self {
            size: size as usize,
            sources: sources as usize,
            selects,
            destinations: destinations as usize,
            const_a: regs.get(c::SET_REMAP_CONST_A),
            const_b: regs.get(c::SET_REMAP_CONST_B),
        })
    }

    /// Bytes of one source element.
    fn source_bytes(&self) -> usize {
        self.size * self.sources
    }

    /// Bytes of one destination element.
    fn destination_bytes(&self) -> usize {
        self.size * self.destinations
    }

    /// Whether a destination component is left as it was, so the destination must be read first.
    fn keeps_destination(&self) -> bool {
        self.selects[..self.destinations].contains(&Select::NoWrite)
    }

    /// Remaps whole elements of `source` into `destination`.
    fn apply(&self, source: &[u8], destination: &mut [u8]) {
        let a = self.const_a.to_le_bytes();
        let b = self.const_b.to_le_bytes();
        for (input, output) in source
            .chunks_exact(self.source_bytes())
            .zip(destination.chunks_exact_mut(self.destination_bytes()))
        {
            for (select, out) in self.selects[..self.destinations]
                .iter()
                .zip(output.chunks_exact_mut(self.size))
            {
                let bytes = match *select {
                    Select::Source(n) => {
                        let n = n as usize;
                        input.get(n * self.size..(n + 1) * self.size)
                    }
                    Select::ConstA => a.get(..self.size),
                    Select::ConstB => b.get(..self.size),
                    Select::NoWrite => None,
                };
                if let Some(bytes) = bytes {
                    out.copy_from_slice(bytes);
                }
            }
        }
    }
}

/// A launch that moves bytes, checked.
struct Plan {
    source: Surface,
    destination: Surface,
    remap: Option<Remap>,
    /// Elements per line.
    elements: u64,
    lines: u32,
    source_region: Region,
    destination_region: Region,
    source_cover: Option<Range<u64>>,
    destination_cover: Option<Range<u64>>,
}

impl Plan {
    /// Checks a launch in the module docs' order, from the address types to the regions.
    fn new(regs: Registers<'_>, launch: u32) -> Result<Self, CopyError> {
        if c::LAUNCH_DMA_SRC_TYPE.get(launch) == c::LAUNCH_DMA_SRC_TYPE_PHYSICAL {
            return Err(CopyError::Physical {
                side: CopySide::Source,
            });
        }
        if c::LAUNCH_DMA_DST_TYPE.get(launch) == c::LAUNCH_DMA_DST_TYPE_PHYSICAL {
            return Err(CopyError::Physical {
                side: CopySide::Destination,
            });
        }
        let remap = (c::LAUNCH_DMA_REMAP_ENABLE.get(launch) == c::LAUNCH_DMA_REMAP_ENABLE_TRUE)
            .then(|| Remap::decode(regs))
            .transpose()?;
        let (source_bytes, destination_bytes) =
            remap.map_or((1, 1), |r| (r.source_bytes(), r.destination_bytes()));
        let lines = if c::LAUNCH_DMA_MULTI_LINE_ENABLE.get(launch)
            == c::LAUNCH_DMA_MULTI_LINE_ENABLE_TRUE
        {
            regs.get(c::LINE_COUNT)
        } else {
            1
        };
        let elements = u64::from(regs.get(c::LINE_LENGTH_IN));
        // At most 2^32 elements of at most 16 bytes: these fit 64 bits.
        let source_region = Region {
            width: elements * source_bytes as u64,
            lines,
        };
        let destination_region = Region {
            width: elements * destination_bytes as u64,
            lines,
        };
        check_size(source_region.bytes().max(destination_region.bytes()))?;
        let source = side(
            regs,
            CopySide::Source,
            c::LAUNCH_DMA_SRC_MEMORY_LAYOUT.get(launch) == c::LAUNCH_DMA_SRC_MEMORY_LAYOUT_PITCH,
            source_bytes as u64,
        )?;
        let destination = side(
            regs,
            CopySide::Destination,
            c::LAUNCH_DMA_DST_MEMORY_LAYOUT.get(launch) == c::LAUNCH_DMA_DST_MEMORY_LAYOUT_PITCH,
            destination_bytes as u64,
        )?;
        let source_cover = source.cover(source_region)?;
        let destination_cover = destination.cover(destination_region)?;
        Ok(Self {
            source,
            destination,
            remap,
            elements,
            lines,
            source_region,
            destination_region,
            source_cover,
            destination_cover,
        })
    }

    /// Probes both sides, then moves the bytes.
    fn run<M, S>(&self, memory: &M, sink: &mut S) -> Result<(), CopyError>
    where
        M: GpuMemory + ?Sized,
        S: CopySink + ?Sized,
    {
        let (Some(source_cover), Some(destination_cover)) =
            (self.source_cover.clone(), self.destination_cover.clone())
        else {
            return Ok(());
        };
        self.source
            .probe(memory, 0..self.lines, 0..self.source_region.width)?;
        self.destination
            .probe(memory, 0..self.lines, 0..self.destination_region.width)?;
        sink.make_current(source_cover);
        sink.make_current(destination_cover.clone());
        let (source_bytes, destination_bytes) = self.remap.map_or((1, 1), |r| {
            (r.source_bytes() as u64, r.destination_bytes() as u64)
        });
        let per_chunk = (CHUNK_BYTES / source_bytes.max(destination_bytes)).max(1);
        let mut input = Vec::new();
        let mut output = Vec::new();
        for line in 0..self.lines {
            let mut first = 0;
            while first < self.elements {
                let last = (first + per_chunk).min(self.elements);
                let from = first * source_bytes..last * source_bytes;
                let to = first * destination_bytes..last * destination_bytes;
                input.resize(buffer_len(&from)?, 0);
                self.source.read(memory, line, from, &mut input)?;
                match &self.remap {
                    None => self.destination.write(memory, line, to, &input, |_| ())?,
                    Some(remap) => {
                        output.clear();
                        output.resize(buffer_len(&to)?, 0);
                        if remap.keeps_destination() {
                            self.destination
                                .read(memory, line, to.clone(), &mut output)?;
                        }
                        remap.apply(&input, &mut output);
                        self.destination.write(memory, line, to, &output, |_| ())?;
                    }
                }
                first = last;
            }
        }
        sink.invalidate(destination_cover);
        Ok(())
    }
}

/// A chunk's length as a buffer length (at most [`CHUNK_BYTES`] times the element size).
fn buffer_len(range: &Range<u64>) -> Result<usize, CopyError> {
    usize::try_from(range.end - range.start).map_err(|_| CopyError::TooLarge {
        bytes: u128::from(range.end - range.start),
        max: super::MAX_COPY_BYTES,
    })
}

/// One side of a launch, from its registers: pitch-linear when `pitch` is set, else
/// block-linear with its width and origin x counted in elements of `element` bytes.
fn side(
    regs: Registers<'_>,
    side: CopySide,
    pitch: bool,
    element: u64,
) -> Result<Surface, CopyError> {
    let (address, pitch_method) = match side {
        CopySide::Source => (
            regs.address(
                c::OFFSET_IN_UPPER,
                c::OFFSET_IN_UPPER_UPPER,
                c::OFFSET_IN_LOWER,
            ),
            c::PITCH_IN,
        ),
        CopySide::Destination => (
            regs.address(
                c::OFFSET_OUT_UPPER,
                c::OFFSET_OUT_UPPER_UPPER,
                c::OFFSET_OUT_LOWER,
            ),
            c::PITCH_OUT,
        ),
    };
    if pitch {
        let pitch = i64::from(regs.get(pitch_method).cast_signed());
        return Ok(Surface::pitch(side, address, pitch));
    }
    let desc = match side {
        CopySide::Source => {
            let block = regs.get(c::SET_SRC_BLOCK_SIZE);
            let origin = regs.get(c::SET_SRC_ORIGIN);
            BlockLinearDesc {
                block: BlockFields {
                    width: c::SET_SRC_BLOCK_SIZE_WIDTH.get(block),
                    height: c::SET_SRC_BLOCK_SIZE_HEIGHT.get(block),
                    depth: c::SET_SRC_BLOCK_SIZE_DEPTH.get(block),
                },
                width_b: u64::from(regs.get(c::SET_SRC_WIDTH)) * element,
                height: regs.get(c::SET_SRC_HEIGHT),
                depth: regs.get(c::SET_SRC_DEPTH),
                origin_x_b: u64::from(c::SET_SRC_ORIGIN_X.get(origin)) * element,
                origin_y: c::SET_SRC_ORIGIN_Y.get(origin),
            }
        }
        CopySide::Destination => {
            let block = regs.get(c::SET_DST_BLOCK_SIZE);
            let origin = regs.get(c::SET_DST_ORIGIN);
            BlockLinearDesc {
                block: BlockFields {
                    width: c::SET_DST_BLOCK_SIZE_WIDTH.get(block),
                    height: c::SET_DST_BLOCK_SIZE_HEIGHT.get(block),
                    depth: c::SET_DST_BLOCK_SIZE_DEPTH.get(block),
                },
                width_b: u64::from(regs.get(c::SET_DST_WIDTH)) * element,
                height: regs.get(c::SET_DST_HEIGHT),
                depth: regs.get(c::SET_DST_DEPTH),
                origin_x_b: u64::from(c::SET_DST_ORIGIN_X.get(origin)) * element,
                origin_y: c::SET_DST_ORIGIN_Y.get(origin),
            }
        }
    };
    Surface::block_linear(side, address, &desc)
}
