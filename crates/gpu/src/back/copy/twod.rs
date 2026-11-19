//! The 2D class, `902D` (`FERMI_TWOD_A`): surface-to-surface blits on the CPU.
//!
//! Which 2D class NVN2 binds is not in any reference: `902D` is a **guess** (Mesa has no Ampere
//! 2D class; the spec's verification item 10, a bind trace, confirms it).
//!
//! # Sources
//!
//! Paths are relative to the reference folder (`tools/refs-path.sh`), at the commits pinned in
//! `docs/references.md`.
//!
//! - Every method and field is read through [`crate::classes::cl902d`], generated from
//!   `mesa-nouveau/src/nouveau/headers/nvidia/classes/cl902d.h` @ 52238762 (Copyright (c) 2001-2001
//!   NVIDIA CORPORATION, MIT licence).
//! - The behaviour is ported from `Fermi2D::Blit` in
//!   `eden-nx2/src/video_core/engines/fermi_2d.{h,cpp}` @ 50fa86bb (Copyright 2026 Eden Emulator
//!   Project, GPL-3.0-or-later; Copyright 2018 yuzu Emulator Project, GPL-2.0-or-later) and
//!   `SoftwareBlitEngine::Blit` and `NearestNeighbor` in
//!   `eden-nx2/src/video_core/engines/sw_blitter/blitter.cpp` (Copyright 2022 yuzu Emulator
//!   Project, GPL-3.0-or-later).
//!
//! # What a blit does
//!
//! `PIXELS_FROM_MEMORY_SRC_Y0_INT` (the last of the blit's parameters, Eden-NX2's trigger)
//! starts it. The rectangles are Eden-NX2's `Config`: the destination is `DST_WIDTH` by
//! `DST_HEIGHT` pixels at `DST_X0`, `DST_Y0` (signed 32-bit); the source starts at the 32.32
//! fixed-point `SRC_X0`, `SRC_Y0` (each `_INT` above `_FRAC`), less half a step's integer part
//! when `SAMPLE_MODE_ORIGIN` is `CORNER` (`(du_dx >> 33) << 32`), and ends at the start plus
//! `DU_DX * DST_WIDTH` (`DV_DY * DST_HEIGHT`), both rounded down to whole pixels. Each side is
//! pitch-linear (`SET_*_PITCH`) or block-linear (`SET_*_BLOCK_SIZE`, `_WIDTH` in pixels,
//! `_HEIGHT`, `_DEPTH`) by `SET_*_MEMORY_LAYOUT`, with pixels of the size [`bytes_per_pixel`]
//! gives its format.
//!
//! - With the same format and the same extents on both sides, the pixels are copied
//!   unchanged, whatever the filter.
//! - With the same format and different extents, `SAMPLE_MODE_FILTER` `POINT` copies, for each
//!   destination pixel `(x, y)`, the source pixel `(x * step_x >> 32, y * step_y >> 32)`, where
//!   `step_x` is the source width over the destination width in 32.32 fixed point, rounded to
//!   nearest with halves away from zero (Eden-NX2's `llround(src / dst * 2^32)`), and the result
//!   clamped to the source's last pixel. `BILINEAR` is implemented for
//!   [`BILINEAR_FORMATS`] only ([`CopyError::BlitFilter`] for the others): each byte is an 8-bit
//!   unorm component, sampled at `x * (src_width - 1) / (dst_width - 1)` (0 for a one-pixel-wide
//!   destination) and likewise in y, from the four neighbours with 16-bit fixed-point weights,
//!   the result truncated as Eden-NX2's unorm conversion truncates (`static_cast<u32>(value *
//!   255)`).
//! - Different formats need a conversion: [`CopyError::BlitConversion`].
//!
//! The checks run in this order, before anything is written: the register file
//! ([`CopyError::Registers`]), the operation ([`CopyError::BlitOperation`]), clipping
//! ([`CopyError::BlitClip`]), the source's then the destination's format
//! ([`CopyError::BlitFormat`]), the conversion ([`CopyError::BlitConversion`]), the destination's
//! then the source's rectangle ([`CopyError::BlitRect`]); an empty destination then ends the
//! blit with nothing done; the filter ([`CopyError::BlitFilter`]), the size
//! ([`CopyError::TooLarge`]), the source then destination surfaces and regions, and the probe of
//! the source then the destination ([`CopyError::Fault`]).
//!
//! # Divergences from Eden-NX2
//!
//! Each is pinned by the test named, in `copy/tests/twod.rs`.
//!
//! - **Point sampling reads the pixel under the sample.** Eden-NX2's `NearestNeighbor` reads
//!   element `(src_y * src_width + src_x) >> 32` with both coordinates in 32.32, so a fractional
//!   source row adds `frac(y) * src_width` pixels; here the row and column are each rounded down
//!   first. The step is computed exactly in integers rather than in `f64`. Pinned by
//!   `divergence_point_sampling_reads_the_pixel_under_the_sample`.
//! - **Bilinear samples the neighbours.** Eden-NX2's `Bilinear` reads texel
//!   `(x * src_width + y) >> 32` (element 0 for any image under 2^32 pixels) in `f32`; here it
//!   samples the four neighbours in integers, for [`BILINEAR_FORMATS`] only. Pinned by
//!   `bilinear_blends_the_four_neighbours`.
//! - **Refused where Eden-NX2 goes on or converts:** an operation other than `SRCCOPY` and
//!   clipping (Eden-NX2's `UNIMPLEMENTED_IF_MSG`, which goes on), a format without a size here
//!   (`Y1_8X8`, `AY8`, undefined values), different formats (Eden-NX2 converts through
//!   `converter.cpp`; not ported), bilinear filtering of other formats, a negative or inverted
//!   rectangle or an empty source for a non-empty destination (Eden-NX2 casts to `u32`), a
//!   block-linear region outside its surface, and a blit past [`super::MAX_COPY_BYTES`]. Pinned
//!   by `refusals_are_exact`.
//! - **Eden-NX2's pitch adjustment changes nothing here.** `need_align_to_pitch` moves a pitch
//!   source's address by `src_x0` pixels when its region passes `pitch / bpp`; pitch regions are
//!   not bounded by their width here, so the same bytes are read either way.
//! - **Not read, as in Eden-NX2:** `SET_DST_LAYER` (the class has no source layer method;
//!   Eden-NX2 logs a non-zero layer of either surface and goes on),
//!   `SET_DST_COLOR_RENDER_TO_ZETA_SURFACE` (so a depth surface uses the colour GOB too; the
//!   spec's verification item 9), `SET_PIXELS_FROM_MEMORY_DIRECTION`, `SET_COMPRESSION`, the
//!   render enable and the sector promotion. Eden-NX2's delegation of large converted blits to
//!   the host GPU (`delegate_to_gpu`) is the host path's (G17 and G19). Pinned by
//!   `fields_that_are_not_read_change_nothing`.

use super::surface::{BlockFields, BlockLinearDesc, CopySide, ONE_GOB, Region, Surface};
use super::{CopyError, CopySink, PendingWrites, Registers, check_size, registers_needed};
use crate::classes::Method;
use crate::classes::cl902d as d;
use crate::memory::GpuMemory;

/// The register file must reach every method a blit reads.
pub const BLIT_REGISTERS_NEEDED: usize = registers_needed(&[
    d::SET_DST_FORMAT,
    d::SET_DST_MEMORY_LAYOUT,
    d::SET_DST_BLOCK_SIZE,
    d::SET_DST_DEPTH,
    d::SET_DST_PITCH,
    d::SET_DST_WIDTH,
    d::SET_DST_HEIGHT,
    d::SET_DST_OFFSET_UPPER,
    d::SET_DST_OFFSET_LOWER,
    d::SET_SRC_FORMAT,
    d::SET_SRC_MEMORY_LAYOUT,
    d::SET_SRC_BLOCK_SIZE,
    d::SET_SRC_DEPTH,
    d::SET_SRC_PITCH,
    d::SET_SRC_WIDTH,
    d::SET_SRC_HEIGHT,
    d::SET_SRC_OFFSET_UPPER,
    d::SET_SRC_OFFSET_LOWER,
    d::SET_CLIP_ENABLE,
    d::SET_OPERATION,
    d::SET_PIXELS_FROM_MEMORY_SAMPLE_MODE,
    d::SET_PIXELS_FROM_MEMORY_DST_X0,
    d::SET_PIXELS_FROM_MEMORY_DST_Y0,
    d::SET_PIXELS_FROM_MEMORY_DST_WIDTH,
    d::SET_PIXELS_FROM_MEMORY_DST_HEIGHT,
    d::SET_PIXELS_FROM_MEMORY_DU_DX_FRAC,
    d::SET_PIXELS_FROM_MEMORY_DU_DX_INT,
    d::SET_PIXELS_FROM_MEMORY_DV_DY_FRAC,
    d::SET_PIXELS_FROM_MEMORY_DV_DY_INT,
    d::SET_PIXELS_FROM_MEMORY_SRC_X0_FRAC,
    d::SET_PIXELS_FROM_MEMORY_SRC_X0_INT,
    d::SET_PIXELS_FROM_MEMORY_SRC_Y0_FRAC,
    d::PIXELS_FROM_MEMORY_SRC_Y0_INT,
]);

/// Each format value this module has a pixel size for, with that size in bytes: the sum of the
/// component widths the format's name gives (`A8R8G8B8` is four 8-bit components; `RF32_GF32` two
/// 32-bit floats; `BF10GF11RF11` 32 bits). The values are `SET_SRC_FORMAT_V`'s, which equal
/// `SET_DST_FORMAT_V`'s for every name both define
/// (`tests::twod::source_and_destination_formats_agree`). `Y1_8X8` (one bit per pixel) and the
/// source-only `AY8`, whose name gives no layout, are left out.
const FORMATS: [(u32, u32); 39] = [
    (d::SET_SRC_FORMAT_V_A8R8G8B8, 4),
    (d::SET_SRC_FORMAT_V_A8RL8GL8BL8, 4),
    (d::SET_SRC_FORMAT_V_A2R10G10B10, 4),
    (d::SET_SRC_FORMAT_V_A8B8G8R8, 4),
    (d::SET_SRC_FORMAT_V_A8BL8GL8RL8, 4),
    (d::SET_SRC_FORMAT_V_A2B10G10R10, 4),
    (d::SET_SRC_FORMAT_V_X8R8G8B8, 4),
    (d::SET_SRC_FORMAT_V_X8RL8GL8BL8, 4),
    (d::SET_SRC_FORMAT_V_X8B8G8R8, 4),
    (d::SET_SRC_FORMAT_V_X8BL8GL8RL8, 4),
    (d::SET_SRC_FORMAT_V_R5G6B5, 2),
    (d::SET_SRC_FORMAT_V_A1R5G5B5, 2),
    (d::SET_SRC_FORMAT_V_X1R5G5B5, 2),
    (d::SET_SRC_FORMAT_V_Y8, 1),
    (d::SET_SRC_FORMAT_V_Y16, 2),
    (d::SET_SRC_FORMAT_V_Y32, 4),
    (d::SET_SRC_FORMAT_V_Z1R5G5B5, 2),
    (d::SET_SRC_FORMAT_V_O1R5G5B5, 2),
    (d::SET_SRC_FORMAT_V_Z8R8G8B8, 4),
    (d::SET_SRC_FORMAT_V_O8R8G8B8, 4),
    (d::SET_SRC_FORMAT_V_RF16, 2),
    (d::SET_SRC_FORMAT_V_RF32, 4),
    (d::SET_SRC_FORMAT_V_RF32_GF32, 8),
    (d::SET_SRC_FORMAT_V_RF16_GF16_BF16_AF16, 8),
    (d::SET_SRC_FORMAT_V_RF16_GF16_BF16_X16, 8),
    (d::SET_SRC_FORMAT_V_RF32_GF32_BF32_AF32, 16),
    (d::SET_SRC_FORMAT_V_RF32_GF32_BF32_X32, 16),
    (d::SET_SRC_FORMAT_V_R16_G16_B16_A16, 8),
    (d::SET_SRC_FORMAT_V_RN16_GN16_BN16_AN16, 8),
    (d::SET_SRC_FORMAT_V_BF10GF11RF11, 4),
    (d::SET_SRC_FORMAT_V_AN8BN8GN8RN8, 4),
    (d::SET_SRC_FORMAT_V_RF16_GF16, 4),
    (d::SET_SRC_FORMAT_V_R16_G16, 4),
    (d::SET_SRC_FORMAT_V_RN16_GN16, 4),
    (d::SET_SRC_FORMAT_V_G8R8, 2),
    (d::SET_SRC_FORMAT_V_GN8RN8, 2),
    (d::SET_SRC_FORMAT_V_RN16, 2),
    (d::SET_SRC_FORMAT_V_RN8, 1),
    (d::SET_SRC_FORMAT_V_A8, 1),
];

/// The formats a scaled bilinear blit is implemented for: four 8-bit unorm components
/// (Eden-NX2's `A8R8G8B8_UNORMTraits` and `A8B8G8R8_UNORMTraits` in `converter.cpp`).
pub const BILINEAR_FORMATS: [u32; 2] = [d::SET_SRC_FORMAT_V_A8R8G8B8, d::SET_SRC_FORMAT_V_A8B8G8R8];

/// A pixel's size in bytes for a 2D format value, `None` for a value with no size here.
pub fn bytes_per_pixel(format: u32) -> Option<u32> {
    FORMATS
        .iter()
        .find(|(value, _)| *value == format)
        .map(|&(_, bytes)| bytes)
}

/// One in 16-bit fixed point, the bilinear weights' scale.
const WEIGHT_ONE: u64 = 1 << 16;

/// Handles one 2D method, called after the channel's `902D` register mirror (`registers`,
/// indexed by method number) has stored its value. Only `PIXELS_FROM_MEMORY_SRC_Y0_INT` does
/// anything: it first flushes `pending` to `sink`, then blits as the module docs describe. Every
/// other method returns `Ok(())` at once.
pub fn blit_method<M, S>(
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
    if method != d::PIXELS_FROM_MEMORY_SRC_Y0_INT.method() {
        return Ok(());
    }
    pending.flush(sink);
    let regs = Registers::new(registers, BLIT_REGISTERS_NEEDED)?;
    if let Some(blit) = Blit::new(regs)? {
        blit.run(memory, sink)?;
    }
    Ok(())
}

/// A rectangle in pixels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Rect {
    x: u32,
    y: u32,
    width: u32,
    height: u32,
}

/// A signed 64-bit 32.32 value from its `_INT` and `_FRAC` registers.
fn fixed(regs: Registers<'_>, int: Method, frac: Method) -> i64 {
    ((u64::from(regs.get(int)) << u32::BITS) | u64::from(regs.get(frac))).cast_signed()
}

/// A signed 32-bit register.
fn signed(regs: Registers<'_>, method: Method) -> i64 {
    i64::from(regs.get(method).cast_signed())
}

/// `start..end` as a rectangle span, refused when negative, inverted or not 32-bit.
fn span(start: i128, end: i128, side: CopySide) -> Result<(u32, u32), CopyError> {
    let refuse = CopyError::BlitRect { side };
    if end < start {
        return Err(refuse);
    }
    let start = u32::try_from(start).map_err(|_| refuse)?;
    let end = u32::try_from(end).map_err(|_| refuse)?;
    Ok((start, end - start))
}

/// The destination and source rectangles (Eden-NX2's `Config`).
fn rects(regs: Registers<'_>) -> Result<(Rect, Rect), CopyError> {
    let dst_x = i128::from(signed(regs, d::SET_PIXELS_FROM_MEMORY_DST_X0));
    let dst_y = i128::from(signed(regs, d::SET_PIXELS_FROM_MEMORY_DST_Y0));
    let dst_w = i128::from(signed(regs, d::SET_PIXELS_FROM_MEMORY_DST_WIDTH));
    let dst_h = i128::from(signed(regs, d::SET_PIXELS_FROM_MEMORY_DST_HEIGHT));
    let (x, width) = span(dst_x, dst_x + dst_w, CopySide::Destination)?;
    let (y, height) = span(dst_y, dst_y + dst_h, CopySide::Destination)?;
    let destination = Rect {
        x,
        y,
        width,
        height,
    };
    let du_dx = fixed(
        regs,
        d::SET_PIXELS_FROM_MEMORY_DU_DX_INT,
        d::SET_PIXELS_FROM_MEMORY_DU_DX_FRAC,
    );
    let dv_dy = fixed(
        regs,
        d::SET_PIXELS_FROM_MEMORY_DV_DY_INT,
        d::SET_PIXELS_FROM_MEMORY_DV_DY_FRAC,
    );
    let mut src_x = i128::from(fixed(
        regs,
        d::SET_PIXELS_FROM_MEMORY_SRC_X0_INT,
        d::SET_PIXELS_FROM_MEMORY_SRC_X0_FRAC,
    ));
    let mut src_y = i128::from(fixed(
        regs,
        d::PIXELS_FROM_MEMORY_SRC_Y0_INT,
        d::SET_PIXELS_FROM_MEMORY_SRC_Y0_FRAC,
    ));
    if d::SET_PIXELS_FROM_MEMORY_SAMPLE_MODE_ORIGIN.get(regs.get(d::SET_PIXELS_FROM_MEMORY_SAMPLE_MODE))
        == d::SET_PIXELS_FROM_MEMORY_SAMPLE_MODE_ORIGIN_CORNER
    {
        src_x -= i128::from((du_dx >> 33) << 32);
        src_y -= i128::from((dv_dy >> 33) << 32);
    }
    let (x, width) = span(
        src_x >> 32,
        (src_x + i128::from(du_dx) * dst_w) >> 32,
        CopySide::Source,
    )?;
    let (y, height) = span(
        src_y >> 32,
        (src_y + i128::from(dv_dy) * dst_h) >> 32,
        CopySide::Source,
    )?;
    Ok((
        destination,
        Rect {
            x,
            y,
            width,
            height,
        },
    ))
}

/// How destination pixels come from source pixels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Sampling {
    /// Same extents: copied unchanged.
    Copy,
    /// Point sampling with these 32.32 steps.
    Point { step_x: u64, step_y: u64 },
    /// Bilinear filtering of 8-bit components.
    Bilinear,
}

/// A checked blit.
struct Blit {
    bpp: u32,
    source_rect: Rect,
    destination_rect: Rect,
    sampling: Sampling,
    source: Surface,
    destination: Surface,
    source_cover: core::ops::Range<u64>,
    destination_cover: core::ops::Range<u64>,
}

/// `numerator * 2^32 / denominator`, rounded to nearest with halves up (Eden-NX2's
/// `llround(static_cast<f64>(src) / dst * (1ULL << 32))`, computed exactly). `denominator` is
/// not zero.
fn step(numerator: u32, denominator: u32) -> u64 {
    let numerator = u128::from(numerator) << 32;
    let denominator = u128::from(denominator);
    let rounded = (numerator + denominator / 2) / denominator;
    u64::try_from(rounded).unwrap_or(u64::MAX)
}

impl Blit {
    /// Checks a blit in the module docs' order; `None` when the destination is empty.
    fn new(regs: Registers<'_>) -> Result<Option<Self>, CopyError> {
        let operation = d::SET_OPERATION_V.get(regs.get(d::SET_OPERATION));
        if operation != d::SET_OPERATION_V_SRCCOPY {
            return Err(CopyError::BlitOperation(operation));
        }
        if d::SET_CLIP_ENABLE_V.get(regs.get(d::SET_CLIP_ENABLE)) == d::SET_CLIP_ENABLE_V_TRUE {
            return Err(CopyError::BlitClip);
        }
        let source_format = d::SET_SRC_FORMAT_V.get(regs.get(d::SET_SRC_FORMAT));
        let destination_format = d::SET_DST_FORMAT_V.get(regs.get(d::SET_DST_FORMAT));
        let bpp = bytes_per_pixel(source_format).ok_or(CopyError::BlitFormat {
            side: CopySide::Source,
            format: source_format,
        })?;
        bytes_per_pixel(destination_format).ok_or(CopyError::BlitFormat {
            side: CopySide::Destination,
            format: destination_format,
        })?;
        if source_format != destination_format {
            return Err(CopyError::BlitConversion {
                from: source_format,
                to: destination_format,
            });
        }
        let (destination_rect, source_rect) = rects(regs)?;
        if destination_rect.width == 0 || destination_rect.height == 0 {
            return Ok(None);
        }
        if source_rect.width == 0 || source_rect.height == 0 {
            return Err(CopyError::BlitRect {
                side: CopySide::Source,
            });
        }
        let same = (source_rect.width, source_rect.height)
            == (destination_rect.width, destination_rect.height);
        let bilinear = d::SET_PIXELS_FROM_MEMORY_SAMPLE_MODE_FILTER
            .get(regs.get(d::SET_PIXELS_FROM_MEMORY_SAMPLE_MODE))
            == d::SET_PIXELS_FROM_MEMORY_SAMPLE_MODE_FILTER_BILINEAR;
        let sampling = if same {
            Sampling::Copy
        } else if bilinear {
            if !BILINEAR_FORMATS.contains(&source_format) {
                return Err(CopyError::BlitFilter(source_format));
            }
            Sampling::Bilinear
        } else {
            Sampling::Point {
                step_x: step(source_rect.width, destination_rect.width),
                step_y: step(source_rect.height, destination_rect.height),
            }
        };
        let bytes = |rect: Rect| {
            u128::from(rect.width) * u128::from(rect.height) * u128::from(bpp)
        };
        check_size(bytes(source_rect).max(bytes(destination_rect)))?;
        let source = surface(regs, CopySide::Source, bpp, source_rect)?;
        let destination = surface(regs, CopySide::Destination, bpp, destination_rect)?;
        let region = |rect: Rect| Region {
            width: u64::from(rect.width) * u64::from(bpp),
            lines: rect.height,
        };
        // Neither region is empty, so both covers exist.
        let empty = CopyError::BlitRect {
            side: CopySide::Source,
        };
        let source_cover = source.cover(region(source_rect))?.ok_or(empty)?;
        let destination_cover = destination
            .cover(region(destination_rect))?
            .ok_or(empty)?;
        Ok(Some(Self {
            bpp,
            source_rect,
            destination_rect,
            sampling,
            source,
            destination,
            source_cover,
            destination_cover,
        }))
    }

    /// Probes both sides, then writes the destination row by row.
    fn run<M, S>(&self, memory: &M, sink: &mut S) -> Result<(), CopyError>
    where
        M: GpuMemory + ?Sized,
        S: CopySink + ?Sized,
    {
        let bpp = u64::from(self.bpp);
        let source_row = u64::from(self.source_rect.width) * bpp;
        let destination_row = u64::from(self.destination_rect.width) * bpp;
        self.source
            .probe(memory, 0..self.source_rect.height, 0..source_row)?;
        self.destination
            .probe(memory, 0..self.destination_rect.height, 0..destination_row)?;
        sink.make_current(self.source_cover.clone());
        sink.make_current(self.destination_cover.clone());
        let row_len = |len: u64| {
            usize::try_from(len).map_err(|_| CopyError::TooLarge {
                bytes: u128::from(len),
                max: super::MAX_COPY_BYTES,
            })
        };
        let mut rows = SourceRows::new(row_len(source_row)?);
        let mut out = vec![0u8; row_len(destination_row)?];
        for y in 0..self.destination_rect.height {
            match self.sampling {
                Sampling::Copy => {
                    let row = rows.get(&self.source, memory, y)?;
                    out.copy_from_slice(row);
                }
                Sampling::Point { step_x, step_y } => {
                    let sy = sample(y, step_y, self.source_rect.height);
                    let row = rows.get(&self.source, memory, sy)?;
                    let bpp = self.bpp as usize;
                    for (x, pixel) in (0u32..).zip(out.chunks_exact_mut(bpp)) {
                        let sx = sample(x, step_x, self.source_rect.width) as usize;
                        if let Some(texel) = row.get(sx * bpp..(sx + 1) * bpp) {
                            pixel.copy_from_slice(texel);
                        }
                    }
                }
                Sampling::Bilinear => self.bilinear_row(memory, &mut rows, y, &mut out)?,
            }
            self.destination
                .write(memory, y, 0..destination_row, &out, |_| ())?;
        }
        sink.invalidate(self.destination_cover.clone());
        Ok(())
    }

    /// One destination row of a bilinear blit.
    fn bilinear_row<M: GpuMemory + ?Sized>(
        &self,
        memory: &M,
        rows: &mut SourceRows,
        y: u32,
        out: &mut [u8],
    ) -> Result<(), CopyError> {
        let (y_low, y_high, wy) = position(y, self.source_rect.height, self.destination_rect.height);
        let low = rows.get(&self.source, memory, y_low)?.to_vec();
        let high = rows.get(&self.source, memory, y_high)?;
        let bpp = self.bpp as usize;
        for (x, pixel) in (0u32..).zip(out.chunks_exact_mut(bpp)) {
            let (x_low, x_high, wx) =
                position(x, self.source_rect.width, self.destination_rect.width);
            let at = |row: &[u8], column: u32, byte: usize| -> u64 {
                row.get(column as usize * bpp + byte)
                    .copied()
                    .map_or(0, u64::from)
            };
            for (byte, value) in pixel.iter_mut().enumerate() {
                let top = at(&low, x_low, byte) * (WEIGHT_ONE - wx) + at(&low, x_high, byte) * wx;
                let bottom =
                    at(high, x_low, byte) * (WEIGHT_ONE - wx) + at(high, x_high, byte) * wx;
                let blended = (top * (WEIGHT_ONE - wy) + bottom * wy) >> 32;
                *value = u8::try_from(blended).unwrap_or(u8::MAX);
            }
        }
        Ok(())
    }
}

/// The source index point sampling reads for destination index `index`: `index * step >> 32`,
/// clamped to the last of `count`.
fn sample(index: u32, step: u64, count: u32) -> u32 {
    let at = (u128::from(index) * u128::from(step)) >> 32;
    u32::try_from(at).unwrap_or(u32::MAX).min(count - 1)
}

/// The two source indices and the 16-bit weight of the upper one for destination index `index`
/// of `destination` samples over `source`: the sample sits at `index * (source - 1) /
/// (destination - 1)`, or 0 when `destination` is 1.
fn position(index: u32, source: u32, destination: u32) -> (u32, u32, u64) {
    if destination <= 1 {
        return (0, 0, 0);
    }
    let at = u128::from(index) * u128::from(source - 1) * u128::from(WEIGHT_ONE)
        / u128::from(destination - 1);
    let low = u32::try_from(at >> 16).unwrap_or(u32::MAX);
    let weight = u64::try_from(at & u128::from(WEIGHT_ONE - 1)).unwrap_or(0);
    let high = if weight == 0 { low } else { low + 1 };
    (low, high.min(source - 1), weight)
}

/// The last source rows read, so a row sampled by consecutive destination rows is read once.
struct SourceRows {
    rows: [(Option<u32>, Vec<u8>); 2],
    next: usize,
}

impl SourceRows {
    fn new(len: usize) -> Self {
        Self {
            rows: [(None, vec![0; len]), (None, vec![0; len])],
            next: 0,
        }
    }

    /// Source row `y` of the blit's region.
    fn get<M: GpuMemory + ?Sized>(
        &mut self,
        source: &Surface,
        memory: &M,
        y: u32,
    ) -> Result<&[u8], CopyError> {
        if let Some(slot) = self.rows.iter().position(|(row, _)| *row == Some(y)) {
            return Ok(&self.rows[slot].1);
        }
        let slot = self.next;
        self.next = 1 - self.next;
        let (row, buffer) = &mut self.rows[slot];
        *row = None;
        let len = buffer.len() as u64;
        source.read(memory, y, 0..len, buffer)?;
        *row = Some(y);
        Ok(buffer)
    }
}

/// One side of a blit: its surface, with the rectangle's origin applied.
fn surface(
    regs: Registers<'_>,
    side: CopySide,
    bpp: u32,
    rect: Rect,
) -> Result<Surface, CopyError> {
    struct Methods {
        layout: Method,
        layout_field: crate::field::Field,
        pitch_value: u32,
        block: Method,
        height_field: crate::field::Field,
        depth_field: crate::field::Field,
        depth: Method,
        pitch: Method,
        width: Method,
        height: Method,
        upper: Method,
        upper_field: crate::field::Field,
        lower: Method,
    }
    let m = match side {
        CopySide::Source => Methods {
            layout: d::SET_SRC_MEMORY_LAYOUT,
            layout_field: d::SET_SRC_MEMORY_LAYOUT_V,
            pitch_value: d::SET_SRC_MEMORY_LAYOUT_V_PITCH,
            block: d::SET_SRC_BLOCK_SIZE,
            height_field: d::SET_SRC_BLOCK_SIZE_HEIGHT,
            depth_field: d::SET_SRC_BLOCK_SIZE_DEPTH,
            depth: d::SET_SRC_DEPTH,
            pitch: d::SET_SRC_PITCH,
            width: d::SET_SRC_WIDTH,
            height: d::SET_SRC_HEIGHT,
            upper: d::SET_SRC_OFFSET_UPPER,
            upper_field: d::SET_SRC_OFFSET_UPPER_V,
            lower: d::SET_SRC_OFFSET_LOWER,
        },
        CopySide::Destination => Methods {
            layout: d::SET_DST_MEMORY_LAYOUT,
            layout_field: d::SET_DST_MEMORY_LAYOUT_V,
            pitch_value: d::SET_DST_MEMORY_LAYOUT_V_PITCH,
            block: d::SET_DST_BLOCK_SIZE,
            height_field: d::SET_DST_BLOCK_SIZE_HEIGHT,
            depth_field: d::SET_DST_BLOCK_SIZE_DEPTH,
            depth: d::SET_DST_DEPTH,
            pitch: d::SET_DST_PITCH,
            width: d::SET_DST_WIDTH,
            height: d::SET_DST_HEIGHT,
            upper: d::SET_DST_OFFSET_UPPER,
            upper_field: d::SET_DST_OFFSET_UPPER_V,
            lower: d::SET_DST_OFFSET_LOWER,
        },
    };
    let address = regs.address(m.upper, m.upper_field, m.lower);
    let bpp = u64::from(bpp);
    let origin_x_b = u64::from(rect.x) * bpp;
    if m.layout_field.get(regs.get(m.layout)) == m.pitch_value {
        let pitch = i64::from(regs.get(m.pitch));
        // The region's first byte; line `l` then follows at `l * pitch`.
        let start = i128::from(address)
            + i128::from(rect.y) * i128::from(pitch)
            + i128::from(origin_x_b);
        let start = u64::try_from(start).map_err(|_| CopyError::Extent { side })?;
        return Ok(Surface::pitch(side, start, pitch));
    }
    let block = regs.get(m.block);
    Surface::block_linear(
        side,
        address,
        &BlockLinearDesc {
            block: BlockFields {
                width: ONE_GOB,
                height: m.height_field.get(block),
                depth: m.depth_field.get(block),
            },
            width_b: u64::from(regs.get(m.width)) * bpp,
            height: regs.get(m.height),
            depth: regs.get(m.depth),
            origin_x_b,
            origin_y: rect.y,
        },
    )
}
