//! `902D` blits: unscaled copies between layouts against the layout crate, point and bilinear
//! scaling against a model, refusals and divergences.

use super::super::surface::CopySide;
use super::super::{
    BILINEAR_FORMATS, BLIT_REGISTERS_NEEDED, CopyError, MAX_COPY_BYTES, PendingWrites, blit_method,
    bytes_per_pixel,
};
use super::*;
use crate::classes::cl902d as d;

const SRC: u64 = GPU_BASE + 0x1_0000;
const DST: u64 = GPU_BASE + 0x9_0000;
const RGBA: u32 = d::SET_SRC_FORMAT_V_A8R8G8B8;

/// One side of a test blit; block-linear widths are in pixels here.
#[derive(Clone, Copy, Debug)]
enum Side {
    Pitch {
        address: u64,
        pitch: u32,
    },
    Bl {
        address: u64,
        width_px: u32,
        height: u32,
        bh: u8,
    },
}

impl Side {
    fn bl(self, bpp: u32) -> Option<(u64, Bl)> {
        match self {
            Side::Bl {
                address,
                width_px,
                height,
                bh,
            } => Some((
                address,
                Bl {
                    width: width_px * bpp,
                    height,
                    depth: 1,
                    bh,
                    bd: 0,
                },
            )),
            Side::Pitch { .. } => None,
        }
    }
}

/// A rectangle in pixels.
#[derive(Clone, Copy, Debug)]
struct R {
    x: u32,
    y: u32,
    w: u32,
    h: u32,
}

/// A blit's settings.
#[derive(Clone, Copy, Debug)]
struct Blit {
    format: u32,
    source: Side,
    destination: Side,
    from: R,
    to: R,
    bilinear: bool,
}

/// `numerator / denominator` in 32.32, rounded up, so the source end comes out exact.
fn ratio(numerator: u32, denominator: u32) -> i64 {
    (((u64::from(numerator) << 32) + u64::from(denominator) - 1) / u64::from(denominator))
        .cast_signed()
}

fn set_fixed(regs: &mut Regs, int: Method, frac: Method, value: i64) {
    regs.set(int, hi(value.cast_unsigned()))
        .set(frac, lo(value.cast_unsigned()));
}

fn regs_for(blit: &Blit) -> Regs {
    let mut regs = Regs::new(BLIT_REGISTERS_NEEDED);
    regs.set(d::SET_OPERATION, d::SET_OPERATION_V_SRCCOPY)
        .set(d::SET_SRC_FORMAT, blit.format)
        .set(d::SET_DST_FORMAT, blit.format)
        .set(d::SET_SRC_DEPTH, 1)
        .set(d::SET_DST_DEPTH, 1)
        .set(d::SET_PIXELS_FROM_MEMORY_DST_X0, blit.to.x)
        .set(d::SET_PIXELS_FROM_MEMORY_DST_Y0, blit.to.y)
        .set(d::SET_PIXELS_FROM_MEMORY_DST_WIDTH, blit.to.w)
        .set(d::SET_PIXELS_FROM_MEMORY_DST_HEIGHT, blit.to.h)
        .field(
            d::SET_PIXELS_FROM_MEMORY_SAMPLE_MODE,
            d::SET_PIXELS_FROM_MEMORY_SAMPLE_MODE_FILTER,
            u32::from(blit.bilinear),
        );
    if blit.to.w > 0 && blit.to.h > 0 {
        let du = ratio(blit.from.w, blit.to.w);
        let dv = ratio(blit.from.h, blit.to.h);
        set_fixed(
            &mut regs,
            d::SET_PIXELS_FROM_MEMORY_DU_DX_INT,
            d::SET_PIXELS_FROM_MEMORY_DU_DX_FRAC,
            du,
        );
        set_fixed(
            &mut regs,
            d::SET_PIXELS_FROM_MEMORY_DV_DY_INT,
            d::SET_PIXELS_FROM_MEMORY_DV_DY_FRAC,
            dv,
        );
    }
    set_fixed(
        &mut regs,
        d::SET_PIXELS_FROM_MEMORY_SRC_X0_INT,
        d::SET_PIXELS_FROM_MEMORY_SRC_X0_FRAC,
        i64::from(blit.from.x) << 32,
    );
    set_fixed(
        &mut regs,
        d::PIXELS_FROM_MEMORY_SRC_Y0_INT,
        d::SET_PIXELS_FROM_MEMORY_SRC_Y0_FRAC,
        i64::from(blit.from.y) << 32,
    );
    for (side, source) in [(blit.source, true), (blit.destination, false)] {
        let (layout, block, pitch, width, height, upper, lower) = if source {
            (
                d::SET_SRC_MEMORY_LAYOUT,
                d::SET_SRC_BLOCK_SIZE,
                d::SET_SRC_PITCH,
                d::SET_SRC_WIDTH,
                d::SET_SRC_HEIGHT,
                d::SET_SRC_OFFSET_UPPER,
                d::SET_SRC_OFFSET_LOWER,
            )
        } else {
            (
                d::SET_DST_MEMORY_LAYOUT,
                d::SET_DST_BLOCK_SIZE,
                d::SET_DST_PITCH,
                d::SET_DST_WIDTH,
                d::SET_DST_HEIGHT,
                d::SET_DST_OFFSET_UPPER,
                d::SET_DST_OFFSET_LOWER,
            )
        };
        let address = match side {
            Side::Pitch { address, pitch: p } => {
                regs.set(layout, d::SET_SRC_MEMORY_LAYOUT_V_PITCH)
                    .set(pitch, p);
                address
            }
            Side::Bl {
                address,
                width_px,
                height: h,
                bh,
            } => {
                regs.set(layout, d::SET_SRC_MEMORY_LAYOUT_V_BLOCKLINEAR)
                    .field(block, d::SET_SRC_BLOCK_SIZE_HEIGHT, bh.into())
                    .set(width, width_px)
                    .set(height, h);
                address
            }
        };
        // The 2D class keeps 8 upper bits: use an address that fits them.
        regs.set(upper, hi(address) & 0xff).set(lower, lo(address));
    }
    regs
}

/// `address` moved from the shared test base to this file's low base.
fn low(address: u64) -> u64 {
    address - GPU_BASE + LOW_BASE
}

fn side_low(side: Side) -> Side {
    match side {
        Side::Pitch { address, pitch } => Side::Pitch {
            address: low(address),
            pitch,
        },
        Side::Bl {
            address,
            width_px,
            height,
            bh,
        } => Side::Bl {
            address: low(address),
            width_px,
            height,
            bh,
        },
    }
}

fn run(mem: &Memory, blit: &Blit, sink: &mut Recorder) -> Result<(), CopyError> {
    let blit = Blit {
        source: side_low(blit.source),
        destination: side_low(blit.destination),
        ..*blit
    };
    blit_method(
        d::PIXELS_FROM_MEMORY_SRC_Y0_INT.method(),
        &regs_for(&blit).0,
        mem,
        &mut PendingWrites::new(),
        sink,
    )
}

/// The pixels of `rect` on `side`, as rows of `rect.w * bpp` bytes.
fn pixels(mem: &Memory, side: Side, rect: R, bpp: u32) -> Vec<u8> {
    match side_low(side) {
        Side::Pitch { address, pitch } => (0..rect.h)
            .flat_map(|y| {
                let at =
                    address + u64::from(rect.y + y) * u64::from(pitch) + u64::from(rect.x * bpp);
                read(mem, at, u64::from(rect.w * bpp))
            })
            .collect(),
        side @ Side::Bl { address, .. } => {
            let (_, bl) = side.bl(bpp).unwrap();
            bl.untile(
                &read(mem, address, bl.size()),
                (rect.x * bpp, rect.y),
                rect.w * bpp,
                rect.h,
            )
        }
    }
}

/// Memory after writing `rows` to `rect` of `side`.
fn expect(before: &[u8], side: Side, rect: R, bpp: u32, rows: &[u8]) -> Vec<u8> {
    let mut after = before.to_vec();
    match side {
        Side::Pitch { address, pitch } => {
            let row = (rect.w * bpp) as usize;
            for y in 0..rect.h as usize {
                let at = us(address - GPU_BASE)
                    + (rect.y as usize + y) * pitch as usize
                    + (rect.x * bpp) as usize;
                after[at..at + row].copy_from_slice(&rows[y * row..(y + 1) * row]);
            }
        }
        Side::Bl { address, .. } => {
            let (_, bl) = side.bl(bpp).unwrap();
            let start = us(address - GPU_BASE);
            bl.tile(
                &mut after[start..start + us(bl.size())],
                rows,
                (rect.w * bpp) as usize,
                (rect.x * bpp, rect.y),
                rect.w * bpp,
                rect.h,
            );
        }
    }
    after
}

/// Runs an unscaled blit and checks memory against the layout crate's oracle.
fn check_copy(blit: Blit) {
    let mem = memory();
    let bpp = bytes_per_pixel(blit.format).unwrap();
    let rows = pixels(&mem, blit.source, blit.from, bpp);
    let expected = expect(
        &read(&mem, LOW_BASE, PAGES * PAGE_SIZE),
        blit.destination,
        blit.to,
        bpp,
        &rows,
    );
    run(&mem, &blit, &mut Recorder::default()).unwrap();
    assert!(
        read(&mem, LOW_BASE, PAGES * PAGE_SIZE) == expected,
        "{blit:?}"
    );
}

#[test]
fn an_unscaled_blit_copies_between_every_layout_pair() {
    let pitch_src = Side::Pitch {
        address: SRC + 4,
        pitch: 0x200,
    };
    let pitch_dst = Side::Pitch {
        address: DST + 8,
        pitch: 0x180,
    };
    let bl_src = Side::Bl {
        address: SRC,
        width_px: 100,
        height: 60,
        bh: 2,
    };
    let bl_dst = Side::Bl {
        address: DST,
        width_px: 70,
        height: 50,
        bh: 1,
    };
    let from = R {
        x: 3,
        y: 5,
        w: 40,
        h: 30,
    };
    let to = R {
        x: 20,
        y: 7,
        w: 40,
        h: 30,
    };
    for (source, destination) in [
        (pitch_src, pitch_dst),
        (pitch_src, bl_dst),
        (bl_src, pitch_dst),
        (bl_src, bl_dst),
    ] {
        for bilinear in [false, true] {
            check_copy(Blit {
                format: RGBA,
                source,
                destination,
                from,
                to,
                bilinear,
            });
        }
    }
    // A 16-byte format and a 1-byte one.
    for format in [
        d::SET_SRC_FORMAT_V_RF32_GF32_BF32_AF32,
        d::SET_SRC_FORMAT_V_A8,
    ] {
        check_copy(Blit {
            format,
            source: bl_src,
            destination: bl_dst,
            from: R { w: 10, ..from },
            to: R { w: 10, ..to },
            bilinear: false,
        });
    }
}

/// The model of point sampling: destination `(x, y)` takes source `(x * sw / dw, y * sh / dh)`
/// rounded down, through the exact 32.32 step.
fn point_model(src: &[u8], from: R, to: R, bpp: usize) -> Vec<u8> {
    let step = |s: u32, d: u32| ((u128::from(s) << 32) + u128::from(d) / 2) / u128::from(d);
    let (sx, sy) = (step(from.w, to.w), step(from.h, to.h));
    let mut out = Vec::new();
    for y in 0..to.h {
        let row = ((u128::from(y) * sy) >> 32).min(u128::from(from.h - 1)) as usize;
        for x in 0..to.w {
            let column = ((u128::from(x) * sx) >> 32).min(u128::from(from.w - 1)) as usize;
            let at = (row * from.w as usize + column) * bpp;
            out.extend_from_slice(&src[at..at + bpp]);
        }
    }
    out
}

fn check_point(from: R, to: R) {
    let mem = memory();
    let source = Side::Bl {
        address: SRC,
        width_px: 64,
        height: 64,
        bh: 1,
    };
    let destination = Side::Pitch {
        address: DST,
        pitch: 0x400,
    };
    let blit = Blit {
        format: RGBA,
        source,
        destination,
        from,
        to,
        bilinear: false,
    };
    let src = pixels(&mem, source, from, 4);
    let expected = expect(
        &read(&mem, LOW_BASE, PAGES * PAGE_SIZE),
        destination,
        to,
        4,
        &point_model(&src, from, to, 4),
    );
    run(&mem, &blit, &mut Recorder::default()).unwrap();
    assert!(
        read(&mem, LOW_BASE, PAGES * PAGE_SIZE) == expected,
        "{from:?} -> {to:?}"
    );
}

#[test]
fn point_sampling_scales_down_and_up() {
    // 4 to 6: destination 3 sits exactly on source 2, which a step rounded down would miss.
    check_point(
        R {
            x: 1,
            y: 1,
            w: 4,
            h: 4,
        },
        R {
            x: 0,
            y: 0,
            w: 6,
            h: 6,
        },
    );
    check_point(
        R {
            x: 2,
            y: 4,
            w: 16,
            h: 12,
        },
        R {
            x: 1,
            y: 1,
            w: 8,
            h: 6,
        },
    );
    check_point(
        R {
            x: 0,
            y: 0,
            w: 5,
            h: 3,
        },
        R {
            x: 9,
            y: 2,
            w: 15,
            h: 7,
        },
    );
    check_point(
        R {
            x: 7,
            y: 1,
            w: 13,
            h: 9,
        },
        R {
            x: 0,
            y: 0,
            w: 5,
            h: 4,
        },
    );
}

#[test]
fn divergence_point_sampling_reads_the_pixel_under_the_sample() {
    // A 4 by 2 source doubled to 8 by 4: destination row 1 samples source row 0.5, so row 0.
    // Eden-NX2's index `(src_y * W + src_x) >> 32` reads row 0, column 2 (0.5 * 4) for x = 0.
    let mem = memory();
    let source = Side::Pitch {
        address: SRC,
        pitch: 16,
    };
    let destination = Side::Pitch {
        address: DST,
        pitch: 32,
    };
    let from = R {
        x: 0,
        y: 0,
        w: 4,
        h: 2,
    };
    let to = R {
        x: 0,
        y: 0,
        w: 8,
        h: 4,
    };
    let src = pixels(&mem, source, from, 4);
    run(
        &mem,
        &Blit {
            format: RGBA,
            source,
            destination,
            from,
            to,
            bilinear: false,
        },
        &mut Recorder::default(),
    )
    .unwrap();
    let row1 = read(&mem, low(DST) + 32, 4);
    assert_eq!(row1, src[0..4], "row 1, x 0 is source (0, 0)");
    assert_ne!(row1, src[8..12], "not Eden-NX2's source (2, 0)");
}

#[test]
fn bilinear_blends_the_four_neighbours() {
    let mem = memory();
    let source = Side::Pitch {
        address: SRC,
        pitch: 8,
    };
    let destination = Side::Pitch {
        address: DST,
        pitch: 12,
    };
    // 2 by 2 source pixels with chosen bytes.
    let src: [[u8; 4]; 4] = [
        [10, 0, 255, 1],
        [21, 100, 0, 2],
        [30, 50, 255, 3],
        [41, 200, 1, 5],
    ];
    mem.write(low(SRC), &[src[0], src[1]].concat()).unwrap();
    mem.write(low(SRC) + 8, &[src[2], src[3]].concat()).unwrap();
    let from = R {
        x: 0,
        y: 0,
        w: 2,
        h: 2,
    };
    let to = R {
        x: 0,
        y: 0,
        w: 3,
        h: 3,
    };
    run(
        &mem,
        &Blit {
            format: RGBA,
            source,
            destination,
            from,
            to,
            bilinear: true,
        },
        &mut Recorder::default(),
    )
    .unwrap();
    let pixel = |x: u64, y: u64| read(&mem, low(DST) + y * 12 + x * 4, 4);
    // Corners are the source pixels.
    assert_eq!(pixel(0, 0), src[0]);
    assert_eq!(pixel(2, 0), src[1]);
    assert_eq!(pixel(0, 2), src[2]);
    assert_eq!(pixel(2, 2), src[3]);
    // Edge midpoints average two, the centre averages four, rounding down.
    let avg2 = |a: [u8; 4], b: [u8; 4]| -> Vec<u8> {
        (0..4)
            .map(|i| u8::try_from(u32::midpoint(u32::from(a[i]), u32::from(b[i]))).unwrap())
            .collect()
    };
    assert_eq!(pixel(1, 0), avg2(src[0], src[1]));
    assert_eq!(pixel(0, 1), avg2(src[0], src[2]));
    let centre: Vec<u8> = (0..4)
        .map(|i| u8::try_from(src.iter().map(|p| u32::from(p[i])).sum::<u32>() / 4).unwrap())
        .collect();
    assert_eq!(pixel(1, 1), centre);
    assert_eq!(centre, vec![25, 87, 127, 2]);
}

#[test]
fn a_corner_origin_starts_half_a_step_earlier() {
    // Halving with du_dx 2.0: CORNER moves the source start back by (2.0 >> 1) = 1 pixel.
    let mem = memory();
    let source = Side::Pitch {
        address: SRC,
        pitch: 64,
    };
    let destination = Side::Pitch {
        address: DST,
        pitch: 64,
    };
    let blit = Blit {
        format: d::SET_SRC_FORMAT_V_A8,
        source,
        destination,
        from: R {
            x: 1,
            y: 1,
            w: 8,
            h: 2,
        },
        to: R {
            x: 0,
            y: 0,
            w: 4,
            h: 1,
        },
        bilinear: false,
    };
    let mut regs = regs_for(&Blit {
        source: side_low(source),
        destination: side_low(destination),
        ..blit
    });
    regs.field(
        d::SET_PIXELS_FROM_MEMORY_SAMPLE_MODE,
        d::SET_PIXELS_FROM_MEMORY_SAMPLE_MODE_ORIGIN,
        d::SET_PIXELS_FROM_MEMORY_SAMPLE_MODE_ORIGIN_CORNER,
    );
    blit_method(
        d::PIXELS_FROM_MEMORY_SRC_Y0_INT.method(),
        &regs.0,
        &mem,
        &mut PendingWrites::new(),
        &mut Recorder::default(),
    )
    .unwrap();
    // The source rectangle is (0, 0) 8 by 2: columns 0, 2, 4, 6 of row 0.
    let row = read(&mem, low(SRC), 8);
    assert_eq!(
        read(&mem, low(DST), 4),
        vec![row[0], row[2], row[4], row[6]]
    );
}

#[test]
fn the_sink_sees_pending_then_current_then_invalidate() {
    let mem = memory();
    let blit = Blit {
        format: RGBA,
        source: Side::Pitch {
            address: SRC,
            pitch: 0x100,
        },
        destination: Side::Bl {
            address: DST,
            width_px: 32,
            height: 16,
            bh: 1,
        },
        from: R {
            x: 1,
            y: 2,
            w: 4,
            h: 3,
        },
        to: R {
            x: 2,
            y: 1,
            w: 4,
            h: 3,
        },
        bilinear: false,
    };
    let regs = regs_for(&Blit {
        source: side_low(blit.source),
        destination: side_low(blit.destination),
        ..blit
    });
    let mut pending = PendingWrites::new();
    pending.add(5..9);
    let mut sink = Recorder::default();
    blit_method(
        d::PIXELS_FROM_MEMORY_SRC_Y0_INT.method(),
        &regs.0,
        &mem,
        &mut pending,
        &mut sink,
    )
    .unwrap();
    let src_start = low(SRC) + 2 * 0x100 + 4;
    assert_eq!(
        sink.events,
        vec![
            Event::Invalidate(5..9),
            Event::MakeCurrent(src_start..src_start + 2 * 0x100 + 16),
            Event::MakeCurrent(low(DST)..low(DST) + 128 * 16),
            Event::Invalidate(low(DST)..low(DST) + 128 * 16),
        ]
    );
}

/// The blit the refusal tests change one register of.
fn blit_base() -> Blit {
    Blit {
        format: RGBA,
        source: Side::Pitch {
            address: SRC,
            pitch: 0x100,
        },
        destination: Side::Pitch {
            address: DST,
            pitch: 0x100,
        },
        from: R {
            x: 0,
            y: 0,
            w: 8,
            h: 8,
        },
        to: R {
            x: 0,
            y: 0,
            w: 4,
            h: 4,
        },
        bilinear: false,
    }
}

/// Runs `blit` changed by `f`; a refused blit must leave memory and the sink untouched.
fn run_with(blit: &Blit, f: &dyn Fn(&mut Regs)) -> Result<(), CopyError> {
    let mem = memory();
    let mut regs = regs_for(&Blit {
        source: side_low(blit.source),
        destination: side_low(blit.destination),
        ..*blit
    });
    f(&mut regs);
    let before = read(&mem, LOW_BASE, PAGES * PAGE_SIZE);
    let mut sink = Recorder::default();
    let result = blit_method(
        d::PIXELS_FROM_MEMORY_SRC_Y0_INT.method(),
        &regs.0,
        &mem,
        &mut PendingWrites::new(),
        &mut sink,
    );
    if result.is_err() {
        assert!(read(&mem, LOW_BASE, PAGES * PAGE_SIZE) == before);
        assert_eq!(sink.events, vec![]);
    }
    result
}

#[test]
fn refusals_are_exact() {
    let mem = memory();
    let base = blit_base();
    assert_eq!(
        blit_method(
            d::PIXELS_FROM_MEMORY_SRC_Y0_INT.method(),
            &vec![0; BLIT_REGISTERS_NEEDED - 1],
            &mem,
            &mut PendingWrites::new(),
            &mut Recorder::default()
        ),
        Err(CopyError::Registers {
            len: BLIT_REGISTERS_NEEDED - 1,
            needed: BLIT_REGISTERS_NEEDED
        })
    );
    // Operation before clip before formats.
    assert_eq!(
        run_with(&base, &|r| {
            r.set(d::SET_OPERATION, d::SET_OPERATION_V_BLEND_AND)
                .set(d::SET_CLIP_ENABLE, d::SET_CLIP_ENABLE_V_TRUE)
                .set(d::SET_SRC_FORMAT, 0);
        }),
        Err(CopyError::BlitOperation(d::SET_OPERATION_V_BLEND_AND))
    );
    assert_eq!(
        run_with(&base, &|r| {
            r.set(d::SET_CLIP_ENABLE, d::SET_CLIP_ENABLE_V_TRUE)
                .set(d::SET_SRC_FORMAT, 0);
        }),
        Err(CopyError::BlitClip)
    );
    // Source format before destination format before the conversion.
    assert_eq!(
        run_with(&base, &|r| {
            r.set(d::SET_SRC_FORMAT, d::SET_SRC_FORMAT_V_Y1_8X8)
                .set(d::SET_DST_FORMAT, d::SET_SRC_FORMAT_V_AY8);
        }),
        Err(CopyError::BlitFormat {
            side: CopySide::Source,
            format: d::SET_SRC_FORMAT_V_Y1_8X8
        })
    );
    assert_eq!(
        run_with(&base, &|r| {
            r.set(d::SET_DST_FORMAT, d::SET_SRC_FORMAT_V_AY8);
        }),
        Err(CopyError::BlitFormat {
            side: CopySide::Destination,
            format: d::SET_SRC_FORMAT_V_AY8
        })
    );
    assert_eq!(
        run_with(&base, &|r| {
            r.set(d::SET_DST_FORMAT, d::SET_DST_FORMAT_V_A8B8G8R8);
        }),
        Err(CopyError::BlitConversion {
            from: RGBA,
            to: d::SET_DST_FORMAT_V_A8B8G8R8
        })
    );
}

#[test]
fn rectangle_filter_and_size_refusals_are_exact() {
    let base = blit_base();
    // Destination rectangle before source rectangle.
    assert_eq!(
        run_with(&base, &|r| {
            r.set(d::SET_PIXELS_FROM_MEMORY_DST_X0, (-1i32).cast_unsigned())
                .set(
                    d::SET_PIXELS_FROM_MEMORY_SRC_X0_INT,
                    (-1i32).cast_unsigned(),
                );
        }),
        Err(CopyError::BlitRect {
            side: CopySide::Destination
        })
    );
    assert_eq!(
        run_with(&base, &|r| {
            r.set(d::SET_PIXELS_FROM_MEMORY_DST_WIDTH, (-1i32).cast_unsigned());
        }),
        Err(CopyError::BlitRect {
            side: CopySide::Destination
        })
    );
    assert_eq!(
        run_with(&base, &|r| {
            r.set(
                d::SET_PIXELS_FROM_MEMORY_SRC_X0_INT,
                (-1i32).cast_unsigned(),
            );
        }),
        Err(CopyError::BlitRect {
            side: CopySide::Source
        })
    );
    // A non-empty destination from an empty source.
    assert_eq!(
        run_with(&base, &|r| {
            r.set(d::SET_PIXELS_FROM_MEMORY_DU_DX_INT, 0)
                .set(d::SET_PIXELS_FROM_MEMORY_DU_DX_FRAC, 0);
        }),
        Err(CopyError::BlitRect {
            side: CopySide::Source
        })
    );
    // The empty source is reported before the filter.
    let rf32_bilinear = Blit {
        format: d::SET_SRC_FORMAT_V_RF32,
        bilinear: true,
        ..base
    };
    assert_eq!(
        run_with(&rf32_bilinear, &|r| {
            r.set(d::SET_PIXELS_FROM_MEMORY_DU_DX_INT, 0)
                .set(d::SET_PIXELS_FROM_MEMORY_DU_DX_FRAC, 0);
        }),
        Err(CopyError::BlitRect {
            side: CopySide::Source
        })
    );
    // An empty destination does nothing, even from an empty source.
    assert_eq!(
        run_with(&base, &|r| {
            r.set(d::SET_PIXELS_FROM_MEMORY_DST_WIDTH, 0)
                .set(d::SET_PIXELS_FROM_MEMORY_DU_DX_INT, 0);
        }),
        Ok(())
    );
    // Bilinear scaling of a format it is not implemented for; unscaled bilinear copies.
    let rf32 = Blit {
        format: d::SET_SRC_FORMAT_V_RF32,
        bilinear: true,
        ..base
    };
    assert_eq!(
        run_with(&rf32, &|_| ()),
        Err(CopyError::BlitFilter(d::SET_SRC_FORMAT_V_RF32))
    );
}

#[test]
fn the_size_bound_is_checked_at_the_edge() {
    let base = blit_base();
    // The size: 4096 by 4096 at 16 bytes is MAX_COPY_BYTES; one more row is refused.
    // A source pitch of 1 MiB puts line 2 past the mapping, so the probe stops there.
    let huge = Blit {
        format: d::SET_SRC_FORMAT_V_RF32_GF32_BF32_AF32,
        source: Side::Pitch {
            address: SRC,
            pitch: 0x10_0000,
        },
        from: R {
            x: 0,
            y: 0,
            w: 4096,
            h: 4097,
        },
        to: R {
            x: 0,
            y: 0,
            w: 4096,
            h: 4097,
        },
        ..base
    };
    assert_eq!(
        run_with(&huge, &|_| ()),
        Err(CopyError::TooLarge {
            bytes: u128::from(MAX_COPY_BYTES) + 4096 * 16,
            max: MAX_COPY_BYTES
        })
    );
    // At the bound itself the blit gets past the size check, to the probe.
    let edge = Blit {
        from: R {
            h: 4096,
            ..huge.from
        },
        to: R { h: 4096, ..huge.to },
        ..huge
    };
    assert_eq!(
        run_with(&edge, &|_| ()),
        Err(CopyError::Fault {
            side: CopySide::Source,
            fault: crate::memory::GpuFault::Unmapped {
                va: low(SRC) + 2 * 0x10_0000
            },
        })
    );
}

#[test]
fn fields_that_are_not_read_change_nothing() {
    let blit = Blit {
        format: RGBA,
        source: Side::Bl {
            address: SRC,
            width_px: 40,
            height: 20,
            bh: 1,
        },
        destination: Side::Pitch {
            address: DST,
            pitch: 0x100,
        },
        from: R {
            x: 1,
            y: 1,
            w: 10,
            h: 6,
        },
        to: R {
            x: 2,
            y: 3,
            w: 10,
            h: 6,
        },
        bilinear: false,
    };
    let plain = memory();
    run(&plain, &blit, &mut Recorder::default()).unwrap();
    let noisy = memory();
    let mut regs = regs_for(&Blit {
        source: side_low(blit.source),
        destination: side_low(blit.destination),
        ..blit
    });
    regs.set(d::SET_DST_LAYER, 3)
        .set(d::SET_DST_COLOR_RENDER_TO_ZETA_SURFACE, 1)
        .set(d::SET_PIXELS_FROM_MEMORY_DIRECTION, 0x22)
        .set(d::SET_COMPRESSION, 1)
        .set(d::SET_RENDER_ENABLE_C, 0)
        .set(d::SET_PIXELS_FROM_MEMORY_SECTOR_PROMOTION, 3);
    blit_method(
        d::PIXELS_FROM_MEMORY_SRC_Y0_INT.method(),
        &regs.0,
        &noisy,
        &mut PendingWrites::new(),
        &mut Recorder::default(),
    )
    .unwrap();
    assert!(read(&plain, LOW_BASE, PAGES * PAGE_SIZE) == read(&noisy, LOW_BASE, PAGES * PAGE_SIZE));
    // Other methods do nothing.
    assert_eq!(
        blit_method(
            d::SET_PIXELS_FROM_MEMORY_SRC_Y0_FRAC.method(),
            &regs.0,
            &noisy,
            &mut PendingWrites::new(),
            &mut Recorder::default()
        ),
        Ok(())
    );
}

/// Each format's pixel size, written out from its name, with its source and destination
/// values (`cl902d.h`).
const FORMAT_TABLE: [(u32, u32, u32); 39] = [
    (
        d::SET_SRC_FORMAT_V_A8R8G8B8,
        d::SET_DST_FORMAT_V_A8R8G8B8,
        4,
    ),
    (
        d::SET_SRC_FORMAT_V_A8RL8GL8BL8,
        d::SET_DST_FORMAT_V_A8RL8GL8BL8,
        4,
    ),
    (
        d::SET_SRC_FORMAT_V_A2R10G10B10,
        d::SET_DST_FORMAT_V_A2R10G10B10,
        4,
    ),
    (
        d::SET_SRC_FORMAT_V_A8B8G8R8,
        d::SET_DST_FORMAT_V_A8B8G8R8,
        4,
    ),
    (
        d::SET_SRC_FORMAT_V_A8BL8GL8RL8,
        d::SET_DST_FORMAT_V_A8BL8GL8RL8,
        4,
    ),
    (
        d::SET_SRC_FORMAT_V_A2B10G10R10,
        d::SET_DST_FORMAT_V_A2B10G10R10,
        4,
    ),
    (
        d::SET_SRC_FORMAT_V_X8R8G8B8,
        d::SET_DST_FORMAT_V_X8R8G8B8,
        4,
    ),
    (
        d::SET_SRC_FORMAT_V_X8RL8GL8BL8,
        d::SET_DST_FORMAT_V_X8RL8GL8BL8,
        4,
    ),
    (
        d::SET_SRC_FORMAT_V_X8B8G8R8,
        d::SET_DST_FORMAT_V_X8B8G8R8,
        4,
    ),
    (
        d::SET_SRC_FORMAT_V_X8BL8GL8RL8,
        d::SET_DST_FORMAT_V_X8BL8GL8RL8,
        4,
    ),
    (d::SET_SRC_FORMAT_V_R5G6B5, d::SET_DST_FORMAT_V_R5G6B5, 2),
    (
        d::SET_SRC_FORMAT_V_A1R5G5B5,
        d::SET_DST_FORMAT_V_A1R5G5B5,
        2,
    ),
    (
        d::SET_SRC_FORMAT_V_X1R5G5B5,
        d::SET_DST_FORMAT_V_X1R5G5B5,
        2,
    ),
    (d::SET_SRC_FORMAT_V_Y8, d::SET_DST_FORMAT_V_Y8, 1),
    (d::SET_SRC_FORMAT_V_Y16, d::SET_DST_FORMAT_V_Y16, 2),
    (d::SET_SRC_FORMAT_V_Y32, d::SET_DST_FORMAT_V_Y32, 4),
    (
        d::SET_SRC_FORMAT_V_Z1R5G5B5,
        d::SET_DST_FORMAT_V_Z1R5G5B5,
        2,
    ),
    (
        d::SET_SRC_FORMAT_V_O1R5G5B5,
        d::SET_DST_FORMAT_V_O1R5G5B5,
        2,
    ),
    (
        d::SET_SRC_FORMAT_V_Z8R8G8B8,
        d::SET_DST_FORMAT_V_Z8R8G8B8,
        4,
    ),
    (
        d::SET_SRC_FORMAT_V_O8R8G8B8,
        d::SET_DST_FORMAT_V_O8R8G8B8,
        4,
    ),
    (d::SET_SRC_FORMAT_V_RF16, d::SET_DST_FORMAT_V_RF16, 2),
    (d::SET_SRC_FORMAT_V_RF32, d::SET_DST_FORMAT_V_RF32, 4),
    (
        d::SET_SRC_FORMAT_V_RF32_GF32,
        d::SET_DST_FORMAT_V_RF32_GF32,
        8,
    ),
    (
        d::SET_SRC_FORMAT_V_RF16_GF16_BF16_AF16,
        d::SET_DST_FORMAT_V_RF16_GF16_BF16_AF16,
        8,
    ),
    (
        d::SET_SRC_FORMAT_V_RF16_GF16_BF16_X16,
        d::SET_DST_FORMAT_V_RF16_GF16_BF16_X16,
        8,
    ),
    (
        d::SET_SRC_FORMAT_V_RF32_GF32_BF32_AF32,
        d::SET_DST_FORMAT_V_RF32_GF32_BF32_AF32,
        16,
    ),
    (
        d::SET_SRC_FORMAT_V_RF32_GF32_BF32_X32,
        d::SET_DST_FORMAT_V_RF32_GF32_BF32_X32,
        16,
    ),
    (
        d::SET_SRC_FORMAT_V_R16_G16_B16_A16,
        d::SET_DST_FORMAT_V_R16_G16_B16_A16,
        8,
    ),
    (
        d::SET_SRC_FORMAT_V_RN16_GN16_BN16_AN16,
        d::SET_DST_FORMAT_V_RN16_GN16_BN16_AN16,
        8,
    ),
    (
        d::SET_SRC_FORMAT_V_BF10GF11RF11,
        d::SET_DST_FORMAT_V_BF10GF11RF11,
        4,
    ),
    (
        d::SET_SRC_FORMAT_V_AN8BN8GN8RN8,
        d::SET_DST_FORMAT_V_AN8BN8GN8RN8,
        4,
    ),
    (
        d::SET_SRC_FORMAT_V_RF16_GF16,
        d::SET_DST_FORMAT_V_RF16_GF16,
        4,
    ),
    (d::SET_SRC_FORMAT_V_R16_G16, d::SET_DST_FORMAT_V_R16_G16, 4),
    (
        d::SET_SRC_FORMAT_V_RN16_GN16,
        d::SET_DST_FORMAT_V_RN16_GN16,
        4,
    ),
    (d::SET_SRC_FORMAT_V_G8R8, d::SET_DST_FORMAT_V_G8R8, 2),
    (d::SET_SRC_FORMAT_V_GN8RN8, d::SET_DST_FORMAT_V_GN8RN8, 2),
    (d::SET_SRC_FORMAT_V_RN16, d::SET_DST_FORMAT_V_RN16, 2),
    (d::SET_SRC_FORMAT_V_RN8, d::SET_DST_FORMAT_V_RN8, 1),
    (d::SET_SRC_FORMAT_V_A8, d::SET_DST_FORMAT_V_A8, 1),
];

/// Each format's pixel size, written out from its name, and the destination values that equal
/// the source ones (`cl902d.h`).
#[test]
fn source_and_destination_formats_agree() {
    let table = FORMAT_TABLE;
    for (source, destination, bytes) in table {
        assert_eq!(source, destination);
        assert_eq!(bytes_per_pixel(source), Some(bytes), "{source:#x}");
    }
    // Some values written out from the header.
    assert_eq!(d::SET_SRC_FORMAT_V_A8R8G8B8, 0xcf);
    assert_eq!(d::SET_SRC_FORMAT_V_RF32_GF32_BF32_AF32, 0xc0);
    assert_eq!(d::SET_SRC_FORMAT_V_A8, 0xf7);
    // Every value with no size here is refused.
    let sized: Vec<u32> = table.iter().map(|t| t.0).collect();
    for value in 0..=0xff {
        assert_eq!(
            bytes_per_pixel(value).is_some(),
            sized.contains(&value),
            "{value:#x}"
        );
    }
    assert_eq!(bytes_per_pixel(d::SET_SRC_FORMAT_V_Y1_8X8), None);
    assert_eq!(bytes_per_pixel(d::SET_SRC_FORMAT_V_AY8), None);
    assert_eq!(BILINEAR_FORMATS, [0xcf, 0xd5]);
}

#[test]
fn blit_registers_needed_covers_every_method_read() {
    assert_eq!(d::PIXELS_FROM_MEMORY_SRC_Y0_INT.byte_address, 0x8dc);
    assert_eq!(BLIT_REGISTERS_NEEDED, 0x8dc / 4 + 1);
    let methods = [
        (d::SET_DST_FORMAT, 0x200),
        (d::SET_DST_MEMORY_LAYOUT, 0x204),
        (d::SET_DST_BLOCK_SIZE, 0x208),
        (d::SET_DST_PITCH, 0x214),
        (d::SET_DST_OFFSET_UPPER, 0x220),
        (d::SET_SRC_FORMAT, 0x230),
        (d::SET_SRC_PITCH, 0x244),
        (d::SET_SRC_OFFSET_LOWER, 0x254),
        (d::SET_CLIP_ENABLE, 0x290),
        (d::SET_OPERATION, 0x2ac),
        (d::SET_PIXELS_FROM_MEMORY_SAMPLE_MODE, 0x88c),
        (d::SET_PIXELS_FROM_MEMORY_DST_X0, 0x8b0),
        (d::SET_PIXELS_FROM_MEMORY_DU_DX_FRAC, 0x8c0),
        (d::SET_PIXELS_FROM_MEMORY_DU_DX_INT, 0x8c4),
        (d::SET_PIXELS_FROM_MEMORY_SRC_X0_FRAC, 0x8d0),
    ];
    for (method, byte) in methods {
        assert_eq!(method.byte_address, byte, "{}", method.name);
    }
    assert_eq!(d::SET_DST_OFFSET_UPPER_V, Field::new(7, 0));
    assert_eq!(d::SET_SRC_BLOCK_SIZE_HEIGHT, Field::new(6, 4));
    assert_eq!(d::SET_SRC_BLOCK_SIZE_DEPTH, Field::new(10, 8));
}

#[test]
fn a_pitch_region_is_not_bounded_by_its_width() {
    // SET_SRC_WIDTH and SET_DST_WIDTH say 2 pixels; the regions are 6 wide from column 4.
    let source = Side::Pitch {
        address: SRC,
        pitch: 0x40,
    };
    let destination = Side::Pitch {
        address: DST,
        pitch: 0x40,
    };
    let blit = Blit {
        format: RGBA,
        source,
        destination,
        from: R {
            x: 4,
            y: 1,
            w: 6,
            h: 2,
        },
        to: R {
            x: 4,
            y: 2,
            w: 6,
            h: 2,
        },
        bilinear: false,
    };
    let mut regs = regs_for(&Blit {
        source: side_low(source),
        destination: side_low(destination),
        ..blit
    });
    regs.set(d::SET_SRC_WIDTH, 2).set(d::SET_DST_WIDTH, 2);
    let mem = memory();
    let rows = pixels(&mem, source, blit.from, 4);
    blit_method(
        d::PIXELS_FROM_MEMORY_SRC_Y0_INT.method(),
        &regs.0,
        &mem,
        &mut PendingWrites::new(),
        &mut Recorder::default(),
    )
    .unwrap();
    assert_eq!(pixels(&mem, destination, blit.to, 4), rows);
}

#[test]
fn point_sampling_clamps_to_the_last_source_pixel() {
    // One source pixel stretched over 92,819, the narrowest destination whose rounded step
    // (2^32 / 92,819, rounded up to 46,273) times its last index reaches 1.0, past the last
    // source pixel; it is clamped back to it.
    let width = 92_819u32;
    let step = ((1u128 << 32) + u128::from(width / 2)) / u128::from(width);
    assert_eq!(step, 46_273);
    assert!(
        (u128::from(width - 1) * step) >> 32 >= 1,
        "the case must reach the clamp"
    );
    let mem = memory();
    let source = Side::Pitch {
        address: SRC,
        pitch: 0x10,
    };
    let destination = Side::Pitch {
        address: DST,
        pitch: 0x20_0000,
    };
    let blit = Blit {
        format: d::SET_SRC_FORMAT_V_A8,
        source,
        destination,
        from: R {
            x: 0,
            y: 0,
            w: 1,
            h: 1,
        },
        to: R {
            x: 0,
            y: 0,
            w: width,
            h: 1,
        },
        bilinear: false,
    };
    run(&mem, &blit, &mut Recorder::default()).unwrap();
    let pixel = read(&mem, low(SRC), 1)[0];
    assert_eq!(
        read(&mem, low(DST), u64::from(width)),
        vec![pixel; width as usize]
    );
}

// --- Register reset values (the values themselves: `regs::tests`) ---

/// A block-linear to block-linear blit for the depth tests.
fn depth_blit() -> Blit {
    Blit {
        format: RGBA,
        source: Side::Bl {
            address: SRC,
            width_px: 32,
            height: 16,
            bh: 1,
        },
        destination: Side::Bl {
            address: DST,
            width_px: 32,
            height: 16,
            bh: 1,
        },
        from: R {
            x: 1,
            y: 2,
            w: 4,
            h: 3,
        },
        to: R {
            x: 2,
            y: 1,
            w: 4,
            h: 3,
        },
        bilinear: false,
    }
}

#[test]
fn a_block_linear_side_of_depth_zero_is_refused() {
    let out = starquake_layout::LayoutError::RegionOutOfBounds;
    let blit = depth_blit();
    assert_eq!(
        run_with(&blit, &|r| {
            r.set(d::SET_SRC_DEPTH, 0);
        }),
        Err(CopyError::Layout {
            side: CopySide::Source,
            error: out
        })
    );
    assert_eq!(
        run_with(&blit, &|r| {
            r.set(d::SET_DST_DEPTH, 0);
        }),
        Err(CopyError::Layout {
            side: CopySide::Destination,
            error: out
        })
    );
    // Both at 0: the source is reported.
    assert_eq!(
        run_with(&blit, &|r| {
            r.set(d::SET_SRC_DEPTH, 0).set(d::SET_DST_DEPTH, 0);
        }),
        Err(CopyError::Layout {
            side: CopySide::Source,
            error: out
        })
    );
    // The depths a new `902D` mirror holds (the class's reset values, `regs::DEFAULTS_2D`) make
    // the same blit work.
    let mirror = crate::back::state::Mirror::new(
        crate::interface::ChannelId(3),
        crate::ring::ClassId(d::CLASS),
    );
    assert_eq!(
        run_with(&blit, &|r| {
            r.set(d::SET_SRC_DEPTH, 0).set(d::SET_DST_DEPTH, 0);
            for method in [d::SET_SRC_DEPTH, d::SET_DST_DEPTH] {
                r.set(method, mirror.value(method.method()).unwrap());
            }
        }),
        Ok(())
    );
    // A pitch side never reads its depth.
    assert_eq!(
        run_with(&blit_base(), &|r| {
            r.set(d::SET_SRC_DEPTH, 0).set(d::SET_DST_DEPTH, 0);
        }),
        Ok(())
    );
}

// --- Overlapping sides ---

#[test]
fn an_overlapping_same_surface_blit_reads_the_whole_source_first() {
    // `check_copy` reads the source before the blit and expects it written whole, as
    // Eden-NX2's `SoftwareBlitEngine::Blit`. Rows 0..4 to rows 1..5 of one pitch surface: read
    // row by row, every destination row would become old row 0.
    let surface = Side::Pitch {
        address: SRC,
        pitch: 0x100,
    };
    check_copy(Blit {
        format: RGBA,
        source: surface,
        destination: surface,
        from: R {
            x: 0,
            y: 0,
            w: 8,
            h: 4,
        },
        to: R {
            x: 0,
            y: 1,
            w: 8,
            h: 4,
        },
        bilinear: false,
    });
    // The same on one block-linear surface.
    let surface = Side::Bl {
        address: SRC,
        width_px: 32,
        height: 16,
        bh: 1,
    };
    check_copy(Blit {
        format: RGBA,
        source: surface,
        destination: surface,
        from: R {
            x: 0,
            y: 0,
            w: 9,
            h: 7,
        },
        to: R {
            x: 1,
            y: 1,
            w: 9,
            h: 7,
        },
        bilinear: false,
    });
}

#[test]
fn an_overlapping_scaled_blit_samples_the_source_as_it_was() {
    // Rows 0..4 doubled onto rows 1..9 of the same surface: each source row is sampled by two
    // consecutive destination rows, some after the row was overwritten.
    let mem = memory();
    let surface = Side::Pitch {
        address: SRC,
        pitch: 0x100,
    };
    let from = R {
        x: 0,
        y: 0,
        w: 8,
        h: 4,
    };
    let to = R {
        x: 0,
        y: 1,
        w: 8,
        h: 8,
    };
    let blit = Blit {
        format: RGBA,
        source: surface,
        destination: surface,
        from,
        to,
        bilinear: false,
    };
    let src = pixels(&mem, surface, from, 4);
    let expected = expect(
        &read(&mem, LOW_BASE, PAGES * PAGE_SIZE),
        surface,
        to,
        4,
        &point_model(&src, from, to, 4),
    );
    run(&mem, &blit, &mut Recorder::default()).unwrap();
    assert!(read(&mem, LOW_BASE, PAGES * PAGE_SIZE) == expected);
}

// --- Size bound on scaled blits ---

#[test]
fn the_size_bound_counts_the_larger_side_of_a_scaled_blit() {
    let base = blit_base();
    let format = d::SET_SRC_FORMAT_V_RF32_GF32_BF32_AF32;
    let bpp = u128::from(bytes_per_pixel(format).unwrap());
    let one = R {
        x: 0,
        y: 0,
        w: 1,
        h: 1,
    };
    let big = |h: u32| R {
        x: 0,
        y: 0,
        w: 4096,
        h,
    };
    let wide = Side::Pitch {
        address: SRC,
        pitch: 0x10_0000,
    };
    // One pixel stretched over 4096 by 4097: the destination's bytes are counted.
    let up = Blit {
        format,
        destination: Side::Pitch {
            address: DST,
            pitch: 0x10_0000,
        },
        from: one,
        to: big(4097),
        ..base
    };
    assert_eq!(
        run_with(&up, &|_| ()),
        Err(CopyError::TooLarge {
            bytes: 4096 * 4097 * bpp,
            max: MAX_COPY_BYTES
        })
    );
    // At the bound the destination gets past the size check, to the probe (line 2 is past
    // the mapping).
    assert_eq!(
        run_with(
            &Blit {
                to: big(4096),
                ..up
            },
            &|_| ()
        ),
        Err(CopyError::Fault {
            side: CopySide::Destination,
            fault: crate::memory::GpuFault::Unmapped {
                va: low(DST) + 2 * 0x10_0000
            },
        })
    );
    // 4096 by 4097 shrunk to one pixel: the source's bytes are counted.
    let down = Blit {
        format,
        source: wide,
        from: big(4097),
        to: one,
        ..base
    };
    assert_eq!(
        run_with(&down, &|_| ()),
        Err(CopyError::TooLarge {
            bytes: 4096 * 4097 * bpp,
            max: MAX_COPY_BYTES
        })
    );
    assert_eq!(
        run_with(
            &Blit {
                from: big(4096),
                ..down
            },
            &|_| ()
        ),
        Err(CopyError::Fault {
            side: CopySide::Source,
            fault: crate::memory::GpuFault::Unmapped {
                va: low(SRC) + 2 * 0x10_0000
            },
        })
    );
}

// --- Pending writes on failure ---

#[test]
fn a_blit_with_a_short_register_file_still_flushes_the_pending_writes() {
    let mem = memory();
    let before = snapshot(&mem);
    let mut pending = PendingWrites::new();
    pending.add(5..9);
    let mut sink = Recorder::default();
    assert_eq!(
        blit_method(
            d::PIXELS_FROM_MEMORY_SRC_Y0_INT.method(),
            &vec![0; BLIT_REGISTERS_NEEDED - 1],
            &mem,
            &mut pending,
            &mut sink
        ),
        Err(CopyError::Registers {
            len: BLIT_REGISTERS_NEEDED - 1,
            needed: BLIT_REGISTERS_NEEDED
        })
    );
    assert_eq!(sink.events, vec![Event::Invalidate(5..9)]);
    assert!(pending.is_empty());
    assert!(snapshot(&mem) == before);
}

// --- Unsigned pitch ---

/// A pitch with bit 31 set: Eden-NX2 types `SET_SRC_PITCH` and `SET_DST_PITCH` `u32`
/// (`fermi_2d.h`, `u32 pitch`), so line 1 is `2^31` bytes above line 0, not below.
const HIGH_PITCH: u32 = 1 << 31;

#[test]
fn a_source_pitch_with_bit_31_set_is_unsigned() {
    let mem = memory();
    // Line 1 of the source, through a second mapping of a page of the same guest memory.
    mem.map(
        PageRange::new(low(SRC) + u64::from(HIGH_PITCH), PAGE_SIZE).unwrap(),
        CPU_BASE + 0x10_0000,
    )
    .unwrap();
    let source = Side::Pitch {
        address: SRC,
        pitch: HIGH_PITCH,
    };
    let destination = Side::Pitch {
        address: DST,
        pitch: 0x100,
    };
    let rect = R {
        x: 0,
        y: 0,
        w: 4,
        h: 2,
    };
    let rows = pixels(&mem, source, rect, 4);
    assert_eq!(
        rows[16..],
        read(&mem, low(SRC) + u64::from(HIGH_PITCH), 16)[..]
    );
    let blit = Blit {
        format: RGBA,
        source,
        destination,
        from: rect,
        to: rect,
        bilinear: false,
    };
    run(&mem, &blit, &mut Recorder::default()).unwrap();
    assert_eq!(pixels(&mem, destination, rect, 4), rows);
}

#[test]
fn a_destination_pitch_with_bit_31_set_is_unsigned() {
    let mem = memory();
    let line1 = low(DST) + u64::from(HIGH_PITCH);
    mem.map(
        PageRange::new(line1, PAGE_SIZE).unwrap(),
        CPU_BASE + 0x10_0000,
    )
    .unwrap();
    let source = Side::Pitch {
        address: SRC,
        pitch: 0x100,
    };
    let destination = Side::Pitch {
        address: DST,
        pitch: HIGH_PITCH,
    };
    let rect = R {
        x: 0,
        y: 0,
        w: 4,
        h: 2,
    };
    let rows = pixels(&mem, source, rect, 4);
    let blit = Blit {
        format: RGBA,
        source,
        destination,
        from: rect,
        to: rect,
        bilinear: false,
    };
    run(&mem, &blit, &mut Recorder::default()).unwrap();
    assert_eq!(read(&mem, low(DST), 16), rows[..16]);
    assert_eq!(read(&mem, line1, 16), rows[16..]);
}

// --- Depth slices ---

#[test]
fn divergence_slices_keep_the_rows_contiguous() {
    // Source origin row 6 of 8, three rows: rows 0 and 1 in slice 0, row 2 in slice 1 at row 6.
    // Destination row 2 takes source slice 1 row 6, not Eden-NX2's linear offset
    // 1 * 64 * 8 + 0 into its source buffer.
    let bl = Bl {
        width: 64,
        height: 8,
        depth: 2,
        bh: 0,
        bd: 0,
    };
    let mem = memory();
    let destination = Side::Pitch {
        address: DST,
        pitch: 0x100,
    };
    let blit = Blit {
        format: RGBA,
        source: Side::Bl {
            address: SRC,
            width_px: 16,
            height: 8,
            bh: 0,
        },
        destination,
        from: R {
            x: 0,
            y: 6,
            w: 16,
            h: 3,
        },
        to: R {
            x: 0,
            y: 0,
            w: 16,
            h: 3,
        },
        bilinear: false,
    };
    let mut regs = regs_for(&Blit {
        source: side_low(blit.source),
        destination: side_low(blit.destination),
        ..blit
    });
    regs.set(d::SET_SRC_DEPTH, bl.depth);
    let tiled = read(&mem, low(SRC), bl.size());
    blit_method(
        d::PIXELS_FROM_MEMORY_SRC_Y0_INT.method(),
        &regs.0,
        &mem,
        &mut PendingWrites::new(),
        &mut Recorder::default(),
    )
    .unwrap();
    for row in 0..3u32 {
        let (z, y) = (row / 2, 6 + row % 2);
        let offset = us(starquake_layout::block_linear_offset(
            &bl.tiling(),
            bl.extent(),
            Offset4D::new(0, y, z, 0),
        )
        .unwrap());
        assert_eq!(
            read(&mem, low(DST) + u64::from(row) * 0x100, 16),
            tiled[offset..offset + 16],
            "row {row}"
        );
    }
}
