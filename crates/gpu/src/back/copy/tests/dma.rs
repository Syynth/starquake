//! `C7B5` DMA copies: each layout pair against the layout crate, multi-line, remap, the
//! semaphore, faults at the edge, every refusal and every divergence the module docs list.

use proptest::prelude::*;
use proptest::test_runner::{Config as ProptestConfig, RngSeed};
use starquake_layout::LayoutError;
use starquake_memory::PAGE_SIZE;

use super::super::surface::CopySide;
use super::super::{
    CopyError, DMA_REGISTERS_NEEDED, MAX_COPY_BYTES, PendingWrites, check_size, dma_method,
};
use super::*;
use crate::classes::clc7b5 as c;
use crate::memory::GpuFault;
use crate::ring::ReleaseOp;
use crate::sema::{
    self, DecodeError, PayloadSize, Reduction, ReductionFormat, ReductionOp, ReleaseForm,
};

/// One side of a test copy.
#[derive(Clone, Copy, Debug)]
enum Side {
    Pitch {
        address: u64,
        pitch: i32,
    },
    Bl {
        address: u64,
        bl: Bl,
        origin: (u32, u32),
    },
}

impl Side {
    fn address(self) -> u64 {
        match self {
            Side::Pitch { address, .. } | Side::Bl { address, .. } => address,
        }
    }
}

/// A launch's settings.
#[derive(Clone, Copy, Debug)]
struct Copy {
    source: Side,
    destination: Side,
    length: u32,
    lines: u32,
    multi_line: bool,
}

fn regs_for(copy: &Copy) -> Regs {
    let mut regs = Regs::new(DMA_REGISTERS_NEEDED);
    regs.field(
        c::LAUNCH_DMA,
        c::LAUNCH_DMA_DATA_TRANSFER_TYPE,
        c::LAUNCH_DMA_DATA_TRANSFER_TYPE_NON_PIPELINED,
    )
    .field(
        c::LAUNCH_DMA,
        c::LAUNCH_DMA_MULTI_LINE_ENABLE,
        u32::from(copy.multi_line),
    )
    .set(c::LINE_LENGTH_IN, copy.length)
    .set(c::LINE_COUNT, copy.lines);
    for (side, is_source) in [(copy.source, true), (copy.destination, false)] {
        let (upper, lower, pitch_m) = if is_source {
            (c::OFFSET_IN_UPPER, c::OFFSET_IN_LOWER, c::PITCH_IN)
        } else {
            (c::OFFSET_OUT_UPPER, c::OFFSET_OUT_LOWER, c::PITCH_OUT)
        };
        regs.set(upper, hi(side.address()))
            .set(lower, lo(side.address()));
        let layout = if is_source {
            c::LAUNCH_DMA_SRC_MEMORY_LAYOUT
        } else {
            c::LAUNCH_DMA_DST_MEMORY_LAYOUT
        };
        match side {
            Side::Pitch { pitch, .. } => {
                regs.field(c::LAUNCH_DMA, layout, c::LAUNCH_DMA_SRC_MEMORY_LAYOUT_PITCH)
                    .set(pitch_m, pitch.cast_unsigned());
            }
            Side::Bl { bl, origin, .. } => {
                regs.field(
                    c::LAUNCH_DMA,
                    layout,
                    c::LAUNCH_DMA_SRC_MEMORY_LAYOUT_BLOCKLINEAR,
                );
                if is_source {
                    regs.field(
                        c::SET_SRC_BLOCK_SIZE,
                        c::SET_SRC_BLOCK_SIZE_HEIGHT,
                        bl.bh.into(),
                    )
                    .field(
                        c::SET_SRC_BLOCK_SIZE,
                        c::SET_SRC_BLOCK_SIZE_DEPTH,
                        bl.bd.into(),
                    )
                    .set(c::SET_SRC_WIDTH, bl.width)
                    .set(c::SET_SRC_HEIGHT, bl.height)
                    .set(c::SET_SRC_DEPTH, bl.depth)
                    .field(c::SET_SRC_ORIGIN, c::SET_SRC_ORIGIN_X, origin.0)
                    .field(c::SET_SRC_ORIGIN, c::SET_SRC_ORIGIN_Y, origin.1);
                } else {
                    regs.field(
                        c::SET_DST_BLOCK_SIZE,
                        c::SET_DST_BLOCK_SIZE_HEIGHT,
                        bl.bh.into(),
                    )
                    .field(
                        c::SET_DST_BLOCK_SIZE,
                        c::SET_DST_BLOCK_SIZE_DEPTH,
                        bl.bd.into(),
                    )
                    .set(c::SET_DST_WIDTH, bl.width)
                    .set(c::SET_DST_HEIGHT, bl.height)
                    .set(c::SET_DST_DEPTH, bl.depth)
                    .field(c::SET_DST_ORIGIN, c::SET_DST_ORIGIN_X, origin.0)
                    .field(c::SET_DST_ORIGIN, c::SET_DST_ORIGIN_Y, origin.1);
                }
            }
        }
    }
    regs
}

fn launch(mem: &Memory, regs: &Regs, sink: &mut Recorder) -> Result<(), CopyError> {
    dma_method(
        c::LAUNCH_DMA.method(),
        &regs.0,
        mem,
        &mut PendingWrites::new(),
        sink,
    )
}

/// The lines of the copy's region on `side`, read as linear rows of `length` bytes.
fn gather(mem: &Memory, side: Side, length: u32, lines: u32) -> Vec<u8> {
    match side {
        Side::Pitch { address, pitch } => (0..lines)
            .flat_map(|l| {
                let start =
                    u64::try_from(i128::from(address) + i128::from(l) * i128::from(pitch)).unwrap();
                read(mem, start, length.into())
            })
            .collect(),
        Side::Bl {
            address,
            bl,
            origin,
        } => {
            let tiled = read(mem, address, bl.size());
            bl.untile(&tiled, origin, length, lines)
        }
    }
}

/// What memory must hold after copying `linear` (rows of `length`) to `side`, given `before`.
fn scatter(before: &[u8], side: Side, linear: &[u8], length: u32, lines: u32) -> Vec<u8> {
    let mut after = before.to_vec();
    match side {
        Side::Pitch { address, pitch } => {
            for l in 0..lines {
                let start = usize::try_from(
                    i128::from(address - GPU_BASE) + i128::from(l) * i128::from(pitch),
                )
                .unwrap();
                let row = &linear[(l * length) as usize..((l + 1) * length) as usize];
                after[start..start + length as usize].copy_from_slice(row);
            }
        }
        Side::Bl {
            address,
            bl,
            origin,
        } => {
            let start = us(address - GPU_BASE);
            let end = start + us(bl.size());
            bl.tile(
                &mut after[start..end],
                linear,
                length as usize,
                origin,
                length,
                lines,
            );
        }
    }
    after
}

/// Runs `copy` and checks every byte of memory against the layout crate's oracle.
fn check_against_layout(copy: Copy) {
    let mem = memory();
    let lines = if copy.multi_line { copy.lines } else { 1 };
    let linear = gather(&mem, copy.source, copy.length, lines);
    let expected = scatter(
        &snapshot(&mem),
        copy.destination,
        &linear,
        copy.length,
        lines,
    );
    let mut sink = Recorder::default();
    launch(&mem, &regs_for(&copy), &mut sink).unwrap();
    assert!(snapshot(&mem) == expected, "{copy:?}");
}

const SRC: u64 = GPU_BASE + 0x1_0000;
const DST: u64 = GPU_BASE + 0x9_0000;

#[test]
fn pitch_to_pitch_copies_each_line_at_its_own_pitch() {
    check_against_layout(Copy {
        source: Side::Pitch {
            address: SRC + 3,
            pitch: 0x150,
        },
        destination: Side::Pitch {
            address: DST + 5,
            pitch: 0x230,
        },
        length: 0x123,
        lines: 7,
        multi_line: true,
    });
}

#[test]
fn pitch_to_block_linear_matches_the_layout_crates_tile() {
    let bl = Bl {
        width: 300,
        height: 70,
        depth: 1,
        bh: 2,
        bd: 0,
    };
    check_against_layout(Copy {
        source: Side::Pitch {
            address: SRC + 1,
            pitch: 211,
        },
        destination: Side::Bl {
            address: DST,
            bl,
            origin: (37, 9),
        },
        length: 200,
        lines: 50,
        multi_line: true,
    });
}

#[test]
fn block_linear_to_pitch_matches_the_layout_crates_untile() {
    let bl = Bl {
        width: 256,
        height: 40,
        depth: 1,
        bh: 1,
        bd: 0,
    };
    check_against_layout(Copy {
        source: Side::Bl {
            address: SRC,
            bl,
            origin: (17, 3),
        },
        destination: Side::Pitch {
            address: DST + 7,
            pitch: 250,
        },
        length: 230,
        lines: 33,
        multi_line: true,
    });
}

#[test]
fn block_linear_to_block_linear_matches_untile_then_tile() {
    check_against_layout(Copy {
        source: Side::Bl {
            address: SRC,
            bl: Bl {
                width: 192,
                height: 64,
                depth: 1,
                bh: 3,
                bd: 0,
            },
            origin: (64, 8),
        },
        destination: Side::Bl {
            address: DST,
            bl: Bl {
                width: 500,
                height: 30,
                depth: 1,
                bh: 0,
                bd: 0,
            },
            origin: (3, 1),
        },
        length: 128,
        lines: 29,
        multi_line: true,
    });
}

fn bl_side(address: u64, region: (u32, u32)) -> impl Strategy<Value = Side> {
    let (w, h) = region;
    (0u32..64, 0u32..20, 0u8..=2, 0u32..=40, 0u32..=10).prop_map(
        move |(ox, oy, bh, extra_w, extra_h)| Side::Bl {
            address,
            bl: Bl {
                width: ox + w + extra_w,
                height: oy + h + extra_h,
                depth: 1,
                bh,
                bd: 0,
            },
            origin: (ox, oy),
        },
    )
}

fn pitch_side(address: u64, w: u32) -> impl Strategy<Value = Side> {
    (0u32..60).prop_map(move |extra| Side::Pitch {
        address,
        pitch: (w + extra).cast_signed(),
    })
}

fn any_side(address: u64, region: (u32, u32)) -> impl Strategy<Value = Side> {
    prop_oneof![pitch_side(address, region.0), bl_side(address, region)]
}

fn any_copy() -> impl Strategy<Value = Copy> {
    (1u32..=150, 1u32..=24).prop_flat_map(|region| {
        (any_side(SRC, region), any_side(DST, region)).prop_map(move |(source, destination)| Copy {
            source,
            destination,
            length: region.0,
            lines: region.1,
            multi_line: true,
        })
    })
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 96,
        rng_seed: RngSeed::Fixed(0x6331_3132),
        ..ProptestConfig::default()
    })]

    /// Every layout pair, generated geometry: the copy equals the layout crate's untile and
    /// tile, byte for byte across the whole memory.
    #[test]
    fn generated_copies_match_the_layout_crate(copy in any_copy()) {
        check_against_layout(copy);
    }
}

#[test]
fn without_multi_line_only_the_first_line_is_copied() {
    check_against_layout(Copy {
        source: Side::Pitch {
            address: SRC,
            pitch: 0x100,
        },
        destination: Side::Pitch {
            address: DST,
            pitch: 0x100,
        },
        length: 0x80,
        lines: 9,
        multi_line: false,
    });
}

#[test]
fn a_line_longer_than_a_chunk_is_copied_whole() {
    // Two and a half chunks in one line, each side unaligned.
    let length = u32::try_from(super::super::dma::CHUNK_BYTES * 5 / 2).unwrap() + 3;
    check_against_layout(Copy {
        source: Side::Pitch {
            address: SRC + 1,
            pitch: 0,
        },
        destination: Side::Pitch {
            address: DST + 2,
            pitch: 0,
        },
        length,
        lines: 1,
        multi_line: true,
    });
}

#[test]
fn lines_past_the_last_row_continue_in_the_next_slice() {
    // 3 rows per slice from origin row 5 of 8: lines 0..3 in slice 0, 3..6 in slice 1.
    let bl = Bl {
        width: 64,
        height: 8,
        depth: 4,
        bh: 0,
        bd: 1,
    };
    let mem = memory();
    let copy = Copy {
        source: Side::Pitch {
            address: SRC,
            pitch: 40,
        },
        destination: Side::Bl {
            address: DST,
            bl,
            origin: (8, 5),
        },
        length: 40,
        lines: 6,
        multi_line: true,
    };
    let linear = gather(&mem, copy.source, 40, 6);
    launch(&mem, &regs_for(&copy), &mut Recorder::default()).unwrap();
    for line in 0..6u32 {
        let (z, y) = (line / 3, 5 + line % 3);
        for x in 0..40u32 {
            let offset = starquake_layout::block_linear_offset(
                &bl.tiling(),
                bl.extent(),
                Offset4D::new(8 + x, y, z, 0),
            )
            .unwrap();
            assert_eq!(
                read(&mem, DST + offset, 1)[0],
                linear[(line * 40 + x) as usize],
                "line {line} x {x}"
            );
        }
    }
}

// --- Remap ---

fn remap(regs: &mut Regs, size: u32, sources: u32, destinations: u32, selects: [u32; 4]) {
    regs.field(
        c::LAUNCH_DMA,
        c::LAUNCH_DMA_REMAP_ENABLE,
        c::LAUNCH_DMA_REMAP_ENABLE_TRUE,
    )
    .field(
        c::SET_REMAP_COMPONENTS,
        c::SET_REMAP_COMPONENTS_COMPONENT_SIZE,
        size - 1,
    )
    .field(
        c::SET_REMAP_COMPONENTS,
        c::SET_REMAP_COMPONENTS_NUM_SRC_COMPONENTS,
        sources - 1,
    )
    .field(
        c::SET_REMAP_COMPONENTS,
        c::SET_REMAP_COMPONENTS_NUM_DST_COMPONENTS,
        destinations - 1,
    )
    .field(
        c::SET_REMAP_COMPONENTS,
        c::SET_REMAP_COMPONENTS_DST_X,
        selects[0],
    )
    .field(
        c::SET_REMAP_COMPONENTS,
        c::SET_REMAP_COMPONENTS_DST_Y,
        selects[1],
    )
    .field(
        c::SET_REMAP_COMPONENTS,
        c::SET_REMAP_COMPONENTS_DST_Z,
        selects[2],
    )
    .field(
        c::SET_REMAP_COMPONENTS,
        c::SET_REMAP_COMPONENTS_DST_W,
        selects[3],
    )
    .set(c::SET_REMAP_CONST_A, 0xa1a2_a3a4)
    .set(c::SET_REMAP_CONST_B, 0xb1b2_b3b4);
}

const X: u32 = c::SET_REMAP_COMPONENTS_DST_X_SRC_X;
const Y: u32 = c::SET_REMAP_COMPONENTS_DST_X_SRC_Y;
const Z: u32 = c::SET_REMAP_COMPONENTS_DST_X_SRC_Z;
const W: u32 = c::SET_REMAP_COMPONENTS_DST_X_SRC_W;
const A: u32 = c::SET_REMAP_COMPONENTS_DST_X_CONST_A;
const B: u32 = c::SET_REMAP_COMPONENTS_DST_X_CONST_B;
const KEEP: u32 = c::SET_REMAP_COMPONENTS_DST_X_NO_WRITE;

/// The model of one remapped element: each destination component from its selector.
fn remap_model(src: &[u8], old: &[u8], size: usize, selects: &[u32]) -> Vec<u8> {
    let mut out = old.to_vec();
    for (j, &select) in selects.iter().enumerate() {
        let part = &mut out[j * size..(j + 1) * size];
        match select {
            s if s <= W => part.copy_from_slice(&src[s as usize * size..(s as usize + 1) * size]),
            s if s == A => part.copy_from_slice(&0xa1a2_a3a4u32.to_le_bytes()[..size]),
            s if s == B => part.copy_from_slice(&0xb1b2_b3b4u32.to_le_bytes()[..size]),
            _ => {}
        }
    }
    out
}

/// A pitch-to-pitch remapped copy of 5 elements by 3 lines, checked against the model.
fn check_remap(size: u32, sources: u32, selects: &[u32]) {
    let destinations = u32::try_from(selects.len()).unwrap();
    let mut all = [KEEP; 4];
    all[..selects.len()].copy_from_slice(selects);
    let (s, d) = ((size * sources) as usize, (size * destinations) as usize);
    let copy = Copy {
        source: Side::Pitch {
            address: SRC + 1,
            pitch: 0x40,
        },
        destination: Side::Pitch {
            address: DST + 3,
            pitch: 0x50,
        },
        length: 5,
        lines: 3,
        multi_line: true,
    };
    let mem = memory();
    let before = snapshot(&mem);
    let mut regs = regs_for(&copy);
    remap(&mut regs, size, sources, destinations, all);
    launch(&mem, &regs, &mut Recorder::default()).unwrap();
    let mut expected = before.clone();
    for line in 0..3usize {
        for e in 0..5usize {
            let src_at = us(SRC + 1 - GPU_BASE) + line * 0x40 + e * s;
            let dst_at = us(DST + 3 - GPU_BASE) + line * 0x50 + e * d;
            let element = remap_model(
                &before[src_at..src_at + s],
                &before[dst_at..dst_at + d],
                size as usize,
                selects,
            );
            expected[dst_at..dst_at + d].copy_from_slice(&element);
        }
    }
    assert!(
        snapshot(&mem) == expected,
        "size {size} sources {sources} {selects:?}"
    );
}

#[test]
fn remap_swaps_components() {
    check_remap(1, 4, &[Z, Y, X, W]);
    check_remap(2, 4, &[W, Z, Y, X]);
}

#[test]
fn remap_fills_constants_and_keeps_no_write_components() {
    check_remap(4, 1, &[A]);
    check_remap(1, 2, &[Y, KEEP, B, X]);
    check_remap(3, 3, &[KEEP, Z, A]);
}

#[test]
fn remap_changes_the_element_size_between_sides() {
    // Three 2-byte source components to four destination ones.
    check_remap(2, 3, &[Z, Y, X, B]);
    // Four 4-byte components to one.
    check_remap(4, 4, &[W]);
}

#[test]
fn remap_with_a_block_linear_destination_counts_width_and_origin_in_elements() {
    // Elements of 4 one-byte components: width 20 elements is 80 bytes, origin x 3 is 12 bytes.
    let bl = Bl {
        width: 80,
        height: 16,
        depth: 1,
        bh: 1,
        bd: 0,
    };
    let mem = memory();
    let before = snapshot(&mem);
    let copy = Copy {
        source: Side::Pitch {
            address: SRC,
            pitch: 64,
        },
        destination: Side::Bl {
            address: DST,
            bl: Bl { width: 20, ..bl },
            origin: (3, 2),
        },
        length: 10,
        lines: 4,
        multi_line: true,
    };
    let mut regs = regs_for(&copy);
    remap(&mut regs, 1, 4, 4, [W, Z, Y, X]);
    launch(&mem, &regs, &mut Recorder::default()).unwrap();
    // The model: reverse each 4-byte element, then tile 40 bytes by 4 rows at (12, 2).
    let mut linear = Vec::new();
    for line in 0..4u64 {
        let start = us(SRC - GPU_BASE + line * 64);
        for e in 0..10 {
            let mut element = before[start + e * 4..start + e * 4 + 4].to_vec();
            element.reverse();
            linear.extend(element);
        }
    }
    let mut tiled = before[us(DST - GPU_BASE)..us(DST - GPU_BASE + bl.size())].to_vec();
    bl.tile(&mut tiled, &linear, 40, (12, 2), 40, 4);
    assert_eq!(read(&mem, DST, bl.size()), tiled);
}

// --- The semaphore ---

/// A launch that copies nothing (`LINE_LENGTH_IN` 0) with semaphore settings applied by `f`.
fn semaphore_launch(f: impl FnOnce(&mut Regs)) -> Result<Vec<Release>, CopyError> {
    let mem = memory();
    let mut regs = regs_for(&Copy {
        source: Side::Pitch {
            address: SRC,
            pitch: 0,
        },
        destination: Side::Pitch {
            address: DST,
            pitch: 0,
        },
        length: 0,
        lines: 1,
        multi_line: false,
    });
    regs.set(c::SET_SEMAPHORE_A, 0x1_2345)
        .set(c::SET_SEMAPHORE_B, 0x6789_abc0)
        .set(c::SET_SEMAPHORE_PAYLOAD, 0x1122_3344)
        .set(c::SET_SEMAPHORE_PAYLOAD_UPPER, 0x5566_7788);
    f(&mut regs);
    let mut sink = Recorder::default();
    launch(&mem, &regs, &mut sink).map(|()| sink.releases())
}

const SEM_ADDRESS: u64 = 0x1_2345_6789_abc0;

fn semaphore_type(regs: &mut Regs, value: u32) -> &mut Regs {
    regs.field(c::LAUNCH_DMA, c::LAUNCH_DMA_SEMAPHORE_TYPE, value)
}

#[test]
fn semaphore_fields_give_the_exact_release() {
    // No semaphore: nothing released.
    assert_eq!(semaphore_launch(|_| ()), Ok(vec![]));
    // One word, 32-bit payload, written.
    assert_eq!(
        semaphore_launch(|r| {
            semaphore_type(r, c::LAUNCH_DMA_SEMAPHORE_TYPE_RELEASE_ONE_WORD_SEMAPHORE);
        }),
        Ok(vec![Release {
            address: SEM_ADDRESS,
            payload: 0x1122_3344,
            form: ReleaseForm {
                size: PayloadSize::Four,
                timestamp: false
            },
            op: ReleaseOp::Write,
        }])
    );
    // Four words: with a timestamp, and a two-word payload.
    assert_eq!(
        semaphore_launch(|r| {
            semaphore_type(r, c::LAUNCH_DMA_SEMAPHORE_TYPE_RELEASE_FOUR_WORD_SEMAPHORE).field(
                c::LAUNCH_DMA,
                c::LAUNCH_DMA_SEMAPHORE_PAYLOAD_SIZE,
                c::LAUNCH_DMA_SEMAPHORE_PAYLOAD_SIZE_TWO_WORD,
            );
        }),
        Ok(vec![Release {
            address: SEM_ADDRESS,
            payload: 0x5566_7788_1122_3344,
            form: ReleaseForm {
                size: PayloadSize::Eight,
                timestamp: true
            },
            op: ReleaseOp::Write,
        }])
    );
    // Each reduction and sign the header defines.
    let reductions = [
        (c::LAUNCH_DMA_SEMAPHORE_REDUCTION_IMIN, ReductionOp::Min),
        (c::LAUNCH_DMA_SEMAPHORE_REDUCTION_IMAX, ReductionOp::Max),
        (c::LAUNCH_DMA_SEMAPHORE_REDUCTION_IXOR, ReductionOp::Xor),
        (c::LAUNCH_DMA_SEMAPHORE_REDUCTION_IAND, ReductionOp::And),
        (c::LAUNCH_DMA_SEMAPHORE_REDUCTION_IOR, ReductionOp::Or),
        (c::LAUNCH_DMA_SEMAPHORE_REDUCTION_IADD, ReductionOp::Add),
        (c::LAUNCH_DMA_SEMAPHORE_REDUCTION_INC, ReductionOp::Inc),
        (c::LAUNCH_DMA_SEMAPHORE_REDUCTION_DEC, ReductionOp::Dec),
    ];
    for (value, op) in reductions {
        for (sign, format) in [
            (
                c::LAUNCH_DMA_SEMAPHORE_REDUCTION_SIGN_SIGNED,
                ReductionFormat::Signed,
            ),
            (
                c::LAUNCH_DMA_SEMAPHORE_REDUCTION_SIGN_UNSIGNED,
                ReductionFormat::Unsigned,
            ),
        ] {
            let releases = semaphore_launch(|r| {
                semaphore_type(r, c::LAUNCH_DMA_SEMAPHORE_TYPE_RELEASE_ONE_WORD_SEMAPHORE)
                    .field(
                        c::LAUNCH_DMA,
                        c::LAUNCH_DMA_SEMAPHORE_REDUCTION_ENABLE,
                        c::LAUNCH_DMA_SEMAPHORE_REDUCTION_ENABLE_TRUE,
                    )
                    .field(c::LAUNCH_DMA, c::LAUNCH_DMA_SEMAPHORE_REDUCTION, value)
                    .field(c::LAUNCH_DMA, c::LAUNCH_DMA_SEMAPHORE_REDUCTION_SIGN, sign);
            });
            assert_eq!(
                releases,
                Ok(vec![Release {
                    address: SEM_ADDRESS,
                    payload: 0x1122_3344,
                    form: ReleaseForm {
                        size: PayloadSize::Four,
                        timestamp: false
                    },
                    op: ReleaseOp::Reduce(Reduction { op, format }),
                }]),
                "reduction {value} sign {sign}"
            );
        }
    }
}

#[test]
fn a_reduction_field_without_enable_is_a_write() {
    let releases = semaphore_launch(|r| {
        semaphore_type(r, c::LAUNCH_DMA_SEMAPHORE_TYPE_RELEASE_ONE_WORD_SEMAPHORE).field(
            c::LAUNCH_DMA,
            c::LAUNCH_DMA_SEMAPHORE_REDUCTION,
            c::LAUNCH_DMA_SEMAPHORE_REDUCTION_FADD,
        );
    })
    .unwrap();
    assert_eq!(releases[0].op, ReleaseOp::Write);
}

#[test]
fn the_dma_reduction_values_are_the_host_encodings() {
    // `sema::host_reduction` decodes `LAUNCH_DMA_SEMAPHORE_REDUCTION`: the values must agree.
    assert_eq!(
        c::LAUNCH_DMA_SEMAPHORE_REDUCTION_IMIN,
        sema::HOST_REDUCTION_MIN
    );
    assert_eq!(
        c::LAUNCH_DMA_SEMAPHORE_REDUCTION_IMAX,
        sema::HOST_REDUCTION_MAX
    );
    assert_eq!(
        c::LAUNCH_DMA_SEMAPHORE_REDUCTION_IXOR,
        sema::HOST_REDUCTION_XOR
    );
    assert_eq!(
        c::LAUNCH_DMA_SEMAPHORE_REDUCTION_IAND,
        sema::HOST_REDUCTION_AND
    );
    assert_eq!(
        c::LAUNCH_DMA_SEMAPHORE_REDUCTION_IOR,
        sema::HOST_REDUCTION_OR
    );
    assert_eq!(
        c::LAUNCH_DMA_SEMAPHORE_REDUCTION_IADD,
        sema::HOST_REDUCTION_ADD
    );
    assert_eq!(
        c::LAUNCH_DMA_SEMAPHORE_REDUCTION_INC,
        sema::HOST_REDUCTION_INC
    );
    assert_eq!(
        c::LAUNCH_DMA_SEMAPHORE_REDUCTION_DEC,
        sema::HOST_REDUCTION_DEC
    );
    assert_eq!(
        c::LAUNCH_DMA_SEMAPHORE_REDUCTION_SIGN_SIGNED,
        sema::HOST_FORMAT_SIGNED
    );
    assert_eq!(
        c::LAUNCH_DMA_SEMAPHORE_REDUCTION_SIGN_UNSIGNED,
        sema::HOST_FORMAT_UNSIGNED
    );
}

// --- Faults at the edge ---

/// A pitch copy of one line of `len` bytes ending at the hole's first byte or one past.
fn edge_copy(source_len_past: Option<u64>, destination_len_past: Option<u64>) -> Copy {
    // The hole is page 100; the mapped edge is its first byte.
    let edge = GPU_BASE + 100 * PAGE_SIZE;
    let source = match source_len_past {
        Some(past) => edge - 0x40 + past,
        None => SRC,
    };
    let destination = match destination_len_past {
        Some(past) => edge - 0x40 + past,
        None => DST,
    };
    Copy {
        source: Side::Pitch {
            address: source,
            pitch: 0,
        },
        destination: Side::Pitch {
            address: destination,
            pitch: 0,
        },
        length: 0x40,
        lines: 1,
        multi_line: false,
    }
}

const HOLE: u64 = GPU_BASE + 100 * PAGE_SIZE;

#[test]
fn a_source_ending_at_the_hole_copies_and_one_byte_more_faults_there() {
    let mem = memory_with_hole(100);
    assert_eq!(
        launch(
            &mem,
            &regs_for(&edge_copy(Some(0), None)),
            &mut Recorder::default()
        ),
        Ok(())
    );
    assert_eq!(
        launch(
            &mem,
            &regs_for(&edge_copy(Some(1), None)),
            &mut Recorder::default()
        ),
        Err(CopyError::Fault {
            side: CopySide::Source,
            fault: GpuFault::Unmapped { va: HOLE },
        })
    );
}

#[test]
fn a_destination_ending_at_the_hole_copies_and_one_byte_more_faults_there() {
    let mem = memory_with_hole(100);
    assert_eq!(
        launch(
            &mem,
            &regs_for(&edge_copy(None, Some(0))),
            &mut Recorder::default()
        ),
        Ok(())
    );
    assert_eq!(
        launch(
            &mem,
            &regs_for(&edge_copy(None, Some(1))),
            &mut Recorder::default()
        ),
        Err(CopyError::Fault {
            side: CopySide::Destination,
            fault: GpuFault::Unmapped { va: HOLE },
        })
    );
}

#[test]
fn a_block_linear_destination_reaching_the_hole_faults_at_its_first_byte() {
    // A 64-byte-wide surface of 8 rows per GOB: 512 bytes per GOB, 8 GOBs a page. Rows 0..64
    // fill page 99 exactly (64 rows of 64 bytes); row 64 starts in the hole.
    let bl = Bl {
        width: 64,
        height: 72,
        depth: 1,
        bh: 0,
        bd: 0,
    };
    let start = HOLE - PAGE_SIZE;
    let mem = memory_with_hole(100);
    let mut copy = Copy {
        source: Side::Pitch {
            address: SRC,
            pitch: 64,
        },
        destination: Side::Bl {
            address: start,
            bl,
            origin: (0, 0),
        },
        length: 64,
        lines: 64,
        multi_line: true,
    };
    assert_eq!(
        launch(&mem, &regs_for(&copy), &mut Recorder::default()),
        Ok(())
    );
    copy.lines = 65;
    assert_eq!(
        launch(&mem, &regs_for(&copy), &mut Recorder::default()),
        Err(CopyError::Fault {
            side: CopySide::Destination,
            fault: GpuFault::Unmapped { va: HOLE },
        })
    );
}

#[test]
fn with_both_sides_faulting_the_source_is_reported() {
    let mem = memory_with_hole(100);
    assert_eq!(
        launch(
            &mem,
            &regs_for(&edge_copy(Some(1), Some(1))),
            &mut Recorder::default()
        ),
        Err(CopyError::Fault {
            side: CopySide::Source,
            fault: GpuFault::Unmapped { va: HOLE },
        })
    );
}

#[test]
fn a_fault_found_by_the_probe_writes_nothing_and_the_next_launch_works() {
    let mem = memory_with_hole(100);
    // Many lines are fine; only the last line's end reaches the hole.
    let mut copy = Copy {
        source: Side::Pitch {
            address: HOLE - 0x40 * 8,
            pitch: 0x40,
        },
        destination: Side::Pitch {
            address: DST,
            pitch: 0x40,
        },
        length: 0x40,
        lines: 9,
        multi_line: true,
    };
    let mut regs = regs_for(&copy);
    semaphore_type(
        &mut regs,
        c::LAUNCH_DMA_SEMAPHORE_TYPE_RELEASE_ONE_WORD_SEMAPHORE,
    );
    let before = read(&mem, DST, 0x40 * 9);
    let mut sink = Recorder::default();
    assert_eq!(
        launch(&mem, &regs, &mut sink),
        Err(CopyError::Fault {
            side: CopySide::Source,
            fault: GpuFault::Unmapped { va: HOLE },
        })
    );
    assert_eq!(read(&mem, DST, 0x40 * 9), before);
    assert_eq!(sink.events, vec![]);
    copy.lines = 8;
    let mut regs = regs_for(&copy);
    semaphore_type(
        &mut regs,
        c::LAUNCH_DMA_SEMAPHORE_TYPE_RELEASE_ONE_WORD_SEMAPHORE,
    );
    assert_eq!(launch(&mem, &regs, &mut sink), Ok(()));
    assert_eq!(
        read(&mem, DST, 0x40 * 8),
        read(&mem, HOLE - 0x40 * 8, 0x40 * 8)
    );
    assert_eq!(sink.releases().len(), 1);
}

// --- What the sink sees ---

#[test]
fn the_sink_sees_pending_then_current_then_invalidate_then_release() {
    let mem = memory();
    let bl = Bl {
        width: 128,
        height: 16,
        depth: 1,
        bh: 1,
        bd: 0,
    };
    let copy = Copy {
        source: Side::Pitch {
            address: SRC + 0x10,
            pitch: -0x100,
        },
        destination: Side::Bl {
            address: DST,
            bl,
            origin: (4, 2),
        },
        length: 0x20,
        lines: 3,
        multi_line: true,
    };
    let mut regs = regs_for(&copy);
    semaphore_type(
        &mut regs,
        c::LAUNCH_DMA_SEMAPHORE_TYPE_RELEASE_ONE_WORD_SEMAPHORE,
    )
    .set(c::SET_SEMAPHORE_B, 0x40);
    let mut pending = PendingWrites::new();
    pending.add(0x300..0x310);
    pending.add(0x100..0x108);
    let mut sink = Recorder::default();
    dma_method(
        c::LAUNCH_DMA.method(),
        &regs.0,
        &mem,
        &mut pending,
        &mut sink,
    )
    .unwrap();
    assert!(pending.is_empty());
    assert_eq!(
        sink.events,
        vec![
            Event::Invalidate(0x100..0x108),
            Event::Invalidate(0x300..0x310),
            // Lines at +0x10, -0xf0 and -0x1f0: the cover runs from the lowest start to the end
            // of the highest line.
            Event::MakeCurrent(SRC + 0x10 - 0x200..SRC + 0x10 + 0x20),
            // The whole surface padded to blocks of 16 rows: 128 by 16 bytes.
            Event::MakeCurrent(DST..DST + 128 * 16),
            Event::Invalidate(DST..DST + 128 * 16),
            Event::Release(Release {
                address: 0x40,
                payload: 0,
                form: ReleaseForm {
                    size: PayloadSize::Four,
                    timestamp: false
                },
                op: ReleaseOp::Write,
            }),
        ]
    );
}

#[test]
fn a_launch_that_fails_still_flushes_the_pending_writes() {
    let mem = memory();
    let mut regs = regs_for(&edge_copy(None, None));
    semaphore_type(
        &mut regs,
        c::LAUNCH_DMA_SEMAPHORE_TYPE_RELEASE_CONDITIONAL_INTR_SEMAPHORE,
    );
    let mut pending = PendingWrites::new();
    pending.add(0x10..0x20);
    let mut sink = Recorder::default();
    assert_eq!(
        dma_method(
            c::LAUNCH_DMA.method(),
            &regs.0,
            &mem,
            &mut pending,
            &mut sink
        ),
        Err(CopyError::SemaphoreType(
            c::LAUNCH_DMA_SEMAPHORE_TYPE_RELEASE_CONDITIONAL_INTR_SEMAPHORE
        ))
    );
    assert_eq!(sink.events, vec![Event::Invalidate(0x10..0x20)]);
}

#[test]
fn other_methods_do_nothing_and_leave_pending_writes_alone() {
    let mem = memory();
    let before = snapshot(&mem);
    let regs = regs_for(&edge_copy(None, None));
    let mut pending = PendingWrites::new();
    pending.add(0x10..0x20);
    let mut sink = Recorder::default();
    for method in [c::LINE_COUNT.method(), c::LAUNCH_DMA.method() + 1, 0] {
        assert_eq!(
            dma_method(method, &regs.0, &mem, &mut pending, &mut sink),
            Ok(())
        );
    }
    assert_eq!(pending.ranges(), vec![0x10..0x20]);
    assert_eq!(sink.events, vec![]);
    assert!(snapshot(&mem) == before);
}

// --- Refusals ---

/// The copy the refusal tests change one register of.
fn refusal_base() -> Copy {
    Copy {
        source: Side::Pitch {
            address: SRC,
            pitch: 0x40,
        },
        destination: Side::Bl {
            address: DST,
            bl: Bl {
                width: 64,
                height: 8,
                depth: 1,
                bh: 0,
                bd: 0,
            },
            origin: (0, 0),
        },
        length: 0x40,
        lines: 8,
        multi_line: true,
    }
}

/// Launches the refusal base changed by `f`; a refused launch must leave memory and the sink
/// untouched.
fn refused(f: &dyn Fn(&mut Regs)) -> Result<(), CopyError> {
    let mem = memory();
    let mut regs = regs_for(&refusal_base());
    f(&mut regs);
    let before = snapshot(&mem);
    let mut sink = Recorder::default();
    let result = launch(&mem, &regs, &mut sink);
    if result.is_err() {
        assert!(snapshot(&mem) == before, "a refused launch wrote");
        assert_eq!(sink.events, vec![], "a refused launch reached the sink");
    }
    result
}

#[test]
fn refusals_are_exact() {
    let mem = memory();
    let base = refusal_base();
    // The register file.
    assert_eq!(
        dma_method(
            c::LAUNCH_DMA.method(),
            &regs_for(&base).0[..DMA_REGISTERS_NEEDED - 1],
            &mem,
            &mut PendingWrites::new(),
            &mut Recorder::default()
        ),
        Err(CopyError::Registers {
            len: DMA_REGISTERS_NEEDED - 1,
            needed: DMA_REGISTERS_NEEDED
        })
    );
    // The semaphore.
    assert_eq!(
        refused(&|r| {
            semaphore_type(
                r,
                c::LAUNCH_DMA_SEMAPHORE_TYPE_RELEASE_CONDITIONAL_INTR_SEMAPHORE,
            );
        }),
        Err(CopyError::SemaphoreType(
            c::LAUNCH_DMA_SEMAPHORE_TYPE_RELEASE_CONDITIONAL_INTR_SEMAPHORE
        ))
    );
    for float in [
        c::LAUNCH_DMA_SEMAPHORE_REDUCTION_FADD,
        c::LAUNCH_DMA_SEMAPHORE_REDUCTION_FMIN,
        c::LAUNCH_DMA_SEMAPHORE_REDUCTION_FMAX,
    ] {
        assert_eq!(
            refused(&|r| {
                semaphore_type(r, c::LAUNCH_DMA_SEMAPHORE_TYPE_RELEASE_ONE_WORD_SEMAPHORE)
                    .field(
                        c::LAUNCH_DMA,
                        c::LAUNCH_DMA_SEMAPHORE_REDUCTION_ENABLE,
                        c::LAUNCH_DMA_SEMAPHORE_REDUCTION_ENABLE_TRUE,
                    )
                    .field(c::LAUNCH_DMA, c::LAUNCH_DMA_SEMAPHORE_REDUCTION, float);
            }),
            Err(CopyError::FloatReduction(float))
        );
    }
    for invalid in [
        c::LAUNCH_DMA_SEMAPHORE_REDUCTION_INVALIDA,
        c::LAUNCH_DMA_SEMAPHORE_REDUCTION_INVALIDB,
        c::LAUNCH_DMA_SEMAPHORE_REDUCTION_INVALIDC,
        c::LAUNCH_DMA_SEMAPHORE_REDUCTION_INVALIDD,
        c::LAUNCH_DMA_SEMAPHORE_REDUCTION_INVALIDE,
    ] {
        assert_eq!(
            refused(&|r| {
                semaphore_type(r, c::LAUNCH_DMA_SEMAPHORE_TYPE_RELEASE_ONE_WORD_SEMAPHORE)
                    .field(
                        c::LAUNCH_DMA,
                        c::LAUNCH_DMA_SEMAPHORE_REDUCTION_ENABLE,
                        c::LAUNCH_DMA_SEMAPHORE_REDUCTION_ENABLE_TRUE,
                    )
                    .field(c::LAUNCH_DMA, c::LAUNCH_DMA_SEMAPHORE_REDUCTION, invalid);
            }),
            Err(CopyError::Reduction(DecodeError::UndefinedReduction {
                word: "LAUNCH_DMA",
                value: invalid
            }))
        );
    }
}

#[test]
fn address_remap_and_block_refusals_are_exact() {
    // Physical addresses, source before destination.
    assert_eq!(
        refused(&|r| {
            r.field(
                c::LAUNCH_DMA,
                c::LAUNCH_DMA_SRC_TYPE,
                c::LAUNCH_DMA_SRC_TYPE_PHYSICAL,
            )
            .field(
                c::LAUNCH_DMA,
                c::LAUNCH_DMA_DST_TYPE,
                c::LAUNCH_DMA_DST_TYPE_PHYSICAL,
            );
        }),
        Err(CopyError::Physical {
            side: CopySide::Source
        })
    );
    assert_eq!(
        refused(&|r| {
            r.field(
                c::LAUNCH_DMA,
                c::LAUNCH_DMA_DST_TYPE,
                c::LAUNCH_DMA_DST_TYPE_PHYSICAL,
            );
        }),
        Err(CopyError::Physical {
            side: CopySide::Destination
        })
    );
    // Remap: an undefined selector, and a source component past the count.
    assert_eq!(
        refused(&|r| remap(r, 1, 4, 2, [X, 7, KEEP, KEEP])),
        Err(CopyError::RemapSelector {
            component: 1,
            value: 7
        })
    );
    assert_eq!(
        refused(&|r| remap(r, 1, 2, 3, [X, Y, Z, 7])),
        Err(CopyError::RemapSource {
            component: 2,
            read: 2,
            sources: 2
        })
    );
    // A selector past NUM_DST_COMPONENTS is not read.
    assert_eq!(refused(&|r| remap(r, 1, 1, 1, [X, 7, 7, 7])), Ok(()));
    // Block fields, width before height before depth.
    assert_eq!(
        refused(&|r| {
            r.field(c::SET_DST_BLOCK_SIZE, c::SET_DST_BLOCK_SIZE_WIDTH, 1)
                .field(c::SET_DST_BLOCK_SIZE, c::SET_DST_BLOCK_SIZE_HEIGHT, 6)
                .field(c::SET_DST_BLOCK_SIZE, c::SET_DST_BLOCK_SIZE_DEPTH, 6);
        }),
        Err(CopyError::BlockWidth {
            side: CopySide::Destination,
            value: 1
        })
    );
    let top = c::SET_DST_BLOCK_SIZE_HEIGHT_THIRTYTWO_GOBS;
    assert_eq!(
        refused(&|r| {
            r.field(c::SET_DST_BLOCK_SIZE, c::SET_DST_BLOCK_SIZE_HEIGHT, top + 1)
                .field(c::SET_DST_BLOCK_SIZE, c::SET_DST_BLOCK_SIZE_DEPTH, top + 1);
        }),
        Err(CopyError::BlockHeight {
            side: CopySide::Destination,
            value: top + 1
        })
    );
    assert_eq!(
        refused(&|r| {
            r.field(c::SET_DST_BLOCK_SIZE, c::SET_DST_BLOCK_SIZE_DEPTH, top + 1);
        }),
        Err(CopyError::BlockDepth {
            side: CopySide::Destination,
            value: top + 1
        })
    );
    // The largest defined block is accepted (the region still fits).
    assert_eq!(
        refused(&|r| {
            r.field(c::SET_DST_BLOCK_SIZE, c::SET_DST_BLOCK_SIZE_HEIGHT, top)
                .field(c::SET_DST_BLOCK_SIZE, c::SET_DST_BLOCK_SIZE_DEPTH, 0);
        })
        .map_err(|e| e.to_string()),
        Ok(())
    );
    // A pitch line start below address 0, and an end past 2^64.
    assert_eq!(
        refused(&|r| {
            r.set(c::OFFSET_IN_UPPER, 0)
                .set(c::OFFSET_IN_LOWER, 0x100)
                .set(c::PITCH_IN, (-0x40i32).cast_unsigned());
        }),
        Err(CopyError::Extent {
            side: CopySide::Source
        })
    );
}

#[test]
fn error_messages_are_plain_words() {
    let cases: [(CopyError, &str); 6] = [
        (
            CopyError::Fault {
                side: CopySide::Source,
                fault: GpuFault::Unmapped { va: 0x1000 },
            },
            "the copy's source: GPU address 0x1000 is not mapped",
        ),
        (
            CopyError::Layout {
                side: CopySide::Destination,
                error: LayoutError::RegionOutOfBounds,
            },
            "the copy's destination surface: the region does not fit in the level",
        ),
        (
            CopyError::Registers { len: 0, needed: 1 },
            "the register file holds 0 registers, the engine reads 1",
        ),
        (
            CopyError::TooLarge {
                bytes: u128::from(MAX_COPY_BYTES) + 1,
                max: MAX_COPY_BYTES,
            },
            "the copy moves 268435457 bytes, more than the 268435456 allowed",
        ),
        (
            CopyError::BlockWidth {
                side: CopySide::Source,
                value: 2,
            },
            "the copy's source block width field is 2, but only one GOB (0) is defined",
        ),
        (
            CopyError::RemapSource {
                component: 0,
                read: 3,
                sources: 1,
            },
            "remap component 0 reads source component 3 of 1",
        ),
    ];
    for (error, text) in cases {
        assert_eq!(error.to_string(), text);
    }
}

#[test]
fn the_size_bound_is_checked_at_the_edge() {
    assert_eq!(check_size(u128::from(MAX_COPY_BYTES)), Ok(()));
    assert_eq!(
        check_size(u128::from(MAX_COPY_BYTES) + 1),
        Err(CopyError::TooLarge {
            bytes: u128::from(MAX_COPY_BYTES) + 1,
            max: MAX_COPY_BYTES
        })
    );
    // Through a launch: 2^14 lines of 2^14 bytes is MAX_COPY_BYTES; one byte more per line is
    // refused before anything is read (the source would fault). Remap makes the destination the
    // larger side: four 1-byte components from one.
    let mem = memory_with_hole(0);
    let mut copy = Copy {
        source: Side::Pitch {
            address: GPU_BASE,
            pitch: 0,
        },
        destination: Side::Pitch {
            address: GPU_BASE,
            pitch: 0,
        },
        length: (1 << 14) + 1,
        lines: 1 << 14,
        multi_line: true,
    };
    let refused = Err(CopyError::TooLarge {
        bytes: u128::from(MAX_COPY_BYTES) + (1 << 14),
        max: MAX_COPY_BYTES,
    });
    assert_eq!(
        launch(&mem, &regs_for(&copy), &mut Recorder::default()),
        refused
    );
    copy.length = 1 << 14;
    assert_eq!(
        launch(&mem, &regs_for(&copy), &mut Recorder::default()),
        Err(CopyError::Fault {
            side: CopySide::Source,
            fault: GpuFault::Unmapped { va: GPU_BASE }
        })
    );
    copy.length = (1 << 12) + 1;
    copy.lines = 1 << 14;
    let mut regs = regs_for(&copy);
    remap(&mut regs, 1, 1, 4, [X, X, X, X]);
    assert_eq!(
        launch(&mem, &regs, &mut Recorder::default()),
        Err(CopyError::TooLarge {
            bytes: u128::from(MAX_COPY_BYTES) + 4 * (1 << 14),
            max: MAX_COPY_BYTES
        })
    );
}

// --- Divergences from Eden-NX2 ---

#[test]
fn divergence_upper_address_bits_are_seventeen() {
    // GPU_BASE's upper word is 0x1_2345: bit 16 and bits 15:8 are set, which an 8-bit upper
    // field (Eden-NX2's `upper & 0xff`) would drop.
    assert!(hi(GPU_BASE) > 0xff);
    check_against_layout(Copy {
        source: Side::Pitch {
            address: SRC,
            pitch: 0x10,
        },
        destination: Side::Pitch {
            address: DST,
            pitch: 0x10,
        },
        length: 0x10,
        lines: 2,
        multi_line: true,
    });
    assert_eq!(c::OFFSET_IN_UPPER_UPPER, Field::new(16, 0));
    assert_eq!(c::OFFSET_OUT_UPPER_UPPER, Field::new(16, 0));
    assert_eq!(c::SET_SEMAPHORE_A_UPPER, Field::new(16, 0));
    // Bits above the field are dropped.
    let releases = semaphore_launch(|r| {
        semaphore_type(r, c::LAUNCH_DMA_SEMAPHORE_TYPE_RELEASE_ONE_WORD_SEMAPHORE)
            .set(c::SET_SEMAPHORE_A, 0xfffe_0001);
    })
    .unwrap();
    assert_eq!(releases[0].address, 0x1_6789_abc0);
}

#[test]
fn divergence_transfer_type_none_moves_nothing() {
    let mem = memory();
    let before = snapshot(&mem);
    let mut regs = regs_for(&edge_copy(None, None));
    regs.field(
        c::LAUNCH_DMA,
        c::LAUNCH_DMA_DATA_TRANSFER_TYPE,
        c::LAUNCH_DMA_DATA_TRANSFER_TYPE_NONE,
    );
    semaphore_type(
        &mut regs,
        c::LAUNCH_DMA_SEMAPHORE_TYPE_RELEASE_ONE_WORD_SEMAPHORE,
    );
    let mut sink = Recorder::default();
    launch(&mem, &regs, &mut sink).unwrap();
    assert!(snapshot(&mem) == before);
    assert_eq!(sink.events.len(), 1);
    assert_eq!(sink.releases().len(), 1);
    // Pipelined moves bytes as non-pipelined does.
    regs.field(
        c::LAUNCH_DMA,
        c::LAUNCH_DMA_DATA_TRANSFER_TYPE,
        c::LAUNCH_DMA_DATA_TRANSFER_TYPE_PIPELINED,
    );
    launch(&mem, &regs, &mut Recorder::default()).unwrap();
    assert_eq!(read(&mem, DST, 0x40), read(&mem, SRC, 0x40));
}

#[test]
fn divergence_a_one_line_copy_follows_the_layout_bits() {
    let bl = Bl {
        width: 128,
        height: 16,
        depth: 1,
        bh: 1,
        bd: 0,
    };
    check_against_layout(Copy {
        source: Side::Pitch {
            address: SRC,
            pitch: 0,
        },
        destination: Side::Bl {
            address: DST,
            bl,
            origin: (5, 11),
        },
        length: 100,
        lines: 3,
        multi_line: false,
    });
    check_against_layout(Copy {
        source: Side::Bl {
            address: SRC,
            bl,
            origin: (20, 9),
        },
        destination: Side::Pitch {
            address: DST,
            pitch: 0,
        },
        length: 100,
        lines: 3,
        multi_line: false,
    });
}

#[test]
fn divergence_remap_applies_the_selectors_on_every_path() {
    // Multi-line pitch to pitch: Eden-NX2 copies LINE_LENGTH_IN bytes unchanged.
    check_remap(1, 4, &[W, Z, Y, X]);
    // One line, multi-line clear.
    let mem = memory();
    let before = read(&mem, SRC, 8);
    let mut regs = regs_for(&Copy {
        source: Side::Pitch {
            address: SRC,
            pitch: 0,
        },
        destination: Side::Pitch {
            address: DST,
            pitch: 0,
        },
        length: 2,
        lines: 1,
        multi_line: false,
    });
    remap(&mut regs, 1, 4, 4, [W, Z, Y, X]);
    launch(&mem, &regs, &mut Recorder::default()).unwrap();
    let expected = [
        before[3], before[2], before[1], before[0], before[7], before[6], before[5], before[4],
    ];
    assert_eq!(read(&mem, DST, 8), expected);
}

#[test]
fn divergence_one_byte_constant_fill_uses_the_low_byte() {
    // Eden-NX2 would write a1 a2 a3 a4 a1 a2 (the 32-bit constant's bytes in turn, from
    // 0xa1a2a3a4 stored little-endian: a4 a3 a2 a1 ...); here each element is the low byte.
    let mem = memory();
    let mut regs = regs_for(&Copy {
        source: Side::Pitch {
            address: SRC,
            pitch: 0,
        },
        destination: Side::Pitch {
            address: DST,
            pitch: 0,
        },
        length: 6,
        lines: 1,
        multi_line: false,
    });
    remap(&mut regs, 1, 1, 1, [A, KEEP, KEEP, KEEP]);
    launch(&mem, &regs, &mut Recorder::default()).unwrap();
    assert_eq!(read(&mem, DST, 6), vec![0xa4; 6]);
}

#[test]
fn divergence_a_region_past_the_surface_is_refused() {
    let mem = memory();
    let bl = Bl {
        width: 64,
        height: 16,
        depth: 2,
        bh: 1,
        bd: 0,
    };
    let copy = |length: u32, lines: u32, origin: (u32, u32)| Copy {
        source: Side::Pitch {
            address: SRC,
            pitch: 64,
        },
        destination: Side::Bl {
            address: DST,
            bl,
            origin,
        },
        length,
        lines,
        multi_line: true,
    };
    let out = Err(CopyError::Layout {
        side: CopySide::Destination,
        error: LayoutError::RegionOutOfBounds,
    });
    // x: the end exactly at the width fits, one past does not.
    assert_eq!(
        launch(
            &mem,
            &regs_for(&copy(60, 1, (4, 0))),
            &mut Recorder::default()
        ),
        Ok(())
    );
    assert_eq!(
        launch(
            &mem,
            &regs_for(&copy(61, 1, (4, 0))),
            &mut Recorder::default()
        ),
        out
    );
    // Lines: 12 rows from row 4, twice (two slices), fit; one more line does not.
    assert_eq!(
        launch(
            &mem,
            &regs_for(&copy(8, 24, (0, 4))),
            &mut Recorder::default()
        ),
        Ok(())
    );
    assert_eq!(
        launch(
            &mem,
            &regs_for(&copy(8, 25, (0, 4))),
            &mut Recorder::default()
        ),
        out
    );
    // An origin row at the height.
    assert_eq!(
        launch(
            &mem,
            &regs_for(&copy(8, 1, (0, 16))),
            &mut Recorder::default()
        ),
        out
    );
    assert_eq!(
        launch(
            &mem,
            &regs_for(&copy(8, 1, (0, 15))),
            &mut Recorder::default()
        ),
        Ok(())
    );
}

#[test]
fn divergence_slices_keep_the_pitch_stride() {
    // Origin row 6 of 8: two lines per slice. Line 2 (slice 1, row 6) reads the pitch side's
    // line 2, at 2 * pitch, not Eden-NX2's 1 * pitch * 8 + 0.
    let bl = Bl {
        width: 64,
        height: 8,
        depth: 2,
        bh: 0,
        bd: 0,
    };
    let mem = memory();
    let copy = Copy {
        source: Side::Pitch {
            address: SRC,
            pitch: 0x40,
        },
        destination: Side::Bl {
            address: DST,
            bl,
            origin: (0, 6),
        },
        length: 0x40,
        lines: 3,
        multi_line: true,
    };
    let line2 = read(&mem, SRC + 2 * 0x40, 0x40);
    launch(&mem, &regs_for(&copy), &mut Recorder::default()).unwrap();
    let tiled = read(&mem, DST, bl.size());
    let offset = us(starquake_layout::block_linear_offset(
        &bl.tiling(),
        bl.extent(),
        Offset4D::new(0, 6, 1, 0),
    )
    .unwrap());
    assert_eq!(&tiled[offset..offset + 16], &line2[..16]);
}

#[test]
fn divergence_a_negative_pitch_walks_down_on_every_path() {
    let bl = Bl {
        width: 128,
        height: 16,
        depth: 1,
        bh: 1,
        bd: 0,
    };
    check_against_layout(Copy {
        source: Side::Pitch {
            address: SRC + 0x1000,
            pitch: -0x90,
        },
        destination: Side::Bl {
            address: DST,
            bl,
            origin: (1, 1),
        },
        length: 0x70,
        lines: 9,
        multi_line: true,
    });
    check_against_layout(Copy {
        source: Side::Bl {
            address: SRC,
            bl,
            origin: (1, 1),
        },
        destination: Side::Pitch {
            address: DST + 0x1000,
            pitch: -0x90,
        },
        length: 0x70,
        lines: 9,
        multi_line: true,
    });
}

#[test]
fn fields_that_are_not_read_change_nothing() {
    let copy = Copy {
        source: Side::Bl {
            address: SRC,
            bl: Bl {
                width: 128,
                height: 16,
                depth: 1,
                bh: 1,
                bd: 0,
            },
            origin: (2, 3),
        },
        destination: Side::Pitch {
            address: DST,
            pitch: 0x80,
        },
        length: 0x50,
        lines: 5,
        multi_line: true,
    };
    let plain = memory();
    launch(&plain, &regs_for(&copy), &mut Recorder::default()).unwrap();
    let noisy = memory();
    let mut regs = regs_for(&copy);
    regs.field(
        c::LAUNCH_DMA,
        c::LAUNCH_DMA_INTERRUPT_TYPE,
        c::LAUNCH_DMA_INTERRUPT_TYPE_BLOCKING,
    )
    .field(
        c::LAUNCH_DMA,
        c::LAUNCH_DMA_FLUSH_ENABLE,
        c::LAUNCH_DMA_FLUSH_ENABLE_TRUE,
    )
    .set(c::SET_SRC_LAYER, 3)
    .set(c::SET_DST_LAYER, 4)
    .field(
        c::SET_SRC_BLOCK_SIZE,
        c::SET_SRC_BLOCK_SIZE_GOB_HEIGHT,
        c::SET_SRC_BLOCK_SIZE_GOB_HEIGHT_GOB_HEIGHT_FERMI_8,
    )
    .set(c::SET_SRC_PHYS_MODE, 1)
    .set(c::SET_DST_PHYS_MODE, 1)
    .set(c::SET_RENDER_ENABLE_C, 0);
    // The 32-bit origins sit past the methods a launch reads; widen the file to hold them.
    regs.0.resize(c::DST_ORIGIN_Y.method() as usize + 1, 0);
    regs.set(c::SRC_ORIGIN_X, 9)
        .set(c::SRC_ORIGIN_Y, 9)
        .set(c::DST_ORIGIN_X, 9)
        .set(c::DST_ORIGIN_Y, 9);
    launch(&noisy, &regs, &mut Recorder::default()).unwrap();
    assert!(snapshot(&plain) == snapshot(&noisy));
}

#[test]
fn dma_registers_needed_covers_every_method_read() {
    // The highest method a launch reads is SET_SRC_ORIGIN (byte 0x73C).
    assert_eq!(DMA_REGISTERS_NEEDED, 0x73c / 4 + 1);
    assert_eq!(
        c::SET_SRC_ORIGIN.method() as usize + 1,
        DMA_REGISTERS_NEEDED
    );
}

/// The header's byte addresses and field positions, written out (`clc7b5.h`), and the register
/// positions Eden-NX2 asserts (`maxwell_dma.h`, `ASSERT_REG_POSITION`), which agree.
#[test]
fn constants_match_the_header_written_out() {
    let methods = [
        (c::SET_SEMAPHORE_A, 0x240),
        (c::SET_SEMAPHORE_B, 0x244),
        (c::SET_SEMAPHORE_PAYLOAD, 0x248),
        (c::SET_SEMAPHORE_PAYLOAD_UPPER, 0x24c),
        (c::LAUNCH_DMA, 0x300),
        (c::OFFSET_IN_UPPER, 0x400),
        (c::OFFSET_IN_LOWER, 0x404),
        (c::OFFSET_OUT_UPPER, 0x408),
        (c::OFFSET_OUT_LOWER, 0x40c),
        (c::PITCH_IN, 0x410),
        (c::PITCH_OUT, 0x414),
        (c::LINE_LENGTH_IN, 0x418),
        (c::LINE_COUNT, 0x41c),
        (c::SET_REMAP_CONST_A, 0x700),
        (c::SET_REMAP_CONST_B, 0x704),
        (c::SET_REMAP_COMPONENTS, 0x708),
        (c::SET_DST_BLOCK_SIZE, 0x70c),
        (c::SET_DST_WIDTH, 0x710),
        (c::SET_DST_HEIGHT, 0x714),
        (c::SET_DST_DEPTH, 0x718),
        (c::SET_DST_LAYER, 0x71c),
        (c::SET_DST_ORIGIN, 0x720),
        (c::SET_SRC_BLOCK_SIZE, 0x728),
        (c::SET_SRC_WIDTH, 0x72c),
        (c::SET_SRC_HEIGHT, 0x730),
        (c::SET_SRC_DEPTH, 0x734),
        (c::SET_SRC_LAYER, 0x738),
        (c::SET_SRC_ORIGIN, 0x73c),
    ];
    for (method, byte) in methods {
        assert_eq!(method.byte_address, byte, "{}", method.name);
    }
    let fields = [
        (c::LAUNCH_DMA_DATA_TRANSFER_TYPE, (1, 0)),
        (c::LAUNCH_DMA_SEMAPHORE_TYPE, (4, 3)),
        (c::LAUNCH_DMA_INTERRUPT_TYPE, (6, 5)),
        (c::LAUNCH_DMA_SRC_MEMORY_LAYOUT, (7, 7)),
        (c::LAUNCH_DMA_DST_MEMORY_LAYOUT, (8, 8)),
        (c::LAUNCH_DMA_MULTI_LINE_ENABLE, (9, 9)),
        (c::LAUNCH_DMA_REMAP_ENABLE, (10, 10)),
        (c::LAUNCH_DMA_SRC_TYPE, (12, 12)),
        (c::LAUNCH_DMA_DST_TYPE, (13, 13)),
        (c::LAUNCH_DMA_SEMAPHORE_REDUCTION, (17, 14)),
        (c::LAUNCH_DMA_SEMAPHORE_REDUCTION_SIGN, (18, 18)),
        (c::LAUNCH_DMA_SEMAPHORE_REDUCTION_ENABLE, (19, 19)),
        (c::LAUNCH_DMA_SEMAPHORE_PAYLOAD_SIZE, (27, 27)),
        (c::SET_REMAP_COMPONENTS_DST_X, (2, 0)),
        (c::SET_REMAP_COMPONENTS_DST_Y, (6, 4)),
        (c::SET_REMAP_COMPONENTS_DST_Z, (10, 8)),
        (c::SET_REMAP_COMPONENTS_DST_W, (14, 12)),
        (c::SET_REMAP_COMPONENTS_COMPONENT_SIZE, (17, 16)),
        (c::SET_REMAP_COMPONENTS_NUM_SRC_COMPONENTS, (21, 20)),
        (c::SET_REMAP_COMPONENTS_NUM_DST_COMPONENTS, (25, 24)),
        (c::SET_DST_BLOCK_SIZE_WIDTH, (3, 0)),
        (c::SET_DST_BLOCK_SIZE_HEIGHT, (7, 4)),
        (c::SET_DST_BLOCK_SIZE_DEPTH, (11, 8)),
        (c::SET_DST_ORIGIN_X, (15, 0)),
        (c::SET_DST_ORIGIN_Y, (31, 16)),
    ];
    for (field, (high, low)) in fields {
        assert_eq!(field, Field::new(high, low));
    }
}

/// The values the module matches on, written out from `clc7b5.h`.
#[test]
fn values_match_the_header_written_out() {
    assert_eq!(
        [
            c::LAUNCH_DMA_SEMAPHORE_TYPE_NONE,
            c::LAUNCH_DMA_SEMAPHORE_TYPE_RELEASE_ONE_WORD_SEMAPHORE,
            c::LAUNCH_DMA_SEMAPHORE_TYPE_RELEASE_FOUR_WORD_SEMAPHORE,
            c::LAUNCH_DMA_SEMAPHORE_TYPE_RELEASE_CONDITIONAL_INTR_SEMAPHORE,
        ],
        [0, 1, 2, 3]
    );
    assert_eq!(c::LAUNCH_DMA_SRC_MEMORY_LAYOUT_PITCH, 1);
    assert_eq!(c::LAUNCH_DMA_DST_MEMORY_LAYOUT_PITCH, 1);
    assert_eq!(c::LAUNCH_DMA_DATA_TRANSFER_TYPE_NONE, 0);
    assert_eq!(
        [
            c::LAUNCH_DMA_SEMAPHORE_REDUCTION_FADD,
            c::LAUNCH_DMA_SEMAPHORE_REDUCTION_FMIN,
            c::LAUNCH_DMA_SEMAPHORE_REDUCTION_FMAX
        ],
        [0xa, 0xb, 0xc]
    );
    assert_eq!(c::LAUNCH_DMA_SEMAPHORE_PAYLOAD_SIZE_TWO_WORD, 1);
}

/// The remap selector values: `DST_Y`, `_Z` and `_W` name the same values as `DST_X`
/// (`clc7b5.h`).
#[test]
fn remap_selector_values_match_the_header_written_out() {
    let x = [
        c::SET_REMAP_COMPONENTS_DST_X_SRC_X,
        c::SET_REMAP_COMPONENTS_DST_X_SRC_Y,
        c::SET_REMAP_COMPONENTS_DST_X_SRC_Z,
        c::SET_REMAP_COMPONENTS_DST_X_SRC_W,
        c::SET_REMAP_COMPONENTS_DST_X_CONST_A,
        c::SET_REMAP_COMPONENTS_DST_X_CONST_B,
        c::SET_REMAP_COMPONENTS_DST_X_NO_WRITE,
    ];
    assert_eq!(x, [0, 1, 2, 3, 4, 5, 6]);
    for other in [
        [
            c::SET_REMAP_COMPONENTS_DST_Y_SRC_X,
            c::SET_REMAP_COMPONENTS_DST_Y_SRC_Y,
            c::SET_REMAP_COMPONENTS_DST_Y_SRC_Z,
            c::SET_REMAP_COMPONENTS_DST_Y_SRC_W,
            c::SET_REMAP_COMPONENTS_DST_Y_CONST_A,
            c::SET_REMAP_COMPONENTS_DST_Y_CONST_B,
            c::SET_REMAP_COMPONENTS_DST_Y_NO_WRITE,
        ],
        [
            c::SET_REMAP_COMPONENTS_DST_Z_SRC_X,
            c::SET_REMAP_COMPONENTS_DST_Z_SRC_Y,
            c::SET_REMAP_COMPONENTS_DST_Z_SRC_Z,
            c::SET_REMAP_COMPONENTS_DST_Z_SRC_W,
            c::SET_REMAP_COMPONENTS_DST_Z_CONST_A,
            c::SET_REMAP_COMPONENTS_DST_Z_CONST_B,
            c::SET_REMAP_COMPONENTS_DST_Z_NO_WRITE,
        ],
        [
            c::SET_REMAP_COMPONENTS_DST_W_SRC_X,
            c::SET_REMAP_COMPONENTS_DST_W_SRC_Y,
            c::SET_REMAP_COMPONENTS_DST_W_SRC_Z,
            c::SET_REMAP_COMPONENTS_DST_W_SRC_W,
            c::SET_REMAP_COMPONENTS_DST_W_CONST_A,
            c::SET_REMAP_COMPONENTS_DST_W_CONST_B,
            c::SET_REMAP_COMPONENTS_DST_W_NO_WRITE,
        ],
    ] {
        assert_eq!(other, x);
    }
}

#[test]
fn a_destination_line_crossing_into_the_hole_writes_nothing() {
    // Three lines, the last starting 0x20 before the hole: its run crosses a page into it. The
    // probe reads the first byte of each page a run reaches, so the hole is found before line 0
    // is written.
    let mem = memory_with_hole(100);
    let copy = Copy {
        source: Side::Pitch {
            address: SRC,
            pitch: 0x40,
        },
        destination: Side::Pitch {
            address: HOLE - 0x20 - 2 * 0x100,
            pitch: 0x100,
        },
        length: 0x40,
        lines: 3,
        multi_line: true,
    };
    let before = snapshot_with_hole(&mem);
    assert_eq!(
        launch(&mem, &regs_for(&copy), &mut Recorder::default()),
        Err(CopyError::Fault {
            side: CopySide::Destination,
            fault: GpuFault::Unmapped { va: HOLE },
        })
    );
    assert!(snapshot_with_hole(&mem) == before);
}

/// The mapped bytes of a `memory_with_hole(100)`.
fn snapshot_with_hole(mem: &Memory) -> Vec<u8> {
    [
        read(mem, GPU_BASE, 100 * PAGE_SIZE),
        read(mem, HOLE + PAGE_SIZE, (PAGES - 101) * PAGE_SIZE),
    ]
    .concat()
}

// --- Overlapping sides ---

#[test]
fn an_overlapping_one_line_copy_longer_than_a_chunk_reads_the_whole_line_first() {
    // One and a half chunks, the destination one byte past the source: the second chunk's
    // source starts inside the first chunk's destination, so streaming chunk by chunk would
    // copy one byte the first chunk had already overwritten.
    let length = super::super::dma::CHUNK_BYTES * 3 / 2;
    let mem = memory();
    let before = snapshot(&mem);
    let old = read(&mem, SRC, length);
    let copy = Copy {
        source: Side::Pitch {
            address: SRC,
            pitch: 0,
        },
        destination: Side::Pitch {
            address: SRC + 1,
            pitch: 0,
        },
        length: u32::try_from(length).unwrap(),
        lines: 1,
        multi_line: false,
    };
    launch(&mem, &regs_for(&copy), &mut Recorder::default()).unwrap();
    let mut expected = before;
    let at = us(SRC + 1 - GPU_BASE);
    expected[at..at + us(length)].copy_from_slice(&old);
    assert!(snapshot(&mem) == expected);
}

#[test]
fn overlapping_multi_line_pitch_copies_read_each_line_before_writing_it() {
    // Each destination line is the next source line: as Eden-NX2's per-line `CopyBlock`, line
    // 1's source already holds line 0's copy when it is read, so every line becomes old line 0
    // (reading the whole source first would give old lines 0 to 3 instead).
    let mem = memory();
    let before = snapshot(&mem);
    let old_line0 = read(&mem, SRC, 0x100);
    let copy = Copy {
        source: Side::Pitch {
            address: SRC,
            pitch: 0x100,
        },
        destination: Side::Pitch {
            address: SRC + 0x100,
            pitch: 0x100,
        },
        length: 0x100,
        lines: 4,
        multi_line: true,
    };
    launch(&mem, &regs_for(&copy), &mut Recorder::default()).unwrap();
    let mut expected = before;
    for line in 1..=4 {
        let at = us(SRC - GPU_BASE) + line * 0x100;
        expected[at..at + 0x100].copy_from_slice(&old_line0);
    }
    assert!(snapshot(&mem) == expected);
}

#[test]
fn overlapping_copies_with_a_block_linear_side_read_the_whole_source_first() {
    // `check_against_layout` reads the source before the launch and expects it written whole:
    // Eden-NX2's model for both block-linear paths, with each side on the other's bytes.
    let bl = Bl {
        width: 64,
        height: 32,
        depth: 1,
        bh: 1,
        bd: 0,
    };
    check_against_layout(Copy {
        source: Side::Pitch {
            address: DST,
            pitch: 64,
        },
        destination: Side::Bl {
            address: DST,
            bl,
            origin: (0, 0),
        },
        length: 64,
        lines: 32,
        multi_line: true,
    });
    check_against_layout(Copy {
        source: Side::Bl {
            address: DST,
            bl,
            origin: (0, 0),
        },
        destination: Side::Pitch {
            address: DST + 0x10,
            pitch: 64,
        },
        length: 64,
        lines: 32,
        multi_line: true,
    });
    check_against_layout(Copy {
        source: Side::Bl {
            address: DST,
            bl,
            origin: (0, 0),
        },
        destination: Side::Bl {
            address: DST,
            bl,
            origin: (16, 3),
        },
        length: 48,
        lines: 29,
        multi_line: true,
    });
}

#[test]
fn a_launch_with_a_short_register_file_still_flushes_the_pending_writes() {
    let mem = memory();
    let before = snapshot(&mem);
    let mut pending = PendingWrites::new();
    pending.add(0x10..0x20);
    let mut sink = Recorder::default();
    assert_eq!(
        dma_method(
            c::LAUNCH_DMA.method(),
            &vec![0; DMA_REGISTERS_NEEDED - 1],
            &mem,
            &mut pending,
            &mut sink
        ),
        Err(CopyError::Registers {
            len: DMA_REGISTERS_NEEDED - 1,
            needed: DMA_REGISTERS_NEEDED
        })
    );
    assert_eq!(sink.events, vec![Event::Invalidate(0x10..0x20)]);
    assert!(pending.is_empty());
    assert!(snapshot(&mem) == before);
}
