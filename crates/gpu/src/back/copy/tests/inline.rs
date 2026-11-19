//! Inline-to-memory in all three classes: writes in order, pitch and block-linear destinations
//! against the layout crate, the completion semaphore, faults, refusals and divergences.

use starquake_layout::LayoutError;
use starquake_memory::PAGE_SIZE;

use super::super::surface::CopySide;
use super::super::{
    CopyError, INLINE_A140, INLINE_C797, INLINE_C7C0, InlineClass, InlineUpload, PendingWrites,
    dma_method,
};
use super::*;
use crate::classes::{cla140, clc797, clc7b5, clc7c0};
use crate::memory::GpuFault;
use crate::ring::ReleaseOp;
use crate::sema::{DecodeError, PayloadSize, Reduction, ReductionFormat, ReductionOp, ReleaseForm};

const CLASSES: [InlineClass; 3] = [INLINE_A140, INLINE_C797, INLINE_C7C0];

/// Low enough for every class's upper address field.
const DST: u64 = LOW_BASE + 0x4_0000;

/// A destination for a test upload.
#[derive(Clone, Copy, Debug)]
enum Dest {
    Pitch { address: u64, pitch: u32 },
    Bl { address: u64, bl: Bl, origin: (u32, u32) },
}

/// An engine with its register file and pending list.
struct Rig {
    class: InlineClass,
    regs: Regs,
    engine: InlineUpload,
    pending: PendingWrites,
    sink: Recorder,
}

impl Rig {
    fn new(class: InlineClass) -> Self {
        Self {
            class,
            regs: Regs::new(class.registers_needed()),
            engine: InlineUpload::new(),
            pending: PendingWrites::new(),
            sink: Recorder::default(),
        }
    }

    /// Sets up a transfer of `lines` lines of `length` bytes to `dest` (without launching).
    fn configure(&mut self, dest: Dest, length: u32, lines: u32) -> &mut Self {
        let k = self.class;
        self.regs.set(k.line_length_in, length).set(k.line_count, lines);
        let address = match dest {
            Dest::Pitch { address, pitch } => {
                self.regs.set(k.pitch_out, pitch).field(k.launch, k.layout, k.layout_pitch);
                address
            }
            Dest::Bl { address, bl, origin } => {
                self.regs
                    .field(k.launch, k.layout, 1 - k.layout_pitch)
                    .field(k.block_size, k.block_height, bl.bh.into())
                    .field(k.block_size, k.block_depth, bl.bd.into())
                    .set(k.width, bl.width)
                    .set(k.height, bl.height)
                    .set(k.depth, bl.depth)
                    .field(k.origin_x, k.origin_x_value, origin.0)
                    .field(k.origin_y, k.origin_y_value, origin.1);
                address
            }
        };
        self.regs.set(k.offset_out_upper, hi(address)).set(k.offset_out, lo(address));
        self
    }

    fn method(&mut self, mem: &Memory, method: Method) -> Result<(), CopyError> {
        let class = self.class;
        self.engine.method(
            &class,
            method.method(),
            &self.regs.0,
            mem,
            &mut self.pending,
            &mut self.sink,
        )
    }

    fn launch(&mut self, mem: &Memory) -> Result<(), CopyError> {
        self.method(mem, self.class.launch)
    }

    fn word(&mut self, mem: &Memory, word: u32) -> Result<(), CopyError> {
        self.regs.set(self.class.data, word);
        self.method(mem, self.class.data)
    }

    /// Sends `bytes` as little-endian words, the last padded with 0xee.
    fn send(&mut self, mem: &Memory, bytes: &[u8]) {
        for chunk in bytes.chunks(4) {
            let mut word = [0xee; 4];
            word[..chunk.len()].copy_from_slice(chunk);
            self.word(mem, u32::from_le_bytes(word)).unwrap();
        }
    }
}

/// Data bytes distinct from the memory pattern.
fn data(len: usize) -> Vec<u8> {
    (0..len).map(|i| (i as u8).wrapping_mul(13).wrapping_add(0x5b)).collect()
}

#[test]
fn each_word_is_written_as_it_arrives() {
    for class in CLASSES {
        let mem = memory();
        let before = snapshot(&mem);
        let mut rig = Rig::new(class);
        rig.configure(Dest::Pitch { address: DST + 2, pitch: 0x20 }, 10, 3).launch(&mem).unwrap();
        let bytes = data(30);
        let mut expected = before.clone();
        for (n, chunk) in bytes.chunks(4).enumerate() {
            rig.send(&mem, chunk);
            for (i, byte) in chunk.iter().enumerate() {
                let k = n * 4 + i;
                let at = (DST + 2 - LOW_BASE) as usize + (k / 10) * 0x20 + k % 10;
                expected[at] = *byte;
            }
            assert!(snapshot(&mem) == expected, "after word {n}");
        }
        assert_eq!(rig.engine.remaining(), 0);
    }
}

#[test]
fn a_block_linear_destination_matches_the_layout_crates_tile() {
    for class in CLASSES {
        let mem = memory();
        let bl = Bl { width: 200, height: 40, depth: 1, bh: 2, bd: 0 };
        let mut rig = Rig::new(class);
        rig.configure(Dest::Bl { address: DST, bl, origin: (5, 3) }, 37, 5).launch(&mem).unwrap();
        let mut tiled = read(&mem, DST, bl.size());
        let bytes = data(37 * 5);
        rig.send(&mem, &bytes);
        bl.tile(&mut tiled, &bytes, 37, (5, 3), 37, 5);
        assert_eq!(read(&mem, DST, bl.size()), tiled, "{}", class.launch_word);
    }
}

#[test]
fn completion_semaphore_gives_the_exact_release() {
    for class in CLASSES {
        let k = class;
        let semaphore = 0x0012_3456_789a_bcd0u64 & ((u64::from(k.semaphore_a_upper.max()) << 32) | 0xffff_ffff);
        let cases = [
            // (struct size, reduction enable, op, format, expected op, timestamp)
            (k.struct_four_words, false, 0, 0, ReleaseOp::Write, true),
            (1 - k.struct_four_words, false, 0, 0, ReleaseOp::Write, false),
            (
                1 - k.struct_four_words,
                true,
                sema_red::RED_MAX,
                sema_red::RED_FORMAT_SIGNED,
                ReleaseOp::Reduce(Reduction { op: ReductionOp::Max, format: ReductionFormat::Signed }),
                false,
            ),
            (
                k.struct_four_words,
                true,
                sema_red::RED_INC,
                sema_red::RED_FORMAT_UNSIGNED,
                ReleaseOp::Reduce(Reduction { op: ReductionOp::Inc, format: ReductionFormat::Unsigned }),
                true,
            ),
        ];
        for (struct_size, enable, op, format, expected, timestamp) in cases {
            let mem = memory();
            let mut rig = Rig::new(class);
            rig.configure(Dest::Pitch { address: DST, pitch: 8 }, 6, 2)
                .regs
                .field(k.launch, k.completion, k.completion_release)
                .field(k.launch, k.struct_size, struct_size)
                .field(k.launch, k.reduction_enable, u32::from(enable))
                .field(k.launch, k.reduction_op, op)
                .field(k.launch, k.reduction_format, format)
                .set(k.semaphore_a, hi(semaphore))
                .set(k.semaphore_b, lo(semaphore))
                .set(k.semaphore_c, 0xc0ff_ee11);
            rig.launch(&mem).unwrap();
            rig.send(&mem, &data(8));
            assert_eq!(rig.sink.releases(), vec![], "released before the last word");
            rig.send(&mem, &data(4));
            assert_eq!(
                rig.sink.releases(),
                vec![Release {
                    address: semaphore,
                    payload: 0xc0ff_ee11,
                    form: ReleaseForm { size: PayloadSize::Four, timestamp },
                    op: expected,
                }],
                "{}",
                k.launch_word
            );
        }
    }
}

mod sema_red {
    pub(super) use crate::sema::{RED_FORMAT_SIGNED, RED_FORMAT_UNSIGNED, RED_INC, RED_MAX};
}

#[test]
fn completion_types_other_than_release_release_nothing() {
    let k = INLINE_A140;
    for completion in [
        cla140::LAUNCH_DMA_COMPLETION_TYPE_FLUSH_DISABLE,
        cla140::LAUNCH_DMA_COMPLETION_TYPE_FLUSH_ONLY,
    ] {
        let mem = memory();
        let mut rig = Rig::new(k);
        rig.configure(Dest::Pitch { address: DST, pitch: 4 }, 4, 1)
            .regs
            .field(k.launch, k.completion, completion);
        rig.launch(&mem).unwrap();
        rig.send(&mem, &data(4));
        assert_eq!(rig.sink.releases(), vec![]);
    }
}

#[test]
fn an_undefined_reduction_format_refuses_the_launch() {
    let k = INLINE_C7C0;
    let mem = memory();
    let mut rig = Rig::new(k);
    rig.configure(Dest::Pitch { address: DST, pitch: 4 }, 4, 1)
        .regs
        .field(k.launch, k.completion, k.completion_release)
        .field(k.launch, k.reduction_enable, k.reduction_enable_true)
        .field(k.launch, k.reduction_format, 2);
    assert_eq!(
        rig.launch(&mem),
        Err(CopyError::Reduction(DecodeError::UndefinedFormat { word: "C7C0 LAUNCH_DMA", value: 2 }))
    );
    assert_eq!(rig.engine.remaining(), 0);
}

#[test]
fn a_launch_of_zero_bytes_completes_and_releases_at_once() {
    let k = INLINE_A140;
    let mem = memory();
    let mut rig = Rig::new(k);
    rig.configure(Dest::Pitch { address: DST, pitch: 4 }, 0, 5)
        .regs
        .field(k.launch, k.completion, k.completion_release)
        .set(k.semaphore_b, 0x80);
    rig.launch(&mem).unwrap();
    assert_eq!(rig.sink.releases().len(), 1);
    assert_eq!(rig.engine.remaining(), 0);
    // No make_current for an empty region.
    assert_eq!(rig.sink.events.len(), 1);
}

#[test]
fn a_launch_reports_the_destination_cover_before_any_word() {
    let mem = memory();
    let mut rig = Rig::new(INLINE_C797);
    rig.configure(Dest::Pitch { address: DST + 4, pitch: 0x30 }, 0x10, 3).launch(&mem).unwrap();
    assert_eq!(rig.sink.events, vec![Event::MakeCurrent(DST + 4..DST + 4 + 0x60 + 0x10)]);
}

#[test]
fn a_word_split_across_lines_lands_on_both() {
    let mem = memory();
    let mut rig = Rig::new(INLINE_A140);
    rig.configure(Dest::Pitch { address: DST, pitch: 0x100 }, 3, 3).launch(&mem).unwrap();
    rig.word(&mem, 0x4433_2211).unwrap();
    assert_eq!(read(&mem, DST, 3), vec![0x11, 0x22, 0x33]);
    assert_eq!(read(&mem, DST + 0x100, 1), vec![0x44]);
    assert_eq!(rig.engine.remaining(), 5);
}

#[test]
fn inline_writes_merge_until_the_next_copy_flushes_them() {
    let mem = memory();
    let mut rig = Rig::new(INLINE_C7C0);
    rig.configure(Dest::Pitch { address: DST, pitch: 0x100 }, 8, 2).launch(&mem).unwrap();
    rig.send(&mem, &data(16));
    // Words 0 and 1 merge into line 0, words 2 and 3 into line 1.
    assert_eq!(rig.pending.ranges(), vec![DST..DST + 8, DST + 0x100..DST + 0x108]);
    // A DMA launch hands them over first.
    let mut dma = Regs::new(super::super::DMA_REGISTERS_NEEDED);
    dma.set(clc7b5::LAUNCH_DMA, 0);
    let mut sink = Recorder::default();
    dma_method(clc7b5::LAUNCH_DMA.method(), &dma.0, &mem, &mut rig.pending, &mut sink).unwrap();
    assert!(rig.pending.is_empty());
    assert_eq!(
        sink.events,
        vec![Event::Invalidate(DST..DST + 8), Event::Invalidate(DST + 0x100..DST + 0x108)]
    );
}

#[test]
fn a_word_that_faults_keeps_the_words_before_it() {
    let hole = GPU_BASE + 100 * PAGE_SIZE;
    let k = INLINE_A140;
    let mem = memory_with_hole(100);
    let mut rig = Rig::new(k);
    rig.configure(Dest::Pitch { address: hole - 8, pitch: 0 }, 16, 1)
        .regs
        .field(k.launch, k.completion, k.completion_release);
    rig.launch(&mem).unwrap();
    rig.word(&mem, 0x0403_0201).unwrap();
    rig.word(&mem, 0x0807_0605).unwrap();
    assert_eq!(
        rig.word(&mem, 0x0c0b_0a09),
        Err(CopyError::Fault { side: CopySide::Destination, fault: GpuFault::Unmapped { va: hole } })
    );
    assert_eq!(read(&mem, hole - 8, 8), vec![1, 2, 3, 4, 5, 6, 7, 8]);
    // The transfer ended: later words are ignored and nothing is released.
    assert_eq!(rig.engine.remaining(), 0);
    assert_eq!(rig.word(&mem, 0xffff_ffff), Ok(()));
    assert_eq!(rig.sink.releases(), vec![]);
    // The engine takes a new launch.
    rig.configure(Dest::Pitch { address: hole - 8, pitch: 0 }, 4, 1);
    rig.launch(&mem).unwrap();
    rig.word(&mem, 0xaabb_ccdd).unwrap();
    assert_eq!(read(&mem, hole - 8, 4), vec![0xdd, 0xcc, 0xbb, 0xaa]);
    assert_eq!(rig.sink.releases().len(), 1);
}

#[test]
fn a_word_straddling_the_hole_writes_none_of_its_bytes() {
    let hole = GPU_BASE + 100 * PAGE_SIZE;
    let mem = memory_with_hole(100);
    let mut rig = Rig::new(INLINE_A140);
    rig.configure(Dest::Pitch { address: hole - 2, pitch: 0 }, 4, 1).launch(&mem).unwrap();
    let before = read(&mem, hole - 2, 2);
    assert_eq!(
        rig.word(&mem, 0x0403_0201),
        Err(CopyError::Fault { side: CopySide::Destination, fault: GpuFault::Unmapped { va: hole } })
    );
    assert_eq!(read(&mem, hole - 2, 2), before);
    assert!(rig.pending.is_empty());
}

#[test]
fn a_launch_abandons_the_transfer_in_progress() {
    let mem = memory();
    let k = INLINE_C797;
    let mut rig = Rig::new(k);
    rig.configure(Dest::Pitch { address: DST, pitch: 0 }, 8, 1)
        .regs
        .field(k.launch, k.completion, k.completion_release);
    rig.launch(&mem).unwrap();
    rig.word(&mem, 1).unwrap();
    rig.configure(Dest::Pitch { address: DST + 0x100, pitch: 0 }, 4, 1);
    rig.launch(&mem).unwrap();
    assert_eq!(rig.engine.remaining(), 4);
    rig.word(&mem, 0x0d0c_0b0a).unwrap();
    assert_eq!(read(&mem, DST + 0x100, 4), vec![0x0a, 0x0b, 0x0c, 0x0d]);
    assert_eq!(rig.sink.releases().len(), 1);
    // A launch that fails also abandons it.
    rig.configure(Dest::Pitch { address: DST, pitch: 0 }, 8, 1).launch(&mem).unwrap();
    rig.regs.field(k.block_size, k.block_width, 1).field(k.launch, k.layout, 1 - k.layout_pitch);
    assert!(rig.launch(&mem).is_err());
    assert_eq!(rig.engine.remaining(), 0);
}

#[test]
fn divergence_upper_address_bits_follow_each_class() {
    // Garbage above each class's field: A140 keeps 25 bits, C7C0 17, C797 8.
    // The target's upper word, 0x1_2345, fits 25 and 17 bits but not 8.
    let target = GPU_BASE + 0x4_0000;
    for (class, width) in [(INLINE_A140, 25), (INLINE_C7C0, 17), (INLINE_C797, 8)] {
        assert_eq!(class.offset_out_upper_value, Field::new(width - 1, 0));
        let mem = memory();
        let mut rig = Rig::new(class);
        rig.configure(Dest::Pitch { address: target, pitch: 0 }, 4, 1);
        rig.regs.set(class.offset_out_upper, hi(target) | !class.offset_out_upper_value.max());
        rig.launch(&mem).unwrap();
        rig.word(&mem, 0x0403_0201).unwrap();
        // The word lands where the class's field puts it: for C797 the low alias of the guest
        // memory, at upper word 0x45.
        let kept = u64::from(hi(target) & class.offset_out_upper_value.max());
        let landed = (kept << 32) | u64::from(lo(target));
        assert_eq!(landed == target, width > 8, "{}", class.launch_word);
        assert_eq!(read(&mem, landed, 4), vec![1, 2, 3, 4], "{}", class.launch_word);
    }
}

#[test]
fn divergence_bad_destinations_are_refused() {
    let k = INLINE_A140;
    let mem = memory();
    let bl = Bl { width: 64, height: 16, depth: 1, bh: 1, bd: 0 };
    let refused = |f: &dyn Fn(&mut Regs), length: u32, lines: u32| {
        let mut rig = Rig::new(k);
        rig.configure(Dest::Bl { address: DST, bl, origin: (4, 2) }, length, lines);
        f(&mut rig.regs);
        let result = rig.launch(&mem);
        if result.is_err() {
            assert_eq!(rig.sink.events, vec![]);
            assert_eq!(rig.engine.remaining(), 0);
        }
        result
    };
    assert_eq!(refused(&|_| (), 60, 14), Ok(()));
    assert_eq!(
        refused(&|_| (), 61, 1),
        Err(CopyError::Layout { side: CopySide::Destination, error: LayoutError::RegionOutOfBounds })
    );
    assert_eq!(
        refused(&|_| (), 1, 15),
        Err(CopyError::Layout { side: CopySide::Destination, error: LayoutError::RegionOutOfBounds })
    );
    assert_eq!(
        refused(&|r| { r.field(k.block_size, k.block_width, 1); }, 1, 1),
        Err(CopyError::BlockWidth { side: CopySide::Destination, value: 1 })
    );
    let top = cla140::SET_DST_BLOCK_SIZE_HEIGHT_THIRTYTWO_GOBS;
    assert_eq!(
        refused(&|r| { r.field(k.block_size, k.block_height, top + 1); }, 1, 1),
        Err(CopyError::BlockHeight { side: CopySide::Destination, value: top + 1 })
    );
    assert_eq!(
        refused(&|r| { r.field(k.block_size, k.block_depth, top + 1); }, 1, 1),
        Err(CopyError::BlockDepth { side: CopySide::Destination, value: top + 1 })
    );
}

#[test]
fn extra_words_and_words_without_a_launch_are_ignored() {
    let k = INLINE_C7C0;
    let mem = memory();
    let before = snapshot(&mem);
    let mut rig = Rig::new(k);
    // No launch yet.
    rig.word(&mem, 0xdead_beef).unwrap();
    assert!(snapshot(&mem) == before);
    // A 6-byte transfer: the second word supplies two bytes, the third none.
    rig.configure(Dest::Pitch { address: DST, pitch: 0 }, 6, 1).launch(&mem).unwrap();
    rig.word(&mem, 0x0403_0201).unwrap();
    rig.word(&mem, 0x0807_0605).unwrap();
    rig.word(&mem, 0x0c0b_0a09).unwrap();
    let rest = (DST - LOW_BASE + 6) as usize..(DST - LOW_BASE + 8) as usize;
    assert_eq!(read(&mem, DST, 8), [vec![1, 2, 3, 4, 5, 6], before[rest].to_vec()].concat());
    // Not read: the layer, interrupt and sysmembar fields.
    let other = memory();
    let mut noisy = Rig::new(k);
    noisy.configure(Dest::Pitch { address: DST, pitch: 0 }, 6, 1)
        .regs
        .set(clc7c0::SET_DST_LAYER, 7)
        .field(k.launch, clc7c0::LAUNCH_DMA_INTERRUPT_TYPE, clc7c0::LAUNCH_DMA_INTERRUPT_TYPE_INTERRUPT)
        .field(k.launch, clc7c0::LAUNCH_DMA_SYSMEMBAR_DISABLE, clc7c0::LAUNCH_DMA_SYSMEMBAR_DISABLE_TRUE);
    noisy.launch(&other).unwrap();
    noisy.word(&other, 0x0403_0201).unwrap();
    noisy.word(&other, 0x0807_0605).unwrap();
    assert!(snapshot(&other) == snapshot(&mem));
}

#[test]
fn other_methods_do_nothing() {
    let mem = memory();
    let mut rig = Rig::new(INLINE_A140);
    rig.configure(Dest::Pitch { address: DST, pitch: 0 }, 4, 1);
    for method in [cla140::LINE_COUNT, cla140::SET_I2M_SEMAPHORE_A, cla140::NO_OPERATION] {
        rig.method(&mem, method).unwrap();
    }
    assert_eq!(rig.engine.remaining(), 0);
    assert_eq!(rig.sink.events, vec![]);
}

#[test]
fn a_short_register_file_is_refused() {
    for class in CLASSES {
        let mem = memory();
        let regs = vec![0; class.registers_needed() - 1];
        for method in [class.launch, class.data] {
            assert_eq!(
                InlineUpload::new().method(
                    &class,
                    method.method(),
                    &regs,
                    &mem,
                    &mut PendingWrites::new(),
                    &mut Recorder::default()
                ),
                Err(CopyError::Registers { len: regs.len(), needed: class.registers_needed() })
            );
        }
    }
}

/// The byte addresses and the fields whose widths differ, written out from each header
/// (`cla140.h`, `clc797.h`, `clc7c0.h`).
#[test]
fn constants_match_the_headers_written_out() {
    for class in CLASSES {
        let methods = [
            (class.line_length_in, 0x180),
            (class.line_count, 0x184),
            (class.offset_out_upper, 0x188),
            (class.offset_out, 0x18c),
            (class.pitch_out, 0x190),
            (class.block_size, 0x194),
            (class.width, 0x198),
            (class.height, 0x19c),
            (class.depth, 0x1a0),
            (class.origin_x, 0x1a8),
            (class.origin_y, 0x1ac),
            (class.launch, 0x1b0),
            (class.data, 0x1b4),
            (class.semaphore_a, 0x1dc),
            (class.semaphore_b, 0x1e0),
            (class.semaphore_c, 0x1e4),
        ];
        for (method, byte) in methods {
            assert_eq!(method.byte_address, byte, "{} {}", class.launch_word, method.name);
        }
        assert_eq!(class.registers_needed(), 0x1e4 / 4 + 1);
        assert_eq!(class.layout, Field::new(0, 0));
        assert_eq!(class.layout_pitch, 1);
        assert_eq!(class.reduction_enable, Field::new(1, 1));
        assert_eq!(class.reduction_format, Field::new(3, 2));
        assert_eq!(class.completion, Field::new(5, 4));
        assert_eq!(class.completion_release, 2);
        assert_eq!(class.struct_size, Field::new(12, 12));
        assert_eq!(class.struct_four_words, 0);
        assert_eq!(class.reduction_op, Field::new(15, 13));
        assert_eq!(class.origin_x_value, Field::new(20, 0));
        assert_eq!(class.origin_y_value, Field::new(16, 0));
        assert_eq!(class.block_width, Field::new(3, 0));
        assert_eq!(class.block_height, Field::new(7, 4));
        assert_eq!(class.block_depth, Field::new(11, 8));
    }
    assert_eq!(cla140::OFFSET_OUT_UPPER_VALUE, Field::new(24, 0));
    assert_eq!(clc797::OFFSET_OUT_UPPER_VALUE, Field::new(7, 0));
    assert_eq!(clc7c0::OFFSET_OUT_UPPER_VALUE, Field::new(16, 0));
    assert_eq!(cla140::SET_I2M_SEMAPHORE_A_OFFSET_UPPER, Field::new(24, 0));
    assert_eq!(clc797::SET_I2M_SEMAPHORE_A_OFFSET_UPPER, Field::new(7, 0));
    assert_eq!(clc7c0::SET_I2M_SEMAPHORE_A_OFFSET_UPPER, Field::new(7, 0));
    // The reduction format values are the `RED_*` encoding's.
    assert_eq!(cla140::LAUNCH_DMA_REDUCTION_FORMAT_UNSIGNED_32, crate::sema::RED_FORMAT_UNSIGNED);
    assert_eq!(cla140::LAUNCH_DMA_REDUCTION_FORMAT_SIGNED_32, crate::sema::RED_FORMAT_SIGNED);
    assert_eq!(cla140::LAUNCH_DMA_REDUCTION_OP_RED_ADD, crate::sema::RED_ADD);
    assert_eq!(cla140::LAUNCH_DMA_REDUCTION_OP_RED_XOR, crate::sema::RED_XOR);
}
