//! Golden resource tables, the merge rules, the ordering of OR-ed words, and array lengths.

use super::{
    BOUND_BANK, Engine, FLOAT, answered, buffer, info_at, kind_of, op_at, parse, pass, sample_of,
    sources, untraced, value,
};
use crate::exchange::{Question, Questioner, SampledType, TextureUsage};
use crate::opt::resources::trace::dynamic_count;
use crate::opt::resources::{
    DESCRIPTOR_MAX_COUNT, DESCRIPTOR_SIZE, DESCRIPTOR_SIZE_SHIFT, DYNAMIC_ARRAY_BYTES, Untraced,
    run,
};
use crate::translate::texture::ABSENT;

/// Bound, bindless (bank 24), two-word, indexed and uniformly indexed handles, a storage image
/// read and written, a texel buffer fetched and one read as an image, and an integer texture.
const GOLDEN: &str = "block0:
    %0: u32 = GetRegister r0
    %50: f32x2 = CompositeConstructF32x2 f32 0x0, f32 0x0
    %51: u32x2 = CompositeConstructU32x2 u32 0x0, u32 0x0
    %1: f32x4 = ImageSampleImplicitLod u32 0x40, %50, u32 0x0, u32 0x0 {type=2d}
    %2: u32 = GetCbufU32 u32 0x18, u32 0x10
    %3: f32 = ImageSampleDrefImplicitLod %2, %50, f32 0x0, u32 0x0, u32 0x0 {type=2d, depth}
    %4: u32 = GetCbufU32 u32 0x3, u32 0x20
    %5: u32 = BitwiseAnd32 %4, u32 0xfffff
    %6: u32 = GetCbufU32 u32 0x3, u32 0x24
    %7: u32 = ShiftLeftLogical32 %6, u32 0x14
    %8: u32 = BitwiseOr32 %5, %7
    %9: f32x4 = ImageSampleImplicitLod %8, %50, u32 0x0, u32 0x0 {type=3d}
    %10: u32 = ShiftLeftLogical32 %0, u32 0x3
    %11: u32 = IAdd32 %10, u32 0x100
    %12: u32 = GetCbufU32 u32 0x3, %11
    %13: f32x4 = ImageSampleImplicitLod %12, %50, u32 0x0, u32 0x0 {type=2d}
    %14: u32 = GetCbufU32 u32 0x5, u32 0x8
    %15: u32 = ShiftLeftLogical32 %14, u32 0x2
    %16: u32 = IAdd32 u32 0x200, %15
    %17: u32 = GetCbufU32 u32 0x3, %16
    %18: f32x4 = ImageSampleImplicitLod %17, %50, u32 0x0, u32 0x0 {type=cube}
    %19: u32 = GetCbufU32 u32 0x4, u32 0x30
    %20: u32x4 = ImageRead %19, %51 {type=2d, format=r32_uint}
    %21 = ImageWrite %19, %51, %20 {type=2d, format=r32_uint}
    %22: u32 = GetCbufU32 u32 0x4, u32 0x34
    %23: f32x4 = ImageFetch %22, %0, u32 0x0, u32 0x0, u32 0x0 {type=1d}
    %24: u32 = GetCbufU32 u32 0x4, u32 0x38
    %25: u32x4 = ImageRead %24, %0 {type=buffer, format=r32_uint}
    %26: f32x4 = ImageSampleImplicitLod u32 0x44, %50, u32 0x0, u32 0x0 {type=2d}
    return
";

/// The engine state [`GOLDEN`] runs against: each word and handle distinct.
fn golden_engine() -> Engine {
    Engine::new()
        .word(BOUND_BANK, 0x40, 0x1001)
        .kind(0x1001, FLOAT)
        .word(24, 0x10, 0x2002)
        .kind(0x2002, FLOAT)
        .word(3, 0x20, 0xfff3_3003)
        .word(3, 0x24, 0x5)
        .kind(0x0053_3003, FLOAT)
        .word(3, 0x100, 0x4004)
        .kind(0x4004, FLOAT)
        .word(3, 0x200, 0x6006)
        .kind(0x6006, kind_of(SampledType::SignedInt, false, 0x111))
        .word(4, 0x30, 0x7007)
        .kind(0x7007, kind_of(SampledType::UnsignedInt, false, 0x12345))
        .word(4, 0x34, 0x8008)
        .kind(0x8008, kind_of(SampledType::Float, true, 0))
        .word(4, 0x38, 0x9009)
        .kind(0x9009, kind_of(SampledType::UnsignedInt, true, 0x23))
        .word(BOUND_BANK, 0x44, 0xa00a)
        .kind(0xa00a, kind_of(SampledType::SignedInt, false, 0))
}

#[test]
fn golden_tables() {
    let (function, table) = pass(GOLDEN, &golden_engine());
    insta::assert_snapshot!("golden_table", table.to_string());
    insta::assert_snapshot!("golden_ir", function.to_string());
}

#[test]
fn a_uniform_index_keeps_one_descriptor() {
    let (function, table) = pass(GOLDEN, &golden_engine());
    let cube = table
        .textures()
        .iter()
        .find(|d| d.ty == crate::ir::TextureType::ColorCube)
        .expect("the cube");
    assert_eq!(cube.source.count(), 1);
    let index = cube.source.index().expect("an index word");
    assert_eq!((index.bank(), index.offset(), index.shift()), (5, 8, 2));
    assert_eq!(op_at(&function, 18).args.first(), Some(&ABSENT));
    // The array indexed at run time takes the clamped index instead.
    assert_ne!(op_at(&function, 13).args.first(), Some(&ABSENT));
}

#[test]
fn an_index_alone_starts_at_zero() {
    let text = sample_of(
        "    %0: u32 = GetRegister r0\n    %1: u32 = GetCbufU32 u32 3, %0\n",
        "%1",
    );
    assert_eq!(
        sources(&text),
        format!("%60: cbuf 3 at 0x0, {DESCRIPTOR_MAX_COUNT} handles 8 bytes apart\n")
    );
}

/// An array OR-ed with a single word: the array is the primary word, on either side.
#[test]
fn an_array_or_a_single_word() {
    for (a, b) in [("%2", "%3"), ("%3", "%2")] {
        let lines = format!(
            "    %0: u32 = GetRegister r0\n    %1: u32 = IAdd32 %0, u32 0x100\n    \
             %2: u32 = GetCbufU32 u32 3, %1\n    %3: u32 = GetCbufU32 u32 3, u32 0x8\n    \
             %4: u32 = BitwiseOr32 {a}, {b}\n"
        );
        assert_eq!(
            sources(&sample_of(&lines, "%4")),
            "%60: cbuf 3 at 0x100 | cbuf 3 at 0x8, 1024 handles 8 bytes apart\n"
        );
    }
}

/// Eden's order of two single words: the left one is primary unless it is shifted, or in a
/// higher bank, or at a higher offset.
#[test]
fn single_words_are_ordered_as_eden_orders_them() {
    let or = |left: (u32, u32, u32), right: (u32, u32, u32)| {
        let lines = format!(
            "    %0: u32 = GetCbufU32 u32 {:#x}, u32 {:#x}\n    \
             %1: u32 = ShiftLeftLogical32 %0, u32 {:#x}\n    \
             %2: u32 = GetCbufU32 u32 {:#x}, u32 {:#x}\n    \
             %3: u32 = ShiftLeftLogical32 %2, u32 {:#x}\n    %4: u32 = BitwiseOr32 %1, %3\n",
            left.0, left.1, left.2, right.0, right.1, right.2
        );
        sources(&sample_of(&lines, "%4"))
    };
    // Kept: unshifted, lower bank, lower offset.
    assert_eq!(
        or((3, 0x10, 0), (4, 0x20, 8)),
        "%60: cbuf 3 at 0x10 | cbuf 4 at 0x20 << 8 & 0xffffff00\n"
    );
    // Equal bank and offset: kept.
    assert_eq!(
        or((3, 0x10, 0), (3, 0x10, 8)),
        "%60: cbuf 3 at 0x10 | cbuf 3 at 0x10 << 8 & 0xffffff00\n"
    );
    // Swapped: the left one is shifted.
    assert_eq!(
        or((3, 0x10, 8), (4, 0x20, 0)),
        "%60: cbuf 4 at 0x20 | cbuf 3 at 0x10 << 8 & 0xffffff00\n"
    );
    // Swapped: a higher bank, even at a lower offset.
    assert_eq!(
        or((4, 0x10, 0), (3, 0x20, 0)),
        "%60: cbuf 3 at 0x20 | cbuf 4 at 0x10\n"
    );
    // Swapped: a higher offset in a lower bank (Eden compares each field on its own).
    assert_eq!(
        or((3, 0x20, 0), (4, 0x10, 0)),
        "%60: cbuf 4 at 0x10 | cbuf 3 at 0x20\n"
    );
    // Swapped: a higher offset, one word past.
    assert_eq!(
        or((3, 0x14, 0), (3, 0x10, 0)),
        "%60: cbuf 3 at 0x10 | cbuf 3 at 0x14\n"
    );
}

#[test]
fn two_arrays_or_two_index_words_are_refused() {
    let arrays = "    %0: u32 = GetRegister r0\n    %1: u32 = GetCbufU32 u32 3, %0\n    \
                  %2: u32 = GetCbufU32 u32 4, %0\n    %3: u32 = BitwiseOr32 %1, %2\n";
    assert_eq!(
        untraced(&sample_of(arrays, "%3")),
        Untraced::TwoArrays(value(3))
    );
    // Two index words on arrays of one handle each (both start past the array limit).
    let indexed = format!(
        "    %0: u32 = GetCbufU32 u32 5, u32 0x0\n    %1: u32 = IAdd32 %0, u32 {DYNAMIC_ARRAY_BYTES:#x}\n    \
         %2: u32 = GetCbufU32 u32 3, %1\n    %3: u32 = GetCbufU32 u32 4, %1\n    \
         %4: u32 = BitwiseOr32 %2, %3\n"
    );
    assert_eq!(
        untraced(&sample_of(&indexed, "%4")),
        Untraced::TwoIndexWords(value(4))
    );
    // One index word and one single word: the indexed one is primary, on either side.
    for (a, b) in [("%2", "%3"), ("%3", "%2")] {
        let lines = format!(
            "    %0: u32 = GetCbufU32 u32 5, u32 0x0\n    %1: u32 = IAdd32 %0, u32 0x100\n    \
             %2: u32 = GetCbufU32 u32 3, %1\n    %3: u32 = GetCbufU32 u32 3, u32 0x8\n    \
             %4: u32 = BitwiseOr32 {a}, {b}\n"
        );
        assert_eq!(
            sources(&sample_of(&lines, "%4")),
            "%60: cbuf 3 at 0x100 | cbuf 3 at 0x8, 1024 handles 8 bytes apart, index from cbuf 5 at 0x0 << 0\n"
        );
    }
}

/// Pinned for the conflict with PR #4588 (module docs): a single shifted word's handle is
/// `word << shift`, not `(word << shift) | word`.
#[test]
fn a_shifted_single_word() {
    let text = sample_of(
        "    %0: u32 = GetCbufU32 u32 3, u32 0x10\n    %1: u32 = ShiftLeftLogical32 %0, u32 0x8\n",
        "%1",
    );
    let engine = Engine::new().word(3, 0x10, 0x12).kind(0x1200, FLOAT);
    let function = parse(&text);
    let (found, answers) = answered(&function, buffer(BOUND_BANK), &engine);
    assert!(found.questions(&answers).contains(&Question::TextureKind {
        handle: 0x1200,
        usage: TextureUsage::Sampled
    }));
    let mut function = function;
    assert!(run(&mut function, &found, &answers).is_ok());
}

/// Eden's `DynamicDescriptorCount`, at each edge, with literal expectations.
#[test]
fn array_lengths_at_the_edge() {
    assert_eq!(
        (DESCRIPTOR_SIZE, DESCRIPTOR_MAX_COUNT, DYNAMIC_ARRAY_BYTES),
        (8, 1024, 0x4000)
    );
    // From offset 0, eight bytes apart, 2048 fit: held to the limit.
    assert_eq!(dynamic_count(0, DESCRIPTOR_SIZE_SHIFT), 1024);
    // Sixteen bytes apart: 1 + (0x4000 - 8) / 16 = 1024, exactly the limit.
    assert_eq!(dynamic_count(0, 4), 1024);
    assert_eq!(dynamic_count(16, 4), 1023);
    // Room for exactly one descriptor, then one byte too little.
    assert_eq!(dynamic_count(0x4000 - 8, DESCRIPTOR_SIZE_SHIFT), 1);
    assert_eq!(dynamic_count(0x4000 - 16, DESCRIPTOR_SIZE_SHIFT), 2);
    assert_eq!(dynamic_count(0x4000 - 7, DESCRIPTOR_SIZE_SHIFT), 1);
    // At and past the bytes an array may start in.
    assert_eq!(dynamic_count(0x4000, DESCRIPTOR_SIZE_SHIFT), 1);
    assert_eq!(dynamic_count(0x3fff, 2), 1);
    // A shift of 31 or more is one handle; 30 spaces them 1 GiB apart.
    assert_eq!(dynamic_count(0, 31), 1);
    assert_eq!(dynamic_count(0, 30), 1);
    assert_eq!(dynamic_count(0, 13), 2);
    assert_eq!(dynamic_count(u32::MAX, 2), 1);
}

#[test]
fn descriptors_merge_as_edens_do() {
    // The same handle sampled as 2D twice and as 3D once; then read and written as an image;
    // and fetched with and without a sample index.
    let text = "block0:
    %0: u32 = GetCbufU32 u32 0x3, u32 0x10
    %1: f32x2 = CompositeConstructF32x2 f32 0x0, f32 0x0
    %2: u32x2 = CompositeConstructU32x2 u32 0x0, u32 0x0
    %3: f32x4 = ImageSampleImplicitLod %0, %1, u32 0x0, u32 0x0 {type=2d}
    %4: f32x4 = ImageSampleImplicitLod %0, %1, u32 0x0, u32 0x0 {type=3d}
    %5: f32x4 = ImageSampleExplicitLod %0, %1, u32 0x0, u32 0x0 {type=2d}
    %6: u32 = GetCbufU32 u32 0x3, u32 0x20
    %7: u32x4 = ImageRead %6, %2 {type=2d, format=r32_uint}
    %8 = ImageWrite %6, %2, %7 {type=2d, format=r32_uint}
    %9: u32x4 = ImageRead %6, %2 {type=2d, format=r8_uint}
    %10: u32 = GetCbufU32 u32 0x3, u32 0x30
    %11: f32x4 = ImageFetch %10, %2, u32 0x0, u32 0x0, u32 0x0 {type=2d}
    %12: f32x4 = ImageFetch %10, %2, u32 0x0, u32 0x0, u32 0x1 {type=2d}
    return
";
    let engine = Engine::new()
        .word(3, 0x10, 0x11)
        .kind(0x11, FLOAT)
        .word(3, 0x20, 0x22)
        .kind(0x22, FLOAT)
        .word(3, 0x30, 0x33)
        .kind(0x33, FLOAT);
    let (_, table) = pass(text, &engine);
    insta::assert_snapshot!("merged_table", table.to_string());
    assert_eq!(table.textures().len(), 3);
    assert_eq!(table.images().len(), 2);
    let descriptor = |at: u32| {
        let (function, _) = pass(text, &engine);
        info_at(&function, at).descriptor_index
    };
    assert_eq!(
        [3, 4, 5, 7, 8, 9, 11, 12].map(descriptor),
        [0, 1, 0, 0, 0, 1, 2, 2]
    );
}

/// Texel buffers read as images merge by format and source, as Eden's image buffers do.
#[test]
fn image_buffers_merge_by_format() {
    let text = "block0:
    %0: u32 = GetCbufU32 u32 0x3, u32 0x10
    %1: u32 = GetRegister r0
    %2: u32x4 = ImageRead %0, %1 {type=buffer, format=r32_uint}
    %3 = ImageWrite %0, %1, %2 {type=buffer, format=r32_uint}
    %4: u32x4 = ImageRead %0, %1 {type=buffer, format=r16_uint}
    return
";
    let engine = Engine::new()
        .word(3, 0x10, 0x11)
        .kind(0x11, kind_of(SampledType::UnsignedInt, true, 0x7));
    let (function, table) = pass(text, &engine);
    assert_eq!(
        table.to_string(),
        "image buffer 0: r32_uint, unsigned, header format 0x7, read and written, from cbuf 3 at 0x10\n\
         image buffer 1: r16_uint, unsigned, header format 0x7, read, from cbuf 3 at 0x10\n"
    );
    let index = |at| info_at(&function, at).descriptor_index;
    assert_eq!([2, 3, 4].map(index), [0, 0, 1]);
}

/// An array that holds one handle (it starts at the array limit) is spaced
/// `DESCRIPTOR_SIZE_SHIFT` whatever its index's scale, as Eden's descriptor of one handle is,
/// so it merges with a single word at the same place.
#[test]
fn an_array_of_one_handle_is_a_single_word() {
    let text = format!(
        "block0:
    %0: u32 = GetRegister r0
    %1: u32 = ShiftLeftLogical32 %0, u32 0x4
    %2: u32 = IAdd32 %1, u32 {DYNAMIC_ARRAY_BYTES:#x}
    %3: u32 = GetCbufU32 u32 0x3, %2
    %4: f32x2 = CompositeConstructF32x2 f32 0x0, f32 0x0
    %5: f32x4 = ImageSampleImplicitLod %3, %4, u32 0x0, u32 0x0 {{type=2d}}
    %6: u32 = GetCbufU32 u32 0x3, u32 {DYNAMIC_ARRAY_BYTES:#x}
    %7: f32x4 = ImageSampleImplicitLod %6, %4, u32 0x0, u32 0x0 {{type=2d}}
    return
"
    );
    let engine = Engine::new()
        .word(3, DYNAMIC_ARRAY_BYTES, 0x5)
        .kind(0x5, FLOAT);
    let (function, table) = pass(&text, &engine);
    let only = table.textures();
    assert_eq!(only.len(), 1, "{table}");
    assert_eq!(
        only.first().map(|d| d.source.size_shift()),
        Some(DESCRIPTOR_SIZE_SHIFT)
    );
    assert_eq!(op_at(&function, 5).args.first(), Some(&ABSENT));
}

/// A depth sample and a colour sample of the same handle and type are two descriptors (Eden
/// compares `is_depth`).
#[test]
fn depth_keeps_descriptors_apart() {
    let text = "block0:
    %0: u32 = GetCbufU32 u32 0x3, u32 0x10
    %1: f32x2 = CompositeConstructF32x2 f32 0x0, f32 0x0
    %2: f32x4 = ImageSampleImplicitLod %0, %1, u32 0x0, u32 0x0 {type=2d}
    %3: f32 = ImageSampleDrefImplicitLod %0, %1, f32 0x0, u32 0x0, u32 0x0 {type=2d, depth}
    return
";
    let engine = Engine::new().word(3, 0x10, 0x11).kind(0x11, FLOAT);
    let (function, table) = pass(text, &engine);
    assert_eq!(
        table.to_string(),
        "texture 0: 2d, float, from cbuf 3 at 0x10\n\
         texture 1: 2d, float, depth, from cbuf 3 at 0x10\n"
    );
    let index = |at| info_at(&function, at).descriptor_index;
    assert_eq!([2, 3].map(index), [0, 1]);
}

/// A merged descriptor keeps a multisample flag an earlier use set (Eden ORs it in).
#[test]
fn a_later_use_keeps_the_multisample_flag() {
    let text = "block0:
    %0: u32 = GetCbufU32 u32 0x3, u32 0x30
    %1: u32x2 = CompositeConstructU32x2 u32 0x0, u32 0x0
    %2: f32x4 = ImageFetch %0, %1, u32 0x0, u32 0x0, u32 0x1 {type=2d}
    %3: f32x4 = ImageFetch %0, %1, u32 0x0, u32 0x0, u32 0x0 {type=2d}
    return
";
    let engine = Engine::new().word(3, 0x30, 0x33).kind(0x33, FLOAT);
    let (_, table) = pass(text, &engine);
    assert_eq!(
        table.to_string(),
        "texture 0: 2d, float, multisampled, from cbuf 3 at 0x30\n"
    );
}
