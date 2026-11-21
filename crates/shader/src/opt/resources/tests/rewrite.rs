//! What each texture operation becomes: the array index, the type changes, the sample
//! argument, the integer flag, descriptor order, and interpreter equivalence.

use super::{BOUND_BANK, Engine, FLOAT, kind_of, parse, pass, value};
use crate::exchange::SampledType;
use crate::interp::{ImageCall, Machine, ScriptedImage, Val};
use crate::ir::{Flags, Function, Inst, Opcode, Operation, TextureInfo, TextureType};
use crate::opt::resources::DESCRIPTOR_MAX_COUNT;
use crate::translate::texture::ABSENT;

/// The operation at `at`.
fn op_at(function: &Function, at: u32) -> Operation {
    match function.inst(value(at)) {
        Some(Inst::Op(op)) => op.clone(),
        other => panic!("no operation at %{at}: {other:?}"),
    }
}

/// The texture information of the operation at `at`.
fn info_at(function: &Function, at: u32) -> TextureInfo {
    match op_at(function, at).flags {
        Flags::Texture(info) => info,
        other => panic!("not texture flags: {other:?}"),
    }
}

/// An array of handles at 0x100 of bank 3, eight bytes apart, indexed by r0; the red channel
/// of the sample goes to r1.
const ARRAY: &str = "block0:
    %0: u32 = GetRegister r0
    %1: u32 = ShiftLeftLogical32 %0, u32 0x3
    %2: u32 = IAdd32 %1, u32 0x100
    %3: u32 = GetCbufU32 u32 0x3, %2
    %4: f32x2 = CompositeConstructF32x2 f32 0x0, f32 0x0
    %5: f32x4 = ImageSampleImplicitLod %3, %4, u32 0x0, u32 0x0 {type=2d}
    %6: f32 = CompositeExtractF32x4 %5, u32 0x0
    %7: u32 = BitCastU32F32 %6
    %8 = SetRegister r1, %7
    return
";

/// The handle of array element `k`.
const fn handle_of(k: u32) -> u32 {
    0x5000_0000 | (k * 0x11)
}

/// What the texture of handle `handle` returns: a texel unique to it.
const fn texel(handle: u32) -> Val {
    Val::F32x4([handle, handle ^ 1, handle ^ 2, handle ^ 3])
}

/// Bank 3 filled with the array's handles, the whole 64 KiB a bank holds.
fn array_machine(r0: u32) -> Machine {
    let mut machine = Machine::new();
    let mut bytes = vec![0_u8; 0x1_0000];
    for k in 0..DESCRIPTOR_MAX_COUNT {
        let at = usize::try_from(0x100 + 8 * k).expect("fits");
        bytes[at..at + 4].copy_from_slice(&handle_of(k).to_le_bytes());
    }
    machine.cbufs.insert(3, bytes);
    machine.regs[0] = r0;
    machine
}

/// A sample call with `first` as its handle argument and `info`, returning `reply`.
fn sample(first: Val, info: TextureInfo, reply: Val) -> ScriptedImage {
    ScriptedImage {
        call: ImageCall {
            opcode: Opcode::ImageSampleImplicitLod,
            info,
            args: vec![first, Val::F32x2([0, 0]), Val::U32(0), Val::U32(0)],
        },
        reply: Some(reply),
    }
}

#[test]
fn an_indexed_sample_gives_the_same_texel_after_the_pass() {
    let before = parse(ARRAY);
    let engine = Engine::new().word(3, 0x100, handle_of(0)).kind(handle_of(0), FLOAT);
    let (after, _) = pass(ARRAY, &engine);
    let last = DESCRIPTOR_MAX_COUNT - 1;
    for r0 in [0, 1, 7, last] {
        let handle = handle_of(r0);
        let mut start = array_machine(r0);
        start.images.push_back(sample(
            Val::U32(handle),
            info_at(&before, 5),
            texel(handle),
        ));
        let mut ran_before = start.clone();
        ran_before.run(&before).expect("runs before");
        // After the pass the operation names element r0 of descriptor 0; the texel is that
        // element's texture.
        let mut start = array_machine(r0);
        start.images.push_back(sample(
            Val::U32(r0),
            info_at(&after, 5),
            texel(handle_of(r0)),
        ));
        let mut ran_after = start.clone();
        ran_after.run(&after).expect("runs after");
        assert_eq!(ran_after.regs[1], ran_before.regs[1], "r0 = {r0}");
        assert_eq!(ran_after.regs[1], handle);
    }
    // Past the array the index is clamped to its last element, as Eden clamps it.
    let mut start = array_machine(last + 1);
    start
        .images
        .push_back(sample(Val::U32(last), info_at(&after, 5), texel(handle_of(last))));
    start.run(&after).expect("runs after");
    assert_eq!(start.regs[1], handle_of(last));
}

#[test]
fn a_buffer_query_and_fetch() {
    let text = "block0:
    %0: u32 = GetCbufU32 u32 0x3, u32 0x10
    %1: u32 = GetCbufU32 u32 0x3, u32 0x14
    %2: u32x2 = CompositeConstructU32x2 u32 0x0, u32 0x0
    %3: u32x4 = ImageQueryDimensions %0, u32 0x0, u1 false {type=2d}
    %4: u32x4 = ImageQueryDimensions %1, u32 0x0, u1 false {type=2d}
    %5: f32x4 = ImageFetch %0, %2, u32 0x0, u32 0x0, u32 0x0 {type=1d}
    %6: f32x4 = ImageFetch %1, %2, u32 0x0, u32 0x0, u32 0x0 {type=1d}
    %7: f32x4 = ImageSampleImplicitLod %0, %2, u32 0x0, u32 0x0 {type=2d}
    return
";
    let engine = Engine::new()
        .word(3, 0x10, 0x10)
        .kind(0x10, kind_of(SampledType::Float, true, 0))
        .word(3, 0x14, 0x14)
        .kind(0x14, FLOAT);
    let (function, table) = pass(text, &engine);
    // A buffer answer: the query and the 1D fetch become buffer operations.
    assert_eq!(info_at(&function, 3).ty, TextureType::Buffer);
    assert_eq!(info_at(&function, 5).ty, TextureType::Buffer);
    // Not a buffer: both keep their types.
    assert_eq!(info_at(&function, 4).ty, TextureType::Color2D);
    assert_eq!(info_at(&function, 6).ty, TextureType::Color1D);
    // A 2D sample of a buffer answer keeps its type: only a 1D fetch is changed.
    assert_eq!(info_at(&function, 7).ty, TextureType::Color2D);
    // The query and the fetch of the buffer share one texel-buffer descriptor.
    assert_eq!(
        table.to_string(),
        "texture buffer 0: float, from cbuf 3 at 0x10\n\
         texture 0: 2d, float, from cbuf 3 at 0x10\n\
         texture 1: 2d, float, from cbuf 3 at 0x14\n\
         texture 2: 1d, float, from cbuf 3 at 0x14\n"
    );
}

#[test]
fn a_fetch_keeps_its_sample_only_where_the_type_takes_one() {
    let text = "block0:
    %0: u32 = GetCbufU32 u32 0x3, u32 0x10
    %1: u32x2 = CompositeConstructU32x2 u32 0x0, u32 0x0
    %2: u32 = GetRegister r0
    %3: f32x4 = ImageFetch %0, %1, u32 0x0, u32 0x0, %2 {type=3d}
    %4: f32x4 = ImageFetch %0, %1, u32 0x0, u32 0x0, %2 {type=array_2d}
    %5: f32x4 = ImageFetch %0, %1, u32 0x0, u32 0x0, u32 0x0 {type=2d_rect}
    return
";
    let engine = Engine::new().word(3, 0x10, 0x10).kind(0x10, FLOAT);
    let (function, table) = pass(text, &engine);
    assert_eq!(op_at(&function, 3).args.get(4), Some(&ABSENT));
    assert_eq!(
        op_at(&function, 4).args.get(4),
        Some(&crate::ir::Operand::Value(value(2)))
    );
    let multisampled: Vec<(TextureType, bool)> = table
        .textures()
        .iter()
        .map(|d| (d.ty, d.is_multisample))
        .collect();
    assert_eq!(
        multisampled,
        [
            (TextureType::Color3D, false),
            (TextureType::ColorArray2D, true),
            (TextureType::Color2DRect, false)
        ]
    );
}

#[test]
fn integer_textures_set_the_flag() {
    for (sampled, integer) in [
        (SampledType::Float, false),
        (SampledType::SignedInt, true),
        (SampledType::UnsignedInt, true),
    ] {
        let text = "block0:
    %0: f32x2 = CompositeConstructF32x2 f32 0x0, f32 0x0
    %1: f32x4 = ImageSampleImplicitLod u32 0x40, %0, u32 0x0, u32 0x0 {type=2d}
    return
";
        let engine = Engine::new()
            .word(BOUND_BANK, 0x40, 0x99)
            .kind(0x99, kind_of(sampled, false, 0));
        let (function, table) = pass(text, &engine);
        assert_eq!(info_at(&function, 1).is_integer, integer, "{sampled:?}");
        assert_eq!(table.textures().first().map(|d| d.component), Some(sampled));
    }
}

/// Descriptors are numbered in the order the operations sort: by bank, then offset, then the
/// post order of the blocks (here the later block first).
#[test]
fn descriptors_follow_the_sort_and_the_post_order() {
    let text = "block0:
    %0: f32x2 = CompositeConstructF32x2 f32 0x0, f32 0x0
    %1: u32 = GetCbufU32 u32 0x3, u32 0x20
    %2: f32x4 = ImageSampleImplicitLod %1, %0, u32 0x0, u32 0x0 {type=2d}
    %3: f32x4 = ImageSampleImplicitLod u32 0x40, %0, u32 0x0, u32 0x0 {type=2d}
    %4: u32 = GetCbufU32 u32 0x3, u32 0x10
    %5: f32x4 = ImageSampleImplicitLod %4, %0, u32 0x0, u32 0x0 {type=2d}
    %6: f32x4 = ImageSampleImplicitLod %1, %0, u32 0x0, u32 0x0 {type=3d}
    jump block1
block1:
    %7: f32x4 = ImageSampleImplicitLod %1, %0, u32 0x0, u32 0x0 {type=cube}
    return
";
    let engine = Engine::new()
        .word(3, 0x20, 0x20)
        .kind(0x20, FLOAT)
        .word(3, 0x10, 0x10)
        .kind(0x10, FLOAT)
        .word(BOUND_BANK, 0x40, 0x40)
        .kind(0x40, FLOAT);
    let (function, table) = pass(text, &engine);
    assert_eq!(
        table.to_string(),
        "texture 0: 2d, float, from cbuf 2 at 0x40\n\
         texture 1: 2d, float, from cbuf 3 at 0x10\n\
         texture 2: cube, float, from cbuf 3 at 0x20\n\
         texture 3: 2d, float, from cbuf 3 at 0x20\n\
         texture 4: 3d, float, from cbuf 3 at 0x20\n"
    );
    let index = |at| info_at(&function, at).descriptor_index;
    assert_eq!([2, 3, 5, 6, 7].map(index), [3, 0, 1, 4, 2]);
}
