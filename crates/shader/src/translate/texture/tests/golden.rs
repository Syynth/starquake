//! Golden IR: the exact text of representative forms, so a change to what a form translates to
//! shows in review.

use super::*;

/// `TEX.LB.DC` on a 2D array with a fault predicate: the array index first, then the
/// coordinates, the bias and the reference; red, green and blue from the comparison, alpha
/// 1.0, and the fault predicate written false.
#[test]
fn biased_depth_sample_of_an_array() {
    let op = OpTex {
        dim: TexDim::Array2D,
        lod_mode: TexLodMode::Bias,
        z_cmpr: true,
        fault: p(2),
        ..tex()
    };
    assert_eq!(
        translated(Op::Tex(op)).to_string(),
        "block0:\n\
         \x20   %0: u32 = GetRegister r10\n\
         \x20   %1: u32 = BitFieldUExtract %0, u32 0x0, u32 0x10\n\
         \x20   %2: f32 = ConvertF32U16 %1\n\
         \x20   %3: u32 = GetRegister r11\n\
         \x20   %4: f32 = BitCastF32U32 %3\n\
         \x20   %5: u32 = GetRegister r20\n\
         \x20   %6: f32 = BitCastF32U32 %5\n\
         \x20   %7: f32x3 = CompositeConstructF32x3 %4, %6, %2\n\
         \x20   %8: u32 = GetRegister r21\n\
         \x20   %9: f32 = BitCastF32U32 %8\n\
         \x20   %10: u32 = GetRegister r22\n\
         \x20   %11: f32 = BitCastF32U32 %10\n\
         \x20   %12: f32 = ImageSampleDrefImplicitLod u32 0x40, %7, %11, %9, u32 0x0 {type=array_2d, depth, bias}\n\
         \x20   %13: u32 = BitCastU32F32 %12\n\
         \x20   %14 = SetRegister r0, %13\n\
         \x20   %15: u32 = BitCastU32F32 %12\n\
         \x20   %16 = SetRegister r1, %15\n\
         \x20   %17: u32 = BitCastU32F32 %12\n\
         \x20   %18 = SetRegister r2, %17\n\
         \x20   %19: u32 = BitCastU32F32 f32 0x3f800000\n\
         \x20   %20 = SetRegister r3, %19\n\
         \x20   %21 = SetPred p2, u1 false\n\
         \x20   return\n"
    );
}

/// `TLD4.PTP`: four gathers, each with one texel's two offset fields, their component 3
/// combined into the result.
#[test]
fn per_texel_offset_gather() {
    let op = OpTld4 {
        offset_mode: TexOffsetMode::PerPx,
        channel_mask: mask(0x3),
        dsts: [r(0), Gpr::RZ],
        ..tld4()
    };
    assert_eq!(
        translated(Op::Tld4(op)).to_string(),
        "block0:\n\
         \x20   %0: u32 = GetRegister r10\n\
         \x20   %1: f32 = BitCastF32U32 %0\n\
         \x20   %2: u32 = GetRegister r11\n\
         \x20   %3: f32 = BitCastF32U32 %2\n\
         \x20   %4: f32x2 = CompositeConstructF32x2 %1, %3\n\
         \x20   %5: u32 = GetRegister r20\n\
         \x20   %6: u32 = BitFieldSExtract %5, u32 0x0, u32 0x6\n\
         \x20   %7: u32 = BitFieldSExtract %5, u32 0x8, u32 0x6\n\
         \x20   %8: u32 = BitFieldSExtract %5, u32 0x10, u32 0x6\n\
         \x20   %9: u32 = BitFieldSExtract %5, u32 0x18, u32 0x6\n\
         \x20   %10: u32 = GetRegister r21\n\
         \x20   %11: u32 = BitFieldSExtract %10, u32 0x0, u32 0x6\n\
         \x20   %12: u32 = BitFieldSExtract %10, u32 0x8, u32 0x6\n\
         \x20   %13: u32 = BitFieldSExtract %10, u32 0x10, u32 0x6\n\
         \x20   %14: u32 = BitFieldSExtract %10, u32 0x18, u32 0x6\n\
         \x20   %15: u32x2 = CompositeConstructU32x2 %6, %7\n\
         \x20   %16: f32x4 = ImageGather u32 0x40, %4, %15, u32 0x0 {type=2d, gather=1}\n\
         \x20   %17: f32 = CompositeExtractF32x4 %16, u32 0x3\n\
         \x20   %18: u32x2 = CompositeConstructU32x2 %8, %9\n\
         \x20   %19: f32x4 = ImageGather u32 0x40, %4, %18, u32 0x0 {type=2d, gather=1}\n\
         \x20   %20: f32 = CompositeExtractF32x4 %19, u32 0x3\n\
         \x20   %21: u32x2 = CompositeConstructU32x2 %11, %12\n\
         \x20   %22: f32x4 = ImageGather u32 0x40, %4, %21, u32 0x0 {type=2d, gather=1}\n\
         \x20   %23: f32 = CompositeExtractF32x4 %22, u32 0x3\n\
         \x20   %24: u32x2 = CompositeConstructU32x2 %13, %14\n\
         \x20   %25: f32x4 = ImageGather u32 0x40, %4, %24, u32 0x0 {type=2d, gather=1}\n\
         \x20   %26: f32 = CompositeExtractF32x4 %25, u32 0x3\n\
         \x20   %27: f32x4 = CompositeConstructF32x4 %17, %20, %23, %26\n\
         \x20   %28: f32 = CompositeExtractF32x4 %27, u32 0x0\n\
         \x20   %29: f32 = CompositeExtractF32x4 %27, u32 0x1\n\
         \x20   %30: u32 = BitCastU32F32 %28\n\
         \x20   %31 = SetRegister r0, %30\n\
         \x20   %32: u32 = BitCastU32F32 %29\n\
         \x20   %33 = SetRegister r1, %32\n\
         \x20   return\n"
    );
}

/// `TMML` of both channels: the relative level (the host's second) signed, the absolute one
/// unsigned, each times 256 and masked to 16 bits.
#[test]
fn level_query_of_both_channels() {
    let op = OpTmml {
        channel_mask: mask(0x3),
        ..tmml()
    };
    assert_eq!(
        translated(Op::Tmml(op)).to_string(),
        "block0:\n\
         \x20   %0: u32 = GetRegister r10\n\
         \x20   %1: f32 = BitCastF32U32 %0\n\
         \x20   %2: u32 = GetRegister r11\n\
         \x20   %3: f32 = BitCastF32U32 %2\n\
         \x20   %4: f32x2 = CompositeConstructF32x2 %1, %3\n\
         \x20   %5: f32x4 = ImageQueryLod u32 0x40, %4 {type=2d}\n\
         \x20   %6: f32 = CompositeExtractF32x4 %5, u32 0x1\n\
         \x20   %7: f32 = FPMul32 %6, f32 0x43800000\n\
         \x20   %8: u32 = ConvertS32F32 %7\n\
         \x20   %9: u32 = BitwiseAnd32 %8, u32 0xffff\n\
         \x20   %10 = SetRegister r4, %9\n\
         \x20   %11: f32 = CompositeExtractF32x4 %5, u32 0x0\n\
         \x20   %12: f32 = FPMul32 %11, f32 0x43800000\n\
         \x20   %13: u32 = ConvertU32F32 %12\n\
         \x20   %14: u32 = BitwiseAnd32 %13, u32 0xffff\n\
         \x20   %15 = SetRegister r5, %14\n\
         \x20   return\n"
    );
}

/// `SUATOM.MIN.S32` on a 1D array: integer coordinates with the array index last, the signed
/// minimum, the old value written back.
#[test]
fn signed_surface_atomic_of_an_array() {
    let op = OpSuAtom {
        atom_op: AtomOp::Min,
        atom_type: AtomType::I32,
        dim: ImageDim::Array1D,
        ..suatom()
    };
    assert_eq!(
        translated(Op::SuAtom(op)).to_string(),
        "block0:\n\
         \x20   %0: u32 = GetRegister r10\n\
         \x20   %1: u32 = GetRegister r11\n\
         \x20   %2: u32 = BitFieldUExtract %1, u32 0x0, u32 0x10\n\
         \x20   %3: u32x2 = CompositeConstructU32x2 %0, %2\n\
         \x20   %4: u32 = GetRegister r30\n\
         \x20   %5: u32 = GetRegister r20\n\
         \x20   %6: u32 = ImageAtomicSMin32 %4, %3, %5 {type=array_1d}\n\
         \x20   %7 = SetRegister r0, %6\n\
         \x20   return\n"
    );
}
