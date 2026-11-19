//! `SULD`, `SUST`, `SUATOM` and `SURED` with a register handle: every dimension, formatted and
//! binary accesses of every size, every atomic operation and both types, and the fault
//! predicate.

use super::*;

/// The handle register's value.
const HANDLE_BITS: u32 = 0x0012_3456;
const HANDLE: Val = Val::U32(HANDLE_BITS);
const X: u32 = 0x0000_0011;
const Y: u32 = 0x0000_0022;
const Z: u32 = 0x0000_0033;
/// An array index register: layer 4 below noise.
const LAYER_BITS: u32 = 0xabcd_0004;
/// A texel's four words.
const WORDS: [u32; 4] = [0xa0, 0xb1, 0xc2, 0xd3];

/// Each dimension's coordinate registers and the coordinates read from them: integers, an array
/// index last.
fn dimensions() -> [(ImageDim, TextureType, Vec<(u8, u32)>, Val); 6] {
    [
        (ImageDim::D1, TextureType::Color1D, vec![(10, X)], Val::U32(X)),
        (
            ImageDim::Buffer1D,
            TextureType::Buffer,
            vec![(10, X)],
            Val::U32(X),
        ),
        (
            ImageDim::Array1D,
            TextureType::ColorArray1D,
            vec![(10, X), (11, LAYER_BITS)],
            Val::U32x2([X, 4]),
        ),
        (
            ImageDim::D2,
            TextureType::Color2D,
            vec![(10, X), (11, Y)],
            Val::U32x2([X, Y]),
        ),
        (
            ImageDim::Array2D,
            TextureType::ColorArray2D,
            vec![(10, X), (11, Y), (12, LAYER_BITS)],
            Val::U32x3([X, Y, 4]),
        ),
        (
            ImageDim::D3,
            TextureType::Color3D,
            vec![(10, X), (11, Y), (12, Z)],
            Val::U32x3([X, Y, Z]),
        ),
    ]
}

fn surface_info(ty: TextureType, image_format: ImageFormat) -> TextureInfo {
    TextureInfo {
        ty,
        image_format,
        ..TextureInfo::default()
    }
}

/// A formatted load of every dimension reads the texel's selected channels into consecutive
/// registers; the fault predicate reports resident.
#[test]
fn formatted_loads() {
    for (dim, ty, mut regs, coords) in dimensions() {
        regs.push((30, HANDLE_BITS));
        let op = OpSuLd { dim, ..suld() };
        let machine = run_op(
            Op::SuLd(op),
            &regs,
            vec![call(
                Opcode::ImageRead,
                surface_info(ty, ImageFormat::Typeless),
                vec![HANDLE, coords],
                Some(Val::U32x4(WORDS)),
            )],
        );
        assert_eq!(machine.regs[..4], WORDS, "{dim}");
    }
    let op = OpSuLd {
        access: ImageAccess::Formatted(mask(0xa)),
        fault: p(3),
        ..suld()
    };
    let function = translated(Op::SuLd(op));
    let mut machine = machine_with(&[(10, X), (11, Y), (30, HANDLE_BITS), (2, 9)]);
    machine.preds[3] = true;
    run_script(
        &function,
        &mut machine,
        vec![call(
            Opcode::ImageRead,
            surface_info(TextureType::Color2D, ImageFormat::Typeless),
            vec![HANDLE, Val::U32x2([X, Y])],
            Some(Val::U32x4(WORDS)),
        )],
    );
    assert_eq!(machine.regs[..3], [WORDS[1], WORDS[3], 9]);
    assert!(!machine.preds[3]);
}

/// The memory types of the binary forms: each size's image format and register count (Eden's
/// `Format` and `SizeInRegs`).
const BINARY: [(MemType, ImageFormat, usize); 7] = [
    (MemType::U8, ImageFormat::R8Uint, 1),
    (MemType::I8, ImageFormat::R8Sint, 1),
    (MemType::U16, ImageFormat::R16Uint, 1),
    (MemType::I16, ImageFormat::R16Sint, 1),
    (MemType::B32, ImageFormat::R32Uint, 1),
    (MemType::B64, ImageFormat::R32G32Uint, 2),
    (MemType::B128, ImageFormat::R32G32B32A32Uint, 4),
];

/// A binary load reads the words its size covers into consecutive registers.
#[test]
fn binary_loads() {
    for (mem_type, format, count) in BINARY {
        let op = OpSuLd {
            access: ImageAccess::Binary(mem_type),
            ..suld()
        };
        let mut regs = vec![(10, X), (11, Y), (30, HANDLE_BITS)];
        regs.extend((0..4).map(|index| (index, 9)));
        let machine = run_op(
            Op::SuLd(op),
            &regs,
            vec![call(
                Opcode::ImageRead,
                surface_info(TextureType::Color2D, format),
                vec![HANDLE, Val::U32x2([X, Y])],
                Some(Val::U32x4(WORDS)),
            )],
        );
        let mut expected = [9; 4];
        expected[..count].copy_from_slice(&WORDS[..count]);
        assert_eq!(machine.regs[..4], expected, "{mem_type:?}");
    }
}

/// A formatted store writes the data registers its mask covers and zero for the rest; a binary
/// store the registers its size covers.
#[test]
fn stores() {
    let data = [(20, WORDS[0]), (21, WORDS[1]), (22, WORDS[2]), (23, WORDS[3])];
    let mut cases: Vec<(ImageAccess, ImageFormat, usize)> = SURFACE_STORE_MASKS
        .iter()
        .map(|&bits| {
            let count = usize::try_from(bits.count_ones()).expect("small");
            (ImageAccess::Formatted(mask(bits)), ImageFormat::Typeless, count)
        })
        .collect();
    cases.extend(
        BINARY
            .iter()
            .map(|&(mem_type, format, count)| (ImageAccess::Binary(mem_type), format, count)),
    );
    for (access, format, count) in cases {
        let op = OpSuSt { access, ..sust() };
        let mut regs = vec![(10, X), (11, Y), (30, HANDLE_BITS)];
        regs.extend(data);
        let mut color = [0; 4];
        color[..count].copy_from_slice(&WORDS[..count]);
        run_op(
            Op::SuSt(op),
            &regs,
            vec![call(
                Opcode::ImageWrite,
                surface_info(TextureType::Color2D, format),
                vec![HANDLE, Val::U32x2([X, Y]), Val::U32x4(color)],
                None,
            )],
        );
    }
    for (dim, ty, mut regs, coords) in dimensions() {
        regs.push((30, HANDLE_BITS));
        regs.extend(data);
        run_op(
            Op::SuSt(OpSuSt { dim, ..sust() }),
            &regs,
            vec![call(
                Opcode::ImageWrite,
                surface_info(ty, ImageFormat::Typeless),
                vec![HANDLE, coords, Val::U32x4(WORDS)],
                None,
            )],
        );
    }
}

/// Every operation of both types maps to its 32-bit image atomic, the minimum and maximum
/// signed for `S32`, with the data register as the value and the old value written back.
#[test]
fn atomics() {
    let cases = [
        (AtomOp::Add, AtomType::U32, Opcode::ImageAtomicIAdd32),
        (AtomOp::Add, AtomType::I32, Opcode::ImageAtomicIAdd32),
        (AtomOp::Min, AtomType::U32, Opcode::ImageAtomicUMin32),
        (AtomOp::Min, AtomType::I32, Opcode::ImageAtomicSMin32),
        (AtomOp::Max, AtomType::U32, Opcode::ImageAtomicUMax32),
        (AtomOp::Max, AtomType::I32, Opcode::ImageAtomicSMax32),
        (AtomOp::Inc, AtomType::U32, Opcode::ImageAtomicInc32),
        (AtomOp::Dec, AtomType::U32, Opcode::ImageAtomicDec32),
        (AtomOp::And, AtomType::U32, Opcode::ImageAtomicAnd32),
        (AtomOp::Or, AtomType::I32, Opcode::ImageAtomicOr32),
        (AtomOp::Xor, AtomType::U32, Opcode::ImageAtomicXor32),
        (AtomOp::Exch, AtomType::I32, Opcode::ImageAtomicExchange32),
    ];
    for (atom_op, atom_type, opcode) in cases {
        let op = OpSuAtom {
            atom_op,
            atom_type,
            ..suatom()
        };
        let machine = run_op(
            Op::SuAtom(op),
            &[(10, X), (11, Y), (20, 0x77), (30, HANDLE_BITS)],
            vec![call(
                opcode,
                surface_info(TextureType::Color2D, ImageFormat::Typeless),
                vec![HANDLE, Val::U32x2([X, Y]), Val::U32(0x77)],
                Some(Val::U32(0x1234)),
            )],
        );
        assert_eq!(machine.regs[0], 0x1234, "{atom_op:?} {atom_type:?}");
    }
    for (dim, ty, mut regs, coords) in dimensions() {
        regs.extend([(20, 0x77), (30, HANDLE_BITS)]);
        let machine = run_op(
            Op::SuAtom(OpSuAtom { dim, ..suatom() }),
            &regs,
            vec![call(
                Opcode::ImageAtomicIAdd32,
                surface_info(ty, ImageFormat::Typeless),
                vec![HANDLE, coords, Val::U32(0x77)],
                Some(Val::U32(0x1234)),
            )],
        );
        assert_eq!(machine.regs[0], 0x1234, "{dim}");
    }
}

/// `SURED` performs the atomic and writes nothing, its destination field included; the fault
/// predicate of `SUATOM` reports resident.
#[test]
fn reductions_discard_the_old_value() {
    let op = OpSuAtom {
        reduce: true,
        atom_op: AtomOp::Xor,
        dst: r(0),
        ..suatom()
    };
    let machine = run_op(
        Op::SuAtom(op),
        &[(0, 9), (10, X), (11, Y), (20, 0x77), (30, HANDLE_BITS)],
        vec![call(
            Opcode::ImageAtomicXor32,
            surface_info(TextureType::Color2D, ImageFormat::Typeless),
            vec![HANDLE, Val::U32x2([X, Y]), Val::U32(0x77)],
            Some(Val::U32(0x1234)),
        )],
    );
    assert_eq!(machine.regs[0], 9);
    let op = OpSuAtom {
        fault: p(6),
        ..suatom()
    };
    let function = translated(Op::SuAtom(op));
    let mut machine = machine_with(&[(10, X), (11, Y), (20, 0x77), (30, HANDLE_BITS)]);
    machine.preds[6] = true;
    run_script(
        &function,
        &mut machine,
        vec![call(
            Opcode::ImageAtomicIAdd32,
            surface_info(TextureType::Color2D, ImageFormat::Typeless),
            vec![HANDLE, Val::U32x2([X, Y]), Val::U32(0x77)],
            Some(Val::U32(0x1234)),
        )],
    );
    assert!(!machine.preds[6]);
}

/// A handle in `RZ` reads zero, as Eden's `X(RZ)` does; surfaces leave the descriptor buffer
/// alone.
#[test]
fn a_handle_in_rz_and_the_descriptor_buffer() {
    let op = OpSuLd {
        handle: Gpr::RZ,
        ..suld()
    };
    run_op(
        Op::SuLd(op),
        &[(10, X), (11, Y)],
        vec![call(
            Opcode::ImageRead,
            surface_info(TextureType::Color2D, ImageFormat::Typeless),
            vec![Val::U32(0), Val::U32x2([X, Y])],
            Some(Val::U32x4(WORDS)),
        )],
    );
    let bound = DescriptorBuffer::new().bind(3).expect("binds");
    for op in [Op::SuLd(suld()), Op::SuSt(sust()), Op::SuAtom(suatom())] {
        let (_, buffer) = translate_raw(Instr::new(op), bound).expect("translates");
        assert_eq!(buffer, bound);
    }
}
