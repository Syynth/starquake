//! `ULDC`: every type, the bound and indexed forms, and the offsets the lowering reads as
//! signed.

use super::*;

/// Constant buffer `bank` holding `words` from offset 0, little-endian.
fn with_cbuf(machine: &mut Machine, bank: u32, words: &[u32]) {
    let bytes = words.iter().flat_map(|word| word.to_le_bytes()).collect();
    machine.cbufs.insert(bank, bytes);
}

/// A bound `ULDC` of `mem_type` from buffer `bank` at `offset` into `dst`.
fn bound(dst: u8, bank: u8, offset: u16, mem_type: MemType) -> Op {
    Op::ULdc(OpULdc {
        dst: ur(dst),
        src: ULdcSrc::CBuf(CBufRef {
            buf: CBuf::Binding(bank),
            offset,
        }),
        mem_type,
    })
}

/// An indexed `ULDC` (`0xabb`) of `mem_type` from buffer `bank` at `reg` plus `offset`.
fn indexed(dst: u8, bank: u8, reg: UniformReg, offset: u16, mem_type: MemType) -> Op {
    Op::ULdc(OpULdc {
        dst: ur(dst),
        src: ULdcSrc::Indexed {
            index: bank,
            reg,
            offset,
        },
        mem_type,
    })
}

/// The words the tests' buffers hold: each byte distinct, the high bit set in some bytes and
/// half-words and clear in others.
const WORDS: [u32; 6] = [
    0x8381_8280,
    0x0706_0504,
    0x8b8a_7f89,
    0xfedc_ba98,
    0x7654_3210,
    0x1357_9bdf,
];

/// Every type of the bound form, from buffer 5 at offset 8: the 8- and 16-bit ones zero- or
/// sign-extended, 32 bits as they are, 64 bits into a pair and 128 bits into a quad.
#[test]
fn every_type_of_the_bound_form() {
    let cases: [(MemType, &[u32]); 7] = [
        (MemType::U8, &[0x89]),
        (MemType::I8, &[0xffff_ff89]),
        (MemType::U16, &[0x7f89]),
        (MemType::I16, &[0x7f89]),
        (MemType::B32, &[WORDS[2]]),
        (MemType::B64, &[WORDS[2], WORDS[3]]),
        (MemType::B128, &[WORDS[2], WORDS[3], WORDS[4], WORDS[5]]),
    ];
    for (mem_type, expected) in cases {
        let mut machine = machine_with(&[]);
        with_cbuf(&mut machine, 5, &WORDS);
        run_op(bound(8, 5, 8, mem_type), &mut machine);
        for (k, word) in expected.iter().enumerate() {
            assert_eq!(machine.uniform_regs[8 + k], *word, "{mem_type:?} {k}");
        }
        let next = 8 + expected.len();
        assert_eq!(machine.uniform_regs[next], 0, "{mem_type:?} writes no more");
        assert_eq!(machine.regs[8], 0xbad0_0008, "{mem_type:?}");
    }
}

/// The 8- and 16-bit types with the high bit of the element set: sign-extended or not.
#[test]
fn narrow_types_extend_by_their_sign() {
    let cases = [
        (MemType::U8, 0x80),
        (MemType::I8, 0xffff_ff80),
        (MemType::U16, 0x8280),
        (MemType::I16, 0xffff_8280),
    ];
    for (mem_type, expected) in cases {
        let mut machine = machine_with(&[]);
        with_cbuf(&mut machine, 2, &WORDS);
        run_op(bound(1, 2, 0, mem_type), &mut machine);
        assert_eq!(machine.uniform_regs[1], expected, "{mem_type:?}");
    }
}

/// The indexed form reads at the uniform register plus the offset, every type; with `URZ` the
/// register reads zero.
#[test]
fn the_indexed_form_adds_its_register() {
    let cases: [(MemType, &[u32]); 3] = [
        (MemType::B32, &[WORDS[3]]),
        (MemType::B64, &[WORDS[2], WORDS[3]]),
        (MemType::B128, &[WORDS[2], WORDS[3], WORDS[4], WORDS[5]]),
    ];
    for (mem_type, expected) in cases {
        let mut machine = machine_with(&[]);
        machine.uniform_regs[9] = 4;
        with_cbuf(&mut machine, 31, &WORDS);
        // A 64-bit read is made at the offset aligned down to 8 (Eden's `GetCbuf`, the
        // interpreter's rule), so the wide cases read at 8.
        let offset = if mem_type == MemType::B32 { 8 } else { 4 };
        run_op(indexed(12, 31, ur(9), offset, mem_type), &mut machine);
        for (k, word) in expected.iter().enumerate() {
            assert_eq!(machine.uniform_regs[12 + k], *word, "{mem_type:?} {k}");
        }
    }
    let mut machine = machine_with(&[]);
    with_cbuf(&mut machine, 31, &WORDS);
    run_op(
        indexed(1, 31, UniformReg::URZ, 4, MemType::B32),
        &mut machine,
    );
    assert_eq!(machine.uniform_regs[1], WORDS[1]);
}

/// The lowering reads the offset of `0xabb` with a 32- or 64-bit type, and of `0xab9` with a
/// 64-bit type, as signed (its Maxwell `LDC`); the other types keep the native, unsigned,
/// reading on both opcodes (the lowering refuses them on `0xabb` and reads `0xab9` `.32`
/// unsigned). Pinned on the IR's offsets for `0x8000` and `0x7ffc` for every type of `0xabb`,
/// and on the interpreter for an indexed 32-bit read that wraps back into the buffer and an
/// indexed 16-bit read that does not.
#[test]
fn the_lowering_s_offsets_are_signed() {
    let unsigned_indexed = [MemType::U8, MemType::I8, MemType::U16, MemType::I16];
    for mem_type in unsigned_indexed {
        let text = text(&[Instr::new(indexed(2, 5, ur(9), 0x8000, mem_type))]);
        assert!(
            text.contains("IAdd32 %0, u32 0x8000\n"),
            "{mem_type:?}: {text}"
        );
    }
    let quad = text(&[Instr::new(indexed(4, 5, ur(9), 0x8000, MemType::B128))]);
    assert!(quad.contains("IAdd32 %0, u32 0x8000\n"), "{quad}");
    assert!(quad.contains("IAdd32 %0, u32 0x800c\n"), "{quad}");
    for (offset, expected) in [(0x8000, "0xffff8000"), (0x7ff8, "0x7ff8")] {
        let pair = text(&[Instr::new(indexed(2, 5, ur(9), offset, MemType::B64))]);
        assert!(
            pair.contains(&format!("IAdd32 %0, u32 {expected}\n")),
            "{pair}"
        );
    }
    let mut machine = machine_with(&[]);
    machine.uniform_regs[9] = 8;
    let mut bytes = vec![0; usize::from(u16::MAX) + 1];
    bytes[0x8008..0x800a].copy_from_slice(&0x9abc_u16.to_le_bytes());
    machine.cbufs.insert(5, bytes);
    run_op(indexed(2, 5, ur(9), 0x8000, MemType::U16), &mut machine);
    assert_eq!(machine.uniform_regs[2], 0x9abc);
    let cases = [
        (
            bound(2, 5, 0x8000, MemType::B32),
            "GetCbufU32 u32 0x5, u32 0x8000",
        ),
        (
            bound(2, 5, 0x8000, MemType::U16),
            "GetCbufU16 u32 0x5, u32 0x8000",
        ),
        (
            bound(4, 5, 0x8000, MemType::B128),
            "GetCbufU32 u32 0x5, u32 0x8000",
        ),
        (
            bound(2, 5, 0x8000, MemType::B64),
            "GetCbufU32x2 u32 0x5, u32 0xffff8000",
        ),
        (
            bound(2, 5, 0x7ffc, MemType::B64),
            "GetCbufU32x2 u32 0x5, u32 0x7ffc",
        ),
        (
            indexed(2, 5, ur(9), 0x8000, MemType::B32),
            "IAdd32 %0, u32 0xffff8000",
        ),
        (
            indexed(2, 5, ur(9), 0x7ffc, MemType::B32),
            "IAdd32 %0, u32 0x7ffc",
        ),
    ];
    for (op, expected) in cases {
        let text = text(&[Instr::new(op)]);
        assert!(text.contains(expected), "{op}: {text}");
    }
    let mut machine = machine_with(&[]);
    machine.uniform_regs[9] = 0x8008;
    with_cbuf(&mut machine, 5, &WORDS);
    run_op(indexed(2, 5, ur(9), 0x8000, MemType::B32), &mut machine);
    assert_eq!(machine.uniform_regs[2], WORDS[2]);
}

/// The 32-bit form at the highest word-aligned offset reads the last word of a full buffer.
#[test]
fn the_last_word_of_a_buffer() {
    let mut machine = machine_with(&[]);
    let mut bytes = vec![0; usize::from(u16::MAX) + 1];
    bytes[0xfffc..].copy_from_slice(&0xcafe_f00d_u32.to_le_bytes());
    machine.cbufs.insert(7, bytes);
    run_op(bound(3, 7, 0xfffc, MemType::B32), &mut machine);
    assert_eq!(machine.uniform_regs[3], 0xcafe_f00d);
}

/// A 32-bit read into `URZ` is dropped.
#[test]
fn a_read_into_urz_is_dropped() {
    let function = build(&[Instr::new(Op::ULdc(OpULdc {
        dst: UniformReg::URZ,
        src: ULdcSrc::CBuf(CBufRef {
            buf: CBuf::Binding(1),
            offset: 0,
        }),
        mem_type: MemType::B32,
    }))])
    .expect("translates");
    assert_eq!(count(&function, Opcode::SetUniformRegister), 0);
}
