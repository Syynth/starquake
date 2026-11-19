//! `LDC`: every type, the signed offset, and which reads are indexed.

use super::*;
use crate::ir::Inst;

fn ldc(dst: u8, mem_type: MemType, bank: u8, offset_reg: u8, offset: u16) -> OpLdc {
    OpLdc {
        dst: r(dst),
        cbuf: cbuf(bank, offset),
        offset: r(offset_reg),
        mode: LdcMode::Indexed,
        mem_type,
    }
}

/// A machine with bank 31 holding 16 distinct words.
fn machine() -> Machine {
    let mut machine = Machine::new();
    let words: Vec<u32> = (0..16).map(|k| 0x8000_00f1 + 0x0101_0000 * k).collect();
    with_cbuf(&mut machine, 31, &words);
    machine
}

/// `LDC` of every type from the last bank (31: the 32 slots of the five-bit field) at `R6 +
/// 0x8`: narrow types zero- or sign-extend; `.64` fills a pair and `.128` a quad.
#[test]
fn ldc_reads_every_type() {
    let word = |k: u32| 0x8000_00f1_u32 + 0x0101_0000 * k;
    for (mem_type, expected) in [
        (MemType::U8, vec![0xf1]),
        (MemType::I8, vec![0xffff_fff1]),
        (MemType::U16, vec![0xf1]),
        (MemType::I16, vec![0xf1]),
        (MemType::B32, vec![word(2)]),
        (MemType::B64, vec![word(2), word(3)]),
        (MemType::B128, vec![word(2), word(3), word(4), word(5)]),
    ] {
        let mut machine = machine();
        set_reg(&mut machine, 6, 4);
        let dst = if mem_type == MemType::B128 { 8 } else { 10 };
        run_op(Op::Ldc(ldc(dst, mem_type, 31, 6, 4)), &mut machine);
        for (k, value) in (0_u8..).zip(&expected) {
            assert_eq!(
                machine.regs[usize::from(dst + k)],
                *value,
                "{mem_type:?} word {k}"
            );
        }
    }
}

/// A signed 16-bit read takes the half's top bit.
#[test]
fn a_signed_half_extends_its_sign() {
    let mut machine = Machine::new();
    with_cbuf(&mut machine, 2, &[0x8001_7ffe]);
    run_op(Op::Ldc(ldc(1, MemType::I16, 2, 255, 2)), &mut machine);
    assert_eq!(machine.regs[1], 0xffff_8001);
    run_op(Op::Ldc(ldc(1, MemType::U16, 2, 255, 2)), &mut machine);
    assert_eq!(machine.regs[1], 0x8001);
}

/// The 16-bit offset is signed, as PR #4588's `Visitor::LDC` reads it
/// (`static_cast<s16>(Take(38, 16))`), where the decoder keeps it unsigned: `c[3][R48 +
/// 0xfe05]` with `R48 = 0x800` reads at `0x800 - 0x1fb` (aligned down to
/// `0x604`; read unsigned, `0x10605` would be past the buffer), and with `RZ` the field
/// `0xfffc` is the offset `0xfffffffc`.
#[test]
fn ldc_offsets_are_signed() {
    let mut machine = Machine::new();
    let mut words = vec![0; 0x200];
    words[0x604 / 4] = 0x1234_5678;
    with_cbuf(&mut machine, 3, &words);
    set_reg(&mut machine, 48, 0x800);
    run_op(Op::Ldc(ldc(7, MemType::B32, 3, 48, 0xfe05)), &mut machine);
    assert_eq!(machine.regs[7], 0x1234_5678);
    let function = build_op(Op::Ldc(ldc(1, MemType::B32, 3, 255, 0xfffc))).expect("translates");
    assert_eq!(
        function.to_string(),
        "block0:\n\
         \x20   %0: u32 = GetCbufU32 u32 0x3, u32 0xfffffffc\n\
         \x20   %1 = SetRegister r1, %0\n\
         \x20   return\n"
    );
}

/// A read with `RZ` as the offset register has an immediate offset; with any other register
/// it has a value, so slice 13 can tell the indexed reads it bounds-checks.
#[test]
fn only_a_register_offset_is_indexed() {
    for (offset_reg, indexed) in [(255, false), (0, true), (6, true)] {
        for mem_type in [MemType::U8, MemType::B32, MemType::B64, MemType::B128] {
            let function =
                build_op(Op::Ldc(ldc(8, mem_type, 1, offset_reg, 0x10))).expect("translates");
            let reads: Vec<bool> = function
                .listed()
                .filter_map(|(_, inst)| match inst {
                    Inst::Op(op) if op.opcode.name().starts_with("GetCbuf") => {
                        Some(matches!(op.args.get(1), Some(Operand::Value(_))))
                    }
                    _ => None,
                })
                .collect();
            assert_ne!(reads.len(), 0);
            assert!(
                reads.iter().all(|&read| read == indexed),
                "{offset_reg} {mem_type:?}"
            );
        }
    }
}
