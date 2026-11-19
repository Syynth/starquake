//! Local loads and stores: words indexed by the byte address shifted right by 2, narrow
//! accesses within their word.

use super::*;

/// A machine with eight words of local memory, distinct values.
fn machine() -> Machine {
    let mut machine = Machine::new();
    machine.local = (0..8).map(|k| 0x1000_0000 * k + 0x0403_0201).collect();
    machine
}

/// `LDL` of every type from `R2 + 4`: a narrow load extracts its byte or half from the word at
/// its address and zero- or sign-extends it; a wide one reads consecutive words.
#[test]
fn ldl_reads_every_type() {
    for (byte, mem_type, expected) in [
        (5_u32, MemType::U8, vec![0x02]),
        (7, MemType::U8, vec![0x14]),
        (7, MemType::I8, vec![0x14]),
        (6, MemType::U16, vec![0x1403]),
        (4, MemType::B32, vec![0x1403_0201]),
        (8, MemType::B64, vec![0x2403_0201, 0x3403_0201]),
        (
            16,
            MemType::B128,
            vec![0x4403_0201, 0x5403_0201, 0x6403_0201, 0x7403_0201],
        ),
    ] {
        let mut machine = machine();
        set_reg(&mut machine, 2, byte - 4);
        run_op(Op::Ld(ldl(8, mem_type, 2, 4)), &mut machine);
        for (k, value) in (0_u8..).zip(&expected) {
            assert_eq!(
                machine.regs[usize::from(8 + k)],
                *value,
                "{mem_type:?} at {byte}"
            );
        }
    }
}

/// A signed narrow load extends the field's top bit.
#[test]
fn signed_narrow_loads_extend_their_sign() {
    let mut machine = Machine::new();
    machine.local = vec![0x80f0_7f81];
    for (byte, mem_type, expected) in [
        (0_i32, MemType::I8, 0xffff_ff81_u32),
        (1, MemType::I8, 0x7f),
        (3, MemType::I8, 0xffff_ff80),
        (0, MemType::I16, 0x7f81),
        (2, MemType::I16, 0xffff_80f0),
        (2, MemType::U16, 0x80f0),
    ] {
        run_op(Op::Ld(ldl(1, mem_type, 255, byte)), &mut machine);
        assert_eq!(machine.regs[1], expected, "{mem_type:?} at {byte}");
    }
}

/// `STL` of a byte or half reads the word, inserts its bits at the address and writes the word
/// back; the other bytes of the word keep their values. Wider stores write whole words.
#[test]
fn stl_writes_every_type() {
    for (byte, mem_type, expected) in [
        (5_i32, MemType::U8, vec![(1, 0x1403_cc01)]),
        (7, MemType::I8, vec![(1, 0xcc03_0201)]),
        (6, MemType::U16, vec![(1, 0xbbcc_0201)]),
        (6, MemType::I16, vec![(1, 0xbbcc_0201)]),
        (4, MemType::B32, vec![(1, 0xaabb_bbcc)]),
        (8, MemType::B64, vec![(2, 0xaabb_bbcc), (3, 0x1122_3344)]),
        (
            16,
            MemType::B128,
            vec![
                (4, 0xaabb_bbcc),
                (5, 0x1122_3344),
                (6, 0x5566_7788),
                (7, 0x99aa_0000),
            ],
        ),
    ] {
        let mut machine = machine();
        let data = [0xaabb_bbcc_u32, 0x1122_3344, 0x5566_7788, 0x99aa_0000];
        for (k, value) in (0_u8..).zip(data) {
            set_reg(&mut machine, 8 + k, value);
        }
        let before = machine.local.clone();
        run_op(
            Op::St(st(8, mem_type, LOCAL, plain(255, false, byte))),
            &mut machine,
        );
        let mut after = before;
        for &(word, value) in &expected {
            after[word] = value;
        }
        assert_eq!(machine.local, after, "{mem_type:?} at {byte}");
    }
}

/// A store from `RZ` writes zero words.
#[test]
fn a_store_from_rz_writes_zeros() {
    let mut machine = machine();
    let op = OpSt {
        data: Gpr::RZ,
        ..st(0, MemType::B64, LOCAL, plain(255, false, 8))
    };
    run_op(Op::St(op), &mut machine);
    assert_eq!(machine.local[2..4], [0, 0]);
    assert_eq!(machine.local[4], 0x4403_0201);
}

/// The word index is the byte address shifted right arithmetically: a negative register plus
/// offset gives a negative index, which the interpreter refuses as out of bounds rather than
/// reading word `0x3fff_ffff`.
#[test]
fn the_word_index_shifts_arithmetically() {
    let function = build_op(Op::Ld(ldl(1, MemType::B32, 2, -4))).expect("translates");
    let mut machine = machine();
    set_reg(&mut machine, 2, 0);
    assert!(matches!(
        machine.run(&function),
        Err(InterpError::OutOfBounds {
            offset: 0x3_ffff_fffc,
            ..
        })
    ));
}

/// `LDL` reads with no bound: PR #4588 selects zero past the local memory size
/// (`env.LocalMemorySize()`), which this slice leaves to slice 16's pass 6 (the module docs'
/// divergence table), so the IR is the load alone.
#[test]
fn ldl_reads_without_a_bound() {
    let function = build_op(Op::Ld(ldl(1, MemType::B32, 2, 4))).expect("translates");
    assert_eq!(
        function.to_string(),
        "block0:\n\
         \x20   %0: u32 = GetRegister r2\n\
         \x20   %1: u32 = IAdd32 %0, u32 0x4\n\
         \x20   %2: u32 = ShiftRightArithmetic32 %1, u32 0x2\n\
         \x20   %3: u32 = LoadLocal %2\n\
         \x20   %4 = SetRegister r1, %3\n\
         \x20   return\n"
    );
}
