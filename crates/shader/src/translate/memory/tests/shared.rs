//! Shared loads and stores and shared addresses (`Visitor::SharedOffset`).

use super::*;

/// A machine with `len` bytes of shared memory, every byte `0xa5`.
fn machine(len: usize) -> Machine {
    let mut machine = Machine::new();
    machine.shared = vec![0xa5; len];
    machine
}

/// `LDS` of every type from `R2 + 0x10`: the narrow types zero- or sign-extend, the wide ones
/// fill a pair or quad.
#[test]
fn lds_reads_every_type() {
    let bytes: [u8; 16] = [
        0x81, 0xf2, 0x13, 0x84, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff,
        0x10,
    ];
    let word = |k: usize| u32::from_le_bytes(bytes[4 * k..4 * k + 4].try_into().expect("4"));
    let cases: [(MemType, &[u32]); 7] = [
        (MemType::U8, &[0x81]),
        (MemType::I8, &[0xffff_ff81]),
        (MemType::U16, &[0xf281]),
        (MemType::I16, &[0xffff_f281]),
        (MemType::B32, &[word(0)]),
        (MemType::B64, &[word(0), word(1)]),
        (MemType::B128, &[word(0), word(1), word(2), word(3)]),
    ];
    for (mem_type, expected) in cases {
        let mut machine = machine(0x40);
        machine.shared[0x20..0x30].copy_from_slice(&bytes);
        set_reg(&mut machine, 2, 0x10);
        for k in 0..4 {
            set_reg(&mut machine, 8 + k, 0xdead_0000 + u32::from(k));
        }
        run_op(
            Op::Ld(lds(8, mem_type, 2, OffsetStride::X1, 0x10)),
            &mut machine,
        );
        for (k, value) in (0_u8..).zip(expected) {
            assert_eq!(
                machine.regs[usize::from(8 + k)],
                *value,
                "{mem_type:?} word {k}"
            );
        }
        let words = u8::try_from(expected.len()).expect("small");
        for k in words..4 {
            assert_eq!(machine.regs[usize::from(8 + k)], 0xdead_0000 + u32::from(k));
        }
    }
}

/// `STS` of every type writes exactly its bytes; the next byte keeps its value.
#[test]
fn sts_writes_every_type() {
    let data = [0x8899_aabb_u32, 0x1122_3344, 0x5566_7788, 0x99aa_bbcc];
    let written: Vec<u8> = data.iter().flat_map(|word| word.to_le_bytes()).collect();
    for (mem_type, len) in [
        (MemType::U8, 1),
        (MemType::I8, 1),
        (MemType::U16, 2),
        (MemType::I16, 2),
        (MemType::B32, 4),
        (MemType::B64, 8),
        (MemType::B128, 16),
    ] {
        let mut machine = machine(0x40);
        set_reg(&mut machine, 2, 0x10);
        for (k, value) in (0_u8..).zip(data) {
            set_reg(&mut machine, 8 + k, value);
        }
        let op = st(
            8,
            mem_type,
            MemSpace::Shared(OffsetStride::X1),
            plain(2, false, 0x10),
        );
        run_op(Op::St(op), &mut machine);
        assert_eq!(
            &machine.shared[0x20..0x20 + len],
            &written[..len],
            "{mem_type:?}"
        );
        assert_eq!(machine.shared[0x20 + len], 0xa5, "{mem_type:?}");
    }
}

/// The address register is scaled by its stride: `X4`, `X8` and `X16` shift it left by 2, 3
/// and 4 ("Native paired shared values keep both words under scaled addressing").
#[test]
fn strides_scale_the_register() {
    for (stride, scale) in [
        (OffsetStride::X1, 1),
        (OffsetStride::X4, 4),
        (OffsetStride::X8, 8),
        (OffsetStride::X16, 16),
    ] {
        let mut machine = machine(0x200);
        set_reg(&mut machine, 19, 20);
        let address = 20 * scale + 8;
        put_shared_word(&mut machine, address, 0x1234_5678);
        run_op(Op::Ld(lds(1, MemType::B32, 19, stride, 8)), &mut machine);
        assert_eq!(machine.regs[1], 0x1234_5678, "{stride:?}");
    }
}

/// With a register, the offset is signed: `R2 - 4`.
#[test]
fn a_register_offset_is_signed() {
    let mut machine = machine(0x40);
    set_reg(&mut machine, 2, 0x20);
    put_shared_word(&mut machine, 0x1c, 0x0f0f_0f0f);
    run_op(
        Op::Ld(lds(1, MemType::B32, 2, OffsetStride::X1, -4)),
        &mut machine,
    );
    assert_eq!(machine.regs[1], 0x0f0f_0f0f);
}

/// With `RZ`, the 24-bit offset field is read unsigned, as PR #4588's `Visitor::SharedOffset`
/// returns it (`ir.Imm32(offset)`), where the decoder sign-extends it: the field `0x800000` is
/// the address `0x800000`, not `-0x800000`, and the stride does not apply.
#[test]
fn an_rz_offset_is_read_unsigned() {
    let function = build_op(Op::Ld(lds(
        1,
        MemType::B32,
        255,
        OffsetStride::X16,
        -0x80_0000,
    )))
    .expect("translates");
    assert_eq!(
        function.to_string(),
        "block0:\n\
         \x20   %0: u32 = LoadSharedU32 u32 0x800000\n\
         \x20   %1 = SetRegister r1, %0\n\
         \x20   return\n"
    );
    let mut machine = machine(0x40);
    put_shared_word(&mut machine, 0x30, 0x7777_1111);
    run_op(
        Op::Ld(lds(1, MemType::B32, 255, OffsetStride::X16, 0x30)),
        &mut machine,
    );
    assert_eq!(machine.regs[1], 0x7777_1111);
}

/// A store and a load of a pair at the same scaled address round-trip both words.
#[test]
fn a_pair_round_trips() {
    let mut machine = machine(0x200);
    set_reg(&mut machine, 19, 17);
    set_reg(&mut machine, 4, 0x1234_5678);
    set_reg(&mut machine, 5, 0x9abc_def0);
    let stride = OffsetStride::X8;
    let store = st(
        4,
        MemType::B64,
        MemSpace::Shared(stride),
        plain(19, false, 0),
    );
    run(
        &[
            Instr::new(Op::St(store)),
            Instr::new(Op::Ld(lds(2, MemType::B64, 19, stride, 0))),
        ],
        &mut machine,
    );
    assert_eq!(shared_word(&machine, 17 * 8), 0x1234_5678);
    assert_eq!(shared_word(&machine, 17 * 8 + 4), 0x9abc_def0);
    assert_eq!(
        [machine.regs[2], machine.regs[3]],
        [0x1234_5678, 0x9abc_def0]
    );
}

/// A store from `RZ` writes zeros in every word ("Native shared vector zero stores preserve
/// their address and predicates").
#[test]
fn a_store_from_rz_writes_zeros() {
    let mut machine = machine(0x40);
    set_reg(&mut machine, 2, 0x10);
    let op = OpSt {
        data: Gpr::RZ,
        ..st(
            0,
            MemType::B128,
            MemSpace::Shared(OffsetStride::X1),
            plain(2, false, 0),
        )
    };
    run_op(Op::St(op), &mut machine);
    assert!(machine.shared[0x10..0x20].iter().all(|&byte| byte == 0));
    assert_eq!(machine.shared[0x20], 0xa5);
}
