//! Global loads and stores, global addresses (`Visitor::GlobalAddress`), and the
//! uniform-address bits of each form.

use super::*;

/// The base of the global region the cases use: above 4 GiB, so a lost high word shows.
const BASE: u64 = 0x21_0000_0000;
/// The region's length.
const LEN: usize = 0x100;

/// A machine with the global region, every byte `0xa5`.
fn machine() -> Machine {
    let mut machine = Machine::new();
    global_region(&mut machine, BASE, LEN, 0xa5);
    machine
}

/// `R2:R3` holding [`BASE`] plus `low`.
fn pointer(machine: &mut Machine, low: u32) {
    let address = BASE + u64::from(low);
    set_reg(machine, 2, lo(address));
    set_reg(machine, 3, hi(address));
}

fn lo(value: u64) -> u32 {
    u32::try_from(value & u64::from(u32::MAX)).unwrap_or(0)
}

fn hi(value: u64) -> u32 {
    lo(value >> 32)
}

/// `LDG` of every type from `R2:R3 + 0x10`: the narrow types zero- or sign-extend their byte or
/// half, the wide ones fill a pair or quad.
#[test]
fn ldg_reads_every_type() {
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
        let mut machine = machine();
        machine.global.bytes[0x10..0x20].copy_from_slice(&bytes);
        pointer(&mut machine, 0);
        for k in 0..4 {
            set_reg(&mut machine, 8 + k, 0xdead_0000 + u32::from(k));
        }
        run_op(Op::Ld(ldg(8, mem_type, plain(2, true, 0x10))), &mut machine);
        for (k, value) in (0_u8..).zip(expected) {
            assert_eq!(
                machine.regs[usize::from(8 + k)],
                *value,
                "{mem_type:?} word {k}"
            );
        }
        // The registers past the access keep their values.
        let words = u8::try_from(expected.len()).expect("small");
        for k in words..4 {
            assert_eq!(
                machine.regs[usize::from(8 + k)],
                0xdead_0000 + u32::from(k),
                "{mem_type:?} register {k}"
            );
        }
    }
}

/// `STG` of every type to `R2:R3 + 0x10` writes exactly its bytes, little-endian; the bytes
/// after it keep their values.
#[test]
fn stg_writes_every_type() {
    let data = [0x8899_aabb_u32, 0x1122_3344, 0x5566_7788, 0x99aa_bbcc];
    let cases = [
        (MemType::U8, 1),
        (MemType::I8, 1),
        (MemType::U16, 2),
        (MemType::I16, 2),
        (MemType::B32, 4),
        (MemType::B64, 8),
        (MemType::B128, 16),
    ];
    for (mem_type, len) in cases {
        let mut machine = machine();
        pointer(&mut machine, 0);
        for (k, value) in (0_u8..).zip(data) {
            set_reg(&mut machine, 8 + k, value);
        }
        run_op(
            Op::St(st(8, mem_type, GLOBAL, plain(2, true, 0x10))),
            &mut machine,
        );
        let written: Vec<u8> = data.iter().flat_map(|word| word.to_le_bytes()).collect();
        assert_eq!(
            &machine.global.bytes[0x10..0x10 + len],
            &written[..len],
            "{mem_type:?}"
        );
        assert!(
            machine.global.bytes[0x10 + len..0x30]
                .iter()
                .all(|&byte| byte == 0xa5),
            "{mem_type:?}"
        );
    }
}

/// A store from `RZ` writes zeros, for one word and for a quad.
#[test]
fn a_store_from_rz_writes_zeros() {
    for (mem_type, len) in [(MemType::B32, 4), (MemType::B128, 16)] {
        let mut machine = machine();
        pointer(&mut machine, 0);
        let op = OpSt {
            data: Gpr::RZ,
            ..st(0, mem_type, GLOBAL, plain(2, true, 0))
        };
        run_op(Op::St(op), &mut machine);
        assert!(machine.global.bytes[..len].iter().all(|&byte| byte == 0));
        assert_eq!(machine.global.bytes[len], 0xa5);
    }
}

/// The offset is sign-extended and added as 64 bits; a low word that carries reaches the high
/// word ("Native scalar global loads retain address carry and signed offsets").
#[test]
fn offsets_are_signed_and_carry_into_the_high_word() {
    for offset in [0, 0x10, -0x10, 0x80, 0xf0, -0x80] {
        let mut machine = Machine::new();
        // R2:R3 = 0x21_ffff_ff80, just below a high-word carry.
        let pointer = 0x21_ffff_ff80_u64;
        set_reg(&mut machine, 2, lo(pointer));
        set_reg(&mut machine, 3, hi(pointer));
        let address = pointer.wrapping_add(sext64(offset));
        global_region(&mut machine, address & !0xff, 0x200, 0);
        put_global_word(&mut machine, address, 0x8765_4321);
        run_op(
            Op::Ld(ldg(9, MemType::B32, plain(2, true, offset))),
            &mut machine,
        );
        assert_eq!(machine.regs[9], 0x8765_4321, "offset {offset}");
    }
}

/// The 24-bit offset field near its edges: `0x7ffffc`, the largest word-aligned offset, adds
/// 8 MiB less 4 bytes, and `0x800000` subtracts 8 MiB.
#[test]
fn the_offset_field_edges_are_signed() {
    for offset in [0x7f_fffc, -0x80_0000] {
        let mut machine = Machine::new();
        let pointer = 0x40_0000_0000_u64;
        set_reg(&mut machine, 2, lo(pointer));
        set_reg(&mut machine, 3, hi(pointer));
        let address = pointer.wrapping_add(sext64(offset));
        global_region(&mut machine, address, 4, 0);
        put_global_word(&mut machine, address, 0x0bad_cafe);
        run_op(
            Op::Ld(ldg(9, MemType::B32, plain(2, true, offset))),
            &mut machine,
        );
        assert_eq!(machine.regs[9], 0x0bad_cafe, "offset {offset:#x}");
    }
}

/// A register without its 64-bit flag is the low word, with a zero high word; `RZ` adds
/// nothing.
#[test]
fn a_single_register_has_a_zero_high_word() {
    let mut machine = Machine::new();
    global_region(&mut machine, 0x1000, 0x10, 0);
    put_global_word(&mut machine, 0x1004, 0x0123_4567);
    set_reg(&mut machine, 2, 0x1000);
    set_reg(&mut machine, 3, 0xffff_ffff);
    run_op(
        Op::Ld(ldg(9, MemType::B32, plain(2, false, 4))),
        &mut machine,
    );
    assert_eq!(machine.regs[9], 0x0123_4567);
    let mut machine = Machine::new();
    global_region(&mut machine, 0, 0x10, 0);
    put_global_word(&mut machine, 8, 0x7654_3210);
    run_op(
        Op::Ld(ldg(9, MemType::B32, plain(255, true, 8))),
        &mut machine,
    );
    assert_eq!(machine.regs[9], 0x7654_3210);
}

/// With the uniform address on, the uniform register (a pair when `.E`) is added to the
/// register (a pair when bit 90) with a carry between the words ("Native global loads combine
/// a uniform pointer with the scalar byte displacement").
#[test]
fn a_uniform_pointer_adds_to_the_register_with_a_carry() {
    for (reg_64, high) in [(false, 0), (true, 1)] {
        for (reg_low, offset) in [(16_u32, 0), (0x10, 0x10), (0x20, -0x10)] {
            let mut machine = Machine::new();
            set_uniform(&mut machine, 4, 0xffff_fff8);
            set_uniform(&mut machine, 5, 0x21);
            set_reg(&mut machine, 32, reg_low);
            set_reg(&mut machine, 33, high);
            let address = 0x21_ffff_fff8_u64
                .wrapping_add(u64::from(reg_low) | (u64::from(high) << 32))
                .wrapping_add(sext64(offset));
            global_region(&mut machine, address & !0xff, 0x200, 0);
            put_global_word(&mut machine, address, 0x8765_4321);
            let addr = with_uniform(32, reg_64, 4, true, true, offset);
            run_op(Op::Ld(ldg(29, MemType::B32, addr)), &mut machine);
            assert_eq!(machine.regs[29], 0x8765_4321, "{reg_64} {reg_low} {offset}");
            assert_eq!(machine.uniform_regs[4], 0xffff_fff8);
            assert_eq!(machine.regs[32], reg_low);
        }
    }
}

/// A uniform register without `.E` is the low word with a zero high word; with `RZ` the uniform
/// register is the whole address.
#[test]
fn a_uniform_word_alone_and_without_its_pair() {
    let mut machine = Machine::new();
    set_uniform(&mut machine, 6, 0x3000);
    set_uniform(&mut machine, 7, 0xffff_ffff);
    global_region(&mut machine, 0x3000, 0x10, 0);
    put_global_word(&mut machine, 0x3008, 0x1357_9bdf);
    let addr = with_uniform(255, false, 6, false, true, 8);
    run_op(Op::Ld(ldg(1, MemType::B32, addr)), &mut machine);
    assert_eq!(machine.regs[1], 0x1357_9bdf);
}

/// In the uniform `LDG`, bit 90 is the register's 64-bit flag and bit 72 the uniform
/// register's, whether or not the uniform address is on (`Visitor::LDG`: `reg_pair = Take(90,
/// 1)`): with the uniform address off, a set bit 72 changes nothing and bit 90 makes the
/// register a pair.
#[test]
fn uniform_forms_read_bit_90_as_the_register_pair() {
    for (reg_64, wide) in [(false, false), (false, true), (true, false), (true, true)] {
        let mut machine = Machine::new();
        set_reg(&mut machine, 2, 0x100);
        set_reg(&mut machine, 3, 0x21);
        set_uniform(&mut machine, 4, 0x5000);
        let address = if reg_64 { 0x21_0000_0100 } else { 0x100 };
        global_region(&mut machine, address, 4, 0);
        put_global_word(&mut machine, address, 0x2468_ace0);
        let addr = with_uniform(2, reg_64, 4, wide, false, 0);
        run_op(Op::Ld(ldg(1, MemType::B32, addr)), &mut machine);
        assert_eq!(machine.regs[1], 0x2468_ace0, "{reg_64} {wide}");
    }
}

/// `RED` reads bit 72 (the decoder's uniform `.E`) as its register's 64-bit flag, as PR #4588's
/// `Visitor::ATOMG` does for `RED` (`reg_pair = Take(72, 1)`).
#[test]
fn red_reads_its_register_pair_from_bit_72() {
    for pair in [false, true] {
        let mut machine = Machine::new();
        set_reg(&mut machine, 2, 0x40);
        set_reg(&mut machine, 3, 0x21);
        set_reg(&mut machine, 5, 7);
        let address = if pair { 0x21_0000_0040 } else { 0x40 };
        global_region(&mut machine, address, 4, 0);
        put_global_word(&mut machine, address, 100);
        run_op(
            Op::Atom(red(AtomOp::Add, AtomType::U32, 2, pair, 0, 5)),
            &mut machine,
        );
        assert_eq!(global_word(&machine, address), 107, "pair {pair}");
    }
}

/// The uniform `ATOMG` reads bit 70 as its register's 64-bit flag (`reg_pair = Take(70, 1)`),
/// which the decoder carries as the address's `reg_64`.
#[test]
fn the_uniform_atomg_reads_bit_70_as_the_register_pair() {
    for reg_64 in [false, true] {
        let mut machine = Machine::new();
        set_reg(&mut machine, 2, 0x80);
        set_reg(&mut machine, 3, 0x22);
        set_reg(&mut machine, 5, 3);
        let address = if reg_64 { 0x22_0000_0080 } else { 0x80 };
        global_region(&mut machine, address, 4, 0);
        put_global_word(&mut machine, address, 40);
        let addr = with_uniform(2, reg_64, 0, false, false, 0);
        run_op(
            Op::Atom(atom(9, AtomOp::Add, AtomType::U32, GLOBAL, addr, 5)),
            &mut machine,
        );
        assert_eq!(global_word(&machine, address), 43);
        assert_eq!(machine.regs[9], 40);
    }
}

/// A false load predicate gives zero in every destination register (the load still runs); a
/// true one, or a negated false one, gives the loaded value.
#[test]
fn a_false_load_predicate_gives_zero() {
    for (pred, value, expected) in [
        (ps(3, false), false, [0, 0]),
        (ps(3, false), true, [0x1111_1111, 0x2222_2222]),
        (ps(3, true), false, [0x1111_1111, 0x2222_2222]),
        (ps(7, true), true, [0, 0]),
    ] {
        let mut machine = machine();
        put_global_word(&mut machine, BASE, 0x1111_1111);
        put_global_word(&mut machine, BASE + 4, 0x2222_2222);
        pointer(&mut machine, 0);
        machine.preds[3] = value;
        let op = OpLd {
            pred,
            ..ldg(8, MemType::B64, with_uniform(2, true, 63, false, false, 0))
        };
        run_op(Op::Ld(op), &mut machine);
        assert_eq!(
            [machine.regs[8], machine.regs[9]],
            expected,
            "{pred:?} {value}"
        );
    }
}

/// A load into `RZ` still reads memory (an unmapped address faults) but writes nothing.
#[test]
fn a_load_into_rz_reads_but_writes_nothing() {
    let mut machine = machine();
    pointer(&mut machine, 0);
    let regs = machine.regs;
    run_op(
        Op::Ld(ldg(255, MemType::B64, plain(2, true, 0))),
        &mut machine,
    );
    assert_eq!(machine.regs, regs);
    let mut empty = Machine::new();
    let function = build_op(Op::Ld(ldg(255, MemType::B32, plain(2, true, 0)))).expect("translates");
    assert!(matches!(
        empty.run(&function),
        Err(InterpError::OutOfBounds { .. })
    ));
}

/// Every memory order but the system scope's translates, and none changes the access.
#[test]
fn every_order_but_the_system_scope_translates() {
    for &order in MemOrder::ALL {
        let result = build_op(
            Op::Ld(ldg(1, MemType::B32, plain(2, true, 0))).with_space(global_with(order)),
        );
        if order == MemOrder::Strong(MemScope::System) {
            assert_eq!(
                result,
                Err(MemoryError::Unsupported(MemoryUnsupported::SystemOrder))
            );
        } else {
            let mut machine = machine();
            put_global_word(&mut machine, BASE, 0x600d_f00d);
            pointer(&mut machine, 0);
            machine.run(&result.expect("translates")).expect("runs");
            assert_eq!(machine.regs[1], 0x600d_f00d, "{order:?}");
        }
    }
}

/// Builds a copy of a load with another space.
trait WithSpace {
    fn with_space(self, space: MemSpace) -> Self;
}

impl WithSpace for Op {
    fn with_space(self, space: MemSpace) -> Self {
        match self {
            Self::Ld(ld) => Self::Ld(OpLd { space, ..ld }),
            Self::St(st) => Self::St(OpSt { space, ..st }),
            Self::Atom(atom) => Self::Atom(OpAtom { space, ..atom }),
            other => other,
        }
    }
}

/// Bit 76 of the uniform `LDG` and `STG` with the uniform address on makes the uniform
/// register a memory descriptor: the address is the register pair plus the offset alone. With
/// the uniform address off, or without bit 76, the uniform register is added (when on) as
/// usual.
#[test]
fn bit_76_makes_the_uniform_register_a_descriptor() {
    for (bit76, enabled, added) in [
        (true, true, false),
        (false, true, true),
        (true, false, false),
        (false, false, false),
    ] {
        let mut machine = Machine::new();
        set_reg(&mut machine, 2, 0x100);
        set_reg(&mut machine, 3, 0x21);
        set_uniform(&mut machine, 8, 0x40);
        set_uniform(&mut machine, 9, 0);
        set_reg(&mut machine, 7, 0x5151_5151);
        let address = if added {
            0x21_0000_0140
        } else {
            0x21_0000_0100
        };
        global_region(&mut machine, 0x21_0000_0000, 0x200, 0);
        put_global_word(&mut machine, address + 0x10, 0x600d_cafe);
        let addr = with_uniform(2, true, 8, true, enabled, 0x10);
        let load = OpLd {
            bit76,
            ..ldg(1, MemType::B32, addr)
        };
        let store = OpSt {
            bit76,
            ..st(7, MemType::B32, GLOBAL, addr)
        };
        run(
            &[Instr::new(Op::Ld(load)), Instr::new(Op::St(store))],
            &mut machine,
        );
        assert_eq!(machine.regs[1], 0x600d_cafe, "{bit76} {enabled}");
        assert_eq!(global_word(&machine, address + 0x10), 0x5151_5151);
        // The descriptor is still checked as an address register would be.
        if bit76 && enabled {
            let odd = with_uniform(2, true, 9, true, true, 0);
            assert_eq!(
                build_op(Op::Ld(OpLd {
                    bit76,
                    ..ldg(1, MemType::B32, odd)
                })),
                Err(MemoryError::Unsupported(
                    MemoryUnsupported::GlobalUniformAddress {
                        reg: ur(9),
                        pair: true,
                    }
                ))
            );
        }
    }
}
