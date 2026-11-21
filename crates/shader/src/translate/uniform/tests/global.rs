//! `LDCG`: a global load at a uniform register pair, or zero where its uniform predicate holds
//! (NAK `OpLdcg`).

use super::*;
use crate::interp::Region;

/// The base of the tests' global region: above 4 GiB, so the high address word matters.
const BASE: u64 = 0x1_0000_0000;

/// An `LDCG` of `mem_type` into `dst` from `addr` plus `offset`, zero where `pred` holds.
fn ldcg(dst: u8, addr: UniformReg, offset: i32, pred: UniformPredSrc, mem_type: MemType) -> Op {
    Op::ULdc(OpULdc {
        dst: ur(dst),
        src: ULdcSrc::Global { addr, offset, pred },
        mem_type,
    })
}

/// A machine whose global region at [`BASE`] holds 64 bytes `0x10`, `0x11`, ..., with
/// `UR4:UR5` holding `BASE + 0x20`.
fn machine() -> Machine {
    let mut machine = machine_with(&[]);
    machine.global = Region {
        base: BASE,
        bytes: (0x10..0x50).collect(),
    };
    let address = BASE + 0x20;
    machine.uniform_regs[4] = u32::try_from(address & u64::from(u32::MAX)).expect("low");
    machine.uniform_regs[5] = u32::try_from(address >> 32).expect("high");
    machine
}

/// The 32-bit word of the region at byte `at`.
fn word_at(at: usize) -> u32 {
    let bytes: Vec<u8> = (0x10..0x50).collect();
    u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
}

/// Where the predicate does not hold, the load at the pair plus the signed offset; where it
/// holds, zero, and no load is made (the address of the second case is outside every region,
/// which a load would fault on).
#[test]
fn ldcg_loads_or_returns_zero() {
    for offset in [0x8_i32, -0x10, 0] {
        for holds in [false, true] {
            let mut machine = machine();
            machine.uniform_preds[1] = holds;
            machine.preds[1] = !holds;
            run_op(
                ldcg(8, ur(4), offset, ups(1, false), MemType::B64),
                &mut machine,
            );
            let at = usize::try_from(0x20 + offset).expect("in the region");
            let (low, high) = if holds {
                (0, 0)
            } else {
                (word_at(at), word_at(at + 4))
            };
            assert_eq!(machine.uniform_regs[8], low, "{offset} {holds}");
            assert_eq!(machine.uniform_regs[9], high, "{offset} {holds}");
        }
    }
    let mut machine = machine();
    machine.uniform_regs[5] = 0xdead;
    machine.uniform_preds[2] = true;
    machine.uniform_regs[8] = 0x5555;
    run_op(
        ldcg(8, ur(4), 0, ups(2, false), MemType::B32),
        &mut machine,
    );
    assert_eq!(machine.uniform_regs[8], 0, "no load where the predicate holds");
}

/// A negated predicate loads where the uniform predicate is false.
#[test]
fn a_negated_predicate_inverts_it() {
    for value in [false, true] {
        let mut machine = machine();
        machine.uniform_preds[3] = value;
        run_op(
            ldcg(8, ur(4), 0, ups(3, true), MemType::B32),
            &mut machine,
        );
        let expected = if value { word_at(0x20) } else { 0 };
        assert_eq!(machine.uniform_regs[8], expected, "{value}");
    }
}

/// Every type but `.128`: the narrow ones zero- or sign-extended.
#[test]
fn every_type_loads() {
    let cases = [
        (MemType::U8, 0x34_u32),
        (MemType::I8, 0x34),
        (MemType::U16, 0x3534),
        (MemType::I16, 0x3534),
        (MemType::B32, 0x3736_3534),
    ];
    for (mem_type, expected) in cases {
        let mut machine = machine();
        run_op(
            ldcg(8, ur(4), 4, ups(1, false), mem_type),
            &mut machine,
        );
        // `UP1` is false: the load is made.
        assert_eq!(machine.uniform_regs[8], expected, "{mem_type:?}");
    }
    let mut machine = machine();
    machine.global.bytes[0x24] = 0x80;
    run_op(
        ldcg(8, ur(4), 4, ups(1, false), MemType::I8),
        &mut machine,
    );
    assert_eq!(machine.uniform_regs[8], 0xffff_ff80);
}

/// `UPT` always gives zero, with no load and no block; `!UPT` always loads, with no block.
#[test]
fn constant_predicates_make_no_block() {
    let never = build(&[Instr::new(ldcg(
        8,
        ur(4),
        0,
        UniformPredSrc {
            reg: UniformPred::UPT,
            negated: false,
        },
        MemType::B64,
    ))])
    .expect("translates");
    assert_eq!(count(&never, Opcode::LoadGlobal64), 0);
    assert_eq!(never.to_string().matches("block").count(), 1);
    let mut machine = machine();
    machine.uniform_regs[8] = 1;
    machine.uniform_regs[9] = 1;
    machine.run(&never).expect("runs");
    assert_eq!((machine.uniform_regs[8], machine.uniform_regs[9]), (0, 0));
    let always = build(&[Instr::new(ldcg(
        8,
        ur(4),
        0,
        UniformPredSrc {
            reg: UniformPred::UPT,
            negated: true,
        },
        MemType::B32,
    ))])
    .expect("translates");
    assert_eq!(count(&always, Opcode::LoadGlobal32), 1);
    assert_eq!(count(&always, Opcode::GetUniformPred), 0);
    assert_eq!(always.to_string().matches("block").count(), 1);
}

/// The address is read before the destination is cleared, so a destination that overlaps the
/// address pair still loads from the address it held.
#[test]
fn the_address_is_read_before_the_destination_is_written() {
    let mut machine = machine();
    run_op(
        ldcg(4, ur(4), 0, ups(1, false), MemType::B64),
        &mut machine,
    );
    assert_eq!(machine.uniform_regs[4], word_at(0x20));
    assert_eq!(machine.uniform_regs[5], word_at(0x24));
}

/// `URZ` as the address pair reads address zero plus the offset.
#[test]
fn urz_addresses_zero() {
    let mut machine = machine();
    machine.global.base = 0x40;
    run_op(
        ldcg(8, UniformReg::URZ, 0x44, ups(1, false), MemType::B32),
        &mut machine,
    );
    assert_eq!(machine.uniform_regs[8], word_at(4));
}
