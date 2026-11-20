//! Atomics: every operation with each type the PR takes (`U32`, `S32`), in shared and global
//! memory and as `RED`, against a model of each operation.

use super::*;

/// The new value of a word holding `old` after `op` with `value`, of type `atom_type` (the
/// interpreter's model is Eden's: `Inc` wraps to 0 at `old >= value`, `Dec` to `value` at
/// `old == 0 || old > value`).
fn model(op: AtomOp, signed: bool, old: u32, value: u32) -> u32 {
    let as_signed = |x: u32| i32::from_ne_bytes(x.to_ne_bytes());
    match op {
        AtomOp::Add => old.wrapping_add(value),
        AtomOp::Min if signed => to_u32(as_signed(old).min(as_signed(value))),
        AtomOp::Min => old.min(value),
        AtomOp::Max if signed => to_u32(as_signed(old).max(as_signed(value))),
        AtomOp::Max => old.max(value),
        AtomOp::Inc => {
            if old >= value {
                0
            } else {
                old + 1
            }
        }
        AtomOp::Dec => {
            if old == 0 || old > value {
                value
            } else {
                old - 1
            }
        }
        AtomOp::And => old & value,
        AtomOp::Or => old | value,
        AtomOp::Xor => old ^ value,
        AtomOp::Exch | AtomOp::CmpExch(_) => value,
    }
}

const TYPES: [AtomType; 2] = [AtomType::U32, AtomType::I32];

/// `ATOMS` of every operation and type: the shared word becomes the model's value and the
/// destination the old value; `RZ` discards it.
#[test]
fn shared_atomics_follow_the_model() {
    for op in AtomOp::ENCODED {
        for atom_type in TYPES {
            for old in VALUES {
                for value in VALUES {
                    for dst in [5_u8, 255] {
                        let mut machine = Machine::new();
                        machine.shared = vec![0; 0x20];
                        put_shared_word(&mut machine, 0x10, old);
                        set_reg(&mut machine, 2, 0x8);
                        set_reg(&mut machine, 3, value);
                        set_reg(&mut machine, 5, 0x0123_4567);
                        let space = MemSpace::Shared(OffsetStride::X1);
                        let addr = plain(2, false, 8);
                        run_op(
                            Op::Atom(atom(dst, op, atom_type, space, addr, 3)),
                            &mut machine,
                        );
                        let signed = atom_type == AtomType::I32;
                        assert_eq!(
                            shared_word(&machine, 0x10),
                            model(op, signed, old, value),
                            "{op:?} {atom_type:?} {old:#x} {value:#x}"
                        );
                        let kept = if dst == 5 { old } else { 0x0123_4567 };
                        assert_eq!(machine.regs[5], kept);
                        assert_eq!(machine.regs[3], value);
                    }
                }
            }
        }
    }
}

/// `ATOMG` of every operation and type, through a 64-bit pointer.
#[test]
fn global_atomics_follow_the_model() {
    for op in AtomOp::ENCODED {
        for atom_type in TYPES {
            for old in VALUES {
                for value in VALUES {
                    let mut machine = Machine::new();
                    global_region(&mut machine, 0x21_0000_0000, 0x10, 0);
                    put_global_word(&mut machine, 0x21_0000_0008, old);
                    set_reg(&mut machine, 2, 0);
                    set_reg(&mut machine, 3, 0x21);
                    set_reg(&mut machine, 4, value);
                    let addr = plain(2, true, 8);
                    run_op(
                        Op::Atom(atom(6, op, atom_type, GLOBAL, addr, 4)),
                        &mut machine,
                    );
                    let signed = atom_type == AtomType::I32;
                    assert_eq!(
                        global_word(&machine, 0x21_0000_0008),
                        model(op, signed, old, value),
                        "{op:?} {atom_type:?} {old:#x} {value:#x}"
                    );
                    assert_eq!(machine.regs[6], old);
                }
            }
        }
    }
}

/// `RED` of every operation its three-bit field encodes (all but `EXCH`) changes memory as
/// `ATOMG` does and writes no register.
#[test]
fn reductions_write_no_register() {
    for op in AtomOp::ENCODED.into_iter().filter(|&op| op != AtomOp::Exch) {
        for atom_type in TYPES {
            let mut machine = Machine::new();
            global_region(&mut machine, 0x22_0000_0000, 0x10, 0);
            put_global_word(&mut machine, 0x22_0000_0004, 0x8000_0005);
            set_reg(&mut machine, 2, 0);
            set_reg(&mut machine, 3, 0x22);
            set_reg(&mut machine, 4, 7);
            let regs = machine.regs;
            run_op(Op::Atom(red(op, atom_type, 2, true, 4, 4)), &mut machine);
            let signed = atom_type == AtomType::I32;
            assert_eq!(
                global_word(&machine, 0x22_0000_0004),
                model(op, signed, 0x8000_0005, 7),
                "{op:?} {atom_type:?}"
            );
            assert_eq!(machine.regs, regs);
        }
    }
}

/// The signed and unsigned extrema pick different opcodes: a case where they differ.
#[test]
fn signedness_picks_the_extremum() {
    for (atom_type, expected) in [(AtomType::U32, 0xffff_ffff), (AtomType::I32, 1)] {
        let mut machine = Machine::new();
        machine.shared = vec![0; 8];
        put_shared_word(&mut machine, 0, 1);
        set_reg(&mut machine, 3, 0xffff_ffff);
        let space = MemSpace::Shared(OffsetStride::X1);
        run_op(
            Op::Atom(atom(
                255,
                AtomOp::Max,
                atom_type,
                space,
                plain(255, false, 0),
                3,
            )),
            &mut machine,
        );
        assert_eq!(shared_word(&machine, 0), expected, "{atom_type:?}");
    }
}

/// `ATOMS.POPC` adds one to the shared word at the scaled register, with no offset ("Native
/// histogram shared increment scales its address and increments once").
#[test]
fn atoms_popc_adds_one() {
    let mut machine = Machine::new();
    machine.shared = vec![0; 0x100];
    put_shared_word(&mut machine, 252, 0xffff_ffff);
    set_reg(&mut machine, 3, 63);
    run_op(
        Op::AtomsPopc(OpAtomsPopc {
            addr: r(3),
            stride: OffsetStride::X4,
        }),
        &mut machine,
    );
    assert_eq!(shared_word(&machine, 252), 0);
    assert!(
        machine.shared[..252].iter().all(|&byte| byte == 0),
        "only the word at 252 changes"
    );
}
