//! Barriers (`MEMBAR`, `ERRBAR`, `LDGDEPBAR`) and the global-to-shared copy `LDGSTS`.

use super::*;
use crate::decode::OpLdGsts;
use crate::ir::Inst;

/// The opcodes `function` holds, in order, without the register accesses.
fn opcodes(function: &Function) -> Vec<Opcode> {
    function
        .listed()
        .filter_map(|(_, inst)| match inst {
            Inst::Op(op) => Some(op.opcode),
            Inst::Phi(_) => None,
        })
        .collect()
}

/// `MEMBAR.CTA` is a workgroup barrier; the GPU and system scopes and scope 5 are device
/// barriers (`Visitor::MEMBAR`); `ERRBAR` is a device barrier and `LDGDEPBAR` a workgroup one
/// (`Visitor::Run`).
#[test]
fn barriers_by_scope() {
    let membar = |scope| {
        Op::MemBar(OpMemBar {
            scope,
            mmio: false,
            sc: false,
        })
    };
    for (op, expected) in [
        (membar(MemBarScope::Cta), Opcode::WorkgroupMemoryBarrier),
        (membar(MemBarScope::Gpu), Opcode::DeviceMemoryBarrier),
        (membar(MemBarScope::System), Opcode::DeviceMemoryBarrier),
        (membar(MemBarScope::Scope5), Opcode::DeviceMemoryBarrier),
        (Op::ErrBar(OpErrBar {}), Opcode::DeviceMemoryBarrier),
        (
            Op::LdgDepBar(OpLdgDepBar {}),
            Opcode::WorkgroupMemoryBarrier,
        ),
    ] {
        let function = build_op(op).expect("translates");
        assert_eq!(opcodes(&function), [expected], "{op}");
        Machine::new().run(&function).expect("runs");
    }
}

/// An `LDGSTS` from `R4:R5 + global` to `R6 + 16 * shared`, enabled by `enable`.
fn ldgsts(global: i16, shared: i16, enable: PredSrc) -> OpLdGsts {
    OpLdGsts {
        shared: r(6),
        shared_offset: shared,
        global: r(4),
        global_offset: global,
        descriptor: 0x2a,
        enable,
    }
}

/// `LDGSTS` copies four words from global to shared memory at both offsets, or zero-fills them
/// when the enable predicate is false ("Native global-to-shared copies transfer four words or
/// zero-fill without reading"; the load still runs here, as in the PR's native translator).
#[test]
fn ldgsts_copies_four_words_or_zero_fills() {
    for (global, shared) in [(0_i16, 0_i16), (-0x10, 2), (0x7f0, -1)] {
        for enabled in [false, true] {
            for negated in [false, true] {
                let mut machine = Machine::new();
                let source = 0x21_0000_0800_u64;
                global_region(&mut machine, 0x21_0000_0000, 0x1000, 0);
                let start = source.wrapping_add(sext64(i32::from(global)));
                for k in 0..4_u32 {
                    put_global_word(
                        &mut machine,
                        start + u64::from(k) * 4,
                        0x1111_1111 * (k + 1),
                    );
                }
                set_reg(&mut machine, 4, 0x800);
                set_reg(&mut machine, 5, 0x21);
                set_reg(&mut machine, 6, 0x40);
                machine.shared = vec![0xa5; 0x100];
                machine.preds[0] = enabled;
                run_op(
                    Op::LdGsts(ldgsts(global, shared, ps(0, negated))),
                    &mut machine,
                );
                let target = usize::try_from(0x40 + i32::from(shared) * 16).expect("inside");
                for k in 0..4_u32 {
                    let at = target + usize::try_from(k * 4).expect("small");
                    let expected = if enabled == negated {
                        0
                    } else {
                        0x1111_1111 * (k + 1)
                    };
                    assert_eq!(shared_word(&machine, at), expected, "{global} {shared} {k}");
                }
                assert_eq!(machine.shared[target + 16], 0xa5);
            }
        }
    }
}

/// `LDGSTS` ends with a workgroup barrier after its four shared writes, and a following
/// `LDGDEPBAR` adds a second one, so a later `LDS` reads the copied data after both.
#[test]
fn ldgsts_then_ldgdepbar_orders_the_copy_before_the_read() {
    let program = [
        Instr::new(Op::LdGsts(ldgsts(0, 0, ps(0, false)))),
        Instr::new(Op::LdgDepBar(OpLdgDepBar {})),
        Instr::new(Op::Ld(lds(8, MemType::B128, 6, OffsetStride::X1, 0))),
    ];
    let function = build(&program).expect("translates");
    let ops: Vec<Opcode> = opcodes(&function)
        .into_iter()
        .filter(|op| {
            matches!(
                op,
                Opcode::WriteSharedU32 | Opcode::WorkgroupMemoryBarrier | Opcode::LoadSharedU128
            )
        })
        .collect();
    assert_eq!(
        ops,
        [
            Opcode::WriteSharedU32,
            Opcode::WriteSharedU32,
            Opcode::WriteSharedU32,
            Opcode::WriteSharedU32,
            Opcode::WorkgroupMemoryBarrier,
            Opcode::WorkgroupMemoryBarrier,
            Opcode::LoadSharedU128,
        ]
    );
    let mut machine = Machine::new();
    global_region(&mut machine, 0x1000, 0x10, 0);
    for k in 0..4_u32 {
        put_global_word(&mut machine, 0x1000 + u64::from(k) * 4, 0xc0de_0000 + k);
    }
    set_reg(&mut machine, 4, 0x1000);
    set_reg(&mut machine, 6, 0x20);
    machine.shared = vec![0; 0x40];
    machine.preds[0] = true;
    machine.run(&function).expect("runs");
    assert_eq!(
        machine.regs[8..12],
        [0xc0de_0000, 0xc0de_0001, 0xc0de_0002, 0xc0de_0003]
    );
}

/// Every enable predicate is taken, negated or not, and enables the copy exactly where it
/// holds (the module docs' divergence table: PR #4588's native translator takes only `P0`, its
/// lowering any predicate).
#[test]
fn any_predicate_enables_a_copy() {
    for index in 0..=7 {
        for negated in [false, true] {
            for value in [false, true] {
                let mut machine = Machine::new();
                global_region(&mut machine, 0x1000, 0x10, 0x77);
                set_reg(&mut machine, 4, 0x1000);
                set_reg(&mut machine, 6, 0);
                machine.shared = vec![0xa5; 0x10];
                if let Some(slot) = machine.preds.get_mut(usize::from(index)) {
                    *slot = value;
                }
                run_op(Op::LdGsts(ldgsts(0, 0, ps(index, negated))), &mut machine);
                let holds = (index == 7 || value) != negated;
                let expected = if holds { 0x7777_7777 } else { 0 };
                assert_eq!(
                    shared_word(&machine, 0),
                    expected,
                    "p{index} {negated} {value}"
                );
            }
        }
    }
}
