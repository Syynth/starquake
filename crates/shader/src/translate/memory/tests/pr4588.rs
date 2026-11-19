//! PR #4588's cases for this group (`eden-pr4588/src/tests/video_core/native_shader.cpp` @
//! e5fb3b97), rebuilt with the test assembler from the instructions their words encode, with
//! the PR's expected values. Each test names the PR's `TEST_CASE`. The PR packs uniform
//! registers into the top of the register file (`UR4` is its `R220`); here they are uniform
//! registers.
//!
//! The PR runs most of these through its lowering to Maxwell (`TranslateScalarNative`, which
//! calls `LowerToMaxwell`), not through the native translator this module ports. Left out and
//! named here:
//!
//! - "Native CTA-strong global loads retain memory ordering around the read" and "Native scalar
//!   strong global loads retain device memory ordering": the lowering puts a device barrier
//!   before and after a `.STRONG.SM` load; the native translator ignores the order
//!   (`Ignore(76, 4)`), as this module does. The loaded values are checked by
//!   `scalar_global_loads_retain_address_carry_and_signed_offsets`.
//! - "Native uniform global reduction preserves its full pointer and wrapping addend", "Native
//!   shared loads retain uniform offsets and their byte displacement" and "Native uniform shared
//!   stores and loads retain their data and displacement": the lowering takes a uniform `RED`,
//!   `LDS` and `STS` (`0x988`); the native translator refuses them ("native global atomic
//!   flags", "native shared uniform address", no case for `0x988`), as this module does
//!   (`rejections::every_refusal_has_its_message`).
//! - "Native signed address carry and byte stores preserve pointer and width" and the address
//!   arithmetic of "Native histogram pointer carry and global reduction retain the address and
//!   addend": their `IADD3` and `LEA` are the integer translator's (slice 12a); the `RED` is
//!   ported below with the address the arithmetic gives.
//! - The `BAR` of "Native shared atomics and barriers retain real shared-memory effects" and the
//!   `DEPBAR` of "Native global-to-shared copies transfer four words or zero-fill without
//!   reading": the control group's (slice 12h).
//! - The words the lowering alone refuses: "Native local spill retains byte address data and
//!   execution predicate"'s unaligned offset, "Native unsigned global maximum"'s `RED.MIN`,
//!   "Native global-to-shared copy"'s guarded copy (a guard here makes a conditional block), and
//!   "Native scalar global stores"' other store width.
//! - "Native uniform 64-bit loads populate consecutive registers", "Native uniform constant
//!   pairs remain contiguous at bank boundaries" and "Native uniform loads retain the full
//!   five-bit constant buffer bank": `ULDC`, slice 6d's. The captured-shader cases need
//!   fixtures from games.

use super::*;
use crate::decode::OpLdGsts;
use crate::ir::Inst;

fn guarded(op: Op, guard: PredSrc) -> Instr {
    Instr { guard, op }
}

/// The CTA-strong order NAK and the PR's words use (`0x5` in bits 77..81).
const STRONG_CTA: MemSpace = global_with(MemOrder::Strong(MemScope::Cta));

/// "Native scalar global stores retain both uniform address words and source data": `STG`
/// with `RZ` and the uniform pair `UR4` stores `R7` at `UR4:UR5`; an odd uniform pair is
/// refused.
#[test]
fn scalar_global_stores_retain_both_uniform_address_words_and_source_data() {
    let store = |uniform| {
        Op::St(st(
            7,
            MemType::B32,
            STRONG_CTA,
            with_uniform(255, true, uniform, true, true, 0),
        ))
    };
    let mut machine = Machine::new();
    set_uniform(&mut machine, 4, 0x1000);
    set_uniform(&mut machine, 5, 0x21);
    set_reg(&mut machine, 7, 0xdead_beef);
    global_region(&mut machine, 0x21_0000_1000, 4, 0);
    run_op(store(4), &mut machine);
    assert_eq!(global_word(&machine, 0x21_0000_1000), 0xdead_beef);
    assert_eq!(
        build_op(store(5)),
        Err(MemoryError::Unsupported(
            MemoryUnsupported::GlobalUniformAddress {
                reg: ur(5),
                pair: true,
            }
        ))
    );
}

/// "Native shared atomic extrema retain unsigned values and optional old results".
#[test]
fn shared_atomic_extrema_retain_unsigned_values_and_optional_old_results() {
    for minimum in [false, true] {
        for discard in [false, true] {
            let op = if minimum { AtomOp::Min } else { AtomOp::Max };
            let dst = if discard { 255 } else { 5 };
            let space = MemSpace::Shared(OffsetStride::X1);
            let instr = Op::Atom(atom(dst, op, AtomType::U32, space, plain(255, false, 8), 1));
            for old in [0_u32, 1, 0x8000_0000, 0xffff_ffff] {
                for value in [0_u32, 1, 0x8000_0000, 0xffff_ffff] {
                    let mut machine = Machine::new();
                    machine.shared = vec![0; 0x10];
                    put_shared_word(&mut machine, 8, old);
                    set_reg(&mut machine, 1, value);
                    set_reg(&mut machine, 5, 0x1234_5678);
                    run_op(instr, &mut machine);
                    let expected = if minimum {
                        old.min(value)
                    } else {
                        old.max(value)
                    };
                    assert_eq!(shared_word(&machine, 8), expected);
                    assert_eq!(machine.regs[1], value);
                    assert_eq!(machine.regs[5], if discard { 0x1234_5678 } else { old });
                }
            }
        }
    }
}

/// "Native global stores carry a register offset into the high address word": `UR6:UR7` plus
/// `R2` with a carry into the high word.
#[test]
fn global_stores_carry_a_register_offset_into_the_high_address_word() {
    let mut machine = Machine::new();
    set_uniform(&mut machine, 6, 0xffff_fff0);
    set_uniform(&mut machine, 7, 0x21);
    set_reg(&mut machine, 2, 0x30);
    set_reg(&mut machine, 9, 0xdead_beef);
    global_region(&mut machine, 0x22_0000_0000, 0x40, 0);
    let addr = with_uniform(2, false, 6, true, true, 0);
    run_op(Op::St(st(9, MemType::B32, STRONG_CTA, addr)), &mut machine);
    assert_eq!(global_word(&machine, 0x22_0000_0020), 0xdead_beef);
    let written = machine
        .global
        .bytes
        .iter()
        .filter(|&&byte| byte != 0)
        .count();
    assert_eq!(written, 4, "one word written");
}

/// "Native shared atomics and barriers retain real shared-memory effects" without its `BAR`:
/// `STS.X4`, `LDS.X4`, `ATOMS.ADD` and `MEMBAR` at scope 5.
#[test]
fn shared_atomics_and_barriers_retain_real_shared_memory_effects() {
    let x4 = MemSpace::Shared(OffsetStride::X4);
    let program = [
        Instr::new(Op::St(st(7, MemType::B32, x4, plain(3, false, 4)))),
        Instr::new(Op::Ld(lds(9, MemType::B32, 3, OffsetStride::X4, 4))),
        Instr::new(Op::Atom(atom(
            10,
            AtomOp::Add,
            AtomType::U32,
            MemSpace::Shared(OffsetStride::X1),
            plain(255, false, 0x7fc),
            9,
        ))),
        Instr::new(Op::MemBar(OpMemBar {
            scope: MemBarScope::Scope5,
            mmio: false,
            sc: false,
        })),
    ];
    let function = build(&program).expect("translates");
    let barriers = function
        .listed()
        .filter(
            |(_, inst)| matches!(inst, Inst::Op(op) if op.opcode == Opcode::DeviceMemoryBarrier),
        )
        .count();
    assert_eq!(barriers, 1);
    let mut machine = Machine::new();
    machine.shared = vec![0; 0x800];
    set_reg(&mut machine, 3, 3);
    set_reg(&mut machine, 7, 5);
    put_shared_word(&mut machine, 0x7fc, 100);
    machine.run(&function).expect("runs");
    assert_eq!(shared_word(&machine, 16), 5);
    assert_eq!(shared_word(&machine, 0x7fc), 105);
    assert_eq!(machine.regs[10], 100);
}

/// "Native global vector loads preserve both address words and each result channel": one
/// 64- or 128-bit load at `R8:R9` plus the sign-extended offset, its words written in order;
/// an odd address pair is refused.
#[test]
fn global_vector_loads_preserve_both_address_words_and_each_result_channel() {
    for pair in [false, true] {
        for offset in [0_i32, 16, -16] {
            for dst in [8_u8, 16] {
                let mem_type = if pair { MemType::B64 } else { MemType::B128 };
                let function = build_op(Op::Ld(ldg(dst, mem_type, plain(8, true, offset))))
                    .expect("translates");
                let load = if pair {
                    Opcode::LoadGlobal64
                } else {
                    Opcode::LoadGlobal128
                };
                let mut reads = Vec::new();
                let mut writes = Vec::new();
                let mut loads = 0;
                for (_, inst) in function.listed() {
                    let Inst::Op(op) = inst else { continue };
                    match (op.opcode, op.args.as_slice()) {
                        (Opcode::GetRegister, [Operand::Imm(Imm::Reg(reg))]) => {
                            reads.push(reg.index());
                        }
                        (Opcode::SetRegister, [Operand::Imm(Imm::Reg(reg)), _]) => {
                            writes.push(reg.index());
                        }
                        (Opcode::IAdd64, [_, Operand::Imm(Imm::U64(added))]) => {
                            assert_eq!(*added, sext64(offset));
                        }
                        (opcode, _) if opcode == load => loads += 1,
                        _ => {}
                    }
                }
                assert_eq!(loads, 1);
                assert_eq!(reads, [8, 9]);
                let words = if pair { 2 } else { 4 };
                assert_eq!(writes, (dst..dst + words).collect::<Vec<_>>());
            }
        }
    }
    assert_eq!(
        build_op(Op::Ld(ldg(8, MemType::B64, plain(9, true, 0)))),
        Err(MemoryError::Unsupported(
            MemoryUnsupported::GlobalRegisterAddress(r(9))
        ))
    );
}

/// "Native shared atomic OR retains the shared offset and source bits".
#[test]
fn shared_atomic_or_retains_the_shared_offset_and_source_bits() {
    let space = MemSpace::Shared(OffsetStride::X1);
    let op = Op::Atom(atom(
        255,
        AtomOp::Or,
        AtomType::U32,
        space,
        plain(255, false, 0x68),
        7,
    ));
    for initial in [0_u32, 0x1234_5678, 0xffff_ffff] {
        for operand in [0_u32, 0x8765_4321, 0xffff_ffff] {
            let mut machine = Machine::new();
            machine.shared = vec![0; 0x70];
            set_reg(&mut machine, 7, operand);
            put_shared_word(&mut machine, 0x68, initial);
            run_op(op, &mut machine);
            assert_eq!(shared_word(&machine, 0x68), initial | operand);
            assert_eq!(machine.regs[7], operand);
        }
    }
}

/// "Native scalar global loads retain address carry and signed offsets".
#[test]
fn scalar_global_loads_retain_address_carry_and_signed_offsets() {
    for offset in [0_i32, 16, -16] {
        let constant = global_with(MemOrder::Constant);
        let op = Op::Ld(OpLd {
            space: constant,
            ..ldg(29, MemType::B32, plain(0, true, offset))
        });
        let mut machine = Machine::new();
        set_reg(&mut machine, 0, 0xffff_fff8);
        set_reg(&mut machine, 1, 0x21);
        let address = 0x21_ffff_fff8_u64.wrapping_add(sext64(offset));
        global_region(&mut machine, 0x21_ffff_ff00, 0x200, 0);
        put_global_word(&mut machine, address, 0x8765_4321);
        run_op(op, &mut machine);
        assert_eq!(machine.regs[29], 0x8765_4321);
        assert_eq!([machine.regs[0], machine.regs[1]], [0xffff_fff8, 0x21]);
    }
}

/// "Native global loads combine a uniform pointer with the scalar byte displacement".
#[test]
fn global_loads_combine_a_uniform_pointer_with_the_scalar_byte_displacement() {
    for wide_offset in [false, true] {
        for offset in [0_i32, 16, -16] {
            let source = if wide_offset { 32 } else { 33 };
            let op = Op::Ld(OpLd {
                space: global_with(MemOrder::Constant),
                ..ldg(
                    29,
                    MemType::B32,
                    with_uniform(source, wide_offset, 4, true, true, offset),
                )
            });
            let mut machine = Machine::new();
            set_uniform(&mut machine, 4, 0xffff_fff8);
            set_uniform(&mut machine, 5, 0x21);
            set_reg(&mut machine, 32, 16);
            set_reg(&mut machine, 33, if wide_offset { 1 } else { 16 });
            let added = if wide_offset { 0x1_0000_0010 } else { 16 };
            let address = 0x21_ffff_fff8_u64
                .wrapping_add(added)
                .wrapping_add(sext64(offset));
            global_region(&mut machine, address & !0xff, 0x200, 0);
            put_global_word(&mut machine, address, 0x8765_4321);
            run_op(op, &mut machine);
            assert_eq!(machine.regs[29], 0x8765_4321);
            assert_eq!(machine.uniform_regs[4], 0xffff_fff8);
            assert_eq!(machine.regs[usize::from(source)], 16);
        }
    }
}

/// "Native paired shared values keep both words under scaled addressing".
#[test]
fn paired_shared_values_keep_both_words_under_scaled_addressing() {
    let x8 = MemSpace::Shared(OffsetStride::X8);
    let program = [
        Instr::new(Op::St(st(4, MemType::B64, x8, plain(19, false, 0)))),
        Instr::new(Op::Ld(lds(2, MemType::B64, 19, OffsetStride::X8, 0))),
    ];
    let mut machine = Machine::new();
    machine.shared = vec![0; 0x100];
    set_reg(&mut machine, 19, 17);
    set_reg(&mut machine, 4, 0x1234_5678);
    set_reg(&mut machine, 5, 0x9abc_def0);
    run(&program, &mut machine);
    assert_eq!(shared_word(&machine, 17 * 8), 0x1234_5678);
    assert_eq!(shared_word(&machine, 17 * 8 + 4), 0x9abc_def0);
    assert_eq!(
        [machine.regs[2], machine.regs[3]],
        [0x1234_5678, 0x9abc_def0]
    );
}

/// "Native histogram pointer carry and global reduction retain the address and addend": the
/// `RED.E.ADD.STRONG.GPU [R0.64], R3` at the address the case's arithmetic gives, with the
/// case's low words but `0xffffffff` replaced by `0xfffffffc` (the interpreter refuses a
/// misaligned access, as the hardware faults on one).
#[test]
fn histogram_pointer_carry_and_global_reduction_retain_the_address_and_addend() {
    let op = Op::Atom(OpAtom {
        space: global_with(MemOrder::Strong(MemScope::Gpu)),
        ..red(AtomOp::Add, AtomType::U32, 0, true, 0, 3)
    });
    for low in [0_u32, 0xffff_fff0, 0xffff_fffc] {
        let address = 0x21_0000_0000_u64 + u64::from(low) + 32;
        let mut machine = Machine::new();
        set_reg(
            &mut machine,
            0,
            u32::try_from(address & 0xffff_ffff).expect("low"),
        );
        set_reg(&mut machine, 1, u32::try_from(address >> 32).expect("high"));
        set_reg(&mut machine, 3, 17);
        global_region(&mut machine, address & !0xff, 0x200, 0);
        put_global_word(&mut machine, address, 0xffff_fff8);
        run_op(op, &mut machine);
        assert_eq!(global_word(&machine, address), 9);
    }
}

/// "Native histogram shared increment scales its address and increments once".
#[test]
fn histogram_shared_increment_scales_its_address_and_increments_once() {
    let mut machine = Machine::new();
    machine.shared = vec![0; 0x100];
    set_reg(&mut machine, 3, 63);
    put_shared_word(&mut machine, 252, 0xffff_ffff);
    run_op(
        Op::AtomsPopc(OpAtomsPopc {
            addr: r(3),
            stride: OffsetStride::X4,
        }),
        &mut machine,
    );
    assert_eq!(shared_word(&machine, 252), 0);
    assert!(machine.shared.iter().all(|&byte| byte == 0));
}

/// "Native indexed constant load adds a signed byte displacement": `c[3][R48 + 0xf814]` with
/// `R48 = 0x800` reads at `0x14`.
#[test]
fn indexed_constant_load_adds_a_signed_byte_displacement() {
    let mut machine = Machine::new();
    with_cbuf(&mut machine, 3, &[0, 0, 0, 0, 0, 0x1234_5678]);
    set_reg(&mut machine, 48, 0x800);
    let op = Op::Ldc(OpLdc {
        dst: r(7),
        cbuf: cbuf(3, 0xf814),
        offset: r(48),
        mode: LdcMode::Indexed,
        mem_type: MemType::B32,
    });
    run_op(op, &mut machine);
    assert_eq!(machine.regs[7], 0x1234_5678);
}

/// "Native local spill retains byte address data and execution predicate": `@P1 STL [RZ +
/// offset], R85` then `LDL R42, [RZ + offset]`.
#[test]
fn local_spill_retains_byte_address_data_and_execution_predicate() {
    for offset in [0_i32, 4, 28] {
        for active in [false, true] {
            let store = st(85, MemType::B32, LOCAL, plain(255, false, offset));
            let load = ldl(42, MemType::B32, 255, offset);
            let mut machine = Machine::new();
            machine.local = vec![0xdead_beef; 8];
            machine.preds[1] = active;
            set_reg(&mut machine, 85, 0x1234_abcd);
            run(
                &[
                    guarded(Op::St(store), ps(1, false)),
                    Instr::new(Op::Ld(load)),
                ],
                &mut machine,
            );
            let word = usize::try_from(offset / 4).expect("small");
            let expected = if active { 0x1234_abcd } else { 0xdead_beef };
            assert_eq!(machine.local[word], expected);
            assert_eq!(machine.regs[42], expected);
        }
    }
}

/// "Native wide global stores preserve all words across a low-address carry", with each
/// pointer aligned to its access (the interpreter refuses a misaligned access, as the hardware
/// faults on one): the last aligned low word below the carry, so the access ends at it.
#[test]
fn wide_global_stores_preserve_all_words_across_a_low_address_carry() {
    for (mem_type, words) in [(MemType::B32, 1_u8), (MemType::B64, 2), (MemType::B128, 4)] {
        let low = 0_u32.wrapping_sub(4 * u32::from(words));
        let mut machine = Machine::new();
        set_reg(&mut machine, 0, low);
        set_reg(&mut machine, 1, 0x21);
        for k in 0..words {
            set_reg(&mut machine, 12 + k, 0x1234_5678 + u32::from(k));
        }
        let pointer = 0x21_0000_0000 + u64::from(low);
        global_region(&mut machine, 0x21_ffff_fff0, 16, 0);
        let space = global_with(MemOrder::Strong(MemScope::Cta));
        run_op(
            Op::St(st(12, mem_type, space, plain(0, true, 0))),
            &mut machine,
        );
        for k in 0..words {
            let at = pointer + u64::from(k) * 4;
            assert_eq!(global_word(&machine, at), 0x1234_5678 + u32::from(k));
        }
    }
}

/// "Native descriptor-addressed global loads retain the explicit 64-bit pointer": with bit 76
/// and the uniform address on, `UR8` is a descriptor and the address is `R2:R3 + 0x20`.
#[test]
fn descriptor_addressed_global_loads_retain_the_explicit_64_bit_pointer() {
    for (mem_type, words) in [(MemType::B32, 1_usize), (MemType::B64, 2)] {
        let op = Op::Ld(OpLd {
            bit76: true,
            ..ldg(4, mem_type, with_uniform(2, true, 8, true, true, 0x20))
        });
        let mut machine = Machine::new();
        set_reg(&mut machine, 2, 0xffff_fff0);
        set_reg(&mut machine, 3, 0x1234_5678);
        set_uniform(&mut machine, 8, 0xabcd_ef01);
        set_uniform(&mut machine, 9, 0x8765_4321);
        global_region(&mut machine, 0x1234_5679_0000_0010, 8, 0);
        put_global_word(&mut machine, 0x1234_5679_0000_0010, 0x1357_9bdf);
        put_global_word(&mut machine, 0x1234_5679_0000_0014, 0x2468_ace0);
        run_op(op, &mut machine);
        assert_eq!(
            machine.regs[4..4 + words],
            [0x1357_9bdf, 0x2468_ace0][..words]
        );
        assert_eq!(machine.uniform_regs[8..10], [0xabcd_ef01, 0x8765_4321]);
    }
}

/// "Native global-to-shared copies transfer four words or zero-fill without reading", with its
/// `LDGDEPBAR`: two workgroup barriers here (the case's third is its `DEPBAR`'s, the control
/// group's), and the global memory present, since the copy reads it even when disabled (module
/// docs).
#[test]
fn global_to_shared_copies_transfer_four_words_or_zero_fill() {
    let data = [0x1234_5678_u32, 0x9abc_def0, 0x7654_3210, 0xfedc_ba98];
    for zero_fill in [false, true] {
        for global_offset in [0_i16, 32, 512, -16] {
            for shared_offset in [0_i32, 16, 4096, -16] {
                let copy = OpLdGsts {
                    shared: r(4),
                    shared_offset: i16::try_from(shared_offset / 16).expect("small"),
                    global: r(2),
                    global_offset,
                    descriptor: 8,
                    enable: ps(7, zero_fill),
                };
                let program = [
                    Instr::new(Op::LdGsts(copy)),
                    Instr::new(Op::LdgDepBar(OpLdgDepBar {})),
                ];
                let function = build(&program).expect("translates");
                let barriers = function
                    .listed()
                    .filter(|(_, inst)| {
                        matches!(inst, Inst::Op(op) if op.opcode == Opcode::WorkgroupMemoryBarrier)
                    })
                    .count();
                assert_eq!(barriers, 2);
                let mut machine = Machine::new();
                set_reg(&mut machine, 2, 0xffff_fff0);
                set_reg(&mut machine, 3, 1);
                set_reg(&mut machine, 4, 64);
                let source = 0x1_ffff_fff0_u64.wrapping_add(sext64(i32::from(global_offset)));
                global_region(&mut machine, 0x1_ffff_fe00, 0x600, 0);
                for (k, word) in (0_u64..).zip(data) {
                    put_global_word(&mut machine, source + 4 * k, word);
                }
                machine.shared = vec![0; 0x1100];
                let target = usize::try_from(64 + shared_offset).expect("inside");
                put_shared_word(&mut machine, target - 4, 0xabab_abab);
                put_shared_word(&mut machine, target + 16, 0xcdcd_cdcd);
                machine.run(&function).expect("runs");
                for (k, word) in (0_usize..).zip(data) {
                    let expected = if zero_fill { 0 } else { word };
                    assert_eq!(shared_word(&machine, target + 4 * k), expected);
                }
                assert_eq!(shared_word(&machine, target - 4), 0xabab_abab);
                assert_eq!(shared_word(&machine, target + 16), 0xcdcd_cdcd);
                assert_eq!(machine.regs[2..5], [0xffff_fff0, 1, 64]);
            }
        }
    }
}

/// "Native global-to-shared copy retains its source predicate and rejects other modes": each
/// predicate code (the low three bits the register, bit 3 the negation) is read as the enable
/// predicate; an odd global pair is refused.
#[test]
fn global_to_shared_copy_retains_its_source_predicate() {
    for condition in [0_u8, 4, 7, 8, 12] {
        let enable = ps(condition & 7, condition & 8 != 0);
        let copy = |global| {
            Op::LdGsts(OpLdGsts {
                shared: r(4),
                shared_offset: 0,
                global: r(global),
                global_offset: 0,
                descriptor: 8,
                enable,
            })
        };
        let function = build_op(copy(2)).expect("translates");
        let first = function.listed().next().map(|(_, inst)| inst.clone());
        if condition & 7 == 7 {
            // PT reads as an immediate: no predicate is read.
            assert!(!matches!(first, Some(Inst::Op(ref op)) if op.opcode == Opcode::GetPred));
        } else {
            let read = match first {
                Some(Inst::Op(op)) => Some((op.opcode, op.args)),
                _ => None,
            };
            assert_eq!(
                read,
                Some((
                    Opcode::GetPred,
                    vec![Operand::Imm(Imm::Pred(p(condition & 7)))]
                ))
            );
        }
        assert_eq!(
            build_op(copy(3)),
            Err(MemoryError::Unsupported(MemoryUnsupported::CopyRegisters {
                shared: r(4),
                global: r(3),
            }))
        );
    }
}

/// "Native global stores preserve scalar widths paired data and signed displacement": a
/// descriptor store (bit 76, the uniform address on) of each width at `R2:R3` plus the offset,
/// with the pointer's low word `0xfffffff8` rather than the case's `0xfffffffc` so the pair is
/// aligned (the interpreter refuses a misaligned access).
#[test]
fn global_stores_preserve_scalar_widths_paired_data_and_signed_displacement() {
    for (mem_type, expected) in [
        (MemType::U8, vec![0xef_u32]),
        (MemType::U16, vec![0xcdef]),
        (MemType::B32, vec![0x89ab_cdef]),
        (MemType::B64, vec![0x89ab_cdef, 0x7654_3210]),
    ] {
        for displacement in [0_i32, 8, -8] {
            let op = Op::St(OpSt {
                bit76: true,
                ..st(
                    6,
                    mem_type,
                    GLOBAL,
                    with_uniform(2, true, 4, true, true, displacement),
                )
            });
            let mut machine = Machine::new();
            set_reg(&mut machine, 2, 0xffff_fff8);
            set_reg(&mut machine, 3, 0x10);
            set_uniform(&mut machine, 4, 0x1000);
            set_uniform(&mut machine, 5, 2);
            set_reg(&mut machine, 6, 0x89ab_cdef);
            set_reg(&mut machine, 7, 0x7654_3210);
            let address = 0x10_ffff_fff8_u64.wrapping_add(sext64(displacement));
            global_region(&mut machine, address & !0xff, 0x200, 0);
            run_op(op, &mut machine);
            for (k, word) in (0_u64..).zip(&expected) {
                assert_eq!(global_word(&machine, address + 4 * k), *word);
            }
            let written = machine
                .global
                .bytes
                .iter()
                .filter(|&&byte| byte != 0)
                .count();
            assert_eq!(
                written,
                expected.len() * 4
                    - usize::from(mem_type == MemType::U8) * 3
                    - usize::from(mem_type == MemType::U16) * 2
            );
        }
    }
}

/// "Native constant-buffer logic and byte loads retain their values": the `LDC.U8 R5,
/// c[0x0][0x178]`.
#[test]
fn constant_buffer_byte_loads_retain_their_values() {
    let mut words = vec![0_u32; 0x60];
    words[0x178 / 4] = 0xffff_ffab;
    let mut machine = Machine::new();
    with_cbuf(&mut machine, 0, &words);
    let op = Op::Ldc(OpLdc {
        dst: r(5),
        cbuf: cbuf(0, 0x178),
        offset: Gpr::RZ,
        mode: LdcMode::Indexed,
        mem_type: MemType::U8,
    });
    run_op(op, &mut machine);
    assert_eq!(machine.regs[5], 0xab);
}

/// "Native shared vector zero stores preserve their address and predicates": `@P0 STS.64` or
/// `.128` of `RZ` at `R4` scaled plus 0x20.
#[test]
fn shared_vector_zero_stores_preserve_their_address_and_predicates() {
    for (mem_type, count) in [(MemType::B64, 2_usize), (MemType::B128, 4)] {
        for stride in [OffsetStride::X1, OffsetStride::X4, OffsetStride::X16] {
            for active in [false, true] {
                let space = MemSpace::Shared(stride);
                let op = Op::St(OpSt {
                    data: Gpr::RZ,
                    ..st(0, mem_type, space, plain(4, false, 0x20))
                });
                let mut machine = Machine::new();
                machine.shared = vec![0; 0x200];
                set_reg(&mut machine, 4, 16);
                machine.preds[0] = active;
                let scale = match stride {
                    OffsetStride::X1 => 1,
                    OffsetStride::X4 => 4,
                    OffsetStride::X8 => 8,
                    OffsetStride::X16 => 16,
                };
                let address = 16 * scale + 32;
                for k in 0..count + 2 {
                    put_shared_word(&mut machine, address - 4 + k * 4, 0xdead_beef);
                }
                run(&[guarded(op, ps(0, false))], &mut machine);
                for k in 0..count + 2 {
                    let zeroed = active && k > 0 && k <= count;
                    let expected = if zeroed { 0 } else { 0xdead_beef };
                    assert_eq!(shared_word(&machine, address - 4 + k * 4), expected);
                }
                assert_eq!(machine.regs[4], 16);
            }
        }
    }
}

/// "Native unsigned halfword global loads preserve addresses and ignore descriptors": a guarded
/// descriptor `LDG.U16` at `R2:R3`, `UR6` not added.
#[test]
fn unsigned_halfword_global_loads_preserve_addresses_and_ignore_descriptors() {
    for (guard, negated) in [(0_u8, false), (0, true), (7, false)] {
        for enabled in [false, true] {
            for address in [0x1234_5002_u64, 0x1234_5678_abcd_ef02] {
                let op = Op::Ld(OpLd {
                    bit76: true,
                    ..ldg(4, MemType::U16, with_uniform(2, true, 6, true, true, 0))
                });
                let mut machine = Machine::new();
                set_reg(
                    &mut machine,
                    2,
                    u32::try_from(address & 0xffff_ffff).expect("low"),
                );
                set_reg(&mut machine, 3, u32::try_from(address >> 32).expect("high"));
                set_reg(&mut machine, 4, 0xdead_beef);
                set_uniform(&mut machine, 6, 0x1111_1111);
                set_uniform(&mut machine, 7, 0x2222_2222);
                global_region(&mut machine, address, 0x10, 0);
                put_global_word(&mut machine, address, 0xabcd_8765);
                machine.preds[0] = enabled;
                run(&[guarded(op, ps(guard, negated))], &mut machine);
                let active = (guard == 7 || enabled) != negated;
                assert_eq!(machine.regs[4], if active { 0x8765 } else { 0xdead_beef });
            }
        }
    }
}

/// "Native local spills retain dynamic byte addresses offsets and predication": a guarded
/// `STL [R1 + displacement], R4` then `LDL R5` from the same address.
#[test]
fn local_spills_retain_dynamic_byte_addresses_offsets_and_predication() {
    for displacement in [0_i32, 4, 12] {
        for (guard, negated) in [(0_u8, false), (0, true), (7, false)] {
            for last_use in [false, true] {
                for active in [false, true] {
                    for base in [0_u32, 16, 32] {
                        let store = st(4, MemType::B32, LOCAL, plain(1, false, displacement));
                        let eviction = if last_use {
                            MemEvictionPriority::LastUse
                        } else {
                            MemEvictionPriority::Normal
                        };
                        let load = OpLd {
                            space: MemSpace::Local(eviction),
                            ..ldl(5, MemType::B32, 1, displacement)
                        };
                        let mut machine = Machine::new();
                        machine.local = vec![0xdead_beef; 16];
                        set_reg(&mut machine, 1, base);
                        set_reg(&mut machine, 4, 0x1234_abcd);
                        machine.preds[0] = active;
                        run(
                            &[
                                guarded(Op::St(store), ps(guard, negated)),
                                Instr::new(Op::Ld(load)),
                            ],
                            &mut machine,
                        );
                        let enabled = (guard == 7 || active) != negated;
                        let expected = if enabled { 0x1234_abcd } else { 0xdead_beef };
                        assert_eq!(machine.regs[5], expected);
                        let word =
                            usize::try_from((base + to_u32(displacement)) / 4).expect("small");
                        for (index, value) in machine.local.iter().enumerate() {
                            let changed = enabled && index == word;
                            assert_eq!(*value, if changed { 0x1234_abcd } else { 0xdead_beef });
                        }
                    }
                }
            }
        }
    }
}

/// "Native unsigned global maximum preserves full pointers and all registers": `RED.MAX` at
/// `R6:R7`; `R7` as the pair and a uniform field of `UR1` are refused.
#[test]
fn unsigned_global_maximum_preserves_full_pointers_and_all_registers() {
    let op = |reg| {
        Op::Atom(OpAtom {
            space: global_with(MemOrder::Strong(MemScope::Gpu)),
            ..red(AtomOp::Max, AtomType::U32, reg, true, 0, 4)
        })
    };
    for current in [0_u32, 1, 0x7fff_ffff, 0x8000_0000, 0xffff_ffff] {
        for operand in [0_u32, 1, 0x7fff_ffff, 0x8000_0000, 0xffff_ffff] {
            let mut machine = Machine::new();
            for (index, slot) in (0_u32..).zip(machine.regs.iter_mut()) {
                *slot = index * 37 + 1;
            }
            set_reg(&mut machine, 6, 0xffff_fff0);
            set_reg(&mut machine, 7, 0x29);
            set_reg(&mut machine, 4, operand);
            machine.preds = [true; 8];
            let regs = machine.regs;
            global_region(&mut machine, 0x29_ffff_fff0, 8, 0);
            put_global_word(&mut machine, 0x29_ffff_fff0, current);
            put_global_word(&mut machine, 0x29_ffff_fff4, 0xcafe_beef);
            run_op(op(6), &mut machine);
            assert_eq!(global_word(&machine, 0x29_ffff_fff0), current.max(operand));
            assert_eq!(global_word(&machine, 0x29_ffff_fff4), 0xcafe_beef);
            assert_eq!(machine.regs, regs);
        }
    }
    assert_eq!(
        build_op(op(7)),
        Err(MemoryError::Unsupported(
            MemoryUnsupported::GlobalRegisterAddress(r(7))
        ))
    );
    let uniform_field = Op::Atom(OpAtom {
        addr: with_uniform(6, false, 1, true, false, 0),
        ..red(AtomOp::Max, AtomType::U32, 6, true, 0, 4)
    });
    assert_eq!(
        build_op(uniform_field),
        Err(MemoryError::Unsupported(
            MemoryUnsupported::ReductionUniformField(ur(1))
        ))
    );
}
