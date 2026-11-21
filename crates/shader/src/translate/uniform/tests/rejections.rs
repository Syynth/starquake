//! Every refusal with its exact error and message, each boundary on both sides, refused
//! instructions leaving the function untouched, and the order of the checks.

use super::alu::lea;
use super::*;
use crate::decode::{OpBMsk, OpIMad64, OpLea, OpPrmt, OpShf, OpUP2UR, PrmtMode, ShfType};

/// The error `instr` is refused with; the function is checked to be untouched.
fn refusal(instr: Instr) -> UniformError {
    let mut function = Function::new();
    let entry = function.entry();
    let mut builder = Builder::new(&mut function, entry);
    let error = translate_uniform(&mut builder, &instr, GuardMode::Honour)
        .expect_err("the case is refused");
    assert_eq!(builder.function(), &Function::new(), "{instr}");
    error
}

/// Whether `instr` translates.
fn accepted(instr: Instr) -> bool {
    let mut function = Function::new();
    let entry = function.entry();
    let mut builder = Builder::new(&mut function, entry);
    translate_uniform(&mut builder, &instr, GuardMode::Honour).is_ok()
}

/// A `ULDC` of `mem_type` into `dst` from buffer 1 at `offset`.
fn uldc(dst: UniformReg, offset: u16, mem_type: MemType) -> Op {
    Op::ULdc(OpULdc {
        dst,
        src: ULdcSrc::CBuf(CBufRef {
            buf: CBuf::Binding(1),
            offset,
        }),
        mem_type,
    })
}

/// An `LDCG` of `mem_type` into `dst`.
fn ldcg(dst: UniformReg, mem_type: MemType) -> Op {
    Op::ULdc(OpULdc {
        dst,
        src: ULdcSrc::Global {
            addr: ur(4),
            offset: 0,
            pred: ups(1, false),
        },
        mem_type,
    })
}

/// `UBMSK` and `UP2UR` are refused by name.
#[test]
fn instructions_neither_reference_translates() {
    let ubmsk = Instr::new(Op::UBMsk(twin(OpBMsk {
        dst: r(1),
        pos: reg(2),
        width: reg(3),
        wrap: false,
    })));
    let error = refusal(ubmsk);
    assert_eq!(
        error,
        UniformError::Unsupported(UniformUnsupported::Instruction("ubmsk"))
    );
    assert_eq!(
        error.to_string(),
        "ubmsk, which neither reference translates, is not supported"
    );
    assert_eq!(
        refusal(Instr::new(Op::UP2UR(OpUP2UR {}))),
        UniformError::Unsupported(UniformUnsupported::Instruction("up2ur"))
    );
}

/// Each twin with a source 2, in its form 2 (an immediate source 2), is refused by name; the
/// same twin with a register source 2 translates.
#[test]
fn every_form_2_is_refused() {
    let imm = AluSrc::Imm(0x10);
    let lea_with = |a_high: AluSrc| OpLea {
        a_high,
        dst_high: true,
        ..lea(4, reg(1), reg(2), 3)
    };
    let shf_with = |high: AluSrc| OpShf {
        dst: r(4),
        low: reg(1),
        shift: reg(2),
        high,
        data_type: ShfType::U32,
        wrap: false,
        right: false,
        dst_high: false,
    };
    let prmt_with = |high: AluSrc| OpPrmt {
        dst: r(4),
        srcs: [reg(1), high],
        sel: reg(2),
        mode: PrmtMode::Index,
    };
    let imad_with = |c: AluSrc| OpIMad64 {
        dst: r(4),
        srcs: [reg(1), reg(2), c],
        signed: false,
        carry: NOT_PT,
    };
    let cases = [
        (
            Op::UIAdd3(twin(iadd3(4, [reg(1), reg(2), imm]))),
            Op::UIAdd3(twin(iadd3(4, [reg(1), reg(2), reg(3)]))),
        ),
        (
            Op::ULea(twin(lea_with(imm))),
            Op::ULea(twin(lea_with(reg(3)))),
        ),
        (
            Op::UShf(twin(shf_with(imm))),
            Op::UShf(twin(shf_with(reg(3)))),
        ),
        (
            Op::UPrmt(twin(prmt_with(imm))),
            Op::UPrmt(twin(prmt_with(reg(3)))),
        ),
        (
            Op::UIMad64(twin(imad_with(imm))),
            Op::UIMad64(twin(imad_with(reg(6)))),
        ),
    ];
    for (form_2, form_1) in cases {
        let mnemonic = form_2.mnemonic();
        assert_eq!(
            refusal(Instr::new(form_2)),
            UniformError::Unsupported(UniformUnsupported::ImmediateSource2(mnemonic)),
            "{form_2}"
        );
        // `UPRMT` with a register selector is the integer group's refusal, not this one.
        if matches!(form_1, Op::UPrmt(_)) {
            assert_eq!(
                refusal(Instr::new(form_1)),
                UniformError::Twin(IntegerUnsupported::PermuteSelector)
            );
        } else {
            assert!(accepted(Instr::new(form_1)), "{form_1}");
        }
    }
    assert_eq!(
        UniformError::from(UniformUnsupported::ImmediateSource2("uiadd3")).to_string(),
        "uiadd3 with an immediate source 2 is not supported"
    );
}

/// `R2UR` into `URZ` is refused (`Require(destination < 63)`); into `UR62` it translates.
#[test]
fn r2ur_into_urz() {
    let r2ur = |dst| Instr::new(Op::R2UR(OpR2UR { dst, src: r(1) }));
    let error = refusal(r2ur(UniformReg::URZ));
    assert_eq!(
        error,
        UniformError::Unsupported(UniformUnsupported::R2urDestination)
    );
    assert_eq!(error.to_string(), "an r2ur into urz is not supported");
    let last = UniformReg::new(UniformReg::URZ.index() - 1).expect("UR62");
    assert!(accepted(r2ur(last)));
}

/// A bindless `ULDC` is refused.
#[test]
fn a_bindless_uldc() {
    let error = refusal(Instr::new(Op::ULdc(OpULdc {
        dst: ur(1),
        src: ULdcSrc::CBuf(CBufRef {
            buf: CBuf::BindlessUniform(ur(2)),
            offset: 0,
        }),
        mem_type: MemType::B32,
    })));
    assert_eq!(
        error,
        UniformError::Unsupported(UniformUnsupported::BindlessConstantLoad)
    );
    assert_eq!(error.to_string(), "a bindless uldc is not supported");
}

/// A `ULDC` offset must be a multiple of four, in the bound and the indexed form and for
/// every type: one past and one before a multiple are refused, the multiple accepted.
#[test]
fn a_uldc_offset_is_a_multiple_of_four() {
    let indexed = |offset: u16, mem_type| {
        Op::ULdc(OpULdc {
            dst: ur(4),
            src: ULdcSrc::Indexed {
                index: 1,
                reg: ur(2),
                offset,
            },
            mem_type,
        })
    };
    for mem_type in [MemType::U8, MemType::I16, MemType::B32] {
        for offset in [0x247_u16, 0x249, 0x24a] {
            let expected =
                UniformError::Unsupported(UniformUnsupported::ConstantLoadOffset(offset));
            assert_eq!(refusal(Instr::new(uldc(ur(4), offset, mem_type))), expected);
            assert_eq!(refusal(Instr::new(indexed(offset, mem_type))), expected);
        }
        assert!(accepted(Instr::new(uldc(ur(4), 0x248, mem_type))));
        assert!(accepted(Instr::new(indexed(0x248, mem_type))));
    }
    assert_eq!(
        UniformError::from(UniformUnsupported::ConstantLoadOffset(0x246)).to_string(),
        "a uldc at offset 0x246, which is not a multiple of four is not supported"
    );
}

/// A 64-bit destination must be even and below [`ULDC_PAIR_LIMIT`], a 128-bit one a multiple
/// of four below [`ULDC_QUAD_LIMIT`], for `ULDC`; `LDCG`'s 64-bit destination the same: each
/// edge on both sides.
#[test]
fn wide_destinations() {
    let pair_last = ULDC_PAIR_LIMIT - 2;
    let quad_last = ULDC_QUAD_LIMIT - 4;
    let reg = |index: u8| UniformReg::new(index).expect("a uniform register");
    let cases = [
        (MemType::B64, pair_last, true),
        (MemType::B64, ULDC_PAIR_LIMIT, false),
        (MemType::B64, 3, false),
        (MemType::B64, 0, true),
        (MemType::B128, quad_last, true),
        (MemType::B128, ULDC_QUAD_LIMIT, false),
        (MemType::B128, quad_last + 2, false),
        (MemType::B128, 0, true),
        (MemType::B32, UniformReg::URZ.index(), true),
    ];
    for (mem_type, index, ok) in cases {
        let words = words_of(mem_type);
        let expected = UniformError::Unsupported(UniformUnsupported::ConstantLoadDestination {
            dst: reg(index),
            words,
        });
        let op = uldc(reg(index), 0, mem_type);
        if ok {
            assert!(accepted(Instr::new(op)), "{op}");
        } else {
            assert_eq!(refusal(Instr::new(op)), expected, "{op}");
        }
        if mem_type != MemType::B128 {
            let op = ldcg(reg(index), mem_type);
            if ok {
                assert!(accepted(Instr::new(op)), "{op}");
            } else {
                assert_eq!(refusal(Instr::new(op)), expected, "{op}");
            }
        }
    }
    assert_eq!(
        refusal(Instr::new(uldc(UniformReg::URZ, 0, MemType::B64))),
        UniformError::Unsupported(UniformUnsupported::ConstantLoadDestination {
            dst: UniformReg::URZ,
            words: 2,
        })
    );
    assert_eq!(
        UniformError::from(UniformUnsupported::ConstantLoadDestination {
            dst: reg(3),
            words: 2,
        })
        .to_string(),
        "a 2-word load into ur3 is not supported"
    );
}

/// A 128-bit `LDCG` is refused (NAK's assertion below SM 10.0).
#[test]
fn a_128_bit_ldcg() {
    let error = refusal(Instr::new(ldcg(ur(4), MemType::B128)));
    assert_eq!(
        error,
        UniformError::Unsupported(UniformUnsupported::GlobalLoad128)
    );
    assert_eq!(error.to_string(), "a 128-bit ldcg is not supported");
}

/// `UIMAD.WIDE`'s pairs start at an even uniform register below
/// [`integer::UNIFORM_WIDE_PAIR_LIMIT`]: the last one accepted, the limit and an odd register
/// refused, by uniform register; a `URZ` addend reads zero.
#[test]
fn uimad_wide_pairs() {
    let limit = integer::UNIFORM_WIDE_PAIR_LIMIT;
    let wide = |dst: Gpr, c: AluSrc| {
        Instr::new(Op::UIMad64(twin(OpIMad64 {
            dst,
            srcs: [reg(1), reg(2), c],
            signed: false,
            carry: NOT_PT,
        })))
    };
    assert!(accepted(wide(r(limit - 2), reg(limit - 2))));
    assert!(accepted(wide(r(4), RZ)));
    for bad in [limit, 5] {
        assert_eq!(
            refusal(wide(r(bad), reg(4))),
            UniformError::Twin(IntegerUnsupported::WideUniformDestination(ur(bad)))
        );
        assert_eq!(
            refusal(wide(r(4), reg(bad))),
            UniformError::Twin(IntegerUnsupported::WideAddendUniform(ur(bad)))
        );
    }
    assert_eq!(
        refusal(wide(r(limit), reg(4))).to_string(),
        "an imad.wide destination pair at ur62 is not supported"
    );
}

/// Instructions of other groups are refused by their mnemonic.
#[test]
fn other_groups() {
    let error = refusal(Instr::new(Op::Nop(crate::decode::OpNop {})));
    assert_eq!(error, UniformError::OtherGroup("nop"));
    assert_eq!(
        error.to_string(),
        "nop is not a uniform datapath instruction"
    );
}

/// The documented order, each neighbouring pair with both faults at once: a twin's form 2
/// before the integer group's checks; `ULDC`'s bindless buffer before its offset, its offset
/// before its destination; an instruction's own check before a guard's successor phi.
#[test]
fn the_checks_come_in_their_order() {
    // Form 2 with a carry input the integer group refuses.
    let form_2 = Op::UIAdd3(twin(OpIAdd3 {
        carry: [ps(1, false), NOT_PT],
        ..iadd3(4, [reg(1), reg(2), AluSrc::Imm(1)])
    }));
    assert_eq!(
        refusal(Instr::new(form_2)),
        UniformError::Unsupported(UniformUnsupported::ImmediateSource2("uiadd3"))
    );
    let bindless_misaligned = Op::ULdc(OpULdc {
        dst: ur(1),
        src: ULdcSrc::CBuf(CBufRef {
            buf: CBuf::BindlessUniform(ur(2)),
            offset: 2,
        }),
        mem_type: MemType::B64,
    });
    assert_eq!(
        refusal(Instr::new(bindless_misaligned)),
        UniformError::Unsupported(UniformUnsupported::BindlessConstantLoad)
    );
    assert_eq!(
        refusal(Instr::new(uldc(ur(3), 2, MemType::B64))),
        UniformError::Unsupported(UniformUnsupported::ConstantLoadOffset(2))
    );
    // A bad destination under a guard, in a block whose successor has a phi naming it.
    let mut function = Function::new();
    let entry = function.entry();
    let mut builder = Builder::new(&mut function, entry);
    let exit = builder.add_block().expect("adds");
    builder
        .terminate(Terminator::Jump(exit))
        .expect("terminates");
    builder.switch_to(exit).expect("switches");
    builder
        .phi(
            crate::ir::Type::U32,
            vec![(entry, Operand::Imm(Imm::U32(1)))],
        )
        .expect("appends");
    builder.terminate(Terminator::Return).expect("terminates");
    let mut builder = Builder::new(&mut function, entry);
    assert_eq!(
        translate_uniform(
            &mut builder,
            &under(ps(1, false), uldc(ur(3), 0, MemType::B64)),
            GuardMode::Honour
        ),
        Err(UniformError::Unsupported(
            UniformUnsupported::ConstantLoadDestination {
                dst: ur(3),
                words: 2
            }
        ))
    );
}
