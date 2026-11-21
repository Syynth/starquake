//! The refusals: each with its exact error and message, the first fault in walk order, the
//! passes' order when both refuse, and a refused function left as it was.

use super::super::{HostLowerError, HostPass, lower_for_host, run};
use super::reg;
use crate::host::HostCaps;
use crate::ir::{BlockId, Builder, Function, Imm, Opcode, Operand, Terminator, Type, Value};

fn value(b: &mut Builder<'_>, opcode: Opcode, args: &[Operand]) -> Value {
    b.op(opcode, args).expect("builds")
}

/// A 64-bit float from registers 0 and 1, appended through `b`.
fn double(b: &mut Builder<'_>) -> Operand {
    let lo = Operand::Value(value(b, Opcode::GetRegister, &[reg(0)]));
    let hi = Operand::Value(value(b, Opcode::GetRegister, &[reg(1)]));
    let words = Operand::Value(value(b, Opcode::CompositeConstructU32x2, &[lo, hi]));
    Operand::Value(value(b, Opcode::PackDouble2x32, &[words]))
}

/// A 16-bit float from register 2, appended through `b`.
fn half(b: &mut Builder<'_>) -> Operand {
    let word = Operand::Value(value(b, Opcode::GetRegister, &[reg(2)]));
    let halves = Operand::Value(value(b, Opcode::UnpackFloat2x16, &[word]));
    Operand::Value(value(
        b,
        Opcode::CompositeExtractF16x2,
        &[halves, Operand::Imm(Imm::U32(0))],
    ))
}

/// A one-block function whose body `body` appends, ending in `return`.
fn build(body: impl FnOnce(&mut Builder<'_>) -> Vec<Value>) -> (Function, Vec<Value>) {
    let mut function = Function::new();
    let entry = function.entry();
    let mut b = Builder::new(&mut function, entry);
    let marked = body(&mut b);
    b.terminate(Terminator::Return).expect("returns");
    (function, marked)
}

/// `BitCastU16F16` and `BitCastF16U16` are refused by the f16 pass with their exact error and
/// message, and the function is left as it was.
#[test]
fn a_16_bit_bit_cast_is_refused() {
    let (cast_out, marked) = build(|b| {
        let h = half(b);
        vec![value(b, Opcode::BitCastU16F16, &[h])]
    });
    let (cast_in, marked_in) = build(|b| {
        let bits = Operand::Value(value(b, Opcode::UndefU16, &[]));
        vec![value(b, Opcode::BitCastF16U16, &[bits])]
    });
    for (original, opcode, at) in [
        (cast_out, Opcode::BitCastU16F16, marked[0]),
        (cast_in, Opcode::BitCastF16U16, marked_in[0]),
    ] {
        let mut function = original.clone();
        let error = run(&mut function, HostPass::Fp16ToFp32).expect_err("refused");
        assert_eq!(
            error,
            HostLowerError::NoNarrowForm {
                pass: HostPass::Fp16ToFp32,
                at,
                opcode
            }
        );
        assert_eq!(
            error.to_string(),
            format!(
                "{at}: {opcode} has no 32-bit form, which a host target without 16-bit floats \
                 needs"
            )
        );
        assert_eq!(function, original);
        assert!(run(&mut function, HostPass::Fp64ToFp32).is_ok(), "{opcode}: no f64 here");
    }
}

/// A narrowed value in an `Opaque` slot of an operation other than `Identity` is refused with
/// the slot and the type, directly and through an `Identity`; in `Identity`'s own slot it is
/// accepted.
#[test]
fn a_narrowed_value_in_an_opaque_slot_is_refused() {
    let cases: [(HostPass, fn(&mut Builder<'_>) -> Operand, Type); 2] = [
        (HostPass::Fp64ToFp32, double, Type::F64),
        (HostPass::Fp16ToFp32, half, Type::F16),
    ];
    for (pass, make, ty) in cases {
        for through_identity in [false, true] {
            let (original, marked) = build(|b| {
                let mut x = make(b);
                if through_identity {
                    x = Operand::Value(value(b, Opcode::Identity, &[x]));
                }
                vec![value(b, Opcode::Reference, &[x])]
            });
            let mut function = original.clone();
            let error = run(&mut function, pass).expect_err("refused");
            assert_eq!(
                error,
                HostLowerError::OpaqueSlot {
                    pass,
                    at: marked[0],
                    opcode: Opcode::Reference,
                    index: 0,
                    ty
                }
            );
            assert_eq!(
                error.to_string(),
                format!(
                    "{}: argument 0 of Reference is a {ty} value, which {pass} cannot narrow \
                     there",
                    marked[0]
                )
            );
            assert_eq!(function, original);
        }
        let (mut identity_only, _) = build(|b| {
            let x = make(b);
            vec![value(b, Opcode::Identity, &[x])]
        });
        assert!(run(&mut identity_only, pass).is_ok(), "{pass}");
    }
}

/// The scan reports the first fault in walk order (blocks in arena order, each block's
/// instructions in order), whichever check finds it: with an `Opaque`-slot fault in block 0
/// and a missing 32-bit form in block 1, the slot is reported, and with the two swapped the
/// missing form is; with both in one block, the one listed first.
#[test]
fn the_first_fault_in_walk_order_is_reported() {
    let build = |slot_first: bool, one_block: bool| {
        let mut function = Function::new();
        let entry = function.entry();
        let mut b = Builder::new(&mut function, entry);
        let second: BlockId = if one_block {
            entry
        } else {
            b.add_block().expect("a block")
        };
        let mut marked = Vec::new();
        for slot in [slot_first, !slot_first] {
            let h = half(&mut b);
            marked.push(if slot {
                value(&mut b, Opcode::Reference, &[h])
            } else {
                value(&mut b, Opcode::BitCastU16F16, &[h])
            });
            if !one_block && b.block() == entry {
                b.terminate(Terminator::Jump(second)).expect("jumps");
                b.switch_to(second).expect("a block");
            }
        }
        b.terminate(Terminator::Return).expect("returns");
        (function, marked)
    };
    for one_block in [false, true] {
        for slot_first in [true, false] {
            let (mut function, marked) = build(slot_first, one_block);
            let error = run(&mut function, HostPass::Fp16ToFp32).expect_err("refused");
            let wanted = if slot_first {
                HostLowerError::OpaqueSlot {
                    pass: HostPass::Fp16ToFp32,
                    at: marked[0],
                    opcode: Opcode::Reference,
                    index: 0,
                    ty: Type::F16,
                }
            } else {
                HostLowerError::NoNarrowForm {
                    pass: HostPass::Fp16ToFp32,
                    at: marked[0],
                    opcode: Opcode::BitCastU16F16,
                }
            };
            assert_eq!(error, wanted, "slot first {slot_first}, one block {one_block}");
        }
    }
}

/// With both passes needed and each refusing, the f64 pass's refusal is reported (the order);
/// with only the f16 pass refusing, nothing is changed, not even the 64-bit operations the f64
/// pass would have rewritten, since every check runs before either pass changes anything.
#[test]
fn a_refused_function_is_unchanged() {
    let (both, marked) = build(|b| {
        let h = half(b);
        let refused_half = value(b, Opcode::BitCastU16F16, &[h]);
        let d = double(b);
        let refused_double = value(b, Opcode::Reference, &[d]);
        vec![refused_half, refused_double]
    });
    let mut function = both.clone();
    let error = lower_for_host(&mut function, HostCaps::NONE).expect_err("refused");
    assert_eq!(
        error,
        HostLowerError::OpaqueSlot {
            pass: HostPass::Fp64ToFp32,
            at: marked[1],
            opcode: Opcode::Reference,
            index: 0,
            ty: Type::F64
        }
    );
    assert_eq!(function, both);

    let (half_only, marked) = build(|b| {
        let d = double(b);
        let neg = Operand::Value(value(b, Opcode::FPNeg64, &[d]));
        value(b, Opcode::UnpackDouble2x32, &[neg]);
        let h = half(b);
        vec![value(b, Opcode::BitCastU16F16, &[h])]
    });
    let mut function = half_only.clone();
    let error = lower_for_host(&mut function, HostCaps::NONE).expect_err("refused");
    assert_eq!(
        error,
        HostLowerError::NoNarrowForm {
            pass: HostPass::Fp16ToFp32,
            at: marked[0],
            opcode: Opcode::BitCastU16F16
        }
    );
    assert_eq!(function, half_only);
    let f16_target = HostCaps {
        f16: true,
        ..HostCaps::NONE
    };
    assert!(lower_for_host(&mut function, f16_target).is_ok(), "f64 alone lowers");
}

/// A malformed function is refused before anything is scanned, with the verifier's error.
#[test]
fn a_malformed_function_is_refused() {
    let mut function = Function::new();
    let error = function.verify().expect_err("an unterminated block");
    assert_eq!(
        run(&mut function.clone(), HostPass::Fp64ToFp32),
        Err(HostLowerError::Verify(error.clone()))
    );
    assert_eq!(
        lower_for_host(&mut function, HostCaps::NONE),
        Err(HostLowerError::Verify(error.clone()))
    );
    assert_eq!(
        HostLowerError::Verify(error.clone()).to_string(),
        format!("the IR is malformed: {error}")
    );
}
