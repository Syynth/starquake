//! IR tests: the text form and each verifier rejection.

use super::*;

fn control() -> FpControl {
    FpControl {
        rounding: RoundMode::NearestEven,
        ftz: false,
        fmz: false,
    }
}

/// A function whose arenas are given directly, to build malformed cases `push` cannot.
fn raw(insts: Vec<Inst>, order: Vec<u32>) -> Function {
    Function {
        insts,
        blocks: vec![Block {
            insts: order.into_iter().map(Value).collect(),
        }],
    }
}

#[test]
fn text_form_lists_every_instruction_in_order() {
    let mut f = Function::new();
    let b = f.entry();
    let imm = f.push(b, Inst::Imm32(0x3f80_0000)).expect("room");
    let one = f.push(b, Inst::BitCastF32(imm)).expect("room");
    let attr = f.push(b, Inst::GetAttribute(Attribute(32))).expect("room");
    let sum = f
        .push(
            b,
            Inst::FFma(
                one,
                attr,
                one,
                FpControl {
                    rounding: RoundMode::Zero,
                    ftz: true,
                    fmz: true,
                },
            ),
        )
        .expect("room");
    let neg = f.push(b, Inst::FNeg(sum)).expect("room");
    let abs = f.push(b, Inst::FAbs(neg)).expect("room");
    let sat = f.push(b, Inst::FSaturate(abs)).expect("room");
    f.push(b, Inst::SetAttribute(Attribute(28), sat))
        .expect("room");
    let bits = f.push(b, Inst::BitCastU32(sat)).expect("room");
    f.push(b, Inst::SetReg(Gpr::new(3), bits)).expect("room");
    let reg = f.push(b, Inst::GetReg(Gpr::new(3))).expect("room");
    let lane = f.push(b, Inst::LaneId).expect("room");
    let lane_f = f.push(b, Inst::BitCastF32(lane)).expect("room");
    f.push(
        b,
        Inst::SetFragColor {
            target: 2,
            component: 3,
            value: lane_f,
        },
    )
    .expect("room");
    assert_eq!(f.verify(), Ok(()));
    assert_eq!(reg.index(), 10);
    assert_eq!(
        f.to_string(),
        "block0:\n\
         \x20   %0: u32 = imm32 0x3f800000\n\
         \x20   %1: f32 = bitcast_f32 %0\n\
         \x20   %2: f32 = get_attribute a0x80\n\
         \x20   %3: f32 = ffma.rz.ftz.fmz %1, %2, %1\n\
         \x20   %4: f32 = fneg %3\n\
         \x20   %5: f32 = fabs %4\n\
         \x20   %6: f32 = fsaturate %5\n\
         \x20   set_attribute a0x70, %6\n\
         \x20   %8: u32 = bitcast_u32 %6\n\
         \x20   set_reg r3, %8\n\
         \x20   %10: u32 = get_reg r3\n\
         \x20   %11: u32 = lane_id\n\
         \x20   %12: f32 = bitcast_f32 %11\n\
         \x20   set_frag_color 2.3, %12\n"
    );
}

#[test]
fn verifier_rejects_an_id_with_no_instruction() {
    let f = raw(vec![Inst::LaneId], vec![0, 1]);
    assert_eq!(f.verify(), Err(VerifyError::Missing(Value(1))));
}

#[test]
fn verifier_rejects_an_id_listed_twice() {
    let f = raw(vec![Inst::LaneId], vec![0, 0]);
    assert_eq!(f.verify(), Err(VerifyError::Repeated(Value(0))));
}

#[test]
fn verifier_rejects_rz() {
    let f = raw(vec![Inst::GetReg(Gpr::RZ)], vec![0]);
    assert_eq!(f.verify(), Err(VerifyError::ZeroRegister(Value(0))));
    let f = raw(
        vec![Inst::LaneId, Inst::SetReg(Gpr::RZ, Value(0))],
        vec![0, 1],
    );
    assert_eq!(f.verify(), Err(VerifyError::ZeroRegister(Value(1))));
}

#[test]
fn verifier_rejects_a_use_before_its_definition() {
    // %1 is defined after %0 uses it.
    let f = raw(vec![Inst::BitCastF32(Value(1)), Inst::LaneId], vec![0, 1]);
    assert_eq!(
        f.verify(),
        Err(VerifyError::Undefined {
            at: Value(0),
            operand: Value(1)
        })
    );
    // An instruction using itself.
    let f = raw(vec![Inst::FNeg(Value(0))], vec![0]);
    assert_eq!(
        f.verify(),
        Err(VerifyError::Undefined {
            at: Value(0),
            operand: Value(0)
        })
    );
    // An operand in the arena but not in any block.
    let f = raw(vec![Inst::LaneId, Inst::BitCastF32(Value(0))], vec![1]);
    assert_eq!(
        f.verify(),
        Err(VerifyError::Undefined {
            at: Value(1),
            operand: Value(0)
        })
    );
}

#[test]
fn verifier_rejects_wrong_operand_types() {
    let f = raw(vec![Inst::LaneId, Inst::FNeg(Value(0))], vec![0, 1]);
    assert_eq!(
        f.verify(),
        Err(VerifyError::Type {
            at: Value(1),
            operand: Value(0),
            wanted: Type::F32
        })
    );
    let f = raw(
        vec![
            Inst::GetAttribute(Attribute(32)),
            Inst::SetReg(Gpr::new(0), Value(0)),
        ],
        vec![0, 1],
    );
    assert_eq!(
        f.verify(),
        Err(VerifyError::Type {
            at: Value(1),
            operand: Value(0),
            wanted: Type::U32
        })
    );
    // A store has no value to use.
    let f = raw(
        vec![
            Inst::GetAttribute(Attribute(32)),
            Inst::SetAttribute(Attribute(28), Value(0)),
            Inst::FNeg(Value(1)),
        ],
        vec![0, 1, 2],
    );
    assert_eq!(
        f.verify(),
        Err(VerifyError::Type {
            at: Value(2),
            operand: Value(1),
            wanted: Type::F32
        })
    );
    let c = control();
    let f = raw(
        vec![Inst::LaneId, Inst::FFma(Value(0), Value(0), Value(0), c)],
        vec![0, 1],
    );
    assert!(matches!(f.verify(), Err(VerifyError::Type { .. })));
}

#[test]
fn verifier_errors_read_as_words() {
    assert_eq!(
        VerifyError::Undefined {
            at: Value(4),
            operand: Value(7)
        }
        .to_string(),
        "%4 uses %7 before it is defined"
    );
    assert_eq!(
        VerifyError::Type {
            at: Value(1),
            operand: Value(0),
            wanted: Type::F32
        }
        .to_string(),
        "%1 needs %0 to be f32"
    );
}
