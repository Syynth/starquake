//! Every refusal with its exact error and message, the order of the checks with two faults
//! present at once, and each limit at its edge on both sides.

use super::*;
use crate::decode::{
    IMadMode, IntCmpOp, MovSrc, OpBMsk, OpBRev, OpFlo, OpIAbs, OpIAdd3X, OpIDp4, OpIMad, OpIMad64,
    OpIMnMx, OpISetP, OpLea, OpLeaX, OpLop3, OpPopC, OpPrmt, OpSel, OpSgxt, OpShf, PredSetOp,
    PrmtMode, ShfType,
};

const BINDLESS: AluSrc = AluSrc::CBuf {
    cbuf: CBufRef {
        buf: CBuf::BindlessUniform(UniformReg::URZ),
        offset: 0,
    },
    abs: false,
    neg: false,
};

fn lea() -> OpLea {
    OpLea {
        dst: r(0),
        overflow: Pred::PT,
        a: reg(1),
        b: reg(2),
        a_high: AluSrc::reg(Gpr::RZ),
        shift: 1,
        dst_high: false,
        sx32: false,
        intermediate: false,
        carry: NOT_PT,
    }
}

fn mov(src: MovSrc, quad_lanes: u8) -> Op {
    Op::Mov(OpMov {
        dst: r(1),
        src,
        quad_lanes,
    })
}

fn wide(dst: u8, c: AluSrc) -> OpIMad64 {
    OpIMad64 {
        dst: r(dst),
        srcs: [reg(1), reg(2), c],
        signed: false,
        carry: NOT_PT,
    }
}

fn isetp() -> OpISetP {
    OpISetP {
        dsts: [p(0), Pred::PT],
        srcs: [reg(1), reg(2)],
        cmp_op: IntCmpOp::Lt,
        signed: true,
        set_op: PredSetOp::And,
        ex: false,
        accum: PredSrc::TRUE,
        low_cmp: PredSrc::TRUE,
    }
}

fn prmt(sel: AluSrc, mode: PrmtMode) -> OpPrmt {
    OpPrmt {
        dst: r(0),
        srcs: [reg(1), reg(2)],
        sel,
        mode,
    }
}

fn idp(two_a: bool, signed: [bool; 2], srcs: [AluSrc; 3]) -> OpIDp4 {
    OpIDp4 {
        dst: r(0),
        srcs,
        signed,
        two_a,
    }
}

fn imad(srcs: [AluSrc; 3], mode: IMadMode) -> Op {
    Op::IMad(OpIMad {
        dst: r(0),
        srcs,
        signed: false,
        mode,
    })
}

/// Translates `op` through its word into a fresh function and returns the refusal; the
/// function is untouched.
fn refusal(op: Op) -> IntegerError {
    let mut function = Function::new();
    let entry = function.entry();
    let mut builder = Builder::new(&mut function, entry);
    let error = translate_integer_word(&mut builder, word(&Instr::new(op)), GuardMode::Honour)
        .expect_err("refused");
    assert_eq!(
        builder.function(),
        &Function::new(),
        "{op:?} touched the function"
    );
    error
}

fn accepted(op: Op) {
    assert!(build(&[Instr::new(op)]).is_ok(), "{op:?} refused");
}

/// Each check refuses with its named value and a message in plain words: `BMSK`, the bindless
/// source, the checks of `MOV`, `IADD3` and `LEA`, and `PRMT`'s mode.
#[test]
fn each_check_names_its_reason() {
    use IntegerUnsupported as U;
    let cases: Vec<(Op, IntegerUnsupported, &str)> = vec![
        (
            Op::BMsk(OpBMsk {
                dst: r(0),
                pos: reg(1),
                width: reg(2),
                wrap: false,
            }),
            U::Instruction("bmsk"),
            "bmsk, which neither reference translates, is not supported",
        ),
        (
            Op::IAdd3(iadd3(0, [reg(1), BINDLESS, reg(3)])),
            U::BindlessConstantBuffer,
            "a bindless constant-buffer source is not supported",
        ),
        (
            mov(MovSrc::Reg(r(2)), 0x7),
            U::MovLaneMask,
            "a mov with a partial lane mask is not supported",
        ),
        (
            Op::IAdd3(OpIAdd3 {
                carry: [ps(0, false), NOT_PT],
                ..iadd3(0, [reg(1), reg(2), reg(3)])
            }),
            U::Iadd3CarryPredicate,
            "an iadd3 carry input other than !pt is not supported",
        ),
        (
            Op::Lea(OpLea {
                intermediate: true,
                ..lea()
            }),
            U::LeaIntermediate,
            "a lea with its shifted value negated is not supported",
        ),
        (
            Op::Lea(OpLea {
                carry: ps(1, true),
                ..lea()
            }),
            U::LeaCarryPredicate,
            "a lea carry input other than !pt is not supported",
        ),
        (
            Op::Lea(OpLea {
                dst_high: true,
                overflow: p(2),
                ..lea()
            }),
            U::LeaHighCarryOutput,
            "a lea.hi with a carry output is not supported",
        ),
        (
            Op::Lea(OpLea {
                sx32: true,
                ..lea()
            }),
            U::LeaSignExtendWithoutHigh,
            "a lea.sx32 without .hi is not supported",
        ),
        (
            Op::Lea(OpLea {
                sx32: true,
                dst_high: true,
                a_high: reg(3),
                ..lea()
            }),
            U::LeaSignExtendWithHighWord,
            "a lea.hi.sx32 with a high word other than rz is not supported",
        ),
        (
            Op::Prmt(prmt(AluSrc::Imm(0x3210), PrmtMode::Forward4Extract)),
            U::PermuteMode,
            "a prmt mode other than the default is not supported",
        ),
    ];
    assert_refusals(cases);
}

/// The checks of `PRMT`'s selector, `IMAD`, `IMAD.WIDE`, `SGXT`, `IDP` and `ISETP`, as above.
#[test]
fn each_later_check_names_its_reason() {
    use IntegerUnsupported as U;
    let cases: Vec<(Op, IntegerUnsupported, &str)> = vec![
        (
            Op::Prmt(prmt(reg(3), PrmtMode::Index)),
            U::PermuteSelector,
            "a prmt selector that is not an immediate is not supported",
        ),
        (
            imad(
                [reg(1), reg(2), reg(3)],
                IMadMode::Lo {
                    x: false,
                    carry: ps(0, true),
                },
            ),
            U::ImadCarryInput,
            "an imad carry input other than !pt is not supported",
        ),
        (
            imad(
                [reg(1), neg(2), reg(3)],
                IMadMode::Lo {
                    x: true,
                    carry: ps(0, false),
                },
            ),
            U::ImadCarryNegatedFactor,
            "an imad.x with a negated factor is not supported",
        ),
        (
            Op::IMad64(OpIMad64 {
                carry: ps(3, true),
                ..wide(4, reg(6))
            }),
            U::WideCarryInput,
            "an imad.wide carry input other than !pt is not supported",
        ),
        (
            Op::IMad64(wide(5, reg(6))),
            U::WideDestination(r(5)),
            "an imad.wide destination pair at r5 is not supported",
        ),
        (
            Op::IMad64(wide(4, reg(7))),
            U::WideAddendRegister(r(7)),
            "an imad.wide addend pair at r7 is not supported",
        ),
        (
            Op::IMad64(wide(4, ureg(5))),
            U::WideAddendUniform(ur(5)),
            "an imad.wide addend pair at ur5 is not supported",
        ),
        (
            Op::IMad64(OpIMad64 {
                srcs: [reg(1), reg(2), AluSrc::Imm(9)],
                ..wide(4, reg(6))
            }),
            U::WideAddendImmediate,
            "an imad.wide immediate addend is not supported",
        ),
        (
            Op::Sgxt(OpSgxt {
                dst: r(0),
                a: reg(1),
                bits: reg(2),
                signed: true,
                wrap: true,
            }),
            U::SgxtWrap,
            "an sgxt.w is not supported",
        ),
        (
            Op::IDp4(idp(false, [false, false], [reg(1), reg(2), reg(3)])),
            U::IdpFourWay,
            "idp.4a, which neither reference translates, is not supported",
        ),
        (
            Op::IDp4(idp(true, [true, false], [reg(1), reg(2), reg(3)])),
            U::IdpModifiers,
            "an idp.2a with signed or negated sources is not supported",
        ),
        (
            Op::ISetP(OpISetP {
                low_cmp: ps(3, false),
                ..isetp()
            }),
            U::IsetpLowPredicate,
            "an isetp low-word predicate other than pt is not supported",
        ),
        (
            Op::ISetP(OpISetP {
                ex: true,
                ..isetp()
            }),
            U::IsetpExtended,
            "an isetp.ex is not supported",
        ),
    ];
    assert_refusals(cases);
}

/// Each case is refused with its value and message, and leaves the function untouched.
fn assert_refusals(cases: Vec<(Op, IntegerUnsupported, &str)>) {
    for (op, what, message) in cases {
        let error = refusal(op);
        assert_eq!(error, IntegerError::Unsupported(what), "{op:?}");
        assert_eq!(error.to_string(), message);
    }
}

/// Each modifier the `IDP.2A` check covers refuses on its own: either byte signedness, and the
/// negation of each source.
#[test]
fn idp_2a_refuses_each_modifier() {
    let plain = [reg(1), reg(2), reg(3)];
    let cases = [
        idp(true, [true, false], plain),
        idp(true, [false, true], plain),
        idp(true, [false, false], [neg(1), reg(2), reg(3)]),
        idp(true, [false, false], [reg(1), neg(2), reg(3)]),
        idp(true, [false, false], [reg(1), reg(2), neg(3)]),
    ];
    for op in cases {
        assert_eq!(
            refusal(Op::IDp4(op)),
            IntegerUnsupported::IdpModifiers.into(),
            "{op:?}"
        );
    }
    accepted(Op::IDp4(idp(true, [false, false], plain)));
}

/// The carry checks of `IADD3`: either carry input other than `!PT` refuses one without `.X`
/// (also with carry outputs), and `.X` accepts any.
#[test]
fn iadd3_carry_inputs() {
    let plain = iadd3(0, [reg(1), reg(2), reg(3)]);
    for carry in [
        [ps(0, false), NOT_PT],
        [NOT_PT, ps(0, false)],
        [PredSrc::TRUE, NOT_PT],
    ] {
        for overflow in [[Pred::PT; 2], [p(1), Pred::PT]] {
            let op = OpIAdd3 {
                overflow,
                carry,
                ..plain
            };
            assert_eq!(
                refusal(Op::IAdd3(op)),
                IntegerUnsupported::Iadd3CarryPredicate.into()
            );
            accepted(Op::IAdd3X(crate::decode::OpIAdd3X { add: op }));
        }
    }
    accepted(Op::IAdd3(plain));
}

/// `LEA.X` takes any carry input and `LEA.HI` without a carry output and `.SX32` with `RZ` as
/// its high word are accepted: the one-step-off cases of the checks above.
#[test]
fn lea_checks_accept_one_step_off() {
    accepted(Op::LeaX(OpLeaX {
        lea: OpLea {
            carry: ps(1, true),
            ..lea()
        },
    }));
    accepted(Op::Lea(OpLea {
        dst_high: true,
        ..lea()
    }));
    accepted(Op::Lea(OpLea {
        overflow: p(2),
        ..lea()
    }));
    accepted(Op::Lea(OpLea {
        sx32: true,
        dst_high: true,
        ..lea()
    }));
    accepted(Op::Prmt(prmt(AluSrc::Imm(0x3210), PrmtMode::Index)));
    accepted(imad(
        [reg(1), reg(2), neg(3)],
        IMadMode::Lo {
            x: true,
            carry: ps(0, false),
        },
    ));
    accepted(imad(
        [neg(1), neg(2), reg(3)],
        IMadMode::Lo {
            x: false,
            carry: NOT_PT,
        },
    ));
}

/// `IMAD.WIDE`'s pairs at their edges: the last even register below [`WIDE_PAIR_LIMIT`] is
/// accepted, the limit itself and the odd registers on either side refused; `RZ` is a zero
/// addend but no destination; a uniform pair must start even, so `URZ` is refused.
#[test]
fn wide_pairs_at_their_edges() {
    use IntegerUnsupported as U;
    let last = WIDE_PAIR_LIMIT - 2;
    accepted(Op::IMad64(wide(last, reg(6))));
    for dst in [last + 1, WIDE_PAIR_LIMIT, WIDE_PAIR_LIMIT + 1] {
        assert_eq!(
            refusal(Op::IMad64(wide(dst, reg(6)))),
            U::WideDestination(r(dst)).into()
        );
    }
    accepted(Op::IMad64(wide(4, reg(last))));
    accepted(Op::IMad64(wide(4, AluSrc::reg(Gpr::RZ))));
    for addend in [last + 1, WIDE_PAIR_LIMIT] {
        assert_eq!(
            refusal(Op::IMad64(wide(4, reg(addend)))),
            U::WideAddendRegister(r(addend)).into()
        );
    }
    let last_uniform = UniformReg::URZ.index() - 1;
    accepted(Op::IMad64(wide(4, ureg(last_uniform))));
    for addend in [last_uniform - 1, UniformReg::URZ.index()] {
        assert_eq!(
            refusal(Op::IMad64(wide(4, ureg(addend)))),
            U::WideAddendUniform(ur(addend)).into()
        );
    }
    accepted(Op::IMad64(wide(4, cb(1, 0xfffc))));
}

/// `WIDE_PAIR_LIMIT` against the reference's literal (`Visitor::IMAD_WIDE`: `limit = uniform_op
/// ? 62 : 254`, the vector datapath's 254): `R252` starts the last pair, as the destination and
/// as the addend, and `R254` is refused in both places.
#[test]
fn wide_pairs_match_the_reference_literal() {
    use IntegerUnsupported as U;
    accepted(Op::IMad64(wide(252, reg(6))));
    accepted(Op::IMad64(wide(4, reg(252))));
    assert_eq!(
        refusal(Op::IMad64(wide(254, reg(6)))),
        U::WideDestination(r(254)).into()
    );
    assert_eq!(
        refusal(Op::IMad64(wide(4, reg(254)))),
        U::WideAddendRegister(r(254)).into()
    );
}

/// When two checks fail, the one PR #4588 makes first is reported: the bindless source before
/// any instruction check, then each instruction's checks in the PR's order.
#[test]
fn the_first_failing_check_is_reported() {
    use IntegerUnsupported as U;
    let cases: Vec<(Op, IntegerUnsupported)> = vec![
        (
            mov(
                MovSrc::CBuf(CBufRef {
                    buf: CBuf::BindlessUniform(ur(2)),
                    offset: 0,
                }),
                0x3,
            ),
            U::BindlessConstantBuffer,
        ),
        (
            Op::IAdd3(OpIAdd3 {
                carry: [ps(0, false), NOT_PT],
                ..iadd3(0, [reg(1), reg(2), BINDLESS])
            }),
            U::BindlessConstantBuffer,
        ),
        (
            Op::Lea(OpLea {
                intermediate: true,
                carry: ps(1, false),
                ..lea()
            }),
            U::LeaIntermediate,
        ),
        (
            Op::Lea(OpLea {
                carry: ps(1, false),
                dst_high: true,
                overflow: p(2),
                ..lea()
            }),
            U::LeaCarryPredicate,
        ),
        (
            Op::Lea(OpLea {
                carry: ps(1, false),
                sx32: true,
                ..lea()
            }),
            U::LeaCarryPredicate,
        ),
        (
            Op::Lea(OpLea {
                dst_high: true,
                overflow: p(2),
                sx32: true,
                a_high: reg(3),
                ..lea()
            }),
            U::LeaHighCarryOutput,
        ),
        (Op::Prmt(prmt(reg(3), PrmtMode::Replicate8)), U::PermuteMode),
        (
            Op::IMad64(OpIMad64 {
                carry: ps(0, false),
                ..wide(5, reg(6))
            }),
            U::WideCarryInput,
        ),
        (Op::IMad64(wide(5, reg(7))), U::WideDestination(r(5))),
        (
            Op::IMad64(OpIMad64 {
                srcs: [reg(1), reg(2), AluSrc::Imm(1)],
                ..wide(5, reg(6))
            }),
            U::WideDestination(r(5)),
        ),
        (
            Op::IDp4(idp(false, [true, true], [neg(1), reg(2), reg(3)])),
            U::IdpFourWay,
        ),
        (
            Op::ISetP(OpISetP {
                ex: true,
                low_cmp: ps(3, false),
                ..isetp()
            }),
            U::IsetpLowPredicate,
        ),
        (
            Op::ISetP(OpISetP {
                srcs: [reg(1), BINDLESS],
                ex: true,
                ..isetp()
            }),
            U::BindlessConstantBuffer,
        ),
    ];
    for (op, what) in cases {
        assert_eq!(refusal(op), what.into(), "{op:?}");
    }
}

/// A bindless constant buffer is refused in every source position of every instruction that
/// reads ALU sources.
#[test]
fn bindless_buffers_are_refused_in_every_position() {
    let cases = [
        Op::IAdd3(iadd3(0, [reg(1), BINDLESS, reg(3)])),
        Op::IAdd3(iadd3(0, [reg(1), reg(2), BINDLESS])),
        Op::IMad64(wide(4, BINDLESS)),
        Op::Lea(OpLea {
            b: BINDLESS,
            ..lea()
        }),
        Op::Lea(OpLea {
            a_high: BINDLESS,
            dst_high: true,
            ..lea()
        }),
        Op::Sgxt(OpSgxt {
            dst: r(0),
            a: reg(1),
            bits: BINDLESS,
            signed: false,
            wrap: false,
        }),
        Op::BRev(crate::decode::OpBRev {
            dst: r(0),
            src: BINDLESS,
        }),
        mov(
            MovSrc::CBuf(CBufRef {
                buf: CBuf::BindlessUniform(ur(2)),
                offset: 4,
            }),
            OpMov::ALL_LANES,
        ),
    ];
    for op in cases {
        assert_eq!(
            build(&[Instr::new(op)]).err(),
            Some(IntegerUnsupported::BindlessConstantBuffer.into()),
            "{op:?}"
        );
    }
    let bound = Op::IAdd3(iadd3(0, [reg(1), cb(0, 0), reg(3)]));
    assert!(build(&[Instr::new(bound)]).is_ok());
}

/// Translates `op`, built by hand (not every case has a word), into a fresh function and
/// returns the result; a refusal leaves the function untouched.
fn translate_direct(op: Op) -> Result<(), IntegerError> {
    let mut function = Function::new();
    let entry = function.entry();
    let mut builder = Builder::new(&mut function, entry);
    let result = translate_integer(&mut builder, &Instr::new(op), GuardMode::Honour);
    if result.is_err() {
        assert_eq!(
            builder.function(),
            &Function::new(),
            "{op:?} touched the function"
        );
    }
    result
}

/// An instruction named for the failure message, how many sources it reads, and how to build
/// it from three sources in the order the bindless check scans them.
type SourceCase = (&'static str, usize, fn([AluSrc; 3]) -> Op);

/// Every source position of every instruction that reads ALU sources.
fn every_source_position() -> Vec<SourceCase> {
    let mut cases = three_source_cases();
    cases.extend(fewer_source_cases());
    cases
}

/// The instructions with three ALU sources.
fn three_source_cases() -> Vec<SourceCase> {
    vec![
        ("iadd3", 3, |s| Op::IAdd3(iadd3(0, s))),
        ("iadd3.x", 3, |s| Op::IAdd3X(OpIAdd3X { add: iadd3(0, s) })),
        ("lop3", 3, |s| {
            Op::Lop3(OpLop3 {
                pred: Pred::PT,
                dst: r(0),
                srcs: s,
                lut: 0x96,
            })
        }),
        ("imad", 3, |s| {
            imad(
                s,
                IMadMode::Lo {
                    x: false,
                    carry: NOT_PT,
                },
            )
        }),
        ("imad.hi", 3, |s| imad(s, IMadMode::Hi)),
        ("imad.wide", 3, |s| {
            Op::IMad64(OpIMad64 {
                srcs: s,
                ..wide(4, AluSrc::reg(Gpr::RZ))
            })
        }),
        ("idp", 3, |s| Op::IDp4(idp(true, [false; 2], s))),
        ("lea", 3, |[a, b, a_high]| {
            Op::Lea(OpLea {
                a,
                b,
                a_high,
                dst_high: true,
                ..lea()
            })
        }),
        ("lea.x", 3, |[a, b, a_high]| {
            Op::LeaX(OpLeaX {
                lea: OpLea {
                    a,
                    b,
                    a_high,
                    dst_high: true,
                    ..lea()
                },
            })
        }),
        // The selector is the middle source: a register when another source is bindless, so a
        // missed `b` would be refused as `PermuteSelector` instead.
        ("prmt", 3, |[a, sel, b]| {
            Op::Prmt(OpPrmt {
                dst: r(0),
                srcs: [a, b],
                sel,
                mode: PrmtMode::Index,
            })
        }),
        ("shf", 3, |[low, shift, high]| {
            Op::Shf(OpShf {
                dst: r(0),
                low,
                shift,
                high,
                data_type: ShfType::U32,
                wrap: false,
                right: false,
                dst_high: false,
            })
        }),
    ]
}

/// The instructions with one or two ALU sources.
fn fewer_source_cases() -> Vec<SourceCase> {
    vec![
        ("iabs", 1, |[src, ..]| Op::IAbs(OpIAbs { dst: r(0), src })),
        ("brev", 1, |[src, ..]| Op::BRev(OpBRev { dst: r(0), src })),
        ("popc", 1, |[src, ..]| Op::PopC(OpPopC { dst: r(0), src })),
        ("flo", 1, |[src, ..]| {
            Op::Flo(OpFlo {
                dst: r(0),
                src,
                signed: false,
                return_shift_amount: false,
            })
        }),
        ("imnmx", 2, |[a, b, _]| {
            Op::IMnMx(OpIMnMx {
                dst: r(0),
                srcs: [a, b],
                signed: false,
                min: PredSrc::TRUE,
            })
        }),
        ("sel", 2, |[a, b, _]| {
            Op::Sel(OpSel {
                dst: r(0),
                srcs: [a, b],
                cond: PredSrc::TRUE,
            })
        }),
        ("isetp", 2, |[a, b, _]| {
            Op::ISetP(OpISetP {
                srcs: [a, b],
                ..isetp()
            })
        }),
        ("sgxt", 2, |[a, bits, _]| {
            Op::Sgxt(OpSgxt {
                dst: r(0),
                a,
                bits,
                signed: false,
                wrap: false,
            })
        }),
        ("bmsk", 2, |[pos, width, _]| {
            Op::BMsk(OpBMsk {
                dst: r(0),
                pos,
                width,
                wrap: false,
            })
        }),
    ]
}

/// A bindless constant buffer is refused in each source position of each instruction, built
/// by hand (`IADD3`'s `a` and others have no word with a constant buffer there), whatever
/// check the instruction would make next: `PRMT`'s `b` with a register selector included.
#[test]
fn bindless_buffers_are_refused_in_every_source_of_every_instruction() {
    for (name, count, make) in every_source_position() {
        for position in 0..count {
            let mut srcs = [reg(2), reg(4), reg(6)];
            if let Some(slot) = srcs.get_mut(position) {
                *slot = BINDLESS;
            }
            assert_eq!(
                translate_direct(make(srcs)),
                Err(IntegerUnsupported::BindlessConstantBuffer.into()),
                "{name} source {position}"
            );
        }
    }
}

/// A hand-built source modifier this module would not apply is refused by the instruction's
/// mnemonic, with the function untouched: a negation on `IMAD.WIDE`'s, `IMAD.HI`'s, `IABS`'s
/// and `SHF`'s sources and on `LEA`'s `a` and high word (`IMAD.WIDE` with `-R2 * R4` would
/// otherwise write the product un-negated), and an absolute value on a source that takes a
/// negation. The negations this module applies still translate.
#[test]
fn modifiers_the_form_drops_are_refused() {
    let abs = AluSrc::Reg {
        reg: r(2),
        abs: true,
        neg: false,
    };
    let refused: [(Op, &str); 9] = [
        (
            Op::IMad64(OpIMad64 {
                srcs: [neg(2), reg(4), AluSrc::reg(Gpr::RZ)],
                ..wide(4, AluSrc::reg(Gpr::RZ))
            }),
            "imad.wide",
        ),
        (imad([reg(2), reg(4), neg(6)], IMadMode::Hi), "imad"),
        (
            Op::IAbs(OpIAbs {
                dst: r(0),
                src: neg(2),
            }),
            "iabs",
        ),
        (Op::Lea(OpLea { a: neg(1), ..lea() }), "lea"),
        (
            Op::Lea(OpLea {
                a_high: neg(3),
                dst_high: true,
                ..lea()
            }),
            "lea",
        ),
        (
            Op::Shf(OpShf {
                dst: r(0),
                low: reg(1),
                shift: neg(2),
                high: reg(3),
                data_type: ShfType::U32,
                wrap: false,
                right: false,
                dst_high: false,
            }),
            "shf",
        ),
        (Op::IAdd3(iadd3(0, [reg(1), abs, reg(3)])), "iadd3"),
        (
            Op::PopC(OpPopC {
                dst: r(0),
                src: abs,
            }),
            "popc",
        ),
        (Op::Lea(OpLea { b: abs, ..lea() }), "lea"),
    ];
    for (op, mnemonic) in refused {
        let refusal = translate_direct(op);
        assert_eq!(refusal, Err(IntegerError::Modifier(mnemonic)), "{op:?}");
        assert_eq!(
            refusal.map_err(|error| error.to_string()),
            Err(format!("{mnemonic} has a source modifier it does not take"))
        );
    }
    let applied = [
        Op::IAdd3(iadd3(0, [neg(1), neg(2), neg(3)])),
        imad(
            [neg(1), neg(2), neg(3)],
            IMadMode::Lo {
                x: false,
                carry: NOT_PT,
            },
        ),
        Op::Lea(OpLea { b: neg(2), ..lea() }),
        Op::PopC(OpPopC {
            dst: r(0),
            src: neg(2),
        }),
        Op::Flo(OpFlo {
            dst: r(0),
            src: neg(2),
            signed: false,
            return_shift_amount: false,
        }),
    ];
    for op in applied {
        assert_eq!(translate_direct(op), Ok(()), "{op:?}");
    }
}

/// `PRMT`'s forms 2, 3 and 7 put the selector in the register slot (and `b` in the wide
/// slot), so they are refused as PR #4588 refuses them; form 1 (a register selector) too.
/// Form 4, the immediate selector, is the one translated.
#[test]
fn prmt_takes_only_an_immediate_selector() {
    let mut machine = Machine::new();
    machine.uniform_regs[1] = 1;
    for b in [AluSrc::Imm(5), cb(0, 0), ureg(1), reg(2)] {
        let op = OpPrmt {
            dst: r(0),
            srcs: [reg(1), b],
            sel: reg(3),
            mode: PrmtMode::Index,
        };
        assert_eq!(
            refusal(Op::Prmt(op)),
            IntegerUnsupported::PermuteSelector.into(),
            "{b:?}"
        );
    }
    accepted(Op::Prmt(prmt(AluSrc::Imm(0x7654), PrmtMode::Index)));
}
