//! Tests of slice 6h, the control and warp group (`op::control`): NAK's expected text, one
//! hand-built word per form, branch offsets at their edges, the values and bits the group
//! refuses, the divergences from PR #4588's table, and generated round trips of the group alone.

use proptest::prelude::*;
use proptest::strategy::Union;

use super::super::op::control::{self, *};
use super::*;
use crate::asm::{EncodeError, assemble};

/// A word built by hand: `opcode`, the guard `PT` (7 at bit 12), and each `(start, value)` placed
/// at its bit, with positions written out here rather than taken from the decoder's constants.
fn word(opcode: u128, fields: &[(u32, u128)]) -> Instruction {
    let mut bits = opcode | (7 << 12);
    for &(start, value) in fields {
        bits |= value << start;
    }
    Instruction::from_bits(bits)
}

/// Decodes a hand-built word and prints it.
fn text(word: Instruction) -> String {
    match decode(word) {
        Ok(instr) => instr.to_string(),
        Err(error) => format!("error: {error}"),
    }
}

fn ur(index: u8) -> UniformReg {
    UniformReg::new(index).expect("a uniform register")
}

fn b(index: u8) -> Barrier {
    Barrier::new(index).expect("a barrier")
}

fn pt() -> PredSrc {
    PredSrc::TRUE
}

/// NAK's `test_nop`, ported (`nvdisasm_tests.rs` @ 52238762): `OpNop` prints `nop;`.
#[test]
fn nop_text_matches_nak_nvdisasm_case() {
    assert_eq!(round_text(Instr::new(Op::Nop(OpNop {}))), "nop;");
}

/// NAK's `test_redux`, ported: `ur0` and `r1` with each operation and its text on SM 8.0 and
/// later.
#[test]
fn redux_text_matches_nak_nvdisasm_cases() {
    let cases = [
        (ReduxOp::And, ""),
        (ReduxOp::Or, ".or"),
        (ReduxOp::Xor, ".xor"),
        (ReduxOp::Sum, ".sum"),
        (ReduxOp::Min { signed: false }, ".min"),
        (ReduxOp::Max { signed: false }, ".max"),
        (ReduxOp::Min { signed: true }, ".min.s32"),
        (ReduxOp::Max { signed: true }, ".max.s32"),
    ];
    for (op, suffix) in cases {
        let instr = Instr::new(Op::Redux(OpRedux {
            op,
            dst: ur(0),
            src: r(1),
        }));
        assert_eq!(round_text(instr), format!("redux{suffix} ur0, r1;"));
    }
}

/// NAK's `test_match`, ported: `.all` with the predicate `p1`, `.any` with none (`PT`), each on
/// a 32-bit (`r4`) and a 64-bit (`r4` and `r5`, `.u64`) value.
#[test]
fn match_text_matches_nak_nvdisasm_cases() {
    let mut checked = 0;
    for (op, pred, pred_str, op_str) in [
        (MatchOp::All, p(1), "p1, ", ".all"),
        (MatchOp::Any, Pred::PT, "", ".any"),
    ] {
        for (u64, u64_str) in [(false, ""), (true, ".u64")] {
            let instr = Instr::new(Op::Match(OpMatch {
                op,
                u64,
                mask: r(3),
                src: r(4),
                pred,
            }));
            assert_eq!(
                round_text(instr),
                format!("match{op_str}{u64_str} {pred_str}r3, r4;")
            );
            checked += 1;
        }
    }
    assert_eq!(checked, 4);
}

/// NAK's `test_nanosleep`, ported: the sources `0` (encoded as `RZ`), `0x87654321`, `r3`,
/// `c[0x5][0x100]` and `cx[ur2][0x100]`. NAK's expected strings end in ` ;`: the test gives
/// the instruction a delay of 1, `nvdisasm` prints the scheduling text after the operands, and
/// NAK's harness cuts it off and appends ` ;` (`disassemble_instrs`, "Remove possible scheduling
/// infos"). The decoder ignores scheduling bits, so the space is dropped before comparing.
#[test]
fn nanosleep_text_matches_nak_nvdisasm_cases() {
    let unmodified = |cbuf| AluSrc::CBuf {
        cbuf,
        abs: false,
        neg: false,
    };
    let cases = [
        (AluSrc::reg(Gpr::RZ), "nanosleep rz ;"),
        (AluSrc::Imm(0x8765_4321), "nanosleep 0x87654321 ;"),
        (AluSrc::reg(r(3)), "nanosleep r3 ;"),
        (
            unmodified(CBufRef {
                buf: CBuf::Binding(5),
                offset: 0x100,
            }),
            "nanosleep c[0x5][0x100] ;",
        ),
        (
            unmodified(CBufRef {
                buf: CBuf::BindlessUniform(ur(2)),
                offset: 0x100,
            }),
            "nanosleep cx[ur2][0x100] ;",
        ),
    ];
    for (time, nak) in cases {
        let instr = Instr::new(Op::Nanosleep(OpNanosleep {
            time,
            clear: false,
            warp: false,
            rand: false,
            cond: pt(),
        }));
        assert_eq!(round_text(instr), nak.replace(" ;", ";"));
    }
}

/// One hand-built word per form, and each modifier bit, condition negation and operation field
/// set alone in some word, with the text it decodes to. Field positions are written out as
/// numbers, so a wrong position in the decoder fails here. **Unconfirmed text** beyond NAK's
/// cases above (see `print` and `op::control`).
#[test]
fn hand_built_words_of_each_form_decode() {
    let cases = [branch_words(), barrier_words(), warp_words()].concat();
    let mut opcodes: Vec<u16> = Vec::new();
    for (word, expected) in &cases {
        assert_eq!(text(*word), *expected, "{:#x}", word.bits());
        opcodes.push(u16::try_from(word.field(OPCODE)).expect("12 bits"));
    }
    // Every form of the group has a hand-built word here.
    for form in control::FORMS {
        assert!(opcodes.contains(&form.opcode), "{:#x}", form.opcode);
    }
}

/// Hand-built words of `BRA`, `BRX`, `CALL`, `RET`, `EXIT` and `KILL`.
fn branch_words() -> Vec<(Instruction, &'static str)> {
    vec![
        // BRA 0x947: condition PT (7 at 87), offset 4 words at 34.
        (word(0x947, &[(87, 7), (34, 4)]), "bra +0x10;"),
        // The condition !p2 (2 at 87, bit 90).
        (word(0x947, &[(87, 2), (90, 1), (34, 4)]), "bra !p2, +0x10;"),
        // Bit 32 (.U) alone.
        (word(0x947, &[(87, 7), (32, 1), (34, 4)]), "bra.u +0x10;"),
        // A negative offset: -3 words in 48 bits.
        (word(0x947, &[(87, 7), (34, (1 << 48) - 3)]), "bra -0xc;"),
        // BRA 0x547: fixed 7 at 87 and bit 91; uniform condition up1 at 24, its negation bit 27.
        (
            word(0x547, &[(87, 7), (91, 1), (24, 1), (27, 1), (34, 4)]),
            "bra !up1, +0x10;",
        ),
        (
            word(0x547, &[(87, 7), (91, 1), (24, 7), (32, 1), (34, 1)]),
            "bra.u upt, +0x4;",
        ),
        // BRX 0x949: r5 at 24, condition PT.
        (word(0x949, &[(24, 5), (87, 7), (34, 4)]), "brx r5 +0x10;"),
        (
            word(0x949, &[(24, 5), (87, 0), (34, (1 << 48) - 4)]),
            "brx p0, r5 -0x10;",
        ),
        // BRXU 0x958: ur3 at 24, fixed bit 91.
        (
            word(0x958, &[(24, 3), (91, 1), (87, 7), (34, 4)]),
            "brx ur3 +0x10;",
        ),
        // CALL 0x944: bit 86 (.NOINC) set and clear.
        (
            word(0x944, &[(86, 1), (87, 7), (34, 4)]),
            "call.rel.noinc +0x10;",
        ),
        (
            word(0x944, &[(87, 6), (90, 1), (34, 4)]),
            "call.rel !p6, +0x10;",
        ),
        // RET 0x950: r20 at 24, bit 86 (.NODEC).
        (
            word(0x950, &[(24, 20), (86, 1), (87, 7), (34, 4)]),
            "ret.rel.nodec r20 +0x10;",
        ),
        (
            word(0x950, &[(24, 20), (87, 7), (34, (1 << 48) - 1)]),
            "ret.rel r20 -0x4;",
        ),
        // EXIT 0x94d: bits 84 and 85 alone.
        (word(0x94d, &[(87, 7), (84, 1)]), "exit.keeprefcount;"),
        (word(0x94d, &[(87, 7), (85, 1)]), "exit.no_atexit;"),
        // KILL 0x95b.
        (word(0x95b, &[(87, 7)]), "kill;"),
        (word(0x95b, &[(87, 3), (90, 1)]), "kill !p3;"),
    ]
}

/// Hand-built words of `BSSY`, `BSYNC`, `BREAK`, `BMOV`, `WARPSYNC` and `BAR`.
fn barrier_words() -> Vec<(Instruction, &'static str)> {
    vec![
        // BSSY 0x945: b2 at 16, offset 8 words.
        (word(0x945, &[(16, 2), (87, 7), (34, 8)]), "bssy b2, +0x20;"),
        // BSYNC 0x941: b15.
        (word(0x941, &[(16, 15), (87, 7)]), "bsync b15;"),
        // The condition !PT (never) is printed; only plain PT is left out.
        (word(0x941, &[(87, 7), (90, 1)]), "bsync !pt, b0;"),
        // BREAK 0x942: b1 under p0.
        (word(0x942, &[(16, 1), (87, 0)]), "break p0, b1;"),
        // BMOV 0x356: b4 at 24, r9 at 32; bit 84 (.CLEAR) alone.
        (word(0x356, &[(24, 4), (32, 9)]), "bmov.32 b4, r9;"),
        (
            word(0x356, &[(24, 4), (32, 9), (84, 1)]),
            "bmov.32.clear b4, r9;",
        ),
        // BMOV 0x355: r6 at 16, b7 at 24.
        (word(0x355, &[(16, 6), (24, 7)]), "bmov.32 r6, b7;"),
        // WARPSYNC 0x948 (form 4): the immediate at 32.
        (
            word(0x948, &[(32, 0xffff_ffff), (87, 7)]),
            "warpsync 0xffffffff;",
        ),
        // 0x348 (form 1): r4 at 32, under !p5.
        (
            word(0x348, &[(32, 4), (87, 5), (90, 1)]),
            "warpsync !p5, r4;",
        ),
        // 0xb48 (form 5): offset 0x40 at 38, buffer 2 at 54.
        (
            word(0xb48, &[(38, 0x40), (54, 2), (87, 7)]),
            "warpsync c[0x2][0x40];",
        ),
        // 0xd48 (form 6): ur5 at 32, fixed bit 91.
        (word(0xd48, &[(32, 5), (91, 1), (87, 7)]), "warpsync ur5;"),
        // BAR 0xb1d: every field zero (NAK's encoding), then each field alone.
        (word(0xb1d, &[]), "bar.sync r0, p0;"),
        (word(0xb1d, &[(87, 7)]), "bar.sync r0;"),
        (word(0xb1d, &[(77, 1), (87, 7)]), "bar.arv r0;"),
        (word(0xb1d, &[(77, 2), (87, 7)]), "bar.red r0;"),
        (word(0xb1d, &[(77, 3), (87, 7)]), "bar.scan r0;"),
        (word(0xb1d, &[(74, 1), (87, 7)]), "bar.sync.and r0;"),
        (word(0xb1d, &[(74, 2), (87, 7)]), "bar.sync.or r0;"),
        (
            word(0xb1d, &[(80, 1), (87, 7)]),
            "bar.sync.defer_blocking r0;",
        ),
        (
            word(0xb1d, &[(32, 3), (87, 2), (90, 1)]),
            "bar.sync r3, !p2;",
        ),
    ]
}

/// Hand-built words of `VOTE`, `SHFL`, `MATCH`, `REDUX`, `NOP`, `DEPBAR` and `NANOSLEEP`.
fn warp_words() -> Vec<(Instruction, &'static str)> {
    vec![
        // VOTE 0x806: r2 at 16, any (1 at 72), p3 at 81, voted !p1.
        (
            word(0x806, &[(16, 2), (72, 1), (81, 3), (87, 1), (90, 1)]),
            "vote.any r2, p3, !p1;",
        ),
        (
            word(0x806, &[(16, 0xff), (81, 7), (87, 7)]),
            "vote.all rz, pt, pt;",
        ),
        // VOTEU 0x886: ur4 at 16, eq (2 at 72), upt at 81.
        (
            word(0x886, &[(16, 4), (72, 2), (81, 7), (87, 7)]),
            "voteu.eq ur4, upt, pt;",
        ),
        // SHFL 0x389: r1 at 16, r2 at 24, lane r3 at 32, clamp r4 at 64, up (1 at 58), PT at 81.
        (
            word(
                0x389,
                &[(16, 1), (24, 2), (32, 3), (64, 4), (58, 1), (81, 7)],
            ),
            "shfl.up pt, r1, r2, r3, r4;",
        ),
        // 0x589: the clamp 0x1f1f at 40, in-bounds p0.
        (
            word(0x589, &[(16, 1), (24, 2), (32, 3), (40, 0x1f1f)]),
            "shfl.idx p0, r1, r2, r3, 0x1f1f;",
        ),
        // 0x989: the lane 0x1f at 53, down (2 at 58).
        (
            word(
                0x989,
                &[(16, 1), (24, 2), (53, 0x1f), (64, 4), (58, 2), (81, 7)],
            ),
            "shfl.down pt, r1, r2, 0x1f, r4;",
        ),
        // 0xf89: the lane 1 at 53, the clamp 0x1c03 at 40, bfly (3 at 58).
        (
            word(
                0xf89,
                &[(16, 1), (24, 2), (53, 1), (40, 0x1c03), (58, 3), (81, 7)],
            ),
            "shfl.bfly pt, r1, r2, 0x1, 0x1c03;",
        ),
        // MATCH 0x3a1: r3 at 16, r4 at 24, p1 at 81; bit 73 (.U64) alone, bit 79 (.ANY) alone.
        (
            word(0x3a1, &[(16, 3), (24, 4), (81, 1)]),
            "match.all p1, r3, r4;",
        ),
        (
            word(0x3a1, &[(16, 3), (24, 4), (81, 7), (73, 1)]),
            "match.all.u64 r3, r4;",
        ),
        (
            word(0x3a1, &[(16, 3), (24, 4), (81, 7), (79, 1)]),
            "match.any r3, r4;",
        ),
        // REDUX 0x3c4: ur6 at 16, r7 at 24, the operation at 78; bit 73 with MIN.
        (
            word(0x3c4, &[(16, 6), (24, 7), (78, 3)]),
            "redux.sum ur6, r7;",
        ),
        (
            word(0x3c4, &[(16, 6), (24, 7), (78, 4), (73, 1)]),
            "redux.min.s32 ur6, r7;",
        ),
        (
            word(0x3c4, &[(16, 6), (24, 7), (78, 5)]),
            "redux.max ur6, r7;",
        ),
        // NOP 0x918.
        (word(0x918, &[]), "nop;"),
        // DEPBAR 0x91a: PR #4588's `DEPBAR.LE SB5, 1` (count 1 at 38, 0xd at 44).
        (word(0x91a, &[(38, 1), (44, 0xd)]), "depbar.le sb5, 0x1;"),
        (word(0x91a, &[(38, 0x3f), (44, 7)]), "depbar sb7, 0x3f;"),
        // NANOSLEEP 0x95d (form 4); bits 83, 85 and 86 alone.
        (
            word(0x95d, &[(32, 0x8765_4321), (87, 7)]),
            "nanosleep 0x87654321;",
        ),
        (word(0x95d, &[(87, 7), (83, 1)]), "nanosleep.clear 0x0;"),
        (word(0x95d, &[(87, 7), (85, 1)]), "nanosleep.warp 0x0;"),
        (word(0x95d, &[(87, 7), (86, 1)]), "nanosleep.rand 0x0;"),
        // 0x35d (form 1): RZ at 32, under p4.
        (word(0x35d, &[(32, 0xff), (87, 4)]), "nanosleep p4, rz;"),
        // 0xb5d (form 5): bindless, ur2 at 32, offset 0x100 at 38, bit 91.
        (
            word(0xb5d, &[(32, 2), (38, 0x100), (91, 1), (87, 7)]),
            "nanosleep cx[ur2][0x100];",
        ),
        // 0xd5d (form 6): ur2 at 32, fixed bit 91.
        (word(0xd5d, &[(32, 2), (91, 1), (87, 7)]), "nanosleep ur2;"),
    ]
}

/// `BCLEAR`: NAK's `impl SM70Op for OpBClear` writes `0x355`, the destination `RZ`, the barrier
/// at 24..28 and bit 84; it decodes as `BMOV` from the barrier into `RZ` with `.CLEAR`.
#[test]
fn bclear_decodes_as_bmov_clear_into_rz() {
    let bclear = word(0x355, &[(16, 0xff), (24, 3), (84, 1)]);
    assert_eq!(
        decode(bclear).map(|instr| instr.op),
        Ok(Op::BMov(OpBMov {
            dir: BMovDir::FromBarrier {
                dst: Gpr::RZ,
                bar: b(3),
            },
            clear: true,
        }))
    );
    assert_eq!(text(bclear), "bmov.32.clear rz, b3;");
}

/// The decoded value of a hand-built word's branch offset.
fn offset_of(word: Instruction) -> Option<i64> {
    match decode(word).ok()?.op {
        Op::Bra(op) => Some(op.offset.words()),
        Op::Brx(op) => Some(op.offset.words()),
        Op::Call(op) => Some(op.offset.words()),
        Op::Ret(op) => Some(op.offset.words()),
        Op::BSSy(op) => Some(op.offset.words()),
        _ => None,
    }
}

/// Branch offsets at the most negative and most positive values of their fields, in hand-built
/// words: 48 bits for `BRA`, `BRX`, `CALL` and `RET`, 30 for `BSSY`. Bit 63 alone is the sign of
/// `BSSY`'s offset and a positive offset of `BRA`'s, so a wrong width fails.
#[test]
fn branch_offsets_at_the_edges_of_their_fields() {
    let min48 = -(1_i64 << 47);
    let max48 = (1_i64 << 47) - 1;
    for (opcode, extra) in [
        (0x947, vec![(87, 7)]),
        (0x949, vec![(87, 7), (24, 1)]),
        (0x944, vec![(87, 7)]),
        (0x950, vec![(87, 7), (24, 1)]),
    ] {
        let at = |raw: u128| {
            let mut fields = extra.clone();
            fields.push((34, raw));
            word(opcode, &fields)
        };
        assert_eq!(offset_of(at(1 << 47)), Some(min48), "{opcode:#x}");
        assert_eq!(offset_of(at((1 << 47) - 1)), Some(max48), "{opcode:#x}");
        assert_eq!(offset_of(at((1 << 48) - 1)), Some(-1), "{opcode:#x}");
        assert_eq!(offset_of(at(1 << 29)), Some(1 << 29), "{opcode:#x}");
    }
    let min30 = -(1_i64 << 29);
    let max30 = (1_i64 << 29) - 1;
    let bssy = |raw: u128| word(0x945, &[(87, 7), (34, raw)]);
    assert_eq!(offset_of(bssy(1 << 29)), Some(min30));
    assert_eq!(offset_of(bssy((1 << 29) - 1)), Some(max30));
    assert_eq!(offset_of(bssy((1 << 30) - 1)), Some(-1));
    // The bytes and the text at the edges.
    let bra = |raw| word(0x947, &[(87, 7), (34, raw)]);
    assert_eq!(text(bra(1 << 47)), "bra -0x2000000000000;");
    assert_eq!(text(bra((1 << 47) - 1)), "bra +0x1fffffffffffc;");
    assert_eq!(text(bssy(1 << 29)), "bssy b0, -0x80000000;");
    assert_eq!(text(bssy((1 << 29) - 1)), "bssy b0, +0x7ffffffc;");
    assert_eq!(
        BranchOffset::from_words(min48).bytes(),
        Some(min48 * BranchOffset::WORD_BYTES)
    );
    // Only offsets no field can hold overflow in bytes.
    let last = i64::MAX / BranchOffset::WORD_BYTES;
    assert_eq!(
        BranchOffset::from_words(last).bytes(),
        Some(last * BranchOffset::WORD_BYTES)
    );
    assert_eq!(BranchOffset::from_words(last + 1).bytes(), None);
    assert_eq!(BranchOffset::from_words(i64::MIN).bytes(), None);
    assert_eq!(
        BranchOffset::from_words(i64::MIN).to_string(),
        "-0x20000000000000000"
    );
}

/// The assembler refuses an offset one past either end of its field, naming the field's width.
#[test]
fn offsets_past_their_field_are_refused() {
    type Encoder<'a> = &'a dyn Fn(i64) -> Result<Instruction, EncodeError>;
    let bra = |words| {
        assemble(&Instr::new(Op::Bra(OpBra {
            cond: BraCond::Pred(pt()),
            uniform: false,
            offset: BranchOffset::from_words(words),
        })))
    };
    let bssy = |words| {
        assemble(&Instr::new(Op::BSSy(OpBSSy {
            bar: b(0),
            offset: BranchOffset::from_words(words),
            cond: pt(),
        })))
    };
    let encoders: [(Encoder<'_>, u32); 2] = [(&bra, 48), (&bssy, 30)];
    for (encode, width) in encoders {
        let half = 1_i64 << (width - 1);
        for words in [-half, half - 1] {
            let word = encode(words).expect("in range");
            assert_eq!(offset_of(word), Some(words));
        }
        for value in [-half - 1, half] {
            assert_eq!(
                encode(value),
                Err(EncodeError::OutOfRange {
                    what: "branch offset",
                    value,
                    bits: width
                })
            );
        }
    }
    assert_eq!(
        bssy(1 << 29).map_err(|error| error.to_string()),
        Err("branch offset 536870912 does not fit in 30 signed bits".into())
    );
}

/// Field values that name nothing are refused at step 4 with the exact value and field, and
/// the last defined value next to each is accepted.
#[test]
fn undefined_field_values_are_refused() {
    // VOTE's operation at 72..74: 2 (EQ) decodes, 3 does not.
    let vote = |op| word(0x806, &[(72, op), (81, 7), (87, 7)]);
    assert_eq!(text(vote(2)), "vote.eq r0, pt, pt;");
    assert_eq!(
        decode(vote(3)),
        Err(DecodeError::UnknownValue {
            opcode: 0x806,
            bit: 72,
            width: 2,
            value: 3
        })
    );
    let voteu = word(0x886, &[(72, 3), (81, 7), (87, 7)]);
    assert_eq!(
        decode(voteu),
        Err(DecodeError::UnknownValue {
            opcode: 0x886,
            bit: 72,
            width: 2,
            value: 3
        })
    );
    // REDUX's operation at 78..81: 5 (MAX) decodes, 6 and 7 do not.
    let redux = |op| word(0x3c4, &[(78, op)]);
    assert_eq!(text(redux(5)), "redux.max ur0, r0;");
    for value in [6, 7] {
        assert_eq!(
            decode(redux(value)),
            Err(DecodeError::UnknownValue {
                opcode: 0x3c4,
                bit: 78,
                width: 3,
                value: u64::try_from(value).expect("small")
            })
        );
    }
    // BAR's reduction at 74..76: 2 (OR) decodes, 3 does not.
    let bar = |red| word(0xb1d, &[(74, red), (87, 7)]);
    assert_eq!(text(bar(2)), "bar.sync.or r0;");
    assert_eq!(
        decode(bar(3)),
        Err(DecodeError::UnknownValue {
            opcode: 0xb1d,
            bit: 74,
            width: 2,
            value: 3
        })
    );
    assert_eq!(
        decode(vote(3)).map_err(|error| error.to_string()),
        Err(
            "opcode 0x806 has the value 3 in its 2-bit field at bit 72, which its form does not \
             define"
                .into()
        )
    );
}

/// `REDUX`'s signed bit is read only for `MIN` and `MAX`: with another operation it is an
/// unknown bit (step 5).
#[test]
fn redux_signed_bit_only_with_min_and_max() {
    for op in 0..4 {
        assert_eq!(
            decode(word(0x3c4, &[(78, op), (73, 1)])),
            Err(DecodeError::UnknownBit {
                opcode: 0x3c4,
                bit: 73
            }),
            "{op}"
        );
    }
    assert_eq!(
        text(word(0x3c4, &[(78, 4), (73, 1)])),
        "redux.min.s32 ur0, r0;"
    );
    assert_eq!(
        text(word(0x3c4, &[(78, 5), (73, 1)])),
        "redux.max.s32 ur0, r0;"
    );
}

/// The order of checks with the group's faults: a stray bit (step 2) before a missing fixed bit
/// (step 3); a missing fixed bit before an undefined value (step 4) cannot meet in one form, so
/// step 2 before step 4 and step 4 before step 5 are checked instead.
#[test]
fn control_checks_run_in_the_documented_order() {
    // Step 2 before step 3: a uniform BRA without bit 91 and with bit 100 set.
    let uniform_bra = word(0x547, &[(87, 7), (100, 1)]);
    assert_eq!(
        decode(uniform_bra),
        Err(DecodeError::UnknownBit {
            opcode: 0x547,
            bit: 100
        })
    );
    // Step 3 alone.
    assert_eq!(
        decode(word(0x547, &[(87, 7)])),
        Err(DecodeError::MissingBit {
            opcode: 0x547,
            bit: 91
        })
    );
    // Step 2 before step 4: a VOTE with operation 3 and bit 100 set.
    assert_eq!(
        decode(word(0x806, &[(72, 3), (100, 1)])),
        Err(DecodeError::UnknownBit {
            opcode: 0x806,
            bit: 100
        })
    );
    // Step 4 before step 5: a REDUX with operation 6 and the signed bit, which it does not read.
    assert_eq!(
        decode(word(0x3c4, &[(78, 6), (73, 1)])),
        Err(DecodeError::UnknownValue {
            opcode: 0x3c4,
            bit: 78,
            width: 3,
            value: 6
        })
    );
}

/// The uniform `BRA` requires each of bits 87, 88, 89 and 91 set and `BRXU` bit 91, naming the
/// lowest clear one; with all of them set the words decode.
#[test]
fn control_fixed_bits_are_required() {
    let all = [(87, 1), (88, 1), (89, 1), (91, 1)];
    for missing in 0..all.len() {
        let fields: Vec<(u32, u128)> = all
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != missing)
            .map(|(_, &field)| field)
            .collect();
        assert_eq!(
            decode(word(0x547, &fields)),
            Err(DecodeError::MissingBit {
                opcode: 0x547,
                bit: all[missing].0
            })
        );
    }
    // Two missing at once: the lowest is named.
    assert_eq!(
        decode(word(0x547, &[(87, 1), (91, 1)])),
        Err(DecodeError::MissingBit {
            opcode: 0x547,
            bit: 88
        })
    );
    assert!(decode(word(0x547, &all)).is_ok());
    assert_eq!(
        decode(word(0x958, &[(87, 7)])),
        Err(DecodeError::MissingBit {
            opcode: 0x958,
            bit: 91
        })
    );
    assert!(decode(word(0x958, &[(87, 7), (91, 1)])).is_ok());
}

/// `WARPSYNC` and `NANOSLEEP` reject the wide slot's modifier bits (62, 63) in each form whose
/// PR #4588 row marks them known, and the assembler refuses a modified source.
#[test]
fn control_sources_reject_modifier_bits() {
    // Every row of the group that drops bits drops exactly these two.
    for row in control::REFERENCE_ROWS {
        assert!(
            row.dropped == 0 || row.dropped == SLOT_ABS.mask() | SLOT_NEG.mask(),
            "{:#x}",
            row.opcode
        );
    }
    let dropping: Vec<u16> = control::REFERENCE_ROWS
        .iter()
        .filter(|row| row.dropped != 0)
        .map(|row| row.opcode)
        .collect();
    assert_eq!(dropping, [0x348, 0xb48, 0xd48, 0x35d, 0xb5d, 0xd5d]);
    for opcode in dropping {
        let fixed = lookup(opcode).expect("a form").fixed;
        let plain = Instruction::from_bits(word(u128::from(opcode), &[(87, 7)]).bits() | fixed);
        assert!(decode(plain).is_ok(), "{opcode:#x}");
        for bit in [62, 63] {
            assert_eq!(
                decode(plain.with_flag(Field::bit(bit), true)),
                Err(DecodeError::UnknownBit { opcode, bit }),
                "{opcode:#x}"
            );
        }
    }
    let modified = AluSrc::Reg {
        reg: r(1),
        abs: false,
        neg: true,
    };
    assert_eq!(
        assemble(&Instr::new(Op::WarpSync(OpWarpSync {
            mask: modified,
            cond: pt()
        }))),
        Err(EncodeError::Modifier {
            mnemonic: "warpsync",
            index: 1
        })
    );
    assert_eq!(
        assemble(&Instr::new(Op::Nanosleep(OpNanosleep {
            time: modified,
            clear: false,
            warp: false,
            rand: false,
            cond: pt()
        }))),
        Err(EncodeError::Modifier {
            mnemonic: "nanosleep",
            index: 1
        })
    );
}

/// The forms PR #4588's table has rows for that this decoder leaves unknown: `WARPSYNC` and
/// `NANOSLEEP` forms 2, 3 and 7, and `BAR`'s `0x31d`; and opcodes next to the group's.
#[test]
fn control_rows_left_undecoded_are_unknown_opcodes() {
    let unknown = [
        AluForm::ImmSrc2.opcode(WARPSYNC_OPCODE),
        AluForm::CBufSrc2.opcode(WARPSYNC_OPCODE),
        AluForm::UniformSrc2.opcode(WARPSYNC_OPCODE),
        AluForm::ImmSrc2.opcode(NANOSLEEP_OPCODE),
        AluForm::CBufSrc2.opcode(NANOSLEEP_OPCODE),
        AluForm::UniformSrc2.opcode(NANOSLEEP_OPCODE),
        0x31d,
        // No row: JMX and BPT have no known opcode; these are neighbours of decoded ones.
        0x943,
        0x946,
        0x951,
    ];
    assert_eq!(
        unknown[..6],
        [0x548, 0x748, 0xf48, 0x55d, 0x75d, 0xf5d],
        "the rows' opcodes"
    );
    for opcode in unknown {
        let word = Instruction::default().with_field(OPCODE, u64::from(opcode));
        assert_eq!(decode(word), Err(DecodeError::UnknownOpcode { opcode }));
    }
}

/// The two shells of the group print their mnemonic and have no encoding.
#[test]
fn jmx_and_bpt_stay_shells() {
    let shells: Vec<&str> = control::SHELLS.iter().map(Op::mnemonic).collect();
    assert_eq!(shells, ["jmx", "bpt"]);
}

/// Barrier registers at the edges of their field.
#[test]
fn barriers_at_the_edges_of_their_encoding() {
    let last = Barrier::COUNT - 1;
    assert_eq!(u64::from(last), (1 << Barrier::BITS) - 1);
    assert_eq!(Barrier::new(last).map(Barrier::index), Some(15));
    assert_eq!(Barrier::new(last + 1), None);
    assert_eq!(Barrier::new(u8::MAX), None);
    for index in [0, last] {
        let instr = Instr::new(Op::BSync(OpBSync {
            bar: b(index),
            cond: pt(),
        }));
        assert_eq!(round_text(instr), format!("bsync b{index};"));
    }
}

/// The assembler refuses `SHFL` immediates and `DEPBAR` fields one past their widths, and
/// accepts them at the edge.
#[test]
fn control_immediates_past_their_field_are_refused() {
    let shfl = |lane, c| {
        assemble(&Instr::new(Op::Shfl(OpShfl {
            op: ShflOp::Idx,
            dst: r(1),
            in_bounds: Pred::PT,
            src: r(2),
            lane,
            c,
        })))
    };
    assert!(shfl(ShflSrc::Imm(0x1f), ShflSrc::Imm(0x1fff)).is_ok());
    assert_eq!(
        shfl(ShflSrc::Imm(0x20), ShflSrc::Reg(r(4))),
        Err(EncodeError::TooWide {
            what: "shuffle lane",
            value: 0x20,
            bits: 5
        })
    );
    assert_eq!(
        shfl(ShflSrc::Reg(r(3)), ShflSrc::Imm(0x2000)),
        Err(EncodeError::TooWide {
            what: "shuffle clamp",
            value: 0x2000,
            bits: 13
        })
    );
    assert_eq!(
        shfl(ShflSrc::Imm(0x20), ShflSrc::Imm(0x2000)),
        Err(EncodeError::TooWide {
            what: "shuffle clamp",
            value: 0x2000,
            bits: 13
        })
    );
    let depbar = |scoreboard, count| {
        assemble(&Instr::new(Op::DepBar(OpDepBar {
            scoreboard,
            le: false,
            count,
        })))
    };
    assert!(depbar(7, 63).is_ok());
    assert_eq!(
        depbar(8, 0),
        Err(EncodeError::TooWide {
            what: "dependency scoreboard",
            value: 8,
            bits: 3
        })
    );
    assert_eq!(
        depbar(0, 64),
        Err(EncodeError::TooWide {
            what: "dependency count",
            value: 64,
            bits: 6
        })
    );
}

/// The group's opcodes against the cited reference values, written out: NAK's `sm70_encode.rs`
/// @ 52238762 for its encoders, PR #4588's `ampere_isa_table.h` and `visitor.cpp` @ e5fb3b97 for
/// `BRX`, `CALL`, `RET` and `DEPBAR`.
#[test]
fn control_opcodes_match_the_references() {
    let opcodes = [
        (BRA_OPCODE, 0x947),
        (BRA_UNIFORM_OPCODE, 0x547),
        (BRX_OPCODE, 0x949),
        (BRXU_OPCODE, 0x958),
        (CALL_OPCODE, 0x944),
        (RET_OPCODE, 0x950),
        (EXIT_OPCODE, 0x94d),
        (KILL_OPCODE, 0x95b),
        (BSSY_OPCODE, 0x945),
        (BSYNC_OPCODE, 0x941),
        (BREAK_OPCODE, 0x942),
        (BMOV_TO_BARRIER_OPCODE, 0x356),
        (BMOV_FROM_BARRIER_OPCODE, 0x355),
        (WARPSYNC_OPCODE, 0x148),
        (NANOSLEEP_OPCODE, 0x15d),
        (BAR_OPCODE, 0xb1d),
        (VOTE_OPCODE, 0x806),
        (VOTEU_OPCODE, 0x886),
        (SHFL_OPCODE, 0x389),
        (SHFL_IMM_C_OPCODE, 0x589),
        (SHFL_IMM_LANE_OPCODE, 0x989),
        (SHFL_IMM_BOTH_OPCODE, 0xf89),
        (MATCH_OPCODE, 0x3a1),
        (REDUX_OPCODE, 0x3c4),
        (NOP_OPCODE, 0x918),
        (DEPBAR_OPCODE, 0x91a),
    ];
    for (shipping, reference) in opcodes {
        assert_eq!(shipping, reference);
    }
}

/// The group's field positions against the cited reference ranges, written out (NAK's
/// `set_field`, `set_bit` and helper ranges; PR #4588's `Take` calls).
#[test]
fn control_fields_match_the_references() {
    let fields = [
        (COND, (87, 3)),
        (COND_NOT, (90, 1)),
        (PRED_DST, (81, 3)),
        (UNIFORM_DST, (16, 6)),
        (BRANCH_OFFSET, (34, 48)),
        (BSSY_OFFSET, (34, 30)),
        (CONV_BARRIER, (16, 4)),
        (UNIFORM_BIT, (91, 1)),
        (BRA_U, (32, 1)),
        (BRA_UPRED, (24, 3)),
        (BRA_UPRED_NOT, (27, 1)),
        (BRX_REG, (24, 8)),
        (BRX_UNIFORM_REG, (24, 6)),
        (CALL_NO_INC, (86, 1)),
        (RET_REG, (24, 8)),
        (RET_NO_DEC, (86, 1)),
        (EXIT_KEEP_REF_COUNT, (84, 1)),
        (EXIT_NO_AT_EXIT, (85, 1)),
        (BMOV_BARRIER, (24, 4)),
        (BMOV_SRC, (32, 8)),
        (BMOV_CLEAR, (84, 1)),
        (NANOSLEEP_CLEAR, (83, 1)),
        (NANOSLEEP_WARP, (85, 1)),
        (NANOSLEEP_RAND, (86, 1)),
        (BAR_SRC, (32, 8)),
        (BAR_RED, (74, 2)),
        (BAR_MODE, (77, 2)),
        (BAR_DEFER_BLOCKING, (80, 1)),
        (VOTE_OP, (72, 2)),
        (SHFL_SRC, (24, 8)),
        (SHFL_LANE_REG, (32, 8)),
        (SHFL_LANE_IMM, (53, 5)),
        (SHFL_C_REG, (64, 8)),
        (SHFL_C_IMM, (40, 13)),
        (SHFL_OP, (58, 2)),
        (MATCH_SRC, (24, 8)),
        (MATCH_U64, (73, 1)),
        (MATCH_ANY, (79, 1)),
        (REDUX_SRC, (24, 8)),
        (REDUX_SIGNED, (73, 1)),
        (REDUX_OP, (78, 3)),
        (DEPBAR_COUNT, (38, 6)),
        (DEPBAR_SCOREBOARD, (44, 3)),
        (DEPBAR_LE, (47, 1)),
    ];
    for (shipping, (start, width)) in fields {
        assert_eq!((shipping.start, shipping.width), (start, width));
    }
}

/// The enumerations' encodings against NAK's `VoteOp`, `ShflOp` and `ReduxOp` encoders and its
/// comment on `BAR`'s fields, written out.
#[test]
fn control_enumerations_match_the_references() {
    let votes: Vec<u64> = VoteOp::ALL.iter().map(|op| op.bits()).collect();
    assert_eq!(votes, [0, 1, 2]);
    assert_eq!(VoteOp::ALL, [VoteOp::All, VoteOp::Any, VoteOp::Eq]);
    assert_eq!(VoteOp::from_bits(3), None);
    assert_eq!(
        ShflOp::ALL,
        [ShflOp::Idx, ShflOp::Up, ShflOp::Down, ShflOp::Bfly]
    );
    let shfls: Vec<u64> = ShflOp::ALL.iter().map(|op| op.bits()).collect();
    assert_eq!(shfls, [0, 1, 2, 3]);
    let reduxes = [
        (ReduxOp::And, 0),
        (ReduxOp::Or, 1),
        (ReduxOp::Xor, 2),
        (ReduxOp::Sum, 3),
        (ReduxOp::Min { signed: true }, 4),
        (ReduxOp::Max { signed: false }, 5),
    ];
    for (op, bits) in reduxes {
        assert_eq!(op.bits(), bits);
    }
    assert_eq!(
        BarMode::ALL,
        [
            BarMode::Sync,
            BarMode::Arrive,
            BarMode::Reduce,
            BarMode::Scan
        ]
    );
    let modes: Vec<u64> = BarMode::ALL.iter().map(|mode| mode.bits()).collect();
    assert_eq!(modes, [0, 1, 2, 3]);
    assert_eq!(BarRed::ALL, [BarRed::Popc, BarRed::And, BarRed::Or]);
    let reds: Vec<u64> = BarRed::ALL.iter().map(|red| red.bits()).collect();
    assert_eq!(reds, [0, 1, 2]);
    assert_eq!(BarRed::from_bits(3), None);
    assert_eq!(BranchOffset::WORD_BYTES, 4);
    assert_eq!(Barrier::COUNT, 16);
}

/// More unconfirmed text: the printer's guesses for operands the hand-built words do not reach.
#[test]
fn unconfirmed_text_of_the_control_group() {
    let cases = [
        (
            Op::Bra(OpBra {
                cond: BraCond::UniformPred(UniformPredSrc {
                    reg: UniformPred::UPT,
                    negated: false,
                }),
                uniform: true,
                offset: BranchOffset::from_words(0),
            }),
            "bra.u upt, +0x0;",
        ),
        (
            Op::Vote(OpVote {
                op: VoteOp::All,
                dst: VoteDst::Uniform {
                    ballot: UniformReg::URZ,
                    vote: UniformPred::new(2).expect("up2"),
                },
                pred: PredSrc {
                    reg: p(6),
                    negated: true,
                },
            }),
            "voteu.all urz, up2, !p6;",
        ),
        (
            Op::BMov(OpBMov {
                dir: BMovDir::ToBarrier {
                    bar: b(15),
                    src: Gpr::RZ,
                },
                clear: true,
            }),
            "bmov.32.clear b15, rz;",
        ),
    ];
    for (op, expected) in cases {
        assert_eq!(round_text(Instr::new(op)), expected);
    }
}

/// A word of one of the group's forms (as `testing::any_form_word`, over this group alone).
fn any_control_word() -> impl Strategy<Value = Instruction> {
    let forms: Vec<(u16, u128, u128)> = control::FORMS
        .iter()
        .map(|form| (form.opcode, form.known, form.fixed))
        .collect();
    (0..forms.len(), any::<u128>()).prop_map(move |(index, noise)| {
        let (opcode, known, fixed) = forms[index];
        Instruction::from_bits((noise & known) | fixed).with_field(OPCODE, u64::from(opcode))
    })
}

proptest! {
    #![proptest_config(config())]

    /// The group's operations alone, so each gets many more cases than in the crate-wide round
    /// trip: assembling then decoding gives the instruction back.
    #[test]
    fn control_assemble_then_decode_round_trips(
        guard in crate::testing::any_pred_src(),
        op in Union::new(control::any_ops()),
    ) {
        let instr = Instr { guard, op };
        let word = assemble(&instr).expect("the generator makes encodable instructions");
        prop_assert_eq!(decode(word), Ok(instr));
    }

    /// The group's words alone: a word decodes and re-assembles to itself (scheduling bits
    /// aside), or is refused for a bit its layout does not read (`REDUX`'s signed bit, or the
    /// other constant-buffer layout's bits in `WARPSYNC` and `NANOSLEEP`) or for a value that
    /// names nothing.
    #[test]
    fn control_decode_then_assemble_round_trips(word in any_control_word()) {
        match decode(word) {
            Ok(instr) => {
                let again = assemble(&instr).expect("a decoded instruction is encodable");
                prop_assert_eq!(again.bits(), word.bits() & !SCHEDULING.mask());
            }
            Err(DecodeError::UnknownBit { opcode, bit }) => {
                let read = bits_read(word).expect("the form's function accepts the word");
                prop_assert_eq!(read & (1 << bit), 0);
                let layout = (opcode == REDUX_OPCODE && bit == REDUX_SIGNED.start)
                    || [
                        AluForm::CBufSrc1.opcode(WARPSYNC_OPCODE),
                        AluForm::CBufSrc1.opcode(NANOSLEEP_OPCODE),
                    ]
                    .contains(&opcode);
                prop_assert!(layout, "{opcode:#x} bit {bit}");
            }
            Err(DecodeError::UnknownValue { opcode, bit, .. }) => {
                prop_assert!(
                    [(VOTE_OPCODE, VOTE_OP.start), (VOTEU_OPCODE, VOTE_OP.start),
                     (REDUX_OPCODE, REDUX_OP.start), (BAR_OPCODE, BAR_RED.start)]
                        .contains(&(opcode, bit))
                );
            }
            Err(error) => prop_assert!(false, "{error}"),
        }
    }
}
