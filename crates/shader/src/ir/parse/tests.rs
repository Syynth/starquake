//! Parser tests: one hand-written accepting case per feature of the format, and each refusal
//! with its exact error and line.

use super::super::*;
use super::{ParseError, ParseErrorKind};
use crate::decode::{Gpr, Pred};

fn ir(text: &str) -> Function {
    Function::parse(text).expect("parses")
}

/// The error parsing `text` gives, as words.
fn refused(text: &str) -> String {
    Function::parse(text)
        .err()
        .map_or_default(|error| error.to_string())
}

/// The first argument of the only instruction of `operand`'s one-line function.
fn operand(text: &str) -> Operand {
    let f = ir(&format!("block0:\n    %0 = Reference {text}\n"));
    first_op(&f).args[0]
}

/// The first instruction of `f`, which must be an operation.
fn first_op(f: &Function) -> &Operation {
    match f.inst(Value(0)) {
        Some(Inst::Op(op)) => Some(op),
        _ => None,
    }
    .expect("an operation")
}

fn kind(text: &str) -> Option<ParseErrorKind> {
    Function::parse(text).err().map(|error| error.kind)
}

/// The kind of error parsing `text` as an operand gives.
fn operand_kind(text: &str) -> Option<ParseErrorKind> {
    kind(&format!("block0:\n    %0 = Reference {text}\n"))
}

#[test]
fn comments_blank_lines_and_spacing_are_ignored() {
    let plain = ir("block0:\n    %0: u32 = LaneId\n    return\n");
    let noisy = ir("// a comment\n\nblock0:   // the entry\n  %0:u32=LaneId\n\n\treturn // done\n");
    assert_eq!(noisy, plain);
    assert_eq!(
        "block0:\n    return\n".parse::<Function>(),
        Ok(ir("block0:\n    return\n"))
    );
}

#[test]
fn result_types_may_be_left_out_on_operations() {
    assert_eq!(
        ir("block0:\n    %0 = LaneId\n"),
        ir("block0:\n    %0: u32 = LaneId\n")
    );
    assert_eq!(
        kind("block0:\n    %0: f32 = LaneId\n"),
        Some(ParseErrorKind::Annotation {
            opcode: Opcode::LaneId,
            wanted: Type::U32,
            found: Type::F32
        })
    );
    assert_eq!(
        kind("block0:\n    %0: u32 = Barrier\n"),
        Some(ParseErrorKind::Annotation {
            opcode: Opcode::Barrier,
            wanted: Type::Void,
            found: Type::U32
        })
    );
    assert_eq!(
        kind("block0:\n    %0 = Phi\n"),
        Some(ParseErrorKind::PhiType)
    );
}

#[test]
fn numbers_read_in_decimal_and_hexadecimal() {
    assert_eq!(operand("u32 42"), operand("u32 0x2a"));
    assert_eq!(operand("u8 255"), Operand::Imm(Imm::U8(u8::MAX)));
    assert_eq!(operand("u16 0xffff"), Operand::Imm(Imm::U16(u16::MAX)));
    assert_eq!(operand("u32 0xffffffff"), Operand::Imm(Imm::U32(u32::MAX)));
    assert_eq!(
        operand("u64 18446744073709551615"),
        Operand::Imm(Imm::U64(u64::MAX))
    );
    assert_eq!(
        operand("f32 1.5"),
        Operand::Imm(Imm::F32(1.5_f32.to_bits()))
    );
    assert_eq!(
        operand("f32 -2"),
        Operand::Imm(Imm::F32((-2.0_f32).to_bits()))
    );
    assert_eq!(
        operand("f32 0x7fc00001"),
        Operand::Imm(Imm::F32(0x7fc0_0001))
    );
    assert_eq!(
        operand("f64 0.25"),
        Operand::Imm(Imm::F64(0.25_f64.to_bits()))
    );
    assert_eq!(operand("u1 false"), Operand::Imm(Imm::U1(false)));
    let number = |text: &str| Some(ParseErrorKind::Number(text.to_owned()));
    assert_eq!(operand_kind("u8 256"), number("256"));
    assert_eq!(operand_kind("u16 0x10000"), number("0x10000"));
    assert_eq!(operand_kind("u32 0x100000000"), number("0x100000000"));
    assert_eq!(
        operand_kind("u64 18446744073709551616"),
        number("18446744073709551616")
    );
    assert_eq!(operand_kind("f32 0x100000000"), number("0x100000000"));
    assert_eq!(operand_kind("f32 one"), number("one"));
    assert_eq!(operand_kind("f64 x"), number("x"));
    assert_eq!(operand_kind("u1 1"), number("1"));
    assert_eq!(operand_kind("u32 -1"), number("-1"));
    assert_eq!(operand_kind("%0x1"), number("%0x1"));
}

/// Each name kind at its last valid number and one past it.
#[test]
fn names_read_up_to_their_edges() {
    let imm = |text: &str| operand(text).imm().expect("an immediate");
    let unknown = |text: &str| Some(ParseErrorKind::UnknownOperand(text.to_owned()));
    let last_r = Gpr::RZ.index() - 1;
    assert_eq!(imm(&format!("r{last_r}")), Imm::Reg(Gpr::new(last_r)));
    assert_eq!(imm("rz"), Imm::Reg(Gpr::RZ));
    assert_eq!(
        operand_kind(&format!("r{}", Gpr::RZ.index())),
        unknown("r255")
    );
    let last_p = Pred::PT.index() - 1;
    assert_eq!(
        imm(&format!("p{last_p}")),
        Imm::Pred(Pred::new(last_p).expect("exists"))
    );
    assert_eq!(imm("pt"), Imm::Pred(Pred::PT));
    assert_eq!(
        operand_kind(&format!("p{}", Pred::PT.index())),
        unknown("p7")
    );
    let last_uniform = UniformReg::URZ.index() - 1;
    assert_eq!(
        imm(&format!("ur{last_uniform}")),
        Imm::UniformReg(UniformReg::new(last_uniform).expect("exists"))
    );
    assert_eq!(imm("urz"), Imm::UniformReg(UniformReg::URZ));
    assert_eq!(
        operand_kind(&format!("ur{}", UniformReg::URZ.index())),
        unknown("ur63")
    );
    let last_upred = UniformPred::UPT.index() - 1;
    assert_eq!(
        imm(&format!("up{last_upred}")),
        Imm::UniformPred(UniformPred::new(last_upred).expect("exists"))
    );
    assert_eq!(imm("upt"), Imm::UniformPred(UniformPred::UPT));
    assert_eq!(
        operand_kind(&format!("up{}", UniformPred::UPT.index())),
        unknown("up7")
    );
    let last_patch = Patch::COUNT - 1;
    assert_eq!(
        imm(&format!("patch{last_patch}")),
        Imm::Patch(Patch::new(last_patch).expect("exists"))
    );
    assert_eq!(
        operand_kind(&format!("patch{}", Patch::COUNT)),
        unknown("patch126")
    );
    // Attributes: the byte address of a word; the last word a `u16` can number, and one past.
    let last_address = u32::from(u16::MAX) * ATTRIBUTE_WORD_BYTES;
    assert_eq!(
        imm(&format!("a{last_address:#x}")),
        Imm::Attribute(Attribute(u16::MAX))
    );
    let past = last_address + ATTRIBUTE_WORD_BYTES;
    assert_eq!(operand_kind(&format!("a{past:#x}")), unknown("a0x40000"));
    assert_eq!(operand_kind("a0x81"), unknown("a0x81"));
    assert_eq!(operand_kind("r1x"), unknown("r1x"));
    assert_eq!(operand_kind("r"), unknown("r"));
    assert_eq!(operand_kind("r+1"), unknown("r+1"));
    assert_eq!(operand_kind("u32x2 0x1"), unknown("u32x2 0x1"));
    assert_eq!(
        operand_kind("i32 1"),
        Some(ParseErrorKind::UnknownType("i32".to_owned()))
    );
}

#[test]
fn ids_the_text_skips_become_detached_slots() {
    let f = ir("block0:\n    %3: u32 = LaneId\n    %1 = Barrier\n");
    let arena: Vec<Inst> = f.insts().map(|(_, inst)| inst.clone()).collect();
    assert_eq!(
        arena,
        [
            Inst::detached(),
            Inst::Op(Operation::new(Opcode::Barrier, vec![])),
            Inst::detached(),
            Inst::Op(Operation::new(Opcode::LaneId, vec![])),
        ]
    );
    let listed: Vec<u32> = f.listed().map(|(value, _)| value.index()).collect();
    assert_eq!(listed, [3, 1]);
    assert_eq!(
        f.to_string(),
        "block0:\n    %3: u32 = LaneId\n    %1 = Barrier\n"
    );
}

/// An id may be as large as the text is long, and no larger.
#[test]
fn ids_are_bounded_by_the_text_length() {
    // The smallest text whose one id equals its own length.
    let text_with = |id: usize| format!("block0:\n    %{id} = Barrier\n");
    let mut id = 0;
    while text_with(id).len() != id {
        id = text_with(id).len();
    }
    let at_limit = text_with(id);
    assert_eq!(at_limit.len(), id);
    assert_eq!(ir(&at_limit).insts().count(), id + 1);
    let past = text_with(id + 1);
    assert_eq!(past.len(), id, "same number of digits");
    let value = Value(u32::try_from(id + 1).expect("small"));
    assert_eq!(kind(&past), Some(ParseErrorKind::IdTooLarge(value)));
    assert_eq!(
        kind("block0:\n    %4294967296 = Barrier\n"),
        Some(ParseErrorKind::Number("4294967296".to_owned()))
    );
}

#[test]
fn flags_of_each_kind() {
    let flags = |text: &str| first_op(&ir(&format!("block0:\n    %0 = {text}\n"))).flags;
    assert_eq!(
        flags("FPAdd32 f32 1, f32 2 {fmz_none, rm, no_contraction}"),
        Flags::Fp(FpControl {
            no_contraction: true,
            rounding: FpRounding::Rm,
            fmz_mode: FmzMode::None,
        })
    );
    assert_eq!(
        flags("FPAdd32 f32 1, f32 2 {}"),
        Flags::Fp(FpControl::default())
    );
    assert_eq!(
        flags("ImageRead u32 0, u32 0 {integer, gather=3, type=buffer, format=r8_sint}"),
        Flags::Texture(TextureInfo {
            is_integer: true,
            gather_component: TEXTURE_TWO_BIT_MAX,
            ty: TextureType::Buffer,
            image_format: ImageFormat::R8Sint,
            ..TextureInfo::default()
        })
    );
    let flag = |word: &str| Some(ParseErrorKind::Flag(word.to_owned()));
    let fp = |words: &str| {
        kind(&format!(
            "block0:\n    %0 = FPAdd32 f32 1, f32 2 {{{words}}}\n"
        ))
    };
    let tex = |words: &str| {
        kind(&format!(
            "block0:\n    %0 = ImageRead u32 0, u32 0 {{{words}}}\n"
        ))
    };
    assert_eq!(kind("block0:\n    %0 = LaneId {ftz}\n"), flag("ftz"));
    assert_eq!(fp("rz, rn"), flag("rn"));
    assert_eq!(fp("ftz, fmz"), flag("fmz"));
    assert_eq!(fp("no_contraction, no_contraction"), flag("no_contraction"));
    assert_eq!(fp("depth"), flag("depth"));
    assert_eq!(tex("depth, depth"), flag("depth"));
    assert_eq!(tex("rz"), flag("rz"));
    assert_eq!(tex("depth=1"), flag("depth=1"));
    assert_eq!(tex("type"), flag("type"));
    assert_eq!(tex("type=4d"), flag("type=4d"));
    assert_eq!(tex("format=rgba"), flag("format=rgba"));
    assert_eq!(
        tex(&format!("gather={}", TEXTURE_TWO_BIT_MAX + 1)),
        Some(ParseErrorKind::Number("4".to_owned()))
    );
    assert_eq!(
        tex(&format!("derivatives={}", TEXTURE_TWO_BIT_MAX + 1)),
        Some(ParseErrorKind::Number("4".to_owned()))
    );
    assert_eq!(
        tex("descriptor=65536"),
        Some(ParseErrorKind::Number("65536".to_owned()))
    );
    assert_eq!(
        kind("block0:\n    %0 = FPAdd32 f32 1, f32 2 {rz\n"),
        Some(ParseErrorKind::Expected("} after the flags"))
    );
}

#[test]
fn every_texture_word_reads() {
    let f = ir(
        "block0:\n    %0: u32x4 = ImageRead u32 0, u32 0 {descriptor=9, type=cube, depth, \
                bias, lod_clamp, relaxed_precision, gather=1, derivatives=3, format=r16_uint, ndv, \
                integer}\n",
    );
    assert_eq!(
        first_op(&f).flags,
        Flags::Texture(TextureInfo {
            descriptor_index: 9,
            ty: TextureType::ColorCube,
            is_depth: true,
            has_bias: true,
            has_lod_clamp: true,
            relaxed_precision: true,
            gather_component: 1,
            num_derivatives: 3,
            image_format: ImageFormat::R16Uint,
            ndv_is_active: true,
            is_integer: true,
        })
    );
}

#[test]
fn phis_read_their_incoming_pairs() {
    let f = ir("block0:\n    %0: f32 = Phi [block2: %7], [block0: f32 1.0]\n");
    assert_eq!(
        f.inst(Value(0)),
        Some(&Inst::Phi(Phi {
            ty: Type::F32,
            incoming: vec![
                (BlockId(2), Operand::Value(Value(7))),
                (BlockId(0), Operand::Imm(Imm::F32(1.0_f32.to_bits()))),
            ],
        }))
    );
    let expected = Some(ParseErrorKind::Expected("[block: operand]"));
    assert_eq!(kind("block0:\n    %0: u32 = Phi block1: %7\n"), expected);
    assert_eq!(kind("block0:\n    %0: u32 = Phi [block1 %7]\n"), expected);
}

#[test]
fn terminators() {
    let f = ir(
        "block0:\n    branch %4, block2, block1\nblock1:\n    jump block0\nblock2:\n    \
                unreachable\nblock3:\n    return\n",
    );
    let terminators: Vec<Terminator> = f.blocks().map(|(_, block)| block.terminator).collect();
    assert_eq!(
        terminators,
        [
            Terminator::Branch {
                cond: Operand::Value(Value(4)),
                then: BlockId(2),
                otherwise: BlockId(1)
            },
            Terminator::Jump(BlockId(0)),
            Terminator::Unreachable,
            Terminator::Return,
        ]
    );
    let branch = Some(ParseErrorKind::Expected(
        "branch <condition>, <block>, <block>",
    ));
    assert_eq!(kind("block0:\n    branch %4, block2\n"), branch);
    assert_eq!(
        kind("block0:\n    branch %4, block2, block1, block0\n"),
        branch
    );
    assert_eq!(
        kind("block0:\n    jump 2\n"),
        Some(ParseErrorKind::Expected("a block such as block1"))
    );
    assert_eq!(
        kind("block0:\n    goto block1\n"),
        Some(ParseErrorKind::Expected("an instruction or a terminator"))
    );
    assert_eq!(
        kind("block0:\n    return\n    %0 = Barrier\n"),
        Some(ParseErrorKind::AfterTerminator)
    );
    assert_eq!(
        kind("block0:\n    return\n    return\n"),
        Some(ParseErrorKind::AfterTerminator)
    );
}

#[test]
fn block_headers_and_ids() {
    assert_eq!(
        kind("block1:\n"),
        Some(ParseErrorKind::BlockOrder {
            found: 1,
            expected: 0
        })
    );
    assert_eq!(
        kind("block0:\nblock0:\n"),
        Some(ParseErrorKind::BlockOrder {
            found: 0,
            expected: 1
        })
    );
    assert_eq!(
        kind("block0x1:\n"),
        Some(ParseErrorKind::Number("block0x1".to_owned()))
    );
    assert_eq!(
        kind("entry:\n"),
        Some(ParseErrorKind::Expected("a block such as block1"))
    );
    assert_eq!(kind("%0 = Barrier\n"), Some(ParseErrorKind::OutsideBlock));
    assert_eq!(
        kind("block0:\n    %0 = Barrier\n    %0 = Barrier\n"),
        Some(ParseErrorKind::Redefined(Value(0)))
    );
    assert_eq!(
        kind("block0:\n    %0 = Barrier\nblock1:\n    %0 = Barrier\n"),
        Some(ParseErrorKind::Redefined(Value(0)))
    );
    assert_eq!(
        kind("block0:\n    0 = Barrier\n"),
        Some(ParseErrorKind::Expected("an instruction or a terminator"))
    );
    assert_eq!(
        kind("block0:\n    %0 Barrier\n"),
        Some(ParseErrorKind::Expected("= after the id"))
    );
    assert_eq!(
        kind("block0:\n    %0: q32 = Barrier\n"),
        Some(ParseErrorKind::UnknownType("q32".to_owned()))
    );
    assert_eq!(
        kind("block0:\n    %0 = FooBar\n"),
        Some(ParseErrorKind::UnknownOpcode("FooBar".to_owned()))
    );
}

#[test]
fn structure_lines() {
    let f = ir(
        "block0:\nstructure:\n    loop {\n        if pt {\n            break u1 true\n        }\n        unreachable\n    } repeat %3\n    return\n",
    );
    assert_eq!(
        f.structure(),
        Some(
            &[
                Node::Loop {
                    body: vec![
                        Node::If {
                            cond: Operand::Imm(Imm::Pred(Pred::PT)),
                            body: vec![Node::Break {
                                cond: Operand::Imm(Imm::U1(true))
                            }],
                        },
                        Node::Unreachable,
                    ],
                    repeat: Operand::Value(Value(3)),
                },
                Node::Return,
            ][..]
        )
    );
    assert_eq!(ir("block0:\nstructure:\n").structure(), Some(&[][..]));
    assert_eq!(
        kind("block0:\nstructure:\n    }\n"),
        Some(ParseErrorKind::Unbalanced)
    );
    assert_eq!(
        kind("block0:\nstructure:\n    if pt {\n    } repeat pt\n"),
        Some(ParseErrorKind::Unbalanced)
    );
    assert_eq!(
        kind("block0:\nstructure:\n    loop {\n    }\n"),
        Some(ParseErrorKind::Unbalanced)
    );
    assert_eq!(
        kind("block0:\nstructure:\n    if pt {\n    } else\n"),
        Some(ParseErrorKind::Unbalanced)
    );
    assert_eq!(
        kind("block0:\nstructure:\n    loop {\n"),
        Some(ParseErrorKind::Unclosed)
    );
    assert_eq!(
        kind("block0:\nstructure:\n    while pt {\n"),
        Some(ParseErrorKind::Expected("if or loop"))
    );
    assert_eq!(
        kind("block0:\nstructure:\n    block0:\n"),
        Some(ParseErrorKind::Number("0:".to_owned()))
    );
    assert_eq!(
        kind("block0:\nstructure:\n    break\n"),
        Some(ParseErrorKind::Expected("a block such as block1"))
    );
}

/// The parser nests up to [`MAX_NESTING`] levels and no deeper.
#[test]
fn nesting_is_bounded() {
    let text = |depth: usize| {
        let mut text = String::from("block0:\nstructure:\n");
        for _ in 0..depth {
            text.push_str("if u1 true {\n");
        }
        text.push_str("block0\n");
        for _ in 0..depth {
            text.push_str("}\n");
        }
        text
    };
    let deepest = ir(&text(MAX_NESTING));
    assert_eq!(deepest.verify(), Ok(()));
    let error = Function::parse(&text(MAX_NESTING + 1)).err();
    assert_eq!(
        error,
        Some(ParseError {
            line: 2 + MAX_NESTING + 1,
            kind: ParseErrorKind::TooDeep
        })
    );
}

#[test]
fn errors_name_their_line() {
    assert_eq!(
        refused("block0:\n\n    %0 = FooBar\n"),
        "line 3: unknown opcode FooBar"
    );
    // An unclosed tree is reported at the last line.
    assert_eq!(
        refused("block0:\nstructure:\n    loop {\n\n"),
        "line 4: the structure ends inside an if or a loop"
    );
    let words = [
        (ParseErrorKind::Expected("x"), "expected x"),
        (
            ParseErrorKind::UnknownOpcode("Foo".into()),
            "unknown opcode Foo",
        ),
        (ParseErrorKind::UnknownType("q8".into()), "unknown type q8"),
        (
            ParseErrorKind::UnknownOperand("r999".into()),
            "unknown operand r999",
        ),
        (ParseErrorKind::Number("0xg".into()), "bad number 0xg"),
        (
            ParseErrorKind::Flag("rz".into()),
            "unknown or repeated flag rz",
        ),
        (
            ParseErrorKind::BlockOrder {
                found: 3,
                expected: 1,
            },
            "block3 is out of order: block1 comes next",
        ),
        (ParseErrorKind::Redefined(Value(2)), "%2 is defined twice"),
        (
            ParseErrorKind::IdTooLarge(Value(99)),
            "%99 is larger than the text's length",
        ),
        (
            ParseErrorKind::Annotation {
                opcode: Opcode::LaneId,
                wanted: Type::U32,
                found: Type::F32,
            },
            "LaneId has type u32, not f32",
        ),
        (ParseErrorKind::PhiType, "a phi needs a result type"),
        (
            ParseErrorKind::AfterTerminator,
            "a line after the block's terminator",
        ),
        (
            ParseErrorKind::OutsideBlock,
            "an instruction outside a block",
        ),
        (
            ParseErrorKind::Unbalanced,
            "a closing brace that does not match",
        ),
        (
            ParseErrorKind::Unclosed,
            "the structure ends inside an if or a loop",
        ),
    ];
    for (error, text) in words {
        assert_eq!(error.to_string(), text);
    }
    assert_eq!(
        ParseErrorKind::TooDeep.to_string(),
        format!("the structure nests deeper than {MAX_NESTING} levels")
    );
}
