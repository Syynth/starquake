//! Attribute and system-value translator tests: each instruction and modifier run on the IR
//! interpreter (slice 7) with values derived from the instruction's documented behaviour,
//! golden IR, PR #4588's cases for this group, each rejection with its exact error and order,
//! and generated operations that never panic.

use proptest::prelude::*;

use super::*;
use crate::decode::{
    AluSrc, Instr, Instruction, IsbeAccess, IsbeMemType, OpExit, OpIsberd, OpIsbewr, OpLdTram,
    OpOut, OpOutFinal, OutType, PredSrc, UniformReg, decode,
};
use crate::interp::{Input, InterpError, Machine};
use crate::ir::Function;
use crate::sph::{Components, PixelIo};
use crate::testing::{
    any_gpr, any_pred, any_pred_src, any_uniform_reg, any_unmodified_src, config,
};
use crate::translate::driver::translate_instrs;
use crate::translate::{
    COLOR_TARGETS, ColorOutputs, GuestStage, REGISTER_LIMIT, Stage, TranslateError,
};

/// `program` through the translation driver, with no header.
fn translate_program(stage: Stage, program: &[Instr]) -> Result<Function, TranslateError> {
    translate_instrs(stage, None, program)
}

/// `program` through the translation driver, with the pixel program's `header`.
fn translate_with_header(
    stage: Stage,
    header: &PixelIo,
    program: &[Instr],
) -> Result<Function, TranslateError> {
    translate_instrs(stage, Some(header), program)
}

fn r(index: u8) -> Gpr {
    Gpr::new(index)
}

fn p(index: u8) -> Pred {
    Pred::new(index).expect("a predicate")
}

fn ur(index: u8) -> UniformReg {
    UniformReg::new(index).expect("a uniform register")
}

fn stage(stage: GuestStage) -> Stage {
    Stage {
        stage,
        color_outputs: ColorOutputs {
            targets: [0; COLOR_TARGETS],
        },
    }
}

fn vertex() -> Stage {
    stage(GuestStage::Vertex)
}

fn pixel() -> Stage {
    stage(GuestStage::Pixel)
}

fn exit() -> Instr {
    Instr::new(Op::Exit(OpExit {
        keep_ref_count: false,
        no_at_exit: false,
        pred: PredSrc::TRUE,
    }))
}

/// Translates `op` followed by `EXIT` in a vertex program, with no header.
fn one(op: Op) -> Result<Function, TranslateError> {
    translate_program(vertex(), &[Instr::new(op), exit()])
}

/// Translates `op` followed by `EXIT` in a pixel program with `header`.
fn with_header(header: &PixelIo, op: Op) -> Result<Function, TranslateError> {
    translate_with_header(pixel(), header, &[Instr::new(op), exit()])
}

/// The rejection of `op` as the program's first instruction.
fn rejected(op: Op, what: impl Into<Unsupported>) -> Result<Function, TranslateError> {
    Err(TranslateError::Unsupported {
        index: 0,
        instr: Instr::new(op),
        what: what.into(),
    })
}

/// Runs `ir` on `machine`.
fn run(ir: &Function, machine: &mut Machine) {
    machine.run(ir).expect("runs");
}

fn f32_bits(value: f32) -> u32 {
    value.to_bits()
}

fn attr(word: u16) -> Attribute {
    Attribute(word)
}

/// The byte address of attribute word `word`.
fn addr(word: u16) -> u16 {
    word * 4
}

fn ald(dst: Gpr, addr: u16, comps: u8) -> OpALd {
    OpALd {
        dst,
        vtx: Gpr::RZ,
        offset: Gpr::RZ,
        addr,
        comps,
        patch: false,
        phys: false,
        output: false,
    }
}

fn ast(data: Gpr, addr: u16, comps: u8) -> OpASt {
    OpASt {
        data,
        vtx: Gpr::RZ,
        offset: Gpr::RZ,
        addr,
        comps,
        patch: false,
        phys: false,
    }
}

fn ipa(dst: Gpr, addr: u16) -> OpIpa {
    OpIpa {
        dst,
        addr,
        freq: InterpFreq::Pass,
        loc: InterpLoc::Default,
        offset: Gpr::RZ,
        pred: Pred::PT,
    }
}

/// A header whose generic attribute `index` has the map `map` (x, y, z, w).
fn header_with(index: usize, map: [PixelImap; 4]) -> PixelIo {
    let mut header = PixelIo::default();
    header.generic[index] = Components {
        x: map[0],
        y: map[1],
        z: map[2],
        w: map[3],
    };
    header
}

// ---------------------------------------------------------------------------------------------
// Constants against the references.

/// Every attribute word and special register number this module names, written out from the
/// references (`attribute.h` and `move_special_register.cpp` @ cf654052; PR #4588's `S2R`).
#[test]
fn constants_match_the_references() {
    assert_eq!(ATTRIBUTE_COMPONENTS, 4);
    assert_eq!(crate::ir::POSITION_X, 28);
    assert_eq!(POSITION_W, 31);
    assert_eq!(GENERIC_0_X, 32);
    assert_eq!(GENERIC_31_X, 156);
    assert_eq!(LEGACY_COLORS, 160..=175);
    assert_eq!(FOG_COORDINATE, 186);
    assert_eq!(LEGACY_TEXTURES, 192..=231);
    assert_eq!(CS2R_PAIR, 2);
    assert_eq!(ONE_F32, 1.0_f32.to_bits());
    let srs = [
        (WSCALEFACTOR_XY, 30),
        (WSCALEFACTOR_Z, 31),
        (EQ_MASK, 56),
        (LT_MASK, 57),
        (LE_MASK, 58),
        (GT_MASK, 59),
        (GE_MASK, 60),
    ];
    for (sr, number) in srs {
        assert_eq!(sr, SpecialReg(number));
    }
}

// ---------------------------------------------------------------------------------------------
// ALD and AST.

/// `ALD` and `AST` pass the vertex register to the IR (PR #4588: `X(vertex_reg)`), read once
/// before the first word, so an `ALD` whose destination is its own vertex register still reads
/// every word of the first vertex.
#[test]
fn ald_and_ast_pass_the_vertex_register() {
    let load = Op::ALd(OpALd {
        vtx: r(9),
        ..ald(r(9), addr(32), 2)
    });
    let ir = one(load).expect("translates");
    assert_eq!(
        ir.to_string(),
        "block0:\n\
         \x20   %0: u32 = GetRegister r9\n\
         \x20   %1: f32 = GetAttribute a0x80, %0\n\
         \x20   %2: u32 = BitCastU32F32 %1\n\
         \x20   %3 = SetRegister r9, %2\n\
         \x20   %4: f32 = GetAttribute a0x84, %0\n\
         \x20   %5: u32 = BitCastU32F32 %4\n\
         \x20   %6 = SetRegister r10, %5\n\
         \x20   return\n"
    );
    let mut machine = Machine::default();
    machine.regs[9] = 2;
    machine.attributes.insert((attr(32), 2), f32_bits(5.0));
    machine.attributes.insert((attr(33), 2), f32_bits(6.0));
    run(&ir, &mut machine);
    assert_eq!(machine.regs[9], f32_bits(5.0));
    assert_eq!(machine.regs[10], f32_bits(6.0));

    let store = Op::ASt(OpASt {
        vtx: r(3),
        ..ast(r(4), addr(28), 2)
    });
    let ir = one(store).expect("translates");
    let mut machine = Machine::default();
    machine.regs[3] = 7;
    machine.regs[4] = f32_bits(-1.0);
    machine.regs[5] = f32_bits(2.5);
    run(&ir, &mut machine);
    assert_eq!(
        machine.outputs.into_iter().collect::<Vec<_>>(),
        vec![
            ((attr(28), 7), f32_bits(-1.0)),
            ((attr(29), 7), f32_bits(2.5))
        ]
    );
}

/// Each word count reads and writes exactly that many words, in order, each to its register.
#[test]
fn ald_and_ast_move_each_word_count() {
    for comps in 1_u8..=4 {
        let base = 40_u16;
        let ir = one(Op::ALd(ald(r(20), addr(base), comps))).expect("translates");
        let mut machine = Machine::default();
        for word in 0..4_u16 {
            machine
                .attributes
                .insert((attr(base + word), 0), f32_bits(f32::from(word) + 10.0));
        }
        run(&ir, &mut machine);
        for word in 0..4_u8 {
            let expected = if word < comps {
                f32_bits(f32::from(word) + 10.0)
            } else {
                0
            };
            assert_eq!(
                machine.regs[usize::from(20 + word)],
                expected,
                "{comps} words"
            );
        }

        let ir = one(Op::ASt(ast(r(20), addr(base), comps))).expect("translates");
        let mut machine = Machine::default();
        for word in 0..4_u8 {
            machine.regs[usize::from(20 + word)] = u32::from(word) + 100;
        }
        run(&ir, &mut machine);
        let written: Vec<_> = machine.outputs.into_iter().collect();
        let expected: Vec<_> = (0..u16::from(comps))
            .map(|word| ((attr(base + word), 0), u32::from(word) + 100))
            .collect();
        assert_eq!(written, expected);
    }
}

/// `AST` from `RZ` stores zero for every word, without reaching past `RZ` (PR #4588: "RZ as the
/// data source stores zero for every component").
#[test]
fn ast_from_rz_stores_zero_for_every_word() {
    let ir = one(Op::ASt(ast(Gpr::RZ, addr(28), 4))).expect("translates");
    // Every register holds non-zero bits, so a store that read one would show it.
    let mut machine = Machine {
        regs: [u32::MAX; crate::interp::REG_SLOTS],
        ..Machine::default()
    };
    run(&ir, &mut machine);
    let written: Vec<_> = machine.outputs.into_iter().collect();
    let expected: Vec<_> = (28..32).map(|word| ((attr(word), 0), 0)).collect();
    assert_eq!(written, expected);
}

// ---------------------------------------------------------------------------------------------
// IPA.

/// The value `IPA` gives on the interpreter for attribute `word` reading `value`, with the
/// position's W `w`, under `header`.
fn ipa_value(header: &PixelIo, op: OpIpa, word: u16, value: f32, w: f32) -> u32 {
    let ir = with_header(header, Op::Ipa(op)).expect("translates");
    assert_eq!(ir.verify(), Ok(()));
    let mut machine = Machine::default();
    machine.attributes.insert((attr(word), 0), f32_bits(value));
    machine
        .attributes
        .insert((attr(POSITION_W), 0), f32_bits(w));
    run(&ir, &mut machine);
    machine.regs[usize::from(op.dst.index())]
}

/// A generic attribute is multiplied by the position's W when the first component of its input
/// map that is not unused is perspective, or when every component is unused; constant and
/// screen-linear inputs are not (PR #4588's `IPA`). The first component decides, not any.
#[test]
fn ipa_interpolates_generics_by_the_header() {
    use PixelImap::{Constant, Perspective, ScreenLinear, Unused};
    let word = GENERIC_0_X + 4 * 5 + 2; // generic 5's z
    let cases = [
        ([Perspective, Unused, Unused, Unused], true),
        ([Unused; 4], true),
        ([Constant, Perspective, Perspective, Perspective], false),
        ([ScreenLinear, Perspective, Perspective, Perspective], false),
        ([Unused, Constant, Perspective, Perspective], false),
        ([Unused, Unused, Perspective, Constant], true),
        ([Unused, Unused, Unused, ScreenLinear], false),
    ];
    for (map, perspective) in cases {
        let header = header_with(5, map);
        let got = ipa_value(&header, ipa(r(1), addr(word)), word, 3.0, 0.5);
        let expected = if perspective { 1.5 } else { 3.0 };
        assert_eq!(got, f32_bits(expected), "{map:?}");
    }
    // Another generic's map does not count.
    let header = header_with(4, [Constant; 4]);
    assert_eq!(
        ipa_value(&header, ipa(r(1), addr(word)), word, 3.0, 0.5),
        f32_bits(1.5)
    );
}

/// The perspective input's golden IR: the attribute at vertex 0, the position's W, `FPMul32`
/// with default controls (Eden's `FPMul` with an empty `FpControl`).
#[test]
fn ipa_golden_ir() {
    let header = header_with(0, [PixelImap::Perspective; 4]);
    let ir = with_header(&header, Op::Ipa(ipa(r(2), addr(GENERIC_0_X)))).expect("translates");
    assert_eq!(
        ir.to_string(),
        "block0:\n\
         \x20   %0: f32 = GetAttribute a0x80, u32 0x0\n\
         \x20   %1: f32 = GetAttribute a0x7c, u32 0x0\n\
         \x20   %2: f32 = FPMul32 %0, %1\n\
         \x20   %3: u32 = BitCastU32F32 %2\n\
         \x20   %4 = SetRegister r2, %3\n\
         \x20   return\n"
    );
    let header = header_with(0, [PixelImap::Constant; 4]);
    let ir = with_header(&header, Op::Ipa(ipa(r(2), addr(GENERIC_0_X)))).expect("translates");
    assert_eq!(
        ir.to_string(),
        "block0:\n\
         \x20   %0: f32 = GetAttribute a0x80, u32 0x0\n\
         \x20   %1: u32 = BitCastU32F32 %0\n\
         \x20   %2 = SetRegister r2, %1\n\
         \x20   return\n"
    );
}

/// Which attributes `IPA` multiplies by W, at each edge of Eden's ranges: `IsGeneric` is
/// `Generic0X` (32) through `Generic31X` (156), so generic 31's Y, Z and W (157 to 159) are not
/// multiplied and need no header; `IsLegacyAttribute` is 160 to 175, 186 and 192 to 231.
#[test]
fn ipa_follows_edens_generic_range() {
    let header = PixelIo::default(); // every generic unused: perspective
    // (word, multiplied, needs the header)
    let cases: [(u16, bool, bool); 18] = [
        (0, false, false),
        (28, false, false),
        (30, false, false),
        (32, true, true),
        (156, true, true),
        (157, false, false),
        (159, false, false),
        (160, true, false),
        (175, true, false),
        (176, false, false),
        (185, false, false),
        (186, true, false),
        (187, false, false),
        (191, false, false),
        (192, true, false),
        (231, true, false),
        (232, false, false),
        (255, false, false),
    ];
    for (word, multiplied, needs_header) in cases {
        let got = ipa_value(&header, ipa(r(1), addr(word)), word, 3.0, 0.5);
        let expected = if multiplied { 1.5 } else { 3.0 };
        assert_eq!(got, f32_bits(expected), "word {word}");
        let op = Op::Ipa(ipa(r(1), addr(word)));
        let without = translate_program(pixel(), &[Instr::new(op), exit()]);
        if needs_header {
            assert_eq!(
                without,
                rejected(op, AttributeRejection::IpaNoHeader(attr(word))),
                "word {word}"
            );
        } else {
            assert!(without.is_ok(), "word {word}: {without:?}");
        }
    }
    // The position's W itself (31), just below the generics, reads once, needs no header and is
    // not multiplied (here the attribute and W are the same word, so it is checked by its IR).
    let ir = one(Op::Ipa(ipa(r(1), addr(POSITION_W)))).expect("translates");
    assert_eq!(
        ir.to_string(),
        "block0:\n\
         \x20   %0: f32 = GetAttribute a0x7c, u32 0x0\n\
         \x20   %1: u32 = BitCastU32F32 %0\n\
         \x20   %2 = SetRegister r1, %1\n\
         \x20   return\n"
    );
}

/// The frequency (`.PASS`, `.CONSTANT`) and the centroid location change nothing: the IR is the
/// default's (PR #4588: "the flat state comes from the program header", "Default and centroid
/// locations read the same interpolant").
#[test]
fn ipa_frequency_and_centroid_change_nothing() {
    let header = header_with(1, [PixelImap::Perspective; 4]);
    let base = ipa(r(3), addr(GENERIC_0_X + 4));
    let expected = with_header(&header, Op::Ipa(base)).expect("translates");
    for freq in [InterpFreq::Pass, InterpFreq::Constant] {
        for loc in [InterpLoc::Default, InterpLoc::Centroid] {
            let op = Op::Ipa(OpIpa { freq, loc, ..base });
            assert_eq!(
                with_header(&header, op).expect("translates"),
                expected,
                "{freq:?} {loc:?}"
            );
        }
    }
}

/// `IPA` reads every attribute address, from the first word to the last (`0x3fc`, the most the
/// eight-bit word field holds).
#[test]
fn ipa_addresses_at_both_edges() {
    for word in [0_u16, 255] {
        let ir = one(Op::Ipa(ipa(r(1), addr(word)))).expect("translates");
        let mut machine = Machine::default();
        machine.attributes.insert((attr(word), 0), f32_bits(9.0));
        run(&ir, &mut machine);
        assert_eq!(machine.regs[1], f32_bits(9.0), "word {word}");
    }
}

/// `IPA.OFFSET` reads `InterpolateAtOffset` at the offset register's bits, for the position's W
/// too (a divergence from PR #4588, which rejects the location; see the module docs).
#[test]
fn ipa_offset_is_carried_in_the_ir() {
    let header = header_with(2, [PixelImap::Perspective; 4]);
    let word = GENERIC_0_X + 8;
    let op = OpIpa {
        loc: InterpLoc::Offset,
        offset: r(6),
        ..ipa(r(1), addr(word))
    };
    let ir = with_header(&header, Op::Ipa(op)).expect("translates");
    assert_eq!(
        ir.to_string(),
        "block0:\n\
         \x20   %0: u32 = GetRegister r6\n\
         \x20   %1: f32 = InterpolateAtOffset a0xa0, %0\n\
         \x20   %2: f32 = InterpolateAtOffset a0x7c, %0\n\
         \x20   %3: f32 = FPMul32 %1, %2\n\
         \x20   %4: u32 = BitCastU32F32 %3\n\
         \x20   %5 = SetRegister r1, %4\n\
         \x20   return\n"
    );
    let offset = 0x0100_ff00; // x = -0x100, y = 0x100 in NAK's packing (s12 halves)
    let mut machine = Machine::default();
    machine.regs[6] = offset;
    machine
        .interpolated
        .insert((attr(word), offset), f32_bits(8.0));
    machine
        .interpolated
        .insert((attr(POSITION_W), offset), f32_bits(0.25));
    // The centre's values are not read.
    machine.attributes.insert((attr(word), 0), f32_bits(100.0));
    run(&ir, &mut machine);
    assert_eq!(machine.regs[1], f32_bits(2.0));

    // An offset of `RZ` interpolates at offset 0; a flat input reads once.
    let flat = header_with(2, [PixelImap::Constant; 4]);
    let op = OpIpa {
        offset: Gpr::RZ,
        ..op
    };
    let ir = with_header(&flat, Op::Ipa(op)).expect("translates");
    assert_eq!(
        ir.to_string(),
        "block0:\n\
         \x20   %0: f32 = InterpolateAtOffset a0xa0, u32 0x0\n\
         \x20   %1: u32 = BitCastU32F32 %0\n\
         \x20   %2 = SetRegister r1, %1\n\
         \x20   return\n"
    );
    // A value the test does not give is a missing input, naming the attribute and the offset.
    let mut machine = Machine::default();
    let error = machine
        .run(&ir)
        .map(|_| ())
        .expect_err("the value is not given");
    assert!(
        matches!(
            error,
            InterpError::MissingInput {
                input: Input::Interpolated { attribute, offset: 0 },
                ..
            } if attribute == attr(word)
        ),
        "{error:?}"
    );
    assert_eq!(
        error.to_string(),
        "%0: attribute a0xa0 at offset 0x0 is not given"
    );
}

/// Each `IPA` check, alone and with the next one also failing: the offset register outside
/// `.OFFSET`, the frequency, the predicate output, the header; in that order.
#[test]
fn ipa_checks_report_in_order() {
    let word = GENERIC_0_X;
    let base = ipa(r(1), addr(word));
    let offset_reg = OpIpa {
        offset: r(4),
        ..base
    };
    let centroid_offset_reg = OpIpa {
        loc: InterpLoc::Centroid,
        ..offset_reg
    };
    let state = OpIpa {
        freq: InterpFreq::State,
        ..base
    };
    let predicate = OpIpa { pred: p(1), ..base };
    let cases = [
        (offset_reg, AttributeRejection::IpaOffsetRegister(r(4))),
        (
            centroid_offset_reg,
            AttributeRejection::IpaOffsetRegister(r(4)),
        ),
        (state, AttributeRejection::IpaFrequency(InterpFreq::State)),
        (predicate, AttributeRejection::IpaPredicate(p(1))),
        (base, AttributeRejection::IpaNoHeader(attr(word))),
        // Two faults: the first in the order is reported.
        (
            OpIpa {
                freq: InterpFreq::State,
                ..offset_reg
            },
            AttributeRejection::IpaOffsetRegister(r(4)),
        ),
        (
            OpIpa {
                pred: p(1),
                ..state
            },
            AttributeRejection::IpaFrequency(InterpFreq::State),
        ),
        (predicate, AttributeRejection::IpaPredicate(p(1))),
        // `.OFFSET` takes its register, so the frequency is the fault.
        (
            OpIpa {
                loc: InterpLoc::Offset,
                freq: InterpFreq::State,
                ..offset_reg
            },
            AttributeRejection::IpaFrequency(InterpFreq::State),
        ),
    ];
    for (op, what) in cases {
        assert_eq!(one(Op::Ipa(op)), rejected(Op::Ipa(op), what), "{op}");
    }
    // The predicate is checked before the header: a generic attribute without a header and a
    // predicate output reports the predicate.
    let both = OpIpa { pred: p(2), ..base };
    assert_eq!(
        one(Op::Ipa(both)),
        rejected(Op::Ipa(both), AttributeRejection::IpaPredicate(p(2)))
    );
    // With every fault cleared, it translates.
    assert!(with_header(&PixelIo::default(), Op::Ipa(base)).is_ok());
}

/// `IPA` checks its attribute address as `ALD` and `AST` do (starquake's: the PR reads an
/// 8-bit word number, the decoder a byte address that cannot fail it, but `OpIpa`'s fields are
/// public): the last word, `0x3fc`, is accepted; an address that is not word aligned
/// (`0x3fe`, and `0x3fd` and `0x3ff` beside it) or that starts at or past the end of the
/// attribute space is [`Unsupported::AttributeAddress`] with one word.
#[test]
fn ipa_addresses_outside_the_attribute_space_are_rejected() {
    let end = u16::try_from(ATTRIBUTE_SPACE_END).expect("fits");
    let last = end - 4;
    let ir = one(Op::Ipa(ipa(r(1), last))).expect("translates");
    let mut machine = Machine::default();
    machine
        .attributes
        .insert((attr(last / 4), 0), f32_bits(9.0));
    run(&ir, &mut machine);
    assert_eq!(machine.regs[1], f32_bits(9.0));
    for bad in [last + 1, last + 2, last + 3, end, end + 4, u16::MAX] {
        let op = Op::Ipa(ipa(r(1), bad));
        assert_eq!(
            one(op),
            rejected(
                op,
                Unsupported::AttributeAddress {
                    addr: bad,
                    comps: 1
                }
            ),
            "{bad:#x}"
        );
    }
    let op = Op::Ipa(ipa(r(1), 0x3fe));
    assert_eq!(
        one(op).map_err(|error| error.to_string()),
        Err(
            "instruction 0 (ipa.pass r1, a[0x3fe];): attribute address 0x3fe for a 1-word access is \
             not supported"
                .into()
        )
    );
    // The PR's checks come first: a frequency fault with a bad address reports the frequency.
    let both = OpIpa {
        freq: InterpFreq::State,
        ..ipa(r(1), 0x3fe)
    };
    assert_eq!(
        one(Op::Ipa(both)),
        rejected(
            Op::Ipa(both),
            AttributeRejection::IpaFrequency(InterpFreq::State)
        )
    );
    // And the address before the header: an unaligned address inside generic 0, with no
    // header, reports the address, where the aligned one reports the header.
    let generic = addr(GENERIC_0_X);
    let unaligned = Op::Ipa(ipa(r(1), generic + 2));
    assert_eq!(
        one(unaligned),
        rejected(
            unaligned,
            Unsupported::AttributeAddress {
                addr: generic + 2,
                comps: 1
            }
        )
    );
    let aligned = Op::Ipa(ipa(r(1), generic));
    assert_eq!(
        one(aligned),
        rejected(aligned, AttributeRejection::IpaNoHeader(attr(GENERIC_0_X)))
    );
}

/// Without the header, `IPA` of a generic attribute is rejected (starquake's: the PR always has
/// the header); with it, it translates.
#[test]
fn ipa_of_a_generic_needs_the_header() {
    let op = Op::Ipa(ipa(r(1), addr(GENERIC_31_X)));
    assert_eq!(
        one(op),
        rejected(op, AttributeRejection::IpaNoHeader(attr(GENERIC_31_X)))
    );
    assert_eq!(
        one(op).map_err(|error| error.to_string()),
        Err(
            "instruction 0 (ipa.pass r1, a[0x270];): an ipa of generic attribute a0x270 without \
             the program's header is not supported"
                .into()
        )
    );
    assert!(with_header(&PixelIo::default(), op).is_ok());
}

// ---------------------------------------------------------------------------------------------
// S2R, S2UR, CS2R, PIXLD and AL2P.

/// The system values the machine holds for the special register tests, each distinct.
fn system_machine(lane: u32) -> Machine {
    let mut machine = Machine::default();
    machine.system.lane_id = lane;
    machine.system.local_invocation_id = [11, 12, 13];
    machine.system.workgroup_id = [21, 22, 23];
    machine
}

/// What PR #4588's `S2R` reads for each special register it handles, at lane 5 (written out:
/// the register numbers from its `switch`, the values from the machine's system values).
fn expected_special() -> Vec<(u8, u32)> {
    vec![
        (0, 5),
        (33, 11),
        (34, 12),
        (35, 13),
        (37, 21),
        (38, 22),
        (39, 23),
        (30, 0x3f80_0000),
        (31, 0x3f80_0000),
        (56, 0x20),
        (57, 0x1f),
        (58, 0x3f),
        (59, 0xffff_ffc0),
        (60, 0xffff_ffe0),
    ]
}

/// Each special register the PR reads, into a register (`S2R`) and a uniform register
/// (`S2UR`); every other number is rejected with its name.
#[test]
fn s2r_and_s2ur_read_each_special_register() {
    let handled = expected_special();
    for &(number, value) in &handled {
        let sr = SpecialReg(number);
        let s2r = Op::S2R(OpS2R {
            dst: S2RDst::Reg(r(7)),
            sr,
        });
        let ir = one(s2r).expect("translates");
        assert_eq!(ir.verify(), Ok(()));
        let mut machine = system_machine(5);
        run(&ir, &mut machine);
        assert_eq!(machine.regs[7], value, "{sr}");

        let uniform = Op::S2R(OpS2R {
            dst: S2RDst::Uniform(ur(9)),
            sr,
        });
        let ir = one(uniform).expect("translates");
        let mut machine = system_machine(5);
        run(&ir, &mut machine);
        assert_eq!(machine.uniform_regs[9], value, "{sr} to ur9");
        assert_eq!(machine.regs, [0; crate::interp::REG_SLOTS]);
    }
    for number in 0..=u8::MAX {
        if handled.iter().any(|&(handled, _)| handled == number) {
            continue;
        }
        let sr = SpecialReg(number);
        for dst in [
            S2RDst::Reg(r(7)),
            S2RDst::Reg(Gpr::RZ),
            S2RDst::Uniform(ur(9)),
            S2RDst::Uniform(UniformReg::URZ),
        ] {
            let op = Op::S2R(OpS2R { dst, sr });
            assert_eq!(one(op), rejected(op, Unsupported::SpecialRegister(sr)));
        }
    }
}

/// The thread id reads its component of `LocalInvocationId` (Eden's emitter:
/// `CompositeExtract(Inst(Opcode::LocalInvocationId), 0)`), and the block id of `WorkgroupId`.
#[test]
fn s2r_golden_ir() {
    let ir = one(Op::S2R(OpS2R {
        dst: S2RDst::Reg(r(3)),
        sr: SpecialReg::TID_Z,
    }))
    .expect("translates");
    assert_eq!(
        ir.to_string(),
        "block0:\n\
         \x20   %0: u32x3 = LocalInvocationId\n\
         \x20   %1: u32 = CompositeExtractU32x3 %0, u32 0x2\n\
         \x20   %2 = SetRegister r3, %1\n\
         \x20   return\n"
    );
    let ir = one(Op::S2R(OpS2R {
        dst: S2RDst::Uniform(ur(4)),
        sr: SpecialReg::CTAID_Y,
    }))
    .expect("translates");
    assert_eq!(
        ir.to_string(),
        "block0:\n\
         \x20   %0: u32x3 = WorkgroupId\n\
         \x20   %1: u32 = CompositeExtractU32x3 %0, u32 0x1\n\
         \x20   %2 = SetUniformRegister ur4, %1\n\
         \x20   return\n"
    );
    let ir = one(Op::S2R(OpS2R {
        dst: S2RDst::Reg(r(3)),
        sr: WSCALEFACTOR_Z,
    }))
    .expect("translates");
    assert_eq!(
        ir.to_string(),
        "block0:\n    %0 = SetRegister r3, u32 0x3f800000\n    return\n"
    );
}

/// `S2R` to `RZ` and `S2UR` to `URZ` read nothing.
#[test]
fn s2r_to_the_zero_registers_reads_nothing() {
    for (number, _) in expected_special() {
        for dst in [S2RDst::Reg(Gpr::RZ), S2RDst::Uniform(UniformReg::URZ)] {
            let ir = one(Op::S2R(OpS2R {
                dst,
                sr: SpecialReg(number),
            }))
            .expect("translates");
            assert_eq!(ir.to_string(), "block0:\n    return\n");
        }
    }
}

/// The lane masks at the word's edges: lane 0, lane 31 and lanes past the word (the
/// interpreter's rule for a lane id of 32 or more).
#[test]
fn lane_masks_at_the_word_edges() {
    // (lane, eq, lt, le, gt, ge)
    let cases = [
        (0, 0x1, 0x0, 0x1, 0xffff_fffe, 0xffff_ffff),
        (1, 0x2, 0x1, 0x3, 0xffff_fffc, 0xffff_fffe),
        (31, 0x8000_0000, 0x7fff_ffff, 0xffff_ffff, 0x0, 0x8000_0000),
        (32, 0x0, 0xffff_ffff, 0xffff_ffff, 0x0, 0x0),
        (u32::MAX, 0x0, 0xffff_ffff, 0xffff_ffff, 0x0, 0x0),
    ];
    for (lane, eq, lt, le, gt, ge) in cases {
        for (sr, expected) in [
            (EQ_MASK, eq),
            (LT_MASK, lt),
            (LE_MASK, le),
            (GT_MASK, gt),
            (GE_MASK, ge),
        ] {
            let ir = one(Op::S2R(OpS2R {
                dst: S2RDst::Reg(r(1)),
                sr,
            }))
            .expect("translates");
            let mut machine = system_machine(lane);
            run(&ir, &mut machine);
            assert_eq!(machine.regs[1], expected, "{sr} at lane {lane}");
        }
    }
}

/// `CS2R` reads `SRZ` as zero into one register or, with `.64`, two; the pair must end before
/// `RZ` (`dest + 1 < 255`), at the edge on both sides; any other register is rejected first.
#[test]
fn cs2r_reads_zero_into_one_or_two_registers() {
    let cs2r = |dst, sr, wide| Op::CS2R(OpCS2R { dst, sr, wide });
    for (wide, written) in [(false, 1), (true, 2)] {
        let ir = one(cs2r(r(3), SpecialReg::SRZ, wide)).expect("translates");
        let mut machine = Machine::default();
        machine.regs[3] = 7;
        machine.regs[4] = 8;
        machine.regs[5] = 9;
        run(&ir, &mut machine);
        let expected = if written == 2 { [0, 0, 9] } else { [0, 8, 9] };
        assert_eq!(machine.regs[3..6], expected, "wide {wide}");
    }
    let last_pair = u8::try_from(REGISTER_LIMIT - CS2R_PAIR).expect("fits");
    let ir = one(cs2r(r(last_pair), SpecialReg::SRZ, true)).expect("translates");
    let mut machine = Machine::default();
    machine.regs[usize::from(last_pair)] = 1;
    machine.regs[usize::from(last_pair) + 1] = 1;
    run(&ir, &mut machine);
    assert_eq!(
        machine.regs[usize::from(last_pair)..=usize::from(last_pair) + 1],
        [0, 0]
    );
    for dst in [r(last_pair + 1), Gpr::RZ] {
        let op = cs2r(dst, SpecialReg::SRZ, true);
        assert_eq!(
            one(op),
            rejected(op, AttributeRejection::Cs2rDestination(dst))
        );
        // One register is fine there.
        assert!(one(cs2r(dst, SpecialReg::SRZ, false)).is_ok());
    }
    for sr in [SpecialReg::CLOCK_LO, SpecialReg::LANE_ID, SpecialReg(254)] {
        let op = cs2r(r(3), sr, false);
        assert_eq!(one(op), rejected(op, AttributeRejection::Cs2rRegister(sr)));
    }
    // Both faults: the register is reported first.
    let op = cs2r(Gpr::RZ, SpecialReg::CLOCK_LO, true);
    assert_eq!(
        one(op),
        rejected(op, AttributeRejection::Cs2rRegister(SpecialReg::CLOCK_LO))
    );
}

/// `PIXLD.MY_INDEX` reads the sample's index (Eden's Maxwell `PIXLD`: `ir.SampleId()`); a
/// predicate output and every other value are rejected, the predicate first.
#[test]
fn pixld_my_index_reads_the_sample_id() {
    let pixld = |dst, val, pred| Op::PixLd(OpPixLd { dst, val, pred });
    let ir = one(pixld(r(2), PixVal::MyIndex, Pred::PT)).expect("translates");
    assert_eq!(
        ir.to_string(),
        "block0:\n    %0: u32 = SampleId\n    %1 = SetRegister r2, %0\n    return\n"
    );
    let mut machine = Machine::default();
    machine.system.sample_id = 3;
    run(&ir, &mut machine);
    assert_eq!(machine.regs[2], 3);
    let ir = one(pixld(Gpr::RZ, PixVal::MyIndex, Pred::PT)).expect("translates");
    assert_eq!(ir.to_string(), "block0:\n    return\n");
    for val in PixVal::ALL {
        if val == PixVal::MyIndex {
            continue;
        }
        let op = pixld(r(2), val, Pred::PT);
        assert_eq!(one(op), rejected(op, AttributeRejection::PixLdValue(val)));
    }
    let op = pixld(r(2), PixVal::MyIndex, p(4));
    assert_eq!(
        one(op),
        rejected(op, AttributeRejection::PixLdPredicate(p(4)))
    );
    let op = pixld(r(2), PixVal::CovMask, p(4));
    assert_eq!(
        one(op),
        rejected(op, AttributeRejection::PixLdPredicate(p(4)))
    );
}

/// `AL2P` adds the offset register to the address (Eden's Maxwell `AL2P`); the output bit
/// changes nothing; more than one word is rejected.
#[test]
fn al2p_adds_the_offset_register() {
    let al2p = |dst, offset, addr, comps, output| {
        Op::AL2P(OpAL2P {
            dst,
            offset,
            addr,
            comps,
            output,
        })
    };
    let ir = one(al2p(r(1), r(2), 0x84, 1, false)).expect("translates");
    assert_eq!(
        ir.to_string(),
        "block0:\n\
         \x20   %0: u32 = GetRegister r2\n\
         \x20   %1: u32 = IAdd32 %0, u32 0x84\n\
         \x20   %2 = SetRegister r1, %1\n\
         \x20   return\n"
    );
    let mut machine = Machine::default();
    machine.regs[2] = 0x10;
    run(&ir, &mut machine);
    assert_eq!(machine.regs[1], 0x94);
    assert_eq!(one(al2p(r(1), r(2), 0x84, 1, true)), Ok(ir));
    // The largest address the 10-bit SM70 field holds, read unsigned (NAK); Eden's Maxwell
    // field is signed, a divergence listed in the module docs.
    let ir = one(al2p(r(1), Gpr::RZ, 0x3ff, 1, false)).expect("translates");
    let mut machine = Machine::default();
    run(&ir, &mut machine);
    assert_eq!(machine.regs[1], 0x3ff);
    for comps in 2..=4 {
        let op = al2p(r(1), r(2), 0x84, comps, false);
        assert_eq!(one(op), rejected(op, AttributeRejection::Al2pWidth(comps)));
    }
}

// ---------------------------------------------------------------------------------------------
// Deferred and unknown instructions.

/// `OUT` and `OUT.FINAL` (decision 13) and `ISBERD` and `ISBEWR` (decision 11) are rejected by
/// name, `LDTRAM` by its mnemonic, each with its exact text.
#[test]
fn deferred_instructions_are_rejected_by_name() {
    let mut geometry = vec![Op::OutFinal(OpOutFinal {
        dst: Gpr::RZ,
        handle: r(1),
        stream: AluSrc::reg(Gpr::RZ),
    })];
    for out_type in OutType::ALL {
        geometry.push(Op::Out(OpOut {
            dst: r(0),
            handle: r(1),
            stream: AluSrc::Imm(0),
            out_type,
        }));
    }
    for op in geometry {
        assert_eq!(
            one(op),
            rejected(
                op,
                AttributeRejection::Deferred {
                    mnemonic: op.mnemonic(),
                    deferral: Deferral::Geometry
                }
            )
        );
    }
    let isberd = Op::Isberd(OpIsberd {
        dst: r(1),
        offset: r(2),
        imm_offset: 0x42,
        mem_type: IsbeMemType::B32,
        access: IsbeAccess::Attribute,
        output: false,
        skew: false,
    });
    // `ISBEWR` in the one form PR #4588 translates (`.o.attr.32`) is rejected too.
    let isbewr = Op::Isbewr(OpIsbewr {
        offset: r(2),
        data: r(1),
        imm_offset: 0,
        mem_type: IsbeMemType::B32,
        access: IsbeAccess::Attribute,
        output: true,
        skew: false,
    });
    for op in [isberd, isbewr] {
        assert_eq!(
            one(op),
            rejected(
                op,
                AttributeRejection::Deferred {
                    mnemonic: op.mnemonic(),
                    deferral: Deferral::MeshAndTask
                }
            )
        );
    }
    assert_eq!(
        one(isbewr).map_err(|error| error.to_string()),
        Err(
            "instruction 0 (isbewr.o.attr.32 [r2], r1;): isbewr (mesh and task programs come \
             after milestone 3) is not supported"
                .into()
        )
    );
    let out = Op::Out(OpOut {
        dst: r(0),
        handle: r(1),
        stream: AluSrc::Imm(0),
        out_type: OutType::Emit,
    });
    assert_eq!(
        one(out).map_err(|error| error.to_string()),
        Err(
            "instruction 0 (out.emit r0, r1, 0x0;): out (geometry programs are deferred) is not \
             supported"
                .into()
        )
    );
    let ldtram = Op::LdTram(OpLdTram {
        dst: r(1),
        addr: 0x80,
        use_c: false,
        ureg: UniformReg::URZ,
    });
    assert_eq!(
        one(ldtram),
        rejected(ldtram, Unsupported::Instruction("ldtram"))
    );
}

/// Every rejection of the group reads as plain words.
#[test]
fn rejections_read_as_words() {
    let cases: [(AttributeRejection, &str); 11] = [
        (
            AttributeRejection::Deferred {
                mnemonic: "out.final",
                deferral: Deferral::Geometry,
            },
            "out.final (geometry programs are deferred)",
        ),
        (
            AttributeRejection::Deferred {
                mnemonic: "isberd",
                deferral: Deferral::MeshAndTask,
            },
            "isberd (mesh and task programs come after milestone 3)",
        ),
        (
            AttributeRejection::IpaOffsetRegister(r(4)),
            "an ipa with offset register r4 outside .offset",
        ),
        (
            AttributeRejection::IpaFrequency(InterpFreq::State),
            "an ipa.state",
        ),
        (
            AttributeRejection::IpaPredicate(p(1)),
            "an ipa writing predicate p1",
        ),
        (
            AttributeRejection::IpaNoHeader(attr(GENERIC_0_X)),
            "an ipa of generic attribute a0x80 without the program's header",
        ),
        (
            AttributeRejection::Cs2rRegister(SpecialReg::CLOCK_LO),
            "a cs2r of sr_clocklo",
        ),
        (
            AttributeRejection::Cs2rDestination(r(254)),
            "a cs2r.64 to r254",
        ),
        (AttributeRejection::Al2pWidth(2), "an al2p of 2 words"),
        (
            AttributeRejection::PixLdPredicate(p(0)),
            "a pixld writing predicate p0",
        ),
        (
            AttributeRejection::PixLdValue(PixVal::CovMask),
            "a pixld.covmask",
        ),
    ];
    for (rejection, text) in cases {
        assert_eq!(rejection.to_string(), text);
        assert_eq!(Unsupported::from(rejection).to_string(), text);
    }
}

// ---------------------------------------------------------------------------------------------
// PR #4588's cases for this group (`eden-pr4588/src/tests/video_core/native_shader.cpp`).

/// A PR #4588 instruction: its four 32-bit words, decoded.
fn pr_word(words: [u32; 4]) -> Instr {
    let bits = words
        .iter()
        .rev()
        .fold(0_u128, |bits, &word| bits << 32 | u128::from(word));
    decode(Instruction::from_bits(bits)).expect("decodes")
}

/// `VertexTransform`'s attribute words: `ALD.64 R0, a[0x80]` ("load two input components") and
/// `AST.64 a[0x70], R0` ("store position X/Y"), with the arithmetic between them left out (it is
/// other groups'). Generic 0's two words reach the position's X and Y.
#[test]
fn pr_vertex_transform_attribute_words() {
    let load = pr_word([0xff00_7321, 0x0000_80ff, 0x0000_0400, 0x000e_2200]);
    let store = pr_word([0xff00_7322, 0x0000_7000, 0x0000_04ff, 0x000f_e200]);
    assert_eq!(load, Instr::new(Op::ALd(ald(r(0), 0x80, 2))));
    assert_eq!(store, Instr::new(Op::ASt(ast(r(0), 0x70, 2))));
    let ir = translate_program(vertex(), &[load, store, exit()]).expect("translates");
    for (x, y) in [(2.0_f32, 4.0_f32), (-4.0, 0.5)] {
        let mut machine = Machine::default();
        machine
            .attributes
            .insert((attr(GENERIC_0_X), 0), f32_bits(x));
        machine
            .attributes
            .insert((attr(GENERIC_0_X + 1), 0), f32_bits(y));
        run(&ir, &mut machine);
        assert_eq!(
            machine.outputs.into_iter().collect::<Vec<_>>(),
            vec![((attr(28), 0), f32_bits(x)), ((attr(29), 0), f32_bits(y))]
        );
    }
}

/// "Native scalar attribute stores accept the zero register without wrapping": one store of zero
/// to the position's X. (The PR's broken variant, two words from `RZ`, is a refusal of its
/// Maxwell lowering, `LowerToMaxwell`; its native translator accepts it, `AST`:
/// `data_reg == 255 ||`, and so does this one, `ast_from_rz_stores_zero_for_every_word`.)
#[test]
fn pr_scalar_attribute_store_from_rz() {
    let store = pr_word([0xff00_7322, 0x0000_70ff, 0x0000_00ff, 0]);
    assert_eq!(store, Instr::new(Op::ASt(ast(Gpr::RZ, 0x70, 1))));
    let ir = translate_program(vertex(), &[store, exit()]).expect("translates");
    let mut machine = Machine::default();
    run(&ir, &mut machine);
    assert_eq!(
        machine.outputs.into_iter().collect::<Vec<_>>(),
        vec![((attr(28), 0), 0)]
    );
}

/// "Native constant interpolation retains attribute and flat input behavior": `IPA.CONSTANT` of
/// generic 7's X with a flat input map (`imap_generic_vector[7].raw = 0x55`, constant in every
/// component) reads attribute 60 once and does not multiply.
#[test]
fn pr_constant_interpolation() {
    let instr = pr_word([0x0004_7326, 255, 0x000e_403c, 0]);
    assert_eq!(
        instr,
        Instr::new(Op::Ipa(OpIpa {
            freq: InterpFreq::Constant,
            ..ipa(r(4), addr(60))
        }))
    );
    let header = header_with(7, [PixelImap::Constant; 4]);
    let ir = translate_with_header(pixel(), &header, &[instr, exit()]).expect("translates");
    let text = ir.to_string();
    assert_eq!(text.matches("GetAttribute a0xf0").count(), 1);
    assert_eq!(text.matches("GetAttribute").count(), 1);
    assert!(!text.contains("FPMul32"));
    let mut machine = Machine::default();
    machine.attributes.insert((attr(60), 0), f32_bits(0.75));
    run(&ir, &mut machine);
    assert_eq!(machine.regs[4], f32_bits(0.75));
}

/// "Native VPRS index exports retain their identity separately from viewport masks": one store to
/// attribute word 23 (address `0x5c`) from R4.
#[test]
fn pr_vprs_index_export() {
    let store = pr_word([0xff00_7322, 0x0000_5c04, 0xff, 0]);
    assert_eq!(store, Instr::new(Op::ASt(ast(r(4), 0x5c, 1))));
    let ir = translate_program(vertex(), &[store, exit()]).expect("translates");
    let mut machine = Machine::default();
    machine.regs[4] = 0x1234;
    run(&ir, &mut machine);
    assert_eq!(
        machine.outputs.into_iter().collect::<Vec<_>>(),
        vec![((attr(23), 0), 0x1234)]
    );
}

/// The PR's `S2R R3, SR_TID.X` word (from its trap-guard case, where it is the unknown
/// instruction) and its `ALD R0, a[0x2fc]` (the vertex id, from its integer forms case) read
/// the thread id's X and attribute word 191.
#[test]
fn pr_s2r_and_vertex_id_words() {
    let s2r = pr_word([0x0003_7919, 0, 0x2100, 0]);
    assert_eq!(
        s2r,
        Instr::new(Op::S2R(OpS2R {
            dst: S2RDst::Reg(r(3)),
            sr: SpecialReg::TID_X
        }))
    );
    let load = pr_word([0xff00_7321, 0x0002_fcff, 0, 0x000e_2200]);
    assert_eq!(load, Instr::new(Op::ALd(ald(r(0), 0x2fc, 1))));
    let ir = translate_program(vertex(), &[s2r, load, exit()]).expect("translates");
    let mut machine = system_machine(0);
    machine.attributes.insert((attr(191), 0), 42);
    run(&ir, &mut machine);
    assert_eq!(machine.regs[3], 11);
    assert_eq!(machine.regs[0], 42);
}

// ---------------------------------------------------------------------------------------------
// Generated operations.

/// Any operation of the group, with arbitrary field values (wider than the decoder produces:
/// any word count, any address).
fn any_group_op() -> impl Strategy<Value = Op> {
    let ald = (
        any_gpr(),
        any_gpr(),
        any_gpr(),
        any::<u16>(),
        any::<u8>(),
        any::<[bool; 3]>(),
    )
        .prop_map(|(dst, vtx, offset, addr, comps, [patch, phys, output])| {
            Op::ALd(OpALd {
                dst,
                vtx,
                offset,
                addr,
                comps,
                patch,
                phys,
                output,
            })
        });
    let ast = (
        any_gpr(),
        any_gpr(),
        any_gpr(),
        any::<u16>(),
        any::<u8>(),
        any::<[bool; 2]>(),
    )
        .prop_map(|(data, vtx, offset, addr, comps, [patch, phys])| {
            Op::ASt(OpASt {
                data,
                vtx,
                offset,
                addr,
                comps,
                patch,
                phys,
            })
        });
    let ipa = (
        any_gpr(),
        any::<u16>(),
        0..InterpFreq::ALL.len(),
        0..InterpLoc::ALL.len(),
        any_gpr(),
        any_pred(),
    )
        .prop_map(|(dst, addr, freq, loc, offset, pred)| {
            Op::Ipa(OpIpa {
                dst,
                addr,
                freq: InterpFreq::ALL[freq],
                loc: InterpLoc::ALL[loc],
                offset,
                pred,
            })
        });
    let al2p = (
        any_gpr(),
        any_gpr(),
        any::<u16>(),
        any::<u8>(),
        any::<bool>(),
    )
        .prop_map(|(dst, offset, addr, comps, output)| {
            Op::AL2P(OpAL2P {
                dst,
                offset,
                addr,
                comps,
                output,
            })
        });
    prop_oneof![ald, ast, ipa, al2p, any_system_value_op()]
}

/// Any system-value or geometry operation of the group, with arbitrary field values.
fn any_system_value_op() -> impl Strategy<Value = Op> {
    let s2r = (
        prop_oneof![
            any_gpr().prop_map(S2RDst::Reg),
            any_uniform_reg().prop_map(S2RDst::Uniform)
        ],
        any::<u8>(),
    )
        .prop_map(|(dst, sr)| {
            Op::S2R(OpS2R {
                dst,
                sr: SpecialReg(sr),
            })
        });
    let cs2r = (any_gpr(), any::<u8>(), any::<bool>()).prop_map(|(dst, sr, wide)| {
        Op::CS2R(OpCS2R {
            dst,
            sr: SpecialReg(sr),
            wide,
        })
    });
    let pixld = (any_gpr(), 0..PixVal::ALL.len(), any_pred()).prop_map(|(dst, val, pred)| {
        Op::PixLd(OpPixLd {
            dst,
            val: PixVal::ALL[val],
            pred,
        })
    });
    let out = (
        any_gpr(),
        any_gpr(),
        any_unmodified_src(),
        0..=OutType::ALL.len(),
    )
        .prop_map(
            |(dst, handle, stream, out_type)| match OutType::ALL.get(out_type) {
                Some(&out_type) => Op::Out(OpOut {
                    dst,
                    handle,
                    stream,
                    out_type,
                }),
                None => Op::OutFinal(OpOutFinal {
                    dst,
                    handle,
                    stream,
                }),
            },
        );
    prop_oneof![s2r, cs2r, pixld, out]
}

proptest! {
    #![proptest_config(config())]

    /// Generated operations of the group, under any guard, never panic the translator, with or
    /// without a header, in either stage; what it accepts verifies.
    #[test]
    fn generated_operations_never_panic(
        op in any_group_op(),
        guard in any_pred_src(),
        pixel_stage in any::<bool>(),
        header_index in 0_usize..32,
        imap in 0_usize..4,
    ) {
        let stage = if pixel_stage { pixel() } else { vertex() };
        let maps = [
            PixelImap::Unused,
            PixelImap::Constant,
            PixelImap::Perspective,
            PixelImap::ScreenLinear,
        ];
        let header = header_with(header_index, [maps[imap]; 4]);
        let program = [Instr { guard, op }, exit()];
        for result in [
            translate_program(stage, &program),
            translate_with_header(stage, &header, &program),
        ] {
            if let Ok(ir) = &result {
                prop_assert_eq!(ir.verify(), Ok(()));
            }
        }
    }
}
