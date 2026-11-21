//! Tests of the uniform datapath translator: each instruction and modifier run through the IR
//! interpreter (slice 7) with expected values from the instruction's documented behaviour, the
//! predicate files the lowering reads, the guards, every refusal with its exact error, PR
//! #4588's cases for this group rebuilt with the test assembler, the register classes through
//! SSA, and a no-panic property test.
//!
//! Every case is built as an [`Instr`], assembled by the test assembler and decoded again
//! before it is translated (but `UP2UR`, a shell no word holds), so the words are synthetic.

mod alu;
mod constant;
mod files;
mod global;
mod guards;
mod pr4588;
mod rejections;

use proptest::prelude::*;

use super::*;
use crate::asm::assemble;
use crate::decode::{Gpr, OpIAdd3, UniformPred};
use crate::interp::Machine;
use crate::ir::{Function, Terminator};
use crate::testing::{any_word_of, config};

/// The general register `index`; as a twin's operand, the uniform register `index`.
pub(super) fn r(index: u8) -> Gpr {
    Gpr::new(index)
}

/// A register source (a uniform register in a twin).
pub(super) fn reg(index: u8) -> AluSrc {
    AluSrc::reg(r(index))
}

/// A negated register source.
pub(super) fn neg(index: u8) -> AluSrc {
    AluSrc::Reg {
        reg: r(index),
        abs: false,
        neg: true,
    }
}

/// `RZ` as a source (`URZ` in a twin).
pub(super) const RZ: AluSrc = AluSrc::Reg {
    reg: Gpr::RZ,
    abs: false,
    neg: false,
};

/// The predicate `index` (a uniform predicate in a twin that reads one).
pub(super) fn p(index: u8) -> Pred {
    Pred::new(index).expect("a predicate")
}

/// The predicate source `index`.
pub(super) fn ps(index: u8, negated: bool) -> PredSrc {
    PredSrc {
        reg: p(index),
        negated,
    }
}

/// `!PT`.
pub(super) const NOT_PT: PredSrc = PredSrc {
    reg: Pred::PT,
    negated: true,
};

/// The uniform register `index`.
pub(super) fn ur(index: u8) -> UniformReg {
    UniformReg::new(index).expect("a uniform register")
}

/// The uniform predicate `index`.
pub(super) fn up(index: u8) -> UniformPred {
    UniformPred::new(index).expect("a uniform predicate")
}

/// The uniform predicate source `index`.
pub(super) fn ups(index: u8, negated: bool) -> UniformPredSrc {
    UniformPredSrc {
        reg: up(index),
        negated,
    }
}

/// `op` on the uniform datapath.
pub(super) fn twin<T: TwinOperands>(op: T) -> Uniform<T> {
    Uniform::new(op).expect("the uniform encoding holds the case")
}

/// An `IADD3` with no carry outputs and `!PT` carries.
pub(super) fn iadd3(dst: u8, srcs: [AluSrc; 3]) -> OpIAdd3 {
    OpIAdd3 {
        dst: r(dst),
        overflow: [Pred::PT; 2],
        srcs,
        carry: [NOT_PT; 2],
    }
}

/// `instr` under `guard`.
pub(super) fn under(guard: PredSrc, op: Op) -> Instr {
    Instr { guard, op }
}

/// `instr` round-tripped through its word: assembled, then decoded.
pub(super) fn word(instr: &Instr) -> Instruction {
    let word = assemble(instr).expect("the test assembler encodes the case");
    assert_eq!(decode(word), Ok(*instr), "the case decodes back to itself");
    word
}

/// The function `instrs` translate to with guards honoured, through their words, in one entry
/// block that returns.
pub(super) fn build(instrs: &[Instr]) -> Result<Function, UniformError> {
    let mut function = Function::new();
    let entry = function.entry();
    let mut builder = Builder::new(&mut function, entry);
    for instr in instrs {
        translate_uniform_word(&mut builder, word(instr), GuardMode::Honour)?;
    }
    builder.terminate(Terminator::Return)?;
    Ok(function)
}

/// Translates `instrs` and runs them on `machine`.
pub(super) fn run(instrs: &[Instr], machine: &mut Machine) {
    let function = build(instrs).expect("translates");
    assert_eq!(function.verify(), Ok(()));
    machine.run(&function).expect("runs");
}

/// Runs one unguarded instruction on `machine`.
pub(super) fn run_op(op: Op, machine: &mut Machine) {
    run(&[Instr::new(op)], machine);
}

/// A machine with uniform registers `uregs` set from `UR0` up, and every general register
/// holding a value no uniform case expects (`0xbad0_0000` plus its number), so a read of the
/// wrong file changes a result.
pub(super) fn machine_with(uregs: &[u32]) -> Machine {
    let mut machine = Machine::new();
    for (index, slot) in (0_u32..).zip(machine.regs.iter_mut()) {
        *slot = 0xbad0_0000 | index;
    }
    for (slot, value) in machine.uniform_regs.iter_mut().zip(uregs) {
        *slot = *value;
    }
    machine
}

/// The text of the function `instrs` translate to.
pub(super) fn text(instrs: &[Instr]) -> String {
    build(instrs).expect("translates").to_string()
}

/// How many operations of `function` are `opcode`.
pub(super) fn count(function: &Function, opcode: Opcode) -> usize {
    function
        .to_string()
        .lines()
        .filter(|line| {
            line.split_whitespace()
                .any(|word| word == format!("{opcode:?}"))
        })
        .count()
}

/// The values the generated cases draw from: the edges of the 32-bit range and the carry
/// boundary, and a pattern with every byte distinct.
pub(super) const VALUES: [u32; 6] = [0, 1, 0x7fff_ffff, 0x8000_0000, 0xffff_ffff, 0x1234_5678];

/// `in_group` takes exactly the uniform datapath's 21 operations: every one written out here,
/// each decoded from a sample word but the shell `UP2UR`, and no operation of any other group.
#[test]
fn in_group_is_the_uniform_datapath() {
    let expected = [
        "r2ur",
        "up2ur",
        "umov",
        "uldc",
        "uplop3",
        "uiadd3",
        "uiadd3.x",
        "ulea",
        "ulea.x",
        "ulop3",
        "uprmt",
        "ushf",
        "uimad",
        "uimad.wide",
        "uisetp",
        "usel",
        "usgxt",
        "ubmsk",
        "uflo",
        "ubrev",
        "upopc",
    ];
    let mut seen = std::collections::BTreeSet::new();
    for instr in crate::translate::driver::tests::decoded_samples() {
        if in_group(&instr.op) {
            seen.insert(instr.op.mnemonic());
        } else {
            assert!(!expected.contains(&instr.op.mnemonic()), "{instr}");
        }
    }
    assert!(in_group(&Op::UP2UR(crate::decode::OpUP2UR {})));
    seen.insert("up2ur");
    let mut expected: Vec<_> = expected.to_vec();
    expected.sort_unstable();
    assert_eq!(seen.into_iter().collect::<Vec<_>>(), expected);
}

/// Every decoded sample of the group, in both guard modes and under a guard, translates or is
/// refused (never by the builder), and what translates verifies.
#[test]
fn every_decoded_form_translates_or_is_refused() {
    let mut translated = std::collections::BTreeSet::new();
    for instr in crate::translate::driver::tests::decoded_samples() {
        if !in_group(&instr.op) {
            continue;
        }
        for guard in [PredSrc::TRUE, ps(2, true)] {
            for mode in [GuardMode::Honour, GuardMode::Tested] {
                let mut function = Function::new();
                let entry = function.entry();
                let mut builder = Builder::new(&mut function, entry);
                let instr = Instr { guard, ..instr };
                let result = translate_uniform(&mut builder, &instr, mode);
                assert!(
                    !matches!(result, Err(UniformError::Build(_))),
                    "{instr}: {result:?}"
                );
                if result.is_ok() {
                    translated.insert(instr.op.mnemonic());
                    builder.terminate(Terminator::Return).expect("terminates");
                    assert_eq!(function.verify(), Ok(()), "{instr}");
                }
            }
        }
    }
    // The samples are canonical words and their one-bit changes, whose predicate fields are
    // mostly `P0`; PR #4588 requires `!PT` carries (refused by the integer group's checks) for
    // `UIADD3`, `ULEA`, `UIMAD.WIDE` and `UISETP`'s low-word predicate, so these translate in
    // no sample, and `UBMSK` is refused by name.
    assert_eq!(
        translated.into_iter().collect::<Vec<_>>(),
        [
            "r2ur", "ubrev", "uflo", "uiadd3.x", "uimad", "uldc", "ulea.x", "ulop3", "umov",
            "uplop3", "upopc", "uprmt", "usel", "usgxt", "ushf",
        ]
    );
}

/// Uniform and per-thread registers and predicates stay in their classes through the SSA
/// rewrite (slice 8): a program that writes `R1`, `UR1`, `P1` and `UP1` to different values and
/// reads each back gives the same state before and after the rewrite, and after it no value
/// read from one class names the other.
#[test]
fn register_classes_survive_ssa() {
    use crate::decode::{MovSrc, OpMov};
    use crate::opt::{Pass, run as run_pass};
    let mov = Instr::new(Op::Mov(OpMov {
        dst: r(1),
        src: MovSrc::Imm(0x1111),
        quad_lanes: OpMov::ALL_LANES,
    }));
    let umov = Instr::new(Op::UMov(OpUMov {
        dst: ur(1),
        src: UMovSrc::Imm(0x2222),
    }));
    // UIADD3 UR2 = UR1 + UR1, with UP1 the carry; IADD3 R2 = R1 + UR1 (a vector reading a
    // uniform register).
    let uiadd3 = Instr::new(Op::UIAdd3(twin(OpIAdd3 {
        overflow: [p(1), Pred::PT],
        ..iadd3(2, [reg(1), reg(1), RZ])
    })));
    let vadd = Instr::new(Op::IAdd3(OpIAdd3 {
        srcs: [
            reg(1),
            AluSrc::UniformReg {
                reg: ur(1),
                abs: false,
                neg: false,
            },
            RZ,
        ],
        ..iadd3(2, [RZ; 3])
    }));
    let mut function = Function::new();
    let entry = function.entry();
    let mut builder = Builder::new(&mut function, entry);
    for instr in [mov, umov, uiadd3, vadd] {
        let word = word(&instr);
        if in_group(&instr.op) {
            translate_uniform_word(&mut builder, word, GuardMode::Honour).expect("uniform");
        } else {
            integer::translate_integer_word(&mut builder, word, GuardMode::Honour)
                .expect("integer");
        }
    }
    builder.terminate(Terminator::Return).expect("terminates");
    let mut before = Machine::new();
    before.preds[1] = true;
    let mut after = before.clone();
    before.run(&function).expect("runs");
    assert_eq!(before.regs[1], 0x1111);
    assert_eq!(before.uniform_regs[1], 0x2222);
    assert_eq!(before.uniform_regs[2], 0x4444);
    assert_eq!(before.regs[2], 0x3333);
    assert!(!before.uniform_preds[1], "no carry out of 0x2222 + 0x2222");
    assert!(before.preds[1], "the per-thread predicate is untouched");
    run_pass(&mut function, Pass::SsaRewrite).expect("rewrites");
    assert_eq!(function.verify(), Ok(()));
    after.run(&function).expect("runs");
    assert_eq!(after.regs, before.regs);
    assert_eq!(after.uniform_regs, before.uniform_regs);
    assert_eq!(after.preds, before.preds);
    assert_eq!(after.uniform_preds, before.uniform_preds);
    let text = function.to_string();
    assert!(!text.contains("GetUniformRegister"), "{text}");
    assert!(!text.contains("GetRegister"), "{text}");
}

proptest! {
    #![proptest_config(config())]

    /// Generated words of every form of the group decode or not; what decodes translates or is
    /// refused, never by the builder; what translates verifies and runs on the interpreter.
    #[test]
    fn generated_words_never_panic(
        word in any_word_of(crate::decode::uniform::FORMS),
        uregs in proptest::collection::vec(any::<u32>(), 64),
        upreds in any::<[bool; 8]>(),
        preds in any::<[bool; 8]>(),
        guard in any::<bool>(),
    ) {
        let mut function = Function::new();
        let entry = function.entry();
        let mut builder = Builder::new(&mut function, entry);
        let mode = if guard { GuardMode::Honour } else { GuardMode::Tested };
        match translate_uniform_word(&mut builder, word, mode) {
            Ok(()) => {
                prop_assert!(builder.terminate(Terminator::Return).is_ok());
                prop_assert_eq!(function.verify(), Ok(()));
                let mut machine = Machine::new();
                for (slot, value) in machine.uniform_regs.iter_mut().zip(&uregs) {
                    *slot = *value;
                }
                machine.uniform_preds = upreds;
                machine.preds = preds;
                let bytes = vec![0xa5; usize::from(u16::MAX) + 1];
                for bank in 0..32 {
                    machine.cbufs.insert(bank, bytes.clone());
                }
                // A global load may fault (an address no region covers); the run is checked
                // not to panic, and anything else to succeed.
                let ran = machine.run(&function);
                if !matches!(
                    ran,
                    Err(crate::interp::InterpError::OutOfBounds { .. }
                        | crate::interp::InterpError::Misaligned { .. })
                ) {
                    prop_assert!(ran.is_ok(), "{:?}", ran.err());
                }
            }
            Err(UniformError::Build(error)) => prop_assert!(false, "build error {error}"),
            Err(UniformError::Modifier(name)) => prop_assert!(false, "{name}: modifier"),
            Err(UniformError::OtherGroup(name)) => prop_assert!(false, "{name}: other group"),
            Err(_) => {}
        }
    }
}
