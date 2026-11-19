//! Tests of the integer ALU translator: each instruction and modifier run through the IR
//! interpreter (slice 7) with expected values from the instruction's documented behaviour,
//! golden IR, the guards, every refusal with its exact error, PR #4588's cases for this group
//! and a no-panic property test.
//!
//! Every case is built as an [`Instr`], assembled by the test assembler and decoded again
//! before it is translated, so the words are synthetic (never copied from a game).

mod alu;
mod golden;
mod guards;
mod pr4588;
mod rejections;

use proptest::prelude::*;

use super::*;
use crate::asm::assemble;
use crate::decode::{AluSrc, CBuf, CBufRef, OpIAdd3, UniformPred};
use crate::interp::{InterpError, Machine};
use crate::ir::{Function, Terminator};
use crate::testing::{any_word_of, config};

pub(super) fn r(index: u8) -> Gpr {
    Gpr::new(index)
}

pub(super) fn reg(index: u8) -> AluSrc {
    AluSrc::reg(r(index))
}

pub(super) fn neg(index: u8) -> AluSrc {
    AluSrc::Reg {
        reg: r(index),
        abs: false,
        neg: true,
    }
}

pub(super) fn p(index: u8) -> Pred {
    Pred::new(index).unwrap_or(Pred::PT)
}

pub(super) fn ps(index: u8, negated: bool) -> PredSrc {
    PredSrc {
        reg: p(index),
        negated,
    }
}

pub(super) fn ur(index: u8) -> UniformReg {
    UniformReg::new(index).unwrap_or(UniformReg::URZ)
}

pub(super) fn upred(index: u8) -> UniformPred {
    UniformPred::new(index).unwrap_or(UniformPred::UPT)
}

/// A bound constant-buffer source.
pub(super) fn cb(bank: u8, offset: u16) -> AluSrc {
    AluSrc::CBuf {
        cbuf: CBufRef {
            buf: CBuf::Binding(bank),
            offset,
        },
        abs: false,
        neg: false,
    }
}

/// A uniform-register source.
pub(super) fn ureg(index: u8) -> AluSrc {
    AluSrc::UniformReg {
        reg: ur(index),
        abs: false,
        neg: false,
    }
}

/// `!PT`.
pub(super) const NOT_PT: PredSrc = super::NOT_PT;

/// `instr` round-tripped through its word: assembled, then decoded.
pub(super) fn word(instr: &Instr) -> Instruction {
    let word = assemble(instr).expect("the test assembler encodes the case");
    assert_eq!(decode(word), Ok(*instr), "the case decodes back to itself");
    word
}

/// The function `instrs` translate to, through their words, in one entry block that returns.
pub(super) fn build(instrs: &[Instr]) -> Result<Function, IntegerError> {
    let mut function = Function::new();
    let entry = function.entry();
    let mut builder = Builder::new(&mut function, entry);
    for instr in instrs {
        translate_integer_word(&mut builder, word(instr), GuardMode::Honour)?;
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

/// A machine with `regs` set from register 0 up.
pub(super) fn machine_with(regs: &[u32]) -> Machine {
    let mut machine = Machine::new();
    for (slot, value) in machine.regs.iter_mut().zip(regs) {
        *slot = *value;
    }
    machine
}

/// Constant buffer `bank` holding `words` from offset 0, little-endian.
pub(super) fn with_cbuf(machine: &mut Machine, bank: u32, words: &[u32]) {
    let bytes = words.iter().flat_map(|word| word.to_le_bytes()).collect();
    machine.cbufs.insert(bank, bytes);
}

/// An `IADD3` (`IADD3.X` when `x`) with no carry outputs and `!PT` carries.
pub(super) fn iadd3(dst: u8, srcs: [AluSrc; 3]) -> OpIAdd3 {
    OpIAdd3 {
        dst: r(dst),
        overflow: [Pred::PT; 2],
        srcs,
        carry: [NOT_PT; 2],
    }
}

/// The values the generated cases draw from: the edges of the 32-bit range and the carry
/// boundary, and two patterns with every byte distinct.
pub(super) const VALUES: [u32; 8] = [
    0,
    1,
    2,
    0x7fff_ffff,
    0x8000_0000,
    0xffff_fffe,
    0xffff_ffff,
    0x1234_5678,
];

/// The low word of `value`.
pub(super) fn lo32(value: u64) -> u32 {
    u32::try_from(value & u64::from(u32::MAX)).unwrap_or(0)
}

/// The high word of `value`.
pub(super) fn hi32(value: u64) -> u32 {
    lo32(value >> 32)
}

/// `value`'s bits as a signed integer.
pub(super) fn to_i32(value: u32) -> i32 {
    i32::from_ne_bytes(value.to_ne_bytes())
}

/// `value`'s bits as an unsigned integer.
pub(super) fn to_u32(value: i32) -> u32 {
    u32::from_ne_bytes(value.to_ne_bytes())
}

/// `value`'s bits as a signed 64-bit integer.
pub(super) fn to_i64(value: u64) -> i64 {
    i64::from_ne_bytes(value.to_ne_bytes())
}

/// `value`'s bits as an unsigned 64-bit integer.
pub(super) fn to_u64(value: i64) -> u64 {
    u64::from_ne_bytes(value.to_ne_bytes())
}

/// `value` sign-extended to 64 bits.
pub(super) fn sext(value: u32) -> u64 {
    to_u64(i64::from(to_i32(value)))
}

/// The signed 64-bit product of two words, as its bits.
pub(super) fn signed_product(a: u32, b: u32) -> u64 {
    to_u64(i64::from(to_i32(a)) * i64::from(to_i32(b)))
}

/// The bytes a constant buffer needs for the furthest read a decoded word can make: the high
/// word of an `IMAD.WIDE` addend, one word past a 16-bit offset, read unaligned, which takes
/// the aligned word after it too.
pub(super) fn furthest_cbuf_read() -> usize {
    let word = usize::try_from(WORD_BYTES).unwrap_or(usize::MAX);
    let high = usize::from(u16::MAX) + word;
    // The aligned word holding the high word's first byte, then the word after it.
    let last_word = high / word * word + word;
    last_word + word
}

proptest! {
    #![proptest_config(config())]

    /// Generated words of every integer form decode or not; what decodes translates or is
    /// refused, never panics; what translates verifies and runs on the interpreter with every
    /// register, predicate and constant buffer it may read.
    #[test]
    fn generated_words_never_panic(
        word in any_word_of(crate::decode::integer::FORMS),
        regs in proptest::collection::vec(any::<u32>(), 256),
        preds in any::<[bool; 8]>(),
        guard in any::<bool>(),
    ) {
        let mut function = Function::new();
        let entry = function.entry();
        let mut builder = Builder::new(&mut function, entry);
        let mode = if guard { GuardMode::Honour } else { GuardMode::Tested };
        let before = builder.function().clone();
        match translate_integer_word(&mut builder, word, mode) {
            Ok(()) => {
                prop_assert!(builder.terminate(Terminator::Return).is_ok());
                prop_assert_eq!(function.verify(), Ok(()));
                let mut machine = Machine::new();
                for (slot, value) in machine.regs.iter_mut().zip(&regs) {
                    *slot = *value;
                }
                machine.preds = preds;
                // Every bank, as long as the furthest read any word can make.
                let bytes = vec![0xa5; furthest_cbuf_read()];
                for bank in 0..32 {
                    machine.cbufs.insert(bank, bytes.clone());
                }
                let ran = machine.run(&function);
                prop_assert!(ran.is_ok(), "{:?}", ran.err());
            }
            Err(IntegerError::Build(error)) => prop_assert!(false, "build error {error}"),
            // The decoder never sets a modifier this module refuses.
            Err(IntegerError::Modifier(name)) => prop_assert!(false, "{name}: modifier"),
            Err(_) => prop_assert_eq!(builder.function(), &before),
        }
    }
}

/// An instruction of another group is refused by its mnemonic, and so is a word that does not
/// decode, with the decoder's exact error.
#[test]
fn other_groups_and_unknown_words_are_refused() {
    let mut function = Function::new();
    let entry = function.entry();
    let mut builder = Builder::new(&mut function, entry);
    let nop = Instr::new(Op::Nop(crate::decode::OpNop {}));
    let error = translate_integer(&mut builder, &nop, GuardMode::Honour);
    assert_eq!(error, Err(IntegerError::OtherGroup("nop")));
    assert_eq!(
        error.map_err(|error| error.to_string()),
        Err("nop is not an integer ALU instruction".to_owned())
    );
    // Opcode 0 is no form; a MOV with a bit its form does not explain is refused at that bit.
    let unknown = Instruction::from_bits(0);
    assert_eq!(
        translate_integer_word(&mut builder, unknown, GuardMode::Honour),
        Err(IntegerError::Decode(DecodeError::UnknownOpcode {
            opcode: 0
        }))
    );
    let mov = word(&Instr::new(Op::Mov(OpMov {
        dst: r(1),
        src: crate::decode::MovSrc::Reg(r(2)),
        quad_lanes: OpMov::ALL_LANES,
    })));
    let stray = Instruction::from_bits(mov.bits() | 1 << 62);
    let refused = translate_integer_word(&mut builder, stray, GuardMode::Honour);
    assert_eq!(
        refused,
        Err(IntegerError::Decode(DecodeError::UnknownBit {
            opcode: 0x202,
            bit: 62
        }))
    );
    assert_eq!(
        refused.map_err(|error| error.to_string()),
        Err("opcode 0x202 has bit 62 set, which its form does not use".to_owned())
    );
    assert_eq!(builder.function(), &Function::new());
}

/// [`is_integer`] holds for exactly the group's operations: every decoded form of the integer
/// table, and none of another group's forms.
#[test]
fn the_group_is_the_integer_table() {
    let integer = crate::decode::integer::FORMS;
    let mut seen = 0;
    for group in crate::decode::GROUPS {
        for form in group {
            let ours = integer.iter().any(|row| row.opcode == form.opcode);
            // A word of the form with only its opcode and fixed bits set decodes to the form's
            // operation, whose group this checks.
            let word = Instruction::from_bits(form.fixed)
                .with_field(crate::decode::OPCODE, u64::from(form.opcode));
            if let Ok(instr) = decode(word) {
                assert_eq!(is_integer(&instr.op), ours, "{}", instr.op.mnemonic());
                seen += usize::from(ours);
            }
        }
    }
    // Every integer form decodes with its fields zero (`PLOP3`, `P2R` and `R2P` included).
    assert_eq!(seen, integer.len());
}

/// The interpreter's errors are what a failed case reports; a run that reads a constant buffer
/// the machine lacks fails rather than reading zero, so the cases give every buffer they read.
#[test]
fn a_missing_constant_buffer_fails_the_run() {
    let function = build(&[Instr::new(Op::Mov(OpMov {
        dst: r(1),
        src: crate::decode::MovSrc::CBuf(CBufRef {
            buf: CBuf::Binding(3),
            offset: 8,
        }),
        quad_lanes: OpMov::ALL_LANES,
    }))])
    .expect("translates");
    let ran = Machine::new().run(&function);
    assert!(
        matches!(ran, Err(InterpError::OutOfBounds { .. })),
        "{ran:?}"
    );
}

/// The furthest constant-buffer read: `IMAD.WIDE` with its addend at offset `0xffff` runs on
/// buffers of exactly [`furthest_cbuf_read`] bytes (`0x10008`) and fails on one byte fewer.
#[test]
fn the_furthest_constant_buffer_read_fits() {
    assert_eq!(furthest_cbuf_read(), 0x1_0008);
    let function = build(&[Instr::new(Op::IMad64(crate::decode::OpIMad64 {
        dst: r(4),
        srcs: [reg(1), reg(2), cb(5, u16::MAX)],
        signed: false,
        carry: NOT_PT,
    }))])
    .expect("translates");
    for (len, fits) in [
        (furthest_cbuf_read(), true),
        (furthest_cbuf_read() - 1, false),
    ] {
        let mut machine = Machine::new();
        machine.cbufs.insert(5, vec![0xa5; len]);
        let ran = machine.run(&function);
        if fits {
            assert!(ran.is_ok(), "{ran:?}");
        } else {
            // The last word read, at 0x10004, is the one that no longer fits.
            let fields = match &ran {
                Err(InterpError::OutOfBounds {
                    offset, size, len, ..
                }) => Some((*offset, *size, *len)),
                _ => None,
            };
            assert_eq!(fields, Some((0x1_0004, 4, 0x1_0007)), "{ran:?}");
        }
    }
}
