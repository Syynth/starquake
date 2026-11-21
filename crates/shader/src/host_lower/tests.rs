//! The host lowering passes' tests: the tables against Eden's literals, the written-out
//! conversions against their models and the interpreter, interpreter equivalence before and
//! after each pass (one case per mapped opcode, and generated programs with a fixed seed), the
//! listed precision losses, the refusals and their order, naga validation of the lowered
//! modules under the reduced capabilities, and the production translator end to end.
//!
//! The harness builds programs of single operations from the opcode table's signatures: each
//! argument is read from registers through the boundary conversions (a 64-bit float from a
//! register pair through `PackDouble2x32`, a 16-bit float from a packed word through
//! `UnpackFloat2x16`), and each result is written back to registers the same way, so the
//! interpreter's registers before and after a pass can be compared output by output.

mod bits;
mod end_to_end;
mod equivalence;
mod passes;
mod precision;
mod refusals;
mod table;
mod validation;

use super::{HostPass, Rewrite, run};
use crate::decode::Gpr;
use crate::interp::{Machine, REG_SLOTS};
use crate::ir::{Builder, Function, Imm, Inst, Opcode, Operand, Terminator, Type, Value};

/// The first register of each input bank and how many registers it has.
const F64_BANK: (u8, u8) = (0, 32);
const F32_BANK: (u8, u8) = (32, 16);
const HALF_BANK: (u8, u8) = (48, 16);
const INT_BANK: (u8, u8) = (64, 16);
const BOOL_BANK: (u8, u8) = (80, 16);
/// The first register a program writes its outputs to, above every input bank.
const OUTPUT_BASE: u8 = 128;

/// What one output register (or register pair) holds, to compare it as its type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Out {
    /// A 64-bit float in a register pair, low word first.
    F64(u8),
    /// A 32-bit float.
    F32(u8),
    /// A 16-bit float packed twice into a word.
    F16(u8),
    /// A 32-bit integer.
    U32(u8),
    /// A 64-bit integer or two words in a register pair (compared as a 64-bit float by the f64
    /// pass's tests, whose pairs hold one).
    U64(u8),
    /// A boolean as 1 or 0.
    Bool(u8),
}

/// One statement of a generated program: an opcode, and a pick of input register for each
/// argument (taken modulo the bank's size).
#[derive(Clone, Debug)]
struct Stmt {
    opcode: Opcode,
    picks: [u8; 8],
}

/// A built program: the function and what each output holds, with the opcode that wrote it.
struct Program {
    function: Function,
    outputs: Vec<(Opcode, Out)>,
}

fn reg(index: u8) -> Operand {
    Operand::Imm(Imm::Reg(Gpr::new(index)))
}

fn u32_imm(value: u32) -> Operand {
    Operand::Imm(Imm::U32(value))
}

/// Builds statements into one block, allocating output registers upwards from
/// [`OUTPUT_BASE`].
struct Writer<'f> {
    b: Builder<'f>,
    next_out: u8,
    outputs: Vec<(Opcode, Out)>,
}

impl Writer<'_> {
    fn op(&mut self, opcode: Opcode, args: &[Operand]) -> Operand {
        Operand::Value(
            self.b
                .op(opcode, args)
                .map_err(|error| format!("{opcode}: {error}"))
                .expect("builds"),
        )
    }

    fn get(&mut self, index: u8) -> Operand {
        self.op(Opcode::GetRegister, &[reg(index)])
    }

    fn set(&mut self, index: u8, value: Operand) {
        self.op(Opcode::SetRegister, &[reg(index), value]);
    }

    fn out(&mut self, words: u8) -> u8 {
        let at = self.next_out;
        self.next_out = self.next_out.checked_add(words).expect("output registers");
        assert!(
            usize::from(self.next_out) < REG_SLOTS - 1,
            "too many outputs"
        );
        at
    }

    /// A value of type `ty` read from the bank for it, at pick `pick`.
    fn input(&mut self, ty: Type, pick: u8) -> Operand {
        let at = |(base, count): (u8, u8)| base + pick % count;
        match ty {
            Type::F64 => {
                let pair = self.words(F64_BANK, pick);
                self.op(Opcode::PackDouble2x32, &[pair])
            }
            Type::F32 => {
                let bits = self.get(at(F32_BANK));
                self.op(Opcode::BitCastF32U32, &[bits])
            }
            Type::F16 => {
                let word = self.get(at(HALF_BANK));
                let pair = self.op(Opcode::UnpackFloat2x16, &[word]);
                self.op(
                    Opcode::CompositeExtractF16x2,
                    &[pair, u32_imm(u32::from(pick & 1))],
                )
            }
            Type::U32 => self.get(at(INT_BANK)),
            Type::U64 => {
                let pair = self.words(INT_BANK, pick);
                self.op(Opcode::PackUint2x32, &[pair])
            }
            Type::U32x2 => self.words(F64_BANK, pick),
            Type::U1 => {
                let word = self.get(at(BOOL_BANK));
                self.op(Opcode::INotEqual, &[word, u32_imm(0)])
            }
            Type::F16x2 if pick & 2 == 0 => {
                let word = self.get(at(HALF_BANK));
                self.op(Opcode::UnpackFloat2x16, &[word])
            }
            Type::F32x2 | Type::F16x2 | Type::F64x2 => self.vector(ty, 2, pick),
            Type::F32x3 | Type::F16x3 | Type::F64x3 => self.vector(ty, 3, pick),
            Type::F32x4 | Type::F16x4 | Type::F64x4 => self.vector(ty, 4, pick),
            other => unreachable!("no input of type {other}"),
        }
    }

    /// A register pair of `bank`, at pick `pick`, as two words.
    fn words(&mut self, (base, count): (u8, u8), pick: u8) -> Operand {
        let lo = base + (pick % (count / 2)) * 2;
        let low = self.get(lo);
        let high = self.get(lo + 1);
        self.op(Opcode::CompositeConstructU32x2, &[low, high])
    }

    /// A vector of `n` components of `ty`'s scalar, each picked from the next pick.
    fn vector(&mut self, ty: Type, n: u8, pick: u8) -> Operand {
        let (scalar, construct) = vector_ops(ty);
        let parts: Vec<Operand> = (0..n)
            .map(|i| self.input(scalar, pick.wrapping_add(i.wrapping_mul(5))))
            .collect();
        self.op(construct, &parts)
    }

    /// Writes `value` of type `ty` to fresh output registers, as `opcode`'s output.
    fn output(&mut self, opcode: Opcode, ty: Type, value: Operand) {
        match ty {
            Type::F64 => {
                let pair = self.op(Opcode::UnpackDouble2x32, &[value]);
                let at = self.out(2);
                for i in 0..2 {
                    let word = self.op(Opcode::CompositeExtractU32x2, &[pair, u32_imm(i)]);
                    self.set(at + u8::try_from(i).expect("small"), word);
                }
                self.outputs.push((opcode, Out::F64(at)));
            }
            Type::F32 => {
                let bits = self.op(Opcode::BitCastU32F32, &[value]);
                let at = self.out(1);
                self.set(at, bits);
                self.outputs.push((opcode, Out::F32(at)));
            }
            Type::F16 => {
                let pair = self.op(Opcode::CompositeConstructF16x2, &[value, value]);
                let word = self.op(Opcode::PackFloat2x16, &[pair]);
                let at = self.out(1);
                self.set(at, word);
                self.outputs.push((opcode, Out::F16(at)));
            }
            Type::U32 => {
                let at = self.out(1);
                self.set(at, value);
                let out = if matches!(opcode, Opcode::PackFloat2x16 | Opcode::PackHalf2x16) {
                    Out::F16(at)
                } else {
                    Out::U32(at)
                };
                self.outputs.push((opcode, out));
            }
            Type::U64 | Type::U32x2 => {
                let pair = if ty == Type::U64 {
                    self.op(Opcode::UnpackUint2x32, &[value])
                } else {
                    value
                };
                let at = self.out(2);
                for i in 0..2 {
                    let word = self.op(Opcode::CompositeExtractU32x2, &[pair, u32_imm(i)]);
                    self.set(at + u8::try_from(i).expect("small"), word);
                }
                self.outputs.push((opcode, Out::U64(at)));
            }
            Type::U1 => {
                let word = self.op(Opcode::SelectU32, &[value, u32_imm(1), u32_imm(0)]);
                let at = self.out(1);
                self.set(at, word);
                self.outputs.push((opcode, Out::Bool(at)));
            }
            Type::F32x2 | Type::F16x2 | Type::F64x2 => self.split(opcode, ty, 2, value),
            Type::F32x3 | Type::F16x3 | Type::F64x3 => self.split(opcode, ty, 3, value),
            Type::F32x4 | Type::F16x4 | Type::F64x4 => self.split(opcode, ty, 4, value),
            other => unreachable!("no output of type {other}"),
        }
    }

    fn split(&mut self, opcode: Opcode, ty: Type, n: u32, value: Operand) {
        let (scalar, _) = vector_ops(ty);
        let extract = extract_op(ty);
        for i in 0..n {
            let part = self.op(extract, &[value, u32_imm(i)]);
            self.output(opcode, scalar, part);
        }
    }

    /// One statement: `stmt.opcode` with its arguments read from the banks.
    fn stmt(&mut self, stmt: &Stmt) {
        let opcode = stmt.opcode;
        let index_slot = index_slot(opcode);
        let args: Vec<Operand> = opcode
            .args()
            .iter()
            .enumerate()
            .map(|(i, &ty)| {
                let pick = stmt.picks[i % stmt.picks.len()];
                if Some(i) == index_slot {
                    u32_imm(u32::from(pick) % vector_len(opcode.args()[0]))
                } else if opcode == Opcode::BitCastF64U64 {
                    // The bits of a 64-bit float from its bank, not any integer.
                    let pair = self.words(F64_BANK, pick);
                    self.op(Opcode::PackUint2x32, &[pair])
                } else {
                    self.input(ty, pick)
                }
            })
            .collect();
        let result = self.op(opcode, &args);
        self.output(opcode, opcode.result(), result);
    }
}

/// The scalar type of a vector type and its construct opcode.
fn vector_ops(ty: Type) -> (Type, Opcode) {
    match ty {
        Type::F32x2 => (Type::F32, Opcode::CompositeConstructF32x2),
        Type::F32x3 => (Type::F32, Opcode::CompositeConstructF32x3),
        Type::F32x4 => (Type::F32, Opcode::CompositeConstructF32x4),
        Type::F16x2 => (Type::F16, Opcode::CompositeConstructF16x2),
        Type::F16x3 => (Type::F16, Opcode::CompositeConstructF16x3),
        Type::F16x4 => (Type::F16, Opcode::CompositeConstructF16x4),
        Type::F64x2 => (Type::F64, Opcode::CompositeConstructF64x2),
        Type::F64x3 => (Type::F64, Opcode::CompositeConstructF64x3),
        Type::F64x4 => (Type::F64, Opcode::CompositeConstructF64x4),
        other => unreachable!("{other} is not a float vector"),
    }
}

fn extract_op(ty: Type) -> Opcode {
    match ty {
        Type::F32x2 => Opcode::CompositeExtractF32x2,
        Type::F32x3 => Opcode::CompositeExtractF32x3,
        Type::F32x4 => Opcode::CompositeExtractF32x4,
        Type::F16x2 => Opcode::CompositeExtractF16x2,
        Type::F16x3 => Opcode::CompositeExtractF16x3,
        Type::F16x4 => Opcode::CompositeExtractF16x4,
        Type::F64x2 => Opcode::CompositeExtractF64x2,
        Type::F64x3 => Opcode::CompositeExtractF64x3,
        Type::F64x4 => Opcode::CompositeExtractF64x4,
        other => unreachable!("{other} is not a float vector"),
    }
}

fn vector_len(ty: Type) -> u32 {
    match ty {
        Type::F32x2 | Type::F16x2 | Type::F64x2 | Type::U32x2 => 2,
        Type::F32x3 | Type::F16x3 | Type::F64x3 | Type::U32x3 => 3,
        Type::F32x4 | Type::F16x4 | Type::F64x4 | Type::U32x4 => 4,
        _ => 1,
    }
}

/// The argument of a composite extract or insert that is the component index (an immediate).
fn index_slot(opcode: Opcode) -> Option<usize> {
    let name = opcode.name();
    if name.starts_with("CompositeExtract") {
        Some(1)
    } else if name.starts_with("CompositeInsert") {
        Some(2)
    } else {
        None
    }
}

/// A one-block control-flow graph of `stmts`, ending in `return`.
fn program(stmts: &[Stmt]) -> Program {
    let mut function = Function::new();
    let entry = function.entry();
    let mut writer = Writer {
        b: Builder::new(&mut function, entry),
        next_out: OUTPUT_BASE,
        outputs: Vec::new(),
    };
    for stmt in stmts {
        writer.stmt(stmt);
    }
    let outputs = writer.outputs;
    writer.b.terminate(Terminator::Return).expect("returns");
    Program { function, outputs }
}

/// A one-statement program of `opcode` with fixed picks.
fn single(opcode: Opcode) -> Program {
    program(&[Stmt {
        opcode,
        picks: [1, 6, 11, 4, 9, 14, 3, 8],
    }])
}

/// Every opcode `pass` renames or writes out that the interpreter can run: all but the
/// packed-half atomics, which neither the interpreter nor the lowering takes.
fn runnable(pass: HostPass) -> Vec<Opcode> {
    Opcode::ALL
        .iter()
        .copied()
        .filter(|&opcode| pass.rewrite(opcode) != Rewrite::Keep)
        .filter(|opcode| !opcode.name().contains("Atomic"))
        .collect()
}

/// The machine after running `function` on `machine`.
fn ran(function: &Function, machine: &Machine) -> Machine {
    let mut machine = machine.clone();
    machine
        .run(function)
        .map_err(|error| format!("{error}\n{function}"))
        .expect("runs");
    machine
}

/// `function` after `pass`, which must succeed.
fn passed(function: &Function, pass: HostPass) -> Function {
    let mut function = function.clone();
    run(&mut function, pass)
        .map_err(|error| format!("{error}\n{function}"))
        .expect("the pass runs");
    function
}

/// How many listed operations or phis name a type `pass` narrows.
fn narrowed_left(function: &Function, pass: HostPass) -> usize {
    function
        .listed()
        .filter(|(_, inst)| match inst {
            Inst::Op(op) => pass.names_narrowed_type(op.opcode),
            Inst::Phi(phi) => pass.narrow_type(phi.ty).is_some(),
        })
        .count()
}

/// How many listed operations have `opcode`.
fn count(function: &Function, opcode: Opcode) -> usize {
    function
        .listed()
        .filter(|(_, inst)| matches!(inst, Inst::Op(op) if op.opcode == opcode))
        .count()
}

/// The id of the first listed operation with `opcode`.
fn first(function: &Function, opcode: Opcode) -> Value {
    function
        .listed()
        .find(|(_, inst)| matches!(inst, Inst::Op(op) if op.opcode == opcode))
        .map(|(value, _)| value)
        .expect("listed")
}

/// A register pair's 64 bits.
fn pair(machine: &Machine, lo: u8) -> u64 {
    u64::from(machine.regs[usize::from(lo)]) | u64::from(machine.regs[usize::from(lo) + 1]) << 32
}

/// Puts `bits` in the register pair at `lo`.
fn set_pair(machine: &mut Machine, lo: u8, bits: u64) {
    let (low, high) = words(bits);
    machine.regs[usize::from(lo)] = low;
    machine.regs[usize::from(lo) + 1] = high;
}

/// The low and high words of `bits`.
fn words(bits: u64) -> (u32, u32) {
    let b = bits.to_le_bytes();
    (
        u32::from_le_bytes([b[0], b[1], b[2], b[3]]),
        u32::from_le_bytes([b[4], b[5], b[6], b[7]]),
    )
}

/// The low and high halves of `word`.
fn halves_of(word: u32) -> (u16, u16) {
    let b = word.to_le_bytes();
    (
        u16::from_le_bytes([b[0], b[1]]),
        u16::from_le_bytes([b[2], b[3]]),
    )
}

/// Rust's own `f64` to `f32` conversion: IEEE 754 round to nearest even, the independent
/// reference the narrowing is checked against.
#[expect(
    clippy::cast_possible_truncation,
    reason = "the cast is the IEEE 754 narrowing the tests compare with, not an accident"
)]
fn ieee_f32(x: f64) -> f32 {
    x as f32
}

/// The distance in binary16 ulps between two halves that are not NaNs.
fn half_ulps(a: u16, b: u16) -> u32 {
    let order = |h: u16| -> i32 {
        let magnitude = i32::from(h & 0x7fff);
        if h & 0x8000 == 0 {
            magnitude
        } else {
            -magnitude
        }
    };
    order(a).abs_diff(order(b))
}

fn is_half_nan(h: u16) -> bool {
    h & 0x7c00 == 0x7c00 && h & 0x03ff != 0
}
