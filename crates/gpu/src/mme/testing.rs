//! Test support: a tiny assembler for MME64 programs and a recording [`MacroHost`].
//!
//! Mesa's simulator tests build their programs with `mme_builder`; the instruction constants in
//! `tests.rs` are what those builder calls lower to, written with these helpers instead, and
//! their results are checked against values worked out independently (Mesa's tests compare the
//! simulator with a real GPU instead, which is not available here).

use std::collections::{BTreeMap, VecDeque};

use super::{Alu, AluOp, DataRam, Instruction, MacroError, MacroHost, Out, OutOp, Reg, StepLimit};

/// Arbitrary method id the test host treats as "start a DMA read of `value` words into the load
/// FIFO at the next barrier" (the real method numbers belong to the 3D engine, not to MME).
pub const DMA_FIFOED_METHOD: u16 = 0x100;

/// Hand-derived lane masks for each predicate mode, in encoding order: `(predicate non-zero,
/// predicate zero)`, bit 0 = ALU 0, bit 1 = ALU 1, bit 2 = output 0, bit 3 = output 1.
/// A `T` lane is off when the predicate is non-zero, an `F` lane is off when it is zero.
pub const PRED_LANES: [(u8, u8); 16] = [
    (0xf, 0xf), // UUUU
    (0x0, 0xf), // TTTT
    (0xf, 0x0), // FFFF
    (0xc, 0xf), // TTUU
    (0xf, 0xc), // FFUU
    (0xe, 0xd), // TFUU
    (0xe, 0xf), // TUUU
    (0xf, 0xe), // FUUU
    (0x3, 0xf), // UUTT
    (0xb, 0x7), // UUTF
    (0xb, 0xf), // UUTU
    (0x7, 0xb), // UUFT
    (0xf, 0x3), // UUFF
    (0xf, 0xb), // UUFU
    (0x7, 0xf), // UUUT
    (0xf, 0x7), // UUUF
];

/// A host that records everything a macro does to it.
#[derive(Default)]
pub struct Host {
    /// The load FIFO.
    pub params: VecDeque<u32>,
    /// When set, `load` never runs dry and returns an incrementing counter.
    pub endless: bool,
    /// How many parameters have been handed out.
    pub loads: u32,
    /// 3D register state: the last value emitted to each method.
    pub regs: BTreeMap<u16, u32>,
    /// Every method call, in order.
    pub calls: Vec<(u16, u32)>,
    /// Every method index `state` was asked for.
    pub state_reads: Vec<u16>,
    /// How many barriers ran.
    pub barriers: u32,
    /// Memory the DMA-read hook reads from.
    pub memory: Vec<u32>,
    pub dma_words: usize,
}

impl Host {
    pub fn with_params(params: &[u32]) -> Self {
        Self {
            params: params.iter().copied().collect(),
            ..Self::default()
        }
    }
}

impl MacroHost for Host {
    fn load(&mut self) -> Option<u32> {
        let value = if self.endless {
            Some(self.loads)
        } else {
            self.params.pop_front()
        };
        if value.is_some() {
            self.loads += 1;
        }
        value
    }

    fn state(&mut self, method: u16) -> u32 {
        self.state_reads.push(method);
        self.regs.get(&method).copied().unwrap_or(0)
    }

    fn method(&mut self, method: u16, value: u32) {
        self.calls.push((method, value));
        self.regs.insert(method, value);
        if method == DMA_FIFOED_METHOD {
            self.dma_words = usize::try_from(value).unwrap_or(0);
        }
    }

    fn barrier(&mut self) {
        self.barriers += 1;
        let words = std::mem::take(&mut self.dma_words);
        self.params.extend(self.memory.iter().take(words));
    }
}

pub const fn r(n: u8) -> Reg {
    Reg::Gpr(n)
}

pub fn alu(op: AluOp, dst: Reg, a: Reg, b: Reg) -> Alu {
    Alu {
        op,
        dst,
        src: [a, b],
    }
}

/// An instruction with only ALU slot 0 in use.
pub fn inst0(op: AluOp, dst: Reg, a: Reg, b: Reg) -> Instruction {
    Instruction {
        alu: [alu(op, dst, a, b), Alu::default()],
        ..Instruction::default()
    }
}

/// An instruction with both ALU slots in use.
pub fn inst01(first: Alu, second: Alu) -> Instruction {
    Instruction {
        alu: [first, second],
        ..Instruction::default()
    }
}

pub fn with_imm(inst: Instruction, imm0: u16, imm1: u16) -> Instruction {
    Instruction {
        imm: [imm0, imm1],
        ..inst
    }
}

pub fn with_out(inst: Instruction, slot: usize, mthd: OutOp, emit: OutOp) -> Instruction {
    let mut inst = inst;
    inst.out[slot] = Out { mthd, emit };
    inst
}

/// `dst = value`, with `value` sign-extended from 16 bits.
pub fn mov(dst: u8, value: u16) -> Instruction {
    with_imm(inst0(AluOp::Add, r(dst), Reg::Imm, Reg::Zero), value, 0)
}

/// `dst = src`.
pub fn mov_reg(dst: u8, src: Reg) -> Instruction {
    inst0(AluOp::Add, r(dst), src, Reg::Zero)
}

/// `dst = a op b` on registers.
pub fn op3(op: AluOp, dst: u8, a: u8, b: u8) -> Instruction {
    inst0(op, r(dst), r(a), r(b))
}

/// `dst += value` (sign-extended 16-bit immediate).
pub fn add_imm(dst: u8, value: u16) -> Instruction {
    with_imm(inst0(AluOp::Add, r(dst), r(dst), Reg::Imm), value, 0)
}

/// Data RAM `[index] = src`, with the index a small immediate.
pub fn store(index: u16, src: u8) -> Instruction {
    with_imm(inst0(AluOp::Dwrite, Reg::Zero, Reg::Imm, r(src)), index, 0)
}

/// `LOOP count`, running the next `body_len` instructions.
pub fn loop_(count: Reg, body_len: u16) -> Instruction {
    with_imm(
        inst0(AluOp::Loop, Reg::Zero, count, Reg::Zero),
        body_len + 1,
        0,
    )
}

/// A branch with the comparison outcome `expect` and a signed 13-bit offset.
pub fn branch(op: AluOp, a: Reg, b: Reg, expect: bool, offset: i16) -> Instruction {
    let imm = (u16::from(expect) << 15) | (offset.cast_unsigned() & 0x1fff);
    with_imm(inst0(op, Reg::Zero, a, b), imm, 0)
}

/// `JAL` forward by `offset` instructions, with the bit 15 the NVIDIA blob always sets.
pub fn jal(offset: u16) -> Instruction {
    with_imm(
        inst0(AluOp::Jal, Reg::Zero, Reg::Zero, Reg::Zero),
        0x8000 | offset,
        0,
    )
}

/// `MERGE dst = (x with y[src_pos..+bits] inserted at dst_pos)`.
pub fn merge(dst: u8, x: u8, y: u8, dst_pos: u16, bits: u16, src_pos: u16) -> Instruction {
    let imm = (dst_pos << 10) | (bits << 5) | src_pos;
    with_imm(op3(AluOp::Merge, dst, x, y), imm, 0)
}

/// Appends what Mesa's `mme_tu104_builder_finish` appends: two nops, the first marking the end.
pub fn finish(mut program: Vec<Instruction>) -> Vec<Instruction> {
    program.push(Instruction {
        end_next: true,
        ..Instruction::default()
    });
    program.push(Instruction::default());
    program
}

/// Runs a finished program with the default step limit.
pub fn run_program(program: &[Instruction], params: &[u32]) -> Result<(DataRam, Host), MacroError> {
    let mut ram = DataRam::new();
    let mut host = Host::with_params(params);
    super::run(program, &mut ram, &mut host, StepLimit::DEFAULT)?;
    Ok((ram, host))
}
