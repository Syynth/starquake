//! The Turing MME64 interpreter: runs a decoded macro against a [`MacroHost`] and a [`DataRam`].
//!
//! Ported from `mesa-nouveau/src/nouveau/mme/mme_tu104_sim.c` and `mme_sim.h` @ 52238762 (MIT,
//! Copyright 2022 Collabora Ltd.), cross-checked against `eden-nx2/src/video_core/macro_turing.h`
//! @ 50fa86bb (GPL-3.0-or-later). Where the two disagree this follows Mesa, which was checked
//! against hardware:
//!
//! - `LOOP` with a count of zero runs the body once (`MAX2(1, x) - 1` extra passes); Eden-NX2
//!   skips the body, citing macros the native SDK emits around FIFO reads. Mesa's hardware tests
//!   cannot tell the two apart (their loop body adds the count), so this is an open question
//!   for a real title.
//! - The second ALU slot sees the first slot's register write; Eden-NX2 reads both slots'
//!   operands before committing either. Programs from the Mesa builder never depend on the
//!   difference (`mme_tu104_alus_have_dependency` keeps dependent ALU ops in separate instructions).
//! - Bit 14 of a branch immediate is ignored; Eden-NX2 treats it as "taking this back edge
//!   cancels a pending end".
//! - `STATE` masks the method index to 14 bits (`(uint16_t)(x + y) * 4` stored in a 16-bit
//!   method address); Eden-NX2 masks to 16 bits.
//!
//! Guest programs are untrusted: every failure Mesa asserts on is an [`MacroError`] here, and a
//! [`StepLimit`] bounds the number of instructions executed.

use thiserror::Error;

use super::inst::{AluOp, GPR_COUNT, Instruction, OutOp, Pred, Reg};

/// Words of macro data RAM (`MME_TU104_DRAM_COUNT` in `mme_tu104.h`).
pub const DATA_RAM_WORDS: usize = 0xc00;

/// The extended-operation operand that means "load barrier" (`mme_tu104_load_barrier` in
/// `mme_tu104_builder.c`; `eval_extended` in `mme_tu104_sim.c`). Mesa also passes `y == 1`;
/// Eden-NX2 ignores `y`, and so does this.
pub const LOAD_BARRIER: u32 = 0x1000;

/// Branch immediate (low 13 bits) that means "stop the macro" instead of a jump
/// (`mme_tu104_sim.c`: `(uint16_t)offset == 0xf000`; `mme_tu104_exit_if` adds `0x1000`).
const EXIT_OFFSET: u16 = 0x1000;

/// Method indices are dwords (byte address / 4) and wrap at 14 bits, because Mesa keeps the
/// byte address in 16 bits.
pub const METHOD_MASK: u16 = 0x3fff;

/// The macro data RAM that `DREAD` and `DWRITE` access.
///
/// Each GPU channel has its own (Eden-NX2 notes, row 84), so a run borrows one rather than
/// owning it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DataRam {
    words: Box<[u32]>,
}

impl DataRam {
    /// A zeroed data RAM of [`DATA_RAM_WORDS`] words.
    pub fn new() -> Self {
        Self {
            words: vec![0; DATA_RAM_WORDS].into_boxed_slice(),
        }
    }

    /// The word at `index`, or `None` past the end.
    pub fn get(&self, index: u32) -> Option<u32> {
        self.words.get(usize::try_from(index).ok()?).copied()
    }

    /// Mutable access to the word at `index`, or `None` past the end.
    pub fn get_mut(&mut self, index: u32) -> Option<&mut u32> {
        self.words.get_mut(usize::try_from(index).ok()?)
    }

    /// All words, for hosts that implement DMA into data RAM.
    pub fn as_slice(&self) -> &[u32] {
        &self.words
    }

    /// All words, mutable.
    pub fn as_mut_slice(&mut self) -> &mut [u32] {
        &mut self.words
    }
}

impl Default for DataRam {
    fn default() -> Self {
        Self::new()
    }
}

/// What a running macro needs from the rest of the GPU front: Mesa's `mme_sim_state_ops`
/// (`load`, `state`, `mthd`, `barrier`; `map_dram` is the [`DataRam`] argument of [`run`]).
///
/// Method numbers are dword indices (byte address / 4).
pub trait MacroHost {
    /// The next parameter from the load FIFO, or `None` when it is exhausted (the run then fails
    /// with [`MacroError::ParameterUnderflow`]).
    fn load(&mut self) -> Option<u32>;

    /// The current value of the 3D register at `method`.
    fn state(&mut self, method: u16) -> u32;

    /// A method call emitted by the macro.
    fn method(&mut self, method: u16, value: u32);

    /// The load barrier (`EXTENDED` with `x == 0x1000`).
    ///
    /// Extension hook for MME DMA reads (`MME_DMA_READ_FIFOED`, Eden-NX2 notes, row 58): a host
    /// that supports them appends the words it fetched to its load FIFO here, so that later
    /// `LOAD0`/`LOAD1` reads see them. The default does nothing, which is correct for a host
    /// without DMA reads.
    fn barrier(&mut self) {}
}

/// The most instructions one run may execute.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StepLimit(u32);

impl StepLimit {
    /// Eden-NX2's limit (`macro_turing.h`: `steps < 1000000`), generous for real macros.
    pub const DEFAULT: Self = Self(1_000_000);

    /// A limit of `steps` instructions.
    pub const fn new(steps: u32) -> Self {
        Self(steps)
    }

    /// The number of instructions.
    pub const fn get(self) -> u32 {
        self.0
    }
}

impl Default for StepLimit {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// Why a macro run failed. `pc` is the index of the instruction that failed.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum MacroError {
    /// The macro ran for the whole step limit without finishing.
    #[error("macro exceeded its limit of {limit} steps")]
    StepLimitExceeded {
        /// The limit that was hit.
        limit: u32,
    },
    /// Execution reached an instruction index past the end of the program.
    #[error("instruction {pc} is outside the program")]
    PcOutOfRange {
        /// The index.
        pc: usize,
    },
    /// An instruction asked for a parameter and the load FIFO was empty.
    #[error("parameter FIFO underflow at instruction {pc}")]
    ParameterUnderflow {
        /// The instruction.
        pc: usize,
    },
    /// A general register was read before anything wrote it.
    #[error("instruction {pc} reads uninitialized register R{reg}")]
    UninitializedRegister {
        /// The instruction.
        pc: usize,
        /// The register number.
        reg: u8,
    },
    /// A register selector that cannot be read or written here.
    #[error("instruction {pc} uses invalid register selector {bits}")]
    InvalidRegister {
        /// The instruction.
        pc: usize,
        /// The selector.
        bits: u8,
    },
    /// An output source that does not exist.
    #[error("instruction {pc} uses invalid output source {bits}")]
    InvalidOutput {
        /// The instruction.
        pc: usize,
        /// The selector.
        bits: u8,
    },
    /// An `EXTENDED` operation other than the load barrier.
    #[error("instruction {pc} uses unsupported extended operation {x:#x}")]
    UnsupportedExtended {
        /// The instruction.
        pc: usize,
        /// The operand.
        x: u32,
    },
    /// A `MERGE` whose destination bit position is 32 or more.
    #[error("instruction {pc} merges into bit {dst_pos}")]
    InvalidMerge {
        /// The instruction.
        pc: usize,
        /// The destination bit position.
        dst_pos: u16,
    },
    /// A `LOOP` whose length field is below 2.
    #[error("instruction {pc} starts a loop with length field {length}")]
    InvalidLoop {
        /// The instruction.
        pc: usize,
        /// The length field.
        length: u16,
    },
    /// A `LOOP` inside a running loop.
    #[error("instruction {pc} starts a loop inside a loop")]
    NestedLoop {
        /// The instruction.
        pc: usize,
    },
    /// A jump or branch leaving the program.
    #[error("instruction {pc} jumps by {offset} out of the program")]
    BranchOutOfRange {
        /// The instruction.
        pc: usize,
        /// The signed offset.
        offset: i32,
    },
    /// `DREAD` or `DWRITE` outside data RAM.
    #[error("instruction {pc} accesses data RAM word {index:#x}")]
    DataRamOutOfBounds {
        /// The instruction.
        pc: usize,
        /// The word index.
        index: u32,
    },
}

fn signed(value: u32) -> i32 {
    value.cast_signed()
}

fn low16(value: u32) -> u16 {
    let bytes = value.to_le_bytes();
    u16::from_le_bytes([bytes[0], bytes[1]])
}

fn low8(value: u32) -> u8 {
    value.to_le_bytes()[0]
}

fn split64(value: u64) -> (u32, u32) {
    let bytes = value.to_le_bytes();
    (
        u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
        u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
    )
}

/// Sign-extends a 16-bit immediate the way ALU operands do.
fn sext(imm: u16) -> u32 {
    i32::from(imm.cast_signed()).cast_unsigned()
}

fn imm32(imm: [u16; 2]) -> u32 {
    (u32::from(imm[0]) << 16) | u32::from(imm[1])
}

fn compare(op: AluOp, x: u32, y: u32) -> bool {
    match op {
        AluOp::Blt | AluOp::Slt => signed(x) < signed(y),
        AluOp::Bltu | AluOp::Sltu => x < y,
        AluOp::Ble | AluOp::Sle => signed(x) <= signed(y),
        AluOp::Bleu | AluOp::Sleu => x <= y,
        _ => x == y,
    }
}

struct Machine<'a, H> {
    program: &'a [Instruction],
    ram: &'a mut DataRam,
    host: &'a mut H,
    /// Index of the executing instruction.
    ip: usize,
    next_ip: usize,
    stop: bool,
    load: [u32; 2],
    regs: [u32; GPR_COUNT as usize],
    /// Bit `n` is set once `R<n>` has been written.
    written: u32,
    alu_res: [u32; 2],
    /// Carry or borrow of the last add or subtract, high half of the last multiply.
    carry: u32,
    /// Current method index and the increment applied after each emit.
    method: u16,
    increment: u16,
    loop_count: u32,
    loop_start: usize,
    loop_end: usize,
}

impl<H: MacroHost> Machine<'_, H> {
    fn read_reg(
        &self,
        inst: &Instruction,
        slot: Option<usize>,
        reg: Reg,
    ) -> Result<u32, MacroError> {
        let pc = self.ip;
        match (reg, slot) {
            (Reg::Gpr(n), _) if n < GPR_COUNT => {
                if self.written & (1 << n) == 0 {
                    return Err(MacroError::UninitializedRegister { pc, reg: n });
                }
                Ok(self.regs[usize::from(n)])
            }
            (Reg::Zero, _) => Ok(0),
            // Immediates are sign-extended for ALU operands. The predicate register has no slot.
            (Reg::Imm, Some(slot)) => Ok(sext(inst.imm[slot])),
            (Reg::ImmPair, Some(slot)) => Ok(sext(inst.imm[1 - slot])),
            (Reg::Imm32, _) => Ok(imm32(inst.imm)),
            (Reg::Load0, _) => Ok(self.load[0]),
            (Reg::Load1, _) => Ok(self.load[1]),
            (reg, _) => Err(MacroError::InvalidRegister {
                pc,
                bits: reg.bits(),
            }),
        }
    }

    fn write_reg(&mut self, reg: Reg, value: u32) -> Result<(), MacroError> {
        match reg {
            Reg::Gpr(n) if n < GPR_COUNT => {
                self.written |= 1 << n;
                self.regs[usize::from(n)] = value;
                Ok(())
            }
            Reg::Zero => Ok(()),
            reg => Err(MacroError::InvalidRegister {
                pc: self.ip,
                bits: reg.bits(),
            }),
        }
    }

    /// Fetches parameters for the `LOAD0`/`LOAD1` an instruction mentions, in that order, before
    /// anything else in the instruction runs (`load_params`).
    fn load_params(&mut self, inst: &Instruction) -> Result<(), MacroError> {
        for index in 0..2 {
            let (reg, out) = if index == 0 {
                (Reg::Load0, OutOp::Load0)
            } else {
                (Reg::Load1, OutOp::Load1)
            };
            let used = inst.pred == reg
                || inst.alu.iter().any(|alu| alu.src.contains(&reg))
                || inst.out.iter().any(|o| o.mthd == out || o.emit == out);
            if used {
                self.load[index] = self
                    .host
                    .load()
                    .ok_or(MacroError::ParameterUnderflow { pc: self.ip })?;
            }
        }
        Ok(())
    }

    fn branch_to(&mut self, offset: i32) -> Result<(), MacroError> {
        let pc = self.ip;
        let target = isize::try_from(offset)
            .ok()
            .and_then(|offset| pc.checked_add_signed(offset))
            .filter(|target| *target < self.program.len())
            .ok_or(MacroError::BranchOutOfRange { pc, offset })?;
        self.next_ip = target;
        Ok(())
    }

    fn eval_alu(&mut self, inst: &Instruction, slot: usize) -> Result<(), MacroError> {
        let pc = self.ip;
        let alu = inst.alu[slot];
        let imm = inst.imm[slot];
        let x = self.read_reg(inst, Some(slot), alu.src[0])?;
        let y = self.read_reg(inst, Some(slot), alu.src[1])?;

        let mut res = 0;
        match alu.op {
            AluOp::Add => {
                res = x.wrapping_add(y);
                self.carry = u32::from(res < x);
            }
            AluOp::Addc => res = x.wrapping_add(y).wrapping_add(self.carry),
            AluOp::Sub => {
                res = x.wrapping_sub(y);
                self.carry = u32::from(res > x);
            }
            AluOp::Subb => res = x.wrapping_sub(y).wrapping_sub(self.carry),
            AluOp::Mul => {
                let product = i64::from(signed(x)) * i64::from(signed(y));
                (res, self.carry) = split64(product.cast_unsigned());
            }
            AluOp::Mulh => res = self.carry,
            AluOp::Mulu => {
                (res, self.carry) = split64(u64::from(x) * u64::from(y));
            }
            AluOp::Extended => {
                if x != LOAD_BARRIER {
                    return Err(MacroError::UnsupportedExtended { pc, x });
                }
                self.host.barrier();
            }
            AluOp::Clz => res = x.leading_zeros(),
            AluOp::Sll => res = x << (y & 31),
            AluOp::Srl => res = x >> (y & 31),
            AluOp::Sra => res = (signed(x) >> (y & 31)).cast_unsigned(),
            AluOp::And => res = x & y,
            AluOp::Nand => res = !(x & y),
            AluOp::Or => res = x | y,
            AluOp::Xor => res = x ^ y,
            AluOp::Merge => {
                // Immediate: bits 0..5 source position, 5..10 width, 10..16 destination position.
                let src_pos = u32::from(imm & 0x1f);
                let width = u32::from((imm >> 5) & 0x1f);
                let dst_pos = (imm >> 10) & 0x3f;
                if dst_pos >= 32 {
                    return Err(MacroError::InvalidMerge { pc, dst_pos });
                }
                let mask = (1_u32 << width) - 1;
                res = (x & !(mask << dst_pos)) | (((y >> src_pos) & mask) << dst_pos);
            }
            AluOp::Slt | AluOp::Sltu | AluOp::Sle | AluOp::Sleu | AluOp::Seq => {
                res = if compare(alu.op, x, y) { u32::MAX } else { 0 };
            }
            AluOp::State => {
                res = self.host.state(low16(x.wrapping_add(y)) & METHOD_MASK);
            }
            AluOp::Loop => {
                if self.loop_count != 0 {
                    return Err(MacroError::NestedLoop { pc });
                }
                if imm < 2 {
                    return Err(MacroError::InvalidLoop { pc, length: imm });
                }
                // Mesa: a count of zero still runs the body once (see the module docs).
                self.loop_count = x.max(1) - 1;
                self.loop_start = pc;
                self.loop_end = pc + usize::from(imm) - 1;
            }
            AluOp::Jal => {
                // Mesa asserts bit 15 is set ("no idea what bit 15 does"); it is ignored here.
                self.branch_to(i32::from(imm & 0x7fff))?;
            }
            AluOp::Blt | AluOp::Bltu | AluOp::Ble | AluOp::Bleu | AluOp::Beq => {
                // Bit 15 says which outcome of the comparison takes the branch; the low 13 bits
                // are a signed offset.
                if compare(alu.op, x, y) == (imm & 0x8000 != 0) {
                    let raw = imm & 0x1fff;
                    if raw == EXIT_OFFSET {
                        self.stop = true;
                    } else {
                        let offset = (raw << 3).cast_signed() >> 3;
                        self.branch_to(i32::from(offset))?;
                    }
                }
            }
            AluOp::Dread => {
                res = self
                    .ram
                    .get(x)
                    .ok_or(MacroError::DataRamOutOfBounds { pc, index: x })?;
            }
            AluOp::Dwrite => {
                *self
                    .ram
                    .get_mut(x)
                    .ok_or(MacroError::DataRamOutOfBounds { pc, index: x })? = y;
            }
        }

        self.alu_res[slot] = res;
        self.write_reg(alu.dst, res)
    }

    fn out_value(&self, inst: &Instruction, op: OutOp) -> Result<u32, MacroError> {
        match op {
            OutOp::Alu0 => Ok(self.alu_res[0]),
            OutOp::Alu1 => Ok(self.alu_res[1]),
            OutOp::Load0 => Ok(self.load[0]),
            OutOp::Load1 => Ok(self.load[1]),
            OutOp::Imm0 => Ok(u32::from(inst.imm[0])),
            OutOp::Imm1 => Ok(u32::from(inst.imm[1])),
            OutOp::ImmHigh0 => Ok(u32::from(inst.imm[0] >> 12)),
            OutOp::ImmHigh1 => Ok(u32::from(inst.imm[1] >> 12)),
            OutOp::Imm32 => Ok(imm32(inst.imm)),
            op => Err(MacroError::InvalidOutput {
                pc: self.ip,
                bits: op.bits(),
            }),
        }
    }

    fn eval_out(&mut self, inst: &Instruction, slot: usize) -> Result<(), MacroError> {
        let out = inst.out[slot];
        if out.mthd != OutOp::None {
            let data = self.out_value(inst, out.mthd)?;
            self.method = low16(data & 0xfff);
            self.increment = u16::from(low8((data >> 12) & 0xf));
        }
        if out.emit != OutOp::None {
            let data = self.out_value(inst, out.emit)?;
            self.host.method(self.method, data);
            self.method = self.method.wrapping_add(self.increment) & METHOD_MASK;
        }
        Ok(())
    }

    /// Runs one instruction; returns whether the macro is finished.
    fn step(&mut self, end_next: &mut bool) -> Result<bool, MacroError> {
        let inst = *self
            .program
            .get(self.ip)
            .ok_or(MacroError::PcOutOfRange { pc: self.ip })?;
        self.next_ip = self.ip + 1;

        self.load_params(&inst)?;

        let lanes = if inst.pred_mode == Pred::Uuuu {
            0xf
        } else {
            let value = self.read_reg(&inst, None, inst.pred)?;
            inst.pred_mode.lane_mask(value != 0)
        };
        for slot in 0..2 {
            if lanes & (1 << slot) != 0 {
                self.eval_alu(&inst, slot)?;
            }
        }
        for slot in 0..2 {
            if lanes & (4 << slot) != 0 {
                self.eval_out(&inst, slot)?;
            }
        }

        if *end_next || self.stop {
            return Ok(true);
        }
        *end_next = inst.end_next;

        if self.loop_count > 0 && self.ip == self.loop_end {
            self.loop_count -= 1;
            self.next_ip = self.loop_start + 1;
        }
        self.ip = self.next_ip;
        Ok(false)
    }
}

/// Runs `program` from its first instruction until it ends or fails.
///
/// A macro ends one instruction after one with `end_next` set, or immediately after a branch
/// with the exit offset. `data_ram` persists across calls (it belongs to the GPU channel); the
/// registers do not.
///
/// # Errors
///
/// A [`MacroError`] for any malformed instruction or out-of-range access that is reached, and
/// [`MacroError::StepLimitExceeded`] after `limit` instructions. The host and data RAM keep
/// whatever effects happened before the failure.
pub fn run(
    program: &[Instruction],
    data_ram: &mut DataRam,
    host: &mut impl MacroHost,
    limit: StepLimit,
) -> Result<(), MacroError> {
    let mut machine = Machine {
        program,
        ram: data_ram,
        host,
        ip: 0,
        next_ip: 0,
        stop: false,
        load: [0; 2],
        regs: [0; GPR_COUNT as usize],
        written: 0,
        alu_res: [0; 2],
        carry: 0,
        method: 0,
        increment: 0,
        loop_count: 0,
        loop_start: 0,
        loop_end: 0,
    };
    let mut end_next = false;
    for _ in 0..limit.get() {
        if machine.step(&mut end_next)? {
            return Ok(());
        }
    }
    Err(MacroError::StepLimitExceeded { limit: limit.get() })
}
