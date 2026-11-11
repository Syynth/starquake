//! The Turing MME64 instruction format: types, decoder and encoder.
//!
//! Ported from `mesa-nouveau/src/nouveau/mme/mme_tu104.h` and `mme_tu104.c` @ 52238762 (MIT,
//! Copyright 2022 Collabora Ltd.), cross-checked against `eden-nx2/src/video_core/macro_turing.h`
//! @ 50fa86bb (GPL-3.0-or-later).
//!
//! One instruction is 96 bits (three words) and carries two ALU slots, four output sources (a
//! method selector and an emit source for each of two output slots), two 16-bit immediates and a
//! predicate. Decoding is total: every 96-bit pattern maps to an [`Instruction`] and encodes back
//! to the same words, so a guest program is never rejected up front; malformed encodings (a
//! register selector of 30 or 31, an output selector of 11 to 15) are reported when, and only if,
//! they are executed.

use thiserror::Error;

/// Words per encoded instruction.
pub const INSTRUCTION_WORDS: usize = 3;

/// Number of general registers, `R0` to `R23` (`mme_tu104.h`, `enum mme_tu104_reg`).
pub const GPR_COUNT: u8 = 24;

/// Branch immediate (low 13 bits) that means "stop the macro" instead of a jump
/// (`mme_tu104_sim.c`: `(uint16_t)offset == 0xf000`; `mme_tu104_exit_if` adds `0x1000`).
pub const EXIT_OFFSET: u16 = 0x1000;

/// The signed instruction offset of a conditional branch: the low 13 bits of its immediate,
/// sign-extended (`mme_tu104_sim.c`). Bit 15 (which outcome branches) and bit 14 (ignored) are
/// not part of the offset. The [`EXIT_OFFSET`] marker is not a real offset; callers check
/// [`is_exit_offset`] first.
pub const fn branch_offset(imm: u16) -> i16 {
    ((imm & 0x1fff) << 3).cast_signed() >> 3
}

/// Whether the branch immediate `imm` carries the [`EXIT_OFFSET`] marker in its low 13 bits.
pub const fn is_exit_offset(imm: u16) -> bool {
    imm & 0x1fff == EXIT_OFFSET
}

/// Predicate mode: one letter per lane (ALU 0, ALU 1, output 0, output 1).
///
/// A `T` lane is disabled when the predicate register is non-zero, an `F` lane is disabled when it
/// is zero, and a `U` lane always runs (`mme_tu104_sim.c` `load_pred` and `eden-nx2`
/// `macro_turing.h` `Decode` agree). The mode `UUUU` does not read the predicate register.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Pred {
    /// `UUUU`: every lane always runs.
    #[default]
    Uuuu,
    /// `TTTT`.
    Tttt,
    /// `FFFF`.
    Ffff,
    /// `TTUU`.
    Ttuu,
    /// `FFUU`.
    Ffuu,
    /// `TFUU`.
    Tfuu,
    /// `TUUU`.
    Tuuu,
    /// `FUUU`.
    Fuuu,
    /// `UUTT`.
    Uutt,
    /// `UUTF`.
    Uutf,
    /// `UUTU`.
    Uutu,
    /// `UUFT`.
    Uuft,
    /// `UUFF`.
    Uuff,
    /// `UUFU`.
    Uufu,
    /// `UUUT`.
    Uuut,
    /// `UUUF`.
    Uuuf,
}

impl Pred {
    /// All sixteen modes in encoding order.
    pub const ALL: [Self; 16] = [
        Self::Uuuu,
        Self::Tttt,
        Self::Ffff,
        Self::Ttuu,
        Self::Ffuu,
        Self::Tfuu,
        Self::Tuuu,
        Self::Fuuu,
        Self::Uutt,
        Self::Uutf,
        Self::Uutu,
        Self::Uuft,
        Self::Uuff,
        Self::Uufu,
        Self::Uuut,
        Self::Uuuf,
    ];

    const LETTERS: [&'static str; 16] = [
        "UUUU", "TTTT", "FFFF", "TTUU", "FFUU", "TFUU", "TUUU", "FUUU", "UUTT", "UUTF", "UUTU",
        "UUFT", "UUFF", "UUFU", "UUUT", "UUUF",
    ];

    /// Decodes the 4-bit field; higher bits are ignored.
    pub fn from_bits(bits: u8) -> Self {
        Self::ALL[usize::from(bits & 0xf)]
    }

    /// The 4-bit encoding.
    pub fn bits(self) -> u8 {
        self as u8
    }

    /// The four lane letters, for example `"TTUU"`.
    pub fn letters(self) -> &'static str {
        Self::LETTERS[usize::from(self.bits())]
    }

    /// Which lanes run (bit 0 is ALU 0, bit 3 is output 1) when the predicate register is
    /// non-zero (`value` is true) or zero.
    pub fn lane_mask(self, value: bool) -> u8 {
        let disabled_letter = if value { b'T' } else { b'F' };
        self.letters()
            .bytes()
            .enumerate()
            .filter(|(_, letter)| *letter != disabled_letter)
            .fold(0, |mask, (lane, _)| mask | (1 << lane))
    }
}

/// A source or destination register selector (5 bits).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Reg {
    /// General register `R0` to `R23`.
    Gpr(u8),
    /// Reads as zero; writes are discarded.
    #[default]
    Zero,
    /// This slot's 16-bit immediate, sign-extended.
    Imm,
    /// The other slot's 16-bit immediate, sign-extended.
    ImmPair,
    /// Both immediates as one word: `imm[0]` high, `imm[1]` low.
    Imm32,
    /// The first parameter loaded by this instruction.
    Load0,
    /// The second parameter loaded by this instruction.
    Load1,
    /// A selector no register uses (30 or 31). Executing it is an error.
    Invalid(u8),
}

impl Reg {
    /// Decodes the 5-bit field; higher bits are ignored.
    pub fn from_bits(bits: u8) -> Self {
        match bits & 0x1f {
            n @ 0..=23 => Self::Gpr(n),
            24 => Self::Zero,
            25 => Self::Imm,
            26 => Self::ImmPair,
            27 => Self::Imm32,
            28 => Self::Load0,
            29 => Self::Load1,
            n => Self::Invalid(n),
        }
    }

    /// The selector value.
    pub fn bits(self) -> u8 {
        match self {
            Self::Gpr(n) | Self::Invalid(n) => n,
            Self::Zero => 24,
            Self::Imm => 25,
            Self::ImmPair => 26,
            Self::Imm32 => 27,
            Self::Load0 => 28,
            Self::Load1 => 29,
        }
    }
}

/// An ALU operation (5 bits). Every value is defined.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum AluOp {
    /// `x + y`, sets the carry.
    #[default]
    Add,
    /// `x + y + carry`.
    Addc,
    /// `x - y`, sets the borrow.
    Sub,
    /// `x - y - borrow`.
    Subb,
    /// Signed 32x32 multiply; low half to the result, high half to the carry register.
    Mul,
    /// The high half left by the last multiply.
    Mulh,
    /// Unsigned 32x32 multiply; low half to the result, high half to the carry register.
    Mulu,
    /// Extended operation; only the load barrier (`x == 0x1000`) is known.
    Extended,
    /// Count leading zeros.
    Clz,
    /// Shift left by `y & 31`.
    Sll,
    /// Logical shift right by `y & 31`.
    Srl,
    /// Arithmetic shift right by `y & 31`.
    Sra,
    /// `x & y`.
    And,
    /// `!(x & y)`.
    Nand,
    /// `x | y`.
    Or,
    /// `x ^ y`.
    Xor,
    /// Bitfield insert; the slot's immediate holds `src_pos`, `bits` and `dst_pos`.
    Merge,
    /// Signed `x < y`, as all-ones or zero.
    Slt,
    /// Unsigned `x < y`.
    Sltu,
    /// Signed `x <= y`.
    Sle,
    /// Unsigned `x <= y`.
    Sleu,
    /// `x == y`.
    Seq,
    /// Read a 3D register: method index `x + y`.
    State,
    /// Repeat the following instructions; the slot's immediate is the body length plus one.
    Loop,
    /// Jump forward by the slot's immediate.
    Jal,
    /// Branch if signed `x < y`.
    Blt,
    /// Branch if unsigned `x < y`.
    Bltu,
    /// Branch if signed `x <= y`.
    Ble,
    /// Branch if unsigned `x <= y`.
    Bleu,
    /// Branch if `x == y`.
    Beq,
    /// Read data RAM word `x`.
    Dread,
    /// Write `y` to data RAM word `x`.
    Dwrite,
}

impl AluOp {
    /// All thirty-two operations in encoding order.
    pub const ALL: [Self; 32] = [
        Self::Add,
        Self::Addc,
        Self::Sub,
        Self::Subb,
        Self::Mul,
        Self::Mulh,
        Self::Mulu,
        Self::Extended,
        Self::Clz,
        Self::Sll,
        Self::Srl,
        Self::Sra,
        Self::And,
        Self::Nand,
        Self::Or,
        Self::Xor,
        Self::Merge,
        Self::Slt,
        Self::Sltu,
        Self::Sle,
        Self::Sleu,
        Self::Seq,
        Self::State,
        Self::Loop,
        Self::Jal,
        Self::Blt,
        Self::Bltu,
        Self::Ble,
        Self::Bleu,
        Self::Beq,
        Self::Dread,
        Self::Dwrite,
    ];

    const NAMES: [&'static str; 32] = [
        "ADD", "ADDC", "SUB", "SUBB", "MUL", "MULH", "MULU", "EXTENDED", "CLZ", "SLL", "SRL",
        "SRA", "AND", "NAND", "OR", "XOR", "MERGE", "SLT", "SLTU", "SLE", "SLEU", "SEQ", "STATE",
        "LOOP", "JAL", "BLT", "BLTU", "BLE", "BLEU", "BEQ", "DREAD", "DWRITE",
    ];

    /// Decodes the 5-bit field; higher bits are ignored.
    pub fn from_bits(bits: u8) -> Self {
        Self::ALL[usize::from(bits & 0x1f)]
    }

    /// The 5-bit encoding.
    pub fn bits(self) -> u8 {
        self as u8
    }

    /// The assembler mnemonic, for example `"ADDC"`.
    pub fn name(self) -> &'static str {
        Self::NAMES[usize::from(self.bits())]
    }

    /// Whether the operation changes control flow (`mme_tu104_alu_op_is_control_flow`).
    pub fn is_control_flow(self) -> bool {
        matches!(
            self,
            Self::Loop | Self::Jal | Self::Blt | Self::Bltu | Self::Ble | Self::Bleu | Self::Beq
        )
    }

    /// Whether the operation matters even when its result is discarded
    /// (`mme_tu104_alu_op_has_side_effects`).
    pub fn has_side_effects(self) -> bool {
        self.is_control_flow() || matches!(self, Self::Extended | Self::Dwrite)
    }
}

/// One ALU slot: operation, destination and two sources.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Alu {
    /// The operation.
    pub op: AluOp,
    /// Where the result goes; [`Reg::Zero`] discards it.
    pub dst: Reg,
    /// The two source operands `x` and `y`.
    pub src: [Reg; 2],
}

/// Where an output slot takes its method selector or emitted value from (4 bits; the method
/// selector field is only 3 bits wide, so it cannot name the higher sources).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum OutOp {
    /// Nothing.
    #[default]
    None,
    /// The result of ALU slot 0.
    Alu0,
    /// The result of ALU slot 1.
    Alu1,
    /// The first loaded parameter.
    Load0,
    /// The second loaded parameter.
    Load1,
    /// Immediate 0, zero-extended.
    Imm0,
    /// Immediate 1, zero-extended.
    Imm1,
    /// Reserved; executing it is an error.
    Reserved,
    /// The top four bits of immediate 0.
    ImmHigh0,
    /// The top four bits of immediate 1.
    ImmHigh1,
    /// Both immediates as one word.
    Imm32,
    /// A selector no output uses (11 to 15). Executing it is an error.
    Invalid(u8),
}

impl OutOp {
    /// Decodes the 4-bit field; higher bits are ignored.
    pub fn from_bits(bits: u8) -> Self {
        match bits & 0xf {
            0 => Self::None,
            1 => Self::Alu0,
            2 => Self::Alu1,
            3 => Self::Load0,
            4 => Self::Load1,
            5 => Self::Imm0,
            6 => Self::Imm1,
            7 => Self::Reserved,
            8 => Self::ImmHigh0,
            9 => Self::ImmHigh1,
            10 => Self::Imm32,
            n => Self::Invalid(n),
        }
    }

    /// The selector value.
    pub fn bits(self) -> u8 {
        match self {
            Self::None => 0,
            Self::Alu0 => 1,
            Self::Alu1 => 2,
            Self::Load0 => 3,
            Self::Load1 => 4,
            Self::Imm0 => 5,
            Self::Imm1 => 6,
            Self::Reserved => 7,
            Self::ImmHigh0 => 8,
            Self::ImmHigh1 => 9,
            Self::Imm32 => 10,
            Self::Invalid(n) => n,
        }
    }
}

/// One output slot: an optional method selector, then an optional emit.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Out {
    /// Source of the method selector: bits 0 to 11 are the method index, bits 12 to 15 the
    /// increment applied after each emit.
    pub mthd: OutOp,
    /// Source of the value to send to the current method.
    pub emit: OutOp,
}

/// One decoded 96-bit instruction.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Instruction {
    /// The next instruction is the last one to run.
    pub end_next: bool,
    /// Which lanes the predicate enables.
    pub pred_mode: Pred,
    /// The predicate register.
    pub pred: Reg,
    /// The two 16-bit immediates.
    pub imm: [u16; 2],
    /// The two ALU slots.
    pub alu: [Alu; 2],
    /// The two output slots.
    pub out: [Out; 2],
}

/// Why an instruction could not be encoded.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum EncodeError {
    /// A field value does not round-trip through its bit field (for example `Reg::Gpr(30)`, or
    /// an output source the 3-bit method selector cannot hold).
    #[error("{field} does not fit its {bits}-bit field")]
    FieldOverflow {
        /// The instruction field.
        field: &'static str,
        /// The field width.
        bits: u32,
    },
}

/// Why a program could not be decoded.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum DecodeError {
    /// The program is not a whole number of 3-word instructions.
    #[error("program has {words} words, not a multiple of {INSTRUCTION_WORDS}")]
    TruncatedProgram {
        /// The program length in words.
        words: usize,
    },
}

// Field positions, `mme_tu104_encode` / `mme_tu104_decode` (bit 0 is the least significant bit of
// the 96-bit value). Per ALU slot: op 5 bits, dst 5, src0 5, src1 5, imm 16.
const END_NEXT: u32 = 0;
const PRED_MODE: u32 = 1;
const PRED_REG: u32 = 5;
const ALU_BASE: [u32; 2] = [10, 46];
const ALU_DST: u32 = 5;
const ALU_SRC0: u32 = 10;
const ALU_SRC1: u32 = 15;
const ALU_IMM: u32 = 20;
// Per output slot: method selector 3 bits, emit source 4 bits.
const OUT_BASE: [u32; 2] = [82, 89];
const OUT_EMIT: u32 = 3;

fn bits_of(raw: u128, low: u32, width: u32) -> [u8; 2] {
    let mask = (1_u128 << width) - 1;
    let bytes = ((raw >> low) & mask).to_le_bytes();
    [bytes[0], bytes[1]]
}

fn field8(raw: u128, low: u32, width: u32) -> u8 {
    bits_of(raw, low, width)[0]
}

fn field16(raw: u128, low: u32, width: u32) -> u16 {
    u16::from_le_bytes(bits_of(raw, low, width))
}

impl Instruction {
    /// Decodes one instruction from three words in program-stream order.
    ///
    /// Mesa notes that "the words are reversed in the actual encoding": stream word 0 holds bits
    /// 64 to 95 and word 2 holds bits 0 to 31.
    pub fn decode(words: [u32; INSTRUCTION_WORDS]) -> Self {
        let raw =
            u128::from(words[2]) | (u128::from(words[1]) << 32) | (u128::from(words[0]) << 64);
        let alu = |slot: usize| {
            let base = ALU_BASE[slot];
            Alu {
                op: AluOp::from_bits(field8(raw, base, 5)),
                dst: Reg::from_bits(field8(raw, base + ALU_DST, 5)),
                src: [
                    Reg::from_bits(field8(raw, base + ALU_SRC0, 5)),
                    Reg::from_bits(field8(raw, base + ALU_SRC1, 5)),
                ],
            }
        };
        let out = |slot: usize| {
            let base = OUT_BASE[slot];
            Out {
                mthd: OutOp::from_bits(field8(raw, base, 3)),
                emit: OutOp::from_bits(field8(raw, base + OUT_EMIT, 4)),
            }
        };
        Self {
            end_next: field8(raw, END_NEXT, 1) != 0,
            pred_mode: Pred::from_bits(field8(raw, PRED_MODE, 4)),
            pred: Reg::from_bits(field8(raw, PRED_REG, 5)),
            imm: [
                field16(raw, ALU_BASE[0] + ALU_IMM, 16),
                field16(raw, ALU_BASE[1] + ALU_IMM, 16),
            ],
            alu: [alu(0), alu(1)],
            out: [out(0), out(1)],
        }
    }

    /// Encodes the instruction into three words in program-stream order.
    ///
    /// # Errors
    ///
    /// [`EncodeError::FieldOverflow`] if a register or output selector would not decode back to
    /// the same value.
    pub fn encode(&self) -> Result<[u32; INSTRUCTION_WORDS], EncodeError> {
        let reg = |reg: Reg, field: &'static str| {
            if Reg::from_bits(reg.bits()) == reg {
                Ok(u128::from(reg.bits()))
            } else {
                Err(EncodeError::FieldOverflow { field, bits: 5 })
            }
        };
        let out_op = |op: OutOp, width: u32, field: &'static str| {
            if OutOp::from_bits(op.bits()) == op && u32::from(op.bits()) >> width == 0 {
                Ok(u128::from(op.bits()))
            } else {
                Err(EncodeError::FieldOverflow { field, bits: width })
            }
        };

        let mut raw = u128::from(self.end_next)
            | (u128::from(self.pred_mode.bits()) << PRED_MODE)
            | (reg(self.pred, "pred")? << PRED_REG);
        for slot in 0..2 {
            let base = ALU_BASE[slot];
            let alu = &self.alu[slot];
            raw |= u128::from(alu.op.bits()) << base;
            raw |= reg(alu.dst, "alu.dst")? << (base + ALU_DST);
            raw |= reg(alu.src[0], "alu.src")? << (base + ALU_SRC0);
            raw |= reg(alu.src[1], "alu.src")? << (base + ALU_SRC1);
            raw |= u128::from(self.imm[slot]) << (base + ALU_IMM);

            let base = OUT_BASE[slot];
            let out = &self.out[slot];
            raw |= out_op(out.mthd, 3, "out.mthd")? << base;
            raw |= out_op(out.emit, 4, "out.emit")? << (base + OUT_EMIT);
        }

        let word = |index: u32| {
            let bytes = (raw >> (32 * index)).to_le_bytes();
            u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
        };
        Ok([word(2), word(1), word(0)])
    }
}

/// Decodes a whole program from its words.
///
/// # Errors
///
/// [`DecodeError::TruncatedProgram`] if the length is not a multiple of [`INSTRUCTION_WORDS`].
pub fn decode_program(words: &[u32]) -> Result<Vec<Instruction>, DecodeError> {
    let (chunks, remainder) = words.as_chunks::<INSTRUCTION_WORDS>();
    if !remainder.is_empty() {
        return Err(DecodeError::TruncatedProgram { words: words.len() });
    }
    Ok(chunks
        .iter()
        .map(|chunk| Instruction::decode(*chunk))
        .collect())
}

/// Encodes a whole program into words.
///
/// # Errors
///
/// The first [`EncodeError`] of any instruction.
pub fn encode_program(program: &[Instruction]) -> Result<Vec<u32>, EncodeError> {
    let mut words = Vec::with_capacity(program.len() * INSTRUCTION_WORDS);
    for inst in program {
        words.extend(inst.encode()?);
    }
    Ok(words)
}
