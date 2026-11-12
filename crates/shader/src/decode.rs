//! The SM86 decoder: one 128-bit instruction word to an [`Instr`].
//!
//! Slice 1 of the spec's port slices decodes six forms: `MOV` with a register or immediate
//! source, `FFMA` with registers or one immediate, `ALD`, `AST`, `EXIT` and `S2R`. The field
//! positions are the inverse of NAK's encoder (`fields`); the forms the decoder does not know
//! (every other opcode, and the constant-buffer and uniform-register forms of `MOV` and `FFMA`)
//! are [`DecodeError::UnknownOpcode`].
//!
//! Decoding is strict (owner's decision 4, 2026-11-12): each form has a mask of the bits it
//! explains, and a word with any other bit set is [`DecodeError::UnknownBit`], naming the lowest
//! such bit, as PR #4588's `FirstUnknownBit` does (`frontend/ampere/isa.h`). The opcode is
//! checked first, then the stray bits. The scheduling bits 105..128 are part of every mask and
//! are not decoded: they encode stalls and dependency barriers, which NAK computes for timing
//! only (`nak/calc_instr_deps.rs`). That they never change a result is a guess (spec,
//! "Instruction layout facts used before full decoding").
//!
//! Divergence from PR #4588's table: `MOV`'s register form rejects bits 62 and 63, which the
//! table marks known (`fields::MOV_REG_KNOWN`); pinned by
//! `tests::mov_register_form_rejects_source_modifier_bits`.

mod fields;
mod instruction;
mod op;
mod print;
#[cfg(test)]
mod tests;

use thiserror::Error;

pub(crate) use fields::*;
pub use instruction::{Field, Instruction, LengthError};
pub use op::{
    FloatSrc, Gpr, Instr, MovSrc, Op, OpALd, OpASt, OpExit, OpFFma, OpMov, OpS2R, Pred, PredSrc,
    RoundMode, SpecialReg,
};

/// Why a word is not a known instruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum DecodeError {
    /// The 12-bit opcode is not one of the forms this decoder knows.
    #[error("unknown opcode {opcode:#05x}")]
    UnknownOpcode {
        /// Bits 0..12.
        opcode: u16,
    },
    /// A bit is set that the opcode's form does not explain.
    #[error("opcode {opcode:#05x} has bit {bit} set, which its form does not use")]
    UnknownBit {
        /// Bits 0..12.
        opcode: u16,
        /// The lowest unexplained set bit.
        bit: u32,
    },
}

/// Decodes one instruction.
pub fn decode(word: Instruction) -> Result<Instr, DecodeError> {
    let raw = word.field(OPCODE);
    // Cannot fail: the field is 12 bits wide.
    let opcode = u16::try_from(raw).unwrap_or(u16::MAX);
    let (known, decode_op): (u128, fn(Instruction, u64) -> Op) = match raw {
        MOV_REG => (MOV_REG_KNOWN, decode_mov),
        MOV_IMM => (MOV_IMM_KNOWN, decode_mov),
        FFMA_REG => (FFMA_REG_KNOWN, decode_ffma),
        FFMA_IMM_SRC2 => (FFMA_IMM_SRC2_KNOWN, decode_ffma),
        FFMA_IMM_SRC1 => (FFMA_IMM_SRC1_KNOWN, decode_ffma),
        ALD_OPCODE => (ALD_KNOWN, decode_attr),
        AST_OPCODE => (AST_KNOWN, decode_attr),
        EXIT_OPCODE => (EXIT_KNOWN, decode_exit),
        S2R_OPCODE => (S2R_KNOWN, decode_s2r),
        _ => return Err(DecodeError::UnknownOpcode { opcode }),
    };
    let stray = word.bits() & !known;
    if stray != 0 {
        return Err(DecodeError::UnknownBit {
            opcode,
            bit: stray.trailing_zeros(),
        });
    }
    Ok(Instr {
        guard: pred_src(word, GUARD, GUARD_NOT),
        op: decode_op(word, raw),
    })
}

fn decode_mov(word: Instruction, raw: u64) -> Op {
    let src = if raw == MOV_REG {
        MovSrc::Reg(gpr(word, SLOT_REG))
    } else {
        MovSrc::Imm(imm32(word))
    };
    Op::Mov(OpMov {
        dst: gpr(word, DST),
        src,
        quad_lanes: byte(word, MOV_QUAD_LANES),
    })
}

fn decode_exit(word: Instruction, _raw: u64) -> Op {
    Op::Exit(OpExit {
        keep_ref_count: word.flag(EXIT_KEEP_REF_COUNT),
        no_at_exit: word.flag(EXIT_NO_AT_EXIT),
        pred: pred_src(word, EXIT_PRED, EXIT_PRED_NOT),
    })
}

fn decode_s2r(word: Instruction, _raw: u64) -> Op {
    Op::S2R(OpS2R {
        dst: gpr(word, DST),
        sr: SpecialReg(byte(word, S2R_SR)),
    })
}

/// Decodes `ALD` or `AST`, which share their address fields.
fn decode_attr(word: Instruction, raw: u64) -> Op {
    match raw {
        ALD_OPCODE => Op::ALd(OpALd {
            dst: gpr(word, DST),
            vtx: gpr(word, ALD_VTX),
            offset: gpr(word, ATTR_OFFSET),
            addr: attr_addr(word),
            comps: attr_comps(word),
            patch: word.flag(ATTR_PATCH),
            phys: word.flag(ATTR_PHYS),
            output: word.flag(ALD_OUTPUT),
        }),
        _ => Op::ASt(OpASt {
            data: gpr(word, AST_DATA),
            vtx: gpr(word, AST_VTX),
            offset: gpr(word, ATTR_OFFSET),
            addr: attr_addr(word),
            comps: attr_comps(word),
            patch: word.flag(ATTR_PATCH),
            phys: word.flag(ATTR_PHYS),
        }),
    }
}

fn decode_ffma(word: Instruction, raw: u64) -> Op {
    let a = float_reg(word, SRC0, SRC0_ABS, SRC0_NEG);
    let c_reg = float_reg(word, SRC2, SRC2_ABS, SRC2_NEG);
    let srcs = if raw == FFMA_IMM_SRC2 {
        // Form 2 moves source 1 into source 2's register field (`encode_alu_base`).
        [a, c_reg, FloatSrc::Imm(imm32(word))]
    } else if raw == FFMA_IMM_SRC1 {
        [a, FloatSrc::Imm(imm32(word)), c_reg]
    } else {
        [a, float_reg(word, SLOT_REG, SLOT_ABS, SLOT_NEG), c_reg]
    };
    Op::FFma(OpFFma {
        dst: gpr(word, DST),
        srcs,
        saturate: word.flag(FFMA_SAT),
        rnd: RoundMode::from_bits(word.field(FFMA_RND)),
        ftz: word.flag(FFMA_FTZ),
        dnz: word.flag(FFMA_DNZ),
    })
}

/// Reads an eight-bit or narrower field. Cannot truncate: every caller's field is at most
/// eight bits wide.
fn byte(word: Instruction, field: Field) -> u8 {
    u8::try_from(word.field(field)).unwrap_or(u8::MAX)
}

fn gpr(word: Instruction, field: Field) -> Gpr {
    Gpr::new(byte(word, field))
}

fn imm32(word: Instruction) -> u32 {
    // Cannot truncate: the field is 32 bits wide.
    u32::try_from(word.field(SLOT_IMM)).unwrap_or(u32::MAX)
}

fn float_reg(word: Instruction, reg: Field, abs: Field, neg: Field) -> FloatSrc {
    FloatSrc::Reg {
        reg: gpr(word, reg),
        abs: word.flag(abs),
        neg: word.flag(neg),
    }
}

fn pred_src(word: Instruction, reg: Field, not: Field) -> PredSrc {
    PredSrc {
        // Cannot fail: the field is three bits wide.
        reg: Pred::new(byte(word, reg)).unwrap_or(Pred::PT),
        negated: word.flag(not),
    }
}

fn attr_addr(word: Instruction) -> u16 {
    // Cannot truncate: the field is ten bits wide.
    u16::try_from(word.field(ATTR_ADDR)).unwrap_or(u16::MAX)
}

fn attr_comps(word: Instruction) -> u8 {
    // The field holds the count minus one (`set_field(74..76, comps - 1)`); two bits, so 1..=4.
    byte(word, ATTR_COMPS).saturating_add(1)
}
