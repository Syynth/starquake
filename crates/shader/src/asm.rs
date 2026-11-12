//! The test assembler: a decoded [`Instr`] back to its instruction word.
//!
//! Ported from `mesa-nouveau/src/nouveau/compiler/nak/sm70_encode.rs` @ 52238762 (MIT,
//! Copyright 2022 Collabora, Ltd.): `encode_alu_base` (operand forms), `impl SM70Op for` `OpMov`,
//! `OpFFma`, `OpALd`, `OpASt`, `OpExit`, `OpS2R`, and `set_pred`. It is the fixture generator for
//! the crate's tests (spec, "Test assembler"): words copied from real shaders cannot be
//! committed. The scheduling bits (105..128, NAK's `set_instr_deps`) are left zero, which the
//! decoder ignores.
//!
//! Exported behind the `test-support` feature for downstream crates' tests.

use thiserror::Error;

use crate::decode::{
    ALD_OPCODE, ALD_OUTPUT, ALD_VTX, AST_DATA, AST_OPCODE, AST_VTX, ATTR_ADDR, ATTR_COMPS,
    ATTR_OFFSET, ATTR_PATCH, ATTR_PHYS, DST, EXIT_KEEP_REF_COUNT, EXIT_NO_AT_EXIT, EXIT_OPCODE,
    EXIT_PRED, EXIT_PRED_NOT, FFMA_DNZ, FFMA_FTZ, FFMA_IMM_SRC1, FFMA_IMM_SRC2, FFMA_REG, FFMA_RND,
    FFMA_SAT, Field, FloatSrc, GUARD, GUARD_NOT, Gpr, Instr, Instruction, MOV_IMM, MOV_QUAD_LANES,
    MOV_REG, MovSrc, OPCODE, Op, OpALd, OpASt, OpExit, OpFFma, OpMov, OpS2R, PredSrc, S2R_OPCODE,
    S2R_SR, SLOT_ABS, SLOT_IMM, SLOT_NEG, SLOT_REG, SRC0, SRC0_ABS, SRC0_NEG, SRC2, SRC2_ABS,
    SRC2_NEG,
};

/// Why an [`Instr`] has no encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum EncodeError {
    /// `FFMA`'s first source must be a register (`encode_alu_src0` takes only registers).
    #[error("the first source of ffma must be a register")]
    ImmediateFirstSource,
    /// `FFMA` has one wide slot, so at most one immediate.
    #[error("ffma can take only one immediate source")]
    TwoImmediates,
    /// A field value does not fit its bits.
    #[error("{what} {value} does not fit in {bits} bits")]
    TooWide {
        /// Which operand.
        what: &'static str,
        /// The value given.
        value: u64,
        /// The field's width.
        bits: u32,
    },
    /// `ALD` and `AST` move one to four words.
    #[error("an attribute access moves 1 to 4 words, not {0}")]
    Comps(u8),
}

/// Encodes `instr`.
pub fn assemble(instr: &Instr) -> Result<Instruction, EncodeError> {
    let word = pred(Instruction::default(), GUARD, GUARD_NOT, instr.guard);
    match instr.op {
        Op::Mov(op) => mov(word, op),
        Op::FFma(op) => ffma(word, op),
        Op::ALd(op) => ald(word, op),
        Op::ASt(op) => ast(word, op),
        Op::Exit(op) => Ok(exit(word, op)),
        Op::S2R(op) => Ok(s2r(word, op)),
    }
}

/// Sets `field` to `value`, which must fit.
fn checked(
    word: Instruction,
    field: Field,
    what: &'static str,
    value: u64,
) -> Result<Instruction, EncodeError> {
    let fits = value.checked_shr(field.width).is_none_or(|rest| rest == 0);
    if fits {
        Ok(word.with_field(field, value))
    } else {
        Err(EncodeError::TooWide {
            what,
            value,
            bits: field.width,
        })
    }
}

fn reg(word: Instruction, field: Field, reg: Gpr) -> Instruction {
    word.with_field(field, u64::from(reg.index()))
}

fn pred(word: Instruction, reg: Field, not: Field, src: PredSrc) -> Instruction {
    word.with_field(reg, u64::from(src.reg.index()))
        .with_flag(not, src.negated)
}

/// `set_alu_reg` for a 32-bit source: the register and its abs and neg bits.
fn float_reg(
    word: Instruction,
    fields: (Field, Field, Field),
    src: FloatSrc,
) -> Result<Instruction, EncodeError> {
    match src {
        FloatSrc::Reg { reg: r, abs, neg } => Ok(reg(word, fields.0, r)
            .with_flag(fields.1, abs)
            .with_flag(fields.2, neg)),
        FloatSrc::Imm(_) => Err(EncodeError::ImmediateFirstSource),
    }
}

fn mov(word: Instruction, op: OpMov) -> Result<Instruction, EncodeError> {
    let word = reg(word, DST, op.dst);
    let word = match op.src {
        MovSrc::Reg(src) => reg(word.with_field(OPCODE, MOV_REG), SLOT_REG, src),
        MovSrc::Imm(imm) => word
            .with_field(OPCODE, MOV_IMM)
            .with_field(SLOT_IMM, u64::from(imm)),
    };
    checked(
        word,
        MOV_QUAD_LANES,
        "quad lane mask",
        u64::from(op.quad_lanes),
    )
}

fn ffma(word: Instruction, op: OpFFma) -> Result<Instruction, EncodeError> {
    const SLOT: (Field, Field, Field) = (SLOT_REG, SLOT_ABS, SLOT_NEG);
    const SOURCE_2: (Field, Field, Field) = (SRC2, SRC2_ABS, SRC2_NEG);
    let [a, b, c] = op.srcs;
    let word = float_reg(word, (SRC0, SRC0_ABS, SRC0_NEG), a)?;
    let word = match (b, c) {
        (FloatSrc::Reg { .. }, FloatSrc::Reg { .. }) => float_reg(
            float_reg(word.with_field(OPCODE, FFMA_REG), SLOT, b)?,
            SOURCE_2,
            c,
        )?,
        (FloatSrc::Imm(imm), FloatSrc::Reg { .. }) => float_reg(
            word.with_field(OPCODE, FFMA_IMM_SRC1)
                .with_field(SLOT_IMM, u64::from(imm)),
            SOURCE_2,
            c,
        )?,
        // Form 2: source 1 goes in source 2's register field.
        (FloatSrc::Reg { .. }, FloatSrc::Imm(imm)) => float_reg(
            word.with_field(OPCODE, FFMA_IMM_SRC2)
                .with_field(SLOT_IMM, u64::from(imm)),
            SOURCE_2,
            b,
        )?,
        (FloatSrc::Imm(_), FloatSrc::Imm(_)) => return Err(EncodeError::TwoImmediates),
    };
    Ok(reg(word, DST, op.dst)
        .with_flag(FFMA_DNZ, op.dnz)
        .with_flag(FFMA_SAT, op.saturate)
        .with_field(FFMA_RND, op.rnd.bits())
        .with_flag(FFMA_FTZ, op.ftz))
}

/// The address and width fields `ALD` and `AST` share.
fn attr(
    word: Instruction,
    offset: Gpr,
    addr: u16,
    comps: u8,
    patch: bool,
    phys: bool,
) -> Result<Instruction, EncodeError> {
    if !(1..=4).contains(&comps) {
        return Err(EncodeError::Comps(comps));
    }
    let word = checked(word, ATTR_ADDR, "attribute address", u64::from(addr))?;
    Ok(reg(word, ATTR_OFFSET, offset)
        .with_field(ATTR_COMPS, u64::from(comps - 1))
        .with_flag(ATTR_PATCH, patch)
        .with_flag(ATTR_PHYS, phys))
}

fn ald(word: Instruction, op: OpALd) -> Result<Instruction, EncodeError> {
    let word = attr(word, op.offset, op.addr, op.comps, op.patch, op.phys)?;
    Ok(reg(
        reg(word.with_field(OPCODE, ALD_OPCODE), DST, op.dst),
        ALD_VTX,
        op.vtx,
    )
    .with_flag(ALD_OUTPUT, op.output))
}

fn ast(word: Instruction, op: OpASt) -> Result<Instruction, EncodeError> {
    let word = attr(word, op.offset, op.addr, op.comps, op.patch, op.phys)?;
    Ok(reg(
        reg(word.with_field(OPCODE, AST_OPCODE), AST_DATA, op.data),
        AST_VTX,
        op.vtx,
    ))
}

fn exit(word: Instruction, op: OpExit) -> Instruction {
    pred(
        word.with_field(OPCODE, EXIT_OPCODE)
            .with_flag(EXIT_KEEP_REF_COUNT, op.keep_ref_count)
            .with_flag(EXIT_NO_AT_EXIT, op.no_at_exit),
        EXIT_PRED,
        EXIT_PRED_NOT,
        op.pred,
    )
}

fn s2r(word: Instruction, op: OpS2R) -> Instruction {
    reg(word.with_field(OPCODE, S2R_OPCODE), DST, op.dst).with_field(S2R_SR, u64::from(op.sr.0))
}

/// Assembles a program: each instruction's word, as the little-endian bytes guest memory
/// holds.
pub fn assemble_program(instrs: &[Instr]) -> Result<Vec<u8>, EncodeError> {
    let mut bytes = Vec::with_capacity(instrs.len().saturating_mul(Instruction::SIZE));
    for instr in instrs {
        bytes.extend_from_slice(&assemble(instr)?.to_le_bytes());
    }
    Ok(bytes)
}
