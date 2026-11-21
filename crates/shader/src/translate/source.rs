//! The shared source reader (slice 12i, spec "The translation driver"): how a translator turns an
//! instruction's source operands into IR values. The group translators ([`super::integer`],
//! [`super::float`], [`super::texture`], [`super::control`] and the attribute group) read
//! registers, uniform registers, constant-buffer words and predicates through these functions
//! rather than each keeping its own copy.
//!
//! Each reader follows PR #4588's visitor (`eden-pr4588/src/shader_recompiler/frontend/ampere/`
//! @ e5fb3b97, GPL-3.0-or-later, Copyright 2026 Eden Emulator Project, `visitor.cpp`):
//!
//! - a register through `GetRegister`, `RZ` as the immediate 0 (`Visitor::X`);
//! - a uniform register through `GetUniformRegister`, `URZ` as the immediate 0 (`Visitor::UR`);
//! - a constant-buffer word through `GetCbufU32` at its aligned byte offset, or, for an offset
//!   that is not word aligned, the tail of one word and the head of the next joined by shifts
//!   (`Visitor::Cbuf`);
//! - a predicate through `GetPred`, `PT` as the immediate (`Visitor::Pred`), and a uniform
//!   predicate through `GetUniformPred`, `UPT` as the immediate (`Visitor::UPred`); a negated
//!   one is then `LogicalNot`.
//!
//! The readers append through [`Emit`], so a translator that splits its block lazily for a guard
//! (the integer group's `Emitter`) splits before the first read, as before.

use crate::decode::{AluSrc, CBuf, CBufRef, Gpr, PredSrc, UniformPred, UniformPredSrc, UniformReg};
use crate::ir::{BuildError, Builder, Imm, Opcode, Operand};

/// The bytes of one constant-buffer word: the unit of `GetCbufU32` (`Visitor::Cbuf`:
/// `byte_offset % 4`). Public through [`super::float::CBUF_WORD_BYTES`].
pub const CBUF_WORD_BYTES: u32 = 4;

/// Where a reader appends its operations: a [`Builder`], or a translator's emitter around one.
pub(crate) trait Emit {
    /// Appends `opcode` with `args` and the default flags of its kind, returning its value.
    fn emit_op(&mut self, opcode: Opcode, args: &[Operand]) -> Result<Operand, BuildError>;
}

impl Emit for Builder<'_> {
    fn emit_op(&mut self, opcode: Opcode, args: &[Operand]) -> Result<Operand, BuildError> {
        Ok(self.op(opcode, args)?.into())
    }
}

/// A 32-bit immediate operand.
const fn imm32(bits: u32) -> Operand {
    Operand::Imm(Imm::U32(bits))
}

/// A register's raw bits; `RZ` reads zero (`Visitor::X`).
pub(crate) fn reg(e: &mut impl Emit, reg: Gpr) -> Result<Operand, BuildError> {
    if reg.is_zero() {
        return Ok(imm32(0));
    }
    e.emit_op(Opcode::GetRegister, &[Operand::Imm(Imm::Reg(reg))])
}

/// A uniform register's raw bits; `URZ` reads zero (`Visitor::UR`).
pub(crate) fn uniform_reg(e: &mut impl Emit, reg: UniformReg) -> Result<Operand, BuildError> {
    if reg.is_zero() {
        return Ok(imm32(0));
    }
    e.emit_op(
        Opcode::GetUniformRegister,
        &[Operand::Imm(Imm::UniformReg(reg))],
    )
}

/// The 32 bits at byte `offset` of constant buffer `bank`: one aligned read, or the tail of one
/// word and the head of the next for an unaligned offset (`Visitor::Cbuf`). The second word's
/// offset saturates at `u32::MAX`; the callers pass a 16-bit field, at most plus one word.
pub(crate) fn cbuf_word(e: &mut impl Emit, bank: u8, offset: u32) -> Result<Operand, BuildError> {
    let bank = imm32(u32::from(bank));
    let skew = offset % CBUF_WORD_BYTES;
    let aligned = offset - skew;
    if skew == 0 {
        return e.emit_op(Opcode::GetCbufU32, &[bank, imm32(aligned)]);
    }
    let skew_bits = skew * u8::BITS;
    let low = e.emit_op(Opcode::GetCbufU32, &[bank, imm32(aligned)])?;
    let high = e.emit_op(
        Opcode::GetCbufU32,
        &[bank, imm32(aligned.saturating_add(CBUF_WORD_BYTES))],
    )?;
    let low = e.emit_op(Opcode::ShiftRightLogical32, &[low, imm32(skew_bits)])?;
    let high = e.emit_op(
        Opcode::ShiftLeftLogical32,
        &[high, imm32(u32::BITS - skew_bits)],
    )?;
    e.emit_op(Opcode::BitwiseOr32, &[low, high])
}

/// An ALU source's raw bits, without its modifiers (`Visitor::Value`), or `None` for a
/// constant buffer named by a uniform register, which each group refuses in its own way.
pub(crate) fn alu_bits(e: &mut impl Emit, src: AluSrc) -> Result<Option<Operand>, BuildError> {
    Ok(Some(match src {
        AluSrc::Reg { reg: r, .. } => reg(e, r)?,
        AluSrc::Imm(bits) => imm32(bits),
        AluSrc::UniformReg { reg: r, .. } => uniform_reg(e, r)?,
        AluSrc::CBuf {
            cbuf:
                CBufRef {
                    buf: CBuf::Binding(bank),
                    offset,
                },
            ..
        } => cbuf_word(e, bank, u32::from(offset))?,
        AluSrc::CBuf { .. } => return Ok(None),
    }))
}

/// A predicate source's value; `PT` reads as the immediate (`Visitor::Pred`).
pub(crate) fn pred(e: &mut impl Emit, src: PredSrc) -> Result<Operand, BuildError> {
    if src.reg.is_true() {
        return Ok(Operand::Imm(Imm::U1(!src.negated)));
    }
    let value = e.emit_op(Opcode::GetPred, &[Operand::Imm(Imm::Pred(src.reg))])?;
    if src.negated {
        return e.emit_op(Opcode::LogicalNot, &[value]);
    }
    Ok(value)
}

/// The guard of a uniform datapath instruction (`classify.h`'s `uniform`: `UMOV`, `ULDC`,
/// `ULDC.64`, `S2UR`, `UPLOP3`, `VOTEU` and the uniform ALU forms, each of which has a decode
/// form), read as the uniform predicate of the same number (`Visitor::Run`: `UPred(guard.index,
/// guard.negated)`); `PT` names `UPT`, which reads as the immediate.
pub(crate) fn uniform_guard(e: &mut impl Emit, guard: PredSrc) -> Result<Operand, BuildError> {
    let reg = UniformPred::new(guard.reg.index()).unwrap_or(UniformPred::UPT);
    uniform_pred(
        e,
        UniformPredSrc {
            reg,
            negated: guard.negated,
        },
    )
}

/// A uniform predicate source's value; `UPT` reads as the immediate (`Visitor::UPred`).
pub(crate) fn uniform_pred(e: &mut impl Emit, src: UniformPredSrc) -> Result<Operand, BuildError> {
    if src.reg.is_true() {
        return Ok(Operand::Imm(Imm::U1(!src.negated)));
    }
    let value = e.emit_op(
        Opcode::GetUniformPred,
        &[Operand::Imm(Imm::UniformPred(src.reg))],
    )?;
    if src.negated {
        return e.emit_op(Opcode::LogicalNot, &[value]);
    }
    Ok(value)
}
