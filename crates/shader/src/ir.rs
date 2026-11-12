//! The intermediate representation, minimal form (slice 1 of the spec's port slices).
//!
//! The owner's decision 1 (2026-11-12): a new Rust IR with Eden's operation set and pass
//! behaviour, stored in arenas and referred to by typed ids. This slice builds the skeleton of
//! that design for straight-line code: a [`Function`] owns an arena of instructions, each
//! result is a [`Value`] id (its index in the arena, in creation order), and blocks list
//! instruction ids. Guest registers are read and written with [`Inst::GetReg`] and
//! [`Inst::SetReg`], as Eden's IR does before SSA (`eden/src/shader_recompiler/frontend/ir/
//! opcodes.inc` @ cf654052: `GetRegister`, `SetRegister`, `GetAttribute`, `SetAttribute`,
//! `FPFma32`, `FPSaturate32`, `FPAbs32`, `FPNeg32`, `BitCastF32U32`, `BitCastU32F32`, `LaneId`,
//! `SetFragColor`). Slice 5 (IR core) widens it: more types and operations, control flow, a
//! parser for the text form and a full verifier.
//!
//! There is one block and no terminator: the function ends where the program's `EXIT` was.
//! Iteration is in id order everywhere, so the same program always prints the same text.

#[cfg(test)]
mod tests;

use std::fmt::{self, Display, Formatter};

use thiserror::Error;

use crate::decode::{Gpr, RoundMode};

/// The id of an instruction and of the value it produces.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Value(u32);

impl Value {
    /// The instruction's index in its function's arena.
    pub const fn index(self) -> u32 {
        self.0
    }
}

impl Display for Value {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "%{}", self.0)
    }
}

/// The id of a block.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BlockId(u32);

/// A value's type (Eden's `frontend/ir/type.h`, the subset slice 1 needs).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Type {
    /// A 32-bit unsigned integer (a guest register's raw bits).
    U32,
    /// A 32-bit float.
    F32,
}

impl Display for Type {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::U32 => "u32",
            Self::F32 => "f32",
        })
    }
}

/// An attribute word, numbered as Eden numbers them: the byte address divided by four
/// (`eden/src/shader_recompiler/frontend/ir/attribute.h`: `PositionX = 28` is address `0x70`,
/// `Generic0X = 32` is address `0x80`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Attribute(pub u16);

impl Display for Attribute {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "a{:#x}", u32::from(self.0) * 4)
    }
}

/// Floating-point controls carried on an operation even where the backend cannot honour them
/// (spec, "The intermediate representation"; decision 10: rounding and flush-to-zero are
/// ignored on arithmetic by the naga backend, a listed divergence). Eden's `FpControl`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FpControl {
    /// The rounding mode.
    pub rounding: RoundMode,
    /// Flush denormals to zero.
    pub ftz: bool,
    /// The multiply-by-zero mode (`FFMA`'s `dnz` bit).
    pub fmz: bool,
}

impl Display for FpControl {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.rounding)?;
        if self.ftz {
            f.write_str(".ftz")?;
        }
        if self.fmz {
            f.write_str(".fmz")?;
        }
        Ok(())
    }
}

/// One instruction.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Inst {
    /// A 32-bit constant: `u32`.
    Imm32(u32),
    /// Reads a register (never `RZ`; the translator turns `RZ` into `Imm32(0)`): `u32`.
    GetReg(Gpr),
    /// Writes a `u32` to a register (never `RZ`).
    SetReg(Gpr, Value),
    /// Reinterprets a `u32` as `f32`.
    BitCastF32(Value),
    /// Reinterprets an `f32` as `u32`.
    BitCastU32(Value),
    /// `|x|`: `f32`.
    FAbs(Value),
    /// `-x`: `f32`.
    FNeg(Value),
    /// `a * b + c`, fused: `f32`.
    FFma(Value, Value, Value, FpControl),
    /// Clamps to [0, 1]: `f32`.
    FSaturate(Value),
    /// Reads an input attribute word as `f32`.
    GetAttribute(Attribute),
    /// Writes an `f32` to an output attribute word.
    SetAttribute(Attribute, Value),
    /// The lane's index in its warp: `u32`.
    LaneId,
    /// Writes an `f32` to a component of a colour target.
    SetFragColor {
        /// The render target, 0 to 7.
        target: u8,
        /// The component, 0 to 3.
        component: u8,
        /// The value.
        value: Value,
    },
}

impl Inst {
    /// The type of the value the instruction produces, if it produces one.
    pub const fn ty(&self) -> Option<Type> {
        match self {
            Self::Imm32(_) | Self::GetReg(_) | Self::BitCastU32(_) | Self::LaneId => {
                Some(Type::U32)
            }
            Self::BitCastF32(_)
            | Self::FAbs(_)
            | Self::FNeg(_)
            | Self::FFma(..)
            | Self::FSaturate(_)
            | Self::GetAttribute(_) => Some(Type::F32),
            Self::SetReg(..) | Self::SetAttribute(..) | Self::SetFragColor { .. } => None,
        }
    }

    /// The operands with the type each must have.
    fn operands(&self) -> Vec<(Value, Type)> {
        match *self {
            Self::Imm32(_) | Self::GetReg(_) | Self::GetAttribute(_) | Self::LaneId => vec![],
            Self::SetReg(_, v) | Self::BitCastF32(v) => vec![(v, Type::U32)],
            Self::BitCastU32(v)
            | Self::FAbs(v)
            | Self::FNeg(v)
            | Self::FSaturate(v)
            | Self::SetAttribute(_, v)
            | Self::SetFragColor { value: v, .. } => vec![(v, Type::F32)],
            Self::FFma(a, b, c, _) => vec![(a, Type::F32), (b, Type::F32), (c, Type::F32)],
        }
    }
}

/// A basic block: instruction ids in execution order.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Block {
    /// The instructions.
    pub insts: Vec<Value>,
}

/// A function: arenas of instructions and blocks.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Function {
    insts: Vec<Inst>,
    blocks: Vec<Block>,
}

/// The IR's limits were exceeded.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
#[error("the function has more than {max} instructions", max = u32::MAX)]
pub struct ArenaFull;

impl Function {
    /// A function with one empty block, the entry.
    pub fn new() -> Self {
        Self {
            insts: Vec::new(),
            blocks: vec![Block::default()],
        }
    }

    /// The entry block.
    pub const fn entry(&self) -> BlockId {
        BlockId(0)
    }

    /// Appends `inst` to the end of `block` and returns its id.
    pub fn push(&mut self, block: BlockId, inst: Inst) -> Result<Value, ArenaFull> {
        let id = Value(u32::try_from(self.insts.len()).map_err(|_| ArenaFull)?);
        let block = usize::try_from(block.0)
            .ok()
            .and_then(|index| self.blocks.get_mut(index))
            .ok_or(ArenaFull)?;
        block.insts.push(id);
        self.insts.push(inst);
        Ok(id)
    }

    /// The instruction `value` names.
    pub fn inst(&self, value: Value) -> Option<&Inst> {
        usize::try_from(value.0)
            .ok()
            .and_then(|index| self.insts.get(index))
    }

    /// The blocks, entry first.
    pub fn blocks(&self) -> &[Block] {
        &self.blocks
    }

    /// Every instruction with its id, in creation order.
    pub fn insts(&self) -> impl Iterator<Item = (Value, &Inst)> {
        (0_u32..).map(Value).zip(self.insts.iter())
    }

    /// Checks that every operand is defined earlier in the same block, with the type the
    /// operation needs, and that no instruction writes or reads `RZ` or appears twice.
    pub fn verify(&self) -> Result<(), VerifyError> {
        let mut defined = vec![false; self.insts.len()];
        for block in &self.blocks {
            for &value in &block.insts {
                let inst = self.inst(value).ok_or(VerifyError::Missing(value))?;
                let slot = usize::try_from(value.0)
                    .ok()
                    .and_then(|index| defined.get_mut(index))
                    .ok_or(VerifyError::Missing(value))?;
                if *slot {
                    return Err(VerifyError::Repeated(value));
                }
                if let Inst::GetReg(reg) | Inst::SetReg(reg, _) = *inst
                    && reg.is_zero()
                {
                    return Err(VerifyError::ZeroRegister(value));
                }
                for (operand, wanted) in inst.operands() {
                    let ready = usize::try_from(operand.0)
                        .ok()
                        .and_then(|index| defined.get(index))
                        .copied()
                        .unwrap_or(false);
                    if !ready {
                        return Err(VerifyError::Undefined { at: value, operand });
                    }
                    let found = self.inst(operand).and_then(Inst::ty);
                    if found != Some(wanted) {
                        return Err(VerifyError::Type {
                            at: value,
                            operand,
                            wanted,
                        });
                    }
                }
                if let Some(slot) = usize::try_from(value.0)
                    .ok()
                    .and_then(|index| defined.get_mut(index))
                {
                    *slot = true;
                }
            }
        }
        Ok(())
    }
}

/// Why [`Function::verify`] failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum VerifyError {
    /// A block lists an id with no instruction.
    #[error("{0} is listed in a block but does not exist")]
    Missing(Value),
    /// An id is listed twice.
    #[error("{0} is listed twice")]
    Repeated(Value),
    /// An instruction reads or writes `RZ`.
    #[error("{0} reads or writes rz")]
    ZeroRegister(Value),
    /// An operand is not defined before its use.
    #[error("{at} uses {operand} before it is defined")]
    Undefined {
        /// The instruction.
        at: Value,
        /// The operand.
        operand: Value,
    },
    /// An operand has the wrong type.
    #[error("{at} needs {operand} to be {wanted}")]
    Type {
        /// The instruction.
        at: Value,
        /// The operand.
        operand: Value,
        /// The type it needs.
        wanted: Type,
    },
}

impl Display for Inst {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match *self {
            Self::Imm32(bits) => write!(f, "imm32 {bits:#x}"),
            Self::GetReg(reg) => write!(f, "get_reg {reg}"),
            Self::SetReg(reg, value) => write!(f, "set_reg {reg}, {value}"),
            Self::BitCastF32(value) => write!(f, "bitcast_f32 {value}"),
            Self::BitCastU32(value) => write!(f, "bitcast_u32 {value}"),
            Self::FAbs(value) => write!(f, "fabs {value}"),
            Self::FNeg(value) => write!(f, "fneg {value}"),
            Self::FFma(a, b, c, control) => write!(f, "ffma{control} {a}, {b}, {c}"),
            Self::FSaturate(value) => write!(f, "fsaturate {value}"),
            Self::GetAttribute(attribute) => write!(f, "get_attribute {attribute}"),
            Self::SetAttribute(attribute, value) => {
                write!(f, "set_attribute {attribute}, {value}")
            }
            Self::LaneId => f.write_str("lane_id"),
            Self::SetFragColor {
                target,
                component,
                value,
            } => write!(f, "set_frag_color {target}.{component}, {value}"),
        }
    }
}

impl Display for Function {
    /// The text form: one line per block label and per instruction, `%n = op.type args` for
    /// instructions with a result.
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        for (index, block) in self.blocks.iter().enumerate() {
            writeln!(f, "block{index}:")?;
            for &value in &block.insts {
                let Some(inst) = self.inst(value) else {
                    writeln!(f, "    {value} = <missing>")?;
                    continue;
                };
                match inst.ty() {
                    Some(ty) => writeln!(f, "    {value}: {ty} = {inst}")?,
                    None => writeln!(f, "    {inst}")?,
                }
            }
        }
        Ok(())
    }
}
