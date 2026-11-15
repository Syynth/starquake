//! Operands: values and immediates, and the names immediates carry (registers, predicates,
//! attributes, patches).
//!
//! An operand is either the result of an instruction ([`Value`]) or an immediate, as in Eden's
//! `IR::Value` (`eden/src/shader_recompiler/frontend/ir/value.h` @ cf654052, GPL-3.0-or-later),
//! which holds an instruction pointer or one of the immediate kinds below. Floats are kept as
//! their bits, so equal immediates compare equal and print exactly.

use std::fmt::{self, Display, Formatter};

use super::{Type, Value};
use crate::decode::{Gpr, Pred};

/// A uniform register, `UR0` to `UR62`, or `URZ`, which reads zero.
///
/// SM86 has 63 uniform registers and `URZ` is number 63: NAK encodes the uniform zero register
/// as `ugpr_max()`, 63 below SM 10.0 (`mesa-nouveau/src/nouveau/compiler/nak/sm70_encode.rs` @
/// 52238762, `zero_reg`), and gives SM 7.x and 8.x 63 allocatable uniform registers
/// (`sm70.rs`, `num_regs`: `RegFile::UGPR => 63`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct UniformReg(u8);

impl UniformReg {
    /// `URZ`.
    pub const URZ: Self = Self(63);

    /// Uniform register `index`, if it exists (`URZ` included).
    pub const fn new(index: u8) -> Option<Self> {
        if index <= Self::URZ.0 {
            Some(Self(index))
        } else {
            None
        }
    }

    /// The register's number.
    pub const fn index(self) -> u8 {
        self.0
    }

    /// Whether this is `URZ`.
    pub const fn is_zero(self) -> bool {
        self.0 == Self::URZ.0
    }
}

impl Display for UniformReg {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        if self.is_zero() {
            f.write_str("urz")
        } else {
            write!(f, "ur{}", self.0)
        }
    }
}

/// A uniform predicate, `UP0` to `UP6`, or `UPT`, which reads true.
///
/// NAK's true register is number 7 in every predicate file (`sm70_encode.rs`, `true_reg`), and
/// SM 7.x and 8.x have seven allocatable uniform predicates (`sm70.rs`, `num_regs`:
/// `RegFile::UPred => 7`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct UniformPred(u8);

impl UniformPred {
    /// `UPT`.
    pub const UPT: Self = Self(7);

    /// Uniform predicate `index`, if it exists (`UPT` included).
    pub const fn new(index: u8) -> Option<Self> {
        if index <= Self::UPT.0 {
            Some(Self(index))
        } else {
            None
        }
    }

    /// The predicate's number.
    pub const fn index(self) -> u8 {
        self.0
    }

    /// Whether this is `UPT`.
    pub const fn is_true(self) -> bool {
        self.0 == Self::UPT.0
    }
}

impl Display for UniformPred {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        if self.is_true() {
            f.write_str("upt")
        } else {
            write!(f, "up{}", self.0)
        }
    }
}

/// An attribute word, numbered as Eden numbers them: the byte address divided by four
/// (`eden/src/shader_recompiler/frontend/ir/attribute.h`: `PositionX = 28` is address `0x70`,
/// `Generic0X = 32` is address `0x80`). The text form writes the byte address.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Attribute(pub u16);

/// Bytes per attribute word.
pub const ATTRIBUTE_WORD_BYTES: u32 = 4;

impl Display for Attribute {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "a{:#x}", u32::from(self.0) * ATTRIBUTE_WORD_BYTES)
    }
}

/// A tessellation patch word: the six level-of-detail words, then 120 components
/// (`eden/src/shader_recompiler/frontend/ir/patch.h`: `TessellationLodLeft` to
/// `TessellationLodInteriorV`, then `Component0` to `Component119`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Patch(u8);

impl Patch {
    /// How many patch words there are (`patch.h`: 6 level-of-detail words and 120 components).
    pub const COUNT: u8 = 126;

    /// Patch word `index`, if it exists.
    pub const fn new(index: u8) -> Option<Self> {
        if index < Self::COUNT {
            Some(Self(index))
        } else {
            None
        }
    }

    /// The word's number.
    pub const fn index(self) -> u8 {
        self.0
    }
}

impl Display for Patch {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "patch{}", self.0)
    }
}

/// An immediate (Eden's immediate `IR::Value` kinds, plus the uniform register class).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Imm {
    /// A per-thread register name.
    Reg(Gpr),
    /// A predicate name.
    Pred(Pred),
    /// A uniform register name.
    UniformReg(UniformReg),
    /// A uniform predicate name.
    UniformPred(UniformPred),
    /// An attribute word.
    Attribute(Attribute),
    /// A patch word.
    Patch(Patch),
    /// A boolean.
    U1(bool),
    /// An 8-bit integer.
    U8(u8),
    /// A 16-bit integer.
    U16(u16),
    /// A 32-bit integer.
    U32(u32),
    /// A 64-bit integer.
    U64(u64),
    /// A 32-bit float, as its bits.
    F32(u32),
    /// A 64-bit float, as its bits.
    F64(u64),
}

impl Imm {
    /// The immediate's type.
    pub const fn ty(self) -> Type {
        match self {
            Self::Reg(_) => Type::Reg,
            Self::Pred(_) => Type::Pred,
            Self::UniformReg(_) => Type::UniformReg,
            Self::UniformPred(_) => Type::UniformPred,
            Self::Attribute(_) => Type::Attribute,
            Self::Patch(_) => Type::Patch,
            Self::U1(_) => Type::U1,
            Self::U8(_) => Type::U8,
            Self::U16(_) => Type::U16,
            Self::U32(_) => Type::U32,
            Self::U64(_) => Type::U64,
            Self::F32(_) => Type::F32,
            Self::F64(_) => Type::F64,
        }
    }
}

impl Display for Imm {
    /// Names print as themselves (`r3`, `pt`, `ur5`, `upt`, `a0x80`, `patch7`); numbers print
    /// as their type and their bits in hexadecimal (`u32 0x5`, `f32 0x3f800000`), booleans as
    /// `u1 true` or `u1 false`.
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match *self {
            Self::Reg(reg) => write!(f, "{reg}"),
            Self::Pred(pred) => write!(f, "{pred}"),
            Self::UniformReg(reg) => write!(f, "{reg}"),
            Self::UniformPred(pred) => write!(f, "{pred}"),
            Self::Attribute(attribute) => write!(f, "{attribute}"),
            Self::Patch(patch) => write!(f, "{patch}"),
            Self::U1(value) => write!(f, "u1 {value}"),
            Self::U8(value) => write!(f, "u8 {value:#x}"),
            Self::U16(value) => write!(f, "u16 {value:#x}"),
            Self::U32(value) | Self::F32(value) => write!(f, "{} {value:#x}", self.ty()),
            Self::U64(value) | Self::F64(value) => write!(f, "{} {value:#x}", self.ty()),
        }
    }
}

/// An instruction's argument: a value or an immediate.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Operand {
    /// The result of an instruction.
    Value(Value),
    /// An immediate.
    Imm(Imm),
}

impl Operand {
    /// The value, if the operand is one.
    pub const fn value(self) -> Option<Value> {
        match self {
            Self::Value(value) => Some(value),
            Self::Imm(_) => None,
        }
    }

    /// The immediate, if the operand is one.
    pub const fn imm(self) -> Option<Imm> {
        match self {
            Self::Imm(imm) => Some(imm),
            Self::Value(_) => None,
        }
    }
}

impl From<Value> for Operand {
    fn from(value: Value) -> Self {
        Self::Value(value)
    }
}

impl From<Imm> for Operand {
    fn from(imm: Imm) -> Self {
        Self::Imm(imm)
    }
}

impl Display for Operand {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Value(value) => write!(f, "{value}"),
            Self::Imm(imm) => write!(f, "{imm}"),
        }
    }
}
