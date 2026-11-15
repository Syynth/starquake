//! The IR's value types.
//!
//! Ported from Eden's `eden/src/shader_recompiler/frontend/ir/type.h` and `type.cpp` @ cf654052
//! (GPL-3.0-or-later). Eden's `Type` is a bit set because its C++ emitter has typed wrappers that
//! accept several types (`F16F32F64`); its opcode table only ever names one type per slot, so
//! here a type is a plain enum. Two additions for the uniform register class (owner's decision 1,
//! 2026-11-12): [`Type::UniformReg`] and [`Type::UniformPred`], the types of uniform register
//! and uniform predicate names, as [`Type::Reg`] and [`Type::Pred`] are for the per-thread ones.

use std::fmt::{self, Display, Formatter};

/// The type of an operand or a result.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Type {
    /// No value: the result type of an operation that only has effects.
    Void,
    /// Any type: an argument slot that accepts every type, and the result of `Identity`, whose
    /// value has its argument's type (Eden's `AreTypesCompatible`, `Value::Type`).
    Opaque,
    /// A per-thread register name (`r0` to `r254`, `rz`).
    Reg,
    /// A predicate name (`p0` to `p6`, `pt`).
    Pred,
    /// A uniform register name (`ur0` to `ur62`, `urz`): starquake's own register class.
    UniformReg,
    /// A uniform predicate name (`up0` to `up6`, `upt`): starquake's own register class.
    UniformPred,
    /// An attribute word.
    Attribute,
    /// A tessellation patch word.
    Patch,
    /// A boolean (a predicate's value).
    U1,
    /// An 8-bit integer.
    U8,
    /// A 16-bit integer.
    U16,
    /// A 32-bit integer.
    U32,
    /// A 64-bit integer.
    U64,
    /// A 16-bit float.
    F16,
    /// A 32-bit float.
    F32,
    /// A 64-bit float.
    F64,
    /// Two 32-bit integers.
    U32x2,
    /// Three 32-bit integers.
    U32x3,
    /// Four 32-bit integers.
    U32x4,
    /// Two 16-bit floats.
    F16x2,
    /// Three 16-bit floats.
    F16x3,
    /// Four 16-bit floats.
    F16x4,
    /// Two 32-bit floats.
    F32x2,
    /// Three 32-bit floats.
    F32x3,
    /// Four 32-bit floats.
    F32x4,
    /// Two 64-bit floats.
    F64x2,
    /// Three 64-bit floats.
    F64x3,
    /// Four 64-bit floats.
    F64x4,
}

impl Type {
    /// Every type, in declaration order.
    pub const ALL: [Self; 28] = [
        Self::Void,
        Self::Opaque,
        Self::Reg,
        Self::Pred,
        Self::UniformReg,
        Self::UniformPred,
        Self::Attribute,
        Self::Patch,
        Self::U1,
        Self::U8,
        Self::U16,
        Self::U32,
        Self::U64,
        Self::F16,
        Self::F32,
        Self::F64,
        Self::U32x2,
        Self::U32x3,
        Self::U32x4,
        Self::F16x2,
        Self::F16x3,
        Self::F16x4,
        Self::F32x2,
        Self::F32x3,
        Self::F32x4,
        Self::F64x2,
        Self::F64x3,
        Self::F64x4,
    ];

    /// The type's name in the text form.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Void => "void",
            Self::Opaque => "opaque",
            Self::Reg => "reg",
            Self::Pred => "pred",
            Self::UniformReg => "ureg",
            Self::UniformPred => "upred",
            Self::Attribute => "attribute",
            Self::Patch => "patch",
            Self::U1 => "u1",
            Self::U8 => "u8",
            Self::U16 => "u16",
            Self::U32 => "u32",
            Self::U64 => "u64",
            Self::F16 => "f16",
            Self::F32 => "f32",
            Self::F64 => "f64",
            Self::U32x2 => "u32x2",
            Self::U32x3 => "u32x3",
            Self::U32x4 => "u32x4",
            Self::F16x2 => "f16x2",
            Self::F16x3 => "f16x3",
            Self::F16x4 => "f16x4",
            Self::F32x2 => "f32x2",
            Self::F32x3 => "f32x3",
            Self::F32x4 => "f32x4",
            Self::F64x2 => "f64x2",
            Self::F64x3 => "f64x3",
            Self::F64x4 => "f64x4",
        }
    }

    /// The type named `name` in the text form.
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|ty| ty.name() == name)
    }

    /// Whether a value of type `found` may fill a slot of type `self`: the same type, or any type
    /// in an [`Type::Opaque`] slot (Eden's `AreTypesCompatible`, read with the argument's type
    /// already resolved, so it is never `Opaque` itself).
    pub fn accepts(self, found: Self) -> bool {
        self == Self::Opaque || self == found
    }

    /// Whether a value can have this type: every type but [`Type::Void`], [`Type::Opaque`] and
    /// the name types (registers, predicates, attributes, patches), which only immediates have.
    /// A phi's type must be one of these.
    pub const fn is_value(self) -> bool {
        !matches!(
            self,
            Self::Void
                | Self::Opaque
                | Self::Reg
                | Self::Pred
                | Self::UniformReg
                | Self::UniformPred
                | Self::Attribute
                | Self::Patch
        )
    }
}

impl Display for Type {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}
