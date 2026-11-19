//! The values the interpreter computes: one variant per value type of the IR ([`Type`]), with
//! floats kept as their bits so a NaN's payload and a zero's sign survive exactly and two
//! values compare equal only when their bits do.

use crate::ir::{Imm, Type};

/// A computed value: one variant per value type of the IR (every [`Type`] for which
/// [`Type::is_value`] holds). Floats are their IEEE 754 bits (binary16, binary32, binary64);
/// vectors list their components in order, component 0 first.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Val {
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
    /// A 16-bit float, as its bits.
    F16(u16),
    /// A 32-bit float, as its bits.
    F32(u32),
    /// A 64-bit float, as its bits.
    F64(u64),
    /// Two 32-bit integers.
    U32x2([u32; 2]),
    /// Three 32-bit integers.
    U32x3([u32; 3]),
    /// Four 32-bit integers.
    U32x4([u32; 4]),
    /// Two 16-bit floats, as their bits.
    F16x2([u16; 2]),
    /// Three 16-bit floats, as their bits.
    F16x3([u16; 3]),
    /// Four 16-bit floats, as their bits.
    F16x4([u16; 4]),
    /// Two 32-bit floats, as their bits.
    F32x2([u32; 2]),
    /// Three 32-bit floats, as their bits.
    F32x3([u32; 3]),
    /// Four 32-bit floats, as their bits.
    F32x4([u32; 4]),
    /// Two 64-bit floats, as their bits.
    F64x2([u64; 2]),
    /// Three 64-bit floats, as their bits.
    F64x3([u64; 3]),
    /// Four 64-bit floats, as their bits.
    F64x4([u64; 4]),
}

/// The scalar kind of a vector's components.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Lane {
    /// 32-bit integers.
    U32,
    /// 16-bit floats.
    F16,
    /// 32-bit floats.
    F32,
    /// 64-bit floats.
    F64,
}

impl Lane {
    /// The scalar value of this kind holding `bits` (truncated to the kind's width).
    pub(super) fn scalar(self, bits: u64) -> Val {
        match self {
            Self::U32 => Val::U32(low32(bits)),
            Self::F16 => Val::F16(low16(bits)),
            Self::F32 => Val::F32(low32(bits)),
            Self::F64 => Val::F64(bits),
        }
    }

    /// The bits of `value` if it is a scalar of this kind.
    pub(super) fn bits_of(self, value: Val) -> Option<u64> {
        match (self, value) {
            (Self::U32, Val::U32(bits)) | (Self::F32, Val::F32(bits)) => Some(u64::from(bits)),
            (Self::F16, Val::F16(bits)) => Some(u64::from(bits)),
            (Self::F64, Val::F64(bits)) => Some(bits),
            _ => None,
        }
    }

    /// The scalar type of this kind.
    pub(super) const fn ty(self) -> Type {
        match self {
            Self::U32 => Type::U32,
            Self::F16 => Type::F16,
            Self::F32 => Type::F32,
            Self::F64 => Type::F64,
        }
    }
}

/// The low 32 bits of `bits`.
pub(super) const fn low32(bits: u64) -> u32 {
    let [b0, b1, b2, b3, ..] = bits.to_le_bytes();
    u32::from_le_bytes([b0, b1, b2, b3])
}

/// The low 16 bits of `bits`.
pub(super) const fn low16(bits: u64) -> u16 {
    let [b0, b1, ..] = bits.to_le_bytes();
    u16::from_le_bytes([b0, b1])
}

macro_rules! widen {
    ($array:expr) => {{
        let mut bits = [0_u64; 4];
        for (slot, &component) in bits.iter_mut().zip($array.iter()) {
            *slot = u64::from(component);
        }
        bits
    }};
}

macro_rules! narrow {
    ($bits:expr, $len:literal, $f:ident) => {{
        let mut out = [Default::default(); $len];
        for (slot, &component) in out.iter_mut().zip($bits.iter()) {
            *slot = $f(component);
        }
        out
    }};
}

impl Val {
    /// The value's type.
    pub const fn ty(self) -> Type {
        match self {
            Self::U1(_) => Type::U1,
            Self::U8(_) => Type::U8,
            Self::U16(_) => Type::U16,
            Self::U32(_) => Type::U32,
            Self::U64(_) => Type::U64,
            Self::F16(_) => Type::F16,
            Self::F32(_) => Type::F32,
            Self::F64(_) => Type::F64,
            Self::U32x2(_) => Type::U32x2,
            Self::U32x3(_) => Type::U32x3,
            Self::U32x4(_) => Type::U32x4,
            Self::F16x2(_) => Type::F16x2,
            Self::F16x3(_) => Type::F16x3,
            Self::F16x4(_) => Type::F16x4,
            Self::F32x2(_) => Type::F32x2,
            Self::F32x3(_) => Type::F32x3,
            Self::F32x4(_) => Type::F32x4,
            Self::F64x2(_) => Type::F64x2,
            Self::F64x3(_) => Type::F64x3,
            Self::F64x4(_) => Type::F64x4,
        }
    }

    /// A 32-bit float value.
    pub const fn f32(value: f32) -> Self {
        Self::F32(value.to_bits())
    }

    /// A 64-bit float value.
    pub const fn f64(value: f64) -> Self {
        Self::F64(value.to_bits())
    }

    /// The value of an immediate; `None` for a name (a register, predicate, attribute or
    /// patch), which is not a value.
    pub const fn from_imm(imm: Imm) -> Option<Self> {
        match imm {
            Imm::U1(value) => Some(Self::U1(value)),
            Imm::U8(value) => Some(Self::U8(value)),
            Imm::U16(value) => Some(Self::U16(value)),
            Imm::U32(value) => Some(Self::U32(value)),
            Imm::U64(value) => Some(Self::U64(value)),
            Imm::F32(bits) => Some(Self::F32(bits)),
            Imm::F64(bits) => Some(Self::F64(bits)),
            Imm::Reg(_)
            | Imm::Pred(_)
            | Imm::UniformReg(_)
            | Imm::UniformPred(_)
            | Imm::Attribute(_)
            | Imm::Patch(_) => None,
        }
    }

    /// The value every bit of which is zero, of type `ty`; `None` for a type no value has.
    pub const fn zero(ty: Type) -> Option<Self> {
        Some(match ty {
            Type::U1 => Self::U1(false),
            Type::U8 => Self::U8(0),
            Type::U16 => Self::U16(0),
            Type::U32 => Self::U32(0),
            Type::U64 => Self::U64(0),
            Type::F16 => Self::F16(0),
            Type::F32 => Self::F32(0),
            Type::F64 => Self::F64(0),
            Type::U32x2 => Self::U32x2([0; 2]),
            Type::U32x3 => Self::U32x3([0; 3]),
            Type::U32x4 => Self::U32x4([0; 4]),
            Type::F16x2 => Self::F16x2([0; 2]),
            Type::F16x3 => Self::F16x3([0; 3]),
            Type::F16x4 => Self::F16x4([0; 4]),
            Type::F32x2 => Self::F32x2([0; 2]),
            Type::F32x3 => Self::F32x3([0; 3]),
            Type::F32x4 => Self::F32x4([0; 4]),
            Type::F64x2 => Self::F64x2([0; 2]),
            Type::F64x3 => Self::F64x3([0; 3]),
            Type::F64x4 => Self::F64x4([0; 4]),
            Type::Void
            | Type::Opaque
            | Type::Reg
            | Type::Pred
            | Type::UniformReg
            | Type::UniformPred
            | Type::Attribute
            | Type::Patch => return None,
        })
    }

    /// The components of a vector as bits, zero past its length; `None` for a scalar.
    pub(super) fn lanes(self) -> Option<[u64; 4]> {
        Some(match self {
            Self::U32x2(v) | Self::F32x2(v) => widen!(v),
            Self::U32x3(v) | Self::F32x3(v) => widen!(v),
            Self::U32x4(v) | Self::F32x4(v) => widen!(v),
            Self::F16x2(v) => widen!(v),
            Self::F16x3(v) => widen!(v),
            Self::F16x4(v) => widen!(v),
            Self::F64x2(v) => widen!(v),
            Self::F64x3(v) => widen!(v),
            Self::F64x4(v) => widen!(v),
            Self::U1(_)
            | Self::U8(_)
            | Self::U16(_)
            | Self::U32(_)
            | Self::U64(_)
            | Self::F16(_)
            | Self::F32(_)
            | Self::F64(_) => return None,
        })
    }

    /// The vector of kind `lane` with the first `len` of `bits` as its components; `None` for
    /// a length other than 2, 3 or 4.
    pub(super) fn from_lanes(lane: Lane, bits: [u64; 4], len: usize) -> Option<Self> {
        let identity = |bits: u64| bits;
        Some(match (lane, len) {
            (Lane::U32, 2) => Self::U32x2(narrow!(bits, 2, low32)),
            (Lane::U32, 3) => Self::U32x3(narrow!(bits, 3, low32)),
            (Lane::U32, 4) => Self::U32x4(narrow!(bits, 4, low32)),
            (Lane::F16, 2) => Self::F16x2(narrow!(bits, 2, low16)),
            (Lane::F16, 3) => Self::F16x3(narrow!(bits, 3, low16)),
            (Lane::F16, 4) => Self::F16x4(narrow!(bits, 4, low16)),
            (Lane::F32, 2) => Self::F32x2(narrow!(bits, 2, low32)),
            (Lane::F32, 3) => Self::F32x3(narrow!(bits, 3, low32)),
            (Lane::F32, 4) => Self::F32x4(narrow!(bits, 4, low32)),
            (Lane::F64, 2) => Self::F64x2(narrow!(bits, 2, identity)),
            (Lane::F64, 3) => Self::F64x3(narrow!(bits, 3, identity)),
            (Lane::F64, 4) => Self::F64x4(narrow!(bits, 4, identity)),
            _ => return None,
        })
    }
}
