//! The operations: one arm per supported opcode, grouped as the parent module's "Semantics"
//! lists them, with the source of each group there.

use crate::ir::{Flags, FlagsKind, Imm, Opcode, Operation, Site, TextureInfo, Type, Value};

use super::float::{self, Compare, Control, DirectedF64, Format, Integral};
use super::integer;
use super::memory::Place;
use super::value::{Lane, low16, low32};
use super::{ImageCall, Input, InterpError, Run, Space, Val, type_error};

mod warp;

/// The most arguments any opcode takes.
const MAX_ARGS: usize = {
    let mut most = 0;
    let mut index = 0;
    while index < Opcode::ALL.len() {
        let count = Opcode::ALL[index].args().len();
        if count > most {
            most = count;
        }
        index += 1;
    }
    most
};

/// Whether a group of operations handled an opcode.
enum Handled {
    /// The opcode is not in the group.
    Not,
    /// It is, with this value (`None` for an operation with no result).
    Done(Option<Val>),
}

/// An operation's arguments: each a value, or the type of a name (which is not a value).
struct Args {
    at: Value,
    slots: [Result<Val, Type>; MAX_ARGS],
    /// Each argument's immediate, if it is one (names are immediates).
    names: [Option<Imm>; MAX_ARGS],
}

/// Generates a typed accessor for [`Args`].
macro_rules! accessor {
    ($(#[$meta:meta])* $name:ident, $variant:ident, $out:ty) => {
        $(#[$meta])*
        fn $name(&self, index: usize) -> Result<$out, InterpError> {
            match self.val(index, Type::$variant)? {
                Val::$variant(value) => Ok(value.into()),
                other => Err(self.mismatch(index, Type::$variant, other.ty())),
            }
        }
    };
}

impl Args {
    fn site(&self) -> Site {
        Site::Inst(self.at)
    }

    fn mismatch(&self, index: usize, wanted: Type, found: Type) -> InterpError {
        type_error(self.site(), index, wanted, found)
    }

    /// Argument `index`, which must be a value (of type `wanted`, for the error).
    fn val(&self, index: usize, wanted: Type) -> Result<Val, InterpError> {
        match self.slots.get(index) {
            Some(Ok(value)) => Ok(*value),
            Some(Err(found)) => Err(self.mismatch(index, wanted, *found)),
            None => Err(self.mismatch(index, wanted, Type::Void)),
        }
    }

    accessor!(
        /// A `u1` argument.
        u1,
        U1,
        bool
    );
    accessor!(
        /// A `u8` argument.
        u8,
        U8,
        u8
    );
    accessor!(
        /// A `u16` argument.
        u16,
        U16,
        u16
    );
    accessor!(
        /// A `u32` argument.
        u32,
        U32,
        u32
    );
    accessor!(
        /// A `u64` argument.
        u64,
        U64,
        u64
    );

    /// The name argument `index`, of type `wanted`.
    fn name(&self, index: usize, wanted: Type) -> Result<Imm, InterpError> {
        match self.names.get(index).copied().flatten() {
            Some(imm) if imm.ty() == wanted => Ok(imm),
            other => Err(self.mismatch(index, wanted, other.map_or(Type::Void, Imm::ty))),
        }
    }

    /// A register name argument, as its slot.
    fn reg(&self, index: usize) -> Result<usize, InterpError> {
        match self.name(index, Type::Reg)? {
            Imm::Reg(reg) => Ok(usize::from(reg.index())),
            other => Err(self.mismatch(index, Type::Reg, other.ty())),
        }
    }

    /// A predicate name argument, as its slot.
    fn pred(&self, index: usize) -> Result<usize, InterpError> {
        match self.name(index, Type::Pred)? {
            Imm::Pred(pred) => Ok(usize::from(pred.index())),
            other => Err(self.mismatch(index, Type::Pred, other.ty())),
        }
    }

    /// A uniform register name argument, as its slot.
    fn ureg(&self, index: usize) -> Result<usize, InterpError> {
        match self.name(index, Type::UniformReg)? {
            Imm::UniformReg(reg) => Ok(usize::from(reg.index())),
            other => Err(self.mismatch(index, Type::UniformReg, other.ty())),
        }
    }

    /// A uniform predicate name argument, as its slot.
    fn upred(&self, index: usize) -> Result<usize, InterpError> {
        match self.name(index, Type::UniformPred)? {
            Imm::UniformPred(pred) => Ok(usize::from(pred.index())),
            other => Err(self.mismatch(index, Type::UniformPred, other.ty())),
        }
    }

    /// An attribute name argument.
    fn attribute(&self, index: usize) -> Result<crate::ir::Attribute, InterpError> {
        match self.name(index, Type::Attribute)? {
            Imm::Attribute(attribute) => Ok(attribute),
            other => Err(self.mismatch(index, Type::Attribute, other.ty())),
        }
    }

    /// A patch name argument.
    fn patch(&self, index: usize) -> Result<crate::ir::Patch, InterpError> {
        match self.name(index, Type::Patch)? {
            Imm::Patch(patch) => Ok(patch),
            other => Err(self.mismatch(index, Type::Patch, other.ty())),
        }
    }

    /// A float argument of `format`, as its bits.
    fn float(&self, index: usize, format: Format) -> Result<u64, InterpError> {
        match (format, self.val(index, float_type(format))?) {
            (Format::F16, Val::F16(bits)) => Ok(u64::from(bits)),
            (Format::F32, Val::F32(bits)) => Ok(u64::from(bits)),
            (Format::F64, Val::F64(bits)) => Ok(bits),
            (_, other) => Err(self.mismatch(index, float_type(format), other.ty())),
        }
    }

    /// A vector argument of type `ty`, as its components.
    fn vector(&self, index: usize, ty: Type) -> Result<[u64; 4], InterpError> {
        let value = self.val(index, ty)?;
        match value.lanes() {
            Some(bits) if value.ty() == ty => Ok(bits),
            _ => Err(self.mismatch(index, ty, value.ty())),
        }
    }
}

/// The scalar type of a float format.
const fn float_type(format: Format) -> Type {
    match format {
        Format::F16 => Type::F16,
        Format::F32 => Type::F32,
        Format::F64 => Type::F64,
    }
}

/// A float value of `format` from its bits.
fn float_val(format: Format, bits: u64) -> Val {
    match format {
        Format::F16 => Val::F16(low16(bits)),
        Format::F32 => Val::F32(low32(bits)),
        Format::F64 => Val::F64(bits),
    }
}

/// The components' kind and count of a vector type.
const fn vector_shape(ty: Type) -> Option<(Lane, usize)> {
    Some(match ty {
        Type::U32x2 => (Lane::U32, 2),
        Type::U32x3 => (Lane::U32, 3),
        Type::U32x4 => (Lane::U32, 4),
        Type::F16x2 => (Lane::F16, 2),
        Type::F16x3 => (Lane::F16, 3),
        Type::F16x4 => (Lane::F16, 4),
        Type::F32x2 => (Lane::F32, 2),
        Type::F32x3 => (Lane::F32, 3),
        Type::F32x4 => (Lane::F32, 4),
        Type::F64x2 => (Lane::F64, 2),
        Type::F64x3 => (Lane::F64, 3),
        Type::F64x4 => (Lane::F64, 4),
        _ => return None,
    })
}

/// The width and extension of a memory access.
#[derive(Clone, Copy)]
enum Width {
    U8,
    S8,
    U16,
    S16,
    B32,
    B64,
    B128,
}

impl Width {
    const fn size(self) -> u64 {
        match self {
            Self::U8 | Self::S8 => 1,
            Self::U16 | Self::S16 => 2,
            Self::B32 => 4,
            Self::B64 => 8,
            Self::B128 => 16,
        }
    }

    /// The value loaded from the little-endian bytes `raw`.
    fn loaded(self, raw: u128) -> Val {
        let words = raw.to_le_bytes();
        let word = |index: usize| {
            let at = index * 4;
            match words.get(at..at + 4) {
                Some(&[b0, b1, b2, b3]) => u32::from_le_bytes([b0, b1, b2, b3]),
                _ => 0,
            }
        };
        let [b0, b1, ..] = words;
        match self {
            Self::U8 => Val::U32(u32::from(b0)),
            Self::S8 => Val::U32(i32::from(b0.cast_signed()).cast_unsigned()),
            Self::U16 => Val::U32(u32::from(u16::from_le_bytes([b0, b1]))),
            Self::S16 => Val::U32(i32::from(i16::from_le_bytes([b0, b1])).cast_unsigned()),
            Self::B32 => Val::U32(word(0)),
            Self::B64 => Val::U32x2([word(0), word(1)]),
            Self::B128 => Val::U32x4([word(0), word(1), word(2), word(3)]),
        }
    }

    /// The type of the value stored.
    const fn stored_type(self) -> Type {
        match self {
            Self::U8 | Self::S8 | Self::U16 | Self::S16 | Self::B32 => Type::U32,
            Self::B64 => Type::U32x2,
            Self::B128 => Type::U32x4,
        }
    }

    /// The bytes stored for `value` (its low `size` bytes are written).
    fn stored(self, args: &Args, index: usize) -> Result<u128, InterpError> {
        let ty = self.stored_type();
        if ty == Type::U32 {
            return Ok(u128::from(args.u32(index)?));
        }
        let words = args.vector(index, ty)?;
        Ok(words
            .iter()
            .rev()
            .fold(0_u128, |acc, &word| (acc << 32) | u128::from(word)))
    }
}

/// An integer atomic operation.
#[derive(Clone, Copy)]
enum Atomic {
    Add,
    SMin,
    UMin,
    SMax,
    UMax,
    Inc,
    Dec,
    And,
    Or,
    Xor,
    Exchange,
}

impl Atomic {
    fn apply(self, old: u32, value: u32) -> u32 {
        match self {
            Self::Add => old.wrapping_add(value),
            Self::SMin => old.cast_signed().min(value.cast_signed()).cast_unsigned(),
            Self::UMin => old.min(value),
            Self::SMax => old.cast_signed().max(value.cast_signed()).cast_unsigned(),
            Self::UMax => old.max(value),
            Self::Inc => integer::atomic_inc(old, value),
            Self::Dec => integer::atomic_dec(old, value),
            Self::And => old & value,
            Self::Or => old | value,
            Self::Xor => old ^ value,
            Self::Exchange => value,
        }
    }
}

/// Where an atomic goes.
#[derive(Clone, Copy)]
enum AtomicSpace {
    Shared,
    Global,
    Storage,
}

/// The space and operation of a 32-bit integer atomic opcode.
const fn atomic32(opcode: Opcode) -> Option<(AtomicSpace, Atomic)> {
    use AtomicSpace::{Global, Shared, Storage};
    use Opcode as O;
    Some(match opcode {
        O::SharedAtomicIAdd32 => (Shared, Atomic::Add),
        O::SharedAtomicSMin32 => (Shared, Atomic::SMin),
        O::SharedAtomicUMin32 => (Shared, Atomic::UMin),
        O::SharedAtomicSMax32 => (Shared, Atomic::SMax),
        O::SharedAtomicUMax32 => (Shared, Atomic::UMax),
        O::SharedAtomicInc32 => (Shared, Atomic::Inc),
        O::SharedAtomicDec32 => (Shared, Atomic::Dec),
        O::SharedAtomicAnd32 => (Shared, Atomic::And),
        O::SharedAtomicOr32 => (Shared, Atomic::Or),
        O::SharedAtomicXor32 => (Shared, Atomic::Xor),
        O::SharedAtomicExchange32 => (Shared, Atomic::Exchange),
        O::GlobalAtomicIAdd32 => (Global, Atomic::Add),
        O::GlobalAtomicSMin32 => (Global, Atomic::SMin),
        O::GlobalAtomicUMin32 => (Global, Atomic::UMin),
        O::GlobalAtomicSMax32 => (Global, Atomic::SMax),
        O::GlobalAtomicUMax32 => (Global, Atomic::UMax),
        O::GlobalAtomicInc32 => (Global, Atomic::Inc),
        O::GlobalAtomicDec32 => (Global, Atomic::Dec),
        O::GlobalAtomicAnd32 => (Global, Atomic::And),
        O::GlobalAtomicOr32 => (Global, Atomic::Or),
        O::GlobalAtomicXor32 => (Global, Atomic::Xor),
        O::GlobalAtomicExchange32 => (Global, Atomic::Exchange),
        O::StorageAtomicIAdd32 => (Storage, Atomic::Add),
        O::StorageAtomicSMin32 => (Storage, Atomic::SMin),
        O::StorageAtomicUMin32 => (Storage, Atomic::UMin),
        O::StorageAtomicSMax32 => (Storage, Atomic::SMax),
        O::StorageAtomicUMax32 => (Storage, Atomic::UMax),
        O::StorageAtomicInc32 => (Storage, Atomic::Inc),
        O::StorageAtomicDec32 => (Storage, Atomic::Dec),
        O::StorageAtomicAnd32 => (Storage, Atomic::And),
        O::StorageAtomicOr32 => (Storage, Atomic::Or),
        O::StorageAtomicXor32 => (Storage, Atomic::Xor),
        O::StorageAtomicExchange32 => (Storage, Atomic::Exchange),
        _ => return None,
    })
}

/// The format, comparison and orderedness of a float comparison opcode.
const fn fp_compare(opcode: Opcode) -> Option<(Format, Compare, bool)> {
    use Compare::{Equal, Greater, GreaterEqual, Less, LessEqual, NotEqual};
    use Format::{F16, F32, F64};
    use Opcode as O;
    Some(match opcode {
        O::FPOrdEqual16 => (F16, Equal, true),
        O::FPOrdEqual32 => (F32, Equal, true),
        O::FPOrdEqual64 => (F64, Equal, true),
        O::FPUnordEqual16 => (F16, Equal, false),
        O::FPUnordEqual32 => (F32, Equal, false),
        O::FPUnordEqual64 => (F64, Equal, false),
        O::FPOrdNotEqual16 => (F16, NotEqual, true),
        O::FPOrdNotEqual32 => (F32, NotEqual, true),
        O::FPOrdNotEqual64 => (F64, NotEqual, true),
        O::FPUnordNotEqual16 => (F16, NotEqual, false),
        O::FPUnordNotEqual32 => (F32, NotEqual, false),
        O::FPUnordNotEqual64 => (F64, NotEqual, false),
        O::FPOrdLessThan16 => (F16, Less, true),
        O::FPOrdLessThan32 => (F32, Less, true),
        O::FPOrdLessThan64 => (F64, Less, true),
        O::FPUnordLessThan16 => (F16, Less, false),
        O::FPUnordLessThan32 => (F32, Less, false),
        O::FPUnordLessThan64 => (F64, Less, false),
        O::FPOrdGreaterThan16 => (F16, Greater, true),
        O::FPOrdGreaterThan32 => (F32, Greater, true),
        O::FPOrdGreaterThan64 => (F64, Greater, true),
        O::FPUnordGreaterThan16 => (F16, Greater, false),
        O::FPUnordGreaterThan32 => (F32, Greater, false),
        O::FPUnordGreaterThan64 => (F64, Greater, false),
        O::FPOrdLessThanEqual16 => (F16, LessEqual, true),
        O::FPOrdLessThanEqual32 => (F32, LessEqual, true),
        O::FPOrdLessThanEqual64 => (F64, LessEqual, true),
        O::FPUnordLessThanEqual16 => (F16, LessEqual, false),
        O::FPUnordLessThanEqual32 => (F32, LessEqual, false),
        O::FPUnordLessThanEqual64 => (F64, LessEqual, false),
        O::FPOrdGreaterThanEqual16 => (F16, GreaterEqual, true),
        O::FPOrdGreaterThanEqual32 => (F32, GreaterEqual, true),
        O::FPOrdGreaterThanEqual64 => (F64, GreaterEqual, true),
        O::FPUnordGreaterThanEqual16 => (F16, GreaterEqual, false),
        O::FPUnordGreaterThanEqual32 => (F32, GreaterEqual, false),
        O::FPUnordGreaterThanEqual64 => (F64, GreaterEqual, false),
        _ => return None,
    })
}

/// An integer comparison.
#[derive(Clone, Copy)]
enum IntCompare {
    SLess,
    ULess,
    Equal,
    SLessEqual,
    ULessEqual,
    SGreater,
    UGreater,
    NotEqual,
    SGreaterEqual,
    UGreaterEqual,
}

impl IntCompare {
    const fn eval(self, a: u32, b: u32) -> bool {
        let (sa, sb) = (a.cast_signed(), b.cast_signed());
        match self {
            Self::SLess => sa < sb,
            Self::ULess => a < b,
            Self::Equal => a == b,
            Self::SLessEqual => sa <= sb,
            Self::ULessEqual => a <= b,
            Self::SGreater => sa > sb,
            Self::UGreater => a > b,
            Self::NotEqual => a != b,
            Self::SGreaterEqual => sa >= sb,
            Self::UGreaterEqual => a >= b,
        }
    }
}

const fn int_compare(opcode: Opcode) -> Option<IntCompare> {
    Some(match opcode {
        Opcode::SLessThan => IntCompare::SLess,
        Opcode::ULessThan => IntCompare::ULess,
        Opcode::IEqual => IntCompare::Equal,
        Opcode::SLessThanEqual => IntCompare::SLessEqual,
        Opcode::ULessThanEqual => IntCompare::ULessEqual,
        Opcode::SGreaterThan => IntCompare::SGreater,
        Opcode::UGreaterThan => IntCompare::UGreater,
        Opcode::INotEqual => IntCompare::NotEqual,
        Opcode::SGreaterThanEqual => IntCompare::SGreaterEqual,
        Opcode::UGreaterThanEqual => IntCompare::UGreaterEqual,
        _ => return None,
    })
}

/// The float format, if any, of a unary float operation's operand and result, and how it
/// computes its result.
#[derive(Clone, Copy)]
enum FloatUnary {
    Abs,
    Neg,
    Saturate,
    Integral(Integral),
    IsNan,
}

const fn float_unary(opcode: Opcode) -> Option<(Format, FloatUnary)> {
    use Format::{F16, F32, F64};
    use Opcode as O;
    Some(match opcode {
        O::FPAbs16 => (F16, FloatUnary::Abs),
        O::FPAbs32 => (F32, FloatUnary::Abs),
        O::FPAbs64 => (F64, FloatUnary::Abs),
        O::FPNeg16 => (F16, FloatUnary::Neg),
        O::FPNeg32 => (F32, FloatUnary::Neg),
        O::FPNeg64 => (F64, FloatUnary::Neg),
        O::FPSaturate16 => (F16, FloatUnary::Saturate),
        O::FPSaturate32 => (F32, FloatUnary::Saturate),
        O::FPSaturate64 => (F64, FloatUnary::Saturate),
        O::FPRoundEven16 => (F16, FloatUnary::Integral(Integral::Even)),
        O::FPRoundEven32 => (F32, FloatUnary::Integral(Integral::Even)),
        O::FPRoundEven64 => (F64, FloatUnary::Integral(Integral::Even)),
        O::FPFloor16 => (F16, FloatUnary::Integral(Integral::Floor)),
        O::FPFloor32 => (F32, FloatUnary::Integral(Integral::Floor)),
        O::FPFloor64 => (F64, FloatUnary::Integral(Integral::Floor)),
        O::FPCeil16 => (F16, FloatUnary::Integral(Integral::Ceil)),
        O::FPCeil32 => (F32, FloatUnary::Integral(Integral::Ceil)),
        O::FPCeil64 => (F64, FloatUnary::Integral(Integral::Ceil)),
        O::FPTrunc16 => (F16, FloatUnary::Integral(Integral::Trunc)),
        O::FPTrunc32 => (F32, FloatUnary::Integral(Integral::Trunc)),
        O::FPTrunc64 => (F64, FloatUnary::Integral(Integral::Trunc)),
        O::FPIsNan16 => (F16, FloatUnary::IsNan),
        O::FPIsNan32 => (F32, FloatUnary::IsNan),
        O::FPIsNan64 => (F64, FloatUnary::IsNan),
        _ => return None,
    })
}

/// A binary or ternary float arithmetic operation.
#[derive(Clone, Copy)]
enum FloatArith {
    Add,
    Mul,
    Fma,
    Max,
    Min,
    Clamp,
}

const fn float_arith(opcode: Opcode) -> Option<(Format, FloatArith)> {
    use Format::{F16, F32, F64};
    use Opcode as O;
    Some(match opcode {
        O::FPAdd16 => (F16, FloatArith::Add),
        O::FPAdd32 => (F32, FloatArith::Add),
        O::FPAdd64 => (F64, FloatArith::Add),
        O::FPMul16 => (F16, FloatArith::Mul),
        O::FPMul32 => (F32, FloatArith::Mul),
        O::FPMul64 => (F64, FloatArith::Mul),
        O::FPFma16 => (F16, FloatArith::Fma),
        O::FPFma32 => (F32, FloatArith::Fma),
        O::FPFma64 => (F64, FloatArith::Fma),
        O::FPMax32 => (F32, FloatArith::Max),
        O::FPMax64 => (F64, FloatArith::Max),
        O::FPMin32 => (F32, FloatArith::Min),
        O::FPMin64 => (F64, FloatArith::Min),
        O::FPClamp16 => (F16, FloatArith::Clamp),
        O::FPClamp32 => (F32, FloatArith::Clamp),
        O::FPClamp64 => (F64, FloatArith::Clamp),
        _ => return None,
    })
}

/// The integer source of an integer-to-float conversion.
#[derive(Clone, Copy)]
enum IntSource {
    S8,
    S16,
    S32,
    S64,
    U8,
    U16,
    U32,
    U64,
}

impl IntSource {
    /// The integer argument 0 holds: the low 8 or 16 bits of a 32-bit argument sign- or
    /// zero-extended, as Eden's `ExtractS8` and siblings take them.
    fn read(self, args: &Args) -> Result<i128, InterpError> {
        let low_bytes = |args: &Args| args.u32(0).map(u32::to_le_bytes);
        Ok(match self {
            Self::S8 => i128::from(low_bytes(args)?[0].cast_signed()),
            Self::S16 => {
                let [b0, b1, ..] = low_bytes(args)?;
                i128::from(i16::from_le_bytes([b0, b1]))
            }
            Self::S32 => i128::from(args.u32(0)?.cast_signed()),
            Self::S64 => i128::from(args.u64(0)?.cast_signed()),
            Self::U8 => i128::from(low_bytes(args)?[0]),
            Self::U16 => {
                let [b0, b1, ..] = low_bytes(args)?;
                i128::from(u16::from_le_bytes([b0, b1]))
            }
            Self::U32 => i128::from(args.u32(0)?),
            Self::U64 => i128::from(args.u64(0)?),
        })
    }
}

const fn int_to_float(opcode: Opcode) -> Option<(IntSource, Format)> {
    use Format::{F16, F32, F64};
    use IntSource::{S8, S16, S32, S64, U8, U16, U32, U64};
    use Opcode as O;
    Some(match opcode {
        O::ConvertF16S8 => (S8, F16),
        O::ConvertF16S16 => (S16, F16),
        O::ConvertF16S32 => (S32, F16),
        O::ConvertF16S64 => (S64, F16),
        O::ConvertF16U8 => (U8, F16),
        O::ConvertF16U16 => (U16, F16),
        O::ConvertF16U32 => (U32, F16),
        O::ConvertF16U64 => (U64, F16),
        O::ConvertF32S8 => (S8, F32),
        O::ConvertF32S16 => (S16, F32),
        O::ConvertF32S32 => (S32, F32),
        O::ConvertF32S64 => (S64, F32),
        O::ConvertF32U8 => (U8, F32),
        O::ConvertF32U16 => (U16, F32),
        O::ConvertF32U32 => (U32, F32),
        O::ConvertF32U64 => (U64, F32),
        O::ConvertF64S8 => (S8, F64),
        O::ConvertF64S16 => (S16, F64),
        O::ConvertF64S32 => (S32, F64),
        O::ConvertF64S64 => (S64, F64),
        O::ConvertF64U8 => (U8, F64),
        O::ConvertF64U16 => (U16, F64),
        O::ConvertF64U32 => (U32, F64),
        O::ConvertF64U64 => (U64, F64),
        _ => return None,
    })
}

/// The destination of a float-to-integer conversion: its range and whether the result is
/// 64 bits wide.
#[derive(Clone, Copy)]
enum IntDest {
    S16,
    S32,
    S64,
    U16,
    U32,
    U64,
}

impl IntDest {
    const fn range(self) -> (i128, i128) {
        match self {
            Self::S16 => (i16::MIN as i128, i16::MAX as i128),
            Self::S32 => (i32::MIN as i128, i32::MAX as i128),
            Self::S64 => (i64::MIN as i128, i64::MAX as i128),
            Self::U16 => (0, u16::MAX as i128),
            Self::U32 => (0, u32::MAX as i128),
            Self::U64 => (0, u64::MAX as i128),
        }
    }

    /// The result value for the in-range integer `value`: a 16-bit result sign- or
    /// zero-extended into 32 bits, as Eden's `OpSConvert`/`OpUConvert` to `U32` does.
    fn result(self, value: i128) -> Val {
        // The value is in the destination's range, so each conversion is exact.
        let bits = value.cast_unsigned();
        match self {
            Self::S16 | Self::S32 | Self::U16 | Self::U32 => Val::U32(low32(low128(bits))),
            Self::S64 | Self::U64 => Val::U64(low128(bits)),
        }
    }
}

/// The low 64 bits of `value`.
const fn low128(value: u128) -> u64 {
    let bytes = value.to_le_bytes();
    let [b0, b1, b2, b3, b4, b5, b6, b7, ..] = bytes;
    u64::from_le_bytes([b0, b1, b2, b3, b4, b5, b6, b7])
}

const fn float_to_int(opcode: Opcode) -> Option<(Format, IntDest)> {
    use Format::{F16, F32, F64};
    use IntDest::{S16, S32, S64, U16, U32, U64};
    use Opcode as O;
    Some(match opcode {
        O::ConvertS16F16 => (F16, S16),
        O::ConvertS16F32 => (F32, S16),
        O::ConvertS16F64 => (F64, S16),
        O::ConvertS32F16 => (F16, S32),
        O::ConvertS32F32 => (F32, S32),
        O::ConvertS32F64 => (F64, S32),
        O::ConvertS64F16 => (F16, S64),
        O::ConvertS64F32 => (F32, S64),
        O::ConvertS64F64 => (F64, S64),
        O::ConvertU16F16 => (F16, U16),
        O::ConvertU16F32 => (F32, U16),
        O::ConvertU16F64 => (F64, U16),
        O::ConvertU32F16 => (F16, U32),
        O::ConvertU32F32 => (F32, U32),
        O::ConvertU32F64 => (F64, U32),
        O::ConvertU64F16 => (F16, U64),
        O::ConvertU64F32 => (F32, U64),
        O::ConvertU64F64 => (F64, U64),
        _ => return None,
    })
}

const fn float_to_float(opcode: Opcode) -> Option<(Format, Format)> {
    use Format::{F16, F32, F64};
    Some(match opcode {
        Opcode::ConvertF16F32 => (F32, F16),
        Opcode::ConvertF32F16 => (F16, F32),
        Opcode::ConvertF32F64 => (F64, F32),
        Opcode::ConvertF64F32 => (F32, F64),
        _ => return None,
    })
}

/// The width of a memory load or store opcode and whether it stores.
const fn memory_width(opcode: Opcode) -> Option<(Width, bool)> {
    use Opcode as O;
    use Width::{B32, B64, B128, S8, S16, U8, U16};
    Some(match opcode {
        O::LoadGlobalU8 | O::LoadStorageU8 | O::LoadSharedU8 => (U8, false),
        O::LoadGlobalS8 | O::LoadStorageS8 | O::LoadSharedS8 => (S8, false),
        O::LoadGlobalU16 | O::LoadStorageU16 | O::LoadSharedU16 => (U16, false),
        O::LoadGlobalS16 | O::LoadStorageS16 | O::LoadSharedS16 => (S16, false),
        O::LoadGlobal32 | O::LoadStorage32 | O::LoadSharedU32 => (B32, false),
        O::LoadGlobal64 | O::LoadStorage64 | O::LoadSharedU64 => (B64, false),
        O::LoadGlobal128 | O::LoadStorage128 | O::LoadSharedU128 => (B128, false),
        O::WriteGlobalU8
        | O::WriteGlobalS8
        | O::WriteStorageU8
        | O::WriteStorageS8
        | O::WriteSharedU8 => (U8, true),
        O::WriteGlobalU16
        | O::WriteGlobalS16
        | O::WriteStorageU16
        | O::WriteStorageS16
        | O::WriteSharedU16 => (U16, true),
        O::WriteGlobal32 | O::WriteStorage32 | O::WriteSharedU32 => (B32, true),
        O::WriteGlobal64 | O::WriteStorage64 | O::WriteSharedU64 => (B64, true),
        O::WriteGlobal128 | O::WriteStorage128 | O::WriteSharedU128 => (B128, true),
        _ => return None,
    })
}

/// The memory space of a load, store or atomic opcode's name prefix.
#[derive(Clone, Copy)]
enum MemorySpace {
    Global,
    Storage,
    Shared,
}

const fn memory_space(opcode: Opcode) -> MemorySpace {
    use Opcode as O;
    match opcode {
        O::LoadStorageU8
        | O::LoadStorageS8
        | O::LoadStorageU16
        | O::LoadStorageS16
        | O::LoadStorage32
        | O::LoadStorage64
        | O::LoadStorage128
        | O::WriteStorageU8
        | O::WriteStorageS8
        | O::WriteStorageU16
        | O::WriteStorageS16
        | O::WriteStorage32
        | O::WriteStorage64
        | O::WriteStorage128 => MemorySpace::Storage,
        O::LoadSharedU8
        | O::LoadSharedS8
        | O::LoadSharedU16
        | O::LoadSharedS16
        | O::LoadSharedU32
        | O::LoadSharedU64
        | O::LoadSharedU128
        | O::WriteSharedU8
        | O::WriteSharedU16
        | O::WriteSharedU32
        | O::WriteSharedU64
        | O::WriteSharedU128 => MemorySpace::Shared,
        _ => MemorySpace::Global,
    }
}

/// The size and result shape of a constant buffer read.
const fn cbuf_read(opcode: Opcode) -> Option<(Width, bool)> {
    use Opcode as O;
    Some(match opcode {
        O::GetCbufU8 => (Width::U8, false),
        O::GetCbufS8 => (Width::S8, false),
        O::GetCbufU16 => (Width::U16, false),
        O::GetCbufS16 => (Width::S16, false),
        O::GetCbufU32 => (Width::B32, false),
        O::GetCbufF32 => (Width::B32, true),
        O::GetCbufU32x2 => (Width::B64, false),
        _ => return None,
    })
}

/// The bits of `1.0` in `format`.
fn one(format: Format) -> u64 {
    format.encode(1.0)
}

/// The sign bit of `format`.
fn sign_bit(format: Format) -> u64 {
    format.encode(-0.0)
}

impl Run<'_> {
    /// The arguments of `op`, whose id is `at`: each value read. The verifier has checked each
    /// value's type against its slot; the accessors check it again, and the type an `opaque`
    /// slot needs, as they read it.
    fn args(&self, at: Value, op: &Operation) -> Result<Args, InterpError> {
        let here = Site::Inst(at);
        let mut slots = [Err(Type::Void); MAX_ARGS];
        let mut names = [None; MAX_ARGS];
        for ((slot, name), &operand) in slots.iter_mut().zip(names.iter_mut()).zip(&op.args) {
            *slot = self.operand(here, operand)?;
            *name = operand.imm();
        }
        Ok(Args { at, slots, names })
    }

    /// Runs operation `op`, whose id is `at`, and returns its value (`None` for an operation
    /// with no result).
    pub(super) fn op(&mut self, at: Value, op: &Operation) -> Result<Option<Val>, InterpError> {
        let args = self.args(at, op)?;
        let opcode = op.opcode;
        if opcode.flags_kind() == FlagsKind::Texture {
            return self.image(&args, op);
        }
        if let Some((space, kind)) = atomic32(opcode) {
            return self.atomic(&args, space, kind).map(Some);
        }
        if let Some((width, store)) = memory_width(opcode) {
            return self.memory(&args, opcode, width, store);
        }
        if let Some((width, is_float)) = cbuf_read(opcode) {
            return self.cbuf(&args, width, is_float).map(Some);
        }
        if let Handled::Done(effect) = self.warp(&args, op)? {
            return Ok(effect);
        }
        if let Handled::Done(effect) = self.context(&args, opcode)? {
            return Ok(effect);
        }
        if let Handled::Done(effect) = self.io(&args, opcode)? {
            return Ok(effect);
        }
        if let Handled::Done(effect) = self.masks_and_offsets(&args, opcode)? {
            return Ok(effect);
        }
        if opcode == Opcode::FSwizzleAdd {
            return swizzle_add(&args, op, self.machine.system.lane_id).map(Some);
        }
        let value = match float_op(&args, op)? {
            Some(value) => Some(value),
            None => match bits_op(&args, opcode)? {
                Some(value) => Some(value),
                None => int_op(&args, opcode)?,
            },
        };
        value
            .map(Some)
            .ok_or(InterpError::Unsupported { at, opcode })
    }

    /// A constant buffer read, at the offset aligned down to its size.
    fn cbuf(&self, args: &Args, width: Width, is_float: bool) -> Result<Val, InterpError> {
        let index = args.u32(0)?;
        let offset = u64::from(args.u32(1)?);
        let size = width.size();
        let place = Place {
            at: args.at,
            space: Space::ConstantBuffer(index),
            address: offset - offset % size,
            base: 0,
        };
        let bytes = self
            .machine
            .cbufs
            .get(&index)
            .map_or(&[][..], Vec::as_slice);
        let loaded = width.loaded(place.read(bytes, size)?);
        Ok(match (is_float, loaded) {
            (true, Val::U32(bits)) => Val::F32(bits),
            (_, other) => other,
        })
    }

    /// The no-ops and the register, predicate, flag and variable operations: whether the
    /// opcode is one of them and, if so, its value.
    fn context(&mut self, args: &Args, opcode: Opcode) -> Result<Handled, InterpError> {
        use Opcode as O;
        let machine = &mut *self.machine;
        let value = match opcode {
            O::Identity => args.val(0, Type::Opaque)?,
            O::ConditionRef => Val::U1(args.u1(0)?),
            O::Void
            | O::Reference
            | O::Prologue
            | O::Epilogue
            | O::Barrier
            | O::WorkgroupMemoryBarrier
            | O::DeviceMemoryBarrier => return Ok(Handled::Done(None)),
            O::DemoteToHelperInvocation => {
                machine.demoted = true;
                return Ok(Handled::Done(None));
            }

            O::GetRegister => Val::U32(*machine.regs.get(args.reg(0)?).unwrap_or(&0)),
            O::SetRegister => {
                let value = args.u32(1)?;
                if let Some(slot) = machine.regs.get_mut(args.reg(0)?) {
                    *slot = value;
                }
                return Ok(Handled::Done(None));
            }
            O::GetPred => Val::U1(*machine.preds.get(args.pred(0)?).unwrap_or(&false)),
            O::SetPred => {
                let value = args.u1(1)?;
                if let Some(slot) = machine.preds.get_mut(args.pred(0)?) {
                    *slot = value;
                }
                return Ok(Handled::Done(None));
            }
            O::GetUniformRegister => {
                Val::U32(*machine.uniform_regs.get(args.ureg(0)?).unwrap_or(&0))
            }
            O::SetUniformRegister => {
                let value = args.u32(1)?;
                if let Some(slot) = machine.uniform_regs.get_mut(args.ureg(0)?) {
                    *slot = value;
                }
                return Ok(Handled::Done(None));
            }
            O::GetUniformPred => {
                Val::U1(*machine.uniform_preds.get(args.upred(0)?).unwrap_or(&false))
            }
            O::SetUniformPred => {
                let value = args.u1(1)?;
                if let Some(slot) = machine.uniform_preds.get_mut(args.upred(0)?) {
                    *slot = value;
                }
                return Ok(Handled::Done(None));
            }
            O::GetGotoVariable => Val::U1(
                machine
                    .goto_variables
                    .get(&args.u32(0)?)
                    .copied()
                    .unwrap_or(false),
            ),
            O::SetGotoVariable => {
                let (index, value) = (args.u32(0)?, args.u1(1)?);
                machine.goto_variables.insert(index, value);
                return Ok(Handled::Done(None));
            }
            O::GetIndirectBranchVariable => Val::U32(machine.indirect_branch),
            O::SetIndirectBranchVariable => {
                machine.indirect_branch = args.u32(0)?;
                return Ok(Handled::Done(None));
            }
            O::GetZFlag => Val::U1(machine.flags.zero),
            O::GetSFlag => Val::U1(machine.flags.sign),
            O::GetCFlag => Val::U1(machine.flags.carry),
            O::GetOFlag => Val::U1(machine.flags.overflow),
            O::SetZFlag => {
                machine.flags.zero = args.u1(0)?;
                return Ok(Handled::Done(None));
            }
            O::SetSFlag => {
                machine.flags.sign = args.u1(0)?;
                return Ok(Handled::Done(None));
            }
            O::SetCFlag => {
                machine.flags.carry = args.u1(0)?;
                return Ok(Handled::Done(None));
            }
            O::SetOFlag => {
                machine.flags.overflow = args.u1(0)?;
                return Ok(Handled::Done(None));
            }
            _ => return Ok(Handled::Not),
        };
        Ok(Handled::Done(Some(value)))
    }

    /// The attribute, patch, fragment output, system value and local memory operations: whether
    /// the opcode is one of them and, if so, its value.
    fn io(&mut self, args: &Args, opcode: Opcode) -> Result<Handled, InterpError> {
        use Opcode as O;
        let at = args.at;
        let machine = &mut *self.machine;
        let value = match opcode {
            O::GetAttribute | O::GetAttributeU32 | O::GetAttributeIndexed => {
                let attribute = if opcode == O::GetAttributeIndexed {
                    indexed_attribute(at, args.u32(0)?)?
                } else {
                    args.attribute(0)?
                };
                let vertex = args.u32(1)?;
                let bits = machine
                    .attributes
                    .get(&(attribute, vertex))
                    .copied()
                    .ok_or(InterpError::MissingInput {
                        at,
                        input: Input::Attribute { attribute, vertex },
                    })?;
                if opcode == O::GetAttributeU32 {
                    Val::U32(bits)
                } else {
                    Val::F32(bits)
                }
            }
            O::SetAttribute | O::SetAttributeIndexed => {
                let attribute = if opcode == O::SetAttributeIndexed {
                    indexed_attribute(at, args.u32(0)?)?
                } else {
                    args.attribute(0)?
                };
                let value = args.float(1, Format::F32)?;
                let vertex = args.u32(2)?;
                machine.outputs.insert((attribute, vertex), low32(value));
                return Ok(Handled::Done(None));
            }
            O::GetPatch => {
                let patch = args.patch(0)?;
                let bits =
                    machine
                        .patches
                        .get(&patch)
                        .copied()
                        .ok_or(InterpError::MissingInput {
                            at,
                            input: Input::Patch(patch),
                        })?;
                Val::F32(bits)
            }
            O::SetPatch => {
                let patch = args.patch(0)?;
                let value = args.float(1, Format::F32)?;
                machine.patches.insert(patch, low32(value));
                return Ok(Handled::Done(None));
            }
            O::SetFragColor => {
                let (target, component) = (args.u32(0)?, args.u32(1)?);
                let value = args.float(2, Format::F32)?;
                machine
                    .frag_colors
                    .insert((target, component), low32(value));
                return Ok(Handled::Done(None));
            }
            O::SetSampleMask => {
                machine.sample_mask = Some(args.u32(0)?);
                return Ok(Handled::Done(None));
            }
            O::SetFragDepth => {
                machine.frag_depth = Some(low32(args.float(0, Format::F32)?));
                return Ok(Handled::Done(None));
            }

            O::WorkgroupId => Val::U32x3(machine.system.workgroup_id),
            O::LocalInvocationId => Val::U32x3(machine.system.local_invocation_id),
            O::InvocationId => Val::U32(machine.system.invocation_id),
            O::SampleId => Val::U32(machine.system.sample_id),
            O::IsHelperInvocation => Val::U1(machine.system.is_helper || machine.demoted),
            O::YDirection => Val::F32(machine.system.y_direction),
            O::LaneId => Val::U32(machine.system.lane_id),

            O::LoadLocal => {
                let word = args.u32(0)?;
                let index = usize::try_from(word).ok();
                match index.and_then(|index| machine.local.get(index)) {
                    Some(&value) => Val::U32(value),
                    None => return Err(local_out_of_bounds(at, word, machine.local.len())),
                }
            }
            O::WriteLocal => {
                let (word, value) = (args.u32(0)?, args.u32(1)?);
                let len = machine.local.len();
                let index = usize::try_from(word).ok();
                match index.and_then(|index| machine.local.get_mut(index)) {
                    Some(slot) => *slot = value,
                    None => return Err(local_out_of_bounds(at, word, len)),
                }
                return Ok(Handled::Done(None));
            }
            _ => return Ok(Handled::Not),
        };
        Ok(Handled::Done(Some(value)))
    }

    /// The lane masks and the interpolation at an offset (see the module docs): whether the
    /// opcode is one of them and, if so, its value.
    fn masks_and_offsets(&mut self, args: &Args, opcode: Opcode) -> Result<Handled, InterpError> {
        use Opcode as O;
        let machine = &*self.machine;
        let (equal, below) = lane_masks(machine.system.lane_id);
        let value = match opcode {
            O::SubgroupEqMask => Val::U32(equal),
            O::SubgroupLtMask => Val::U32(below),
            O::SubgroupLeMask => Val::U32(below | equal),
            O::SubgroupGtMask => Val::U32(!(below | equal)),
            O::SubgroupGeMask => Val::U32(!below),
            O::InterpolateAtOffset => {
                let attribute = args.attribute(0)?;
                let offset = args.u32(1)?;
                let bits = machine
                    .interpolated
                    .get(&(attribute, offset))
                    .copied()
                    .ok_or(InterpError::MissingInput {
                        at: args.at,
                        input: Input::Interpolated { attribute, offset },
                    })?;
                Val::F32(bits)
            }
            _ => return Ok(Handled::Not),
        };
        Ok(Handled::Done(Some(value)))
    }

    /// A load from or a store to global, storage or shared memory.
    fn memory(
        &mut self,
        args: &Args,
        opcode: Opcode,
        width: Width,
        store: bool,
    ) -> Result<Option<Val>, InterpError> {
        let at = args.at;
        let size = width.size();
        let machine = &mut *self.machine;
        // The place, the bytes and the index of the stored value's argument.
        let (place, bytes, value_index): (Place, &mut [u8], usize) = match memory_space(opcode) {
            MemorySpace::Global => (
                Place {
                    at,
                    space: Space::Global,
                    address: args.u64(0)?,
                    base: machine.global.base,
                },
                &mut machine.global.bytes,
                1,
            ),
            MemorySpace::Storage => {
                let binding = args.u32(0)?;
                let bytes = machine
                    .storage
                    .get_mut(&binding)
                    .map_or(&mut [][..], Vec::as_mut_slice);
                (
                    Place {
                        at,
                        space: Space::Storage(binding),
                        address: u64::from(args.u32(1)?),
                        base: 0,
                    },
                    bytes,
                    2,
                )
            }
            MemorySpace::Shared => (
                Place {
                    at,
                    space: Space::Shared,
                    address: u64::from(args.u32(0)?),
                    base: 0,
                },
                &mut machine.shared,
                1,
            ),
        };
        if store {
            let value = width.stored(args, value_index)?;
            place.write(bytes, size, value)?;
            Ok(None)
        } else {
            Ok(Some(width.loaded(place.read(bytes, size)?)))
        }
    }

    /// A 32-bit integer atomic: the old value.
    fn atomic(
        &mut self,
        args: &Args,
        space: AtomicSpace,
        kind: Atomic,
    ) -> Result<Val, InterpError> {
        let at = args.at;
        let machine = &mut *self.machine;
        let (place, bytes, value): (Place, &mut [u8], u32) = match space {
            AtomicSpace::Shared => (
                Place {
                    at,
                    space: Space::Shared,
                    address: u64::from(args.u32(0)?),
                    base: 0,
                },
                &mut machine.shared,
                args.u32(1)?,
            ),
            AtomicSpace::Global => (
                Place {
                    at,
                    space: Space::Global,
                    address: args.u64(0)?,
                    base: machine.global.base,
                },
                &mut machine.global.bytes,
                args.u32(1)?,
            ),
            AtomicSpace::Storage => {
                let binding = args.u32(0)?;
                let bytes = machine
                    .storage
                    .get_mut(&binding)
                    .map_or(&mut [][..], Vec::as_mut_slice);
                (
                    Place {
                        at,
                        space: Space::Storage(binding),
                        address: u64::from(args.u32(1)?),
                        base: 0,
                    },
                    bytes,
                    args.u32(2)?,
                )
            }
        };
        let size = Width::B32.size();
        let old = low32(low128(place.read(bytes, size)?));
        place.write(bytes, size, u128::from(kind.apply(old, value)))?;
        Ok(Val::U32(old))
    }

    /// A texture or image operation: checked against the next scripted call, whose reply it
    /// returns.
    fn image(&mut self, args: &Args, op: &Operation) -> Result<Option<Val>, InterpError> {
        let at = args.at;
        let opcode = op.opcode;
        let info = match op.flags {
            Flags::Texture(info) => info,
            Flags::None | Flags::Fp(_) => TextureInfo::default(),
        };
        let mut values = Vec::with_capacity(op.args.len());
        for index in 0..op.args.len() {
            values.push(args.val(
                index,
                opcode.args().get(index).copied().unwrap_or(Type::Void),
            )?);
        }
        let call = ImageCall {
            opcode,
            info,
            args: values,
        };
        let index = self.image_index;
        let Some(scripted) = self.machine.images.front() else {
            return Err(InterpError::ImageScriptEnded { at, opcode });
        };
        if scripted.call != call {
            return Err(InterpError::ImageMismatch { at, opcode, index });
        }
        let result = opcode.result();
        let reply = scripted.reply;
        let fits = match reply {
            None => result == Type::Void,
            Some(value) => value.ty() == result,
        };
        if !fits {
            return Err(InterpError::ImageReply { at, opcode, index });
        }
        self.machine.images.pop_front();
        self.image_index += 1;
        Ok(reply)
    }
}

/// The float operations: `None` for any other opcode.
fn float_op(args: &Args, op: &Operation) -> Result<Option<Val>, InterpError> {
    use Opcode as O;
    let opcode = op.opcode;
    let control = match op.flags {
        Flags::Fp(control) => Control::of(control),
        Flags::None | Flags::Texture(_) => Control::NONE,
    };
    let directed = |_: DirectedF64| InterpError::DirectedRounding {
        at: args.at,
        opcode,
    };
    if let Some((format, how, ordered)) = fp_compare(opcode) {
        let (a, b) = (args.float(0, format)?, args.float(1, format)?);
        let result = float::compare(format, a, b, how, ordered, control);
        return Ok(Some(Val::U1(result)));
    }
    if let Some((format, unary)) = float_unary(opcode) {
        let a = args.float(0, format)?;
        return Ok(Some(match unary {
            FloatUnary::Abs => float_val(format, a & !sign_bit(format)),
            FloatUnary::Neg => float_val(format, a ^ sign_bit(format)),
            FloatUnary::Saturate => float_val(format, float::clamp(format, a, 0, one(format))),
            FloatUnary::Integral(how) => {
                float_val(format, float::integral(format, a, how, control))
            }
            FloatUnary::IsNan => Val::U1(format.is_nan(a)),
        }));
    }
    if let Some((format, arith)) = float_arith(opcode) {
        let (a, b) = (args.float(0, format)?, args.float(1, format)?);
        let bits = match arith {
            FloatArith::Add => float::add(format, a, b, control).map_err(directed)?,
            FloatArith::Mul => float::mul(format, a, b, control).map_err(directed)?,
            FloatArith::Fma => {
                let c = args.float(2, format)?;
                float::fma(format, a, b, c, control).map_err(directed)?
            }
            FloatArith::Max => float::min_max(format, a, b, control, true),
            FloatArith::Min => float::min_max(format, a, b, control, false),
            FloatArith::Clamp => float::clamp(format, a, b, args.float(2, format)?),
        };
        return Ok(Some(float_val(format, bits)));
    }
    if let Some((from, to)) = float_to_float(opcode) {
        let bits = float::convert(from, to, args.float(0, from)?, control);
        return Ok(Some(float_val(to, bits)));
    }
    if let Some((source, to)) = int_to_float(opcode) {
        let value = source.read(args)?;
        return Ok(Some(float_val(to, float::from_integer(to, value, control))));
    }
    if let Some((from, dest)) = float_to_int(opcode) {
        let (min, max) = dest.range();
        let value = float::to_integer(from, args.float(0, from)?, min, max);
        return Ok(Some(dest.result(value)));
    }
    Ok(Some(match opcode {
        O::FPRecip32 => f32_fn(args, |x| 1.0 / x)?,
        O::FPRecip64 => f64_fn(args, |x| 1.0 / x)?,
        O::FPRecipSqrt32 => f32_fn(args, |x| 1.0 / x.sqrt())?,
        O::FPRecipSqrt64 => f64_fn(args, |x| 1.0 / x.sqrt())?,
        O::FPSqrt => f32_fn(args, f64::sqrt)?,
        O::FPSin => f32_fn(args, f64::sin)?,
        O::FPCos => f32_fn(args, f64::cos)?,
        O::FPExp2 => f32_fn(args, f64::exp2)?,
        O::FPLog2 => f32_fn(args, f64::log2)?,
        _ => return Ok(None),
    }))
}

/// The composite, select, bit cast and packing operations: `None` for any other opcode.
fn bits_op(args: &Args, opcode: Opcode) -> Result<Option<Val>, InterpError> {
    use Opcode as O;
    let at = args.at;
    if let Some(shape) = vector_shape(opcode.result())
        && let Some(value) = composite(args, opcode, shape)?
    {
        return Ok(Some(value));
    }
    Ok(Some(match opcode {
        O::SelectU1
        | O::SelectU8
        | O::SelectU16
        | O::SelectU32
        | O::SelectU64
        | O::SelectF16
        | O::SelectF32
        | O::SelectF64 => {
            let wanted = opcode.result();
            if args.u1(0)? {
                args.val(1, wanted)?
            } else {
                args.val(2, wanted)?
            }
        }
        O::BitCastU16F16 => Val::U16(low16(args.float(0, Format::F16)?)),
        O::BitCastU32F32 => Val::U32(low32(args.float(0, Format::F32)?)),
        O::BitCastU64F64 => Val::U64(args.float(0, Format::F64)?),
        O::BitCastF16U16 => Val::F16(args.u16(0)?),
        O::BitCastF32U32 => Val::F32(args.u32(0)?),
        O::BitCastF64U64 => Val::F64(args.u64(0)?),
        O::PackUint2x32 => {
            let [low, high, ..] = args.vector(0, Type::U32x2)?;
            Val::U64(low | (high << 32))
        }
        O::UnpackUint2x32 => {
            let value = args.u64(0)?;
            Val::U32x2([low32(value), low32(value >> 32)])
        }
        O::PackFloat2x16 => {
            let [low, high, ..] = args.vector(0, Type::F16x2)?;
            Val::U32(low32(low | (high << 16)))
        }
        O::UnpackFloat2x16 => {
            let value = u64::from(args.u32(0)?);
            Val::F16x2([low16(value), low16(value >> 16)])
        }
        O::PackHalf2x16 => {
            let [low, high, ..] = args.vector(0, Type::F32x2)?;
            let half = |bits: u64| float::convert(Format::F32, Format::F16, bits, Control::NONE);
            Val::U32(low32(half(low) | (half(high) << 16)))
        }
        O::UnpackHalf2x16 => {
            let value = u64::from(args.u32(0)?);
            let single = |bits: u64| float::convert(Format::F16, Format::F32, bits, Control::NONE);
            Val::F32x2([low32(single(value & 0xffff)), low32(single(value >> 16))])
        }
        O::PackDouble2x32 => {
            let [low, high, ..] = args.vector(0, Type::U32x2)?;
            Val::F64(low | (high << 32))
        }
        O::UnpackDouble2x32 => {
            let value = args.float(0, Format::F64)?;
            Val::U32x2([low32(value), low32(value >> 32)])
        }

        O::CompositeExtractU32x2
        | O::CompositeExtractU32x3
        | O::CompositeExtractU32x4
        | O::CompositeExtractF16x2
        | O::CompositeExtractF16x3
        | O::CompositeExtractF16x4
        | O::CompositeExtractF32x2
        | O::CompositeExtractF32x3
        | O::CompositeExtractF32x4
        | O::CompositeExtractF64x2
        | O::CompositeExtractF64x3
        | O::CompositeExtractF64x4 => {
            let ty = opcode.args().first().copied().unwrap_or(Type::Void);
            let Some((lane, len)) = vector_shape(ty) else {
                return Err(InterpError::Unsupported { at, opcode });
            };
            let bits = args.vector(0, ty)?;
            let index = args.u32(1)?;
            let component = usize::try_from(index)
                .ok()
                .filter(|&slot| slot < len)
                .and_then(|slot| bits.get(slot))
                .ok_or(InterpError::Component { at, index, ty })?;
            lane.scalar(*component)
        }
        _ => return Ok(None),
    }))
}

/// The integer, logic and undefined-value operations: `None` for any other opcode.
fn int_op(args: &Args, opcode: Opcode) -> Result<Option<Val>, InterpError> {
    use Opcode as O;
    let at = args.at;
    if let Some(how) = int_compare(opcode) {
        let (a, b) = (args.u32(0)?, args.u32(1)?);
        return Ok(Some(Val::U1(how.eval(a, b))));
    }
    Ok(Some(match opcode {
        O::UndefU1 | O::UndefU8 | O::UndefU16 | O::UndefU32 | O::UndefU64 => {
            match Val::zero(opcode.result()) {
                Some(zero) => zero,
                None => return Err(InterpError::Unsupported { at, opcode }),
            }
        }

        O::IAdd32 => Val::U32(args.u32(0)?.wrapping_add(args.u32(1)?)),
        O::ISub32 => Val::U32(args.u32(0)?.wrapping_sub(args.u32(1)?)),
        O::IMul32 => Val::U32(args.u32(0)?.wrapping_mul(args.u32(1)?)),
        O::SDiv32 => Val::U32(integer::sdiv32(args.u32(0)?, args.u32(1)?)),
        O::UDiv32 => Val::U32(integer::udiv32(args.u32(0)?, args.u32(1)?)),
        O::INeg32 => Val::U32(args.u32(0)?.wrapping_neg()),
        O::IAbs32 => Val::U32(args.u32(0)?.cast_signed().wrapping_abs().cast_unsigned()),
        O::ShiftLeftLogical32 => Val::U32(integer::shl32(args.u32(0)?, args.u32(1)?)),
        O::ShiftRightLogical32 => Val::U32(integer::shr32(args.u32(0)?, args.u32(1)?)),
        O::ShiftRightArithmetic32 => Val::U32(integer::sar32(args.u32(0)?, args.u32(1)?)),
        O::BitwiseAnd32 => Val::U32(args.u32(0)? & args.u32(1)?),
        O::BitwiseOr32 => Val::U32(args.u32(0)? | args.u32(1)?),
        O::BitwiseXor32 => Val::U32(args.u32(0)? ^ args.u32(1)?),
        O::BitFieldInsert => Val::U32(integer::bitfield_insert(
            args.u32(0)?,
            args.u32(1)?,
            args.u32(2)?,
            args.u32(3)?,
        )),
        O::BitFieldSExtract => Val::U32(integer::bitfield_sextract(
            args.u32(0)?,
            args.u32(1)?,
            args.u32(2)?,
        )),
        O::BitFieldUExtract => Val::U32(integer::bitfield_uextract(
            args.u32(0)?,
            args.u32(1)?,
            args.u32(2)?,
        )),
        O::BitReverse32 => Val::U32(args.u32(0)?.reverse_bits()),
        O::BitCount32 => Val::U32(args.u32(0)?.count_ones()),
        O::BitwiseNot32 => Val::U32(!args.u32(0)?),
        O::FindSMsb32 => Val::U32(integer::find_smsb(args.u32(0)?)),
        O::FindUMsb32 => Val::U32(integer::find_umsb(args.u32(0)?)),
        O::SMin32 => Val::U32(
            args.u32(0)?
                .cast_signed()
                .min(args.u32(1)?.cast_signed())
                .cast_unsigned(),
        ),
        O::UMin32 => Val::U32(args.u32(0)?.min(args.u32(1)?)),
        O::SMax32 => Val::U32(
            args.u32(0)?
                .cast_signed()
                .max(args.u32(1)?.cast_signed())
                .cast_unsigned(),
        ),
        O::UMax32 => Val::U32(args.u32(0)?.max(args.u32(1)?)),
        O::SClamp32 => {
            let (value, low, high) = (
                args.u32(0)?.cast_signed(),
                args.u32(1)?.cast_signed(),
                args.u32(2)?.cast_signed(),
            );
            Val::U32(value.max(low).min(high).cast_unsigned())
        }
        O::UClamp32 => {
            let (value, low, high) = (args.u32(0)?, args.u32(1)?, args.u32(2)?);
            Val::U32(value.max(low).min(high))
        }

        O::IAdd64 => Val::U64(args.u64(0)?.wrapping_add(args.u64(1)?)),
        O::ISub64 => Val::U64(args.u64(0)?.wrapping_sub(args.u64(1)?)),
        O::INeg64 => Val::U64(args.u64(0)?.wrapping_neg()),
        O::IAbs64 => Val::U64(args.u64(0)?.cast_signed().wrapping_abs().cast_unsigned()),
        O::ShiftLeftLogical64 => Val::U64(integer::shl64(args.u64(0)?, args.u32(1)?)),
        O::ShiftRightLogical64 => Val::U64(integer::shr64(args.u64(0)?, args.u32(1)?)),
        O::ShiftRightArithmetic64 => Val::U64(integer::sar64(args.u64(0)?, args.u32(1)?)),

        O::LogicalOr => Val::U1(args.u1(0)? | args.u1(1)?),
        O::LogicalAnd => Val::U1(args.u1(0)? & args.u1(1)?),
        O::LogicalXor => Val::U1(args.u1(0)? ^ args.u1(1)?),
        O::LogicalNot => Val::U1(!args.u1(0)?),

        O::ConvertU64U32 => Val::U64(u64::from(args.u32(0)?)),
        O::ConvertU32U64 => Val::U32(low32(args.u64(0)?)),
        O::ConvertU16U32 => Val::U16(low16(u64::from(args.u32(0)?))),
        O::ConvertU32U16 => Val::U32(u32::from(args.u16(0)?)),
        O::ConvertU8U32 => Val::U8(args.u32(0)?.to_le_bytes()[0]),
        O::ConvertU32U8 => Val::U32(u32::from(args.u8(0)?)),
        O::ConvertS32S8 => Val::U32(i32::from(args.u8(0)?.cast_signed()).cast_unsigned()),
        O::ConvertS32S16 => Val::U32(i32::from(args.u16(0)?.cast_signed()).cast_unsigned()),
        _ => return Ok(None),
    }))
}

/// The composite construct and insert operations, whose result is the vector type of shape
/// `(lane, len)`; `None` for any other opcode with a vector result.
fn composite(
    args: &Args,
    opcode: Opcode,
    (lane, len): (Lane, usize),
) -> Result<Option<Val>, InterpError> {
    use Opcode as O;
    let result = opcode.result();
    let component = |index: usize| -> Result<u64, InterpError> {
        let value = args.val(index, lane.ty())?;
        lane.bits_of(value)
            .ok_or_else(|| args.mismatch(index, lane.ty(), value.ty()))
    };
    let bits = match opcode {
        O::CompositeConstructU32x2
        | O::CompositeConstructU32x3
        | O::CompositeConstructU32x4
        | O::CompositeConstructF16x2
        | O::CompositeConstructF16x3
        | O::CompositeConstructF16x4
        | O::CompositeConstructF32x2
        | O::CompositeConstructF32x3
        | O::CompositeConstructF32x4
        | O::CompositeConstructF64x2
        | O::CompositeConstructF64x3
        | O::CompositeConstructF64x4 => {
            let mut bits = [0; 4];
            for (index, slot) in bits.iter_mut().enumerate().take(len) {
                *slot = component(index)?;
            }
            bits
        }
        O::CompositeInsertU32x2
        | O::CompositeInsertU32x3
        | O::CompositeInsertU32x4
        | O::CompositeInsertF16x2
        | O::CompositeInsertF16x3
        | O::CompositeInsertF16x4
        | O::CompositeInsertF32x2
        | O::CompositeInsertF32x3
        | O::CompositeInsertF32x4
        | O::CompositeInsertF64x2
        | O::CompositeInsertF64x3
        | O::CompositeInsertF64x4 => {
            let mut bits = args.vector(0, result)?;
            let index = args.u32(2)?;
            let slot = usize::try_from(index)
                .ok()
                .filter(|&slot| slot < len)
                .and_then(|slot| bits.get_mut(slot))
                .ok_or(InterpError::Component {
                    at: args.at,
                    index,
                    ty: result,
                })?;
            *slot = component(1)?;
            bits
        }
        _ => return Ok(None),
    };
    Ok(Val::from_lanes(lane, bits, len))
}

/// The attribute word an indexed attribute access at byte `offset` names: `offset >> 2`, as
/// Eden's `make_load` computes it.
fn indexed_attribute(at: Value, offset: u32) -> Result<crate::ir::Attribute, InterpError> {
    u16::try_from(offset >> 2)
        .map(crate::ir::Attribute)
        .map_err(|_| InterpError::AttributeOffset { at, offset })
}

/// The masks of a warp of 32 lanes with the lanes equal to `lane` and below it: bit `n` for
/// lane `n` (see the module docs). A lane of 32 or more equals no lane and is above them all.
fn lane_masks(lane: u32) -> (u32, u32) {
    let equal = 1_u32.checked_shl(lane).unwrap_or(0);
    // For a lane below 32, the bits under its own; past the word, every bit (0 - 1 wraps).
    (equal, equal.wrapping_sub(1))
}

/// `FSwizzleAdd`'s multipliers of `a` and of `b`, by the lane's two bits of the swizzle: Eden's
/// `fswzadd_lut_a` and `fswzadd_lut_b` (`spirv_emit_context.cpp` @ cf654052: `-1, 1, -1, 0` and
/// `-1, -1, 1, -1`), as binary32 bits.
pub(super) const SWIZZLE_ADD_FACTORS: [(u32, u32); 4] = [
    (0xbf80_0000, 0xbf80_0000),
    (0x3f80_0000, 0xbf80_0000),
    (0xbf80_0000, 0x3f80_0000),
    (0x0000_0000, 0xbf80_0000),
];

/// `FSwizzleAdd a, b, swizzle` (Eden's `EmitFSwizzleAdd`, `emit_spirv_warp.cpp`): the lane's
/// two bits of the swizzle, at `(lane_id & 3) * 2`, pick a multiplier for each operand, and the
/// products are added. Only the lane id is read, not other lanes' values. The operation's
/// controls apply to the two multiplications and the addition (Eden's emitter writes plain
/// `OpFMul` and `OpFAdd`; multiplying by `1`, `-1` or `0` is exact in every mode).
fn swizzle_add(args: &Args, op: &Operation, lane_id: u32) -> Result<Val, InterpError> {
    let control = match op.flags {
        Flags::Fp(control) => Control::of(control),
        Flags::None | Flags::Texture(_) => Control::NONE,
    };
    let directed = |_: DirectedF64| InterpError::DirectedRounding {
        at: args.at,
        opcode: op.opcode,
    };
    let (a, b) = (args.float(0, Format::F32)?, args.float(1, Format::F32)?);
    let swizzle = args.u32(2)?;
    let shift = (lane_id & 3) * 2;
    let index = (swizzle >> shift) & 3;
    let (factor_a, factor_b) = usize::try_from(index)
        .ok()
        .and_then(|slot| SWIZZLE_ADD_FACTORS.get(slot))
        .copied()
        .unwrap_or_default();
    let a = float::mul(Format::F32, a, u64::from(factor_a), control).map_err(directed)?;
    let b = float::mul(Format::F32, b, u64::from(factor_b), control).map_err(directed)?;
    let sum = float::add(Format::F32, a, b, control).map_err(directed)?;
    Ok(float_val(Format::F32, sum))
}

/// A binary32 function of argument 0 (see [`float::unary_f32`]).
fn f32_fn(args: &Args, f: impl FnOnce(f64) -> f64) -> Result<Val, InterpError> {
    let bits = float::unary_f32(args.float(0, Format::F32)?, f);
    Ok(Val::F32(low32(bits)))
}

/// A binary64 function of argument 0 (see [`float::unary_f64`]).
fn f64_fn(args: &Args, f: impl FnOnce(f64) -> f64) -> Result<Val, InterpError> {
    Ok(Val::F64(float::unary_f64(args.float(0, Format::F64)?, f)))
}

/// Bytes per local memory word: local memory is an array of 32-bit words, indexed by word
/// (Eden's `EmitLoadLocal`, `local_memory`).
const LOCAL_WORD_BYTES: u64 = 4;

/// The out-of-bounds error of local word `word` in local memory of `len` words, in bytes.
fn local_out_of_bounds(at: Value, word: u32, len: usize) -> InterpError {
    let bytes = u64::try_from(len)
        .unwrap_or(u64::MAX)
        .saturating_mul(LOCAL_WORD_BYTES);
    InterpError::OutOfBounds {
        at,
        space: Space::Local,
        offset: u64::from(word) * LOCAL_WORD_BYTES,
        size: LOCAL_WORD_BYTES,
        len: bytes,
    }
}
