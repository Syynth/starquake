//! One IR operation as naga expressions and statements (the parent module's docs,
//! "Operations").
//!
//! Each arm builds the expression tree its semantics need; where naga's (WGSL's) definition
//! leaves an input undefined or differs from the interpreter's ([`crate::interp`],
//! "Semantics"), the arm selects or clamps explicitly. The edge rules, with the interpreter
//! function each mirrors (`interp/integer.rs`, `interp/float.rs`):
//!
//! - Shifts: the amount is clamped to `width - 1` before naga's shift, then a logical shift
//!   by the width or more selects 0 (`shl32`, `shr32`, `shl64`, `shr64`); an arithmetic
//!   shift by `width - 1` already fills with the sign (`sar32`, `sar64`).
//! - Division: a zero divisor is replaced by 1 before naga's division, and the result by
//!   `0xffff_ffff` (`udiv32`, `sdiv32`); naga defines `i32::MIN / -1` as `i32::MIN`, as the
//!   interpreter does.
//! - Bit fields: offset `o = min(offset, 32)` and count `c = min(count, 32 - o)`, the
//!   interpreter's clamps written so naga's `extractBits` and `insertBits` see only fields
//!   inside the word (`bitfield_uextract`, `bitfield_insert`). `BitFieldSExtract` shifts the
//!   unsigned field up by `32 - min(count, 32)` and arithmetically back, so the sign is the
//!   top bit of the field the count names, and a zero count gives 0 (`bitfield_sextract`).
//! - Float to integer: naga's conversion truncates and saturates; a NaN selects 0.

use naga::{
    BinaryOperator, Block, Expression, Handle, Literal, MathFunction, RelationalFunction,
    ScalarKind, Statement, UnaryOperator,
};

use super::builder::push;
use super::{
    AttributeKind, LowerError, Lowering, Var, attribute_arg, classify, directed, imm_arg,
};
use crate::ir::{Flags, Imm, Opcode, Operation, Type, Value};

/// The bits of a 32-bit integer.
const BITS32: u32 = u32::BITS;
/// The bits of a 64-bit integer.
const BITS64: u32 = u64::BITS;
/// The bytes of a 16-, 32- and 64-bit scalar, as naga's conversion widths.
const WIDTH16: u8 = 2;
const WIDTH32: u8 = 4;
const WIDTH64: u8 = 8;

/// How a shift fills.
#[derive(Clone, Copy)]
enum Shift {
    Left,
    Logical,
    Arithmetic,
}

/// A float comparison: the naga operator and whether the comparison is unordered.
fn float_comparison(opcode: Opcode) -> Option<(BinaryOperator, bool)> {
    use BinaryOperator::{Equal, Greater, GreaterEqual, Less, LessEqual, NotEqual};
    Some(match opcode {
        Opcode::FPOrdEqual16 | Opcode::FPOrdEqual32 => (Equal, false),
        Opcode::FPUnordEqual16 | Opcode::FPUnordEqual32 => (Equal, true),
        Opcode::FPOrdNotEqual16 | Opcode::FPOrdNotEqual32 => (NotEqual, false),
        Opcode::FPUnordNotEqual16 | Opcode::FPUnordNotEqual32 => (NotEqual, true),
        Opcode::FPOrdLessThan16 | Opcode::FPOrdLessThan32 => (Less, false),
        Opcode::FPUnordLessThan16 | Opcode::FPUnordLessThan32 => (Less, true),
        Opcode::FPOrdGreaterThan16 | Opcode::FPOrdGreaterThan32 => (Greater, false),
        Opcode::FPUnordGreaterThan16 | Opcode::FPUnordGreaterThan32 => (Greater, true),
        Opcode::FPOrdLessThanEqual16 | Opcode::FPOrdLessThanEqual32 => (LessEqual, false),
        Opcode::FPUnordLessThanEqual16 | Opcode::FPUnordLessThanEqual32 => (LessEqual, true),
        Opcode::FPOrdGreaterThanEqual16 | Opcode::FPOrdGreaterThanEqual32 => {
            (GreaterEqual, false)
        }
        Opcode::FPUnordGreaterThanEqual16 | Opcode::FPUnordGreaterThanEqual32 => {
            (GreaterEqual, true)
        }
        _ => return None,
    })
}

/// An integer comparison: the naga operator and whether its operands are signed.
fn integer_comparison(opcode: Opcode) -> Option<(BinaryOperator, bool)> {
    use BinaryOperator::{Equal, Greater, GreaterEqual, Less, LessEqual, NotEqual};
    Some(match opcode {
        Opcode::IEqual => (Equal, false),
        Opcode::INotEqual => (NotEqual, false),
        Opcode::SLessThan => (Less, true),
        Opcode::ULessThan => (Less, false),
        Opcode::SLessThanEqual => (LessEqual, true),
        Opcode::ULessThanEqual => (LessEqual, false),
        Opcode::SGreaterThan => (Greater, true),
        Opcode::UGreaterThan => (Greater, false),
        Opcode::SGreaterThanEqual => (GreaterEqual, true),
        Opcode::UGreaterThanEqual => (GreaterEqual, false),
        _ => return None,
    })
}

/// A float operation that is one naga math function of its arguments.
fn float_math(opcode: Opcode) -> Option<MathFunction> {
    Some(match opcode {
        Opcode::FPAbs16 | Opcode::FPAbs32 => MathFunction::Abs,
        Opcode::FPSaturate16 | Opcode::FPSaturate32 => MathFunction::Saturate,
        Opcode::FPRecipSqrt32 => MathFunction::InverseSqrt,
        Opcode::FPSqrt => MathFunction::Sqrt,
        Opcode::FPSin => MathFunction::Sin,
        Opcode::FPCos => MathFunction::Cos,
        Opcode::FPExp2 => MathFunction::Exp2,
        Opcode::FPLog2 => MathFunction::Log2,
        // naga's `round` rounds half to even (WGSL's `round`).
        Opcode::FPRoundEven16 | Opcode::FPRoundEven32 => MathFunction::Round,
        Opcode::FPFloor16 | Opcode::FPFloor32 => MathFunction::Floor,
        Opcode::FPCeil16 | Opcode::FPCeil32 => MathFunction::Ceil,
        Opcode::FPTrunc16 | Opcode::FPTrunc32 => MathFunction::Trunc,
        _ => return None,
    })
}

/// A two-operand operation that is one naga binary operator (wrapping on integers).
fn plain_binary(opcode: Opcode) -> Option<BinaryOperator> {
    Some(match opcode {
        Opcode::FPAdd16 | Opcode::FPAdd32 | Opcode::IAdd32 | Opcode::IAdd64 => BinaryOperator::Add,
        Opcode::FPMul16 | Opcode::FPMul32 | Opcode::IMul32 => BinaryOperator::Multiply,
        Opcode::ISub32 | Opcode::ISub64 => BinaryOperator::Subtract,
        Opcode::BitwiseAnd32 => BinaryOperator::And,
        Opcode::BitwiseOr32 => BinaryOperator::InclusiveOr,
        Opcode::BitwiseXor32 => BinaryOperator::ExclusiveOr,
        Opcode::LogicalOr => BinaryOperator::LogicalOr,
        Opcode::LogicalAnd => BinaryOperator::LogicalAnd,
        // `!=` on booleans is their exclusive or.
        Opcode::LogicalXor => BinaryOperator::NotEqual,
        _ => return None,
    })
}

/// The vector type and size of a composite operation, and whether it constructs (else extracts
/// or inserts).
fn composite(opcode: Opcode) -> Option<(Type, u32, CompositeOp)> {
    use CompositeOp::{Construct, Extract, Insert};
    Some(match opcode {
        Opcode::CompositeConstructU32x2 => (Type::U32x2, 2, Construct),
        Opcode::CompositeConstructU32x3 => (Type::U32x3, 3, Construct),
        Opcode::CompositeConstructU32x4 => (Type::U32x4, 4, Construct),
        Opcode::CompositeExtractU32x2 => (Type::U32x2, 2, Extract),
        Opcode::CompositeExtractU32x3 => (Type::U32x3, 3, Extract),
        Opcode::CompositeExtractU32x4 => (Type::U32x4, 4, Extract),
        Opcode::CompositeInsertU32x2 => (Type::U32x2, 2, Insert),
        Opcode::CompositeInsertU32x3 => (Type::U32x3, 3, Insert),
        Opcode::CompositeInsertU32x4 => (Type::U32x4, 4, Insert),
        Opcode::CompositeConstructF16x2 => (Type::F16x2, 2, Construct),
        Opcode::CompositeConstructF16x3 => (Type::F16x3, 3, Construct),
        Opcode::CompositeConstructF16x4 => (Type::F16x4, 4, Construct),
        Opcode::CompositeExtractF16x2 => (Type::F16x2, 2, Extract),
        Opcode::CompositeExtractF16x3 => (Type::F16x3, 3, Extract),
        Opcode::CompositeExtractF16x4 => (Type::F16x4, 4, Extract),
        Opcode::CompositeInsertF16x2 => (Type::F16x2, 2, Insert),
        Opcode::CompositeInsertF16x3 => (Type::F16x3, 3, Insert),
        Opcode::CompositeInsertF16x4 => (Type::F16x4, 4, Insert),
        Opcode::CompositeConstructF32x2 => (Type::F32x2, 2, Construct),
        Opcode::CompositeConstructF32x3 => (Type::F32x3, 3, Construct),
        Opcode::CompositeConstructF32x4 => (Type::F32x4, 4, Construct),
        Opcode::CompositeExtractF32x2 => (Type::F32x2, 2, Extract),
        Opcode::CompositeExtractF32x3 => (Type::F32x3, 3, Extract),
        Opcode::CompositeExtractF32x4 => (Type::F32x4, 4, Extract),
        Opcode::CompositeInsertF32x2 => (Type::F32x2, 2, Insert),
        Opcode::CompositeInsertF32x3 => (Type::F32x3, 3, Insert),
        Opcode::CompositeInsertF32x4 => (Type::F32x4, 4, Insert),
        _ => return None,
    })
}

/// What a composite operation does.
#[derive(Clone, Copy, PartialEq, Eq)]
enum CompositeOp {
    Construct,
    Extract,
    Insert,
}

/// An integer-to-float conversion: the float width, and the source's bits and signedness.
fn int_to_float(opcode: Opcode) -> Option<(u8, u32, bool)> {
    Some(match opcode {
        Opcode::ConvertF16S8 => (WIDTH16, 8, true),
        Opcode::ConvertF16S16 => (WIDTH16, 16, true),
        Opcode::ConvertF16S32 => (WIDTH16, BITS32, true),
        Opcode::ConvertF16U8 => (WIDTH16, 8, false),
        Opcode::ConvertF16U16 => (WIDTH16, 16, false),
        Opcode::ConvertF16U32 => (WIDTH16, BITS32, false),
        Opcode::ConvertF32S8 => (WIDTH32, 8, true),
        Opcode::ConvertF32S16 => (WIDTH32, 16, true),
        Opcode::ConvertF32S32 => (WIDTH32, BITS32, true),
        Opcode::ConvertF32U8 => (WIDTH32, 8, false),
        Opcode::ConvertF32U16 => (WIDTH32, 16, false),
        Opcode::ConvertF32U32 => (WIDTH32, BITS32, false),
        _ => return None,
    })
}

/// A float-to-integer conversion: whether the source is f16, and the result's bits and
/// signedness (16-bit results are extended into the `u32`).
fn float_to_int(opcode: Opcode) -> Option<(bool, u32, bool)> {
    Some(match opcode {
        Opcode::ConvertS16F16 => (true, 16, true),
        Opcode::ConvertS16F32 => (false, 16, true),
        Opcode::ConvertS32F16 => (true, BITS32, true),
        Opcode::ConvertS32F32 => (false, BITS32, true),
        Opcode::ConvertU16F16 => (true, 16, false),
        Opcode::ConvertU16F32 => (false, 16, false),
        Opcode::ConvertU32F16 => (true, BITS32, false),
        Opcode::ConvertU32F32 => (false, BITS32, false),
        _ => return None,
    })
}

impl Lowering {
    pub(super) fn lit(&mut self, body: &mut Block, literal: Literal) -> Handle<Expression> {
        self.expr(body, Expression::Literal(literal))
    }

    fn binary(
        &mut self,
        body: &mut Block,
        op: BinaryOperator,
        left: Handle<Expression>,
        right: Handle<Expression>,
    ) -> Handle<Expression> {
        self.expr(body, Expression::Binary { op, left, right })
    }

    fn unary(
        &mut self,
        body: &mut Block,
        op: UnaryOperator,
        expr: Handle<Expression>,
    ) -> Handle<Expression> {
        self.expr(body, Expression::Unary { op, expr })
    }

    /// `fun(arg, rest...)`.
    pub(super) fn math(
        &mut self,
        body: &mut Block,
        fun: MathFunction,
        arg: Handle<Expression>,
        rest: &[Handle<Expression>],
    ) -> Handle<Expression> {
        let mut rest = rest.iter().copied();
        let (arg1, arg2, arg3) = (rest.next(), rest.next(), rest.next());
        self.expr(
            body,
            Expression::Math {
                fun,
                arg,
                arg1,
                arg2,
                arg3,
            },
        )
    }

    /// Reinterprets (`convert` `None`) or converts (`Some(width)`) `expr` to `kind`.
    pub(super) fn cast(
        &mut self,
        body: &mut Block,
        expr: Handle<Expression>,
        kind: ScalarKind,
        convert: Option<u8>,
    ) -> Handle<Expression> {
        self.expr(
            body,
            Expression::As {
                expr,
                kind,
                convert,
            },
        )
    }

    /// `condition ? accept : reject`.
    pub(super) fn select(
        &mut self,
        body: &mut Block,
        condition: Handle<Expression>,
        accept: Handle<Expression>,
        reject: Handle<Expression>,
    ) -> Handle<Expression> {
        self.expr(
            body,
            Expression::Select {
                condition,
                accept,
                reject,
            },
        )
    }

    fn is_nan(&mut self, body: &mut Block, argument: Handle<Expression>) -> Handle<Expression> {
        self.expr(
            body,
            Expression::Relational {
                fun: RelationalFunction::IsNan,
                argument,
            },
        )
    }

    pub(super) fn component(
        &mut self,
        body: &mut Block,
        base: Handle<Expression>,
        index: u32,
    ) -> Handle<Expression> {
        self.expr(body, Expression::AccessIndex { base, index })
    }

    /// `max` (or `min`) that returns the operand that is not a NaN.
    fn nan_aware(
        &mut self,
        body: &mut Block,
        fun: MathFunction,
        a: Handle<Expression>,
        b: Handle<Expression>,
    ) -> Handle<Expression> {
        let both = self.math(body, fun, a, &[b]);
        let a_nan = self.is_nan(body, a);
        let picked = self.select(body, a_nan, b, both);
        let b_nan = self.is_nan(body, b);
        self.select(body, b_nan, a, picked)
    }

    /// A shift of a `bits`-wide integer by a `u32` amount (see the module docs).
    fn shift(
        &mut self,
        body: &mut Block,
        kind: Shift,
        value: Handle<Expression>,
        amount: Handle<Expression>,
        bits: u32,
    ) -> Handle<Expression> {
        let top = self.lit(body, Literal::U32(bits - 1));
        let clamped = self.math(body, MathFunction::Min, amount, &[top]);
        match kind {
            Shift::Arithmetic => {
                let signed = self.cast(body, value, ScalarKind::Sint, None);
                let shifted = self.binary(body, BinaryOperator::ShiftRight, signed, clamped);
                self.cast(body, shifted, ScalarKind::Uint, None)
            }
            Shift::Left | Shift::Logical => {
                let op = if matches!(kind, Shift::Left) {
                    BinaryOperator::ShiftLeft
                } else {
                    BinaryOperator::ShiftRight
                };
                let shifted = self.binary(body, op, value, clamped);
                let width = self.lit(body, Literal::U32(bits));
                let out = self.binary(body, BinaryOperator::GreaterEqual, amount, width);
                let zero = self.zero_int(body, bits);
                self.select(body, out, zero, shifted)
            }
        }
    }

    /// Zero of a `bits`-wide unsigned integer.
    fn zero_int(&mut self, body: &mut Block, bits: u32) -> Handle<Expression> {
        let literal = if bits == BITS64 {
            Literal::U64(0)
        } else {
            Literal::U32(0)
        };
        self.lit(body, literal)
    }

    /// The bit field offset and count clamped as the module docs say: `min(offset, 32)` and
    /// `min(count, 32 - that)`.
    fn field(
        &mut self,
        body: &mut Block,
        offset: Handle<Expression>,
        count: Handle<Expression>,
    ) -> (Handle<Expression>, Handle<Expression>) {
        let width = self.lit(body, Literal::U32(BITS32));
        let offset = self.math(body, MathFunction::Min, offset, &[width]);
        let room = self.binary(body, BinaryOperator::Subtract, width, offset);
        let count = self.math(body, MathFunction::Min, count, &[room]);
        (offset, count)
    }

    /// The low `bits` of `value`, sign- or zero-extended to 32.
    fn low_bits(
        &mut self,
        body: &mut Block,
        value: Handle<Expression>,
        bits: u32,
        signed: bool,
    ) -> Handle<Expression> {
        if bits == BITS32 {
            return if signed {
                self.cast(body, value, ScalarKind::Sint, None)
            } else {
                value
            };
        }
        let source = if signed {
            self.cast(body, value, ScalarKind::Sint, None)
        } else {
            value
        };
        let zero = self.lit(body, Literal::U32(0));
        let count = self.lit(body, Literal::U32(bits));
        self.math(body, MathFunction::ExtractBits, source, &[zero, count])
    }

    /// A float (f16 or f32) to a signed or unsigned integer of `bits` (16 or 32), truncating,
    /// saturating and giving 0 for a NaN; the result is a `u32` (see the module docs).
    fn float_to_int(
        &mut self,
        body: &mut Block,
        value: Handle<Expression>,
        f16: bool,
        bits: u32,
        signed: bool,
    ) -> Handle<Expression> {
        let value = if f16 {
            self.cast(body, value, ScalarKind::Float, Some(WIDTH32))
        } else {
            value
        };
        let kind = if signed {
            ScalarKind::Sint
        } else {
            ScalarKind::Uint
        };
        let mut int = self.cast(body, value, kind, Some(WIDTH32));
        if bits < BITS32 {
            // Saturate to 16 bits; truncation and saturation commute.
            let (low, high) = if signed {
                (
                    Literal::I32(i32::from(i16::MIN)),
                    Literal::I32(i32::from(i16::MAX)),
                )
            } else {
                (Literal::U32(0), Literal::U32(u32::from(u16::MAX)))
            };
            let low = self.lit(body, low);
            let high = self.lit(body, high);
            let above = self.math(body, MathFunction::Max, int, &[low]);
            int = self.math(body, MathFunction::Min, above, &[high]);
        }
        let int = if signed {
            self.cast(body, int, ScalarKind::Uint, None)
        } else {
            int
        };
        let nan = self.is_nan(body, value);
        let zero = self.lit(body, Literal::U32(0));
        self.select(body, nan, zero, int)
    }

    /// Lowers instruction `at`, storing its value.
    #[expect(
        clippy::too_many_lines,
        reason = "one arm per group of opcodes, in the module docs' order; splitting the match \
                  would hide which opcodes are lowered"
    )]
    pub(super) fn inst(
        &mut self,
        body: &mut Block,
        at: Value,
        op: &Operation,
    ) -> Result<(), LowerError> {
        let opcode = op.opcode;
        if let Some(var) = Var::of(at, op)? {
            let pointer = |this: &mut Self, body: &mut Block| this.var(body, var);
            let expr = if opcode.result() == Type::Void {
                let value_index = op.args.len().saturating_sub(1);
                let value = self.arg(body, at, op, value_index)?;
                let pointer = pointer(self, body)?;
                push(body, Statement::Store { pointer, value });
                None
            } else {
                let pointer = pointer(self, body)?;
                Some(self.expr(body, Expression::Load { pointer }))
            };
            return self.done(at, expr);
        }
        if let Some((cmp, unordered)) = float_comparison(opcode) {
            let a = self.arg(body, at, op, 0)?;
            let b = self.arg(body, at, op, 1)?;
            let expr = self.float_compare(body, cmp, unordered, a, b);
            return self.done(at, Some(expr));
        }
        if let Some((cmp, signed)) = integer_comparison(opcode) {
            let mut a = self.arg(body, at, op, 0)?;
            let mut b = self.arg(body, at, op, 1)?;
            if signed {
                a = self.cast(body, a, ScalarKind::Sint, None);
                b = self.cast(body, b, ScalarKind::Sint, None);
            }
            let expr = self.binary(body, cmp, a, b);
            return self.done(at, Some(expr));
        }
        if let Some(fun) = float_math(opcode) {
            let value = self.arg(body, at, op, 0)?;
            let expr = self.math(body, fun, value, &[]);
            return self.done(at, Some(expr));
        }
        if let Some(binop) = plain_binary(opcode) {
            let a = self.arg(body, at, op, 0)?;
            let b = self.arg(body, at, op, 1)?;
            let expr = self.binary(body, binop, a, b);
            return self.done(at, Some(expr));
        }
        if let Some((ty, size, kind)) = composite(opcode) {
            let expr = self.composite(body, at, op, ty, size, kind)?;
            return self.done(at, Some(expr));
        }
        if let Some((width, bits, signed)) = int_to_float(opcode) {
            self.check_rounding(at, op)?;
            let value = self.arg(body, at, op, 0)?;
            let source = self.low_bits(body, value, bits, signed);
            let expr = self.cast(body, source, ScalarKind::Float, Some(width));
            return self.done(at, Some(expr));
        }
        if let Some((f16, bits, signed)) = float_to_int(opcode) {
            let value = self.arg(body, at, op, 0)?;
            let expr = self.float_to_int(body, value, f16, bits, signed);
            return self.done(at, Some(expr));
        }
        if super::memory::is_local(opcode) || super::memory::is_shared(opcode) {
            let expr = self.memory_op(body, at, op)?;
            return self.done(at, expr);
        }
        let expr = match opcode {
            Opcode::Void | Opcode::Prologue | Opcode::Epilogue => None,
            Opcode::Identity | Opcode::ConditionRef => Some(self.arg(body, at, op, 0)?),
            Opcode::UndefU1 => Some(self.lit(body, Literal::Bool(false))),
            Opcode::UndefU32 => Some(self.lit(body, Literal::U32(0))),
            Opcode::UndefU64 => Some(self.lit(body, Literal::U64(0))),
            Opcode::DemoteToHelperInvocation => {
                push(body, Statement::Kill);
                None
            }
            Opcode::Barrier => {
                push(body, Statement::ControlBarrier(naga::Barrier::WORK_GROUP));
                None
            }
            Opcode::WorkgroupMemoryBarrier => {
                push(body, Statement::MemoryBarrier(naga::Barrier::WORK_GROUP));
                None
            }
            Opcode::DeviceMemoryBarrier => {
                push(body, Statement::MemoryBarrier(naga::Barrier::STORAGE));
                None
            }
            Opcode::BitCastF32U32 | Opcode::BitCastU32F32 => {
                let kind = if opcode == Opcode::BitCastF32U32 {
                    ScalarKind::Float
                } else {
                    ScalarKind::Uint
                };
                let value = self.arg(body, at, op, 0)?;
                Some(self.cast(body, value, kind, None))
            }
            Opcode::FPNeg16 | Opcode::FPNeg32 => {
                let value = self.arg(body, at, op, 0)?;
                Some(self.unary(body, UnaryOperator::Negate, value))
            }
            // The controls are not expressed (decision 10, in the module docs).
            Opcode::FPFma16 | Opcode::FPFma32 => {
                let factor = self.arg(body, at, op, 0)?;
                let multiplier = self.arg(body, at, op, 1)?;
                let addend = self.arg(body, at, op, 2)?;
                Some(self.math(body, MathFunction::Fma, factor, &[multiplier, addend]))
            }
            Opcode::FPMax32 | Opcode::FPMin32 => {
                let fun = if opcode == Opcode::FPMax32 {
                    MathFunction::Max
                } else {
                    MathFunction::Min
                };
                let a = self.arg(body, at, op, 0)?;
                let b = self.arg(body, at, op, 1)?;
                Some(self.nan_aware(body, fun, a, b))
            }
            Opcode::FPClamp16 | Opcode::FPClamp32 => {
                let value = self.arg(body, at, op, 0)?;
                let low = self.arg(body, at, op, 1)?;
                let high = self.arg(body, at, op, 2)?;
                let above = self.nan_aware(body, MathFunction::Max, value, low);
                Some(self.nan_aware(body, MathFunction::Min, above, high))
            }
            Opcode::FPRecip32 => {
                let value = self.arg(body, at, op, 0)?;
                let one = self.lit(body, Literal::F32(1.0));
                Some(self.binary(body, BinaryOperator::Divide, one, value))
            }
            Opcode::FPIsNan16 | Opcode::FPIsNan32 => {
                let value = self.arg(body, at, op, 0)?;
                Some(self.is_nan(body, value))
            }
            Opcode::INeg32 | Opcode::INeg64 => {
                let value = self.arg(body, at, op, 0)?;
                let bits = if opcode == Opcode::INeg64 {
                    BITS64
                } else {
                    BITS32
                };
                let zero = self.zero_int(body, bits);
                Some(self.binary(body, BinaryOperator::Subtract, zero, value))
            }
            Opcode::IAbs32 | Opcode::IAbs64 => {
                let value = self.arg(body, at, op, 0)?;
                let bits = if opcode == Opcode::IAbs64 {
                    BITS64
                } else {
                    BITS32
                };
                let zero = self.zero_int(body, bits);
                let negated = self.binary(body, BinaryOperator::Subtract, zero, value);
                let signed = self.cast(body, value, ScalarKind::Sint, None);
                let signed_zero = self.cast(body, zero, ScalarKind::Sint, None);
                let negative = self.binary(body, BinaryOperator::Less, signed, signed_zero);
                Some(self.select(body, negative, negated, value))
            }
            Opcode::UDiv32 | Opcode::SDiv32 => Some(self.divide(body, at, op)?),
            Opcode::ShiftLeftLogical32
            | Opcode::ShiftRightLogical32
            | Opcode::ShiftRightArithmetic32
            | Opcode::ShiftLeftLogical64
            | Opcode::ShiftRightLogical64
            | Opcode::ShiftRightArithmetic64 => {
                let (kind, bits) = match opcode {
                    Opcode::ShiftLeftLogical32 => (Shift::Left, BITS32),
                    Opcode::ShiftRightLogical32 => (Shift::Logical, BITS32),
                    Opcode::ShiftRightArithmetic32 => (Shift::Arithmetic, BITS32),
                    Opcode::ShiftLeftLogical64 => (Shift::Left, BITS64),
                    Opcode::ShiftRightLogical64 => (Shift::Logical, BITS64),
                    _ => (Shift::Arithmetic, BITS64),
                };
                let value = self.arg(body, at, op, 0)?;
                let amount = self.arg(body, at, op, 1)?;
                Some(self.shift(body, kind, value, amount, bits))
            }
            Opcode::BitwiseNot32 => {
                let value = self.arg(body, at, op, 0)?;
                Some(self.unary(body, UnaryOperator::BitwiseNot, value))
            }
            Opcode::LogicalNot => {
                let value = self.arg(body, at, op, 0)?;
                Some(self.unary(body, UnaryOperator::LogicalNot, value))
            }
            Opcode::BitFieldInsert => {
                let base = self.arg(body, at, op, 0)?;
                let insert = self.arg(body, at, op, 1)?;
                let offset = self.arg(body, at, op, 2)?;
                let count = self.arg(body, at, op, 3)?;
                let (offset, count) = self.field(body, offset, count);
                Some(self.math(
                    body,
                    MathFunction::InsertBits,
                    base,
                    &[insert, offset, count],
                ))
            }
            Opcode::BitFieldUExtract | Opcode::BitFieldSExtract => {
                let base = self.arg(body, at, op, 0)?;
                let offset = self.arg(body, at, op, 1)?;
                let raw_count = self.arg(body, at, op, 2)?;
                let (offset, count) = self.field(body, offset, raw_count);
                let field = self.math(body, MathFunction::ExtractBits, base, &[offset, count]);
                if opcode == Opcode::BitFieldUExtract {
                    Some(field)
                } else {
                    Some(self.sign_extend_field(body, field, raw_count))
                }
            }
            Opcode::BitReverse32 | Opcode::BitCount32 | Opcode::FindUMsb32 => {
                let fun = match opcode {
                    Opcode::BitReverse32 => MathFunction::ReverseBits,
                    Opcode::BitCount32 => MathFunction::CountOneBits,
                    _ => MathFunction::FirstLeadingBit,
                };
                let value = self.arg(body, at, op, 0)?;
                Some(self.math(body, fun, value, &[]))
            }
            Opcode::FindSMsb32 => {
                let value = self.arg(body, at, op, 0)?;
                let signed = self.cast(body, value, ScalarKind::Sint, None);
                let found = self.math(body, MathFunction::FirstLeadingBit, signed, &[]);
                Some(self.cast(body, found, ScalarKind::Uint, None))
            }
            Opcode::SMin32
            | Opcode::SMax32
            | Opcode::UMin32
            | Opcode::UMax32
            | Opcode::SClamp32
            | Opcode::UClamp32 => Some(self.integer_min_max(body, at, op)?),
            Opcode::SelectU1
            | Opcode::SelectU32
            | Opcode::SelectU64
            | Opcode::SelectF16
            | Opcode::SelectF32 => {
                let condition = self.arg(body, at, op, 0)?;
                let accept = self.arg(body, at, op, 1)?;
                let reject = self.arg(body, at, op, 2)?;
                Some(self.select(body, condition, accept, reject))
            }
            Opcode::PackUint2x32 => {
                let pair = self.arg(body, at, op, 0)?;
                let low = self.component(body, pair, 0);
                let high = self.component(body, pair, 1);
                let low = self.cast(body, low, ScalarKind::Uint, Some(WIDTH64));
                let high = self.cast(body, high, ScalarKind::Uint, Some(WIDTH64));
                let half = self.lit(body, Literal::U32(BITS32));
                let high = self.binary(body, BinaryOperator::ShiftLeft, high, half);
                Some(self.binary(body, BinaryOperator::InclusiveOr, low, high))
            }
            Opcode::UnpackUint2x32 => {
                let value = self.arg(body, at, op, 0)?;
                let low = self.cast(body, value, ScalarKind::Uint, Some(WIDTH32));
                let half = self.lit(body, Literal::U32(BITS32));
                let high = self.binary(body, BinaryOperator::ShiftRight, value, half);
                let high = self.cast(body, high, ScalarKind::Uint, Some(WIDTH32));
                let ty = self.ir_type(Type::U32x2)?;
                Some(self.expr(
                    body,
                    Expression::Compose {
                        ty,
                        components: vec![low, high],
                    },
                ))
            }
            Opcode::PackHalf2x16 | Opcode::PackFloat2x16 => {
                let mut pair = self.arg(body, at, op, 0)?;
                if opcode == Opcode::PackFloat2x16 {
                    pair = self.cast(body, pair, ScalarKind::Float, Some(WIDTH32));
                }
                Some(self.math(body, MathFunction::Pack2x16float, pair, &[]))
            }
            Opcode::UnpackHalf2x16 | Opcode::UnpackFloat2x16 => {
                let value = self.arg(body, at, op, 0)?;
                let pair = self.math(body, MathFunction::Unpack2x16float, value, &[]);
                Some(if opcode == Opcode::UnpackFloat2x16 {
                    self.cast(body, pair, ScalarKind::Float, Some(WIDTH16))
                } else {
                    pair
                })
            }
            Opcode::ConvertF16F32 | Opcode::ConvertF32F16 => {
                self.check_rounding(at, op)?;
                let width = if opcode == Opcode::ConvertF16F32 {
                    WIDTH16
                } else {
                    WIDTH32
                };
                let value = self.arg(body, at, op, 0)?;
                Some(self.cast(body, value, ScalarKind::Float, Some(width)))
            }
            Opcode::ConvertU64U32 | Opcode::ConvertU32U64 => {
                let width = if opcode == Opcode::ConvertU64U32 {
                    WIDTH64
                } else {
                    WIDTH32
                };
                let value = self.arg(body, at, op, 0)?;
                Some(self.cast(body, value, ScalarKind::Uint, Some(width)))
            }
            Opcode::LaneId => {
                let lane = self
                    .lane_arg
                    .ok_or(LowerError::Internal("no lane id argument".into()))?;
                Some(self.expr(body, Expression::FunctionArgument(lane)))
            }
            Opcode::WorkgroupId | Opcode::LocalInvocationId => {
                let arg = if opcode == Opcode::WorkgroupId {
                    self.workgroup_id_arg
                } else {
                    self.local_invocation_id_arg
                };
                let arg = arg.ok_or(LowerError::Internal(format!("no argument for {opcode}")))?;
                Some(self.expr(body, Expression::FunctionArgument(arg)))
            }
            Opcode::GetAttribute => {
                let attribute = attribute_arg(at, op, 1)?;
                let Some(AttributeKind::Generic { index, component }) = classify(attribute) else {
                    return Err(LowerError::Internal(format!("input {attribute}")));
                };
                let input = *self
                    .input_args
                    .get(&index)
                    .ok_or(LowerError::Internal(format!("input {attribute}")))?;
                let base = self.expr(body, Expression::FunctionArgument(input));
                Some(self.component(body, base, component))
            }
            Opcode::SetAttribute => {
                let attribute = attribute_arg(at, op, 2)?;
                let (member, component) = match classify(attribute) {
                    Some(AttributeKind::Position { component }) => (self.position, component),
                    Some(AttributeKind::Generic { index, component }) => {
                        (self.generic_members.get(&index).copied(), component)
                    }
                    None => (None, 0),
                };
                let value = Self::raw_arg(at, op, 1)?;
                self.store_output(body, at, member, component, (1, value))?;
                None
            }
            Opcode::SetFragColor => {
                let (Some(Imm::U32(target)), Some(Imm::U32(component))) =
                    (imm_arg(op, 0), imm_arg(op, 1))
                else {
                    return Err(LowerError::Internal(format!("colour output at {at}")));
                };
                let member = u8::try_from(target)
                    .ok()
                    .and_then(|target| self.target_members.get(&target).copied());
                let value = Self::raw_arg(at, op, 2)?;
                self.store_output(body, at, member, component, (2, value))?;
                None
            }
            opcode => return Err(LowerError::Operation { at, opcode }),
        };
        self.done(at, expr)
    }

    /// Records the expression of instruction `at`, if it has one.
    fn done(&mut self, at: Value, expr: Option<Handle<Expression>>) -> Result<(), LowerError> {
        if let Some(expr) = expr {
            let slot = usize::try_from(at.index())
                .ok()
                .and_then(|slot| self.values.get_mut(slot))
                .ok_or_else(|| LowerError::Internal(format!("{at} is outside the arena")))?;
            *slot = Some(expr);
        }
        Ok(())
    }

    /// Refuses a directed rounding mode on a conversion (the parent module's divergence
    /// table).
    fn check_rounding(&self, at: Value, op: &Operation) -> Result<(), LowerError> {
        if let Flags::Fp(control) = op.flags
            && let Some(rounding) = directed(control.rounding)
        {
            return Err(LowerError::Rounding {
                at,
                opcode: op.opcode,
                rounding,
            });
        }
        Ok(())
    }

    /// A float comparison; an unordered one is the negation of the ordered complement, so a
    /// NaN operand makes it true.
    fn float_compare(
        &mut self,
        body: &mut Block,
        cmp: BinaryOperator,
        unordered: bool,
        a: Handle<Expression>,
        b: Handle<Expression>,
    ) -> Handle<Expression> {
        use BinaryOperator::{Equal, Greater, GreaterEqual, Less, LessEqual, NotEqual};
        // The ordered complement of each operator (`a != b` ordered is `a < b || a > b`).
        let ordered_not_equal = |this: &mut Self, body: &mut Block| {
            let less = this.binary(body, Less, a, b);
            let greater = this.binary(body, Greater, a, b);
            this.binary(body, BinaryOperator::LogicalOr, less, greater)
        };
        if !unordered {
            return if cmp == NotEqual {
                ordered_not_equal(self, body)
            } else {
                self.binary(body, cmp, a, b)
            };
        }
        let complement = match cmp {
            Equal => ordered_not_equal(self, body),
            NotEqual => self.binary(body, Equal, a, b),
            Less => self.binary(body, GreaterEqual, a, b),
            Greater => self.binary(body, LessEqual, a, b),
            LessEqual => self.binary(body, Greater, a, b),
            _ => self.binary(body, Less, a, b),
        };
        self.unary(body, UnaryOperator::LogicalNot, complement)
    }

    /// `UDiv32` and `SDiv32` (see the module docs).
    fn divide(
        &mut self,
        body: &mut Block,
        at: Value,
        op: &Operation,
    ) -> Result<Handle<Expression>, LowerError> {
        let a = self.arg(body, at, op, 0)?;
        let b = self.arg(body, at, op, 1)?;
        let zero = self.lit(body, Literal::U32(0));
        let by_zero = self.binary(body, BinaryOperator::Equal, b, zero);
        let one = self.lit(body, Literal::U32(1));
        let divisor = self.select(body, by_zero, one, b);
        let quotient = if op.opcode == Opcode::SDiv32 {
            let a = self.cast(body, a, ScalarKind::Sint, None);
            let divisor = self.cast(body, divisor, ScalarKind::Sint, None);
            let quotient = self.binary(body, BinaryOperator::Divide, a, divisor);
            self.cast(body, quotient, ScalarKind::Uint, None)
        } else {
            self.binary(body, BinaryOperator::Divide, a, divisor)
        };
        let all_ones = self.lit(body, Literal::U32(u32::MAX));
        Ok(self.select(body, by_zero, all_ones, quotient))
    }

    /// `field` (already extracted unsigned) sign-extended from bit `min(count, 32) - 1`; 0 for
    /// a zero count.
    fn sign_extend_field(
        &mut self,
        body: &mut Block,
        field: Handle<Expression>,
        count: Handle<Expression>,
    ) -> Handle<Expression> {
        let width = self.lit(body, Literal::U32(BITS32));
        let count = self.math(body, MathFunction::Min, count, &[width]);
        let spare = self.binary(body, BinaryOperator::Subtract, width, count);
        let top = self.lit(body, Literal::U32(BITS32 - 1));
        let spare = self.math(body, MathFunction::Min, spare, &[top]);
        let raised = self.binary(body, BinaryOperator::ShiftLeft, field, spare);
        let signed = self.cast(body, raised, ScalarKind::Sint, None);
        let lowered = self.binary(body, BinaryOperator::ShiftRight, signed, spare);
        let extended = self.cast(body, lowered, ScalarKind::Uint, None);
        let zero = self.lit(body, Literal::U32(0));
        let empty = self.binary(body, BinaryOperator::Equal, count, zero);
        self.select(body, empty, zero, extended)
    }

    /// The integer minimum, maximum and clamp (`min(max(value, low), high)`).
    fn integer_min_max(
        &mut self,
        body: &mut Block,
        at: Value,
        op: &Operation,
    ) -> Result<Handle<Expression>, LowerError> {
        let signed = matches!(
            op.opcode,
            Opcode::SMin32 | Opcode::SMax32 | Opcode::SClamp32
        );
        let mut args = Vec::with_capacity(op.args.len());
        for index in 0..op.opcode.args().len() {
            let arg = self.arg(body, at, op, index)?;
            args.push(if signed {
                self.cast(body, arg, ScalarKind::Sint, None)
            } else {
                arg
            });
        }
        let get = |index: usize| {
            args.get(index)
                .copied()
                .ok_or(LowerError::Operand { at, index })
        };
        let result = match op.opcode {
            Opcode::SMin32 | Opcode::UMin32 => {
                self.math(body, MathFunction::Min, get(0)?, &[get(1)?])
            }
            Opcode::SMax32 | Opcode::UMax32 => {
                self.math(body, MathFunction::Max, get(0)?, &[get(1)?])
            }
            _ => {
                let above = self.math(body, MathFunction::Max, get(0)?, &[get(1)?]);
                self.math(body, MathFunction::Min, above, &[get(2)?])
            }
        };
        Ok(if signed {
            self.cast(body, result, ScalarKind::Uint, None)
        } else {
            result
        })
    }

    /// A composite construct, extract or insert; the component of an extract or insert must be
    /// an immediate inside the vector.
    fn composite(
        &mut self,
        body: &mut Block,
        at: Value,
        op: &Operation,
        ty: Type,
        size: u32,
        kind: CompositeOp,
    ) -> Result<Handle<Expression>, LowerError> {
        let component_at = |index: usize| match imm_arg(op, index) {
            Some(Imm::U32(component)) if component < size => Ok(component),
            _ => Err(LowerError::Operand { at, index }),
        };
        match kind {
            CompositeOp::Construct => {
                let mut components = Vec::with_capacity(op.args.len());
                for index in 0..op.args.len() {
                    components.push(self.arg(body, at, op, index)?);
                }
                let ty = self.ir_type(ty)?;
                Ok(self.expr(body, Expression::Compose { ty, components }))
            }
            CompositeOp::Extract => {
                let component = component_at(1)?;
                let vector = self.arg(body, at, op, 0)?;
                Ok(self.component(body, vector, component))
            }
            CompositeOp::Insert => {
                let replaced = component_at(2)?;
                let vector = self.arg(body, at, op, 0)?;
                let value = self.arg(body, at, op, 1)?;
                let mut components = Vec::new();
                for index in 0..size {
                    components.push(if index == replaced {
                        value
                    } else {
                        self.component(body, vector, index)
                    });
                }
                let ty = self.ir_type(ty)?;
                Ok(self.expr(body, Expression::Compose { ty, components }))
            }
        }
    }
}
