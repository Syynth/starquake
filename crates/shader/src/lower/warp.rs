//! Warp operations and derivatives as naga subgroup and derivative operations: shader slice
//! 16b (decided by the owner, 2026-11-20, decision log "Owner rulings on the wave 46
//! questions", row 1).
//!
//! Each operation means what Eden's SPIR-V backend makes of it on a host whose subgroups are
//! no wider than the guest's 32-lane warp and whose stage has subgroups
//! (`eden/src/shader_recompiler/backend/spirv/emit_spirv_warp.cpp` @ cf654052, GPL-3.0-or-later:
//! the paths where `warp_size_potentially_larger_than_guest` is false and
//! `StageSupportsSubgroups` is true):
//!
//! - `VoteAll` and `VoteAny` are naga's subgroup `All` and `Any` reductions
//!   (`EmitVoteAll`, `EmitVoteAny`: `OpGroupNonUniformAll`, `OpGroupNonUniformAny`).
//!   `VoteEqual` is `All(p) || All(!p)`: naga has no `AllEqual` (`EmitVoteEqual`:
//!   `OpGroupNonUniformAllEqual`, which is true exactly when every active lane's value is
//!   the same).
//! - `SubgroupBallot` is component 0 of naga's ballot (`EmitSubgroupBallot`).
//! - The lane masks `SubgroupEqMask` to `SubgroupGeMask` are computed from the lane id: naga
//!   has no mask built-ins, and Eden loads SPIR-V's (`EmitSubgroupEqMask` and its siblings,
//!   `LoadMask`, component 0). For a lane below 32 the words are the same; for a lane of 32
//!   or more (a wider subgroup) the equal mask is 0 and the below mask all ones, as the
//!   interpreter defines them ([`crate::interp`], "Warp operations").
//! - The four shuffles compute their source lane and in-range flag exactly as
//!   `EmitShuffleIndex`, `EmitShuffleUp`, `EmitShuffleDown` and `EmitShuffleButterfly` do
//!   (`ComputeMinThreadId`, `ComputeMaxThreadId`, `GuestLane`'s `& 31`, and their signed
//!   comparisons `OpSLessThanEqual` and `OpSGreaterThanEqual`), and select the gathered value
//!   only in range (`OpSelect`). The gather is naga's `Shuffle` of the source lane for
//!   `ShuffleIndex` (`ShuffleAbsolute`), or its `Broadcast` of `lane & 31` when the lane and
//!   the segment mask are immediates and the mask is 0 (Eden's `BroadcastLane` case);
//!   `ShuffleUp` and `ShuffleDown` of the delta (`ShuffleRelative` where the host supports
//!   relative shuffles); `ShuffleXor` of the delta for `ShuffleButterfly`.
//!   `GetInBoundsFromOp` of a shuffle is that shuffle's in-range flag (`SetInBoundsFlag`).
//! - `QuadBroadcast` with an immediate index is naga's `QuadBroadcast` of `index & 3`
//!   (`EmitQuadBroadcast` with quad shuffles); with an index computed at run time it is
//!   Eden's fallback, a `Shuffle` from lane `(lane & !3) | (index & 3)`, since naga needs a
//!   constant index (`valid/function.rs`: `InvalidInvocationIdExprType`).
//! - `QuadSwap` takes an immediate direction 0, 1 or 2, naga's `QuadSwap` along `X`, `Y` or
//!   `Diagonal` (`EmitQuadSwap` with quad shuffles: `OpGroupNonUniformQuadSwap`). naga's MSL
//!   writer swaps with lane `lane ^ 1`, `^ 2` and `^ 3` for them (`back/msl/writer.rs`:
//!   `quad_shuffle_xor` with `1u`, `2u`, `3u`), which is Eden's fallback's
//!   `lane ^ (direction + 1)` and the interpreter's.
//! - `FSwizzleAdd a, b, swizzle` is `a * A[i] + b * B[i]`, `i = (swizzle >> ((lane & 3) << 1))
//!   & 3`, with Eden's tables ([`crate::interp`]'s `SWIZZLE_ADD_FACTORS`, `fswzadd_lut_a` and
//!   `fswzadd_lut_b`) indexed at run time as `EmitFSwizzleAdd` does (`OpVectorExtractDynamic`).
//!   Its controls are not expressed, as on other arithmetic (decision 10; the parent module's
//!   divergence table).
//! - `DPdxFine`, `DPdyFine`, `DPdxCoarse` and `DPdyCoarse` are naga's derivatives along that
//!   axis with that control (`EmitDPdxFine` and its siblings).
//!
//! No warp operation compares or converts a float: the range checks are signed integer
//! comparisons, as Eden's are, so the NaN-safe helpers of the `ops` module (the owner's ruling
//! of 2026-11-20, row 3) have nothing to take here.
//!
//! # Requirements
//!
//! Every warp operation needs subgroups ([`crate::host::Requirements::subgroup`]): they are
//! Eden's subgroup lists ([`crate::info::opcode_requirements`]), and each one lowers to a naga
//! subgroup statement or reads the lane id. [`needs_subgroup_size_32`] says which also need
//! the narrower size flag (spec, "Compute shaders"): those whose result depends on the warp
//! having 32 lanes. Derivatives need neither.
//!
//! # Divergences
//!
//! | What | Here | Reference | Pinned by |
//! |---|---|---|---|
//! | A warp operation in a vertex program | Refused ([`LowerError::NotInStage`]): naga 30.0.1 allows subgroup operations in a vertex stage only with `Capabilities::SUBGROUP_VERTEX_STAGE` (`valid/mod.rs`), for which [`crate::host::Requirements`] has no flag yet, and does not allow the lane id there at all (as slice 16 found for `LaneId`) | Eden writes each one's fallback for a stage without subgroups (`StageSupportsSubgroups`): a vote gives its argument, a ballot `p ? 1 : 0`, the lane id 0, and so on | `tests::warp::warp_operations_are_refused_in_vertex_programs` |
//! | A derivative outside a pixel program | Refused ([`LowerError::NotInStage`]): naga 30.0.1 validates `Derivative` in the fragment stage only (`valid/expression.rs`) | Eden writes `OpDPdx*` whatever the stage | `tests::warp::derivatives_are_refused_outside_pixel_programs` |
//! | A host whose subgroups are wider than 32 lanes | Not followed: each operation whose result depends on 32 lanes records "needs subgroup size 32", and the backend refuses the shader where that is not met | Eden partitions a wider subgroup into guest warps (`warp_size_potentially_larger_than_guest`: `WarpExtract`, `HostThreadId`, the masked lane id) | `tests::warp::every_warp_operation_lowers_with_its_requirements` |
//! | The lane masks | Computed from the lane id | SPIR-V's mask built-ins (the words agree for every lane below 32) | `tests::warp::warp_operations_match_the_interpreter_on_the_gpu`, `tests::warp::warp_operations_match_a_32_lane_model_on_the_gpu` |
//! | `QuadSwap` whose direction is not the immediate 0, 1 or 2 | Refused ([`LowerError::Operand`], argument 1): naga's `QuadSwap` takes only the three directions | Eden passes the direction on (SPIR-V needs a constant); its fallback without quad shuffles, and the interpreter, read lane `lane ^ (direction + 1)` for any value | `tests::warp::quad_swap_takes_an_immediate_direction` |
//! | A coarse or fine derivative on Metal | The module keeps the control, but naga's MSL writer writes `dfdx` and `dfdy` for every control (Metal has no other), so the host's own derivative runs, coarse or fine | `OpDPdxCoarse`, `OpDPdxFine` and their `y` forms | `tests::warp::warp_snapshot` (`warp_derivatives`) |
//! | `QuadBroadcast` with an immediate index of 4 or more | `index & 3`, as the interpreter and Eden's fallback | Eden passes the index on unmasked to `OpGroupNonUniformQuadBroadcast` | `tests::warp::quad_broadcast_masks_an_immediate_index_and_shuffles_a_computed_one` |

use naga::{
    BinaryOperator, Block, CollectiveOperation, DerivativeAxis, DerivativeControl, Direction,
    Expression, GatherMode, Handle, Literal, ScalarKind, Statement, SubgroupOperation,
    UnaryOperator,
};

use super::builder::push;
use super::ops::Shift;
use super::{LowerError, Lowering};
use crate::interp::{QUAD_LANE_MASK, SWIZZLE_ADD_FACTORS};
use crate::ir::{Imm, Opcode, Operand, Operation, Type, Value, WARP_SIZE};
use crate::translate::GuestStage;

/// `GuestLane`'s mask: a lane index modulo the warp size (`& 31U`).
const LANE_MASK: u32 = WARP_SIZE - 1;
/// How far `FSwizzleAdd` shifts the lane's place in its quad to find its two bits of the
/// swizzle (`EmitFSwizzleAdd`: `OpShiftLeftLogical(mask, Const(1U))`).
const SWIZZLE_PLACE_SHIFT: u32 = 1;

/// Which lane mask.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Mask {
    Equal,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
}

/// Which shuffle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Shuffle {
    Index,
    Up,
    Down,
    Butterfly,
}

/// A warp operation or derivative, as the lowering groups them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Warp {
    /// `VoteAll` or `VoteAny`: a naga reduction.
    Vote(SubgroupOperation),
    VoteEqual,
    Ballot,
    Mask(Mask),
    Shuffle(Shuffle),
    QuadBroadcast,
    QuadSwap,
    SwizzleAdd,
    Derivative(DerivativeAxis, DerivativeControl),
    /// `GetInBoundsFromOp`.
    InBounds,
}

impl Warp {
    /// The group of `opcode`, or `None` when it is not one of this module's (`LaneId`, which
    /// slice 16 lowers, is not).
    pub(super) const fn of(opcode: Opcode) -> Option<Self> {
        use Opcode as O;
        Some(match opcode {
            O::VoteAll => Self::Vote(SubgroupOperation::All),
            O::VoteAny => Self::Vote(SubgroupOperation::Any),
            O::VoteEqual => Self::VoteEqual,
            O::SubgroupBallot => Self::Ballot,
            O::SubgroupEqMask => Self::Mask(Mask::Equal),
            O::SubgroupLtMask => Self::Mask(Mask::Less),
            O::SubgroupLeMask => Self::Mask(Mask::LessEqual),
            O::SubgroupGtMask => Self::Mask(Mask::Greater),
            O::SubgroupGeMask => Self::Mask(Mask::GreaterEqual),
            O::ShuffleIndex => Self::Shuffle(Shuffle::Index),
            O::ShuffleUp => Self::Shuffle(Shuffle::Up),
            O::ShuffleDown => Self::Shuffle(Shuffle::Down),
            O::ShuffleButterfly => Self::Shuffle(Shuffle::Butterfly),
            O::QuadBroadcast => Self::QuadBroadcast,
            O::QuadSwap => Self::QuadSwap,
            O::FSwizzleAdd => Self::SwizzleAdd,
            O::DPdxFine => Self::Derivative(DerivativeAxis::X, DerivativeControl::Fine),
            O::DPdyFine => Self::Derivative(DerivativeAxis::Y, DerivativeControl::Fine),
            O::DPdxCoarse => Self::Derivative(DerivativeAxis::X, DerivativeControl::Coarse),
            O::DPdyCoarse => Self::Derivative(DerivativeAxis::Y, DerivativeControl::Coarse),
            O::GetInBoundsFromOp => Self::InBounds,
            _ => return None,
        })
    }
}

/// Whether an operation of `opcode` needs the host to run 32-wide subgroups, the narrower
/// "needs subgroup size 32" flag ([`crate::host::Requirements::subgroup_size_32`]; spec,
/// "Compute shaders"; decided by the owner, 2026-11-20): its result depends on the warp having
/// 32 lanes.
///
/// That is the lane id, the votes and the ballot (over every lane of the warp), the lane masks
/// and the four shuffles (whose source lanes are taken modulo 32). Not the quad operations nor
/// `FSwizzleAdd`, which read only the lane's place in its quad (`lane & 3`) and lanes of the
/// same quad, nor the derivatives, which use no subgroup.
pub const fn needs_subgroup_size_32(opcode: Opcode) -> bool {
    use Opcode as O;
    matches!(
        opcode,
        O::LaneId
            | O::VoteAll
            | O::VoteAny
            | O::VoteEqual
            | O::SubgroupBallot
            | O::SubgroupEqMask
            | O::SubgroupLtMask
            | O::SubgroupLeMask
            | O::SubgroupGtMask
            | O::SubgroupGeMask
            | O::ShuffleIndex
            | O::ShuffleUp
            | O::ShuffleDown
            | O::ShuffleButterfly
    )
}

/// Whether `stage` may use `opcode`, as far as this module is concerned (see the module docs'
/// divergences): derivatives only in pixel programs, the other warp operations in any stage
/// but vertex; `true` for every other opcode.
pub(super) fn allowed_in(opcode: Opcode, stage: GuestStage) -> bool {
    match Warp::of(opcode) {
        Some(Warp::Derivative(..)) => stage == GuestStage::Pixel,
        Some(Warp::InBounds) | None => true,
        Some(_) => stage != GuestStage::Vertex,
    }
}

/// Whether lowering `op` reads the lane id (the subgroup invocation id built-in): the lane
/// masks, the shuffles, `FSwizzleAdd`, and a `QuadBroadcast` whose index is not an immediate.
pub(super) fn reads_lane_id(op: &Operation) -> bool {
    match Warp::of(op.opcode) {
        Some(Warp::Mask(_) | Warp::Shuffle(_) | Warp::SwizzleAdd) => true,
        Some(Warp::QuadBroadcast) => !matches!(op.args.get(1), Some(Operand::Imm(Imm::U32(_)))),
        _ => false,
    }
}

/// The naga direction of a `QuadSwap` direction operand, if it is one of the three.
const fn quad_direction(operand: Operand) -> Option<Direction> {
    match operand {
        Operand::Imm(Imm::U32(0)) => Some(Direction::X),
        Operand::Imm(Imm::U32(1)) => Some(Direction::Y),
        Operand::Imm(Imm::U32(2)) => Some(Direction::Diagonal),
        _ => None,
    }
}

impl Lowering {
    /// Lowers warp operation `at` of group `kind` (see the module docs).
    pub(super) fn warp(
        &mut self,
        body: &mut Block,
        at: Value,
        op: &Operation,
        kind: Warp,
    ) -> Result<Option<Handle<Expression>>, LowerError> {
        let expr = match kind {
            Warp::Vote(operation) => {
                let predicate = self.arg(body, at, op, 0)?;
                self.vote(body, operation, predicate)?
            }
            Warp::VoteEqual => {
                let predicate = self.arg(body, at, op, 0)?;
                let all = self.vote(body, SubgroupOperation::All, predicate)?;
                let negated = self.not(body, predicate);
                let none = self.vote(body, SubgroupOperation::All, negated)?;
                self.binary(body, BinaryOperator::LogicalOr, all, none)
            }
            Warp::Ballot => {
                let predicate = self.arg(body, at, op, 0)?;
                let result = self.fb.result(Expression::SubgroupBallotResult);
                push(
                    body,
                    Statement::SubgroupBallot {
                        result,
                        predicate: Some(predicate),
                    },
                );
                self.component(body, result, 0)
            }
            Warp::Mask(mask) => self.mask(body, mask)?,
            Warp::Shuffle(shuffle) => self.shuffle(body, at, op, shuffle)?,
            Warp::QuadBroadcast => {
                let value = self.arg(body, at, op, 0)?;
                let mode = if let Operand::Imm(Imm::U32(index)) = Self::raw_arg(at, op, 1)? {
                    let index = self.lit(body, Literal::U32(index & QUAD_LANE_MASK));
                    GatherMode::QuadBroadcast(index)
                } else {
                    let index = self.arg(body, at, op, 1)?;
                    let lane = self.lane(body)?;
                    let quad = self.lit(body, Literal::U32(QUAD_LANE_MASK));
                    let base_mask = self.lit(body, Literal::U32(!QUAD_LANE_MASK));
                    let base = self.binary(body, BinaryOperator::And, lane, base_mask);
                    let place = self.binary(body, BinaryOperator::And, index, quad);
                    let source = self.binary(body, BinaryOperator::InclusiveOr, base, place);
                    GatherMode::Shuffle(source)
                };
                self.gather(body, mode, value)
            }
            Warp::QuadSwap => {
                let direction = quad_direction(Self::raw_arg(at, op, 1)?)
                    .ok_or(LowerError::Operand { at, index: 1 })?;
                let value = self.arg(body, at, op, 0)?;
                self.gather(body, GatherMode::QuadSwap(direction), value)
            }
            Warp::SwizzleAdd => self.swizzle_add(body, at, op)?,
            Warp::Derivative(axis, ctrl) => {
                let value = self.arg(body, at, op, 0)?;
                self.expr(
                    body,
                    Expression::Derivative {
                        axis,
                        ctrl,
                        expr: value,
                    },
                )
            }
            Warp::InBounds => {
                let Operand::Value(of) = Self::raw_arg(at, op, 0)? else {
                    return Err(LowerError::Operand { at, index: 0 });
                };
                // The shuffle's own value must be in scope here, as any operand's.
                self.value(of)?;
                *self
                    .in_bounds
                    .get(&of)
                    .ok_or(LowerError::Operand { at, index: 0 })?
            }
        };
        Ok(Some(expr))
    }

    /// The lane id: the subgroup invocation id argument.
    fn lane(&mut self, body: &mut Block) -> Result<Handle<Expression>, LowerError> {
        let lane = self
            .lane_arg
            .ok_or(LowerError::Internal("no lane id argument".into()))?;
        Ok(self.expr(body, Expression::FunctionArgument(lane)))
    }

    fn not(&mut self, body: &mut Block, value: Handle<Expression>) -> Handle<Expression> {
        self.unary(body, UnaryOperator::LogicalNot, value)
    }

    /// `operation` (`All` or `Any`) of `predicate` over the subgroup's active lanes.
    fn vote(
        &mut self,
        body: &mut Block,
        operation: SubgroupOperation,
        predicate: Handle<Expression>,
    ) -> Result<Handle<Expression>, LowerError> {
        let ty = self.ir_type(Type::U1)?;
        let result = self.fb.result(Expression::SubgroupOperationResult { ty });
        push(
            body,
            Statement::SubgroupCollectiveOperation {
                op: operation,
                collective_op: CollectiveOperation::Reduce,
                argument: predicate,
                result,
            },
        );
        Ok(result)
    }

    /// `value` (a `u32`) gathered from the lane `mode` names.
    fn gather(
        &mut self,
        body: &mut Block,
        mode: GatherMode,
        value: Handle<Expression>,
    ) -> Handle<Expression> {
        let ty = self.types.u32;
        let result = self.fb.result(Expression::SubgroupOperationResult { ty });
        push(
            body,
            Statement::SubgroupGather {
                mode,
                argument: value,
                result,
            },
        );
        result
    }

    /// A lane mask (see the module docs): `equal = 1 << lane` (0 from lane 32 up, the shift
    /// helper's rule) and `below = equal - 1`, wrapping.
    fn mask(&mut self, body: &mut Block, mask: Mask) -> Result<Handle<Expression>, LowerError> {
        let lane = self.lane(body)?;
        let one = self.lit(body, Literal::U32(1));
        let equal = self.shift(body, Shift::Left, one, lane, u32::BITS);
        let below = self.binary(body, BinaryOperator::Subtract, equal, one);
        Ok(match mask {
            Mask::Equal => equal,
            Mask::Less => below,
            Mask::LessEqual => self.binary(body, BinaryOperator::InclusiveOr, below, equal),
            Mask::Greater => {
                let at_or_below = self.binary(body, BinaryOperator::InclusiveOr, below, equal);
                self.unary(body, UnaryOperator::BitwiseNot, at_or_below)
            }
            Mask::GreaterEqual => self.unary(body, UnaryOperator::BitwiseNot, below),
        })
    }

    /// `a <= b` (or `>=`) of two `u32`s compared as signed values.
    fn signed_compare(
        &mut self,
        body: &mut Block,
        cmp: BinaryOperator,
        a: Handle<Expression>,
        b: Handle<Expression>,
    ) -> Handle<Expression> {
        let a = self.cast(body, a, ScalarKind::Sint, None);
        let b = self.cast(body, b, ScalarKind::Sint, None);
        self.binary(body, cmp, a, b)
    }

    /// A shuffle and its in-range flag, which [`Lowering::in_bounds`] keeps for a
    /// `GetInBoundsFromOp` of it (see the module docs).
    fn shuffle(
        &mut self,
        body: &mut Block,
        at: Value,
        op: &Operation,
        shuffle: Shuffle,
    ) -> Result<Handle<Expression>, LowerError> {
        let value = self.arg(body, at, op, 0)?;
        let index = self.arg(body, at, op, 1)?;
        let clamp = self.arg(body, at, op, 2)?;
        let segment = self.arg(body, at, op, 3)?;
        let lane = self.lane(body)?;
        // `ComputeMinThreadId` and `ComputeMaxThreadId`.
        let not_segment = self.unary(body, UnaryOperator::BitwiseNot, segment);
        let min_lane = self.binary(body, BinaryOperator::And, lane, segment);
        let clamp_bits = self.binary(body, BinaryOperator::And, clamp, not_segment);
        let max_lane = self.binary(body, BinaryOperator::InclusiveOr, min_lane, clamp_bits);
        // `GuestLane`.
        let lane_mask = self.lit(body, Literal::U32(LANE_MASK));
        let delta = self.binary(body, BinaryOperator::And, index, lane_mask);
        let (source, cmp, mode) = match shuffle {
            Shuffle::Index => {
                let low = self.binary(body, BinaryOperator::And, delta, not_segment);
                let source = self.binary(body, BinaryOperator::InclusiveOr, low, min_lane);
                let mode = match (Self::raw_arg(at, op, 1)?, Self::raw_arg(at, op, 3)?) {
                    (Operand::Imm(Imm::U32(lane)), Operand::Imm(Imm::U32(0))) => {
                        let read = self.lit(body, Literal::U32(lane & LANE_MASK));
                        GatherMode::Broadcast(read)
                    }
                    _ => GatherMode::Shuffle(source),
                };
                (source, BinaryOperator::LessEqual, mode)
            }
            Shuffle::Up => {
                let source = self.binary(body, BinaryOperator::Subtract, lane, delta);
                (
                    source,
                    BinaryOperator::GreaterEqual,
                    GatherMode::ShuffleUp(delta),
                )
            }
            Shuffle::Down => {
                let source = self.binary(body, BinaryOperator::Add, lane, delta);
                (
                    source,
                    BinaryOperator::LessEqual,
                    GatherMode::ShuffleDown(delta),
                )
            }
            Shuffle::Butterfly => {
                let source = self.binary(body, BinaryOperator::ExclusiveOr, lane, delta);
                (
                    source,
                    BinaryOperator::LessEqual,
                    GatherMode::ShuffleXor(delta),
                )
            }
        };
        let in_range = self.signed_compare(body, cmp, source, max_lane);
        self.in_bounds.insert(at, in_range);
        let gathered = self.gather(body, mode, value);
        Ok(self.select(body, in_range, gathered, value))
    }

    /// `FSwizzleAdd a, b, swizzle` (see the module docs).
    fn swizzle_add(
        &mut self,
        body: &mut Block,
        at: Value,
        op: &Operation,
    ) -> Result<Handle<Expression>, LowerError> {
        let a = self.arg(body, at, op, 0)?;
        let b = self.arg(body, at, op, 1)?;
        let swizzle = self.arg(body, at, op, 2)?;
        let lane = self.lane(body)?;
        let quad = self.lit(body, Literal::U32(QUAD_LANE_MASK));
        let place = self.binary(body, BinaryOperator::And, lane, quad);
        let step = self.lit(body, Literal::U32(SWIZZLE_PLACE_SHIFT));
        let shift = self.binary(body, BinaryOperator::ShiftLeft, place, step);
        let shifted = self.binary(body, BinaryOperator::ShiftRight, swizzle, shift);
        let index = self.binary(body, BinaryOperator::And, shifted, quad);
        let table_a = self.factor_table(body, |(factor, _)| factor);
        let table_b = self.factor_table(body, |(_, factor)| factor);
        let factor_a = self.expr(
            body,
            Expression::Access {
                base: table_a,
                index,
            },
        );
        let factor_b = self.expr(
            body,
            Expression::Access {
                base: table_b,
                index,
            },
        );
        let scaled_a = self.binary(body, BinaryOperator::Multiply, a, factor_a);
        let scaled_b = self.binary(body, BinaryOperator::Multiply, b, factor_b);
        Ok(self.binary(body, BinaryOperator::Add, scaled_a, scaled_b))
    }

    /// One of `FSwizzleAdd`'s tables as a `vec4<f32>`: the factor `pick` takes from each row
    /// of [`SWIZZLE_ADD_FACTORS`] (all finite, so naga takes them as literals).
    fn factor_table(
        &mut self,
        body: &mut Block,
        pick: fn((u32, u32)) -> u32,
    ) -> Handle<Expression> {
        let components = SWIZZLE_ADD_FACTORS
            .iter()
            .map(|&row| self.lit(body, Literal::F32(f32::from_bits(pick(row)))))
            .collect();
        let ty = self.types.vec4;
        self.expr(body, Expression::Compose { ty, components })
    }
}
