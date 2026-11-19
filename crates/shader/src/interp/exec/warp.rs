//! The warp operations on a warp whose only active lane is the running invocation (the parent
//! module's "Semantics", "Warp operations"; added by slice 12h for the control and warp
//! translator's tests).
//!
//! Each operation follows the code Eden's SPIR-V emitter writes for it when the host stage has
//! subgroups and its warp is not wider than the guest's
//! (`eden/src/shader_recompiler/backend/spirv/emit_spirv_warp.cpp` @ cf654052,
//! GPL-3.0-or-later):
//!
//! - `VoteAll` and `VoteAny` (`OpGroupNonUniformAll`, `OpGroupNonUniformAny`) give their
//!   argument, and `VoteEqual` (`OpGroupNonUniformAllEqual`) gives true: one lane votes.
//! - `SubgroupBallot` (`OpGroupNonUniformBallot`, component 0) gives `1 << lane` when its
//!   argument is true, else 0.
//! - The shuffles compute their source lane and in-range flag as `EmitShuffleIndex`,
//!   `EmitShuffleUp`, `EmitShuffleDown` and `EmitShuffleButterfly` do (`ComputeMinThreadId`,
//!   `ComputeMaxThreadId`, `GuestLane`'s `& 31`, and their signed comparisons), and select the
//!   shuffled value only in range, as each does with `OpSelect`. The shuffled value is this
//!   lane's own when the source is this lane, and [`INACTIVE_LANE_VALUE`] otherwise.
//! - `QuadBroadcast` reads lane `(lane & !3) | (index & 3)` and `QuadSwap` lane
//!   `lane ^ (direction + 1)`, the lanes Eden's fallbacks without quad shuffles read
//!   (`EmitQuadBroadcast`, `EmitQuadSwap`), with the same rule for another lane.
//! - `GetInBoundsFromOp` gives the in-range flag the shuffle it names last computed
//!   (`SetInBoundsFlag`); naming anything else is [`InterpError::Unsupported`].
//!
//! The lane id is checked against [`WARP_SIZE`] by every operation that reads it.

use super::super::{INACTIVE_LANE_VALUE, InterpError, Run, Val};
use super::{Args, Handled};
use crate::ir::{Opcode, Operand, Operation, Value, WARP_SIZE};

/// `GuestLane`'s mask: a lane index modulo the warp size.
const LANE_MASK: u32 = WARP_SIZE - 1;
/// The lanes of a quad, less one: `EmitQuadBroadcast`'s `~3u` and `3u`.
const QUAD_LANE_MASK: u32 = 3;

/// Which shuffle.
#[derive(Clone, Copy)]
enum Shuffle {
    Index,
    Up,
    Down,
    Butterfly,
}

/// The source lane and in-range flag of a shuffle run by `lane` (Eden's emitters, see the
/// module docs).
fn shuffle_source(kind: Shuffle, lane: u32, index: u32, clamp: u32, segment: u32) -> (u32, bool) {
    let not_segment = !segment;
    let min_lane = lane & segment;
    let max_lane = min_lane | (clamp & not_segment);
    let delta = index & LANE_MASK;
    // The emitters compare as signed 32-bit values (`OpSLessThanEqual`,
    // `OpSGreaterThanEqual`).
    let signed = |value: u32| value.cast_signed();
    match kind {
        Shuffle::Index => {
            let source = (delta & not_segment) | min_lane;
            (source, signed(source) <= signed(max_lane))
        }
        Shuffle::Up => {
            let source = lane.wrapping_sub(delta);
            (source, signed(source) >= signed(max_lane))
        }
        Shuffle::Down => {
            let source = lane.wrapping_add(delta);
            (source, signed(source) <= signed(max_lane))
        }
        Shuffle::Butterfly => {
            let source = lane ^ delta;
            (source, signed(source) <= signed(max_lane))
        }
    }
}

/// What reading `value` from lane `source` gives on the one-lane warp of `lane`.
const fn read_lane(value: u32, lane: u32, source: u32) -> u32 {
    if source == lane {
        value
    } else {
        INACTIVE_LANE_VALUE
    }
}

impl Run<'_> {
    /// The running invocation's lane, checked against the warp size.
    fn lane(&self, at: Value) -> Result<u32, InterpError> {
        let lane = self.machine.system.lane_id;
        if lane < WARP_SIZE {
            Ok(lane)
        } else {
            Err(InterpError::LaneOutsideWarp { at, lane })
        }
    }

    /// The warp operations: whether the opcode is one of them and, if so, its value.
    pub(super) fn warp(&mut self, args: &Args, op: &Operation) -> Result<Handled, InterpError> {
        use Opcode as O;
        let at = args.at;
        let shuffle = match op.opcode {
            O::ShuffleIndex => Some(Shuffle::Index),
            O::ShuffleUp => Some(Shuffle::Up),
            O::ShuffleDown => Some(Shuffle::Down),
            O::ShuffleButterfly => Some(Shuffle::Butterfly),
            _ => None,
        };
        if let Some(kind) = shuffle {
            let (value, index, clamp, segment) =
                (args.u32(0)?, args.u32(1)?, args.u32(2)?, args.u32(3)?);
            let lane = self.lane(at)?;
            let (source, in_range) = shuffle_source(kind, lane, index, clamp, segment);
            self.in_bounds.insert(at, in_range);
            let result = if in_range {
                read_lane(value, lane, source)
            } else {
                value
            };
            return Ok(Handled::Done(Some(Val::U32(result))));
        }
        let value = match op.opcode {
            O::VoteAll | O::VoteAny => Val::U1(args.u1(0)?),
            O::VoteEqual => {
                args.u1(0)?;
                Val::U1(true)
            }
            O::SubgroupBallot => {
                let vote = args.u1(0)?;
                let lane = self.lane(at)?;
                Val::U32(if vote { 1 << lane } else { 0 })
            }
            O::QuadBroadcast => {
                let (value, index) = (args.u32(0)?, args.u32(1)?);
                let lane = self.lane(at)?;
                let source = (lane & !QUAD_LANE_MASK) | (index & QUAD_LANE_MASK);
                Val::U32(read_lane(value, lane, source))
            }
            O::QuadSwap => {
                let (value, direction) = (args.u32(0)?, args.u32(1)?);
                let lane = self.lane(at)?;
                Val::U32(read_lane(value, lane, lane ^ direction.wrapping_add(1)))
            }
            O::GetInBoundsFromOp => {
                let of = match op.args.first() {
                    Some(Operand::Value(of)) => Some(*of),
                    _ => None,
                };
                match of.and_then(|of| self.in_bounds.get(&of)) {
                    Some(&in_range) => Val::U1(in_range),
                    None => {
                        return Err(InterpError::Unsupported {
                            at,
                            opcode: op.opcode,
                        });
                    }
                }
            }
            _ => return Ok(Handled::Not),
        };
        Ok(Handled::Done(Some(value)))
    }
}
