//! Slice 16b, the warp operations and derivatives (the `warp` module): what each lowers to, its
//! requirements and stages, and its results on the host's GPU, against the interpreter on a
//! warp of one active lane and against a 32-lane model on a full warp.

use naga::{DerivativeAxis, DerivativeControl, Direction, Expression, GatherMode, Statement};

use super::ops::{
    Case, ONE_INVOCATION, bit, boolean, cases_function, f, float, gpu_mismatches, int, op, u,
};
use super::run::gpu_lanes;
use super::*;
use crate::interp::{QUAD_LANE_MASK, SWIZZLE_ADD_FACTORS};
use crate::ir::{Builder, WARP_SIZE};
use crate::translate::control::{QUAD_SHUFFLE_MASK, SHFL_FIELD_MASK, SHFL_SEGMENT_SHIFT};

/// The value every shuffle and quad case moves.
const VALUE: u32 = 0x1234_5678;
/// The clamp and segment mask of a `SHFL` with the quad mask, as the control translator splits
/// them (`translate::control`'s `QUAD_SHUFFLE_MASK`): clamp 3, segment mask `0b1_1100`, so
/// each group of four lanes shuffles among itself.
const QUAD_CLAMP: u32 = QUAD_SHUFFLE_MASK & SHFL_FIELD_MASK;
const QUAD_SEGMENT: u32 = (QUAD_SHUFFLE_MASK >> SHFL_SEGMENT_SHIFT) & SHFL_FIELD_MASK;

/// Builds the arguments of a case's instruction.
type Args = fn(&mut Builder<'_>) -> Vec<Operand>;

/// `value` as an operand computed at run time (`IAdd32 value, 0`), not an immediate.
fn computed(builder: &mut Builder<'_>, value: u32) -> Operand {
    op(builder, Opcode::IAdd32, &[u(value), u(0)]).into()
}

/// A function of one block holding the arguments `args` builds and then `opcode` of them;
/// returns it and the instruction's id.
fn alone(opcode: Opcode, args: Args) -> (Function, Value) {
    let mut function = Function::new();
    let entry = function.entry();
    let mut builder = Builder::new(&mut function, entry);
    let args = args(&mut builder);
    let at = op(&mut builder, opcode, &args);
    builder
        .terminate(Terminator::Return)
        .expect("the block returns");
    (function, at)
}

/// One row of [`ROWS`]: an opcode, its arguments, whether it needs subgroups, whether it needs
/// a subgroup size of 32, and whether it reads the lane id.
type Row = (Opcode, Args, bool, bool, bool);

/// Every warp operation and derivative the slice lowers, with the requirements the owner's
/// ruling of 2026-11-20 gives each, written out by hand (not read from
/// `needs_subgroup_size_32`): the size flag exactly where the result depends on 32 lanes.
/// `QuadBroadcast` has a row for each of its two lowerings.
const ROWS: &[Row] = &[
    (Opcode::LaneId, |_| vec![], true, true, true),
    (Opcode::VoteAll, |_| vec![bit(true)], true, true, false),
    (Opcode::VoteAny, |_| vec![bit(false)], true, true, false),
    (Opcode::VoteEqual, |_| vec![bit(true)], true, true, false),
    (
        Opcode::SubgroupBallot,
        |_| vec![bit(true)],
        true,
        true,
        false,
    ),
    (Opcode::SubgroupEqMask, |_| vec![], true, true, true),
    (Opcode::SubgroupLtMask, |_| vec![], true, true, true),
    (Opcode::SubgroupLeMask, |_| vec![], true, true, true),
    (Opcode::SubgroupGtMask, |_| vec![], true, true, true),
    (Opcode::SubgroupGeMask, |_| vec![], true, true, true),
    (
        Opcode::ShuffleIndex,
        |b| vec![u(VALUE), computed(b, 3), u(31), u(0)],
        true,
        true,
        true,
    ),
    (
        Opcode::ShuffleUp,
        |_| vec![u(VALUE), u(1), u(0), u(0)],
        true,
        true,
        true,
    ),
    (
        Opcode::ShuffleDown,
        |_| vec![u(VALUE), u(2), u(31), u(0)],
        true,
        true,
        true,
    ),
    (
        Opcode::ShuffleButterfly,
        |_| vec![u(VALUE), u(3), u(31), u(0)],
        true,
        true,
        true,
    ),
    (
        Opcode::QuadBroadcast,
        |_| vec![u(VALUE), u(2)],
        true,
        false,
        false,
    ),
    (
        Opcode::QuadBroadcast,
        |b| vec![u(VALUE), computed(b, 2)],
        true,
        false,
        true,
    ),
    (
        Opcode::QuadSwap,
        |_| vec![u(VALUE), u(1)],
        true,
        false,
        false,
    ),
    (
        Opcode::FSwizzleAdd,
        |_| vec![f(1.5), f(2.25), u(0xe4)],
        true,
        false,
        true,
    ),
    (Opcode::DPdxFine, |_| vec![f(1.5)], false, false, false),
    (Opcode::DPdyFine, |_| vec![f(1.5)], false, false, false),
    (Opcode::DPdxCoarse, |_| vec![f(1.5)], false, false, false),
    (Opcode::DPdyCoarse, |_| vec![f(1.5)], false, false, false),
];

/// Whether `opcode` is a derivative (pixel programs only).
const fn is_derivative(opcode: Opcode) -> bool {
    matches!(
        opcode,
        Opcode::DPdxFine | Opcode::DPdyFine | Opcode::DPdxCoarse | Opcode::DPdyCoarse
    )
}

/// Each row, alone, in each stage that may use it (compute and pixel programs; derivatives
/// pixel only): the module validates with exactly the capabilities its requirements name,
/// under this crate's flags and every flag; the requirements are the row's; the lane id is an
/// input exactly when the row reads it.
#[test]
fn every_warp_operation_lowers_with_its_requirements() {
    for &(opcode, args, subgroup, subgroup_size_32, lane) in ROWS {
        let stages: &[GuestStage] = if is_derivative(opcode) {
            &[GuestStage::Pixel]
        } else {
            &[GuestStage::Compute, GuestStage::Pixel]
        };
        for &stage in stages {
            let (function, _) = alone(opcode, args);
            let shader = lowered(stage, &function, &LowerInputs::DEFAULT);
            assert_eq!(
                shader.requirements,
                Requirements {
                    subgroup,
                    subgroup_size_32,
                    ..Requirements::default()
                },
                "{opcode} in a {stage} program"
            );
            let reads_lane = shader
                .inputs
                .iter()
                .any(|slot| slot.guest == GuestSlot::LaneId);
            assert_eq!(reads_lane, lane, "{opcode} in a {stage} program");
        }
    }
}

/// A module whose only subgroup operations are quad operations needs the subgroup capability
/// but not the size flag (the spec's slice 16b row), so it lowers for a host with subgroups of
/// another size; adding a vote makes it need the size, refused there with the exact feature.
#[test]
fn a_quad_only_module_needs_subgroups_but_not_size_32() {
    let quads = |vote: bool| {
        let mut function = Function::new();
        let entry = function.entry();
        let mut b = Builder::new(&mut function, entry);
        let index = computed(&mut b, 1);
        op(&mut b, Opcode::QuadBroadcast, &[u(VALUE), u(3)]);
        op(&mut b, Opcode::QuadBroadcast, &[u(VALUE), index]);
        op(&mut b, Opcode::QuadSwap, &[u(VALUE), u(2)]);
        op(&mut b, Opcode::FSwizzleAdd, &[f(1.0), f(2.0), u(0x1b)]);
        if vote {
            op(&mut b, Opcode::VoteAny, &[bit(true)]);
        }
        b.terminate(Terminator::Return).expect("the block returns");
        function
    };
    let other_size = LowerInputs {
        caps: HostCaps {
            subgroup_size_32: false,
            ..HostCaps::ALL
        },
        ..LowerInputs::DEFAULT
    };
    let shader = lowered(GuestStage::Pixel, &quads(false), &other_size);
    assert_eq!(
        shader.requirements,
        Requirements {
            subgroup: true,
            ..Requirements::default()
        }
    );
    assert_eq!(
        lower_with(GuestStage::Pixel, &quads(true), &other_size).err(),
        Some(LowerError::Missing(Feature::SubgroupSize32))
    );
    assert_eq!(
        LowerError::Missing(Feature::SubgroupSize32).to_string(),
        "the shader needs a subgroup size of 32, which the host target lacks"
    );
    let shader = lowered(GuestStage::Pixel, &quads(true), &LowerInputs::DEFAULT);
    assert_eq!(
        shader.requirements,
        Requirements {
            subgroup: true,
            subgroup_size_32: true,
            ..Requirements::default()
        }
    );
}

/// Every warp operation but `LaneId` (which has its own error) is refused in a vertex program,
/// naming it (the `warp` module's divergence table).
#[test]
fn warp_operations_are_refused_in_vertex_programs() {
    let mut refused = 0;
    for &(opcode, args, ..) in ROWS {
        if opcode == Opcode::LaneId {
            continue;
        }
        let (function, at) = alone(opcode, args);
        assert_eq!(
            lower_with(GuestStage::Vertex, &function, &LowerInputs::DEFAULT).err(),
            Some(LowerError::NotInStage {
                at,
                opcode,
                stage: GuestStage::Vertex
            })
        );
        refused += 1;
    }
    // Every row but `LaneId`, `QuadBroadcast` twice.
    assert_eq!(refused, ROWS.len() - 1);
    let (function, _) = alone(Opcode::LaneId, |_| vec![]);
    assert_eq!(
        lower_with(GuestStage::Vertex, &function, &LowerInputs::DEFAULT).err(),
        Some(LowerError::LaneIdInVertex)
    );
}

/// The derivatives are refused outside pixel programs, in vertex and compute programs alike,
/// and lower in a pixel program as naga's derivative along their axis with their control.
#[test]
fn derivatives_are_refused_outside_pixel_programs() {
    let cases = [
        (Opcode::DPdxFine, DerivativeAxis::X, DerivativeControl::Fine),
        (Opcode::DPdyFine, DerivativeAxis::Y, DerivativeControl::Fine),
        (
            Opcode::DPdxCoarse,
            DerivativeAxis::X,
            DerivativeControl::Coarse,
        ),
        (
            Opcode::DPdyCoarse,
            DerivativeAxis::Y,
            DerivativeControl::Coarse,
        ),
    ];
    for (opcode, axis, ctrl) in cases {
        let (function, at) = alone(opcode, |_| vec![f(1.5)]);
        for stage in [GuestStage::Vertex, GuestStage::Compute] {
            assert_eq!(
                lower_with(stage, &function, &LowerInputs::DEFAULT).err(),
                Some(LowerError::NotInStage { at, opcode, stage })
            );
        }
        let shader = lowered(GuestStage::Pixel, &function, &LowerInputs::DEFAULT);
        let derivatives: Vec<_> = naga_module(&shader).entry_points[0]
            .function
            .expressions
            .iter()
            .filter_map(|(_, expression)| match *expression {
                Expression::Derivative { axis, ctrl, .. } => Some((axis, ctrl)),
                _ => None,
            })
            .collect();
        assert_eq!(derivatives, [(axis, ctrl)], "{opcode}");
    }
}

/// What a naga gather reads, with a constant index read back as its value (`None` when the
/// index is not a literal).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Gather {
    BroadcastFirst,
    Broadcast(Option<u32>),
    Shuffle,
    Up,
    Down,
    Xor,
    QuadBroadcast(Option<u32>),
    QuadSwap(Direction),
}

/// The gathers of `shader`'s entry point (one block), in order.
fn gathers(shader: &HostShader) -> Vec<Gather> {
    let function = &naga_module(shader).entry_points[0].function;
    let constant = |index| match function.expressions[index] {
        Expression::Literal(naga::Literal::U32(value)) => Some(value),
        _ => None,
    };
    function
        .body
        .iter()
        .filter_map(|statement| match *statement {
            Statement::SubgroupGather { mode, .. } => Some(match mode {
                GatherMode::BroadcastFirst => Gather::BroadcastFirst,
                GatherMode::Broadcast(index) => Gather::Broadcast(constant(index)),
                GatherMode::Shuffle(_) => Gather::Shuffle,
                GatherMode::ShuffleUp(_) => Gather::Up,
                GatherMode::ShuffleDown(_) => Gather::Down,
                GatherMode::ShuffleXor(_) => Gather::Xor,
                GatherMode::QuadBroadcast(index) => Gather::QuadBroadcast(constant(index)),
                GatherMode::QuadSwap(direction) => Gather::QuadSwap(direction),
            }),
            _ => None,
        })
        .collect()
}

/// `opcode` of the arguments `args` builds, lowered as a compute shader; its gathers.
fn gathers_of(opcode: Opcode, args: Args) -> Vec<Gather> {
    let (function, _) = alone(opcode, args);
    gathers(&lowered(
        GuestStage::Compute,
        &function,
        &LowerInputs::DEFAULT,
    ))
}

/// `QuadSwap` takes the immediate directions 0, 1 and 2 as naga's `X`, `Y` and `Diagonal`;
/// the immediate 3 (one past the last), `u32::MAX` and a computed direction are refused as
/// argument 1.
#[test]
fn quad_swap_takes_an_immediate_direction() {
    assert_eq!(
        gathers_of(Opcode::QuadSwap, |_| vec![u(VALUE), u(0)]),
        [Gather::QuadSwap(Direction::X)]
    );
    assert_eq!(
        gathers_of(Opcode::QuadSwap, |_| vec![u(VALUE), u(1)]),
        [Gather::QuadSwap(Direction::Y)]
    );
    assert_eq!(
        gathers_of(Opcode::QuadSwap, |_| vec![u(VALUE), u(2)]),
        [Gather::QuadSwap(Direction::Diagonal)]
    );
    let refused: [Args; 3] = [
        |_| vec![u(VALUE), u(3)],
        |_| vec![u(VALUE), u(u32::MAX)],
        |b| vec![u(VALUE), computed(b, 1)],
    ];
    for args in refused {
        let (function, at) = alone(Opcode::QuadSwap, args);
        assert_eq!(
            lower_with(GuestStage::Compute, &function, &LowerInputs::DEFAULT).err(),
            Some(LowerError::Operand { at, index: 1 })
        );
    }
}

/// `QuadBroadcast` with an immediate index is naga's quad broadcast of `index & 3` (3 stays 3,
/// 4 becomes 0, 7 becomes 3); with a computed index it is a shuffle (Eden's fallback).
#[test]
fn quad_broadcast_masks_an_immediate_index_and_shuffles_a_computed_one() {
    let cases: [(Args, Gather); 4] = [
        (|_| vec![u(VALUE), u(3)], Gather::QuadBroadcast(Some(3))),
        (|_| vec![u(VALUE), u(4)], Gather::QuadBroadcast(Some(0))),
        (|_| vec![u(VALUE), u(7)], Gather::QuadBroadcast(Some(3))),
        (|b| vec![u(VALUE), computed(b, 3)], Gather::Shuffle),
    ];
    for (args, gather) in cases {
        assert_eq!(gathers_of(Opcode::QuadBroadcast, args), [gather]);
    }
    assert_eq!(QUAD_LANE_MASK, 3, "Eden's `3u`");
}

/// `ShuffleIndex` broadcasts `lane & 31` when the lane and the segment mask are immediates and
/// the mask is 0, and shuffles otherwise (a segment of 1, a computed lane, a computed segment
/// of 0); the relative shuffles gather up, down and by exclusive or.
#[test]
fn each_shuffle_gathers_as_edens_does() {
    let cases: [(Opcode, Args, Gather); 8] = [
        (
            Opcode::ShuffleIndex,
            |_| vec![u(VALUE), u(5), u(31), u(0)],
            Gather::Broadcast(Some(5)),
        ),
        (
            Opcode::ShuffleIndex,
            |_| vec![u(VALUE), u(37), u(31), u(0)],
            Gather::Broadcast(Some(5)),
        ),
        (
            Opcode::ShuffleIndex,
            |_| vec![u(VALUE), u(5), u(31), u(1)],
            Gather::Shuffle,
        ),
        (
            Opcode::ShuffleIndex,
            |b| vec![u(VALUE), computed(b, 5), u(31), u(0)],
            Gather::Shuffle,
        ),
        (
            Opcode::ShuffleIndex,
            |b| vec![u(VALUE), u(5), u(31), computed(b, 0)],
            Gather::Shuffle,
        ),
        (
            Opcode::ShuffleUp,
            |_| vec![u(VALUE), u(1), u(0), u(0)],
            Gather::Up,
        ),
        (
            Opcode::ShuffleDown,
            |_| vec![u(VALUE), u(1), u(31), u(0)],
            Gather::Down,
        ),
        (
            Opcode::ShuffleButterfly,
            |_| vec![u(VALUE), u(1), u(31), u(0)],
            Gather::Xor,
        ),
    ];
    for (opcode, args, gather) in cases {
        assert_eq!(gathers_of(opcode, args), [gather], "{opcode}");
    }
}

/// `GetInBoundsFromOp` of each shuffle is the very flag the shuffle selects its value on; of
/// anything else (a quad operation, an integer operation) it is refused as argument 0.
#[test]
fn in_bounds_is_its_shuffles_flag() {
    for opcode in [
        Opcode::ShuffleIndex,
        Opcode::ShuffleUp,
        Opcode::ShuffleDown,
        Opcode::ShuffleButterfly,
    ] {
        let mut function = Function::new();
        let entry = function.entry();
        let mut b = Builder::new(&mut function, entry);
        let index = computed(&mut b, 1);
        let shuffle = op(&mut b, opcode, &[u(VALUE), index, u(31), u(0)]);
        let in_bounds = op(&mut b, Opcode::GetInBoundsFromOp, &[shuffle.into()]);
        let pred = Operand::Imm(Imm::Pred(crate::decode::Pred::new(1).expect("p1")));
        op(&mut b, Opcode::SetPred, &[pred, in_bounds.into()]);
        b.terminate(Terminator::Return).expect("the block returns");
        let shader = lowered(GuestStage::Compute, &function, &LowerInputs::DEFAULT);
        let naga_function = &naga_module(&shader).entry_points[0].function;
        let condition = naga_function
            .expressions
            .iter()
            .find_map(|(_, expression)| match *expression {
                Expression::Select { condition, .. } => Some(condition),
                _ => None,
            })
            .expect("the shuffle's select");
        let stored = naga_function
            .body
            .iter()
            .find_map(|statement| match *statement {
                Statement::Store { value, .. } => Some(value),
                _ => None,
            })
            .expect("the predicate's store");
        assert_eq!(stored, condition, "{opcode}");
    }
    let others: [(Opcode, Args); 2] = [
        (Opcode::QuadSwap, |_| vec![u(VALUE), u(0)]),
        (Opcode::IAdd32, |_| vec![u(1), u(2)]),
    ];
    for (opcode, args) in others {
        let mut function = Function::new();
        let entry = function.entry();
        let mut b = Builder::new(&mut function, entry);
        let args = args(&mut b);
        let of = op(&mut b, opcode, &args);
        let at = op(&mut b, Opcode::GetInBoundsFromOp, &[of.into()]);
        b.terminate(Terminator::Return).expect("the block returns");
        assert_eq!(
            lower_with(GuestStage::Compute, &function, &LowerInputs::DEFAULT).err(),
            Some(LowerError::Operand { at, index: 0 }),
            "{opcode}"
        );
    }
}

/// `opcode` of `args` (a shuffle), and its `GetInBoundsFromOp` as 1 or 0.
fn in_bounds(builder: &mut Builder<'_>, opcode: Opcode, args: &[Operand]) -> Value {
    let shuffle = op(builder, opcode, args);
    boolean(builder, Opcode::GetInBoundsFromOp, &[shuffle.into()])
}

/// The warp operations on a warp whose only active lane is lane 0, where the interpreter models
/// them ([`crate::interp`], "Warp operations"): each shuffle where its source is out of range
/// or is lane 0 (another lane's value is undefined on the GPU), the in-range flags on both
/// sides of each comparison's edge and of `GuestLane`'s `& 31`, and `FSwizzleAdd` for each of
/// lane 0's swizzle pairs.
const ONE_LANE_CASES: &[Case] = &[
    ("vote all true", |b| {
        boolean(b, Opcode::VoteAll, &[bit(true)])
    }),
    ("vote all false", |b| {
        boolean(b, Opcode::VoteAll, &[bit(false)])
    }),
    ("vote any true", |b| {
        boolean(b, Opcode::VoteAny, &[bit(true)])
    }),
    ("vote any false", |b| {
        boolean(b, Opcode::VoteAny, &[bit(false)])
    }),
    ("vote equal true", |b| {
        boolean(b, Opcode::VoteEqual, &[bit(true)])
    }),
    ("vote equal false", |b| {
        boolean(b, Opcode::VoteEqual, &[bit(false)])
    }),
    ("ballot true", |b| {
        int(b, Opcode::SubgroupBallot, &[bit(true)])
    }),
    ("ballot false", |b| {
        int(b, Opcode::SubgroupBallot, &[bit(false)])
    }),
    ("eq mask", |b| int(b, Opcode::SubgroupEqMask, &[])),
    ("lt mask", |b| int(b, Opcode::SubgroupLtMask, &[])),
    ("le mask", |b| int(b, Opcode::SubgroupLeMask, &[])),
    ("gt mask", |b| int(b, Opcode::SubgroupGtMask, &[])),
    ("ge mask", |b| int(b, Opcode::SubgroupGeMask, &[])),
    ("index broadcast own lane", |b| {
        int(b, Opcode::ShuffleIndex, &[u(VALUE), u(0), u(31), u(0)])
    }),
    ("index broadcast lane 32", |b| {
        int(b, Opcode::ShuffleIndex, &[u(VALUE), u(32), u(0), u(0)])
    }),
    ("index broadcast out of range", |b| {
        int(b, Opcode::ShuffleIndex, &[u(VALUE), u(5), u(0), u(0)])
    }),
    ("index quad segment own lane", |b| {
        int(
            b,
            Opcode::ShuffleIndex,
            &[u(VALUE), u(0), u(31), u(QUAD_SEGMENT)],
        )
    }),
    ("index quad segment out of range", |b| {
        int(
            b,
            Opcode::ShuffleIndex,
            &[u(VALUE), u(3), u(1), u(QUAD_SEGMENT)],
        )
    }),
    ("index in bounds lane 32 clamp 0", |b| {
        in_bounds(b, Opcode::ShuffleIndex, &[u(VALUE), u(32), u(0), u(0)])
    }),
    ("index in bounds lane 1 clamp 0", |b| {
        in_bounds(b, Opcode::ShuffleIndex, &[u(VALUE), u(1), u(0), u(0)])
    }),
    ("index in bounds lane 1 clamp 1", |b| {
        in_bounds(b, Opcode::ShuffleIndex, &[u(VALUE), u(1), u(1), u(0)])
    }),
    ("index in bounds quad lane 2", |b| {
        in_bounds(
            b,
            Opcode::ShuffleIndex,
            &[u(VALUE), u(2), u(31), u(QUAD_SEGMENT)],
        )
    }),
    ("index in bounds quad lane 3 clamp 1", |b| {
        in_bounds(
            b,
            Opcode::ShuffleIndex,
            &[u(VALUE), u(3), u(1), u(QUAD_SEGMENT)],
        )
    }),
    ("up 0", |b| {
        int(b, Opcode::ShuffleUp, &[u(VALUE), u(0), u(0), u(0)])
    }),
    ("up 1", |b| {
        int(b, Opcode::ShuffleUp, &[u(VALUE), u(1), u(0), u(0)])
    }),
    ("up in bounds 0", |b| {
        in_bounds(b, Opcode::ShuffleUp, &[u(VALUE), u(0), u(0), u(0)])
    }),
    ("up in bounds 0 clamp 1", |b| {
        in_bounds(b, Opcode::ShuffleUp, &[u(VALUE), u(0), u(1), u(0)])
    }),
    ("up in bounds 1", |b| {
        in_bounds(b, Opcode::ShuffleUp, &[u(VALUE), u(1), u(0), u(0)])
    }),
    ("up in bounds 32", |b| {
        in_bounds(b, Opcode::ShuffleUp, &[u(VALUE), u(32), u(0), u(0)])
    }),
    ("down 0", |b| {
        int(b, Opcode::ShuffleDown, &[u(VALUE), u(0), u(31), u(0)])
    }),
    ("down 3 clamp 2", |b| {
        int(b, Opcode::ShuffleDown, &[u(VALUE), u(3), u(2), u(0)])
    }),
    ("down in bounds 2 clamp 2", |b| {
        in_bounds(b, Opcode::ShuffleDown, &[u(VALUE), u(2), u(2), u(0)])
    }),
    ("down in bounds 3 clamp 2", |b| {
        in_bounds(b, Opcode::ShuffleDown, &[u(VALUE), u(3), u(2), u(0)])
    }),
    ("down in bounds 32 clamp 0", |b| {
        in_bounds(b, Opcode::ShuffleDown, &[u(VALUE), u(32), u(0), u(0)])
    }),
    ("butterfly 0", |b| {
        int(b, Opcode::ShuffleButterfly, &[u(VALUE), u(0), u(31), u(0)])
    }),
    ("butterfly 4 clamp 3", |b| {
        int(b, Opcode::ShuffleButterfly, &[u(VALUE), u(4), u(3), u(0)])
    }),
    ("butterfly in bounds 3 clamp 3", |b| {
        in_bounds(b, Opcode::ShuffleButterfly, &[u(VALUE), u(3), u(3), u(0)])
    }),
    ("butterfly in bounds 4 clamp 3", |b| {
        in_bounds(b, Opcode::ShuffleButterfly, &[u(VALUE), u(4), u(3), u(0)])
    }),
    ("butterfly in bounds 32 clamp 0", |b| {
        in_bounds(b, Opcode::ShuffleButterfly, &[u(VALUE), u(32), u(0), u(0)])
    }),
    ("quad broadcast 0", |b| {
        int(b, Opcode::QuadBroadcast, &[u(VALUE), u(0)])
    }),
    ("quad broadcast 4", |b| {
        int(b, Opcode::QuadBroadcast, &[u(VALUE), u(4)])
    }),
    ("quad broadcast computed 4", |b| {
        let index = computed(b, 4);
        int(b, Opcode::QuadBroadcast, &[u(VALUE), index])
    }),
    ("swizzle add 0", |b| {
        float(b, Opcode::FSwizzleAdd, &[f(1.5), f(2.25), u(0)])
    }),
    ("swizzle add 1", |b| {
        float(b, Opcode::FSwizzleAdd, &[f(1.5), f(2.25), u(1)])
    }),
    ("swizzle add 2", |b| {
        float(b, Opcode::FSwizzleAdd, &[f(1.5), f(2.25), u(2)])
    }),
    ("swizzle add 3", |b| {
        float(b, Opcode::FSwizzleAdd, &[f(1.5), f(2.25), u(3)])
    }),
    ("swizzle add other lanes' bits", |b| {
        float(b, Opcode::FSwizzleAdd, &[f(1.5), f(2.25), u(0xfffc | 1)])
    }),
];

/// The warp operations give the interpreter's results on the host's GPU, on a workgroup of one
/// invocation (lane 0 alone in its subgroup).
#[test]
fn warp_operations_match_the_interpreter_on_the_gpu() {
    let (function, _) = cases_function(ONE_LANE_CASES);
    let shader = lowered(GuestStage::Compute, &function, &ONE_INVOCATION);
    assert!(shader.requirements.subgroup && shader.requirements.subgroup_size_32);
    let mismatches = gpu_mismatches("warp_operations", ONE_LANE_CASES);
    assert!(mismatches.is_empty(), "{mismatches:#?}");
}

/// The registers of [`full_warp_function`]: the lane id, the active-lane ballot, then each
/// case [`full_warp_model`] computes, in order.
const FULL_WARP_REGS: u8 = 30;

/// `v(lane)`, the value each lane moves in the full-warp cases.
const fn lane_value(lane: u32) -> u32 {
    lane.wrapping_mul(0x101).wrapping_add(7)
}

/// `FSwizzleAdd a, b, swizzle` on `lane`, from Eden's tables in `f32` arithmetic (each product
/// is exact, so the sum's one rounding is the only one).
fn swizzle_add_model(a: f32, b: f32, swizzle: u32, lane: u32) -> u32 {
    let index = (swizzle >> ((lane & QUAD_LANE_MASK) * 2)) & QUAD_LANE_MASK;
    let (factor_a, factor_b) = SWIZZLE_ADD_FACTORS[usize::try_from(index).expect("small")];
    (a * f32::from_bits(factor_a) + b * f32::from_bits(factor_b)).to_bits()
}

/// A function of one block that writes, in each lane, register 0 the lane id, register 1
/// `SubgroupBallot(true)`, and registers 2 up the cases [`full_warp_model`] lists.
fn full_warp_function() -> Function {
    let mut function = Function::new();
    let entry = function.entry();
    let mut b = Builder::new(&mut function, entry);
    let lane: Operand = op(&mut b, Opcode::LaneId, &[]).into();
    let scaled = op(&mut b, Opcode::IMul32, &[lane, u(0x101)]);
    let value: Operand = op(&mut b, Opcode::IAdd32, &[scaled.into(), u(7)]).into();
    let low_bit = op(&mut b, Opcode::BitwiseAnd32, &[lane, u(1)]);
    let odd: Operand = op(&mut b, Opcode::INotEqual, &[low_bit.into(), u(0)]).into();
    let cmp =
        |b: &mut Builder<'_>, opcode, right| -> Operand { op(b, opcode, &[lane, u(right)]).into() };
    let mut results = vec![
        lane,
        int(&mut b, Opcode::SubgroupBallot, &[bit(true)]).into(),
    ];
    let mut push = |value: Value| results.push(value.into());
    push(int(&mut b, Opcode::SubgroupBallot, &[odd]));
    let below_32 = cmp(&mut b, Opcode::ULessThan, WARP_SIZE);
    push(boolean(&mut b, Opcode::VoteAll, &[below_32]));
    let not_5 = cmp(&mut b, Opcode::INotEqual, 5);
    push(boolean(&mut b, Opcode::VoteAll, &[not_5]));
    let is_5 = cmp(&mut b, Opcode::IEqual, 5);
    push(boolean(&mut b, Opcode::VoteAny, &[is_5]));
    let is_40 = cmp(&mut b, Opcode::IEqual, 40);
    push(boolean(&mut b, Opcode::VoteAny, &[is_40]));
    push(boolean(&mut b, Opcode::VoteEqual, &[odd]));
    let below_40 = cmp(&mut b, Opcode::ULessThan, 40);
    push(boolean(&mut b, Opcode::VoteEqual, &[below_40]));
    push(boolean(&mut b, Opcode::VoteEqual, &[is_40]));
    for mask in [
        Opcode::SubgroupEqMask,
        Opcode::SubgroupLtMask,
        Opcode::SubgroupLeMask,
        Opcode::SubgroupGtMask,
        Opcode::SubgroupGeMask,
    ] {
        push(int(&mut b, mask, &[]));
    }
    push(int(
        &mut b,
        Opcode::ShuffleIndex,
        &[value, u(5), u(31), u(0)],
    ));
    let flipped: Operand = op(&mut b, Opcode::BitwiseXor32, &[lane, u(3)]).into();
    push(int(
        &mut b,
        Opcode::ShuffleIndex,
        &[value, flipped, u(31), u(0)],
    ));
    push(int(
        &mut b,
        Opcode::ShuffleIndex,
        &[value, u(1), u(QUAD_CLAMP), u(QUAD_SEGMENT)],
    ));
    let up = op(&mut b, Opcode::ShuffleUp, &[value, u(1), u(0), u(0)]);
    push(up);
    let down = op(&mut b, Opcode::ShuffleDown, &[value, u(2), u(31), u(0)]);
    push(down);
    push(int(
        &mut b,
        Opcode::ShuffleButterfly,
        &[value, u(1), u(31), u(0)],
    ));
    push(boolean(&mut b, Opcode::GetInBoundsFromOp, &[up.into()]));
    push(boolean(&mut b, Opcode::GetInBoundsFromOp, &[down.into()]));
    push(int(&mut b, Opcode::QuadBroadcast, &[value, u(2)]));
    let three = computed(&mut b, 3);
    push(int(&mut b, Opcode::QuadBroadcast, &[value, three]));
    for direction in 0..3 {
        push(int(&mut b, Opcode::QuadSwap, &[value, u(direction)]));
    }
    let as_float: Operand = op(&mut b, Opcode::ConvertF32U32, &[lane]).into();
    for swizzle in [0xe4, 0x1b] {
        push(float(
            &mut b,
            Opcode::FSwizzleAdd,
            &[as_float, f(0.5), u(swizzle)],
        ));
    }
    assert_eq!(results.len(), usize::from(FULL_WARP_REGS));
    for (reg, result) in (0..FULL_WARP_REGS).zip(results) {
        op(
            &mut b,
            Opcode::SetRegister,
            &[Operand::Imm(Imm::Reg(Gpr::new(reg))), result],
        );
    }
    b.terminate(Terminator::Return).expect("the block returns");
    function
}

/// What [`full_warp_function`] leaves in `lane` on a warp of 32 active lanes, from Eden's
/// definitions (`emit_spirv_warp.cpp` @ cf654052: the votes and ballot over the active lanes,
/// the shuffles' source lanes and range checks, the quad lanes, `EmitFSwizzleAdd`'s tables).
fn full_warp_model(lane: u32) -> Vec<u32> {
    let v = lane_value;
    let all_lanes = u32::MAX;
    let bool_word = u32::from;
    let equal = 1_u32 << lane;
    let below = equal - 1;
    let up_in_range = lane >= 1;
    let down_in_range = lane + 2 < WARP_SIZE;
    let quad_base = lane & !QUAD_LANE_MASK;
    let lane_float = f32::from(u16::try_from(lane).expect("a lane of the warp"));
    vec![
        lane,
        all_lanes,
        0xaaaa_aaaa,
        1,
        0,
        1,
        0,
        0,
        1,
        1,
        equal,
        below,
        below | equal,
        !(below | equal),
        !below,
        v(5),
        v(lane ^ 3),
        v((lane & QUAD_SEGMENT) | 1),
        if up_in_range { v(lane - 1) } else { v(lane) },
        if down_in_range { v(lane + 2) } else { v(lane) },
        v(lane ^ 1),
        bool_word(up_in_range),
        bool_word(down_in_range),
        v(quad_base | 2),
        v(quad_base | 3),
        v(lane ^ 1),
        v(lane ^ 2),
        v(lane ^ 3),
        swizzle_add_model(lane_float, 0.5, 0xe4, lane),
        swizzle_add_model(lane_float, 0.5, 0x1b, lane),
    ]
}

/// The warp operations on a full warp of 32 active lanes, where each reads other lanes,
/// against [`full_warp_model`]. Skipped, with a notice, where the host's subgroup holding the
/// workgroup is not 32 lanes wide (lane 0's ballot of every lane is not all ones).
#[test]
fn warp_operations_match_a_32_lane_model_on_the_gpu() {
    let function = full_warp_function();
    let inputs = LowerInputs {
        workgroup_size: [WARP_SIZE, 1, 1],
        ..LowerInputs::DEFAULT
    };
    let shader = lowered(GuestStage::Compute, &function, &inputs);
    let regs: Vec<u8> = (0..FULL_WARP_REGS).collect();
    let test = "warp_operations_match_a_32_lane_model_on_the_gpu";
    let Some(lanes) = gpu_lanes(test, &shader, &regs, WARP_SIZE) else {
        return;
    };
    if lanes.iter().any(|lane| lane[1] != u32::MAX) {
        eprintln!("skipping the GPU half of {test}: the subgroup is not 32 lanes wide");
        return;
    }
    let mut seen: Vec<u32> = lanes.iter().map(|lane| lane[0]).collect();
    seen.sort_unstable();
    assert_eq!(seen, (0..WARP_SIZE).collect::<Vec<_>>());
    let mut mismatches = Vec::new();
    for registers in &lanes {
        let lane = registers[0];
        let expected = full_warp_model(lane);
        for (reg, (expected, found)) in expected.iter().zip(registers).enumerate() {
            if expected != found {
                mismatches.push(format!(
                    "lane {lane}, r{reg}: model {expected:#x}, GPU {found:#x}"
                ));
            }
        }
    }
    assert!(mismatches.is_empty(), "{mismatches:#?}");
}

/// The MSL naga writes for the full-warp module and for the four derivatives in a pixel
/// program (snapshots: the golden text for the derivatives, which the interpreter does not
/// model, and for the gathers' forms).
#[test]
fn warp_snapshot() {
    let inputs = LowerInputs {
        workgroup_size: [WARP_SIZE, 1, 1],
        ..LowerInputs::DEFAULT
    };
    let shader = lowered(GuestStage::Compute, &full_warp_function(), &inputs);
    insta::assert_snapshot!("warp_operations", msl(&shader));
    let mut function = Function::new();
    let entry = function.entry();
    let mut b = Builder::new(&mut function, entry);
    let x = op(
        &mut b,
        Opcode::GetRegister,
        &[Operand::Imm(Imm::Reg(Gpr::new(0)))],
    );
    let x = op(&mut b, Opcode::BitCastF32U32, &[x.into()]);
    for (component, opcode) in (0..).zip([
        Opcode::DPdxFine,
        Opcode::DPdyFine,
        Opcode::DPdxCoarse,
        Opcode::DPdyCoarse,
    ]) {
        let derivative = op(&mut b, opcode, &[x.into()]);
        op(
            &mut b,
            Opcode::SetFragColor,
            &[u(0), u(component), derivative.into()],
        );
    }
    b.terminate(Terminator::Return).expect("the block returns");
    let shader = lowered(GuestStage::Pixel, &function, &LowerInputs::DEFAULT);
    insta::assert_snapshot!("warp_derivatives", msl(&shader));
}
