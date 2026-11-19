//! Dead code elimination: unused operations leave in one walk, side effects and conditions
//! stay, the walk order's limits are Eden's, and the side-effect list is Eden's.

use super::{after, after_each};
use crate::ir::Opcode;
use crate::opt::{Pass, may_have_side_effects};

#[test]
fn unused_operations_leave_and_side_effects_stay() {
    assert_eq!(
        after(
            "block0:
    %0: u32 = GetCbufU32 u32 0, u32 0
    %1: u32 = IAdd32 %0, u32 1
    %2: u32 = IMul32 %1, %1
    %3: f32 = BitCastF32U32 %0
    %4 = SetAttribute a0x80, %3, u32 0
    %5 = SetRegister r0, %2
    %6: u1 = GetPred p0
    %7 = DemoteToHelperInvocation
    return
",
            Pass::DeadCodeElimination
        ),
        "block0:
    %0: u32 = GetCbufU32 u32 0x0, u32 0x0
    %3: f32 = BitCastF32U32 %0
    %4 = SetAttribute a0x80, %3, u32 0x0
    %7 = DemoteToHelperInvocation
    return
"
    );
}

#[test]
fn conditions_keep_their_values() {
    assert_eq!(
        after(
            "block0:
    %0: u32 = GetCbufU32 u32 0, u32 0
    %1: u1 = INotEqual %0, u32 0
    %2: u1 = IEqual %0, u32 0
    branch %1, block1, block1
block1:
    return
",
            Pass::DeadCodeElimination
        ),
        "block0:
    %0: u32 = GetCbufU32 u32 0x0, u32 0x0
    %1: u1 = INotEqual %0, u32 0x0
    branch %1, block1, block1
block1:
    return
"
    );
    assert_eq!(
        after(
            "block0:
    %0: u1 = GetPred p0
    %1: u1 = GetPred p1
    %2: u1 = GetPred p2
    %3: u1 = GetPred p3
block1:
    %4: u32 = GetRegister r0
structure:
    block0
    if %0 {
        loop {
            block1
            break %1
        } repeat %2
    }
    return
",
            Pass::DeadCodeElimination
        ),
        "block0:
    %0: u1 = GetPred p0
    %1: u1 = GetPred p1
    %2: u1 = GetPred p2
block1:
structure:
    block0
    if %0 {
        loop {
            block1
            break %1
        } repeat %2
    }
    return
"
    );
}

#[test]
fn users_in_later_blocks_are_removed_first() {
    // Post order visits block1 before block0, so %0, used only by block1's dead %1, goes too.
    assert_eq!(
        after(
            "block0:
    %0: u32 = GetCbufU32 u32 0, u32 0
    jump block1
block1:
    %1: u32 = IAdd32 %0, u32 1
    return
",
            Pass::DeadCodeElimination
        ),
        "block0:
    jump block1
block1:
    return
"
    );
}

#[test]
fn the_walk_is_post_order_not_reverse_arena_order() {
    // Post order is block1, block2, block0; reverse arena order would visit block2 before
    // block1 and keep %0, whose only user is block1's dead %1.
    assert_eq!(
        after(
            "block0:
    jump block2
block1:
    %1: u32 = IAdd32 %0, u32 1
    return
block2:
    %0: u32 = GetCbufU32 u32 0, u32 0
    jump block1
",
            Pass::DeadCodeElimination
        ),
        "block0:
    jump block2
block1:
    return
block2:
    jump block1
"
    );
}

#[test]
fn dead_loop_values_stay_as_in_eden() {
    // The counter's phi and its increment use each other round the loop, and nothing else
    // uses them: one walk in post order (block2, block1, block0) does not remove them.
    let text = "block0:
    jump block1
block1:
    %0: u32 = Phi [block0: u32 0], [block1: %1]
    %1: u32 = IAdd32 %0, u32 1
    %2: u1 = GetPred p0
    branch %2, block1, block2
block2:
    return
";
    let out = after(text, Pass::DeadCodeElimination);
    assert!(out.contains("%0: u32 = Phi"), "{out}");
    assert!(out.contains("%1: u32 = IAdd32 %0"), "{out}");
}

#[test]
fn unreachable_users_are_removed_first() {
    // block2 is unreachable; its dead %2 is the only user of block0's %0, so %0 goes too.
    // block3 is unreachable as well and comes later in the arena: its dead %3 uses block2's
    // %1, so block3 is walked before block2.
    assert_eq!(
        after(
            "block0:
    %0: u32 = GetCbufU32 u32 0, u32 0
    jump block1
block1:
    return
block2:
    %1: u32 = GetCbufU32 u32 0, u32 4
    %2: u32 = IAdd32 %0, u32 1
    jump block1
block3:
    %3: u32 = IAdd32 %1, u32 1
    jump block1
",
            Pass::DeadCodeElimination
        ),
        "block0:
    jump block1
block1:
    return
block2:
    jump block1
block3:
    jump block1
"
    );
}

#[test]
fn a_full_pipeline_leaves_only_what_is_used() {
    let text = "block0:
    %0 = SetRegister r0, u32 3
    %1 = SetRegister r1, u32 0
    jump block1
block1:
    %2: u32 = GetRegister r0
    %3: u1 = INotEqual %2, u32 0
    branch %3, block2, block3
block2:
    %4: u32 = GetRegister r1
    %5: u32 = IAdd32 %4, %2
    %6 = SetRegister r1, %5
    %7: u32 = ISub32 %2, u32 1
    %8 = SetRegister r0, %7
    jump block1
block3:
    %9: u32 = GetRegister r1
    %10: f32 = BitCastF32U32 %9
    %11 = SetAttribute a0x80, %10, u32 0
    return
";
    insta::assert_snapshot!("dce_pipeline", after_each(text, &Pass::ORDER));
}

/// Eden's `Inst::MayHaveSideEffects` (`frontend/ir/microinstruction.cpp` @ cf654052), in its
/// order, written out.
const EDEN_SIDE_EFFECTS: [&str; 159] = [
    "ConditionRef",
    "Reference",
    "PhiMove",
    "Prologue",
    "Epilogue",
    "Join",
    "DemoteToHelperInvocation",
    "Barrier",
    "WorkgroupMemoryBarrier",
    "DeviceMemoryBarrier",
    "EmitVertex",
    "EndPrimitive",
    "SetAttribute",
    "SetAttributeIndexed",
    "SetPatch",
    "SetFragColor",
    "SetSampleMask",
    "SetFragDepth",
    "WriteGlobalU8",
    "WriteGlobalS8",
    "WriteGlobalU16",
    "WriteGlobalS16",
    "WriteGlobal32",
    "WriteGlobal64",
    "WriteGlobal128",
    "WriteStorageU8",
    "WriteStorageS8",
    "WriteStorageU16",
    "WriteStorageS16",
    "WriteStorage32",
    "WriteStorage64",
    "WriteStorage128",
    "WriteLocal",
    "WriteSharedU8",
    "WriteSharedU16",
    "WriteSharedU32",
    "WriteSharedU64",
    "WriteSharedU128",
    "SharedAtomicIAdd32",
    "SharedAtomicSMin32",
    "SharedAtomicUMin32",
    "SharedAtomicSMax32",
    "SharedAtomicUMax32",
    "SharedAtomicInc32",
    "SharedAtomicDec32",
    "SharedAtomicAnd32",
    "SharedAtomicOr32",
    "SharedAtomicXor32",
    "SharedAtomicExchange32",
    "SharedAtomicExchange64",
    "SharedAtomicExchange32x2",
    "GlobalAtomicIAdd32",
    "GlobalAtomicSMin32",
    "GlobalAtomicUMin32",
    "GlobalAtomicSMax32",
    "GlobalAtomicUMax32",
    "GlobalAtomicInc32",
    "GlobalAtomicDec32",
    "GlobalAtomicAnd32",
    "GlobalAtomicOr32",
    "GlobalAtomicXor32",
    "GlobalAtomicExchange32",
    "GlobalAtomicIAdd64",
    "GlobalAtomicSMin64",
    "GlobalAtomicUMin64",
    "GlobalAtomicSMax64",
    "GlobalAtomicUMax64",
    "GlobalAtomicAnd64",
    "GlobalAtomicOr64",
    "GlobalAtomicXor64",
    "GlobalAtomicExchange64",
    "GlobalAtomicIAdd32x2",
    "GlobalAtomicSMin32x2",
    "GlobalAtomicUMin32x2",
    "GlobalAtomicSMax32x2",
    "GlobalAtomicUMax32x2",
    "GlobalAtomicAnd32x2",
    "GlobalAtomicOr32x2",
    "GlobalAtomicXor32x2",
    "GlobalAtomicExchange32x2",
    "GlobalAtomicAddF32",
    "GlobalAtomicAddF16x2",
    "GlobalAtomicAddF32x2",
    "GlobalAtomicMinF16x2",
    "GlobalAtomicMinF32x2",
    "GlobalAtomicMaxF16x2",
    "GlobalAtomicMaxF32x2",
    "StorageAtomicIAdd32",
    "StorageAtomicSMin32",
    "StorageAtomicUMin32",
    "StorageAtomicSMax32",
    "StorageAtomicUMax32",
    "StorageAtomicInc32",
    "StorageAtomicDec32",
    "StorageAtomicAnd32",
    "StorageAtomicOr32",
    "StorageAtomicXor32",
    "StorageAtomicExchange32",
    "StorageAtomicIAdd64",
    "StorageAtomicSMin64",
    "StorageAtomicUMin64",
    "StorageAtomicSMax64",
    "StorageAtomicUMax64",
    "StorageAtomicAnd64",
    "StorageAtomicOr64",
    "StorageAtomicXor64",
    "StorageAtomicExchange64",
    "StorageAtomicIAdd32x2",
    "StorageAtomicSMin32x2",
    "StorageAtomicUMin32x2",
    "StorageAtomicSMax32x2",
    "StorageAtomicUMax32x2",
    "StorageAtomicAnd32x2",
    "StorageAtomicOr32x2",
    "StorageAtomicXor32x2",
    "StorageAtomicExchange32x2",
    "StorageAtomicAddF32",
    "StorageAtomicAddF16x2",
    "StorageAtomicAddF32x2",
    "StorageAtomicMinF16x2",
    "StorageAtomicMinF32x2",
    "StorageAtomicMaxF16x2",
    "StorageAtomicMaxF32x2",
    "BindlessImageWrite",
    "BoundImageWrite",
    "ImageWrite",
    "BindlessImageAtomicIAdd32",
    "BindlessImageAtomicSMin32",
    "BindlessImageAtomicUMin32",
    "BindlessImageAtomicSMax32",
    "BindlessImageAtomicUMax32",
    "BindlessImageAtomicInc32",
    "BindlessImageAtomicDec32",
    "BindlessImageAtomicAnd32",
    "BindlessImageAtomicOr32",
    "BindlessImageAtomicXor32",
    "BindlessImageAtomicExchange32",
    "BoundImageAtomicIAdd32",
    "BoundImageAtomicSMin32",
    "BoundImageAtomicUMin32",
    "BoundImageAtomicSMax32",
    "BoundImageAtomicUMax32",
    "BoundImageAtomicInc32",
    "BoundImageAtomicDec32",
    "BoundImageAtomicAnd32",
    "BoundImageAtomicOr32",
    "BoundImageAtomicXor32",
    "BoundImageAtomicExchange32",
    "ImageAtomicIAdd32",
    "ImageAtomicSMin32",
    "ImageAtomicUMin32",
    "ImageAtomicSMax32",
    "ImageAtomicUMax32",
    "ImageAtomicInc32",
    "ImageAtomicDec32",
    "ImageAtomicAnd32",
    "ImageAtomicOr32",
    "ImageAtomicXor32",
    "ImageAtomicExchange32",
];

#[test]
fn side_effects_match_eden() {
    let mut eden: Vec<&str> = EDEN_SIDE_EFFECTS.to_vec();
    eden.sort_unstable();
    let mut ours: Vec<&str> = Opcode::ALL
        .iter()
        .filter(|op| may_have_side_effects(**op))
        .map(|op| op.name())
        .collect();
    ours.sort_unstable();
    assert_eq!(ours, eden);
    // Neither the per-thread nor the uniform set operations are side effects.
    for op in [
        Opcode::SetRegister,
        Opcode::SetPred,
        Opcode::SetUniformRegister,
        Opcode::SetUniformPred,
    ] {
        assert!(!may_have_side_effects(op), "{op}");
    }
}
