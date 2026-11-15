//! IR tests: the opcode table, types, the arenas, the builder, the printer's golden text, and
//! the parse–print round trip on generated IR. The verifier's and the parser's rejections are
//! in `verify::tests` and `parse::tests`.

use proptest::prelude::*;

use super::*;
use crate::decode::{Gpr, Pred};
use crate::testing::config;

/// Rows in Eden's table (`opcodes.h`: `META_TABLE[534]`).
const EDEN_ROWS: usize = 534;
/// Eden's rows this table leaves out: `Phi` and the three rescaling opcodes.
const LEFT_OUT: [&str; 4] = [
    "Phi",
    "IsTextureScaled",
    "IsImageScaled",
    "ResolutionDownFactor",
];
/// The uniform register class's opcodes, which Eden does not have.
const ADDED: [Opcode; 4] = [
    Opcode::GetUniformRegister,
    Opcode::SetUniformRegister,
    Opcode::GetUniformPred,
    Opcode::SetUniformPred,
];

fn ir(text: &str) -> Function {
    Function::parse(text).expect("parses")
}

fn u32_imm(bits: u32) -> Operand {
    Operand::Imm(Imm::U32(bits))
}

#[test]
fn opcode_table_differs_from_eden_as_listed() {
    assert_eq!(Opcode::ALL.len(), EDEN_ROWS - LEFT_OUT.len() + ADDED.len());
    for name in LEFT_OUT {
        assert_eq!(Opcode::from_name(name), None, "{name}");
    }
    let signature = |op: Opcode| (op.result(), op.args().to_vec(), op.flags_kind());
    assert_eq!(
        signature(Opcode::GetUniformRegister),
        (Type::U32, vec![Type::UniformReg], FlagsKind::None)
    );
    assert_eq!(
        signature(Opcode::SetUniformRegister),
        (
            Type::Void,
            vec![Type::UniformReg, Type::U32],
            FlagsKind::None
        )
    );
    assert_eq!(
        signature(Opcode::GetUniformPred),
        (Type::U1, vec![Type::UniformPred], FlagsKind::None)
    );
    assert_eq!(
        signature(Opcode::SetUniformPred),
        (
            Type::Void,
            vec![Type::UniformPred, Type::U1],
            FlagsKind::None
        )
    );
}

/// Spot checks of rows against Eden's `opcodes.inc`, one per shape: a `Void` argument list
/// (no arguments), five arguments, floating-point and texture flags, Eden's one opcode name
/// with an underscore, the first and last rows.
#[test]
fn opcode_rows_match_eden() {
    use Type::{Attribute as A, F32, F32x4, Opaque, U1, U32, Void};
    let row = |op: Opcode| (op.name(), op.result(), op.args().to_vec(), op.flags_kind());
    assert_eq!(Opcode::ALL.first(), Some(&Opcode::Identity));
    assert_eq!(Opcode::ALL.last(), Some(&Opcode::DPdyCoarse));
    assert_eq!(
        row(Opcode::Identity),
        ("Identity", Opaque, vec![Opaque], FlagsKind::None)
    );
    assert_eq!(
        row(Opcode::GetZFlag),
        ("GetZFlag", U1, vec![], FlagsKind::None)
    );
    assert_eq!(
        row(Opcode::GetAttribute),
        ("GetAttribute", F32, vec![A, U32], FlagsKind::None)
    );
    assert_eq!(
        row(Opcode::SetFragColor),
        ("SetFragColor", Void, vec![U32, U32, F32], FlagsKind::None)
    );
    assert_eq!(
        row(Opcode::FPFma32),
        ("FPFma32", F32, vec![F32, F32, F32], FlagsKind::Fp)
    );
    assert_eq!(
        row(Opcode::FPOrdLessThan32),
        ("FPOrdLessThan32", U1, vec![F32, F32], FlagsKind::Fp)
    );
    assert_eq!(
        row(Opcode::FPIsNan32),
        ("FPIsNan32", U1, vec![F32], FlagsKind::None)
    );
    assert_eq!(
        row(Opcode::ImageFetch),
        (
            "ImageFetch",
            F32x4,
            vec![Opaque, Opaque, Opaque, U32, Opaque],
            FlagsKind::Texture
        )
    );
    assert_eq!(
        row(Opcode::BindlessImageSampleDrefImplicitLod),
        (
            "BindlessImageSampleDrefImplicitLod",
            F32,
            vec![U32, Opaque, F32, Opaque, Opaque],
            FlagsKind::Texture
        )
    );
    assert_eq!(
        row(Opcode::SR_WScaleFactorXY),
        ("SR_WScaleFactorXY", U32, vec![], FlagsKind::None)
    );
    assert_eq!(
        row(Opcode::LaneId),
        ("LaneId", U32, vec![], FlagsKind::None)
    );
}

#[test]
fn every_name_maps_back() {
    for &op in Opcode::ALL {
        assert_eq!(Opcode::from_name(op.name()), Some(op));
        assert_eq!(op.to_string(), op.name());
    }
    for ty in Type::ALL {
        assert_eq!(Type::from_name(ty.name()), Some(ty));
    }
    assert_eq!(Opcode::from_name("fpfma32"), None);
    assert_eq!(Type::from_name("U32"), None);
}

#[test]
fn types_accept_themselves_and_opaque_slots_accept_all() {
    for slot in Type::ALL {
        for found in Type::ALL {
            assert_eq!(
                slot.accepts(found),
                slot == Type::Opaque || slot == found,
                "{slot} {found}"
            );
        }
    }
    let values: Vec<Type> = Type::ALL.into_iter().filter(|ty| ty.is_value()).collect();
    assert_eq!(values.first(), Some(&Type::U1));
    assert_eq!(values.last(), Some(&Type::F64x4));
    assert!(!Type::Opaque.is_value());
    assert!(!Type::UniformPred.is_value());
    assert!(!Type::Patch.is_value());
}

/// The name types' edges: the reserved names are the last numbers, and one past them does not
/// exist.
#[test]
fn name_edges() {
    assert_eq!(
        UniformReg::new(UniformReg::URZ.index()),
        Some(UniformReg::URZ)
    );
    assert_eq!(UniformReg::new(UniformReg::URZ.index() + 1), None);
    assert!(!UniformReg::new(UniformReg::URZ.index() - 1).is_some_and(UniformReg::is_zero));
    assert_eq!(
        UniformPred::new(UniformPred::UPT.index()),
        Some(UniformPred::UPT)
    );
    assert_eq!(UniformPred::new(UniformPred::UPT.index() + 1), None);
    assert!(!UniformPred::new(UniformPred::UPT.index() - 1).is_some_and(UniformPred::is_true));
    let last_patch = Patch::new(Patch::COUNT - 1).expect("exists");
    assert_eq!(last_patch.index(), Patch::COUNT - 1);
    assert_eq!(Patch::new(Patch::COUNT), None);
    let shown: Vec<String> = [
        Imm::UniformReg(UniformReg::URZ),
        Imm::UniformReg(UniformReg::new(5).expect("exists")),
        Imm::UniformPred(UniformPred::UPT),
        Imm::UniformPred(UniformPred::new(2).expect("exists")),
        Imm::Patch(last_patch),
        Imm::Attribute(Attribute(32)),
    ]
    .iter()
    .map(ToString::to_string)
    .collect();
    assert_eq!(shown, ["urz", "ur5", "upt", "up2", "patch125", "a0x80"]);
}

#[test]
fn arenas_hand_out_ids_in_creation_order() {
    let mut f = Function::new();
    assert_eq!(f.block_count(), 1);
    assert_eq!(f.entry().index(), 0);
    let second = f.add_block().expect("room");
    assert_eq!(second.index(), 1);
    let a = f
        .push(second, Inst::Op(Operation::new(Opcode::LaneId, vec![])))
        .expect("room");
    let b = f
        .push(f.entry(), Inst::Op(Operation::new(Opcode::Barrier, vec![])))
        .expect("room");
    assert_eq!((a.index(), b.index()), (0, 1));
    assert_eq!(
        f.push(
            BlockId(2),
            Inst::Op(Operation::new(Opcode::Barrier, vec![]))
        ),
        Err(BuildError::NoBlock(BlockId(2)))
    );
    // Listed order is block by block; arena order is creation order.
    let listed: Vec<Value> = f.listed().map(|(value, _)| value).collect();
    assert_eq!(listed, [b, a]);
    let arena: Vec<Value> = f.insts().map(|(value, _)| value).collect();
    assert_eq!(arena, [a, b]);
    let ids: Vec<BlockId> = f.blocks().map(|(id, _)| id).collect();
    assert_eq!(ids, [BlockId(0), BlockId(1)]);
    assert!(f.inst(Value(2)).is_none());
    assert!(f.block(BlockId(2)).is_none());
    if let Some(Inst::Op(op)) = f.inst_mut(a) {
        op.opcode = Opcode::InvocationId;
    }
    assert_eq!(
        f.inst(a),
        Some(&Inst::Op(Operation::new(Opcode::InvocationId, vec![])))
    );
    if let Some(block) = f.block_mut(second) {
        block.terminator = Terminator::Return;
    }
    assert_eq!(
        f.block(second).map(|b| b.terminator),
        Some(Terminator::Return)
    );
    assert_eq!(Function::default(), Function::new());
}

#[test]
fn listed_skips_ids_with_no_instruction_and_detached_slots() {
    // %1 is a detached slot the text skips; block1 lists %7, which does not exist.
    let mut f = ir("block0:\n    %0: u32 = LaneId\n    %2: u32 = LaneId\n    return\n");
    let missing = Value(7);
    f.blocks.push(Block {
        insts: vec![missing],
        terminator: Terminator::Return,
    });
    let listed: Vec<u32> = f.listed().map(|(value, _)| value.index()).collect();
    assert_eq!(listed, [0, 2]);
    let arena: Vec<u32> = f.insts().map(|(value, _)| value.index()).collect();
    assert_eq!(arena, [0, 1, 2]);
    assert_eq!(f.inst(Value(1)), Some(&Inst::detached()));
}

#[test]
fn type_of_resolves_identity_chains_and_stops_on_cycles() {
    let f = ir("block0:\n\
         \x20   %0: f32 = BitCastF32U32 u32 1\n\
         \x20   %1: opaque = Identity %0\n\
         \x20   %2: opaque = Identity %1\n\
         \x20   %3: opaque = Identity r4\n\
         \x20   %4: opaque = Identity %5\n\
         \x20   %5: opaque = Identity %4\n\
         \x20   %6 = Barrier\n\
         \x20   %7: opaque = Identity\n\
         \x20   return\n");
    let ty = |n| f.type_of(Operand::Value(Value(n)));
    assert_eq!(ty(0), Some(Type::F32));
    assert_eq!(ty(2), Some(Type::F32));
    assert_eq!(ty(3), Some(Type::Reg));
    assert_eq!(ty(4), None, "a cycle");
    assert_eq!(ty(6), Some(Type::Void));
    assert_eq!(ty(7), None, "an identity with no argument");
    assert_eq!(ty(9), None, "no instruction");
    assert_eq!(f.type_of(u32_imm(0)), Some(Type::U32));
}

#[test]
fn successors_list_targets_in_order() {
    let cond = Operand::Imm(Imm::U1(true));
    let list = |t: Terminator| t.successors().collect::<Vec<_>>();
    assert_eq!(list(Terminator::None), []);
    assert_eq!(list(Terminator::Return), []);
    assert_eq!(list(Terminator::Unreachable), []);
    assert_eq!(list(Terminator::Jump(BlockId(3))), [BlockId(3)]);
    assert_eq!(
        list(Terminator::Branch {
            cond,
            then: BlockId(2),
            otherwise: BlockId(1)
        }),
        [BlockId(2), BlockId(1)]
    );
}

#[test]
fn operation_new_takes_the_default_flags_of_its_kind() {
    assert_eq!(Operation::new(Opcode::LaneId, vec![]).flags, Flags::None);
    assert_eq!(
        Operation::new(Opcode::FPAdd32, vec![]).flags,
        Flags::Fp(FpControl::default())
    );
    assert_eq!(
        Operation::new(Opcode::ImageRead, vec![]).flags,
        Flags::Texture(TextureInfo::default())
    );
    for kind in [FlagsKind::None, FlagsKind::Fp, FlagsKind::Texture] {
        assert_eq!(Flags::default_of(kind).kind(), kind);
    }
    let operand = Operand::from(Value(3));
    assert_eq!(operand.value(), Some(Value(3)));
    assert_eq!(operand.imm(), None);
    let imm = Operand::from(Imm::U1(false));
    assert_eq!(imm.imm(), Some(Imm::U1(false)));
    assert_eq!(imm.value(), None);
}

/// Appends through `b`, expecting a failure, and checks that nothing was appended.
fn fail(
    b: &mut Builder<'_>,
    opcode: Opcode,
    args: &[Operand],
    flags: Option<Flags>,
) -> Option<BuildError> {
    let before = b.function().insts.len();
    let result = match flags {
        Some(flags) => b.op_with(opcode, args, flags),
        None => b.op(opcode, args),
    };
    assert_eq!(b.function().insts.len(), before, "nothing appended");
    result.err()
}

/// The builder appends a checked instruction, and appends nothing when a check fails. Each
/// failure is the verifier's error for the same fault.
#[test]
fn builder_checks_each_instruction() {
    let mut f = Function::new();
    let entry = f.entry();
    let mut b = Builder::new(&mut f, entry);
    assert_eq!(b.block(), entry);
    let lane = b.op(Opcode::LaneId, &[]).expect("valid");
    let barrier = b.op(Opcode::Barrier, &[]).expect("valid");
    let next = Value(2);
    let invalid = |error| Some(BuildError::Invalid(error));
    assert_eq!(
        fail(&mut b, Opcode::IAdd32, &[lane.into()], None),
        invalid(VerifyError::ArgCount {
            at: next,
            opcode: Opcode::IAdd32,
            wanted: 2,
            found: 1
        })
    );
    assert_eq!(
        fail(
            &mut b,
            Opcode::IAdd32,
            &[lane.into(), lane.into()],
            Some(Flags::Fp(FpControl::default()))
        ),
        invalid(VerifyError::FlagsKind {
            at: next,
            opcode: Opcode::IAdd32
        })
    );
    let bad_gather = TextureInfo {
        gather_component: TEXTURE_TWO_BIT_MAX + 1,
        ..TextureInfo::default()
    };
    let read_args = [u32_imm(0), u32_imm(0)];
    assert_eq!(
        fail(
            &mut b,
            Opcode::ImageRead,
            &read_args,
            Some(Flags::Texture(bad_gather))
        ),
        invalid(VerifyError::FlagsRange(next))
    );
    let bad_derivatives = TextureInfo {
        num_derivatives: TEXTURE_TWO_BIT_MAX + 1,
        ..TextureInfo::default()
    };
    assert_eq!(
        fail(
            &mut b,
            Opcode::ImageRead,
            &read_args,
            Some(Flags::Texture(bad_derivatives))
        ),
        invalid(VerifyError::FlagsRange(next))
    );
    assert_eq!(
        fail(
            &mut b,
            Opcode::IAdd32,
            &[lane.into(), Value(9).into()],
            None
        ),
        invalid(VerifyError::Undefined {
            at: Site::Inst(next),
            operand: Value(9)
        })
    );
    assert_eq!(
        fail(&mut b, Opcode::IAdd32, &[lane.into(), barrier.into()], None),
        invalid(VerifyError::NoResult {
            at: Site::Inst(next),
            operand: barrier
        })
    );
    assert_eq!(
        fail(&mut b, Opcode::FPAbs32, &[lane.into()], None),
        invalid(VerifyError::Type {
            at: Site::Inst(next),
            index: 0,
            wanted: Type::F32,
            found: Type::U32
        })
    );
    // The edges of the 2-bit fields are accepted.
    let edge = TextureInfo {
        gather_component: TEXTURE_TWO_BIT_MAX,
        num_derivatives: TEXTURE_TWO_BIT_MAX,
        ..TextureInfo::default()
    };
    b.op_with(Opcode::ImageRead, &read_args, Flags::Texture(edge))
        .expect("in range");
}

/// A phi's type and incoming values, and a terminator's targets and condition, are checked as
/// they are added.
#[test]
fn builder_checks_phis_and_terminators() {
    let mut f = Function::new();
    let entry = f.entry();
    let mut b = Builder::new(&mut f, entry);
    let lane = b.op(Opcode::LaneId, &[]).expect("valid");
    b.op(Opcode::Barrier, &[]).expect("valid");
    b.op(Opcode::Barrier, &[]).expect("valid");
    assert_eq!(
        b.phi(Type::Reg, vec![]),
        Err(BuildError::Invalid(VerifyError::PhiType {
            at: Value(3),
            ty: Type::Reg
        }))
    );
    assert_eq!(
        b.phi(Type::U32, vec![(BlockId(5), u32_imm(0))]),
        Err(BuildError::Invalid(VerifyError::NoSuchBlock {
            at: Site::Inst(Value(3)),
            block: BlockId(5)
        }))
    );
    assert_eq!(
        b.phi(Type::F32, vec![(entry, u32_imm(0))]),
        Err(BuildError::Invalid(VerifyError::Type {
            at: Site::Inst(Value(3)),
            index: 0,
            wanted: Type::F32,
            found: Type::U32
        }))
    );
    let phi = b.phi(Type::U32, vec![(entry, lane.into())]).expect("valid");
    assert_eq!(phi, Value(3));
    // Terminators: targets must exist and a branch condition must be a `u1`.
    assert_eq!(
        b.terminate(Terminator::Jump(BlockId(1))),
        Err(BuildError::Invalid(VerifyError::NoSuchBlock {
            at: Site::Terminator(entry),
            block: BlockId(1)
        }))
    );
    let other = b.add_block().expect("room");
    assert_eq!(
        b.terminate(Terminator::Branch {
            cond: lane.into(),
            then: other,
            otherwise: other
        }),
        Err(BuildError::Invalid(VerifyError::Type {
            at: Site::Terminator(entry),
            index: 0,
            wanted: Type::U1,
            found: Type::U32
        }))
    );
    b.terminate(Terminator::Jump(other)).expect("valid");
    assert_eq!(
        b.switch_to(BlockId(9)),
        Err(BuildError::NoBlock(BlockId(9)))
    );
    assert_eq!(b.block(), entry, "a failed switch leaves the block");
    b.switch_to(other).expect("exists");
    b.terminate(Terminator::Return).expect("valid");
    // The builder does not check where a phi goes; the verifier does.
    assert_eq!(f.verify(), Err(VerifyError::PhiAfterOperation(phi)));
}

#[test]
fn builder_refuses_a_block_that_does_not_exist() {
    let mut f = Function::new();
    let mut b = Builder::new(&mut f, BlockId(4));
    assert_eq!(
        b.op(Opcode::Barrier, &[]),
        Err(BuildError::NoBlock(BlockId(4)))
    );
    assert_eq!(
        b.phi(Type::U1, vec![]),
        Err(BuildError::NoBlock(BlockId(4)))
    );
    assert_eq!(
        b.terminate(Terminator::Return),
        Err(BuildError::NoBlock(BlockId(4)))
    );
    assert_eq!(f.insts, Vec::<Inst>::new());
}

/// Each get and set opcode refuses its class's reserved name and accepts the name just below it,
/// in the builder and the verifier alike.
#[test]
fn reserved_names_are_refused() {
    let uniform = |index| Imm::UniformReg(UniformReg::new(index).expect("exists"));
    let upred = |index| Imm::UniformPred(UniformPred::new(index).expect("exists"));
    let pred = |index| Imm::Pred(Pred::new(index).expect("exists"));
    let cases = [
        (
            Opcode::GetRegister,
            Opcode::SetRegister,
            Imm::Reg(Gpr::RZ),
            Imm::Reg(Gpr::new(Gpr::RZ.index() - 1)),
            u32_imm(0),
        ),
        (
            Opcode::GetPred,
            Opcode::SetPred,
            Imm::Pred(Pred::PT),
            pred(Pred::PT.index() - 1),
            Operand::Imm(Imm::U1(true)),
        ),
        (
            Opcode::GetUniformRegister,
            Opcode::SetUniformRegister,
            Imm::UniformReg(UniformReg::URZ),
            uniform(UniformReg::URZ.index() - 1),
            u32_imm(0),
        ),
        (
            Opcode::GetUniformPred,
            Opcode::SetUniformPred,
            Imm::UniformPred(UniformPred::UPT),
            upred(UniformPred::UPT.index() - 1),
            Operand::Imm(Imm::U1(true)),
        ),
    ];
    for (get, set, reserved, below, data) in cases {
        for (opcode, args) in [
            (get, vec![Operand::Imm(reserved)]),
            (set, vec![Operand::Imm(reserved), data]),
        ] {
            let error = VerifyError::ReservedName {
                at: Value(0),
                opcode,
                name: reserved,
            };
            let mut f = Function::new();
            let entry = f.entry();
            assert_eq!(
                Builder::new(&mut f, entry).op(opcode, &args),
                Err(BuildError::Invalid(error))
            );
            f.push(entry, Inst::Op(Operation::new(opcode, args.clone())))
                .expect("room");
            f.blocks[0].terminator = Terminator::Return;
            assert_eq!(f.verify(), Err(error));
            let mut below_args = args;
            below_args[0] = Operand::Imm(below);
            let mut f = Function::new();
            let mut b = Builder::new(&mut f, entry);
            b.op(opcode, &below_args).expect("not reserved");
            b.terminate(Terminator::Return).expect("valid");
            assert_eq!(f.verify(), Ok(()));
        }
    }
    // Other opcodes may name the reserved names (a name is only reserved for access).
    let identity = ir("block0:\n    %0: opaque = Identity rz\n    return\n");
    assert_eq!(identity.verify(), Ok(()));
}

/// A function that uses every feature of the text form: every immediate kind, every flag word,
/// phis, every terminator; and a structured function with every node.
const GOLDEN_CFG: &str = "\
block0:
    %0: u32 = GetRegister r7
    %1 = SetRegister r254, %0
    %2: u1 = GetPred p3
    %3 = SetPred p6, u1 false
    %4: u32 = GetUniformRegister ur9
    %5 = SetUniformRegister ur62, u32 0x2a
    %6: u1 = GetUniformPred up1
    %7 = SetUniformPred up6, u1 true
    %8: f32 = GetAttribute a0x7c, u32 0x0
    %9: f32 = GetPatch patch125
    %10: u32 = ConvertU32U8 u8 0xff
    %11: u32 = ConvertS32S16 u16 0x8000
    %12: u64 = IAdd64 u64 0xfedcba9876543210, u64 0x1
    %13: f64 = FPAdd64 f64 0x3ff0000000000000, f64 0x8000000000000000 {no_contraction, rm, fmz}
    %14: f32 = FPMul32 %8, f32 0x3f000000 {rp, fmz_none}
    %15: f32 = FPFma32 %8, %9, %14 {rn, ftz}
    %16: f32 = FPAdd32 %15, %15 {rz}
    %17: f32 = FPAdd32 %16, %16
    %18: f32x4 = ImageSampleImplicitLod u32 0x1, %8, %8, %8 {descriptor=7, type=array_cube, depth, bias, lod_clamp, relaxed_precision, gather=3, derivatives=2, format=r32g32b32a32_uint, ndv, integer}
    %19: f32x4 = ImageSampleImplicitLod u32 0x1, %8, %8, %8 {type=2d_rect}
    %20: opaque = Identity %17
    branch %2, block1, block2
block1:
    jump block3
block2:
    %21: u1 = GetZFlag
    jump block3
block3:
    %22: f32 = Phi [block1: %20], [block2: f32 0x0]
    return
block4:
    %23: u32 = Phi
    unreachable
";

/// The structured counterpart: every node, nested.
const GOLDEN_STRUCTURED: &str = "\
block0:
    %0: u1 = GetPred p0
block1:
    %1: u1 = GetPred p1
block2:
    %2: u1 = GetPred p2
block3:
structure:
    block0
    if %0 {
        loop {
            block1
            break %1
            block2
        } repeat %2
        unreachable
    }
    block3
    return
";

/// Block 0 of [`GOLDEN_CFG`], first part: every name kind. Returns `%2`, `%8` and `%9`.
fn golden_names(b: &mut Builder<'_>) -> (Value, Value, Value) {
    let reg = |index| Operand::Imm(Imm::Reg(Gpr::new(index)));
    let pred = |index| Operand::Imm(Imm::Pred(Pred::new(index).expect("exists")));
    let ureg = |index| Operand::Imm(Imm::UniformReg(UniformReg::new(index).expect("exists")));
    let upred = |index| Operand::Imm(Imm::UniformPred(UniformPred::new(index).expect("exists")));
    let r7 = b.op(Opcode::GetRegister, &[reg(7)]).expect("ok");
    b.op(Opcode::SetRegister, &[reg(254), r7.into()])
        .expect("ok");
    let p3 = b.op(Opcode::GetPred, &[pred(3)]).expect("ok");
    b.op(Opcode::SetPred, &[pred(6), Operand::Imm(Imm::U1(false))])
        .expect("ok");
    b.op(Opcode::GetUniformRegister, &[ureg(9)]).expect("ok");
    b.op(Opcode::SetUniformRegister, &[ureg(62), u32_imm(42)])
        .expect("ok");
    b.op(Opcode::GetUniformPred, &[upred(1)]).expect("ok");
    b.op(
        Opcode::SetUniformPred,
        &[upred(6), Operand::Imm(Imm::U1(true))],
    )
    .expect("ok");
    let attr = b
        .op(
            Opcode::GetAttribute,
            &[Operand::Imm(Imm::Attribute(Attribute(31))), u32_imm(0)],
        )
        .expect("ok");
    let patch = b
        .op(
            Opcode::GetPatch,
            &[Operand::Imm(Imm::Patch(
                Patch::new(Patch::COUNT - 1).expect("exists"),
            ))],
        )
        .expect("ok");
    (p3, attr, patch)
}

/// Block 0 of [`GOLDEN_CFG`], second part: every number kind and flag word. Returns `%20`.
fn golden_numbers(b: &mut Builder<'_>, attr: Value, patch: Value) -> Value {
    let fp = |no_contraction, rounding, fmz_mode| {
        Flags::Fp(FpControl {
            no_contraction,
            rounding,
            fmz_mode,
        })
    };
    b.op(Opcode::ConvertU32U8, &[Operand::Imm(Imm::U8(0xff))])
        .expect("ok");
    b.op(Opcode::ConvertS32S16, &[Operand::Imm(Imm::U16(0x8000))])
        .expect("ok");
    b.op(
        Opcode::IAdd64,
        &[
            Operand::Imm(Imm::U64(0xfedc_ba98_7654_3210)),
            Operand::Imm(Imm::U64(1)),
        ],
    )
    .expect("ok");
    b.op_with(
        Opcode::FPAdd64,
        &[
            Operand::Imm(Imm::F64(1.0_f64.to_bits())),
            Operand::Imm(Imm::F64((-0.0_f64).to_bits())),
        ],
        fp(true, FpRounding::Rm, FmzMode::Fmz),
    )
    .expect("ok");
    let mul = b
        .op_with(
            Opcode::FPMul32,
            &[attr.into(), Operand::Imm(Imm::F32(0.5_f32.to_bits()))],
            fp(false, FpRounding::Rp, FmzMode::None),
        )
        .expect("ok");
    let fma = b
        .op_with(
            Opcode::FPFma32,
            &[attr.into(), patch.into(), mul.into()],
            fp(false, FpRounding::Rn, FmzMode::Ftz),
        )
        .expect("ok");
    let add = b
        .op_with(
            Opcode::FPAdd32,
            &[fma.into(), fma.into()],
            fp(false, FpRounding::Rz, FmzMode::DontCare),
        )
        .expect("ok");
    let add2 = b
        .op(Opcode::FPAdd32, &[add.into(), add.into()])
        .expect("ok");
    let every = TextureInfo {
        descriptor_index: 7,
        ty: TextureType::ColorArrayCube,
        is_depth: true,
        has_bias: true,
        has_lod_clamp: true,
        relaxed_precision: true,
        gather_component: TEXTURE_TWO_BIT_MAX,
        num_derivatives: 2,
        image_format: ImageFormat::R32G32B32A32Uint,
        ndv_is_active: true,
        is_integer: true,
    };
    let sample = [u32_imm(1), attr.into(), attr.into(), attr.into()];
    b.op_with(
        Opcode::ImageSampleImplicitLod,
        &sample,
        Flags::Texture(every),
    )
    .expect("ok");
    let rect = TextureInfo {
        ty: TextureType::Color2DRect,
        ..TextureInfo::default()
    };
    b.op_with(
        Opcode::ImageSampleImplicitLod,
        &sample,
        Flags::Texture(rect),
    )
    .expect("ok");
    b.op(Opcode::Identity, &[add2.into()]).expect("ok")
}

#[test]
fn golden_text_prints_what_was_built() {
    let mut f = Function::new();
    let entry = f.entry();
    let mut b = Builder::new(&mut f, entry);
    let one = b.add_block().expect("room");
    let two = b.add_block().expect("room");
    let three = b.add_block().expect("room");
    let four = b.add_block().expect("room");
    let (p3, attr, patch) = golden_names(&mut b);
    let identity = golden_numbers(&mut b, attr, patch);
    b.terminate(Terminator::Branch {
        cond: p3.into(),
        then: one,
        otherwise: two,
    })
    .expect("ok");
    b.switch_to(one).expect("exists");
    b.terminate(Terminator::Jump(three)).expect("ok");
    b.switch_to(two).expect("exists");
    b.op(Opcode::GetZFlag, &[]).expect("ok");
    b.terminate(Terminator::Jump(three)).expect("ok");
    b.switch_to(three).expect("exists");
    b.phi(
        Type::F32,
        vec![(one, identity.into()), (two, Operand::Imm(Imm::F32(0)))],
    )
    .expect("ok");
    b.terminate(Terminator::Return).expect("ok");
    // No block reaches block4, so its phi has no incoming values.
    b.switch_to(four).expect("exists");
    b.phi(Type::U32, vec![]).expect("ok");
    b.terminate(Terminator::Unreachable).expect("ok");
    assert_eq!(f.to_string(), GOLDEN_CFG);
    assert_eq!(f.verify(), Ok(()));
    assert_eq!(Function::parse(GOLDEN_CFG), Ok(f));
}

#[test]
fn golden_structured_text() {
    let entry = BlockId(0);
    let pred = |index| Operand::Imm(Imm::Pred(Pred::new(index).expect("exists")));
    let mut s = Function::new();
    let mut b = Builder::new(&mut s, entry);
    let mut conds = Vec::new();
    for index in 0..3 {
        conds.push(b.op(Opcode::GetPred, &[pred(index)]).expect("ok"));
        let next = b.add_block().expect("room");
        b.switch_to(next).expect("exists");
    }
    s.set_structure(Some(vec![
        Node::Block(BlockId(0)),
        Node::If {
            cond: conds[0].into(),
            body: vec![
                Node::Loop {
                    body: vec![
                        Node::Block(BlockId(1)),
                        Node::Break {
                            cond: conds[1].into(),
                        },
                        Node::Block(BlockId(2)),
                    ],
                    repeat: conds[2].into(),
                },
                Node::Unreachable,
            ],
        },
        Node::Block(BlockId(3)),
        Node::Return,
    ]));
    assert_eq!(s.to_string(), GOLDEN_STRUCTURED);
    assert_eq!(s.verify(), Ok(()));
    assert_eq!(Function::parse(GOLDEN_STRUCTURED), Ok(s.clone()));
    s.set_structure(None);
    assert_eq!(s.structure(), None);
    assert_eq!(s.verify(), Err(VerifyError::Unterminated(BlockId(0))));
}

/// The printer shows an id a block lists but the arena lacks, rather than skipping it.
#[test]
fn printer_marks_missing_instructions() {
    let mut f = Function::new();
    f.blocks[0].insts.push(Value(3));
    assert_eq!(f.to_string(), "block0:\n    %3 = <missing>\n");
}

/// A tiny deterministic generator (xorshift64*), seeded by proptest, so one `u64` builds a whole
/// function.
struct Gen(u64);

impl Gen {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    fn below(&mut self, bound: usize) -> usize {
        usize::try_from(self.next() % u64::try_from(bound).expect("fits")).expect("fits")
    }

    fn bool(&mut self) -> bool {
        self.next() & 1 == 1
    }

    fn byte(&mut self) -> u8 {
        self.next().to_le_bytes()[0]
    }

    fn value(&mut self, count: usize) -> Value {
        Value(u32::try_from(self.below(count.max(1) + 4)).expect("small"))
    }

    fn block(&mut self, count: usize) -> BlockId {
        BlockId(u32::try_from(self.below(count + 1)).expect("small"))
    }

    /// An operand for a slot of type `ty`: a name for a name type, else a value or an
    /// immediate of the type (or of any kind for an opaque slot). Types are not matched to
    /// values: the round trip is about syntax, and every field gets an independent random value.
    fn operand(&mut self, ty: Type, values: usize) -> Operand {
        let bits = self.next();
        let imm = match ty {
            Type::Reg => Imm::Reg(Gpr::new(self.byte())),
            Type::Pred => Imm::Pred(Pred::new(self.byte() % 8).expect("in range")),
            Type::UniformReg => Imm::UniformReg(
                UniformReg::new(self.byte() % (UniformReg::URZ.index() + 1)).expect("in range"),
            ),
            Type::UniformPred => Imm::UniformPred(
                UniformPred::new(self.byte() % (UniformPred::UPT.index() + 1)).expect("in range"),
            ),
            Type::Attribute => Imm::Attribute(Attribute(bits.to_le_bytes()[1].into())),
            Type::Patch => Imm::Patch(Patch::new(self.byte() % Patch::COUNT).expect("in range")),
            _ if self.bool() => return Operand::Value(self.value(values)),
            Type::U1 => Imm::U1(self.bool()),
            Type::U8 => Imm::U8(self.byte()),
            Type::U16 => Imm::U16(u16::from_le_bytes([self.byte(), self.byte()])),
            Type::U32 => Imm::U32(u32::try_from(bits >> 32).expect("fits")),
            Type::U64 => Imm::U64(bits),
            Type::F32 => Imm::F32(u32::try_from(bits >> 32).expect("fits")),
            Type::F64 => Imm::F64(bits),
            Type::Opaque => {
                let kinds = [Type::Reg, Type::U32, Type::F64, Type::Attribute, Type::U1];
                let kind = kinds[self.below(kinds.len())];
                return self.operand(kind, values);
            }
            _ => return Operand::Value(self.value(values)),
        };
        Operand::Imm(imm)
    }

    fn flags(&mut self, kind: FlagsKind) -> Flags {
        match kind {
            FlagsKind::None => Flags::None,
            FlagsKind::Fp => Flags::Fp(FpControl {
                no_contraction: self.bool(),
                rounding: [
                    FpRounding::DontCare,
                    FpRounding::Rn,
                    FpRounding::Rm,
                    FpRounding::Rp,
                    FpRounding::Rz,
                ][self.below(5)],
                fmz_mode: [FmzMode::DontCare, FmzMode::Ftz, FmzMode::Fmz, FmzMode::None]
                    [self.below(4)],
            }),
            FlagsKind::Texture => Flags::Texture(TextureInfo {
                descriptor_index: u16::from_le_bytes([self.byte(), self.byte()]),
                ty: TextureType::ALL[self.below(TextureType::ALL.len())],
                is_depth: self.bool(),
                has_bias: self.bool(),
                has_lod_clamp: self.bool(),
                relaxed_precision: self.bool(),
                gather_component: self.byte() % (TEXTURE_TWO_BIT_MAX + 1),
                num_derivatives: self.byte() % (TEXTURE_TWO_BIT_MAX + 1),
                image_format: ImageFormat::ALL[self.below(ImageFormat::ALL.len())],
                ndv_is_active: self.bool(),
                is_integer: self.bool(),
            }),
        }
    }

    fn inst(&mut self, values: usize, blocks: usize) -> Inst {
        if self.below(8) == 0 {
            let values_types: Vec<Type> = Type::ALL.into_iter().filter(|t| t.is_value()).collect();
            let ty = values_types[self.below(values_types.len())];
            let incoming = (0..self.below(4))
                .map(|_| (self.block(blocks), self.operand(ty, values)))
                .collect();
            return Inst::Phi(Phi { ty, incoming });
        }
        let opcode = Opcode::ALL[self.below(Opcode::ALL.len())];
        let args = opcode
            .args()
            .iter()
            .map(|&ty| self.operand(ty, values))
            .collect();
        let flags = self.flags(opcode.flags_kind());
        Inst::Op(Operation {
            opcode,
            args,
            flags,
        })
    }

    fn terminator(&mut self, blocks: usize, values: usize) -> Terminator {
        match self.below(5) {
            0 => Terminator::None,
            1 => Terminator::Jump(self.block(blocks)),
            2 => Terminator::Branch {
                cond: self.operand(Type::U1, values),
                then: self.block(blocks),
                otherwise: self.block(blocks),
            },
            3 => Terminator::Return,
            _ => Terminator::Unreachable,
        }
    }

    fn nodes(&mut self, depth: usize, blocks: usize, values: usize) -> Vec<Node> {
        (0..self.below(4))
            .map(|_| match self.below(if depth < 3 { 6 } else { 4 }) {
                0 => Node::Block(self.block(blocks)),
                1 => Node::Break {
                    cond: self.operand(Type::U1, values),
                },
                2 => Node::Return,
                3 => Node::Unreachable,
                4 => Node::If {
                    cond: self.operand(Type::U1, values),
                    body: self.nodes(depth + 1, blocks, values),
                },
                _ => Node::Loop {
                    body: self.nodes(depth + 1, blocks, values),
                    repeat: self.operand(Type::U1, values),
                },
            })
            .collect()
    }

    /// A function of up to four blocks, a control-flow graph or structured, with detached slots
    /// before some instructions (never at the end of the arena, which the text cannot show).
    fn function(&mut self) -> Function {
        let block_count = 1 + self.below(4);
        let mut f = Function {
            insts: Vec::new(),
            blocks: vec![Block::default(); block_count],
            structure: None,
        };
        let structured = self.bool();
        for index in 0..block_count {
            for _ in 0..self.below(6) {
                if self.below(8) == 0 {
                    f.insts.push(Inst::detached());
                }
                let inst = self.inst(f.insts.len(), block_count);
                let id = Value(u32::try_from(f.insts.len()).expect("small"));
                f.insts.push(inst);
                f.blocks[index].insts.push(id);
            }
            if !structured {
                f.blocks[index].terminator = self.terminator(block_count, f.insts.len());
            }
        }
        if structured {
            let nodes = self.nodes(0, block_count, f.insts.len());
            f.structure = Some(nodes);
        }
        f
    }
}

proptest! {
    #![proptest_config(config())]

    /// Printing then parsing gives the function back, and printing that gives the same text.
    /// Every field of every instruction, terminator and node holds an independent random value,
    /// so two fields read in swapped order fail the comparison.
    #[test]
    fn parse_print_round_trips(seed in any::<u64>()) {
        let f = Gen(seed | 1).function();
        let text = f.to_string();
        let parsed = Function::parse(&text).map_err(|error| {
            TestCaseError::fail(format!("{error}\n{text}"))
        })?;
        prop_assert_eq!(&parsed, &f, "{}", text);
        prop_assert_eq!(parsed.to_string(), text);
    }
}

/// The skeleton's translated programs survive the round trip and verify.
#[test]
fn translated_programs_round_trip() {
    let f = ir("block0:\n\
         \x20   %0: f32 = GetAttribute a0x80, u32 0x0\n\
         \x20   %1: u32 = BitCastU32F32 %0\n\
         \x20   %2 = SetRegister r0, %1\n\
         \x20   %3: f32 = FPFma32 %0, %0, %0 {no_contraction, rz, ftz}\n\
         \x20   %4 = SetFragColor u32 0x0, u32 0x3, %3\n\
         \x20   return\n");
    assert_eq!(f.verify(), Ok(()));
    assert_eq!(Function::parse(&f.to_string()), Ok(f));
}

/// Every verifier error about blocks and structure reads as plain words.
#[test]
fn structure_errors_read_as_words() {
    let at = Value(4);
    let block = BlockId(2);
    let cases: Vec<(VerifyError, &str)> = vec![
        (VerifyError::NoBlocks, "the function has no blocks"),
        (
            VerifyError::Missing(at),
            "%4 is listed in a block but does not exist",
        ),
        (VerifyError::Repeated(at), "%4 is listed twice"),
        (VerifyError::Unterminated(block), "block2 has no terminator"),
        (
            VerifyError::NoSuchBlock {
                at: Site::Terminator(block),
                block: BlockId(9),
            },
            "the terminator of block2 names block9, which does not exist",
        ),
        (
            VerifyError::NoSuchBlock {
                at: Site::Structure,
                block: BlockId(9),
            },
            "the structure names block9, which does not exist",
        ),
        (
            VerifyError::TerminatorInStructure(block),
            "block2 has a terminator in a structured function",
        ),
        (
            VerifyError::NotPlaced(block),
            "block2 is not placed in the structure",
        ),
        (
            VerifyError::PlacedTwice(block),
            "block2 is placed in the structure twice",
        ),
        (
            VerifyError::BreakOutsideLoop,
            "the structure has a break outside every loop",
        ),
    ];
    for (error, text) in cases {
        assert_eq!(error.to_string(), text);
    }
}

/// Every verifier error about instructions, and every builder error, reads as plain words.
#[test]
fn instruction_errors_read_as_words() {
    let at = Value(4);
    let block = BlockId(2);
    let cases: Vec<(VerifyError, &str)> = vec![
        (
            VerifyError::Undefined {
                at: Site::Inst(at),
                operand: Value(7),
            },
            "%4 uses %7 before it is defined",
        ),
        (
            VerifyError::NoResult {
                at: Site::Inst(at),
                operand: Value(1),
            },
            "%4 uses %1, which has no result",
        ),
        (
            VerifyError::ArgCount {
                at,
                opcode: Opcode::IAdd32,
                wanted: 2,
                found: 3,
            },
            "%4: IAdd32 takes 2 arguments, not 3",
        ),
        (
            VerifyError::FlagsKind {
                at,
                opcode: Opcode::IAdd32,
            },
            "%4: IAdd32 does not carry these flags",
        ),
        (
            VerifyError::FlagsRange(at),
            "%4: a texture field is out of range",
        ),
        (
            VerifyError::Type {
                at: Site::Inst(at),
                index: 1,
                wanted: Type::F32,
                found: Type::U32,
            },
            "%4 needs argument 1 to be f32, not u32",
        ),
        (
            VerifyError::ReservedName {
                at,
                opcode: Opcode::SetUniformRegister,
                name: Imm::UniformReg(UniformReg::URZ),
            },
            "%4: SetUniformRegister cannot access urz",
        ),
        (
            VerifyError::PhiType { at, ty: Type::Pred },
            "%4 is a phi of type pred, which no value has",
        ),
        (
            VerifyError::PhiAfterOperation(at),
            "%4 is a phi after an operation in its block",
        ),
        (
            VerifyError::PhiRepeated { at, block },
            "%4 has two incoming values from block2",
        ),
        (
            VerifyError::PhiNotPredecessor { at, block },
            "%4 has an incoming value from block2, which is not a predecessor",
        ),
        (
            VerifyError::PhiMissingPredecessor { at, block },
            "%4 has no incoming value from predecessor block2",
        ),
    ];
    for (error, text) in cases {
        assert_eq!(error.to_string(), text);
    }
    assert_eq!(
        VerifyError::TooDeep.to_string(),
        format!("the structure nests deeper than {MAX_NESTING} levels")
    );
}

/// Every builder error reads as plain words.
#[test]
fn build_errors_read_as_words() {
    let at = Value(4);
    let block = BlockId(2);
    let builds = [
        (
            BuildError::Full,
            format!(
                "the function has more than {} instructions or blocks",
                u32::MAX
            ),
        ),
        (
            BuildError::NoBlock(block),
            "block2 does not exist".to_owned(),
        ),
        (
            BuildError::Invalid(VerifyError::Repeated(at)),
            "%4 is listed twice".to_owned(),
        ),
    ];
    for (error, text) in builds {
        assert_eq!(error.to_string(), text);
    }
}
