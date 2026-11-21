//! Which passes run for which caps, their order, phis, immediates, flags, unreachable blocks,
//! the written-out conversions' sizes, the room check and the work bound.

use super::super::bits::{NARROW_OPS, PACK_HALF_OPS, UNPACK_HALF_OPS, WIDEN_OPS, narrow_f64};
use super::super::{Expansion, HostPass, MAX_INSTRUCTIONS, check_room, fits, lower_for_host, run};
use super::{count, narrowed_left, passed, ran, reg, set_pair, single};
use crate::host::HostCaps;
use crate::host_lower::HostLowerError;
use crate::interp::Machine;
use crate::ir::{
    Builder, Flags, FmzMode, FpControl, FpRounding, Function, Imm, Inst, Opcode, Operand,
    Terminator, Type,
};
use crate::opt::arena_len;

/// A one-block function with one 64-bit and one 16-bit operation (`FPAdd64` and `FPAdd16`).
fn mixed() -> Function {
    let mut function = Function::new();
    let entry = function.entry();
    let mut b = Builder::new(&mut function, entry);
    let mut op = |opcode, args: &[Operand]| Operand::Value(b.op(opcode, args).expect("builds"));
    let lo = op(Opcode::GetRegister, &[reg(0)]);
    let hi = op(Opcode::GetRegister, &[reg(1)]);
    let words = op(Opcode::CompositeConstructU32x2, &[lo, hi]);
    let x = op(Opcode::PackDouble2x32, &[words]);
    let sum = op(Opcode::FPAdd64, &[x, x]);
    let out = op(Opcode::UnpackDouble2x32, &[sum]);
    let low = op(
        Opcode::CompositeExtractU32x2,
        &[out, Operand::Imm(Imm::U32(0))],
    );
    op(Opcode::SetRegister, &[reg(10), low]);
    let packed = op(Opcode::GetRegister, &[reg(2)]);
    let halves = op(Opcode::UnpackFloat2x16, &[packed]);
    let h = op(
        Opcode::CompositeExtractF16x2,
        &[halves, Operand::Imm(Imm::U32(0))],
    );
    let hsum = op(Opcode::FPAdd16, &[h, h]);
    let pair = op(Opcode::CompositeConstructF16x2, &[hsum, hsum]);
    let word = op(Opcode::PackFloat2x16, &[pair]);
    op(Opcode::SetRegister, &[reg(11), word]);
    drop(op);
    b.terminate(Terminator::Return).expect("returns");
    function
}

fn caps_of(set: u8) -> HostCaps {
    let bit = |i: u8| set & (1 << i) != 0;
    HostCaps {
        f16: bit(0),
        f64: bit(1),
        i64: bit(2),
        subgroup: bit(3),
        subgroup_size_32: bit(4),
    }
}

/// For every combination of the target's features, each pass runs exactly when the target
/// lacks its feature (the other features change nothing), and the function keeps the 64-bit
/// operation exactly when the target has 64-bit floats and the 16-bit one exactly when it has
/// 16-bit floats.
#[test]
fn each_pass_runs_exactly_when_the_target_lacks_its_feature() {
    for set in 0_u8..32 {
        let caps = caps_of(set);
        let mut function = mixed();
        let applied = lower_for_host(&mut function, caps).expect("lowers");
        assert_eq!(applied.ran(HostPass::Fp64ToFp32), !caps.f64, "{caps:?}");
        assert_eq!(applied.ran(HostPass::Fp16ToFp32), !caps.f16, "{caps:?}");
        assert_eq!(HostPass::Fp64ToFp32.needed(caps), !caps.f64, "{caps:?}");
        assert_eq!(HostPass::Fp16ToFp32.needed(caps), !caps.f16, "{caps:?}");
        assert_eq!(
            count(&function, Opcode::FPAdd64),
            usize::from(caps.f64),
            "{caps:?}"
        );
        assert_eq!(
            count(&function, Opcode::FPAdd16),
            usize::from(caps.f16),
            "{caps:?}"
        );
        assert_eq!(
            count(&function, Opcode::FPAdd32),
            usize::from(!caps.f64) + usize::from(!caps.f16),
            "{caps:?}"
        );
    }
}

/// A target with both features gets the function back as it was, with no work done; the
/// function is not even verified then (module docs, "Checks and their order", step 1).
#[test]
fn a_target_with_both_features_is_untouched() {
    let mut function = mixed();
    let applied = lower_for_host(&mut function, HostCaps::ALL).expect("lowers");
    assert_eq!(function, mixed());
    assert_eq!(applied.work().steps(), 0);
    assert!(!applied.ran(HostPass::Fp64ToFp32) && !applied.ran(HostPass::Fp16ToFp32));
    let mut malformed = Function::new();
    assert!(malformed.verify().is_err(), "an unterminated block");
    assert!(lower_for_host(&mut malformed, HostCaps::ALL).is_ok());
    assert!(matches!(
        lower_for_host(&mut malformed, HostCaps::NONE),
        Err(HostLowerError::Verify(_))
    ));
}

/// The passes run in Eden's order, the f64 pass first, and the names and features read as the
/// errors print them.
#[test]
fn the_passes_run_in_eden_s_order() {
    assert_eq!(
        HostPass::ORDER,
        [HostPass::Fp64ToFp32, HostPass::Fp16ToFp32]
    );
    assert_eq!(
        HostPass::Fp64ToFp32.to_string(),
        "the 64-bit float lowering"
    );
    assert_eq!(
        HostPass::Fp16ToFp32.to_string(),
        "the 16-bit float lowering"
    );
    assert_eq!(HostPass::Fp64ToFp32.feature(), "64-bit floats");
    assert_eq!(HostPass::Fp16ToFp32.feature(), "16-bit floats");
}

/// A control-flow graph whose two arms each make a value of type `ty` (from registers 0 and 1
/// through `make`) that a phi of `ty` in the join block takes and `store` writes to register
/// 10; the branch is on register 4.
fn diamond(
    ty: Type,
    make: impl Fn(&mut Builder<'_>, u8) -> Operand,
    store: impl Fn(&mut Builder<'_>, Operand),
) -> Function {
    let mut function = Function::new();
    let entry = function.entry();
    let mut b = Builder::new(&mut function, entry);
    let then = b.add_block().expect("a block");
    let otherwise = b.add_block().expect("a block");
    let join = b.add_block().expect("a block");
    let word = Operand::Value(b.op(Opcode::GetRegister, &[reg(4)]).expect("builds"));
    let cond = Operand::Value(
        b.op(Opcode::INotEqual, &[word, Operand::Imm(Imm::U32(0))])
            .expect("builds"),
    );
    b.terminate(Terminator::Branch {
        cond,
        then,
        otherwise,
    })
    .expect("branches");
    let mut arms = Vec::new();
    for (block, base) in [(then, 0), (otherwise, 2)] {
        b.switch_to(block).expect("a block");
        arms.push((block, make(&mut b, base)));
        b.terminate(Terminator::Jump(join)).expect("jumps");
    }
    b.switch_to(join).expect("a block");
    let phi = Operand::Value(b.phi(ty, arms).expect("a phi"));
    store(&mut b, phi);
    b.terminate(Terminator::Return).expect("returns");
    function
}

fn value(b: &mut Builder<'_>, opcode: Opcode, args: &[Operand]) -> Operand {
    Operand::Value(b.op(opcode, args).expect("builds"))
}

/// A phi of a 64-bit float and a phi of a 16-bit float take the 32-bit type, and the function
/// gives the same result down both arms (Eden leaves such phis 64- and 16-bit).
#[test]
fn phis_take_the_narrowed_type() {
    let double = diamond(
        Type::F64,
        |b, base| {
            let lo = value(b, Opcode::GetRegister, &[reg(base)]);
            let hi = value(b, Opcode::GetRegister, &[reg(base + 1)]);
            let words = value(b, Opcode::CompositeConstructU32x2, &[lo, hi]);
            value(b, Opcode::PackDouble2x32, &[words])
        },
        |b, phi| {
            let words = value(b, Opcode::UnpackDouble2x32, &[phi]);
            let hi = value(
                b,
                Opcode::CompositeExtractU32x2,
                &[words, Operand::Imm(Imm::U32(1))],
            );
            value(b, Opcode::SetRegister, &[reg(10), hi]);
        },
    );
    let half = diamond(
        Type::F16,
        |b, base| {
            let word = value(b, Opcode::GetRegister, &[reg(base)]);
            let halves = value(b, Opcode::UnpackFloat2x16, &[word]);
            value(
                b,
                Opcode::CompositeExtractF16x2,
                &[halves, Operand::Imm(Imm::U32(1))],
            )
        },
        |b, phi| {
            let pair = value(b, Opcode::CompositeConstructF16x2, &[phi, phi]);
            let word = value(b, Opcode::PackFloat2x16, &[pair]);
            value(b, Opcode::SetRegister, &[reg(10), word]);
        },
    );
    for (function, pass, ty) in [
        (double, HostPass::Fp64ToFp32, Type::F64),
        (half, HostPass::Fp16ToFp32, Type::F16),
    ] {
        let after = passed(&function, pass);
        let phis: Vec<Type> = after
            .listed()
            .filter_map(|(_, inst)| match inst {
                Inst::Phi(phi) => Some(phi.ty),
                Inst::Op(_) => None,
            })
            .collect();
        assert_eq!(phis, [Type::F32], "{ty}");
        assert_eq!(narrowed_left(&after, pass), 0);
        for taken in [0, 1] {
            let mut machine = Machine::default();
            if ty == Type::F64 {
                set_pair(&mut machine, 0, 1.5_f64.to_bits());
                set_pair(&mut machine, 2, (-0.25_f64).to_bits());
            } else {
                machine.regs[0] = 0x3c00_4000;
                machine.regs[2] = 0xbc00_c400;
            }
            machine.regs[4] = taken;
            let (before, after) = (ran(&function, &machine), ran(&after, &machine));
            assert_eq!(before.regs[10], after.regs[10], "{ty}, arm {taken}");
            assert_ne!(before.regs[10], 0, "{ty}, arm {taken}");
        }
    }
}

/// A 64-bit float immediate is narrowed with the boundary's conversion wherever it is: an
/// operand of a renamed operation, an `Identity` of one, a phi's incoming value, and the
/// operand of a written-out `UnpackDouble2x32`; an immediate of another type is kept.
#[test]
fn f64_immediates_are_narrowed() {
    let third = (1.0_f64 / 3.0).to_bits();
    let narrowed = Operand::Imm(Imm::F32(narrow_f64(third)));
    let mut function = Function::new();
    let entry = function.entry();
    let mut b = Builder::new(&mut function, entry);
    let joined = b.add_block().expect("a block");
    let imm = Operand::Imm(Imm::F64(third));
    let lo = value(&mut b, Opcode::GetRegister, &[reg(0)]);
    let hi = value(&mut b, Opcode::GetRegister, &[reg(1)]);
    let words = value(&mut b, Opcode::CompositeConstructU32x2, &[lo, hi]);
    let x = value(&mut b, Opcode::PackDouble2x32, &[words]);
    let sum = value(&mut b, Opcode::FPAdd64, &[x, imm]);
    let same = value(&mut b, Opcode::Identity, &[imm]);
    let unpacked = value(&mut b, Opcode::UnpackDouble2x32, &[imm]);
    let kept = value(&mut b, Opcode::IAdd32, &[lo, Operand::Imm(Imm::U32(7))]);
    b.terminate(Terminator::Jump(joined)).expect("jumps");
    b.switch_to(joined).expect("a block");
    let phi = value_phi(&mut b, entry, imm);
    for (i, v) in [sum, same, phi].into_iter().enumerate() {
        let w = value(&mut b, Opcode::UnpackDouble2x32, &[v]);
        let hi = value(
            &mut b,
            Opcode::CompositeExtractU32x2,
            &[w, Operand::Imm(Imm::U32(1))],
        );
        value(
            &mut b,
            Opcode::SetRegister,
            &[reg(10 + u8::try_from(i).expect("small")), hi],
        );
    }
    let hi = value(
        &mut b,
        Opcode::CompositeExtractU32x2,
        &[unpacked, Operand::Imm(Imm::U32(1))],
    );
    value(&mut b, Opcode::SetRegister, &[reg(13), hi]);
    value(&mut b, Opcode::SetRegister, &[reg(14), kept]);
    b.terminate(Terminator::Return).expect("returns");

    let after = passed(&function, HostPass::Fp64ToFp32);
    let args = |value: Operand| match value {
        Operand::Value(v) => match after.inst(v) {
            Some(Inst::Op(op)) => op.args.clone(),
            Some(Inst::Phi(phi)) => phi.incoming.iter().map(|&(_, operand)| operand).collect(),
            None => Vec::new(),
        },
        Operand::Imm(_) => Vec::new(),
    };
    assert_eq!(args(sum).get(1), Some(&narrowed));
    assert_eq!(args(same), [narrowed]);
    assert_eq!(args(phi), [narrowed]);
    assert_eq!(args(kept).get(1), Some(&Operand::Imm(Imm::U32(7))));
    let mut machine = Machine::default();
    set_pair(&mut machine, 0, 2.0_f64.to_bits());
    let (before, after) = (ran(&function, &machine), ran(&after, &machine));
    let wide = f64::from(f32::from_bits(narrow_f64(third))).to_bits();
    for at in 11..=13 {
        assert_eq!(after.regs[at], (wide >> 32) as u32, "r{at}");
        assert_eq!(before.regs[at], (third >> 32) as u32, "r{at}");
    }
}

fn value_phi(b: &mut Builder<'_>, from: crate::ir::BlockId, imm: Operand) -> Operand {
    Operand::Value(b.phi(Type::F64, vec![(from, imm)]).expect("a phi"))
}

/// A renamed operation keeps its flags when its twin carries the same kind (`FPAdd64`'s
/// rounding and flushing; `FPAbs64`'s none), and takes the twin's defaults when the kind
/// changes (`ConvertF32F64` to `Identity`, `GlobalAtomicAddF16x2` to `GlobalAtomicAddF32x2`).
#[test]
fn flags_follow_the_renamed_opcode() {
    let control = Flags::Fp(FpControl {
        no_contraction: true,
        rounding: FpRounding::Rz,
        fmz_mode: FmzMode::Ftz,
    });
    let mut function = Function::new();
    let entry = function.entry();
    let mut b = Builder::new(&mut function, entry);
    let lo = value(&mut b, Opcode::GetRegister, &[reg(0)]);
    let words = value(&mut b, Opcode::CompositeConstructU32x2, &[lo, lo]);
    let x = value(&mut b, Opcode::PackDouble2x32, &[words]);
    let add = b
        .op_with(Opcode::FPAdd64, &[x, x], control)
        .expect("builds");
    let abs = b.op(Opcode::FPAbs64, &[x]).expect("builds");
    let narrow = b
        .op_with(Opcode::ConvertF32F64, &[x], control)
        .expect("builds");
    let address = value(&mut b, Opcode::PackUint2x32, &[words]);
    let halves = value(&mut b, Opcode::UnpackFloat2x16, &[lo]);
    let atomic = b
        .op_with(Opcode::GlobalAtomicAddF16x2, &[address, halves], control)
        .expect("builds");
    b.terminate(Terminator::Return).expect("returns");
    let after = passed(
        &passed(&function, HostPass::Fp64ToFp32),
        HostPass::Fp16ToFp32,
    );
    let op = |v| match after.inst(v) {
        Some(Inst::Op(op)) => (op.opcode, op.flags),
        other => unreachable!("{other:?}"),
    };
    assert_eq!(op(add), (Opcode::FPAdd32, control));
    assert_eq!(op(abs), (Opcode::FPAbs32, Flags::None));
    assert_eq!(op(narrow), (Opcode::Identity, Flags::None));
    assert_eq!(op(atomic), (Opcode::GlobalAtomicAddF32x2, Flags::None));
}

/// Each written-out conversion writes exactly its stated number of operations, pinned here as
/// literals: one `PackDouble2x32` adds 30 instructions to its function, and so on.
#[test]
fn written_out_conversions_have_fixed_sizes() {
    assert_eq!(
        (NARROW_OPS, WIDEN_OPS, PACK_HALF_OPS, UNPACK_HALF_OPS),
        (27, 20, 41, 25)
    );
    let cases = [
        (
            Opcode::PackDouble2x32,
            Expansion::PackDouble,
            30,
            HostPass::Fp64ToFp32,
        ),
        (
            Opcode::UnpackDouble2x32,
            Expansion::UnpackDouble,
            22,
            HostPass::Fp64ToFp32,
        ),
        (
            Opcode::BitCastF64U64,
            Expansion::DoubleFromBits,
            31,
            HostPass::Fp64ToFp32,
        ),
        (
            Opcode::BitCastU64F64,
            Expansion::BitsFromDouble,
            23,
            HostPass::Fp64ToFp32,
        ),
        (
            Opcode::PackHalf2x16,
            Expansion::PackHalves,
            88,
            HostPass::Fp16ToFp32,
        ),
        (
            Opcode::UnpackHalf2x16,
            Expansion::UnpackHalves,
            55,
            HostPass::Fp16ToFp32,
        ),
    ];
    for (opcode, expansion, size, pass) in cases {
        assert_eq!(expansion.ops(), size, "{opcode}");
        let program = single(opcode);
        let others: u64 = program
            .function
            .listed()
            .filter_map(|(_, inst)| match inst {
                Inst::Op(op) if op.opcode != opcode => match pass.rewrite(op.opcode) {
                    super::super::Rewrite::Expand(e) => Some(e.ops()),
                    _ => None,
                },
                _ => None,
            })
            .sum();
        let before = arena_len(&program.function);
        let after = passed(&program.function, pass);
        let grown = u64::try_from(arena_len(&after) - before).expect("fits");
        assert_eq!(grown, size + others, "{opcode}");
    }
}

/// Blocks the entry cannot reach are rewritten too.
#[test]
fn unreachable_blocks_are_rewritten() {
    let mut function = mixed();
    let entry = function.entry();
    let mut b = Builder::new(&mut function, entry);
    let dead = b.add_block().expect("a block");
    b.switch_to(dead).expect("a block");
    let lo = value(&mut b, Opcode::GetRegister, &[reg(0)]);
    let words = value(&mut b, Opcode::CompositeConstructU32x2, &[lo, lo]);
    let x = value(&mut b, Opcode::PackDouble2x32, &[words]);
    let neg = value(&mut b, Opcode::FPNeg64, &[x]);
    let out = value(&mut b, Opcode::UnpackDouble2x32, &[neg]);
    let w = value(
        &mut b,
        Opcode::CompositeExtractU32x2,
        &[out, Operand::Imm(Imm::U32(0))],
    );
    value(&mut b, Opcode::SetRegister, &[reg(20), w]);
    b.terminate(Terminator::Return).expect("returns");
    let after = passed(&function, HostPass::Fp64ToFp32);
    assert_eq!(narrowed_left(&after, HostPass::Fp64ToFp32), 0);
    assert_eq!(count(&after, Opcode::FPNeg32), 1);
    assert!(after.block(dead).is_some_and(|block| block.insts.len() > 7));
}

/// The room check accepts a function that fills the arena exactly and refuses one more
/// instruction, with the total it would need; an overflowing sum is refused too.
#[test]
fn room_is_checked_at_the_edge() {
    assert_eq!(MAX_INSTRUCTIONS, 1 << 32);
    assert!(fits(MAX_INSTRUCTIONS - 5, 5));
    assert!(!fits(MAX_INSTRUCTIONS - 5, 6));
    assert!(!fits(u64::MAX, 1));
    assert!(fits(0, 0));
    let len = usize::try_from(MAX_INSTRUCTIONS - 5).expect("a 64-bit host");
    assert_eq!(check_room(len, 5), Ok(()));
    let refused = check_room(len, 6).expect_err("refused");
    assert_eq!(
        refused,
        HostLowerError::Full {
            needed: MAX_INSTRUCTIONS + 1
        }
    );
    assert_eq!(
        refused.to_string(),
        "the host lowering would need 4294967297 instructions, more than the 4294967296 a \
         function holds"
    );
}

/// The work of both passes on [`mixed`] (15 listed instructions) is pinned. The f64 pass
/// alone: its scan (15 steps), its first phase (15, plus the 30 operations `PackDouble2x32`
/// writes), and its second phase over the 45 instructions then listed (plus the 22
/// `UnpackDouble2x32` writes). Both passes: the two scans (15 each); the f64 pass as above
/// (112); and the f16 pass over the 67 instructions the f64 pass leaves (67, plus 55 for
/// `UnpackFloat2x16`; then 122, plus 88 for `PackFloat2x16`).
#[test]
fn work_is_pinned() {
    let mut alone = mixed();
    let work = run(&mut alone, HostPass::Fp64ToFp32).expect("runs");
    assert_eq!(work.steps(), 15 + (15 + 30) + (45 + 22));
    let mut function = mixed();
    let applied = lower_for_host(&mut function, HostCaps::NONE).expect("lowers");
    assert_eq!(applied.work().steps(), 474);
    assert_eq!(474, 2 * 15 + 112 + (67 + 55) + (122 + 88));
}
