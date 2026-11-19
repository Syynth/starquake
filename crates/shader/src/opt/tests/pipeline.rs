//! The order, the per-pass verification and the errors' words.

use super::{parse, value};
use crate::ir::{BuildError, Site, Type, VerifyError};
use crate::opt::{Pass, PassError, Work, optimize, run};

#[test]
fn the_order_is_edens_with_identity_removal_after_the_rewrite() {
    assert_eq!(
        Pass::ORDER,
        [
            Pass::SsaRewrite,
            Pass::IdentityRemoval,
            Pass::ConstantPropagation,
            Pass::DeadCodeElimination,
        ]
    );
    let names: Vec<String> = Pass::ORDER.iter().map(ToString::to_string).collect();
    assert_eq!(
        names,
        [
            "the SSA rewrite",
            "identity removal",
            "constant propagation",
            "dead code elimination"
        ]
    );
}

/// A loop that sums a constant buffer word's worth of numbers into an output.
const SUM: &str = "block0:
    %0: u32 = GetCbufU32 u32 0, u32 0
    %1 = SetRegister r1, %0
    %2 = SetRegister r0, u32 0
    jump block1
block1:
    %3: u32 = GetRegister r0
    %4: u32 = GetRegister r1
    %5: u32 = IAdd32 %3, %4
    %6 = SetRegister r0, %5
    %7: u32 = ISub32 %4, u32 1
    %8 = SetRegister r1, %7
    %9: u32 = IAdd32 u32 2, u32 3
    %10 = SetRegister r5, %9
    %11: u1 = INotEqual %7, u32 0
    branch %11, block1, block2
block2:
    %12: u32 = GetRegister r0
    %13: u32 = GetRegister r5
    %14: u32 = IAdd32 %12, %13
    %15: f32 = BitCastF32U32 %14
    %16 = SetAttribute a0x80, %15, u32 0
    return
";

#[test]
fn optimize_runs_each_pass_in_order() {
    let mut by_hand = parse(SUM);
    let mut work = 0;
    for pass in Pass::ORDER {
        work += run(&mut by_hand, pass).expect("runs").steps();
    }
    let mut together = parse(SUM);
    let total = optimize(&mut together).expect("optimizes");
    assert_eq!(together, by_hand);
    assert_eq!(total.steps(), work);
    insta::assert_snapshot!("pipeline_sum", together.to_string());
}

#[test]
fn optimize_keeps_the_interpreters_result() {
    let mut machine = crate::interp::Machine::new();
    machine.cbufs.insert(0, 4_u32.to_le_bytes().to_vec());
    let before = super::ran(&parse(SUM), &machine);
    let mut function = parse(SUM);
    optimize(&mut function).expect("optimizes");
    let after = super::ran(&function, &machine);
    assert_eq!(after.outputs, before.outputs);
    assert_eq!(
        before.outputs.values().copied().collect::<Vec<_>>(),
        [4 + 3 + 2 + 1 + 5]
    );
}

#[test]
fn a_pass_whose_output_does_not_verify_is_an_error() {
    // Dead code elimination keeps the store, whose value has the wrong type.
    let text = "block0:
    %0 = SetAttribute a0x80, u32 1, u32 0
    return
";
    let mut function = parse(text);
    let error = run(&mut function, Pass::DeadCodeElimination).expect_err("does not verify");
    assert_eq!(
        error,
        PassError::Verify {
            pass: Pass::DeadCodeElimination,
            error: VerifyError::Type {
                at: Site::Inst(value(0)),
                index: 1,
                wanted: Type::F32,
                found: Type::U32,
            },
        }
    );
    assert_eq!(
        error.to_string(),
        "after dead code elimination: %0 needs argument 1 to be f32, not u32"
    );
    // optimize stops at the first pass that fails, which here is the SSA rewrite.
    let mut function = parse(text);
    assert_eq!(
        optimize(&mut function).map(Work::steps),
        Err(PassError::Verify {
            pass: Pass::SsaRewrite,
            error: VerifyError::Type {
                at: Site::Inst(value(0)),
                index: 1,
                wanted: Type::F32,
                found: Type::U32,
            },
        })
    );
}

#[test]
fn errors_read_as_words() {
    assert_eq!(
        PassError::Missing(value(3)).to_string(),
        "%3 is listed in a block but does not exist"
    );
    assert_eq!(
        PassError::Build(BuildError::Full).to_string(),
        "the function has more than 4294967295 instructions or blocks"
    );
}

#[test]
fn every_pass_refuses_a_listed_id_with_no_instruction() {
    for pass in Pass::ORDER {
        let mut function = parse("block0:\n    %0 = Void\n    return\n");
        if let Some(block) = function.block_mut(function.entry()) {
            block.insts.push(value(7));
        }
        assert_eq!(
            run(&mut function, pass),
            Err(PassError::Missing(value(7))),
            "{pass}"
        );
    }
}
