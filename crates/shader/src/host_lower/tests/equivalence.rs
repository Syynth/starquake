//! Interpreter equivalence before and after each pass: one case per mapped opcode with fixed
//! inputs, and generated programs of several statements with a fixed seed, compared within the
//! module's stated precision ([`F64_RELATIVE_TOLERANCE`], [`F16_FMA_TOLERANCE_ULPS`]).

use proptest::collection::vec;
use proptest::prelude::*;

use super::super::{F16_FMA_TOLERANCE_ULPS, F64_RELATIVE_TOLERANCE};
use super::{
    BOOL_BANK, F32_BANK, F64_BANK, HALF_BANK, INT_BANK, Out, Stmt, half_ulps, halves_of,
    is_half_nan, narrowed_left, pair, passed, program, ran, runnable, set_pair, single,
};
use crate::host_lower::HostPass;
use crate::interp::Machine;
use crate::ir::Opcode;
use crate::testing::config;

/// The f64 pass's operations whose result is rounded at binary32 (module docs, "Precision");
/// every other one is exact for binary32 operands.
const F64_ROUNDED: [Opcode; 9] = [
    Opcode::FPAdd64,
    Opcode::FPMul64,
    Opcode::FPFma64,
    Opcode::FPRecip64,
    Opcode::FPRecipSqrt64,
    Opcode::ConvertF64S32,
    Opcode::ConvertF64U32,
    Opcode::ConvertF64S64,
    Opcode::ConvertF64U64,
];

/// Whether two 64-bit results agree: both NaN, or `after` within the relative tolerance of
/// `before` when `rounded`, else bit for bit.
fn f64_agrees(before: u64, after: u64, rounded: bool) -> bool {
    let (b, a) = (f64::from_bits(before), f64::from_bits(after));
    if b.is_nan() || a.is_nan() {
        return b.is_nan() && a.is_nan();
    }
    if !rounded {
        return before == after;
    }
    if b.is_infinite() || a.is_infinite() {
        return before == after;
    }
    (a - b).abs() <= F64_RELATIVE_TOLERANCE * b.abs()
}

fn f32_agrees(before: u32, after: u32) -> bool {
    let (b, a) = (f32::from_bits(before), f32::from_bits(after));
    (b.is_nan() && a.is_nan()) || before == after
}

/// Whether two packed-half outputs agree, lane by lane: both NaN, within
/// [`F16_FMA_TOLERANCE_ULPS`] for `FPFma16`, else bit for bit.
fn halves_agree(before: u32, after: u32, fma: bool) -> bool {
    (0..2).all(|lane| {
        let pick = |word: u32| {
            let (low, high) = halves_of(word);
            if lane == 0 { low } else { high }
        };
        let (b, a) = (pick(before), pick(after));
        if is_half_nan(b) || is_half_nan(a) {
            return is_half_nan(b) && is_half_nan(a);
        }
        if fma {
            half_ulps(b, a) <= F16_FMA_TOLERANCE_ULPS
        } else {
            b == a
        }
    })
}

/// Compares every output of a program run before and after `pass`.
fn compare(
    pass: HostPass,
    before: &Machine,
    after: &Machine,
    outputs: &[(Opcode, Out)],
) -> Result<(), String> {
    for &(opcode, out) in outputs {
        let word = |m: &Machine, at: u8| m.regs[usize::from(at)];
        let ok = match out {
            Out::F64(at) => f64_agrees(
                pair(before, at),
                pair(after, at),
                F64_ROUNDED.contains(&opcode),
            ),
            Out::U64(at) if pass == HostPass::Fp64ToFp32 => {
                f64_agrees(pair(before, at), pair(after, at), false)
            }
            Out::U64(at) => pair(before, at) == pair(after, at),
            Out::F32(at) => f32_agrees(word(before, at), word(after, at)),
            Out::F16(at) => {
                halves_agree(word(before, at), word(after, at), opcode == Opcode::FPFma16)
            }
            Out::U32(at) | Out::Bool(at) => word(before, at) == word(after, at),
        };
        if !ok {
            return Err(format!("{opcode} ({out:?}) differs"));
        }
    }
    Ok(())
}

/// A binary32 value as the f64 pass's generated operands take it: a special, or a normal
/// number with an exponent from -30 to 30, so that products and fused multiply-adds stay in
/// binary32's normal range.
fn f32_operand() -> BoxedStrategy<u32> {
    prop_oneof![
        1 => prop::sample::select(vec![
            0_u32, 0x8000_0000, 0x3f80_0000, 0xbf80_0000, 0x3f00_0000, 0x7f80_0000, 0xff80_0000,
            0x7fc0_0000, 0x7f80_0001, 0x4b80_0000, 0xcf00_0000,
        ]),
        6 => (any::<bool>(), -30_i32..=30, 0_u32..(1 << 23)).prop_map(|(negative, exponent, mantissa)| {
            let field = u32::try_from(exponent + 127).expect("in range");
            u32::from(negative) << 31 | field << 23 | mantissa
        }),
    ]
    .boxed()
}

/// A machine whose banks hold the given values: `f64s` as binary32 values widened, `f32s`,
/// `halves` packed, `ints`, and `bools` as 0 or 1.
fn machine(f64s: &[u32], f32s: &[u32], halves: &[u32], ints: &[u32], bools: &[bool]) -> Machine {
    let mut machine = Machine::default();
    for (i, &bits) in f64s.iter().enumerate().take(usize::from(F64_BANK.1 / 2)) {
        let wide = f64::from(f32::from_bits(bits)).to_bits();
        set_pair(
            &mut machine,
            F64_BANK.0 + 2 * u8::try_from(i).expect("small"),
            wide,
        );
    }
    let fill = |machine: &mut Machine, (base, count): (u8, u8), words: &[u32]| {
        for (i, &word) in words.iter().enumerate().take(usize::from(count)) {
            machine.regs[usize::from(base) + i] = word;
        }
    };
    fill(&mut machine, F32_BANK, f32s);
    fill(&mut machine, HALF_BANK, halves);
    fill(&mut machine, INT_BANK, ints);
    let bools: Vec<u32> = bools.iter().map(|&b| u32::from(b)).collect();
    fill(&mut machine, BOOL_BANK, &bools);
    machine
}

/// Fixed, distinct inputs: binary32 values of every sign and several exponents, halves of
/// every class, and integers of every width.
fn fixed_machine() -> Machine {
    let f64s = [
        0x3fc0_0000,
        0xc0a0_0000,
        0x3eaa_aaab,
        0x4b00_0001,
        0xbf80_0000,
        0x7f80_0000,
        0x3f80_0000,
        0x0000_0000,
        0x4049_0fdb,
        0xc2f6_e979,
        0x7fc0_0000,
        0x3dcc_cccd,
        0x8000_0000,
        0x4f00_0000,
        0xbe80_0000,
        0x4120_0000,
    ];
    let f32s = [
        0x3fc0_0000,
        0xc0a0_0000,
        0x3eaa_aaab,
        0x4b00_0001,
        0xbf80_0000,
        0x7f80_0000,
        0x3f80_0000,
        0x0000_0000,
        0x4049_0fdb,
        0xc2f6_e979,
        0x7fc0_0000,
        0x3dcc_cccd,
        0x8000_0000,
        0x4f00_0000,
        0xbe80_0000,
        0x4120_0000,
    ];
    let halves = [
        0x3c00_4200,
        0xbc00_3555,
        0x7c00_fc00,
        0x7e00_0001,
        0x8000_0000,
        0x7bff_0400,
        0x4900_c780,
        0x03ff_8001,
        0x5640_3800,
        0xd640_4248,
        0x3e00_bd00,
        0x0200_7c01,
        0x4400_c400,
        0x3a00_3d00,
        0x6000_e000,
        0x2e66_ae66,
    ];
    let ints = [
        7,
        0xffff_ff85,
        300,
        65_504,
        0x8000_0000,
        0x7fff_ffff,
        0xffff_ffff,
        1 << 24 | 1,
        42,
        0,
        0x1234_5678,
        0xfedc_ba98,
        255,
        0xff80,
        3,
        0xdead_beef,
    ];
    let bools = [
        true, false, true, true, false, true, false, false, true, false, true, false, false, true,
        true, false,
    ];
    machine(&f64s, &f32s, &halves, &ints, &bools)
}

/// Every opcode each pass renames or writes out (the packed-half atomics aside, which nothing
/// runs) gives, alone, the same outputs before and after the pass within the stated precision,
/// and leaves no narrowed type behind.
#[test]
fn every_mapped_opcode_matches_the_interpreter() {
    let machine = fixed_machine();
    for pass in HostPass::ORDER {
        let opcodes = runnable(pass);
        assert_ne!(opcodes.len(), 0);
        for opcode in opcodes {
            let program = single(opcode);
            let after = passed(&program.function, pass);
            assert_eq!(narrowed_left(&after, pass), 0, "{opcode}: {after}");
            let before_run = ran(&program.function, &machine);
            let after_run = ran(&after, &machine);
            compare(pass, &before_run, &after_run, &program.outputs)
                .map_err(|why| format!("{pass}: {why}\n{}", program.function))
                .expect("the outputs agree");
        }
    }
}

fn stmts(pass: HostPass) -> BoxedStrategy<Vec<Stmt>> {
    let opcodes = runnable(pass);
    vec(
        (prop::sample::select(opcodes), any::<[u8; 8]>())
            .prop_map(|(opcode, picks)| Stmt { opcode, picks }),
        1..6,
    )
    .boxed()
}

fn generated(pass: HostPass, stmts: &[Stmt], machine: &Machine) -> Result<(), TestCaseError> {
    let program = program(stmts);
    let after = passed(&program.function, pass);
    prop_assert_eq!(narrowed_left(&after, pass), 0);
    let before_run = ran(&program.function, machine);
    let after_run = ran(&after, machine);
    compare(pass, &before_run, &after_run, &program.outputs).map_err(TestCaseError::fail)
}

proptest! {
    #![proptest_config(config())]

    /// Generated programs of 64-bit operations on binary32 operands give the same outputs
    /// before and after the f64 pass within [`F64_RELATIVE_TOLERANCE`] (exactly where the
    /// module docs say so).
    #[test]
    fn f64_programs_match_within_tolerance(
        stmts in stmts(HostPass::Fp64ToFp32),
        f64s in vec(f32_operand(), 16),
        f32s in vec(f32_operand(), 16),
        ints in vec(any::<u32>(), 16),
        bools in vec(any::<bool>(), 16),
    ) {
        let machine = machine(&f64s, &f32s, &[], &ints, &bools);
        generated(HostPass::Fp64ToFp32, &stmts, &machine)?;
    }

    /// Generated programs of 16-bit operations on any halves give the same outputs before and
    /// after the f16 pass: exactly, and `FPFma16` within [`F16_FMA_TOLERANCE_ULPS`].
    #[test]
    fn f16_programs_match_within_tolerance(
        stmts in stmts(HostPass::Fp16ToFp32),
        f32s in vec(any::<u32>(), 16),
        halves in vec(any::<u32>(), 16),
        ints in vec(any::<u32>(), 16),
        bools in vec(any::<bool>(), 16),
    ) {
        let machine = machine(&[], &f32s, &halves, &ints, &bools);
        generated(HostPass::Fp16ToFp32, &stmts, &machine)?;
    }
}
