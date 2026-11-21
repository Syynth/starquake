//! The written-out conversions: run in the interpreter against [`narrow_f64`], against Rust's
//! own IEEE 754 conversions (an independent source) where the `bits` module says they agree,
//! and against the interpreter's `PackHalf2x16` and `UnpackHalf2x16`; and each divergence from
//! Eden the `bits` module lists.

use proptest::collection::vec;
use proptest::prelude::*;

use super::super::bits::{
    Emitter, F16_CANONICAL_NAN, F32_CANONICAL_NAN, F64_CANONICAL_NAN, F64_TO_F32_BIAS,
    narrow_f64,
};
use super::{pair, reg, set_pair, u32_imm};
use crate::interp::Machine;
use crate::ir::{Builder, Function, Opcode, Operand, Terminator};
use crate::opt::Work;
use crate::testing::config;

/// How many conversions one function runs, each on its own input registers.
const BATCH: usize = 32;
/// The first output register.
const OUT: u8 = 128;
/// Where the interpreter's own conversion of input `i` goes: `OUT + THEIRS + i`.
const THEIRS: u8 = 64;

/// Which conversion a batch function runs.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Conversion {
    /// Register pair `2i` narrowed to register `OUT + i`.
    Narrow,
    /// Register `i` widened to the pair at `OUT + 2i`.
    Widen,
    /// Register `i`'s binary32 narrowed to a half in register `OUT + i`, and the interpreter's
    /// own `PackHalf2x16` of it in register `OUT + THEIRS + i`.
    PackHalf,
    /// Register `i`'s low half widened to register `OUT + i`, and the interpreter's own
    /// `UnpackHalf2x16` of it in register `OUT + THEIRS + i`.
    UnpackHalf,
}

fn op(emitter: &mut Emitter<'_, '_>, opcode: Opcode, args: &[Operand]) -> Operand {
    emitter
        .op(opcode, args)
        .unwrap_or_else(|error| panic!("{opcode}: {error}"))
}

fn small(i: usize) -> u8 {
    u8::try_from(i).expect("a small index")
}

/// The function running `conversion` [`BATCH`] times.
fn batch(conversion: Conversion) -> Function {
    let mut function = Function::new();
    let entry = function.entry();
    let mut work = Work::default();
    {
        let mut e = Emitter::new(Builder::new(&mut function, entry), &mut work);
        for i in 0..BATCH {
            let i = small(i);
            match conversion {
                Conversion::Narrow => {
                    let lo = op(&mut e, Opcode::GetRegister, &[reg(2 * i)]);
                    let hi = op(&mut e, Opcode::GetRegister, &[reg(2 * i + 1)]);
                    let bits = e.narrow(lo, hi).expect("emits");
                    op(&mut e, Opcode::SetRegister, &[reg(OUT + i), bits]);
                }
                Conversion::Widen => {
                    let x = op(&mut e, Opcode::GetRegister, &[reg(i)]);
                    let (lo, hi) = e.widen(x).expect("emits");
                    op(&mut e, Opcode::SetRegister, &[reg(OUT + 2 * i), lo]);
                    op(&mut e, Opcode::SetRegister, &[reg(OUT + 2 * i + 1), hi]);
                }
                Conversion::PackHalf => {
                    let x = op(&mut e, Opcode::GetRegister, &[reg(i)]);
                    let half = e.pack_half(x).expect("emits");
                    op(&mut e, Opcode::SetRegister, &[reg(OUT + i), half]);
                    let f = op(&mut e, Opcode::BitCastF32U32, &[x]);
                    let v = op(&mut e, Opcode::CompositeConstructF32x2, &[f, f]);
                    let word = op(&mut e, Opcode::PackHalf2x16, &[v]);
                    let low = op(&mut e, Opcode::BitwiseAnd32, &[word, u32_imm(0xffff)]);
                    op(&mut e, Opcode::SetRegister, &[reg(OUT + THEIRS + i), low]);
                }
                Conversion::UnpackHalf => {
                    let x = op(&mut e, Opcode::GetRegister, &[reg(i)]);
                    let low = op(&mut e, Opcode::BitwiseAnd32, &[x, u32_imm(0xffff)]);
                    let single = e.unpack_half(low).expect("emits");
                    op(&mut e, Opcode::SetRegister, &[reg(OUT + i), single]);
                    let v = op(&mut e, Opcode::UnpackHalf2x16, &[x]);
                    let c = op(&mut e, Opcode::CompositeExtractF32x2, &[v, u32_imm(0)]);
                    let bits = op(&mut e, Opcode::BitCastU32F32, &[c]);
                    op(&mut e, Opcode::SetRegister, &[reg(OUT + THEIRS + i), bits]);
                }
            }
        }
    }
    Builder::new(&mut function, entry)
        .terminate(Terminator::Return)
        .expect("returns");
    function
}

/// Runs `conversion` on up to [`BATCH`] inputs at a time and returns, for each input, what the
/// written-out conversion gave and what the interpreter's own conversion gave (the same for
/// the narrowing and widening, which have none).
fn convert(conversion: Conversion, inputs: &[u64]) -> Vec<(u64, u64)> {
    let function = batch(conversion);
    let mut results = Vec::with_capacity(inputs.len());
    for chunk in inputs.chunks(BATCH) {
        let mut machine = Machine::default();
        for (i, &input) in chunk.iter().enumerate() {
            let i = small(i);
            match conversion {
                Conversion::Narrow => set_pair(&mut machine, 2 * i, input),
                _ => machine.regs[usize::from(i)] = input as u32,
            }
        }
        machine.run(&function).expect("runs");
        for i in 0..chunk.len() {
            let i = small(i);
            let ours = match conversion {
                Conversion::Widen => pair(&machine, OUT + 2 * i),
                _ => u64::from(machine.regs[usize::from(OUT + i)]),
            };
            let theirs = match conversion {
                Conversion::PackHalf | Conversion::UnpackHalf => {
                    u64::from(machine.regs[usize::from(OUT + THEIRS + i)])
                }
                _ => ours,
            };
            results.push((ours, theirs));
        }
    }
    results
}

fn narrowed(bits: u64) -> u32 {
    convert(Conversion::Narrow, &[bits])[0].0 as u32
}

fn widened(bits: u32) -> u64 {
    convert(Conversion::Widen, &[u64::from(bits)])[0].0
}

/// Eden's `PackedF64ToF32` (module docs of `bits`), written out from its source to show the
/// divergence: the exponent field plus `F64ToF32Exp = +1023 - 127`, and the top 23 mantissa
/// bits kept.
fn eden_narrow(bits: u64) -> u32 {
    let (lo, hi) = (bits as u32, (bits >> 32) as u32);
    let sign = hi >> 31;
    let exp = (hi >> 20) & 0x7ff;
    let mantissa = ((hi & 0xf_ffff) << 3) | (lo >> 29);
    let exp_if_subnorm = if exp == 0 { 0 } else { exp.wrapping_add(1023 - 127) };
    let exp_if_infnan = if exp == 0x7ff { 0xff } else { exp_if_subnorm };
    (sign << 31) | (exp_if_infnan.wrapping_shl(23) | mantissa)
}

/// Inputs every narrowing test includes: zeros, ones, the binary32 range's edges and the
/// rounding ties around them, infinities and NaNs of both kinds of payload.
fn narrow_edges() -> Vec<u64> {
    let max_tie = f64::from(f32::MAX) + 2.0_f64.powi(103);
    let mut edges = vec![
        0,
        1 << 63,
        1.0_f64.to_bits(),
        (-1.0_f64).to_bits(),
        f64::from(f32::MAX).to_bits(),
        f64::MAX.to_bits(),
        f64::MIN_POSITIVE.to_bits(),
        f64::INFINITY.to_bits(),
        f64::NEG_INFINITY.to_bits(),
        0x7ff0_0000_0000_0001,
        0xfff8_0000_0000_0000,
        0x7ff0_0001_0000_0000,
        f64::from(f32::MIN_POSITIVE).to_bits(),
        (f64::from(f32::MIN_POSITIVE) / 2.0).to_bits(),
        1,
        (1.0 + 2.0_f64.powi(-24)).to_bits(),
        (1.0 + 3.0 * 2.0_f64.powi(-24)).to_bits(),
        (1.0 + 2.0_f64.powi(-24) + 2.0_f64.powi(-50)).to_bits(),
    ];
    edges.extend([max_tie, -max_tie].map(|x| x.to_bits()));
    edges.extend([max_tie.next_down(), max_tie.next_up()].map(f64::to_bits));
    edges
}

/// Whether the binary64 `bits` narrows to a normal binary32 or overflows (no NaN, no result
/// below binary32's normal range), where the narrowing is IEEE 754's.
fn ieee_range(bits: u64) -> bool {
    let x = f64::from_bits(bits);
    !x.is_nan() && x.abs() >= f64::from(f32::MIN_POSITIVE)
}

/// The written-out narrowing gives [`narrow_f64`]'s bits on the edges, and the model gives
/// Rust's `as f32` (IEEE 754 round to nearest even) wherever the result is normal or infinite.
#[test]
fn narrow_model_matches_the_emitted_operations_on_the_edges() {
    let edges = narrow_edges();
    for (&input, (ours, _)) in edges.iter().zip(convert(Conversion::Narrow, &edges)) {
        assert_eq!(ours, u64::from(narrow_f64(input)), "{input:#x}");
        if ieee_range(input) {
            assert_eq!(narrow_f64(input), (f64::from_bits(input) as f32).to_bits(), "{input:#x}");
        }
    }
}

/// The narrowing's divergence from Eden's rebias: 1.0, 2.0 and 0.5 narrow to themselves here,
/// where Eden's formula gives the bits of -1.0 for 1.0 (its exponent, 1919, overflows the
/// field into the sign).
#[test]
fn narrowing_rebiases_the_exponent_the_right_way() {
    for x in [1.0_f64, 2.0, 0.5, -3.0] {
        assert_eq!(narrowed(x.to_bits()), (x as f32).to_bits(), "{x}");
    }
    assert_eq!(eden_narrow(1.0_f64.to_bits()), (-1.0_f32).to_bits());
    assert_eq!(F64_TO_F32_BIAS, 896);
}

/// Narrowing rounds to nearest even where Eden truncates: a tie rounds to the even neighbour
/// (down from `1 + 2^-24`, up from `1 + 3 * 2^-24`), and anything past a tie rounds up.
#[test]
fn narrowing_rounds_to_nearest_even() {
    let ulp = 2.0_f32.powi(-23);
    let cases = [
        (1.0 + 2.0_f64.powi(-24), 1.0_f32),
        (1.0 + 3.0 * 2.0_f64.powi(-24), 1.0 + 2.0 * ulp),
        (1.0 + 2.0_f64.powi(-24) + 2.0_f64.powi(-50), 1.0 + ulp),
        (1.0 + 2.0_f64.powi(-24) - 2.0_f64.powi(-50), 1.0),
    ];
    for (input, wanted) in cases {
        assert_eq!(narrowed(input.to_bits()), wanted.to_bits(), "{input}");
    }
    let past_tie = (1.0 + 2.0_f64.powi(-24) + 2.0_f64.powi(-50)).to_bits();
    assert_eq!(eden_narrow(past_tie) & 0x007f_ffff, 0, "Eden truncates");
}

/// Past binary32's range the narrowing saturates to an infinity of the value's sign (from the
/// tie at the top of the range up; below it, the largest finite binary32), and below the
/// smallest normal binary32 it flushes to a zero of the value's sign, where IEEE 754 would
/// make a subnormal.
#[test]
fn narrowing_saturates_and_flushes() {
    let max_tie = f64::from(f32::MAX) + 2.0_f64.powi(103);
    let inf = f32::INFINITY.to_bits();
    assert_eq!(narrowed(max_tie.to_bits()), inf);
    assert_eq!(narrowed((-max_tie).to_bits()), f32::NEG_INFINITY.to_bits());
    assert_eq!(narrowed(max_tie.next_down().to_bits()), f32::MAX.to_bits());
    assert_eq!(narrowed(f64::MAX.to_bits()), inf);
    let smallest = f64::from(f32::MIN_POSITIVE);
    assert_eq!(narrowed(smallest.to_bits()), f32::MIN_POSITIVE.to_bits());
    assert_eq!(narrowed(smallest.next_down().to_bits()), 0);
    assert_eq!(narrowed((-smallest / 2.0).to_bits()), (-0.0_f32).to_bits());
    assert_ne!((smallest / 2.0) as f32, 0.0, "IEEE 754 makes a subnormal");
}

/// A binary32 subnormal widens to a zero of its sign (Eden's widening keeps its mantissa under
/// a zero exponent, a binary64 subnormal of the wrong value); the smallest normal widens
/// exactly.
#[test]
fn widening_flushes_subnormals() {
    assert_eq!(widened(1), 0);
    assert_eq!(widened(0x8040_0000), 1 << 63);
    assert_eq!(widened(f32::MIN_POSITIVE.to_bits()), f64::from(f32::MIN_POSITIVE).to_bits());
}

/// Every NaN becomes the canonical NaN: a binary64 NaN whose payload is only in the bits the
/// narrowing drops (which Eden's truncation makes an infinity), and a binary32 NaN widened.
#[test]
fn nans_become_canonical() {
    let low_payload = 0x7ff0_0000_0000_0001;
    assert_eq!(narrowed(low_payload), F32_CANONICAL_NAN);
    assert_eq!(eden_narrow(low_payload), f32::INFINITY.to_bits());
    assert_eq!(narrowed(0xfff8_0000_0000_0000), F32_CANONICAL_NAN);
    assert_eq!(widened(0x7f80_0001), F64_CANONICAL_NAN);
    assert_eq!(widened(0xffc0_0000), F64_CANONICAL_NAN);
}

/// Every binary16 widens to the interpreter's `UnpackHalf2x16` of it, bit for bit, with any
/// high half beside it ignored.
#[test]
fn unpack_half_matches_the_interpreter_for_every_half() {
    let inputs: Vec<u64> = (0..=u16::MAX)
        .map(|half| u64::from(half) | u64::from(half ^ 0x5a5a) << 16)
        .collect();
    for (&input, (ours, theirs)) in inputs.iter().zip(convert(Conversion::UnpackHalf, &inputs)) {
        assert_eq!(ours, theirs, "{input:#x}");
    }
}

/// Binary32 inputs for the packing: any bits, values near a binary16 rounding boundary (a
/// half widened, plus half a binary16 ulp, plus -1, 0 or 1 binary32 ulp), and values in and
/// around the binary16 subnormal range.
fn pack_input() -> BoxedStrategy<u32> {
    prop_oneof![
        any::<u32>(),
        (any::<u16>(), -1_i32..=1).prop_map(|(half, nudge)| {
            let wide = f32::from_bits(convert_half(half));
            let step = 2.0_f32.powi(-11) * wide.abs().max(2.0_f32.powi(-14));
            (wide + step).to_bits().wrapping_add_signed(nudge)
        }),
        (any::<bool>(), 90_u32..=115, 0_u32..(1 << 23))
            .prop_map(|(sign, exponent, mantissa)| u32::from(sign) << 31 | exponent << 23 | mantissa),
    ]
    .boxed()
}

/// A binary16's value as a binary32, by Rust's arithmetic (exact).
fn convert_half(half: u16) -> u32 {
    let magnitude = f32::from(half & 0x3ff);
    let exponent = i32::from((half >> 10) & 0x1f);
    let value = match exponent {
        0 => magnitude * 2.0_f32.powi(-24),
        0x1f => f32::INFINITY,
        _ => (1.0 + magnitude / 1024.0) * 2.0_f32.powi(exponent - 15),
    };
    (if half & 0x8000 == 0 { value } else { -value }).to_bits()
}

/// The packing's edges: zeros, the largest finite binary16 and the tie past it, the smallest
/// subnormal and the ties around it, infinities and NaNs.
fn pack_edges() -> Vec<u64> {
    [
        0.0_f32, -0.0, 65504.0, 65519.996, 65520.0, -65520.0, 2.0_f32.powi(-24),
        2.0_f32.powi(-25), 2.0_f32.powi(-25) * 1.000_001, 3.0 * 2.0_f32.powi(-25),
        2.0_f32.powi(-14), 2.0_f32.powi(-14) * (1.0 - 2.0_f32.powi(-12)), f32::INFINITY,
        f32::NEG_INFINITY, f32::MIN_POSITIVE, 1.0,
    ]
    .iter()
    .map(|x| u64::from(x.to_bits()))
    .chain([0x7f80_0001_u32, 0xffc0_0000, 1].map(u64::from))
    .collect()
}

/// The packing gives the interpreter's `PackHalf2x16` on its edges, a NaN giving the
/// canonical half NaN.
#[test]
fn pack_half_matches_the_interpreter_on_the_edges() {
    let edges = pack_edges();
    for (&input, (ours, theirs)) in edges.iter().zip(convert(Conversion::PackHalf, &edges)) {
        assert_eq!(ours, theirs, "{input:#x}");
    }
    let nan = convert(Conversion::PackHalf, &[0x7f80_0001])[0].0;
    assert_eq!(nan, u64::from(F16_CANONICAL_NAN));
}

proptest! {
    #![proptest_config(config())]

    /// The written-out narrowing gives [`narrow_f64`]'s bits for any input, and the model gives
    /// Rust's `as f32` wherever the result is normal or infinite.
    #[test]
    fn narrow_model_matches_the_emitted_operations(inputs in vec(any::<u64>(), BATCH)) {
        for (&input, (ours, _)) in inputs.iter().zip(convert(Conversion::Narrow, &inputs)) {
            prop_assert_eq!(ours, u64::from(narrow_f64(input)));
            if ieee_range(input) {
                prop_assert_eq!(narrow_f64(input), (f64::from_bits(input) as f32).to_bits());
            }
        }
    }

    /// The written-out widening gives Rust's `as f64` (exact) for every binary32 that is not a
    /// subnormal or a NaN.
    #[test]
    fn widening_matches_ieee(inputs in vec(any::<u32>(), BATCH)) {
        let wide: Vec<u64> = inputs.iter().map(|&x| u64::from(x)).collect();
        for (&input, (ours, _)) in inputs.iter().zip(convert(Conversion::Widen, &wide)) {
            let x = f32::from_bits(input);
            if x.is_nan() {
                prop_assert_eq!(ours, F64_CANONICAL_NAN);
            } else if x.is_subnormal() {
                prop_assert_eq!(ours, u64::from(input >> 31) << 63);
            } else {
                prop_assert_eq!(ours, f64::from(x).to_bits());
            }
        }
    }

    /// The written-out packing gives the interpreter's `PackHalf2x16`, bit for bit.
    #[test]
    fn pack_half_matches_the_interpreter(inputs in vec(pack_input(), BATCH)) {
        let wide: Vec<u64> = inputs.iter().map(|&x| u64::from(x)).collect();
        for (ours, theirs) in convert(Conversion::PackHalf, &wide) {
            prop_assert_eq!(ours, theirs);
        }
    }
}
