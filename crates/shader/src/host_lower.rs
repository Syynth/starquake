//! The host lowering passes (slice 18 of the spec's port slices; "Passes needed for
//! correctness", item 8): IR passes that run only where the host target needs them. The f64
//! pass turns 64-bit float operations into 32-bit ones where the target lacks 64-bit floats
//! (Metal has none, a fact from the Metal Shading Language specification that the spec has not
//! re-checked), and the f16 pass turns 16-bit float operations into 32-bit ones where the
//! target lacks 16-bit floats (experiment E1 found f16 accepted on Metal with `SHADER_F16`, so
//! there it runs only when the target's caps lack the feature).
//!
//! Ported by behaviour from Eden's `eden/src/shader_recompiler/ir_opt/lower_fp64_to_fp32.cpp`
//! and `lower_fp16_to_fp32.cpp` @ cf654052 (the same at `eden-pr4588` @ e5fb3b97); the `fp64`
//! and `fp16` submodules hold the tables and the `bits` submodule the conversions written out
//! with integer operations. No code is translated line by line.
//!
//! # Which passes run
//!
//! [`lower_for_host`] runs, in [`HostPass::ORDER`] (Eden's order,
//! `frontend/maxwell/translate_program.cpp` @ cf654052, lines 377–382), each pass the target's
//! [`HostCaps`] need ([`HostPass::needed`]): the f64 pass exactly when `caps.f64` is false and
//! the f16 pass exactly when `caps.f16` is false (`tests::passes::each_pass_runs_exactly_when_the_target_lacks_its_feature`).
//! The production translator calls it after collect info and before the lowering
//! ([`crate::pipeline`], step 6), with the caps of the [`crate::cache::HostTarget`] the cache
//! was created with. A target that has the feature keeps the operations: the lowering then
//! lowers 16-bit floats natively and refuses every 64-bit float operation by name
//! ([`crate::lower::LowerError::Operation`]), since no slice lowers 64-bit floats to naga yet
//! (`tests::end_to_end::a_double_program_is_refused_where_the_target_has_f64`).
//!
//! **Where it runs, against Eden.** Eden runs these passes before its SSA rewrite. The spec
//! lists them last, after dead code elimination and collect info ("Passes needed for
//! correctness", items 7 and 8), and the production translator follows the spec. The order
//! changes nothing the passes compute: they rewrite each operation by its opcode alone. The
//! operations they write are not constant-folded afterwards, which Eden's order would allow.
//!
//! # What each pass does
//!
//! Each opcode whose result or arguments name the pass's narrowed types (64-bit floats and
//! their vectors, or 16-bit floats and their vectors) is either **renamed** to its 32-bit twin
//! (its arguments unchanged; its flags kept when the twin carries the same kind, else reset to
//! the twin's defaults, as for `ConvertF32F64` becoming `Identity` or `GlobalAtomicAddF16x2`
//! becoming `GlobalAtomicAddF32x2`), **written out** with 32-bit integer operations (the
//! conversions at the register boundary: `PackDouble2x32`, `UnpackDouble2x32`,
//! `BitCastF64U64` and `BitCastU64F64`; `PackFloat2x16`, `UnpackFloat2x16`, `PackHalf2x16` and
//! `UnpackHalf2x16`), or **refused** ([`HostLowerError::NoNarrowForm`]). A written-out
//! operation's slot becomes an `Identity` of the result, as Eden's `ReplaceUsesWith` leaves it,
//! and the operations are placed just before it. A phi of a narrowed type takes the 32-bit
//! type. A 64-bit float immediate in a narrowed slot (or an `Identity` of one, or a phi's) is
//! narrowed as the written-out conversion narrows a value ([`bits`]'s `narrow_f64`). The exact
//! table is `tests::table::mappings_match_the_reference_literals`.
//!
//! # Precision
//!
//! **The f64 pass.** A 64-bit value enters 32-bit arithmetic through `PackDouble2x32` or
//! `BitCastF64U64` (narrowed: [`bits`]) and leaves through `UnpackDouble2x32` or
//! `BitCastU64F64` (widened). For operands that are binary32 values, every mapped operation
//! gives the 64-bit operation's result within a relative error of [`F64_RELATIVE_TOLERANCE`]
//! (one binary32 ulp), and exactly for comparisons, selects, composites, sign operations,
//! minimum and maximum, clamps, roundings to an integer and conversions to integers
//! (`tests::equivalence::f64_programs_match_within_tolerance`, generated, and
//! `tests::equivalence::every_mapped_opcode_matches_the_interpreter`, one case per opcode).
//! What is lost, each pinned by a test in `tests::precision`:
//!
//! | Loss | Pinned by |
//! |---|---|
//! | A 64-bit operand is rounded to the nearest binary32 (24 significant bits) | `f64_operands_round_to_binary32` |
//! | Arithmetic (`FPAdd`, `FPMul`, `FPFma`, `FPRecip`, `FPRecipSqrt`) is rounded at binary32 | `f64_arithmetic_rounds_at_binary32` |
//! | A finite value past binary32's range becomes an infinity | `f64_values_past_binary32_become_infinities` |
//! | A value below 2^-126 in magnitude (binary32's smallest normal) becomes a zero of its sign | `f64_values_below_binary32_normals_become_zeros` |
//! | A binary32 subnormal result widens to a zero of its sign | `binary32_subnormal_results_widen_to_zero` |
//! | A NaN's payload is lost: every NaN that crosses the boundary is the canonical one | `nan_payloads_are_lost` |
//! | Comparisons and conversions to integers see the rounded operand | `f64_comparisons_and_conversions_see_rounded_values` |
//! | An integer converted to a 64-bit float is rounded to binary32 | `integers_convert_to_binary32` |
//! | A rounding mode carried by `ConvertF32F64` is lost (it becomes `Identity`; the value was already rounded to nearest at the boundary) | `f64_to_f32_conversion_modes_are_lost` |
//!
//! **The f16 pass.** A packed half enters through `UnpackFloat2x16` (exact) and leaves through
//! `PackFloat2x16` (rounded to nearest even, as the interpreter rounds); in between the values
//! are binary32. One operation on binary16 operands then packed gives the 16-bit operation's
//! result exactly for every mapped operation but `FPFma16`, which may differ by
//! [`F16_FMA_TOLERANCE_ULPS`] binary16 ulp (rounding the exact result to binary32 and then to
//! binary16 is innocuous for addition and multiplication, since binary32 has more than twice
//! binary16's precision plus two bits, but not for a fused multiply-add)
//! (`tests::equivalence::f16_programs_match_within_tolerance`). What is lost, each pinned by a
//! test in `tests::precision`:
//!
//! | Loss | Pinned by |
//! |---|---|
//! | Intermediate results keep binary32 precision and range until they are packed, so a chain of operations, a comparison or a conversion to an integer of an intermediate can differ | `f16_intermediates_keep_binary32_precision_and_range` |
//! | `ConvertF16F32` becomes `Identity`: the narrowing waits for the pack | `f16_narrowing_waits_for_the_pack` |
//! | Flushing (`.FTZ`, `.FMZ`) applies to binary32 denormals, so a binary16 denormal is not flushed | `half_flushing_is_lost` |
//! | A rounding mode on 16-bit arithmetic rounds at binary32; the pack rounds to nearest even | `half_rounding_modes_apply_at_binary32` |
//! | `FPFma16` rounds twice | `half_fma_rounds_twice` |
//! | A NaN's payload is lost | `nan_payloads_are_lost` |
//!
//! The packed-half atomics (`GlobalAtomicAddF16x2` and the rest) become Eden's `F32x2` forms,
//! which Eden's backend defines as the same packed halves computed in binary32; no test runs
//! them (the interpreter and the lowering refuse both forms), so only the table pins them.
//!
//! # Checks and their order
//!
//! Nothing is changed until every check has passed (`tests::refusals::a_refused_function_is_unchanged`):
//!
//! 1. The function verifies ([`HostLowerError::Verify`]); skipped when no pass is needed.
//! 2. For each pass in [`HostPass::ORDER`], each instruction the blocks list, blocks in arena
//!    order (unreachable blocks included): an opcode naming a narrowed type that the pass has
//!    no rewrite for ([`HostLowerError::NoNarrowForm`]), then a narrowed value in an `Opaque`
//!    slot of an operation other than `Identity` ([`HostLowerError::OpaqueSlot`]).
//! 3. The arena has room for every operation the passes write ([`HostLowerError::Full`]).
//!
//! Then the passes run and, in tests and debug builds, the function is verified after each
//! ([`HostLowerError::Unverified`], a bug in this module if it happens). After the checks only
//! a bug can fail a pass ([`HostLowerError::Build`]).
//!
//! # Cost
//!
//! Each pass visits each listed instruction three times (the scan and two rewrite phases) and
//! each operand of an `Opaque` slot once, and writes a fixed number of operations per
//! written-out conversion (`tests::passes::written_out_conversions_have_fixed_sizes`), so its
//! work and the function's growth are linear in the function's size. The scan resolves an
//! `Opaque` slot's type through `Identity` chains as [`Function::type_of`] does, which the
//! verifier also does for every operand. Pinned by `tests::passes::work_is_pinned`.
//!
//! # Divergences from Eden
//!
//! | What | Here | Eden | Pinned by |
//! |---|---|---|---|
//! | Where the passes run | After collect info (the spec's order) | Before the SSA rewrite | `tests::end_to_end::the_production_translator_lowers_a_double_program` (through the production translator) |
//! | Narrowing and widening at the boundary | Correct rebias, rounding to nearest even, saturation and flushing, canonical NaNs | Wrong-signed rebias, truncation, no range handling (the `bits` module) | `tests::bits` |
//! | `FPMax64`, `FPMin64`, `FPRecip64`, `FPRecipSqrt64`, `SelectF64`, `BitCastF64U64`, `BitCastU64F64`, phis of 64-bit float types | Lowered | Left as 64-bit operations | `tests::table::every_narrowed_opcode_is_mapped_or_refused`, `tests::passes::phis_take_the_narrowed_type` |
//! | `SelectF16`, phis of 16-bit float types | Lowered | Left | the same tests |
//! | The four half-packing operations | Written out with integer operations | `PackHalf2x16` and `UnpackHalf2x16` kept | `tests::equivalence::every_mapped_opcode_matches_the_interpreter` |
//! | `BitCastU16F16`, `BitCastF16U16`, a narrowed value in another operation's `Opaque` slot | Refused | Left | `tests::refusals` |
//! | A 64-bit float immediate | Narrowed with the boundary's conversion | Kept: the renamed operation then takes a 64-bit immediate | `tests::passes::f64_immediates_are_narrowed` |

mod bits;
mod fp16;
mod fp64;
#[cfg(test)]
mod tests;

use std::fmt::{self, Display, Formatter};

use thiserror::Error;

use crate::host::HostCaps;
use crate::ir::{
    BlockId, BuildError, Builder, Flags, Function, Imm, Inst, Opcode, Operand, Type, Value,
    VerifyError,
};
use crate::opt::{Work, arena_len, identity_of};
use bits::Emitter;

/// The f64 pass's stated precision for operands that are binary32 values (module docs,
/// "Precision"): a relative error of at most one binary32 ulp, `2^-23`.
pub const F64_RELATIVE_TOLERANCE: f64 = f32::EPSILON as f64;

/// The f16 pass's stated precision for `FPFma16` (module docs, "Precision"): at most one
/// binary16 ulp; every other mapped 16-bit operation is exact.
pub const F16_FMA_TOLERANCE_ULPS: u32 = 1;

/// The most instructions a function's arena can hold: ids are `u32`.
const MAX_INSTRUCTIONS: u64 = u32::MAX as u64 + 1;

/// One of the host lowering passes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum HostPass {
    /// 64-bit floats become 32-bit floats (Eden's `LowerFp64ToFp32`).
    Fp64ToFp32,
    /// 16-bit floats become 32-bit floats (Eden's `LowerFp16ToFp32`).
    Fp16ToFp32,
}

impl HostPass {
    /// The order [`lower_for_host`] runs the passes in: Eden's (module docs).
    pub const ORDER: [Self; 2] = [Self::Fp64ToFp32, Self::Fp16ToFp32];

    /// The pass's name, for errors.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Fp64ToFp32 => "the 64-bit float lowering",
            Self::Fp16ToFp32 => "the 16-bit float lowering",
        }
    }

    /// The feature whose absence calls for the pass, in plain words.
    pub const fn feature(self) -> &'static str {
        match self {
            Self::Fp64ToFp32 => "64-bit floats",
            Self::Fp16ToFp32 => "16-bit floats",
        }
    }

    /// Whether a target with `caps` needs the pass: exactly when it lacks the pass's feature.
    pub const fn needed(self, caps: HostCaps) -> bool {
        match self {
            Self::Fp64ToFp32 => !caps.f64,
            Self::Fp16ToFp32 => !caps.f16,
        }
    }

    /// What `opcode` becomes under this pass.
    const fn rewrite(self, opcode: Opcode) -> Rewrite {
        match self {
            Self::Fp64ToFp32 => fp64::rewrite(opcode),
            Self::Fp16ToFp32 => fp16::rewrite(opcode),
        }
    }

    /// The 32-bit type `ty` becomes under this pass, if it is one the pass narrows.
    const fn narrow_type(self, ty: Type) -> Option<Type> {
        match self {
            Self::Fp64ToFp32 => fp64::narrow_type(ty),
            Self::Fp16ToFp32 => fp16::narrow_type(ty),
        }
    }

    /// Whether `opcode`'s result or an argument is of a type this pass narrows.
    fn names_narrowed_type(self, opcode: Opcode) -> bool {
        std::iter::once(opcode.result())
            .chain(opcode.args().iter().copied())
            .any(|ty| self.narrow_type(ty).is_some())
    }

    /// `imm` narrowed, if it is of a type this pass narrows (only 64-bit float immediates
    /// exist among the narrowed types).
    const fn narrow_imm(self, imm: Imm) -> Option<Imm> {
        match (self, imm) {
            (Self::Fp64ToFp32, Imm::F64(bits)) => Some(Imm::F32(bits::narrow_f64(bits))),
            _ => None,
        }
    }
}

impl Display for HostPass {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// What a pass does with one opcode (module docs, "What each pass does").
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Rewrite {
    /// Nothing: the opcode names no narrowed type, or it does and is refused.
    Keep,
    /// The opcode becomes this one.
    Rename(Opcode),
    /// The operation is written out.
    Expand(Expansion),
}

/// A conversion a pass writes out with integer operations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Expansion {
    /// `PackDouble2x32`: two words narrowed to a 32-bit float.
    PackDouble,
    /// `UnpackDouble2x32`: a 32-bit float widened to two words.
    UnpackDouble,
    /// `BitCastF64U64`: a 64-bit integer's bits narrowed to a 32-bit float.
    DoubleFromBits,
    /// `BitCastU64F64`: a 32-bit float widened to a 64-bit integer's bits.
    BitsFromDouble,
    /// `PackFloat2x16` and `PackHalf2x16`: two 32-bit floats narrowed to two packed halves.
    PackHalves,
    /// `UnpackFloat2x16` and `UnpackHalf2x16`: two packed halves widened to two 32-bit floats.
    UnpackHalves,
}

impl Expansion {
    /// How many operations the conversion writes.
    const fn ops(self) -> u64 {
        match self {
            Self::PackDouble => 2 + bits::NARROW_OPS + 1,
            Self::UnpackDouble => 1 + bits::WIDEN_OPS + 1,
            Self::DoubleFromBits => 3 + bits::NARROW_OPS + 1,
            Self::BitsFromDouble => 1 + bits::WIDEN_OPS + 2,
            Self::PackHalves => 4 + 2 * bits::PACK_HALF_OPS + 2,
            Self::UnpackHalves => 2 + 2 * bits::UNPACK_HALF_OPS + 3,
        }
    }

    /// Whether the conversion's input is of a type no pass narrows, so it can be written out
    /// in the first phase, before the operations that read the narrowed values have their new
    /// types (the others read a narrowed value and wait for the second phase).
    const fn first_phase(self) -> bool {
        matches!(
            self,
            Self::PackDouble | Self::DoubleFromBits | Self::UnpackHalves
        )
    }

    /// Writes the conversion of `arg` and returns its result.
    fn emit(self, emitter: &mut Emitter<'_, '_>, arg: Operand) -> Result<Operand, BuildError> {
        use Opcode as O;
        let first = Operand::Imm(Imm::U32(0));
        let second = Operand::Imm(Imm::U32(1));
        match self {
            Self::PackDouble | Self::DoubleFromBits => {
                let pair = if self == Self::DoubleFromBits {
                    emitter.op(O::UnpackUint2x32, &[arg])?
                } else {
                    arg
                };
                let lo = emitter.op(O::CompositeExtractU32x2, &[pair, first])?;
                let hi = emitter.op(O::CompositeExtractU32x2, &[pair, second])?;
                let narrowed = emitter.narrow(lo, hi)?;
                emitter.op(O::BitCastF32U32, &[narrowed])
            }
            Self::UnpackDouble | Self::BitsFromDouble => {
                let raw = emitter.op(O::BitCastU32F32, &[arg])?;
                let (lo, hi) = emitter.widen(raw)?;
                let pair = emitter.op(O::CompositeConstructU32x2, &[lo, hi])?;
                if self == Self::BitsFromDouble {
                    emitter.op(O::PackUint2x32, &[pair])
                } else {
                    Ok(pair)
                }
            }
            Self::PackHalves => {
                let low = emitter.op(O::CompositeExtractF32x2, &[arg, first])?;
                let high = emitter.op(O::CompositeExtractF32x2, &[arg, second])?;
                let low = emitter.op(O::BitCastU32F32, &[low])?;
                let high = emitter.op(O::BitCastU32F32, &[high])?;
                let low = emitter.pack_half(low)?;
                let high = emitter.pack_half(high)?;
                let shift = Operand::Imm(Imm::U32(bits::F16_SIGN_SHIFT));
                let high = emitter.op(O::ShiftLeftLogical32, &[high, shift])?;
                emitter.op(O::BitwiseOr32, &[low, high])
            }
            Self::UnpackHalves => {
                let mask = Operand::Imm(Imm::U32(u32::from(u16::MAX)));
                let shift = Operand::Imm(Imm::U32(bits::F16_SIGN_SHIFT));
                let low = emitter.op(O::BitwiseAnd32, &[arg, mask])?;
                let high = emitter.op(O::ShiftRightLogical32, &[arg, shift])?;
                let low = emitter.unpack_half(low)?;
                let high = emitter.unpack_half(high)?;
                let low = emitter.op(O::BitCastF32U32, &[low])?;
                let high = emitter.op(O::BitCastF32U32, &[high])?;
                emitter.op(O::CompositeConstructF32x2, &[low, high])
            }
        }
    }
}

/// What [`lower_for_host`] did.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Applied {
    ran: [bool; 2],
    work: Work,
}

impl Applied {
    /// Whether `pass` ran.
    pub const fn ran(&self, pass: HostPass) -> bool {
        self.ran[pass_index(pass)]
    }

    /// The work the passes did, the scans included.
    pub const fn work(&self) -> Work {
        self.work
    }
}

/// `pass`'s index in [`HostPass::ORDER`].
const fn pass_index(pass: HostPass) -> usize {
    match pass {
        HostPass::Fp64ToFp32 => 0,
        HostPass::Fp16ToFp32 => 1,
    }
}

/// Why a host lowering pass did not run (module docs, "Checks and their order").
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum HostLowerError {
    /// The function is malformed.
    #[error("the IR is malformed: {0}")]
    Verify(VerifyError),
    /// An operation names a narrowed type and has no 32-bit form.
    #[error(
        "{at}: {opcode} has no 32-bit form, which a host target without {feature} needs",
        feature = pass.feature()
    )]
    NoNarrowForm {
        /// The pass.
        pass: HostPass,
        /// The instruction.
        at: Value,
        /// Its opcode.
        opcode: Opcode,
    },
    /// A narrowed value fills an `Opaque` slot of an operation other than `Identity`, whose
    /// meaning the pass cannot know to keep.
    #[error("{at}: argument {index} of {opcode} is a {ty} value, which {pass} cannot narrow there")]
    OpaqueSlot {
        /// The pass.
        pass: HostPass,
        /// The instruction.
        at: Value,
        /// Its opcode.
        opcode: Opcode,
        /// The argument's index.
        index: usize,
        /// The argument's type.
        ty: Type,
    },
    /// The operations the passes would write do not fit the function's arena.
    #[error(
        "the host lowering would need {needed} instructions, more than the {max} a function holds",
        max = MAX_INSTRUCTIONS
    )]
    Full {
        /// How many instructions the function would hold.
        needed: u64,
    },
    /// A pass could not write an operation after its checks passed: a bug in this module.
    #[error("{pass} could not write an operation: {error}")]
    Build {
        /// The pass.
        pass: HostPass,
        /// Why.
        error: BuildError,
    },
    /// A pass's output does not verify: a bug in this module.
    #[error("after {pass}: {error}")]
    Unverified {
        /// The pass.
        pass: HostPass,
        /// What the verifier found.
        error: VerifyError,
    },
}

/// Whether [`run`] and [`lower_for_host`] verify each pass's output: in tests and debug builds.
const VERIFY: bool = cfg!(any(test, debug_assertions));

/// Runs each pass a target with `caps` needs on `function`, in [`HostPass::ORDER`] (module
/// docs, "Which passes run"). A refusal leaves the function unchanged.
pub fn lower_for_host(function: &mut Function, caps: HostCaps) -> Result<Applied, HostLowerError> {
    let mut applied = Applied::default();
    let needed = HostPass::ORDER.map(|pass| pass.needed(caps));
    if needed == [false; 2] {
        return Ok(applied);
    }
    function.verify().map_err(HostLowerError::Verify)?;
    let mut work = Work::default();
    let mut added = 0_u64;
    for (pass, _) in HostPass::ORDER.into_iter().zip(needed).filter(|(_, n)| *n) {
        added = added.saturating_add(scan(function, pass, &mut work)?);
    }
    check_room(arena_len(function), added)?;
    for (pass, _) in HostPass::ORDER.into_iter().zip(needed).filter(|(_, n)| *n) {
        apply(function, pass, &mut work)?;
        if let Some(ran) = applied.ran.get_mut(pass_index(pass)) {
            *ran = true;
        }
    }
    applied.work = work;
    Ok(applied)
}

/// Runs `pass` on `function` whatever the target, with the same checks as [`lower_for_host`].
/// A refusal leaves the function unchanged.
pub fn run(function: &mut Function, pass: HostPass) -> Result<Work, HostLowerError> {
    function.verify().map_err(HostLowerError::Verify)?;
    let mut work = Work::default();
    let added = scan(function, pass, &mut work)?;
    check_room(arena_len(function), added)?;
    apply(function, pass, &mut work)?;
    Ok(work)
}

/// Whether a function whose arena holds `len` instructions has room for `added` more.
const fn fits(len: u64, added: u64) -> bool {
    match len.checked_add(added) {
        Some(total) => total <= MAX_INSTRUCTIONS,
        None => false,
    }
}

/// [`fits`] for a function's arena, as a refusal.
fn check_room(len: usize, added: u64) -> Result<(), HostLowerError> {
    let len = u64::try_from(len).unwrap_or(u64::MAX);
    if fits(len, added) {
        Ok(())
    } else {
        Err(HostLowerError::Full {
            needed: len.saturating_add(added),
        })
    }
}

/// Checks that `pass` can rewrite every listed instruction (module docs, "Checks and their
/// order", step 2) and returns how many operations it will write.
fn scan(function: &Function, pass: HostPass, work: &mut Work) -> Result<u64, HostLowerError> {
    let mut added = 0_u64;
    for (at, inst) in function.listed() {
        work.step();
        let Inst::Op(op) = inst else { continue };
        match pass.rewrite(op.opcode) {
            Rewrite::Rename(_) => {}
            Rewrite::Expand(expansion) => added = added.saturating_add(expansion.ops()),
            Rewrite::Keep => {
                if pass.names_narrowed_type(op.opcode) {
                    return Err(HostLowerError::NoNarrowForm {
                        pass,
                        at,
                        opcode: op.opcode,
                    });
                }
                if op.opcode == Opcode::Identity {
                    continue;
                }
                for (index, (&slot, &arg)) in op.opcode.args().iter().zip(&op.args).enumerate() {
                    if slot != Type::Opaque {
                        continue;
                    }
                    work.step();
                    if let Some(ty) = function.type_of(arg)
                        && pass.narrow_type(ty).is_some()
                    {
                        return Err(HostLowerError::OpaqueSlot {
                            pass,
                            at,
                            opcode: op.opcode,
                            index,
                            ty,
                        });
                    }
                }
            }
        }
    }
    Ok(added)
}

/// Rewrites every listed instruction under `pass` (module docs, "What each pass does"), after
/// [`scan`] accepted the function, then verifies it in tests and debug builds.
fn apply(function: &mut Function, pass: HostPass, work: &mut Work) -> Result<(), HostLowerError> {
    let blocks: Vec<BlockId> = function.blocks().map(|(id, _)| id).collect();
    for first in [true, false] {
        for &block in &blocks {
            rewrite_block(function, block, pass, first, work)
                .map_err(|error| HostLowerError::Build { pass, error })?;
        }
    }
    if VERIFY {
        function
            .verify()
            .map_err(|error| HostLowerError::Unverified { pass, error })?;
    }
    Ok(())
}

/// One phase of [`apply`] on one block: the first renames, retypes phis, narrows immediates
/// and writes out the conversions whose input no pass narrows; the second writes out the rest.
fn rewrite_block(
    function: &mut Function,
    block: BlockId,
    pass: HostPass,
    first: bool,
    work: &mut Work,
) -> Result<(), BuildError> {
    let listed = function
        .block(block)
        .map(|b| b.insts.clone())
        .ok_or(BuildError::NoBlock(block))?;
    let base = listed.len();
    let mut order = Vec::with_capacity(base);
    let mut grew = false;
    for value in listed {
        work.step();
        if let Some(expansion) = rewrite_in_place(function, value, pass, first) {
            // Every written-out conversion takes one argument, which the verified function
            // has.
            let Some(arg) = (match function.inst(value) {
                Some(Inst::Op(op)) => op.args.first().copied(),
                _ => None,
            }) else {
                order.push(value);
                continue;
            };
            let result = expansion.emit(
                &mut Emitter::new(Builder::new(function, block), work),
                arg,
            )?;
            let written = function
                .block_mut(block)
                .map(|b| b.insts.split_off(base))
                .ok_or(BuildError::NoBlock(block))?;
            order.extend(written);
            if let Some(inst) = function.inst_mut(value) {
                *inst = identity_of(result);
            }
            grew = true;
        }
        order.push(value);
    }
    if grew && let Some(b) = function.block_mut(block) {
        b.insts = order;
    }
    Ok(())
}

/// Rewrites `value`'s instruction in place for this phase, and returns the conversion to write
/// out for it, if one is due in this phase.
fn rewrite_in_place(
    function: &mut Function,
    value: Value,
    pass: HostPass,
    first: bool,
) -> Option<Expansion> {
    match function.inst_mut(value)? {
        Inst::Phi(phi) => {
            if first && let Some(ty) = pass.narrow_type(phi.ty) {
                phi.ty = ty;
                for (_, operand) in &mut phi.incoming {
                    narrow_operand(pass, operand);
                }
            }
            None
        }
        Inst::Op(op) => match pass.rewrite(op.opcode) {
            Rewrite::Keep => {
                if first && op.opcode == Opcode::Identity {
                    op.args.iter_mut().for_each(|arg| narrow_operand(pass, arg));
                }
                None
            }
            Rewrite::Rename(renamed) => {
                if first {
                    for (slot, arg) in op.opcode.args().iter().zip(op.args.iter_mut()) {
                        if pass.narrow_type(*slot).is_some() {
                            narrow_operand(pass, arg);
                        }
                    }
                    if renamed.flags_kind() != op.opcode.flags_kind() {
                        op.flags = Flags::default_of(renamed.flags_kind());
                    }
                    op.opcode = renamed;
                }
                None
            }
            Rewrite::Expand(expansion) => (expansion.first_phase() == first).then(|| {
                op.args.iter_mut().for_each(|arg| narrow_operand(pass, arg));
                expansion
            }),
        },
    }
}

/// Narrows `operand` in place if it is an immediate of a type `pass` narrows.
fn narrow_operand(pass: HostPass, operand: &mut Operand) {
    if let Operand::Imm(imm) = *operand
        && let Some(narrowed) = pass.narrow_imm(imm)
    {
        *operand = Operand::Imm(narrowed);
    }
}
