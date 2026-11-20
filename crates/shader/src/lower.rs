//! Lowering the IR to a validated naga module and its interface tables.
//!
//! This module (with its submodules) is the only code that builds naga IR (spec, "The backend
//! through naga"), so a naga upgrade touches one place. Slice 1 built the skeleton for
//! straight-line vertex and pixel functions; slice 16 widens it to the structured IR the
//! structurizer makes (slice 11), compute entry points, arithmetic, conversions, comparisons and
//! predicates, and the local and shared memory pass ("Passes needed for correctness", item 6).
//!
//! # The entry points
//!
//! [`lower_with`] takes the stage, the function and [`LowerInputs`]: the workgroup size and
//! shared-memory size (from the variant key, [`crate::variant::VariantKey`]), the local-memory
//! size (from the SPH or the QMD) and the host target's features ([`HostCaps`]). [`lower`] is
//! [`lower_with`] with [`LowerInputs::DEFAULT`], what slice 1's [`crate::program`] chain uses.
//!
//! # What it takes
//!
//! - **Control flow.** A structured function (the tree of [`Node`]s), or a control-flow graph of
//!   one block that ends in `return` or `unreachable`, which lowers as that block followed by
//!   the node of the same name. Any other control-flow graph is [`LowerError::ControlFlow`]:
//!   it is to be structured first ([`crate::structure`]).
//! - **Form.** The form the structurizer leaves, or the SSA form the passes leave after it
//!   (the owner's ruling of 2026-11-19, decision log, "Owner rulings on the wave 44 and 45
//!   questions", row 3: Eden's order, the structurizer first and SSA after it, on the
//!   structured function; slice 8b, `crate::opt`). Guest registers, predicates, uniform
//!   registers and predicates, the four condition-code flags, the goto variables and the
//!   indirect branch variable that are still read and written through their get and set
//!   opcodes each become a local variable of the entry point, zero (or false) at entry, as the
//!   interpreter reads an unwritten one ([`crate::interp`]); after the SSA rewrite and dead
//!   code elimination none is left. Each phi becomes a local variable of its type, stored at
//!   the end of each block it names and loaded at the start of its own block, which is the
//!   structured-phi rule (the value comes from the block run last; "Structured control
//!   flow"). A phi of a type with no naga type here (the 64-bit float types) is
//!   [`LowerError::Phi`].
//!
//! # Structured control flow
//!
//! naga's blocks scope their expressions as the IR's structured scope rule does (`ir` module
//! docs), so each IR value is the naga expression made where its instruction is. An `if` is a
//! naga `If` with an empty `reject`; a loop is a naga `Loop` whose body ends with
//! `if repeat {} else { break }`, so the repeat condition is read in the body's scope, after it,
//! as the verifier checks it; a `break` is `if cond { break }` (naga's `If` is not a break
//! target, so it leaves the innermost loop); `return` returns the output struct (or nothing);
//! the entry point's body always ends with that return. A phi's incoming value is stored at
//! the end of the block it names, where it must already be computed and in scope (an `if` or
//! loop body's values leave scope when the body ends, as naga's do); the SSA rewrite only
//! names such values (`opt::ssa`), and any other is [`LowerError::PhiIncoming`]
//! (`tests::structured::a_phi_value_out_of_scope_is_refused`). **Divergence:** `unreachable` lowers as
//! the same return. Eden emits `OpUnreachable` (`emit_spirv_control_flow.cpp`), whose
//! behaviour is undefined when reached; naga has no such statement, and a return is defined.
//! Pinned by `tests::structured::unreachable_returns`.
//!
//! # Operations
//!
//! The opcodes lowered, by group; every other opcode is [`LowerError::Operation`]
//! (`tests::opcodes::every_opcode_is_lowered_or_refused` enumerates [`Opcode::ALL`]):
//!
//! - No-ops: `Void`, `Prologue`, `Epilogue` (the interpreter's no-ops); `Identity` and
//!   `ConditionRef` are their argument.
//! - Variables: `GetRegister` to `SetIndirectBranchVariable` and the flag gets and sets, as
//!   above (a goto variable's index must be an immediate); `UndefU1`, `UndefU32` and `UndefU64`
//!   are zero, as in the interpreter.
//! - The interface: `GetAttribute`, `SetAttribute` and `SetFragColor` as slice 1 lowered them
//!   (below), `LaneId`, and in compute programs `WorkgroupId` and `LocalInvocationId`.
//! - Control: `DemoteToHelperInvocation` (pixel programs only) and the three barriers (compute
//!   programs only, as naga allows: `Barrier` is `workgroupBarrier`, `WorkgroupMemoryBarrier`
//!   and `DeviceMemoryBarrier` are memory barriers on workgroup and storage memory).
//! - Selects (`SelectU1`, `SelectU32`, `SelectU64`, `SelectF16`, `SelectF32`), bit casts
//!   (`BitCastU32F32`, `BitCastF32U32`), packing (`PackUint2x32`, `UnpackUint2x32`,
//!   `PackHalf2x16`, `UnpackHalf2x16`, `PackFloat2x16`, `UnpackFloat2x16`), and the 32-bit
//!   integer and 16- and 32-bit float composites (construct, and extract and insert at an
//!   immediate component).
//! - The 32-bit and 16-bit float operations of the table from `FPAbs16` to `FPIsNan64`
//!   (`FPRecip32`, `FPRecipSqrt32`, `FPSqrt` and the transcendentals have only a 32-bit form);
//!   every 64-bit float operation is refused until slice 18 lowers f64.
//! - 32-bit integer arithmetic, bit operations and comparisons (`IAdd32` to
//!   `UGreaterThanEqual`), the 64-bit `IAdd64`, `ISub64`, `INeg64`, `IAbs64` and shifts, and
//!   the logical operations.
//! - Conversions: f16 and f32 to 16- and 32-bit integers (`ConvertS16F16` to `ConvertU32F32`
//!   without the f64 and 64-bit ones), 8-, 16- and 32-bit integers to f16 and f32
//!   (`ConvertF16S8` to `ConvertF32U32` without the 64-bit ones), f16 to and from f32, and u32
//!   to and from u64 (`ConvertU64U32`, `ConvertU32U64`). The integer-to-integer conversions
//!   with an 8- or 16-bit type (`ConvertU16U32` to `ConvertS32S16`) are refused: the lowering
//!   has no 8- or 16-bit integer type.
//! - Local and shared memory (below).
//!
//! Each operation follows the interpreter's semantics ([`crate::interp`], "Semantics") where
//! naga's (which are WGSL's) leave a case undefined or differ: a shift by the width or more
//! gives 0 (logical) or the sign (arithmetic); `UDiv32` by zero gives `0xffff_ffff` and
//! `SDiv32` by zero `-1`; a bit field's offset and count are clamped as the interpreter clamps
//! them, and `BitFieldSExtract` sign-extends from the top of the field the count names, not of
//! the bits that exist (naga's `extractBits` would extend from bit 31); `IAbs` of the minimum
//! is itself; `FPMax32` and `FPMin32` return the operand that is not a NaN; `FPClamp` is
//! `min(max(value, low), high)` with those; a float-to-integer conversion truncates, saturates
//! and gives 0 for a NaN; an ordered comparison is false and an unordered one true when an
//! operand is a NaN. Pinned on the host's GPU, with operands the host compiler cannot fold,
//! against the interpreter by `tests::ops` (the `*_match_the_interpreter_on_the_gpu` tests,
//! which skip without a GPU adapter), and everywhere by the MSL of the same cases
//! (`tests::ops::operations_snapshot`).
//!
//! # Interface
//!
//! - Each guest register the function uses becomes a `u32` local variable, zero at entry;
//!   `GetReg` and `SetReg` load and store it.
//! - A vertex function's input attributes become vertex inputs: generic attribute `n` is a
//!   `vec4<f32>` at location `n`. Its outputs are one struct: the position built-in, then
//!   generic `n` at location `n`, with perspective interpolation at the pixel centre. That is
//!   still slice 1's guess: slice 15 now collects the pixel header's modes
//!   ([`crate::info::ShaderInfo::interpolation`]), but nothing here reads them yet. Slice 17
//!   lowers them, subject to the open owner question on where a vertex program's outputs get
//!   the pixel program's modes ([`crate::info`], "Open, for the owner"). Pixel generic inputs
//!   are refused here ([`LowerError::Input`]) until then. Components the program does not
//!   write are zero. Generics from [`HOST_LOCATIONS`] up (generic 31) are refused, read or
//!   written, because wgpu on Metal binds no location past 30.
//! - A pixel function's colour target `n` is a `vec4<f32>` output at location `n`; unwritten
//!   components are zero. `SR_LANEID` is the subgroup invocation id built-in, which needs the
//!   subgroup capability; naga 30.0.1 does not allow it in a vertex shader
//!   (`naga/src/valid/interface.rs`: `SubgroupInvocationId` is an input of compute, fragment,
//!   task, mesh and ray stages, not vertex).
//! - A compute function has no outputs. Its workgroup and local invocation ids are the
//!   built-ins of those names; they are arguments of the entry point but not interface slots,
//!   since nothing outside the shader binds them. The workgroup size is the entry point's
//!   constant size ([`HostShader::workgroup_size`]).
//! - A 32-bit immediate operand becomes a literal at its use, where slice 1's IR had an `Imm32`
//!   instruction, so the expressions keep their order (`tests::skeleton_modules_are_unchanged`).
//!
//! Generic attributes are attribute words 32 to 159, `n = (word - 32) / 4`, as Eden numbers
//! them (`eden/src/shader_recompiler/frontend/ir/attribute.h` @ cf654052: `Generic0X = 32`,
//! `Generic31W = 159`, `GenericAttributeIndex`). Divergence: Eden's `IsGeneric` stops at
//! `Generic31X` (156), so its words 157 to 159 are not generics there; here they are generic
//! 31's Y, Z and W, and so refused with the rest of generic 31 as
//! [`LowerError::GenericLocation`] rather than as an attribute the stage cannot use. Pinned by
//! `tests::generic_31_w_is_a_generic`.
//!
//! # Local and shared memory
//!
//! The `memory` submodule: local memory is a private array of `u32` words, sized from the
//! SPH (graphics) or the QMD (compute), shared memory a workgroup array sized from the QMD
//! (through the variant key), each `ceil(bytes / 4)` words as Eden sizes them
//! (`backend/spirv/spirv_emit_context.cpp` @ cf654052, `DefineLocalMemory` and the
//! non-explicit-layout arm of `DefineSharedMemory`). Every word an access touches is
//! bounds-checked: a read past the end gives 0 and a write past the end is dropped (the
//! submodule lists where this differs from Eden).
//!
//! # Requirements and the host target
//!
//! A function needs [`Requirements::f16`] when an operation it lowers has a 16-bit float in its
//! signature, [`Requirements::i64`] when one has a 64-bit integer, and
//! [`Requirements::subgroup`] when it reads the lane id. A requirement the target lacks
//! ([`LowerInputs::caps`]) is [`LowerError::Missing`], checked after the body is lowered, in
//! [`Feature`] order. The module is validated with exactly the capabilities its requirements
//! name ([`capabilities`]).
//!
//! # Checks and their order
//!
//! 1. The function verifies ([`LowerError::Verify`]).
//! 2. The stage is lowered ([`LowerError::Stage`]: vertex, pixel and compute).
//! 3. The inputs: a compute program's workgroup size ([`LowerError::WorkgroupSize`]), the
//!    local-memory size ([`LowerError::LocalMemory`]), a compute program's shared-memory size
//!    ([`LowerError::SharedMemory`]).
//! 4. The control-flow form ([`LowerError::ControlFlow`]).
//! 5. Each listed instruction, blocks in arena order, for what the interface and the variables
//!    need: phi types, name and attribute operands, stages, colour outputs.
//! 6. The body, in tree order: unlowered operations and operand forms.
//! 7. The requirements against the target ([`LowerError::Missing`]).
//! 8. naga's validation ([`LowerError::Validation`], a bug in this module if it happens).
//!
//! # Divergences
//!
//! | What | Here | Reference | Pinned by |
//! |---|---|---|---|
//! | Rounding modes and flush-to-zero on arithmetic | Not expressed (decision 10): `FPFma32` is naga's fused `fma` whatever its `FpControl` | Eden's SPIR-V decorations | `tests::skeleton_modules_are_unchanged` (the `modifiers` program's `rz` and `ftz`) |
//! | A rounding mode or flushing mode carried on a conversion (`ConvertF*` with `rn`, `rm`, `rp`, `rz`, `ftz`, `fmz` or `none`) | Ignored: the module is the one the unset controls give | Eden emits them as SPIR-V decorations | `tests::opcodes::conversions_lower_whatever_their_rounding` |
//! | `F2F` (to half) with a directed rounding mode or `.FTZ` | Emulated upstream, as decision 10 (a) asks: `translate::float` converts to nearest and steps the half, and flushes the operand and a denormal result ("Rounding modes and flush-to-zero"); the mode and flushing it also carries on `ConvertF16F32` are ignored here | Eden: SPIR-V decorations | `tests::ops::translated_f2f_rounding_matches_the_interpreter_on_the_gpu` |
//! | `I2F` and `I2FP` (integer to float) with a directed rounding mode | Rounds as the host does (to nearest even on an Apple GPU, measured), with or without a mode: decision 10 (a) treats it like arithmetic and asks for no emulation | The interpreter honours the carried mode; Eden: a SPIR-V decoration | `tests::ops::int_to_float_rounds_as_the_host_does_on_the_gpu` |
//! | `F2I.FTZ` (float to integer) | The translator carries it on the rounding operation in front of the conversion, which is ignored here as on arithmetic (the first row); with a 32-bit operand the translator also flushes upstream, and with a half operand it does not, as the owner's ruling of 2026-11-19 (row 5) accepts | SPIR-V decoration | `translate::float`'s `tests::convert::f2i_flushes_denormals_before_rounding` (upstream) |
//! | `DemoteToHelperInvocation` | naga's `Kill`, which ends the invocation; naga has no demote (spec, the naga table) | `OpDemoteToHelperInvocationEXT`: the invocation continues as a helper | `tests::structured::demote_is_kill` |
//! | `unreachable` | A return | `OpUnreachable` | `tests::structured::unreachable_returns` |
//! | `FPSaturate16` and `FPSaturate32` of a NaN | naga's `saturate`, whose NaN result is the host's | The interpreter gives `+0` | `tests::skeleton_modules_are_unchanged` |
//! | NaN handling (comparisons, NaN-aware minimum, maximum and clamp, NaN to 0 on conversion) | Written out with `isnan`: wgpu-hal 30.0.2's Metal backend leaves the Metal compiler's default fast math on, under which a plain comparison treats a NaN as an ordinary value (measured on an Apple GPU: two NaNs compared equal); `isnan` and selects on it gave the interpreter's results there. Other hosts are not measured | The interpreter's results | `tests::ops::float_operations_match_the_interpreter_on_the_gpu` |
//! | `f32` and `f16` from an integer, no mode carried | naga's conversion, rounding as the host does (to nearest even on an Apple GPU, measured) | Round to nearest even (the interpreter) | `tests::ops::int_to_float_rounds_as_the_host_does_on_the_gpu` (the same module, by `tests::opcodes::conversions_lower_whatever_their_rounding`) |
//! | NaN result bits of float arithmetic: `FPAdd32` of a NaN, `FPMul32` of infinity by 0, `FPSqrt` of -1, `FPMax32` of two NaNs, an `FPAdd16` NaN widened to f32 (the cases measured) | The host's bits, not canonicalised: an Apple GPU gave 0x7fc0_0000 for each | The interpreter's canonical NaN (the binary32 one is itself a guess, `interp::float`) | `tests::ops::nan_results_are_the_hosts_on_the_gpu` |
//! | A phi | A local variable: stored at the end of each block it names, loaded at the start of its block | Eden's `EmitPhi` (`emit_spirv.cpp` @ cf654052), an `OpPhi`, whose operands it patches in afterwards | `tests::structured::structured_phis_match_the_interpreter_on_the_gpu` |
//! | A phi in the first block run (no block ran before it) | Reads its local variable's zero | The interpreter refuses it (`PhiAtStart`); Eden has no such phi (its first block has no predecessors, and its rewrite makes no phi there) | `tests::unlowered_forms_are_refused` |
//! | A non-finite `f32` immediate | Its bits, reinterpreted: naga 30.0.1 refuses a NaN or infinite literal | A literal | `tests::opcodes::non_finite_immediates_are_bit_casts` |
//! | Float to integer at or past 2^31 (2^32 unsigned) | Selects the maximum: naga's saturating conversion stops at the largest float below it (`naga_f2i32` gave 0x7fffff80, measured) | The interpreter saturates to the maximum | `tests::ops::conversions_and_variables_match_the_interpreter_on_the_gpu` |
//! | Shared-memory word index | `offset >> 2`, logical | `OpShiftRightArithmetic` (`emit_spirv_shared_memory.cpp`), which differs only for offsets of 2^31 and up, all out of bounds here | `tests::memory::shared_offsets_at_and_past_the_end` |
//! | `PackHalf2x16` and `UnpackHalf2x16` | Need [`Requirements::f16`] (see [`capabilities`]) | Eden needs no feature (`OpExtInst PackHalf2x16`) | `tests::opcodes::f16_packing_needs_f16` |
//! | Out-of-bounds local and shared accesses | Read 0, write dropped, word by word | Eden does not check; the interpreter refuses | `tests::memory` |

pub(crate) mod builder;
mod control;
#[cfg(test)]
mod e1;
mod memory;
mod ops;
#[cfg(test)]
mod tests;

use std::collections::{BTreeMap, BTreeSet};

use naga::valid::{Capabilities, ValidationFlags, Validator};
use naga::{
    Binding, Block, BuiltIn, Expression, FunctionArgument, FunctionResult, Handle, Interpolation,
    Literal, LocalVariable, Module, Sampling, Scalar, ShaderStage, Statement, StructMember,
    TypeInner, VectorSize,
};
use thiserror::Error;

pub use crate::host::HostCaps;
use crate::host::{
    Feature, GuestSlot, HostBuiltIn, HostCode, HostShader, HostStage, InterfaceSlot, Requirements,
    SlotBinding, SlotType,
};
use crate::ir::ATTRIBUTE_COMPONENTS as COMPONENTS;
use crate::ir::{
    Attribute, BlockId, Function, Imm, Inst, Node, Opcode, Operand, Operation, Terminator, Type,
    Value, VerifyError,
};
pub use crate::ir::{GENERIC_0_X, POSITION_X};
use crate::translate::{COLOR_TARGETS, GuestStage};
use builder::{FunctionBuilder, named_ty, push, ty};

/// The entry point's name in every module.
pub const ENTRY_POINT: &str = "main";

/// How many generic attributes there are (`attribute.h`: `NUM_GENERICS = 32`).
pub const GENERICS: u16 = 32;
/// How many locations a host stage binds: generic `n` uses location `n`, so generics from this
/// one up are refused. wgpu 30.0.1 on Metal binds vertex attributes below 31 (`wgpu-hal`
/// `metal/adapter.rs`: `max_vertex_attributes: 31`) and vertex outputs at locations up to
/// `max_inter_stage_shader_variables - 1` (`wgpu-core` `validation.rs`), which is 31 on Apple
/// GPUs (`metal/adapter.rs`: "min(124, 124 / 4)"); measured on an Apple M2. Devices with lower
/// limits (other Mac GPUs, Vulkan drivers) are the GPU crate's to check against the device. To
/// support generic 31 later, remap locations: the interface tables already keep the guest slot
/// apart from the host binding.
pub const HOST_LOCATIONS: u8 = 31;
/// The size in bytes of a `vec4<f32>`, the stride of the output structs.
const VEC4_BYTES: u32 = 16;
/// The bytes of one local- or shared-memory word: both arrays are of `u32` (Eden's
/// `DivCeil(size, 4U)`).
pub const MEMORY_WORD_BYTES: u32 = 4;
/// The most local memory a program may ask for, in bytes: 512 KiB, the CUDA programming
/// guide's "maximum amount of local memory per thread" for every compute capability from 2.0
/// (its table of technical specifications). A **guess** for the Switch 2's GPU (compute
/// capability 8.6 class), to be checked against the SPH and QMD fields' real range (E3).
pub const MAX_LOCAL_MEMORY_BYTES: u64 = 512 * 1024;
/// The most shared memory a compute program may ask for, in bytes: 99 KiB, the CUDA
/// programming guide's "maximum amount of shared memory per thread block" for compute
/// capability 8.6. A **guess** for the Switch 2's GPU; the host's own limit is wgpu's to check
/// when the pipeline is made.
pub const MAX_SHARED_MEMORY_BYTES: u32 = 99 * 1024;
/// The most invocations a workgroup may have: 1,024, the CUDA programming guide's "maximum
/// number of threads per block" for compute capability 8.6. A **guess** for the Switch 2's
/// GPU; the host's own limit is wgpu's to check when the pipeline is made.
pub const MAX_WORKGROUP_INVOCATIONS: u32 = 1024;

/// What an attribute word is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AttributeKind {
    Position { component: u32 },
    Generic { index: u8, component: u32 },
}

fn classify(attribute: Attribute) -> Option<AttributeKind> {
    let word = attribute.0;
    let component = u32::from(word % COMPONENTS);
    if (POSITION_X..POSITION_X + COMPONENTS).contains(&word) {
        return Some(AttributeKind::Position { component });
    }
    let generic = word.checked_sub(GENERIC_0_X)? / COMPONENTS;
    (generic < GENERICS).then(|| AttributeKind::Generic {
        index: u8::try_from(generic).unwrap_or(u8::MAX),
        component,
    })
}

/// Why a function was not lowered.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum LowerError {
    /// The function is malformed.
    #[error("the IR is malformed: {0}")]
    Verify(#[from] VerifyError),
    /// The stage is not handled yet.
    #[error("{0} programs are not lowered yet")]
    Stage(GuestStage),
    /// A control-flow graph of more than one block, or whose one block does not end in
    /// `return` or `unreachable`: it is to be structured first.
    #[error("a control-flow graph other than one block that returns must be structured first")]
    ControlFlow,
    /// An operation the lowering does not handle yet.
    #[error("{at}: {opcode} is not lowered yet")]
    Operation {
        /// The instruction.
        at: Value,
        /// Its opcode.
        opcode: Opcode,
    },
    /// A phi of a type the lowering has no naga type for (see the module docs, "What it
    /// takes").
    #[error("{at}: a phi of type {ty} is not lowered")]
    Phi {
        /// The phi.
        at: Value,
        /// Its type.
        ty: Type,
    },
    /// A phi names a value that is not computed, or not in scope, at the end of the block it
    /// takes it from (see the module docs, "Structured control flow").
    #[error("{at} takes {value} from {from}, where it is not in scope")]
    PhiIncoming {
        /// The phi.
        at: Value,
        /// The block it takes the value from.
        from: BlockId,
        /// The value.
        value: Value,
    },
    /// An argument the lowering needs as an immediate of a particular kind is something else.
    #[error("{at}: argument {index} is not an immediate the lowering handles")]
    Operand {
        /// The instruction.
        at: Value,
        /// The argument's index.
        index: usize,
    },
    /// An attribute the stage cannot read.
    #[error("a {stage} program cannot read attribute {attribute}")]
    Input {
        /// The stage.
        stage: GuestStage,
        /// The attribute.
        attribute: Attribute,
    },
    /// An attribute the stage cannot write.
    #[error("a {stage} program cannot write attribute {attribute}")]
    Output {
        /// The stage.
        stage: GuestStage,
        /// The attribute.
        attribute: Attribute,
    },
    /// A generic attribute with no host location (see [`HOST_LOCATIONS`]).
    #[error("generic {generic} has no host location: locations stop at {last}", last = HOST_LOCATIONS - 1)]
    GenericLocation {
        /// The generic attribute's index.
        generic: u8,
    },
    /// The lane id in a vertex program, which naga cannot express.
    #[error("the lane id cannot be read in a vertex program")]
    LaneIdInVertex,
    /// A colour output outside the targets and components that exist.
    #[error("colour output {target}.{component} does not exist")]
    ColorOutput {
        /// The target.
        target: u32,
        /// The component.
        component: u32,
    },
    /// A colour output in a stage that has none.
    #[error("a {0} program cannot write colour outputs")]
    ColorInStage(GuestStage),
    /// An operation only another stage has (a barrier or a workgroup id outside compute, a
    /// discard outside pixel programs, shared memory outside compute).
    #[error("{at}: a {stage} program cannot use {opcode}")]
    NotInStage {
        /// The instruction.
        at: Value,
        /// Its opcode.
        opcode: Opcode,
        /// The stage.
        stage: GuestStage,
    },
    /// A compute program's workgroup size is zero along an axis or has more than
    /// [`MAX_WORKGROUP_INVOCATIONS`] invocations.
    #[error(
        "a workgroup of {x} by {y} by {z} is not between 1 and {max} invocations",
        x = size[0], y = size[1], z = size[2], max = MAX_WORKGROUP_INVOCATIONS
    )]
    WorkgroupSize {
        /// The size, x, y and z.
        size: [u32; 3],
    },
    /// More local memory than [`MAX_LOCAL_MEMORY_BYTES`].
    #[error("{bytes} bytes of local memory are more than the {max} the lowering allows", max = MAX_LOCAL_MEMORY_BYTES)]
    LocalMemory {
        /// The size asked for.
        bytes: u64,
    },
    /// More shared memory than [`MAX_SHARED_MEMORY_BYTES`].
    #[error("{bytes} bytes of shared memory are more than the {max} the lowering allows", max = MAX_SHARED_MEMORY_BYTES)]
    SharedMemory {
        /// The size asked for.
        bytes: u32,
    },
    /// The shader needs a feature the host target lacks.
    #[error("the shader needs {0}, which the host target lacks")]
    Missing(Feature),
    /// naga rejected the module: a bug in this module, reported rather than handed out.
    #[error("naga rejected the module: {0}")]
    Validation(String),
    /// The lowering's own bookkeeping disagrees with the function (a bug in this module).
    #[error("internal lowering error: {0}")]
    Internal(String),
}

/// The validation flags this crate uses: every flag but the uniformity analysis.
pub fn validation_flags() -> ValidationFlags {
    ValidationFlags::all() - ValidationFlags::CONTROL_FLOW_UNIFORMITY
}

/// The naga capabilities a shader with `requirements` is validated with.
///
/// [`Requirements::f16`] grants `SHADER_FLOAT16_IN_FLOAT32` too, which naga 30.0.1 needs for
/// `pack2x16float` and `unpack2x16float` (`valid/expression.rs`), so `PackHalf2x16` and
/// `UnpackHalf2x16` count as needing f16. wgpu grants that capability from the downlevel flag
/// `SHADER_F16_IN_F32` (`wgpu-naga-bridge` 30.0.2, `features_to_naga_capabilities`), which
/// `wgpu-types` 30.0.2 documents as missing only on Mesa's Vulkan drivers without
/// `SHADER_F16`, so a device with the f16 feature has it: stricter than needed on a device
/// with the flag but not the feature, which is listed as a divergence in the module docs.
pub fn capabilities(requirements: Requirements) -> Capabilities {
    let mut caps = Capabilities::empty();
    caps.set(Capabilities::SUBGROUP, requirements.subgroup);
    caps.set(
        Capabilities::SHADER_FLOAT16 | Capabilities::SHADER_FLOAT16_IN_FLOAT32,
        requirements.f16,
    );
    caps.set(Capabilities::FLOAT64, requirements.f64);
    caps.set(Capabilities::SHADER_INT64, requirements.i64);
    caps.set(
        Capabilities::SHADER_INT64_ATOMIC_ALL_OPS,
        requirements.i64_atomics,
    );
    caps
}

/// What the lowering needs beyond the IR (see the module docs, "The entry points").
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LowerInputs {
    /// A compute program's workgroup size, x, y and z (the variant key's); ignored for other
    /// stages.
    pub workgroup_size: [u32; 3],
    /// The local-memory size in bytes: the SPH's for a graphics program, the QMD's for a
    /// compute program.
    pub local_memory_bytes: u64,
    /// A compute program's shared-memory size in bytes (the variant key's, from the QMD);
    /// ignored for other stages.
    pub shared_memory_bytes: u32,
    /// The features of the host target.
    pub caps: HostCaps,
}

impl LowerInputs {
    /// A workgroup of one invocation, no local or shared memory, and a target with every
    /// feature: what [`lower`] uses.
    pub const DEFAULT: Self = Self {
        workgroup_size: [1, 1, 1],
        local_memory_bytes: 0,
        shared_memory_bytes: 0,
        caps: HostCaps::ALL,
    };
}

/// The location of generic `generic`, if a host stage can bind it.
fn host_location(generic: u8) -> Result<u8, LowerError> {
    if generic < HOST_LOCATIONS {
        Ok(generic)
    } else {
        Err(LowerError::GenericLocation { generic })
    }
}

/// A variable of the guest's state, which becomes a local variable (see the module docs, "What
/// it takes"). Declared in this order, so registers come first, as in slice 1.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Var {
    Reg(u8),
    Pred(u8),
    UniformReg(u8),
    UniformPred(u8),
    Goto(u32),
    IndirectBranch,
    ZeroFlag,
    SignFlag,
    CarryFlag,
    OverflowFlag,
}

impl Var {
    /// The variable's name in the module and whether it is a boolean (else a `u32`).
    fn name(self) -> (String, bool) {
        match self {
            Self::Reg(index) => (format!("r{index}"), false),
            Self::Pred(index) => (format!("p{index}"), true),
            Self::UniformReg(index) => (format!("ur{index}"), false),
            Self::UniformPred(index) => (format!("up{index}"), true),
            Self::Goto(index) => (format!("goto{index}"), true),
            Self::IndirectBranch => ("indirect_branch".into(), false),
            Self::ZeroFlag => ("zero_flag".into(), true),
            Self::SignFlag => ("sign_flag".into(), true),
            Self::CarryFlag => ("carry_flag".into(), true),
            Self::OverflowFlag => ("overflow_flag".into(), true),
        }
    }

    /// The variable a get or set operation names, `Ok(None)` for any other operation.
    fn of(at: Value, op: &Operation) -> Result<Option<Self>, LowerError> {
        let name = || imm_arg(op, 0);
        let refused = LowerError::Operand { at, index: 0 };
        let var = match op.opcode {
            Opcode::GetRegister | Opcode::SetRegister => match name() {
                Some(Imm::Reg(reg)) => Self::Reg(reg.index()),
                _ => return Err(refused),
            },
            Opcode::GetPred | Opcode::SetPred => match name() {
                Some(Imm::Pred(pred)) => Self::Pred(pred.index()),
                _ => return Err(refused),
            },
            Opcode::GetUniformRegister | Opcode::SetUniformRegister => match name() {
                Some(Imm::UniformReg(reg)) => Self::UniformReg(reg.index()),
                _ => return Err(refused),
            },
            Opcode::GetUniformPred | Opcode::SetUniformPred => match name() {
                Some(Imm::UniformPred(pred)) => Self::UniformPred(pred.index()),
                _ => return Err(refused),
            },
            Opcode::GetGotoVariable | Opcode::SetGotoVariable => match name() {
                Some(Imm::U32(index)) => Self::Goto(index),
                _ => return Err(refused),
            },
            Opcode::GetIndirectBranchVariable | Opcode::SetIndirectBranchVariable => {
                Self::IndirectBranch
            }
            Opcode::GetZFlag | Opcode::SetZFlag => Self::ZeroFlag,
            Opcode::GetSFlag | Opcode::SetSFlag => Self::SignFlag,
            Opcode::GetCFlag | Opcode::SetCFlag => Self::CarryFlag,
            Opcode::GetOFlag | Opcode::SetOFlag => Self::OverflowFlag,
            _ => return Ok(None),
        };
        Ok(Some(var))
    }
}

/// What a function reads and writes, collected before the module is built.
#[derive(Default)]
struct Interface {
    vars: BTreeSet<Var>,
    inputs: BTreeSet<u8>,
    generic_outputs: BTreeSet<u8>,
    targets: BTreeSet<u8>,
    system: SystemValues,
    memory: MemoryUse,
    requirements: Requirements,
}

/// The system values a function reads that become built-in arguments.
#[derive(Clone, Copy, Default)]
struct SystemValues {
    lane_id: bool,
    workgroup_id: bool,
    local_invocation_id: bool,
}

/// The memory spaces a function uses.
#[derive(Clone, Copy, Default)]
struct MemoryUse {
    local: bool,
    shared: bool,
}

/// Which stages may use `opcode`, when not every stage may.
fn stage_allows(opcode: Opcode, stage: GuestStage) -> bool {
    let compute = stage == GuestStage::Compute;
    match opcode {
        Opcode::Barrier
        | Opcode::WorkgroupMemoryBarrier
        | Opcode::DeviceMemoryBarrier
        | Opcode::WorkgroupId
        | Opcode::LocalInvocationId => compute,
        Opcode::DemoteToHelperInvocation => stage == GuestStage::Pixel,
        _ if memory::is_shared(opcode) => compute,
        _ => true,
    }
}

/// Whether `ty` is a 16-bit float type, which needs [`Requirements::f16`].
const fn is_f16(ty: Type) -> bool {
    matches!(ty, Type::F16 | Type::F16x2 | Type::F16x3 | Type::F16x4)
}

impl Interface {
    fn collect(stage: GuestStage, function: &Function) -> Result<Self, LowerError> {
        let vertex = stage == GuestStage::Vertex;
        let mut found = Self::default();
        for (at, inst) in function.listed() {
            let op = match inst {
                Inst::Op(op) => op,
                Inst::Phi(phi) => {
                    if naga_inner(phi.ty).is_none() {
                        return Err(LowerError::Phi { at, ty: phi.ty });
                    }
                    found.requirements.f16 |= is_f16(phi.ty);
                    found.requirements.i64 |= phi.ty == Type::U64;
                    continue;
                }
            };
            if let Some(var) = Var::of(at, op)? {
                found.vars.insert(var);
                continue;
            }
            if !stage_allows(op.opcode, stage) {
                return Err(LowerError::NotInStage {
                    at,
                    opcode: op.opcode,
                    stage,
                });
            }
            let result = [op.opcode.result()];
            for &ty in op.opcode.args().iter().chain(&result) {
                found.requirements.f16 |= is_f16(ty);
                found.requirements.i64 |= ty == Type::U64;
            }
            // naga's `pack2x16float` and `unpack2x16float` need `SHADER_FLOAT16_IN_FLOAT32`,
            // which counts as part of the f16 requirement (see [`capabilities`]).
            found.requirements.f16 |=
                matches!(op.opcode, Opcode::PackHalf2x16 | Opcode::UnpackHalf2x16);
            found.memory.local |= memory::is_local(op.opcode);
            found.memory.shared |= memory::is_shared(op.opcode);
            match op.opcode {
                Opcode::GetAttribute => match classify(attribute_arg(at, op, 1)?) {
                    Some(AttributeKind::Generic { index, .. }) if vertex => {
                        found.inputs.insert(host_location(index)?);
                    }
                    _ => {
                        let attribute = attribute_arg(at, op, 1)?;
                        return Err(LowerError::Input { stage, attribute });
                    }
                },
                Opcode::SetAttribute => match classify(attribute_arg(at, op, 2)?) {
                    Some(AttributeKind::Generic { index, .. }) if vertex => {
                        found.generic_outputs.insert(host_location(index)?);
                    }
                    Some(AttributeKind::Position { .. }) if vertex => {}
                    _ => {
                        let attribute = attribute_arg(at, op, 2)?;
                        return Err(LowerError::Output { stage, attribute });
                    }
                },
                Opcode::LaneId if vertex => return Err(LowerError::LaneIdInVertex),
                Opcode::LaneId => {
                    found.system.lane_id = true;
                    found.requirements.subgroup = true;
                }
                Opcode::WorkgroupId => found.system.workgroup_id = true,
                Opcode::LocalInvocationId => found.system.local_invocation_id = true,
                Opcode::SetFragColor => {
                    let Some(Imm::U32(target)) = imm_arg(op, 0) else {
                        return Err(LowerError::Operand { at, index: 0 });
                    };
                    let Some(Imm::U32(component)) = imm_arg(op, 1) else {
                        return Err(LowerError::Operand { at, index: 1 });
                    };
                    if stage != GuestStage::Pixel {
                        return Err(LowerError::ColorInStage(stage));
                    }
                    let target = u8::try_from(target)
                        .ok()
                        .filter(|&t| usize::from(t) < COLOR_TARGETS)
                        .ok_or(LowerError::ColorOutput { target, component })?;
                    if component >= u32::from(COMPONENTS) {
                        return Err(LowerError::ColorOutput {
                            target: u32::from(target),
                            component,
                        });
                    }
                    found.targets.insert(target);
                }
                _ => {}
            }
        }
        Ok(found)
    }
}

/// A location binding; varyings (a vertex program's generic outputs, the only ones slice 1
/// lowers) get perspective interpolation at the pixel centre, slice 1's guess. The modes slice
/// 15 collects ([`crate::info::ShaderInfo::interpolation`]) replace it in slice 17.
fn location(location: u8, varying: bool) -> Binding {
    Binding::Location {
        location: u32::from(location),
        interpolation: varying.then_some(Interpolation::Perspective),
        sampling: varying.then_some(Sampling::Center),
        blend_src: None,
        per_primitive: false,
    }
}

/// The member of the output struct a write lands in.
#[derive(Clone, Copy)]
struct OutputMember {
    index: u32,
}

/// Lowers `function` of `stage` to a validated naga module with [`LowerInputs::DEFAULT`] (see
/// the module docs).
pub fn lower(stage: GuestStage, function: &Function) -> Result<HostShader, LowerError> {
    lower_with(stage, function, &LowerInputs::DEFAULT)
}

/// Checks the inputs that apply to `stage` (see the module docs, "Checks and their order",
/// step 3).
fn check_inputs(stage: GuestStage, inputs: &LowerInputs) -> Result<(), LowerError> {
    let compute = stage == GuestStage::Compute;
    if compute {
        let size = inputs.workgroup_size;
        let invocations = size
            .iter()
            .try_fold(1_u32, |product, &axis| product.checked_mul(axis));
        if size.contains(&0) || invocations.is_none_or(|n| n > MAX_WORKGROUP_INVOCATIONS) {
            return Err(LowerError::WorkgroupSize { size });
        }
    }
    if inputs.local_memory_bytes > MAX_LOCAL_MEMORY_BYTES {
        return Err(LowerError::LocalMemory {
            bytes: inputs.local_memory_bytes,
        });
    }
    if compute && inputs.shared_memory_bytes > MAX_SHARED_MEMORY_BYTES {
        return Err(LowerError::SharedMemory {
            bytes: inputs.shared_memory_bytes,
        });
    }
    Ok(())
}

/// The tree a function runs: its own, or, for a control-flow graph of one block that ends in
/// `return` or `unreachable`, that block and the node of the same name.
fn tree(function: &Function) -> Result<Vec<Node>, LowerError> {
    if let Some(nodes) = function.structure() {
        return Ok(nodes.to_vec());
    }
    let entry = function.entry();
    let end = match function.block(entry).map(|block| block.terminator) {
        Some(Terminator::Return) if function.block_count() == 1 => Node::Return,
        Some(Terminator::Unreachable) if function.block_count() == 1 => Node::Unreachable,
        _ => return Err(LowerError::ControlFlow),
    };
    Ok(vec![Node::Block(entry), end])
}

/// Lowers `function` of `stage` to a validated naga module (see the module docs).
pub fn lower_with(
    stage: GuestStage,
    function: &Function,
    inputs: &LowerInputs,
) -> Result<HostShader, LowerError> {
    function.verify()?;
    let (host_stage, shader_stage) = match stage {
        GuestStage::Vertex => (HostStage::Vertex, ShaderStage::Vertex),
        GuestStage::Pixel => (HostStage::Fragment, ShaderStage::Fragment),
        GuestStage::Compute => (HostStage::Compute, ShaderStage::Compute),
        other => return Err(LowerError::Stage(other)),
    };
    check_inputs(stage, inputs)?;
    let nodes = tree(function)?;
    let interface = Interface::collect(stage, function)?;
    let requirements = interface.requirements;

    let mut module = Module::default();
    let types = Types {
        u32: ty(&mut module, TypeInner::Scalar(Scalar::U32)),
        vec4: ty(
            &mut module,
            TypeInner::Vector {
                size: VectorSize::Quad,
                scalar: Scalar::F32,
            },
        ),
    };
    let mut lowering = Lowering::new(FunctionBuilder::new(ENTRY_POINT), module, types);
    let interface_inputs = lowering.declare_inputs(&interface);
    let outputs = lowering.declare_outputs(stage, &interface);
    for &var in &interface.vars {
        let (name, boolean) = var.name();
        let var_ty = if boolean {
            lowering.ir_type(Type::U1)?
        } else {
            types.u32
        };
        let local = lowering.fb.local(&name, var_ty);
        lowering.vars.insert(var, local);
    }
    lowering.declare_memory(&interface, inputs)?;
    lowering.values = vec![None; function.insts().count()];
    lowering.declare_phis(function)?;

    let mut body = Block::new();
    lowering.nodes(function, &nodes, &mut body)?;
    if !matches!(nodes.last(), Some(Node::Return | Node::Unreachable)) {
        lowering.ret(&mut body);
    }
    if let Some(feature) = requirements.missing(inputs.caps) {
        return Err(LowerError::Missing(feature));
    }
    let Lowering { fb, mut module, .. } = lowering;
    let compute = stage == GuestStage::Compute;
    let workgroup_size = compute.then_some(inputs.workgroup_size);
    module.entry_points.push(naga::EntryPoint {
        name: ENTRY_POINT.into(),
        stage: shader_stage,
        early_depth_test: None,
        workgroup_size: workgroup_size.unwrap_or([0, 0, 0]),
        workgroup_size_overrides: None,
        function: fb.finish(body),
        mesh_info: None,
        task_payload: None,
        incoming_ray_payload: None,
    });

    let info = Validator::new(validation_flags(), capabilities(requirements))
        .validate(&module)
        .map_err(|error| LowerError::Validation(error_chain(&error)))?;
    Ok(HostShader {
        stage: host_stage,
        code: HostCode::Naga {
            module: Box::new(module),
            info: Box::new(info),
        },
        entry: ENTRY_POINT.into(),
        inputs: interface_inputs,
        outputs,
        requirements,
        workgroup_size,
    })
}

/// The types every module uses.
#[derive(Clone, Copy)]
struct Types {
    u32: Handle<naga::Type>,
    vec4: Handle<naga::Type>,
}

/// The naga type of an IR value type the lowering handles.
fn naga_inner(ty: Type) -> Option<TypeInner> {
    let vector = |size, scalar| Some(TypeInner::Vector { size, scalar });
    match ty {
        Type::U1 => Some(TypeInner::Scalar(Scalar::BOOL)),
        Type::U32 => Some(TypeInner::Scalar(Scalar::U32)),
        Type::U64 => Some(TypeInner::Scalar(Scalar::U64)),
        Type::F16 => Some(TypeInner::Scalar(Scalar::F16)),
        Type::F32 => Some(TypeInner::Scalar(Scalar::F32)),
        Type::U32x2 => vector(VectorSize::Bi, Scalar::U32),
        Type::U32x3 => vector(VectorSize::Tri, Scalar::U32),
        Type::U32x4 => vector(VectorSize::Quad, Scalar::U32),
        Type::F16x2 => vector(VectorSize::Bi, Scalar::F16),
        Type::F16x3 => vector(VectorSize::Tri, Scalar::F16),
        Type::F16x4 => vector(VectorSize::Quad, Scalar::F16),
        Type::F32x2 => vector(VectorSize::Bi, Scalar::F32),
        Type::F32x3 => vector(VectorSize::Tri, Scalar::F32),
        Type::F32x4 => vector(VectorSize::Quad, Scalar::F32),
        _ => None,
    }
}

fn arg_index(fb: &FunctionBuilder) -> u32 {
    u32::try_from(fb.function.arguments.len()).unwrap_or(u32::MAX)
}

/// The state of lowering one function.
struct Lowering {
    fb: FunctionBuilder,
    module: Module,
    types: Types,
    /// Each IR value's expression, by value index.
    values: Vec<Option<Handle<Expression>>>,
    vars: BTreeMap<Var, Handle<LocalVariable>>,
    /// Each phi's local variable.
    phis: BTreeMap<Value, Handle<LocalVariable>>,
    /// The stores at the end of each block: the phi, its local and the incoming value.
    phi_stores: BTreeMap<BlockId, Vec<(Value, Handle<LocalVariable>, Operand)>>,
    /// The values lowered, in order, so a body's can leave scope when it ends.
    scope_log: Vec<Value>,
    input_args: BTreeMap<u8, u32>,
    lane_arg: Option<u32>,
    workgroup_id_arg: Option<u32>,
    local_invocation_id_arg: Option<u32>,
    output: Option<Handle<LocalVariable>>,
    members: Vec<StructMember>,
    position: Option<OutputMember>,
    generic_members: BTreeMap<u8, OutputMember>,
    target_members: BTreeMap<u8, OutputMember>,
    memory: memory::Memory,
}

impl Lowering {
    fn new(fb: FunctionBuilder, module: Module, types: Types) -> Self {
        Self {
            fb,
            module,
            types,
            values: Vec::new(),
            vars: BTreeMap::new(),
            phis: BTreeMap::new(),
            phi_stores: BTreeMap::new(),
            scope_log: Vec::new(),
            input_args: BTreeMap::new(),
            lane_arg: None,
            workgroup_id_arg: None,
            local_invocation_id_arg: None,
            output: None,
            members: Vec::new(),
            position: None,
            generic_members: BTreeMap::new(),
            target_members: BTreeMap::new(),
            memory: memory::Memory::default(),
        }
    }

    /// The naga type of IR type `ir`.
    fn ir_type(&mut self, ir: Type) -> Result<Handle<naga::Type>, LowerError> {
        let inner =
            naga_inner(ir).ok_or_else(|| LowerError::Internal(format!("no naga type for {ir}")))?;
        Ok(ty(&mut self.module, inner))
    }

    /// Adds a built-in argument and returns its index.
    fn builtin_arg(&mut self, name: &str, ty: Handle<naga::Type>, built_in: BuiltIn) -> u32 {
        let index = arg_index(&self.fb);
        self.fb.function.arguments.push(FunctionArgument {
            name: Some(name.into()),
            ty,
            binding: Some(Binding::BuiltIn(built_in)),
        });
        index
    }

    /// Declares the entry point's arguments: generic inputs by location, then the lane id, then
    /// the compute built-ins (which are not interface slots).
    fn declare_inputs(&mut self, interface: &Interface) -> Vec<InterfaceSlot> {
        let types = self.types;
        let mut inputs = Vec::new();
        for &generic in &interface.inputs {
            self.input_args.insert(generic, arg_index(&self.fb));
            self.fb.function.arguments.push(FunctionArgument {
                name: Some(format!("in{generic}")),
                ty: types.vec4,
                binding: Some(location(generic, false)),
            });
            inputs.push(InterfaceSlot {
                binding: SlotBinding::Location(u32::from(generic)),
                ty: SlotType::F32x4,
                guest: GuestSlot::Generic(generic),
            });
        }
        let system = interface.system;
        if system.lane_id {
            let lane = self.builtin_arg("lane_id", types.u32, BuiltIn::SubgroupInvocationId);
            self.lane_arg = Some(lane);
            inputs.push(InterfaceSlot {
                binding: SlotBinding::BuiltIn(HostBuiltIn::SubgroupInvocationId),
                ty: SlotType::U32,
                guest: GuestSlot::LaneId,
            });
        }
        if system.workgroup_id || system.local_invocation_id {
            let vec3 = ty(
                &mut self.module,
                TypeInner::Vector {
                    size: VectorSize::Tri,
                    scalar: Scalar::U32,
                },
            );
            if system.workgroup_id {
                let arg = self.builtin_arg("workgroup_id", vec3, BuiltIn::WorkGroupId);
                self.workgroup_id_arg = Some(arg);
            }
            if system.local_invocation_id {
                let arg = self.builtin_arg("local_id", vec3, BuiltIn::LocalInvocationId);
                self.local_invocation_id_arg = Some(arg);
            }
        }
        inputs
    }

    /// Adds a `vec4<f32>` member to the output struct.
    fn member(&mut self, name: String, binding: Binding) -> OutputMember {
        let index = u32::try_from(self.members.len()).unwrap_or(u32::MAX);
        self.members.push(StructMember {
            name: Some(name),
            ty: self.types.vec4,
            binding: Some(binding),
            offset: index.saturating_mul(VEC4_BYTES),
        });
        OutputMember { index }
    }

    /// Declares the output struct (built-ins first, then locations ascending) and the local
    /// variable the body writes it through.
    fn declare_outputs(&mut self, stage: GuestStage, interface: &Interface) -> Vec<InterfaceSlot> {
        let mut outputs = Vec::new();
        if stage == GuestStage::Vertex {
            let position = Binding::BuiltIn(BuiltIn::Position { invariant: false });
            self.position = Some(self.member("position".into(), position));
            outputs.push(InterfaceSlot {
                binding: SlotBinding::BuiltIn(HostBuiltIn::Position),
                ty: SlotType::F32x4,
                guest: GuestSlot::Position,
            });
        }
        for &generic in &interface.generic_outputs {
            let member = self.member(format!("out{generic}"), location(generic, true));
            self.generic_members.insert(generic, member);
            outputs.push(InterfaceSlot {
                binding: SlotBinding::Location(u32::from(generic)),
                ty: SlotType::F32x4,
                guest: GuestSlot::Generic(generic),
            });
        }
        for &target in &interface.targets {
            let member = self.member(format!("color{target}"), location(target, false));
            self.target_members.insert(target, member);
            outputs.push(InterfaceSlot {
                binding: SlotBinding::Location(u32::from(target)),
                ty: SlotType::F32x4,
                guest: GuestSlot::RenderTarget(target),
            });
        }
        if !self.members.is_empty() {
            let members = std::mem::take(&mut self.members);
            let span = u32::try_from(members.len())
                .unwrap_or(u32::MAX)
                .saturating_mul(VEC4_BYTES);
            let struct_ty = named_ty(
                &mut self.module,
                "Output",
                TypeInner::Struct { members, span },
            );
            self.fb.function.result = Some(FunctionResult {
                ty: struct_ty,
                binding: None,
            });
            self.output = Some(self.fb.local("out", struct_ty));
        }
        outputs
    }

    /// Appends `expression` to `body` (see `builder`).
    fn expr(&mut self, body: &mut Block, expression: Expression) -> Handle<Expression> {
        self.fb.expr(body, expression)
    }

    /// The expression of an operand; the verifier has checked it is defined earlier.
    fn value(&self, value: Value) -> Result<Handle<Expression>, LowerError> {
        usize::try_from(value.index())
            .ok()
            .and_then(|slot| self.values.get(slot).copied().flatten())
            .ok_or(LowerError::Verify(VerifyError::Missing(value)))
    }

    /// The expression of an operand of instruction `at`: a value's, or a literal made here for
    /// an immediate (where slice 1's IR had an `Imm32` instruction, so the module's
    /// expressions keep their order); a NaN or infinite `f32` is its bits reinterpreted, since
    /// naga refuses such literals. `index` names the argument in the error for an immediate
    /// the lowering has no literal for.
    fn operand(
        &mut self,
        body: &mut Block,
        at: Value,
        index: usize,
        operand: Operand,
    ) -> Result<Handle<Expression>, LowerError> {
        let literal = match operand {
            Operand::Value(value) => return self.value(value),
            Operand::Imm(Imm::U1(value)) => Literal::Bool(value),
            Operand::Imm(Imm::U32(bits)) => Literal::U32(bits),
            // naga refuses a NaN or infinite float literal ("Float literal is NaN"), so a
            // non-finite immediate is its bits, reinterpreted.
            Operand::Imm(Imm::F32(bits)) if !f32::from_bits(bits).is_finite() => {
                let bits = self.expr(body, Expression::Literal(Literal::U32(bits)));
                return Ok(self.expr(
                    body,
                    Expression::As {
                        expr: bits,
                        kind: naga::ScalarKind::Float,
                        convert: None,
                    },
                ));
            }
            Operand::Imm(Imm::F32(bits)) => Literal::F32(f32::from_bits(bits)),
            Operand::Imm(Imm::U64(bits)) => Literal::U64(bits),
            Operand::Imm(_) => return Err(LowerError::Operand { at, index }),
        };
        Ok(self.expr(body, Expression::Literal(literal)))
    }

    /// Argument `index` of `op`, lowered.
    fn arg(
        &mut self,
        body: &mut Block,
        at: Value,
        op: &Operation,
        index: usize,
    ) -> Result<Handle<Expression>, LowerError> {
        let operand = Self::raw_arg(at, op, index)?;
        self.operand(body, at, index, operand)
    }

    /// Argument `index` of `op`, as it is.
    fn raw_arg(at: Value, op: &Operation, index: usize) -> Result<Operand, LowerError> {
        op.args
            .get(index)
            .copied()
            .ok_or(LowerError::Operand { at, index })
    }

    /// A pointer to the local variable of `var`.
    fn var(&mut self, body: &mut Block, var: Var) -> Result<Handle<Expression>, LowerError> {
        let local = *self
            .vars
            .get(&var)
            .ok_or_else(|| LowerError::Internal(format!("no variable for {}", var.name().0)))?;
        Ok(self.expr(body, Expression::LocalVariable(local)))
    }

    /// Stores `value` to component `component` of output member `member`.
    fn store_output(
        &mut self,
        body: &mut Block,
        at: Value,
        member: Option<OutputMember>,
        component: u32,
        value: (usize, Operand),
    ) -> Result<(), LowerError> {
        let value = self.operand(body, at, value.0, value.1)?;
        let (Some(output), Some(member)) = (self.output, member) else {
            return Err(LowerError::Internal("no output member".into()));
        };
        let base = self.expr(body, Expression::LocalVariable(output));
        let member = self.expr(
            body,
            Expression::AccessIndex {
                base,
                index: member.index,
            },
        );
        let pointer = self.expr(
            body,
            Expression::AccessIndex {
                base: member,
                index: component,
            },
        );
        push(body, Statement::Store { pointer, value });
        Ok(())
    }

    /// Declares a local variable for each phi and records the stores each block ends with.
    fn declare_phis(&mut self, function: &Function) -> Result<(), LowerError> {
        for (at, inst) in function.listed() {
            let Inst::Phi(phi) = inst else { continue };
            let ty = self.ir_type(phi.ty)?;
            let local = self.fb.local(&format!("phi{}", at.index()), ty);
            self.phis.insert(at, local);
            for &(from, operand) in &phi.incoming {
                self.phi_stores
                    .entry(from)
                    .or_default()
                    .push((at, local, operand));
            }
        }
        Ok(())
    }

    /// Loads phi `at`'s local variable as its value.
    pub(super) fn phi_load(&mut self, body: &mut Block, at: Value) -> Result<(), LowerError> {
        let local = *self
            .phis
            .get(&at)
            .ok_or_else(|| LowerError::Internal(format!("no variable for {at}")))?;
        let pointer = self.expr(body, Expression::LocalVariable(local));
        let value = self.expr(body, Expression::Load { pointer });
        self.record(at, value)
    }

    /// Stores, at the end of `block`, each value a phi takes from it.
    pub(super) fn phi_stores(
        &mut self,
        body: &mut Block,
        block: BlockId,
    ) -> Result<(), LowerError> {
        let Some(stores) = self.phi_stores.get(&block).cloned() else {
            return Ok(());
        };
        for (at, local, operand) in stores {
            let value = match operand {
                Operand::Value(value) => {
                    self.value(value).map_err(|_| LowerError::PhiIncoming {
                        at,
                        from: block,
                        value,
                    })?
                }
                Operand::Imm(_) => self.operand(body, at, 0, operand)?,
            };
            let pointer = self.expr(body, Expression::LocalVariable(local));
            push(body, Statement::Store { pointer, value });
        }
        Ok(())
    }

    /// Records the expression of instruction `at`, which is in scope until the body that
    /// holds it ends ([`Lowering::end_scope`]).
    pub(super) fn record(&mut self, at: Value, expr: Handle<Expression>) -> Result<(), LowerError> {
        let slot = usize::try_from(at.index())
            .ok()
            .and_then(|slot| self.values.get_mut(slot))
            .ok_or_else(|| LowerError::Internal(format!("{at} is outside the arena")))?;
        *slot = Some(expr);
        self.scope_log.push(at);
        Ok(())
    }

    /// Takes the values recorded since `mark` out of scope (an `if` or loop body has ended).
    pub(super) fn end_scope(&mut self, mark: usize) {
        let ended: Vec<Value> = self.scope_log.drain(mark..).collect();
        for value in ended {
            if let Some(slot) = usize::try_from(value.index())
                .ok()
                .and_then(|slot| self.values.get_mut(slot))
            {
                *slot = None;
            }
        }
    }

    /// Ends the invocation: returns the output struct, or nothing.
    fn ret(&mut self, body: &mut Block) {
        let value = self.output.map(|output| {
            let pointer = self.expr(body, Expression::LocalVariable(output));
            self.expr(body, Expression::Load { pointer })
        });
        push(body, Statement::Return { value });
    }
}

/// `error` and each of its sources, joined by ": ", so naga's nested validation errors read in
/// full.
fn error_chain(error: &dyn std::error::Error) -> String {
    let mut text = error.to_string();
    let mut source = error.source();
    while let Some(inner) = source {
        text.push_str(": ");
        text.push_str(&inner.to_string());
        source = inner.source();
    }
    text
}

/// Immediate argument `index` of `op`, if it is one.
fn imm_arg(op: &Operation, index: usize) -> Option<Imm> {
    op.args.get(index)?.imm()
}

/// The attribute an attribute access names (argument 0), after checking that its vertex index
/// (argument `vertex`) is the immediate zero: the stages lowered here have one vertex.
fn attribute_arg(at: Value, op: &Operation, vertex: usize) -> Result<Attribute, LowerError> {
    if imm_arg(op, vertex) != Some(Imm::U32(0)) {
        return Err(LowerError::Operand { at, index: vertex });
    }
    match imm_arg(op, 0) {
        Some(Imm::Attribute(attribute)) => Ok(attribute),
        _ => Err(LowerError::Operand { at, index: 0 }),
    }
}
