//! The IR interpreter: the test oracle for the translators and passes (slice 7 of the spec's
//! port slices; owner's decision 15, 2026-11-12).
//!
//! [`Machine::run`] runs one invocation of an IR [`Function`] on the machine's state and inputs
//! and leaves the outputs and the final state in the machine; [`Finished`] says how many steps
//! the run took and holds every value it computed. It runs all three forms the IR takes (spec,
//! "The intermediate representation"):
//!
//! - a **control-flow graph** before structuring, from the entry block along each block's
//!   [`Terminator`], with guest registers, predicates, uniform registers and predicates, the
//!   condition-code flags and the goto and indirect-branch variables read and written through
//!   their get and set opcodes;
//! - a **structured** function, along its tree of [`Node`]s;
//! - **SSA form** in either: a block's phis take, all at once, the values that arrive from the
//!   block that ran before it (in a control-flow graph its predecessor; in a structured
//!   function the block the tree ran last), so a pass can be checked by running its input and
//!   its output on the same machine.
//!
//! The function is verified first ([`Function::verify`]); a malformed one is
//! [`InterpError::Invalid`]. The approach follows PR #4588's tests, which evaluate Eden's IR
//! for a vertex program (`eden-pr4588/src/tests/video_core/native_shader.cpp`, `EvaluateVertex`
//! and `ExecuteNativeWarp`); it is a reference for the approach, not for code.
//!
//! # Semantics
//!
//! Each operation follows Eden's definition of its opcode (`opcodes.inc`, the table in
//! `ir::Opcode`), which Eden gives by the code its SPIR-V emitter writes
//! (`eden/src/shader_recompiler/backend/spirv/emit_spirv_*.cpp` @ cf654052, GPL-3.0-or-later):
//! each group below names its file. Where that code is undefined for some inputs (SPIR-V leaves
//! a shift past the width, a bit field past bit 31, a division by zero and an out-of-range
//! float-to-integer conversion undefined), the interpreter gives a defined result rather than
//! an error, because translators compute such values and then discard them with a select
//! (Eden's `F2I` translator converts a NaN and selects a constant instead,
//! `frontend/maxwell/translate/impl/floating_point_conversion_integer.cpp`); each choice is
//! listed below and pinned by a test in `tests`.
//!
//! - **Context** (`emit_spirv_context_get_set.cpp`). Registers, predicates, uniform registers
//!   and predicates, the flags (`GetZFlag` and the rest), goto variables and the indirect
//!   branch variable are state in the [`Machine`]; Eden's backend never sees them (SSA removes
//!   them), and the interpreter reads an unwritten flag or goto variable as false and the
//!   unwritten indirect branch variable as 0. `GetCbuf*` reads constant buffer bytes,
//!   little-endian, at the offset aligned down to the access size (Eden's `GetCbuf`: "Hardware
//!   been proved to read the aligned offset"). `GetAttribute`, `GetAttributeU32` and
//!   `GetAttributeIndexed` read [`Machine::attributes`] (an absent one is
//!   [`InterpError::MissingInput`]; the indexed form names the word `offset >> 2`, as Eden's
//!   `make_load` does), and the set forms write [`Machine::outputs`]. `InterpolateAtOffset`, a
//!   starquake opcode, reads [`Machine::interpolated`] by word and offset bits (an absent one
//!   is [`InterpError::MissingInput`]). `GetPatch` and `SetPatch`
//!   share [`Machine::patches`] (Eden reads a control program's output patch and an evaluation
//!   program's input patch; one map serves both). `SetFragColor`, `SetFragDepth` and
//!   `SetSampleMask` write their fields. System values come from [`SystemValues`];
//!   `IsHelperInvocation` is also true once the program has run
//!   `DemoteToHelperInvocation`, which sets [`Machine::demoted`] and lets the program continue
//!   (`OpDemoteToHelperInvocationEXT`, `emit_spirv_control_flow.cpp`).
//! - **Undefined values** (`emit_spirv_undefined.cpp`): `UndefU1` and its siblings give zero,
//!   which is what an unwritten register, predicate or flag reads before SSA, so a function
//!   gives the same result before and after the SSA rewrite turns such reads into undefined
//!   values (Eden's `ssa_rewrite_pass.cpp`, `ReadVariable`).
//! - **No-ops**: `Void`, `Reference`, `Prologue`, `Epilogue` (Eden's are host fix-ups,
//!   `emit_spirv_special.cpp`), and the barriers, which one invocation has nothing to wait for.
//!   `Identity` and `ConditionRef` give their argument.
//! - **Memory** (`emit_spirv_memory.cpp`, `emit_spirv_shared_memory.cpp`,
//!   `emit_spirv_atomic.cpp`). Global (a [`Region`] at a 64-bit address), storage (one byte
//!   buffer per binding), shared (bytes) and local memory (32-bit words, indexed by word as
//!   Eden's `EmitLoadLocal` does). Byte-addressed accesses are little-endian. An access is
//!   checked for alignment to its size first ([`InterpError::Misaligned`]; the hardware faults,
//!   so a test relying on one is wrong), then for bounds ([`InterpError::OutOfBounds`]). 8- and
//!   16-bit loads zero- or sign-extend as their names say. The 32-bit integer atomics of the
//!   global, storage and shared spaces return the old value; `Inc` and `Dec` wrap as Eden's
//!   `CasFunction` does (`spirv_emit_context.cpp`: increment gives 0 when `old >= value`,
//!   decrement gives `value` when `old == 0 || old > value`).
//! - **Composites, selects, bit casts and packing** (`emit_spirv_composite.cpp`,
//!   `emit_spirv_select.cpp`, `emit_spirv_bitwise_conversion.cpp`): component 0 is the low
//!   half when packing; `PackHalf2x16` rounds to nearest even. An extract or insert index past
//!   the vector is [`InterpError::Component`].
//! - **Integers** (`emit_spirv_integer.cpp`, `emit_spirv_logical.cpp`): wrapping arithmetic.
//!   Undefined in SPIR-V, defined here: a logical shift by the width or more gives 0 (as PR
//!   #4588's `ExecuteNativeWarp` computes `ShiftLeftLogical32` and `ShiftRightLogical32`); an
//!   arithmetic shift by the width or more fills with the sign bit (a guess); `UDiv32` by zero
//!   gives `0xffff_ffff` and `SDiv32` by zero gives `-1` (guesses), and `SDiv32` of `i32::MIN` by
//!   `-1` wraps to `i32::MIN`; `IAbs` of the minimum wraps to itself; a bit field's offset and
//!   count are clamped to 32 and the field reads zeros above bit 31 (`BitFieldUExtract`), sign
//!   extends from the field's top bit (`BitFieldSExtract`) and inserts nothing past bit 31
//!   (`BitFieldInsert`); `SClamp32` and `UClamp32` are `min(max(value, low), high)`.
//!   `FindSMsb32` and `FindUMsb32` follow GLSL.std.450 (`-1` for no bit).
//! - **Floating point** (`emit_spirv_floating_point.cpp`): see the `float` module, which states
//!   the rounding, denormal and NaN rules and the one host-dependent group (transcendentals).
//!   Rounding modes and flush-to-zero are modelled as the IR defines them, on arithmetic and on
//!   conversions alike: owner's decision 10 (ignore them on arithmetic, emulate them on
//!   conversions) is the lowering's concern, and an interpreter that honours them lets a test
//!   see what the lowering gives up. `FPMax*` and `FPMin*` return the operand that is not a NaN
//!   (as `std::fmax` in PR #4588's `EvaluateVertex`); `FPClamp*` and `FPSaturate*` are
//!   `min(max(value, low), high)`, Eden's `Clamp` helper's first form, so a NaN clamps to the
//!   low bound and saturates to `+0`.
//! - **Conversions** (`emit_spirv_convert.cpp`): float to integer truncates toward zero; out of
//!   range it saturates and a NaN gives 0 (SPIR-V leaves both undefined). The 8- and 16-bit
//!   integer sources take the low bits of their 32-bit argument, sign- or zero-extended (Eden's
//!   `ExtractS8` and siblings); 16-bit integer results are sign- or zero-extended into 32 bits.
//! - **Texture and image operations** (every opcode whose flags are texture information) run
//!   against a script: [`Machine::images`] lists the calls a test expects, in order, each with
//!   its reply ([`ScriptedImage`]). A call after the last scripted one
//!   ([`InterpError::ImageScriptEnded`]), a call that differs from the next one in its opcode,
//!   texture information or argument values ([`InterpError::ImageMismatch`]), and then a reply
//!   not of the opcode's result type ([`InterpError::ImageReply`]) are errors, checked in that
//!   order, and a refused call leaves the script as it was. Tests stay pure.
//! - **Warp operations** (`emit_spirv_warp.cpp`): `LaneId` (from [`SystemValues`]); the lane
//!   masks `SubgroupEqMask`, `SubgroupLtMask`, `SubgroupLeMask`, `SubgroupGtMask` and
//!   `SubgroupGeMask` computed from it for a warp of 32 lanes: bit `n` is set when lane `n` is
//!   equal to, below, at or below, above, or at or above the lane id (`LoadMask` loads the
//!   first word of SPIR-V's `SubgroupEqMask` and its siblings); a lane id of 32 or more is
//!   above every lane of the word, so the equal mask is 0 and the below masks are all ones;
//!   and the votes, `SubgroupBallot`, the four shuffles, `QuadBroadcast`, `QuadSwap` and
//!   `GetInBoundsFromOp` of a shuffle on a warp whose **only active lane is this invocation**
//!   (`exec`'s `warp` module, added by slice 12h): a vote gives its argument (`VoteEqual`
//!   gives true), a ballot gives this lane's bit or zero, and a shuffle computes its source
//!   lane and in-range flag by Eden's formulas, then gives its own value when the source is
//!   out of range or is this lane, and [`INACTIVE_LANE_VALUE`] when it is another lane (SPIR-V
//!   leaves a read of an inactive invocation undefined; translators read every lane and
//!   select the active ones, as PR #4588's `REDUX` does). A lane id of [`WARP_SIZE`] or more
//!   is [`InterpError::LaneOutsideWarp`] for the ballot, the shuffles and the quad operations
//!   (not for the lane masks). Results that need several active lanes are checked as golden
//!   IR by the translators' tests. Derivatives are not supported.
//!
//! An operation not listed (for example `PhiMove`, `Join`, the `Get*FromOp` pseudo-operations
//! other than `GetInBoundsFromOp` of a shuffle, `EmitVertex`, `EndPrimitive`,
//! `InvocationInfo`, `RenderArea`, the `SR_WScaleFactor` values, the 64-bit and float atomics,
//! `FSwizzleAdd` and the derivatives) is
//! [`InterpError::Unsupported`], naming the opcode; translator slices add semantics as they
//! need them.
//!
//! # Where the interpreter differs from its references
//!
//! Each is a choice for a test oracle, pinned by the named test in `tests`:
//!
//! - `FPClamp32` of a NaN gives the low bound; PR #4588's `EvaluateVertex` uses `std::clamp`,
//!   which passes the NaN through (`float::clamp_and_saturate_take_a_nan_to_the_low_bound`).
//! - A misaligned global, storage or shared access is [`InterpError::Misaligned`] (a local
//!   access takes a word index, so it cannot be misaligned);
//!   Eden's emitter divides the offset by the access size, which reads the aligned-down
//!   address (`StorageIndex`, `emit_spirv_shared_memory.cpp`'s `Pointer`), and its shared
//!   64- and 128-bit word path aligns only to 4 (`memory::global_alignment_then_bounds`,
//!   `memory::shared_memory`). Constant buffers do align down, as Eden does.
//! - An access outside the bytes a test gave is [`InterpError::OutOfBounds`], where a host
//!   with robust buffer access returns zero (`memory::constant_buffer_bounds` and the other
//!   memory tests).
//! - An attribute or patch the test did not give is [`InterpError::MissingInput`]; Eden's
//!   indexed load returns `0.0` for an attribute the program does not declare (`make_load`'s
//!   default case) (`memory::attributes_by_word_and_vertex`).
//! - Directed rounding on `FPAdd64`, `FPMul64` and `FPFma64` is
//!   [`InterpError::DirectedRounding`] rather than modelled
//!   (`float::double_precision_rounds_to_nearest_and_refuses_directed_rounding`).
//!
//! # Limits
//!
//! A run is bounded by [`Machine::step_limit`] (by default [`DEFAULT_STEP_LIMIT`]). A step is one
//! instruction (a phi or an operation), one terminator, or one evaluation in the structured tree
//! of an `if` condition, a `break` condition, a loop's repeat condition, a `return` or an
//! `unreachable`. A run that would take a step past the limit stops with
//! [`InterpError::StepLimit`], so a run of exactly the limit finishes. Memory is bounded by the
//! function (one value slot per instruction) and by the steps (each step adds at most one entry
//! to a map of the machine).

mod exec;
mod float;
mod integer;
mod memory;
#[cfg(test)]
mod tests;
mod value;

use std::collections::{BTreeMap, VecDeque};

use thiserror::Error;

use crate::decode::{Gpr, Pred, UniformPred, UniformReg};
use crate::ir::{
    Attribute, BlockId, Function, Inst, Node, Opcode, Operand, Patch, Site, Terminator,
    TextureInfo, Type, Value, VerifyError, WARP_SIZE,
};

pub use value::Val;

/// The default [`Machine::step_limit`]: about a million steps, far more than a test of one
/// instruction or a short sequence takes.
pub const DEFAULT_STEP_LIMIT: u64 = 1 << 20;

/// What a shuffle gives when its source is a lane other than the running invocation's, which
/// is inactive on the interpreter's one-lane warp (see the module docs). The value is the
/// interpreter's choice, not a reference's: SPIR-V leaves it undefined
/// (`OpGroupNonUniformShuffle`). It is not zero, so a translator that uses such a read
/// without selecting it away changes a test's result.
pub const INACTIVE_LANE_VALUE: u32 = 0xdead_beef;

/// How many per-thread register slots there are: every name a [`Gpr`] can hold (`RZ`'s slot
/// is never read or written; the verifier refuses accesses to it).
pub const REG_SLOTS: usize = 1 << Gpr::BITS;
/// How many predicate slots there are (`PT`'s slot is never read or written).
pub const PRED_SLOTS: usize = 1 << Pred::BITS;
/// How many uniform register slots there are (`URZ`'s slot is never read or written).
pub const UNIFORM_REG_SLOTS: usize = 1 << UniformReg::BITS;
/// How many uniform predicate slots there are (`UPT`'s slot is never read or written).
pub const UNIFORM_PRED_SLOTS: usize = 1 << UniformPred::BITS;

/// The condition-code flags (Eden's `GetZFlag`, `GetSFlag`, `GetCFlag`, `GetOFlag`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "the four independent condition-code flags, one bool each as Eden's flag opcodes read them"
)]
pub struct ConditionCodes {
    /// Zero.
    pub zero: bool,
    /// Sign.
    pub sign: bool,
    /// Carry.
    pub carry: bool,
    /// Overflow.
    pub overflow: bool,
}

/// A run of bytes at a 64-bit address: the global memory a test gives.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Region {
    /// The address of the first byte.
    pub base: u64,
    /// The bytes.
    pub bytes: Vec<u8>,
}

/// The system values a program can read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SystemValues {
    /// `WorkgroupId`.
    pub workgroup_id: [u32; 3],
    /// `LocalInvocationId`.
    pub local_invocation_id: [u32; 3],
    /// `InvocationId`.
    pub invocation_id: u32,
    /// `SampleId`.
    pub sample_id: u32,
    /// `IsHelperInvocation`, before any `DemoteToHelperInvocation`.
    pub is_helper: bool,
    /// `LaneId`.
    pub lane_id: u32,
    /// `YDirection`, as its bits.
    pub y_direction: u32,
}

impl Default for SystemValues {
    /// Everything zero, except `YDirection`, which is `1.0`: Eden's value when the host does not
    /// negate Y (`EmitYDirection`: `y_negate ? -1.0f : 1.0f`).
    fn default() -> Self {
        Self {
            workgroup_id: [0; 3],
            local_invocation_id: [0; 3],
            invocation_id: 0,
            sample_id: 0,
            is_helper: false,
            lane_id: 0,
            y_direction: 1.0_f32.to_bits(),
        }
    }
}

/// A texture or image call: the opcode, its texture information and its argument values.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImageCall {
    /// The opcode.
    pub opcode: Opcode,
    /// Its texture information.
    pub info: TextureInfo,
    /// Its arguments, in order.
    pub args: Vec<Val>,
}

/// A texture or image call a test expects, and what it returns.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScriptedImage {
    /// The call expected.
    pub call: ImageCall,
    /// Its result: `None` for an operation with no result (`ImageWrite` and its siblings),
    /// else a value of the opcode's result type.
    pub reply: Option<Val>,
}

/// A memory space, for errors.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Space {
    /// A constant buffer, by index.
    ConstantBuffer(u32),
    /// Global memory.
    Global,
    /// A storage buffer, by binding.
    Storage(u32),
    /// Local memory.
    Local,
    /// Shared memory.
    Shared,
}

impl std::fmt::Display for Space {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ConstantBuffer(index) => write!(f, "constant buffer {index}"),
            Self::Global => f.write_str("global memory"),
            Self::Storage(binding) => write!(f, "storage buffer {binding}"),
            Self::Local => f.write_str("local memory"),
            Self::Shared => f.write_str("shared memory"),
        }
    }
}

/// An input the program read that the machine does not have.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Input {
    /// An input attribute word of a vertex.
    Attribute {
        /// The attribute word.
        attribute: Attribute,
        /// The vertex.
        vertex: u32,
    },
    /// A patch word.
    Patch(Patch),
    /// An input attribute word interpolated at an offset.
    Interpolated {
        /// The attribute word.
        attribute: Attribute,
        /// The offset's bits.
        offset: u32,
    },
}

impl std::fmt::Display for Input {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Attribute { attribute, vertex } => {
                write!(f, "attribute {attribute} of vertex {vertex}")
            }
            Self::Interpolated { attribute, offset } => {
                write!(f, "attribute {attribute} at offset {offset:#x}")
            }
            Self::Patch(patch) => write!(f, "{patch}"),
        }
    }
}

/// Why a run failed.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum InterpError {
    /// The function does not verify.
    #[error("the function is malformed: {0}")]
    Invalid(#[from] VerifyError),
    /// The interpreter has no semantics for the opcode.
    #[error("{at}: the interpreter does not support {opcode}")]
    Unsupported {
        /// The instruction.
        at: Value,
        /// Its opcode.
        opcode: Opcode,
    },
    /// A 64-bit operation asks for a rounding mode other than to nearest.
    #[error("{at}: the interpreter does not support directed rounding on {opcode}")]
    DirectedRounding {
        /// The instruction.
        at: Value,
        /// Its opcode.
        opcode: Opcode,
    },
    /// The run reached its step limit.
    #[error("the run stopped at its limit of {}", count(*limit, "step"))]
    StepLimit {
        /// The limit.
        limit: u64,
    },
    /// A value was read before the run computed it.
    #[error("{at} reads {value}, which the run has not computed")]
    NotComputed {
        /// Where.
        at: Site,
        /// The value.
        value: Value,
    },
    /// An argument has a type other than the one the operation needs.
    #[error("{at}: argument {index} is {found}, not {wanted}")]
    Type {
        /// Where.
        at: Site,
        /// The argument's index (0 for a condition).
        index: usize,
        /// The type needed.
        wanted: Type,
        /// The type found.
        found: Type,
    },
    /// The first block run has a phi.
    #[error("{0} is a phi in the first block run, so no value flows into it")]
    PhiAtStart(Value),
    /// A phi has no value from the block that ran before its own.
    #[error("{at} has no incoming value from {from}")]
    PhiNoIncoming {
        /// The phi.
        at: Value,
        /// The block that ran before.
        from: BlockId,
    },
    /// Control reached an `unreachable`.
    #[error("control reached the unreachable point in {0}")]
    ReachedUnreachable(Site),
    /// A memory access is not aligned to its size.
    #[error(
        "{at}: an access of {} to {space} at {offset:#x} is not aligned",
        count(*size, "byte")
    )]
    Misaligned {
        /// The instruction.
        at: Value,
        /// The space.
        space: Space,
        /// The offset (an address for global memory).
        offset: u64,
        /// The access size in bytes.
        size: u64,
    },
    /// A memory access falls outside the bytes the machine holds.
    #[error(
        "{at}: an access of {} to {space} at {offset:#x} is outside the {} it holds",
        count(*size, "byte"),
        count(*len, "byte")
    )]
    OutOfBounds {
        /// The instruction.
        at: Value,
        /// The space.
        space: Space,
        /// The offset (an address for global memory).
        offset: u64,
        /// The access size in bytes.
        size: u64,
        /// How many bytes the space holds.
        len: u64,
    },
    /// An input the program read is not given.
    #[error("{at}: {input} is not given")]
    MissingInput {
        /// The instruction.
        at: Value,
        /// The input.
        input: Input,
    },
    /// An indexed attribute offset names no attribute word.
    #[error("{at}: attribute offset {offset:#x} names no attribute")]
    AttributeOffset {
        /// The instruction.
        at: Value,
        /// The byte offset.
        offset: u32,
    },
    /// A composite index is past the vector's last component.
    #[error("{at}: component {index} is past the end of a {ty}")]
    Component {
        /// The instruction.
        at: Value,
        /// The index.
        index: u32,
        /// The vector type.
        ty: Type,
    },
    /// A texture or image call came after the last scripted one.
    #[error("{at}: {opcode} has no scripted call left")]
    ImageScriptEnded {
        /// The instruction.
        at: Value,
        /// Its opcode.
        opcode: Opcode,
    },
    /// A texture or image call differs from the scripted one.
    #[error("{at}: the {opcode} call differs from scripted call {index}")]
    ImageMismatch {
        /// The instruction.
        at: Value,
        /// Its opcode.
        opcode: Opcode,
        /// The scripted call's index (counting from 0 in this run).
        index: usize,
    },
    /// A scripted reply is not of the opcode's result type.
    #[error("{at}: scripted call {index} replies with the wrong type for {opcode}")]
    ImageReply {
        /// The instruction.
        at: Value,
        /// Its opcode.
        opcode: Opcode,
        /// The scripted call's index (counting from 0 in this run).
        index: usize,
    },
    /// A warp operation ran with a lane id the warp does not have ([`SystemValues::lane_id`]
    /// of [`WARP_SIZE`] or more).
    #[error("{at}: lane {lane} is outside the warp of {WARP_SIZE} lanes")]
    LaneOutsideWarp {
        /// The instruction.
        at: Value,
        /// The lane id.
        lane: u32,
    },
}

/// One invocation's state, inputs and outputs. Every field is plain data a test sets before a
/// run and reads after it; [`Machine::default`] is all zero and empty, with
/// [`DEFAULT_STEP_LIMIT`] and [`SystemValues::default`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Machine {
    /// Per-thread registers, by [`Gpr::index`].
    pub regs: [u32; REG_SLOTS],
    /// Predicates, by [`Pred::index`].
    pub preds: [bool; PRED_SLOTS],
    /// Uniform registers, by [`UniformReg::index`].
    pub uniform_regs: [u32; UNIFORM_REG_SLOTS],
    /// Uniform predicates, by [`UniformPred::index`].
    pub uniform_preds: [bool; UNIFORM_PRED_SLOTS],
    /// The condition-code flags.
    pub flags: ConditionCodes,
    /// Goto variables, by index; an absent one reads false.
    pub goto_variables: BTreeMap<u32, bool>,
    /// The indirect branch variable.
    pub indirect_branch: u32,
    /// Constant buffers' bytes, by index.
    pub cbufs: BTreeMap<u32, Vec<u8>>,
    /// Input attribute words, by word and vertex, as their bits.
    pub attributes: BTreeMap<(Attribute, u32), u32>,
    /// Input attribute words interpolated at an offset, by word and the offset's bits, as their
    /// bits: what `InterpolateAtOffset` reads (one invocation cannot interpolate, so a test
    /// gives the result).
    pub interpolated: BTreeMap<(Attribute, u32), u32>,
    /// Output attribute words, by word and vertex, as their bits.
    pub outputs: BTreeMap<(Attribute, u32), u32>,
    /// Patch words, as their bits (read and written; see the module docs).
    pub patches: BTreeMap<Patch, u32>,
    /// Fragment colours, by render target and component, as their bits.
    pub frag_colors: BTreeMap<(u32, u32), u32>,
    /// The fragment depth, as its bits, once written.
    pub frag_depth: Option<u32>,
    /// The sample mask, once written.
    pub sample_mask: Option<u32>,
    /// Local memory, in 32-bit words.
    pub local: Vec<u32>,
    /// Shared memory.
    pub shared: Vec<u8>,
    /// Global memory.
    pub global: Region,
    /// Storage buffers' bytes, by binding.
    pub storage: BTreeMap<u32, Vec<u8>>,
    /// System values.
    pub system: SystemValues,
    /// The texture and image calls expected, in order; a run takes them from the front.
    pub images: VecDeque<ScriptedImage>,
    /// Whether the program ran `DemoteToHelperInvocation`.
    pub demoted: bool,
    /// The most steps a run may take.
    pub step_limit: u64,
}

impl Default for Machine {
    fn default() -> Self {
        Self {
            regs: [0; REG_SLOTS],
            preds: [false; PRED_SLOTS],
            uniform_regs: [0; UNIFORM_REG_SLOTS],
            uniform_preds: [false; UNIFORM_PRED_SLOTS],
            flags: ConditionCodes::default(),
            goto_variables: BTreeMap::new(),
            indirect_branch: 0,
            cbufs: BTreeMap::new(),
            attributes: BTreeMap::new(),
            interpolated: BTreeMap::new(),
            outputs: BTreeMap::new(),
            patches: BTreeMap::new(),
            frag_colors: BTreeMap::new(),
            frag_depth: None,
            sample_mask: None,
            local: Vec::new(),
            shared: Vec::new(),
            global: Region::default(),
            storage: BTreeMap::new(),
            system: SystemValues::default(),
            images: VecDeque::new(),
            demoted: false,
            step_limit: DEFAULT_STEP_LIMIT,
        }
    }
}

/// What a finished run leaves besides the machine's state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Finished {
    /// How many steps the run took.
    pub steps: u64,
    values: Vec<Option<Val>>,
}

impl Finished {
    /// The value the instruction with id `index` (`%index` in the text form) last computed, if
    /// the run computed one.
    pub fn value(&self, index: u32) -> Option<Val> {
        usize::try_from(index)
            .ok()
            .and_then(|slot| self.values.get(slot).copied().flatten())
    }
}

impl Machine {
    /// A machine with every register, predicate and flag clear, no inputs and the default step
    /// limit.
    pub fn new() -> Self {
        Self::default()
    }

    /// Runs one invocation of `function` (see the module docs). The machine keeps the state
    /// the run leaves, also when it fails part-way.
    pub fn run(&mut self, function: &Function) -> Result<Finished, InterpError> {
        function.verify()?;
        let mut run = Run {
            function,
            machine: self,
            values: vec![None; function.insts().count()],
            steps: 0,
            previous: None,
            image_index: 0,
            in_bounds: BTreeMap::new(),
        };
        match function.structure() {
            None => run.graph()?,
            Some(nodes) => {
                run.nodes(nodes)?;
            }
        }
        Ok(Finished {
            steps: run.steps,
            values: run.values,
        })
    }
}

/// How a structured statement list ended.
enum Flow {
    /// It ran to its end.
    Next,
    /// A `break` left the innermost loop.
    Break,
    /// A `return` ended the program.
    Return,
}

/// One run in progress.
struct Run<'a> {
    function: &'a Function,
    machine: &'a mut Machine,
    /// Each instruction's latest value.
    values: Vec<Option<Val>>,
    steps: u64,
    /// The block that ran last.
    previous: Option<BlockId>,
    /// How many scripted image calls this run has taken.
    image_index: usize,
    /// Each shuffle's latest in-range flag, by its id, for `GetInBoundsFromOp` (one entry per
    /// shuffle instruction, so bounded by the function's size).
    in_bounds: BTreeMap<Value, bool>,
}

impl Run<'_> {
    /// Counts one step, refusing one past the limit.
    fn step(&mut self) -> Result<(), InterpError> {
        if self.steps >= self.machine.step_limit {
            return Err(InterpError::StepLimit {
                limit: self.machine.step_limit,
            });
        }
        self.steps += 1;
        Ok(())
    }

    /// The value of `operand`, read at `at`.
    fn operand(&self, at: Site, operand: Operand) -> Result<Result<Val, Type>, InterpError> {
        match operand {
            Operand::Imm(imm) => Ok(Val::from_imm(imm).ok_or(imm.ty())),
            Operand::Value(value) => usize::try_from(value.index())
                .ok()
                .and_then(|slot| self.values.get(slot).copied().flatten())
                .map(Ok)
                .ok_or(InterpError::NotComputed { at, value }),
        }
    }

    /// The boolean `cond` at `at`.
    fn condition(&self, at: Site, cond: Operand) -> Result<bool, InterpError> {
        match self.operand(at, cond)? {
            Ok(Val::U1(value)) => Ok(value),
            Ok(other) => Err(type_error(at, 0, Type::U1, other.ty())),
            Err(found) => Err(type_error(at, 0, Type::U1, found)),
        }
    }

    /// Runs a control-flow graph from its entry.
    fn graph(&mut self) -> Result<(), InterpError> {
        let mut block = self.function.entry();
        loop {
            self.block(block)?;
            let terminator = self
                .function
                .block(block)
                .map_or(Terminator::None, |b| b.terminator);
            self.step()?;
            let at = Site::Terminator(block);
            block = match terminator {
                Terminator::Jump(target) => target,
                Terminator::Branch {
                    cond,
                    then,
                    otherwise,
                } => {
                    if self.condition(at, cond)? {
                        then
                    } else {
                        otherwise
                    }
                }
                Terminator::Return => return Ok(()),
                Terminator::Unreachable => return Err(InterpError::ReachedUnreachable(at)),
                Terminator::None => return Err(VerifyError::Unterminated(block).into()),
            };
        }
    }

    /// Runs a structured statement list.
    fn nodes(&mut self, nodes: &[Node]) -> Result<Flow, InterpError> {
        let at = Site::Structure;
        for node in nodes {
            match node {
                Node::Block(block) => self.block(*block)?,
                Node::If { cond, body } => {
                    self.step()?;
                    if self.condition(at, *cond)? {
                        match self.nodes(body)? {
                            Flow::Next => {}
                            flow @ (Flow::Break | Flow::Return) => return Ok(flow),
                        }
                    }
                }
                Node::Loop { body, repeat } => loop {
                    match self.nodes(body)? {
                        Flow::Next => {}
                        Flow::Break => break,
                        Flow::Return => return Ok(Flow::Return),
                    }
                    self.step()?;
                    if !self.condition(at, *repeat)? {
                        break;
                    }
                },
                Node::Break { cond } => {
                    self.step()?;
                    if self.condition(at, *cond)? {
                        return Ok(Flow::Break);
                    }
                }
                Node::Return => {
                    self.step()?;
                    return Ok(Flow::Return);
                }
                Node::Unreachable => {
                    self.step()?;
                    return Err(InterpError::ReachedUnreachable(at));
                }
            }
        }
        Ok(Flow::Next)
    }

    /// Runs a block's instructions: its phis together, then its operations in order.
    fn block(&mut self, block: BlockId) -> Result<(), InterpError> {
        let function = self.function;
        let Some(listed) = function.block(block) else {
            return Err(VerifyError::NoSuchBlock {
                at: Site::Structure,
                block,
            }
            .into());
        };
        // Phis read every incoming value before any of them is written.
        let mut pending: Vec<(Value, Val)> = Vec::new();
        let mut first_op = listed.insts.len();
        for (position, &value) in listed.insts.iter().enumerate() {
            let Some(Inst::Phi(phi)) = function.inst(value) else {
                first_op = position;
                break;
            };
            self.step()?;
            let Some(from) = self.previous else {
                return Err(InterpError::PhiAtStart(value));
            };
            let Some(&(_, incoming)) = phi.incoming.iter().find(|(pred, _)| *pred == from) else {
                return Err(InterpError::PhiNoIncoming { at: value, from });
            };
            let at = Site::Inst(value);
            let val = match self.operand(at, incoming)? {
                Ok(val) => val,
                Err(found) => return Err(type_error(at, 0, phi.ty, found)),
            };
            if val.ty() != phi.ty {
                return Err(type_error(at, 0, phi.ty, val.ty()));
            }
            pending.push((value, val));
        }
        for (value, val) in pending {
            self.set(value, val);
        }
        for &value in listed.insts.iter().skip(first_op) {
            let Some(Inst::Op(op)) = function.inst(value) else {
                // The verifier refuses a phi after an operation.
                return Err(VerifyError::PhiAfterOperation(value).into());
            };
            self.step()?;
            if let Some(val) = self.op(value, op)? {
                self.set(value, val);
            }
        }
        self.previous = Some(block);
        Ok(())
    }

    /// Records `val` as the value of `value`.
    fn set(&mut self, value: Value, val: Val) {
        if let Some(slot) = usize::try_from(value.index())
            .ok()
            .and_then(|slot| self.values.get_mut(slot))
        {
            *slot = Some(val);
        }
    }
}

/// `n` and `noun`, plural unless `n` is 1, for error words.
fn count(n: u64, noun: &str) -> String {
    if n == 1 {
        format!("1 {noun}")
    } else {
        format!("{n} {noun}s")
    }
}

/// A [`InterpError::Type`].
const fn type_error(at: Site, index: usize, wanted: Type, found: Type) -> InterpError {
    InterpError::Type {
        at,
        index,
        wanted,
        found,
    }
}
