//! Slice 14: global memory to storage buffers (`docs/specs/shader-pipeline.md`, "Passes needed
//! for correctness", item 4).
//!
//! [`to_storage_buffers`] turns each global memory access (`LoadGlobal*`, `WriteGlobal*`,
//! `GlobalAtomic*`: what 12e translates `LDG`, `STG`, `ATOMG` and `RED` to) into the matching
//! storage-buffer access, by tracing its 64-bit address back to a pointer stored in a constant
//! buffer, and returns the storage buffers the shader uses ([`StorageBuffers`]): for each, the
//! constant-buffer bank and offset of its pointer and whether the shader writes through it.
//! That is the part of the resource table this pass owns. The GPU crate binds each buffer per
//! draw from the pointer it reads at that place; in Eden that is
//! `BufferCache::StorageBufferBinding` (`eden/src/video_core/buffer_cache/buffer_cache.h`
//! @ cf654052), which reads the 64-bit pointer, aligns it down to the host's storage-buffer
//! alignment and reads the buffer's size from the next 64-bit word when it holds one.
//!
//! # Sources
//!
//! Ported by behaviour from Eden's
//! `eden/src/shader_recompiler/ir_opt/global_memory_to_storage_buffer_pass.cpp` @ cf654052
//! (GPL-3.0-or-later, over GPL-2.0-or-later yuzu code; Copyright 2026 Eden Emulator Project,
//! Copyright 2021 yuzu Emulator Project), with its search, Eden's `BreadthFirstSearch`
//! (`frontend/ir/breadth_first_search.h`), computed in linear time (the scoping doc's
//! performance lesson 7, Eden-NX2's row 121; "Cost" below).
//!
//! # What the pass does, in Eden's order
//!
//! 1. **Collect.** Blocks are walked in post order (the parent module's "Block order",
//!    reversed: Eden walks `post_order_blocks`), each block's instructions in order. For each
//!    global memory access (Eden's `IsGlobalMemory`, the 50 opcodes of [`storage_opcode`]):
//!    - **The low address** (Eden's `TrackLowAddress`). The address, argument 0, must not be
//!      an immediate; an `IAdd64` of it is looked through when its offset, argument 1, is a
//!      `u64` immediate, whose low 32 bits, read as signed, are the immediate offset (Eden's
//!      `static_cast<s32>(static_cast<s64>(U64()))`), and whose base, argument 0, is not an
//!      immediate; then a `PackUint2x32` is looked through when its argument is not an
//!      immediate; what is left must be a `CompositeConstructU32x2`, whose argument 0 is the
//!      low address. `Identity` chains are looked through at each of these steps, as Eden's
//!      `InstRecursive` and `IsImmediate` do (`tests::look_through`). An argument one of these
//!      steps reads that is not there (the access's argument 0, the `IAdd64`'s argument 1 or
//!      0, the `PackUint2x32`'s argument 0, the pair's argument 0) is
//!      [`UntracedReason::MissingArgument`], reported at that step
//!      (`tests::rejections::a_missing_address_argument_is_refused`). Only a function that does
//!      not verify has one; Eden's instructions always carry their opcode's arguments, and its
//!      `Inst::Arg` does not check the index (`frontend/ir/value.h`).
//!    - **The pointer** (Eden's `Track`). A search from the low address for a `GetCbufU32` or
//!      `GetCbufU32x2` whose bank and offset are `u32` immediates, the offset a multiple of
//!      the alignment, first with NVN's bias ([`NVN_BIAS`]: bank 0, offsets `0x100` to `0x700`,
//!      the end excluded, aligned to 16), then, if that finds none, without a bias, aligned to
//!      [`UNBIASED_ALIGNMENT`]. The search finds what Eden's breadth-first search finds: the
//!      nearest such read along arguments (a phi's incoming values included), and among the
//!      nearest, the one Eden's walk meets first, which visits each instruction's arguments
//!      right to left.
//!    - The access is recorded with its pointer, and the pointer's buffer as written when the
//!      access is a store or an atomic (Eden's `IsGlobalMemoryWrite`).
//! 2. **The table.** The buffers, each once, sorted by bank then offset (Eden's `flat_set` of
//!    `StorageBufferAddr`, compared by index then offset), become [`StorageBuffers`]; a buffer
//!    is written when any access to it writes. A buffer's binding is its position in the table.
//! 3. **Replace**, in collection order (Eden's `StorageOffset` and `Replace`). In front of the
//!    access: the low address plus the immediate offset when that is not zero (`IAdd32`), a
//!    `GetCbufU32` of the pointer's bank and offset, that word with its bits below the host
//!    alignment cleared (`BitwiseAnd32` with `!(alignment - 1)`), and the offset into the
//!    buffer, the first minus the second (`ISub32`); then the storage opcode with the binding,
//!    the offset and, for a store or an atomic, the access's argument 1. A load or an atomic
//!    becomes an `Identity` of the new instruction (Eden's `ReplaceUsesWith`); a store becomes
//!    a `Void` with no arguments (Eden's `Invalidate`) and stays in its block, as in Eden,
//!    until identity removal or dead code elimination takes it out. The new instructions
//!    carry their opcode's default flags, as Eden's `PrependNewInst` gives them.
//!
//! Every refusal but one happens while collecting, before anything is changed, so a function
//! refused with [`GlobalMemoryError::Untraced`], [`GlobalMemoryError::Bank`],
//! [`GlobalMemoryError::Offset`], [`GlobalMemoryError::MissingData`] or
//! [`GlobalMemoryError::Pass`] is left as it was (`tests::rejections::a_refused_function_is_unchanged`;
//! every refusal test checks it). The checks run access by access in collection order, and for
//! one access in this order: the low address (each [`UntracedReason`] in the order above), the
//! pointer ([`UntracedReason::ImmediateLowWord`], then [`UntracedReason::NoPointer`]), then the
//! pointer's bank ([`GlobalMemoryError::Bank`]), its offset ([`GlobalMemoryError::Offset`])
//! and, for a store or an atomic, its data argument ([`GlobalMemoryError::MissingData`]). The
//! first fault is the error (`tests::rejections::the_first_access_in_post_order_is_reported`,
//! `tests::rejections::bank_is_checked_before_offset`,
//! `tests::rejections::offset_is_checked_before_data`, and bank before data in
//! `tests::rejections::a_missing_data_argument_is_refused`). A malformed function
//! ([`GlobalMemoryError::Pass`]: an `Identity` chain that loops, an id with no instruction) is
//! reported when the walk meets it, before those checks: every access's low address is walked
//! before any access's pointer is checked, so a loop in a later access's low address is the
//! error even when an earlier access would be refused
//! (`tests::rejections::a_malformed_low_address_is_reported_before_an_earlier_refusal`). Once
//! every access has passed them, an arena without room for the replacements is refused
//! ([`BuildError::Full`]; `tests::arena_room_edges` pins the decision, the pure `arena_fits`; no
//! test reaches its call, which needs an arena of about 2^32 instructions).
//!
//! The replacement then changes the function block by block. [`Function::push`] refuses a
//! full arena, which `arena_fits` has ruled out, and a block that does not exist, which cannot
//! happen, as the blocks replaced are the ones the walk listed; either would be returned as
//! [`GlobalMemoryError::Pass`] with the blocks before it already converted. The one refusal
//! that comes after the change is [`GlobalMemoryError::Verify`]: in tests and debug builds the
//! output is verified, and a function that does not verify is returned converted, as the
//! parent module's `run` leaves a pass's output
//! (`tests::a_function_that_does_not_verify_is_converted_then_refused`).
//!
//! # Where this pass departs from Eden's
//!
//! Each pinned by a test in `tests`:
//!
//! - **An address that cannot be traced is a rejection** naming the instruction and the reason
//!   ([`GlobalMemoryError::Untraced`]), where Eden logs a warning and leaves the access as
//!   global memory for its backend's NVN fallbacks. This is the owner's decision 9 of the
//!   spec (2026-11-12): "(a), rejected at first, revisited when a direct backend can share
//!   memory"; naga has no pointers made from integers ("The backend through naga"). Census E4
//!   counts such addresses for the revisit (`tests::rejections::each_reason_is_refused_exactly`).
//! - **The search runs in linear time** over the whole function ("Cost"), and gives Eden's
//!   result: `tests::search::the_search_matches_edens_breadth_first_search` runs a direct port
//!   of Eden's `BreadthFirstSearch` beside it on generated graphs with shared arguments and
//!   phi cycles.
//! - **A pointer in a bank past the last** (bank [`CBUF_BANKS`] or more) is refused
//!   ([`GlobalMemoryError::Bank`]), and so is **a pointer whose 8 bytes ([`POINTER_BYTES`]) end
//!   past a bank's 64 KiB** ([`INDEXED_CBUF_SIZE`]; [`GlobalMemoryError::Offset`]). Eden takes
//!   any bank and offset the search finds. The translator gives banks below 32 (a five-bit
//!   field), but `LDC`'s signed 16-bit offset can make an offset such as `0xfffffff8`
//!   (`tests::rejections::bank_edges`, `tests::rejections::offset_edges`).
//! - **The host alignment must be a power of two** ([`StorageAlignment::new`]). Eden's
//!   `HostTranslateInfo::ApplyDescriptorLimitPolicy` turns 0 into 1 and takes any other value,
//!   whose `~(alignment - 1)` is not an alignment mask unless it is a power of two
//!   (`tests::alignment_is_a_power_of_two`).
//! - **Blocks the entry cannot reach are walked too**, and the accesses in them converted or
//!   refused; Eden removes such blocks before its passes run (`RemoveUnreachableBlocks`), so
//!   their pointers never reach its table (`tests::unreachable_blocks_are_walked`).
//! - **An immediate of another kind than Eden's read** (a bank or offset that is not a `u32`,
//!   an `IAdd64` offset that is not a `u64`) is not a match, or not an immediate offset
//!   ([`UntracedReason::OffsetNotImmediate`]); Eden reads its bits under a debug assertion.
//!   Only a function that does not verify has one (`tests::rejections::wrong_immediate_kinds`).
//! - Eden's two warnings ("failed to track", "tracked without bias") are not logged: the first
//!   is an error here and the second is not observable. The descriptor's `count`, always 1 in
//!   Eden (storage-buffer arrays are its "TODO"), is not kept. Eden's `StorageOffset` fallback
//!   for an access whose low address does not track (`UConvert` of the address) is not ported:
//!   such an access is never collected, in Eden or here. Eden's `JoinStorageInfo`, which
//!   merges two programs' tables, has no caller in this crate yet.
//!
//! Ported as Eden has it, though it could surprise: the immediate offset keeps only its low 32
//! bits, so an `IAdd64` offset whose low word is zero adds nothing to the offset
//! (`tests::golden::an_offset_with_a_zero_low_word_adds_nothing`), and the float atomics'
//! controls are dropped, as the storage opcodes carry none in Eden's table.
//!
//! **Conflict: PR #4588's search, not followed.** The two Switch 2 references disagree here.
//! PR #4588's version of this pass
//! (`eden-pr4588/src/shader_recompiler/ir_opt/global_memory_to_storage_buffer_pass.cpp`
//! @ e5fb3b97, `CollectStorageBuffers` and `Track` with `native_env`) takes a different search
//! whenever the environment's `HasNativeRegisterCopies()` is true, and that is every Switch 2
//! program it lowers: `GenericEnvironment::HasNativeRegisterCopies` returns true whenever
//! `native_source_size` is set (`video_core/shader_environment.cpp` @ e5fb3b97), and the
//! default lowering path sets it (`native_source_size = native_header_size +
//! lowered.source_size;` after `LowerToMaxwell`), so it runs no Switch 2 program through
//! Eden's search. That search has no NVN bias, follows only
//! `IAdd32`, phis, the two values of a `SelectU32` and component 0 of a
//! `CompositeExtractU32x2`, and accepts a pointer only when the environment finds a mapped GPU
//! address in that constant-buffer word (`IsCbufStorageBufferPointer`). The PR's stated
//! reason for dropping the bias is that applying it "can skip the pointer and select a
//! dimension or stride instead". Eden-NX2's copy of the pass
//! (`eden-nx2/src/shader_recompiler/ir_opt/global_memory_to_storage_buffer_pass.cpp`
//! @ 50fa86bb) keeps Eden's biased breadth-first search, adding only a memo of each start's
//! result. Graphics-server decision 8 (the larger answer where one contains the other,
//! otherwise the one a title ran on) names no winner: neither search contains the other, and
//! both references ran titles. So this follows the spec's sentence for the pass, "Ported as
//! is" (shader-pipeline, "Passes needed for correctness", item 4), which is Eden's and
//! Eden-NX2's search. Whether Switch 2 programs should take PR #4588's search instead is an
//! **owner question**.
//!
//! # Where it runs
//!
//! Eden runs the pass after constant propagation and its position pass, before its geometry
//! compaction and texture passes (`frontend/maxwell/translate_program.cpp` @ cf654052, lines
//! 389–396). It is not yet in [`super::Pass::ORDER`]: it takes the host alignment and returns
//! the table, which the chain that runs it (the production translation, which wires the
//! passes) passes and keeps. The host alignment changes the IR, so it belongs to the
//! translation's inputs. It takes a control-flow graph or a structured function
//! (`tests::golden::a_structured_function_converts`).
//!
//! # Cost
//!
//! Eden searches again from every access, and its search keeps visited instructions in a list
//! it scans, so one access can cost the square of the instructions it walks and a program the
//! square of its accesses (Eden-NX2's row 121: 3.4 s for one 28,102-instruction shader; NX2's
//! fix, a hash set and a memo per start, still searches once per distinct start). Here the
//! search is made once for all accesses: the instructions their low addresses reach are walked
//! once along their arguments, then a breadth-first walk from every matching read backwards
//! along those arguments gives each instruction its distance to the nearest match, and each
//! start's match is found by stepping to the rightmost argument one step nearer, remembering
//! each instruction's match. Each instruction and argument is visited a fixed number of times,
//! so the [`Work`] the pass returns is linear in the function's size, and so is the memory it
//! holds (eight 32-bit words and an optional bank and offset per arena slot, two 32-bit words
//! per argument of a reached instruction, and the access lists): `tests::linear` pins it on a program
//! at the translation driver's cap ([`crate::translate::driver::MAX_TRANSLATED_INSTRUCTIONS`])
//! and the exact work of one fixed function.

#[cfg(test)]
mod tests;

use thiserror::Error;

use crate::info::INDEXED_CBUF_SIZE;
use crate::ir::{
    BlockId, BuildError, Function, Imm, Inst, Opcode, Operand, Operation, Value, VerifyError,
};
use crate::translate::float::CBUF_BANKS;

use super::identity::Resolver;
use super::{PassError, VERIFY, Work, arena_len, block_order, identity_of, invalidated};

/// A range of constant-buffer words a pointer search prefers (Eden's `Bias`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Bias {
    /// The bank.
    bank: u32,
    /// The first offset in the range.
    begin: u32,
    /// The offset just past the range.
    end: u32,
    /// The alignment of the offset.
    alignment: u32,
}

/// NVN's storage-buffer range, which the search tries first (Eden's `nvn_bias` in
/// `CollectStorageBuffers`: index 0, `offset_begin` `0x100`, `offset_end` `0x700`, alignment
/// 16).
const NVN_BIAS: Bias = Bias {
    bank: 0,
    begin: 0x100,
    end: 0x700,
    alignment: 16,
};

/// The alignment of a pointer found without the bias (Eden's `Track`: `bias ? bias->alignment
/// : 8U`).
const UNBIASED_ALIGNMENT: u32 = 8;

/// The bytes of a storage-buffer pointer in a constant buffer: Eden's binding reads a `u64`
/// there (`StorageBufferBinding`: `gpu_memory->Read<u64>(ssbo_addr)`).
const POINTER_BYTES: u32 = 8;

/// The host's storage-buffer offset alignment (Eden's `HostTranslateInfo::min_ssbo_alignment`;
/// wgpu's `min_storage_buffer_offset_alignment`): a power of two.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct StorageAlignment(u32);

impl StorageAlignment {
    /// The alignment of `bytes`, which must be a power of two ([`AlignmentError`]).
    pub const fn new(bytes: u32) -> Result<Self, AlignmentError> {
        if bytes.is_power_of_two() {
            Ok(Self(bytes))
        } else {
            Err(AlignmentError(bytes))
        }
    }

    /// The alignment in bytes.
    pub const fn bytes(self) -> u32 {
        self.0
    }

    /// The mask that aligns a value down to this alignment (Eden's `~(alignment - 1U)`).
    const fn mask(self) -> u32 {
        !self.0.wrapping_sub(1)
    }
}

/// A storage-buffer alignment that is not a power of two.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Error)]
#[error("a storage buffer alignment of {0} bytes is not a power of two")]
pub struct AlignmentError(pub u32);

/// One storage buffer a shader uses: where its pointer is, and whether the shader writes
/// through it (Eden's `StorageBufferDescriptor`, without its `count`, which Eden always sets
/// to 1).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct StorageBufferDescriptor {
    bank: u8,
    offset: u32,
    written: bool,
}

impl StorageBufferDescriptor {
    /// The constant-buffer bank that holds the pointer, below [`CBUF_BANKS`].
    pub const fn bank(self) -> u8 {
        self.bank
    }

    /// The pointer's byte offset in its bank: its 8 bytes end at or before
    /// [`INDEXED_CBUF_SIZE`].
    pub const fn offset(self) -> u32 {
        self.offset
    }

    /// Whether the shader stores to the buffer or runs an atomic on it.
    pub const fn is_written(self) -> bool {
        self.written
    }
}

/// What [`to_storage_buffers`] found: the shader's storage buffers, by binding, and the work it
/// did.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StorageBuffers {
    descriptors: Vec<StorageBufferDescriptor>,
    work: Work,
}

impl StorageBuffers {
    /// The storage buffers, sorted by bank then offset, each once; a buffer's binding (the
    /// first argument of its storage accesses) is its position here.
    pub fn descriptors(&self) -> &[StorageBufferDescriptor] {
        &self.descriptors
    }

    /// The work the pass did, in the parent module's steps.
    pub const fn work(&self) -> Work {
        self.work
    }
}

/// Why a global memory access's address could not be traced to a constant buffer (module
/// docs, "What the pass does").
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UntracedReason {
    /// The access, or an instruction its low address is built from, lacks an argument the
    /// walk reads: the access's argument 0, an `IAdd64`'s argument 0 or 1, a `PackUint2x32`'s
    /// argument 0 or a `CompositeConstructU32x2`'s argument 0 (only a function that does not
    /// verify has one).
    MissingArgument,
    /// The address is an immediate.
    ImmediateAddress,
    /// The address is an `IAdd64` whose offset is not a `u64` immediate.
    OffsetNotImmediate,
    /// The address is an `IAdd64` whose base is an immediate.
    ImmediateBase,
    /// The address is a `PackUint2x32` of an immediate (only a function that does not verify
    /// has one: no immediate is a vector).
    ImmediatePair,
    /// The address is not built from two 32-bit words (`CompositeConstructU32x2`).
    NotAPair,
    /// The address's low word is an immediate.
    ImmediateLowWord,
    /// No constant-buffer word the search accepts holds the low word.
    NoPointer,
}

impl UntracedReason {
    /// The reason in words.
    pub const fn describe(self) -> &'static str {
        match self {
            Self::MissingArgument => {
                "the access or an instruction its address is built from lacks an argument"
            }
            Self::ImmediateAddress => "the address is an immediate",
            Self::OffsetNotImmediate => "the 64-bit addition's offset is not a u64 immediate",
            Self::ImmediateBase => "the 64-bit addition's base is an immediate",
            Self::ImmediatePair => "the packed pair is an immediate",
            Self::NotAPair => "the address is not built from two 32-bit words",
            Self::ImmediateLowWord => "the address's low word is an immediate",
            Self::NoPointer => "no constant buffer word holds the address's low word",
        }
    }
}

impl std::fmt::Display for UntracedReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.describe())
    }
}

/// Why [`to_storage_buffers`] failed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Error)]
pub enum GlobalMemoryError {
    /// An access's address cannot be traced to a pointer in a constant buffer (decision 9).
    #[error("{at}: the address of {opcode} cannot be traced to a storage buffer pointer: {reason}")]
    Untraced {
        /// The access.
        at: Value,
        /// Its opcode.
        opcode: Opcode,
        /// Why.
        reason: UntracedReason,
    },
    /// An access's pointer is in a bank past the last.
    #[error(
        "{at}: {opcode} finds its storage buffer pointer in constant buffer bank {bank}, which does not exist; banks stop at {last}",
        last = CBUF_BANKS - 1
    )]
    Bank {
        /// The access.
        at: Value,
        /// Its opcode.
        opcode: Opcode,
        /// The bank.
        bank: u32,
    },
    /// An access's pointer ends past its bank's 64 KiB.
    #[error(
        "{at}: {opcode} finds its storage buffer pointer at offset {offset:#x} of constant buffer bank {bank}, and its 8 bytes end past the bank's {INDEXED_CBUF_SIZE:#x}"
    )]
    Offset {
        /// The access.
        at: Value,
        /// Its opcode.
        opcode: Opcode,
        /// The bank.
        bank: u8,
        /// The offset.
        offset: u32,
    },
    /// A store or an atomic without its data argument (argument 1), which only a function that
    /// does not verify has.
    #[error("{at}: {opcode} has no data argument")]
    MissingData {
        /// The access.
        at: Value,
        /// Its opcode.
        opcode: Opcode,
    },
    /// The function is malformed (an `Identity` chain that loops, an id with no instruction),
    /// or its arena has no room for the replacements ([`BuildError::Full`]).
    #[error(transparent)]
    Pass(#[from] PassError),
    /// The output does not verify (checked in tests and debug builds, as the parent module's
    /// `run` checks). The function is left converted, as `run` leaves a pass's output.
    #[error("after global memory to storage buffers: {0}")]
    Verify(VerifyError),
}

/// What an access is, for its replacement and its buffer's written flag.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Access {
    /// A load: the result is the storage load's.
    Load,
    /// A store: it writes and has no result.
    Write,
    /// An atomic: it writes, and the result is the storage atomic's.
    Atomic,
}

/// The storage opcode a global memory opcode becomes and what it is (Eden's `IsGlobalMemory`,
/// `IsGlobalMemoryWrite` and `GlobalToStorage`); `None` for any other opcode.
const fn storage_opcode(opcode: Opcode) -> Option<(Opcode, Access)> {
    use Access::{Atomic, Load, Write};
    use Opcode as O;
    Some(match opcode {
        O::LoadGlobalS8 => (O::LoadStorageS8, Load),
        O::LoadGlobalU8 => (O::LoadStorageU8, Load),
        O::LoadGlobalS16 => (O::LoadStorageS16, Load),
        O::LoadGlobalU16 => (O::LoadStorageU16, Load),
        O::LoadGlobal32 => (O::LoadStorage32, Load),
        O::LoadGlobal64 => (O::LoadStorage64, Load),
        O::LoadGlobal128 => (O::LoadStorage128, Load),
        O::WriteGlobalS8 => (O::WriteStorageS8, Write),
        O::WriteGlobalU8 => (O::WriteStorageU8, Write),
        O::WriteGlobalS16 => (O::WriteStorageS16, Write),
        O::WriteGlobalU16 => (O::WriteStorageU16, Write),
        O::WriteGlobal32 => (O::WriteStorage32, Write),
        O::WriteGlobal64 => (O::WriteStorage64, Write),
        O::WriteGlobal128 => (O::WriteStorage128, Write),
        O::GlobalAtomicIAdd32 => (O::StorageAtomicIAdd32, Atomic),
        O::GlobalAtomicSMin32 => (O::StorageAtomicSMin32, Atomic),
        O::GlobalAtomicUMin32 => (O::StorageAtomicUMin32, Atomic),
        O::GlobalAtomicSMax32 => (O::StorageAtomicSMax32, Atomic),
        O::GlobalAtomicUMax32 => (O::StorageAtomicUMax32, Atomic),
        O::GlobalAtomicInc32 => (O::StorageAtomicInc32, Atomic),
        O::GlobalAtomicDec32 => (O::StorageAtomicDec32, Atomic),
        O::GlobalAtomicAnd32 => (O::StorageAtomicAnd32, Atomic),
        O::GlobalAtomicOr32 => (O::StorageAtomicOr32, Atomic),
        O::GlobalAtomicXor32 => (O::StorageAtomicXor32, Atomic),
        O::GlobalAtomicExchange32 => (O::StorageAtomicExchange32, Atomic),
        O::GlobalAtomicIAdd64 => (O::StorageAtomicIAdd64, Atomic),
        O::GlobalAtomicSMin64 => (O::StorageAtomicSMin64, Atomic),
        O::GlobalAtomicUMin64 => (O::StorageAtomicUMin64, Atomic),
        O::GlobalAtomicSMax64 => (O::StorageAtomicSMax64, Atomic),
        O::GlobalAtomicUMax64 => (O::StorageAtomicUMax64, Atomic),
        O::GlobalAtomicAnd64 => (O::StorageAtomicAnd64, Atomic),
        O::GlobalAtomicOr64 => (O::StorageAtomicOr64, Atomic),
        O::GlobalAtomicXor64 => (O::StorageAtomicXor64, Atomic),
        O::GlobalAtomicExchange64 => (O::StorageAtomicExchange64, Atomic),
        O::GlobalAtomicIAdd32x2 => (O::StorageAtomicIAdd32x2, Atomic),
        O::GlobalAtomicSMin32x2 => (O::StorageAtomicSMin32x2, Atomic),
        O::GlobalAtomicUMin32x2 => (O::StorageAtomicUMin32x2, Atomic),
        O::GlobalAtomicSMax32x2 => (O::StorageAtomicSMax32x2, Atomic),
        O::GlobalAtomicUMax32x2 => (O::StorageAtomicUMax32x2, Atomic),
        O::GlobalAtomicAnd32x2 => (O::StorageAtomicAnd32x2, Atomic),
        O::GlobalAtomicOr32x2 => (O::StorageAtomicOr32x2, Atomic),
        O::GlobalAtomicXor32x2 => (O::StorageAtomicXor32x2, Atomic),
        O::GlobalAtomicExchange32x2 => (O::StorageAtomicExchange32x2, Atomic),
        O::GlobalAtomicAddF32 => (O::StorageAtomicAddF32, Atomic),
        O::GlobalAtomicAddF16x2 => (O::StorageAtomicAddF16x2, Atomic),
        O::GlobalAtomicAddF32x2 => (O::StorageAtomicAddF32x2, Atomic),
        O::GlobalAtomicMinF16x2 => (O::StorageAtomicMinF16x2, Atomic),
        O::GlobalAtomicMinF32x2 => (O::StorageAtomicMinF32x2, Atomic),
        O::GlobalAtomicMaxF16x2 => (O::StorageAtomicMaxF16x2, Atomic),
        O::GlobalAtomicMaxF32x2 => (O::StorageAtomicMaxF32x2, Atomic),
        _ => return None,
    })
}

/// Runs the pass on `function` with the host's storage-buffer alignment (see the module docs):
/// every global memory access becomes a storage-buffer access, and the storage buffers are
/// returned. A function refused with any error but [`GlobalMemoryError::Verify`] is left
/// unchanged; `Verify`, checked in tests and debug builds after the replacement, is returned
/// with the function already converted (module docs).
pub fn to_storage_buffers(
    function: &mut Function,
    alignment: StorageAlignment,
) -> Result<StorageBuffers, GlobalMemoryError> {
    let mut work = Work::default();
    let accesses = find_accesses(function, &mut work)?;
    if accesses.is_empty() {
        return Ok(StorageBuffers {
            descriptors: Vec::new(),
            work,
        });
    }
    let collected = collect(function, &accesses, &mut work)?;
    let descriptors = table(&accesses, &collected);
    // Replace, block by block in collection order (a block's accesses are consecutive there).
    let mut start = 0;
    while let Some(first) = accesses.get(start) {
        let block = first.block;
        let end = accesses
            .iter()
            .skip(start)
            .position(|found| found.block != block)
            .map_or(accesses.len(), |len| start + len);
        replace_block(
            function,
            block,
            accesses.get(start..end).unwrap_or_default(),
            collected.get(start..end).unwrap_or_default(),
            &descriptors,
            alignment,
            &mut work,
        )?;
        start = end;
    }
    if VERIFY {
        function.verify().map_err(GlobalMemoryError::Verify)?;
    }
    Ok(StorageBuffers { descriptors, work })
}

/// The global memory accesses, in collection order: blocks in post order, each block's
/// instructions in order.
fn find_accesses(function: &Function, work: &mut Work) -> Result<Vec<Found>, PassError> {
    let mut order = block_order(function, work);
    order.reverse();
    let mut accesses = Vec::new();
    for &block in &order {
        let Some(listed) = function.block(block) else {
            continue;
        };
        for &value in &listed.insts {
            work.step();
            let Some(inst) = function.inst(value) else {
                return Err(PassError::Missing(value));
            };
            if let Inst::Op(op) = inst
                && let Some((storage, access)) = storage_opcode(op.opcode)
            {
                accesses.push(Found {
                    block,
                    value,
                    opcode: op.opcode,
                    storage,
                    access,
                });
            }
        }
    }
    Ok(accesses)
}

/// Collects every access (module docs, "What the pass does", step 1): the low addresses, the
/// search over all of them at once, then each access's checks in collection order, then the
/// arena's room. Changes nothing.
fn collect(
    function: &Function,
    accesses: &[Found],
    work: &mut Work,
) -> Result<Vec<Collected>, GlobalMemoryError> {
    let mut resolver = Resolver::new(function);
    let mut lows = Vec::with_capacity(accesses.len());
    for found in accesses {
        work.step();
        lows.push(low_address(function, &mut resolver, found.value, work)?);
    }
    let starts: Vec<Value> = lows
        .iter()
        .filter_map(|low| low.as_ref().ok().map(|low| low.start))
        .collect();
    let mut tracker = Tracker::new(function, &mut resolver, &starts, work)?;
    let mut collected = Vec::with_capacity(accesses.len());
    for (found, low) in accesses.iter().zip(lows) {
        work.step();
        collected.push(check(function, &mut tracker, found, low, work)?);
    }
    if !arena_fits(arena_len(function), accesses.len()) {
        return Err(PassError::Build(BuildError::Full).into());
    }
    Ok(collected)
}

/// One access's checks after its low address, in the module docs' order: the pointer, its
/// bank, its offset, then a store's or an atomic's data argument.
fn check(
    function: &Function,
    tracker: &mut Tracker,
    found: &Found,
    low: Result<LowAddress, UntracedReason>,
    work: &mut Work,
) -> Result<Collected, GlobalMemoryError> {
    let untraced = |reason| GlobalMemoryError::Untraced {
        at: found.value,
        opcode: found.opcode,
        reason,
    };
    let low = low.map_err(untraced)?;
    let pointer = tracker
        .find(low.start, Search::Biased, work)
        .or_else(|| tracker.find(low.start, Search::Unbiased, work))
        .ok_or_else(|| untraced(UntracedReason::NoPointer))?;
    let (bank, offset) = tracker.place(pointer);
    let bank = u8::try_from(bank)
        .ok()
        .filter(|&bank| bank < CBUF_BANKS)
        .ok_or(GlobalMemoryError::Bank {
            at: found.value,
            opcode: found.opcode,
            bank,
        })?;
    if offset
        .checked_add(POINTER_BYTES)
        .is_none_or(|end| end > INDEXED_CBUF_SIZE)
    {
        return Err(GlobalMemoryError::Offset {
            at: found.value,
            opcode: found.opcode,
            bank,
            offset,
        });
    }
    let data = match found.access {
        Access::Load => None,
        Access::Write | Access::Atomic => {
            Some(operation(function, found.value)?.1.get(1).copied().ok_or(
                GlobalMemoryError::MissingData {
                    at: found.value,
                    opcode: found.opcode,
                },
            )?)
        }
    };
    Ok(Collected {
        low: low.low_word,
        imm_offset: low.imm_offset,
        bank,
        offset,
        data,
    })
}

/// The table (module docs, step 2): each buffer once, by bank then offset, written when any
/// access to it writes.
fn table(accesses: &[Found], collected: &[Collected]) -> Vec<StorageBufferDescriptor> {
    let mut descriptors: Vec<StorageBufferDescriptor> = accesses
        .iter()
        .zip(collected)
        .map(|(found, c)| StorageBufferDescriptor {
            bank: c.bank,
            offset: c.offset,
            written: found.access != Access::Load,
        })
        .collect();
    descriptors.sort_unstable_by_key(|d| (d.bank, d.offset, !d.written));
    descriptors.dedup_by(|later, kept| {
        // Sorted with a written entry first among equal places, so the kept one is written
        // when any is.
        later.bank == kept.bank && later.offset == kept.offset
    });
    descriptors
}

/// A global memory access found in the walk.
#[derive(Clone, Copy, Debug)]
struct Found {
    block: BlockId,
    value: Value,
    opcode: Opcode,
    storage: Opcode,
    access: Access,
}

/// What collecting found for an access.
#[derive(Clone, Copy, Debug)]
struct Collected {
    /// The low address as the pair names it, unresolved (Eden's `addr_inst->Arg(0)`).
    low: Operand,
    /// The immediate offset of an `IAdd64`, 0 without one.
    imm_offset: u32,
    bank: u8,
    offset: u32,
    /// A store's or an atomic's argument 1; `None` for a load.
    data: Option<Operand>,
}

/// The instructions the replacement makes per access, at most: `IAdd32`, `GetCbufU32`,
/// `BitwiseAnd32`, `ISub32` and the storage opcode.
const MADE_PER_ACCESS: u64 = 5;

/// How many instructions an arena can hold: ids are 32-bit ([`Value::index`]), and
/// [`Function::push`] refuses an instruction whose id would not fit.
const ARENA_CAPACITY: u64 = 1 << u32::BITS;

/// Whether an arena of `len` instructions has room for the replacements of `accesses`
/// accesses (at most [`ARENA_CAPACITY`] instructions after them), checked before the first
/// change so that [`Function::push`] cannot refuse part-way.
fn arena_fits(len: usize, accesses: usize) -> bool {
    let (Ok(len), Ok(accesses)) = (u64::try_from(len), u64::try_from(accesses)) else {
        return false;
    };
    accesses
        .checked_mul(MADE_PER_ACCESS)
        .and_then(|made| made.checked_add(len))
        .is_some_and(|total| total <= ARENA_CAPACITY)
}

/// The low address of an access (Eden's `LowAddrInfo`).
#[derive(Clone, Copy, Debug)]
struct LowAddress {
    /// The pair's argument 0, as written.
    low_word: Operand,
    /// The same, resolved: where the search starts.
    start: Value,
    /// The immediate offset's low 32 bits (Eden's `s32`, kept as its bits).
    imm_offset: u32,
}

/// Eden's `TrackLowAddress`, then the search's immediate check (module docs, "What the pass
/// does"): an `Err` is the reason the address does not trace; a malformed function is the
/// outer error.
fn low_address(
    function: &Function,
    resolver: &mut Resolver,
    access: Value,
    work: &mut Work,
) -> Result<Result<LowAddress, UntracedReason>, PassError> {
    let mut arg =
        |value: Value, index: usize, work: &mut Work| -> Result<Option<Operand>, PassError> {
            match operation(function, value)?.1.get(index) {
                Some(&operand) => resolver.resolve(function, operand, work).map(Some),
                None => Ok(None),
            }
        };
    let Some(address) = arg(access, 0, work)? else {
        return Ok(Err(UntracedReason::MissingArgument));
    };
    let Operand::Value(mut current) = address else {
        return Ok(Err(UntracedReason::ImmediateAddress));
    };
    let mut imm_offset = 0_u32;
    if operation(function, current)?.0 == Some(Opcode::IAdd64) {
        let Some(offset) = arg(current, 1, work)? else {
            return Ok(Err(UntracedReason::MissingArgument));
        };
        let Operand::Imm(Imm::U64(bits)) = offset else {
            return Ok(Err(UntracedReason::OffsetNotImmediate));
        };
        // The low 32 bits: Eden's `static_cast<s32>(static_cast<s64>(U64()))`.
        imm_offset = u32::try_from(bits & u64::from(u32::MAX)).unwrap_or_default();
        // An operation with argument 1 has argument 0, so this `None` cannot be reached; it
        // is kept so that a missing base reads as what it is.
        let Some(base) = arg(current, 0, work)? else {
            return Ok(Err(UntracedReason::MissingArgument));
        };
        let Operand::Value(base) = base else {
            return Ok(Err(UntracedReason::ImmediateBase));
        };
        current = base;
    }
    if operation(function, current)?.0 == Some(Opcode::PackUint2x32) {
        let Some(vector) = arg(current, 0, work)? else {
            return Ok(Err(UntracedReason::MissingArgument));
        };
        let Operand::Value(vector) = vector else {
            return Ok(Err(UntracedReason::ImmediatePair));
        };
        current = vector;
    }
    let (opcode, args) = operation(function, current)?;
    if opcode != Some(Opcode::CompositeConstructU32x2) {
        return Ok(Err(UntracedReason::NotAPair));
    }
    let Some(&low_word) = args.first() else {
        return Ok(Err(UntracedReason::MissingArgument));
    };
    let Operand::Value(start) = resolver.resolve(function, low_word, work)? else {
        return Ok(Err(UntracedReason::ImmediateLowWord));
    };
    Ok(Ok(LowAddress {
        low_word,
        start,
        imm_offset,
    }))
}

/// The opcode and arguments of the instruction at `value`; a phi has no opcode and no
/// arguments here, so it matches nothing the low-address walk looks for.
fn operation(function: &Function, value: Value) -> Result<(Option<Opcode>, &[Operand]), PassError> {
    match function.inst(value) {
        Some(Inst::Op(op)) => Ok((Some(op.opcode), &op.args)),
        Some(Inst::Phi(_)) => Ok((None, &[])),
        None => Err(PassError::Missing(value)),
    }
}

/// Which constant-buffer reads a search accepts (Eden's `Track` with and without a bias).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Search {
    /// With [`NVN_BIAS`].
    Biased,
    /// Without a bias, aligned to [`UNBIASED_ALIGNMENT`].
    Unbiased,
}

impl Search {
    const fn slot(self) -> usize {
        match self {
            Self::Biased => 0,
            Self::Unbiased => 1,
        }
    }

    /// Whether a read of `bank` at `offset` is a pointer to this search (Eden's `pred`).
    const fn accepts(self, bank: u32, offset: u32) -> bool {
        match self {
            Self::Biased => {
                offset.is_multiple_of(NVN_BIAS.alignment)
                    && bank == NVN_BIAS.bank
                    && offset >= NVN_BIAS.begin
                    && offset < NVN_BIAS.end
            }
            Self::Unbiased => offset.is_multiple_of(UNBIASED_ALIGNMENT),
        }
    }
}

/// The users of each reached instruction ([`Tracker::users`]).
struct Users {
    /// Where each instruction's users start in `users`, by arena index, and one past the last.
    start: Vec<u32>,
    /// The users, as arena indices.
    users: Vec<u32>,
}

/// No distance, or no match remembered.
const NONE: u32 = u32::MAX;

/// The pointer search over every instruction the starts reach (module docs, "Cost").
struct Tracker {
    /// Each reached instruction's first argument edge in `args`, by arena index ([`NONE`] for
    /// one not reached).
    first: Vec<u32>,
    /// How many argument edges each reached instruction has.
    count: Vec<u32>,
    /// The argument edges: each reached instruction's arguments that resolve to values, in
    /// argument order, as arena indices.
    args: Vec<u32>,
    /// The bank and offset of each reached constant-buffer read with immediate `u32` bank and
    /// offset, by arena index.
    places: Vec<Option<(u32, u32)>>,
    /// Each search's distance from each instruction to its nearest match ([`NONE`] for none).
    dist: [Vec<u32>; 2],
    /// Each search's remembered match per instruction ([`NONE`] until found).
    found: [Vec<u32>; 2],
}

impl Tracker {
    /// Walks every instruction `starts` reach along resolved arguments, then finds each
    /// search's distances.
    fn new(
        function: &Function,
        resolver: &mut Resolver,
        starts: &[Value],
        work: &mut Work,
    ) -> Result<Self, PassError> {
        let len = arena_len(function);
        let mut tracker = Self {
            first: vec![NONE; len],
            count: vec![0; len],
            args: Vec::new(),
            places: vec![None; len],
            dist: [Vec::new(), Vec::new()],
            found: [vec![NONE; len], vec![NONE; len]],
        };
        let reached = tracker.walk(function, resolver, starts, work)?;
        let users = tracker.users(&reached, work);
        for search in [Search::Biased, Search::Unbiased] {
            let dist = tracker.distances(&reached, &users, search, work);
            if let Some(cell) = tracker.dist.get_mut(search.slot()) {
                *cell = dist;
            }
        }
        Ok(tracker)
    }

    /// Walks every instruction `starts` reach along resolved arguments, once each, recording
    /// each one's argument edges and, for a constant-buffer read with immediate `u32` bank and
    /// offset, its place. Returns the reached instructions.
    fn walk(
        &mut self,
        function: &Function,
        resolver: &mut Resolver,
        starts: &[Value],
        work: &mut Work,
    ) -> Result<Vec<Value>, PassError> {
        let mut reached = Vec::new();
        let mut stack: Vec<Value> = Vec::new();
        for &start in starts {
            self.reach(function, start, &mut reached, &mut stack)?;
        }
        while let Some(value) = stack.pop() {
            work.step();
            let index = slot(value);
            let first =
                u32::try_from(self.args.len()).map_err(|_| PassError::Build(BuildError::Full))?;
            let Some(inst) = function.inst(value) else {
                return Err(PassError::Missing(value));
            };
            let operands: &[Operand] = match inst {
                Inst::Op(op) => &op.args,
                Inst::Phi(_) => &[],
            };
            let mut edges = 0_u32;
            let incoming = match inst {
                Inst::Phi(phi) => phi.incoming.as_slice(),
                Inst::Op(_) => &[],
            };
            for &arg in operands.iter().chain(incoming.iter().map(|(_, arg)| arg)) {
                work.step();
                if let Operand::Value(target) = resolver.resolve(function, arg, work)? {
                    self.reach(function, target, &mut reached, &mut stack)?;
                    self.args.push(target.index());
                    edges = edges.saturating_add(1);
                }
            }
            if let Some(slot) = self.first.get_mut(index) {
                *slot = first;
            }
            if let Some(slot) = self.count.get_mut(index) {
                *slot = edges;
            }
            if let Inst::Op(op) = inst
                && matches!(op.opcode, Opcode::GetCbufU32 | Opcode::GetCbufU32x2)
                && let (Some(&bank), Some(&offset)) = (op.args.first(), op.args.get(1))
                && let Operand::Imm(Imm::U32(bank)) = resolver.resolve(function, bank, work)?
                && let Operand::Imm(Imm::U32(offset)) = resolver.resolve(function, offset, work)?
                && let Some(place) = self.places.get_mut(index)
            {
                *place = Some((bank, offset));
            }
        }
        Ok(reached)
    }

    /// The users of each reached instruction, as a reverse adjacency in one array: user ids
    /// of instruction `i` are `users[start[i]..start[i + 1]]`.
    fn users(&self, reached: &[Value], work: &mut Work) -> Users {
        let mut user_start = vec![0_u32; self.first.len().saturating_add(1)];
        for &target in &self.args {
            work.step();
            if let Some(count) = user_start.get_mut(slot_index(target).saturating_add(1)) {
                *count = count.saturating_add(1);
            }
        }
        for index in 1..user_start.len() {
            let previous = user_start.get(index - 1).copied().unwrap_or(0);
            if let Some(current) = user_start.get_mut(index) {
                *current = current.saturating_add(previous);
            }
        }
        let mut fill = user_start.clone();
        let mut users = vec![0_u32; self.args.len()];
        for &value in reached {
            let index = slot(value);
            for target in self.edges(index) {
                work.step();
                if let Some(next) = fill.get_mut(slot_index(target))
                    && let Some(cell) = users.get_mut(slot_index(*next))
                {
                    *cell = value.index();
                    *next = next.saturating_add(1);
                }
            }
        }
        Users {
            start: user_start,
            users,
        }
    }

    /// `search`'s distance from each reached instruction to its nearest match: a
    /// breadth-first walk from every match along users ([`NONE`] for none).
    fn distances(
        &self,
        reached: &[Value],
        users: &Users,
        search: Search,
        work: &mut Work,
    ) -> Vec<u32> {
        let mut dist = vec![NONE; self.first.len()];
        let mut queue: Vec<u32> = Vec::new();
        for &value in reached {
            work.step();
            let index = slot(value);
            if let Some(Some((bank, offset))) = self.places.get(index)
                && search.accepts(*bank, *offset)
                && let Some(d) = dist.get_mut(index)
            {
                *d = 0;
                queue.push(value.index());
            }
        }
        let mut head = 0;
        while let Some(&node) = queue.get(head) {
            head += 1;
            work.step();
            let next = dist
                .get(slot_index(node))
                .copied()
                .unwrap_or(NONE)
                .saturating_add(1);
            let from = users.start.get(slot_index(node)).copied().unwrap_or(0);
            let to = users
                .start
                .get(slot_index(node).saturating_add(1))
                .copied()
                .unwrap_or(from);
            for &user in users
                .users
                .get(slot_index(from)..slot_index(to))
                .unwrap_or_default()
            {
                work.step();
                if let Some(d) = dist.get_mut(slot_index(user))
                    && *d == NONE
                {
                    *d = next;
                    queue.push(user);
                }
            }
        }
        dist
    }

    /// Marks `value` reached and queues it, once.
    fn reach(
        &mut self,
        function: &Function,
        value: Value,
        reached: &mut Vec<Value>,
        stack: &mut Vec<Value>,
    ) -> Result<(), PassError> {
        if function.inst(value).is_none() {
            return Err(PassError::Missing(value));
        }
        let Some(first) = self.first.get_mut(slot(value)) else {
            return Err(PassError::Missing(value));
        };
        if *first == NONE {
            // Reached and queued: a placeholder until its edges are recorded (no edge array
            // has `NONE - 1` entries).
            *first = NONE - 1;
            reached.push(value);
            stack.push(value);
        }
        Ok(())
    }

    /// The argument edges of the instruction at arena index `index`, in argument order.
    fn edges(&self, index: usize) -> impl DoubleEndedIterator<Item = u32> + '_ {
        let first = self.first.get(index).copied().unwrap_or(NONE);
        let count = self.count.get(index).copied().unwrap_or(0);
        let range = if first >= NONE - 1 {
            0..0
        } else {
            slot_index(first)..slot_index(first).saturating_add(slot_index(count))
        };
        self.args.get(range).unwrap_or_default().iter().copied()
    }

    /// The match `search` finds from `start`, if any: Eden's breadth-first search's result
    /// (module docs).
    fn find(&mut self, start: Value, search: Search, work: &mut Work) -> Option<u32> {
        let s = search.slot();
        let dist = self.dist.get(s)?;
        if dist.get(slot(start)).copied().unwrap_or(NONE) == NONE {
            return None;
        }
        // Step to the rightmost argument one step nearer until a remembered match or a match.
        let mut path = Vec::new();
        let mut node = start.index();
        let answer = loop {
            work.step();
            let remembered = self
                .found
                .get(s)
                .and_then(|found| found.get(slot_index(node)))
                .copied()
                .unwrap_or(NONE);
            if remembered != NONE {
                break remembered;
            }
            let here = dist.get(slot_index(node)).copied().unwrap_or(NONE);
            if here == 0 {
                break node;
            }
            path.push(node);
            let next = self.edges(slot_index(node)).rev().find(|&arg| {
                work.step();
                dist.get(slot_index(arg)).copied().unwrap_or(NONE) == here.wrapping_sub(1)
            })?;
            node = next;
        };
        if let Some(found) = self.found.get_mut(s) {
            for node in path.into_iter().chain([node]) {
                if let Some(cell) = found.get_mut(slot_index(node)) {
                    *cell = answer;
                }
            }
        }
        Some(answer)
    }

    /// The bank and offset of a match [`Tracker::find`] returned.
    fn place(&self, node: u32) -> (u32, u32) {
        self.places
            .get(slot_index(node))
            .copied()
            .flatten()
            .unwrap_or((0, 0))
    }
}

/// An arena index as a vector index (a `u32` always fits on the hosts the crate builds for;
/// an index that does not would read past every vector and match nothing).
fn slot_index(index: u32) -> usize {
    usize::try_from(index).unwrap_or(usize::MAX)
}

/// A value's arena index as a vector index.
fn slot(value: Value) -> usize {
    slot_index(value.index())
}

/// Rewrites `block`'s list with `run`'s replacements in front of each access (module docs,
/// "Replace").
fn replace_block(
    function: &mut Function,
    block: BlockId,
    run: &[Found],
    collected_run: &[Collected],
    descriptors: &[StorageBufferDescriptor],
    alignment: StorageAlignment,
    work: &mut Work,
) -> Result<(), PassError> {
    let list = function
        .block_mut(block)
        .map_or_default(|b| std::mem::take(&mut b.insts));
    let mut out = Vec::with_capacity(
        list.len().saturating_add(
            run.len()
                .saturating_mul(usize::try_from(MADE_PER_ACCESS).unwrap_or(usize::MAX)),
        ),
    );
    let mut next = run.iter().zip(collected_run).peekable();
    for value in list {
        work.step();
        if let Some(&(found, collected)) = next.peek()
            && found.value == value
        {
            next.next();
            let binding = descriptors
                .binary_search_by_key(&(collected.bank, collected.offset), |d| (d.bank, d.offset))
                .unwrap_or(0);
            // A binding is below the number of accesses, which are instructions with 32-bit ids.
            let binding = u32::try_from(binding).unwrap_or(u32::MAX);
            let data = collected.data;
            let mut emit = |opcode: Opcode, args: Vec<Operand>| -> Result<Operand, PassError> {
                work.step();
                let new_value = function.push(block, Inst::Op(Operation::new(opcode, args)))?;
                out.push(new_value);
                Ok(Operand::Value(new_value))
            };
            let mut offset = collected.low;
            if collected.imm_offset != 0 {
                offset = emit(
                    Opcode::IAdd32,
                    vec![offset, Operand::Imm(Imm::U32(collected.imm_offset))],
                )?;
            }
            let cbuf_word = emit(
                Opcode::GetCbufU32,
                vec![
                    Operand::Imm(Imm::U32(u32::from(collected.bank))),
                    Operand::Imm(Imm::U32(collected.offset)),
                ],
            )?;
            let base = emit(
                Opcode::BitwiseAnd32,
                vec![cbuf_word, Operand::Imm(Imm::U32(alignment.mask()))],
            )?;
            let offset = emit(Opcode::ISub32, vec![offset, base])?;
            let mut args = vec![Operand::Imm(Imm::U32(binding)), offset];
            args.extend(data);
            let storage = emit(found.storage, args)?;
            if let Some(inst) = function.inst_mut(value) {
                *inst = match (found.access, storage) {
                    (Access::Write, _) => invalidated(),
                    (Access::Load | Access::Atomic, storage) => identity_of(storage),
                };
            }
        }
        out.push(value);
    }
    if let Some(b) = function.block_mut(block) {
        b.insts = out;
    }
    Ok(())
}
