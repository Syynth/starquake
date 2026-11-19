//! Slice 12e: the memory group's translator, decoded instructions of slice 6e to IR memory
//! operations by space.
//!
//! [`translate_memory`] appends one instruction's IR at a [`Builder`]'s position, and
//! [`translate_memory_word`] decodes a word first, so an unknown opcode or bit is refused with
//! the decoder's exact error ([`MemoryError::Decode`]). [`is_memory`] says which operations are
//! this group's: the loads and stores of every space (`LDG`, `LDL`, `LDS`, `STG`, `STL`, `STS`),
//! `LDC`, the atomics (`ATOMG`, `ATOMS`, `RED`, their compare-and-swap forms, `ATOMS.POPC`),
//! `CCTL`, `MEMBAR`, `ERRBAR`, `LDGDEPBAR` and `LDGSTS`; any other is
//! [`MemoryError::OtherGroup`].
//!
//! # Sources
//!
//! Ported by behaviour from PR #4588's native frontend
//! (`eden-pr4588/src/shader_recompiler/frontend/ampere/` @ e5fb3b97, GPL-3.0-or-later, Copyright
//! 2026 Eden Emulator Project): `ops_memory.cpp` (`Visitor::GlobalAddress`, `LDG`, `STG`,
//! `SharedOffset`, `LDS`, `STS`, `ATOMS`, `LDGSTS`, `MEMBAR`, `ATOMS_POPC`, `LDL`, `STL`,
//! `ATOMG`), `ops_misc.cpp` (`LDC`) and `visitor.cpp` (`Visitor::Run`: which opcode reaches which
//! translator, `ERRBAR` as a device memory barrier and `LDGDEPBAR` as a workgroup memory
//! barrier, and `CheckConsumed`). Cross-checked with Eden's Maxwell translators
//! (`eden/src/shader_recompiler/frontend/maxwell/translate/impl/` @ cf654052,
//! GPL-2.0-or-later): `load_store_memory.cpp`, `load_store_local_shared.cpp`,
//! `load_constant.cpp`, `atomic_operations_global_memory.cpp`,
//! `atomic_operations_shared_memory.cpp` and `barrier_operations.cpp` give the same IR
//! operations for the same accesses; where Eden translates more (64-bit and float atomics,
//! `ATOM` with `.CAS` refused there too), the PR's narrower set is kept (below).
//!
//! # What each instruction becomes
//!
//! - **Global addresses** (`LDG`, `STG`, `ATOMG`, `RED`, `LDGSTS`): `Visitor::GlobalAddress`. The
//!   register (a pair when its 64-bit flag is set, else the low word with a zero high word) and,
//!   when the uniform address is on, the uniform register (likewise a pair or a word) are added
//!   as two 32-bit words with an explicit carry (`IAdd32`, `ULessThan`, `SelectU32`), packed with
//!   `CompositeConstructU32x2` and `PackUint2x32`, and the sign-extended 24-bit offset is added
//!   with `IAdd64` when it is not zero. `RZ` and an unused uniform register add nothing.
//! - **Shared and local addresses**: `Visitor::SharedOffset`. With `RZ` the address is the
//!   24-bit offset field read unsigned, an immediate; with a register it is the register,
//!   shifted left by the stride code plus one when the stride is not `X1`, plus the
//!   sign-extended offset when that is not zero.
//! - **`LDG`, `STG`**: `LoadGlobal*` and `WriteGlobal*` by type; a 64- or 128-bit access moves a
//!   register pair or quad through `CompositeExtractU32x2`/`x4` and `CompositeConstructU32x2`/`x4`.
//!   `LDG`'s load predicate, where it is not `PT`, selects zero for each destination register
//!   where it is false (the load itself still runs, as in the PR).
//! - **`LDS`, `STS`**: `LoadShared*` and `WriteShared*` by type (the 8- and 16-bit stores take
//!   the unsigned opcode for either signedness, as Eden's `WriteShared(8, ..)` does).
//! - **`LDL`, `STL`**: local memory is an array of 32-bit words (`LoadLocal`, `WriteLocal`,
//!   indexed by word): the word index is the byte address shifted right arithmetically by 2.
//!   An 8- or 16-bit load extracts its byte or half from the word (`BitFieldUExtract` or
//!   `BitFieldSExtract` at the address's low bits times 8, masked to 24 or 16), and an 8- or
//!   16-bit store reads the word, inserts its bits (`BitFieldInsert`) and writes it back. A 64- or
//!   128-bit access moves two or four consecutive words.
//! - **`LDC`**: `GetCbufU8`, `GetCbufS8`, `GetCbufU16`, `GetCbufS16`, `GetCbufU32` or
//!   `GetCbufU32x2` of the bank, at the offset register plus the sign-extended 16-bit offset;
//!   `.128` is four `GetCbufU32` four bytes apart.
//! - **Atomics**: `SharedAtomic*32` and `GlobalAtomic*32` by operation (`Min` and `Max` signed for
//!   `.S32`); the old value goes to the destination unless it is `RZ` or the instruction is a
//!   `RED`. `ATOMS.POPC` is `SharedAtomicIAdd32` of 1 at the shared address with a zero offset.
//! - **Barriers**: `MEMBAR.CTA` and `LDGDEPBAR` are `WorkgroupMemoryBarrier`; `MEMBAR` at the
//!   GPU or system scope or scope 5, and `ERRBAR`, are `DeviceMemoryBarrier`.
//! - **`LDGSTS`**: four words, each `LoadGlobal32` at the global pair plus the offset plus 4
//!   times the word, selected to zero where the enable predicate is false (the load still runs,
//!   as in the PR), and `WriteSharedU32` at the shared register plus 16 times the shared offset
//!   plus 4 times the word; then a `WorkgroupMemoryBarrier`, so the copy completes before any
//!   later instruction, as the PR's comment says ("the copy completes synchronously"). A
//!   following `LDGDEPBAR` adds a second workgroup barrier
//!   (`tests::barriers::ldgsts_then_ldgdepbar_orders_the_copy_before_the_read`).
//!
//! # Guards
//!
//! An instruction under a guard predicate other than `PT` becomes a conditional block (spec,
//! "Instruction translation"), made as the integer translator makes it (`integer`'s
//! `split_for_guard`): the block tests the guard and branches to a new block holding the
//! instruction, which jumps to a second new block that takes the first block's terminator, and
//! the builder is left there. The split is made after every check, so a refused instruction
//! leaves the function untouched (`tests::guards::refusals_leave_the_function_untouched`). A
//! guarded instruction in a block with a successor whose phi names the block is refused
//! ([`MemoryError::SuccessorPhi`], as the integer translator refuses it). An instruction under
//! `!PT` never runs and appends nothing, without its checks, as PR #4588's `Visitor::Run`
//! returns when `guard.Never()`; with [`GuardMode::Tested`] the guard is ignored (flow analysis
//! has tested it). Every instruction of this group but `LDC` is a side effect in the PR's
//! `classify.h`, which flow analysis also puts in a conditional block; `LDC` is pure there and
//! keeps its guard as a select (divergence below).
//!
//! # Checks
//!
//! The checks run in this order: the group; the guard (`!PT` appends nothing); the forms PR
//! #4588's `Visitor::Run` sends to no translator or neither reference translates
//! ([`MemoryUnsupported::Instruction`] for `CCTL`, [`MemoryUnsupported::CompareAndSwap`],
//! [`MemoryUnsupported::UniformStore`], [`MemoryUnsupported::UniformSharedAtomic`],
//! [`MemoryUnsupported::BindlessConstantLoad`]); each `Require` of the PR's translator for the
//! form, in the PR's order, as a named [`MemoryUnsupported`] whose doc names the PR's message;
//! then the bits the decoder reads but the PR's translator does not consume, in bit order, as
//! `Visitor::CheckConsumed` refuses the lowest stray bit ("bit N not modelled"); then, for a
//! guarded instruction, the successor phis. Every check runs before anything is appended.
//!
//! The PR's checks the decoder makes first are not repeated: the types and atomic operations no
//! encoding names (`Require(type <= 6)`, `Require(atom_op <= 8)`), bit 91 of the forms without a
//! uniform register (`"native global load flags"` and the others), `LDG`'s bits 68..72
//! (`"native global load predicate"`), `LDC`'s bank (`cbuf_index < 32`: five bits), and the
//! fixed and dropped bits of `LDGSTS` and `ATOMS.POPC` (the decoder's module docs). The unread
//! bits mirrored here are: bit 80 of the global forms' memory order, which only the system scope
//! sets (the PR ignores bits 76..80 of `LDG`, `STG`, `ATOMG` and `RED`: [`MemoryUnsupported::SystemOrder`]);
//! `LDS`'s uniform register field and bit 87 ([`MemoryUnsupported::SharedUniformField`],
//! [`MemoryUnsupported::SharedLoadZd`]); `RED`'s destination field, uniform register field and
//! bit 90 ([`MemoryUnsupported::ReductionDestination`],
//! [`MemoryUnsupported::ReductionUniformField`], [`MemoryUnsupported::ReductionRegister64`]); and
//! `MEMBAR`'s bits 72 and 80 ([`MemoryUnsupported::BarrierMmio`],
//! [`MemoryUnsupported::BarrierSc`]).
//!
//! # Decisions taken here (the decoder's open readings)
//!
//! The decoder (slice 6e) left these readings to this slice. Each follows PR #4588's native
//! translator, the frontend the spec ports, except bit 76 (the last bullet), and each is a guess
//! until an `nvdisasm` or hardware check. The PR runs titles through its lowering to Maxwell by
//! default and through the native translator only when asked
//! (`eden-pr4588/src/video_core/shader_environment.cpp`: `EDEN_NATIVE_FRONTEND` unset keeps the
//! lowering, `lower_to_maxwell.h`), so where the two parts read a field differently and the
//! native translator would translate the word to something else, decision 8 of the 2026-11-12
//! graphics rulings ("otherwise the one a title ran on") takes the lowering's reading:
//!
//! - **The uniform-address bits.** Bit 91 turns the uniform register on (`LDG`, `STG`, `ATOMG`:
//!   `use_uniform = Take(91, 1)`). In the uniform `LDG` and `STG` the register's 64-bit flag is
//!   bit 90 and the uniform register's is bit 72 (`.E`) whether or not bit 91 is set, and in the
//!   uniform `ATOMG` the register's is bit 70; the forms without a uniform register read bit 72
//!   as the register's. `RED` is not a uniform form to the PR (`uniform_form = op == 0x9a8`): it
//!   reads bit 72 as the register's flag, refuses bit 91 and leaves bits 64..70 and 90 unread,
//!   so they must be clear. The decoder carries bit 72 of `RED` as `UniformAddr::wide`, which is
//!   read here as the register's flag
//!   (`tests::global::red_reads_its_register_pair_from_bit_72`). `LDL`, `LDS` and their uniform
//!   forms refuse bit 91 (the PR's `Require`), and the uniform `STL`, `STS` and `ATOMS`
//!   (`0x987`, `0x988`, `0x98c`) reach no translator of the PR.
//! - **The offset of a shared or local address with `RZ`** is the 24-bit field read unsigned
//!   (`Visitor::SharedOffset`: `if (reg == 255) return ir.Imm32(offset);`), where the decoder
//!   sign-extends it; the field is recovered from the decoder's `i32`
//!   (`tests::shared::an_rz_offset_is_read_unsigned`).
//! - **`LDC`'s 16-bit offset** is signed (`Visitor::LDC`: `static_cast<s16>(Take(38, 16))`),
//!   where the decoder keeps NAK's unsigned `u16`
//!   (`tests::constant::ldc_offsets_are_signed`).
//! - **Bit 76 of the uniform `LDG` and `STG`** (`0x981`, `0x986`; the decoder's `bit76`, meaning
//!   unknown there) with the uniform address on makes the uniform register a memory descriptor,
//!   which is checked as an address register would be but not added to the address: the
//!   lowering reads the word `0x0c1e1100` (bits 72, 76, 81..85, 90 and 91; the type masked) of
//!   `LDG` as a "descriptor address" whose address is the register pair alone, and `0x0c101100`
//!   of `STG` likewise (`lower_to_maxwell.h`, `descriptor_address` and `sized_store`; PR #4588's
//!   test "Native descriptor-addressed global loads retain the explicit 64-bit pointer"). The
//!   native translator ignores the bit with the memory order (`Ignore(76, 4)`) and adds the
//!   uniform register. With the uniform address off the bit changes nothing
//!   (`tests::global::bit_76_makes_the_uniform_register_a_descriptor`).
//!
//! # Constant buffers
//!
//! Banks are five bits wide, so the host model has 32 slots (bits 54..59, `trap_guards.h`; the
//! scoping doc's row 65); a decoded bank cannot name a 33rd. An indexed read, which slice 13
//! bounds-checks, is a `GetCbuf*` whose offset argument is a value rather than an immediate:
//! `LDC` with `RZ` as its offset register reads at an immediate offset (the PR adds the zero
//! register; the sum is folded here), and with any other register at a value
//! (`tests::constant::only_a_register_offset_is_indexed`).
//!
//! **Bindless handles (from the review of #201).** A bindless `LDC` (`0x582`) is refused, as PR
//! #4588 refuses it (`Require(Take(91, 1) == 0, "native bindless constant load")`, and
//! `Visitor::Run` sends `0x582` to no translator), so no handle register, odd, even or `URZ`,
//! is accepted here. Textures are traced before true bindless is admitted (decision 8 of the
//! coupled graphics rulings); when it is, NAK's reading, a 64-bit handle in an even uniform
//! register pair (NAK always passes a pair), is the one with a source. Slice 13 has no bindless
//! read to bound until then. **Owner question** for slice 13 to agree: refuse an odd or `URZ`
//! handle then, or accept it as the encoding allows.
//!
//! # Local and shared memory
//!
//! This slice emits the accesses; the private and workgroup arrays, their sizes (from the SPH
//! or the QMD) and the bounds check of every indexed access are slice 16's pass 6 (decided by
//! the owner, 2026-11-18). So `LDL` here does not select zero past the local memory size as
//! the PR does (divergence below), and nothing here reads a size.
//!
//! # Divergences from PR #4588
//!
//! | What | PR #4588 | Here | Pinned by |
//! |---|---|---|---|
//! | A guarded `LDC` | a select of each written register (`classify.h` lists `LDC` as pure) | a conditional block (spec, "Instruction translation"); the registers come out the same | `tests::guards::a_guarded_ldc_is_a_conditional_block` |
//! | `LDL` past the local memory size | each word selects 0 where the byte offset is not below `env.LocalMemorySize()` | no bound here: slice 16's pass 6 bounds every indexed local access | `tests::local::ldl_reads_without_a_bound` |
//! | `LDC` with `RZ` | `IAdd(X(255), Imm32(offset))` | the immediate offset alone (the same value) | `tests::constant::only_a_register_offset_is_indexed` |
//! | `LDG`'s load predicate | writes the registers, then rewrites each with a select of its new value | selects the loaded value before writing (the same values) | `tests::global::a_false_load_predicate_gives_zero` |
//! | The uniform `LDG` and `STG` with bit 76 and the uniform address on | the native translator adds the uniform register; the lowering does not (above) | the lowering's reading: a descriptor, not added | `tests::global::bit_76_makes_the_uniform_register_a_descriptor`, `tests::pr4588::descriptor_addressed_global_loads_retain_the_explicit_64_bit_pointer` |
//! | `LDGSTS`'s enable predicate | the native translator requires bits 85..90 clear ("native global-to-shared copy mode", `Take(85, 5) == 0`) and then reads bits 87..90 as the predicate, so only `P0` passes, against its own comment ("the enable predicate (87..89 plus the negate bit 70)"); the lowering takes any predicate (its test "Native global-to-shared copy retains its source predicate", `PT` included) | any predicate, as the decoder reads it (the lowering's reading, decision 8). **Owner question**: refuse all but `P0` as the native translator does | `tests::barriers::any_predicate_enables_a_copy` |
//!
//! # Not translated
//!
//! Refused by name, with the exact error ([`MemoryUnsupported`]): `CCTL`, which neither
//! reference translates (no case in `Visitor::Run`; Eden's Maxwell `CCTL` throws
//! `NotImplemented`); the compare-and-swap atomics `ATOMG.CAS` and `ATOMS.CAS` (`0x3a9`,
//! `0x38d`; no case in `Visitor::Run`, and Eden's IR has no compare-and-swap operation); the
//! uniform `STL`, `STS` and `ATOMS`; the bindless `LDC`; atomic types other than `U32` and
//! `S32` (the PR's `Require(type <= 1)`, where Eden's Maxwell translator also takes 64-bit and
//! float atomics); and `MEMBAR.SM` (scope 1, the PR's `Require(scope == 0 || scope == 2 ||
//! scope == 3 || scope == 5)`). `ULDC` and `ULDC.64` are slice 6d's and `LDCG` has no slice:
//! neither decodes yet, so neither reaches this module.

#[cfg(test)]
mod tests;

use std::fmt::{self, Display, Formatter};

use thiserror::Error;

use super::integer::{GuardMode, split_for_guard, successor_phi};
use crate::decode::memory::MEM_OFFSET;
use crate::decode::{
    AtomDst, AtomOp, AtomType, CBuf, DecodeError, Gpr, Instr, Instruction, LdcMode, MemAddr,
    MemBarScope, MemOrder, MemScope, MemSpace, MemType, OffsetStride, Op, OpAtom, OpAtomsPopc,
    OpLd, OpLdGsts, OpLdc, OpMemBar, OpSt, Pred, UniformReg, decode,
};
use crate::flow::read_pred;
use crate::ir::{BlockId, BuildError, Builder, Imm, Opcode, Operand};

/// A memory space, for error messages.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Space {
    /// Global memory.
    Global,
    /// Per-thread local memory.
    Local,
    /// Workgroup shared memory.
    Shared,
}

impl Space {
    /// The space's name in plain words.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Global => "global",
            Self::Local => "local",
            Self::Shared => "shared",
        }
    }

    const fn of(space: MemSpace) -> Self {
        match space {
            MemSpace::Global(_) => Self::Global,
            MemSpace::Local(_) => Self::Local,
            MemSpace::Shared(_) => Self::Shared,
        }
    }
}

impl Display for Space {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// An atomic type in plain words, for error messages.
struct TypeName(AtomType);

impl Display for TypeName {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(match self.0 {
            AtomType::U32 => "u32",
            AtomType::I32 => "s32",
            AtomType::U64 => "u64",
            AtomType::F32 => "f32",
            AtomType::F16v2 => "f16x2",
            AtomType::I64 => "s64",
            AtomType::F64 => "f64",
        })
    }
}

/// A form or field value of this group that is not translated: one of PR #4588's `Require`
/// checks (named by its message), a bit its `Visitor::CheckConsumed` refuses because its
/// translator does not read it, or a form no reference translates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum MemoryUnsupported {
    /// An instruction neither reference translates (`CCTL`: no case in `Visitor::Run`; Eden's
    /// Maxwell `CCTL` throws `NotImplemented`).
    #[error("{0}, which neither reference translates,")]
    Instruction(&'static str),
    /// A compare-and-swap atomic (`ATOMG.CAS` `0x3a9`, `ATOMS.CAS` `0x38d`): no case in
    /// `Visitor::Run`, and Eden's IR has no compare-and-swap operation.
    #[error("a {0} compare-and-swap atomic, which neither reference translates,")]
    CompareAndSwap(Space),
    /// The uniform-address form of `STL` (`0x987`) or `STS` (`0x988`), which `Visitor::Run`
    /// sends to no translator (it takes `0x387` and `0x388`).
    #[error("the uniform-address form of a {0} store")]
    UniformStore(Space),
    /// The uniform-address form of `ATOMS` (`0x98c`), which `Visitor::Run` sends to no
    /// translator (it takes `0x38c`).
    #[error("the uniform-address form of a shared atomic")]
    UniformSharedAtomic,
    /// An atomic in local memory, which no encoding has (NAK's `OpAtom`: "Atomics do not
    /// support local"); only a hand-built operation reaches it.
    #[error("a local atomic, which no encoding has,")]
    LocalAtomic,
    /// A bindless `LDC` (`0x582`; `Visitor::LDC`: "native bindless constant load", and
    /// `Visitor::Run` takes only `0xb82`).
    #[error("a bindless ldc")]
    BindlessConstantLoad,
    /// An `LDC` mode other than plain indexing (`Visitor::LDC`: "native indexed constant load
    /// mode", `Take(78, 2) == 0`).
    #[error("an ldc with an indexing mode other than plain indexing")]
    ConstantLoadMode(LdcMode),
    /// A 64- or 128-bit `LDC` whose destination is not an even register below 254 or a
    /// multiple of four below 252 (`Visitor::LDC`: "native constant load destination"; `RZ` is
    /// refused too).
    #[error("an ldc destination of {words} registers at {dst}")]
    ConstantLoadDestination {
        /// The first destination register.
        dst: Gpr,
        /// The registers written.
        words: u8,
    },
    /// An `LDG` predicate destination other than `PT` (`Visitor::LDG`: "native global load
    /// residency", `Take(81, 3) == 7`).
    #[error("an ldg predicate destination {0}")]
    LoadResidency(Pred),
    /// A global address whose uniform register is `URZ`, or whose 64-bit uniform pair does not
    /// start at an even register below `UR62` (`Visitor::GlobalAddress`: "native global uniform
    /// address", `uniform < 63 && (!uniform_pair || (uniform % 2 == 0 && uniform + 1 < 63))`).
    #[error("a global address in uniform register {}{reg}", if *.pair { "pair " } else { "" })]
    GlobalUniformAddress {
        /// The uniform register.
        reg: UniformReg,
        /// Whether it is the first of a pair.
        pair: bool,
    },
    /// A global address register pair that does not start at an even register below `R254`
    /// (`Visitor::GlobalAddress`: "native global register address", `reg % 2 == 0 && reg + 1
    /// < 255`).
    #[error("a global address register pair at {0}")]
    GlobalRegisterAddress(Gpr),
    /// A 64- or 128-bit load whose destination, other than `RZ`, does not start at a multiple
    /// of its register count and end below `RZ` ("native global load destination", "native
    /// shared load destination", "native local load destination").
    #[error("a {space} load destination of {words} registers at {dst}")]
    LoadDestination {
        /// The space.
        space: Space,
        /// The first destination register.
        dst: Gpr,
        /// The registers written.
        words: u8,
    },
    /// A 64- or 128-bit store whose data, other than `RZ`, does not start at a multiple of its
    /// register count and end below `RZ` ("native global store data", "native shared store
    /// data", "native local store data").
    #[error("{space} store data of {words} registers at {data}")]
    StoreData {
        /// The space.
        space: Space,
        /// The first data register.
        data: Gpr,
        /// The registers read.
        words: u8,
    },
    /// A shared or local load with the uniform address on (bit 91), or a local load whose
    /// uniform register field is not 0 (`Visitor::LDS`: "native shared uniform address",
    /// `Take(91, 1) == 0`; `Visitor::LDL`: "native local uniform address", `Take(32, 6) == 0 &&
    /// Take(91, 1) == 0`).
    #[error("a uniform register in a {0} address")]
    UniformAddress(Space),
    /// An atomic type other than `U32` and `S32` (`Visitor::ATOMS`: "native shared atomic
    /// type", `Visitor::ATOMG`: "native global atomic type", `type <= 1`).
    #[error("a {space} atomic of type {}", TypeName(*.atom_type))]
    AtomicType {
        /// The space.
        space: Space,
        /// The type.
        atom_type: AtomType,
    },
    /// An `ATOMG` predicate destination other than `PT` (`Visitor::ATOMG`: "native global
    /// atomic predicate output", `Take(81, 3) == 7`).
    #[error("an atomg predicate destination {0}")]
    AtomicPredicateOutput(Pred),
    /// A `RED` with the uniform address on (`Visitor::ATOMG`, which reads `RED` as a form
    /// without a uniform register: "native global atomic flags", `Take(91, 1) == 0`).
    #[error("a red with a uniform address")]
    ReductionUniformAddress,
    /// `MEMBAR.SM` (`Visitor::MEMBAR`: "native memory barrier scope", `scope == 0 || scope == 2
    /// || scope == 3 || scope == 5`).
    #[error("a membar at the sm scope")]
    BarrierScope(MemBarScope),
    /// `LDGSTS` registers out of range (`Visitor::LDGSTS`: "native global-to-shared copy
    /// registers", `global_reg % 2 == 0 && global_reg + 1 < 255 && shared_reg < 255`).
    #[error("ldgsts registers {shared} (shared) and {global} (global pair)")]
    CopyRegisters {
        /// The shared address register.
        shared: Gpr,
        /// The global address register pair.
        global: Gpr,
    },
    /// The system-scope memory order of a global access, which sets bit 80: the PR ignores bits
    /// 76..80 only (`Ignore(76, 4)`, "memory ordering"), so `CheckConsumed` refuses it.
    #[error("a global access with the system-scope memory order")]
    SystemOrder,
    /// An `LDS` whose uniform register field (bits 32..38) is not 0 with the uniform address
    /// off: `Visitor::LDS` does not read the field, so `CheckConsumed` refuses it.
    #[error("an lds uniform register field of {0}")]
    SharedUniformField(UniformReg),
    /// `LDS`'s bit 87 (`.ZD`): `Visitor::LDS` does not read it, so `CheckConsumed` refuses it.
    #[error("an lds with bit 87 set")]
    SharedLoadZd,
    /// A `RED` destination field (bits 16..24) other than `R0`: `Visitor::ATOMG` does not read it
    /// for `RED`, so `CheckConsumed` refuses it.
    #[error("a red destination field of {0}")]
    ReductionDestination(Gpr),
    /// A `RED` uniform register field (bits 64..70) other than `UR0`: `Visitor::ATOMG` does not
    /// read it for `RED`, so `CheckConsumed` refuses it.
    #[error("a red uniform register field of {0}")]
    ReductionUniformField(UniformReg),
    /// `RED`'s bit 90, the uniform `STG`'s register flag: `Visitor::ATOMG` does not read it for
    /// `RED`, so `CheckConsumed` refuses it.
    #[error("a red with bit 90 set")]
    ReductionRegister64,
    /// `MEMBAR.MMIO` (bit 72): `Visitor::MEMBAR` reads only the scope, so `CheckConsumed`
    /// refuses it.
    #[error("a membar.mmio")]
    BarrierMmio,
    /// `MEMBAR.SC` (bit 80): `Visitor::MEMBAR` reads only the scope, so `CheckConsumed` refuses
    /// it.
    #[error("a membar.sc")]
    BarrierSc,
}

/// Why a memory instruction was not translated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum MemoryError {
    /// The word does not decode.
    #[error(transparent)]
    Decode(#[from] DecodeError),
    /// The instruction belongs to another group.
    #[error("{0} is not a memory instruction")]
    OtherGroup(&'static str),
    /// The form or a field value is not translated; nothing was appended.
    #[error("{0} is not supported")]
    Unsupported(MemoryUnsupported),
    /// A guarded instruction would split `block`, whose successor `successor` has a phi naming
    /// it (see the module docs, "Guards"); nothing was appended.
    #[error("a guarded instruction cannot split {block}: {successor} has a phi naming it")]
    SuccessorPhi {
        /// The block the builder is in.
        block: BlockId,
        /// Its successor with the phi.
        successor: BlockId,
    },
    /// The IR could not be built: an arena is full, or this module built a malformed
    /// instruction (a bug). The function may hold part of the instruction's IR.
    #[error(transparent)]
    Build(#[from] BuildError),
}

impl From<MemoryUnsupported> for MemoryError {
    fn from(what: MemoryUnsupported) -> Self {
        Self::Unsupported(what)
    }
}

/// The register a register pair or quad must end below: `RZ` (`ops_memory.cpp`: `dest + 1 <
/// 255`, `dest + 3 < 255`).
pub const REGISTER_END: u8 = Gpr::RZ.index();

/// The uniform register a uniform pair must end below: `URZ` (`Visitor::GlobalAddress`:
/// `uniform + 1 < 63`).
pub const UNIFORM_REGISTER_END: u8 = UniformReg::URZ.index();

/// The register a 64-bit `LDC` destination must start below (`Visitor::LDC`: `dest < 254`).
pub const LDC_PAIR_LIMIT: u8 = 254;

/// The register a 128-bit `LDC` destination must start below (`Visitor::LDC`: `dest < 252`).
pub const LDC_QUAD_LIMIT: u8 = 252;

/// The bytes of one 32-bit word: the step between the words of a wide access.
pub const WORD_BYTES: u32 = 4;

/// The shift from a byte address to a local word index: 4 bytes a word (`Visitor::LDL`:
/// `ShiftRightArithmetic(byte_offset, Imm32(2))`).
pub const LOCAL_WORD_SHIFT: u32 = 2;

/// The shift from a byte address to its bit position within a word (`Visitor::LDL`:
/// `ShiftLeftLogical(offset, Imm32(3))`).
pub const BYTE_BIT_SHIFT: u32 = 3;

/// The bit positions an 8-bit local access can start at, as a mask of the byte address times
/// 8 (`Visitor::LDL`, `STL`: `BitwiseAnd(.., Imm32(24))`).
pub const BYTE_POSITION_MASK: u32 = 24;

/// The bit positions a 16-bit local access can start at (`Visitor::LDL`, `STL`:
/// `BitwiseAnd(.., Imm32(16))`).
pub const HALF_POSITION_MASK: u32 = 16;

/// The bytes one unit of `LDGSTS`'s shared offset spans (`Visitor::LDGSTS`: `* 16`).
pub const COPY_SHARED_UNIT: i32 = crate::decode::memory::LDGSTS_SHARED_UNIT;

/// The words one `LDGSTS` copies: 16 bytes (`Visitor::LDGSTS`: `component < 4`).
pub const COPY_WORDS: u32 = 4;

/// The bits of a shared or local offset field read unsigned with `RZ` (the decoder's 24-bit
/// [`MEM_OFFSET`]).
const OFFSET_FIELD: u32 = (1 << MEM_OFFSET.width) - 1;

/// Whether `op` belongs to this group.
pub const fn is_memory(op: &Op) -> bool {
    matches!(
        op,
        Op::Ld(_)
            | Op::St(_)
            | Op::Ldc(_)
            | Op::Atom(_)
            | Op::CCtl(_)
            | Op::MemBar(_)
            | Op::LdGsts(_)
            | Op::LdgDepBar(_)
            | Op::ErrBar(_)
            | Op::AtomsPopc(_)
    )
}

/// Decodes `word` and translates it ([`translate_memory`]).
pub fn translate_memory_word(
    builder: &mut Builder<'_>,
    word: Instruction,
    guard: GuardMode,
) -> Result<(), MemoryError> {
    let instr = decode(word)?;
    translate_memory(builder, &instr, guard)
}

/// Appends the IR of `instr`, a memory instruction, at `builder`'s position, and leaves the
/// builder where translation continues (see the module docs). The checks run in this order:
/// the group, then the guard (`!PT` appends nothing), then the forms no translator takes, then
/// PR #4588's `Require` checks in its order, then the bits its translator does not read, then a
/// guard's successor phis. A refusal appends nothing.
pub fn translate_memory(
    builder: &mut Builder<'_>,
    instr: &Instr,
    guard: GuardMode,
) -> Result<(), MemoryError> {
    if !is_memory(&instr.op) {
        return Err(MemoryError::OtherGroup(instr.op.mnemonic()));
    }
    let pending = match guard {
        GuardMode::Tested => None,
        GuardMode::Honour if instr.guard.is_true() => None,
        GuardMode::Honour if instr.guard.reg.is_true() => return Ok(()),
        GuardMode::Honour => Some(instr.guard),
    };
    check(&instr.op)?;
    let join = match pending {
        Some(guard) => {
            let block = builder.block();
            if let Some(successor) = successor_phi(builder.function(), block) {
                return Err(MemoryError::SuccessorPhi { block, successor });
            }
            Some(split_for_guard(builder, guard)?)
        }
        None => None,
    };
    Emit { b: builder }.instr(instr.op)?;
    if let Some(join) = join {
        builder.switch_to(join)?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// Checks: every refusal, in the documented order, before anything is appended.

const fn require(ok: bool, what: MemoryUnsupported) -> Result<(), MemoryUnsupported> {
    if ok { Ok(()) } else { Err(what) }
}

/// The registers a type moves: 1, or 2 and 4 for the 64- and 128-bit types.
const fn words_of(mem_type: MemType) -> u8 {
    match mem_type {
        MemType::B64 => 2,
        MemType::B128 => 4,
        MemType::U8 | MemType::I8 | MemType::U16 | MemType::I16 | MemType::B32 => 1,
    }
}

/// Whether a register run of `words` starting at `first` is valid for a wide access: `RZ`, or
/// a multiple of `words` whose last register is below `RZ` (`ops_memory.cpp`: `dest == 255 ||
/// (dest % count == 0 && dest + count - 1 < 255)`). A one-register run is always valid.
fn run_ok(first: Gpr, words: u8) -> bool {
    if words <= 1 || first.is_zero() {
        return true;
    }
    let index = u32::from(first.index());
    index.is_multiple_of(u32::from(words)) && index + u32::from(words) - 1 < u32::from(REGISTER_END)
}

/// How PR #4588 reads a global address (`Visitor::GlobalAddress`'s arguments).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct GlobalAddr {
    /// The register; `RZ` adds nothing.
    reg: Gpr,
    /// The register is a pair.
    reg_pair: bool,
    /// The uniform register and whether it is a pair, when the uniform address is on.
    uniform: Option<(UniformReg, bool)>,
    /// The uniform register is a memory descriptor, not part of the address (module docs,
    /// "Decisions taken here"): it is checked but not added.
    descriptor: bool,
    /// The sign-extended byte offset.
    offset: i32,
}

/// `addr` read as PR #4588 reads it in the forms of `LDG`, `STG` and `ATOMG` (not `RED`, see
/// [`reduction_addr`]): the decoder's fields carry the uniform and non-uniform layouts' flags
/// where the PR reads them. With `bit76` (the uniform `LDG`'s and `STG`'s bit 76) and the
/// uniform address on, the uniform register is a memory descriptor (module docs, "Decisions
/// taken here").
const fn global_addr(addr: MemAddr, bit76: bool) -> GlobalAddr {
    let uniform = match addr.uniform {
        Some(uniform) if uniform.enabled => Some((uniform.reg, uniform.wide)),
        _ => None,
    };
    GlobalAddr {
        reg: addr.reg,
        reg_pair: addr.reg_64,
        uniform,
        descriptor: bit76 && uniform.is_some(),
        offset: addr.offset,
    }
}

/// A `RED`'s address as PR #4588 reads it: the register's flag is bit 72, which the decoder
/// carries as the uniform register's `.E`; there is no uniform register.
const fn reduction_addr(addr: MemAddr) -> GlobalAddr {
    GlobalAddr {
        reg: addr.reg,
        reg_pair: match addr.uniform {
            Some(uniform) => uniform.wide,
            None => addr.reg_64,
        },
        uniform: None,
        descriptor: false,
        offset: addr.offset,
    }
}

/// `Visitor::GlobalAddress`'s checks: the uniform register, then the register.
fn check_global_addr(addr: GlobalAddr) -> Result<(), MemoryUnsupported> {
    if let Some((reg, pair)) = addr.uniform {
        let index = u32::from(reg.index());
        let end = u32::from(UNIFORM_REGISTER_END);
        require(
            index < end && (!pair || (index.is_multiple_of(2) && index + 1 < end)),
            MemoryUnsupported::GlobalUniformAddress { reg, pair },
        )?;
    }
    if !addr.reg.is_zero() && addr.reg_pair {
        let index = u32::from(addr.reg.index());
        require(
            index.is_multiple_of(2) && index + 1 < u32::from(REGISTER_END),
            MemoryUnsupported::GlobalRegisterAddress(addr.reg),
        )?;
    }
    Ok(())
}

/// The bit-80 check of a global access's order (module docs, "Checks").
const fn check_order(space: MemSpace) -> Result<(), MemoryUnsupported> {
    match space {
        MemSpace::Global(access) => require(
            !matches!(access.order, MemOrder::Strong(MemScope::System)),
            MemoryUnsupported::SystemOrder,
        ),
        MemSpace::Local(_) | MemSpace::Shared(_) => Ok(()),
    }
}

fn check(op: &Op) -> Result<(), MemoryUnsupported> {
    match *op {
        Op::Ld(ld) => check_ld(ld),
        Op::St(st) => check_st(st),
        Op::Ldc(ldc) => check_ldc(ldc),
        Op::Atom(atom) => check_atom(atom),
        Op::MemBar(bar) => check_membar(bar),
        Op::LdGsts(copy) => check_ldgsts(copy),
        Op::CCtl(_) => Err(MemoryUnsupported::Instruction(op.mnemonic())),
        _ => Ok(()),
    }
}

fn check_ld(ld: OpLd) -> Result<(), MemoryUnsupported> {
    let space = Space::of(ld.space);
    let words = words_of(ld.mem_type);
    let destination = MemoryUnsupported::LoadDestination {
        space,
        dst: ld.dst,
        words,
    };
    match ld.space {
        MemSpace::Global(_) => {
            require(
                ld.pred_dst.is_true(),
                MemoryUnsupported::LoadResidency(ld.pred_dst),
            )?;
            check_global_addr(global_addr(ld.addr, ld.bit76))?;
            require(run_ok(ld.dst, words), destination)?;
            check_order(ld.space)
        }
        MemSpace::Shared(_) => {
            let uniform = ld.addr.uniform;
            require(
                !uniform.is_some_and(|uniform| uniform.enabled),
                MemoryUnsupported::UniformAddress(space),
            )?;
            require(run_ok(ld.dst, words), destination)?;
            if let Some(uniform) = uniform {
                require(
                    uniform.reg.index() == 0,
                    MemoryUnsupported::SharedUniformField(uniform.reg),
                )?;
            }
            require(!ld.zd, MemoryUnsupported::SharedLoadZd)
        }
        MemSpace::Local(_) => {
            require(
                !ld.addr
                    .uniform
                    .is_some_and(|uniform| uniform.enabled || uniform.reg.index() != 0),
                MemoryUnsupported::UniformAddress(space),
            )?;
            require(run_ok(ld.dst, words), destination)
        }
    }
}

fn check_st(st: OpSt) -> Result<(), MemoryUnsupported> {
    let space = Space::of(st.space);
    let words = words_of(st.mem_type);
    let data = MemoryUnsupported::StoreData {
        space,
        data: st.data,
        words,
    };
    match st.space {
        MemSpace::Global(_) => {
            check_global_addr(global_addr(st.addr, st.bit76))?;
            require(run_ok(st.data, words), data)?;
            check_order(st.space)
        }
        MemSpace::Local(_) | MemSpace::Shared(_) => {
            require(
                st.addr.uniform.is_none(),
                MemoryUnsupported::UniformStore(space),
            )?;
            require(run_ok(st.data, words), data)
        }
    }
}

fn check_ldc(ldc: OpLdc) -> Result<(), MemoryUnsupported> {
    require(
        matches!(ldc.cbuf.buf, CBuf::Binding(_)),
        MemoryUnsupported::BindlessConstantLoad,
    )?;
    require(
        ldc.mode == LdcMode::Indexed,
        MemoryUnsupported::ConstantLoadMode(ldc.mode),
    )?;
    let words = words_of(ldc.mem_type);
    let index = ldc.dst.index();
    let ok = match ldc.mem_type {
        MemType::B64 => index.is_multiple_of(2) && index < LDC_PAIR_LIMIT,
        MemType::B128 => index.is_multiple_of(4) && index < LDC_QUAD_LIMIT,
        MemType::U8 | MemType::I8 | MemType::U16 | MemType::I16 | MemType::B32 => true,
    };
    require(
        ok,
        MemoryUnsupported::ConstantLoadDestination {
            dst: ldc.dst,
            words,
        },
    )
}

fn check_atom(atom: OpAtom) -> Result<(), MemoryUnsupported> {
    let space = Space::of(atom.space);
    require(
        !matches!(atom.op, AtomOp::CmpExch(_)),
        MemoryUnsupported::CompareAndSwap(space),
    )?;
    let atom_type = MemoryUnsupported::AtomicType {
        space,
        atom_type: atom.atom_type,
    };
    let narrow = matches!(atom.atom_type, AtomType::U32 | AtomType::I32);
    match (atom.space, atom.dst) {
        (MemSpace::Global(_), AtomDst::Reg(_)) => {
            require(
                atom.pred_dst.is_true(),
                MemoryUnsupported::AtomicPredicateOutput(atom.pred_dst),
            )?;
            require(narrow, atom_type)?;
            check_global_addr(global_addr(atom.addr, false))?;
            check_order(atom.space)
        }
        (MemSpace::Global(_), AtomDst::Reduction { ignored }) => {
            let uniform = atom.addr.uniform;
            require(
                !uniform.is_some_and(|uniform| uniform.enabled),
                MemoryUnsupported::ReductionUniformAddress,
            )?;
            require(narrow, atom_type)?;
            check_global_addr(reduction_addr(atom.addr))?;
            require(
                ignored.index() == 0,
                MemoryUnsupported::ReductionDestination(ignored),
            )?;
            if let Some(uniform) = uniform {
                require(
                    uniform.reg.index() == 0,
                    MemoryUnsupported::ReductionUniformField(uniform.reg),
                )?;
            }
            check_order(atom.space)?;
            require(!atom.addr.reg_64, MemoryUnsupported::ReductionRegister64)
        }
        (MemSpace::Shared(_), _) => {
            require(
                atom.addr.uniform.is_none(),
                MemoryUnsupported::UniformSharedAtomic,
            )?;
            require(narrow, atom_type)
        }
        (MemSpace::Local(_), _) => Err(MemoryUnsupported::LocalAtomic),
    }
}

fn check_membar(bar: OpMemBar) -> Result<(), MemoryUnsupported> {
    require(
        bar.scope != MemBarScope::Sm,
        MemoryUnsupported::BarrierScope(bar.scope),
    )?;
    require(!bar.mmio, MemoryUnsupported::BarrierMmio)?;
    require(!bar.sc, MemoryUnsupported::BarrierSc)
}

/// `Visitor::LDGSTS`'s register check. Its `Require(Take(85, 5) == 0)` is not mirrored: it
/// leaves only `P0` as the enable predicate (module docs, "Divergences from PR #4588").
fn check_ldgsts(copy: OpLdGsts) -> Result<(), MemoryUnsupported> {
    let global = u32::from(copy.global.index());
    require(
        global.is_multiple_of(2)
            && global + 1 < u32::from(REGISTER_END)
            && copy.shared.index() < REGISTER_END,
        MemoryUnsupported::CopyRegisters {
            shared: copy.shared,
            global: copy.global,
        },
    )
}

// ---------------------------------------------------------------------------------------------
// Emission: every check has passed.

/// A 32-bit immediate operand.
const fn imm32(bits: u32) -> Operand {
    Operand::Imm(Imm::U32(bits))
}

/// The register `k` places after `reg`; the checks keep every register a form reads below
/// `RZ`.
fn next(reg: Gpr, k: u8) -> Gpr {
    Gpr::new(reg.index().saturating_add(k))
}

/// The left shift that scales a shared address register by its stride: the stride code plus
/// one (`Visitor::SharedOffset`: `ShiftLeftLogical(base, Imm32(stride + 1))`), none for `X1`.
const fn stride_shift(stride: OffsetStride) -> Option<u32> {
    match stride {
        OffsetStride::X1 => None,
        OffsetStride::X4 => Some(2),
        OffsetStride::X8 => Some(3),
        OffsetStride::X16 => Some(4),
    }
}

/// The opcodes of one type's access in one space.
struct Access {
    load: Opcode,
    store: Opcode,
}

const fn global_access(mem_type: MemType) -> Access {
    let (load, store) = match mem_type {
        MemType::U8 => (Opcode::LoadGlobalU8, Opcode::WriteGlobalU8),
        MemType::I8 => (Opcode::LoadGlobalS8, Opcode::WriteGlobalS8),
        MemType::U16 => (Opcode::LoadGlobalU16, Opcode::WriteGlobalU16),
        MemType::I16 => (Opcode::LoadGlobalS16, Opcode::WriteGlobalS16),
        MemType::B32 => (Opcode::LoadGlobal32, Opcode::WriteGlobal32),
        MemType::B64 => (Opcode::LoadGlobal64, Opcode::WriteGlobal64),
        MemType::B128 => (Opcode::LoadGlobal128, Opcode::WriteGlobal128),
    };
    Access { load, store }
}

/// Shared memory: the 8- and 16-bit stores take the unsigned opcode for either signedness
/// (Eden's `WriteShared(8, ..)` has no signed form).
const fn shared_access(mem_type: MemType) -> Access {
    let (load, store) = match mem_type {
        MemType::U8 => (Opcode::LoadSharedU8, Opcode::WriteSharedU8),
        MemType::I8 => (Opcode::LoadSharedS8, Opcode::WriteSharedU8),
        MemType::U16 => (Opcode::LoadSharedU16, Opcode::WriteSharedU16),
        MemType::I16 => (Opcode::LoadSharedS16, Opcode::WriteSharedU16),
        MemType::B32 => (Opcode::LoadSharedU32, Opcode::WriteSharedU32),
        MemType::B64 => (Opcode::LoadSharedU64, Opcode::WriteSharedU64),
        MemType::B128 => (Opcode::LoadSharedU128, Opcode::WriteSharedU128),
    };
    Access { load, store }
}

/// The 32-bit atomic of `op` in `space` (signed `Min` and `Max` for `S32`).
const fn atomic(space: Space, op: AtomOp, signed: bool) -> Opcode {
    let shared = matches!(space, Space::Shared);
    match (op, shared) {
        (AtomOp::Add, true) => Opcode::SharedAtomicIAdd32,
        (AtomOp::Min, true) if signed => Opcode::SharedAtomicSMin32,
        (AtomOp::Min, true) => Opcode::SharedAtomicUMin32,
        (AtomOp::Max, true) if signed => Opcode::SharedAtomicSMax32,
        (AtomOp::Max, true) => Opcode::SharedAtomicUMax32,
        (AtomOp::Inc, true) => Opcode::SharedAtomicInc32,
        (AtomOp::Dec, true) => Opcode::SharedAtomicDec32,
        (AtomOp::And, true) => Opcode::SharedAtomicAnd32,
        (AtomOp::Or, true) => Opcode::SharedAtomicOr32,
        (AtomOp::Xor, true) => Opcode::SharedAtomicXor32,
        (AtomOp::Exch | AtomOp::CmpExch(_), true) => Opcode::SharedAtomicExchange32,
        (AtomOp::Add, false) => Opcode::GlobalAtomicIAdd32,
        (AtomOp::Min, false) if signed => Opcode::GlobalAtomicSMin32,
        (AtomOp::Min, false) => Opcode::GlobalAtomicUMin32,
        (AtomOp::Max, false) if signed => Opcode::GlobalAtomicSMax32,
        (AtomOp::Max, false) => Opcode::GlobalAtomicUMax32,
        (AtomOp::Inc, false) => Opcode::GlobalAtomicInc32,
        (AtomOp::Dec, false) => Opcode::GlobalAtomicDec32,
        (AtomOp::And, false) => Opcode::GlobalAtomicAnd32,
        (AtomOp::Or, false) => Opcode::GlobalAtomicOr32,
        (AtomOp::Xor, false) => Opcode::GlobalAtomicXor32,
        (AtomOp::Exch | AtomOp::CmpExch(_), false) => Opcode::GlobalAtomicExchange32,
    }
}

/// Appends one checked instruction's IR.
struct Emit<'b, 'f> {
    b: &'b mut Builder<'f>,
}

impl Emit<'_, '_> {
    fn op(&mut self, opcode: Opcode, args: &[Operand]) -> Result<Operand, BuildError> {
        Ok(self.b.op(opcode, args)?.into())
    }

    /// Reads a register; `RZ` reads zero (`Visitor::X`).
    fn get(&mut self, reg: Gpr) -> Result<Operand, BuildError> {
        if reg.is_zero() {
            return Ok(imm32(0));
        }
        self.op(Opcode::GetRegister, &[Operand::Imm(Imm::Reg(reg))])
    }

    /// Writes a register; a write to `RZ` is dropped (`Visitor::X`).
    fn set(&mut self, reg: Gpr, value: Operand) -> Result<(), BuildError> {
        if !reg.is_zero() {
            self.op(Opcode::SetRegister, &[Operand::Imm(Imm::Reg(reg)), value])?;
        }
        Ok(())
    }

    /// Reads a uniform register; `URZ` reads zero (`Visitor::UR`).
    fn uniform(&mut self, reg: UniformReg) -> Result<Operand, BuildError> {
        if reg.is_zero() {
            return Ok(imm32(0));
        }
        self.op(
            Opcode::GetUniformRegister,
            &[Operand::Imm(Imm::UniformReg(reg))],
        )
    }

    /// `base + k`, without an addition for 0; an immediate base is added at once.
    fn add(&mut self, base: Operand, k: u32) -> Result<Operand, BuildError> {
        match base {
            _ if k == 0 => Ok(base),
            Operand::Imm(Imm::U32(bits)) => Ok(imm32(bits.wrapping_add(k))),
            _ => self.op(Opcode::IAdd32, &[base, imm32(k)]),
        }
    }

    fn instr(&mut self, op: Op) -> Result<(), BuildError> {
        match op {
            Op::Ld(ld) => self.ld(ld),
            Op::St(st) => self.st(st),
            Op::Ldc(ldc) => self.ldc(ldc),
            Op::Atom(atom) => self.atom(atom),
            Op::AtomsPopc(popc) => self.atoms_popc(popc),
            Op::MemBar(bar) => {
                let opcode = match bar.scope {
                    MemBarScope::Cta => Opcode::WorkgroupMemoryBarrier,
                    MemBarScope::Sm
                    | MemBarScope::Gpu
                    | MemBarScope::System
                    | MemBarScope::Scope5 => Opcode::DeviceMemoryBarrier,
                };
                self.op(opcode, &[]).map(drop)
            }
            Op::ErrBar(_) => self.op(Opcode::DeviceMemoryBarrier, &[]).map(drop),
            Op::LdgDepBar(_) => self.op(Opcode::WorkgroupMemoryBarrier, &[]).map(drop),
            Op::LdGsts(copy) => self.ldgsts(copy),
            _ => Ok(()),
        }
    }

    /// `Visitor::GlobalAddress`: the 64-bit address.
    fn global_address(&mut self, addr: GlobalAddr) -> Result<Operand, BuildError> {
        let zero = imm32(0);
        let have_reg = !addr.reg.is_zero();
        let uniform = if addr.descriptor { None } else { addr.uniform };
        let (low, high) = match uniform {
            Some((reg, pair)) => {
                let u_low = self.uniform(reg)?;
                let u_high = if pair {
                    let next =
                        UniformReg::new(reg.index().saturating_add(1)).unwrap_or(UniformReg::URZ);
                    self.uniform(next)?
                } else {
                    zero
                };
                if have_reg {
                    let r_low = self.get(addr.reg)?;
                    let low = self.op(Opcode::IAdd32, &[u_low, r_low])?;
                    let carry = self.op(Opcode::ULessThan, &[low, u_low])?;
                    let r_high = if addr.reg_pair {
                        self.get(next(addr.reg, 1))?
                    } else {
                        zero
                    };
                    let sum = self.op(Opcode::IAdd32, &[u_high, r_high])?;
                    let carried = self.op(Opcode::SelectU32, &[carry, imm32(1), zero])?;
                    (low, self.op(Opcode::IAdd32, &[sum, carried])?)
                } else {
                    (u_low, u_high)
                }
            }
            None if have_reg => {
                let low = self.get(addr.reg)?;
                let high = if addr.reg_pair {
                    self.get(next(addr.reg, 1))?
                } else {
                    zero
                };
                (low, high)
            }
            None => (zero, zero),
        };
        let pair = self.op(Opcode::CompositeConstructU32x2, &[low, high])?;
        let address = self.op(Opcode::PackUint2x32, &[pair])?;
        if addr.offset == 0 {
            return Ok(address);
        }
        let offset = i64::from(addr.offset).cast_unsigned();
        self.op(Opcode::IAdd64, &[address, Operand::Imm(Imm::U64(offset))])
    }

    /// `Visitor::SharedOffset`: the byte address of a shared or local access.
    fn shared_offset(
        &mut self,
        reg: Gpr,
        stride: OffsetStride,
        offset: i32,
    ) -> Result<Operand, BuildError> {
        if reg.is_zero() {
            return Ok(imm32(offset.cast_unsigned() & OFFSET_FIELD));
        }
        let mut base = self.get(reg)?;
        if let Some(shift) = stride_shift(stride) {
            base = self.op(Opcode::ShiftLeftLogical32, &[base, imm32(shift)])?;
        }
        self.add(base, offset.cast_unsigned())
    }

    /// Writes the words of a wide value to `dst` onwards: one word as it is, a pair or quad
    /// extracted; nothing for `RZ`.
    fn set_words(&mut self, dst: Gpr, value: Operand, words: u8) -> Result<(), BuildError> {
        let extract = match words {
            2 => Opcode::CompositeExtractU32x2,
            4 => Opcode::CompositeExtractU32x4,
            _ => return self.set(dst, value),
        };
        if dst.is_zero() {
            return Ok(());
        }
        for k in 0..words {
            let word = self.op(extract, &[value, imm32(u32::from(k))])?;
            self.set(next(dst, k), word)?;
        }
        Ok(())
    }

    /// The words of a wide value read from `data` onwards: one word, or a pair or quad built;
    /// `RZ` gives zeros.
    fn get_words(&mut self, data: Gpr, words: u8) -> Result<Operand, BuildError> {
        let construct = match words {
            2 => Opcode::CompositeConstructU32x2,
            4 => Opcode::CompositeConstructU32x4,
            _ => return self.get(data),
        };
        let mut parts = Vec::with_capacity(usize::from(words));
        for k in 0..words {
            let reg = if data.is_zero() { data } else { next(data, k) };
            parts.push(self.get(reg)?);
        }
        self.op(construct, &parts)
    }

    fn ld(&mut self, ld: OpLd) -> Result<(), BuildError> {
        let words = words_of(ld.mem_type);
        match ld.space {
            MemSpace::Global(_) => {
                let address = self.global_address(global_addr(ld.addr, ld.bit76))?;
                let value = self.op(global_access(ld.mem_type).load, &[address])?;
                if ld.pred.is_true() || ld.dst.is_zero() {
                    return self.set_words(ld.dst, value, words);
                }
                let enabled = read_pred(self.b, ld.pred)?;
                let mut parts = Vec::with_capacity(usize::from(words));
                if words == 1 {
                    parts.push(value);
                } else {
                    let extract = if words == 2 {
                        Opcode::CompositeExtractU32x2
                    } else {
                        Opcode::CompositeExtractU32x4
                    };
                    for k in 0..words {
                        parts.push(self.op(extract, &[value, imm32(u32::from(k))])?);
                    }
                }
                for (k, part) in (0..).zip(parts) {
                    let selected = self.op(Opcode::SelectU32, &[enabled, part, imm32(0)])?;
                    self.set(next(ld.dst, k), selected)?;
                }
                Ok(())
            }
            MemSpace::Shared(stride) => {
                let address = self.shared_offset(ld.addr.reg, stride, ld.addr.offset)?;
                let value = self.op(shared_access(ld.mem_type).load, &[address])?;
                self.set_words(ld.dst, value, words)
            }
            MemSpace::Local(_) => self.ldl(ld),
        }
    }

    /// `Visitor::LDL`, without the bound (module docs, "Local and shared memory").
    fn ldl(&mut self, ld: OpLd) -> Result<(), BuildError> {
        let offset = self.shared_offset(ld.addr.reg, OffsetStride::X1, ld.addr.offset)?;
        let field = match ld.mem_type {
            MemType::U8 | MemType::I8 => Some((BYTE_POSITION_MASK, 8, ld.mem_type == MemType::I8)),
            MemType::U16 | MemType::I16 => {
                Some((HALF_POSITION_MASK, 16, ld.mem_type == MemType::I16))
            }
            MemType::B32 | MemType::B64 | MemType::B128 => None,
        };
        if let Some((mask, width, signed)) = field {
            let word = self.local_word(offset)?;
            let value = self.op(Opcode::LoadLocal, &[word])?;
            let bits = self.op(Opcode::ShiftLeftLogical32, &[offset, imm32(BYTE_BIT_SHIFT)])?;
            let bit = self.op(Opcode::BitwiseAnd32, &[bits, imm32(mask)])?;
            let extract = if signed {
                Opcode::BitFieldSExtract
            } else {
                Opcode::BitFieldUExtract
            };
            let field = self.op(extract, &[value, bit, imm32(width)])?;
            return self.set(ld.dst, field);
        }
        for k in 0..words_of(ld.mem_type) {
            let byte = self.add(offset, u32::from(k) * WORD_BYTES)?;
            let word = self.local_word(byte)?;
            let value = self.op(Opcode::LoadLocal, &[word])?;
            if !ld.dst.is_zero() {
                self.set(next(ld.dst, k), value)?;
            }
        }
        Ok(())
    }

    /// The local word index of byte address `byte` (`ShiftRightArithmetic(byte, Imm32(2))`).
    fn local_word(&mut self, byte: Operand) -> Result<Operand, BuildError> {
        self.op(
            Opcode::ShiftRightArithmetic32,
            &[byte, imm32(LOCAL_WORD_SHIFT)],
        )
    }

    fn st(&mut self, st: OpSt) -> Result<(), BuildError> {
        let words = words_of(st.mem_type);
        match st.space {
            MemSpace::Global(_) => {
                let address = self.global_address(global_addr(st.addr, st.bit76))?;
                let value = self.get_words(st.data, words)?;
                self.op(global_access(st.mem_type).store, &[address, value])
                    .map(drop)
            }
            MemSpace::Shared(stride) => {
                let address = self.shared_offset(st.addr.reg, stride, st.addr.offset)?;
                let value = self.get_words(st.data, words)?;
                self.op(shared_access(st.mem_type).store, &[address, value])
                    .map(drop)
            }
            MemSpace::Local(_) => self.stl(st),
        }
    }

    /// `Visitor::STL`.
    fn stl(&mut self, st: OpSt) -> Result<(), BuildError> {
        let offset = self.shared_offset(st.addr.reg, OffsetStride::X1, st.addr.offset)?;
        let word = self.local_word(offset)?;
        let field = match st.mem_type {
            MemType::U8 | MemType::I8 => Some((BYTE_POSITION_MASK, 8)),
            MemType::U16 | MemType::I16 => Some((HALF_POSITION_MASK, 16)),
            MemType::B32 | MemType::B64 | MemType::B128 => None,
        };
        if let Some((mask, width)) = field {
            let bits = self.op(Opcode::ShiftLeftLogical32, &[offset, imm32(BYTE_BIT_SHIFT)])?;
            let bit = self.op(Opcode::BitwiseAnd32, &[bits, imm32(mask)])?;
            let old = self.op(Opcode::LoadLocal, &[word])?;
            let data = self.get(st.data)?;
            let merged = self.op(Opcode::BitFieldInsert, &[old, data, bit, imm32(width)])?;
            return self.op(Opcode::WriteLocal, &[word, merged]).map(drop);
        }
        for k in 0..words_of(st.mem_type) {
            let index = self.add(word, u32::from(k))?;
            let reg = if st.data.is_zero() {
                st.data
            } else {
                next(st.data, k)
            };
            let value = self.get(reg)?;
            self.op(Opcode::WriteLocal, &[index, value])?;
        }
        Ok(())
    }

    /// `Visitor::LDC`: the offset register plus the signed offset (folded for `RZ`).
    fn ldc(&mut self, ldc: OpLdc) -> Result<(), BuildError> {
        let CBuf::Binding(bank) = ldc.cbuf.buf else {
            return Ok(());
        };
        let bank = imm32(u32::from(bank));
        let displacement = i32::from(ldc.cbuf.offset.cast_signed()).cast_unsigned();
        let offset = if ldc.offset.is_zero() {
            imm32(displacement)
        } else {
            let reg = self.get(ldc.offset)?;
            self.op(Opcode::IAdd32, &[reg, imm32(displacement)])?
        };
        let read = match ldc.mem_type {
            MemType::U8 => Opcode::GetCbufU8,
            MemType::I8 => Opcode::GetCbufS8,
            MemType::U16 => Opcode::GetCbufU16,
            MemType::I16 => Opcode::GetCbufS16,
            MemType::B32 => Opcode::GetCbufU32,
            MemType::B64 => Opcode::GetCbufU32x2,
            MemType::B128 => {
                for k in 0..4_u8 {
                    let at = self.add(offset, u32::from(k) * WORD_BYTES)?;
                    let word = self.op(Opcode::GetCbufU32, &[bank, at])?;
                    self.set(next(ldc.dst, k), word)?;
                }
                return Ok(());
            }
        };
        let value = self.op(read, &[bank, offset])?;
        self.set_words(ldc.dst, value, words_of(ldc.mem_type))
    }

    fn atom(&mut self, atom: OpAtom) -> Result<(), BuildError> {
        let space = Space::of(atom.space);
        let signed = atom.atom_type == AtomType::I32;
        let address = match (atom.space, atom.dst) {
            (MemSpace::Global(_), AtomDst::Reg(_)) => {
                self.global_address(global_addr(atom.addr, false))?
            }
            (MemSpace::Global(_), AtomDst::Reduction { .. }) => {
                self.global_address(reduction_addr(atom.addr))?
            }
            (MemSpace::Shared(stride), _) => {
                self.shared_offset(atom.addr.reg, stride, atom.addr.offset)?
            }
            // Refused by the checks.
            (MemSpace::Local(_), _) => return Ok(()),
        };
        let value = self.get(atom.data)?;
        let old = self.op(atomic(space, atom.op, signed), &[address, value])?;
        if let AtomDst::Reg(dst) = atom.dst {
            self.set(dst, old)?;
        }
        Ok(())
    }

    /// `Visitor::ATOMS_POPC`: one added to the shared word, at the address with no offset.
    fn atoms_popc(&mut self, popc: OpAtomsPopc) -> Result<(), BuildError> {
        let address = self.shared_offset(popc.addr, popc.stride, 0)?;
        self.op(Opcode::SharedAtomicIAdd32, &[address, imm32(1)])
            .map(drop)
    }

    /// `Visitor::LDGSTS`: four words copied, then a workgroup memory barrier.
    fn ldgsts(&mut self, copy: OpLdGsts) -> Result<(), BuildError> {
        let enabled = read_pred(self.b, copy.enable)?;
        let global_offset = i32::from(copy.global_offset);
        let shared_offset = i32::from(copy.shared_offset).saturating_mul(COPY_SHARED_UNIT);
        let mut step = 0_i32;
        for _ in 0..COPY_WORDS {
            let address = self.global_address(GlobalAddr {
                reg: copy.global,
                reg_pair: true,
                uniform: None,
                descriptor: false,
                offset: global_offset.saturating_add(step),
            })?;
            let loaded = self.op(Opcode::LoadGlobal32, &[address])?;
            let value = self.op(Opcode::SelectU32, &[enabled, loaded, imm32(0)])?;
            let base = self.get(copy.shared)?;
            let target = self.add(base, shared_offset.saturating_add(step).cast_unsigned())?;
            self.op(Opcode::WriteSharedU32, &[target, value])?;
            step = step.saturating_add(WORD_BYTES.cast_signed());
        }
        self.op(Opcode::WorkgroupMemoryBarrier, &[]).map(drop)
    }
}
