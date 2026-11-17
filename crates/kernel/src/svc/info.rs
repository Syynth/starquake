//! `GetInfo` (0x29): facts about a process. This slice answers the ids a libnx program asks at
//! start-up (the alias, heap, ASLR and stack regions, the alias region's extra size, the total
//! and used memory), the random entropy that rtld asks for after its module walk, and whether a
//! debugger is attached.
//!
//! Ported from Eden `eden/src/core/hle/kernel/svc/svc_info.cpp` @ cf654052 (GPL-3.0-or-later),
//! `GetInfo`; the ids are `InfoType` in `eden/src/core/hle/kernel/svc_types.h` @ cf654052.
//! Eden's checks, in its order, for each group of ids answered here:
//!
//! - **The memory ids** (2 to 7, 12 to 15 and 28, in the first `case` group of Eden's
//!   `switch`):
//!   1. `info_sub_id == 0`, else `ResultInvalidEnumValue`.
//!   2. The handle names a process (`GetObject<KProcess>`: the current-process pseudo-handle,
//!      or a handle to a process), else `ResultInvalidHandle`.
//!
//!   Then the value is read from that process's page table, or (ids 6 and 7) from the process.
//! - **`DebuggerAttached`** (8): Eden checks neither the handle nor the sub id; the value is
//!   zero (`*result = 0`). No debugger is ever attached here, so the value is the fact, not a
//!   stub. Eden, and so starquake, is looser than Horizon here: Horizon checks
//!   `handle == InvalidHandle`, else `ResultInvalidHandle`, and then `info_subtype == 0`, else
//!   `ResultInvalidCombination` (fact only,
//!   `atmosphere/libraries/libmesosphere/source/svc/kern_svc_info.cpp` @ 4e3f3703,
//!   `InfoType_DebuggerAttached`). Test: `a_debugger_is_never_attached_whatever_is_passed`.
//! - **`RandomEntropy`** (11):
//!   1. `handle == 0` (the invalid handle, not a process), else `ResultInvalidHandle`.
//!   2. `info_sub_id < 4` ([`ENTROPY_COUNT`]), else `ResultInvalidCombination`.
//!
//!   Then the value is word `info_sub_id` of the calling process's entropy
//!   (`GetCurrentProcess(...).GetRandomEntropy(info_sub_id)`).
//!
//! PR #4588 (`eden-pr4588/src/core/hle/kernel/svc/svc_info.cpp` @ e5fb3b97) changes none of
//! these, so a Switch 2 program is answered the same.
//!
//! # Divergences from Eden
//!
//! - **Ids Eden answers that this slice does not** return [`KernelError::NotImplemented`]
//!   and are logged: the core and priority masks, the resource-limit and tick-count ids, the
//!   system-resource sizes, the program id, the initial process id range, the exception
//!   context address, the non-system memory sizes, `IsApplication`, the free thread count,
//!   ids 37 and 38, and Atmosphère's two. Eden answers each. They belong to the slices that
//!   build what they report (process creation, threads, resource limits). Test:
//!   `an_id_this_slice_does_not_answer_is_not_implemented_whatever_else_is_passed`.
//! - **The Switch 2 ids 29 to 33** (`eden-pr4588/src/core/hle/kernel/svc/svc_info.cpp` @
//!   e5fb3b97; scoping doc, "New `svcGetInfo` ids") are not built either and return
//!   [`KernelError::NotImplemented`]. PR #4588 answers them only for a Switch 2 program; Eden
//!   returns `ResultInvalidEnumValue`. Same test.
//! - **Any process handle names the one process there is.** With one process, a handle to a
//!   process object and the pseudo-handle are the same process (kernel slice 8 makes
//!   processes rows of a table). Test: `a_handle_must_name_a_process`.
//! - **The entropy is drawn when it is first asked for**, not when the process is created
//!   (Eden's `KProcess::Initialize`, `GenerateRandom(m_entropy)`): the dispatcher draws the
//!   [`ENTROPY_COUNT`] words from the environment's random numbers at the first
//!   `RandomEntropy` call, whatever its handle and sub id, and keeps them in the process's
//!   row ([`crate::process::ProcessRecord::entropy`]). Creating a process has no route to the
//!   environment yet. The guest cannot tell: the words are random either way, and never change
//!   once drawn. Test (in `dispatch/tests/info_and_break.rs`):
//!   `the_entropy_is_drawn_once_at_the_first_ask_and_kept`. A process whose entropy was not
//!   drawn is an internal failure here, not a value. Test:
//!   `entropy_that_was_never_drawn_is_an_internal_failure_after_the_checks`.
//!
//! Ids Eden does not list (`IsSvcPermitted` 26, `IoRegionHint` 27, `TransferMemoryHint` 34 and
//! every number that is not an `InfoType`) return [`KernelError::InvalidEnumValue`], as Eden's
//! `default` does.

use crate::handle::{Handle, HandleTable, Resolved};
use crate::id::ProcessId;
use crate::memory::AddressSpace;
use crate::object::Object;
use crate::process::{Console, ENTROPY_COUNT, ProcessInfo};
use crate::result::{CallResult, KernelError, SvcError};
use crate::tables::{Processes, Scheduler};
use starquake_memory::PAGE_SIZE;

/// Defines [`InfoType`] from `Name = number;` rows, so an id and its number are written once.
macro_rules! info_types {
    ($($(#[$doc:meta])* $name:ident = $number:literal;)*) => {
        /// What `GetInfo` is asked for (`Svc::InfoType`): the ids Eden answers, and the
        /// Switch 2 ids 29 to 33.
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub enum InfoType {
            $($(#[$doc])* $name,)*
        }

        impl InfoType {
            /// Every id, in number order.
            pub const ALL: &'static [Self] = &[$(Self::$name,)*];

            /// The id's number.
            #[must_use]
            pub const fn number(self) -> u32 {
                match self {
                    $(Self::$name => $number,)*
                }
            }
        }
    };
}

info_types! {
    /// The cores the process may run on.
    CoreMask = 0;
    /// The priorities the process may use.
    PriorityMask = 1;
    /// Where the alias region starts.
    AliasRegionAddress = 2;
    /// The alias region's size.
    AliasRegionSize = 3;
    /// Where the heap region starts.
    HeapRegionAddress = 4;
    /// The heap region's size.
    HeapRegionSize = 5;
    /// The memory the process can use in all.
    TotalMemorySize = 6;
    /// The memory the process uses.
    UsedMemorySize = 7;
    /// Whether a debugger is attached.
    DebuggerAttached = 8;
    /// A handle to the process's resource limit.
    ResourceLimit = 9;
    /// The idle thread's tick count.
    IdleTickCount = 10;
    /// Random entropy.
    RandomEntropy = 11;
    /// Where the ASLR region (the code region) starts.
    AslrRegionAddress = 12;
    /// The ASLR region's size.
    AslrRegionSize = 13;
    /// Where the stack region starts.
    StackRegionAddress = 14;
    /// The stack region's size.
    StackRegionSize = 15;
    /// The size of the process's system resource.
    SystemResourceSizeTotal = 16;
    /// How much of its system resource the process uses.
    SystemResourceSizeUsed = 17;
    /// The program id.
    ProgramId = 18;
    /// The initial process id range.
    InitialProcessIdRange = 19;
    /// Where the user exception context is.
    UserExceptionContextAddress = 20;
    /// The memory the process can use in all, without its system resource.
    TotalNonSystemMemorySize = 21;
    /// The memory the process uses, without its system resource.
    UsedNonSystemMemorySize = 22;
    /// Whether the process is an application.
    IsApplication = 23;
    /// How many more threads the process may create.
    FreeThreadCount = 24;
    /// A thread's tick count.
    ThreadTickCount = 25;
    /// How much of the alias region is its extra size.
    AliasRegionExtraSize = 28;
    /// Switch 2: where the remote region starts.
    RemoteRegionAddress = 29;
    /// Switch 2: the remote region's size.
    RemoteRegionSize = 30;
    /// Switch 2: remote memory used.
    RemoteMemoryUsed = 31;
    /// Switch 2: remote memory in all.
    RemoteMemoryTotal = 32;
    /// Switch 2: the mapping alignment.
    MemoryMappingAlignment = 33;
    /// Unknown; Eden returns zero.
    Unknown37 = 37;
    /// Unknown; Eden returns zero.
    Unknown38 = 38;
    /// Atmosphère's kernel facts.
    MesosphereMeta = 65000;
    /// Atmosphère's handle to the current process.
    MesosphereCurrentProcess = 65001;
}

/// What `RemoteRegionAddress` (29) tells a Switch 2 program: no remote region. A guess, from
/// PR #4588 (`eden-pr4588/src/core/hle/kernel/svc/svc_info.cpp` @ e5fb3b97, `GetInfo`:
/// "Remote mappings are absent", `*result = 0`). A trace from a console, or the SDK's use of
/// the value, would confirm it.
pub const REMOTE_REGION_ADDRESS: u64 = 0;

/// What `RemoteRegionSize` (30) tells a Switch 2 program: no remote region. A guess, from the
/// same line of PR #4588 as [`REMOTE_REGION_ADDRESS`], confirmed the same way.
pub const REMOTE_REGION_SIZE: u64 = 0;

/// What `RemoteMemoryUsed` (31) tells a Switch 2 program: no remote memory. A guess, from the
/// same line of PR #4588 as [`REMOTE_REGION_ADDRESS`], confirmed the same way.
pub const REMOTE_MEMORY_USED: u64 = 0;

/// What `RemoteMemoryTotal` (32) tells a Switch 2 program: no remote memory. A guess, from the
/// same line of PR #4588 as [`REMOTE_REGION_ADDRESS`], confirmed the same way.
pub const REMOTE_MEMORY_TOTAL: u64 = 0;

/// What `MemoryMappingAlignment` (33) tells a Switch 2 program: the page size, [`PAGE_SIZE`]. A
/// guess, from PR #4588 (`eden-pr4588/src/core/hle/kernel/svc/svc_info.cpp` @ e5fb3b97,
/// `GetInfo`: "ordinary mappings use the kernel page size", `*result = PageSize`; the scoping
/// doc's row, "page granule 0x1000"). A trace from a console, or the SDK's use of the value,
/// would confirm it.
pub const MEMORY_MAPPING_ALIGNMENT: u64 = PAGE_SIZE;

impl InfoType {
    /// The id with this number, if it is listed.
    #[must_use]
    pub fn from_raw(raw: u32) -> Option<Self> {
        Self::ALL.iter().copied().find(|id| id.number() == raw)
    }
}

/// The process `handle` names (Eden's `GetObject<KProcess>`): the calling process
/// ([`Processes::own`]) for the current-process pseudo-handle, or the process a handle to a
/// process object names. [`KernelError::InvalidHandle`] for any other handle.
fn process_from_handle(
    handles: &HandleTable,
    scheduler: &Scheduler,
    processes: &Processes,
    handle: Handle,
) -> CallResult<ProcessId> {
    match handles.resolve(handle)? {
        Resolved::CurrentProcess => Ok(processes.own()),
        Resolved::CurrentThread => Err(KernelError::InvalidHandle.into()),
        Resolved::Object(object) => match scheduler.objects().get(object)? {
            Object::Process(process) => Ok(process),
            Object::Thread(_)
            | Object::Session
            | Object::Event(_)
            | Object::ReadableEvent(_)
            | Object::SharedMemory(_)
            | Object::TransferMemory(_) => Err(KernelError::InvalidHandle.into()),
        },
    }
}

/// Answers `GetInfo(info_type, handle, info_sub_id)` for the calling process
/// ([`Processes::own`] of `processes`), whose handle table is `handles` and whose address space
/// is `space`. `info_type` is the raw value the guest passed. Each group of ids checks what
/// Eden checks, in its order (see the module docs).
pub fn get_info(
    handles: &HandleTable,
    space: &AddressSpace,
    scheduler: &Scheduler,
    processes: &Processes,
    info_type: u32,
    handle: Handle,
    info_sub_id: u64,
) -> CallResult<u64> {
    let id = InfoType::from_raw(info_type).ok_or(KernelError::InvalidEnumValue)?;
    // PR #4588 answers its ids in an `if` before Eden's `switch`.
    if let Some(value) = switch_2_fact(id) {
        return switch_2_info(handles, scheduler, processes, handle, info_sub_id, value);
    }
    match id {
        // `case InfoType::DebuggerAttached: *result = 0; R_SUCCEED();`
        InfoType::DebuggerAttached => Ok(0),
        InfoType::RandomEntropy => {
            random_entropy(scheduler, processes.own(), handle, info_sub_id)
        }
        _ => memory_info(handles, space, scheduler, processes, id, handle, info_sub_id),
    }
}

/// What a Switch 2 id answers ([`REMOTE_REGION_ADDRESS`] to [`MEMORY_MAPPING_ALIGNMENT`]), or
/// `None` for any other id.
const fn switch_2_fact(id: InfoType) -> Option<u64> {
    Some(match id {
        InfoType::RemoteRegionAddress => REMOTE_REGION_ADDRESS,
        InfoType::RemoteRegionSize => REMOTE_REGION_SIZE,
        InfoType::RemoteMemoryUsed => REMOTE_MEMORY_USED,
        InfoType::RemoteMemoryTotal => REMOTE_MEMORY_TOTAL,
        InfoType::MemoryMappingAlignment => MEMORY_MAPPING_ALIGNMENT,
        _ => return None,
    })
}

/// A Switch 2 id, whose answer is `value`: for a Switch 1 program, Eden's `default`; for a
/// Switch 2 program, PR #4588's three checks in its order (see the module docs).
fn switch_2_info(
    handles: &HandleTable,
    scheduler: &Scheduler,
    processes: &Processes,
    handle: Handle,
    info_sub_id: u64,
    value: u64,
) -> CallResult<u64> {
    // A Switch 1 kernel does not know the id: Eden's `default`, `R_THROW(ResultInvalidEnumValue)`.
    if console_of(processes, processes.own())? != Console::Switch2 {
        return Err(KernelError::InvalidEnumValue.into());
    }
    // `R_UNLESS(info_sub_id == 0, ResultInvalidEnumValue)`.
    if info_sub_id != 0 {
        return Err(KernelError::InvalidEnumValue.into());
    }
    // `R_UNLESS(process.IsNotNull(), ResultInvalidHandle)`.
    let named = process_from_handle(handles, scheduler, processes, handle)?;
    // `R_UNLESS(process->UsesPAuthReturns(), ResultInvalidEnumValue)`: the process the handle
    // names was built for Switch 2 (an extended ACID).
    if console_of(processes, named)? != Console::Switch2 {
        return Err(KernelError::InvalidEnumValue.into());
    }
    tracing::debug!(value, "GetInfo Switch 2 id answered with PR #4588's guessed value");
    Ok(value)
}

/// The console `process` was built for. A process the table has no row for is an internal
/// failure: a process object or the calling process always names a created process.
fn console_of(processes: &Processes, process: ProcessId) -> CallResult<Console> {
    processes
        .get(process)
        .map(ProcessInfo::console)
        .map_err(|error| SvcError::internal(&error))
}

/// `RandomEntropy`: word `info_sub_id` of `process`'s entropy, after Eden's two checks.
fn random_entropy(
    scheduler: &Scheduler,
    process: ProcessId,
    handle: Handle,
    info_sub_id: u64,
) -> CallResult<u64> {
    // `R_UNLESS(handle == 0, ResultInvalidHandle)`.
    if handle != Handle::INVALID {
        return Err(KernelError::InvalidHandle.into());
    }
    // `R_UNLESS(info_sub_id < 4, ResultInvalidCombination)`.
    let index = usize::try_from(info_sub_id)
        .ok()
        .filter(|&index| index < ENTROPY_COUNT)
        .ok_or(KernelError::InvalidCombination)?;
    let entropy = scheduler.process(process)?.entropy().ok_or_else(|| {
        SvcError::Internal("the process's entropy was asked for before it was drawn".to_owned())
    })?;
    entropy
        .get(index)
        .copied()
        .ok_or_else(|| KernelError::InvalidCombination.into())
}

/// The memory ids: Eden's checks for its first group, then the value. An id of that group, or
/// of any other, that this slice does not answer is [`KernelError::NotImplemented`] before any
/// check.
fn memory_info(
    handles: &HandleTable,
    space: &AddressSpace,
    scheduler: &Scheduler,
    processes: &Processes,
    id: InfoType,
    handle: Handle,
    info_sub_id: u64,
) -> CallResult<u64> {
    // The value is a plain read, so it is taken before the checks; what is not built has none.
    let Some(value) = memory_fact(space, id) else {
        tracing::warn!(?id, number = id.number(), "GetInfo id not implemented");
        return Err(KernelError::NotImplemented.into());
    };
    if info_sub_id != 0 {
        return Err(KernelError::InvalidEnumValue.into());
    }
    process_from_handle(handles, scheduler, processes, handle)?;
    Ok(value)
}

/// The fact `id` asks for, if it is one this slice answers (see the module docs).
fn memory_fact(space: &AddressSpace, id: InfoType) -> Option<u64> {
    let regions = space.regions();
    Some(match id {
        InfoType::AliasRegionAddress => regions.alias.start,
        InfoType::AliasRegionSize => regions.alias.size(),
        InfoType::HeapRegionAddress => regions.heap.start,
        InfoType::HeapRegionSize => regions.heap.size(),
        InfoType::AslrRegionAddress => regions.code.start,
        InfoType::AslrRegionSize => regions.code.size(),
        InfoType::StackRegionAddress => regions.stack.start,
        InfoType::StackRegionSize => regions.stack.size(),
        InfoType::TotalMemorySize => space.total_user_physical_memory_size(),
        InfoType::UsedMemorySize => space.used_user_physical_memory_size(),
        InfoType::AliasRegionExtraSize => regions.alias_region_extra_size,
        _ => return None,
    })
}

#[cfg(test)]
mod tests;
