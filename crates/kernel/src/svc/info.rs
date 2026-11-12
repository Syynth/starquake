//! `GetInfo` (0x29): the memory facts of a process. This slice answers the ids a libnx program
//! asks at start-up: the alias, heap, ASLR and stack regions, the alias region's extra size,
//! and the total and used memory.
//!
//! Ported from Eden `eden/src/core/hle/kernel/svc/svc_info.cpp` @ cf654052 (GPL-3.0-or-later),
//! `GetInfo`; the ids are `InfoType` in `eden/src/core/hle/kernel/svc_types.h` @ cf654052.
//! For the ids answered here, Eden checks in this order:
//!
//! 1. `info_sub_id == 0`, else `ResultInvalidEnumValue`.
//! 2. The handle names a process (`GetObject<KProcess>`: the current-process pseudo-handle, or
//!    a handle to a process), else `ResultInvalidHandle`.
//!
//! and then reads the value from that process's page table, or (ids 6 and 7) from the process.
//!
//! # Divergences from Eden
//!
//! - **Ids Eden answers that this slice does not** return [`KernelError::NotImplemented`]
//!   and are logged: the core and priority masks,
//!   the debugger, resource-limit, tick-count and entropy ids, the system-resource sizes, the
//!   program id, the initial process id range, the exception context address, the non-system
//!   memory sizes, `IsApplication`, the free thread count, ids 37 and 38, and Atmosphère's two.
//!   Eden answers each. They belong to the slices that build what they report (process
//!   creation, threads, resource limits). Test:
//!   `an_id_this_slice_does_not_answer_is_not_implemented_whatever_else_is_passed`.
//! - **The Switch 2 ids 29 to 33** (`eden-pr4588/src/core/hle/kernel/svc/svc_info.cpp` @
//!   e5fb3b97; scoping doc, "New `svcGetInfo` ids") are not built either and return
//!   [`KernelError::NotImplemented`]. PR #4588 answers them only for a Switch 2 program; Eden
//!   returns `ResultInvalidEnumValue`. Same test.
//! - **Any process handle names the one process there is.** With one process, a handle to a
//!   process object and the pseudo-handle are the same process (kernel slice 8 makes
//!   processes rows of a table). Test: `a_handle_must_name_a_process`.
//!
//! Ids Eden does not list (`IsSvcPermitted` 26, `IoRegionHint` 27, `TransferMemoryHint` 34 and
//! every number that is not an `InfoType`) return [`KernelError::InvalidEnumValue`], as Eden's
//! `default` does.

use crate::handle::{Handle, HandleTable, Resolved};
use crate::memory::AddressSpace;
use crate::object::Object;
use crate::result::{CallResult, KernelError};
use crate::tables::Scheduler;

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

impl InfoType {
    /// The id with this number, if it is listed.
    #[must_use]
    pub fn from_raw(raw: u32) -> Option<Self> {
        Self::ALL.iter().copied().find(|id| id.number() == raw)
    }
}

/// Whether `handle` names a process (Eden's `GetObject<KProcess>`): the current-process
/// pseudo-handle, or a handle to a process object. [`KernelError::InvalidHandle`] otherwise.
fn process_from_handle(
    handles: &HandleTable,
    scheduler: &Scheduler,
    handle: Handle,
) -> CallResult<()> {
    match handles.resolve(handle)? {
        Resolved::CurrentProcess => Ok(()),
        Resolved::CurrentThread => Err(KernelError::InvalidHandle.into()),
        Resolved::Object(object) => match scheduler.objects().get(object)? {
            Object::Process(_) => Ok(()),
            Object::Thread(_) | Object::Session | Object::Event(_) | Object::ReadableEvent(_) => {
                Err(KernelError::InvalidHandle.into())
            }
        },
    }
}

/// Answers `GetInfo(info_type, handle, info_sub_id)` for the calling process, whose handle
/// table is `handles` and whose address space is `space`. `info_type` is the raw value the
/// guest passed.
pub fn get_info(
    handles: &HandleTable,
    space: &AddressSpace,
    scheduler: &Scheduler,
    info_type: u32,
    handle: Handle,
    info_sub_id: u64,
) -> CallResult<u64> {
    let id = InfoType::from_raw(info_type).ok_or(KernelError::InvalidEnumValue)?;
    // The value is a plain read, so it is taken before the checks; what is not built has none.
    let Some(value) = memory_fact(space, id) else {
        tracing::warn!(?id, number = id.number(), "GetInfo id not implemented");
        return Err(KernelError::NotImplemented.into());
    };
    if info_sub_id != 0 {
        return Err(KernelError::InvalidEnumValue.into());
    }
    process_from_handle(handles, scheduler, handle)?;
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
