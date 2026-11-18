//! `GetInfo` (0x29): facts about a process, a thread or the kernel. Every id Eden's `GetInfo`
//! answers is answered here, and so are the Switch 2 ids 29 to 33 (the remote region's address
//! and size, remote memory used and in all, and the mapping alignment; scoping doc, "Process,
//! memory, kernel", "New `svcGetInfo` ids").
//!
//! Ported from Eden `eden/src/core/hle/kernel/svc/svc_info.cpp` @ cf654052 (GPL-3.0-or-later),
//! `GetInfo`; the ids are `InfoType` in `eden/src/core/hle/kernel/svc_types.h` @ cf654052, the
//! process's facts are `KProcess`'s (`eden/src/core/hle/kernel/k_process.h` and
//! `k_process.cpp` @ cf654052), and PR #4588's changes are from
//! `eden-pr4588/src/core/hle/kernel/svc/svc_info.cpp` @ e5fb3b97. Eden's checks, in its order,
//! for each group of ids:
//!
//! - **The process ids** (0 to 7, 12 to 18, 20 to 24 and 28: the first `case` group of Eden's
//!   `switch`):
//!   1. `info_sub_id == 0`, else `ResultInvalidEnumValue`.
//!   2. The handle names a process (`GetObject<KProcess>`: the current-process pseudo-handle,
//!      or a handle to a process), else `ResultInvalidHandle`. The process id a process
//!      object holds is then checked against what was allocated: the scheduler has a row for
//!      it, whose own object ([`crate::process::ProcessRecord::object`]) is the one the
//!      handle names. Either failing is the kernel's own inconsistency, an internal failure
//!      after the guest's checks (Eden has nothing to check: its object is the process). Tests:
//!      `a_fact_about_a_process_with_no_row_is_an_internal_failure_after_the_checks`,
//!      `a_process_object_that_is_not_its_process_s_own_is_an_internal_failure_after_the_checks`.
//!
//!   Then the value. The **address-space ids** (2 to 7, 21, 22 and 28) read the calling
//!   process's address space (see the divergences); the **fact ids** (0, 1, 16, 17, 18, 23 and
//!   24) read the process the handle names; 20 is a constant (see the divergences):
//!   - 0 `CoreMask` and 1 `PriorityMask`: `GetCoreMask` and `GetPriorityMask`, the masks of
//!     the process's thread-info capability ([`ProcessInfo::core_mask`],
//!     [`ProcessInfo::priority_mask`]).
//!   - 2 to 5, 12 to 15: the alias, heap, alias-code (ASLR) and stack regions. 28: the alias
//!     region's extra size, which Eden reads from the calling process
//!     (`GetCurrentProcessPointer`) whatever the handle names, after a second
//!     `R_UNLESS(info_sub_id == 0, ResultInvalidCombination)` that the first check makes
//!     unreachable.
//!   - 6 `TotalMemorySize` and 7 `UsedMemorySize`: `GetTotalUserPhysicalMemorySize` and
//!     `GetUsedUserPhysicalMemorySize`, the system resource's charge counted
//!     ([`AddressSpace::total_user_physical_memory_size`],
//!     [`AddressSpace::used_user_physical_memory_size`]; "The system resource", below).
//!   - 16 `SystemResourceSizeTotal`: `GetTotalSystemResourceSize`, the size of the process's
//!     secure system resource ([`secure_resource_size`]).
//!   - 17 `SystemResourceSizeUsed`: `GetUsedSystemResourceSize`; [`SYSTEM_RESOURCE_SIZE_USED`]
//!     here (see the divergences).
//!   - 18 `ProgramId`: `GetProgramId` ([`ProcessInfo::program_id`]).
//!   - 20 `UserExceptionContextAddress`: `GetProcessLocalRegionAddress`;
//!     [`USER_EXCEPTION_CONTEXT_ADDRESS`] here (see the divergences).
//!   - 21 `TotalNonSystemMemorySize` and 22 `UsedNonSystemMemorySize`:
//!     `GetTotalNonSystemUserPhysicalMemorySize` and `GetUsedNonSystemUserPhysicalMemorySize`
//!     ([`non_system_total`] with the calling process's charge; the used size is the normal
//!     memory (the heap and what `MapPhysicalMemory` mapped), the code and the main thread's
//!     stack, [`AddressSpace::used_non_system_user_physical_memory_size`]).
//!   - 23 `IsApplication`: `IsApplication` ([`is_application`]): one when the named process's
//!     pool is the application pool, as `LoadFromMetadata` sets `m_is_application`. Test:
//!     `is_application_is_whether_the_named_process_draws_from_the_application_pool`.
//!   - 24 `FreeThreadCount`: the process's resource limit's `ThreadCountMax` limit less its
//!     current value (`process->GetResourceLimit()`; [`crate::resource_limit`]), read through
//!     the values `GetResourceLimitLimitValue` and `GetResourceLimitCurrentValue` answer
//!     ([`crate::svc::resource_limit`]), so the three agree. Tests:
//!     `the_free_thread_count_is_the_limit_less_the_process_s_threads`,
//!     `the_free_thread_count_agrees_with_the_resource_limit_calls`.
//! - **`DebuggerAttached`** (8): Eden checks neither the handle nor the sub id; the value is
//!   zero (`*result = 0`). No debugger is ever attached here, so the value is the fact, not a
//!   stub. Eden, and so starquake, is looser than Horizon here: Horizon checks
//!   `handle == InvalidHandle`, else `ResultInvalidHandle`, and then `info_subtype == 0`, else
//!   `ResultInvalidCombination` (fact only,
//!   `atmosphere/libraries/libmesosphere/source/svc/kern_svc_info.cpp` @ 4e3f3703,
//!   `InfoType_DebuggerAttached`). Test: `a_debugger_is_never_attached_whatever_is_passed`.
//! - **`ResourceLimit`** (9):
//!   1. `handle == 0`, else `ResultInvalidHandle`.
//!   2. `info_sub_id == 0`, else `ResultInvalidCombination`.
//!
//!   Then a new handle to the calling process's resource limit is added to its handle table
//!   (`handle_table.Add`), whose failure (`ResultOutOfHandles` for a full table) is the call's
//!   result; the value is the new handle. Eden answers `Svc::InvalidHandle` for a process with
//!   no limit, which every process here has ([`crate::sched::Scheduler::add_process`]). Tests:
//!   `the_resource_limit_is_a_new_handle_to_the_calling_process_s_limit_after_edens_two_checks`,
//!   `the_resource_limit_with_a_full_handle_table_is_out_of_handles`.
//! - **`IdleTickCount`** (10):
//!   1. `info_sub_id` is [`ALL_CORES`] or the calling core's index, else
//!      `ResultInvalidCombination`.
//!   2. `handle == InvalidHandle`, else `ResultInvalidHandle`.
//!
//!   Then the calling core's idle thread's CPU time; [`IDLE_TICK_COUNT`] here (see the
//!   divergences). Test: `the_idle_tick_count_checks_the_core_then_the_handle`.
//! - **`RandomEntropy`** (11):
//!   1. `handle == 0` (the invalid handle, not a process), else `ResultInvalidHandle`.
//!   2. `info_sub_id < 4` ([`ENTROPY_COUNT`]), else `ResultInvalidCombination`.
//!
//!   Then the value is word `info_sub_id` of the calling process's entropy
//!   (`GetCurrentProcess(...).GetRandomEntropy(info_sub_id)`).
//! - **`InitialProcessIdRange`** (19):
//!   1. `handle == InvalidHandle`, else `ResultInvalidHandle`.
//!   2. `info_sub_id` is `InitialProcessIdRangeInfo::Minimum` (0) or `Maximum` (1)
//!      (`svc_types.h`), else `ResultInvalidCombination`.
//!
//!   Then Eden's values, which Eden itself marks as a stub ("(STUBBED) ... returned 0/64",
//!   `//todo`): [`INITIAL_PROCESS_ID_MINIMUM`] and [`INITIAL_PROCESS_ID_MAXIMUM`]. Each is a
//!   guess. Horizon's real range ends below the first user process id
//!   ([`crate::tables::FIRST_USER_PROCESS_ID`] is Eden's `InitialProcessIdMax + 1`), which
//!   Eden's stub does not match; a trace from a console would settle it. Test:
//!   `the_initial_process_id_range_is_edens_stub_after_its_checks`.
//! - **`ThreadTickCount`** (25):
//!   1. `info_sub_id` is [`ALL_CORES`] or below the calling process's core count, else
//!      `ResultInvalidCombination`. The count is PR #4588's: `NUM_CPU_CORES`
//!      ([`APPLICATION_CORE_COUNT`]) for a process built for Switch 2 (`UsesPAuthReturns`),
//!      `NUM_LEGACY_CPU_CORES` ([`LEGACY_VIRTUAL_CORE_COUNT`]) for any other; Eden mainline
//!      has `num_cpus = 4` for every process, which PR #4588 keeps for a Switch 1 program.
//!   2. The handle names a thread (`GetObject<KThread>`: the current-thread pseudo-handle, or a
//!      handle to a thread; [`thread_from_handle`]), else `ResultInvalidHandle`.
//!
//!   Then the thread's CPU time: zero for any thread but the caller, and for the caller the
//!   time since the core's last context switch (with the thread's CPU time added for
//!   [`ALL_CORES`]); [`THREAD_TICK_COUNT`] here for every thread (see the divergences). Tests:
//!   `the_thread_tick_count_checks_the_core_count_of_the_console_then_the_handle`,
//!   `the_thread_tick_count_is_zero_for_every_thread`.
//! - **`Unknown37` and `Unknown38`** (37, 38): no checks; the value is zero (Eden's
//!   "(STUBBED)" case). Test: `ids_37_and_38_are_zero_whatever_is_passed`.
//! - **`MesosphereMeta`** (65000):
//!   1. `handle == InvalidHandle`, else `ResultInvalidHandle`.
//!   2. `info_sub_id` is one of Eden's `MesosphereMetaInfo` values, else
//!      `ResultInvalidCombination`: [`MESOSPHERE_META_KERNEL_VERSION`] (0) answers
//!      [`SUPPORTED_KERNEL_VERSION`]; [`MESOSPHERE_META_IS_KTRACE_ENABLED`] (1) and
//!      [`MESOSPHERE_META_IS_SINGLE_STEP_ENABLED`] (2) answer zero.
//!
//!   Test: `the_mesosphere_meta_id_checks_the_handle_then_the_sub_id`.
//! - **`MesosphereCurrentProcess`** (65001):
//!   1. `handle == InvalidHandle`, else `ResultInvalidHandle`.
//!   2. `info_sub_id == 0`, else `ResultInvalidCombination`.
//!   3. A new handle to the calling process is added to its handle table
//!      (`handle_table.Add`), whose failure (`ResultOutOfHandles` for a full table) is the
//!      call's result.
//!
//!   Then the value is the new handle. Tests:
//!   `the_current_process_id_adds_a_handle_to_the_calling_process`,
//!   `the_current_process_id_with_a_full_handle_table_is_out_of_handles`.
//!
//! PR #4588 changes `ThreadTickCount`'s core count (above) and none of the other checks or
//! values, so a Switch 2 program is answered the same otherwise.
//!
//! - **The Switch 2 ids** (29 to 33), from PR #4588's `GetInfo`, which answers them in an `if`
//!   before Eden's `switch`:
//!   1. The calling process was built for Switch 2, else [`KernelError::InvalidEnumValue`]
//!      (Eden's `default`, `R_THROW(ResultInvalidEnumValue)`: a Switch 1 kernel does not know
//!      the ids; see the divergences).
//!   2. `info_sub_id == 0`, else `ResultInvalidEnumValue`.
//!   3. The handle names a process (`GetObject<KProcess>`), else `ResultInvalidHandle`.
//!   4. That process was built for Switch 2 (`process->UsesPAuthReturns()`, which Eden PR
//!      #4588 sets from the NPDM's extended ACID: [`crate::process::Console::Switch2`]), else
//!      `ResultInvalidEnumValue`.
//!
//!   Then the value is PR #4588's: zero for ids 29 to 32 ("Remote mappings are absent") and
//!   the page size for id 33 ("ordinary mappings use the kernel page size"). **Each value is a
//!   guess** ([`REMOTE_REGION_ADDRESS`], [`REMOTE_REGION_SIZE`], [`REMOTE_MEMORY_USED`],
//!   [`REMOTE_MEMORY_TOTAL`], [`MEMORY_MAPPING_ALIGNMENT`]): a trace from a console, or the
//!   SDK's use of the value, would confirm it. The references disagree: Eden-NX2
//!   (`eden-nx2/src/core/hle/kernel/svc/svc_info.cpp` @ 50fa86bb) answers ids 29 and 30 with
//!   the upper half of the alias region, answers every process whatever it was built for, and
//!   names 31 and 32 the other way round (size, then used). starquake follows PR #4588 (the
//!   lead's ruling under the delegation, `docs/proposals.md`, #223). A process a handle names
//!   that the process table has no row for is an internal failure, after the guest's checks.
//!   (A calling process with no row would be one before them, at step 1; it cannot arise, as
//!   the table never drops its own process.) Tests: `each_switch_2_id_answers_a_switch_2_program`,
//!   `a_switch_2_id_with_a_sub_id_other_than_zero_is_an_invalid_enum_value_before_the_handle`,
//!   `a_switch_2_id_needs_a_handle_to_a_process`,
//!   `a_switch_2_id_is_answered_by_the_console_of_the_process_the_handle_names`,
//!   `a_handle_to_a_process_the_table_has_no_row_for_is_an_internal_failure`.
//!
//! No id returns [`KernelError::NotImplemented`]. Test: `no_id_eden_answers_is_not_implemented`.
//!
//! # The system resource
//!
//! A process's system resource (`META` 0x14, [`ProcessInfo::system_resource`]) is what ids 6,
//! 7, 16, 17 and 21 count. Eden's `KProcess::LoadFromMetadata` asks for
//! `system_resource_num_pages = metadata.GetSystemResourceSize() / PageSize`; `Initialize`
//! makes a `KSecureSystemResource` of that many pages when it is not zero, reserving its
//! charge from the process's `PhysicalMemoryMax` limit, and otherwise gives the process the
//! shared default resource, which every one of Eden's `GetTotalSystemResourceSize`,
//! `GetUsedSystemResourceSize`, `GetRequiredSecureMemorySize` and
//! `GetRequiredSecureMemorySizeNonDefault` answers with zero. Here the address space is made
//! with the resource and charges it to its memory count
//! ([`crate::memory::AddressSpace::with_system_resource`], [`crate::memory::system_resource`]).
//! So:
//!
//! - 16 is the size in whole pages, whatever the pool ([`secure_resource_size`]: `GetSize`).
//! - 6, 7 and 21 take the charge, `S` below (`GetRequiredSecureMemorySize`, and the
//!   `NonDefault` form, which is the same for a secure resource): the size for any pool but the
//!   applet pool, zero for that one (`CalculateRequiredSecureMemorySize`,
//!   `eden/src/core/hle/kernel/board/nintendo/nx/k_system_control.cpp` @ cf654052;
//!   [`AddressSpace::secure_memory_size`]). With `U` the non-system used size, `F` the free
//!   memory after the charge and `M` the process maximum ([`AddressSpace::max_process_memory`]),
//!   7 is `U + S`; 6 is `M` when `U + S + F > M`, else `F + U + S`; 21 is `M - S` when
//!   `U + S + F > M`, else `F + U`, Eden's formulas. `U + S + F` is the whole pool, which is
//!   below `M` on both arrangements (4 GiB: pool 3285 MiB, `M` 8 GiB; 12 GiB: pool 9809 MiB,
//!   `M` 12 GiB), so a live address space always takes the second branch: 6 is the pool and 21
//!   the pool less `S`. The first branches are pinned on the functions
//!   (`the_non_system_total_takes_edens_branch_at_the_exact_edge`, and the space's own tests).
//!   So the `S` that 21 passes to [`non_system_total`] (the calling address space's charge) is
//!   not observable through a live process: passing the size instead, or another process's,
//!   gives the same answer, as the second branch does not read it. A mutation of it survives
//!   every test but the function's own, and is listed here as unobservable.
//!   Tests: `the_non_system_total_is_the_pool_less_the_calling_process_s_charge`,
//!   `the_total_and_used_memory_count_the_system_resource_as_eden_does`.
//! - The resource limit's `PhysicalMemoryMax` current value is the same count (#242,
//!   [`crate::resource_limit`]), so it holds `S` too: in the second of those tests.
//!
//! # Divergences from Eden
//!
//! - **[`non_system_total`] answers zero for an `S` larger than `M`**, rather than wrapping.
//!   Neither Eden nor a live address space here can have one (the charge is at most the pool,
//!   and its reservation fails with `ResultLimitReached` otherwise). Test:
//!   `a_system_resource_larger_than_the_maximum_leaves_no_non_system_total`.
//! - **17 is [`SYSTEM_RESOURCE_SIZE_USED`] (zero) for every process.** Eden's is the pages the
//!   secure resource's dynamic page manager has handed out for the kernel's own structures
//!   (`m_dynamic_page_manager.GetUsed() * PageSize`), which nothing here keeps; for a process
//!   whose `S` is zero Eden's is zero too. A guess; a trace from a console would show the real
//!   value. Test: `the_system_resource_used_is_zero_whatever_its_size`.
//! - **20 is [`USER_EXCEPTION_CONTEXT_ADDRESS`] (zero) for every process.** Eden's is the
//!   process-local region, which no process here has (`crate::process`, "Divergences from
//!   `KProcess::Initialize`"). A guess, until the region is reserved. Test:
//!   `the_exception_context_address_is_zero`.
//! - **No CPU time is kept**, so 10 is [`IDLE_TICK_COUNT`] and 25 [`THREAD_TICK_COUNT`], both
//!   zero, after Eden's checks. Eden's are the idle thread's CPU time (10) and, for the calling
//!   thread, the ticks since its core's last context switch (25, with its CPU time added for
//!   [`ALL_CORES`]); 25 is zero in Eden too for any other thread, or for the caller with
//!   another core's index. Each is a guess. Tests:
//!   `the_idle_tick_count_checks_the_core_then_the_handle`,
//!   `the_thread_tick_count_is_zero_for_every_thread`.
//! - **A Switch 1 program asking a Switch 2 id** is told [`KernelError::InvalidEnumValue`]
//!   before any check, whatever its handle and sub id, as Eden answers every program
//!   (`eden/src/core/hle/kernel/svc/svc_info.cpp` @ cf654052, `GetInfo`'s `default`) and as
//!   Horizon's Switch 1 kernel does (fact only,
//!   `atmosphere/libraries/libmesosphere/source/svc/kern_svc_info.cpp` @ 4e3f3703, `GetInfo`'s
//!   `default`). PR #4588 has no caller gate: it checks the sub id, the handle and then only
//!   the named process's `UsesPAuthReturns()`, for every program. So a Switch 1 program
//!   passing sub id 0 is answered differently in exactly two cases (with any other sub id, or
//!   a handle naming a Switch 1 process, both say `InvalidEnumValue`):
//!   - the handle names no process: PR #4588 says `ResultInvalidHandle`, here it is
//!     `InvalidEnumValue`. Test:
//!     `a_switch_1_program_is_told_the_switch_2_ids_are_an_invalid_enum_value_whatever_is_passed`.
//!   - the handle names a Switch 2 process: PR #4588 answers the value, here it is
//!     `InvalidEnumValue`. Test (its last loop):
//!     `a_switch_2_id_is_answered_by_the_console_of_the_process_the_handle_names`.
//! - **The address-space ids (2 to 7, 21, 22 and 28) read the calling process's address
//!   space, whatever process the handle names.** With one address space beside the tables, a
//!   handle to another process object is answered as the pseudo-handle is; 6, 7 and 21 also
//!   take the charge of the calling process's system resource, the one its address space was
//!   made with. Test, the second process's resource differing from the calling process's:
//!   `the_non_system_total_is_the_pool_less_the_calling_process_s_charge`.
//!   The fact ids (0, 1, 16, 17, 18, 23 and 24) and the Switch 2 ids do read the process the
//!   handle names, from the process table and the scheduler; one the tables have no row for
//!   is an internal failure, after the guest's checks. Tests: `a_handle_must_name_a_process`,
//!   `the_facts_are_the_named_process_s_and_the_sizes_the_caller_s`,
//!   `a_fact_about_a_process_with_no_row_is_an_internal_failure_after_the_checks`.
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
use crate::process::{Console, ENTROPY_COUNT, LEGACY_VIRTUAL_CORE_COUNT, ProcessInfo};
use crate::resource_limit::LimitableResource;
use crate::result::{CallResult, KernelError, SvcError};
use crate::sched::{APPLICATION_CORE_COUNT, CoreId};
use crate::svc::resource_limit::{self, LimitTables};
use crate::svc::thread::thread_from_handle;
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

/// What `SystemResourceSizeUsed` (17) tells every process: nothing used. A guess: Eden's is the
/// pages the secure resource's dynamic page manager has handed out
/// (`KSecureSystemResource::GetUsedSize`, `eden/src/core/hle/kernel/k_system_resource.h` @
/// cf654052), which nothing here keeps. A trace from a console would show the real value.
pub const SYSTEM_RESOURCE_SIZE_USED: u64 = 0;

/// What `UserExceptionContextAddress` (20) tells every process. A guess: Eden's is the
/// process-local region's address (`GetProcessLocalRegionAddress`), which no process here has
/// (`crate::process`, "Divergences from `KProcess::Initialize`"). Reserving that region gives
/// the real value.
pub const USER_EXCEPTION_CONTEXT_ADDRESS: u64 = 0;

/// The sub id `IdleTickCount` (10) and `ThreadTickCount` (25) take for "every core" rather than
/// one core's index: Eden compares with `0xFFFFFFFFFFFFFFFF`.
pub const ALL_CORES: u64 = u64::MAX;

/// What `IdleTickCount` (10) tells every process: no idle time. A guess: Eden's is the calling
/// core's idle thread's CPU time (`GetIdleThread()->GetCpuTime()`), which nothing here keeps.
pub const IDLE_TICK_COUNT: u64 = 0;

/// What `ThreadTickCount` (25) tells every process, for every thread. Eden's is zero for any
/// thread but the caller, and for the caller with another core's index; for the caller it is
/// the ticks since its core's last context switch, with the thread's CPU time added for
/// [`ALL_CORES`]. Nothing here keeps either, so for the caller this is a guess.
pub const THREAD_TICK_COUNT: u64 = 0;

/// `InitialProcessIdRangeInfo::Minimum` (`eden/src/core/hle/kernel/svc_types.h` @ cf654052):
/// the `InitialProcessIdRange` (19) sub id for the range's lowest id.
pub const INITIAL_PROCESS_ID_RANGE_MINIMUM: u64 = 0;

/// `InitialProcessIdRangeInfo::Maximum`: the `InitialProcessIdRange` (19) sub id for the
/// range's highest id.
pub const INITIAL_PROCESS_ID_RANGE_MAXIMUM: u64 = 1;

/// What `InitialProcessIdRange` (19) answers for [`INITIAL_PROCESS_ID_RANGE_MINIMUM`]: Eden's
/// stub value (`*result = 0; //todo`, logged as "(STUBBED) ... returned 0/64"). A guess Eden
/// marks as one; a trace from a console would settle it.
pub const INITIAL_PROCESS_ID_MINIMUM: u64 = 0;

/// What `InitialProcessIdRange` (19) answers for [`INITIAL_PROCESS_ID_RANGE_MAXIMUM`]: Eden's
/// stub value (`*result = 64; //todo`). A guess, as [`INITIAL_PROCESS_ID_MINIMUM`] is.
pub const INITIAL_PROCESS_ID_MAXIMUM: u64 = 64;

/// `MesosphereMetaInfo::KernelVersion`: the `MesosphereMeta` (65000) sub id for the kernel
/// version (Eden's `GetInfo`, `svc_info.cpp` @ cf654052).
pub const MESOSPHERE_META_KERNEL_VERSION: u64 = 0;

/// `MesosphereMetaInfo::IsKTraceEnabled`: the `MesosphereMeta` sub id for whether kernel
/// tracing is on. Eden answers zero.
pub const MESOSPHERE_META_IS_KTRACE_ENABLED: u64 = 1;

/// `MesosphereMetaInfo::IsSingleStepEnabled`: the `MesosphereMeta` sub id for whether single
/// stepping is on. Eden answers zero.
pub const MESOSPHERE_META_IS_SINGLE_STEP_ENABLED: u64 = 2;

/// Where a kernel version's minor number starts: `KernelVersion::minor_version`,
/// `BitField<0, 4, u32>` (`eden/src/core/hle/kernel/svc_version.h` @ cf654052).
pub const KERNEL_VERSION_MINOR_SHIFT: u32 = 0;

/// How many bits a kernel version's minor number has (`BitField<0, 4, u32>`).
pub const KERNEL_VERSION_MINOR_WIDTH: u32 = 4;

/// Where a kernel version's major number starts: `KernelVersion::major_version`,
/// `BitField<4, 13, u32>`.
pub const KERNEL_VERSION_MAJOR_SHIFT: u32 = 4;

/// How many bits a kernel version's major number has (`BitField<4, 13, u32>`).
pub const KERNEL_VERSION_MAJOR_WIDTH: u32 = 13;

/// How much an SDK major version is below the kernel's: `ConvertToSvcMajorVersion(sdk)` is
/// `sdk + 4` (`svc_version.h`).
pub const SVC_MAJOR_VERSION_OFFSET: u32 = 4;

/// The SDK major version of the newest kernel Eden supports: `ConvertToSvcMajorVersion(22)`
/// (`SupportedKernelMajorVersion`, `svc_version.h` @ cf654052; the same in Eden PR #4588 @
/// e5fb3b97).
pub const SUPPORTED_SDK_MAJOR_VERSION: u32 = 22;

/// The minor version of the newest kernel Eden supports: `ConvertToSvcMinorVersion(2)`, which
/// keeps the SDK's minor number (`SupportedKernelMinorVersion`).
pub const SUPPORTED_KERNEL_MINOR_VERSION: u32 = 2;

/// The kernel version `MesosphereMeta`'s [`MESOSPHERE_META_KERNEL_VERSION`] answers:
/// `SupportedKernelVersion`, Eden's encoding ([`encode_kernel_version`]) of
/// [`SUPPORTED_SDK_MAJOR_VERSION`] plus [`SVC_MAJOR_VERSION_OFFSET`] and
/// [`SUPPORTED_KERNEL_MINOR_VERSION`].
pub const SUPPORTED_KERNEL_VERSION: u32 = encode_kernel_version(
    SUPPORTED_SDK_MAJOR_VERSION + SVC_MAJOR_VERSION_OFFSET,
    SUPPORTED_KERNEL_MINOR_VERSION,
);

/// A mask of `width` low bits; `width` is below 32.
const fn low_bits(width: u32) -> u32 {
    (1 << width) - 1
}

/// Eden's `EncodeKernelVersion(major, minor)`: each number cut to its field's width
/// ([`KERNEL_VERSION_MINOR_WIDTH`], [`KERNEL_VERSION_MAJOR_WIDTH`]) and put at its shift, as
/// `BitField::FormatValue` does.
#[must_use]
pub const fn encode_kernel_version(major: u32, minor: u32) -> u32 {
    ((minor & low_bits(KERNEL_VERSION_MINOR_WIDTH)) << KERNEL_VERSION_MINOR_SHIFT)
        | ((major & low_bits(KERNEL_VERSION_MAJOR_WIDTH)) << KERNEL_VERSION_MAJOR_SHIFT)
}

/// The size of the secure system resource of `process` (Eden's `GetTotalSystemResourceSize`,
/// `GetInfo` 16): its recorded size ([`ProcessInfo::system_resource_size`]) rounded down to
/// whole pages, which is zero below one page, whatever its pool (see "The system resource" in
/// the module docs). What the resource charges is
/// [`crate::memory::SystemResource::required_secure_memory_size`].
#[must_use]
pub const fn secure_resource_size(process: &ProcessInfo) -> u64 {
    process.system_resource().size()
}

/// What `IsApplication` (23) tells of `process`: one for an application
/// ([`ProcessInfo::is_application`]: its pool is the application pool), zero otherwise
/// (`*result = process->IsApplication()`).
#[must_use]
pub fn is_application(process: &ProcessInfo) -> u64 {
    u64::from(process.is_application())
}

/// `KProcess::GetTotalNonSystemUserPhysicalMemorySize` (`eden/src/core/hle/kernel/k_process.cpp`
/// @ cf654052) of a process that uses `used_non_system` bytes (its normal memory, the code and
/// the main thread's stack: `GetUsedNonSystemUserPhysicalMemorySize`), can still reserve
/// `free`, may use at most `max` (`m_max_process_memory`) and whose secure system resource
/// needs `secure` ([`secure_resource_size`]): `max - secure` when
/// `used_non_system + secure + free > max`, else `free + used_non_system`. A `secure` above
/// `max`, which Eden cannot reach, answers zero (see the module docs).
#[must_use]
pub const fn non_system_total(used_non_system: u64, free: u64, max: u64, secure: u64) -> u64 {
    let used = used_non_system.saturating_add(secure);
    if used.saturating_add(free) > max {
        max.saturating_sub(secure)
    } else {
        free.saturating_add(used_non_system)
    }
}

impl InfoType {
    /// The id with this number, if it is listed.
    #[must_use]
    pub fn from_raw(raw: u32) -> Option<Self> {
        Self::ALL.iter().copied().find(|id| id.number() == raw)
    }
}

/// The tables `GetInfo` reads, for the calling process ([`Processes::own`] of `processes`),
/// whose handle table is `handles` and whose address space is `space`. Only
/// `MesosphereCurrentProcess` (65001) writes: it adds a handle, which takes a reference in the
/// scheduler's object table.
#[derive(Debug)]
pub struct InfoTables<'t> {
    /// The calling process's handle table.
    pub handles: &'t mut HandleTable,
    /// The calling process's address space.
    pub space: &'t AddressSpace,
    /// The scheduler, with its object table and the processes' rows.
    pub scheduler: &'t mut Scheduler,
    /// The process facts.
    pub processes: &'t Processes,
}

/// The process `handle` names (Eden's `GetObject<KProcess>`): the calling process
/// ([`Processes::own`]) for the current-process pseudo-handle, or the process a handle to a
/// process object names. [`KernelError::InvalidHandle`] for any other handle.
///
/// The id a process object holds is checked against what was allocated: the scheduler has a
/// row for that process ([`Scheduler::process`]'s error otherwise), and the row's own object
/// ([`crate::process::ProcessRecord::object`]) is the one the handle names (an internal
/// failure otherwise). Both are the kernel's own inconsistency, which no guest input makes:
/// only [`Scheduler::add_process`] makes a process object.
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
            Object::Process(process) => {
                if scheduler.process(process)?.object() != object {
                    return Err(SvcError::Internal(
                        "a handle names a process object that is not its process's own".into(),
                    ));
                }
                Ok(process)
            }
            Object::Thread(_)
            | Object::Session
            | Object::Event(_)
            | Object::ReadableEvent(_)
            | Object::SharedMemory(_)
            | Object::TransferMemory(_)
            | Object::ResourceLimit(_) => Err(KernelError::InvalidHandle.into()),
        },
    }
}

/// Answers `GetInfo(info_type, handle, info_sub_id)` for the calling process, on `core`, the
/// core the call was made on, from `tables`. `info_type` is the raw value the guest passed.
/// Each group of ids checks what Eden checks, in its order (see the module docs).
pub fn get_info(
    tables: InfoTables<'_>,
    core: CoreId,
    info_type: u32,
    handle: Handle,
    info_sub_id: u64,
) -> CallResult<u64> {
    let id = InfoType::from_raw(info_type).ok_or(KernelError::InvalidEnumValue)?;
    let InfoTables {
        handles,
        space,
        scheduler,
        processes,
    } = tables;
    let shared = (&*handles, &*scheduler, processes);
    // Eden's first group: its two checks, then the value for the process `named`.
    let process = |value: &dyn Fn(ProcessId) -> CallResult<u64>| {
        process_info(shared, handle, info_sub_id, value)
    };
    let regions = space.regions();
    match id {
        InfoType::CoreMask => process(&|named| Ok(info_of(processes, named)?.core_mask())),
        InfoType::PriorityMask => process(&|named| Ok(info_of(processes, named)?.priority_mask())),
        InfoType::AliasRegionAddress => process(&|_| Ok(regions.alias.start)),
        InfoType::AliasRegionSize => process(&|_| Ok(regions.alias.size())),
        InfoType::HeapRegionAddress => process(&|_| Ok(regions.heap.start)),
        InfoType::HeapRegionSize => process(&|_| Ok(regions.heap.size())),
        InfoType::TotalMemorySize => process(&|_| Ok(space.total_user_physical_memory_size())),
        InfoType::UsedMemorySize => process(&|_| Ok(space.used_user_physical_memory_size())),
        InfoType::UsedNonSystemMemorySize => {
            process(&|_| Ok(space.used_non_system_user_physical_memory_size()))
        }
        InfoType::AslrRegionAddress => process(&|_| Ok(regions.code.start)),
        InfoType::AslrRegionSize => process(&|_| Ok(regions.code.size())),
        InfoType::StackRegionAddress => process(&|_| Ok(regions.stack.start)),
        InfoType::StackRegionSize => process(&|_| Ok(regions.stack.size())),
        InfoType::SystemResourceSizeTotal => {
            process(&|named| Ok(secure_resource_size(info_of(processes, named)?)))
        }
        // The named process must have a row, as Eden's `GetUsedSystemResourceSize` reads it.
        InfoType::SystemResourceSizeUsed => {
            process(&|named| info_of(processes, named).map(|_| SYSTEM_RESOURCE_SIZE_USED))
        }
        InfoType::ProgramId => process(&|named| Ok(info_of(processes, named)?.program_id())),
        InfoType::UserExceptionContextAddress => process(&|_| Ok(USER_EXCEPTION_CONTEXT_ADDRESS)),
        InfoType::TotalNonSystemMemorySize => process(&|_| {
            Ok(non_system_total(
                space.used_non_system_user_physical_memory_size(),
                space.limit().free(),
                space.max_process_memory(),
                space.secure_memory_size(),
            ))
        }),
        InfoType::IsApplication => process(&|named| Ok(is_application(info_of(processes, named)?))),
        InfoType::FreeThreadCount => process(&|named| free_thread_count(shared, space, named)),
        InfoType::AliasRegionExtraSize => process(&|_| Ok(regions.alias_region_extra_size)),
        // `case InfoType::DebuggerAttached: *result = 0; R_SUCCEED();`
        InfoType::DebuggerAttached => Ok(0),
        InfoType::ResourceLimit => {
            resource_limit(handles, scheduler, processes, handle, info_sub_id)
        }
        InfoType::IdleTickCount => idle_tick_count(core, handle, info_sub_id),
        InfoType::RandomEntropy => random_entropy(scheduler, processes.own(), handle, info_sub_id),
        InfoType::InitialProcessIdRange => initial_process_id_range(handle, info_sub_id),
        InfoType::ThreadTickCount => thread_tick_count(shared, core, handle, info_sub_id),
        // PR #4588 answers its ids in an `if` before Eden's `switch`.
        InfoType::RemoteRegionAddress => {
            switch_2_info(shared, handle, info_sub_id, REMOTE_REGION_ADDRESS)
        }
        InfoType::RemoteRegionSize => {
            switch_2_info(shared, handle, info_sub_id, REMOTE_REGION_SIZE)
        }
        InfoType::RemoteMemoryUsed => {
            switch_2_info(shared, handle, info_sub_id, REMOTE_MEMORY_USED)
        }
        InfoType::RemoteMemoryTotal => {
            switch_2_info(shared, handle, info_sub_id, REMOTE_MEMORY_TOTAL)
        }
        InfoType::MemoryMappingAlignment => {
            switch_2_info(shared, handle, info_sub_id, MEMORY_MAPPING_ALIGNMENT)
        }
        InfoType::Unknown37 | InfoType::Unknown38 => {
            tracing::warn!(
                number = id.number(),
                info_sub_id,
                handle = handle.raw(),
                "GetInfo id answered with Eden's stubbed zero"
            );
            Ok(0)
        }
        InfoType::MesosphereMeta => mesosphere_meta(handle, info_sub_id),
        InfoType::MesosphereCurrentProcess => {
            mesosphere_current_process(handles, scheduler, processes, handle, info_sub_id)
        }
    }
}

/// The handle table, the scheduler and the process facts, shared.
type Shared<'a> = (&'a HandleTable, &'a Scheduler, &'a Processes);

/// Eden's first group, the process ids: its two checks, then `value` of the process the handle
/// names (see the module docs).
fn process_info(
    (handles, scheduler, processes): Shared<'_>,
    handle: Handle,
    info_sub_id: u64,
    value: &dyn Fn(ProcessId) -> CallResult<u64>,
) -> CallResult<u64> {
    // `R_UNLESS(info_sub_id == 0, ResultInvalidEnumValue)`.
    if info_sub_id != 0 {
        return Err(KernelError::InvalidEnumValue.into());
    }
    // `R_UNLESS(process.IsNotNull(), ResultInvalidHandle)`.
    value(process_from_handle(handles, scheduler, processes, handle)?)
}

/// `FreeThreadCount` (24) of `process`: its resource limit's thread limit less its current
/// value (`limit_value - current_value`), read through the same values
/// `GetResourceLimitLimitValue` and `GetResourceLimitCurrentValue` answer with
/// ([`resource_limit::values`]), so the three always agree.
fn free_thread_count(
    (handles, scheduler, processes): Shared<'_>,
    space: &AddressSpace,
    process: ProcessId,
) -> CallResult<u64> {
    let limit = scheduler.resource_limit(process)?;
    let tables = LimitTables {
        handles,
        space,
        scheduler,
        processes,
    };
    Ok(resource_limit::values(tables, limit, LimitableResource::ThreadCount)?.free())
}

/// `ResourceLimit` (9): Eden's two checks, then a new handle to the calling process's resource
/// limit in its handle table (`handle_table.Add`), whose failure (`ResultOutOfHandles` for a
/// full table) is the call's result.
fn resource_limit(
    handles: &mut HandleTable,
    scheduler: &mut Scheduler,
    processes: &Processes,
    handle: Handle,
    info_sub_id: u64,
) -> CallResult<u64> {
    // `R_UNLESS(handle == 0, ResultInvalidHandle)`.
    if handle != Handle::INVALID {
        return Err(KernelError::InvalidHandle.into());
    }
    // `R_UNLESS(info_sub_id == 0, ResultInvalidCombination)`.
    if info_sub_id != 0 {
        return Err(KernelError::InvalidCombination.into());
    }
    // `R_TRY(handle_table.Add(system.Kernel(), std::addressof(resource_handle),
    // resource_limit))`. Every process here has a limit, so Eden's `Svc::InvalidHandle` answer
    // for a process with none is never given.
    let object = scheduler.process(processes.own())?.resource_limit();
    let added = handles.add(scheduler.objects_mut(), object)?;
    Ok(u64::from(added.raw()))
}

/// `IdleTickCount` (10) for a call made on `core`: Eden's two checks, then
/// [`IDLE_TICK_COUNT`].
fn idle_tick_count(core: CoreId, handle: Handle, info_sub_id: u64) -> CallResult<u64> {
    // `R_UNLESS(core_valid, ResultInvalidCombination)`, where `core_valid` is
    // `info_sub_id == 0xFFFFFFFFFFFFFFFF || info_sub_id == CurrentPhysicalCoreIndex()`.
    if info_sub_id != ALL_CORES && info_sub_id != u64::from(core.index()) {
        return Err(KernelError::InvalidCombination.into());
    }
    // `R_UNLESS(handle == InvalidHandle, ResultInvalidHandle)`.
    if handle != Handle::INVALID {
        return Err(KernelError::InvalidHandle.into());
    }
    Ok(IDLE_TICK_COUNT)
}

/// `InitialProcessIdRange` (19): Eden's handle check, then its stub values by sub id.
fn initial_process_id_range(handle: Handle, info_sub_id: u64) -> CallResult<u64> {
    tracing::warn!("GetInfo InitialProcessIdRange answered with Eden's stubbed 0/64");
    // `R_UNLESS(handle == InvalidHandle, ResultInvalidHandle)`.
    if handle != Handle::INVALID {
        return Err(KernelError::InvalidHandle.into());
    }
    match info_sub_id {
        INITIAL_PROCESS_ID_RANGE_MINIMUM => Ok(INITIAL_PROCESS_ID_MINIMUM),
        INITIAL_PROCESS_ID_RANGE_MAXIMUM => Ok(INITIAL_PROCESS_ID_MAXIMUM),
        // `default: R_THROW(ResultInvalidCombination);`
        _ => Err(KernelError::InvalidCombination.into()),
    }
}

/// The cores `ThreadTickCount` (25) accepts an index below, for a process built for `console`:
/// PR #4588's `UsesPAuthReturns() ? NUM_CPU_CORES : NUM_LEGACY_CPU_CORES`.
const fn tick_count_cores(console: Console) -> u8 {
    match console {
        Console::Switch2 => APPLICATION_CORE_COUNT,
        Console::Switch1 => LEGACY_VIRTUAL_CORE_COUNT,
    }
}

/// `ThreadTickCount` (25) for a call made on `core`: PR #4588's two checks, then
/// [`THREAD_TICK_COUNT`].
fn thread_tick_count(
    (handles, scheduler, processes): Shared<'_>,
    core: CoreId,
    handle: Handle,
    info_sub_id: u64,
) -> CallResult<u64> {
    let cores = u64::from(tick_count_cores(console_of(processes, processes.own())?));
    // `if (info_sub_id != 0xFFFFFFFFFFFFFFFF && info_sub_id >= num_cpus)
    // R_THROW(ResultInvalidCombination)`.
    if info_sub_id != ALL_CORES && info_sub_id >= cores {
        return Err(KernelError::InvalidCombination.into());
    }
    // `if (thread.IsNull()) R_THROW(ResultInvalidHandle)`.
    thread_from_handle(scheduler, handles, core, handle)?;
    Ok(THREAD_TICK_COUNT)
}

/// `MesosphereMeta` (65000): Eden's handle check, then the value by sub id.
fn mesosphere_meta(handle: Handle, info_sub_id: u64) -> CallResult<u64> {
    // `R_UNLESS(handle == InvalidHandle, ResultInvalidHandle)`.
    if handle != Handle::INVALID {
        return Err(KernelError::InvalidHandle.into());
    }
    match info_sub_id {
        MESOSPHERE_META_KERNEL_VERSION => Ok(u64::from(SUPPORTED_KERNEL_VERSION)),
        MESOSPHERE_META_IS_KTRACE_ENABLED | MESOSPHERE_META_IS_SINGLE_STEP_ENABLED => Ok(0),
        // `default: R_THROW(ResultInvalidCombination);`
        _ => Err(KernelError::InvalidCombination.into()),
    }
}

/// `MesosphereCurrentProcess` (65001): Eden's two checks, then a new handle to the calling
/// process in its handle table.
fn mesosphere_current_process(
    handles: &mut HandleTable,
    scheduler: &mut Scheduler,
    processes: &Processes,
    handle: Handle,
    info_sub_id: u64,
) -> CallResult<u64> {
    // `R_UNLESS(handle == InvalidHandle, ResultInvalidHandle)`.
    if handle != Handle::INVALID {
        return Err(KernelError::InvalidHandle.into());
    }
    // `R_UNLESS(info_sub_id == 0, ResultInvalidCombination)`.
    if info_sub_id != 0 {
        return Err(KernelError::InvalidCombination.into());
    }
    // `R_TRY(handle_table.Add(system.Kernel(), std::addressof(tmp), current_process))`.
    let object = scheduler.process(processes.own())?.object();
    let added = handles.add(scheduler.objects_mut(), object)?;
    Ok(u64::from(added.raw()))
}

/// A Switch 2 id, whose answer is `value`: for a Switch 1 program, Eden's `default`; for a
/// Switch 2 program, PR #4588's three checks in its order (see the module docs).
fn switch_2_info(
    (handles, scheduler, processes): Shared<'_>,
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
    tracing::debug!(
        value,
        "GetInfo Switch 2 id answered with PR #4588's guessed value"
    );
    Ok(value)
}

/// The facts of `process`. A process the table has no row for is an internal failure: a
/// process object or the calling process always names a created process.
fn info_of(processes: &Processes, process: ProcessId) -> CallResult<&ProcessInfo> {
    processes
        .get(process)
        .map_err(|error| SvcError::internal(&error))
}

/// The console `process` was built for ([`info_of`]).
fn console_of(processes: &Processes, process: ProcessId) -> CallResult<Console> {
    info_of(processes, process).map(ProcessInfo::console)
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

#[cfg(test)]
mod tests;
