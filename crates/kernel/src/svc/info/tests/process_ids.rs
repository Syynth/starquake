//! The ids Eden answers beyond the address-space, entropy, debugger and Switch 2 ids: the
//! process facts (0, 1, 16 to 18, 20 to 24), the resource limit (9), the tick counts (10, 25),
//! the initial process id range (19), the two unknown ids (37, 38) and Atmosphère's two
//! (65000, 65001). Driven against a created and started process whose main thread runs on
//! [`CALLER_CORE`], beside a second process the tables also hold.

use starquake_memory::{FlatMemory, PAGE_SIZE};
use starquake_trace::GuestTicks;

use super::*;
use crate::dispatch::Tables;
use crate::handle::MAX_TABLE_SIZE;
use crate::id::ThreadId;
use crate::memory::GUARD_PAGES;
use crate::process::{LEGACY_VIRTUAL_CORE_COUNT, MainThread, Range};
use crate::sched::{APPLICATION_CORE_COUNT, CoreSignals, UserThread};

struct NoSignals;

impl CoreSignals for NoSignals {
    fn interrupt(&mut self, _: CoreId) {}

    fn wake(&mut self, _: CoreId) {}
}

const ENTRY: u64 = 0x8000_0040;
const STACK_TOP: u64 = 0x10_8000_0000;
/// Pages of guest memory the tests give the thread-local pages: more than they use.
const TLS_PAGES: u64 = 64;

/// The calling process's description: a Switch 2 program on virtual cores 1-4 with
/// priorities 24-59, its main thread on [`CALLER_CORE`], and a system resource of
/// [`OWN_RESOURCE_PAGES`] pages and a few bytes more.
fn own_description(console: Console) -> ProcessDescription {
    let mut description = ProcessDescription::homebrew(console);
    description.program_id = 0x0100_aaaa_bbbb_c000;
    description.cores = Range { first: 1, last: 4 };
    description.priorities = Range {
        first: 24,
        last: 59,
    };
    description.main_thread_core = CALLER_CORE;
    description.main_thread_priority = 44;
    description.system_resource_size =
        u32::try_from(OWN_RESOURCE_PAGES * PAGE_SIZE + OWN_RESOURCE_EXTRA).unwrap();
    description.handle_table_size = i32::from(MAX_TABLE_SIZE);
    description
}

/// Whole pages of the calling process's system resource.
const OWN_RESOURCE_PAGES: u64 = 0x18;
/// Bytes of the calling process's system resource past its whole pages: less than a page.
const OWN_RESOURCE_EXTRA: u64 = 0x123;

/// The second process's description: a Switch 2 program on virtual cores 0-2 with priorities
/// 30-40 and a system resource of [`SECOND_RESOURCE_PAGES`] whole pages; every fact differs
/// from the calling process's.
fn second_description() -> ProcessDescription {
    let mut description = ProcessDescription::homebrew(Console::Switch2);
    description.program_id = 0x0100_dddd_eeee_f000;
    description.cores = Range { first: 0, last: 2 };
    description.priorities = Range {
        first: 30,
        last: 40,
    };
    description.main_thread_priority = 33;
    description.system_resource_size = u32::try_from(SECOND_RESOURCE_PAGES * PAGE_SIZE).unwrap();
    description
}

/// Whole pages of the second process's system resource.
const SECOND_RESOURCE_PAGES: u64 = 0x27;

/// A created and started process whose main thread is current on [`CALLER_CORE`] and whose
/// entropy is [`ENTROPY`], and a second process in the same tables with a handle to it in the
/// calling process's table.
struct Rig {
    tables: Tables,
    main: MainThread,
    second: ProcessId,
    second_handle: Handle,
}

impl Rig {
    /// The rig, its calling process built for `console`.
    fn of(console: Console) -> Self {
        Self::with(&own_description(console))
    }

    /// The rig, its calling process built for Switch 2.
    fn new() -> Self {
        Self::of(Console::Switch2)
    }

    /// The rig, its calling process made from `description`.
    fn with(description: &ProcessDescription) -> Self {
        let mut tables = crate::process::create(description, loaded().0).unwrap();
        let first = tables.address_space.regions().kernel_map.start + GUARD_PAGES * PAGE_SIZE;
        let memory = FlatMemory::new(first, usize::try_from(TLS_PAGES * PAGE_SIZE).unwrap());
        let main = crate::process::run(&mut tables, &memory, ENTRY, STACK_TOP).unwrap();
        // As the dispatcher draws it at the first `RandomEntropy` ask.
        let own = tables.processes.own();
        tables.scheduler.set_entropy(own, ENTROPY).unwrap();
        tables
            .scheduler
            .update(GuestTicks(0), None, &mut NoSignals)
            .unwrap();
        assert_eq!(
            tables.scheduler.schedule(caller_core(), &mut NoSignals),
            Ok(Some(main.thread))
        );
        let second = tables.processes.create(&second_description()).unwrap();
        tables.scheduler.add_process(second).unwrap();
        let object = tables.scheduler.process(second).unwrap().object();
        let second_handle = tables
            .handles
            .add(tables.scheduler.objects_mut(), object)
            .unwrap();
        Self {
            tables,
            main,
            second,
            second_handle,
        }
    }

    /// `GetInfo(id, handle, sub_id)` from `core`.
    fn ask_on(
        &mut self,
        core: CoreId,
        id: InfoType,
        handle: Handle,
        sub_id: u64,
    ) -> CallResult<u64> {
        let Tables {
            handles,
            address_space,
            scheduler,
            processes,
        } = &mut self.tables;
        let tables = InfoTables {
            handles,
            space: address_space,
            scheduler,
            processes,
        };
        get_info(tables, core, id.number(), handle, sub_id)
    }

    /// `GetInfo(id, handle, sub_id)` from [`CALLER_CORE`].
    fn ask(&mut self, id: InfoType, handle: Handle, sub_id: u64) -> CallResult<u64> {
        self.ask_on(caller_core(), id, handle, sub_id)
    }

    /// Creates a thread of `process` (not started), as `CreateThread` does before its handle.
    fn create_thread(&mut self, process: ProcessId) -> (ThreadId, crate::id::ObjectId) {
        let spec = UserThread {
            entry: ENTRY,
            argument: 0,
            stack_top: STACK_TOP,
            priority: Priority::new(44).unwrap(),
            virtual_core: 0,
        };
        self.tables
            .scheduler
            .create_user_thread(&mut self.tables.address_space, process, &spec)
            .unwrap()
    }

    /// The calling process.
    fn own(&self) -> ProcessId {
        self.tables.processes.own()
    }
}

/// The process ids this slice adds to Eden's first group, the ones that were not answered
/// before.
const ADDED_PROCESS_IDS: [InfoType; 10] = [
    InfoType::CoreMask,
    InfoType::PriorityMask,
    InfoType::SystemResourceSizeTotal,
    InfoType::SystemResourceSizeUsed,
    InfoType::ProgramId,
    InfoType::UserExceptionContextAddress,
    InfoType::TotalNonSystemMemorySize,
    InfoType::UsedNonSystemMemorySize,
    InfoType::IsApplication,
    InfoType::FreeThreadCount,
];

/// Handles that name no process: the invalid handle, the current-thread pseudo-handle and a
/// handle no entry has.
const NOT_A_PROCESS: [Handle; 3] = [
    Handle::INVALID,
    Handle::CURRENT_THREAD,
    Handle::from_raw(0x0001_0001),
];

#[test]
fn no_id_eden_answers_is_not_implemented() {
    // Every id, with every handle and sub id a check of any group looks for, from both
    // consoles: none is `NotImplemented`, and each answers some input of a Switch 2 program.
    for console in [Console::Switch1, Console::Switch2] {
        let mut rig = Rig::of(console);
        let handles = [
            Handle::CURRENT_PROCESS,
            Handle::INVALID,
            Handle::CURRENT_THREAD,
            rig.second_handle,
            rig.main.handle,
        ];
        let sub_ids = [
            0,
            INITIAL_PROCESS_ID_RANGE_MAXIMUM,
            MESOSPHERE_META_IS_SINGLE_STEP_ENABLED,
            u64::from(CALLER_CORE),
            ALL_CORES,
        ];
        for &id in InfoType::ALL {
            let mut answered = false;
            for handle in handles {
                for sub_id in sub_ids {
                    let result = rig.ask(id, handle, sub_id);
                    assert_ne!(
                        result,
                        kernel(KernelError::NotImplemented),
                        "{console:?} {id:?} handle {:#x} sub id {sub_id:#x}",
                        handle.raw()
                    );
                    answered |= result.is_ok();
                }
            }
            if console == Console::Switch2 {
                assert!(answered, "{id:?} answered nothing");
            }
        }
    }
}

#[test]
fn only_the_current_process_id_changes_the_tables() {
    // Every id but `MesosphereCurrentProcess` reads; that one adds a handle when it succeeds,
    // and changes nothing when it fails.
    let mut rig = Rig::new();
    for &id in InfoType::ALL {
        for (handle, sub_id) in [
            (Handle::INVALID, 0),
            (Handle::CURRENT_PROCESS, 0),
            (Handle::INVALID, 1),
        ] {
            let (handles, scheduler) = (rig.tables.handles.clone(), rig.tables.scheduler.clone());
            let result = rig.ask(id, handle, sub_id);
            let changed = rig.tables.handles != handles || rig.tables.scheduler != scheduler;
            let adds = id == InfoType::MesosphereCurrentProcess && result.is_ok();
            assert_eq!(
                changed,
                adds,
                "{id:?} handle {:#x} sub id {sub_id}: {result:?}",
                handle.raw()
            );
        }
    }
}

#[test]
fn the_facts_are_the_named_process_s_and_the_sizes_the_caller_s() {
    let mut rig = Rig::new();
    let (own, second) = (rig.own(), rig.second);
    let own_info = rig.tables.processes.get(own).unwrap().clone();
    let second_info = rig.tables.processes.get(second).unwrap().clone();
    // The second process has three threads, the calling process its main thread only.
    for _ in 0..3 {
        rig.create_thread(second);
    }
    let own_description = own_description(Console::Switch2);
    let second_description = second_description();
    let facts = [
        (
            InfoType::CoreMask,
            own_info.core_mask(),
            second_info.core_mask(),
        ),
        (
            InfoType::PriorityMask,
            own_info.priority_mask(),
            second_info.priority_mask(),
        ),
        (
            InfoType::ProgramId,
            own_description.program_id,
            second_description.program_id,
        ),
        (
            InfoType::SystemResourceSizeTotal,
            OWN_RESOURCE_PAGES * PAGE_SIZE,
            SECOND_RESOURCE_PAGES * PAGE_SIZE,
        ),
        (
            InfoType::FreeThreadCount,
            THREAD_COUNT_LIMIT - 1,
            THREAD_COUNT_LIMIT - 3,
        ),
    ];
    for (id, own_value, second_value) in facts {
        assert_ne!(own_value, second_value, "{id:?}");
        assert_eq!(
            rig.ask(id, Handle::CURRENT_PROCESS, 0),
            Ok(own_value),
            "{id:?}"
        );
        assert_eq!(
            rig.ask(id, rig.second_handle, 0),
            Ok(second_value),
            "{id:?}"
        );
    }
    // The address-space sizes are the calling process's, whichever process is named.
    let space = &rig.tables.address_space;
    let used = space.used_user_physical_memory_size();
    let total = non_system_total(
        used,
        space.limit().free(),
        space.max_process_memory(),
        OWN_RESOURCE_PAGES * PAGE_SIZE,
    );
    for (id, value) in [
        (InfoType::UsedNonSystemMemorySize, used),
        (InfoType::TotalNonSystemMemorySize, total),
    ] {
        assert_ne!(value, 0, "{id:?}");
        for handle in [Handle::CURRENT_PROCESS, rig.second_handle] {
            assert_eq!(rig.ask(id, handle, 0), Ok(value), "{id:?}");
        }
    }
}

#[test]
fn the_first_group_checks_the_sub_id_then_the_handle() {
    let mut rig = Rig::new();
    let thread_handle = rig.main.handle;
    for id in ADDED_PROCESS_IDS {
        // `R_UNLESS(info_sub_id == 0, ResultInvalidEnumValue)` first: with the handle wrong
        // too, the sub id is reported.
        for sub_id in [1, ALL_CORES] {
            for handle in [Handle::CURRENT_PROCESS, Handle::INVALID, thread_handle] {
                assert_eq!(
                    rig.ask(id, handle, sub_id),
                    kernel(KernelError::InvalidEnumValue),
                    "{id:?} sub id {sub_id:#x}"
                );
            }
        }
        // Then `R_UNLESS(process.IsNotNull(), ResultInvalidHandle)`: a handle naming no
        // process, or a thread.
        for handle in NOT_A_PROCESS.into_iter().chain([thread_handle]) {
            assert_eq!(
                rig.ask(id, handle, 0),
                kernel(KernelError::InvalidHandle),
                "{id:?} handle {:#x}",
                handle.raw()
            );
        }
        assert!(rig.ask(id, Handle::CURRENT_PROCESS, 0).is_ok(), "{id:?}");
    }
}

#[test]
fn a_fact_about_a_process_with_no_row_is_an_internal_failure_after_the_checks() {
    // A process object always names a created process; one that does not is the kernel's own
    // inconsistency, reported after the guest's checks. The ids that read the caller's
    // address space, or a constant, need no row for the named process.
    let mut rig = Rig::new();
    let ghost = ProcessId::from_parts(9, 7);
    let object = rig
        .tables
        .scheduler
        .objects_mut()
        .create(Object::Process(ghost))
        .unwrap();
    let ghost_handle = rig
        .tables
        .handles
        .add(rig.tables.scheduler.objects_mut(), object)
        .unwrap();
    let no_fact = SvcError::internal(&rig.tables.processes.get(ghost).unwrap_err());
    let no_record = SvcError::from(rig.tables.scheduler.process(ghost).unwrap_err());
    for id in ADDED_PROCESS_IDS {
        let expected = match id {
            InfoType::CoreMask
            | InfoType::PriorityMask
            | InfoType::SystemResourceSizeTotal
            | InfoType::SystemResourceSizeUsed
            | InfoType::ProgramId => Err(no_fact.clone()),
            InfoType::FreeThreadCount => Err(no_record.clone()),
            _ => rig.ask(id, Handle::CURRENT_PROCESS, 0),
        };
        assert_eq!(rig.ask(id, ghost_handle, 0), expected, "{id:?}");
        assert_eq!(
            rig.ask(id, ghost_handle, 1),
            kernel(KernelError::InvalidEnumValue),
            "{id:?}"
        );
    }
}

#[test]
fn the_system_resource_size_is_whole_pages_of_the_recorded_size() {
    // `system_resource_num_pages = GetSystemResourceSize() / PageSize`: a size below one page
    // is none, and a size past a page boundary counts only the whole pages.
    let max = u64::from(u32::MAX);
    for (size, expected) in [
        (0, 0),
        (1, 0),
        (PAGE_SIZE - 1, 0),
        (PAGE_SIZE, PAGE_SIZE),
        (PAGE_SIZE + 1, PAGE_SIZE),
        (2 * PAGE_SIZE - 1, PAGE_SIZE),
        (max, max - (PAGE_SIZE - 1)),
    ] {
        let mut description = ProcessDescription::homebrew(Console::Switch2);
        description.system_resource_size = u32::try_from(size).unwrap();
        let processes = Processes::new(&description).unwrap();
        assert_eq!(
            secure_resource_size(processes.own_info().unwrap()),
            expected,
            "size {size:#x}"
        );
    }
}

#[test]
fn the_system_resource_used_is_zero_whatever_its_size() {
    // Divergence: Eden's is the pages the secure resource has handed out; nothing here keeps
    // that. Both processes have a resource of whole pages.
    let mut rig = Rig::new();
    assert_eq!(SYSTEM_RESOURCE_SIZE_USED, 0);
    for handle in [Handle::CURRENT_PROCESS, rig.second_handle] {
        assert_eq!(
            rig.ask(InfoType::SystemResourceSizeUsed, handle, 0),
            Ok(SYSTEM_RESOURCE_SIZE_USED)
        );
        assert_ne!(rig.ask(InfoType::SystemResourceSizeTotal, handle, 0), Ok(0));
    }
}

#[test]
fn the_non_system_total_takes_edens_branch_at_the_exact_edge() {
    // `used_size + free_size > max_size`, where `used_size` adds the secure size: at the edge
    // the total is `free + used`, one byte of free memory past it is `max - secure`.
    let (used, secure, max) = (0x30_0000, 0x5000, 0x100_0000);
    let edge = max - used - secure;
    assert_eq!(non_system_total(used, edge, max, secure), edge + used);
    assert_eq!(non_system_total(used, edge + 1, max, secure), max - secure);
    // With no secure resource the edge is `free + used == max`.
    assert_eq!(non_system_total(used, max - used, max, 0), max);
    assert_eq!(non_system_total(used, max - used + 1, max, 0), max);
    assert_eq!(non_system_total(used, max - used - 1, max, 0), max - 1);
}

/// Eden's `GetTotalNonSystemUserPhysicalMemorySize`, with the free memory Eden would have:
/// the secure size already reserved from it.
fn edens_non_system_total(used: u64, uncharged_free: u64, max: u64, secure: u64) -> u64 {
    let free = uncharged_free - secure;
    if used + secure + free > max {
        max - secure
    } else {
        free + used
    }
}

#[test]
fn the_non_system_total_is_s_higher_than_edens_while_the_resource_is_not_charged() {
    let (used, secure, max) = (0x30_0000, 0x5000, 0x100_0000);
    // `used + free > max`: the two agree.
    let free = max - used + 1;
    assert_eq!(
        non_system_total(used, free, max, secure),
        edens_non_system_total(used, free, max, secure)
    );
    // `used + secure + free <= max`: `secure` higher here.
    let free = max - used - secure;
    assert_eq!(
        non_system_total(used, free, max, secure),
        edens_non_system_total(used, free, max, secure) + secure
    );
    // In between: `max - secure` here, `free - secure + used` in Eden.
    let free = max - used;
    assert_eq!(non_system_total(used, free, max, secure), max - secure);
    assert_eq!(
        edens_non_system_total(used, free, max, secure),
        free - secure + used
    );
    // The process the rig makes is the second case: its pool is below its maximum, so its
    // total is its free memory and its used memory, its system resource not taken off.
    let mut rig = Rig::new();
    let space = &rig.tables.address_space;
    let (used, free) = (space.used_user_physical_memory_size(), space.limit().free());
    assert!(used + OWN_RESOURCE_PAGES * PAGE_SIZE + free <= space.max_process_memory());
    assert_eq!(
        rig.ask(
            InfoType::TotalNonSystemMemorySize,
            Handle::CURRENT_PROCESS,
            0
        ),
        Ok(free + used)
    );
}

#[test]
fn a_system_resource_larger_than_the_maximum_leaves_no_non_system_total() {
    // Eden refuses to create such a process; here the total is zero, not a wrapped number.
    let (used, free, max) = (0x30_0000, 0x10_0000, 0x100_0000);
    assert_eq!(non_system_total(used, free, max, max + 1), 0);
    assert_eq!(non_system_total(used, free, max, max), 0);
    assert_eq!(non_system_total(used, free, max, max - 1), 1);
    assert_eq!(non_system_total(u64::MAX, u64::MAX, max, u64::MAX), 0);
}

#[test]
fn the_total_and_used_memory_leave_the_system_resource_out() {
    // Divergence: Eden adds the secure size to 7, and to what 6 compares. Here a process with
    // a system resource is told what one without is told, and 7 is 22's value.
    let mut with = Rig::new();
    let mut description = own_description(Console::Switch2);
    description.system_resource_size = 0;
    let mut without = Rig::with(&description);
    for id in [InfoType::TotalMemorySize, InfoType::UsedMemorySize] {
        let value = without.ask(id, Handle::CURRENT_PROCESS, 0);
        assert!(value.is_ok());
        assert_eq!(with.ask(id, Handle::CURRENT_PROCESS, 0), value, "{id:?}");
    }
    assert_eq!(
        with.ask(InfoType::UsedMemorySize, Handle::CURRENT_PROCESS, 0),
        with.ask(
            InfoType::UsedNonSystemMemorySize,
            Handle::CURRENT_PROCESS,
            0
        )
    );
}

#[test]
fn the_exception_context_address_is_zero() {
    // Divergence: no process has a process-local region here.
    let mut rig = Rig::new();
    assert_eq!(USER_EXCEPTION_CONTEXT_ADDRESS, 0);
    for handle in [Handle::CURRENT_PROCESS, rig.second_handle] {
        assert_eq!(
            rig.ask(InfoType::UserExceptionContextAddress, handle, 0),
            Ok(USER_EXCEPTION_CONTEXT_ADDRESS)
        );
    }
}

#[test]
fn every_process_is_an_application() {
    // Divergence: the description keeps no pool partition.
    let mut rig = Rig::new();
    assert_eq!(IS_APPLICATION, 1);
    for handle in [Handle::CURRENT_PROCESS, rig.second_handle] {
        assert_eq!(
            rig.ask(InfoType::IsApplication, handle, 0),
            Ok(IS_APPLICATION)
        );
    }
}

#[test]
fn the_free_thread_count_is_edens_limit_less_the_process_s_threads() {
    let mut rig = Rig::new();
    let (own, second) = (rig.own(), rig.second);
    let free = |rig: &mut Rig, handle| rig.ask(InfoType::FreeThreadCount, handle, 0);
    let second_handle = rig.second_handle;
    // The main thread counts; the second process has none.
    assert_eq!(
        free(&mut rig, Handle::CURRENT_PROCESS),
        Ok(THREAD_COUNT_LIMIT - 1)
    );
    assert_eq!(free(&mut rig, second_handle), Ok(THREAD_COUNT_LIMIT));
    // Two more threads of the calling process: the second's count does not move.
    let (_, first_object) = rig.create_thread(own);
    rig.create_thread(own);
    assert_eq!(
        free(&mut rig, Handle::CURRENT_PROCESS),
        Ok(THREAD_COUNT_LIMIT - 3)
    );
    assert_eq!(free(&mut rig, second_handle), Ok(THREAD_COUNT_LIMIT));
    // A thread destroyed gives its count back (`KThread::PostDestroy`).
    rig.tables.scheduler.close_object(first_object).unwrap();
    assert_eq!(
        free(&mut rig, Handle::CURRENT_PROCESS),
        Ok(THREAD_COUNT_LIMIT - 2)
    );
    rig.create_thread(second);
    assert_eq!(free(&mut rig, second_handle), Ok(THREAD_COUNT_LIMIT - 1));
}

#[test]
fn free_threads_stop_at_zero_past_the_limit() {
    let limit = usize::try_from(THREAD_COUNT_LIMIT).unwrap();
    assert_eq!(free_threads(0), THREAD_COUNT_LIMIT);
    assert_eq!(free_threads(limit - 1), 1);
    assert_eq!(free_threads(limit), 0);
    assert_eq!(free_threads(limit + 1), 0);
    assert_eq!(free_threads(usize::MAX), 0);
    // Eden's value, a guess for an application (see the constant).
    assert_eq!(THREAD_COUNT_LIMIT, 800);
}

/// Handles other than the invalid one, for the ids that take only it.
fn not_invalid(rig: &Rig) -> [Handle; 4] {
    [
        Handle::CURRENT_PROCESS,
        Handle::CURRENT_THREAD,
        rig.second_handle,
        Handle::from_raw(1),
    ]
}

#[test]
fn the_resource_limit_is_the_invalid_handle_after_edens_two_checks() {
    let mut rig = Rig::new();
    let id = InfoType::ResourceLimit;
    let before = rig.tables.handles.clone();
    assert_eq!(
        rig.ask(id, Handle::INVALID, 0),
        Ok(u64::from(Handle::INVALID.raw()))
    );
    // No handle was added.
    assert_eq!(rig.tables.handles, before);
    // `R_UNLESS(handle == 0, ResultInvalidHandle)` comes before the sub id's check.
    for handle in not_invalid(&rig) {
        for sub_id in [0, 1] {
            assert_eq!(
                rig.ask(id, handle, sub_id),
                kernel(KernelError::InvalidHandle),
                "handle {:#x} sub id {sub_id}",
                handle.raw()
            );
        }
    }
    // `R_UNLESS(info_sub_id == 0, ResultInvalidCombination)`.
    for sub_id in [1, ALL_CORES] {
        assert_eq!(
            rig.ask(id, Handle::INVALID, sub_id),
            kernel(KernelError::InvalidCombination)
        );
    }
}

#[test]
fn the_idle_tick_count_checks_the_core_then_the_handle() {
    let mut rig = Rig::new();
    let id = InfoType::IdleTickCount;
    let caller = u64::from(CALLER_CORE);
    assert_eq!(IDLE_TICK_COUNT, 0);
    // Every core, or the calling core's own index.
    for sub_id in [ALL_CORES, caller] {
        assert_eq!(rig.ask(id, Handle::INVALID, sub_id), Ok(IDLE_TICK_COUNT));
    }
    // Any other index, either side of the calling core's, is refused; the core is checked
    // before the handle.
    for sub_id in [0, caller - 1, caller + 1, ALL_CORES - 1] {
        for handle in [Handle::INVALID, Handle::CURRENT_PROCESS] {
            assert_eq!(
                rig.ask(id, handle, sub_id),
                kernel(KernelError::InvalidCombination),
                "sub id {sub_id:#x}"
            );
        }
    }
    for handle in not_invalid(&rig) {
        assert_eq!(
            rig.ask(id, handle, caller),
            kernel(KernelError::InvalidHandle),
            "handle {:#x}",
            handle.raw()
        );
    }
    // The core is the one the call was made on: from core 0, core 0's index is accepted and
    // the calling core's is not.
    let core_0 = CoreId::new(0).unwrap();
    assert_eq!(
        rig.ask_on(core_0, id, Handle::INVALID, 0),
        Ok(IDLE_TICK_COUNT)
    );
    assert_eq!(
        rig.ask_on(core_0, id, Handle::INVALID, caller),
        kernel(KernelError::InvalidCombination)
    );
}

#[test]
fn the_initial_process_id_range_is_edens_stub_after_its_checks() {
    let mut rig = Rig::new();
    let id = InfoType::InitialProcessIdRange;
    assert_eq!(
        rig.ask(id, Handle::INVALID, INITIAL_PROCESS_ID_RANGE_MINIMUM),
        Ok(INITIAL_PROCESS_ID_MINIMUM)
    );
    assert_eq!(
        rig.ask(id, Handle::INVALID, INITIAL_PROCESS_ID_RANGE_MAXIMUM),
        Ok(INITIAL_PROCESS_ID_MAXIMUM)
    );
    // Eden's stub values ("returned 0/64"), and `InitialProcessIdRangeInfo`'s numbering.
    assert_eq!(
        (INITIAL_PROCESS_ID_MINIMUM, INITIAL_PROCESS_ID_MAXIMUM),
        (0, 64)
    );
    assert_eq!(
        (
            INITIAL_PROCESS_ID_RANGE_MINIMUM,
            INITIAL_PROCESS_ID_RANGE_MAXIMUM
        ),
        (0, 1)
    );
    // Any other sub id is refused, after the handle.
    for sub_id in [INITIAL_PROCESS_ID_RANGE_MAXIMUM + 1, ALL_CORES] {
        assert_eq!(
            rig.ask(id, Handle::INVALID, sub_id),
            kernel(KernelError::InvalidCombination),
            "sub id {sub_id:#x}"
        );
    }
    for handle in not_invalid(&rig) {
        for sub_id in [INITIAL_PROCESS_ID_RANGE_MINIMUM, ALL_CORES] {
            assert_eq!(
                rig.ask(id, handle, sub_id),
                kernel(KernelError::InvalidHandle),
                "handle {:#x}",
                handle.raw()
            );
        }
    }
}

#[test]
fn the_thread_tick_count_checks_the_core_count_of_the_console_then_the_handle() {
    let id = InfoType::ThreadTickCount;
    for (console, cores) in [
        (Console::Switch1, LEGACY_VIRTUAL_CORE_COUNT),
        (Console::Switch2, APPLICATION_CORE_COUNT),
    ] {
        let mut rig = Rig::of(console);
        let cores = u64::from(cores);
        // `info_sub_id != 0xFFFFFFFFFFFFFFFF && info_sub_id >= num_cpus`: the last core is
        // accepted, the count itself is not.
        for sub_id in [0, cores - 1, ALL_CORES] {
            assert_eq!(
                rig.ask(id, Handle::CURRENT_THREAD, sub_id),
                Ok(THREAD_TICK_COUNT),
                "{console:?} sub id {sub_id:#x}"
            );
        }
        // Refused before the handle is looked at.
        for sub_id in [cores, cores + 1, ALL_CORES - 1] {
            for handle in [Handle::CURRENT_THREAD, Handle::CURRENT_PROCESS] {
                assert_eq!(
                    rig.ask(id, handle, sub_id),
                    kernel(KernelError::InvalidCombination),
                    "{console:?} sub id {sub_id:#x}"
                );
            }
        }
        // `GetObject<KThread>`: a handle naming no thread.
        for handle in [
            Handle::INVALID,
            Handle::CURRENT_PROCESS,
            rig.second_handle,
            Handle::from_raw(0x0001_0001),
        ] {
            assert_eq!(
                rig.ask(id, handle, 0),
                kernel(KernelError::InvalidHandle),
                "{console:?} handle {:#x}",
                handle.raw()
            );
        }
    }
}

#[test]
fn the_thread_tick_count_is_zero_for_every_thread() {
    // Divergence: no CPU time is kept, so the calling thread is told zero as any other is.
    let mut rig = Rig::new();
    assert_eq!(THREAD_TICK_COUNT, 0);
    let (_, object) = rig.create_thread(rig.second);
    let other = rig
        .tables
        .handles
        .add(rig.tables.scheduler.objects_mut(), object)
        .unwrap();
    for handle in [Handle::CURRENT_THREAD, rig.main.handle, other] {
        for sub_id in [ALL_CORES, u64::from(CALLER_CORE), 0] {
            assert_eq!(
                rig.ask(InfoType::ThreadTickCount, handle, sub_id),
                Ok(THREAD_TICK_COUNT),
                "handle {:#x} sub id {sub_id:#x}",
                handle.raw()
            );
        }
    }
    // The current-thread pseudo-handle is the thread the calling core runs: a core running
    // nothing has none, which is the kernel's failure, not the guest's.
    let idle = CoreId::new(4).unwrap();
    assert_eq!(
        rig.ask_on(idle, InfoType::ThreadTickCount, Handle::CURRENT_THREAD, 0),
        Err(SchedError::NoCurrentThread { core: 4 }.into())
    );
    assert_eq!(
        rig.ask_on(idle, InfoType::ThreadTickCount, rig.main.handle, 0),
        Ok(0)
    );
}

#[test]
fn ids_37_and_38_are_zero_whatever_is_passed() {
    let mut rig = Rig::new();
    for id in [InfoType::Unknown37, InfoType::Unknown38] {
        for handle in not_invalid(&rig).into_iter().chain([Handle::INVALID]) {
            for sub_id in [0, 1, ALL_CORES] {
                assert_eq!(rig.ask(id, handle, sub_id), Ok(0), "{id:?}");
            }
        }
    }
}

#[test]
fn the_mesosphere_meta_id_checks_the_handle_then_the_sub_id() {
    let mut rig = Rig::new();
    let id = InfoType::MesosphereMeta;
    for (sub_id, value) in [
        (
            MESOSPHERE_META_KERNEL_VERSION,
            u64::from(SUPPORTED_KERNEL_VERSION),
        ),
        (MESOSPHERE_META_IS_KTRACE_ENABLED, 0),
        (MESOSPHERE_META_IS_SINGLE_STEP_ENABLED, 0),
    ] {
        assert_eq!(
            rig.ask(id, Handle::INVALID, sub_id),
            Ok(value),
            "sub id {sub_id}"
        );
    }
    // `MesosphereMetaInfo`'s numbering.
    assert_eq!(
        [
            MESOSPHERE_META_KERNEL_VERSION,
            MESOSPHERE_META_IS_KTRACE_ENABLED,
            MESOSPHERE_META_IS_SINGLE_STEP_ENABLED
        ],
        [0, 1, 2]
    );
    for sub_id in [MESOSPHERE_META_IS_SINGLE_STEP_ENABLED + 1, ALL_CORES] {
        assert_eq!(
            rig.ask(id, Handle::INVALID, sub_id),
            kernel(KernelError::InvalidCombination)
        );
        for handle in not_invalid(&rig) {
            assert_eq!(
                rig.ask(id, handle, sub_id),
                kernel(KernelError::InvalidHandle),
                "handle {:#x}",
                handle.raw()
            );
        }
    }
}

/// A kernel version's major and minor numbers, read with the shipping fields.
fn decode_kernel_version(encoded: u32) -> (u32, u32) {
    let field = |shift: u32, width: u32| (encoded >> shift) & ((1 << width) - 1);
    (
        field(KERNEL_VERSION_MAJOR_SHIFT, KERNEL_VERSION_MAJOR_WIDTH),
        field(KERNEL_VERSION_MINOR_SHIFT, KERNEL_VERSION_MINOR_WIDTH),
    )
}

#[test]
fn the_kernel_version_is_encoded_as_edens_bit_fields() {
    // `SupportedKernelVersion`: SDK 22.2, kernel major 26.
    assert_eq!(
        decode_kernel_version(SUPPORTED_KERNEL_VERSION),
        (
            SUPPORTED_SDK_MAJOR_VERSION + SVC_MAJOR_VERSION_OFFSET,
            SUPPORTED_KERNEL_MINOR_VERSION
        )
    );
    // The golden word: major 26 in bits 4-16, minor 2 in bits 0-3.
    assert_eq!(SUPPORTED_KERNEL_VERSION, (26 << 4) | 2);
    // Distinct values at each field's widest come back; wider ones are cut to the field.
    let (major, minor) = (0x1abc, 0x9);
    assert_eq!(
        decode_kernel_version(encode_kernel_version(major, minor)),
        (major, minor)
    );
    let major_max = (1 << KERNEL_VERSION_MAJOR_WIDTH) - 1;
    let minor_max = (1 << KERNEL_VERSION_MINOR_WIDTH) - 1;
    assert_eq!(
        decode_kernel_version(encode_kernel_version(major_max, minor_max)),
        (major_max, minor_max)
    );
    assert_eq!(
        encode_kernel_version(major_max + 1 + 5, minor_max + 1 + 3),
        encode_kernel_version(5, 3)
    );
    // The two fields fill bits 0-16 and nothing else.
    assert_eq!(
        encode_kernel_version(u32::MAX, u32::MAX),
        (1 << (KERNEL_VERSION_MAJOR_SHIFT + KERNEL_VERSION_MAJOR_WIDTH)) - 1
    );
}

#[test]
fn the_current_process_id_adds_a_handle_to_the_calling_process() {
    let mut rig = Rig::new();
    let id = InfoType::MesosphereCurrentProcess;
    let own_object = rig.tables.scheduler.process(rig.own()).unwrap().object();
    let mut added = Vec::new();
    for _ in 0..2 {
        let raw = rig.ask(id, Handle::INVALID, 0).unwrap();
        let handle = Handle::from_raw(u32::try_from(raw).unwrap());
        assert_eq!(
            rig.tables.handles.resolve(handle),
            Ok(Resolved::Object(own_object))
        );
        assert_eq!(
            rig.tables.scheduler.objects().get(own_object),
            Ok(Object::Process(rig.own()))
        );
        added.push(handle);
    }
    // Each ask adds a handle of its own.
    assert_ne!(added[0], added[1]);
    // `R_UNLESS(handle == InvalidHandle, ResultInvalidHandle)`, then
    // `R_UNLESS(info_sub_id == 0, ResultInvalidCombination)`; neither adds a handle.
    let before = rig.tables.handles.clone();
    for handle in not_invalid(&rig) {
        assert_eq!(
            rig.ask(id, handle, 1),
            kernel(KernelError::InvalidHandle),
            "handle {:#x}",
            handle.raw()
        );
    }
    for sub_id in [1, ALL_CORES] {
        assert_eq!(
            rig.ask(id, Handle::INVALID, sub_id),
            kernel(KernelError::InvalidCombination)
        );
    }
    assert_eq!(rig.tables.handles, before);
}

#[test]
fn the_current_process_id_with_a_full_handle_table_is_out_of_handles() {
    // A table of four: the main thread's handle and the rig's handle to the second process
    // leave room for two.
    let mut description = own_description(Console::Switch2);
    description.handle_table_size = 4;
    let mut rig = Rig::with(&description);
    let id = InfoType::MesosphereCurrentProcess;
    for _ in 0..2 {
        assert!(rig.ask(id, Handle::INVALID, 0).is_ok());
    }
    let before = (rig.tables.handles.clone(), rig.tables.scheduler.clone());
    assert_eq!(
        rig.ask(id, Handle::INVALID, 0),
        kernel(KernelError::OutOfHandles)
    );
    assert_eq!(
        (rig.tables.handles.clone(), rig.tables.scheduler.clone()),
        before
    );
}
