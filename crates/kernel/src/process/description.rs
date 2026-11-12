//! What a process is created from ([`ProcessDescription`]), what the kernel keeps of it
//! ([`ProcessInfo`]), and the translation from the cores a program names (virtual) to the cores
//! the scheduler has (physical).
//!
//! Ported from Eden @ cf654052 and Eden PR #4588 @ e5fb3b97 (GPL-3.0-or-later), paths relative
//! to the reference folder:
//!
//! - `eden/src/core/hle/kernel/k_capabilities.cpp`, `SetCorePriorityCapability`: the checks on
//!   the core and priority ranges, in its order (see [`ProcessInfo::new`]).
//! - `eden/src/core/hle/kernel/k_process.h`: `CheckThreadPriority` (a priority is allowed when
//!   its bit is in the priority mask), `GetIdealCoreId` (the NPDM's main thread core,
//!   `LoadFromMetadata`).
//! - `eden-pr4588/src/core/hardware_properties.h`: `VirtualToPhysicalCoreMap`
//!   ([`VIRTUAL_TO_PHYSICAL_CORE`]), `NUM_LEGACY_CPU_CORES` and `NumVirtualCores`.
//! - `eden-pr4588/src/core/hle/kernel/svc/svc_thread.cpp`, `IsValidVirtualCoreId`: a process built
//!   for Switch 2 (`UsesPAuthReturns`, set from `UsesExtendedAcid`) may name any of the 64
//!   virtual cores; any other process only cores 0-3 ([`Console::virtual_core_count`]).
//! - `eden/src/core/file_sys/program_metadata.cpp`, `GetDefault`: the description of a program
//!   with no NPDM ([`ProcessDescription::homebrew`]).
//!
//! # Divergences
//!
//! - Eden asserts in `KThread::Initialize` that the main thread's core is in the process's core
//!   mask and its priority in the priority mask. Here a description that breaks either is
//!   refused when the process is created ([`ProcessError::MainThreadCore`],
//!   [`ProcessError::MainThreadPriority`]). Test:
//!   `a_main_thread_outside_the_process_ranges_is_refused`.
//! - A priority range ending past [`LOWEST_THREAD_PRIORITY`] is
//!   [`KernelError::InvalidArgument`]. Eden's capability word has six bits for it, so it cannot
//!   arise there, and Eden asserts it (`ASSERT(max_prio < Common::BitSize<u64>())`). Test:
//!   `a_kernel_priority_or_one_past_the_lowest_is_an_invalid_argument`.
//! - Only the 39-bit address space is built (`crate::memory::layout`); a description that asks
//!   for another is [`ProcessError::AddressSpace`]. Eden lays the others out. Test:
//!   `only_the_39_bit_address_space_is_accepted`.

use thiserror::Error;

use crate::result::KernelError;
use crate::sched::{AffinityMask, CoreId, LOWEST_THREAD_PRIORITY, MAX_CORES, Priority};
use crate::time::CounterRate;

/// The number of virtual cores a program can name: one per bit of a `u64` core mask
/// (`NumVirtualCores = Common::BitSize<u64>()`, `eden-pr4588/src/core/hardware_properties.h`).
pub const VIRTUAL_CORE_COUNT: u8 = MAX_CORES;

/// The virtual cores a process not built for Switch 2 may name in a thread call: Eden PR #4588's
/// `NUM_LEGACY_CPU_CORES = 4`.
pub const LEGACY_VIRTUAL_CORE_COUNT: u8 = 4;

/// The physical core each virtual core runs on: Eden PR #4588's `VirtualToPhysicalCoreMap`
/// (`eden-pr4588/src/core/hardware_properties.h` @ e5fb3b97). Virtual cores 0-5 are the
/// application cores; 6, 7 and 63 (the system modules' cores) map to the system core, 3; every
/// other virtual core maps to core 0.
pub const VIRTUAL_TO_PHYSICAL_CORE: [u8; VIRTUAL_CORE_COUNT as usize] = [
    0, 1, 2, 3, 4, 5, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, //
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3,
];

/// Priorities a process may never give its threads: 0-3 are the kernel's
/// (`SetCorePriorityCapability`: `R_UNLESS((m_priority_mask & 0xF) == 0, ResultInvalidArgument)`).
pub const KERNEL_PRIORITY_MASK: u64 = 0xF;

/// The physical core a virtual core runs on, or `None` for a virtual core at or above
/// [`VIRTUAL_CORE_COUNT`].
#[must_use]
pub fn physical_core(virtual_core: u8) -> Option<CoreId> {
    let physical = *VIRTUAL_TO_PHYSICAL_CORE.get(usize::from(virtual_core))?;
    CoreId::new(physical)
}

/// The physical cores a virtual core mask names (`ConvertVirtualCoreMaskToPhysical`): each set
/// bit through [`VIRTUAL_TO_PHYSICAL_CORE`].
#[must_use]
pub fn physical_mask(virtual_mask: u64) -> AffinityMask {
    let bits = AffinityMask::from_bits(virtual_mask)
        .iter()
        .filter_map(|virtual_core| physical_core(virtual_core.index()))
        .fold(0, |bits, core| bits | core.bit());
    AffinityMask::from_bits(bits)
}

/// Which console a program was built for. It decides the rate of the process's counter (kernel
/// spec, "Guest time") and which virtual cores its thread calls may name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Console {
    /// The first Switch: a 19.2 MHz counter, virtual cores 0-3.
    Switch1,
    /// Switch 2 (an extended ACID): a 31.25 MHz counter, any of the 64 virtual cores.
    Switch2,
}

impl Console {
    /// The rate of the counter a process built for this console reads: the kernel's choice, the
    /// one place it is made.
    #[must_use]
    pub const fn counter_rate(self) -> CounterRate {
        match self {
            Self::Switch1 => CounterRate::SWITCH_1,
            Self::Switch2 => CounterRate::SWITCH_2,
        }
    }

    /// How many virtual cores, from 0, a thread call of such a process may name
    /// (`IsValidVirtualCoreId` in Eden PR #4588).
    #[must_use]
    pub const fn virtual_core_count(self) -> u8 {
        match self {
            Self::Switch1 => LEGACY_VIRTUAL_CORE_COUNT,
            Self::Switch2 => VIRTUAL_CORE_COUNT,
        }
    }

    /// Whether a thread call of such a process may name `core_id` (`IsValidVirtualCoreId`).
    #[must_use]
    pub fn is_valid_virtual_core(self, core_id: i32) -> bool {
        (0..i32::from(self.virtual_core_count())).contains(&core_id)
    }
}

/// The width of a program's address space, as its NPDM gives it (`ProgramAddressSpaceType`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AddressSpaceWidth {
    /// 32 bits.
    Is32Bit,
    /// 36 bits.
    Is36Bit,
    /// 32 bits, with no map region.
    Is32BitNoMap,
    /// 39 bits: the only width the kernel lays out.
    Is39Bit,
}

/// An inclusive range of cores or priorities, as an NPDM's thread-info capability gives it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Range {
    /// The first value.
    pub first: u8,
    /// The last value, included.
    pub last: u8,
}

/// What a process is created from: the values the loader's NPDM parser produces.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProcessDescription {
    /// The program's name.
    pub name: String,
    /// The program id (the ACI's title id).
    pub program_id: u64,
    /// The width of its address space.
    pub address_space: AddressSpaceWidth,
    /// The virtual cores its threads may use (`minimum_core_id..=maximum_core_id`).
    pub cores: Range,
    /// The priorities its threads may have, most urgent first
    /// (`highest_thread_priority..=lowest_thread_priority`).
    pub priorities: Range,
    /// The main thread's priority.
    pub main_thread_priority: u8,
    /// The main thread's virtual core, and the process's ideal core (`GetIdealCoreId`).
    pub main_thread_core: u8,
    /// Bytes of the main thread's stack.
    pub main_thread_stack_size: u64,
    /// Bytes of the system resource.
    pub system_resource_size: u32,
    /// The handle table's size; zero or less means the maximum (`KHandleTable::Initialize`).
    pub handle_table_size: i32,
    /// The console it was built for.
    pub console: Console,
}

impl ProcessDescription {
    /// The description Eden gives a program that has no NPDM, as homebrew NROs do
    /// (`ProgramMetadata::GetDefault`, `eden/src/core/file_sys/program_metadata.cpp` @
    /// cf654052): cores 0-3, priorities 16-63 (capability `0x30043F7`), the main thread at
    /// priority 44 (`0x2c`) on core 0 with a 1 MiB stack, program id zero, no system resource,
    /// a 39-bit address space and the largest handle table. The console is the caller's: an
    /// NRO does not say which it was built for.
    #[must_use]
    pub fn homebrew(console: Console) -> Self {
        Self {
            name: String::new(),
            program_id: 0,
            address_space: AddressSpaceWidth::Is39Bit,
            cores: Range { first: 0, last: 3 },
            priorities: Range {
                first: 16,
                last: LOWEST_THREAD_PRIORITY,
            },
            main_thread_priority: 0x2c,
            main_thread_core: 0,
            main_thread_stack_size: 0x10_0000,
            system_resource_size: 0,
            handle_table_size: 0,
            console,
        }
    }
}

/// Why a process could not be created from its description.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ProcessError {
    /// A capability check failed, with the result Eden's `SetCorePriorityCapability` gives.
    #[error("the description's thread capability is refused: {0}")]
    Capability(KernelError),
    /// The main thread's core is not in the process's core range.
    #[error("the main thread's core {core} is outside the process's cores")]
    MainThreadCore {
        /// The main thread's virtual core.
        core: u8,
    },
    /// The main thread's priority is not in the process's priority range.
    #[error("the main thread's priority {priority} is outside the process's priorities")]
    MainThreadPriority {
        /// The main thread's priority.
        priority: u8,
    },
    /// The address space is not 39 bits wide.
    #[error("only a 39-bit address space is built, not {0:?}")]
    AddressSpace(AddressSpaceWidth),
    /// The process table could not take another row.
    #[error("the process table is full")]
    TableFull,
    /// Process ids ran out.
    #[error("no process id is left")]
    OutOfIds,
}

/// What the kernel keeps of a process's description: the facts fixed at creation that system
/// calls check (kernel spec: the `Processes` table).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProcessInfo {
    name: String,
    process_id: u64,
    program_id: u64,
    core_mask: u64,
    priority_mask: u64,
    ideal_core: u8,
    main_thread_priority: Priority,
    main_thread_stack_size: u64,
    system_resource_size: u32,
    handle_table_size: i32,
    console: Console,
}

/// A mask with bits `range.first..=range.last` set; `range.last` is below 64.
fn range_mask(range: Range) -> u64 {
    (range.first..=range.last).fold(0, |mask, bit| mask | (1_u64 << bit))
}

impl ProcessInfo {
    /// Checks `description` and keeps what the kernel needs of it, for the process with guest id
    /// `process_id`.
    ///
    /// The checks are `SetCorePriorityCapability`'s, in its order: an empty core range, then an
    /// empty priority range, is [`KernelError::InvalidCombination`]; a last core at or above
    /// [`VIRTUAL_CORE_COUNT`] is [`KernelError::InvalidCoreId`]; a priority above
    /// [`LOWEST_THREAD_PRIORITY`] (Eden asserts the range fits a `u64` mask) or a priority range
    /// that includes a kernel priority ([`KERNEL_PRIORITY_MASK`]) is
    /// [`KernelError::InvalidArgument`]; all inside [`ProcessError::Capability`]. Then the main
    /// thread's core and priority must be in the ranges, and the address space 39 bits wide.
    pub fn new(description: &ProcessDescription, process_id: u64) -> Result<Self, ProcessError> {
        let refused = ProcessError::Capability;
        let ProcessDescription {
            cores, priorities, ..
        } = *description;
        if cores.first > cores.last || priorities.first > priorities.last {
            return Err(refused(KernelError::InvalidCombination));
        }
        if cores.last >= VIRTUAL_CORE_COUNT {
            return Err(refused(KernelError::InvalidCoreId));
        }
        if priorities.last > LOWEST_THREAD_PRIORITY {
            return Err(refused(KernelError::InvalidArgument));
        }
        let core_mask = range_mask(cores);
        let priority_mask = range_mask(priorities);
        if priority_mask & KERNEL_PRIORITY_MASK != 0 {
            return Err(refused(KernelError::InvalidArgument));
        }
        let core = description.main_thread_core;
        if core >= VIRTUAL_CORE_COUNT || core_mask & (1 << core) == 0 {
            return Err(ProcessError::MainThreadCore { core });
        }
        let priority = description.main_thread_priority;
        let main_thread_priority = Priority::new(priority)
            .filter(|p| priority_mask & (1 << p.get()) != 0)
            .ok_or(ProcessError::MainThreadPriority { priority })?;
        if description.address_space != AddressSpaceWidth::Is39Bit {
            return Err(ProcessError::AddressSpace(description.address_space));
        }
        Ok(Self {
            name: description.name.clone(),
            process_id,
            program_id: description.program_id,
            core_mask,
            priority_mask,
            ideal_core: core,
            main_thread_priority,
            main_thread_stack_size: description.main_thread_stack_size,
            system_resource_size: description.system_resource_size,
            handle_table_size: description.handle_table_size,
            console: description.console,
        })
    }

    /// The program's name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The process id as the guest knows it (`GetProcessId`).
    #[must_use]
    pub const fn process_id(&self) -> u64 {
        self.process_id
    }

    /// The program id.
    #[must_use]
    pub const fn program_id(&self) -> u64 {
        self.program_id
    }

    /// The virtual cores its threads may use (`GetCoreMask`).
    #[must_use]
    pub const fn core_mask(&self) -> u64 {
        self.core_mask
    }

    /// The priorities its threads may have, one bit each (`GetPriorityMask`).
    #[must_use]
    pub const fn priority_mask(&self) -> u64 {
        self.priority_mask
    }

    /// Whether its threads may have `priority` (`CheckThreadPriority`). A priority outside
    /// 0-63 has no bit, so it is never allowed.
    #[must_use]
    pub fn check_thread_priority(&self, priority: i32) -> bool {
        u32::try_from(priority)
            .ok()
            .and_then(|bit| 1_u64.checked_shl(bit))
            .is_some_and(|bit| bit & self.priority_mask != 0)
    }

    /// The virtual core its threads go on when a call asks for the process's value
    /// (`GetIdealCoreId`): the main thread's.
    #[must_use]
    pub const fn ideal_core(&self) -> u8 {
        self.ideal_core
    }

    /// The main thread's priority.
    #[must_use]
    pub const fn main_thread_priority(&self) -> Priority {
        self.main_thread_priority
    }

    /// Bytes of the main thread's stack.
    #[must_use]
    pub const fn main_thread_stack_size(&self) -> u64 {
        self.main_thread_stack_size
    }

    /// Bytes of the system resource.
    #[must_use]
    pub const fn system_resource_size(&self) -> u32 {
        self.system_resource_size
    }

    /// The handle table's size, as [`crate::handle::HandleTable::new`] takes it.
    #[must_use]
    pub const fn handle_table_size(&self) -> i32 {
        self.handle_table_size
    }

    /// The console it was built for.
    #[must_use]
    pub const fn console(&self) -> Console {
        self.console
    }

    /// The rate of its counter.
    #[must_use]
    pub const fn counter_rate(&self) -> CounterRate {
        self.console.counter_rate()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sched::HIGHEST_THREAD_PRIORITY;

    fn homebrew() -> ProcessDescription {
        ProcessDescription::homebrew(Console::Switch1)
    }

    fn refused(error: KernelError) -> Result<ProcessInfo, ProcessError> {
        Err(ProcessError::Capability(error))
    }

    #[test]
    fn the_homebrew_description_is_edens_default_capability() {
        // `0x30043F7` decodes (`KCapabilities::CorePriority`: bits 4-9 lowest priority, 10-15
        // highest priority, 16-23 minimum core, 24-31 maximum core) to priorities 16..=63 and
        // cores 0..=3; checked against the packed word so the decode itself is pinned.
        let word: u32 = 0x0300_43F7;
        let lowest = (word >> 4) & 0x3F;
        let highest = (word >> 10) & 0x3F;
        let (min_core, max_core) = ((word >> 16) & 0xFF, (word >> 24) & 0xFF);
        let description = homebrew();
        assert_eq!(u32::from(description.priorities.first), highest);
        assert_eq!(u32::from(description.priorities.last), lowest);
        assert_eq!(u32::from(description.cores.first), min_core);
        assert_eq!(u32::from(description.cores.last), max_core);
        let info = ProcessInfo::new(&description, 0x51).unwrap();
        assert_eq!(info.core_mask(), 0b1111);
        assert_eq!(info.priority_mask(), u64::MAX << 16);
        assert_eq!(info.main_thread_priority().get(), 44);
        assert_eq!(info.ideal_core(), 0);
        assert_eq!(info.process_id(), 0x51);
    }

    #[test]
    fn the_capability_checks_come_in_edens_order() {
        // An empty core range and an empty priority range at once: the core check is first,
        // and both give the same result.
        let mut both = homebrew();
        both.cores = Range { first: 2, last: 1 };
        both.priorities = Range {
            first: 40,
            last: 30,
        };
        assert_eq!(
            ProcessInfo::new(&both, 1),
            refused(KernelError::InvalidCombination)
        );
        // A core range ending past the virtual cores, with an empty priority range: the empty
        // range is checked first.
        let mut order = homebrew();
        order.cores = Range {
            first: 0,
            last: VIRTUAL_CORE_COUNT,
        };
        order.priorities = Range {
            first: 40,
            last: 30,
        };
        assert_eq!(
            ProcessInfo::new(&order, 1),
            refused(KernelError::InvalidCombination)
        );
        order.priorities = homebrew().priorities;
        assert_eq!(
            ProcessInfo::new(&order, 1),
            refused(KernelError::InvalidCoreId)
        );
        // The last virtual core itself is allowed.
        let mut top = homebrew();
        top.cores.last = VIRTUAL_CORE_COUNT - 1;
        assert!(ProcessInfo::new(&top, 1).is_ok());
    }

    #[test]
    fn a_kernel_priority_or_one_past_the_lowest_is_an_invalid_argument() {
        let mut kernel = homebrew();
        // 3 is the last kernel priority, 4 the first a process may have.
        let first_allowed = u8::try_from(KERNEL_PRIORITY_MASK.count_ones()).unwrap();
        kernel.priorities.first = first_allowed - 1;
        assert_eq!(
            ProcessInfo::new(&kernel, 1),
            refused(KernelError::InvalidArgument)
        );
        kernel.priorities.first = first_allowed;
        assert!(ProcessInfo::new(&kernel, 1).is_ok());
        let mut past = homebrew();
        past.priorities.last = LOWEST_THREAD_PRIORITY + 1;
        assert_eq!(
            ProcessInfo::new(&past, 1),
            refused(KernelError::InvalidArgument)
        );
    }

    #[test]
    fn a_main_thread_outside_the_process_ranges_is_refused() {
        let mut core = homebrew();
        core.main_thread_core = core.cores.last + 1;
        assert_eq!(
            ProcessInfo::new(&core, 1),
            Err(ProcessError::MainThreadCore {
                core: core.cores.last + 1
            })
        );
        core.main_thread_core = core.cores.last;
        assert!(ProcessInfo::new(&core, 1).is_ok());
        let mut priority = homebrew();
        priority.main_thread_priority = priority.priorities.first - 1;
        assert_eq!(
            ProcessInfo::new(&priority, 1),
            Err(ProcessError::MainThreadPriority {
                priority: priority.priorities.first - 1
            })
        );
        priority.main_thread_priority = priority.priorities.first;
        assert!(ProcessInfo::new(&priority, 1).is_ok());
    }

    #[test]
    fn only_the_39_bit_address_space_is_accepted() {
        for width in [
            AddressSpaceWidth::Is32Bit,
            AddressSpaceWidth::Is36Bit,
            AddressSpaceWidth::Is32BitNoMap,
        ] {
            let mut description = homebrew();
            description.address_space = width;
            assert_eq!(
                ProcessInfo::new(&description, 1),
                Err(ProcessError::AddressSpace(width))
            );
        }
    }

    #[test]
    fn a_thread_priority_is_allowed_by_its_bit_in_the_mask() {
        let info = ProcessInfo::new(&homebrew(), 1).unwrap();
        let first = i32::from(homebrew().priorities.first);
        assert!(!info.check_thread_priority(first - 1));
        assert!(info.check_thread_priority(first));
        assert!(info.check_thread_priority(i32::from(LOWEST_THREAD_PRIORITY)));
        for outside in [
            i32::from(HIGHEST_THREAD_PRIORITY) - 1,
            i32::from(LOWEST_THREAD_PRIORITY) + 1,
            i32::MIN,
            i32::MAX,
        ] {
            assert!(!info.check_thread_priority(outside), "{outside}");
        }
    }

    #[test]
    fn the_console_decides_the_rate_and_the_virtual_cores() {
        assert_eq!(Console::Switch1.counter_rate(), CounterRate::SWITCH_1);
        assert_eq!(Console::Switch2.counter_rate(), CounterRate::SWITCH_2);
        for (console, count) in [
            (Console::Switch1, LEGACY_VIRTUAL_CORE_COUNT),
            (Console::Switch2, VIRTUAL_CORE_COUNT),
        ] {
            assert!(!console.is_valid_virtual_core(-1));
            assert!(console.is_valid_virtual_core(0));
            assert!(console.is_valid_virtual_core(i32::from(count) - 1));
            assert!(!console.is_valid_virtual_core(i32::from(count)));
        }
    }

    #[test]
    fn virtual_cores_map_as_eden_pr_4588_maps_them() {
        for core in 0..6 {
            assert_eq!(physical_core(core), CoreId::new(core));
        }
        let system = CoreId::new(3);
        assert_eq!(physical_core(6), system);
        assert_eq!(physical_core(7), system);
        assert_eq!(physical_core(VIRTUAL_CORE_COUNT - 1), system);
        assert_eq!(physical_core(8), CoreId::new(0));
        assert_eq!(physical_core(VIRTUAL_CORE_COUNT), None);
        // Virtual 1, 6 and 63: physical 1 and 3.
        let mask = (1 << 1) | (1 << 6) | (1 << 63);
        assert_eq!(physical_mask(mask).bits(), (1 << 1) | (1 << 3));
        assert_eq!(physical_mask(0), AffinityMask::EMPTY);
    }
}
