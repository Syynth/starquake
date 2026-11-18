//! A program's `main.npdm` as the kernel's process description: the table in "The process's
//! description" in the parent module's docs says where each value comes from.

use std::ops::RangeInclusive;

use starquake_kernel::memory::Pool;
use starquake_kernel::process::{AddressSpaceWidth, Console, ProcessDescription, Range};
use starquake_loader::npdm::{AcidLayout, AddressSpaceType, Npdm, PoolPartition};

use super::{LoadError, SystemResourceFault};

/// The block a system resource size must be a whole number of, as Horizon's loader checks it:
/// `os::MemoryBlockUnitSize`, 2 MiB (fact only, path relative to the reference folder,
/// `atmosphere/libraries/libstratosphere/include/stratosphere/os/os_memory_heap_common.hpp`
/// @ 4e3f3703).
pub const MEMORY_BLOCK_UNIT_SIZE: u32 = 2 << 20;

/// The largest system resource size a program may ask for: `SystemResourceSizeMax =
/// 0x1FE00000` (fact only, `atmosphere/stratosphere/loader/source/ldr_process_creation.cpp`
/// @ 4e3f3703).
pub const SYSTEM_RESOURCE_SIZE_MAX: u32 = 0x1FE0_0000;

/// The application-type bits of a program's info flags (`ProgramInfoFlag_ApplicationTypeMask
/// = 3`, `atmosphere/libraries/libstratosphere/include/stratosphere/ldr/ldr_types.hpp` @
/// 4e3f3703, fact only), which Horizon's loader fills from the `ACI0` program-type capability.
pub(crate) const PROGRAM_TYPE_MASK: u8 = 3;

/// The program type of an application: `ProgramInfoFlag_Application = 1` (same source).
pub(crate) const PROGRAM_TYPE_APPLICATION: u8 = 1;

/// The program type of an applet: `ProgramInfoFlag_Applet = 2` (same source).
pub(crate) const PROGRAM_TYPE_APPLET: u8 = 2;

/// The program type Horizon's loader drops: `ProgramInfoFlag_InvalidType = 3` (same source).
pub(crate) const PROGRAM_TYPE_INVALID: u8 = 3;

/// Bytes of a process's name: `CreateProcessParameter::name` is `std::array<char, 12>`
/// (`eden/src/core/hle/kernel/svc_types.h` @ cf654052), and `KProcess::LoadFromMetadata` copies
/// that many bytes of the metadata's name into it.
pub(crate) const PROCESS_NAME_SIZE: usize = 12;

/// The longest prefix of `name` that fits [`PROCESS_NAME_SIZE`] bytes and ends on a character
/// boundary (see the parent module's divergences).
fn process_name(name: &str) -> String {
    let mut end = name.len().min(PROCESS_NAME_SIZE);
    while !name.is_char_boundary(end) {
        end -= 1;
    }
    name.get(..end).unwrap_or_default().to_owned()
}

/// The kernel's name for an NPDM's address-space type.
const fn width(kind: AddressSpaceType) -> AddressSpaceWidth {
    match kind {
        AddressSpaceType::Is32Bit => AddressSpaceWidth::Is32Bit,
        AddressSpaceType::Is36Bit => AddressSpaceWidth::Is36Bit,
        AddressSpaceType::Is32BitNoMap => AddressSpaceWidth::Is32BitNoMap,
        AddressSpaceType::Is39Bit => AddressSpaceWidth::Is39Bit,
    }
}

/// The console an `ACID` layout is built for: Eden PR #4588 sets `uses_extended_acid` when the
/// `ACID` magic is at its Switch 2 offset, and that makes the process a Switch 2 one.
const fn console(layout: AcidLayout) -> Console {
    match layout {
        AcidLayout::Switch1 => Console::Switch1,
        AcidLayout::Switch2 => Console::Switch2,
    }
}

/// The kernel's pool for an `ACID` pool partition (`static_cast<KMemoryManager::Pool>(
/// metadata.GetPoolPartition())` in Eden's `KProcess::LoadFromMetadata`), or
/// [`LoadError::PoolPartition`] for a number no pool has, as Horizon's loader refuses it
/// (`R_THROW(ldr::ResultInvalidMeta())` in its pool switch's `default`; fact only,
/// `atmosphere/stratosphere/loader/source/ldr_process_creation.cpp` @ 4e3f3703).
const fn pool(partition: PoolPartition) -> Result<Pool, LoadError> {
    match partition {
        PoolPartition::Application => Ok(Pool::Application),
        PoolPartition::Applet => Ok(Pool::Applet),
        PoolPartition::System => Ok(Pool::System),
        PoolPartition::SystemNonSecure => Ok(Pool::SystemNonSecure),
        PoolPartition::Other(number) => Err(LoadError::PoolPartition(number)),
    }
}

/// The application-type bits Horizon's loader takes from the `ACI0` program-type capability
/// (`MakeProgramInfoFlag`: the type masked with [`PROGRAM_TYPE_MASK`], unless it is
/// [`PROGRAM_TYPE_INVALID`]), zero with no such capability (a system module's).
fn program_type(npdm: &Npdm) -> u8 {
    npdm.aci
        .kernel_capabilities
        .program_type
        .map(|kind| kind & PROGRAM_TYPE_MASK)
        .filter(|&kind| kind != PROGRAM_TYPE_INVALID)
        .unwrap_or(0)
}

/// Whether an address space type passes Horizon's loader's test for a system resource,
/// `out->flags & svc::CreateProcessParameterFlag_AddressSpace64Bit39`: a bitwise test of the
/// address-space field against 64-bit-39's value (3), which every type here but the 32-bit one
/// (0) has a bit of. (The loader's comment says "Process must be 64-bit"; the test it makes is
/// this one, so a 36-bit or 32-bit-without-alias process passes it.)
const fn passes_address_space_test(kind: AddressSpaceType) -> bool {
    match kind {
        AddressSpaceType::Is32Bit => false,
        AddressSpaceType::Is36Bit | AddressSpaceType::Is32BitNoMap | AddressSpaceType::Is39Bit => {
            true
        }
    }
}

/// Horizon's loader's checks on the system resource size, in its order
/// (`GetCreateProcessParameter`, fact only,
/// `atmosphere/stratosphere/loader/source/ldr_process_creation.cpp` @ 4e3f3703): a whole
/// number of [`MEMORY_BLOCK_UNIT_SIZE`] blocks (`ldr::ResultInvalidSize`); then, for a size
/// that is not zero, an address space that passes [`passes_address_space_test`], an
/// application or applet program type, and at most [`SYSTEM_RESOURCE_SIZE_MAX`] (each
/// `ldr::ResultInvalidMeta`).
fn check_system_resource(npdm: &Npdm) -> Result<(), SystemResourceFault> {
    let size = npdm.system_resource_size;
    if !size.is_multiple_of(MEMORY_BLOCK_UNIT_SIZE) {
        return Err(SystemResourceFault::Unaligned { size });
    }
    if size == 0 {
        return Ok(());
    }
    if !passes_address_space_test(npdm.address_space) {
        return Err(SystemResourceFault::AddressSpace);
    }
    if !matches!(
        program_type(npdm),
        PROGRAM_TYPE_APPLICATION | PROGRAM_TYPE_APPLET
    ) {
        return Err(SystemResourceFault::ProgramType);
    }
    if size > SYSTEM_RESOURCE_SIZE_MAX {
        return Err(SystemResourceFault::TooLarge { size });
    }
    Ok(())
}

fn range(values: &RangeInclusive<u8>) -> Range {
    Range {
        first: *values.start(),
        last: *values.end(),
    }
}

/// The process `npdm` describes. In this order: [`LoadError::Not64Bit`] when its `META` flags
/// mark it as 32-bit, [`LoadError::NoThreadInfo`] when its `ACI0` kernel capabilities have no
/// thread info, [`LoadError::PoolPartition`] when its `ACID` names no pool ([`pool`]), and
/// [`LoadError::SystemResource`] when its system resource size fails Horizon's loader's checks
/// ([`check_system_resource`]).
pub(crate) fn describe(npdm: &Npdm) -> Result<ProcessDescription, LoadError> {
    // Eden's `KProcess::LoadFromMetadata` sets `CreateProcessFlag::Is64Bit` only when
    // `metadata.Is64BitProgram()`; without it the process runs as AArch32, which is refused here.
    if !npdm.is_64_bit {
        return Err(LoadError::Not64Bit);
    }
    let capabilities = &npdm.aci.kernel_capabilities;
    let threads = capabilities
        .thread_info
        .as_ref()
        .ok_or(LoadError::NoThreadInfo)?;
    let pool = pool(npdm.acid.pool_partition)?;
    check_system_resource(npdm).map_err(LoadError::SystemResource)?;
    Ok(ProcessDescription {
        name: process_name(&npdm.name),
        program_id: npdm.aci.title_id,
        address_space: width(npdm.address_space),
        cores: range(&threads.cores),
        priorities: range(&threads.priorities),
        main_thread_priority: npdm.main_thread_priority,
        main_thread_core: npdm.main_thread_core,
        main_thread_stack_size: u64::from(npdm.main_thread_stack_size),
        system_resource_size: npdm.system_resource_size,
        handle_table_size: capabilities.handle_table_size.map_or(0, i32::from),
        console: console(npdm.acid.layout),
        pool,
    })
}

#[cfg(test)]
mod tests {
    use starquake_loader::npdm::capabilities::ThreadInfo;

    use super::*;

    const NPDM: &[u8] = include_bytes!("../../../../tests/homebrew/hello/main.npdm");

    /// The hello fixture's metadata, with every value `describe` reads set to something
    /// distinct.
    fn npdm() -> Npdm {
        let mut npdm = Npdm::parse(NPDM).unwrap();
        npdm.name = "describe-me-please".to_owned();
        npdm.aci.title_id = 0x0100_0000_0000_1234;
        npdm.main_thread_priority = 45;
        npdm.main_thread_core = 2;
        npdm.main_thread_stack_size = 0x0003_4000;
        npdm.system_resource_size = 0x0040_0000;
        npdm.aci.kernel_capabilities.thread_info = Some(ThreadInfo {
            priorities: 24..=58,
            cores: 1..=3,
        });
        npdm.aci.kernel_capabilities.handle_table_size = Some(0x123);
        npdm.aci.kernel_capabilities.program_type = Some(PROGRAM_TYPE_APPLET);
        npdm.acid.pool_partition = PoolPartition::Applet;
        npdm
    }

    #[test]
    fn each_value_comes_from_its_place_in_the_metadata() {
        assert_eq!(
            describe(&npdm()).unwrap(),
            ProcessDescription {
                name: "describe-me-".to_owned(),
                program_id: 0x0100_0000_0000_1234,
                address_space: AddressSpaceWidth::Is39Bit,
                cores: Range { first: 1, last: 3 },
                priorities: Range {
                    first: 24,
                    last: 58
                },
                main_thread_priority: 45,
                main_thread_core: 2,
                main_thread_stack_size: 0x0003_4000,
                system_resource_size: 0x0040_0000,
                handle_table_size: 0x123,
                console: Console::Switch1,
                pool: Pool::Applet,
            }
        );
    }

    #[test]
    fn the_acid_layout_decides_the_console() {
        let mut switch2 = npdm();
        switch2.acid.layout = AcidLayout::Switch2;
        assert_eq!(describe(&switch2).unwrap().console, Console::Switch2);
        assert_eq!(describe(&npdm()).unwrap().console, Console::Switch1);
    }

    #[test]
    fn no_handle_table_capability_means_the_largest_table() {
        // `m_handle_table_size = 0`, which `KHandleTable::Initialize` reads as the maximum.
        let mut none = npdm();
        none.aci.kernel_capabilities.handle_table_size = None;
        assert_eq!(describe(&none).unwrap().handle_table_size, 0);
    }

    #[test]
    fn every_address_space_type_maps_to_its_width() {
        for (kind, expected) in [
            (AddressSpaceType::Is32Bit, AddressSpaceWidth::Is32Bit),
            (AddressSpaceType::Is36Bit, AddressSpaceWidth::Is36Bit),
            (
                AddressSpaceType::Is32BitNoMap,
                AddressSpaceWidth::Is32BitNoMap,
            ),
            (AddressSpaceType::Is39Bit, AddressSpaceWidth::Is39Bit),
        ] {
            let mut file = npdm();
            file.address_space = kind;
            // No system resource, whose check would refuse the 32-bit type
            // (`only_a_32_bit_address_space_fails_the_address_space_test`).
            file.system_resource_size = 0;
            assert_eq!(describe(&file).unwrap().address_space, expected);
        }
    }

    #[test]
    fn metadata_without_thread_info_is_refused() {
        // Divergence from Eden, which goes on with empty masks (see the parent module's docs).
        let mut none = npdm();
        none.aci.kernel_capabilities.thread_info = None;
        assert_eq!(describe(&none), Err(LoadError::NoThreadInfo));
        assert_eq!(
            LoadError::NoThreadInfo.to_string(),
            "the program's metadata gives its process no cores or priorities"
        );
    }

    #[test]
    fn a_32_bit_program_is_refused() {
        // Divergence from Eden, which creates the process without `Is64Bit` and runs it as
        // AArch32 code (see the parent module's docs).
        let mut a32 = npdm();
        a32.is_64_bit = false;
        assert_eq!(describe(&a32), Err(LoadError::Not64Bit));
        assert_eq!(
            LoadError::Not64Bit.to_string(),
            "the program's metadata marks it as 32-bit (AArch32) code, which cannot be run"
        );
        // Checked before the thread info: with both faults, the width is reported.
        a32.aci.kernel_capabilities.thread_info = None;
        assert_eq!(describe(&a32), Err(LoadError::Not64Bit));
        // The fixture is 64-bit and is described.
        assert!(npdm().is_64_bit);
        assert!(describe(&npdm()).is_ok());
    }

    #[test]
    fn each_pool_partition_names_its_pool_and_one_no_pool_has_is_refused() {
        // `static_cast<KMemoryManager::Pool>(metadata.GetPoolPartition())`, the numbers written
        // out; past `SystemNonSecure` Horizon's loader refuses the metadata.
        for (partition, expected) in [
            (PoolPartition::Application, Pool::Application),
            (PoolPartition::Applet, Pool::Applet),
            (PoolPartition::System, Pool::System),
            (PoolPartition::SystemNonSecure, Pool::SystemNonSecure),
        ] {
            let mut file = npdm();
            file.acid.pool_partition = partition;
            assert_eq!(describe(&file).unwrap().pool, expected, "{partition:?}");
        }
        for number in [4, 0xF] {
            let mut file = npdm();
            file.acid.pool_partition = PoolPartition::Other(number);
            assert_eq!(describe(&file), Err(LoadError::PoolPartition(number)));
        }
        assert_eq!(
            LoadError::PoolPartition(4).to_string(),
            "the program's metadata names memory pool 4, which does not exist"
        );
        // After the thread info: with both faults, the thread info is reported.
        let mut both = npdm();
        both.acid.pool_partition = PoolPartition::Other(4);
        both.aci.kernel_capabilities.thread_info = None;
        assert_eq!(describe(&both), Err(LoadError::NoThreadInfo));
    }

    /// The fixture's metadata with a system resource of `size` bytes.
    fn with_resource(size: u32) -> Npdm {
        let mut file = npdm();
        file.system_resource_size = size;
        file
    }

    /// The fault `describe` reports for `file`'s system resource, if any.
    fn fault(file: &Npdm) -> Option<SystemResourceFault> {
        let result = describe(file);
        let fault = match &result {
            Err(LoadError::SystemResource(fault)) => Some(*fault),
            _ => None,
        };
        // Anything but a system resource fault must be a description.
        assert!(fault.is_some() || result.is_ok(), "{result:?}");
        fault
    }

    #[test]
    fn the_constants_are_horizons_loaders() {
        // `os::MemoryBlockUnitSize = 2_MB`, `SystemResourceSizeMax = 0x1FE00000`, and the
        // `ProgramInfoFlag` application types.
        assert_eq!(MEMORY_BLOCK_UNIT_SIZE, 0x20_0000);
        assert_eq!(SYSTEM_RESOURCE_SIZE_MAX, 0x1FE0_0000);
        assert_eq!(
            (
                PROGRAM_TYPE_MASK,
                PROGRAM_TYPE_APPLICATION,
                PROGRAM_TYPE_APPLET,
                PROGRAM_TYPE_INVALID
            ),
            (3, 1, 2, 3)
        );
    }

    #[test]
    fn a_system_resource_must_be_whole_2_mib_blocks() {
        // `R_UNLESS(util::IsAligned(size, os::MemoryBlockUnitSize), ldr::ResultInvalidSize())`,
        // made for every size, zero included.
        let unit = MEMORY_BLOCK_UNIT_SIZE;
        for size in [0, unit, 2 * unit, SYSTEM_RESOURCE_SIZE_MAX] {
            assert_eq!(fault(&with_resource(size)), None, "{size:#x}");
        }
        for size in [1, 0x1000, unit - 1, unit + 1, unit + 0x1000, 2 * unit - 1] {
            assert_eq!(
                fault(&with_resource(size)),
                Some(SystemResourceFault::Unaligned { size }),
                "{size:#x}"
            );
        }
        assert_eq!(
            LoadError::SystemResource(SystemResourceFault::Unaligned { size: 0x1000 }).to_string(),
            "the program's metadata asks for a system resource the console's loader refuses: \
             0x1000 bytes is not a whole number of 2 MiB blocks"
        );
    }

    #[test]
    fn a_system_resource_may_be_up_to_the_maximum_and_no_more() {
        // `R_UNLESS(size <= SystemResourceSizeMax, ldr::ResultInvalidMeta())`: the maximum is a
        // whole number of blocks, so one block more is the first aligned size past it.
        let max = SYSTEM_RESOURCE_SIZE_MAX;
        assert!(max.is_multiple_of(MEMORY_BLOCK_UNIT_SIZE));
        assert_eq!(fault(&with_resource(max)), None);
        let over = max + MEMORY_BLOCK_UNIT_SIZE;
        assert_eq!(
            fault(&with_resource(over)),
            Some(SystemResourceFault::TooLarge { size: over })
        );
        assert_eq!(
            LoadError::SystemResource(SystemResourceFault::TooLarge { size: over }).to_string(),
            "the program's metadata asks for a system resource the console's loader refuses: \
             0x20000000 bytes is more than the 0x1fe00000 bytes a program may ask for"
        );
    }

    #[test]
    fn only_a_32_bit_address_space_fails_the_address_space_test() {
        // `out->flags & CreateProcessParameterFlag_AddressSpace64Bit39`, a bitwise test: every
        // type with a bit of 3 passes. The kernel then refuses all but the 39-bit one
        // (`only_the_39_bit_address_space_is_accepted`, in the kernel).
        for (kind, passes) in [
            (AddressSpaceType::Is32Bit, false),
            (AddressSpaceType::Is36Bit, true),
            (AddressSpaceType::Is32BitNoMap, true),
            (AddressSpaceType::Is39Bit, true),
        ] {
            let mut file = with_resource(MEMORY_BLOCK_UNIT_SIZE);
            file.address_space = kind;
            let expected = (!passes).then_some(SystemResourceFault::AddressSpace);
            assert_eq!(fault(&file), expected, "{kind:?}");
            // With no system resource the test is not made.
            file.system_resource_size = 0;
            assert_eq!(fault(&file), None, "{kind:?}");
        }
        assert_eq!(
            SystemResourceFault::AddressSpace.to_string(),
            "a process with a 32-bit address space cannot have one"
        );
    }

    #[test]
    fn only_an_application_or_an_applet_may_have_a_system_resource() {
        // `R_UNLESS(IsApplication(meta) || IsApplet(meta), ldr::ResultInvalidMeta())`, from the
        // `ACI0` program type masked to its two bits; no program type is a system module's, and
        // the invalid type is dropped.
        for (program_type, allowed) in [
            (None, false),
            (Some(0), false),
            (Some(PROGRAM_TYPE_APPLICATION), true),
            (Some(PROGRAM_TYPE_APPLET), true),
            (Some(PROGRAM_TYPE_INVALID), false),
            // The capability's third bit is outside the mask: 5 is an application, 6 an applet.
            (Some(4), false),
            (Some(5), true),
            (Some(6), true),
            (Some(7), false),
        ] {
            let mut file = with_resource(MEMORY_BLOCK_UNIT_SIZE);
            file.aci.kernel_capabilities.program_type = program_type;
            let expected = (!allowed).then_some(SystemResourceFault::ProgramType);
            assert_eq!(fault(&file), expected, "{program_type:?}");
            file.system_resource_size = 0;
            assert_eq!(fault(&file), None, "{program_type:?}");
        }
        assert_eq!(
            SystemResourceFault::ProgramType.to_string(),
            "only an application or an applet may have one"
        );
    }

    #[test]
    fn the_system_resource_checks_come_in_horizons_order() {
        let unaligned = MEMORY_BLOCK_UNIT_SIZE + 1;
        let over = SYSTEM_RESOURCE_SIZE_MAX + MEMORY_BLOCK_UNIT_SIZE;
        // An unaligned size with every later fault: the alignment.
        let mut all = with_resource(unaligned);
        all.address_space = AddressSpaceType::Is32Bit;
        all.aci.kernel_capabilities.program_type = None;
        assert_eq!(
            fault(&all),
            Some(SystemResourceFault::Unaligned { size: unaligned })
        );
        // The address space before the program type, which is before the size.
        let mut late = with_resource(over);
        late.address_space = AddressSpaceType::Is32Bit;
        late.aci.kernel_capabilities.program_type = None;
        assert_eq!(fault(&late), Some(SystemResourceFault::AddressSpace));
        late.address_space = AddressSpaceType::Is39Bit;
        assert_eq!(fault(&late), Some(SystemResourceFault::ProgramType));
        late.aci.kernel_capabilities.program_type = Some(PROGRAM_TYPE_APPLICATION);
        assert_eq!(
            fault(&late),
            Some(SystemResourceFault::TooLarge { size: over })
        );
        // The pool partition is checked before the system resource.
        let mut pool_too = with_resource(unaligned);
        pool_too.acid.pool_partition = PoolPartition::Other(4);
        assert_eq!(describe(&pool_too), Err(LoadError::PoolPartition(4)));
    }

    #[test]
    fn the_fixtures_metadata_is_an_application_with_no_system_resource() {
        // The homebrew fixture as built: what every test that loads it starts from.
        let file = Npdm::parse(NPDM).unwrap();
        assert_eq!(file.system_resource_size, 0);
        assert_eq!(program_type(&file), PROGRAM_TYPE_APPLICATION);
        assert_eq!(describe(&file).unwrap().pool, Pool::Application);
    }

    #[test]
    fn the_process_name_is_the_first_twelve_bytes_of_the_metadatas() {
        let fits = "a".repeat(PROCESS_NAME_SIZE);
        assert_eq!(process_name(&fits), fits);
        assert_eq!(process_name(&format!("{fits}b")), fits);
        assert_eq!(process_name("short"), "short");
        assert_eq!(process_name(""), "");
        // Divergence: a character the cut would split is left out whole (`é` is two bytes, at
        // bytes 11 and 12).
        let split = format!("{}é", "a".repeat(PROCESS_NAME_SIZE - 1));
        assert_eq!(process_name(&split), "a".repeat(PROCESS_NAME_SIZE - 1));
        // Ending exactly at the limit keeps it.
        let whole = format!("{}é", "a".repeat(PROCESS_NAME_SIZE - 2));
        assert_eq!(process_name(&whole), whole);
    }
}
