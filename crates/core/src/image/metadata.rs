//! A program's `main.npdm` as the kernel's process description: the table in "The process's
//! description" in the parent module's docs says where each value comes from.

use std::ops::RangeInclusive;

use starquake_kernel::process::{AddressSpaceWidth, Console, ProcessDescription, Range};
use starquake_loader::npdm::{AcidLayout, AddressSpaceType, Npdm};

use super::LoadError;

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

fn range(values: &RangeInclusive<u8>) -> Range {
    Range {
        first: *values.start(),
        last: *values.end(),
    }
}

/// The process `npdm` describes. [`LoadError::NoThreadInfo`] when its `ACI0` kernel
/// capabilities have no thread info.
pub(crate) fn describe(npdm: &Npdm) -> Result<ProcessDescription, LoadError> {
    let capabilities = &npdm.aci.kernel_capabilities;
    let threads = capabilities
        .thread_info
        .as_ref()
        .ok_or(LoadError::NoThreadInfo)?;
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
        npdm.main_thread_core = 1;
        npdm.main_thread_stack_size = 0x0003_4000;
        npdm.system_resource_size = 0x0020_0000;
        npdm.aci.kernel_capabilities.thread_info = Some(ThreadInfo {
            priorities: 24..=58,
            cores: 1..=2,
        });
        npdm.aci.kernel_capabilities.handle_table_size = Some(0x123);
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
                cores: Range { first: 1, last: 2 },
                priorities: Range {
                    first: 24,
                    last: 58
                },
                main_thread_priority: 45,
                main_thread_core: 1,
                main_thread_stack_size: 0x0003_4000,
                system_resource_size: 0x0020_0000,
                handle_table_size: 0x123,
                console: Console::Switch1,
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
