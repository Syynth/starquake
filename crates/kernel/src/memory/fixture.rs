//! An address space laid out like the homebrew fixture's (`tests/homebrew/hello/hello.nro`), for
//! this crate's tests: seven pages of code (three of text, one of read-only data, three of data
//! and `.bss`), a main thread's stack, and a recording [`Mapper`](super::Mapper) behind it.

use starquake_memory::PAGE_SIZE;

use super::testing::{MapperLog, RecordingMapper};
use super::{AddressSpace, MemoryArrangement, ProcessMemory, RegionOffsets, SvcPermission};

/// Where the code goes: Eden's address for 39-bit code without randomisation
/// (`KProcess::LoadFromMetadata`).
pub(crate) const CODE: u64 = 0x8000_0000;
/// Pages of text, of read-only data, and of data with `.bss`.
pub(crate) const TEXT_PAGES: u64 = 3;
pub(crate) const RODATA_PAGES: u64 = 1;
pub(crate) const DATA_PAGES: u64 = 3;
/// Pages of code in all.
pub(crate) const CODE_PAGES: u64 = TEXT_PAGES + RODATA_PAGES + DATA_PAGES;
/// The size of the code in bytes.
pub(crate) const CODE_SIZE: u64 = CODE_PAGES * PAGE_SIZE;
/// Where the read-only data and the data start.
pub(crate) const RODATA: u64 = CODE + TEXT_PAGES * PAGE_SIZE;
pub(crate) const DATA: u64 = RODATA + RODATA_PAGES * PAGE_SIZE;
/// The main thread's stack: the size Eden gives a program with no NPDM
/// (`Core::Memory::DEFAULT_STACK_SIZE`).
pub(crate) const STACK_SIZE: u64 = 0x10_0000;

/// What the fixture's address space is created from, with the console's arrangement given.
pub(crate) fn memory(arrangement: MemoryArrangement) -> ProcessMemory {
    ProcessMemory {
        code_address: CODE,
        code_size: CODE_SIZE,
        arrangement,
        alias_region_extra: false,
        offsets: RegionOffsets::default(),
    }
}

/// A new address space on a 4 GiB console: the code mapped and nothing else. The log holds
/// the code's mapping.
pub(crate) fn created() -> (AddressSpace, MapperLog) {
    let (mapper, log) = RecordingMapper::new();
    let space = AddressSpace::new(&memory(MemoryArrangement::FourGb), Box::new(mapper)).unwrap();
    (space, log)
}

/// An address space as a loaded program's is when it starts: the segments reprotected and the
/// main thread's stack mapped. The log is empty. Returns the stack's top as well.
pub(crate) fn loaded() -> (AddressSpace, MapperLog, u64) {
    let (mut space, log) = created();
    for (address, pages, permission) in [
        (CODE, TEXT_PAGES, SvcPermission::ReadExecute),
        (RODATA, RODATA_PAGES, SvcPermission::Read),
        (DATA, DATA_PAGES, SvcPermission::ReadWrite),
    ] {
        space
            .set_process_memory_permission(address, pages * PAGE_SIZE, permission)
            .unwrap();
    }
    let stack_top = space.create_main_thread_stack(STACK_SIZE).unwrap();
    drop(log.take());
    (space, log, stack_top)
}
