//! Loading a program into a guest address space: where an NRO's segments, its `.bss` and the
//! counter trampolines go ([`place`]), and creating the process's memory around them with the
//! kernel ([`load`]): the code, the main thread's stack and its TLS page.
//!
//! [`load`] follows Eden's process start for a program (paths relative to the reference folder,
//! all @ cf654052):
//!
//! 1. `KProcess::Initialize` (`eden/src/core/hle/kernel/k_process.cpp`): the kernel lays out
//!    the address space around the code, reserves the code's memory and maps it with no guest
//!    access (`starquake_kernel::memory::AddressSpace::new`).
//! 2. `KProcess::LoadModule`: the image is written, then each segment is reprotected
//!    (`set_process_memory_permission`).
//! 3. `KProcess::Run`: the main thread's stack is reserved and mapped in the stack region
//!    (`create_main_thread_stack`).
//! 4. `KThread::InitializeUserThread` to `KThreadLocalPage::Initialize`
//!    (`eden/src/core/hle/kernel/k_thread_local_page.cpp`): one thread-local page, read-write,
//!    in the kernel-map region (`map_pages_in_region`).
//!
//! ```text
//! IMAGE_BASE - n pages   counter trampolines (only when a read was patched)
//! IMAGE_BASE             the image (text, rodata, data), then .bss            "main"
//! above the code, each 2 MiB aligned (starquake_kernel::memory::layout):
//!   kernel-map region    TLS page, four pages in (`TPIDRRO_EL0` points at it)
//!   stack region         the stack, four pages in
//!   alias region, heap region
//! ```
//!
//! PROVISIONAL, and what replaces each part: kernel slice 8 (process creation) owns the
//! process, its main thread and the registers it starts with; until then this module does the
//! steps above itself, with these fixed choices:
//!
//! - **The code goes at [`IMAGE_BASE`]**, Eden's address for 39-bit code with no randomisation
//!   (`KProcess::LoadFromMetadata`: `code_address = 0x8000'0000 + aslr_space_offset`), and the
//!   regions are not randomised either (kernel spec: deterministic by default).
//! - **The stack is [`STACK_SIZE`]**, what Eden gives a program with no NPDM, and there is one
//!   thread.
//! - **Every program gets the 4 GiB memory arrangement** ([`MEMORY_ARRANGEMENT`]), as every
//!   program counts time at the Switch 1 rate (`crate::clock`). Process creation picks both by
//!   what the program was built for.
//!
//! # Permissions
//!
//! [`Layout::module`] says what the module's pages are: whole pages, each with the permissions
//! guest code has there. The permissions are Eden's (`eden/src/core/hle/kernel/k_process.cpp`
//! @ cf654052):
//!
//! | What | Guest code may | Eden |
//! |---|---|---|
//! | `.text` | read, execute | `KProcess::LoadModule`, `ReprotectSegment(CodeSegment, ReadExecute)` |
//! | `.rodata` | read | `ReprotectSegment(RODataSegment, Read)` |
//! | `.data`, `.bss` | read, write | `ReprotectSegment(DataSegment, ReadWrite)`; `LoadNroImpl` (`eden/src/core/loader/nro.cpp`) adds `.bss` to the data segment |
//! | pages of the module in no segment | nothing (mapped, so the loader can write the image) | `KProcess::Initialize`, "Map the code region": `KernelRead \| NotMapped` |
//! | stack | read, write | `KProcess::Run`, `KMemoryState::Stack`, `UserReadWrite` |
//! | TLS page | read, write | `KThreadLocalPage::Initialize`, `KMemoryState::ThreadLocal`, `UserReadWrite` |
//! | counter trampolines | read, execute | starquake's own: see below |
//!
//! A segment covers whole pages: its size is rounded up to a page, as `LoadNroImpl` does
//! (`PageAlignSize`). Divergences from Eden, each pinned by a test here:
//!
//! - **The counter trampolines** are pages a console does not have. They are mapped as code,
//!   read-execute, in the pages right below the image, where no region is placed and nothing a
//!   program asks about lies: the pages after the module, where a console's loader would put
//!   a program's arguments, are free as on a console
//!   (`the_trampolines_lie_below_the_image_and_the_page_after_the_module_is_free`). They are
//!   not counted as the process's memory, so `GetInfo` reports what a console would
//!   (`the_trampolines_are_not_counted_as_the_processs_memory`). `QueryMemory` does show them.
//!   Under NCE, Eden appends its patch section to the module's image instead
//!   (`eden/src/core/loader/nro.cpp`, `LoadNroImpl`, `HAS_NCE`), where it is counted.
//! - A segment that does not start on a page boundary is [`LoadError::MisalignedSegment`]
//!   (`a_segment_that_starts_inside_a_page_is_an_error`). Eden does not refuse it.
//!   `ReprotectSegment` in `KProcess::LoadModule` discards the result of
//!   `KPageTableBase::SetProcessMemoryPermission`
//!   (`eden/src/core/hle/kernel/k_page_table_base.cpp` @ cf654052), and that function has no
//!   alignment check that returns an error: `CheckMemoryState` passes for a range inside one
//!   block of the code region, and `Operate` only soft-asserts
//!   `Common::IsAligned(GetInteger(virt_addr), PageSize)` (`ASSERT` is `AssertFailSoftImpl` in
//!   `eden/src/common/assert.cpp`, which carries on unless debug asserts are switched on)
//!   before it calls `ProtectRegion` on the unaligned range and succeeds. Guess: what guest
//!   code can then do with the pages that range touches was not traced past `ProtectRegion`;
//!   running Eden on such a file would confirm it.
//! - Two segments that share a page (after rounding) are [`LoadError::OverlappingSegments`]
//!   (`segments_that_share_a_page_are_an_error`). Eden reprotects in the order text, rodata,
//!   data. When every page of the later segment has one memory state and one permission at
//!   that moment (true of the three layouts the test uses), the later segment's permissions
//!   win for the shared pages. Otherwise `CheckMemoryState` fails
//!   (`R_UNLESS(info.m_permission == first_perm, ResultInvalidCurrentMemory)`, and the same
//!   check on `m_state`), the result is discarded, the shared page keeps the earlier segment's
//!   permissions and the rest of the later segment stays `KernelRead | NotMapped`.
//! - `.bss` is the pages `[image_end, bss_end)`: it starts where the image, rounded up to a
//!   page, ends. `LoadNroImpl` does `codeset.DataSegment().size += bss_size` instead, so Eden's
//!   read-write range is the data segment's pages plus `bss_size` bytes directly after them.
//!   The two are the same pages only when the data segment ends at the page-rounded end of the
//!   image (`bss_starts_on_the_page_after_the_image_and_is_read_write`). When it ends earlier,
//!   Eden's range stops short of the end of `.bss` and takes in the image pages after the data
//!   segment; here those pages keep what their own segment gives them, or no guest access
//!   (`bss_is_read_write_where_it_lies_even_when_the_data_segment_ends_before_the_image`).
//! - A reprotection the kernel refuses is [`LoadError::Map`], not ignored as Eden's
//!   `ReprotectSegment` ignores it. [`place`] has refused the layouts that would make it fail.
//!
//! The module's size follows Eden's NRO loader (`eden/src/core/loader/nro.cpp` @ cf654052,
//! `LoadNroImpl`): the image rounded up to a page, plus `.bss` rounded up to a page, with the
//! `MOD0` header's `.bss` size preferred over the NRO header's when `MOD0` is present. Divergence:
//! a `MOD0` whose `.bss` end precedes its start is [`LoadError::BadBss`] here; Eden subtracts
//! the two unchecked.
//!
//! Memory the kernel maps reads as zero until it is written (`crate::space`), so `.bss`, the
//! TLS page and the stack need no writes.

use starquake_kernel::memory::{
    AddressSpace as KernelSpace, CreateError, Mapper, MemoryArrangement, MemoryPermission,
    MemoryState, ProcessMemory, RegionOffsets, SvcPermission,
};
use starquake_kernel::result::SvcError;
use starquake_loader::nro::{Nro, NroError};
use starquake_loader::patch::{CounterRescale, PatchConfig, PatchError, patch_text};
use starquake_memory::{Fault, GuestMemory, PAGE_SIZE, Perms};

use crate::crash::Module;
use crate::report::Mapping;

/// The guest's page size (`docs/specs/execution-model.md`: a 4 KiB granule).
pub(crate) const PAGE: u64 = PAGE_SIZE;
/// Guest virtual address the program's image goes at: Eden's address for 39-bit code without
/// randomisation (see the module docs).
pub(crate) const IMAGE_BASE: u64 = 0x8000_0000;
/// Size of the main thread's stack: what Eden gives a program with no NPDM
/// (`Core::Memory::DEFAULT_STACK_SIZE` in `eden/src/core/memory.h` @ cf654052, used by
/// `eden/src/core/loader/nro.cpp`). PROVISIONAL: Horizon takes it from the NPDM.
pub(crate) const STACK_SIZE: u64 = 0x10_0000;
/// Size of the TLS block: one page, as Horizon allocates thread-local regions out of pages.
pub(crate) const TLS_SIZE: u64 = PAGE;
/// The console's memory arrangement every program runs with. PROVISIONAL (see the module
/// docs).
pub(crate) const MEMORY_ARRANGEMENT: MemoryArrangement = MemoryArrangement::FourGb;
/// The name the program's module goes by in the memory map and crash reports.
pub(crate) const MODULE_NAME: &str = "main";

/// Why a program could not be loaded into guest memory.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum LoadError {
    /// The file is not a well-formed NRO.
    #[error("the program is not a loadable NRO: {0}")]
    Nro(#[from] NroError),
    /// The `MOD0` header's `.bss` end precedes its start.
    #[error("the NRO's MOD0 header describes a .bss that ends before it starts")]
    BadBss,
    /// The counter patcher rejected the text segment.
    #[error("the program's text could not be patched: {0}")]
    Patch(#[from] PatchError),
    /// A segment does not start on a page boundary.
    #[error("the NRO's segment at offset {offset:#x} does not start on a page boundary")]
    MisalignedSegment {
        /// The segment's offset in the image.
        offset: u32,
    },
    /// Two segments, or a segment and `.bss`, cover the same page.
    #[error("two of the NRO's segments share the page at image offset {offset:#x}")]
    OverlappingSegments {
        /// Offset in the image of the first shared page.
        offset: u64,
    },
    /// A size in the NRO, or an address derived from one, does not fit 64 bits.
    #[error("the program's sizes overflow the address space")]
    Overflow,
    /// The kernel could not create the program's address space: its code does not fit the
    /// code region or the memory an application may use.
    #[error("the program's address space could not be created: {0}")]
    AddressSpace(#[from] CreateError),
    /// The kernel refused one of the mappings a program starts with.
    #[error("the program's {what} could not be mapped: {error}")]
    Map {
        /// What was being mapped or reprotected.
        what: &'static str,
        /// Why the kernel refused.
        error: SvcError,
    },
    /// The program could not be copied into guest memory.
    #[error("the program could not be written into guest memory: {0}")]
    Write(Fault),
}

/// A parsed program: the NRO's header fields and its image bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Program {
    nro: Nro,
    /// The image: the first `nro.size` bytes of the file.
    image: Vec<u8>,
}

impl Program {
    /// Parses `bytes` as an NRO.
    pub(crate) fn parse(bytes: &[u8]) -> Result<Self, LoadError> {
        let nro = Nro::parse(bytes)?;
        // `Nro::parse` has checked that the image fits the file.
        let image = usize::try_from(nro.size)
            .ok()
            .and_then(|size| bytes.get(..size))
            .ok_or(LoadError::Overflow)?
            .to_vec();
        Ok(Self { nro, image })
    }

    /// Size of `.bss` in bytes, before rounding (see the module docs).
    fn bss_size(&self) -> Result<u32, LoadError> {
        match &self.nro.mod0 {
            Some(mod0) => mod0.bss_size().ok_or(LoadError::BadBss),
            None => Ok(self.nro.bss_size),
        }
    }
}

/// A run of whole pages, and what guest code may do there.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Region {
    /// Guest virtual address of the first page.
    pub(crate) start: u64,
    /// Size in bytes: a non-zero multiple of [`PAGE`].
    pub(crate) size: u64,
    /// What guest code may do with the pages.
    pub(crate) perms: Perms,
}

/// Where a program's module and its trampolines go, and the bytes to put there: what [`place`]
/// works out from the file alone.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Layout {
    /// Address the guest starts executing at: the first byte of the image.
    pub(crate) entry: u64,
    /// `(address, bytes)` to copy into guest memory, in address order.
    pub(crate) writes: Vec<(u64, Vec<u8>)>,
    /// Everything that was laid out, in address order.
    pub(crate) mappings: Vec<Mapping>,
    /// The module's pages, in address order, none overlapping: every page from the module's
    /// base to its end (see "Permissions" in the module docs).
    pub(crate) module_regions: Vec<Region>,
    /// The counter trampolines' pages, right below the module, if a read was patched.
    pub(crate) trampolines: Option<Region>,
    /// The program's module.
    pub(crate) module: Module,
    /// Addresses of the counter reads that were patched, in address order.
    pub(crate) counter_reads: Vec<u64>,
}

/// A loaded program, as the core loop and the run's report need it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Placement {
    /// Address the guest starts executing at: the first byte of the image.
    pub(crate) entry: u64,
    /// Initial stack pointer: the end of the stack.
    pub(crate) stack_top: u64,
    /// Address of the TLS block, for `TPIDRRO_EL0`.
    pub(crate) tls: u64,
    /// Everything that was mapped, in address order.
    pub(crate) mappings: Vec<Mapping>,
    /// The program's module.
    pub(crate) module: Module,
    /// Addresses of the counter reads that were patched, in address order.
    pub(crate) counter_reads: Vec<u64>,
}

/// What [`load`] hands back: where things are, and the kernel's address space that holds them.
#[derive(Debug)]
pub(crate) struct Loaded {
    /// Where the program, its stack and its TLS block are.
    pub(crate) placed: Placement,
    /// The process's address space, for the kernel's tables.
    pub(crate) address_space: KernelSpace,
}

/// `value` rounded up to a page.
fn page_up(value: u64) -> Result<u64, LoadError> {
    value
        .checked_next_multiple_of(PAGE)
        .ok_or(LoadError::Overflow)
}

fn add(a: u64, b: u64) -> Result<u64, LoadError> {
    a.checked_add(b).ok_or(LoadError::Overflow)
}

/// The module's pages: its segments and `.bss` with their permissions, and every other page of
/// `base..bss_end` with none (see "Permissions" in the module docs).
fn module_regions(
    program: &Program,
    base: u64,
    image_end: u64,
    bss_end: u64,
) -> Result<Vec<Region>, LoadError> {
    let segment_perms = [Perms::ReadExecute, Perms::Read, Perms::ReadWrite];
    let mut parts = Vec::new();
    for (segment, perms) in program.nro.segments().into_iter().zip(segment_perms) {
        if segment.size == 0 {
            continue;
        }
        if !u64::from(segment.offset).is_multiple_of(PAGE) {
            return Err(LoadError::MisalignedSegment {
                offset: segment.offset,
            });
        }
        let start = add(base, u64::from(segment.offset))?;
        let end = add(start, page_up(u64::from(segment.size))?)?;
        parts.push((start, end, perms));
    }
    if image_end < bss_end {
        parts.push((image_end, bss_end, Perms::ReadWrite));
    }
    parts.sort_by_key(|&(start, _, _)| start);

    let mut regions = Vec::new();
    let mut push = |start: u64, end: u64, perms| {
        if start < end {
            regions.push(Region {
                start,
                size: end - start,
                perms,
            });
        }
    };
    // Walk the parts in address order, giving the pages between them no guest access.
    let mut at = base;
    for (start, end, perms) in parts {
        if start < at {
            return Err(LoadError::OverlappingSegments {
                offset: start - base,
            });
        }
        push(at, start, Perms::None);
        push(start, end, perms);
        at = end;
    }
    push(at, bss_end, Perms::None);
    Ok(regions)
}

/// The program's image with its counter reads rewritten for `counter` to branch to
/// trampolines at `trampolines_at`, and what the patcher did.
fn patch(
    program: &Program,
    base: u64,
    trampolines_at: u64,
    counter: Option<CounterRescale>,
) -> Result<(Vec<u8>, starquake_loader::patch::Patched), LoadError> {
    let nro = &program.nro;
    let mut image = program.image.clone();
    let text_address = add(base, u64::from(nro.text.offset))?;
    let text = usize::try_from(nro.text.offset)
        .ok()
        .zip(usize::try_from(nro.text.size).ok())
        .and_then(|(start, len)| Some(start..start.checked_add(len)?))
        .and_then(|range| image.get_mut(range))
        .ok_or(LoadError::Overflow)?;
    let patched = patch_text(text, text_address, trampolines_at, &PatchConfig { counter })?;
    Ok((image, patched))
}

/// Lays `program`'s module out at `base` (see the module docs), rewriting its counter reads
/// for `counter` if one is given, with the trampolines in the pages right below `base`.
pub(crate) fn place(
    program: &Program,
    base: u64,
    counter: Option<CounterRescale>,
) -> Result<Layout, LoadError> {
    let nro = &program.nro;
    let image_end = add(base, page_up(u64::from(nro.size))?)?;
    let bss_end = add(image_end, page_up(u64::from(program.bss_size()?))?)?;

    // The trampolines end where the image starts, so where they start depends on how many
    // there are: patch once to count them (how many there are does not depend on where they
    // go; the page after the module is as good a place as any), then again with their address.
    let (mut image, mut patched) = patch(program, base, bss_end, counter)?;
    let trampolines_len = u64::try_from(patched.trampolines.len())
        .map_err(|_| LoadError::Overflow)
        .and_then(page_up)?;
    let trampolines_at = base
        .checked_sub(trampolines_len)
        .ok_or(LoadError::Overflow)?;
    if trampolines_len != 0 {
        let first_len = patched.trampolines.len();
        (image, patched) = patch(program, base, trampolines_at, counter)?;
        if patched.trampolines.len() != first_len {
            return Err(LoadError::Overflow);
        }
    }

    let mut mappings = Vec::new();
    let mut map = |name: &str, start: u64, end: u64| {
        if start < end {
            mappings.push(Mapping {
                name: name.to_owned(),
                start,
                end,
            });
        }
    };
    map("trampolines", trampolines_at, base);
    for (name, segment) in [".text", ".rodata", ".data"]
        .into_iter()
        .zip(nro.segments())
    {
        let start = add(base, u64::from(segment.offset))?;
        let end = add(start, u64::from(segment.size))?;
        map(&format!("{MODULE_NAME}{name}"), start, end);
    }
    map(&format!("{MODULE_NAME}.bss"), image_end, bss_end);
    mappings.sort_by_key(|mapping| mapping.start);

    let mut writes = Vec::new();
    if !patched.trampolines.is_empty() {
        writes.push((trampolines_at, patched.trampolines));
    }
    writes.push((base, image));
    Ok(Layout {
        entry: base,
        writes,
        mappings,
        module_regions: module_regions(program, base, image_end, bss_end)?,
        trampolines: (trampolines_len != 0).then_some(Region {
            start: trampolines_at,
            size: trampolines_len,
            perms: Perms::ReadExecute,
        }),
        module: Module {
            name: MODULE_NAME.to_owned(),
            base,
            size: bss_end - base,
            build_id: nro.build_id.to_vec(),
        },
        counter_reads: patched.sites.iter().map(|site| site.address).collect(),
    })
}

/// What the kernel is asked for a module region's permissions: `None` for pages with no guest
/// access, which stay as the kernel mapped them.
fn reprotection(perms: Perms) -> Option<SvcPermission> {
    match perms {
        Perms::ReadExecute => Some(SvcPermission::ReadExecute),
        Perms::Read => Some(SvcPermission::Read),
        Perms::ReadWrite => Some(SvcPermission::ReadWrite),
        // `module_regions` gives a region one of the three above or none.
        Perms::None | Perms::Execute | Perms::ReadWriteExecute => None,
    }
}

/// Loads `program` at [`IMAGE_BASE`]: creates its address space with the kernel, with `mapper`
/// making the kernel's mappings real, copies the image into `memory` (that same address space,
/// as host code sees it) and gives the process a main thread's stack and TLS page. Counter
/// reads are rewritten for `counter` if one is given. See the module docs for the steps.
pub(crate) fn load(
    program: &Program,
    counter: Option<CounterRescale>,
    mapper: Box<dyn Mapper>,
    memory: &dyn GuestMemory,
) -> Result<Loaded, LoadError> {
    let layout = place(program, IMAGE_BASE, counter)?;
    let refused = |what: &'static str| move |error: SvcError| LoadError::Map { what, error };
    let mut space = KernelSpace::new(
        &ProcessMemory {
            code_address: layout.module.base,
            code_size: layout.module.size,
            arrangement: MEMORY_ARRANGEMENT,
            alias_region_extra: false,
            offsets: RegionOffsets::default(),
        },
        mapper,
    )?;
    if let Some(trampolines) = layout.trampolines {
        space
            .map_pages(
                trampolines.start,
                trampolines.size / PAGE,
                MemoryState::Code,
                MemoryPermission::USER_READ_EXECUTE,
            )
            .map_err(refused("counter trampolines"))?;
    }
    for (address, bytes) in &layout.writes {
        memory.write(*address, bytes).map_err(LoadError::Write)?;
    }
    for region in &layout.module_regions {
        if let Some(permission) = reprotection(region.perms) {
            space
                .set_process_memory_permission(region.start, region.size, permission)
                .map_err(refused("segments"))?;
        }
    }
    let stack_top = space
        .create_main_thread_stack(STACK_SIZE)
        .map_err(refused("stack"))?;
    let tls = space
        .map_pages_in_region(
            TLS_SIZE / PAGE,
            MemoryState::ThreadLocal,
            MemoryPermission::USER_READ_WRITE,
        )
        .map_err(refused("TLS page"))?;

    let mut mappings = layout.mappings;
    for (name, start, size) in [
        ("tls", tls, TLS_SIZE),
        ("stack", stack_top - STACK_SIZE, STACK_SIZE),
    ] {
        mappings.push(Mapping {
            name: name.to_owned(),
            start,
            end: start + size,
        });
    }
    mappings.sort_by_key(|mapping| mapping.start);
    Ok(Loaded {
        placed: Placement {
            entry: layout.entry,
            stack_top,
            tls,
            mappings,
            module: layout.module,
            counter_reads: layout.counter_reads,
        },
        address_space: space,
    })
}

#[cfg(test)]
mod tests {
    use starquake_kernel::memory::GUARD_PAGES;
    use starquake_kernel::memory::testing::{MapperCall, RecordingMapper};
    use starquake_kernel::result::KernelError;
    use starquake_memory::FlatMemory;
    use starquake_testkit::nro::{BUILD_ID_SIZE, CODE_OFFSET, NroBuilder};
    use starquake_testkit::{a64, programs};

    use super::*;

    /// Where a real run puts the program.
    const BASE: u64 = IMAGE_BASE;
    /// The counter rates of a real run: Apple Silicon's and the kernel's.
    const RESCALE: CounterRescale = CounterRescale {
        host_hz: 24_000_000,
        guest_hz: crate::clock::PROCESS_COUNTER_RATE.hz(),
    };
    /// How much flat guest memory the loading tests have around [`IMAGE_BASE`]: a megabyte
    /// below it (for trampolines) and one from it (for the image).
    const FLAT_BELOW: u64 = 1 << 20;
    const FLAT_SIZE: usize = 2 << 20;

    fn mapping(name: &str, start: u64, end: u64) -> Mapping {
        Mapping {
            name: name.to_owned(),
            start,
            end,
        }
    }

    fn region(start: u64, pages: u64, perms: Perms) -> Region {
        Region {
            start,
            size: pages * PAGE,
            perms,
        }
    }

    /// Overwrites segment `index`'s offset and size in an NRO's header.
    fn set_segment(file: &mut [u8], index: usize, offset: u32, size: u32) {
        use starquake_testkit::nro::{SEGMENT_HEADER_SIZE, SEGMENT_SIZE_OFFSET, SEGMENTS_OFFSET};
        let at = SEGMENTS_OFFSET + index * SEGMENT_HEADER_SIZE;
        file[at..at + 4].copy_from_slice(&offset.to_le_bytes());
        let at = at + SEGMENT_SIZE_OFFSET;
        file[at..at + 4].copy_from_slice(&size.to_le_bytes());
    }

    fn word_at(bytes: &[u8], offset: u64) -> u32 {
        let at = usize::try_from(offset).unwrap();
        u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap())
    }

    impl Layout {
        /// Every run of pages the layout maps, in address order.
        fn regions(&self) -> Vec<Region> {
            self.trampolines
                .iter()
                .chain(&self.module_regions)
                .copied()
                .collect()
        }
    }

    /// Flat guest memory around [`IMAGE_BASE`].
    fn flat() -> FlatMemory {
        FlatMemory::new(BASE - FLAT_BELOW, FLAT_SIZE)
    }

    // ----- place: the module and its trampolines ------------------------------------------------

    #[test]
    fn a_program_is_laid_out_from_its_base() {
        let file = programs::hello().unwrap();
        let placed = place(&Program::parse(&file).unwrap(), BASE, Some(RESCALE)).unwrap();
        // hello is one page of text and one of rodata, with no data, .bss or counter reads.
        assert_eq!(file.len(), 2 * 0x1000);
        assert_eq!(
            placed.mappings,
            [
                mapping("main.text", BASE, BASE + PAGE),
                mapping("main.rodata", BASE + PAGE, BASE + 2 * PAGE),
            ]
        );
        // Text may be read and executed, rodata read.
        assert_eq!(
            placed.regions(),
            [
                region(BASE, 1, Perms::ReadExecute),
                region(BASE + PAGE, 1, Perms::Read),
            ]
        );
        assert_eq!(placed.entry, BASE);
        // Nothing was patched: the image is written as it is in the file, and nothing else.
        assert_eq!(placed.writes, [(BASE, file)]);
        assert_eq!(placed.counter_reads, Vec::<u64>::new());
        assert_eq!(placed.trampolines, None);
        assert_eq!(
            placed.module,
            Module {
                name: "main".into(),
                base: BASE,
                size: 2 * PAGE,
                build_id: vec![0; BUILD_ID_SIZE],
            }
        );
    }

    #[test]
    fn counter_reads_become_branches_to_trampolines_placed_below_the_module() {
        let counter = programs::counter(RESCALE.guest_hz).unwrap();
        let program = Program::parse(&counter.nro).unwrap();
        let placed = place(&program, BASE, Some(RESCALE)).unwrap();

        assert_eq!(
            placed.counter_reads,
            counter.read_offsets.map(|offset| BASE + offset)
        );
        let trampolines = placed
            .mappings
            .iter()
            .find(|mapping| mapping.name == "trampolines")
            .expect("a trampoline mapping");
        // They fit one page, which ends exactly where the image starts.
        assert_eq!((trampolines.start, trampolines.end), (BASE - PAGE, BASE));
        // The trampolines are code: readable and executable, not writable.
        assert_eq!(
            placed.trampolines,
            Some(region(BASE - PAGE, 1, Perms::ReadExecute))
        );
        assert_eq!(
            placed.regions()[..2],
            [
                region(BASE - PAGE, 1, Perms::ReadExecute),
                region(BASE, 1, Perms::ReadExecute),
            ]
        );
        assert_eq!(placed.writes.len(), 2, "trampolines and an image");
        let (trampolines_at, code) = &placed.writes[0];
        let (image_at, image) = &placed.writes[1];
        assert_eq!((*image_at, *trampolines_at), (BASE, trampolines.start));
        assert!(!code.is_empty() && code.len() <= usize::try_from(PAGE).unwrap());
        // Each read is now `b <trampoline>` (imm26 in words, sign-extended), and each lands in
        // the trampoline area at a different place.
        let mut targets = Vec::new();
        for offset in counter.read_offsets {
            let word = word_at(image, offset);
            assert_eq!(word & 0xFC00_0000, 0x1400_0000, "not a b: {word:#x}");
            let distance = (i64::from(word & 0x03FF_FFFF) << 38) >> 36;
            assert!(distance < 0, "the trampolines are below the text");
            let target = (BASE + offset).checked_add_signed(distance).unwrap();
            assert!(trampolines.contains(target), "{target:#x}");
            targets.push(target);
        }
        targets.dedup();
        assert_eq!(targets.len(), counter.read_offsets.len());
        // Nothing else in the image changed.
        let mut restored = image.clone();
        for offset in counter.read_offsets {
            let at = usize::try_from(offset).unwrap();
            restored[at..at + 4].copy_from_slice(&counter.nro[at..at + 4]);
        }
        assert_eq!(restored, counter.nro);
    }

    #[test]
    fn trampolines_need_room_below_the_base() {
        let counter = programs::counter(RESCALE.guest_hz).unwrap();
        let program = Program::parse(&counter.nro).unwrap();
        // One page below the base is exactly enough; none is not.
        assert!(place(&program, PAGE, Some(RESCALE)).is_ok());
        assert_eq!(place(&program, 0, Some(RESCALE)), Err(LoadError::Overflow));
        // Without trampolines a base of zero is fine.
        assert!(place(&program, 0, None).is_ok());
    }

    #[test]
    fn without_a_counter_rescale_nothing_is_patched() {
        let counter = programs::counter(RESCALE.guest_hz).unwrap();
        let placed = place(&Program::parse(&counter.nro).unwrap(), BASE, None).unwrap();
        assert_eq!(placed.writes, [(BASE, counter.nro)]);
        assert_eq!(placed.counter_reads, Vec::<u64>::new());
        assert!(placed.mappings.iter().all(|m| m.name != "trampolines"));
        assert_eq!(placed.trampolines, None);
    }

    #[test]
    fn bss_follows_the_image_rounded_up_to_a_page() {
        let file = NroBuilder::new(&[a64::nop()])
            .data(b"data")
            .bss_size(0x1234)
            .build()
            .unwrap();
        let placed = place(&Program::parse(&file).unwrap(), BASE, None).unwrap();
        // One page of text, none of rodata, one of data, then two pages of .bss.
        assert_eq!(
            placed.mappings,
            [
                mapping("main.text", BASE, BASE + PAGE),
                mapping("main.data", BASE + PAGE, BASE + 2 * PAGE),
                mapping("main.bss", BASE + 2 * PAGE, BASE + 4 * PAGE),
            ]
        );
        assert_eq!(placed.module.size, 4 * PAGE);
        assert!(CODE_OFFSET < usize::try_from(PAGE).unwrap());
        // Data and .bss are both read-write, each its own run of pages.
        assert_eq!(
            placed.regions(),
            [
                region(BASE, 1, Perms::ReadExecute),
                region(BASE + PAGE, 1, Perms::ReadWrite),
                region(BASE + 2 * PAGE, 2, Perms::ReadWrite),
            ]
        );
    }

    /// Text, rodata and data of one page each, at image offsets 0, one page and two pages.
    fn three_segments() -> Vec<u8> {
        let file = NroBuilder::new(&[a64::nop()])
            .rodata(b"rodata")
            .data(b"data")
            .build()
            .unwrap();
        assert_eq!(file.len() as u64, 3 * PAGE);
        file
    }

    #[test]
    fn regions_cover_every_write_and_mapping_and_never_overlap() {
        let counter = programs::counter(RESCALE.guest_hz).unwrap();
        for file in [three_segments(), programs::hello().unwrap(), counter.nro] {
            let program = Program::parse(&file).unwrap();
            let placed = place(&program, BASE, Some(RESCALE)).unwrap();
            let regions = placed.regions();
            let inside = |start: u64, end: u64| {
                regions
                    .iter()
                    .any(|r| r.start <= start && end <= r.start + r.size)
            };
            for pair in regions.windows(2) {
                assert!(pair[0].start + pair[0].size <= pair[1].start, "{pair:?}");
            }
            for r in &regions {
                assert!(
                    r.size > 0 && r.start % PAGE == 0 && r.size % PAGE == 0,
                    "{r:?}"
                );
            }
            for mapping in &placed.mappings {
                assert!(inside(mapping.start, mapping.end), "{mapping:?}");
            }
            // A write may span several regions, so check it page by page.
            for (address, bytes) in &placed.writes {
                let end = address + bytes.len() as u64;
                let mut page = *address & !(PAGE - 1);
                while page < end {
                    assert!(inside(page, page + PAGE), "{page:#x}");
                    page += PAGE;
                }
            }
            // The module's regions are exactly the module, with no hole.
            let module_end = placed.module.base + placed.module.size;
            let mut at = placed.module.base;
            for r in &placed.module_regions {
                assert_eq!(r.start, at);
                at += r.size;
            }
            assert_eq!(at, module_end);
        }
    }

    #[test]
    fn pages_of_the_module_in_no_segment_are_mapped_without_guest_access() {
        // Drop rodata from the header: its page stays part of the image, between text and data.
        let mut file = three_segments();
        set_segment(&mut file, 1, 0, 0);
        let placed = place(&Program::parse(&file).unwrap(), BASE, None).unwrap();
        assert_eq!(
            placed.regions(),
            [
                region(BASE, 1, Perms::ReadExecute),
                region(BASE + PAGE, 1, Perms::None),
                region(BASE + 2 * PAGE, 1, Perms::ReadWrite),
            ]
        );
        // The image is still written whole, across the page no guest code can touch.
        assert_eq!(placed.writes, [(BASE, file)]);
    }

    #[test]
    fn pages_at_the_end_of_the_image_in_no_segment_are_mapped_without_guest_access() {
        // Drop data from the header: its page is still the last page of the image, and there is
        // no .bss after it.
        let mut file = three_segments();
        set_segment(&mut file, 2, 0, 0);
        let placed = place(&Program::parse(&file).unwrap(), BASE, None).unwrap();
        assert_eq!(
            placed.regions(),
            [
                region(BASE, 1, Perms::ReadExecute),
                region(BASE + PAGE, 1, Perms::Read),
                region(BASE + 2 * PAGE, 1, Perms::None),
            ]
        );
        // The module ends with that page, and the image is written whole.
        assert_eq!(placed.module.size, 3 * PAGE);
        assert_eq!(placed.writes, [(BASE, file)]);
    }

    #[test]
    fn a_segment_is_rounded_up_to_whole_pages() {
        // A rodata of one byte still gets its whole page, read-only.
        let mut file = three_segments();
        set_segment(&mut file, 1, u32::try_from(PAGE).unwrap(), 1);
        let placed = place(&Program::parse(&file).unwrap(), BASE, None).unwrap();
        assert_eq!(placed.regions()[1], region(BASE + PAGE, 1, Perms::Read));
        // The mapping in the report keeps the segment's own size.
        assert_eq!(
            placed.mappings[1],
            mapping("main.rodata", BASE + PAGE, BASE + PAGE + 1)
        );
    }

    #[test]
    fn a_segment_that_starts_inside_a_page_is_an_error() {
        // Divergence from Eden, which has no check that refuses this: it discards the result of
        // `SetProcessMemoryPermission`, whose only alignment check is a soft assert before it
        // reprotects the unaligned range (see "Permissions" in the module docs).
        let page = u32::try_from(PAGE).unwrap();
        for offset in [page + 1, 2 * page - 1] {
            let mut file = three_segments();
            set_segment(&mut file, 1, offset, 1);
            assert_eq!(
                place(&Program::parse(&file).unwrap(), BASE, None),
                Err(LoadError::MisalignedSegment { offset }),
                "{offset:#x}"
            );
        }
        assert_eq!(
            LoadError::MisalignedSegment { offset: 0x1001 }.to_string(),
            "the NRO's segment at offset 0x1001 does not start on a page boundary"
        );
    }

    #[test]
    fn segments_that_share_a_page_are_an_error() {
        // Divergence from Eden. In each of these three layouts every page of the later segment
        // has one state and one permission when Eden reprotects it, so there the later segment's
        // permissions win for the shared page (the module docs say what Eden does otherwise).
        let page = u32::try_from(PAGE).unwrap();
        // Text one instruction into rodata's page (text is whole instructions, or the patcher
        // refuses it first); rodata on top of text; rodata on top of data.
        for (index, offset, size, shared) in [
            (0, 0, page + 4, PAGE),
            (1, 0, 1, 0),
            (1, 2 * page, 1, 2 * PAGE),
        ] {
            let mut file = three_segments();
            set_segment(&mut file, index, offset, size);
            assert_eq!(
                place(&Program::parse(&file).unwrap(), BASE, None),
                Err(LoadError::OverlappingSegments { offset: shared }),
                "segment {index} at {offset:#x}, {size:#x} bytes"
            );
        }
        // Text ending exactly where rodata starts shares nothing.
        let mut file = three_segments();
        set_segment(&mut file, 0, 0, page);
        assert!(place(&Program::parse(&file).unwrap(), BASE, None).is_ok());
        assert_eq!(
            LoadError::OverlappingSegments { offset: 0x2000 }.to_string(),
            "two of the NRO's segments share the page at image offset 0x2000"
        );
    }

    #[test]
    fn bss_starts_on_the_page_after_the_image_and_is_read_write() {
        // The loader only accepts segments that lie inside the image, and .bss starts at the
        // image's size rounded up to a page, so .bss never shares a page with a segment.
        let file = NroBuilder::new(&[a64::nop()])
            .data(b"data")
            .bss_size(1)
            .build()
            .unwrap();
        let program = Program::parse(&file).unwrap();
        let placed = place(&program, BASE, None).unwrap();
        let image_end = BASE + (file.len() as u64).next_multiple_of(PAGE);
        let regions = placed.regions();
        let bss = regions
            .iter()
            .find(|r| r.start == image_end)
            .expect("a .bss region");
        assert_eq!(*bss, region(image_end, 1, Perms::ReadWrite));
        for segment in program.nro.segments() {
            let end = BASE + u64::from(segment.offset) + u64::from(segment.size);
            assert!(end.next_multiple_of(PAGE) <= image_end);
        }
    }

    #[test]
    fn bss_is_read_write_where_it_lies_even_when_the_data_segment_ends_before_the_image() {
        // Divergence from Eden, which grows the data segment by the size of .bss. Data is moved
        // to the image's second page and rodata dropped, so the third page is in no segment and
        // data ends a page before the image does.
        let page = u32::try_from(PAGE).unwrap();
        let mut file = NroBuilder::new(&[a64::nop()])
            .rodata(b"rodata")
            .data(b"data")
            .bss_size(1)
            .build()
            .unwrap();
        assert_eq!(file.len() as u64, 3 * PAGE);
        set_segment(&mut file, 1, 0, 0);
        set_segment(&mut file, 2, page, page);
        let placed = place(&Program::parse(&file).unwrap(), BASE, None).unwrap();
        // Eden's read-write range would be the second and third pages, leaving .bss (the
        // fourth) without guest access. Here the third page has none and .bss is read-write.
        assert_eq!(
            placed.regions(),
            [
                region(BASE, 1, Perms::ReadExecute),
                region(BASE + PAGE, 1, Perms::ReadWrite),
                region(BASE + 2 * PAGE, 1, Perms::None),
                region(BASE + 3 * PAGE, 1, Perms::ReadWrite),
            ]
        );
        assert!(
            placed
                .mappings
                .contains(&mapping("main.bss", BASE + 3 * PAGE, BASE + 4 * PAGE))
        );
    }

    #[test]
    fn a_file_that_is_not_an_nro_is_an_error() {
        assert_eq!(
            Program::parse(b"not an nro"),
            Err(LoadError::Nro(NroError::Truncated {
                len: 10,
                needed: starquake_loader::nro::HEADER_SIZE
            }))
        );
        let mut file = programs::hello().unwrap();
        file[starquake_testkit::nro::MAGIC_OFFSET] ^= 0xff;
        assert_eq!(
            Program::parse(&file),
            Err(LoadError::Nro(NroError::BadMagic))
        );
    }

    #[test]
    fn a_mod0_whose_bss_ends_before_it_starts_is_an_error() {
        // Divergence from Eden, which subtracts the two offsets unchecked.
        use starquake_testkit::nro::{MOD0_FIELD_SIZE, MOD0_OFFSET};
        let mut file = programs::hello().unwrap();
        // bss_start_offset is MOD0's second field after the magic; make it larger than the end.
        let start_at = MOD0_OFFSET + 2 * MOD0_FIELD_SIZE;
        let end = word_at(&file, u64::try_from(start_at + MOD0_FIELD_SIZE).unwrap());
        file[start_at..start_at + 4].copy_from_slice(&(end + 1).to_le_bytes());
        let program = Program::parse(&file).unwrap();
        assert_eq!(place(&program, BASE, None), Err(LoadError::BadBss));
    }

    // ----- load: the process's memory around the module -------------------------------------------

    /// Loads `file` into flat memory with a recording mapper behind the kernel's address
    /// space.
    fn loaded(
        file: &[u8],
        counter: Option<CounterRescale>,
    ) -> (Loaded, FlatMemory, Vec<MapperCall>) {
        let (mapper, log) = RecordingMapper::new();
        let memory = flat();
        let loaded = load(
            &Program::parse(file).unwrap(),
            counter,
            Box::new(mapper),
            &memory,
        )
        .unwrap();
        (loaded, memory, log.take())
    }

    /// What the kernel's address space holds at `address`: where the block starts, its size,
    /// its state and what the guest may do with it.
    fn block(space: &KernelSpace, address: u64) -> (u64, u64, MemoryState, MemoryPermission) {
        let block = space.query_info(address).unwrap();
        (
            block.address(),
            block.size(),
            block.state(),
            block.permission(),
        )
    }

    #[test]
    fn a_loaded_program_has_its_stack_and_tls_page_where_the_kernel_places_them() {
        let file = three_segments();
        let (
            Loaded {
                placed,
                address_space,
            },
            memory,
            _,
        ) = loaded(&file, None);
        let regions = *address_space.regions();
        // Four free pages, then the mapping (`KPageTableBase::GetNumGuardPages`).
        let guard = GUARD_PAGES * PAGE;
        let stack_bottom = regions.stack.start + guard;
        let tls = regions.kernel_map.start + guard;
        assert_eq!(placed.entry, IMAGE_BASE);
        assert_eq!(placed.stack_top, stack_bottom + STACK_SIZE);
        assert_eq!(placed.stack_top % 16, 0, "SP must be 16-byte aligned");
        assert_eq!(placed.tls, tls);
        assert_eq!(
            placed.mappings,
            [
                mapping("main.text", BASE, BASE + PAGE),
                mapping("main.rodata", BASE + PAGE, BASE + 2 * PAGE),
                mapping("main.data", BASE + 2 * PAGE, BASE + 3 * PAGE),
                mapping("tls", tls, tls + TLS_SIZE),
                mapping("stack", stack_bottom, placed.stack_top),
            ]
        );
        assert_eq!(placed.module.size, 3 * PAGE);

        // The kernel's blocks: each segment with Eden's state and permission, the stack and
        // the thread-local page.
        let at = |address| block(&address_space, address);
        assert_eq!(
            at(BASE),
            (
                BASE,
                PAGE,
                MemoryState::Code,
                MemoryPermission::USER_READ_EXECUTE
            )
        );
        assert_eq!(
            at(BASE + PAGE),
            (
                BASE + PAGE,
                PAGE,
                MemoryState::Code,
                MemoryPermission::USER_READ
            )
        );
        assert_eq!(
            at(BASE + 2 * PAGE),
            (
                BASE + 2 * PAGE,
                PAGE,
                MemoryState::CodeData,
                MemoryPermission::USER_READ_WRITE
            )
        );
        assert_eq!(
            at(placed.stack_top - 1),
            (
                stack_bottom,
                STACK_SIZE,
                MemoryState::Stack,
                MemoryPermission::USER_READ_WRITE
            )
        );
        assert_eq!(
            at(tls),
            (
                tls,
                TLS_SIZE,
                MemoryState::ThreadLocal,
                MemoryPermission::USER_READ_WRITE
            )
        );
        // The code and the stack are the process's memory; the thread-local page is not
        // (`KProcess::GetUsedUserPhysicalMemorySize`).
        assert_eq!(
            address_space.used_user_physical_memory_size(),
            3 * PAGE + STACK_SIZE
        );
        assert_eq!(
            address_space.limit().limit(),
            MEMORY_ARRANGEMENT.application_pool_size()
        );

        // The image is in guest memory.
        let mut read = vec![0; file.len()];
        memory.read(BASE, &mut read).unwrap();
        assert_eq!(read, file);
    }

    #[test]
    fn loading_asks_the_translation_for_each_mapping_in_edens_order() {
        let (Loaded { placed, .. }, _, calls) = loaded(&three_segments(), None);
        let stack_bottom = placed.stack_top - STACK_SIZE;
        let tls = placed.tls;
        // The code with no guest access, the three segments reprotected, the stack, the
        // thread-local page.
        let map = |address, size, perms| MapperCall::Map {
            address,
            size,
            perms,
        };
        let protect = |address, perms| MapperCall::Protect {
            address,
            size: PAGE,
            perms,
        };
        assert_eq!(
            calls,
            [
                map(BASE, 3 * PAGE, Perms::None),
                protect(BASE, Perms::ReadExecute),
                protect(BASE + PAGE, Perms::Read),
                protect(BASE + 2 * PAGE, Perms::ReadWrite),
                map(stack_bottom, STACK_SIZE, Perms::ReadWrite),
                map(tls, TLS_SIZE, Perms::ReadWrite),
            ]
        );
    }

    #[test]
    fn pages_in_no_segment_stay_as_the_kernel_mapped_the_code() {
        let mut file = three_segments();
        set_segment(&mut file, 1, 0, 0);
        let (Loaded { address_space, .. }, _, calls) = loaded(&file, None);
        // `KProcess::Initialize`, "Map the code region": `KernelRead | NotMapped`.
        assert_eq!(
            block(&address_space, BASE + PAGE),
            (
                BASE + PAGE,
                PAGE,
                MemoryState::Code,
                MemoryPermission::KERNEL_READ | MemoryPermission::NOT_MAPPED
            )
        );
        // Two reprotections, not three.
        let protects = calls
            .iter()
            .filter(|call| matches!(call, MapperCall::Protect { .. }))
            .count();
        assert_eq!(protects, 2);
    }

    #[test]
    fn the_trampolines_lie_below_the_image_and_the_page_after_the_module_is_free() {
        let counter = programs::counter(RESCALE.guest_hz).unwrap();
        let (
            Loaded {
                placed,
                address_space,
            },
            memory,
            calls,
        ) = loaded(&counter.nro, Some(RESCALE));
        assert_eq!(
            placed.mappings[0],
            mapping("trampolines", BASE - PAGE, BASE)
        );
        // Code, read-execute, a block of its own (it is no part of the module).
        assert_eq!(
            block(&address_space, BASE - 1),
            (
                BASE - PAGE,
                PAGE,
                MemoryState::Code,
                MemoryPermission::USER_READ_EXECUTE
            )
        );
        assert_eq!(block(&address_space, BASE).0, BASE);
        // The pages after the module are free, up to the thread-local page, as they are on
        // a console (whose loader maps a program's arguments there, when it has any).
        let module_end = BASE + placed.module.size;
        let (start, size, state, _) = block(&address_space, module_end);
        assert_eq!((start, state), (module_end, MemoryState::Free));
        assert_eq!(start + size, placed.tls);
        // Mapped before anything is written, and the trampolines' bytes are in guest memory.
        assert_eq!(
            calls[1],
            MapperCall::Map {
                address: BASE - PAGE,
                size: PAGE,
                perms: Perms::ReadExecute
            }
        );
        let mut first = [0; 4];
        memory.read(BASE - PAGE, &mut first).unwrap();
        assert_ne!(first, [0; 4]);
    }

    #[test]
    fn the_trampolines_are_not_counted_as_the_processs_memory() {
        let counter = programs::counter(RESCALE.guest_hz).unwrap();
        let (with, _, _) = loaded(&counter.nro, Some(RESCALE));
        let (without, _, _) = loaded(&counter.nro, None);
        assert!(with.placed.mappings.iter().any(|m| m.name == "trampolines"));
        // `GetInfo` reports what a console would: the module and the stack.
        let expected = with.placed.module.size + STACK_SIZE;
        for loaded in [&with, &without] {
            let space = &loaded.address_space;
            assert_eq!(space.used_user_physical_memory_size(), expected);
            assert_eq!(space.limit().reserved(), expected);
        }
        // And everything else is where it is without them.
        assert_eq!(with.placed.stack_top, without.placed.stack_top);
        assert_eq!(with.placed.tls, without.placed.tls);
    }

    /// A [`Mapper`] that refuses its `n`th call (from zero) and accepts the rest.
    struct RefuseCall(usize);

    impl RefuseCall {
        fn next(&mut self) -> Result<(), starquake_kernel::memory::MapperError> {
            let refused = self.0 == 0;
            self.0 = self.0.wrapping_sub(1);
            if refused {
                Err(starquake_kernel::memory::MapperError("refused".into()))
            } else {
                Ok(())
            }
        }
    }

    impl Mapper for RefuseCall {
        fn map(
            &mut self,
            _: u64,
            _: u64,
            _: Perms,
        ) -> Result<(), starquake_kernel::memory::MapperError> {
            self.next()
        }

        fn unmap(&mut self, _: u64, _: u64) -> Result<(), starquake_kernel::memory::MapperError> {
            self.next()
        }

        fn protect(
            &mut self,
            _: u64,
            _: u64,
            _: Perms,
        ) -> Result<(), starquake_kernel::memory::MapperError> {
            self.next()
        }
    }

    #[test]
    fn a_mapping_the_translation_refuses_fails_the_load_and_says_which() {
        // The counter program's calls, in order: the code, the trampolines, one segment (its
        // text), the stack, the thread-local page.
        let counter = programs::counter(RESCALE.guest_hz).unwrap();
        let program = Program::parse(&counter.nro).unwrap();
        let attempt = |call: usize| {
            load(&program, Some(RESCALE), Box::new(RefuseCall(call)), &flat()).map(|_| ())
        };
        assert!(matches!(
            attempt(0),
            Err(LoadError::AddressSpace(CreateError::Code(
                SvcError::Internal(_)
            )))
        ));
        for (call, what) in [
            (1, "counter trampolines"),
            (2, "segments"),
            (3, "stack"),
            (4, "TLS page"),
        ] {
            match attempt(call) {
                Err(LoadError::Map {
                    what: failed,
                    error,
                }) => {
                    assert_eq!(failed, what, "call {call}");
                    assert!(matches!(error, SvcError::Internal(_)), "{error:?}");
                }
                other => assert_eq!(other.map_err(|e| e.to_string()), Err(what.to_owned())),
            }
        }
        // With no call refused the load succeeds, after exactly those five.
        assert_eq!(attempt(5), Ok(()));
        assert_eq!(
            LoadError::Map {
                what: "stack",
                error: SvcError::Kernel(KernelError::LimitReached)
            }
            .to_string(),
            "the program's stack could not be mapped: limit reached"
        );
    }

    #[test]
    fn a_program_larger_than_the_application_pool_is_refused() {
        // More `.bss` than an application may use: the kernel cannot reserve the code.
        let pool = u32::try_from(MEMORY_ARRANGEMENT.application_pool_size()).unwrap();
        let build = |bss: u32| {
            let file = NroBuilder::new(&[a64::nop()])
                .bss_size(bss)
                .build()
                .unwrap();
            let (mapper, _log) = RecordingMapper::new();
            load(
                &Program::parse(&file).unwrap(),
                None,
                Box::new(mapper),
                &flat(),
            )
            .map(|_| ())
        };
        // One page of image: with a page less than the pool of `.bss` the code is exactly the
        // pool, which fits, and the stack then does not; with the whole pool it is one page
        // too much.
        let page = u32::try_from(PAGE).unwrap();
        assert_eq!(
            build(pool - page),
            Err(LoadError::Map {
                what: "stack",
                error: SvcError::Kernel(KernelError::LimitReached)
            })
        );
        let error = LoadError::AddressSpace(CreateError::Code(SvcError::Kernel(
            KernelError::LimitReached,
        )));
        assert_eq!(build(pool), Err(error.clone()));
        assert_eq!(
            error.to_string(),
            "the program's address space could not be created: the program's code could not be \
             mapped: limit reached"
        );
    }

    #[test]
    fn an_image_guest_memory_cannot_hold_is_a_write_fault() {
        let file = programs::hello().unwrap();
        let program = Program::parse(&file).unwrap();
        let (mapper, _log) = RecordingMapper::new();
        let too_small = FlatMemory::new(BASE, file.len() - 1);
        let error = load(&program, None, Box::new(mapper), &too_small).unwrap_err();
        assert_eq!(error, LoadError::Write(Fault { address: BASE }));
        assert_eq!(
            error.to_string(),
            "the program could not be written into guest memory: guest memory fault at \
             0x80000000"
        );
        let (mapper, _log) = RecordingMapper::new();
        let exact = FlatMemory::new(BASE, file.len());
        assert!(load(&program, None, Box::new(mapper), &exact).is_ok());
    }
}
