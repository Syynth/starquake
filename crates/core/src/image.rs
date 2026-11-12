//! Loading a program into a guest address space: what process it is ([`Program`], with its
//! description from its metadata), where its modules, their `.bss` and the counter trampolines
//! go ([`place`]), and creating the process's memory around them with the kernel ([`load`]):
//! the code and the main thread's stack. The main thread itself, and its thread-local page, are
//! the kernel's process creation's (`starquake_kernel::process::run`, called by the core loop).
//!
//! A program is an NRO ([`Program::parse`]: one module, `main`) or an ExeFS
//! ([`Program::from_exefs`]: `main.npdm` and up to thirteen NSO modules, in the loader's
//! `starquake_loader::exefs::MODULE_NAMES` order). What each module puts in its pages is the
//! loader's ([`CodeSet`]: segments, `MOD0` and `.bss`, the module's size); relocations are the
//! guest's own (the loader's `codeset` module docs, "Relocations").
//!
//! [`load`] follows Eden's process start for a program (paths relative to the reference folder,
//! all @ cf654052):
//!
//! 1. `KProcess::LoadFromMetadata` and `KProcess::Initialize`
//!    (`eden/src/core/hle/kernel/k_process.cpp`): the kernel lays out the address space around
//!    the code, reserves the code's memory and maps it with no guest access
//!    (`starquake_kernel::memory::AddressSpace::new`). The code is every module, one after the
//!    other from [`IMAGE_BASE`] (`AppLoader_DeconstructedRomDirectory::Load`,
//!    `eden/src/core/loader/deconstructed_rom_directory.cpp`: `next_load_addr` is the end of the
//!    module before).
//! 2. `KProcess::LoadModule`, for each module: the module is written, then each segment is
//!    reprotected (`set_process_memory_permission`).
//! 3. `KProcess::Run`: the main thread's stack, of the description's size, is reserved and
//!    mapped in the stack region (`create_main_thread_stack`). The rest of `KProcess::Run` (the
//!    main thread and its thread-local page, in the kernel-map region) is the kernel's
//!    `process::run`.
//!
//! ```text
//! IMAGE_BASE - n pages   counter trampolines of every module (only when a read was patched)
//! IMAGE_BASE             the first module (rtld, or main), then .bss
//!                        each later module, from the end of the one before
//! above the code, each 2 MiB aligned (starquake_kernel::memory::layout):
//!   kernel-map region    the kernel's thread-local pages (not this module's)
//!   stack region         the stack, four pages in
//!   alias region, heap region
//! ```
//!
//! # The process's description
//!
//! [`Program::description`] is what the kernel creates the process from. For an ExeFS it comes
//! from `main.npdm` as Eden's `KProcess::LoadFromMetadata` reads it
//! (`eden/src/core/hle/kernel/k_process.cpp` and
//! `eden-pr4588/src/core/file_sys/program_metadata.cpp` @ e5fb3b97):
//!
//! | Description | NPDM | Eden |
//! |---|---|---|
//! | name | `META` name, its first [`metadata::PROCESS_NAME_SIZE`] bytes | `std::memcpy(params.name.data(), name.data(), sizeof(params.name))`, `CreateProcessParameter::name` is 12 bytes (`eden/src/core/hle/kernel/svc_types.h`) |
//! | program id | `ACI0` title id | `GetTitleID` returns `aci_header.title_id` |
//! | address space | `META` flags bits 1-3 | `GetAddressSpaceType` |
//! | cores, priorities | the `ACI0` kernel capabilities' thread info | `Initialize(kernel, params, metadata.GetKernelCapabilities(), ...)`, the `ACI0` list (`aci_kernel_capabilities`) |
//! | main thread priority and core, stack size | `META` 0xE, 0xF, 0x1C | `LoadParameters{GetMainThreadPriority(), GetMainThreadStackSize()}`, `m_ideal_core_id = metadata.GetMainThreadCore()` |
//! | system resource size | `META` 0x14 | `system_resource_num_pages = GetSystemResourceSize() / PageSize` |
//! | handle table size | the `ACI0` handle-table capability, zero without one | `KCapabilities::SetHandleTableCapability`; `m_handle_table_size = 0` before the list is read |
//! | console | Switch 2 when the `ACID` is in the Switch 2 layout | `m_uses_pauth_returns = metadata.UsesExtendedAcid()` (Eden PR #4588) |
//!
//! The console decides the counter rate (`crate::clock`), so the patcher's rate and the
//! kernel's `GetSystemTick` follow the program's metadata, and the memory arrangement
//! ([`memory_arrangement`]). The description is checked with the kernel's own checks
//! (`starquake_kernel::process::ProcessInfo::new`) when the program is parsed, so a program
//! the kernel would refuse is refused before a VM is made.
//!
//! The `ACI0` is checked against its `ACID` (`Npdm::validate_against_acid`) and a violation is
//! logged, never refused: the loader's `npdm::validate` docs say to use it as a diagnostic, and
//! Eden trusts both blocks. Test: `an_aci0_its_acid_does_not_permit_is_logged_not_refused`.
//!
//! An NRO has no metadata: it is described by Eden's defaults for a program with no NPDM
//! (`ProcessDescription::homebrew`), built for [`crate::clock::NRO_CONSOLE`].
//!
//! Divergences from Eden in the description, each pinned by a test here:
//!
//! - **No thread-info capability** is [`LoadError::NoThreadInfo`]. Eden leaves the process's
//!   core and priority masks empty and goes on; its main thread then fails a soft assert
//!   (`KThread::Initialize`). Test: `metadata_without_thread_info_is_refused`.
//! - **A name cut inside a character** keeps the characters before the cut: the parser reads
//!   the name as UTF-8 text, where Eden copies twelve raw bytes. Test:
//!   `the_process_name_is_the_first_twelve_bytes_of_the_metadatas`.
//!
//! # Placement
//!
//! - **The code goes at [`IMAGE_BASE`]**, Eden's address for 39-bit code with no randomisation
//!   (`KProcess::LoadFromMetadata`: `code_address = 0x8000'0000 + aslr_space_offset`), and the
//!   regions are not randomised either (kernel spec: deterministic by default). Only a 39-bit
//!   address space is built (the kernel's `ProcessError::AddressSpace`).
//! - **The memory arrangement** is the console's ([`memory_arrangement`]): 4 GiB for a Switch 1
//!   program, 12 GiB for a Switch 2 one.
//!
//! # Permissions
//!
//! [`Layout::module_regions`] says what the modules' pages are: whole pages, each with the
//! permissions guest code has there. The permissions are Eden's
//! (`eden/src/core/hle/kernel/k_process.cpp` @ cf654052):
//!
//! | What | Guest code may | Eden |
//! |---|---|---|
//! | `.text` | read, execute | `KProcess::LoadModule`, `ReprotectSegment(CodeSegment, ReadExecute)` |
//! | `.rodata` | read | `ReprotectSegment(RODataSegment, Read)` |
//! | `.data`, `.bss` | read, write | `ReprotectSegment(DataSegment, ReadWrite)`; both loaders add `.bss` to the data segment (the loader's `codeset` module docs) |
//! | pages of a module in no segment | nothing (mapped, so the loader can write the image) | `KProcess::Initialize`, "Map the code region": `KernelRead \| NotMapped` |
//! | stack | read, write | `KProcess::Run`, `KMemoryState::Stack`, `UserReadWrite` |
//! | counter trampolines | read, execute | starquake's own: see below |
//!
//! A segment covers whole pages: its size is rounded up to a page, as `LoadNroImpl` and
//! `AppLoader_NSO::LoadModule` do (`PageAlignSize`). Divergences from Eden, each pinned by a
//! test here:
//!
//! - **The counter trampolines** are pages a console does not have. Every module's are in one
//!   run of pages, mapped as code, read-execute, right below the first module, where no region
//!   is placed and nothing a program asks about lies: the pages after the last module, where a
//!   console's loader would put a program's arguments, are free as on a console
//!   (`the_trampolines_lie_below_the_image_and_the_page_after_the_module_is_free`). They are
//!   not counted as the process's memory, so `GetInfo` reports what a console would
//!   (`the_trampolines_are_not_counted_as_the_processs_memory`). `QueryMemory` does show them.
//!   Under NCE, Eden appends its patch section to each module's image instead
//!   (`eden/src/core/loader/nro.cpp`, `LoadNroImpl`, `HAS_NCE`), where it is counted.
//!   A module's trampolines must be within a branch's reach (128 MiB) of every read the module
//!   has: a set whose patched reads lie further above the trampolines is refused with the
//!   patcher's `BranchOutOfRange` (`a_read_out_of_the_trampolines_reach_is_refused`); placing
//!   trampolines between modules would lift that limit and move the modules off Eden's
//!   addresses. Guess, not checked against any source here: a title's `rtld` finds modules by
//!   walking the address space for code; it walks upwards from its own module, so it does not
//!   meet pages below the first module. Running a retail title would confirm it.
//!   While guest code is inside a trampoline its program counter is the trampoline's address,
//!   not the patched read's: an exit there (a fault on the two stack words a trampoline saves,
//!   or the timeout's interrupt) is reported at that address, which the memory map names
//!   `trampolines`, and a resumed guest carries on in the trampoline. Nothing maps it back to
//!   the read it stands for.
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
//!   running Eden on such a file would confirm it. Only an NRO reaches this check: an NSO's
//!   code set has refused a misaligned segment already.
//! - Two segments that share a page (after rounding) are [`LoadError::OverlappingSegments`]
//!   (`segments_that_share_a_page_are_an_error`). Eden reprotects in the order text, rodata,
//!   data. When every page of the later segment has one memory state and one permission at
//!   that moment (true of the three layouts the test uses), the later segment's permissions
//!   win for the shared pages. Otherwise `CheckMemoryState` fails
//!   (`R_UNLESS(info.m_permission == first_perm, ResultInvalidCurrentMemory)`, and the same
//!   check on `m_state`), the result is discarded, the shared page keeps the earlier segment's
//!   permissions and the rest of the later segment stays `KernelRead | NotMapped`. Only an NRO
//!   reaches this check: an NSO's code set has refused segments out of order already.
//! - An NRO's `.bss` is the pages `[image_end, bss_end)`, its own range, where `LoadNroImpl`
//!   adds it to the data segment (the loader's `codeset` module docs;
//!   `bss_is_read_write_where_it_lies_even_when_the_data_segment_ends_before_the_image`).
//! - A reprotection the kernel refuses is [`LoadError::Map`], not ignored as Eden's
//!   `ReprotectSegment` ignores it. [`place`] has refused the layouts that would make it fail.
//!
//! Memory the kernel maps reads as zero until it is written (`crate::space`), so `.bss` and the
//! stack need no writes.

mod metadata;

use starquake_kernel::memory::{
    AddressSpace as KernelSpace, CreateError, Mapper, MemoryArrangement, MemoryPermission,
    MemoryState, ProcessMemory, RegionOffsets, SvcPermission,
};
use starquake_kernel::process::{Console, ProcessDescription, ProcessError};
use starquake_kernel::result::SvcError;
use starquake_loader::codeset::{CodeSet, CodeSetError, NroOrCodeSetError};
use starquake_loader::exefs::ExeFs;
use starquake_loader::npdm::validate::AcidViolation;
use starquake_loader::nro::NroError;
use starquake_loader::patch::{CounterRescale, PatchConfig, PatchError, patch_text};
use starquake_memory::{Fault, GuestMemory, PAGE_SIZE, Perms};

use crate::crash::Module;
use crate::report::Mapping;

/// The guest's page size (`docs/specs/execution-model.md`: a 4 KiB granule).
pub(crate) const PAGE: u64 = PAGE_SIZE;
/// Guest virtual address the program's first module goes at: Eden's address for 39-bit code
/// without randomisation (see the module docs).
pub(crate) const IMAGE_BASE: u64 = 0x8000_0000;
/// The name an NRO's one module goes by in the memory map and crash reports.
pub(crate) const MODULE_NAME: &str = starquake_loader::exefs::MAIN_MODULE;
/// Alignment of each module's first trampoline (`patch_text` requires 8).
const TRAMPOLINE_ALIGN: u64 = 8;

/// Why a program could not be loaded into guest memory.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum LoadError {
    /// The file is not a well-formed NRO.
    #[error("the program is not a loadable NRO: {0}")]
    Nro(#[from] NroError),
    /// A module cannot be loaded (the loader's code set refused it).
    #[error(transparent)]
    CodeSet(#[from] CodeSetError),
    /// The program's metadata has no thread-info capability, so its process has no cores or
    /// priorities.
    #[error("the program's metadata gives its process no cores or priorities")]
    NoThreadInfo,
    /// The kernel would refuse the process the metadata describes.
    #[error("the program's metadata describes a process the kernel refuses: {0}")]
    Process(#[from] ProcessError),
    /// The counter patcher rejected a module's text.
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
    /// A size in the program, or an address derived from one, does not fit 64 bits.
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

impl From<NroOrCodeSetError> for LoadError {
    fn from(error: NroOrCodeSetError) -> Self {
        match error {
            NroOrCodeSetError::Nro(error) => Self::Nro(error),
            NroOrCodeSetError::CodeSet(error) => Self::CodeSet(error),
        }
    }
}

/// One module of a program.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ProgramModule {
    /// Its name in the memory map and crash reports: its ExeFS file name, or `main`.
    pub(crate) name: String,
    /// What it puts in its pages.
    pub(crate) code: CodeSet,
}

/// A parsed program: its modules in placement order, and the process it runs as.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Program {
    modules: Vec<ProgramModule>,
    description: ProcessDescription,
    acid_violation: Option<AcidViolation>,
}

impl Program {
    /// Parses `bytes` as an NRO: one module, described by Eden's defaults for a program with no
    /// NPDM (see the module docs).
    pub(crate) fn parse(bytes: &[u8]) -> Result<Self, LoadError> {
        let code = CodeSet::parse_nro(bytes)?;
        Self::new(
            vec![ProgramModule {
                name: MODULE_NAME.to_owned(),
                code,
            }],
            ProcessDescription::homebrew(crate::clock::NRO_CONSOLE),
            None,
        )
    }

    /// The program an ExeFS holds: its modules, and the process its `main.npdm` describes (see
    /// "The process's description" in the module docs). An `ACI0` its `ACID` does not permit
    /// is logged, not refused.
    pub(crate) fn from_exefs(exefs: ExeFs) -> Result<Self, LoadError> {
        let ExeFs { npdm, modules } = exefs;
        let description = metadata::describe(&npdm)?;
        let acid_violation = npdm.validate_against_acid().err();
        if let Some(violation) = &acid_violation {
            tracing::warn!(%violation, "the program's ACI0 asks for more than its ACID permits");
        }
        let modules = modules
            .into_iter()
            .map(|module| ProgramModule {
                name: module.name.to_owned(),
                code: module.code,
            })
            .collect();
        Self::new(modules, description, acid_violation)
    }

    /// Checks `description` as the kernel will when it creates the process.
    fn new(
        modules: Vec<ProgramModule>,
        description: ProcessDescription,
        acid_violation: Option<AcidViolation>,
    ) -> Result<Self, LoadError> {
        // The id is the kernel's to give; the checks do not read it.
        starquake_kernel::process::ProcessInfo::new(&description, 0)?;
        Ok(Self {
            modules,
            description,
            acid_violation,
        })
    }

    /// What the kernel creates the program's process from.
    pub(crate) fn description(&self) -> &ProcessDescription {
        &self.description
    }

    /// What the program's `ACI0` breaks against its `ACID`, if anything (logged when parsed).
    #[cfg(test)]
    pub(crate) fn acid_violation(&self) -> Option<&AcidViolation> {
        self.acid_violation.as_ref()
    }

    /// The modules, in placement order.
    #[cfg(test)]
    pub(crate) fn modules(&self) -> &[ProgramModule] {
        &self.modules
    }
}

/// The memory arrangement of a console: 4 GiB for a Switch 1, 12 GiB for a Switch 2.
pub(crate) const fn memory_arrangement(console: Console) -> MemoryArrangement {
    match console {
        Console::Switch1 => MemoryArrangement::FourGb,
        Console::Switch2 => MemoryArrangement::TwelveGb,
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

/// Where a program's modules and their trampolines go, and the bytes to put there: what
/// [`place`] works out from the files alone.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Layout {
    /// Address the guest starts executing at: the first byte of the first module.
    pub(crate) entry: u64,
    /// `(address, bytes)` to copy into guest memory, in address order.
    pub(crate) writes: Vec<(u64, Vec<u8>)>,
    /// Everything that was laid out, in address order.
    pub(crate) mappings: Vec<Mapping>,
    /// The modules' pages, in address order, none overlapping: every page from the first
    /// module's base to the last one's end (see "Permissions" in the module docs).
    pub(crate) module_regions: Vec<Region>,
    /// The counter trampolines' pages, right below the first module, if a read was patched.
    pub(crate) trampolines: Option<Region>,
    /// The program's modules, in address order.
    pub(crate) modules: Vec<Module>,
    /// Addresses of the counter reads that were patched, in address order.
    pub(crate) counter_reads: Vec<u64>,
}

impl Layout {
    /// Where the code ends: one past the last module's last page.
    fn code_end(&self) -> u64 {
        self.modules
            .last()
            .map_or(self.entry, |module| module.base.saturating_add(module.size))
    }
}

/// A loaded program, as the core loop and the run's report need it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Placement {
    /// Address the guest starts executing at: the first byte of the first module.
    pub(crate) entry: u64,
    /// Initial stack pointer: the end of the stack.
    pub(crate) stack_top: u64,
    /// Everything that was mapped, in address order.
    pub(crate) mappings: Vec<Mapping>,
    /// The program's modules, in address order.
    pub(crate) modules: Vec<Module>,
    /// Addresses of the counter reads that were patched, in address order.
    pub(crate) counter_reads: Vec<u64>,
    /// What the kernel creates the process from.
    pub(crate) description: ProcessDescription,
}

/// What [`load`] hands back: where things are, and the kernel's address space that holds them.
#[derive(Debug)]
pub(crate) struct Loaded {
    /// Where the program and its stack are.
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

/// A module's pages at `base`: its protected ranges with their permissions, and every other page
/// of the module with none (see "Permissions" in the module docs).
fn module_regions(code: &CodeSet, base: u64) -> Result<Vec<Region>, LoadError> {
    let mut parts = Vec::new();
    for (range, perms) in code.protections() {
        if !range.offset.is_multiple_of(PAGE) {
            return Err(LoadError::MisalignedSegment {
                offset: u32::try_from(range.offset).map_err(|_| LoadError::Overflow)?,
            });
        }
        let start = add(base, range.offset)?;
        let end = add(start, page_up(range.size)?)?;
        parts.push((start, end, perms));
    }
    parts.sort_by_key(|&(start, _, _)| start);

    let module_end = add(base, code.size)?;
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
    push(at, module_end, Perms::None);
    Ok(regions)
}

/// `code` with its counter reads rewritten for `counter` to branch to trampolines at
/// `trampolines_at`, for the module at `base`, and what the patcher did.
fn patch(
    code: &CodeSet,
    base: u64,
    trampolines_at: u64,
    counter: Option<CounterRescale>,
) -> Result<(CodeSet, starquake_loader::patch::Patched), LoadError> {
    let mut code = code.clone();
    let text_address = add(base, code.text.offset)?;
    let text = code.text_mut().ok_or(LoadError::Overflow)?;
    let patched = patch_text(text, text_address, trampolines_at, &PatchConfig { counter })?;
    Ok((code, patched))
}

/// Lays `program`'s modules out from `base`, one after the other (see the module docs),
/// rewriting their counter reads for `counter` if one is given, with every module's
/// trampolines in one run of pages right below `base`.
pub(crate) fn place(
    program: &Program,
    base: u64,
    counter: Option<CounterRescale>,
) -> Result<Layout, LoadError> {
    // Where each module goes.
    let mut bases = Vec::with_capacity(program.modules.len());
    let mut at = base;
    for module in &program.modules {
        bases.push(at);
        at = add(at, module.code.size)?;
    }

    // The trampolines end where the first module starts, so where they start depends on how
    // many there are: patch once to count each module's (how many there are does not depend on
    // where they go; the start of the module's own text, rounded down, is always in reach),
    // then again with their addresses.
    let mut offsets = Vec::with_capacity(program.modules.len());
    let mut total = 0_u64;
    for (module, &module_base) in program.modules.iter().zip(&bases) {
        let text_address = add(module_base, module.code.text.offset)?;
        let near = text_address - text_address % TRAMPOLINE_ALIGN;
        let (_, first_pass) = patch(&module.code, module_base, near, counter)?;
        let len = u64::try_from(first_pass.trampolines.len()).map_err(|_| LoadError::Overflow)?;
        offsets.push((total, len));
        total = add(total, len)?
            .checked_next_multiple_of(TRAMPOLINE_ALIGN)
            .ok_or(LoadError::Overflow)?;
    }
    let trampolines_len = page_up(total)?;
    let trampolines_at = base
        .checked_sub(trampolines_len)
        .ok_or(LoadError::Overflow)?;

    let mut area = Vec::new();
    let mut writes = Vec::new();
    let mut mappings = Vec::new();
    let mut module_regions_all = Vec::new();
    let mut modules = Vec::new();
    let mut counter_reads = Vec::new();
    let mut map = |name: String, start: u64, end: u64| {
        if start < end {
            mappings.push(Mapping { name, start, end });
        }
    };
    map("trampolines".to_owned(), trampolines_at, base);
    for ((module, &module_base), &(offset, len)) in program.modules.iter().zip(&bases).zip(&offsets)
    {
        // Patched again even with no trampolines: a read into `xzr` becomes a `nop` and needs
        // none.
        let (code, patched) = patch(
            &module.code,
            module_base,
            add(trampolines_at, offset)?,
            counter,
        )?;
        let placed_len =
            u64::try_from(patched.trampolines.len()).map_err(|_| LoadError::Overflow)?;
        if placed_len != len {
            return Err(LoadError::Overflow);
        }
        if !patched.trampolines.is_empty() {
            let start = usize::try_from(offset).map_err(|_| LoadError::Overflow)?;
            area.resize(start, 0);
            area.extend_from_slice(&patched.trampolines);
        }
        counter_reads.extend(patched.sites.iter().map(|site| site.address));

        let name = &module.name;
        for (segment, range) in [".text", ".rodata", ".data"]
            .into_iter()
            .zip(code.segments())
        {
            let start = add(module_base, range.offset)?;
            map(format!("{name}{segment}"), start, add(start, range.size)?);
        }
        let module_end = add(module_base, code.size)?;
        if code.bss.size != 0 {
            map(
                format!("{name}.bss"),
                add(module_base, code.bss.offset)?,
                module_end,
            );
        }
        module_regions_all.extend(module_regions(&code, module_base)?);
        for piece in &code.pieces {
            writes.push((add(module_base, piece.offset)?, piece.bytes.clone()));
        }
        modules.push(Module {
            name: name.clone(),
            base: module_base,
            size: code.size,
            build_id: code.build_id.to_vec(),
        });
    }
    mappings.sort_by_key(|mapping| mapping.start);
    if !area.is_empty() {
        writes.insert(0, (trampolines_at, area));
    }
    Ok(Layout {
        entry: base,
        writes,
        mappings,
        module_regions: module_regions_all,
        trampolines: (trampolines_len != 0).then_some(Region {
            start: trampolines_at,
            size: trampolines_len,
            perms: Perms::ReadExecute,
        }),
        modules,
        counter_reads,
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
/// making the kernel's mappings real, copies the modules into `memory` (that same address
/// space, as host code sees it) and gives the process a main thread's stack of the size its
/// description asks for. Counter reads are rewritten for `counter` if one is given. See the
/// module docs for the steps.
pub(crate) fn load(
    program: &Program,
    counter: Option<CounterRescale>,
    mapper: Box<dyn Mapper>,
    memory: &dyn GuestMemory,
) -> Result<Loaded, LoadError> {
    let description = program.description();
    let layout = place(program, IMAGE_BASE, counter)?;
    let refused = |what: &'static str| move |error: SvcError| LoadError::Map { what, error };
    let mut space = KernelSpace::new(
        &ProcessMemory {
            code_address: layout.entry,
            code_size: layout.code_end() - layout.entry,
            arrangement: memory_arrangement(description.console),
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
        .create_main_thread_stack(description.main_thread_stack_size)
        .map_err(refused("stack"))?;

    let mut mappings = layout.mappings;
    if stack_top != 0 {
        // The kernel rounds the stack up to whole pages; a stack of no bytes has no top.
        let stack_size = page_up(description.main_thread_stack_size)?;
        mappings.push(Mapping {
            name: "stack".to_owned(),
            start: stack_top - stack_size,
            end: stack_top,
        });
    }
    mappings.sort_by_key(|mapping| mapping.start);
    Ok(Loaded {
        placed: Placement {
            entry: layout.entry,
            stack_top,
            mappings,
            modules: layout.modules,
            counter_reads: layout.counter_reads,
            description: description.clone(),
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
    use crate::clock::NRO_CONSOLE;

    /// Where a real run puts the program.
    const BASE: u64 = IMAGE_BASE;
    /// The counter rates of a real run: Apple Silicon's and the kernel's.
    const RESCALE: CounterRescale = CounterRescale {
        host_hz: 24_000_000,
        guest_hz: NRO_CONSOLE.counter_rate().hz(),
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

    /// The stack an NRO's process gets: Eden's default description's.
    fn stack_size() -> u64 {
        ProcessDescription::homebrew(NRO_CONSOLE).main_thread_stack_size
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
            placed.modules,
            [Module {
                name: "main".into(),
                base: BASE,
                size: 2 * PAGE,
                build_id: vec![0; BUILD_ID_SIZE],
            }]
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
        assert_eq!(placed.modules[0].size, 4 * PAGE);
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
            let module_end = placed.code_end();
            let mut at = placed.modules[0].base;
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
        assert_eq!(placed.modules[0].size, 3 * PAGE);
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
        for segment in program.modules()[0].code.segments() {
            let end = BASE + segment.offset + segment.size;
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
        assert_eq!(
            Program::parse(&file),
            Err(LoadError::CodeSet(CodeSetError::BadBss))
        );
        assert_eq!(
            LoadError::CodeSet(CodeSetError::BadBss).to_string(),
            "the NRO's MOD0 header describes a .bss that ends before it starts"
        );
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
    fn a_loaded_program_has_its_stack_where_the_kernel_places_it() {
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
        let first_tls_page = regions.kernel_map.start + guard;
        assert_eq!(placed.entry, IMAGE_BASE);
        assert_eq!(placed.stack_top, stack_bottom + stack_size());
        assert_eq!(placed.stack_top % 16, 0, "SP must be 16-byte aligned");
        assert_eq!(
            placed.mappings,
            [
                mapping("main.text", BASE, BASE + PAGE),
                mapping("main.rodata", BASE + PAGE, BASE + 2 * PAGE),
                mapping("main.data", BASE + 2 * PAGE, BASE + 3 * PAGE),
                mapping("stack", stack_bottom, placed.stack_top),
            ]
        );
        assert_eq!(placed.modules[0].size, 3 * PAGE);
        assert_eq!(
            placed.description,
            ProcessDescription::homebrew(NRO_CONSOLE)
        );

        // The kernel's blocks: each segment with Eden's state and permission, and the stack. The
        // thread-local pages are the kernel's process creation's: none is mapped here.
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
                stack_size(),
                MemoryState::Stack,
                MemoryPermission::USER_READ_WRITE
            )
        );
        assert_eq!(at(first_tls_page).2, MemoryState::Free);
        // The code and the stack are the process's memory
        // (`KProcess::GetUsedUserPhysicalMemorySize`).
        assert_eq!(
            address_space.used_user_physical_memory_size(),
            3 * PAGE + stack_size()
        );
        assert_eq!(
            address_space.limit().limit(),
            MemoryArrangement::FourGb.application_pool_size()
        );

        // The image is in guest memory.
        let mut read = vec![0; file.len()];
        memory.read(BASE, &mut read).unwrap();
        assert_eq!(read, file);
    }

    #[test]
    fn loading_asks_the_translation_for_each_mapping_in_edens_order() {
        let (Loaded { placed, .. }, _, calls) = loaded(&three_segments(), None);
        let stack_bottom = placed.stack_top - stack_size();
        // The code with no guest access, the three segments reprotected, the stack.
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
                map(stack_bottom, stack_size(), Perms::ReadWrite),
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
        // The pages after the module are free, up to the kernel-map region (where the kernel
        // puts thread-local pages), as they are on a console (whose loader maps a program's
        // arguments there, when it has any).
        let module_end = BASE + placed.modules[0].size;
        let (start, size, state, _) = block(&address_space, module_end);
        assert_eq!((start, state), (module_end, MemoryState::Free));
        assert!(start + size >= address_space.regions().kernel_map.start);
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
        let expected = with.placed.modules[0].size + stack_size();
        for loaded in [&with, &without] {
            let space = &loaded.address_space;
            assert_eq!(space.used_user_physical_memory_size(), expected);
            assert_eq!(space.limit().reserved(), expected);
        }
        // And everything else is where it is without them.
        assert_eq!(with.placed.stack_top, without.placed.stack_top);
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
        // text), the stack.
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
        for (call, what) in [(1, "counter trampolines"), (2, "segments"), (3, "stack")] {
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
        // With no call refused the load succeeds, after exactly those four.
        assert_eq!(attempt(4), Ok(()));
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
        let pool = u32::try_from(MemoryArrangement::FourGb.application_pool_size()).unwrap();
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

    // ----- ExeFS programs: several modules, and a description from the metadata ---------------

    const HELLO_NSO: &[u8] = include_bytes!("../../../tests/homebrew/hello/hello.nso");
    const HELLO_NPDM: &[u8] = include_bytes!("../../../tests/homebrew/hello/main.npdm");
    const HELLO_JSON: &str = include_str!("../../../tests/homebrew/hello/hello.json");

    /// An ExeFS of the hello fixture's metadata and `modules`, each the fixture's NSO.
    fn exefs(modules: &[&'static str]) -> ExeFs {
        let mut files = vec![(starquake_loader::exefs::NPDM_FILE, HELLO_NPDM)];
        files.extend(modules.iter().map(|name| (*name, HELLO_NSO)));
        ExeFs::parse(&files).unwrap()
    }

    /// An NSO's code set: `text` padded with `nop`s to whole pages, no rodata, eight bytes of
    /// data on the page after the text, and `bss_size` bytes of `.bss`.
    fn nso(text: &[u32], bss_size: u32) -> CodeSet {
        use starquake_loader::nso::{Nso, NsoSegment, RodataExtent};
        let page_words = usize::try_from(PAGE).unwrap() / 4;
        let mut words = text.to_vec();
        words.resize(words.len().max(1).next_multiple_of(page_words), a64::nop());
        let text = a64::to_bytes(&words);
        let end = u32::try_from(text.len()).unwrap();
        let none = RodataExtent { offset: 0, size: 0 };
        CodeSet::from_nso(Nso {
            text: NsoSegment {
                memory_offset: 0,
                data: text,
            },
            rodata: NsoSegment {
                memory_offset: end,
                data: Vec::new(),
            },
            data: NsoSegment {
                memory_offset: end,
                data: vec![0xD7; 8],
            },
            bss_size,
            build_id: [0x42; starquake_loader::codeset::BUILD_ID_SIZE],
            flags: 0,
            module_name_offset: 0,
            module_name_size: 0,
            api_info: none,
            dynstr: none,
            dynsym: none,
        })
        .unwrap()
    }

    /// A program of `modules`, described as an NRO is.
    fn program_of(modules: Vec<(&str, CodeSet)>) -> Program {
        Program::new(
            modules
                .into_iter()
                .map(|(name, code)| ProgramModule {
                    name: name.to_owned(),
                    code,
                })
                .collect(),
            ProcessDescription::homebrew(NRO_CONSOLE),
            None,
        )
        .unwrap()
    }

    fn read_counter() -> u32 {
        a64::mrs(0, a64::SysReg::CntPctEl0).unwrap()
    }

    /// Where the `b` at `address`, in `bytes` placed at `at`, branches to.
    fn branch_target(bytes: &[u8], at: u64, address: u64) -> u64 {
        let word = word_at(bytes, address - at);
        assert_eq!(word & 0xFC00_0000, 0x1400_0000, "not a b: {word:#x}");
        let distance = (i64::from(word & 0x03FF_FFFF) << 38) >> 36;
        address.checked_add_signed(distance).unwrap()
    }

    /// Loads `program` into flat memory with a recording mapper.
    fn load_program(program: &Program) -> Loaded {
        let (mapper, _log) = RecordingMapper::new();
        load(program, None, Box::new(mapper), &flat()).unwrap()
    }

    #[test]
    fn an_exefs_is_described_by_its_metadata() {
        let json: serde_json::Value = serde_json::from_str(HELLO_JSON).unwrap();
        let number = |value: &serde_json::Value| match value {
            serde_json::Value::String(text) => {
                u64::from_str_radix(text.trim_start_matches("0x"), 16).unwrap()
            }
            other => other.as_u64().unwrap(),
        };
        let byte = |value: &serde_json::Value| u8::try_from(number(value)).unwrap();
        let capability = |kind: &str| {
            json["kernel_capabilities"]
                .as_array()
                .unwrap()
                .iter()
                .find(|entry| entry["type"] == kind)
                .unwrap()["value"]
                .clone()
        };
        let flags = capability("kernel_flags");
        let program = Program::from_exefs(exefs(&["main"])).unwrap();
        let description = program.description();
        let name = json["name"].as_str().unwrap();
        assert_eq!(description.name, name[..metadata::PROCESS_NAME_SIZE]);
        assert_eq!(description.program_id, number(&json["title_id"]));
        assert_eq!(
            description.address_space,
            starquake_kernel::process::AddressSpaceWidth::Is39Bit
        );
        // npdmtool's "highest" and "lowest" name the numbers, not the urgency.
        assert_eq!(
            (description.priorities.first, description.priorities.last),
            (
                byte(&flags["lowest_thread_priority"]),
                byte(&flags["highest_thread_priority"])
            )
        );
        assert_eq!(
            (description.cores.first, description.cores.last),
            (
                byte(&flags["lowest_cpu_id"]),
                byte(&flags["highest_cpu_id"])
            )
        );
        assert_eq!(
            description.main_thread_priority,
            byte(&json["main_thread_priority"])
        );
        assert_eq!(description.main_thread_core, byte(&json["default_cpu_id"]));
        assert_eq!(
            description.main_thread_stack_size,
            number(&json["main_thread_stack_size"])
        );
        assert_eq!(
            u64::try_from(description.handle_table_size).unwrap(),
            number(&capability("handle_table_size"))
        );
        assert_eq!(description.console, Console::Switch1);
        // The fixture's ACI0 is inside its ACID.
        assert_eq!(program.acid_violation(), None);
        // It differs from what an NRO gets, so the metadata is what was used.
        assert_ne!(*description, ProcessDescription::homebrew(NRO_CONSOLE));
    }

    #[test]
    fn modules_follow_one_another_from_the_base() {
        let program = Program::from_exefs(exefs(&["sdk", "main", "rtld"])).unwrap();
        let placed = place(&program, BASE, None).unwrap();
        let size = program.modules()[0].code.size;
        let names: Vec<_> = placed.modules.iter().map(|m| m.name.as_str()).collect();
        assert_eq!(names, ["rtld", "main", "sdk"]);
        for (index, module) in (0_u64..).zip(&placed.modules) {
            let base = BASE + size * index;
            assert_eq!((module.base, module.size), (base, size), "{}", module.name);
            let text = format!("{}.text", module.name);
            let found = placed.mappings.iter().find(|m| m.name == text).unwrap();
            assert_eq!(found.start, base);
        }
        assert_eq!(placed.entry, BASE);
        assert_eq!(placed.code_end(), BASE + 3 * size);
        // The regions cover the three modules with no hole and nothing outside them.
        let mut at = BASE;
        for region in &placed.module_regions {
            assert_eq!(region.start, at);
            at += region.size;
        }
        assert_eq!(at, placed.code_end());
        // Every module's segments are written at its base, and nothing else is.
        let code = &program.modules()[0].code;
        let expected: Vec<_> = (0..3_u64)
            .flat_map(|index| {
                code.pieces
                    .iter()
                    .map(move |piece| (BASE + size * index + piece.offset, piece.bytes.clone()))
            })
            .collect();
        assert_eq!(placed.writes, expected);
    }

    #[test]
    fn an_nso_modules_bss_shares_its_datas_pages_and_runs_to_the_modules_end() {
        let placed = place(&program_of(vec![("main", nso(&[], 0x1234))]), BASE, None).unwrap();
        // One page of text; data's eight bytes and 0x1234 of .bss: two read-write pages.
        assert_eq!(
            placed.regions(),
            [
                region(BASE, 1, Perms::ReadExecute),
                region(BASE + PAGE, 2, Perms::ReadWrite),
            ]
        );
        assert_eq!(
            placed.mappings,
            [
                mapping("main.text", BASE, BASE + PAGE),
                mapping("main.data", BASE + PAGE, BASE + PAGE + 8),
                mapping("main.bss", BASE + PAGE + 8, BASE + 3 * PAGE),
            ]
        );
        // With no .bss there is no .bss mapping, and data keeps its one page.
        let placed = place(&program_of(vec![("main", nso(&[], 0))]), BASE, None).unwrap();
        assert!(placed.mappings.iter().all(|m| m.name != "main.bss"));
        assert_eq!(
            placed.regions()[1],
            region(BASE + PAGE, 1, Perms::ReadWrite)
        );
    }

    #[test]
    fn every_modules_trampolines_share_one_run_below_the_first() {
        let first = nso(&[read_counter()], 0);
        let second = nso(&[a64::nop(), read_counter()], 0);
        let size = first.size;
        let program = program_of(vec![("rtld", first), ("main", second)]);
        let placed = place(&program, BASE, Some(RESCALE)).unwrap();
        let reads = [BASE, BASE + size + 4];
        assert_eq!(placed.counter_reads, reads);
        assert_eq!(
            placed.trampolines,
            Some(region(BASE - PAGE, 1, Perms::ReadExecute))
        );
        assert_eq!(
            placed.mappings[0],
            mapping("trampolines", BASE - PAGE, BASE)
        );
        // The area is written first, then each module's pieces (text, data); each read
        // branches into the area, the second module's past the first's, 8-aligned.
        let (area_at, area) = &placed.writes[0];
        assert_eq!(*area_at, BASE - PAGE);
        assert_eq!(placed.writes[1].0, BASE);
        assert_eq!(placed.writes[3].0, BASE + size);
        let first_target = branch_target(&placed.writes[1].1, BASE, reads[0]);
        let second_target = branch_target(&placed.writes[3].1, BASE + size, reads[1]);
        assert_eq!(first_target, BASE - PAGE);
        assert!(first_target < second_target && second_target < BASE);
        assert_eq!(second_target % TRAMPOLINE_ALIGN, 0);
        // Both reads are the same shape, so each module's trampolines are one module's alone:
        // the second starts at the first's length rounded up to 8, and the area ends with it.
        let alone = place(
            &program_of(vec![("main", nso(&[read_counter()], 0))]),
            BASE,
            Some(RESCALE),
        )
        .unwrap();
        let one = alone.writes[0].1.len() as u64;
        assert_eq!(
            second_target - first_target,
            one.next_multiple_of(TRAMPOLINE_ALIGN)
        );
        assert_eq!(area.len() as u64, second_target - first_target + one);
        assert_eq!(&area[..alone.writes[0].1.len()], &alone.writes[0].1[..]);
    }

    #[test]
    fn a_read_out_of_the_trampolines_reach_is_refused() {
        // A first module with no reads, sized so that the second module starts 2^27 - PAGE
        // above the trampolines (one page below the first module). `b` reaches -2^27..2^27
        // bytes, so a read may be at most 2^27 above its trampoline.
        let reach = 1_u64 << 27;
        // One page of text, then eight bytes of data: .bss takes the module to 2^27 - 2 pages.
        let filler_bss = u32::try_from(reach - 3 * PAGE - 8).unwrap();
        let filler = || nso(&[], filler_bss);
        assert_eq!(filler().size, reach - 2 * PAGE);
        let read_at = |offset: u64| {
            let mut words = vec![a64::nop(); usize::try_from(offset / 4).unwrap()];
            words.push(read_counter());
            nso(&words, 0)
        };
        // The first word of the second module's second page is 2^27 above: in reach.
        let edge = place(
            &program_of(vec![("rtld", filler()), ("main", read_at(PAGE))]),
            BASE,
            Some(RESCALE),
        )
        .unwrap();
        assert_eq!(edge.counter_reads[0] - (BASE - PAGE), reach);
        // The word after it is 2^27 + 4 above: out of reach.
        let from = BASE + filler().size + PAGE + 4;
        assert_eq!(from - (BASE - PAGE), reach + 4);
        assert_eq!(
            place(
                &program_of(vec![("rtld", filler()), ("main", read_at(PAGE + 4))]),
                BASE,
                Some(RESCALE),
            ),
            Err(LoadError::Patch(PatchError::BranchOutOfRange {
                from,
                to: BASE - PAGE
            }))
        );
    }

    #[test]
    fn a_read_into_xzr_in_any_module_becomes_a_nop_with_no_trampoline() {
        let xzr = a64::mrs(31, a64::SysReg::CntPctEl0).unwrap();
        let first = nso(&[], 0);
        let size = first.size;
        let program = program_of(vec![("rtld", first), ("main", nso(&[xzr], 0))]);
        let placed = place(&program, BASE, Some(RESCALE)).unwrap();
        assert_eq!(placed.counter_reads, [BASE + size]);
        assert_eq!(placed.trampolines, None);
        let (at, text) = &placed.writes[2];
        assert_eq!(*at, BASE + size);
        assert_eq!(word_at(text, 0), a64::nop());
    }

    #[test]
    fn a_switch2_program_gets_the_12_gb_arrangement_and_its_own_stack_size() {
        let mut files = exefs(&["main"]);
        files.npdm.acid.layout = starquake_loader::npdm::AcidLayout::Switch2;
        // Not a whole number of pages: the kernel rounds it up.
        files.npdm.main_thread_stack_size = 0x3_4001;
        let program = Program::from_exefs(files).unwrap();
        assert_eq!(program.description().console, Console::Switch2);
        let Loaded {
            placed,
            address_space,
        } = load_program(&program);
        assert_eq!(
            address_space.limit().limit(),
            MemoryArrangement::TwelveGb.application_pool_size()
        );
        let stack = placed.mappings.iter().find(|m| m.name == "stack").unwrap();
        assert_eq!(stack.end - stack.start, 0x3_5000);
        assert_eq!(stack.end, placed.stack_top);
        assert_eq!(placed.description, *program.description());
        assert_eq!(
            memory_arrangement(Console::Switch1),
            MemoryArrangement::FourGb
        );
    }

    #[test]
    fn a_stack_of_no_bytes_has_no_mapping() {
        let mut files = exefs(&["main"]);
        files.npdm.main_thread_stack_size = 0;
        let placed = load_program(&Program::from_exefs(files).unwrap()).placed;
        assert_eq!(placed.stack_top, 0);
        assert!(placed.mappings.iter().all(|m| m.name != "stack"));
    }

    #[test]
    fn an_aci0_its_acid_does_not_permit_is_logged_not_refused() {
        let mut files = exefs(&["main"]);
        let title_id = files.npdm.acid.title_id_max + 1;
        files.npdm.aci.title_id = title_id;
        let program = Program::from_exefs(files).unwrap();
        assert!(matches!(
            program.acid_violation(),
            Some(AcidViolation::TitleIdOutOfRange { title_id: id, .. }) if *id == title_id
        ));
        assert_eq!(program.description().program_id, title_id);
    }

    #[test]
    fn a_description_the_kernel_refuses_is_refused_when_parsed() {
        let first = Program::from_exefs(exefs(&["main"]))
            .unwrap()
            .description()
            .priorities
            .first;
        let mut below = exefs(&["main"]);
        below.npdm.main_thread_priority = first - 1;
        let error = Program::from_exefs(below).unwrap_err();
        assert_eq!(
            error,
            LoadError::Process(ProcessError::MainThreadPriority {
                priority: first - 1
            })
        );
        assert_eq!(
            error.to_string(),
            format!(
                "the program's metadata describes a process the kernel refuses: the main \
                 thread's priority {} is outside the process's priorities",
                first - 1
            )
        );
        let mut at = exefs(&["main"]);
        at.npdm.main_thread_priority = first;
        assert!(Program::from_exefs(at).is_ok());
    }

    #[test]
    fn an_exefs_loads_every_module_into_guest_memory() {
        let program = Program::from_exefs(exefs(&["rtld", "main"])).unwrap();
        let (mapper, _log) = RecordingMapper::new();
        let memory = flat();
        let Loaded {
            placed,
            address_space,
        } = load(&program, Some(RESCALE), Box::new(mapper), &memory).unwrap();
        let size = program.modules()[0].code.size;
        // The code is both modules, and the process counts it with its stack.
        assert_eq!(
            address_space.used_user_physical_memory_size(),
            2 * size + program.description().main_thread_stack_size
        );
        // Each module's rodata is in guest memory at its base plus rodata's offset (the
        // patcher changes only text).
        let rodata = &program.modules()[0].code.pieces[1];
        for module in &placed.modules {
            let mut read = vec![0; rodata.bytes.len()];
            memory.read(module.base + rodata.offset, &mut read).unwrap();
            assert_eq!(read, rodata.bytes, "{}", module.name);
        }
    }
}
