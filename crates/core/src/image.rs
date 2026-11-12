//! Placing a program in a guest address space: where an NRO's segments, its `.bss`, the
//! counter trampolines, a TLS block and a stack go, what guest code may do with each, and what
//! bytes to write there.
//!
//! PROVISIONAL (walking skeleton, #74). This lays one program out in a window of guest virtual
//! addresses as large as guest RAM, starting at [`IMAGE_BASE`]. There is no address-space
//! bookkeeping and no relocation: the layout is whatever fits, lowest address first.
//!
//! ```text
//! window start  the image (text, rodata, data), then .bss            "main"
//!               counter trampolines (only when a read was patched)
//!               TLS block (one page; `TPIDRRO_EL0` points at it)
//!               ... not mapped ...
//! window end    stack (the last [`STACK_SIZE`] bytes; `SP` starts at the end of the window)
//! ```
//!
//! What replaces it: the kernel's address space (kernel slice 6) owns regions, memory states
//! and where things go; process creation (kernel slice 8) owns the stack, the TLS page and the
//! registers a thread starts with.
//!
//! # Permissions
//!
//! [`Placement::regions`] says what to map: whole pages, each with the permissions guest code
//! has there. Everything between them is not mapped, so the page after the TLS block and the
//! page below the stack fault. The permissions are Eden's (`eden/src/core/hle/kernel/k_process.cpp`
//! @ cf654052):
//!
//! | What | Guest code may | Eden |
//! |---|---|---|
//! | `.text` | read, execute | `KProcess::LoadModule`, `ReprotectSegment(CodeSegment, ReadExecute)` |
//! | `.rodata` | read | `ReprotectSegment(RODataSegment, Read)` |
//! | `.data`, `.bss` | read, write | `ReprotectSegment(DataSegment, ReadWrite)`; `LoadNroImpl` (`eden/src/core/loader/nro.cpp`) adds `.bss` to the data segment |
//! | pages of the module in no segment | nothing (mapped, so the loader can write the image) | `KProcess::Initialize`, "Map the code region": `KernelRead \| NotMapped` |
//! | stack | read, write | `KProcess::Run`, `KMemoryState::Stack`, `UserReadWrite` |
//! | TLS block | read, write | `eden/src/core/hle/kernel/k_thread_local_page.cpp`, `KThreadLocalPage::Initialize`, `UserReadWrite` |
//! | counter trampolines | read, execute | starquake's own: Eden has no such page under a hypervisor |
//!
//! A segment covers whole pages: its size is rounded up to a page, as `LoadNroImpl` does
//! (`PageAlignSize`). Divergences from Eden, each pinned by a test here:
//!
//! - A segment that does not start on a page boundary is [`LoadError::MisalignedSegment`]. Eden
//!   passes it to `SetProcessMemoryPermission`, ignores the failure, and runs the program with
//!   that segment inaccessible.
//! - Two segments that share a page (after rounding) are [`LoadError::OverlappingSegments`].
//!   Eden reprotects in order, so the later segment's permissions win for the shared pages.
//!
//! The module's size follows Eden's NRO loader (`eden/src/core/loader/nro.cpp` @ cf654052,
//! `LoadNroImpl`): the image rounded up to a page, plus `.bss` rounded up to a page, with the
//! `MOD0` header's `.bss` size preferred over the NRO header's when `MOD0` is present. Divergence:
//! a `MOD0` whose `.bss` end precedes its start is [`LoadError::BadBss`] here; Eden subtracts
//! the two unchecked.
//!
//! It relies on guest RAM starting out zeroed (a fresh VM per run), so `.bss`, the TLS block
//! and the stack need no writes.

use starquake_loader::nro::{Nro, NroError};
use starquake_loader::patch::{CounterRescale, PatchConfig, PatchError, patch_text};
use starquake_memory::{Fault, GuestMemory, PAGE_SIZE, Perms};

use crate::crash::Module;
use crate::report::Mapping;

/// The guest's page size (`docs/specs/execution-model.md`: a 4 KiB granule).
pub(crate) const PAGE: u64 = PAGE_SIZE;
/// Guest virtual address the program's window starts at, and so where its image goes.
/// PROVISIONAL: the address the walking skeleton has always used (it was the base of the
/// identity-mapped bring-up RAM). Kernel slice 6 places a program in Horizon's code region.
pub(crate) const IMAGE_BASE: u64 = 0x8000_0000;
/// Size of the one guest thread's stack. PROVISIONAL: Horizon takes the main thread's stack size
/// from the NPDM, which an NRO does not have.
pub(crate) const STACK_SIZE: u64 = 0x10_0000;
/// Size of the TLS block: one page, as Horizon allocates thread-local regions out of pages.
pub(crate) const TLS_SIZE: u64 = PAGE;
/// The name the program's module goes by in the memory map and crash reports.
pub(crate) const MODULE_NAME: &str = "main";

/// Why a program could not be placed in guest memory.
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
    /// The program, its TLS block and its stack need more guest memory than there is.
    #[error("the program needs {needed:#x} bytes of guest memory and there are {available:#x}")]
    DoesNotFit {
        /// Bytes needed, stack included.
        needed: u64,
        /// Bytes of guest RAM.
        available: u64,
    },
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
    /// Guest RAM does not start and end on a page boundary, or runs past the end of the
    /// address space.
    #[error(
        "guest RAM at {base:#x}, {size:#x} bytes long, is not page-aligned or does not fit the \
         address space"
    )]
    BadRam {
        /// RAM base.
        base: u64,
        /// RAM size.
        size: u64,
    },
    /// A size in the NRO, or an address derived from one, does not fit 64 bits.
    #[error("the program's sizes overflow the address space")]
    Overflow,
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

/// A run of whole pages to map, and what guest code may do there.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Region {
    /// Guest virtual address of the first page.
    pub(crate) start: u64,
    /// Size in bytes: a non-zero multiple of [`PAGE`].
    pub(crate) size: u64,
    /// What guest code may do with the pages.
    pub(crate) perms: Perms,
}

/// Where everything goes, and the bytes to put there.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Placement {
    /// Address the guest starts executing at: the first byte of the image.
    pub(crate) entry: u64,
    /// Initial stack pointer: the end of the stack.
    pub(crate) stack_top: u64,
    /// Address of the TLS block, for `TPIDRRO_EL0`.
    pub(crate) tls: u64,
    /// `(address, bytes)` to copy into guest memory, in address order.
    pub(crate) writes: Vec<(u64, Vec<u8>)>,
    /// Everything that was laid out, in address order.
    pub(crate) mappings: Vec<Mapping>,
    /// The pages to map, in address order, none overlapping (see "Permissions" in the module
    /// docs). Every byte of `writes` and of `mappings` is inside one.
    pub(crate) regions: Vec<Region>,
    /// The program's module.
    pub(crate) module: Module,
    /// Addresses of the counter reads that were patched, in address order.
    pub(crate) counter_reads: Vec<u64>,
}

impl Placement {
    /// Copies the program into `memory`.
    pub(crate) fn write_into(&self, memory: &dyn GuestMemory) -> Result<(), Fault> {
        self.writes
            .iter()
            .try_for_each(|(address, bytes)| memory.write(*address, bytes))
    }
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

/// Lays `program` out in a window of `ram_size` bytes of guest virtual addresses at `ram_base`
/// (see the module docs), rewriting its counter reads for `counter` if one is given. The window
/// is as large as guest RAM, so a program that fits the window fits RAM.
pub(crate) fn place(
    program: &Program,
    ram_base: u64,
    ram_size: u64,
    counter: Option<CounterRescale>,
) -> Result<Placement, LoadError> {
    let bad_ram = LoadError::BadRam {
        base: ram_base,
        size: ram_size,
    };
    if !ram_base.is_multiple_of(PAGE) || !ram_size.is_multiple_of(PAGE) {
        return Err(bad_ram);
    }
    let ram_end = ram_base.checked_add(ram_size).ok_or(bad_ram)?;
    let nro = &program.nro;
    let base = ram_base;

    let image_end = add(base, page_up(u64::from(nro.size))?)?;
    let bss_end = add(image_end, page_up(u64::from(program.bss_size()?))?)?;

    // Patch a copy of the image's text; the trampolines go right after the module.
    let trampolines_at = bss_end;
    let mut image = program.image.clone();
    let text_address = add(base, u64::from(nro.text.offset))?;
    let text = usize::try_from(nro.text.offset)
        .ok()
        .zip(usize::try_from(nro.text.size).ok())
        .and_then(|(start, len)| Some(start..start.checked_add(len)?))
        .and_then(|range| image.get_mut(range))
        .ok_or(LoadError::Overflow)?;
    let patched = patch_text(text, text_address, trampolines_at, &PatchConfig { counter })?;
    let trampolines_len = u64::try_from(patched.trampolines.len())
        .map_err(|_| LoadError::Overflow)
        .and_then(page_up)?;

    let tls = add(trampolines_at, trampolines_len)?;
    let tls_end = add(tls, TLS_SIZE)?;
    let needed = add(tls_end - base, STACK_SIZE)?;
    if needed > ram_size {
        return Err(LoadError::DoesNotFit {
            needed,
            available: ram_size,
        });
    }
    let stack_bottom = ram_end - STACK_SIZE;

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
    for (name, segment) in [".text", ".rodata", ".data"]
        .into_iter()
        .zip(nro.segments())
    {
        let start = add(base, u64::from(segment.offset))?;
        let end = add(start, u64::from(segment.size))?;
        map(&format!("{MODULE_NAME}{name}"), start, end);
    }
    map(&format!("{MODULE_NAME}.bss"), image_end, bss_end);
    map("trampolines", trampolines_at, tls);
    map("tls", tls, tls_end);
    map("stack", stack_bottom, ram_end);
    mappings.sort_by_key(|mapping| mapping.start);

    let mut regions = module_regions(program, base, image_end, bss_end)?;
    for (start, end, perms) in [
        (trampolines_at, tls, Perms::ReadExecute),
        (tls, tls_end, Perms::ReadWrite),
        (stack_bottom, ram_end, Perms::ReadWrite),
    ] {
        if start < end {
            regions.push(Region {
                start,
                size: end - start,
                perms,
            });
        }
    }

    let mut writes = vec![(base, image)];
    if !patched.trampolines.is_empty() {
        writes.push((trampolines_at, patched.trampolines));
    }
    Ok(Placement {
        entry: base,
        stack_top: ram_end,
        tls,
        writes,
        mappings,
        regions,
        module: Module {
            name: MODULE_NAME.to_owned(),
            base,
            size: bss_end - base,
            build_id: nro.build_id.to_vec(),
        },
        counter_reads: patched.sites.iter().map(|site| site.address).collect(),
    })
}

#[cfg(test)]
mod tests {
    use starquake_memory::FlatMemory;
    use starquake_testkit::nro::{BUILD_ID_SIZE, CODE_OFFSET, NroBuilder};
    use starquake_testkit::{a64, programs};

    use super::*;

    /// Where a real run puts the program.
    const RAM_BASE: u64 = IMAGE_BASE;
    const RAM_SIZE: u64 = 16 << 20;
    /// The counter rates of a real run: Apple Silicon's and the kernel's.
    const RESCALE: CounterRescale = CounterRescale {
        host_hz: 24_000_000,
        guest_hz: starquake_kernel::svc::tick::CNTFREQ,
    };

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

    #[test]
    fn a_program_is_laid_out_from_the_start_of_ram_with_its_stack_at_the_end() {
        let file = programs::hello().unwrap();
        let placed = place(
            &Program::parse(&file).unwrap(),
            RAM_BASE,
            RAM_SIZE,
            Some(RESCALE),
        )
        .unwrap();
        // hello is one page of text and one of rodata, with no data, .bss or counter reads.
        assert_eq!(file.len(), 2 * 0x1000);
        assert_eq!(
            placed.mappings,
            [
                mapping("main.text", RAM_BASE, RAM_BASE + PAGE),
                mapping("main.rodata", RAM_BASE + PAGE, RAM_BASE + 2 * PAGE),
                mapping("tls", RAM_BASE + 2 * PAGE, RAM_BASE + 2 * PAGE + TLS_SIZE),
                mapping(
                    "stack",
                    RAM_BASE + RAM_SIZE - STACK_SIZE,
                    RAM_BASE + RAM_SIZE
                ),
            ]
        );
        // Text may be read and executed, rodata read, the TLS block and the stack read and
        // written; nothing lies between the TLS block and the stack.
        assert_eq!(
            placed.regions,
            [
                region(RAM_BASE, 1, Perms::ReadExecute),
                region(RAM_BASE + PAGE, 1, Perms::Read),
                region(RAM_BASE + 2 * PAGE, TLS_SIZE / PAGE, Perms::ReadWrite),
                region(
                    RAM_BASE + RAM_SIZE - STACK_SIZE,
                    STACK_SIZE / PAGE,
                    Perms::ReadWrite
                ),
            ]
        );
        assert_eq!(placed.entry, RAM_BASE);
        assert_eq!(placed.tls, RAM_BASE + 2 * PAGE);
        assert_eq!(placed.stack_top, RAM_BASE + RAM_SIZE);
        assert_eq!(placed.stack_top % 16, 0, "SP must be 16-byte aligned");
        // Nothing was patched: the image is written as it is in the file, and nothing else.
        assert_eq!(placed.writes, [(RAM_BASE, file)]);
        assert_eq!(placed.counter_reads, Vec::<u64>::new());
        assert_eq!(
            placed.module,
            Module {
                name: "main".into(),
                base: RAM_BASE,
                size: 2 * PAGE,
                build_id: vec![0; BUILD_ID_SIZE],
            }
        );
    }

    #[test]
    fn counter_reads_become_branches_to_trampolines_placed_after_the_module() {
        let counter = programs::counter(RESCALE.guest_hz).unwrap();
        let program = Program::parse(&counter.nro).unwrap();
        let placed = place(&program, RAM_BASE, RAM_SIZE, Some(RESCALE)).unwrap();

        assert_eq!(
            placed.counter_reads,
            counter.read_offsets.map(|offset| RAM_BASE + offset)
        );
        let trampolines = placed
            .mappings
            .iter()
            .find(|mapping| mapping.name == "trampolines")
            .expect("a trampoline mapping");
        // One page of text, so the trampolines start on the next page, before the TLS block.
        assert_eq!(trampolines.start, RAM_BASE + PAGE);
        assert_eq!(trampolines.end, placed.tls);
        // The trampolines are code: readable and executable, not writable.
        assert_eq!(
            placed.regions[..3],
            [
                region(RAM_BASE, 1, Perms::ReadExecute),
                region(trampolines.start, 1, Perms::ReadExecute),
                region(placed.tls, 1, Perms::ReadWrite),
            ]
        );
        assert_eq!(placed.writes.len(), 2, "an image and its trampolines");
        let (image_at, image) = &placed.writes[0];
        let (trampolines_at, code) = &placed.writes[1];
        assert_eq!((*image_at, *trampolines_at), (RAM_BASE, trampolines.start));
        assert!(!code.is_empty() && code.len() <= usize::try_from(PAGE).unwrap());
        // Each read is now `b <trampoline>` (imm26 in words, sign-extended), and each lands in
        // the trampoline area at a different place.
        let mut targets = Vec::new();
        for offset in counter.read_offsets {
            let word = word_at(image, offset);
            assert_eq!(word & 0xFC00_0000, 0x1400_0000, "not a b: {word:#x}");
            let distance = (i64::from(word & 0x03FF_FFFF) << 38) >> 36;
            let target = (RAM_BASE + offset).checked_add_signed(distance).unwrap();
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
    fn without_a_counter_rescale_nothing_is_patched() {
        let counter = programs::counter(RESCALE.guest_hz).unwrap();
        let placed = place(
            &Program::parse(&counter.nro).unwrap(),
            RAM_BASE,
            RAM_SIZE,
            None,
        )
        .unwrap();
        assert_eq!(placed.writes, [(RAM_BASE, counter.nro)]);
        assert_eq!(placed.counter_reads, Vec::<u64>::new());
        assert!(placed.mappings.iter().all(|m| m.name != "trampolines"));
        // With no trampolines the TLS block follows the module directly.
        assert_eq!(placed.tls, RAM_BASE + PAGE);
    }

    #[test]
    fn bss_follows_the_image_rounded_up_to_a_page() {
        let file = NroBuilder::new(&[a64::nop()])
            .data(b"data")
            .bss_size(0x1234)
            .build()
            .unwrap();
        let placed = place(&Program::parse(&file).unwrap(), RAM_BASE, RAM_SIZE, None).unwrap();
        // One page of text, none of rodata, one of data, then two pages of .bss.
        assert_eq!(
            placed.mappings[..4],
            [
                mapping("main.text", RAM_BASE, RAM_BASE + PAGE),
                mapping("main.data", RAM_BASE + PAGE, RAM_BASE + 2 * PAGE),
                mapping("main.bss", RAM_BASE + 2 * PAGE, RAM_BASE + 4 * PAGE),
                mapping("tls", RAM_BASE + 4 * PAGE, RAM_BASE + 5 * PAGE),
            ]
        );
        assert_eq!(placed.module.size, 4 * PAGE);
        assert!(CODE_OFFSET < usize::try_from(PAGE).unwrap());
        // Data and .bss are both read-write, each its own run of pages.
        assert_eq!(
            placed.regions[..4],
            [
                region(RAM_BASE, 1, Perms::ReadExecute),
                region(RAM_BASE + PAGE, 1, Perms::ReadWrite),
                region(RAM_BASE + 2 * PAGE, 2, Perms::ReadWrite),
                region(RAM_BASE + 4 * PAGE, 1, Perms::ReadWrite),
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
            let placed = place(&program, RAM_BASE, RAM_SIZE, Some(RESCALE)).unwrap();
            let inside = |start: u64, end: u64| {
                placed
                    .regions
                    .iter()
                    .any(|r| r.start <= start && end <= r.start + r.size)
            };
            for pair in placed.regions.windows(2) {
                assert!(pair[0].start + pair[0].size <= pair[1].start, "{pair:?}");
            }
            for r in &placed.regions {
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
        }
    }

    #[test]
    fn pages_of_the_module_in_no_segment_are_mapped_without_guest_access() {
        // Drop rodata from the header: its page stays part of the image, between text and data.
        let mut file = three_segments();
        set_segment(&mut file, 1, 0, 0);
        let placed = place(&Program::parse(&file).unwrap(), RAM_BASE, RAM_SIZE, None).unwrap();
        assert_eq!(
            placed.regions[..3],
            [
                region(RAM_BASE, 1, Perms::ReadExecute),
                region(RAM_BASE + PAGE, 1, Perms::None),
                region(RAM_BASE + 2 * PAGE, 1, Perms::ReadWrite),
            ]
        );
        // The image is still written whole, across the page no guest code can touch.
        assert_eq!(placed.writes, [(RAM_BASE, file)]);
    }

    #[test]
    fn a_segment_is_rounded_up_to_whole_pages() {
        // A rodata of one byte still gets its whole page, read-only.
        let mut file = three_segments();
        set_segment(&mut file, 1, u32::try_from(PAGE).unwrap(), 1);
        let placed = place(&Program::parse(&file).unwrap(), RAM_BASE, RAM_SIZE, None).unwrap();
        assert_eq!(placed.regions[1], region(RAM_BASE + PAGE, 1, Perms::Read));
        // The mapping in the report keeps the segment's own size.
        assert_eq!(
            placed.mappings[1],
            mapping("main.rodata", RAM_BASE + PAGE, RAM_BASE + PAGE + 1)
        );
    }

    #[test]
    fn a_segment_that_starts_inside_a_page_is_an_error() {
        // Divergence from Eden, which runs the program with the segment inaccessible.
        let page = u32::try_from(PAGE).unwrap();
        for offset in [page + 1, 2 * page - 1] {
            let mut file = three_segments();
            set_segment(&mut file, 1, offset, 1);
            assert_eq!(
                place(&Program::parse(&file).unwrap(), RAM_BASE, RAM_SIZE, None),
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
        // Divergence from Eden, where the later segment's permissions win.
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
                place(&Program::parse(&file).unwrap(), RAM_BASE, RAM_SIZE, None),
                Err(LoadError::OverlappingSegments { offset: shared }),
                "segment {index} at {offset:#x}, {size:#x} bytes"
            );
        }
        // Text ending exactly where rodata starts shares nothing.
        let mut file = three_segments();
        set_segment(&mut file, 0, 0, page);
        assert!(place(&Program::parse(&file).unwrap(), RAM_BASE, RAM_SIZE, None).is_ok());
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
        let placed = place(&program, RAM_BASE, RAM_SIZE, None).unwrap();
        let image_end = RAM_BASE + (file.len() as u64).next_multiple_of(PAGE);
        let bss = placed
            .regions
            .iter()
            .find(|r| r.start == image_end)
            .expect("a .bss region");
        assert_eq!(*bss, region(image_end, 1, Perms::ReadWrite));
        for segment in program.nro.segments() {
            let end = RAM_BASE + u64::from(segment.offset) + u64::from(segment.size);
            assert!(end.next_multiple_of(PAGE) <= image_end);
        }
    }

    #[test]
    fn a_program_that_exactly_fits_is_placed_and_one_page_less_is_an_error() {
        let file = programs::hello().unwrap();
        let program = Program::parse(&file).unwrap();
        // Two pages of image, the TLS block and the stack.
        let needed = 2 * PAGE + TLS_SIZE + STACK_SIZE;
        let placed = place(&program, RAM_BASE, needed, None).unwrap();
        // The TLS block ends exactly where the stack begins.
        assert_eq!(placed.tls + TLS_SIZE, RAM_BASE + needed - STACK_SIZE);
        assert_eq!(
            place(&program, RAM_BASE, needed - PAGE, None),
            Err(LoadError::DoesNotFit {
                needed,
                available: needed - PAGE
            })
        );
        // RAM smaller than the stack alone.
        assert_eq!(
            place(&program, RAM_BASE, PAGE, None),
            Err(LoadError::DoesNotFit {
                needed,
                available: PAGE
            })
        );
    }

    #[test]
    fn ram_must_be_page_aligned_and_inside_the_address_space() {
        let program = Program::parse(&programs::hello().unwrap()).unwrap();
        for (base, size) in [
            (RAM_BASE + 1, RAM_SIZE),
            (RAM_BASE, RAM_SIZE + 8),
            (u64::MAX - PAGE + 1, 2 * PAGE),
        ] {
            assert_eq!(
                place(&program, base, size, None),
                Err(LoadError::BadRam { base, size }),
                "{base:#x} {size:#x}"
            );
        }
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
        assert_eq!(
            place(&program, RAM_BASE, RAM_SIZE, None),
            Err(LoadError::BadBss)
        );
    }

    #[test]
    fn writing_copies_the_image_into_guest_memory_and_reports_a_fault() {
        let file = programs::hello().unwrap();
        let placed = place(&Program::parse(&file).unwrap(), RAM_BASE, RAM_SIZE, None).unwrap();
        let memory = FlatMemory::new(RAM_BASE, file.len());
        placed.write_into(&memory).unwrap();
        let mut read = vec![0; file.len()];
        memory.read(RAM_BASE, &mut read).unwrap();
        assert_eq!(read, file);
        let too_small = FlatMemory::new(RAM_BASE, file.len() - 1);
        assert_eq!(
            placed.write_into(&too_small),
            Err(Fault { address: RAM_BASE })
        );
    }
}
