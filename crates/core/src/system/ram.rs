//! How much guest RAM a run gives a program, and the check, made before a VM exists or a byte is
//! loaded, that the program fits at all.
//!
//! Design: `docs/specs/execution-model.md`, "Guest memory" ("Backing memory": host allocations
//! that stand for the console's RAM, split into Horizon's pools; "Sparse and lazy memory"), and
//! `docs/specs/kernel.md`, "Address space" (pool accounting: what an application may use is its
//! console's application pool). PROVISIONAL until the backing pools replace the one RAM region
//! (`crate::space`): a run's guest RAM stands for the part of the application pool the program
//! uses, sized from the program rather than the whole pool.
//!
//! # What is counted
//!
//! [`GuestRam::for_program`] adds up what loading the program and starting its process back
//! with RAM (`needed`), then gives it room to run:
//!
//! | Part | Bytes | Where it comes from |
//! |---|---|---|
//! | the program | `Layout::extent`: the counter trampolines and every module with its `.bss`, from the first page to the last | `crate::image`, "Placement" |
//! | the main thread's stack | the description's stack size, rounded up to a page | `starquake_kernel::memory::AddressSpace::create_main_thread_stack` (Eden's `KProcess::Run`) |
//! | the main thread's thread-local page | [`THREAD_LOCAL_RAM`], one page | `starquake_kernel::process::run` takes the thread's region from a new page (`ThreadLocalPages::reserve`) and writes it |
//! | working RAM | [`WORKING_RAM`], or what the maximum leaves of it | a guess: see the constant |
//!
//! The process's system resource (`META` 0x14, `ProcessDescription::system_resource`) is counted
//! against the pool but given no guest RAM. Eden makes it a `KSecureSystemResource` for the
//! kernel's own structures for the process, reserved from the same `PhysicalMemoryMax` limit as
//! the code and the stack, and so does the kernel here
//! (`starquake_kernel::memory::AddressSpace::with_system_resource`, which charges
//! `SystemResource::required_secure_memory_size`: its size, or nothing in the applet pool). So
//! the fit check takes that charge off the pool before comparing (`available` below), and
//! working RAM is cut to what the pool leaves after it. Nothing is allocated from the
//! resource: the translation tables live in the backend's own table pool, outside guest RAM
//! (`starquake_cpu::hvf`, "Guest-physical memory"; the limit that pool sets is below). GPU-only
//! memory is never guest memory (execution model, "Sparse and lazy memory").
//!
//! # The maximum and the fit check
//!
//! The maximum is the application pool of the console the program is built for
//! (`starquake_kernel::memory::MemoryArrangement::application_pool_size`, through
//! `crate::image::memory_arrangement`): 3285 MiB for a Switch 1 program, 9809 MiB for a
//! Switch 2 one, less the system resource's charge (`available`; zero when the charge is more
//! than the pool). A program whose `needed` is above it is [`Error::OutOfRam`], before a VM is
//! made. A program that fits has RAM for every page loading writes (the trampolines and the
//! modules' bytes, all inside the counted extent), so loading it does not run out of RAM (the
//! tables that map them are another matter: "The table pool", below); a
//! guest that touches more than its RAM once it runs still ends as an emulator error
//! (`crate::space`).
//!
//! A program that fits gets `needed` plus [`WORKING_RAM`], or the maximum if that is less: a
//! program that fits with no byte to spare loads, with no working RAM. The backend rounds the
//! size up to its own pages (`starquake_cpu::hvf::RamConfig::at_least`, 16 KiB), which never
//! passes the pool: both pool sizes are whole 16 KiB pages
//! (`both_maximums_are_whole_host_pages`). It can pass `available` by less than 16 KiB only
//! for a charge that is not whole 16 KiB pages, which no metadata Horizon's loader takes has
//! (its sizes are whole 2 MiB blocks, `crate::image::MEMORY_BLOCK_UNIT_SIZE`).
//!
//! # The table pool: a second limit, not counted
//!
//! Guest RAM is not the only memory a program's pages need. The backend maps guest RAM through
//! translation tables taken from its own table pool (`starquake_cpu::hvf`, `TABLE_POOL_SIZE`),
//! whose size is fixed (4 MiB, 1024 table pages; by its own doc, enough to map about 2 GiB of
//! scattered 2 MiB windows) and is not sized with guest RAM, which can now be up to 3285 or
//! 9809 MiB. The fit check does not count table pages. So a program that fits here, but whose
//! loaded bytes or the pages it touches need more tables than the pool holds, fails when the
//! tables run out, on the mapper's error rather than [`Error::OutOfRam`]: while loading, a
//! module's bytes are written through the guest's mappings and that surfaces as a guest write
//! fault (`LoadError::Write`), the failure this check exists to rule out; once running, as an
//! emulator error. PROVISIONAL until the kernel sizes address spaces and the table pool with
//! them.
//!
//! # Divergence from Eden
//!
//! Eden reserves against the process's `PhysicalMemoryMax` limit the code
//! (`KProcess::Initialize`, the overload `LoadFromMetadata` calls: a
//! `KScopedResourceReservation` of `code_size`), the system resource
//! (`secure_resource->Initialize(kernel, system_resource_size, res_limit, m_memory_pool)` in
//! the same function, which reserves `PhysicalMemoryMax`, `secure_size` in
//! `KSecureSystemResource::Initialize`, `eden/src/core/hle/kernel/k_system_resource.cpp`), and
//! the main thread's stack (`KProcess::Run`), all in `eden/src/core/hle/kernel/k_process.cpp` @
//! cf654052. The check here differs both ways:
//!
//! - **Stricter.** It counts what Eden does not reserve: the thread-local page (Eden's comes
//!   from the kernel's page buffers, `KPageBuffer::Allocate` in
//!   `KThreadLocalPage::Initialize`, `k_thread_local_page.cpp`) and the counter trampolines
//!   (Eden has none). So a program whose code and stack fill the pool to within less than a
//!   page (and its trampolines) loads under Eden and is [`Error::OutOfRam`] here. Test:
//!   `code_and_stack_that_fill_the_pool_leave_no_room_for_the_thread_local_page`.
//!
//! The system resource is counted as Eden reserves it, so a program with a non-zero system
//! resource whose code and stack fill the pool is refused here as Eden refuses it
//! (`ResultLimitReached` there, [`Error::OutOfRam`] here, before a VM is made). Tests:
//! `a_system_resource_is_counted_so_a_title_that_eden_refuses_is_refused`,
//! `a_system_resource_that_fits_with_no_byte_to_spare_loads_and_one_page_more_does_not`.

use starquake_kernel::process::ProcessDescription;
use starquake_memory::PAGE_SIZE;

use super::Error;
use crate::image::{Layout, LoadError, memory_arrangement};

/// Guest RAM counted for the main thread's thread-local page: one page, which the kernel maps
/// and writes when it starts the process (see the module docs).
pub const THREAD_LOCAL_RAM: u64 = PAGE_SIZE;

/// Guest RAM a run gives a program beyond what loading and starting it back: room for what it
/// maps and touches once it runs (its heap, more threads' stacks and thread-local pages, memory
/// a service shares with it). The kernel specs reserve none of this up front: the heap is set
/// at run time (`SetHeapSize`), and a libnx program asks for gigabytes of it and touches a few
/// pages (`crate::space`).
///
/// Guess: 16 MiB, the fixed guest RAM every run had before RAM was sized from the program (the
/// backend's default), within which the homebrew tests run. A program that touches more ends
/// its run as an emulator error, naming the page that could not be backed. A trace of a
/// retail title's start-up would show what it touches.
pub const WORKING_RAM: u64 = 16 << 20;

/// The guest RAM a run gives a program (see the module docs).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct GuestRam {
    /// What loading the program and starting its process back: a whole number of pages, at
    /// most the maximum.
    pub(crate) needed: u64,
    /// The RAM to make: `needed` plus [`WORKING_RAM`], or the maximum if that is less.
    pub(crate) size: u64,
}

impl GuestRam {
    /// The RAM for a program laid out as `layout` whose process is `description`, or
    /// [`Error::OutOfRam`] when the program does not fit its console's application pool.
    /// [`LoadError::Overflow`] when the sizes do not add up in 64 bits.
    pub(crate) fn for_program(
        layout: &Layout,
        description: &ProcessDescription,
    ) -> Result<Self, Error> {
        let overflow = || Error::Load(LoadError::Overflow);
        let stack = description
            .main_thread_stack_size
            .checked_next_multiple_of(PAGE_SIZE)
            .ok_or_else(overflow)?;
        let needed = layout
            .extent()
            .checked_add(stack)
            .and_then(|sum| sum.checked_add(THREAD_LOCAL_RAM))
            .ok_or_else(overflow)?;
        // The system resource's charge comes off the pool, as the kernel reserves it from the
        // same count when it makes the address space.
        let charge = description.system_resource().required_secure_memory_size();
        let available = memory_arrangement(description.console)
            .application_pool_size()
            .saturating_sub(charge);
        if needed > available {
            return Err(Error::OutOfRam { needed, available });
        }
        // `needed` is at most `available`, so the room left cannot underflow.
        let size = needed + WORKING_RAM.min(available - needed);
        Ok(Self { needed, size })
    }
}

#[cfg(test)]
mod tests {
    use starquake_kernel::memory::{MemoryArrangement, Pool};
    use starquake_kernel::process::Console;
    use starquake_loader::exefs::{ExeFs, MAIN_MODULE, NPDM_FILE};
    use starquake_loader::npdm::AcidLayout;
    use starquake_loader::patch::CounterRescale;
    use starquake_testkit::nro::NroBuilder;
    use starquake_testkit::{a64, programs};

    use super::*;
    use crate::clock::NRO_CONSOLE;
    use crate::image::{IMAGE_BASE, MEMORY_BLOCK_UNIT_SIZE, Program, place};

    const NSO: &[u8] = include_bytes!("../../../../tests/homebrew/hello/hello.nso");
    const NPDM: &[u8] = include_bytes!("../../../../tests/homebrew/hello/main.npdm");

    /// A host counter rate for the patcher: what the trampolines are written for. Any non-zero
    /// rate will do; the sizing reads the layout the patcher made.
    const RESCALE: CounterRescale = CounterRescale {
        host_hz: 24_000_000,
        guest_hz: NRO_CONSOLE.counter_rate().hz(),
    };

    /// The application pool of the console an NRO runs as.
    fn nro_pool() -> u64 {
        memory_arrangement(NRO_CONSOLE).application_pool_size()
    }

    /// An NRO of one page of text and `bss` bytes of `.bss`.
    fn nro(bss: u32) -> Program {
        Program::parse(
            &NroBuilder::new(&[a64::nop()])
                .bss_size(bss)
                .build()
                .unwrap(),
        )
        .unwrap()
    }

    /// The hello fixture's ExeFS (the program a NUP title holds), with its metadata changed by
    /// `change`.
    fn exefs(change: impl FnOnce(&mut ExeFs)) -> Program {
        let mut exefs = ExeFs::parse(&[(NPDM_FILE, NPDM), (MAIN_MODULE, NSO)]).unwrap();
        change(&mut exefs);
        Program::from_exefs(exefs).unwrap()
    }

    fn ram(program: &Program) -> Result<GuestRam, Error> {
        let layout = place(program, IMAGE_BASE, Some(RESCALE)).unwrap();
        GuestRam::for_program(&layout, program.description())
    }

    /// What `program` needs, counted part by part from its layout's own fields: the
    /// trampolines' pages, each module's size, the stack rounded up to a page, and the
    /// thread-local page.
    fn needed_by_parts(program: &Program) -> u64 {
        let layout = place(program, IMAGE_BASE, Some(RESCALE)).unwrap();
        let trampolines = layout.trampolines.map_or(0, |region| region.size);
        let modules: u64 = layout.modules.iter().map(|module| module.size).sum();
        let stack = program
            .description()
            .main_thread_stack_size
            .next_multiple_of(PAGE_SIZE);
        trampolines + modules + stack + THREAD_LOCAL_RAM
    }

    #[test]
    fn needed_is_the_trampolines_the_modules_the_stack_and_one_thread_local_page() {
        // A program with counter reads, so it has trampolines below its module.
        let counter = Program::parse(&programs::counter(RESCALE.guest_hz).unwrap().nro).unwrap();
        let layout = place(&counter, IMAGE_BASE, Some(RESCALE)).unwrap();
        let trampolines = layout
            .trampolines
            .expect("the counter program has trampolines");
        assert_ne!(trampolines.size, 0);
        let needed = needed_by_parts(&counter);
        assert_eq!(
            ram(&counter),
            Ok(GuestRam {
                needed,
                size: needed + WORKING_RAM
            })
        );
        // Without trampolines the program's pages start at its module.
        let plain = nro(0);
        assert_eq!(
            place(&plain, IMAGE_BASE, Some(RESCALE))
                .unwrap()
                .trampolines,
            None
        );
        assert_eq!(ram(&plain).unwrap().needed, needed_by_parts(&plain));
    }

    #[test]
    fn an_exefs_counts_every_module_and_its_metadatas_stack_rounded_up_to_a_page() {
        // Two modules, and a stack one byte over three pages: four pages are counted.
        let files = [(NPDM_FILE, NPDM), (MAIN_MODULE, NSO), ("rtld", NSO)];
        let mut parsed = ExeFs::parse(&files).unwrap();
        parsed.npdm.main_thread_stack_size = u32::try_from(3 * PAGE_SIZE + 1).unwrap();
        let program = Program::from_exefs(parsed).unwrap();
        let layout = place(&program, IMAGE_BASE, Some(RESCALE)).unwrap();
        assert_eq!(layout.modules.len(), 2);
        let needed = ram(&program).unwrap().needed;
        assert_eq!(needed, needed_by_parts(&program));
        assert_eq!(needed, layout.extent() + 4 * PAGE_SIZE + THREAD_LOCAL_RAM);
    }

    #[test]
    fn an_nro_that_fits_its_pool_with_no_byte_to_spare_gets_all_of_it_and_one_byte_more_does_not_fit()
     {
        let pool = nro_pool();
        // `.bss` is whole pages after the image, so filling the rest of the pool with it leaves
        // nothing to spare.
        let rest = pool - ram(&nro(0)).unwrap().needed;
        let bss = u32::try_from(rest).unwrap();
        assert_eq!(
            ram(&nro(bss)),
            Ok(GuestRam {
                needed: pool,
                size: pool
            })
        );
        // One byte more of `.bss` is one page more of module.
        assert_eq!(
            ram(&nro(bss + 1)),
            Err(Error::OutOfRam {
                needed: pool + PAGE_SIZE,
                available: pool
            })
        );
    }

    #[test]
    fn an_exefs_that_fits_its_pool_with_no_byte_to_spare_loads_and_one_byte_more_does_not_fit() {
        let pool = memory_arrangement(Console::Switch1).application_pool_size();
        let no_stack = ram(&exefs(|e| e.npdm.main_thread_stack_size = 0))
            .unwrap()
            .needed;
        let stack = u32::try_from(pool - no_stack).unwrap();
        let fits = exefs(|e| e.npdm.main_thread_stack_size = stack);
        assert_eq!(fits.description().console, Console::Switch1);
        assert_eq!(
            ram(&fits),
            Ok(GuestRam {
                needed: pool,
                size: pool
            })
        );
        // One byte more of stack is a page more.
        assert_eq!(
            ram(&exefs(|e| e.npdm.main_thread_stack_size = stack + 1)),
            Err(Error::OutOfRam {
                needed: pool + PAGE_SIZE,
                available: pool
            })
        );
    }

    #[test]
    fn the_maximum_is_the_application_pool_of_the_programs_console() {
        // A stack as large as a Switch 1 application pool: too much for a Switch 1 program,
        // room to spare for the same program built for a Switch 2.
        let four = MemoryArrangement::FourGb.application_pool_size();
        let twelve = MemoryArrangement::TwelveGb.application_pool_size();
        let stack = u32::try_from(four).unwrap();
        let switch1 = exefs(|e| e.npdm.main_thread_stack_size = stack);
        let needed = needed_by_parts(&switch1);
        assert_eq!(
            ram(&switch1),
            Err(Error::OutOfRam {
                needed,
                available: four
            })
        );
        let switch2 = exefs(|e| {
            e.npdm.main_thread_stack_size = stack;
            e.npdm.acid.layout = AcidLayout::Switch2;
        });
        assert_eq!(switch2.description().console, Console::Switch2);
        assert_eq!(
            ram(&switch2),
            Ok(GuestRam {
                needed,
                size: needed + WORKING_RAM
            })
        );
        assert!(needed + WORKING_RAM < twelve);
    }

    #[test]
    fn working_ram_is_given_in_full_up_to_the_maximum_and_cut_at_it() {
        let pool = nro_pool();
        let base = ram(&nro(0)).unwrap().needed;
        // `.bss` that leaves the pool `left` bytes past `needed`.
        let with_room = |left: u64| ram(&nro(u32::try_from(pool - base - left).unwrap())).unwrap();
        // A page more room than working RAM: working RAM in full, a page short of the pool.
        let roomy = with_room(WORKING_RAM + PAGE_SIZE);
        assert_eq!(roomy.size, roomy.needed + WORKING_RAM);
        assert_eq!(roomy.size, pool - PAGE_SIZE);
        // Exactly working RAM's room: in full, and the whole pool.
        let exact = with_room(WORKING_RAM);
        assert_eq!(exact.size, exact.needed + WORKING_RAM);
        assert_eq!(exact.size, pool);
        // A page less: cut to the pool.
        let short = with_room(WORKING_RAM - PAGE_SIZE);
        assert_eq!(short.size, pool);
        assert_eq!(short.size - short.needed, WORKING_RAM - PAGE_SIZE);
    }

    #[test]
    fn code_and_stack_that_fill_the_pool_leave_no_room_for_the_thread_local_page() {
        // The kernel would reserve this program's code and stack: together they are exactly the
        // pool (the divergence in the module docs).
        let pool = memory_arrangement(Console::Switch1).application_pool_size();
        let program = exefs(|e| e.npdm.main_thread_stack_size = 0);
        let layout = place(&program, IMAGE_BASE, Some(RESCALE)).unwrap();
        let code: u64 = layout.modules.iter().map(|module| module.size).sum();
        let stack = u32::try_from(pool - code).unwrap();
        let program = exefs(|e| e.npdm.main_thread_stack_size = stack);
        let layout = place(&program, IMAGE_BASE, Some(RESCALE)).unwrap();
        let trampolines = layout.trampolines.map_or(0, |region| region.size);
        assert_eq!(
            ram(&program),
            Err(Error::OutOfRam {
                needed: pool + trampolines + THREAD_LOCAL_RAM,
                available: pool
            })
        );
    }

    #[test]
    fn a_system_resource_is_counted_so_a_title_that_eden_refuses_is_refused() {
        // A Switch 1 program that fits its pool with no byte to spare.
        let pool = memory_arrangement(Console::Switch1).application_pool_size();
        let no_stack = ram(&exefs(|e| e.npdm.main_thread_stack_size = 0))
            .unwrap()
            .needed;
        let stack = u32::try_from(pool - no_stack).unwrap();
        let program = exefs(|e| e.npdm.main_thread_stack_size = stack);
        assert_eq!(program.description().system_resource_size, 0);
        let layout = place(&program, IMAGE_BASE, Some(RESCALE)).unwrap();
        assert_eq!(
            GuestRam::for_program(&layout, program.description()),
            Ok(GuestRam {
                needed: pool,
                size: pool,
            })
        );
        // The same program with a system resource of one 2 MiB block: Eden, which reserves code,
        // stack and system resource from one limit, refuses it, and so does the fit check, with
        // the same `needed` and the pool less the charge.
        let mut description = program.description().clone();
        description.system_resource_size = MEMORY_BLOCK_UNIT_SIZE;
        let charge = u64::from(MEMORY_BLOCK_UNIT_SIZE);
        assert_eq!(
            GuestRam::for_program(&layout, &description),
            Err(Error::OutOfRam {
                needed: pool,
                available: pool - charge
            })
        );
        // In the applet pool the same resource charges nothing (`CalculateRequiredSecureMemorySize`),
        // and the program fits again.
        description.pool = Pool::Applet;
        assert_eq!(
            GuestRam::for_program(&layout, &description),
            Ok(GuestRam {
                needed: pool,
                size: pool,
            })
        );
    }

    #[test]
    fn a_system_resource_that_fits_with_no_byte_to_spare_loads_and_one_page_more_does_not() {
        // The program's own pages and a resource that together are the pool, exactly.
        let program = nro(0);
        let layout = place(&program, IMAGE_BASE, Some(RESCALE)).unwrap();
        let mut description = program.description().clone();
        let needed = GuestRam::for_program(&layout, &description).unwrap().needed;
        let pool = nro_pool();
        let charge = pool - needed;
        description.system_resource_size = u32::try_from(charge).unwrap();
        assert_eq!(description.system_resource().size(), charge);
        assert_eq!(
            GuestRam::for_program(&layout, &description),
            Ok(GuestRam {
                needed,
                size: needed
            })
        );
        // A page more of resource: a page short.
        description.system_resource_size += u32::try_from(PAGE_SIZE).unwrap();
        assert_eq!(
            GuestRam::for_program(&layout, &description),
            Err(Error::OutOfRam {
                needed,
                available: needed - PAGE_SIZE
            })
        );
        // A resource larger than the pool leaves nothing, rather than wrapping.
        description.system_resource_size = u32::MAX;
        assert!(description.system_resource().size() > pool);
        assert_eq!(
            GuestRam::for_program(&layout, &description),
            Err(Error::OutOfRam {
                needed,
                available: 0
            })
        );
    }

    #[test]
    fn working_ram_is_cut_to_what_the_resource_leaves() {
        // A resource that leaves the program exactly working RAM's room, then a page less.
        let program = nro(0);
        let layout = place(&program, IMAGE_BASE, Some(RESCALE)).unwrap();
        let mut description = program.description().clone();
        let needed = GuestRam::for_program(&layout, &description).unwrap().needed;
        let pool = nro_pool();
        let roomy = pool - needed - WORKING_RAM;
        description.system_resource_size = u32::try_from(roomy).unwrap();
        assert_eq!(
            GuestRam::for_program(&layout, &description).unwrap().size,
            needed + WORKING_RAM
        );
        description.system_resource_size += u32::try_from(PAGE_SIZE).unwrap();
        assert_eq!(
            GuestRam::for_program(&layout, &description).unwrap().size,
            needed + WORKING_RAM - PAGE_SIZE
        );
    }

    #[test]
    fn sizes_that_overflow_are_an_overflow_not_a_shortage() {
        let program = nro(0);
        let layout = place(&program, IMAGE_BASE, Some(RESCALE)).unwrap();
        let mut description = program.description().clone();
        // The stack cannot be rounded up to a page.
        description.main_thread_stack_size = u64::MAX;
        assert_eq!(
            GuestRam::for_program(&layout, &description),
            Err(Error::Load(LoadError::Overflow))
        );
        // The stack rounds up, and adding the program's pages to it overflows.
        description.main_thread_stack_size = u64::MAX - (PAGE_SIZE - 1);
        assert_eq!(
            GuestRam::for_program(&layout, &description),
            Err(Error::Load(LoadError::Overflow))
        );
        // The stack and the program's pages add up, and the thread-local page overflows.
        description.main_thread_stack_size = u64::MAX - (PAGE_SIZE - 1) - layout.extent();
        assert_eq!(
            GuestRam::for_program(&layout, &description),
            Err(Error::Load(LoadError::Overflow))
        );
        // A page less of stack adds up, and is a shortage.
        description.main_thread_stack_size -= PAGE_SIZE;
        assert!(matches!(
            GuestRam::for_program(&layout, &description),
            Err(Error::OutOfRam { .. })
        ));
    }

    #[test]
    fn running_out_says_how_much_was_needed_and_how_much_there_is() {
        assert_eq!(
            Error::OutOfRam {
                needed: 0xcd50_1000,
                available: 0xcd50_0000
            }
            .to_string(),
            "the program needs 0xcd501000 bytes of guest RAM, more than the 0xcd500000 bytes an \
             application may use on its console"
        );
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn both_maximums_are_whole_host_pages() {
        use starquake_cpu::hvf::RamConfig;
        for arrangement in [MemoryArrangement::FourGb, MemoryArrangement::TwelveGb] {
            let pool = arrangement.application_pool_size();
            assert_eq!(RamConfig::at_least(pool).unwrap().size(), pool);
        }
    }
}
