//! How much guest RAM a run gives a program, and the check, made before a VM exists or a byte is
//! loaded, that the program fits at all.
//!
//! Design: `docs/specs/execution-model.md`, "Guest memory" ("Backing memory": host allocations
//! that stand for the console's RAM, split into Horizon's pools; "Sparse and lazy memory"), and
//! `docs/specs/kernel.md`, "Address space" (pool accounting: what an application may use is its
//! console's application pool). PROVISIONAL until the backing pools replace the one RAM region
//! (`crate::space`): a run's guest RAM stands for the program's application pool, as one region.
//!
//! # How much RAM a run has
//!
//! As the owner decided (decision log, 2026-11-19, "Owner rulings on the wave 44 and 45
//! questions and the first title bring-up", ruling 13; execution model, "Sparse and lazy
//! memory"): what the program may use, its console's application pool
//! (`starquake_kernel::memory::MemoryArrangement::application_pool_size`, through
//! `crate::image::memory_arrangement`: 3285 MiB for a Switch 1 program, 9809 MiB for a
//! Switch 2 one), less the charge of its system resource (`available`; zero when the charge is
//! more than the pool). [`GuestRam::for_program`] gives every program that fits all of it
//! (`size`), whatever the program needs: RAM is reserved whole and backed lazily, so a run costs
//! the host what the guest touches, not what it may map (the backend's host memory is an
//! anonymous mapping the host commits a page at a time, `starquake_cpu::hvf`, "Guest-physical
//! memory"; `crate::space` hands RAM out a page at a time, as pages are first touched). This
//! replaces a fixed 16 MiB allowance on top of what loading needs, which the first retail title
//! brought up used up about 0.4 s into its start-up.
//!
//! The process's system resource (`META` 0x14, `ProcessDescription::system_resource`) is counted
//! against the pool but given no guest RAM. Eden makes it a `KSecureSystemResource` for the
//! kernel's own structures for the process, reserved from the same `PhysicalMemoryMax` limit as
//! the code and the stack, and so does the kernel here
//! (`starquake_kernel::memory::AddressSpace::with_system_resource`, which charges
//! `SystemResource::required_secure_memory_size`: its size, or nothing in the applet pool). So
//! the charge comes off the pool, both for the fit check and for the RAM made. Nothing is
//! allocated from the resource: the translation tables live in the backend's own table pool,
//! outside guest RAM (`starquake_cpu::hvf`, "Guest-physical memory"; "The table pool", below).
//! GPU-only memory is never guest memory (execution model, "Sparse and lazy memory").
//!
//! The backend rounds the size up to its own pages (`starquake_cpu::hvf::RamConfig::at_least`,
//! 16 KiB), which never passes the pool: both pool sizes are whole 16 KiB pages
//! (`both_maximums_are_whole_host_pages`). It can pass `available` by less than 16 KiB only
//! for a charge that is not whole 16 KiB pages, which no metadata Horizon's loader takes has
//! (its sizes are whole 2 MiB blocks, `crate::image::MEMORY_BLOCK_UNIT_SIZE`).
//!
//! # What is counted, and the fit check
//!
//! [`GuestRam::for_program`] adds up what loading the program and starting its process back
//! with RAM (`needed`):
//!
//! | Part | Bytes | Where it comes from |
//! |---|---|---|
//! | the program | `Layout::extent`: the counter trampolines and every module with its `.bss`, from the first page to the last | `crate::image`, "Placement" |
//! | the main thread's stack | the description's stack size, rounded up to a page | `starquake_kernel::memory::AddressSpace::create_main_thread_stack` (Eden's `KProcess::Run`) |
//! | the main thread's thread-local page | [`THREAD_LOCAL_RAM`], one page | `starquake_kernel::process::run` takes the thread's region from a new page (`ThreadLocalPages::reserve`) and writes it |
//!
//! A program whose `needed` is above `available` is [`Error::OutOfRam`], before a VM is made.
//! A program that fits has RAM for every page loading writes (the trampolines and the modules'
//! bytes, all inside the counted extent), so loading it does not run out of RAM; a guest that
//! touches more than its RAM once it runs (more pages than the pool, counting the ones loading
//! wrote) still ends as an emulator error (`crate::space`).
//!
//! # The table pool
//!
//! Guest RAM is not the only memory a program's pages need: the backend maps it through
//! translation tables taken from its own table pool (`starquake_cpu::hvf`, `TABLE_POOL_SIZE`),
//! which the fit check does not count. That pool holds every table one address space can have
//! at once (`starquake_memory::stage1::MAX_TABLE_PAGES`, a little over 1 GiB, backed lazily
//! too), so it is sized with the address space, not with how scattered a program's pages are.
//! A map is refused for want of a table only in the window that constant's docs name: while
//! tables an unmapping freed wait for their flush, and the tables in use, the waiting ones and
//! the ones the map needs come to more than every table of the space. Loading unmaps nothing,
//! so it cannot run out of tables.
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
//! `a_system_resource_that_fits_with_no_byte_to_spare_fits_and_one_page_more_does_not`.

use starquake_kernel::process::ProcessDescription;
use starquake_memory::PAGE_SIZE;

use super::Error;
use crate::image::{Layout, LoadError, memory_arrangement};

/// Guest RAM counted for the main thread's thread-local page: one page, which the kernel maps
/// and writes when it starts the process (see the module docs).
pub const THREAD_LOCAL_RAM: u64 = PAGE_SIZE;

/// The guest RAM a run gives a program (see the module docs).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct GuestRam {
    /// What loading the program and starting its process back: a whole number of pages, at
    /// most `size`.
    pub(crate) needed: u64,
    /// The RAM to make: the program's application pool less its system resource's charge,
    /// whatever `needed` is.
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
        // The whole of what the program may use, reserved and backed lazily (module docs).
        Ok(Self {
            needed,
            size: available,
        })
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
                size: nro_pool()
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
        // The Switch 2 program gets its own console's pool, all of it.
        assert_eq!(
            ram(&switch2),
            Ok(GuestRam {
                needed,
                size: twelve
            })
        );
        assert_ne!(four, twelve);
    }

    #[test]
    fn the_ram_is_the_whole_pool_whatever_the_program_needs() {
        // The pool as the console's memory arrangement gives it, written out: 3285 MiB for the
        // console an NRO runs as (a Switch 1), from
        // `starquake_kernel::memory::MemoryArrangement::application_pool_size`.
        let pool = nro_pool();
        assert_eq!(pool, 3285 << 20);
        let base = ram(&nro(0)).unwrap().needed;
        // `.bss` that leaves the pool `left` bytes past `needed`: from far more room than the
        // 16 MiB allowance this replaced to none at all, the RAM made is the pool.
        let with_room = |left: u64| ram(&nro(u32::try_from(pool - base - left).unwrap())).unwrap();
        for left in [pool - base, 64 << 20, 16 << 20, PAGE_SIZE, 0] {
            let ram = with_room(left);
            assert_eq!(ram.needed, pool - left, "{left:#x}");
            assert_eq!(ram.size, pool, "{left:#x}");
        }
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
    fn a_system_resource_that_fits_with_no_byte_to_spare_fits_and_one_page_more_does_not() {
        // The program's own pages and a resource that together are the pool, exactly. This pins
        // the fit check only: the charge is generally not whole `SECURE_ALIGNMENT`s, so the
        // kernel would refuse the resource (`InvalidSize`) when the address space is made, and
        // this description would not load.
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
    fn the_ram_is_the_pool_less_the_resources_charge() {
        // Resources of one 2 MiB block and of sixteen: the RAM made is the pool less each
        // charge, with the same `needed`.
        let program = nro(0);
        let layout = place(&program, IMAGE_BASE, Some(RESCALE)).unwrap();
        let mut description = program.description().clone();
        let needed = GuestRam::for_program(&layout, &description).unwrap().needed;
        let pool = nro_pool();
        for blocks in [1, 16] {
            description.system_resource_size = blocks * MEMORY_BLOCK_UNIT_SIZE;
            let charge = description.system_resource().required_secure_memory_size();
            assert_eq!(charge, u64::from(blocks * MEMORY_BLOCK_UNIT_SIZE));
            assert_eq!(
                GuestRam::for_program(&layout, &description),
                Ok(GuestRam {
                    needed,
                    size: pool - charge
                }),
                "{blocks}"
            );
        }
        // In the applet pool the resource charges nothing, and the RAM is the whole pool.
        description.pool = Pool::Applet;
        assert_eq!(
            GuestRam::for_program(&layout, &description),
            Ok(GuestRam { needed, size: pool })
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

    /// The guest RAM a run makes for `program`, as `crate::system::host::run` makes it, with
    /// an address space over it; then touched far past the 16 MiB the old allowance gave,
    /// filled to its last byte, and measured by the backing's own counter.
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    fn touch_and_fill(program: &Program, pool: u64) {
        use std::sync::PoisonError;

        use starquake_cpu::hvf::{HvfBackend, PhysicalMemory, RamConfig};
        use starquake_memory::{GuestMemory, Perms};

        use crate::space::{AddressSpace, SpaceError};
        use crate::system::host::ONE_VM;

        let guest = ram(program).unwrap();
        assert_eq!(guest.size, pool);
        let _one_vm = ONE_VM.lock().unwrap_or_else(PoisonError::into_inner);
        let backend =
            HvfBackend::new(RamConfig::at_least(guest.size).unwrap()).expect("create the VM");
        let physical = backend.physical();
        let ram = backend.ram();
        assert_eq!(ram.size(), pool);
        // The whole pool, reserved: nothing of it costs the host yet.
        assert_eq!(physical.ram_resident_bytes(), Ok(0));
        let space =
            AddressSpace::<PhysicalMemory>::new(physical.clone(), ram, backend.table_pool())
                .unwrap();
        // A 256 MiB mapping, a doubleword written every 16 MiB of it: sixteen pages backed, the
        // first sixteen pages of RAM in the order they were touched, four host pages of 16 KiB.
        let (va, size, step): (u64, u64, u64) = (0x10_0000_0000, 256 << 20, 16 << 20);
        let every = || (va..va + size).step_by(usize::try_from(step).unwrap());
        space.map(va, size, Perms::ReadWrite).unwrap();
        for at in every() {
            space.write(at, &at.to_le_bytes()).unwrap();
        }
        // The host's page (16 KiB on Apple Silicon), the smallest RAM a `RamConfig` holds.
        let host_page = RamConfig::at_least(1).unwrap().size();
        let touched = size / step * PAGE_SIZE;
        assert_eq!(physical.ram_resident_bytes(), Ok(touched.div_ceil(host_page) * host_page));
        for at in every() {
            let mut back = [0; 8];
            space.read(at, &mut back).unwrap();
            assert_eq!(u64::from_le_bytes(back), at);
        }
        // The pool's edge: the rest of RAM is one run that ends exactly at the pool's size;
        // a page more is refused with what is left, and handing out the rest costs nothing.
        let rest = ram.size() - touched;
        assert_eq!(
            space.allocate_shared(rest + PAGE_SIZE),
            Err(SpaceError::OutOfRam {
                size: rest + PAGE_SIZE,
                available: rest
            })
        );
        let last = space.allocate_shared(rest).unwrap();
        assert_eq!((last.base(), last.end()), (ram.base() + touched, ram.end()));
        assert_eq!(physical.ram_resident_bytes(), Ok(touched.div_ceil(host_page) * host_page));
        assert_eq!(
            space.allocate_shared(PAGE_SIZE),
            Err(SpaceError::OutOfRam {
                size: PAGE_SIZE,
                available: 0
            })
        );
        // Its last byte is guest RAM's last byte: written, it costs one host page more.
        physical.write(ram.end() - 1, &[0xee]).unwrap();
        assert_eq!(
            physical.ram_resident_bytes(),
            Ok(touched.div_ceil(host_page) * host_page + host_page)
        );
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn a_runs_guest_ram_is_its_whole_pool_and_costs_the_host_only_what_is_touched() {
        if std::env::var_os("CI").is_some() {
            return;
        }
        // An NRO, a Switch 1 program; and an ExeFS built for a Switch 2.
        touch_and_fill(&nro(0), nro_pool());
        let switch2 = exefs(|e| e.npdm.acid.layout = AcidLayout::Switch2);
        assert_eq!(switch2.description().console, Console::Switch2);
        touch_and_fill(
            &switch2,
            MemoryArrangement::TwelveGb.application_pool_size(),
        );
    }
}
