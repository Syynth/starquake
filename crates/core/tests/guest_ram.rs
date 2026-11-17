//! Guest RAM sized from the program a run loads, at the exact edge, on both ways a program
//! arrives: an NRO ([`System::new`]) and a NUP title ([`System::from_nup`]). A program that
//! fits its console's application pool with no byte to spare loads and runs; one byte more is
//! [`Error::OutOfRam`], found before the VM is made (`crates/core/src/system/ram.rs`).
//!
//! The programs are synthetic: an NRO of one instruction, which exits, the same with a counter
//! read before it (so that the program has counter trampolines), and a container built
//! around the homebrew hello fixture (`tests/homebrew/hello/`) with its `.bss` made larger,
//! whose libnx start-up stops at `Break` once the kernel refuses it a heap in the full pool.
//!
//! These need a Mac with Apple Silicon. They skip themselves only when the `CI` environment
//! variable is set (hosted runners can't create VMs); anywhere else, failing to create a VM fails
//! the test. Each fitting run makes a VM whose guest RAM is a whole Switch 1 application pool
//! (3285 MiB).

#![cfg(all(target_os = "macos", target_arch = "aarch64"))]

use std::io::Cursor;

use starquake_core::crash::Exception;
use starquake_core::system::THREAD_LOCAL_RAM;
use starquake_core::{Error, ExitReason, RunOptions, RunReport, System, exit_code};
use starquake_kernel::memory::MemoryArrangement;
use starquake_kernel::process::{Console, ProcessDescription};
use starquake_kernel::result::KernelError;
use starquake_kernel::svc::SvcId;
use starquake_kernel::svc::exception::{BreakReason, RESULT_ARGUMENT_SIZE};
use starquake_loader::exefs::{ExeFs, MAIN_MODULE, NPDM_FILE};
use starquake_loader::npdm::AcidLayout;
use starquake_loader::nso::Nso;
use starquake_loader::nup::PackageKind;
use starquake_loader::nup::build::{self, Content, EXEFS_DIR_NAME, PackageSpec, Tree};
use starquake_memory::PAGE_SIZE;
use starquake_testkit::a64::{self, SysReg};
use starquake_testkit::nro::NroBuilder;
use starquake_trace::{Dump, EventKind};

#[cfg(test)]
mod common;

use common::{GENEROUS, run};

const NSO: &[u8] = include_bytes!("../../../tests/homebrew/hello/hello.nso");
const NPDM: &[u8] = include_bytes!("../../../tests/homebrew/hello/main.npdm");

/// A synthetic title id.
const TITLE_ID: u64 = 0x0100_0000_0000_5000;

/// Offset in an NSO's header of the `.bss` size: the last word of the third (data) segment
/// header, which starts at 0x30 (`NSOSegmentHeader`, 0x10 bytes each from 0x10, in
/// `eden/src/core/loader/nso.h`; the loader's `nso` module docs: "The third segment header's
/// last word is the bss size"). The loader keeps its offsets private; the tests check this one
/// by parsing the patched file.
const NSO_BSS_SIZE_OFFSET: usize = 0x3C;

/// Shared setup. In a `cfg(test)` module so it may unwrap like the tests do.
#[cfg(test)]
mod support {
    use super::*;

    /// Runs `system` (a title), or returns `None` on hosted CI (see the module docs).
    pub(crate) fn run_title(system: &System) -> Option<RunReport> {
        if std::env::var_os("CI").is_some() {
            return None;
        }
        Some(system.run(&RunOptions {
            timeout: Some(GENEROUS),
            command_line: Vec::new(),
            sd_card: None,
        }))
    }

    /// The bytes of the mapping called `name` in a run's memory map, zero if there is none.
    pub(crate) fn mapped(report: &RunReport, name: &str) -> u64 {
        report
            .memory_map
            .iter()
            .find(|mapping| mapping.name == name)
            .map_or(0, |mapping| mapping.end - mapping.start)
    }

    /// The run ended with the program's `ExitProcess`.
    pub(crate) fn assert_exited(report: &RunReport) {
        assert_eq!(
            report.exit,
            ExitReason::ProcessExited,
            "{:?}\n{}",
            report
                .crash()
                .map(starquake_core::crash::CrashReport::to_text),
            report.trace
        );
    }

    /// The run was refused for want of guest RAM before anything ran, needing `needed` bytes of
    /// the `available` an application may use.
    pub(crate) fn assert_out_of_ram(report: &RunReport, needed: u64, available: u64) {
        let error = Error::OutOfRam { needed, available };
        assert_eq!(report.exit, ExitReason::EmulatorError(error.to_string()));
        assert_eq!(report.exit_code(), exit_code::EMULATOR_ERROR);
        assert_eq!(report.trace, "");
        assert!(report.memory_map.is_empty(), "{:?}", report.memory_map);
        assert_eq!(
            error.to_string(),
            format!(
                "the program needs {needed:#x} bytes of guest RAM, more than the {available:#x} \
                 bytes an application may use on its console"
            )
        );
    }

    /// The hello fixture's NSO with `bss` bytes of `.bss`.
    pub(crate) fn nso_with_bss(bss: u32) -> Vec<u8> {
        let mut nso = NSO.to_vec();
        nso[NSO_BSS_SIZE_OFFSET..NSO_BSS_SIZE_OFFSET + 4].copy_from_slice(&bss.to_le_bytes());
        assert_eq!(Nso::parse(&nso).unwrap().bss_size, bss);
        nso
    }

    /// A NUP title whose base holds the fixture's metadata and `main`.
    pub(crate) fn title(main: &[u8]) -> System {
        let tree = [(NPDM_FILE, NPDM), (MAIN_MODULE, main)].iter().fold(
            Tree::new(),
            |tree, (name, bytes)| {
                tree.with_file(&format!("{EXEFS_DIR_NAME}/{name}"), Content::raw(*bytes))
                    .unwrap()
            },
        );
        let built =
            build::build(&[PackageSpec::new(PackageKind::Base, TITLE_ID, 0, tree)]).unwrap();
        System::from_nup(Cursor::new(built.bytes)).unwrap()
    }
}

use support::{assert_exited, assert_out_of_ram, mapped, nso_with_bss, run_title, title};

#[test]
fn an_nro_that_fits_its_pool_with_no_byte_to_spare_runs_and_one_byte_more_is_out_of_ram() {
    let exit = a64::svc(u32::from(SvcId::ExitProcess.number())).unwrap();
    let file = |bss: u32| NroBuilder::new(&[exit]).bss_size(bss).build().unwrap();
    // An NRO runs as a Switch 1 program, with Eden's defaults for a program with no NPDM.
    let pool = MemoryArrangement::FourGb.application_pool_size();
    let stack = ProcessDescription::homebrew(Console::Switch1).main_thread_stack_size;
    let Some(small) = run(&file(0), GENEROUS) else {
        return;
    };
    assert_exited(&small);
    // It reads no counter, so it has no trampolines: what it needs is its image, its stack and
    // its thread-local page.
    assert_eq!(mapped(&small, "trampolines"), 0);
    let image = u64::try_from(file(0).len()).unwrap();
    let needed = image.next_multiple_of(PAGE_SIZE) + stack + THREAD_LOCAL_RAM;
    // `.bss` is whole pages after the image: this much of it leaves nothing to spare.
    let bss = u32::try_from(pool - needed).unwrap();
    let fits = run(&file(bss), GENEROUS).unwrap();
    assert_exited(&fits);
    assert_eq!(mapped(&fits, "main.bss"), u64::from(bss));
    // One byte more of `.bss` is one page more.
    let over = run(&file(bss + 1), GENEROUS).unwrap();
    assert_out_of_ram(&over, pool + PAGE_SIZE, pool);
}

#[test]
fn an_nro_with_a_counter_read_that_fits_its_pool_with_no_byte_to_spare_runs_and_one_byte_more_is_out_of_ram()
 {
    // One counter read, which the patcher rewrites to branch to a trampoline below the image,
    // then `ExitProcess`.
    let words = [
        a64::mrs(0, SysReg::CntVctEl0).unwrap(),
        a64::svc(u32::from(SvcId::ExitProcess.number())).unwrap(),
    ];
    let file = |bss: u32| NroBuilder::new(&words).bss_size(bss).build().unwrap();
    let pool = MemoryArrangement::FourGb.application_pool_size();
    let stack = ProcessDescription::homebrew(Console::Switch1).main_thread_stack_size;
    let Some(small) = run(&file(0), GENEROUS) else {
        return;
    };
    assert_exited(&small);
    // The trampolines' pages are part of what the program needs; measured, since their code
    // depends on the host's counter rate.
    let trampolines = mapped(&small, "trampolines");
    assert_ne!(trampolines, 0);
    let image = u64::try_from(file(0).len()).unwrap();
    let needed = trampolines + image.next_multiple_of(PAGE_SIZE) + stack + THREAD_LOCAL_RAM;
    // `.bss` that leaves nothing to spare once the trampolines are counted.
    let bss = u32::try_from(pool - needed).unwrap();
    let fits = run(&file(bss), GENEROUS).unwrap();
    assert_exited(&fits);
    assert_eq!(mapped(&fits, "trampolines"), trampolines);
    assert_eq!(mapped(&fits, "main.bss"), u64::from(bss));
    // One byte more of `.bss` is one page more.
    let over = run(&file(bss + 1), GENEROUS).unwrap();
    assert_out_of_ram(&over, pool + PAGE_SIZE, pool);
}

#[test]
fn a_nup_title_that_fits_its_pool_with_no_byte_to_spare_runs_and_one_byte_more_is_out_of_ram() {
    let exefs = ExeFs::parse(&[(NPDM_FILE, NPDM), (MAIN_MODULE, NSO)]).unwrap();
    // The fixture is a Switch 1 program.
    assert_eq!(exefs.npdm.acid.layout, AcidLayout::Switch1);
    let pool = MemoryArrangement::FourGb.application_pool_size();
    let stack = u64::from(exefs.npdm.main_thread_stack_size).next_multiple_of(PAGE_SIZE);
    let code = &exefs.modules[0].code;
    let data_end = code.data.offset + code.data.size;

    // The title as it is, to measure its trampolines rather than assume them. The patcher
    // makes trampolines only for the counter reads it patches, and the fixture may have none,
    // in which case this is zero; this test then does not exercise trampolines, and
    // `an_nro_with_a_counter_read_that_fits_its_pool_with_no_byte_to_spare_runs_and_one_byte_more_is_out_of_ram`
    // is the one that checks they are counted.
    let Some(plain) = run_title(&title(NSO)) else {
        return;
    };
    assert_exited(&plain);
    assert_eq!(plain.guest_log.len(), 3);
    let trampolines = mapped(&plain, "trampolines");

    // The module's size is its data's end plus `.bss`, rounded up to a page: `.bss` that ends
    // the module exactly where the pool is full leaves nothing to spare.
    let module = pool - trampolines - stack - THREAD_LOCAL_RAM;
    let bss = u32::try_from(module - data_end).unwrap();
    let fits = run_title(&title(&nso_with_bss(bss))).unwrap();
    // It loads and runs. With its module and stack filling the pool, libnx's start-up then
    // asks for a heap and is told the limit is reached, as on a console whose pool is full,
    // and stops at `Break`: the guest's own end, not the emulator's. libnx breaks with a
    // panic and its own result in a four-byte buffer (`diagAbortWithResult`,
    // `libnx/nx/source/runtime/diag.c` and `runtime/init.c` @ b156b477):
    // `MAKERESULT(Module_Libnx, LibnxError_HeapAllocFailed)`: module 345 (0x159) and
    // description 15 (`libnx/nx/include/switch/result.h`), the description shifted left
    // 9 bits.
    let crash = fits
        .crash()
        .expect("libnx stops at Break when it has no heap");
    let heap_alloc_failed = 0x159 | (0xf << 9);
    // The buffer is the result on libnx's stack, its address in `x1`.
    let address = crash.registers.x[1];
    assert_eq!(
        crash.exception,
        Exception::SvcBreak {
            reason: BreakReason::Panic.raw(),
            reason_name: Some("panic"),
            address,
            size: RESULT_ARGUMENT_SIZE,
            code: Some(heap_alloc_failed),
        },
        "{}",
        crash.to_text()
    );
    assert_eq!(fits.system_calls.get("Break"), Some(&1));
    let set_heap = u32::from(SvcId::SetHeapSize.number());
    let heap_results: Vec<_> = Dump::from_bytes(&fits.events)
        .expect("a readable dump")
        .merged()
        .iter()
        .filter_map(|event| match event.kind {
            EventKind::Svc { id, result, .. } if id == set_heap => Some(result.0),
            _ => None,
        })
        .collect();
    assert_eq!(heap_results, [KernelError::LimitReached.code().raw()]);
    assert_eq!(mapped(&fits, "trampolines"), trampolines);
    assert_eq!(mapped(&fits, "main.bss"), u64::from(bss));
    // One byte more of `.bss` is one page more of module.
    let over = run_title(&title(&nso_with_bss(bss + 1))).unwrap();
    assert_out_of_ram(&over, pool + PAGE_SIZE, pool);
}
