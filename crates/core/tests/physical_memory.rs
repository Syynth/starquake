//! `MapPhysicalMemory` and `UnmapPhysicalMemory` from synthetic programs run natively under the
//! hypervisor backend: the memory a program maps in its alias region is memory the vCPU can read
//! and write (fresh pages read zero), exactly the range asked for, and gone once unmapped. A
//! program with no system resource is refused, as Horizon refuses it.
//!
//! The programs are NROs built with the testkit, run as the `main` module of an ExeFS whose
//! metadata is the homebrew fixture's (`tests/homebrew/hello/main.npdm`) with its system
//! resource size set: an NRO alone runs with Eden's defaults for a program with no NPDM, which
//! have no system resource, and both calls refuse a process without one.
//!
//! These need a Mac with Apple Silicon. They skip themselves only when the `CI` environment
//! variable is set (hosted runners can't create VMs); anywhere else, failing to create a VM fails
//! the test. The kernel's side of both calls is tested on every host in the kernel crate
//! (`memory/space/physical_tests.rs`, `svc/physical_memory/tests.rs`).

#![cfg(all(target_os = "macos", target_arch = "aarch64"))]

use starquake_core::crash::Exception;
use starquake_core::{ExitReason, RunOptions, RunReport, System};
use starquake_kernel::handle::Handle;
use starquake_kernel::result::KernelError;
use starquake_kernel::svc::SvcId;
use starquake_kernel::svc::info::InfoType;
use starquake_loader::codeset::CodeSet;
use starquake_loader::exefs::{ExeFs, ExeFsModule, MAIN_MODULE, NPDM_FILE};
use starquake_memory::PAGE_SIZE;
use starquake_testkit::a64::{self, Assembler};
use starquake_testkit::nro::NroBuilder;
use starquake_trace::{Dump, EventKind};

#[cfg(test)]
mod common;

use common::GENEROUS;

const NSO: &[u8] = include_bytes!("../../../tests/homebrew/hello/hello.nso");
const NPDM: &[u8] = include_bytes!("../../../tests/homebrew/hello/main.npdm");

/// Shared setup. In a `cfg(test)` module so it may unwrap like the tests do.
#[cfg(test)]
mod support {
    use super::*;

    /// How many bytes the programs map: two pages.
    pub(crate) const SIZE: u64 = 2 * PAGE_SIZE;
    /// The value the programs store and compare.
    pub(crate) const VALUE: u32 = 0x1234;
    /// A system resource of one 2 MiB block: the smallest a program's metadata may ask for
    /// (Horizon's loader refuses a size that is not whole blocks).
    pub(crate) fn resource() -> u32 {
        starquake_core::MEMORY_BLOCK_UNIT_SIZE
    }

    /// What a program does once it has mapped the memory and used it.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub(crate) enum Then {
        /// Unmaps it and exits.
        UnmapAndExit,
        /// Stores to the first word past the mapped range.
        StorePastTheEnd,
        /// Unmaps it and loads from its first word.
        LoadAfterUnmapping,
    }

    fn svc(id: SvcId) -> u32 {
        a64::svc(u32::from(id.number())).unwrap()
    }

    /// The 16-bit half of `value` at `shift`.
    fn halfword(value: u64, shift: u32) -> u32 {
        u32::try_from((value >> shift) & u64::from(a64::MAX_IMM16)).unwrap()
    }

    /// A program that asks `GetInfo` where its alias region starts, maps [`SIZE`] bytes there
    /// with `MapPhysicalMemory`, checks that the second page reads zero, stores [`VALUE`] to the
    /// first and last words and reads it back from each, and then does `then`. A call that
    /// fails, and a value that does not read back, ends at `brk #1`.
    ///
    /// ```text
    ///       movz x1, #AliasRegionAddress
    ///       movz/movk x2, #CURRENT_PROCESS
    ///       movz x3, #0
    ///       svc  #0x29             ; GetInfo: x1 = the alias region's start
    ///       cbnz x0, fail
    ///       mov  x9, x1
    ///       mov  x0, x9
    ///       movz x1, #SIZE
    ///       svc  #0x2c             ; MapPhysicalMemory
    ///       cbnz x0, fail
    ///       ldr  x3, [x9, #0x1000] ; a fresh page reads zero
    ///       cbnz x3, fail
    ///       movz x2, #VALUE
    ///       str  x2, [x9]
    ///       ldr  x3, [x9]
    ///       sub  x4, x2, x3
    ///       cbnz x4, fail
    ///       str  x2, [x9, #SIZE - 8]
    ///       ldr  x3, [x9, #SIZE - 8]
    ///       sub  x4, x2, x3
    ///       cbnz x4, fail
    ///       ...                    ; then
    ///       svc  #0x7
    /// fail: brk  #1
    /// ```
    pub(crate) fn uses_physical_memory(then: Then) -> Vec<u8> {
        let size = u32::try_from(SIZE).unwrap();
        let last_word = size - a64::MEM_OFFSET_ALIGN;
        let process = u64::from(Handle::CURRENT_PROCESS.raw());
        let mut asm = Assembler::new();
        let fail = asm.label();
        asm.emit(a64::movz(1, InfoType::AliasRegionAddress.number(), 0).unwrap());
        asm.emit(a64::movz(2, halfword(process, 0), 0).unwrap());
        asm.emit(a64::movk(2, halfword(process, 16), 16).unwrap());
        asm.emit(a64::movz(3, 0, 0).unwrap());
        asm.emit(svc(SvcId::GetInfo));
        asm.cbnz(0, fail).unwrap();
        asm.emit(a64::mov(9, 1).unwrap());
        let unmap = |asm: &mut Assembler| {
            asm.emit(a64::mov(0, 9).unwrap());
            asm.emit(a64::movz(1, size, 0).unwrap());
            asm.emit(svc(SvcId::UnmapPhysicalMemory));
            asm.cbnz(0, fail).unwrap();
        };
        asm.emit(a64::mov(0, 9).unwrap());
        asm.emit(a64::movz(1, size, 0).unwrap());
        asm.emit(svc(SvcId::MapPhysicalMemory));
        asm.cbnz(0, fail).unwrap();
        asm.emit(a64::ldr(3, 9, u32::try_from(PAGE_SIZE).unwrap()).unwrap());
        asm.cbnz(3, fail).unwrap();
        asm.emit(a64::movz(2, VALUE, 0).unwrap());
        for offset in [0, last_word] {
            asm.emit(a64::str(2, 9, offset).unwrap());
            asm.emit(a64::ldr(3, 9, offset).unwrap());
            asm.emit(a64::sub(4, 2, 3).unwrap());
            asm.cbnz(4, fail).unwrap();
        }
        match then {
            Then::UnmapAndExit => unmap(&mut asm),
            Then::StorePastTheEnd => asm.emit(a64::str(2, 9, size).unwrap()),
            Then::LoadAfterUnmapping => {
                unmap(&mut asm);
                asm.emit(a64::ldr(3, 9, 0).unwrap());
            }
        }
        asm.emit(svc(SvcId::ExitProcess));
        asm.bind(fail).unwrap();
        asm.emit(a64::brk(1).unwrap());
        NroBuilder::new(&asm.finish().unwrap()).build().unwrap()
    }

    /// Runs `nro` as the `main` module of the fixture's ExeFS with a system resource of
    /// `resource` bytes, or returns `None` on hosted CI (see the module docs).
    pub(crate) fn run(nro: &[u8], resource: u32) -> Option<RunReport> {
        if std::env::var_os("CI").is_some() {
            return None;
        }
        let mut exefs = ExeFs::parse(&[(NPDM_FILE, NPDM), (MAIN_MODULE, NSO)]).unwrap();
        exefs.npdm.system_resource_size = resource;
        exefs.modules = vec![ExeFsModule {
            name: MAIN_MODULE,
            code: CodeSet::parse_nro(nro).unwrap(),
        }];
        let system = System::from_exefs(exefs).unwrap();
        Some(system.run(&RunOptions {
            timeout: Some(GENEROUS),
            command_line: Vec::new(),
            sd_card: None,
            romfs: None,
        }))
    }

    /// The results of the run's calls to `id`, in order.
    pub(crate) fn results(report: &RunReport, id: SvcId) -> Vec<u32> {
        let number = u32::from(id.number());
        Dump::from_bytes(&report.events)
            .expect("a readable dump")
            .merged()
            .iter()
            .filter_map(|event| match event.kind {
                EventKind::Svc { id, result, .. } if id == number => Some(result.0),
                _ => None,
            })
            .collect()
    }

    /// The alias region's start, as the program's `GetInfo` was told: its one argument-free
    /// answer is in the crash report's `x9` for the crashing programs.
    pub(crate) fn alias_start(report: &RunReport) -> u64 {
        report.crash().expect("a crash report").registers.x[9]
    }
}

use support::{SIZE, Then, alias_start, resource, results, run, uses_physical_memory};

#[test]
fn a_program_reads_and_writes_the_memory_it_maps_and_unmaps_it() {
    let Some(report) = run(&uses_physical_memory(Then::UnmapAndExit), resource()) else {
        return;
    };
    assert_eq!(
        report.exit,
        ExitReason::ProcessExited,
        "{:?}",
        report
            .crash()
            .map(starquake_core::crash::CrashReport::to_text)
    );
    for id in [
        SvcId::GetInfo,
        SvcId::MapPhysicalMemory,
        SvcId::UnmapPhysicalMemory,
    ] {
        assert_eq!(results(&report, id), [0], "{}", id.name());
    }
}

#[test]
fn the_word_past_the_mapped_range_is_not_there() {
    let Some(report) = run(&uses_physical_memory(Then::StorePastTheEnd), resource()) else {
        return;
    };
    // Everything up to the store worked; the store, one byte past the range, faults there.
    let crash = report.crash().expect("a crash report");
    let address = alias_start(&report) + SIZE;
    assert_eq!(
        crash.exception,
        Exception::DataAbort {
            address,
            write: true
        },
        "{}",
        crash.to_text()
    );
    assert_eq!(crash.fault_mapping, None);
    assert_eq!(results(&report, SvcId::MapPhysicalMemory), [0]);
}

#[test]
fn memory_that_was_unmapped_is_not_there() {
    let Some(report) = run(&uses_physical_memory(Then::LoadAfterUnmapping), resource()) else {
        return;
    };
    let crash = report.crash().expect("a crash report");
    let address = alias_start(&report);
    assert_eq!(
        crash.exception,
        Exception::DataAbort {
            address,
            write: false
        },
        "{}",
        crash.to_text()
    );
    assert_eq!(results(&report, SvcId::MapPhysicalMemory), [0]);
    assert_eq!(results(&report, SvcId::UnmapPhysicalMemory), [0]);
}

#[test]
fn a_program_with_no_system_resource_is_refused_with_invalid_state() {
    let program = uses_physical_memory(Then::UnmapAndExit);
    // The same program with no system resource in its metadata, and as an NRO alone (Eden's
    // defaults for a program with no NPDM: no system resource). (A size short of a whole page,
    // which the kernel would count as none, is not a whole 2 MiB block, and Horizon's loader
    // refuses it before a process exists.)
    let Some(none) = run(&program, 0) else {
        return;
    };
    let alone = common::run(&program, GENEROUS).unwrap();
    for report in [none, alone] {
        // The refusal ends the program at its `brk`, an undefined-instruction crash.
        let crash = report.crash().expect("a crash report");
        assert_eq!(
            crash.exception,
            Exception::UndefinedInstruction,
            "{}",
            crash.to_text()
        );
        assert_eq!(results(&report, SvcId::GetInfo), [0]);
        assert_eq!(
            results(&report, SvcId::MapPhysicalMemory),
            [KernelError::InvalidState.code().raw()]
        );
    }
}
