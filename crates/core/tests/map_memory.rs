//! `MapMemory` and `UnmapMemory` from synthetic programs run natively under the hypervisor
//! backend: the alias a program maps in its stack region reads what it wrote to the source
//! before, what it writes through the alias is in the source once the alias is gone, the
//! source cannot be reached while the alias exists, and the alias is gone once unmapped.
//!
//! These need a Mac with Apple Silicon. They skip themselves only when the `CI` environment
//! variable is set (hosted runners can't create VMs); anywhere else, failing to create a VM fails
//! the test. The kernel's side of both calls is tested on every host in the kernel crate
//! (`memory/space/alias_tests.rs`, `svc/memory/alias_tests.rs`, `dispatch/tests/map_memory.rs`),
//! and the translation's in `src/space/tests/alias.rs`.

#![cfg(all(target_os = "macos", target_arch = "aarch64"))]

use starquake_core::crash::Exception;
use starquake_core::{ExitReason, RunReport};
use starquake_kernel::handle::Handle;
use starquake_kernel::svc::SvcId;
use starquake_kernel::svc::info::InfoType;
use starquake_kernel::svc::memory::HEAP_SIZE_ALIGNMENT;
use starquake_memory::PAGE_SIZE;
use starquake_testkit::a64::{self, Assembler};
use starquake_testkit::nro::NroBuilder;
use starquake_trace::{Dump, EventKind};

#[cfg(test)]
mod common;

use common::{GENEROUS, run};

/// Shared setup. In a `cfg(test)` module so it may unwrap like the tests do.
#[cfg(test)]
mod support {
    use super::*;

    /// Register 31 as an operand of `sub` (shifted register): `XZR`.
    const XZR: u8 = 31;
    /// The value the program stores in the source before it maps the alias.
    pub(crate) const BEFORE: u32 = 0x1234;
    /// The value the program stores through the alias.
    pub(crate) const THROUGH: u32 = 0x5678;
    /// Where in the page the program stores [`THROUGH`]: a word that is not the first.
    const THROUGH_OFFSET: u32 = a64::MEM_OFFSET_ALIGN;

    /// What a program does besides the round trip.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub(crate) enum Then {
        /// Nothing: unmaps the alias, checks the source and exits.
        Exit,
        /// Loads from the source while the alias exists.
        LoadSourceWhileAliased,
        /// Loads from the alias after unmapping it.
        LoadAliasAfterUnmapping,
    }

    fn svc(id: SvcId) -> u32 {
        a64::svc(u32::from(id.number())).unwrap()
    }

    /// The 16-bit half of `value` at `shift`.
    fn halfword(value: u64, shift: u32) -> u32 {
        u32::try_from((value >> shift) & u64::from(a64::MAX_IMM16)).unwrap()
    }

    /// `GetInfo(id, current process, 0)` into x1, failing to `fail`.
    fn get_info(asm: &mut Assembler, id: InfoType, fail: a64::Label) {
        let process = u64::from(Handle::CURRENT_PROCESS.raw());
        asm.emit(a64::movz(1, id.number(), 0).unwrap());
        asm.emit(a64::movz(2, halfword(process, 0), 0).unwrap());
        asm.emit(a64::movk(2, halfword(process, 16), 16).unwrap());
        asm.emit(a64::movz(3, 0, 0).unwrap());
        asm.emit(svc(SvcId::GetInfo));
        asm.cbnz(0, fail).unwrap();
    }

    /// `id(x9, x10, one page)`, failing to `fail`.
    fn alias_call(asm: &mut Assembler, id: SvcId, fail: a64::Label) {
        asm.emit(a64::mov(0, 9).unwrap());
        asm.emit(a64::mov(1, 10).unwrap());
        asm.emit(a64::movz(2, u32::try_from(PAGE_SIZE).unwrap(), 0).unwrap());
        asm.emit(svc(id));
        asm.cbnz(0, fail).unwrap();
    }

    /// Loads the word at `offset` from `base` and fails to `fail` unless it is `value`.
    fn expect(asm: &mut Assembler, base: u8, offset: u32, value: u32, fail: a64::Label) {
        asm.emit(a64::ldr(3, base, offset).unwrap());
        asm.emit(a64::movz(2, value, 0).unwrap());
        asm.emit(a64::sub(4, 2, 3).unwrap());
        asm.cbnz(4, fail).unwrap();
    }

    /// A program that sets a heap of [`HEAP_SIZE_ALIGNMENT`] bytes, stores [`BEFORE`] in its
    /// first word, maps the heap's first page with `MapMemory` at the last page of the stack
    /// region, checks that the alias reads [`BEFORE`], stores [`THROUGH`] through it, unmaps it
    /// with `UnmapMemory`, checks that the source reads [`THROUGH`], and exits; `then` adds a
    /// load that faults. A call that fails, and a value that does not read back, ends at
    /// `brk #1`.
    ///
    /// ```text
    ///       movz x1, #HEAP, lsl #16
    ///       svc  #0x1              ; SetHeapSize: x1 = the heap
    ///       cbnz x0, fail
    ///       mov  x10, x1           ; the source
    ///       ...                    ; GetInfo(StackRegionAddress) -> x11
    ///       ...                    ; GetInfo(StackRegionSize) -> x1
    ///       sub  x3, xzr, x1
    ///       sub  x9, x11, x3       ; the stack region's end
    ///       movz x4, #PAGE
    ///       sub  x9, x9, x4        ; the alias: the region's last page
    ///       movz x2, #BEFORE
    ///       str  x2, [x10]
    ///       ...                    ; MapMemory(x9, x10, PAGE)
    ///       ...                    ; [x9] == BEFORE
    ///       movz x2, #THROUGH
    ///       str  x2, [x9, #8]
    ///       ...                    ; LoadSourceWhileAliased: ldr x3, [x10]
    ///       ...                    ; UnmapMemory(x9, x10, PAGE)
    ///       ...                    ; [x10, #8] == THROUGH
    ///       ...                    ; LoadAliasAfterUnmapping: ldr x3, [x9]
    ///       svc  #0x7
    /// fail: brk  #1
    /// ```
    pub(crate) fn aliases_a_heap_page(then: Then) -> Vec<u8> {
        let mut asm = Assembler::new();
        let fail = asm.label();
        asm.emit(a64::movz(1, halfword(HEAP_SIZE_ALIGNMENT, 16), 16).unwrap());
        asm.emit(svc(SvcId::SetHeapSize));
        asm.cbnz(0, fail).unwrap();
        asm.emit(a64::mov(10, 1).unwrap());
        get_info(&mut asm, InfoType::StackRegionAddress, fail);
        asm.emit(a64::mov(11, 1).unwrap());
        get_info(&mut asm, InfoType::StackRegionSize, fail);
        asm.emit(a64::sub(3, XZR, 1).unwrap());
        asm.emit(a64::sub(9, 11, 3).unwrap());
        asm.emit(a64::movz(4, u32::try_from(PAGE_SIZE).unwrap(), 0).unwrap());
        asm.emit(a64::sub(9, 9, 4).unwrap());
        asm.emit(a64::movz(2, BEFORE, 0).unwrap());
        asm.emit(a64::str(2, 10, 0).unwrap());
        alias_call(&mut asm, SvcId::MapMemory, fail);
        expect(&mut asm, 9, 0, BEFORE, fail);
        asm.emit(a64::movz(2, THROUGH, 0).unwrap());
        asm.emit(a64::str(2, 9, THROUGH_OFFSET).unwrap());
        if then == Then::LoadSourceWhileAliased {
            asm.emit(a64::ldr(3, 10, 0).unwrap());
        }
        alias_call(&mut asm, SvcId::UnmapMemory, fail);
        expect(&mut asm, 10, THROUGH_OFFSET, THROUGH, fail);
        if then == Then::LoadAliasAfterUnmapping {
            asm.emit(a64::ldr(3, 9, 0).unwrap());
        }
        asm.emit(svc(SvcId::ExitProcess));
        asm.bind(fail).unwrap();
        asm.emit(a64::brk(1).unwrap());
        NroBuilder::new(&asm.finish().unwrap()).build().unwrap()
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
}

use support::{Then, aliases_a_heap_page, results};

#[test]
fn what_is_written_through_either_address_is_read_through_the_other() {
    let Some(report) = run(&aliases_a_heap_page(Then::Exit), GENEROUS) else {
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
        SvcId::SetHeapSize,
        SvcId::MapMemory,
        SvcId::UnmapMemory,
        SvcId::ExitProcess,
    ] {
        assert_eq!(results(&report, id).first(), Some(&0), "{}", id.name());
    }
    assert_eq!(results(&report, SvcId::GetInfo), [0, 0]);
}

#[test]
fn the_source_cannot_be_reached_while_the_alias_exists() {
    let Some(report) = run(&aliases_a_heap_page(Then::LoadSourceWhileAliased), GENEROUS) else {
        return;
    };
    let crash = report.crash().expect("a crash report");
    // The heap's start is in x10.
    let address = crash.registers.x[10];
    assert_eq!(
        crash.exception,
        Exception::DataAbort {
            address,
            write: false
        },
        "{}",
        crash.to_text()
    );
    assert_eq!(results(&report, SvcId::MapMemory), [0]);
    assert_eq!(results(&report, SvcId::UnmapMemory), Vec::<u32>::new());
}

#[test]
fn the_alias_is_gone_once_unmapped() {
    let Some(report) = run(
        &aliases_a_heap_page(Then::LoadAliasAfterUnmapping),
        GENEROUS,
    ) else {
        return;
    };
    let crash = report.crash().expect("a crash report");
    // The alias's address is in x9; the source's round trip passed, or the run would have
    // ended at `brk`.
    let address = crash.registers.x[9];
    assert_eq!(
        crash.exception,
        Exception::DataAbort {
            address,
            write: false
        },
        "{}",
        crash.to_text()
    );
    assert_eq!(results(&report, SvcId::MapMemory), [0]);
    assert_eq!(results(&report, SvcId::UnmapMemory), [0]);
}
