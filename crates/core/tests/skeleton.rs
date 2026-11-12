//! The walking skeleton, end to end: synthetic guest programs run natively under the hypervisor
//! backend, their system calls go through the kernel's dispatcher, and what they did is read
//! back from the run's report.
//!
//! These need a Mac with Apple Silicon. They skip themselves only when the `CI` environment
//! variable is set (hosted runners can't create VMs); anywhere else, failing to create a VM fails
//! the test. The same loop is tested on every host against a scripted vCPU in
//! `src/machine/tests.rs`.
//!
//! The evidence is meant to be checkable: what the hello program must print is written here as
//! constants ([`EXPECTED_OUTPUT`]), independent of the program's own copy of the text, its trace
//! is compared with a committed snapshot, and a control program that prints something else is
//! shown to fail both comparisons.

#![cfg(all(target_os = "macos", target_arch = "aarch64"))]

use std::time::Duration;

use starquake_core::crash::{Exception, Location};
use starquake_core::{Error, ExitReason, LoadError, RunReport, exit_code};
use starquake_kernel::memory::layout::{LayoutRequest, RegionOffsets, Regions};
use starquake_kernel::memory::{CreateError, MemoryArrangement};
use starquake_kernel::result::{KernelError, SvcError};
use starquake_kernel::svc::SvcId;
use starquake_memory::PAGE_SIZE;
use starquake_testkit::a64::SysReg;
use starquake_testkit::nro::{CODE_OFFSET, NroBuilder};
use starquake_testkit::programs::COUNTER_HZ;
use starquake_testkit::{a64, programs};
use starquake_trace::{Dump, EventKind};

/// What the hello program must print, one `OutputDebugString` call per line.
const EXPECTED_OUTPUT: [&str; 2] = ["hello from the guest", "starquake walking skeleton"];

/// What the control program prints: the same two lines with the last letter changed.
const CONTROL_OUTPUT: [&str; 2] = ["hello from the guest", "starquake walking skeletoN"];

/// The committed snapshot of the hello program's trace, as `insta` stores it: a header between
/// two `---` lines, then the text.
const HELLO_SNAPSHOT: &str = include_str!("snapshots/skeleton__hello_trace.snap");

/// How far apart two readings of the counter at one instant can be, one through a patched
/// instruction and one through `GetSystemTick`: each rounds down and may be one tick below the
/// exact value (`crates/core/src/clock.rs`), so one tick.
const ROUNDING: u64 = 1;

#[cfg(test)]
mod common;

/// Shared setup. In a `cfg(test)` module so it may unwrap like the tests do.
#[cfg(test)]
mod support {
    use super::*;

    /// The guest's output as text.
    pub(crate) fn output(report: &RunReport) -> Vec<String> {
        report
            .guest_log
            .iter()
            .map(|line| String::from_utf8(line.clone()).expect("the guest printed UTF-8"))
            .collect()
    }

    /// The comparison of a run's output with [`EXPECTED_OUTPUT`].
    pub(crate) fn output_is_expected(report: &RunReport) -> bool {
        output(report) == EXPECTED_OUTPUT
    }

    /// The text of the committed hello snapshot, without `insta`'s header.
    pub(crate) fn golden_trace() -> &'static str {
        let mut parts = HELLO_SNAPSHOT.splitn(3, "---\n");
        assert_eq!(
            parts.next(),
            Some(""),
            "the snapshot starts with its header"
        );
        parts.next().expect("the snapshot has a header");
        parts.next().expect("the snapshot has a body").trim_end()
    }

    /// The comparison of a run's trace with the committed snapshot.
    pub(crate) fn trace_is_golden(report: &RunReport) -> bool {
        report.trace.trim_end() == golden_trace()
    }

    /// Where the program's image was placed.
    pub(crate) fn image_base(report: &RunReport) -> u64 {
        report
            .memory_map
            .iter()
            .find(|mapping| mapping.name == "main.text")
            .expect("a text mapping")
            .start
    }

    /// The events in the report's flight-recorder dump, in order.
    pub(crate) fn events(report: &RunReport) -> Vec<EventKind> {
        Dump::from_bytes(&report.events)
            .expect("a readable dump")
            .merged()
            .iter()
            .map(|event| event.kind)
            .collect()
    }
}

use common::{GENEROUS, run};
use support::{events, golden_trace, image_base, output, output_is_expected, trace_is_golden};

#[test]
fn hello_prints_the_expected_lines_and_its_trace_matches_the_snapshot() {
    let Some(report) = run(&programs::hello().unwrap(), GENEROUS) else {
        return;
    };
    assert_eq!(report.exit, ExitReason::ProcessExited);
    assert_eq!(report.exit_code(), exit_code::STOP_CONDITION);
    assert_eq!(output(&report), EXPECTED_OUTPUT);
    insta::assert_snapshot!("hello_trace", report.trace);
    // The two comparisons the control test below must fail. Checking them here as well shows
    // that they agree with the assertions above when a run is right.
    assert!(output_is_expected(&report));
    assert!(trace_is_golden(&report));

    assert_eq!(
        report.system_calls.iter().collect::<Vec<_>>(),
        [
            (&"ExitProcess".to_owned(), &1),
            (&"OutputDebugString".to_owned(), &2)
        ]
    );
    assert!(report.stubbed.is_empty(), "{:?}", report.stubbed);
    assert_eq!(report.patched_counter_reads, Vec::<u64>::new());
    // Each line's event carries the length and hash of the text this test expects, so the trace
    // is tied to the output and not only to itself.
    let logs: Vec<EventKind> = events(&report)
        .into_iter()
        .filter(|kind| matches!(kind, EventKind::GuestLog { .. }))
        .collect();
    assert_eq!(
        logs,
        EXPECTED_OUTPUT.map(|line| EventKind::guest_log(line.as_bytes()))
    );
}

#[test]
fn a_program_that_prints_something_else_fails_the_same_comparisons() {
    // The control: a run that is healthy in every other way, and wrong in one letter.
    let lines = CONTROL_OUTPUT.map(str::as_bytes);
    let Some(report) = run(&programs::print_lines(&lines).unwrap(), GENEROUS) else {
        return;
    };
    assert_eq!(report.exit, ExitReason::ProcessExited);
    assert_eq!(output(&report), CONTROL_OUTPUT);

    assert!(!output_is_expected(&report));
    assert!(!trace_is_golden(&report));
    // The trace differs from the snapshot in exactly one line: the changed line's log event,
    // whose hash is of different text. Its length, and so the system call's arguments, are the
    // same.
    let golden: Vec<&str> = golden_trace().lines().collect();
    let control: Vec<&str> = report.trace.trim_end().lines().collect();
    assert_eq!(golden.len(), control.len());
    let differing: Vec<(&str, &str)> = golden
        .into_iter()
        .zip(control)
        .filter(|(golden, control)| golden != control)
        .collect();
    let line = |text: &str| format!("  {}", EventKind::guest_log(text.as_bytes()));
    assert_eq!(
        differing,
        [(
            line(EXPECTED_OUTPUT[1]).as_str(),
            line(CONTROL_OUTPUT[1]).as_str()
        )]
    );
}

#[test]
fn the_counter_advances_through_patched_reads_and_reads_the_guest_rate() {
    let counter = programs::counter(COUNTER_HZ).unwrap();
    let Some(report) = run(&counter.nro, GENEROUS) else {
        return;
    };
    // The program exits only if its second read was above its first and `cntfrq_el0` read the
    // guest's rate; otherwise it stops at a `brk` and this is a crash.
    assert_eq!(
        report.exit,
        ExitReason::ProcessExited,
        "{:?}",
        report.crash()
    );
    // All three reads were patched, so none of them reached the backend's own counter emulation
    // (which would have read the host's rate, not the guest's).
    let base = image_base(&report);
    assert_eq!(
        report.patched_counter_reads,
        counter.read_offsets.map(|offset| base + offset)
    );
    assert!(
        report
            .memory_map
            .iter()
            .any(|mapping| mapping.name == "trampolines")
    );
    insta::assert_snapshot!("counter_trace", report.trace);
}

#[test]
fn a_counter_program_expecting_another_rate_stops_at_its_rate_check() {
    // The control for the counter test: the rate the program compares against is off by one, so
    // the same run must end at the program's `brk #3`, and the crash report says so.
    let counter = programs::counter(COUNTER_HZ + 1).unwrap();
    let Some(report) = run(&counter.nro, GENEROUS) else {
        return;
    };
    assert_eq!(report.exit_code(), exit_code::GUEST_CRASHED);
    let crash = report.crash().expect("a crash report");
    assert_eq!(crash.exception, Exception::UndefinedInstruction);
    assert_eq!(crash.pc, image_base(&report) + counter.wrong_rate_offset);
    // x23 is what `cntfrq_el0` read through the patched site, x24 what the program expected.
    assert_eq!(crash.registers.x[23], COUNTER_HZ);
    assert_eq!(crash.registers.x[24], COUNTER_HZ + 1);
    // The two counter reads before it passed their checks: the second is above the first.
    assert!(
        crash.registers.x[20] > crash.registers.x[19],
        "{:#x?}",
        crash.registers.x
    );
}

#[test]
fn a_tick_read_by_system_call_lies_between_the_patched_counter_reads_around_it() {
    // Guest time has one zero point (kernel spec, "Guest time"): `GetSystemTick` and a counter
    // read through a patched instruction are readings of one counter. The program reads the
    // counter, asks the kernel for the tick, reads the counter again, and stops where the
    // crash report shows all three:
    //
    //   mrs x19, cntpct_el0     (patched)
    //   svc #0x1e               GetSystemTick: the tick in x0
    //   mov x20, x0
    //   mrs x21, cntpct_el0     (patched)
    //   brk #0
    let words = [
        a64::mrs(19, SysReg::CntPctEl0).unwrap(),
        a64::svc(u32::from(SvcId::GetSystemTick.number())).unwrap(),
        a64::mov(20, 0).unwrap(),
        a64::mrs(21, SysReg::CntPctEl0).unwrap(),
        a64::brk(0).unwrap(),
    ];
    let stop_offset = CODE_OFFSET as u64 + 4 * u64::from(a64::INSN_SIZE);
    let file = NroBuilder::new(&words).build().unwrap();
    let Some(report) = run(&file, GENEROUS) else {
        return;
    };
    assert_eq!(report.exit_code(), exit_code::GUEST_CRASHED);
    let crash = report.crash().expect("the program stops at its brk");
    assert_eq!(crash.exception, Exception::UndefinedInstruction);
    assert_eq!(crash.pc, image_base(&report) + stop_offset);
    // Both counter reads went through the patcher, and the kernel answered the system call.
    assert_eq!(report.patched_counter_reads.len(), 2);
    assert_eq!(
        report.system_calls.iter().collect::<Vec<_>>(),
        [(&SvcId::GetSystemTick.name().to_owned(), &1)]
    );
    assert!(report.stubbed.is_empty(), "{:?}", report.stubbed);

    // Apart from rounding, the system call's tick is bracketed by the reads around it.
    let [before, tick, after] = [19, 20, 21].map(|register| crash.registers.x[register]);
    assert!(before > ROUNDING, "the counter read as {before}");
    assert!(
        before - ROUNDING <= tick && tick <= after + ROUNDING,
        "patched read {before}, GetSystemTick {tick}, patched read {after}"
    );
    // The bracket is one system call wide, far under a second of ticks, while the two zero
    // points this replaces (the host's boot and the start of the run) are further apart than
    // that on any machine that has been up for a second. So this fails if they differ.
    assert!(
        after - before < COUNTER_HZ,
        "the reads are {} ticks apart",
        after - before
    );
}

#[test]
fn a_store_to_an_unmapped_address_is_a_crash_naming_the_address_and_the_instruction() {
    let crash = programs::crash().unwrap();
    let Some(report) = run(&crash.nro, GENEROUS) else {
        return;
    };
    assert_eq!(report.exit_code(), exit_code::GUEST_CRASHED);
    let found = report.crash().expect("a crash report");
    assert_eq!(
        found.exception,
        Exception::DataAbort {
            address: crash.fault_address,
            write: true
        }
    );
    assert_eq!(found.pc, image_base(&report) + crash.store_offset);
    assert_eq!(
        found.pc_location,
        Some(Location {
            module: "main".into(),
            offset: crash.store_offset
        })
    );
    assert_eq!(found.fault_mapping, None);
    // The registers are the ones the program set before the store.
    assert_eq!(found.registers.x[0], crash.fault_address - 8);
    assert_eq!(found.registers.x[1], 0x1234);
    let text = found.to_text();
    assert!(
        text.contains(&format!(
            "a write at address {:#x} (not mapped)",
            crash.fault_address
        )),
        "{text}"
    );
    assert!(
        text.contains(&format!(
            "pc {:#x} (main+{:#x})",
            found.pc, crash.store_offset
        )),
        "{text}"
    );
    // The store never happened and the program never reached its ExitProcess.
    assert!(report.system_calls.is_empty(), "{:?}", report.system_calls);
    insta::assert_snapshot!("crash_trace", report.trace);
}

/// Programs that use their own address space, to show that each part of it has the permissions
/// it should. Each has eight bytes of read-only data, eight of data and a page of `.bss`.
#[cfg(test)]
mod permissions {
    use starquake_testkit::a64::{Assembler, SysReg};
    use starquake_testkit::nro::CODE_OFFSET;

    use super::*;

    const RODATA: [u8; 8] = *b"readonly";
    const DATA: [u8; 8] = *b"writable";
    const BSS_SIZE: u32 = 0x1000;
    /// The link register, which `ret` jumps to: the one indirect branch the assembler has.
    const LR: u8 = 30;
    /// The stack pointer, as the base of an `add` or `sub`.
    const SP: u8 = 31;

    /// Where the parts of the image are, as offsets from its first byte.
    #[derive(Debug, Clone, Copy, Default)]
    pub(crate) struct Offsets {
        pub(crate) rodata: u64,
        pub(crate) data: u64,
        pub(crate) bss: u64,
    }

    /// The image offset of instruction `index` of a program's code.
    pub(crate) fn code_offset(index: usize) -> u64 {
        u64::try_from(CODE_OFFSET).unwrap()
            + u64::try_from(index).unwrap() * u64::from(a64::INSN_SIZE)
    }

    /// The distance for an `adr` at instruction `index` to reach image offset `target`.
    pub(crate) fn distance(target: u64, index: usize) -> i64 {
        i64::try_from(target).unwrap() - i64::try_from(code_offset(index)).unwrap()
    }

    /// Builds the NRO whose code `code` returns when told where the image's parts are. `code`
    /// must return the same number of instructions whatever it is told.
    pub(crate) fn build(code: impl Fn(Offsets) -> Vec<u32>) -> (Vec<u8>, Offsets) {
        let builder = |words: &[u32]| {
            NroBuilder::new(words)
                .rodata(&RODATA)
                .data(&DATA)
                .bss_size(BSS_SIZE)
        };
        // The layout depends on how many instructions there are, not on what they are.
        let placeholder = vec![a64::nop(); code(Offsets::default()).len()];
        let [_, rodata, data] = builder(&placeholder).segment_offsets().unwrap();
        let offsets = Offsets {
            rodata: u64::from(rodata),
            data: u64::from(data),
            // `.bss` follows the image, which the builder writes as whole pages.
            bss: builder(&placeholder).build().unwrap().len() as u64,
        };
        let words = code(offsets);
        assert_eq!(words.len(), placeholder.len());
        (builder(&words).build().unwrap(), offsets)
    }

    /// A program that reads its read-only data, then writes that value to its data, its
    /// `.bss`, its stack and its TLS block and reads it back from each. It exits only if every
    /// value read back is the one written and is not zero; otherwise it stops at `brk #1`.
    ///
    /// ```text
    ///       adr  x0, rodata
    ///       ldr  x1, [x0]
    ///       cbz  x1, fail
    ///       adr  x2, data          ; then .bss, sp - 16, tpidrro_el0
    ///       str  x1, [x2]
    ///       ldr  x3, [x2]
    ///       sub  x9, x1, x3
    ///       cbnz x9, fail
    ///       ...
    ///       svc  #0x7
    /// fail: brk  #1
    /// ```
    pub(crate) fn uses_every_part() -> Vec<u8> {
        let (file, _) = build(|at| {
            let mut asm = Assembler::new();
            let fail = asm.label();
            asm.emit(a64::adr(0, distance(at.rodata, asm.len())).unwrap());
            asm.emit(a64::ldr(1, 0, 0).unwrap());
            asm.cbz(1, fail).unwrap();
            for part in 0..4 {
                let address = match part {
                    0 => a64::adr(2, distance(at.data, asm.len())),
                    1 => a64::adr(2, distance(at.bss, asm.len())),
                    2 => a64::sub_imm(2, SP, 16),
                    _ => a64::mrs(2, SysReg::TpidrroEl0),
                };
                asm.emit(address.unwrap());
                asm.emit(a64::str(1, 2, 0).unwrap());
                asm.emit(a64::ldr(3, 2, 0).unwrap());
                asm.emit(a64::sub(9, 1, 3).unwrap());
                asm.cbnz(9, fail).unwrap();
            }
            asm.emit(a64::svc(programs::SVC_EXIT_PROCESS).unwrap());
            asm.bind(fail).unwrap();
            asm.emit(a64::brk(1).unwrap());
            asm.finish().unwrap()
        });
        file
    }

    /// What a program that does one forbidden thing does, and what must come of it.
    pub(crate) struct Forbidden {
        pub(crate) name: &'static str,
        /// The NRO.
        pub(crate) file: Vec<u8>,
        /// The name of the mapping the faulting address is in.
        pub(crate) mapping: &'static str,
        /// The faulting address, as an offset from the start of that mapping. `None` means
        /// sixteen bytes below its end (the stack's top).
        pub(crate) offset: Option<u64>,
        /// A store (`Some(true)`), a load (`Some(false)`) or a jump (`None`).
        pub(crate) write: Option<bool>,
    }

    /// A program that stores x1 to the address `target` puts in x0, then exits.
    fn store_to(target: impl Fn(Offsets) -> u32) -> Vec<u8> {
        build(|at| {
            vec![
                target(at),
                a64::str(1, 0, 0).unwrap(),
                a64::svc(programs::SVC_EXIT_PROCESS).unwrap(),
            ]
        })
        .0
    }

    /// A program that jumps to the address `target` puts in the link register.
    fn jump_to(target: impl Fn(Offsets) -> u32) -> Vec<u8> {
        build(|at| vec![target(at), a64::ret()]).0
    }

    /// One program per thing the permissions must stop.
    pub(crate) fn forbidden() -> Vec<Forbidden> {
        let case = |name, file, mapping, offset, write| Forbidden {
            name,
            file,
            mapping,
            offset,
            write,
        };
        let adr = |rd: u8, target: u64| a64::adr(rd, distance(target, 0)).unwrap();
        // Rodata and data follow the text in the image; the first instruction is in the text.
        let own_code = code_offset(0);
        vec![
            case(
                "a store to the text",
                store_to(|_| adr(0, own_code)),
                "main.text",
                Some(own_code),
                Some(true),
            ),
            case(
                "a store to the read-only data",
                store_to(|at| adr(0, at.rodata)),
                "main.rodata",
                Some(0),
                Some(true),
            ),
            case(
                "a jump into the read-only data",
                jump_to(|at| adr(LR, at.rodata)),
                "main.rodata",
                Some(0),
                None,
            ),
            case(
                "a jump into the data",
                jump_to(|at| adr(LR, at.data)),
                "main.data",
                Some(0),
                None,
            ),
            case(
                "a jump into .bss",
                jump_to(|at| adr(LR, at.bss)),
                "main.bss",
                Some(0),
                None,
            ),
            case(
                "a jump into the stack",
                jump_to(|_| a64::sub_imm(LR, SP, 16).unwrap()),
                "stack",
                None,
                None,
            ),
            case(
                "a jump into the TLS block",
                jump_to(|_| a64::mrs(LR, SysReg::TpidrroEl0).unwrap()),
                "tls",
                Some(0),
                None,
            ),
        ]
    }

    /// A program that loads from the absolute address `address`, then exits.
    pub(crate) fn load_from(address: u64) -> Vec<u8> {
        let halfword =
            |shift: u32| u32::try_from((address >> shift) & u64::from(a64::MAX_IMM16)).unwrap();
        build(|_| {
            vec![
                a64::movz(0, halfword(0), 0).unwrap(),
                a64::movk(0, halfword(16), 16).unwrap(),
                a64::movk(0, halfword(32), 32).unwrap(),
                a64::ldr(1, 0, 0).unwrap(),
                a64::svc(programs::SVC_EXIT_PROCESS).unwrap(),
            ]
        })
        .0
    }

    /// Image offset of the `ldr` in a [`load_from`] program.
    pub(crate) fn load_offset() -> u64 {
        code_offset(3)
    }
}

#[test]
fn a_program_can_use_every_part_of_its_address_space_as_that_part_allows() {
    let Some(report) = run(&permissions::uses_every_part(), GENEROUS) else {
        return;
    };
    // Reading rodata, and writing and reading back data, .bss, the stack and the TLS block all
    // worked; a wrong value, or a fault, would have ended the run as a crash.
    assert_eq!(
        report.exit,
        ExitReason::ProcessExited,
        "{:?}",
        report
            .crash()
            .map(starquake_core::crash::CrashReport::to_text)
    );
    let names: Vec<&str> = report
        .memory_map
        .iter()
        .map(|mapping| mapping.name.as_str())
        .collect();
    assert_eq!(
        names,
        [
            "main.text",
            "main.rodata",
            "main.data",
            "main.bss",
            "tls",
            "stack"
        ]
    );
}

#[test]
fn a_store_to_code_or_read_only_data_and_a_jump_into_data_are_crashes_naming_the_mapping() {
    for case in permissions::forbidden() {
        let Some(report) = run(&case.file, GENEROUS) else {
            return;
        };
        let name = case.name;
        assert_eq!(report.exit_code(), exit_code::GUEST_CRASHED, "{name}");
        let crash = report.crash().expect("a crash report");
        let mapping = report
            .memory_map
            .iter()
            .find(|mapping| mapping.name == case.mapping)
            .expect("the mapping is in the memory map");
        let address = match case.offset {
            Some(offset) => mapping.start + offset,
            None => mapping.end - 16,
        };
        let (exception, pc, what) = match case.write {
            // A refused store or load: the program counter is still on it, the program's
            // second instruction.
            Some(write) => (
                Exception::DataAbort { address, write },
                image_base(&report) + permissions::code_offset(1),
                if write {
                    "data abort, a write"
                } else {
                    "data abort, a read"
                },
            ),
            // A refused jump: the program counter is where it jumped to.
            None => (
                Exception::InstructionAbort { address },
                address,
                "instruction abort, a fetch",
            ),
        };
        assert_eq!(crash.exception, exception, "{name}");
        assert_eq!(crash.pc, pc, "{name}");
        assert_eq!(crash.fault_mapping.as_deref(), Some(case.mapping), "{name}");
        assert_eq!(
            crash.exception_text(),
            format!("{what} at address {address:#x} (in {})", case.mapping),
            "{name}"
        );
        // The program never got as far as its ExitProcess.
        assert!(report.system_calls.is_empty(), "{name}");
    }
}

#[test]
fn the_pages_next_to_the_tls_block_and_the_stack_are_not_mapped() {
    // Where the TLS block and the stack are is read off a run of a program with the same
    // segments, not restated.
    let Some(healthy) = run(&permissions::uses_every_part(), GENEROUS) else {
        return;
    };
    let mapping = |name: &str| {
        healthy
            .memory_map
            .iter()
            .find(|mapping| mapping.name == name)
            .expect("the mapping is in the memory map")
    };
    let (tls, stack) = (mapping("tls"), mapping("stack"));
    assert!(tls.end < stack.start);
    // The first byte after the TLS block, the last eight below the stack, and the first after
    // the stack (which is also the first after the program's whole window).
    for address in [tls.end, stack.start - 8, stack.end] {
        let report = run(&permissions::load_from(address), GENEROUS).unwrap();
        assert_eq!(report.memory_map, healthy.memory_map, "{address:#x}");
        let crash = report.crash().expect("a crash report");
        assert_eq!(
            crash.exception,
            Exception::DataAbort {
                address,
                write: false
            }
        );
        assert_eq!(crash.fault_mapping, None, "{address:#x}");
        assert_eq!(crash.pc, image_base(&report) + permissions::load_offset());
        assert_eq!(
            crash.exception_text(),
            format!("data abort, a read at address {address:#x} (not mapped)")
        );
    }
    // The last eight bytes of the TLS block and the first eight of the stack can be loaded.
    for address in [tls.end - 8, stack.start] {
        let report = run(&permissions::load_from(address), GENEROUS).unwrap();
        assert_eq!(report.exit, ExitReason::ProcessExited, "{address:#x}");
    }
}

#[test]
fn a_program_whose_segments_cannot_be_mapped_is_an_emulator_error() {
    use starquake_testkit::nro::{SEGMENT_HEADER_SIZE, SEGMENTS_OFFSET};
    // The read-only data segment is made to start one byte into its page.
    let (mut file, offsets) = permissions::build(|_| vec![a64::nop()]);
    let misaligned = u32::try_from(offsets.rodata + 1).unwrap();
    let at = SEGMENTS_OFFSET + SEGMENT_HEADER_SIZE;
    file[at..at + 4].copy_from_slice(&misaligned.to_le_bytes());
    let Some(report) = run(&file, GENEROUS) else {
        return;
    };
    let error = Error::Load(LoadError::MisalignedSegment { offset: misaligned });
    assert_eq!(report.exit, ExitReason::EmulatorError(error.to_string()));
    assert_eq!(report.exit_code(), exit_code::EMULATOR_ERROR);
    assert_eq!(
        error.to_string(),
        format!("the NRO's segment at offset {misaligned:#x} does not start on a page boundary")
    );
}

#[test]
fn a_second_thread_runs_natively_on_its_own_core_and_is_joined() {
    // The program starts a thread on core 1, waits for it to end (a `WaitSynchronization` on
    // its handle, with no timeout), and exits only if the thread reported core 1 and left its
    // marker; any failed check is a `brk` with its own number. Core 1 is another host thread
    // with its own vCPU.
    let program = programs::threads().unwrap();
    let Some(report) = run(&program.nro, GENEROUS) else {
        return;
    };
    assert_eq!(
        report.exit,
        ExitReason::ProcessExited,
        "{:?}\n{}",
        report.crash().map(|crash| crash.to_text()),
        report.trace
    );
    let made: Vec<_> = report
        .system_calls
        .iter()
        .map(|(name, count)| (name.as_str(), *count))
        .collect();
    assert_eq!(
        made,
        [
            ("CreateThread", 1),
            ("ExitProcess", 1),
            ("ExitThread", 1),
            ("GetCurrentProcessorNumber", 1),
            ("StartThread", 1),
            ("WaitSynchronization", 1),
        ]
    );
    // Each thread's calls were made on its own core.
    let dump = Dump::from_bytes(&report.events).expect("a readable dump");
    let second = u32::from(SvcId::GetCurrentProcessorNumber.number());
    for event in dump.merged() {
        if let EventKind::Svc { id, .. } = event.kind {
            let core = if id == second || id == u32::from(SvcId::ExitThread.number()) {
                u8::try_from(programs::SECOND_THREAD_CORE).unwrap()
            } else {
                0
            };
            assert_eq!(event.core, core, "{event:?}");
        }
    }
}

#[test]
fn an_endless_loop_is_ended_by_the_timeout() {
    let timeout = Duration::from_millis(300);
    let Some(report) = run(&programs::endless_loop().unwrap(), timeout) else {
        return;
    };
    assert_eq!(report.exit, ExitReason::Timeout);
    assert_eq!(report.exit_code(), exit_code::TIMEOUT);
    assert!(report.duration >= timeout, "{:?}", report.duration);
    assert!(report.system_calls.is_empty());
}

#[test]
fn a_program_larger_than_an_application_may_be_is_an_emulator_error() {
    // As much `.bss` as the application pool of the console holds (with the page of text, one
    // page too much): the kernel cannot reserve the program's code.
    let pool = MemoryArrangement::FourGb.application_pool_size();
    let file = NroBuilder::new(&[a64::nop()])
        .bss_size(u32::try_from(pool).unwrap())
        .build()
        .unwrap();
    let Some(report) = run(&file, GENEROUS) else {
        return;
    };
    let error = Error::Load(LoadError::AddressSpace(CreateError::Code(
        SvcError::Kernel(KernelError::LimitReached),
    )));
    assert_eq!(report.exit, ExitReason::EmulatorError(error.to_string()));
    assert_eq!(report.exit_code(), exit_code::EMULATOR_ERROR);
    assert_eq!(report.trace, "");
    assert_eq!(
        error.to_string(),
        "the program's address space could not be created: the program's code could not be \
         mapped: limit reached"
    );
}

/// Programs that make the kernel's memory calls and then use, or misuse, the memory: what a
/// call did must be what the vCPU's translation does.
#[cfg(test)]
mod memory_calls {
    use starquake_testkit::a64::Assembler;

    use super::permissions::{Offsets, build, distance};
    use super::*;

    /// The value the programs store and compare.
    const VALUE: u32 = 0x1234;
    /// The size of the heap the heap programs ask for: one alignment unit of `SetHeapSize`.
    pub(crate) const HEAP_SIZE: u64 = starquake_kernel::svc::memory::HEAP_SIZE_ALIGNMENT;
    /// `Svc::MemoryPermission::Read` and `ReadWrite`, as a guest passes them.
    pub(crate) const PERM_READ: u32 = 1;
    pub(crate) const PERM_READ_WRITE: u32 = 3;

    fn svc(id: SvcId) -> u32 {
        a64::svc(u32::from(id.number())).unwrap()
    }

    /// A program that stores to the first page of its data, asks `SetMemoryPermission` to give
    /// that page `permission`, checks that the call succeeded and that a load still reads what
    /// was stored, stores again and exits. Returns the NRO, where its parts are, and the index
    /// of the second store.
    ///
    /// ```text
    ///       adr  x9, data
    ///       movz x1, #VALUE
    ///       str  x1, [x9]          ; backs the page; the vCPU now caches a writable entry
    ///       mov  x0, x9
    ///       movz x1, #0x1000
    ///       movz x2, #permission
    ///       svc  #0x2
    ///       cbnz x0, fail
    ///       ldr  x3, [x9]
    ///       movz x5, #VALUE
    ///       sub  x4, x3, x5
    ///       cbnz x4, fail
    ///       str  x5, [x9]          ; the second store
    ///       svc  #0x7
    /// fail: brk  #1
    /// ```
    pub(crate) fn reprotects_its_data(permission: u32) -> (Vec<u8>, Offsets, usize) {
        let second_store = std::cell::Cell::new(0);
        let (file, offsets) = build(|at| {
            let mut asm = Assembler::new();
            let fail = asm.label();
            asm.emit(a64::adr(9, distance(at.data, asm.len())).unwrap());
            asm.emit(a64::movz(1, VALUE, 0).unwrap());
            asm.emit(a64::str(1, 9, 0).unwrap());
            asm.emit(a64::mov(0, 9).unwrap());
            asm.emit(a64::movz(1, u32::try_from(PAGE_SIZE).unwrap(), 0).unwrap());
            asm.emit(a64::movz(2, permission, 0).unwrap());
            asm.emit(svc(SvcId::SetMemoryPermission));
            asm.cbnz(0, fail).unwrap();
            asm.emit(a64::ldr(3, 9, 0).unwrap());
            asm.emit(a64::movz(5, VALUE, 0).unwrap());
            asm.emit(a64::sub(4, 3, 5).unwrap());
            asm.cbnz(4, fail).unwrap();
            second_store.set(asm.len());
            asm.emit(a64::str(5, 9, 0).unwrap());
            asm.emit(svc(SvcId::ExitProcess));
            asm.bind(fail).unwrap();
            asm.emit(a64::brk(1).unwrap());
            asm.finish().unwrap()
        });
        (file, offsets, second_store.get())
    }

    /// What a heap program does once its heap works.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub(crate) enum Then {
        /// Exits.
        Exit,
        /// Stores to the first word after the heap.
        StorePastTheEnd,
        /// Sets the heap's size to zero and loads from where the heap was.
        LoadAfterFreeing,
    }

    /// A program that sets its heap to [`HEAP_SIZE`] bytes, stores [`VALUE`] to the heap's
    /// first and last words and reads it back from each, and then does `then`. Any call that
    /// fails, and any value that does not read back, ends at `brk #1`.
    ///
    /// ```text
    ///       movz x1, #HEAP_SIZE >> 16, lsl #16
    ///       svc  #0x1
    ///       cbnz x0, fail
    ///       mov  x9, x1            ; the heap's address
    ///       movz x2, #VALUE
    ///       str  x2, [x9]
    ///       ldr  x3, [x9]
    ///       sub  x4, x2, x3
    ///       cbnz x4, fail
    ///       movz/movk x11, #-(HEAP_SIZE - 0x1000)
    ///       sub  x10, x9, x11      ; the heap's last page
    ///       str  x2, [x10, #0xff8]
    ///       ldr  x3, [x10, #0xff8]
    ///       sub  x4, x2, x3
    ///       cbnz x4, fail
    ///       ...                    ; then
    ///       svc  #0x7
    /// fail: brk  #1
    /// ```
    pub(crate) fn uses_a_heap(then: Then) -> Vec<u8> {
        let page = u32::try_from(PAGE_SIZE).unwrap();
        let last_word = page - a64::MEM_OFFSET_ALIGN;
        let halfword = |value: u64, shift: u32| {
            u32::try_from((value >> shift) & u64::from(a64::MAX_IMM16)).unwrap()
        };
        let mut asm = Assembler::new();
        let fail = asm.label();
        asm.emit(a64::movz(1, halfword(HEAP_SIZE, 16), 16).unwrap());
        asm.emit(svc(SvcId::SetHeapSize));
        asm.cbnz(0, fail).unwrap();
        asm.emit(a64::mov(9, 1).unwrap());
        asm.emit(a64::movz(2, VALUE, 0).unwrap());
        asm.emit(a64::str(2, 9, 0).unwrap());
        asm.emit(a64::ldr(3, 9, 0).unwrap());
        asm.emit(a64::sub(4, 2, 3).unwrap());
        asm.cbnz(4, fail).unwrap();
        // x10 = x9 + (HEAP_SIZE - one page), as a subtraction of its negative.
        let back = (HEAP_SIZE - PAGE_SIZE).wrapping_neg();
        asm.emit(a64::movz(11, halfword(back, 0), 0).unwrap());
        for shift in [16, 32, 48] {
            asm.emit(a64::movk(11, halfword(back, shift), shift).unwrap());
        }
        asm.emit(a64::sub(10, 9, 11).unwrap());
        asm.emit(a64::str(2, 10, last_word).unwrap());
        asm.emit(a64::ldr(3, 10, last_word).unwrap());
        asm.emit(a64::sub(4, 2, 3).unwrap());
        asm.cbnz(4, fail).unwrap();
        match then {
            Then::Exit => {}
            Then::StorePastTheEnd => asm.emit(a64::str(2, 10, page).unwrap()),
            Then::LoadAfterFreeing => {
                asm.emit(a64::movz(1, 0, 0).unwrap());
                asm.emit(svc(SvcId::SetHeapSize));
                asm.cbnz(0, fail).unwrap();
                asm.emit(a64::ldr(3, 9, 0).unwrap());
            }
        }
        asm.emit(svc(SvcId::ExitProcess));
        asm.bind(fail).unwrap();
        asm.emit(a64::brk(1).unwrap());
        NroBuilder::new(&asm.finish().unwrap()).build().unwrap()
    }

    /// A program with `bss` bytes of `.bss` that stores to the first word of each of the
    /// first `pages` pages of it, and exits.
    ///
    /// ```text
    ///       adr  x0, bss
    ///       movz/movk x1, #pages
    ///       movz/movk x2, #-0x1000
    /// loop: str  x1, [x0]
    ///       sub  x0, x0, x2        ; the next page
    ///       sub  x1, x1, #1
    ///       cbnz x1, loop
    ///       svc  #0x7
    /// ```
    pub(crate) fn touches_bss(bss: u32, pages: u32) -> Vec<u8> {
        let halfword = |value: u64, shift: u32| {
            u32::try_from((value >> shift) & u64::from(a64::MAX_IMM16)).unwrap()
        };
        let code = |bss_offset: u64| {
            let mut asm = Assembler::new();
            asm.emit(a64::adr(0, distance(bss_offset, asm.len())).unwrap());
            asm.emit(a64::movz(1, halfword(u64::from(pages), 0), 0).unwrap());
            asm.emit(a64::movk(1, halfword(u64::from(pages), 16), 16).unwrap());
            let step = PAGE_SIZE.wrapping_neg();
            asm.emit(a64::movz(2, halfword(step, 0), 0).unwrap());
            for shift in [16, 32, 48] {
                asm.emit(a64::movk(2, halfword(step, shift), shift).unwrap());
            }
            let again = asm.label();
            asm.bind(again).unwrap();
            asm.emit(a64::str(1, 0, 0).unwrap());
            asm.emit(a64::sub(0, 0, 2).unwrap());
            asm.emit(a64::sub_imm(1, 1, 1).unwrap());
            asm.cbnz(1, again).unwrap();
            asm.emit(svc(SvcId::ExitProcess));
            asm.finish().unwrap()
        };
        // `.bss` follows the image, which is one page of text here whatever the code says.
        let image = NroBuilder::new(&code(0))
            .bss_size(bss)
            .build()
            .unwrap()
            .len() as u64;
        NroBuilder::new(&code(image)).bss_size(bss).build().unwrap()
    }
}

/// Where the kernel put the heap region of a run: the layout for the module in the run's
/// memory map, on the console every program gets.
#[cfg(test)]
fn heap_region_start(report: &RunReport) -> u64 {
    let module: Vec<_> = report
        .memory_map
        .iter()
        .filter(|mapping| mapping.name.starts_with("main."))
        .collect();
    let base = image_base(report);
    let end = module.iter().map(|mapping| mapping.end).max().unwrap();
    Regions::new(&LayoutRequest {
        code_address: base,
        code_size: end.next_multiple_of(PAGE_SIZE) - base,
        physical_memory_size: MemoryArrangement::FourGb.physical_memory_size(),
        alias_region_extra: false,
        offsets: RegionOffsets::default(),
    })
    .unwrap()
    .heap
    .start
}

#[test]
fn a_store_to_a_page_made_read_only_faults_at_that_page_and_a_load_still_works() {
    let (file, at, second_store) = memory_calls::reprotects_its_data(memory_calls::PERM_READ);
    let Some(report) = run(&file, GENEROUS) else {
        return;
    };
    // The first store, the call and the load after it all worked (anything else ends at the
    // program's `brk`, an undefined-instruction crash); the second store is the crash.
    let crash = report.crash().expect("a crash report");
    let data = image_base(&report) + at.data;
    assert_eq!(
        crash.exception,
        Exception::DataAbort {
            address: data,
            write: true
        },
        "{}",
        crash.to_text()
    );
    assert_eq!(
        crash.pc,
        image_base(&report) + permissions::code_offset(second_store)
    );
    assert_eq!(crash.fault_mapping.as_deref(), Some("main.data"));
    // The one call the program made, with the page it named, succeeded.
    assert_eq!(
        events(&report)
            .into_iter()
            .filter(|kind| matches!(kind, EventKind::Svc { .. }))
            .collect::<Vec<_>>(),
        [EventKind::Svc {
            id: u32::from(SvcId::SetMemoryPermission.number()),
            args: [
                data,
                PAGE_SIZE,
                u64::from(memory_calls::PERM_READ),
                0,
                0,
                0,
                0,
                0
            ],
            result: starquake_trace::ResultCode(0),
        }]
    );
    assert!(report.stubbed.is_empty(), "{:?}", report.stubbed);
}

#[test]
fn the_same_program_with_the_page_left_writable_exits_normally() {
    // The control for the test above: the only difference is the permission the program asks
    // for, so the crash there is the call's doing.
    let (file, _, _) = memory_calls::reprotects_its_data(memory_calls::PERM_READ_WRITE);
    let Some(report) = run(&file, GENEROUS) else {
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
    assert_eq!(
        report.system_calls.iter().collect::<Vec<_>>(),
        [
            (&"ExitProcess".to_owned(), &1),
            (&"SetMemoryPermission".to_owned(), &1)
        ]
    );
}

#[test]
fn a_heap_can_be_read_and_written_from_its_first_word_to_its_last() {
    let file = memory_calls::uses_a_heap(memory_calls::Then::Exit);
    let Some(report) = run(&file, GENEROUS) else {
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
    assert_eq!(
        report.system_calls.iter().collect::<Vec<_>>(),
        [
            (&"ExitProcess".to_owned(), &1),
            (&"SetHeapSize".to_owned(), &1)
        ]
    );
}

#[test]
fn the_first_word_after_a_heap_and_a_heap_that_was_freed_are_not_there() {
    use memory_calls::{HEAP_SIZE, Then};
    for (then, offset, write) in [
        (Then::StorePastTheEnd, HEAP_SIZE, true),
        (Then::LoadAfterFreeing, 0, false),
    ] {
        let Some(report) = run(&memory_calls::uses_a_heap(then), GENEROUS) else {
            return;
        };
        let crash = report.crash().expect("a crash report");
        // The heap worked up to here, at the address the kernel answered: the fault is where
        // the layout puts the heap region, plus the offset.
        let address = heap_region_start(&report) + offset;
        assert_eq!(
            crash.exception,
            Exception::DataAbort { address, write },
            "{then:?}: {}",
            crash.to_text()
        );
        assert_eq!(crash.fault_mapping, None, "{then:?}");
        // Freeing took a second call; both succeeded.
        let calls = if then == Then::LoadAfterFreeing { 2 } else { 1 };
        assert_eq!(
            report.system_calls.get("SetHeapSize"),
            Some(&calls),
            "{then:?}"
        );
        let results: Vec<u32> = events(&report)
            .into_iter()
            .filter_map(|kind| match kind {
                EventKind::Svc { result, .. } => Some(result.0),
                _ => None,
            })
            .collect();
        assert_eq!(
            results,
            vec![0; usize::try_from(calls).unwrap()],
            "{then:?}"
        );
    }
}

#[test]
fn memory_a_program_never_touches_needs_no_guest_ram() {
    /// More `.bss` than the backend's guest RAM holds (16 MiB); a whole number of pages.
    const BSS: u32 = 64 << 20;
    // Touching a few pages of it is fine.
    let Some(report) = run(&memory_calls::touches_bss(BSS, 8), GENEROUS) else {
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

    // Touching all of it runs out of guest RAM, which is the emulator's failure (PROVISIONAL:
    // the kernel's accounting lets an application use far more than the backend backs). The
    // run names the page that could not be backed: one of `.bss`, on a page boundary.
    let pages = BSS / u32::try_from(PAGE_SIZE).unwrap();
    let report = run(&memory_calls::touches_bss(BSS, pages), GENEROUS).unwrap();
    assert_eq!(report.exit_code(), exit_code::EMULATOR_ERROR);
    let ExitReason::EmulatorError(message) = &report.exit else {
        unreachable!("{:?}", report.exit);
    };
    let out_of_ram = starquake_core::SpaceError::OutOfRam {
        size: PAGE_SIZE,
        available: 0,
    };
    let address = message
        .strip_prefix("guest memory at 0x")
        .and_then(|rest| rest.strip_suffix(&format!(" could not be backed: {out_of_ram}")))
        .and_then(|digits| u64::from_str_radix(digits, 16).ok())
        .unwrap_or_else(|| unreachable!("{message}"));
    let bss = report
        .memory_map
        .iter()
        .find(|mapping| mapping.name == "main.bss")
        .expect("a .bss mapping");
    assert!(bss.contains(address), "{address:#x}");
    assert!(address.is_multiple_of(PAGE_SIZE), "{address:#x}");
    // Not the first page: RAM held what came before it.
    assert!(address > bss.start);
}

#[test]
fn runs_in_one_process_do_not_disturb_each_other() {
    // Each run makes a fresh VM at the same guest addresses. A stale instruction or leftover
    // memory from the run before would change what the next one does.
    let hello = programs::hello().unwrap();
    let Some(first) = run(&hello, GENEROUS) else {
        return;
    };
    let crash = run(&programs::crash().unwrap().nro, GENEROUS).unwrap();
    let again = run(&hello, GENEROUS).unwrap();
    assert_eq!(crash.exit_code(), exit_code::GUEST_CRASHED);
    assert_eq!(first.exit, ExitReason::ProcessExited);
    assert_eq!(again.exit, ExitReason::ProcessExited);
    assert_eq!(first.guest_log, again.guest_log);
    assert_eq!(first.trace, again.trace);
    assert!(output_is_expected(&again));
}
