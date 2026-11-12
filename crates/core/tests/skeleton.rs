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
use starquake_core::{Error, ExitReason, LoadError, RunOptions, RunReport, System, exit_code};
use starquake_kernel::svc::SvcId;
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

/// Long enough for any of these programs on a loaded machine; they finish in milliseconds.
const GENEROUS: Duration = Duration::from_secs(30);

/// Shared setup. In a `cfg(test)` module so it may unwrap like the tests do.
#[cfg(test)]
mod support {
    use super::*;

    /// Runs an NRO to completion, or returns `None` on hosted CI.
    pub(crate) fn run(file: &[u8], timeout: Duration) -> Option<RunReport> {
        if std::env::var_os("CI").is_some() {
            return None;
        }
        let options = RunOptions {
            timeout: Some(timeout),
            command_line: Vec::new(),
        };
        Some(System::new(file).expect("load the program").run(&options))
    }

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

use support::{events, golden_trace, image_base, output, output_is_expected, run, trace_is_golden};

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
fn a_program_too_large_for_guest_memory_is_an_emulator_error() {
    /// More `.bss` than the backend's guest RAM holds; a whole number of pages.
    const BSS: u32 = 64 << 20;
    let file = NroBuilder::new(&[a64::nop()])
        .bss_size(BSS)
        .build()
        .unwrap();
    // What the layout costs is read off a run that fits, not restated: the TLS block and the
    // stack are in its memory map, and guest RAM runs from the image to the end of the stack.
    let Some(fits) = run(&programs::hello().unwrap(), GENEROUS) else {
        return;
    };
    let mapping = |name: &str| {
        fits.memory_map
            .iter()
            .find(|mapping| mapping.name == name)
            .expect("the mapping is in the memory map")
    };
    let size = |name: &str| mapping(name).end - mapping(name).start;
    let available = mapping("stack").end - image_base(&fits);
    // The image is already whole pages, as the builder writes it.
    let needed = file.len() as u64 + u64::from(BSS) + size("tls") + size("stack");
    assert!(needed > available);

    let report = run(&file, GENEROUS).unwrap();
    let error = Error::Load(LoadError::DoesNotFit { needed, available });
    assert_eq!(report.exit, ExitReason::EmulatorError(error.to_string()));
    assert_eq!(report.exit_code(), exit_code::EMULATOR_ERROR);
    assert_eq!(report.trace, "");
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
