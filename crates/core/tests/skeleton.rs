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
use starquake_kernel::svc::tick::CNTFREQ;
use starquake_testkit::nro::NroBuilder;
use starquake_testkit::{a64, programs};
use starquake_trace::{Dump, EventKind};

/// What the hello program must print, one `OutputDebugString` call per line.
const EXPECTED_OUTPUT: [&str; 2] = ["hello from the guest", "starquake walking skeleton"];

/// What the control program prints: the same two lines with the last letter changed.
const CONTROL_OUTPUT: [&str; 2] = ["hello from the guest", "starquake walking skeletoN"];

/// The committed snapshot of the hello program's trace, as `insta` stores it: a header between
/// two `---` lines, then the text.
const HELLO_SNAPSHOT: &str = include_str!("snapshots/skeleton__hello_trace.snap");

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
    let counter = programs::counter(CNTFREQ).unwrap();
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
    let counter = programs::counter(CNTFREQ + 1).unwrap();
    let Some(report) = run(&counter.nro, GENEROUS) else {
        return;
    };
    assert_eq!(report.exit_code(), exit_code::GUEST_CRASHED);
    let crash = report.crash().expect("a crash report");
    assert_eq!(crash.exception, Exception::UndefinedInstruction);
    assert_eq!(crash.pc, image_base(&report) + counter.wrong_rate_offset);
    // x23 is what `cntfrq_el0` read through the patched site, x24 what the program expected.
    assert_eq!(crash.registers.x[23], CNTFREQ);
    assert_eq!(crash.registers.x[24], CNTFREQ + 1);
    // The two counter reads before it passed their checks: the second is above the first.
    assert!(
        crash.registers.x[20] > crash.registers.x[19],
        "{:#x?}",
        crash.registers.x
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
    /// More `.bss` than the backend's bring-up RAM holds; a whole number of pages.
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
