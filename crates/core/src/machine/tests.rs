//! The core loop driven by a scripted vCPU, on any host: the scripted guest makes the exits a
//! real program would, against a real placed image in flat memory.

use std::time::{Duration, Instant};

use starquake_cpu::testing::ScriptedVcpu;
use starquake_cpu::{Exit, Interrupter, Vcpu};
use starquake_kernel::result::KernelError;
use starquake_kernel::svc::SvcId;
use starquake_memory::FlatMemory;
use starquake_testkit::programs;
use starquake_trace::{Dump, Event, EventKind, FaultAction, LifecyclePhase};

use super::*;
use crate::crash::Location;
use crate::image::{Program, place};
use crate::report::exit_code;

const RAM_BASE: u64 = 0x8000_0000;
/// Room for a small image, its TLS block and the stack.
const RAM_SIZE: usize = 2 << 20;

/// A program placed in flat memory.
struct Guest {
    memory: FlatMemory,
    placed: Placement,
}

impl Guest {
    fn new(file: &[u8]) -> Self {
        let program = Program::parse(file).unwrap();
        let placed = place(&program, RAM_BASE, RAM_SIZE as u64, None).unwrap();
        let memory = FlatMemory::new(RAM_BASE, RAM_SIZE);
        placed.write_into(&memory).unwrap();
        Self { memory, placed }
    }

    /// Runs `vcpu` as this guest.
    fn run(&self, mut vcpu: ScriptedVcpu, options: &RunOptions) -> RunReport {
        run(
            &mut vcpu,
            &self.memory,
            &self.placed,
            options,
            Instant::now(),
        )
    }

    /// Where the mapping called `name` starts.
    fn start_of(&self, name: &str) -> u64 {
        self.placed
            .mappings
            .iter()
            .find(|mapping| mapping.name == name)
            .unwrap()
            .start
    }
}

fn svc(id: SvcId) -> Exit {
    Exit::Svc(id.number())
}

fn lifecycle(phase: LifecyclePhase) -> EventKind {
    EventKind::Lifecycle {
        phase,
        subject: PROCESS_ID,
    }
}

/// The kinds of the events in a report's flight-recorder dump, in order, after checking that
/// they are numbered from zero and stamped with the one core and thread.
fn recorded(report: &RunReport) -> Vec<EventKind> {
    let events = Dump::from_bytes(&report.events).unwrap().merged();
    for (index, event) in (0u64..).zip(&events) {
        let Event {
            seq, core, thread, ..
        } = event;
        assert_eq!((*seq, *core, *thread), (index, CORE, MAIN_THREAD));
    }
    events.iter().map(|event| event.kind).collect()
}

#[test]
fn the_guest_programs_use_the_kernels_system_call_numbers() {
    // The testkit restates the two numbers from the reference; this is the cross-check.
    assert_eq!(
        programs::SVC_EXIT_PROCESS,
        u32::from(SvcId::ExitProcess.number())
    );
    assert_eq!(
        programs::SVC_OUTPUT_DEBUG_STRING,
        u32::from(SvcId::OutputDebugString.number())
    );
}

#[test]
fn a_guest_that_prints_two_lines_and_exits_is_reported_in_full() {
    let guest = Guest::new(&programs::hello().unwrap());
    let [first, second] = programs::HELLO_LINES;
    let first_at = guest.start_of("main.rodata");
    let second_at = first_at + first.len() as u64;
    let (first_len, second_len) = (first.len() as u64, second.len() as u64);
    let (entry, stack_top, tls) = (guest.placed.entry, guest.placed.stack_top, guest.placed.tls);

    let vcpu = ScriptedVcpu::new()
        .then(
            move |regs| {
                // What the loop starts the thread with.
                assert_eq!((regs.pc, regs.sp, regs.tpidrro), (entry, stack_top, tls));
                assert_eq!((regs.x, regs.tpidr, regs.pstate), ([0; 31], 0, 0));
                regs.x[0] = first_at;
                regs.x[1] = first_len;
            },
            svc(SvcId::OutputDebugString),
        )
        .then(
            move |regs| {
                // The call's result came back in x0, and x1 was left alone.
                assert_eq!((regs.x[0], regs.x[1]), (0, first_len));
                regs.x[0] = second_at;
                regs.x[1] = second_len;
            },
            svc(SvcId::OutputDebugString),
        )
        .then(
            move |regs| assert_eq!((regs.x[0], regs.x[1]), (0, second_len)),
            svc(SvcId::ExitProcess),
        );
    let report = guest.run(vcpu, &RunOptions::default());

    assert_eq!(report.exit, ExitReason::ProcessExited);
    assert_eq!(report.exit_code(), exit_code::STOP_CONDITION);
    assert_eq!(report.guest_log, [first.to_vec(), second.to_vec()]);
    let output = u32::from(SvcId::OutputDebugString.number());
    let exit = u32::from(SvcId::ExitProcess.number());
    let call = |id, x0, x1| EventKind::Svc {
        id,
        args: [x0, x1, 0, 0, 0, 0, 0, 0],
        result: starquake_trace::ResultCode(0),
    };
    let expected = [
        lifecycle(LifecyclePhase::ProcessStart),
        EventKind::guest_log(first),
        call(output, first_at, first_len),
        EventKind::guest_log(second),
        call(output, second_at, second_len),
        call(exit, 0, second_len),
        lifecycle(LifecyclePhase::ProcessExit),
    ];
    assert_eq!(recorded(&report), expected);
    // The trace is the same events, one per line, under the one thread.
    let lines: Vec<String> = std::iter::once("thread 1".to_owned())
        .chain(expected.iter().map(|kind| format!("  {kind}")))
        .collect();
    assert_eq!(report.trace, lines.join("\n") + "\n");
    assert_eq!(
        report.system_calls,
        BTreeMap::from([
            ("ExitProcess".to_owned(), 1),
            ("OutputDebugString".to_owned(), 2)
        ])
    );
    assert_eq!(report.stubbed, BTreeSet::new());
    assert_eq!(report.memory_map, guest.placed.mappings);
    assert_eq!(report.patched_counter_reads, Vec::<u64>::new());
}

#[test]
fn an_unimplemented_call_returns_not_implemented_and_is_listed_as_stubbed() {
    let guest = Guest::new(&programs::hello().unwrap());
    let not_implemented = u64::from(KernelError::NotImplemented.code().raw());
    // A number in the gap after GetResourceLimitPeakValue, which Horizon does not list.
    let unlisted = SvcId::GetResourceLimitPeakValue.number() + 1;
    assert_eq!(SvcId::from_number(unlisted), None);
    let vcpu = ScriptedVcpu::new()
        .then(|_| {}, svc(SvcId::Break))
        .then(
            move |regs| assert_eq!(regs.x[0], not_implemented),
            svc(SvcId::Break),
        )
        .then(|_| {}, Exit::Svc(unlisted))
        .then(
            move |regs| assert_eq!(regs.x[0], not_implemented),
            svc(SvcId::ExitProcess),
        );
    let report = guest.run(vcpu, &RunOptions::default());
    assert_eq!(report.exit, ExitReason::ProcessExited);
    assert_eq!(
        report.stubbed,
        BTreeSet::from([
            format!("Break ({:#x})", SvcId::Break.number()),
            format!("{unlisted:#x}")
        ])
    );
    assert_eq!(
        report.system_calls,
        BTreeMap::from([
            ("Break".to_owned(), 2),
            ("ExitProcess".to_owned(), 1),
            (format!("{unlisted:#x}"), 1)
        ])
    );
}

#[test]
fn a_data_abort_is_a_crash_that_names_the_address_and_the_instruction() {
    let crash = programs::crash().unwrap();
    let guest = Guest::new(&crash.nro);
    let instruction = guest.placed.entry + crash.store_offset;
    let address = crash.fault_address;
    let vcpu = ScriptedVcpu::new().then(
        move |regs| {
            regs.x[0] = address - 8;
            regs.x[1] = 0x1234;
            regs.pc = instruction;
        },
        Exit::DataAbort {
            address,
            write: true,
        },
    );
    let options = RunOptions {
        timeout: None,
        command_line: vec!["starquake".into(), "run".into(), "crash.nro".into()],
    };
    let report = guest.run(vcpu, &options);

    assert_eq!(report.exit_code(), exit_code::GUEST_CRASHED);
    let report_crash = report.crash().expect("a crash report");
    assert_eq!(
        report_crash.exception,
        Exception::DataAbort {
            address,
            write: true
        }
    );
    assert_eq!(report_crash.pc, instruction);
    assert_eq!(
        report_crash.pc_location,
        Some(Location {
            module: "main".into(),
            offset: crash.store_offset
        })
    );
    assert_eq!(report_crash.fault_mapping, None);
    assert_eq!(report_crash.registers.x[..2], [address - 8, 0x1234]);
    assert_eq!(report_crash.registers.sp, guest.placed.stack_top);
    assert_eq!(report_crash.command_line, options.command_line);
    assert_eq!(report_crash.memory_map, guest.placed.mappings);
    assert_eq!(
        report_crash.modules,
        std::slice::from_ref(&guest.placed.module)
    );
    // The fault is in the flight recorder and the crash report's last events, and not in the
    // golden trace, which leaves faults out.
    let fault = EventKind::Fault {
        address,
        write: true,
        action: FaultAction::Fatal,
    };
    assert_eq!(
        recorded(&report),
        [lifecycle(LifecyclePhase::ProcessStart), fault]
    );
    assert_eq!(
        report_crash.last_events,
        [
            "#0 core 0 thread 1: lifecycle ProcessStart 1".to_owned(),
            format!("#1 core 0 thread 1: {fault}")
        ]
    );
    assert_eq!(report.trace, "thread 1\n  lifecycle ProcessStart 1\n");
}

#[test]
fn an_undefined_instruction_is_a_crash_at_its_address() {
    let guest = Guest::new(&programs::hello().unwrap());
    let pc = guest.placed.entry + 0x40;
    let vcpu = ScriptedVcpu::new().then(move |regs| regs.pc = pc, Exit::Undefined { pc });
    let report = guest.run(vcpu, &RunOptions::default());
    assert_eq!(report.exit_code(), exit_code::GUEST_CRASHED);
    let crash = report.crash().expect("a crash report");
    assert_eq!(crash.exception, Exception::UndefinedInstruction);
    assert_eq!(crash.pc, pc);
    assert_eq!(
        crash.pc_location.as_ref().map(|location| location.offset),
        Some(0x40)
    );
}

#[test]
fn a_guest_that_never_stops_is_ended_by_the_timeout() {
    let guest = Guest::new(&programs::endless_loop().unwrap());
    let timeout = Duration::from_millis(30);
    let options = RunOptions {
        timeout: Some(timeout),
        command_line: Vec::new(),
    };
    let report = guest.run(ScriptedVcpu::new().then_spin(), &options);
    assert_eq!(report.exit, ExitReason::Timeout);
    assert_eq!(report.exit_code(), exit_code::TIMEOUT);
    assert!(report.duration >= timeout, "{:?}", report.duration);
    assert_eq!(report.trace, "thread 1\n  lifecycle ProcessStart 1\n");
}

#[test]
fn a_run_that_ends_before_its_timeout_does_not_wait_for_it() {
    let guest = Guest::new(&programs::hello().unwrap());
    let options = RunOptions {
        timeout: Some(Duration::from_secs(3600)),
        command_line: Vec::new(),
    };
    let vcpu = ScriptedVcpu::new().then(|_| {}, svc(SvcId::ExitProcess));
    let started = Instant::now();
    let report = guest.run(vcpu, &options);
    assert_eq!(report.exit, ExitReason::ProcessExited);
    assert!(started.elapsed() < Duration::from_secs(60));
}

#[test]
fn an_interrupt_that_is_not_the_timeout_resumes_the_guest() {
    let guest = Guest::new(&programs::hello().unwrap());
    let vcpu = ScriptedVcpu::new().then(|_| {}, svc(SvcId::ExitProcess));
    vcpu.interrupter().interrupt().unwrap();
    let report = guest.run(vcpu, &RunOptions::default());
    assert_eq!(report.exit, ExitReason::ProcessExited);
}

#[test]
fn a_failing_vcpu_is_an_emulator_error() {
    let guest = Guest::new(&programs::hello().unwrap());
    // A script with no steps: the first run fails.
    let report = guest.run(ScriptedVcpu::new(), &RunOptions::default());
    assert_eq!(
        report.exit,
        ExitReason::EmulatorError(
            "the vCPU failed: host API error: the scripted vCPU has no more exits".into()
        )
    );
    assert_eq!(report.exit_code(), exit_code::EMULATOR_ERROR);
    assert_eq!(
        recorded(&report),
        [
            lifecycle(LifecyclePhase::ProcessStart),
            lifecycle(LifecyclePhase::EmulatorError)
        ]
    );
}

#[test]
fn a_string_the_guest_cannot_read_is_an_error_result_and_no_log_line() {
    let guest = Guest::new(&programs::hello().unwrap());
    let invalid = u64::from(KernelError::InvalidCurrentMemory.code().raw());
    let vcpu = ScriptedVcpu::new()
        .then(
            |regs| {
                // Below the start of guest memory.
                regs.x[0] = RAM_BASE - 0x10;
                regs.x[1] = 4;
            },
            svc(SvcId::OutputDebugString),
        )
        .then(
            move |regs| assert_eq!(regs.x[0], invalid),
            svc(SvcId::ExitProcess),
        );
    let report = guest.run(vcpu, &RunOptions::default());
    assert_eq!(report.exit, ExitReason::ProcessExited);
    assert_eq!(report.guest_log, Vec::<Vec<u8>>::new());
    assert!(
        report.trace.contains(&format!(
            "  svc 0x27({:#x}, 0x4) -> {invalid:#x}\n",
            RAM_BASE - 0x10
        )),
        "{}",
        report.trace
    );
}
