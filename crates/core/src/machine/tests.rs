//! The core loop driven by a scripted vCPU, on any host: the scripted guest makes the exits a
//! real program would, against a real placed image in flat memory.

use std::cell::Cell;
use std::rc::Rc;
use std::time::{Duration, Instant};

use starquake_cpu::testing::ScriptedVcpu;
use starquake_cpu::{Exit, FaultStatus, Interrupter, Syndrome, Vcpu};
use starquake_kernel::result::KernelError;
use starquake_kernel::svc::SvcId;
use starquake_memory::{Fault, FlatMemory};
use starquake_testkit::programs;
use starquake_trace::{
    Dump, DumpError, Env, EnvRecord, Event, EventKind, FaultAction, FixedClock, HostNanos,
    LifecyclePhase, SystemClock,
};

use super::*;
use crate::clock::PROCESS_COUNTER_RATE;
use crate::crash::Location;
use crate::image::{Program, place};
use crate::report::{empty_dump, exit_code};

/// Where a real run puts the program.
const RAM_BASE: u64 = crate::image::IMAGE_BASE;
/// Room for a small image, its TLS block and the stack.
const RAM_SIZE: usize = 2 << 20;

/// The time a scripted run counts with: the rate every process has for now, and the host's
/// monotonic clock. Nothing here reads a counter in the guest, so the clock's zero is free.
fn timebase() -> Timebase<SystemClock> {
    Timebase {
        rate: PROCESS_COUNTER_RATE,
        clock: SystemClock::new(),
    }
}

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
            timebase(),
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

/// The syndrome a real vCPU reports for a store to an address nothing maps, missing at the
/// first level of the walk.
const UNMAPPED_STORE: Syndrome = Syndrome::data_abort(FaultStatus::Translation { level: 1 }, true);

/// Where the tests put strings for the guest to print: free memory between the image and the
/// stack.
const SCRATCH: u64 = RAM_BASE + 0x8_0000;

/// Guest memory whose every read takes a while, so two clock readings on either side of one
/// cannot agree by accident.
struct SlowMemory<'a>(&'a FlatMemory);

impl GuestMemory for SlowMemory<'_> {
    fn read(&self, va: u64, buf: &mut [u8]) -> Result<(), Fault> {
        std::thread::sleep(Duration::from_millis(2));
        self.0.read(va, buf)
    }

    fn write(&self, va: u64, data: &[u8]) -> Result<(), Fault> {
        self.0.write(va, data)
    }
}

/// A script that prints each of `lines` with `OutputDebugString` and then exits. The lines are
/// written back to back at [`SCRATCH`] in `memory`.
fn printing(memory: &FlatMemory, lines: &[&[u8]]) -> ScriptedVcpu {
    let mut vcpu = ScriptedVcpu::new();
    let mut at = SCRATCH;
    for line in lines {
        memory.write(at, line).unwrap();
        let (address, len) = (at, line.len() as u64);
        vcpu = vcpu.then(
            move |regs| {
                regs.x[0] = address;
                regs.x[1] = len;
            },
            svc(SvcId::OutputDebugString),
        );
        at += len;
    }
    vcpu.then(|_| {}, svc(SvcId::ExitProcess))
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
    // And the rate the example's counter program expects is the one the kernel gives a process.
    assert_eq!(programs::COUNTER_HZ, PROCESS_COUNTER_RATE.hz());
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
        // ExitProcess takes no arguments: what the calls before it left in x0 and x1 is not
        // part of the event.
        call(exit, 0, 0),
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
            syndrome: UNMAPPED_STORE,
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
fn an_instruction_abort_is_a_crash_that_names_the_address_fetched_from() {
    // A jump into the stack, which is mapped but not executable.
    let guest = Guest::new(&programs::hello().unwrap());
    let address = guest.start_of("stack") + 0x40;
    let vcpu = ScriptedVcpu::new().then(
        move |regs| regs.pc = address,
        Exit::InstructionAbort {
            address,
            syndrome: Syndrome::instruction_abort(FaultStatus::Permission { level: 3 }),
        },
    );
    let report = guest.run(vcpu, &RunOptions::default());
    assert_eq!(report.exit_code(), exit_code::GUEST_CRASHED);
    let crash = report.crash().expect("a crash report");
    assert_eq!(crash.exception, Exception::InstructionAbort { address });
    assert_eq!(crash.pc, address);
    assert_eq!(crash.pc_location, None);
    assert_eq!(crash.fault_mapping.as_deref(), Some("stack"));
    // The fault is in the event stream, as a read.
    assert_eq!(
        recorded(&report).last().copied(),
        Some(EventKind::Fault {
            address,
            write: false,
            action: FaultAction::Fatal,
        })
    );
}

#[test]
fn an_undefined_instruction_is_a_crash_at_its_address() {
    let guest = Guest::new(&programs::hello().unwrap());
    let pc = guest.placed.entry + 0x40;
    let vcpu = ScriptedVcpu::new().then(
        move |regs| regs.pc = pc,
        Exit::Undefined {
            pc,
            syndrome: Syndrome::undefined(),
        },
    );
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

#[test]
fn a_log_line_and_the_call_that_wrote_it_carry_one_time() {
    let guest = Guest::new(&programs::hello().unwrap());
    let mut vcpu = printing(&guest.memory, &[b"one", b"two"]);
    // Reading the string takes 2 ms, far longer than a tick of the event clock: a second clock
    // reading for the log line would be later than the call's.
    let report = run(
        &mut vcpu,
        &SlowMemory(&guest.memory),
        &guest.placed,
        &RunOptions::default(),
        Instant::now(),
        timebase(),
    );
    assert_eq!(report.guest_log, [b"one".to_vec(), b"two".to_vec()]);
    let events = Dump::from_bytes(&report.events).unwrap().merged();
    let mut pairs = 0;
    for pair in events.windows(2) {
        if matches!(pair[0].kind, EventKind::GuestLog { .. }) {
            assert!(matches!(pair[1].kind, EventKind::Svc { .. }));
            assert_eq!(pair[0].time, pair[1].time, "{pair:?}");
            pairs += 1;
        }
    }
    assert_eq!(pairs, 2);
    // Time still moves from one call to the next.
    let times: Vec<_> = events.iter().map(|event| event.time).collect();
    assert!(times.is_sorted(), "{times:?}");
    assert!(times.first() < times.last(), "{times:?}");
}

#[test]
fn get_system_tick_is_the_clock_at_the_rate_the_run_was_given() {
    // A clock that reads a whole number of seconds, so the expected ticks are exact, and both
    // console rates: the run's rate reaches the kernel, and the kernel's conversion reaches the
    // guest.
    const SECONDS: u64 = 1_000_003;
    let start = SECONDS * starquake_kernel::time::NANOS_PER_SECOND;
    for rate in [CounterRate::SWITCH_1, CounterRate::SWITCH_2] {
        let guest = Guest::new(&programs::hello().unwrap());
        let tick = Rc::new(Cell::new(0));
        let seen = Rc::clone(&tick);
        // The second step's closure sees what the first call wrote to x0.
        let mut vcpu = ScriptedVcpu::new()
            .then(|_| {}, svc(SvcId::GetSystemTick))
            .then(move |regs| seen.set(regs.x[0]), svc(SvcId::ExitProcess));
        let report = run(
            &mut vcpu,
            &guest.memory,
            &guest.placed,
            &RunOptions::default(),
            Instant::now(),
            Timebase {
                rate,
                clock: FixedClock::new(start, 1),
            },
        );
        assert_eq!(report.exit, ExitReason::ProcessExited);
        // The environment's first reading is the clock's first: guest tick zero is its zero.
        assert_eq!(tick.get(), rate.ticks_at(HostNanos(start)).0);
        assert_eq!(tick.get(), SECONDS * rate.hz());
        // The call's event carries the same guest time.
        let events = Dump::from_bytes(&report.events).unwrap().merged();
        let stamped: Vec<u64> = events
            .iter()
            .filter(|event| {
                matches!(event.kind, EventKind::Svc { id, .. } if id == u32::from(SvcId::GetSystemTick.number()))
            })
            .map(|event| event.time.0)
            .collect();
        assert_eq!(stamped, [tick.get()]);
    }
}

#[test]
fn the_environments_recording_is_dropped_as_it_is_made() {
    // Every dispatch reads the environment's clock once, and a live environment keeps every
    // answer it gives. Nothing in a run uses that recording, so the loop must not let it grow
    // with the number of system calls the guest makes.
    const CALLS: u64 = 1_000;
    let guest = Guest::new(&programs::hello().unwrap());
    let options = RunOptions::default();
    let session = Session {
        memory: &guest.memory,
        placed: &guest.placed,
        options: &options,
        clock: RunClock::new(timebase()),
        recorder: Recorder::new(),
    };
    let mut log = GuestLog {
        lines: Vec::new(),
        kept_bytes: 0,
        limit: Limits::DEFAULT.guest_log_bytes,
        dropped: 0,
        session: &session,
    };
    let mut tally = Tally::default();
    let env = session.environment();
    let mut vcpu = ScriptedVcpu::new();
    for _ in 0..CALLS {
        vcpu = vcpu.then(|_| {}, svc(SvcId::GetSystemTick));
    }
    let mut vcpu = vcpu.then(|_| {}, svc(SvcId::ExitProcess));

    let ending = drive(&mut vcpu, &session, &env, &mut log, &mut tally);
    assert!(matches!(ending, Ending::Exited));
    // The calls were made, through this environment...
    assert_eq!(
        tally.calls,
        BTreeMap::from([
            ("ExitProcess".to_owned(), 1),
            ("GetSystemTick".to_owned(), CALLS)
        ])
    );
    // ...and none of their clock readings is still held. (The count, not the records: a
    // failure would otherwise print one record per call.)
    assert_eq!(env.recording().len(), 0);
    // It is an environment that records: one more reading is one record.
    let reading = env.now().unwrap();
    assert_eq!(env.recording(), [EnvRecord::Clock(reading)]);
}

#[test]
fn a_crash_report_quotes_the_newest_events_and_no_more_than_its_window() {
    let guest = Guest::new(&programs::hello().unwrap());
    // More calls than the window holds, then the crash.
    let calls = LAST_EVENTS + 8;
    let mut vcpu = ScriptedVcpu::new();
    for _ in 0..calls {
        vcpu = vcpu.then(|_| {}, svc(SvcId::GetCurrentProcessorNumber));
    }
    let address = programs::CRASH_ADDRESS;
    let vcpu = vcpu.then(
        |_| {},
        Exit::DataAbort {
            address,
            write: true,
            syndrome: UNMAPPED_STORE,
        },
    );
    let report = guest.run(vcpu, &RunOptions::default());
    // The process start, the calls, and the fault.
    let total = calls + 2;
    assert_eq!(recorded(&report).len(), total);
    let quoted = &report.crash().expect("a crash report").last_events;
    assert_eq!(quoted.len(), LAST_EVENTS);
    // The window ends at the newest event, the fault, and holds the events just before it in
    // order.
    let fault = EventKind::Fault {
        address,
        write: true,
        action: FaultAction::Fatal,
    };
    assert_eq!(
        quoted.last(),
        Some(&format!("#{} core 0 thread 1: {fault}", total - 1))
    );
    for (line, seq) in quoted.iter().zip(total - LAST_EVENTS..total) {
        assert!(line.starts_with(&format!("#{seq} core 0 ")), "{line}");
    }
}

#[test]
fn the_trace_and_the_guest_log_stop_growing_at_their_limits() {
    let guest = Guest::new(&programs::hello().unwrap());
    let lines: [&[u8]; 4] = [b"aaaa", b"bbbbbb", b"cc", b"d"];
    // The first two lines are exactly the byte limit. The third does not fit, and the fourth,
    // which would, comes after a dropped line, so it is not kept either: what is kept is a
    // prefix of what the guest wrote.
    let limits = Limits {
        trace_events: 4,
        guest_log_bytes: lines[0].len() + lines[1].len(),
    };
    let mut vcpu = printing(&guest.memory, &lines);
    let report = run_within(
        &mut vcpu,
        &guest.memory,
        &guest.placed,
        &RunOptions::default(),
        Instant::now(),
        timebase(),
        limits,
    );
    assert_eq!(report.exit, ExitReason::ProcessExited);
    assert_eq!(report.guest_log, [lines[0].to_vec(), lines[1].to_vec()]);
    assert_eq!(report.guest_log_dropped, 2);

    // Eleven events in all: the start, a log line and a call for each of the four lines, the
    // exit call and the exit. The flight recorder has every one.
    let events = recorded(&report);
    assert_eq!(events.len(), 11);
    // The trace keeps the first four and says how many it left out.
    let kept: Vec<String> = events[..limits.trace_events]
        .iter()
        .map(|kind| format!("  {kind}\n"))
        .collect();
    assert_eq!(
        report.trace,
        format!(
            "thread 1\n{}truncated: 7 later events are not shown (the trace keeps its first 4)\n",
            kept.concat()
        )
    );
    assert_eq!(report.trace_events_dropped, 7);
    assert_eq!(report.summary().guest_log_lines, 4);
    assert_eq!(report.summary().guest_log_lines_dropped, 2);
    assert_eq!(report.summary().trace_events_dropped, 7);
    // Counts are of what the guest did, not of what was kept.
    assert_eq!(report.system_calls["OutputDebugString"], 4);
}

#[test]
fn a_run_within_its_limits_is_not_truncated() {
    let guest = Guest::new(&programs::hello().unwrap());
    let lines: [&[u8]; 2] = [b"aaaa", b"bbbbbb"];
    // Start, two lines with their calls, the exit call and the exit: seven events, ten bytes.
    let limits = Limits {
        trace_events: 7,
        guest_log_bytes: 10,
    };
    let mut vcpu = printing(&guest.memory, &lines);
    let report = run_within(
        &mut vcpu,
        &guest.memory,
        &guest.placed,
        &RunOptions::default(),
        Instant::now(),
        timebase(),
        limits,
    );
    assert_eq!(report.guest_log, [lines[0].to_vec(), lines[1].to_vec()]);
    assert_eq!(
        (report.guest_log_dropped, report.trace_events_dropped),
        (0, 0)
    );
    assert!(!report.trace.contains("truncated"), "{}", report.trace);
    assert_eq!(report.trace.lines().count(), 1 + limits.trace_events);
    // The limits of a real run are the documented ones.
    assert_eq!(
        Limits::DEFAULT,
        Limits {
            trace_events: 100_000,
            guest_log_bytes: 4 << 20
        }
    );
}

#[test]
fn a_flight_recorder_that_cannot_be_written_out_is_reported_however_the_run_ended() {
    let lost = || Err(DumpError::BadMagic);
    let message = format!(
        "the flight recorder could not be written out: {}",
        DumpError::BadMagic
    );
    // A clean exit does not stay one.
    assert_eq!(
        settle_dump(lost(), ExitReason::ProcessExited),
        (
            ExitReason::EmulatorError(message.clone()),
            empty_dump(),
            Some(message.clone())
        )
    );
    // A timeout, a crash and an emulator error keep their reason, and the loss is still said.
    let crash = ExitReason::GuestCrashed(Box::new(CrashReport::new(
        Exception::UndefinedInstruction,
        &GuestRegs::default(),
        &[],
        &[],
        Vec::new(),
        &[],
    )));
    for exit in [
        ExitReason::Timeout,
        crash,
        ExitReason::EmulatorError("the vCPU failed".into()),
    ] {
        assert_eq!(
            settle_dump(lost(), exit.clone()),
            (exit, empty_dump(), Some(message.clone()))
        );
    }
    // A dump that was written is passed through.
    assert_eq!(
        settle_dump(Ok(vec![1, 2, 3]), ExitReason::Timeout),
        (ExitReason::Timeout, vec![1, 2, 3], None)
    );
    // The file is still a readable dump, with nothing in it.
    assert_eq!(Dump::from_bytes(&empty_dump()), Ok(Dump::new(Vec::new())));
}
