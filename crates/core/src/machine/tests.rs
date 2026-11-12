//! The core loop driven by a scripted vCPU, on any host: the scripted guest makes the exits a
//! real program would, against a real placed image in flat memory.

use std::cell::Cell;
use std::rc::Rc;
use std::sync::atomic::AtomicUsize;
use std::sync::{Mutex, PoisonError};
use std::time::{Duration, Instant};

use starquake_cpu::testing::ScriptedVcpu;
use starquake_cpu::{Exit, FaultStatus, Interrupter, Syndrome, Vcpu};
use starquake_kernel::handle::{Handle, MAX_TABLE_SIZE, MIN_LINEAR_ID};
use starquake_kernel::memory::GUARD_PAGES;
use starquake_kernel::memory::testing::RecordingMapper;
use starquake_kernel::result::KernelError;
use starquake_kernel::svc::SvcId;
use starquake_memory::{Fault, FlatMemory};
use starquake_testkit::programs;
use starquake_trace::{
    Dump, DumpError, Env, EnvRecord, Event, EventKind, FaultAction, FixedClock, HostNanos,
    LifecyclePhase, SystemClock,
};

use super::*;
use starquake_kernel::process::{Console, ProcessDescription};

use crate::clock::NRO_CONSOLE;
use crate::crash::{Location, SyndromeReport};
use crate::image::{Loaded, Program, load};
use crate::report::{empty_dump, exit_code};
use crate::space::SpaceError;

/// Where a real run puts the program, and where the tests' flat memory starts.
const RAM_BASE: u64 = crate::image::IMAGE_BASE;
/// Room for a small image and the strings the tests have the guest print. The stack and the
/// TLS block are elsewhere in the address space, where no scripted guest reaches.
const RAM_SIZE: usize = 2 << 20;

/// The time a scripted run counts with: the rate every process has for now, and the host's
/// monotonic clock. Nothing here reads a counter in the guest, so the clock's zero is free.
fn timebase() -> Timebase<SystemClock> {
    Timebase {
        console: NRO_CONSOLE,
        clock: SystemClock::new(),
    }
}

/// Makes the vCPU of every core but core 0 in a scripted run: a script with no steps, so a thread
/// the scheduler gives one of those cores fails the run ("no more exits").
static IDLE: fn(CoreId) -> Result<ScriptedVcpu, starquake_cpu::Error> = |_| Ok(ScriptedVcpu::new());

/// A scripted run's vCPUs: `first` for core 0, and [`IDLE`] ones for the others.
fn vcpus(first: &mut ScriptedVcpu) -> Vcpus<'_, ScriptedVcpu> {
    Vcpus { first, make: &IDLE }
}

/// Flat memory as a guest's address space: it backs nothing lazily (every byte it has is
/// there from the start) and owes its vCPUs nothing.
impl<V> GuestSpace<V> for FlatMemory {
    fn touch(&self, _: u64, _: Access) -> Touch {
        Touch::Refused
    }

    fn settle(&self, _: &mut V) -> Result<(), starquake_cpu::Error> {
        Ok(())
    }
}

/// A program loaded into flat memory, with the kernel's address space behind a mapper that
/// changes nothing.
struct Guest {
    memory: FlatMemory,
    placed: Placement,
    /// The kernel's address space, until a run takes it.
    address_space: Cell<Option<AddressSpace>>,
    /// Where the kernel will put the main thread's thread-local region: the first page of the
    /// kernel-map region past its guard pages.
    tls: u64,
}

impl Guest {
    fn new(file: &[u8]) -> Self {
        let program = Program::parse(file).unwrap();
        let memory = FlatMemory::new(RAM_BASE, RAM_SIZE);
        let (mapper, _log) = RecordingMapper::new();
        let Loaded {
            placed,
            address_space,
        } = load(&program, None, Box::new(mapper), &memory).unwrap();
        let tls = address_space.regions().kernel_map.start + GUARD_PAGES * crate::image::PAGE;
        Self {
            memory,
            placed,
            address_space: Cell::new(Some(address_space)),
            tls,
        }
    }

    /// The kernel's address space. A guest runs once.
    fn address_space(&self) -> AddressSpace {
        self.address_space.take().expect("one run per guest")
    }

    /// What the loop takes, with `space` as the guest's address space.
    fn program<'a, V>(&'a self, space: &'a dyn GuestSpace<V>) -> GuestProgram<'a, V> {
        GuestProgram {
            space,
            placed: &self.placed,
            address_space: self.address_space(),
        }
    }

    /// Runs `vcpu` as this guest.
    fn run(&self, mut vcpu: ScriptedVcpu, options: &RunOptions) -> RunReport {
        run(
            vcpus(&mut vcpu),
            self.program(&self.memory),
            options,
            Instant::now(),
            timebase(),
        )
    }

    /// What a run's memory map holds: the loader's mappings, and the page of the main thread's
    /// thread-local region, in address order.
    fn memory_map(&self) -> Vec<crate::report::Mapping> {
        let mut map = self.placed.mappings.clone();
        map.push(crate::report::Mapping {
            name: "tls".to_owned(),
            start: self.tls,
            end: self.tls + crate::image::PAGE,
        });
        map.sort_by_key(|mapping| mapping.start);
        map
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

/// Where the tests put strings for the guest to print: flat memory above the image.
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

impl<V> GuestSpace<V> for SlowMemory<'_> {
    fn touch(&self, _: u64, _: Access) -> Touch {
        Touch::Refused
    }

    fn settle(&self, _: &mut V) -> Result<(), starquake_cpu::Error> {
        Ok(())
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
    assert_eq!(programs::COUNTER_HZ, NRO_CONSOLE.counter_rate().hz());
    // The threads program's calls, and its second thread's priority: the main thread's.
    for (restated, id) in [
        (programs::SVC_CREATE_THREAD, SvcId::CreateThread),
        (programs::SVC_START_THREAD, SvcId::StartThread),
        (programs::SVC_EXIT_THREAD, SvcId::ExitThread),
        (
            programs::SVC_GET_CURRENT_PROCESSOR_NUMBER,
            SvcId::GetCurrentProcessorNumber,
        ),
        (
            programs::SVC_WAIT_SYNCHRONIZATION,
            SvcId::WaitSynchronization,
        ),
    ] {
        assert_eq!(restated, u32::from(id.number()), "{}", id.name());
    }
    assert_eq!(
        programs::THREAD_PRIORITY,
        u32::from(ProcessDescription::homebrew(NRO_CONSOLE).main_thread_priority)
    );
}

#[test]
fn a_guest_that_prints_two_lines_and_exits_is_reported_in_full() {
    let guest = Guest::new(&programs::hello().unwrap());
    let [first, second] = programs::HELLO_LINES;
    let first_at = guest.start_of("main.rodata");
    let second_at = first_at + first.len() as u64;
    let (first_len, second_len) = (first.len() as u64, second.len() as u64);
    let (entry, stack_top, tls) = (guest.placed.entry, guest.placed.stack_top, guest.tls);
    let main_thread = Handle::from_parts(MAX_TABLE_SIZE - 1, MIN_LINEAR_ID).unwrap();
    assert_eq!(main_thread.raw(), 0x83ff);

    let vcpu = ScriptedVcpu::new()
        .then(
            move |regs| {
                // What the loop starts the thread with: `x1` is a handle to the main thread,
                // the first handle of a full-size table (its last index, the first linear
                // id), and everything else is zero. The reference emulator's trace of the
                // homebrew fixture shows the same value, 0x83ff, left in `x2`.
                assert_eq!((regs.pc, regs.sp, regs.tpidrro), (entry, stack_top, tls));
                let mut x = [0; 31];
                x[1] = u64::from(main_thread.raw());
                assert_eq!((regs.x, regs.tpidr, regs.pstate), (x, 0, 0));
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
    // The loader's mappings, and the main thread's thread-local page the kernel placed.
    assert_eq!(report.memory_map, guest.memory_map());
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
        sd_card: None,
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
        report_crash.syndrome,
        Some(SyndromeReport::new(UNMAPPED_STORE)),
        "the report carries the exit's syndrome"
    );
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
    assert_eq!(report_crash.memory_map, guest.memory_map());
    assert_eq!(report_crash.modules, guest.placed.modules);
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
    assert_eq!(
        crash.syndrome,
        Some(SyndromeReport::new(Syndrome::instruction_abort(
            FaultStatus::Permission { level: 3 }
        )))
    );
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
fn each_abort_crash_carries_its_own_exits_syndrome() {
    // Two different syndromes per abort kind, so a report that carried a fixed syndrome in place
    // of the exit's would get one of each pair wrong.
    let crash = programs::crash().unwrap();
    for syndrome in [
        UNMAPPED_STORE,
        Syndrome::data_abort(FaultStatus::Permission { level: 3 }, true),
    ] {
        let guest = Guest::new(&crash.nro);
        let instruction = guest.placed.entry + crash.store_offset;
        let address = crash.fault_address;
        let vcpu = ScriptedVcpu::new().then(
            move |regs| regs.pc = instruction,
            Exit::DataAbort {
                address,
                write: true,
                syndrome,
            },
        );
        let report = guest.run(vcpu, &RunOptions::default());
        let report_crash = report.crash().expect("a crash report");
        assert_eq!(
            report_crash.syndrome,
            Some(SyndromeReport::new(syndrome)),
            "{:#x}",
            syndrome.esr()
        );
    }
    for syndrome in [
        Syndrome::instruction_abort(FaultStatus::Translation { level: 2 }),
        Syndrome::instruction_abort(FaultStatus::Permission { level: 3 }),
    ] {
        let guest = Guest::new(&programs::hello().unwrap());
        let address = guest.start_of("stack") + 0x40;
        let vcpu = ScriptedVcpu::new().then(
            move |regs| regs.pc = address,
            Exit::InstructionAbort { address, syndrome },
        );
        let report = guest.run(vcpu, &RunOptions::default());
        let crash = report.crash().expect("a crash report");
        assert_eq!(
            crash.syndrome,
            Some(SyndromeReport::new(syndrome)),
            "{:#x}",
            syndrome.esr()
        );
    }
}

#[test]
fn an_undefined_instruction_is_a_crash_at_its_address() {
    let guest = Guest::new(&programs::hello().unwrap());
    let pc = guest.placed.entry + 0x40;
    // A `brk #0x123` (class 0x3c, `IL`, the immediate in `ISS`), not the plain undefined
    // syndrome, so the report shows the one the exit carried.
    let brk = Syndrome::from_esr(0xf200_0123);
    let vcpu = ScriptedVcpu::new().then(
        move |regs| regs.pc = pc,
        Exit::Undefined { pc, syndrome: brk },
    );
    let report = guest.run(vcpu, &RunOptions::default());
    assert_eq!(report.exit_code(), exit_code::GUEST_CRASHED);
    let crash = report.crash().expect("a crash report");
    assert_eq!(crash.exception, Exception::UndefinedInstruction);
    assert_eq!(crash.pc, pc);
    assert_eq!(crash.syndrome, Some(SyndromeReport::new(brk)));
    assert!(
        crash
            .to_text()
            .contains("syndrome: 0xf2000123 (brk #0x123)\n"),
        "{}",
        crash.to_text()
    );
    assert_eq!(
        crash.pc_location.as_ref().map(|location| location.offset),
        Some(0x40)
    );
}

/// The vCPUs of a threaded run, to interrupt if the run does not stop them.
type Interrupters = Arc<Mutex<Vec<starquake_cpu::testing::ScriptedInterrupter>>>;

/// Runs `body`, interrupting every vCPU in `interrupters` if it has not finished within twenty
/// seconds: a run whose stop fails to interrupt its spinning cores then ends late (its cores
/// see the run is stopping when this interrupt arrives) and the test fails on its time instead
/// of hanging the gate.
fn bounded<R>(interrupters: &Interrupters, body: impl FnOnce() -> R) -> R {
    let (done, finished) = std::sync::mpsc::channel::<()>();
    std::thread::scope(|scope| {
        scope.spawn(move || {
            if finished.recv_timeout(Duration::from_secs(20)).is_err() {
                for interrupter in lock(interrupters).iter() {
                    interrupter.interrupt().unwrap();
                }
            }
        });
        let result = body();
        assert!(
            done.send(()).is_ok(),
            "the watchdog fired: the test ran out of time"
        );
        result
    })
}

#[test]
fn a_guest_that_never_stops_is_ended_by_the_timeout() {
    let guest = Guest::new(&programs::endless_loop().unwrap());
    let timeout = Duration::from_millis(30);
    let options = RunOptions {
        timeout: Some(timeout),
        command_line: Vec::new(),
        sd_card: None,
    };
    let vcpu = ScriptedVcpu::new().then_spin();
    let interrupters = Interrupters::default();
    lock(&interrupters).push(vcpu.interrupter());
    let report = bounded(&interrupters, || guest.run(vcpu, &options));
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
        sd_card: None,
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
        vcpus(&mut vcpu),
        guest.program(&SlowMemory(&guest.memory)),
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
    for console in [Console::Switch1, Console::Switch2] {
        let rate = console.counter_rate();
        let mut guest = Guest::new(&programs::hello().unwrap());
        // The process is described as built for the console the clock counts for.
        guest.placed.description.console = console;
        let tick = Rc::new(Cell::new(0));
        let seen = Rc::clone(&tick);
        // The second step's closure sees what the first call wrote to x0.
        let mut vcpu = ScriptedVcpu::new()
            .then(|_| {}, svc(SvcId::GetSystemTick))
            .then(move |regs| seen.set(regs.x[0]), svc(SvcId::ExitProcess));
        let report = run(
            vcpus(&mut vcpu),
            guest.program(&guest.memory),
            &RunOptions::default(),
            Instant::now(),
            Timebase {
                console,
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
        options: &options,
        clock: RunClock::new(timebase()),
        recorder: Recorder::new(),
    };
    let env = session.environment();
    let mut vcpu = ScriptedVcpu::new();
    for _ in 0..CALLS {
        vcpu = vcpu.then(|_| {}, svc(SvcId::GetSystemTick));
    }
    let mut vcpu = vcpu.then(|_| {}, svc(SvcId::ExitProcess));

    let (dispatcher, _) = boot(guest.address_space(), &guest.placed, &guest.memory).unwrap();
    let (ending, _, tally) = drive(
        vcpus(&mut vcpu),
        dispatcher,
        &guest.memory,
        &session,
        &env,
        Limits::DEFAULT,
        &real_services,
    );
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
        vcpus(&mut vcpu),
        guest.program(&guest.memory),
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
        vcpus(&mut vcpu),
        guest.program(&guest.memory),
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

// ----- The guest's address space: lazy backing and settling ----------------------------------------

/// A guest address space for the loop's own tests: flat memory, a script of what each touch
/// comes to, and a record of what the loop asked of it.
struct ScriptedSpace<'a> {
    memory: &'a FlatMemory,
    /// What the touches come to, in order; when the script runs out, a touch is refused.
    script: Mutex<std::vec::IntoIter<Touch>>,
    /// Every touch the loop made.
    touched: Mutex<Vec<(u64, Access)>>,
    /// How many times the loop settled the address space.
    settled: Arc<AtomicUsize>,
    /// The settle (from zero) that fails, if any.
    failing_settle: Option<usize>,
}

impl<'a> ScriptedSpace<'a> {
    fn new(memory: &'a FlatMemory, script: Vec<Touch>) -> Self {
        Self {
            memory,
            script: Mutex::new(script.into_iter()),
            touched: Mutex::new(Vec::new()),
            settled: Arc::new(AtomicUsize::new(0)),
            failing_settle: None,
        }
    }

    fn touched(&self) -> Vec<(u64, Access)> {
        self.touched
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }
}

impl GuestMemory for ScriptedSpace<'_> {
    fn read(&self, va: u64, buf: &mut [u8]) -> Result<(), Fault> {
        self.memory.read(va, buf)
    }

    fn write(&self, va: u64, data: &[u8]) -> Result<(), Fault> {
        self.memory.write(va, data)
    }
}

impl<V> GuestSpace<V> for ScriptedSpace<'_> {
    fn touch(&self, address: u64, access: Access) -> Touch {
        self.touched
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push((address, access));
        self.script
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .next()
            .unwrap_or(Touch::Refused)
    }

    fn settle(&self, _: &mut V) -> Result<(), starquake_cpu::Error> {
        let call = self.settled.fetch_add(1, Ordering::SeqCst);
        if self.failing_settle == Some(call) {
            return Err(starquake_cpu::Error::Host("the flush did not run".into()));
        }
        Ok(())
    }
}

/// The syndrome a real vCPU reports for a fetch from an address nothing maps yet.
const UNMAPPED_FETCH: Syndrome = Syndrome::instruction_abort(FaultStatus::Translation { level: 3 });

#[test]
fn a_fault_on_a_page_that_only_lacked_backing_resumes_the_guest_and_is_not_an_event() {
    let guest = Guest::new(&programs::hello().unwrap());
    let stack = guest.start_of("stack");
    let text = guest.start_of("main.text");
    let space = ScriptedSpace::new(
        &guest.memory,
        vec![Touch::Backed, Touch::Backed, Touch::Backed],
    );
    // A store, a load and a fetch that each fault once; the guest then goes on to exit.
    let vcpu = ScriptedVcpu::new()
        .then(
            |_| {},
            Exit::DataAbort {
                address: stack + 0x18,
                write: true,
                syndrome: UNMAPPED_STORE,
            },
        )
        .then(
            |_| {},
            Exit::DataAbort {
                address: stack + 0x1000,
                write: false,
                syndrome: Syndrome::data_abort(FaultStatus::Translation { level: 3 }, false),
            },
        )
        .then(
            |_| {},
            Exit::InstructionAbort {
                address: text + 0x40,
                syndrome: UNMAPPED_FETCH,
            },
        )
        .then(|_| {}, svc(SvcId::ExitProcess));
    let mut vcpu = vcpu;
    let report = run(
        vcpus(&mut vcpu),
        guest.program(&space),
        &RunOptions::default(),
        Instant::now(),
        timebase(),
    );
    assert_eq!(report.exit, ExitReason::ProcessExited);
    // The loop asked about each access as it was: the address that faulted, and its kind.
    assert_eq!(
        space.touched(),
        [
            (stack + 0x18, Access::Write),
            (stack + 0x1000, Access::Read),
            (text + 0x40, Access::Execute),
        ]
    );
    // Backing a page is not an event: the stream holds the start, the exit call and the exit.
    assert_eq!(
        recorded(&report),
        [
            lifecycle(LifecyclePhase::ProcessStart),
            EventKind::Svc {
                id: u32::from(SvcId::ExitProcess.number()),
                args: [0; starquake_trace::SVC_ARG_COUNT],
                result: starquake_trace::ResultCode(0),
            },
            lifecycle(LifecyclePhase::ProcessExit),
        ]
    );
}

#[test]
fn a_fault_the_address_space_refuses_after_one_it_backed_is_still_a_crash() {
    let guest = Guest::new(&programs::hello().unwrap());
    let stack = guest.start_of("stack");
    // The first touch is backed, the second (the script has run out) refused.
    let space = ScriptedSpace::new(&guest.memory, vec![Touch::Backed]);
    let abort = |address| Exit::DataAbort {
        address,
        write: true,
        syndrome: UNMAPPED_STORE,
    };
    let mut vcpu = ScriptedVcpu::new()
        .then(|_| {}, abort(stack))
        .then(|_| {}, abort(stack - 8));
    let report = run(
        vcpus(&mut vcpu),
        guest.program(&space),
        &RunOptions::default(),
        Instant::now(),
        timebase(),
    );
    assert_eq!(report.exit_code(), exit_code::GUEST_CRASHED);
    assert_eq!(
        report.crash().unwrap().exception,
        Exception::DataAbort {
            address: stack - 8,
            write: true
        }
    );
    // Only the fault that ended the run is an event.
    let faults: Vec<EventKind> = recorded(&report)
        .into_iter()
        .filter(|kind| matches!(kind, EventKind::Fault { .. }))
        .collect();
    assert_eq!(
        faults,
        [EventKind::Fault {
            address: stack - 8,
            write: true,
            action: FaultAction::Fatal,
        }]
    );
}

#[test]
fn a_page_that_cannot_be_backed_is_an_emulator_error_not_a_crash() {
    let out_of_ram = SpaceError::OutOfRam {
        size: starquake_memory::PAGE_SIZE,
        available: 0,
    };
    assert_eq!(
        out_of_ram.to_string(),
        "mapping 0x1000 bytes needs more guest RAM than the 0x0 bytes left"
    );
    for fetch in [false, true] {
        let guest = Guest::new(&programs::hello().unwrap());
        let address = guest.start_of("stack") + 0x2000;
        let space = ScriptedSpace::new(&guest.memory, vec![Touch::Failed(out_of_ram)]);
        let exit = if fetch {
            Exit::InstructionAbort {
                address,
                syndrome: UNMAPPED_FETCH,
            }
        } else {
            Exit::DataAbort {
                address,
                write: true,
                syndrome: UNMAPPED_STORE,
            }
        };
        let mut vcpu = ScriptedVcpu::new().then(|_| {}, exit);
        let report = run(
            vcpus(&mut vcpu),
            guest.program(&space),
            &RunOptions::default(),
            Instant::now(),
            timebase(),
        );
        assert_eq!(
            report.exit,
            ExitReason::EmulatorError(format!(
                "guest memory at {address:#x} could not be backed: {out_of_ram}"
            )),
            "fetch: {fetch}"
        );
        assert_eq!(report.exit_code(), exit_code::EMULATOR_ERROR);
        assert!(report.crash().is_none());
    }
}

#[test]
fn the_address_space_is_settled_after_every_system_call_before_the_guest_runs_again() {
    let guest = Guest::new(&programs::hello().unwrap());
    let space = ScriptedSpace::new(&guest.memory, Vec::new());
    let settled = Arc::clone(&space.settled);
    // Each closure is the guest running: before its first instruction the address space has
    // been settled once (loading reprotected the segments), and by each later one the call
    // before it has been, whatever it returned (a result, "not implemented", an error result).
    let expect = move |calls: usize| {
        let settled = Arc::clone(&settled);
        move |_: &mut GuestRegs| assert_eq!(settled.load(Ordering::SeqCst), calls)
    };
    let mut vcpu = ScriptedVcpu::new()
        .then(expect(1), svc(SvcId::GetSystemTick))
        .then(expect(2), svc(SvcId::Break))
        .then(expect(3), svc(SvcId::CloseHandle))
        .then(expect(4), svc(SvcId::ExitProcess));
    let report = run(
        vcpus(&mut vcpu),
        guest.program(&space),
        &RunOptions::default(),
        Instant::now(),
        timebase(),
    );
    assert_eq!(report.exit, ExitReason::ProcessExited);
    assert_eq!(space.settled.load(Ordering::SeqCst), 5);
    // A fault settles nothing: only a system call changes mappings.
    assert_eq!(space.touched(), []);
}

#[test]
fn a_settle_that_fails_ends_the_run_as_an_emulator_error() {
    let guest = Guest::new(&programs::hello().unwrap());
    let mut space = ScriptedSpace::new(&guest.memory, Vec::new());
    // The settle after the second call (the third in all: one comes before the guest starts).
    space.failing_settle = Some(2);
    let reached = Rc::new(Cell::new(0));
    let count = |reached: &Rc<Cell<u32>>| {
        let reached = Rc::clone(reached);
        move |_: &mut GuestRegs| reached.set(reached.get() + 1)
    };
    let mut vcpu = ScriptedVcpu::new()
        .then(count(&reached), svc(SvcId::GetSystemTick))
        .then(count(&reached), svc(SvcId::GetSystemTick))
        .then(count(&reached), svc(SvcId::ExitProcess));
    let report = run(
        vcpus(&mut vcpu),
        guest.program(&space),
        &RunOptions::default(),
        Instant::now(),
        timebase(),
    );
    assert_eq!(
        report.exit,
        ExitReason::EmulatorError("the vCPU failed: host API error: the flush did not run".into())
    );
    // The guest did not run again after the call whose flush failed.
    assert_eq!(reached.get(), 2);
}

#[test]
fn the_main_threads_handle_names_a_thread_with_the_default_priority() {
    let guest = Guest::new(&programs::hello().unwrap());
    // `GetThreadPriority` takes its handle in x1, which is where the loop put the main
    // thread's: the guest passes it back untouched.
    let vcpu = ScriptedVcpu::new()
        .then(|_| {}, svc(SvcId::GetThreadPriority))
        .then(
            |regs| {
                let priority = ProcessDescription::homebrew(NRO_CONSOLE).main_thread_priority;
                assert_eq!((regs.x[0], regs.x[1]), (0, u64::from(priority)));
            },
            svc(SvcId::ExitProcess),
        );
    let report = guest.run(vcpu, &RunOptions::default());
    assert_eq!(report.exit, ExitReason::ProcessExited);
    assert!(report.stubbed.is_empty(), "{:?}", report.stubbed);
}

#[test]
fn the_process_is_created_from_the_placements_description() {
    // A description whose main thread priority differs from the default: the main thread runs
    // at that priority.
    let mut guest = Guest::new(&programs::hello().unwrap());
    let default = ProcessDescription::homebrew(NRO_CONSOLE);
    let priority = default.main_thread_priority + 1;
    guest.placed.description.main_thread_priority = priority;
    let vcpu = ScriptedVcpu::new()
        .then(|_| {}, svc(SvcId::GetThreadPriority))
        .then(
            move |regs| assert_eq!((regs.x[0], regs.x[1]), (0, u64::from(priority))),
            svc(SvcId::ExitProcess),
        );
    let report = guest.run(vcpu, &RunOptions::default());
    assert_eq!(report.exit, ExitReason::ProcessExited);
    // A description the kernel refuses ends the run before the guest runs.
    let mut refused = Guest::new(&programs::hello().unwrap());
    refused.placed.description.main_thread_priority = default.priorities.first - 1;
    let report = refused.run(ScriptedVcpu::new(), &RunOptions::default());
    assert_eq!(
        report.exit,
        ExitReason::EmulatorError(format!(
            "the kernel could not be set up: the main thread's priority {} is outside the \
             process's priorities",
            default.priorities.first - 1
        ))
    );
}

#[test]
fn a_clock_for_another_console_than_the_descriptions_ends_the_run_before_the_guest_runs() {
    // The program's reads were patched for its description's console; a clock counting at the
    // other console's rate would give `GetSystemTick` a different rate.
    let guest = Guest::new(&programs::hello().unwrap());
    assert_eq!(guest.placed.description.console, Console::Switch1);
    let report = run(
        vcpus(&mut ScriptedVcpu::new()),
        guest.program(&guest.memory),
        &RunOptions::default(),
        Instant::now(),
        Timebase {
            console: Console::Switch2,
            clock: SystemClock::new(),
        },
    );
    assert_eq!(
        report.exit,
        ExitReason::EmulatorError(CONSOLE_MISMATCH.to_owned())
    );
    assert!(report.system_calls.is_empty());
    assert_eq!(
        CONSOLE_MISMATCH,
        "the run's clock and the process's description name different consoles, so the \
         counter rates would differ"
    );
    // The same guest with the description's console runs.
    let matching = Guest::new(&programs::hello().unwrap());
    let vcpu = ScriptedVcpu::new().then(|_| {}, svc(SvcId::ExitProcess));
    assert_eq!(
        matching.run(vcpu, &RunOptions::default()).exit,
        ExitReason::ProcessExited
    );
}

mod six_cores;
