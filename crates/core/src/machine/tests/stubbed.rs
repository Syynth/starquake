//! A service nothing provides, in a real run (`run`): a scripted guest asks `sm:` for one and
//! sends it a command, which the run's auto-stub policy answers. The summary and the crash
//! report name both the service and the command (the owner's ruling of 2026-11-17 on issue
//! #207).
//!
//! Every message is in the main thread's thread-local page, as in `reads.rs`, whose helpers
//! these steps use. Each run has a timeout, so a step that fails ends the run instead of
//! leaving the other threads waiting.

use std::cell::RefCell;

use starquake_cpu::Exit;
use starquake_services::framework::Protocol;
use starquake_services::framework::Session;
use starquake_services::sm::{self, MAX_NAME_LEN, ServiceName, Sm};
use starquake_trace::{Dump, EventKind, IpcOutcome};

use super::reads::{Script, Seen, WithTls, libnx, send};
use super::*;
use crate::services_report::{StubbedCommand, StubbedRequestReport};

/// Where in the thread-local page the port's name is.
const NAME_AT: u64 = 0x300;
/// The service the guest asks for, which nothing provides.
const UNKNOWN: &str = "lm";
/// The command it sends the service.
const COMMAND: u32 = 5;
/// The longest a run here may take: its timeout.
const RUN_LIMIT: Duration = Duration::from_secs(30);

/// The conversation: `sm:`'s `Initialize`, `GetService(UNKNOWN)`, then [`COMMAND`] on the
/// session it gave, then `ending` (an exit or a crash), after which the run is over.
fn asking_guest(guest: &Script, ending: Exit) -> ScriptedVcpu {
    let name = guest.base + NAME_AT;
    guest.tls.write(name, b"sm:\0").unwrap();
    let sm_handle = Rc::new(RefCell::new(0u64));
    let (keep, take) = (Rc::clone(&sm_handle), Rc::clone(&sm_handle));
    ScriptedVcpu::new()
        .then(move |regs| regs.x[1] = name, svc(SvcId::ConnectToNamedPort))
        .then(
            {
                let guest = guest.clone();
                move |regs| {
                    assert_eq!(regs.x[0], 0, "ConnectToNamedPort failed");
                    *keep.borrow_mut() = regs.x[1];
                    guest.write(
                        &libnx(sm::command::INITIALIZE)
                            .send_pid(0)
                            .data(&0_u64.to_le_bytes()),
                    );
                    regs.x[0] = regs.x[1];
                }
            },
            svc(SvcId::SendSyncRequest),
        )
        .then(
            send(
                guest,
                move |_| *take.borrow(),
                |guest| {
                    guest.read_reply(0);
                    let name: [u8; MAX_NAME_LEN] = ServiceName::new(UNKNOWN).unwrap().to_field();
                    guest.write(&libnx(sm::command::GET_SERVICE).data(&name));
                },
            ),
            svc(SvcId::SendSyncRequest),
        )
        .then(
            send(
                guest,
                |guest| {
                    guest.read_reply(0);
                    guest.last_handle()
                },
                |guest| guest.write(&libnx(COMMAND).data(&[0xEE; 8])),
            ),
            svc(SvcId::SendSyncRequest),
        )
        .then(
            {
                let guest = guest.clone();
                move |regs| {
                    assert_eq!(regs.x[0], 0, "the stubbed command failed");
                    guest.read_reply(0);
                }
            },
            ending,
        )
}

/// Runs [`asking_guest`] ending with `ending`; returns the report and what the guest saw.
fn run_asking(ending: impl FnOnce(&Guest) -> Exit) -> (RunReport, Seen) {
    let guest = Guest::new(&programs::hello().unwrap());
    let page = usize::try_from(crate::image::PAGE).unwrap();
    let tls = Arc::new(FlatMemory::new(guest.tls, page));
    let space = WithTls {
        image: &guest.memory,
        tls: Arc::clone(&tls),
    };
    let seen = Rc::new(RefCell::new(Seen::default()));
    let view = Script {
        tls,
        base: guest.tls,
        seen: Rc::clone(&seen),
    };
    let mut main = asking_guest(&view, ending(&guest));
    let options = RunOptions {
        timeout: Some(RUN_LIMIT),
        ..RunOptions::default()
    };
    let report = run(
        vcpus(&mut main),
        guest.program(&space),
        &options,
        Instant::now(),
        timebase(),
    );
    drop((view, main));
    let seen = Rc::try_unwrap(seen).ok().unwrap().into_inner();
    (report, seen)
}

/// What `sm:`'s table calls a CMIF command.
fn sm_name(id: u32) -> &'static str {
    Sm::default().commands().name(Protocol::Cmif, id).unwrap()
}

/// The summary's view of the services, which every ending of the run shares: the requests by
/// command, the stubbed command, the unknown name, exactly.
fn assert_services_named(report: &RunReport) {
    let summary = serde_json::to_value(report.summary()).unwrap();
    let service = sm::PORT_NAME;
    assert_eq!(
        summary["service_commands"],
        serde_json::json!({
            format!("{service}:Sm::{}", sm_name(sm::command::INITIALIZE)): 1,
            format!("{service}:Sm::{}", sm_name(sm::command::GET_SERVICE)): 1,
            format!("{service}:{UNKNOWN}::{COMMAND}"): 1,
        })
    );
    assert_eq!(summary["service_commands_uncounted"], 0);
    assert_eq!(
        report.services.stubbed,
        [StubbedCommand {
            service: service.to_owned(),
            interface: UNKNOWN.to_owned(),
            protocol: "cmif",
            command: COMMAND,
        }]
    );
    assert_eq!(
        summary["stubbed_commands"],
        serde_json::json!([{
            "service": service,
            "interface": UNKNOWN,
            "protocol": "cmif",
            "command": COMMAND,
        }])
    );
    assert_eq!(summary["unknown_services"], serde_json::json!([UNKNOWN]));
}

#[test]
fn an_unknown_service_and_its_command_are_named_in_the_summary() {
    let (report, seen) = run_asking(|_| svc(SvcId::ExitProcess));
    assert_eq!(report.exit, ExitReason::ProcessExited, "{report:?}");
    // Initialize, GetService and the command: each answered with success.
    assert_eq!(seen.results, [0, 0, 0]);
    assert_services_named(&report);
    assert_eq!(report.crash(), None);
}

#[test]
fn a_crash_after_it_names_the_last_auto_stubbed_command_and_its_service() {
    let (report, seen) = run_asking(|guest| {
        let pc = guest.placed.entry + 0x40;
        Exit::Undefined {
            pc,
            syndrome: Syndrome::undefined(),
        }
    });
    assert_eq!(report.exit_code(), exit_code::GUEST_CRASHED, "{report:?}");
    assert_eq!(seen.results, [0, 0, 0]);
    assert_services_named(&report);

    // The one stubbed request, as the event stream numbered it.
    let events = Dump::from_bytes(&report.events).unwrap().merged();
    let stubbed: Vec<u64> = events
        .iter()
        .filter(|event| {
            matches!(
                event.kind,
                EventKind::Ipc {
                    outcome: IpcOutcome::Stubbed,
                    ..
                }
            )
        })
        .map(|event| event.seq)
        .collect();
    assert_eq!(stubbed.len(), 1);
    let crash = report.crash().unwrap();
    assert_eq!(
        crash.last_stubbed,
        [StubbedRequestReport {
            seq: stubbed[0],
            thread: MAIN_THREAD.0,
            service: sm::PORT_NAME.to_owned(),
            interface: UNKNOWN.to_owned(),
            protocol: "cmif",
            command: COMMAND,
        }]
    );
    let line = format!(
        "  #{} thread {}: {UNKNOWN} (on {}) cmif command {COMMAND}\n",
        stubbed[0],
        MAIN_THREAD.0,
        sm::PORT_NAME
    );
    assert!(
        crash.to_text().ends_with(&format!(
            "last auto-stubbed requests, oldest first:\n{line}"
        )),
        "{}",
        crash.to_text()
    );
    let json = serde_json::to_value(crash).unwrap();
    assert_eq!(json["last_stubbed"][0]["interface"], UNKNOWN);
    assert_eq!(json["last_stubbed"][0]["command"], COMMAND);
}
