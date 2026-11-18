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
use starquake_ipc::{CommandType, RequestBuilder, RequestLayout};
use starquake_services::framework::Protocol;
use starquake_services::framework::Session;
use starquake_services::framework::control::{CONTROL_COMMANDS, CONTROL_INTERFACE};
use starquake_services::sm::{self, MAX_NAME_LEN, ServiceName, Sm};
use starquake_trace::{Dump, EventKind, IpcOutcome, IpcProtocol};

use super::reads::{Script, Seen, WithTls, libnx, send};
use super::*;
use crate::services_report::{StubbedCommand, StubbedRequestReport};

/// Where in the thread-local page the port's name is.
const NAME_AT: u64 = 0x300;
/// The service the guest asks for, which nothing provides: a made-up name, so that no service
/// a later change registers can take it (`lm`, the name used first, is now registered).
const UNKNOWN: &str = "absent";
/// The command it sends the service.
const COMMAND: u32 = 5;
/// The longest a run here may take: its timeout.
const RUN_LIMIT: Duration = Duration::from_secs(30);

/// The conversation: `sm:`'s `Initialize`, `GetService(UNKNOWN)`, then `request` on the
/// session it gave, then `ending` (an exit or a crash), after which the run is over.
fn asking_guest(guest: &Script, request: RequestBuilder, ending: Exit) -> ScriptedVcpu {
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
                move |guest| guest.write(&request),
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

/// [`COMMAND`], as libnx sends it, with some arguments.
fn command_request() -> RequestBuilder {
    libnx(COMMAND).data(&[0xEE; 8])
}

/// Runs [`asking_guest`] sending `request` and ending with `ending`; returns the report and
/// what the guest saw.
fn run_asking(request: RequestBuilder, ending: impl FnOnce(&Guest) -> Exit) -> (RunReport, Seen) {
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
    let mut main = asking_guest(&view, request, ending(&guest));
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
            protocol: IpcProtocol::Cmif,
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
    let (report, seen) = run_asking(command_request(), |_| svc(SvcId::ExitProcess));
    assert_eq!(report.exit, ExitReason::ProcessExited, "{report:?}");
    // Initialize, GetService and the command: each answered with success.
    assert_eq!(seen.results, [0, 0, 0]);
    assert_services_named(&report);
    assert_eq!(report.crash(), None);
}

#[test]
fn a_crash_after_it_names_the_last_auto_stubbed_command_and_its_service() {
    let (report, seen) = run_asking(command_request(), |guest| {
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
            protocol: IpcProtocol::Cmif,
            command: COMMAND,
        }]
    );
    assert_eq!(crash.unknown_services, [UNKNOWN]);
    assert_eq!(crash.stubbed_commands, report.services.stubbed);
    let service = sm::PORT_NAME;
    let line = format!(
        "  #{} thread {}: {UNKNOWN} (on {service}) cmif command {COMMAND}\n",
        stubbed[0], MAIN_THREAD.0,
    );
    assert!(
        crash.to_text().ends_with(&format!(
            "\nservices nothing provides:\n  {UNKNOWN}\n\nauto-stubbed commands:\n  {UNKNOWN} \
             (on {service}) cmif command {COMMAND}\n\nlast auto-stubbed requests, oldest \
             first:\n{line}"
        )),
        "{}",
        crash.to_text()
    );
    let json = serde_json::to_value(crash).unwrap();
    assert_eq!(json["unknown_services"], serde_json::json!([UNKNOWN]));
    assert_eq!(json["stubbed_commands"][0]["interface"], UNKNOWN);
    assert_eq!(json["last_stubbed"][0]["interface"], UNKNOWN);
    assert_eq!(json["last_stubbed"][0]["command"], COMMAND);
}

/// A crash after `request`, which the run auto-stubs: the report, and the stubbed request's
/// place in the event stream.
fn crash_after(request: RequestBuilder) -> (RunReport, u64) {
    let (report, seen) = run_asking(request, |guest| Exit::Undefined {
        pc: guest.placed.entry + 0x40,
        syndrome: Syndrome::undefined(),
    });
    assert_eq!(report.exit_code(), exit_code::GUEST_CRASHED, "{report:?}");
    assert_eq!(seen.results, [0, 0, 0]);
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
    (report, stubbed[0])
}

/// A control command the framework does not have, sent to the stub: the summary's stubbed
/// commands and the crash report's last stubbed requests write it the same way, as a control
/// command on `IpcController`, in their records, their JSON and their text.
#[test]
fn a_stubbed_control_command_is_written_alike_in_both_lists() {
    // The first id past the control table's commands.
    let id = (CONTROL_COMMANDS.cmif.iter())
        .map(|command| command.id)
        .max()
        .unwrap()
        + 1;
    assert_eq!(CONTROL_COMMANDS.name(Protocol::Cmif, id), None);
    let request = RequestBuilder::cmif(CommandType::Control, id).layout(RequestLayout::Libnx);
    let (report, seq) = crash_after(request);
    let service = sm::PORT_NAME;
    let command = StubbedCommand {
        service: service.to_owned(),
        interface: CONTROL_INTERFACE.to_owned(),
        protocol: IpcProtocol::Control,
        command: id,
    };
    assert_eq!(report.services.stubbed, std::slice::from_ref(&command));
    let crash = report.crash().unwrap();
    assert_eq!(crash.stubbed_commands, std::slice::from_ref(&command));
    assert_eq!(
        crash.last_stubbed,
        [StubbedRequestReport {
            seq,
            thread: MAIN_THREAD.0,
            service: service.to_owned(),
            interface: CONTROL_INTERFACE.to_owned(),
            protocol: IpcProtocol::Control,
            command: id,
        }]
    );
    // The two lists agree on the request, field by field.
    let (listed, last) = (&crash.stubbed_commands[0], &crash.last_stubbed[0]);
    assert_eq!(
        (
            &listed.service,
            &listed.interface,
            listed.protocol,
            listed.command
        ),
        (&last.service, &last.interface, last.protocol, last.command)
    );
    let json = serde_json::to_value(crash).unwrap();
    assert_eq!(json["stubbed_commands"][0]["protocol"], "control");
    assert_eq!(json["last_stubbed"][0]["protocol"], "control");
    let summary = serde_json::to_value(report.summary()).unwrap();
    assert_eq!(summary["stubbed_commands"][0]["protocol"], "control");
    assert_eq!(
        command.text(),
        format!("{CONTROL_INTERFACE} (on {service}) control command {id}")
    );
    assert!(
        crash.to_text().ends_with(&format!(
            "
auto-stubbed commands:
  {}

last auto-stubbed requests, oldest first:
  {}
",
            command.text(),
            crash.last_stubbed[0].text()
        )),
        "{}",
        crash.to_text()
    );
    assert_eq!(
        crash.last_stubbed[0].text(),
        format!(
            "#{seq} thread {}: {CONTROL_INTERFACE} (on {service}) control command {id}",
            MAIN_THREAD.0
        )
    );
}

#[test]
fn the_requests_programs_calls_and_buffer_are_the_kernels() {
    // The testkit restates them from the reference; this is the cross-check.
    for (restated, id) in [
        (
            programs::SVC_CONNECT_TO_NAMED_PORT,
            SvcId::ConnectToNamedPort,
        ),
        (programs::SVC_SEND_SYNC_REQUEST, SvcId::SendSyncRequest),
    ] {
        assert_eq!(restated, u32::from(id.number()), "{}", id.name());
    }
    assert_eq!(programs::MESSAGE_LIMIT, starquake_ipc::BUFFER_SIZE);
}
