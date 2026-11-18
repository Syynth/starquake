//! The pieces of the services' report on every host: the key a count has, the sink of the last
//! stubbed requests, and reading the servers back. A real run's report is checked end to end in
//! `machine/tests/stubbed.rs`.

use starquake_kernel::dispatch::IpcContext;
use starquake_kernel::id::ObjectId;
use starquake_kernel::ipc::{Handled, HandlerError, SessionHandler};
use starquake_services::apm::{self, Apm};
use starquake_services::framework::control::{
    CONTROL_COMMANDS, CONTROL_INTERFACE, QUERY_POINTER_BUFFER_SIZE,
};
use starquake_services::framework::{CountedCommand, Protocol, Session, UnknownPolicy};
use starquake_services::sm::{self, Sm};
use starquake_trace::{GuestTicks, ResultCode, SessionId, ThreadId};

use super::*;
use crate::hle;
use crate::hle::tests::{Rig as HleRig, dispatcher, libnx};

/// A count of `count` requests for `id` on `interface` in `protocol`, named `name`.
fn count(
    interface: &str,
    protocol: IpcProtocol,
    id: u32,
    name: Option<&'static str>,
) -> RequestCount {
    RequestCount {
        command: CountedCommand {
            interface: InterfaceName::new(interface).unwrap(),
            protocol,
            id,
        },
        name,
        count: 1,
    }
}

#[test]
fn a_count_is_keyed_by_service_interface_and_command_name_or_id() {
    let get_service = Sm::default()
        .commands()
        .name(Protocol::Tipc, sm::command::GET_SERVICE);
    let query = CONTROL_COMMANDS.name(Protocol::Cmif, QUERY_POINTER_BUFFER_SIZE);
    let cases = [
        (
            count(
                "Sm",
                IpcProtocol::Cmif,
                sm::command::GET_SERVICE,
                get_service,
            ),
            format!("sm::Sm::{}", get_service.unwrap()),
        ),
        // A command the table does not name is written by its id, in decimal.
        (
            count("lm", IpcProtocol::Cmif, 0x15, None),
            "sm::lm::21".to_owned(),
        ),
        // A TIPC command says so: TIPC numbers its commands apart from CMIF.
        (
            count(
                "Sm",
                IpcProtocol::Tipc,
                sm::command::GET_SERVICE,
                get_service,
            ),
            format!("sm::Sm::{} (tipc)", get_service.unwrap()),
        ),
        (
            count("lm", IpcProtocol::Tipc, 0x15, None),
            "sm::lm::21 (tipc)".to_owned(),
        ),
        // A control request's interface says it is one.
        (
            count(
                CONTROL_INTERFACE,
                IpcProtocol::Control,
                QUERY_POINTER_BUFFER_SIZE,
                query,
            ),
            format!("sm::{CONTROL_INTERFACE}::{}", query.unwrap()),
        ),
    ];
    for (counted, key) in cases {
        assert_eq!(command_key("sm:", &counted), key);
    }
}

/// An IPC event `seq` with `outcome`, for `command` on `interface`.
fn ipc(seq: u64, interface: &str, command: u32, outcome: IpcOutcome) -> Event {
    Event {
        seq,
        time: GuestTicks(seq * 10),
        core: 2,
        thread: ThreadId(0x40 + seq),
        kind: EventKind::Ipc {
            session: SessionId(9),
            service: ServiceId(0x31),
            interface: InterfaceName::new(interface).unwrap(),
            protocol: IpcProtocol::Tipc,
            command,
            request_size: 0,
            reply_size: 0,
            result: ResultCode(0),
            outcome,
        },
    }
}

/// What a kept request of [`ipc`]'s is.
fn kept(seq: u64, interface: &str, command: u32) -> StubbedRequest {
    StubbedRequest {
        seq,
        thread: 0x40 + seq,
        service: ServiceId(0x31),
        interface: InterfaceName::new(interface).unwrap(),
        protocol: IpcProtocol::Tipc,
        command,
    }
}

#[test]
fn the_sink_keeps_only_stubbed_requests() {
    let sink = LastStubbed::new();
    sink.record(&ipc(1, "lm", 5, IpcOutcome::Done));
    sink.record(&ipc(2, "lm", 6, IpcOutcome::Stubbed));
    sink.record(&ipc(3, "lm", 7, IpcOutcome::Deferred));
    sink.record(&Event {
        kind: EventKind::GuestLog { len: 1, hash: 2 },
        ..ipc(4, "lm", 8, IpcOutcome::Stubbed)
    });
    assert_eq!(sink.requests(), [kept(2, "lm", 6)]);
}

#[test]
fn the_sink_keeps_the_newest_up_to_its_bound_oldest_first() {
    let bound = u64::try_from(LAST_STUBBED).unwrap();
    let sink = LastStubbed::new();
    for seq in 0..bound {
        sink.record(&ipc(
            seq,
            "lm",
            u32::try_from(seq).unwrap(),
            IpcOutcome::Stubbed,
        ));
    }
    // Exactly at the bound: all of them.
    let all: Vec<_> = (0..bound)
        .map(|seq| kept(seq, "lm", u32::try_from(seq).unwrap()))
        .collect();
    assert_eq!(sink.requests(), all);
    // One more: the oldest goes.
    sink.record(&ipc(bound, "aoc:u", 9, IpcOutcome::Stubbed));
    let mut newest = all[1..].to_vec();
    newest.push(kept(bound, "aoc:u", 9));
    assert_eq!(sink.requests(), newest);
    assert_eq!(sink.requests().len(), LAST_STUBBED);
}

/// A handler that is not a service framework server.
struct Other;

impl SessionHandler for Other {
    fn accept(&mut self, _: ObjectId) -> Result<(), HandlerError> {
        Ok(())
    }

    fn closed(&mut self, _: ObjectId) {}

    fn handle(
        &mut self,
        _: &mut IpcContext<'_>,
        _: ObjectId,
        _: &mut [u8],
    ) -> Result<Handled, HandlerError> {
        Ok(Handled::Replied)
    }

    fn take_reply(
        &mut self,
        _: &mut IpcContext<'_>,
        _: &mut [u8],
    ) -> Option<(ObjectId, Result<(), HandlerError>)> {
        None
    }
}

#[test]
fn reading_takes_each_servers_name_and_skips_handlers_of_other_kinds() {
    let dispatcher = dispatcher();
    let manager = sm::server(Registry::new(), UnknownPolicy::AutoStub);
    dispatcher
        .add_server("sm", ServiceId(0x31), None, Box::new(manager))
        .unwrap();
    dispatcher
        .add_server("other", ServiceId(0x32), None, Box::new(Other))
        .unwrap();
    let report = ServicesReport::read(&dispatcher);
    // The server's own name, not the one the kernel was given for logs.
    assert_eq!(
        report.names,
        BTreeMap::from([(ServiceId(0x31), sm::PORT_NAME.to_owned())])
    );
    assert_eq!(report.service_name(ServiceId(0x31)), sm::PORT_NAME);
    // A number no server has is written in hex.
    assert_eq!(report.service_name(ServiceId(0x32)), "0x32");
    // Nothing was sent, stubbed or asked for.
    assert_eq!(report.commands, BTreeMap::new());
    assert_eq!((report.uncounted, report.stubbed.len()), (0, 0));
    assert_eq!(report.unknown_services, Vec::<String>::new());
}

#[test]
fn a_stubbed_request_is_named_after_its_server() {
    let report = ServicesReport {
        names: BTreeMap::from([(ServiceId(0x31), sm::PORT_NAME.to_owned())]),
        ..ServicesReport::default()
    };
    let other = StubbedRequest {
        service: ServiceId(0x77),
        ..kept(4, "IFoo", 2)
    };
    assert_eq!(
        report.name_requests(&[kept(3, "lm", 5), other]),
        [
            StubbedRequestReport {
                seq: 3,
                thread: 0x43,
                service: sm::PORT_NAME.to_owned(),
                interface: "lm".to_owned(),
                protocol: IpcProtocol::Tipc,
                command: 5,
            },
            StubbedRequestReport {
                seq: 4,
                thread: 0x44,
                service: "0x77".to_owned(),
                interface: "IFoo".to_owned(),
                protocol: IpcProtocol::Tipc,
                command: 2,
            },
        ]
    );
}

/// The stubbed commands are sorted by service, not listed in the order the kernel holds the
/// servers in: `sm:` is added before `apm` and has the lower number, but `apm` sorts first.
#[test]
fn the_stubbed_commands_are_in_service_order_whatever_the_kernels_order() {
    let rig = HleRig::new();
    // The kernel's order and the names' order differ.
    assert!(hle::SM < hle::APM);
    assert!(apm::NAME < sm::PORT_NAME);
    // A command `apm`'s root does not have, and one sent to a session `sm:` hands out for a
    // name nothing provides: each stubbed by its own server.
    let missing = 99;
    assert_eq!(Apm.commands().name(Protocol::Cmif, missing), None);
    let performance = rig.service(apm::NAME);
    // A made-up name, which no service a later change registers can take.
    let unknown = "absent";
    let stub = rig.service(unknown);
    for session in [stub, performance] {
        let (code, _) = rig.send(session, &libnx(missing));
        assert_eq!(code, starquake_kernel::result::ResultCode::SUCCESS);
    }
    let report = ServicesReport::read(&rig.dispatcher);
    let stubbed = |service: &str, interface: &str| StubbedCommand {
        service: service.to_owned(),
        interface: interface.to_owned(),
        protocol: IpcProtocol::Cmif,
        command: missing,
    };
    assert_eq!(
        report.stubbed,
        [
            stubbed(apm::NAME, Apm::interface()),
            stubbed(sm::PORT_NAME, unknown),
        ]
    );
}
