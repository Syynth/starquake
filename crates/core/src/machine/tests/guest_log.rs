//! The services' way into the guest log ([`ServicesLog`]), how the report takes the log
//! ([`take_log`]), and a real run whose `lm` line reaches the report.

use std::cell::RefCell;

use starquake_ipc::{BufferDescriptor, RequestBuilder, StaticDescriptor};
use starquake_services::sm::{self, MAX_NAME_LEN, ServiceName};

use super::reads::{Script, Seen, WithTls, libnx, send};
use super::*;
use crate::hle::tests::log_packet;

/// `lm`'s lines and `OutputDebugString`'s go into one log, in the order they were written, under
/// one limit: once a line does not fit, it and every later line, of either kind, are dropped.
#[test]
fn services_lines_share_the_debug_log_and_its_limit() {
    let log = Arc::new(Mutex::new(GuestLog::new(10)));
    let services = ServicesLog(Arc::clone(&log));
    // `LogSink::output` keeps a debug string this way.
    lock(&log).keep(b"abc");
    lm::GuestLog::line(&services, b"defg");
    lock(&log).keep(b"hij");
    assert_eq!(
        lock(&log).lines,
        [b"abc".to_vec(), b"defg".to_vec(), b"hij".to_vec()]
    );
    assert_eq!(lock(&log).kept_bytes, 10);
    // Exactly at the limit; the next byte is dropped, then everything after it.
    lm::GuestLog::line(&services, b"k");
    lock(&log).keep(b"");
    lm::GuestLog::line(&services, b"");
    let log = take_log(&log);
    assert_eq!(log.lines.len(), 3);
    assert_eq!(log.dropped, 3);
}

/// Taking the log leaves an empty one with the same limit behind, for anything that still
/// holds it.
#[test]
fn taking_the_log_leaves_an_empty_one_with_the_same_limit() {
    let log = Arc::new(Mutex::new(GuestLog::new(4)));
    let services = ServicesLog(Arc::clone(&log));
    lm::GuestLog::line(&services, b"four");
    lm::GuestLog::line(&services, b"x");
    let taken = take_log(&log);
    assert_eq!((taken.lines, taken.dropped), (vec![b"four".to_vec()], 1));
    let left = lock(&log);
    assert_eq!(
        (left.lines.len(), left.kept_bytes, left.limit, left.dropped),
        (0, 0, 4, 0)
    );
}

/// Where in the thread-local page [`lm_and_debug_lines_reach_the_runs_guest_log`] puts the
/// port's name, the debug strings and the log packet, past the message at its start.
const NAME_AT: u64 = 0x300;
const FIRST_AT: u64 = 0x380;
const LAST_AT: u64 = 0x390;
const PACKET_AT: u64 = 0x400;
/// The longest that run may take: its timeout.
const RUN_LIMIT: Duration = Duration::from_secs(30);

/// `Log` of the `len` bytes at `address`, as libnx sends an auto-select buffer: an A descriptor
/// and an empty X one.
fn log_request(address: u64, len: u64) -> RequestBuilder {
    libnx(lm::command::logger::LOG)
        .send_buffer(BufferDescriptor {
            address,
            size: len,
            flags: 0,
        })
        .static_descriptor(StaticDescriptor {
            counter: 0,
            address: 0,
            size: 0,
        })
}

/// The conversation of [`lm_and_debug_lines_reach_the_runs_guest_log`]: `OutputDebugString("a")`,
/// `sm:`'s `Initialize` and `GetService("lm")`, `OpenLogger`, a `Log` of one whole message
/// whose text is `b`, `OutputDebugString("c")`, then `ExitProcess`.
fn logging_guest(view: &Script) -> ScriptedVcpu {
    let base = view.base;
    view.tls.write(base + NAME_AT, b"sm:\0").unwrap();
    view.tls.write(base + FIRST_AT, b"a").unwrap();
    view.tls.write(base + LAST_AT, b"c").unwrap();
    let packet = log_packet(b"b");
    view.tls.write(base + PACKET_AT, &packet).unwrap();
    let packet_len = u64::try_from(packet.len()).unwrap();

    let sm_handle = Rc::new(RefCell::new(0u64));
    let (keep, take) = (Rc::clone(&sm_handle), Rc::clone(&sm_handle));
    ScriptedVcpu::new()
        .then(
            move |regs| {
                regs.x[0] = base + FIRST_AT;
                regs.x[1] = 1;
            },
            svc(SvcId::OutputDebugString),
        )
        .then(
            move |regs| {
                assert_eq!(regs.x[0], 0, "OutputDebugString failed");
                regs.x[1] = base + NAME_AT;
            },
            svc(SvcId::ConnectToNamedPort),
        )
        .then(
            {
                let guest = view.clone();
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
                view,
                move |_| *take.borrow(),
                |guest| {
                    guest.read_reply(0);
                    let name: [u8; MAX_NAME_LEN] = ServiceName::new(lm::NAME).unwrap().to_field();
                    guest.write(&libnx(sm::command::GET_SERVICE).data(&name));
                },
            ),
            svc(SvcId::SendSyncRequest),
        )
        .then(
            send(
                view,
                |guest| {
                    guest.read_reply(0);
                    guest.last_handle()
                },
                |guest| {
                    guest.write(
                        &libnx(lm::command::lm::OPEN_LOGGER)
                            .send_pid(0)
                            .data(&0_u64.to_le_bytes()),
                    );
                },
            ),
            svc(SvcId::SendSyncRequest),
        )
        .then(
            send(
                view,
                |guest| {
                    guest.read_reply(0);
                    guest.last_handle()
                },
                move |guest| guest.write(&log_request(base + PACKET_AT, packet_len)),
            ),
            svc(SvcId::SendSyncRequest),
        )
        .then(
            {
                let guest = view.clone();
                move |regs| {
                    assert_eq!(regs.x[0], 0, "Log failed");
                    guest.read_reply(0);
                    regs.x[0] = base + LAST_AT;
                    regs.x[1] = 1;
                }
            },
            svc(SvcId::OutputDebugString),
        )
        .then(
            |regs| assert_eq!(regs.x[0], 0, "OutputDebugString failed"),
            svc(SvcId::ExitProcess),
        )
}

/// A real run (`run`, so `real_services`): the guest prints `a` with `OutputDebugString`, asks
/// `sm:` for `lm`, opens a logger and logs a whole message whose text is `b`, then prints `c`
/// ([`logging_guest`]). The run's guest log (`guest_log.txt`) holds all three, in that order:
/// `lm`'s line reaches the run's own log, not one the services keep apart.
#[test]
fn lm_and_debug_lines_reach_the_runs_guest_log() {
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
    let mut main = logging_guest(&view);
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
    assert_eq!(report.exit, ExitReason::ProcessExited, "{report:?}");
    // Initialize, GetService, OpenLogger and Log: each answered with success.
    assert_eq!(seen.results, [0, 0, 0, 0]);
    assert_eq!(
        report.guest_log,
        [b"a".to_vec(), b"b".to_vec(), b"c".to_vec()]
    );
    assert_eq!(report.guest_log_dropped, 0);
}
