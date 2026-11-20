//! The graphics server in a real run (`run`): a scripted guest gets the graphics server's root
//! from `appletOE`, asks its gpu worker for an ordered job (a sparse region's destroy), and waits
//! on the job's event with no timeout. Only the timer thread's turn over the graphics server's
//! completion queue (`machine/timer.rs`, "The graphics server's turn") can end that wait, and
//! the core loop must not take the run for stalled while the queue holds the job's completion
//! (`crate::hle::graphics`, "Telling a stalled run").
//!
//! A second scripted guest binds the shared syncpoint page with 204 and asks it, again and
//! again, about a counter that does not move, so each request is held to the page's bound and
//! only the timer thread's turn at that bound answers it, well before the next preemption. On a
//! machine whose cores never step, a completion queued for the graphics server rings the timer
//! thread, and the turn reads the environment only when the graphics server is not idle.
//!
//! Requests are laid out by hand: libnx's for `sm:` and `appletOE` (`libnx/nx/source/services/
//! sm.c`, `applet.c` @ b156b477), and the TIPC header words PR compares for gsv's root and gpu
//! sessions (`eden-pr4588/src/core/hle/service/am/service/graphics_supervisor.cpp` and
//! `gpu_session.cpp` @ e5fb3b97), as `crate::hle::tests::graphics` lays them out.

use std::cell::RefCell;

use starquake_gpu::interface::ChannelId;
use starquake_gsv::gpu::command as gpu_command;
use starquake_gsv::note::Note;
use starquake_gsv::protocol::command as root_command;
use starquake_gsv::shared::page::{
    ANSWER_BOUND, GROUP_REQUEST_OFFSET, READ_ONLY_SIZE, SLOT_REQUEST_OFFSET, WRITABLE_SIZE,
};
use starquake_ipc::{
    BUFFER_SIZE, Body, CommandType, Message, RequestBuilder, RequestLayout, SessionKind,
    TIPC_COMMAND_OFFSET,
};
use starquake_kernel::memory::SvcPermission;
use starquake_kernel::sched::PREEMPTION_INTERVAL_NS;
use starquake_services::am;
use starquake_services::framework::control::CONVERT_CURRENT_OBJECT_TO_DOMAIN;
use starquake_services::sm::{self, MAX_NAME_LEN, ServiceName};
use starquake_testkit::a64;
use starquake_testkit::nro::NroBuilder;

use super::reads::WithTls;
use super::*;

mod frames;
mod stall;
mod vsync;

/// Where in the thread-local page the port's name and the waited handle are.
const NAME_AT: u64 = 0x300;
const HANDLES_AT: u64 = 0x400;

/// The longest the run may take: its timeout.
const RUN_LIMIT: Duration = Duration::from_secs(30);

/// A sparse region inside NVN's window (row 25), eight pages.
const REGION: u64 = 0x81_0000_0000;
const REGION_SIZE: u64 = 8 * 0x1000;

/// What the steps carry from one reply to the next.
#[derive(Default)]
struct Vars {
    sm: u32,
    oe: u32,
    object: u32,
    gsv: u32,
    gpu: u32,
    worker: u32,
    job: u32,
    device: u32,
    space: u32,
    /// Set by the last step: the wait ended with success.
    waited: Option<u64>,
    /// The shared page's guest: its two transfer memories, and 204's request and answer events.
    writable: u32,
    read_only: u32,
    request: u32,
    answer: u32,
    /// When the page guest's last request was signalled.
    asked: Option<Instant>,
    /// Each of the page guest's waits for an answer: its result and how long from the signal.
    answers: Vec<(u64, Duration)>,
    /// The presenting guest (`frames`): its native memory, its image and its vi2 session.
    native: u32,
    image: u32,
    vi2: u32,
}

/// The scripted guest's view of its thread-local page and its variables.
#[derive(Clone)]
struct Script {
    tls: Arc<FlatMemory>,
    base: u64,
    vars: Rc<RefCell<Vars>>,
}

impl Script {
    fn write_message(&self, bytes: &[u8]) {
        let mut buffer = [0; BUFFER_SIZE];
        buffer[..bytes.len()].copy_from_slice(bytes);
        self.tls.write(self.base, &buffer).unwrap();
    }

    fn write(&self, request: &RequestBuilder) {
        let mut buffer = [0; BUFFER_SIZE];
        request.build(&mut buffer).unwrap();
        self.write_message(&buffer);
    }

    fn write_words(&self, words: &[u32]) {
        let bytes: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
        self.write_message(&bytes);
    }

    fn message(&self) -> [u8; BUFFER_SIZE] {
        let mut buffer = [0; BUFFER_SIZE];
        self.tls.read(self.base, &mut buffer).unwrap();
        buffer
    }

    /// A CMIF reply on a session of `kind` with `size` data bytes: its data, object ids, and
    /// copied and moved handles. Its result must be success.
    fn cmif(&self, kind: SessionKind, size: usize) -> (Vec<u8>, Vec<u32>, Vec<u32>, Vec<u32>) {
        let buffer = self.message();
        let message = Message::parse_reply_sized(&buffer, kind, size).unwrap();
        let Body::CmifReply(body) = message.body else {
            unreachable!("not a CMIF reply");
        };
        assert_eq!(body.header.result, 0);
        (
            body.data.to_vec(),
            body.object_ids,
            message.copy_handles,
            message.move_handles,
        )
    }

    /// A TIPC reply: its data words and copied handles. Its result must be success.
    fn tipc(&self) -> (Vec<u32>, Vec<u32>) {
        let buffer = self.message();
        let message = Message::parse_reply(&buffer, SessionKind::Session).unwrap();
        let Body::TipcReply { result, data } = message.body else {
            unreachable!("not a TIPC reply");
        };
        assert_eq!(result, 0);
        let words = data
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| u32::from_le_bytes(*word))
            .collect();
        (words, message.copy_handles)
    }
}

fn libnx(id: u32) -> RequestBuilder {
    RequestBuilder::cmif(CommandType::Request, id).layout(RequestLayout::Libnx)
}

fn tipc_type(command: u32) -> u32 {
    u32::from(TIPC_COMMAND_OFFSET) + command
}

/// The id of the command `name` of `interface`, from its table.
fn command_of(interface: &dyn starquake_services::framework::Session, name: &str) -> u32 {
    let table = interface.commands();
    table
        .cmif
        .iter()
        .find(|command| command.name == name)
        .map(|command| command.id)
        .unwrap()
}

fn halves(value: u64) -> [u32; 2] {
    let bytes = value.to_le_bytes();
    [
        u32::from_le_bytes(bytes[..4].try_into().unwrap()),
        u32::from_le_bytes(bytes[4..].try_into().unwrap()),
    ]
}

/// A step that, once the call before succeeded, runs `next` on the script and sends its
/// request to the handle it returns.
fn step(
    script: &Script,
    next: impl FnOnce(&Script, &mut Vars) -> u32 + 'static,
) -> impl FnOnce(&mut GuestRegs) + 'static {
    let script = script.clone();
    move |regs| {
        assert_eq!(regs.x[0], 0, "the call before failed");
        let mut vars = script.vars.borrow_mut();
        regs.x[0] = u64::from(next(&script, &mut vars));
    }
}

/// The scripted guest's first steps: from `sm:` to `ISelfController` 2000, whose reply (the
/// graphics server's root, copied) the next step reads.
fn asking_for_the_root(script: &Script) -> ScriptedVcpu {
    let name = script.base + NAME_AT;
    script.tls.write(name, b"sm:\0").unwrap();
    let send = || svc(SvcId::SendSyncRequest);
    let take_sm = script.clone();
    ScriptedVcpu::new()
        .then(move |regs| regs.x[1] = name, svc(SvcId::ConnectToNamedPort))
        .then(
            move |regs| {
                assert_eq!(regs.x[0], 0, "ConnectToNamedPort failed");
                let sm = u32::try_from(regs.x[1]).unwrap();
                take_sm.vars.borrow_mut().sm = sm;
                take_sm.write(
                    &libnx(sm::command::INITIALIZE)
                        .send_pid(0)
                        .data(&0_u64.to_le_bytes()),
                );
                regs.x[0] = u64::from(sm);
            },
            send(),
        )
        .then(
            step(script, |script, vars| {
                script.cmif(SessionKind::Session, 0);
                let name: [u8; MAX_NAME_LEN] = ServiceName::new(am::OE_NAME).unwrap().to_field();
                script.write(&libnx(sm::command::GET_SERVICE).data(&name));
                vars.sm
            }),
            send(),
        )
        .then(
            step(script, |script, vars| {
                vars.oe = script.cmif(SessionKind::Session, 0).3[0];
                script.write(
                    &RequestBuilder::cmif(CommandType::Control, CONVERT_CURRENT_OBJECT_TO_DOMAIN)
                        .layout(RequestLayout::Libnx),
                );
                vars.oe
            }),
            send(),
        )
        .then(
            step(script, |script, vars| {
                let (data, ..) = script.cmif(SessionKind::Session, 4);
                let root = u32::from_le_bytes(data.try_into().unwrap());
                let open = command_of(&am::IApplicationProxyService, "open_application_proxy");
                script.write(
                    &RequestBuilder::domain(CommandType::Request, root, open)
                        .send_pid(0)
                        .copy_handle(Handle::CURRENT_PROCESS.raw())
                        .data(&0_u64.to_le_bytes()),
                );
                vars.oe
            }),
            send(),
        )
        .then(
            step(script, |script, vars| {
                vars.object = script.cmif(SessionKind::Domain, 0).1[0];
                let proxy = am::IApplicationProxy::default();
                let get = command_of(&proxy, "get_self_controller");
                script.write(&RequestBuilder::domain(
                    CommandType::Request,
                    vars.object,
                    get,
                ));
                vars.oe
            }),
            send(),
        )
        .then(
            step(script, |script, vars| {
                vars.object = script.cmif(SessionKind::Domain, 0).1[0];
                let get = command_of(&am::ISelfController, "get_graphics_supervisor");
                script.write(&RequestBuilder::domain(
                    CommandType::Request,
                    vars.object,
                    get,
                ));
                vars.oe
            }),
            send(),
        )
}

/// The scripted guest (module docs), ending with `ExitProcess`.
#[expect(
    clippy::too_many_lines,
    reason = "one step per request of the conversation, in order"
)]
fn waiting_guest(script: &Script) -> ScriptedVcpu {
    let send = || svc(SvcId::SendSyncRequest);
    let wait = script.clone();
    let done = script.clone();
    let [first_low, first_high] = halves(REGION);
    let [last_low, last_high] = halves(REGION + REGION_SIZE - 1);
    asking_for_the_root(script)
        .then(
            step(script, |script, vars| {
                vars.gsv = script.cmif(SessionKind::Domain, 0).2[0];
                // 497 (`0x201, 2`, the flags 4 at words 2-3).
                script.write_words(&[tipc_type(root_command::OPEN_GPU_SESSION), 2, 4, 0]);
                vars.gsv
            }),
            send(),
        )
        .then(
            step(script, |script, vars| {
                vars.gpu = script.tipc().1[0];
                // gpu 12 on the last index (`0x1c, 1`, the index at word 2).
                script.write_words(&[tipc_type(gpu_command::GET_NOTIFICATION_EVENT), 1, 63]);
                vars.gpu
            }),
            send(),
        )
        .then(
            step(script, |script, vars| {
                script.tipc();
                script.write_words(&[tipc_type(gpu_command::REQUEST_WORKER), 1, 63]);
                vars.gpu
            }),
            send(),
        )
        .then(
            step(script, |script, vars| {
                script.tipc();
                script.write_words(&[tipc_type(gpu_command::ACQUIRE_WORKER), 0]);
                vars.gpu
            }),
            send(),
        )
        .then(
            step(script, |script, vars| {
                vars.worker = script.tipc().1[0];
                script.write_words(&[tipc_type(gpu_command::GET_NOTIFICATION_EVENT), 1, 0]);
                vars.gpu
            }),
            send(),
        )
        .then(
            step(script, |script, vars| {
                vars.job = script.tipc().1[0];
                // 41 (`0x39, 6`): name, flags 0, label.
                script.write_words(&[
                    tipc_type(gpu_command::CREATE_GRAPHICS_DEVICE),
                    6,
                    0x11,
                    0,
                    0,
                    0,
                    0x22,
                    0,
                ]);
                vars.gpu
            }),
            send(),
        )
        .then(
            step(script, |script, vars| {
                vars.device = script.tipc().0[0];
                script.write_words(&[tipc_type(gpu_command::CREATE_ADDRESS_SPACE), 1, 0]);
                vars.gpu
            }),
            send(),
        )
        .then(
            step(script, move |script, vars| {
                vars.space = script.tipc().0[0];
                // 214 (`0xe6, 10`): space, first, last, page shift 12, mode 0, device, sync 0,
                // slot 0.
                script.write_words(&[
                    tipc_type(gpu_command::CREATE_SPARSE_REGION),
                    10,
                    vars.space,
                    first_low,
                    first_high,
                    last_low,
                    last_high,
                    12,
                    0,
                    vars.device,
                    0,
                    0,
                ]);
                vars.worker
            }),
            send(),
        )
        .then(
            step(script, |script, vars| {
                script.tipc();
                // The create signalled the job's event at once: clear it.
                vars.job
            }),
            svc(SvcId::ResetSignal),
        )
        .then(
            step(script, move |script, vars| {
                // 215 (`0xe7, 7`): space, first, last, sync 0, slot 0.
                script.write_words(&[
                    tipc_type(gpu_command::DESTROY_SPARSE_REGION),
                    7,
                    vars.space,
                    first_low,
                    first_high,
                    last_low,
                    last_high,
                    0,
                    0,
                ]);
                vars.worker
            }),
            send(),
        )
        .then(
            move |regs| {
                assert_eq!(regs.x[0], 0, "215 failed");
                wait.tipc();
                let job = wait.vars.borrow().job;
                wait.tls
                    .write(wait.base + HANDLES_AT, &job.to_le_bytes())
                    .unwrap();
                regs.x[1] = wait.base + HANDLES_AT;
                regs.x[2] = 1;
                regs.x[3] = u64::MAX;
            },
            svc(SvcId::WaitSynchronization),
        )
        .then(
            move |regs| done.vars.borrow_mut().waited = Some(regs.x[0]),
            svc(SvcId::ExitProcess),
        )
}

/// The guest waits, with no timeout, on an event only the timer thread's graphics turn signals:
/// the run carries on past the wait and the guest exits, and the turn's note is in the trace.
#[test]
fn the_timer_thread_s_graphics_turn_ends_a_guest_s_wait() {
    let guest = super::Guest::new(&programs::hello().unwrap());
    let page = usize::try_from(crate::image::PAGE).unwrap();
    let tls = Arc::new(FlatMemory::new(guest.tls, page));
    let space = WithTls {
        image: &guest.memory,
        tls: Arc::clone(&tls),
    };
    let vars = Rc::new(RefCell::new(Vars::default()));
    let script = Script {
        tls,
        base: guest.tls,
        vars: Rc::clone(&vars),
    };
    let mut main = waiting_guest(&script);
    let options = RunOptions {
        timeout: Some(RUN_LIMIT),
        command_line: Vec::new(),
        sd_card: None,
        romfs: None,
        rtc: RtcAtBoot::eden_default(),
        environment: crate::EnvSource::Live,
        until_frames: None,
    };
    let report = run(
        vcpus(&mut main),
        guest.program(&space),
        &options,
        Instant::now(),
        timebase(),
    );
    drop((script, main));
    assert_eq!(report.exit, ExitReason::ProcessExited, "{}", report.trace);
    assert_eq!(vars.borrow().waited, Some(0));
    // The drain's note: gsv's code 3, `GpuOrderedDone`, recorded for the root's number.
    let note = format!(
        "note service={} code={}",
        crate::hle::GSV_ROOT.0,
        starquake_gsv::note::GPU_ORDERED_DONE
    );
    assert!(report.trace.contains(&note), "{}", report.trace);
}

// ----- The shared page through the real timer thread -------------------------------------------

/// The events of a run's flight recorder, in order, each with its time on the environment's
/// clock in nanoseconds: the event's guest time at the run's counter rate, rounded down by less
/// than [`counter_tick`].
fn timed_events(report: &RunReport) -> Vec<(EventKind, u64)> {
    let hz = u128::from(timebase().rate().hz());
    Dump::from_bytes(&report.events)
        .unwrap()
        .merged()
        .into_iter()
        .map(|event| {
            let nanos = u128::from(event.time.0) * 1_000_000_000 / hz;
            (event.kind, u64::try_from(nanos).unwrap())
        })
        .collect()
}

/// One tick of the run's counter in nanoseconds, rounded up: the most [`timed_events`] rounds
/// a time down by.
fn counter_tick() -> u64 {
    1_000_000_000_u64.div_ceil(timebase().rate().hz())
}

/// The notes with `code` that gsv's root recorded in `events`, in order: each one's value and
/// time.
fn gsv_notes(events: &[(EventKind, u64)], code: u32) -> Vec<(u64, u64)> {
    events
        .iter()
        .filter_map(|&(kind, time)| match kind {
            EventKind::ServiceNote {
                service,
                code: noted,
                value,
            } if service == crate::hle::GSV_ROOT && noted == code => Some((value, time)),
            _ => None,
        })
        .collect()
}

/// How many times the page guest asks: the first may be answered at once (the page shows no
/// value yet), every later one is held to the bound, its counter unmoved.
const PAGE_ROUNDS: usize = 9;

/// The longest the page guest waits for one answer: a wait that ends so is a failure, and keeps
/// a run whose turns never answer (a turn at a time that never passes the bound) bounded.
const ANSWER_WAIT: Duration = Duration::from_millis(500);

/// A syncpoint the page guest asks about: group 0, bit 40, as `crate::hle`'s page tests ask.
const ASKED: u8 = 40;

/// A program whose `.bss` holds the page's two transfer memories, page-aligned: its code is
/// never run (a scripted vCPU makes its exits).
fn page_program() -> Vec<u8> {
    let pages = (WRITABLE_SIZE + READ_ONLY_SIZE) / 0x1000 + 1;
    NroBuilder::new(&[a64::nop()])
        .bss_size(pages * 0x1000)
        .build()
        .unwrap()
}

/// A page guest's memory: the page's two transfer memories (`.bss` pages the kernel maps, held
/// apart so the script can write its requests there), over the image and its thread-local page.
struct WithPage<'a> {
    page: Arc<FlatMemory>,
    rest: WithTls<'a>,
}

impl GuestMemory for WithPage<'_> {
    fn read(&self, va: u64, buf: &mut [u8]) -> Result<(), Fault> {
        self.page.read(va, buf).or_else(|_| self.rest.read(va, buf))
    }

    fn write(&self, va: u64, data: &[u8]) -> Result<(), Fault> {
        self.page
            .write(va, data)
            .or_else(|_| self.rest.write(va, data))
    }
}

impl<V> GuestSpace<V> for WithPage<'_> {
    fn touch(&self, _: u64, _: Access) -> Touch {
        Touch::Refused
    }

    fn settle(&self, _: &mut V) -> Result<(), starquake_cpu::Error> {
        Ok(())
    }
}

/// The page guest's first steps: gets the root, lends the writable memory at `writable` and the
/// read-only one right after it, and sends 204 with both, whose reply the next step reads.
fn binding_the_page(script: &Script, writable: u64) -> ScriptedVcpu {
    let read_only = writable + u64::from(WRITABLE_SIZE);
    let lend_writable = script.clone();
    let lend_read_only = script.clone();
    let bind = script.clone();
    asking_for_the_root(script)
        .then(
            move |regs| {
                assert_eq!(regs.x[0], 0, "2000 failed");
                let mut vars = lend_writable.vars.borrow_mut();
                vars.gsv = lend_writable.cmif(SessionKind::Domain, 0).2[0];
                regs.x[1] = writable;
                regs.x[2] = u64::from(WRITABLE_SIZE);
                regs.x[3] = u64::from(SvcPermission::None.bits());
            },
            svc(SvcId::CreateTransferMemory),
        )
        .then(
            move |regs| {
                assert_eq!(regs.x[0], 0, "lending the writable memory failed");
                lend_read_only.vars.borrow_mut().writable = u32::try_from(regs.x[1]).unwrap();
                regs.x[1] = read_only;
                regs.x[2] = u64::from(READ_ONLY_SIZE);
                regs.x[3] = u64::from(SvcPermission::Read.bits());
            },
            svc(SvcId::CreateTransferMemory),
        )
        .then(
            move |regs| {
                assert_eq!(regs.x[0], 0, "lending the read-only memory failed");
                let mut vars = bind.vars.borrow_mut();
                vars.read_only = u32::try_from(regs.x[1]).unwrap();
                // 204 as PR checks it (`0xdc, 0x80000006, 4`, two copied handles, then `1,
                // 0x1ff, 0x1000, 0, 0x2000, 0`), as `crate::hle`'s page tests send it.
                bind.write_words(&[
                    tipc_type(root_command::BIND_SYNCPOINT_MEMORY),
                    0x8000_0006,
                    4,
                    vars.writable,
                    vars.read_only,
                    1,
                    0x1FF,
                    WRITABLE_SIZE,
                    0,
                    READ_ONLY_SIZE,
                    0,
                ]);
                regs.x[0] = u64::from(vars.gsv);
            },
            svc(SvcId::SendSyncRequest),
        )
}

/// The scripted page guest: gets the root, lends the writable memory at the start of `page`
/// and the read-only one right after it, binds them with 204, then [`PAGE_ROUNDS`] times flips
/// its request bits (group 0 and [`ASKED`]), signals the request event, waits for the answer
/// event (at most [`ANSWER_WAIT`]) and clears it, timing each wait from the signal; then exits.
fn page_guest(script: &Script, page: &Arc<FlatMemory>, writable: u64) -> ScriptedVcpu {
    let mut guest = binding_the_page(script, writable);
    for round in 0..PAGE_ROUNDS {
        let ask = script.clone();
        let wait = script.clone();
        let clear = script.clone();
        let requests = Arc::clone(page);
        guest = guest
            .then(
                move |regs| {
                    let mut vars = ask.vars.borrow_mut();
                    if round == 0 {
                        assert_eq!(regs.x[0], 0, "204 failed");
                        let handles = ask.tipc().1;
                        (vars.request, vars.answer) = (handles[0], handles[1]);
                    } else {
                        assert_eq!(regs.x[0], 0, "clearing the answer failed");
                    }
                    // Flip the group's summary bit and the slot's bit.
                    let summary = u64::from(round % 2 == 0);
                    let slot = summary << ASKED;
                    let request_at = writable + u64::from(GROUP_REQUEST_OFFSET);
                    let slot_at = writable + u64::from(SLOT_REQUEST_OFFSET);
                    requests.write(request_at, &summary.to_le_bytes()).unwrap();
                    requests.write(slot_at, &slot.to_le_bytes()).unwrap();
                    vars.asked = Some(Instant::now());
                    regs.x[0] = u64::from(vars.request);
                },
                svc(SvcId::SignalEvent),
            )
            .then(
                move |regs| {
                    assert_eq!(regs.x[0], 0, "signalling the request failed");
                    let answer = wait.vars.borrow().answer;
                    wait.tls
                        .write(wait.base + HANDLES_AT, &answer.to_le_bytes())
                        .unwrap();
                    regs.x[1] = wait.base + HANDLES_AT;
                    regs.x[2] = 1;
                    regs.x[3] = u64::try_from(ANSWER_WAIT.as_nanos()).unwrap();
                },
                svc(SvcId::WaitSynchronization),
            )
            .then(
                move |regs| {
                    let mut vars = clear.vars.borrow_mut();
                    let waited = vars.asked.take().unwrap().elapsed();
                    vars.answers.push((regs.x[0], waited));
                    regs.x[0] = u64::from(vars.answer);
                },
                svc(SvcId::ResetSignal),
            );
    }
    let done = script.clone();
    guest.then(
        move |regs| {
            assert_eq!(regs.x[0], 0, "clearing the answer failed");
            done.vars.borrow_mut().waited = Some(0);
        },
        svc(SvcId::ExitProcess),
    )
}

/// A held page request is answered by the real timer thread at the bound, well before the next
/// preemption: each request after the first (one whose counter has not moved, so held to
/// [`ANSWER_BOUND`]) is answered, by the median, at least the bound after the guest asked (by the
/// guest's own clock) and less than half of [`PREEMPTION_INTERVAL_NS`] after it (by the flight
/// recorder: from the request's `SignalEvent` to the `SyncptPageAcknowledged` note the timer
/// thread records as it answers; `observability.md`, "Stalls and host timing"). Each of these
/// makes that so, and without either the request waits for the next preemption tick: the
/// thread sleeps no later than the page's deadline, and the turn reads its time from the
/// environment's clock (a turn at a time that never passes the bound never answers: every wait
/// then ends at [`ANSWER_WAIT`]).
///
/// It times the timer thread, not the guest's wake, which also waits for the core's host thread
/// to be scheduled: timed by the guest's clock, under `taskpolicy -c utility` with 16 busy
/// processes on the same Mac, it failed 1 run in 40 and 9 in 60 with the timer thread on time.
/// So it no longer sees that the cores are rescheduled after the turn, which wakes the waiting
/// thread's core at once. It cannot see whether the queue rings the timer thread either: the
/// guest's timed wait gives the kernel a deadline, which wakes the thread soon after the signal
/// anyway. `a_completion_queued_for_the_graphics_server_rings_the_timer_thread` pins the ring.
///
/// The thread must also sleep *to* the bound where the host ends timed waits late on purpose
/// (macOS timer coalescing; `machine/timer.rs`, "Sleeping to a deadline"). Asking for each sleep
/// in one wait, as it did before it measured that lateness, CI's hosted macOS runners saw
/// medians of 5.07 to 5.19 ms by the guest's clock (held waits of 1.75 to 5.68 ms, 5 runs), and
/// this test failed 17 runs of 20 under `taskpolicy -c utility` on an Apple Silicon Mac. With
/// the waits sized by the measured lateness its median was 1.14 ms, 2.6 ms at worst (30 runs,
/// with other builds loading the Mac); without the sleep to the page's deadline it failed 20
/// runs of 20.
#[test]
fn a_held_page_request_is_answered_at_the_bound_by_the_timer_thread() {
    let guest = super::Guest::new(&page_program());
    let bss = guest.start_of("main.bss");
    let writable = bss.next_multiple_of(crate::image::PAGE);
    let page_size = usize::try_from(crate::image::PAGE).unwrap();
    let tls = Arc::new(FlatMemory::new(guest.tls, page_size));
    let lent = usize::try_from(WRITABLE_SIZE + READ_ONLY_SIZE).unwrap();
    let page = Arc::new(FlatMemory::new(writable, lent));
    let space = WithPage {
        page: Arc::clone(&page),
        rest: WithTls {
            image: &guest.memory,
            tls: Arc::clone(&tls),
        },
    };
    let vars = Rc::new(RefCell::new(Vars::default()));
    let script = Script {
        tls,
        base: guest.tls,
        vars: Rc::clone(&vars),
    };
    let mut main = page_guest(&script, &page, writable);
    let options = RunOptions {
        timeout: Some(RUN_LIMIT),
        command_line: Vec::new(),
        sd_card: None,
        romfs: None,
        rtc: RtcAtBoot::eden_default(),
        environment: crate::EnvSource::Live,
        until_frames: None,
    };
    let report = run(
        vcpus(&mut main),
        guest.program(&space),
        &options,
        Instant::now(),
        timebase(),
    );
    drop((script, main));
    assert_eq!(report.exit, ExitReason::ProcessExited, "{}", report.trace);
    let vars = vars.borrow();
    assert_eq!(vars.waited, Some(0));
    assert_eq!(vars.answers.len(), PAGE_ROUNDS);
    for (round, (result, _)) in vars.answers.iter().enumerate() {
        assert_eq!(*result, 0, "wait {round} ended with {result:#x}");
    }
    // At least the bound, by the guest's own clock (read before its signal and after its wake).
    let mut waited: Vec<Duration> = vars.answers[1..]
        .iter()
        .map(|(_, waited)| *waited)
        .collect();
    waited.sort();
    let bound = Duration::from_nanos(ANSWER_BOUND);
    assert!(waited[waited.len() / 2] >= bound, "held waits {waited:?}");
    // Well under half a preemption interval, by the flight recorder: from each request's
    // `SignalEvent` to the `SyncptPageAcknowledged` note the timer thread records as it answers.
    let events = timed_events(&report);
    let asked: Vec<u64> = events
        .iter()
        .filter_map(|&(kind, time)| match kind {
            EventKind::Svc { id, result, .. }
                if id == u32::from(SvcId::SignalEvent.number()) && result.0 == 0 =>
            {
                Some(time)
            }
            _ => None,
        })
        .collect();
    let answered = gsv_notes(&events, starquake_gsv::note::SYNCPT_PAGE_ACKNOWLEDGED);
    assert_eq!((asked.len(), answered.len()), (PAGE_ROUNDS, PAGE_ROUNDS));
    let mut held: Vec<Duration> = asked[1..]
        .iter()
        .zip(&answered[1..])
        .map(|(&asked, &(_, answered))| {
            assert!(answered > asked, "{asked} {answered}");
            Duration::from_nanos(answered + counter_tick() - asked)
        })
        .collect();
    held.sort();
    let median = held[held.len() / 2];
    let preemption = Duration::from_nanos(PREEMPTION_INTERVAL_NS);
    assert!(median < preemption / 2, "held answers {held:?}");
    // Each held request's answer is in the trace: the value and the acknowledge notes.
    let acknowledged = format!(
        "note service={} code={}",
        crate::hle::GSV_ROOT.0,
        starquake_gsv::note::SYNCPT_PAGE_ACKNOWLEDGED
    );
    assert_eq!(
        report.trace.matches(&acknowledged).count(),
        PAGE_ROUNDS,
        "{}",
        report.trace
    );
}

// ----- The turn on a machine whose cores never step ---------------------------------------------

/// How far each reading of the machine's environment clock is from the last.
const CLOCK_STEP: u64 = 1_000;

/// Runs `check` on a machine of the hello program with a real run's services and no thread of
/// its own running (no core steps, no timer thread), its environment's clock reading
/// [`CLOCK_STEP`] further each time it is read.
fn on_a_machine(check: impl FnOnce(&Machine<'_, ScriptedVcpu, FixedClock>)) {
    let guest = super::Guest::new(&programs::hello().unwrap());
    let space: &dyn GuestSpace<ScriptedVcpu> = &guest.memory;
    let options = RunOptions::default();
    let session = Session {
        memory: space,
        options: &options,
        clock: RunClock::new(Timebase {
            console: NRO_CONSOLE,
            clock: FixedClock::new(CLOCK_STEP, CLOCK_STEP),
        }),
        recorder: Arc::new(Recorder::new()),
        gpu_memory: None,
        render: crate::renderer::RenderSetup::recording(),
    };
    let env = session.environment();
    let (dispatcher, _) = boot(guest.address_space(), &guest.placed, space, &*env).unwrap();
    let Ok(machine) = Machine::start(
        dispatcher,
        space,
        &session,
        &env,
        Limits::DEFAULT,
        &real_services,
    ) else {
        unreachable!("the services could not be installed");
    };
    check(&machine);
    drop(machine.finish());
}

/// A completion queued for the run's graphics server rings the timer thread's bell: the machine
/// gives the graphics server its bell as the queue's wake (`machine/timer.rs`, "The graphics
/// server's turn").
#[test]
fn a_completion_queued_for_the_graphics_server_rings_the_timer_thread() {
    on_a_machine(|machine| {
        assert!(!machine.bell.take_ring());
        let room = Note::QueueRoom {
            channel: ChannelId(crate::hle::graphics::FIRST_CHANNEL),
        };
        machine.graphics.replay(room).unwrap();
        assert!(machine.bell.take_ring());
        assert!(!machine.bell.take_ring());
    });
}

/// The timer thread's graphics turn reads nothing while the graphics server is idle (the clock
/// read just before and just after it are one step apart), and reads the clock, at least, once
/// something is queued, which it drains.
#[test]
fn an_idle_graphics_turn_reads_nothing_and_a_busy_one_reads_the_clock() {
    on_a_machine(|machine| {
        let reading = || machine.env.now().unwrap().0;
        assert!(machine.graphics.is_idle());
        let before = reading();
        assert!(timer::graphics(machine).is_ok());
        assert_eq!(reading() - before, CLOCK_STEP);

        let room = Note::QueueRoom {
            channel: ChannelId(crate::hle::graphics::FIRST_CHANNEL),
        };
        machine.graphics.replay(room).unwrap();
        assert!(!machine.graphics.is_idle());
        let before = reading();
        assert!(timer::graphics(machine).is_ok());
        assert!(reading() - before > CLOCK_STEP);
        assert!(machine.graphics.is_idle());
    });
}
