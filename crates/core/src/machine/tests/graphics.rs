//! The graphics server in a real run (`run`): a scripted guest gets the graphics server's root
//! from `appletOE`, asks its gpu worker for an ordered job (a sparse region's destroy), and waits
//! on the job's event with no timeout. Only the timer thread's turn over the graphics server's
//! completion queue (`machine/timer.rs`, "The graphics server's turn") can end that wait, and
//! the core loop must not take the run for stalled while the queue holds the job's completion
//! (`crate::hle::graphics`, "Telling a stalled run").
//!
//! Requests are laid out by hand: libnx's for `sm:` and `appletOE` (`libnx/nx/source/services/
//! sm.c`, `applet.c` @ b156b477), and the TIPC header words PR compares for gsv's root and gpu
//! sessions (`eden-pr4588/src/core/hle/service/am/service/graphics_supervisor.cpp` and
//! `gpu_session.cpp` @ e5fb3b97), as `crate::hle::tests::graphics` lays them out.

use std::cell::RefCell;

use starquake_gsv::gpu::command as gpu_command;
use starquake_gsv::protocol::command as root_command;
use starquake_ipc::{
    BUFFER_SIZE, Body, CommandType, Message, RequestBuilder, RequestLayout, SessionKind,
    TIPC_COMMAND_OFFSET,
};
use starquake_services::am;
use starquake_services::framework::control::CONVERT_CURRENT_OBJECT_TO_DOMAIN;
use starquake_services::sm::{self, MAX_NAME_LEN, ServiceName};

use super::reads::WithTls;
use super::*;

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

/// The scripted guest (module docs), ending with `ExitProcess`.
#[expect(
    clippy::too_many_lines,
    reason = "one step per request of the conversation, in order"
)]
fn waiting_guest(script: &Script) -> ScriptedVcpu {
    let name = script.base + NAME_AT;
    script.tls.write(name, b"sm:\0").unwrap();
    let send = || svc(SvcId::SendSyncRequest);
    let take_sm = script.clone();
    let wait = script.clone();
    let done = script.clone();
    let [first_low, first_high] = halves(REGION);
    let [last_low, last_high] = halves(REGION + REGION_SIZE - 1);
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
