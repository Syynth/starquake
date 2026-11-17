//! `appletOE` driven through raw messages laid out as libnx lays them out
//! (`libnx/nx/source/services/applet.c`), over a fake kernel. libnx converts the `appletOE`
//! session to a domain before it opens the proxy (the startup fixture's trace, line 45), so
//! most tests talk to domain objects, as the fixture does.

use starquake_ipc::{
    BUFFER_SIZE, Body, CMIF_HEADER_BYTES, CommandHeader, CommandType, DomainCommand,
    DomainRequestHeader, Message, ReceiveList, RequestBuilder, RequestLayout, SFCI_MAGIC,
    SFCO_MAGIC, SessionKind, SpecialHeader,
};
use starquake_kernel::handle::Handle;
use starquake_kernel::id::ObjectId;
use starquake_kernel::ipc::message::Current;
use starquake_kernel::result::{KernelError, ResultCode};

use super::*;
use crate::framework::control::CONVERT_CURRENT_OBJECT_TO_DOMAIN;
use crate::framework::fake::FakeKernel;
use crate::framework::result::{INVALID_CMIF_REQUEST, UNKNOWN_COMMAND_ID};
use crate::framework::{Handled, Kernel, Protocol, Session, SnapshotError};

use super::command::application_functions::{
    GET_SCREEN_SHOT_AND_CONTINUOUS_RECORDING_INITIALLY_ENABLED, NOTIFY_RUNNING, UNKNOWN_321,
};
use super::command::common_state_getter::{
    GET_CURRENT_FOCUS_STATE, GET_EVENT_HANDLE, GET_OPERATION_MODE, GET_PERFORMANCE_MODE,
};
use super::command::graphics_scaling_parameters_functions::{
    GET_GRAPHICS_SCALING_PARAMETERS, SET_GRAPHICS_SCALE_CHANGED_NOTIFICATION_ENABLED,
};
use super::command::proxy::{
    GET_APPLICATION_FUNCTIONS, GET_AUDIO_CONTROLLER, GET_COMMON_STATE_GETTER, GET_DEBUG_FUNCTIONS,
    GET_DISPLAY_CONTROLLER, GET_GRAPHICS_SCALING_PARAMETERS_FUNCTIONS, GET_LIBRARY_APPLET_CREATOR,
    GET_PERFORMANCE_FUNCTIONS, GET_SELF_CONTROLLER, GET_WINDOW_CONTROLLER,
};
use super::command::proxy_service::{OPEN_APPLICATION_PROXY, OPEN_APPLICATION_PROXY_WITH_FLAGS};
use super::command::self_controller::{
    EXIT, GET_ACCUMULATED_SUSPENDED_TICK_CHANGED_EVENT, GET_GRAPHICS_SUPERVISOR,
    SET_FOCUS_HANDLING_MODE, SET_OPERATION_MODE_CHANGED_NOTIFICATION,
    SET_OUT_OF_FOCUS_SUSPENDING_ENABLED, SET_PERFORMANCE_MODE_CHANGED_NOTIFICATION,
};
use super::command::window_controller::{ACQUIRE_FOREGROUND_RIGHTS, GET_APPLET_RESOURCE_USER_ID};

/// `ISelfController::LockExit`, which this port leaves out, so it has no shipping constant
/// (`eden/src/core/hle/service/am/service/self_controller.cpp`: `{1,
/// D<&ISelfController::LockExit>, "LockExit"}`).
const LOCK_EXIT: u32 = 1;
/// `ICommonStateGetter::ReceiveMessage`, which this port leaves out, so it has no shipping
/// constant (`common_state_getter.cpp`: `{1,
/// D<&ICommonStateGetter::ReceiveMessage>, "ReceiveMessage"}`).
const RECEIVE_MESSAGE: u32 = 1;

/// The caller's process id: distinct from every other number the tests use.
const PROCESS_ID: u64 = 0x0123_4567_89AB_CDEF;
/// What fills the message buffer beforehand.
const STALE: u8 = 0xA5;

/// An `appletOE` server, a kernel, one session, and a handle to stand for the caller's process.
struct Rig {
    kernel: FakeKernel,
    server: Server,
    session: ObjectId,
    process: Handle,
}

/// A parsed reply.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Answered {
    result: ResultCode,
    data: Vec<u8>,
    copies: Vec<u32>,
    moves: Vec<u32>,
    objects: Vec<u32>,
}

fn parse(buffer: &[u8], kind: SessionKind) -> Answered {
    let message = Message::parse_reply(buffer, kind).unwrap();
    let Body::CmifReply(reply) = &message.body else {
        unreachable!("not a CMIF reply: {:?}", message.body);
    };
    Answered {
        result: ResultCode::from_raw(reply.header.result),
        data: reply.data.to_vec(),
        copies: message.copy_handles.clone(),
        moves: message.move_handles.clone(),
        objects: reply.object_ids.clone(),
    }
}

/// A whole message buffer that starts with `words` and is zero after them.
fn buffer_of(words: &[u32]) -> Vec<u8> {
    let mut buffer: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    buffer.resize(BUFFER_SIZE, 0);
    buffer
}

/// The words of a domain reply with no handles: header, padding, the domain header counting
/// `objects.len()` objects, `SFCO`, version, result, token, `data` (as words, zero-padded), then
/// the object ids.
fn domain_reply(result: ResultCode, data: &[u8], objects: &[u32]) -> Vec<u8> {
    let mut words = vec![0, 0, 0, 0];
    words.extend([u32::try_from(objects.len()).unwrap(), 0, 0, 0]);
    words.extend([SFCO_MAGIC, 0, result.raw(), 0]);
    for chunk in data.chunks(4) {
        let mut word = [0; 4];
        word[..chunk.len()].copy_from_slice(chunk);
        words.push(u32::from_le_bytes(word));
    }
    words.extend(objects);
    // Raw data runs from after the two header words to the end.
    words[1] = u32::try_from(words.len() - 2).unwrap();
    buffer_of(&words)
}

/// The words of a domain reply that copies one handle and holds nothing else: the header with
/// the special header announced, the special header counting one copied handle, the handle, then
/// (already on a 16-byte boundary) the domain header with no objects and `SFCO`.
fn copied_handle_reply(handle: u32) -> Vec<u8> {
    let header = CommandHeader {
        command_type: CommandType::Invalid,
        num_x: 0,
        num_a: 0,
        num_b: 0,
        num_w: 0,
        raw_data_words: 8,
        receive_list: ReceiveList::Disabled,
        has_special_header: true,
    };
    let special = SpecialHeader {
        send_pid: false,
        num_copy_handles: 1,
        num_move_handles: 0,
    };
    let [w0, w1] = header.encode().unwrap();
    let success = ResultCode::SUCCESS.raw();
    let special = special.encode().unwrap();
    buffer_of(&[
        w0, w1, special, handle, 0, 0, 0, 0, SFCO_MAGIC, 0, success, 0,
    ])
}

/// The interfaces the startup fixture opens, in its order, each with the proxy's command.
const OPENED: [(u32, &str); 8] = [
    (GET_APPLICATION_FUNCTIONS, "IApplicationFunctions"),
    (GET_LIBRARY_APPLET_CREATOR, "ILibraryAppletCreator"),
    (GET_COMMON_STATE_GETTER, "ICommonStateGetter"),
    (GET_SELF_CONTROLLER, "ISelfController"),
    (GET_WINDOW_CONTROLLER, "IWindowController"),
    (GET_AUDIO_CONTROLLER, "IAudioController"),
    (GET_DISPLAY_CONTROLLER, "IDisplayController"),
    (GET_DEBUG_FUNCTIONS, "IDebugFunctions"),
];

impl Rig {
    fn with(settings: AppletSettings) -> Self {
        let mut kernel = FakeKernel::new(0).unwrap();
        kernel.set_process_id(PROCESS_ID);
        let session = kernel.new_object();
        let process = kernel.new_object();
        let process = kernel.insert_handle(process).unwrap();
        let mut server = oe_server(settings, UnknownPolicy::Strict);
        server.accept(session).unwrap();
        Self {
            kernel,
            server,
            session,
            process,
        }
    }

    fn new() -> Self {
        Self::with(AppletSettings::eden_default())
    }

    fn send_raw(&mut self, session: ObjectId, request: &RequestBuilder) -> Vec<u8> {
        let mut buffer = vec![STALE; BUFFER_SIZE];
        request.build(&mut buffer).unwrap();
        let handled = self.server.handle(&mut self.kernel, session, &mut buffer);
        assert_eq!(handled, Ok(Handled::Replied));
        buffer
    }

    /// Sends `command` with `data` to object `object` of the rig's domain; the raw reply.
    fn object_raw(&mut self, object: u32, command: u32, data: &[u8]) -> Vec<u8> {
        let request = RequestBuilder::domain(CommandType::Request, object, command).data(data);
        self.send_raw(self.session, &request)
    }

    fn object(&mut self, object: u32, command: u32, data: &[u8]) -> Answered {
        parse(&self.object_raw(object, command, data), SessionKind::Domain)
    }

    /// Converts the rig's session to a domain, as libnx does; the root's object id.
    fn convert(&mut self) -> u32 {
        let convert = RequestBuilder::cmif(CommandType::Control, CONVERT_CURRENT_OBJECT_TO_DOMAIN);
        let reply = parse(&self.send_raw(self.session, &convert), SessionKind::Session);
        assert_eq!(reply.result, ResultCode::SUCCESS);
        u32::from_le_bytes(reply.data.try_into().unwrap())
    }

    /// `OpenApplicationProxy` as libnx sends it (`_appletGetSessionProxy`) to the domain's root
    /// `root`, with process id `pid`; the raw reply.
    fn open_proxy_raw(&mut self, root: u32, pid: u64) -> Vec<u8> {
        let request = RequestBuilder::domain(CommandType::Request, root, OPEN_APPLICATION_PROXY)
            .send_pid(pid)
            .copy_handle(self.process.raw())
            .data(&0_u64.to_le_bytes());
        self.send_raw(self.session, &request)
    }

    /// Converts, opens the proxy, and returns its object id.
    fn proxy(&mut self) -> u32 {
        let root = self.convert();
        let reply = parse(&self.open_proxy_raw(root, PROCESS_ID), SessionKind::Domain);
        assert_eq!(reply.result, ResultCode::SUCCESS);
        reply.objects[0]
    }

    /// Opens the interface `command` returns from the proxy `proxy`; its object id.
    fn open(&mut self, proxy: u32, command: u32) -> u32 {
        let reply = self.object(proxy, command, &[]);
        assert_eq!(reply.result, ResultCode::SUCCESS);
        assert_eq!(reply.objects.len(), 1);
        reply.objects[0]
    }

    fn applet(&self) -> Applet {
        *self.server.family::<Applet>().unwrap()
    }

    /// The object behind a handle a reply copied.
    fn object_of(&self, handle: u32) -> ObjectId {
        self.kernel
            .resolve_handle(Handle::from_raw(handle))
            .unwrap()
    }

    /// Whether the server's snapshot names `interface`: how a test tells which interface an
    /// object id is when it has no command to tell by. A snapshot writes a name as `postcard`
    /// writes a string, its length (one byte, for a name this short) then its bytes, so a name
    /// that starts another (`IApplicationProxy`, `IApplicationProxyService`) is told apart.
    fn holds(&mut self, interface: &str) -> bool {
        let saved = self.server.save().unwrap();
        let mut name = vec![u8::try_from(interface.len()).unwrap()];
        assert!(name[0] < 0x80, "a one-byte length");
        name.extend(interface.as_bytes());
        saved.windows(name.len()).any(|window| window == name)
    }
}

// ----- appletOE: OpenApplicationProxy ---------------------------------------------------------

#[test]
fn open_application_proxy_returns_an_object_and_the_reply_is_these_bytes() {
    let mut rig = Rig::new();
    let root = rig.convert();
    let reply = rig.open_proxy_raw(root, PROCESS_ID);
    let proxy = root + 1;
    assert_eq!(reply, domain_reply(ResultCode::SUCCESS, &[], &[proxy]));
    assert!(rig.holds("IApplicationProxy"));
    assert_eq!(rig.server.interface_count(), 2);
    // No kernel session and no handle beyond the test's own: the proxy lives in the domain.
    assert_eq!(rig.kernel.sessions(), []);
    assert_eq!(rig.kernel.handle_count(), 1);
}

#[test]
fn open_application_proxy_on_a_plain_session_moves_a_session_handle() {
    let mut rig = Rig::new();
    let request = RequestBuilder::cmif(CommandType::Request, OPEN_APPLICATION_PROXY)
        .layout(RequestLayout::Libnx)
        .send_pid(PROCESS_ID)
        .copy_handle(rig.process.raw())
        .data(&0_u64.to_le_bytes());
    let reply = parse(&rig.send_raw(rig.session, &request), SessionKind::Session);
    assert_eq!(reply.result, ResultCode::SUCCESS);
    assert_eq!(reply.moves.len(), 1);
    let proxy = rig.object_of(reply.moves[0]);
    assert_eq!(rig.kernel.sessions(), [proxy]);
    // The new session answers as a proxy.
    let window = parse(
        &rig.send_raw(
            proxy,
            &RequestBuilder::cmif(CommandType::Request, GET_WINDOW_CONTROLLER),
        ),
        SessionKind::Session,
    );
    assert_eq!(window.result, ResultCode::SUCCESS);
    assert_eq!(window.moves.len(), 1);
}

/// Divergence from Eden, which takes the process id from the kernel whether or not the request
/// asked for it, and reads copy handle 0 without checking there is one. The id is checked
/// first: a request without either is answered as one without the id.
#[test]
fn open_application_proxy_needs_the_process_id_and_the_process_handle() {
    let mut rig = Rig::new();
    let root = rig.convert();
    let ask = |pid: bool, handle: Option<u32>| {
        let mut request =
            RequestBuilder::domain(CommandType::Request, root, OPEN_APPLICATION_PROXY)
                .data(&0_u64.to_le_bytes());
        if pid {
            request = request.send_pid(PROCESS_ID);
        }
        if let Some(handle) = handle {
            request = request.copy_handle(handle);
        }
        request
    };
    let process = rig.process.raw();
    let cases = [
        (false, Some(process), UNKNOWN),
        (true, None, INVALID_CMIF_REQUEST),
        (false, None, UNKNOWN),
        // A handle that names nothing is the kernel's error: here the current-process
        // pseudo-handle, before the rig says which process it names.
        (
            true,
            Some(Handle::CURRENT_PROCESS.raw()),
            KernelError::InvalidHandle.code(),
        ),
    ];
    for (pid, handle, result) in cases {
        let reply = rig.send_raw(rig.session, &ask(pid, handle));
        assert_eq!(
            reply,
            domain_reply(result, &[], &[]),
            "pid {pid}, handle {handle:?}"
        );
    }
    assert_eq!(rig.server.interface_count(), 1);
}

/// libnx's exact request: `_appletGetSessionProxy` copies `CUR_PROCESS_HANDLE`, which the kernel
/// translates to the sending process (`starquake_kernel::ipc::message`), as the fake does once it
/// knows the process (`framework::fake`, "Handle translation").
#[test]
fn libnx_s_request_with_the_current_process_pseudo_handle_opens_the_proxy() {
    let mut rig = Rig::new();
    let root = rig.convert();
    let process = rig.kernel.new_object();
    rig.kernel.set_current(Current {
        thread: None,
        process: Some(process),
    });
    let request = RequestBuilder::domain(CommandType::Request, root, OPEN_APPLICATION_PROXY)
        .send_pid(PROCESS_ID)
        .copy_handle(Handle::CURRENT_PROCESS.raw())
        .data(&0_u64.to_le_bytes());
    let reply = rig.send_raw(rig.session, &request);
    assert_eq!(reply, domain_reply(ResultCode::SUCCESS, &[], &[root + 1]));
    assert!(rig.holds("IApplicationProxy"));
    // The pseudo-handle took no handle in the caller's table: only the rig's own is there.
    assert_eq!(rig.kernel.handle_count(), 1);
}

/// Divergence from Eden, which answers `ResultUnknown` for a process it did not launch: here
/// every process id opens the proxy, and is the applet resource user id.
#[test]
fn any_process_id_opens_the_proxy_and_is_its_applet_resource_user_id() {
    const OTHER: u64 = 0xFEDC_BA98_7654_3210;
    let mut rig = Rig::new();
    let root = rig.convert();
    for pid in [PROCESS_ID, OTHER] {
        // The id is the kernel's, whatever the message's id slot holds (`Kernel::process_id`).
        rig.kernel.set_process_id(pid);
        let reply = parse(&rig.open_proxy_raw(root, 0), SessionKind::Domain);
        assert_eq!(reply.result, ResultCode::SUCCESS);
        let window = rig.open(reply.objects[0], GET_WINDOW_CONTROLLER);
        let raw = rig.object_raw(window, GET_APPLET_RESOURCE_USER_ID, &[]);
        assert_eq!(
            raw,
            domain_reply(ResultCode::SUCCESS, &pid.to_le_bytes(), &[])
        );
    }
}

// ----- IApplicationProxy ---------------------------------------------------------------------

#[test]
fn each_proxy_command_returns_its_interface() {
    for (command, interface) in OPENED {
        let mut rig = Rig::new();
        let proxy = rig.proxy();
        assert!(!rig.holds(interface), "{interface} before it is opened");
        let reply = rig.object_raw(proxy, command, &[]);
        assert_eq!(
            reply,
            domain_reply(ResultCode::SUCCESS, &[], &[proxy + 1]),
            "{interface}"
        );
        assert!(rig.holds(interface), "{interface}");
        assert_eq!(rig.server.interface_count(), 3, "{interface}");
    }
}

// ----- ICommonStateGetter --------------------------------------------------------------------

#[test]
fn get_event_handle_copies_the_message_event_signalled_at_launch() {
    let mut rig = Rig::new();
    let proxy = rig.proxy();
    let getter = rig.open(proxy, GET_COMMON_STATE_GETTER);
    assert_eq!(rig.applet().message_event(), None);

    let raw = rig.object_raw(getter, GET_EVENT_HANDLE, &[]);
    let first = parse(&raw, SessionKind::Domain).copies[0];
    assert_eq!(raw, copied_handle_reply(first));
    let event = rig.object_of(first);
    assert_eq!(rig.applet().message_event(), Some(event));
    // A focus change is pending from launch, with notifications on: signalled once.
    assert!(rig.applet().lifecycle().message_available());
    assert_eq!(rig.kernel.signals(event), Some(1));

    // Asked again: another handle to the same event, not signalled again.
    let second = parse(
        &rig.object_raw(getter, GET_EVENT_HANDLE, &[]),
        SessionKind::Domain,
    )
    .copies[0];
    assert_ne!(second, first);
    assert_eq!(rig.object_of(second), event);
    assert_eq!(rig.kernel.signals(event), Some(1));
}

#[test]
fn a_message_event_made_after_notifications_went_off_is_not_signalled() {
    let mut rig = Rig::new();
    let proxy = rig.proxy();
    let getter = rig.open(proxy, GET_COMMON_STATE_GETTER);
    let controller = rig.open(proxy, GET_SELF_CONTROLLER);
    // libnx's `SetFocusHandlingMode(SuspendHomeSleep)`: notify off.
    let off = rig.object(controller, SET_FOCUS_HANDLING_MODE, &[0, 0, 1]);
    assert_eq!(off.result, ResultCode::SUCCESS);
    let copied = rig.object(getter, GET_EVENT_HANDLE, &[]).copies[0];
    let event = rig.object_of(copied);
    assert_eq!(rig.kernel.signals(event), Some(0));
}

#[test]
fn the_reported_modes_and_focus_state_are_the_settings() {
    let cases = [
        // Eden's defaults: docked (so boost), in focus.
        (AppletSettings::eden_default(), [1_u8], 1_i32, [1_u8]),
        (
            AppletSettings {
                operation_mode: OperationMode::Handheld,
                focus_state: FocusState::NotInFocus,
            },
            [0],
            0,
            [2],
        ),
        (
            AppletSettings {
                operation_mode: OperationMode::Docked,
                focus_state: FocusState::Background,
            },
            [1],
            1,
            [3],
        ),
    ];
    for (settings, operation, performance, focus) in cases {
        let mut rig = Rig::with(settings);
        assert_eq!(rig.applet().settings(), &settings);
        let proxy = rig.proxy();
        let getter = rig.open(proxy, GET_COMMON_STATE_GETTER);
        let asked = [
            (GET_OPERATION_MODE, operation.to_vec()),
            (GET_PERFORMANCE_MODE, performance.to_le_bytes().to_vec()),
            (GET_CURRENT_FOCUS_STATE, focus.to_vec()),
        ];
        for (command, data) in asked {
            assert_eq!(
                rig.object_raw(getter, command, &[]),
                domain_reply(ResultCode::SUCCESS, &data, &[]),
                "{settings:?}, command {command}"
            );
        }
    }
}

#[test]
fn the_reply_values_are_edens_numbers() {
    // `am_types.h`: `OperationMode : u8 { Handheld = 0, Docked = 1 }`, `FocusState : u8 {
    // InFocus = 1, NotInFocus = 2, Background = 3 }`.
    assert_eq!(OperationMode::Handheld.raw(), 0);
    assert_eq!(OperationMode::Docked.raw(), 1);
    assert_eq!(FocusState::InFocus.raw(), 1);
    assert_eq!(FocusState::NotInFocus.raw(), 2);
    assert_eq!(FocusState::Background.raw(), 3);
    assert!(OperationMode::Docked.is_docked());
    assert!(!OperationMode::Handheld.is_docked());
    assert_eq!(
        AppletSettings::eden_default(),
        AppletSettings {
            operation_mode: OperationMode::Docked,
            focus_state: FocusState::InFocus,
        }
    );
    assert_eq!(
        AppletSettings::eden_default().performance_mode(),
        PerformanceMode::Boost
    );
}

// ----- ISelfController -----------------------------------------------------------------------

/// Divergence from Eden, which clears the event where a message stops being available.
#[test]
fn turning_focus_notifications_off_clears_the_message_event() {
    let mut rig = Rig::new();
    let proxy = rig.proxy();
    let getter = rig.open(proxy, GET_COMMON_STATE_GETTER);
    let controller = rig.open(proxy, GET_SELF_CONTROLLER);
    let copied = rig.object(getter, GET_EVENT_HANDLE, &[]).copies[0];
    let event = rig.object_of(copied);
    assert_eq!(rig.kernel.signals(event), Some(1));
    assert_eq!(rig.kernel.is_signalled(event), Some(true));

    // notify stays on: nothing changes.
    let on = rig.object_raw(controller, SET_FOCUS_HANDLING_MODE, &[1, 0, 1]);
    assert_eq!(on, domain_reply(ResultCode::SUCCESS, &[], &[]));
    assert!(rig.applet().lifecycle().message_available());
    assert_eq!(rig.kernel.signals(event), Some(1));
    assert_eq!(rig.kernel.is_signalled(event), Some(true));

    // notify off, libnx's start-up: no message is available any more, and the event is cleared.
    let off = rig.object_raw(controller, SET_FOCUS_HANDLING_MODE, &[0, 0, 1]);
    assert_eq!(off, domain_reply(ResultCode::SUCCESS, &[], &[]));
    assert!(!rig.applet().lifecycle().message_available());
    assert_eq!(rig.kernel.signals(event), Some(1));
    assert_eq!(rig.kernel.is_signalled(event), Some(false));

    // notify off again: still nothing, and the event stays clear.
    rig.object(controller, SET_FOCUS_HANDLING_MODE, &[0, 0, 1]);
    assert_eq!(rig.kernel.signals(event), Some(1));
    assert_eq!(rig.kernel.is_signalled(event), Some(false));

    // notify on again: the pending focus change is a message again, and the event is signalled.
    rig.object(controller, SET_FOCUS_HANDLING_MODE, &[1, 0, 1]);
    assert!(rig.applet().lifecycle().message_available());
    assert_eq!(rig.kernel.signals(event), Some(2));
    assert_eq!(rig.kernel.is_signalled(event), Some(true));
}

#[test]
fn notifications_turned_off_before_the_event_exists_clear_nothing() {
    // No event yet: the change is only the applet's, and the event is made clear.
    let mut rig = Rig::new();
    let proxy = rig.proxy();
    let controller = rig.open(proxy, GET_SELF_CONTROLLER);
    let off = rig.object(controller, SET_FOCUS_HANDLING_MODE, &[0, 0, 1]);
    assert_eq!(off.result, ResultCode::SUCCESS);
    let getter = rig.open(proxy, GET_COMMON_STATE_GETTER);
    let copied = rig.object(getter, GET_EVENT_HANDLE, &[]).copies[0];
    let event = rig.object_of(copied);
    assert_eq!(rig.kernel.is_signalled(event), Some(false));
    assert_eq!(rig.kernel.signals(event), Some(0));
}

// ----- ISelfController: Exit -----------------------------------------------------------------

#[test]
fn exit_answers_success_and_asks_once_to_end_the_process() {
    let mut rig = Rig::new();
    let proxy = rig.proxy();
    let controller = rig.open(proxy, GET_SELF_CONTROLLER);
    assert_eq!(rig.kernel.ends(), 0);
    // libnx's `_appletSelfExit`: no payload.
    let reply = rig.object_raw(controller, EXIT, &[]);
    assert_eq!(reply, domain_reply(ResultCode::SUCCESS, &[], &[]));
    assert_eq!(rig.kernel.ends(), 1);
}

#[test]
fn exit_on_a_kernel_that_cannot_end_the_process_answers_its_error() {
    let mut rig = Rig::new();
    let proxy = rig.proxy();
    let controller = rig.open(proxy, GET_SELF_CONTROLLER);
    let placeholder = FakeKernel::new(0).unwrap();
    let mut kernel = RefusesSignals(std::mem::replace(&mut rig.kernel, placeholder));
    let request = RequestBuilder::domain(CommandType::Request, controller, EXIT);
    let mut buffer = vec![STALE; BUFFER_SIZE];
    request.build(&mut buffer).unwrap();
    let handled = rig.server.handle(&mut kernel, rig.session, &mut buffer);
    assert_eq!(handled, Ok(Handled::Replied));
    assert_eq!(
        buffer,
        domain_reply(KernelError::NotImplemented.code(), &[], &[])
    );
    assert_eq!(kernel.0.ends(), 0);
}

#[test]
fn set_focus_handling_mode_reads_its_three_flags_at_their_offsets() {
    // `background` (offset 1) is not read; `notify` (0) and `suspend` (2) are. Each case has
    // the other two flags at the values that would show a swap.
    let cases: [([u8; 3], bool, FocusHandlingMode); 4] = [
        ([0, 1, 1], false, FocusHandlingMode::SuspendHomeSleep),
        ([1, 0, 0], true, FocusHandlingMode::NoSuspend),
        ([0, 1, 0], false, FocusHandlingMode::NoSuspend),
        ([1, 1, 1], true, FocusHandlingMode::SuspendHomeSleep),
    ];
    for (flags, notify, mode) in cases {
        let mut rig = Rig::new();
        let proxy = rig.proxy();
        let controller = rig.open(proxy, GET_SELF_CONTROLLER);
        let reply = rig.object(controller, SET_FOCUS_HANDLING_MODE, &flags);
        assert_eq!(reply.result, ResultCode::SUCCESS, "{flags:?}");
        let lifecycle = *rig.applet().lifecycle();
        assert_eq!(
            (
                lifecycle.notifications.focus_state_changed,
                lifecycle.focus_handling_mode
            ),
            (notify, mode),
            "{flags:?}"
        );
    }
}

#[test]
fn a_flag_is_any_byte_that_is_not_zero() {
    let mut rig = Rig::new();
    let proxy = rig.proxy();
    let controller = rig.open(proxy, GET_SELF_CONTROLLER);
    let enabled = |rig: &Rig| {
        rig.applet()
            .lifecycle()
            .notifications
            .operation_mode_changed
    };
    rig.object(controller, SET_OPERATION_MODE_CHANGED_NOTIFICATION, &[0]);
    assert!(!enabled(&rig));
    rig.object(controller, SET_OPERATION_MODE_CHANGED_NOTIFICATION, &[2]);
    assert!(enabled(&rig));
    rig.object(controller, SET_OPERATION_MODE_CHANGED_NOTIFICATION, &[0]);
    rig.object(controller, SET_OPERATION_MODE_CHANGED_NOTIFICATION, &[0xFF]);
    assert!(enabled(&rig));
}

#[test]
fn a_payload_too_short_for_the_flags_is_refused_and_changes_nothing() {
    let mut rig = Rig::new();
    let proxy = rig.proxy();
    let controller = rig.open(proxy, GET_SELF_CONTROLLER);
    let before = rig.applet();
    let refused = domain_reply(INVALID_CMIF_REQUEST, &[], &[]);
    // A message carries its payload in whole words, so the shortest payload short of the flags
    // is none at all.
    for command in [
        SET_FOCUS_HANDLING_MODE,
        SET_OPERATION_MODE_CHANGED_NOTIFICATION,
        SET_PERFORMANCE_MODE_CHANGED_NOTIFICATION,
        SET_OUT_OF_FOCUS_SUSPENDING_ENABLED,
    ] {
        assert_eq!(
            rig.object_raw(controller, command, &[]),
            refused,
            "{command}"
        );
    }
    assert_eq!(rig.applet(), before);
    // Exactly the three bytes libnx sends are enough.
    let accepted = rig.object(controller, SET_FOCUS_HANDLING_MODE, &[0, 0, 0]);
    assert_eq!(accepted.result, ResultCode::SUCCESS);
}

#[test]
fn the_mode_change_notifications_are_set_and_cleared_without_a_signal() {
    let mut rig = Rig::new();
    let proxy = rig.proxy();
    let getter = rig.open(proxy, GET_COMMON_STATE_GETTER);
    let controller = rig.open(proxy, GET_SELF_CONTROLLER);
    let copied = rig.object(getter, GET_EVENT_HANDLE, &[]).copies[0];
    let event = rig.object_of(copied);
    let flags = |applet: &Applet| {
        let lifecycle = applet.lifecycle();
        (
            lifecycle.notifications.operation_mode_changed,
            lifecycle.notifications.performance_mode_changed,
        )
    };
    assert_eq!(flags(&rig.applet()), (true, true));
    let steps = [
        (SET_OPERATION_MODE_CHANGED_NOTIFICATION, 0, (false, true)),
        (SET_PERFORMANCE_MODE_CHANGED_NOTIFICATION, 0, (false, false)),
        (SET_OPERATION_MODE_CHANGED_NOTIFICATION, 1, (true, false)),
        (SET_OPERATION_MODE_CHANGED_NOTIFICATION, 1, (true, false)),
        (SET_PERFORMANCE_MODE_CHANGED_NOTIFICATION, 1, (true, true)),
    ];
    for (command, flag, expected) in steps {
        let reply = rig.object_raw(controller, command, &[flag]);
        assert_eq!(reply, domain_reply(ResultCode::SUCCESS, &[], &[]));
        assert_eq!(flags(&rig.applet()), expected, "{command} {flag}");
    }
    assert_eq!(rig.kernel.signals(event), Some(1));
}

#[test]
fn set_out_of_focus_suspending_enabled_moves_the_focus_handling_mode() {
    let mut rig = Rig::new();
    let proxy = rig.proxy();
    let controller = rig.open(proxy, GET_SELF_CONTROLLER);
    let mode = |rig: &Rig| rig.applet().lifecycle().focus_handling_mode;
    assert_eq!(mode(&rig), FocusHandlingMode::SuspendHomeSleep);
    let steps = [
        (0, FocusHandlingMode::SuspendHomeSleep),
        (1, FocusHandlingMode::AlwaysSuspend),
        (1, FocusHandlingMode::AlwaysSuspend),
        (0, FocusHandlingMode::SuspendHomeSleep),
    ];
    for (flag, expected) in steps {
        let reply = rig.object_raw(controller, SET_OUT_OF_FOCUS_SUSPENDING_ENABLED, &[flag]);
        assert_eq!(reply, domain_reply(ResultCode::SUCCESS, &[], &[]));
        assert_eq!(mode(&rig), expected, "{flag}");
    }
}

#[test]
fn the_focus_handling_mode_moves_as_edens_does_for_every_mode_and_flag() {
    use FocusHandlingMode::{AlwaysSuspend, NoSuspend, SuspendHomeSleep};
    // `LifecycleManager::SetFocusHandlingMode(suspend)`.
    let set_focus = [
        (AlwaysSuspend, false, NoSuspend),
        (AlwaysSuspend, true, AlwaysSuspend),
        (SuspendHomeSleep, false, NoSuspend),
        (SuspendHomeSleep, true, SuspendHomeSleep),
        (NoSuspend, false, NoSuspend),
        (NoSuspend, true, SuspendHomeSleep),
    ];
    for (from, suspend, to) in set_focus {
        let mut lifecycle = Lifecycle {
            focus_handling_mode: from,
            ..Lifecycle::default()
        };
        lifecycle.set_focus_handling_mode(suspend);
        assert_eq!(lifecycle.focus_handling_mode, to, "{from:?} {suspend}");
    }
    // `LifecycleManager::SetOutOfFocusSuspendingEnabled(enabled)`.
    let set_out_of_focus = [
        (AlwaysSuspend, false, SuspendHomeSleep),
        (AlwaysSuspend, true, AlwaysSuspend),
        (SuspendHomeSleep, false, SuspendHomeSleep),
        (SuspendHomeSleep, true, AlwaysSuspend),
        (NoSuspend, false, NoSuspend),
        (NoSuspend, true, AlwaysSuspend),
    ];
    for (from, enabled, to) in set_out_of_focus {
        let mut lifecycle = Lifecycle {
            focus_handling_mode: from,
            ..Lifecycle::default()
        };
        lifecycle.set_out_of_focus_suspending_enabled(enabled);
        assert_eq!(lifecycle.focus_handling_mode, to, "{from:?} {enabled}");
    }
}

#[test]
fn a_message_is_available_only_with_notifications_on_and_a_focus_change_pending() {
    for (notify, changed) in [(false, false), (false, true), (true, false), (true, true)] {
        let lifecycle = Lifecycle {
            notifications: Notifications {
                focus_state_changed: notify,
                ..Notifications::default()
            },
            has_focus_state_changed: changed,
            ..Lifecycle::default()
        };
        assert_eq!(lifecycle.message_available(), notify && changed);
    }
    // Eden's initial values.
    assert_eq!(
        Lifecycle::default(),
        Lifecycle {
            notifications: Notifications {
                focus_state_changed: true,
                operation_mode_changed: true,
                performance_mode_changed: true,
            },
            has_focus_state_changed: true,
            focus_handling_mode: FocusHandlingMode::SuspendHomeSleep,
        }
    );
}

#[test]
fn the_suspended_tick_event_is_one_event_never_signalled() {
    let mut rig = Rig::new();
    let proxy = rig.proxy();
    let getter = rig.open(proxy, GET_COMMON_STATE_GETTER);
    let controller = rig.open(proxy, GET_SELF_CONTROLLER);
    let raw = rig.object_raw(
        controller,
        GET_ACCUMULATED_SUSPENDED_TICK_CHANGED_EVENT,
        &[],
    );
    let first = parse(&raw, SessionKind::Domain).copies[0];
    assert_eq!(raw, copied_handle_reply(first));
    let event = rig.object_of(first);
    assert_eq!(rig.applet().suspended_tick_event(), Some(event));
    assert_eq!(rig.kernel.signals(event), Some(0));

    let again = rig.object(
        controller,
        GET_ACCUMULATED_SUSPENDED_TICK_CHANGED_EVENT,
        &[],
    );
    assert_eq!(rig.object_of(again.copies[0]), event);
    assert_eq!(rig.kernel.signals(event), Some(0));
    // It is not the message event.
    let copied = rig.object(getter, GET_EVENT_HANDLE, &[]).copies[0];
    let message = rig.object_of(copied);
    assert_ne!(message, event);
    assert_eq!(rig.kernel.signals(event), Some(0));
}

// ----- IWindowController and IApplicationFunctions -------------------------------------------

#[test]
fn acquire_foreground_rights_succeeds_and_changes_nothing() {
    let mut rig = Rig::new();
    let proxy = rig.proxy();
    let window = rig.open(proxy, GET_WINDOW_CONTROLLER);
    let before = rig.applet();
    assert_eq!(
        rig.object_raw(window, ACQUIRE_FOREGROUND_RIGHTS, &[]),
        domain_reply(ResultCode::SUCCESS, &[], &[])
    );
    assert_eq!(rig.applet(), before);
}

#[test]
fn notify_running_answers_true() {
    let mut rig = Rig::new();
    let proxy = rig.proxy();
    let functions = rig.open(proxy, GET_APPLICATION_FUNCTIONS);
    assert_eq!(
        rig.object_raw(functions, NOTIFY_RUNNING, &[]),
        domain_reply(ResultCode::SUCCESS, &[1], &[])
    );
}

// ----- What is left out ----------------------------------------------------------------------

#[test]
fn commands_that_are_left_out_go_to_the_unknown_command_policy() {
    let mut rig = Rig::new();
    let proxy = rig.proxy();
    let controller = rig.open(proxy, GET_SELF_CONTROLLER);
    let getter = rig.open(proxy, GET_COMMON_STATE_GETTER);
    let refused = domain_reply(UNKNOWN_COMMAND_ID, &[], &[]);
    assert_eq!(rig.object_raw(getter, RECEIVE_MESSAGE, &[]), refused);
    assert_eq!(rig.object_raw(controller, LOCK_EXIT, &[]), refused);
    for command in [
        GET_AUDIO_CONTROLLER,
        GET_DISPLAY_CONTROLLER,
        GET_LIBRARY_APPLET_CREATOR,
        GET_DEBUG_FUNCTIONS,
    ] {
        let opened = rig.open(proxy, command);
        assert_eq!(rig.object_raw(opened, 0, &[]), refused, "{command}");
    }
}

#[test]
fn the_tables_name_the_commands_as_eden_does() {
    let names = |session: &dyn Session| -> Vec<(u32, &'static str)> {
        session
            .commands()
            .cmif
            .iter()
            .map(|row| (row.id, row.name))
            .collect()
    };
    assert_eq!(
        names(&IApplicationProxyService),
        [
            (OPEN_APPLICATION_PROXY, "open_application_proxy"),
            (
                OPEN_APPLICATION_PROXY_WITH_FLAGS,
                "open_application_proxy_with_flags"
            ),
        ]
    );
    assert_eq!(
        names(&IApplicationProxy::default()),
        [
            (GET_COMMON_STATE_GETTER, "get_common_state_getter"),
            (GET_SELF_CONTROLLER, "get_self_controller"),
            (GET_WINDOW_CONTROLLER, "get_window_controller"),
            (GET_AUDIO_CONTROLLER, "get_audio_controller"),
            (GET_DISPLAY_CONTROLLER, "get_display_controller"),
            (GET_LIBRARY_APPLET_CREATOR, "get_library_applet_creator"),
            (GET_APPLICATION_FUNCTIONS, "get_application_functions"),
            (GET_PERFORMANCE_FUNCTIONS, "get_performance_functions"),
            (
                GET_GRAPHICS_SCALING_PARAMETERS_FUNCTIONS,
                "get_graphics_scaling_parameters_functions"
            ),
            (GET_DEBUG_FUNCTIONS, "get_debug_functions"),
        ]
    );
    assert_eq!(
        names(&IApplicationFunctions),
        [
            (NOTIFY_RUNNING, "notify_running"),
            (
                GET_SCREEN_SHOT_AND_CONTINUOUS_RECORDING_INITIALLY_ENABLED,
                "get_screen_shot_and_continuous_recording_initially_enabled"
            ),
            (UNKNOWN_321, "unknown_321"),
        ]
    );
    assert_eq!(
        names(&IGraphicsScalingParametersFunctions),
        [
            (
                GET_GRAPHICS_SCALING_PARAMETERS,
                "get_graphics_scaling_parameters"
            ),
            (
                SET_GRAPHICS_SCALE_CHANGED_NOTIFICATION_ENABLED,
                "set_graphics_scale_changed_notification_enabled"
            ),
        ]
    );
    assert_eq!(
        ICommonStateGetter
            .commands()
            .name(Protocol::Cmif, GET_EVENT_HANDLE),
        Some("get_event_handle")
    );
    for empty in [
        names(&IAudioController),
        names(&IDisplayController),
        names(&ILibraryAppletCreator),
        names(&IDebugFunctions),
        names(&IPerformanceFunctions),
    ] {
        assert_eq!(empty, []);
    }
}

// ----- The start-up and snapshots ------------------------------------------------------------

/// The startup fixture's requests to `appletOE`, in its trace's order (lines 45 to 87 of
/// `tests/homebrew/startup/reference/startup.trace`), each answered with success.
#[test]
fn libnx_start_up_runs_through_in_the_traces_order() {
    let mut rig = Rig::new();
    let proxy = rig.proxy();
    let mut objects = std::collections::BTreeMap::new();
    for (command, interface) in OPENED.iter().take(5) {
        objects.insert(*interface, rig.open(proxy, *command));
    }
    let window = objects["IWindowController"];
    let aruid = rig.object(window, GET_APPLET_RESOURCE_USER_ID, &[]);
    assert_eq!(aruid.data, PROCESS_ID.to_le_bytes());
    for (command, interface) in OPENED.iter().skip(5) {
        objects.insert(*interface, rig.open(proxy, *command));
    }
    let controller = objects["ISelfController"];
    let getter = objects["ICommonStateGetter"];
    let functions = objects["IApplicationFunctions"];
    let steps: [(u32, u32, &[u8]); 11] = [
        (
            controller,
            GET_ACCUMULATED_SUSPENDED_TICK_CHANGED_EVENT,
            &[],
        ),
        (getter, GET_EVENT_HANDLE, &[]),
        (getter, GET_CURRENT_FOCUS_STATE, &[]),
        (window, ACQUIRE_FOREGROUND_RIGHTS, &[]),
        (controller, SET_FOCUS_HANDLING_MODE, &[0, 0, 1]),
        (controller, SET_OUT_OF_FOCUS_SUSPENDING_ENABLED, &[0]),
        (functions, NOTIFY_RUNNING, &[]),
        (getter, GET_OPERATION_MODE, &[]),
        (getter, GET_PERFORMANCE_MODE, &[]),
        (controller, SET_OPERATION_MODE_CHANGED_NOTIFICATION, &[1]),
        (controller, SET_PERFORMANCE_MODE_CHANGED_NOTIFICATION, &[1]),
    ];
    for (object, command, data) in steps {
        let reply = rig.object(object, command, data);
        assert_eq!(reply.result, ResultCode::SUCCESS, "command {command}");
    }
    // In focus, so libnx never waits on the message event.
    let focus = rig.object(getter, GET_CURRENT_FOCUS_STATE, &[]);
    assert_eq!(focus.data[0], FocusState::InFocus.raw());
    let lifecycle = *rig.applet().lifecycle();
    assert!(!lifecycle.message_available());
    assert_eq!(
        lifecycle.focus_handling_mode,
        FocusHandlingMode::SuspendHomeSleep
    );
    // So libnx's main loop finds the message event clear, and does not wake on every poll.
    let event = rig.applet().message_event().unwrap();
    assert_eq!(rig.kernel.is_signalled(event), Some(false));
    // The proxy, the root, and the eight interfaces it opened.
    assert_eq!(rig.server.interface_count(), 10);
}

#[test]
fn a_server_in_mid_start_up_saves_and_restores() {
    let mut rig = Rig::new();
    let proxy = rig.proxy();
    for (command, _) in OPENED {
        rig.open(proxy, command);
    }
    let getter = rig.open(proxy, GET_COMMON_STATE_GETTER);
    let controller = rig.open(proxy, GET_SELF_CONTROLLER);
    rig.object(getter, GET_EVENT_HANDLE, &[]);
    rig.object(
        controller,
        GET_ACCUMULATED_SUSPENDED_TICK_CHANGED_EVENT,
        &[],
    );
    rig.object(controller, SET_FOCUS_HANDLING_MODE, &[0, 0, 0]);
    let saved = rig.server.save().unwrap();

    let mut restored = oe_server(AppletSettings::eden_default(), UnknownPolicy::Strict);
    restored.restore(&saved).unwrap();
    assert_eq!(restored.family::<Applet>(), Some(&rig.applet()));
    assert_eq!(restored.interface_count(), rig.server.interface_count());
    assert_eq!(restored.save().unwrap(), saved);
}

#[test]
fn a_restored_server_answers_with_the_saved_applet_resource_user_id() {
    let mut rig = Rig::new();
    let proxy = rig.proxy();
    let window = rig.open(proxy, GET_WINDOW_CONTROLLER);
    let saved = rig.server.save().unwrap();

    // The restored server takes the rig's place, on the same kernel and session.
    let mut restored = oe_server(AppletSettings::eden_default(), UnknownPolicy::Strict);
    restored.restore(&saved).unwrap();
    rig.server = restored;

    // The window controller opened before the save.
    let reply = rig.object(window, GET_APPLET_RESOURCE_USER_ID, &[]);
    assert_eq!(reply.result, ResultCode::SUCCESS);
    assert_eq!(reply.data, PROCESS_ID.to_le_bytes());
    // One opened after the restore, from the proxy opened before it.
    let fresh = rig.open(proxy, GET_WINDOW_CONTROLLER);
    let reply = rig.object(fresh, GET_APPLET_RESOURCE_USER_ID, &[]);
    assert_eq!(reply.result, ResultCode::SUCCESS);
    assert_eq!(reply.data, PROCESS_ID.to_le_bytes());
}

/// `ISelfController` writes the applet back before it reports a failed signal (`Applet`'s
/// docs). A server restored onto a kernel where its saved message event is not an event fails
/// the signal, and the flag the command set is still kept.
#[test]
fn a_failed_signal_still_keeps_the_lifecycle_change() {
    let mut rig = Rig::new();
    let proxy = rig.proxy();
    let getter = rig.open(proxy, GET_COMMON_STATE_GETTER);
    let controller = rig.open(proxy, GET_SELF_CONTROLLER);
    rig.object(getter, GET_EVENT_HANDLE, &[]);
    let off = rig.object(controller, SET_FOCUS_HANDLING_MODE, &[0, 0, 1]);
    assert_eq!(off.result, ResultCode::SUCCESS);
    let event = rig.applet().message_event().unwrap();
    let saved = rig.server.save().unwrap();

    // A fresh kernel mints the same session id, but nothing has made the event there.
    let mut fresh = Rig::new();
    assert_eq!(fresh.session, rig.session);
    assert_eq!(fresh.kernel.signals(event), None);
    fresh.server.restore(&saved).unwrap();
    assert!(!fresh.applet().lifecycle().notifications.focus_state_changed);

    // Notify on, with a focus change pending: a message becomes available, and the signal fails.
    let on = fresh.object(controller, SET_FOCUS_HANDLING_MODE, &[1, 0, 1]);
    assert_eq!(on.result, KernelError::InvalidHandle.code());
    let lifecycle = *fresh.applet().lifecycle();
    assert!(lifecycle.notifications.focus_state_changed);
    assert!(lifecycle.message_available());
    assert_eq!(fresh.applet().message_event(), Some(event));
}

/// `ISelfController` writes the applet back before it reports a failed clear, as it does a
/// failed signal: a server restored onto a kernel where its saved message event is not an
/// event fails the clear with the kernel's exact error, and the flag the command set is kept.
#[test]
fn a_failed_clear_still_keeps_the_lifecycle_change() {
    let mut rig = Rig::new();
    let proxy = rig.proxy();
    let getter = rig.open(proxy, GET_COMMON_STATE_GETTER);
    let controller = rig.open(proxy, GET_SELF_CONTROLLER);
    rig.object(getter, GET_EVENT_HANDLE, &[]);
    assert!(rig.applet().lifecycle().message_available());
    let event = rig.applet().message_event().unwrap();
    let saved = rig.server.save().unwrap();

    // A fresh kernel did not make the event, so the server may not clear it.
    let mut fresh = Rig::new();
    assert_eq!(fresh.kernel.is_signalled(event), None);
    fresh.server.restore(&saved).unwrap();
    let off = fresh.object(controller, SET_FOCUS_HANDLING_MODE, &[0, 0, 1]);
    assert_eq!(off.result, KernelError::InvalidHandle.code());
    let lifecycle = *fresh.applet().lifecycle();
    assert!(!lifecycle.notifications.focus_state_changed);
    assert!(!lifecycle.message_available());
    assert_eq!(fresh.applet().message_event(), Some(event));
}

/// The error every signal of a [`RefusesSignals`] kernel reports: one no command of the port
/// reports otherwise.
const REFUSED: KernelError = KernelError::OutOfResource;

/// A [`FakeKernel`] whose `signal_event` always fails with [`REFUSED`], even for an event it just
/// made: how a test reaches the failure path of signalling a new event, which the fake alone
/// never takes. It does not offer `clear_event` or `end_process`, so they answer the trait's
/// default, [`KernelError::NotImplemented`].
struct RefusesSignals(FakeKernel);

impl Kernel for RefusesSignals {
    fn process_id(&self) -> u64 {
        self.0.process_id()
    }

    fn resolve_handle(&self, handle: Handle) -> starquake_kernel::result::SvcResult<ObjectId> {
        self.0.resolve_handle(handle)
    }

    fn insert_handle(&mut self, object: ObjectId) -> starquake_kernel::result::SvcResult<Handle> {
        self.0.insert_handle(object)
    }

    fn create_event(&mut self) -> starquake_kernel::result::SvcResult<ObjectId> {
        self.0.create_event()
    }

    fn signal_event(&mut self, _: ObjectId) -> starquake_kernel::result::SvcResult<()> {
        Err(REFUSED)
    }

    fn create_shared_memory(
        &mut self,
        size: u64,
        permission: starquake_kernel::memory::SvcPermission,
    ) -> starquake_kernel::result::SvcResult<(ObjectId, starquake_kernel::memory::SharedPages)>
    {
        self.0.create_shared_memory(size, permission)
    }

    fn create_session(&mut self) -> starquake_kernel::result::SvcResult<(ObjectId, Handle)> {
        self.0.create_session()
    }

    fn connect(
        &mut self,
        service: starquake_trace::ServiceId,
    ) -> starquake_kernel::result::SvcResult<(ObjectId, Handle)> {
        self.0.connect(service)
    }

    fn read_memory(&self, address: u64, out: &mut [u8]) -> starquake_kernel::result::SvcResult<()> {
        self.0.read_memory(address, out)
    }

    fn write_memory(
        &mut self,
        address: u64,
        data: &[u8],
    ) -> starquake_kernel::result::SvcResult<()> {
        self.0.write_memory(address, data)
    }

    fn record(&mut self, kind: starquake_trace::EventKind) {
        self.0.record(kind);
    }
}

/// `Applet::message_event_or_create` keeps a new event before it signals it, so a failed signal
/// does not lose the kernel object: the applet names it, and the next `GetEventHandle` returns
/// it rather than making another.
#[test]
fn a_new_message_event_whose_signal_fails_is_still_kept() {
    let mut rig = Rig::new();
    let proxy = rig.proxy();
    let getter = rig.open(proxy, GET_COMMON_STATE_GETTER);
    // A focus change is pending from launch, with notifications on: the new event is signalled.
    assert!(rig.applet().lifecycle().message_available());

    let placeholder = FakeKernel::new(0).unwrap();
    let mut kernel = RefusesSignals(std::mem::replace(&mut rig.kernel, placeholder));
    let request = RequestBuilder::domain(CommandType::Request, getter, GET_EVENT_HANDLE);
    let mut buffer = vec![STALE; BUFFER_SIZE];
    request.build(&mut buffer).unwrap();
    let handled = rig.server.handle(&mut kernel, rig.session, &mut buffer);
    assert_eq!(handled, Ok(Handled::Replied));
    rig.kernel = kernel.0;
    let refused = parse(&buffer, SessionKind::Domain);
    assert_eq!(refused.result, REFUSED.code());
    assert_eq!(refused.copies, Vec::<u32>::new());

    let event = rig.applet().message_event().unwrap();
    assert_eq!(rig.kernel.signals(event), Some(0));
    let copied = rig.object(getter, GET_EVENT_HANDLE, &[]).copies[0];
    assert_eq!(rig.object_of(copied), event);
    assert_eq!(rig.kernel.signals(event), Some(0));
}

// ----- ISelfController: GetGraphicsSupervisor (Switch 2) -------------------------------------

/// The number the tests' graphics server's root is known by: distinct from every other number
/// the fake kernel knows.
const GRAPHICS: ServiceId = ServiceId(0x6575);

impl Rig {
    /// A rig whose applet names [`GRAPHICS`] as the graphics server, which the kernel knows.
    fn with_graphics() -> Self {
        let mut rig = Self::new();
        rig.kernel.add_service(GRAPHICS);
        let applet = Applet::new(AppletSettings::eden_default()).with_graphics_server(GRAPHICS);
        let mut server = oe_server_for(applet, UnknownPolicy::Strict);
        server.accept(rig.session).unwrap();
        rig.server = server;
        rig
    }
}

#[test]
fn get_graphics_supervisor_copies_a_session_on_the_graphics_server_and_the_reply_is_these_bytes() {
    let mut rig = Rig::with_graphics();
    let proxy = rig.proxy();
    let controller = rig.open(proxy, GET_SELF_CONTROLLER);
    let handles = rig.kernel.handle_count();
    let reply = rig.object_raw(controller, GET_GRAPHICS_SUPERVISOR, &[]);
    // One more handle: the copied one, which names the new session.
    assert_eq!(rig.kernel.handle_count(), handles + 1);
    let message = Message::parse_reply(&reply, SessionKind::Domain).unwrap();
    assert_eq!(message.move_handles, Vec::<u32>::new());
    let [handle] = message.copy_handles[..] else {
        unreachable!("one copied handle: {:?}", message.copy_handles);
    };
    assert_eq!(reply, copied_handle_reply(handle));
    let session = rig.object_of(handle);
    assert_eq!(rig.kernel.owner(session), Some(GRAPHICS));
    // The graphics server's server is told of it, to open its root interface.
    assert_eq!(rig.kernel.take_connections(), [(GRAPHICS, session)]);
    // appletOE itself opened nothing more: the root, the proxy and the controller.
    assert_eq!(rig.server.interface_count(), 3);
}

/// The conflict the module docs list ("The graphics server"): NX2's new session per call, not
/// PR's one cached session.
#[test]
fn get_graphics_supervisor_makes_a_new_session_on_every_call() {
    let mut rig = Rig::with_graphics();
    let proxy = rig.proxy();
    let controller = rig.open(proxy, GET_SELF_CONTROLLER);
    let first = rig.object(controller, GET_GRAPHICS_SUPERVISOR, &[]);
    let second = rig.object(controller, GET_GRAPHICS_SUPERVISOR, &[]);
    assert_eq!(first.result, ResultCode::SUCCESS);
    assert_eq!(second.result, ResultCode::SUCCESS);
    let first = rig.object_of(first.copies[0]);
    let second = rig.object_of(second.copies[0]);
    assert_ne!(first, second);
    assert_eq!(
        rig.kernel.take_connections(),
        [(GRAPHICS, first), (GRAPHICS, second)]
    );
}

/// Starquake's own answer (module docs, "The graphics server"): no handle, no session.
#[test]
fn get_graphics_supervisor_without_a_graphics_server_is_unknown() {
    let mut rig = Rig::new();
    rig.kernel.add_service(GRAPHICS);
    let proxy = rig.proxy();
    let controller = rig.open(proxy, GET_SELF_CONTROLLER);
    let handles = rig.kernel.handle_count();
    let reply = rig.object_raw(controller, GET_GRAPHICS_SUPERVISOR, &[]);
    assert_eq!(reply, domain_reply(UNKNOWN, &[], &[]));
    assert_eq!(rig.kernel.handle_count(), handles);
    assert_eq!(rig.kernel.take_connections(), []);
    assert_eq!(rig.applet().graphics_server(), None);
}

/// An `appletOE` server built for a run whose graphics server is `graphics`, if it has one.
fn oe_server_with(graphics: Option<ServiceId>) -> Server {
    let applet = Applet::new(AppletSettings::eden_default());
    let applet = match graphics {
        Some(graphics) => applet.with_graphics_server(graphics),
        None => applet,
    };
    oe_server_for(applet, UnknownPolicy::Strict)
}

/// A restored server of the same run still hands out sessions on the graphics server, from a
/// controller opened before the save.
#[test]
fn a_restored_server_still_hands_out_sessions_on_the_graphics_server() {
    let mut rig = Rig::with_graphics();
    let proxy = rig.proxy();
    let controller = rig.open(proxy, GET_SELF_CONTROLLER);
    let saved = rig.server.save().unwrap();
    let mut restored = oe_server_with(Some(GRAPHICS));
    restored.restore(&saved).unwrap();
    assert_eq!(
        restored.family::<Applet>().unwrap().graphics_server(),
        Some(GRAPHICS)
    );
    rig.server = restored;
    let reply = rig.object(controller, GET_GRAPHICS_SUPERVISOR, &[]);
    assert_eq!(reply.result, ResultCode::SUCCESS);
    let session = rig.object_of(reply.copies[0]);
    assert_eq!(rig.kernel.take_connections(), [(GRAPHICS, session)]);
}

/// The graphics server's number is the run's wiring (module docs, "The graphics server"): a
/// snapshot whose number is not the restoring server's is refused, whichever side has one, and
/// the server keeps its own number and nothing of the snapshot.
#[test]
fn a_snapshot_with_another_graphics_server_is_refused() {
    let another = ServiceId(0x101);
    let cases = [
        (Some(GRAPHICS), Some(another)),
        (Some(another), Some(GRAPHICS)),
        (Some(GRAPHICS), None),
        (None, Some(GRAPHICS)),
    ];
    for (saved_with, run) in cases {
        let mut saving = oe_server_with(saved_with);
        let mut kernel = FakeKernel::new(0).unwrap();
        let session = kernel.new_object();
        saving.accept(session).unwrap();
        let saved = saving.save().unwrap();

        let mut restored = oe_server_with(run);
        let refused = SnapshotError::FamilyWiring("its applet names another graphics server");
        assert_eq!(
            restored.restore(&saved),
            Err(refused.clone()),
            "{saved_with:?} {run:?}"
        );
        assert_eq!(
            refused.to_string(),
            "the snapshot does not fit this run: its applet names another graphics server"
        );
        assert_eq!(restored.family::<Applet>().unwrap().graphics_server(), run);
        assert_eq!(restored.session_count(), 0, "{saved_with:?} {run:?}");
    }
    // The same number on both sides, or none on both, restores.
    for graphics in [Some(GRAPHICS), None] {
        let mut saving = oe_server_with(graphics);
        let mut kernel = FakeKernel::new(0).unwrap();
        saving.accept(kernel.new_object()).unwrap();
        let saved = saving.save().unwrap();
        let mut restored = oe_server_with(graphics);
        restored.restore(&saved).unwrap();
        assert_eq!(restored.session_count(), 1, "{graphics:?}");
    }
}

// ----- The Switch 2 additions (Eden PR #4588) ------------------------------------------------

/// `IPerformanceFunctions`' rows in PR #4588, both `nullptr`, so they have no shipping constant
/// (`eden-pr4588/src/core/hle/service/am/service/performance_functions.cpp` @ e5fb3b97: `{0,
/// nullptr, "IsCpuOverclockEnabled"}, {1, nullptr, "SetCpuOverclockEnabled"}`).
const PERFORMANCE_FUNCTIONS_ROWS: [u32; 2] = [0, 1];
/// `IGraphicsScalingParametersFunctions`' `nullptr` rows in PR #4588, so they have no shipping
/// constant (`graphics_scaling_parameters_functions.cpp` @ e5fb3b97: 1000
/// `SetSupportingGraphicsScalingParametersForDebug`, 1010 `SetGpuQuotaForDebug`, 1020
/// `ClearGpuQuotaForDebug`, 1030 `SetScreenRatioForDebug`, 1040 `ClearScreenRatioForDebug`).
const GRAPHICS_SCALING_DEBUG_ROWS: [u32; 5] = [1000, 1010, 1020, 1030, 1040];
/// `IApplicationFunctions::Unknown330`, which PR #4588 has and this port leaves out, so it has
/// no shipping constant (`application_functions.cpp` @ e5fb3b97: `{330,
/// D<&IApplicationFunctions::Unknown330>, "Unknown330"}`).
const UNKNOWN_330: u32 = 330;

/// The interfaces the Switch 2 proxy commands open, each with the proxy's command.
const SWITCH2_OPENED: [(u32, &str); 2] = [
    (GET_PERFORMANCE_FUNCTIONS, "IPerformanceFunctions"),
    (
        GET_GRAPHICS_SCALING_PARAMETERS_FUNCTIONS,
        "IGraphicsScalingParametersFunctions",
    ),
];

/// The 16-byte payload of `OpenApplicationProxyWithFlags` in PR #4588's layout (module docs,
/// "Layouts"): `flags` at offset 0, padding, the process id's `u64` slot at offset 8.
fn with_flags_payload(flags: u32, pid_slot: u64) -> Vec<u8> {
    let mut payload = flags.to_le_bytes().to_vec();
    payload.extend([0; 4]);
    payload.extend(pid_slot.to_le_bytes());
    payload
}

/// The buffer of `OpenApplicationProxyWithFlags` sent to domain object `root`, laid out word by
/// word: the header announcing a special header and 16 raw words (libnx counts 16 bytes of
/// alignment slack into the raw size, `cmifMakeRequest`), the special header asking for the
/// process id and one copied handle, the id's two words (the kernel fills them; here zero), the
/// handle, padding to the 16-byte boundary, the domain header (`SendMessage`, no input objects,
/// the `SFCI` header plus 16 payload bytes), `SFCI`, version 0, the command id, token 0, then
/// [`with_flags_payload`].
fn with_flags_by_hand(root: u32, process: u32, flags: u32) -> Vec<u8> {
    let payload = with_flags_payload(flags, 0);
    let header = CommandHeader {
        command_type: CommandType::Request,
        num_x: 0,
        num_a: 0,
        num_b: 0,
        num_w: 0,
        raw_data_words: 16,
        receive_list: ReceiveList::Disabled,
        has_special_header: true,
    };
    let special = SpecialHeader {
        send_pid: true,
        num_copy_handles: 1,
        num_move_handles: 0,
    };
    let domain = DomainRequestHeader {
        command: DomainCommand::SendMessage,
        input_object_count: 0,
        size: u32::try_from(CMIF_HEADER_BYTES + payload.len()).unwrap(),
        object_id: root,
        token: 0,
    };
    let [w0, w1] = header.encode().unwrap();
    // Six words to here: two of padding reach the 16-byte boundary.
    let mut words = vec![w0, w1, special.encode().unwrap(), 0, 0, process, 0, 0];
    words.extend(domain.encode().unwrap());
    words.extend([SFCI_MAGIC, 0, OPEN_APPLICATION_PROXY_WITH_FLAGS, 0]);
    words.extend(
        payload
            .chunks(4)
            .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap())),
    );
    buffer_of(&words)
}

impl Rig {
    /// `OpenApplicationProxyWithFlags` with `flags` and PR's 16-byte payload, to the domain's
    /// root `root`, with process id `pid`; the raw reply.
    fn open_proxy_with_flags_raw(&mut self, root: u32, flags: u32, pid: u64) -> Vec<u8> {
        let request = RequestBuilder::domain(
            CommandType::Request,
            root,
            OPEN_APPLICATION_PROXY_WITH_FLAGS,
        )
        .send_pid(pid)
        .copy_handle(self.process.raw())
        .data(&with_flags_payload(flags, 0));
        self.send_raw(self.session, &request)
    }

    /// Sends a whole message buffer laid out by hand; the raw reply.
    fn send_buffer(&mut self, mut buffer: Vec<u8>) -> Vec<u8> {
        let handled = self
            .server
            .handle(&mut self.kernel, self.session, &mut buffer);
        assert_eq!(handled, Ok(Handled::Replied));
        buffer
    }
}

#[test]
fn open_application_proxy_with_flags_laid_out_by_hand_opens_the_proxy_on_a_domain() {
    for flags in PROXY_FLAGS_ACCEPTED {
        let mut rig = Rig::new();
        let root = rig.convert();
        let request = with_flags_by_hand(root, rig.process.raw(), flags);
        let reply = rig.send_buffer(request);
        assert_eq!(
            reply,
            domain_reply(ResultCode::SUCCESS, &[], &[root + 1]),
            "flags {flags}"
        );
        assert!(rig.holds("IApplicationProxy"), "flags {flags}");
        assert_eq!(rig.server.interface_count(), 2, "flags {flags}");
    }
}

/// PR #4588's check, `R_UNLESS(flags == 0 || flags == 2, ResultUnknown)`: 0 and 2 open the
/// proxy; 1 and 3, the neighbours of 2, the largest value, and values whose low byte or low
/// half is 0 or 2 but whose higher bytes are not zero are refused with `ResultUnknown`, and open
/// nothing. The last set shows the whole `u32` is read, not only its low byte or half.
#[test]
fn open_application_proxy_with_flags_accepts_only_0_and_2() {
    assert_eq!(PROXY_FLAGS_ACCEPTED, [0, 2]);
    let cases = [
        (0, true),
        (1, false),
        (2, true),
        (3, false),
        (u32::MAX, false),
        // Low byte accepted, the second byte set.
        (0x0000_0100, false),
        (0x0000_0102, false),
        // Low half accepted, the third byte set.
        (0x0001_0000, false),
        (0x0001_0002, false),
        // Low three bytes accepted, the top byte set.
        (0x0200_0000, false),
    ];
    for (flags, accepted) in cases {
        let mut rig = Rig::new();
        let root = rig.convert();
        let reply = rig.open_proxy_with_flags_raw(root, flags, PROCESS_ID);
        let (expected, interfaces) = if accepted {
            (domain_reply(ResultCode::SUCCESS, &[], &[root + 1]), 2)
        } else {
            (domain_reply(UNKNOWN, &[], &[]), 1)
        };
        assert_eq!(reply, expected, "flags {flags}");
        assert_eq!(rig.server.interface_count(), interfaces, "flags {flags}");
    }
    // Every value up to 64 other than the two is refused too.
    let mut rig = Rig::new();
    let root = rig.convert();
    for flags in (0..=64).filter(|flags| !PROXY_FLAGS_ACCEPTED.contains(flags)) {
        let reply = rig.open_proxy_with_flags_raw(root, flags, PROCESS_ID);
        assert_eq!(reply, domain_reply(UNKNOWN, &[], &[]), "flags {flags}");
    }
    assert_eq!(rig.server.interface_count(), 1);
}

/// The checks in order (module docs, "The Switch 2 additions"): the payload must hold the flags
/// (starquake's divergence), then the flags (PR's `R_UNLESS`), then command 0's checks, the
/// process id before the handle. Each case with two faults shows which comes first.
#[test]
fn open_application_proxy_with_flags_checks_its_flags_first() {
    const REFUSED_FLAGS: u32 = 1;
    let mut rig = Rig::new();
    let root = rig.convert();
    let process = rig.process.raw();
    let accepted = PROXY_FLAGS_ACCEPTED[0];
    // (payload, pid, handle, result)
    let cases: [(Vec<u8>, bool, bool, ResultCode); 9] = [
        // No payload at all, with or without the id and handle: the flags cannot be read.
        (Vec::new(), true, true, INVALID_CMIF_REQUEST),
        (Vec::new(), false, false, INVALID_CMIF_REQUEST),
        // Refused flags win over a missing id or a missing handle.
        (with_flags_payload(REFUSED_FLAGS, 0), false, true, UNKNOWN),
        (with_flags_payload(REFUSED_FLAGS, 0), true, false, UNKNOWN),
        (with_flags_payload(REFUSED_FLAGS, 0), false, false, UNKNOWN),
        // Accepted flags: command 0's checks, the id first.
        (with_flags_payload(accepted, 0), false, true, UNKNOWN),
        (
            with_flags_payload(accepted, 0),
            true,
            false,
            INVALID_CMIF_REQUEST,
        ),
        (with_flags_payload(accepted, 0), false, false, UNKNOWN),
        // Exactly the flags' four bytes are enough: the id's slot is not read.
        (
            accepted.to_le_bytes().to_vec(),
            true,
            true,
            ResultCode::SUCCESS,
        ),
    ];
    for (payload, pid, handle, result) in cases {
        let mut request = RequestBuilder::domain(
            CommandType::Request,
            root,
            OPEN_APPLICATION_PROXY_WITH_FLAGS,
        )
        .data(&payload);
        if pid {
            request = request.send_pid(PROCESS_ID);
        }
        if handle {
            request = request.copy_handle(process);
        }
        let reply = parse(&rig.send_raw(rig.session, &request), SessionKind::Domain);
        assert_eq!(
            reply.result, result,
            "payload {payload:?}, pid {pid}, handle {handle}"
        );
    }
    // Only the last case opened anything.
    assert_eq!(rig.server.interface_count(), 2);

    // The exact edge, with libnx's layout, whose domain `size` counts the exact argument bytes
    // (the default layout rounds them up to a whole word, so a short payload would arrive as
    // four bytes): every length short of the flags is refused, the flags' own length opens the
    // proxy.
    let mut rig = Rig::new();
    let root = rig.convert();
    let process = rig.process.raw();
    let flags = accepted.to_le_bytes();
    for len in 0..=flags.len() {
        let request = RequestBuilder::domain(
            CommandType::Request,
            root,
            OPEN_APPLICATION_PROXY_WITH_FLAGS,
        )
        .layout(RequestLayout::Libnx)
        .send_pid(PROCESS_ID)
        .copy_handle(process)
        .data(&flags[..len]);
        let reply = parse(&rig.send_raw(rig.session, &request), SessionKind::Domain);
        let (result, interfaces) = if len == flags.len() {
            (ResultCode::SUCCESS, 2)
        } else {
            (INVALID_CMIF_REQUEST, 1)
        };
        assert_eq!(reply.result, result, "{len} bytes");
        assert_eq!(rig.server.interface_count(), interfaces, "{len} bytes");
    }
}

#[test]
fn open_application_proxy_with_flags_on_a_plain_session_moves_a_session_handle() {
    let mut rig = Rig::new();
    let request = RequestBuilder::cmif(CommandType::Request, OPEN_APPLICATION_PROXY_WITH_FLAGS)
        .layout(RequestLayout::Libnx)
        .send_pid(PROCESS_ID)
        .copy_handle(rig.process.raw())
        .data(&with_flags_payload(PROXY_FLAGS_ACCEPTED[1], 0));
    let reply = parse(&rig.send_raw(rig.session, &request), SessionKind::Session);
    assert_eq!(reply.result, ResultCode::SUCCESS);
    assert_eq!(reply.moves.len(), 1);
    let proxy = rig.object_of(reply.moves[0]);
    let window = parse(
        &rig.send_raw(
            proxy,
            &RequestBuilder::cmif(CommandType::Request, GET_WINDOW_CONTROLLER),
        ),
        SessionKind::Session,
    );
    assert_eq!(window.result, ResultCode::SUCCESS);
    assert_eq!(window.moves.len(), 1);
}

/// The proxy command 10 returns is the one command 0 returns: two rigs, one opened each way,
/// answer every proxy command with the same bytes, and save the same snapshot.
#[test]
fn the_proxy_opened_with_flags_answers_as_the_one_opened_without() {
    let mut plain = Rig::new();
    let root = plain.convert();
    let reply = parse(&plain.open_proxy_raw(root, PROCESS_ID), SessionKind::Domain);
    let plain_proxy = reply.objects[0];

    let mut flagged = Rig::new();
    let root = flagged.convert();
    let raw = flagged.open_proxy_with_flags_raw(root, PROXY_FLAGS_ACCEPTED[1], PROCESS_ID);
    let flagged_proxy = parse(&raw, SessionKind::Domain).objects[0];
    assert_eq!(flagged_proxy, plain_proxy);
    assert_eq!(flagged.server.save().unwrap(), plain.server.save().unwrap());

    for (command, interface) in OPENED.iter().chain(SWITCH2_OPENED.iter()) {
        let plain_reply = plain.object_raw(plain_proxy, *command, &[]);
        let flagged_reply = flagged.object_raw(flagged_proxy, *command, &[]);
        assert_eq!(flagged_reply, plain_reply, "{interface}");
        assert_eq!(
            parse(&flagged_reply, SessionKind::Domain).result,
            ResultCode::SUCCESS,
            "{interface}"
        );
    }
    // The window controller opened from the flagged proxy reports the caller's process id.
    let window = flagged.open(flagged_proxy, GET_WINDOW_CONTROLLER);
    assert_eq!(
        flagged.object_raw(window, GET_APPLET_RESOURCE_USER_ID, &[]),
        domain_reply(ResultCode::SUCCESS, &PROCESS_ID.to_le_bytes(), &[])
    );
    plain.open(plain_proxy, GET_WINDOW_CONTROLLER);
    assert_eq!(flagged.server.save().unwrap(), plain.server.save().unwrap());
}

#[test]
fn each_switch_2_proxy_command_returns_its_interface() {
    for (command, interface) in SWITCH2_OPENED {
        let mut rig = Rig::new();
        let proxy = rig.proxy();
        assert!(!rig.holds(interface), "{interface} before it is opened");
        let reply = rig.object_raw(proxy, command, &[]);
        assert_eq!(
            reply,
            domain_reply(ResultCode::SUCCESS, &[], &[proxy + 1]),
            "{interface}"
        );
        assert!(rig.holds(interface), "{interface}");
        assert_eq!(rig.server.interface_count(), 3, "{interface}");
    }
}

/// `GraphicsScalingParameters` as the reply carries it: each field a little-endian `u32`, in
/// declaration order. Distinct values, so a swap of any two fails.
#[test]
fn graphics_scaling_parameters_are_written_in_declaration_order() {
    let parameters = GraphicsScalingParameters {
        graphics_scale: 0x0403_0201,
        screen_ratio_numerator: 0x1413_1211,
        screen_ratio_denominator: 0x2423_2221,
    };
    let mut writer = Writer::new();
    parameters.write(&mut writer);
    assert_eq!(
        writer.as_bytes(),
        [
            0x01, 0x02, 0x03, 0x04, 0x11, 0x12, 0x13, 0x14, 0x21, 0x22, 0x23, 0x24
        ]
    );
}

/// The values the Switch 2 commands answer are PR #4588's (module docs, "The Switch 2
/// additions"), checked against the reference here once; the other tests derive from the
/// constants.
#[test]
fn the_switch_2_answers_are_pr_4588_s() {
    // `graphics_scaling_parameters_functions.cpp`: `*out_parameters = {10, 1, 1}`.
    assert_eq!(
        GraphicsScalingParameters::PR_4588,
        GraphicsScalingParameters {
            graphics_scale: 10,
            screen_ratio_numerator: 1,
            screen_ratio_denominator: 1,
        }
    );
    // `application_functions.cpp`: 320 over `applet.h`'s initial values (`Inherit !=
    // Disable`, and `false && ..`), then 321's `*out_enabled = false`.
    assert_eq!(
        [
            SCREENSHOT_INITIALLY_ENABLED,
            CONTINUOUS_RECORDING_INITIALLY_ENABLED,
            UNKNOWN_321_ENABLED,
        ],
        [true, false, false]
    );
}

#[test]
fn get_graphics_scaling_parameters_answers_pr_4588_s_parameters() {
    let mut rig = Rig::new();
    let proxy = rig.proxy();
    let scaling = rig.open(proxy, GET_GRAPHICS_SCALING_PARAMETERS_FUNCTIONS);
    let mut expected = Writer::new();
    GraphicsScalingParameters::PR_4588.write(&mut expected);
    assert_eq!(
        rig.object_raw(scaling, GET_GRAPHICS_SCALING_PARAMETERS, &[]),
        domain_reply(ResultCode::SUCCESS, expected.as_bytes(), &[])
    );
    // The notification flag changes no answer (PR reads it nowhere else).
    rig.object(
        scaling,
        SET_GRAPHICS_SCALE_CHANGED_NOTIFICATION_ENABLED,
        &[1],
    );
    assert!(rig.applet().graphics_scale_changed_notification_enabled());
    assert_eq!(
        rig.object_raw(scaling, GET_GRAPHICS_SCALING_PARAMETERS, &[]),
        domain_reply(ResultCode::SUCCESS, expected.as_bytes(), &[])
    );
}

/// The flag is kept in the applet, initially false; each step changes it or leaves it, and a
/// payload with no byte is refused and changes nothing, whichever value it holds. Only the first
/// byte is the `bool`: a zero byte followed by a non-zero one clears the flag.
#[test]
fn set_graphics_scale_changed_notification_enabled_keeps_its_flag() {
    let mut rig = Rig::new();
    let proxy = rig.proxy();
    let scaling = rig.open(proxy, GET_GRAPHICS_SCALING_PARAMETERS_FUNCTIONS);
    assert!(!rig.applet().graphics_scale_changed_notification_enabled());
    let refused = domain_reply(INVALID_CMIF_REQUEST, &[], &[]);
    let steps: [(&[u8], bool); 8] = [
        (&[], false),
        (&[0], false),
        (&[1], true),
        (&[], true),
        (&[0xFF], true),
        (&[0], false),
        (&[2], true),
        (&[0, 1], false),
    ];
    for (data, expected) in steps {
        let before = rig.applet();
        let reply = rig.object_raw(
            scaling,
            SET_GRAPHICS_SCALE_CHANGED_NOTIFICATION_ENABLED,
            data,
        );
        if data.is_empty() {
            assert_eq!(reply, refused, "{data:?}");
            assert_eq!(rig.applet(), before, "{data:?}");
        } else {
            assert_eq!(
                reply,
                domain_reply(ResultCode::SUCCESS, &[], &[]),
                "{data:?}"
            );
        }
        assert_eq!(
            rig.applet().graphics_scale_changed_notification_enabled(),
            expected,
            "{data:?}"
        );
    }
}

#[test]
fn get_screen_shot_and_continuous_recording_initially_enabled_answers_two_bytes() {
    let mut rig = Rig::new();
    let proxy = rig.proxy();
    let functions = rig.open(proxy, GET_APPLICATION_FUNCTIONS);
    let expected = [
        u8::from(SCREENSHOT_INITIALLY_ENABLED),
        u8::from(CONTINUOUS_RECORDING_INITIALLY_ENABLED),
    ];
    assert_eq!(
        rig.object_raw(
            functions,
            GET_SCREEN_SHOT_AND_CONTINUOUS_RECORDING_INITIALLY_ENABLED,
            &[]
        ),
        domain_reply(ResultCode::SUCCESS, &expected, &[])
    );
}

#[test]
fn unknown_321_answers_pr_4588_s_guess() {
    let mut rig = Rig::new();
    let proxy = rig.proxy();
    let functions = rig.open(proxy, GET_APPLICATION_FUNCTIONS);
    assert_eq!(
        rig.object_raw(functions, UNKNOWN_321, &[]),
        domain_reply(ResultCode::SUCCESS, &[u8::from(UNKNOWN_321_ENABLED)], &[])
    );
}

/// PR #4588's `nullptr` rows, and 330, which the scoping doc does not name, go to the
/// unknown-command policy.
#[test]
fn the_switch_2_commands_pr_4588_leaves_unimplemented_go_to_the_unknown_command_policy() {
    let mut rig = Rig::new();
    let proxy = rig.proxy();
    let performance = rig.open(proxy, GET_PERFORMANCE_FUNCTIONS);
    let scaling = rig.open(proxy, GET_GRAPHICS_SCALING_PARAMETERS_FUNCTIONS);
    let functions = rig.open(proxy, GET_APPLICATION_FUNCTIONS);
    let refused = domain_reply(UNKNOWN_COMMAND_ID, &[], &[]);
    for command in PERFORMANCE_FUNCTIONS_ROWS {
        assert_eq!(
            rig.object_raw(performance, command, &[]),
            refused,
            "{command}"
        );
    }
    for command in GRAPHICS_SCALING_DEBUG_ROWS {
        assert_eq!(rig.object_raw(scaling, command, &[]), refused, "{command}");
    }
    assert_eq!(rig.object_raw(functions, UNKNOWN_330, &[]), refused);
}

/// A server saved with the Switch 2 interfaces open and the notification flag set restores, and
/// the objects opened before the save answer after it.
#[test]
fn a_server_with_the_switch_2_interfaces_saves_and_restores() {
    let mut rig = Rig::new();
    let root = rig.convert();
    let raw = rig.open_proxy_with_flags_raw(root, PROXY_FLAGS_ACCEPTED[1], PROCESS_ID);
    let proxy = parse(&raw, SessionKind::Domain).objects[0];
    let performance = rig.open(proxy, GET_PERFORMANCE_FUNCTIONS);
    let scaling = rig.open(proxy, GET_GRAPHICS_SCALING_PARAMETERS_FUNCTIONS);
    let functions = rig.open(proxy, GET_APPLICATION_FUNCTIONS);
    rig.object(
        scaling,
        SET_GRAPHICS_SCALE_CHANGED_NOTIFICATION_ENABLED,
        &[1],
    );
    let saved = rig.server.save().unwrap();

    let mut restored = oe_server(AppletSettings::eden_default(), UnknownPolicy::Strict);
    restored.restore(&saved).unwrap();
    assert_eq!(restored.family::<Applet>(), Some(&rig.applet()));
    assert_eq!(restored.save().unwrap(), saved);
    rig.server = restored;
    assert!(rig.applet().graphics_scale_changed_notification_enabled());

    let mut parameters = Writer::new();
    GraphicsScalingParameters::PR_4588.write(&mut parameters);
    assert_eq!(
        rig.object_raw(scaling, GET_GRAPHICS_SCALING_PARAMETERS, &[]),
        domain_reply(ResultCode::SUCCESS, parameters.as_bytes(), &[])
    );
    rig.object(
        scaling,
        SET_GRAPHICS_SCALE_CHANGED_NOTIFICATION_ENABLED,
        &[0],
    );
    assert!(!rig.applet().graphics_scale_changed_notification_enabled());
    assert_eq!(
        rig.object_raw(functions, UNKNOWN_321, &[]),
        domain_reply(ResultCode::SUCCESS, &[u8::from(UNKNOWN_321_ENABLED)], &[])
    );
    assert_eq!(
        rig.object_raw(performance, PERFORMANCE_FUNCTIONS_ROWS[0], &[]),
        domain_reply(UNKNOWN_COMMAND_ID, &[], &[])
    );
    // The proxy opened with flags still opens interfaces.
    let window = rig.open(proxy, GET_WINDOW_CONTROLLER);
    assert_eq!(
        rig.object_raw(window, GET_APPLET_RESOURCE_USER_ID, &[]),
        domain_reply(ResultCode::SUCCESS, &PROCESS_ID.to_le_bytes(), &[])
    );
}
