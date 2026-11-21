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
use starquake_kernel::id::{ObjectId, SlabId};
use starquake_kernel::ipc::message::Current;
use starquake_kernel::result::{KernelError, ResultCode};

use super::*;
use crate::framework::control::CONVERT_CURRENT_OBJECT_TO_DOMAIN;
use crate::framework::fake::FakeKernel;
use crate::framework::result::{INVALID_CMIF_REQUEST, UNKNOWN_COMMAND_ID};
use crate::framework::{Handled, Kernel, Protocol, Session, SnapshotError};

use super::command::application_functions::{
    GET_DESIRED_LANGUAGE, GET_GPU_ERROR_DETECTED_SYSTEM_EVENT,
    GET_SCREEN_SHOT_AND_CONTINUOUS_RECORDING_INITIALLY_ENABLED, GET_UNKNOWN_EVENT_210,
    IS_GAME_PLAY_RECORDING_SUPPORTED, NOTIFY_RUNNING, POP_LAUNCH_PARAMETER,
    SET_GAME_PLAY_RECORDING_STATE, UNKNOWN_321, UNKNOWN_330,
};
use super::command::common_state_getter::{
    GET_CURRENT_FOCUS_STATE, GET_EVENT_HANDLE, GET_OPERATION_MODE, GET_PERFORMANCE_MODE,
    RECEIVE_MESSAGE, SET_CPU_BOOST_MODE,
    SET_REQUEST_EXIT_TO_LIBRARY_APPLET_AT_EXECUTE_NEXT_PROGRAM_ENABLED,
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
    SET_RESTART_MESSAGE_ENABLED, SET_RTC_SCREEN_SHARING_AUDIO_ENABLED, SET_SCREEN_SHOT_PERMISSION,
};
use super::command::window_controller::{ACQUIRE_FOREGROUND_RIGHTS, GET_APPLET_RESOURCE_USER_ID};

/// `ISelfController::LockExit`, which this port leaves out, so it has no shipping constant
/// (`eden/src/core/hle/service/am/service/self_controller.cpp`: `{1,
/// D<&ISelfController::LockExit>, "LockExit"}`).
const LOCK_EXIT: u32 = 1;
/// `ICommonStateGetter::GetBootMode`, which this port leaves out, so it has no shipping
/// constant (`common_state_getter.cpp`: `{8, D<&ICommonStateGetter::GetBootMode>,
/// "GetBootMode"}`).
const GET_BOOT_MODE: u32 = 8;

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
        self.server.family::<Applet>().unwrap().clone()
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
        let lifecycle = rig.applet().lifecycle().clone();
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
    // Each other kept term alone makes a message available, focus notifications on or off.
    for notify in [false, true] {
        let quiet = Lifecycle {
            notifications: Notifications {
                focus_state_changed: notify,
                ..Notifications::default()
            },
            has_focus_state_changed: false,
            ..Lifecycle::default()
        };
        assert!(!quiet.message_available());
        let terms = [
            Lifecycle {
                has_operation_mode_changed: true,
                ..quiet.clone()
            },
            Lifecycle {
                has_performance_mode_changed: true,
                ..quiet.clone()
            },
            Lifecycle {
                unordered_messages: [AppletMessage::Resume].into_iter().collect(),
                ..quiet.clone()
            },
        ];
        for term in terms {
            assert!(term.message_available(), "{term:?}");
        }
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
            has_operation_mode_changed: false,
            has_performance_mode_changed: false,
            unordered_messages: VecDeque::new(),
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
    assert_eq!(rig.object_raw(getter, GET_BOOT_MODE, &[]), refused);
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

/// The ids and method names of `session`'s CMIF table, in its order.
fn names(session: &dyn Session) -> Vec<(u32, &'static str)> {
    session
        .commands()
        .cmif
        .iter()
        .map(|row| (row.id, row.name))
        .collect()
}

#[test]
fn the_tables_name_the_commands_as_eden_does() {
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
            (POP_LAUNCH_PARAMETER, "pop_launch_parameter"),
            (GET_DESIRED_LANGUAGE, "get_desired_language"),
            (NOTIFY_RUNNING, "notify_running"),
            (
                IS_GAME_PLAY_RECORDING_SUPPORTED,
                "is_game_play_recording_supported"
            ),
            (
                SET_GAME_PLAY_RECORDING_STATE,
                "set_game_play_recording_state"
            ),
            (
                GET_GPU_ERROR_DETECTED_SYSTEM_EVENT,
                "get_gpu_error_detected_system_event"
            ),
            (GET_UNKNOWN_EVENT_210, "get_unknown_event_210"),
            (
                GET_SCREEN_SHOT_AND_CONTINUOUS_RECORDING_INITIALLY_ENABLED,
                "get_screen_shot_and_continuous_recording_initially_enabled"
            ),
            (UNKNOWN_321, "unknown_321"),
            (UNKNOWN_330, "unknown_330"),
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

#[test]
fn the_getter_and_controller_tables_name_the_commands_as_eden_does() {
    assert_eq!(
        names(&ICommonStateGetter),
        [
            (GET_EVENT_HANDLE, "get_event_handle"),
            (RECEIVE_MESSAGE, "receive_message"),
            (GET_OPERATION_MODE, "get_operation_mode"),
            (GET_PERFORMANCE_MODE, "get_performance_mode"),
            (GET_CURRENT_FOCUS_STATE, "get_current_focus_state"),
            (SET_CPU_BOOST_MODE, "set_cpu_boost_mode"),
            (
                SET_REQUEST_EXIT_TO_LIBRARY_APPLET_AT_EXECUTE_NEXT_PROGRAM_ENABLED,
                "set_request_exit_to_library_applet_at_execute_next_program_enabled"
            ),
        ]
    );
    assert_eq!(
        names(&ISelfController),
        [
            (EXIT, "exit"),
            (SET_SCREEN_SHOT_PERMISSION, "set_screen_shot_permission"),
            (
                SET_OPERATION_MODE_CHANGED_NOTIFICATION,
                "set_operation_mode_changed_notification"
            ),
            (
                SET_PERFORMANCE_MODE_CHANGED_NOTIFICATION,
                "set_performance_mode_changed_notification"
            ),
            (SET_FOCUS_HANDLING_MODE, "set_focus_handling_mode"),
            (SET_RESTART_MESSAGE_ENABLED, "set_restart_message_enabled"),
            (
                SET_OUT_OF_FOCUS_SUSPENDING_ENABLED,
                "set_out_of_focus_suspending_enabled"
            ),
            (
                GET_ACCUMULATED_SUSPENDED_TICK_CHANGED_EVENT,
                "get_accumulated_suspended_tick_changed_event"
            ),
            (GET_GRAPHICS_SUPERVISOR, "get_graphics_supervisor"),
            (
                SET_RTC_SCREEN_SHARING_AUDIO_ENABLED,
                "set_rtc_screen_sharing_audio_enabled"
            ),
        ]
    );
    assert_eq!(
        ICommonStateGetter
            .commands()
            .name(Protocol::Cmif, GET_EVENT_HANDLE),
        Some("get_event_handle")
    );
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
    let lifecycle = rig.applet().lifecycle().clone();
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
    restored.restore_beside(&saved, &rig.kernel).unwrap();
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
    restored.restore_beside(&saved, &rig.kernel).unwrap();
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
    // Restored beside the kernel that made the event, so the restore's check passes; the
    // commands then run on the fresh kernel, which did not make it.
    fresh.server.restore_beside(&saved, &rig.kernel).unwrap();
    assert!(!fresh.applet().lifecycle().notifications.focus_state_changed);

    // Notify on, with a focus change pending: a message becomes available, and the signal fails.
    let on = fresh.object(controller, SET_FOCUS_HANDLING_MODE, &[1, 0, 1]);
    assert_eq!(on.result, KernelError::InvalidHandle.code());
    let lifecycle = fresh.applet().lifecycle().clone();
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
    // Restored beside the kernel that made the event, so the restore's check passes; the
    // commands then run on the fresh kernel, which did not make it.
    fresh.server.restore_beside(&saved, &rig.kernel).unwrap();
    let off = fresh.object(controller, SET_FOCUS_HANDLING_MODE, &[0, 0, 1]);
    assert_eq!(off.result, KernelError::InvalidHandle.code());
    let lifecycle = fresh.applet().lifecycle().clone();
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
    restored.restore_beside(&saved, &rig.kernel).unwrap();
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
            restored.restore_beside(&saved, &kernel),
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
        restored.restore_beside(&saved, &kernel).unwrap();
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
        Applet::new(AppletSettings::eden_default())
            .screenshot_and_continuous_recording_initially_enabled(),
        (true, false)
    );
    // `applet.h`'s initial fields: `ScreenshotPermission screenshot_permission{}` (`Inherit`),
    // and `GamePlayRecordingState game_play_recording_state{GamePlayRecordingState::Disabled}`.
    let applet = Applet::new(AppletSettings::eden_default());
    assert_eq!(
        applet.screenshot_permission(),
        ScreenshotPermission::INHERIT
    );
    assert_eq!(
        applet.game_play_recording_state(),
        GamePlayRecordingState::DISABLED
    );
    // 321's `false`, `applet.h`'s `bool game_play_recording_supported{}`, and 330's `*out = 0`.
    assert_eq!(
        (
            UNKNOWN_321_ENABLED,
            GAME_PLAY_RECORDING_SUPPORTED,
            UNKNOWN_330_VALUE
        ),
        (false, false, 0)
    );
    // `am_types.h`: `ScreenshotPermission { Inherit = 0, Enable = 1, Disable = 2 }` and
    // `GamePlayRecordingState { Disabled, Enabled }`.
    assert_eq!(
        [
            ScreenshotPermission::INHERIT,
            ScreenshotPermission::ENABLE,
            ScreenshotPermission::DISABLE,
        ]
        .map(ScreenshotPermission::raw),
        [0, 1, 2]
    );
    assert_eq!(
        [
            GamePlayRecordingState::DISABLED,
            GamePlayRecordingState::ENABLED
        ]
        .map(GamePlayRecordingState::raw),
        [0, 1]
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
    // Screenshots on (`Inherit`), recording off: distinct bytes, so a swap of the two fails.
    let expected = [1, 0];
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

/// PR #4588's `nullptr` rows go to the unknown-command policy.
#[test]
fn the_switch_2_commands_pr_4588_leaves_unimplemented_go_to_the_unknown_command_policy() {
    let mut rig = Rig::new();
    let proxy = rig.proxy();
    let performance = rig.open(proxy, GET_PERFORMANCE_FUNCTIONS);
    let scaling = rig.open(proxy, GET_GRAPHICS_SCALING_PARAMETERS_FUNCTIONS);
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
    restored.restore_beside(&saved, &rig.kernel).unwrap();
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

// ----- The title's start-up commands: 900, 130 and 330 ---------------------------------------

/// `ICommonStateGetter` 900 takes nothing, answers nothing, and sets the applet's flag; a second
/// call leaves it set. Nothing else in the applet changes.
#[test]
fn set_request_exit_to_library_applet_at_execute_next_program_enabled_sets_the_flag() {
    let mut rig = Rig::new();
    let proxy = rig.proxy();
    let getter = rig.open(proxy, GET_COMMON_STATE_GETTER);
    let before = rig.applet();
    assert!(!before.request_exit_to_library_applet_at_execute_next_program_enabled());
    for _ in 0..2 {
        assert_eq!(
            rig.object_raw(
                getter,
                SET_REQUEST_EXIT_TO_LIBRARY_APPLET_AT_EXECUTE_NEXT_PROGRAM_ENABLED,
                &[]
            ),
            domain_reply(ResultCode::SUCCESS, &[], &[])
        );
        let after = rig.applet();
        assert!(after.request_exit_to_library_applet_at_execute_next_program_enabled());
        assert_eq!(
            Applet {
                request_exit_to_library_applet_at_execute_next_program_enabled: false,
                ..after
            },
            before
        );
    }
}

/// `IApplicationFunctions` 130 copies one event, made the first time and never signalled: the
/// same event on every call, and not the message or suspended-tick event.
#[test]
fn get_gpu_error_detected_system_event_copies_one_event_never_signalled() {
    let mut rig = Rig::new();
    let proxy = rig.proxy();
    let functions = rig.open(proxy, GET_APPLICATION_FUNCTIONS);
    let getter = rig.open(proxy, GET_COMMON_STATE_GETTER);
    let controller = rig.open(proxy, GET_SELF_CONTROLLER);
    assert_eq!(rig.applet().gpu_error_detected_event(), None);
    let raw = rig.object_raw(functions, GET_GPU_ERROR_DETECTED_SYSTEM_EVENT, &[]);
    let first = parse(&raw, SessionKind::Domain).copies[0];
    assert_eq!(raw, copied_handle_reply(first));
    let event = rig.object_of(first);
    assert_eq!(rig.applet().gpu_error_detected_event(), Some(event));
    assert_eq!(rig.kernel.signals(event), Some(0));

    let again = rig.object(functions, GET_GPU_ERROR_DETECTED_SYSTEM_EVENT, &[]);
    assert_eq!(again.result, ResultCode::SUCCESS);
    assert_eq!(rig.object_of(again.copies[0]), event);

    let message = rig.object(getter, GET_EVENT_HANDLE, &[]).copies[0];
    let tick = rig
        .object(
            controller,
            GET_ACCUMULATED_SUSPENDED_TICK_CHANGED_EVENT,
            &[],
        )
        .copies[0];
    assert_ne!(rig.object_of(message), event);
    assert_ne!(rig.object_of(tick), event);
    assert_eq!(rig.applet().gpu_error_detected_event(), Some(event));
    assert_eq!(rig.kernel.signals(event), Some(0));
}

#[test]
fn unknown_330_answers_the_references_byte() {
    let mut rig = Rig::new();
    let proxy = rig.proxy();
    let functions = rig.open(proxy, GET_APPLICATION_FUNCTIONS);
    let before = rig.applet();
    assert_eq!(
        rig.object_raw(functions, UNKNOWN_330, &[]),
        domain_reply(ResultCode::SUCCESS, &[UNKNOWN_330_VALUE], &[])
    );
    assert_eq!(rig.applet(), before);
}

// ----- Screenshots and recording: 10, 65, 67 and 320 -----------------------------------------

#[test]
fn is_game_play_recording_supported_answers_the_references_false() {
    let mut rig = Rig::new();
    let proxy = rig.proxy();
    let functions = rig.open(proxy, GET_APPLICATION_FUNCTIONS);
    let supported = domain_reply(
        ResultCode::SUCCESS,
        &[u8::from(GAME_PLAY_RECORDING_SUPPORTED)],
        &[],
    );
    assert_eq!(
        rig.object_raw(functions, IS_GAME_PLAY_RECORDING_SUPPORTED, &[]),
        supported
    );
    // Setting the recording state on does not make it supported.
    let on = rig.object(
        functions,
        SET_GAME_PLAY_RECORDING_STATE,
        &GamePlayRecordingState::ENABLED.raw().to_le_bytes(),
    );
    assert_eq!(on.result, ResultCode::SUCCESS);
    assert_eq!(
        rig.object_raw(functions, IS_GAME_PLAY_RECORDING_SUPPORTED, &[]),
        supported
    );
}

/// Words that differ from `value` only above its low byte, one for each higher byte.
fn above_the_low_byte(value: u32) -> [u32; 3] {
    [value | 1 << 8, value | 1 << 16, value | 1 << 24]
}

/// Screenshots are off exactly for `Disable`: every other word, including ones whose low byte
/// is `Disable`'s, enables them.
#[test]
fn screenshots_are_enabled_for_every_permission_but_disable() {
    let disable = ScreenshotPermission::DISABLE.raw();
    assert!(!ScreenshotPermission::DISABLE.screenshots_enabled());
    let mut enabled = vec![
        ScreenshotPermission::INHERIT.raw(),
        ScreenshotPermission::ENABLE.raw(),
        disable - 1,
        disable + 1,
        u32::MAX,
    ];
    enabled.extend(above_the_low_byte(disable));
    for raw in enabled {
        assert!(
            ScreenshotPermission::from_raw(raw).screenshots_enabled(),
            "{raw:#x}"
        );
    }
}

/// Recording is on exactly when the applet supports it and the state is `Enabled`.
#[test]
fn continuous_recording_needs_support_and_the_enabled_state() {
    let enabled = GamePlayRecordingState::ENABLED.raw();
    let mut others = vec![
        GamePlayRecordingState::DISABLED.raw(),
        enabled + 1,
        u32::MAX,
    ];
    others.extend(above_the_low_byte(enabled));
    for supported in [false, true] {
        assert_eq!(
            GamePlayRecordingState::ENABLED.continuous_recording_enabled(supported),
            supported
        );
        for raw in &others {
            assert!(
                !GamePlayRecordingState::from_raw(*raw).continuous_recording_enabled(supported),
                "{raw:#x} {supported}"
            );
        }
    }
}

/// `SetScreenShotPermission` keeps the word it is sent, whatever it is, and 320's first byte
/// follows it: off exactly for `Disable`, on for a word that differs from it only above the low
/// byte. Each step changes the permission or leaves it as it was.
#[test]
fn set_screen_shot_permission_keeps_its_word_and_320_follows_it() {
    let mut rig = Rig::new();
    let proxy = rig.proxy();
    let controller = rig.open(proxy, GET_SELF_CONTROLLER);
    let functions = rig.open(proxy, GET_APPLICATION_FUNCTIONS);
    let disable = ScreenshotPermission::DISABLE.raw();
    let mut steps = vec![
        disable,
        disable,
        ScreenshotPermission::ENABLE.raw(),
        disable,
        ScreenshotPermission::INHERIT.raw(),
        u32::MAX,
    ];
    for above in above_the_low_byte(disable) {
        steps.extend([disable, above]);
    }
    for raw in steps {
        let reply = rig.object_raw(controller, SET_SCREEN_SHOT_PERMISSION, &raw.to_le_bytes());
        assert_eq!(
            reply,
            domain_reply(ResultCode::SUCCESS, &[], &[]),
            "{raw:#x}"
        );
        assert_eq!(rig.applet().screenshot_permission().raw(), raw);
        let screenshot = u8::from(raw != disable);
        assert_eq!(
            rig.object_raw(
                functions,
                GET_SCREEN_SHOT_AND_CONTINUOUS_RECORDING_INITIALLY_ENABLED,
                &[]
            ),
            domain_reply(ResultCode::SUCCESS, &[screenshot, 0], &[]),
            "{raw:#x}"
        );
    }
}

/// `SetGamePlayRecordingState` keeps the word it is sent, whatever it is. 320's second byte
/// stays off whatever the state, because the applet does not support recording.
#[test]
fn set_game_play_recording_state_keeps_its_word_and_recording_stays_off() {
    let mut rig = Rig::new();
    let proxy = rig.proxy();
    let functions = rig.open(proxy, GET_APPLICATION_FUNCTIONS);
    let enabled = GamePlayRecordingState::ENABLED.raw();
    let mut steps = vec![
        enabled,
        enabled,
        GamePlayRecordingState::DISABLED.raw(),
        enabled,
        u32::MAX,
    ];
    steps.extend(above_the_low_byte(enabled));
    for raw in steps {
        let reply = rig.object_raw(functions, SET_GAME_PLAY_RECORDING_STATE, &raw.to_le_bytes());
        assert_eq!(
            reply,
            domain_reply(ResultCode::SUCCESS, &[], &[]),
            "{raw:#x}"
        );
        assert_eq!(rig.applet().game_play_recording_state().raw(), raw);
        assert_eq!(
            rig.object_raw(
                functions,
                GET_SCREEN_SHOT_AND_CONTINUOUS_RECORDING_INITIALLY_ENABLED,
                &[]
            ),
            domain_reply(ResultCode::SUCCESS, &[1, 0], &[]),
            "{raw:#x}"
        );
    }
}

/// Both `u32` setters refuse a payload one, two or three bytes short of the word (and an empty
/// one) with [`INVALID_CMIF_REQUEST`], and change nothing; the full word is accepted. The
/// requests use libnx's layout, whose domain `size` counts the exact argument bytes (the default
/// layout rounds them up to a whole word, so a short payload would arrive as four bytes).
#[test]
fn a_payload_too_short_for_the_word_is_refused_and_changes_nothing() {
    let mut rig = Rig::new();
    let proxy = rig.proxy();
    let controller = rig.open(proxy, GET_SELF_CONTROLLER);
    let functions = rig.open(proxy, GET_APPLICATION_FUNCTIONS);
    let word = u32::MAX.to_le_bytes();
    for (object, command) in [
        (controller, SET_SCREEN_SHOT_PERMISSION),
        (functions, SET_GAME_PLAY_RECORDING_STATE),
    ] {
        for len in 0..=word.len() {
            let before = rig.applet();
            let request = RequestBuilder::domain(CommandType::Request, object, command)
                .layout(RequestLayout::Libnx)
                .data(&word[..len]);
            let reply = parse(&rig.send_raw(rig.session, &request), SessionKind::Domain);
            if len == word.len() {
                assert_eq!(reply.result, ResultCode::SUCCESS, "{command}");
            } else {
                assert_eq!(reply.result, INVALID_CMIF_REQUEST, "{command} {len}");
                assert_eq!(rig.applet(), before, "{command} {len}");
            }
        }
    }
    assert_eq!(rig.applet().screenshot_permission().raw(), u32::MAX);
    assert_eq!(rig.applet().game_play_recording_state().raw(), u32::MAX);
}

/// A server saved after the title's start-up commands restores, and answers from what it saved:
/// the screenshot permission, the recording state, the flag and the GPU-error event.
#[test]
fn a_server_after_the_titles_start_up_commands_saves_and_restores() {
    let mut rig = Rig::new();
    let proxy = rig.proxy();
    let getter = rig.open(proxy, GET_COMMON_STATE_GETTER);
    let controller = rig.open(proxy, GET_SELF_CONTROLLER);
    let functions = rig.open(proxy, GET_APPLICATION_FUNCTIONS);
    let steps: [(u32, u32, Vec<u8>); 4] = [
        (
            getter,
            SET_REQUEST_EXIT_TO_LIBRARY_APPLET_AT_EXECUTE_NEXT_PROGRAM_ENABLED,
            Vec::new(),
        ),
        (functions, GET_GPU_ERROR_DETECTED_SYSTEM_EVENT, Vec::new()),
        (
            controller,
            SET_SCREEN_SHOT_PERMISSION,
            ScreenshotPermission::DISABLE.raw().to_le_bytes().to_vec(),
        ),
        (
            functions,
            SET_GAME_PLAY_RECORDING_STATE,
            GamePlayRecordingState::ENABLED.raw().to_le_bytes().to_vec(),
        ),
    ];
    for (object, command, data) in &steps {
        let reply = rig.object(*object, *command, data);
        assert_eq!(reply.result, ResultCode::SUCCESS, "{command}");
    }
    let event = rig.applet().gpu_error_detected_event().unwrap();
    let saved = rig.server.save().unwrap();

    let mut restored = oe_server(AppletSettings::eden_default(), UnknownPolicy::Strict);
    restored.restore_beside(&saved, &rig.kernel).unwrap();
    assert_eq!(restored.family::<Applet>(), Some(&rig.applet()));
    assert_eq!(restored.save().unwrap(), saved);
    rig.server = restored;

    let applet = rig.applet();
    assert!(applet.request_exit_to_library_applet_at_execute_next_program_enabled());
    assert_eq!(
        applet.screenshot_permission(),
        ScreenshotPermission::DISABLE
    );
    assert_eq!(
        applet.game_play_recording_state(),
        GamePlayRecordingState::ENABLED
    );
    assert_eq!(
        rig.object_raw(
            functions,
            GET_SCREEN_SHOT_AND_CONTINUOUS_RECORDING_INITIALLY_ENABLED,
            &[]
        ),
        domain_reply(ResultCode::SUCCESS, &[0, 0], &[])
    );
    let again = rig.object(functions, GET_GPU_ERROR_DETECTED_SYSTEM_EVENT, &[]);
    assert_eq!(again.result, ResultCode::SUCCESS);
    assert_eq!(rig.object_of(again.copies[0]), event);
}

/// How a test reads the id the applet keeps for one of its events.
type KeptEvent = fn(&Applet) -> Option<ObjectId>;

/// The three never-signalled events, each with the interface that returns it, the command, and
/// how to read the id the applet keeps for it.
fn never_signalled_events() -> [(u32, u32, KeptEvent); 3] {
    [
        (
            GET_SELF_CONTROLLER,
            GET_ACCUMULATED_SUSPENDED_TICK_CHANGED_EVENT,
            Applet::suspended_tick_event,
        ),
        (
            GET_APPLICATION_FUNCTIONS,
            GET_GPU_ERROR_DETECTED_SYSTEM_EVENT,
            Applet::gpu_error_detected_event,
        ),
        (
            GET_APPLICATION_FUNCTIONS,
            GET_UNKNOWN_EVENT_210,
            Applet::unknown_event,
        ),
    ]
}

/// A restored suspended-tick, GPU-error or unknown event is checked against the kernel before
/// its handle is copied (module docs, "The applet"): restored onto a kernel where the saved id
/// names another kind of object, or nothing the kernel made, the command answers the kernel's
/// `InvalidHandle`, copies no handle and leaves the applet as it was, on every call.
#[test]
fn a_restored_event_the_kernel_did_not_make_is_refused() {
    for (interface, command, kept) in never_signalled_events() {
        let mut rig = Rig::new();
        let proxy = rig.proxy();
        let object = rig.open(proxy, interface);
        let reply = rig.object(object, command, &[]);
        assert_eq!(reply.result, ResultCode::SUCCESS, "{command}");
        let event = kept(&rig.applet()).unwrap();
        let saved = rig.server.save().unwrap();

        for names_an_object in [true, false] {
            // A fresh kernel mints the same session id; the event's id is the next it mints,
            // made here as an object that is not an event, or never made.
            let mut fresh = Rig::new();
            assert_eq!(fresh.session, rig.session);
            if names_an_object {
                assert_eq!(fresh.kernel.new_object(), event);
            }
            assert_eq!(fresh.kernel.signals(event), None);
            // Restored beside the kernel that made the event, so the restore's check passes;
            // the command then runs on the fresh kernel, which did not make it.
            fresh.server.restore_beside(&saved, &rig.kernel).unwrap();
            let before = fresh.applet();
            let handles = fresh.kernel.handle_count();
            for _ in 0..2 {
                assert_eq!(
                    fresh.object_raw(object, command, &[]),
                    domain_reply(KernelError::InvalidHandle.code(), &[], &[]),
                    "{command} {names_an_object}"
                );
                assert_eq!(fresh.kernel.handle_count(), handles);
                assert_eq!(fresh.applet(), before);
                assert_eq!(kept(&fresh.applet()), Some(event));
            }
        }
    }
}

/// The check of a kept event does not signal or clear it: restored onto the kernel that made
/// it, the command copies the same event, still never signalled.
#[test]
fn a_restored_event_the_kernel_made_is_copied_unchanged() {
    for (interface, command, kept) in never_signalled_events() {
        let mut rig = Rig::new();
        let proxy = rig.proxy();
        let object = rig.open(proxy, interface);
        rig.object(object, command, &[]);
        let event = kept(&rig.applet()).unwrap();
        let saved = rig.server.save().unwrap();
        let mut restored = oe_server(AppletSettings::eden_default(), UnknownPolicy::Strict);
        restored.restore_beside(&saved, &rig.kernel).unwrap();
        rig.server = restored;

        let raw = rig.object_raw(object, command, &[]);
        let copied = parse(&raw, SessionKind::Domain).copies[0];
        assert_eq!(raw, copied_handle_reply(copied), "{command}");
        assert_eq!(rig.object_of(copied), event);
        assert_eq!(rig.kernel.signals(event), Some(0));
        assert_eq!(rig.kernel.is_signalled(event), Some(false));
    }
}

/// How a test sets the id the applet keeps for one of its events.
type SetKeptEvent = fn(&mut Applet, Option<ObjectId>);

/// The four events the applet keeps, by name, with how to set each.
fn kept_event_slots() -> [(&'static str, SetKeptEvent); 4] {
    [
        ("message", |applet, id| applet.message_event = id),
        ("suspended tick", |applet, id| {
            applet.suspended_tick_event = id;
        }),
        ("GPU error", |applet, id| {
            applet.gpu_error_detected_event = id;
        }),
        ("unknown", |applet, id| applet.unknown_event = id),
    ]
}

/// An applet that keeps a distinct event in each slot: slot `i` keeps
/// `ObjectId::from_parts(i + 1, 1)`.
fn applet_with_distinct_events() -> Applet {
    let mut applet = Applet::new(AppletSettings::eden_default());
    for (index, (_, set)) in (1_u32..).zip(kept_event_slots()) {
        set(&mut applet, Some(ObjectId::from_parts(index, 1)));
    }
    applet
}

/// A restore with `applet` as the snapshot's family state, into a fresh server, beside a kernel
/// that holds every event the applet names ([`EveryEvent`]): the result, and the server.
fn restore_applet(applet: Applet) -> (Result<(), SnapshotError>, Server) {
    let saved = oe_server_for(applet, UnknownPolicy::Strict).save().unwrap();
    let mut server = oe_server(AppletSettings::eden_default(), UnknownPolicy::Strict);
    let result = server.restore_beside(&saved, &EveryEvent);
    (result, server)
}

/// A kernel, for tests of what a restore refuses before it asks the kernel, that answers every
/// id as a live readable event of every server's, and no id as a shared memory: so only the
/// checks before the kernel's can refuse.
struct EveryEvent;

impl RestoredKernel for EveryEvent {
    fn is_service_readable_event(&self, _: ServiceId, _: ObjectId) -> bool {
        true
    }

    fn shared_memory(&self, _: ObjectId) -> Option<starquake_kernel::memory::SharedMemory> {
        None
    }
}

/// Live operation makes each kept event once, so the four ids are distinct; a snapshot that
/// keeps one id in two slots is refused with the exact error, whichever two, and the server is
/// left as it was (module docs, "The applet"). Each refused case has an accepted neighbour that
/// differs only in that slot's id: its generation, or no event at all.
#[test]
fn a_snapshot_that_keeps_one_event_in_two_slots_is_refused() {
    let fresh = Applet::new(AppletSettings::eden_default());
    let refused = SnapshotError::Decode(postcard::Error::SerdeDeCustom);
    let slots = kept_event_slots();
    let mut pairs = 0;
    for (first, (first_name, _)) in slots.iter().enumerate() {
        for (second_name, set_second) in slots.iter().skip(first + 1) {
            pairs += 1;
            let shared = ObjectId::from_parts(u32::try_from(first + 1).unwrap(), 1);
            let mut applet = applet_with_distinct_events();
            set_second(&mut applet, Some(shared));
            let (result, server) = restore_applet(applet);
            assert_eq!(result, Err(refused.clone()), "{first_name} {second_name}");
            assert_eq!(server.family::<Applet>(), Some(&fresh));
            assert_eq!(server.session_count(), 0);

            // The same index in another generation is another object; no event is none.
            let other_generation = ObjectId::from_parts(u32::try_from(first + 1).unwrap(), 2);
            for id in [Some(other_generation), None] {
                let mut applet = applet_with_distinct_events();
                set_second(&mut applet, id);
                let (result, server) = restore_applet(applet.clone());
                assert_eq!(result, Ok(()), "{first_name} {second_name} {id:?}");
                assert_eq!(server.family::<Applet>(), Some(&applet));
            }
        }
    }
    assert_eq!(pairs, 6);
    // No event in any slot, as at launch, restores: `None` is not an id.
    assert_eq!(restore_applet(fresh.clone()).0, Ok(()));
}

/// The forgery the check closes: an applet whose unknown event is its message event. Restored,
/// `GetUnknownEvent210`'s check would clear the message event while a message is pending, and a
/// guest waiting on it would never wake; the snapshot is refused instead, so the message event
/// of a run stays signalled.
#[test]
fn a_snapshot_whose_unknown_event_is_the_message_event_is_refused() {
    let mut rig = Rig::new();
    let (_, _, event) = rig.getter_with_event();
    assert_eq!(rig.kernel.is_signalled(event), Some(true));
    let mut forged = rig.applet();
    forged.unknown_event = Some(event);
    let saved = oe_server_for(forged, UnknownPolicy::Strict).save().unwrap();
    let before = rig.applet();
    assert_eq!(
        rig.server.restore_beside(&saved, &rig.kernel),
        Err(SnapshotError::Decode(postcard::Error::SerdeDeCustom))
    );
    assert_eq!(rig.applet(), before);
    assert_eq!(rig.kernel.is_signalled(event), Some(true));
}

// ----- ICommonStateGetter: ReceiveMessage and the messages -----------------------------------

/// `am_types.h`'s `enum class AppletMessage : u32` @ cf654052, every value but `None = 0`, in
/// declaration order: the reference's list, written out here to check [`AppletMessage::raw`].
const EDEN_MESSAGE_WORDS: [u32; 31] = [
    1, 2, 4, 6, 15, 16, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 50, 51, 55, 56, 57,
    60, 90, 91, 92, 93, 95,
];

/// The reply to a `ReceiveMessage` that delivers `message`.
fn message_reply(message: AppletMessage) -> Vec<u8> {
    domain_reply(ResultCode::SUCCESS, &message.raw().to_le_bytes(), &[])
}

impl Rig {
    /// A rig whose server starts from `applet` (made with [`oe_server_for`]).
    fn with_applet(applet: Applet) -> Self {
        let mut rig = Self::new();
        rig.server = oe_server_for(applet, UnknownPolicy::Strict);
        rig.server.accept(rig.session).unwrap();
        rig
    }

    /// Opens the proxy and the common state getter, and takes the message event; the proxy's
    /// and the getter's object ids, and the event.
    fn getter_with_event(&mut self) -> (u32, u32, ObjectId) {
        let proxy = self.proxy();
        let getter = self.open(proxy, GET_COMMON_STATE_GETTER);
        let copied = self.object(getter, GET_EVENT_HANDLE, &[]).copies[0];
        (proxy, getter, self.object_of(copied))
    }
}

/// An application's applet with every kept message term pending: the focus change (Eden's
/// initial `m_has_focus_state_changed`), an operation-mode change, a performance-mode change,
/// and `queued` in the unordered queue.
fn applet_with_every_message(queued: &[AppletMessage]) -> Applet {
    let mut applet = Applet::new(AppletSettings::eden_default());
    applet.lifecycle.has_operation_mode_changed = true;
    applet.lifecycle.has_performance_mode_changed = true;
    applet.lifecycle.unordered_messages = queued.iter().copied().collect();
    applet
}

/// The issue's acceptance: the first `ReceiveMessage` of a fresh application answers the focus
/// change; the event, signalled at launch, is then clear; the next answers `ResultNoMessages`
/// with nothing, and the event stays clear.
#[test]
fn receive_message_answers_the_focus_change_then_no_messages() {
    let mut rig = Rig::new();
    let (_, getter, event) = rig.getter_with_event();
    assert_eq!(rig.kernel.is_signalled(event), Some(true));

    let first = rig.object_raw(getter, RECEIVE_MESSAGE, &[]);
    assert_eq!(first, message_reply(AppletMessage::FocusStateChanged));
    // The exact words after the header and the domain header: `SFCO`, version, success, token,
    // then the message, 15.
    let words: Vec<u32> = first[32..52]
        .chunks(4)
        .map(|word| u32::from_le_bytes(word.try_into().unwrap()))
        .collect();
    assert_eq!(words, [SFCO_MAGIC, 0, 0, 0, 15]);
    assert!(!rig.applet().lifecycle().has_focus_state_changed);
    assert_eq!(rig.kernel.is_signalled(event), Some(false));
    assert_eq!(rig.kernel.signals(event), Some(1));

    for _ in 0..2 {
        let none = rig.object_raw(getter, RECEIVE_MESSAGE, &[]);
        assert_eq!(none, domain_reply(NO_MESSAGES, &[], &[]));
        assert_eq!(rig.kernel.is_signalled(event), Some(false));
        assert_eq!(rig.kernel.signals(event), Some(1));
    }
}

#[test]
fn no_messages_is_edens_result_and_libnx_s_word() {
    assert_eq!(ResultCode::new(AM_MODULE, 3), Some(NO_MESSAGES));
    // libnx's `appletMainLoop`: `if (R_VALUE(rc) == 0x680) continue;`.
    assert_eq!(NO_MESSAGES.raw(), 0x680);
    assert_eq!(NO_MESSAGES.module(), AM_MODULE);
    assert_eq!(NO_MESSAGES.description(), 3);
}

/// Before the event exists, a `ReceiveMessage` only takes the message; the event made after is
/// made clear.
#[test]
fn a_message_received_before_the_event_exists_leaves_it_clear() {
    let mut rig = Rig::new();
    let proxy = rig.proxy();
    let getter = rig.open(proxy, GET_COMMON_STATE_GETTER);
    let first = rig.object_raw(getter, RECEIVE_MESSAGE, &[]);
    assert_eq!(first, message_reply(AppletMessage::FocusStateChanged));
    assert_eq!(rig.applet().message_event(), None);
    let copied = rig.object(getter, GET_EVENT_HANDLE, &[]).copies[0];
    let event = rig.object_of(copied);
    assert_eq!(rig.kernel.signals(event), Some(0));
    assert_eq!(rig.kernel.is_signalled(event), Some(false));
}

/// libnx's start-up turns focus notifications off (`SetFocusHandlingMode(0, 0, 1)`): the focus
/// change stays pending but is not a message, so `ReceiveMessage` answers `ResultNoMessages` and
/// changes nothing. Turned on again, the pending change is delivered.
#[test]
fn a_focus_change_is_not_received_while_its_notifications_are_off() {
    let mut rig = Rig::new();
    let (proxy, getter, event) = rig.getter_with_event();
    let controller = rig.open(proxy, GET_SELF_CONTROLLER);
    rig.object(controller, SET_FOCUS_HANDLING_MODE, &[0, 0, 1]);
    let before = rig.applet();
    assert!(before.lifecycle().has_focus_state_changed);

    let none = rig.object_raw(getter, RECEIVE_MESSAGE, &[]);
    assert_eq!(none, domain_reply(NO_MESSAGES, &[], &[]));
    assert_eq!(rig.applet(), before);
    assert_eq!(rig.kernel.is_signalled(event), Some(false));

    rig.object(controller, SET_FOCUS_HANDLING_MODE, &[1, 0, 1]);
    assert_eq!(rig.kernel.is_signalled(event), Some(true));
    let first = rig.object_raw(getter, RECEIVE_MESSAGE, &[]);
    assert_eq!(first, message_reply(AppletMessage::FocusStateChanged));
    assert_eq!(rig.kernel.is_signalled(event), Some(false));
}

/// The priority order, through the server: with every kept term pending and two messages
/// queued, `ReceiveMessage` answers the focus change, the operation-mode change, the
/// performance-mode change, then the queue oldest first, and only then `ResultNoMessages`. The
/// event stays signalled until the last message is taken, and is cleared exactly then.
#[test]
fn receive_message_answers_in_edens_order_of_priority() {
    let queued = [
        AppletMessage::StartupLogoDisappeared,
        AppletMessage::DetectShortPressingHomeButton,
    ];
    let mut rig = Rig::with_applet(applet_with_every_message(&queued));
    let (_, getter, event) = rig.getter_with_event();
    let expected = [
        AppletMessage::FocusStateChanged,
        AppletMessage::OperationModeChanged,
        AppletMessage::PerformanceModeChanged,
        queued[0],
        queued[1],
    ];
    for (index, message) in expected.into_iter().enumerate() {
        assert_eq!(rig.kernel.is_signalled(event), Some(true), "before {index}");
        let reply = rig.object_raw(getter, RECEIVE_MESSAGE, &[]);
        assert_eq!(reply, message_reply(message), "message {index}");
    }
    assert_eq!(rig.kernel.is_signalled(event), Some(false));
    assert_eq!(rig.kernel.signals(event), Some(1));
    let none = rig.object_raw(getter, RECEIVE_MESSAGE, &[]);
    assert_eq!(none, domain_reply(NO_MESSAGES, &[], &[]));
    assert_eq!(
        rig.applet().lifecycle(),
        &Lifecycle {
            has_focus_state_changed: false,
            ..Lifecycle::default()
        }
    );
}

/// Which message source is pending, in Eden's order of priority.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Source {
    Focus,
    OperationMode,
    PerformanceMode,
    Queue,
}

/// The pop order checked against a model: for every set of pending sources, with focus
/// notifications on and off, the messages come in [`Source`]'s order, the focus change only
/// with its notifications on, the queue oldest first; then `None`, and again `None`. At every
/// step `message_available` holds exactly when the next pop returns a message.
#[test]
fn the_pop_order_is_edens_for_every_set_of_pending_messages() {
    const ORDER: [Source; 4] = [
        Source::Focus,
        Source::OperationMode,
        Source::PerformanceMode,
        Source::Queue,
    ];
    let queued = [
        AppletMessage::DetectLongPressingCaptureButton,
        AppletMessage::ChangeIntoForeground,
    ];
    for set in 0_u32..16 {
        for notify in [false, true] {
            let pending = |source: Source| {
                let index = ORDER.iter().position(|s| *s == source).unwrap();
                set & (1 << index) != 0
            };
            let mut lifecycle = Lifecycle {
                notifications: Notifications {
                    focus_state_changed: notify,
                    ..Notifications::default()
                },
                has_focus_state_changed: pending(Source::Focus),
                has_operation_mode_changed: pending(Source::OperationMode),
                has_performance_mode_changed: pending(Source::PerformanceMode),
                unordered_messages: if pending(Source::Queue) {
                    queued.into_iter().collect()
                } else {
                    VecDeque::new()
                },
                ..Lifecycle::default()
            };
            let mut expected = Vec::new();
            for source in ORDER.into_iter().filter(|s| pending(*s)) {
                match source {
                    Source::Focus if notify => expected.push(AppletMessage::FocusStateChanged),
                    Source::Focus => {}
                    Source::OperationMode => expected.push(AppletMessage::OperationModeChanged),
                    Source::PerformanceMode => {
                        expected.push(AppletMessage::PerformanceModeChanged);
                    }
                    Source::Queue => expected.extend(queued),
                }
            }
            let mut popped = Vec::new();
            loop {
                let available = lifecycle.message_available();
                let message = lifecycle.pop_message_in_order_of_priority();
                assert_eq!(
                    available,
                    message.is_some(),
                    "set {set:#06b}, notify {notify}"
                );
                let Some(message) = message else { break };
                popped.push(message);
            }
            assert_eq!(popped, expected, "set {set:#06b}, notify {notify}");
            assert_eq!(lifecycle.pop_message_in_order_of_priority(), None);
            // A focus change pending with notifications off is kept for later.
            assert_eq!(
                lifecycle.has_focus_state_changed,
                pending(Source::Focus) && !notify,
                "set {set:#06b}, notify {notify}"
            );
        }
    }
}

#[test]
fn applet_messages_are_edens_words() {
    let words: Vec<u32> = AppletMessage::ALL.iter().map(|m| m.raw()).collect();
    assert_eq!(words, EDEN_MESSAGE_WORDS);
    for message in AppletMessage::ALL {
        assert_eq!(AppletMessage::from_raw(message.raw()), Ok(message));
        assert_eq!(AppletMessage::try_from(message.raw()), Ok(message));
        assert_eq!(u32::from(message), message.raw());
    }
    // Each word, and every neighbour that is not a word, both sides; `None` (0) is not a
    // message; a word with bits set above its low byte is not one either.
    for raw in (0..=0x1FF).chain([u32::MAX, 15 | 0x100, 15 | 0x1_0000, 15 | 0x100_0000]) {
        let expected = if EDEN_MESSAGE_WORDS.contains(&raw) {
            Ok(raw)
        } else {
            Err(NotAnAppletMessage(raw))
        };
        assert_eq!(
            AppletMessage::from_raw(raw).map(AppletMessage::raw),
            expected,
            "{raw}"
        );
    }
    assert_eq!(
        NotAnAppletMessage(3).to_string(),
        "3 is not an applet message"
    );
}

/// A message is saved as its word: `postcard`'s `u32`, so each message's bytes are its word's.
/// Read back, every word Eden declares is that message, and any other word is refused.
#[test]
fn a_message_is_saved_as_its_word_and_read_back_through_from_raw() {
    for message in AppletMessage::ALL {
        let bytes = postcard::to_allocvec(&message).unwrap();
        assert_eq!(bytes, postcard::to_allocvec(&message.raw()).unwrap());
    }
    // One message's bytes, golden.
    assert_eq!(
        postcard::to_allocvec(&AppletMessage::StartupLogoDisappeared).unwrap(),
        [95]
    );
    for raw in (0..=0x1FF).chain([u32::MAX, 15 | 0x100]) {
        let bytes = postcard::to_allocvec(&raw).unwrap();
        let read = postcard::from_bytes::<AppletMessage>(&bytes);
        match AppletMessage::from_raw(raw) {
            Ok(message) => assert_eq!(read, Ok(message), "{raw}"),
            Err(_) => assert_eq!(read, Err(postcard::Error::SerdeDeCustom), "{raw}"),
        }
    }
}

/// The lifecycle's saved layout, golden: the fields in declaration order, each with a value
/// that differs from its neighbours where the type allows.
#[test]
fn the_lifecycle_is_saved_in_this_layout() {
    let lifecycle = Lifecycle {
        notifications: Notifications {
            focus_state_changed: true,
            operation_mode_changed: false,
            performance_mode_changed: true,
        },
        has_focus_state_changed: false,
        focus_handling_mode: FocusHandlingMode::NoSuspend,
        has_operation_mode_changed: true,
        has_performance_mode_changed: false,
        unordered_messages: [
            AppletMessage::DetectShortPressingHomeButton,
            AppletMessage::StartupLogoDisappeared,
            AppletMessage::ApplicationExited,
        ]
        .into_iter()
        .collect(),
    };
    let bytes = postcard::to_allocvec(&lifecycle).unwrap();
    assert_eq!(bytes, [1, 0, 1, 0, 2, 1, 0, 3, 20, 95, 6]);
    assert_eq!(postcard::from_bytes::<Lifecycle>(&bytes), Ok(lifecycle));
}

/// A server with messages pending saves, restores and saves again to the same bytes, and the
/// restored server delivers them in the same order, the queue after the flags. The queue holds
/// a message Eden raises only through its flags: it is delivered as pushed.
#[test]
fn a_restored_queue_is_delivered_in_order_after_the_flags() {
    let queued = [
        AppletMessage::FocusStateChanged,
        AppletMessage::AlbumRecordingSaved,
    ];
    let mut rig = Rig::with_applet(applet_with_every_message(&queued));
    let (_, getter, event) = rig.getter_with_event();
    // Take the focus change first, so the restored state is part-way through.
    let first = rig.object_raw(getter, RECEIVE_MESSAGE, &[]);
    assert_eq!(first, message_reply(AppletMessage::FocusStateChanged));
    let saved = rig.server.save().unwrap();

    let mut restored = oe_server(AppletSettings::eden_default(), UnknownPolicy::Strict);
    restored.restore_beside(&saved, &rig.kernel).unwrap();
    assert_eq!(restored.family::<Applet>(), Some(&rig.applet()));
    assert_eq!(restored.save().unwrap(), saved);
    rig.server = restored;

    let expected = [
        AppletMessage::OperationModeChanged,
        AppletMessage::PerformanceModeChanged,
        queued[0],
        queued[1],
    ];
    for (index, message) in expected.into_iter().enumerate() {
        assert_eq!(rig.kernel.is_signalled(event), Some(true), "before {index}");
        let reply = rig.object_raw(getter, RECEIVE_MESSAGE, &[]);
        assert_eq!(reply, message_reply(message), "message {index}");
    }
    assert_eq!(rig.kernel.is_signalled(event), Some(false));
    let none = rig.object_raw(getter, RECEIVE_MESSAGE, &[]);
    assert_eq!(none, domain_reply(NO_MESSAGES, &[], &[]));
}

/// A snapshot whose queue holds a word that is not a message is refused with the exact error,
/// and the server is left as it was; the declared neighbours of each refused word restore.
#[test]
fn a_snapshot_with_an_invalid_queued_message_is_refused() {
    // Two snapshots that differ only in the queued message's one byte, to find where it is.
    let saved_with = |message: AppletMessage| {
        let mut rig = Rig::with_applet(applet_with_every_message(&[message]));
        rig.proxy();
        rig.server.save().unwrap()
    };
    let saved = saved_with(AppletMessage::StartupLogoDisappeared);
    let other = saved_with(AppletMessage::AlbumScreenShotTaken);
    assert_eq!(saved.len(), other.len());
    let differing: Vec<usize> = (0..saved.len()).filter(|&i| saved[i] != other[i]).collect();
    assert_eq!(differing.len(), 1);
    let at = differing[0];
    assert_eq!(
        u32::from(saved[at]),
        AppletMessage::StartupLogoDisappeared.raw()
    );

    let restore = |word: u8| {
        let mut bytes = saved.clone();
        bytes[at] = word;
        let mut server = oe_server(AppletSettings::eden_default(), UnknownPolicy::Strict);
        let result = server.restore_beside(&bytes, &FakeKernel::new(0).unwrap());
        (result, server)
    };
    let refused = SnapshotError::Decode(postcard::Error::SerdeDeCustom);
    // `None`, and a word on each side of declared words; the last single-byte word.
    for word in [0_u8, 3, 5, 7, 14, 17, 34, 49, 94, 96, 0x7F] {
        let (result, server) = restore(word);
        assert_eq!(result, Err(refused.clone()), "{word}");
        assert_eq!(server.session_count(), 0, "{word}");
        assert_eq!(
            server.family::<Applet>(),
            Some(&Applet::new(AppletSettings::eden_default())),
            "{word}"
        );
    }
    assert_eq!(
        refused.to_string(),
        "could not decode the snapshot: Serde Deserialization Error"
    );
    // Their declared neighbours restore, and are the queued message.
    for word in [1_u8, 2, 4, 6, 15, 16, 33, 50, 93, 95] {
        let (result, server) = restore(word);
        assert_eq!(result, Ok(()), "{word}");
        let queue = &server
            .family::<Applet>()
            .unwrap()
            .lifecycle()
            .unordered_messages;
        let words: Vec<u32> = queue.iter().map(|m| m.raw()).collect();
        assert_eq!(words, [u32::from(word)]);
    }
}

/// Divergence from Eden, whose clear cannot fail: restored onto a kernel that did not make the
/// message event, `ReceiveMessage` of the last message fails the clear with the kernel's exact
/// error and writes nothing back, so the message is still there to be received.
#[test]
fn a_failed_clear_leaves_the_message_to_be_received_again() {
    let mut rig = Rig::new();
    let (_, getter, event) = rig.getter_with_event();
    let saved = rig.server.save().unwrap();

    let mut fresh = Rig::new();
    assert_eq!(fresh.kernel.is_signalled(event), None);
    // Restored beside the kernel that made the event, so the restore's check passes; the
    // commands then run on the fresh kernel, which did not make it.
    fresh.server.restore_beside(&saved, &rig.kernel).unwrap();
    let before = fresh.applet();
    assert!(before.lifecycle().has_focus_state_changed);
    for _ in 0..2 {
        let reply = fresh.object(getter, RECEIVE_MESSAGE, &[]);
        assert_eq!(reply.result, KernelError::InvalidHandle.code());
        assert_eq!(reply.data, Vec::<u8>::new());
        assert_eq!(fresh.applet(), before);
    }

    // With a message left after it nothing is cleared, so the same restore delivers one.
    let mut rig = Rig::with_applet(applet_with_every_message(&[]));
    let (_, getter, _) = rig.getter_with_event();
    let saved = rig.server.save().unwrap();
    let mut fresh = Rig::new();
    // Restored beside the kernel that made the event, so the restore's check passes; the
    // commands then run on the fresh kernel, which did not make it.
    fresh.server.restore_beside(&saved, &rig.kernel).unwrap();
    let first = fresh.object_raw(getter, RECEIVE_MESSAGE, &[]);
    assert_eq!(first, message_reply(AppletMessage::FocusStateChanged));
}

/// Eden's `SignalSystemEventIfNeeded` clears the event only when its cached "a message is
/// available" was true. Restored onto a kernel that did not make the message event, with no
/// message pending, `ReceiveMessage` and a lifecycle command that leaves none pending touch no
/// event: they answer `ResultNoMessages` and success, not the kernel's `InvalidHandle`.
#[test]
fn with_no_message_pending_nothing_clears_the_message_event() {
    let mut rig = Rig::new();
    let (proxy, getter, event) = rig.getter_with_event();
    let controller = rig.open(proxy, GET_SELF_CONTROLLER);
    let first = rig.object_raw(getter, RECEIVE_MESSAGE, &[]);
    assert_eq!(first, message_reply(AppletMessage::FocusStateChanged));
    let saved = rig.server.save().unwrap();

    let mut fresh = Rig::new();
    assert_eq!(fresh.kernel.is_signalled(event), None);
    // Restored beside the kernel that made the event, so the restore's check passes; the
    // commands then run on the fresh kernel, which did not make it.
    fresh.server.restore_beside(&saved, &rig.kernel).unwrap();
    assert!(!fresh.applet().lifecycle().message_available());
    for _ in 0..2 {
        let none = fresh.object_raw(getter, RECEIVE_MESSAGE, &[]);
        assert_eq!(none, domain_reply(NO_MESSAGES, &[], &[]));
    }
    let on = fresh.object(controller, SET_FOCUS_HANDLING_MODE, &[1, 0, 1]);
    assert_eq!(on.result, ResultCode::SUCCESS);
    assert!(!fresh.applet().lifecycle().message_available());
}

/// The other side of [`with_no_message_pending_nothing_clears_the_message_event`]: Eden signals
/// only when its cached "a message is available" was false. Restored onto a kernel that did not
/// make the message event, with the focus change still pending, a lifecycle command that keeps
/// it pending touches no event and succeeds.
#[test]
fn with_a_message_still_pending_nothing_signals_the_message_event() {
    let mut rig = Rig::new();
    let (proxy, _, event) = rig.getter_with_event();
    let controller = rig.open(proxy, GET_SELF_CONTROLLER);
    assert!(rig.applet().lifecycle().message_available());
    let saved = rig.server.save().unwrap();

    let mut fresh = Rig::new();
    assert_eq!(fresh.kernel.is_signalled(event), None);
    // Restored beside the kernel that made the event, so the restore's check passes; the
    // commands then run on the fresh kernel, which did not make it.
    fresh.server.restore_beside(&saved, &rig.kernel).unwrap();
    let on = fresh.object(controller, SET_FOCUS_HANDLING_MODE, &[1, 0, 1]);
    assert_eq!(on.result, ResultCode::SUCCESS);
    assert!(fresh.applet().lifecycle().message_available());
}

/// The unordered messages queued in [`per_call_work_does_not_grow_with_the_queue`]: a queue a
/// restored snapshot may hold, long enough that touching every message on each call costs far
/// more than [`QUEUE_PATIENCE`] over [`QUEUE_ROUNDS`].
const LONG_QUEUE: usize = 1 << 25;

/// How many times [`per_call_work_does_not_grow_with_the_queue`] sends each command. Each round
/// copies three event handles, so the rounds stay well inside the fake kernel's handle table.
const QUEUE_ROUNDS: usize = 200;

/// The wall-clock bound on [`per_call_work_does_not_grow_with_the_queue`]'s rounds.
const QUEUE_PATIENCE: std::time::Duration = std::time::Duration::from_secs(20);

/// Every command that reads or changes the applet works on it in place, not on a copy: with a
/// queue of [`LONG_QUEUE`] messages, [`QUEUE_ROUNDS`] rounds of `ReceiveMessage`, the three
/// event getters and `SetFocusHandlingMode` finish inside [`QUEUE_PATIENCE`].
///
/// Measured in the gate's debug build, all the rounds take about a fifth of a second alone and
/// up to two and a half seconds under the full parallel gate. A command that copies the applet, queue and all,
/// on each call (as `ReceiveMessage`, the `ISelfController` lifecycle commands and the event
/// getters once did, which made draining a restored queue quadratic) spends a quarter to half a
/// second per call on it: adding the copy back to one of them at a time crossed the bound in
/// round 78 (`SetFocusHandlingMode`), 46 (`ReceiveMessage`) and 13 (the event getters, three
/// per round). The bound is checked after every command, so a regression fails at the bound
/// instead of running on. A wall-clock bound with this margin (about eight times the gate's
/// time, under half a regression's) is the only observation of the cost short of counting
/// allocations.
#[test]
fn per_call_work_does_not_grow_with_the_queue() {
    let mut applet = Applet::new(AppletSettings::eden_default());
    applet.lifecycle.unordered_messages =
        std::iter::repeat_n(AppletMessage::AlbumRecordingSaved, LONG_QUEUE).collect();
    let mut rig = Rig::with_applet(applet);
    let (proxy, getter, _) = rig.getter_with_event();
    let controller = rig.open(proxy, GET_SELF_CONTROLLER);
    let functions = rig.open(proxy, GET_APPLICATION_FUNCTIONS);
    // The focus change comes first.
    let first = rig.object_raw(getter, RECEIVE_MESSAGE, &[]);
    assert_eq!(first, message_reply(AppletMessage::FocusStateChanged));

    let deadline = std::time::Instant::now() + QUEUE_PATIENCE;
    let in_time = |what: &str, round: usize| {
        assert!(
            std::time::Instant::now() < deadline,
            "{what} in round {round} ran past the bound"
        );
    };
    for round in 0..QUEUE_ROUNDS {
        let reply = rig.object_raw(getter, RECEIVE_MESSAGE, &[]);
        assert_eq!(reply, message_reply(AppletMessage::AlbumRecordingSaved));
        in_time("ReceiveMessage", round);
        let mode = rig.object(controller, SET_FOCUS_HANDLING_MODE, &[1, 0, 1]);
        assert_eq!(mode.result, ResultCode::SUCCESS);
        in_time("SetFocusHandlingMode", round);
        for (object, command) in [
            (getter, GET_EVENT_HANDLE),
            (controller, GET_ACCUMULATED_SUSPENDED_TICK_CHANGED_EVENT),
            (functions, GET_GPU_ERROR_DETECTED_SYSTEM_EVENT),
        ] {
            let reply = rig.object(object, command, &[]);
            assert_eq!(reply.result, ResultCode::SUCCESS);
            assert_eq!(reply.copies.len(), 1);
            in_time("an event getter", round);
        }
    }
    let applet = rig.server.family::<Applet>().unwrap();
    assert_eq!(
        applet.lifecycle().unordered_messages.len(),
        LONG_QUEUE - QUEUE_ROUNDS
    );
    assert!(applet.lifecycle().message_available());
}

/// [`Lifecycle::message_available_after_pop`], which reads at most the first two queued
/// messages, agrees with a pop on a full copy for every set of pending flags, with notifications
/// on and off, and every queue length up to two past where it stops reading, at every step of a
/// drain.
#[test]
fn message_available_after_pop_agrees_with_a_pop() {
    for flags in 0_u32..8 {
        for notify in [false, true] {
            for queued in 0..5 {
                let mut lifecycle = Lifecycle {
                    notifications: Notifications {
                        focus_state_changed: notify,
                        ..Notifications::default()
                    },
                    has_focus_state_changed: flags & 1 != 0,
                    has_operation_mode_changed: flags & 2 != 0,
                    has_performance_mode_changed: flags & 4 != 0,
                    unordered_messages: AppletMessage::ALL.into_iter().take(queued).collect(),
                    ..Lifecycle::default()
                };
                loop {
                    let mut popped = lifecycle.clone();
                    let message = popped.pop_message_in_order_of_priority();
                    assert_eq!(
                        lifecycle.message_available_after_pop(),
                        popped.message_available(),
                        "flags {flags:#05b}, notify {notify}, {queued} queued, at {lifecycle:?}"
                    );
                    lifecycle = popped;
                    if message.is_none() {
                        break;
                    }
                }
            }
        }
    }
}

// ----- The title's later start-up commands: 21, 210, 66, 14 and 2400 -------------------------

/// `IApplicationFunctions` 21 answers Eden's language for an application whose metadata Eden
/// cannot read, under Eden's default system language: `en-US`, as `set`'s language code writes
/// it (a string, zero-padded to eight bytes). Nothing in the applet changes, and a second call
/// answers the same.
#[test]
fn get_desired_language_answers_eden_s_answer_with_no_metadata() {
    assert_eq!(LANGUAGE_CODE_EN_US.to_le_bytes(), *b"en-US\0\0\0");
    assert_eq!(DESIRED_LANGUAGE_CODE, LANGUAGE_CODE_EN_US);
    let mut rig = Rig::new();
    let proxy = rig.proxy();
    let functions = rig.open(proxy, GET_APPLICATION_FUNCTIONS);
    let before = rig.applet();
    for _ in 0..2 {
        assert_eq!(
            rig.object_raw(functions, GET_DESIRED_LANGUAGE, &[]),
            domain_reply(ResultCode::SUCCESS, b"en-US\0\0\0", &[])
        );
        assert_eq!(rig.applet(), before);
    }
}

/// `IApplicationFunctions` 210 copies one event, made the first time and never signalled: the
/// same event on every call, and none of the applet's other events.
#[test]
fn get_unknown_event_210_copies_one_event_never_signalled() {
    let mut rig = Rig::new();
    let proxy = rig.proxy();
    let functions = rig.open(proxy, GET_APPLICATION_FUNCTIONS);
    let getter = rig.open(proxy, GET_COMMON_STATE_GETTER);
    let controller = rig.open(proxy, GET_SELF_CONTROLLER);
    assert_eq!(rig.applet().unknown_event(), None);
    let raw = rig.object_raw(functions, GET_UNKNOWN_EVENT_210, &[]);
    let first = parse(&raw, SessionKind::Domain).copies[0];
    assert_eq!(raw, copied_handle_reply(first));
    let event = rig.object_of(first);
    assert_eq!(rig.applet().unknown_event(), Some(event));
    assert_eq!(rig.kernel.signals(event), Some(0));
    assert_eq!(rig.kernel.is_signalled(event), Some(false));

    let again = rig.object_raw(functions, GET_UNKNOWN_EVENT_210, &[]);
    let second = parse(&again, SessionKind::Domain).copies[0];
    assert_eq!(again, copied_handle_reply(second));
    assert_eq!(rig.object_of(second), event);

    let others = [
        rig.object(getter, GET_EVENT_HANDLE, &[]).copies[0],
        rig.object(
            controller,
            GET_ACCUMULATED_SUSPENDED_TICK_CHANGED_EVENT,
            &[],
        )
        .copies[0],
        rig.object(functions, GET_GPU_ERROR_DETECTED_SYSTEM_EVENT, &[])
            .copies[0],
    ];
    for other in others {
        assert_ne!(rig.object_of(other), event);
    }
    assert_eq!(rig.applet().unknown_event(), Some(event));
    assert_eq!(rig.kernel.signals(event), Some(0));
}

/// Eden's `CpuBoostMode` words, checked against libnx's `ApmCpuBoostMode` (`Normal = 0`,
/// `FastLoad = 1`, `Type2 = 2`), and Eden's check: a word is a mode exactly when it is below the
/// number of modes, so the last mode is accepted and the next word refused, and a word that
/// differs from a mode only above its low byte is refused.
#[test]
fn cpu_boost_mode_accepts_eden_s_three_words_and_no_other() {
    assert_eq!(
        [
            CpuBoostMode::Normal,
            CpuBoostMode::FastLoad,
            CpuBoostMode::Partial
        ]
        .map(CpuBoostMode::raw),
        [0, 1, 2]
    );
    let modes = u32::try_from(CpuBoostMode::ALL.len()).unwrap();
    let last = CpuBoostMode::ALL[CpuBoostMode::ALL.len() - 1];
    assert_eq!(CpuBoostMode::from_raw(modes - 1), Some(last));
    assert_eq!(CpuBoostMode::from_raw(modes), None);
    for mode in CpuBoostMode::ALL {
        assert_eq!(CpuBoostMode::from_raw(mode.raw()), Some(mode));
        for above in [0x100, 0x1_0000, 0x100_0000] {
            assert_eq!(CpuBoostMode::from_raw(mode.raw() | above), None, "{mode:?}");
        }
    }
    for raw in (modes..=0x1FF).chain([u32::MAX]) {
        assert_eq!(CpuBoostMode::from_raw(raw), None, "{raw}");
    }
}

/// The ids of the five later start-up commands are the reference's rows, written out here, since
/// every other test sends the shipping constant and so passes whatever its value: Eden @
/// cf654052, `src/core/hle/service/am/service/application_functions.cpp`
/// (`{21, .., "GetDesiredLanguage"}`, `{210, .., "Unknown210"}`), `common_state_getter.cpp`
/// (`{66, .., "SetCpuBoostMode"}`) and `self_controller.cpp` (`{14, ..,
/// "SetRestartMessageEnabled"}`); PR #4588 @ e5fb3b97,
/// `eden-pr4588/src/core/hle/service/am/service/self_controller.cpp` (`{2400, ..,
/// "SetRtcScreenSharingAudioEnabled"}`).
#[test]
fn the_later_start_up_command_ids_are_the_reference_s() {
    let rows = [
        ("GetDesiredLanguage", GET_DESIRED_LANGUAGE, 21),
        ("GetUnknownEvent210", GET_UNKNOWN_EVENT_210, 210),
        ("SetCpuBoostMode", SET_CPU_BOOST_MODE, 66),
        ("SetRestartMessageEnabled", SET_RESTART_MESSAGE_ENABLED, 14),
        (
            "SetRtcScreenSharingAudioEnabled",
            SET_RTC_SCREEN_SHARING_AUDIO_ENABLED,
            2400,
        ),
    ];
    for (name, shipping, reference) in rows {
        assert_eq!(shipping, reference, "{name}");
    }
}

/// `ICommonStateGetter` 66 answers success and nothing else for every word, a mode or not, as
/// Eden's does, and keeps nothing: Eden's performance configuration for the mode is not kept
/// here (module docs, "The title's later start-up commands").
#[test]
fn set_cpu_boost_mode_answers_success_and_keeps_nothing() {
    let mut rig = Rig::new();
    let proxy = rig.proxy();
    let getter = rig.open(proxy, GET_COMMON_STATE_GETTER);
    let before = rig.applet();
    let modes = u32::try_from(CpuBoostMode::ALL.len()).unwrap();
    let words = CpuBoostMode::ALL.map(CpuBoostMode::raw).into_iter().chain([
        modes,
        modes - 1 + 0x100,
        u32::MAX,
    ]);
    for word in words {
        assert_eq!(
            rig.object_raw(getter, SET_CPU_BOOST_MODE, &word.to_le_bytes()),
            domain_reply(ResultCode::SUCCESS, &[], &[]),
            "{word}"
        );
        assert_eq!(rig.applet(), before, "{word}");
    }
    // The reported performance mode is the settings', boost or not.
    let reply = rig.object_raw(getter, GET_PERFORMANCE_MODE, &[]);
    let mode = AppletSettings::eden_default().performance_mode().raw();
    assert_eq!(
        reply,
        domain_reply(ResultCode::SUCCESS, &mode.to_le_bytes(), &[])
    );
}

/// `ISelfController` 14 keeps its flag in the applet, initially false (Eden's
/// `m_resume_notification_enabled{}`), reading any byte that is not zero as true. It never
/// touches the message event, signalled or not, and changes nothing else in the applet.
#[test]
fn set_restart_message_enabled_keeps_its_flag_and_leaves_the_message_event() {
    assert!(!Applet::new(AppletSettings::eden_default()).resume_notification_enabled());
    let mut rig = Rig::new();
    let proxy = rig.proxy();
    let getter = rig.open(proxy, GET_COMMON_STATE_GETTER);
    let controller = rig.open(proxy, GET_SELF_CONTROLLER);
    let copied = rig.object(getter, GET_EVENT_HANDLE, &[]).copies[0];
    let event = rig.object_of(copied);
    let before = rig.applet();
    assert!(!before.resume_notification_enabled());
    let resume = |rig: &Rig| rig.applet().resume_notification_enabled();
    // Signalled at launch; then with the event clear (the focus change received).
    for signalled in [true, false] {
        if !signalled {
            rig.object(getter, RECEIVE_MESSAGE, &[]);
        }
        assert_eq!(rig.kernel.is_signalled(event), Some(signalled));
        let signals = rig.kernel.signals(event);
        for (byte, expected) in [(1, true), (1, true), (0, false), (2, true), (0, false)] {
            assert_eq!(
                rig.object_raw(controller, SET_RESTART_MESSAGE_ENABLED, &[byte]),
                domain_reply(ResultCode::SUCCESS, &[], &[]),
                "{byte}"
            );
            assert_eq!(resume(&rig), expected, "{byte}");
            assert_eq!(rig.kernel.is_signalled(event), Some(signalled), "{byte}");
            assert_eq!(rig.kernel.signals(event), signals, "{byte}");
        }
    }
    rig.object(controller, SET_RESTART_MESSAGE_ENABLED, &[1]);
    let mut expected = before;
    expected.resume_notification_enabled = true;
    expected.lifecycle.has_focus_state_changed = false;
    assert_eq!(rig.applet(), expected);
}

/// `ISelfController` 2400 (Switch 2) answers success and nothing else for either flag, and keeps
/// nothing, as PR #4588's only logs it.
#[test]
fn set_rtc_screen_sharing_audio_enabled_answers_success_and_keeps_nothing() {
    let mut rig = Rig::new();
    let proxy = rig.proxy();
    let controller = rig.open(proxy, GET_SELF_CONTROLLER);
    let before = rig.applet();
    for byte in [1, 0, 2, 0xFF] {
        assert_eq!(
            rig.object_raw(controller, SET_RTC_SCREEN_SHARING_AUDIO_ENABLED, &[byte]),
            domain_reply(ResultCode::SUCCESS, &[], &[]),
            "{byte}"
        );
        assert_eq!(rig.applet(), before, "{byte}");
    }
}

/// The two new flag commands refuse a payload with no byte for the flag, and the boost-mode one
/// a payload of fewer than its word's four bytes (libnx's layout, whose domain `size` counts the
/// exact argument bytes), with [`INVALID_CMIF_REQUEST`], and change nothing; the exact argument
/// is accepted.
#[test]
fn the_later_commands_refuse_a_payload_too_short_and_change_nothing() {
    let mut rig = Rig::new();
    let proxy = rig.proxy();
    let getter = rig.open(proxy, GET_COMMON_STATE_GETTER);
    let controller = rig.open(proxy, GET_SELF_CONTROLLER);
    let word = CpuBoostMode::FastLoad.raw().to_le_bytes();
    let cases: [(u32, u32, &[u8]); 3] = [
        (getter, SET_CPU_BOOST_MODE, &word),
        (controller, SET_RESTART_MESSAGE_ENABLED, &[1]),
        (controller, SET_RTC_SCREEN_SHARING_AUDIO_ENABLED, &[1]),
    ];
    for (object, command, argument) in cases {
        for len in 0..=argument.len() {
            let before = rig.applet();
            let request = RequestBuilder::domain(CommandType::Request, object, command)
                .layout(RequestLayout::Libnx)
                .data(&argument[..len]);
            let reply = rig.send_raw(rig.session, &request);
            if len == argument.len() {
                assert_eq!(
                    reply,
                    domain_reply(ResultCode::SUCCESS, &[], &[]),
                    "{command}"
                );
            } else {
                assert_eq!(
                    reply,
                    domain_reply(INVALID_CMIF_REQUEST, &[], &[]),
                    "{command} {len}"
                );
                assert_eq!(rig.applet(), before, "{command} {len}");
            }
        }
    }
    assert!(rig.applet().resume_notification_enabled());
}

/// A server saved after the later start-up commands restores, and answers from what it saved:
/// the restart-message flag, and the unknown event, copied again unchanged.
#[test]
fn a_server_after_the_later_start_up_commands_saves_and_restores() {
    let mut rig = Rig::new();
    let proxy = rig.proxy();
    let controller = rig.open(proxy, GET_SELF_CONTROLLER);
    let functions = rig.open(proxy, GET_APPLICATION_FUNCTIONS);
    rig.object(controller, SET_RESTART_MESSAGE_ENABLED, &[1]);
    rig.object(functions, GET_UNKNOWN_EVENT_210, &[]);
    let event = rig.applet().unknown_event().unwrap();
    let saved = rig.server.save().unwrap();

    let mut restored = oe_server(AppletSettings::eden_default(), UnknownPolicy::Strict);
    restored.restore_beside(&saved, &rig.kernel).unwrap();
    assert_eq!(restored.family::<Applet>(), Some(&rig.applet()));
    assert_eq!(restored.save().unwrap(), saved);
    rig.server = restored;

    assert!(rig.applet().resume_notification_enabled());
    let raw = rig.object_raw(functions, GET_UNKNOWN_EVENT_210, &[]);
    let copied = parse(&raw, SessionKind::Domain).copies[0];
    assert_eq!(raw, copied_handle_reply(copied));
    assert_eq!(rig.object_of(copied), event);
    assert_eq!(rig.kernel.signals(event), Some(0));
    // The flag goes off again on the restored server.
    rig.object(controller, SET_RESTART_MESSAGE_ENABLED, &[0]);
    assert!(!rig.applet().resume_notification_enabled());
}

// ----- restoring the applet's events beside the kernel ---------------------------------------

/// A hook that does nothing, for a hooked event's writable end.
struct QuietHook;

impl starquake_kernel::ipc::EventHook for QuietHook {
    fn signalled(&self, _: ObjectId) {}
}

/// Another server's number, whose events are not `appletOE`'s.
const OTHER_SERVER: ServiceId = ServiceId(0x0777);

/// Each kept event's slot with the refusal a restore answers for an id in it that is not a live
/// readable event this server made, in the order of the applet's fields.
fn kept_event_refusals() -> [(SetKeptEvent, SnapshotError); 4] {
    let [message, tick, gpu, unknown] = kept_event_slots().map(|(_, set)| set);
    [
        (
            message,
            SnapshotError::FamilyObject(
                "its applet's message event is not a live readable event its server made",
            ),
        ),
        (
            tick,
            SnapshotError::FamilyObject(
                "its applet's suspended-tick event is not a live readable event its server made",
            ),
        ),
        (
            gpu,
            SnapshotError::FamilyObject(
                "its applet's GPU-error event is not a live readable event its server made",
            ),
        ),
        (
            unknown,
            SnapshotError::FamilyObject(
                "its applet's unknown event is not a live readable event its server made",
            ),
        ),
    ]
}

impl Rig {
    /// Has every command that makes one of the applet's four events run, so each slot keeps
    /// one.
    fn with_every_event(&mut self) {
        let proxy = self.proxy();
        let getter = self.open(proxy, GET_COMMON_STATE_GETTER);
        let controller = self.open(proxy, GET_SELF_CONTROLLER);
        let functions = self.open(proxy, GET_APPLICATION_FUNCTIONS);
        let steps = [
            (getter, GET_EVENT_HANDLE),
            (controller, GET_ACCUMULATED_SUSPENDED_TICK_CHANGED_EVENT),
            (functions, GET_GPU_ERROR_DETECTED_SYSTEM_EVENT),
            (functions, GET_UNKNOWN_EVENT_210),
        ];
        for (object, command) in steps {
            assert_eq!(
                self.object(object, command, &[]).result,
                ResultCode::SUCCESS
            );
        }
        let applet = self.applet();
        assert!(
            [
                applet.message_event(),
                applet.suspended_tick_event(),
                applet.gpu_error_detected_event(),
                applet.unknown_event()
            ]
            .iter()
            .all(Option::is_some)
        );
    }

    /// Ids that are not a readable event this server made, each made on the rig's kernel (whose
    /// current service is the server's), by what each is.
    fn not_the_servers(&mut self) -> [(&'static str, ObjectId); 4] {
        let nothing = self.kernel.new_object();
        self.kernel.set_current_service(OTHER_SERVER);
        let others = self.kernel.create_event().unwrap();
        self.kernel.set_current_service(ServiceId(0));
        let hooked = self
            .kernel
            .create_hooked_event(std::sync::Arc::new(QuietHook));
        let page = starquake_memory::PAGE_SIZE;
        let memory = self
            .kernel
            .create_shared_memory(page, starquake_kernel::memory::SvcPermission::Read);
        [
            ("an id that names nothing", nothing),
            ("another server's event", others),
            ("a writable end this server made", hooked.unwrap().0),
            ("a shared memory", memory.unwrap().0),
        ]
    }
}

/// Each of the four events is restored only as a live readable event this server made (module
/// docs, "The applet"): any other id in one slot, the other three the server's own, is refused
/// with that slot's exact error, the server is left as it was, and the next restore of the
/// server's own snapshot succeeds.
#[test]
fn each_kept_event_restores_only_as_a_live_readable_event_its_server_made() {
    let mut rig = Rig::new();
    rig.with_every_event();
    let live = rig.applet();
    let own = rig.server.save().unwrap();
    let fresh = Applet::new(AppletSettings::eden_default());
    for (what, bad) in rig.not_the_servers() {
        for (set, refused) in kept_event_refusals() {
            let mut forged = live.clone();
            set(&mut forged, Some(bad));
            let saved = oe_server_for(forged, UnknownPolicy::Strict).save().unwrap();
            let mut restored = oe_server(AppletSettings::eden_default(), UnknownPolicy::Strict);
            assert_eq!(
                restored.restore_beside(&saved, &rig.kernel),
                Err(refused.clone()),
                "{what}: {refused}"
            );
            assert_eq!(restored.family::<Applet>(), Some(&fresh));
            assert_eq!(restored.restore_beside(&own, &rig.kernel), Ok(()));
            assert_eq!(restored.family::<Applet>(), Some(&live));
        }
    }
}

/// The slots are checked in the order of the applet's fields: with two slots wrong, the earlier
/// one is reported, for every pair.
#[test]
fn the_earliest_wrong_event_slot_is_reported() {
    let mut rig = Rig::new();
    rig.with_every_event();
    let live = rig.applet();
    let [(_, nothing), (_, others), ..] = rig.not_the_servers();
    let slots = kept_event_refusals();
    for (index, (set_first, refused)) in slots.iter().enumerate() {
        for (set_second, _) in slots.iter().skip(index + 1) {
            let mut forged = live.clone();
            set_first(&mut forged, Some(nothing));
            set_second(&mut forged, Some(others));
            let saved = oe_server_for(forged, UnknownPolicy::Strict).save().unwrap();
            let mut restored = oe_server(AppletSettings::eden_default(), UnknownPolicy::Strict);
            assert_eq!(
                restored.restore_beside(&saved, &rig.kernel),
                Err(refused.clone())
            );
        }
    }
}

/// The events are checked under the number the server is known by: a server known by another
/// number refuses the events this one made, and takes the events made while the kernel's
/// current service was that number.
#[test]
fn the_events_are_checked_under_the_server_s_own_number() {
    let mut rig = Rig::new();
    rig.with_every_event();
    let saved = rig.server.save().unwrap();
    let elsewhere = || {
        oe_server(AppletSettings::eden_default(), UnknownPolicy::Strict)
            .with_service_id(OTHER_SERVER)
    };
    let [(_, message_refused), ..] = kept_event_refusals();
    assert_eq!(
        elsewhere().restore_beside(&saved, &rig.kernel),
        Err(message_refused)
    );

    let mut other = Rig::new();
    other.kernel.set_current_service(OTHER_SERVER);
    other.with_every_event();
    let saved = other.server.save().unwrap();
    assert_eq!(elsewhere().restore_beside(&saved, &other.kernel), Ok(()));
}

/// The graphics server's number is checked before the events: a snapshot with both wrong is
/// refused for the number, and the same snapshot with the server's own events for nothing.
#[test]
fn the_graphics_server_is_checked_before_the_events() {
    let mut rig = Rig::with_graphics();
    rig.with_every_event();
    let live = rig.applet();
    let [(_, nothing), ..] = rig.not_the_servers();
    let mut forged = live.clone();
    forged.message_event = Some(nothing);
    forged.graphics_server = None;
    let saved = oe_server_for(forged, UnknownPolicy::Strict).save().unwrap();
    let mut restored = oe_server_with(Some(GRAPHICS));
    assert_eq!(
        restored.restore_beside(&saved, &rig.kernel),
        Err(SnapshotError::FamilyWiring(
            "its applet names another graphics server"
        ))
    );
    let mut own_events = live;
    own_events.graphics_server = None;
    let saved = oe_server_for(own_events, UnknownPolicy::Strict)
        .save()
        .unwrap();
    assert_eq!(
        restored.restore_beside(&saved, &rig.kernel),
        Err(SnapshotError::FamilyWiring(
            "its applet names another graphics server"
        ))
    );
    assert_eq!(
        oe_server_with(None).restore_beside(&saved, &rig.kernel),
        Ok(())
    );
}

/// An `appletOE` server keeps kernel objects, so `restore` refuses its snapshots whatever they
/// hold, an applet with no events included, and leaves the server as it was.
#[test]
fn an_appletoe_server_is_restored_only_beside_the_kernel() {
    let mut rig = Rig::new();
    rig.with_every_event();
    let with_events = rig.server.save().unwrap();
    let without = Rig::new().server.save().unwrap();
    let fresh = Applet::new(AppletSettings::eden_default());
    for bytes in [with_events, without] {
        let mut restored = oe_server(AppletSettings::eden_default(), UnknownPolicy::Strict);
        assert_eq!(restored.restore(&bytes), Err(SnapshotError::NeedsKernel));
        assert_eq!(restored.family::<Applet>(), Some(&fresh));
        assert_eq!(restored.restore_beside(&bytes, &rig.kernel), Ok(()));
    }
}

// ----- The launch parameter: `IApplicationFunctions` 1 ---------------------------------------

/// [`NO_DATA_IN_CHANNEL`] is Eden's `ResultNoDataInChannel`, written out:
/// `eden/src/core/hle/service/am/am_results.h` @ cf654052, `constexpr Result
/// ResultNoDataInChannel{ErrorModule::AM, 2}` (the same line in Eden-NX2 @ 50fa86bb and PR #4588
/// @ e5fb3b97): module 128, description 2, so the word `128 | 2 << 9`.
#[test]
fn no_data_in_channel_is_edens_result() {
    assert_eq!(ResultCode::new(128, 2), Some(NO_DATA_IN_CHANNEL));
    assert_eq!(NO_DATA_IN_CHANNEL.raw(), 0x480);
    assert_eq!(NO_DATA_IN_CHANNEL.module(), AM_MODULE);
    assert_eq!(NO_DATA_IN_CHANNEL.description(), 2);
    assert_ne!(NO_DATA_IN_CHANNEL, NO_MESSAGES);
}

/// The id and the kinds are the reference's, written out, since every other test sends the
/// shipping constants and so passes whatever their values: Eden @ cf654052,
/// `src/core/hle/service/am/service/application_functions.cpp` (`{1,
/// D<&IApplicationFunctions::PopLaunchParameter>, "PopLaunchParameter"}`), and `am_types.h`
/// (`enum class LaunchParameterKind : u32 { UserChannel = 1, AccountPreselectedUser = 2 }`);
/// libnx's `AppletLaunchParameterKind` gives the two the same values.
#[test]
fn the_launch_parameter_command_and_kinds_are_the_reference_s() {
    assert_eq!(POP_LAUNCH_PARAMETER, 1);
    assert_eq!(LaunchParameterKind::UserChannel.raw(), 1);
    assert_eq!(LaunchParameterKind::AccountPreselectedUser.raw(), 2);
    assert_eq!(
        LaunchParameterKind::ALL,
        [
            LaunchParameterKind::UserChannel,
            LaunchParameterKind::AccountPreselectedUser
        ]
    );
}

/// Eden's channel for each word: `launch_parameter_kind == LaunchParameterKind::UserChannel ?
/// user_channel_launch_parameter : preselected_user_launch_parameter`, so the user channel for
/// exactly the word 1, and the preselected user's for every other word: 0, 2, libnx's `Unknown`
/// 3, words whose low byte is 1 but whose upper bytes are not zero, and the largest word.
#[test]
fn eden_pops_the_user_channel_for_kind_1_and_the_preselected_user_s_for_any_other_word() {
    assert_eq!(
        LaunchParameterKind::channel_for(1),
        LaunchParameterKind::UserChannel
    );
    for raw in [0, 2, 3, 0x101, 0x0100_0001, u32::MAX] {
        assert_eq!(
            LaunchParameterKind::channel_for(raw),
            LaunchParameterKind::AccountPreselectedUser,
            "{raw:#x}"
        );
    }
    for kind in LaunchParameterKind::ALL {
        assert_eq!(LaunchParameterKind::channel_for(kind.raw()), kind);
    }
}

/// The issue's acceptance: both kinds answer Eden's empty-channel result, with no object and no
/// data, as libnx sends the command (`appletPopLaunchParameter`: the kind as one `u32`, one out
/// object). A second call answers the same, and nothing in the applet changes.
#[test]
fn pop_launch_parameter_answers_no_data_in_channel_for_both_kinds() {
    let mut rig = Rig::new();
    let proxy = rig.proxy();
    let functions = rig.open(proxy, GET_APPLICATION_FUNCTIONS);
    let before = rig.applet();
    for kind in LaunchParameterKind::ALL {
        for _ in 0..2 {
            assert_eq!(
                rig.object_raw(functions, POP_LAUNCH_PARAMETER, &kind.raw().to_le_bytes()),
                domain_reply(NO_DATA_IN_CHANNEL, &[], &[]),
                "{kind:?}"
            );
            assert_eq!(rig.applet(), before, "{kind:?}");
        }
    }
}

/// A word that is not a kind answers as the preselected user's kind does, since Eden pops the
/// preselected user's channel for it: here, too, no data.
#[test]
fn pop_launch_parameter_answers_an_unknown_kind_as_the_preselected_user_s() {
    let mut rig = Rig::new();
    let proxy = rig.proxy();
    let functions = rig.open(proxy, GET_APPLICATION_FUNCTIONS);
    let preselected = rig.object_raw(
        functions,
        POP_LAUNCH_PARAMETER,
        &LaunchParameterKind::AccountPreselectedUser
            .raw()
            .to_le_bytes(),
    );
    assert_eq!(preselected, domain_reply(NO_DATA_IN_CHANNEL, &[], &[]));
    for raw in [0_u32, 3, 0x101, u32::MAX] {
        assert_eq!(
            LaunchParameterKind::channel_for(raw),
            LaunchParameterKind::AccountPreselectedUser,
            "{raw:#x}"
        );
        assert_eq!(
            rig.object_raw(functions, POP_LAUNCH_PARAMETER, &raw.to_le_bytes()),
            preselected,
            "{raw:#x}"
        );
    }
}

/// A payload of fewer than the kind's four bytes (libnx's layout, whose domain `size` counts the
/// exact argument bytes) is refused with [`INVALID_CMIF_REQUEST`] and changes nothing; the whole
/// word reaches the channel and answers no data.
#[test]
fn pop_launch_parameter_refuses_a_payload_too_short() {
    let mut rig = Rig::new();
    let proxy = rig.proxy();
    let functions = rig.open(proxy, GET_APPLICATION_FUNCTIONS);
    let word = LaunchParameterKind::UserChannel.raw().to_le_bytes();
    let before = rig.applet();
    for len in 0..=word.len() {
        let request = RequestBuilder::domain(CommandType::Request, functions, POP_LAUNCH_PARAMETER)
            .layout(RequestLayout::Libnx)
            .data(&word[..len]);
        let reply = rig.send_raw(rig.session, &request);
        let expected = if len == word.len() {
            NO_DATA_IN_CHANNEL
        } else {
            INVALID_CMIF_REQUEST
        };
        assert_eq!(reply, domain_reply(expected, &[], &[]), "{len}");
        assert_eq!(rig.applet(), before, "{len}");
    }
}
