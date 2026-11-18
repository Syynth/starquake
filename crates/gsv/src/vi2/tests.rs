//! The vi2 session driven through raw TIPC messages, over the services framework's fake
//! kernel, with gsv's shared state behind it and the timer thread's vsync turn played by the
//! test.
//!
//! Requests are laid out by hand, word by word, from the layouts PR checks
//! (`eden-pr4588/src/core/hle/service/am/service/graphics_supervisor.cpp` and
//! `graphics_supervisor_protocol.h` @ e5fb3b97): each handler compares the two header words
//! with literals (`OpenDisplay`: `packet[0] == 0x75 && packet[1] == 3`, the name at words 2–3,
//! the flags at 4; `DecodeRequestGroupEventDuplicate`: `0x30, 0x80000001, 2`, the copied handle
//! at word 3, the group at 4), which the tests write as they are. Replies are compared as whole
//! message buffers.

mod saved;

use std::sync::Arc;

use starquake_gpu::interface::Completions;
use starquake_gpu::syncpoint::Syncpoints;
use starquake_kernel::dispatch::DeviceShareError;
use starquake_kernel::handle::Handle;
use starquake_kernel::memory::DeviceShareToken;
use starquake_kernel::result::SvcResult;
use starquake_kernel::slab::SlabId;
use starquake_services::framework::fake::FakeKernel;
use starquake_services::framework::result::INVALID_CMIF_REQUEST;
use starquake_services::framework::{Halt, Handled, Kernel, Session, Stop, UNKNOWN_COMMAND_ID};
use starquake_trace::{Env, EventKind, FixedClock, LiveEnv, ServiceId};

use super::command::*;
use super::*;
use crate::completion::CompletionQueue;
use crate::shared::TimerHost;
use crate::shared::vsync::INTERVAL_NS;
use crate::testing::{SPECIAL, STALE, copied, halves, le, result_only, success, tipc_type};

/// The number the tests' vi2 server is known by.
const VI2: ServiceId = ServiceId(0x0305);
/// The number gsv records its notes under: not the server's, so mixing them up fails.
const GSV: ServiceId = ServiceId(0x0657);
/// The `Env` clock's first reading.
const START: u64 = 0x10_0000;
/// A special header that counts one copied handle (PR `DecodeRequestGroupEventDuplicate`:
/// `packet[2] != 2`).
const ONE_COPY: u32 = 2;
/// Where 21's out buffer is in the fake caller's memory.
const BUFFER: u64 = 0x8000;

// ----- Requests, word by word -------------------------------------------------------------------

/// A `u64` as its two words, low first.
fn w(value: u64) -> [u32; 2] {
    halves(value)
}

/// 1, 3 and 5: PR `GetSessionLimit`, `packet[0] == 16 + ctx.GetCommand() && packet[1] == 0`.
fn get_notification_limit() -> Vec<u32> {
    vec![0x11, 0]
}
fn get_session_limit() -> Vec<u32> {
    vec![0x13, 0]
}
fn get_request_group_limit() -> Vec<u32> {
    vec![0x15, 0]
}

/// 11 (PR `GetNotificationEvent`: `packet[0] == 0x1b && packet[1] == 1 && packet[2] == 0`).
fn get_notification_event(word: u32) -> Vec<u32> {
    vec![0x1B, 1, word]
}

/// 13 (PR `ReleaseNotificationEvent`: `packet[0] == 0x1d && packet[1] == 1 && packet[2] == 0`).
fn release_notification_event(word: u32) -> Vec<u32> {
    vec![0x1D, 1, word]
}

/// 21 (PR `GetNotifications`, which checks no header word): type `0x25`, one B descriptor
/// (header bits 24–27), no data; the descriptor's size, address (low 32 bits) and a third word
/// of zero (no high address or size bits, flags 0).
fn get_notifications(size: u32) -> Vec<u32> {
    vec![0x0100_0025, 0, size, u32::try_from(BUFFER).unwrap(), 0]
}

/// 31 (PR `CreateRequestGroupEvent`: `packet[0] == 0x2f && packet[1] == 1`, the group at 2).
fn create_group(group: u32) -> Vec<u32> {
    vec![0x2F, 1, group]
}

/// 32 (PR `DecodeRequestGroupEventDuplicate`: `0x30, 0x80000001, 2`, handle, group).
fn duplicate_group(handle: u32, group: u32) -> Vec<u32> {
    vec![0x30, 0x8000_0001, 2, handle, group]
}

/// 51 (PR `ConfigureRequestGroups`: `packet[0] == 0x43 && packet[1] == 1`).
fn configure_groups(count: u32) -> Vec<u32> {
    vec![0x43, 1, count]
}

/// 101 (PR `OpenDisplay`: `0x75, 3`, name at 2–3, flags at 4).
fn open_display(name: u64, flags: u32) -> Vec<u32> {
    let [lo, hi] = w(name);
    vec![0x75, 3, lo, hi, flags]
}

/// 102 (PR `CloseDisplay`: `0x76, 2`, name at 2–3).
fn close_display(name: u64) -> Vec<u32> {
    let [lo, hi] = w(name);
    vec![0x76, 2, lo, hi]
}

/// A request of the name alone: 104 (`0x78, 2`), 105 (`0x79`), 106 (`0x7a`), 107 (`0x7b`) and
/// 354 (`0x172`), PR's header words for each.
fn named(type_word: u32, name: u64) -> Vec<u32> {
    let [lo, hi] = w(name);
    vec![type_word, 2, lo, hi]
}

/// 111 (PR `OpenEndpoint`: `0x7f, 6`, name, display name, client).
fn open_endpoint(name: u64, display: u64, token: u64) -> Vec<u32> {
    let mut words = vec![0x7F, 6];
    words.extend(w(name));
    words.extend(w(display));
    words.extend(w(token));
    words
}

/// 112 (PR `CloseEndpoint`: `0x80, 4`, name, display name).
fn close_endpoint(name: u64, display: u64) -> Vec<u32> {
    let mut words = vec![0x80, 4];
    words.extend(w(name));
    words.extend(w(display));
    words
}

/// 201 (PR `RegisterClientState`: `0xd9, 4`, client at 2–3, bits at 4 and 5).
fn register_client(token: u64, retire: u32, available: u32) -> Vec<u32> {
    let [lo, hi] = w(token);
    vec![0xD9, 4, lo, hi, retire, available]
}

/// 202 (PR `UnregisterClientState`: `0xda, 2`), 212 (`GetClientAvailability`: `0xe4, 2`) and
/// 252 (`UnbindClientDisplay`: `0x10c, 2`): the client alone.
fn client_only(type_word: u32, token: u64) -> Vec<u32> {
    let [lo, hi] = w(token);
    vec![type_word, 2, lo, hi]
}

/// 251 (PR `BindClientDisplay`: `0x10b, 4`, client, name).
fn bind(token: u64, name: u64) -> Vec<u32> {
    let mut words = vec![0x10B, 4];
    words.extend(w(token));
    words.extend(w(name));
    words
}

/// 261–272 (PR `RegisterRequestGroupMonitor`: `packet[0] == ctx.GetCommand() + 16 && packet[1]
/// == 3`, name at 2–3, group at 4).
fn monitor(command: u32, name: u64, group: u32) -> Vec<u32> {
    let [lo, hi] = w(name);
    vec![command + 16, 3, lo, hi, group]
}

/// 301 (PR `IsSupportedDisplayParameter301`: `0x13d, 6`, name, value, mask).
fn parameter_301(name: u64, value: u64, mask: u64) -> Vec<u32> {
    let mut words = vec![0x13D, 6];
    words.extend(w(name));
    words.extend(w(value));
    words.extend(w(mask));
    words
}

/// 351 (PR `SetClientOutputEncoding`: `0x16f, 4`, client, encoding).
fn output_encoding(token: u64, encoding: u64) -> Vec<u32> {
    let mut words = vec![0x16F, 4];
    words.extend(w(token));
    words.extend(w(encoding));
    words
}

/// 352 (PR `SetClientOutputLevels`: `0x170, 5`, client, three floats at 4–6).
fn output_levels(token: u64, levels: [f32; 3]) -> Vec<u32> {
    let mut words = vec![0x170, 5];
    words.extend(w(token));
    words.extend(levels.map(f32::to_bits));
    words
}

#[test]
fn the_reference_header_words_are_the_command_ids() {
    let cases = [
        (get_notification_limit(), GET_NOTIFICATION_LIMIT),
        (get_session_limit(), GET_SESSION_LIMIT),
        (get_request_group_limit(), GET_REQUEST_GROUP_LIMIT),
        (get_notification_event(0), GET_NOTIFICATION_EVENT),
        (release_notification_event(0), RELEASE_NOTIFICATION_EVENT),
        (create_group(0), CREATE_REQUEST_GROUP_EVENT),
        (configure_groups(0), CONFIGURE_REQUEST_GROUPS),
        (open_display(0, 0), OPEN_DISPLAY),
        (close_display(0), CLOSE_DISPLAY),
        (named(0x78, 0), GET_HDR_STATUS),
        (named(0x79, 0), GET_DISPLAY_STATE_105),
        (named(0x7A, 0), GET_BUFFER_MAX_RESOLUTION),
        (named(0x7B, 0), GET_DISPLAY_STATE_107),
        (named(0x172, 0), GET_DISPLAY_STATE_354),
        (open_endpoint(0, 0, 0), OPEN_ENDPOINT),
        (close_endpoint(0, 0), CLOSE_ENDPOINT),
        (register_client(0, 0, 0), REGISTER_CLIENT_STATE),
        (client_only(0xDA, 0), UNREGISTER_CLIENT_STATE),
        (client_only(0xE4, 0), GET_CLIENT_AVAILABILITY),
        (bind(0, 0), BIND_CLIENT_DISPLAY),
        (client_only(0x10C, 0), UNBIND_CLIENT_DISPLAY),
        (parameter_301(0, 0, 0), SET_DISPLAY_PARAMETER_301),
        (output_encoding(0, 0), SET_CLIENT_OUTPUT_ENCODING),
        (output_levels(0, [0.0; 3]), SET_CLIENT_OUTPUT_LEVELS),
    ];
    for (words, command) in cases {
        assert_eq!(words[0], tipc_type(command), "{command}");
        assert_eq!(
            usize::try_from(words[1]).unwrap(),
            words.len() - 2,
            "{command}"
        );
    }
    // 21 and 32 carry a descriptor and a special header: their type words.
    assert_eq!(
        get_notifications(0)[0] & 0xFFFF,
        tipc_type(GET_NOTIFICATIONS)
    );
    assert_eq!(
        duplicate_group(0, 0)[0],
        tipc_type(DUPLICATE_REQUEST_GROUP_EVENT)
    );
    for command in MONITORS {
        assert_eq!(monitor(command, 0, 0)[0], tipc_type(command));
    }
}

/// Every command id against PR's table, written out (`{1, ...}`, `{3, ...}` and so on), and the
/// monitors as PR lists them, 261 to 272.
#[test]
fn the_command_ids_are_prs() {
    assert_eq!(
        [
            GET_NOTIFICATION_LIMIT,
            GET_SESSION_LIMIT,
            GET_REQUEST_GROUP_LIMIT,
            GET_NOTIFICATION_EVENT,
            RELEASE_NOTIFICATION_EVENT,
            GET_NOTIFICATIONS,
            CREATE_REQUEST_GROUP_EVENT,
            DUPLICATE_REQUEST_GROUP_EVENT,
            CONFIGURE_REQUEST_GROUPS,
            OPEN_DISPLAY,
            CLOSE_DISPLAY,
            GET_HDR_STATUS,
            GET_DISPLAY_STATE_105,
            GET_BUFFER_MAX_RESOLUTION,
            GET_DISPLAY_STATE_107,
            OPEN_ENDPOINT,
            CLOSE_ENDPOINT,
            REGISTER_CLIENT_STATE,
            UNREGISTER_CLIENT_STATE,
            GET_CLIENT_AVAILABILITY,
            BIND_CLIENT_DISPLAY,
            UNBIND_CLIENT_DISPLAY,
            SET_DISPLAY_PARAMETER_301,
            SET_CLIENT_OUTPUT_ENCODING,
            SET_CLIENT_OUTPUT_LEVELS,
            GET_DISPLAY_STATE_354,
        ],
        [
            1, 3, 5, 11, 13, 21, 31, 32, 51, 101, 102, 104, 105, 106, 107, 111, 112, 201, 202, 212,
            251, 252, 301, 351, 352, 354
        ]
    );
    assert_eq!(
        MONITORS,
        [261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272]
    );
    assert_eq!(REGISTER_DISPLAY_EVENT_MONITOR, 261);
    assert_eq!(UNREGISTER_DISPLAY_MONITOR_272, 272);
}

/// The constants against the values the references write out.
#[test]
fn the_constants_are_the_references_values() {
    // `Result{ErrorModule::VI, n}`: module 114 (0x72) in the low nine bits, the description above.
    assert_eq!(VI_OUT_OF_RANGE.raw(), 0x72 | (4 << 9));
    assert_eq!(VI_PERMISSION_DENIED.raw(), 0x72 | (5 << 9));
    assert_eq!(VI_NOT_FOUND.raw(), 0x72 | (7 << 9));
    assert_eq!(DEFAULT_DISPLAY, 0x0074_6C75_6166_6564);
    assert_eq!(
        (
            REQUEST_GROUPS,
            NOTIFICATION_WORDS,
            CLIENT_NOTIFICATIONS,
            MAX_CLIENTS
        ),
        (8, 32, 32, 16)
    );
    assert_eq!((MAX_DISPLAY_REFERENCES, OPEN_DISPLAY_FLAGS), (1024, 1));
    assert_eq!((ALL_SLOTS, OUTPUT_ENCODING), (0x7FFF, 0x14));
    assert_eq!((HDR_FORMAT_SDR, BUFFER_RESOLUTION_720P), (1, 1));
    assert_eq!(
        (DISPLAY_STATE_105, DISPLAY_STATE_107, DISPLAY_STATE_354),
        (0, [1, 1, 0], 1)
    );
    assert_eq!(DISPLAY_FLAG_63, 0x8000_0000_0000_0000);
    assert_eq!(NOTIFICATIONS_SIZE, 128);
    assert_eq!((VSYNC_KIND, MONITOR_KINDS), (0, 6));
    assert_eq!(
        (NOTIFICATION_LIMIT, SESSION_LIMIT, REQUEST_GROUP_LIMIT),
        (1, 1, 7)
    );
}

// ----- The rig ------------------------------------------------------------------------------------

/// A halt that nothing in these tests should reach: the environment never fails.
struct NoHalt;

impl Halt for NoHalt {
    fn halt(&self, stop: Stop) {
        unreachable!("the environment failed: {stop:?}");
    }
}

/// A [`TimerHost`] that signals through the fake kernel, as the wiring's does through the real
/// one, and keeps what was recorded.
struct KernelHost<'a> {
    kernel: &'a mut FakeKernel,
    recorded: Vec<EventKind>,
}

impl TimerHost for KernelHost<'_> {
    fn record(&mut self, kind: EventKind) {
        self.recorded.push(kind);
    }

    fn unlock_device_share(&mut self, _token: DeviceShareToken) -> Result<(), DeviceShareError> {
        Ok(())
    }

    fn signal_event(&mut self, event: ObjectId) -> SvcResult<()> {
        Kernel::signal_event(self.kernel, event)
    }
}

/// gsv's shared state over fresh syncpoints and a fresh queue.
fn fresh_shared() -> Arc<Shared> {
    let queue = Arc::new(CompletionQueue::new());
    let completions: Arc<dyn Completions> = Arc::<CompletionQueue>::clone(&queue);
    Arc::new(Shared::new(
        Arc::new(Syncpoints::new(completions)),
        queue,
        GSV,
    ))
}

/// A vi2 server over gsv's shared state, reading a fixed `Env` clock, a kernel, and one session.
struct Rig {
    kernel: FakeKernel,
    server: Server,
    session: ObjectId,
    shared: Arc<Shared>,
}

/// The vi2 server over `shared`, with a fixed clock that reads [`START`], then 7 ns more each
/// time.
fn server_over(shared: &Arc<Shared>, policy: UnknownPolicy) -> Server {
    let env: Arc<dyn Env> = Arc::new(LiveEnv::new(FixedClock::new(START, 7), 0));
    vi2_server(policy, Arc::clone(shared))
        .with_service_id(VI2)
        .with_env(env, Arc::new(NoHalt))
}

impl Rig {
    fn with(policy: UnknownPolicy) -> Self {
        let mut kernel = FakeKernel::new(0).unwrap();
        kernel.set_current_service(VI2);
        kernel.map(BUFFER, vec![STALE; 0x200]);
        let session = kernel.new_object();
        let shared = fresh_shared();
        let mut server = server_over(&shared, policy);
        server.accept(session).unwrap();
        Self {
            kernel,
            server,
            session,
            shared,
        }
    }

    fn new() -> Self {
        Self::with(UnknownPolicy::Strict)
    }

    /// Sends the message `words` lays out on `session`; the message buffer afterwards.
    fn send_on(&mut self, session: ObjectId, words: &[u32]) -> Vec<u8> {
        let mut buffer = vec![STALE; starquake_ipc::BUFFER_SIZE];
        let bytes = le(words);
        buffer[..bytes.len()].copy_from_slice(&bytes);
        let handled = self.server.handle(&mut self.kernel, session, &mut buffer);
        assert_eq!(handled, Ok(Handled::Replied));
        buffer
    }

    fn send(&mut self, words: &[u32]) -> Vec<u8> {
        self.send_on(self.session, words)
    }

    /// Sends `words` and expects success with no data and no handle.
    fn ok(&mut self, words: &[u32]) {
        let command = words[0] & 0xFFFF;
        let id = command - tipc_type(0);
        assert_eq!(self.send(words), success(id, &[], &[]), "{words:x?}");
    }

    /// The object behind a handle the session handed out.
    fn object(&self, handle: u32) -> ObjectId {
        self.kernel
            .resolve_handle(Handle::from_raw(handle))
            .unwrap()
    }

    /// The handle a reply copied, checked to be the reply's only one.
    fn handed(reply: &[u8]) -> u32 {
        let handles = copied(reply);
        assert_eq!(handles.len(), 1);
        handles[0]
    }

    /// 31 for `group`, expected to succeed: its event.
    fn group_event(&mut self, group: u32) -> ObjectId {
        let reply = self.send(&create_group(group));
        let handle = Self::handed(&reply);
        assert_eq!(reply, success(CREATE_REQUEST_GROUP_EVENT, &[handle], &[]));
        self.object(handle)
    }

    /// 11 with 0, expected to succeed: the notification event.
    fn notification_event(&mut self) -> ObjectId {
        let reply = self.send(&get_notification_event(0));
        let handle = Self::handed(&reply);
        assert_eq!(reply, success(GET_NOTIFICATION_EVENT, &[handle], &[]));
        self.object(handle)
    }

    /// The session's state, read from the server's snapshot: the interface's name with its
    /// length, then the session's saved bytes with theirs (the framework's `SavedTarget`). The
    /// rig's server holds one session at a time.
    fn session_state(&self) -> Vi2Session {
        let snapshot = self.server.save().unwrap();
        let name = Vi2Session::interface().as_bytes();
        let mut row = vec![u8::try_from(name.len()).unwrap()];
        row.extend(name);
        let at = snapshot
            .windows(row.len())
            .position(|window| window == row.as_slice())
            .unwrap();
        let (bytes, _) = postcard::take_from_bytes::<Vec<u8>>(&snapshot[at + row.len()..]).unwrap();
        postcard::from_bytes(&bytes).unwrap()
    }

    /// Closes the session and opens a new one in `state`, as a restore of it would.
    fn replace(&mut self, state: Vi2Session) {
        self.server.close(self.session).unwrap();
        self.session = self.kernel.new_object();
        self.server.open(self.session, Box::new(state)).unwrap();
    }

    /// Opens the default display and registers `token` with bits `retire` and `available`,
    /// bound to it.
    fn bound_client(&mut self, token: u64, retire: u32, available: u32) {
        self.ok(&open_display(DEFAULT_DISPLAY, OPEN_DISPLAY_FLAGS));
        self.ok(&register_client(token, retire, available));
        self.ok(&bind(token, DEFAULT_DISPLAY));
    }

    /// Plays the timer thread's live vsync turn at `now`; what it recorded.
    fn tick(&mut self, now: u64) -> Vec<EventKind> {
        let mut host = KernelHost {
            kernel: &mut self.kernel,
            recorded: Vec::new(),
        };
        self.shared.tick_vsync(&mut host, HostNanos(now));
        host.recorded
    }

    fn signals(&self, event: ObjectId) -> u32 {
        self.kernel.signals(event).unwrap()
    }

    fn listeners(&self) -> Vec<ObjectId> {
        self.shared.lock().vsync().listeners().collect()
    }
}

fn refused(command: u32, error: KernelError) -> Vec<u8> {
    result_only(command, error.code())
}

/// A display name that is not in [`KNOWN_DISPLAYS`]: `"unknown"`.
const UNKNOWN: u64 = u64::from_le_bytes(*b"unknown\0");

// ----- Limits, and every reply ------------------------------------------------------------------

/// 1 and 3 answer the chosen side of their conflict (1, NX2's), 5 the references' 7.
#[test]
fn the_limits_are_the_chosen_answers() {
    let mut rig = Rig::new();
    assert_eq!(
        rig.send(&get_notification_limit()),
        success(GET_NOTIFICATION_LIMIT, &[], &[1])
    );
    assert_eq!(
        rig.send(&get_session_limit()),
        success(GET_SESSION_LIMIT, &[], &[1])
    );
    assert_eq!(
        rig.send(&get_request_group_limit()),
        success(GET_REQUEST_GROUP_LIMIT, &[], &[7])
    );
}

/// Each command's success reply, as whole message buffers, in an order that reaches each: the
/// data words written out from PR's `IPC::ResponseBuilder` and what it pushes.
#[test]
fn every_command_s_reply_bytes() {
    let mut rig = Rig::new();
    let token = 0x0000_00A1_0000_00B2;
    let notification = rig.notification_event();
    rig.ok(&release_notification_event(0));
    let event = rig.group_event(3);
    let handle = rig.kernel.insert_handle(event).unwrap().raw();
    let reply = rig.send(&duplicate_group(handle, 3));
    let again = Rig::handed(&reply);
    assert_eq!(reply, success(DUPLICATE_REQUEST_GROUP_EVENT, &[again], &[]));
    assert_eq!(rig.object(again), event);
    rig.ok(&configure_groups(REQUEST_GROUPS - 1));
    rig.ok(&open_display(DEFAULT_DISPLAY, OPEN_DISPLAY_FLAGS));
    // 104: `rb.Push(format)` (a `u64`), `rb.Push<u32>(0)`.
    assert_eq!(
        rig.send(&named(0x78, DEFAULT_DISPLAY)),
        success(GET_HDR_STATUS, &[], &[1, 0, 0])
    );
    // 105: `rb.Push<u32>(0)`.
    assert_eq!(
        rig.send(&named(0x79, DEFAULT_DISPLAY)),
        success(GET_DISPLAY_STATE_105, &[], &[0])
    );
    // 106: `rb.Push(resolution)`, a `u64`.
    assert_eq!(
        rig.send(&named(0x7A, DEFAULT_DISPLAY)),
        success(GET_BUFFER_MAX_RESOLUTION, &[], &[1, 0])
    );
    // 107: three `u64`: 1, 1, 0.
    assert_eq!(
        rig.send(&named(0x7B, DEFAULT_DISPLAY)),
        success(GET_DISPLAY_STATE_107, &[], &[1, 0, 1, 0, 0, 0])
    );
    // 354: one `u64`, 1.
    assert_eq!(
        rig.send(&named(0x172, DEFAULT_DISPLAY)),
        success(GET_DISPLAY_STATE_354, &[], &[1, 0])
    );
    rig.ok(&register_client(token, 4, 5));
    // 212: `rb.Push(available)`, a `u64`: every slot free.
    assert_eq!(
        rig.send(&client_only(0xE4, token)),
        success(GET_CLIENT_AVAILABILITY, &[], &[0x7FFF, 0])
    );
    rig.ok(&bind(token, DEFAULT_DISPLAY));
    rig.ok(&open_endpoint(DEFAULT_DISPLAY, DEFAULT_DISPLAY, token));
    rig.ok(&close_endpoint(DEFAULT_DISPLAY, DEFAULT_DISPLAY));
    rig.ok(&output_encoding(token, OUTPUT_ENCODING));
    rig.ok(&output_levels(token, [0.0, 1.0, 0.8]));
    rig.ok(&parameter_301(
        DEFAULT_DISPLAY,
        DISPLAY_FLAG_63,
        DISPLAY_FLAG_63,
    ));
    for command in MONITORS {
        rig.ok(&monitor(command, DEFAULT_DISPLAY, 3));
    }
    rig.ok(&client_only(0x10C, token));
    rig.ok(&client_only(0xDA, token));
    rig.ok(&close_display(DEFAULT_DISPLAY));
    // 21: no data; the bitmap goes to the buffer: client 4's retire bit and the display's.
    assert_eq!(rig.signals(notification), 2);
    rig.ok(&get_notifications(128));
    let mut expected = [0; 128];
    expected[0] = (1 << 4) | (1 << OPEN_DISPLAY_FLAGS);
    assert_eq!(rig.kernel.memory(BUFFER, 128).unwrap(), &expected[..]);
}

/// A failure's reply is its result word alone: PR also writes the data words, zero.
#[test]
fn a_failure_reply_is_the_result_word_alone() {
    let mut rig = Rig::new();
    assert_eq!(
        rig.send(&named(0x78, DEFAULT_DISPLAY)),
        result_only(GET_HDR_STATUS, VI_NOT_FOUND)
    );
    assert_eq!(
        rig.send(&named(0x7B, UNKNOWN)),
        result_only(GET_DISPLAY_STATE_107, VI_NOT_FOUND)
    );
    assert_eq!(
        rig.send(&client_only(0xE4, 9)),
        refused(GET_CLIENT_AVAILABILITY, KernelError::NotFound)
    );
}

/// A request one data word short of what its command reads is refused, changing nothing; one
/// with a word more than PR's header counts is read as if it were not there.
#[test]
fn a_short_request_is_refused_and_a_longer_one_is_read() {
    let mut rig = Rig::new();
    let mut short = open_display(DEFAULT_DISPLAY, OPEN_DISPLAY_FLAGS);
    short.pop();
    short[1] -= 1;
    assert_eq!(
        rig.send(&short),
        result_only(OPEN_DISPLAY, INVALID_CMIF_REQUEST)
    );
    assert_eq!(rig.session_state().display_references(), 0);
    let mut long = open_display(DEFAULT_DISPLAY, OPEN_DISPLAY_FLAGS);
    long.push(0xDEAD_BEEF);
    long[1] += 1;
    rig.ok(&long);
    assert_eq!(rig.session_state().display_references(), 1);
    // A 32 with no handle is refused before the group is looked at.
    assert_eq!(
        rig.send(&[0x30, 1, 0]),
        result_only(DUPLICATE_REQUEST_GROUP_EVENT, INVALID_CMIF_REQUEST)
    );
}

/// The server is named, serves TIPC only, and leaves 228 and 229 (slice 14) to its policy.
#[test]
fn the_server_is_named_and_leaves_present_to_the_policy() {
    let mut rig = Rig::new();
    assert_eq!(rig.server.name(), VI2_NAME);
    assert_eq!(VI2_NAME, "gsv:vi2");
    assert_eq!(Vi2Session::new().commands().cmif, []);
    for command in [228, 229] {
        assert_eq!(
            rig.send(&[tipc_type(command), 0]),
            result_only(command, UNKNOWN_COMMAND_ID)
        );
    }
}

// ----- Notifications ------------------------------------------------------------------------------

/// 11 makes one event per session and hands it out each time, signalled only while a
/// notification is raised.
#[test]
fn the_notification_event_is_made_once_and_signalled_while_raised() {
    let mut rig = Rig::new();
    assert_eq!(rig.session_state().notification_event(), None);
    let event = rig.notification_event();
    assert_eq!(rig.notification_event(), event);
    assert_eq!(rig.signals(event), 0);
    // A raised notification: the display's, at its last close.
    rig.ok(&open_display(DEFAULT_DISPLAY, OPEN_DISPLAY_FLAGS));
    rig.ok(&close_display(DEFAULT_DISPLAY));
    assert_eq!(rig.signals(event), 1);
    // While raised, every 11 signals it again.
    assert_eq!(rig.notification_event(), event);
    assert_eq!(rig.signals(event), 2);
    rig.ok(&get_notifications(128));
    assert!(!rig.kernel.is_signalled(event).unwrap());
    assert_eq!(rig.notification_event(), event);
    assert_eq!(rig.signals(event), 2);
}

/// 21 hands out the bitmap and clears it and the event; a buffer one byte short is refused and
/// leaves both, and the exact size is accepted.
#[test]
fn get_notifications_hands_out_the_bitmap_and_clears_it() {
    let mut rig = Rig::new();
    let event = rig.notification_event();
    rig.ok(&register_client(7, 31, 30));
    rig.ok(&client_only(0xDA, 7));
    assert!(rig.kernel.is_signalled(event).unwrap());
    let short = u32::try_from(NOTIFICATIONS_SIZE - 1).unwrap();
    assert_eq!(
        rig.send(&get_notifications(short)),
        refused(GET_NOTIFICATIONS, KernelError::InvalidSize)
    );
    assert_eq!(rig.kernel.memory(BUFFER, 1).unwrap(), [STALE]);
    assert!(rig.kernel.is_signalled(event).unwrap());
    assert_eq!(rig.session_state().notifications()[0], 1 << 31);
    let exact = u32::try_from(NOTIFICATIONS_SIZE).unwrap();
    rig.ok(&get_notifications(exact));
    assert_eq!(
        rig.kernel.memory(BUFFER, 4).unwrap(),
        (1_u32 << 31).to_le_bytes()
    );
    assert_eq!(
        rig.kernel.memory(BUFFER + NOTIFICATIONS_SIZE, 1).unwrap(),
        [STALE]
    );
    assert!(!rig.kernel.is_signalled(event).unwrap());
    assert_eq!(
        rig.session_state().notifications(),
        &[0; NOTIFICATION_WORDS]
    );
    // Read again: all zero.
    rig.ok(&get_notifications(exact));
    assert_eq!(rig.kernel.memory(BUFFER, 128).unwrap(), &[0; 128][..]);
}

/// A signal the kernel refuses is the request's error, after the change it notifies of: a
/// restored session whose notification event the kernel does not hold unregisters the client,
/// raises its bit, and answers `InvalidHandle`.
#[test]
fn a_refused_signal_is_the_request_s_error_after_the_change() {
    let mut rig = Rig::new();
    rig.ok(&register_client(7, 2, 3));
    let mut state = rig.session_state();
    let foreign = rig.kernel.new_object();
    state.notification_event = Some(foreign);
    rig.replace(state);
    assert_eq!(
        rig.send(&client_only(0xDA, 7)),
        refused(UNREGISTER_CLIENT_STATE, KernelError::InvalidHandle)
    );
    let state = rig.session_state();
    assert_eq!(state.clients(), []);
    assert_eq!(state.notifications()[0], 1 << 2);
}

// ----- Request groups -----------------------------------------------------------------------------

/// 31 refuses a group past the last and one whose event exists; the last group is accepted.
#[test]
fn create_request_group_event_checks_its_group() {
    let mut rig = Rig::new();
    assert_eq!(
        rig.send(&create_group(REQUEST_GROUPS)),
        result_only(CREATE_REQUEST_GROUP_EVENT, VI_OUT_OF_RANGE)
    );
    assert_eq!(
        rig.send(&create_group(u32::MAX)),
        result_only(CREATE_REQUEST_GROUP_EVENT, VI_OUT_OF_RANGE)
    );
    let last = rig.group_event(REQUEST_GROUPS - 1);
    assert_eq!(
        rig.send(&create_group(REQUEST_GROUPS - 1)),
        result_only(CREATE_REQUEST_GROUP_EVENT, VI_PERMISSION_DENIED)
    );
    let first = rig.group_event(0);
    assert_ne!(first, last);
    let groups: Vec<_> = rig
        .session_state()
        .groups()
        .iter()
        .map(super::RequestGroup::event)
        .collect();
    let mut expected = vec![None; 8];
    expected[0] = Some(first);
    expected[7] = Some(last);
    assert_eq!(groups, expected);
}

/// 31's event is copied: a special header counting one copied handle, then the result.
#[test]
fn the_request_group_event_is_handed_out_copied() {
    let mut rig = Rig::new();
    let reply = rig.send(&create_group(0));
    let handle = Rig::handed(&reply);
    assert_eq!(
        &reply[..20],
        &le(&[0x2F, SPECIAL | 1, ONE_COPY, handle, 0])[..]
    );
}

/// 32 hands the group's event out again; refuses a group past the last, a group with no event
/// and a handle naming another object; and a handle naming nothing is the kernel's error.
#[test]
fn duplicate_hands_out_the_same_event_and_refuses_another() {
    let mut rig = Rig::new();
    let zero = rig.group_event(0);
    let one = rig.group_event(1);
    let zero_handle = rig.kernel.insert_handle(zero).unwrap().raw();
    let one_handle = rig.kernel.insert_handle(one).unwrap().raw();
    let reply = rig.send(&duplicate_group(zero_handle, 0));
    assert_eq!(rig.object(Rig::handed(&reply)), zero);
    assert_eq!(
        rig.send(&duplicate_group(one_handle, 0)),
        refused(DUPLICATE_REQUEST_GROUP_EVENT, KernelError::InvalidHandle)
    );
    assert_eq!(
        rig.send(&duplicate_group(zero_handle, 2)),
        result_only(DUPLICATE_REQUEST_GROUP_EVENT, VI_NOT_FOUND)
    );
    assert_eq!(
        rig.send(&duplicate_group(zero_handle, REQUEST_GROUPS)),
        result_only(DUPLICATE_REQUEST_GROUP_EVENT, VI_OUT_OF_RANGE)
    );
    assert_eq!(
        rig.send(&duplicate_group(0x7FFF, 0)),
        refused(DUPLICATE_REQUEST_GROUP_EVENT, KernelError::InvalidHandle)
    );
}

/// The listed divergence: with a handle naming an object that is not the group's event and a
/// group past the last, the group's refusal is answered (PR answers `InvalidHandle` when the
/// object is not a readable event); with the group in range but no event, `VI_NOT_FOUND`.
#[test]
fn duplicate_checks_the_group_before_the_identity() {
    let mut rig = Rig::new();
    let other = rig.kernel.new_object();
    let handle = rig.kernel.insert_handle(other).unwrap().raw();
    assert_eq!(
        rig.send(&duplicate_group(handle, REQUEST_GROUPS)),
        result_only(DUPLICATE_REQUEST_GROUP_EVENT, VI_OUT_OF_RANGE)
    );
    assert_eq!(
        rig.send(&duplicate_group(handle, 0)),
        result_only(DUPLICATE_REQUEST_GROUP_EVENT, VI_NOT_FOUND)
    );
}

/// 51 refuses a count at the number of groups (the out-of-range check first, also when 51 was
/// sent before), takes one below it once, and refuses a second.
#[test]
fn configure_request_groups_takes_one_count_once() {
    let mut rig = Rig::new();
    assert_eq!(
        rig.send(&configure_groups(REQUEST_GROUPS)),
        result_only(CONFIGURE_REQUEST_GROUPS, VI_OUT_OF_RANGE)
    );
    rig.ok(&configure_groups(REQUEST_GROUPS - 1));
    assert_eq!(rig.session_state().group_count(), Some(REQUEST_GROUPS - 1));
    assert_eq!(
        rig.send(&configure_groups(0)),
        result_only(CONFIGURE_REQUEST_GROUPS, VI_PERMISSION_DENIED)
    );
    assert_eq!(
        rig.send(&configure_groups(REQUEST_GROUPS)),
        result_only(CONFIGURE_REQUEST_GROUPS, VI_OUT_OF_RANGE)
    );
    assert_eq!(rig.session_state().group_count(), Some(REQUEST_GROUPS - 1));
}

// ----- Vsync ----------------------------------------------------------------------------------------

/// The first group's event starts the vsync clock one interval after the `Env` clock's reading
/// on the handler's thread, and listens from then on; a second group joins the running clock.
#[test]
fn a_new_group_listens_and_starts_the_clock_from_the_env_reading() {
    let mut rig = Rig::new();
    assert_eq!(rig.shared.vsync_deadline(), None);
    let first = rig.group_event(0);
    assert_eq!(
        rig.shared.vsync_deadline(),
        Some(HostNanos(START + INTERVAL_NS))
    );
    let second = rig.group_event(5);
    assert_eq!(
        rig.shared.vsync_deadline(),
        Some(HostNanos(START + INTERVAL_NS))
    );
    assert_eq!(rig.listeners(), [first, second]);
}

/// The fallback: a group's event with no registration is ticked on the clock's first tick and
/// every one after, and not before the deadline.
#[test]
fn an_unregistered_group_is_ticked_by_the_fallback() {
    let mut rig = Rig::new();
    let event = rig.group_event(2);
    rig.tick(START + INTERVAL_NS - 1);
    assert_eq!(rig.signals(event), 0);
    rig.tick(START + INTERVAL_NS);
    assert_eq!(rig.signals(event), 1);
    rig.tick(START + 2 * INTERVAL_NS);
    assert_eq!(rig.signals(event), 2);
}

/// A group registered for kind 0 (261) on a known display is ticked, as PR's `StartGroupTick`
/// ticks it; its registration does not signal it.
#[test]
fn a_kind_zero_monitor_ticks_its_group() {
    let mut rig = Rig::new();
    let event = rig.group_event(1);
    rig.ok(&monitor(REGISTER_DISPLAY_EVENT_MONITOR, DEFAULT_DISPLAY, 1));
    assert_eq!(rig.signals(event), 0);
    assert_eq!(
        rig.session_state().groups()[1].monitor(),
        Some(Monitor {
            display: DEFAULT_DISPLAY,
            kind: VSYNC_KIND
        })
    );
    assert_eq!(rig.listeners(), [event]);
    rig.tick(START + INTERVAL_NS);
    assert_eq!(rig.signals(event), 1);
}

/// A group registered for each other kind is signalled once at registration and not ticked;
/// unregistering it signals it again and the fallback ticks it again. The last group stopping
/// listening stops the clock.
#[test]
fn a_monitor_of_another_kind_signals_once_and_is_not_ticked() {
    for command in MONITORS.into_iter().skip(2).step_by(2) {
        let (kind, registers) = monitor_kind(command).unwrap();
        assert!(registers && kind != VSYNC_KIND);
        let mut rig = Rig::new();
        let event = rig.group_event(4);
        rig.ok(&monitor(command, DEFAULT_DISPLAY, 4));
        assert_eq!(rig.signals(event), 1, "{command}");
        assert_eq!(rig.listeners(), [], "{command}");
        assert_eq!(rig.shared.vsync_deadline(), None);
        rig.tick(START + INTERVAL_NS);
        assert_eq!(rig.signals(event), 1, "{command}");
        rig.ok(&monitor(command + 1, DEFAULT_DISPLAY, 4));
        assert_eq!(rig.signals(event), 2, "{command}");
        assert_eq!(rig.listeners(), [event], "{command}");
        assert_eq!(rig.session_state().groups()[4].monitor(), None);
        let deadline = rig.shared.vsync_deadline().unwrap();
        rig.tick(deadline.0);
        assert_eq!(rig.signals(event), 3, "{command}");
    }
}

/// 262 on a kind-0 registration takes it out and signals the group; the fallback keeps it
/// listening.
#[test]
fn unregistering_kind_zero_signals_and_the_fallback_keeps_ticking() {
    let mut rig = Rig::new();
    let event = rig.group_event(0);
    rig.ok(&monitor(REGISTER_DISPLAY_EVENT_MONITOR, DEFAULT_DISPLAY, 0));
    rig.ok(&monitor(
        UNREGISTER_DISPLAY_EVENT_MONITOR,
        DEFAULT_DISPLAY,
        0,
    ));
    assert_eq!(rig.signals(event), 1);
    assert_eq!(rig.session_state().groups()[0].monitor(), None);
    assert_eq!(rig.listeners(), [event]);
}

/// An unregister that does not match (another name, another kind, another group, a group with
/// no registration, a group past the last, a zero name) succeeds and changes nothing.
#[test]
fn an_unregister_that_does_not_match_changes_nothing() {
    let mut rig = Rig::new();
    let event = rig.group_event(0);
    let other = rig.group_event(1);
    rig.ok(&monitor(REGISTER_HDR_STATUS_MONITOR, DEFAULT_DISPLAY, 0));
    let before = rig.session_state();
    let listeners = rig.listeners();
    let null = u64::from_le_bytes(*b"null\0\0\0\0");
    for words in [
        monitor(UNREGISTER_HDR_STATUS_MONITOR, null, 0),
        monitor(UNREGISTER_DISPLAY_EVENT_MONITOR, DEFAULT_DISPLAY, 0),
        monitor(UNREGISTER_DISPLAY_MONITOR_268, DEFAULT_DISPLAY, 0),
        monitor(UNREGISTER_HDR_STATUS_MONITOR, DEFAULT_DISPLAY, 1),
        monitor(UNREGISTER_HDR_STATUS_MONITOR, DEFAULT_DISPLAY, 2),
        monitor(
            UNREGISTER_HDR_STATUS_MONITOR,
            DEFAULT_DISPLAY,
            REQUEST_GROUPS,
        ),
        monitor(UNREGISTER_HDR_STATUS_MONITOR, 0, 0),
    ] {
        rig.ok(&words);
    }
    assert_eq!(rig.session_state(), before);
    assert_eq!(rig.listeners(), listeners);
    assert_eq!((rig.signals(event), rig.signals(other)), (1, 0));
}

/// 261 refuses, in PR's order, a group past the last or a zero name ([`VI_OUT_OF_RANGE`]), an
/// unknown name or a group with no event ([`VI_NOT_FOUND`]), and a registered group
/// ([`VI_PERMISSION_DENIED`]); with two faults at once the earlier check's answer comes back.
#[test]
fn registering_a_monitor_checks_in_prs_order() {
    let mut rig = Rig::new();
    let register = REGISTER_DISPLAY_EVENT_MONITOR;
    rig.group_event(0);
    let cases = [
        (
            monitor(register, DEFAULT_DISPLAY, REQUEST_GROUPS),
            VI_OUT_OF_RANGE,
        ),
        (monitor(register, UNKNOWN, REQUEST_GROUPS), VI_OUT_OF_RANGE),
        (monitor(register, 0, 0), VI_OUT_OF_RANGE),
        (monitor(register, 0, 1), VI_OUT_OF_RANGE),
        (monitor(register, UNKNOWN, 0), VI_NOT_FOUND),
        (monitor(register, DEFAULT_DISPLAY, 1), VI_NOT_FOUND),
        (monitor(register, UNKNOWN, 1), VI_NOT_FOUND),
    ];
    for (words, code) in cases {
        assert_eq!(rig.send(&words), result_only(register, code), "{words:x?}");
    }
    rig.ok(&monitor(register, DEFAULT_DISPLAY, 0));
    let player = u64::from_le_bytes(*b"player2\0");
    for command in [register, REGISTER_DISPLAY_MONITOR_271] {
        assert_eq!(
            rig.send(&monitor(command, player, 0)),
            result_only(command, VI_PERMISSION_DENIED)
        );
    }
    assert_eq!(
        rig.send(&monitor(register, UNKNOWN, 0)),
        result_only(register, VI_NOT_FOUND)
    );
}

/// When the vsync clock holds its most listeners, 31 answers `OutOfResource` and makes nothing;
/// one below it, 31 is accepted and fills it.
#[test]
fn a_full_vsync_clock_refuses_a_new_group() {
    let mut rig = Rig::new();
    for index in 0..MAX_LISTENERS - 1 {
        let filler = ObjectId::from_parts(u32::try_from(0x1000 + index).unwrap(), 9);
        rig.shared
            .lock()
            .listen_vsync(filler, HostNanos(0))
            .unwrap();
    }
    rig.group_event(0);
    assert_eq!(rig.shared.lock().vsync().listeners().count(), MAX_LISTENERS);
    let made = rig.kernel.new_object();
    assert_eq!(
        rig.send(&create_group(1)),
        refused(CREATE_REQUEST_GROUP_EVENT, KernelError::OutOfResource)
    );
    // Nothing was made: the next object the kernel makes follows the one minted above.
    assert_eq!(rig.session_state().groups()[1].event(), None);
    let next = rig.kernel.new_object();
    assert_eq!(SlabId::index(next), SlabId::index(made) + 1);
}

/// With the clock full, a kind-0 registration of a group the clock refused succeeds and leaves
/// it not listening; once room is made, a later kind-0 registration lets it in. An unregister
/// that should make a group listen on a full clock succeeds too.
#[test]
fn a_full_clock_leaves_a_registered_group_unticked() {
    let mut rig = Rig::new();
    let event = rig.group_event(0);
    rig.ok(&monitor(REGISTER_HDR_STATUS_MONITOR, DEFAULT_DISPLAY, 0));
    let fillers: Vec<ObjectId> = (0..MAX_LISTENERS)
        .map(|index| ObjectId::from_parts(u32::try_from(0x1000 + index).unwrap(), 9))
        .collect();
    for filler in &fillers {
        rig.shared
            .lock()
            .listen_vsync(*filler, HostNanos(0))
            .unwrap();
    }
    rig.ok(&monitor(UNREGISTER_HDR_STATUS_MONITOR, DEFAULT_DISPLAY, 0));
    assert!(!rig.listeners().contains(&event));
    rig.ok(&monitor(REGISTER_DISPLAY_EVENT_MONITOR, DEFAULT_DISPLAY, 0));
    assert!(!rig.listeners().contains(&event));
    rig.shared.lock().unlisten_vsync(fillers[0]);
    rig.ok(&monitor(
        UNREGISTER_DISPLAY_EVENT_MONITOR,
        DEFAULT_DISPLAY,
        0,
    ));
    assert!(rig.listeners().contains(&event));
}

// ----- The display ------------------------------------------------------------------------------

/// 101 refuses another name, other flags and a display opened the most times; 102 counts down,
/// raising the display's notification at the last close only, and refuses a display not open.
#[test]
fn the_display_opens_and_closes_by_references() {
    let mut rig = Rig::new();
    assert_eq!(
        rig.send(&open_display(UNKNOWN, OPEN_DISPLAY_FLAGS)),
        refused(OPEN_DISPLAY, KernelError::NotFound)
    );
    // Both faults: the name's answer.
    assert_eq!(
        rig.send(&open_display(UNKNOWN, 0)),
        refused(OPEN_DISPLAY, KernelError::NotFound)
    );
    for flags in [0, OPEN_DISPLAY_FLAGS + 1] {
        assert_eq!(
            rig.send(&open_display(DEFAULT_DISPLAY, flags)),
            refused(OPEN_DISPLAY, KernelError::NotImplemented)
        );
    }
    assert_eq!(
        rig.send(&close_display(DEFAULT_DISPLAY)),
        refused(CLOSE_DISPLAY, KernelError::NotFound)
    );
    rig.ok(&open_display(DEFAULT_DISPLAY, OPEN_DISPLAY_FLAGS));
    rig.ok(&open_display(DEFAULT_DISPLAY, OPEN_DISPLAY_FLAGS));
    rig.ok(&close_display(DEFAULT_DISPLAY));
    assert_eq!(rig.session_state().notifications()[0], 0);
    rig.ok(&close_display(DEFAULT_DISPLAY));
    assert_eq!(
        rig.session_state().notifications()[0],
        1 << OPEN_DISPLAY_FLAGS
    );
    assert_eq!(
        rig.send(&close_display(UNKNOWN)),
        refused(CLOSE_DISPLAY, KernelError::InvalidCombination)
    );
}

/// 101 counts up to [`MAX_DISPLAY_REFERENCES`] and refuses the one after.
#[test]
fn the_display_opens_at_most_the_most_references() {
    let mut rig = Rig::new();
    for _ in 0..MAX_DISPLAY_REFERENCES {
        rig.ok(&open_display(DEFAULT_DISPLAY, OPEN_DISPLAY_FLAGS));
    }
    assert_eq!(
        rig.session_state().display_references(),
        MAX_DISPLAY_REFERENCES
    );
    assert_eq!(
        rig.send(&open_display(DEFAULT_DISPLAY, OPEN_DISPLAY_FLAGS)),
        refused(OPEN_DISPLAY, KernelError::OutOfResource)
    );
    rig.ok(&close_display(DEFAULT_DISPLAY));
    rig.ok(&open_display(DEFAULT_DISPLAY, OPEN_DISPLAY_FLAGS));
}

/// Player displays are not opened: 101 answers `NotFound`, 102 `NotFound` and 251
/// `InvalidState`, PR's answers for a player display with no reference.
#[test]
fn player_displays_are_not_opened() {
    let mut rig = Rig::new();
    for name in [*b"player1\0", *b"player3\0"] {
        let name = u64::from_le_bytes(name);
        assert!(player_display(name).is_some());
        assert_eq!(
            rig.send(&open_display(name, OPEN_DISPLAY_FLAGS)),
            refused(OPEN_DISPLAY, KernelError::NotFound)
        );
        assert_eq!(
            rig.send(&close_display(name)),
            refused(CLOSE_DISPLAY, KernelError::NotFound)
        );
    }
    rig.ok(&register_client(5, 0, 1));
    let player = u64::from_le_bytes(*b"player2\0");
    assert_eq!(
        rig.send(&bind(5, player)),
        refused(BIND_CLIENT_DISPLAY, KernelError::InvalidState)
    );
}

/// PR's `GetPlayerDisplayIndex` at its edges: `'1'` to `'3'` in the seventh byte, a zero eighth.
#[test]
fn a_player_display_is_player_one_to_three() {
    let name = |bytes: &[u8; 8]| u64::from_le_bytes(*bytes);
    assert_eq!(player_display(name(b"player1\0")), Some(0));
    assert_eq!(player_display(name(b"player3\0")), Some(2));
    assert_eq!(player_display(name(b"player0\0")), None);
    assert_eq!(player_display(name(b"player4\0")), None);
    assert_eq!(player_display(name(b"player1x")), None);
    assert_eq!(player_display(name(b"Player1\0")), None);
    assert_eq!(player_display(DEFAULT_DISPLAY), None);
}

/// PR's `IsKnownDisplayName`: the text up to the first zero byte, so a name with bytes after a
/// zero byte is known by its text; the empty name and a near miss are not.
#[test]
fn a_known_display_is_read_up_to_its_first_zero_byte() {
    let name = |bytes: &[u8; 8]| u64::from_le_bytes(*bytes);
    assert!(is_known_display(DEFAULT_DISPLAY));
    assert!(is_known_display(name(b"overlay0")));
    assert!(is_known_display(name(b"null\0abc")));
    assert!(is_known_display(name(b"#ld01\0\0\0")));
    assert!(!is_known_display(name(b"#ld02\0\0\0")));
    assert!(!is_known_display(name(b"defaulT\0")));
    assert!(!is_known_display(0));
    assert_eq!(KNOWN_DISPLAYS.len(), 42);
    for known in KNOWN_DISPLAYS {
        let mut bytes = [0; 8];
        bytes[..known.len()].copy_from_slice(known.as_bytes());
        assert!(is_known_display(u64::from_le_bytes(bytes)), "{known}");
    }
}

/// 104 and 106 answer only for the open default display; 105, 107 and 354 answer any known name,
/// open or not, and refuse a zero name and an unknown one.
#[test]
fn display_state_answers_by_name() {
    let mut rig = Rig::new();
    let null = u64::from_le_bytes(*b"null\0abc");
    for (type_word, command) in [(0x78, GET_HDR_STATUS), (0x7A, GET_BUFFER_MAX_RESOLUTION)] {
        for name in [DEFAULT_DISPLAY, null, 0] {
            assert_eq!(
                rig.send(&named(type_word, name)),
                result_only(command, VI_NOT_FOUND),
                "{command}"
            );
        }
    }
    let states = [
        (0x79, GET_DISPLAY_STATE_105, vec![0]),
        (0x7B, GET_DISPLAY_STATE_107, vec![1, 0, 1, 0, 0, 0]),
        (0x172, GET_DISPLAY_STATE_354, vec![1, 0]),
    ];
    for (type_word, command, data) in &states {
        assert_eq!(
            rig.send(&named(*type_word, null)),
            success(*command, &[], data)
        );
        assert_eq!(
            rig.send(&named(*type_word, 0)),
            result_only(*command, VI_OUT_OF_RANGE)
        );
        assert_eq!(
            rig.send(&named(*type_word, UNKNOWN)),
            result_only(*command, VI_NOT_FOUND)
        );
    }
    rig.ok(&open_display(DEFAULT_DISPLAY, OPEN_DISPLAY_FLAGS));
    assert_eq!(
        rig.send(&named(0x78, null)),
        result_only(GET_HDR_STATUS, VI_NOT_FOUND)
    );
    assert_eq!(
        rig.send(&named(0x78, DEFAULT_DISPLAY)),
        success(GET_HDR_STATUS, &[], &[1, 0, 0])
    );
}

/// The listed conflict: 301 takes PR's two forms and every one-field change of them PR refuses,
/// on the default display, whose flags the system module's handler masks in; another name,
/// even a known one, is refused, with PR's flags or any.
#[test]
fn display_parameter_301_takes_any_flags_on_the_default_display() {
    let mut rig = Rig::new();
    for (value, mask) in [
        (0, DISPLAY_FLAG_63),
        (DISPLAY_FLAG_63, DISPLAY_FLAG_63),
        (1, DISPLAY_FLAG_63),
        (DISPLAY_FLAG_63 | 1, DISPLAY_FLAG_63),
        (0, DISPLAY_FLAG_63 | 1),
        (0, 0),
        (u64::MAX, u64::MAX),
    ] {
        rig.ok(&parameter_301(DEFAULT_DISPLAY, value, mask));
    }
    let null = u64::from_le_bytes(*b"null\0\0\0\0");
    for name in [UNKNOWN, null, 0] {
        for (value, mask) in [(0, DISPLAY_FLAG_63), (1, 1)] {
            assert_eq!(
                rig.send(&parameter_301(name, value, mask)),
                refused(SET_DISPLAY_PARAMETER_301, KernelError::InvalidCombination)
            );
        }
    }
    assert_eq!(rig.session_state(), Vi2Session::new());
}

// ----- Clients ------------------------------------------------------------------------------------

/// 201 refuses a zero token, a bit at [`CLIENT_NOTIFICATIONS`] on either side and the same bit
/// twice; the last bit is accepted; a registered token and a held bit are `InvalidState`.
#[test]
fn register_client_checks_its_bits() {
    let mut rig = Rig::new();
    let last = CLIENT_NOTIFICATIONS - 1;
    for words in [
        register_client(0, 0, 1),
        register_client(1, CLIENT_NOTIFICATIONS, 0),
        register_client(1, 0, CLIENT_NOTIFICATIONS),
        register_client(1, 6, 6),
        // Both faults: the zero token and a bit past the last.
        register_client(0, CLIENT_NOTIFICATIONS, 0),
    ] {
        assert_eq!(
            rig.send(&words),
            refused(REGISTER_CLIENT_STATE, KernelError::InvalidCombination)
        );
    }
    rig.ok(&register_client(1, last, 0));
    let registered = rig.session_state().clients()[0];
    assert_eq!(registered.bits(), (1 << last) | 1);
    // A fresh client: its fields as 201 read them, every slot free, no display, layer or levels.
    assert_eq!(
        (
            registered.token(),
            registered.retire(),
            registered.available_bit()
        ),
        (1, last, 0)
    );
    assert_eq!(registered.available(), ALL_SLOTS);
    assert!(!registered.is_bound());
    assert_eq!((registered.layer(), registered.levels()), (None, None));
    for words in [
        register_client(1, 2, 3),
        register_client(2, 3, last),
        register_client(2, 0, 4),
    ] {
        assert_eq!(
            rig.send(&words),
            refused(REGISTER_CLIENT_STATE, KernelError::InvalidState),
            "{words:x?}"
        );
    }
    // Both faults: a registered token and a bad bit: the bit's answer.
    assert_eq!(
        rig.send(&register_client(1, 2, 2)),
        refused(REGISTER_CLIENT_STATE, KernelError::InvalidCombination)
    );
}

/// 201 registers up to [`MAX_CLIENTS`] clients, refusing the next with `OutOfResource`, and a
/// clashing one with `InvalidState` even then.
#[test]
fn register_client_holds_at_most_the_most_clients() {
    let mut rig = Rig::new();
    for index in 0..u32::try_from(MAX_CLIENTS).unwrap() {
        rig.ok(&register_client(
            u64::from(index) + 1,
            2 * index,
            2 * index + 1,
        ));
    }
    let more = u64::try_from(MAX_CLIENTS).unwrap() + 1;
    // Every bit is held now, so a new client clashes first.
    assert_eq!(
        rig.send(&register_client(more, 0, 1)),
        refused(REGISTER_CLIENT_STATE, KernelError::InvalidState)
    );
    rig.ok(&client_only(0xDA, 1));
    rig.ok(&register_client(more, 0, 1));
    assert_eq!(rig.session_state().clients().len(), MAX_CLIENTS);
    let mut state = rig.session_state();
    // A full table with a free bit: drop one client's bits by hand through a restore.
    state.clients[0].retire = 0;
    state.clients[0].available_bit = 1;
    rig.replace(state);
    rig.ok(&client_only(0xDA, 2));
    rig.ok(&register_client(more + 1, 2, 3));
    assert_eq!(
        rig.send(&register_client(more + 2, 2, 3)),
        refused(REGISTER_CLIENT_STATE, KernelError::InvalidState)
    );
}

/// 202 forgets the client, closes its endpoint, frees its bits for another, and raises its
/// retire notification; a zero token and an unknown one are refused.
#[test]
fn unregister_client_retires_it() {
    let mut rig = Rig::new();
    let event = rig.notification_event();
    rig.bound_client(0x51, 9, 10);
    rig.ok(&open_endpoint(DEFAULT_DISPLAY, DEFAULT_DISPLAY, 0x51));
    assert_eq!(
        rig.send(&client_only(0xDA, 0)),
        refused(UNREGISTER_CLIENT_STATE, KernelError::InvalidCombination)
    );
    assert_eq!(
        rig.send(&client_only(0xDA, 0x52)),
        refused(UNREGISTER_CLIENT_STATE, KernelError::NotFound)
    );
    rig.ok(&client_only(0xDA, 0x51));
    let state = rig.session_state();
    assert_eq!((state.clients(), state.endpoint()), (&[][..], None));
    assert_eq!(state.notifications()[0], 1 << 9);
    assert_eq!(rig.signals(event), 1);
    rig.ok(&register_client(0x52, 10, 9));
}

/// 212 makes a client's layer on first use, numbered from [`FIRST_LAYER`], and keeps it; it
/// needs the display open, and a layer number past `u64::MAX` is `OutOfResource`.
#[test]
fn availability_makes_the_layer_once() {
    let mut rig = Rig::new();
    rig.ok(&register_client(0x61, 0, 1));
    rig.ok(&register_client(0x62, 2, 3));
    assert_eq!(
        rig.send(&client_only(0xE4, 0x61)),
        refused(GET_CLIENT_AVAILABILITY, KernelError::NotFound)
    );
    assert_eq!(
        rig.send(&client_only(0xE4, 0)),
        refused(GET_CLIENT_AVAILABILITY, KernelError::InvalidCombination)
    );
    rig.ok(&open_display(DEFAULT_DISPLAY, OPEN_DISPLAY_FLAGS));
    for token in [0x62, 0x61, 0x62] {
        assert_eq!(
            rig.send(&client_only(0xE4, token)),
            success(GET_CLIENT_AVAILABILITY, &[], &[0x7FFF, 0])
        );
    }
    let state = rig.session_state();
    assert_eq!(state.clients()[0].layer(), Some(LayerId(FIRST_LAYER + 1)));
    assert_eq!(state.clients()[1].layer(), Some(LayerId(FIRST_LAYER)));
    assert_eq!(state.next_layer(), FIRST_LAYER + 2);

    // The last number is handed out; the one after it is not.
    let mut state = Vi2Session::new();
    state.display_references = 1;
    state.next_layer = u64::MAX - 1;
    state.clients = vec![Client::new(1, 0, 1), Client::new(2, 2, 3)];
    rig.replace(state);
    rig.ok_data(&client_only(0xE4, 1));
    assert_eq!(rig.session_state().next_layer(), u64::MAX);
    assert_eq!(
        rig.send(&client_only(0xE4, 2)),
        refused(GET_CLIENT_AVAILABILITY, KernelError::OutOfResource)
    );
    assert_eq!(rig.session_state().clients()[1].layer(), None);
}

impl Rig {
    /// Sends `words` and expects success, whatever its data.
    fn ok_data(&mut self, words: &[u32]) {
        let reply = self.send(words);
        let at = 8;
        assert_eq!(&reply[at..at + 4], &[0; 4], "{words:x?}");
    }
}

/// 251 binds a registered client to the open default display once; 252 unbinds a bound one.
#[test]
fn bind_and_unbind_follow_prs_checks() {
    let mut rig = Rig::new();
    assert_eq!(
        rig.send(&bind(0x71, DEFAULT_DISPLAY)),
        refused(BIND_CLIENT_DISPLAY, KernelError::NotFound)
    );
    rig.ok(&register_client(0x71, 0, 1));
    assert_eq!(
        rig.send(&bind(0x71, DEFAULT_DISPLAY)),
        refused(BIND_CLIENT_DISPLAY, KernelError::NotFound)
    );
    rig.ok(&open_display(DEFAULT_DISPLAY, OPEN_DISPLAY_FLAGS));
    assert_eq!(
        rig.send(&bind(0x71, UNKNOWN)),
        refused(BIND_CLIENT_DISPLAY, KernelError::NotFound)
    );
    assert_eq!(
        rig.send(&client_only(0x10C, 0x71)),
        refused(UNBIND_CLIENT_DISPLAY, KernelError::NotFound)
    );
    rig.ok(&bind(0x71, DEFAULT_DISPLAY));
    assert!(rig.session_state().clients()[0].is_bound());
    assert_eq!(
        rig.send(&bind(0x71, DEFAULT_DISPLAY)),
        refused(BIND_CLIENT_DISPLAY, KernelError::InvalidState)
    );
    for token in [0, 0x72] {
        assert_eq!(
            rig.send(&client_only(0x10C, token)),
            refused(UNBIND_CLIENT_DISPLAY, KernelError::NotFound)
        );
    }
    rig.ok(&client_only(0x10C, 0x71));
    assert!(!rig.session_state().clients()[0].is_bound());
}

/// 111 opens one endpoint for a bound client of the open display, refusing every other case
/// with `NotFound` and a second with `InvalidState`; 112 checks its names, then the endpoint.
#[test]
fn the_endpoint_opens_once_for_a_bound_client() {
    let mut rig = Rig::new();
    rig.ok(&register_client(0x81, 0, 1));
    let d = DEFAULT_DISPLAY;
    // Not bound, display not open.
    assert_eq!(
        rig.send(&open_endpoint(d, d, 0x81)),
        refused(OPEN_ENDPOINT, KernelError::NotFound)
    );
    rig.ok(&open_display(d, OPEN_DISPLAY_FLAGS));
    assert_eq!(
        rig.send(&open_endpoint(d, d, 0x81)),
        refused(OPEN_ENDPOINT, KernelError::NotFound)
    );
    rig.ok(&bind(0x81, d));
    for words in [
        open_endpoint(d, d, 0),
        open_endpoint(d, d, 0x82),
        open_endpoint(UNKNOWN, UNKNOWN, 0x81),
        open_endpoint(d, UNKNOWN, 0x81),
    ] {
        assert_eq!(
            rig.send(&words),
            refused(OPEN_ENDPOINT, KernelError::NotFound)
        );
    }
    assert_eq!(
        rig.send(&close_endpoint(d, d)),
        refused(CLOSE_ENDPOINT, KernelError::NotFound)
    );
    rig.ok(&open_endpoint(d, d, 0x81));
    assert_eq!(
        rig.send(&open_endpoint(d, d, 0x81)),
        refused(OPEN_ENDPOINT, KernelError::InvalidState)
    );
    for words in [close_endpoint(UNKNOWN, UNKNOWN), close_endpoint(d, UNKNOWN)] {
        assert_eq!(
            rig.send(&words),
            refused(CLOSE_ENDPOINT, KernelError::InvalidCombination)
        );
    }
    assert_eq!(rig.session_state().endpoint(), Some(0x81));
    rig.ok(&close_endpoint(d, d));
    assert_eq!(rig.session_state().endpoint(), None);
    // The display closed: the bound client cannot open one.
    rig.ok(&close_display(d));
    assert_eq!(
        rig.send(&open_endpoint(d, d, 0x81)),
        refused(OPEN_ENDPOINT, KernelError::NotFound)
    );
}

/// 351 takes only [`OUTPUT_ENCODING`]; 352 takes levels within PR's ranges, at their edges, and
/// refuses one past each, a NaN, and an unknown client first.
#[test]
fn output_encoding_and_levels_follow_prs_ranges() {
    let mut rig = Rig::new();
    assert_eq!(
        rig.send(&output_encoding(0x91, OUTPUT_ENCODING)),
        refused(SET_CLIENT_OUTPUT_ENCODING, KernelError::NotFound)
    );
    rig.ok(&register_client(0x91, 0, 1));
    rig.ok(&output_encoding(0x91, OUTPUT_ENCODING));
    for encoding in [OUTPUT_ENCODING - 1, OUTPUT_ENCODING + 1] {
        assert_eq!(
            rig.send(&output_encoding(0x91, encoding)),
            refused(SET_CLIENT_OUTPUT_ENCODING, KernelError::NotImplemented)
        );
    }
    // An unknown client with bad levels: the client's answer.
    assert_eq!(
        rig.send(&output_levels(0x92, [2.0, 0.0, 0.0])),
        refused(SET_CLIENT_OUTPUT_LEVELS, KernelError::NotFound)
    );
    let above = f32::from_bits(1.0_f32.to_bits() + 1);
    let below = -f32::from_bits(1);
    for levels in [
        [above, 0.0, 0.0],
        [0.0, above, 0.0],
        [below, 0.0, 0.0],
        [0.0, below, 0.0],
        [0.0, 0.0, below],
        [f32::NAN, 0.0, 0.0],
        [0.0, 0.0, f32::NAN],
    ] {
        assert_eq!(
            rig.send(&output_levels(0x91, levels)),
            refused(SET_CLIENT_OUTPUT_LEVELS, KernelError::InvalidCombination),
            "{levels:?}"
        );
    }
    assert_eq!(rig.session_state().clients()[0].levels(), None);
    for levels in [[0.0, 0.0, 0.0], [1.0, 1.0, f32::MAX], [0.0, 1.0, 0.8]] {
        rig.ok(&output_levels(0x91, levels));
        assert_eq!(rig.session_state().clients()[0].levels(), Some(levels));
    }
}

/// The monitor kinds: 261 to 272 in pairs, the odd ids registering; none outside.
#[test]
fn monitor_kinds_pair_up() {
    let kinds: Vec<_> = MONITORS.iter().map(|&c| monitor_kind(c).unwrap()).collect();
    assert_eq!(
        kinds,
        [
            (0, true),
            (0, false),
            (1, true),
            (1, false),
            (2, true),
            (2, false),
            (3, true),
            (3, false),
            (4, true),
            (4, false),
            (5, true),
            (5, false)
        ]
    );
    assert_eq!(monitor_kind(260), None);
    assert_eq!(monitor_kind(273), None);
    assert_eq!(monitor_kind(0), None);
}

/// The listed conflict: 11 and 13 take any word (NX2's reading), where PR refuses all but 0;
/// a word with only its high bits set and one with only its low bit set each get the one event.
#[test]
fn notification_commands_take_any_word() {
    let mut rig = Rig::new();
    let event = rig.notification_event();
    for word in [1, 0x8000_0000, u32::MAX] {
        let reply = rig.send(&get_notification_event(word));
        let handle = Rig::handed(&reply);
        assert_eq!(reply, success(GET_NOTIFICATION_EVENT, &[handle], &[]));
        assert_eq!(rig.object(handle), event);
        rig.ok(&release_notification_event(word));
    }
}

/// The listed gap: a closed session's groups keep listening, so after eight sessions that made
/// eight groups each are closed, a new session's 31 answers `OutOfResource`; a group registered
/// for another kind does not listen, so it does not count. Each group's state is what
/// `RequestGroup::listens` says it is.
#[test]
fn closed_sessions_keep_their_groups_listening() {
    let mut rig = Rig::new();
    let per_session = u32::try_from(MAX_LISTENERS / 8).unwrap();
    assert_eq!(per_session, REQUEST_GROUPS);
    for session in 0..8 {
        if session > 0 {
            rig.session = rig.kernel.new_object();
            rig.server.accept(rig.session).unwrap();
        }
        for group in 0..per_session {
            rig.group_event(group);
        }
        if session == 0 {
            // One group of the first session monitors the HDR status, kind 2: it stops listening.
            rig.ok(&monitor(REGISTER_HDR_STATUS_MONITOR, DEFAULT_DISPLAY, 0));
            let state = rig.session_state();
            let listening: Vec<bool> = state.groups().iter().map(RequestGroup::listens).collect();
            let mut expected = vec![true; 8];
            expected[0] = false;
            assert_eq!(listening, expected);
            let monitor = state.groups()[0].monitor().unwrap();
            assert_eq!((monitor.display(), monitor.kind()), (DEFAULT_DISPLAY, 2));
        }
        rig.server.close(rig.session).unwrap();
    }
    assert_eq!(rig.listeners().len(), MAX_LISTENERS - 1);
    rig.session = rig.kernel.new_object();
    rig.server.accept(rig.session).unwrap();
    rig.group_event(0);
    assert_eq!(
        rig.send(&create_group(1)),
        refused(CREATE_REQUEST_GROUP_EVENT, KernelError::OutOfResource)
    );
}
