//! The root driven through raw TIPC messages, over the services framework's fake kernel.
//!
//! Requests are laid out by hand, word by word, from the layouts PR checks
//! (`eden-pr4588/src/core/hle/service/am/service/graphics_supervisor.cpp` and
//! `graphics_supervisor_protocol.h` @ e5fb3b97): each handler compares the header words with
//! literals (`DecodeBootstrap`: `packet[0] != 0x14 || packet[1] != 0x8000000c || packet[2] != 2`),
//! which the tests write as they are, next to the command id they encode. Replies are compared
//! as whole message buffers, laid out by hand the same way.

use std::collections::BTreeSet;

use proptest::prelude::*;
use proptest::test_runner::{Config, RngAlgorithm, RngSeed};
use starquake_ipc::{BUFFER_SIZE, Message, SessionKind, TIPC_COMMAND_OFFSET};
use starquake_kernel::handle::Handle;
use starquake_kernel::id::ObjectId;
use starquake_kernel::memory::{SharedPages, SvcPermission};
use starquake_kernel::result::{KernelError, ResultCode, SvcResult};
use starquake_services::framework::fake::FakeKernel;
use starquake_services::framework::result::INVALID_CMIF_REQUEST;
use starquake_services::framework::{Handled, Kernel, Protocol, Session, UNKNOWN_COMMAND_ID};
use starquake_trace::{EventKind, IpcOutcome, ServiceId};

use super::*;
use crate::protocol::{HANDLE_KIND, Mark, ROOT_COMMANDS};

/// The number the tests' root server is known by.
const ROOT: ServiceId = ServiceId(0x0657);
/// What fills a message buffer before a request is laid into it.
const STALE: u8 = 0xA5;
/// The TIPC header's "a special header follows" bit (PR `DecodeBootstrap`: `packet[1] !=
/// 0x8000000c`, a special header and twelve data words).
const SPECIAL: u32 = 0x8000_0000;
/// A special header that counts one copied handle (PR `DecodeBootstrap`: `packet[2] != 2`).
const ONE_COPY: u32 = 2;

/// A root server, a kernel, and one session on it.
struct Rig {
    kernel: FakeKernel,
    server: Server,
    session: ObjectId,
}

impl Rig {
    fn with(policy: UnknownPolicy) -> Self {
        let mut kernel = FakeKernel::new(0).unwrap();
        kernel.set_current_service(ROOT);
        let session = kernel.new_object();
        let mut server = root_server(policy).with_service_id(ROOT);
        server.accept(session).unwrap();
        Self {
            kernel,
            server,
            session,
        }
    }

    fn new() -> Self {
        Self::with(UnknownPolicy::Strict)
    }

    /// Sends the message `words` lays out on `session`; the message buffer afterwards.
    fn send_on(&mut self, session: ObjectId, words: &[u32]) -> Vec<u8> {
        let mut buffer = vec![STALE; BUFFER_SIZE];
        let bytes = le(words);
        buffer[..bytes.len()].copy_from_slice(&bytes);
        let handled = self.server.handle(&mut self.kernel, session, &mut buffer);
        assert_eq!(handled, Ok(Handled::Replied));
        buffer
    }

    fn send(&mut self, words: &[u32]) -> Vec<u8> {
        self.send_on(self.session, words)
    }

    /// A handle in the caller's table to a fresh object: what stands for the process.
    fn process(&mut self) -> u32 {
        let object = self.kernel.new_object();
        self.kernel.insert_handle(object).unwrap().raw()
    }

    fn state(&self) -> RootState {
        *self.server.family::<RootState>().unwrap()
    }

    /// `Initialize` with a fresh process handle.
    fn initialize(&mut self, request: &BootstrapRequest) -> Vec<u8> {
        let process = self.process();
        self.send(&initialize_words(process, request))
    }

    fn bind_process(&mut self) -> Vec<u8> {
        let process = self.process();
        self.send(&[tipc_type(BIND_PROCESS), SPECIAL, ONE_COPY, process])
    }

    fn complete_bootstrap(&mut self) -> Vec<u8> {
        self.send(&[tipc_type(COMPLETE_BOOTSTRAP), 0])
    }
}

/// Little-endian words as bytes.
fn le(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

/// A whole message buffer that starts with `words` and is zero after them.
fn buffer_of(words: &[u32]) -> Vec<u8> {
    let mut buffer = le(words);
    buffer.resize(BUFFER_SIZE, 0);
    buffer
}

/// A TIPC message's type word for command `id`.
fn tipc_type(id: u32) -> u32 {
    u32::from(TIPC_COMMAND_OFFSET) + id
}

/// The low and high words of a `u64`.
fn halves(value: u64) -> [u32; 2] {
    let bytes = value.to_le_bytes();
    let word = |at: usize| u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
    [word(0), word(4)]
}

/// `Initialize` (4) as PR's `DecodeBootstrap` reads it: `0x14, 0x8000000c, 2`, the process
/// handle at word 3, the applet resource user id at words 4–5, the flags at 6–7 and the four
/// parameters at 8–15.
fn initialize_words(process: u32, request: &BootstrapRequest) -> Vec<u32> {
    // The header PR checks, word for word.
    assert_eq!(tipc_type(INITIALIZE), 0x14);
    let mut words = vec![0x14, 0x8000_000C, 2, process];
    words.extend(halves(request.applet_resource_user_id));
    words.extend(halves(request.flags));
    for parameter in request.parameters {
        words.extend(halves(parameter));
    }
    assert_eq!(words.len(), 16);
    words
}

/// A TIPC reply to command `id` that holds a result and nothing else.
fn result_only(id: u32, result: ResultCode) -> Vec<u8> {
    buffer_of(&[tipc_type(id), 1, result.raw()])
}

/// A successful TIPC reply to `id` with these copied handles and data words: the header (raw
/// size: the result word and the data), the special header when there are handles, the
/// handles, the result, the data.
fn success(id: u32, copies: &[u32], data: &[u32]) -> Vec<u8> {
    let raw = 1 + u32::try_from(data.len()).unwrap();
    let mut words = vec![tipc_type(id)];
    if copies.is_empty() {
        words.push(raw);
    } else {
        words.push(SPECIAL | raw);
        words.push(u32::try_from(copies.len()).unwrap() << 1);
        words.extend(copies);
    }
    words.push(ResultCode::SUCCESS.raw());
    words.extend(data);
    buffer_of(&words)
}

/// The handles a reply copied.
fn copied(buffer: &[u8]) -> Vec<u32> {
    Message::parse_reply(buffer, SessionKind::Session)
        .unwrap()
        .copy_handles
}

/// A local bootstrap request whose fields are all distinct and non-zero except the parameters,
/// which a local bootstrap needs to be zero.
fn local() -> BootstrapRequest {
    BootstrapRequest {
        applet_resource_user_id: 0x0000_0051_0000_00A1,
        flags: bootstrap::LOCAL_FLAGS[0],
        parameters: [0; bootstrap::PARAMETERS],
    }
}

// ----- The command table -----------------------------------------------------------------------

/// What each root command replies, after its result: written out by hand from PR's handlers (the
/// `IPC::ResponseBuilder` and what is pushed after the result; a `u64` is two words), with the
/// spec's mark for it ("The root").
const SHAPES: &[(u32, Mark, &[Handed], u32)] = &[
    (INITIALIZE, Mark::Read, &[], 0),
    (BIND_PROCESS, Mark::Read, &[], 0),
    (COMPLETE_BOOTSTRAP, Mark::Read, &[], 0),
    (QUERY_MEMORY_MODE, Mark::Read, &[], 1),
    (CONFIGURE_NATIVE_MEMORY, Mark::Read, &[], 2),
    (INITIALIZE_SYNCPOINT_CONTEXT, Mark::Read, &[], 0),
    (GET_SYNCPOINT_SESSION, Mark::Observed, &[Handed::Session], 0),
    (
        BIND_SYNCPOINT_MEMORY,
        Mark::Observed,
        &[Handed::Event, Handed::Event],
        0,
    ),
    (INITIALIZE_DISPLAY_CONTEXT, Mark::Read, &[], 0),
    (FINALIZE_DISPLAY_CONTEXT, Mark::Read, &[], 0),
    (OPEN_DISPLAY_SESSION, Mark::Observed, &[Handed::Session], 0),
    (INITIALIZE_GPU_CONTEXT, Mark::Read, &[], 0),
    (ACTIVATE_GPU_CONTEXT, Mark::Read, &[], 0),
    (REGISTER_GPU_ADDRESS_SPACE, Mark::Read, &[], 0),
    (OPEN_GPU_SESSION, Mark::Observed, &[Handed::Session], 0),
    (GET_NVIDIA_DRIVER, Mark::Observed, &[Handed::Session], 0),
    (CREATE_NATIVE_MEMORY, Mark::Observed, &[], 1),
    (DESTROY_NATIVE_MEMORY, Mark::Observed, &[], 0),
    (BIND_NATIVE_MEMORY, Mark::Observed, &[], 0),
    (BIND_NATIVE_GPU_MEMORY, Mark::Observed, &[], 0),
    (CREATE_NATIVE_IMAGE, Mark::Observed, &[], 1),
    (DESTROY_NATIVE_IMAGE, Mark::Observed, &[], 0),
    (CALCULATE_NATIVE_IMAGE_SIZE, Mark::Observed, &[], 2),
    (CALCULATE_IMAGE_REQUIREMENTS, Mark::Observed, &[], 4),
    (
        DESTROY_NATIVE_WINDOW_ALIAS_IMAGE_POOL,
        Mark::Observed,
        &[],
        0,
    ),
    (
        CREATE_NATIVE_WINDOW_ALIAS_IMAGE_POOL,
        Mark::Observed,
        &[],
        0,
    ),
    (MAP_NATIVE_GPU_IMAGE, Mark::Read, &[], 37),
    (UNMAP_NATIVE_GPU_IMAGE, Mark::Read, &[], 0),
    (QUERY_NATIVE_GPU_IMAGE, Mark::Read, &[], 37),
    (INITIALIZE_VIDEO_DECODER, Mark::Observed, &[], 0),
    (
        OPEN_VIDEO_DECODER_CONTEXT_902,
        Mark::Observed,
        &[Handed::Session],
        0,
    ),
    (FINALIZE_VIDEO_DECODER, Mark::Observed, &[], 0),
    (
        OPEN_VIDEO_DECODER_CONTEXT,
        Mark::Observed,
        &[Handed::Session],
        0,
    ),
    (CREATE_VIDEO_IMAGE_POOL, Mark::Observed, &[], 1),
    (DESTROY_VIDEO_IMAGE_POOL, Mark::Observed, &[], 0),
    (QUERY_VIDEO_IMAGE, Mark::Observed, &[], 32),
    (CALCULATE_VIDEO_IMAGE_REQUIREMENTS, Mark::Observed, &[], 4),
    (INITIALIZE_VIDEO_ENCODER, Mark::Conflict, &[], 0),
    (
        OPEN_VIDEO_ENCODER_CONTEXT,
        Mark::Conflict,
        &[Handed::Session],
        0,
    ),
    (FINALIZE_VIDEO_ENCODER, Mark::Conflict, &[], 0),
    (INITIALIZE_FRAME_CONTEXT, Mark::Read, &[], 0),
    (ACTIVATE_FRAME_CONTEXT, Mark::Read, &[], 0),
    (SET_GRAPHICS_CLASS_1111, Mark::Read, &[], 0),
    (GET_NATIVE_EVENT_1121, Mark::Read, &[Handed::Session], 0),
    (SEQUENCE_COMMAND_1201, Mark::Read, &[], 0),
    (SEQUENCE_COMMAND_1202, Mark::Read, &[Handed::Session], 0),
    (SEQUENCE_COMMAND_1203, Mark::Read, &[], 0),
    (CREATE_CHAT_IMAGE_POOL, Mark::Read, &[], 1),
    (CALCULATE_REQUIRED_BUFFER_MEMORY, Mark::Read, &[], 4),
    (INITIALIZE_IMAGE_CONTEXT_1501, Mark::Read, &[], 0),
    (INITIALIZE_IMAGE_CONTEXT_1502, Mark::Read, &[], 0),
    (CREATE_CAMERA_CONTEXT, Mark::Read, &[], 0),
    (OPEN_CAMERA_SESSION, Mark::Read, &[Handed::Session], 0),
];

/// The table, the hand-written shapes and the root's dispatch name the same commands, each
/// once, with the same marks and shapes.
#[test]
fn the_table_the_shapes_and_the_dispatch_list_the_same_commands() {
    let table: Vec<u32> = ROOT_COMMANDS.iter().map(|c| c.id).collect();
    let shapes: Vec<u32> = SHAPES.iter().map(|s| s.0).collect();
    let dispatch = GsvRoot.commands();
    let tipc: Vec<u32> = dispatch.tipc.iter().map(|c| c.id).collect();
    let distinct: BTreeSet<u32> = table.iter().copied().collect();
    assert_eq!(distinct.len(), table.len(), "an id is listed twice");
    assert_eq!(distinct, shapes.iter().copied().collect());
    assert_eq!(distinct, tipc.iter().copied().collect());
    assert_eq!(table.len(), shapes.len());
    assert_eq!(table.len(), tipc.len());
    // Every root command is TIPC: the CMIF list is empty.
    assert_eq!(dispatch.cmif, []);
    assert_eq!(Placeholder::default().commands().tipc, []);
    assert_eq!(Placeholder::default().commands().cmif, []);
    for &(id, mark, handles, words) in SHAPES {
        let row = root_command(id).unwrap();
        assert_eq!(row.mark, mark, "{id}");
        assert_eq!(row.reply.handles, handles, "{id}");
        assert_eq!(row.reply.words, words, "{id}");
        assert!(!row.name.is_empty() && !row.source.is_empty(), "{id}");
    }
    assert_eq!(root_command(5), None);
    assert_eq!(HANDLE_KIND, Mark::Conflict);
}

/// Every root command's reply, as whole message buffers, sent on a root no bootstrap command
/// has reached: before the bootstrap, nothing is refused (decision 7). Each session it hands out
/// is a placeholder on the root's server, and each event a kernel event, unsignalled.
#[test]
fn every_root_command_s_reply_bytes() {
    let mut rig = Rig::new();
    for &(id, _, handles, words) in SHAPES {
        let request = match id {
            INITIALIZE => initialize_words(rig.process(), &local()),
            BIND_PROCESS => vec![0x1B, SPECIAL, ONE_COPY, rig.process()],
            // PR `ConfigureNativeMemory`: `packet[0] == 0x83 && packet[1] == 2`.
            CONFIGURE_NATIVE_MEMORY => vec![0x83, 2, 0, 0],
            _ => vec![tipc_type(id), 0],
        };
        let reply = rig.send(&request);
        let copies = copied(&reply);
        let data = vec![0; usize::try_from(words).unwrap()];
        assert_eq!(reply, success(id, &copies, &data), "{id}");
        assert_eq!(copies.len(), handles.len(), "{id}");
        for (handle, handed) in copies.iter().zip(handles) {
            let object = rig
                .kernel
                .resolve_handle(Handle::from_raw(*handle))
                .unwrap();
            match handed {
                Handed::Session => {
                    assert_eq!(rig.kernel.owner(object), Some(ROOT), "{id}");
                    // It answers as a placeholder: by the server's policy.
                    let asked = rig.send_on(object, &[tipc_type(0), 0]);
                    assert_eq!(asked, result_only(0, UNKNOWN_COMMAND_ID), "{id}");
                }
                Handed::Event => assert_eq!(rig.kernel.signals(object), Some(0), "{id}"),
            }
        }
    }
    // The header words written as PR checks them are the ones the ids make.
    assert_eq!(tipc_type(BIND_PROCESS), 0x1B);
    assert_eq!(tipc_type(CONFIGURE_NATIVE_MEMORY), 0x83);
}

#[test]
fn a_command_the_table_does_not_list_goes_to_the_policy() {
    // 5 is NX2's `Unknown5`; 904 and 905 fall in the spec's "901–906" but no reference has them.
    for id in [5, 904, 905] {
        let mut rig = Rig::new();
        let reply = rig.send(&[tipc_type(id), 0]);
        assert_eq!(reply, result_only(id, UNKNOWN_COMMAND_ID), "{id}");
        let mut rig = Rig::with(UnknownPolicy::AutoStub);
        let reply = rig.send(&[tipc_type(id), 0]);
        assert_eq!(reply, result_only(id, ResultCode::SUCCESS), "{id}");
        assert_eq!(rig.server.stubbed().count(), 1, "{id}");
    }
}

// ----- The bootstrap ---------------------------------------------------------------------------

#[test]
fn the_bootstrap_in_order_reaches_ready_and_records_the_request() {
    let mut rig = Rig::new();
    assert_eq!(rig.state(), RootState::new());
    assert_eq!(rig.initialize(&local()), success(INITIALIZE, &[], &[]));
    assert_eq!(rig.state().phase(), Phase::Initialized);
    assert_eq!(rig.state().bootstrap(), Some(&local()));
    assert_eq!(rig.bind_process(), success(BIND_PROCESS, &[], &[]));
    assert_eq!(rig.state().phase(), Phase::ProcessBound);
    assert_eq!(
        rig.complete_bootstrap(),
        success(COMPLETE_BOOTSTRAP, &[], &[])
    );
    assert_eq!(rig.state().phase(), Phase::Ready);
    assert_eq!(rig.state().out_of_order(), 0);
    // Once ready, the memory modes answer as before it.
    let mode = rig.send(&[tipc_type(QUERY_MEMORY_MODE), 0]);
    assert_eq!(
        mode,
        success(QUERY_MEMORY_MODE, &[], &[bootstrap::NO_REMOTE_MEMORY_MODE])
    );
    // PR `QueryMemoryMode`: `packet[0] == 0x8e && packet[1] == 0`.
    assert_eq!(tipc_type(QUERY_MEMORY_MODE), 0x8E);
}

/// Decision 7: each bootstrap command out of PR's order is answered with success, counted and
/// logged, and still moves the phase to its own. Each command is seen both in order (the count
/// unchanged) and out of it (the count up by one).
#[test]
fn bootstrap_out_of_order_is_logged_not_refused() {
    let mut rig = Rig::new();
    // 101 first: PR wants `ProcessBound`.
    assert_eq!(
        rig.complete_bootstrap(),
        success(COMPLETE_BOOTSTRAP, &[], &[])
    );
    assert_eq!(
        (rig.state().phase(), rig.state().out_of_order()),
        (Phase::Ready, 1)
    );
    // 11 after 101: PR wants `Initialized`.
    assert_eq!(rig.bind_process(), success(BIND_PROCESS, &[], &[]));
    assert_eq!(
        (rig.state().phase(), rig.state().out_of_order()),
        (Phase::ProcessBound, 2)
    );
    // 4 after 11: PR wants `Created`.
    assert_eq!(rig.initialize(&local()), success(INITIALIZE, &[], &[]));
    assert_eq!(
        (rig.state().phase(), rig.state().out_of_order()),
        (Phase::Initialized, 3)
    );
    assert_eq!(rig.state().bootstrap(), Some(&local()));
    // From there the order is PR's again: 11 and 101 count nothing.
    rig.bind_process();
    rig.complete_bootstrap();
    assert_eq!(
        (rig.state().phase(), rig.state().out_of_order()),
        (Phase::Ready, 3)
    );
    // A second `Initialize` replaces the request and counts.
    let again = BootstrapRequest {
        flags: bootstrap::LOCAL_FLAGS[2],
        ..local()
    };
    assert_eq!(rig.initialize(&again), success(INITIALIZE, &[], &[]));
    assert_eq!(rig.state().bootstrap(), Some(&again));
    assert_eq!(
        (rig.state().phase(), rig.state().out_of_order()),
        (Phase::Initialized, 4)
    );
}

#[test]
fn the_out_of_order_count_stops_at_its_limit() {
    for (count, after) in [(u32::MAX - 1, u32::MAX), (u32::MAX, u32::MAX)] {
        let mut state = RootState {
            out_of_order: count,
            ..RootState::new()
        };
        state.step(COMPLETE_BOOTSTRAP);
        assert_eq!(state.out_of_order(), after, "{count}");
        assert_eq!(state.phase(), Phase::Ready);
    }
}

/// A command that is not a bootstrap command leaves the phase and the count alone.
#[test]
fn only_the_three_bootstrap_commands_step_the_phase() {
    for row in ROOT_COMMANDS {
        let mut state = RootState::new();
        state.step(row.id);
        let stepped = [INITIALIZE, BIND_PROCESS, COMPLETE_BOOTSTRAP].contains(&row.id);
        assert_eq!(state.phase() != Phase::Created, stepped, "{}", row.id);
        // Only 4 starts in the phase PR wants it in.
        let counted = stepped && row.id != INITIALIZE;
        assert_eq!(state.out_of_order(), u32::from(counted), "{}", row.id);
    }
}

/// The conflict the module docs list: PR answers `ResultNotImplemented`, NX2 success. The
/// request is kept with its mode either way. Each local-flag edge and each parameter is tried.
#[test]
fn a_remote_memory_bootstrap_is_accepted_and_recorded() {
    let [first, .., last] = bootstrap::LOCAL_FLAGS;
    let mut cases = vec![
        (first - 1, [0; bootstrap::PARAMETERS], false),
        (first, [0; bootstrap::PARAMETERS], true),
        (last, [0; bootstrap::PARAMETERS], true),
        (last + 1, [0; bootstrap::PARAMETERS], false),
    ];
    for at in 0..bootstrap::PARAMETERS {
        let mut parameters = [0; bootstrap::PARAMETERS];
        parameters[at] = 1;
        cases.push((first, parameters, false));
    }
    for (flags, parameters, is_local) in cases {
        let mut rig = Rig::new();
        let request = BootstrapRequest {
            flags,
            parameters,
            ..local()
        };
        assert_eq!(rig.initialize(&request), success(INITIALIZE, &[], &[]));
        let recorded = *rig.state().bootstrap().unwrap();
        assert_eq!(recorded, request);
        assert_eq!(recorded.is_local(), is_local, "{flags:#x} {parameters:?}");
        assert_eq!(rig.state().phase(), Phase::Initialized);
    }
}

#[test]
fn the_gpu_profile_is_prs_for_each_flags_word() {
    let [five, six, seven] = bootstrap::LOCAL_FLAGS;
    let cases = [
        (five - 1, None),
        (five, Some(bootstrap::GPU_PROFILE_FLAGS_5)),
        (six, Some(bootstrap::GPU_PROFILE_FLAGS_6_AND_7)),
        (seven, Some(bootstrap::GPU_PROFILE_FLAGS_6_AND_7)),
        (seven + 1, None),
        (0, None),
    ];
    for (flags, profile) in cases {
        let request = BootstrapRequest { flags, ..local() };
        assert_eq!(request.gpu_profile(), profile, "{flags:#x}");
    }
    // PR's numbers, which the constants hold: `return 0x807`, `return 0x808`.
    assert_eq!(bootstrap::GPU_PROFILE_FLAGS_5, 0x807);
    assert_eq!(bootstrap::GPU_PROFILE_FLAGS_6_AND_7, 0x808);
}

/// The listed divergence: the root does not know the process or the applet resource user id,
/// so any object and any id are taken.
#[test]
fn the_bootstrap_takes_any_object_and_any_user_id() {
    let mut rig = Rig::new();
    let request = BootstrapRequest {
        applet_resource_user_id: u64::MAX,
        ..local()
    };
    // The session's own handle stands for the process.
    let session = rig.kernel.insert_handle(rig.session).unwrap().raw();
    let reply = rig.send(&initialize_words(session, &request));
    assert_eq!(reply, success(INITIALIZE, &[], &[]));
    assert_eq!(rig.state().bootstrap(), Some(&request));
}

/// Twelve data words are what 4 reads: eleven is too few (the data is read before anything is
/// recorded, so nothing is), thirteen is read. A request with no handle is refused too.
#[test]
fn a_short_bootstrap_request_is_refused_and_a_longer_one_is_read() {
    let mut rig = Rig::new();
    let process = rig.process();
    let full = initialize_words(process, &local());

    let mut short = full.clone();
    short.pop();
    short[1] = SPECIAL | 0xB;
    assert_eq!(
        rig.send(&short),
        result_only(INITIALIZE, INVALID_CMIF_REQUEST)
    );
    assert_eq!(rig.state(), RootState::new());

    // The same twelve words with no special header and no handle.
    let mut no_handle = vec![tipc_type(INITIALIZE), 12];
    no_handle.extend(&full[4..]);
    assert_eq!(
        rig.send(&no_handle),
        result_only(INITIALIZE, INVALID_CMIF_REQUEST)
    );
    assert_eq!(rig.state(), RootState::new());

    let mut long = full.clone();
    long.push(0xFFFF_FFFF);
    long[1] = SPECIAL | 0xD;
    assert_eq!(rig.send(&long), success(INITIALIZE, &[], &[]));
    assert_eq!(rig.state().bootstrap(), Some(&local()));

    // `BindProcess` without its handle.
    let reply = rig.send(&[tipc_type(BIND_PROCESS), 0]);
    assert_eq!(reply, result_only(BIND_PROCESS, INVALID_CMIF_REQUEST));
    assert_eq!(rig.state().phase(), Phase::Initialized);
}

/// The handle is checked before the data, as PR checks the process before the fields: a
/// request with both faults gets the handle's error.
#[test]
fn the_handle_is_checked_before_the_data() {
    let mut rig = Rig::new();
    let names_nothing = Handle::INVALID.raw();
    let mut both = initialize_words(names_nothing, &local());
    both.pop();
    both[1] = SPECIAL | 0xB;
    let reply = rig.send(&both);
    assert_eq!(
        reply,
        result_only(INITIALIZE, KernelError::InvalidHandle.code())
    );
    // Each fault alone.
    let reply = rig.send(&initialize_words(names_nothing, &local()));
    assert_eq!(
        reply,
        result_only(INITIALIZE, KernelError::InvalidHandle.code())
    );
    assert_eq!(rig.state(), RootState::new());
}

/// `ConfigureNativeMemory` records the mode it is given, whatever it is, and answers zero.
#[test]
fn configure_native_memory_records_any_mode() {
    for mode in [0, 2, 1, 3, u64::MAX] {
        let mut rig = Rig::new();
        let [low, high] = halves(mode);
        let reply = rig.send(&[tipc_type(CONFIGURE_NATIVE_MEMORY), 2, low, high]);
        let [zero_low, zero_high] = halves(bootstrap::NO_NATIVE_MEMORY_MODE);
        assert_eq!(
            reply,
            success(CONFIGURE_NATIVE_MEMORY, &[], &[zero_low, zero_high]),
            "{mode:#x}"
        );
        assert_eq!(rig.state().native_memory_mode(), Some(mode));
    }
    // With no mode at all it is refused, and nothing is recorded.
    let mut rig = Rig::new();
    let reply = rig.send(&[tipc_type(CONFIGURE_NATIVE_MEMORY), 1, 2]);
    assert_eq!(
        reply,
        result_only(CONFIGURE_NATIVE_MEMORY, INVALID_CMIF_REQUEST)
    );
    assert_eq!(rig.state().native_memory_mode(), None);
}

fn config() -> ProptestConfig {
    ProptestConfig {
        cases: 256,
        failure_persistence: None,
        rng_seed: RngSeed::Fixed(0x4753_5652_4F4F_5431),
        rng_algorithm: RngAlgorithm::ChaCha,
        ..Config::default()
    }
}

proptest! {
    #![proptest_config(config())]

    /// Any request laid out as PR reads it is recorded field for field: swapping any two reads
    /// (or reading a parameter out of place) records a different request.
    #[test]
    fn a_bootstrap_request_is_read_field_for_field(
        applet_resource_user_id in any::<u64>(),
        flags in any::<u64>(),
        parameters in any::<[u64; bootstrap::PARAMETERS]>(),
    ) {
        let request = BootstrapRequest { applet_resource_user_id, flags, parameters };
        let mut rig = Rig::new();
        prop_assert_eq!(rig.initialize(&request), success(INITIALIZE, &[], &[]));
        prop_assert_eq!(rig.state().bootstrap().copied(), Some(request));
    }
}

/// One request with every field distinct, so a swap shows even without the generated cases.
#[test]
fn a_bootstrap_request_with_distinct_fields() {
    let request = BootstrapRequest {
        applet_resource_user_id: 0x1111_1111_2222_2222,
        flags: 0x3333_3333_4444_4444,
        parameters: [
            0x5555_5555_6666_6666,
            0x7777_7777_8888_8888,
            0x9999_9999_AAAA_AAAA,
            0xBBBB_BBBB_CCCC_CCCC,
        ],
    };
    let mut rig = Rig::new();
    rig.initialize(&request);
    assert_eq!(rig.state().bootstrap(), Some(&request));
}

// ----- Stubs -----------------------------------------------------------------------------------

/// 204 makes its two events once and hands the same two out again.
#[test]
fn the_syncpoint_memory_stub_hands_out_the_same_two_events() {
    let mut rig = Rig::new();
    let words = [tipc_type(BIND_SYNCPOINT_MEMORY), 0];
    let first = copied(&rig.send(&words));
    let second = copied(&rig.send(&words));
    let objects = |rig: &Rig, handles: &[u32]| -> Vec<ObjectId> {
        let resolve = |h: &u32| rig.kernel.resolve_handle(Handle::from_raw(*h)).unwrap();
        handles.iter().map(resolve).collect()
    };
    let made = objects(&rig, &first);
    assert_eq!(made.len(), 2);
    assert_ne!(made[0], made[1]);
    assert_eq!(objects(&rig, &second), made);
    assert_eq!(
        rig.state().syncpoint_events(),
        SyncpointEvents::Both(made[0], made[1])
    );
}

/// When the kernel refuses the second event, the reply is its error and the first is kept,
/// then handed out first once the kernel makes the second.
#[test]
fn a_refused_second_event_keeps_the_first() {
    let mut rig = Rig::new();
    let placeholder = FakeKernel::new(0).unwrap();
    let mut kernel = RefusesAfter {
        kernel: std::mem::replace(&mut rig.kernel, placeholder),
        events_left: 1,
    };
    let mut buffer = vec![STALE; BUFFER_SIZE];
    let request = le(&[tipc_type(BIND_SYNCPOINT_MEMORY), 0]);
    buffer[..request.len()].copy_from_slice(&request);
    let handled = rig.server.handle(&mut kernel, rig.session, &mut buffer);
    assert_eq!(handled, Ok(Handled::Replied));
    rig.kernel = kernel.kernel;
    assert_eq!(
        buffer,
        result_only(BIND_SYNCPOINT_MEMORY, KernelError::OutOfResource.code())
    );
    let SyncpointEvents::One(first) = rig.state().syncpoint_events() else {
        unreachable!("one event kept: {:?}", rig.state().syncpoint_events());
    };
    assert_eq!(rig.kernel.signals(first), Some(0));

    let reply = rig.send(&[tipc_type(BIND_SYNCPOINT_MEMORY), 0]);
    let copies = copied(&reply);
    let object = |at: usize| {
        let handle = Handle::from_raw(copies[at]);
        rig.kernel.resolve_handle(handle).unwrap()
    };
    assert_eq!(object(0), first);
    assert_eq!(
        rig.state().syncpoint_events(),
        SyncpointEvents::Both(first, object(1))
    );
}

/// A [`FakeKernel`] that makes `events_left` events and then refuses with
/// [`KernelError::OutOfResource`]: how a test reaches the failure between 204's two events.
struct RefusesAfter {
    kernel: FakeKernel,
    events_left: u32,
}

impl Kernel for RefusesAfter {
    fn process_id(&self) -> u64 {
        self.kernel.process_id()
    }

    fn resolve_handle(&self, handle: Handle) -> SvcResult<ObjectId> {
        self.kernel.resolve_handle(handle)
    }

    fn insert_handle(&mut self, object: ObjectId) -> SvcResult<Handle> {
        self.kernel.insert_handle(object)
    }

    fn create_event(&mut self) -> SvcResult<ObjectId> {
        if self.events_left == 0 {
            return Err(KernelError::OutOfResource);
        }
        self.events_left -= 1;
        self.kernel.create_event()
    }

    fn signal_event(&mut self, event: ObjectId) -> SvcResult<()> {
        self.kernel.signal_event(event)
    }

    fn create_shared_memory(
        &mut self,
        size: u64,
        permission: SvcPermission,
    ) -> SvcResult<(ObjectId, SharedPages)> {
        self.kernel.create_shared_memory(size, permission)
    }

    fn create_session(&mut self) -> SvcResult<(ObjectId, Handle)> {
        self.kernel.create_session()
    }

    fn connect(&mut self, service: ServiceId) -> SvcResult<(ObjectId, Handle)> {
        self.kernel.connect(service)
    }

    fn read_memory(&self, address: u64, out: &mut [u8]) -> SvcResult<()> {
        self.kernel.read_memory(address, out)
    }

    fn write_memory(&mut self, address: u64, data: &[u8]) -> SvcResult<()> {
        self.kernel.write_memory(address, data)
    }

    fn record(&mut self, kind: EventKind) {
        self.kernel.record(kind);
    }
}

/// Each session a stub hands out is a new one, recording the command that made it.
#[test]
fn each_session_stub_hands_out_a_new_placeholder() {
    let mut rig = Rig::new();
    let words = [tipc_type(OPEN_GPU_SESSION), 0];
    let first = copied(&rig.send(&words))[0];
    let second = copied(&rig.send(&words))[0];
    assert_ne!(first, second);
    // The first session, and one per placeholder.
    assert_eq!(rig.server.session_count(), 3);
    let saved = rig.server.save().unwrap();
    let name = b"Placeholder";
    let count = saved.windows(name.len()).filter(|w| w == name).count();
    assert_eq!(count, 2);
    let placeholder = Placeholder {
        command: OPEN_GPU_SESSION,
    };
    assert_eq!(placeholder.command(), OPEN_GPU_SESSION);
}

// ----- Recording and snapshots -----------------------------------------------------------------

/// Every request is one IPC event of the run's stream, with the root's number, the command, its
/// result and "done": the bootstrap out of order too.
#[test]
fn each_request_is_recorded() {
    let mut rig = Rig::new();
    rig.complete_bootstrap();
    rig.initialize(&local());
    rig.send(&[tipc_type(OPEN_DISPLAY_SESSION), 0]);
    rig.send(&[tipc_type(5), 0]);
    let recorded: Vec<_> = rig
        .kernel
        .recorded()
        .into_iter()
        .filter_map(|event| match event.kind {
            EventKind::Ipc {
                service,
                command,
                result,
                outcome,
                ..
            } => Some((service, command, result.0, outcome)),
            _ => None,
        })
        .collect();
    let done = |command| (ROOT, command, ResultCode::SUCCESS.raw(), IpcOutcome::Done);
    assert_eq!(
        recorded,
        [
            done(COMPLETE_BOOTSTRAP),
            done(INITIALIZE),
            done(OPEN_DISPLAY_SESSION),
            (ROOT, 5, UNKNOWN_COMMAND_ID.raw(), IpcOutcome::Done),
        ]
    );
}

/// A root saved mid-bootstrap, with a placeholder and 204's events out, restores to the same
/// state, and the restored server carries on from it: the next bootstrap command counts as in
/// order, 204 hands out the saved events, and the placeholder still answers.
#[test]
fn a_root_saved_mid_bootstrap_restores_and_carries_on() {
    let mut rig = Rig::new();
    rig.complete_bootstrap();
    rig.initialize(&local());
    let [low, high] = halves(2);
    rig.send(&[tipc_type(CONFIGURE_NATIVE_MEMORY), 2, low, high]);
    let events = copied(&rig.send(&[tipc_type(BIND_SYNCPOINT_MEMORY), 0]));
    let placeholder = copied(&rig.send(&[tipc_type(GET_NVIDIA_DRIVER), 0]))[0];
    let placeholder = rig
        .kernel
        .resolve_handle(Handle::from_raw(placeholder))
        .unwrap();
    let state = rig.state();
    // 101 in `Created` and 4 in `Ready` were both out of order.
    assert_eq!(
        (state.phase(), state.out_of_order()),
        (Phase::Initialized, 2)
    );
    let saved = rig.server.save().unwrap();

    let mut restored = root_server(UnknownPolicy::Strict).with_service_id(ROOT);
    restored.restore(&saved).unwrap();
    assert_eq!(restored.family::<RootState>(), Some(&state));
    assert_eq!(restored.save().unwrap(), saved);
    rig.server = restored;

    // In order from the restored phase: nothing more is counted.
    assert_eq!(rig.bind_process(), success(BIND_PROCESS, &[], &[]));
    assert_eq!(
        rig.complete_bootstrap(),
        success(COMPLETE_BOOTSTRAP, &[], &[])
    );
    assert_eq!(
        (rig.state().phase(), rig.state().out_of_order()),
        (Phase::Ready, 2)
    );
    assert_eq!(rig.state().native_memory_mode(), Some(2));
    assert_eq!(rig.state().bootstrap(), Some(&local()));
    let again = copied(&rig.send(&[tipc_type(BIND_SYNCPOINT_MEMORY), 0]));
    let object = |rig: &Rig, h: u32| rig.kernel.resolve_handle(Handle::from_raw(h)).unwrap();
    assert_eq!(object(&rig, again[0]), object(&rig, events[0]));
    assert_eq!(object(&rig, again[1]), object(&rig, events[1]));
    let asked = rig.send_on(placeholder, &[tipc_type(0), 0]);
    assert_eq!(asked, result_only(0, UNKNOWN_COMMAND_ID));
}

#[test]
fn the_server_is_named_and_serves_tipc_only() {
    let rig = Rig::new();
    assert_eq!(rig.server.name(), ROOT_NAME);
    assert_eq!(GsvRoot.commands().name(Protocol::Cmif, INITIALIZE), None);
    assert_eq!(
        GsvRoot.commands().name(Protocol::Tipc, INITIALIZE),
        Some("initialize")
    );
    assert_eq!(Phase::Ready.word(), "ready");
    assert_eq!(Mark::Conflict.word(), "conflict");
}
