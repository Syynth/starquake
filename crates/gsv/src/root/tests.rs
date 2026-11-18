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
use starquake_ipc::{BUFFER_SIZE, CommandType, RequestBuilder, SFCO_MAGIC};
use starquake_kernel::handle::Handle;
use starquake_kernel::id::ObjectId;
use starquake_kernel::memory::{SharedPages, SvcPermission};
use starquake_kernel::result::{KernelError, ResultCode, SvcResult};
use starquake_kernel::slab::SlabId;
use starquake_services::framework::fake::FakeKernel;
use starquake_services::framework::result::INVALID_CMIF_REQUEST;
use starquake_services::framework::{
    Family, Handled, Kernel, Protocol, Session, SnapshotError, StubbedCommand, UNKNOWN_COMMAND_ID,
};
use starquake_trace::{EventKind, IpcOutcome, ServiceId};

use super::*;
use crate::protocol::{HANDLE_KIND, Mark, ROOT_COMMANDS};
use crate::testing::{
    SPECIAL, STALE, buffer_of, copied, halves, le, result_only, success, tipc_type,
};

/// The number the tests' root server is known by.
const ROOT: ServiceId = ServiceId(0x0657);
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
        *self.server.family::<RootFamily>().unwrap().state()
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
                    // It records the command that handed it out: the snapshot holds a
                    // placeholder whose saved state is this row's id.
                    let saved = rig.server.save().unwrap();
                    let row = saved_placeholder(id);
                    let found = saved.windows(row.len()).any(|w| w == row.as_slice());
                    assert!(found, "no placeholder saved with command {id}");
                }
                Handed::Event => assert_eq!(rig.kernel.signals(object), Some(0), "{id}"),
            }
        }
    }
    // The header words written as PR checks them are the ones the ids make.
    assert_eq!(tipc_type(BIND_PROCESS), 0x1B);
    assert_eq!(tipc_type(CONFIGURE_NATIVE_MEMORY), 0x83);
}

/// A placeholder session as a server snapshot holds it: its interface's name and its saved
/// state, each with its length (`SavedTarget::Interface` in the framework's `targets.rs`), for
/// a placeholder handed out by command `id`.
fn saved_placeholder(id: u32) -> Vec<u8> {
    let name = Placeholder::interface().as_bytes();
    let state = Placeholder { command: id }.save().unwrap();
    let mut row = vec![u8::try_from(name.len()).unwrap()];
    row.extend(name);
    row.push(u8::try_from(state.len()).unwrap());
    row.extend(state);
    row
}

/// A CMIF request to the root, with a TIPC command's id in it, is answered by the server's
/// unknown-command policy: the root's CMIF table is empty. The replies are CMIF replies on a
/// plain session: the header (raw size 6), padding, `SFCO`, the version, the result, and the
/// token.
#[test]
fn a_cmif_request_to_the_root_goes_to_the_policy() {
    for (policy, result) in [
        (UnknownPolicy::Strict, UNKNOWN_COMMAND_ID),
        (UnknownPolicy::AutoStub, ResultCode::SUCCESS),
    ] {
        let mut rig = Rig::with(policy);
        let mut buffer = vec![STALE; BUFFER_SIZE];
        RequestBuilder::cmif(CommandType::Request, INITIALIZE)
            .build(&mut buffer)
            .unwrap();
        let handled = rig.server.handle(&mut rig.kernel, rig.session, &mut buffer);
        assert_eq!(handled, Ok(Handled::Replied), "{policy:?}");
        let expected = buffer_of(&[0, 6, 0, 0, SFCO_MAGIC, 0, result.raw(), 0]);
        assert_eq!(buffer, expected, "{policy:?}");
        // Not the bootstrap's `Initialize`: nothing is recorded.
        assert_eq!(rig.state(), RootState::new(), "{policy:?}");
        let stubbed: Vec<_> = rig.server.stubbed().copied().collect();
        let expected = match policy {
            UnknownPolicy::Strict => vec![],
            UnknownPolicy::AutoStub => vec![StubbedCommand {
                interface: starquake_trace::InterfaceName::lossy(GsvRoot.name()),
                protocol: starquake_trace::IpcProtocol::Cmif,
                id: INITIALIZE,
            }],
        };
        assert_eq!(stubbed, expected, "{policy:?}");
    }
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

/// The listed divergence from NX2 ("The server"): a second root session, as a second 2000 makes,
/// shares the first one's state. The bootstrap begun on one is carried on in order on the
/// other, and both get the same two 204 events. In NX2 the second session would start in no
/// phase of its own and make two events of its own.
#[test]
fn two_root_sessions_share_the_phase_and_the_syncpoint_events() {
    let mut rig = Rig::new();
    let second = rig.kernel.new_object();
    rig.server.accept(second).unwrap();
    assert_eq!(rig.initialize(&local()), success(INITIALIZE, &[], &[]));
    let process = rig.process();
    let bind = rig.send_on(
        second,
        &[tipc_type(BIND_PROCESS), SPECIAL, ONE_COPY, process],
    );
    assert_eq!(bind, success(BIND_PROCESS, &[], &[]));
    // In order on the second session: the first session's 4 counts there.
    assert_eq!(
        (rig.state().phase(), rig.state().out_of_order()),
        (Phase::ProcessBound, 0)
    );
    let words = [tipc_type(BIND_SYNCPOINT_MEMORY), 0];
    let on_first = copied(&rig.send(&words));
    let on_second = copied(&rig.send_on(second, &words));
    let objects = |rig: &Rig, handles: &[u32]| -> Vec<ObjectId> {
        let resolve = |h: &u32| rig.kernel.resolve_handle(Handle::from_raw(*h)).unwrap();
        handles.iter().map(resolve).collect()
    };
    let made = objects(&rig, &on_first);
    assert_eq!(made.len(), 2);
    assert_eq!(objects(&rig, &on_second), made);
    assert_eq!(
        rig.state().syncpoint_events(),
        SyncpointEvents::Both(made[0], made[1])
    );
    // 101 on the second session completes the bootstrap the first began.
    let complete = rig.send_on(second, &[tipc_type(COMPLETE_BOOTSTRAP), 0]);
    assert_eq!(complete, success(COMPLETE_BOOTSTRAP, &[], &[]));
    assert_eq!(
        (rig.state().phase(), rig.state().out_of_order()),
        (Phase::Ready, 0)
    );
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
    assert_eq!(
        restored.family::<RootFamily>().map(RootFamily::state),
        Some(&state)
    );
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

// ----- Saved layouts ---------------------------------------------------------------------------

/// An object id with distinct, non-zero parts.
fn object(index: u32, generation: u32) -> ObjectId {
    ObjectId::from_parts(index, generation)
}

/// A bootstrap request whose six fields are `first`, `first + 1`, ... `first + 5`.
fn counting_request(first: u64) -> BootstrapRequest {
    BootstrapRequest {
        applet_resource_user_id: first,
        flags: first + 1,
        parameters: [first + 2, first + 3, first + 4, first + 5],
    }
}

/// The saved layouts of the root's state and sessions, as `postcard` writes them (field order,
/// a variant's index, `Option`'s tag; every value here below 0x80, one byte each, except the
/// placeholder's command). One state per phase, and each kind of syncpoint events among them,
/// with every value distinct and non-zero where the state allows it (`Created` has no
/// bootstrap and counts nothing). Each decodes back to itself.
#[test]
fn the_saved_root_layouts_are_pinned() {
    let cases: [(RootState, &[u8]); 4] = [
        (
            RootState {
                phase: Phase::Created,
                bootstrap: None,
                out_of_order: 0,
                native_memory_mode: Some(0x0B),
                syncpoint_events: SyncpointEvents::One(object(0x0C, 0x0D)),
                gpu_server: None,
                syncpt_server: Some(ServiceId(0x0E)),
                nvsrv_server: None,
                video_server: Some(ServiceId(0x0F)),
            },
            // phase 0; no bootstrap; count 0; mode Some(0x0B); One(index, generation); no gpu
            // server; Some(syncpt server); no nvsrv server; Some(video server).
            &[0, 0, 0, 1, 0x0B, 1, 0x0C, 0x0D, 0, 1, 0x0E, 0, 1, 0x0F],
        ),
        (
            RootState {
                phase: Phase::Initialized,
                bootstrap: Some(counting_request(0x11)),
                out_of_order: 0x17,
                native_memory_mode: None,
                syncpoint_events: SyncpointEvents::Both(object(0x18, 0x19), object(0x1A, 0x1B)),
                gpu_server: Some(ServiceId(0x1C)),
                syncpt_server: None,
                nvsrv_server: Some(ServiceId(0x1D)),
                video_server: None,
            },
            // phase 1; Some(user id, flags, four parameters); count; no mode; Both(a, b); Some(gpu
            // server); no syncpt server; Some(nvsrv server); no video server.
            &[
                1, 1, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0, 2, 0x18, 0x19, 0x1A, 0x1B, 1,
                0x1C, 0, 1, 0x1D, 0,
            ],
        ),
        (
            RootState {
                phase: Phase::ProcessBound,
                bootstrap: None,
                out_of_order: 0x21,
                native_memory_mode: Some(0x22),
                syncpoint_events: SyncpointEvents::None,
                gpu_server: None,
                syncpt_server: None,
                nvsrv_server: None,
                video_server: None,
            },
            // phase 2; no bootstrap; count; mode; no events; no gpu, syncpt, nvsrv or video
            // server.
            &[2, 0, 0x21, 1, 0x22, 0, 0, 0, 0, 0],
        ),
        (
            RootState {
                phase: Phase::Ready,
                bootstrap: Some(counting_request(0x31)),
                out_of_order: 0x37,
                native_memory_mode: Some(0x38),
                syncpoint_events: SyncpointEvents::One(object(0x39, 0x3A)),
                gpu_server: Some(ServiceId(0x3B)),
                syncpt_server: Some(ServiceId(0x3C)),
                nvsrv_server: Some(ServiceId(0x3D)),
                video_server: Some(ServiceId(0x3E)),
            },
            &[
                3, 1, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37, 1, 0x38, 1, 0x39, 0x3A, 1, 0x3B, 1,
                0x3C, 1, 0x3D, 1, 0x3E,
            ],
        ),
    ];
    for (state, bytes) in cases {
        assert_eq!(state.save_state().unwrap(), bytes, "{state:?}");
        let mut loaded = RootState::new();
        loaded.load_state(bytes).unwrap();
        assert_eq!(loaded, state);
    }
    // The placeholder: its command, 202 as a varint (0x4A | 0x80, then 1).
    assert_eq!(GET_SYNCPOINT_SESSION, 0xCA);
    let placeholder = Placeholder {
        command: GET_SYNCPOINT_SESSION,
    };
    assert_eq!(placeholder.save().unwrap(), [0xCA, 0x01]);
    // The root session keeps nothing.
    assert_eq!(GsvRoot.save().unwrap(), Vec::<u8>::new());
}

/// Each saved state no sequence of commands reaches, and the error it is refused with: the
/// first of them is the hostile snapshot that would hand out one event twice.
fn unreachable_states() -> Vec<(RootState, RootStateError)> {
    let reachable = RootState::new();
    let event = object(0x0C, 0x0D);
    vec![
        (
            RootState {
                bootstrap: Some(local()),
                out_of_order: 7,
                syncpoint_events: SyncpointEvents::Both(event, event),
                ..reachable
            },
            // Both faults: the phase's is reported first.
            RootStateError::CreatedWithBootstrap,
        ),
        (
            RootState {
                bootstrap: Some(local()),
                ..reachable
            },
            RootStateError::CreatedWithBootstrap,
        ),
        (
            RootState {
                out_of_order: 1,
                ..reachable
            },
            RootStateError::CreatedWithCount { count: 1 },
        ),
        (
            RootState {
                phase: Phase::Initialized,
                out_of_order: 1,
                ..reachable
            },
            RootStateError::InitializedWithoutBootstrap,
        ),
        (
            RootState {
                phase: Phase::ProcessBound,
                ..reachable
            },
            RootStateError::UncountedWithoutBootstrap,
        ),
        (
            RootState {
                phase: Phase::Ready,
                ..reachable
            },
            RootStateError::UncountedWithoutBootstrap,
        ),
        (
            RootState {
                syncpoint_events: SyncpointEvents::Both(event, event),
                ..reachable
            },
            RootStateError::SameSyncpointEvents,
        ),
    ]
}

/// The saved form of `state`, field for field.
fn saved(state: RootState) -> SavedRootState {
    SavedRootState {
        phase: state.phase,
        bootstrap: state.bootstrap,
        out_of_order: state.out_of_order,
        native_memory_mode: state.native_memory_mode,
        syncpoint_events: state.syncpoint_events,
        gpu_server: state.gpu_server,
        syncpt_server: state.syncpt_server,
        nvsrv_server: state.nvsrv_server,
        video_server: state.video_server,
    }
}

#[test]
fn a_saved_state_no_command_reaches_is_refused() {
    for (state, error) in unreachable_states() {
        assert_eq!(RootState::try_from(saved(state)), Err(error), "{state:?}");
        // Through a whole server snapshot: refused, and the server is left as it was.
        let bytes = root_server(UnknownPolicy::Strict)
            .with_family(state)
            .save()
            .unwrap();
        let mut restored = root_server(UnknownPolicy::Strict);
        let refused = SnapshotError::Decode(postcard::Error::SerdeDeCustom);
        assert_eq!(restored.restore(&bytes), Err(refused), "{state:?}");
        assert_eq!(
            restored.family::<RootFamily>().map(RootFamily::state),
            Some(&RootState::new())
        );
    }
}

/// Each relation's edge, on the side that is reachable: accepted as it is.
#[test]
fn a_saved_state_at_each_edge_is_accepted() {
    let reachable = RootState::new();
    let cases = [
        reachable,
        RootState {
            phase: Phase::Initialized,
            bootstrap: Some(local()),
            ..reachable
        },
        RootState {
            phase: Phase::ProcessBound,
            out_of_order: 1,
            ..reachable
        },
        RootState {
            phase: Phase::Ready,
            out_of_order: 1,
            ..reachable
        },
        RootState {
            phase: Phase::Ready,
            bootstrap: Some(local()),
            ..reachable
        },
        RootState {
            syncpoint_events: SyncpointEvents::Both(object(1, 0), object(1, 1)),
            ..reachable
        },
    ];
    for state in cases {
        assert_eq!(RootState::try_from(saved(state)), Ok(state), "{state:?}");
        let bytes = root_server(UnknownPolicy::Strict)
            .with_family(state)
            .save()
            .unwrap();
        let mut restored = root_server(UnknownPolicy::Strict);
        restored.restore(&bytes).unwrap();
        assert_eq!(
            restored.family::<RootFamily>().map(RootFamily::state),
            Some(&state)
        );
    }
}

#[test]
fn a_refused_state_is_said_in_plain_words() {
    let said = [
        (
            RootStateError::CreatedWithBootstrap,
            "a root that no bootstrap command has reached has a recorded bootstrap",
        ),
        (
            RootStateError::CreatedWithCount { count: 7 },
            "a root that no bootstrap command has reached has counted 7 out of order",
        ),
        (
            RootStateError::InitializedWithoutBootstrap,
            "an initialized root has no recorded bootstrap",
        ),
        (
            RootStateError::UncountedWithoutBootstrap,
            "a root that left the created phase without a bootstrap has counted nothing out of order",
        ),
        (
            RootStateError::SameSyncpointEvents,
            "the two syncpoint events are the same object",
        ),
    ];
    for (error, words) in said {
        assert_eq!(error.to_string(), words);
    }
}

// ----- 497 and the gpu server ------------------------------------------------------------------

/// The number the tests' gpu server is known by.
const GPU: ServiceId = ServiceId(0x0749);

/// A root wired to a gpu server.
fn wired() -> Rig {
    let mut kernel = FakeKernel::new(0).unwrap();
    kernel.set_current_service(ROOT);
    kernel.add_service(GPU);
    let session = kernel.new_object();
    let state = RootState::new().with_gpu_server(GPU);
    let mut server = root_server_for(state, UnknownPolicy::Strict).with_service_id(ROOT);
    server.accept(session).unwrap();
    Rig {
        kernel,
        server,
        session,
    }
}

/// 497 (`0x201, 2`, PR `OpenGpuSession`: flags at words 2-3) copies a new session on the gpu
/// server, with flags 4 or any other (logged, decision 7), bootstrap complete or not.
#[test]
fn open_gpu_session_copies_a_session_on_the_gpu_server() {
    let mut rig = wired();
    assert_eq!(tipc_type(OPEN_GPU_SESSION), 0x201);
    for flags in [GPU_SESSION_FLAGS, 5] {
        let [low, high] = halves(flags);
        let reply = rig.send(&[0x201, 2, low, high]);
        assert_eq!(copied(&reply).len(), 1);
        let connections = rig.kernel.take_connections();
        assert_eq!(connections.len(), 1);
        assert_eq!(connections[0].0, GPU);
    }
    // Only the root's own session: no placeholder was made.
    assert_eq!(rig.server.session_count(), 1);
    assert_eq!(rig.state().gpu_server(), Some(GPU));
    // A short request is refused.
    let short = rig.send(&[0x201, 1, 4]);
    assert_eq!(short, result_only(OPEN_GPU_SESSION, INVALID_CMIF_REQUEST));
}

/// A snapshot that names another gpu server, or none, is refused, and the server keeps its own.
#[test]
fn a_snapshot_with_another_gpu_server_is_refused() {
    let unwired = Rig::new();
    let saved = unwired.server.save().unwrap();
    let mut rig = wired();
    assert_eq!(
        rig.server.restore(&saved),
        Err(SnapshotError::FamilyWiring(
            "its root names another gpu server"
        ))
    );
    assert_eq!(rig.state().gpu_server(), Some(GPU));
    let own = rig.server.save().unwrap();
    assert_eq!(rig.server.restore(&own), Ok(()));
}

// ----- 202, 1402 and their servers -------------------------------------------------------------

/// The numbers the tests' syncpt and nvsrv servers are known by.
const SYNCPT: ServiceId = ServiceId(0x0202);
const NVSRV: ServiceId = ServiceId(0x1402);

/// A root over `state`, with every family server it names known to the kernel.
fn wired_with(state: RootState) -> Rig {
    let mut kernel = FakeKernel::new(0).unwrap();
    kernel.set_current_service(ROOT);
    for service in [state.gpu_server, state.syncpt_server, state.nvsrv_server]
        .into_iter()
        .flatten()
    {
        kernel.add_service(service);
    }
    let session = kernel.new_object();
    let mut server = root_server_for(state, UnknownPolicy::Strict).with_service_id(ROOT);
    server.accept(session).unwrap();
    Rig {
        kernel,
        server,
        session,
    }
}

/// 202 (PR `GetSyncpointSession`: `packet[0] == 0xda && packet[1] == 0`) and 1402 (PR
/// `GetNvidiaDriver`: `packet[0] != 0x58a || packet[1] != 0`), as the header words PR compares.
const GET_SYNCPOINT_SESSION_WORDS: [u32; 2] = [0xDA, 0];
const GET_NVIDIA_DRIVER_WORDS: [u32; 2] = [0x58A, 0];

#[test]
fn the_family_session_header_words_are_the_command_ids() {
    assert_eq!(
        GET_SYNCPOINT_SESSION_WORDS[0],
        tipc_type(GET_SYNCPOINT_SESSION)
    );
    assert_eq!(GET_NVIDIA_DRIVER_WORDS[0], tipc_type(GET_NVIDIA_DRIVER));
    assert_eq!((GET_SYNCPOINT_SESSION, GET_NVIDIA_DRIVER), (202, 1402));
}

/// The server one of 202 and 1402 connected to, or `None` when it answered with the stub's
/// placeholder (a new session on the root's own server).
fn opened(rig: &mut Rig, words: &[u32]) -> Option<ServiceId> {
    let sessions = rig.server.session_count();
    let reply = rig.send(words);
    assert_eq!(copied(&reply).len(), 1, "{words:x?}");
    let connections = rig.kernel.take_connections();
    match connections.as_slice() {
        [] => {
            assert_eq!(rig.server.session_count(), sessions + 1);
            None
        }
        [(service, _)] => {
            assert_eq!(rig.server.session_count(), sessions);
            Some(*service)
        }
        more => unreachable!("one request made {} connections", more.len()),
    }
}

/// 202 and 1402 each copy a session on their own server once the root names it, before the
/// bootstrap and after it, and the stub's placeholder while it does not: each server is wired
/// alone too, so a root that sent one command to another's server would fail.
#[test]
fn each_family_command_hands_out_a_session_on_its_own_server() {
    let cases = [
        (RootState::new(), None, None),
        (
            RootState::new().with_syncpt_server(SYNCPT),
            Some(SYNCPT),
            None,
        ),
        (RootState::new().with_nvsrv_server(NVSRV), None, Some(NVSRV)),
        (RootState::new().with_gpu_server(GPU), None, None),
        (
            RootState::new()
                .with_gpu_server(GPU)
                .with_syncpt_server(SYNCPT)
                .with_nvsrv_server(NVSRV),
            Some(SYNCPT),
            Some(NVSRV),
        ),
    ];
    for (state, syncpt, nvsrv) in cases {
        let mut rig = wired_with(state);
        assert_eq!(
            opened(&mut rig, &GET_SYNCPOINT_SESSION_WORDS),
            syncpt,
            "{state:?}"
        );
        assert_eq!(
            opened(&mut rig, &GET_NVIDIA_DRIVER_WORDS),
            nvsrv,
            "{state:?}"
        );
        rig.initialize(&local());
        rig.bind_process();
        rig.complete_bootstrap();
        assert_eq!(rig.state().phase(), Phase::Ready);
        assert_eq!(
            opened(&mut rig, &GET_SYNCPOINT_SESSION_WORDS),
            syncpt,
            "{state:?}"
        );
        assert_eq!(
            opened(&mut rig, &GET_NVIDIA_DRIVER_WORDS),
            nvsrv,
            "{state:?}"
        );
        assert_eq!(rig.state().syncpt_server(), syncpt);
        assert_eq!(rig.state().nvsrv_server(), nvsrv);
    }
}

/// A wired 202 or 1402 replies a success word and one copied handle, nothing more.
#[test]
fn a_family_session_reply_is_one_copied_handle_and_success() {
    let state = RootState::new()
        .with_syncpt_server(SYNCPT)
        .with_nvsrv_server(NVSRV);
    for (words, command) in [
        (GET_SYNCPOINT_SESSION_WORDS, GET_SYNCPOINT_SESSION),
        (GET_NVIDIA_DRIVER_WORDS, GET_NVIDIA_DRIVER),
    ] {
        let mut rig = wired_with(state);
        let reply = rig.send(&words);
        let handle = copied(&reply)[0];
        // A special header with one data word (the result) and one copied handle.
        let expected = le(&[tipc_type(command), SPECIAL | 1, ONE_COPY, handle, 0]);
        assert_eq!(&reply[..expected.len()], &expected[..], "{command}");
    }
}

/// A 202 or 1402 whose header is not the one PR compares (`0xda, 0`, `0x58a, 0`) because it
/// counts data words is accepted, the data not read: PR answers `ResultInvalidState` (202) and
/// `ResultInvalidCombination` (1402), NX2 success (module docs, "Where this differs from the
/// references"). The data words differ from each other and from the reply's, so a handler that
/// read or echoed them would show it.
#[test]
fn a_family_command_with_a_data_word_is_accepted() {
    let state = RootState::new()
        .with_syncpt_server(SYNCPT)
        .with_nvsrv_server(NVSRV);
    for (command, server) in [(GET_SYNCPOINT_SESSION, SYNCPT), (GET_NVIDIA_DRIVER, NVSRV)] {
        for data in [&[0xDEAD_BEEF][..], &[0x1234_5678, 0x9ABC_DEF0][..]] {
            let mut words = vec![tipc_type(command), u32::try_from(data.len()).unwrap()];
            words.extend(data);
            let mut rig = wired_with(state);
            assert_eq!(opened(&mut rig, &words), Some(server), "{words:x?}");
            let reply = rig.send(&words);
            let handle = copied(&reply)[0];
            let expected = le(&[tipc_type(command), SPECIAL | 1, ONE_COPY, handle, 0]);
            assert_eq!(&reply[..expected.len()], &expected[..], "{words:x?}");
            assert_eq!(rig.state(), state);
        }
    }
}

/// A snapshot whose gpu, syncpt or nvsrv server is not the run's is refused, and the server
/// keeps its own. With the gpu server and the nvsrv server both different, the gpu server's
/// refusal is reported; with the syncpt and nvsrv servers both different, the syncpt server's.
#[test]
fn a_snapshot_with_other_family_servers_is_refused_gpu_first() {
    let run = RootState::new()
        .with_gpu_server(GPU)
        .with_syncpt_server(SYNCPT)
        .with_nvsrv_server(NVSRV);
    let other = ServiceId(0x0999);
    let cases = [
        (
            RootState::new()
                .with_gpu_server(other)
                .with_syncpt_server(SYNCPT)
                .with_nvsrv_server(other),
            "its root names another gpu server",
        ),
        (
            RootState::new()
                .with_gpu_server(GPU)
                .with_syncpt_server(other)
                .with_nvsrv_server(other),
            "its root names another syncpt server",
        ),
        (
            RootState::new()
                .with_gpu_server(GPU)
                .with_syncpt_server(SYNCPT),
            "its root names another nvsrv server",
        ),
        (
            RootState::new()
                .with_gpu_server(GPU)
                .with_nvsrv_server(NVSRV),
            "its root names another syncpt server",
        ),
    ];
    for (saved, refusal) in cases {
        let bytes = wired_with(saved).server.save().unwrap();
        let mut rig = wired_with(run);
        assert_eq!(
            rig.server.restore(&bytes),
            Err(SnapshotError::FamilyWiring(refusal)),
            "{saved:?}"
        );
        assert_eq!(rig.state(), run);
        let own = rig.server.save().unwrap();
        assert_eq!(rig.server.restore(&own), Ok(()));
    }
}
