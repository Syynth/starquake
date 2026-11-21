//! `aud:u` driven through raw messages over a fake kernel, laid out as Eden's CMIF
//! serialisation reads them (module docs).

use starquake_ipc::{
    BUFFER_SIZE, Body, CommandType, Message, RequestBuilder, RequestLayout, SessionKind,
};
use starquake_kernel::handle::Handle;
use starquake_kernel::id::ObjectId;
use starquake_kernel::result::ResultCode;

use super::command::renderer_manager::GET_WORK_BUFFER_SIZE;
use super::command::user_client::GET_AUDIO_RENDERER_MANAGER;
use super::work_buffer::{CURRENT_REVISION, PARAMETER_SIZE};
use super::*;
use crate::framework::control::CONVERT_CURRENT_OBJECT_TO_DOMAIN;
use crate::framework::fake::FakeKernel;
use crate::framework::result::{INVALID_CMIF_REQUEST, UNKNOWN_COMMAND_ID};
use crate::framework::{Handled, Kernel, Protocol, Session, StubbedCommand};

/// `aud:u`'s commands left to the unknown-command policy (module docs, "Left out"; deferred
/// against decision 8's rule, "The two references"), so they have no shipping constants: PR
/// #4588's `GetAudioInManager` (0) and `GetAudioOutManager` (1)
/// (`eden-pr4588/src/core/hle/service/audio/audio.cpp` @ e5fb3b97).
const USER_CLIENT_LEFT_OUT: [u32; 2] = [0, 1];
/// `IAudioRendererManager`'s commands left out likewise
/// (`eden-nx2/src/core/hle/service/audio/audio_renderer_manager.cpp` @ 50fa86bb):
/// `OpenAudioRenderer` (0), `GetAudioDeviceService` (2), `OpenAudioRendererForManualExecution`
/// (3) and `GetAudioDeviceServiceWithRevisionInfo` (4).
const MANAGER_LEFT_OUT: [u32; 4] = [0, 2, 3, 4];

/// What fills the message buffer beforehand.
const STALE: u8 = 0xA5;

/// Eden-NX2's size for [`parameter`] at revision 15 (`work_buffer/tests.rs`, golden row 15).
const SIZE_AT_15: u64 = 266_240;

/// An `aud:u` server, a kernel, and one session.
struct Rig {
    kernel: FakeKernel,
    server: Server,
    session: ObjectId,
}

impl Rig {
    fn new(policy: UnknownPolicy) -> Self {
        let mut kernel = FakeKernel::new(0).unwrap();
        let session = kernel.new_object();
        let mut server = u_server(policy);
        server.accept(session).unwrap();
        Self {
            kernel,
            server,
            session,
        }
    }

    fn send_raw(&mut self, session: ObjectId, request: &RequestBuilder) -> Vec<u8> {
        let mut buffer = vec![STALE; BUFFER_SIZE];
        request.build(&mut buffer).unwrap();
        let handled = self.server.handle(&mut self.kernel, session, &mut buffer);
        assert_eq!(handled, Ok(Handled::Replied));
        buffer
    }

    /// The result, data and moved handles of the reply to `request`.
    fn send(
        &mut self,
        session: ObjectId,
        request: &RequestBuilder,
    ) -> (ResultCode, Vec<u8>, Vec<u32>) {
        let reply = self.send_raw(session, request);
        let message = Message::parse_reply(&reply, SessionKind::Session).unwrap();
        let Body::CmifReply(body) = &message.body else {
            unreachable!("not a CMIF reply: {:?}", message.body);
        };
        (
            ResultCode::from_raw(body.header.result),
            body.data.to_vec(),
            message.move_handles.clone(),
        )
    }

    /// Sends `GetAudioRendererManager` and returns the manager's session.
    fn manager(&mut self) -> ObjectId {
        let (result, _, moves) = self.send(self.session, &request(GET_AUDIO_RENDERER_MANAGER));
        assert_eq!((result, moves.len()), (ResultCode::SUCCESS, 1));
        self.kernel
            .resolve_handle(Handle::from_raw(moves[0]))
            .unwrap()
    }
}

fn request(command: u32) -> RequestBuilder {
    RequestBuilder::cmif(CommandType::Request, command).layout(RequestLayout::Libnx)
}

/// The renderer parameter's bytes, as `AudioRendererParameterInternal` lays them out: sample
/// rate 48000, sample count 240, mixes 3, sub-mixes 5, voices 7, sinks 2, effects 11, perf
/// frames 13, voice drop 1, three zero bytes, splitter infos 17, splitter destinations 19,
/// external context size 23, and `revision`.
fn parameter(revision: u32) -> Vec<u8> {
    let mut bytes = Vec::new();
    for word in [48_000_u32, 240, 3, 5, 7, 2, 11, 13] {
        bytes.extend(word.to_le_bytes());
    }
    bytes.extend([1, 0, 0, 0]);
    for word in [17_u32, 19, 23, revision] {
        bytes.extend(word.to_le_bytes());
    }
    assert_eq!(bytes.len(), PARAMETER_SIZE);
    bytes
}

/// `GetWorkBufferSize` with `data` as its payload.
fn work_buffer_size_request(data: &[u8]) -> RequestBuilder {
    request(GET_WORK_BUFFER_SIZE).data(data)
}

#[test]
fn get_audio_renderer_manager_hands_out_a_manager_that_sizes_a_work_buffer() {
    let mut rig = Rig::new(UnknownPolicy::Strict);
    let manager = rig.manager();
    assert_eq!(rig.kernel.sessions(), [manager]);
    assert_eq!(
        (rig.server.session_count(), rig.server.interface_count()),
        (2, 2)
    );
    let (result, data, moves) = rig.send(
        manager,
        &work_buffer_size_request(&parameter(CURRENT_REVISION)),
    );
    // The reply's payload is the size and nothing else (`Out<u64>`).
    assert_eq!(
        (result, data, moves.len()),
        (ResultCode::SUCCESS, SIZE_AT_15.to_le_bytes().to_vec(), 0)
    );
    // Asked again, the same answer: the manager keeps nothing.
    let again = rig.send(
        manager,
        &work_buffer_size_request(&parameter(CURRENT_REVISION)),
    );
    assert_eq!(again.1, SIZE_AT_15.to_le_bytes().to_vec());
}

#[test]
fn every_call_hands_out_a_new_manager() {
    // Eden-NX2's `PushIpcInterface<IAudioRendererManager>(ctx, system)` makes one per call.
    let mut rig = Rig::new(UnknownPolicy::Strict);
    let first = rig.manager();
    let second = rig.manager();
    assert_ne!(first, second);
    assert_eq!(rig.server.session_count(), 3);
    for manager in [first, second] {
        let (result, data, _) = rig.send(
            manager,
            &work_buffer_size_request(&parameter(CURRENT_REVISION)),
        );
        assert_eq!(
            (result, data),
            (ResultCode::SUCCESS, SIZE_AT_15.to_le_bytes().to_vec())
        );
    }
}

#[test]
fn get_audio_renderer_manager_reads_no_word() {
    // Eden-NX2 pops a `u32` it only logs; PR #4588 reads nothing. With either, a manager.
    for with_word in [false, true] {
        let mut rig = Rig::new(UnknownPolicy::Strict);
        let mut get = request(GET_AUDIO_RENDERER_MANAGER);
        if with_word {
            get = get.data(&0x0102_0304_u32.to_le_bytes());
        }
        let (result, data, moves) = rig.send(rig.session, &get);
        assert_eq!(
            (result, data, moves.len()),
            (ResultCode::SUCCESS, vec![], 1),
            "{with_word}"
        );
    }
}

#[test]
fn get_audio_renderer_manager_in_a_domain_returns_an_object() {
    let mut rig = Rig::new(UnknownPolicy::Strict);
    let convert = RequestBuilder::cmif(CommandType::Control, CONVERT_CURRENT_OBJECT_TO_DOMAIN);
    let converted = rig.send_raw(rig.session, &convert);
    let root = u32::from_le_bytes(converted[32..36].try_into().unwrap());
    let get = RequestBuilder::domain(CommandType::Request, root, GET_AUDIO_RENDERER_MANAGER);
    let reply = rig.send_raw(rig.session, &get);
    let message = Message::parse_reply(&reply, SessionKind::Domain).unwrap();
    let Body::CmifReply(body) = &message.body else {
        unreachable!("not a CMIF reply");
    };
    assert_eq!(body.header.result, ResultCode::SUCCESS.raw());
    let object = body.object_ids[0];
    assert_eq!(object, root + 1);
    assert_eq!(rig.kernel.sessions(), []);
    // The object answers, as a manager.
    let size = RequestBuilder::domain(CommandType::Request, object, GET_WORK_BUFFER_SIZE)
        .data(&parameter(CURRENT_REVISION));
    let reply = rig.send_raw(rig.session, &size);
    let message = Message::parse_reply(&reply, SessionKind::Domain).unwrap();
    let Body::CmifReply(body) = &message.body else {
        unreachable!("not a CMIF reply");
    };
    assert_eq!(body.header.result, ResultCode::SUCCESS.raw());
    assert_eq!(body.data, SIZE_AT_15.to_le_bytes());
}

#[test]
fn get_work_buffer_size_refuses_a_revision_past_its_bound_with_invalid_revision() {
    let mut rig = Rig::new(UnknownPolicy::Strict);
    let manager = rig.manager();
    // Past 15 as a number or `REV` tag, past 16 as a native `REv` tag, and a native tag that
    // names no revision.
    for revision in [
        CURRENT_REVISION + 1,
        u32::from_le_bytes(*b"REV@"),
        u32::from_le_bytes(*b"REvA"),
        u32::from_le_bytes(*b"REv/"),
    ] {
        let (result, data, _) = rig.send(manager, &work_buffer_size_request(&parameter(revision)));
        assert_eq!((result, data), (INVALID_REVISION, vec![]), "{revision:#x}");
    }
    // At each bound itself, a size: 15 as a number or `REV?`, 16 as `REv@`, the word a native
    // title sends, which has revision 15's features and so its size.
    for revision in [
        CURRENT_REVISION,
        u32::from_le_bytes(*b"REV?"),
        u32::from_le_bytes(*b"REv@"),
    ] {
        let (result, data, _) = rig.send(manager, &work_buffer_size_request(&parameter(revision)));
        assert_eq!(
            (result, data),
            (ResultCode::SUCCESS, SIZE_AT_15.to_le_bytes().to_vec()),
            "{revision:#x}"
        );
    }
}

#[test]
fn get_work_buffer_size_refuses_a_size_that_would_wrap_with_invalid_argument() {
    let mut rig = Rig::new(UnknownPolicy::Strict);
    let manager = rig.manager();
    // Sub-mixes at offset 0x0C: 2^24 makes `sub_mixes * MAX_EFFECTS` 2^32.
    let mut wraps = parameter(CURRENT_REVISION);
    wraps[0x0C..0x10].copy_from_slice(&0x0100_0000_u32.to_le_bytes());
    let (result, data, _) = rig.send(manager, &work_buffer_size_request(&wraps));
    assert_eq!((result, data), (INVALID_ARGUMENT, vec![]));
    // Splitter destinations at offset 0x28: -1, which the splitter reads at revision 15.
    let mut negative = parameter(CURRENT_REVISION);
    negative[0x28..0x2C].copy_from_slice(&(-1_i32).to_le_bytes());
    let (result, data, _) = rig.send(manager, &work_buffer_size_request(&negative));
    assert_eq!((result, data), (INVALID_ARGUMENT, vec![]));
    // Both with a revision past 15: the revision is reported.
    let mut both = wraps;
    both[0x30..0x34].copy_from_slice(&(CURRENT_REVISION + 1).to_le_bytes());
    let (result, _, _) = rig.send(manager, &work_buffer_size_request(&both));
    assert_eq!(result, INVALID_REVISION);
}

#[test]
fn the_results_are_mapped_by_error() {
    assert_eq!(
        result_of(WorkBufferError::InvalidRevision {
            word: 16,
            number: 16
        }),
        INVALID_REVISION
    );
    assert_eq!(
        result_of(WorkBufferError::NoRevision { word: 0x2f76_4552 }),
        INVALID_REVISION
    );
    assert_eq!(result_of(WorkBufferError::Overflow), INVALID_ARGUMENT);
    assert_eq!(
        result_of(WorkBufferError::NegativeSplitterDestinations(-1)),
        INVALID_ARGUMENT
    );
}

#[test]
fn a_parameter_short_of_its_last_words_is_refused() {
    let mut rig = Rig::new(UnknownPolicy::Strict);
    let manager = rig.manager();
    let whole = parameter(CURRENT_REVISION);
    // A word short, two words short, and nothing at all: each read past the end is refused. A
    // message's data is counted in whole words, so a payload one byte short reaches the server
    // padded to a word; `work_buffer/tests.rs` (`a_parameter_one_byte_short_is_refused`) reads
    // one byte short directly.
    for len in [PARAMETER_SIZE - 4, PARAMETER_SIZE - 8, 0] {
        let short =
            RequestBuilder::cmif(CommandType::Request, GET_WORK_BUFFER_SIZE).data(&whole[..len]);
        let (result, data, _) = rig.send(manager, &short);
        assert_eq!((result, data), (INVALID_CMIF_REQUEST, vec![]), "{len}");
    }
    let exact = RequestBuilder::cmif(CommandType::Request, GET_WORK_BUFFER_SIZE).data(&whole);
    let (result, data, _) = rig.send(manager, &exact);
    assert_eq!(
        (result, data),
        (ResultCode::SUCCESS, SIZE_AT_15.to_le_bytes().to_vec())
    );
}

#[test]
fn the_result_codes_are_edens() {
    // `{ErrorModule::Audio, 1537}` and `{ErrorModule::Audio, 900}`; `Audio = 153`.
    assert_eq!(AUDIO_MODULE, 153);
    assert_eq!(
        (INVALID_REVISION.module(), INVALID_REVISION.description()),
        (153, 1537)
    );
    assert_eq!(
        (INVALID_ARGUMENT.module(), INVALID_ARGUMENT.description()),
        (153, 900)
    );
}

#[test]
fn the_manager_is_not_the_root_and_the_root_is_not_the_manager() {
    let mut rig = Rig::new(UnknownPolicy::Strict);
    let manager = rig.manager();
    // The root's 2 is the manager's left-out `GetAudioDeviceService`, and the manager's 1 is
    // not the root's.
    let (result, _, moves) = rig.send(manager, &request(GET_AUDIO_RENDERER_MANAGER));
    assert_eq!((result, moves.len()), (UNKNOWN_COMMAND_ID, 0));
    let (result, data, _) = rig.send(
        rig.session,
        &work_buffer_size_request(&parameter(CURRENT_REVISION)),
    );
    assert_eq!((result, data), (UNKNOWN_COMMAND_ID, vec![]));
    assert_eq!(rig.server.session_count(), 2);
}

#[test]
fn commands_that_are_left_out_go_to_the_unknown_command_policy() {
    // Under the strict policy, each is unknown.
    let mut rig = Rig::new(UnknownPolicy::Strict);
    let manager = rig.manager();
    for (session, commands) in [
        (rig.session, &USER_CLIENT_LEFT_OUT[..]),
        (manager, &MANAGER_LEFT_OUT[..]),
    ] {
        for &command in commands {
            assert_eq!(
                rig.send(session, &request(command)).0,
                UNKNOWN_COMMAND_ID,
                "{command}"
            );
        }
    }
    // Under the auto-stub policy, each answers success with a payload of zeroes and no object,
    // and each is recorded against its own interface.
    let mut rig = Rig::new(UnknownPolicy::AutoStub);
    let manager = rig.manager();
    let mut expected = Vec::new();
    for (session, interface, commands) in [
        (rig.session, "IAudioUserClient", &USER_CLIENT_LEFT_OUT[..]),
        (manager, "IAudioRendererManager", &MANAGER_LEFT_OUT[..]),
    ] {
        for &command in commands {
            let (result, data, moves) = rig.send(session, &request(command));
            assert_eq!((result, moves.len()), (ResultCode::SUCCESS, 0), "{command}");
            assert!(data.iter().all(|byte| *byte == 0), "{command}");
            expected.push(StubbedCommand {
                interface: starquake_trace::InterfaceName::new(interface).unwrap(),
                protocol: starquake_trace::IpcProtocol::Cmif,
                id: command,
            });
        }
    }
    let mut stubbed: Vec<_> = rig.server.stubbed().copied().collect();
    stubbed.sort_by_key(|stub| (stub.interface, stub.id));
    expected.sort_by_key(|stub| (stub.interface, stub.id));
    assert_eq!(stubbed, expected);
}

#[test]
fn a_tipc_request_goes_to_the_unknown_command_policy() {
    // PR #4588 answers a TIPC request `Kernel::ResultNotImplemented`; here no interface has a
    // TIPC table (module docs, "The two references").
    let id = u16::try_from(GET_AUDIO_RENDERER_MANAGER).unwrap();
    let mut rig = Rig::new(UnknownPolicy::Strict);
    let reply = rig.send_raw(rig.session, &RequestBuilder::tipc(id));
    let message = Message::parse_reply(&reply, SessionKind::Session).unwrap();
    let Body::TipcReply { result, .. } = message.body else {
        unreachable!("not a TIPC reply: {:?}", message.body);
    };
    assert_eq!(result, UNKNOWN_COMMAND_ID.raw());
    assert_eq!(message.move_handles, Vec::<u32>::new());
    assert_eq!(rig.server.stubbed().count(), 0);

    let mut rig = Rig::new(UnknownPolicy::AutoStub);
    let reply = rig.send_raw(rig.session, &RequestBuilder::tipc(id));
    let message = Message::parse_reply(&reply, SessionKind::Session).unwrap();
    let Body::TipcReply { result, .. } = message.body else {
        unreachable!("not a TIPC reply: {:?}", message.body);
    };
    assert_eq!(result, ResultCode::SUCCESS.raw());
    assert_eq!(message.move_handles, Vec::<u32>::new());
    let stubbed: Vec<_> = rig.server.stubbed().copied().collect();
    assert_eq!(
        stubbed,
        [StubbedCommand {
            interface: starquake_trace::InterfaceName::new("IAudioUserClient").unwrap(),
            protocol: starquake_trace::IpcProtocol::Tipc,
            id: GET_AUDIO_RENDERER_MANAGER,
        }]
    );
}

#[test]
fn a_server_with_a_manager_open_saves_and_restores() {
    let mut rig = Rig::new(UnknownPolicy::Strict);
    let manager = rig.manager();
    let saved = rig.server.save().unwrap();
    let mut restored = u_server(UnknownPolicy::Strict);
    restored.restore(&saved).unwrap();
    assert_eq!(restored.save().unwrap(), saved);
    // The restored server's manager sizes a work buffer, and its root still makes managers.
    rig.server = restored;
    let (result, data, _) = rig.send(
        manager,
        &work_buffer_size_request(&parameter(CURRENT_REVISION)),
    );
    assert_eq!(
        (result, data),
        (ResultCode::SUCCESS, SIZE_AT_15.to_le_bytes().to_vec())
    );
    let another = rig.manager();
    assert_ne!(another, manager);
    assert_eq!(rig.server.session_count(), 3);
}

#[test]
fn the_command_ids_and_name_are_the_references() {
    // Eden-NX2's `{2, .., "GetAudioRendererManager (guessed)"}` and Eden's `{1,
    // D<&IAudioRendererManager::GetWorkBufferSize>, "GetWorkBufferSize"}`;
    // `RegisterNamedService("aud:u", ..)`.
    assert_eq!((GET_AUDIO_RENDERER_MANAGER, GET_WORK_BUFFER_SIZE), (2, 1));
    assert_eq!(U_NAME, "aud:u");
}

#[test]
fn the_tables_name_the_commands() {
    assert_eq!(
        IAudioUserClient
            .commands()
            .name(Protocol::Cmif, GET_AUDIO_RENDERER_MANAGER),
        Some("get_audio_renderer_manager")
    );
    assert_eq!(IAudioUserClient.commands().cmif.len(), 1);
    assert_eq!(
        IAudioRendererManager
            .commands()
            .name(Protocol::Cmif, GET_WORK_BUFFER_SIZE),
        Some("get_work_buffer_size")
    );
    assert_eq!(IAudioRendererManager.commands().cmif.len(), 1);
    assert_eq!(IAudioUserClient.commands().tipc, []);
    assert_eq!(IAudioRendererManager.commands().tipc, []);
    assert_eq!(IAudioUserClient::interface(), "IAudioUserClient");
    assert_eq!(IAudioRendererManager::interface(), "IAudioRendererManager");
}
