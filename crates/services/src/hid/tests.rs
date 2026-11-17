//! `hid` driven through raw messages laid out as libnx lays them out
//! (`libnx/nx/source/services/hid.c`, `_hidCreateAppletResource` and
//! `_hidGetSharedMemoryHandle`), over a fake kernel. libnx keeps its `hid` session plain
//! (`NX_SERVICE_ASSUME_NON_DOMAIN`), as the startup fixture's trace shows (line 92 sends no
//! conversion), so these tests do too.

use starquake_ipc::{
    BUFFER_SIZE, Body, CommandHeader, CommandType, Message, ReceiveList, RequestBuilder,
    RequestLayout, SFCO_MAGIC, SessionKind, SpecialHeader,
};
use starquake_kernel::handle::Handle;
use starquake_kernel::id::ObjectId;
use starquake_kernel::memory::SvcPermission;
use starquake_kernel::result::{KernelError, ResultCode};

use super::command::applet_resource::GET_SHARED_MEMORY_HANDLE;
use super::command::hid_server::CREATE_APPLET_RESOURCE;
use super::shared_memory::{
    LIFO_BUFFER_COUNT, LIFO_ENTRY_COUNT, NPAD_ENTRY_COUNT, SIZE, initial_image, lifo_offsets,
};
use super::*;
use crate::framework::fake::FakeKernel;
use crate::framework::result::UNKNOWN_COMMAND_ID;
use crate::framework::{Handled, Kernel, Protocol, Session};

/// `IHidServer::ActivateDebugPad`, which this port leaves out, so it has no shipping constant
/// (`eden/src/core/hle/service/hid/hid_server.cpp`: `{1, C<&IHidServer::ActivateDebugPad>, ..}`).
const ACTIVATE_DEBUG_PAD: u32 = 1;

/// The caller's process id: distinct from every other number the tests use.
const PROCESS_ID: u64 = 0x0123_4567_89AB_CDEF;
/// What libnx puts in the payload: `appletGetAppletResourceUserId()`. Distinct from
/// [`PROCESS_ID`], so a reply that used it instead of the kernel's id would show.
const PAYLOAD_ARUID: u64 = 0x0FED_CBA9_8765_4321;
/// What fills the message buffer beforehand.
const STALE: u8 = 0xA5;

/// A `hid` server, a kernel, and one session.
struct Rig {
    kernel: FakeKernel,
    server: Server,
    session: ObjectId,
}

/// A parsed reply.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Answered {
    result: ResultCode,
    copies: Vec<u32>,
    moves: Vec<u32>,
}

impl Rig {
    fn new() -> Self {
        let mut kernel = FakeKernel::new(0).unwrap();
        kernel.set_process_id(PROCESS_ID);
        let session = kernel.new_object();
        let mut server = server(UnknownPolicy::Strict);
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

    fn send(&mut self, session: ObjectId, request: &RequestBuilder) -> Answered {
        let reply = self.send_raw(session, request);
        let message = Message::parse_reply(&reply, SessionKind::Session).unwrap();
        let Body::CmifReply(body) = &message.body else {
            unreachable!("not a CMIF reply: {:?}", message.body);
        };
        Answered {
            result: ResultCode::from_raw(body.header.result),
            copies: message.copy_handles.clone(),
            moves: message.move_handles.clone(),
        }
    }

    /// `CreateAppletResource` as libnx sends it; the raw reply.
    fn create_raw(&mut self) -> Vec<u8> {
        self.send_raw(self.session, &create())
    }

    /// Sends `CreateAppletResource` for the kernel's current process id, and returns the
    /// resource's session.
    fn resource(&mut self) -> ObjectId {
        let reply = self.send(self.session, &create());
        assert_eq!(reply.result, ResultCode::SUCCESS);
        self.object_of(reply.moves[0])
    }

    /// Sends `GetSharedMemoryHandle` to `resource`; the parsed reply.
    fn memory_of(&mut self, resource: ObjectId) -> Answered {
        self.send(resource, &libnx_request(GET_SHARED_MEMORY_HANDLE))
    }

    /// Sends `GetSharedMemoryHandle` to `resource`; the memory its copied handle names.
    fn memory(&mut self, resource: ObjectId) -> ObjectId {
        let handle = self.memory_of(resource).copies[0];
        self.object_of(handle)
    }

    /// The object a handle in the caller's table names.
    fn object_of(&self, handle: u32) -> ObjectId {
        self.kernel
            .resolve_handle(Handle::from_raw(handle))
            .unwrap()
    }

    fn resources(&self) -> AppletResources {
        self.server.family::<AppletResources>().unwrap().clone()
    }
}

/// A request laid out as libnx lays one out (`cmifMakeRequest`), with no payload.
fn libnx_request(command: u32) -> RequestBuilder {
    RequestBuilder::cmif(CommandType::Request, command).layout(RequestLayout::Libnx)
}

/// `CreateAppletResource` as libnx sends it: the process id asked for, and libnx's own applet
/// resource user id in the payload.
fn create() -> RequestBuilder {
    libnx_request(CREATE_APPLET_RESOURCE)
        .send_pid(PROCESS_ID)
        .data(&PAYLOAD_ARUID.to_le_bytes())
}

/// A whole message buffer that starts with `words` and is zero after them.
fn buffer_of(words: &[u32]) -> Vec<u8> {
    let mut buffer: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    buffer.resize(BUFFER_SIZE, 0);
    buffer
}

/// A reply with one handle, copied or moved, and nothing else: the header with the special
/// header announced, the special header, the handle, then `SFCO`, version, result and token,
/// already on a 16-byte boundary.
fn one_handle_reply(handle: u32, copied: bool) -> Vec<u8> {
    let header = CommandHeader {
        command_type: CommandType::Invalid,
        num_x: 0,
        num_a: 0,
        num_b: 0,
        num_w: 0,
        raw_data_words: 4,
        receive_list: ReceiveList::Disabled,
        has_special_header: true,
    };
    let special = SpecialHeader {
        send_pid: false,
        num_copy_handles: u32::from(copied),
        num_move_handles: u32::from(!copied),
    };
    let [w0, w1] = header.encode().unwrap();
    let success = ResultCode::SUCCESS.raw();
    let special = special.encode().unwrap();
    buffer_of(&[w0, w1, special, handle, SFCO_MAGIC, 0, success, 0])
}

/// A reply that holds only a result.
fn only(result: ResultCode) -> Vec<u8> {
    buffer_of(&[0, 6, 0, 0, SFCO_MAGIC, 0, result.raw(), 0])
}

#[test]
fn create_applet_resource_hands_out_a_session_and_the_reply_is_these_bytes() {
    let mut rig = Rig::new();
    let reply = rig.create_raw();
    let message = Message::parse_reply(&reply, SessionKind::Session).unwrap();
    let handle = message.move_handles[0];
    assert_eq!(reply, one_handle_reply(handle, false));
    let resource = rig.object_of(handle);
    assert_eq!(rig.kernel.sessions(), [resource]);
    assert_eq!(
        (rig.server.session_count(), rig.server.interface_count()),
        (2, 2)
    );
    // The resource is an `IAppletResource`: its command 0 is `GetSharedMemoryHandle`, which
    // copies a handle, where `hid`'s command 0 would move one.
    assert_eq!(rig.memory_of(resource).copies.len(), 1);
}

#[test]
fn get_shared_memory_handle_copies_the_memory_and_the_reply_is_these_bytes() {
    let mut rig = Rig::new();
    let resource = rig.resource();
    let reply = rig.send_raw(resource, &libnx_request(GET_SHARED_MEMORY_HANDLE));
    let message = Message::parse_reply(&reply, SessionKind::Session).unwrap();
    let handle = message.copy_handles[0];
    assert_eq!(reply, one_handle_reply(handle, true));
    // The handle names the memory the table holds for the caller's id, which processes map
    // read-only, of the size libnx maps.
    let memory = rig.object_of(handle);
    assert_eq!(rig.resources().memory(PROCESS_ID), Some(memory));
    let (pages, permission) = rig.kernel.shared_memory(memory).unwrap();
    assert_eq!((pages.size(), permission), (SIZE, SvcPermission::Read));
}

/// The golden bytes of a fresh input shared memory. The offsets are written out from the
/// sources, not computed from this port's constants: Eden's `static_assert`s in
/// `eden/src/hid_core/resources/shared_memory_format.h` and libnx's `HidNpadInternalState`
/// (`libnx/nx/include/switch/services/hid.h`), whose lifos are 0x350 bytes for a controller and
/// 0x708 for a six-axis sensor.
fn golden() -> Vec<u8> {
    let mut image = vec![0_u8; 0x40000];
    let mut put = |at: usize, bytes: &[u8]| image[at..at + bytes.len()].copy_from_slice(bytes);
    // Debug pad, touch screen, mouse, keyboard, home, sleep and capture buttons, gesture and
    // debug mouse: each a lifo whose `total_buffer_count` (at +8) is 17.
    for lifo in [
        0x0, 0x400, 0x3400, 0x3800, 0x4C00, 0x4E00, 0x5000, 0x3BA00, 0x3DC00,
    ] {
        put(lifo + 8, &17_u64.to_le_bytes());
    }
    for entry in 0..10 {
        let base = 0x9A00 + entry * 0x5000;
        // `NoController` in the full-key and the Joy-Con colour states.
        put(base + 0x8, &2_u32.to_le_bytes());
        put(base + 0x14, &2_u32.to_le_bytes());
        // Seven controller lifos, six six-axis lifos, the GameCube trigger lifo.
        for lifo in [
            0x28, 0x378, 0x6C8, 0xA18, 0xD68, 0x10B8, 0x1408, 0x1758, 0x1E60, 0x2568, 0x2C70,
            0x3378, 0x3A80, 0x4228,
        ] {
            put(base + lifo + 8, &17_u64.to_le_bytes());
        }
    }
    // `npad_condition`: `_00`, then `is_initialized`, `hold_type`, `is_valid`.
    put(0x3E200, &[0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]);
    image
}

#[test]
fn the_memory_is_a_fresh_input_shared_memory() {
    let golden = golden();
    assert_eq!(initial_image(), golden);
    // And it is what the guest maps: the pages hold the image.
    let mut rig = Rig::new();
    let resource = rig.resource();
    let memory = rig.memory(resource);
    let (pages, _) = rig.kernel.shared_memory(memory).unwrap();
    let mut held = vec![STALE; golden.len()];
    pages.read(0, &mut held).unwrap();
    assert_eq!(held, golden);
}

#[test]
fn every_lifo_is_listed_once_in_order_and_lies_inside_the_memory() {
    let lifos = lifo_offsets();
    let per_entry = usize::try_from(
        shared_memory::npad::CONTROLLER_LIFO_COUNT + shared_memory::npad::SIX_AXIS_LIFO_COUNT + 1,
    )
    .unwrap();
    let entries = usize::try_from(NPAD_ENTRY_COUNT).unwrap();
    assert_eq!(
        lifos.len(),
        shared_memory::LIFOS.len() + entries * per_entry
    );
    assert!(lifos.windows(2).all(|pair| pair[0] < pair[1]), "{lifos:x?}");
    let last = lifos.last().copied().unwrap();
    assert!(last + LIFO_BUFFER_COUNT + 8 <= SIZE);
    // Every lifo's count is the entry count, and the count of every lifo is set.
    let image = initial_image();
    for lifo in lifos {
        let at = usize::try_from(lifo + LIFO_BUFFER_COUNT).unwrap();
        let count = u64::from_le_bytes(image[at..at + 8].try_into().unwrap());
        assert_eq!(count, LIFO_ENTRY_COUNT, "lifo at {lifo:#x}");
    }
}

#[test]
fn the_applet_resource_user_id_is_the_kernel_s_not_the_payload_s() {
    let mut rig = Rig::new();
    rig.resource();
    assert_eq!(rig.resources().len(), 1);
    assert!(rig.resources().memory(PROCESS_ID).is_some());
    assert_eq!(rig.resources().memory(PAYLOAD_ARUID), None);
}

#[test]
fn an_id_that_asks_again_gets_the_same_memory_and_another_id_its_own() {
    const OTHER: u64 = 0x1111_2222_3333_4444;
    let mut rig = Rig::new();
    let first = rig.resource();
    let first_memory = rig.memory(first);
    let again = rig.resource();
    assert_ne!(again, first);
    let again_memory = rig.memory(again);
    assert_eq!(again_memory, first_memory);
    assert_eq!(rig.resources().len(), 1);

    rig.kernel.set_process_id(OTHER);
    let other = rig.resource();
    let other_memory = rig.memory(other);
    assert_ne!(other_memory, first_memory);
    assert_eq!(rig.resources().memory(OTHER), Some(other_memory));
    assert_eq!(rig.resources().len(), 2);
}

/// Divergence from Eden (module docs): any id gets a row while the table has room; with the
/// table full a new id still gets a resource, whose memory request is Eden's
/// `ResultAruidNotRegistered`.
#[test]
fn a_full_table_gives_a_new_id_no_memory() {
    let mut rig = Rig::new();
    let ids: Vec<u64> = (1..=u64::try_from(ARUID_INDEX_MAX).unwrap()).collect();
    for id in &ids {
        rig.kernel.set_process_id(*id);
        let resource = rig.resource();
        assert_eq!(
            rig.memory_of(resource).result,
            ResultCode::SUCCESS,
            "id {id}"
        );
    }
    // Exactly at the limit, every id has its row.
    assert_eq!(rig.resources().len(), ARUID_INDEX_MAX);

    // One more id: the resource is handed out, but there is no memory for it.
    let late = u64::try_from(ARUID_INDEX_MAX).unwrap() + 1;
    rig.kernel.set_process_id(late);
    let resource = rig.resource();
    let reply = rig.send_raw(resource, &libnx_request(GET_SHARED_MEMORY_HANDLE));
    assert_eq!(reply, only(ARUID_NOT_REGISTERED));
    assert_eq!(rig.resources().len(), ARUID_INDEX_MAX);
    assert_eq!(rig.resources().memory(late), None);

    // An id that has its row still gets its memory with the table full.
    rig.kernel.set_process_id(ids[0]);
    let resource = rig.resource();
    assert_eq!(rig.memory_of(resource).result, ResultCode::SUCCESS);
}

/// Divergence from Eden, which takes the id from the kernel whether or not the request asked.
#[test]
fn create_applet_resource_needs_the_process_id() {
    let mut rig = Rig::new();
    let request = libnx_request(CREATE_APPLET_RESOURCE).data(&PAYLOAD_ARUID.to_le_bytes());
    let reply = rig.send_raw(rig.session, &request);
    assert_eq!(reply, only(INVALID_CMIF_REQUEST));
    assert!(rig.resources().is_empty());
    assert_eq!(rig.server.interface_count(), 1);
}

/// Divergence from Eden, which ignores the resource manager's result.
#[test]
fn a_kernel_failure_to_make_the_memory_is_the_reply_and_keeps_no_row() {
    let mut rig = Rig::new();
    rig.kernel.refuse_objects(Some(KernelError::OutOfResource));
    let reply = rig.create_raw();
    assert_eq!(reply, only(KernelError::OutOfResource.code()));
    assert!(rig.resources().is_empty());
    assert_eq!(rig.server.interface_count(), 1);
    // Once the kernel has room again, the same id gets its row.
    rig.kernel.refuse_objects(None);
    let resource = rig.resource();
    assert_eq!(rig.memory_of(resource).result, ResultCode::SUCCESS);
    assert_eq!(rig.resources().len(), 1);
}

#[test]
fn commands_that_are_left_out_go_to_the_unknown_command_policy() {
    let mut rig = Rig::new();
    let activate = libnx_request(ACTIVATE_DEBUG_PAD)
        .send_pid(PROCESS_ID)
        .data(&PAYLOAD_ARUID.to_le_bytes());
    assert_eq!(
        rig.send_raw(rig.session, &activate),
        only(UNKNOWN_COMMAND_ID)
    );
    // `IAppletResource` has only command 0.
    let resource = rig.resource();
    assert_eq!(
        rig.send_raw(resource, &libnx_request(ACTIVATE_DEBUG_PAD)),
        only(UNKNOWN_COMMAND_ID)
    );
}

#[test]
fn the_tables_name_the_commands_as_eden_does() {
    let name = |session: &dyn Session, id| session.commands().name(Protocol::Cmif, id);
    assert_eq!(
        name(&IHidServer, CREATE_APPLET_RESOURCE),
        Some("create_applet_resource")
    );
    assert_eq!(
        name(&IAppletResource::default(), GET_SHARED_MEMORY_HANDLE),
        Some("get_shared_memory_handle")
    );
    assert_eq!(IHidServer::interface(), "IHidServer");
    assert_eq!(IAppletResource::interface(), "IAppletResource");
}

#[test]
fn a_restored_server_hands_out_the_saved_memory() {
    let mut rig = Rig::new();
    let resource = rig.resource();
    let memory = rig.memory(resource);
    let saved = rig.server.save().unwrap();

    let mut restored = server(UnknownPolicy::Strict);
    restored.restore(&saved).unwrap();
    assert_eq!(restored.save().unwrap(), saved);
    rig.server = restored;
    // The restored resource answers with the same memory, and the restored table gives the
    // same id the same memory again.
    let reply = rig.memory_of(resource);
    assert_eq!(reply.result, ResultCode::SUCCESS);
    assert_eq!(rig.object_of(reply.copies[0]), memory);
    let again = rig.resource();
    assert_eq!(rig.memory(again), memory);
}
