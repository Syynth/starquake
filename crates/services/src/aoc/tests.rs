//! `aoc:u` driven through raw messages over a fake kernel, laid out as Eden's CMIF
//! serialisation reads them (module docs).

use starquake_ipc::{
    BUFFER_SIZE, Body, BufferDescriptor, CommandType, Message, RequestBuilder, RequestLayout,
    SessionKind,
};
use starquake_kernel::handle::Handle;
use starquake_kernel::id::ObjectId;
use starquake_kernel::result::{KernelError, ResultCode};

use super::command::manager::{
    COUNT_ADD_ON_CONTENT, CREATE_EC_PURCHASED_EVENT_MANAGER,
    CREATE_PERMANENT_EC_PURCHASED_EVENT_MANAGER, GET_ADD_ON_CONTENT_LIST_CHANGED_EVENT,
    GET_ADD_ON_CONTENT_LIST_CHANGED_EVENT_WITH_PROCESS_ID, LIST_ADD_ON_CONTENT,
};
use super::command::purchase::{
    GET_PURCHASED_EVENT, POP_PURCHASED_PRODUCT_INFO, POP_PURCHASED_PRODUCT_INFO_WITH_UID,
};
use super::*;
use crate::framework::fake::FakeKernel;
use crate::framework::result::{INVALID_CMIF_REQUEST, UNKNOWN_COMMAND_ID};
use crate::framework::{Handled, Kernel, Protocol, Session};

/// `aoc:u`'s commands this port leaves to the unknown-command policy (module docs, "Left out"),
/// so they have no shipping constants. From Eden's table
/// (`eden/src/core/hle/service/aoc/addon_content_manager.cpp` @ cf654052): those listed without
/// a handler (0, 1, 4, 6, 9, 13, 110, 200, 300, 301, 302), those whose handler only answers
/// success (`PrepareAddOnContent` 7, `NotifyMountAddOnContent` 11, `NotifyUnmountAddOnContent`
/// 12, `CheckAddOnContentMountStatus` 50), and `GetAddOnContentBaseId` (5).
const MANAGER_LEFT_OUT: [u32; 16] = [0, 1, 4, 5, 6, 7, 9, 11, 12, 13, 50, 110, 200, 300, 301, 302];
/// `IPurchaseEventManager`'s commands left out likewise
/// (`eden/src/core/hle/service/aoc/purchase_event_manager.cpp` @ cf654052): whose handlers only
/// answer success, `SetDefaultDeliveryTarget` (0) and `SetDeliveryTarget` (1).
const PURCHASE_LEFT_OUT: [u32; 2] = [0, 1];

/// The caller's process id.
const PROCESS_ID: u64 = 0x0123_4567_89AB_CDEF;
/// Where tests map the room `ListAddOnContent` may write to.
const LIST_AT: u64 = 0x0020_0000;
/// The size of that room: eight ids.
const LIST_ROOM: usize = 32;
/// What fills the message buffer and the list's room beforehand.
const STALE: u8 = 0xA5;

/// A parsed reply.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Answered {
    result: ResultCode,
    data: Vec<u8>,
    copies: Vec<u32>,
    moves: Vec<u32>,
}

/// An `aoc:u` server, a kernel, and one session.
struct Rig {
    kernel: FakeKernel,
    server: Server,
    session: ObjectId,
}

impl Rig {
    fn new() -> Self {
        Self::with_policy(UnknownPolicy::Strict)
    }

    fn with_policy(policy: UnknownPolicy) -> Self {
        let mut kernel = FakeKernel::new(0).unwrap();
        kernel.set_process_id(PROCESS_ID);
        let session = kernel.new_object();
        kernel.map(LIST_AT, vec![STALE; LIST_ROOM]);
        let mut server = u_server(policy);
        server.accept(session).unwrap();
        Self {
            kernel,
            server,
            session,
        }
    }

    fn send(&mut self, session: ObjectId, request: &RequestBuilder) -> Answered {
        let mut buffer = vec![STALE; BUFFER_SIZE];
        request.build(&mut buffer).unwrap();
        let handled = self.server.handle(&mut self.kernel, session, &mut buffer);
        assert_eq!(handled, Ok(Handled::Replied));
        let message = Message::parse_reply(&buffer, SessionKind::Session).unwrap();
        let Body::CmifReply(body) = &message.body else {
            unreachable!("not a CMIF reply: {:?}", message.body);
        };
        Answered {
            result: ResultCode::from_raw(body.header.result),
            data: body.data.to_vec(),
            copies: message.copy_handles.clone(),
            moves: message.move_handles.clone(),
        }
    }

    fn object_of(&self, handle: u32) -> ObjectId {
        self.kernel
            .resolve_handle(Handle::from_raw(handle))
            .unwrap()
    }

    /// Sends a command that returns an event to `session`; the event.
    fn event(&mut self, session: ObjectId, request: &RequestBuilder) -> ObjectId {
        let reply = self.send(session, request);
        assert_eq!(reply.result, ResultCode::SUCCESS);
        assert_eq!(reply.copies.len(), 1);
        self.object_of(reply.copies[0])
    }

    /// Sends `command` (100 or 101); the new manager's session.
    fn manager(&mut self, command: u32) -> ObjectId {
        let reply = self.send(self.session, &request(command));
        assert_eq!(reply.result, ResultCode::SUCCESS);
        self.object_of(reply.moves[0])
    }

    fn family(&self) -> AddOnContent {
        self.server.family::<AddOnContent>().unwrap().clone()
    }
}

fn request(command: u32) -> RequestBuilder {
    RequestBuilder::cmif(CommandType::Request, command).layout(RequestLayout::Libnx)
}

/// A command whose one in argument is a client process id.
fn with_process_id(command: u32) -> RequestBuilder {
    request(command)
        .send_pid(PROCESS_ID)
        .data(&0u64.to_le_bytes())
}

/// `ListAddOnContent`: offset, count, then the process id's word at its alignment, and the
/// room for the list as a B descriptor.
fn list(offset: u32, count: u32) -> RequestBuilder {
    let mut data = Vec::new();
    data.extend(offset.to_le_bytes());
    data.extend(count.to_le_bytes());
    data.extend(0u64.to_le_bytes());
    request(LIST_ADD_ON_CONTENT)
        .send_pid(PROCESS_ID)
        .data(&data)
        .receive_buffer(BufferDescriptor {
            address: LIST_AT,
            size: u64::try_from(LIST_ROOM).unwrap(),
            flags: 0,
        })
}

fn reply_u32(reply: &Answered) -> u32 {
    u32::from_le_bytes(reply.data[..4].try_into().unwrap())
}

#[test]
fn count_add_on_content_counts_none() {
    let mut rig = Rig::new();
    let reply = rig.send(rig.session, &with_process_id(COUNT_ADD_ON_CONTENT));
    assert_eq!(reply.result, ResultCode::SUCCESS);
    // The count, not a stale word of the message buffer.
    assert_eq!(reply_u32(&reply), 0);
}

#[test]
fn list_add_on_content_lists_none_at_offset_zero_whatever_the_count() {
    for count in [0, 1, 8, u32::MAX] {
        let mut rig = Rig::new();
        let reply = rig.send(rig.session, &list(0, count));
        assert_eq!(reply.result, ResultCode::SUCCESS, "count {count}");
        assert_eq!(reply_u32(&reply), 0, "count {count}");
        // Nothing is written to the room.
        assert_eq!(
            rig.kernel.memory(LIST_AT, LIST_ROOM),
            Some(&[STALE; LIST_ROOM][..])
        );
    }
}

#[test]
fn an_offset_past_the_empty_list_is_application_record_not_found() {
    // Eden's `R_UNLESS(out.size() >= offset, ..)` with no add-on content: offset 0 is the edge
    // that passes (above), and every offset past it fails.
    for offset in [1, 2, u32::MAX] {
        let mut rig = Rig::new();
        let reply = rig.send(rig.session, &list(offset, 1));
        assert_eq!(
            reply.result, APPLICATION_RECORD_NOT_FOUND,
            "offset {offset}"
        );
        assert_eq!(
            rig.kernel.memory(LIST_AT, LIST_ROOM),
            Some(&[STALE; LIST_ROOM][..])
        );
    }
    // An offset that differs from 0 only above its low byte is past the list too.
    let mut rig = Rig::new();
    assert_eq!(
        rig.send(rig.session, &list(0x100, 1)).result,
        APPLICATION_RECORD_NOT_FOUND
    );
}

#[test]
fn the_result_codes_are_edens() {
    // `{ErrorModule::AM, 37}` and `{ErrorModule::NIMShop, 400}`.
    assert_eq!(
        (
            APPLICATION_RECORD_NOT_FOUND.module(),
            APPLICATION_RECORD_NOT_FOUND.description()
        ),
        (AM_MODULE, 37)
    );
    assert_eq!(
        (
            NO_PURCHASED_PRODUCT_INFO_AVAILABLE.module(),
            NO_PURCHASED_PRODUCT_INFO_AVAILABLE.description()
        ),
        (NIM_SHOP_MODULE, 400)
    );
    assert_eq!((AM_MODULE, NIM_SHOP_MODULE), (128, 164));
}

#[test]
fn a_list_request_too_short_for_its_offset_and_count_is_refused() {
    // A payload of one word, the offset only, in the exact layout (no reserve of padding after
    // it): the count's read passes the end of the payload. The offset is 0, so a read of the
    // count that went on past the end would answer success.
    let mut rig = Rig::new();
    let short = RequestBuilder::cmif(CommandType::Request, LIST_ADD_ON_CONTENT)
        .data(&0u32.to_le_bytes())
        .receive_buffer(BufferDescriptor {
            address: LIST_AT,
            size: u64::try_from(LIST_ROOM).unwrap(),
            flags: 0,
        });
    let reply = rig.send(rig.session, &short);
    assert_eq!(reply.result, INVALID_CMIF_REQUEST);
}

#[test]
fn a_list_request_without_its_room_is_refused() {
    let mut rig = Rig::new();
    let mut data = Vec::new();
    data.extend(0u32.to_le_bytes());
    data.extend(1u32.to_le_bytes());
    data.extend(0u64.to_le_bytes());
    let reply = rig.send(rig.session, &request(LIST_ADD_ON_CONTENT).data(&data));
    assert_eq!(reply.result, INVALID_CMIF_REQUEST);
}

#[test]
fn a_new_server_has_no_list_changed_event_and_a_new_manager_no_purchased_event() {
    assert_eq!(AddOnContent::default().list_changed_event(), None);
    assert_eq!(IPurchaseEventManager::default().purchased_event(), None);
    assert_eq!(Rig::new().family().list_changed_event(), None);
}

#[test]
fn both_list_changed_commands_hand_out_one_event_for_every_session() {
    let mut rig = Rig::new();
    let first = rig.event(rig.session, &request(GET_ADD_ON_CONTENT_LIST_CHANGED_EVENT));
    assert_eq!(rig.family().list_changed_event(), Some(first));
    // Never signalled.
    assert_eq!(rig.kernel.signals(first), Some(0));
    let with_id = rig.event(
        rig.session,
        &with_process_id(GET_ADD_ON_CONTENT_LIST_CHANGED_EVENT_WITH_PROCESS_ID),
    );
    assert_eq!(with_id, first);
    // A second session of the server gets the same event.
    let other = rig.kernel.new_object();
    rig.server.accept(other).unwrap();
    let again = rig.event(other, &request(GET_ADD_ON_CONTENT_LIST_CHANGED_EVENT));
    assert_eq!(again, first);
    assert_eq!(rig.kernel.signals(first), Some(0));
    // The other command first makes it too.
    let mut rig = Rig::new();
    let event = rig.event(
        rig.session,
        &with_process_id(GET_ADD_ON_CONTENT_LIST_CHANGED_EVENT_WITH_PROCESS_ID),
    );
    assert_eq!(rig.family().list_changed_event(), Some(event));
}

#[test]
fn both_create_commands_hand_out_a_manager_with_its_own_event() {
    let mut rig = Rig::new();
    let managers = [
        rig.manager(CREATE_EC_PURCHASED_EVENT_MANAGER),
        rig.manager(CREATE_PERMANENT_EC_PURCHASED_EVENT_MANAGER),
    ];
    assert_eq!(rig.server.session_count(), 3);
    let events = managers.map(|manager| rig.event(manager, &request(GET_PURCHASED_EVENT)));
    assert_ne!(events[0], events[1]);
    // Asked again, a manager hands out the event it made.
    for (manager, event) in managers.into_iter().zip(events) {
        assert_eq!(rig.event(manager, &request(GET_PURCHASED_EVENT)), event);
        assert_eq!(rig.kernel.signals(event), Some(0));
    }
    // Neither is the server's list-changed event.
    let list_changed = rig.event(rig.session, &request(GET_ADD_ON_CONTENT_LIST_CHANGED_EVENT));
    assert!(!events.contains(&list_changed));
}

#[test]
fn a_manager_has_no_purchased_product_info() {
    let mut rig = Rig::new();
    let manager = rig.manager(CREATE_EC_PURCHASED_EVENT_MANAGER);
    for command in [
        POP_PURCHASED_PRODUCT_INFO,
        POP_PURCHASED_PRODUCT_INFO_WITH_UID,
    ] {
        let reply = rig.send(manager, &request(command));
        assert_eq!(
            reply.result, NO_PURCHASED_PRODUCT_INFO_AVAILABLE,
            "{command}"
        );
    }
}

#[test]
fn the_root_is_not_a_manager_and_a_manager_is_not_the_root() {
    let mut rig = Rig::new();
    let manager = rig.manager(CREATE_EC_PURCHASED_EVENT_MANAGER);
    // The root's 100 is not a manager's command, and a manager's 3 is not the root's.
    assert_eq!(
        rig.send(manager, &request(CREATE_EC_PURCHASED_EVENT_MANAGER))
            .result,
        UNKNOWN_COMMAND_ID
    );
    assert_eq!(
        rig.send(rig.session, &request(POP_PURCHASED_PRODUCT_INFO))
            .result,
        COUNT_IS_NOT_POP
    );
}

/// What the root answers for command 3, `ListAddOnContent`, sent with no room: the framework's
/// refusal of a request without its declared buffer, not the manager's pop result.
const COUNT_IS_NOT_POP: ResultCode = INVALID_CMIF_REQUEST;

#[test]
fn commands_that_are_left_out_go_to_the_unknown_command_policy() {
    // Under the strict policy, each is unknown.
    let mut rig = Rig::new();
    let manager = rig.manager(CREATE_EC_PURCHASED_EVENT_MANAGER);
    for command in MANAGER_LEFT_OUT {
        assert_eq!(
            rig.send(rig.session, &with_process_id(command)).result,
            UNKNOWN_COMMAND_ID,
            "{command}"
        );
    }
    for command in PURCHASE_LEFT_OUT {
        assert_eq!(
            rig.send(manager, &with_process_id(command)).result,
            UNKNOWN_COMMAND_ID,
            "{command}"
        );
    }
    // Under the auto-stub policy, each answers success with a payload of zeroes, and each is
    // recorded against its own interface (module docs, "Left out").
    let mut rig = Rig::with_policy(UnknownPolicy::AutoStub);
    let manager = rig.manager(CREATE_EC_PURCHASED_EVENT_MANAGER);
    let mut expected = Vec::new();
    for (session, interface, commands) in [
        (rig.session, "IAddOnContentManager", &MANAGER_LEFT_OUT[..]),
        (manager, "IPurchaseEventManager", &PURCHASE_LEFT_OUT[..]),
    ] {
        for &command in commands {
            let reply = rig.send(session, &with_process_id(command));
            assert_eq!(reply.result, ResultCode::SUCCESS, "{command}");
            assert!(reply.data.iter().all(|byte| *byte == 0), "{command}");
            expected.push(crate::framework::StubbedCommand {
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
fn a_server_restored_onto_its_kernel_hands_out_the_same_events() {
    let mut rig = Rig::new();
    let list_changed = rig.event(rig.session, &request(GET_ADD_ON_CONTENT_LIST_CHANGED_EVENT));
    let manager = rig.manager(CREATE_EC_PURCHASED_EVENT_MANAGER);
    let purchased = rig.event(manager, &request(GET_PURCHASED_EVENT));
    let saved = rig.server.save().unwrap();
    let mut restored = u_server(UnknownPolicy::Strict);
    restored.restore(&saved).unwrap();
    assert_eq!(restored.save().unwrap(), saved);
    rig.server = restored;
    assert_eq!(rig.family().list_changed_event(), Some(list_changed));
    assert_eq!(
        rig.event(rig.session, &request(GET_ADD_ON_CONTENT_LIST_CHANGED_EVENT)),
        list_changed
    );
    assert_eq!(rig.event(manager, &request(GET_PURCHASED_EVENT)), purchased);
    assert_eq!(rig.kernel.signals(list_changed), Some(0));
    assert_eq!(rig.kernel.signals(purchased), Some(0));
}

#[test]
fn a_restored_event_the_kernel_did_not_make_is_refused() {
    let mut rig = Rig::new();
    let list_changed = rig.event(rig.session, &request(GET_ADD_ON_CONTENT_LIST_CHANGED_EVENT));
    let manager = rig.manager(CREATE_EC_PURCHASED_EVENT_MANAGER);
    let purchased = rig.event(manager, &request(GET_PURCHASED_EVENT));
    let saved = rig.server.save().unwrap();

    for names_an_object in [true, false] {
        // A fresh kernel mints the same session ids; the events' ids are made here as objects
        // that are not events, or never made.
        let mut fresh = Rig::new();
        assert_eq!(fresh.session, rig.session);
        if names_an_object {
            let mut minted = Vec::new();
            while minted.last() != Some(&list_changed.max(purchased).max(manager)) {
                minted.push(fresh.kernel.new_object());
            }
            assert!(minted.contains(&list_changed) && minted.contains(&purchased));
        }
        fresh.server.restore(&saved).unwrap();
        let handles = fresh.kernel.handle_count();
        for _ in 0..2 {
            let reply = fresh.send(
                fresh.session,
                &request(GET_ADD_ON_CONTENT_LIST_CHANGED_EVENT),
            );
            assert_eq!(
                reply.result,
                KernelError::InvalidHandle.code(),
                "{names_an_object}"
            );
            assert_eq!(fresh.family().list_changed_event(), Some(list_changed));
            let reply = fresh.send(manager, &request(GET_PURCHASED_EVENT));
            assert_eq!(
                reply.result,
                KernelError::InvalidHandle.code(),
                "{names_an_object}"
            );
            assert_eq!(fresh.kernel.handle_count(), handles);
        }
    }
}

#[test]
fn the_tables_name_the_commands() {
    let manager = IAddOnContentManager.commands();
    let names = [
        (COUNT_ADD_ON_CONTENT, "count_add_on_content"),
        (LIST_ADD_ON_CONTENT, "list_add_on_content"),
        (GET_ADD_ON_CONTENT_LIST_CHANGED_EVENT, "list_changed_event"),
        (
            GET_ADD_ON_CONTENT_LIST_CHANGED_EVENT_WITH_PROCESS_ID,
            "list_changed_event",
        ),
        (
            CREATE_EC_PURCHASED_EVENT_MANAGER,
            "create_purchased_event_manager",
        ),
        (
            CREATE_PERMANENT_EC_PURCHASED_EVENT_MANAGER,
            "create_purchased_event_manager",
        ),
    ];
    for (id, name) in names {
        assert_eq!(manager.name(Protocol::Cmif, id), Some(name));
    }
    assert_eq!(manager.cmif.len(), names.len());
    let purchase = IPurchaseEventManager::default().commands();
    assert_eq!(
        purchase.name(Protocol::Cmif, GET_PURCHASED_EVENT),
        Some("get_purchased_event")
    );
    assert_eq!(purchase.cmif.len(), 3);
    assert_eq!(IAddOnContentManager::interface(), "IAddOnContentManager");
    assert_eq!(IPurchaseEventManager::interface(), "IPurchaseEventManager");
    assert_eq!(U_NAME, "aoc:u");
}
