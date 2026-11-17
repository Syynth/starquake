//! New sessions handed out as copied handles ([`Reply::copy_session`], [`Reply::copy_service`]),
//! over a [`FakeHost`] so that a copied session on another service can be sent requests.
//!
//! A desk hands out sessions: on itself, on a second service, and a mix with a moved one, so
//! that each handle's list and place, and the server each session is on, are seen.

use serde::{Deserialize, Serialize};
use starquake_ipc::{BUFFER_SIZE, Body, CommandType, Message, RequestBuilder, SessionKind};
use starquake_kernel::handle::Handle;
use starquake_kernel::result::ResultCode;
use starquake_trace::ServiceId;

use super::super::control::CONVERT_CURRENT_OBJECT_TO_DOMAIN;
use super::super::fake::{FakeHost, FakeKernel};
use super::super::{Context, Handled, Kernel, Outcome, Reply, Request, Server, UnknownPolicy};
use super::{STALE, request};
use crate::commands;

/// The desk's server and the number it is known by.
const DESK: ServiceId = ServiceId(11);
/// The server whose sessions the desk hands out with [`Reply::copy_service`].
const LOBBY: ServiceId = ServiceId(12);

/// The desk's commands.
mod desk {
    /// Copies a session on the desk's own server, leading to a [`Card`](super::Card) whose tag
    /// is [`TAG`](super::TAG).
    pub const COPY_CARD: u32 = 0;
    /// Copies a session on the lobby's server.
    pub const COPY_LOBBY: u32 = 1;
    /// Copies a card, then a lobby session, and moves a card: three handles, two lists.
    pub const MIXED: u32 = 2;
}

/// The tag the desk gives the card it copies.
const TAG: u32 = 0x5EED_0001;
/// The tag of the card the desk moves in [`desk::MIXED`].
const MOVED_TAG: u32 = 0x5EED_0002;
/// The word every lobby session answers its command 0 with.
const LOBBY_WORD: u32 = 0x10BB_0001;

/// The desk: the root of the desk's server.
#[derive(Default, Serialize, Deserialize)]
struct Desk;

/// A card the desk hands out. Command 0 answers its tag.
#[derive(Default, Serialize, Deserialize)]
struct Card {
    tag: u32,
}

/// The lobby's root. Command 0 answers [`LOBBY_WORD`].
#[derive(Default, Serialize, Deserialize)]
struct Lobby;

commands! {
    impl Desk {
        desk::COPY_CARD => copy_card,
        desk::COPY_LOBBY => copy_lobby,
        desk::MIXED => mixed,
    }
}

commands! {
    impl Card {
        0 => tag,
    }
}

commands! {
    impl Lobby {
        0 => word,
    }
}

impl Desk {
    fn copy_card(_: &mut Self, _: &mut Context<'_>, _: &Request<'_>, reply: &mut Reply) -> Outcome {
        reply.copy_session(Box::new(Card { tag: TAG }));
        reply.ok()
    }

    fn copy_lobby(_: &mut Self, _: &mut Context<'_>, _: &Request<'_>, reply: &mut Reply) -> Outcome {
        reply.copy_service(LOBBY);
        reply.ok()
    }

    fn mixed(_: &mut Self, _: &mut Context<'_>, _: &Request<'_>, reply: &mut Reply) -> Outcome {
        reply.push_session(Box::new(Card { tag: MOVED_TAG }));
        reply.copy_session(Box::new(Card { tag: TAG }));
        reply.copy_service(LOBBY);
        reply.ok()
    }
}

impl Card {
    fn tag(&mut self, _: &mut Context<'_>, _: &Request<'_>, reply: &mut Reply) -> Outcome {
        reply.payload().write_u32(self.tag);
        reply.ok()
    }
}

impl Lobby {
    fn word(_: &mut Self, _: &mut Context<'_>, _: &Request<'_>, reply: &mut Reply) -> Outcome {
        reply.payload().write_u32(LOBBY_WORD);
        reply.ok()
    }
}

/// A host with the desk and the lobby, and a session on the desk.
fn host() -> (FakeHost, Handle) {
    let mut host = FakeHost::new(FakeKernel::new(0).unwrap());
    host.add(
        Server::new("desk", UnknownPolicy::Strict)
            .with_service_id(DESK)
            .with_root(|| Box::new(Desk)),
    );
    host.add(
        Server::new("lobby", UnknownPolicy::Strict)
            .with_service_id(LOBBY)
            .with_root(|| Box::new(Lobby)),
    );
    let desk = host.connect(DESK).unwrap();
    (host, desk)
}

/// Sends `request` on `session` and returns the reply's message buffer.
fn send(host: &mut FakeHost, session: Handle, request: &RequestBuilder) -> Vec<u8> {
    let mut buffer = vec![STALE; BUFFER_SIZE];
    request.build(&mut buffer).unwrap();
    assert_eq!(host.send(session, &mut buffer), Ok(Handled::Replied));
    buffer
}

/// The reply's result, and its copied and moved handles.
fn handles(buffer: &[u8], kind: SessionKind) -> (u32, Vec<Handle>, Vec<Handle>) {
    let message = Message::parse_reply(buffer, kind).unwrap();
    let result = match &message.body {
        Body::CmifReply(reply) => reply.header.result,
        other => unreachable!("not a CMIF reply: {other:?}"),
    };
    let to = |list: &[u32]| list.iter().copied().map(Handle::from_raw).collect();
    (result, to(&message.copy_handles), to(&message.move_handles))
}

/// The word a session answers command 0 with, and the service its session is on.
fn answer(host: &mut FakeHost, session: Handle) -> (u32, Option<ServiceId>) {
    let object = host.kernel.resolve_handle(session).unwrap();
    let owner = host.kernel.owner(object);
    let reply = send(host, session, &request(0));
    let message = Message::parse_reply(&reply, SessionKind::Session).unwrap();
    let Body::CmifReply(reply) = &message.body else {
        unreachable!("not a CMIF reply");
    };
    assert_eq!(reply.header.result, ResultCode::SUCCESS.raw());
    let word = u32::from_le_bytes(reply.data[..4].try_into().unwrap());
    (word, owner)
}

#[test]
fn a_copied_session_is_a_new_session_on_this_server_in_the_copy_list() {
    let (mut host, desk) = host();
    let reply = send(&mut host, desk, &request(desk::COPY_CARD));
    let (result, copies, moves) = handles(&reply, SessionKind::Session);
    assert_eq!(result, ResultCode::SUCCESS.raw());
    assert_eq!(moves, []);
    assert_eq!(copies.len(), 1);
    assert_eq!(host.server(DESK).unwrap().session_count(), 2);
    assert_eq!(answer(&mut host, copies[0]), (TAG, Some(DESK)));
}

#[test]
fn a_copied_service_is_a_new_session_on_its_server_in_the_copy_list() {
    let (mut host, desk) = host();
    let reply = send(&mut host, desk, &request(desk::COPY_LOBBY));
    let (result, copies, moves) = handles(&reply, SessionKind::Session);
    assert_eq!(result, ResultCode::SUCCESS.raw());
    assert_eq!(moves, []);
    assert_eq!(copies.len(), 1);
    // The lobby's server opened its root for it.
    assert_eq!(host.server(LOBBY).unwrap().session_count(), 1);
    assert_eq!(answer(&mut host, copies[0]), (LOBBY_WORD, Some(LOBBY)));
}

/// Copies are listed in the order the method made them, whatever it moved in between.
#[test]
fn copied_and_moved_handles_each_keep_their_own_list_and_order() {
    let (mut host, desk) = host();
    let reply = send(&mut host, desk, &request(desk::MIXED));
    let (result, copies, moves) = handles(&reply, SessionKind::Session);
    assert_eq!(result, ResultCode::SUCCESS.raw());
    assert_eq!(copies.len(), 2);
    assert_eq!(moves.len(), 1);
    assert_eq!(answer(&mut host, copies[0]), (TAG, Some(DESK)));
    assert_eq!(answer(&mut host, copies[1]), (LOBBY_WORD, Some(LOBBY)));
    assert_eq!(answer(&mut host, moves[0]), (MOVED_TAG, Some(DESK)));
}

/// In a domain a pushed interface becomes an object id, but a copied one is still a session of
/// its own: a copied handle cannot be a domain object.
#[test]
fn in_a_domain_a_copied_session_is_still_a_session_handle() {
    let (mut host, desk) = host();
    let convert = RequestBuilder::cmif(CommandType::Control, CONVERT_CURRENT_OBJECT_TO_DOMAIN);
    send(&mut host, desk, &convert);
    let ask = RequestBuilder::domain(CommandType::Request, 1, desk::COPY_CARD);
    let reply = send(&mut host, desk, &ask);
    let message = Message::parse_reply(&reply, SessionKind::Domain).unwrap();
    let Body::CmifReply(body) = &message.body else {
        unreachable!("not a CMIF reply");
    };
    assert_eq!(body.header.result, ResultCode::SUCCESS.raw());
    assert_eq!(body.object_ids, Vec::<u32>::new());
    assert_eq!(message.move_handles, Vec::<u32>::new());
    assert_eq!(message.copy_handles.len(), 1);
    let copied = Handle::from_raw(message.copy_handles[0]);
    assert_eq!(answer(&mut host, copied), (TAG, Some(DESK)));
}
