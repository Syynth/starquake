//! The framework driven end to end: a toy service, a [`FakeKernel`], and raw messages built with
//! `starquake-ipc`'s request builder.
//!
//! Each test sends the bytes a guest would put in its message buffer to [`Server::handle`] and
//! checks the bytes that come back, and what changed in the server and the kernel.

use std::cell::RefCell;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use serde::{Deserialize, Serialize};
use starquake_ipc::{BUFFER_SIZE, Body, CommandType, Message, RequestBuilder, SessionKind};
use starquake_kernel::handle::Handle;
use starquake_kernel::id::{ObjectId, SlabId};
use starquake_kernel::result::ResultCode;
use starquake_trace::ServiceId;

use super::control::CONVERT_CURRENT_OBJECT_TO_DOMAIN;
use super::fake::FakeKernel;
use super::{
    Context, Fault, Handled, Interfaces, Kernel, Notify, Outcome, Protocol, Reply, ReplyToken,
    Request, Server, ServerError, UnknownPolicy, Writer,
};
use crate::commands;

mod buffers;
mod control;
mod deferred;
mod domain;
mod handles;
mod later;
mod plain;
mod queue;
mod server;
mod shared_memory;
mod snapshot;
mod trace;
mod unknown;

/// What fills a message buffer before a request is built into it, and the caller's memory
/// before a test writes to it: a byte that is neither zero nor part of any expected value.
const STALE: u8 = 0xA5;

/// The toy service's root interface.
#[derive(Default, Serialize, Deserialize)]
struct Toy {
    /// Events made by `make_event`, oldest first.
    events: Vec<ObjectId>,
}

// The CMIF rows name their ids by constant and the TIPC rows by literal: the macro takes both.
commands! {
    impl Toy {
        toy::ADD => add,
        toy::OPEN_CHILD => open_child,
        toy::CHILD_TAG => child_tag,
        toy::INSPECT => inspect,
        toy::UPPER => upper [In(AutoSelect), Out(AutoSelect)],
        toy::MAKE_EVENT => make_event,
        toy::FAIL => fail,
        toy::FLOOD => flood,
        toy::SPAWN => spawn,
        toy::INTERLEAVE => interleave,
        toy::POKE => poke,
        toy::SIGNAL => signal,
        toy::OPEN_MUTE => open_mute,
        toy::ODD => odd,
        toy::WAIT => wait,
        toy::WAIT_TWICE => wait_twice,
        toy::FORGET => forget,
        toy::HASTY => hasty,
        toy::BUMP => bump,
        toy::LEND => lend,
        toy::COPY_MIXED => copy_mixed [In(Pointer), In(MapAlias), Out(MapAlias), Out(Pointer)],
        toy::PEEK_UNDECLARED => peek_undeclared [In(MapAlias)],
        toy::FILL => fill [Out(MapAlias)],
    }
    tipc {
        0 => sum,
        1 => open_child,
        2 => inspect,
        3 => wait,
        // CMIF `upper` (also id 4) declares auto-select buffers; this row map-alias ones, the
        // only kind the console's TIPC carries.
        4 => upper [In(MapAlias), Out(MapAlias)],
    }
}

thread_local! {
    /// Where the toy's deferring methods leave their reply tokens: the test's stand-in for the
    /// worker a real service hands its token to. One per test thread, so tests do not see each
    /// other's tokens.
    static PARKED: RefCell<Vec<ReplyToken>> = const { RefCell::new(Vec::new()) };
}

/// The tokens the toy has parked on this thread since the last call, oldest first.
fn parked() -> Vec<ReplyToken> {
    PARKED.take()
}

/// The one token the toy has parked since the last call.
fn parked_one() -> ReplyToken {
    let mut tokens = parked();
    assert_eq!(tokens.len(), 1);
    tokens.remove(0)
}

/// The toy family's state: what every toy session of a server shares.
#[derive(Debug, Default, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct Tally {
    bumps: u32,
}

impl Toy {
    /// in: `u32 a`, 4 bytes of padding, `u64 b`. out: `u64 a + b`.
    fn add(_: &mut Self, _cx: &mut Context<'_>, req: &Request<'_>, reply: &mut Reply) -> Outcome {
        let mut r = req.payload();
        let a = r.read_u32()?;
        r.align(8)?;
        let b = r.read_u64()?;
        reply.payload().write_u64(u64::from(a).wrapping_add(b));
        reply.ok()
    }

    /// in: `u32 tag`. out: a new [`Child`] with that tag.
    fn open_child(
        _: &mut Self,
        _cx: &mut Context<'_>,
        req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        let tag = req.payload().read_u32()?;
        reply.push_session(Box::new(Child { tag }));
        reply.ok()
    }

    /// in: input object 0, a [`Child`]. out: `u32` its tag. The child's tag is then bumped, to
    /// show the method reached the object itself.
    fn child_tag(
        _: &mut Self,
        cx: &mut Context<'_>,
        req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        let child = req.in_object::<Child>(cx, 0)?;
        reply.payload().write_u32(child.tag);
        child.tag = child.tag.wrapping_add(1);
        reply.ok()
    }

    /// Reports what the request carried, so tests can tell lists apart. out: `u64` process id
    /// ([`ABSENT`] widened if none was sent); the object index behind copy handles 0 and 1 and
    /// move handles 0 and 1 as `u32`s ([`ABSENT`] for a handle that is missing or names nothing);
    /// `u32` command id; `u32` 1 for TIPC and 0 for CMIF.
    fn inspect(
        _: &mut Self,
        _cx: &mut Context<'_>,
        req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        let index = |object: Result<ObjectId, Fault>| object.map_or(ABSENT, SlabId::index);
        let w = reply.payload();
        w.write_u64(req.process_id().unwrap_or(u64::from(ABSENT)));
        w.write_u32(index(req.copy_handle(0)));
        w.write_u32(index(req.copy_handle(1)));
        w.write_u32(index(req.move_handle(0)));
        w.write_u32(index(req.move_handle(1)));
        w.write_u32(req.command_id());
        w.write_u32(u32::from(req.protocol() == Protocol::Tipc));
        reply.ok()
    }

    /// in: copy handle 0, an event. Signals it.
    fn signal(_: &mut Self, cx: &mut Context<'_>, req: &Request<'_>, reply: &mut Reply) -> Outcome {
        cx.signal_event(req.copy_handle(0)?)?;
        reply.ok()
    }

    /// out: `u8` [`ODD_BYTE`], and a new [`Child`] tagged with it.
    fn odd(_: &mut Self, _cx: &mut Context<'_>, _req: &Request<'_>, reply: &mut Reply) -> Outcome {
        reply.payload().write_u8(ODD_BYTE);
        let tag = u32::from(ODD_BYTE);
        reply.push_session(Box::new(Child { tag }));
        reply.ok()
    }

    /// out: a new [`Mute`].
    fn open_mute(
        _: &mut Self,
        _cx: &mut Context<'_>,
        _req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        reply.push_session(Box::new(Mute));
        reply.ok()
    }

    /// in: in buffer 0 (auto-select), at most [`UPPER_LIMIT`] bytes of text. out: out buffer 0
    /// (auto-select), the text in upper case; `u32` its length.
    fn upper(_: &mut Self, cx: &mut Context<'_>, req: &Request<'_>, reply: &mut Reply) -> Outcome {
        let text = req.in_buffer(cx, 0)?.read_all(UPPER_LIMIT)?;
        let mut out = req.out_buffer(cx, 0)?;
        out.write(&text.to_ascii_uppercase())?;
        let written = out.written();
        reply
            .payload()
            .write_u32(u32::try_from(written).unwrap_or(u32::MAX));
        reply.ok()
    }

    /// out: copy handle 0, a new event, already signalled once.
    fn make_event(
        &mut self,
        cx: &mut Context<'_>,
        _req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        let event = cx.create_event()?;
        cx.signal_event(event)?;
        self.events.push(event);
        reply.copy_handle(event);
        reply.ok()
    }

    /// in: `u32` a result code. Fills the reply as a successful command would, then fails with
    /// that code.
    fn fail(_: &mut Self, cx: &mut Context<'_>, req: &Request<'_>, reply: &mut Reply) -> Outcome {
        let code = ResultCode::from_raw(req.payload().read_u32()?);
        reply.payload().write_u64(FAIL_PAYLOAD);
        reply.copy_handle(cx.create_event()?);
        reply.push_session(Box::new(Child { tag: 0 }));
        Err(code.into())
    }

    /// in: `u32 n`. out: `n` bytes of payload, a copy handle and a new [`Child`].
    fn flood(_: &mut Self, cx: &mut Context<'_>, req: &Request<'_>, reply: &mut Reply) -> Outcome {
        let n = usize::try_from(req.payload().read_u32()?).unwrap_or(usize::MAX);
        reply
            .payload()
            .write_bytes(&vec![0xEE; n.min(2 * BUFFER_SIZE)]);
        reply.copy_handle(cx.create_event()?);
        reply.push_session(Box::new(Child { tag: 0 }));
        reply.ok()
    }

    /// in: `u32 n`, `u32 first_tag`. out: `n` new children tagged from `first_tag` up.
    fn spawn(_: &mut Self, _cx: &mut Context<'_>, req: &Request<'_>, reply: &mut Reply) -> Outcome {
        let mut r = req.payload();
        let (n, first_tag) = (r.read_u32()?, r.read_u32()?);
        for i in 0..n {
            reply.push_session(Box::new(Child {
                tag: first_tag.wrapping_add(i),
            }));
        }
        reply.ok()
    }

    /// out, in this order: a moved event, a new child tagged [`INTERLEAVED_TAG`], a second moved
    /// event, and one copied event.
    fn interleave(
        &mut self,
        cx: &mut Context<'_>,
        _req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        let (first, second, copied) = (cx.create_event()?, cx.create_event()?, cx.create_event()?);
        self.events.extend([first, second, copied]);
        reply.move_handle(first);
        reply.push_session(Box::new(Child {
            tag: INTERLEAVED_TAG,
        }));
        reply.move_handle(second);
        reply.copy_handle(copied);
        reply.ok()
    }

    /// in: `u64 from`, `u64 to`. Copies four bytes of the caller's memory from one address to
    /// the other, reversed.
    fn poke(_: &mut Self, cx: &mut Context<'_>, req: &Request<'_>, reply: &mut Reply) -> Outcome {
        let mut r = req.payload();
        let (from, to) = (r.read_u64()?, r.read_u64()?);
        let mut bytes = [0; 4];
        cx.read_memory(from, &mut bytes)?;
        bytes.reverse();
        cx.write_memory(to, &bytes)?;
        reply.ok()
    }

    /// Takes the request's reply token, parks it, and defers. What it writes into the reply
    /// first must never be sent.
    fn wait(_: &mut Self, cx: &mut Context<'_>, _req: &Request<'_>, reply: &mut Reply) -> Outcome {
        reply.payload().write_u64(FAIL_PAYLOAD);
        let token = cx.defer()?;
        PARKED.with_borrow_mut(|parked| parked.push(token));
        reply.deferred()
    }

    /// Asks for two reply tokens.
    fn wait_twice(
        this: &mut Self,
        cx: &mut Context<'_>,
        req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        Self::wait(this, cx, req, reply)?;
        Self::wait(this, cx, req, reply)
    }

    /// Defers without taking a reply token.
    fn forget(
        _: &mut Self,
        _cx: &mut Context<'_>,
        _req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        reply.deferred()
    }

    /// Takes a reply token, parks it, and answers at once. out: `u32` [`HASTY`].
    fn hasty(_: &mut Self, cx: &mut Context<'_>, _req: &Request<'_>, reply: &mut Reply) -> Outcome {
        let token = cx.defer()?;
        PARKED.with_borrow_mut(|parked| parked.push(token));
        reply.payload().write_u32(HASTY);
        reply.ok()
    }

    /// Counts in the family's state. out: `u32` the count so far, this call included.
    fn bump(_: &mut Self, cx: &mut Context<'_>, _req: &Request<'_>, reply: &mut Reply) -> Outcome {
        let tally = cx.family::<Tally>()?;
        tally.bumps = tally.bumps.wrapping_add(1);
        reply.payload().write_u32(tally.bumps);
        reply.ok()
    }

    /// in: `u32` a service's number. out: `u8` [`ODD_BYTE`]; a new session on that service,
    /// after a moved event.
    fn lend(_: &mut Self, cx: &mut Context<'_>, req: &Request<'_>, reply: &mut Reply) -> Outcome {
        let service = ServiceId(req.payload().read_u32()?);
        reply.payload().write_u8(ODD_BYTE);
        reply.move_handle(cx.create_event()?);
        reply.push_service(service);
        reply.ok()
    }

    /// in: in buffer 0 (pointer) and in buffer 1 (map-alias), at most [`UPPER_LIMIT`] bytes each.
    /// out: out buffer 0 (map-alias) gets in buffer 1's bytes and out buffer 1 (pointer) in
    /// buffer 0's, so a test sees which descriptor each index was.
    fn copy_mixed(
        _: &mut Self,
        cx: &mut Context<'_>,
        req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        let pointer = req.in_buffer(cx, 0)?.read_all(UPPER_LIMIT)?;
        let map_alias = req.in_buffer(cx, 1)?.read_all(UPPER_LIMIT)?;
        req.out_buffer(cx, 0)?.write(&map_alias)?;
        req.out_buffer(cx, 1)?.write(&pointer)?;
        reply.ok()
    }

    /// Declares one in buffer and asks for a second.
    fn peek_undeclared(
        _: &mut Self,
        cx: &mut Context<'_>,
        req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        req.in_buffer(cx, 1)?;
        reply.ok()
    }

    /// out: out buffer 0 (map-alias) filled with [`FILL_BYTE`], up to [`UPPER_LIMIT`] bytes;
    /// `u32` how many were written.
    fn fill(_: &mut Self, cx: &mut Context<'_>, req: &Request<'_>, reply: &mut Reply) -> Outcome {
        let mut out = req.out_buffer(cx, 0)?;
        let room = usize::try_from(out.len()).unwrap_or(usize::MAX);
        out.write(&vec![FILL_BYTE; room.min(UPPER_LIMIT)])?;
        let written = out.written();
        reply
            .payload()
            .write_u32(u32::try_from(written).unwrap_or(u32::MAX));
        reply.ok()
    }

    /// TIPC 0. in: `u32 a`, `u32 b`. out: `u32 a + b`.
    fn sum(_: &mut Self, _cx: &mut Context<'_>, req: &Request<'_>, reply: &mut Reply) -> Outcome {
        let mut r = req.payload();
        let (a, b) = (r.read_u32()?, r.read_u32()?);
        reply.payload().write_u32(a.wrapping_add(b));
        reply.ok()
    }
}

/// The most text `upper` accepts.
const UPPER_LIMIT: usize = 16;
/// What `fail` writes into the payload it then throws away.
const FAIL_PAYLOAD: u64 = 0x7777_6666_5555_4444;
/// The tag of the child `interleave` returns.
const INTERLEAVED_TAG: u32 = 0x1E;
/// The one byte of payload `odd` returns.
const ODD_BYTE: u8 = 0x0D;
/// What `inspect` reports for something the request did not carry.
const ABSENT: u32 = u32::MAX;
/// What `fill` writes.
const FILL_BYTE: u8 = 0xF1;
/// What `hasty` answers.
const HASTY: u32 = 0x4A57;
/// The number the toy server is known by.
const SERVICE: ServiceId = ServiceId(0x51);

/// Toy command ids, named after their methods.
mod toy {
    pub(super) const ADD: u32 = 0;
    pub(super) const OPEN_CHILD: u32 = 1;
    pub(super) const CHILD_TAG: u32 = 2;
    pub(super) const INSPECT: u32 = 3;
    pub(super) const UPPER: u32 = 4;
    pub(super) const MAKE_EVENT: u32 = 5;
    pub(super) const FAIL: u32 = 6;
    pub(super) const FLOOD: u32 = 7;
    pub(super) const SPAWN: u32 = 8;
    pub(super) const INTERLEAVE: u32 = 9;
    pub(super) const POKE: u32 = 10;
    pub(super) const SIGNAL: u32 = 11;
    pub(super) const OPEN_MUTE: u32 = 12;
    pub(super) const ODD: u32 = 13;
    pub(super) const WAIT: u32 = 14;
    pub(super) const WAIT_TWICE: u32 = 15;
    pub(super) const FORGET: u32 = 16;
    pub(super) const HASTY: u32 = 17;
    pub(super) const BUMP: u32 = 18;
    pub(super) const LEND: u32 = 19;
    pub(super) const COPY_MIXED: u32 = 20;
    pub(super) const PEEK_UNDECLARED: u32 = 21;
    pub(super) const FILL: u32 = 22;
    /// An id the toy does not list in either block.
    pub(super) const MISSING: u16 = 77;
    pub(super) const TIPC_SUM: u16 = 0;
    pub(super) const TIPC_OPEN_CHILD: u16 = 1;
    pub(super) const TIPC_INSPECT: u16 = 2;
    pub(super) const TIPC_WAIT: u16 = 3;
    pub(super) const TIPC_UPPER: u16 = 4;
}

/// The interface the toy hands out.
#[derive(Serialize, Deserialize)]
struct Child {
    tag: u32,
}

commands! {
    impl Child {
        0 => tag,
        1 => set_tag,
    }
}

impl Child {
    /// out: `u32 tag`.
    fn tag(&mut self, _cx: &mut Context<'_>, _req: &Request<'_>, reply: &mut Reply) -> Outcome {
        reply.payload().write_u32(self.tag);
        reply.ok()
    }

    /// in: `u32 tag`.
    fn set_tag(&mut self, _cx: &mut Context<'_>, req: &Request<'_>, reply: &mut Reply) -> Outcome {
        self.tag = req.payload().read_u32()?;
        reply.ok()
    }
}

/// Child command ids.
mod child {
    pub(super) const TAG: u32 = 0;
    pub(super) const SET_TAG: u32 = 1;
}

/// An interface with no commands at all.
#[derive(Serialize, Deserialize)]
struct Mute;

commands! {
    impl Mute {}
}

/// Counts how often a server said a reply token completed.
#[derive(Default)]
struct Wakes(AtomicUsize);

impl Wakes {
    fn count(&self) -> usize {
        self.0.load(Ordering::SeqCst)
    }
}

impl Notify for Wakes {
    fn completion_posted(&self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

/// The interfaces a toy server's sessions can be.
fn toy_interfaces() -> Interfaces {
    Interfaces::new()
        .with::<Toy>()
        .with::<Child>()
        .with::<Mute>()
}

/// A toy server with no sessions, told to wake `wakes`.
fn toy_server(policy: UnknownPolicy, wakes: &Arc<Wakes>) -> Server {
    let notify: Arc<dyn Notify> = Arc::<Wakes>::clone(wakes);
    Server::new("toy", policy)
        .with_service_id(SERVICE)
        .with_interfaces(toy_interfaces())
        .with_family(Tally::default())
        .with_root(|| Box::new(Toy::default()))
        .with_notify(notify)
}

/// A toy server, a fake kernel and one open session.
struct Rig {
    kernel: FakeKernel,
    server: Server,
    session: ObjectId,
    /// Told when a reply token of the server completes.
    wakes: Arc<Wakes>,
}

impl Rig {
    fn new(policy: UnknownPolicy) -> Self {
        Self::with_handle_table(policy, 0)
    }

    fn strict() -> Self {
        Self::new(UnknownPolicy::Strict)
    }

    /// A rig whose caller can hold `size` handles.
    fn with_handle_table(policy: UnknownPolicy, size: i32) -> Self {
        let mut kernel = FakeKernel::new(size).unwrap();
        let session = kernel.new_object();
        let wakes = Arc::new(Wakes::default());
        let mut server = toy_server(policy, &wakes);
        server.open(session, Box::new(Toy::default())).unwrap();
        Self {
            kernel,
            server,
            session,
            wakes,
        }
    }

    /// Opens a second session on the server, with a toy of its own.
    fn second_session(&mut self) -> ObjectId {
        let session = self.kernel.new_object();
        self.server.accept(session).unwrap();
        session
    }

    /// Sends a request that must be deferred, and returns its token id. The buffer must come
    /// back untouched.
    fn defer_on(&mut self, session: ObjectId, request: &RequestBuilder) -> super::TokenId {
        let (handled, buffer) = self.try_send_on(session, request);
        assert_eq!(buffer, built(request));
        match handled {
            Ok(Handled::Deferred(token)) => token,
            other => unreachable!("not deferred: {other:?}"),
        }
    }

    /// Delivers the oldest completed deferred request into a buffer that held [`STALE`] bytes,
    /// and returns the session it was sent on and the buffer.
    fn deliver(&mut self) -> (ObjectId, Vec<u8>) {
        let completed = self.server.take_completed().unwrap();
        let session = completed.session();
        let mut buffer = vec![STALE; BUFFER_SIZE];
        let delivered = self
            .server
            .deliver(&mut self.kernel, completed, &mut buffer);
        assert_eq!(delivered, Ok(()));
        (session, buffer)
    }

    /// Sends a request on the rig's session and returns the buffer, which must hold a reply.
    fn send(&mut self, request: &RequestBuilder) -> Vec<u8> {
        self.send_on(self.session, request)
    }

    fn send_on(&mut self, session: ObjectId, request: &RequestBuilder) -> Vec<u8> {
        let (handled, buffer) = self.try_send_on(session, request);
        assert_eq!(handled, Ok(Handled::Replied));
        buffer
    }

    fn try_send_on(
        &mut self,
        session: ObjectId,
        request: &RequestBuilder,
    ) -> (Result<Handled, ServerError>, Vec<u8>) {
        let mut buffer = built(request);
        let handled = self.server.handle(&mut self.kernel, session, &mut buffer);
        (handled, buffer)
    }

    /// Converts the rig's session to a domain and checks the reply.
    fn convert(&mut self) {
        let convert = control_request(CONVERT_CURRENT_OBJECT_TO_DOMAIN);
        let reply = plain(&self.send(&convert));
        assert_eq!(reply.result, ResultCode::SUCCESS.raw());
    }

    /// A rig whose session is already a domain.
    fn domain() -> Self {
        let mut rig = Self::strict();
        rig.convert();
        rig
    }

    /// The kernel object behind a handle the caller was given.
    fn object_of(&self, handle: u32) -> ObjectId {
        self.kernel
            .resolve_handle(Handle::from_raw(handle))
            .unwrap()
    }

    /// Opens a child on the rig's ordinary session and returns the child's session.
    fn open_child(&mut self, tag: u32) -> ObjectId {
        let reply = plain(&self.send(&request(toy::OPEN_CHILD).data(&tag.to_le_bytes())));
        assert_eq!(reply.result, ResultCode::SUCCESS.raw());
        assert_eq!(reply.moves.len(), 1);
        self.object_of(reply.moves[0])
    }

    /// The tag of the child behind an ordinary session.
    fn child_tag(&mut self, child: ObjectId) -> u32 {
        let reply = plain(&self.send_on(child, &request(child::TAG)));
        assert_eq!(reply.result, ResultCode::SUCCESS.raw());
        u32::from_le_bytes(reply.data.try_into().unwrap())
    }
}

/// A CMIF request for a command.
fn request(command_id: u32) -> RequestBuilder {
    RequestBuilder::cmif(CommandType::Request, command_id)
}

/// A CMIF control request.
fn control_request(command_id: u32) -> RequestBuilder {
    RequestBuilder::cmif(CommandType::Control, command_id)
}

/// A request to an object of a domain.
fn domain_request(object_id: u32, command_id: u32) -> RequestBuilder {
    RequestBuilder::domain(CommandType::Request, object_id, command_id)
}

/// A payload written with the framework's own writer.
fn payload(write: impl FnOnce(&mut Writer)) -> Vec<u8> {
    let mut writer = Writer::new();
    write(&mut writer);
    writer.as_bytes().to_vec()
}

/// A whole message buffer that starts with `words` and is zero after them.
fn buffer_of(words: &[u32]) -> Vec<u8> {
    let mut buffer = le(words);
    assert!(buffer.len() <= BUFFER_SIZE);
    buffer.resize(BUFFER_SIZE, 0);
    buffer
}

/// A request as the bytes of a message buffer that held [`STALE`] bytes before.
fn built(request: &RequestBuilder) -> Vec<u8> {
    let mut buffer = vec![STALE; BUFFER_SIZE];
    request.build(&mut buffer).unwrap();
    buffer
}

/// Little-endian words as bytes.
fn le(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

/// A parsed reply.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Answer {
    result: u32,
    data: Vec<u8>,
    copies: Vec<u32>,
    moves: Vec<u32>,
    objects: Vec<u32>,
}

fn answer(buffer: &[u8], kind: SessionKind) -> Answer {
    let message = Message::parse_reply(buffer, kind).unwrap();
    let (result, data, objects) = match &message.body {
        Body::CmifReply(reply) => (
            reply.header.result,
            reply.data.to_vec(),
            reply.object_ids.clone(),
        ),
        Body::TipcReply { result, data } => (*result, data.to_vec(), Vec::new()),
        other => unreachable!("not a reply: {other:?}"),
    };
    Answer {
        result,
        data,
        copies: message.copy_handles.clone(),
        moves: message.move_handles.clone(),
        objects,
    }
}

/// A reply without a domain header: ordinary sessions, control requests, and TIPC.
fn plain(buffer: &[u8]) -> Answer {
    answer(buffer, SessionKind::Session)
}

/// A reply with a domain header.
fn in_domain(buffer: &[u8]) -> Answer {
    answer(buffer, SessionKind::Domain)
}

/// A reply that holds a result and nothing else.
fn only(result: ResultCode) -> Answer {
    Answer {
        result: result.raw(),
        data: Vec::new(),
        copies: Vec::new(),
        moves: Vec::new(),
        objects: Vec::new(),
    }
}
