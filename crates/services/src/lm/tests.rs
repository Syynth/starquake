//! `lm` driven through raw messages over a fake kernel: `OpenLogger` as the console's
//! interface declares it (a client process id), and `Log` with its auto-select buffer sent
//! either way a client may send one.

use std::sync::Mutex;

use starquake_ipc::{
    BUFFER_SIZE, Body, BufferDescriptor, CommandType, Message, RequestBuilder, RequestLayout,
    SessionKind, StaticDescriptor,
};
use starquake_kernel::handle::Handle;
use starquake_kernel::id::ObjectId;
use starquake_kernel::result::ResultCode;

use super::command::lm::OPEN_LOGGER;
use super::command::logger::{LOG, SET_DESTINATION};
use super::packet::{
    ChunkKey, FLAG_HEAD, FLAG_TAIL, LEB128_CONTINUE, LEB128_SHIFT, LEB128_VALUE_MASK,
};
use super::*;
use crate::framework::control::CONVERT_CURRENT_OBJECT_TO_DOMAIN;
use crate::framework::fake::FakeKernel;
use crate::framework::result::{INVALID_CMIF_REQUEST, UNKNOWN_COMMAND_ID};
use crate::framework::{Handled, Kernel, Protocol, Session};

/// `ILogger::TransmitHashedLog`, which Eden lists without a handler, so it has no shipping
/// constant (`eden/src/core/hle/service/lm/lm.cpp`: `{2, nullptr, "TransmitHashedLog"}`).
const TRANSMIT_HASHED_LOG: u32 = 2;

/// The caller's process id.
const PROCESS_ID: u64 = 0x0123_4567_89AB_CDEF;
/// Where tests map the packet `Log` reads.
const PACKET_AT: u64 = 0x0010_0000;
/// What fills the message buffer beforehand.
const STALE: u8 = 0xA5;

/// A guest log that keeps every line.
#[derive(Default)]
struct Lines(Mutex<Vec<Vec<u8>>>);

impl GuestLog for Lines {
    fn line(&self, text: &[u8]) {
        self.0.lock().unwrap().push(text.to_vec());
    }
}

/// An `lm` server, a kernel, one session, and the guest log the server writes to.
struct Rig {
    kernel: FakeKernel,
    server: Server,
    session: ObjectId,
    lines: Arc<Lines>,
}

impl Rig {
    fn new() -> Self {
        Self::with_policy(UnknownPolicy::Strict)
    }

    fn with_policy(policy: UnknownPolicy) -> Self {
        let mut kernel = FakeKernel::new(0).unwrap();
        kernel.set_process_id(PROCESS_ID);
        let session = kernel.new_object();
        let lines = Arc::new(Lines::default());
        let log: Arc<dyn GuestLog> = Arc::<Lines>::clone(&lines);
        let mut server = server(policy, log);
        server.accept(session).unwrap();
        Self {
            kernel,
            server,
            session,
            lines,
        }
    }

    fn send_raw(&mut self, session: ObjectId, request: &RequestBuilder) -> Vec<u8> {
        let mut buffer = vec![STALE; BUFFER_SIZE];
        request.build(&mut buffer).unwrap();
        let handled = self.server.handle(&mut self.kernel, session, &mut buffer);
        assert_eq!(handled, Ok(Handled::Replied));
        buffer
    }

    /// Sends `request` to `session`: the reply's result and moved handles.
    fn send(&mut self, session: ObjectId, request: &RequestBuilder) -> (ResultCode, Vec<u32>) {
        let reply = self.send_raw(session, request);
        let message = Message::parse_reply(&reply, SessionKind::Session).unwrap();
        let Body::CmifReply(body) = &message.body else {
            unreachable!("not a CMIF reply: {:?}", message.body);
        };
        (
            ResultCode::from_raw(body.header.result),
            message.move_handles.clone(),
        )
    }

    /// Sends `OpenLogger` and returns the logger's session.
    fn logger(&mut self) -> ObjectId {
        let (result, moves) = self.send(self.session, &open_logger());
        assert_eq!(result, ResultCode::SUCCESS);
        self.kernel
            .resolve_handle(Handle::from_raw(moves[0]))
            .unwrap()
    }

    /// Maps `packet` and sends it to `logger` through an A descriptor; the result.
    fn log(&mut self, logger: ObjectId, packet: &[u8]) -> ResultCode {
        self.kernel.map(PACKET_AT, packet.to_vec());
        self.send(logger, &log_mapped(packet.len())).0
    }

    fn lines(&self) -> Vec<Vec<u8>> {
        self.lines.0.lock().unwrap().clone()
    }
}

fn request(command: u32) -> RequestBuilder {
    RequestBuilder::cmif(CommandType::Request, command).layout(RequestLayout::Libnx)
}

/// `OpenLogger` with a client process id: the process id sent, and a `u64` in the payload.
fn open_logger() -> RequestBuilder {
    request(OPEN_LOGGER)
        .send_pid(PROCESS_ID)
        .data(&0u64.to_le_bytes())
}

fn a(address: u64, size: usize) -> BufferDescriptor {
    BufferDescriptor {
        address,
        size: u64::try_from(size).unwrap(),
        flags: 0,
    }
}

fn x(address: u64, size: usize) -> StaticDescriptor {
    StaticDescriptor {
        counter: 0,
        address,
        size: u16::try_from(size).unwrap(),
    }
}

/// `Log` with the packet at [`PACKET_AT`] in the A descriptor and an empty X beside it.
fn log_mapped(len: usize) -> RequestBuilder {
    request(LOG)
        .send_buffer(a(PACKET_AT, len))
        .static_descriptor(x(0, 0))
}

/// `Log` with the packet at [`PACKET_AT`] in the X descriptor and an empty A beside it.
fn log_pointer(len: usize) -> RequestBuilder {
    request(LOG)
        .send_buffer(a(0, 0))
        .static_descriptor(x(PACKET_AT, len))
}

fn leb128(mut value: u64) -> Vec<u8> {
    let mut bytes = Vec::new();
    loop {
        let byte = u8::try_from(value & u64::from(LEB128_VALUE_MASK)).unwrap();
        value >>= LEB128_SHIFT;
        if value == 0 {
            bytes.push(byte);
            return bytes;
        }
        bytes.push(byte | LEB128_CONTINUE);
    }
}

fn chunk(key: ChunkKey, data: &[u8]) -> Vec<u8> {
    let mut bytes = leb128(key.raw());
    bytes.extend(leb128(u64::try_from(data.len()).unwrap()));
    bytes.extend(data);
    bytes
}

/// A packet with `flags`, of thread `thread`, carrying `payload`.
fn packet(flags: u8, thread: u64, payload: &[u8]) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend(PROCESS_ID.to_le_bytes());
    bytes.extend(thread.to_le_bytes());
    bytes.extend([flags, 0, 1, 0]);
    bytes.extend(u32::try_from(payload.len()).unwrap().to_le_bytes());
    bytes.extend(payload);
    bytes
}

/// A whole message in one packet: a session-begin chunk, `text`, a line number and a file
/// name, as a client writes one.
fn whole(text: &[u8]) -> Vec<u8> {
    let payload = [
        chunk(ChunkKey::SessionBegin, &[1]),
        chunk(ChunkKey::TextLog, text),
        chunk(ChunkKey::LineNumber, &7u32.to_le_bytes()),
        chunk(ChunkKey::FileName, b"main.cpp"),
    ]
    .concat();
    packet(FLAG_HEAD | FLAG_TAIL, 1, &payload)
}

#[test]
fn open_logger_hands_out_a_logger_that_logs() {
    let mut rig = Rig::new();
    let logger = rig.logger();
    assert_eq!(
        (rig.server.session_count(), rig.server.interface_count()),
        (2, 2)
    );
    assert_eq!(rig.log(logger, &whole(b"hello")), ResultCode::SUCCESS);
    assert_eq!(rig.lines(), [b"hello".to_vec()]);
    // The root is not a logger: its command 0 is `OpenLogger`, which takes no buffer, so a
    // `Log` sent to it is refused before any command runs, and nothing is logged.
    rig.kernel.map(PACKET_AT, whole(b"x"));
    assert_eq!(
        rig.send(rig.session, &log_mapped(whole(b"x").len())).0,
        INVALID_CMIF_REQUEST
    );
    assert_eq!(rig.lines().len(), 1);
}

#[test]
fn open_logger_in_a_domain_returns_a_logger_object() {
    let mut rig = Rig::new();
    let convert = RequestBuilder::cmif(CommandType::Control, CONVERT_CURRENT_OBJECT_TO_DOMAIN);
    let converted = rig.send_raw(rig.session, &convert);
    let root = u32::from_le_bytes(converted[32..36].try_into().unwrap());
    let open = RequestBuilder::domain(CommandType::Request, root, OPEN_LOGGER)
        .send_pid(PROCESS_ID)
        .data(&0u64.to_le_bytes());
    let reply = rig.send_raw(rig.session, &open);
    let message = Message::parse_reply(&reply, SessionKind::Domain).unwrap();
    let Body::CmifReply(body) = &message.body else {
        unreachable!("not a CMIF reply");
    };
    assert_eq!(body.header.result, ResultCode::SUCCESS.raw());
    let object = body.object_ids[0];
    let packet = whole(b"in a domain");
    rig.kernel.map(PACKET_AT, packet.clone());
    let log = RequestBuilder::domain(CommandType::Request, object, LOG)
        .send_buffer(a(PACKET_AT, packet.len()))
        .static_descriptor(x(0, 0));
    let reply = rig.send_raw(rig.session, &log);
    let message = Message::parse_reply(&reply, SessionKind::Domain).unwrap();
    let Body::CmifReply(body) = &message.body else {
        unreachable!("not a CMIF reply");
    };
    assert_eq!(body.header.result, ResultCode::SUCCESS.raw());
    assert_eq!(rig.lines(), [b"in a domain".to_vec()]);
}

#[test]
fn a_packet_sent_through_the_pointer_descriptor_is_read_too() {
    let mut rig = Rig::new();
    let logger = rig.logger();
    let packet = whole(b"by pointer");
    rig.kernel.map(PACKET_AT, packet.clone());
    assert_eq!(
        rig.send(logger, &log_pointer(packet.len())).0,
        ResultCode::SUCCESS
    );
    assert_eq!(rig.lines(), [b"by pointer".to_vec()]);
}

#[test]
fn a_message_in_several_packets_is_one_line_at_its_tail() {
    let mut rig = Rig::new();
    let logger = rig.logger();
    let text = chunk(ChunkKey::TextLog, b"split across three");
    let (first, rest) = text.split_at(3);
    let (second, third) = rest.split_at(5);
    assert_eq!(
        rig.log(logger, &packet(FLAG_HEAD, 1, first)),
        ResultCode::SUCCESS
    );
    assert_eq!(rig.log(logger, &packet(0, 1, second)), ResultCode::SUCCESS);
    assert_eq!(rig.lines(), Vec::<Vec<u8>>::new());
    assert_eq!(
        rig.log(logger, &packet(FLAG_TAIL, 1, third)),
        ResultCode::SUCCESS
    );
    assert_eq!(rig.lines(), [b"split across three".to_vec()]);
}

#[test]
fn messages_in_progress_belong_to_their_logger() {
    // Eden's `entries` are per `ILogger`: a tail sent to another logger finds nothing.
    let mut rig = Rig::new();
    let first = rig.logger();
    let second = rig.logger();
    let text = chunk(ChunkKey::TextLog, b"mine");
    rig.log(first, &packet(FLAG_HEAD, 1, &text[..2]));
    rig.log(second, &packet(FLAG_TAIL, 1, &text[2..]));
    assert_eq!(rig.lines(), Vec::<Vec<u8>>::new());
    rig.log(first, &packet(FLAG_TAIL, 1, &text[2..]));
    assert_eq!(rig.lines(), [b"mine".to_vec()]);
}

#[test]
fn a_message_without_text_adds_no_line() {
    let mut rig = Rig::new();
    let logger = rig.logger();
    let payload = chunk(ChunkKey::ModuleName, b"module");
    assert_eq!(
        rig.log(logger, &packet(FLAG_HEAD | FLAG_TAIL, 1, &payload)),
        ResultCode::SUCCESS
    );
    assert_eq!(rig.lines(), Vec::<Vec<u8>>::new());
}

#[test]
fn a_malformed_packet_is_answered_with_success_and_adds_no_line() {
    let mut rig = Rig::new();
    let logger = rig.logger();
    // A header cut short, a continuation of nothing, and a chunk past the end of its message:
    // each answered with success, as Eden answers every `Log`, and none logged.
    let short = packet(FLAG_HEAD | FLAG_TAIL, 1, b"")[..10].to_vec();
    let orphan = packet(FLAG_TAIL, 1, &chunk(ChunkKey::TextLog, b"orphan"));
    let past_end = packet(FLAG_HEAD | FLAG_TAIL, 1, &[0x02, 0x09, b'a']);
    for bad in [short, orphan, past_end] {
        assert_eq!(rig.log(logger, &bad), ResultCode::SUCCESS);
    }
    assert_eq!(rig.lines(), Vec::<Vec<u8>>::new());
    // The logger still works.
    assert_eq!(rig.log(logger, &whole(b"after")), ResultCode::SUCCESS);
    assert_eq!(rig.lines(), [b"after".to_vec()]);
}

#[test]
fn a_log_request_without_both_descriptors_is_refused() {
    // Eden answers each of these with success (`ctx.ReadBuffer()`, no descriptor count); here
    // the framework refuses them before `Log` runs (module docs, "Where this differs from
    // Eden").
    let mut rig = Rig::new();
    let logger = rig.logger();
    let packet = whole(b"refused");
    rig.kernel.map(PACKET_AT, packet.clone());
    let len = packet.len();
    let no_descriptors = request(LOG);
    let only_a = request(LOG).send_buffer(a(PACKET_AT, len));
    let only_x = request(LOG).static_descriptor(x(PACKET_AT, len));
    for refused in [no_descriptors, only_a, only_x] {
        assert_eq!(rig.send(logger, &refused).0, INVALID_CMIF_REQUEST);
    }
    assert_eq!(rig.lines(), Vec::<Vec<u8>>::new());
    // The same packet with both descriptors is logged.
    assert_eq!(rig.send(logger, &log_mapped(len)).0, ResultCode::SUCCESS);
    assert_eq!(rig.lines(), [b"refused".to_vec()]);
}

#[test]
fn a_packet_larger_than_the_limit_is_dropped() {
    // A packet of exactly the limit is read: a whole message of the largest size, its text
    // filling it.
    let mut rig = Rig::new();
    let logger = rig.logger();
    let room = MAX_PACKET_BYTES - HEADER_SIZE - leb128(ChunkKey::TextLog.raw()).len();
    // The text's size takes as many bytes as its own LEB128 form: the first width that fits.
    let text_len = (1..=packet::MAX_LEB128_BYTES)
        .map(|width| room - width)
        .find(|len| leb128(u64::try_from(*len).unwrap()).len() == room - len)
        .unwrap();
    let text = vec![b't'; text_len];
    let largest = packet(FLAG_HEAD | FLAG_TAIL, 1, &chunk(ChunkKey::TextLog, &text));
    assert_eq!(largest.len(), MAX_PACKET_BYTES);
    assert_eq!(rig.log(logger, &largest), ResultCode::SUCCESS);
    assert_eq!(rig.lines(), [text]);
    // One byte more is not read at all, though its message would parse.
    let mut over = largest;
    over.push(0);
    assert_eq!(rig.log(logger, &over), ResultCode::SUCCESS);
    assert_eq!(rig.lines().len(), 1);
}

#[test]
fn a_new_logger_sends_everywhere_and_set_destination_is_kept() {
    // `LogDestination destination{LogDestination::All}`.
    assert_eq!(ILogger::new().destination(), DESTINATION_ALL);
    assert!(ILogger::new().pending().is_empty());
    assert_eq!(ILogger::default(), ILogger::new());
    let mut rig = Rig::new();
    let logger = rig.logger();
    let set = request(SET_DESTINATION).data(&0x0000_0102u32.to_le_bytes());
    assert_eq!(rig.send(logger, &set).0, ResultCode::SUCCESS);
    // Lines still go to the guest log whatever the destination.
    assert_eq!(rig.log(logger, &whole(b"still")), ResultCode::SUCCESS);
    assert_eq!(rig.lines(), [b"still".to_vec()]);
    // The destination is the logger's state: saved with it.
    let saved = rig.server.save().unwrap();
    let mut fresh = ILogger::new();
    fresh.destination = 0x0000_0102;
    let state = crate::framework::snapshot::encode(&fresh).unwrap();
    let encoded_default = crate::framework::snapshot::encode(&ILogger::new()).unwrap();
    assert!(saved.windows(state.len()).any(|w| w == state));
    assert!(
        !saved
            .windows(encoded_default.len())
            .any(|w| w == encoded_default)
    );
}

#[test]
fn a_message_in_progress_survives_a_save_and_restore() {
    let mut rig = Rig::new();
    let logger = rig.logger();
    let text = chunk(ChunkKey::TextLog, b"restored");
    rig.log(logger, &packet(FLAG_HEAD, 1, &text[..4]));
    let saved = rig.server.save().unwrap();

    let lines = Arc::new(Lines::default());
    let log: Arc<dyn GuestLog> = Arc::<Lines>::clone(&lines);
    let mut restored = server(UnknownPolicy::Strict, log);
    restored.restore(&saved).unwrap();
    assert_eq!(restored.save().unwrap(), saved);
    // The restored logger finishes the message, into the restored server's own guest log.
    rig.server = restored;
    rig.lines = lines;
    assert_eq!(
        rig.log(logger, &packet(FLAG_TAIL, 1, &text[4..])),
        ResultCode::SUCCESS
    );
    assert_eq!(rig.lines(), [b"restored".to_vec()]);
}

#[test]
fn the_family_saves_nothing_and_loads_only_nothing() {
    let lines: Arc<dyn GuestLog> = Arc::new(Lines::default());
    let mut loggers = Loggers { guest_log: lines };
    assert_eq!(loggers.save_state(), Ok(Vec::new()));
    assert_eq!(loggers.load_state(&[]), Ok(()));
    assert_eq!(
        loggers.load_state(&[1, 2]),
        Err(SnapshotError::TrailingBytes { count: 2 })
    );
}

#[test]
fn commands_that_are_left_out_go_to_the_unknown_command_policy() {
    let mut rig = Rig::new();
    let logger = rig.logger();
    assert_eq!(
        rig.send(logger, &request(TRANSMIT_HASHED_LOG)).0,
        UNKNOWN_COMMAND_ID
    );
    let mut stubbing = Rig::with_policy(UnknownPolicy::AutoStub);
    let logger = stubbing.logger();
    assert_eq!(
        stubbing.send(logger, &request(TRANSMIT_HASHED_LOG)).0,
        ResultCode::SUCCESS
    );
    let stubbed: Vec<_> = stubbing.server.stubbed().copied().collect();
    assert_eq!(
        stubbed,
        [crate::framework::StubbedCommand {
            interface: starquake_trace::InterfaceName::new("ILogger").unwrap(),
            protocol: starquake_trace::IpcProtocol::Cmif,
            id: TRANSMIT_HASHED_LOG,
        }]
    );
}

#[test]
fn the_tables_name_the_commands() {
    assert_eq!(
        Lm.commands().name(Protocol::Cmif, OPEN_LOGGER),
        Some("open_logger")
    );
    let logger = ILogger::new();
    assert_eq!(logger.commands().name(Protocol::Cmif, LOG), Some("log"));
    assert_eq!(
        logger.commands().name(Protocol::Cmif, SET_DESTINATION),
        Some("set_destination")
    );
    assert_eq!(Lm::interface(), "Lm");
    assert_eq!(ILogger::interface(), "ILogger");
    assert_eq!(NAME, "lm");
}
