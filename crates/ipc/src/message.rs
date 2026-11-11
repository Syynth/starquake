//! Parsing a whole IPC buffer into a [`Message`].
//!
//! Layout, in order (`eden/src/core/hle/service/hle_ipc.cpp`, `ParseCommandBuffer`):
//!
//! 1. the two header words ([`CommandHeader`]); a close command stops here;
//! 2. if flagged, the special header, then the process id (two words, if flagged), then the copy
//!    handles, then the move handles;
//! 3. the X descriptors (2 words each), then A, B and W descriptors (3 words each);
//! 4. the raw data section, `raw_data_words` words long;
//! 5. the C (receive list) descriptors, immediately after the raw data section.
//!
//! Inside the raw data section a CMIF message is padded to a 16-byte boundary (measured from the
//! start of the buffer), then carries an optional domain header, the `SFCI`/`SFCO` header, the
//! data, and in a domain the trailing object ids. A TIPC message has none of that: its raw data
//! is the payload itself, and a reply's payload starts with one result word.
//!
//! The parser is zero-copy for raw data and never panics: every offset is checked.
//!
//! # Guess: where a domain reply keeps its object ids
//!
//! This is a guess, not a protocol fact. The parser takes a domain reply's object ids from the
//! **end** of the raw data section and treats everything between the `SFCO` header and them as
//! data. That only holds when the raw size is exact, as [`crate::ReplyBuilder`] writes it. Eden's
//! `ResponseBuilder` reserves four extra padding words (`ipc_helpers.h`), and libnx's
//! `cmifParseResponse` (`libnx/nx/include/switch/sf/cmif.h`) instead finds the objects at the
//! `SFCO` header plus the data size it already knows. On a reply that keeps such a reserve this parser would return the padding as
//! object ids and truncate the data. Parsing replies from a real guest needs the expected data
//! size passed in, as libnx does.

use crate::cmif::{
    CMIF_HEADER_BYTES, CmifReplyHeader, CmifRequestHeader, DomainCommand, DomainReplyHeader,
    DomainRequestHeader,
};
use crate::error::{IpcError, Result};
use crate::hipc::{
    BUFFER_SIZE, BufferDescriptor, CommandHeader, CommandType, ReceiveDescriptor, ReceiveList,
    SpecialHeader, StaticDescriptor,
};

/// Whether the session the message travels on has been converted to a domain.
///
/// Eden decides this from session state, not from the bytes, so the caller supplies it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SessionKind {
    /// An ordinary session: no domain headers.
    Session,
    /// A domain: requests of type request/request-with-context carry a domain header, and so do
    /// their replies. Eden writes the reply's domain header only when the request had one
    /// (`ipc_helpers.h`: `mgr->IsDomain() && ctx.HasDomainMessageHeader()`), so a reply to a
    /// control or control-with-context request on a domain has none: parse such a reply with
    /// [`SessionKind::Session`].
    Domain,
}

/// A CMIF request payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CmifRequest<'a> {
    /// The domain header, present on domain requests of type request/request-with-context.
    pub domain: Option<DomainRequestHeader>,
    /// The `SFCI` header.
    pub header: CmifRequestHeader,
    /// The argument bytes. In a domain this is exactly `domain.size` minus the `SFCI` header; in
    /// a plain session it runs to the end of the raw data section and so may include trailing
    /// alignment padding.
    pub data: &'a [u8],
    /// The input object ids that trail the arguments in a domain request.
    pub object_ids: Vec<u32>,
}

/// A CMIF reply payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CmifReply<'a> {
    /// The domain header, present on replies to domain request/request-with-context messages
    /// (parsed with [`SessionKind::Domain`]).
    pub domain: Option<DomainReplyHeader>,
    /// The `SFCO` header.
    pub header: CmifReplyHeader,
    /// The data bytes between the `SFCO` header and the object ids.
    ///
    /// Guess: in a domain reply this assumes the raw size is exact (see the module docs). If the
    /// sender kept padding after the object ids, it is wrongly included here.
    pub data: &'a [u8],
    /// The object ids that trail the data in a domain reply.
    ///
    /// Guess: they are taken from the end of the raw data section, which assumes the raw size is
    /// exact (see the module docs). Padding after them would be returned as ids.
    pub object_ids: Vec<u32>,
}

/// What a message carries after its descriptors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Body<'a> {
    /// A close command: nothing else in the buffer is populated.
    Close,
    /// A CMIF request.
    CmifRequest(CmifRequest<'a>),
    /// A domain request to close a virtual handle: no `SFCI` header and no data.
    DomainClose(DomainRequestHeader),
    /// A CMIF reply.
    CmifReply(CmifReply<'a>),
    /// A TIPC request; the command id is in [`Message::command_type`]. All of the raw data is
    /// the arguments.
    TipcRequest {
        /// The raw data section.
        data: &'a [u8],
    },
    /// A TIPC reply: a single result word, then the data directly. This is the layout Eden got
    /// wrong (it wrote a two-word result), see scoping doc row 22.
    TipcReply {
        /// The result word.
        result: u32,
        /// The raw data after the result word.
        data: &'a [u8],
    },
}

/// A parsed IPC buffer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Message<'a> {
    /// The message kind from the command header.
    pub command_type: CommandType,
    /// The process id the sender asked to attach (the slot holds zero until the kernel fills it).
    pub pid: Option<u64>,
    /// Handles copied to the receiver.
    pub copy_handles: Vec<u32>,
    /// Handles moved to the receiver.
    pub move_handles: Vec<u32>,
    /// X (static send) descriptors.
    pub static_descriptors: Vec<StaticDescriptor>,
    /// A (send) descriptors.
    pub send_buffers: Vec<BufferDescriptor>,
    /// B (receive) descriptors.
    pub receive_buffers: Vec<BufferDescriptor>,
    /// W (exchange) descriptors.
    pub exchange_buffers: Vec<BufferDescriptor>,
    /// How the receive list is described.
    pub receive_list: ReceiveList,
    /// The C descriptors after the raw data section.
    pub receive_descriptors: Vec<ReceiveDescriptor>,
    /// The payload.
    pub body: Body<'a>,
}

/// A bounds-checked cursor over a byte slice.
struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    fn at(buf: &'a [u8], pos: usize) -> Self {
        Self { buf, pos }
    }

    fn bytes(&mut self, count: usize) -> Result<&'a [u8]> {
        let truncated = IpcError::Truncated {
            offset: self.pos,
            needed: count,
            len: self.buf.len(),
        };
        let end = self
            .pos
            .checked_add(count)
            .ok_or_else(|| truncated.clone())?;
        let slice = self.buf.get(self.pos..end).ok_or(truncated)?;
        self.pos = end;
        Ok(slice)
    }

    fn words<const N: usize>(&mut self) -> Result<[u32; N]> {
        let mut out = [0_u32; N];
        for slot in &mut out {
            *slot = match *self.bytes(4)? {
                [a, b, c, d] => u32::from_le_bytes([a, b, c, d]),
                _ => return Err(IpcError::Inconsistent("short word read")),
            };
        }
        Ok(out)
    }

    fn word(&mut self) -> Result<u32> {
        let [word] = self.words()?;
        Ok(word)
    }

    fn rest(&mut self) -> &'a [u8] {
        let slice = self.buf.get(self.pos..).unwrap_or_default();
        self.pos = self.buf.len();
        slice
    }

    fn list<T, const N: usize>(&mut self, count: u32, decode: fn([u32; N]) -> T) -> Result<Vec<T>> {
        (0..count).map(|_| self.words().map(decode)).collect()
    }
}

fn to_usize(value: u32) -> usize {
    usize::try_from(value).unwrap_or(usize::MAX)
}

impl<'a> Message<'a> {
    /// Parses a request out of a thread's IPC buffer.
    ///
    /// Only the first [`BUFFER_SIZE`] bytes are read. `kind` says whether the session is a
    /// domain; it only matters for CMIF request/request-with-context messages.
    pub fn parse_request(buf: &'a [u8], kind: SessionKind) -> Result<Self> {
        Self::parse(buf, kind, false)
    }

    /// Parses a reply out of a thread's IPC buffer.
    ///
    /// A CMIF reply has command type zero; a TIPC reply echoes the request's TIPC type.
    pub fn parse_reply(buf: &'a [u8], kind: SessionKind) -> Result<Self> {
        Self::parse(buf, kind, true)
    }

    fn parse(buf: &'a [u8], kind: SessionKind, reply: bool) -> Result<Self> {
        let usable = buf.get(..BUFFER_SIZE).unwrap_or(buf);
        let mut r = Reader::new(usable);
        let header = CommandHeader::decode(r.words()?)?;
        let command_type = header.command_type;

        let valid = if reply {
            matches!(command_type, CommandType::Invalid) || command_type.is_tipc()
        } else {
            !matches!(command_type, CommandType::Invalid)
        };
        if !valid {
            return Err(IpcError::UnexpectedCommandType(command_type));
        }

        let mut message = Message {
            command_type,
            pid: None,
            copy_handles: Vec::new(),
            move_handles: Vec::new(),
            static_descriptors: Vec::new(),
            send_buffers: Vec::new(),
            receive_buffers: Vec::new(),
            exchange_buffers: Vec::new(),
            receive_list: ReceiveList::Disabled,
            receive_descriptors: Vec::new(),
            body: Body::Close,
        };
        if command_type.is_close() {
            // Close does not populate the rest of the header.
            return Ok(message);
        }

        if header.has_special_header {
            let special = SpecialHeader::decode(r.word()?);
            if special.send_pid {
                let [low, high] = r.words()?;
                message.pid = Some(u64::from(low) | u64::from(high) << 32);
            }
            message.copy_handles = r.list(special.num_copy_handles, |[h]: [u32; 1]| h)?;
            message.move_handles = r.list(special.num_move_handles, |[h]: [u32; 1]| h)?;
        }
        message.static_descriptors = r.list(header.num_x, StaticDescriptor::decode)?;
        message.send_buffers = r.list(header.num_a, BufferDescriptor::decode)?;
        message.receive_buffers = r.list(header.num_b, BufferDescriptor::decode)?;
        message.exchange_buffers = r.list(header.num_w, BufferDescriptor::decode)?;

        let raw_start = r.pos;
        let raw_end = to_usize(header.raw_data_words)
            .checked_mul(4)
            .and_then(|bytes| raw_start.checked_add(bytes))
            .ok_or(IpcError::Inconsistent("raw data size overflows"))?;
        // Reading the section as a slice checks that it lies inside the buffer.
        let section = Reader::at(usable, raw_start).bytes(raw_end - raw_start)?;

        message.receive_list = header.receive_list;
        message.receive_descriptors = Reader::at(usable, raw_end).list(
            header.receive_list.descriptor_count(),
            ReceiveDescriptor::decode,
        )?;

        message.body = if command_type.is_tipc() {
            Self::tipc_body(section, reply)?
        } else {
            // Padding to a 16-byte boundary, measured from the start of the buffer.
            let padding = raw_start
                .checked_next_multiple_of(16)
                .and_then(|aligned| aligned.checked_sub(raw_start))
                .ok_or(IpcError::Inconsistent("alignment overflows"))?;
            let content = section.get(padding..).ok_or(IpcError::Inconsistent(
                "raw data section ends inside its alignment padding",
            ))?;
            if reply {
                Self::cmif_reply_body(content, kind)?
            } else {
                Self::cmif_request_body(content, kind, command_type)?
            }
        };
        Ok(message)
    }

    fn tipc_body(section: &'a [u8], reply: bool) -> Result<Body<'a>> {
        if reply {
            let mut r = Reader::new(section);
            let result = r.word()?;
            Ok(Body::TipcReply {
                result,
                data: r.rest(),
            })
        } else {
            Ok(Body::TipcRequest { data: section })
        }
    }

    fn cmif_request_body(
        content: &'a [u8],
        kind: SessionKind,
        command_type: CommandType,
    ) -> Result<Body<'a>> {
        let mut r = Reader::new(content);
        // Only request and request-with-context carry a domain header
        // (`hle_ipc.cpp`, `ParseCommandBuffer`).
        let has_domain_header = kind == SessionKind::Domain
            && matches!(
                command_type,
                CommandType::Request | CommandType::RequestWithContext
            );
        let domain = if has_domain_header {
            Some(DomainRequestHeader::decode(r.words()?)?)
        } else {
            None
        };
        if let Some(domain) = domain
            && domain.command == DomainCommand::CloseVirtualHandle
        {
            // CloseVirtualHandle has no SFCI header and no data.
            return Ok(Body::DomainClose(domain));
        }
        let header = CmifRequestHeader::decode(r.words()?)?;
        let rest = r.rest();
        let (data, object_ids) = match domain {
            Some(domain) => {
                let args_len = to_usize(domain.size).checked_sub(CMIF_HEADER_BYTES).ok_or(
                    IpcError::Inconsistent("domain message size is smaller than the SFCI header"),
                )?;
                let (data, tail) =
                    rest.split_at_checked(args_len)
                        .ok_or(IpcError::Inconsistent(
                            "domain message size overruns the raw data",
                        ))?;
                let count = to_usize(domain.input_object_count);
                let ids_len = count
                    .checked_mul(4)
                    .ok_or(IpcError::Inconsistent("domain object count overflows"))?;
                let ids = tail.get(..ids_len).ok_or(IpcError::Inconsistent(
                    "domain object ids overrun the raw data",
                ))?;
                (data, Self::read_ids(ids, count)?)
            }
            None => (rest, Vec::new()),
        };
        Ok(Body::CmifRequest(CmifRequest {
            domain,
            header,
            data,
            object_ids,
        }))
    }

    fn cmif_reply_body(content: &'a [u8], kind: SessionKind) -> Result<Body<'a>> {
        let mut r = Reader::new(content);
        let domain = if kind == SessionKind::Domain {
            Some(DomainReplyHeader::decode(r.words()?))
        } else {
            None
        };
        let header = CmifReplyHeader::decode(r.words()?)?;
        let rest = r.rest();
        let (data, object_ids) = match domain {
            Some(domain) => {
                let count = to_usize(domain.num_objects);
                let ids_len = count
                    .checked_mul(4)
                    .ok_or(IpcError::Inconsistent("domain object count overflows"))?;
                let data_len = rest
                    .len()
                    .checked_sub(ids_len)
                    .ok_or(IpcError::Inconsistent(
                        "domain object ids overrun the raw data",
                    ))?;
                let (data, ids) = rest.split_at(data_len);
                (data, Self::read_ids(ids, count)?)
            }
            None => (rest, Vec::new()),
        };
        Ok(Body::CmifReply(CmifReply {
            domain,
            header,
            data,
            object_ids,
        }))
    }

    fn read_ids(bytes: &[u8], count: usize) -> Result<Vec<u32>> {
        let mut r = Reader::new(bytes);
        (0..count).map(|_| r.word()).collect()
    }
}
