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
//! # Replies: the caller says how much data it expects
//!
//! [`Message::parse_reply_sized`] reads a reply the way libnx's `cmifParseResponse`
//! (`libnx/nx/include/switch/sf/cmif.h:260-288`) and `tipcParseResponse` (`tipc.h:199-228`) do:
//! the caller passes the data size it expects, the data starts right after the `SFCO` header (or
//! the TIPC result word), and a domain reply's object ids start right after that many bytes
//! (`cmif.h:270`), with no padding between. A reply whose result is not zero (`R_FAILED`,
//! `libnx/nx/include/switch/result.h:12`) has no data and no object ids: libnx returns the result
//! before it looks for them (`cmif.h:277-278`, `tipc.h:208-209`).
//!
//! It differs from libnx (`hipcParseResponse`, `libnx/nx/include/switch/sf/hipc.h:316-363`, then
//! the two functions above) in these ways, each pinned by an exact-value or exact-error test in
//! `tests/libnx_golden.rs`:
//!
//! - **Command type.** libnx reads no type word from a reply; its caller picks the CMIF or the
//!   TIPC parser. Here the type word picks: zero is a CMIF reply, a TIPC type (16 and above) is a
//!   TIPC reply, and any other type is [`IpcError::UnexpectedCommandType`]
//!   (`sized_parse_picks_cmif_or_tipc_from_the_type_word`).
//! - **Descriptors.** libnx steps over the copy and move handles and the X descriptors only
//!   (`hipc.h:347-360`) and reads no C descriptor. Here the A, B and W descriptors the header
//!   counts are stepped over too, so the raw data section starts after them, and the C descriptors
//!   the receive-list flag names are read after the section
//!   (`sized_parse_steps_over_abw_descriptors_and_reads_the_receive_list`).
//! - **Buffer bounds.** libnx never checks the sizes it reads against the buffer. Here a raw data
//!   section or a descriptor list that leaves the first [`BUFFER_SIZE`] bytes is
//!   [`IpcError::Truncated`] (`sized_parse_refuses_a_raw_section_that_leaves_the_buffer` and the
//!   test above).
//! - **Data and object bounds.** libnx reads the data and the object ids wherever the sizes put
//!   them, inside the raw data section or not. Here they must lie inside the raw data section, or
//!   the parse is an [`IpcError::Inconsistent`] error
//!   (`sized_parse_refuses_data_or_objects_past_the_raw_data`). Bytes past the section are not
//!   part of the message.
//! - **Object count.** libnx ignores the domain reply header's `num_out_objects` and reads as many
//!   ids as the caller asks for (`cmifResponseGetObject`, `cmif.h:290-293`). Here the header's
//!   count says how many ids to read (`domain_reply_object_count_comes_from_its_header`).
//!
//! [`Message::parse_reply`] is for a caller that does not know the data size. The data then runs
//! to the end of the raw data section, and a domain reply's object ids are taken from the end of
//! that section. That finds the ids only when nothing follows them, which holds for a
//! [`crate::ReplyBuilder`] reply whose data is a whole number of words (its raw size is exact).
//! After data that is not, the section ends in padding and this split is wrong; libnx's split is
//! the one [`Message::parse_reply_sized`] makes. Pinned by
//! `unsized_parse_splits_domain_replies_only_after_whole_words` in `tests/libnx_golden.rs`.

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
    /// The data bytes after the `SFCO` header. From [`Message::parse_reply_sized`], exactly the
    /// expected size, or empty when the result is not zero. From [`Message::parse_reply`], the
    /// bytes up to the object ids, or to the end of the raw data section (see the module docs).
    pub data: &'a [u8],
    /// The object ids that follow the data in a domain reply, as many as the domain reply header
    /// counts. Empty from [`Message::parse_reply_sized`] when the result is not zero.
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
        /// The data after the result word: from [`Message::parse_reply_sized`], exactly the
        /// expected size, or empty when the result is not zero; from [`Message::parse_reply`],
        /// the rest of the raw data section.
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

/// Which way a message travels, and for a reply the data size the caller expects, if known.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Direction {
    Request,
    Reply(Option<usize>),
}

/// The expected reply data runs past the raw data section.
const DATA_OVERRUNS: IpcError = IpcError::Inconsistent("reply data overruns the raw data");
/// A domain message's object ids run past the raw data section.
const IDS_OVERRUN: IpcError = IpcError::Inconsistent("domain object ids overrun the raw data");

fn to_usize(value: u32) -> usize {
    usize::try_from(value).unwrap_or(usize::MAX)
}

impl<'a> Message<'a> {
    /// Parses a request out of a thread's IPC buffer.
    ///
    /// Only the first [`BUFFER_SIZE`] bytes are read. `kind` says whether the session is a
    /// domain; it only matters for CMIF request/request-with-context messages.
    pub fn parse_request(buf: &'a [u8], kind: SessionKind) -> Result<Self> {
        Self::parse(buf, kind, Direction::Request)
    }

    /// Parses a reply out of a thread's IPC buffer without knowing its data size: the data runs
    /// to the end of the raw data section, and a domain reply's object ids are taken from the
    /// end of the section. See the module docs for when that split is right; prefer
    /// [`Message::parse_reply_sized`].
    ///
    /// A CMIF reply has command type zero; a TIPC reply echoes the request's TIPC type.
    pub fn parse_reply(buf: &'a [u8], kind: SessionKind) -> Result<Self> {
        Self::parse(buf, kind, Direction::Reply(None))
    }

    /// Parses a reply out of a thread's IPC buffer as libnx does, given the number of data bytes
    /// the caller expects (`cmifParseResponse`'s `size`, `tipcParseResponse`'s `out_size`).
    ///
    /// The data is the `data_size` bytes after the `SFCO` header (or the TIPC result word), and a
    /// domain reply's object ids follow them at once. When the result is not zero the data and the
    /// object ids are empty. Fails with [`IpcError::Inconsistent`] if the data or the object ids
    /// would run past the raw data section. See the module docs for where this differs from
    /// libnx.
    pub fn parse_reply_sized(buf: &'a [u8], kind: SessionKind, data_size: usize) -> Result<Self> {
        Self::parse(buf, kind, Direction::Reply(Some(data_size)))
    }

    fn parse(buf: &'a [u8], kind: SessionKind, direction: Direction) -> Result<Self> {
        let reply = matches!(direction, Direction::Reply(_));
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
            Self::tipc_body(section, direction)?
        } else {
            // Padding to a 16-byte boundary, measured from the start of the buffer.
            let padding = raw_start
                .checked_next_multiple_of(16)
                .and_then(|aligned| aligned.checked_sub(raw_start))
                .ok_or(IpcError::Inconsistent("alignment overflows"))?;
            let content = section.get(padding..).ok_or(IpcError::Inconsistent(
                "raw data section ends inside its alignment padding",
            ))?;
            match direction {
                Direction::Reply(data_size) => Self::cmif_reply_body(content, kind, data_size)?,
                Direction::Request => Self::cmif_request_body(content, kind, command_type)?,
            }
        };
        Ok(message)
    }

    fn tipc_body(section: &'a [u8], direction: Direction) -> Result<Body<'a>> {
        let Direction::Reply(data_size) = direction else {
            return Ok(Body::TipcRequest { data: section });
        };
        // tipc.h:207: the result is the first data word.
        let mut r = Reader::new(section);
        let result = r.word()?;
        let rest = r.rest();
        let data = match data_size {
            None => rest,
            // tipc.h:208-209: a failed reply returns before the data.
            Some(_) if result != 0 => &[],
            // tipc.h:211-212: the data follows the result word.
            Some(size) => rest.get(..size).ok_or(DATA_OVERRUNS)?,
        };
        Ok(Body::TipcReply { result, data })
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

    fn cmif_reply_body(
        content: &'a [u8],
        kind: SessionKind,
        data_size: Option<usize>,
    ) -> Result<Body<'a>> {
        let mut r = Reader::new(content);
        // cmif.h:267-269: in a domain the domain reply header comes first.
        let domain = if kind == SessionKind::Domain {
            Some(DomainReplyHeader::decode(r.words()?))
        } else {
            None
        };
        // cmif.h:275: the magic is checked.
        let header = CmifReplyHeader::decode(r.words()?)?;
        let rest = r.rest();
        let count = domain.map_or(0, |domain| to_usize(domain.num_objects));
        let ids_len = count
            .checked_mul(4)
            .ok_or(IpcError::Inconsistent("domain object count overflows"))?;
        let (data, ids) = match data_size {
            None => {
                let data_len = rest.len().checked_sub(ids_len).ok_or(IDS_OVERRUN)?;
                rest.split_at_checked(data_len).ok_or(IDS_OVERRUN)?
            }
            // cmif.h:277-278: a failed reply returns before the data and the objects.
            Some(_) if header.result != 0 => (&[][..], &[][..]),
            // cmif.h:281: the data is right after the header; cmif.h:270: the objects follow the
            // expected data size.
            Some(size) => {
                let (data, tail) = rest.split_at_checked(size).ok_or(DATA_OVERRUNS)?;
                (data, tail.get(..ids_len).ok_or(IDS_OVERRUN)?)
            }
        };
        let object_ids = if ids.is_empty() {
            Vec::new()
        } else {
            Self::read_ids(ids, count)?
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
