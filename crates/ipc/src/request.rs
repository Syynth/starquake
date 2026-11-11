//! Building requests into an IPC buffer: the client side, the mirror of [`crate::ReplyBuilder`].
//!
//! The layout is the one the server side parses, so it is taken from
//! `eden/src/core/hle/service/hle_ipc.cpp` (`ParseCommandBuffer`, GPL-3.0-or-later) and
//! `eden/src/core/hle/ipc.h` (`CommandHeader`, `HandleDescriptorHeader`, `DomainMessageHeader`,
//! `DataPayloadHeader`), the same sources as [`crate::Message::parse_request`]:
//!
//! 1. the two header words, then (if there are a pid or handles) the special header, the two
//!    pid words, the copy handles and the move handles;
//! 2. the X descriptors, then the A, B and W descriptors;
//! 3. the raw data section: for CMIF, padding to a 16-byte boundary measured from the start of
//!    the buffer, the domain header (domain `Request`/`RequestWithContext` only), the `SFCI`
//!    header, the arguments and the domain input object ids; for TIPC, the arguments alone;
//! 4. the C (receive list) descriptors straight after the raw data section.
//!
//! Eden's HLE only parses requests (`IPC::RequestParser`) and has no request builder to port, so
//! this builder is the inverse of the parser. Its conventions are checked against libnx, the
//! client side of the protocol (`libnx/nx/include/switch/sf/hipc.h`, `cmif.h` and `tipc.h`,
//! ISC, pinned in `docs/references.md`); `tests/libnx_conventions.rs` pins each point below.
//!
//! # Where the builder agrees with libnx
//!
//! - Field order: special header, pid slot, copy handles, move handles, X, A, B and W
//!   descriptors, raw data, C descriptors (`hipcMakeRequest`, `hipcCalcRequestLayout`).
//! - CMIF content starts on a 16-byte boundary measured from the start of the buffer
//!   (`cmifGetAlignedDataStart`), with an optional domain header, then the `SFCI` header.
//! - The domain header is a send-message or close command, an input object count, a size, an
//!   object id, a padding word and a token (`CmifDomainInHeader`); a request to a domain object
//!   has the `Request` type, and a control message carries no domain header
//!   (`cmifMakeControlRequest`).
//! - Closing a domain object is a `Request` with a close-command domain header and no `SFCI`
//!   header; closing a plain session is a bare `Close` message (`cmifMakeCloseRequest`).
//! - TIPC: the type is `request_id + 16`, the raw data is the arguments rounded up to words with
//!   no `SFCI` header and no padding, and closing is a bare type 15 message (`tipcMakeRequest`,
//!   `tipcClose`).
//! - The receive list flag is `2 + n` for `n` C descriptors (`hipcMakeRequest`), so one
//!   descriptor is flag 3; the builder follows that unless told otherwise (see
//!   [`RequestBuilder::receive_list`]).
//! - The `SFCI` token and version words are free for the caller to set. libnx writes version 0
//!   and token 0 for ordinary requests and control messages; with a context it uses the
//!   `RequestWithContext` type, version 1, and puts the context in the domain header's token
//!   (in a domain) or the `SFCI` token (otherwise).
//!
//! # Where it differs, on purpose
//!
//! - **Raw data size.** libnx starts every CMIF raw size from a fixed 16-byte reserve for the
//!   alignment padding (`actual_size = 16`), then adds the headers and arguments, rounds up to
//!   2 bytes, and adds a table of `u16` sizes for out pointers. This builder writes the exact size
//!   (the real padding, 0 to 3 words, and no table) as [`crate::ReplyBuilder`] does. The parser
//!   takes both. TIPC sizes are exact in libnx too.
//! - **Domain `size` and object ids.** libnx's `size` is `CMIF_HEADER_BYTES` plus the exact
//!   argument bytes and the object ids follow the arguments at once, unaligned. This builder
//!   rounds the arguments up to a whole word with zero bytes, counts that padding in `size` and
//!   starts the object ids on a word boundary. The two agree when the arguments are a whole
//!   number of words; otherwise the builder's `size` is larger. The parser takes both because it
//!   locates the ids from `size`.
//! - **Field limits.** libnx casts the object count to 8 bits and the size to 16 bits without a
//!   check; the builder returns [`IpcError::FieldOutOfRange`] above
//!   [`crate::MAX_DOMAIN_INPUT_OBJECTS`] and [`crate::MAX_DOMAIN_SIZE`].
//! - **Out pointer sizes.** The table of out pointer sizes (placed after the arguments and object
//!   ids, `cmifMakeRequest`) is not modelled. A caller that needs it for a request without
//!   object ids can append it to the arguments.
//!
//! # Beyond libnx
//!
//! - The process id slot: libnx leaves the two words unwritten (`hipcMakeRequest` only skips
//!   them) and the kernel fills them in. [`RequestBuilder::send_pid`] writes the value it is
//!   given, which a libnx program never does; it exists so a test can see the value come back.

use crate::builder::{Parts, fill_header, out_of_range, push_special_header, to_u32};
use crate::cmif::{CMIF_HEADER_BYTES, CmifRequestHeader, DomainCommand, DomainRequestHeader};
use crate::error::{IpcError, Result};
use crate::hipc::{
    BufferDescriptor, CommandHeader, CommandType, ReceiveDescriptor, ReceiveList, StaticDescriptor,
    align_raw, push_data, write_words,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Layout {
    Cmif {
        command_type: CommandType,
        command_id: u32,
    },
    Domain {
        command_type: CommandType,
        object_id: u32,
        command_id: u32,
    },
    DomainClose {
        object_id: u32,
    },
    Tipc {
        command_id: u16,
    },
    Close {
        tipc: bool,
    },
}

/// Accumulates the parts of a request and writes them into a buffer.
///
/// The constructors pick the framing: [`RequestBuilder::cmif`] for an ordinary session (and for
/// control messages on a domain), [`RequestBuilder::domain`] for a request to a domain object,
/// [`RequestBuilder::domain_close`] to close a domain object's virtual handle,
/// [`RequestBuilder::tipc`] for TIPC, and [`RequestBuilder::close`] /
/// [`RequestBuilder::tipc_close`] to close the session. Data is padded with zeros to a whole number of words.
///
/// Nothing is checked until [`RequestBuilder::build`], which reports every problem as an
/// [`IpcError`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RequestBuilder {
    layout: Layout,
    version: Option<u32>,
    token: Option<u32>,
    domain_token: Option<u32>,
    pid: Option<u64>,
    parts: Parts,
    statics: Vec<StaticDescriptor>,
    sends: Vec<BufferDescriptor>,
    receives: Vec<BufferDescriptor>,
    exchanges: Vec<BufferDescriptor>,
    receive_list: Option<ReceiveList>,
    receive_descriptors: Vec<ReceiveDescriptor>,
}

impl RequestBuilder {
    fn new(layout: Layout) -> Self {
        Self {
            layout,
            version: None,
            token: None,
            domain_token: None,
            pid: None,
            parts: Parts::default(),
            statics: Vec::new(),
            sends: Vec::new(),
            receives: Vec::new(),
            exchanges: Vec::new(),
            receive_list: None,
            receive_descriptors: Vec::new(),
        }
    }

    /// A CMIF request with an `SFCI` header and no domain header: the message type is
    /// `command_type`, which must be one of the CMIF request types (legacy request, request,
    /// legacy control, control, request with context, control with context). Building fails with
    /// [`IpcError::UnexpectedCommandType`] for any other type.
    ///
    /// Use this for every message on an ordinary session, and for control messages on a domain
    /// (only request and request-with-context carry a domain header,
    /// `eden/src/core/hle/service/hle_ipc.cpp`).
    pub fn cmif(command_type: CommandType, command_id: u32) -> Self {
        Self::new(Layout::Cmif {
            command_type,
            command_id,
        })
    }

    /// A request to the object `object_id` of a domain: a domain header with the send-message
    /// command, then the `SFCI` header. `command_type` must be [`CommandType::Request`] or
    /// [`CommandType::RequestWithContext`]; building fails with
    /// [`IpcError::UnexpectedCommandType`] otherwise. Input object ids are added with
    /// [`RequestBuilder::object_id`].
    pub fn domain(command_type: CommandType, object_id: u32, command_id: u32) -> Self {
        Self::new(Layout::Domain {
            command_type,
            object_id,
            command_id,
        })
    }

    /// A domain request that closes the virtual handle of `object_id`: type
    /// [`CommandType::Request`], a domain header with the close-virtual-handle command, and no
    /// `SFCI` header and no data. Building fails if data or object ids were added.
    pub fn domain_close(object_id: u32) -> Self {
        Self::new(Layout::DomainClose { object_id })
    }

    /// A TIPC request for the command with this id: type `16 + command_id`, and the raw data is
    /// the arguments, with no `SFCI` header and no alignment padding. Building fails with
    /// [`IpcError::TipcCommandOutOfRange`] if the type would not fit 16 bits.
    pub fn tipc(command_id: u16) -> Self {
        Self::new(Layout::Tipc { command_id })
    }

    /// A close-session message ([`CommandType::Close`]): only the two header words are
    /// populated. Building fails if anything else was added.
    pub fn close() -> Self {
        Self::new(Layout::Close { tipc: false })
    }

    /// A TIPC close-session message ([`CommandType::TipcClose`]); as [`RequestBuilder::close`].
    pub fn tipc_close() -> Self {
        Self::new(Layout::Close { tipc: true })
    }

    /// Sets the `SFCI` version word (default zero). Only CMIF and domain-send requests have one.
    #[must_use]
    pub fn version(mut self, version: u32) -> Self {
        self.version = Some(version);
        self
    }

    /// Sets the `SFCI` token word (default zero). Only CMIF and domain-send requests have one.
    #[must_use]
    pub fn token(mut self, token: u32) -> Self {
        self.token = Some(token);
        self
    }

    /// Sets the token word of the domain header (default zero). Only domain requests have one.
    #[must_use]
    pub fn domain_token(mut self, token: u32) -> Self {
        self.domain_token = Some(token);
        self
    }

    /// Asks for the sender's process id to be attached, with `pid` in the two pid words.
    ///
    /// A libnx program writes nothing into those words and the kernel fills them in
    /// (`libnx/nx/include/switch/sf/hipc.h`, `hipcMakeRequest`), so the value matters only to a
    /// test that wants to see it come back; pass zero for the nearest thing to an unwritten slot.
    #[must_use]
    pub fn send_pid(mut self, pid: u64) -> Self {
        self.pid = Some(pid);
        self
    }

    /// Adds a handle to copy to the receiver.
    #[must_use]
    pub fn copy_handle(mut self, handle: u32) -> Self {
        self.parts.copy_handles.push(handle);
        self
    }

    /// Adds a handle to move to the receiver.
    #[must_use]
    pub fn move_handle(mut self, handle: u32) -> Self {
        self.parts.move_handles.push(handle);
        self
    }

    /// Adds an X (static send) descriptor.
    #[must_use]
    pub fn static_descriptor(mut self, descriptor: StaticDescriptor) -> Self {
        self.statics.push(descriptor);
        self
    }

    /// Adds an A (send) descriptor.
    #[must_use]
    pub fn send_buffer(mut self, descriptor: BufferDescriptor) -> Self {
        self.sends.push(descriptor);
        self
    }

    /// Adds a B (receive) descriptor.
    #[must_use]
    pub fn receive_buffer(mut self, descriptor: BufferDescriptor) -> Self {
        self.receives.push(descriptor);
        self
    }

    /// Adds a W (exchange) descriptor.
    #[must_use]
    pub fn exchange_buffer(mut self, descriptor: BufferDescriptor) -> Self {
        self.exchanges.push(descriptor);
        self
    }

    /// Sets how the receive list is described. Without this it follows the number of C
    /// descriptors added: none is [`ReceiveList::Disabled`] and `n` is [`ReceiveList::Many`]`(n)`,
    /// so a single descriptor is flag 3, as libnx writes it (`flag = 2 + n`,
    /// `libnx/nx/include/switch/sf/hipc.h`, `hipcMakeRequest`). Set it to say
    /// [`ReceiveList::Inline`], or to use [`ReceiveList::One`] (flag 2, libnx's
    /// `HIPC_AUTO_RECV_STATIC`) for one descriptor. Building fails with [`IpcError::Inconsistent`]
    /// if it disagrees with the C descriptors added.
    #[must_use]
    pub fn receive_list(mut self, receive_list: ReceiveList) -> Self {
        self.receive_list = Some(receive_list);
        self
    }

    /// Adds a C (receive list) descriptor, written after the raw data section.
    #[must_use]
    pub fn receive_descriptor(mut self, descriptor: ReceiveDescriptor) -> Self {
        self.receive_descriptors.push(descriptor);
        self
    }

    /// Appends argument bytes.
    #[must_use]
    pub fn data(mut self, bytes: &[u8]) -> Self {
        self.parts.data.extend_from_slice(bytes);
        self
    }

    /// Appends a domain input object id after the arguments. Building fails with
    /// [`IpcError::ObjectIdsOutsideDomain`] unless the request comes from
    /// [`RequestBuilder::domain`].
    #[must_use]
    pub fn object_id(mut self, id: u32) -> Self {
        self.parts.object_ids.push(id);
        self
    }

    /// Writes the request into `buf` and returns the number of bytes used.
    ///
    /// At most the first [`BUFFER_SIZE`] bytes of `buf` are used. The whole usable area is zeroed
    /// first, so no stale words survive. Fails without panicking if the message does not fit, if
    /// a count, size or id is out of range, or if the parts added do not fit the constructor
    /// used; on failure `buf` is left unchanged.
    pub fn build(&self, buf: &mut [u8]) -> Result<usize> {
        write_words(&self.words()?, buf)
    }

    fn words(&self) -> Result<Vec<u32>> {
        let command_type = self.validate()?;

        // The two header words are filled in last, once the raw data size is known.
        let mut words = vec![0; CommandHeader::WORDS];
        let has_special_header = push_special_header(
            &mut words,
            self.pid,
            &self.parts.copy_handles,
            &self.parts.move_handles,
        )?;
        for descriptor in &self.statics {
            words.extend(descriptor.encode()?);
        }
        for descriptor in self
            .sends
            .iter()
            .chain(&self.receives)
            .chain(&self.exchanges)
        {
            words.extend(descriptor.encode()?);
        }

        let raw_start = words.len();
        self.push_raw(&mut words)?;
        let raw_data_words = to_u32("raw data words", words.len() - raw_start)?;

        let receive_list = self.resolve_receive_list()?;
        for descriptor in &self.receive_descriptors {
            words.extend(descriptor.encode()?);
        }

        let header = CommandHeader {
            command_type,
            num_x: to_u32("X descriptor count", self.statics.len())?,
            num_a: to_u32("A descriptor count", self.sends.len())?,
            num_b: to_u32("B descriptor count", self.receives.len())?,
            num_w: to_u32("W descriptor count", self.exchanges.len())?,
            raw_data_words,
            receive_list,
            has_special_header,
        };
        fill_header(&mut words, &header)?;
        Ok(words)
    }

    /// Appends the raw data section.
    fn push_raw(&self, words: &mut Vec<u32>) -> Result<()> {
        match self.layout {
            Layout::Tipc { .. } => push_data(words, &self.parts.data),
            Layout::Close { .. } => {}
            Layout::Cmif { command_id, .. } => {
                align_raw(words);
                self.push_cmif_header(words, command_id);
                push_data(words, &self.parts.data);
            }
            Layout::Domain {
                object_id,
                command_id,
                ..
            } => {
                align_raw(words);
                // The size counts the SFCI header and the arguments, padded to whole words.
                let size = self
                    .parts
                    .data
                    .len()
                    .checked_next_multiple_of(size_of::<u32>())
                    .and_then(|bytes| bytes.checked_add(CMIF_HEADER_BYTES))
                    .ok_or_else(|| out_of_range("domain message size", usize::MAX))?;
                words.extend(
                    DomainRequestHeader {
                        command: DomainCommand::SendMessage,
                        input_object_count: to_u32(
                            "domain input object count",
                            self.parts.object_ids.len(),
                        )?,
                        size: to_u32("domain message size", size)?,
                        object_id,
                        token: self.domain_token.unwrap_or_default(),
                    }
                    .encode()?,
                );
                self.push_cmif_header(words, command_id);
                push_data(words, &self.parts.data);
                words.extend_from_slice(&self.parts.object_ids);
            }
            Layout::DomainClose { object_id } => {
                align_raw(words);
                words.extend(
                    DomainRequestHeader {
                        command: DomainCommand::CloseVirtualHandle,
                        input_object_count: 0,
                        size: 0,
                        object_id,
                        token: self.domain_token.unwrap_or_default(),
                    }
                    .encode()?,
                );
            }
        }
        Ok(())
    }

    /// Checks the parts against the constructor used. Returns the command type for the header.
    fn validate(&self) -> Result<CommandType> {
        let (command_type, domain) = match self.layout {
            Layout::Cmif { command_type, .. } => {
                if matches!(
                    command_type,
                    CommandType::Invalid
                        | CommandType::Close
                        | CommandType::TipcClose
                        | CommandType::Tipc(_)
                ) {
                    return Err(IpcError::UnexpectedCommandType(command_type));
                }
                (command_type, false)
            }
            Layout::Domain { command_type, .. } => {
                if !matches!(
                    command_type,
                    CommandType::Request | CommandType::RequestWithContext
                ) {
                    return Err(IpcError::UnexpectedCommandType(command_type));
                }
                (command_type, true)
            }
            Layout::DomainClose { .. } => (CommandType::Request, true),
            Layout::Tipc { command_id } => (CommandType::Tipc(command_id), false),
            Layout::Close { tipc } => (
                if tipc {
                    CommandType::TipcClose
                } else {
                    CommandType::Close
                },
                false,
            ),
        };
        match self.layout {
            Layout::Domain { .. } => {}
            Layout::DomainClose { .. } => {
                if !self.parts.object_ids.is_empty() || !self.parts.data.is_empty() {
                    return Err(IpcError::Inconsistent(
                        "closing a domain virtual handle carries no data or object ids",
                    ));
                }
            }
            _ => {
                if !self.parts.object_ids.is_empty() {
                    return Err(IpcError::ObjectIdsOutsideDomain);
                }
            }
        }
        let has_sfci = matches!(self.layout, Layout::Cmif { .. } | Layout::Domain { .. });
        if !has_sfci && (self.version.is_some() || self.token.is_some()) {
            return Err(IpcError::Inconsistent(
                "version and token belong to the SFCI header, which this message does not have",
            ));
        }
        if !domain && self.domain_token.is_some() {
            return Err(IpcError::Inconsistent(
                "the domain token belongs to the domain header, which this message does not have",
            ));
        }
        if matches!(self.layout, Layout::Close { .. }) && self.carries_anything() {
            return Err(IpcError::Inconsistent(
                "a close message carries nothing but its header",
            ));
        }
        Ok(command_type)
    }

    /// Whether anything beyond the layout itself was added.
    fn carries_anything(&self) -> bool {
        self.pid.is_some()
            || !self.parts.copy_handles.is_empty()
            || !self.parts.move_handles.is_empty()
            || !self.statics.is_empty()
            || !self.sends.is_empty()
            || !self.receives.is_empty()
            || !self.exchanges.is_empty()
            || self.receive_list.is_some()
            || !self.receive_descriptors.is_empty()
            || !self.parts.data.is_empty()
            || !self.parts.object_ids.is_empty()
    }

    /// The receive list flag: as set, or following the C descriptors added; either way it must
    /// announce exactly as many descriptors as there are.
    fn resolve_receive_list(&self) -> Result<ReceiveList> {
        let count = to_u32("receive descriptor count", self.receive_descriptors.len())?;
        let list = self.receive_list.unwrap_or(match count {
            0 => ReceiveList::Disabled,
            n => ReceiveList::Many(n),
        });
        if list.descriptor_count() == count {
            Ok(list)
        } else {
            Err(IpcError::Inconsistent(
                "the receive list flag does not match the number of receive descriptors",
            ))
        }
    }

    fn push_cmif_header(&self, words: &mut Vec<u32>, command_id: u32) {
        words.extend(
            CmifRequestHeader {
                version: self.version.unwrap_or_default(),
                command_id,
                token: self.token.unwrap_or_default(),
            }
            .encode(),
        );
    }
}
