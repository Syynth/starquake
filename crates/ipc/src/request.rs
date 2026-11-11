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
//! Eden only ever *parses* requests, so no Eden code builds one; this builder is a design choice
//! that follows from the parser, not a protocol fact. Two details are guesses beyond the parser:
//!
//! - Guess: a real client (libnx) may reserve extra padding words in the raw data size; this
//!   builder writes the exact size, as [`crate::ReplyBuilder`] does.
//! - Guess: in a domain request the `size` field counts the `SFCI` header plus the arguments
//!   **including** the zero padding that rounds them up to a whole word (see `cmif.rs`: the only
//!   reading under which the trailing object ids can be located).

use crate::cmif::{CMIF_HEADER_BYTES, CmifRequestHeader, DomainCommand, DomainRequestHeader};
use crate::error::{IpcError, Result};
use crate::hipc::{
    BUFFER_SIZE, BufferDescriptor, CommandHeader, CommandType, ReceiveDescriptor, ReceiveList,
    SpecialHeader, StaticDescriptor, high32, low32,
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
/// [`RequestBuilder::tipc`] for TIPC, and [`RequestBuilder::close`] / [`RequestBuilder::tipc_close`]
/// to close the session. Data is padded with zeros to a whole number of words.
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
    copy_handles: Vec<u32>,
    move_handles: Vec<u32>,
    statics: Vec<StaticDescriptor>,
    sends: Vec<BufferDescriptor>,
    receives: Vec<BufferDescriptor>,
    exchanges: Vec<BufferDescriptor>,
    receive_list: Option<ReceiveList>,
    receive_descriptors: Vec<ReceiveDescriptor>,
    data: Vec<u8>,
    object_ids: Vec<u32>,
}

impl RequestBuilder {
    fn new(layout: Layout) -> Self {
        Self {
            layout,
            version: None,
            token: None,
            domain_token: None,
            pid: None,
            copy_handles: Vec::new(),
            move_handles: Vec::new(),
            statics: Vec::new(),
            sends: Vec::new(),
            receives: Vec::new(),
            exchanges: Vec::new(),
            receive_list: None,
            receive_descriptors: Vec::new(),
            data: Vec::new(),
            object_ids: Vec::new(),
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

    /// Asks for the sender's process id to be attached, with `pid` in the two pid words. A real
    /// client passes zero: the slot holds zero until the kernel fills it
    /// (`eden/src/core/hle/service/hle_ipc.cpp` skips the two words and reads the pid from the
    /// owning process).
    #[must_use]
    pub fn send_pid(mut self, pid: u64) -> Self {
        self.pid = Some(pid);
        self
    }

    /// Adds a handle to copy to the receiver.
    #[must_use]
    pub fn copy_handle(mut self, handle: u32) -> Self {
        self.copy_handles.push(handle);
        self
    }

    /// Adds a handle to move to the receiver.
    #[must_use]
    pub fn move_handle(mut self, handle: u32) -> Self {
        self.move_handles.push(handle);
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
    /// descriptors added: none is [`ReceiveList::Disabled`], one is [`ReceiveList::One`], `n` is
    /// [`ReceiveList::Many`]`(n)`. Set it to say [`ReceiveList::Inline`], or to use
    /// [`ReceiveList::Many`]`(1)` (flag 3) rather than [`ReceiveList::One`] (flag 2). Building
    /// fails with [`IpcError::Inconsistent`] if it disagrees with the C descriptors added.
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
        self.data.extend_from_slice(bytes);
        self
    }

    /// Appends a domain input object id after the arguments. Building fails with
    /// [`IpcError::ObjectIdsOutsideDomain`] unless the request comes from
    /// [`RequestBuilder::domain`].
    #[must_use]
    pub fn object_id(mut self, id: u32) -> Self {
        self.object_ids.push(id);
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
        let has_special_header =
            self.pid.is_some() || !self.copy_handles.is_empty() || !self.move_handles.is_empty();

        // The two header words are filled in last, once the raw data size is known.
        let mut words = vec![0, 0];
        if has_special_header {
            let special = SpecialHeader {
                send_pid: self.pid.is_some(),
                num_copy_handles: to_u32("copy handle count", self.copy_handles.len())?,
                num_move_handles: to_u32("move handle count", self.move_handles.len())?,
            };
            words.push(special.encode()?);
            if let Some(pid) = self.pid {
                words.extend([low32(pid), high32(pid)]);
            }
            words.extend_from_slice(&self.copy_handles);
            words.extend_from_slice(&self.move_handles);
        }
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
        let [w0, w1] = header.encode()?;
        if let [first, second, ..] = words.as_mut_slice() {
            *first = w0;
            *second = w1;
        }
        Ok(words)
    }

    /// Appends the raw data section.
    fn push_raw(&self, words: &mut Vec<u32>) -> Result<()> {
        match self.layout {
            Layout::Tipc { .. } => self.push_data(words),
            Layout::Close { .. } => {}
            Layout::Cmif { command_id, .. } => {
                align(words);
                self.push_cmif_header(words, command_id);
                self.push_data(words);
            }
            Layout::Domain {
                object_id,
                command_id,
                ..
            } => {
                align(words);
                // The size counts the SFCI header and the arguments, padded to whole words.
                let size = self
                    .data
                    .len()
                    .div_ceil(4)
                    .checked_mul(4)
                    .and_then(|bytes| bytes.checked_add(CMIF_HEADER_BYTES))
                    .ok_or_else(|| out_of_range("domain message size", usize::MAX))?;
                words.extend(
                    DomainRequestHeader {
                        command: DomainCommand::SendMessage,
                        input_object_count: to_u32(
                            "domain input object count",
                            self.object_ids.len(),
                        )?,
                        size: to_u32("domain message size", size)?,
                        object_id,
                        token: self.domain_token.unwrap_or_default(),
                    }
                    .encode()?,
                );
                self.push_cmif_header(words, command_id);
                self.push_data(words);
                words.extend_from_slice(&self.object_ids);
            }
            Layout::DomainClose { object_id } => {
                align(words);
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
                if !self.object_ids.is_empty() || !self.data.is_empty() {
                    return Err(IpcError::Inconsistent(
                        "closing a domain virtual handle carries no data or object ids",
                    ));
                }
            }
            _ => {
                if !self.object_ids.is_empty() {
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
            || !self.copy_handles.is_empty()
            || !self.move_handles.is_empty()
            || !self.statics.is_empty()
            || !self.sends.is_empty()
            || !self.receives.is_empty()
            || !self.exchanges.is_empty()
            || self.receive_list.is_some()
            || !self.receive_descriptors.is_empty()
            || !self.data.is_empty()
            || !self.object_ids.is_empty()
    }

    /// The receive list flag: as set, or following the C descriptors added; either way it must
    /// announce exactly as many descriptors as there are.
    fn resolve_receive_list(&self) -> Result<ReceiveList> {
        let count = to_u32("receive descriptor count", self.receive_descriptors.len())?;
        let list = self.receive_list.unwrap_or(match count {
            0 => ReceiveList::Disabled,
            1 => ReceiveList::One,
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

    fn push_data(&self, words: &mut Vec<u32>) {
        for chunk in self.data.chunks(4) {
            let mut word = [0_u8; 4];
            for (slot, byte) in word.iter_mut().zip(chunk) {
                *slot = *byte;
            }
            words.push(u32::from_le_bytes(word));
        }
    }
}

/// Pads with zero words so the next word starts a 16-byte boundary, measured from the start of
/// the buffer.
fn align(words: &mut Vec<u32>) {
    words.resize(words.len().next_multiple_of(4), 0);
}

fn out_of_range(field: &'static str, value: usize) -> IpcError {
    IpcError::FieldOutOfRange {
        field,
        value: u64::try_from(value).unwrap_or(u64::MAX),
    }
}

/// A count as a header word, or the out-of-range error naming `field`.
fn to_u32(field: &'static str, value: usize) -> Result<u32> {
    u32::try_from(value).map_err(|_| out_of_range(field, value))
}

/// Writes `words` little-endian into the usable start of `buf` and zeroes the rest of the usable
/// area; `buf` is untouched if they do not fit. Returns the bytes written.
fn write_words(words: &[u32], buf: &mut [u8]) -> Result<usize> {
    let needed = words.len().saturating_mul(4);
    let capacity = buf.len().min(BUFFER_SIZE);
    let usable = buf
        .get_mut(..capacity)
        .ok_or(IpcError::DoesNotFit { needed, capacity })?;
    let (out, tail) = usable
        .split_at_mut_checked(needed)
        .ok_or(IpcError::DoesNotFit { needed, capacity })?;
    for (chunk, word) in out.as_chunks_mut::<4>().0.iter_mut().zip(words) {
        *chunk = word.to_le_bytes();
    }
    tail.fill(0);
    Ok(needed)
}
