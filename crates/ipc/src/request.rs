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
//! this builder is the inverse of the parser. Its layout is checked against libnx, the client
//! side of the protocol (`libnx/nx/include/switch/sf/hipc.h`, `cmif.h` and `tipc.h`, ISC, commit
//! `b156b4774fb2`, pinned in `docs/references.md`).
//!
//! # Two layouts
//!
//! [`RequestLayout`] picks how the raw data section of a CMIF request is sized and filled. TIPC
//! and close messages are the same in both, byte for byte
//! (`tipc_request_is_byte_exact_in_both_layouts` and `close_requests_are_byte_exact_in_both_layouts`
//! in `tests/libnx_golden.rs`).
//!
//! - [`RequestLayout::Libnx`] writes what libnx writes, byte for byte: `cmifMakeRequest`
//!   (`cmif.h:100-157`), `cmifMakeControlRequest` (`cmif.h:159-174`), `cmifMakeCloseRequest`
//!   (`cmif.h:176-193`) and `tipcMakeRequest` (`tipc.h:127-172`). The raw size starts from a
//!   16-byte reserve for the alignment padding (`cmif.h:103`), adds the domain header and object
//!   ids (`cmif.h:105`), the `SFCI` header and the arguments (`cmif.h:106`), rounds up to 2 bytes
//!   (`cmif.h:107`), adds one `u16` per out pointer (`cmif.h:108-110`) and rounds up to words
//!   (`cmif.h:111`). A domain header's `size` is the `SFCI` header plus the exact argument bytes,
//!   and the input object ids follow the arguments at once, unaligned (`cmif.h:131-141`). The
//!   out-pointer size table ([`RequestBuilder::out_pointer`]) sits at the rounded offset, measured
//!   from the start of the raw data section with the reserve counted (`cmif.h:108`,
//!   `cmif.h:153`): after the content (the object ids in a domain, the arguments otherwise), by
//!   16 bytes less the alignment padding (0 to 12 bytes) plus one more byte when the content's
//!   length is odd, so 4 to 17 bytes. Pinned by the golden-bytes tests in `tests/libnx_golden.rs`
//!   (`plain_request_is_byte_exact`, `domain_request_is_byte_exact`,
//!   `pointer_request_is_byte_exact`, `tipc_request_is_byte_exact_in_both_layouts`,
//!   `control_request_is_byte_exact`, `domain_close_request_is_byte_exact`), for each of the
//!   paddings 0, 4, 8 and 12 bytes by `out_pointer_table_follows_every_alignment_padding`, and,
//!   over generated inputs (handles, a process id and X descriptors move the section start), by
//!   `libnx_layout_matches_the_libnx_arithmetic`, all in `tests/libnx_golden.rs`.
//! - [`RequestLayout::Exact`], the default, writes the exact size as [`crate::ReplyBuilder`]
//!   does: the real padding (0 to 3 words), no reserve and no out-pointer size table. A domain
//!   header's `size` counts the arguments rounded up to a whole word, and the object ids start on
//!   a word boundary. The two layouts agree on a domain `size` when the arguments are a whole
//!   number of words. Pinned by `cmif_raw_size_differs_from_libnx_by_the_padding_reserve`,
//!   `builder_pads_domain_arguments_where_libnx_does_not` and
//!   `receive_descriptors_add_no_size_table_where_libnx_adds_one` in
//!   `tests/libnx_conventions.rs`.
//!
//! The parser takes both: it locates domain arguments and object ids from `size`, and the slack
//! of libnx's reserve is part of a plain request's data.
//!
//! # Where both layouts agree with libnx
//!
//! - Field order: special header, pid slot, copy handles, move handles, X, A, B and W
//!   descriptors, raw data, C descriptors (`hipcMakeRequest`, `hipcCalcRequestLayout`). Pinned by
//!   `every_list_lands_in_its_own_place` in `tests/request_builder.rs`.
//! - CMIF content starts on a 16-byte boundary measured from the start of the buffer
//!   (`cmifGetAlignedDataStart`), with an optional domain header, then the `SFCI` header. Pinned
//!   by `domain_requests_are_aligned_to_sixteen_bytes` in `tests/request_builder.rs`.
//! - The domain header is a send-message or close command, an input object count, a size, an
//!   object id, a padding word and a token (`CmifDomainInHeader`); a request to a domain object
//!   has the `Request` type, and a control message carries no domain header
//!   (`cmifMakeControlRequest`). Pinned by `parser_accepts_the_domain_layout_libnx_writes` and
//!   `control_and_context_conventions_are_expressible` in `tests/libnx_conventions.rs`.
//! - Closing a domain object is a `Request` with a close-command domain header and no `SFCI`
//!   header; closing a plain session is a bare `Close` message (`cmifMakeCloseRequest`). Pinned
//!   by `domain_close_matches_libnx` and `close_messages_match_libnx` in
//!   `tests/libnx_conventions.rs`.
//! - TIPC: the type is `request_id + 16`, the raw data is the arguments rounded up to words with
//!   no `SFCI` header and no padding, and closing is a bare type 15 message (`tipcMakeRequest`,
//!   `tipcClose`). Pinned by `tipc_requests_match_libnx` and `close_messages_match_libnx` in
//!   `tests/libnx_conventions.rs`.
//! - The receive list flag is `2 + n` for `n` C descriptors (`hipcMakeRequest`), so one
//!   descriptor is flag 3; the builder follows that unless told otherwise (see
//!   [`RequestBuilder::receive_list`]). Pinned by `one_receive_descriptor_is_flag_three_unless_set`
//!   in `tests/libnx_conventions.rs`.
//! - The `SFCI` token and version words are free for the caller to set. libnx writes version 0
//!   and token 0 for ordinary requests and control messages; with a context it uses the
//!   `RequestWithContext` type, version 1, and puts the context in the domain header's token
//!   (in a domain) or the `SFCI` token (otherwise). Pinned by
//!   `control_and_context_conventions_are_expressible` in `tests/libnx_conventions.rs`.
//!
//! # Where both layouts differ from libnx, on purpose
//!
//! - **Field limits.** libnx casts the object count to 8 bits and the size to 16 bits without a
//!   check (`cmif.h:134-135`); the builder returns [`IpcError::FieldOutOfRange`] above
//!   [`crate::MAX_DOMAIN_INPUT_OBJECTS`] and [`crate::MAX_DOMAIN_SIZE`]. The errors are pinned by
//!   `sized_bounds_name_the_field_that_overflowed` in `tests/request_builder.rs`; the limits are
//!   the field widths in `domain_limits_are_the_libnx_field_widths` in
//!   `tests/libnx_conventions.rs`.
//! - **Descriptor address bits.** libnx stores address bits 36 and up in a 6-bit field of an X
//!   descriptor (`HipcStaticDescriptor::address_high`, `hipc.h:51`) and a 22-bit field of an
//!   A/B/W descriptor (`hipc.h:61`). The descriptors here follow Eden's 39-bit layout
//!   (`eden/src/core/hle/ipc.h`), so an address of `1 << 39` or more is an
//!   [`IpcError::FieldOutOfRange`] where libnx would write the high bits; pinned by
//!   `descriptors_reject_values_wider_than_their_fields` in `tests/properties.rs`. For the same
//!   reason an X descriptor's counter bits 9..12 (Eden's) are address bits 39..42 to libnx, whose
//!   index is 6 bits (`hipc.h:50`); a counter below 64 and an address below `1 << 39` encode the
//!   same either way. Pinned by `static_counter_high_bits_are_libnx_address_bits` in
//!   `tests/libnx_golden.rs`.
//! - **The process id slot.** libnx leaves the two words unwritten (`hipc.h:256-257` only skips
//!   them) and the kernel fills them in. [`RequestBuilder::send_pid`] writes the value it is
//!   given, which a libnx program never does; it exists so a test can see the value come back
//!   (`every_list_lands_in_its_own_place` in `tests/request_builder.rs`). The golden-bytes tests
//!   pass zero, the value of a zeroed buffer.
//! - **Command types libnx never writes.** libnx builds only `Request`, `RequestWithContext`,
//!   `Control` and `Close` CMIF messages. For `LegacyRequest`, `LegacyControl` and
//!   `ControlWithContext`, [`RequestLayout::Libnx`] applies the same arithmetic; that is this
//!   builder's choice, not something libnx shows.

use crate::builder::{Parts, extend_words, fill_header, out_of_range, push_special_header, to_u32};
use crate::cmif::{CMIF_HEADER_BYTES, CmifRequestHeader, DomainCommand, DomainRequestHeader};
use crate::error::{IpcError, Result};
use crate::hipc::{
    BufferDescriptor, CommandHeader, CommandType, RAW_ALIGNMENT_WORDS, ReceiveDescriptor,
    ReceiveList, StaticDescriptor, align_raw, push_data, write_words,
};

const WORD: usize = size_of::<u32>();

/// The fixed reserve libnx counts for the alignment padding at the start of every CMIF raw size
/// (`actual_size = 16`, `cmif.h:103`; `16 + ...`, `cmif.h:161` and `cmif.h:181`). It is one
/// alignment unit.
const LIBNX_RESERVE_BYTES: usize = RAW_ALIGNMENT_WORDS * WORD;

/// Which conventions [`RequestBuilder`] follows for the raw data section of a CMIF request. See
/// the [module docs](self) for both layouts and where each one differs from libnx.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum RequestLayout {
    /// The exact size: the real alignment padding, the arguments rounded up to whole words before
    /// any object ids, and no out-pointer size table.
    #[default]
    Exact,
    /// libnx's layout, byte for byte: the 16-byte reserve, the exact argument length with the
    /// object ids straight after it, and the out-pointer size table.
    Libnx,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Framing {
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
/// [`RequestBuilder::tipc_close`] to close the session. [`RequestBuilder::layout`] picks the
/// raw data conventions. Data is padded with zeros to a whole number of words.
///
/// Nothing is checked until [`RequestBuilder::build`], which reports every problem as an
/// [`IpcError`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RequestBuilder {
    framing: Framing,
    layout: RequestLayout,
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
    out_pointer_sizes: Vec<u16>,
}

impl RequestBuilder {
    fn new(framing: Framing) -> Self {
        Self {
            framing,
            layout: RequestLayout::default(),
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
            out_pointer_sizes: Vec::new(),
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
        Self::new(Framing::Cmif {
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
        Self::new(Framing::Domain {
            command_type,
            object_id,
            command_id,
        })
    }

    /// A domain request that closes the virtual handle of `object_id`: type
    /// [`CommandType::Request`], a domain header with the close-virtual-handle command, and no
    /// `SFCI` header and no data. Building fails if data or object ids were added.
    pub fn domain_close(object_id: u32) -> Self {
        Self::new(Framing::DomainClose { object_id })
    }

    /// A TIPC request for the command with this id: type `16 + command_id`, and the raw data is
    /// the arguments, with no `SFCI` header and no alignment padding. Building fails with
    /// [`IpcError::TipcCommandOutOfRange`] if the type would not fit 16 bits.
    pub fn tipc(command_id: u16) -> Self {
        Self::new(Framing::Tipc { command_id })
    }

    /// A close-session message ([`CommandType::Close`]): only the two header words are
    /// populated. Building fails if anything else was added.
    pub fn close() -> Self {
        Self::new(Framing::Close { tipc: false })
    }

    /// A TIPC close-session message ([`CommandType::TipcClose`]); as [`RequestBuilder::close`].
    pub fn tipc_close() -> Self {
        Self::new(Framing::Close { tipc: true })
    }

    /// Picks the raw data conventions (default [`RequestLayout::Exact`]). It changes CMIF and
    /// domain requests only; TIPC and close messages are the same in both layouts.
    #[must_use]
    pub fn layout(mut self, layout: RequestLayout) -> Self {
        self.layout = layout;
        self
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

    /// Adds a C (receive list) descriptor, written after the raw data section. This is libnx's
    /// out *fixed* pointer (`cmifRequestOutFixedPointer`, `cmif.h:216-220`): it has no entry in
    /// the out-pointer size table.
    #[must_use]
    pub fn receive_descriptor(mut self, descriptor: ReceiveDescriptor) -> Self {
        self.receive_descriptors.push(descriptor);
        self
    }

    /// Adds an out pointer, as libnx's `cmifRequestOutPointer` does (`cmif.h:222-226`): a C
    /// descriptor in the receive list, in order with those from
    /// [`RequestBuilder::receive_descriptor`], and the descriptor's size as the next `u16` of the
    /// out-pointer size table. libnx's out auto-select buffers (`cmif.h:239-248`) add one too.
    ///
    /// The table exists only in [`RequestLayout::Libnx`] and only in a request with an `SFCI`
    /// header; building fails with [`IpcError::Inconsistent`] otherwise.
    #[must_use]
    pub fn out_pointer(mut self, descriptor: ReceiveDescriptor) -> Self {
        self.out_pointer_sizes.push(descriptor.size);
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
    /// At most the first [`crate::BUFFER_SIZE`] bytes of `buf` are used. The whole usable area is
    /// zeroed first, so no stale words survive. Fails without panicking if the message does not
    /// fit, if a count, size or id is out of range, or if the parts added do not fit the
    /// constructor or the layout used; on failure `buf` is left unchanged.
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
        let content = self.raw_content()?;
        if matches!(self.framing, Framing::Tipc { .. } | Framing::Close { .. }) {
            // No padding and no reserve: TIPC sizes are exact in libnx too (`tipc.h:152`).
            push_data(words, &content);
            return Ok(());
        }
        let raw_start = words.len();
        // Align the content to 16 bytes, measured from the start of the buffer
        // (`cmifGetAlignedDataStart`, `cmif.h:94-98`).
        align_raw(words);
        match self.layout {
            RequestLayout::Exact => push_data(words, &content),
            RequestLayout::Libnx => {
                let padding = words
                    .len()
                    .checked_sub(raw_start)
                    .and_then(|padding| padding.checked_mul(WORD))
                    .ok_or(IpcError::Inconsistent("alignment overflows"))?;
                push_data(words, &self.libnx_section(&content, padding)?);
            }
        }
        Ok(())
    }

    /// The raw data section after `padding` alignment bytes, as `cmifMakeRequest` sizes and fills
    /// it: the content, the slack of the reserve, and the out-pointer size table.
    fn libnx_section(&self, content: &[u8], padding: usize) -> Result<Vec<u8>> {
        let overflow = || out_of_range("raw data words", usize::MAX);
        // cmif.h:103-107: the reserve, then the content, rounded up to a half word.
        let table_offset = LIBNX_RESERVE_BYTES
            .checked_add(content.len())
            .and_then(|size| size.checked_next_multiple_of(size_of::<u16>()))
            .ok_or_else(overflow)?;
        // cmif.h:108-110: one u16 per out pointer at that offset.
        let table_end = self
            .out_pointer_sizes
            .len()
            .checked_mul(size_of::<u16>())
            .and_then(|table| table.checked_add(table_offset))
            .ok_or_else(overflow)?;
        // cmif.h:111: whole words.
        let raw_len = table_end
            .checked_next_multiple_of(WORD)
            .ok_or_else(overflow)?;
        // The offsets count from the start of the section; this vector starts after the padding.
        let too_small =
            IpcError::Inconsistent("the libnx raw data section is smaller than its content");
        let mut section = vec![0_u8; raw_len.checked_sub(padding).ok_or(too_small.clone())?];
        section
            .get_mut(..content.len())
            .ok_or(too_small.clone())?
            .copy_from_slice(content);
        let table = section
            .get_mut(table_offset.saturating_sub(padding)..table_end.saturating_sub(padding))
            .ok_or(too_small)?;
        for (slot, size) in table
            .as_chunks_mut::<{ size_of::<u16>() }>()
            .0
            .iter_mut()
            .zip(&self.out_pointer_sizes)
        {
            *slot = size.to_le_bytes();
        }
        Ok(section)
    }

    /// The bytes of the raw data section after the alignment padding, before any reserve or
    /// table: the domain header, the `SFCI` header, the arguments and the object ids.
    fn raw_content(&self) -> Result<Vec<u8>> {
        let data = &self.parts.data;
        let mut out = Vec::new();
        match self.framing {
            Framing::Tipc { .. } => out.extend_from_slice(data),
            Framing::Close { .. } => {}
            Framing::Cmif { command_id, .. } => {
                extend_words(&mut out, &self.cmif_header(command_id));
                out.extend_from_slice(data);
            }
            Framing::Domain {
                object_id,
                command_id,
                ..
            } => {
                let too_big = || out_of_range("domain message size", usize::MAX);
                // libnx counts the exact argument bytes (`payload_size`, `cmif.h:131`); the exact
                // layout rounds them up to a whole word.
                let args_len = match self.layout {
                    RequestLayout::Exact => data
                        .len()
                        .checked_next_multiple_of(WORD)
                        .ok_or_else(too_big)?,
                    RequestLayout::Libnx => data.len(),
                };
                let size = args_len
                    .checked_add(CMIF_HEADER_BYTES)
                    .ok_or_else(too_big)?;
                let header = DomainRequestHeader {
                    command: DomainCommand::SendMessage,
                    input_object_count: to_u32(
                        "domain input object count",
                        self.parts.object_ids.len(),
                    )?,
                    size: to_u32("domain message size", size)?,
                    object_id,
                    token: self.domain_token.unwrap_or_default(),
                };
                extend_words(&mut out, &header.encode()?);
                extend_words(&mut out, &self.cmif_header(command_id));
                let args_end = out.len().checked_add(args_len).ok_or_else(too_big)?;
                out.extend_from_slice(data);
                out.resize(args_end, 0);
                // The object ids follow the arguments (`cmif.h:141`).
                extend_words(&mut out, &self.parts.object_ids);
            }
            Framing::DomainClose { object_id } => {
                let header = DomainRequestHeader {
                    command: DomainCommand::CloseVirtualHandle,
                    input_object_count: 0,
                    size: 0,
                    object_id,
                    token: self.domain_token.unwrap_or_default(),
                };
                extend_words(&mut out, &header.encode()?);
            }
        }
        Ok(out)
    }

    /// Checks the parts against the constructor used. Returns the command type for the header.
    fn validate(&self) -> Result<CommandType> {
        let (command_type, domain) = match self.framing {
            Framing::Cmif { command_type, .. } => {
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
            Framing::Domain { command_type, .. } => {
                if !matches!(
                    command_type,
                    CommandType::Request | CommandType::RequestWithContext
                ) {
                    return Err(IpcError::UnexpectedCommandType(command_type));
                }
                (command_type, true)
            }
            Framing::DomainClose { .. } => (CommandType::Request, true),
            Framing::Tipc { command_id } => (CommandType::Tipc(command_id), false),
            Framing::Close { tipc } => (
                if tipc {
                    CommandType::TipcClose
                } else {
                    CommandType::Close
                },
                false,
            ),
        };
        match self.framing {
            Framing::Domain { .. } => {}
            Framing::DomainClose { .. } => {
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
        let has_sfci = matches!(self.framing, Framing::Cmif { .. } | Framing::Domain { .. });
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
        if matches!(self.framing, Framing::Close { .. }) && self.carries_anything() {
            return Err(IpcError::Inconsistent(
                "a close message carries nothing but its header",
            ));
        }
        if !self.out_pointer_sizes.is_empty() {
            if !has_sfci {
                return Err(IpcError::Inconsistent(
                    "only a CMIF request with an SFCI header has an out-pointer size table",
                ));
            }
            if self.layout != RequestLayout::Libnx {
                return Err(IpcError::Inconsistent(
                    "the out-pointer size table is written only in the libnx layout",
                ));
            }
        }
        Ok(command_type)
    }

    /// Whether anything beyond the framing itself was added.
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

    fn cmif_header(&self, command_id: u32) -> [u32; 4] {
        CmifRequestHeader {
            version: self.version.unwrap_or_default(),
            command_id,
            token: self.token.unwrap_or_default(),
        }
        .encode()
    }
}
