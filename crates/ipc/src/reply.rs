//! Building replies into an IPC buffer.
//!
//! Ported from `eden/src/core/hle/service/ipc_helpers.h` (`ResponseBuilder`), with three
//! deliberate differences:
//!
//! - A TIPC reply's raw data is **one** result word followed directly by the data, as in
//!   `eden-nx2/src/core/hle/service/nx2_gsv.h` (`TipcReplyWords`). Eden's builder wrote the
//!   two-word CMIF result there, which put TIPC data one word late (scoping doc row 22).
//! - The raw data size in the header is exact (alignment padding plus the words actually
//!   written). Eden's `ResponseBuilder` adds the payload size twice and reserves four extra
//!   padding words. libnx locates reply fields by the sizes it knows, not by this field
//!   (`cmifParseResponse`, `libnx/nx/include/switch/sf/cmif.h:260-288`, never reads it for CMIF;
//!   `tipcParseResponse`, `tipc.h:199-228`, neither), so an exact count is the least surprising
//!   choice. This is a design choice, not a protocol fact.
//! - A domain reply's object ids follow the data at once, unaligned, where libnx's
//!   `cmifParseResponse` reads them: at the `SFCO` header plus the data size the caller expects
//!   (`cmif.h:270`). Data that is not a whole number of words is not padded before them; only the
//!   end of the raw data section is padded to a word. Pinned by
//!   `domain_reply_puts_objects_where_libnx_reads_them` in `tests/libnx_golden.rs`.
//!
//! Neither Eden nor libnx says what type word a reply carries (libnx's `hipcParseResponse`,
//! `hipc.h:316-363`, reads only the counts, the size and the special header). This builder writes
//! zero for CMIF and echoes the request's type for TIPC, as Eden's `ResponseBuilder` does.

use crate::builder::{Parts, fill_header, push_special_header, to_u32};
use crate::cmif::{CmifReplyHeader, DomainReplyHeader};
use crate::error::{IpcError, Result};
use crate::hipc::{
    CommandHeader, CommandType, MAX_TIPC_COMMAND_ID, ReceiveList, align_raw, push_data, write_words,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Layout {
    Cmif { domain: bool },
    Tipc { command_id: u16 },
}

/// Accumulates the parts of a reply and writes them into a buffer.
///
/// The three constructors pick the framing: [`ReplyBuilder::cmif`] for an ordinary session,
/// [`ReplyBuilder::domain`] for a domain, [`ReplyBuilder::tipc`] for TIPC. Data is padded with
/// zeros to a whole number of words.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplyBuilder {
    layout: Layout,
    result: u32,
    parts: Parts,
}

impl ReplyBuilder {
    fn new(layout: Layout, result: u32) -> Self {
        Self {
            layout,
            result,
            parts: Parts::default(),
        }
    }

    /// A reply on an ordinary CMIF session: command type zero, `SFCO` header with `result`.
    pub fn cmif(result: u32) -> Self {
        Self::new(Layout::Cmif { domain: false }, result)
    }

    /// A reply on a domain: as [`ReplyBuilder::cmif`], plus the domain reply header and any
    /// object ids added with [`ReplyBuilder::object_id`].
    ///
    /// Use this only for replies to request/request-with-context messages that carried a domain
    /// header. Eden writes the domain reply header only in that case
    /// (`eden/src/core/hle/service/ipc_helpers.h`: `mgr->IsDomain() && ctx.HasDomainMessageHeader()`),
    /// so replies to control requests on a domain use [`ReplyBuilder::cmif`].
    pub fn domain(result: u32) -> Self {
        Self::new(Layout::Cmif { domain: true }, result)
    }

    /// A TIPC reply to the TIPC request with this command id: the reply echoes the request's
    /// type (`16 + command_id`), and its raw data is `result` followed by the data. No `SFCO`
    /// header, no alignment padding.
    pub fn tipc(command_id: u16, result: u32) -> Self {
        Self::new(Layout::Tipc { command_id }, result)
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

    /// Appends data bytes.
    #[must_use]
    pub fn data(mut self, bytes: &[u8]) -> Self {
        self.parts.data.extend_from_slice(bytes);
        self
    }

    /// Appends a domain object id. The ids follow the data at once, with no padding between, in
    /// the order added. Building fails unless the reply is a domain reply.
    #[must_use]
    pub fn object_id(mut self, id: u32) -> Self {
        self.parts.object_ids.push(id);
        self
    }

    /// Writes the reply into `buf` and returns the number of bytes used.
    ///
    /// At most the first [`BUFFER_SIZE`] bytes of `buf` are used. The whole usable area is zeroed
    /// first, so no stale words survive. Fails without panicking if the message does not fit, if
    /// a count or id is out of range, or if object ids were added to a non-domain reply; on
    /// failure `buf` is left unchanged.
    pub fn build(&self, buf: &mut [u8]) -> Result<usize> {
        write_words(&self.words()?, buf)
    }

    fn words(&self) -> Result<Vec<u32>> {
        let parts = &self.parts;
        if !parts.object_ids.is_empty() && self.layout != (Layout::Cmif { domain: true }) {
            return Err(IpcError::ObjectIdsOutsideDomain);
        }

        // The two header words are filled in last, once the raw data size is known.
        let mut words = vec![0; CommandHeader::WORDS];
        let has_special_header =
            push_special_header(&mut words, None, &parts.copy_handles, &parts.move_handles)?;
        let raw_start = words.len();

        let command_type = match self.layout {
            Layout::Tipc { command_id } => {
                if command_id > MAX_TIPC_COMMAND_ID {
                    return Err(IpcError::TipcCommandOutOfRange(command_id));
                }
                words.push(self.result);
                push_data(&mut words, &parts.data);
                CommandType::Tipc(command_id)
            }
            Layout::Cmif { domain } => {
                // Align the payload to 16 bytes, measured from the start of the buffer.
                align_raw(&mut words);
                if domain {
                    words.extend_from_slice(
                        &DomainReplyHeader {
                            num_objects: to_u32("domain object count", parts.object_ids.len())?,
                        }
                        .encode(),
                    );
                }
                words.extend_from_slice(
                    &CmifReplyHeader {
                        version: 0,
                        result: self.result,
                        token: 0,
                    }
                    .encode(),
                );
                // The object ids follow the data at once, unaligned: libnx reads them at the
                // `SFCO` header plus the data size it expects (`cmif.h:270`).
                let mut tail = parts.data.clone();
                for id in &parts.object_ids {
                    tail.extend_from_slice(&id.to_le_bytes());
                }
                push_data(&mut words, &tail);
                CommandType::Invalid
            }
        };

        let header = CommandHeader {
            command_type,
            num_x: 0,
            num_a: 0,
            num_b: 0,
            num_w: 0,
            raw_data_words: to_u32("raw data words", words.len() - raw_start)?,
            receive_list: ReceiveList::Disabled,
            has_special_header,
        };
        fill_header(&mut words, &header)?;
        Ok(words)
    }
}
