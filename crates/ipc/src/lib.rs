//! CMIF and TIPC message framing.
//!
//! Parses and builds the request/response words in a thread's IPC buffer: headers, handle
//! descriptors, buffer descriptors and raw data. Everything here is pure data handling over
//! `&[u8]` / `&mut [u8]` (the buffer is [`BUFFER_SIZE`] bytes of little-endian words); there are no
//! sessions, ports, handle tables or guest-memory reads, and no input can make it panic.
//!
//! The layers, from the outside in:
//!
//! - [`hipc`]: the two header words, the special header, handles and the X/A/B/W/C descriptors.
//! - [`cmif`]: the `SFCI`/`SFCO` payload headers and the domain message headers.
//! - TIPC: command type `16 + command id`, no `SFCI`/`SFCO` header and no alignment padding. A
//!   reply's raw data starts with a single result word. Eden's `ResponseBuilder::Push(Result)`
//!   wrote the two-word CMIF result into TIPC replies, which put reply data one word late
//!   (`docs/switch2-eden-diff-scoping.md`, row 22; `eden-nx2/NX2_NOTES.md`, row 22).
//! - [`Message`] parses a whole buffer; [`ReplyBuilder`] writes CMIF, domain and TIPC replies.
//!
//! Ported from `eden/src/core/hle/ipc.h`, `eden/src/core/hle/service/hle_ipc.cpp` and
//! `eden/src/core/hle/service/ipc_helpers.h` (GPL-3.0-or-later), with the TIPC reply layout taken
//! from `eden-nx2/src/core/hle/service/nx2_gsv.h` (`TipcReplyWords`).

pub mod cmif;
pub mod error;
pub mod hipc;
pub mod message;
pub mod reply;

pub use cmif::{
    CMIF_HEADER_BYTES, CmifReplyHeader, CmifRequestHeader, DOMAIN_HEADER_BYTES, DomainCommand,
    DomainReplyHeader, DomainRequestHeader, SFCI_MAGIC, SFCO_MAGIC,
};
pub use error::{IpcError, Result};
pub use hipc::{
    BUFFER_SIZE, BufferDescriptor, CommandHeader, CommandType, MAX_DESCRIPTORS, MAX_HANDLES,
    MAX_RAW_WORDS, MAX_TIPC_COMMAND_ID, ReceiveDescriptor, ReceiveList, SpecialHeader,
    StaticDescriptor, TIPC_COMMAND_OFFSET,
};
pub use message::{Body, CmifReply, CmifRequest, Message, SessionKind};
pub use reply::ReplyBuilder;
