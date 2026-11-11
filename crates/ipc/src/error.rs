//! The error type shared by every layer of the framing code.

use crate::hipc::CommandType;

/// Result alias for this crate.
pub type Result<T, E = IpcError> = core::result::Result<T, E>;

/// Why a message could not be parsed or built.
///
/// Parsing never panics: every malformed buffer maps to one of these.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum IpcError {
    /// A read or write went past the end of the usable buffer.
    #[error("message truncated: {needed} bytes needed at offset {offset}, buffer holds {len}")]
    Truncated {
        /// Byte offset of the access.
        offset: usize,
        /// Bytes the access needed.
        needed: usize,
        /// Usable length of the buffer.
        len: usize,
    },
    /// The command-type field holds a value the protocol does not define.
    #[error("unknown command type {0:#x}")]
    UnknownCommandType(u16),
    /// A defined command type that is not valid in this direction (for example a request whose
    /// type is zero, or a reply that claims to be a close).
    #[error("command type {0:?} is not valid here")]
    UnexpectedCommandType(CommandType),
    /// An `SFCI`/`SFCO` magic did not match.
    #[error("bad magic {found:#010x}, expected {expected:#010x}")]
    BadMagic {
        /// The word found in the buffer.
        found: u32,
        /// The word the layout requires.
        expected: u32,
    },
    /// A domain message header carried a command other than send-message or close-virtual-handle.
    #[error("unknown domain command {0}")]
    UnknownDomainCommand(u8),
    /// The counts and sizes in the headers contradict each other or the buffer layout.
    #[error("inconsistent message: {0}")]
    Inconsistent(&'static str),
    /// A value does not fit the bit-field it must be encoded into.
    #[error("{field} value {value:#x} does not fit its field")]
    FieldOutOfRange {
        /// Name of the field.
        field: &'static str,
        /// The offending value.
        value: u64,
    },
    /// A TIPC command id whose command type (`16 + id`) would not fit the 16-bit type field.
    #[error("TIPC command id {0} out of range")]
    TipcCommandOutOfRange(u16),
    /// The built reply does not fit the buffer it was asked to be written into.
    #[error("reply needs {needed} bytes but the buffer holds {capacity}")]
    DoesNotFit {
        /// Bytes the reply needs.
        needed: usize,
        /// Usable capacity of the buffer.
        capacity: usize,
    },
    /// Domain object ids were added to a reply that is not for a domain session.
    #[error("object ids can only be sent in a domain reply")]
    ObjectIdsOutsideDomain,
}
