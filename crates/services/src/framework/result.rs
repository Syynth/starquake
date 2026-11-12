//! What a service method returns, and the result codes the framework itself replies with.
//!
//! A method returns an [`Outcome`]: `Ok` once it has filled in its [`Reply`](super::Reply), or a
//! [`Fault`]. A fault becomes the reply's result code ([`Fault::result`]) and nothing else of the
//! reply is sent.
//!
//! # Result codes
//!
//! The module numbers are Eden's `ErrorModule::CMIF` and `ErrorModule::HIPC`
//! (`eden/src/core/hle/result.h`), and [`NOT_SUPPORTED`] is Eden's `IPC::ResultNotSupported`
//! (`eden/src/core/hle/service/ipc_helpers.h`).
//!
//! **Every other description number here is a guess.** Eden defines none of them (what it
//! does in the cases they answer is listed in the [module docs](super), in
//! [`control`](super::control) and in [`request`](super::request)), and libnx, a client, has no
//! use for them. The numbers are the ones Atmosphère's `sf` results
//! list under these names, recalled from memory; Atmosphère is not a pinned reference
//! (`docs/references.md`), so nothing has checked them. Pinning Atmosphère's `sf_results.hpp` and
//! `hipc_results.hpp`, or a trace from a console, would confirm or correct them. No starquake
//! test depends on a console agreeing: the tests compare replies with these constants.

use starquake_kernel::result::{KernelError, MODULE_BITS, ResultCode};

/// `ErrorModule::CMIF` (`eden/src/core/hle/result.h`).
pub const CMIF_MODULE: u32 = 10;
/// `ErrorModule::HIPC` (`eden/src/core/hle/result.h`).
pub const HIPC_MODULE: u32 = 11;

/// Packs a module and a description (the bit layout is the kernel crate's, from Eden's
/// `result.h`). Every description below fits its field, which a test checks.
const fn code(module: u32, description: u32) -> ResultCode {
    ResultCode::from_raw(module | (description << MODULE_BITS))
}

/// The request is one the framework does not serve. Eden's `IPC::ResultNotSupported`
/// (`ErrorModule::HIPC`, 1; `eden/src/core/hle/service/ipc_helpers.h`). Where the framework
/// replies with it is starquake's own choice, listed in the [module docs](super).
pub const NOT_SUPPORTED: ResultCode = code(HIPC_MODULE, 1);
/// The interface has no command with the request's id (strict policy). Guess: CMIF 221.
pub const UNKNOWN_COMMAND_ID: ResultCode = code(CMIF_MODULE, 221);
/// A domain request names an object id that is not in the domain. Guess: CMIF 261.
pub const TARGET_NOT_FOUND: ResultCode = code(CMIF_MODULE, 261);
/// The domain has no room for the objects a reply returns. Guess: CMIF 301.
pub const OUT_OF_DOMAIN_ENTRIES: ResultCode = code(CMIF_MODULE, 301);
/// The request lacks something the command reads: payload bytes, a buffer, a handle or an input
/// object. Guess: HIPC 420.
pub const INVALID_CMIF_REQUEST: ResultCode = code(HIPC_MODULE, 420);
/// A control request that needs a domain arrived on a session that is not one. Guess: HIPC 491.
pub const TARGET_NOT_DOMAIN: ResultCode = code(HIPC_MODULE, 491);
/// A control request names an object id that is not in the session's domain. Guess: HIPC 492.
pub const DOMAIN_OBJECT_NOT_FOUND: ResultCode = code(HIPC_MODULE, 492);

/// How a service method finished.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Completion {
    /// The reply is complete and is sent now.
    Done,
}

/// What a service method returns. `reply.ok()` builds the success value; `?` on any framework
/// call, or `Err(code.into())` with the service's own result code, builds the failure.
pub type Outcome = Result<Completion, Fault>;

/// Which way a buffer carries data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    /// From the caller to the service: an A or X descriptor.
    In,
    /// From the service to the caller: a B or C descriptor.
    Out,
}

/// Which handle list of a message.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HandleKind {
    /// The handles the caller copies to the service.
    Copy,
    /// The handles the caller moves to the service.
    Move,
}

/// Why a service method could not answer its request.
///
/// The first seven variants mean the request does not hold what the command reads; they all
/// reply [`INVALID_CMIF_REQUEST`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum Fault {
    /// A payload read went past the end of the payload.
    #[error("payload read of {needed} bytes at offset {offset} passes its end ({len} bytes)")]
    PayloadTruncated {
        /// Byte offset of the read.
        offset: usize,
        /// Bytes the read needed.
        needed: usize,
        /// Length of the payload.
        len: usize,
    },
    /// The request has no buffer with this index in this direction.
    #[error("the request has no {direction:?} buffer {index}")]
    NoBuffer {
        /// Which way the buffer was meant to carry data.
        direction: Direction,
        /// The index asked for.
        index: usize,
    },
    /// A read or write went past the end of a buffer.
    #[error("buffer access of {needed} bytes at offset {offset} passes its end ({len} bytes)")]
    BufferOverrun {
        /// Byte offset of the access inside the buffer.
        offset: u64,
        /// Bytes the access needed.
        needed: usize,
        /// Length of the buffer.
        len: u64,
    },
    /// An in buffer holds more than the service accepts.
    #[error("in buffer holds {len} bytes, more than the {limit} the command accepts")]
    BufferTooLarge {
        /// Bytes left in the buffer.
        len: u64,
        /// The most the command takes.
        limit: usize,
    },
    /// The request has no handle with this index in this list.
    #[error("the request has no {kind:?} handle {index}")]
    NoHandle {
        /// The handle list asked.
        kind: HandleKind,
        /// The index asked for.
        index: usize,
    },
    /// The request has no input object with this index, or the id there names no object.
    #[error("the request has no input object {index}")]
    NoObject {
        /// The index asked for.
        index: usize,
    },
    /// The input object is a different interface from the one the command takes, or it is the
    /// object the command was sent to.
    #[error("input object {index} is not the interface the command takes")]
    WrongObject {
        /// The index asked for.
        index: usize,
    },
    /// A kernel call failed: an invalid handle, unmapped guest memory, a full handle table.
    /// Replies the kernel error's own code.
    #[error("kernel: {0}")]
    Kernel(#[from] KernelError),
    /// The service's own failure. Replies this code.
    #[error("service result {:#x}", .0.raw())]
    Result(ResultCode),
}

impl Fault {
    /// The result code the caller sees.
    #[must_use]
    pub fn result(&self) -> ResultCode {
        match self {
            Self::PayloadTruncated { .. }
            | Self::NoBuffer { .. }
            | Self::BufferOverrun { .. }
            | Self::BufferTooLarge { .. }
            | Self::NoHandle { .. }
            | Self::NoObject { .. }
            | Self::WrongObject { .. } => INVALID_CMIF_REQUEST,
            Self::Kernel(error) => error.code(),
            Self::Result(code) => *code,
        }
    }
}

impl From<ResultCode> for Fault {
    fn from(code: ResultCode) -> Self {
        Self::Result(code)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FRAMEWORK_CODES: [(ResultCode, u32); 7] = [
        (NOT_SUPPORTED, HIPC_MODULE),
        (UNKNOWN_COMMAND_ID, CMIF_MODULE),
        (TARGET_NOT_FOUND, CMIF_MODULE),
        (OUT_OF_DOMAIN_ENTRIES, CMIF_MODULE),
        (INVALID_CMIF_REQUEST, HIPC_MODULE),
        (TARGET_NOT_DOMAIN, HIPC_MODULE),
        (DOMAIN_OBJECT_NOT_FOUND, HIPC_MODULE),
    ];

    #[test]
    fn every_framework_code_is_a_well_formed_failure_in_its_module() {
        for (code, module) in FRAMEWORK_CODES {
            assert!(!code.is_success(), "{code:?}");
            assert_eq!(code.module(), module, "{code:?}");
            // Nothing spilled out of the description field: rebuilding from the two fields gives
            // the same word.
            assert_eq!(ResultCode::new(module, code.description()), Some(code));
        }
    }

    #[test]
    fn framework_codes_are_distinct() {
        for (i, (a, _)) in FRAMEWORK_CODES.iter().enumerate() {
            for (b, _) in FRAMEWORK_CODES.iter().skip(i + 1) {
                assert_ne!(a, b);
            }
        }
    }

    #[test]
    fn not_supported_is_edens_word() {
        // Golden word: ipc_helpers.h `ResultNotSupported{ErrorModule::HIPC, 1}` with result.h's
        // `HIPC = 11`, module in bits 0..9 and description from bit 9. Horizon prints it 2011-0001.
        assert_eq!(NOT_SUPPORTED.raw(), 0x20B);
    }

    #[test]
    fn a_fault_replies_the_code_its_docs_name() {
        let malformed = [
            Fault::PayloadTruncated {
                offset: 1,
                needed: 2,
                len: 3,
            },
            Fault::NoBuffer {
                direction: Direction::In,
                index: 0,
            },
            Fault::BufferOverrun {
                offset: 1,
                needed: 2,
                len: 3,
            },
            Fault::BufferTooLarge { len: 2, limit: 1 },
            Fault::NoHandle {
                kind: HandleKind::Move,
                index: 0,
            },
            Fault::NoObject { index: 0 },
            Fault::WrongObject { index: 0 },
        ];
        for fault in malformed {
            assert_eq!(fault.result(), INVALID_CMIF_REQUEST, "{fault}");
        }
        assert_eq!(
            Fault::from(KernelError::InvalidHandle).result(),
            KernelError::InvalidHandle.code()
        );
        assert_eq!(
            Fault::from(TARGET_NOT_FOUND),
            Fault::Result(TARGET_NOT_FOUND)
        );
        assert_eq!(Fault::from(TARGET_NOT_FOUND).result(), TARGET_NOT_FOUND);
    }

    #[test]
    fn fault_messages_are_stable() {
        assert_eq!(
            Fault::PayloadTruncated {
                offset: 4,
                needed: 8,
                len: 6
            }
            .to_string(),
            "payload read of 8 bytes at offset 4 passes its end (6 bytes)"
        );
        assert_eq!(
            Fault::NoBuffer {
                direction: Direction::Out,
                index: 2
            }
            .to_string(),
            "the request has no Out buffer 2"
        );
        assert_eq!(
            Fault::Result(NOT_SUPPORTED).to_string(),
            format!("service result {:#x}", NOT_SUPPORTED.raw())
        );
    }
}
