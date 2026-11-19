//! The content hash of a captured program (spec, "Identity, caching and invalidation", "Hash").
//!
//! The function is XXH3's 128-bit variant (`twox-hash`'s `XxHash3_128`, pure Rust): fast, not
//! cryptographic, and defined by the xxHash specification byte for byte, independent of the
//! host's word size and byte order, so the value is stable across runs and platforms as the
//! disk cache needs. The seed is the header's length in bytes (0 or 128): a compute program and
//! a graphics program whose bytes happen to be equal decode differently (the graphics one skips
//! its header), so they must not share a program entry. That seed is this slice's choice; the
//! spec leaves the exact function to the slice.

use std::fmt::{self, Display, Formatter};

use twox_hash::XxHash3_128;

/// A program's content hash: 128 bits over its header and every captured instruction byte.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ContentHash(u128);

impl ContentHash {
    /// The hash with this value, as a stored entry carries it.
    pub const fn new(value: u128) -> Self {
        Self(value)
    }

    /// The hash's value.
    pub const fn value(self) -> u128 {
        self.0
    }

    /// The hash of `bytes`, a program whose first `header_len` bytes are its header.
    pub(crate) fn of(header_len: usize, bytes: &[u8]) -> Self {
        // Cannot fail: a header is 0 or 128 bytes.
        let seed = u64::try_from(header_len).unwrap_or(u64::MAX);
        Self(XxHash3_128::oneshot_with_seed(seed, bytes))
    }
}

impl Display for ContentHash {
    /// Thirty-two hex digits.
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{:032x}", self.0)
    }
}
