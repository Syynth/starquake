//! The shader-entry key (layer 3) and its digest, the [`ShaderStamp`] (spec, "Identity, caching
//! and invalidation", layer 3; "Compilation, threads and stutter": "A shader's identity across
//! boots is a `ShaderStamp`, a digest of its entry key"; owner's ruling 4 of 2026-11-19).
//!
//! # Provisional
//!
//! Slice 21 defines the stamp's persistent form with the disk cache's entry encoding. Until
//! then the stamp is XXH3-128 over the key fed through the key types' derived `Hash`, with
//! every integer written little-endian at a fixed width ([`KeyWriter`]). So, within one build,
//! the stamp is the same on every run and every host for the same key. Across builds it
//! changes whenever one of the key's types gains, loses or reorders a field, and integer
//! arrays and slices reach the writer in the host's byte order (the standard library's
//! `hash_slice` writes their memory): the same stamp on the two supported host families, which
//! are both little-endian, but not on a big-endian host. Slice 21 replaces this with a
//! versioned encoding; the 128-bit width stays a guess (ruling 4).
//!
//! A stamp collision is treated as impossible, as a content-hash collision is (spec, "Hash"):
//! the shader layer is keyed by stamp.

use std::fmt::{self, Display, Formatter};
use std::hash::{Hash, Hasher};

use twox_hash::XxHash3_128;

use super::HostTarget;
use crate::capture::ContentHash;
use crate::exchange::Answers;
use crate::translate::GuestStage;
use crate::variant::VariantKey;

/// A shader entry's identity: the digest of its key (module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ShaderStamp(pub(super) u128);

impl ShaderStamp {
    /// The stamp's value.
    pub const fn value(self) -> u128 {
        self.0
    }
}

impl Display for ShaderStamp {
    /// Thirty-two hex digits.
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{:032x}", self.0)
    }
}

/// The key of a shader entry (spec, layer 3: "content hash, answers, variant key, host target,
/// translator version"), with the stage the program runs as (module docs of the parent).
#[derive(Hash)]
pub(super) struct ShaderKey<'a> {
    /// The translator version.
    pub(super) version: u32,
    /// The host target.
    pub(super) target: &'a HostTarget,
    /// The stage.
    pub(super) stage: GuestStage,
    /// The program's content hash.
    pub(super) hash: ContentHash,
    /// The answers its questions got.
    pub(super) answers: &'a Answers,
    /// The variant key.
    pub(super) variant: &'a VariantKey,
}

impl ShaderKey<'_> {
    /// The key's stamp.
    pub(super) fn stamp(&self) -> ShaderStamp {
        let mut writer = KeyWriter::default();
        self.hash(&mut writer);
        ShaderStamp(XxHash3_128::oneshot(&writer.bytes))
    }
}

/// Collects what a key's `Hash` writes, every integer little-endian at a fixed width (`usize`
/// and `isize` as eight bytes), for the stamp's digest (module docs).
#[derive(Debug, Default)]
pub(super) struct KeyWriter {
    pub(super) bytes: Vec<u8>,
}

impl Hasher for KeyWriter {
    /// The low 64 bits of the digest of what was written so far.
    fn finish(&self) -> u64 {
        let digest = XxHash3_128::oneshot(&self.bytes);
        // Cannot fail: the value is masked to 64 bits.
        u64::try_from(digest & u128::from(u64::MAX)).unwrap_or(u64::MAX)
    }

    fn write(&mut self, bytes: &[u8]) {
        self.bytes.extend_from_slice(bytes);
    }

    fn write_u8(&mut self, value: u8) {
        self.bytes.push(value);
    }

    fn write_u16(&mut self, value: u16) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn write_u32(&mut self, value: u32) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn write_u64(&mut self, value: u64) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn write_u128(&mut self, value: u128) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn write_usize(&mut self, value: usize) {
        self.write_u64(u64::try_from(value).unwrap_or(u64::MAX));
    }

    fn write_i8(&mut self, value: i8) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn write_i16(&mut self, value: i16) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn write_i32(&mut self, value: i32) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn write_i64(&mut self, value: i64) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn write_i128(&mut self, value: i128) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn write_isize(&mut self, value: isize) {
        self.write_i64(i64::try_from(value).unwrap_or(i64::MAX));
    }
}
