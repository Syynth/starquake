//! The list of pipeline keys seen, as a versioned file's bytes, so that the next boot can
//! precompile them (slice G23; render-backend, "Compiling without stalling a frame": "the gpu
//! side's list of pipeline keys seen ... versioned and refused on mismatch").
//!
//! This module owns the bytes and does no file I/O: `core` owns the folder and the file, which
//! is derived from a title and so lives in the user's data folder, never in the repository
//! (`docs/specs/observability.md`, "Storage and what may be committed").
//!
//! # Layout (version 1)
//!
//! Everything is `postcard`: unsigned integers are little-endian base-128 varints, an enum is
//! its variant's index as a varint and then its fields, an `Option` is `0x00` or `0x01` and then
//! the value, a fixed-size array is its elements with no count, and a `Vec` is its count and then
//! its elements.
//!
//! | Bytes | What |
//! |---|---|
//! | 4 | [`MAGIC`], `SQPK` |
//! | varint | [`VERSION`] (`u32`) |
//! | varint | the number of keys (`u32`), at most [`MAX_PIPELINES`] |
//! | per key | `0x00` and a graphics key, or `0x01` and a compute key's shader stamp (`u128`) |
//!
//! A graphics key is its [`StageStamps`](super::StageStamps) (the vertex stamp, then an optional
//! stamp per tessellation control, tessellation evaluation, geometry and fragment), then the
//! fields of `FixedState` in `key.rs`, in declaration order: the dynamic-state mask, the vertex
//! input, the topology, primitive restart, cull mode, front face, polygon mode, depth clamp, the
//! depth bias's three `f32` bit patterns (when not dynamic), the line width's bit pattern (when
//! not dynamic), the depth test, depth write, the stencil test (its two faces and, when not
//! dynamic, its compare and write masks), the eight blend states, the logic operation, the
//! target formats and the emulation flags. Each render type inside is written as its own
//! `serde` derive has it, so a change to the field or variant order of one of those types (in
//! `starquake-render`) changes this file too. The bytes are pinned by
//! `tests::list::golden_bytes_of_every_key_kind`; any change to them needs a new [`VERSION`] in the
//! same change.
//!
//! # What reading checks
//!
//! In this order: the magic ([`KeyListError::BadMagic`]); the version
//! ([`KeyListError::UnsupportedVersion`], whatever follows); the number of keys
//! ([`KeyListError::TooManyKeys`], before any key is decoded); then each key in turn, decoded
//! ([`KeyListError::Truncated`] where the bytes run out, [`KeyListError::Decode`] for anything
//! else) and checked as the parent module's `key.rs` docs say ([`KeyListError::InvalidKey`]);
//! then that nothing follows the last key ([`KeyListError::TrailingBytes`]); then that no key
//! repeats an earlier one ([`KeyListError::DuplicateKey`], naming the first repeat). Writing
//! applies the same key-count, key and duplicate checks, in the same order, so a list that was
//! written can be read.
//!
//! A file too short to hold the magic is truncated when it is the start of the magic, and
//! otherwise not a key list. Reading accepts what `postcard` accepts, which includes numbers
//! written with more bytes than they need; [`KeyList::to_bytes`] always writes the shortest.
//!
//! # Memory
//!
//! At most [`MAX_PIPELINES`] keys are kept, each checked as it is decoded, so a kept key has at
//! most the guest's 32 vertex buffers and 32 attributes. The key being decoded can be as large
//! as what is left of the file before the check refuses it; a caller that reads a file it does
//! not trust bounds its size first.

use super::MAX_PIPELINES;
use super::key::{KeyFault, PipelineKey, check_graphics};
use serde::Deserialize;
use std::collections::HashSet;
use thiserror::Error;

/// First bytes of every key list, so that a file of another kind is refused before its body is
/// decoded.
pub const MAGIC: [u8; 4] = *b"SQPK";

/// Layout version written into every key list and required when reading one.
pub const VERSION: u32 = 1;

/// Why a key list could not be written or read.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum KeyListError {
    /// Encoding failed. No list reaches this today: `postcard` refuses to write only a sequence
    /// whose length is not known when it starts, which nothing in a key is, and a buffer that is
    /// full, which a `Vec` never is. The variant is here so that this module does not unwrap.
    #[error("could not encode the key list: {0}")]
    Encode(postcard::Error),
    /// The bytes end before the list does.
    #[error("the key list ends early")]
    Truncated,
    /// The bytes are not a well-formed encoding (a field out of range).
    #[error("could not decode the key list: {0}")]
    Decode(postcard::Error),
    /// The bytes do not start with [`MAGIC`].
    #[error("not a pipeline key list (bad magic)")]
    BadMagic,
    /// The list was written with a different layout version.
    #[error(
        "pipeline key list layout version {found} is not supported (this build reads {VERSION})"
    )]
    UnsupportedVersion {
        /// The version found in the list.
        found: u32,
    },
    /// The list has more keys than [`MAX_PIPELINES`].
    #[error("the key list has {count} keys, more than the {MAX_PIPELINES} allowed")]
    TooManyKeys {
        /// The number of keys it has or says it has.
        count: u64,
    },
    /// A key fails the checks every key passes.
    #[error("key {index} of the list is refused: {fault}")]
    InvalidKey {
        /// The key's place in the list, from zero.
        index: usize,
        /// What is wrong with it.
        fault: KeyFault,
    },
    /// Bytes remain after the last key.
    #[error("{count} trailing bytes after the key list")]
    TrailingBytes {
        /// How many bytes were left over.
        count: usize,
    },
    /// A key repeats an earlier one.
    #[error("key {index} of the list repeats an earlier key")]
    DuplicateKey {
        /// The repeat's place in the list, from zero.
        index: usize,
    },
}

/// The pipeline keys a [`super::PipelineCache`] has seen, in the order it first recorded them,
/// each checked (module docs). Made by [`super::PipelineCache::key_list`] or read from a file's
/// bytes, and handed to [`super::PipelineCache::precompile`] on the next boot.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct KeyList {
    pub(super) keys: Vec<PipelineKey>,
}

impl KeyList {
    /// How many keys the list has.
    pub fn len(&self) -> usize {
        self.keys.len()
    }

    /// Whether the list has no keys.
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    /// The list as a file's bytes (module docs, "Layout").
    ///
    /// Errors, in this order: [`KeyListError::TooManyKeys`], [`KeyListError::InvalidKey`] for
    /// the first key that fails its checks, and [`KeyListError::DuplicateKey`]: what reading
    /// refuses.
    pub fn to_bytes(&self) -> Result<Vec<u8>, KeyListError> {
        let count = key_count(self.keys.len())?;
        for (index, key) in self.keys.iter().enumerate() {
            check_key(index, key)?;
        }
        check_duplicates(&self.keys)?;
        let mut bytes =
            postcard::to_allocvec(&(MAGIC, VERSION, count)).map_err(KeyListError::Encode)?;
        for key in &self.keys {
            bytes = postcard::to_extend(key, bytes).map_err(KeyListError::Encode)?;
        }
        Ok(bytes)
    }

    /// Reads a key list from a file's bytes. Malformed, truncated, foreign, other-version,
    /// over-long or failing input is an `Err` (module docs, "What reading checks").
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, KeyListError> {
        let Some((magic, rest)) = bytes.split_first_chunk() else {
            return Err(if MAGIC.starts_with(bytes) {
                KeyListError::Truncated
            } else {
                KeyListError::BadMagic
            });
        };
        if *magic != MAGIC {
            return Err(KeyListError::BadMagic);
        }
        let (version, rest) = take::<u32>(rest)?;
        if version != VERSION {
            return Err(KeyListError::UnsupportedVersion { found: version });
        }
        let (count, mut rest) = take::<u32>(rest)?;
        let count = usize::try_from(count)
            .ok()
            .filter(|&n| n <= MAX_PIPELINES)
            .ok_or(KeyListError::TooManyKeys {
                count: count.into(),
            })?;
        // The count is at most `MAX_PIPELINES`, so the room taken is bounded however few bytes
        // follow; each key also takes at least one byte, so a count the bytes cannot hold ends
        // in `Truncated`.
        let mut keys = Vec::with_capacity(count);
        for index in 0..count {
            let (key, after) = take::<PipelineKey>(rest)?;
            check_key(index, &key)?;
            keys.push(key);
            rest = after;
        }
        if !rest.is_empty() {
            return Err(KeyListError::TrailingBytes { count: rest.len() });
        }
        check_duplicates(&keys)?;
        Ok(Self { keys })
    }
}

/// The number of keys as the header writes it, or [`KeyListError::TooManyKeys`] past
/// [`MAX_PIPELINES`].
fn key_count(count: usize) -> Result<u32, KeyListError> {
    u32::try_from(count)
        .ok()
        .filter(|_| count <= MAX_PIPELINES)
        .ok_or(KeyListError::TooManyKeys {
            count: u64::try_from(count).unwrap_or(u64::MAX),
        })
}

/// Checks one key, as writing and reading both do.
fn check_key(index: usize, key: &PipelineKey) -> Result<(), KeyListError> {
    match key {
        PipelineKey::Graphics(key) => {
            check_graphics(key).map_err(|fault| KeyListError::InvalidKey { index, fault })
        }
        PipelineKey::Compute(_) => Ok(()),
    }
}

/// [`KeyListError::DuplicateKey`] for the first key that repeats an earlier one.
fn check_duplicates(keys: &[PipelineKey]) -> Result<(), KeyListError> {
    let mut seen = HashSet::with_capacity(keys.len());
    match keys.iter().position(|key| !seen.insert(key)) {
        Some(index) => Err(KeyListError::DuplicateKey { index }),
        None => Ok(()),
    }
}

/// Reads one value from the front of `bytes` and returns the bytes after it.
fn take<'a, T: Deserialize<'a>>(bytes: &'a [u8]) -> Result<(T, &'a [u8]), KeyListError> {
    postcard::take_from_bytes(bytes).map_err(|error| match error {
        postcard::Error::DeserializeUnexpectedEnd => KeyListError::Truncated,
        other => KeyListError::Decode(other),
    })
}
