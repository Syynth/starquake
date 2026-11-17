//! A size-capped cache of whole decompressed files, evicted least-recently-used.
//!
//! A zstd file of a container is one frame with no random access, so a read of part of it
//! decompresses it whole (decision log, 2026-11-16, "RomFS from a NUP: positional reads and a
//! capped decompression cache"). [`DecompressedCache`] keeps the decompressed bytes so that the
//! next read of the same file is a copy, and drops the least recently used files when a new one
//! would take it past its cap; a file that was dropped is decompressed again on its next read.
//!
//! The cap bounds the bytes it holds, counted as the files' sizes:
//!
//! - A file larger than the cap is [`CacheError::TooLarge`], before anything is allocated or
//!   decompressed, so a file record claiming a huge size cannot force an allocation.
//! - Room is made **before** the new file is decompressed: least recently used files are
//!   dropped until the ones left and the new one fit within the cap, so the cache's bytes and
//!   the new file's never add up to more than the cap.
//! - Files that fill the cap exactly all stay: eviction starts one byte past it.
//!
//! "Used" is a lookup that found the file or the fill that added it. Pinned by the tests in
//! `nup/cache/tests.rs`.

use std::collections::BTreeMap;

use super::FileId;

/// Why the cache could not hand out a file's bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CacheError<E> {
    /// The file is larger than the whole cache.
    TooLarge {
        /// The file's size.
        size: u64,
        /// The cache's cap.
        cap: u64,
    },
    /// Filling it failed.
    Fill(E),
    /// The fill returned a different number of bytes than the file's size.
    WrongSize {
        /// The file's size.
        size: u64,
        /// The bytes the fill returned.
        got: u64,
    },
}

/// One cached file.
#[derive(Debug)]
struct Cached {
    bytes: Vec<u8>,
    /// When it was last used: a value of [`DecompressedCache::clock`].
    used: u64,
}

/// Whole files' bytes, up to a cap. See the module docs.
#[derive(Debug)]
pub struct DecompressedCache {
    cap: u64,
    held: u64,
    clock: u64,
    files: BTreeMap<FileId, Cached>,
    /// The files by when they were last used, oldest first.
    by_use: BTreeMap<u64, FileId>,
}

impl DecompressedCache {
    /// An empty cache that holds at most `cap` bytes.
    pub fn new(cap: u64) -> Self {
        Self {
            cap,
            held: 0,
            clock: 0,
            files: BTreeMap::new(),
            by_use: BTreeMap::new(),
        }
    }

    /// The cap.
    pub fn cap(&self) -> u64 {
        self.cap
    }

    /// The bytes held now.
    pub fn held(&self) -> u64 {
        self.held
    }

    /// Whether `file` is held.
    #[cfg(test)]
    pub fn contains(&self, file: FileId) -> bool {
        self.files.contains_key(&file)
    }

    /// The next use's stamp.
    fn tick(&mut self) -> u64 {
        self.clock = self.clock.saturating_add(1);
        self.clock
    }

    /// `file`'s bytes, `size` of them: from the cache, or from `decompress` after making room.
    /// See the module docs for the order of checks.
    pub fn get_or_fill<E>(
        &mut self,
        file: FileId,
        size: u64,
        decompress: impl FnOnce() -> Result<Vec<u8>, E>,
    ) -> Result<&[u8], CacheError<E>> {
        let now = self.tick();
        if let Some(cached) = self.files.get_mut(&file) {
            self.by_use.remove(&cached.used);
            cached.used = now;
            self.by_use.insert(now, file);
        } else {
            if size > self.cap {
                return Err(CacheError::TooLarge {
                    size,
                    cap: self.cap,
                });
            }
            // Room first: `held + size <= cap` before the new bytes exist.
            while self.held > self.cap.saturating_sub(size) {
                let Some((_, oldest)) = self.by_use.pop_first() else {
                    break;
                };
                if let Some(dropped) = self.files.remove(&oldest) {
                    self.held = self.held.saturating_sub(len_of(&dropped.bytes));
                }
            }
            let bytes = decompress().map_err(CacheError::Fill)?;
            let got = len_of(&bytes);
            if got != size {
                return Err(CacheError::WrongSize { size, got });
            }
            self.held = self.held.saturating_add(got);
            self.by_use.insert(now, file);
            self.files.insert(file, Cached { bytes, used: now });
        }
        Ok(self
            .files
            .get(&file)
            .map_or_default(|cached| cached.bytes.as_slice()))
    }
}

/// A buffer's length in bytes.
fn len_of(bytes: &[u8]) -> u64 {
    super::to_u64(bytes.len())
}

#[cfg(test)]
mod tests;
