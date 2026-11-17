//! A package's `RomFS` directory served as a `RomFS` image: the data storage a title opens with
//! `OpenDataStorageByCurrentProcess` and reads at any offset.
//!
//! A NUP stores a `RomFS` as a directory tree of files (SPEC §3.1.3), while a title's
//! file-system library reads a `RomFS` image. [`RomFsStorage`] builds that image's header and
//! tables from the tree with the loader's port of Eden's builder ([`crate::romfs::build`], whose
//! docs give the layout), the way Eden-NX2 serves a loose `RomFS` directory, and answers
//! positional reads ([`RomFsStorage::read_at`], through `&self`) by copying the header and
//! tables and reading each file's data from the container where it lies:
//!
//! - **A raw file** is read straight from its extent ([`Nup::read_raw_at`]). Its bytes are not
//!   checked against its SHA-256 (a read of part of a file cannot check a hash of all of it);
//!   `starquake verify` checks every file.
//! - **A zstd file** is one frame with no random access, so its first read decompresses it whole
//!   into a [`DecompressedCache`] of at most the cap the storage was made with
//!   ([`CACHE_CAP`] for a run), checking its size and SHA-256 as [`Nup::read_to_vec_at`] does;
//!   later reads copy from the cache. The cache drops the least recently used files to make
//!   room, and a file it dropped is decompressed again on its next read (decision log,
//!   2026-11-16, "RomFS from a NUP: positional reads and a capped decompression cache"). A file
//!   larger than the cap is [`StorageError::TooLargeForCache`]: the size is compared with the cap
//!   before anything is allocated.
//!
//! Building the image reads nothing from the container: the tree, names and sizes are in its
//! table of contents. The image's header and tables are held in memory (bounded by
//! `romfs::build::MAX_METADATA_SIZE`).
//!
//! # Failures are sticky per file
//!
//! A read that fails on a file's data (the source failed, the file is damaged, or it is too large
//! for the cache) fails, and every later read that touches that file fails with the same error,
//! never with other bytes. Reads that touch no failed file are unaffected. Pinned by
//! `a_failed_read_is_sticky_for_its_file_and_no_other`.
//!
//! # Locking
//!
//! The cache and the record of failures each sit behind a mutex; neither is held while the
//! other is taken. The cache's lock is held while a file decompresses, so a second read of a file
//! that is being decompressed waits for it rather than decompressing it again. Pinned by
//! `two_threads_reading_one_compressed_file_decompress_it_once`.

use std::collections::BTreeMap;
use std::io;
use std::sync::{Mutex, MutexGuard, PoisonError};

use super::cache::{CacheError, DecompressedCache};
use super::{Codec, DirId, FileId, Node, Nup, NupError};
use crate::read_at::ReadAt;
use crate::romfs::build::{self, BuildError, DirIn, FileIn, Image, Placed};

/// The most bytes of decompressed files a run's `RomFS` keeps in memory: 1 GiB. A starquake
/// limit. It bounds what the cache holds whatever a title asks for, and it is also the largest
/// compressed file a run can serve, so it is set well above the largest compressed file the
/// project has seen in a title's `RomFS` (a guess: no reference bounds it; a title that needs
/// more says so in plain words, [`StorageError::TooLargeForCache`], and would settle it).
pub const CACHE_CAP: u64 = 1 << 30;

/// Why the `RomFS` storage could not be built or read.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum StorageError {
    /// The directory given is not one of the container's.
    #[error("the RomFS directory {0} is not in this container")]
    NoSuchDir(usize),
    /// The tree could not be built into an image.
    #[error("the RomFS could not be laid out as an image: {0}")]
    Build(#[from] BuildError),
    /// A compressed file is larger than the cache that would hold it decompressed.
    #[error(
        "file {file} of the RomFS is {size} bytes decompressed, more than the {cap}-byte \
         cache of decompressed files holds"
    )]
    TooLargeForCache {
        /// The file's index in the container.
        file: usize,
        /// Its size.
        size: u64,
        /// The cache's cap.
        cap: u64,
    },
    /// The container could not be read.
    #[error("file {file} of the RomFS could not be read: {message}")]
    Io {
        /// The file's index in the container.
        file: usize,
        /// The source's error, in words.
        message: String,
    },
    /// A file's stored bytes are not what its record says (a bad frame, the wrong size or the
    /// wrong hash).
    #[error("file {file} of the RomFS is damaged: {message}")]
    Damaged {
        /// The file's index in the container.
        file: usize,
        /// What is wrong, in words.
        message: String,
    },
}

impl StorageError {
    /// `error`, from reading `file`, as a storage error.
    fn from_nup(file: FileId, error: &NupError) -> Self {
        let (file, message) = (file.index(), error.to_string());
        if error.is_io() {
            Self::Io { file, message }
        } else {
            Self::Damaged { file, message }
        }
    }
}

/// A file of the image: which file of the container, and how it is stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Data {
    file: FileId,
    codec: Codec,
}

/// A package's `RomFS` directory as a `RomFS` image. See the module docs.
#[derive(Debug)]
pub struct RomFsStorage<S> {
    nup: Nup<S>,
    image: Image<Data>,
    cache: Mutex<DecompressedCache>,
    failed: Mutex<BTreeMap<FileId, StorageError>>,
}

/// Takes `mutex`'s lock; a poisoned lock is still taken (what it guards is never left half
/// changed: each update is one insert or one field).
fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

impl<S> RomFsStorage<S> {
    /// Builds the image of the tree below `dir` of `nup`, whose zstd files are cached
    /// decompressed up to `cache_cap` bytes. Nothing is read from the container's source.
    pub fn new(nup: Nup<S>, dir: DirId, cache_cap: u64) -> Result<Self, StorageError> {
        if dir.index() >= nup.dir_count() {
            return Err(StorageError::NoSuchDir(dir.index()));
        }
        let image = layout(&nup, dir)?;
        Ok(Self {
            nup,
            image,
            cache: Mutex::new(DecompressedCache::new(cache_cap)),
            failed: Mutex::new(BTreeMap::new()),
        })
    }

    /// The image's size in bytes.
    pub fn size(&self) -> u64 {
        self.image.size()
    }

    /// The container.
    pub fn nup(&self) -> &Nup<S> {
        &self.nup
    }

    /// The bytes of decompressed files held now.
    pub fn cached_bytes(&self) -> u64 {
        lock(&self.cache).held()
    }
}

impl<S: ReadAt> RomFsStorage<S> {
    /// Reads up to `buf.len()` bytes of the image from `offset` and returns how many: fewer than
    /// `buf.len()` only at the end of the image, and zero at or past it. See the module docs for
    /// how file data is read and when a read fails.
    pub fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<usize, StorageError> {
        self.image.read_with(offset, buf, |placed, into, out| {
            self.read_file(placed, into, out)
        })
    }

    /// Fills `out` from `into` bytes into `placed`'s data.
    fn read_file(
        &self,
        placed: &Placed<Data>,
        into: u64,
        out: &mut [u8],
    ) -> Result<(), StorageError> {
        let Data { file, codec } = placed.key;
        if let Some(error) = lock(&self.failed).get(&file) {
            return Err(error.clone());
        }
        let result = match codec {
            Codec::Raw => self.read_raw(file, into, out),
            Codec::Zstd => self.read_cached(file, placed.size, into, out),
        };
        if let Err(error) = &result {
            lock(&self.failed).insert(file, error.clone());
        }
        result
    }

    /// A raw file's bytes, from its extent.
    fn read_raw(&self, file: FileId, into: u64, out: &mut [u8]) -> Result<(), StorageError> {
        let count = self
            .nup
            .read_raw_at(file, into, out)
            .map_err(|error| StorageError::from_nup(file, &error))?;
        if count == out.len() {
            Ok(())
        } else {
            // The image placed `out` inside the file's size, so a short read is a source that
            // ended early.
            let error = NupError::Io(io::Error::from(io::ErrorKind::UnexpectedEof));
            Err(StorageError::from_nup(file, &error))
        }
    }

    /// A compressed file's bytes, from the cache, decompressing it first if it is not there.
    fn read_cached(
        &self,
        file: FileId,
        size: u64,
        into: u64,
        out: &mut [u8],
    ) -> Result<(), StorageError> {
        let mut cache = lock(&self.cache);
        let cap = cache.cap();
        let bytes = cache
            .get_or_fill(file, size, || self.nup.read_to_vec_at(file, cap))
            .map_err(|error| match error {
                CacheError::TooLarge { size, cap } => StorageError::TooLargeForCache {
                    file: file.index(),
                    size,
                    cap,
                },
                CacheError::Fill(error) => StorageError::from_nup(file, &error),
                CacheError::WrongSize { size, got } => StorageError::Damaged {
                    file: file.index(),
                    message: format!("it decompressed to {got} bytes, not its size {size}"),
                },
            })?;
        let part = usize::try_from(into)
            .ok()
            .and_then(|start| bytes.get(start..start.checked_add(out.len())?));
        match part {
            Some(part) => {
                out.copy_from_slice(part);
                Ok(())
            }
            // The image placed `out` inside the file's size, and the cache holds exactly that.
            None => Err(StorageError::Damaged {
                file: file.index(),
                message: "a read fell outside its decompressed bytes".to_owned(),
            }),
        }
    }
}

/// The storage as a positional source, so that it can stand behind anything that reads one. A
/// [`StorageError`] reaches the caller as an `io::Error` of kind `InvalidData` that carries it.
impl<S: ReadAt> ReadAt for RomFsStorage<S> {
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> io::Result<usize> {
        Self::read_at(self, offset, buf)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))
    }

    fn size(&self) -> io::Result<u64> {
        Ok(Self::size(self))
    }
}

/// The image of the tree below `root`: every directory and file reached from it, a directory's
/// parent before it. The walk keeps a visited set, so it ends whatever the tree (`Nup::open`
/// already makes it a forest).
fn layout<S>(nup: &Nup<S>, root: DirId) -> Result<Image<Data>, StorageError> {
    let mut visited = vec![false; nup.dir_count()];
    let mut dirs = vec![DirIn {
        parent: 0,
        name: &[],
    }];
    let mut files = Vec::new();
    let mut queue = vec![(root, 0usize)];
    if let Some(seen) = visited.get_mut(root.index()) {
        *seen = true;
    }
    let mut next = 0;
    while let Some(&(dir, index)) = queue.get(next) {
        next += 1;
        for &child in nup.children(dir) {
            let name = nup.name(child);
            match child {
                Node::Dir(sub) => {
                    let Some(seen) = visited.get_mut(sub.index()) else {
                        continue;
                    };
                    if std::mem::replace(seen, true) {
                        continue;
                    }
                    queue.push((sub, dirs.len()));
                    dirs.push(DirIn {
                        parent: index,
                        name,
                    });
                }
                Node::File(file) => {
                    let Some(entry) = nup.file(file) else {
                        continue;
                    };
                    files.push(FileIn {
                        parent: index,
                        name,
                        size: entry.size,
                        key: Data {
                            file,
                            codec: entry.codec,
                        },
                    });
                }
            }
        }
    }
    Ok(build::build(&dirs, &files)?)
}

#[cfg(test)]
mod tests;
