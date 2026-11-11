//! `RomFS` images: header, directory and file tables, and the path hash tables.
//!
//! Read-only: [`RomFs::new`] parses the 0x50-byte header and loads the four tables it points at
//! (directory and file metadata, directory and file hash buckets) from any `Read + Seek` source.
//! File data is never loaded up front: [`RomFs::read_file_at`] pulls windows on demand, so a
//! multi-gigabyte image costs only its metadata. Each table is checked against the source length
//! before it is allocated.
//!
//! Ported from `eden/src/core/file_sys/romfs.cpp` @ cf654052 (GPL-3.0-or-later): `TableLocation`,
//! `RomFSHeader`, `DirectoryEntry`, `FileEntry`, `GetEntry` and `ExtractRomFS`. The hash tables,
//! which `ExtractRomFS` never reads, follow the builder in
//! `eden/src/core/file_sys/fsmitm_romfsbuild.cpp` (`romfs_calc_path_hash`, the bucket loops in
//! `RomFSBuildContext::Build`): an entry's `hash` field is the next entry in its bucket's chain,
//! chains are headed by the bucket table, and the bucket is `hash % bucket_count`. The path limit
//! is `EntryNameLengthMax` (`eden/src/core/file_sys/fs_directory.h`).
//!
//! Intentional divergences from Eden, each pinned by a test:
//!
//! - Eden's `GetEntry` returns a zeroed entry for an offset outside the table, whose `sibling` of
//!   0 sends `ProcessFile` and `ProcessDirectory` round the same entry forever; here that is
//!   [`RomFsError::EntryOutOfBounds`]. Sibling and child links that revisit an entry are
//!   [`RomFsError::Cycle`].
//! - Eden clamps a name that runs past the table; here that is [`RomFsError::NameOutOfBounds`].
//! - Eden's names are raw bytes; here a name must be UTF-8 ([`RomFsError::NameNotUtf8`]).
//! - Eden ignores a file whose data lies past the end of the source until it is read; here
//!   [`RomFs::file_extent`] and the reads reject it, checking `data offset + offset + size` for
//!   overflow as well.
//! - Eden asserts that the root directory is named `""`; here a non-empty root name is
//!   [`RomFsError::BadRoot`].
//!
//! Guesses, not backed by a pinned reference source (the builder never produces these, so they
//! cannot be checked against it):
//!
//! - A hash table must be a non-zero multiple of 4 bytes (`romfs_get_hash_table_count` never
//!   returns fewer than 3 buckets); anything else is [`RomFsError::BadHashTable`]. The reader
//!   takes the bucket count from the table size, not from the entry count.
//! - Metadata tables are addressed by `u32`, so a table over `u32::MAX` bytes is rejected.
//! - [`path_hash`] feeds name bytes in as unsigned. Eden's builder XORs `char`, which is signed
//!   on some hosts, so names with bytes of 0x80 or above hash differently there.
//! - Lookup paths must already be normalised: absolute, `/`-separated, with no empty, `.` or `..`
//!   components. Eden's `PathNormalizer` is not ported.
//!
//! Lookups follow the hash chains and check each candidate's `parent` field and name; listings
//! and [`RomFs::walk`] follow the child and sibling links. For a consistent image the two agree.

use std::collections::HashSet;
use std::io::{Read, Seek};
use std::ops::{ControlFlow, Range};

use crate::pfs0::{Le, checked_extent, read_exact_at, read_vec_at, read_window, source_len};

/// Size of the header (`static_assert(sizeof(RomFSHeader) == 0x50)`).
pub const HEADER_SIZE: usize = 0x50;
/// Size of a directory entry without its name (`static_assert(sizeof(DirectoryEntry) == 0x18)`).
pub const DIR_ENTRY_SIZE: usize = 0x18;
/// Size of a file entry without its name (`static_assert(sizeof(FileEntry) == 0x20)`).
pub const FILE_ENTRY_SIZE: usize = 0x20;
/// The "no entry" link value (`ROMFS_ENTRY_EMPTY`).
pub const ENTRY_EMPTY: u32 = 0xFFFF_FFFF;
/// Table offset of the root directory entry (`ProcessDirectory(ctx, 0, ...)`).
pub const ROOT_OFFSET: u32 = 0;
/// Longest path or name in bytes (`EntryNameLengthMax`).
pub const MAX_PATH_LEN: usize = 0x300;
/// Seed of the path hash (`romfs_calc_path_hash`: `parent ^ 123456789`).
pub const HASH_SEED: u32 = 123_456_789;
/// Longest metadata table: entries are addressed by `u32` offsets (a guess; see the module docs).
pub const MAX_TABLE_SIZE: u64 = u32::MAX as u64;

/// The hash of `name` under the directory entry at table offset `parent`
/// (`romfs_calc_path_hash`). The root directory hashes with parent 0 and an empty name.
pub fn path_hash(parent: u32, name: &[u8]) -> u32 {
    let mut hash = parent ^ HASH_SEED;
    for &byte in name {
        hash = hash.rotate_right(5);
        hash ^= u32::from(byte);
    }
    hash
}

/// Where a table sits in the image (`TableLocation`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TableLocation {
    /// Offset from the start of the image.
    pub offset: u64,
    /// Size in bytes.
    pub size: u64,
}

/// The `RomFS` header (`RomFSHeader`, without `header_size`, which must be [`HEADER_SIZE`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RomFsHeader {
    /// Directory hash buckets (`directory_hash`).
    pub dir_hash: TableLocation,
    /// Directory entries (`directory_meta`).
    pub dir_meta: TableLocation,
    /// File hash buckets (`file_hash`).
    pub file_hash: TableLocation,
    /// File entries (`file_meta`).
    pub file_meta: TableLocation,
    /// Offset of the file data area; file entry offsets are relative to it (`data_offset`).
    pub data_offset: u64,
}

/// Which of the four tables an error is about.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Table {
    /// Directory hash buckets.
    DirHash,
    /// Directory entries.
    DirMeta,
    /// File hash buckets.
    FileHash,
    /// File entries.
    FileMeta,
}

/// A directory entry (`DirectoryEntry` plus its name).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DirEntry {
    /// This entry's offset in the directory table; the id links refer to.
    pub id: u32,
    /// The parent directory's id (`parent`; the root's is 0, itself).
    pub parent: u32,
    /// The next directory with the same parent (`sibling`).
    pub sibling: Option<u32>,
    /// The first subdirectory (`child_dir`).
    pub child_dir: Option<u32>,
    /// The first file (`child_file`).
    pub child_file: Option<u32>,
    /// The next directory in this one's hash bucket (the `hash` field).
    pub hash_next: Option<u32>,
    /// The directory's name; empty for the root.
    pub name: String,
}

/// A file entry (`FileEntry` plus its name).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileEntry {
    /// This entry's offset in the file table; the id links refer to.
    pub id: u32,
    /// The parent directory's id (`parent`).
    pub parent: u32,
    /// The next file with the same parent (`sibling`).
    pub sibling: Option<u32>,
    /// Offset of the data from [`RomFsHeader::data_offset`] (`offset`).
    pub offset: u64,
    /// Size of the data in bytes (`size`).
    pub size: u64,
    /// The next file in this one's hash bucket (the `hash` field).
    pub hash_next: Option<u32>,
    /// The file's name.
    pub name: String,
}

/// The direct children of a directory, in sibling order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DirContents {
    /// Subdirectories.
    pub dirs: Vec<DirEntry>,
    /// Files.
    pub files: Vec<FileEntry>,
}

/// Every directory and file of an image with its absolute path (`/` for the root).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RomFsTree {
    /// Directories, parents before children.
    pub dirs: Vec<(String, DirEntry)>,
    /// Files.
    pub files: Vec<(String, FileEntry)>,
}

/// Why a `RomFS` could not be parsed, searched or read.
#[derive(Debug, thiserror::Error)]
pub enum RomFsError {
    /// The source failed to seek or read.
    #[error("RomFS source I/O failed")]
    Io(#[from] std::io::Error),
    /// The source is shorter than the header.
    #[error("RomFS is {len} bytes, shorter than the {HEADER_SIZE}-byte header")]
    Truncated {
        /// Length of the source.
        len: u64,
    },
    /// `header_size` is not [`HEADER_SIZE`].
    #[error("RomFS header size is {found:#x}, not {HEADER_SIZE:#x}")]
    BadHeaderSize {
        /// The value found.
        found: u64,
    },
    /// A table lies (partly) outside the source, or its end overflows.
    #[error(
        "RomFS {table:?} table ({size:#x} bytes at {offset:#x}) exceeds the {len:#x}-byte source"
    )]
    TableOutOfBounds {
        /// The table.
        table: Table,
        /// Declared offset.
        offset: u64,
        /// Declared size.
        size: u64,
        /// Length of the source.
        len: u64,
    },
    /// A table is larger than [`MAX_TABLE_SIZE`].
    #[error("RomFS {table:?} table is {size:#x} bytes, larger than {MAX_TABLE_SIZE:#x}")]
    TableTooLarge {
        /// The table.
        table: Table,
        /// Declared size.
        size: u64,
    },
    /// A hash table is empty or not a multiple of 4 bytes.
    #[error("RomFS {table:?} table is {size:#x} bytes, not a non-zero multiple of 4")]
    BadHashTable {
        /// The table.
        table: Table,
        /// Declared size.
        size: u64,
    },
    /// An entry (or the part of it before its name) does not fit in its table.
    #[error("RomFS {table:?} entry at {id:#x} does not fit in the table")]
    EntryOutOfBounds {
        /// The table.
        table: Table,
        /// The entry's offset.
        id: u32,
    },
    /// An entry's name runs past the end of its table.
    #[error(
        "RomFS {table:?} entry at {id:#x}: a {name_len:#x}-byte name does not fit in the table"
    )]
    NameOutOfBounds {
        /// The table.
        table: Table,
        /// The entry's offset.
        id: u32,
        /// Declared name length.
        name_len: u32,
    },
    /// An entry's name is not valid UTF-8.
    #[error("RomFS {table:?} entry at {id:#x}: name is not valid UTF-8")]
    NameNotUtf8 {
        /// The table.
        table: Table,
        /// The entry's offset.
        id: u32,
    },
    /// The root directory (the entry at offset 0) has a name.
    #[error("RomFS root directory has a non-empty name")]
    BadRoot,
    /// Following links or a hash chain came back to an entry already visited.
    #[error("RomFS {table:?} links loop back to the entry at {id:#x}")]
    Cycle {
        /// The table.
        table: Table,
        /// The entry's offset.
        id: u32,
    },
    /// A path (or the path of an entry being listed) is longer than [`MAX_PATH_LEN`].
    #[error("RomFS path is longer than {MAX_PATH_LEN} bytes")]
    PathTooLong,
    /// A lookup path is not absolute or has an empty, `.` or `..` component.
    #[error("RomFS lookup path {path:?} is not a normalised absolute path")]
    InvalidPath {
        /// The path as given.
        path: String,
    },
    /// The file entry handed to a read does not lie inside this image's source.
    #[error(
        "RomFS file data ({size:#x} bytes at data offset {offset:#x}) exceeds the {len:#x}-byte source"
    )]
    FileOutOfBounds {
        /// The entry's offset field.
        offset: u64,
        /// The entry's size field.
        size: u64,
        /// Length of the source.
        len: u64,
    },
}

/// A parsed `RomFS` over a seekable source.
#[derive(Debug)]
pub struct RomFs<R> {
    reader: R,
    len: u64,
    header: RomFsHeader,
    root: DirEntry,
    dir_meta: Vec<u8>,
    file_meta: Vec<u8>,
    dir_buckets: Vec<u32>,
    file_buckets: Vec<u32>,
}

impl<R: Read + Seek> RomFs<R> {
    /// Parses the image that starts at offset 0 of `reader`.
    ///
    /// Every offset and size is checked against the source length before use, so malformed input
    /// is an error, never a panic or a runaway allocation.
    pub fn new(mut reader: R) -> Result<Self, RomFsError> {
        let len = source_len(&mut reader)?;
        if len < HEADER_SIZE as u64 {
            return Err(RomFsError::Truncated { len });
        }
        let mut raw = [0u8; HEADER_SIZE];
        read_exact_at(&mut reader, 0, &mut raw)?;
        let truncated = || RomFsError::Truncated { len };
        let mut fields = Le::new(&raw);
        let header_size = fields.u64().ok_or_else(truncated)?;
        let mut location = || -> Option<TableLocation> {
            Some(TableLocation {
                offset: fields.u64()?,
                size: fields.u64()?,
            })
        };
        let dir_hash = location().ok_or_else(truncated)?;
        let dir_meta = location().ok_or_else(truncated)?;
        let file_hash = location().ok_or_else(truncated)?;
        let file_meta = location().ok_or_else(truncated)?;
        let data_offset = fields.u64().ok_or_else(truncated)?;
        if header_size != HEADER_SIZE as u64 {
            return Err(RomFsError::BadHeaderSize { found: header_size });
        }
        let header = RomFsHeader {
            dir_hash,
            dir_meta,
            file_hash,
            file_meta,
            data_offset,
        };

        let dir_buckets = load_buckets(&mut reader, len, Table::DirHash, dir_hash)?;
        let dir_meta = load_table(&mut reader, len, Table::DirMeta, dir_meta)?;
        let file_buckets = load_buckets(&mut reader, len, Table::FileHash, file_hash)?;
        let file_meta = load_table(&mut reader, len, Table::FileMeta, file_meta)?;

        let root = dir_at(&dir_meta, ROOT_OFFSET)?;
        if !root.name.is_empty() {
            return Err(RomFsError::BadRoot);
        }
        Ok(Self {
            reader,
            len,
            header,
            root,
            dir_meta,
            file_meta,
            dir_buckets,
            file_buckets,
        })
    }

    /// The header.
    pub fn header(&self) -> &RomFsHeader {
        &self.header
    }

    /// The root directory.
    pub fn root(&self) -> &DirEntry {
        &self.root
    }

    /// The directory entry at table offset `id`.
    pub fn dir(&self, id: u32) -> Result<DirEntry, RomFsError> {
        dir_at(&self.dir_meta, id)
    }

    /// The file entry at table offset `id`.
    pub fn file(&self, id: u32) -> Result<FileEntry, RomFsError> {
        file_at(&self.file_meta, id)
    }

    /// The children of `dir`, following its `child_dir` / `child_file` links and their sibling
    /// chains (Eden's `ProcessDirectory` and `ProcessFile`).
    pub fn read_dir(&self, dir: &DirEntry) -> Result<DirContents, RomFsError> {
        Ok(DirContents {
            dirs: self.collect_dirs(dir.child_dir, &mut HashSet::new())?,
            files: self.collect_files(dir.child_file, &mut HashSet::new())?,
        })
    }

    /// Lists every directory and file with its absolute path, parents before children and
    /// siblings in link order. An entry reached twice is [`RomFsError::Cycle`]; a path over
    /// [`MAX_PATH_LEN`] is [`RomFsError::PathTooLong`].
    pub fn walk(&self) -> Result<RomFsTree, RomFsError> {
        let mut tree = RomFsTree {
            dirs: Vec::new(),
            files: Vec::new(),
        };
        let mut seen_dirs = HashSet::from([self.root.id]);
        let mut seen_files = HashSet::new();
        let mut pending = vec![("/".to_owned(), self.root.clone())];
        while let Some((path, dir)) = pending.pop() {
            for file in self.collect_files(dir.child_file, &mut seen_files)? {
                tree.files.push((join(&path, &file.name)?, file));
            }
            let children = self.collect_dirs(dir.child_dir, &mut seen_dirs)?;
            let mut named = Vec::with_capacity(children.len());
            for child in children {
                named.push((join(&path, &child.name)?, child));
            }
            // Reversed so the first child is popped first.
            pending.extend(named.into_iter().rev());
            tree.dirs.push((path, dir));
        }
        Ok(tree)
    }

    /// The directory at the absolute path `path` (`/` is the root), found through the directory
    /// hash table. `Ok(None)` if there is no such directory.
    pub fn find_dir(&self, path: &str) -> Result<Option<DirEntry>, RomFsError> {
        let components = split_path(path)?;
        let mut current = self.root.clone();
        for name in components {
            match self.child_dir(current.id, name)? {
                Some(next) => current = next,
                None => return Ok(None),
            }
        }
        Ok(Some(current))
    }

    /// The file at the absolute path `path`, found through the directory and file hash tables.
    /// `Ok(None)` if there is no such file (including for `/`).
    pub fn find_file(&self, path: &str) -> Result<Option<FileEntry>, RomFsError> {
        let mut components = split_path(path)?;
        let Some(name) = components.pop() else {
            return Ok(None);
        };
        let mut parent = self.root.id;
        for dir in components {
            match self.child_dir(parent, dir)? {
                Some(next) => parent = next.id,
                None => return Ok(None),
            }
        }
        let hash = path_hash(parent, name.as_bytes());
        let limit = self.file_meta.len();
        lookup_chain(Table::FileMeta, &self.file_buckets, limit, hash, |id| {
            let entry = file_at(&self.file_meta, id)?;
            Ok(if entry.parent == parent && entry.name == name {
                ControlFlow::Break(entry)
            } else {
                ControlFlow::Continue(entry.hash_next)
            })
        })
    }

    /// The byte range of the source that `file`'s data occupies, checked for overflow and against
    /// the source length.
    pub fn file_extent(&self, file: &FileEntry) -> Result<Range<u64>, RomFsError> {
        checked_extent(self.header.data_offset, file.offset, file.size, self.len).ok_or(
            RomFsError::FileOutOfBounds {
                offset: file.offset,
                size: file.size,
                len: self.len,
            },
        )
    }

    /// Reads the whole of `file`.
    pub fn read_file(&mut self, file: &FileEntry) -> Result<Vec<u8>, RomFsError> {
        let extent = self.file_extent(file)?;
        Ok(read_vec_at(
            &mut self.reader,
            extent.start,
            extent.end - extent.start,
        )?)
    }

    /// Reads up to `buf.len()` bytes of `file` starting `offset` bytes into it, and returns how
    /// many were read: fewer than `buf.len()` only at the end of the file, and zero at or past it.
    pub fn read_file_at(
        &mut self,
        file: &FileEntry,
        offset: u64,
        buf: &mut [u8],
    ) -> Result<usize, RomFsError> {
        let extent = self.file_extent(file)?;
        Ok(read_window(&mut self.reader, extent, offset, buf)?)
    }

    /// The subdirectory `name` of the directory at `parent`, through the directory hash table.
    fn child_dir(&self, parent: u32, name: &str) -> Result<Option<DirEntry>, RomFsError> {
        let hash = path_hash(parent, name.as_bytes());
        let limit = self.dir_meta.len();
        lookup_chain(Table::DirMeta, &self.dir_buckets, limit, hash, |id| {
            let entry = dir_at(&self.dir_meta, id)?;
            Ok(if entry.parent == parent && entry.name == name {
                ControlFlow::Break(entry)
            } else {
                ControlFlow::Continue(entry.hash_next)
            })
        })
    }

    /// The directories on the sibling chain starting at `start`; an entry already in `seen` (or
    /// repeated on the chain) is a cycle.
    fn collect_dirs(
        &self,
        start: Option<u32>,
        seen: &mut HashSet<u32>,
    ) -> Result<Vec<DirEntry>, RomFsError> {
        let mut out = Vec::new();
        let mut next = start;
        while let Some(id) = next {
            if !seen.insert(id) {
                return Err(RomFsError::Cycle {
                    table: Table::DirMeta,
                    id,
                });
            }
            let entry = dir_at(&self.dir_meta, id)?;
            next = entry.sibling;
            out.push(entry);
        }
        Ok(out)
    }

    /// The files on the sibling chain starting at `start`; an entry already in `seen` (or
    /// repeated on the chain) is a cycle.
    fn collect_files(
        &self,
        start: Option<u32>,
        seen: &mut HashSet<u32>,
    ) -> Result<Vec<FileEntry>, RomFsError> {
        let mut out = Vec::new();
        let mut next = start;
        while let Some(id) = next {
            if !seen.insert(id) {
                return Err(RomFsError::Cycle {
                    table: Table::FileMeta,
                    id,
                });
            }
            let entry = file_at(&self.file_meta, id)?;
            next = entry.sibling;
            out.push(entry);
        }
        Ok(out)
    }
}

/// Walks the hash chain for `hash`, handing each entry offset to `step` until it breaks with a
/// result or the chain ends. A chain longer than `limit` (the table's byte size, an upper bound
/// on the distinct entry offsets) must revisit an entry and is a cycle.
fn lookup_chain<T>(
    table: Table,
    buckets: &[u32],
    limit: usize,
    hash: u32,
    mut step: impl FnMut(u32) -> Result<ControlFlow<T, Option<u32>>, RomFsError>,
) -> Result<Option<T>, RomFsError> {
    let bucket = usize::try_from(hash)
        .ok()
        .and_then(|hash| hash.checked_rem(buckets.len()))
        .and_then(|bucket| buckets.get(bucket))
        .copied();
    let mut next = bucket.and_then(link);
    let mut last = ENTRY_EMPTY;
    for _ in 0..=limit {
        let Some(id) = next else {
            return Ok(None);
        };
        last = id;
        match step(id)? {
            ControlFlow::Break(found) => return Ok(Some(found)),
            ControlFlow::Continue(after) => next = after,
        }
    }
    Err(RomFsError::Cycle { table, id: last })
}

/// A link word as an optional entry offset.
fn link(word: u32) -> Option<u32> {
    (word != ENTRY_EMPTY).then_some(word)
}

/// Splits an absolute lookup path into its components (`/` has none).
fn split_path(path: &str) -> Result<Vec<&str>, RomFsError> {
    if path.len() > MAX_PATH_LEN {
        return Err(RomFsError::PathTooLong);
    }
    let invalid = || RomFsError::InvalidPath {
        path: path.to_owned(),
    };
    let rest = path.strip_prefix('/').ok_or_else(invalid)?;
    if rest.is_empty() {
        return Ok(Vec::new());
    }
    let components: Vec<&str> = rest.split('/').collect();
    if components
        .iter()
        .any(|part| part.is_empty() || *part == "." || *part == "..")
    {
        return Err(invalid());
    }
    Ok(components)
}

/// `name` below the directory path `parent`, within [`MAX_PATH_LEN`].
fn join(parent: &str, name: &str) -> Result<String, RomFsError> {
    let mut path = if parent == "/" {
        String::new()
    } else {
        parent.to_owned()
    };
    path.push('/');
    path.push_str(name);
    if path.len() > MAX_PATH_LEN {
        return Err(RomFsError::PathTooLong);
    }
    Ok(path)
}

/// Reads a metadata table, checking it against the source length first.
fn load_table<R: Read + Seek>(
    reader: &mut R,
    len: u64,
    table: Table,
    location: TableLocation,
) -> Result<Vec<u8>, RomFsError> {
    if location.size > MAX_TABLE_SIZE {
        return Err(RomFsError::TableTooLarge {
            table,
            size: location.size,
        });
    }
    let extent = checked_extent(0, location.offset, location.size, len).ok_or(
        RomFsError::TableOutOfBounds {
            table,
            offset: location.offset,
            size: location.size,
            len,
        },
    )?;
    Ok(read_vec_at(reader, extent.start, location.size)?)
}

/// Reads a hash table as bucket words.
fn load_buckets<R: Read + Seek>(
    reader: &mut R,
    len: u64,
    table: Table,
    location: TableLocation,
) -> Result<Vec<u32>, RomFsError> {
    if location.size == 0 || !location.size.is_multiple_of(4) {
        return Err(RomFsError::BadHashTable {
            table,
            size: location.size,
        });
    }
    let bytes = load_table(reader, len, table, location)?;
    // A multiple of 4 bytes (checked above), so there is no remainder to drop.
    let (words, _) = bytes.as_chunks::<4>();
    Ok(words.iter().copied().map(u32::from_le_bytes).collect())
}

/// The name that follows an entry: `name_len` bytes at the reader's position.
fn read_name(
    fields: &mut Le<'_>,
    table: Table,
    id: u32,
    name_len: u32,
) -> Result<String, RomFsError> {
    let out_of_bounds = RomFsError::NameOutOfBounds {
        table,
        id,
        name_len,
    };
    let bytes = usize::try_from(name_len)
        .ok()
        .and_then(|len| fields.bytes(len))
        .ok_or(out_of_bounds)?;
    std::str::from_utf8(bytes)
        .map(str::to_owned)
        .map_err(|_| RomFsError::NameNotUtf8 { table, id })
}

/// Decodes the directory entry at `id` in `table`.
fn dir_at(table: &[u8], id: u32) -> Result<DirEntry, RomFsError> {
    let bad = || RomFsError::EntryOutOfBounds {
        table: Table::DirMeta,
        id,
    };
    let rest = usize::try_from(id)
        .ok()
        .and_then(|at| table.get(at..))
        .ok_or_else(bad)?;
    let mut fields = Le::new(rest);
    let parent = fields.u32().ok_or_else(bad)?;
    let sibling = fields.u32().ok_or_else(bad)?;
    let child_dir = fields.u32().ok_or_else(bad)?;
    let child_file = fields.u32().ok_or_else(bad)?;
    let hash_next = fields.u32().ok_or_else(bad)?;
    let name_len = fields.u32().ok_or_else(bad)?;
    let name = read_name(&mut fields, Table::DirMeta, id, name_len)?;
    Ok(DirEntry {
        id,
        parent,
        sibling: link(sibling),
        child_dir: link(child_dir),
        child_file: link(child_file),
        hash_next: link(hash_next),
        name,
    })
}

/// Decodes the file entry at `id` in `table`.
fn file_at(table: &[u8], id: u32) -> Result<FileEntry, RomFsError> {
    let bad = || RomFsError::EntryOutOfBounds {
        table: Table::FileMeta,
        id,
    };
    let rest = usize::try_from(id)
        .ok()
        .and_then(|at| table.get(at..))
        .ok_or_else(bad)?;
    let mut fields = Le::new(rest);
    let parent = fields.u32().ok_or_else(bad)?;
    let sibling = fields.u32().ok_or_else(bad)?;
    let offset = fields.u64().ok_or_else(bad)?;
    let size = fields.u64().ok_or_else(bad)?;
    let hash_next = fields.u32().ok_or_else(bad)?;
    let name_len = fields.u32().ok_or_else(bad)?;
    let name = read_name(&mut fields, Table::FileMeta, id, name_len)?;
    Ok(FileEntry {
        id,
        parent,
        sibling: link(sibling),
        offset,
        size,
        hash_next: link(hash_next),
        name,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use std::io::{Cursor, SeekFrom};

    /// Fixed seed so property runs are reproducible.
    const PROPTEST_SEED: u64 = 0x5EED_0000_20F5;

    /// Field offsets inside a directory entry (`DirectoryEntry`).
    const DIR_SIBLING: usize = 4;
    const DIR_CHILD_DIR: usize = 8;
    const DIR_CHILD_FILE: usize = 0xC;
    const DIR_HASH_NEXT: usize = 0x10;
    const DIR_NAME_LEN: usize = 0x14;
    /// Field offsets inside a file entry (`FileEntry`).
    const FILE_PARENT: usize = 0;
    const FILE_SIBLING: usize = 4;
    const FILE_SIZE: usize = 0x10;
    const FILE_HASH_NEXT: usize = 0x18;
    const FILE_NAME_LEN: usize = 0x1C;
    /// Offset of `data_offset` in the header.
    const DATA_OFFSET_AT: usize = 0x48;

    /// A tree to build into an image. Parent indices refer to directories: 0 is the root and `i`
    /// is the `i`th (1-based) entry of `dirs`.
    #[derive(Debug, Clone)]
    struct TreeSpec {
        dirs: Vec<(usize, String)>,
        files: Vec<(usize, String, Vec<u8>)>,
    }

    /// An image and what the builder knows about it.
    struct Built {
        bytes: Vec<u8>,
        header: RomFsHeader,
        /// Directories in table order, with absolute paths (`/` for the root).
        dirs: Vec<(String, DirEntry)>,
        /// Files in table order, with absolute paths and contents.
        files: Vec<(String, FileEntry, Vec<u8>)>,
    }

    /// Eden's `romfs_get_hash_table_count`, kept separate from the reader (which takes the bucket
    /// count from the table size).
    fn bucket_count(entries: usize) -> usize {
        if entries < 3 {
            return 3;
        }
        if entries < 19 {
            return entries | 1;
        }
        let mut count = entries;
        while [2, 3, 5, 7, 11, 13, 17].iter().any(|p| count % p == 0) {
            count += 1;
        }
        count
    }

    fn u32_of(n: usize) -> u32 {
        u32::try_from(n).unwrap()
    }

    fn u64_of(n: usize) -> u64 {
        u64::try_from(n).unwrap()
    }

    fn usize_of(n: u32) -> usize {
        usize::try_from(n).unwrap()
    }

    fn usize_of64(n: u64) -> usize {
        usize::try_from(n).unwrap()
    }

    fn put(bytes: &mut [u8], at: usize, value: &[u8]) {
        bytes[at..at + value.len()].copy_from_slice(value);
    }

    /// Lays out an image the way Eden's `RomFSBuildContext::Build` does (entries sorted by path,
    /// 4-aligned names, 16-aligned file data, chained hash buckets), except that the file data
    /// starts `data_pad` * 16 bytes later than the minimum and the four tables are placed in the
    /// order `table_order` (a permutation of 0..4: dir hash, dir meta, file hash, file meta).
    fn build(spec: &TreeSpec, data_pad: usize, table_order: &[usize]) -> Built {
        let mut dir_paths = vec![String::new()];
        for (parent, name) in &spec.dirs {
            let path = format!("{}/{}", dir_paths[*parent], name);
            dir_paths.push(path);
        }
        let dir_name = |i: usize| -> &str { if i == 0 { "" } else { &spec.dirs[i - 1].1 } };
        let dir_parent = |i: usize| -> usize { if i == 0 { 0 } else { spec.dirs[i - 1].0 } };
        let file_paths: Vec<String> = spec
            .files
            .iter()
            .map(|(parent, name, _)| format!("{}/{}", dir_paths[*parent], name))
            .collect();

        let mut dir_order: Vec<usize> = (0..dir_paths.len()).collect();
        dir_order.sort_by(|&a, &b| dir_paths[a].cmp(&dir_paths[b]));
        let mut file_order: Vec<usize> = (0..spec.files.len()).collect();
        file_order.sort_by(|&a, &b| file_paths[a].cmp(&file_paths[b]));

        // Entry offsets.
        let mut dir_id = vec![0u32; dir_paths.len()];
        let mut cursor = 0usize;
        for &i in &dir_order {
            dir_id[i] = u32_of(cursor);
            cursor += DIR_ENTRY_SIZE + dir_name(i).len().next_multiple_of(4);
        }
        let dir_table_size = cursor;
        let mut file_id = vec![0u32; spec.files.len()];
        let mut file_data = vec![0usize; spec.files.len()];
        let mut cursor = 0usize;
        let mut partition = 0usize;
        for &i in &file_order {
            file_id[i] = u32_of(cursor);
            cursor += FILE_ENTRY_SIZE + spec.files[i].1.len().next_multiple_of(4);
            partition = partition.next_multiple_of(16);
            file_data[i] = partition;
            partition += spec.files[i].2.len();
        }
        let file_table_size = cursor;

        // Links.
        let mut dir_buckets = vec![ENTRY_EMPTY; bucket_count(dir_paths.len())];
        let mut dirs = Vec::new();
        for (n, &i) in dir_order.iter().enumerate() {
            let parent = dir_parent(i);
            let sibling = dir_order[n + 1..]
                .iter()
                .copied()
                .find(|&j| j != 0 && dir_parent(j) == parent)
                .map(|j| dir_id[j]);
            let child_dir = dir_order
                .iter()
                .copied()
                .find(|&j| j != 0 && dir_parent(j) == i)
                .map(|j| dir_id[j]);
            let child_file = file_order
                .iter()
                .copied()
                .find(|&f| spec.files[f].0 == i)
                .map(|f| file_id[f]);
            let parent_id = if i == 0 { 0 } else { dir_id[parent] };
            let bucket = usize_of(path_hash(parent_id, dir_name(i).as_bytes())) % dir_buckets.len();
            let hash_next = link(dir_buckets[bucket]);
            dir_buckets[bucket] = dir_id[i];
            let path = if i == 0 {
                "/".to_owned()
            } else {
                dir_paths[i].clone()
            };
            dirs.push((
                path,
                DirEntry {
                    id: dir_id[i],
                    parent: parent_id,
                    sibling: if i == 0 { None } else { sibling },
                    child_dir,
                    child_file,
                    hash_next,
                    name: dir_name(i).to_owned(),
                },
            ));
        }
        let mut file_buckets = vec![ENTRY_EMPTY; bucket_count(spec.files.len())];
        let mut files = Vec::new();
        for (n, &i) in file_order.iter().enumerate() {
            let (parent, name, content) = &spec.files[i];
            let sibling = file_order[n + 1..]
                .iter()
                .copied()
                .find(|&j| spec.files[j].0 == *parent)
                .map(|j| file_id[j]);
            let bucket = usize_of(path_hash(dir_id[*parent], name.as_bytes())) % file_buckets.len();
            let hash_next = link(file_buckets[bucket]);
            file_buckets[bucket] = file_id[i];
            files.push((
                file_paths[i].clone(),
                FileEntry {
                    id: file_id[i],
                    parent: dir_id[*parent],
                    sibling,
                    offset: u64_of(file_data[i]),
                    size: u64_of(content.len()),
                    hash_next,
                    name: name.clone(),
                },
                content.clone(),
            ));
        }

        // Encode the tables.
        let word = |value: Option<u32>| value.unwrap_or(ENTRY_EMPTY).to_le_bytes();
        let mut dir_meta = Vec::new();
        for (_, entry) in &dirs {
            dir_meta.extend_from_slice(&entry.parent.to_le_bytes());
            dir_meta.extend_from_slice(&word(entry.sibling));
            dir_meta.extend_from_slice(&word(entry.child_dir));
            dir_meta.extend_from_slice(&word(entry.child_file));
            dir_meta.extend_from_slice(&word(entry.hash_next));
            dir_meta.extend_from_slice(&u32_of(entry.name.len()).to_le_bytes());
            dir_meta.extend_from_slice(entry.name.as_bytes());
            dir_meta.resize(dir_meta.len().next_multiple_of(4), 0);
        }
        assert_eq!(dir_meta.len(), dir_table_size);
        let mut file_meta = Vec::new();
        for (_, entry, _) in &files {
            file_meta.extend_from_slice(&entry.parent.to_le_bytes());
            file_meta.extend_from_slice(&word(entry.sibling));
            file_meta.extend_from_slice(&entry.offset.to_le_bytes());
            file_meta.extend_from_slice(&entry.size.to_le_bytes());
            file_meta.extend_from_slice(&word(entry.hash_next));
            file_meta.extend_from_slice(&u32_of(entry.name.len()).to_le_bytes());
            file_meta.extend_from_slice(entry.name.as_bytes());
            file_meta.resize(file_meta.len().next_multiple_of(4), 0);
        }
        assert_eq!(file_meta.len(), file_table_size);
        let encode_buckets =
            |buckets: &[u32]| -> Vec<u8> { buckets.iter().flat_map(|b| b.to_le_bytes()).collect() };
        let tables = [
            encode_buckets(&dir_buckets),
            dir_meta,
            encode_buckets(&file_buckets),
            file_meta,
        ];

        // Place the file data, then the tables in the requested order.
        let data_offset = HEADER_SIZE + 16 * data_pad;
        let mut bytes = vec![0u8; data_offset + partition];
        for (_, entry, content) in &files {
            let at = data_offset + usize_of64(entry.offset);
            bytes[at..at + content.len()].copy_from_slice(content);
        }
        let mut locations = [TableLocation { offset: 0, size: 0 }; 4];
        for &t in table_order {
            bytes.resize(bytes.len().next_multiple_of(4), 0);
            locations[t] = TableLocation {
                offset: u64_of(bytes.len()),
                size: u64_of(tables[t].len()),
            };
            bytes.extend_from_slice(&tables[t]);
        }
        let header = RomFsHeader {
            dir_hash: locations[0],
            dir_meta: locations[1],
            file_hash: locations[2],
            file_meta: locations[3],
            data_offset: u64_of(data_offset),
        };
        put(&mut bytes, 0, &u64_of(HEADER_SIZE).to_le_bytes());
        for (n, location) in locations.iter().enumerate() {
            put(&mut bytes, location_at(n), &location.offset.to_le_bytes());
            put(&mut bytes, location_at(n) + 8, &location.size.to_le_bytes());
        }
        put(
            &mut bytes,
            DATA_OFFSET_AT,
            &header.data_offset.to_le_bytes(),
        );
        Built {
            bytes,
            header,
            dirs,
            files,
        }
    }

    /// Offset within the header of table `n`'s offset (`+ 8` for its size).
    fn location_at(table: usize) -> usize {
        8 + 16 * table
    }

    const IDENTITY: [usize; 4] = [0, 1, 2, 3];

    /// `/`, `/root.bin`, `/docs/{a.bin, readme.txt}`, `/docs/img/`.
    fn sample_spec() -> TreeSpec {
        TreeSpec {
            dirs: vec![(0, "docs".into()), (1, "img".into())],
            files: vec![
                (0, "root.bin".into(), b"ROOT".to_vec()),
                (1, "readme.txt".into(), b"hello romfs".to_vec()),
                (1, "a.bin".into(), vec![1, 2, 3]),
            ],
        }
    }

    fn sample() -> Built {
        build(&sample_spec(), 1, &IDENTITY)
    }

    fn open(bytes: &[u8]) -> Result<RomFs<Cursor<&[u8]>>, RomFsError> {
        RomFs::new(Cursor::new(bytes))
    }

    /// Where a table starts in the image, and where entry `id` of it does.
    fn dir_entry_at(built: &Built, id: u32) -> usize {
        usize_of64(built.header.dir_meta.offset) + usize_of(id)
    }

    fn file_entry_at(built: &Built, id: u32) -> usize {
        usize_of64(built.header.file_meta.offset) + usize_of(id)
    }

    fn dir_named(built: &Built, path: &str) -> DirEntry {
        built
            .dirs
            .iter()
            .find(|(p, _)| p == path)
            .unwrap()
            .1
            .clone()
    }

    /// Everything the builder knows must come back out of the reader, through every public path.
    fn check_image(built: &Built) {
        let mut fs = open(&built.bytes).unwrap();
        assert_eq!(*fs.header(), built.header);
        assert_eq!(*fs.root(), built.dirs[0].1);

        let mut tree = fs.walk().unwrap();
        let mut expected_dirs = built.dirs.clone();
        tree.dirs.sort_by(|a, b| a.0.cmp(&b.0));
        expected_dirs.sort_by(|a, b| a.0.cmp(&b.0));
        assert_eq!(tree.dirs, expected_dirs);
        let mut expected_files: Vec<(String, FileEntry)> = built
            .files
            .iter()
            .map(|(path, entry, _)| (path.clone(), entry.clone()))
            .collect();
        tree.files.sort_by(|a, b| a.0.cmp(&b.0));
        expected_files.sort_by(|a, b| a.0.cmp(&b.0));
        assert_eq!(tree.files, expected_files);

        for (path, entry) in &built.dirs {
            assert_eq!(fs.dir(entry.id).unwrap(), *entry);
            assert_eq!(fs.find_dir(path).unwrap().as_ref(), Some(entry));
            let listing = fs.read_dir(entry).unwrap();
            let want_dirs: Vec<&DirEntry> = built
                .dirs
                .iter()
                .map(|(_, d)| d)
                .filter(|d| d.id != ROOT_OFFSET && d.parent == entry.id)
                .collect();
            assert_eq!(listing.dirs.iter().collect::<Vec<_>>(), want_dirs);
            let want_files: Vec<&FileEntry> = built
                .files
                .iter()
                .map(|(_, f, _)| f)
                .filter(|f| f.parent == entry.id)
                .collect();
            assert_eq!(listing.files.iter().collect::<Vec<_>>(), want_files);
            let missing = format!("{}/missing", path.trim_end_matches('/'));
            assert!(fs.find_dir(&missing).unwrap().is_none());
        }
        for (path, entry, content) in &built.files {
            assert_eq!(fs.file(entry.id).unwrap(), *entry);
            assert_eq!(fs.find_file(path).unwrap().as_ref(), Some(entry));
            assert_eq!(fs.read_file(entry).unwrap(), *content);
            assert_eq!(
                fs.file_extent(entry).unwrap().start,
                built.header.data_offset + entry.offset
            );
            assert!(fs.find_file(&format!("{path}/missing")).unwrap().is_none());
        }
    }

    #[test]
    fn path_hash_known_answers() {
        // Worked by hand from `hash = parent ^ SEED; for each byte: hash = rotr(hash, 5) ^ byte`,
        // with the rotation written out as shifts.
        let rotr5 = |h: u32| (h >> 5) | (h << 27);
        assert_eq!(path_hash(0, b""), HASH_SEED);
        assert_eq!(path_hash(0x40, b""), 0x40 ^ HASH_SEED);
        assert_eq!(path_hash(0, b"a"), rotr5(HASH_SEED) ^ 0x61);
        assert_eq!(path_hash(0, b"ab"), rotr5(rotr5(HASH_SEED) ^ 0x61) ^ 0x62);
        assert_eq!(path_hash(7, b"x"), rotr5(7 ^ HASH_SEED) ^ 0x78);
        // Bytes are unsigned: no sign extension (a guess; see the module docs).
        assert_eq!(path_hash(0, &[0x80]), rotr5(HASH_SEED) ^ 0x80);
    }

    #[test]
    fn parses_a_small_image() {
        let built = sample();
        check_image(&built);
        let mut fs = open(&built.bytes).unwrap();
        assert_eq!(fs.root().name, "");
        let readme = fs.find_file("/docs/readme.txt").unwrap().unwrap();
        assert_eq!(readme.size, 11);
        assert_eq!(fs.read_file(&readme).unwrap(), b"hello romfs");
        assert!(fs.find_file("/readme.txt").unwrap().is_none());
        assert!(fs.find_file("/docs").unwrap().is_none());
        assert!(fs.find_dir("/docs/img").unwrap().is_some());
        assert_eq!(fs.find_dir("/").unwrap().as_ref(), Some(fs.root()));
        assert!(fs.find_file("/").unwrap().is_none());
        let tree = fs.walk().unwrap();
        let paths: Vec<&str> = tree.files.iter().map(|(p, _)| p.as_str()).collect();
        assert_eq!(paths, ["/root.bin", "/docs/a.bin", "/docs/readme.txt"]);
    }

    #[test]
    fn a_file_and_a_directory_may_share_a_name() {
        let spec = TreeSpec {
            dirs: vec![(0, "same".into())],
            files: vec![
                (0, "same".into(), b"file".to_vec()),
                (1, "same".into(), b"inner".to_vec()),
            ],
        };
        let built = build(&spec, 0, &IDENTITY);
        check_image(&built);
        let mut fs = open(&built.bytes).unwrap();
        let file = fs.find_file("/same").unwrap().unwrap();
        assert_eq!(fs.read_file(&file).unwrap(), b"file");
        assert!(fs.find_dir("/same").unwrap().is_some());
        let inner = fs.find_file("/same/same").unwrap().unwrap();
        assert_eq!(fs.read_file(&inner).unwrap(), b"inner");
    }

    #[test]
    fn an_empty_image_has_only_a_root() {
        let built = build(
            &TreeSpec {
                dirs: vec![],
                files: vec![],
            },
            0,
            &IDENTITY,
        );
        check_image(&built);
        let fs = open(&built.bytes).unwrap();
        assert!(fs.walk().unwrap().files.is_empty());
    }

    #[test]
    fn rejects_short_input_and_a_wrong_header_size() {
        assert!(matches!(open(&[]), Err(RomFsError::Truncated { len: 0 })));
        let wrong = u64_of(HEADER_SIZE) + 8;
        let mut bytes = sample().bytes;
        put(&mut bytes, 0, &wrong.to_le_bytes());
        assert!(matches!(
            open(&bytes),
            Err(RomFsError::BadHeaderSize { found }) if found == wrong
        ));
    }

    #[test]
    fn rejects_tables_outside_the_source() {
        let tables = [
            Table::DirHash,
            Table::DirMeta,
            Table::FileHash,
            Table::FileMeta,
        ];
        let built = sample();
        let len = u64_of(built.bytes.len());
        let locations = [
            built.header.dir_hash,
            built.header.dir_meta,
            built.header.file_hash,
            built.header.file_meta,
        ];
        for (n, table) in tables.into_iter().enumerate() {
            let at = location_at(n);
            // Offset past the end; offset + size overflowing u64; size past the end.
            for (offset, size) in [
                (u64::MAX, 4),
                (u64::MAX - 3, 8),
                (locations[n].offset, len),
                (len, 4),
                (len + 4, 4),
            ] {
                let mut bytes = built.bytes.clone();
                put(&mut bytes, at, &offset.to_le_bytes());
                put(&mut bytes, at + 8, &size.to_le_bytes());
                assert!(
                    matches!(open(&bytes), Err(RomFsError::TableOutOfBounds { table: t, .. }) if t == table),
                    "table {table:?} at {offset:#x} + {size:#x}"
                );
            }
            // A table that ends exactly at the end of the source is within bounds.
            let mut bytes = built.bytes.clone();
            put(&mut bytes, at, &(len - locations[n].size).to_le_bytes());
            assert!(!matches!(
                open(&bytes),
                Err(RomFsError::TableOutOfBounds { .. })
            ));
        }
    }

    #[test]
    fn rejects_tables_larger_than_the_offset_space() {
        let mut bytes = sample().bytes;
        put(
            &mut bytes,
            location_at(1) + 8,
            &(MAX_TABLE_SIZE + 1).to_le_bytes(),
        );
        assert!(matches!(
            open(&bytes),
            Err(RomFsError::TableTooLarge {
                table: Table::DirMeta,
                ..
            })
        ));
    }

    #[test]
    fn hash_tables_must_be_a_non_zero_multiple_of_four_bytes() {
        for (n, table) in [(0, Table::DirHash), (2, Table::FileHash)] {
            for size in [0u64, 1, 6] {
                let mut bytes = sample().bytes;
                put(&mut bytes, location_at(n) + 8, &size.to_le_bytes());
                assert!(
                    matches!(open(&bytes), Err(RomFsError::BadHashTable { table: t, size: s }) if t == table && s == size),
                    "{table:?} size {size}"
                );
            }
        }
    }

    #[test]
    fn an_image_without_a_root_entry_is_rejected() {
        // Eden would assert; here the empty directory table has no entry at offset 0.
        let mut bytes = sample().bytes;
        put(&mut bytes, location_at(1) + 8, &0u64.to_le_bytes());
        assert!(matches!(
            open(&bytes),
            Err(RomFsError::EntryOutOfBounds {
                table: Table::DirMeta,
                id: 0
            })
        ));
    }

    #[test]
    fn the_root_must_be_unnamed() {
        let built = sample();
        let mut bytes = built.bytes.clone();
        put(
            &mut bytes,
            dir_entry_at(&built, ROOT_OFFSET) + DIR_NAME_LEN,
            &1u32.to_le_bytes(),
        );
        assert!(matches!(open(&bytes), Err(RomFsError::BadRoot)));
    }

    #[test]
    fn names_are_bounded_by_their_table_and_must_be_utf8() {
        let built = sample();
        let docs = dir_named(&built, "/docs").id;
        let docs_at = dir_entry_at(&built, docs);

        // Eden clamps an over-long name to the table; here it is an error.
        for name_len in [u32::MAX, 0x1_0000] {
            let mut bytes = built.bytes.clone();
            put(&mut bytes, docs_at + DIR_NAME_LEN, &name_len.to_le_bytes());
            let fs = open(&bytes).unwrap();
            assert!(matches!(
                fs.dir(docs),
                Err(RomFsError::NameOutOfBounds { table: Table::DirMeta, name_len: n, .. }) if n == name_len
            ));
            assert!(fs.walk().is_err());
        }

        let mut bytes = built.bytes.clone();
        put(&mut bytes, docs_at + DIR_ENTRY_SIZE, &[0xFF]);
        let fs = open(&bytes).unwrap();
        assert!(matches!(
            fs.dir(docs),
            Err(RomFsError::NameNotUtf8 {
                table: Table::DirMeta,
                ..
            })
        ));

        let file = built.files[0].1.clone();
        let mut bytes = built.bytes.clone();
        put(
            &mut bytes,
            file_entry_at(&built, file.id) + FILE_NAME_LEN,
            &u32::MAX.to_le_bytes(),
        );
        let fs = open(&bytes).unwrap();
        assert!(matches!(
            fs.file(file.id),
            Err(RomFsError::NameOutOfBounds {
                table: Table::FileMeta,
                ..
            })
        ));
    }

    #[test]
    fn links_outside_the_table_are_errors_not_zeroed_entries() {
        // Eden's GetEntry returns a zeroed entry here, and its sibling of 0 loops forever.
        let built = sample();
        let dir_len = u32_of(usize_of64(built.header.dir_meta.size));
        let root_at = dir_entry_at(&built, ROOT_OFFSET);
        for target in [
            dir_len,
            dir_len - 4,
            dir_len - u32_of(DIR_ENTRY_SIZE) + 4,
            0x7FFF_FFF0,
        ] {
            let mut bytes = built.bytes.clone();
            put(&mut bytes, root_at + DIR_CHILD_DIR, &target.to_le_bytes());
            let fs = open(&bytes).unwrap();
            assert!(
                matches!(fs.walk(), Err(RomFsError::EntryOutOfBounds { table: Table::DirMeta, id }) if id == target),
                "target {target:#x}"
            );
        }
        let mut bytes = built.bytes.clone();
        put(
            &mut bytes,
            root_at + DIR_CHILD_FILE,
            &0x7FFF_FFF0u32.to_le_bytes(),
        );
        let fs = open(&bytes).unwrap();
        assert!(matches!(
            fs.walk(),
            Err(RomFsError::EntryOutOfBounds {
                table: Table::FileMeta,
                ..
            })
        ));
    }

    #[test]
    fn linked_entries_that_loop_are_cycles() {
        let built = sample();
        let first_file = built.files[0].1.clone();

        // A file that is its own sibling.
        let mut bytes = built.bytes.clone();
        put(
            &mut bytes,
            file_entry_at(&built, first_file.id) + FILE_SIBLING,
            &first_file.id.to_le_bytes(),
        );
        let fs = open(&bytes).unwrap();
        assert!(matches!(
            fs.walk(),
            Err(RomFsError::Cycle {
                table: Table::FileMeta,
                ..
            })
        ));

        // The root as someone's child.
        let mut bytes = built.bytes.clone();
        put(
            &mut bytes,
            dir_entry_at(&built, ROOT_OFFSET) + DIR_CHILD_DIR,
            &ROOT_OFFSET.to_le_bytes(),
        );
        let fs = open(&bytes).unwrap();
        assert!(matches!(
            fs.walk(),
            Err(RomFsError::Cycle {
                table: Table::DirMeta,
                id: ROOT_OFFSET
            })
        ));

        // A directory that is its own sibling.
        let docs = dir_named(&built, "/docs");
        let mut bytes = built.bytes.clone();
        put(
            &mut bytes,
            dir_entry_at(&built, docs.id) + DIR_SIBLING,
            &docs.id.to_le_bytes(),
        );
        let fs = open(&bytes).unwrap();
        assert!(matches!(
            fs.read_dir(fs.root()),
            Err(RomFsError::Cycle {
                table: Table::DirMeta,
                ..
            })
        ));
    }

    #[test]
    fn hash_chains_that_loop_are_cycles() {
        let built = sample();

        // Every file entry's chain link points at itself, and every file bucket at the first
        // file, so any name lookup walks into the loop.
        let mut bytes = built.bytes.clone();
        for (_, entry, _) in &built.files {
            put(
                &mut bytes,
                file_entry_at(&built, entry.id) + FILE_HASH_NEXT,
                &entry.id.to_le_bytes(),
            );
        }
        let first = built.files[0].1.id;
        let buckets_at = usize_of64(built.header.file_hash.offset);
        let bucket_bytes = usize_of64(built.header.file_hash.size);
        for word in bytes[buckets_at..buckets_at + bucket_bytes].chunks_exact_mut(4) {
            word.copy_from_slice(&first.to_le_bytes());
        }
        let fs = open(&bytes).unwrap();
        assert!(matches!(
            fs.find_file("/missing"),
            Err(RomFsError::Cycle {
                table: Table::FileMeta,
                id
            }) if id == first
        ));

        // Same for directories, with the root linked to itself.
        let mut bytes = built.bytes.clone();
        put(
            &mut bytes,
            dir_entry_at(&built, ROOT_OFFSET) + DIR_HASH_NEXT,
            &ROOT_OFFSET.to_le_bytes(),
        );
        let buckets_at = usize_of64(built.header.dir_hash.offset);
        let bucket_bytes = usize_of64(built.header.dir_hash.size);
        for word in bytes[buckets_at..buckets_at + bucket_bytes].chunks_exact_mut(4) {
            word.copy_from_slice(&ROOT_OFFSET.to_le_bytes());
        }
        let fs = open(&bytes).unwrap();
        assert!(matches!(
            fs.find_dir("/nothing/here"),
            Err(RomFsError::Cycle {
                table: Table::DirMeta,
                id: ROOT_OFFSET
            })
        ));
    }

    #[test]
    fn lookup_checks_the_parent_field_not_just_the_name() {
        // Eden never reads the hash tables, so this is behaviour of the hash-table lookup only.
        let built = sample();
        let file = built
            .files
            .iter()
            .find(|(p, _, _)| p == "/docs/a.bin")
            .unwrap()
            .1
            .clone();
        let mut bytes = built.bytes.clone();
        put(
            &mut bytes,
            file_entry_at(&built, file.id) + FILE_PARENT,
            &(file.parent + 1).to_le_bytes(),
        );
        let fs = open(&bytes).unwrap();
        assert!(fs.find_file("/docs/a.bin").unwrap().is_none());
    }

    #[test]
    fn file_data_is_checked_over_its_whole_extent() {
        let built = sample();
        let file = built.files[0].1.clone();
        let mut fs = open(&built.bytes).unwrap();
        let data_offset = built.header.data_offset;
        let len = u64_of(built.bytes.len());
        let room = len - data_offset;
        let forged = |offset: u64, size: u64| FileEntry {
            offset,
            size,
            ..file.clone()
        };

        assert_eq!(fs.file_extent(&forged(0, room)).unwrap(), data_offset..len);
        assert_eq!(fs.file_extent(&forged(room, 0)).unwrap(), len..len);
        for (offset, size) in [
            (0, room + 1),
            (1, room),
            (room + 1, 0),
            (u64::MAX, 1),
            (1, u64::MAX),
            (u64::MAX, u64::MAX),
            (u64::MAX / 2 + 1, u64::MAX / 2 + 1),
        ] {
            let entry = forged(offset, size);
            assert!(
                matches!(
                    fs.file_extent(&entry),
                    Err(RomFsError::FileOutOfBounds { .. })
                ),
                "{offset:#x} + {size:#x}"
            );
            assert!(matches!(
                fs.read_file(&entry),
                Err(RomFsError::FileOutOfBounds { .. })
            ));
            assert!(matches!(
                fs.read_file_at(&entry, 0, &mut [0u8; 2]),
                Err(RomFsError::FileOutOfBounds { .. })
            ));
        }

        // A data offset near u64::MAX overflows the base itself.
        let mut bytes = built.bytes.clone();
        put(&mut bytes, DATA_OFFSET_AT, &(u64::MAX - 1).to_le_bytes());
        let fs = open(&bytes).unwrap();
        assert!(matches!(
            fs.file_extent(&forged(2, 0)),
            Err(RomFsError::FileOutOfBounds { .. })
        ));

        // A stored entry that overruns the source is reported when it is read.
        let mut bytes = built.bytes.clone();
        put(
            &mut bytes,
            file_entry_at(&built, file.id) + FILE_SIZE,
            &u64::MAX.to_le_bytes(),
        );
        let mut fs = open(&bytes).unwrap();
        let stored = fs.file(file.id).unwrap();
        assert!(matches!(
            fs.read_file(&stored),
            Err(RomFsError::FileOutOfBounds { .. })
        ));
    }

    #[test]
    fn read_file_at_returns_windows_and_stops_at_the_end_of_the_file() {
        let built = sample();
        let mut fs = open(&built.bytes).unwrap();
        let file = fs.find_file("/docs/readme.txt").unwrap().unwrap();
        let mut buf = [0u8; 5];
        assert_eq!(fs.read_file_at(&file, 6, &mut buf).unwrap(), 5);
        assert_eq!(&buf, b"romfs");
        // A window crossing the end of the file is cut there.
        let mut buf = [0xFFu8; 8];
        assert_eq!(fs.read_file_at(&file, 8, &mut buf).unwrap(), 3);
        assert_eq!(&buf[..3], b"mfs");
        assert_eq!(&buf[3..], &[0xFF; 5]);
        assert_eq!(fs.read_file_at(&file, 11, &mut buf).unwrap(), 0);
        assert_eq!(fs.read_file_at(&file, u64::MAX, &mut buf).unwrap(), 0);
        assert_eq!(fs.read_file_at(&file, 0, &mut []).unwrap(), 0);
    }

    #[test]
    fn lookup_paths_must_be_normalised_absolute_paths() {
        let built = sample();
        let fs = open(&built.bytes).unwrap();
        for path in [
            "",
            "docs",
            "docs/a.bin",
            "//",
            "/docs//a.bin",
            "/docs/",
            "/./docs",
            "/docs/../docs",
            "/..",
        ] {
            assert!(
                matches!(fs.find_file(path), Err(RomFsError::InvalidPath { .. })),
                "{path:?}"
            );
            assert!(matches!(
                fs.find_dir(path),
                Err(RomFsError::InvalidPath { .. })
            ));
        }
        let longest = format!("/{}", "a".repeat(MAX_PATH_LEN - 1));
        assert!(fs.find_file(&longest).unwrap().is_none());
        let too_long = format!("{longest}a");
        assert!(matches!(
            fs.find_file(&too_long),
            Err(RomFsError::PathTooLong)
        ));
        assert!(matches!(
            fs.find_dir(&too_long),
            Err(RomFsError::PathTooLong)
        ));
    }

    #[test]
    fn walked_paths_are_limited_to_the_maximum_path_length() {
        let fits = "a".repeat(MAX_PATH_LEN - 1);
        let spec = TreeSpec {
            dirs: vec![(0, fits.clone())],
            files: vec![],
        };
        let built = build(&spec, 0, &IDENTITY);
        let fs = open(&built.bytes).unwrap();
        let tree = fs.walk().unwrap();
        assert_eq!(tree.dirs[1].0.len(), MAX_PATH_LEN);
        assert!(fs.find_dir(&format!("/{fits}")).unwrap().is_some());

        let spec = TreeSpec {
            dirs: vec![(0, "a".repeat(MAX_PATH_LEN))],
            files: vec![],
        };
        let built = build(&spec, 0, &IDENTITY);
        let fs = open(&built.bytes).unwrap();
        assert!(matches!(fs.walk(), Err(RomFsError::PathTooLong)));

        // Depth cannot get round the limit either.
        let dirs: Vec<(usize, String)> = (0..MAX_PATH_LEN).map(|i| (i, "d".to_owned())).collect();
        let built = build(
            &TreeSpec {
                dirs,
                files: vec![],
            },
            0,
            &IDENTITY,
        );
        let fs = open(&built.bytes).unwrap();
        assert!(matches!(fs.walk(), Err(RomFsError::PathTooLong)));
    }

    #[test]
    fn io_errors_from_the_source_are_reported() {
        struct Failing;
        impl Read for Failing {
            fn read(&mut self, _: &mut [u8]) -> std::io::Result<usize> {
                Err(std::io::Error::other("boom"))
            }
        }
        impl Seek for Failing {
            fn seek(&mut self, _: SeekFrom) -> std::io::Result<u64> {
                Ok(0x1000)
            }
        }
        assert!(matches!(RomFs::new(Failing), Err(RomFsError::Io(_))));
    }

    #[test]
    fn file_data_is_read_on_demand_from_a_large_image() {
        let mut spec = sample_spec();
        spec.files[0].2 = vec![0xAB; 4 << 20];
        let built = build(&spec, 0, &IDENTITY);
        let mut fs = open(&built.bytes).unwrap();
        let big = fs.find_file("/root.bin").unwrap().unwrap();
        let mut window = [0u8; 4];
        assert_eq!(
            fs.read_file_at(&big, (4 << 20) - 2, &mut window).unwrap(),
            2
        );
        assert_eq!(&window[..2], [0xAB, 0xAB]);
    }

    fn tree_specs() -> impl Strategy<Value = TreeSpec> {
        (0usize..5, 0usize..9).prop_flat_map(|(dir_count, file_count)| {
            let dirs = proptest::collection::vec((any::<usize>(), "[a-zé]{0,3}"), dir_count);
            let files = proptest::collection::vec(
                (
                    any::<usize>(),
                    "[a-zé.]{0,3}",
                    proptest::collection::vec(any::<u8>(), 0..20),
                ),
                file_count,
            );
            (dirs, files).prop_map(move |(dirs, files)| TreeSpec {
                // Directory i's parent is the root or an earlier directory.
                dirs: dirs
                    .into_iter()
                    .enumerate()
                    .map(|(i, (pick, suffix))| (pick % (i + 1), format!("n{i}{suffix}")))
                    .collect(),
                files: files
                    .into_iter()
                    .enumerate()
                    .map(|(i, (pick, suffix, content))| {
                        (pick % (dir_count + 1), format!("n{i}{suffix}"), content)
                    })
                    .collect(),
            })
        })
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            rng_seed: proptest::test_runner::RngSeed::Fixed(PROPTEST_SEED),
            ..ProptestConfig::default()
        })]

        /// Build then parse: every field of every entry must come back exactly, and every lookup
        /// path must work, so reading any two fields or tables in the wrong order fails here.
        #[test]
        fn build_then_parse_round_trips(
            spec in tree_specs(),
            data_pad in 0usize..4,
            order in Just(vec![0usize, 1, 2, 3]).prop_shuffle(),
        ) {
            check_image(&build(&spec, data_pad, &order));
        }

        #[test]
        fn arbitrary_bytes_never_panic(bytes in proptest::collection::vec(any::<u8>(), 0..512)) {
            exercise(&bytes);
        }

        #[test]
        fn arbitrary_tables_behind_a_valid_header_never_panic(
            offsets in proptest::collection::vec(0u64..0x100, 5),
            sizes in proptest::collection::vec(0u64..0x60, 4),
            tail in proptest::collection::vec(any::<u8>(), 0x100..0x180),
        ) {
            let mut bytes = u64_of(HEADER_SIZE).to_le_bytes().to_vec();
            for (n, (offset, size)) in offsets.iter().zip(&sizes).enumerate() {
                // Hash table sizes are kept to multiples of 4 so they get past the shape check.
                let size = if n % 2 == 0 { size & !3 } else { *size };
                bytes.extend_from_slice(&offset.to_le_bytes());
                bytes.extend_from_slice(&size.to_le_bytes());
            }
            bytes.extend_from_slice(&offsets[4].to_le_bytes());
            bytes.extend_from_slice(&tail);
            exercise(&bytes);
        }

        #[test]
        fn mutated_valid_images_never_panic(
            spec in tree_specs(),
            edits in proptest::collection::vec((any::<usize>(), any::<u8>()), 1..6),
        ) {
            let mut bytes = build(&spec, 0, &IDENTITY).bytes;
            for (index, value) in edits {
                let at = index % bytes.len();
                bytes[at] = value;
            }
            exercise(&bytes);
        }
    }

    /// Parses `bytes` and, if that works, uses every public method on whatever it can reach.
    fn exercise(bytes: &[u8]) {
        let Ok(mut fs) = open(bytes) else {
            return;
        };
        let root = fs.root().clone();
        let _ = fs.read_dir(&root);
        for id in (0..0x80).step_by(4) {
            let _ = fs.dir(id);
            let _ = fs.file(id);
        }
        let _ = fs.find_dir("/");
        for path in ["/a", "/a/b", "/n0", "/n0/n1", "/n0n", "/x/y/z"] {
            let _ = fs.find_dir(path);
            let _ = fs.find_file(path);
        }
        if let Ok(tree) = fs.walk() {
            for (path, entry) in tree.files {
                let _ = fs.find_file(&path);
                let _ = fs.file_extent(&entry);
                let _ = fs.read_file(&entry);
                let _ = fs.read_file_at(&entry, 1, &mut [0u8; 8]);
            }
            for (path, _) in tree.dirs {
                let _ = fs.find_dir(&path);
            }
        }
    }
}
