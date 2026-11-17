//! Building a `RomFS` image from a directory tree, as the console's file-system service serves
//! a title's data storage (`OpenDataStorageByCurrentProcess`) and the title's own file-system
//! library then reads it.
//!
//! Ported from Eden's `RomFSBuildContext` (`eden/src/core/file_sys/fsmitm_romfsbuild.cpp` @
//! cf654052, GPL-2.0-or-later, which this project's GPL-3.0-or-later may carry), the builder
//! Eden-NX2 uses to serve a loose `RomFS` directory as a `RomFS` image
//! (`eden-nx2/src/core/file_sys/romfs.cpp` `CreateRomFS`, called from `FindLooseRomFS` in
//! `eden-nx2/src/core/loader/deconstructed_rom_directory.cpp`, @ 50fa86bb). Paths are relative to
//! the reference folder (`docs/references.md`).
//!
//! What [`build`] does, step by step as `RomFSBuildContext` does it:
//!
//! - Every directory and file has a path: `/` and its name after its parent's path, the root's
//!   path empty (`VisitDirectory`). A path must be shorter than `FS_MAX_PATH` (0x301), which
//!   Eden asserts (`ASSERT(child->path_len < FS_MAX_PATH)`); here a longer one is
//!   [`BuildError::PathTooLong`], with the limit [`MAX_PATH_LEN`] (0x300).
//! - Directories and files are each sorted by path, bytewise (`std::sort` on `std::string`),
//!   which is not a depth-first walk of sibling names: `.` (0x2E) sorts below `/` (0x2F), so
//!   `/a.b` comes between `/a` and `/a/c`. Pinned by
//!   `entries_sort_by_whole_path_bytewise_not_depth_first`.
//!   Entry offsets are given in that order, each entry 0x18 (a directory) or 0x20 (a file) bytes
//!   and its name padded to 4 bytes ([`TABLE_ALIGNMENT`]); the root sorts first, at offset 0.
//! - File data is laid out in the same order from [`FILE_PARTITION_OFFSET`], each file's start
//!   aligned to [`FILE_ALIGNMENT`] bytes, the gaps zero (`ConcatenatedVfsFile`'s filler byte is 0
//!   in `CreateRomFS`).
//! - A directory's files and subdirectories are linked in path order through `sibling`, headed by
//!   its `file` and `child` fields; an empty link is [`ENTRY_EMPTY`].
//! - Each table's hash buckets number [`hash_table_count`] of its entries
//!   (`romfs_get_hash_table_count`); an entry goes at the head of bucket
//!   `path_hash(parent, name) % count`, its `hash` field the previous head. Entries are added in
//!   path order, so a bucket's chain runs from the last added back to the first.
//! - The metadata follows the file data, at the next 4-byte boundary: directory buckets,
//!   directory entries, file buckets, file entries, one after the other, and the header points at
//!   each.
//!
//! # Divergences
//!
//! - **The name hash is over unsigned bytes** ([`path_hash`]). Eden's builder XORs `char`,
//!   which is signed on some hosts, so a name with a byte of 0x80 or above hashes differently
//!   there. The console's file-system library, which reads the image, hashes unsigned bytes
//!   (Atmosphère `fs_dbm_hierarchical_rom_file_table.hpp`, `EntryKey::Hash`, casts each
//!   character to its unsigned type; fact only), and so does libnx's reader
//!   (`libnx/nx/source/runtime/devices/romfs_dev.c`, `calcHash`, over `uint8_t`). Pinned by
//!   `a_name_with_high_bytes_hashes_as_the_console_reads_it`.
//! - **A path limit is an error, not an assertion** (see above). Pinned by
//!   `a_path_at_the_limit_builds_and_one_past_it_does_not`.
//! - **The metadata is bounded** by [`MAX_METADATA_SIZE`], a starquake limit checked before
//!   anything is allocated, so a tree with millions of long names cannot make the builder
//!   allocate gigabytes ([`BuildError::MetadataTooLarge`]). Eden has no limit. Pinned by
//!   `the_metadata_limit_holds_at_its_edge`.
//! - **Ex-FS layering** (Eden's `ext` directory of `.stub` and `.ips` files) is not ported:
//!   nothing here patches a title's files.
//!
//! Every directory's parent must come before it in the list, and the first directory is the
//! root, so the input is a tree by construction ([`BuildError::BadParent`] otherwise).

use super::{DIR_ENTRY_SIZE, ENTRY_EMPTY, FILE_ENTRY_SIZE, HEADER_SIZE, MAX_PATH_LEN, path_hash};

/// Where file data starts (`ROMFS_FILEPARTITION_OFS`).
pub const FILE_PARTITION_OFFSET: u64 = 0x200;
/// Each file's data starts at a multiple of this (`Common::AlignUp(file_partition_size, 16)`).
pub const FILE_ALIGNMENT: u64 = 16;
/// Names are padded, and the metadata placed, to a multiple of this (`Common::AlignUp(.., 4)`).
pub const TABLE_ALIGNMENT: u64 = 4;
/// The largest metadata (both hash tables and both entry tables) [`build`] makes: 256 MiB. A
/// starquake limit, not the format's: at 0x20 bytes and a short name per file it leaves room for
/// millions of files, and it keeps every entry offset inside the `u32` the format stores.
pub const MAX_METADATA_SIZE: u64 = 256 << 20;
/// The size of a hash bucket.
pub(crate) const BUCKET_SIZE: u64 = 4;

/// The number of hash buckets for `entries` entries (`romfs_get_hash_table_count`): at least 3,
/// odd below 19, and otherwise the first count from `entries` on that no prime up to 17
/// divides.
pub fn hash_table_count(entries: u64) -> u64 {
    if entries < 3 {
        return 3;
    }
    if entries < 19 {
        return entries | 1;
    }
    let mut count = entries;
    while [2, 3, 5, 7, 11, 13, 17]
        .iter()
        .any(|prime| count.is_multiple_of(*prime))
    {
        count = count.saturating_add(1);
    }
    count
}

/// A directory of the tree to build: the index of its parent (the root's is ignored) and its
/// name (the root's is ignored too: the root has no name).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DirIn<'a> {
    /// The index of the parent directory, which must be smaller than this one's.
    pub parent: usize,
    /// The name.
    pub name: &'a [u8],
}

/// A file of the tree to build.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FileIn<'a, K> {
    /// The index of the directory it is in.
    pub parent: usize,
    /// The name.
    pub name: &'a [u8],
    /// Its size in bytes.
    pub size: u64,
    /// What the caller knows it by.
    pub key: K,
}

/// Where a file's data lies in the image.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Placed<K> {
    /// The offset from the start of the image.
    pub offset: u64,
    /// Its size in bytes.
    pub size: u64,
    /// The caller's key.
    pub key: K,
}

impl<K> Placed<K> {
    /// The offset just past its data.
    pub fn end(&self) -> u64 {
        // `build` checked that every file ends inside the image.
        self.offset.saturating_add(self.size)
    }
}

/// What lies at an offset of an [`Image`] ([`Image::segment_at`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Segment<'a, K> {
    /// Bytes the image holds itself (the header or the metadata), from the offset on.
    Bytes(&'a [u8]),
    /// This many zero bytes: padding.
    Zeros(u64),
    /// A file's data, from `offset` bytes into it.
    File {
        /// The file.
        placed: &'a Placed<K>,
        /// How far into it.
        offset: u64,
    },
}

/// A built image: its header and metadata, and where each file's data lies. The data itself is
/// the caller's.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Image<K> {
    header: [u8; HEADER_SIZE],
    metadata: Vec<u8>,
    metadata_offset: u64,
    files: Vec<Placed<K>>,
    size: u64,
}

impl<K> Image<K> {
    /// The image's size in bytes.
    pub fn size(&self) -> u64 {
        self.size
    }

    /// The header's bytes.
    pub fn header(&self) -> &[u8; HEADER_SIZE] {
        &self.header
    }

    /// Where the metadata starts, and its bytes.
    pub fn metadata(&self) -> (u64, &[u8]) {
        (self.metadata_offset, &self.metadata)
    }

    /// The files, in the order of their data (which is path order).
    pub fn files(&self) -> &[Placed<K>] {
        &self.files
    }

    /// What lies at `offset`, up to the end of that segment; `None` at or past the end.
    pub fn segment_at(&self, offset: u64) -> Option<Segment<'_, K>> {
        if offset >= self.size {
            return None;
        }
        if let Some(header) = usize::try_from(offset)
            .ok()
            .and_then(|at| self.header.get(at..))
            .filter(|rest| !rest.is_empty())
        {
            return Some(Segment::Bytes(header));
        }
        if let Some(into) = offset.checked_sub(self.metadata_offset) {
            let rest = usize::try_from(into)
                .ok()
                .and_then(|at| self.metadata.get(at..))
                .unwrap_or_default();
            return Some(Segment::Bytes(rest));
        }
        // Files are in offset order and do not overlap, so their ends only grow.
        let next = self.files.partition_point(|file| file.end() <= offset);
        match self.files.get(next) {
            Some(placed) if placed.offset <= offset => Some(Segment::File {
                placed,
                offset: offset.saturating_sub(placed.offset),
            }),
            Some(placed) => Some(Segment::Zeros(
                placed
                    .offset
                    .min(self.metadata_offset)
                    .saturating_sub(offset),
            )),
            None => Some(Segment::Zeros(self.metadata_offset.saturating_sub(offset))),
        }
    }

    /// Fills `buf` with the image's own bytes at `offset` (the header, the metadata and the
    /// padding) and hands each piece of file data to `file`, which gets the file, the offset
    /// into it and the part of `buf` it fills. Returns how many bytes were filled: fewer than
    /// `buf.len()` only at the end of the image. The first error `file` returns is the error.
    pub fn read_with<E>(
        &self,
        offset: u64,
        buf: &mut [u8],
        mut file: impl FnMut(&Placed<K>, u64, &mut [u8]) -> Result<(), E>,
    ) -> Result<usize, E> {
        let mut done = 0usize;
        while let Some(rest) = buf.get_mut(done..).filter(|rest| !rest.is_empty()) {
            let at = u64::try_from(done)
                .ok()
                .and_then(|done| offset.checked_add(done));
            let Some(segment) = at.and_then(|at| self.segment_at(at)) else {
                break;
            };
            let count = match segment {
                Segment::Bytes(bytes) => {
                    let count = bytes.len().min(rest.len());
                    if let (Some(to), Some(from)) = (rest.get_mut(..count), bytes.get(..count)) {
                        to.copy_from_slice(from);
                    }
                    count
                }
                Segment::Zeros(len) => {
                    let count = usize::try_from(len).map_or(rest.len(), |len| len.min(rest.len()));
                    rest.get_mut(..count).unwrap_or_default().fill(0);
                    count
                }
                Segment::File { placed, offset } => {
                    let left = placed.size.saturating_sub(offset);
                    let count =
                        usize::try_from(left).map_or(rest.len(), |left| left.min(rest.len()));
                    file(placed, offset, rest.get_mut(..count).unwrap_or_default())?;
                    count
                }
            };
            if count == 0 {
                break;
            }
            done = done.saturating_add(count);
        }
        Ok(done)
    }
}

/// Which of the tree's two lists an entry is in: [`DirIn`]s or [`FileIn`]s.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntryKind {
    /// A directory.
    Directory,
    /// A file.
    File,
}

impl std::fmt::Display for EntryKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Directory => "directory",
            Self::File => "file",
        })
    }
}

/// Why a tree could not be built into an image.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum BuildError {
    /// The tree has no root (no directories at all).
    #[error("the RomFS tree has no root directory")]
    NoRoot,
    /// A directory's parent does not come before it, or a file's parent is not a directory of
    /// the tree.
    #[error("{kind} {index} of the RomFS tree names parent {parent}, which is not before it")]
    BadParent {
        /// Whether it is a directory or a file.
        kind: EntryKind,
        /// Its index.
        index: usize,
        /// The parent it names.
        parent: usize,
    },
    /// A path is longer than a `RomFS` path may be.
    #[error("a path of the RomFS is {len} bytes, longer than the {MAX_PATH_LEN} a path may be")]
    PathTooLong {
        /// The path's length.
        len: usize,
    },
    /// The metadata would be larger than the limit ([`MAX_METADATA_SIZE`] for [`build`]).
    #[error("the RomFS's tables would be {size} bytes, more than the {limit} this emulator builds")]
    MetadataTooLarge {
        /// The size it would be (saturated at `u64::MAX`).
        size: u64,
        /// The limit.
        limit: u64,
    },
    /// The files add up to more than an image can address.
    #[error("the RomFS's files add up to more bytes than an image can hold")]
    TooLarge,
}

/// `value` rounded up to a multiple of `align` (a power of two), if that fits.
fn align_up(value: u64, align: u64) -> Option<u64> {
    let mask = align.checked_sub(1)?;
    value.checked_add(mask).map(|value| value & !mask)
}

/// An entry's size in its table: the fixed part and the name padded to [`TABLE_ALIGNMENT`].
fn entry_size(fixed: usize, name: &[u8]) -> Option<u64> {
    let name = align_up(u64::try_from(name.len()).ok()?, TABLE_ALIGNMENT)?;
    u64::try_from(fixed).ok()?.checked_add(name)
}

/// A node of the tree, for building paths.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Of {
    Dir(usize),
    File(usize),
}

/// The tree being built, with what [`path_of`](Tree::path_of) needs.
struct Tree<'t, 'a, K> {
    dirs: &'t [DirIn<'a>],
    files: &'t [FileIn<'a, K>],
}

impl<K> Tree<'_, '_, K> {
    /// Writes `node`'s path (`/a/b`; the root's is empty) into `out`, through `chain`, both
    /// reused between calls. Every path was checked against [`MAX_PATH_LEN`], and parents come
    /// before their children, so this ends.
    fn path_of(&self, node: Of, chain: &mut Vec<usize>, out: &mut Vec<u8>) {
        chain.clear();
        out.clear();
        let (mut dir, leaf) = match node {
            Of::Dir(dir) => (dir, None),
            Of::File(file) => match self.files.get(file) {
                Some(entry) => (entry.parent, Some(entry.name)),
                None => return,
            },
        };
        while dir != 0 {
            chain.push(dir);
            dir = self.dirs.get(dir).map_or(0, |d| d.parent);
        }
        for &index in chain.iter().rev() {
            out.push(b'/');
            out.extend_from_slice(self.dirs.get(index).map_or_default(|d| d.name));
        }
        if let Some(name) = leaf {
            out.push(b'/');
            out.extend_from_slice(name);
        }
    }

    /// The indices of `nodes` sorted by path, bytewise, as Eden's `std::sort` sorts them.
    fn sorted(&self, mut nodes: Vec<usize>, of: impl Fn(usize) -> Of) -> Vec<usize> {
        let (mut chain, mut a, mut b) = (Vec::new(), Vec::new(), Vec::new());
        let mut other = Vec::new();
        nodes.sort_by(|&x, &y| {
            self.path_of(of(x), &mut chain, &mut a);
            self.path_of(of(y), &mut other, &mut b);
            a.cmp(&b).then(x.cmp(&y))
        });
        nodes
    }
}

/// An entry offset as the `u32` the format stores. Every offset is below
/// [`MAX_METADATA_SIZE`], which `build` checked, so it fits.
fn link(offset: u64) -> u32 {
    u32::try_from(offset).unwrap_or(ENTRY_EMPTY)
}

/// Builds the image of the tree `dirs` and `files` (see the module docs). `dirs[0]` is the root.
pub fn build<K: Copy>(dirs: &[DirIn<'_>], files: &[FileIn<'_, K>]) -> Result<Image<K>, BuildError> {
    build_within(dirs, files, MAX_METADATA_SIZE)
}

/// [`build`] with a metadata limit of `limit` bytes.
pub(crate) fn build_within<K: Copy>(
    dirs: &[DirIn<'_>],
    files: &[FileIn<'_, K>],
    limit: u64,
) -> Result<Image<K>, BuildError> {
    check_tree(dirs, files)?;
    // Sizes before anything is allocated.
    let tables = Tables::of(dirs, files, limit)?;
    let tree = Tree { dirs, files };
    let placement = Placement::of(&tree, limit)?;
    let links = Links::of(&tree, &placement);
    let metadata = tables.write(&tree, &placement, &links)?;

    let metadata_offset = FILE_PARTITION_OFFSET
        .checked_add(placement.partition)
        .and_then(|end| align_up(end, TABLE_ALIGNMENT))
        .ok_or(BuildError::TooLarge)?;
    let size = metadata_offset
        .checked_add(tables.total)
        .ok_or(BuildError::TooLarge)?;
    let header_fields = [
        to_u64(HEADER_SIZE),
        metadata_offset,
        tables.dir_hash,
        metadata_offset.saturating_add(tables.dir_table_at()),
        tables.dir_table,
        metadata_offset.saturating_add(tables.file_hash_at()),
        tables.file_hash,
        metadata_offset.saturating_add(tables.file_table_at()),
        tables.file_table,
        FILE_PARTITION_OFFSET,
    ];
    let mut header = [0u8; HEADER_SIZE];
    for (field, value) in header.as_chunks_mut::<8>().0.iter_mut().zip(header_fields) {
        field.copy_from_slice(&value.to_le_bytes());
    }
    // Path order is also data order.
    let placed = placement
        .file_order
        .iter()
        .filter_map(|&file| {
            let entry = files.get(file)?;
            Some(Placed {
                offset: FILE_PARTITION_OFFSET.saturating_add(*placement.data.get(file)?),
                size: entry.size,
                key: entry.key,
            })
        })
        .collect();
    Ok(Image {
        header,
        metadata,
        metadata_offset,
        files: placed,
        size,
    })
}

/// `usize` to `u64`; lossless on the hosts starquake builds for, saturating elsewhere.
fn to_u64(value: usize) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}

/// A directory's name; the root's is empty, whatever its record says.
fn dir_name<'a>(dirs: &[DirIn<'a>], dir: usize) -> &'a [u8] {
    if dir == 0 {
        return &[];
    }
    dirs.get(dir).map_or_default(|d| d.name)
}

/// Checks that there is a root, that every parent comes before its child and exists, and that
/// every path is at most [`MAX_PATH_LEN`] bytes: a path is its parent's, a `/` and its name.
fn check_tree<K>(dirs: &[DirIn<'_>], files: &[FileIn<'_, K>]) -> Result<(), BuildError> {
    if dirs.is_empty() {
        return Err(BuildError::NoRoot);
    }
    let mut dir_len = vec![0usize; dirs.len()];
    for (index, dir) in dirs.iter().enumerate().skip(1) {
        let parent = dir_len
            .get(dir.parent)
            .copied()
            .filter(|_| dir.parent < index)
            .ok_or(BuildError::BadParent {
                kind: EntryKind::Directory,
                index,
                parent: dir.parent,
            })?;
        let len = path_len(parent, dir.name)?;
        if let Some(slot) = dir_len.get_mut(index) {
            *slot = len;
        }
    }
    for (index, file) in files.iter().enumerate() {
        let parent = dir_len
            .get(file.parent)
            .copied()
            .ok_or(BuildError::BadParent {
                kind: EntryKind::File,
                index,
                parent: file.parent,
            })?;
        path_len(parent, file.name)?;
    }
    Ok(())
}

/// The four tables' sizes, and where each starts in the metadata: directory buckets, directory
/// entries, file buckets, file entries.
struct Tables {
    dir_buckets: u64,
    file_buckets: u64,
    dir_hash: u64,
    dir_table: u64,
    file_hash: u64,
    file_table: u64,
    total: u64,
}

impl Tables {
    /// The tables of `dirs` and `files`, if they fit in `limit` bytes.
    fn of<K>(dirs: &[DirIn<'_>], files: &[FileIn<'_, K>], limit: u64) -> Result<Self, BuildError> {
        let too_large = |size: Option<u64>| BuildError::MetadataTooLarge {
            size: size.unwrap_or(u64::MAX),
            limit,
        };
        let dir_table =
            sum((0..dirs.len()).map(|dir| entry_size(DIR_ENTRY_SIZE, dir_name(dirs, dir))));
        let file_table = sum(files
            .iter()
            .map(|file| entry_size(FILE_ENTRY_SIZE, file.name)));
        let dir_buckets = hash_table_count(to_u64(dirs.len()));
        let file_buckets = hash_table_count(to_u64(files.len()));
        let dir_hash = dir_buckets.checked_mul(BUCKET_SIZE);
        let file_hash = file_buckets.checked_mul(BUCKET_SIZE);
        let total = sum([dir_hash, dir_table, file_hash, file_table]);
        let (Some(dir_hash), Some(dir_table), Some(file_hash), Some(file_table), Some(total)) =
            (dir_hash, dir_table, file_hash, file_table, total)
        else {
            return Err(too_large(None));
        };
        if total > limit {
            return Err(too_large(Some(total)));
        }
        Ok(Self {
            dir_buckets,
            file_buckets,
            dir_hash,
            dir_table,
            file_hash,
            file_table,
            total,
        })
    }

    fn dir_table_at(&self) -> u64 {
        self.dir_hash
    }

    fn file_hash_at(&self) -> u64 {
        self.dir_table_at().saturating_add(self.dir_table)
    }

    fn file_table_at(&self) -> u64 {
        self.file_hash_at().saturating_add(self.file_hash)
    }

    /// The metadata's bytes: every bucket empty, then each entry added in path order at the head
    /// of its bucket's chain, files first and then directories, as Eden adds them.
    fn write<K>(
        &self,
        tree: &Tree<'_, '_, K>,
        placement: &Placement,
        links: &Links,
    ) -> Result<Vec<u8>, BuildError> {
        let len = usize::try_from(self.total).map_err(|_| BuildError::MetadataTooLarge {
            size: self.total,
            limit: self.total,
        })?;
        let mut out = Table(vec![0u8; len]);
        for bucket in 0..self.dir_buckets {
            out.put_u32(bucket.saturating_mul(BUCKET_SIZE), ENTRY_EMPTY);
        }
        for bucket in 0..self.file_buckets {
            let at = self
                .file_hash_at()
                .saturating_add(bucket.saturating_mul(BUCKET_SIZE));
            out.put_u32(at, ENTRY_EMPTY);
        }
        let entry = |offsets: &[u64], index: usize| offsets.get(index).copied().unwrap_or_default();
        let linked = |links: &[u32], index: usize| links.get(index).copied().unwrap_or(ENTRY_EMPTY);
        for &file in &placement.file_order {
            let Some(record) = tree.files.get(file) else {
                continue;
            };
            let parent = link(entry(&placement.dir_offset, record.parent));
            let own = entry(&placement.file_offset, file);
            let chained = out.chain(
                self.file_hash_at(),
                self.file_buckets,
                parent,
                record.name,
                own,
            );
            let at = self.file_table_at().saturating_add(own);
            out.put_u32(at, parent);
            out.put_u32(at.saturating_add(4), linked(&links.file_sibling, file));
            out.put_u64(at.saturating_add(8), entry(&placement.data, file));
            out.put_u64(at.saturating_add(0x10), record.size);
            out.put_u32(at.saturating_add(0x18), chained);
            out.put_name(at.saturating_add(0x1C), record.name);
        }
        for &dir in &placement.dir_order {
            let parent = if dir == 0 {
                0
            } else {
                let parent = tree.dirs.get(dir).map_or(0, |d| d.parent);
                link(entry(&placement.dir_offset, parent))
            };
            let name = dir_name(tree.dirs, dir);
            let own = entry(&placement.dir_offset, dir);
            let chained = out.chain(0, self.dir_buckets, parent, name, own);
            let at = self.dir_table_at().saturating_add(own);
            out.put_u32(at, parent);
            out.put_u32(at.saturating_add(4), linked(&links.dir_sibling, dir));
            out.put_u32(at.saturating_add(8), linked(&links.first_child, dir));
            out.put_u32(at.saturating_add(0xC), linked(&links.first_file, dir));
            out.put_u32(at.saturating_add(0x10), chained);
            out.put_name(at.saturating_add(0x14), name);
        }
        Ok(out.0)
    }
}

/// Where each entry and each file's data go, in path order.
struct Placement {
    dir_order: Vec<usize>,
    file_order: Vec<usize>,
    /// Each directory's entry offset, by index.
    dir_offset: Vec<u64>,
    /// Each file's entry offset, by index.
    file_offset: Vec<u64>,
    /// Each file's data offset from [`FILE_PARTITION_OFFSET`], by index.
    data: Vec<u64>,
    /// The file data's size, padding included.
    partition: u64,
}

impl Placement {
    /// Entry offsets in path order (the root sorts first: its path is empty), and each file's
    /// data at the next [`FILE_ALIGNMENT`] boundary in the same order.
    fn of<K>(tree: &Tree<'_, '_, K>, limit: u64) -> Result<Self, BuildError> {
        let too_large = BuildError::MetadataTooLarge {
            size: u64::MAX,
            limit,
        };
        let dir_order = tree.sorted((0..tree.dirs.len()).collect(), Of::Dir);
        let file_order = tree.sorted((0..tree.files.len()).collect(), Of::File);
        let mut dir_offset = vec![0u64; tree.dirs.len()];
        let mut next = 0u64;
        for &dir in &dir_order {
            if let Some(slot) = dir_offset.get_mut(dir) {
                *slot = next;
            }
            let size = entry_size(DIR_ENTRY_SIZE, dir_name(tree.dirs, dir));
            next = size
                .and_then(|size| next.checked_add(size))
                .ok_or_else(|| too_large.clone())?;
        }
        let mut file_offset = vec![0u64; tree.files.len()];
        let mut data = vec![0u64; tree.files.len()];
        let mut next = 0u64;
        let mut partition = 0u64;
        for &file in &file_order {
            let Some(record) = tree.files.get(file) else {
                continue;
            };
            if let Some(slot) = file_offset.get_mut(file) {
                *slot = next;
            }
            next = entry_size(FILE_ENTRY_SIZE, record.name)
                .and_then(|size| next.checked_add(size))
                .ok_or_else(|| too_large.clone())?;
            partition = align_up(partition, FILE_ALIGNMENT).ok_or(BuildError::TooLarge)?;
            if let Some(slot) = data.get_mut(file) {
                *slot = partition;
            }
            partition = partition
                .checked_add(record.size)
                .ok_or(BuildError::TooLarge)?;
        }
        Ok(Self {
            dir_order,
            file_order,
            dir_offset,
            file_offset,
            data,
            partition,
        })
    }
}

/// Each directory's first file and first subdirectory, and each entry's next sibling, in path
/// order: built back to front, each entry put at the head of its parent's list.
struct Links {
    first_file: Vec<u32>,
    file_sibling: Vec<u32>,
    first_child: Vec<u32>,
    dir_sibling: Vec<u32>,
}

impl Links {
    fn of<K>(tree: &Tree<'_, '_, K>, placement: &Placement) -> Self {
        let mut links = Self {
            first_file: vec![ENTRY_EMPTY; tree.dirs.len()],
            file_sibling: vec![ENTRY_EMPTY; tree.files.len()],
            first_child: vec![ENTRY_EMPTY; tree.dirs.len()],
            dir_sibling: vec![ENTRY_EMPTY; tree.dirs.len()],
        };
        for &file in placement.file_order.iter().rev() {
            let parent = tree.files.get(file).map_or(0, |f| f.parent);
            let offset = link(placement.file_offset.get(file).copied().unwrap_or_default());
            if let (Some(head), Some(sibling)) = (
                links.first_file.get_mut(parent),
                links.file_sibling.get_mut(file),
            ) {
                *sibling = std::mem::replace(head, offset);
            }
        }
        for &dir in placement.dir_order.iter().rev().filter(|&&dir| dir != 0) {
            let parent = tree.dirs.get(dir).map_or(0, |d| d.parent);
            let offset = link(placement.dir_offset.get(dir).copied().unwrap_or_default());
            if let (Some(head), Some(sibling)) = (
                links.first_child.get_mut(parent),
                links.dir_sibling.get_mut(dir),
            ) {
                *sibling = std::mem::replace(head, offset);
            }
        }
        links
    }
}

/// The metadata being written. Every write lands inside it: `Tables::of` sized it for every
/// entry `Placement::of` placed.
struct Table(Vec<u8>);

impl Table {
    /// Writes `bytes` at `at`.
    fn put(&mut self, at: u64, bytes: &[u8]) {
        let field = usize::try_from(at)
            .ok()
            .and_then(|at| self.0.get_mut(at..at.checked_add(bytes.len())?));
        if let Some(field) = field {
            field.copy_from_slice(bytes);
        }
    }

    fn put_u32(&mut self, at: u64, value: u32) {
        self.put(at, &value.to_le_bytes());
    }

    fn put_u64(&mut self, at: u64, value: u64) {
        self.put(at, &value.to_le_bytes());
    }

    /// Writes a name's length at `at` and the name after it (the padding stays zero).
    fn put_name(&mut self, at: u64, name: &[u8]) {
        self.put_u32(at, u32::try_from(name.len()).unwrap_or(u32::MAX));
        self.put(at.saturating_add(4), name);
    }

    /// The `u32` at `at`.
    fn get_u32(&self, at: u64) -> u32 {
        usize::try_from(at)
            .ok()
            .and_then(|at| self.0.get(at..at.checked_add(4)?))
            .and_then(|field| <[u8; 4]>::try_from(field).ok())
            .map_or(ENTRY_EMPTY, u32::from_le_bytes)
    }

    /// Puts the entry at `own` at the head of the chain of `name` under `parent`, in the
    /// `count` buckets at `buckets`, and returns the entry it pushed down the chain.
    fn chain(&mut self, buckets: u64, count: u64, parent: u32, name: &[u8], own: u64) -> u32 {
        let bucket = u64::from(path_hash(parent, name)) % count;
        let at = buckets.saturating_add(bucket.saturating_mul(BUCKET_SIZE));
        let previous = self.get_u32(at);
        self.put_u32(at, link(own));
        previous
    }
}

/// The sum of `sizes`, if every one is known and the total fits.
fn sum(sizes: impl IntoIterator<Item = Option<u64>>) -> Option<u64> {
    sizes
        .into_iter()
        .try_fold(0u64, |total, size| total.checked_add(size?))
}

/// The length of a path that is `parent` bytes long and gains `/` and `name`, if it is at most
/// [`MAX_PATH_LEN`].
fn path_len(parent: usize, name: &[u8]) -> Result<usize, BuildError> {
    let len = parent.saturating_add(1).saturating_add(name.len());
    if len > MAX_PATH_LEN {
        return Err(BuildError::PathTooLong { len });
    }
    Ok(len)
}

#[cfg(test)]
mod tests;
