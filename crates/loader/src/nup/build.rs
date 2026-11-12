//! Builds NUP containers in memory, for tests (feature `test-support`).
//!
//! A test describes packages as trees of files; [`build`] lays them out the way the reference
//! packer does and returns the bytes with a [`Layout`] that says where every record and extent
//! went, so a test can corrupt one field and call [`rehash_toc`] to get past the hash check.
//!
//! Ported from `nup/tools/nup/nup/pack.py` and `discover.py` @ a4840f06 (MIT, Copyright (c) 2026
//! a55uka; the MIT notice is in `nup/LICENSE`): `Tree.assign` (directories numbered in pre-order,
//! files in first-reference order, each directory's child slots contiguous and in name order),
//! `Tree.intern` (names stored once in the pool), `_locate_content_dirs` (a package's `exefs`
//! and `romfs` are root children of those names that hold at least one file), `_derive_uuid`
//! (SPEC §2.1), `_write_extents` (extents 16-aligned with zero padding, identical content stored
//! once), `_write_meta_blocks` (meta blocks after the extents) and `_build_toc`. Unlike the
//! packer, the codec of each file is the test's choice, not the smaller of the two, and two files
//! share an extent only when both their content and their codec match.
//!
//! The packer compresses with the `zstandard` module, which embeds the content size in each
//! frame (SPEC §4.1). `ruzstd`'s encoder does not, so [`zstd_frame`] adds the field to the frame
//! header it writes (RFC 8878, section 3.1.1.1).

use std::collections::BTreeMap;

use ruzstd::encoding::{CompressionLevel, compress_to_vec};
use sha2::{Digest, Sha256};

use super::layout::{
    CHILD_SLOT_SIZE, Codec, DIR_RECORD_SIZE, EXTENT_ALIGN, FILE_RECORD_SIZE, MAGIC, NO_DIR,
    PACKAGE_RECORD_SIZE, PackageKind, RawDir, RawFile, RawPackage, RawSuperblock, RawTocHeader,
    SHA256_SIZE, SUPERBLOCK_SIZE, TOC_HEADER_SIZE, TOC_MAGIC,
};
use super::to_u64;

/// The name of the root child that holds a package's `ExeFS` (SPEC §3.1.3).
pub const EXEFS_DIR_NAME: &str = "exefs";
/// The name of the root child that holds a package's `RomFS` (SPEC §3.1.3).
pub const ROMFS_DIR_NAME: &str = "romfs";

/// Why a container could not be built.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum BuildError {
    /// A count or offset does not fit its field.
    #[error("{0} does not fit its field")]
    TooLarge(&'static str),
    /// A path names a file where a directory is needed, or the reverse.
    #[error("`{0}` is both a file and a directory")]
    PathConflict(String),
    /// The zstd encoder wrote a frame header this builder does not know how to extend.
    #[error("unexpected zstd frame header")]
    UnexpectedFrame,
    /// The bytes are too short to hold a superblock and the table of contents it describes.
    #[error("not a container this builder can rehash")]
    NotAContainer,
}

/// A file to store.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Content {
    /// The file's bytes.
    pub data: Vec<u8>,
    /// How to store them. An empty file is always stored raw with no extent (SPEC §3.6).
    pub codec: Codec,
    /// Stored bytes to write instead of encoding `data` (for corrupt extents); the record keeps
    /// `data`'s size and hash.
    pub stored: Option<Vec<u8>>,
}

impl Content {
    /// `data`, stored raw.
    pub fn raw(data: impl Into<Vec<u8>>) -> Self {
        Self {
            data: data.into(),
            codec: Codec::Raw,
            stored: None,
        }
    }

    /// `data`, stored as a zstd frame.
    pub fn zstd(data: impl Into<Vec<u8>>) -> Self {
        Self {
            data: data.into(),
            codec: Codec::Zstd,
            stored: None,
        }
    }

    /// The same content with `stored` written as its extent.
    #[must_use]
    pub fn with_stored(mut self, stored: impl Into<Vec<u8>>) -> Self {
        self.stored = Some(stored.into());
        self
    }
}

/// An entry of a [`Tree`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Entry {
    /// A subdirectory.
    Dir(Tree),
    /// A file.
    File(Content),
}

/// A directory: entries by name, in bytewise order (SPEC §3.3).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Tree {
    /// The entries.
    pub entries: BTreeMap<Vec<u8>, Entry>,
}

impl Tree {
    /// An empty directory.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds `content` at the `/`-separated `path`, making the directories on the way.
    pub fn with_file(mut self, path: &str, content: Content) -> Result<Self, BuildError> {
        self.insert(path, Entry::File(content))?;
        Ok(self)
    }

    /// Adds an empty directory at `path`.
    pub fn with_dir(mut self, path: &str) -> Result<Self, BuildError> {
        self.insert(path, Entry::Dir(Tree::new()))?;
        Ok(self)
    }

    fn insert(&mut self, path: &str, entry: Entry) -> Result<(), BuildError> {
        let mut parts: Vec<&str> = path.split('/').filter(|p| !p.is_empty()).collect();
        let last = parts
            .pop()
            .ok_or_else(|| BuildError::PathConflict(path.to_owned()))?;
        let mut dir = self;
        for part in parts {
            let next = dir
                .entries
                .entry(part.as_bytes().to_vec())
                .or_insert_with(|| Entry::Dir(Tree::new()));
            dir = match next {
                Entry::Dir(sub) => sub,
                Entry::File(_) => return Err(BuildError::PathConflict(path.to_owned())),
            };
        }
        dir.entries.insert(last.as_bytes().to_vec(), entry);
        Ok(())
    }

    /// Whether any file lies below this directory.
    fn holds_a_file(&self) -> bool {
        self.entries.values().any(|entry| match entry {
            Entry::File(_) => true,
            Entry::Dir(sub) => sub.holds_a_file(),
        })
    }
}

/// A package to store.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackageSpec {
    /// Its title id.
    pub title_id: u64,
    /// Its kind.
    pub kind: PackageKind,
    /// Its version.
    pub version: u32,
    /// Its flags byte.
    pub flags: u8,
    /// Its tree; root children called `exefs` and `romfs` become its content directories.
    pub tree: Tree,
    /// Its meta block (empty for none).
    pub meta: Vec<u8>,
}

impl PackageSpec {
    /// A package with no flags and no meta block.
    pub fn new(kind: PackageKind, title_id: u64, version: u32, tree: Tree) -> Self {
        Self {
            title_id,
            kind,
            version,
            flags: 0,
            tree,
            meta: Vec::new(),
        }
    }
}

/// Where [`build`] put things, as absolute offsets into the bytes.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Layout {
    /// Start of the table of contents.
    pub toc_offset: usize,
    /// Size of the table of contents.
    pub toc_size: usize,
    /// Start of the blob region.
    pub blob_offset: usize,
    /// Each package record.
    pub package_records: Vec<usize>,
    /// Each directory record.
    pub dir_records: Vec<usize>,
    /// Each file record.
    pub file_records: Vec<usize>,
    /// The children array.
    pub children_offset: usize,
    /// The string pool.
    pub pool_offset: usize,
    /// Each file's extent as `(offset, size)` (`(0, 0)` for an empty file).
    pub extents: Vec<(usize, usize)>,
    /// Each package's meta block as `(offset, size)`, if it has one.
    pub meta: Vec<Option<(usize, usize)>>,
    /// Each file's index, by package index and `/`-separated path from the package root.
    pub files: BTreeMap<(usize, String), usize>,
    /// Each directory's index, by package index and path (`""` is the root).
    pub dirs: BTreeMap<(usize, String), usize>,
}

/// A built container.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Built {
    /// The container's bytes.
    pub bytes: Vec<u8>,
    /// Where everything is.
    pub layout: Layout,
}

/// A child slot before node ids are final.
#[derive(Debug, Clone, Copy)]
enum Slot {
    Dir(usize),
    File(usize),
}

/// The tree numbered the way `Tree.assign` numbers it.
#[derive(Default)]
struct Numbered<'a> {
    /// `(name, first_child, child_count)` per directory.
    dirs: Vec<(&'a [u8], usize, usize)>,
    /// `(name, content)` per file.
    files: Vec<(&'a [u8], &'a Content)>,
    children: Vec<Slot>,
    paths: Vec<((usize, String), Slot)>,
}

impl<'a> Numbered<'a> {
    /// Numbers `dir` (called `name`) and everything below it; returns its index.
    fn assign(&mut self, package: usize, name: &'a [u8], path: &str, dir: &'a Tree) -> usize {
        let index = self.dirs.len();
        self.dirs.push((name, 0, 0));
        self.paths
            .push(((package, path.to_owned()), Slot::Dir(index)));
        let first = self.children.len();
        // Files take their numbers first, then each subdirectory in turn (`Tree.assign`).
        for (child, entry) in &dir.entries {
            let slot = match entry {
                Entry::File(content) => {
                    let file = self.files.len();
                    self.files.push((child, content));
                    let child_path = join(path, child);
                    self.paths.push(((package, child_path), Slot::File(file)));
                    Slot::File(file)
                }
                Entry::Dir(_) => Slot::Dir(usize::MAX),
            };
            self.children.push(slot);
        }
        for (position, (child, entry)) in dir.entries.iter().enumerate() {
            if let Entry::Dir(sub) = entry {
                let sub_index = self.assign(package, child, &join(path, child), sub);
                if let Some(slot) = self.children.get_mut(first + position) {
                    *slot = Slot::Dir(sub_index);
                }
            }
        }
        if let Some(record) = self.dirs.get_mut(index) {
            record.1 = first;
            record.2 = dir.entries.len();
        }
        index
    }
}

/// `path/child`, or `child` at the root.
fn join(path: &str, child: &[u8]) -> String {
    let child = String::from_utf8_lossy(child);
    if path.is_empty() {
        child.into_owned()
    } else {
        format!("{path}/{child}")
    }
}

/// `value` rounded up to the extent alignment.
fn align(value: usize) -> usize {
    let align = usize::try_from(EXTENT_ALIGN).unwrap_or(1);
    value.div_ceil(align).saturating_mul(align)
}

fn u32_of(value: usize, what: &'static str) -> Result<u32, BuildError> {
    u32::try_from(value).map_err(|_| BuildError::TooLarge(what))
}

/// What identifies stored bytes for sharing: content, codec and any override.
type ExtentKey = (Vec<u8>, Codec, Option<Vec<u8>>);

/// Lays out `packages` as a NUP container (see the module docs).
pub fn build(packages: &[PackageSpec]) -> Result<Built, BuildError> {
    build_with_header(packages, SUPERBLOCK_SIZE)
}

/// Like [`build`], with a superblock that declares `header_size` bytes: the bytes after the
/// 0x80 the format defines are zero and the table of contents starts at `header_size` (SPEC §2,
/// rules 2 and 3). A `header_size` below 0x80 cannot hold the superblock and is refused as
/// [`BuildError::NotAContainer`].
pub fn build_with_header(
    packages: &[PackageSpec],
    header_size: usize,
) -> Result<Built, BuildError> {
    if header_size < SUPERBLOCK_SIZE {
        return Err(BuildError::NotAContainer);
    }
    let declared = u16::try_from(header_size).map_err(|_| BuildError::TooLarge("header"))?;
    let mut numbered = Numbered::default();
    let roots: Vec<usize> = packages
        .iter()
        .enumerate()
        .map(|(index, package)| numbered.assign(index, b"", "", &package.tree))
        .collect();

    // The string pool, each name once, directories' names first (`_pack_container`).
    let mut pool: Vec<u8> = Vec::new();
    let mut interned: BTreeMap<&[u8], usize> = BTreeMap::new();
    let mut intern = |name| -> usize {
        *interned.entry(name).or_insert_with(|| {
            let at = pool.len();
            pool.extend_from_slice(name);
            at
        })
    };
    let dir_names: Vec<usize> = numbered.dirs.iter().map(|d| intern(d.0)).collect();
    let file_names: Vec<usize> = numbered.files.iter().map(|f| intern(f.0)).collect();
    drop(intern);

    let toc_size = TOC_HEADER_SIZE
        + packages.len() * PACKAGE_RECORD_SIZE
        + numbered.dirs.len() * DIR_RECORD_SIZE
        + numbered.files.len() * FILE_RECORD_SIZE
        + numbered.children.len() * CHILD_SLOT_SIZE
        + pool.len();
    let toc_offset = header_size;
    let blob_offset = align(toc_offset + toc_size);

    let mut bytes = vec![0u8; blob_offset];
    let extents = write_extents(&mut bytes, &numbered.files)?;
    // Meta blocks after the extents (`_write_meta_blocks`).
    let mut meta = Vec::with_capacity(packages.len());
    for package in packages {
        if package.meta.is_empty() {
            meta.push(None);
        } else {
            meta.push(Some((bytes.len(), package.meta.len())));
            bytes.extend_from_slice(&package.meta);
        }
    }

    let mut toc = Vec::with_capacity(toc_size);
    toc.extend_from_slice(
        &RawTocHeader {
            magic: TOC_MAGIC,
            package_count: u32_of(packages.len(), "package count")?,
            dir_count: to_u64(numbered.dirs.len()),
            file_count: to_u64(numbered.files.len()),
            string_pool_offset: to_u64(toc_size - pool.len()),
            string_pool_size: to_u64(pool.len()),
            reserved: 0,
        }
        .encode(),
    );
    let mut layout = Layout {
        toc_offset,
        toc_size,
        blob_offset,
        extents,
        meta,
        ..Layout::default()
    };
    let uuid_input = write_packages(&mut toc, &mut layout, packages, &roots, &numbered)?;
    write_nodes(&mut toc, &mut layout, &numbered, &dir_names, &file_names)?;
    layout.pool_offset = toc_offset + toc.len();
    toc.extend_from_slice(&pool);

    let superblock = RawSuperblock {
        magic: MAGIC,
        header_size: declared,
        header_version: 1,
        toc_offset: to_u64(toc_offset),
        toc_size: to_u64(toc_size),
        toc_sha256: sha256(&toc),
        blob_offset: to_u64(blob_offset),
        file_size: to_u64(bytes.len()),
        package_count: u32_of(packages.len(), "package count")?,
        flags: 0,
        uuid: first_16(&sha256(&uuid_input)),
        reserved: [0; 32],
    };
    if let Some(window) = bytes.get_mut(..SUPERBLOCK_SIZE) {
        window.copy_from_slice(&superblock.encode());
    }
    if let Some(window) = bytes.get_mut(toc_offset..toc_offset + toc_size) {
        window.copy_from_slice(&toc);
    }
    for ((package, path), slot) in numbered.paths {
        match slot {
            Slot::File(file) => layout.files.insert((package, path), file),
            Slot::Dir(dir) => layout.dirs.insert((package, path), dir),
        };
    }
    Ok(Built { bytes, layout })
}

/// Appends the files' extents to `bytes` in file order, 16-aligned, identical content and codec
/// stored once (`_write_extents`); returns each file's `(offset, size)`.
fn write_extents(
    bytes: &mut Vec<u8>,
    files: &[(&[u8], &Content)],
) -> Result<Vec<(usize, usize)>, BuildError> {
    let mut extents = Vec::with_capacity(files.len());
    let mut stored_once: BTreeMap<ExtentKey, (usize, usize)> = BTreeMap::new();
    for (_, content) in files {
        if content.data.is_empty() && content.stored.is_none() {
            extents.push((0, 0));
            continue;
        }
        let key = (content.data.clone(), content.codec, content.stored.clone());
        if let Some(&extent) = stored_once.get(&key) {
            extents.push(extent);
            continue;
        }
        let payload = match (&content.stored, content.codec) {
            (Some(stored), _) => stored.clone(),
            (None, Codec::Raw) => content.data.clone(),
            (None, Codec::Zstd) => zstd_frame(&content.data)?,
        };
        let at = align(bytes.len());
        bytes.resize(at, 0);
        bytes.extend_from_slice(&payload);
        let extent = (at, payload.len());
        stored_once.insert(key, extent);
        extents.push(extent);
    }
    Ok(extents)
}

/// Appends the package records to `toc` (`_build_toc`) and returns the input of the UUID hash
/// (SPEC §2.1: `title_id`, `version`, `kind` of each package in order).
fn write_packages(
    toc: &mut Vec<u8>,
    layout: &mut Layout,
    packages: &[PackageSpec],
    roots: &[usize],
    numbered: &Numbered<'_>,
) -> Result<Vec<u8>, BuildError> {
    let mut uuid_input = Vec::new();
    for ((index, package), &root) in packages.iter().enumerate().zip(roots) {
        // `_locate_content_dirs`: a root child of that name holding at least one file.
        let content_dir = |name: &str| -> Result<u32, BuildError> {
            let Some(Entry::Dir(tree)) = package.tree.entries.get(name.as_bytes()) else {
                return Ok(NO_DIR);
            };
            if !tree.holds_a_file() {
                return Ok(NO_DIR);
            }
            let dir = numbered
                .paths
                .iter()
                .find_map(|((p, path), slot)| match slot {
                    Slot::Dir(d) if *p == index && path == name => Some(*d),
                    _ => None,
                })
                .unwrap_or(usize::MAX);
            u32_of(dir, "content directory")
        };
        let (meta_offset, meta_size) = layout.meta.get(index).copied().flatten().unwrap_or((0, 0));
        layout.package_records.push(layout.toc_offset + toc.len());
        toc.extend_from_slice(
            &RawPackage {
                title_id: package.title_id,
                root_dir: u32_of(root, "root directory")?,
                exefs_dir: content_dir(EXEFS_DIR_NAME)?,
                romfs_dir: content_dir(ROMFS_DIR_NAME)?,
                version: package.version,
                kind: package.kind.to_byte(),
                flags: package.flags,
                reserved: 0,
                reserved2: 0,
                meta_offset: to_u64(meta_offset),
                meta_size: u32_of(meta_size, "meta size")?,
                reserved3: [0; 20],
            }
            .encode(),
        );
        uuid_input.extend_from_slice(&package.title_id.to_le_bytes());
        uuid_input.extend_from_slice(&package.version.to_le_bytes());
        uuid_input.push(package.kind.to_byte());
    }
    Ok(uuid_input)
}

/// Appends the directory records, the file records and the children array to `toc`
/// (`_build_toc`; a file's node id is the directory count plus its index, SPEC §3.3).
fn write_nodes(
    toc: &mut Vec<u8>,
    layout: &mut Layout,
    numbered: &Numbered<'_>,
    dir_names: &[usize],
    file_names: &[usize],
) -> Result<(), BuildError> {
    for (&(name, first, count), &name_at) in numbered.dirs.iter().zip(dir_names) {
        layout.dir_records.push(layout.toc_offset + toc.len());
        toc.extend_from_slice(
            &RawDir {
                name_offset: u32_of(name_at, "name offset")?,
                name_len: u32_of(name.len(), "name length")?,
                first_child: u32_of(first, "first child")?,
                child_count: u32_of(count, "child count")?,
                reserved: 0,
            }
            .encode(),
        );
    }
    for ((&(name, content), &name_at), &(offset, size)) in
        numbered.files.iter().zip(file_names).zip(&layout.extents)
    {
        layout.file_records.push(layout.toc_offset + toc.len());
        let codec = if size == 0 { Codec::Raw } else { content.codec };
        toc.extend_from_slice(
            &RawFile {
                name_offset: u32_of(name_at, "name offset")?,
                name_len: u32_of(name.len(), "name length")?,
                size: to_u64(content.data.len()),
                extent_offset: to_u64(offset),
                extent_size: to_u64(size),
                sha256: sha256(&content.data),
                codec: codec.to_byte(),
                codec_flags: 0,
                reserved2: 0,
                reserved3: [0; 12],
            }
            .encode(),
        );
    }
    layout.children_offset = layout.toc_offset + toc.len();
    for slot in &numbered.children {
        let id = match *slot {
            Slot::Dir(dir) => dir,
            Slot::File(file) => numbered.dirs.len() + file,
        };
        toc.extend_from_slice(&u32_of(id, "node id")?.to_le_bytes());
    }
    Ok(())
}

/// The SHA-256 of `data`.
pub fn sha256(data: &[u8]) -> [u8; SHA256_SIZE] {
    let mut out = [0u8; SHA256_SIZE];
    out.copy_from_slice(Sha256::digest(data).as_slice());
    out
}

fn first_16(digest: &[u8; SHA256_SIZE]) -> [u8; 16] {
    let mut out = [0u8; 16];
    out.copy_from_slice(digest.get(..16).unwrap_or(&[0; 16]));
    out
}

/// Recomputes the superblock's table of contents hash after a test changed the table.
pub fn rehash_toc(bytes: &mut [u8]) -> Result<(), BuildError> {
    let superblock = RawSuperblock::decode(bytes).map_err(|_| BuildError::NotAContainer)?;
    let start = usize::try_from(superblock.toc_offset).map_err(|_| BuildError::NotAContainer)?;
    let size = usize::try_from(superblock.toc_size).map_err(|_| BuildError::NotAContainer)?;
    let toc = bytes
        .get(start..start.saturating_add(size))
        .ok_or(BuildError::NotAContainer)?;
    let rehashed = RawSuperblock {
        toc_sha256: sha256(toc),
        ..superblock
    };
    let head = bytes
        .get_mut(..SUPERBLOCK_SIZE)
        .ok_or(BuildError::NotAContainer)?;
    head.copy_from_slice(&rehashed.encode());
    Ok(())
}

impl Built {
    /// Rewrites the superblock through `edit`. The table of contents hash is not recomputed, so
    /// an edit of `toc_sha256` stays as written.
    pub fn edit_superblock(&mut self, edit: impl FnOnce(&mut RawSuperblock)) {
        if let Ok(mut superblock) = RawSuperblock::decode(&self.bytes) {
            edit(&mut superblock);
            self.write(0, &superblock.encode());
        }
    }

    /// Rewrites the table of contents header through `edit`, then rehashes the table.
    pub fn edit_toc_header(&mut self, edit: impl FnOnce(&mut RawTocHeader)) {
        let at = self.layout.toc_offset;
        if let Some(mut header) = self
            .bytes
            .get(at..)
            .and_then(|b| RawTocHeader::decode(b).ok())
        {
            edit(&mut header);
            self.write(at, &header.encode());
        }
        self.rehash();
    }

    /// Rewrites package record `index` through `edit`, then rehashes the table of contents.
    pub fn edit_package(&mut self, index: usize, edit: impl FnOnce(&mut RawPackage)) {
        if let Some(&at) = self.layout.package_records.get(index)
            && let Some(mut record) = self
                .bytes
                .get(at..)
                .and_then(|b| RawPackage::decode(b).ok())
        {
            edit(&mut record);
            self.write(at, &record.encode());
        }
        self.rehash();
    }

    /// Rewrites directory record `index` through `edit`, then rehashes the table of contents.
    pub fn edit_dir(&mut self, index: usize, edit: impl FnOnce(&mut RawDir)) {
        if let Some(&at) = self.layout.dir_records.get(index)
            && let Some(mut record) = self.bytes.get(at..).and_then(|b| RawDir::decode(b).ok())
        {
            edit(&mut record);
            self.write(at, &record.encode());
        }
        self.rehash();
    }

    /// Rewrites file record `index` through `edit`, then rehashes the table of contents.
    pub fn edit_file(&mut self, index: usize, edit: impl FnOnce(&mut RawFile)) {
        if let Some(&at) = self.layout.file_records.get(index)
            && let Some(mut record) = self.bytes.get(at..).and_then(|b| RawFile::decode(b).ok())
        {
            edit(&mut record);
            self.write(at, &record.encode());
        }
        self.rehash();
    }

    /// Sets child slot `slot` to `node`, then rehashes the table of contents.
    pub fn set_child(&mut self, slot: usize, node: u32) {
        let at = self.layout.children_offset + slot * CHILD_SLOT_SIZE;
        self.write(at, &node.to_le_bytes());
        self.rehash();
    }

    /// The index of the file at `path` in package `package`.
    pub fn file_index(&self, package: usize, path: &str) -> Option<usize> {
        self.layout.files.get(&(package, path.to_owned())).copied()
    }

    /// The index of the directory at `path` in package `package` (`""` is the root).
    pub fn dir_index(&self, package: usize, path: &str) -> Option<usize> {
        self.layout.dirs.get(&(package, path.to_owned())).copied()
    }

    fn write(&mut self, at: usize, data: &[u8]) {
        if let Some(window) = self.bytes.get_mut(at..at.saturating_add(data.len())) {
            window.copy_from_slice(data);
        }
    }

    fn rehash(&mut self) {
        // A container this builder made always rehashes; a test that broke the superblock's
        // table of contents fields first gets the bytes it asked for.
        let _ = rehash_toc(&mut self.bytes);
    }
}

/// Frame header descriptor bits (RFC 8878, section 3.1.1.1.1): `Frame_Content_Size_flag` in bits
/// 7-6, `Single_Segment_flag` bit 5, `Dictionary_ID_flag` bits 1-0.
const FCS_FLAG_MASK: u8 = 0b1100_0000;
pub(super) const SINGLE_SEGMENT_FLAG: u8 = 0b0010_0000;
const DICTIONARY_ID_MASK: u8 = 0b0000_0011;
/// `Frame_Content_Size_flag` 3: an 8-byte content size field.
pub(super) const FCS_FLAG_8_BYTES: u8 = 0b1100_0000;
/// Offset of the frame header descriptor, after the 4-byte magic.
const DESCRIPTOR_OFFSET: usize = 4;

/// `data` as one zstd frame that embeds its content size (see the module docs).
pub fn zstd_frame(data: &[u8]) -> Result<Vec<u8>, BuildError> {
    let frame = compress_to_vec(data, CompressionLevel::Fastest);
    let descriptor = *frame
        .get(DESCRIPTOR_OFFSET)
        .ok_or(BuildError::UnexpectedFrame)?;
    // `ruzstd` writes no content size, no dictionary id and a window descriptor; anything else
    // would need a different splice.
    if descriptor & (FCS_FLAG_MASK | SINGLE_SEGMENT_FLAG | DICTIONARY_ID_MASK) != 0 {
        return Err(BuildError::UnexpectedFrame);
    }
    let window_end = DESCRIPTOR_OFFSET + 2;
    let mut out = Vec::with_capacity(frame.len() + 8);
    out.extend_from_slice(frame.get(..DESCRIPTOR_OFFSET).unwrap_or_default());
    out.push(descriptor | FCS_FLAG_8_BYTES);
    out.extend_from_slice(
        frame
            .get(DESCRIPTOR_OFFSET + 1..window_end)
            .unwrap_or_default(),
    );
    out.extend_from_slice(&to_u64(data.len()).to_le_bytes());
    out.extend_from_slice(frame.get(window_end..).unwrap_or_default());
    Ok(out)
}
