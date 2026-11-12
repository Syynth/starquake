//! NUP containers: decrypted Switch 1 and Switch 2 dumps in one file.
//!
//! A NUP file is a superblock, a SHA-256-protected table of contents (packages, a directory
//! tree, file records, a string pool) and a blob region of file extents, each stored raw or as
//! one Zstandard frame. A container holds a base program, its updates and add-ons as separate
//! packages. Nothing in it is encrypted, so reading it needs no keys or firmware.
//!
//! [`Nup::open`] reads the superblock and the table of contents from any `Read + Seek` source
//! and checks them; it never reads the blob region, so a title of tens of gigabytes costs only
//! its table of contents in memory (bounded by [`Limits::max_toc_size`] before anything is
//! allocated). [`Nup::reader`] then streams one file's bytes, decompressing as it goes and
//! checking the size and the SHA-256 at the end; [`Nup::verify`] runs that over every file.
//! [`Nup::title`] layers the newest update over the base ([`title`]).
//!
//! # Sources
//!
//! The format is specified by `nup/SPEC.md` @ a4840f06 (version 1.0-draft, MIT), and its
//! reference tool is `nup/tools/nup/nup/` at the same commit (`docs/references.md`). Record
//! layouts are in [`layout`]. The checks follow the spec's reader conformance rules (§6.1) and,
//! where the spec is silent, the reference reader (`read.py`, `NupReader._parse_toc` and
//! `_check_bounds`; `format.py`, `parse_super`, `parse_pkg` and `parse_file`) and verifier
//! (`verify.py`). Nothing here is ported line by line; the checks are restated in Rust and each
//! cites the rule it implements.
//!
//! What [`Nup::open`] checks, in this order:
//!
//! - The superblock (SPEC §2, rules 1 to 5): magic `NUP` with major version 1, `header_size`
//!   at least 0x80, `toc_offset == header_size`, the table of contents inside the file,
//!   `blob_offset` 16-aligned, not before the end of the table of contents and not past the end
//!   of the file, `file_size` equal to the source's length, at least one package, no flags.
//!   Its 32 reserved bytes are zero (`parse_super`).
//! - The table of contents' SHA-256, before any value in it is used (§6.1 rule 1).
//! - The table of contents: magic `TOC\x01`, the package count equal to the superblock's, the
//!   records inside it, the string pool after the records and inside it, and the children array
//!   (the bytes between the records and the pool, as `_parse_toc` derives it) a whole number of
//!   4-byte slots.
//! - Each package: a known kind (§6.1 rule 4), only flag bit 0, its two reserved fields at 0x1A
//!   and 0x1C zero (`parse_pkg`), its directories in range, and its meta block inside the file.
//!   An update in a container without a base has the standalone flag (§3.1.2; `_check_bounds`).
//! - Each directory and file name inside the string pool; each directory's child range inside
//!   the children array; each file's codec known, its codec flags zero, an empty file with no
//!   extent, a non-empty one with one, and that extent inside the blob region and 16-aligned
//!   (`parse_file`, `_check_bounds`; §4).
//!
//! # Divergences
//!
//! Stricter than the reference reader, each pinned by a test in `nup/tests.rs`:
//!
//! - **One parent per node, every node reachable.** Every node id appears in exactly one place:
//!   as one package's root, or in one child slot. The directory child ranges are disjoint and
//!   cover the children array. Every directory and file is reachable from a package root. The
//!   reference reader checks none of this (its verifier reports overlapping ranges, cycles and
//!   orphans as problems). It is what makes the tree a forest, so that a walk visits each node
//!   once, a path names one file, and a cycle cannot exist. A child slot naming a node past the
//!   last file is an error here; the reference reader would raise an index error.
//!   Tests: `a_node_with_two_parents_is_refused`, `an_unreachable_directory_is_refused`,
//!   `a_child_slot_past_the_last_node_is_refused`, `overlapping_child_ranges_are_refused`.
//! - **Sorted children.** Each directory's children are in strictly increasing bytewise order
//!   of name, so names are unique and lookups can binary search. The spec makes this a producer
//!   rule (§3.3, §6.2 rule 1); the reference reader assumes it (`find_child` binary searches)
//!   and only its verifier checks it. Test: `unsorted_or_duplicate_children_are_refused`.
//! - **At most one base package** (§3.1.2 says so; the reference reader does not check it).
//!   Test: `two_base_packages_are_refused`.
//! - **A package's `ExeFS` and `RomFS` directories lie inside its own tree.** Not in the spec;
//!   without it an update could name a directory of the base. Test:
//!   `a_content_directory_outside_its_package_is_refused`.
//! - **A raw extent is exactly the file's size** at open (§4.1 says `extent_size == size`; the
//!   reference verifier checks it only with `--deep`). Test: `a_raw_extent_of_the_wrong_size_is_refused`.
//! - **No overlaps.** A meta block starts at or after the end of the table of contents and
//!   overlaps no extent and no other meta block; two extents either are the same extent (same
//!   offset and size, shared by dedup, §3.6) or do not overlap; and files that share an extent
//!   agree on size, codec and SHA-256 (§3.6: producers must not share otherwise). §6.1 rule 2
//!   asks for the overlap check without saying which regions; the reference reader checks only
//!   that a meta block ends inside the file, and its verifier checks shared hashes with
//!   `--deep`. Tests: `overlapping_extents_are_refused`, `a_meta_block_over_an_extent_is_refused`,
//!   `files_sharing_an_extent_must_agree`.
//! - **A size limit on the table of contents** ([`DEFAULT_MAX_TOC_SIZE`]), so a corrupt size
//!   cannot make the reader allocate gigabytes. The reference tool maps the whole file instead.
//!   Test: `the_table_of_contents_limit_holds_at_its_edge`.
//!
//! Looser, matching the reference reader: the reserved fields of the table of contents header,
//! of directory and file records, and the package record's 20 bytes at 0x2C are not checked;
//! names are bytes, not required to be UTF-8; a package's `standalone_update` flag is not
//! refused when the container also has a base; bytes after the string pool inside the table of
//! contents are allowed.
//!
//! # Meta blocks
//!
//! A package's meta block (SPEC §5) is only located and bounds-checked. The spec gives its
//! position as an absolute offset and says nothing more about where it goes, so it may lie
//! anywhere after the table of contents that no extent or other meta block uses: the reference
//! packer at the pinned commit writes it after the extents (`_write_meta_blocks`), and the gap
//! between the table of contents and the blob region is accepted too. Its TLV records are not
//! read: the reference tool at the pinned commit numbers their types from 0 (`format.py`,
//! `class TVL`: `NPDM = 0`) while the spec numbers them from 1 (`0x0001` is the NPDM), and
//! `main.npdm` is read from the `ExeFS` tree, which the spec makes authoritative for content
//! (§5, "the tree wins for content").

mod check;
mod error;
pub mod layout;
mod read;
pub mod title;

#[cfg(any(test, feature = "test-support"))]
pub mod build;
#[cfg(test)]
mod tests;

use std::collections::BTreeSet;
use std::io::{Read, Seek};
use std::ops::Range;

use sha2::{Digest, Sha256};

use crate::pfs0::{read_exact_at, read_vec_at, source_len};
use check::{check_superblock, parse_toc};
pub use error::{NupError, Region, VerifyError, VerifySummary};
pub use layout::{Codec, PackageKind};
use layout::{RawSuperblock, SHA256_SIZE, SUPERBLOCK_SIZE};
pub use read::FileReader;
pub use title::{ExeFs, Title};

/// The largest table of contents [`Nup::open`] reads: 256 MiB. A starquake limit, not the
/// spec's: at 0x50 bytes per file record it leaves room for over three million files.
pub const DEFAULT_MAX_TOC_SIZE: u64 = 256 << 20;

/// How much [`Nup::open_with_limits`] is willing to read into memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Limits {
    /// The largest table of contents accepted, in bytes.
    pub max_toc_size: u64,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_toc_size: DEFAULT_MAX_TOC_SIZE,
        }
    }
}

/// A directory of an open container.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DirId(u32);

/// A file of an open container.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FileId(u32);

impl DirId {
    /// The directory's index in the table of contents.
    pub fn index(self) -> usize {
        index_of(self.0)
    }
}

impl FileId {
    /// The file's index in the table of contents.
    pub fn index(self) -> usize {
        index_of(self.0)
    }
}

/// A `u32` id as an index (a `usize` holds every `u32` on the hosts starquake builds for).
fn index_of(id: u32) -> usize {
    usize::try_from(id).unwrap_or(usize::MAX)
}

/// One entry of a directory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Node {
    /// A subdirectory.
    Dir(DirId),
    /// A file.
    File(FileId),
}

/// A package record, checked.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Package {
    /// The package's position in the table of contents.
    pub index: usize,
    /// The application or add-on id, verbatim (opaque, SPEC §6.1 rule 3).
    pub title_id: u64,
    /// What the package holds.
    pub kind: PackageKind,
    /// The content version, verbatim (SPEC §3.1.1); see [`Package::version_parts`].
    pub version: u32,
    /// Flag bit 0: an update stored without its base (SPEC §3.1.2).
    pub standalone_update: bool,
    /// The root of the package's tree.
    pub root: DirId,
    /// The package's `ExeFS` directory, if it has one.
    pub exefs: Option<DirId>,
    /// The package's `RomFS` directory, if it has one.
    pub romfs: Option<DirId>,
    /// Where the package's meta block lies in the file, if it has one.
    pub meta: Option<Range<u64>>,
}

impl Package {
    /// The version as `(major, minor, micro)`, packed `(major << 16) | (minor << 8) | micro`
    /// (SPEC §3.1.1; `PackageInfo.version_str`). The spec warns that a few titles encode their
    /// version differently, so this is for display only.
    pub fn version_parts(&self) -> (u16, u8, u8) {
        let [micro, minor, low, high] = self.version.to_le_bytes();
        (u16::from_le_bytes([low, high]), minor, micro)
    }
}

/// A file record, checked.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileEntry {
    /// Uncompressed size in bytes.
    pub size: u64,
    /// Where the stored bytes start in the file (0 for an empty file).
    pub extent_offset: u64,
    /// How many bytes are stored.
    pub extent_size: u64,
    /// SHA-256 of the uncompressed content.
    pub sha256: [u8; SHA256_SIZE],
    /// How the extent is stored.
    pub codec: Codec,
    /// The name, as a range of the string pool.
    name: Range<usize>,
}

/// A directory record, checked.
#[derive(Debug, Clone, PartialEq, Eq)]
struct DirEntry {
    /// The name, as a range of the string pool.
    name: Range<usize>,
    /// The children, as a range of the children array.
    children: Range<usize>,
}

/// An open NUP container.
#[derive(Debug)]
pub struct Nup<S> {
    source: S,
    superblock: RawSuperblock,
    packages: Vec<Package>,
    dirs: Vec<DirEntry>,
    files: Vec<FileEntry>,
    children: Vec<Node>,
    pool: Vec<u8>,
}

impl<S: Read + Seek> Nup<S> {
    /// Opens the container in `source` with the [default limits](Limits::default).
    pub fn open(source: S) -> Result<Self, NupError> {
        Self::open_with_limits(source, Limits::default())
    }

    /// Opens the container in `source`: reads and checks the superblock and the table of
    /// contents (see the module docs for every check). File content is not read.
    pub fn open_with_limits(mut source: S, limits: Limits) -> Result<Self, NupError> {
        let actual = source_len(&mut source)?;
        if actual < to_u64(SUPERBLOCK_SIZE) {
            return Err(NupError::SourceTooShort { len: actual });
        }
        let mut head = [0u8; SUPERBLOCK_SIZE];
        read_exact_at(&mut source, 0, &mut head)?;
        let superblock = RawSuperblock::decode(&head)?;
        check_superblock(&superblock, actual, limits)?;

        let toc = read_vec_at(&mut source, superblock.toc_offset, superblock.toc_size)?;
        if Sha256::digest(&toc).as_slice() != superblock.toc_sha256.as_slice() {
            return Err(NupError::TocHashMismatch);
        }
        let parsed = parse_toc(&toc, &superblock)?;
        Ok(Self {
            source,
            superblock,
            packages: parsed.packages,
            dirs: parsed.dirs,
            files: parsed.files,
            children: parsed.children,
            pool: parsed.pool,
        })
    }

    /// Checks every file's content against its SHA-256, package by package in table of contents
    /// order and each tree in name order, and stops at the first file that fails. The table of
    /// contents' own hash was checked by [`Nup::open`]. Files that share an extent (which
    /// [`Nup::open`] made sure agree on their hash) are read once.
    pub fn verify(&mut self) -> Result<VerifySummary, VerifyError> {
        let mut summary = VerifySummary::default();
        let mut hashed = BTreeSet::new();
        for package in 0..self.packages.len() {
            let Some(root) = self.packages.get(package).map(|p| p.root) else {
                continue;
            };
            for (path, file) in self.walk_files(root) {
                let Some(entry) = self.files.get(file.index()) else {
                    continue;
                };
                summary.files = summary.files.saturating_add(1);
                let stored = entry.extent_size != 0;
                let size = entry.size;
                // A shared extent that was already hashed: its files agree on the hash.
                if stored && !hashed.insert((entry.extent_offset, entry.extent_size)) {
                    continue;
                }
                if let Err(source) = self.verify_file(file) {
                    return Err(VerifyError {
                        package,
                        path: String::from_utf8_lossy(&path).into_owned(),
                        source,
                    });
                }
                if stored {
                    summary.extents = summary.extents.saturating_add(1);
                }
                summary.bytes = summary.bytes.saturating_add(size);
            }
        }
        Ok(summary)
    }
}

impl<S> Nup<S> {
    /// The superblock.
    pub fn superblock(&self) -> &RawSuperblock {
        &self.superblock
    }

    /// The packages, in table of contents order.
    pub fn packages(&self) -> &[Package] {
        &self.packages
    }

    /// Number of directories.
    pub fn dir_count(&self) -> usize {
        self.dirs.len()
    }

    /// Number of files.
    pub fn file_count(&self) -> usize {
        self.files.len()
    }

    /// A file's record, or `None` for an id from another container.
    pub fn file(&self, file: FileId) -> Option<&FileEntry> {
        self.files.get(file.index())
    }

    /// A node's name (empty for an id from another container).
    pub fn name(&self, node: Node) -> &[u8] {
        name_in(&self.dirs, &self.files, &self.pool, node)
    }

    /// A directory's entries in name order (empty for an id from another container).
    pub fn children(&self, dir: DirId) -> &[Node] {
        self.dirs
            .get(dir.index())
            .and_then(|d| self.children.get(d.children.clone()))
            .unwrap_or_default()
    }

    /// The entry of `dir` called `name`. Names are unique and sorted in each directory
    /// ([`Nup::open`] checks it), so this is a binary search.
    pub fn lookup(&self, dir: DirId, name: &[u8]) -> Option<Node> {
        let children = self.children(dir);
        children
            .binary_search_by(|&child| self.name(child).cmp(name))
            .ok()
            .and_then(|at| children.get(at).copied())
    }

    /// The node at `path` below `dir`, `/`-separated; empty segments are skipped.
    pub fn resolve(&self, dir: DirId, path: &str) -> Option<Node> {
        let mut at = Node::Dir(dir);
        for segment in path.split('/').filter(|segment| !segment.is_empty()) {
            let Node::Dir(current) = at else {
                return None;
            };
            at = self.lookup(current, segment.as_bytes())?;
        }
        Some(at)
    }

    /// Every file below `dir` with its `/`-separated path from `dir`, depth first in name order.
    /// The tree is a forest ([`Nup::open`] checks it), so each directory is visited once; the
    /// walk also keeps a visited set, so it ends whatever the tree.
    pub fn walk_files(&self, dir: DirId) -> Vec<(Vec<u8>, FileId)> {
        let mut out = Vec::new();
        let mut visited = vec![false; self.dirs.len()];
        if let Some(seen) = visited.get_mut(dir.index()) {
            *seen = true;
        }
        // Each stack entry is a directory, its path prefix ("" or "a/b/") and the next child.
        let mut stack = vec![(dir, Vec::new(), 0usize)];
        while let Some((current, prefix, next)) = stack.last_mut() {
            let Some(&child) = self.children(*current).get(*next) else {
                stack.pop();
                continue;
            };
            *next += 1;
            let mut path = prefix.clone();
            path.extend_from_slice(self.name(child));
            match child {
                Node::File(file) => out.push((path, file)),
                Node::Dir(sub) => {
                    if let Some(seen) = visited.get_mut(sub.index())
                        && !std::mem::replace(seen, true)
                    {
                        path.push(b'/');
                        stack.push((sub, path, 0));
                    }
                }
            }
        }
        out
    }
}

/// A node's name from the records and the pool (empty for an id past the records).
fn name_in<'p>(dirs: &[DirEntry], files: &[FileEntry], pool: &'p [u8], node: Node) -> &'p [u8] {
    let range = match node {
        Node::Dir(dir) => dirs.get(dir.index()).map(|d| d.name.clone()),
        Node::File(file) => files.get(file.index()).map(|f| f.name.clone()),
    };
    range.and_then(|range| pool.get(range)).unwrap_or_default()
}

/// `usize` to `u64`; lossless on the hosts starquake builds for, saturating elsewhere.
pub(crate) fn to_u64(value: usize) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}
