//! The structural checks [`Nup::open`](super::Nup::open) runs on the superblock and the table
//! of contents. The module docs of [`nup`](super) list them and say where each comes from.

use std::collections::BTreeMap;
use std::ops::Range;

use super::layout::{
    CHILD_SLOT_SIZE, DIR_RECORD_SIZE, EXTENT_ALIGN, FILE_RECORD_SIZE, MAGIC, NO_DIR,
    PACKAGE_FLAG_STANDALONE_UPDATE, PACKAGE_RECORD_SIZE, RawDir, RawFile, RawPackage,
    RawSuperblock, RawTocHeader, SUPERBLOCK_SIZE, TOC_HEADER_SIZE, TOC_MAGIC,
};
use super::{
    Codec, DirEntry, DirId, FileEntry, FileId, Limits, MAX_TREE_DEPTH, Node, NupError, Package,
    PackageKind, Region, index_of, name_in, to_u64,
};

/// The superblock checks of SPEC §2 (rules 1 to 5, except the hash, which needs the bytes) and
/// `parse_super`, against the source's length and the reader's limits.
pub(super) fn check_superblock(
    sb: &RawSuperblock,
    actual: u64,
    limits: Limits,
) -> Result<(), NupError> {
    let [n, u, p, major] = sb.magic;
    let [want_n, want_u, want_p, want_major] = MAGIC;
    if [n, u, p] != [want_n, want_u, want_p] {
        return Err(NupError::BadMagic(sb.magic));
    }
    if major != want_major {
        return Err(NupError::UnsupportedVersion(major));
    }
    if usize::from(sb.header_size) < SUPERBLOCK_SIZE {
        return Err(NupError::HeaderTooSmall(sb.header_size));
    }
    if sb.reserved.iter().any(|&byte| byte != 0) {
        return Err(NupError::SuperblockReservedNotZero);
    }
    if sb.flags != 0 {
        return Err(NupError::UnknownSuperblockFlags(sb.flags));
    }
    if sb.file_size != actual {
        return Err(NupError::FileSizeMismatch {
            declared: sb.file_size,
            actual,
        });
    }
    if sb.toc_offset != u64::from(sb.header_size) {
        return Err(NupError::TocNotAfterHeader {
            toc_offset: sb.toc_offset,
            header_size: sb.header_size,
        });
    }
    let toc_end = sb
        .toc_offset
        .checked_add(sb.toc_size)
        .filter(|&end| end <= sb.file_size)
        .ok_or(NupError::TocOutOfBounds)?;
    if !sb.blob_offset.is_multiple_of(EXTENT_ALIGN) {
        return Err(NupError::BlobMisaligned(sb.blob_offset));
    }
    if sb.blob_offset < toc_end {
        return Err(NupError::BlobOverlapsToc {
            blob_offset: sb.blob_offset,
            toc_end,
        });
    }
    if sb.blob_offset > sb.file_size {
        return Err(NupError::BlobOutOfBounds);
    }
    if sb.package_count == 0 {
        return Err(NupError::NoPackages);
    }
    if sb.toc_size > limits.max_toc_size {
        return Err(NupError::TocTooLarge {
            size: sb.toc_size,
            limit: limits.max_toc_size,
        });
    }
    Ok(())
}

/// The checked contents of a table of contents.
pub(super) struct ParsedToc {
    pub(super) packages: Vec<Package>,
    pub(super) dirs: Vec<DirEntry>,
    pub(super) files: Vec<FileEntry>,
    pub(super) children: Vec<Node>,
    pub(super) pool: Vec<u8>,
}

/// `count` records of `size` bytes as a byte count, `None` on overflow.
fn records(count: u64, size: usize) -> Option<u64> {
    count.checked_mul(to_u64(size))
}

/// Where the parts of a table of contents lie, as offsets from its start.
struct Areas {
    /// Package records.
    packages: Range<usize>,
    /// Directory records.
    dirs: Range<usize>,
    /// File records.
    files: Range<usize>,
    /// The children array (between the records and the string pool, as `_parse_toc` has it).
    children: Range<usize>,
    /// The string pool.
    pool: Range<usize>,
}

/// Checks that the records, the string pool and the children array of `head` fit in a table of
/// contents of `toc_len` bytes, and says where they are.
fn areas(head: &RawTocHeader, toc_len: usize) -> Result<Areas, NupError> {
    let len = to_u64(toc_len);
    let records_end = records(u64::from(head.package_count), PACKAGE_RECORD_SIZE)
        .zip(records(head.dir_count, DIR_RECORD_SIZE))
        .zip(records(head.file_count, FILE_RECORD_SIZE))
        .and_then(|((packages, dirs), files)| {
            to_u64(TOC_HEADER_SIZE)
                .checked_add(packages)?
                .checked_add(dirs)?
                .checked_add(files)
        })
        .filter(|&end| end <= len)
        .ok_or_else(|| NupError::TocTruncated {
            needed: needed_saturating(head),
            size: len,
        })?;
    let pool_end = head
        .string_pool_offset
        .checked_add(head.string_pool_size)
        .filter(|&end| head.string_pool_offset >= records_end && end <= len)
        .ok_or(NupError::StringPoolOutOfBounds)?;
    let children_bytes = head.string_pool_offset - records_end;
    if !children_bytes.is_multiple_of(to_u64(CHILD_SLOT_SIZE)) {
        return Err(NupError::ChildrenMisaligned(children_bytes));
    }
    // Every value here is at most `len`, so it fits a `usize`; the record counts multiplied by
    // their sizes were checked above.
    let at = |offset: u64| usize::try_from(offset).unwrap_or(usize::MAX);
    let packages_end = TOC_HEADER_SIZE
        .saturating_add(at(u64::from(head.package_count)).saturating_mul(PACKAGE_RECORD_SIZE));
    let dirs_end = packages_end.saturating_add(at(head.dir_count).saturating_mul(DIR_RECORD_SIZE));
    let files_end = dirs_end.saturating_add(at(head.file_count).saturating_mul(FILE_RECORD_SIZE));
    Ok(Areas {
        packages: TOC_HEADER_SIZE..packages_end,
        dirs: packages_end..dirs_end,
        files: dirs_end..files_end,
        children: files_end..at(head.string_pool_offset),
        pool: at(head.string_pool_offset)..at(pool_end),
    })
}

/// Parses and checks the table of contents `toc` (whose hash has been checked).
pub(super) fn parse_toc(toc: &[u8], sb: &RawSuperblock) -> Result<ParsedToc, NupError> {
    let head = RawTocHeader::decode(toc)?;
    if head.magic != TOC_MAGIC {
        return Err(NupError::BadTocMagic(head.magic));
    }
    if head.package_count != sb.package_count {
        return Err(NupError::PackageCountMismatch {
            superblock: sb.package_count,
            toc: head.package_count,
        });
    }
    let areas = areas(&head, toc.len())?;
    let area = |range: Range<usize>| toc.get(range).unwrap_or_default();
    let pool = area(areas.pool).to_vec();
    let dir_records = area(areas.dirs).as_chunks::<DIR_RECORD_SIZE>().0;
    let file_records = area(areas.files).as_chunks::<FILE_RECORD_SIZE>().0;
    let slots = area(areas.children).as_chunks::<CHILD_SLOT_SIZE>().0;

    let mut packages = Vec::new();
    for (index, record) in area(areas.packages)
        .as_chunks::<PACKAGE_RECORD_SIZE>()
        .0
        .iter()
        .enumerate()
    {
        let raw = RawPackage::decode(record)?;
        packages.push(check_package(index, &raw, dir_records.len(), sb)?);
    }
    check_package_set(&packages)?;

    let mut dirs = Vec::with_capacity(dir_records.len());
    for (index, record) in dir_records.iter().enumerate() {
        dirs.push(check_dir(
            index,
            &RawDir::decode(record)?,
            pool.len(),
            slots.len(),
        )?);
    }
    let mut files = Vec::with_capacity(file_records.len());
    for (index, record) in file_records.iter().enumerate() {
        files.push(check_file(
            index,
            &RawFile::decode(record)?,
            pool.len(),
            sb,
        )?);
    }
    let mut children = Vec::with_capacity(slots.len());
    for (slot, word) in slots.iter().enumerate() {
        children.push(child_node(
            slot,
            u32::from_le_bytes(*word),
            dirs.len(),
            files.len(),
        )?);
    }

    let parsed = ParsedToc {
        packages,
        dirs,
        files,
        children,
        pool,
    };
    check_tree(&parsed)?;
    check_regions(&parsed)?;
    Ok(parsed)
}

/// One directory record's own checks: its name inside the pool and its child range inside the
/// children array (`_pool_str`, `_check_bounds`).
fn check_dir(
    index: usize,
    raw: &RawDir,
    pool_len: usize,
    slot_count: usize,
) -> Result<DirEntry, NupError> {
    let name =
        pool_range(raw.name_offset, raw.name_len, pool_len).ok_or(NupError::NameOutOfPool {
            node: "directory",
            index,
        })?;
    let first = index_of(raw.first_child);
    let children = first
        .checked_add(index_of(raw.child_count))
        .filter(|&end| end <= slot_count)
        .map(|end| first..end)
        .ok_or(NupError::ChildRangeOutOfBounds { dir: index })?;
    Ok(DirEntry { name, children })
}

/// The node a child slot names (SPEC §3.3): an id below the directory count is a directory, and
/// the rest index the files once the directory count is subtracted.
fn child_node(
    slot: usize,
    node: u32,
    dir_count: usize,
    file_count: usize,
) -> Result<Node, NupError> {
    let id = index_of(node);
    if id < dir_count {
        Ok(Node::Dir(DirId(node)))
    } else if id - dir_count < file_count {
        let file = u32::try_from(id - dir_count).unwrap_or(u32::MAX);
        Ok(Node::File(FileId(file)))
    } else {
        Err(NupError::NodeOutOfRange { slot, node })
    }
}

/// What the records of `head` would need, saturating, for the truncation message.
fn needed_saturating(head: &RawTocHeader) -> u64 {
    to_u64(TOC_HEADER_SIZE)
        .saturating_add(u64::from(head.package_count).saturating_mul(to_u64(PACKAGE_RECORD_SIZE)))
        .saturating_add(head.dir_count.saturating_mul(to_u64(DIR_RECORD_SIZE)))
        .saturating_add(head.file_count.saturating_mul(to_u64(FILE_RECORD_SIZE)))
}

/// The pool range of a name, if it fits in a pool of `pool_len` bytes.
fn pool_range(offset: u32, len: u32, pool_len: usize) -> Option<Range<usize>> {
    let start = index_of(offset);
    let end = start.checked_add(index_of(len))?;
    (end <= pool_len).then_some(start..end)
}

/// One package record's own checks (SPEC §3.1, §6.1 rule 4; `parse_pkg`, `_check_bounds`).
fn check_package(
    index: usize,
    raw: &RawPackage,
    dir_count: usize,
    sb: &RawSuperblock,
) -> Result<Package, NupError> {
    let kind = PackageKind::from_byte(raw.kind).ok_or(NupError::UnknownPackageKind {
        package: index,
        kind: raw.kind,
    })?;
    if raw.reserved != 0 || raw.reserved2 != 0 {
        return Err(NupError::PackageReservedNotZero { package: index });
    }
    if raw.flags & !PACKAGE_FLAG_STANDALONE_UPDATE != 0 {
        return Err(NupError::UnknownPackageFlags {
            package: index,
            flags: raw.flags,
        });
    }
    let dir = |field: &'static str, value: u32| {
        if index_of(value) < dir_count {
            Ok(DirId(value))
        } else {
            Err(NupError::DirOutOfRange {
                package: index,
                field,
                dir: value,
            })
        }
    };
    let optional = |field: &'static str, value: u32| {
        if value == NO_DIR {
            Ok(None)
        } else {
            dir(field, value).map(Some)
        }
    };
    let meta = if raw.meta_size == 0 {
        None
    } else {
        let toc_end = sb.toc_offset.saturating_add(sb.toc_size);
        let range = raw
            .meta_offset
            .checked_add(u64::from(raw.meta_size))
            .filter(|&end| raw.meta_offset >= toc_end && end <= sb.file_size)
            .map(|end| raw.meta_offset..end)
            .ok_or(NupError::MetaOutOfBounds { package: index })?;
        Some(range)
    };
    Ok(Package {
        index,
        title_id: raw.title_id,
        kind,
        version: raw.version,
        standalone_update: raw.flags & PACKAGE_FLAG_STANDALONE_UPDATE != 0,
        root: dir("root_dir", raw.root_dir)?,
        exefs: optional("exefs_dir", raw.exefs_dir)?,
        romfs: optional("romfs_dir", raw.romfs_dir)?,
        meta,
    })
}

/// The rules about the set of packages (SPEC §3.1.2): at most one base, and an update in a
/// container without a base is flagged standalone (`_check_bounds`).
fn check_package_set(packages: &[Package]) -> Result<(), NupError> {
    let mut bases = packages.iter().filter(|p| p.kind == PackageKind::Base);
    let first_base = bases.next();
    if let (Some(first), Some(second)) = (first_base, bases.next()) {
        return Err(NupError::MultipleBases {
            first: first.index,
            second: second.index,
        });
    }
    if first_base.is_none()
        && let Some(update) = packages
            .iter()
            .find(|p| p.kind == PackageKind::Update && !p.standalone_update)
    {
        return Err(NupError::UpdateWithoutBase {
            package: update.index,
        });
    }
    Ok(())
}

/// One file record's own checks (SPEC §3.4, §3.6, §4; `parse_file`, `_check_bounds`).
fn check_file(
    index: usize,
    raw: &RawFile,
    pool_len: usize,
    sb: &RawSuperblock,
) -> Result<FileEntry, NupError> {
    let name =
        pool_range(raw.name_offset, raw.name_len, pool_len).ok_or(NupError::NameOutOfPool {
            node: "file",
            index,
        })?;
    let codec = Codec::from_byte(raw.codec).ok_or(NupError::UnknownCodec {
        file: index,
        codec: raw.codec,
    })?;
    if raw.codec_flags != 0 {
        return Err(NupError::CodecFlagsNotZero {
            file: index,
            flags: raw.codec_flags,
        });
    }
    if raw.size == 0 {
        if raw.extent_offset != 0 || raw.extent_size != 0 || codec != Codec::Raw {
            return Err(NupError::EmptyFileWithExtent { file: index });
        }
    } else {
        if raw.extent_size == 0 {
            return Err(NupError::MissingExtent { file: index });
        }
        if codec == Codec::Raw && raw.extent_size != raw.size {
            return Err(NupError::RawSizeMismatch {
                file: index,
                size: raw.size,
                extent_size: raw.extent_size,
            });
        }
        let inside = raw.extent_offset >= sb.blob_offset
            && raw
                .extent_offset
                .checked_add(raw.extent_size)
                .is_some_and(|end| end <= sb.file_size);
        if !inside {
            return Err(NupError::ExtentOutOfBounds { file: index });
        }
        if !raw.extent_offset.is_multiple_of(EXTENT_ALIGN) {
            return Err(NupError::ExtentMisaligned {
                file: index,
                offset: raw.extent_offset,
            });
        }
    }
    Ok(FileEntry {
        size: raw.size,
        extent_offset: raw.extent_offset,
        extent_size: raw.extent_size,
        sha256: raw.sha256,
        codec,
        name,
    })
}

/// The tree rules (see "Divergences" in the module docs): child ranges partition the children
/// array, each node has one place, every node is reachable, children are sorted, and each
/// package's content directories are inside its own tree.
fn check_tree(toc: &ParsedToc) -> Result<(), NupError> {
    check_child_ranges(toc)?;

    // One place per node: package roots first, then child slots.
    let mut placed_dirs = vec![false; toc.dirs.len()];
    let mut placed_files = vec![false; toc.files.len()];
    let mut place = |node: Node| {
        let slot = match node {
            Node::Dir(dir) => placed_dirs.get_mut(dir.index()),
            Node::File(file) => placed_files.get_mut(file.index()),
        };
        match slot {
            Some(seen) if !*seen => {
                *seen = true;
                Ok(())
            }
            _ => Err(NupError::NodeReferencedTwice(node)),
        }
    };
    for package in &toc.packages {
        place(Node::Dir(package.root))?;
    }
    for &child in &toc.children {
        place(child)?;
    }

    // Reachability, recording which package owns each directory. Each directory has one place,
    // so the walk meets it at most once; the owner array doubles as the visited set. The walk
    // also measures depth (a package root is at depth 0) and refuses a tree deeper than
    // `MAX_TREE_DEPTH`, which is what keeps a path (and the walk that builds paths) short.
    let mut owner: Vec<Option<usize>> = vec![None; toc.dirs.len()];
    for package in &toc.packages {
        let mut stack = vec![(package.root, 0usize)];
        while let Some((dir, depth)) = stack.pop() {
            if depth > MAX_TREE_DEPTH {
                return Err(NupError::TreeTooDeep {
                    dir: dir.index(),
                    limit: MAX_TREE_DEPTH,
                });
            }
            let Some(slot) = owner.get_mut(dir.index()) else {
                continue;
            };
            if slot.is_some() {
                continue;
            }
            *slot = Some(package.index);
            let Some(entry) = toc.dirs.get(dir.index()) else {
                continue;
            };
            for &child in toc.children.get(entry.children.clone()).unwrap_or_default() {
                if let Node::Dir(sub) = child {
                    stack.push((sub, depth.saturating_add(1)));
                }
            }
        }
    }
    if let Some(dir) = owner.iter().position(Option::is_none) {
        return Err(NupError::UnreachableNode(Node::Dir(DirId(
            u32::try_from(dir).unwrap_or(u32::MAX),
        ))));
    }
    // Every slot is in some directory's range (checked above) and every directory is
    // reachable, so a file that has a place is reachable; one without a place is not.
    if let Some(file) = placed_files.iter().position(|placed| !placed) {
        return Err(NupError::UnreachableNode(Node::File(FileId(
            u32::try_from(file).unwrap_or(u32::MAX),
        ))));
    }

    check_sorted(toc)?;

    for package in &toc.packages {
        for (field, dir) in [("exefs_dir", package.exefs), ("romfs_dir", package.romfs)] {
            let Some(dir) = dir else { continue };
            if owner.get(dir.index()).copied().flatten() != Some(package.index) {
                return Err(NupError::ContentDirOutsidePackage {
                    package: package.index,
                    field,
                    dir: dir.index(),
                });
            }
        }
    }
    Ok(())
}

/// The directories' child ranges are disjoint and together are exactly the children array.
fn check_child_ranges(toc: &ParsedToc) -> Result<(), NupError> {
    let mut ranges: Vec<&Range<usize>> = toc
        .dirs
        .iter()
        .map(|d| &d.children)
        .filter(|r| !r.is_empty())
        .collect();
    ranges.sort_by_key(|r| (r.start, r.end));
    let mut covered = 0usize;
    let mut end = 0usize;
    for range in ranges {
        if range.start < end {
            return Err(NupError::ChildRangesOverlap);
        }
        end = range.end;
        covered = covered.saturating_add(range.len());
    }
    if covered != toc.children.len() {
        return Err(NupError::ChildRangesOverlap);
    }
    Ok(())
}

/// Each directory's children are in strictly increasing bytewise order of name (SPEC §3.3).
fn check_sorted(toc: &ParsedToc) -> Result<(), NupError> {
    let name = |node| name_in(&toc.dirs, &toc.files, &toc.pool, node);
    for (index, dir) in toc.dirs.iter().enumerate() {
        let children = toc.children.get(dir.children.clone()).unwrap_or_default();
        if children
            .windows(2)
            .any(|pair| matches!(pair, [a, b] if name(*a) >= name(*b)))
        {
            return Err(NupError::ChildrenNotSorted { dir: index });
        }
    }
    Ok(())
}

/// The overlap rules (see "Divergences"): files sharing an extent agree, and extents and meta
/// blocks do not overlap.
fn check_regions(toc: &ParsedToc) -> Result<(), NupError> {
    // Distinct extents, each with the first file that uses it.
    let mut extents: BTreeMap<(u64, u64), usize> = BTreeMap::new();
    for (index, file) in toc.files.iter().enumerate() {
        if file.extent_size == 0 {
            continue;
        }
        let key = (file.extent_offset, file.extent_size);
        match extents.get(&key) {
            None => {
                extents.insert(key, index);
            }
            Some(&first) => {
                let agree = toc.files.get(first).is_some_and(|other| {
                    other.size == file.size
                        && other.codec == file.codec
                        && other.sha256 == file.sha256
                });
                if !agree {
                    return Err(NupError::SharedExtentMismatch {
                        first,
                        second: index,
                    });
                }
            }
        }
    }
    let mut regions: Vec<(u64, u64, Region)> = extents
        .iter()
        .map(|(&(offset, size), &file)| (offset, offset.saturating_add(size), Region::Extent(file)))
        .collect();
    for package in &toc.packages {
        if let Some(meta) = &package.meta {
            regions.push((meta.start, meta.end, Region::Meta(package.index)));
        }
    }
    regions.sort_unstable();
    for pair in regions.windows(2) {
        if let [(_, first_end, first), (second_start, _, second)] = pair
            && second_start < first_end
        {
            return Err(NupError::RegionsOverlap {
                first: *first,
                second: *second,
            });
        }
    }
    Ok(())
}
