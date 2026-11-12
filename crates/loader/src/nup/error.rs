//! The NUP reader's errors.

use super::layout::SUPERBLOCK_SIZE;
use super::Node;

/// Which region of the file an overlap error names.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Region {
    /// The extent of a file.
    Extent(usize),
    /// The meta block of a package.
    Meta(usize),
}

/// Why a container could not be opened or a file of it could not be read.
#[derive(Debug, thiserror::Error)]
pub enum NupError {
    /// The source could not be read.
    #[error("reading the container failed: {0}")]
    Io(#[from] std::io::Error),
    /// The source is shorter than the superblock.
    #[error("the file is {len} bytes, shorter than the {SUPERBLOCK_SIZE}-byte superblock")]
    SourceTooShort {
        /// Length of the source.
        len: u64,
    },
    /// A record was cut short.
    #[error("the {record} record is cut short")]
    RecordTruncated {
        /// Which record.
        record: &'static str,
    },
    /// The first three bytes are not `NUP`.
    #[error("not a NUP file: the magic is {0:02x?}")]
    BadMagic([u8; 4]),
    /// The format's major version is not 1.
    #[error("NUP format version {0} is not supported; only version 1 is")]
    UnsupportedVersion(u8),
    /// `header_size` is below 0x80.
    #[error("the superblock says it is {0:#x} bytes, less than {SUPERBLOCK_SIZE:#x}")]
    HeaderTooSmall(u16),
    /// The superblock's reserved bytes are not zero.
    #[error("the superblock's reserved bytes are not zero")]
    SuperblockReservedNotZero,
    /// The superblock has flags; version 1 defines none.
    #[error("unknown superblock flags {0:#x}")]
    UnknownSuperblockFlags(u32),
    /// `file_size` is not the source's length.
    #[error("the superblock says the file is {declared} bytes, but it is {actual}")]
    FileSizeMismatch {
        /// `file_size`.
        declared: u64,
        /// The source's length.
        actual: u64,
    },
    /// `toc_offset` is not `header_size`.
    #[error("the table of contents starts at {toc_offset:#x}, not at the end of the superblock ({header_size:#x})")]
    TocNotAfterHeader {
        /// `toc_offset`.
        toc_offset: u64,
        /// `header_size`.
        header_size: u16,
    },
    /// The table of contents runs past the end of the file.
    #[error("the table of contents runs past the end of the file")]
    TocOutOfBounds,
    /// `blob_offset` is not 16-aligned.
    #[error("the blob region starts at {0:#x}, which is not 16-byte aligned")]
    BlobMisaligned(u64),
    /// `blob_offset` is before the end of the table of contents.
    #[error("the blob region starts at {blob_offset:#x}, inside the table of contents (which ends at {toc_end:#x})")]
    BlobOverlapsToc {
        /// `blob_offset`.
        blob_offset: u64,
        /// `toc_offset + toc_size`.
        toc_end: u64,
    },
    /// `blob_offset` is past the end of the file.
    #[error("the blob region starts past the end of the file")]
    BlobOutOfBounds,
    /// The container has no packages.
    #[error("the container has no packages")]
    NoPackages,
    /// The table of contents is bigger than [`Limits::max_toc_size`].
    #[error("the table of contents is {size} bytes, more than the {limit} this reader accepts")]
    TocTooLarge {
        /// `toc_size`.
        size: u64,
        /// The limit.
        limit: u64,
    },
    /// The table of contents does not match its SHA-256.
    #[error("the table of contents does not match its SHA-256")]
    TocHashMismatch,
    /// The table of contents magic is not `TOC\x01`.
    #[error("the table of contents magic is {0:02x?}, not TOC 01")]
    BadTocMagic([u8; 4]),
    /// The table of contents disagrees with the superblock on the package count.
    #[error("the superblock lists {superblock} packages and the table of contents {toc}")]
    PackageCountMismatch {
        /// The superblock's count.
        superblock: u32,
        /// The table of contents' count.
        toc: u32,
    },
    /// The records do not fit in the table of contents.
    #[error("the records need {needed} bytes of table of contents, which has {size}")]
    TocTruncated {
        /// Bytes the records need (saturated on overflow).
        needed: u64,
        /// `toc_size`.
        size: u64,
    },
    /// The string pool overlaps the records or runs past the table of contents.
    #[error("the string pool lies outside the table of contents' free space")]
    StringPoolOutOfBounds,
    /// The children array is not a whole number of slots.
    #[error("the children array is {0} bytes, not a whole number of 4-byte slots")]
    ChildrenMisaligned(u64),
    /// A package has a reserved kind.
    #[error("package {package} has unknown kind {kind}")]
    UnknownPackageKind {
        /// Package index.
        package: usize,
        /// The kind byte.
        kind: u8,
    },
    /// A package has flags other than bit 0.
    #[error("package {package} has unknown flags {flags:#x}")]
    UnknownPackageFlags {
        /// Package index.
        package: usize,
        /// The flags byte.
        flags: u8,
    },
    /// A package's reserved fields at 0x1A and 0x1C are not zero.
    #[error("package {package} has reserved bytes that are not zero")]
    PackageReservedNotZero {
        /// Package index.
        package: usize,
    },
    /// A package names a directory that does not exist.
    #[error("package {package}'s {field} is directory {dir}, which does not exist")]
    DirOutOfRange {
        /// Package index.
        package: usize,
        /// `root_dir`, `exefs_dir` or `romfs_dir`.
        field: &'static str,
        /// The value.
        dir: u32,
    },
    /// A package's meta block lies outside the space after the table of contents.
    #[error("package {package}'s meta block lies outside the file or over the table of contents")]
    MetaOutOfBounds {
        /// Package index.
        package: usize,
    },
    /// Two packages are bases.
    #[error("packages {first} and {second} are both base packages")]
    MultipleBases {
        /// The first base.
        first: usize,
        /// The second base.
        second: usize,
    },
    /// An update without a base in the container lacks the standalone flag.
    #[error("package {package} is an update without a base, and is not flagged standalone")]
    UpdateWithoutBase {
        /// Package index.
        package: usize,
    },
    /// A name runs past the string pool.
    #[error("the name of {node} {index} runs past the string pool")]
    NameOutOfPool {
        /// `directory` or `file`.
        node: &'static str,
        /// Record index.
        index: usize,
    },
    /// A directory's child range runs past the children array.
    #[error("directory {dir}'s children run past the children array")]
    ChildRangeOutOfBounds {
        /// Directory index.
        dir: usize,
    },
    /// Two directories' child ranges overlap, or some slots belong to no directory.
    #[error("the directories' child ranges overlap or leave slots unused")]
    ChildRangesOverlap,
    /// A child slot names a node that does not exist.
    #[error("child slot {slot} names node {node}, which does not exist")]
    NodeOutOfRange {
        /// Slot index.
        slot: usize,
        /// The node id.
        node: u32,
    },
    /// A node is a package root or a child more than once.
    #[error("{0:?} appears more than once in the tree")]
    NodeReferencedTwice(Node),
    /// A node cannot be reached from any package root.
    #[error("{0:?} is not reachable from any package root")]
    UnreachableNode(Node),
    /// A directory's children are not in strictly increasing order of name.
    #[error("directory {dir}'s children are not sorted by name, or two share a name")]
    ChildrenNotSorted {
        /// Directory index.
        dir: usize,
    },
    /// A package's `ExeFS` or `RomFS` directory is in another package's tree.
    #[error("package {package}'s {field} is directory {dir}, which is not in its tree")]
    ContentDirOutsidePackage {
        /// Package index.
        package: usize,
        /// `exefs_dir` or `romfs_dir`.
        field: &'static str,
        /// Directory index.
        dir: usize,
    },
    /// A file uses a codec that is not 0 or 1.
    #[error("file {file} uses unknown codec {codec}")]
    UnknownCodec {
        /// File index.
        file: usize,
        /// The codec byte.
        codec: u8,
    },
    /// A file's codec flags are not zero.
    #[error("file {file} has codec flags {flags:#x}; version 1 defines none")]
    CodecFlagsNotZero {
        /// File index.
        file: usize,
        /// The flags byte.
        flags: u8,
    },
    /// An empty file has an extent or a codec.
    #[error("file {file} is empty but has an extent or a codec")]
    EmptyFileWithExtent {
        /// File index.
        file: usize,
    },
    /// A non-empty file has no extent.
    #[error("file {file} is not empty but has no extent")]
    MissingExtent {
        /// File index.
        file: usize,
    },
    /// A raw file's extent is not its size.
    #[error("file {file} is stored raw in {extent_size} bytes but is {size} bytes long")]
    RawSizeMismatch {
        /// File index.
        file: usize,
        /// Uncompressed size.
        size: u64,
        /// Stored size.
        extent_size: u64,
    },
    /// A file's extent lies outside the blob region.
    #[error("file {file}'s extent lies outside the blob region")]
    ExtentOutOfBounds {
        /// File index.
        file: usize,
    },
    /// A file's extent is not 16-aligned.
    #[error("file {file}'s extent starts at {offset:#x}, which is not 16-byte aligned")]
    ExtentMisaligned {
        /// File index.
        file: usize,
        /// The offset.
        offset: u64,
    },
    /// Two regions of the blob area overlap.
    #[error("{first:?} and {second:?} overlap")]
    RegionsOverlap {
        /// The region that starts first.
        first: Region,
        /// The region that starts inside it.
        second: Region,
    },
    /// Two files share an extent but disagree on size, codec or hash.
    #[error("files {first} and {second} share an extent but not their size, codec and hash")]
    SharedExtentMismatch {
        /// The first file.
        first: usize,
        /// The second file.
        second: usize,
    },
    /// A file id from another container.
    #[error("file {0} does not exist in this container")]
    NoSuchFile(usize),
    /// A file is bigger than the caller allowed.
    #[error("file {file} is {size} bytes, more than the {limit} allowed")]
    FileTooLarge {
        /// File index.
        file: usize,
        /// Its size.
        size: u64,
        /// The limit.
        limit: u64,
    },
    /// A file's stored bytes ran out before its size was reached.
    #[error("file {file}'s content ends before its size")]
    ContentTruncated {
        /// File index.
        file: usize,
    },
    /// A file's stored bytes hold more than its size.
    #[error("file {file}'s content is longer than its size")]
    ContentTooLong {
        /// File index.
        file: usize,
    },
    /// A compressed file's frame ends before its extent does.
    #[error("file {file}'s extent has bytes after its zstd frame")]
    TrailingExtentBytes {
        /// File index.
        file: usize,
    },
    /// A compressed file's frame does not declare the file's size.
    #[error("file {file}'s zstd frame declares {frame} bytes of content, not its size {size}")]
    ZstdContentSize {
        /// File index.
        file: usize,
        /// The file's size.
        size: u64,
        /// The frame's content size (0 when the frame declares none).
        frame: u64,
    },
    /// A compressed file's frame could not be decoded.
    #[error("file {file}'s zstd frame could not be decoded: {message}")]
    Zstd {
        /// File index.
        file: usize,
        /// The decoder's message.
        message: String,
    },
    /// A file's content does not match its SHA-256.
    #[error("file {file}'s content does not match its SHA-256")]
    HashMismatch {
        /// File index.
        file: usize,
    },
    /// The container holds neither a base nor an update.
    #[error("the container holds no program: no base and no update package")]
    NoProgram,
    /// Two updates share the highest version, so neither is the newest.
    #[error("updates {first} and {second} both have version {version}")]
    AmbiguousUpdate {
        /// The first update.
        first: usize,
        /// The second update.
        second: usize,
        /// Their version.
        version: u32,
    },
}

impl NupError {
    /// True when the source could not be read, as opposed to the container being malformed.
    pub fn is_io(&self) -> bool {
        matches!(self, Self::Io(_))
    }

    /// The error carried by an `io::Error` that [`FileReader`] returned: its own [`NupError`]
    /// when it carries one, [`NupError::Io`] otherwise.
    pub fn from_io(error: std::io::Error) -> Self {
        if error
            .get_ref()
            .is_some_and(carries_nup_error)
        {
            if let Some(inner) = error.into_inner() {
                return match inner.downcast::<NupError>() {
                    Ok(nup) => *nup,
                    Err(other) => Self::Io(std::io::Error::other(other)),
                };
            }
            return Self::Io(std::io::Error::other("an empty error"));
        }
        Self::Io(error)
    }
}

/// A file whose hash did not match, or that could not be read, during [`Nup::verify`].
#[derive(Debug, thiserror::Error)]
#[error("package {package}, {path}: {source}")]
pub struct VerifyError {
    /// The package whose tree holds the file.
    pub package: usize,
    /// The file's path from the package root, `/`-separated (names shown lossily as UTF-8).
    pub path: String,
    /// What went wrong.
    #[source]
    pub source: NupError,
}

/// What [`Nup::verify`] checked.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct VerifySummary {
    /// Files whose content was checked (every file of every package).
    pub files: u64,
    /// Distinct extents read and hashed (files that share an extent are hashed once).
    pub extents: u64,
    /// Uncompressed bytes hashed.
    pub bytes: u64,
}


/// Whether an `io::Error`'s payload is a [`NupError`].
fn carries_nup_error(inner: &(dyn std::error::Error + Send + Sync + 'static)) -> bool {
    inner.is::<NupError>()
}
