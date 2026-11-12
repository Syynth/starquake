//! The fixed-size records of a NUP container, decoded field by field and nothing more.
//!
//! Every record here is read exactly as `nup/SPEC.md` (§2, §3.1, §3.2, §3.4) and
//! `nup/tools/nup/nup/format.py` @ a4840f06 (MIT) lay it out: the struct layouts and the
//! `struct.Struct` formats `_super_s` (`<4sHHQQ32sQQII16s32s`), `_toc_head_s` (`<4sIQQQQQ`),
//! `_pkg_s` (`<QIIIIBBHIQI20x`), `_dir_s` (`<IIIIQ`) and `_file_s` (`<IIQQQ32sBBH12x`). All
//! integers are little-endian (§1). The checks that relate a field to anything else are in the
//! parent module; a decoder here only fails when its input is shorter than the record.
//!
//! Each record keeps its reserved fields, so a decode followed by an encode gives back the same
//! bytes. The encoders exist for the test builder (`nup::build`) and the round-trip tests.

use super::NupError;
use crate::pfs0::Le;

/// The superblock magic: `NUP` and the format's major version, 1 (SPEC §2, rule 1; `MAGIC`).
pub const MAGIC: [u8; 4] = *b"NUP\x01";
/// The table of contents magic (SPEC §3, `NupToc::magic`; `TOC_MAGIC`).
pub const TOC_MAGIC: [u8; 4] = *b"TOC\x01";
/// Size of the version-1 superblock (SPEC §2 and §8; `HEADER_SIZE`).
pub const SUPERBLOCK_SIZE: usize = 0x80;
/// Size of the table of contents header that precedes the package records (SPEC §3: the
/// `packages` array starts at 0x30; `_toc_head_s.size`).
pub const TOC_HEADER_SIZE: usize = 0x30;
/// Size of a package record (SPEC §3.1 and §8; `PKG_SIZE`).
pub const PACKAGE_RECORD_SIZE: usize = 0x40;
/// Size of a directory record (SPEC §3.2 and §8; `DIR_SIZE`).
pub const DIR_RECORD_SIZE: usize = 0x18;
/// Size of a file record (SPEC §3.4 and §8; `FILE_SIZE`).
pub const FILE_RECORD_SIZE: usize = 0x50;
/// Size of one slot of the children array (SPEC §3.3: a flat `uint32` array).
pub const CHILD_SLOT_SIZE: usize = 4;
/// Alignment of the blob region and of every extent (SPEC §4 and §8; `EXTENT_ALIGN`).
pub const EXTENT_ALIGN: u64 = 16;
/// The `exefs_dir` / `romfs_dir` value that means "no such directory" (SPEC §3.1,
/// `0xFFFFFFFF`; `NO_DIR`).
pub const NO_DIR: u32 = u32::MAX;
/// Package record flag bit 0: an update in a container without its base (SPEC §3.1.2,
/// `PKG_FLAG_STANDALONE_UPDATE`).
pub const PACKAGE_FLAG_STANDALONE_UPDATE: u8 = 0x01;
/// Size of a SHA-256 digest (SPEC §8: the hash is SHA-256).
pub const SHA256_SIZE: usize = 32;

/// What a package holds (SPEC §3.1.2; `Kind`). Kinds 5 to 255 are reserved, and a reader rejects
/// them (SPEC §6.1, rule 4).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PackageKind {
    /// 0: the base application program.
    Base,
    /// 1: a patch program for the base.
    Update,
    /// 2: add-on content, with its own title id.
    AddOn,
    /// 3: reserved by the spec for add-on updates.
    AddOnUpdate,
    /// 4: reserved by the spec for system-data packages.
    System,
}

impl PackageKind {
    /// Every kind, in value order.
    pub const ALL: [Self; 5] = [
        Self::Base,
        Self::Update,
        Self::AddOn,
        Self::AddOnUpdate,
        Self::System,
    ];

    /// The kind a record's `kind` byte names, or `None` for a reserved value.
    pub fn from_byte(value: u8) -> Option<Self> {
        Self::ALL.get(usize::from(value)).copied()
    }

    /// The record's `kind` byte for this kind.
    pub fn to_byte(self) -> u8 {
        match self {
            Self::Base => 0,
            Self::Update => 1,
            Self::AddOn => 2,
            Self::AddOnUpdate => 3,
            Self::System => 4,
        }
    }

    /// The spec's name for the kind (SPEC §3.1.2; `KIND_NAMES`).
    pub fn name(self) -> &'static str {
        match self {
            Self::Base => "base",
            Self::Update => "update",
            Self::AddOn => "addon",
            Self::AddOnUpdate => "addon_update",
            Self::System => "system",
        }
    }
}

/// How a file's extent is stored (SPEC §4.1; `Codec`). Codecs 2 (`lz4`) and 3 (`zstd-chunked`)
/// are reserved for a later version, and a reader rejects them like any unknown codec (SPEC §4.1,
/// §6.1 rule 4).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Codec {
    /// 0: the extent is the file's bytes.
    Raw,
    /// 1: the extent is one Zstandard frame that embeds the content size.
    Zstd,
}

impl Codec {
    /// Every codec, in value order.
    pub const ALL: [Self; 2] = [Self::Raw, Self::Zstd];

    /// The codec a record's `codec` byte names, or `None` for an unknown value.
    pub fn from_byte(value: u8) -> Option<Self> {
        Self::ALL.get(usize::from(value)).copied()
    }

    /// The record's `codec` byte for this codec.
    pub fn to_byte(self) -> u8 {
        match self {
            Self::Raw => 0,
            Self::Zstd => 1,
        }
    }

    /// The spec's name for the codec (SPEC §4.1).
    pub fn name(self) -> &'static str {
        match self {
            Self::Raw => "raw",
            Self::Zstd => "zstd",
        }
    }
}

/// The superblock as stored (SPEC §2, `NupSuperblock`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RawSuperblock {
    /// 0x00: `NUP` and the major version.
    pub magic: [u8; 4],
    /// 0x04: size of the superblock; 0x80 in version 1.
    pub header_size: u16,
    /// 0x06: superblock version; 1.
    pub header_version: u16,
    /// 0x08: where the table of contents starts.
    pub toc_offset: u64,
    /// 0x10: size of the table of contents.
    pub toc_size: u64,
    /// 0x18: SHA-256 of the table of contents.
    pub toc_sha256: [u8; SHA256_SIZE],
    /// 0x38: where the blob region starts.
    pub blob_offset: u64,
    /// 0x40: size of the whole file.
    pub file_size: u64,
    /// 0x48: number of package records.
    pub package_count: u32,
    /// 0x4C: flags; none are defined in version 1.
    pub flags: u32,
    /// 0x50: content-derived id (SPEC §2.1).
    pub uuid: [u8; 16],
    /// 0x60: reserved, zero.
    pub reserved: [u8; 32],
}

/// The table of contents header as stored (SPEC §3, `NupToc` up to `packages`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RawTocHeader {
    /// 0x00: `TOC` and 1.
    pub magic: [u8; 4],
    /// 0x04: number of package records, repeated from the superblock.
    pub package_count: u32,
    /// 0x08: number of directory records.
    pub dir_count: u64,
    /// 0x10: number of file records.
    pub file_count: u64,
    /// 0x18: where the string pool starts, from the start of the table of contents.
    pub string_pool_offset: u64,
    /// 0x20: size of the string pool.
    pub string_pool_size: u64,
    /// 0x28: reserved, zero.
    pub reserved: u64,
}

/// A package record as stored (SPEC §3.1, `NupPackage`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RawPackage {
    /// 0x00: the application or add-on id.
    pub title_id: u64,
    /// 0x08: the package's root directory.
    pub root_dir: u32,
    /// 0x0C: the `ExeFS` directory, or [`NO_DIR`].
    pub exefs_dir: u32,
    /// 0x10: the `RomFS` directory, or [`NO_DIR`].
    pub romfs_dir: u32,
    /// 0x14: the content version, as dump tools export it.
    pub version: u32,
    /// 0x18: the kind byte.
    pub kind: u8,
    /// 0x19: flags; bit 0 is [`PACKAGE_FLAG_STANDALONE_UPDATE`].
    pub flags: u8,
    /// 0x1A: reserved, zero.
    pub reserved: u16,
    /// 0x1C: reserved, zero.
    pub reserved2: u32,
    /// 0x20: where the package's meta block starts, or 0.
    pub meta_offset: u64,
    /// 0x28: size of the meta block, or 0.
    pub meta_size: u32,
    /// 0x2C: reserved, zero.
    pub reserved3: [u8; 20],
}

/// A directory record as stored (SPEC §3.2, `NupDir`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RawDir {
    /// 0x00: where the name starts in the string pool.
    pub name_offset: u32,
    /// 0x04: length of the name.
    pub name_len: u32,
    /// 0x08: first slot of the directory's children.
    pub first_child: u32,
    /// 0x0C: number of children.
    pub child_count: u32,
    /// 0x10: reserved, zero.
    pub reserved: u64,
}

/// A file record as stored (SPEC §3.4, `NupFile`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RawFile {
    /// 0x00: where the name starts in the string pool.
    pub name_offset: u32,
    /// 0x04: length of the name.
    pub name_len: u32,
    /// 0x08: uncompressed size.
    pub size: u64,
    /// 0x10: where the extent starts in the file.
    pub extent_offset: u64,
    /// 0x18: stored size of the extent.
    pub extent_size: u64,
    /// 0x20: SHA-256 of the uncompressed content.
    pub sha256: [u8; SHA256_SIZE],
    /// 0x40: the codec byte.
    pub codec: u8,
    /// 0x41: codec flags; zero in version 1.
    pub codec_flags: u8,
    /// 0x42: reserved, zero.
    pub reserved2: u16,
    /// 0x44: reserved, zero.
    pub reserved3: [u8; 12],
}

/// The error a decoder returns for a record that does not fit in what was read.
fn short(record: &'static str) -> NupError {
    NupError::RecordTruncated { record }
}

impl RawSuperblock {
    /// Decodes the superblock at the start of `bytes`.
    pub fn decode(bytes: &[u8]) -> Result<Self, NupError> {
        let mut le = Le::new(bytes);
        let mut fields = || {
            Some(Self {
                magic: le.take()?,
                header_size: le.take::<2>().map(u16::from_le_bytes)?,
                header_version: le.take::<2>().map(u16::from_le_bytes)?,
                toc_offset: le.u64()?,
                toc_size: le.u64()?,
                toc_sha256: le.take()?,
                blob_offset: le.u64()?,
                file_size: le.u64()?,
                package_count: le.u32()?,
                flags: le.u32()?,
                uuid: le.take()?,
                reserved: le.take()?,
            })
        };
        fields().ok_or_else(|| short("superblock"))
    }

    /// The superblock's bytes.
    #[cfg(any(test, feature = "test-support"))]
    pub fn encode(&self) -> [u8; SUPERBLOCK_SIZE] {
        let mut out = Out::<SUPERBLOCK_SIZE>::new();
        out.put(&self.magic);
        out.put(&self.header_size.to_le_bytes());
        out.put(&self.header_version.to_le_bytes());
        out.put(&self.toc_offset.to_le_bytes());
        out.put(&self.toc_size.to_le_bytes());
        out.put(&self.toc_sha256);
        out.put(&self.blob_offset.to_le_bytes());
        out.put(&self.file_size.to_le_bytes());
        out.put(&self.package_count.to_le_bytes());
        out.put(&self.flags.to_le_bytes());
        out.put(&self.uuid);
        out.put(&self.reserved);
        out.finish()
    }
}

impl RawTocHeader {
    /// Decodes the table of contents header at the start of `bytes`.
    pub fn decode(bytes: &[u8]) -> Result<Self, NupError> {
        let mut le = Le::new(bytes);
        let mut fields = || {
            Some(Self {
                magic: le.take()?,
                package_count: le.u32()?,
                dir_count: le.u64()?,
                file_count: le.u64()?,
                string_pool_offset: le.u64()?,
                string_pool_size: le.u64()?,
                reserved: le.u64()?,
            })
        };
        fields().ok_or_else(|| short("table of contents header"))
    }

    /// The header's bytes.
    #[cfg(any(test, feature = "test-support"))]
    pub fn encode(&self) -> [u8; TOC_HEADER_SIZE] {
        let mut out = Out::<TOC_HEADER_SIZE>::new();
        out.put(&self.magic);
        out.put(&self.package_count.to_le_bytes());
        out.put(&self.dir_count.to_le_bytes());
        out.put(&self.file_count.to_le_bytes());
        out.put(&self.string_pool_offset.to_le_bytes());
        out.put(&self.string_pool_size.to_le_bytes());
        out.put(&self.reserved.to_le_bytes());
        out.finish()
    }
}

impl RawPackage {
    /// Decodes the package record at the start of `bytes`.
    pub fn decode(bytes: &[u8]) -> Result<Self, NupError> {
        let mut le = Le::new(bytes);
        let mut fields = || {
            Some(Self {
                title_id: le.u64()?,
                root_dir: le.u32()?,
                exefs_dir: le.u32()?,
                romfs_dir: le.u32()?,
                version: le.u32()?,
                kind: le.take::<1>()?[0],
                flags: le.take::<1>()?[0],
                reserved: le.take::<2>().map(u16::from_le_bytes)?,
                reserved2: le.u32()?,
                meta_offset: le.u64()?,
                meta_size: le.u32()?,
                reserved3: le.take()?,
            })
        };
        fields().ok_or_else(|| short("package"))
    }

    /// The record's bytes.
    #[cfg(any(test, feature = "test-support"))]
    pub fn encode(&self) -> [u8; PACKAGE_RECORD_SIZE] {
        let mut out = Out::<PACKAGE_RECORD_SIZE>::new();
        out.put(&self.title_id.to_le_bytes());
        out.put(&self.root_dir.to_le_bytes());
        out.put(&self.exefs_dir.to_le_bytes());
        out.put(&self.romfs_dir.to_le_bytes());
        out.put(&self.version.to_le_bytes());
        out.put(&[self.kind, self.flags]);
        out.put(&self.reserved.to_le_bytes());
        out.put(&self.reserved2.to_le_bytes());
        out.put(&self.meta_offset.to_le_bytes());
        out.put(&self.meta_size.to_le_bytes());
        out.put(&self.reserved3);
        out.finish()
    }
}

impl RawDir {
    /// Decodes the directory record at the start of `bytes`.
    pub fn decode(bytes: &[u8]) -> Result<Self, NupError> {
        let mut le = Le::new(bytes);
        let mut fields = || {
            Some(Self {
                name_offset: le.u32()?,
                name_len: le.u32()?,
                first_child: le.u32()?,
                child_count: le.u32()?,
                reserved: le.u64()?,
            })
        };
        fields().ok_or_else(|| short("directory"))
    }

    /// The record's bytes.
    #[cfg(any(test, feature = "test-support"))]
    pub fn encode(&self) -> [u8; DIR_RECORD_SIZE] {
        let mut out = Out::<DIR_RECORD_SIZE>::new();
        out.put(&self.name_offset.to_le_bytes());
        out.put(&self.name_len.to_le_bytes());
        out.put(&self.first_child.to_le_bytes());
        out.put(&self.child_count.to_le_bytes());
        out.put(&self.reserved.to_le_bytes());
        out.finish()
    }
}

impl RawFile {
    /// Decodes the file record at the start of `bytes`.
    pub fn decode(bytes: &[u8]) -> Result<Self, NupError> {
        let mut le = Le::new(bytes);
        let mut fields = || {
            Some(Self {
                name_offset: le.u32()?,
                name_len: le.u32()?,
                size: le.u64()?,
                extent_offset: le.u64()?,
                extent_size: le.u64()?,
                sha256: le.take()?,
                codec: le.take::<1>()?[0],
                codec_flags: le.take::<1>()?[0],
                reserved2: le.take::<2>().map(u16::from_le_bytes)?,
                reserved3: le.take()?,
            })
        };
        fields().ok_or_else(|| short("file"))
    }

    /// The record's bytes.
    #[cfg(any(test, feature = "test-support"))]
    pub fn encode(&self) -> [u8; FILE_RECORD_SIZE] {
        let mut out = Out::<FILE_RECORD_SIZE>::new();
        out.put(&self.name_offset.to_le_bytes());
        out.put(&self.name_len.to_le_bytes());
        out.put(&self.size.to_le_bytes());
        out.put(&self.extent_offset.to_le_bytes());
        out.put(&self.extent_size.to_le_bytes());
        out.put(&self.sha256);
        out.put(&[self.codec, self.codec_flags]);
        out.put(&self.reserved2.to_le_bytes());
        out.put(&self.reserved3);
        out.finish()
    }
}

/// A fixed-size record being written front to back.
#[cfg(any(test, feature = "test-support"))]
struct Out<const N: usize> {
    bytes: [u8; N],
    at: usize,
}

#[cfg(any(test, feature = "test-support"))]
impl<const N: usize> Out<N> {
    fn new() -> Self {
        Self {
            bytes: [0; N],
            at: 0,
        }
    }

    /// Appends `field`. Each encoder above writes exactly `N` bytes, which the golden-bytes tests
    /// pin (every byte of each record is compared); a field past the end would be dropped.
    fn put(&mut self, field: &[u8]) {
        let end = self.at.saturating_add(field.len());
        if let Some(window) = self.bytes.get_mut(self.at..end) {
            window.copy_from_slice(field);
        }
        self.at = end;
    }

    /// The record.
    fn finish(self) -> [u8; N] {
        self.bytes
    }
}
