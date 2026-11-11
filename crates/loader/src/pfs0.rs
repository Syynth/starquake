//! `PFS0` partition filesystems (ExeFS) and their `HFS0` cousin.
//!
//! Read-only: [`Pfs0::new`] parses the header, the entry table and the string table of a
//! partition behind any `Read + Seek` source, and [`Pfs0::read`] / [`Pfs0::read_at`] pull file
//! bytes on demand, so a large image is never loaded as a whole. Only the metadata (header, entry
//! table, string table) is held in memory, and it is bounded by the source length before it is
//! allocated.
//!
//! Ported from `eden/src/core/file_sys/partition_filesystem.cpp` and `.h` @ cf654052
//! (GPL-3.0-or-later): `Header`, `FSEntry`, `PFSEntry`, `HFSEntry` and the
//! `PartitionFilesystem` constructor. Eden accepts both `PFS0` and `HFS0` magics, so this does too.
//!
//! Intentional divergences from Eden, each pinned by a test:
//!
//! - Eden loops over the entries with a `u16` counter although the header count is a `u32`; here
//!   the whole `u32` count is honoured (and bounded by the source length).
//! - Eden does not check at parse time that an entry's data lies inside the source (an
//!   `OffsetVfsFile` just returns fewer bytes when read); here an entry whose
//!   `content start + offset + size` overruns the source, or overflows, is
//!   [`Pfs0Error::EntryOutOfBounds`].
//! - Eden reads a name from a string-table offset it never checks; here an offset past the table
//!   is [`Pfs0Error::NameOffsetOutOfBounds`]. An offset equal to the table size yields an empty
//!   name, as Eden's extra trailing NUL does.
//! - Eden's names are raw bytes; here a name must be UTF-8 ([`Pfs0Error::NameNotUtf8`]).
//! - Eden ignores the `HFS0` hash fields (`hash_region_size`, the 0x20-byte hash); so does this.
//!
//! Matching Eden: a name that is not NUL-terminated ends at the end of the string table; entries
//! may share a name (`VfsDirectory::GetFile` returns the first), so [`Pfs0::find`] does the same.

use std::io::{Read, Seek, SeekFrom};
use std::ops::Range;

/// Magic of a `PFS0` partition (`Header::HasValidMagicValue`).
pub const PFS0_MAGIC: [u8; 4] = *b"PFS0";
/// Magic of an `HFS0` partition (`Header::HasValidMagicValue`).
pub const HFS0_MAGIC: [u8; 4] = *b"HFS0";
/// Size of the header (`static_assert(sizeof(Header) == 0x10)`).
pub const HEADER_SIZE: usize = 0x10;
/// Size of the part of an entry both kinds share (`static_assert(sizeof(FSEntry) == 0x14)`).
pub const FS_ENTRY_SIZE: usize = 0x14;
/// Size of a `PFS0` entry (`static_assert(sizeof(PFSEntry) == 0x18)`).
pub const PFS0_ENTRY_SIZE: usize = 0x18;
/// Size of an `HFS0` entry (`static_assert(sizeof(HFSEntry) == 0x40)`).
pub const HFS0_ENTRY_SIZE: usize = 0x40;

/// Which partition format a source holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pfs0Kind {
    /// `PFS0`: 0x18-byte entries.
    Pfs0,
    /// `HFS0`: 0x40-byte entries.
    Hfs0,
}

impl Pfs0Kind {
    /// The kind a magic selects, if it is one of the two.
    fn from_magic(magic: [u8; 4]) -> Option<Self> {
        match magic {
            PFS0_MAGIC => Some(Self::Pfs0),
            HFS0_MAGIC => Some(Self::Hfs0),
            _ => None,
        }
    }

    /// Size of one entry-table record.
    pub const fn entry_size(self) -> usize {
        match self {
            Self::Pfs0 => PFS0_ENTRY_SIZE,
            Self::Hfs0 => HFS0_ENTRY_SIZE,
        }
    }
}

/// One file of a partition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pfs0Entry {
    /// File name from the string table (`FSEntry::strtab_offset`).
    pub name: String,
    /// Offset of the data from the start of the content area (`FSEntry::offset`); see
    /// [`Pfs0::content_offset`].
    pub offset: u64,
    /// Size of the data in bytes (`FSEntry::size`).
    pub size: u64,
}

/// Why a partition could not be parsed or read.
#[derive(Debug, thiserror::Error)]
pub enum Pfs0Error {
    /// The source failed to seek or read.
    #[error("partition source I/O failed")]
    Io(#[from] std::io::Error),
    /// The source is shorter than the 0x10-byte header.
    #[error("partition is {len} bytes, shorter than the {HEADER_SIZE}-byte header")]
    Truncated {
        /// Length of the source.
        len: u64,
    },
    /// The magic is neither `PFS0` nor `HFS0`.
    #[error("partition magic is neither PFS0 nor HFS0")]
    BadMagic,
    /// The header, entry table and string table do not fit in the source.
    #[error("partition metadata needs {needed} bytes but the source is {len}")]
    MetadataOutOfBounds {
        /// Bytes of metadata the header declares.
        needed: u64,
        /// Length of the source.
        len: u64,
    },
    /// An entry's string-table offset lies past the end of the string table.
    #[error(
        "entry {index}: name offset {offset:#x} is past the {strtab_size:#x}-byte string table"
    )]
    NameOffsetOutOfBounds {
        /// Entry index.
        index: usize,
        /// The declared offset.
        offset: u32,
        /// Size of the string table.
        strtab_size: u32,
    },
    /// An entry's name is not valid UTF-8.
    #[error("entry {index}: name is not valid UTF-8")]
    NameNotUtf8 {
        /// Entry index.
        index: usize,
    },
    /// An entry's data extends past the end of the source (or its end overflows).
    #[error(
        "entry {index}: {size:#x} bytes at content offset {offset:#x} exceed the {len:#x}-byte source"
    )]
    EntryOutOfBounds {
        /// Entry index.
        index: usize,
        /// Declared offset from the content start.
        offset: u64,
        /// Declared size.
        size: u64,
        /// Length of the source.
        len: u64,
    },
    /// The entry handed to a read does not lie inside this partition's source.
    #[error(
        "read of {size:#x} bytes at content offset {offset:#x} exceeds the {len:#x}-byte source"
    )]
    ReadOutOfBounds {
        /// Offset from the content start.
        offset: u64,
        /// Size.
        size: u64,
        /// Length of the source.
        len: u64,
    },
}

/// A parsed `PFS0`/`HFS0` partition over a seekable source.
#[derive(Debug)]
pub struct Pfs0<R> {
    reader: R,
    len: u64,
    kind: Pfs0Kind,
    content_offset: u64,
    entries: Vec<Pfs0Entry>,
}

impl<R: Read + Seek> Pfs0<R> {
    /// Parses the partition that starts at offset 0 of `reader`.
    ///
    /// Every count, offset and size is checked against the source length before use, so malformed
    /// input is an error, never a panic or a runaway allocation.
    pub fn new(mut reader: R) -> Result<Self, Pfs0Error> {
        let len = source_len(&mut reader)?;
        if len < HEADER_SIZE as u64 {
            return Err(Pfs0Error::Truncated { len });
        }
        let mut header = [0u8; HEADER_SIZE];
        read_exact_at(&mut reader, 0, &mut header)?;
        let mut fields = Le::new(&header);
        let truncated = || Pfs0Error::Truncated { len };
        let magic = fields.take::<4>().ok_or_else(truncated)?;
        let num_entries = fields.u32().ok_or_else(truncated)?;
        let strtab_size = fields.u32().ok_or_else(truncated)?;
        let kind = Pfs0Kind::from_magic(magic).ok_or(Pfs0Error::BadMagic)?;

        // Eden: `metadata_size = sizeof(Header) + num_entries * entry_size + strtab_size`. In u64
        // this cannot overflow (2^32 * 0x40 + 2^32 + 0x10).
        let entry_size = kind.entry_size();
        let table_size = u64::from(num_entries) * entry_size as u64;
        let content_offset = HEADER_SIZE as u64 + table_size + u64::from(strtab_size);
        if content_offset > len {
            return Err(Pfs0Error::MetadataOutOfBounds {
                needed: content_offset,
                len,
            });
        }
        let metadata = read_vec_at(
            &mut reader,
            HEADER_SIZE as u64,
            content_offset - HEADER_SIZE as u64,
        )?;
        let table_len =
            usize::try_from(table_size).map_err(|_| Pfs0Error::MetadataOutOfBounds {
                needed: content_offset,
                len,
            })?;
        let (table, strtab) =
            metadata
                .split_at_checked(table_len)
                .ok_or(Pfs0Error::MetadataOutOfBounds {
                    needed: content_offset,
                    len,
                })?;

        let mut entries = Vec::with_capacity(table.len() / entry_size);
        for (index, record) in table.chunks_exact(entry_size).enumerate() {
            let mut fields = Le::new(record);
            let malformed = || Pfs0Error::MetadataOutOfBounds {
                needed: content_offset,
                len,
            };
            let offset = fields.u64().ok_or_else(malformed)?;
            let size = fields.u64().ok_or_else(malformed)?;
            let name_offset = fields.u32().ok_or_else(malformed)?;

            let name = name_at(strtab, name_offset).ok_or(Pfs0Error::NameOffsetOutOfBounds {
                index,
                offset: name_offset,
                strtab_size,
            })?;
            let name = std::str::from_utf8(name)
                .map_err(|_| Pfs0Error::NameNotUtf8 { index })?
                .to_owned();
            if checked_extent(content_offset, offset, size, len).is_none() {
                return Err(Pfs0Error::EntryOutOfBounds {
                    index,
                    offset,
                    size,
                    len,
                });
            }
            entries.push(Pfs0Entry { name, offset, size });
        }

        Ok(Self {
            reader,
            len,
            kind,
            content_offset,
            entries,
        })
    }

    /// Which format the source holds.
    pub fn kind(&self) -> Pfs0Kind {
        self.kind
    }

    /// Offset of the content area: the end of the header, entry table and string table. Entry
    /// offsets are relative to it (Eden: `content_offset`).
    pub fn content_offset(&self) -> u64 {
        self.content_offset
    }

    /// The entries in table order.
    pub fn entries(&self) -> &[Pfs0Entry] {
        &self.entries
    }

    /// The first entry called `name`.
    pub fn find(&self, name: &str) -> Option<&Pfs0Entry> {
        self.entries.iter().find(|entry| entry.name == name)
    }

    /// The byte range `entry` occupies in the source, checked against the source length.
    pub fn extent(&self, entry: &Pfs0Entry) -> Result<Range<u64>, Pfs0Error> {
        checked_extent(self.content_offset, entry.offset, entry.size, self.len).ok_or(
            Pfs0Error::ReadOutOfBounds {
                offset: entry.offset,
                size: entry.size,
                len: self.len,
            },
        )
    }

    /// Reads the whole file `entry` describes.
    pub fn read(&mut self, entry: &Pfs0Entry) -> Result<Vec<u8>, Pfs0Error> {
        let extent = self.extent(entry)?;
        Ok(read_vec_at(
            &mut self.reader,
            extent.start,
            extent.end - extent.start,
        )?)
    }

    /// Reads up to `buf.len()` bytes of `entry` starting `offset` bytes into the file, and returns
    /// how many were read: fewer than `buf.len()` only at the end of the file, and zero at or
    /// past it.
    pub fn read_at(
        &mut self,
        entry: &Pfs0Entry,
        offset: u64,
        buf: &mut [u8],
    ) -> Result<usize, Pfs0Error> {
        let extent = self.extent(entry)?;
        Ok(read_window(&mut self.reader, extent, offset, buf)?)
    }
}

/// Reads from `extent` (a range already checked against the source), `offset` bytes in, into
/// `buf`; returns the byte count. Shared with the `RomFS` reader.
pub(crate) fn read_window<R: Read + Seek>(
    reader: &mut R,
    extent: Range<u64>,
    offset: u64,
    buf: &mut [u8],
) -> std::io::Result<usize> {
    let remaining = (extent.end - extent.start).saturating_sub(offset);
    let count = usize::try_from(remaining).map_or(buf.len(), |remaining| remaining.min(buf.len()));
    let Some(window) = buf.get_mut(..count) else {
        return Ok(0);
    };
    if count == 0 {
        return Ok(0);
    }
    // `offset < extent.end - extent.start` here (count > 0), so the sum cannot pass `extent.end`.
    read_exact_at(reader, extent.start + offset, window)?;
    Ok(count)
}

/// The NUL-terminated name at `offset` in `strtab`; an unterminated name ends with the table.
/// `None` if `offset` is past the table.
fn name_at(strtab: &[u8], offset: u32) -> Option<&[u8]> {
    let tail = strtab.get(usize::try_from(offset).ok()?..)?;
    let end = tail.iter().position(|&b| b == 0).unwrap_or(tail.len());
    tail.get(..end)
}

/// The length of a seekable source (leaves the cursor at the end).
pub(crate) fn source_len<R: Seek>(reader: &mut R) -> std::io::Result<u64> {
    reader.seek(SeekFrom::End(0))
}

/// Fills `buf` from `offset`.
pub(crate) fn read_exact_at<R: Read + Seek>(
    reader: &mut R,
    offset: u64,
    buf: &mut [u8],
) -> std::io::Result<()> {
    reader.seek(SeekFrom::Start(offset))?;
    reader.read_exact(buf)
}

/// Reads `len` bytes from `offset`. The caller has checked `len` against the source length, which
/// is what bounds the allocation.
pub(crate) fn read_vec_at<R: Read + Seek>(
    reader: &mut R,
    offset: u64,
    len: u64,
) -> std::io::Result<Vec<u8>> {
    let len = usize::try_from(len).map_err(|_| {
        std::io::Error::new(std::io::ErrorKind::InvalidInput, "length overflows usize")
    })?;
    let mut buf = vec![0u8; len];
    read_exact_at(reader, offset, &mut buf)?;
    Ok(buf)
}

/// `base + offset .. base + offset + size` if the sum does not overflow and ends within `len`.
pub(crate) fn checked_extent(base: u64, offset: u64, size: u64, len: u64) -> Option<Range<u64>> {
    let start = base.checked_add(offset)?;
    let end = start.checked_add(size)?;
    (end <= len).then_some(start..end)
}

/// Sequential little-endian field reader over a byte slice. Reads return `None` past the end,
/// so a decoder written as a run of reads fails cleanly on a short slice.
pub(crate) struct Le<'a> {
    bytes: &'a [u8],
}

impl<'a> Le<'a> {
    /// Starts reading at the front of `bytes`.
    pub(crate) fn new(bytes: &'a [u8]) -> Self {
        Self { bytes }
    }

    /// The next `N` bytes.
    pub(crate) fn take<const N: usize>(&mut self) -> Option<[u8; N]> {
        let (head, rest) = self.bytes.split_at_checked(N)?;
        self.bytes = rest;
        <[u8; N]>::try_from(head).ok()
    }

    /// The next `len` bytes.
    pub(crate) fn bytes(&mut self, len: usize) -> Option<&'a [u8]> {
        let (head, rest) = self.bytes.split_at_checked(len)?;
        self.bytes = rest;
        Some(head)
    }

    /// The next little-endian `u32`.
    pub(crate) fn u32(&mut self) -> Option<u32> {
        self.take::<4>().map(u32::from_le_bytes)
    }

    /// The next little-endian `u64`.
    pub(crate) fn u64(&mut self) -> Option<u64> {
        self.take::<8>().map(u64::from_le_bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use std::io::Cursor;

    /// Fixed seed so property runs are reproducible.
    const PROPTEST_SEED: u64 = 0x5EED_0000_9F50;

    /// A file to put in a synthetic partition.
    #[derive(Debug, Clone)]
    struct Spec {
        name: String,
        content: Vec<u8>,
        /// Unused bytes between the previous file's end and this file.
        gap: usize,
    }

    /// Where the builder put things, for the tests to assert against.
    struct Built {
        bytes: Vec<u8>,
        content_offset: u64,
        /// Per file: offset from the content start.
        offsets: Vec<u64>,
    }

    /// Lays out a partition: header, entry table, string table (names optionally stored in
    /// reverse order, with `strtab_pad` trailing NULs), then each file after its `gap`.
    fn build(kind: Pfs0Kind, files: &[Spec], reverse_names: bool, strtab_pad: usize) -> Built {
        let mut strtab = Vec::new();
        let mut name_offsets = vec![0u32; files.len()];
        let order: Vec<usize> = if reverse_names {
            (0..files.len()).rev().collect()
        } else {
            (0..files.len()).collect()
        };
        for &i in &order {
            name_offsets[i] = u32::try_from(strtab.len()).unwrap();
            strtab.extend_from_slice(files[i].name.as_bytes());
            strtab.push(0);
        }
        strtab.extend(std::iter::repeat_n(0u8, strtab_pad));

        let mut content = Vec::new();
        let mut offsets = Vec::new();
        for file in files {
            content.extend(std::iter::repeat_n(0xEEu8, file.gap));
            offsets.push(content.len() as u64);
            content.extend_from_slice(&file.content);
        }

        let magic = match kind {
            Pfs0Kind::Pfs0 => PFS0_MAGIC,
            Pfs0Kind::Hfs0 => HFS0_MAGIC,
        };
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&magic);
        bytes.extend_from_slice(&u32::try_from(files.len()).unwrap().to_le_bytes());
        bytes.extend_from_slice(&u32::try_from(strtab.len()).unwrap().to_le_bytes());
        bytes.extend_from_slice(&[0u8; 4]);
        for (i, file) in files.iter().enumerate() {
            let before = bytes.len();
            bytes.extend_from_slice(&offsets[i].to_le_bytes());
            bytes.extend_from_slice(&(file.content.len() as u64).to_le_bytes());
            bytes.extend_from_slice(&name_offsets[i].to_le_bytes());
            // PFS0: 4 bytes of padding. HFS0: hash_region_size, 8 bytes padding, 0x20 hash; fill
            // with a pattern so a reader that mistook them for fields would be caught.
            let rest = kind.entry_size() - (bytes.len() - before);
            bytes.extend(std::iter::repeat_n(0xA5u8, rest));
        }
        bytes.extend_from_slice(&strtab);
        let content_offset = bytes.len() as u64;
        bytes.extend_from_slice(&content);
        Built {
            bytes,
            content_offset,
            offsets,
        }
    }

    fn spec(name: &str, content: &[u8]) -> Spec {
        Spec {
            name: name.to_owned(),
            content: content.to_vec(),
            gap: 0,
        }
    }

    fn parse(bytes: &[u8]) -> Result<Pfs0<Cursor<&[u8]>>, Pfs0Error> {
        Pfs0::new(Cursor::new(bytes))
    }

    fn sample() -> Built {
        build(
            Pfs0Kind::Pfs0,
            &[spec("main", b"MAIN-DATA"), spec("main.npdm", b"NPDM")],
            false,
            0,
        )
    }

    #[test]
    fn parses_and_reads_a_pfs0() {
        let built = sample();
        let mut pfs = parse(&built.bytes).unwrap();
        assert_eq!(pfs.kind(), Pfs0Kind::Pfs0);
        assert_eq!(pfs.content_offset(), built.content_offset);
        assert_eq!(pfs.entries().len(), 2);
        let main = pfs.find("main").unwrap().clone();
        assert_eq!(pfs.read(&main).unwrap(), b"MAIN-DATA");
        let npdm = pfs.find("main.npdm").unwrap().clone();
        assert_eq!(pfs.read(&npdm).unwrap(), b"NPDM");
        assert!(pfs.find("rtld").is_none());
    }

    #[test]
    fn parses_an_hfs0_and_ignores_its_hash_fields() {
        let built = build(
            Pfs0Kind::Hfs0,
            &[spec("a", b"x"), spec("b", b"yz")],
            false,
            0,
        );
        let mut pfs = parse(&built.bytes).unwrap();
        assert_eq!(pfs.kind(), Pfs0Kind::Hfs0);
        let b = pfs.find("b").unwrap().clone();
        assert_eq!(pfs.read(&b).unwrap(), b"yz");
        assert_eq!(pfs.content_offset(), built.content_offset);
    }

    #[test]
    fn an_empty_partition_has_no_entries() {
        let built = build(Pfs0Kind::Pfs0, &[], false, 0);
        let pfs = parse(&built.bytes).unwrap();
        assert_eq!(pfs.entries(), []);
        assert_eq!(pfs.content_offset(), HEADER_SIZE as u64);
    }

    #[test]
    fn rejects_short_and_misidentified_input() {
        assert!(matches!(parse(&[]), Err(Pfs0Error::Truncated { len: 0 })));
        let mut bytes = sample().bytes;
        bytes[0] = b'X';
        assert!(matches!(parse(&bytes), Err(Pfs0Error::BadMagic)));
    }

    #[test]
    fn rejects_metadata_that_does_not_fit() {
        let mut bytes = sample().bytes;
        // Entry count of u32::MAX: 2^32 * 0x18 bytes of table cannot fit (and must not be
        // allocated).
        bytes[4..8].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(matches!(
            parse(&bytes),
            Err(Pfs0Error::MetadataOutOfBounds { .. })
        ));
        let mut bytes = sample().bytes;
        bytes[8..12].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(matches!(
            parse(&bytes),
            Err(Pfs0Error::MetadataOutOfBounds { .. })
        ));
    }

    #[test]
    fn honours_entry_counts_beyond_sixteen_bits() {
        // Eden iterates with a u16 counter, so it never finishes a table with more than 0xFFFF
        // entries; here every entry of the u32 count is parsed.
        let count = usize::from(u16::MAX) + 3;
        let files = vec![spec("", b""); count];
        let built = build(Pfs0Kind::Pfs0, &files, false, 0);
        let pfs = parse(&built.bytes).unwrap();
        assert_eq!(pfs.entries().len(), count);
    }

    /// Patches the first entry's field (`at` is an offset within the entry) of a sample.
    fn with_first_entry(at: usize, value: &[u8]) -> Vec<u8> {
        let mut bytes = sample().bytes;
        let start = HEADER_SIZE + at;
        bytes[start..start + value.len()].copy_from_slice(value);
        bytes
    }

    #[test]
    fn rejects_entries_that_overrun_the_source() {
        // Offsets within FSEntry: offset 0, size 8, strtab_offset 0x10.
        let size_at = 8;
        let offset_at = 0;
        let bytes = with_first_entry(size_at, &u64::MAX.to_le_bytes());
        assert!(matches!(
            parse(&bytes),
            Err(Pfs0Error::EntryOutOfBounds { index: 0, .. })
        ));
        // offset + size overflows u64 even though each is individually plausible.
        let mut bytes = with_first_entry(offset_at, &(u64::MAX - 1).to_le_bytes());
        bytes[HEADER_SIZE + size_at..HEADER_SIZE + size_at + 8]
            .copy_from_slice(&2u64.to_le_bytes());
        assert!(matches!(
            parse(&bytes),
            Err(Pfs0Error::EntryOutOfBounds { index: 0, .. })
        ));
        // One byte past the end.
        let built = sample();
        let tail = built.bytes.len() as u64 - built.content_offset;
        let bytes = with_first_entry(size_at, &(tail + 1).to_le_bytes());
        assert!(matches!(
            parse(&bytes),
            Err(Pfs0Error::EntryOutOfBounds { index: 0, .. })
        ));
        // Exactly to the end is fine.
        let bytes = with_first_entry(size_at, &tail.to_le_bytes());
        assert!(parse(&bytes).is_ok());
    }

    #[test]
    fn name_offsets_are_bounded_by_the_string_table() {
        let name_offset_at = 0x10;
        let strtab_size = u32::from_le_bytes(sample().bytes[8..12].try_into().unwrap());
        let bytes = with_first_entry(name_offset_at, &(strtab_size + 1).to_le_bytes());
        assert!(matches!(
            parse(&bytes),
            Err(Pfs0Error::NameOffsetOutOfBounds { index: 0, .. })
        ));
        let bytes = with_first_entry(name_offset_at, &u32::MAX.to_le_bytes());
        assert!(matches!(
            parse(&bytes),
            Err(Pfs0Error::NameOffsetOutOfBounds { index: 0, .. })
        ));
        // An offset equal to the table size is an empty name (Eden's trailing NUL).
        let bytes = with_first_entry(name_offset_at, &strtab_size.to_le_bytes());
        let pfs = parse(&bytes).unwrap();
        assert_eq!(pfs.entries()[0].name, "");
    }

    #[test]
    fn an_unterminated_name_ends_with_the_string_table() {
        let mut built = build(Pfs0Kind::Pfs0, &[spec("abc", b"")], false, 0);
        // Overwrite the NUL that ends the only name with a letter.
        let nul_at = HEADER_SIZE + PFS0_ENTRY_SIZE + 3;
        assert_eq!(built.bytes[nul_at], 0);
        built.bytes[nul_at] = b'd';
        let pfs = parse(&built.bytes).unwrap();
        assert_eq!(pfs.entries()[0].name, "abcd");
    }

    #[test]
    fn names_must_be_utf8() {
        let mut built = build(Pfs0Kind::Pfs0, &[spec("abc", b"")], false, 0);
        built.bytes[HEADER_SIZE + PFS0_ENTRY_SIZE] = 0xFF;
        assert!(matches!(
            parse(&built.bytes),
            Err(Pfs0Error::NameNotUtf8 { index: 0 })
        ));
    }

    #[test]
    fn duplicate_names_resolve_to_the_first_entry() {
        let built = build(
            Pfs0Kind::Pfs0,
            &[spec("x", b"first"), spec("x", b"second")],
            false,
            0,
        );
        let mut pfs = parse(&built.bytes).unwrap();
        assert_eq!(pfs.entries().len(), 2);
        let entry = pfs.find("x").unwrap().clone();
        assert_eq!(pfs.read(&entry).unwrap(), b"first");
    }

    #[test]
    fn read_at_returns_windows_and_stops_at_the_end_of_the_file() {
        let built = build(
            Pfs0Kind::Pfs0,
            &[spec("a", b"0123456789"), spec("b", b"tail")],
            false,
            0,
        );
        let mut pfs = parse(&built.bytes).unwrap();
        let a = pfs.find("a").unwrap().clone();

        let mut buf = [0u8; 4];
        assert_eq!(pfs.read_at(&a, 2, &mut buf).unwrap(), 4);
        assert_eq!(&buf, b"2345");

        // A window crossing the end of the file is cut there, not extended into the next file.
        let mut buf = [0xFFu8; 8];
        assert_eq!(pfs.read_at(&a, 6, &mut buf).unwrap(), 4);
        assert_eq!(&buf[..4], b"6789");
        assert_eq!(&buf[4..], &[0xFF; 4]);

        assert_eq!(pfs.read_at(&a, 10, &mut buf).unwrap(), 0);
        assert_eq!(pfs.read_at(&a, u64::MAX, &mut buf).unwrap(), 0);
        assert_eq!(pfs.read_at(&a, 0, &mut []).unwrap(), 0);
    }

    #[test]
    fn a_forged_entry_cannot_read_outside_the_source() {
        let built = sample();
        let mut pfs = parse(&built.bytes).unwrap();
        for (offset, size) in [
            (u64::MAX, 1),
            (1, u64::MAX),
            (u64::MAX, u64::MAX),
            (0, built.bytes.len() as u64),
        ] {
            let forged = Pfs0Entry {
                name: "forged".into(),
                offset,
                size,
            };
            assert!(matches!(
                pfs.read(&forged),
                Err(Pfs0Error::ReadOutOfBounds { .. })
            ));
            assert!(matches!(
                pfs.read_at(&forged, 0, &mut [0u8; 4]),
                Err(Pfs0Error::ReadOutOfBounds { .. })
            ));
        }
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
                Ok(0x100)
            }
        }
        assert!(matches!(Pfs0::new(Failing), Err(Pfs0Error::Io(_))));
    }

    #[test]
    fn le_reads_fields_in_order_and_fails_cleanly_when_short() {
        let mut le = Le::new(&[1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 9]);
        assert_eq!(le.u32(), Some(1));
        assert_eq!(le.u64(), Some(2));
        assert_eq!(le.u32(), None);
        assert_eq!(le.take::<1>(), Some([9]));
        assert_eq!(le.take::<1>(), None);
    }

    /// Generated partitions: names are made unique by an index prefix, content lengths and gaps
    /// vary so that offsets, sizes and string-table offsets differ from one another.
    fn specs() -> impl Strategy<Value = Vec<Spec>> {
        proptest::collection::vec(
            (
                "[a-z0-9._é]{0,6}",
                proptest::collection::vec(any::<u8>(), 0..24),
                0usize..9,
            ),
            0..8,
        )
        .prop_map(|raw| {
            raw.into_iter()
                .enumerate()
                .map(|(i, (suffix, content, gap))| Spec {
                    name: format!("{i}{suffix}"),
                    content,
                    gap,
                })
                .collect()
        })
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            rng_seed: proptest::test_runner::RngSeed::Fixed(PROPTEST_SEED),
            ..ProptestConfig::default()
        })]

        /// Build then parse: every field of every entry must come back exactly, so reading any two
        /// fields in the wrong order fails here.
        #[test]
        fn build_then_parse_round_trips(
            files in specs(),
            hfs in any::<bool>(),
            reverse_names in any::<bool>(),
            strtab_pad in 0usize..5,
        ) {
            let kind = if hfs { Pfs0Kind::Hfs0 } else { Pfs0Kind::Pfs0 };
            let built = build(kind, &files, reverse_names, strtab_pad);
            let mut pfs = parse(&built.bytes).unwrap();
            prop_assert_eq!(pfs.kind(), kind);
            prop_assert_eq!(pfs.content_offset(), built.content_offset);
            prop_assert_eq!(pfs.entries().len(), files.len());
            for (i, file) in files.iter().enumerate() {
                let entry = pfs.entries()[i].clone();
                prop_assert_eq!(&entry.name, &file.name);
                prop_assert_eq!(entry.offset, built.offsets[i]);
                prop_assert_eq!(entry.size, file.content.len() as u64);
                prop_assert_eq!(pfs.extent(&entry).unwrap().start, built.content_offset + built.offsets[i]);
                prop_assert_eq!(pfs.read(&entry).unwrap(), file.content.clone());
            }
        }

        #[test]
        fn arbitrary_bytes_never_panic(bytes in proptest::collection::vec(any::<u8>(), 0..512)) {
            exercise(&bytes);
        }

        #[test]
        fn arbitrary_bytes_behind_a_valid_magic_never_panic(
            head in proptest::collection::vec(any::<u8>(), HEADER_SIZE),
            tail in proptest::collection::vec(any::<u8>(), 0..512),
            hfs in any::<bool>(),
        ) {
            let mut bytes = head;
            let magic = if hfs { HFS0_MAGIC } else { PFS0_MAGIC };
            bytes[..4].copy_from_slice(&magic);
            // Keep the counts small so the tail is actually reached.
            bytes[4..6].fill(0);
            bytes[8..10].fill(0);
            bytes.extend_from_slice(&tail);
            exercise(&bytes);
        }

        #[test]
        fn mutated_valid_partitions_never_panic(index in 0usize..0x100, value in any::<u8>()) {
            let mut bytes = sample().bytes;
            let at = index % bytes.len();
            bytes[at] = value;
            exercise(&bytes);
        }
    }

    /// Parses `bytes` and, if that works, uses every public method on every entry.
    fn exercise(bytes: &[u8]) {
        if let Ok(mut pfs) = parse(bytes) {
            let entries = pfs.entries().to_vec();
            for entry in &entries {
                let _ = pfs.find(&entry.name);
                let _ = pfs.extent(entry);
                let _ = pfs.read(entry);
                let _ = pfs.read_at(entry, 1, &mut [0u8; 8]);
            }
        }
    }
}
