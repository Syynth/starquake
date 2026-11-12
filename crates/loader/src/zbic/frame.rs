//! The zstd frame layout that ZBIC keeps: the frame header, the block headers, and the two
//! sections of a compressed block, as far as they must be read to find every FSE table
//! description.
//!
//! [`rewrite_block`] copies a compressed block and hands each table description to a caller's
//! function, which writes it back in the other encoding. Decompression goes from ZBIC to zstd
//! ([`super::decompress`]); the fixture builder goes the other way (`super::build`). Nothing here
//! decodes a Huffman or FSE bit stream.
//!
//! # Sourcing
//!
//! The layout is zstd's, from RFC 8878 (the section is named at each constant). Each fact was
//! also read in the zstd that the reference compiles, `zstd.c` of `kinnay/zbic` @ 11b08f27, whose
//! copy in the pinned references is
//! `atmosphere/libraries/libstratosphere/source/util/zstd.inc` @ 4e3f3703 (fact only; the
//! function is named at each constant). See the module docs of [`super`] for how that file
//! differs from zstd.

/// zstd's frame magic, `0xFD2FB528` little-endian (RFC 8878, section 3.1.1;
/// `ZSTD_MAGICNUMBER` without `ZSTD_ZBIC_SUPPORT`).
pub(crate) const ZSTD_MAGIC: [u8; 4] = [0x28, 0xB5, 0x2F, 0xFD];
/// Size of a frame's magic.
pub(crate) const MAGIC_SIZE: usize = 4;
/// Offset of the frame header descriptor, after the magic (RFC 8878, section 3.1.1.1).
pub(crate) const DESCRIPTOR_OFFSET: usize = MAGIC_SIZE;
/// Descriptor bits 7-6: `Frame_Content_Size_flag` (RFC 8878, section 3.1.1.1.1;
/// `ZSTD_getFrameHeader_advanced`, `fcsID = fhdByte>>6`).
pub(crate) const FCS_FLAG_MASK: u8 = 0b1100_0000;
/// Shift of [`FCS_FLAG_MASK`].
pub(crate) const FCS_FLAG_SHIFT: u32 = 6;
/// Descriptor bit 5: `Single_Segment_flag` (`singleSegment = (fhdByte>>5)&1`).
pub(crate) const SINGLE_SEGMENT_FLAG: u8 = 0b0010_0000;
/// Descriptor bit 3: reserved, must be zero (`(fhdByte & 0x08) != 0` is
/// `frameParameter_unsupported`).
pub(crate) const RESERVED_FLAG: u8 = 0b0000_1000;
/// Descriptor bit 2: `Content_Checksum_flag` (`checksumFlag = (fhdByte>>2)&1`).
pub(crate) const CHECKSUM_FLAG: u8 = 0b0000_0100;
/// Descriptor bits 1-0: `Dictionary_ID_flag` (`dictIDSizeCode = fhdByte&3`).
pub(crate) const DICTIONARY_ID_MASK: u8 = 0b0000_0011;
/// Size of the `Dictionary_ID` field for each flag value (`ZSTD_did_fieldSize`).
pub(crate) const DICTIONARY_ID_SIZES: [usize; 4] = [0, 1, 2, 4];
/// Size of the `Frame_Content_Size` field for each flag value (`ZSTD_fcs_fieldSize`); flag 0 is
/// one byte instead when the frame is a single segment.
pub(crate) const FCS_SIZES: [usize; 4] = [0, 2, 4, 8];
/// Added to a two-byte `Frame_Content_Size` (`MEM_readLE16(ip+pos)+256`).
pub(crate) const FCS_TWO_BYTE_BIAS: u64 = 256;
/// The window descriptor's exponent is its top five bits, the mantissa the low three
/// (RFC 8878, section 3.1.1.1.2; `windowLog = (wlByte >> 3) + ZSTD_WINDOWLOG_ABSOLUTEMIN`).
pub(crate) const WINDOW_MANTISSA_BITS: u32 = 3;
/// The smallest window log (`ZSTD_WINDOWLOG_ABSOLUTEMIN`).
pub(crate) const WINDOW_LOG_MIN: u32 = 10;
/// Size of the content checksum that ends a frame whose descriptor sets [`CHECKSUM_FLAG`]
/// (RFC 8878, section 3.1.1).
pub(crate) const CHECKSUM_SIZE: usize = 4;

/// Size of a block header (RFC 8878, section 3.1.1.2; `ZSTD_blockHeaderSize`).
pub(crate) const BLOCK_HEADER_SIZE: usize = 3;
/// Block header bit 0: `Last_Block` (`ZSTD_getcBlockSize`, `cBlockHeader & 1`).
pub(crate) const BLOCK_LAST_FLAG: u32 = 1;
/// Block header bits 2-1: `Block_Type` (`(cBlockHeader >> 1) & 3`).
pub(crate) const BLOCK_TYPE_SHIFT: u32 = 1;
/// Mask of the block type after shifting.
pub(crate) const BLOCK_TYPE_MASK: u32 = 0b11;
/// Block header bits 23-3: `Block_Size` (`cBlockHeader >> 3`).
pub(crate) const BLOCK_SIZE_SHIFT: u32 = 3;
/// `Block_Type` 0: the body is the content.
pub(crate) const BLOCK_RAW: u32 = 0;
/// `Block_Type` 1: the body is one byte, repeated `Block_Size` times.
pub(crate) const BLOCK_RLE: u32 = 1;
/// `Block_Type` 2: a literals section and a sequences section.
pub(crate) const BLOCK_COMPRESSED: u32 = 2;
/// The largest block, stored or decoded: 128 KiB (RFC 8878, section 3.1.1.2.4;
/// `ZSTD_BLOCKSIZE_MAX`).
pub(crate) const BLOCK_SIZE_MAX: usize = 1 << 17;

/// Literals header bits 1-0: `Literals_Block_Type` (RFC 8878, section 3.1.1.3.1.1;
/// `ZSTD_decodeLiteralsBlock`, `istart[0] & 3`).
const LITERALS_TYPE_MASK: u8 = 0b11;
/// Literals header bits 3-2: `Size_Format` (`(istart[0] >> 2) & 3`).
pub(crate) const LITERALS_FORMAT_SHIFT: u32 = 2;
/// Mask of the size format after shifting.
const LITERALS_FORMAT_MASK: u8 = 0b11;
/// `Literals_Block_Type` 0: the literals themselves.
pub(crate) const LITERALS_RAW: u8 = 0;
/// `Literals_Block_Type` 1: one byte, repeated.
pub(crate) const LITERALS_RLE: u8 = 1;
/// `Literals_Block_Type` 2: Huffman-coded, with a tree description.
pub(crate) const LITERALS_COMPRESSED: u8 = 2;
/// `Literals_Block_Type` 3: Huffman-coded with the previous block's tree.
pub(crate) const LITERALS_TREELESS: u8 = 3;
/// The sizes of a literals header start at bit 4, after the type and the size format.
const LITERALS_SIZE_SHIFT: u32 = 4;
/// For Huffman-coded literals, by size format: the header's length in bytes and the width in
/// bits of each of its two sizes (`lhSize` 3, 3, 4, 5 and the `2 - 2 - 10 - 10`,
/// `2 - 2 - 14 - 14`, `2 - 2 - 18 - 18` layouts in `ZSTD_decodeLiteralsBlock`). Format 0 is one
/// stream; the others are four.
pub(crate) const HUFFMAN_HEADERS: [(usize, u32); 4] = [(3, 10), (3, 10), (4, 14), (5, 18)];
/// The size format of single-stream Huffman literals (`singleStream = !lhlCode`).
pub(crate) const HUFFMAN_ONE_STREAM: u8 = 0;
/// For raw and RLE literals, by size format: the header's length in bytes; the size is every
/// bit above the type and format, so above bit 3 for the one-byte header and above bit 4
/// otherwise (`litSize = istart[0] >> 3`, `MEM_readLE16(istart) >> 4`,
/// `MEM_readLE24(istart) >> 4`).
pub(crate) const PLAIN_HEADERS: [(usize, u32); 4] = [(1, 3), (2, 4), (1, 3), (3, 4)];

/// A Huffman tree description whose first byte is below this is FSE-compressed weights of that
/// many bytes; from it up, `byte - 127` weights follow as nibbles (RFC 8878, section 4.2.1;
/// `HUF_readStats_body`, `iSize >= 128`).
pub(crate) const HUFFMAN_DIRECT_WEIGHTS: u8 = 128;

/// A first sequence-count byte below this is the count (RFC 8878, section 3.1.1.3.2.1;
/// `ZSTD_decodeSeqHeaders`, `nbSeq > 0x7F`).
pub(crate) const SEQUENCE_COUNT_TWO_BYTES: u8 = 0x80;
/// A first sequence-count byte of this value is followed by a 16-bit count less
/// [`SEQUENCE_COUNT_LONG_BIAS`] (`nbSeq == 0xFF`).
pub(crate) const SEQUENCE_COUNT_THREE_BYTES: u8 = 0xFF;
/// Added to the 16-bit count of the three-byte form (`LONGNBSEQ`).
pub(crate) const SEQUENCE_COUNT_LONG_BIAS: u32 = 0x7F00;
/// Mask of one table's mode in the `Symbol_Compression_Modes` byte (RFC 8878, section
/// 3.1.1.3.2.1).
pub(crate) const MODE_MASK: u8 = 0b11;
/// The low two bits of the `Symbol_Compression_Modes` byte are reserved and must be zero
/// (`ZSTD_decodeSeqHeaders`: `*ip & 3` is `corruption_detected`).
pub(crate) const MODES_RESERVED_MASK: u8 = 0b11;
/// Mode 0: the predefined table (`set_basic`).
pub(crate) const MODE_PREDEFINED: u8 = 0;
/// Mode 1: one symbol, in one byte (`set_rle`).
pub(crate) const MODE_RLE: u8 = 1;
/// Mode 2: an FSE table description follows (`set_compressed`).
pub(crate) const MODE_FSE: u8 = 2;
/// Mode 3: the previous block's table (`set_repeat`).
pub(crate) const MODE_REPEAT: u8 = 3;

/// Which FSE table a description belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TableKind {
    /// The weights of a Huffman tree description.
    HuffmanWeights,
    /// A sequences section's literal-length table.
    LiteralLengths,
    /// A sequences section's offset table.
    Offsets,
    /// A sequences section's match-length table.
    MatchLengths,
}

impl TableKind {
    /// The three sequence tables in the order their descriptions are stored, each with the
    /// shift of its mode in the `Symbol_Compression_Modes` byte (`ZSTD_decodeSeqHeaders`:
    /// `LLtype = *ip >> 6`, `OFtype = (*ip >> 4) & 3`, `MLtype = (*ip >> 2) & 3`, built in that
    /// order).
    pub(crate) const SEQUENCE_TABLES: [(Self, u32); 3] = [
        (Self::LiteralLengths, 6),
        (Self::Offsets, 4),
        (Self::MatchLengths, 2),
    ];

    /// The largest symbol the reference lets this table describe: `MaxLL`, `MaxOff` and `MaxML`
    /// passed to `ZSTD_buildSeqTable`, and `FSE_MAX_SYMBOL_VALUE` in `FSE_decompress_wksp_body`.
    pub const fn max_symbol(self) -> u32 {
        match self {
            Self::HuffmanWeights => 255,
            Self::LiteralLengths => 35,
            Self::Offsets => 31,
            Self::MatchLengths => 52,
        }
    }

    /// The largest accuracy log the reference accepts for this table: `LLFSELog`, `OffFSELog`
    /// and `MLFSELog` in `ZSTD_buildSeqTable` (`tableLog > maxLog`), and the 6 that
    /// `HUF_readStats_body` passes to `FSE_decompress_wksp_bmi2`.
    pub const fn max_accuracy_log(self) -> u32 {
        match self {
            Self::HuffmanWeights => 6,
            Self::LiteralLengths | Self::MatchLengths => 9,
            Self::Offsets => 8,
        }
    }
}

impl std::fmt::Display for TableKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::HuffmanWeights => "Huffman weight",
            Self::LiteralLengths => "literal length",
            Self::Offsets => "offset",
            Self::MatchLengths => "match length",
        })
    }
}

/// What a frame header says.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FrameHeader {
    /// Length of the magic and the header: where the first block starts.
    pub len: usize,
    /// `Window_Size`: the content size of a single-segment frame, otherwise what the window
    /// descriptor says.
    pub window: u64,
    /// `Frame_Content_Size`, when the header has the field.
    pub content_size: Option<u64>,
    /// Whether a four-byte content checksum follows the last block.
    pub checksum: bool,
    /// `Dictionary_ID`; 0 when the field is absent or zero.
    pub dictionary: u32,
}

/// Why a frame header could not be read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum HeaderError {
    /// The first four bytes are not the expected magic (or there are fewer than four).
    BadMagic,
    /// The input ends inside the header.
    Truncated,
    /// The descriptor's reserved bit is set.
    ReservedBit,
}

/// A little-endian integer of `bytes.len()` (at most 8) bytes.
fn le(bytes: &[u8]) -> u64 {
    bytes
        .iter()
        .rev()
        .fold(0, |value, &byte| (value << 8) | u64::from(byte))
}

/// Reads the header of the frame that starts `bytes`, whose first four bytes must be `magic`.
///
/// Field order and sizes: `ZSTD_frameHeaderSize_internal` and `ZSTD_getFrameHeader_advanced`
/// (descriptor, window descriptor unless single-segment, dictionary id, content size).
pub(crate) fn parse_header(bytes: &[u8], magic: [u8; 4]) -> Result<FrameHeader, HeaderError> {
    if bytes.get(..MAGIC_SIZE) != Some(&magic[..]) {
        return Err(HeaderError::BadMagic);
    }
    let descriptor = *bytes.get(DESCRIPTOR_OFFSET).ok_or(HeaderError::Truncated)?;
    if descriptor & RESERVED_FLAG != 0 {
        return Err(HeaderError::ReservedBit);
    }
    let single_segment = descriptor & SINGLE_SEGMENT_FLAG != 0;
    let mut at = DESCRIPTOR_OFFSET + 1;
    let mut field = |len: usize| {
        let bytes = bytes.get(at..at + len).ok_or(HeaderError::Truncated)?;
        at += len;
        Ok(bytes)
    };

    let window_descriptor = if single_segment {
        None
    } else {
        field(1)?.first().copied()
    };
    let dictionary_size = DICTIONARY_ID_SIZES
        .get(usize::from(descriptor & DICTIONARY_ID_MASK))
        .copied()
        .unwrap_or_default();
    let dictionary = u32::try_from(le(field(dictionary_size)?)).unwrap_or(u32::MAX);
    let fcs_flag = usize::from((descriptor & FCS_FLAG_MASK) >> FCS_FLAG_SHIFT);
    let fcs_size = match FCS_SIZES.get(fcs_flag).copied().unwrap_or_default() {
        0 if single_segment => 1,
        size => size,
    };
    let content_size = match fcs_size {
        0 => None,
        2 => Some(le(field(2)?) + FCS_TWO_BYTE_BIAS),
        size => Some(le(field(size)?)),
    };
    let window = match window_descriptor {
        // A single-segment frame always has a content size (`windowSize = frameContentSize`).
        None => content_size.unwrap_or_default(),
        Some(byte) => {
            let log = WINDOW_LOG_MIN + u32::from(byte >> WINDOW_MANTISSA_BITS);
            let base = 1u64 << log;
            let mantissa = u64::from(byte & ((1 << WINDOW_MANTISSA_BITS) - 1));
            base + (base >> WINDOW_MANTISSA_BITS) * mantissa
        }
    };
    Ok(FrameHeader {
        len: at,
        window,
        content_size,
        checksum: descriptor & CHECKSUM_FLAG != 0,
        dictionary,
    })
}

/// One block of a frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Block<'a> {
    /// Whether this is the frame's last block.
    pub last: bool,
    /// `Block_Type`: [`BLOCK_RAW`], [`BLOCK_RLE`] or [`BLOCK_COMPRESSED`].
    pub kind: u32,
    /// The stored bytes after the header: `Block_Size` of them, or one for an RLE block.
    pub body: &'a [u8],
    /// The header and the body together.
    pub stored: &'a [u8],
}

/// Why a block could not be read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BlockReadError {
    /// The input ends inside the block's header or body.
    Truncated,
    /// The block has the reserved type 3.
    ReservedType,
}

/// Reads the block that starts `bytes` and returns it with the bytes after it.
///
/// `ZSTD_getcBlockSize`: an RLE block stores one byte whatever its `Block_Size`, and type 3 is
/// an error.
pub(crate) fn read_block(bytes: &[u8]) -> Result<(Block<'_>, &[u8]), BlockReadError> {
    let header = bytes
        .get(..BLOCK_HEADER_SIZE)
        .ok_or(BlockReadError::Truncated)?;
    let header = u32::try_from(le(header)).unwrap_or(u32::MAX);
    let kind = (header >> BLOCK_TYPE_SHIFT) & BLOCK_TYPE_MASK;
    let body_len = match kind {
        BLOCK_RAW | BLOCK_COMPRESSED => {
            usize::try_from(header >> BLOCK_SIZE_SHIFT).unwrap_or(usize::MAX)
        }
        BLOCK_RLE => 1,
        _ => return Err(BlockReadError::ReservedType),
    };
    let end = BLOCK_HEADER_SIZE
        .checked_add(body_len)
        .ok_or(BlockReadError::Truncated)?;
    let stored = bytes.get(..end).ok_or(BlockReadError::Truncated)?;
    let block = Block {
        last: header & BLOCK_LAST_FLAG != 0,
        kind,
        body: stored.get(BLOCK_HEADER_SIZE..).unwrap_or_default(),
        stored,
    };
    Ok((block, bytes.get(end..).unwrap_or_default()))
}

/// The header of a block of `kind` whose `Block_Size` is `size`, or `None` when `size` is over
/// [`BLOCK_SIZE_MAX`].
pub(crate) fn block_header(last: bool, kind: u32, size: usize) -> Option<[u8; BLOCK_HEADER_SIZE]> {
    if size > BLOCK_SIZE_MAX {
        return None;
    }
    let size = u32::try_from(size).ok()?;
    let header = (size << BLOCK_SIZE_SHIFT) | (kind << BLOCK_TYPE_SHIFT) | u32::from(last);
    let [a, b, c, _] = header.to_le_bytes();
    Some([a, b, c])
}

/// What cannot be written back when a re-encoded table description is longer than the one it
/// replaces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Overflow {
    /// A Huffman tree description with FSE-compressed weights would be over 127 bytes, the
    /// most its one-byte length can say.
    HuffmanDescription,
    /// The compressed size of Huffman literals would not fit their header: over 1023 bytes for
    /// single-stream literals, which have one header layout.
    LiteralsSize,
    /// The block would be over 128 KiB.
    BlockSize,
}

impl std::fmt::Display for Overflow {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::HuffmanDescription => "its Huffman tree description would be over 127 bytes",
            Self::LiteralsSize => "the compressed size of its literals would not fit their header",
            Self::BlockSize => "it would be over 128 KiB",
        })
    }
}

/// Why a compressed block could not be rewritten; `E` is the table function's error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum RewriteError<E> {
    /// A section, a size or a table description runs past the end of the block.
    Layout,
    /// The reserved bits of the `Symbol_Compression_Modes` byte are not zero.
    ReservedModes,
    /// A table description was refused.
    Table {
        /// Which table.
        table: TableKind,
        /// The table function's error.
        source: E,
    },
    /// The rewritten block does not fit zstd's layout.
    Overflow(Overflow),
}

/// The header of a block's literals section.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Literals {
    /// `Literals_Block_Type`.
    pub kind: u8,
    /// `Size_Format`.
    pub format: u8,
    /// `Regenerated_Size`.
    pub regenerated: u64,
    /// `Compressed_Size`; 0 for raw and RLE literals, which have no such field.
    pub compressed: u64,
    /// Length of the header.
    pub header_len: usize,
}

impl Literals {
    /// Reads the literals header that starts `body`.
    pub(crate) fn parse(body: &[u8]) -> Option<Self> {
        let first = *body.first()?;
        let kind = first & LITERALS_TYPE_MASK;
        let format = (first >> LITERALS_FORMAT_SHIFT) & LITERALS_FORMAT_MASK;
        if kind != LITERALS_COMPRESSED && kind != LITERALS_TREELESS {
            let (header_len, shift) = *PLAIN_HEADERS.get(usize::from(format))?;
            return Some(Self {
                kind,
                format,
                regenerated: le(body.get(..header_len)?) >> shift,
                compressed: 0,
                header_len,
            });
        }
        let (header_len, width) = *HUFFMAN_HEADERS.get(usize::from(format))?;
        let sizes = le(body.get(..header_len)?) >> LITERALS_SIZE_SHIFT;
        let mask = (1u64 << width) - 1;
        Some(Self {
            kind,
            format,
            regenerated: sizes & mask,
            compressed: (sizes >> width) & mask,
            header_len,
        })
    }

    /// The length of the whole section: the header and what it says follows.
    pub(crate) fn section_len(&self) -> Option<usize> {
        let payload = match self.kind {
            LITERALS_RAW => self.regenerated,
            LITERALS_RLE => 1,
            _ => self.compressed,
        };
        self.header_len.checked_add(usize::try_from(payload).ok()?)
    }
}

/// The header of Huffman-coded literals of `kind` with these sizes, in the smallest layout
/// from `format` up that holds them, or `None` when no layout does. Format 0 is the only
/// single-stream layout, so it is never changed.
pub(crate) fn huffman_literals_header(
    kind: u8,
    format: u8,
    regenerated: u64,
    compressed: u64,
) -> Option<Vec<u8>> {
    let candidates = if format == HUFFMAN_ONE_STREAM {
        HUFFMAN_ONE_STREAM..=HUFFMAN_ONE_STREAM
    } else {
        format..=LITERALS_FORMAT_MASK
    };
    for format in candidates {
        let (header_len, width) = *HUFFMAN_HEADERS.get(usize::from(format))?;
        let limit = 1u64 << width;
        if regenerated < limit && compressed < limit {
            let header = u64::from(kind)
                | (u64::from(format) << LITERALS_FORMAT_SHIFT)
                | (regenerated << LITERALS_SIZE_SHIFT)
                | (compressed << (LITERALS_SIZE_SHIFT + width));
            return Some(header.to_le_bytes().get(..header_len)?.to_vec());
        }
    }
    None
}

/// The number of sequences a sequences section announces and the length of that field
/// (`ZSTD_decodeSeqHeaders`).
pub(crate) fn sequence_count(section: &[u8]) -> Option<(u32, usize)> {
    let first = *section.first()?;
    if first < SEQUENCE_COUNT_TWO_BYTES {
        return Some((u32::from(first), 1));
    }
    if first < SEQUENCE_COUNT_THREE_BYTES {
        let low = *section.get(1)?;
        let high = u32::from(first - SEQUENCE_COUNT_TWO_BYTES);
        return Some(((high << 8) | u32::from(low), 2));
    }
    let count = u32::try_from(le(section.get(1..3)?)).ok()?;
    Some((count + SEQUENCE_COUNT_LONG_BIAS, 3))
}

/// Passes the description that starts `src` through `recode` and returns what follows it.
fn recode_table<'a, E>(
    recode: &mut impl FnMut(&[u8], TableKind, &mut Vec<u8>) -> Result<usize, E>,
    src: &'a [u8],
    table: TableKind,
    out: &mut Vec<u8>,
) -> Result<&'a [u8], RewriteError<E>> {
    let used = recode(src, table, out).map_err(|source| RewriteError::Table { table, source })?;
    src.get(used..).ok_or(RewriteError::Layout)
}

/// Copies the compressed block `body` to `out`, passing each FSE table description through
/// `recode`.
///
/// `recode(src, table, out)` reads the description that starts `src`, appends it to `out` in
/// the other encoding and returns how many bytes of `src` it read. `src` runs to the end of
/// what the reference gives its reader: the weights of a Huffman tree description, or the rest
/// of the block for a sequence table.
///
/// Descriptions are found where `ZSTD_decodeLiteralsBlock`, `HUF_readStats_body` and
/// `ZSTD_decodeSeqHeaders` find them. Everything else is copied as it is, except the sizes
/// that count a description's bytes: the one-byte length of FSE-compressed Huffman weights and
/// the literals header's compressed size (which moves to a wider four-stream layout when the
/// new size needs it). The caller writes the block header.
pub(crate) fn rewrite_block<E>(
    body: &[u8],
    recode: &mut impl FnMut(&[u8], TableKind, &mut Vec<u8>) -> Result<usize, E>,
    out: &mut Vec<u8>,
) -> Result<(), RewriteError<E>> {
    let literals = Literals::parse(body).ok_or(RewriteError::Layout)?;
    let section_len = literals.section_len().ok_or(RewriteError::Layout)?;
    let section = body.get(..section_len).ok_or(RewriteError::Layout)?;
    let payload = section.get(literals.header_len..).unwrap_or_default();
    match payload.split_first() {
        Some((&weights_len, tree))
            if literals.kind == LITERALS_COMPRESSED && weights_len < HUFFMAN_DIRECT_WEIGHTS =>
        {
            let weights_len = usize::from(weights_len);
            let weights = tree.get(..weights_len).ok_or(RewriteError::Layout)?;
            let streams = tree.get(weights_len..).unwrap_or_default();
            let mut description = Vec::new();
            let weights =
                recode_table(recode, weights, TableKind::HuffmanWeights, &mut description)?;
            description.extend_from_slice(weights);
            let new_len = u8::try_from(description.len())
                .ok()
                .filter(|&len| len < HUFFMAN_DIRECT_WEIGHTS)
                .ok_or(RewriteError::Overflow(Overflow::HuffmanDescription))?;
            let compressed = 1 + description.len() + streams.len();
            let header = huffman_literals_header(
                literals.kind,
                literals.format,
                literals.regenerated,
                u64::try_from(compressed).unwrap_or(u64::MAX),
            )
            .ok_or(RewriteError::Overflow(Overflow::LiteralsSize))?;
            out.extend_from_slice(&header);
            out.push(new_len);
            out.extend_from_slice(&description);
            out.extend_from_slice(streams);
        }
        _ => out.extend_from_slice(section),
    }

    let sequences = body.get(section_len..).unwrap_or_default();
    let (count, count_len) = sequence_count(sequences).ok_or(RewriteError::Layout)?;
    if count == 0 {
        // No modes byte and no tables (`nbSeq == 0`: "section ends immediately").
        out.extend_from_slice(sequences);
        return Ok(());
    }
    let modes = *sequences.get(count_len).ok_or(RewriteError::Layout)?;
    if modes & MODES_RESERVED_MASK != 0 {
        return Err(RewriteError::ReservedModes);
    }
    out.extend_from_slice(sequences.get(..=count_len).unwrap_or_default());
    let mut rest = sequences.get(count_len + 1..).unwrap_or_default();
    for (kind, shift) in TableKind::SEQUENCE_TABLES {
        match (modes >> shift) & MODE_MASK {
            MODE_FSE => rest = recode_table(recode, rest, kind, out)?,
            MODE_RLE => {
                let (symbol, after) = rest.split_first().ok_or(RewriteError::Layout)?;
                out.push(*symbol);
                rest = after;
            }
            // The predefined table and the previous block's table store nothing.
            MODE_PREDEFINED | MODE_REPEAT => {}
            // The two-bit mask leaves no other value; were there one, it would be reserved.
            _ => return Err(RewriteError::ReservedModes),
        }
    }
    out.extend_from_slice(rest);
    Ok(())
}
