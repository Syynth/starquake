//! ZBIC: the zstd variant that compresses the segments of a Switch 2 NSO.
//!
//! [`decompress`] turns one ZBIC frame into exactly the number of bytes the NSO header
//! declares. It does not decode blocks itself: it rewrites the frame into the zstd frame that
//! holds the same data, a block at a time, and `ruzstd` (the decoder the NUP reader uses)
//! decodes that.
//!
//! # What ZBIC changes in zstd
//!
//! The reference decompressor, `eden-nx2/src/common/zbic_compression.cpp` @ 50fa86bb, is
//! zstd's `ZSTD_decompress` compiled with `ZSTD_ZBIC_SUPPORT` defined. The zstd it compiles is
//! the single file `src/zstd.c` of `kinnay/zbic` @ 11b08f27 (pinned by the `zbic` entry of
//! `eden-nx2/cpmfile.json`; `eden-pr4588/cpmfile.json` @ e5fb3b97 pins the same commit). That
//! repository says its file is copied from Atmosphère, and the copy in the pinned references,
//! `atmosphere/libraries/libstratosphere/source/util/zstd.inc` @ 4e3f3703, differs from it only
//! in the dictionary builder's `qsort` setup. Both are GPL-2.0 and are used for facts only:
//! nothing here is ported from them.
//!
//! The macro is used in three places:
//!
//! 1. `zstd.h`: `ZSTD_MAGICNUMBER` is `0x4349425A` (the bytes `ZBIC`) instead of `0xFD2FB528`.
//! 2. `FSE_readNCount_body` calls `FSE_readNCount_bic` instead of `FSE_readNCount_normal`. That
//!    function reads every FSE table description a decoder meets: the weights of a Huffman tree
//!    description (`HUF_readStats_body`, through `FSE_decompress_wksp_body`), the three sequence
//!    tables in their `FSE_Compressed` mode (`ZSTD_buildSeqTable`), and the tables of a
//!    dictionary (`ZSTD_loadDEntropy`).
//! 3. `FSE_writeNCount_generic` calls `FSE_writeNCount_bic`, the encoder's side of the same.
//!
//! So ZBIC changes the block format: a compressed block that describes an FSE table cannot be
//! read by a zstd decoder. Nothing else differs. Each section of that `zstd.c` was compared with
//! zstd 1.5.7's own source: besides the three places above, the only changes are the code they
//! call (the two `_bic` functions and their index table) and a `useLowProbCount` argument
//! that the encoder passes to `FSE_writeNCount`.
//!
//! ## The table description
//!
//! A fact read in `FSE_readNCount_bic`. The description is `1 + n` bytes:
//!
//! - Byte 0: bits 6-0 are `n`; bit 7 says every count is stored plus one (so a count of -1,
//!   "less than one", can be stored as 0).
//! - The `n` bytes after it are one number, read from the last byte to the first: the reader
//!   holds part of it and, whenever that part is below `2^40` and bytes remain, shifts the next
//!   byte in below it. A digit is taken as the part modulo the digit's radix, and the part is
//!   then divided by the radix.
//! - The digits, in order: the largest symbol less one (radix 52); the accuracy log less 5
//!   (radix 8); the total of the stored counts less one, or with bit 7 set less the number of
//!   symbols and one (radix: the table size); then the running totals of the stored counts.
//! - The running totals are the entries 1 to `p - 1` of a list whose entry 0 is zero and whose
//!   entry `p` is the total, `p` being the smallest power of two above the largest symbol.
//!   They are stored middle first: the middle entry of the list, then in the same way the upper
//!   half, then the lower half. An entry between two equal entries is equal to them and has no
//!   digit; any other is stored as its distance from the lower entry, in the radix one more
//!   than the distance between the two.
//! - One last digit is read and not used, in the radix one more than entry 1, unless entry 1
//!   is zero.
//! - Symbol `s`'s count is entry `s + 1` less entry `s` (less one more with bit 7 set).
//!
//! The reference refuses a description that does not end before the bytes it is read from do,
//! whose largest symbol is over the table's limit, whose counts' magnitudes do not add up to
//! the table size, or that leaves one of its `n` bytes unread; its caller then refuses an
//! accuracy log over the table's limit. [`TableError`] has one variant for each.
//!
//! # How a frame is decompressed
//!
//! The magic is replaced with zstd's and the frame header is kept. Raw and RLE blocks are
//! passed on as they are. In a compressed block every table description is read and written
//! back as zstd stores it (RFC 8878, section 4.1.1), and the three sizes that count its bytes
//! are corrected: the length byte of the Huffman weights, the compressed size in the literals
//! header, and the block size. The bit streams are not touched.
//!
//! A zstd description can be longer than the ZBIC one it replaces, and zstd's layout has three
//! limits a longer one can cross ([`Overflow`]). (Four-stream literals move to a header with
//! wider sizes instead.) What the reference's encoder (the same `zstd.inc`, fact only) does near
//! each, with what a block can grow by: the rewritten Huffman weights are at most 127 bytes
//! (else [`Overflow::HuffmanDescription`]), the three sequence tables at most 44, 35 and 63
//! bytes (zstd's own bound, `FSE_NCountWriteBound` with `MaxLL` 35 and `LLFSELog` 9, `MaxOff` 31
//! and `OffFSELog` 8, `MaxML` 52 and `MLFSELog` 9), and a four-stream literals header 2 bytes:
//!
//! - FSE-compressed Huffman weights, at most 127 bytes: not kept clear of. `HUF_writeCTable_wksp`
//!   compresses the weights whenever `(hSize>1) & (hSize < maxSymbolValue/2)`, where `hSize` is
//!   the length of the ZBIC weights and `maxSymbolValue` the literals' largest byte, at most 255
//!   (`HUF_SYMBOLVALUE_MAX`). So the encoder can write 126 bytes of weights, and a rewrite that
//!   grows them by two bytes crosses the limit. That a real title does not is a guess, which
//!   the titles read so far have not contradicted.
//! - Single-stream literals, a compressed size of at most 1023: kept clear of when there are
//!   fewer than 256 literals (`singleStream = srcSize < 256` in `ZSTD_compressLiterals`), as
//!   Huffman coding is kept only when the compressed size is below
//!   `srcSize - ZSTD_minGain(srcSize, strategy)`, so it is below 253, and a rewritten tree
//!   description of at most 127 bytes keeps it below 380. Not kept clear of when fewer than
//!   1 KiB of literals may reuse the previous block's tree
//!   (`if (repeat == HUF_repeat_valid && lhSize == 3) singleStream = 1;`) but the encoder writes
//!   a new one: the same condition then leaves a margin below 1023 of more than
//!   `ZSTD_minGain`, which is at least 2 bytes. That a real title does not cross it is a guess.
//! - A block, at most 128 KiB: kept clear of. `ZSTD_entropyCompressSeqStore_wExtLitBuffer` keeps
//!   a compressed block only when it is below `blockSize - ZSTD_minGain(blockSize, strategy)`,
//!   and `ZSTD_minGain` is `(blockSize >> minlog) + 2` with `minlog` at most 8 (`ZSTD_btultra2`
//!   less one), so a block ends at least 514 bytes below 128 KiB, more than the 271 bytes (127,
//!   44, 35, 63 and 2) a block can grow by.
//!
//! Should a title ever cross one, the way to read it is to decode that block's entropy tables
//! here instead of rewriting them, which this design leaves open.
//!
//! Memory: `ruzstd` reserves the frame's window when it reads the frame header, before any
//! block: `FrameDecoder::init` resets its buffer to the window (`DecodeBuffer::reset`, which
//! calls `RingBuffer::reserve`). A window is refused when it is over both [`MAX_WINDOW_SIZE`]
//! and the declared size, but a single-segment frame's window is its content size, which must
//! be the declared size, and a windowed frame may ask for up to the declared size. So a frame
//! of 12 bytes that states a content size of `u32::MAX`, the most an NSO header's 32-bit
//! segment size can declare, makes `ruzstd` reserve about 4 GiB
//! (`a_single_segment_frame_of_the_largest_size_is_refused_after_reserving_it`). Where the
//! host maps that lazily, as macOS and Linux do by default, it costs address space and no
//! memory; where the host refuses the allocation, `ruzstd` panics (`RingBuffer` `expect`s
//! the allocation), which this module cannot catch. The reference allocates the declared size
//! too, as its destination buffer. After that reservation the output grows as blocks are
//! decoded: until the frame ends the decoder holds back one window and at most one block
//! beyond it, as what it offers beyond the window is collected after every block
//! (`output_is_collected_while_a_long_frame_decodes`), and the output collected so far is
//! never over the declared size.
//!
//! # Divergences from the reference
//!
//! The reference is `DecompressDataZBIC` (`ZSTD_decompress` into a buffer of the declared
//! size) and its caller in `eden-nx2/src/core/loader/nso.cpp`, which accepts any result above
//! zero (`ASSERT(r > 0)`). The tests named are in `crates/loader/src/zbic/tests.rs`.
//!
//! Stricter here:
//!
//! - The output must be exactly the declared size; the reference accepts a shorter one
//!   ([`ZbicError::TooShort`], `output_shorter_than_declared_is_refused`).
//! - A frame header that states a content size must state the declared size; the reference
//!   only requires it to be what the frame decodes to and to fit
//!   ([`ZbicError::ContentSizeMismatch`], `a_content_size_must_be_the_declared_size`).
//! - The input must be one frame and nothing else. `ZSTD_decompress` decodes any number of
//!   frames one after another and skips skippable frames ([`ZbicError::TrailingData`] and
//!   [`ZbicError::BadMagic`], `a_second_frame_or_a_skippable_frame_is_refused`).
//! - A window over both [`MAX_WINDOW_SIZE`] and the declared size is refused; the reference
//!   accepts window logs up to 31 ([`ZbicError::WindowTooLarge`],
//!   `a_window_over_the_limit_is_refused_and_one_at_it_is_not`).
//! - A block that crosses one of the three limits above once rewritten is refused
//!   ([`ZbicError::Overflow`], `a_rewritten_block_over_a_zstd_limit_is_refused`).
//! - A raw or RLE block larger than the frame's window, or than 128 KiB, is refused by
//!   `ruzstd`; the reference's one-shot decoder copies such a block whatever its size
//!   ([`ZbicError::Zstd`], `a_block_over_the_window_or_over_128_kib_is_refused`).
//! - Some damaged Huffman-coded literals are refused by `ruzstd` where the reference decodes
//!   them to wrong bytes: its four-stream decoder does not check that every stream ends exactly
//!   ([`ZbicError::Zstd`],
//!   `corrupt_huffman_literals_are_refused_where_the_reference_decodes_them`).
//!
//! Looser here:
//!
//! - A declared size of zero is accepted when the frame is empty; `DecompressDataZBIC`
//!   returns -1 for an empty destination (`an_empty_frame_decodes_to_nothing`).
//! - The content checksum is read but not compared with the content, as in the NUP reader:
//!   `ruzstd` is built without its hash feature. An NSO segment has its own SHA-256
//!   (`a_content_checksum_is_read_but_not_checked`).
//! - Some damaged single-stream Huffman literals decode, to wrong bytes of the declared
//!   length, where the reference answers `corruption_detected`: `ruzstd` does not notice that
//!   the stream no longer ends exactly at its first byte
//!   (`a_corrupt_end_of_single_stream_literals_decodes_where_the_reference_refuses`).
//!
//! The same as the reference: a frame that names a dictionary is refused (`dictionary_wrong`
//! there, [`ZbicError::Dictionary`] here), and so are the frame header's reserved bit and the
//! reserved bits of a sequences section's modes byte, both of which `ruzstd` alone would let
//! through ([`ZbicError::ReservedBit`], [`ZbicError::ReservedModes`]). So are a compressed
//! block's literals when they are more than the frame's largest block, the smaller of its
//! window and 128 KiB (`litSize > blockSizeMax` in `ZSTD_decodeLiteralsBlock`, for raw, RLE
//! and Huffman literals, before anything after their header is read): `ruzstd` alone decodes
//! them when the block has no sequences ([`ZbicError::LiteralsTooLong`],
//! `literals_over_the_largest_block_are_refused`).
//!
//! The list comes from reading both sides and from running this module and a local build of
//! the reference (not part of this repository) over the same inputs: frames from the
//! reference's encoder with one byte changed, cut short or extended, and frames with generated
//! table descriptions, valid and damaged. Every disagreement those runs found is an entry
//! above, and no table description was judged differently. Blocks are decoded by `ruzstd`, so
//! a malformed Huffman or sequence stream that the runs did not produce may still be judged
//! differently from the reference; what `ruzstd` refuses comes back as [`ZbicError::Zstd`].

mod counts;
pub(crate) mod frame;

#[cfg(any(test, feature = "test-support"))]
pub mod build;
#[cfg(test)]
mod tests;

use ruzstd::decoding::{BlockDecodingStrategy, DEFAULT_MAX_WINDOW_SIZE, FrameDecoder};

use frame::{BlockReadError, HeaderError, RewriteError};
pub use frame::{Overflow, TableKind};

/// The first four bytes of a ZBIC frame: `0x4349425A` little-endian (`ZSTD_MAGICNUMBER` under
/// `ZSTD_ZBIC_SUPPORT`; `IsZBIC` in `eden-nx2/src/common/zbic_compression.cpp`).
pub const MAGIC: [u8; 4] = *b"ZBIC";

/// The largest window a frame may ask for when the declared size is smaller: 128 MiB, the
/// limit the NUP reader's decoder has (`ruzstd::decoding::DEFAULT_MAX_WINDOW_SIZE`). A
/// starquake limit; see "Divergences" in the module docs.
pub const MAX_WINDOW_SIZE: u64 = DEFAULT_MAX_WINDOW_SIZE;

#[cfg(test)]
thread_local! {
    /// The most output the decoder offered at once before the frame ended, in the last
    /// [`decompress`] on this thread: what it held beyond its window, for the tests.
    static MOST_OFFERED: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Whether `bytes` starts with the ZBIC magic (`IsZBIC` in
/// `eden-nx2/src/common/zbic_compression.cpp`: at least four bytes, the first four the magic).
pub fn is_zbic(bytes: &[u8]) -> bool {
    bytes.starts_with(&MAGIC)
}

/// Why a ZBIC table description was refused. See "The table description" in the module docs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum TableError {
    /// The description does not end before the bytes it is read from do.
    #[error("it does not end before the data that holds it")]
    Truncated,
    /// The largest symbol is over the table's limit.
    #[error("its largest symbol is {max}, over the limit of {limit}")]
    SymbolOutOfRange {
        /// The largest symbol the description names.
        max: u64,
        /// The table's limit ([`TableKind::max_symbol`]).
        limit: u32,
    },
    /// The accuracy log is over the table's limit.
    #[error("its accuracy log is {log}, over the limit of {limit}")]
    AccuracyLog {
        /// The description's accuracy log.
        log: u32,
        /// The table's limit ([`TableKind::max_accuracy_log`]).
        limit: u32,
    },
    /// The counts' magnitudes do not add up to the table size.
    #[error("its counts add up to {sum}, not to the table size {size}")]
    CountSum {
        /// What they add up to.
        sum: u64,
        /// The table size, `1 << accuracy_log`.
        size: u64,
    },
    /// The description's bytes were not all read.
    #[error("{unread} of its bytes are left over")]
    UnreadBytes {
        /// How many bytes were not read.
        unread: usize,
    },
}

/// Why a ZBIC frame could not be decompressed.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ZbicError {
    /// The first four bytes are not [`MAGIC`], or there are fewer than four.
    #[error("not a ZBIC frame: it does not start with the bytes \"ZBIC\"")]
    BadMagic,
    /// The data ends inside the frame header, a block, or the content checksum, or before a
    /// last block.
    #[error("the ZBIC frame is cut short")]
    Truncated,
    /// The frame header descriptor's reserved bit is set.
    #[error("the ZBIC frame header sets its reserved bit")]
    ReservedBit,
    /// The frame names a dictionary.
    #[error("the ZBIC frame needs dictionary {id:#x}, and an NSO has none")]
    Dictionary {
        /// The dictionary id.
        id: u32,
    },
    /// The frame's window is over the limit.
    #[error("the ZBIC frame's window of {window:#x} bytes is over the limit of {limit:#x}")]
    WindowTooLarge {
        /// The window the frame asks for.
        window: u64,
        /// The larger of [`MAX_WINDOW_SIZE`] and the declared size.
        limit: u64,
    },
    /// The frame header states a content size other than the declared one.
    #[error("the ZBIC frame says it holds {frame:#x} bytes, but {expected:#x} are declared")]
    ContentSizeMismatch {
        /// The frame header's content size.
        frame: u64,
        /// The declared size.
        expected: usize,
    },
    /// A block has the reserved type.
    #[error("ZBIC block {block} has the reserved type")]
    ReservedBlockType {
        /// The block's index in the frame.
        block: usize,
    },
    /// A compressed block's sections or tables run past its end.
    #[error("ZBIC block {block} is malformed: its contents run past its end")]
    BlockLayout {
        /// The block's index in the frame.
        block: usize,
    },
    /// A compressed block's literals are more than its largest block: the frame's window or
    /// 128 KiB, whichever is smaller.
    #[error(
        "ZBIC block {block} regenerates {size:#x} bytes of literals, more than its largest block \
         of {limit:#x}"
    )]
    LiteralsTooLong {
        /// The block's index in the frame.
        block: usize,
        /// The literals header's `Regenerated_Size`.
        size: u64,
        /// The largest block.
        limit: u64,
    },
    /// A compressed block's sequences section sets a reserved bit of its modes byte.
    #[error("ZBIC block {block} sets a reserved bit of its sequence table modes")]
    ReservedModes {
        /// The block's index in the frame.
        block: usize,
    },
    /// A table description was refused.
    #[error("ZBIC block {block}: the {table} table is not valid: {source}")]
    Table {
        /// The block's index in the frame.
        block: usize,
        /// Which table.
        table: TableKind,
        /// What is wrong with it.
        source: TableError,
    },
    /// The block does not fit zstd's layout once its tables are rewritten.
    #[error("ZBIC block {block} cannot be rewritten as zstd: {limit}")]
    Overflow {
        /// The block's index in the frame.
        block: usize,
        /// The limit it crosses.
        limit: Overflow,
    },
    /// `ruzstd` refused the rewritten frame.
    #[error("ZBIC block {block} does not decode: {message}")]
    Zstd {
        /// The block's index in the frame; 0 for the frame header.
        block: usize,
        /// The decoder's message.
        message: String,
    },
    /// The frame holds more than the declared size.
    #[error("the ZBIC frame holds more than the {expected:#x} bytes declared")]
    TooLong {
        /// The declared size.
        expected: usize,
    },
    /// The frame holds less than the declared size.
    #[error("the ZBIC frame holds {actual:#x} bytes, fewer than the {expected:#x} declared")]
    TooShort {
        /// The declared size.
        expected: usize,
        /// What the frame decoded to.
        actual: usize,
    },
    /// Bytes follow the frame.
    #[error("{extra} bytes follow the ZBIC frame")]
    TrailingData {
        /// How many bytes follow.
        extra: usize,
    },
}

impl From<HeaderError> for ZbicError {
    fn from(error: HeaderError) -> Self {
        match error {
            HeaderError::BadMagic => Self::BadMagic,
            HeaderError::Truncated => Self::Truncated,
            HeaderError::ReservedBit => Self::ReservedBit,
        }
    }
}

/// Reads one ZBIC table description and writes it as zstd's, with the reference's two checks
/// on what it describes.
fn bic_to_zstd(src: &[u8], table: TableKind, out: &mut Vec<u8>) -> Result<usize, TableError> {
    let (described, len) = counts::read_bic(src, table.max_symbol())?;
    if described.accuracy_log > table.max_accuracy_log() {
        return Err(TableError::AccuracyLog {
            log: described.accuracy_log,
            limit: table.max_accuracy_log(),
        });
    }
    counts::write_zstd(&described, out)?;
    Ok(len)
}

/// A `ruzstd` failure at `block`.
fn zstd_error(block: usize, error: &dyn std::fmt::Display) -> ZbicError {
    ZbicError::Zstd {
        block,
        message: error.to_string(),
    }
}

/// Appends compressed block `index` to `out` as zstd stores it: its block header with the
/// rewritten size, then its body with every table description rewritten.
///
/// Literals of more than `largest_block` are refused first, before anything after their
/// header is read, as `ZSTD_decodeLiteralsBlock` does (`litSize > blockSizeMax` is
/// `corruption_detected` for raw, RLE and Huffman literals).
fn rewrite_compressed(
    block: &frame::Block<'_>,
    index: usize,
    largest_block: u64,
    out: &mut Vec<u8>,
) -> Result<(), ZbicError> {
    if let Some(literals) = frame::Literals::parse(block.body)
        && literals.regenerated > largest_block
    {
        return Err(ZbicError::LiteralsTooLong {
            block: index,
            size: literals.regenerated,
            limit: largest_block,
        });
    }
    let start = out.len();
    out.extend_from_slice(&[0; frame::BLOCK_HEADER_SIZE]);
    frame::rewrite_block(block.body, &mut bic_to_zstd, out).map_err(|error| match error {
        RewriteError::Layout => ZbicError::BlockLayout { block: index },
        RewriteError::ReservedModes => ZbicError::ReservedModes { block: index },
        RewriteError::Table { table, source } => ZbicError::Table {
            block: index,
            table,
            source,
        },
        RewriteError::Overflow(limit) => ZbicError::Overflow {
            block: index,
            limit,
        },
    })?;
    let body_start = start.saturating_add(frame::BLOCK_HEADER_SIZE);
    let body_len = out.len().saturating_sub(body_start);
    let block_header =
        frame::block_header(block.last, block.kind, body_len).ok_or(ZbicError::Overflow {
            block: index,
            limit: Overflow::BlockSize,
        })?;
    if let Some(slot) = out.get_mut(start..body_start) {
        slot.copy_from_slice(&block_header);
    }
    Ok(())
}

/// Decompresses the ZBIC frame `stored` into exactly `size` bytes.
///
/// `stored` must be one whole frame and nothing else. Problems are reported in the order they
/// are met from the start of the frame: the header, then each block in turn (and, as soon as
/// the output is known to be over `size`, [`ZbicError::TooLong`]), then the output's length,
/// then bytes after the frame. See the module docs for what is checked and how that differs
/// from the reference.
pub fn decompress(stored: &[u8], size: usize) -> Result<Vec<u8>, ZbicError> {
    let header = frame::parse_header(stored, MAGIC)?;
    if header.dictionary != 0 {
        return Err(ZbicError::Dictionary {
            id: header.dictionary,
        });
    }
    let expected = u64::try_from(size).unwrap_or(u64::MAX);
    let limit = MAX_WINDOW_SIZE.max(expected);
    if header.window > limit {
        return Err(ZbicError::WindowTooLarge {
            window: header.window,
            limit,
        });
    }
    if let Some(frame) = header.content_size
        && frame != expected
    {
        return Err(ZbicError::ContentSizeMismatch {
            frame,
            expected: size,
        });
    }

    #[cfg(test)]
    MOST_OFFERED.set(0);
    let mut decoder = FrameDecoder::new();
    decoder.set_max_window_size(limit);
    let mut scratch = frame::ZSTD_MAGIC.to_vec();
    scratch.extend_from_slice(
        stored
            .get(frame::MAGIC_SIZE..header.len)
            .unwrap_or_default(),
    );
    decoder
        .init(scratch.as_slice())
        .map_err(|error| zstd_error(0, &error))?;

    // `blockSizeMax` (`ZSTD_getFrameHeader_advanced`): the smaller of the window and 128 KiB.
    let largest_block = header
        .window
        .min(u64::try_from(frame::BLOCK_SIZE_MAX).unwrap_or(u64::MAX));
    let mut out = Vec::new();
    let mut rest = stored.get(header.len..).unwrap_or_default();
    let mut index = 0usize;
    loop {
        let (block, after) = frame::read_block(rest).map_err(|error| match error {
            BlockReadError::Truncated => ZbicError::Truncated,
            BlockReadError::ReservedType => ZbicError::ReservedBlockType { block: index },
        })?;
        // The content checksum follows the last block and is read with it.
        let checksum_len = if block.last && header.checksum {
            frame::CHECKSUM_SIZE
        } else {
            0
        };
        let checksum = after.get(..checksum_len).ok_or(ZbicError::Truncated)?;
        rest = after.get(checksum_len..).unwrap_or_default();

        scratch.clear();
        if block.kind == frame::BLOCK_COMPRESSED {
            rewrite_compressed(&block, index, largest_block, &mut scratch)?;
        } else {
            scratch.extend_from_slice(block.stored);
        }
        scratch.extend_from_slice(checksum);
        decoder
            .decode_blocks(scratch.as_slice(), BlockDecodingStrategy::UptoBlocks(1))
            .map_err(|error| zstd_error(index, &error))?;

        if block.last {
            break;
        }
        // Until the frame ends the decoder holds back one window, so once it offers anything
        // the output so far is what was collected, the window and what it offers.
        let ready = decoder.can_collect();
        #[cfg(test)]
        MOST_OFFERED.set(MOST_OFFERED.get().max(ready));
        if ready > 0 {
            let total = u64::try_from(out.len().saturating_add(ready))
                .unwrap_or(u64::MAX)
                .saturating_add(header.window);
            if total > expected {
                return Err(ZbicError::TooLong { expected: size });
            }
            decoder
                .collect_to_writer(&mut out)
                .map_err(|error| zstd_error(index, &error))?;
        }
        index += 1;
    }

    let total = out.len().saturating_add(decoder.can_collect());
    if total > size {
        return Err(ZbicError::TooLong { expected: size });
    }
    if total < size {
        return Err(ZbicError::TooShort {
            expected: size,
            actual: total,
        });
    }
    decoder
        .collect_to_writer(&mut out)
        .map_err(|error| zstd_error(index, &error))?;
    if !rest.is_empty() {
        return Err(ZbicError::TrailingData { extra: rest.len() });
    }
    Ok(out)
}
