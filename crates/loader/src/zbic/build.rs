//! Builds ZBIC frames, for tests (feature `test-support`).
//!
//! No ZBIC encoder is ported: the reference's is GPL-2.0-only (module docs of [`super`]).
//! Instead [`from_zstd`] does the opposite of what [`super::decompress`] does to a frame: it
//! takes a zstd frame, replaces the magic, and writes every FSE table description the way ZBIC
//! stores it. [`compress`] makes the zstd frame with `ruzstd`'s encoder first (pure Rust, the
//! crate the loader already depends on), which describes a table in every block that has
//! sequences and compresses the weights of a Huffman tree with more than 16 symbols.
//!
//! The table writer here is this crate's reading of the layout, the inverse of
//! [`super::counts::read_bic`]; it is not the reference's `FSE_writeNCount_bic`, and what it
//! writes need not be byte for byte what a console's packer would. Frames made by the
//! reference's own encoder are in `super::tests` as bytes.

use super::counts::{
    Counts, LOG_RADIX, LOW_COUNTS_FLAG, MIN_ACCURACY_LOG, PAYLOAD_LEN_MASK, REFILL_BELOW,
    SYMBOL_RADIX, ZSTD_LOG_BITS, ZSTD_ZERO_RUN_BITS, ZSTD_ZERO_RUN_MORE,
};
use super::frame::{self, RewriteError, TableKind};
use crate::nso::{
    self, COMPRESSED_SIZES_OFFSET, FLAG_COMPRESSED_SHIFT, FLAG_ZBIC, FLAGS_OFFSET, Nso, NsoError,
    SEGMENT_HEADER_SIZE, SEGMENT_HEADERS_OFFSET,
};

/// Why a ZBIC frame could not be built.
#[derive(Debug, thiserror::Error)]
pub enum BuildError {
    /// The input is not one whole zstd frame.
    #[error("the input is not one whole zstd frame")]
    NotAFrame,
    /// A compressed block could not be rewritten: it is malformed, or one of its tables has
    /// no ZBIC description ([`TableBuildError`]).
    #[error("zstd block {block} cannot be written as ZBIC")]
    Block {
        /// The block's index in the frame.
        block: usize,
        /// What is wrong with the table, when it is a table.
        #[source]
        source: Option<TableBuildError>,
    },
    /// The zstd frame could not be made.
    #[error("the zstd frame could not be made: {0}")]
    Zstd(#[from] crate::nup::build::BuildError),
    /// The NSO to recompress does not parse.
    #[error("the NSO does not parse: {0}")]
    Nso(#[from] NsoError),
}

/// Why a table has no ZBIC description, or why its zstd description could not be read.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum TableBuildError {
    /// The zstd description runs past the end of its data, or its counts overshoot the table
    /// size.
    #[error("the zstd table description is malformed")]
    Malformed,
    /// ZBIC stores accuracy logs 5 to 12, at most 53 symbols, and counts from -1 (or from 0
    /// without the low-counts flag) that add up to the table size.
    #[error("the table is outside what ZBIC can describe")]
    Unsupported,
    /// The number cannot be laid out so that the reader takes each byte when the writer
    /// expects: a digit would leave the part held below the reader's refill threshold. Rare
    /// (about one table in 2^27); change the data.
    #[error("the ZBIC description cannot be laid out for these counts")]
    Unencodable,
}

/// Reads `width` bits at bit `at` of `src`, lowest bit first; bits past the end read as zero.
fn bits_at(src: &[u8], at: usize, width: u32) -> u32 {
    let mut window = [0u8; 8];
    for (slot, byte) in window.iter_mut().zip(src.get(at / 8..).unwrap_or_default()) {
        *slot = *byte;
    }
    let value = u64::from_le_bytes(window) >> (at % 8);
    u32::try_from(value & ((1u64 << width.min(32)) - 1)).unwrap_or(0)
}

/// Reads the zstd table description that starts `src` (RFC 8878, section 4.1.1) and returns
/// it with its length in bytes.
pub(crate) fn read_zstd(src: &[u8]) -> Result<(Counts, usize), TableBuildError> {
    let mut at = 0usize;
    let mut read = |width: u32| {
        let value = bits_at(src, at, width);
        at += usize::try_from(width).unwrap_or(0);
        value
    };
    let accuracy_log = MIN_ACCURACY_LOG + read(ZSTD_LOG_BITS);
    let mut remaining = (1i64 << accuracy_log) + 1;
    let mut threshold = 1i64 << accuracy_log;
    let mut width = accuracy_log + 1;
    let mut counts = Vec::new();
    while remaining > 1 {
        if counts.last() == Some(&0) {
            loop {
                let run = read(ZSTD_ZERO_RUN_BITS);
                counts.extend((0..run).map(|_| 0));
                if run != ZSTD_ZERO_RUN_MORE {
                    break;
                }
            }
        }
        let short = 2 * threshold - 1 - remaining;
        // The value is in the low `width - 1` bits when those are below `short`; otherwise it
        // is `width` bits, and from `threshold` up it stands for the value less `short`.
        let peek = i64::from(read(width - 1));
        let value = if peek < short {
            peek
        } else {
            let full = peek | (i64::from(read(1)) << (width - 1));
            if full >= threshold {
                full - short
            } else {
                full
            }
        };
        let count = i32::try_from(value - 1).map_err(|_| TableBuildError::Malformed)?;
        remaining -= i64::from(count.unsigned_abs());
        if remaining < 1 || counts.len() > usize::from(u8::MAX) {
            return Err(TableBuildError::Malformed);
        }
        counts.push(count);
        while remaining < threshold {
            width -= 1;
            threshold >>= 1;
        }
    }
    let len = at.div_ceil(8);
    if len > src.len() {
        return Err(TableBuildError::Malformed);
    }
    Ok((
        Counts {
            accuracy_log,
            counts,
        },
        len,
    ))
}

/// The running totals of `table`'s stored counts (each count, plus one with `low_counts`):
/// entry 0 is zero, entry `s + 1` is the total up to symbol `s`, and the entries after the last
/// symbol repeat the total, up to entry `slots`.
fn running_totals(table: &Counts, low_counts: bool, slots: usize) -> Option<Vec<u64>> {
    let mut totals = vec![0u64];
    let mut total = 0u64;
    for &count in &table.counts {
        let stored = i64::from(count) + i64::from(low_counts);
        total += u64::try_from(stored).ok()?;
        totals.push(total);
    }
    totals.resize(slots + 1, total);
    Some(totals)
}

/// The digits for the totals strictly between `low` and `high`, in the order the reader
/// takes them (see `fill` in `super::counts`).
fn interval_digits(totals: &[u64], low: usize, high: usize, digits: &mut Vec<(u64, u64)>) {
    if high - low < 2 {
        return;
    }
    let middle = low + (high - low) / 2;
    if let (Some(&first), Some(&mid), Some(&last)) =
        (totals.get(low), totals.get(middle), totals.get(high))
        && first != last
    {
        digits.push((mid - first, last - first + 1));
    }
    interval_digits(totals, middle, high, digits);
    interval_digits(totals, low, middle, digits);
}

/// Writes `table` as ZBIC describes a table (module docs of [`super`], "The table
/// description"), with every count stored plus one when `low_counts` is set. A table with one
/// symbol gets a second with a count of zero, since the largest symbol is stored less one.
pub(crate) fn write_bic(table: &Counts, low_counts: bool) -> Result<Vec<u8>, TableBuildError> {
    let mut table = table.clone();
    if table.counts.len() < 2 {
        table.counts.resize(2, 0);
    }
    let max_symbol = u64::try_from(table.counts.len() - 1).unwrap_or(u64::MAX);
    let log_digit = table
        .accuracy_log
        .checked_sub(MIN_ACCURACY_LOG)
        .map(u64::from)
        .filter(|&digit| digit < LOG_RADIX);
    let (Some(log_digit), true, true) = (
        log_digit,
        max_symbol <= SYMBOL_RADIX,
        table.sum() == table.table_size(),
    ) else {
        return Err(TableBuildError::Unsupported);
    };
    let slots = table.counts.len().next_power_of_two();
    let totals = running_totals(&table, low_counts, slots).ok_or(TableBuildError::Unsupported)?;
    let total = totals.last().copied().unwrap_or_default();
    // The reader adds one to the stored total, and with the flag the number of symbols too;
    // counts that are nearly all -1 leave nothing to store.
    let stored_total = total
        .checked_sub(1 + u64::from(low_counts) * (max_symbol + 1))
        .ok_or(TableBuildError::Unsupported)?;

    let mut digits = vec![
        (max_symbol - 1, SYMBOL_RADIX),
        (log_digit, LOG_RADIX),
        (stored_total, table.table_size()),
    ];
    interval_digits(&totals, 0, slots, &mut digits);
    if let Some(&first) = totals.get(1)
        && first != 0
    {
        // The digit the reader throws away.
        digits.push((0, first + 1));
    }

    let payload = pack_digits(&digits)?;
    // 64 digits of at most 13 bits each are under 127 bytes, so the length always fits.
    let len = u8::try_from(payload.len())
        .ok()
        .filter(|len| len & !PAYLOAD_LEN_MASK == 0)
        .ok_or(TableBuildError::Unsupported)?;
    let mut out = vec![len | if low_counts { LOW_COUNTS_FLAG } else { 0 }];
    out.extend_from_slice(&payload);
    Ok(out)
}

/// The bytes of the number whose digits, as `(digit, radix)` in the order the reader takes
/// them, are `digits`.
///
/// The reader takes the first digit from the whole number, so the number is built from the
/// last digit back. The reader holds at most 48 bits at a time (it refills below
/// [`REFILL_BELOW`], a byte at a time); a byte it will shift in after taking a digit is one
/// the writer moves out before adding that digit. Once a byte has been moved out, the part
/// held must be at least [`REFILL_BELOW`] after every digit, or the reader would refill where
/// the writer did not move a byte out: [`TableBuildError::Unencodable`].
pub(crate) fn pack_digits(digits: &[(u64, u64)]) -> Result<Vec<u8>, TableBuildError> {
    let hold_below = REFILL_BELOW << 8;
    let mut state = 0u64;
    let mut payload = Vec::new();
    for &(digit, radix) in digits.iter().rev() {
        let next = loop {
            match state.checked_mul(radix).and_then(|n| n.checked_add(digit)) {
                Some(next) if next < hold_below => break next,
                _ => {
                    let [byte, ..] = state.to_le_bytes();
                    payload.push(byte);
                    state >>= 8;
                }
            }
        };
        if !payload.is_empty() && next < REFILL_BELOW {
            return Err(TableBuildError::Unencodable);
        }
        state = next;
    }
    while state != 0 {
        let [byte, ..] = state.to_le_bytes();
        payload.push(byte);
        state >>= 8;
    }
    Ok(payload)
}

/// Reads one zstd table description and writes it as ZBIC's. The low-counts flag is set when
/// the table has a count of -1, which cannot be stored without it.
fn zstd_to_bic(src: &[u8], _table: TableKind, out: &mut Vec<u8>) -> Result<usize, TableBuildError> {
    let (described, len) = read_zstd(src)?;
    let low_counts = described.counts.iter().any(|&count| count < 0);
    out.extend_from_slice(&write_bic(&described, low_counts)?);
    Ok(len)
}

/// The ZBIC frame that holds what the zstd frame `zstd` holds: the same header and blocks
/// under the ZBIC magic, with every FSE table description rewritten. `zstd` must be one whole
/// frame.
pub fn from_zstd(zstd: &[u8]) -> Result<Vec<u8>, BuildError> {
    let header = frame::parse_header(zstd, frame::ZSTD_MAGIC).map_err(|_| BuildError::NotAFrame)?;
    let mut out = super::MAGIC.to_vec();
    out.extend_from_slice(zstd.get(frame::MAGIC_SIZE..header.len).unwrap_or_default());
    let mut rest = zstd.get(header.len..).unwrap_or_default();
    let mut index = 0usize;
    loop {
        let (block, after) = frame::read_block(rest).map_err(|_| BuildError::NotAFrame)?;
        rest = after;
        if block.kind == frame::BLOCK_COMPRESSED {
            let mut body = Vec::new();
            frame::rewrite_block(block.body, &mut zstd_to_bic, &mut body).map_err(|error| {
                BuildError::Block {
                    block: index,
                    source: match error {
                        RewriteError::Table { source, .. } => Some(source),
                        RewriteError::Layout
                        | RewriteError::ReservedModes
                        | RewriteError::Overflow(_) => None,
                    },
                }
            })?;
            let block_header = frame::block_header(block.last, block.kind, body.len()).ok_or(
                BuildError::Block {
                    block: index,
                    source: None,
                },
            )?;
            out.extend_from_slice(&block_header);
            out.extend_from_slice(&body);
        } else {
            out.extend_from_slice(block.stored);
        }
        if block.last {
            break;
        }
        index += 1;
    }
    let checksum_len = if header.checksum {
        frame::CHECKSUM_SIZE
    } else {
        0
    };
    if rest.len() != checksum_len {
        return Err(BuildError::NotAFrame);
    }
    out.extend_from_slice(rest);
    Ok(out)
}

/// `data` as one ZBIC frame that states its content size, compressed by `ruzstd`'s encoder
/// (through the NUP builder's [`zstd_frame`](crate::nup::build::zstd_frame), which adds the
/// content size).
pub fn compress(data: &[u8]) -> Result<Vec<u8>, BuildError> {
    from_zstd(&crate::nup::build::zstd_frame(data)?)
}

/// The NSO `file` with all three segments stored as ZBIC frames: the same header, the
/// compression bits and [`FLAG_ZBIC`] set, and the segments laid out back to back after it.
pub fn recompress_nso(file: &[u8]) -> Result<Vec<u8>, BuildError> {
    let parsed = Nso::parse(file)?;
    let mut out = file.get(..nso::HEADER_SIZE).unwrap_or_default().to_vec();
    let mut flags = parsed.flags | FLAG_ZBIC;
    let word = |out: &mut Vec<u8>, at: usize, value: usize| {
        if let (Some(slot), Ok(value)) = (out.get_mut(at..at + 4), u32::try_from(value)) {
            slot.copy_from_slice(&value.to_le_bytes());
        }
    };
    for (index, segment) in parsed.segments().iter().enumerate() {
        let stored = compress(&segment.data)?;
        let offset = out.len();
        word(
            &mut out,
            SEGMENT_HEADERS_OFFSET + index * SEGMENT_HEADER_SIZE,
            offset,
        );
        word(&mut out, COMPRESSED_SIZES_OFFSET + index * 4, stored.len());
        flags |= 1 << (FLAG_COMPRESSED_SHIFT + u32::try_from(index).unwrap_or(0));
        out.extend_from_slice(&stored);
    }
    if let Some(slot) = out.get_mut(FLAGS_OFFSET..FLAGS_OFFSET + 4) {
        slot.copy_from_slice(&flags.to_le_bytes());
    }
    Ok(out)
}
