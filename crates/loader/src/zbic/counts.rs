//! FSE table descriptions: reading the one ZBIC stores, writing the one zstd stores.
//!
//! An FSE table is described by its accuracy log and one normalized count per symbol; the
//! counts' magnitudes add up to `1 << accuracy_log`, and -1 is a "less than one" count that
//! takes one slot. zstd stores the counts one after another in a bit stream (RFC 8878, section
//! 4.1.1). ZBIC stores them as one mixed-radix number, the counts as running totals in binary
//! interpolative order; the layout is in the module docs of [`super`], "The table description".
//!
//! [`read_bic`] is written from that layout, a fact read in `FSE_readNCount_bic`
//! (`atmosphere/libraries/libstratosphere/source/util/zstd.inc` @ 4e3f3703, fact only; the same
//! function is in the `zstd.c` of `kinnay/zbic` @ 11b08f27 that the reference compiles). No code
//! is taken from it: the reference walks a precomputed 768-entry index table, this walks the
//! intervals directly. [`write_zstd`] writes what RFC 8878's decoder reads.

use super::TableError;

/// The smallest accuracy log; both encodings store the log less this (`FSE_MIN_TABLELOG`;
/// RFC 8878, section 4.1.1, "`Accuracy_Log = low4bits + 5`").
pub(crate) const MIN_ACCURACY_LOG: u32 = 5;
/// Width of the accuracy log field of zstd's description (RFC 8878, section 4.1.1).
pub(crate) const ZSTD_LOG_BITS: u32 = 4;
/// Width of one zero-run flag of zstd's description: it follows a count of zero and says how
/// many more zero counts follow (RFC 8878, section 4.1.1).
pub(crate) const ZSTD_ZERO_RUN_BITS: u32 = 2;
/// The zero-run flag that says "three more, and another flag follows".
pub(crate) const ZSTD_ZERO_RUN_MORE: u32 = 3;

/// Bits 6-0 of a ZBIC description's first byte: how many bytes follow
/// (`rawDataSize = bitStream & 0x7F`).
pub(crate) const PAYLOAD_LEN_MASK: u8 = 0x7F;
/// Bit 7 of a ZBIC description's first byte: every count is stored plus one, so that -1 can be
/// stored (`useLowProbCount = bitStream >> 7`).
pub(crate) const LOW_COUNTS_FLAG: u8 = 0x80;
/// The radix of the first digit, the largest symbol less one (`i % 0x34`).
pub(crate) const SYMBOL_RADIX: u64 = 52;
/// The radix of the second digit, the accuracy log less [`MIN_ACCURACY_LOG`]
/// (`encodedCharTable & 0x7`).
pub(crate) const LOG_RADIX: u64 = 8;
/// The number is read a byte at a time, last byte first, while the part held is below this
/// (`((i >> 32) & 0xFFFFFFFF) >= 0x100` stops the reading loop).
pub(crate) const REFILL_BELOW: u64 = 1 << 40;
/// The most running totals a description holds besides the leading zero: the smallest power
/// of two above the largest symbol [`SYMBOL_RADIX`] allows.
const MAX_SLOTS: usize = 64;

/// The contents of an FSE table description.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Counts {
    /// The table has `1 << accuracy_log` slots.
    pub accuracy_log: u32,
    /// The normalized count of each symbol from 0 up; -1 is "less than one".
    pub counts: Vec<i32>,
}

impl Counts {
    /// The number of slots the table has.
    pub(crate) fn table_size(&self) -> u64 {
        1u64.checked_shl(self.accuracy_log).unwrap_or(0)
    }

    /// The number of slots the counts fill.
    pub(crate) fn sum(&self) -> u64 {
        self.counts
            .iter()
            .map(|count| u64::from(count.unsigned_abs()))
            .sum()
    }
}

/// The mixed-radix number of a ZBIC description, read from its last byte to its first.
struct Number<'a> {
    /// The part of the number held.
    state: u64,
    /// The bytes not read yet; the next one is the last.
    unread: &'a [u8],
}

impl<'a> Number<'a> {
    fn new(payload: &'a [u8]) -> Self {
        let mut number = Self {
            state: 0,
            unread: payload,
        };
        number.refill();
        number
    }

    /// Reads bytes while the state is below [`REFILL_BELOW`] and bytes remain.
    fn refill(&mut self) {
        while self.state < REFILL_BELOW {
            let Some((&byte, rest)) = self.unread.split_last() else {
                break;
            };
            self.state = (self.state << 8) | u64::from(byte);
            self.unread = rest;
        }
    }

    /// Takes the next digit, which is below `radix`.
    fn take(&mut self, radix: u64) -> u64 {
        // Every radix this module passes is at least 2.
        let radix = radix.max(1);
        let digit = self.state % radix;
        self.state /= radix;
        self.refill();
        digit
    }
}

/// Fills the running totals strictly between `low` and `high`: the middle one, then the upper
/// half, then the lower half. A middle total between two equal totals is that total and takes
/// no digit; otherwise its distance from the lower total is the next digit, in the radix one
/// more than the distance between the two.
fn fill(sums: &mut [u64; MAX_SLOTS + 1], low: usize, high: usize, number: &mut Number<'_>) {
    if high - low < 2 {
        return;
    }
    let middle = low + (high - low) / 2;
    // The indices are at most `MAX_SLOTS`, and a middle total lies between its neighbours, so
    // `last - first` cannot wrap.
    let (first, last) = (sums[low], sums[high]);
    sums[middle] = if first == last {
        first
    } else {
        first + number.take(last - first + 1)
    };
    fill(sums, middle, high, number);
    fill(sums, low, middle, number);
}

/// Reads the ZBIC table description that starts `src`, for a table whose largest symbol may be
/// `symbol_limit`, and returns it with its length in bytes.
///
/// `src` must run to the end of the bytes the reference hands its reader, because the
/// description has to end before them: `dataSize >= hbSize` is an error there, so a
/// description that fills `src` exactly is [`TableError::Truncated`].
pub(crate) fn read_bic(src: &[u8], symbol_limit: u32) -> Result<(Counts, usize), TableError> {
    let (&head, rest) = src.split_first().ok_or(TableError::Truncated)?;
    let payload_len = usize::from(head & PAYLOAD_LEN_MASK);
    let low_counts = u64::from(head & LOW_COUNTS_FLAG != 0);
    if payload_len >= rest.len() {
        return Err(TableError::Truncated);
    }
    let mut number = Number::new(rest.get(..payload_len).unwrap_or_default());

    let max_symbol = number.take(SYMBOL_RADIX) + 1;
    if max_symbol > u64::from(symbol_limit) {
        return Err(TableError::SymbolOutOfRange {
            max: max_symbol,
            limit: symbol_limit,
        });
    }
    let accuracy_log = MIN_ACCURACY_LOG + u32::try_from(number.take(LOG_RADIX)).unwrap_or(0);
    let table_size = 1u64 << accuracy_log;
    let stored_total = number.take(table_size);
    // Without the flag the total is the table size and is stored less one. With it every
    // count is one more, so the total is larger by the number of symbols, `max_symbol + 1`.
    let total = stored_total + 1 + low_counts * (max_symbol + 1);

    // `max_symbol` is at most `SYMBOL_RADIX`, so `slots` is at most `MAX_SLOTS`.
    let symbols = usize::try_from(max_symbol + 1).unwrap_or(MAX_SLOTS);
    let slots = symbols.next_power_of_two().min(MAX_SLOTS);
    let mut sums = [0u64; MAX_SLOTS + 1];
    sums[slots] = total;
    fill(&mut sums, 0, slots, &mut number);
    // One more digit is stored and thrown away, in the radix one more than the first symbol's
    // stored count, unless that count is zero (the reference's last step, for the indices
    // 0, 0 and 1).
    if sums[1] != 0 {
        number.take(sums[1] + 1);
    }

    let counts: Vec<i32> = sums
        .get(..=symbols)
        .unwrap_or_default()
        .windows(2)
        .map(|pair| match pair {
            [below, above] => {
                let stored = i64::try_from(above - below).unwrap_or(i64::MAX);
                let flag = i64::try_from(low_counts).unwrap_or(0);
                i32::try_from(stored - flag).unwrap_or(i32::MAX)
            }
            _ => 0,
        })
        .collect();
    let table = Counts {
        accuracy_log,
        counts,
    };
    if table.sum() != table_size {
        return Err(TableError::CountSum {
            sum: table.sum(),
            size: table_size,
        });
    }
    if !number.unread.is_empty() {
        return Err(TableError::UnreadBytes {
            unread: number.unread.len(),
        });
    }
    Ok((table, payload_len + 1))
}

/// Appends bits to a byte vector, lowest bit first.
pub(crate) struct BitWriter<'a> {
    out: &'a mut Vec<u8>,
    pending: u64,
    filled: u32,
}

impl<'a> BitWriter<'a> {
    pub(crate) fn new(out: &'a mut Vec<u8>) -> Self {
        Self {
            out,
            pending: 0,
            filled: 0,
        }
    }

    /// Appends the low `width` bits of `value`; `width` is at most 32.
    pub(crate) fn write(&mut self, value: u32, width: u32) {
        let mask = (1u64 << width.min(32)) - 1;
        self.pending |= (u64::from(value) & mask) << self.filled;
        self.filled += width.min(32);
        while self.filled >= 8 {
            self.flush_byte();
            self.filled -= 8;
        }
    }

    fn flush_byte(&mut self) {
        let [byte, ..] = self.pending.to_le_bytes();
        self.out.push(byte);
        self.pending >>= 8;
    }

    /// Pads the last byte with zero bits.
    pub(crate) fn finish(mut self) {
        if self.filled > 0 {
            self.flush_byte();
        }
    }
}

/// Appends `table` to `out` as zstd describes a table (RFC 8878, section 4.1.1): the accuracy
/// log, then each count plus one in a width that shrinks as the slots left do, with a run of
/// two-bit flags after a zero count for the zero counts that follow it. Zero counts after the
/// last other count are not stored.
///
/// The counts must add up to the table size ([`TableError::CountSum`] otherwise), which is
/// what [`read_bic`] returns.
pub(crate) fn write_zstd(table: &Counts, out: &mut Vec<u8>) -> Result<(), TableError> {
    let size = i64::try_from(table.table_size()).unwrap_or(i64::MAX);
    let log_field = table
        .accuracy_log
        .checked_sub(MIN_ACCURACY_LOG)
        .filter(|&field| field < 1 << ZSTD_LOG_BITS);
    let (Some(log_field), true) = (log_field, table.sum() == table.table_size()) else {
        return Err(TableError::CountSum {
            sum: table.sum(),
            size: table.table_size(),
        });
    };
    let mut bits = BitWriter::new(out);
    bits.write(log_field, ZSTD_LOG_BITS);

    // What RFC 8878's decoder tracks: the slots left plus one, the power of two the next
    // value is measured against, and that value's full width.
    let mut remaining = size + 1;
    let mut threshold = size;
    let mut width = table.accuracy_log + 1;
    // The zero counts seen since the zero count that started a run.
    let mut zero_run: Option<u32> = None;
    for &count in &table.counts {
        if remaining <= 1 {
            break;
        }
        if let Some(run) = zero_run.as_mut() {
            if count == 0 {
                *run += 1;
                continue;
            }
            for _ in 0..*run / ZSTD_ZERO_RUN_MORE {
                bits.write(ZSTD_ZERO_RUN_MORE, ZSTD_ZERO_RUN_BITS);
            }
            bits.write(*run % ZSTD_ZERO_RUN_MORE, ZSTD_ZERO_RUN_BITS);
            zero_run = None;
        }
        // Values below `short` take one bit less; values from `threshold` up are moved past
        // the ones that did.
        let short = 2 * threshold - 1 - remaining;
        remaining -= i64::from(count.unsigned_abs());
        let mut value = i64::from(count) + 1;
        if value >= threshold {
            value += short;
        }
        bits.write(
            u32::try_from(value).unwrap_or(0),
            width - u32::from(value < short),
        );
        if count == 0 {
            zero_run = Some(0);
        }
        while remaining < threshold {
            width -= 1;
            threshold >>= 1;
        }
    }
    bits.finish();
    Ok(())
}
