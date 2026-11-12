//! Tests of the ZBIC decompressor.
//!
//! Three kinds of input, so that no property rests only on this crate's own builder:
//!
//! - Table descriptions, block bodies and frames written out by hand, a case for each layout
//!   the decompressor accepts. The table descriptions were worked out on paper from the layout
//!   in the module docs of [`super`]; the arithmetic is in each test.
//! - Frames made by the reference's own encoder from synthetic data (`crates/loader/tests/zbic`,
//!   whose `README.md` says how). The test regenerates the data with [`sample`].
//! - Frames made by [`build`], for the generated round trips.

use std::collections::BTreeSet;

use proptest::prelude::*;
use ruzstd::encoding::{CompressionLevel, compress_to_vec};

use super::build::{self, BuildError, TableBuildError, pack_digits, read_zstd, write_bic};
use super::counts::{
    Counts, LOG_RADIX, LOW_COUNTS_FLAG, MIN_ACCURACY_LOG, PAYLOAD_LEN_MASK, REFILL_BELOW,
    SYMBOL_RADIX, read_bic, write_zstd,
};
use super::frame::{
    BLOCK_COMPRESSED, BLOCK_HEADER_SIZE, BLOCK_RAW, BLOCK_RLE, BLOCK_SIZE_MAX, BLOCK_SIZE_SHIFT,
    BLOCK_TYPE_SHIFT, BlockReadError, CHECKSUM_FLAG, CHECKSUM_SIZE, DESCRIPTOR_OFFSET,
    DICTIONARY_ID_SIZES, FCS_FLAG_SHIFT, FCS_SIZES, FCS_TWO_BYTE_BIAS, FrameHeader,
    HUFFMAN_DIRECT_WEIGHTS, HUFFMAN_HEADERS, HUFFMAN_ONE_STREAM, HeaderError, LITERALS_COMPRESSED,
    LITERALS_FORMAT_SHIFT, LITERALS_RAW, LITERALS_RLE, LITERALS_TREELESS, Literals, MAGIC_SIZE,
    MODE_FSE, MODE_MASK, MODE_PREDEFINED, MODE_REPEAT, MODE_RLE, MODES_RESERVED_MASK,
    PLAIN_HEADERS, RESERVED_FLAG, RewriteError, SEQUENCE_COUNT_LONG_BIAS,
    SEQUENCE_COUNT_THREE_BYTES, SEQUENCE_COUNT_TWO_BYTES, SINGLE_SEGMENT_FLAG, WINDOW_LOG_MIN,
    WINDOW_MANTISSA_BITS, ZSTD_MAGIC, block_header, huffman_literals_header, parse_header,
    read_block, rewrite_block, sequence_count,
};
use super::*;

/// Fixed seed so property runs are reproducible.
const PROPTEST_SEED: u64 = 0x5EED_0000_2B1C;

fn seeded(cases: u32) -> ProptestConfig {
    ProptestConfig {
        cases,
        rng_seed: proptest::test_runner::RngSeed::Fixed(PROPTEST_SEED),
        ..ProptestConfig::default()
    }
}

// ---------------------------------------------------------------------------------------------
// Synthetic data
// ---------------------------------------------------------------------------------------------

/// The generator the fixtures' inputs were made with (`crates/loader/tests/zbic/README.md`).
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 33
    }

    /// The number of low one bits of the next value, at most `cap`, and the value shifted
    /// right by that many.
    fn ones(&mut self, cap: u64) -> (u64, u64) {
        let mut value = self.next();
        let mut ones = 0;
        while value & 1 == 1 && ones < cap {
            ones += 1;
            value >>= 1;
        }
        (ones, value)
    }
}

/// The shapes of synthetic data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    /// Words from a list of sixteen, each followed by one of five separators.
    Text,
    /// Bytes whose values thin out geometrically: Huffman-coded literals, few matches.
    Skewed,
    /// Runs of 1 to 300 equal bytes.
    Runs,
    /// Zero bytes.
    Zeros,
    /// Nine byte values, the low ones far more often: a Huffman tree small enough to store
    /// its weights directly.
    Nibbles,
    /// Four-letter words from a list of eight with nothing between them: many short matches.
    Short,
}

const WORDS: [&[u8]; 16] = [
    b"segment", b"module", b"kernel", b"loader", b"table", b"frame", b"block", b"window",
    b"symbol", b"offset", b"length", b"literal", b"count", b"state", b"stream", b"header",
];
const SEPARATORS: &[u8; 5] = b" \n,.;";
const SHORT_WORDS: [&[u8]; 8] = [
    b"abcd", b"efgh", b"ijkl", b"mnop", b"qrst", b"uvwx", b"yzAB", b"CDEF",
];

/// `len` bytes of `kind` from `seed`.
fn sample(kind: Kind, len: usize, seed: u64) -> Vec<u8> {
    let mut rng = Lcg(seed);
    let mut out = Vec::with_capacity(len + 512);
    let pick = |rng: &mut Lcg, count: usize| usize::try_from(rng.next()).unwrap() % count;
    while out.len() < len {
        match kind {
            Kind::Text => {
                out.extend_from_slice(WORDS[pick(&mut rng, WORDS.len())]);
                out.push(SEPARATORS[pick(&mut rng, SEPARATORS.len())]);
            }
            Kind::Skewed => {
                let (ones, rest) = rng.ones(40);
                out.push(u8::try_from((ones * 7 + (rest & 3)) & 0xFF).unwrap());
            }
            Kind::Runs => {
                let byte = u8::try_from(rng.next() & 0xFF).unwrap();
                let count = 1 + pick(&mut rng, 300);
                out.extend(std::iter::repeat_n(byte, count));
            }
            Kind::Zeros => out.resize(len, 0),
            Kind::Nibbles => out.push(u8::try_from(rng.ones(8).0).unwrap()),
            Kind::Short => out.extend_from_slice(SHORT_WORDS[pick(&mut rng, SHORT_WORDS.len())]),
        }
    }
    out.truncate(len);
    out
}

// ---------------------------------------------------------------------------------------------
// Table descriptions written by hand
// ---------------------------------------------------------------------------------------------

/// A byte that is not part of a description: the reference requires a description to end
/// before the data it is read from does.
const AFTER: u8 = 0xEE;

/// Reads the description `bytes` with one byte after it, as a match-length table (the kind
/// with the highest symbol limit).
fn read(bytes: &[u8]) -> Result<(Counts, usize), TableError> {
    let mut src = bytes.to_vec();
    src.push(AFTER);
    read_bic(&src, TableKind::MatchLengths.max_symbol())
}

fn counts(accuracy_log: u32, counts: &[i32]) -> Counts {
    Counts {
        accuracy_log,
        counts: counts.to_vec(),
    }
}

fn zstd_bytes(table: &Counts) -> Vec<u8> {
    let mut out = Vec::new();
    write_zstd(table, &mut out).unwrap();
    out
}

/// Two symbols with counts 20 and 12 in a table of 32, counts stored as they are.
///
/// Digits: largest symbol less one = 0 (radix 52); accuracy log less 5 = 0 (radix 8); total
/// less one = 31 (radix 32); the one running total, 20, between 0 and 32 (radix 33); the
/// digit that is thrown away, 0 (radix 21). As one number:
/// `0 + 52 * (0 + 8 * (31 + 32 * (20 + 33 * 0))) = 279136 = 0x044260`, three bytes, stored
/// low byte first because the reader starts at the last.
const PLAIN: [u8; 4] = [0x03, 0x60, 0x42, 0x04];
/// [`PLAIN`] with 7 as the digit that is thrown away:
/// `52 * 8 * (31 + 32 * (20 + 33 * 7)) = 3354208 = 0x332E60`.
const PLAIN_OTHER_LAST_DIGIT: [u8; 4] = [0x03, 0x60, 0x2E, 0x33];
/// What zstd stores for counts 20 and 12 in a table of 32: four bits 0 (log 5); 21 in five
/// bits (20 + 1 is below the 30 values that take the short width); then, 13 slots being left,
/// 12 + 1 moved past the 2 short values to 15, in four bits. `0 | 21 << 4 | 15 << 9 = 0x1F50`.
const PLAIN_ZSTD: [u8; 2] = [0x50, 0x1F];

#[test]
fn a_plain_description_reads_as_its_counts_and_writes_as_zstd() {
    let (table, len) = read(&PLAIN).unwrap();
    assert_eq!(table, counts(5, &[20, 12]));
    assert_eq!(len, PLAIN.len());
    assert_eq!(zstd_bytes(&table), PLAIN_ZSTD);

    // The last digit is read and not used.
    assert_eq!(read(&PLAIN_OTHER_LAST_DIGIT).unwrap(), (table, PLAIN.len()));
}

/// Four symbols with counts 29, -1, 0 and 2 in a table of 32, each stored plus one (30, 0,
/// 1, 3), so the running totals are 0, 30, 30, 31, 34.
///
/// Digits: 3 - 1 = 2 (radix 52); 0 (radix 8); the total less the four symbols and one,
/// 34 - 5 = 29 (radix 32); the middle total 30 between 0 and 34 (radix 35); then the upper
/// half, 31 between 30 and 34 as 1 (radix 5); then the lower half, 30 between 0 and 30 (radix
/// 31); the digit thrown away, 0 (radix 31).
/// `2 + 52 * 8 * (29 + 32 * (30 + 35 * (1 + 5 * 30))) = 70765346 = 0x0437CB22`.
const LOW_COUNTS: [u8; 5] = [0x84, 0x22, 0xCB, 0x37, 0x04];

#[test]
fn the_low_counts_flag_stores_each_count_plus_one() {
    assert_eq!(LOW_COUNTS[0] & LOW_COUNTS_FLAG, LOW_COUNTS_FLAG);
    let (table, len) = read(&LOW_COUNTS).unwrap();
    assert_eq!(table, counts(5, &[29, -1, 0, 2]));
    assert_eq!(len, LOW_COUNTS.len());
    // zstd: log 5; 30 as-is in six bits (not below the 30 short values); 3 slots left, so
    // widths of two bits: -1 is stored as 0 in two bits (no short values), then 0 as 1 in two
    // bits (2 slots left, one short value, and 1 is not below it) with a zero-run flag of 0,
    // then 2 as 3 in two bits.
    // `0 | 30 << 4 | 0 << 10 | 1 << 12 | 0 << 14 | 3 << 16 = 0x0311E0`.
    assert_eq!(zstd_bytes(&table), [0xE0, 0x11, 0x03]);
}

/// Four symbols, the first with all 32 slots: the running totals are 0, 32, 32, 32, 32.
///
/// Digits: 2 (radix 52); 0 (radix 8); 31 (radix 32); the middle total 32 between 0 and 32
/// (radix 33); the upper half lies between 32 and 32 and has no digit; the lower half, 32
/// between 0 and 32 (radix 33); the digit thrown away, 0 (radix 33).
/// `2 + 52 * 8 * (31 + 32 * (32 + 33 * 32)) = 14496354 = 0xDD3262`.
const ONE_SYMBOL_OF_FOUR: [u8; 4] = [0x03, 0x62, 0x32, 0xDD];

#[test]
fn a_total_between_two_equal_totals_takes_no_digit() {
    let (table, len) = read(&ONE_SYMBOL_OF_FOUR).unwrap();
    assert_eq!(table, counts(5, &[32, 0, 0, 0]));
    assert_eq!(len, ONE_SYMBOL_OF_FOUR.len());
    // zstd: log 5, then 33 moved past the 30 short values to 63, in six bits; the zero counts
    // after the last other count are not stored. `63 << 4 = 0x03F0`.
    assert_eq!(zstd_bytes(&table), [0xF0, 0x03]);
}

/// Two symbols with counts 0 and 32: the running totals are 0, 0, 32. The first count is
/// zero, so no digit is thrown away after the totals.
///
/// Digits: 0 (radix 52); 0 (radix 8); 31 (radix 32); the total 0 between 0 and 32 (radix 33).
/// `52 * 8 * 31 = 12896 = 0x3260`.
const FIRST_COUNT_ZERO: [u8; 3] = [0x02, 0x60, 0x32];

#[test]
fn the_last_digit_is_skipped_when_the_first_count_is_zero() {
    let (table, len) = read(&FIRST_COUNT_ZERO).unwrap();
    assert_eq!(table, counts(5, &[0, 32]));
    assert_eq!(len, FIRST_COUNT_ZERO.len());
    // zstd: log 5; 0 as 1 in five bits; a zero-run flag of 0; 32 as 63 in six bits.
    // `1 << 4 | 0 << 9 | 63 << 11 = 0x01F810`.
    assert_eq!(zstd_bytes(&table), [0x10, 0xF8, 0x01]);
}

/// [`PLAIN`]'s digits on top of a part of `2^40 + 5` that is left when they have been taken:
/// eight bytes, more than the reader holds at once, so it reads them in steps. Made with the
/// rule in the module docs: before a digit is added, the low bytes of the part are moved out
/// until the result is below `2^48`.
const PLAIN_ON_A_LEFT_OVER_PART: [u8; 9] = [0x08, 0x69, 0x9F, 0x40, 0x03, 0x00, 0x00, 0xC4, 0x8C];

#[test]
fn a_description_longer_than_the_reader_holds_is_read_in_steps() {
    let (table, len) = read(&PLAIN_ON_A_LEFT_OVER_PART).unwrap();
    assert_eq!(table, counts(5, &[20, 12]));
    assert_eq!(len, PLAIN_ON_A_LEFT_OVER_PART.len());
    // The builder's packer makes the same bytes from the same digits and left-over part.
    let digits = [
        (0, SYMBOL_RADIX),
        (0, LOG_RADIX),
        (31, 32),
        (20, 33),
        (0, 21),
    ];
    let mut on_left_over = digits.to_vec();
    on_left_over.extend(left_over_part(REFILL_BELOW + 5));
    assert_eq!(
        pack_digits(&on_left_over).unwrap(),
        PLAIN_ON_A_LEFT_OVER_PART[1..]
    );
}

/// Digits that leave `part` (below `2^48`) when everything before them has been taken:
/// `part` written in radix 4096.
fn left_over_part(part: u64) -> Vec<(u64, u64)> {
    (0..4)
        .map(|index| ((part >> (12 * index)) & 0xFFF, 4096))
        .collect()
}

#[test]
fn bytes_the_reader_never_reaches_are_refused() {
    // The left-over part is at least 2^40, so the reader does not refill after the last digit
    // and a byte before the others stays unread.
    let mut description = vec![PLAIN_ON_A_LEFT_OVER_PART[0] + 1, 0x99];
    description.extend_from_slice(&PLAIN_ON_A_LEFT_OVER_PART[1..]);
    assert_eq!(
        read(&description),
        Err(TableError::UnreadBytes { unread: 1 })
    );
    // With a left-over part just below 2^40 the reader does refill, and reads that byte.
    let mut digits = vec![
        (0, SYMBOL_RADIX),
        (0, LOG_RADIX),
        (31, 32),
        (20, 33),
        (0, 21),
    ];
    digits.extend(left_over_part(REFILL_BELOW - 1));
    let payload = pack_digits(&digits).unwrap();
    let mut description = vec![u8::try_from(payload.len() + 1).unwrap(), 0x99];
    description.extend_from_slice(&payload);
    assert_eq!(read(&description).unwrap().0, counts(5, &[20, 12]));
}

#[test]
fn a_description_must_end_before_the_data_it_is_read_from() {
    let limit = TableKind::MatchLengths.max_symbol();
    assert_eq!(read_bic(&[], limit), Err(TableError::Truncated));
    // The description alone: it ends exactly where its data does.
    assert_eq!(read_bic(&PLAIN, limit), Err(TableError::Truncated));
    // One byte short of that.
    assert_eq!(read_bic(&PLAIN[..3], limit), Err(TableError::Truncated));
    // One byte after it is enough.
    assert!(read(&PLAIN).is_ok());
    // The length is the low seven bits of the first byte: 127 bytes need 129 in all.
    let mut longest = vec![PAYLOAD_LEN_MASK];
    longest.resize(1 + usize::from(PAYLOAD_LEN_MASK), 0);
    assert_eq!(read_bic(&longest, limit), Err(TableError::Truncated));
    longest.push(AFTER);
    assert_ne!(read_bic(&longest, limit), Err(TableError::Truncated));
}

/// A table of 32 whose first symbol has every slot and whose largest symbol is `max_symbol`
/// (33 to 52, so the totals are a list of 64).
fn one_symbol_up_to(max_symbol: u64) -> Vec<u8> {
    // The middle total and five more down the lower half are 32 between 0 and 32; every
    // other total lies between 32 and 32. Then the digit that is thrown away.
    let mut digits = vec![(max_symbol - 1, SYMBOL_RADIX), (0, LOG_RADIX), (31, 32)];
    digits.extend([(32, 33); 6]);
    digits.push((0, 33));
    let payload = pack_digits(&digits).unwrap();
    let mut description = vec![u8::try_from(payload.len()).unwrap()];
    description.extend_from_slice(&payload);
    description
}

#[test]
fn a_symbol_over_the_tables_limit_is_refused_and_one_at_it_is_not() {
    for kind in [
        TableKind::LiteralLengths,
        TableKind::Offsets,
        TableKind::MatchLengths,
    ] {
        let limit = kind.max_symbol();
        // The reader's own bound is the radix of the digit: 52, the match-length limit.
        let over = u64::from(limit) + 1;
        let mut at = one_symbol_up_to(u64::from(limit).max(33));
        at.push(AFTER);
        if limit >= 33 {
            let (table, _) = read_bic(&at, limit).unwrap();
            assert_eq!(table.counts.len(), usize::try_from(limit).unwrap() + 1);
            assert_eq!(table.counts[0], 32);
        }
        if over <= SYMBOL_RADIX {
            let mut description = one_symbol_up_to(over.max(33));
            description.push(AFTER);
            assert_eq!(
                read_bic(&description, limit),
                Err(TableError::SymbolOutOfRange {
                    max: over.max(33),
                    limit
                })
            );
        }
    }
    // Small limits, on both sides: [`PLAIN`]'s largest symbol is 1.
    let mut src = PLAIN.to_vec();
    src.push(AFTER);
    assert!(read_bic(&src, 1).is_ok());
    assert_eq!(
        read_bic(&src, 0),
        Err(TableError::SymbolOutOfRange { max: 1, limit: 0 })
    );
}

#[test]
fn counts_that_do_not_fill_the_table_are_refused() {
    // [`PLAIN`] with a total of 31: `52 * 8 * (30 + 32 * 20) = 278720 = 0x0440C0`; the counts
    // are 20 and 11.
    assert_eq!(
        read(&[0x03, 0xC0, 0x40, 0x04]),
        Err(TableError::CountSum { sum: 31, size: 32 })
    );
    // An empty number: every digit is zero, so one symbol pair shares a total of 1.
    assert_eq!(
        read(&[0x00]),
        Err(TableError::CountSum { sum: 1, size: 32 })
    );
    // With the flag, counts of -1 count as one slot each: 29, -1, 0, 2 fill 32, and the same
    // totals with a last total of 33 (counts 29, -1, 0, 1) do not.
    let digits = [
        (2, SYMBOL_RADIX),
        (0, LOG_RADIX),
        (28, 32),
        (30, 34),
        (1, 4),
        (30, 31),
        (0, 31),
    ];
    let payload = pack_digits(&digits).unwrap();
    let mut description = vec![LOW_COUNTS_FLAG | u8::try_from(payload.len()).unwrap()];
    description.extend_from_slice(&payload);
    assert_eq!(
        read(&description),
        Err(TableError::CountSum { sum: 31, size: 32 })
    );
}

/// A table of `1 << log` whose first of two symbols has every slot.
fn one_symbol_at_log(log: u32) -> Vec<u8> {
    let size = 1u64 << log;
    let digits = [
        (0, SYMBOL_RADIX),
        (u64::from(log - MIN_ACCURACY_LOG), LOG_RADIX),
        (size - 1, size),
        (size, size + 1),
        (0, size + 1),
    ];
    let payload = pack_digits(&digits).unwrap();
    let mut description = vec![u8::try_from(payload.len()).unwrap()];
    description.extend_from_slice(&payload);
    description
}

#[test]
fn every_accuracy_log_the_digit_can_say_is_read() {
    // Written out for log 6: `52 * (1 + 8 * (63 + 64 * 64)) = 1730196 = 0x1A6694`.
    assert_eq!(one_symbol_at_log(6), [0x03, 0x94, 0x66, 0x1A]);
    for log in MIN_ACCURACY_LOG..MIN_ACCURACY_LOG + u32::try_from(LOG_RADIX).unwrap() {
        let (table, _) = read(&one_symbol_at_log(log)).unwrap();
        assert_eq!(table, counts(log, &[1 << log, 0]));
    }
}

#[test]
fn an_accuracy_log_over_the_tables_limit_is_refused_and_one_at_it_is_not() {
    for kind in [
        TableKind::HuffmanWeights,
        TableKind::LiteralLengths,
        TableKind::Offsets,
        TableKind::MatchLengths,
    ] {
        let limit = kind.max_accuracy_log();
        let mut out = Vec::new();
        let mut at = one_symbol_at_log(limit);
        at.push(AFTER);
        assert_eq!(bic_to_zstd(&at, kind, &mut out), Ok(at.len() - 1));
        assert_eq!(read_zstd(&out).unwrap().0, counts(limit, &[1 << limit]));

        let mut over = one_symbol_at_log(limit + 1);
        over.push(AFTER);
        assert_eq!(
            bic_to_zstd(&over, kind, &mut Vec::new()),
            Err(TableError::AccuracyLog {
                log: limit + 1,
                limit
            })
        );
    }
}

#[test]
fn zero_counts_are_written_as_runs() {
    // Counts 16, 0, 0, 0, 0, 16 in a table of 32. zstd: log 5; 17 in five bits; then with 17
    // slots left 0 as 1 in four bits, and for the three zero counts after it a flag of 3 and a
    // flag of 0; then 17 moved past the 14 short values to 31, in five bits.
    // `17 << 4 | 1 << 9 | 3 << 13 | 0 << 15 | 31 << 17 = 0x3E6310`.
    let table = counts(5, &[16, 0, 0, 0, 0, 16]);
    assert_eq!(zstd_bytes(&table), [0x10, 0x63, 0x3E]);
    // Every run length from none to seven reads back.
    for run in 0..8 {
        let mut values = vec![16, 0];
        values.extend(std::iter::repeat_n(0, run));
        values.push(16);
        let table = counts(5, &values);
        let bytes = zstd_bytes(&table);
        assert_eq!(
            read_zstd(&bytes).unwrap(),
            (table, bytes.len()),
            "run {run}"
        );
    }
}

#[test]
fn the_zstd_writer_refuses_counts_that_do_not_fill_the_table() {
    let mut out = Vec::new();
    assert_eq!(
        write_zstd(&counts(5, &[20, 11]), &mut out),
        Err(TableError::CountSum { sum: 31, size: 32 })
    );
    assert_eq!(
        write_zstd(&counts(5, &[20, 13]), &mut out),
        Err(TableError::CountSum { sum: 33, size: 32 })
    );
    // A log its four-bit field cannot say, on both sides of the field's range.
    assert!(write_zstd(&counts(4, &[16]), &mut out).is_err());
    assert!(write_zstd(&counts(20, &[1 << 20]), &mut out).is_ok());
    assert!(write_zstd(&counts(21, &[1 << 21]), &mut out).is_err());
    assert!(write_zstd(&counts(5, &[32]), &mut out).is_ok());
}

// ---------------------------------------------------------------------------------------------
// The builder's table writer and reader
// ---------------------------------------------------------------------------------------------

#[test]
fn the_builder_writes_the_descriptions_written_by_hand() {
    assert_eq!(write_bic(&counts(5, &[20, 12]), false).unwrap(), PLAIN);
    assert_eq!(
        write_bic(&counts(5, &[29, -1, 0, 2]), true).unwrap(),
        LOW_COUNTS
    );
    assert_eq!(
        write_bic(&counts(5, &[32, 0, 0, 0]), false).unwrap(),
        ONE_SYMBOL_OF_FOUR
    );
    assert_eq!(
        write_bic(&counts(5, &[0, 32]), false).unwrap(),
        FIRST_COUNT_ZERO
    );
    // One symbol gets a second with a count of zero.
    assert_eq!(
        read(&write_bic(&counts(5, &[32]), false).unwrap())
            .unwrap()
            .0,
        counts(5, &[32, 0])
    );
}

#[test]
fn the_builder_refuses_tables_zbic_cannot_describe() {
    let unsupported = Err(TableBuildError::Unsupported);
    // The log digit says 5 to 12.
    assert_eq!(write_bic(&counts(4, &[16]), false), unsupported);
    assert!(write_bic(&counts(12, &[4096]), false).is_ok());
    assert_eq!(write_bic(&counts(13, &[8192]), false), unsupported);
    // At most 53 symbols.
    let mut most = vec![0; 53];
    most[52] = 32;
    assert!(write_bic(&counts(5, &most), false).is_ok());
    most.insert(0, 0);
    assert_eq!(write_bic(&counts(5, &most), false), unsupported);
    // The counts must fill the table.
    assert_eq!(write_bic(&counts(5, &[20, 11]), false), unsupported);
    // -1 needs the flag, and nothing below -1 is a count.
    assert_eq!(write_bic(&counts(5, &[31, -1]), false), unsupported);
    assert!(write_bic(&counts(5, &[31, -1]), true).is_ok());
    assert_eq!(write_bic(&counts(5, &[30, -2]), true), unsupported);
    // The stored total is the sum of the counts less one: 16 counts of -1 and one of 16 add
    // up to 0, one fewer -1 and a 17 to 2.
    let mut even = vec![-1; 16];
    even.push(16);
    assert_eq!(write_bic(&counts(5, &even), true), unsupported);
    let mut odd = vec![-1; 15];
    odd.push(17);
    assert_eq!(
        read(&write_bic(&counts(5, &odd), true).unwrap()).unwrap().0,
        counts(5, &odd)
    );
}

#[test]
fn digits_that_would_leave_the_reader_short_are_refused() {
    // The part held is the smallest multiple-of-33 neighbour above 2^48 / 33, so adding a
    // digit in radix 33 needs a byte moved out, and what is left times 33 is 2^40 - 1: the
    // reader would refill one byte too many.
    let part = (1u64 << 48).div_ceil(33);
    assert_eq!((part >> 8) * 33, REFILL_BELOW - 1);
    let mut digits = vec![(0, 33)];
    digits.extend(left_over_part(part));
    assert_eq!(pack_digits(&digits), Err(TableBuildError::Unencodable));
    // One more in the part and the digit fits.
    let mut digits = vec![(0, 33)];
    digits.extend(left_over_part(part + 256));
    assert!(pack_digits(&digits).is_ok());
}

#[test]
fn the_builder_reads_zstd_descriptions() {
    assert_eq!(read_zstd(&PLAIN_ZSTD).unwrap(), (counts(5, &[20, 12]), 2));
    // Bytes after the description are not read.
    assert_eq!(
        read_zstd(&[PLAIN_ZSTD[0], PLAIN_ZSTD[1], 0xFF]).unwrap(),
        (counts(5, &[20, 12]), 2)
    );
    // Cut short: the second count is read from zero bits and overshoots nothing, but the
    // description then needs more bytes than there are.
    assert_eq!(read_zstd(&PLAIN_ZSTD[..1]), Err(TableBuildError::Malformed));
    assert_eq!(read_zstd(&[]), Err(TableBuildError::Malformed));
}

/// A valid table: the accuracy log, and for each symbol whether its count is zero, -1 or a
/// share of the slots.
fn table_strategy() -> impl Strategy<Value = (Counts, bool)> {
    (
        MIN_ACCURACY_LOG..MIN_ACCURACY_LOG + 8,
        proptest::collection::vec((0u8..4, 1u32..1000), 1..=53),
        any::<bool>(),
    )
        .prop_filter_map(
            "the counts must be able to fill the table",
            |(log, symbols, low)| {
                let size = 1u32 << log;
                let minus_ones =
                    u32::try_from(symbols.iter().filter(|(kind, _)| low && *kind == 1).count())
                        .ok()?;
                let shares: u32 = symbols
                    .iter()
                    .filter(|(kind, _)| *kind >= 2)
                    .map(|(_, weight)| weight)
                    .sum();
                let positives =
                    u32::try_from(symbols.iter().filter(|(kind, _)| *kind >= 2).count()).ok()?;
                // Every positive count is at least 1, and the counts add up to at least 1.
                let to_share = size.checked_sub(minus_ones)?.checked_sub(positives)?;
                if positives == 0 || size < 2 * minus_ones + 1 {
                    return None;
                }
                let mut values: Vec<i32> = symbols
                    .iter()
                    .map(|&(kind, weight)| match kind {
                        1 if low => -1,
                        0 | 1 => 0,
                        _ => i32::try_from(1 + to_share * weight / shares).unwrap(),
                    })
                    .collect();
                // What rounding left over goes to the first positive count.
                let given: u32 = values.iter().map(|count| count.unsigned_abs()).sum();
                let first = values.iter_mut().find(|count| **count > 0)?;
                *first += i32::try_from(size - given).ok()?;
                Some((
                    Counts {
                        accuracy_log: log,
                        counts: values,
                    },
                    low,
                ))
            },
        )
}

/// `table` without the zero counts after its last other count, which zstd does not store.
fn trimmed(table: &Counts) -> Counts {
    let len = table
        .counts
        .iter()
        .rposition(|&count| count != 0)
        .map_or(0, |last| last + 1);
    counts(table.accuracy_log, &table.counts[..len])
}

proptest! {
    #![proptest_config(seeded(2000))]

    /// Every digit of a description gets a generated value, so a reader that took two of them
    /// in the wrong order, or walked the totals in another order, returns other counts.
    #[test]
    fn generated_tables_round_trip_through_both_descriptions((table, low) in table_strategy()) {
        let description = write_bic(&table, low).unwrap();
        prop_assert_eq!(usize::from(description[0] & PAYLOAD_LEN_MASK), description.len() - 1);
        prop_assert_eq!(description[0] & LOW_COUNTS_FLAG != 0, low);
        let (read_back, len) = read(&description).unwrap();
        prop_assert_eq!(len, description.len());
        // A single symbol is described with a second one.
        let mut expected = table.clone();
        if expected.counts.len() < 2 {
            expected.counts.push(0);
        }
        prop_assert_eq!(&read_back, &expected);

        let zstd = zstd_bytes(&read_back);
        let (from_zstd, zstd_len) = read_zstd(&zstd).unwrap();
        prop_assert_eq!(zstd_len, zstd.len());
        prop_assert_eq!(from_zstd, trimmed(&table));
    }

    #[test]
    fn arbitrary_descriptions_never_panic_and_what_reads_also_writes(
        bytes in proptest::collection::vec(any::<u8>(), 0..160),
        limit in 0u32..300,
    ) {
        if let Ok((table, len)) = read_bic(&bytes, limit) {
            prop_assert!(len < bytes.len());
            prop_assert_eq!(table.sum(), table.table_size());
            let zstd = zstd_bytes(&table);
            prop_assert_eq!(read_zstd(&zstd).unwrap().0, trimmed(&table));
        }
        let _ = read_zstd(&bytes);
    }
}

// ---------------------------------------------------------------------------------------------
// Frame and block headers
// ---------------------------------------------------------------------------------------------

/// A frame header written field by field: the magic, the descriptor, then whichever of the
/// window descriptor, the dictionary id and the content size the descriptor announces.
fn header_bytes(
    magic: [u8; 4],
    descriptor: u8,
    window: Option<u8>,
    dictionary: &[u8],
    content_size: &[u8],
) -> Vec<u8> {
    let mut out = magic.to_vec();
    out.push(descriptor);
    out.extend(window);
    out.extend_from_slice(dictionary);
    out.extend_from_slice(content_size);
    out
}

#[test]
fn each_frame_header_layout_is_read() {
    let parse = |descriptor: u8, window: Option<u8>, dictionary: &[u8], fcs: &[u8]| {
        parse_header(
            &header_bytes(MAGIC, descriptor, window, dictionary, fcs),
            MAGIC,
        )
    };
    // A window descriptor and nothing else: exponent 3, mantissa 5 is 2^13 + 5 * 2^10.
    assert_eq!(
        parse(0, Some(3 << WINDOW_MANTISSA_BITS | 5), &[], &[]),
        Ok(FrameHeader {
            len: 6,
            window: (1 << (WINDOW_LOG_MIN + 3)) + 5 * (1 << WINDOW_LOG_MIN),
            content_size: None,
            checksum: false,
            dictionary: 0,
        })
    );
    // A single segment has no window descriptor and a one-byte content size, its window.
    assert_eq!(
        parse(SINGLE_SEGMENT_FLAG, None, &[], &[0x2A]),
        Ok(FrameHeader {
            len: 6,
            window: 0x2A,
            content_size: Some(0x2A),
            checksum: false,
            dictionary: 0,
        })
    );
    // Two bytes are the size less 256; four and eight are the size.
    let two = parse(1 << FCS_FLAG_SHIFT, Some(0), &[], &[0x34, 0x12]).unwrap();
    assert_eq!(two.content_size, Some(0x1234 + FCS_TWO_BYTE_BIAS));
    assert_eq!((two.len, two.window), (8, 1 << WINDOW_LOG_MIN));
    let four = parse(2 << FCS_FLAG_SHIFT, Some(0), &[], &[0x78, 0x56, 0x34, 0x12]).unwrap();
    assert_eq!((four.content_size, four.len), (Some(0x1234_5678), 10));
    let eight = parse(3 << FCS_FLAG_SHIFT, Some(0), &[], &[1, 2, 3, 4, 5, 6, 7, 8]).unwrap();
    assert_eq!(
        (eight.content_size, eight.len),
        (Some(0x0807_0605_0403_0201), 14)
    );
    // The dictionary id is one, two or four bytes, between the window and the content size.
    let one = parse(1, Some(0), &[0xAB], &[]).unwrap();
    assert_eq!((one.dictionary, one.len), (0xAB, 7));
    let two = parse(2, Some(0), &[0xCD, 0xAB], &[]).unwrap();
    assert_eq!((two.dictionary, two.len), (0xABCD, 8));
    let four = parse(3, Some(0), &[0x04, 0x03, 0x02, 0x01], &[]).unwrap();
    assert_eq!((four.dictionary, four.len), (0x0102_0304, 10));
    // The checksum flag, and the bit zstd does not use.
    assert!(parse(CHECKSUM_FLAG, Some(0), &[], &[]).unwrap().checksum);
    assert_eq!(
        parse(0b0001_0000, Some(0), &[], &[]),
        parse(0, Some(0), &[], &[])
    );
}

#[test]
fn a_frame_header_is_refused_for_its_magic_its_reserved_bit_or_its_length() {
    let whole = header_bytes(
        MAGIC,
        3 << FCS_FLAG_SHIFT | 3,
        Some(0),
        &[1, 2, 3, 4],
        &[9; 8],
    );
    assert!(parse_header(&whole, MAGIC).is_ok());
    for len in 0..MAGIC_SIZE {
        assert_eq!(
            parse_header(&whole[..len], MAGIC),
            Err(HeaderError::BadMagic)
        );
    }
    for len in MAGIC_SIZE..whole.len() {
        assert_eq!(
            parse_header(&whole[..len], MAGIC),
            Err(HeaderError::Truncated),
            "{len} bytes"
        );
    }
    assert_eq!(parse_header(&whole, ZSTD_MAGIC), Err(HeaderError::BadMagic));
    let mut reserved = whole.clone();
    reserved[DESCRIPTOR_OFFSET] |= RESERVED_FLAG;
    assert_eq!(
        parse_header(&reserved, MAGIC),
        Err(HeaderError::ReservedBit)
    );
    // The reserved bit is looked at before the fields are: a header cut after the descriptor.
    assert_eq!(
        parse_header(&reserved[..=DESCRIPTOR_OFFSET], MAGIC),
        Err(HeaderError::ReservedBit)
    );
}

proptest! {
    #![proptest_config(seeded(512))]

    /// The window descriptor, the dictionary id and the content size each get a generated
    /// value and a generated width, so a parser that read two of them in the wrong order, or
    /// with the wrong width, returns the wrong field.
    #[test]
    fn generated_frame_headers_round_trip(
        single_segment in any::<bool>(),
        checksum in any::<bool>(),
        window in any::<u8>(),
        dictionary_flag in 0usize..4,
        dictionary in any::<u32>(),
        fcs_flag in 0usize..4,
        content_size in any::<u64>(),
    ) {
        let dictionary_size = DICTIONARY_ID_SIZES[dictionary_flag];
        let dictionary = dictionary.to_le_bytes()[..dictionary_size].to_vec();
        let fcs_size = match FCS_SIZES[fcs_flag] {
            0 if single_segment => 1,
            size => size,
        };
        let content = content_size.to_le_bytes()[..fcs_size].to_vec();
        let descriptor = u8::try_from(fcs_flag << FCS_FLAG_SHIFT | dictionary_flag).unwrap()
            | if single_segment { SINGLE_SEGMENT_FLAG } else { 0 }
            | if checksum { CHECKSUM_FLAG } else { 0 };
        let bytes = header_bytes(
            MAGIC,
            descriptor,
            (!single_segment).then_some(window),
            &dictionary,
            &content,
        );
        let header = parse_header(&bytes, MAGIC).unwrap();

        let stated = content.iter().rev().fold(0u64, |value, &byte| value << 8 | u64::from(byte));
        let expected_content = match fcs_size {
            0 => None,
            2 => Some(stated + FCS_TWO_BYTE_BIAS),
            _ => Some(stated),
        };
        let expected_window = if single_segment {
            expected_content.unwrap()
        } else {
            let base = 1u64 << (WINDOW_LOG_MIN + u32::from(window >> WINDOW_MANTISSA_BITS));
            base + base / 8 * u64::from(window & 7)
        };
        prop_assert_eq!(header.len, bytes.len());
        prop_assert_eq!(header.window, expected_window);
        prop_assert_eq!(header.content_size, expected_content);
        prop_assert_eq!(header.checksum, checksum);
        let stated_dictionary =
            dictionary.iter().rev().fold(0u32, |value, &byte| value << 8 | u32::from(byte));
        prop_assert_eq!(header.dictionary, stated_dictionary);
    }
}

#[test]
fn a_block_header_packs_the_last_flag_the_type_and_the_size() {
    // Size 0x12345 in bits 23-3, type 2 in bits 2-1, the last flag in bit 0:
    // `0x12345 << 3 | 2 << 1 | 1 = 0x091A2D`.
    assert_eq!(
        block_header(true, BLOCK_COMPRESSED, 0x12345),
        Some([0x2D, 0x1A, 0x09])
    );
    assert_eq!(block_header(false, BLOCK_RAW, 5), Some([0x28, 0x00, 0x00]));
    assert_eq!(block_header(false, BLOCK_RLE, 3), Some([0x1A, 0x00, 0x00]));
    assert_eq!(
        block_header(true, BLOCK_RAW, BLOCK_SIZE_MAX),
        Some([0x01, 0x00, 0x10])
    );
    assert_eq!(block_header(true, BLOCK_RAW, BLOCK_SIZE_MAX + 1), None);
}

#[test]
fn a_block_is_read_with_the_bytes_its_type_stores() {
    // A raw block stores its size in bytes; what follows is the next block.
    let mut bytes = block_header(false, BLOCK_RAW, 5).unwrap().to_vec();
    bytes.extend_from_slice(b"hello");
    bytes.extend_from_slice(b"next");
    let (block, after) = read_block(&bytes).unwrap();
    assert_eq!(
        (block.last, block.kind, block.body),
        (false, BLOCK_RAW, &b"hello"[..])
    );
    assert_eq!(block.stored, &bytes[..BLOCK_HEADER_SIZE + 5]);
    assert_eq!(after, b"next");
    // An RLE block stores one byte whatever its size.
    let mut bytes = block_header(true, BLOCK_RLE, 300).unwrap().to_vec();
    bytes.extend_from_slice(b"zrest");
    let (block, after) = read_block(&bytes).unwrap();
    assert_eq!(
        (block.last, block.kind, block.body),
        (true, BLOCK_RLE, &b"z"[..])
    );
    assert_eq!(after, b"rest");
    // A compressed block stores its size in bytes.
    let mut bytes = block_header(true, BLOCK_COMPRESSED, 2).unwrap().to_vec();
    bytes.extend_from_slice(&[7, 8]);
    let (block, after) = read_block(&bytes).unwrap();
    assert_eq!(
        (block.kind, block.body, after),
        (BLOCK_COMPRESSED, &[7, 8][..], &[][..])
    );

    // Cut short in the header and in the body, and the reserved type.
    for len in 0..bytes.len() {
        assert_eq!(
            read_block(&bytes[..len]),
            Err(BlockReadError::Truncated),
            "{len}"
        );
    }
    let reserved = u8::try_from(3 << BLOCK_TYPE_SHIFT).unwrap();
    assert_eq!(
        read_block(&[reserved, 0, 0]),
        Err(BlockReadError::ReservedType)
    );
    // The size field is 21 bits wide; a block that claims all of them is cut short.
    assert_eq!(
        read_block(&[0xF8, 0xFF, 0xFF]),
        Err(BlockReadError::Truncated)
    );
    assert_eq!(0xFF_FFF8 >> BLOCK_SIZE_SHIFT, (1 << 21) - 1);
}

// ---------------------------------------------------------------------------------------------
// Compressed blocks written by hand, rewritten with a stand-in for the table function
// ---------------------------------------------------------------------------------------------

/// The header of raw or RLE literals of `size` in `format`.
fn plain_literals_header(kind: u8, format: u8, size: usize) -> Vec<u8> {
    let (len, shift) = PLAIN_HEADERS[usize::from(format)];
    // The one-byte header keeps the size from bit 3 up, so its format (0 or 2) is the size's
    // lowest bit; the others have both format bits.
    let format_bits = if len == 1 {
        0
    } else {
        u64::from(format) << LITERALS_FORMAT_SHIFT
    };
    let header = u64::from(kind) | format_bits | u64::try_from(size).unwrap() << shift;
    header.to_le_bytes()[..len].to_vec()
}

/// A sequences section that announces no sequences.
const NO_SEQUENCES: [u8; 1] = [0];

/// A stand-in for the table function: a "description" is a marker byte, a length byte and
/// that many bytes, and it is written back as the marker, the kind's number, and `grow` extra
/// bytes of `0xAA` after its bytes (or with `-grow` bytes dropped).
fn stand_in(
    grow: isize,
    seen: &mut Vec<TableKind>,
) -> impl FnMut(&[u8], TableKind, &mut Vec<u8>) -> Result<usize, &'static str> + '_ {
    move |src, kind, out| {
        let (&[marker, len], rest) = src.split_first_chunk::<2>().ok_or("no description")?;
        if marker != MARKER {
            return Err("not a description");
        }
        let body = rest
            .get(..usize::from(len))
            .ok_or("description cut short")?;
        // Like the real reader: the description must end before the data does.
        if body.len() == rest.len() {
            return Err("description fills its data");
        }
        seen.push(kind);
        out.push(MARKER);
        out.push(kind_number(kind));
        if grow >= 0 {
            out.extend_from_slice(body);
            out.extend(std::iter::repeat_n(0xAA, grow.unsigned_abs()));
        } else {
            out.extend_from_slice(&body[..body.len() - grow.unsigned_abs()]);
        }
        Ok(2 + body.len())
    }
}

const MARKER: u8 = 0xD5;

fn kind_number(kind: TableKind) -> u8 {
    match kind {
        TableKind::HuffmanWeights => 0xF0,
        TableKind::LiteralLengths => 0xF1,
        TableKind::Offsets => 0xF2,
        TableKind::MatchLengths => 0xF3,
    }
}

/// A stand-in description of `len` bytes, each `fill`.
fn described(fill: u8, len: u8) -> Vec<u8> {
    let mut out = vec![MARKER, len];
    out.extend(std::iter::repeat_n(fill, usize::from(len)));
    out
}

fn rewrite(
    body: &[u8],
    grow: isize,
) -> (Result<Vec<u8>, RewriteError<&'static str>>, Vec<TableKind>) {
    let mut seen = Vec::new();
    let mut out = Vec::new();
    let result = rewrite_block(body, &mut stand_in(grow, &mut seen), &mut out).map(|()| out);
    (result, seen)
}

#[test]
fn raw_and_rle_literals_without_sequences_are_copied() {
    for format in 0..4u8 {
        // Sizes on both sides of the one-byte header's odd/even split.
        for size in [0usize, 1, 6, 7, 30, 31] {
            let mut raw = plain_literals_header(LITERALS_RAW, format, size);
            raw.extend(std::iter::repeat_n(0x61, size));
            raw.extend_from_slice(&NO_SEQUENCES);
            assert_eq!(
                rewrite(&raw, 0),
                (Ok(raw.clone()), vec![]),
                "raw, format {format}"
            );
            // One literal byte short: the sequences section is missing.
            assert_eq!(
                rewrite(&raw[..raw.len() - 1], 0).0,
                Err(RewriteError::Layout),
                "raw, format {format}, size {size}"
            );

            let mut rle = plain_literals_header(LITERALS_RLE, format, size);
            rle.push(0x62);
            rle.extend_from_slice(&NO_SEQUENCES);
            assert_eq!(
                rewrite(&rle, 0),
                (Ok(rle.clone()), vec![]),
                "RLE, format {format}"
            );
        }
    }
    // The widest sizes each header holds: 12 bits in two bytes, 20 in three.
    let header = plain_literals_header(LITERALS_RAW, 1, 0xFFF);
    assert_eq!(header, [0xF4, 0xFF]);
    assert_eq!(Literals::parse(&header).unwrap().regenerated, 0xFFF);
    let header = plain_literals_header(LITERALS_RLE, 3, 0xF_FFFF);
    assert_eq!(header, [0xFD, 0xFF, 0xFF]);
    assert_eq!(Literals::parse(&header).unwrap().regenerated, 0xF_FFFF);
    // The one-byte header holds five bits.
    let header = plain_literals_header(LITERALS_RAW, 0, 31);
    assert_eq!(header, [0xF8]);
    assert_eq!(Literals::parse(&header).unwrap().regenerated, 31);
}

/// A block of empty raw literals and a sequences section: the count bytes, the modes byte and
/// what follows it.
fn sequences_block(count: &[u8], modes: u8, after: &[u8]) -> Vec<u8> {
    let mut body = plain_literals_header(LITERALS_RAW, 0, 0);
    body.extend_from_slice(count);
    body.push(modes);
    body.extend_from_slice(after);
    body
}

/// The modes byte for the three tables' modes.
fn modes(literal_lengths: u8, offsets: u8, match_lengths: u8) -> u8 {
    let [(_, ll), (_, of), (_, ml)] = TableKind::SEQUENCE_TABLES;
    literal_lengths << ll | offsets << of | match_lengths << ml
}

#[test]
fn the_sequence_count_is_one_two_or_three_bytes() {
    assert_eq!(sequence_count(&[]), None);
    assert_eq!(sequence_count(&[0]), Some((0, 1)));
    assert_eq!(
        sequence_count(&[SEQUENCE_COUNT_TWO_BYTES - 1]),
        Some((127, 1))
    );
    // Two bytes: the first less 128 is the high byte.
    assert_eq!(sequence_count(&[SEQUENCE_COUNT_TWO_BYTES]), None);
    assert_eq!(
        sequence_count(&[SEQUENCE_COUNT_TWO_BYTES, 0x80]),
        Some((128, 2))
    );
    assert_eq!(
        sequence_count(&[SEQUENCE_COUNT_THREE_BYTES - 1, 0xFF]),
        Some((0x7EFF, 2))
    );
    // Three bytes: a 16-bit count less the bias.
    assert_eq!(sequence_count(&[SEQUENCE_COUNT_THREE_BYTES, 0x00]), None);
    assert_eq!(
        sequence_count(&[SEQUENCE_COUNT_THREE_BYTES, 0x34, 0x12]),
        Some((0x1234 + SEQUENCE_COUNT_LONG_BIAS, 3))
    );
    assert_eq!(MODE_MASK, 0b11);
}

#[test]
fn the_three_sequence_tables_are_recoded_in_their_stored_order() {
    // Three descriptions of different lengths and contents, then the bit stream.
    let mut after = described(0x11, 1);
    after.extend(described(0x22, 2));
    after.extend(described(0x33, 3));
    after.extend_from_slice(b"bits");
    for count in [&[5u8][..], &[0x81, 0x02], &[0xFF, 0x03, 0x04]] {
        let body = sequences_block(count, modes(MODE_FSE, MODE_FSE, MODE_FSE), &after);
        let (out, seen) = rewrite(&body, 1);
        assert_eq!(
            seen,
            [
                TableKind::LiteralLengths,
                TableKind::Offsets,
                TableKind::MatchLengths
            ]
        );
        let mut expected = plain_literals_header(LITERALS_RAW, 0, 0);
        expected.extend_from_slice(count);
        expected.push(modes(MODE_FSE, MODE_FSE, MODE_FSE));
        expected.extend_from_slice(&[MARKER, 0xF1, 0x11, 0xAA]);
        expected.extend_from_slice(&[MARKER, 0xF2, 0x22, 0x22, 0xAA]);
        expected.extend_from_slice(&[MARKER, 0xF3, 0x33, 0x33, 0x33, 0xAA]);
        expected.extend_from_slice(b"bits");
        assert_eq!(out, Ok(expected));
    }
}

#[test]
fn each_mode_stores_what_zstd_says_it_stores() {
    // Predefined and repeat store nothing, RLE one byte, FSE a description.
    let mut after = vec![0x07];
    after.extend(described(0x44, 2));
    after.extend_from_slice(b"bits");
    let body = sequences_block(&[1], modes(MODE_RLE, MODE_PREDEFINED, MODE_FSE), &after);
    let (out, seen) = rewrite(&body, 0);
    assert_eq!(seen, [TableKind::MatchLengths]);
    let mut expected = sequences_block(&[1], modes(MODE_RLE, MODE_PREDEFINED, MODE_FSE), &[0x07]);
    expected.extend_from_slice(&[MARKER, 0xF3, 0x44, 0x44]);
    expected.extend_from_slice(b"bits");
    assert_eq!(out, Ok(expected));

    let mut after = described(0x55, 1);
    after.push(0x09);
    after.extend_from_slice(b"bits");
    let body = sequences_block(&[1], modes(MODE_REPEAT, MODE_FSE, MODE_RLE), &after);
    let (out, seen) = rewrite(&body, 0);
    assert_eq!(seen, [TableKind::Offsets]);
    let mut expected = sequences_block(&[1], modes(MODE_REPEAT, MODE_FSE, MODE_RLE), &[]);
    expected.extend_from_slice(&[MARKER, 0xF2, 0x55, 0x09]);
    expected.extend_from_slice(b"bits");
    assert_eq!(out, Ok(expected));

    // Nothing but modes that store nothing: the block is copied.
    for mode in [MODE_PREDEFINED, MODE_REPEAT] {
        let body = sequences_block(&[1], modes(mode, mode, mode), b"bits");
        assert_eq!(rewrite(&body, 0), (Ok(body.clone()), vec![]));
    }
    // No sequences: the section is the count alone, and is copied even with bytes after it.
    let body = sequences_block(&[], 0, b"");
    assert_eq!(rewrite(&body, 0), (Ok(body.clone()), vec![]));
    let mut body = plain_literals_header(LITERALS_RAW, 0, 0);
    body.extend_from_slice(&[SEQUENCE_COUNT_TWO_BYTES, 0x00, MARKER, 1, 2]);
    assert_eq!(rewrite(&body, 0), (Ok(body.clone()), vec![]));
}

#[test]
fn a_block_whose_sections_run_past_its_end_is_refused() {
    let layout = Err(RewriteError::Layout);
    // No literals header, or a header cut short.
    assert_eq!(rewrite(&[], 0).0, layout);
    assert_eq!(
        rewrite(&plain_literals_header(LITERALS_RAW, 3, 5)[..2], 0).0,
        layout
    );
    // Sequences: no count, a count cut short, no modes byte, no RLE byte.
    let literals = plain_literals_header(LITERALS_RAW, 0, 0);
    assert_eq!(rewrite(&literals, 0).0, layout);
    for count in [&[0x81u8][..], &[0xFF, 0x01]] {
        let mut body = literals.clone();
        body.extend_from_slice(count);
        assert_eq!(rewrite(&body, 0).0, layout);
    }
    let mut body = literals.clone();
    body.push(1);
    assert_eq!(rewrite(&body, 0).0, layout);
    let body = sequences_block(&[1], modes(MODE_PREDEFINED, MODE_PREDEFINED, MODE_RLE), &[]);
    assert_eq!(rewrite(&body, 0).0, layout);
    let body = sequences_block(
        &[1],
        modes(MODE_PREDEFINED, MODE_PREDEFINED, MODE_RLE),
        &[3],
    );
    assert!(rewrite(&body, 0).0.is_ok());
    // A table function that says it read more than it was given.
    let body = sequences_block(&[1], modes(MODE_FSE, 0, 0), &[1, 2]);
    let mut too_many = |src: &[u8], _: TableKind, _: &mut Vec<u8>| Ok::<_, ()>(src.len() + 1);
    assert_eq!(
        rewrite_block(&body, &mut too_many, &mut Vec::new()),
        Err(RewriteError::Layout)
    );
    let mut all = |src: &[u8], _: TableKind, _: &mut Vec<u8>| Ok::<_, ()>(src.len());
    assert_eq!(rewrite_block(&body, &mut all, &mut Vec::new()), Ok(()));
    // The reserved bits of the modes byte, one at a time; they are looked at before the tables.
    for bit in [1u8, 2] {
        assert_eq!(bit & MODES_RESERVED_MASK, bit);
        let body = sequences_block(&[1], modes(MODE_FSE, 0, 0) | bit, &[1, 2]);
        assert_eq!(rewrite(&body, 0).0, Err(RewriteError::ReservedModes));
    }
    // The table function's own refusal comes back with the table's name.
    let body = sequences_block(&[1], modes(MODE_PREDEFINED, MODE_FSE, 0), &[1, 2]);
    assert_eq!(
        rewrite(&body, 0).0,
        Err(RewriteError::Table {
            table: TableKind::Offsets,
            source: "not a description"
        })
    );
}

/// Huffman-coded literals in `format`: the header, a tree description (`tree`), and `streams`
/// bytes of coded literals.
fn huffman_block(kind: u8, format: u8, regenerated: u64, tree: &[u8], streams: usize) -> Vec<u8> {
    let compressed = u64::try_from(tree.len() + streams).unwrap();
    let mut body = huffman_literals_header(kind, format, regenerated, compressed).unwrap();
    body.extend_from_slice(tree);
    body.extend(std::iter::repeat_n(0x5A, streams));
    body.extend_from_slice(&NO_SEQUENCES);
    body
}

/// A tree description whose weights are FSE-compressed: the length byte, a stand-in
/// description of `description` bytes and `stream` bytes of weights.
fn compressed_weights(description: u8, stream: u8) -> Vec<u8> {
    let mut tree = vec![2 + description + stream];
    tree.extend(described(0x77, description));
    tree.extend(std::iter::repeat_n(0x88, usize::from(stream)));
    tree
}

#[test]
fn a_huffman_literals_header_holds_two_sizes_in_10_14_or_18_bits() {
    // Type 2, format 0, sizes 0x155 and 0x2AA in ten bits each:
    // `2 | 0 << 2 | 0x155 << 4 | 0x2AA << 14 = 0xAA9552`.
    assert_eq!(
        huffman_literals_header(LITERALS_COMPRESSED, 0, 0x155, 0x2AA).unwrap(),
        [0x52, 0x95, 0xAA]
    );
    // Type 3, format 1, the same sizes: `3 | 1 << 2 | ... = 0xAA9557`.
    assert_eq!(
        huffman_literals_header(LITERALS_TREELESS, 1, 0x155, 0x2AA).unwrap(),
        [0x57, 0x95, 0xAA]
    );
    // Format 2, sizes 0x1555 and 0x2AAA in fourteen bits:
    // `2 | 2 << 2 | 0x1555 << 4 | 0x2AAA << 18 = 0xAAA9555A`.
    assert_eq!(
        huffman_literals_header(LITERALS_COMPRESSED, 2, 0x1555, 0x2AAA).unwrap(),
        [0x5A, 0x55, 0xA9, 0xAA]
    );
    // Format 3, sizes 0x15555 and 0x2AAAA in eighteen bits:
    // `2 | 3 << 2 | 0x15555 << 4 | 0x2AAAA << 22 = 0xAA_AA95_555E`.
    assert_eq!(
        huffman_literals_header(LITERALS_COMPRESSED, 3, 0x15555, 0x2AAAA).unwrap(),
        [0x5E, 0x55, 0x95, 0xAA, 0xAA]
    );
    for (format, kind, regenerated, compressed) in [
        (0u8, LITERALS_COMPRESSED, 0x155u64, 0x2AAu64),
        (1, LITERALS_TREELESS, 0x155, 0x2AA),
        (2, LITERALS_COMPRESSED, 0x1555, 0x2AAA),
        (3, LITERALS_TREELESS, 0x15555, 0x2AAAA),
    ] {
        let header = huffman_literals_header(kind, format, regenerated, compressed).unwrap();
        let (len, _) = HUFFMAN_HEADERS[usize::from(format)];
        assert_eq!(
            Literals::parse(&header),
            Some(Literals {
                kind,
                format,
                regenerated,
                compressed,
                header_len: len
            })
        );
        // Cut short by a byte, there is no header.
        assert_eq!(Literals::parse(&header[..len - 1]), None);
    }
}

#[test]
fn sizes_that_outgrow_a_four_stream_header_move_to_a_wider_one() {
    let header = |format, regenerated, compressed| {
        huffman_literals_header(LITERALS_COMPRESSED, format, regenerated, compressed)
            .map(|bytes| Literals::parse(&bytes).unwrap().format)
    };
    for (format, width) in [(1u8, 10u32), (2, 14)] {
        let most = (1u64 << width) - 1;
        assert_eq!(header(format, 6, most), Some(format));
        assert_eq!(header(format, 6, most + 1), Some(format + 1));
        assert_eq!(header(format, most, 6), Some(format));
        assert_eq!(header(format, most + 1, 6), Some(format + 1));
    }
    // From ten bits straight to eighteen.
    assert_eq!(header(1, 6, 1 << 14), Some(3));
    // Eighteen bits are the widest.
    assert_eq!(header(3, 6, (1 << 18) - 1), Some(3));
    assert_eq!(header(3, 6, 1 << 18), None);
    assert_eq!(header(1, 1 << 18, 6), None);
    // The single-stream layout has ten bits and no wider one.
    assert_eq!(
        header(HUFFMAN_ONE_STREAM, 6, (1 << 10) - 1),
        Some(HUFFMAN_ONE_STREAM)
    );
    assert_eq!(header(HUFFMAN_ONE_STREAM, 6, 1 << 10), None);
}

#[test]
fn compressed_huffman_weights_are_recoded_and_the_sizes_that_count_them_corrected() {
    for format in 0..4u8 {
        for grow in [-1isize, 0, 3] {
            let tree = compressed_weights(4, 5);
            let body = huffman_block(LITERALS_COMPRESSED, format, 40, &tree, 20);
            let (out, seen) = rewrite(&body, grow);
            assert_eq!(seen, [TableKind::HuffmanWeights], "format {format}");

            // The description is now `4 + grow` bytes after its two leading ones.
            let description = usize::try_from(4 + grow).unwrap();
            let mut new_tree = vec![u8::try_from(2 + description + 5).unwrap(), MARKER, 0xF0];
            new_tree.extend(std::iter::repeat_n(0x77, description.min(4)));
            new_tree.extend(std::iter::repeat_n(0xAA, description.saturating_sub(4)));
            new_tree.extend(std::iter::repeat_n(0x88, 5));
            let expected = huffman_block(LITERALS_COMPRESSED, format, 40, &new_tree, 20);
            assert_eq!(out, Ok(expected), "format {format}, grow {grow}");
        }
    }
}

#[test]
fn directly_stored_weights_and_treeless_literals_are_copied() {
    // From 128 up the first byte counts weights stored as nibbles: no description.
    let mut tree = vec![HUFFMAN_DIRECT_WEIGHTS + 3, MARKER, 2];
    tree.extend_from_slice(&[9, 9]);
    let body = huffman_block(LITERALS_COMPRESSED, 1, 40, &tree, 20);
    assert_eq!(rewrite(&body, 5), (Ok(body.clone()), vec![]));
    // Treeless literals have no tree description, whatever their first byte looks like.
    let body = huffman_block(LITERALS_TREELESS, 1, 40, &compressed_weights(4, 5), 20);
    assert_eq!(rewrite(&body, 5), (Ok(body.clone()), vec![]));
    // Compressed literals with nothing in them are copied too.
    let body = huffman_block(LITERALS_COMPRESSED, 0, 0, &[], 0);
    assert_eq!(rewrite(&body, 5), (Ok(body.clone()), vec![]));
    // The weights' length byte must not run past the literals.
    let mut short = huffman_block(LITERALS_COMPRESSED, 1, 40, &[30, MARKER, 4], 0);
    assert_eq!(rewrite(&short, 0).0, Err(RewriteError::Layout));
    short = huffman_block(LITERALS_COMPRESSED, 1, 40, &[2, MARKER, 4], 0);
    assert_eq!(
        rewrite(&short, 0).0,
        Err(RewriteError::Table {
            table: TableKind::HuffmanWeights,
            source: "description cut short"
        })
    );
    // The literals' compressed size must not run past the block.
    let body = huffman_block(LITERALS_COMPRESSED, 2, 40, &compressed_weights(4, 5), 20);
    assert_eq!(
        rewrite(&body[..body.len() - 2], 0).0,
        Err(RewriteError::Layout)
    );
}

#[test]
fn a_tree_description_that_grows_past_127_bytes_is_refused() {
    let most = HUFFMAN_DIRECT_WEIGHTS - 1;
    // 2 leading bytes, 100 of description, `stream` of weights.
    let at = huffman_block(
        LITERALS_COMPRESSED,
        1,
        40,
        &compressed_weights(100, most - 104),
        20,
    );
    let (out, _) = rewrite(&at, 2);
    let out = out.unwrap();
    let (header_len, _) = HUFFMAN_HEADERS[1];
    assert_eq!(out[header_len], most);
    let over = huffman_block(
        LITERALS_COMPRESSED,
        1,
        40,
        &compressed_weights(100, most - 103),
        20,
    );
    assert_eq!(
        rewrite(&over, 2).0,
        Err(RewriteError::Overflow(Overflow::HuffmanDescription))
    );
    // Without growth the same block is fine.
    assert!(rewrite(&over, 0).0.is_ok());
}

#[test]
fn literals_that_grow_past_their_header_move_or_are_refused() {
    // A compressed size of exactly the ten-bit limit that grows by one byte.
    let limit = (1usize << HUFFMAN_HEADERS[1].1) - 1;
    let tree = compressed_weights(4, 5);
    let streams = limit - tree.len();
    for (format, grow, expected) in [
        (1u8, 0isize, Ok(1u8)),
        (1, 1, Ok(2)),
        (HUFFMAN_ONE_STREAM, 0, Ok(HUFFMAN_ONE_STREAM)),
        (
            HUFFMAN_ONE_STREAM,
            1,
            Err(RewriteError::Overflow(Overflow::LiteralsSize)),
        ),
    ] {
        let body = huffman_block(LITERALS_COMPRESSED, format, 40, &tree, streams);
        let out = rewrite(&body, grow).0.map(|out| {
            let literals = Literals::parse(&out).unwrap();
            assert_eq!(
                literals.compressed,
                u64::try_from(limit + grow.unsigned_abs()).unwrap()
            );
            assert_eq!(literals.regenerated, 40);
            // The section still ends where the sequences begin.
            assert_eq!(out[literals.section_len().unwrap()..], NO_SEQUENCES);
            literals.format
        });
        assert_eq!(out, expected, "format {format}, grow {grow}");
    }
    // The fourteen-bit limit moves to eighteen bits.
    let limit = (1usize << HUFFMAN_HEADERS[2].1) - 1;
    let body = huffman_block(LITERALS_COMPRESSED, 2, 40, &tree, limit - tree.len());
    let out = rewrite(&body, 1).0.unwrap();
    let literals = Literals::parse(&out).unwrap();
    assert_eq!(
        (literals.format, literals.compressed),
        (3, u64::try_from(limit).unwrap() + 1)
    );
}

// ---------------------------------------------------------------------------------------------
// Whole frames written by hand
// ---------------------------------------------------------------------------------------------

const HELLO: &[u8] = b"hello, zbic";

fn block(last: bool, kind: u32, size: usize, body: &[u8]) -> Vec<u8> {
    let mut out = block_header(last, kind, size).unwrap().to_vec();
    out.extend_from_slice(body);
    out
}

fn raw_block(last: bool, data: &[u8]) -> Vec<u8> {
    block(last, BLOCK_RAW, data.len(), data)
}

fn rle_block(last: bool, byte: u8, count: usize) -> Vec<u8> {
    block(last, BLOCK_RLE, count, &[byte])
}

fn compressed_block(last: bool, body: &[u8]) -> Vec<u8> {
    block(last, BLOCK_COMPRESSED, body.len(), body)
}

/// A frame with a window descriptor of 1 KiB and no content size.
fn windowed(blocks: &[u8]) -> Vec<u8> {
    let mut out = header_bytes(MAGIC, 0, Some(0), &[], &[]);
    out.extend_from_slice(blocks);
    out
}

/// A single-segment frame that states `size` in one byte.
fn single_segment(size: u8, blocks: &[u8]) -> Vec<u8> {
    let mut out = header_bytes(MAGIC, SINGLE_SEGMENT_FLAG, None, &[], &[size]);
    out.extend_from_slice(blocks);
    out
}

#[test]
fn each_frame_header_layout_decodes() {
    let blocks = raw_block(true, HELLO);
    let size = u8::try_from(HELLO.len()).unwrap();
    let frame = |descriptor: u8, window: Option<u8>, dictionary: &[u8], fcs: &[u8]| {
        let mut out = header_bytes(MAGIC, descriptor, window, dictionary, fcs);
        out.extend_from_slice(&blocks);
        out
    };
    let cases = [
        // A window descriptor, no content size.
        frame(0, Some(0), &[], &[]),
        // A single segment and a one-byte content size.
        frame(SINGLE_SEGMENT_FLAG, None, &[], &[size]),
        // Four- and eight-byte content sizes, with a window and as a single segment.
        frame(2 << FCS_FLAG_SHIFT, Some(0), &[], &[size, 0, 0, 0]),
        frame(
            2 << FCS_FLAG_SHIFT | SINGLE_SEGMENT_FLAG,
            None,
            &[],
            &[size, 0, 0, 0],
        ),
        frame(
            3 << FCS_FLAG_SHIFT,
            Some(0),
            &[],
            &[size, 0, 0, 0, 0, 0, 0, 0],
        ),
        // A dictionary id of zero in each of its widths is no dictionary.
        frame(1, Some(0), &[0], &[]),
        frame(2, Some(0), &[0, 0], &[]),
        frame(3 | SINGLE_SEGMENT_FLAG, None, &[0, 0, 0, 0], &[size]),
        // The bit zstd does not use.
        frame(0b0001_0000, Some(0), &[], &[]),
    ];
    for (index, case) in cases.iter().enumerate() {
        assert_eq!(
            decompress(case, HELLO.len()).as_deref(),
            Ok(HELLO),
            "case {index}"
        );
    }
    // A two-byte content size is the size less 256: 300 is stored as 44.
    let long = sample(Kind::Text, 300, 7);
    let mut frame = header_bytes(MAGIC, 1 << FCS_FLAG_SHIFT, Some(0), &[], &[44, 0]);
    frame.extend(raw_block(true, &long));
    assert_eq!(decompress(&frame, 300), Ok(long));
}

#[test]
fn raw_and_rle_blocks_decode() {
    let mut blocks = raw_block(false, b"abc");
    blocks.extend(rle_block(false, b'z', 5));
    blocks.extend(raw_block(false, &[]));
    blocks.extend(rle_block(false, b'!', 0));
    blocks.extend(raw_block(true, b"def"));
    assert_eq!(
        decompress(&windowed(&blocks), 11).as_deref(),
        Ok(&b"abczzzzzdef"[..])
    );
    assert_eq!(
        decompress(&single_segment(11, &blocks), 11).as_deref(),
        Ok(&b"abczzzzzdef"[..])
    );
}

#[test]
fn an_empty_frame_decodes_to_nothing() {
    // Looser than the reference, which returns -1 for an empty destination.
    assert_eq!(decompress(&windowed(&raw_block(true, &[])), 0), Ok(vec![]));
    assert_eq!(
        decompress(&single_segment(0, &raw_block(true, &[])), 0),
        Ok(vec![])
    );
    assert_eq!(
        decompress(&single_segment(0, &rle_block(true, 7, 0)), 0),
        Ok(vec![])
    );
}

#[test]
fn hand_written_compressed_blocks_decode() {
    // Raw literals in each of their headers, and no sequences.
    for format in 0..4u8 {
        let mut body = plain_literals_header(LITERALS_RAW, format, HELLO.len());
        body.extend_from_slice(HELLO);
        body.extend_from_slice(&NO_SEQUENCES);
        let frame = windowed(&compressed_block(true, &body));
        assert_eq!(
            decompress(&frame, HELLO.len()).as_deref(),
            Ok(HELLO),
            "raw, format {format}"
        );

        // RLE literals: one byte, repeated.
        let mut body = plain_literals_header(LITERALS_RLE, format, 9);
        body.push(b'r');
        body.extend_from_slice(&NO_SEQUENCES);
        let frame = windowed(&compressed_block(true, &body));
        assert_eq!(
            decompress(&frame, 9).as_deref(),
            Ok(&b"rrrrrrrrr"[..]),
            "RLE, format {format}"
        );
    }

    // One sequence whose three tables are in RLE mode, so no state bits are stored: four
    // literals (literal-length code 4), a match of three bytes (match-length code 0) at offset
    // code 2, whose two extra bits are 11: offset value 4 + 3 = 7, which is offset 7 - 3 = 4.
    // The bit stream is those two bits under the padding bit: 0b111.
    let mut body = plain_literals_header(LITERALS_RAW, 0, 4);
    body.extend_from_slice(b"abcd");
    body.extend_from_slice(&[1, modes(MODE_RLE, MODE_RLE, MODE_RLE), 4, 2, 0, 0b111]);
    let frame = windowed(&compressed_block(true, &body));
    assert_eq!(decompress(&frame, 7).as_deref(), Ok(&b"abcdabc"[..]));
    // The same block after a raw block, and before one.
    let mut blocks = raw_block(false, b"12");
    blocks.extend(compressed_block(false, &body));
    blocks.extend(raw_block(true, b"34"));
    assert_eq!(
        decompress(&windowed(&blocks), 11).as_deref(),
        Ok(&b"12abcdabc34"[..])
    );
}

#[test]
fn a_content_checksum_is_read_but_not_checked() {
    // Looser than the reference, which compares the checksum with the content.
    let mut frame = header_bytes(MAGIC, CHECKSUM_FLAG, Some(0), &[], &[]);
    frame.extend(raw_block(true, HELLO));
    frame.extend_from_slice(&[0xDE, 0xAD, 0xBE, 0xEF]);
    assert_eq!(decompress(&frame, HELLO.len()).as_deref(), Ok(HELLO));
    // The four bytes must be there, and nothing after them.
    for cut in 1..=CHECKSUM_SIZE {
        assert_eq!(
            decompress(&frame[..frame.len() - cut], HELLO.len()),
            Err(ZbicError::Truncated)
        );
    }
    frame.push(0);
    assert_eq!(
        decompress(&frame, HELLO.len()),
        Err(ZbicError::TrailingData { extra: 1 })
    );
    // The checksum follows the last block only.
    let mut frame = header_bytes(MAGIC, CHECKSUM_FLAG, Some(0), &[], &[]);
    frame.extend(raw_block(false, b"hello"));
    frame.extend(raw_block(true, b", zbic"));
    frame.extend_from_slice(&[1, 2, 3, 4]);
    assert_eq!(decompress(&frame, HELLO.len()).as_deref(), Ok(HELLO));
}

// ---------------------------------------------------------------------------------------------
// Refusals
// ---------------------------------------------------------------------------------------------

#[test]
fn a_wrong_magic_is_refused() {
    let frame = windowed(&raw_block(true, HELLO));
    assert!(is_zbic(&frame));
    let mut wrong = frame.clone();
    wrong[3] ^= 1;
    assert!(!is_zbic(&wrong));
    assert_eq!(decompress(&wrong, HELLO.len()), Err(ZbicError::BadMagic));
    // A zstd frame is not a ZBIC frame.
    let mut zstd = frame.clone();
    zstd[..MAGIC_SIZE].copy_from_slice(&ZSTD_MAGIC);
    assert_eq!(decompress(&zstd, HELLO.len()), Err(ZbicError::BadMagic));
    // Fewer than four bytes, even of the magic.
    for len in 0..MAGIC_SIZE {
        assert!(!is_zbic(&MAGIC[..len]));
        assert_eq!(decompress(&MAGIC[..len], 0), Err(ZbicError::BadMagic));
    }
    assert!(is_zbic(&MAGIC));
    assert_eq!(
        ZbicError::BadMagic.to_string(),
        "not a ZBIC frame: it does not start with the bytes \"ZBIC\""
    );
}

#[test]
fn a_frame_cut_short_is_refused_at_every_length() {
    let mut blocks = raw_block(false, b"hello");
    blocks.extend(rle_block(false, b'-', 3));
    let mut body = plain_literals_header(LITERALS_RAW, 0, 3);
    body.extend_from_slice(b"end");
    body.extend_from_slice(&NO_SEQUENCES);
    blocks.extend(compressed_block(true, &body));
    let mut frame = header_bytes(
        MAGIC,
        CHECKSUM_FLAG | 2 << FCS_FLAG_SHIFT,
        Some(0),
        &[],
        &[11, 0, 0, 0],
    );
    frame.extend_from_slice(&blocks);
    frame.extend_from_slice(&[0; CHECKSUM_SIZE]);
    assert_eq!(decompress(&frame, 11).as_deref(), Ok(&b"hello---end"[..]));
    for len in MAGIC_SIZE..frame.len() {
        assert_eq!(
            decompress(&frame[..len], 11),
            Err(ZbicError::Truncated),
            "{len} bytes"
        );
    }
    // A frame that ends after a block that is not its last.
    assert_eq!(
        decompress(&windowed(&raw_block(false, HELLO)), HELLO.len()),
        Err(ZbicError::Truncated)
    );
    assert_eq!(
        ZbicError::Truncated.to_string(),
        "the ZBIC frame is cut short"
    );
}

#[test]
fn the_reserved_header_bit_and_the_reserved_block_type_are_refused() {
    let mut frame = windowed(&raw_block(true, HELLO));
    frame[DESCRIPTOR_OFFSET] |= RESERVED_FLAG;
    assert_eq!(decompress(&frame, HELLO.len()), Err(ZbicError::ReservedBit));

    let mut blocks = raw_block(false, HELLO);
    blocks.extend_from_slice(&[u8::try_from(3 << BLOCK_TYPE_SHIFT | 1).unwrap(), 0, 0]);
    let error = decompress(&windowed(&blocks), HELLO.len()).unwrap_err();
    assert_eq!(error, ZbicError::ReservedBlockType { block: 1 });
    assert_eq!(error.to_string(), "ZBIC block 1 has the reserved type");
}

#[test]
fn a_dictionary_id_is_refused_unless_zero() {
    // As in the reference, which has no dictionary either (`dictionary_wrong`).
    for (flag, id, expected) in [
        (1u8, &[0x2A][..], 0x2Au32),
        (2, &[0x01, 0x80], 0x8001),
        (3, &[1, 0, 0, 0xFF], 0xFF00_0001),
    ] {
        let mut frame = header_bytes(MAGIC, flag, Some(0), id, &[]);
        frame.extend(raw_block(true, HELLO));
        let error = decompress(&frame, HELLO.len()).unwrap_err();
        assert_eq!(error, ZbicError::Dictionary { id: expected });
    }
    assert_eq!(
        ZbicError::Dictionary { id: 0x2A }.to_string(),
        "the ZBIC frame needs dictionary 0x2a, and an NSO has none"
    );
    // Zero is no dictionary (`each_frame_header_layout_decodes` has every width).
    let mut frame = header_bytes(MAGIC, 1, Some(0), &[0], &[]);
    frame.extend(raw_block(true, HELLO));
    assert!(decompress(&frame, HELLO.len()).is_ok());
}

/// The window descriptor of `MAX_WINDOW_SIZE`, which is a power of two.
fn limit_window() -> u8 {
    let exponent = MAX_WINDOW_SIZE.ilog2() - WINDOW_LOG_MIN;
    assert_eq!(1u64 << (exponent + WINDOW_LOG_MIN), MAX_WINDOW_SIZE);
    u8::try_from(exponent << WINDOW_MANTISSA_BITS).unwrap()
}

#[test]
fn a_window_over_the_limit_is_refused_and_one_at_it_is_not() {
    // Stricter than the reference, which accepts window logs up to 31.
    let frame_with_window = |window: u8| {
        let mut frame = header_bytes(MAGIC, 0, Some(window), &[], &[]);
        frame.extend(raw_block(true, HELLO));
        frame
    };
    assert_eq!(
        decompress(&frame_with_window(limit_window()), HELLO.len()).as_deref(),
        Ok(HELLO)
    );
    // The next window a descriptor can say: one eighth more.
    let error = decompress(&frame_with_window(limit_window() + 1), HELLO.len()).unwrap_err();
    assert_eq!(
        error,
        ZbicError::WindowTooLarge {
            window: MAX_WINDOW_SIZE + MAX_WINDOW_SIZE / 8,
            limit: MAX_WINDOW_SIZE
        }
    );
    assert_eq!(
        error.to_string(),
        "the ZBIC frame's window of 0x9000000 bytes is over the limit of 0x8000000"
    );

    // A declared size over the limit is the limit instead: the same window is then let
    // through, as is a single segment of that size, whose window is its content size. Neither
    // frame is decoded here: both are cut short after the header.
    let size = usize::try_from(MAX_WINDOW_SIZE + MAX_WINDOW_SIZE / 8).unwrap();
    let header = header_bytes(MAGIC, 0, Some(limit_window() + 1), &[], &[]);
    assert_eq!(decompress(&header, size), Err(ZbicError::Truncated));
    assert_eq!(
        decompress(&header, size - 1),
        Err(ZbicError::WindowTooLarge {
            window: MAX_WINDOW_SIZE + MAX_WINDOW_SIZE / 8,
            limit: MAX_WINDOW_SIZE + MAX_WINDOW_SIZE / 8 - 1
        })
    );
    let content = u32::try_from(size).unwrap().to_le_bytes();
    let header = header_bytes(
        MAGIC,
        2 << FCS_FLAG_SHIFT | SINGLE_SEGMENT_FLAG,
        None,
        &[],
        &content,
    );
    assert_eq!(decompress(&header, size), Err(ZbicError::Truncated));
}

#[test]
fn a_content_size_must_be_the_declared_size() {
    // Stricter than the reference, which needs the content size to be what the frame decodes
    // to and no more than the declared size.
    let size = u8::try_from(HELLO.len()).unwrap();
    let frame = single_segment(size, &raw_block(true, HELLO));
    assert_eq!(decompress(&frame, HELLO.len()).as_deref(), Ok(HELLO));
    for declared in [HELLO.len() - 1, HELLO.len() + 1] {
        let error = decompress(&frame, declared).unwrap_err();
        assert_eq!(
            error,
            ZbicError::ContentSizeMismatch {
                frame: u64::from(size),
                expected: declared
            }
        );
    }
    assert_eq!(
        ZbicError::ContentSizeMismatch {
            frame: 11,
            expected: 12
        }
        .to_string(),
        "the ZBIC frame says it holds 0xb bytes, but 0xc are declared"
    );
    // A frame whose header states the declared size and whose blocks hold another.
    let lying = single_segment(size, &raw_block(true, &HELLO[..HELLO.len() - 1]));
    assert_eq!(
        decompress(&lying, HELLO.len()),
        Err(ZbicError::TooShort {
            expected: HELLO.len(),
            actual: HELLO.len() - 1
        })
    );
}

#[test]
fn output_longer_than_declared_is_refused() {
    // Known when the frame ends.
    let frame = windowed(&raw_block(true, HELLO));
    let error = decompress(&frame, HELLO.len() - 1).unwrap_err();
    assert_eq!(
        error,
        ZbicError::TooLong {
            expected: HELLO.len() - 1
        }
    );
    assert_eq!(
        error.to_string(),
        "the ZBIC frame holds more than the 0xa bytes declared"
    );
    assert_eq!(
        decompress(&frame, 0),
        Err(ZbicError::TooLong { expected: 0 })
    );

    // Known before the frame ends: the window is 1 KiB, so after two blocks of 1 KiB the
    // decoder offers the first, and the output so far is 2 KiB. Nothing after that block is
    // read: the frame is cut short there.
    let kib = 1usize << WINDOW_LOG_MIN;
    let mut blocks = rle_block(false, 1, kib);
    blocks.extend(rle_block(false, 2, kib));
    let cut = windowed(&blocks);
    assert_eq!(
        decompress(&cut, 2 * kib - 1),
        Err(ZbicError::TooLong {
            expected: 2 * kib - 1
        })
    );
    // With a declared size that the output has not passed yet, the cut is what is wrong.
    assert_eq!(decompress(&cut, 2 * kib), Err(ZbicError::Truncated));
}

#[test]
fn output_is_collected_while_a_long_frame_decodes() {
    // Six blocks of 1 KiB through a window of 1 KiB: all but the last window is collected
    // before the frame ends.
    let kib = 1usize << WINDOW_LOG_MIN;
    let mut blocks = Vec::new();
    let mut expected = Vec::new();
    for index in 0..6u8 {
        blocks.extend(rle_block(index == 5, index, kib));
        expected.extend(std::iter::repeat_n(index, kib));
    }
    assert_eq!(decompress(&windowed(&blocks), expected.len()), Ok(expected));
}

#[test]
fn output_shorter_than_declared_is_refused() {
    // Stricter than the reference, which accepts any length above zero.
    let frame = windowed(&raw_block(true, HELLO));
    let error = decompress(&frame, HELLO.len() + 1).unwrap_err();
    assert_eq!(
        error,
        ZbicError::TooShort {
            expected: HELLO.len() + 1,
            actual: HELLO.len()
        }
    );
    assert_eq!(
        error.to_string(),
        "the ZBIC frame holds 0xb bytes, fewer than the 0xc declared"
    );
    // The length is looked at before what follows the frame.
    let mut trailing = frame.clone();
    trailing.push(0);
    assert_eq!(
        decompress(&trailing, HELLO.len() + 1),
        Err(ZbicError::TooShort {
            expected: HELLO.len() + 1,
            actual: HELLO.len()
        })
    );
}

#[test]
fn a_second_frame_or_a_skippable_frame_is_refused() {
    // Stricter than the reference, which decodes frames one after another and skips skippable
    // frames.
    let frame = windowed(&raw_block(true, HELLO));
    let mut twice = frame.clone();
    twice.extend_from_slice(&frame);
    let error = decompress(&twice, 2 * HELLO.len()).unwrap_err();
    // The first frame alone is shorter than what was declared for both.
    assert_eq!(
        error,
        ZbicError::TooShort {
            expected: 2 * HELLO.len(),
            actual: HELLO.len()
        }
    );
    let error = decompress(&twice, HELLO.len()).unwrap_err();
    assert_eq!(error, ZbicError::TrailingData { extra: frame.len() });
    assert_eq!(
        error.to_string(),
        format!("{} bytes follow the ZBIC frame", frame.len())
    );

    // A skippable frame (magic 0x184D2A50, a four-byte length, that many bytes) before or
    // after the frame.
    let skippable = [0x50, 0x2A, 0x4D, 0x18, 2, 0, 0, 0, 9, 9];
    let mut before = skippable.to_vec();
    before.extend_from_slice(&frame);
    assert_eq!(decompress(&before, HELLO.len()), Err(ZbicError::BadMagic));
    let mut after = frame.clone();
    after.extend_from_slice(&skippable);
    assert_eq!(
        decompress(&after, HELLO.len()),
        Err(ZbicError::TrailingData {
            extra: skippable.len()
        })
    );
}

/// A frame of one compressed block: empty raw literals, one sequence, a literal-length table
/// described by `description`, and one byte of bit stream.
fn frame_with_table(description: &[u8]) -> Vec<u8> {
    let mut after = description.to_vec();
    after.push(0x01);
    let body = sequences_block(
        &[1],
        modes(MODE_FSE, MODE_PREDEFINED, MODE_PREDEFINED),
        &after,
    );
    windowed(&compressed_block(true, &body))
}

#[test]
fn a_table_description_that_is_not_valid_is_refused_with_its_block_and_table() {
    let table_error = |description: &[u8]| {
        match decompress(&frame_with_table(description), 1) {
            Err(ZbicError::Table {
                block: 0,
                table: TableKind::LiteralLengths,
                source,
            }) => Ok(source),
            other => Err(format!("not a table error: {other:?}")),
        }
        .unwrap()
    };
    // [`PLAIN`] with a total of 31 (`counts_that_do_not_fill_the_table_are_refused`).
    assert_eq!(
        table_error(&[0x03, 0xC0, 0x40, 0x04]),
        TableError::CountSum { sum: 31, size: 32 }
    );
    let limit = TableKind::LiteralLengths.max_symbol();
    assert_eq!(
        table_error(&one_symbol_up_to(u64::from(limit) + 1)),
        TableError::SymbolOutOfRange {
            max: u64::from(limit) + 1,
            limit
        }
    );
    let limit = TableKind::LiteralLengths.max_accuracy_log();
    assert_eq!(
        table_error(&one_symbol_at_log(limit + 1)),
        TableError::AccuracyLog {
            log: limit + 1,
            limit
        }
    );
    let mut unread = vec![PLAIN_ON_A_LEFT_OVER_PART[0] + 1, 0x99];
    unread.extend_from_slice(&PLAIN_ON_A_LEFT_OVER_PART[1..]);
    assert_eq!(table_error(&unread), TableError::UnreadBytes { unread: 1 });
    // A description that runs to the end of the block: the bit stream byte is its last.
    let mut body = sequences_block(&[1], modes(MODE_FSE, 0, 0), &PLAIN);
    body.truncate(body.len() - 1);
    body.push(PLAIN[3]);
    assert_eq!(
        decompress(&windowed(&compressed_block(true, &body)), 1),
        Err(ZbicError::Table {
            block: 0,
            table: TableKind::LiteralLengths,
            source: TableError::Truncated
        })
    );

    // The message names the block and the table, and the block is counted from the frame's
    // first.
    let mut blocks = raw_block(false, HELLO);
    let body = sequences_block(&[1], modes(0, 0, MODE_FSE), &[0x03, 0xC0, 0x40, 0x04, 0x01]);
    blocks.extend(compressed_block(true, &body));
    let error = decompress(&windowed(&blocks), HELLO.len()).unwrap_err();
    assert_eq!(
        error.to_string(),
        "ZBIC block 1: the match length table is not valid: its counts add up to 31, not to \
         the table size 32"
    );
    let messages = [
        (
            TableError::Truncated,
            "it does not end before the data that holds it",
        ),
        (
            TableError::SymbolOutOfRange { max: 36, limit: 35 },
            "its largest symbol is 36, over the limit of 35",
        ),
        (
            TableError::AccuracyLog { log: 10, limit: 9 },
            "its accuracy log is 10, over the limit of 9",
        ),
        (
            TableError::UnreadBytes { unread: 1 },
            "1 of its bytes are left over",
        ),
    ];
    for (error, message) in messages {
        assert_eq!(error.to_string(), message);
    }
    for (kind, name) in [
        (TableKind::HuffmanWeights, "Huffman weight"),
        (TableKind::LiteralLengths, "literal length"),
        (TableKind::Offsets, "offset"),
    ] {
        assert_eq!(kind.to_string(), name);
    }
}

#[test]
fn a_block_whose_contents_run_past_its_end_is_refused() {
    // Raw literals that claim more bytes than the block has.
    let body = plain_literals_header(LITERALS_RAW, 0, 9);
    let mut blocks = raw_block(false, HELLO);
    blocks.extend(compressed_block(true, &body));
    let error = decompress(&windowed(&blocks), HELLO.len()).unwrap_err();
    assert_eq!(error, ZbicError::BlockLayout { block: 1 });
    assert_eq!(
        error.to_string(),
        "ZBIC block 1 is malformed: its contents run past its end"
    );
    // An empty compressed block has no literals header.
    assert_eq!(
        decompress(&windowed(&compressed_block(true, &[])), 0),
        Err(ZbicError::BlockLayout { block: 0 })
    );
}

#[test]
fn a_reserved_bit_in_the_sequence_table_modes_is_refused() {
    // The block of `hand_written_compressed_blocks_decode`, with each reserved bit set.
    let block_with = |modes_byte: u8| {
        let mut body = plain_literals_header(LITERALS_RAW, 0, 4);
        body.extend_from_slice(b"abcd");
        body.extend_from_slice(&[1, modes_byte, 4, 2, 0, 0b111]);
        let mut blocks = raw_block(false, HELLO);
        blocks.extend(compressed_block(true, &body));
        decompress(&windowed(&blocks), HELLO.len() + 7)
    };
    let all_rle = modes(MODE_RLE, MODE_RLE, MODE_RLE);
    assert!(block_with(all_rle).is_ok());
    for bit in [1u8, 2] {
        let error = block_with(all_rle | bit).unwrap_err();
        assert_eq!(error, ZbicError::ReservedModes { block: 1 });
        assert_eq!(
            error.to_string(),
            "ZBIC block 1 sets a reserved bit of its sequence table modes"
        );
    }
}

#[test]
fn a_block_over_the_window_or_over_128_kib_is_refused() {
    // Stricter than the reference, which decodes raw and RLE blocks of any size.
    let kib = 1usize << WINDOW_LOG_MIN;
    for (at, over) in [
        (
            raw_block(true, &vec![7; kib]),
            raw_block(true, &vec![7; kib + 1]),
        ),
        (rle_block(true, 7, kib), rle_block(true, 7, kib + 1)),
    ] {
        assert_eq!(decompress(&windowed(&at), kib), Ok(vec![7; kib]));
        assert!(matches!(
            decompress(&windowed(&over), kib + 1),
            Err(ZbicError::Zstd { block: 0, .. })
        ));
    }
    // A window of 1 MiB: the limit is then the largest block.
    let frame = |blocks: &[u8]| {
        let mut out = header_bytes(MAGIC, 0, Some(10 << WINDOW_MANTISSA_BITS), &[], &[]);
        out.extend_from_slice(blocks);
        out
    };
    let most = vec![7; BLOCK_SIZE_MAX];
    assert_eq!(
        decompress(&frame(&raw_block(true, &most)), BLOCK_SIZE_MAX),
        Ok(most.clone())
    );
    assert_eq!(
        decompress(&frame(&rle_block(true, 7, BLOCK_SIZE_MAX)), BLOCK_SIZE_MAX),
        Ok(most)
    );
    // One byte more does not fit `block_header`, so the header is written out: the size in
    // bits 23-3, the type in bits 2-1, the last flag in bit 0.
    let size = u32::try_from(BLOCK_SIZE_MAX + 1).unwrap();
    let mut raw =
        (size << BLOCK_SIZE_SHIFT | BLOCK_RAW << BLOCK_TYPE_SHIFT | 1).to_le_bytes()[..3].to_vec();
    raw.extend(std::iter::repeat_n(7, BLOCK_SIZE_MAX + 1));
    assert!(matches!(
        decompress(&frame(&raw), BLOCK_SIZE_MAX + 1),
        Err(ZbicError::Zstd { block: 0, .. })
    ));
    let mut rle =
        (size << BLOCK_SIZE_SHIFT | BLOCK_RLE << BLOCK_TYPE_SHIFT | 1).to_le_bytes()[..3].to_vec();
    rle.push(7);
    assert!(matches!(
        decompress(&frame(&rle), BLOCK_SIZE_MAX + 1),
        Err(ZbicError::Zstd { block: 0, .. })
    ));
}

#[test]
fn what_the_zstd_decoder_refuses_comes_back_with_its_block() {
    // A sequence whose bit stream is a zero byte: there is no padding bit to start from.
    let mut body = plain_literals_header(LITERALS_RAW, 0, 4);
    body.extend_from_slice(b"abcd");
    body.extend_from_slice(&[1, modes(MODE_RLE, MODE_RLE, MODE_RLE), 4, 2, 0, 0]);
    let mut blocks = raw_block(false, HELLO);
    blocks.extend(compressed_block(true, &body));
    let error = decompress(&windowed(&blocks), HELLO.len() + 7).unwrap_err();
    let message = match &error {
        ZbicError::Zstd { block: 1, message } => Ok(message),
        other => Err(format!("not a decoder error: {other:?}")),
    }
    .unwrap();
    assert_ne!(message, "");
    assert_eq!(
        error.to_string(),
        format!("ZBIC block 1 does not decode: {message}")
    );
}

/// A table that is far shorter as ZBIC than as zstd: `symbols` symbols in a table of
/// `1 << log`, all but the first with a count of -1. ZBIC stores those as equal running
/// totals, which take no digits; zstd stores each in several bits.
fn mostly_less_than_one(log: u32, symbols: usize) -> (Vec<u8>, Vec<u8>) {
    let mut values = vec![-1; symbols];
    values[0] = (1 << log) - i32::try_from(symbols - 1).unwrap();
    let table = counts(log, &values);
    (write_bic(&table, true).unwrap(), zstd_bytes(&table))
}

#[test]
fn a_rewritten_block_over_a_zstd_limit_is_refused() {
    // Stricter than the reference, which reads the tables where they are.

    // A Huffman tree description of at most 127 bytes: weights with 32 symbols at log 6.
    let (bic, zstd) = mostly_less_than_one(TableKind::HuffmanWeights.max_accuracy_log(), 32);
    let growth = zstd.len() - bic.len();
    assert!(growth > 0);
    let most = usize::from(HUFFMAN_DIRECT_WEIGHTS - 1);
    let literals = |weights_len: usize| {
        let mut tree = vec![u8::try_from(weights_len).unwrap()];
        tree.extend_from_slice(&bic);
        tree.resize(1 + weights_len, 0x01);
        let frame = windowed(&compressed_block(
            true,
            &huffman_block(LITERALS_COMPRESSED, 1, 40, &tree, 20),
        ));
        decompress(&frame, 40)
    };
    assert_eq!(
        literals(most - growth + 1),
        Err(ZbicError::Overflow {
            block: 0,
            limit: Overflow::HuffmanDescription
        })
    );
    // At the limit the block is rewritten, and it is the decoder that refuses the made-up
    // weights.
    assert!(matches!(
        literals(most - growth),
        Err(ZbicError::Zstd { block: 0, .. })
    ));

    // Single-stream literals of at most 1023 bytes.
    let limit = (1usize << HUFFMAN_HEADERS[usize::from(HUFFMAN_ONE_STREAM)].1) - 1;
    let literals = |compressed: usize| {
        let mut tree = vec![u8::try_from(bic.len() + 1).unwrap()];
        tree.extend_from_slice(&bic);
        tree.push(0x01);
        let streams = compressed - tree.len();
        let frame = windowed(&compressed_block(
            true,
            &huffman_block(LITERALS_COMPRESSED, HUFFMAN_ONE_STREAM, 40, &tree, streams),
        ));
        decompress(&frame, 40)
    };
    assert_eq!(
        literals(limit - growth + 1),
        Err(ZbicError::Overflow {
            block: 0,
            limit: Overflow::LiteralsSize
        })
    );
    assert!(matches!(
        literals(limit - growth),
        Err(ZbicError::Zstd { block: 0, .. })
    ));

    // A block of at most 128 KiB: raw literals, then a literal-length table.
    let (bic, zstd) = mostly_less_than_one(
        TableKind::LiteralLengths.max_accuracy_log(),
        usize::try_from(TableKind::LiteralLengths.max_symbol()).unwrap() + 1,
    );
    let growth = zstd.len() - bic.len();
    assert!(growth > 0);
    let block_of = |len: usize| {
        let mut tail = vec![1, modes(MODE_FSE, MODE_PREDEFINED, MODE_PREDEFINED)];
        tail.extend_from_slice(&bic);
        tail.push(0x01);
        let header_len = PLAIN_HEADERS[3].0;
        let literals = len - header_len - tail.len();
        let mut body = plain_literals_header(LITERALS_RAW, 3, literals);
        body.resize(header_len + literals, 0x61);
        body.extend_from_slice(&tail);
        assert_eq!(body.len(), len);
        decompress(&windowed(&compressed_block(true, &body)), literals)
    };
    let error = block_of(BLOCK_SIZE_MAX - growth + 1).unwrap_err();
    assert_eq!(
        error,
        ZbicError::Overflow {
            block: 0,
            limit: Overflow::BlockSize
        }
    );
    assert_eq!(
        error.to_string(),
        "ZBIC block 0 cannot be rewritten as zstd: it would be over 128 KiB"
    );
    assert!(matches!(
        block_of(BLOCK_SIZE_MAX - growth),
        Err(ZbicError::Zstd { block: 0, .. })
    ));
    assert_eq!(
        Overflow::HuffmanDescription.to_string(),
        "its Huffman tree description would be over 127 bytes"
    );
    assert_eq!(
        Overflow::LiteralsSize.to_string(),
        "the compressed size of its literals would not fit their header"
    );
}

// ---------------------------------------------------------------------------------------------
// Frames made by the reference's encoder
// ---------------------------------------------------------------------------------------------

/// What a frame's blocks are made of, read with the shipping parsers.
#[derive(Debug, Default, PartialEq, Eq)]
struct Probe {
    /// Each block's type.
    blocks: Vec<u32>,
    /// Each compressed block's literals type and size format.
    literals: BTreeSet<(u8, u8)>,
    /// For each Huffman tree, whether it stores its weights directly (`true`) or compresses
    /// them (`false`).
    direct_weights: BTreeSet<bool>,
    /// Every mode each sequence table is in, as `(table, mode)`.
    modes: BTreeSet<(u8, u8)>,
    /// The widths of the sequence counts.
    count_widths: BTreeSet<usize>,
    /// For each table description, whether it sets the low-counts flag.
    low_counts: BTreeSet<bool>,
}

fn probe(frame: &[u8]) -> Probe {
    let header = parse_header(frame, MAGIC).unwrap();
    let mut rest = &frame[header.len..];
    let mut probe = Probe::default();
    loop {
        let (block, after) = read_block(rest).unwrap();
        rest = after;
        probe.blocks.push(block.kind);
        if block.kind == BLOCK_COMPRESSED {
            let literals = Literals::parse(block.body).unwrap();
            probe.literals.insert((literals.kind, literals.format));
            if literals.kind == LITERALS_COMPRESSED {
                let direct = block.body[literals.header_len] >= HUFFMAN_DIRECT_WEIGHTS;
                probe.direct_weights.insert(direct);
            }
            let sequences = &block.body[literals.section_len().unwrap()..];
            let (count, width) = sequence_count(sequences).unwrap();
            if count > 0 {
                probe.count_widths.insert(width);
                for (table, (_, shift)) in (0u8..).zip(TableKind::SEQUENCE_TABLES) {
                    probe
                        .modes
                        .insert((table, (sequences[width] >> shift) & MODE_MASK));
                }
            }
            let mut flags = |src: &[u8], kind: TableKind, out: &mut Vec<u8>| {
                probe.low_counts.insert(src[0] & LOW_COUNTS_FLAG != 0);
                bic_to_zstd(src, kind, out)
            };
            rewrite_block(block.body, &mut flags, &mut Vec::new()).unwrap();
        }
        if block.last {
            break;
        }
    }
    probe
}

/// A frame from `crates/loader/tests/zbic` and the synthetic data it holds.
struct Fixture {
    name: &'static str,
    frame: &'static [u8],
    kind: Kind,
    len: usize,
    seed: u64,
}

macro_rules! fixture {
    ($name:literal, $kind:expr, $len:expr, $seed:expr) => {
        Fixture {
            name: $name,
            frame: include_bytes!(concat!("../../tests/zbic/", $name, ".zbic")),
            kind: $kind,
            len: $len,
            seed: $seed,
        }
    };
}

const FIXTURES: [Fixture; 9] = [
    fixture!("text_600", Kind::Text, 600, 1),
    fixture!("skew_200", Kind::Skewed, 200, 5),
    fixture!("skew_900", Kind::Skewed, 900, 5),
    fixture!("skew_2000", Kind::Skewed, 2000, 6),
    fixture!("runs_20k", Kind::Runs, 20000, 11),
    fixture!("zeros_300k", Kind::Zeros, 300_000, 10),
    fixture!("text_8k_tb", Kind::Text, 8000, 14),
    fixture!("nib_300", Kind::Nibbles, 300, 15),
    fixture!("short_20k", Kind::Short, 20000, 21),
];

fn fixture(name: &str) -> &'static Fixture {
    FIXTURES
        .iter()
        .find(|fixture| fixture.name == name)
        .unwrap()
}

#[test]
fn frames_from_the_reference_encoder_decompress() {
    for fixture in &FIXTURES {
        let expected = sample(fixture.kind, fixture.len, fixture.seed);
        assert_eq!(
            decompress(fixture.frame, fixture.len).as_ref(),
            Ok(&expected),
            "{}",
            fixture.name
        );
        // The reference's frames state their content size.
        assert_eq!(
            parse_header(fixture.frame, MAGIC).unwrap().content_size,
            Some(u64::try_from(fixture.len).unwrap()),
            "{}",
            fixture.name
        );
    }
}

#[test]
fn the_reference_frames_hold_the_layouts_they_are_here_for() {
    let [ll, of, ml] = [0u8, 1, 2];
    let all = |mode: u8| BTreeSet::from([(ll, mode), (of, mode), (ml, mode)]);

    // Single-stream Huffman literals with compressed weights, and three described tables.
    let text = probe(fixture("text_600").frame);
    assert_eq!(text.blocks, [BLOCK_COMPRESSED]);
    assert_eq!(
        text.literals,
        BTreeSet::from([(LITERALS_COMPRESSED, HUFFMAN_ONE_STREAM)])
    );
    assert_eq!(text.direct_weights, BTreeSet::from([false]));
    assert_eq!(text.modes, all(MODE_FSE));
    assert_eq!(text.count_widths, BTreeSet::from([1]));
    assert_eq!(text.low_counts, BTreeSet::from([false]));

    // Predefined tables: the only description is the Huffman weights'.
    let skew = probe(fixture("skew_200").frame);
    assert_eq!(skew.modes, all(MODE_PREDEFINED));
    assert!(skew.direct_weights.contains(&false) && skew.low_counts.contains(&false));

    // Four streams, sizes in ten bits and in fourteen; a two-byte sequence count.
    assert_eq!(
        probe(fixture("skew_900").frame).literals,
        BTreeSet::from([(LITERALS_COMPRESSED, 1)])
    );
    let skew = probe(fixture("skew_2000").frame);
    assert_eq!(skew.literals, BTreeSet::from([(LITERALS_COMPRESSED, 2)]));
    assert_eq!(skew.count_widths, BTreeSet::from([2]));

    // Raw literals with a two-byte header, and an offset table in RLE mode.
    let runs = probe(fixture("runs_20k").frame);
    assert_eq!(runs.literals, BTreeSet::from([(LITERALS_RAW, 1)]));
    assert_eq!(
        runs.modes,
        BTreeSet::from([(ll, MODE_FSE), (of, MODE_RLE), (ml, MODE_FSE)])
    );

    // RLE blocks after a compressed one.
    let zeros = probe(fixture("zeros_300k").frame);
    assert_eq!(zeros.blocks, [BLOCK_COMPRESSED, BLOCK_RLE, BLOCK_RLE]);

    // A second block that repeats the first's tables and its Huffman tree.
    let two = probe(fixture("text_8k_tb").frame);
    assert_eq!(two.blocks, [BLOCK_COMPRESSED, BLOCK_COMPRESSED]);
    assert_eq!(
        two.literals,
        BTreeSet::from([
            (LITERALS_COMPRESSED, HUFFMAN_ONE_STREAM),
            (LITERALS_TREELESS, HUFFMAN_ONE_STREAM)
        ])
    );
    assert_eq!(
        two.modes,
        all(MODE_FSE).union(&all(MODE_REPEAT)).copied().collect()
    );

    // Weights stored directly, with a described offset table beside predefined ones.
    let nibbles = probe(fixture("nib_300").frame);
    assert_eq!(nibbles.direct_weights, BTreeSet::from([true]));
    assert_eq!(
        nibbles.modes,
        BTreeSet::from([(ll, MODE_PREDEFINED), (of, MODE_FSE), (ml, MODE_PREDEFINED)])
    );

    // A description with the low-counts flag (the encoder sets it from 2048 sequences up).
    let short = probe(fixture("short_20k").frame);
    assert_eq!(short.low_counts, BTreeSet::from([false, true]));
}

/// Where a fixture's only block keeps its Huffman-coded literals: the offset in the frame of
/// the first byte of the coded streams, and of the byte after the last.
fn huffman_streams(frame: &[u8]) -> (usize, usize) {
    let header = parse_header(frame, MAGIC).unwrap();
    let body = header.len + BLOCK_HEADER_SIZE;
    let literals = Literals::parse(&frame[body..]).unwrap();
    assert_eq!(literals.kind, LITERALS_COMPRESSED);
    let tree = body + literals.header_len;
    // The tree description here is FSE-compressed weights behind their length byte.
    assert!(frame[tree] < HUFFMAN_DIRECT_WEIGHTS);
    (
        tree + 1 + usize::from(frame[tree]),
        body + literals.section_len().unwrap(),
    )
}

#[test]
fn corrupt_huffman_literals_are_refused_where_the_reference_decodes_them() {
    // Stricter than the reference, through `ruzstd`: the reference's four-stream decoder does
    // not check that each stream ends exactly, and decodes this frame to 2000 wrong bytes.
    // The byte is one inside the coded literals for which a run of both decoders differed.
    const DAMAGED: usize = 444;
    let fixture = fixture("skew_2000");
    let (start, end) = huffman_streams(fixture.frame);
    assert!((start..end).contains(&DAMAGED));
    let mut frame = fixture.frame.to_vec();
    frame[DAMAGED] ^= 1;
    assert!(matches!(
        decompress(&frame, fixture.len),
        Err(ZbicError::Zstd { block: 0, .. })
    ));
}

#[test]
fn a_corrupt_end_of_single_stream_literals_decodes_where_the_reference_refuses() {
    // Looser than the reference, through `ruzstd`: a stream is read from its last byte to
    // its first, and `ruzstd` does not notice that this single stream no longer ends exactly at
    // its first byte. The reference answers `corruption_detected`; here the frame decodes, to
    // the declared length and the wrong bytes (an NSO's hash check then fails).
    let fixture = fixture("text_600");
    let (start, _) = huffman_streams(fixture.frame);
    let mut frame = fixture.frame.to_vec();
    frame[start] ^= 0x80;
    let out = decompress(&frame, fixture.len).unwrap();
    assert_eq!(out.len(), fixture.len);
    assert_ne!(out, sample(fixture.kind, fixture.len, fixture.seed));
}

#[test]
fn a_damaged_reference_frame_is_an_error_or_the_declared_size() {
    // Every byte of a small frame, flipped one bit at a time: never a panic, and whatever
    // decodes has the declared length.
    let fixture = fixture("text_600");
    for at in 0..fixture.frame.len() {
        for bit in 0..8 {
            let mut frame = fixture.frame.to_vec();
            frame[at] ^= 1 << bit;
            if let Ok(out) = decompress(&frame, fixture.len) {
                assert_eq!(out.len(), fixture.len, "byte {at}, bit {bit}");
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// The builder
// ---------------------------------------------------------------------------------------------

#[test]
fn the_builder_keeps_a_frame_and_changes_its_magic_and_tables() {
    let data = sample(Kind::Text, 5000, 3);
    let zstd = crate::nup::build::zstd_frame(&data).unwrap();
    let frame = build::from_zstd(&zstd).unwrap();
    assert_eq!(frame[..MAGIC_SIZE], MAGIC);
    // The header after the magic is the zstd frame's.
    let header = parse_header(&frame, MAGIC).unwrap();
    assert_eq!(parse_header(&zstd, ZSTD_MAGIC), Ok(header));
    assert_eq!(frame[MAGIC_SIZE..header.len], zstd[MAGIC_SIZE..header.len]);
    // The frame holds compressed blocks with described tables, written without the flag.
    let probe = probe(&frame);
    assert!(probe.blocks.contains(&BLOCK_COMPRESSED));
    assert!(probe.modes.contains(&(0, MODE_FSE)));
    assert!(probe.low_counts.contains(&false));
    assert_eq!(decompress(&frame, data.len()), Ok(data.clone()));
    assert_eq!(build::compress(&data).unwrap(), frame);

    // A frame without a content size, as `ruzstd` writes it.
    let bare = build::from_zstd(&compress_to_vec(&data[..], CompressionLevel::Fastest)).unwrap();
    assert_eq!(parse_header(&bare, MAGIC).unwrap().content_size, None);
    assert_eq!(decompress(&bare, data.len()), Ok(data));
}

#[test]
fn the_builder_refuses_what_is_not_one_zstd_frame() {
    let zstd = crate::nup::build::zstd_frame(HELLO).unwrap();
    assert!(build::from_zstd(&zstd).is_ok());
    let not_a_frame = |bytes: &[u8]| {
        let error = build::from_zstd(bytes).unwrap_err();
        assert!(matches!(error, BuildError::NotAFrame), "{error}");
        assert_eq!(error.to_string(), "the input is not one whole zstd frame");
    };
    not_a_frame(&[]);
    not_a_frame(&zstd[..zstd.len() - 1]);
    let mut longer = zstd.clone();
    longer.push(0);
    not_a_frame(&longer);
    // A ZBIC frame is not a zstd frame.
    not_a_frame(&build::from_zstd(&zstd).unwrap());

    // A table ZBIC cannot describe: accuracy log 13 (log field 8), one symbol with all 8192
    // slots, which is 8193 moved past the 8190 short values, in fourteen bits.
    // `8 | 16383 << 4 = 0x03FFF8`.
    let mut after = vec![0xF8, 0xFF, 0x03];
    assert_eq!(read_zstd(&after).unwrap().0, counts(13, &[8192]));
    after.push(0x01);
    let body = sequences_block(
        &[1],
        modes(MODE_FSE, MODE_PREDEFINED, MODE_PREDEFINED),
        &after,
    );
    let mut frame = header_bytes(ZSTD_MAGIC, 0, Some(0), &[], &[]);
    frame.extend(raw_block(false, HELLO));
    frame.extend(compressed_block(true, &body));
    let error = build::from_zstd(&frame).unwrap_err();
    assert!(
        matches!(
            error,
            BuildError::Block {
                block: 1,
                source: Some(TableBuildError::Unsupported)
            }
        ),
        "{error}"
    );
    assert_eq!(error.to_string(), "zstd block 1 cannot be written as ZBIC");
    // A block whose sections run past its end.
    let mut frame = header_bytes(ZSTD_MAGIC, 0, Some(0), &[], &[]);
    frame.extend(compressed_block(true, &[]));
    assert!(matches!(
        build::from_zstd(&frame).unwrap_err(),
        BuildError::Block {
            block: 0,
            source: None
        }
    ));
}

proptest! {
    #![proptest_config(seeded(48))]

    /// What the builder makes from generated data decompresses to that data.
    #[test]
    fn generated_frames_round_trip(
        kind in prop_oneof![
            Just(Kind::Text),
            Just(Kind::Skewed),
            Just(Kind::Runs),
            Just(Kind::Nibbles),
            Just(Kind::Short),
        ],
        len in 0usize..40_000,
        seed in any::<u64>(),
    ) {
        let data = sample(kind, len, seed);
        let frame = build::compress(&data).unwrap();
        prop_assert_eq!(decompress(&frame, len), Ok(data));
        // One byte more or less than what the frame says it holds.
        let mismatch = Err(ZbicError::ContentSizeMismatch {
            frame: u64::try_from(len).unwrap(),
            expected: len + 1,
        });
        prop_assert_eq!(decompress(&frame, len + 1), mismatch);
    }
}

proptest! {
    #![proptest_config(seeded(1500))]

    #[test]
    fn arbitrary_bytes_behind_the_magic_never_panic(
        tail in proptest::collection::vec(any::<u8>(), 0..600),
        size in 0usize..4096,
    ) {
        let mut bytes = MAGIC.to_vec();
        bytes.extend_from_slice(&tail);
        if let Ok(out) = decompress(&bytes, size) {
            prop_assert_eq!(out.len(), size);
        }
    }

    #[test]
    fn arbitrary_compressed_blocks_never_panic(
        body in proptest::collection::vec(any::<u8>(), 0..300),
        size in 0usize..4096,
    ) {
        let frame = windowed(&compressed_block(true, &body));
        if let Ok(out) = decompress(&frame, size) {
            prop_assert_eq!(out.len(), size);
        }
        let _ = build::from_zstd(&frame);
    }

    #[test]
    fn mutated_frames_never_panic(
        at in any::<proptest::sample::Index>(),
        value in any::<u8>(),
        cut in any::<proptest::sample::Index>(),
        which in 0usize..FIXTURES.len(),
    ) {
        // The 300 KiB fixture is as good as the others here and far slower.
        let fixture = &FIXTURES[which];
        prop_assume!(fixture.len <= 20_000);
        let mut frame = fixture.frame.to_vec();
        let at = at.index(frame.len());
        frame[at] = value;
        if let Ok(out) = decompress(&frame, fixture.len) {
            prop_assert_eq!(out.len(), fixture.len);
        }
        frame.truncate(cut.index(frame.len()));
        let _ = decompress(&frame, fixture.len);
    }
}
