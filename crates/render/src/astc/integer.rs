//! Bit fields and integer sequence encoding: how a block stores its weights and its colour
//! values (C.2.12).
//!
//! Ported from Eden's `astc.cpp` (`InputBitStream`, `IntegerEncodedValue`, `CreateEncoding`,
//! `DecodeTritBlock`, `DecodeQuintBlock`, `DecodeIntegerSequence`); see the parent module for
//! the attribution.

/// The low 32 bits of `value`.
fn low_u32(value: u128) -> u32 {
    let bytes = value.to_le_bytes();
    u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}

/// A mask of the low `count` bits; all of them for 128 or more.
pub(super) fn mask(count: u32) -> u128 {
    1_u128.checked_shl(count).map_or(u128::MAX, |bit| bit - 1)
}

/// Reads bit fields from a 128-bit value, lowest bit first. Bits past the top read as zero:
/// "if there are insufficient bits in the stream to fill the final block, then unused (higher
/// order) bits are assumed to be 0" (C.2.12).
#[derive(Clone, Copy, Debug)]
pub(super) struct BitReader {
    data: u128,
    position: u32,
}

impl BitReader {
    /// A reader at bit 0 of `data`.
    pub(super) const fn new(data: u128) -> Self {
        Self { data, position: 0 }
    }

    /// A reader at bit `position` of `data`.
    pub(super) const fn at(data: u128, position: u32) -> Self {
        Self { data, position }
    }

    /// The next `count` bits, at most 32, as a number whose lowest bit was read first.
    pub(super) fn take(&mut self, count: u32) -> u32 {
        let rest = self.data.checked_shr(self.position).unwrap_or(0);
        self.position = self.position.saturating_add(count);
        low_u32(rest & mask(count.min(u32::BITS)))
    }

    /// How many bits have been read.
    pub(super) const fn position(&self) -> u32 {
        self.position
    }
}

/// `count` bits of `value` starting at bit `low`.
const fn field(value: u32, low: u32, count: u32) -> u32 {
    (value >> low) & ((1 << count) - 1)
}

/// What a value of a sequence has besides plain bits (table C.2.13).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Digit {
    /// Nothing: the range is a power of two.
    None,
    /// A base-3 digit: the range is 3 times a power of two.
    Trit,
    /// A base-5 digit: the range is 5 times a power of two.
    Quint,
}

/// How the values of a sequence are stored: `bits` plain bits each, and maybe a trit or quint.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Encoding {
    /// The digit each value has.
    pub(super) digit: Digit,
    /// The plain bits each value has.
    pub(super) bits: u32,
}

impl Encoding {
    /// The encoding with the largest range that holds no value above `max` (Eden's
    /// `CreateEncoding`): for a `max` that is itself the top of a range, that range.
    pub(super) const fn for_max(max: u32) -> Self {
        let mut max = max;
        while max > 0 {
            let check = max + 1;
            if check.is_power_of_two() {
                return Self {
                    digit: Digit::None,
                    bits: max.count_ones(),
                };
            }
            if check.is_multiple_of(3) && (check / 3).is_power_of_two() {
                return Self {
                    digit: Digit::Trit,
                    bits: (check / 3 - 1).count_ones(),
                };
            }
            if check.is_multiple_of(5) && (check / 5).is_power_of_two() {
                return Self {
                    digit: Digit::Quint,
                    bits: (check / 5 - 1).count_ones(),
                };
            }
            max -= 1;
        }
        Self {
            digit: Digit::None,
            bits: 0,
        }
    }

    /// The bits `count` values take: `count * bits`, plus `ceil(8 * count / 5)` for trits or
    /// `ceil(7 * count / 3)` for quints (C.2.22). `count` is at most 64.
    pub(super) const fn bit_length(self, count: u32) -> u32 {
        let digits = match self.digit {
            Digit::None => 0,
            Digit::Trit => (count * 8).div_ceil(5),
            Digit::Quint => (count * 7).div_ceil(3),
        };
        count * self.bits + digits
    }
}

/// One decoded value of a sequence, before unquantization.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct Value {
    /// The plain bits.
    pub(super) bits: u32,
    /// The trit (0 to 2) or quint (0 to 4), or 0 when the encoding has neither.
    pub(super) digit: u32,
}

/// Five values packed with their trits (figure C.5, and the bit manipulation below it).
fn trit_block(reader: &mut BitReader, bits: u32) -> [Value; 5] {
    let m0 = reader.take(bits);
    let mut t = reader.take(2);
    let m1 = reader.take(bits);
    t |= reader.take(2) << 2;
    let m2 = reader.take(bits);
    t |= reader.take(1) << 4;
    let m3 = reader.take(bits);
    t |= reader.take(2) << 5;
    let m4 = reader.take(bits);
    t |= reader.take(1) << 7;

    let (c, t3, t4) = if field(t, 2, 3) == 7 {
        ((field(t, 5, 3) << 2) | field(t, 0, 2), 2, 2)
    } else if field(t, 5, 2) == 3 {
        (field(t, 0, 5), field(t, 7, 1), 2)
    } else {
        (field(t, 0, 5), field(t, 5, 2), field(t, 7, 1))
    };
    let (t0, t1, t2) = if field(c, 0, 2) == 3 {
        (
            (field(c, 3, 1) << 1) | (field(c, 2, 1) & !field(c, 3, 1)),
            field(c, 4, 1),
            2,
        )
    } else if field(c, 2, 2) == 3 {
        (field(c, 0, 2), 2, 2)
    } else {
        (
            (field(c, 1, 1) << 1) | (field(c, 0, 1) & !field(c, 1, 1)),
            field(c, 2, 2),
            field(c, 4, 1),
        )
    };
    [(m0, t0), (m1, t1), (m2, t2), (m3, t3), (m4, t4)].map(|(bits, digit)| Value { bits, digit })
}

/// Three values packed with their quints (figure C.6, and the bit manipulation below it).
fn quint_block(reader: &mut BitReader, bits: u32) -> [Value; 3] {
    let m0 = reader.take(bits);
    let mut q = reader.take(3);
    let m1 = reader.take(bits);
    q |= reader.take(2) << 3;
    let m2 = reader.take(bits);
    q |= reader.take(2) << 5;

    let (q0, q1, q2) = if field(q, 1, 2) == 3 && field(q, 5, 2) == 0 {
        let low = field(q, 0, 1);
        (
            4,
            4,
            (low << 2) | ((field(q, 4, 1) & !low) << 1) | (field(q, 3, 1) & !low),
        )
    } else {
        let (c, q2) = if field(q, 1, 2) == 3 {
            (
                (field(q, 3, 2) << 3) | ((!field(q, 5, 2) & 3) << 1) | field(q, 0, 1),
                4,
            )
        } else {
            (field(q, 0, 5), field(q, 5, 2))
        };
        if field(c, 0, 3) == 5 {
            (field(c, 3, 2), 4, q2)
        } else {
            (field(c, 0, 3), field(c, 3, 2), q2)
        }
    };
    [(m0, q0), (m1, q1), (m2, q2)].map(|(bits, digit)| Value { bits, digit })
}

/// Fills `values` with the next `values.len()` values of a sequence stored with `encoding`.
///
/// Only the sequence's own bits are read, `encoding.bit_length(values.len())` of them: a last
/// packed block that is partly wanted takes its missing higher bits as zero (C.2.12),
/// whatever the stream holds after the sequence. Eden's `DecodeIntegerSequence` reads the
/// whole block from the stream (the parent module lists the difference). The reader is left
/// after the last packed block.
pub(super) fn decode_sequence(reader: &mut BitReader, encoding: Encoding, values: &mut [Value]) {
    // A 128-bit stream holds at most 128 values, so a longer sequence is cut where its length
    // has passed the end of any stream.
    let count = u32::try_from(values.len()).map_or(u128::BITS, |count| count.min(u128::BITS));
    let end = reader.position.saturating_add(encoding.bit_length(count));
    let stream = reader.data;
    reader.data &= mask(end);
    read_sequence(reader, encoding, values);
    reader.data = stream;
}

/// [`decode_sequence`] over a stream that ends where the sequence does.
fn read_sequence(reader: &mut BitReader, encoding: Encoding, values: &mut [Value]) {
    match encoding.digit {
        Digit::None => {
            for value in values {
                *value = Value {
                    bits: reader.take(encoding.bits),
                    digit: 0,
                };
            }
        }
        Digit::Trit => {
            for wanted in values.chunks_mut(5) {
                let block = trit_block(reader, encoding.bits);
                for (value, decoded) in wanted.iter_mut().zip(block) {
                    *value = decoded;
                }
            }
        }
        Digit::Quint => {
            for wanted in values.chunks_mut(3) {
                let block = quint_block(reader, encoding.bits);
                for (value, decoded) in wanted.iter_mut().zip(block) {
                    *value = decoded;
                }
            }
        }
    }
}

/// `value`'s low `bits` bits repeated from the top down to fill `to` bits: "simple bit
/// replication from the most significant bit of the value" (C.2.13, C.2.17). `bits` is at
/// most `to`; no bits give 0.
pub(super) fn replicate(value: u32, bits: u32, to: u32) -> u32 {
    if bits == 0 {
        return 0;
    }
    let value = value & ((1 << bits) - 1);
    let mut result = 0;
    let mut filled = 0;
    while filled < to {
        let take = bits.min(to - filled);
        result = (result << take) | (value >> (bits - take));
        filled += take;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::{BitReader, Digit, Encoding, Value, decode_sequence, mask, replicate};

    const fn encoding(digit: Digit, bits: u32) -> Encoding {
        Encoding { digit, bits }
    }

    #[test]
    fn reads_fields_lowest_bit_first_and_zeros_past_the_top() {
        let mut reader = BitReader::new(0xF000_0000_0000_0000_0000_0000_0000_1234);
        assert_eq!(reader.take(4), 0x4);
        assert_eq!(reader.take(8), 0x23);
        assert_eq!(reader.take(0), 0);
        assert_eq!(reader.position(), 12);
        assert_eq!(reader.take(32), 0x1);
        assert_eq!(reader.position(), 44);
        let mut reader = BitReader::at(u128::MAX, 124);
        // Four bits are left; the rest of the eight read as zero, and so does all after.
        assert_eq!(reader.take(8), 0xF);
        assert_eq!(reader.take(8), 0);
        assert_eq!(BitReader::at(u128::MAX, 127).take(1), 1);
        assert_eq!(BitReader::at(u128::MAX, 128).take(1), 0);
        assert_eq!(BitReader::at(u128::MAX, u32::MAX).take(32), 0);
        assert_eq!(BitReader::new(u128::MAX).take(32), u32::MAX);
        assert_eq!(BitReader::new(0x0807_0605_0403_0201).take(32), 0x0403_0201);
        assert_eq!(
            BitReader::at(0x0807_0605_0403_0201, 8).take(32),
            0x0504_0302
        );
    }

    #[test]
    fn masks() {
        assert_eq!(mask(0), 0);
        assert_eq!(mask(1), 1);
        assert_eq!(mask(96), (1 << 96) - 1);
        assert_eq!(mask(127), u128::MAX >> 1);
        assert_eq!(mask(128), u128::MAX);
        assert_eq!(mask(200), u128::MAX);
    }

    /// The ranges of tables C.2.7 (weights) and C.2.16 (colours): each maximum with the
    /// trits, quints and bits its values have.
    #[test]
    fn the_encoding_of_each_range() {
        let ranges = [
            (1, Digit::None, 1),
            (2, Digit::Trit, 0),
            (3, Digit::None, 2),
            (4, Digit::Quint, 0),
            (5, Digit::Trit, 1),
            (7, Digit::None, 3),
            (9, Digit::Quint, 1),
            (11, Digit::Trit, 2),
            (15, Digit::None, 4),
            (19, Digit::Quint, 2),
            (23, Digit::Trit, 3),
            (31, Digit::None, 5),
            (39, Digit::Quint, 3),
            (47, Digit::Trit, 4),
            (63, Digit::None, 6),
            (79, Digit::Quint, 4),
            (95, Digit::Trit, 5),
            (127, Digit::None, 7),
            (159, Digit::Quint, 5),
            (191, Digit::Trit, 6),
            (255, Digit::None, 8),
        ];
        for (index, (max, digit, bits)) in ranges.iter().enumerate() {
            assert_eq!(Encoding::for_max(*max), encoding(*digit, *bits), "{max}");
            // A maximum between two ranges gets the lower one: one short of this range's top
            // is the range before.
            if let Some((_, lower_digit, lower_bits)) = index.checked_sub(1).map(|i| ranges[i]) {
                assert_eq!(
                    Encoding::for_max(max - 1),
                    encoding(lower_digit, lower_bits),
                    "{}",
                    max - 1
                );
            }
        }
        assert_eq!(Encoding::for_max(0), encoding(Digit::None, 0));
    }

    /// C.2.22's bit counts: `ceil(8n/5)` for trits, `ceil(7n/3)` for quints, plus n times the
    /// plain bits.
    #[test]
    fn bit_lengths() {
        // Whole packed blocks, and one value more or fewer.
        assert_eq!(encoding(Digit::Trit, 0).bit_length(5), 8);
        assert_eq!(encoding(Digit::Trit, 0).bit_length(4), 7);
        assert_eq!(encoding(Digit::Trit, 0).bit_length(6), 10);
        assert_eq!(encoding(Digit::Trit, 0).bit_length(1), 2);
        assert_eq!(encoding(Digit::Quint, 0).bit_length(3), 7);
        assert_eq!(encoding(Digit::Quint, 0).bit_length(2), 5);
        assert_eq!(encoding(Digit::Quint, 0).bit_length(4), 10);
        assert_eq!(encoding(Digit::Quint, 0).bit_length(1), 3);
        assert_eq!(encoding(Digit::None, 3).bit_length(7), 21);
        // The specification's own figures: a trit block with 4-bit values is 8 + 5 * 4 bits,
        // a quint block 7 + 3 * 4.
        assert_eq!(encoding(Digit::Trit, 4).bit_length(5), 28);
        assert_eq!(encoding(Digit::Quint, 4).bit_length(3), 19);
        assert_eq!(encoding(Digit::None, 8).bit_length(0), 0);
        // The largest sequences a block can ask for.
        assert_eq!(encoding(Digit::None, 5).bit_length(64), 320);
        assert_eq!(encoding(Digit::Trit, 3).bit_length(64), 103 + 192);
    }

    fn digits<const N: usize>(values: [Value; N]) -> [u32; N] {
        values.map(|value| value.digit)
    }

    /// Five trits are packed into 8 bits (C.2.12): every one of the 256 codes decodes to five
    /// digits of 0 to 2, and all 3^5 = 243 combinations are reachable.
    #[test]
    fn every_trit_code_decodes_and_every_combination_is_reachable() {
        let mut seen = [false; 243];
        for code in 0..256_u128 {
            let mut values = [Value::default(); 5];
            decode_sequence(
                &mut BitReader::new(code),
                encoding(Digit::Trit, 0),
                &mut values,
            );
            let trits = digits(values);
            assert!(trits.iter().all(|t| *t <= 2), "{code}: {trits:?}");
            let index = trits.iter().rev().fold(0, |index, t| index * 3 + t);
            seen[index as usize] = true;
        }
        assert!(seen.iter().all(|seen| *seen));
    }

    /// Three quints are packed into 7 bits: every one of the 128 codes decodes to three
    /// digits of 0 to 4, and all 5^3 = 125 combinations are reachable.
    #[test]
    fn every_quint_code_decodes_and_every_combination_is_reachable() {
        let mut seen = [false; 125];
        for code in 0..128_u128 {
            let mut values = [Value::default(); 3];
            decode_sequence(
                &mut BitReader::new(code),
                encoding(Digit::Quint, 0),
                &mut values,
            );
            let quints = digits(values);
            assert!(quints.iter().all(|q| *q <= 4), "{code}: {quints:?}");
            let index = quints.iter().rev().fold(0, |index, q| index * 5 + q);
            seen[index as usize] = true;
        }
        assert!(seen.iter().all(|seen| *seen));
    }

    /// Codes worked through C.2.12's bit manipulation by hand.
    #[test]
    fn trit_and_quint_codes_by_hand() {
        let trits = |code: u128| {
            let mut values = [Value::default(); 5];
            decode_sequence(
                &mut BitReader::new(code),
                encoding(Digit::Trit, 0),
                &mut values,
            );
            digits(values)
        };
        // C = T[4:0]; t0 = { C[1], C[0] & ~C[1] }, t1 = C[3:2], t2 = C[4], t3 = T[6:5],
        // t4 = T[7].
        assert_eq!(trits(0b0000_0000), [0, 0, 0, 0, 0]);
        assert_eq!(trits(0b0000_0001), [1, 0, 0, 0, 0]);
        assert_eq!(trits(0b0000_0010), [2, 0, 0, 0, 0]);
        assert_eq!(trits(0b0000_0100), [0, 1, 0, 0, 0]);
        assert_eq!(trits(0b0000_1000), [0, 2, 0, 0, 0]);
        assert_eq!(trits(0b0001_0000), [0, 0, 1, 0, 0]);
        assert_eq!(trits(0b0010_0000), [0, 0, 0, 1, 0]);
        assert_eq!(trits(0b0100_0000), [0, 0, 0, 2, 0]);
        assert_eq!(trits(0b1000_0000), [0, 0, 0, 0, 1]);
        // T[4:2] = 111: C = { T[7:5], T[1:0] } = 0, t4 = t3 = 2.
        assert_eq!(trits(0b0001_1100), [0, 0, 0, 2, 2]);
        // T[6:5] = 11 (and T[4:2] not 111): t4 = 2, t3 = T[7].
        assert_eq!(trits(0b1110_0000), [0, 0, 0, 1, 2]);
        // C[1:0] = 11: t2 = 2, t1 = C[4], t0 = { C[3], C[2] & ~C[3] }.
        assert_eq!(trits(0b0000_0111), [1, 0, 2, 0, 0]);
        assert_eq!(trits(0b0001_1011), [2, 1, 2, 0, 0]);
        // C[3:2] = 11 (and C[1:0] not 11): t2 = t1 = 2, t0 = C[1:0].
        assert_eq!(trits(0b0000_1101), [1, 2, 2, 0, 0]);

        let quints = |code: u128| {
            let mut values = [Value::default(); 3];
            decode_sequence(
                &mut BitReader::new(code),
                encoding(Digit::Quint, 0),
                &mut values,
            );
            digits(values)
        };
        // q2 = Q[6:5], C = Q[4:0], q1 = C[4:3], q0 = C[2:0].
        assert_eq!(quints(0b000_0000), [0, 0, 0]);
        assert_eq!(quints(0b000_0100), [4, 0, 0]);
        assert_eq!(quints(0b001_1000), [0, 3, 0]);
        assert_eq!(quints(0b110_0000), [0, 0, 3]);
        assert_eq!(quints(0b100_1000), [0, 1, 2]);
        // C[2:0] = 101: q1 = 4, q0 = C[4:3].
        assert_eq!(quints(0b001_0101), [2, 4, 0]);
        // Q[2:1] = 11 and Q[6:5] = 00: q0 = q1 = 4, q2 = { Q[0], Q[4] & ~Q[0], Q[3] & ~Q[0] }.
        assert_eq!(quints(0b000_0110), [4, 4, 0]);
        assert_eq!(quints(0b001_1110), [4, 4, 3]);
        assert_eq!(quints(0b001_1111), [4, 4, 4]);
        // Q[2:1] = 11 otherwise: q2 = 4, C = { Q[4:3], ~Q[6:5], Q[0] }.
        // Q = 0b010_0110: C = { 00, ~01 = 10, 0 } = 0b00100: q1 = 0, q0 = 4.
        assert_eq!(quints(0b010_0110), [4, 0, 4]);
    }

    /// The plain bits of a packed block sit between the digit bits as figures C.5 and C.6
    /// show (drawn there for 4-bit values).
    #[test]
    fn plain_bits_are_interleaved_with_the_digit_bits() {
        // Trits: m0 at bits 3..0, T1 T0 at 5..4, m1 at 9..6, T3 T2 at 11..10, m2 at 15..12,
        // T4 at 16, m3 at 20..17, T6 T5 at 22..21, m4 at 26..23, T7 at 27.
        let code = 0x1 | (0x2 << 6) | (0x3 << 12) | (0x4 << 17) | (0x5 << 23) | (1 << 27);
        let mut values = [Value::default(); 5];
        decode_sequence(
            &mut BitReader::new(code),
            encoding(Digit::Trit, 4),
            &mut values,
        );
        assert_eq!(values.map(|v| v.bits), [1, 2, 3, 4, 5]);
        assert_eq!(digits(values), [0, 0, 0, 0, 1]);
        // Quints: m0 at 3..0, Q2 Q1 Q0 at 6..4, m1 at 10..7, Q4 Q3 at 12..11, m2 at 16..13,
        // Q6 Q5 at 18..17.
        let code = 0x9 | (0b100 << 4) | (0xA << 7) | (0b01 << 11) | (0xB << 13) | (0b10 << 17);
        let mut values = [Value::default(); 3];
        decode_sequence(
            &mut BitReader::new(code),
            encoding(Digit::Quint, 4),
            &mut values,
        );
        assert_eq!(values.map(|v| v.bits), [9, 10, 11]);
        // Q = 0b10_01_100: q2 = 2, C = 0b01100: q1 = 1, q0 = 4.
        assert_eq!(digits(values), [4, 1, 2]);
    }

    /// A sequence that ends part-way through a packed block takes the block's first values;
    /// the next block starts after a whole one.
    #[test]
    fn sequences_span_blocks_and_stop_part_way() {
        // Seven trits with no plain bits: T = 0b0000_0110 (t0 = 2, t1 = 1) then
        // T = 0b0000_0001.
        let code = 0b0000_0110 | (0b0000_0001 << 8);
        let mut values = [Value::default(); 7];
        decode_sequence(
            &mut BitReader::new(code),
            encoding(Digit::Trit, 0),
            &mut values,
        );
        assert_eq!(digits(values), [2, 1, 0, 0, 0, 1, 0]);
        // Four quints: Q = 0b000_0011 then Q = 0b000_0010.
        let code = 0b000_0011 | (0b000_0010 << 7);
        let mut values = [Value::default(); 4];
        decode_sequence(
            &mut BitReader::new(code),
            encoding(Digit::Quint, 0),
            &mut values,
        );
        assert_eq!(digits(values), [3, 0, 0, 2]);
        // Plain values: three of five bits.
        let mut values = [Value::default(); 3];
        let mut reader = BitReader::new(0b10101_00010_11111);
        decode_sequence(&mut reader, encoding(Digit::None, 5), &mut values);
        assert_eq!(values.map(|v| v.bits), [31, 2, 21]);
        assert_eq!(digits(values), [0, 0, 0]);
        assert_eq!(reader.position(), 15);
        // No values: nothing is read.
        let mut reader = BitReader::new(u128::MAX);
        decode_sequence(&mut reader, encoding(Digit::Trit, 6), &mut []);
        assert_eq!(reader.position(), 0);
    }

    /// Only a sequence's own bits are read. A last packed block that is partly wanted takes
    /// its missing higher bits as zero (C.2.12), whatever follows the sequence in the stream:
    /// in a block that is the unused colour bits, which need not be zero.
    #[test]
    fn bits_after_a_sequence_are_not_read() {
        let decode = |stream: u128, encoding: Encoding, count: usize| {
            let mut values = [Value::default(); 8];
            decode_sequence(&mut BitReader::new(stream), encoding, &mut values[..count]);
            values.map(|value| (value.bits, value.digit))
        };

        // Eight values of a trit and one bit: 8 + ceil(8 * 8 / 5) = 21 bits. The second
        // packed block starts at bit 13 and holds values 5 to 7: m0 at 13, T1 T0 at 15..14,
        // m1 at 16, T3 T2 at 18..17, m2 at 19 and T4 at 20. A whole block would go on with
        // m3 at 21, T6 T5 at 23..22, m4 at 24 and T7 at 25.
        //
        // The first block is zero. The second has m0 = 1, m2 = 1 and T[4:0] = 11100. With
        // T[7:5] = 000, T[4:2] = 111 gives C = { T[7:5], T[1:0] } = 00000, so
        // t0 = t1 = t2 = 0. Were T5 read as 1, C would be 00100 and t1 = 1; were T[7:5]
        // read as 111, C would be 11100 and t1 = t2 = 2.
        let trits = encoding(Digit::Trit, 1);
        assert_eq!(trits.bit_length(8), 21);
        let sequence: u128 = (1 << 13) | (0b11 << 17) | (1 << 19) | (1 << 20);
        let expected = [
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (1, 0),
            (0, 0),
            (1, 0),
        ];
        for after in [0, 1 << 22, 0b111 << 22, 1 << 25, u128::MAX << 21] {
            assert_eq!(decode(sequence | after, trits, 8), expected, "{after:#x}");
        }
        // The same block wanted whole reads its own bits: ten values take 26 bits.
        let mut whole = [Value::default(); 10];
        decode_sequence(
            &mut BitReader::new(sequence | (0b111 << 22)),
            trits,
            &mut whole,
        );
        // C = { T[7:5] = 011, T[1:0] = 00 } = 01100: C[3:2] = 11, so t0 = 0, t1 = t2 = 2;
        // T[4:2] = 111, so t3 = t4 = 2.
        assert_eq!(
            whole[5..].iter().map(|v| v.digit).collect::<Vec<_>>(),
            [0, 2, 2, 2, 2]
        );

        // Two values of a quint and one bit: 2 + ceil(7 * 2 / 3) = 7 bits: m0 at 0,
        // Q2 Q1 Q0 at 3..1, m1 at 4 and Q4 Q3 at 6..5. A whole block would go on with m2 at
        // 7 and Q6 Q5 at 9..8.
        //
        // m0 = 1, Q[2:0] = 110 and Q[4:3] = 01. With Q[6:5] = 00, Q[2:1] = 11 gives
        // q0 = q1 = 4. Were Q5 read as 1: q2 = 4 and C = { Q[4:3], ~Q[6:5], Q0 } = 01100,
        // so q1 = 1.
        let quints = encoding(Digit::Quint, 1);
        assert_eq!(quints.bit_length(2), 7);
        let sequence: u128 = 1 | (0b110 << 1) | (0b01 << 5);
        let mut expected = [(0, 0); 8];
        expected[0] = (1, 4);
        expected[1] = (0, 4);
        for after in [0, 1 << 8, 0b11 << 8, u128::MAX << 7] {
            assert_eq!(decode(sequence | after, quints, 2), expected, "{after:#x}");
        }

        // The exact edge, where the first bit after the sequence is a digit bit. Two trits
        // with no plain bits take ceil(8 * 2 / 5) = 4 bits, T[3:0], and bit 4 would be T4.
        // T[3:0] = 1100 with T4 = 0: C = 01100, C[3:2] = 11, so t0 = C[1:0] = 0 and t1 = 2.
        // Were T4 read as 1, T[4:2] = 111 would give C = 00000 and t1 = 0; were T3 not read,
        // C = 00100 and t1 = 1.
        let bare_trits = encoding(Digit::Trit, 0);
        assert_eq!(bare_trits.bit_length(2), 4);
        let mut expected = [(0, 0); 8];
        expected[1] = (0, 2);
        for after in [0, 1 << 4, u128::MAX << 4] {
            assert_eq!(
                decode(0b1100 | after, bare_trits, 2),
                expected,
                "{after:#x}"
            );
        }
        // One quint with no plain bits takes ceil(7 / 3) = 3 bits, Q[2:0], and bit 3 would
        // be Q3. Q[2:0] = 101 with Q[4:3] = 00: C[2:0] = 101, so q0 = C[4:3] = 0. Were Q3
        // read as 1, q0 = 1; were Q2 not read, C = 00001 and q0 = 1.
        let bare_quints = encoding(Digit::Quint, 0);
        assert_eq!(bare_quints.bit_length(1), 3);
        for after in [0, 1 << 3, u128::MAX << 3] {
            assert_eq!(
                decode(0b101 | after, bare_quints, 1),
                [(0, 0); 8],
                "{after:#x}"
            );
        }

        // A sequence that does not start at bit 0 ends its own length after its start: the
        // two trits again, from bit 9, with ones before and after them.
        let mut reader = BitReader::at((0b1100 << 9) | (u128::MAX << 13) | 0x1FF, 9);
        let mut values = [Value::default(); 2];
        decode_sequence(&mut reader, bare_trits, &mut values);
        assert_eq!(values.map(|value| value.digit), [0, 2]);

        // Plain bits: three values of five bits, and ones after them.
        let mut expected = [(0, 0); 8];
        expected[..3].copy_from_slice(&[(31, 0), (2, 0), (21, 0)]);
        assert_eq!(
            decode(
                0b10101_00010_11111 | (u128::MAX << 15),
                encoding(Digit::None, 5),
                3
            ),
            expected
        );

        // The reader itself goes on from after the last packed block, in the stream as it
        // is: two whole trit blocks of 13 bits.
        let mut reader = BitReader::new(u128::MAX << 21);
        decode_sequence(&mut reader, trits, &mut [Value::default(); 8]);
        assert_eq!(reader.position(), 26);
        assert_eq!(reader.take(4), 0b1111);
    }

    #[test]
    fn replication() {
        // To 8 bits: one bit fills the byte, 2 bits repeat four times, 3 bits "abc" give
        // "abcabcab", 5 bits "abcde" give "abcdeabc", 8 bits are themselves.
        assert_eq!(replicate(1, 1, 8), 0xFF);
        assert_eq!(replicate(0, 1, 8), 0x00);
        assert_eq!(replicate(0b10, 2, 8), 0b1010_1010);
        assert_eq!(replicate(0b110, 3, 8), 0b1101_1011);
        assert_eq!(replicate(0b10011, 5, 8), 0b1001_1100);
        assert_eq!(replicate(0b101_1001, 7, 8), 0b1011_0011);
        assert_eq!(replicate(0xA7, 8, 8), 0xA7);
        // To 6 bits, as weights are: 2 bits "ab" give "ababab", 5 bits "abcde" give "abcdea".
        assert_eq!(replicate(0b01, 2, 6), 0b01_0101);
        assert_eq!(replicate(0b10000, 5, 6), 0b10_0001);
        assert_eq!(replicate(0b1111, 4, 6), 0b11_1111);
        // Bits above the low `bits` are ignored, and no bits give 0.
        assert_eq!(replicate(0b1110, 2, 8), 0b1010_1010);
        assert_eq!(replicate(0xFF, 0, 8), 0);
        // Every full-scale value stays full scale, every zero stays zero.
        for bits in 1..=8 {
            assert_eq!(replicate((1 << bits) - 1, bits, 8), 0xFF, "{bits}");
            assert_eq!(replicate(0, bits, 8), 0, "{bits}");
        }
    }
}
