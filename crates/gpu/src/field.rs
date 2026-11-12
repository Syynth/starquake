//! Bit fields of a 32-bit command word, named the way NVIDIA's class headers name them.
//!
//! The class headers in Mesa (`mesa-nouveau/src/nouveau/headers/nvidia/classes/`) give each
//! field as `high:low` bit positions within a 32-bit word (`NVC76F_DMA_METHOD_COUNT 28:16`).
//! [`Field`] holds one such position, and the decoders and encoders of [`crate::fifo`] and
//! [`crate::sema`] read and write every field through a named `Field` constant, so a decoder
//! and its encoder can never disagree about where a field lives.
//!
//! Every operation is total: a field whose `high` is below its `low`, or above bit 31, is empty
//! (width 0) and reads as zero, rather than panicking. Every `Field` constant in this crate is
//! checked to be non-empty by a test.

/// A bit field `high:low` (inclusive) of a 32-bit word.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Field {
    high: u32,
    low: u32,
}

impl Field {
    /// The field `high:low`, written as in NVIDIA's class headers.
    pub const fn new(high: u32, low: u32) -> Self {
        Self { high, low }
    }

    /// The lowest bit of the field (its shift).
    pub const fn low(self) -> u32 {
        self.low
    }

    /// The number of bits in the field; 0 for an inverted or out-of-range position.
    pub const fn width(self) -> u32 {
        if self.high < self.low || self.high > 31 {
            0
        } else {
            self.high - self.low + 1
        }
    }

    /// The largest value the field holds (its mask, unshifted).
    pub const fn max(self) -> u32 {
        match 1u32.checked_shl(self.width()) {
            Some(bit) => bit.wrapping_sub(1),
            None => u32::MAX,
        }
    }

    /// Whether `value` fits in the field.
    pub const fn fits(self, value: u32) -> bool {
        value & !self.max() == 0
    }

    /// The field's value in `word`.
    pub const fn get(self, word: u32) -> u32 {
        match word.checked_shr(self.low) {
            Some(shifted) => shifted & self.max(),
            None => 0,
        }
    }

    /// Whether the field is non-zero in `word` (for one-bit flags).
    pub const fn is_set(self, word: u32) -> bool {
        self.get(word) != 0
    }

    /// `word` with the field replaced by `value`, or `None` if `value` does not fit.
    pub const fn insert(self, word: u32, value: u32) -> Option<u32> {
        if !self.fits(value) {
            return None;
        }
        // An empty field has max 0, so only value 0 reaches here and nothing changes.
        let (Some(mask), Some(bits)) = (
            self.max().checked_shl(self.low),
            value.checked_shl(self.low),
        ) else {
            return Some(word);
        };
        Some((word & !mask) | bits)
    }

    /// `word` with the field replaced by the low [`Field::width`] bits of `value`. For encoders
    /// whose values come from constants that fit by construction; [`Field::insert`] is for
    /// values a caller supplies.
    pub const fn insert_truncated(self, word: u32, value: u32) -> u32 {
        match self.insert(word, value & self.max()) {
            Some(word) => word,
            None => word,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Field;

    #[test]
    fn a_middle_field_reads_and_writes_only_its_bits() {
        // 28:16 is NVC76F_DMA_METHOD_COUNT: 13 bits.
        let count = Field::new(28, 16);
        assert_eq!(count.width(), 13);
        assert_eq!(count.max(), 0x1fff);
        assert_eq!(count.low(), 16);
        assert_eq!(count.get(0xffff_ffff), 0x1fff);
        assert_eq!(count.get(0x0001_0000), 1);
        assert_eq!(count.get(0x0000_ffff), 0);
        assert_eq!(count.get(0xe000_0000), 0);
        assert_eq!(count.insert(0xffff_ffff, 0), Some(0xe000_ffff));
        assert_eq!(count.insert(0, 0x1fff), Some(0x1fff_0000));
    }

    #[test]
    fn insert_rejects_a_value_one_past_the_field() {
        let count = Field::new(28, 16);
        assert!(count.fits(count.max()));
        assert!(!count.fits(count.max() + 1));
        assert_eq!(count.insert(0, count.max() + 1), None);
    }

    #[test]
    fn insert_truncated_keeps_only_the_fields_width() {
        let op = Field::new(2, 0);
        assert_eq!(op.insert_truncated(0xffff_fff0, 0b101), 0xffff_fff5);
        assert_eq!(op.insert_truncated(0, op.max() + 1), 0);
        assert_eq!(op.insert_truncated(0, 0xffff_ffff), 0b111);
        assert_eq!(Field::new(3, 4).insert_truncated(0x55, 1), 0x55);
    }

    #[test]
    fn a_full_word_field_covers_every_bit() {
        let all = Field::new(31, 0);
        assert_eq!(all.width(), 32);
        assert_eq!(all.max(), u32::MAX);
        assert_eq!(all.get(0x8000_0001), 0x8000_0001);
        assert_eq!(all.insert(0x1234_5678, u32::MAX), Some(u32::MAX));
    }

    #[test]
    fn the_top_bit_is_a_one_bit_field() {
        let top = Field::new(31, 31);
        assert_eq!(top.width(), 1);
        assert!(top.is_set(0x8000_0000));
        assert!(!top.is_set(0x7fff_ffff));
        assert_eq!(top.insert(0, 1), Some(0x8000_0000));
    }

    #[test]
    fn an_inverted_or_out_of_range_field_is_empty_and_never_panics() {
        for field in [Field::new(3, 4), Field::new(32, 0), Field::new(40, 33)] {
            assert_eq!(field.width(), 0, "{field:?}");
            assert_eq!(field.max(), 0);
            assert_eq!(field.get(u32::MAX), 0);
            assert!(field.fits(0));
            assert!(!field.fits(1));
            assert_eq!(field.insert(0x55, 0), Some(0x55));
            assert_eq!(field.insert(0x55, 1), None);
        }
    }
}
