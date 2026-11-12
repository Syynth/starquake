//! One SM86 instruction word and the bit reader over it.
//!
//! An instruction is 128 bits, stored in guest memory as four little-endian 32-bit words; bit
//! `n` of the instruction is bit `n % 32` of word `n / 32` (NAK's encoder writes the words that
//! way: `mesa-nouveau/src/nouveau/compiler/nak/sm70_encode.rs`, `SM70Encoder::inst: [u32; 4]`
//! viewed through `BitMutView`, and `encode_sm70_shader` appends the words in order; PR #4588
//! reads them the same way: `eden-pr4588/src/shader_recompiler/frontend/ampere/instruction.h`,
//! `Instruction::Bits`).

use thiserror::Error;

/// A bit field of an instruction: `width` bits starting at bit `start`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Field {
    /// The first bit.
    pub start: u32,
    /// How many bits.
    pub width: u32,
}

impl Field {
    /// The field of `width` bits starting at `start`.
    pub const fn new(start: u32, width: u32) -> Self {
        Self { start, width }
    }

    /// A one-bit field.
    pub const fn bit(start: u32) -> Self {
        Self::new(start, 1)
    }

    /// The field's bits as a mask over the whole instruction. Bits past bit 127 are dropped.
    pub const fn mask(self) -> u128 {
        let ones = match u128::MAX.checked_shr(128_u32.saturating_sub(self.width)) {
            Some(ones) => ones,
            // A zero width shifts by 128, which is no bits.
            None => 0,
        };
        match ones.checked_shl(self.start) {
            Some(mask) => mask,
            None => 0,
        }
    }
}

/// One 128-bit instruction word.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct Instruction(u128);

/// The bytes of a program are not a whole number of instructions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
#[error("a program of {len} bytes is not a whole number of {size}-byte instructions")]
pub struct LengthError {
    /// The length given.
    pub len: usize,
    /// [`Instruction::SIZE`].
    pub size: usize,
}

impl Instruction {
    /// The size of one instruction in bytes.
    pub const SIZE: usize = 16;

    /// The instruction with these 128 bits.
    pub const fn from_bits(bits: u128) -> Self {
        Self(bits)
    }

    /// The instruction's 128 bits.
    pub const fn bits(self) -> u128 {
        self.0
    }

    /// The instruction stored as these 16 bytes of guest memory.
    pub const fn from_le_bytes(bytes: [u8; Self::SIZE]) -> Self {
        Self(u128::from_le_bytes(bytes))
    }

    /// The 16 bytes this instruction is stored as.
    pub const fn to_le_bytes(self) -> [u8; Self::SIZE] {
        self.0.to_le_bytes()
    }

    /// Splits program bytes into instructions; the length must be a multiple of
    /// [`Instruction::SIZE`].
    pub fn from_program(bytes: &[u8]) -> Result<Vec<Self>, LengthError> {
        let (words, rest) = bytes.as_chunks::<{ Self::SIZE }>();
        if !rest.is_empty() {
            return Err(LengthError {
                len: bytes.len(),
                size: Self::SIZE,
            });
        }
        Ok(words
            .iter()
            .map(|word| Self::from_le_bytes(*word))
            .collect())
    }

    /// Reads `field`. Bits of the field past bit 127 read as zero; a field wider than 64 bits
    /// keeps its low 64 bits.
    pub fn field(self, field: Field) -> u64 {
        let shifted = self.0.checked_shr(field.start).unwrap_or(0);
        let low = Field::new(0, field.width).mask() & u128::from(u64::MAX);
        // Cannot fail: the value was masked to 64 bits.
        u64::try_from(shifted & low).unwrap_or(u64::MAX)
    }

    /// Reads a one-bit field as a flag.
    pub fn flag(self, field: Field) -> bool {
        self.field(field) != 0
    }

    /// This instruction with `field` set to the low bits of `value` (higher bits of `value` are
    /// dropped).
    #[must_use]
    pub fn with_field(self, field: Field, value: u64) -> Self {
        let mask = field.mask();
        let placed = u128::from(value).checked_shl(field.start).unwrap_or(0) & mask;
        Self((self.0 & !mask) | placed)
    }

    /// This instruction with the one-bit `field` set to `value`.
    #[must_use]
    pub fn with_flag(self, field: Field, value: bool) -> Self {
        self.with_field(field, u64::from(value))
    }
}
