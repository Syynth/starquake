//! GPFIFO entries and push-buffer method headers: pure decoders, and an encoder for building
//! method streams in tests (gpu-protocol spec, slice G1, "The method stream").
//!
//! Layouts are NVIDIA's, from `mesa-nouveau/src/nouveau/headers/nvidia/classes/clc76f.h` @
//! 52238762 (class `AMPERE_CHANNEL_GPFIFO_B`; Copyright (c) 2020, NVIDIA CORPORATION, MIT
//! licence), cross-checked against `eden-nx2/src/video_core/dma_pusher.h` @ 50fa86bb
//! (`CommandListHeader`, `CommandHeader`, `SubmissionMode`; GPL-3.0-or-later). The encoder's
//! header forms follow `mesa-nouveau/src/nouveau/headers/nv_push.h` (`NVC0_FIFO_PKHDR_SQ`,
//! `_IL`, `_1I`, `_0I`; MIT). Where the two references disagree, the header settles the layout
//! (owner's decision of 2026-11-12, "one house rule when references disagree").
//!
//! - A **GP entry** ([`GpEntry`]) is 8 bytes naming a run of command words in GPU memory.
//!   Decoding is total: every 64-bit value is an entry.
//! - A **method header** ([`MethodHeader`]) is the word before a method's data words.
//!   [`decode_header`] maps every word to a header or a [`DecodeError`], never a panic.
//! - [`PushBuffer`] lays out headers and data words, as Mesa's `nv_push.h` does.
//!
//! Every bit position is one [`Field`] constant below, read by the decoders and written by the
//! encoders. Method numbers are dword indices (byte address / 4), as in `crate::mme`.
//!
//! What a header *means* to a channel (host methods, parking, split methods, which headers are
//! skipped) is the channel state machine's (slice G8), not this module's.
//!
//! # Divergences from Eden-NX2
//!
//! Each is pinned by a test in `fifo/tests.rs`:
//!
//! - **Method address width.** Eden's `CommandHeader::method` reads 13 bits (12:0); the header's
//!   `NVC76F_DMA_METHOD_ADDRESS` is 11:0, so bit 12 is ignored here
//!   (`bit_12_is_not_part_of_the_method`).
//! - **GP entry address.** Eden's `CommandListHeader::addr` is bits 39:0 of the raw entry; the
//!   header's `NVC76F_GP_ENTRY0_GET` is 31:2, with bit 0 the `FETCH` flag. Bits 1:0 are not
//!   part of the address here (`entry0_low_bits_are_fetch_and_ignored_not_address`).
//! - **GP entry bit 41.** Eden names it `is_push_buffer`; the header names entry 1 bit 9
//!   `LEVEL` (main or subroutine), which this module follows. Eden's `allow_flush` (entry 1
//!   bit 8) is not in the header and is ignored (`entry1_bit_8_is_ignored`).
//! - **Opcodes 0 and 2.** Eden's `ProcessCommands` skips modes 0 and 2 (`IncreasingOld`,
//!   `NonIncreasingOld`) as single words. The header defines them by a tertiary opcode
//!   (`NVC76F_DMA_TERT_OP`, 17:16): tertiary 0 is an old-format method header
//!   ([`MethodHeader::Legacy`], `NVC76F_DMA_METHOD_ADDRESS_OLD` 12:2 and `_COUNT_OLD` 28:18),
//!   and opcode 0 with tertiary 1–3 is a subdevice-mask operation
//!   ([`MethodHeader::SubdeviceMask`]). This module decodes them as the header says; whether
//!   the channel skips them is G8's decision (`opcode_and_tertiary_table`).
//!
//! # Guesses
//!
//! - `clc76f.h` also defines `NVC76F_GP_ENTRY0_OPERAND` (31:0) and `NVC76F_GP_ENTRY1_OPCODE`
//!   (7:0: `NOP`, `ILLEGAL`, `GP_CRC`, `PB_CRC`) over the same bits as the address. NVIDIA's
//!   `dev_pbdma.ref`, which says when they apply, is not in the reference folder; the usual
//!   reading is "when the length is zero" (a **guess**). [`GpEntry`] decodes only the address
//!   form and keeps a zero length as zero; Eden skips zero-length entries (`dma_pusher.cpp`,
//!   `if (header.size > 0)`). A trace with a zero-length entry would confirm either.

use thiserror::Error;

use crate::field::Field;

/// Bytes per GP entry (`NVC76F_GP_ENTRY__SIZE`).
pub const GP_ENTRY_BYTES: usize = 8;

/// Bytes per command word.
const WORD_BYTES: u64 = 4;

/// Entry word 0: `FETCH` (`NVC76F_GP_ENTRY0_FETCH`, 0:0).
pub const GP_ENTRY0_FETCH: Field = Field::new(0, 0);
/// `FETCH_CONDITIONAL` (`NVC76F_GP_ENTRY0_FETCH_CONDITIONAL`); 0 is unconditional.
pub const GP_ENTRY0_FETCH_CONDITIONAL: u32 = 1;
/// Entry word 0: address bits 31:2 (`NVC76F_GP_ENTRY0_GET`, 31:2). The field's value is the
/// address shifted right by [`Field::low`], so the address is the field in place.
pub const GP_ENTRY0_GET: Field = Field::new(31, 2);
/// Entry word 1: address bits 39:32 (`NVC76F_GP_ENTRY1_GET_HI`, 7:0).
pub const GP_ENTRY1_GET_HI: Field = Field::new(7, 0);
/// Where [`GP_ENTRY1_GET_HI`] sits in the address: bit 32 (Eden's `CommandListHeader::addr`
/// is bits 39:0 of the raw entry, entry 1 being its upper word).
pub const GP_ENTRY_GET_HI_SHIFT: u32 = 32;
/// Entry word 1: `LEVEL` (`NVC76F_GP_ENTRY1_LEVEL`, 9:9).
pub const GP_ENTRY1_LEVEL: Field = Field::new(9, 9);
/// `LEVEL_SUBROUTINE` (`NVC76F_GP_ENTRY1_LEVEL_SUBROUTINE`); 0 is main.
pub const GP_ENTRY1_LEVEL_SUBROUTINE: u32 = 1;
/// Entry word 1: length in words (`NVC76F_GP_ENTRY1_LENGTH`, 30:10; Eden's `size`, bits 62:42
/// of the raw entry).
pub const GP_ENTRY1_LENGTH: Field = Field::new(30, 10);
/// Entry word 1: `SYNC` (`NVC76F_GP_ENTRY1_SYNC`, 31:31; Eden's `sync`, bit 63).
pub const GP_ENTRY1_SYNC: Field = Field::new(31, 31);
/// `SYNC_WAIT` (`NVC76F_GP_ENTRY1_SYNC_WAIT`); 0 is proceed.
pub const GP_ENTRY1_SYNC_WAIT: u32 = 1;

/// Method header: dword method address (`NVC76F_DMA_METHOD_ADDRESS`, 11:0).
pub const METHOD_ADDRESS: Field = Field::new(11, 0);
/// Method header: subchannel (`NVC76F_DMA_METHOD_SUBCHANNEL`, 15:13).
pub const METHOD_SUBCHANNEL: Field = Field::new(15, 13);
/// Method header: data word count (`NVC76F_DMA_METHOD_COUNT`, 28:16).
pub const METHOD_COUNT: Field = Field::new(28, 16);
/// Immediate header: the data itself (`NVC76F_DMA_IMMD_DATA`, 28:16).
pub const IMMD_DATA: Field = Field::new(28, 16);
/// Method header: opcode (`NVC76F_DMA_SEC_OP`, 31:29).
pub const SEC_OP: Field = Field::new(31, 29);
/// Method header: tertiary opcode for opcodes 0 and 2 (`NVC76F_DMA_TERT_OP`, 17:16).
pub const TERT_OP: Field = Field::new(17, 16);
/// Old-format method header: dword method address (`NVC76F_DMA_METHOD_ADDRESS_OLD`, 12:2).
pub const METHOD_ADDRESS_OLD: Field = Field::new(12, 2);
/// Old-format method header: data word count (`NVC76F_DMA_METHOD_COUNT_OLD`, 28:18).
pub const METHOD_COUNT_OLD: Field = Field::new(28, 18);
/// Subdevice-mask operations: the mask (`NVC76F_DMA_SUBDEVICE_MASK`, 15:4).
pub const SUBDEVICE_MASK: Field = Field::new(15, 4);
/// Subdevice-mask operations: the 16-bit opcode (`NVC76F_DMA_SET_SUBDEVICE_MASK_OPCODE` and
/// its siblings, 31:16).
pub const SUBDEVICE_OPCODE: Field = Field::new(31, 16);

/// `NVC76F_DMA_SEC_OP_GRP0_USE_TERT`: the tertiary opcode decides.
pub const SEC_OP_GRP0_USE_TERT: u32 = 0;
/// `NVC76F_DMA_SEC_OP_INC_METHOD`.
pub const SEC_OP_INC_METHOD: u32 = 1;
/// `NVC76F_DMA_SEC_OP_GRP2_USE_TERT`: the tertiary opcode decides.
pub const SEC_OP_GRP2_USE_TERT: u32 = 2;
/// `NVC76F_DMA_SEC_OP_NON_INC_METHOD`.
pub const SEC_OP_NON_INC_METHOD: u32 = 3;
/// `NVC76F_DMA_SEC_OP_IMMD_DATA_METHOD`.
pub const SEC_OP_IMMD_DATA_METHOD: u32 = 4;
/// `NVC76F_DMA_SEC_OP_ONE_INC`.
pub const SEC_OP_ONE_INC: u32 = 5;
/// `NVC76F_DMA_SEC_OP_RESERVED6`.
pub const SEC_OP_RESERVED6: u32 = 6;
/// `NVC76F_DMA_SEC_OP_END_PB_SEGMENT`.
pub const SEC_OP_END_PB_SEGMENT: u32 = 7;

/// `NVC76F_DMA_TERT_OP_GRP0_INC_METHOD`: an old-format incrementing method.
pub const TERT_OP_GRP0_INC_METHOD: u32 = 0;
/// `NVC76F_DMA_TERT_OP_GRP2_NON_INC_METHOD`: an old-format non-incrementing method.
pub const TERT_OP_GRP2_NON_INC_METHOD: u32 = 0;
/// `NVC76F_DMA_SET_SUBDEVICE_MASK_OPCODE_VALUE` (also `TERT_OP_GRP0_SET_SUB_DEV_MASK`).
pub const SUBDEVICE_OPCODE_SET: u32 = 1;
/// `NVC76F_DMA_STORE_SUBDEVICE_MASK_OPCODE_VALUE` (also `TERT_OP_GRP0_STORE_SUB_DEV_MASK`).
pub const SUBDEVICE_OPCODE_STORE: u32 = 2;
/// `NVC76F_DMA_USE_SUBDEVICE_MASK_OPCODE_VALUE` (also `TERT_OP_GRP0_USE_SUB_DEV_MASK`).
pub const SUBDEVICE_OPCODE_USE: u32 = 3;

/// Whether the GPU fetches a GP entry's words unconditionally (`NVC76F_GP_ENTRY0_FETCH`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Fetch {
    /// `FETCH_UNCONDITIONAL`.
    #[default]
    Unconditional,
    /// `FETCH_CONDITIONAL`.
    Conditional,
}

/// Whether a GP entry is a main or a subroutine segment (`NVC76F_GP_ENTRY1_LEVEL`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Level {
    /// `LEVEL_MAIN`.
    #[default]
    Main,
    /// `LEVEL_SUBROUTINE`.
    Subroutine,
}

/// Whether the GPU waits before fetching a GP entry (`NVC76F_GP_ENTRY1_SYNC`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Sync {
    /// `SYNC_PROCEED`.
    #[default]
    Proceed,
    /// `SYNC_WAIT`.
    Wait,
}

/// One decoded GP entry: a run of `length` command words at GPU address `address`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct GpEntry {
    /// GPU virtual address of the first word: 40 bits, 4-byte aligned.
    pub address: u64,
    /// Number of 32-bit words (21 bits).
    pub length: u32,
    /// Main or subroutine segment.
    pub level: Level,
    /// Conditional or unconditional fetch.
    pub fetch: Fetch,
    /// Whether to wait before fetching.
    pub sync: Sync,
}

impl GpEntry {
    /// Decodes an entry from its two words (word 0 at the lower address). Total: unused bits
    /// (entry 0 bit 1, entry 1 bit 8) are ignored.
    pub fn from_words(words: [u32; 2]) -> Self {
        let [entry0, entry1] = words;
        let low = u64::from(GP_ENTRY0_GET.get(entry0)) << GP_ENTRY0_GET.low();
        let high = u64::from(GP_ENTRY1_GET_HI.get(entry1)) << GP_ENTRY_GET_HI_SHIFT;
        Self {
            address: high | low,
            length: GP_ENTRY1_LENGTH.get(entry1),
            level: if GP_ENTRY1_LEVEL.get(entry1) == GP_ENTRY1_LEVEL_SUBROUTINE {
                Level::Subroutine
            } else {
                Level::Main
            },
            fetch: if GP_ENTRY0_FETCH.get(entry0) == GP_ENTRY0_FETCH_CONDITIONAL {
                Fetch::Conditional
            } else {
                Fetch::Unconditional
            },
            sync: if GP_ENTRY1_SYNC.get(entry1) == GP_ENTRY1_SYNC_WAIT {
                Sync::Wait
            } else {
                Sync::Proceed
            },
        }
    }

    /// Decodes an entry from its 8 bytes read as one little-endian `u64` (word 0 in the low
    /// half), the form gsv copies out of a submit request.
    pub fn from_raw(raw: u64) -> Self {
        Self::from_words(split_raw(raw))
    }

    /// Encodes the entry as its two words, with every unused bit zero.
    pub fn to_words(&self) -> Result<[u32; 2], EncodeError> {
        // The whole address must be a field-in-place GET plus GET_HI: 4-byte aligned, 40 bits.
        let low_mask = u64::from(GP_ENTRY0_GET.max()) << GP_ENTRY0_GET.low();
        let low_bits = u64::from(u32::MAX);
        if self.address & low_bits & !low_mask != 0 {
            return Err(EncodeError::MisalignedAddress {
                address: self.address,
            });
        }
        let address_bits = GP_ENTRY_GET_HI_SHIFT + GP_ENTRY1_GET_HI.width();
        let high = self.address >> GP_ENTRY_GET_HI_SHIFT;
        let high = u32::try_from(high)
            .ok()
            .filter(|&high| GP_ENTRY1_GET_HI.fits(high))
            .ok_or(EncodeError::FieldOverflow {
                field: "GP entry address",
                value: self.address,
                bits: address_bits,
            })?;
        let get =
            u32::try_from((self.address & low_mask) >> GP_ENTRY0_GET.low()).map_err(|_| {
                EncodeError::FieldOverflow {
                    field: "GP entry address",
                    value: self.address,
                    bits: address_bits,
                }
            })?;

        let fetch = match self.fetch {
            Fetch::Unconditional => 0,
            Fetch::Conditional => GP_ENTRY0_FETCH_CONDITIONAL,
        };
        let level = match self.level {
            Level::Main => 0,
            Level::Subroutine => GP_ENTRY1_LEVEL_SUBROUTINE,
        };
        let sync = match self.sync {
            Sync::Proceed => 0,
            Sync::Wait => GP_ENTRY1_SYNC_WAIT,
        };

        let entry0 = put(GP_ENTRY0_GET, "GP entry address", 0, get)?;
        let entry0 = put(GP_ENTRY0_FETCH, "GP entry fetch", entry0, fetch)?;
        let entry1 = put(GP_ENTRY1_GET_HI, "GP entry address", 0, high)?;
        let entry1 = put(GP_ENTRY1_LENGTH, "GP entry length", entry1, self.length)?;
        let entry1 = put(GP_ENTRY1_LEVEL, "GP entry level", entry1, level)?;
        let entry1 = put(GP_ENTRY1_SYNC, "GP entry sync", entry1, sync)?;
        Ok([entry0, entry1])
    }

    /// Encodes the entry as one little-endian `u64`, the inverse of [`GpEntry::from_raw`].
    pub fn to_raw(&self) -> Result<u64, EncodeError> {
        let [entry0, entry1] = self.to_words()?;
        Ok(u64::from(entry0) | (u64::from(entry1) << u32::BITS))
    }

    /// The entry's extent in bytes (`length` words of 4 bytes). Cannot overflow: the length
    /// is at most 21 bits.
    pub fn byte_len(&self) -> u64 {
        u64::from(self.length) * WORD_BYTES
    }
}

/// Splits a raw little-endian entry into its two words.
fn split_raw(raw: u64) -> [u32; 2] {
    let b = raw.to_le_bytes();
    [
        u32::from_le_bytes([b[0], b[1], b[2], b[3]]),
        u32::from_le_bytes([b[4], b[5], b[6], b[7]]),
    ]
}

/// How a method header's data words map to methods.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MethodMode {
    /// Each data word goes to the next method (`SEC_OP_INC_METHOD`; old format
    /// `TERT_OP_GRP0_INC_METHOD`).
    Incrementing,
    /// Every data word goes to the same method (`SEC_OP_NON_INC_METHOD`; old format
    /// `TERT_OP_GRP2_NON_INC_METHOD`).
    NonIncrementing,
    /// The first data word goes to the method, the rest to the next one (`SEC_OP_ONE_INC`).
    /// The old format has no such mode.
    IncrementOnce,
}

/// A subdevice-mask operation (opcode 0 with tertiary opcode 1–3). Starquake drives one GPU,
/// so the channel treats these as no-ops (spec, "The method stream").
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SubdeviceMask {
    /// `SET_SUBDEVICE_MASK` with its 12-bit mask.
    Set {
        /// The mask (`NVC76F_DMA_SET_SUBDEVICE_MASK_VALUE`, 15:4).
        mask: u16,
    },
    /// `STORE_SUBDEVICE_MASK` with its 12-bit mask.
    Store {
        /// The mask (`NVC76F_DMA_STORE_SUBDEVICE_MASK_VALUE`, 15:4).
        mask: u16,
    },
    /// `USE_SUBDEVICE_MASK`.
    Use,
}

/// One decoded method header.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MethodHeader {
    /// `count` data words follow, for `method` and on (by `mode`) on `subchannel`
    /// (opcodes 1, 3 and 5).
    Methods {
        /// How the data words map to methods.
        mode: MethodMode,
        /// Subchannel, 0–7.
        subchannel: u8,
        /// Dword method address, 12 bits.
        method: u16,
        /// Data words that follow, 13 bits.
        count: u16,
    },
    /// One write of `data` to `method`, carried in the header itself (opcode 4).
    Immediate {
        /// Subchannel, 0–7.
        subchannel: u8,
        /// Dword method address, 12 bits.
        method: u16,
        /// The data, 13 bits.
        data: u16,
    },
    /// An old-format method header (opcode 0 or 2 with tertiary opcode 0): an 11-bit method and
    /// an 11-bit count. `mode` is never [`MethodMode::IncrementOnce`]. The all-zero word
    /// (`NVC76F_DMA_NOP`) decodes as one of these with a count of 0.
    Legacy {
        /// [`MethodMode::Incrementing`] (opcode 0) or [`MethodMode::NonIncrementing`]
        /// (opcode 2).
        mode: MethodMode,
        /// Subchannel, 0–7.
        subchannel: u8,
        /// Dword method address, 11 bits.
        method: u16,
        /// Data words that follow, 11 bits.
        count: u16,
    },
    /// A subdevice-mask operation; no data words follow.
    SubdeviceMask(SubdeviceMask),
    /// End of the push-buffer segment (opcode 7); the rest of the word is ignored.
    EndSegment,
}

/// A word that is not a method header `clc76f.h` defines.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum DecodeError {
    /// Opcode 6 (`NVC76F_DMA_SEC_OP_RESERVED6`).
    #[error("method header {word:#010x} uses the reserved opcode 6")]
    ReservedOpcode {
        /// The header word.
        word: u32,
    },
    /// Opcode 0 with a tertiary opcode other than 0, whose bits 31:16 are not one of the three
    /// subdevice-mask opcodes.
    #[error("method header {word:#010x} is not a defined subdevice-mask operation")]
    UndefinedSubdeviceOp {
        /// The header word.
        word: u32,
    },
    /// Opcode 2 with a tertiary opcode other than 0 (only `GRP2_NON_INC_METHOD` is defined).
    #[error("method header {word:#010x} has an undefined tertiary opcode for opcode 2")]
    UndefinedGroup2Op {
        /// The header word.
        word: u32,
    },
}

/// A value the encoders cannot lay out.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum EncodeError {
    /// A value does not fit its field.
    #[error("{field} {value:#x} does not fit in {bits} bits")]
    FieldOverflow {
        /// What the value is.
        field: &'static str,
        /// The value.
        value: u64,
        /// The field's width.
        bits: u32,
    },
    /// A GP entry address with bit 0 or 1 set.
    #[error("GP entry address {address:#x} is not 4-byte aligned")]
    MisalignedAddress {
        /// The address.
        address: u64,
    },
    /// [`MethodHeader::Legacy`] with [`MethodMode::IncrementOnce`].
    #[error("an old-format method header has no increment-once mode")]
    LegacyIncrementOnce,
}

/// Writes `value` into `field` of `word`, or names the field that overflowed.
fn put(field: Field, name: &'static str, word: u32, value: u32) -> Result<u32, EncodeError> {
    field.insert(word, value).ok_or(EncodeError::FieldOverflow {
        field: name,
        value: u64::from(value),
        bits: field.width(),
    })
}

/// Narrows a decoded field to `u8`. Fields read through this are at most 3 bits wide, so the
/// fallback is unreachable; it exists so no decode path can panic.
fn narrow8(value: u32) -> u8 {
    u8::try_from(value).unwrap_or(u8::MAX)
}

/// Narrows a decoded field to `u16`. Fields read through this are at most 13 bits wide.
fn narrow16(value: u32) -> u16 {
    u16::try_from(value).unwrap_or(u16::MAX)
}

/// Decodes one method header word.
pub fn decode_header(word: u32) -> Result<MethodHeader, DecodeError> {
    let subchannel = narrow8(METHOD_SUBCHANNEL.get(word));
    let methods = |mode| MethodHeader::Methods {
        mode,
        subchannel,
        method: narrow16(METHOD_ADDRESS.get(word)),
        count: narrow16(METHOD_COUNT.get(word)),
    };
    let legacy = |mode| MethodHeader::Legacy {
        mode,
        subchannel,
        method: narrow16(METHOD_ADDRESS_OLD.get(word)),
        count: narrow16(METHOD_COUNT_OLD.get(word)),
    };
    match SEC_OP.get(word) {
        SEC_OP_GRP0_USE_TERT => {
            if TERT_OP.get(word) == TERT_OP_GRP0_INC_METHOD {
                return Ok(legacy(MethodMode::Incrementing));
            }
            let mask = narrow16(SUBDEVICE_MASK.get(word));
            match SUBDEVICE_OPCODE.get(word) {
                SUBDEVICE_OPCODE_SET => {
                    Ok(MethodHeader::SubdeviceMask(SubdeviceMask::Set { mask }))
                }
                SUBDEVICE_OPCODE_STORE => {
                    Ok(MethodHeader::SubdeviceMask(SubdeviceMask::Store { mask }))
                }
                SUBDEVICE_OPCODE_USE => Ok(MethodHeader::SubdeviceMask(SubdeviceMask::Use)),
                _ => Err(DecodeError::UndefinedSubdeviceOp { word }),
            }
        }
        SEC_OP_INC_METHOD => Ok(methods(MethodMode::Incrementing)),
        SEC_OP_GRP2_USE_TERT => {
            if TERT_OP.get(word) == TERT_OP_GRP2_NON_INC_METHOD {
                Ok(legacy(MethodMode::NonIncrementing))
            } else {
                Err(DecodeError::UndefinedGroup2Op { word })
            }
        }
        SEC_OP_NON_INC_METHOD => Ok(methods(MethodMode::NonIncrementing)),
        SEC_OP_IMMD_DATA_METHOD => Ok(MethodHeader::Immediate {
            subchannel,
            method: narrow16(METHOD_ADDRESS.get(word)),
            data: narrow16(IMMD_DATA.get(word)),
        }),
        SEC_OP_ONE_INC => Ok(methods(MethodMode::IncrementOnce)),
        SEC_OP_END_PB_SEGMENT => Ok(MethodHeader::EndSegment),
        // SEC_OP_RESERVED6, the only value a 3-bit field has left.
        _ => Err(DecodeError::ReservedOpcode { word }),
    }
}

/// Encodes one method header, with every bit the header ignores zero.
pub fn encode_header(header: &MethodHeader) -> Result<u32, EncodeError> {
    match *header {
        MethodHeader::Methods {
            mode,
            subchannel,
            method,
            count,
        } => {
            let op = match mode {
                MethodMode::Incrementing => SEC_OP_INC_METHOD,
                MethodMode::NonIncrementing => SEC_OP_NON_INC_METHOD,
                MethodMode::IncrementOnce => SEC_OP_ONE_INC,
            };
            let word = put(SEC_OP, "opcode", 0, op)?;
            let word = put(METHOD_SUBCHANNEL, "subchannel", word, subchannel.into())?;
            let word = put(METHOD_ADDRESS, "method", word, method.into())?;
            put(METHOD_COUNT, "method count", word, count.into())
        }
        MethodHeader::Immediate {
            subchannel,
            method,
            data,
        } => {
            let word = put(SEC_OP, "opcode", 0, SEC_OP_IMMD_DATA_METHOD)?;
            let word = put(METHOD_SUBCHANNEL, "subchannel", word, subchannel.into())?;
            let word = put(METHOD_ADDRESS, "method", word, method.into())?;
            put(IMMD_DATA, "immediate data", word, data.into())
        }
        MethodHeader::Legacy {
            mode,
            subchannel,
            method,
            count,
        } => {
            let (op, tert) = match mode {
                MethodMode::Incrementing => (SEC_OP_GRP0_USE_TERT, TERT_OP_GRP0_INC_METHOD),
                MethodMode::NonIncrementing => (SEC_OP_GRP2_USE_TERT, TERT_OP_GRP2_NON_INC_METHOD),
                MethodMode::IncrementOnce => return Err(EncodeError::LegacyIncrementOnce),
            };
            let word = put(SEC_OP, "opcode", 0, op)?;
            let word = put(TERT_OP, "tertiary opcode", word, tert)?;
            let word = put(METHOD_SUBCHANNEL, "subchannel", word, subchannel.into())?;
            let word = put(METHOD_ADDRESS_OLD, "old-format method", word, method.into())?;
            put(
                METHOD_COUNT_OLD,
                "old-format method count",
                word,
                count.into(),
            )
        }
        MethodHeader::SubdeviceMask(op) => {
            let (opcode, mask) = match op {
                SubdeviceMask::Set { mask } => (SUBDEVICE_OPCODE_SET, mask),
                SubdeviceMask::Store { mask } => (SUBDEVICE_OPCODE_STORE, mask),
                SubdeviceMask::Use => (SUBDEVICE_OPCODE_USE, 0),
            };
            let word = put(SUBDEVICE_OPCODE, "subdevice opcode", 0, opcode)?;
            put(SUBDEVICE_MASK, "subdevice mask", word, mask.into())
        }
        MethodHeader::EndSegment => put(SEC_OP, "opcode", 0, SEC_OP_END_PB_SEGMENT),
    }
}

/// Builds a stream of command words: headers through [`encode_header`], then their data.
///
/// For tests and capture fixtures (spec, "Fixtures and CI"): the shapes are those of Mesa's
/// `nv_push.h`, which takes byte addresses (`mthd >> 2`) where this takes dword methods. A
/// failed call appends nothing.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PushBuffer {
    words: Vec<u32>,
}

impl PushBuffer {
    /// An empty stream.
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends a method header for `data.len()` words, then the words (`NVC0_FIFO_PKHDR_SQ`,
    /// `_0I` and `_1I` by `mode`).
    pub fn methods(
        &mut self,
        mode: MethodMode,
        subchannel: u8,
        method: u16,
        data: &[u32],
    ) -> Result<&mut Self, EncodeError> {
        let count = u16::try_from(data.len())
            .ok()
            .filter(|&count| METHOD_COUNT.fits(count.into()))
            .ok_or(EncodeError::FieldOverflow {
                field: "method count",
                value: u64::try_from(data.len()).unwrap_or(u64::MAX),
                bits: METHOD_COUNT.width(),
            })?;
        self.header(&MethodHeader::Methods {
            mode,
            subchannel,
            method,
            count,
        })?;
        self.words.extend_from_slice(data);
        Ok(self)
    }

    /// Appends an immediate header (`NVC0_FIFO_PKHDR_IL`).
    pub fn immediate(
        &mut self,
        subchannel: u8,
        method: u16,
        data: u16,
    ) -> Result<&mut Self, EncodeError> {
        self.header(&MethodHeader::Immediate {
            subchannel,
            method,
            data,
        })
    }

    /// Appends any header, without data; follow it with [`PushBuffer::data`] for its words.
    /// This is how a test builds a method whose data continues in a later entry.
    pub fn header(&mut self, header: &MethodHeader) -> Result<&mut Self, EncodeError> {
        self.words.push(encode_header(header)?);
        Ok(self)
    }

    /// Appends raw words (data words, or deliberately malformed ones).
    pub fn data(&mut self, words: &[u32]) -> &mut Self {
        self.words.extend_from_slice(words);
        self
    }

    /// The words so far.
    pub fn words(&self) -> &[u32] {
        &self.words
    }

    /// The finished stream.
    pub fn into_words(self) -> Vec<u32> {
        self.words
    }
}

#[cfg(test)]
mod tests;
