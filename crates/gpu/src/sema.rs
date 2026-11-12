//! Semaphore operations: acquire predicates, release forms and reductions, decoded from the
//! words that request them (gpu-protocol spec, slice G2; "Semaphore acquires", "Releases and
//! completion").
//!
//! Four words carry a semaphore operation, in three reduction encodings:
//!
//! | Word | Source | Reduction encoding |
//! |---|---|---|
//! | `SEMAPHORED` (host method 0x7, byte 0x1C) | `classes/clc76f.h` | host: 30:27, `MIN` = 0 … `DEC` = 7; format bit 31, `SIGNED` = 0 |
//! | `SEM_EXECUTE` (host method 0x1B, byte 0x6C) | `classes/clc76f.h` | host, as above |
//! | `MME_DMA_REDUCTION` (3D method 0x15A, byte 0x568) | `classes/clc797.h` | `RED_ADD` = 0 … `RED_XOR` = 7 in 2:0; format 5:4, `UNSIGNED` = 0 |
//! | A QMD 3.0 release slot's second word | `classes/clc7c0qmd.h` (`NVC7C0_QMDV03_00_RELEASE0_*`) | as `MME_DMA_REDUCTION`, op 22:20, format 25:24, enable 26 |
//!
//! `classes/` is `mesa-nouveau/src/nouveau/headers/nvidia/classes/` @ 52238762 (Copyright (c)
//! 2020, NVIDIA CORPORATION, MIT licence). The QMD positions are those of `RELEASE0` (bits
//! 822:820, 825:824 and 826, in word 25) taken relative to the slot's second word; `RELEASE1`
//! (950:948, word 29) and `RELEASE2` (1718:1716, word 53) sit at the same positions in theirs.
//! Note the two encodings order the operations differently and invert the format bit.
//!
//! The acquire conditions are those of `eden-nx2/src/video_core/engines/puller.cpp` @ 50fa86bb
//! (`PendingAcquire::Matches`, `ProcessSemaphoreExecute`; GPL-3.0-or-later), as the spec's
//! acquire table lists them. The reduction arithmetic is that of
//! `eden-nx2/src/video_core/engines/kepler_compute.cpp` (`ProcessNativeReportSemaphore`),
//! which uses the `RED_*` order: `INC` wraps to 0 once the old value reaches the payload, and
//! `DEC` reloads the payload from 0 or from above it.
//!
//! What a channel does with a decoded operation (parking on an acquire, performing a release
//! after host completion) belongs to the channel (G8) and the back's fence path (G15).
//!
//! # Divergences from Eden-NX2
//!
//! Each is pinned by a test in `sema/tests.rs`:
//!
//! - **`SEMAPHORED` operation width.** Eden masks the operation with 0xF
//!   (`ProcessSemaphoreTriggerMethod`); the header's field is 4:0. A word whose bits 4:0 are
//!   not one of the five one-hot values is an error here, so 0x11 is not an acquire
//!   (`semaphored_operation_is_five_bits`).
//! - **`SEMAPHORED` reductions.** Eden has no reduction case (operation 0x10 masks to 0 and is
//!   logged as invalid); here it decodes to [`HostSemaphore::Reduce`]
//!   (`semaphored_reduction_fields`).
//! - **`SEM_EXECUTE` operations 6 and 7.** Eden's `ProcessSemaphoreExecute` treats every
//!   operation it does not name as an acquire already met. Here 6 is a reduction and 7 is an
//!   error (`sem_execute_operation_table`).
//! - **`SEM_EXECUTE` release form.** Eden writes 4 bytes for every `SEM_EXECUTE` release; here
//!   `PAYLOAD_SIZE` and `RELEASE_TIMESTAMP` are decoded into the [`ReleaseForm`]
//!   (`sem_execute_release_fields`).
//! - **Signed min and max.** Eden's reductions compare unsigned whatever the format; here the
//!   signed format compares signed (`signed_format_changes_min_and_max_only`).
//!
//! # Guesses
//!
//! - Under the signed format, `INC` and `DEC` still compare unsigned, as Eden does; no
//!   reference defines a signed `INC` or `DEC`. A trace of a signed `INC` would settle it.
//! - A `SEMAPHORED` reduction is 32 bits wide because its payload register (`SEMAPHOREC`,
//!   31:0) is; `RELEASE_SIZE` is ignored for it.
//! - [`AcquireOp::is_met`] compares 32-bit values, as the spec's table and Eden do. A
//!   `SEM_EXECUTE` acquire with a 64-bit `PAYLOAD_SIZE` decodes with
//!   [`PayloadSize::Eight`]; how it compares is not in any reference and is left to G8.

use std::cmp::Ordering;
use std::ops::{BitAnd, BitOr, BitXor};

use thiserror::Error;

use crate::field::Field;

/// `SEMAPHORED`: the operation (`NVC76F_SEMAPHORED_OPERATION`, 4:0).
pub const SEMAPHORED_OPERATION: Field = Field::new(4, 0);
/// `NVC76F_SEMAPHORED_OPERATION_ACQUIRE`.
pub const SEMAPHORED_OPERATION_ACQUIRE: u32 = 0x1;
/// `NVC76F_SEMAPHORED_OPERATION_RELEASE`.
pub const SEMAPHORED_OPERATION_RELEASE: u32 = 0x2;
/// `NVC76F_SEMAPHORED_OPERATION_ACQ_GEQ`.
pub const SEMAPHORED_OPERATION_ACQ_GEQ: u32 = 0x4;
/// `NVC76F_SEMAPHORED_OPERATION_ACQ_AND`.
pub const SEMAPHORED_OPERATION_ACQ_AND: u32 = 0x8;
/// `NVC76F_SEMAPHORED_OPERATION_REDUCTION`.
pub const SEMAPHORED_OPERATION_REDUCTION: u32 = 0x10;
/// `SEMAPHORED`: `ACQUIRE_SWITCH` (12:12; 1 = enabled).
pub const SEMAPHORED_ACQUIRE_SWITCH: Field = Field::new(12, 12);
/// `SEMAPHORED`: `RELEASE_WFI` (20:20).
pub const SEMAPHORED_RELEASE_WFI: Field = Field::new(20, 20);
/// `NVC76F_SEMAPHORED_RELEASE_WFI_DIS`: note 0 (`_EN`) is the default, wait for idle.
pub const SEMAPHORED_RELEASE_WFI_DIS: u32 = 1;
/// `SEMAPHORED`: `RELEASE_SIZE` (24:24).
pub const SEMAPHORED_RELEASE_SIZE: Field = Field::new(24, 24);
/// `NVC76F_SEMAPHORED_RELEASE_SIZE_4BYTE`; 0 is `_16BYTE` (row 52).
pub const SEMAPHORED_RELEASE_SIZE_4BYTE: u32 = 1;
/// `SEMAPHORED`: the reduction (30:27), in the host encoding (`HOST_REDUCTION_*`).
pub const SEMAPHORED_REDUCTION: Field = Field::new(30, 27);
/// `SEMAPHORED`: the reduction format (31:31), in the host encoding (`HOST_FORMAT_*`).
pub const SEMAPHORED_FORMAT: Field = Field::new(31, 31);

/// `SEM_EXECUTE`: the operation (`NVC76F_SEM_EXECUTE_OPERATION`, 2:0).
pub const SEM_EXECUTE_OPERATION: Field = Field::new(2, 0);
/// `NVC76F_SEM_EXECUTE_OPERATION_ACQUIRE`.
pub const SEM_EXECUTE_OPERATION_ACQUIRE: u32 = 0;
/// `NVC76F_SEM_EXECUTE_OPERATION_RELEASE`.
pub const SEM_EXECUTE_OPERATION_RELEASE: u32 = 1;
/// `NVC76F_SEM_EXECUTE_OPERATION_ACQ_STRICT_GEQ`.
pub const SEM_EXECUTE_OPERATION_ACQ_STRICT_GEQ: u32 = 2;
/// `NVC76F_SEM_EXECUTE_OPERATION_ACQ_CIRC_GEQ`.
pub const SEM_EXECUTE_OPERATION_ACQ_CIRC_GEQ: u32 = 3;
/// `NVC76F_SEM_EXECUTE_OPERATION_ACQ_AND`.
pub const SEM_EXECUTE_OPERATION_ACQ_AND: u32 = 4;
/// `NVC76F_SEM_EXECUTE_OPERATION_ACQ_NOR`.
pub const SEM_EXECUTE_OPERATION_ACQ_NOR: u32 = 5;
/// `NVC76F_SEM_EXECUTE_OPERATION_REDUCTION`.
pub const SEM_EXECUTE_OPERATION_REDUCTION: u32 = 6;
/// `SEM_EXECUTE`: `ACQUIRE_SWITCH_TSG` (12:12; 1 = enabled).
pub const SEM_EXECUTE_ACQUIRE_SWITCH_TSG: Field = Field::new(12, 12);
/// `SEM_EXECUTE`: `RELEASE_WFI` (20:20).
pub const SEM_EXECUTE_RELEASE_WFI: Field = Field::new(20, 20);
/// `NVC76F_SEM_EXECUTE_RELEASE_WFI_EN`: here 0 (`_DIS`) is the default, the opposite of
/// `SEMAPHORED`.
pub const SEM_EXECUTE_RELEASE_WFI_EN: u32 = 1;
/// `SEM_EXECUTE`: `PAYLOAD_SIZE` (24:24).
pub const SEM_EXECUTE_PAYLOAD_SIZE: Field = Field::new(24, 24);
/// `NVC76F_SEM_EXECUTE_PAYLOAD_SIZE_64BIT`; 0 is `_32BIT`.
pub const SEM_EXECUTE_PAYLOAD_SIZE_64BIT: u32 = 1;
/// `SEM_EXECUTE`: `RELEASE_TIMESTAMP` (25:25).
pub const SEM_EXECUTE_RELEASE_TIMESTAMP: Field = Field::new(25, 25);
/// `NVC76F_SEM_EXECUTE_RELEASE_TIMESTAMP_EN`; 0 is `_DIS`.
pub const SEM_EXECUTE_RELEASE_TIMESTAMP_EN: u32 = 1;
/// `SEM_EXECUTE`: the reduction (30:27), in the host encoding (`HOST_REDUCTION_*`).
pub const SEM_EXECUTE_REDUCTION: Field = Field::new(30, 27);
/// `SEM_EXECUTE`: the reduction format (31:31), in the host encoding (`HOST_FORMAT_*`).
pub const SEM_EXECUTE_REDUCTION_FORMAT: Field = Field::new(31, 31);

/// Host reduction `MIN` (`NVC76F_SEMAPHORED_REDUCTION_MIN`, `NVC76F_SEM_EXECUTE_REDUCTION_IMIN`).
pub const HOST_REDUCTION_MIN: u32 = 0;
/// Host reduction `MAX` (`_MAX`, `_IMAX`).
pub const HOST_REDUCTION_MAX: u32 = 1;
/// Host reduction `XOR` (`_XOR`, `_IXOR`).
pub const HOST_REDUCTION_XOR: u32 = 2;
/// Host reduction `AND` (`_AND`, `_IAND`).
pub const HOST_REDUCTION_AND: u32 = 3;
/// Host reduction `OR` (`_OR`, `_IOR`).
pub const HOST_REDUCTION_OR: u32 = 4;
/// Host reduction `ADD` (`_ADD`, `_IADD`).
pub const HOST_REDUCTION_ADD: u32 = 5;
/// Host reduction `INC` (`_INC`).
pub const HOST_REDUCTION_INC: u32 = 6;
/// Host reduction `DEC` (`_DEC`).
pub const HOST_REDUCTION_DEC: u32 = 7;
/// Host format `SIGNED` (`NVC76F_SEMAPHORED_FORMAT_SIGNED`, `_SEM_EXECUTE_REDUCTION_FORMAT_SIGNED`).
pub const HOST_FORMAT_SIGNED: u32 = 0;
/// Host format `UNSIGNED`.
pub const HOST_FORMAT_UNSIGNED: u32 = 1;

/// `MME_DMA_REDUCTION`: the operation (`NVC797_MME_DMA_REDUCTION_REDUCTION_OP`, 2:0), in the
/// `RED_*` encoding.
pub const MME_DMA_REDUCTION_OP: Field = Field::new(2, 0);
/// `MME_DMA_REDUCTION`: the format (`_REDUCTION_FORMAT`, 5:4), in the `RED_FORMAT_*` encoding.
pub const MME_DMA_REDUCTION_FORMAT: Field = Field::new(5, 4);
/// `MME_DMA_REDUCTION`: the size (`_REDUCTION_SIZE`, 8:8).
pub const MME_DMA_REDUCTION_SIZE: Field = Field::new(8, 8);
/// `NVC797_MME_DMA_REDUCTION_REDUCTION_SIZE_EIGHT_BYTES`; 0 is `_FOUR_BYTES`.
pub const MME_DMA_REDUCTION_SIZE_EIGHT_BYTES: u32 = 1;

/// QMD 3.0 release slot, second word: the operation (`RELEASE0_REDUCTION_OP`, 822:820).
pub const QMD_RELEASE_REDUCTION_OP: Field = Field::new(22, 20);
/// QMD 3.0 release slot, second word: the format (`RELEASE0_REDUCTION_FORMAT`, 825:824).
pub const QMD_RELEASE_REDUCTION_FORMAT: Field = Field::new(25, 24);
/// QMD 3.0 release slot, second word: reduction enable (`RELEASE0_REDUCTION_ENABLE`, 826).
pub const QMD_RELEASE_REDUCTION_ENABLE: Field = Field::new(26, 26);
/// `RELEASE0_REDUCTION_ENABLE_TRUE`.
pub const QMD_RELEASE_REDUCTION_ENABLE_TRUE: u32 = 1;

/// `RED_ADD` (MME and QMD encoding).
pub const RED_ADD: u32 = 0;
/// `RED_MIN`.
pub const RED_MIN: u32 = 1;
/// `RED_MAX`.
pub const RED_MAX: u32 = 2;
/// `RED_INC`.
pub const RED_INC: u32 = 3;
/// `RED_DEC`.
pub const RED_DEC: u32 = 4;
/// `RED_AND`.
pub const RED_AND: u32 = 5;
/// `RED_OR`.
pub const RED_OR: u32 = 6;
/// `RED_XOR`.
pub const RED_XOR: u32 = 7;
/// MME and QMD format `UNSIGNED` (`_REDUCTION_FORMAT_UNSIGNED`, `_UNSIGNED_32`).
pub const RED_FORMAT_UNSIGNED: u32 = 0;
/// MME and QMD format `SIGNED`.
pub const RED_FORMAT_SIGNED: u32 = 1;

/// `SEMAPHORED`, as errors name it.
pub const SEMAPHORED: &str = "SEMAPHORED";
/// `SEM_EXECUTE`, as errors name it.
pub const SEM_EXECUTE: &str = "SEM_EXECUTE";
/// `MME_DMA_REDUCTION`, as errors name it.
pub const MME_DMA_REDUCTION: &str = "MME_DMA_REDUCTION";
/// A QMD release slot, as errors name it.
pub const QMD_RELEASE: &str = "QMD release";

/// The condition an acquire waits for, on the 32-bit `value` at the semaphore address and the
/// method's `payload`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AcquireOp {
    /// `value == payload` (`SEMAPHORED` `ACQUIRE`, `SEM_EXECUTE` `ACQUIRE`).
    Equal,
    /// `value >= payload`, unsigned (`SEMAPHORED` `ACQ_GEQ`, `SEM_EXECUTE` `ACQ_STRICT_GEQ`).
    GreaterOrEqual,
    /// `(value - payload) as i32 >= 0`, so it holds across the 32-bit wrap
    /// (`SEM_EXECUTE` `ACQ_CIRC_GEQ`).
    CircularGreaterOrEqual,
    /// `value & payload != 0` (`SEMAPHORED` `ACQ_AND`, `SEM_EXECUTE` `ACQ_AND`).
    And,
    /// `value & payload == 0` (`SEM_EXECUTE` `ACQ_NOR`).
    Nor,
}

impl AcquireOp {
    /// Whether an acquire with this operation and `payload` is met by `value`.
    pub const fn is_met(self, value: u32, payload: u32) -> bool {
        match self {
            Self::Equal => value == payload,
            Self::GreaterOrEqual => value >= payload,
            Self::CircularGreaterOrEqual => value.wrapping_sub(payload).cast_signed() >= 0,
            Self::And => value & payload != 0,
            Self::Nor => value & payload == 0,
        }
    }
}

/// The width of a semaphore payload.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PayloadSize {
    /// 32 bits.
    Four,
    /// 64 bits.
    Eight,
}

impl PayloadSize {
    /// The payload's width in bytes.
    pub const fn bytes(self) -> u64 {
        match self {
            Self::Four => 4,
            Self::Eight => 8,
        }
    }
}

/// What a release writes: the payload, then, with `timestamp`, a 64-bit timestamp.
///
/// With a timestamp the release is the 16-byte semaphore structure: Eden-NX2 writes the
/// payload zero-extended to 64 bits at offset 0 and the timestamp at offset 8
/// (`eden-nx2/src/video_core/query_cache/query_cache.h`, the `QueryType::Payload` case).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ReleaseForm {
    /// The payload's width.
    pub size: PayloadSize,
    /// Whether a timestamp follows the payload.
    pub timestamp: bool,
}

/// Offset of a release's timestamp from the semaphore address (see [`ReleaseForm`]).
pub const RELEASE_TIMESTAMP_OFFSET: u64 = 8;

impl ReleaseForm {
    /// The bytes the release writes: the payload, or 16 with a timestamp.
    pub const fn bytes(self) -> u64 {
        if self.timestamp {
            RELEASE_TIMESTAMP_OFFSET + PayloadSize::Eight.bytes()
        } else {
            self.size.bytes()
        }
    }
}

/// A reduction's operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ReductionOp {
    /// `old + payload`, wrapping.
    Add,
    /// The smaller of the two, by the format.
    Min,
    /// The larger of the two, by the format.
    Max,
    /// `0` if `old >= payload`, else `old + 1` (unsigned).
    Inc,
    /// `payload` if `old == 0 || old > payload`, else `old - 1` (unsigned).
    Dec,
    /// `old & payload`.
    And,
    /// `old | payload`.
    Or,
    /// `old ^ payload`.
    Xor,
}

/// How a reduction compares values.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ReductionFormat {
    /// Unsigned.
    Unsigned,
    /// Two's-complement signed.
    Signed,
}

/// An atomic read-modify-write of a semaphore with a payload.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Reduction {
    /// The operation.
    pub op: ReductionOp,
    /// How `Min` and `Max` compare.
    pub format: ReductionFormat,
}

impl Reduction {
    /// The new 32-bit value from the `old` value and the `payload`.
    pub fn apply_u32(self, old: u32, payload: u32) -> u32 {
        reduce(self, old, payload)
    }

    /// The new 64-bit value from the `old` value and the `payload`.
    pub fn apply_u64(self, old: u64, payload: u64) -> u64 {
        reduce(self, old, payload)
    }
}

/// An unsigned word a reduction works on.
trait Lane: Copy + Ord + BitAnd<Output = Self> + BitOr<Output = Self> + BitXor<Output = Self> {
    const ZERO: Self;
    const ONE: Self;
    fn wrapping_add(self, other: Self) -> Self;
    fn wrapping_sub(self, other: Self) -> Self;
    /// Compares the two as two's-complement signed values.
    fn signed_cmp(self, other: Self) -> Ordering;
}

impl Lane for u32 {
    const ZERO: Self = 0;
    const ONE: Self = 1;
    fn wrapping_add(self, other: Self) -> Self {
        self.wrapping_add(other)
    }
    fn wrapping_sub(self, other: Self) -> Self {
        self.wrapping_sub(other)
    }
    fn signed_cmp(self, other: Self) -> Ordering {
        self.cast_signed().cmp(&other.cast_signed())
    }
}

impl Lane for u64 {
    const ZERO: Self = 0;
    const ONE: Self = 1;
    fn wrapping_add(self, other: Self) -> Self {
        self.wrapping_add(other)
    }
    fn wrapping_sub(self, other: Self) -> Self {
        self.wrapping_sub(other)
    }
    fn signed_cmp(self, other: Self) -> Ordering {
        self.cast_signed().cmp(&other.cast_signed())
    }
}

/// The one implementation of every reduction, for both widths.
fn reduce<T: Lane>(reduction: Reduction, old: T, payload: T) -> T {
    let order = match reduction.format {
        ReductionFormat::Unsigned => old.cmp(&payload),
        ReductionFormat::Signed => old.signed_cmp(payload),
    };
    match reduction.op {
        ReductionOp::Add => old.wrapping_add(payload),
        ReductionOp::Min => {
            if order.is_le() {
                old
            } else {
                payload
            }
        }
        ReductionOp::Max => {
            if order.is_ge() {
                old
            } else {
                payload
            }
        }
        ReductionOp::Inc => {
            if old >= payload {
                T::ZERO
            } else {
                // old < payload <= MAX, so this cannot wrap.
                old.wrapping_add(T::ONE)
            }
        }
        ReductionOp::Dec => {
            if old == T::ZERO || old > payload {
                payload
            } else {
                old.wrapping_sub(T::ONE)
            }
        }
        ReductionOp::And => old & payload,
        ReductionOp::Or => old | payload,
        ReductionOp::Xor => old ^ payload,
    }
}

/// A decoded host semaphore operation (`SEMAPHORED` or `SEM_EXECUTE`). The address and payload
/// come from the channel's other host registers; this is the operation word only.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum HostSemaphore {
    /// Wait until [`AcquireOp::is_met`].
    Acquire {
        /// The condition.
        op: AcquireOp,
        /// The payload width (always [`PayloadSize::Four`] for `SEMAPHORED`).
        size: PayloadSize,
        /// `ACQUIRE_SWITCH` / `ACQUIRE_SWITCH_TSG`: the channel may be switched out while it
        /// waits.
        switch: bool,
    },
    /// Write the payload.
    Release {
        /// What is written.
        form: ReleaseForm,
        /// Wait for idle before releasing.
        wfi: bool,
    },
    /// Apply a reduction with the payload.
    Reduce {
        /// The reduction.
        reduction: Reduction,
        /// The payload width (always [`PayloadSize::Four`] for `SEMAPHORED`).
        size: PayloadSize,
        /// Wait for idle before reducing.
        wfi: bool,
    },
}

/// A decoded `MME_DMA_REDUCTION` word.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MmeDmaReduction {
    /// The reduction.
    pub reduction: Reduction,
    /// The operand width.
    pub size: PayloadSize,
}

/// A semaphore word with a value its header does not define.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum DecodeError {
    /// The operation field holds no defined operation.
    #[error("{word} operation {value:#x} is not defined")]
    UndefinedOperation {
        /// Which word (`SEMAPHORED`, `SEM_EXECUTE`).
        word: &'static str,
        /// The operation field.
        value: u32,
    },
    /// The reduction field holds no defined reduction.
    #[error("{word} reduction {value} is not defined")]
    UndefinedReduction {
        /// Which word.
        word: &'static str,
        /// The reduction field.
        value: u32,
    },
    /// The format field holds no defined format.
    #[error("{word} reduction format {value} is not defined")]
    UndefinedFormat {
        /// Which word (`MME_DMA_REDUCTION`, QMD release).
        word: &'static str,
        /// The format field.
        value: u32,
    },
}

/// An operation `SEMAPHORED` cannot express (`SEM_EXECUTE` expresses every one).
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum EncodeError {
    /// `SEMAPHORED` has no such operation or width.
    #[error("SEMAPHORED cannot express {what}")]
    NotInSemaphored {
        /// What it cannot express.
        what: &'static str,
    },
}

/// Decodes a host-encoded reduction (shared by `SEMAPHORED` and `SEM_EXECUTE`).
fn host_reduction(word: &'static str, op: u32, format: u32) -> Result<Reduction, DecodeError> {
    let op = match op {
        HOST_REDUCTION_MIN => ReductionOp::Min,
        HOST_REDUCTION_MAX => ReductionOp::Max,
        HOST_REDUCTION_XOR => ReductionOp::Xor,
        HOST_REDUCTION_AND => ReductionOp::And,
        HOST_REDUCTION_OR => ReductionOp::Or,
        HOST_REDUCTION_ADD => ReductionOp::Add,
        HOST_REDUCTION_INC => ReductionOp::Inc,
        HOST_REDUCTION_DEC => ReductionOp::Dec,
        value => return Err(DecodeError::UndefinedReduction { word, value }),
    };
    let format = if format == HOST_FORMAT_SIGNED {
        ReductionFormat::Signed
    } else {
        ReductionFormat::Unsigned
    };
    Ok(Reduction { op, format })
}

/// Encodes a reduction in the host encoding: (reduction, format) field values.
fn host_reduction_fields(reduction: Reduction) -> (u32, u32) {
    let op = match reduction.op {
        ReductionOp::Min => HOST_REDUCTION_MIN,
        ReductionOp::Max => HOST_REDUCTION_MAX,
        ReductionOp::Xor => HOST_REDUCTION_XOR,
        ReductionOp::And => HOST_REDUCTION_AND,
        ReductionOp::Or => HOST_REDUCTION_OR,
        ReductionOp::Add => HOST_REDUCTION_ADD,
        ReductionOp::Inc => HOST_REDUCTION_INC,
        ReductionOp::Dec => HOST_REDUCTION_DEC,
    };
    let format = match reduction.format {
        ReductionFormat::Signed => HOST_FORMAT_SIGNED,
        ReductionFormat::Unsigned => HOST_FORMAT_UNSIGNED,
    };
    (op, format)
}

/// Decodes a `RED_*`-encoded reduction (shared by `MME_DMA_REDUCTION` and QMD releases).
pub(crate) fn red_reduction(
    word: &'static str,
    op: u32,
    format: u32,
) -> Result<Reduction, DecodeError> {
    let op = match op {
        RED_ADD => ReductionOp::Add,
        RED_MIN => ReductionOp::Min,
        RED_MAX => ReductionOp::Max,
        RED_INC => ReductionOp::Inc,
        RED_DEC => ReductionOp::Dec,
        RED_AND => ReductionOp::And,
        RED_OR => ReductionOp::Or,
        RED_XOR => ReductionOp::Xor,
        // Both callers read a 3-bit field, so every value is defined; this keeps it total.
        value => return Err(DecodeError::UndefinedReduction { word, value }),
    };
    let format = match format {
        RED_FORMAT_UNSIGNED => ReductionFormat::Unsigned,
        RED_FORMAT_SIGNED => ReductionFormat::Signed,
        value => return Err(DecodeError::UndefinedFormat { word, value }),
    };
    Ok(Reduction { op, format })
}

/// Encodes a reduction in the `RED_*` encoding: (operation, format) field values.
pub(crate) fn red_reduction_fields(reduction: Reduction) -> (u32, u32) {
    let op = match reduction.op {
        ReductionOp::Add => RED_ADD,
        ReductionOp::Min => RED_MIN,
        ReductionOp::Max => RED_MAX,
        ReductionOp::Inc => RED_INC,
        ReductionOp::Dec => RED_DEC,
        ReductionOp::And => RED_AND,
        ReductionOp::Or => RED_OR,
        ReductionOp::Xor => RED_XOR,
    };
    let format = match reduction.format {
        ReductionFormat::Unsigned => RED_FORMAT_UNSIGNED,
        ReductionFormat::Signed => RED_FORMAT_SIGNED,
    };
    (op, format)
}

/// 1 for `true`, for one-bit flags.
fn flag(set: bool) -> u32 {
    u32::from(set)
}

/// Decodes a `SEMAPHORED` word.
pub fn decode_semaphored(word: u32) -> Result<HostSemaphore, DecodeError> {
    let wfi = SEMAPHORED_RELEASE_WFI.get(word) != SEMAPHORED_RELEASE_WFI_DIS;
    let acquire = |op| HostSemaphore::Acquire {
        op,
        size: PayloadSize::Four,
        switch: SEMAPHORED_ACQUIRE_SWITCH.is_set(word),
    };
    match SEMAPHORED_OPERATION.get(word) {
        SEMAPHORED_OPERATION_ACQUIRE => Ok(acquire(AcquireOp::Equal)),
        SEMAPHORED_OPERATION_ACQ_GEQ => Ok(acquire(AcquireOp::GreaterOrEqual)),
        SEMAPHORED_OPERATION_ACQ_AND => Ok(acquire(AcquireOp::And)),
        SEMAPHORED_OPERATION_RELEASE => Ok(HostSemaphore::Release {
            form: ReleaseForm {
                size: PayloadSize::Four,
                timestamp: SEMAPHORED_RELEASE_SIZE.get(word) != SEMAPHORED_RELEASE_SIZE_4BYTE,
            },
            wfi,
        }),
        SEMAPHORED_OPERATION_REDUCTION => Ok(HostSemaphore::Reduce {
            reduction: host_reduction(
                SEMAPHORED,
                SEMAPHORED_REDUCTION.get(word),
                SEMAPHORED_FORMAT.get(word),
            )?,
            size: PayloadSize::Four,
            wfi,
        }),
        value => Err(DecodeError::UndefinedOperation {
            word: SEMAPHORED,
            value,
        }),
    }
}

/// Encodes a `SEMAPHORED` word, with every bit the operation ignores zero.
pub fn encode_semaphored(op: &HostSemaphore) -> Result<u32, EncodeError> {
    let wide = EncodeError::NotInSemaphored {
        what: "a 64-bit payload",
    };
    let wfi_field = |wfi: bool| {
        if wfi { 0 } else { SEMAPHORED_RELEASE_WFI_DIS }
    };
    match *op {
        HostSemaphore::Acquire { op, size, switch } => {
            if size != PayloadSize::Four {
                return Err(wide);
            }
            let value = match op {
                AcquireOp::Equal => SEMAPHORED_OPERATION_ACQUIRE,
                AcquireOp::GreaterOrEqual => SEMAPHORED_OPERATION_ACQ_GEQ,
                AcquireOp::And => SEMAPHORED_OPERATION_ACQ_AND,
                AcquireOp::CircularGreaterOrEqual => {
                    return Err(EncodeError::NotInSemaphored {
                        what: "a circular greater-or-equal acquire",
                    });
                }
                AcquireOp::Nor => {
                    return Err(EncodeError::NotInSemaphored {
                        what: "a nor acquire",
                    });
                }
            };
            let word = SEMAPHORED_OPERATION.insert_truncated(0, value);
            Ok(SEMAPHORED_ACQUIRE_SWITCH.insert_truncated(word, flag(switch)))
        }
        HostSemaphore::Release { form, wfi } => {
            if form.size != PayloadSize::Four {
                return Err(wide);
            }
            let size = if form.timestamp {
                0
            } else {
                SEMAPHORED_RELEASE_SIZE_4BYTE
            };
            let word = SEMAPHORED_OPERATION.insert_truncated(0, SEMAPHORED_OPERATION_RELEASE);
            let word = SEMAPHORED_RELEASE_SIZE.insert_truncated(word, size);
            Ok(SEMAPHORED_RELEASE_WFI.insert_truncated(word, wfi_field(wfi)))
        }
        HostSemaphore::Reduce {
            reduction,
            size,
            wfi,
        } => {
            if size != PayloadSize::Four {
                return Err(wide);
            }
            let (red, format) = host_reduction_fields(reduction);
            let word = SEMAPHORED_OPERATION.insert_truncated(0, SEMAPHORED_OPERATION_REDUCTION);
            let word = SEMAPHORED_REDUCTION.insert_truncated(word, red);
            let word = SEMAPHORED_FORMAT.insert_truncated(word, format);
            Ok(SEMAPHORED_RELEASE_WFI.insert_truncated(word, wfi_field(wfi)))
        }
    }
}

/// Decodes a `SEM_EXECUTE` word.
pub fn decode_sem_execute(word: u32) -> Result<HostSemaphore, DecodeError> {
    let size = if SEM_EXECUTE_PAYLOAD_SIZE.get(word) == SEM_EXECUTE_PAYLOAD_SIZE_64BIT {
        PayloadSize::Eight
    } else {
        PayloadSize::Four
    };
    let wfi = SEM_EXECUTE_RELEASE_WFI.get(word) == SEM_EXECUTE_RELEASE_WFI_EN;
    let acquire = |op| HostSemaphore::Acquire {
        op,
        size,
        switch: SEM_EXECUTE_ACQUIRE_SWITCH_TSG.is_set(word),
    };
    match SEM_EXECUTE_OPERATION.get(word) {
        SEM_EXECUTE_OPERATION_ACQUIRE => Ok(acquire(AcquireOp::Equal)),
        SEM_EXECUTE_OPERATION_ACQ_STRICT_GEQ => Ok(acquire(AcquireOp::GreaterOrEqual)),
        SEM_EXECUTE_OPERATION_ACQ_CIRC_GEQ => Ok(acquire(AcquireOp::CircularGreaterOrEqual)),
        SEM_EXECUTE_OPERATION_ACQ_AND => Ok(acquire(AcquireOp::And)),
        SEM_EXECUTE_OPERATION_ACQ_NOR => Ok(acquire(AcquireOp::Nor)),
        SEM_EXECUTE_OPERATION_RELEASE => Ok(HostSemaphore::Release {
            form: ReleaseForm {
                size,
                timestamp: SEM_EXECUTE_RELEASE_TIMESTAMP.get(word)
                    == SEM_EXECUTE_RELEASE_TIMESTAMP_EN,
            },
            wfi,
        }),
        SEM_EXECUTE_OPERATION_REDUCTION => Ok(HostSemaphore::Reduce {
            reduction: host_reduction(
                SEM_EXECUTE,
                SEM_EXECUTE_REDUCTION.get(word),
                SEM_EXECUTE_REDUCTION_FORMAT.get(word),
            )?,
            size,
            wfi,
        }),
        value => Err(DecodeError::UndefinedOperation {
            word: SEM_EXECUTE,
            value,
        }),
    }
}

/// Encodes a `SEM_EXECUTE` word, with every bit the operation ignores zero. Every
/// [`HostSemaphore`] has a `SEM_EXECUTE` form.
pub fn encode_sem_execute(op: &HostSemaphore) -> u32 {
    let size_field = |size| match size {
        PayloadSize::Four => 0,
        PayloadSize::Eight => SEM_EXECUTE_PAYLOAD_SIZE_64BIT,
    };
    let wfi_field = |wfi: bool| {
        if wfi { SEM_EXECUTE_RELEASE_WFI_EN } else { 0 }
    };
    match *op {
        HostSemaphore::Acquire { op, size, switch } => {
            let value = match op {
                AcquireOp::Equal => SEM_EXECUTE_OPERATION_ACQUIRE,
                AcquireOp::GreaterOrEqual => SEM_EXECUTE_OPERATION_ACQ_STRICT_GEQ,
                AcquireOp::CircularGreaterOrEqual => SEM_EXECUTE_OPERATION_ACQ_CIRC_GEQ,
                AcquireOp::And => SEM_EXECUTE_OPERATION_ACQ_AND,
                AcquireOp::Nor => SEM_EXECUTE_OPERATION_ACQ_NOR,
            };
            let word = SEM_EXECUTE_OPERATION.insert_truncated(0, value);
            let word = SEM_EXECUTE_PAYLOAD_SIZE.insert_truncated(word, size_field(size));
            SEM_EXECUTE_ACQUIRE_SWITCH_TSG.insert_truncated(word, flag(switch))
        }
        HostSemaphore::Release { form, wfi } => {
            let timestamp = if form.timestamp {
                SEM_EXECUTE_RELEASE_TIMESTAMP_EN
            } else {
                0
            };
            let word = SEM_EXECUTE_OPERATION.insert_truncated(0, SEM_EXECUTE_OPERATION_RELEASE);
            let word = SEM_EXECUTE_PAYLOAD_SIZE.insert_truncated(word, size_field(form.size));
            let word = SEM_EXECUTE_RELEASE_TIMESTAMP.insert_truncated(word, timestamp);
            SEM_EXECUTE_RELEASE_WFI.insert_truncated(word, wfi_field(wfi))
        }
        HostSemaphore::Reduce {
            reduction,
            size,
            wfi,
        } => {
            let (red, format) = host_reduction_fields(reduction);
            let word = SEM_EXECUTE_OPERATION.insert_truncated(0, SEM_EXECUTE_OPERATION_REDUCTION);
            let word = SEM_EXECUTE_PAYLOAD_SIZE.insert_truncated(word, size_field(size));
            let word = SEM_EXECUTE_REDUCTION.insert_truncated(word, red);
            let word = SEM_EXECUTE_REDUCTION_FORMAT.insert_truncated(word, format);
            SEM_EXECUTE_RELEASE_WFI.insert_truncated(word, wfi_field(wfi))
        }
    }
}

/// Decodes an `MME_DMA_REDUCTION` word.
pub fn decode_mme_dma_reduction(word: u32) -> Result<MmeDmaReduction, DecodeError> {
    let reduction = red_reduction(
        MME_DMA_REDUCTION,
        MME_DMA_REDUCTION_OP.get(word),
        MME_DMA_REDUCTION_FORMAT.get(word),
    )?;
    let size = if MME_DMA_REDUCTION_SIZE.get(word) == MME_DMA_REDUCTION_SIZE_EIGHT_BYTES {
        PayloadSize::Eight
    } else {
        PayloadSize::Four
    };
    Ok(MmeDmaReduction { reduction, size })
}

/// Encodes an `MME_DMA_REDUCTION` word, with every other bit zero.
pub fn encode_mme_dma_reduction(reduction: &MmeDmaReduction) -> u32 {
    let (op, format) = red_reduction_fields(reduction.reduction);
    let size = match reduction.size {
        PayloadSize::Four => 0,
        PayloadSize::Eight => MME_DMA_REDUCTION_SIZE_EIGHT_BYTES,
    };
    let word = MME_DMA_REDUCTION_OP.insert_truncated(0, op);
    let word = MME_DMA_REDUCTION_FORMAT.insert_truncated(word, format);
    MME_DMA_REDUCTION_SIZE.insert_truncated(word, size)
}

/// Decodes the reduction of a QMD 3.0 release slot from the slot's second word: `None` when
/// reduction enable is clear (a plain release; the operation and format bits are then
/// ignored). The slot's own enable, membar, trap and structure fields are the QMD decoder's
/// (slice G12).
pub fn decode_qmd_release_reduction(flags: u32) -> Result<Option<Reduction>, DecodeError> {
    if QMD_RELEASE_REDUCTION_ENABLE.get(flags) != QMD_RELEASE_REDUCTION_ENABLE_TRUE {
        return Ok(None);
    }
    red_reduction(
        QMD_RELEASE,
        QMD_RELEASE_REDUCTION_OP.get(flags),
        QMD_RELEASE_REDUCTION_FORMAT.get(flags),
    )
    .map(Some)
}

/// Encodes the reduction fields of a QMD 3.0 release slot's second word, with every other bit
/// zero (`None` is a word with reduction enable clear).
pub fn encode_qmd_release_reduction(reduction: Option<Reduction>) -> u32 {
    let Some(reduction) = reduction else {
        return 0;
    };
    let (op, format) = red_reduction_fields(reduction);
    let word = QMD_RELEASE_REDUCTION_ENABLE.insert_truncated(0, QMD_RELEASE_REDUCTION_ENABLE_TRUE);
    let word = QMD_RELEASE_REDUCTION_OP.insert_truncated(word, op);
    QMD_RELEASE_REDUCTION_FORMAT.insert_truncated(word, format)
}

#[cfg(test)]
mod tests;
