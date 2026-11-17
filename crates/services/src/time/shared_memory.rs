//! The time shared memory `time:u` hands a program: its layout, the clock values it carries, and
//! what it holds once the clocks are set up.
//!
//! Ported from Eden @ cf654052 (GPL-3.0-or-later), paths relative to the reference folder:
//! `eden/src/core/hle/service/psc/time/shared_memory.h` and `.cpp` (`SharedMemoryStruct` with
//! the offsets its `static_assert`s pin, `LockFreeAtomicType`, `WriteToLockFreeAtomicType`) and
//! `common.h` (`SteadyClockTimePoint`, `SystemClockContext`, `ContinuousAdjustmentTimePoint`).
//! The reader is libnx (`libnx/nx/source/services/time.c`, ISC): `_timeReadSharedmemObj` reads
//! a value at `offset + 8 + (counter & 1) * size`, the steady clock at 0x0, the local (and user)
//! clock's context at 0x38 and the network clock's at 0x80.
//!
//! # A value and its counter
//!
//! Each value is a `LockFreeAtomicType<T>`: a `u32` counter, then two copies of `T` at `T`'s
//! alignment. A write increments the counter and stores the value in copy `counter % 2`, so
//! after the one write each value gets when the clocks are set up the counter is
//! [`COUNTER_AFTER_SETUP`] and the value is in copy 1 ([`image`]).
//!
//! That is the image this port keeps for the server's life: it writes the memory once and never
//! updates it. Eden does not stop there for two of the values. Its `TimeWorker`'s steady-clock
//! timer fires straight after set-up and every 5 minutes after that
//! (`glue/time/worker.cpp`, `ScheduleLoopingEvent(0ns,
//! standard_steady_clock_rtc_update_interval_minutes)`), and each time
//! `SetStandardSteadyClockBaseTime` rewrites `steady_time_points` and
//! `continuous_adjustment_time_points`; after the first, each has counter 2 and its value in
//! copy 0. So a guest under Eden never sees those two as counter 1 and copy 1. This divergence
//! is listed in the `time` module's docs ("Where this differs from Eden").

use serde::{Deserialize, Serialize};

use crate::shared_image::{blank, put};

/// The name [`put`] logs a misplaced field under.
const MEMORY: &str = "time";

/// The size of the time shared memory: `sizeof(SharedMemoryStruct)`
/// (`static_assert(sizeof(SharedMemoryStruct) == 0x1000)`), the size libnx maps
/// (`shmemLoadRemote(&g_timeSharedmem, shmem, 0x1000, Perm_R)`).
pub const SIZE: u64 = 0x1000;

/// A clock source: a UUID (`ClockSourceId = Common::UUID`, 16 bytes). Two steady time points are
/// comparable only when their sources match (`SteadyClockTimePoint::IdMatches`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClockSourceId(pub [u8; 16]);

/// A point on a steady clock (`SteadyClockTimePoint`: `s64 time_point; ClockSourceId
/// clock_source_id;`, 0x18 bytes).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SteadyClockTimePoint {
    /// The time point, in a unit the holder states.
    pub time_point: i64,
    /// The clock it is on.
    pub source_id: ClockSourceId,
}

/// A system clock's context (`SystemClockContext`: `s64 offset; SteadyClockTimePoint
/// steady_time_point;`, 0x20 bytes): its time is `offset` plus the steady clock's time in
/// seconds, while the steady clock is the one named.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SystemClockContext {
    /// Seconds added to the steady clock's.
    pub offset: i64,
    /// The steady clock's time point, in seconds, when the context was set.
    pub steady_time_point: SteadyClockTimePoint,
}

/// The steady clock's continuous adjustment (`ContinuousAdjustmentTimePoint`: five `s64`s and a
/// `ClockSourceId`, 0x38 bytes).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContinuousAdjustmentTimePoint {
    /// `rtc_offset`: the uptime, in nanoseconds, when the adjustment was set.
    pub rtc_offset: i64,
    /// `diff_scale`.
    pub diff_scale: i64,
    /// `shift_amount`.
    pub shift_amount: i64,
    /// `lower`.
    pub lower: i64,
    /// `upper`.
    pub upper: i64,
    /// The steady clock it adjusts.
    pub source_id: ClockSourceId,
}

/// Where each value lies (`SharedMemoryStruct`'s `offsetof` `static_assert`s).
pub mod offset {
    /// `steady_time_points`: the steady clock's base, a [`super::SteadyClockTimePoint`] in
    /// nanoseconds.
    pub const STEADY_TIME_POINTS: u64 = 0x0;
    /// `local_system_clock_contexts`.
    pub const LOCAL_SYSTEM_CLOCK_CONTEXTS: u64 = 0x38;
    /// `network_system_clock_contexts`.
    pub const NETWORK_SYSTEM_CLOCK_CONTEXTS: u64 = 0x80;
    /// `automatic_corrections`: a `bool`.
    pub const AUTOMATIC_CORRECTIONS: u64 = 0xC8;
    /// `continuous_adjustment_time_points`.
    pub const CONTINUOUS_ADJUSTMENT_TIME_POINTS: u64 = 0xD0;
}

/// The counter of every value once the clocks are set up: one write each
/// (`WriteToLockFreeAtomicType`: `++counter`). In this port it stays so for the server's life.
/// In Eden it is so only until the steady-clock timer's first run, straight after set-up, which
/// writes `steady_time_points` and `continuous_adjustment_time_points` again (module docs, "A
/// value and its counter").
pub const COUNTER_AFTER_SETUP: u32 = 1;

/// Where a value's first copy lies after its counter, for a value of alignment 8 (every value
/// here but the `bool`): the `u32` counter padded to 8.
pub const VALUE_AFTER_COUNTER: u64 = 0x8;
/// Where the `bool`'s first copy lies after its counter: alignment 1, straight after the `u32`.
pub const BOOL_AFTER_COUNTER: u64 = 0x4;

/// What the memory holds once the clocks are set up: every value as Eden's set-up writes it once
/// (`eden/src/core/hle/service/psc/time/service_manager.cpp`, the `Setup...Core` commands).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Contents {
    /// The steady clock's base: its time in nanoseconds at uptime zero, and its source.
    pub steady: SteadyClockTimePoint,
    /// The local system clock's context, which the user system clock reads too.
    pub local: SystemClockContext,
    /// The network system clock's context.
    pub network: SystemClockContext,
    /// Whether the user system clock corrects itself from the network clock.
    pub automatic_correction: bool,
    /// The steady clock's continuous adjustment.
    pub continuous_adjustment: ContinuousAdjustmentTimePoint,
}

impl SteadyClockTimePoint {
    /// The 0x18 bytes of the point: `time_point`, then the source.
    #[must_use]
    pub fn to_bytes(self) -> [u8; 0x18] {
        let mut bytes = [0; 0x18];
        bytes[..8].copy_from_slice(&self.time_point.to_le_bytes());
        bytes[8..].copy_from_slice(&self.source_id.0);
        bytes
    }
}

impl SystemClockContext {
    /// The 0x20 bytes of the context: `offset`, then the steady time point.
    #[must_use]
    pub fn to_bytes(self) -> [u8; 0x20] {
        let mut bytes = [0; 0x20];
        bytes[..8].copy_from_slice(&self.offset.to_le_bytes());
        bytes[8..].copy_from_slice(&self.steady_time_point.to_bytes());
        bytes
    }
}

impl ContinuousAdjustmentTimePoint {
    /// The 0x38 bytes of the point: the five `s64`s in order, then the source.
    #[must_use]
    pub fn to_bytes(self) -> [u8; 0x38] {
        let mut bytes = [0; 0x38];
        let words = [
            self.rtc_offset,
            self.diff_scale,
            self.shift_amount,
            self.lower,
            self.upper,
        ];
        let (chunks, _) = bytes.as_chunks_mut::<8>();
        for (chunk, word) in chunks.iter_mut().zip(words) {
            *chunk = word.to_le_bytes();
        }
        bytes[0x28..].copy_from_slice(&self.source_id.0);
        bytes
    }
}

/// The memory's bytes holding `contents`: [`SIZE`] bytes, each value's counter
/// [`COUNTER_AFTER_SETUP`] and the value in copy 1, everything else zero (Eden's constructor
/// zeroes the memory, `std::memset(m_shared_memory_ptr, 0, ..)`).
#[must_use]
pub fn image(contents: &Contents) -> Vec<u8> {
    let mut image = blank(SIZE);
    let steady = contents.steady.to_bytes();
    let local = contents.local.to_bytes();
    let network = contents.network.to_bytes();
    let automatic = [u8::from(contents.automatic_correction)];
    let continuous = contents.continuous_adjustment.to_bytes();
    let values: [(u64, u64, &[u8]); 5] = [
        (offset::STEADY_TIME_POINTS, VALUE_AFTER_COUNTER, &steady),
        (
            offset::LOCAL_SYSTEM_CLOCK_CONTEXTS,
            VALUE_AFTER_COUNTER,
            &local,
        ),
        (
            offset::NETWORK_SYSTEM_CLOCK_CONTEXTS,
            VALUE_AFTER_COUNTER,
            &network,
        ),
        (
            offset::AUTOMATIC_CORRECTIONS,
            BOOL_AFTER_COUNTER,
            &automatic,
        ),
        (
            offset::CONTINUOUS_ADJUSTMENT_TIME_POINTS,
            VALUE_AFTER_COUNTER,
            &continuous,
        ),
    ];
    for (at, after_counter, value) in values {
        let size = u64::try_from(value.len()).unwrap_or(u64::MAX);
        put(&mut image, at, &COUNTER_AFTER_SETUP.to_le_bytes(), MEMORY);
        put(&mut image, at + after_counter + size, value, MEMORY);
    }
    image
}
