//! Calendar time: what `ToCalendarTimeWithMyRule` makes of a POSIX time, under the one rule this
//! port has, UTC.
//!
//! Eden converts with its port of the tz code's `localtime_rz` under the device's rule
//! (`eden/src/core/hle/service/psc/time/time_zone.cpp` @ cf654052, `TimeZone::ToCalendarTimeImpl`,
//! GPL-3.0-or-later), then copies the fields out: `year = tm_year + 1900`, `month = tm_mon + 1`,
//! `day = tm_mday`, `hour`, `minute`, `second`, `day_of_week = tm_wday` (0 is Sunday),
//! `day_of_year = tm_yday` (0 is the first of January), the zone's abbreviation (at most seven
//! characters and a zero, `name[size - 1] = '\0'`), `is_dst = tm_isdst` and
//! `ut_offset = tm_utoff` (seconds east of UTC). Under UTC there are no transitions, so
//! `localtime_rz` is the proleptic Gregorian calendar with an offset of zero, an abbreviation of
//! `UTC` and no daylight saving. The days are converted with the civil-from-days algorithm
//! (H. Hinnant, "chrono-Compatible Low-Level Date Algorithms", public domain), which is exact
//! for every day an `i64` of seconds can name.
//!
//! # Where this differs from Eden
//!
//! - **A year that does not fit the reply is an error.** Eden answers `ResultOverflow` only when
//!   `localtime_rz` fails, which for UTC is a year past `int`'s range, and otherwise casts the
//!   year to `s16` (`static_cast<s16>(calendar_internal.tm_year + 1900)`), so a year past 32767
//!   comes back wrapped. Here any year outside `i16`'s range is [`OVERFLOW`]: stricter, and pinned
//!   by `a_year_outside_the_reply_s_range_is_an_overflow` in `time/tests.rs`.

use super::OVERFLOW;
use crate::framework::Fault;

/// Seconds in a day.
pub const SECONDS_PER_DAY: i64 = 86_400;
/// Days from 0000-03-01 to 1970-01-01, the shift the civil-from-days algorithm starts with.
const EPOCH_SHIFT_DAYS: i64 = 719_468;
/// Days in a 400-year era.
const DAYS_PER_ERA: i64 = 146_097;
/// The day of the week of 1970-01-01, a Thursday, counting Sunday as 0 (`tm_wday`).
const EPOCH_DAY_OF_WEEK: i64 = 4;

/// The zone abbreviation the UTC rule gives (`tm_zone`), as the reply's eight bytes hold it.
pub const UTC_NAME: [u8; 8] = *b"UTC\0\0\0\0\0";

/// A calendar time (`CalendarTime`, `eden/src/core/hle/service/psc/time/common.h`: `s16 year;
/// s8 month, day, hour, minute, second;`, 8 bytes with one byte of padding).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CalendarTime {
    /// The year.
    pub year: i16,
    /// The month, 1 to 12.
    pub month: i8,
    /// The day of the month, 1 to 31.
    pub day: i8,
    /// The hour, 0 to 23.
    pub hour: i8,
    /// The minute, 0 to 59.
    pub minute: i8,
    /// The second, 0 to 59.
    pub second: i8,
}

/// What a calendar time comes with (`CalendarAdditionalInfo`, `common.h`: `s32 day_of_week; s32
/// day_of_year; std::array<char, 8> name; s32 is_dst; s32 ut_offset;`, 0x18 bytes).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CalendarAdditionalInfo {
    /// The day of the week, 0 (Sunday) to 6.
    pub day_of_week: i32,
    /// The day of the year, 0 (the first of January) to 365.
    pub day_of_year: i32,
    /// The zone's abbreviation, zero-terminated.
    pub name: [u8; 8],
    /// Whether daylight saving is in effect: 0 or 1.
    pub is_dst: i32,
    /// Seconds east of UTC.
    pub ut_offset: i32,
}

/// The size of `ToCalendarTimeWithMyRule`'s reply: the 8-byte calendar time, then the 0x18-byte
/// additional information at its own alignment, 4 (Eden's out arguments in order, each at its
/// alignment, `eden/src/core/hle/service/cmif_serialization.h`; libnx's `struct {
/// TimeCalendarTime caltime; TimeCalendarAdditionalInfo info; }` in `timeToCalendarTimeWithMyRule`).
pub const REPLY_SIZE: usize = 0x20;

impl CalendarTime {
    /// The 8 bytes of the reply's calendar time, the padding byte zero.
    #[must_use]
    pub fn to_bytes(self) -> [u8; 8] {
        let [year_low, year_high] = self.year.to_le_bytes();
        let byte = |value: i8| value.to_le_bytes()[0];
        [
            year_low,
            year_high,
            byte(self.month),
            byte(self.day),
            byte(self.hour),
            byte(self.minute),
            byte(self.second),
            0,
        ]
    }
}

impl CalendarAdditionalInfo {
    /// The 0x18 bytes of the reply's additional information.
    #[must_use]
    pub fn to_bytes(self) -> [u8; 0x18] {
        let mut bytes = [0; 0x18];
        let fields = [
            (0x0, self.day_of_week.to_le_bytes()),
            (0x4, self.day_of_year.to_le_bytes()),
            (0x10, self.is_dst.to_le_bytes()),
            (0x14, self.ut_offset.to_le_bytes()),
        ];
        for (at, field) in fields {
            bytes[at..at + 4].copy_from_slice(&field);
        }
        bytes[0x8..0x10].copy_from_slice(&self.name);
        bytes
    }
}

/// The calendar time of `time` (seconds since 1970-01-01T00:00:00 UTC) under UTC, with its
/// additional information. A year outside `i16`'s range is [`OVERFLOW`] (module docs).
pub fn to_calendar_time_utc(time: i64) -> Result<(CalendarTime, CalendarAdditionalInfo), Fault> {
    // `div_euclid` and `rem_euclid` by a positive constant cannot overflow; every value below
    // is bounded by `|time| / SECONDS_PER_DAY < 1.1e14` days, far inside `i64`.
    let days = time.div_euclid(SECONDS_PER_DAY);
    let seconds = time.rem_euclid(SECONDS_PER_DAY);
    let (year, month, day) = civil_from_days(days);
    let year = i16::try_from(year).map_err(|_| Fault::Result(OVERFLOW))?;
    let day_of_year = days - days_from_civil(i64::from(year), 1, 1);
    let day_of_week = (days + EPOCH_DAY_OF_WEEK).rem_euclid(7);
    // Each of these is inside its documented range, so the narrowing never fails; a failure
    // would be a mistake here, reported as an overflow rather than a panic.
    let narrow = |value: i64| i8::try_from(value).map_err(|_| Fault::Result(OVERFLOW));
    let wide = |value: i64| i32::try_from(value).map_err(|_| Fault::Result(OVERFLOW));
    let calendar = CalendarTime {
        year,
        month: narrow(month)?,
        day: narrow(day)?,
        hour: narrow(seconds / 3600)?,
        minute: narrow(seconds % 3600 / 60)?,
        second: narrow(seconds % 60)?,
    };
    let info = CalendarAdditionalInfo {
        day_of_week: wide(day_of_week)?,
        day_of_year: wide(day_of_year)?,
        name: UTC_NAME,
        is_dst: 0,
        ut_offset: 0,
    };
    Ok((calendar, info))
}

/// The proleptic Gregorian `(year, month, day)` of the day `days` after 1970-01-01
/// (`civil_from_days`).
fn civil_from_days(days: i64) -> (i64, i64, i64) {
    let shifted = days + EPOCH_SHIFT_DAYS;
    let era = shifted.div_euclid(DAYS_PER_ERA);
    let day_of_era = shifted - era * DAYS_PER_ERA;
    let year_of_era =
        (day_of_era - day_of_era / 1460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let day_of_march_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_from_march = (5 * day_of_march_year + 2) / 153;
    let day = day_of_march_year - (153 * month_from_march + 2) / 5 + 1;
    let month = if month_from_march < 10 {
        month_from_march + 3
    } else {
        month_from_march - 9
    };
    let year = year_of_era + era * 400 + i64::from(month <= 2);
    (year, month, day)
}

/// The day, counted from 1970-01-01, of the proleptic Gregorian `year`-`month`-`day`
/// (`days_from_civil`), for a year in `i16`'s range.
fn days_from_civil(year: i64, month: i64, day: i64) -> i64 {
    let year = year - i64::from(month <= 2);
    let era = year.div_euclid(400);
    let year_of_era = year - era * 400;
    let month_from_march = if month > 2 { month - 3 } else { month + 9 };
    let day_of_march_year = (153 * month_from_march + 2) / 5 + day - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_march_year;
    era * DAYS_PER_ERA + day_of_era - EPOCH_SHIFT_DAYS
}
