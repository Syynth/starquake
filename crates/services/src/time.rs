//! `time:u`, the time service for a user program: as far as a libnx start-up asks, which is its
//! five clocks' interfaces, its shared memory, and one calendar conversion.
//!
//! Ported from Eden @ cf654052 (GPL-3.0-or-later), paths relative to the reference folder:
//! `eden/src/core/hle/service/glue/time/static.cpp` (`StaticService`, the table `time:u`
//! answers with), `glue/time/manager.cpp` (`TimeManager`: how the clocks are set up),
//! `glue/time/standard_steady_clock_resource.cpp` (the steady clock's source and base),
//! `eden/src/core/hle/service/psc/time/service_manager.cpp` (the `Setup...Core` commands that
//! write the shared memory), `psc/time/clocks/` (`StandardSteadyClockCore`,
//! `StandardLocalSystemClockCore`, `StandardNetworkSystemClockCore`, `SystemClockCore`),
//! `psc/time/time_zone_service.cpp` and `glue/time/time_zone.cpp` (`ITimeZoneService`),
//! `psc/time/errors.h` (the result codes) and `eden/src/core/hle/service/set/` (the settings'
//! defaults). The request layouts are checked against the client, libnx
//! (`libnx/nx/source/services/time.c` and `libnx/nx/source/runtime/newlib.c`, ISC). The shared
//! memory's layout is [`shared_memory`]; the calendar conversion is [`calendar`].
//!
//! # Commands
//!
//! Ids and names are Eden's tables (`StaticService::StaticService` in `glue/time/static.cpp`,
//! `TimeZoneService::TimeZoneService` in `glue/time/time_zone.cpp`). These are exactly the
//! `time:u` commands the startup fixture's trace sends
//! (`tests/homebrew/startup/reference/startup.trace`, lines 107 to 121); every other command is
//! left out and answered by the server's unknown-command policy.
//!
//! | Interface | Id | Command | In | Out |
//! |---|---|---|---|---|
//! | `time:u` ([`IStaticService`]) | 0 | `GetStandardUserSystemClock` | nothing | an [`ISystemClock`] |
//! | | 1 | `GetStandardNetworkSystemClock` | nothing | an [`ISystemClock`] |
//! | | 2 | `GetStandardSteadyClock` | nothing | an [`ISteadyClock`] |
//! | | 3 | `GetTimeZoneService` | nothing | an [`ITimeZoneService`] |
//! | | 4 | `GetStandardLocalSystemClock` | nothing | an [`ISystemClock`] |
//! | | 20 | `GetSharedMemoryNativeHandle` | nothing | copy handle: the time shared memory |
//! | [`ITimeZoneService`] | 101 | `ToCalendarTimeWithMyRule` | `s64` POSIX time | [`CalendarTime`] then [`CalendarAdditionalInfo`] |
//!
//! [`ISystemClock`] and [`ISteadyClock`] have no command here: libnx reads the clocks from the
//! shared memory when it has one (`timeGetCurrentTime`, `timeGetStandardSteadyClockTimePoint`:
//! `if (!shmemGetAddr(&g_timeSharedmem))` they send a request, otherwise they read), and the
//! startup fixture's trace sends them nothing.
//!
//! Layouts, each the client's and the reference's:
//!
//! - The `Get...Clock` and `GetTimeZoneService` commands: libnx `_timeCmdGetSession` (no
//!   payload, one out object); Eden `OutInterface<..>`.
//! - `GetSharedMemoryNativeHandle`: libnx `_timeGetSharedMemoryNativeHandle` (one copied
//!   handle, `SfOutHandleAttr_HipcCopy`); Eden `OutCopyHandle<Kernel::KSharedMemory>`.
//! - `ToCalendarTimeWithMyRule`: libnx `timeToCalendarTimeWithMyRule` (`serviceDispatchInOut(..,
//!   101, timestamp, out)`: a `u64` in, the 0x20-byte pair out); Eden `(Out<CalendarTime>,
//!   Out<CalendarAdditionalInfo>, s64 time)`.
//!
//! # The clocks
//!
//! Eden sets the clocks up when the emulator starts (`TimeManager::TimeManager`), from the
//! settings of `set:sys` and the host's clock. Here they are set up the first time a program
//! asks for the shared memory ([`IStaticService`]'s `GetSharedMemoryNativeHandle`), because the
//! kernel is reached only while a request is handled; the memory is made then, laid out as
//! Eden's set-up leaves it ([`shared_memory::Contents`], [`set_up`]), and kept for the server's
//! life. That is Eden's set-up with the settings' defaults (`set/setting_formats/
//! system_settings.cpp`, `system_settings_server.cpp`: no steady clock internal or test offset,
//! automatic correction on), on a first boot (no saved clock context), with no RTC reset:
//!
//! - **The steady clock's source** is a random UUID (`StandardSteadyClockResource::Initialize`:
//!   `Common::UUID::MakeRandom()`), drawn here as two `u64`s from the environment
//!   ([`starquake_trace::Env::random_u64`]), little-endian, in order.
//! - **The steady clock's base** is the RTC at uptime zero, in nanoseconds
//!   (`StandardSteadyClockResource::SetCurrentTime`: `boot_time = rtc_time_s * one_second_ns -
//!   uptime`), so the steady clock reads the RTC's seconds plus the uptime. The RTC is the run's
//!   [`TimeSettings::rtc_at_boot`] ("The run's date", below).
//! - **The local system clock's context** is set from the current time
//!   (`StandardLocalSystemClockCore::Initialize` with no saved context: `SetCurrentTime(time)`,
//!   whose `offset = time - steady_time_point`), where Eden's current time is the RTC
//!   (`TimeManager`: `GetRtcTimeInSeconds(epoch_time)` when the saved context is empty). The RTC
//!   and the steady clock agree, so the offset is zero, and the steady time point is the steady
//!   clock's seconds at set-up.
//! - **The network system clock's context** is the local one (Eden's
//!   `SetupStandardNetworkSystemClockCore`: "fix the clock to the local clock").
//! - **The continuous adjustment** is `{rtc_offset: uptime, 0, 0, lower: base, upper: base}`
//!   (`StandardSteadyClockCore::SetContinuousAdjustment`).
//!
//! The uptime is the environment's clock reading ([`Uptime`]), read once, after the two random
//! draws. Eden reads the ticks on both sides of its draw: `StandardSteadyClockResource::Initialize`
//! calls `SetCurrentTime()` first (two `GetClockTicks` reads and the host's RTC), then
//! `Common::UUID::MakeRandom()`, and after the draw it reads the ticks again
//! (`SetupStandardSteadyClockCore`'s `GetRawTime` and `GetClockTicks`,
//! `SetContinuousAdjustment`, and the local clock's `GetCurrentTimePoint`). The one reading here
//! stands for those later reads. The earlier one is not needed: it only turns the host's RTC
//! into a base at uptime zero, and here the RTC is given at uptime zero already
//! ([`TimeSettings::rtc_at_boot`], "The run's date").
//!
//! # The run's date
//!
//! The RTC at uptime zero ([`TimeSettings::rtc_at_boot`]) is the host's date, read once as the
//! run boots, unless a setting pins a date (decision log, 2026-11-17, "The run's date comes from
//! the host, recorded"; `docs/specs/observability.md`, rule 1). [`RtcAtBoot`] says which, and
//! [`TimeSettings::at_boot`] turns it into the settings: [`RtcAtBoot::Host`] reads the
//! environment's wall clock ([`Env::wall_clock`]) once, which a live environment records and a
//! replay feeds back, and [`RtcAtBoot::Pinned`] reads nothing. Tests always pin a date:
//! [`TimeSettings::eden_default`] and [`RtcAtBoot::eden_default`] (which is also
//! [`RtcAtBoot`]'s `Default`) pin the first second of Eden's `standard_user_clock_initial_year`
//! (2023). `starquake-core` reads the date before it installs the services, from the run's
//! settings (its `RunOptions`), and a failure to read it is an emulator error that ends the run
//! before the guest runs (decision log, 2026-11-17, "Env failures"). `time:u` owns turning the
//! RTC and the uptime into calendar time ([`set_up`], [`calendar`]); the kernel keeps owning
//! ticks ([`Uptime`]).
//!
//! # Where this differs from Eden
//!
//! Each difference is pinned by the test named, in `time/tests.rs`:
//!
//! - **The host's date is read once, before the uptime starts to count, and a run can pin it.**
//!   Eden reads the host's wall clock (`GetTimeInSeconds` in `glue/time/
//!   standard_steady_clock_resource.cpp`: `system_clock::now()`, plus `custom_rtc_offset` when
//!   `custom_rtc_enabled`) each time it sets the steady clock's base, and subtracts the uptime of
//!   that moment. Here the one reading taken at boot is the RTC at uptime zero: the time between
//!   guest tick zero and the reading (the run's boot, before the guest runs) is not subtracted, so
//!   the RTC is later than Eden's by that much, which is less than a second on a normal boot
//!   (a guess: nothing here measures it). Eden pins a date as an offset from the host's clock
//!   (`custom_rtc_offset`); here a pinned date is the RTC itself, and the host's clock is not
//!   read (`a_pinned_date_wins_over_the_hosts_and_reads_nothing`,
//!   `the_hosts_date_is_the_rtc_at_boot`).
//! - **The clocks are set up at the first request for the memory**, not at start-up (above).
//!   Only the context's steady time point and the continuous adjustment's `rtc_offset` record
//!   when that was; what the clocks read does not change
//!   (`the_clocks_follow_the_settings_and_the_environment`).
//! - **The memory is written once and never updated.** Eden's `TimeWorker`
//!   (`glue/time/worker.cpp`) schedules its steady-clock timer with
//!   `ScheduleLoopingEvent(0ns, standard_steady_clock_rtc_update_interval_minutes)`, 5 minutes
//!   by default (`set/system_settings_server.cpp`), so straight after set-up, and every 5
//!   minutes after that, it runs `SetStandardSteadyClockBaseTime`
//!   (`psc/time/service_manager.cpp`). That rewrites `steady_time_points`
//!   (`SharedMemory::UpdateBaseTime`: counter 2, the value in copy 0) and
//!   `continuous_adjustment_time_points` (a later `rtc_offset`, and
//!   `StandardSteadyClockCore::UpdateContinuousAdjustmentTime` can set `diff_scale` to -55 or 55
//!   and `shift_amount` to 14). A guest under Eden never sees those two values as this port
//!   leaves them, counter 1 and the value in copy 1 ([`shared_memory`], "A value and its
//!   counter"); here nothing runs on a timer, so every value keeps the image its one write gives
//!   (`the_shared_memory_layout_is_pinned_by_golden_bytes`,
//!   `a_second_request_gets_the_same_memory_and_reads_nothing`).
//! - **The device's time zone is UTC.** Eden uses the zone its settings name, or the host's
//!   (`GetTimeZoneString`), with rules from its time zone archive; this port has the UTC rule
//!   only ([`calendar`], which lists its own divergence)
//!   (`to_calendar_time_with_my_rule_answers_in_utc_and_the_reply_is_these_bytes`).
//! - **A failure to set the clocks up is the reply's.** Eden's set-up cannot fail where a guest
//!   sees it. Here an error from the kernel when the memory is made is the reply's result, and an
//!   uptime past `i64`'s nanoseconds is [`FAILED`], logged; a base past `i64`'s nanoseconds is
//!   [`OVERFLOW`]. Nothing is kept, so the next request tries again
//!   (`a_failed_set_up_is_the_reply_and_keeps_nothing`,
//!   `a_base_past_the_nanoseconds_an_i64_holds_is_an_overflow`).
//!
//! # When the environment fails
//!
//! The set-up reads the environment the server was given ([`server`]), and a failure of it (a
//! replay that ran out of answers or disagreed with what was asked) is not a result the guest
//! sees: it stops the run, through the framework's one path for it
//! ([`halt`](crate::framework::halt); decision log, 2026-11-17, "Env failures"). The run's halt
//! is told [`Stop::Handler`](crate::framework::Stop::Handler) naming [`NAME`] and the error, at
//! the first of the three reads that fails, and no later read is made; the request is answered
//! as one the server gave up on, which the guest never runs to see, and nothing is kept
//! (`an_environment_failure_stops_the_run_and_keeps_nothing`, for each read, with a replay that
//! runs out and one that disagrees). `time:u` sends no host I/O, so it has no host failure to
//! answer with a result code.

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use starquake_kernel::id::ObjectId;
use starquake_kernel::memory::SvcPermission;
use starquake_kernel::result::{KernelError, ResultCode};
use starquake_trace::{Env, EnvError, HostNanos, UnixSeconds};

use crate::commands;
use crate::framework::result::code;
use crate::framework::{
    Context, Fault, Halt, Interfaces, Outcome, Reply, Request, Server, UnknownPolicy,
};

pub mod calendar;
pub mod shared_memory;

pub use calendar::{CalendarAdditionalInfo, CalendarTime};
use shared_memory::{
    ClockSourceId, Contents, ContinuousAdjustmentTimePoint, SteadyClockTimePoint,
    SystemClockContext,
};

#[cfg(test)]
mod tests;

/// The name `time:u` is registered under (`eden/src/core/hle/service/glue/glue.cpp`,
/// `LoopProcess`: `RegisterNamedService("time:u", std::make_shared<Time::StaticService>(..))`).
pub const NAME: &str = "time:u";

/// `ErrorModule::TimeService` (`eden/src/core/hle/result.h`; libnx's clock-mismatch result is
/// `MAKERESULT(116, 102)`).
pub const TIME_MODULE: u32 = 116;

/// `ResultOverflow` (`eden/src/core/hle/service/psc/time/errors.h`: `{ErrorModule::TimeService,
/// 201}`).
pub const OVERFLOW: ResultCode = code(TIME_MODULE, 201);
/// `ResultFailed` (`errors.h`: `{ErrorModule::TimeService, 801}`): what this port answers when
/// the environment's clock reads past the nanoseconds an `s64` holds (module docs).
pub const FAILED: ResultCode = code(TIME_MODULE, 801);

/// Nanoseconds in a second.
pub const NANOSECONDS_PER_SECOND: i64 = 1_000_000_000;

/// The ids of the commands ported here, per interface (Eden's tables).
pub mod command {
    /// `time:u`'s commands (`StaticService::StaticService`).
    pub mod static_service {
        /// `GetStandardUserSystemClock` (`{0, D<&StaticService::GetStandardUserSystemClock>, ..}`).
        pub const GET_STANDARD_USER_SYSTEM_CLOCK: u32 = 0;
        /// `GetStandardNetworkSystemClock` (`{1, ..}`).
        pub const GET_STANDARD_NETWORK_SYSTEM_CLOCK: u32 = 1;
        /// `GetStandardSteadyClock` (`{2, ..}`).
        pub const GET_STANDARD_STEADY_CLOCK: u32 = 2;
        /// `GetTimeZoneService` (`{3, ..}`).
        pub const GET_TIME_ZONE_SERVICE: u32 = 3;
        /// `GetStandardLocalSystemClock` (`{4, ..}`).
        pub const GET_STANDARD_LOCAL_SYSTEM_CLOCK: u32 = 4;
        /// `GetSharedMemoryNativeHandle` (`{20, ..}`).
        pub const GET_SHARED_MEMORY_NATIVE_HANDLE: u32 = 20;
    }

    /// `ITimeZoneService`'s commands (`TimeZoneService::TimeZoneService` in
    /// `glue/time/time_zone.cpp`).
    pub mod time_zone_service {
        /// `ToCalendarTimeWithMyRule` (`{101, D<&TimeZoneService::ToCalendarTimeWithMyRule>,
        /// ..}`).
        pub const TO_CALENDAR_TIME_WITH_MY_RULE: u32 = 101;
    }
}

/// Where a run's RTC at boot comes from (module docs, "The run's date").
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RtcAtBoot {
    /// The host's date: one reading of the environment's wall clock ([`Env::wall_clock`]), taken
    /// as the run boots and recorded, so a replay reproduces it.
    Host,
    /// A pinned date: the RTC at uptime zero is this, and the host's clock is not read.
    Pinned(UnixSeconds),
}

impl RtcAtBoot {
    /// The date tests pin: [`TimeSettings::EDEN_INITIAL_YEAR_START`], as
    /// [`TimeSettings::eden_default`] does.
    #[must_use]
    pub const fn eden_default() -> Self {
        Self::Pinned(UnixSeconds(TimeSettings::EDEN_INITIAL_YEAR_START))
    }
}

impl Default for RtcAtBoot {
    /// [`RtcAtBoot::eden_default`], a pinned date: a run that does not say gets a date that does
    /// not depend on the host. The command line asks for [`RtcAtBoot::Host`] unless it is given
    /// a date.
    fn default() -> Self {
        Self::eden_default()
    }
}

/// The time a run's clocks start from (module docs, "The clocks").
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TimeSettings {
    /// The RTC at uptime zero: seconds since 1970-01-01T00:00:00 UTC.
    pub rtc_at_boot: i64,
    /// Whether the user system clock corrects itself from the network clock (Eden's
    /// `user_system_clock_automatic_correction_enabled`).
    pub automatic_correction: bool,
}

impl TimeSettings {
    /// 2023-01-01T00:00:00 UTC: the first second of Eden's `standard_user_clock_initial_year`
    /// (`set/system_settings_server.cpp`: `ToBytes(s32{2023})`).
    pub const EDEN_INITIAL_YEAR_START: i64 = 1_672_531_200;

    /// Eden's defaults: the RTC at the start of its initial year (a listed divergence: Eden
    /// reads the host's clock), and automatic correction on
    /// (`set/setting_formats/system_settings.cpp`:
    /// `user_system_clock_automatic_correction_enabled = true`).
    #[must_use]
    pub const fn eden_default() -> Self {
        Self::with_rtc(UnixSeconds(Self::EDEN_INITIAL_YEAR_START))
    }

    /// Eden's defaults with the RTC at uptime zero `rtc`: the one place a wall-clock reading
    /// becomes the RTC.
    #[must_use]
    pub const fn with_rtc(rtc: UnixSeconds) -> Self {
        Self {
            rtc_at_boot: rtc.0,
            automatic_correction: true,
        }
    }

    /// The settings a run boots with: Eden's defaults, with the RTC `rtc` says (module docs,
    /// "The run's date"). [`RtcAtBoot::Host`] reads `env`'s wall clock once and
    /// [`RtcAtBoot::Pinned`] reads nothing. The error is the environment's, which ends the run.
    pub fn at_boot(rtc: RtcAtBoot, env: &dyn Env) -> Result<Self, EnvError> {
        let rtc = match rtc {
            RtcAtBoot::Pinned(date) => date,
            RtcAtBoot::Host => env.wall_clock()?,
        };
        Ok(Self::with_rtc(rtc))
    }
}

/// Nanoseconds since guest tick zero, as the environment's clock reads them.
///
/// The environment's clock reads host nanoseconds from a zero that is guest tick zero
/// ([`Env::now`]). The kernel converts that reading to guest ticks; the time service needs
/// nanoseconds, never ticks (Eden converts its ticks back, `ConvertToTimeSpan(GetClockTicks())`),
/// so it takes the reading as it is. This is the one place a reading becomes an uptime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Uptime(i64);

impl Uptime {
    /// The uptime a clock reading stands for; `None` past `i64::MAX` nanoseconds (about 292
    /// years), which an `s64` time point cannot hold.
    #[must_use]
    pub fn from_reading(reading: HostNanos) -> Option<Self> {
        i64::try_from(reading.0).ok().map(Self)
    }

    /// The nanoseconds.
    #[must_use]
    pub const fn nanoseconds(self) -> i64 {
        self.0
    }
}

/// What the clocks hold once set up: `settings` at `uptime`, with steady clock source
/// `source_id` (module docs, "The clocks"). [`OVERFLOW`] when the base or the steady clock's
/// time does not fit `i64` nanoseconds.
pub fn set_up(
    settings: &TimeSettings,
    source_id: ClockSourceId,
    uptime: Uptime,
) -> Result<Contents, Fault> {
    let overflow = Fault::Result(OVERFLOW);
    let base = settings
        .rtc_at_boot
        .checked_mul(NANOSECONDS_PER_SECOND)
        .ok_or(overflow)?;
    let now = base.checked_add(uptime.nanoseconds()).ok_or(overflow)?;
    // `GetCurrentTimePointImpl`: `duration_cast<seconds>`, which truncates toward zero, as `/`
    // does.
    let steady_seconds = now / NANOSECONDS_PER_SECOND;
    // `SetCurrentTime(time)` with `time` the RTC's seconds now, which in this model are the
    // steady clock's: `offset = time - time_point` is zero.
    let context = SystemClockContext {
        offset: 0,
        steady_time_point: SteadyClockTimePoint {
            time_point: steady_seconds,
            source_id,
        },
    };
    Ok(Contents {
        steady: SteadyClockTimePoint {
            time_point: base,
            source_id,
        },
        local: context,
        network: context,
        automatic_correction: settings.automatic_correction,
        continuous_adjustment: ContinuousAdjustmentTimePoint {
            rtc_offset: uptime.nanoseconds(),
            diff_scale: 0,
            shift_amount: 0,
            lower: base,
            upper: base,
            source_id,
        },
    })
}

/// The clocks once set up: what the shared memory holds, and the memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SetUp {
    /// What the shared memory was laid out with.
    pub contents: Contents,
    /// The shared memory's kernel object, which the server keeps for its life.
    pub memory: ObjectId,
}

/// The `time:u` server's family state: the run's settings, and the clocks once set up.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Clocks {
    settings: TimeSettings,
    set_up: Option<SetUp>,
}

impl Clocks {
    /// Clocks that start from `settings` and are not set up yet.
    #[must_use]
    pub const fn new(settings: TimeSettings) -> Self {
        Self {
            settings,
            set_up: None,
        }
    }

    /// The run's settings.
    #[must_use]
    pub const fn settings(&self) -> &TimeSettings {
        &self.settings
    }

    /// The clocks and their memory, once a request has set them up.
    #[must_use]
    pub const fn set_up(&self) -> Option<&SetUp> {
        self.set_up.as_ref()
    }
}

/// The root interface of a `time:u` session (`StaticService`). It keeps nothing.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct IStaticService;

/// Which system clock an [`ISystemClock`] is.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum SystemClockKind {
    /// The standard user system clock (`GetStandardUserSystemClock`).
    #[default]
    StandardUser,
    /// The standard network system clock (`GetStandardNetworkSystemClock`).
    StandardNetwork,
    /// The standard local system clock (`GetStandardLocalSystemClock`).
    StandardLocal,
}

/// A system clock (`ISystemClock`, `psc/time/system_clock.cpp`). None of its commands is ported
/// (module docs, "Commands").
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct ISystemClock {
    kind: SystemClockKind,
}

impl ISystemClock {
    /// Which clock it is.
    #[must_use]
    pub fn kind(&self) -> SystemClockKind {
        self.kind
    }
}

/// The standard steady clock (`ISteadyClock`, `psc/time/steady_clock.cpp`). None of its commands
/// is ported (module docs, "Commands").
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct ISteadyClock;

/// The time zone service (`ITimeZoneService`, `glue/time/time_zone.cpp`). It keeps nothing: the
/// device's rule is UTC (module docs).
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct ITimeZoneService;

commands! {
    impl IStaticService {
        command::static_service::GET_STANDARD_USER_SYSTEM_CLOCK => get_standard_user_system_clock,
        command::static_service::GET_STANDARD_NETWORK_SYSTEM_CLOCK
            => get_standard_network_system_clock,
        command::static_service::GET_STANDARD_STEADY_CLOCK => get_standard_steady_clock,
        command::static_service::GET_TIME_ZONE_SERVICE => get_time_zone_service,
        command::static_service::GET_STANDARD_LOCAL_SYSTEM_CLOCK
            => get_standard_local_system_clock,
        command::static_service::GET_SHARED_MEMORY_NATIVE_HANDLE
            => get_shared_memory_native_handle,
    }
}

// No command of these two is ported (module docs, "Commands"); every request to one goes to the
// unknown-command policy.
commands! {
    impl ISystemClock {}
}

commands! {
    impl ISteadyClock {}
}

commands! {
    impl ITimeZoneService {
        command::time_zone_service::TO_CALENDAR_TIME_WITH_MY_RULE => to_calendar_time_with_my_rule,
    }
}

impl IStaticService {
    /// Hands out a system clock of `kind`.
    fn system_clock(kind: SystemClockKind, reply: &mut Reply) -> Outcome {
        reply.push_session(Box::new(ISystemClock { kind }));
        reply.ok()
    }

    /// `GetStandardUserSystemClock`. out: an [`ISystemClock`].
    fn get_standard_user_system_clock(
        _: &mut Self,
        _cx: &mut Context<'_>,
        _req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        Self::system_clock(SystemClockKind::StandardUser, reply)
    }

    /// `GetStandardNetworkSystemClock`. out: an [`ISystemClock`].
    fn get_standard_network_system_clock(
        _: &mut Self,
        _cx: &mut Context<'_>,
        _req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        Self::system_clock(SystemClockKind::StandardNetwork, reply)
    }

    /// `GetStandardSteadyClock`. out: an [`ISteadyClock`].
    fn get_standard_steady_clock(
        _: &mut Self,
        _cx: &mut Context<'_>,
        _req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        reply.push_session(Box::new(ISteadyClock));
        reply.ok()
    }

    /// `GetTimeZoneService`. out: an [`ITimeZoneService`].
    fn get_time_zone_service(
        _: &mut Self,
        _cx: &mut Context<'_>,
        _req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        reply.push_session(Box::new(ITimeZoneService));
        reply.ok()
    }

    /// `GetStandardLocalSystemClock`. out: an [`ISystemClock`].
    fn get_standard_local_system_clock(
        _: &mut Self,
        _cx: &mut Context<'_>,
        _req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        Self::system_clock(SystemClockKind::StandardLocal, reply)
    }

    /// `GetSharedMemoryNativeHandle`. out: a copy handle to the time shared memory, which the
    /// first request sets up (module docs, "The clocks").
    fn get_shared_memory_native_handle(
        _: &mut Self,
        cx: &mut Context<'_>,
        _req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        let clocks = *cx.family::<Clocks>()?;
        let memory = if let Some(done) = clocks.set_up {
            done.memory
        } else {
            let done = set_up_memory(cx, &clocks.settings)?;
            cx.family::<Clocks>()?.set_up = Some(done);
            done.memory
        };
        reply.copy_handle(memory);
        reply.ok()
    }
}

/// Reads the environment and sets the clocks up (module docs, "The clocks"): two random draws,
/// then one clock reading. An environment that fails stops the run (module docs, "When the
/// environment fails"): `env` is the server's, which has told the run's halt before it returns
/// the error, and `?` answers the request as one the server gave up on.
fn read_clock_inputs(env: &dyn Env) -> Result<(ClockSourceId, Uptime), Fault> {
    let low = env.random_u64()?;
    let high = env.random_u64()?;
    let reading = env.now()?;
    let mut source = [0; 16];
    source[..8].copy_from_slice(&low.to_le_bytes());
    source[8..].copy_from_slice(&high.to_le_bytes());
    let uptime = Uptime::from_reading(reading).ok_or_else(|| {
        tracing::error!(
            reading = reading.0,
            "the clock reads past the nanoseconds an s64 holds"
        );
        Fault::Result(FAILED)
    })?;
    Ok((ClockSourceId(source), uptime))
}

/// Sets the clocks up and makes the shared memory that carries them. The image is written in one
/// write, which touches nothing unless it all fits; the kernel rounds the size up, never down,
/// so it does. Should it fail anyway, the object is not returned: it stays the server's, as
/// every object a server makes does.
fn set_up_memory(cx: &mut Context<'_>, settings: &TimeSettings) -> Result<SetUp, Fault> {
    let (source_id, uptime) = read_clock_inputs(&*cx.env()?)?;
    let contents = set_up(settings, source_id, uptime)?;
    let (memory, pages) = cx.create_shared_memory(shared_memory::SIZE, SvcPermission::Read)?;
    pages
        .write(0, &shared_memory::image(&contents))
        .map_err(|fault| {
            tracing::error!(?fault, "the time shared memory is smaller than its layout");
            Fault::Kernel(KernelError::InvalidState)
        })?;
    Ok(SetUp { contents, memory })
}

impl ITimeZoneService {
    /// `ToCalendarTimeWithMyRule`. in: `s64` POSIX time. out: the calendar time and its
    /// additional information under the device's rule, UTC.
    fn to_calendar_time_with_my_rule(
        _: &mut Self,
        _cx: &mut Context<'_>,
        req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        let time = req.payload().read_i64()?;
        let (calendar, info) = calendar::to_calendar_time_utc(time)?;
        reply
            .payload()
            .write_bytes(&calendar.to_bytes())
            .write_bytes(&info.to_bytes());
        reply.ok()
    }
}

/// The interfaces a `time:u` server's sessions can be.
#[must_use]
pub fn interfaces() -> Interfaces {
    Interfaces::new()
        .with::<IStaticService>()
        .with::<ISystemClock>()
        .with::<ISteadyClock>()
        .with::<ITimeZoneService>()
}

/// The `time:u` server, whose clocks start from `settings` and read `env` (module docs, "The
/// clocks"), telling `halt` when `env` fails it (module docs, "When the environment fails").
#[must_use]
pub fn server(
    settings: TimeSettings,
    policy: UnknownPolicy,
    env: Arc<dyn Env>,
    halt: Arc<dyn Halt>,
) -> Server {
    Server::new(NAME, policy)
        .with_root(|| Box::new(IStaticService))
        .with_interfaces(interfaces())
        .with_family(Clocks::new(settings))
        .with_env(env, halt)
}
