//! `time:u` driven through raw messages laid out as libnx lays them out
//! (`libnx/nx/source/services/time.c`), over a fake kernel and a fixed environment. libnx keeps
//! its `time` sessions plain (`NX_SERVICE_ASSUME_NON_DOMAIN`), as the startup fixture's trace
//! shows (line 106 sends no conversion), so these tests do too.

use std::sync::Arc;

use proptest::prelude::*;
use proptest::test_runner::{Config, RngAlgorithm, RngSeed};
use starquake_ipc::{
    BUFFER_SIZE, CommandHeader, CommandType, Message, ReceiveList, RequestBuilder, RequestLayout,
    SFCO_MAGIC, SessionKind, SpecialHeader,
};
use starquake_kernel::handle::Handle;
use starquake_kernel::id::ObjectId;
use starquake_kernel::memory::SvcPermission;
use starquake_kernel::result::{KernelError, ResultCode};
use starquake_trace::{EnvCall, FixedClock, LiveEnv, ReplayEnv};

use super::calendar::{REPLY_SIZE, SECONDS_PER_DAY, UTC_NAME, to_calendar_time_utc};
use super::command::static_service::{
    GET_SHARED_MEMORY_NATIVE_HANDLE, GET_STANDARD_LOCAL_SYSTEM_CLOCK,
    GET_STANDARD_NETWORK_SYSTEM_CLOCK, GET_STANDARD_STEADY_CLOCK, GET_STANDARD_USER_SYSTEM_CLOCK,
    GET_TIME_ZONE_SERVICE,
};
use super::command::time_zone_service::TO_CALENDAR_TIME_WITH_MY_RULE;
use super::shared_memory::{ClockSourceId, SIZE, image};
use super::*;
use crate::framework::fake::FakeKernel;
use crate::framework::result::UNKNOWN_COMMAND_ID;
use crate::framework::{Handled, Kernel, Protocol, Session};

/// `StaticService::GetEphemeralNetworkSystemClock`, which this port leaves out, so it has no
/// shipping constant (`glue/time/static.cpp`: `{5, ..}`).
const GET_EPHEMERAL_NETWORK_SYSTEM_CLOCK: u32 = 5;
/// `TimeZoneService::ToCalendarTime`, likewise (`glue/time/time_zone.cpp`: `{100, ..}`).
const TO_CALENDAR_TIME: u32 = 100;
/// `SystemClock::GetCurrentTime`, likewise (`psc/time/system_clock.cpp`: `{0, ..}`).
const GET_CURRENT_TIME: u32 = 0;

/// The environment's first clock reading: 3.5 seconds of uptime, so the steady clock's seconds
/// at set-up are the RTC's plus 3.
const UPTIME: u64 = 3_500_000_000;
/// The environment's random seed.
const SEED: u64 = 0x5EED_0F71_AE00;
/// What fills the message buffer beforehand.
const STALE: u8 = 0xA5;

/// A `time:u` server, a kernel, one session, and the environment the server reads.
struct Rig {
    kernel: FakeKernel,
    server: Server,
    session: ObjectId,
    env: Arc<LiveEnv>,
}

impl Rig {
    fn with(settings: TimeSettings, clock_start: u64) -> Self {
        let env = Arc::new(LiveEnv::new(FixedClock::new(clock_start, 1), SEED));
        let mut kernel = FakeKernel::new(0).unwrap();
        let session = kernel.new_object();
        let shared: Arc<dyn Env> = env.clone();
        let mut server = server(settings, UnknownPolicy::Strict, shared);
        server.accept(session).unwrap();
        Self {
            kernel,
            server,
            session,
            env,
        }
    }

    fn new() -> Self {
        Self::with(TimeSettings::eden_default(), UPTIME)
    }

    fn send_raw(&mut self, session: ObjectId, request: &RequestBuilder) -> Vec<u8> {
        let mut buffer = vec![STALE; BUFFER_SIZE];
        request.build(&mut buffer).unwrap();
        let handled = self.server.handle(&mut self.kernel, session, &mut buffer);
        assert_eq!(handled, Ok(Handled::Replied));
        buffer
    }

    /// Sends `command` with no payload to the rig's session; the handles the reply copied and
    /// moved.
    fn send(&mut self, command: u32) -> (Vec<u32>, Vec<u32>) {
        let reply = self.send_raw(self.session, &libnx_request(command));
        let message = Message::parse_reply(&reply, SessionKind::Session).unwrap();
        (message.copy_handles.clone(), message.move_handles.clone())
    }

    /// Opens the interface `command` hands out; its session.
    fn open(&mut self, command: u32) -> ObjectId {
        let (_, moved) = self.send(command);
        self.object_of(moved[0])
    }

    /// Asks for the shared memory; the memory.
    fn memory(&mut self) -> ObjectId {
        let (copied, _) = self.send(GET_SHARED_MEMORY_NATIVE_HANDLE);
        self.object_of(copied[0])
    }

    fn object_of(&self, handle: u32) -> ObjectId {
        self.kernel
            .resolve_handle(Handle::from_raw(handle))
            .unwrap()
    }

    fn clocks(&self) -> Clocks {
        *self.server.family::<Clocks>().unwrap()
    }

    /// What the environment was asked, in order.
    fn env_calls(&self) -> Vec<EnvCall> {
        self.env.recording().iter().map(|r| r.call()).collect()
    }

    /// Whether the server's snapshot holds an interface named `interface` whose state is
    /// `state`: a snapshot writes the name as `postcard` writes a string (its length, one byte
    /// for a name this short, then its bytes), then the state as a byte string.
    fn holds(&self, interface: &str, state: &[u8]) -> bool {
        let saved = self.server.save().unwrap();
        let mut row = vec![u8::try_from(interface.len()).unwrap()];
        row.extend(interface.as_bytes());
        row.push(u8::try_from(state.len()).unwrap());
        row.extend(state);
        saved.windows(row.len()).any(|window| window == row)
    }
}

/// A request laid out as libnx lays one out (`cmifMakeRequest`), with no payload.
fn libnx_request(command: u32) -> RequestBuilder {
    RequestBuilder::cmif(CommandType::Request, command).layout(RequestLayout::Libnx)
}

/// A whole message buffer that starts with `words` and is zero after them.
fn buffer_of(words: &[u32]) -> Vec<u8> {
    let mut buffer: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    buffer.resize(BUFFER_SIZE, 0);
    buffer
}

/// A reply with one handle, copied or moved, and nothing else.
fn one_handle_reply(handle: u32, copied: bool) -> Vec<u8> {
    let header = CommandHeader {
        command_type: CommandType::Invalid,
        num_x: 0,
        num_a: 0,
        num_b: 0,
        num_w: 0,
        raw_data_words: 4,
        receive_list: ReceiveList::Disabled,
        has_special_header: true,
    };
    let special = SpecialHeader {
        send_pid: false,
        num_copy_handles: u32::from(copied),
        num_move_handles: u32::from(!copied),
    };
    let [w0, w1] = header.encode().unwrap();
    let success = ResultCode::SUCCESS.raw();
    let special = special.encode().unwrap();
    buffer_of(&[w0, w1, special, handle, SFCO_MAGIC, 0, success, 0])
}

/// A reply that holds only a result.
fn only(result: ResultCode) -> Vec<u8> {
    buffer_of(&[0, 6, 0, 0, SFCO_MAGIC, 0, result.raw(), 0])
}

/// The clock source the rig's environment draws first: two draws from a generator with the
/// rig's seed.
fn expected_source() -> ClockSourceId {
    let env = LiveEnv::new(FixedClock::new(0, 0), SEED);
    let low = env.random_u64().unwrap();
    let high = env.random_u64().unwrap();
    let mut bytes = [0; 16];
    bytes[..8].copy_from_slice(&low.to_le_bytes());
    bytes[8..].copy_from_slice(&high.to_le_bytes());
    ClockSourceId(bytes)
}

// ----- the interfaces ---------------------------------------------------------------------------

#[test]
fn each_get_command_hands_out_its_interface_and_the_reply_is_these_bytes() {
    // Each command, the interface it hands out, and that interface's state as `postcard`
    // writes it: a system clock's kind is its variant's index.
    let cases: [(u32, &str, &[u8]); 5] = [
        (GET_STANDARD_USER_SYSTEM_CLOCK, "ISystemClock", &[0]),
        (GET_STANDARD_NETWORK_SYSTEM_CLOCK, "ISystemClock", &[1]),
        (GET_STANDARD_STEADY_CLOCK, "ISteadyClock", &[]),
        (GET_TIME_ZONE_SERVICE, "ITimeZoneService", &[]),
        (GET_STANDARD_LOCAL_SYSTEM_CLOCK, "ISystemClock", &[2]),
    ];
    for (command, interface, state) in cases {
        let mut rig = Rig::new();
        let reply = rig.send_raw(rig.session, &libnx_request(command));
        let message = Message::parse_reply(&reply, SessionKind::Session).unwrap();
        let handle = message.move_handles[0];
        assert_eq!(reply, one_handle_reply(handle, false), "command {command}");
        let session = rig.object_of(handle);
        assert_eq!(rig.kernel.sessions(), [session], "command {command}");
        assert_eq!(rig.server.interface_count(), 2, "command {command}");
        assert!(rig.holds(interface, state), "command {command}");
        // Handing out a clock reads nothing: the clocks are set up only for the memory.
        assert_eq!(rig.env_calls(), [], "command {command}");
        assert_eq!(rig.clocks().set_up(), None, "command {command}");
    }
}

#[test]
fn the_tables_name_the_commands_as_eden_does() {
    let name = |session: &dyn Session, id| session.commands().name(Protocol::Cmif, id);
    let root = [
        (
            GET_STANDARD_USER_SYSTEM_CLOCK,
            "get_standard_user_system_clock",
        ),
        (
            GET_STANDARD_NETWORK_SYSTEM_CLOCK,
            "get_standard_network_system_clock",
        ),
        (GET_STANDARD_STEADY_CLOCK, "get_standard_steady_clock"),
        (GET_TIME_ZONE_SERVICE, "get_time_zone_service"),
        (
            GET_STANDARD_LOCAL_SYSTEM_CLOCK,
            "get_standard_local_system_clock",
        ),
        (
            GET_SHARED_MEMORY_NATIVE_HANDLE,
            "get_shared_memory_native_handle",
        ),
    ];
    for (id, method) in root {
        assert_eq!(name(&IStaticService, id), Some(method));
    }
    assert_eq!(
        name(&ITimeZoneService, TO_CALENDAR_TIME_WITH_MY_RULE),
        Some("to_calendar_time_with_my_rule")
    );
    assert_eq!(ISystemClock::default().commands().cmif, []);
    assert_eq!(ISteadyClock.commands().cmif, []);
    // The startup fixture's trace names the sub-interfaces as Eden does.
    assert_eq!(ISystemClock::interface(), "ISystemClock");
    assert_eq!(ISteadyClock::interface(), "ISteadyClock");
    assert_eq!(ITimeZoneService::interface(), "ITimeZoneService");
}

#[test]
fn commands_that_are_left_out_go_to_the_unknown_command_policy() {
    let mut rig = Rig::new();
    let reply = rig.send_raw(
        rig.session,
        &libnx_request(GET_EPHEMERAL_NETWORK_SYSTEM_CLOCK),
    );
    assert_eq!(reply, only(UNKNOWN_COMMAND_ID));
    let clock = rig.open(GET_STANDARD_USER_SYSTEM_CLOCK);
    assert_eq!(
        rig.send_raw(clock, &libnx_request(GET_CURRENT_TIME)),
        only(UNKNOWN_COMMAND_ID)
    );
    let steady = rig.open(GET_STANDARD_STEADY_CLOCK);
    assert_eq!(
        rig.send_raw(steady, &libnx_request(GET_CURRENT_TIME)),
        only(UNKNOWN_COMMAND_ID)
    );
    let zone = rig.open(GET_TIME_ZONE_SERVICE);
    let request = libnx_request(TO_CALENDAR_TIME).data(&0_i64.to_le_bytes());
    assert_eq!(rig.send_raw(zone, &request), only(UNKNOWN_COMMAND_ID));
}

// ----- the shared memory and the clocks ----------------------------------------------------------

/// Contents with a distinct non-zero value in every field, so a field written to another's
/// place, or a copy written to the other slot, shows.
fn distinct() -> Contents {
    let source = |first: u8| {
        let mut bytes = [0; 16];
        for (byte, value) in bytes.iter_mut().zip(first..) {
            *byte = value;
        }
        ClockSourceId(bytes)
    };
    let point = |time_point, first| SteadyClockTimePoint {
        time_point,
        source_id: source(first),
    };
    Contents {
        steady: point(0x0102_0304_0506_0708, 0x10),
        local: SystemClockContext {
            offset: 0x1112_1314_1516_1718,
            steady_time_point: point(0x2122_2324_2526_2728, 0x30),
        },
        network: SystemClockContext {
            offset: 0x4142_4344_4546_4748,
            steady_time_point: point(0x5152_5354_5556_5758, 0x60),
        },
        automatic_correction: true,
        continuous_adjustment: ContinuousAdjustmentTimePoint {
            rtc_offset: 0x7172_7374_7576_7778,
            diff_scale: -2,
            shift_amount: 14,
            lower: 0x0A0B_0C0D_0E0F_1011,
            upper: 0x1A1B_1C1D_1E1F_2021,
            source_id: source(0x80),
        },
    }
}

/// The golden bytes of [`distinct`]. The offsets are written out from the sources, not computed
/// from this port's constants: `SharedMemoryStruct`'s `static_assert`s (counters at 0x0, 0x38,
/// 0x80, 0xC8 and 0xD0) and `LockFreeAtomicType`'s layout, a `u32` counter then two copies at
/// the value's alignment, the value in copy 1 after one write.
fn golden() -> Vec<u8> {
    let contents = distinct();
    let mut image = vec![0_u8; 0x1000];
    let mut put = |at: usize, bytes: &[u8]| image[at..at + bytes.len()].copy_from_slice(bytes);
    let point = |p: SteadyClockTimePoint| {
        let mut bytes = p.time_point.to_le_bytes().to_vec();
        bytes.extend(p.source_id.0);
        bytes
    };
    let context = |c: SystemClockContext| {
        let mut bytes = c.offset.to_le_bytes().to_vec();
        bytes.extend(point(c.steady_time_point));
        bytes
    };
    for counter in [0x0, 0x38, 0x80, 0xC8, 0xD0] {
        put(counter, &1_u32.to_le_bytes());
    }
    // Steady: copies of 0x18 at 0x8, so copy 1 at 0x20.
    put(0x20, &point(contents.steady));
    // Contexts: copies of 0x20 at 0x40 and 0x88, so copy 1 at 0x60 and 0xA8.
    put(0x60, &context(contents.local));
    put(0xA8, &context(contents.network));
    // The `bool`: copies of one byte at 0xCC, so copy 1 at 0xCD.
    put(0xCD, &[1]);
    // Continuous adjustment: copies of 0x38 at 0xD8, so copy 1 at 0x110.
    let adjustment = contents.continuous_adjustment;
    let mut words = Vec::new();
    for word in [
        adjustment.rtc_offset,
        adjustment.diff_scale,
        adjustment.shift_amount,
        adjustment.lower,
        adjustment.upper,
    ] {
        words.extend(word.to_le_bytes());
    }
    words.extend(adjustment.source_id.0);
    put(0x110, &words);
    image
}

#[test]
fn the_shared_memory_layout_is_pinned_by_golden_bytes() {
    assert_eq!(image(&distinct()), golden());
    // A false flag leaves its byte zero, and the counter is still written.
    let mut off = distinct();
    off.automatic_correction = false;
    let mut expected = golden();
    expected[0xCD] = 0;
    assert_eq!(image(&off), expected);
}

#[test]
fn get_shared_memory_native_handle_copies_the_memory_and_the_reply_is_these_bytes() {
    let mut rig = Rig::new();
    let reply = rig.send_raw(rig.session, &libnx_request(GET_SHARED_MEMORY_NATIVE_HANDLE));
    let message = Message::parse_reply(&reply, SessionKind::Session).unwrap();
    let handle = message.copy_handles[0];
    assert_eq!(reply, one_handle_reply(handle, true));
    let memory = rig.object_of(handle);
    let set_up = rig.clocks().set_up().copied().unwrap();
    assert_eq!(set_up.memory, memory);
    // Mapped read-only, of the size libnx maps, and holding the clocks.
    let (pages, permission) = rig.kernel.shared_memory(memory).unwrap();
    assert_eq!((pages.size(), permission), (SIZE, SvcPermission::Read));
    let mut held = vec![STALE; usize::try_from(SIZE).unwrap()];
    pages.read(0, &mut held).unwrap();
    assert_eq!(held, image(&set_up.contents));
}

#[test]
fn the_clocks_follow_the_settings_and_the_environment() {
    for settings in [
        TimeSettings::eden_default(),
        TimeSettings {
            rtc_at_boot: -86_399,
            automatic_correction: false,
        },
    ] {
        let mut rig = Rig::with(settings, UPTIME);
        rig.memory();
        // Two random draws for the source, then one clock reading.
        assert_eq!(
            rig.env_calls(),
            [EnvCall::Random, EnvCall::Random, EnvCall::Clock]
        );
        let source_id = expected_source();
        let base = settings.rtc_at_boot * NANOSECONDS_PER_SECOND;
        let uptime = i64::try_from(UPTIME).unwrap();
        // The steady clock's seconds at set-up, truncated toward zero as Eden's are.
        let seconds = (base + uptime) / NANOSECONDS_PER_SECOND;
        let context = SystemClockContext {
            offset: 0,
            steady_time_point: SteadyClockTimePoint {
                time_point: seconds,
                source_id,
            },
        };
        let expected = Contents {
            steady: SteadyClockTimePoint {
                time_point: base,
                source_id,
            },
            local: context,
            network: context,
            automatic_correction: settings.automatic_correction,
            continuous_adjustment: ContinuousAdjustmentTimePoint {
                rtc_offset: uptime,
                diff_scale: 0,
                shift_amount: 0,
                lower: base,
                upper: base,
                source_id,
            },
        };
        assert_eq!(rig.clocks().set_up().unwrap().contents, expected);
        assert_eq!(rig.clocks().settings(), &settings);
    }
    // What libnx reads (`_timeReadClockFromSharedMem`) at set-up, under Eden's defaults: the
    // context's offset plus the steady clock's seconds, which are the RTC's at uptime zero plus
    // the whole seconds of uptime.
    let mut rig = Rig::new();
    rig.memory();
    let contents = rig.clocks().set_up().unwrap().contents;
    let steady = contents.steady.time_point + i64::try_from(UPTIME).unwrap();
    let read = contents.local.offset + steady / NANOSECONDS_PER_SECOND;
    assert_eq!(read, TimeSettings::EDEN_INITIAL_YEAR_START + 3);
}

#[test]
fn a_second_request_gets_the_same_memory_and_reads_nothing() {
    let mut rig = Rig::new();
    let first = rig.memory();
    let calls = rig.env_calls().len();
    let contents = rig.clocks().set_up().unwrap().contents;
    assert_eq!(rig.memory(), first);
    assert_eq!(rig.env_calls().len(), calls);
    assert_eq!(rig.clocks().set_up().unwrap().contents, contents);
}

#[test]
fn a_failed_set_up_is_the_reply_and_keeps_nothing() {
    // The kernel refuses the memory: its error, and the next request sets the clocks up.
    let mut rig = Rig::new();
    rig.kernel.refuse_objects(Some(KernelError::OutOfResource));
    let ask = libnx_request(GET_SHARED_MEMORY_NATIVE_HANDLE);
    assert_eq!(
        rig.send_raw(rig.session, &ask),
        only(KernelError::OutOfResource.code())
    );
    assert_eq!(rig.clocks().set_up(), None);
    rig.kernel.refuse_objects(None);
    rig.memory();
    assert!(rig.clocks().set_up().is_some());

    // An environment that has nothing to replay.
    let mut rig = Rig::new();
    rig.server = server(
        TimeSettings::eden_default(),
        UnknownPolicy::Strict,
        Arc::new(ReplayEnv::new(Vec::new())),
    );
    rig.server.accept(rig.session).unwrap();
    assert_eq!(rig.send_raw(rig.session, &ask), only(FAILED));
    assert_eq!(rig.clocks().set_up(), None);
    assert_eq!(rig.kernel.sessions(), []);

    // A clock reading past the nanoseconds an `s64` holds, and one exactly at the limit.
    let limit = u64::try_from(i64::MAX).unwrap();
    let mut rig = Rig::with(TimeSettings::eden_default(), limit + 1);
    assert_eq!(rig.send_raw(rig.session, &ask), only(FAILED));
    assert_eq!(rig.clocks().set_up(), None);
    let settings = TimeSettings {
        rtc_at_boot: 0,
        automatic_correction: true,
    };
    let mut rig = Rig::with(settings, limit);
    rig.memory();
    let contents = rig.clocks().set_up().unwrap().contents;
    assert_eq!(contents.continuous_adjustment.rtc_offset, i64::MAX);
}

#[test]
fn a_base_past_the_nanoseconds_an_i64_holds_is_an_overflow() {
    let source = expected_source();
    let at = |rtc_at_boot, uptime| {
        let settings = TimeSettings {
            rtc_at_boot,
            automatic_correction: true,
        };
        set_up(&settings, source, Uptime(uptime)).map(|c| c.steady.time_point)
    };
    let last = i64::MAX / NANOSECONDS_PER_SECOND;
    let first = i64::MIN / NANOSECONDS_PER_SECOND;
    let overflow = Err(Fault::Result(OVERFLOW));
    // The base: the last and first whole seconds that fit, and one past each.
    assert_eq!(at(last, 0), Ok(last * NANOSECONDS_PER_SECOND));
    assert_eq!(at(last + 1, 0), overflow);
    assert_eq!(at(first, 0), Ok(first * NANOSECONDS_PER_SECOND));
    assert_eq!(at(first - 1, 0), overflow);
    // The base plus the uptime: exactly `i64::MAX`, and one past it.
    let room = i64::MAX - last * NANOSECONDS_PER_SECOND;
    assert!(at(last, room).is_ok());
    assert_eq!(at(last, room + 1), overflow);

    // Through the server: the reply is the overflow, and nothing is kept.
    let settings = TimeSettings {
        rtc_at_boot: last + 1,
        automatic_correction: true,
    };
    let mut rig = Rig::with(settings, 0);
    let reply = rig.send_raw(rig.session, &libnx_request(GET_SHARED_MEMORY_NATIVE_HANDLE));
    assert_eq!(reply, only(OVERFLOW));
    assert_eq!(rig.clocks().set_up(), None);
}

#[test]
fn an_uptime_is_a_reading_that_fits_an_i64() {
    let limit = u64::try_from(i64::MAX).unwrap();
    assert_eq!(
        Uptime::from_reading(HostNanos(limit)).map(Uptime::nanoseconds),
        Some(i64::MAX)
    );
    assert_eq!(Uptime::from_reading(HostNanos(limit + 1)), None);
    assert_eq!(
        Uptime::from_reading(HostNanos(0)).map(Uptime::nanoseconds),
        Some(0)
    );
}

#[test]
fn a_restored_server_hands_out_the_saved_memory_without_reading_again() {
    let mut rig = Rig::new();
    let memory = rig.memory();
    let calls = rig.env_calls().len();
    let saved = rig.server.save().unwrap();
    let env: Arc<dyn Env> = rig.env.clone();
    let mut restored = server(TimeSettings::eden_default(), UnknownPolicy::Strict, env);
    restored.restore(&saved).unwrap();
    assert_eq!(restored.save().unwrap(), saved);
    rig.server = restored;
    assert_eq!(rig.memory(), memory);
    assert_eq!(rig.env_calls().len(), calls);
}

// ----- the calendar --------------------------------------------------------------------------

#[test]
fn to_calendar_time_with_my_rule_answers_in_utc_and_the_reply_is_these_bytes() {
    let mut rig = Rig::new();
    let zone = rig.open(GET_TIME_ZONE_SERVICE);
    // 2000-02-29T01:02:03 UTC, a Tuesday, the 60th day of a leap year.
    let time: i64 = 951_786_123;
    let request = libnx_request(TO_CALENDAR_TIME_WITH_MY_RULE).data(&time.to_le_bytes());
    let reply = rig.send_raw(zone, &request);
    #[rustfmt::skip]
    let data: [u8; REPLY_SIZE] = [
        0xD0, 0x07, 2, 29, 1, 2, 3, 0,        // year 2000, month, day, hour, minute, second, pad
        2, 0, 0, 0,                           // day of week: Tuesday
        59, 0, 0, 0,                          // day of year
        b'U', b'T', b'C', 0, 0, 0, 0, 0,      // the zone's abbreviation
        0, 0, 0, 0,                           // not daylight saving
        0, 0, 0, 0,                           // no offset from UTC
    ];
    let mut words = vec![0, 6 + 8, 0, 0, SFCO_MAGIC, 0, ResultCode::SUCCESS.raw(), 0];
    words.extend(
        data.chunks_exact(4)
            .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap())),
    );
    assert_eq!(reply, buffer_of(&words));
    // Nothing about the clocks is read.
    assert_eq!(rig.env_calls(), []);
}

/// `(year, month, day, hour, minute, second, day of week, day of year)` of a time.
fn fields(time: i64) -> Result<(i16, i8, i8, i8, i8, i8, i32, i32), Fault> {
    to_calendar_time_utc(time).map(|(c, i)| {
        assert_eq!((i.name, i.is_dst, i.ut_offset), (UTC_NAME, 0, 0));
        (
            c.year,
            c.month,
            c.day,
            c.hour,
            c.minute,
            c.second,
            i.day_of_week,
            i.day_of_year,
        )
    })
}

#[test]
fn hand_built_times_convert_as_the_calendar_says() {
    // Each checked against a second calendar implementation, weekdays included.
    let cases = [
        (0, (1970, 1, 1, 0, 0, 0, 4, 0)),                    // a Thursday
        (-1, (1969, 12, 31, 23, 59, 59, 3, 364)),            // a Wednesday, before the epoch
        (951_782_400, (2000, 2, 29, 0, 0, 0, 2, 59)), // a leap day in a year divisible by 400
        (1_672_531_200, (2023, 1, 1, 0, 0, 0, 0, 0)), // a Sunday
        (1_735_689_599, (2024, 12, 31, 23, 59, 59, 2, 365)), // the last day of a leap year
        (4_107_542_400, (2100, 3, 1, 0, 0, 0, 1, 59)), // 2100 is not a leap year
        (253_402_300_799, (9999, 12, 31, 23, 59, 59, 5, 364)),
    ];
    for (time, expected) in cases {
        assert_eq!(fields(time), Ok(expected), "time {time}");
    }
}

/// Divergence from Eden, which wraps a year past `s16` instead (`calendar`'s module docs).
#[test]
fn a_year_outside_the_reply_s_range_is_an_overflow() {
    // The last second of year 32767 and the first of year -32768, counted year by year.
    let last = seconds_of(i64::from(i16::MAX) + 1, 1, 1, 0) - 1;
    let first = seconds_of(i64::from(i16::MIN), 1, 1, 0);
    assert_eq!(
        fields(last).map(|f| (f.0, f.1, f.2)),
        Ok((i16::MAX, 12, 31))
    );
    assert_eq!(fields(first).map(|f| (f.0, f.1, f.2)), Ok((i16::MIN, 1, 1)));
    let overflow = Err(Fault::Result(OVERFLOW));
    assert_eq!(fields(last + 1), overflow);
    assert_eq!(fields(first - 1), overflow);
    assert_eq!(fields(i64::MAX), overflow);
    assert_eq!(fields(i64::MIN), overflow);

    // Through the service: the reply is only the result.
    let mut rig = Rig::new();
    let zone = rig.open(GET_TIME_ZONE_SERVICE);
    let request = libnx_request(TO_CALENDAR_TIME_WITH_MY_RULE).data(&(last + 1).to_le_bytes());
    assert_eq!(rig.send_raw(zone, &request), only(OVERFLOW));
}

/// The seconds since the epoch of a calendar time, by counting days year by year from 1970: an
/// independent check of the conversion, slow but plainly right, for years near the epoch.
fn seconds_of(year: i64, month: i64, day: i64, seconds_into_day: i64) -> i64 {
    let leap = |y: i64| (y % 4 == 0 && y % 100 != 0) || y % 400 == 0;
    let year_days = |y: i64| if leap(y) { 366 } else { 365 };
    let mut days: i64 = if year >= 1970 {
        (1970..year).map(year_days).sum()
    } else {
        -(year..1970).map(year_days).sum::<i64>()
    };
    let months = [
        31,
        if leap(year) { 29 } else { 28 },
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    ];
    days += months[..usize::try_from(month - 1).unwrap()]
        .iter()
        .sum::<i64>();
    days += day - 1;
    days * SECONDS_PER_DAY + seconds_into_day
}

fn config() -> ProptestConfig {
    ProptestConfig {
        cases: 512,
        failure_persistence: None,
        rng_seed: RngSeed::Fixed(0x5449_4D45_5554_4331),
        rng_algorithm: RngAlgorithm::ChaCha,
        ..Config::default()
    }
}

proptest! {
    #![proptest_config(config())]

    /// Every field is in its range and adds back up to the time, for times within 3000 years of
    /// the epoch.
    #[test]
    fn a_time_converts_to_fields_that_add_back_up_to_it(
        time in -95_000_000_000_i64..95_000_000_000,
    ) {
        let (year, month, day, hour, minute, second, week_day, year_day) = fields(time).unwrap();
        prop_assert!((1..=12).contains(&month));
        prop_assert!((1..=31).contains(&day));
        prop_assert!((0..24).contains(&hour) && (0..60).contains(&minute));
        prop_assert!((0..60).contains(&second) && (0..7).contains(&week_day));
        let into_day = i64::from(hour) * 3600 + i64::from(minute) * 60 + i64::from(second);
        let back = seconds_of(i64::from(year), i64::from(month), i64::from(day), into_day);
        prop_assert_eq!(back, time);
        let new_year = seconds_of(i64::from(year), 1, 1, 0);
        prop_assert_eq!(i64::from(year_day), (time - new_year).div_euclid(SECONDS_PER_DAY));
        // 1970-01-01 was a Thursday.
        prop_assert_eq!(i64::from(week_day), (time.div_euclid(SECONDS_PER_DAY) + 4).rem_euclid(7));
    }

    /// Any `i64` is a calendar time or an overflow, never a panic.
    #[test]
    fn any_time_converts_or_overflows(time in any::<i64>()) {
        match fields(time) {
            Ok(_) | Err(Fault::Result(OVERFLOW)) => {}
            Err(other) => prop_assert!(false, "{other:?}"),
        }
    }
}
