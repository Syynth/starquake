//! The run's date (module docs, "The run's date"): read once as the run boots, from the
//! environment's wall clock unless the run's settings pin one, and given to `time:u` through a
//! real run's services.

use std::time::{SystemTime, UNIX_EPOCH};

use starquake_services::framework::Server;
use starquake_services::time::Clocks;
use starquake_trace::{EnvCall, EnvError, ReplayEnv, UnixSeconds};

use super::*;
use crate::hle::TIME_U;

/// What a run's services saw as they were installed: the run's environment's recording so far,
/// and the settings `time:u`'s server was installed with.
type Seen = (Vec<EnvRecord>, Option<TimeSettings>);

/// Runs a guest that exits at once, with the run's date from `rtc`, through a real run's
/// services ([`real_services`]); how the run ended and what the services saw.
fn run_dated(rtc: RtcAtBoot) -> (Ending, Seen) {
    let guest = Guest::new(&programs::hello().unwrap());
    let options = RunOptions {
        rtc,
        ..RunOptions::default()
    };
    let session = Session {
        memory: &guest.memory,
        options: &options,
        clock: RunClock::new(timebase()),
        recorder: Recorder::new(),
    };
    let env = session.environment();
    let watched = Arc::clone(&env);
    let seen = Arc::new(Mutex::new(None));
    let kept = Arc::clone(&seen);
    let install = move |dispatcher: &Dispatcher, given: Given| {
        let read = watched.recording();
        real_services(dispatcher, given)?;
        let mut settings = None;
        dispatcher.inspect_handlers(|service, handler| {
            let handler: &dyn core::any::Any = handler;
            if service == TIME_U {
                settings = handler
                    .downcast_ref::<Server>()
                    .and_then(|server| server.family::<Clocks>())
                    .map(|clocks| *clocks.settings());
            }
        });
        *kept.lock().unwrap() = Some((read, settings));
        Ok(())
    };
    let mut vcpu = ScriptedVcpu::new().then(|_| {}, svc(SvcId::ExitProcess));
    let (dispatcher, _) = boot(guest.address_space(), &guest.placed, &guest.memory).unwrap();
    let (ending, ..) = drive(
        vcpus(&mut vcpu),
        dispatcher,
        &guest.memory,
        &session,
        &env,
        Limits::DEFAULT,
        &install,
    );
    let seen = seen
        .lock()
        .unwrap()
        .take()
        .expect("the services were installed");
    (ending, seen)
}

/// The host's seconds since the epoch now, read here independently of the code under test.
fn host_seconds() -> i64 {
    let since = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
    i64::try_from(since.as_secs()).unwrap()
}

#[test]
fn a_pinned_date_reaches_time_u_and_the_host_is_not_read() {
    for date in [-1, 0, 1_709_164_800] {
        let (ending, (read, settings)) = run_dated(RtcAtBoot::Pinned(UnixSeconds(date)));
        assert!(matches!(ending, Ending::Exited), "{date}");
        assert_eq!(read, [], "{date}");
        assert_eq!(settings, Some(TimeSettings::with_rtc(UnixSeconds(date))));
    }
    // A run that does not say pins Eden's date.
    let (_, (read, settings)) = run_dated(RtcAtBoot::default());
    assert_eq!(read, []);
    assert_eq!(settings, Some(TimeSettings::eden_default()));
}

#[test]
fn the_hosts_date_reaches_time_u_through_a_real_runs_services() {
    // The real environment over the host's wall clock: read once, first, before the services
    // are installed, and recorded.
    let before = host_seconds();
    let (ending, (read, settings)) = run_dated(RtcAtBoot::Host);
    let after = host_seconds();
    assert!(matches!(ending, Ending::Exited));
    let [EnvRecord::WallClock(UnixSeconds(date))] = read[..] else {
        unreachable!("not one reading of the date: {read:?}");
    };
    assert!(before <= date && date <= after, "{before} {date} {after}");
    assert_eq!(settings, Some(TimeSettings::with_rtc(UnixSeconds(date))));
}

#[test]
fn a_replay_without_the_date_is_this_emulator_error() {
    let missing = ReplayEnv::new(Vec::new());
    assert_eq!(
        date_at_boot(RtcAtBoot::Host, &missing),
        Err(
            "the run's date could not be read: recording exhausted at record 0: the replay \
             asked for WallClock"
                .to_owned()
        )
    );
    let other = ReplayEnv::new(vec![EnvRecord::Clock(HostNanos(1))]);
    assert_eq!(
        date_at_boot(RtcAtBoot::Host, &other),
        Err(
            "the run's date could not be read: record 0 answers Clock but the replay asked for \
             WallClock"
                .to_owned()
        )
    );
    // The two errors are the environment's own.
    assert_eq!(
        missing.first_error(),
        Some(EnvError::Exhausted {
            index: 0,
            requested: EnvCall::WallClock,
        })
    );
    // A pinned date reads nothing, so a recording with no reading in it replays.
    let empty = ReplayEnv::new(Vec::new());
    assert_eq!(
        date_at_boot(RtcAtBoot::Pinned(UnixSeconds(9)), &empty),
        Ok(TimeSettings::with_rtc(UnixSeconds(9)))
    );
    assert_eq!(empty.finish(), Ok(()));
}
