//! The run's date (module docs, "The run's date"): read once as the run boots, from the
//! environment's wall clock unless the run's settings pin one, and given to `time:u` through a
//! real run's services.

use std::time::{SystemTime, UNIX_EPOCH};

use starquake_services::framework::Server;
use starquake_services::time::Clocks;
use starquake_trace::{EnvCall, EnvError, ReplayEnv, UnixSeconds};

use super::*;
use crate::hle::TIME_U;
use crate::report::DateSource;

/// How many random numbers a run draws as it boots, before the date: the process's entropy and
/// its main thread's `x18`.
const BOOT_DRAWS: usize = starquake_kernel::process::ENTROPY_COUNT + 1;

/// What a run's services saw as they were installed: the run's environment's recording so far,
/// after the process's draws at boot (its entropy and its main thread's `x18`, which come first
/// and are all random numbers), and the settings `time:u`'s server was installed with.
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
        let mut read = watched.recording();
        let after_draws = read.split_off(BOOT_DRAWS);
        assert!(
            read.iter()
                .all(|record| matches!(record, EnvRecord::Random(_))),
            "{read:?}"
        );
        let read = after_draws;
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
    let (dispatcher, _) = boot(guest.address_space(), &guest.placed, &guest.memory, &*env).unwrap();
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

#[test]
fn a_failed_read_of_the_date_ends_the_run_before_the_services_are_installed() {
    // The read fails as a replay's would: the run ends with the read's message, and the
    // services are never installed.
    let missing = ReplayEnv::new(Vec::new());
    let mut installed = Vec::new();
    let failed = set_up_services(date_at_boot(RtcAtBoot::Host, &missing), |time| {
        installed.push(time);
        Ok(())
    });
    let Err((Ending::Failed(message), None)) = failed else {
        unreachable!("not a failed run with no date");
    };
    assert_eq!(
        message,
        "the run's date could not be read: recording exhausted at record 0: the replay asked \
         for WallClock"
    );
    assert_eq!(installed, []);

    // A date that was read is what the services are installed with, and what is returned.
    let read = TimeSettings::with_rtc(UnixSeconds(-86_399));
    let started = set_up_services(Ok(read), |time| {
        installed.push(time);
        Ok(())
    });
    assert!(matches!(started, Ok(time) if time == read));
    assert_eq!(installed, [read]);

    // An install that fails ends the run with its own words, and keeps the date it was given.
    let refused = set_up_services(Ok(read), |_| Err("no room".to_owned()));
    let Err((Ending::Failed(message), Some(time))) = refused else {
        unreachable!("not a failed run with its date");
    };
    assert_eq!(message, "the services could not be set up: no room");
    assert_eq!(time, read);
}

#[test]
fn a_runs_report_keeps_its_date_and_where_it_came_from() {
    // A pinned date, through a real run's services.
    let guest = Guest::new(&programs::hello().unwrap());
    let vcpu = ScriptedVcpu::new().then(|_| {}, svc(SvcId::ExitProcess));
    let options = RunOptions {
        rtc: RtcAtBoot::Pinned(UnixSeconds(-86_399)),
        ..RunOptions::default()
    };
    let report = guest.run(vcpu, &options);
    assert_eq!(report.exit, ExitReason::ProcessExited);
    assert_eq!(
        report.date,
        Some(RunDate {
            rtc_at_boot: UnixSeconds(-86_399),
            source: DateSource::Pinned,
        })
    );

    // The host's date, read on the production environment over the host's wall clock.
    let guest = Guest::new(&programs::hello().unwrap());
    let vcpu = ScriptedVcpu::new().then(|_| {}, svc(SvcId::ExitProcess));
    let options = RunOptions {
        rtc: RtcAtBoot::Host,
        ..RunOptions::default()
    };
    let before = host_seconds();
    let report = guest.run(vcpu, &options);
    let after = host_seconds();
    assert_eq!(report.exit, ExitReason::ProcessExited);
    let Some(RunDate {
        rtc_at_boot: UnixSeconds(date),
        source: DateSource::Host,
    }) = report.date
    else {
        unreachable!("not the host's date: {:?}", report.date);
    };
    assert!(before <= date && date <= after, "{before} {date} {after}");

    // A run whose services could not be set up still says the date it read.
    let guest = Guest::new(&programs::hello().unwrap());
    let mut vcpu = ScriptedVcpu::new();
    let options = RunOptions {
        rtc: RtcAtBoot::Pinned(UnixSeconds(7)),
        ..RunOptions::default()
    };
    let refuse = |_: &Dispatcher, _: Given| Err("no room".to_owned());
    let report = run_serving(
        vcpus(&mut vcpu),
        guest.program(&guest.memory),
        &options,
        (Instant::now(), timebase()),
        Limits::DEFAULT,
        &refuse,
    );
    assert_eq!(
        report.exit,
        ExitReason::EmulatorError("the services could not be set up: no room".to_owned())
    );
    assert_eq!(
        report.date,
        Some(RunDate {
            rtc_at_boot: UnixSeconds(7),
            source: DateSource::Pinned,
        })
    );
}
