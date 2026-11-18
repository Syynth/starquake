//! What a run draws from its environment as it boots (module docs, "The environment"), the
//! recording its report keeps, and a run that replays it: the process's entropy and its main
//! thread's `x18` are drawn once, at creation, in order; a replay of the run gives them back;
//! and a replay that runs out stops the run as an emulator error with the exact words.

use std::cell::RefCell;

use starquake_kernel::svc::info::InfoType;

use super::*;
use crate::EnvSource;

/// How many random numbers a run draws as it boots: the entropy, then the main thread's `x18`.
const BOOT_DRAWS: usize = ENTROPY_COUNT + 1;

/// Boot draws for a replay to hand out instead of the live run's: all different, none of them
/// what the run's seed gives, and the last even, so the bit `x18` gets set shows.
const REPLACED: [u64; BOOT_DRAWS] = [
    0x0101_0000_0000_0011,
    0x0202_0000_0000_0022,
    0x0303_0000_0000_0033,
    0x0404_0000_0000_0044,
    0x0505_0000_0000_0050,
];

/// What a guest saw: its `x18` as the loop started it, then the four entropy words
/// `GetInfo(RandomEntropy)` answered, in sub-id order.
type Seen = Rc<RefCell<Vec<u64>>>;

/// A guest that notes its `x18`, asks for each entropy word in turn and notes it, then exits.
fn entropy_guest(seen: &Seen) -> ScriptedVcpu {
    let ask = |regs: &mut GuestRegs, sub_id: u64| {
        regs.x[1] = u64::from(InfoType::RandomEntropy.number());
        regs.x[2] = u64::from(Handle::INVALID.raw());
        regs.x[3] = sub_id;
    };
    let first = Rc::clone(seen);
    let mut vcpu = ScriptedVcpu::new().then(
        move |regs| {
            // `x18` by its literal index, not through `CFI_REGISTER`, so a wrong constant
            // fails the replay tests: `ctx.r[18]` in `ResetThreadContext64`
            // (`eden/src/core/hle/kernel/k_thread.cpp` @ cf654052, line 61).
            first.borrow_mut().push(regs.x[18]);
            ask(regs, 0);
        },
        svc(SvcId::GetInfo),
    );
    for sub_id in 1..=ENTROPY_COUNT {
        let seen = Rc::clone(seen);
        let next = u64::try_from(sub_id).unwrap();
        let exit = if sub_id == ENTROPY_COUNT {
            svc(SvcId::ExitProcess)
        } else {
            svc(SvcId::GetInfo)
        };
        vcpu = vcpu.then(
            move |regs| {
                assert_eq!(regs.x[0], 0, "GetInfo succeeded");
                seen.borrow_mut().push(regs.x[1]);
                ask(regs, next);
            },
            exit,
        );
    }
    vcpu
}

/// Runs [`entropy_guest`] with `options`: the report, and what the guest saw.
fn run_entropy_guest(options: &RunOptions) -> (RunReport, Vec<u64>) {
    let guest = Guest::new(&programs::hello().unwrap());
    let seen = Seen::default();
    let report = guest.run(entropy_guest(&seen), options);
    let seen = seen.borrow().clone();
    (report, seen)
}

/// Options that replay `records`.
fn replaying(records: Vec<EnvRecord>) -> RunOptions {
    RunOptions {
        environment: EnvSource::Replay(records),
        ..RunOptions::default()
    }
}

/// The emulator error a run ended with.
fn emulator_error(report: &RunReport) -> &str {
    match &report.exit {
        ExitReason::EmulatorError(message) => message,
        other => unreachable!("not an emulator error: {other:?}"),
    }
}

#[test]
fn a_live_run_draws_the_entropy_then_x18_first_and_the_guest_sees_them() {
    let (report, seen) = run_entropy_guest(&RunOptions::default());
    assert_eq!(report.exit, ExitReason::ProcessExited);
    let numbers = boot_draws();
    // The report's recording starts with the five draws, in order, and nothing was cut.
    let records: Vec<EnvRecord> = numbers.into_iter().map(EnvRecord::Random).collect();
    assert_eq!(report.environment[..BOOT_DRAWS], records[..]);
    assert_eq!(report.environment_dropped, 0);
    // No other random number is drawn: the GetInfo calls read what creation drew.
    let randoms = report
        .environment
        .iter()
        .filter(|record| matches!(record, EnvRecord::Random(_)))
        .count();
    assert_eq!(randoms, BOOT_DRAWS);
    // The guest's x18 is the fifth with its bit set, and its entropy the first four.
    let mut expected = vec![boot_x18()];
    expected.extend(&numbers[..ENTROPY_COUNT]);
    assert_eq!(seen, expected);
    assert_eq!(report.exit_code(), exit_code::STOP_CONDITION);
}

#[test]
fn a_replay_of_a_run_gives_the_guest_the_same_draws() {
    let (live, live_seen) = run_entropy_guest(&RunOptions::default());
    assert_eq!(live.exit, ExitReason::ProcessExited);
    let (replayed, seen) = run_entropy_guest(&replaying(live.environment.clone()));
    assert_eq!(replayed.exit, ExitReason::ProcessExited);
    assert_eq!(seen, live_seen);
    // A replay records nothing.
    assert_eq!(replayed.environment, []);
    assert_eq!(replayed.environment_dropped, 0);
}

#[test]
fn the_draws_come_from_the_replay_not_the_seed() {
    // The same recording with the five draws replaced: the guest sees the replacements, so
    // they came from the recording, not from a live environment of the same seed.
    let (live, _) = run_entropy_guest(&RunOptions::default());
    let mut records = live.environment;
    for (record, word) in records.iter_mut().zip(REPLACED) {
        assert!(matches!(record, EnvRecord::Random(_)));
        *record = EnvRecord::Random(word);
    }
    let (replayed, seen) = run_entropy_guest(&replaying(records));
    assert_eq!(replayed.exit, ExitReason::ProcessExited);
    let mut expected = vec![REPLACED[ENTROPY_COUNT] | CFI_VALUE_SET_BITS];
    expected.extend(&REPLACED[..ENTROPY_COUNT]);
    assert_eq!(seen, expected);
    assert_eq!(seen[0], 0x0505_0000_0000_0051);
}

#[test]
fn a_replay_missing_a_draw_stops_the_run_as_this_emulator_error() {
    // Every length short of the five draws: the run ends before the guest runs, with the
    // kernel's words for the draw that found nothing.
    let records: Vec<EnvRecord> = REPLACED.into_iter().map(EnvRecord::Random).collect();
    for given in 0..BOOT_DRAWS {
        let (report, seen) = run_entropy_guest(&replaying(records[..given].to_vec()));
        let what = if given < ENTROPY_COUNT {
            "the process's entropy"
        } else {
            "the main thread's x18"
        };
        assert_eq!(
            emulator_error(&report),
            format!(
                "the kernel could not be set up: the environment failed while {what} was \
                 drawn: recording exhausted at record {given}: the replay asked for Random"
            ),
            "{given}"
        );
        assert_eq!(report.exit_code(), exit_code::EMULATOR_ERROR);
        assert_eq!(seen, Vec::<u64>::new(), "the guest never ran");
        assert_eq!(report.system_calls, BTreeMap::new());
    }
}

#[test]
fn a_replay_whose_next_record_is_not_a_draw_stops_the_run_as_this_emulator_error() {
    // The recording disagrees rather than running out: a clock reading where creation draws.
    let (report, seen) = run_entropy_guest(&replaying(vec![EnvRecord::Clock(HostNanos(1))]));
    assert_eq!(
        emulator_error(&report),
        "the kernel could not be set up: the environment failed while the process's entropy \
         was drawn: record 0 answers Clock but the replay asked for Random"
    );
    assert_eq!(seen, Vec::<u64>::new());
}

#[test]
fn a_replay_that_runs_out_after_boot_stops_the_run_at_the_call_that_found_nothing() {
    // The five draws and the first call's clock reading: the guest's second call finds the
    // recording used up, and the run stops there as an emulator error.
    let (live, _) = run_entropy_guest(&RunOptions::default());
    let records = live.environment[..=BOOT_DRAWS].to_vec();
    assert!(matches!(records[BOOT_DRAWS], EnvRecord::Clock(_)));
    let (report, seen) = run_entropy_guest(&replaying(records));
    assert_eq!(
        emulator_error(&report),
        format!(
            "the environment failed during svc {:#x}: recording exhausted at record {}: the \
             replay asked for Clock",
            SvcId::GetInfo.number(),
            BOOT_DRAWS + 1
        )
    );
    // The guest saw its x18 and the first word, and nothing after.
    assert_eq!(seen.len(), 2);
}

#[test]
fn a_replay_without_the_date_stops_the_run_before_the_services_are_installed() {
    // The run reads the host's date after the process's draws; a recording of only the draws
    // has nothing for it. This is the branch a live run cannot reach (module docs, "The run's
    // date").
    let records: Vec<EnvRecord> = REPLACED.into_iter().map(EnvRecord::Random).collect();
    let options = RunOptions {
        rtc: RtcAtBoot::Host,
        ..replaying(records)
    };
    let (report, seen) = run_entropy_guest(&options);
    assert_eq!(
        emulator_error(&report),
        format!(
            "the run's date could not be read: recording exhausted at record {BOOT_DRAWS}: the \
             replay asked for WallClock"
        )
    );
    assert_eq!(report.date, None);
    assert_eq!(report.services, ServicesReport::default());
    assert_eq!(seen, Vec::<u64>::new());
}

#[test]
fn a_replay_of_the_hosts_date_gives_the_run_that_date() {
    // A live run that read the host's date, replayed: the recorded reading is the run's date.
    let options = RunOptions {
        rtc: RtcAtBoot::Host,
        ..RunOptions::default()
    };
    let (live, _) = run_entropy_guest(&options);
    let mut records = live.environment;
    let EnvRecord::WallClock(_) = records[BOOT_DRAWS] else {
        unreachable!("the date is read right after the draws: {records:?}");
    };
    // A date no host would read today, so it can only have come from the recording.
    let recorded = UnixSeconds(86_400);
    records[BOOT_DRAWS] = EnvRecord::WallClock(recorded);
    let (replayed, _) = run_entropy_guest(&RunOptions {
        rtc: RtcAtBoot::Host,
        ..replaying(records)
    });
    assert_eq!(replayed.exit, ExitReason::ProcessExited);
    assert_eq!(replayed.date, Some(RunDate::new(RtcAtBoot::Host, recorded)));
}

#[test]
fn a_report_keeps_a_prefix_of_the_recording_up_to_its_limit_and_its_replay_runs_out_there() {
    // Room for exactly the five draws and one clock reading: the rest is counted, not kept.
    let kept_records = BOOT_DRAWS + 1;
    let limits = Limits {
        env_recording_bytes: kept_records * size_of::<EnvRecord>(),
        ..Limits::DEFAULT
    };
    let guest = Guest::new(&programs::hello().unwrap());
    let seen = Seen::default();
    let mut vcpu = entropy_guest(&seen);
    let report = run_within(
        vcpus(&mut vcpu),
        guest.program(&guest.memory),
        &RunOptions::default(),
        Instant::now(),
        timebase(),
        limits,
    );
    assert_eq!(report.exit, ExitReason::ProcessExited);
    assert_eq!(report.environment.len(), kept_records);
    // The other four calls' clock readings (three GetInfo and the ExitProcess).
    assert_eq!(report.environment_dropped, 4);
    // Replayed, it runs out at the second call, as a recording cut there would.
    let (replayed, _) = run_entropy_guest(&replaying(report.environment));
    assert_eq!(
        emulator_error(&replayed),
        format!(
            "the environment failed during svc {:#x}: recording exhausted at record \
             {kept_records}: the replay asked for Clock",
            SvcId::GetInfo.number()
        )
    );
}

#[test]
fn a_run_whose_services_could_not_be_set_up_still_keeps_what_it_drew() {
    // The loop never ran, so nothing moved the draws or the date out of the environment; the
    // run's end does.
    let guest = Guest::new(&programs::hello().unwrap());
    let mut vcpu = ScriptedVcpu::new();
    let options = RunOptions {
        rtc: RtcAtBoot::Host,
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
        emulator_error(&report),
        "the services could not be set up: no room"
    );
    let drawn: Vec<EnvRecord> = boot_draws().into_iter().map(EnvRecord::Random).collect();
    assert_eq!(report.environment[..BOOT_DRAWS], drawn[..]);
    assert!(matches!(
        report.environment[BOOT_DRAWS..],
        [EnvRecord::WallClock(_)]
    ));
}
