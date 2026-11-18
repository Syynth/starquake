//! The run's environment and what it keeps of its recording.

use std::sync::Arc;

use starquake_trace::{EnvCall, FixedClock};

use super::*;

/// A live environment whose clock reads 0, 1, 2, ... and whose random numbers come from seed 7.
fn live() -> LiveEnv {
    LiveEnv::new(FixedClock::new(0, 1), 7)
}

/// A clock record of reading `n`.
fn clock(n: u64) -> EnvRecord {
    EnvRecord::Clock(HostNanos(n))
}

/// A host I/O record whose request carries `request` bytes and whose reply carries `reply`.
fn host_io(request: usize, reply: usize) -> EnvRecord {
    EnvRecord::HostIo {
        request: HostIoRequest {
            channel: 3,
            payload: vec![0xab; request],
        },
        reply: HostIoReply::Data(vec![0xcd; reply]),
    }
}

/// The bytes one record that carries nothing counts for.
fn one() -> usize {
    size_of::<EnvRecord>()
}

#[test]
fn a_record_counts_its_own_size_and_the_bytes_it_carries() {
    assert_eq!(record_bytes(&clock(1)), one());
    assert_eq!(record_bytes(&EnvRecord::Random(9)), one());
    assert_eq!(record_bytes(&EnvRecord::WallClock(UnixSeconds(5))), one());
    assert_eq!(record_bytes(&host_io(3, 0)), one() + 3);
    assert_eq!(record_bytes(&host_io(0, 11)), one() + 11);
    assert_eq!(record_bytes(&host_io(3, 11)), one() + 14);
    let failed = EnvRecord::HostIo {
        request: HostIoRequest {
            channel: 3,
            payload: vec![1; 2],
        },
        reply: HostIoReply::Failed(4),
    };
    assert_eq!(record_bytes(&failed), one() + 2);
}

#[test]
fn records_are_kept_up_to_exactly_the_limit() {
    // Three records fit exactly; the fourth is one record past the limit.
    let mut kept = Kept::new(3 * one());
    kept.push(vec![clock(0), clock(1)]);
    kept.push(vec![clock(2), clock(3)]);
    assert_eq!(kept.records(), [clock(0), clock(1), clock(2)]);
    assert_eq!(kept.dropped(), 1);

    // One byte less: only two fit.
    let mut short = Kept::new(3 * one() - 1);
    short.push(vec![clock(0), clock(1), clock(2)]);
    assert_eq!(short.records(), [clock(0), clock(1)]);
    assert_eq!(short.dropped(), 1);
}

#[test]
fn what_is_kept_is_a_prefix() {
    // A host I/O record too large to fit is dropped, and so is every record after it, even
    // one that would fit on its own: a replay of what is kept must be what was answered, from
    // the first.
    let mut kept = Kept::new(3 * one());
    kept.push(vec![clock(0), host_io(0, 2 * one()), clock(1)]);
    kept.push(vec![clock(2)]);
    assert_eq!(kept.records(), [clock(0)]);
    assert_eq!(kept.dropped(), 3);
}

#[test]
fn a_host_io_record_that_fits_exactly_is_kept() {
    let mut kept = Kept::new(2 * one() + 5);
    kept.push(vec![clock(0), host_io(2, 3), clock(1)]);
    assert_eq!(kept.records(), [clock(0), host_io(2, 3)]);
    assert_eq!(kept.dropped(), 1);
}

#[test]
fn a_live_run_env_answers_from_the_host_and_keeps_what_it_moved() {
    let env = RunEnv::new(&EnvSource::Live, live, ENV_RECORDING_BYTES);
    assert_eq!(env.now(), Ok(HostNanos(0)));
    let random = env.random_u64().unwrap();
    // Recorded, not yet kept.
    assert_eq!(env.recording(), [clock(0), EnvRecord::Random(random)]);
    assert_eq!(env.kept(), (Vec::new(), 0));
    env.keep();
    assert_eq!(env.recording(), []);
    assert_eq!(env.kept(), (vec![clock(0), EnvRecord::Random(random)], 0));
    // A second keep appends what came after.
    assert_eq!(env.now(), Ok(HostNanos(1)));
    env.keep();
    assert_eq!(
        env.kept().0,
        [clock(0), EnvRecord::Random(random), clock(1)]
    );
}

#[test]
fn a_live_run_env_keeps_no_more_than_its_limit() {
    let env = RunEnv::new(&EnvSource::Live, live, 2 * one());
    for n in 0..5 {
        assert_eq!(env.now(), Ok(HostNanos(n)));
    }
    env.keep();
    assert_eq!(env.kept(), (vec![clock(0), clock(1)], 3));
}

#[test]
fn a_replay_answers_from_its_records_and_keeps_nothing() {
    let records = vec![clock(40), EnvRecord::Random(41)];
    let env = RunEnv::new(
        &EnvSource::Replay(records),
        || panic!("a replay reads no host"),
        ENV_RECORDING_BYTES,
    );
    assert_eq!(env.now(), Ok(HostNanos(40)));
    assert_eq!(env.random_u64(), Ok(41));
    env.keep();
    assert_eq!(env.kept(), (Vec::new(), 0));
    assert_eq!(env.recording(), []);
    // Run out: the exact error, every time after.
    let exhausted = EnvError::Exhausted {
        index: 2,
        requested: EnvCall::WallClock,
    };
    assert_eq!(env.wall_clock(), Err(exhausted.clone()));
    assert_eq!(env.now(), Err(exhausted));
}

#[test]
fn a_replay_of_what_a_live_run_env_kept_gives_the_same_answers() {
    let live_env = RunEnv::new(&EnvSource::Live, live, ENV_RECORDING_BYTES);
    let answers = (
        live_env.random_u64().unwrap(),
        live_env.now().unwrap(),
        live_env.wall_clock().unwrap(),
        live_env.host_io(&HostIoRequest {
            channel: 2,
            payload: vec![5],
        }),
    );
    live_env.keep();
    let (kept, dropped) = live_env.kept();
    assert_eq!(dropped, 0);
    let replay = RunEnv::new(
        &EnvSource::Replay(kept),
        || panic!("a replay reads no host"),
        ENV_RECORDING_BYTES,
    );
    let replayed = (
        replay.random_u64().unwrap(),
        replay.now().unwrap(),
        replay.wall_clock().unwrap(),
        replay.host_io(&HostIoRequest {
            channel: 2,
            payload: vec![5],
        }),
    );
    assert_eq!(replayed, answers);
}

#[test]
fn keeping_from_several_threads_keeps_the_answers_in_the_order_given() {
    // Each thread reads the clock and keeps, many times over, as the cores and the timer
    // thread do. The clock counts up by one per reading, under the environment's lock, so the
    // order the answers were given in is the order of their values: what is kept must count up
    // by one from zero. Taking outside the kept recording's lock lets two takes append out of
    // order, which this catches.
    const THREADS: u64 = 4;
    const READS: u64 = 2_000;
    let env = Arc::new(RunEnv::new(&EnvSource::Live, live, ENV_RECORDING_BYTES));
    std::thread::scope(|scope| {
        for _ in 0..THREADS {
            let env = Arc::clone(&env);
            scope.spawn(move || {
                for _ in 0..READS {
                    env.now().unwrap();
                    env.keep();
                }
            });
        }
    });
    let (kept, dropped) = env.kept();
    assert_eq!(dropped, 0);
    let expected: Vec<EnvRecord> = (0..THREADS * READS).map(clock).collect();
    assert!(kept == expected, "the kept recording is out of order");
}
