//! Idleness with a channel parked on an acquire (the wiring's module docs, "Idleness"): on the
//! production front clock, and on one held still so that, once the poll's spins are over, no
//! poll runs until the test moves it.

use std::sync::atomic::AtomicU64;

use starquake_render::FenceValue;

use super::*;
use crate::front::POLL_SPINS;

/// A front clock that moves only when told. While it watches a front, it notes at each reading
/// whether that front was quiet: the front reads its clock only inside a turn.
#[derive(Default)]
struct StillClock {
    nanos: AtomicU64,
    watching: Mutex<Option<Arc<FrontShared>>>,
    quiet_seen: Mutex<Vec<bool>>,
}

impl StillClock {
    fn advance(&self, by: Duration) {
        self.nanos
            .fetch_add(u64::try_from(by.as_nanos()).unwrap(), Ordering::SeqCst);
    }

    fn watch(&self, shared: Option<Arc<FrontShared>>) {
        *self.watching.lock().unwrap() = shared;
    }
}

impl FrontClock for StillClock {
    fn now(&self) -> Duration {
        let watching = self.watching.lock().unwrap().clone();
        if let Some(shared) = watching {
            let quiet = shared.is_quiet();
            self.quiet_seen.lock().unwrap().push(quiet);
        }
        Duration::from_nanos(self.nanos.load(Ordering::SeqCst))
    }
}

/// How many times a test asks a question whose wrong answer depends on timing, so that a broken
/// rule is caught on at least one of them.
const ASKS: usize = 20;

/// The polls the front had finished at the end of its last turn, as it published them.
fn polls(r: &Rig) -> u64 {
    r.gpu.status.published().counters.polls
}

/// Makes `SPACE` and a channel, submits a stream that waits for `SEM` to hold 1 (it holds 0)
/// and then releases 2 at `SEM2`, and waits until the channel is parked, the poll's spins are
/// over and the front quiet.
///
/// The first [`POLL_SPINS`] polls after the park are due as soon as they are scheduled
/// ([`crate::front::Backoff::next_sleep`] returns zero for them), so they run on a still clock
/// too, each in an idle turn that marks the front quiet. Only the polls after them wait for the
/// clock to move: a test that holds the clock still has no poll to fear only from then on.
fn park_a_channel(r: &Rig) {
    r.space();
    let channel = r.channel();
    r.gpu.submit(
        channel,
        plain(vec![r.entry(CMD, &acquires_then_releases(SEM, 1, SEM2, 2))]),
    );
    wait_until(
        "the channel parks, the poll's spins end and the front ends a turn idle",
        || {
            let published = r.gpu.status.published();
            published.counters.parks == 1
                && published.waiting
                && published.counters.polls >= u64::from(POLL_SPINS)
                && r.gpu.shared.is_quiet()
        },
    );
}

#[test]
fn a_turn_the_poll_starts_is_busy() {
    // The poll falls due only when the test moves the clock: the turn it starts takes nothing,
    // and the front must read busy all through it.
    let clock = Arc::new(StillClock::default());
    let r = rig_with(false, Arc::<StillClock>::clone(&clock));
    park_a_channel(&r);
    assert_eq!(polls(&r), u64::from(POLL_SPINS), "only the spins polled");
    clock.watch(Some(Arc::clone(&r.gpu.shared)));
    clock.advance(POLL_LONGEST_SLEEP);
    wait_until("a poll runs", || polls(&r) > u64::from(POLL_SPINS));
    clock.watch(None);
    let seen = clock.quiet_seen.lock().unwrap().clone();
    assert_ne!(seen.len(), 0, "the clock was read in a turn");
    assert!(seen.iter().all(|&quiet| !quiet), "{seen:?}");
    assert_eq!(r.word(SEM2), 0, "the acquire still waits");
    r.gpu.stop().unwrap();
}

#[test]
fn a_released_acquire_is_busy_until_a_poll_has_looked() {
    // The CPU writes the value the parked acquire waits for, and no poll can run: the GPU is
    // not idle, since the next poll resumes the channel.
    let clock = Arc::new(StillClock::default());
    let r = rig_with(false, Arc::<StillClock>::clone(&clock));
    park_a_channel(&r);
    r.write(SEM, &[1]);
    for _ in 0..ASKS {
        assert!(!r.gpu.is_idle(), "no poll has looked since the CPU's write");
    }
    assert_eq!(
        polls(&r),
        u64::from(POLL_SPINS),
        "no poll ran after the spins"
    );
    assert!(r.gpu.status.published().waiting, "the channel still waits");
    assert_eq!(r.word(SEM2), 0, "the acquire still waits");
    clock.advance(POLL_LONGEST_SLEEP);
    wait_until("the channel resumes and releases", || r.word(SEM2) == 2);
    wait_until("the GPU is idle", || r.gpu.is_idle());
    r.gpu.stop().unwrap();
}

#[test]
fn a_channel_parked_on_an_unreleased_acquire_is_idle() {
    // On the host clock: the polls find nothing, so only the guest can end the wait.
    let r = rig();
    park_a_channel(&r);
    wait_until("the GPU is idle", || r.gpu.is_idle());
    assert_eq!(r.word(SEM2), 0);
    assert!(r.gpu.status.published().waiting);
    // And the CPU's write still resumes it.
    r.write(SEM, &[1]);
    wait_until("the channel resumes and releases", || r.word(SEM2) == 2);
    wait_until("the GPU is idle again", || r.gpu.is_idle());
    assert!(!r.gpu.status.published().waiting);
    r.gpu.stop().unwrap();
}

#[test]
fn a_cpu_release_of_a_parked_acquire_is_busy_through_the_threads() {
    // The production arrangement: the host clock, and a present behind the parked submission
    // whose list the late backend holds in flight, so the GPU stays busy from the CPU's write
    // until the test finishes the list, whenever the poll runs.
    let r = rig_on(true);
    park_a_channel(&r);
    r.gpu.present(present_request());
    wait_until("the GPU is idle with the present held", || r.gpu.is_idle());
    r.write(SEM, &[1]);
    for _ in 0..ASKS {
        assert!(!r.gpu.is_idle(), "the CPU released the parked acquire");
    }
    wait_until("the present's list is submitted", || {
        r.late().submitted().len() == 1
    });
    assert_eq!(r.word(SEM2), 2);
    assert!(!r.gpu.is_idle());
    r.late().finish(FenceValue::new(1));
    wait_until("the GPU is idle", || r.gpu.is_idle());
    r.gpu.stop().unwrap();
}

/// A quiet wake that counts its calls.
#[derive(Default)]
struct QuietCount(AtomicU64);

impl QuietCount {
    fn count(&self) -> u64 {
        self.0.load(Ordering::SeqCst)
    }
}

impl QuietWake for QuietCount {
    fn front_quiet(&self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

/// The quiet signal through the production front thread on the host clock (the wiring's module
/// docs, "Idleness", "The quiet signal"): the one wake set is told when the front ends a turn
/// idle, here the turn that takes a request; a second wake is refused, with its exact words,
/// and never told. The order inside an idle turn (quiet first, no lock held) is pinned by
/// `the_quiet_wake_is_told_after_the_front_is_quiet_with_no_lock_held`.
#[test]
fn the_production_front_thread_tells_its_one_quiet_wake() {
    let r = rig();
    let first = Arc::new(QuietCount::default());
    let second = Arc::new(QuietCount::default());
    assert_eq!(
        r.gpu.set_quiet_wake(Arc::<QuietCount>::clone(&first)),
        Ok(())
    );
    assert_eq!(
        r.gpu.set_quiet_wake(Arc::<QuietCount>::clone(&second)),
        Err(QuietWakeTaken)
    );
    assert_eq!(
        QuietWakeTaken.to_string(),
        "the GPU front already tells another listener when it ends a turn idle"
    );
    let before = first.count();
    // A request the front takes in a turn of its own, which ends idle.
    r.space();
    wait_until("the front reports quiet after the request's turn", || {
        first.count() > before
    });
    assert_eq!(second.count(), 0);
    r.gpu.stop().unwrap();
}
