//! The semaphore mirror page through the threads (tentative, decision log 2026-11-20): gsv
//! lists a two-page registration in the table it gives the GPU, and a host `SEMAPHORED` release
//! into its second page reaches its first page, where a GPU acquire on another channel sees it.

use super::*;
use crate::vm::{SEMAPHORE_MIRROR_OFFSET, SEMAPHORE_REGISTRATION_BYTES, SemaphoreRegistration};

/// The two-page registration: inside the rig's mapping, clear of its command buffers, frame and
/// semaphores.
const REGISTRATION: u64 = GPU_BASE + 40 * PAGE_SIZE;

/// A front clock that never moves, so a parked acquire is not polled again after its spins: only
/// a release that latches it resumes the channel.
struct Frozen;

impl FrontClock for Frozen {
    fn now(&self) -> Duration {
        Duration::ZERO
    }
}

fn list(r: &Rig) {
    let registration = SemaphoreRegistration::new(
        PageRange::new(REGISTRATION, SEMAPHORE_REGISTRATION_BYTES).unwrap(),
    )
    .unwrap();
    r.vm.add_semaphore_registration(registration);
}

#[test]
fn a_release_into_the_second_page_reaches_the_first_through_the_threads() {
    let r = rig();
    list(&r);
    r.space();
    let channel = r.channel();
    let release = REGISTRATION + SEMAPHORE_MIRROR_OFFSET + 0x10;
    r.gpu
        .submit(channel, plain(vec![r.entry(CMD, &releases(release, 0x77))]));
    wait_until("the copy is written", || r.word(REGISTRATION + 0x10) == 0x77);
    assert_eq!(r.word(release), 0x77);
    r.gpu.stop().unwrap();
}

/// The bring-up's case: one channel waits on a word of the first page, another releases into
/// the second; the copy latches the waiter, and the first channel goes on to its own release.
#[test]
fn an_acquire_on_the_first_page_is_met_by_a_release_into_the_second() {
    let r = rig_with(false, Arc::new(Frozen));
    list(&r);
    r.space();
    let waiting = r.channel();
    let releasing = r.channel();
    r.gpu.submit(
        waiting,
        plain(vec![r.entry(
            CMD,
            &acquires_then_releases(REGISTRATION + 0x10, 0x31, SEM, 0x99),
        )]),
    );
    wait_until("the channel parks", || {
        r.gpu.status.published().counters.parks == 1
    });
    assert_eq!(r.word(SEM), 0);
    r.gpu.submit(
        releasing,
        plain(vec![r.entry(
            CMD2,
            &releases(REGISTRATION + SEMAPHORE_MIRROR_OFFSET + 0x10, 0x31),
        )]),
    );
    wait_until("the waiting channel releases", || r.word(SEM) == 0x99);
    r.gpu.stop().unwrap();
}
