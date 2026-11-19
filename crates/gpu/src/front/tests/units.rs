//! The pure pieces: the poll's schedule, the queue bound, the ring sink's view of the back, and
//! the shared state's refusals and messages.

use super::super::ring_sink::RingSink;
use super::*;
use crate::interface::CHANNEL_QUEUE_ENTRIES;
use crate::ring::{MarkerToken, PushError};

// ---------------------------------------------------------------------------------------------
// The poll's schedule
// ---------------------------------------------------------------------------------------------

#[test]
fn the_polls_constants_are_decision_3s_and_the_guesses() {
    // Decision 3: "sleep in growing steps up to about 100 µs"; the rest are guesses.
    assert_eq!(POLL_LONGEST_SLEEP, Duration::from_micros(100));
    assert_eq!(POLL_SPINS, 64);
    assert_eq!(POLL_FIRST_SLEEP, Duration::from_micros(1));
    assert_eq!(IDLE_PARK, Duration::from_millis(100));
    assert_eq!(QUANTUM, 4096);
}

#[test]
fn backoff_steps_at_their_edges() {
    let mut backoff = Backoff::new();
    // Exactly `POLL_SPINS` polls with no sleep, then the first sleep.
    for spin in 0..POLL_SPINS {
        assert_eq!(backoff.next_sleep(), Duration::ZERO, "spin {spin}");
    }
    assert_eq!(backoff.next_sleep(), POLL_FIRST_SLEEP);
    // Doubling from the first sleep while below the ceiling; the step that would pass it is
    // the ceiling itself, and the schedule stays there.
    let mut expected = POLL_FIRST_SLEEP * 2;
    while expected < POLL_LONGEST_SLEEP {
        assert_eq!(backoff.next_sleep(), expected);
        expected *= 2;
    }
    assert_eq!(backoff.next_sleep(), POLL_LONGEST_SLEEP);
    assert_eq!(backoff.next_sleep(), POLL_LONGEST_SLEEP);
    // The written-out sequence for the shipping constants: 64 spins, then 1, 2, 4, 8, 16, 32,
    // 64 µs, then 100 µs.
    let mut fresh = Backoff::default();
    let steps: Vec<u64> = (0..POLL_SPINS + 9)
        .map(|_| u64::try_from(fresh.next_sleep().as_micros()).unwrap())
        .skip(usize::try_from(POLL_SPINS).unwrap())
        .collect();
    assert_eq!(steps, [1, 2, 4, 8, 16, 32, 64, 100, 100]);
    // A reset starts the spins again.
    backoff.reset();
    assert_eq!(backoff, Backoff::new());
    assert_eq!(backoff.next_sleep(), Duration::ZERO);
}

#[test]
fn the_host_clock_moves_forward_from_zero() {
    let clock = HostClock::default();
    let first = clock.now();
    let second = clock.now();
    assert!(second >= first);
    assert!(first < Duration::from_secs(60));
}

// ---------------------------------------------------------------------------------------------
// The queue bound
// ---------------------------------------------------------------------------------------------

#[test]
fn the_queue_bound_is_decision_12s_and_fits_compares_at_it() {
    assert_eq!(CHANNEL_QUEUE_ENTRIES, 65_536);
    // Exactly full fits; one more does not, from either side.
    assert!(fits(CHANNEL_QUEUE_ENTRIES - 1, 1));
    assert!(fits(CHANNEL_QUEUE_ENTRIES, 0));
    assert!(fits(0, CHANNEL_QUEUE_ENTRIES));
    assert!(!fits(CHANNEL_QUEUE_ENTRIES, 1));
    assert!(!fits(0, CHANNEL_QUEUE_ENTRIES + 1));
    assert!(!fits(1, CHANNEL_QUEUE_ENTRIES));
    assert!(fits(0, 0));
    // A sum past `usize` does not fit.
    assert!(!fits(usize::MAX, 1));
}

// ---------------------------------------------------------------------------------------------
// The ring sink
// ---------------------------------------------------------------------------------------------

fn wfi(channel: u32) -> RingEntry<'static> {
    RingEntry::WaitForIdle {
        channel: ChannelId(channel),
    }
}

#[test]
fn the_back_has_unperformed_work_until_a_marker_after_it_is_reported() {
    let (producer, mut consumer) = ring(64).unwrap();
    let mut sink = RingSink::new(producer);
    // Nothing pushed: nothing unperformed, and a marker alone is no work.
    assert!(!sink.has_unperformed());
    let empty = sink.marker();
    assert!(!sink.has_unperformed());
    sink.send(wfi(1));
    assert!(sink.has_unperformed());
    let first = sink.marker();
    sink.send(wfi(2));
    let second = sink.marker();
    assert!(first > empty && second > first);
    // The first marker reported: the entry after it is still unperformed.
    sink.reported(first);
    assert!(sink.has_unperformed());
    assert!(sink.is_reported(empty));
    assert!(sink.is_reported(first));
    assert!(!sink.is_reported(second));
    // The second: everything is performed.
    sink.reported(second);
    assert!(!sink.has_unperformed());
    assert!(sink.is_reported(second));
    // A token reported again, or never pushed, changes nothing.
    sink.send(wfi(3));
    sink.reported(second);
    sink.reported(MarkerToken(0x1000));
    assert!(sink.has_unperformed());
    assert!(!sink.is_reported(MarkerToken(0x1000)));
    // The entries reached the ring in push order.
    sink.publish();
    let mut popped = Vec::new();
    while let Some(entry) = consumer.try_pop().unwrap() {
        popped.push(entry);
    }
    assert_eq!(
        popped,
        [
            RingEntry::Marker { token: empty },
            wfi(1),
            RingEntry::Marker { token: first },
            wfi(2),
            RingEntry::Marker { token: second },
            wfi(3),
        ]
    );
}

#[test]
fn reporting_a_later_marker_reports_the_earlier_ones() {
    let (producer, _consumer) = ring(64).unwrap();
    let mut sink = RingSink::new(producer);
    sink.send(wfi(1));
    let first = sink.marker();
    sink.send(wfi(2));
    let second = sink.marker();
    sink.reported(second);
    assert!(sink.is_reported(first));
    assert!(!sink.has_unperformed());
    // The earlier token arriving late changes nothing.
    sink.reported(first);
    assert!(!sink.has_unperformed());
}

#[test]
fn a_reserved_token_is_reported_only_once_its_marker_is_pushed() {
    let (producer, _consumer) = ring(64).unwrap();
    let mut sink = RingSink::new(producer);
    let reserved = sink.reserve_token();
    sink.send(wfi(1));
    // Not pushed yet: a report of it is not one the front knows.
    sink.reported(reserved);
    assert!(!sink.is_reported(reserved));
    sink.push_marker(reserved);
    sink.reported(reserved);
    assert!(sink.is_reported(reserved));
    assert!(!sink.has_unperformed());
}

#[test]
fn a_failed_push_is_sticky_and_keeps_the_first_failure() {
    // A ring of 4 words: a run of 8 words takes 2 + 4 and is refused as too large; then the
    // consumer goes, and a later push would be refused as disconnected, but the first failure
    // is the one kept, and nothing more is pushed.
    let (producer, consumer) = ring(4).unwrap();
    let mut sink = RingSink::new(producer);
    sink.send(wfi(1));
    let run = RingEntry::Run {
        channel: ChannelId(1),
        class: class_3d(),
        method: 0x100,
        words: std::borrow::Cow::Owned(vec![1; 8]),
    };
    sink.send(run);
    let too_large = PushError::TooLarge {
        needed: 6,
        capacity: 4,
    };
    assert_eq!(sink.error(), Some(too_large));
    drop(consumer);
    sink.send(wfi(2));
    assert_eq!(sink.error(), Some(too_large));
    // The refused entries count as no work.
    assert!(sink.has_unperformed());
    let token = sink.marker();
    assert!(!sink.is_reported(token));
}

#[test]
fn a_failed_push_stops_the_front_for_good() {
    // A 3D constant-buffer run of 12 words is one `Run` entry of 2 + 6 words, more than a ring
    // of 4 holds: the turn reports the failure, and so does every later turn, even after the
    // consumer has gone (a second, different failure).
    let mut r = rig_with(4, |mem| mem);
    let channel = r.channel(5);
    let mut pb = PushBuffer::new();
    bind_3d(&mut pb);
    let load = m(clc797::LOAD_CONSTANT_BUFFER.method_at(&[0]).unwrap());
    pb.methods(MethodMode::NonIncrementing, 0, load, &[7; 12])
        .unwrap();
    r.submit(channel, vec![r.entry(CMD_A, pb.words())]);
    let too_large = FrontError::Ring(PushError::TooLarge {
        needed: 8,
        capacity: 4,
    });
    let mut first = r.front.turn();
    for _ in 0..4 {
        if first.is_err() {
            break;
        }
        r.back.drain();
        first = r.front.turn();
    }
    assert_eq!(first, Err(too_large));
    drop(r.back);
    assert_eq!(r.front.turn(), Err(too_large));
    assert_eq!(
        too_large.to_string(),
        "the GPU front stopped: the ring refused an entry: the entry takes 8 words and the ring holds 4"
    );
}

// ---------------------------------------------------------------------------------------------
// The shared state
// ---------------------------------------------------------------------------------------------

#[test]
fn a_channel_needs_a_known_address_space_and_a_free_id() {
    let r = rig();
    let config = ChannelConfig {
        space: OTHER_SPACE,
        syncpoint: syncpoint(1),
    };
    assert_eq!(
        r.shared.create_channel(config),
        Err(CreateError::UnknownSpace(OTHER_SPACE))
    );
    // Ids at their bound: the last id is handed out, the one after it is refused.
    let config = ChannelConfig {
        space: SPACE,
        ..config
    };
    r.shared.set_next_channel(u64::from(u32::MAX));
    assert_eq!(r.shared.create_channel(config), Ok(ChannelId(u32::MAX)));
    assert_eq!(r.shared.create_channel(config), Err(CreateError::OutOfIds));
    assert_eq!(
        CreateError::UnknownSpace(OTHER_SPACE).to_string(),
        "address space 34 was never made"
    );
    assert_eq!(
        CreateError::OutOfIds.to_string(),
        "every GPU channel id has been used"
    );
    assert_eq!(
        FrontError::Panicked.to_string(),
        "the GPU front thread panicked"
    );
}

#[test]
fn channels_take_ids_in_turn_from_the_first() {
    let r = rig();
    assert_eq!(r.channel(1), ChannelId(0x40));
    assert_eq!(r.channel(2), ChannelId(0x41));
}

#[test]
fn a_submission_for_an_unknown_channel_is_dropped() {
    let mut r = rig();
    // Never made.
    assert_eq!(
        r.submit(ChannelId(0x99), vec![r.entry(CMD_A, &write_scratch(0, 1))]),
        Submitted::Queued
    );
    assert_eq!(r.shared.dropped_submissions(), 1);
    // Destroyed.
    let channel = r.channel(5);
    r.shared.destroy_channel(channel);
    assert_eq!(
        r.submit(channel, vec![r.entry(CMD_A, &write_scratch(0, 1))]),
        Submitted::Queued
    );
    assert_eq!(r.shared.dropped_submissions(), 2);
    r.settle();
    assert_eq!(r.take(), Vec::<String>::new());
}

#[test]
fn making_an_address_space_twice_keeps_the_first() {
    // The second address space's memory has nothing mapped: had it replaced the first, the
    // channel's entry would fault.
    let mut r = rig();
    let empty = Arc::new(TestGpuMemory::new(FlatMemory::new(CPU_BASE, 0x1000)));
    r.shared
        .create_address_space(SPACE, Arc::new(GpuVm::new()), empty);
    let channel = r.channel(5);
    r.submit(channel, vec![r.entry(CMD_A, &write_scratch(0, 0x11))]);
    r.settle();
    assert_eq!(r.shared.channel_error(channel), None);
    assert!(has(&r.take(), &scratch_text(channel, 0, 0x11)));
}

#[test]
fn a_stop_ends_the_turns() {
    let mut r = rig();
    r.shared.stop();
    assert_eq!(r.front.turn(), Ok(Turn::Stopped));
}

#[test]
fn an_idle_front_reads_no_clock_and_parks_for_the_safety_net() {
    let mut r = rig();
    let channel = r.channel(5);
    r.submit(channel, vec![r.entry(CMD_A, &write_scratch(0, 1))]);
    assert_eq!(r.settle(), Turn::Idle { park: IDLE_PARK });
    assert_eq!(r.clock.reads(), 0);
    assert_eq!(r.front.turn(), Ok(Turn::Idle { park: IDLE_PARK }));
    assert_eq!(r.clock.reads(), 0);
}
