//! Tests for the ring buffer: capacity, back-pressure, publishing, wrap-around, disconnection,
//! corruption, and ordering between two threads. Every wait is bounded, so a broken ring fails
//! these tests instead of hanging them.

use std::borrow::Cow;
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use super::*;
use crate::ring::{
    ChannelId, ClassId, DecodeError, Kind, MarkerToken, PresentRequest, Release, ReleaseOp,
    RingEntry, SyncpointId, encoded_words,
};
use crate::sema::{PayloadSize, ReleaseForm};

/// Long enough for any of these tests' threads to make progress; only reached when broken.
const PATIENCE: Duration = Duration::from_secs(10);

/// Runs `body` on its own thread and returns its result, failing the test after [`PATIENCE`].
fn within<T: Send + 'static>(body: impl FnOnce() -> T + Send + 'static) -> T {
    let (done, result) = mpsc::channel();
    thread::spawn(move || {
        // The receiver is gone only once the test has already failed.
        let _ = done.send(body());
    });
    result
        .recv_timeout(PATIENCE)
        .expect("the ring test did not finish in time")
}

fn write(channel: u32, value: u32) -> RingEntry<'static> {
    RingEntry::Write {
        channel: ChannelId(channel),
        class: ClassId(0xc797),
        method: 0x45,
        value,
    }
}

fn release(payload: u64) -> RingEntry<'static> {
    RingEntry::Release(Release {
        address: 0x1000,
        payload,
        form: ReleaseForm {
            size: PayloadSize::Four,
            timestamp: false,
        },
        op: ReleaseOp::Write,
    })
}

fn run(channel: u32, words: usize) -> RingEntry<'static> {
    RingEntry::Run {
        channel: ChannelId(channel),
        class: ClassId(0xc797),
        method: 0x8e3,
        words: Cow::Owned((0..).take(words).collect()),
    }
}

fn present() -> RingEntry<'static> {
    RingEntry::Present(PresentRequest {
        image: crate::ring::ImageDesc {
            address: 0x4000,
            width: 1,
            height: 2,
            pitch: 3,
            block_height_log2: 4,
            format: 5,
        },
        shown: starquake_render::command::Rect {
            x: 6,
            y: 7,
            width: 8,
            height: 9,
        },
        layer: crate::ring::LayerId(10),
        display: crate::ring::DisplayId(11),
    })
}

/// Words of one [`write`].
fn write_words() -> usize {
    encoded_words(&write(0, 0))
}


#[test]
fn a_capacity_must_be_a_power_of_two() {
    for words in [0, 3, 6, 1000] {
        assert_eq!(ring(words).err(), Some(CapacityError { words }), "{words}");
    }
    assert_eq!(
        ring(6).err().unwrap().to_string(),
        "a ring of 6 words is not a power of two"
    );
    for words in [1, 2, 8, 1024] {
        let (producer, _) = ring(words).unwrap();
        assert_eq!(producer.capacity(), words);
    }
}

#[test]
fn entries_are_invisible_until_published() {
    let (mut producer, mut consumer) = ring(64).unwrap();
    producer.try_push(&write(1, 2)).unwrap();
    assert_eq!(producer.unpublished(), 2);
    assert_eq!(consumer.try_pop(), Ok(None));
    producer.publish();
    assert_eq!(producer.unpublished(), 0);
    assert_eq!(consumer.try_pop(), Ok(Some(write(1, 2))));
    assert_eq!(consumer.try_pop(), Ok(None));
}

#[test]
fn the_producer_publishes_on_its_own_at_the_batch_size() {
    let per_entry = u64::try_from(write_words()).unwrap();
    let below = PUBLISH_EVERY / per_entry - 1;
    let (mut producer, mut consumer) = ring(1024).unwrap();
    for value in 0..below {
        producer.push(&write(1, u32::try_from(value).unwrap())).unwrap();
    }
    assert_eq!(producer.unpublished(), PUBLISH_EVERY - per_entry);
    assert_eq!(consumer.try_pop(), Ok(None));
    // The entry that brings the unpublished words to exactly PUBLISH_EVERY publishes them all.
    producer.push(&write(1, 0xff)).unwrap();
    assert_eq!(producer.unpublished(), 0);
    for value in 0..below {
        assert_eq!(
            consumer.try_pop(),
            Ok(Some(write(1, u32::try_from(value).unwrap())))
        );
    }
    assert_eq!(consumer.try_pop(), Ok(Some(write(1, 0xff))));
}

#[test]
fn a_full_ring_refuses_an_entry_until_one_is_popped() {
    let (mut producer, mut consumer) = ring(8).unwrap();
    for value in 0..4 {
        producer.try_push(&write(1, value)).unwrap();
    }
    assert_eq!(
        producer.try_push(&write(1, 4)),
        Err(PushError::Full { free: 0, needed: 2 })
    );
    let one_word = RingEntry::WaitForIdle {
        channel: ChannelId(1),
    };
    assert_eq!(
        producer.try_push(&one_word),
        Err(PushError::Full { free: 0, needed: 1 })
    );
    producer.publish();
    assert_eq!(consumer.try_pop(), Ok(Some(write(1, 0))));
    producer.try_push(&write(1, 4)).unwrap();
    assert_eq!(
        producer.try_push(&one_word),
        Err(PushError::Full { free: 0, needed: 1 })
    );
    producer.publish();
    for value in 1..5 {
        assert_eq!(consumer.try_pop(), Ok(Some(write(1, value))));
    }
}

#[test]
fn an_entry_of_exactly_the_capacity_fits_only_an_empty_ring() {
    let (mut producer, mut consumer) = ring(8).unwrap();
    assert_eq!(encoded_words(&present()), 8);
    producer.try_push(&present()).unwrap();
    producer.publish();
    assert_eq!(consumer.try_pop(), Ok(Some(present())));

    let one_word = RingEntry::Syncpoint {
        id: SyncpointId::new(3).unwrap(),
    };
    producer.try_push(&one_word).unwrap();
    assert_eq!(
        producer.try_push(&present()),
        Err(PushError::Full { free: 7, needed: 8 })
    );
}

#[test]
fn an_entry_longer_than_the_ring_is_too_large() {
    let (mut producer, _consumer) = ring(8).unwrap();
    // 2 + 12 / 2 = 8 words fits; 2 + 13.div_ceil(2) = 9 does not.
    assert_eq!(encoded_words(&run(1, 12)), 8);
    producer.try_push(&run(1, 12)).unwrap();
    for push in [Producer::try_push, Producer::push] {
        assert_eq!(
            push(&mut producer, &run(1, 13)),
            Err(PushError::TooLarge {
                needed: 9,
                capacity: 8
            })
        );
    }
    assert_eq!(
        producer.try_push(&run(1, 13)).unwrap_err().to_string(),
        "the entry takes 9 words and the ring holds 8"
    );
}

#[test]
fn an_entry_without_an_encoding_is_refused() {
    let (mut producer, _consumer) = ring(1 << 14).unwrap();
    assert_eq!(
        producer.try_push(&run(1, crate::ring::MAX_RUN_WORDS + 1)),
        Err(PushError::Encode(crate::ring::EncodeError::RunTooLong {
            words: crate::ring::MAX_RUN_WORDS + 1
        }))
    );
    assert_eq!(producer.unpublished(), 0);
}

#[test]
fn entries_wrap_around_the_end_of_the_ring() {
    let (mut producer, mut consumer) = ring(8).unwrap();
    for value in 0..3 {
        producer.try_push(&write(1, value)).unwrap();
    }
    producer.publish();
    for value in 0..3 {
        assert_eq!(consumer.try_pop(), Ok(Some(write(1, value))));
    }
    // Written at 6, 7 and 0, then 8 words from 1 around to 0 again.
    producer.try_push(&release(0x55)).unwrap();
    producer.publish();
    assert_eq!(consumer.try_pop(), Ok(Some(release(0x55))));
    producer.try_push(&present()).unwrap();
    producer.publish();
    assert_eq!(consumer.try_pop(), Ok(Some(present())));
    assert_eq!(consumer.try_pop(), Ok(None));
}

#[test]
fn a_dropped_consumer_disconnects_the_producer() {
    let (mut producer, consumer) = ring(8).unwrap();
    producer.try_push(&write(1, 0)).unwrap();
    drop(consumer);
    assert_eq!(producer.try_push(&write(1, 1)), Err(PushError::Disconnected));
    assert_eq!(producer.push(&write(1, 1)), Err(PushError::Disconnected));
    assert_eq!(
        PushError::Disconnected.to_string(),
        "the ring's consumer is gone"
    );
}

#[test]
fn a_dropped_producer_publishes_and_then_disconnects() {
    let (mut producer, mut consumer) = ring(64).unwrap();
    producer.try_push(&write(1, 0)).unwrap();
    producer.try_push(&write(1, 1)).unwrap();
    drop(producer);
    assert_eq!(consumer.try_pop(), Ok(Some(write(1, 0))));
    assert_eq!(consumer.pop(), Ok(write(1, 1)));
    for _ in 0..2 {
        assert_eq!(consumer.try_pop(), Err(PopError::Disconnected));
        assert_eq!(consumer.pop(), Err(PopError::Disconnected));
        assert_eq!(
            consumer.pop_timeout(Duration::ZERO),
            Err(PopError::Disconnected)
        );
    }
    assert_eq!(
        PopError::Disconnected.to_string(),
        "the ring is empty and its producer is gone"
    );
}

#[test]
fn a_corrupt_entry_stops_the_ring_for_good() {
    let (mut producer, mut consumer) = ring(64).unwrap();
    producer.push_raw(&[0]);
    producer.publish();
    let corrupt = PopError::Corrupt(DecodeError::UnknownKind { kind: 0 });
    assert_eq!(consumer.try_pop(), Err(corrupt));
    // A valid entry after it is never reached, and the error does not turn into an end.
    producer.try_push(&write(1, 0)).unwrap();
    producer.publish();
    assert_eq!(consumer.try_pop(), Err(corrupt));
    assert_eq!(consumer.pop_timeout(Duration::ZERO), Err(corrupt));
    drop(producer);
    assert_eq!(consumer.pop(), Err(corrupt));
    assert_eq!(
        corrupt.to_string(),
        "the ring is corrupt: ring entry kind 0 is not defined"
    );
}

#[test]
fn an_entry_longer_than_what_was_published_is_corrupt_for_good() {
    let (mut producer, mut consumer) = ring(64).unwrap();
    // A run header promising two data words (three ring words), published alone.
    producer.push_raw(&[(2 << 32) | 0x45_03]);
    producer.publish();
    let corrupt = PopError::Corrupt(DecodeError::Truncated {
        kind: Kind::Run,
        needed: 3,
        available: 1,
    });
    assert_eq!(consumer.try_pop(), Err(corrupt));
    // With more words published the header would now decode, as a run of whatever follows;
    // the error stays.
    producer.try_push(&write(1, 0)).unwrap();
    producer.try_push(&write(1, 1)).unwrap();
    producer.publish();
    assert_eq!(consumer.try_pop(), Err(corrupt));
}

#[test]
fn pop_timeout_gives_up_on_an_empty_ring() {
    let (_producer, mut consumer) = ring(8).unwrap();
    assert_eq!(consumer.pop_timeout(Duration::ZERO), Ok(None));
    assert_eq!(consumer.pop_timeout(Duration::from_millis(5)), Ok(None));
}

#[test]
fn a_waiting_consumer_wakes_on_a_publish_and_on_a_drop() {
    let (mut producer, mut consumer) = ring(8).unwrap();
    let (popped, results) = mpsc::channel();
    let reader = thread::spawn(move || {
        let _ = popped.send(consumer.pop());
        let _ = popped.send(consumer.pop());
    });
    producer.try_push(&write(1, 7)).unwrap();
    producer.publish();
    assert_eq!(results.recv_timeout(PATIENCE), Ok(Ok(write(1, 7))));
    drop(producer);
    assert_eq!(results.recv_timeout(PATIENCE), Ok(Err(PopError::Disconnected)));
    within(move || reader.join().is_ok());
}

#[test]
fn a_full_producer_publishes_before_it_waits() {
    // Without that publish the producer would wait for space while the consumer waits for
    // entries that were written but never published.
    let (mut producer, mut consumer) = ring(8).unwrap();
    let count = 5_u32;
    let writer = thread::spawn(move || {
        for value in 0..count {
            producer.push(&write(1, value)).unwrap();
        }
        producer.publish();
        // Keep the producer alive until every entry is read, so its drop cannot publish them.
        producer
    });
    for value in 0..count {
        assert_eq!(consumer.pop_timeout(PATIENCE), Ok(Some(write(1, value))));
    }
    within(move || writer.join().is_ok());
}

#[test]
fn a_waiting_producer_wakes_when_the_consumer_goes() {
    let (mut producer, consumer) = ring(8).unwrap();
    for value in 0..4 {
        producer.try_push(&write(1, value)).unwrap();
    }
    let writer = thread::spawn(move || producer.push(&write(1, 4)));
    drop(consumer);
    assert_eq!(
        within(move || writer.join().ok()),
        Some(Err(PushError::Disconnected))
    );
}

/// A fixed-seed generator for the ordering test's schedule (a 64-bit LCG, Knuth's MMIX
/// constants), so every run pushes the same stream.
struct Schedule(u64);

impl Schedule {
    fn next(&mut self, below: u64) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 33) % below
    }
}

/// Entries from four channels interleaved by a fixed schedule, with releases, markers and runs
/// of several lengths among them.
fn interleaved(count: u32) -> Vec<RingEntry<'static>> {
    let mut schedule = Schedule(0x5eed);
    (0..count)
        .map(|serial| {
            let channel = u32::try_from(schedule.next(4)).unwrap();
            match schedule.next(5) {
                0 => release(u64::from(serial)),
                1 => RingEntry::Marker {
                    token: MarkerToken(u64::from(serial)),
                },
                2 => run(channel, usize::try_from(schedule.next(30)).unwrap()),
                _ => write(channel, serial),
            }
        })
        .collect()
}

#[test]
fn two_threads_see_every_entry_in_push_order_through_a_small_ring() {
    let entries = interleaved(20_000);
    let pushed = entries.clone();
    // 32 words: the longest run here (29 words, 17 ring words) fits, and the ring is full often.
    let (mut producer, mut consumer) = ring(32).unwrap();
    let writer = thread::spawn(move || {
        for entry in &pushed {
            producer.push(entry)?;
        }
        Ok::<_, PushError>(())
    });
    let received = within(move || {
        let mut received = Vec::new();
        loop {
            match consumer.pop_timeout(PATIENCE) {
                Ok(Some(entry)) => received.push(entry),
                Ok(None) => return Err("no entry in time".to_owned()),
                Err(PopError::Disconnected) => return Ok(received),
                Err(error) => return Err(error.to_string()),
            }
        }
    })
    .unwrap();
    assert_eq!(within(move || writer.join().ok()), Some(Ok(())));
    assert_eq!(received, entries);
    // Within each channel, the order is the order that channel's entries were pushed.
    for channel in 0..4 {
        let of = |list: &[RingEntry<'static>]| -> Vec<RingEntry<'static>> {
            list.iter()
                .filter(|entry| entry.channel() == Some(ChannelId(channel)))
                .cloned()
                .collect()
        };
        assert_eq!(of(&received), of(&entries));
    }
}
