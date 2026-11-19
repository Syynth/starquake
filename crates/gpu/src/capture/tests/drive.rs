//! The collector that empties the ring, the queue of channels the back forgets, and the hash of
//! a ring entry.

use std::time::Duration;

use crate::capture::ContentHash;
use crate::capture::drive::{CollectError, Collector, Retired, entry_hash};
use crate::front::ChannelRetired;
use crate::interface::ChannelId;
use crate::ring::{ClassId, RingEntry, encode, encoded_words, ring};

/// A `Write` entry of `value`.
fn write(value: u32) -> RingEntry<'static> {
    RingEntry::Write {
        channel: ChannelId(3),
        class: ClassId(0xc797),
        method: 0x40,
        value,
    }
}

/// Bounds every wait of these tests.
const PATIENCE: Duration = Duration::from_secs(20);

#[test]
fn the_collector_keeps_entries_up_to_its_limit_in_words() {
    let words = encoded_words(&write(0));
    // Exactly three entries fit: all are kept.
    let (mut producer, consumer) = ring(1 << 10).unwrap();
    let collector = Collector::start(consumer, 3 * words).unwrap();
    for value in 1..=3 {
        producer.push(&write(value)).unwrap();
    }
    producer.publish();
    assert!(collector.wait_taken(3, PATIENCE));
    assert!(!collector.overflowed());
    assert_eq!(collector.drain(), [write(1), write(2), write(3)]);
    // Draining frees the room again.
    producer.push(&write(4)).unwrap();
    drop(producer);
    let (rest, ended) = collector.finish();
    assert_eq!(rest, [write(4)]);
    assert_eq!(ended, Ok(()));

    // One word short of three entries: the third is dropped and the end is an overflow.
    let (mut producer, consumer) = ring(1 << 10).unwrap();
    let collector = Collector::start(consumer, 3 * words - 1).unwrap();
    for value in 1..=3 {
        producer.push(&write(value)).unwrap();
    }
    drop(producer);
    let (kept, ended) = collector.finish();
    assert_eq!(kept, [write(1), write(2)]);
    assert_eq!(ended, Err(CollectError::Overflow));
    assert_eq!(
        CollectError::Overflow.to_string(),
        "the front pushed more than 16777216 ring words the back had not taken"
    );
}

#[test]
fn a_wait_for_more_than_the_ring_holds_ends_when_the_ring_does() {
    let (mut producer, consumer) = ring(1 << 10).unwrap();
    let collector = Collector::start(consumer, 1 << 10).unwrap();
    producer.push(&write(1)).unwrap();
    drop(producer);
    assert!(!collector.wait_taken(2, PATIENCE));
    assert!(collector.wait_taken(1, Duration::ZERO));
}

#[test]
fn retired_channels_are_due_once_the_back_has_taken_their_entries_in_order() {
    let retired = Retired::default();
    retired.retired(ChannelId(4), 2);
    retired.retired(ChannelId(9), 2);
    retired.retired(ChannelId(1), 5);
    assert_eq!(retired.due(1), []);
    assert_eq!(retired.due(2), [ChannelId(4), ChannelId(9)]);
    assert_eq!(retired.due(4), []);
    assert_eq!(retired.due(5), [ChannelId(1)]);
    assert_eq!(retired.due(6), []);
}

#[test]
fn an_entrys_hash_is_of_its_encoding_in_little_endian_words() {
    let entry = write(0x1234);
    let mut words = Vec::new();
    encode(&entry, &mut words).unwrap();
    let bytes: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
    assert_eq!(entry_hash(&entry), Some(ContentHash::of(&bytes)));
    assert_ne!(entry_hash(&entry), entry_hash(&write(0x1235)));
}
