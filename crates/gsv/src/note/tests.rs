//! Tests of the recorded notes: their codes and values, and reading them back.

use proptest::prelude::*;
use proptest::test_runner::{Config, RngAlgorithm, RngSeed};

use super::*;

fn id(value: u8) -> SyncpointId {
    SyncpointId::new(value).unwrap()
}

fn last_id() -> SyncpointId {
    id(u8::try_from(SYNCPOINT_COUNT - 1).unwrap())
}

#[test]
fn every_note_s_code_and_value_are_pinned() {
    // The recorded layout (module docs): each field a distinct non-zero value.
    let service = ServiceId(0x0657);
    let cases = [
        (
            Note::SyncpointObserved {
                id: id(0x15),
                value: 0x0102_0304,
            },
            1,
            0x15_0102_0304,
        ),
        (
            Note::SyncptPageAnswered {
                id: id(0x2A),
                value: 0x1112_1314,
            },
            2,
            0x2A_1112_1314,
        ),
        (
            Note::GpuOrderedDone {
                ticket: Ticket(0x0506_0708_090A_0B0C),
            },
            3,
            0x0506_0708_090A_0B0C,
        ),
        (
            Note::QueueRoom {
                channel: ChannelId(0x0D0E_0F10),
            },
            8,
            0x0D0E_0F10,
        ),
        (Note::SyncptPageAcknowledged { groups: 0b101 }, 9, 0b101),
    ];
    for (note, code, value) in cases {
        assert_eq!((note.code(), note.value()), (code, value), "{note:?}");
        assert_eq!(
            note.event(service),
            EventKind::ServiceNote {
                service,
                code,
                value
            }
        );
        assert_eq!(Note::decode(code, value), Ok(note));
    }
    assert_eq!(
        [
            SYNCPOINT_OBSERVED,
            SYNCPT_PAGE_ANSWERED,
            GPU_ORDERED_DONE,
            QUEUE_ROOM,
            SYNCPT_PAGE_ACKNOWLEDGED
        ],
        [1, 2, 3, 8, 9]
    );
}

/// A constructor of a note that packs a syncpoint id and a counter value.
type MakeNote = fn(SyncpointId, u32) -> Note;

/// The two notes that pack a syncpoint id and a counter value, each with its constructor.
fn syncpoint_notes() -> [(u32, MakeNote); 2] {
    [
        (SYNCPOINT_OBSERVED, |id, value| Note::SyncpointObserved {
            id,
            value,
        }),
        (SYNCPT_PAGE_ANSWERED, |id, value| Note::SyncptPageAnswered {
            id,
            value,
        }),
    ]
}

#[test]
fn a_syncpoint_note_reads_back_at_its_edges() {
    // Hand-built values: the lowest id with the lowest value, the last id with the highest.
    let top = (u64::from(last_id().get()) << SYNCPOINT_ID_SHIFT) | u64::from(u32::MAX);
    assert_eq!(top, 0xBF_FFFF_FFFF);
    for (code, make) in syncpoint_notes() {
        assert_eq!(Note::decode(code, 0), Ok(make(id(0), 0)));
        assert_eq!(Note::decode(code, top), Ok(make(last_id(), u32::MAX)));
    }
}

#[test]
fn a_syncpoint_note_naming_no_syncpoint_is_an_error() {
    // The first id past the count, and a bit above the id's byte.
    let past = u64::from(SYNCPOINT_COUNT) << SYNCPOINT_ID_SHIFT;
    let high = 1 << (SYNCPOINT_ID_SHIFT + u8::BITS);
    for (code, _) in syncpoint_notes() {
        for value in [past, high, high | 1, u64::MAX] {
            let error = Note::decode(code, value).unwrap_err();
            assert_eq!(error, NoteError::NoSuchSyncpoint { value });
            assert_eq!(
                error.to_string(),
                format!("syncpoint note value {value:#x} names no syncpoint (there are 192)")
            );
        }
    }
}

#[test]
fn a_queue_room_note_wider_than_a_channel_is_an_error() {
    let widest = u64::from(u32::MAX);
    assert_eq!(
        Note::decode(QUEUE_ROOM, widest),
        Ok(Note::QueueRoom {
            channel: ChannelId(u32::MAX)
        })
    );
    let error = Note::decode(QUEUE_ROOM, widest + 1).unwrap_err();
    assert_eq!(error, NoteError::NoSuchChannel { value: widest + 1 });
    assert_eq!(
        error.to_string(),
        "queue room note value 0x100000000 is not a channel number"
    );
}

#[test]
fn a_code_gsv_does_not_write_is_an_error() {
    // Zero, each reserved code, and one past the last.
    for code in [0, 4, 5, 6, 7, 10, u32::MAX] {
        let error = Note::decode(code, 1).unwrap_err();
        assert_eq!(error, NoteError::UnknownCode { code });
        assert_eq!(
            error.to_string(),
            format!("gsv writes no note with code {code}")
        );
    }
}

#[test]
fn a_page_acknowledge_note_names_a_non_empty_set_of_the_page_s_groups() {
    // Each group alone and all of them; the mask is exactly the page's groups.
    let all = (1 << GROUP_COUNT) - 1;
    assert_eq!(GROUP_MASK, all);
    for groups in (0..GROUP_COUNT).map(|group| 1 << group).chain([all]) {
        assert_eq!(
            Note::decode(SYNCPT_PAGE_ACKNOWLEDGED, groups),
            Ok(Note::SyncptPageAcknowledged { groups })
        );
    }
    // No group, the first bit past the last group (alone and with a real one), and every bit.
    let past = 1 << GROUP_COUNT;
    for value in [0, past, past | 1, u64::MAX] {
        let error = Note::decode(SYNCPT_PAGE_ACKNOWLEDGED, value).unwrap_err();
        assert_eq!(error, NoteError::NoSuchGroups { value });
        assert_eq!(
            error.to_string(),
            format!(
                "page acknowledge note value {value:#x} is not a non-empty set of the page's \
                 {GROUP_COUNT} groups"
            )
        );
    }
}

fn note() -> impl Strategy<Value = Note> {
    let ids = 0..u8::try_from(SYNCPOINT_COUNT).unwrap();
    prop_oneof![
        (ids.clone(), any::<u32>())
            .prop_map(|(sp, value)| Note::SyncpointObserved { id: id(sp), value }),
        (ids, any::<u32>()).prop_map(|(sp, value)| Note::SyncptPageAnswered { id: id(sp), value }),
        any::<u64>().prop_map(|ticket| Note::GpuOrderedDone {
            ticket: Ticket(ticket)
        }),
        any::<u32>().prop_map(|channel| Note::QueueRoom {
            channel: ChannelId(channel)
        }),
        (1..=GROUP_MASK).prop_map(|groups| Note::SyncptPageAcknowledged { groups }),
    ]
}

proptest! {
    #![proptest_config(Config {
        cases: 512,
        rng_algorithm: RngAlgorithm::ChaCha,
        rng_seed: RngSeed::Fixed(0x4E4F_5445_5345_4544),
        failure_persistence: None,
        ..Config::default()
    })]

    #[test]
    fn every_note_reads_back_as_written(note in note()) {
        prop_assert_eq!(Note::decode(note.code(), note.value()), Ok(note));
    }
}
