//! The flight recorder: what it keeps, and each event's words.

use crate::back::fence::ReleaseFailure;
use crate::interface::{Fence, RegionId};
use crate::memory::GpuFault;

use super::*;

fn barrier(ticket: u64) -> GpuEvent {
    GpuEvent::Barrier {
        ticket: Ticket(ticket),
    }
}

#[test]
fn it_keeps_the_last_events_up_to_its_capacity() {
    let recorder = FlightRecorder::new(2);
    recorder.record(barrier(1));
    recorder.record(barrier(2));
    assert_eq!(recorder.events(), [barrier(1), barrier(2)]);
    assert_eq!(recorder.text(&[]), "0 barrier, ticket 1\n1 barrier, ticket 2\n");
    recorder.record(barrier(3));
    assert_eq!(recorder.events(), [barrier(2), barrier(3)]);
    assert_eq!(recorder.recorded(), 3);
    assert_eq!(
        recorder.text(&[]),
        "1 earlier events were dropped\n1 barrier, ticket 2\n2 barrier, ticket 3\n"
    );
}

#[test]
fn a_recorder_of_no_events_keeps_none_and_counts_them() {
    let recorder = FlightRecorder::new(0);
    recorder.record(barrier(1));
    assert_eq!(recorder.events(), []);
    assert_eq!(recorder.recorded(), 1);
    assert_eq!(recorder.text(&[]), "1 earlier events were dropped\n");
}

#[test]
fn an_empty_recorder_writes_nothing_but_the_stopped_channels() {
    let recorder = FlightRecorder::new(RECORDER_EVENTS);
    assert_eq!(recorder.text(&[]), "");
    let error = ChannelError::UnalignedEntry { address: 0x1002 };
    assert_eq!(
        recorder.text(&[(ChannelId(3), error)]),
        "channel 3 stopped: the GP entry at GPU address 0x1002 is not 4-byte aligned\n"
    );
}

/// The events about spaces, channels and submissions, each with its text.
fn call_cases() -> Vec<(GpuEvent, &'static str)> {
    vec![
        (
            GpuEvent::SpaceMade { space: AsId(4) },
            "address space 4 made",
        ),
        (
            GpuEvent::ChannelMade {
                channel: ChannelId(2),
                space: AsId(4),
                syncpoint: syncpoint(),
            },
            "channel 2 made in address space 4, incrementing syncpoint 5",
        ),
        (
            GpuEvent::ChannelRefused { space: AsId(4) },
            "a channel in address space 4 was refused",
        ),
        (
            GpuEvent::Submitted {
                channel: ChannelId(2),
                entries: 3,
                wait: Some(Fence {
                    id: syncpoint(),
                    value: 7,
                }),
                increments: 1,
                wfi: true,
                outcome: Submitted::Full,
            },
            "submission on channel 2: 3 entries, 1 increments, waits for syncpoint 5 to reach 7, \
             wait for idle: refused, the queue is full",
        ),
        (
            GpuEvent::Submitted {
                channel: ChannelId(2),
                entries: 0,
                wait: None,
                increments: 0,
                wfi: false,
                outcome: Submitted::Queued,
            },
            "submission on channel 2: 0 entries, 0 increments: queued",
        ),
        (
            GpuEvent::ChannelDestroyed {
                channel: ChannelId(2),
            },
            "channel 2 destroyed",
        ),
    ]
}

/// The events about updates, the back's work and completions, each with its text.
fn work_cases() -> Vec<(GpuEvent, &'static str)> {
    let range = PageRange::new(0x10_0000, 0x2000).unwrap();
    vec![
        (
            GpuEvent::Update {
                space: AsId(4),
                update: VmUpdate::Unmap { range },
                ticket: Some(Ticket(8)),
            },
            "ordered update of address space 4: unmap of 0x2000 bytes at 0x100000, ticket 8",
        ),
        (
            GpuEvent::Update {
                space: AsId(4),
                update: VmUpdate::ModifyKind {
                    range,
                    kind: Kind(0xfe),
                },
                ticket: None,
            },
            "ordered update of address space 4: kind change of 0x2000 bytes at 0x100000 to kind \
             0xfe, no ticket",
        ),
        (
            GpuEvent::Update {
                space: AsId(4),
                update: VmUpdate::DestroyMemory { range },
                ticket: None,
            },
            "ordered update of address space 4: memory destroy of 0x2000 bytes at 0x100000, no \
             ticket",
        ),
        (
            GpuEvent::Update {
                space: AsId(4),
                update: VmUpdate::DestroySparseRegion {
                    range,
                    region: RegionId(6),
                },
                ticket: Some(Ticket(9)),
            },
            "ordered update of address space 4: sparse region destroy of 0x2000 bytes at \
             0x100000 (region 6), ticket 9",
        ),
        (barrier(11), "barrier, ticket 11"),
        (
            GpuEvent::Performed {
                released: 2,
                incremented: 3,
                latched: true,
                markers: 1,
                failed: vec![ReleaseFailure {
                    address: 0x40,
                    fault: GpuFault::Unmapped { va: 0x40 },
                }],
            },
            "performed 2 releases, 3 increments, 1 markers, a waiter latched; the release at \
             GPU address 0x40 was not performed: GPU address 0x40 is not mapped",
        ),
        (
            GpuEvent::Completion(GpuCompletion::SyncpointChanged {
                id: syncpoint(),
                value: 12,
            }),
            "pushed: syncpoint 5 changed to 12",
        ),
        (
            GpuEvent::Completion(GpuCompletion::GpuOrdered { ticket: Ticket(13) }),
            "pushed: ticket 13 done",
        ),
        (
            GpuEvent::Completion(GpuCompletion::GpuQueueRoom {
                channel: ChannelId(2),
            }),
            "pushed: channel 2 has queue room",
        ),
    ]
}

#[test]
fn each_event_is_written_in_plain_words() {
    for (event, text) in call_cases().into_iter().chain(work_cases()) {
        assert_eq!(event.to_string(), text);
    }
    // A present is written as its ring entry's text.
    let present = GpuEvent::Present(present_request());
    assert_eq!(
        present.to_string(),
        RingEntry::Present(present_request()).to_string()
    );
}
