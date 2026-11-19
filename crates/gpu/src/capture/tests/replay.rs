//! Whole captures: built from hand-written streams, written, read back and replayed.

use super::*;
use crate::capture::{
    Access, BackRecord, Capture, Divergence, FrontRecord, Output, ReplayError, ReplayOptions,
    Thread, replay, writes_text,
};
use crate::ring::{RingEntry, write_text};

/// A channel that releases 1 at [`SEM`], then a present: built, written and read back.
fn release_and_present() -> Capture {
    let mut builder = builder();
    let channel = channel(&mut builder);
    let words = releases(SEM, 1);
    let submission = submission_at(&builder, CMD, &words);
    builder.submit(channel, submission).unwrap();
    builder.run().unwrap();
    builder.present(present_request()).unwrap();
    builder.run().unwrap();
    let capture = builder.finish().unwrap();
    Capture::parse(&capture.to_bytes().unwrap()).unwrap()
}

#[test]
fn a_built_capture_replays_to_its_expected_outputs() {
    let capture = release_and_present();
    let output = replay(&capture, ReplayOptions::default()).unwrap();
    assert_eq!(output.presents, 1);
    assert!(!output.stopped_early);
    insta::assert_snapshot!("release_and_present_ring", write_text(&output.ring));
    insta::assert_snapshot!("release_and_present_render", output.render);
    assert_eq!(
        writes_text(&output.writes),
        format!("back space {} wrote 4 bytes at {SEM:#x}: 01 00 00 00\n", SPACE.0)
    );
}

#[test]
fn a_replay_is_the_same_twice_and_the_builder_is_the_same_twice() {
    let first = release_and_present();
    let second = release_and_present();
    assert_eq!(first.to_bytes().unwrap(), second.to_bytes().unwrap());
    let a = replay(&first, ReplayOptions::default()).unwrap();
    let b = replay(&first, ReplayOptions::default()).unwrap();
    assert_eq!(a.render, b.render);
    assert_eq!(a.ring, b.ring);
}

/// A command buffer the CPU writes after its submission, behind an acquire (decision 8): the
/// replay reads what the front read after the CPU wrote, not what memory held at the start.
#[test]
fn a_gated_command_buffer_written_after_submission_replays() {
    let mut builder = builder();
    let channel = channel(&mut builder);
    let mut gate = PushBuffer::new();
    acquire(&mut gate, SEM, 1);
    let gate = gate.into_words();
    let later = releases(SEM2, 7);
    builder.cpu_write(SPACE, CMD, &bytes(&gate)).unwrap();
    // The second entry's words are not written yet: the CPU writes them after the submission.
    let submission = plain(vec![raw_entry(CMD, gate.len()), raw_entry(CMD2, later.len())]);
    builder.submit(channel, submission).unwrap();
    builder.run().unwrap();
    builder.cpu_write(SPACE, CMD2, &bytes(&later)).unwrap();
    builder.cpu_write(SPACE, SEM, &1u32.to_le_bytes()).unwrap();
    builder.run().unwrap();
    let mut word = [0; 4];
    builder.cpu_read(SPACE, SEM2, &mut word).unwrap();
    assert_eq!(u32::from_le_bytes(word), 7, "the gated buffer ran when the capture was made");
    let capture = Capture::parse(&builder.finish().unwrap().to_bytes().unwrap()).unwrap();
    let output = replay(&capture, ReplayOptions::default()).unwrap();
    assert_eq!(
        writes_text(&output.writes),
        format!("back space {} wrote 4 bytes at {SEM2:#x}: 07 00 00 00\n", SPACE.0)
    );
}

/// The index of the `n`-th front read (from 0) in the front log.
fn front_read(capture: &Capture, n: usize) -> usize {
    capture
        .front
        .iter()
        .enumerate()
        .filter(|(_, record)| matches!(record, FrontRecord::Access(Access::Read { .. })))
        .nth(n)
        .map(|(index, _)| index)
        .unwrap()
}

#[test]
fn a_read_moved_to_another_address_is_a_divergence_at_its_record() {
    let mut capture = release_and_present();
    let index = front_read(&capture, 0);
    let FrontRecord::Access(Access::Read { va, len, .. }) = &mut capture.front[index] else {
        unreachable!()
    };
    let read_at = *va;
    let length = *len;
    *va += 4;
    let error = replay(&capture, ReplayOptions::default()).unwrap_err();
    assert_eq!(
        error,
        ReplayError::Diverged(Divergence {
            thread: Thread::Front,
            record: index,
            found: format!(
                "a read of {length} bytes at {:#x} in address space {}",
                read_at + 4,
                SPACE.0
            ),
            made: format!(
                "the GPU read {length} bytes at {read_at:#x} in address space {}",
                SPACE.0
            ),
        })
    );
    assert_eq!(
        error.to_string(),
        format!(
            "the replay diverged at front log record {index}: the capture has a read of {length} \
             bytes at {:#x} in address space {}, but the GPU read {length} bytes at {read_at:#x} \
             in address space {}",
            read_at + 4,
            SPACE.0,
            SPACE.0
        )
    );
}

#[test]
fn a_read_whose_bytes_were_swapped_makes_the_ring_differ_at_the_entry() {
    let mut capture = release_and_present();
    // The command words, with the payload changed: same length, another blob.
    let mut words = releases(SEM, 1);
    let payload = words.iter().position(|&w| w == 1).unwrap();
    words[payload] = 2;
    let other = capture.blobs.insert(&bytes(&words));
    let index = front_read(&capture, 0);
    let FrontRecord::Access(Access::Read { result, .. }) = &mut capture.front[index] else {
        unreachable!()
    };
    *result = Ok(other);
    let error = replay(&capture, ReplayOptions::default()).unwrap_err();
    // The release is the first entry the stream makes.
    assert_eq!(error, ReplayError::RingDiffers { index: 0 });
    assert_eq!(error.to_string(), "ring entry 0 differs from the capture's");
}

#[test]
fn a_command_buffer_read_twice_is_stored_once() {
    let mut builder = builder();
    let channel = channel(&mut builder);
    let words = releases(SEM, 0x5a5a_5a5a);
    let submission = submission_at(&builder, CMD, &words);
    builder.submit(channel, submission.clone()).unwrap();
    builder.submit(channel, submission).unwrap();
    builder.run().unwrap();
    let capture = builder.finish().unwrap();
    let hash = super::super::ContentHash::of(&bytes(&words));
    let reads = capture
        .front
        .iter()
        .filter(|record| {
            matches!(record, FrontRecord::Access(Access::Read { result: Ok(blob), .. }) if *blob == hash)
        })
        .count();
    assert_eq!(reads, 2, "the front read the buffer twice");
    let file = capture.to_bytes().unwrap();
    let needle = bytes(&words);
    let stored = file
        .windows(needle.len())
        .filter(|window| *window == needle.as_slice())
        .count();
    assert_eq!(stored, 1, "the buffer's bytes are in the file once");
    replay(&Capture::parse(&file).unwrap(), ReplayOptions::default()).unwrap();
}

#[test]
fn until_present_stops_the_back_after_that_present() {
    let mut builder = builder();
    let channel = channel(&mut builder);
    let first = submission_at(&builder, CMD, &releases(SEM, 1));
    builder.submit(channel, first).unwrap();
    builder.present(present_request()).unwrap();
    builder.run().unwrap();
    let second = submission_at(&builder, CMD2, &releases(SEM2, 2));
    builder.submit(channel, second).unwrap();
    builder.present(present_request()).unwrap();
    builder.run().unwrap();
    let capture = builder.finish().unwrap();

    let whole = replay(&capture, ReplayOptions::default()).unwrap();
    assert_eq!((whole.presents, whole.stopped_early), (2, false));

    let one = replay(
        &capture,
        ReplayOptions {
            until_present: Some(1),
        },
    )
    .unwrap();
    assert_eq!((one.presents, one.stopped_early), (1, true));
    assert!(matches!(one.ring.last(), Some(RingEntry::Present(_))));
    assert_eq!(
        one.ring.len(),
        whole
            .ring
            .iter()
            .position(|entry| matches!(entry, RingEntry::Present(_)))
            .unwrap()
            + 1
    );
    // The second release has not been applied, so it was not written.
    assert_eq!(one.writes.len(), 1);

    // Asking for more presents than the capture has replays all of it.
    let past = replay(
        &capture,
        ReplayOptions {
            until_present: Some(3),
        },
    )
    .unwrap();
    assert_eq!((past.presents, past.stopped_early), (2, false));
}

#[test]
fn a_ring_hash_the_capture_lacks_is_a_length_difference() {
    let mut capture = release_and_present();
    let last_ring = capture
        .outputs
        .iter()
        .rposition(|output| matches!(output, Output::Ring { .. }))
        .unwrap();
    let rings = capture
        .outputs
        .iter()
        .filter(|output| matches!(output, Output::Ring { .. }))
        .count();
    capture.outputs.remove(last_ring);
    assert_eq!(
        replay(&capture, ReplayOptions::default()).unwrap_err(),
        ReplayError::RingLength {
            captured: rings - 1,
            replayed: rings,
        }
    );
}

#[test]
fn a_render_stream_that_differs_is_reported_after_a_whole_replay() {
    let mut capture = release_and_present();
    let other = capture.blobs.insert(b"not the stream");
    for output in &mut capture.outputs {
        if let Output::Render { text } = output {
            *text = other;
        }
    }
    assert_eq!(
        replay(&capture, ReplayOptions::default()).unwrap_err(),
        ReplayError::RenderDiffers
    );
    // A replay stopped at a present, before the back's log ends, does not compare it.
    let early = replay(
        &capture,
        ReplayOptions {
            until_present: Some(1),
        },
    )
    .unwrap();
    assert!(early.stopped_early);
}

#[test]
fn an_extra_back_step_diverges_at_the_end_of_the_ring() {
    let mut capture = release_and_present();
    capture.back.push(BackRecord::Apply);
    let applied = capture
        .back
        .iter()
        .filter(|record| matches!(record, BackRecord::Apply))
        .count();
    assert_eq!(
        replay(&capture, ReplayOptions::default()).unwrap_err(),
        ReplayError::Diverged(Divergence {
            thread: Thread::Back,
            record: capture.back.len() - 1,
            found: "an applied ring entry".to_owned(),
            made: format!("the front pushed only {} entries", applied - 1),
        })
    );
}

#[test]
fn a_release_that_latches_a_parked_acquire_reaches_the_front_as_a_latch() {
    let mut builder = builder();
    let waiting = channel(&mut builder);
    let releasing = channel(&mut builder);
    let gated = submission_at(&builder, CMD, &acquires_then_releases(SEM, 1, SEM2, 9));
    builder.submit(waiting, gated).unwrap();
    builder.run().unwrap();
    let release = submission_at(&builder, CMD2, &releases(SEM, 1));
    builder.submit(releasing, release).unwrap();
    builder.run().unwrap();
    let mut word = [0; 4];
    builder.cpu_read(SPACE, SEM2, &mut word).unwrap();
    assert_eq!(u32::from_le_bytes(word), 9, "the parked channel ran on");
    let capture = builder.finish().unwrap();
    assert!(capture.front.contains(&FrontRecord::Latch { channel: waiting }));
    assert!(
        capture
            .back
            .iter()
            .any(|record| matches!(record, BackRecord::Waiters { waiters } if !waiters.is_empty()))
    );
    let output = replay(&capture, ReplayOptions::default()).unwrap();
    assert_eq!(
        writes_text(&output.writes),
        format!(
            "back space {space} wrote 4 bytes at {SEM:#x}: 01 00 00 00\n\
             back space {space} wrote 4 bytes at {SEM2:#x}: 09 00 00 00\n",
            space = SPACE.0
        )
    );

    // Without the latch, the front does what it did not do when the capture was made.
    let mut unlatched = capture;
    unlatched
        .front
        .retain(|record| !matches!(record, FrontRecord::Latch { .. }));
    assert!(matches!(
        replay(&unlatched, ReplayOptions::default()).unwrap_err(),
        ReplayError::Diverged(Divergence {
            thread: Thread::Front,
            ..
        })
    ));
}

#[test]
fn a_fence_wait_reads_the_syncpoint_value_the_back_left() {
    let mut builder = builder();
    let first = channel(&mut builder);
    let second = channel(&mut builder);
    let mut increments = submission_at(&builder, CMD, &releases(SEM, 1));
    increments.increments = 1;
    builder.submit(first, increments).unwrap();
    let mut waits = submission_at(&builder, CMD2, &releases(SEM2, 2));
    waits.wait = Some(crate::interface::Fence {
        id: syncpoint(),
        value: 1,
    });
    builder.submit(second, waits).unwrap();
    builder.run().unwrap();
    let capture = builder.finish().unwrap();
    assert!(capture.front.contains(&FrontRecord::Syncpoint {
        id: syncpoint(),
        value: 1
    }));
    let output = replay(&capture, ReplayOptions::default()).unwrap();
    assert_eq!(output.writes.len(), 2);

    let mut unset = capture;
    unset
        .front
        .retain(|record| !matches!(record, FrontRecord::Syncpoint { .. }));
    assert!(replay(&unset, ReplayOptions::default()).is_err());
}
