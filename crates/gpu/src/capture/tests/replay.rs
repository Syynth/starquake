//! Whole captures: built from hand-written streams, written, read back and replayed.

use super::*;
use crate::capture::{
    Access, BackRecord, Capture, Divergence, FrontRecord, Output, ReplayError, ReplayOptions,
    Thread, entry_hash, replay, writes_text,
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
    let output = replay(capture.clone(), ReplayOptions::default()).unwrap();
    assert_eq!(output.presents, 1);
    assert!(!output.stopped_early);
    insta::assert_snapshot!("release_and_present_ring", write_text(&output.ring));
    insta::assert_snapshot!("release_and_present_render", output.render);
    assert_eq!(
        writes_text(&output.writes, &output.blobs),
        format!(
            "back space {} wrote 4 bytes at {SEM:#x}: 01 00 00 00\n",
            SPACE.0
        )
    );
}

#[test]
fn a_replay_is_the_same_twice_and_the_builder_is_the_same_twice() {
    let first = release_and_present();
    let second = release_and_present();
    assert_eq!(first.to_bytes().unwrap(), second.to_bytes().unwrap());
    let a = replay(first.clone(), ReplayOptions::default()).unwrap();
    let b = replay(first.clone(), ReplayOptions::default()).unwrap();
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
    let submission = plain(vec![
        raw_entry(CMD, gate.len()),
        raw_entry(CMD2, later.len()),
    ]);
    builder.submit(channel, submission).unwrap();
    builder.run().unwrap();
    builder.cpu_write(SPACE, CMD2, &bytes(&later)).unwrap();
    builder.cpu_write(SPACE, SEM, &1u32.to_le_bytes()).unwrap();
    builder.run().unwrap();
    let mut word = [0; 4];
    builder.cpu_read(SPACE, SEM2, &mut word).unwrap();
    assert_eq!(
        u32::from_le_bytes(word),
        7,
        "the gated buffer ran when the capture was made"
    );
    let capture = Capture::parse(&builder.finish().unwrap().to_bytes().unwrap()).unwrap();
    let output = replay(capture.clone(), ReplayOptions::default()).unwrap();
    assert_eq!(
        writes_text(&output.writes, &output.blobs),
        format!(
            "back space {} wrote 4 bytes at {SEM2:#x}: 07 00 00 00\n",
            SPACE.0
        )
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
    let error = replay(capture.clone(), ReplayOptions::default()).unwrap_err();
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
    let error = replay(capture.clone(), ReplayOptions::default()).unwrap_err();
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
    replay(Capture::parse(&file).unwrap(), ReplayOptions::default()).unwrap();
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

    let whole = replay(capture.clone(), ReplayOptions::default()).unwrap();
    assert_eq!((whole.presents, whole.stopped_early), (2, false));

    let one = replay(
        capture.clone(),
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
        capture.clone(),
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
        replay(capture.clone(), ReplayOptions::default()).unwrap_err(),
        ReplayError::RingLength {
            captured: rings - 1,
            replayed: rings,
        }
    );
    // With an entry that differs too, the difference is reported first.
    let ring_index = capture
        .outputs
        .iter()
        .position(|output| matches!(output, Output::Ring { .. }))
        .unwrap();
    capture.outputs[ring_index] = Output::Ring {
        hash: super::super::ContentHash::of(b"another entry"),
    };
    assert_eq!(
        replay(capture.clone(), ReplayOptions::default()).unwrap_err(),
        ReplayError::RingDiffers { index: 0 }
    );
    // And a capture with one ring hash more than the front pushes.
    let mut longer = release_and_present();
    longer.outputs.insert(
        0,
        Output::Ring {
            hash: super::super::ContentHash::of(b"x"),
        },
    );
    let mut longer_ok = release_and_present();
    let extra = longer_ok
        .outputs
        .iter()
        .find(|output| matches!(output, Output::Ring { .. }))
        .copied()
        .unwrap();
    let at = longer_ok
        .outputs
        .iter()
        .rposition(|output| matches!(output, Output::Ring { .. }))
        .unwrap();
    longer_ok.outputs.insert(at + 1, extra);
    assert_eq!(
        replay(longer_ok.clone(), ReplayOptions::default()).unwrap_err(),
        ReplayError::RingLength {
            captured: rings + 1,
            replayed: rings,
        }
    );
    assert_eq!(
        replay(longer.clone(), ReplayOptions::default()).unwrap_err(),
        ReplayError::RingDiffers { index: 0 }
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
        replay(capture.clone(), ReplayOptions::default()).unwrap_err(),
        ReplayError::RenderDiffers
    );
    // A replay stopped at a present, before the back's log ends, does not compare it.
    let early = replay(
        capture.clone(),
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
        replay(capture.clone(), ReplayOptions::default()).unwrap_err(),
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
    assert!(
        capture
            .front
            .contains(&FrontRecord::Latch { channel: waiting })
    );
    assert!(
        capture
            .back
            .iter()
            .any(|record| matches!(record, BackRecord::Waiters { waiters } if !waiters.is_empty()))
    );
    let output = replay(capture.clone(), ReplayOptions::default()).unwrap();
    assert_eq!(
        writes_text(&output.writes, &output.blobs),
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
        replay(unlatched.clone(), ReplayOptions::default()).unwrap_err(),
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
    let output = replay(capture.clone(), ReplayOptions::default()).unwrap();
    assert_eq!(output.writes.len(), 2);

    let mut unset = capture;
    unset
        .front
        .retain(|record| !matches!(record, FrontRecord::Syncpoint { .. }));
    assert!(replay(unset.clone(), ReplayOptions::default()).is_err());
}

/// A submission's `u32::MAX` increments are one `ring` record, the hash of the entry with its
/// count (the format stays version 1: no capture has shipped), and the capture replays; the
/// same entry with a count of 1 in that record's place is a difference at its index.
#[test]
fn a_counted_syncpoint_entry_is_one_ring_record() {
    let mut builder = builder();
    let channel = channel(&mut builder);
    let mut submission = submission_at(&builder, CMD, &releases(SEM, 1));
    submission.increments = u32::MAX;
    builder.submit(channel, submission).unwrap();
    builder.run().unwrap();
    let mut capture = Capture::parse(&builder.finish().unwrap().to_bytes().unwrap()).unwrap();
    let counted = RingEntry::Syncpoint {
        id: syncpoint(),
        count: u32::MAX,
    };
    let hash = entry_hash(&counted).unwrap();
    let rings: Vec<usize> = capture
        .outputs
        .iter()
        .enumerate()
        .filter(|(_, output)| matches!(output, Output::Ring { .. }))
        .map(|(at, _)| at)
        .collect();
    let counted_at: Vec<usize> = rings
        .iter()
        .copied()
        .filter(|&at| capture.outputs[at] == Output::Ring { hash })
        .collect();
    assert_eq!(counted_at.len(), 1);
    let output = replay(capture.clone(), ReplayOptions::default()).unwrap();
    let syncpoints: Vec<&RingEntry<'_>> = output
        .ring
        .iter()
        .filter(|entry| matches!(entry, RingEntry::Syncpoint { .. }))
        .collect();
    assert_eq!(syncpoints, [&counted]);
    let index = rings.iter().position(|&at| at == counted_at[0]).unwrap();
    let one = RingEntry::Syncpoint {
        id: syncpoint(),
        count: 1,
    };
    capture.outputs[counted_at[0]] = Output::Ring {
        hash: entry_hash(&one).unwrap(),
    };
    assert_eq!(
        replay(capture, ReplayOptions::default()).unwrap_err(),
        ReplayError::RingDiffers { index }
    );
}

/// A `SYNC_WAIT` entry waits until the back reports the work before it performed: the front
/// learns that only from the back's report.
#[test]
fn a_sync_wait_entry_goes_on_when_the_recorded_report_reaches_the_front() {
    let mut builder = builder();
    let channel = channel(&mut builder);
    let first = releases(SEM, 1);
    let second = releases(SEM2, 2);
    builder.cpu_write(SPACE, CMD, &bytes(&first)).unwrap();
    builder.cpu_write(SPACE, CMD2, &bytes(&second)).unwrap();
    let waits = GpEntry {
        address: CMD2,
        length: u32::try_from(second.len()).unwrap(),
        sync: Sync::Wait,
        ..GpEntry::default()
    };
    let submission = plain(vec![
        raw_entry(CMD, first.len()),
        GpfifoEntry(waits.to_raw().unwrap()),
    ]);
    builder.submit(channel, submission).unwrap();
    builder.run().unwrap();
    let capture = builder.finish().unwrap();
    assert!(capture.front.iter().any(
        |record| matches!(record, FrontRecord::Report { markers, .. } if !markers.is_empty())
    ));
    let output = replay(capture.clone(), ReplayOptions::default()).unwrap();
    assert_eq!(output.writes.len(), 2, "both releases ran");

    let mut unreported = capture;
    unreported
        .front
        .retain(|record| !matches!(record, FrontRecord::Report { .. }));
    assert!(replay(unreported.clone(), ReplayOptions::default()).is_err());
}

/// A second command buffer and semaphore, apart from the others.
const CMD3: u64 = GPU_BASE + 6 * PAGE_SIZE;
const SEM3: u64 = SEM + 0x80;

/// Each `syncpoint` record sets the syncpoint to its value, whatever the replay's syncpoint held:
/// two records for one syncpoint (1, then 2), then a fence wait for 3 that the capture never
/// passed. Adding each record's value instead would reach 3 and run the blocked submission.
#[test]
fn each_syncpoint_record_sets_the_value_the_back_left() {
    let mut builder = builder();
    let increments = channel(&mut builder);
    let waits = channel(&mut builder);
    let mut first = submission_at(&builder, CMD, &releases(SEM, 1));
    first.increments = 1;
    builder.submit(increments, first).unwrap();
    builder.run().unwrap();
    let mut second = submission_at(&builder, CMD2, &releases(SEM2, 2));
    second.increments = 1;
    builder.submit(increments, second).unwrap();
    builder.run().unwrap();
    let mut blocked = submission_at(&builder, CMD3, &releases(SEM3, 3));
    blocked.wait = Some(crate::interface::Fence {
        id: syncpoint(),
        value: 3,
    });
    builder.submit(waits, blocked).unwrap();
    builder.run().unwrap();
    let mut word = [0; 4];
    builder.cpu_read(SPACE, SEM3, &mut word).unwrap();
    assert_eq!(u32::from_le_bytes(word), 0, "the wait for 3 never passed");
    let capture = builder.finish().unwrap();
    let values: Vec<u32> = capture
        .front
        .iter()
        .filter_map(|record| match record {
            FrontRecord::Syncpoint { id, value } if *id == syncpoint() => Some(*value),
            _ => None,
        })
        .collect();
    assert_eq!(values, [1, 2]);
    let output = replay(capture, ReplayOptions::default()).unwrap();
    assert_eq!(
        writes_text(&output.writes, &output.blobs),
        format!(
            "back space {space} wrote 4 bytes at {SEM:#x}: 01 00 00 00\n\
             back space {space} wrote 4 bytes at {SEM2:#x}: 02 00 00 00\n",
            space = SPACE.0
        )
    );
}

/// `release_and_present()` with its back log cut just after the present's entry was applied:
/// after the `apply` record and the reads it made, before the back's next step.
fn ending_at_the_present() -> Capture {
    let mut capture = release_and_present();
    let last_apply = capture
        .back
        .iter()
        .rposition(|record| matches!(record, BackRecord::Apply))
        .unwrap();
    let next_step = capture.back[last_apply + 1..]
        .iter()
        .position(|record| {
            matches!(
                record,
                BackRecord::Apply
                    | BackRecord::Close
                    | BackRecord::Complete { .. }
                    | BackRecord::Waiters { .. }
            )
        })
        .unwrap();
    capture.back.truncate(last_apply + 1 + next_step);
    capture
}

/// `--until present=N` on a capture whose back log ends exactly at present `N` replays the
/// whole capture: it did not stop early, and the render stream is compared.
#[test]
fn until_present_at_the_back_logs_end_is_a_whole_replay() {
    let capture = ending_at_the_present();
    let whole = replay(capture.clone(), ReplayOptions::default()).unwrap();
    assert_eq!((whole.presents, whole.stopped_early), (1, false));
    let at_end = replay(
        capture.clone(),
        ReplayOptions {
            until_present: Some(1),
        },
    )
    .unwrap();
    assert_eq!((at_end.presents, at_end.stopped_early), (1, false));
    assert_eq!(at_end.render, whole.render);
    // The render stream is compared: another stream is a difference.
    let mut other = capture;
    let text = other.blobs.insert(b"not the stream");
    for output in &mut other.outputs {
        if let Output::Render { text: rendered } = output {
            *rendered = text;
        }
    }
    assert_eq!(
        replay(
            other,
            ReplayOptions {
                until_present: Some(1),
            },
        )
        .unwrap_err(),
        ReplayError::RenderDiffers
    );
}

/// A capture made in memory is checked as a file is before it replays: one without a render
/// stream is refused, so a replay that succeeds has always compared one.
#[test]
fn a_capture_without_a_render_stream_does_not_replay() {
    let mut capture = release_and_present();
    capture
        .outputs
        .retain(|output| !matches!(output, Output::Render { .. }));
    let error = replay(capture, ReplayOptions::default()).unwrap_err();
    assert_eq!(
        error,
        ReplayError::Malformed(crate::capture::CaptureError::RenderOutputs { count: 0 })
    );
    assert_eq!(
        error.to_string(),
        "the capture is malformed: the capture's outputs hold 0 render command streams; a \
         capture holds exactly one"
    );
}

/// The same bytes written many times are held once: each write names its blob in the capture's
/// table, which the output shares.
#[test]
fn repeated_writes_name_one_blob() {
    let mut builder = builder();
    let channel = channel(&mut builder);
    let words = releases(SEM, 0x77);
    for _ in 0..4 {
        let submission = submission_at(&builder, CMD, &words);
        builder.submit(channel, submission).unwrap();
    }
    builder.run().unwrap();
    let capture = builder.finish().unwrap();
    let blobs = capture.blobs.len();
    let output = replay(capture, ReplayOptions::default()).unwrap();
    assert_eq!(output.writes.len(), 4);
    let hash = super::super::ContentHash::of(&0x77u32.to_le_bytes());
    assert!(
        output
            .writes
            .iter()
            .all(|write| write.data == hash && write.len == 4)
    );
    assert_eq!(output.blobs.len(), blobs);
}

/// The writes' text: bytes up to [`WRITE_BYTES_SHOWN`] as hex, longer ones and blobs the table
/// lacks as their hash, and a fault after the bytes.
#[test]
fn the_writes_text_shows_short_writes_and_hashes_long_ones() {
    use crate::capture::{Blobs, ContentHash, ReplayedWrite, WRITE_BYTES_SHOWN};
    let mut blobs = Blobs::new();
    let short = vec![0xab; WRITE_BYTES_SHOWN];
    let long = vec![0xcd; WRITE_BYTES_SHOWN + 1];
    let write = |data: &[u8], result| ReplayedWrite {
        thread: Thread::Back,
        space: SPACE,
        va: 0x40,
        data: ContentHash::of(data),
        len: u64::try_from(data.len()).unwrap(),
        result,
    };
    blobs.insert(&short);
    blobs.insert(&long);
    let fault = crate::memory::GpuFault::ShimWrite { va: 0x40 };
    let writes = [
        write(&short, Ok(())),
        write(&long, Err(fault)),
        write(b"absent", Ok(())),
    ];
    let hex = vec!["ab"; WRITE_BYTES_SHOWN].join(" ");
    assert_eq!(
        writes_text(&writes, &blobs),
        format!(
            "back space 33 wrote 64 bytes at 0x40: {hex}\n\
             back space 33 wrote 65 bytes at 0x40: hash {} (it faulted: {fault})\n\
             back space 33 wrote 6 bytes at 0x40: hash {}\n",
            ContentHash::of(&long),
            ContentHash::of(b"absent")
        )
    );
}
