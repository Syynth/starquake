//! The release timestamp in captures (issue #435; gpu-protocol, "Fenced completion": the
//! timestamp is guest time, "so a replay writes the same value"): the builder's back clock is a
//! fake guest clock ([`CaptureBuilder::set_timestamp`]), a capture records each reading of it
//! as a `timestamp` record, and a replay writes the recorded value.

use super::*;
use crate::capture::{
    Access, BackRecord, Capture, ContentHash, Divergence, ReplayError, ReplayOptions,
    ReplayOutput, Thread, replay,
};
use crate::sema::RELEASE_TIMESTAMP_OFFSET;

/// Two guest times: distinct in every byte, with the high half set, so a lost or swapped half
/// shows.
const FIRST: u64 = 0x0102_0304_0506_0708;
const SECOND: u64 = 0x1112_1314_1516_1718;
/// A guest time the builder never gave.
const FORGED: u64 = 0x2122_2324_2526_2728;

/// The payloads of the two releases.
const PAYLOAD: u32 = 0x5a5a_0001;
const PAYLOAD2: u32 = 0x6b6b_0002;

/// The form of the releases here: the 16-byte structure, a timestamp after the payload.
const FORM: ReleaseForm = ReleaseForm {
    size: PayloadSize::Four,
    timestamp: true,
};

/// A stream that releases `payload` at `address` with a timestamp (`SEMAPHORED`'s
/// `RELEASE_SIZE` 16 bytes).
fn releases_with_timestamp(address: u64, payload: u32) -> Vec<u32> {
    let word = sema::encode_semaphored(&HostSemaphore::Release {
        form: FORM,
        wfi: false,
    })
    .unwrap();
    let mut pb = PushBuffer::new();
    semaphored(&mut pb, address, payload, word);
    pb.into_words()
}

/// A channel that releases [`PAYLOAD`] at [`SEM`] while the guest clock reads [`FIRST`], then
/// [`PAYLOAD2`] at [`SEM2`] while it reads [`SECOND`]: built, written and read back.
fn two_timestamped_releases() -> Capture {
    let mut builder = builder();
    let channel = channel(&mut builder);
    builder.set_timestamp(FIRST);
    let first = submission_at(&builder, CMD, &releases_with_timestamp(SEM, PAYLOAD));
    builder.submit(channel, first).unwrap();
    builder.run().unwrap();
    builder.set_timestamp(SECOND);
    let second = submission_at(&builder, CMD2, &releases_with_timestamp(SEM2, PAYLOAD2));
    builder.submit(channel, second).unwrap();
    builder.run().unwrap();
    let capture = builder.finish().unwrap();
    Capture::parse(&capture.to_bytes().unwrap()).unwrap()
}

/// The 16 bytes a release of `payload` with `timestamp` writes: the payload widened to 64 bits,
/// then the timestamp at [`RELEASE_TIMESTAMP_OFFSET`].
fn structure(payload: u32, timestamp: u64) -> Vec<u8> {
    let mut bytes = u64::from(payload).to_le_bytes().to_vec();
    assert_eq!(u64::try_from(bytes.len()).unwrap(), RELEASE_TIMESTAMP_OFFSET);
    bytes.extend_from_slice(&timestamp.to_le_bytes());
    assert_eq!(u64::try_from(bytes.len()).unwrap(), FORM.bytes());
    bytes
}

/// The back's writes in `output`, as (address, bytes).
fn back_writes(output: &ReplayOutput) -> Vec<(u64, Vec<u8>)> {
    output
        .writes
        .iter()
        .filter(|write| write.thread == Thread::Back)
        .map(|write| {
            let bytes = output.blobs.get(write.data).unwrap().to_vec();
            assert_eq!(u64::try_from(bytes.len()).unwrap(), write.len);
            (write.va, bytes)
        })
        .collect()
}

/// The `timestamp` records of a back log, in order.
fn timestamps(capture: &Capture) -> Vec<u64> {
    capture
        .back
        .iter()
        .filter_map(|record| match record {
            BackRecord::Timestamp { value } => Some(*value),
            _ => None,
        })
        .collect()
}

/// The index of the back log's `n`th `timestamp` record.
fn timestamp_record(capture: &Capture, n: usize) -> usize {
    capture
        .back
        .iter()
        .enumerate()
        .filter(|(_, record)| matches!(record, BackRecord::Timestamp { .. }))
        .nth(n)
        .map(|(index, _)| index)
        .unwrap()
}

/// The index of the back log's write at `va`.
fn write_record(capture: &Capture, va: u64) -> usize {
    capture
        .back
        .iter()
        .position(|record| {
            matches!(record, BackRecord::Access(Access::Write { va: at, .. }) if *at == va)
        })
        .unwrap()
}

/// The capture records the guest time each 16-byte release read, in order, and the replay
/// writes each release's structure with the timestamp the capture recorded for it.
#[test]
fn a_replayed_capture_writes_the_timestamps_the_capture_recorded() {
    let capture = two_timestamped_releases();
    assert_eq!(timestamps(&capture), [FIRST, SECOND]);
    // Each reading is recorded before the write that carries it.
    assert!(timestamp_record(&capture, 0) < write_record(&capture, SEM));
    assert!(timestamp_record(&capture, 1) < write_record(&capture, SEM2));
    let output = replay(capture, ReplayOptions::default()).unwrap();
    assert_eq!(
        back_writes(&output),
        [
            (SEM, structure(PAYLOAD, FIRST)),
            (SEM2, structure(PAYLOAD2, SECOND)),
        ]
    );
}

/// The replay's timestamp is the recorded one, not a clock of its own: a capture whose first
/// `timestamp` record and the write it feeds both say [`FORGED`] replays to a write of
/// [`FORGED`], and the second release keeps its own.
#[test]
fn a_replay_writes_the_recorded_timestamp_whatever_its_value() {
    let mut capture = two_timestamped_releases();
    let at = timestamp_record(&capture, 0);
    capture.back[at] = BackRecord::Timestamp { value: FORGED };
    let forged = capture.blobs.insert(&structure(PAYLOAD, FORGED));
    let write = write_record(&capture, SEM);
    let BackRecord::Access(Access::Write { data, .. }) = &mut capture.back[write] else {
        unreachable!("found as a write");
    };
    *data = forged;
    let output = replay(capture, ReplayOptions::default()).unwrap();
    assert_eq!(
        back_writes(&output),
        [
            (SEM, structure(PAYLOAD, FORGED)),
            (SEM2, structure(PAYLOAD2, SECOND)),
        ]
    );
}

/// A `timestamp` record changed alone makes the release write bytes the capture does not hold:
/// the replay diverges at that write, naming both blobs.
#[test]
fn a_changed_timestamp_record_diverges_at_the_write_it_feeds() {
    let mut capture = two_timestamped_releases();
    let at = timestamp_record(&capture, 1);
    capture.back[at] = BackRecord::Timestamp { value: FORGED };
    let write = write_record(&capture, SEM2);
    let recorded = ContentHash::of(&structure(PAYLOAD2, SECOND));
    let made = ContentHash::of(&structure(PAYLOAD2, FORGED));
    let error = replay(capture, ReplayOptions::default()).unwrap_err();
    assert_eq!(
        error,
        ReplayError::Diverged(Divergence {
            thread: Thread::Back,
            record: write,
            found: format!(
                "a write of blob {recorded} at {SEM2:#x} in address space {}",
                SPACE.0
            ),
            made: format!(
                "the GPU wrote blob {made} at {SEM2:#x} in address space {}",
                SPACE.0
            ),
        })
    );
}
