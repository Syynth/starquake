//! Tests of the GPU memory snapshot: its pinned layout, the checks a restore makes, and a
//! restored bookkeeping used afterwards.

use std::sync::Mutex;

use starquake_gpu::interface::VmUpdate;
use starquake_kernel::id::SlabId;
use starquake_kernel::memory::testing::device_share_token;

use super::*;
use crate::completion::testing::{Locks, process};

/// A front that keeps the calls it was given.
#[derive(Default)]
struct Front(Mutex<Vec<String>>);

impl GpuVmFront for Front {
    fn create_address_space(&self, space: AsId, _vm: Arc<GpuVm>) {
        self.0.lock().unwrap().push(format!("create {}", space.0));
    }

    fn update_vm(&self, space: AsId, update: VmUpdate, _done: Option<Ticket>) {
        self.0
            .lock()
            .unwrap()
            .push(format!("update {} {update:?}", space.0));
    }

    fn barrier(&self, done: Ticket) {
        self.0.lock().unwrap().push(format!("barrier {}", done.0));
    }
}

fn object(index: u32, generation: u32) -> ObjectId {
    ObjectId::from_parts(index, generation)
}

/// The start of NVN's window, where the golden state's mapping and region lie.
const W: u64 = 0x800_0000;
const PAGE: u64 = PAGE_SIZE;

/// A state with every part present and every value distinct and non-zero.
fn golden() -> State {
    State {
        events: vec![(3, object(0x11, 0x12))],
        worker: true,
        devices: vec![(5, 0x21, 0x22)],
        spaces: vec![SavedSpace {
            id: 2,
            records: vec![SavedRecord {
                start: W,
                size: 2 * PAGE,
                handle: 7,
                offset: PAGE,
                spans: vec![(W, PAGE, 0x31), (W + PAGE, PAGE, 0x32)],
            }],
            regions: vec![(W + 4 * PAGE, PAGE, 0x33)],
            shim: Some(PAGE),
        }],
        registrations: vec![SavedRegistration {
            handle: 7,
            start: 4 * PAGE,
            size: 3 * PAGE,
            device: 0x41,
            process: process(),
        }],
        pending: vec![SavedJob {
            ticket: 9,
            event: object(0x11, 0x12),
            releases: vec![(process(), 4 * PAGE, 3 * PAGE)],
            reveal: vec![(2, W, 2 * PAGE)],
        }],
        next: (3, 8, 0x34, 0x0A),
    }
}

fn encode(state: &State) -> Vec<u8> {
    let mut bytes = postcard::to_allocvec(&Header {
        magic: SNAPSHOT_MAGIC,
        version: SNAPSHOT_VERSION,
    })
    .unwrap();
    bytes.extend(postcard::to_allocvec(state).unwrap());
    bytes
}

/// A kernel holding `count` locks on the golden registration's range.
fn kernel(count: u32) -> Locks {
    Locks {
        process: process(),
        live: [((4 * PAGE, 3 * PAGE), count)].into_iter().collect(),
    }
}

#[test]
fn the_snapshot_layout_is_pinned() {
    // Varints, by hand: W = 2^27 is 80 80 80 40; a page (2^12) is 80 20.
    let w = [0x80, 0x80, 0x80, 0x40];
    let w1 = [0x80, 0xA0, 0x80, 0x40]; // W + one page
    let w4 = [0x80, 0x80, 0x81, 0x40]; // W + four pages
    let page = [0x80, 0x20];
    let two = [0x80, 0x40];
    let three = [0x80, 0x60];
    let four = [0x80, 0x80, 0x01];
    let (pi, pg) = (0x02, 0x03); // the process: `from_parts(2, 3)`.
    let mut expected: Vec<u8> = b"SQGM".to_vec();
    expected.push(1); // version
    expected.extend([1, 3, 0x11, 0x12]); // events: one, index 3, object
    expected.push(1); // worker
    expected.extend([1, 5, 0x21, 0x22]); // devices: one, id, name, label
    expected.extend([1, 2]); // spaces: one, id 2
    expected.push(1); // its records: one
    expected.extend(w);
    expected.extend(two);
    expected.push(7); // handle
    expected.extend(page); // offset
    expected.push(2); // spans: two
    expected.extend(w);
    expected.extend(page);
    expected.push(0x31);
    expected.extend(w1);
    expected.extend(page);
    expected.push(0x32);
    expected.push(1); // regions: one
    expected.extend(w4);
    expected.extend(page);
    expected.push(0x33);
    expected.push(1); // shim: Some
    expected.extend(page);
    expected.push(1); // registrations: one
    expected.push(7);
    expected.extend(four);
    expected.extend(three);
    expected.push(0x41);
    expected.extend([pi, pg]);
    expected.push(1); // jobs: one
    expected.push(9);
    expected.extend([0x11, 0x12]);
    expected.push(1); // releases: one
    expected.extend([pi, pg]);
    expected.extend(four);
    expected.extend(three);
    expected.push(1); // reveal: one
    expected.push(2);
    expected.extend(w);
    expected.extend(two);
    expected.extend([3, 8, 0x34, 0x0A]); // next ids
    assert_eq!(encode(&golden()), expected);
    let front = Front::default();
    let restored = GpuMemory::restore(&expected, &kernel(2), &front).unwrap();
    assert_eq!(restored.save().unwrap(), expected);
}

/// A restored bookkeeping gives the front its table and the waiting job's barrier, and the job
/// completes against it as before.
#[test]
fn a_restored_job_completes() {
    let bytes = encode(&golden());
    let front = Front::default();
    let mut restored = GpuMemory::restore(&bytes, &kernel(2), &front).unwrap();
    assert_eq!(
        front.0.lock().unwrap().as_slice(),
        ["create 2".to_owned(), "barrier 9".to_owned()]
    );
    // The table holds the registration's own mapping (not in the window: its CPU address), the
    // 211 mapping, the region and the shim.
    let space = AsId(2);
    assert_eq!(restored.mappings(space).len(), 5);
    let finished = restored.ordered_done(Ticket(9)).unwrap();
    assert_eq!(
        finished.releases,
        [device_share_token(process(), 4 * PAGE, 3 * PAGE)]
    );
    assert_eq!(finished.event, object(0x11, 0x12));
    assert_eq!(restored.pending_jobs(), 0);
    // And the next ticket is the saved one.
    let event = object(0x11, 0x12);
    let modify = super::super::memory::ModifyRequest {
        space,
        first: W,
        last: W + PAGE - 1,
        kind: 6,
        fallback: 6,
    };
    restored.modify_kind(&front, &modify, event).unwrap();
    assert_eq!(front.0.lock().unwrap().last().unwrap(), "barrier 10");
}

/// With two locks held, the registration's and the job's tokens are taken back; with one, the
/// second (the job's, index 1) is refused.
#[test]
fn locks_beyond_the_kernel_s_are_refused_at_the_first_extra_one() {
    let bytes = encode(&golden());
    let front = Front::default();
    assert!(GpuMemory::restore(&bytes, &kernel(2), &front).is_ok());
    let refused = GpuMemory::restore(&bytes, &kernel(1), &front);
    assert_eq!(
        refused,
        Err(SnapshotError::Release {
            index: 1,
            error: DeviceShareError::NotLocked {
                address: 4 * PAGE,
                size: 3 * PAGE,
            },
        })
    );
}

/// Each check refuses its fault, with the error it names; where two faults are present, the
/// one checked first answers.
#[test]
fn every_check_refuses_its_fault() {
    type Break = fn(&mut State);
    let cases: [(Break, SnapshotError); 14] = [
        (
            |s| s.next.0 = 0,
            SnapshotError::OutOfRange {
                part: Part::Counters,
                index: 0,
            },
        ),
        // An event index past the range, and a device 0: the event first.
        (
            |s| {
                s.events[0].0 = NOTIFICATION_COUNT;
                s.devices[0].0 = 0;
            },
            SnapshotError::OutOfRange {
                part: Part::Event,
                index: 0,
            },
        ),
        (
            |s| s.devices[0].0 = 0,
            SnapshotError::OutOfRange {
                part: Part::Device,
                index: 0,
            },
        ),
        (
            |s| s.spaces[0].id = 3,
            SnapshotError::OutOfRange {
                part: Part::Space,
                index: 0,
            },
        ),
        (
            |s| s.registrations[0].size = 1,
            SnapshotError::OutOfRange {
                part: Part::Registration,
                index: 0,
            },
        ),
        (
            |s| s.spaces[0].regions[0].0 = W - PAGE,
            SnapshotError::OutOfRange {
                part: Part::Region,
                index: 0,
            },
        ),
        (
            |s| s.spaces[0].regions[0].2 = 0x34,
            SnapshotError::OutOfRange {
                part: Part::Region,
                index: 0,
            },
        ),
        (
            |s| s.spaces[0].records[0].handle = 6,
            SnapshotError::Dangling {
                part: Part::Mapping,
                index: 0,
            },
        ),
        // The offset at its edge is accepted below; one page more is refused.
        (
            |s| s.spaces[0].records[0].offset = 2 * PAGE,
            SnapshotError::OutOfRange {
                part: Part::Mapping,
                index: 0,
            },
        ),
        (
            |s| s.spaces[0].records[0].spans[1].0 += PAGE,
            SnapshotError::Order {
                part: Part::Span,
                index: 1,
            },
        ),
        (
            |s| s.spaces[0].records[0].spans.pop().map(drop).unwrap_or(()),
            SnapshotError::OutOfRange {
                part: Part::Mapping,
                index: 0,
            },
        ),
        (
            |s| s.pending[0].ticket = 0x0A,
            SnapshotError::OutOfRange {
                part: Part::Job,
                index: 0,
            },
        ),
        (
            |s| s.pending[0].event = object(1, 1),
            SnapshotError::Dangling {
                part: Part::Job,
                index: 0,
            },
        ),
        (
            |s| s.pending[0].reveal[0].0 = 1,
            SnapshotError::Dangling {
                part: Part::Job,
                index: 0,
            },
        ),
    ];
    let front = Front::default();
    for (index, (fault, error)) in cases.into_iter().enumerate() {
        let mut state = golden();
        fault(&mut state);
        assert_eq!(
            GpuMemory::restore(&encode(&state), &kernel(2), &front).err(),
            Some(error),
            "case {index}"
        );
    }
    // Edges accepted: the offset that ends the mapping at the registration's end, the last
    // event index, the last device id.
    let mut state = golden();
    state.spaces[0].records[0].offset = PAGE;
    state.events[0].0 = NOTIFICATION_COUNT - 1;
    state.devices[0].0 = u8::try_from(DEVICE_SLOTS - 1).unwrap();
    assert!(GpuMemory::restore(&encode(&state), &kernel(2), &front).is_ok());
}

/// Regions that overlap are refused; a neighbour exactly at the end is accepted.
#[test]
fn regions_may_touch_but_not_overlap() {
    let front = Front::default();
    let mut state = golden();
    state.spaces[0].regions.push((W + 5 * PAGE, PAGE, 0x32));
    assert!(GpuMemory::restore(&encode(&state), &kernel(2), &front).is_ok());
    state.spaces[0].regions[1].0 = W + 4 * PAGE;
    assert_eq!(
        GpuMemory::restore(&encode(&state), &kernel(2), &front).err(),
        Some(SnapshotError::Order {
            part: Part::Region,
            index: 1
        })
    );
    let mut state = golden();
    state.spaces[0].regions[0].1 = 2 * PAGE;
    state.spaces[0].regions.push((W + 5 * PAGE, PAGE, 0x32));
    assert_eq!(
        GpuMemory::restore(&encode(&state), &kernel(2), &front).err(),
        Some(SnapshotError::Overlap {
            part: Part::Region,
            index: 1
        })
    );
}

#[test]
fn the_header_and_the_tail_are_checked_first() {
    let front = Front::default();
    let mut bytes = encode(&golden());
    bytes[0] = b'X';
    assert_eq!(
        GpuMemory::restore(&bytes, &kernel(2), &front).err(),
        Some(SnapshotError::BadMagic)
    );
    let mut bytes = encode(&golden());
    bytes[4] = 2;
    assert_eq!(
        GpuMemory::restore(&bytes, &kernel(2), &front).err(),
        Some(SnapshotError::UnsupportedVersion { found: 2 })
    );
    let mut bytes = encode(&golden());
    bytes.push(0);
    assert_eq!(
        GpuMemory::restore(&bytes, &kernel(2), &front).err(),
        Some(SnapshotError::TrailingBytes { count: 1 })
    );
}

#[test]
fn a_refusal_is_said_in_plain_words() {
    let error = SnapshotError::Dangling {
        part: Part::Mapping,
        index: 2,
    };
    assert_eq!(
        error.to_string(),
        "GPU mapping 2 of the GPU memory snapshot names something that does not exist"
    );
}
