//! Tests of the GPU memory snapshot: its pinned layout, the checks a restore makes, and a
//! restored bookkeeping used afterwards.

use std::sync::Mutex;

use starquake_gpu::interface::{Target, VmUpdate};
use starquake_gpu::vm::Mapping;
use starquake_kernel::id::SlabId;
use starquake_kernel::memory::testing::device_share_token;

use super::super::memory::SPARSE_KIND;
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
    // region and the shim; the 211 mapping's pages are the waiting job's, held until it is done.
    let space = AsId(2);
    assert_eq!(restored.mappings(space).len(), 3);
    let finished = restored.ordered_done(Ticket(9)).unwrap();
    // Then its two runs of one kind show.
    assert_eq!(restored.mappings(space).len(), 5);
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

/// A change to the golden state that breaks one check.
type Break = fn(&mut State);

/// An out-of-range refusal of the first `part`.
fn out(part: Part) -> SnapshotError {
    SnapshotError::OutOfRange { part, index: 0 }
}

/// A dangling-reference refusal of the first `part`.
fn dangling(part: Part) -> SnapshotError {
    SnapshotError::Dangling { part, index: 0 }
}

/// An order refusal of the second `part`.
fn order(part: Part) -> SnapshotError {
    SnapshotError::Order { part, index: 1 }
}

/// An address space with nothing in it.
fn empty_space(id: u32) -> SavedSpace {
    SavedSpace {
        id,
        records: Vec::new(),
        regions: Vec::new(),
        shim: None,
    }
}

/// Each fault, and the error it must be refused with: the groups' lists, joined.
fn faults() -> Vec<(Break, SnapshotError)> {
    let mut all = list_faults();
    all.extend(space_faults());
    all.extend(job_faults());
    all
}

/// The faults of the counters, events, devices, address spaces and registrations.
fn list_faults() -> Vec<(Break, SnapshotError)> {
    vec![
        (|s| s.next.0 = 0, out(Part::Counters)),
        // An event index past the range, and a device 0: the event first.
        (
            |s| {
                s.events[0].0 = NOTIFICATION_COUNT;
                s.devices[0].0 = 0;
            },
            out(Part::Event),
        ),
        // The same object at two indices.
        (
            |s| s.events.push((4, s.events[0].1)),
            SnapshotError::OutOfRange {
                part: Part::Event,
                index: 1,
            },
        ),
        (
            |s| s.events.insert(0, (4, object(0x13, 0x14))),
            order(Part::Event),
        ),
        (|s| s.devices[0].0 = 0, out(Part::Device)),
        (|s| s.spaces[0].id = 3, out(Part::Space)),
        (|s| s.spaces.push(empty_space(1)), order(Part::Space)),
        (|s| s.registrations[0].size = 1, out(Part::Registration)),
        // The handle equal to the next one; the golden one, one below, is accepted.
        (|s| s.next.1 = 7, out(Part::Registration)),
        (
            |s| {
                s.registrations.push(SavedRegistration {
                    handle: 5,
                    start: 8 * PAGE,
                    size: PAGE,
                    device: 0x41,
                    process: process(),
                });
            },
            order(Part::Registration),
        ),
    ]
}

/// The faults of the address spaces' contents.
fn space_faults() -> Vec<(Break, SnapshotError)> {
    vec![
        (|s| s.spaces[0].regions[0].0 = W - PAGE, out(Part::Region)),
        (|s| s.spaces[0].regions[0].2 = 0x34, out(Part::Region)),
        // A region id another address space's region has.
        (
            |s| {
                let mut first = empty_space(1);
                first.regions.push((W, PAGE, s.spaces[0].regions[0].2));
                s.spaces.insert(0, first);
            },
            out(Part::Region),
        ),
        // A mapping that starts below NVN's window.
        (
            |s| s.spaces[0].records[0].start = W - PAGE,
            out(Part::Mapping),
        ),
        (|s| s.spaces[0].records[0].offset = 1, out(Part::Mapping)),
        (
            |s| {
                s.spaces[0].records.push(SavedRecord {
                    start: W,
                    size: PAGE,
                    handle: 7,
                    offset: 0,
                    spans: vec![(W, PAGE, 0x31)],
                });
            },
            order(Part::Mapping),
        ),
        (
            |s| {
                s.spaces[0].records.push(SavedRecord {
                    start: W + PAGE,
                    size: PAGE,
                    handle: 7,
                    offset: 0,
                    spans: vec![(W + PAGE, PAGE, 0x31)],
                });
            },
            SnapshotError::Overlap {
                part: Part::Mapping,
                index: 1,
            },
        ),
        (|s| s.spaces[0].shim = Some(1), out(Part::Shim)),
        (
            |s| s.spaces[0].records[0].handle = 6,
            dangling(Part::Mapping),
        ),
        // The offset at its edge is accepted below; one page more is refused.
        (
            |s| s.spaces[0].records[0].offset = 2 * PAGE,
            out(Part::Mapping),
        ),
        (
            |s| s.spaces[0].records[0].spans[1].0 += PAGE,
            SnapshotError::Order {
                part: Part::Span,
                index: 1,
            },
        ),
        (
            |s| s.spaces[0].records[0].spans.truncate(1),
            out(Part::Mapping),
        ),
    ]
}

/// The faults of the jobs.
fn job_faults() -> Vec<(Break, SnapshotError)> {
    vec![
        (|s| s.pending[0].ticket = 0x0A, out(Part::Job)),
        (|s| s.pending[0].event = object(1, 1), dangling(Part::Job)),
        (|s| s.pending[0].reveal[0].0 = 1, dangling(Part::Job)),
        (|s| s.pending[0].reveal[0].1 = 1, out(Part::Job)),
        (|s| s.pending[0].releases[0].1 = 1, out(Part::Job)),
        (|s| s.pending.push(s.pending[0].clone()), order(Part::Job)),
    ]
}

/// Each check refuses its fault, with the error it names; where two faults are present, the
/// one checked first answers.
#[test]
fn every_check_refuses_its_fault() {
    let front = Front::default();
    for (index, (fault, error)) in faults().into_iter().enumerate() {
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

/// A golden state with `regions` more one-page sparse regions after the golden one, their ids
/// distinct and below the next region id.
fn with_regions(regions: usize) -> State {
    let mut state = golden();
    let count = u32::try_from(regions).unwrap();
    for n in 0..count {
        let start = W + (5 + u64::from(n)) * PAGE;
        state.spaces[0].regions.push((start, PAGE, 0x40 + n));
    }
    state.next.2 = 0x40 + count;
    state
}

/// An address space at exactly [`MAX_PIECES`] pieces is accepted, one more is refused.
#[test]
fn the_pieces_of_an_address_space_are_bounded() {
    let front = Front::default();
    // The golden space holds four: the registration, a region and two runs of one kind.
    let at_limit = with_regions(MAX_PIECES - 4);
    assert!(GpuMemory::restore(&encode(&at_limit), &kernel(2), &front).is_ok());
    let over = with_regions(MAX_PIECES - 3);
    assert_eq!(
        GpuMemory::restore(&encode(&over), &kernel(2), &front).err(),
        Some(SnapshotError::TooMany {
            part: Part::Mapping
        })
    );
}

/// [`MAX_ADDRESS_SPACES`] address spaces are accepted, one more is refused.
#[test]
fn address_spaces_are_bounded() {
    let front = Front::default();
    let spaces = |count: u32| {
        let mut state = golden();
        let golden_space = state.spaces.remove(0);
        state.spaces = (1..=count)
            .map(|id| {
                if id == golden_space.id {
                    golden_space.clone()
                } else {
                    empty_space(id)
                }
            })
            .collect();
        state.next.0 = count + 1;
        state
    };
    let limit = u32::try_from(MAX_ADDRESS_SPACES).unwrap();
    assert!(GpuMemory::restore(&encode(&spaces(limit)), &kernel(2), &front).is_ok());
    assert_eq!(
        GpuMemory::restore(&encode(&spaces(limit + 1)), &kernel(2), &front).err(),
        Some(SnapshotError::TooMany { part: Part::Space })
    );
}

/// [`MAX_PIECES`] registrations are accepted (in an address space holding nothing else), one
/// more is refused.
#[test]
fn registrations_are_bounded() {
    let front = Front::default();
    let registrations = |count: usize| {
        let mut state = golden();
        state.spaces[0].records.clear();
        state.spaces[0].regions.clear();
        let template = state.registrations[0];
        let count = u32::try_from(count).unwrap();
        state.registrations = (1..=count)
            .map(|handle| SavedRegistration { handle, ..template })
            .collect();
        state.next.1 = count + 1;
        state
    };
    // Each registration's lock and the job's, all on the golden range.
    let locks = kernel(u32::try_from(MAX_PIECES).unwrap() + 2);
    assert!(GpuMemory::restore(&encode(&registrations(MAX_PIECES)), &locks, &front).is_ok());
    assert_eq!(
        GpuMemory::restore(&encode(&registrations(MAX_PIECES + 1)), &locks, &front).err(),
        Some(SnapshotError::TooMany {
            part: Part::Registration
        })
    );
}

/// [`MAX_PENDING_JOBS`] waiting jobs are accepted, one more is refused.
#[test]
fn waiting_jobs_are_bounded() {
    let front = Front::default();
    let jobs = |count: usize| {
        let mut state = golden();
        let template = state.pending[0].clone();
        let count = u64::try_from(count).unwrap();
        state.pending = (1..=count)
            .map(|ticket| SavedJob {
                ticket,
                releases: Vec::new(),
                ..template.clone()
            })
            .collect();
        state.next.3 = count + 1;
        state
    };
    assert!(GpuMemory::restore(&encode(&jobs(MAX_PENDING_JOBS)), &kernel(1), &front).is_ok());
    assert_eq!(
        GpuMemory::restore(&encode(&jobs(MAX_PENDING_JOBS + 1)), &kernel(1), &front).err(),
        Some(SnapshotError::TooMany { part: Part::Job })
    );
}

/// A restore does not publish the pages a waiting job holds: a 213 that was waiting inside a
/// sparse region leaves a hole in the new table until it is done, then the region shows there.
#[test]
fn a_restore_leaves_a_pending_unmap_a_hole_until_it_is_done() {
    let mut state = golden();
    state.spaces[0].records.clear();
    state.spaces[0].regions = vec![(W, 4 * PAGE, 0x33)];
    state.pending[0].releases.clear();
    state.pending[0].reveal = vec![(2, W + PAGE, PAGE)];
    let front = Front::default();
    let mut restored = GpuMemory::restore(&encode(&state), &kernel(1), &front).unwrap();
    let space = AsId(2);
    let region = |offset: u64, size: u64| Mapping {
        range: PageRange::new(W + offset, size).unwrap(),
        target: Target::GpuOnly {
            region: RegionId(0x33),
            offset,
        },
        kind: SPARSE_KIND,
    };
    let sparse = |memory: &GpuMemory| -> Vec<Mapping> {
        memory
            .mappings(space)
            .into_iter()
            .filter(|m| matches!(m.target, Target::GpuOnly { .. }))
            .collect()
    };
    assert_eq!(
        sparse(&restored),
        [region(0, PAGE), region(2 * PAGE, 2 * PAGE)]
    );
    restored.ordered_done(Ticket(9)).unwrap();
    assert_eq!(
        sparse(&restored),
        [
            region(0, PAGE),
            region(PAGE, PAGE),
            region(2 * PAGE, 2 * PAGE)
        ]
    );
}

/// Save and restore agree that a destroyed address space is gone: after the golden space (2)
/// is destroyed while its job waits, the save succeeds without it, the restore has only the
/// other space, the job still completes, and the next space takes a new id. A snapshot whose
/// job still names the destroyed space is refused.
#[test]
fn a_destroyed_space_is_not_saved_and_not_restored() {
    let front = Front::default();
    let mut memory = GpuMemory::restore(&encode(&golden()), &kernel(2), &front).unwrap();
    let destroyed = AsId(2);
    let other = memory.create_address_space(&front).unwrap();
    assert_eq!(other, AsId(3));
    memory.destroy_address_space(&front, destroyed).unwrap();

    let bytes = memory.save().unwrap();
    let front = Front::default();
    let mut restored = GpuMemory::restore(&bytes, &kernel(2), &front).unwrap();
    assert_eq!(restored.address_spaces().collect::<Vec<_>>(), [other]);
    assert!(restored.table(destroyed).is_none());
    assert_eq!(
        front.0.lock().unwrap().as_slice(),
        ["create 3".to_owned(), "barrier 9".to_owned()]
    );
    let finished = restored.ordered_done(Ticket(9)).unwrap();
    assert_eq!(finished.event, object(0x11, 0x12));
    assert_eq!(
        finished.releases,
        [device_share_token(process(), 4 * PAGE, 3 * PAGE)]
    );
    assert_eq!(restored.create_address_space(&front), Ok(AsId(4)));

    // The same snapshot with the job's range naming the destroyed space again.
    let (_, body) = postcard::take_from_bytes::<Header>(&bytes).unwrap();
    let mut state: State = postcard::from_bytes(body).unwrap();
    assert_eq!(state.pending[0].reveal, []);
    state.pending[0].reveal.push((destroyed.0, W, PAGE));
    assert_eq!(
        GpuMemory::restore(&encode(&state), &kernel(2), &Front::default()),
        Err(dangling(Part::Job))
    );
}
