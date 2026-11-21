//! The semaphore mirror page (tentative, decision log 2026-11-20): a release in the second page
//! of a two-page registration is also written at the same offset in its first page, over the
//! production memory (`VmMemory`), whose table lists the registration.
//!
//! The rig's table maps four pages from [`BEFORE`]: a page before the registration, the
//! registration's two pages at [`S`], and a neighbour page right after it, which no list holds
//! (a one-page registration, which gsv never lists). Every byte starts as [`FILL`].

use std::sync::atomic::AtomicU64;

use super::*;
use crate::memory::VmMemory;
use crate::vm::{SEMAPHORE_MIRROR_OFFSET, SEMAPHORE_REGISTRATION_BYTES, SemaphoreRegistration};

/// The registration's first page: above 32 bits, so a truncated address misses it.
const S: u64 = 0x3_0000_1000;
/// The page before it, the first the rig maps.
const BEFORE: u64 = S - PAGE_SIZE;
/// The registration's second page.
const SECOND: u64 = S + SEMAPHORE_MIRROR_OFFSET;
/// The neighbour: the page that starts exactly at the registration's end.
const NEIGHBOUR: u64 = S + SEMAPHORE_REGISTRATION_BYTES;
/// Pages the rig maps from [`BEFORE`].
const MAPPED: u64 = 4;

fn registration() -> SemaphoreRegistration {
    SemaphoreRegistration::new(PageRange::new(S, SEMAPHORE_REGISTRATION_BYTES).unwrap()).unwrap()
}

/// The production memory over a table listing [`registration`] and mapping `pages` (each a GPU
/// page start among the rig's four) to guest memory filled with [`FILL`].
struct Mirrored {
    vm: Arc<GpuVm>,
    mem: Arc<VmMemory<FlatMemory>>,
}

fn mirrored_over(r: &Rig, pages: &[u64]) -> Mirrored {
    let size = usize::try_from(MAPPED * PAGE_SIZE).unwrap();
    let guest = Arc::new(FlatMemory::new(CPU_BASE, size));
    guest.write(CPU_BASE, &vec![FILL; size]).unwrap();
    let vm = Arc::new(GpuVm::new());
    for &page in pages {
        vm.map(
            PageRange::new(page, PAGE_SIZE).unwrap(),
            Target::Guest {
                cpu_va: CPU_BASE + (page - BEFORE),
            },
            Kind(0),
        )
        .unwrap();
    }
    vm.add_semaphore_registration(registration());
    let mem = Arc::new(VmMemory::new(
        Arc::clone(&vm),
        guest,
        Arc::clone(&r.syncpoints),
    ));
    Mirrored { vm, mem }
}

/// All four pages mapped.
fn mirrored(r: &Rig) -> Mirrored {
    mirrored_over(r, &[BEFORE, S, SECOND, NEIGHBOUR])
}

impl Mirrored {
    fn action(&self, release: Release) -> FencedAction {
        FencedAction::Release {
            memory: Arc::<VmMemory<FlatMemory>>::clone(&self.mem),
            space: SPACE,
            release,
            failure: FailureLog::Warn,
        }
    }

    fn bytes(&self, va: u64, len: usize) -> Vec<u8> {
        let mut bytes = vec![0; len];
        self.mem.read(va, &mut bytes).unwrap();
        bytes
    }

    fn word(&self, va: u64) -> u32 {
        u32::from_le_bytes(self.bytes(va, 4).try_into().unwrap())
    }

    /// Whether the `len` bytes at `va` are untouched.
    fn untouched(&self, va: u64, len: usize) -> bool {
        self.bytes(va, len) == vec![FILL; len]
    }
}

fn sixteen(address: u64, payload: u64) -> Release {
    Release {
        form: ReleaseForm {
            size: PayloadSize::Eight,
            timestamp: true,
        },
        ..four(address, payload)
    }
}

/// The 16 bytes [`sixteen`] writes with the rig's clock: the payload, then [`TIMESTAMP`].
fn sixteen_bytes(payload: u64) -> Vec<u8> {
    [payload.to_le_bytes(), TIMESTAMP.to_le_bytes()].concat()
}

/// The issue's case: a two-page registration and a release at +0x1010; the word at +0x10 holds
/// the payload too.
#[test]
fn a_release_in_the_second_page_is_copied_to_the_first() {
    let mut r = rig();
    let m = mirrored(&r);
    let attached = r.fences.attach(m.action(four(S + 0x1010, 0xdead_beef)));
    assert_eq!(attached, Ok(performed(false)));
    assert_eq!(m.word(S + 0x1010), 0xdead_beef);
    assert_eq!(m.word(S + 0x10), 0xdead_beef);
    // Only the release's 4 bytes, at each place.
    assert!(m.untouched(S + 0xc, 4));
    assert!(m.untouched(S + 0x14, 4));
    assert!(m.untouched(S + 0x100c, 4));
    assert!(m.untouched(S + 0x1014, 4));
}

/// The memory without a table answers no mirror (the trait's default), so the rig's other tests
/// copy nothing.
#[test]
fn a_memory_without_a_table_lists_no_mirror() {
    let r = rig();
    assert_eq!(r.mem.semaphore_mirror(SEM), None);
    assert_eq!(r.mem.semaphore_mirror(SEM + SEMAPHORE_MIRROR_OFFSET), None);
}

/// Each edge of the second page on both sides, with the neighbour (a one-page registration
/// starting exactly at the end bound, not listed) and the page before it watched.
#[test]
fn only_a_release_starting_in_the_second_page_is_copied() {
    for (address, copied) in [
        (S + 0x10, None),
        (SECOND - 4, None),
        (SECOND, Some(S)),
        (NEIGHBOUR - 4, Some(SECOND - 4)),
        (NEIGHBOUR, None),
        (NEIGHBOUR + 0x10, None),
    ] {
        let mut r = rig();
        let m = mirrored(&r);
        r.fences
            .attach(m.action(four(address, 0x5a5a_0001)))
            .unwrap();
        assert_eq!(m.word(address), 0x5a5a_0001, "{address:#x}");
        // The release's 4 bytes and, when copied, the copy's 4 bytes changed; nothing else in
        // the four pages did.
        let all = m.bytes(BEFORE, usize::try_from(MAPPED * PAGE_SIZE).unwrap());
        let changed: Vec<u64> = (0u64..)
            .zip(&all)
            .filter(|&(_, &byte)| byte != FILL)
            .map(|(index, _)| BEFORE + index)
            .collect();
        let mut expected: Vec<u64> = copied
            .into_iter()
            .chain([address])
            .flat_map(|at| at..at + 4)
            .collect();
        expected.sort_unstable();
        assert_eq!(changed, expected, "{address:#x}");
        if let Some(at) = copied {
            assert_eq!(m.word(at), 0x5a5a_0001, "{address:#x}");
        }
    }
}

/// `Nx2SemaphoreMirror` looks at the release's address only: a 16-byte release that starts in the
/// first page and runs into the second is not copied; one that starts 8 bytes before the
/// registration's end is copied whole, the copy's second half landing at the start of the
/// second page.
#[test]
fn a_release_across_a_page_edge_is_decided_by_its_first_byte() {
    let mut r = rig();
    let m = mirrored(&r);
    r.fences
        .attach(m.action(sixteen(SECOND - 8, 0x11)))
        .unwrap();
    assert_eq!(m.bytes(SECOND - 8, 16), sixteen_bytes(0x11));
    assert!(m.untouched(S - 8, 16));

    let mut r = rig();
    let m = mirrored(&r);
    r.fences
        .attach(m.action(sixteen(NEIGHBOUR - 8, 0x22)))
        .unwrap();
    assert_eq!(m.bytes(NEIGHBOUR - 8, 16), sixteen_bytes(0x22));
    assert_eq!(m.bytes(SECOND - 8, 16), sixteen_bytes(0x22));
    assert!(m.untouched(SECOND + 8, 8));
}

/// Every release form, each at its own offset: the copy is the release's bytes, as long as its
/// form, and nothing past them.
#[test]
fn every_release_form_is_copied() {
    let four_ts = Release {
        form: ReleaseForm {
            size: PayloadSize::Four,
            timestamp: true,
        },
        ..four(0, 0)
    };
    let mut r = rig();
    let m = mirrored(&r);
    let forms = [
        (four(SECOND + 0x100, 0x0102_0304_0506_0708), 4),
        (eight(SECOND + 0x200, 0x1112_1314_1516_1718), 8),
        (
            Release {
                address: SECOND + 0x300,
                payload: 0x2122_2324_2526_2728,
                ..four_ts
            },
            16,
        ),
        (sixteen(SECOND + 0x400, 0x3132_3334_3536_3738), 16),
    ];
    for (release, len) in forms {
        r.fences.attach(m.action(release)).unwrap();
        let written = m.bytes(release.address, len);
        assert_ne!(written, vec![FILL; len]);
        let copy = release.address - SEMAPHORE_MIRROR_OFFSET;
        assert_eq!(m.bytes(copy, len), written, "{release:?}");
        assert!(
            m.untouched(copy + u64::try_from(len).unwrap(), 4),
            "{release:?}"
        );
    }
}

/// The copy is the reduction's result, read from the second page; reducing again at the copy
/// would give 103.
#[test]
fn a_reductions_result_is_copied() {
    let mut r = rig();
    let m = mirrored(&r);
    m.mem.write(S + 0x20, &100u32.to_le_bytes()).unwrap();
    m.mem.write(SECOND + 0x20, &5u32.to_le_bytes()).unwrap();
    r.fences
        .attach(m.action(reduce(SECOND + 0x20, 3, ReductionOp::Add)))
        .unwrap();
    assert_eq!(m.word(SECOND + 0x20), 8);
    assert_eq!(m.word(S + 0x20), 8);
}

/// A clock whose every timestamp reading differs and is counted.
#[derive(Default)]
struct Ticking(AtomicU64);

impl FenceClock for Ticking {
    fn elapsed(&self) -> Duration {
        Duration::ZERO
    }
    fn gpu_timestamp(&self) -> u64 {
        self.0.fetch_add(1, Ordering::Relaxed) + 0x1000
    }
}

/// The copy carries the release's timestamp, and the clock is read once; NX2's second query
/// would read it again.
#[test]
fn a_wide_copy_has_the_releases_timestamp() {
    let r = rig();
    let clock = Arc::new(Ticking::default());
    let env = FenceEnv {
        syncpoints: Arc::clone(&r.syncpoints),
        waiters: Arc::clone(&r.waiters),
        completions: Arc::<Pushed>::clone(&r.pushed),
        clock: Arc::<Ticking>::clone(&clock),
        front: Arc::<Front>::clone(&r.front),
    };
    let mut fences = Fences::new(Arc::<LateBackend>::clone(&r.backend), env, Mode::Fenced);
    let m = mirrored(&r);
    fences.attach(m.action(sixteen(SECOND + 0x40, 7))).unwrap();
    let expected = [7u64.to_le_bytes(), 0x1000u64.to_le_bytes()].concat();
    assert_eq!(m.bytes(SECOND + 0x40, 16), expected);
    assert_eq!(m.bytes(S + 0x40, 16), expected);
    assert_eq!(clock.0.load(Ordering::Relaxed), 1);
}

/// The second page is not mapped: the release faults, is reported once, and nothing is copied.
#[test]
fn a_release_that_faults_is_not_copied() {
    let mut r = rig();
    let m = mirrored_over(&r, &[BEFORE, S, NEIGHBOUR]);
    let address = SECOND + 0x10;
    assert_eq!(
        r.fences.attach(m.action(four(address, 9))),
        Ok(Attached::Performed(Report {
            failed: vec![ReleaseFailure {
                address,
                fault: GpuFault::Unmapped { va: address },
            }],
            ..Report::default()
        }))
    );
    assert!(m.untouched(S + 0x10, 4));
    assert_eq!(r.fences.counters().failed_releases, 1);
}

/// The first page is not mapped: the release is written and counted, and the copy is a failed
/// release at the copy's address, logged at the action's level (warning, then debug).
#[test]
fn a_copy_that_faults_is_reported_at_the_copys_address() {
    for (failure, level) in [
        (FailureLog::Warn, tracing::Level::WARN),
        (FailureLog::Debug, tracing::Level::DEBUG),
    ] {
        let mut r = rig();
        let m = mirrored_over(&r, &[BEFORE, SECOND, NEIGHBOUR]);
        let action = FencedAction::Release {
            memory: Arc::<VmMemory<FlatMemory>>::clone(&m.mem),
            space: SPACE,
            release: four(SECOND + 0x10, 9),
            failure,
        };
        let (attached, events) = captured(|| r.fences.attach(action));
        let copy = S + 0x10;
        assert_eq!(
            attached,
            Ok(Attached::Performed(Report {
                released: 1,
                failed: vec![ReleaseFailure {
                    address: copy,
                    fault: GpuFault::Unmapped { va: copy },
                }],
                ..Report::default()
            }))
        );
        assert_eq!(m.word(SECOND + 0x10), 9);
        assert_eq!(r.fences.counters().failed_releases, 1);
        let logged = at(&events, level);
        assert_eq!(logged.len(), 1, "{events:?}");
        assert!(logged[0].contains(&format!("{copy:#x}")), "{logged:?}");
    }
}

/// A waiter on a word of the copy is latched by it; a waiter on the word after the copy is not.
#[test]
fn a_waiter_on_the_copy_is_latched() {
    let mut r = rig();
    let m = mirrored(&r);
    for (channel, address) in [(ChannelId(3), S + 0x10), (ChannelId(4), S + 0x14)] {
        r.waiters.register(Waiter {
            space: SPACE,
            channel,
            address,
            op: AcquireOp::Equal,
            payload: 7,
        });
    }
    assert_eq!(
        r.fences.attach(m.action(four(SECOND + 0x10, 7))),
        Ok(performed(true))
    );
    assert!(r.waiters.is_latched(ChannelId(3)));
    assert!(!r.waiters.is_latched(ChannelId(4)));
}

/// A waiter on the copy's word closes the open list when the release is attached; the same
/// release to the neighbour, which is not copied, leaves it open.
#[test]
fn a_waiter_on_the_copy_closes_the_open_list() {
    for (address, closed) in [(SECOND + 0x10, true), (NEIGHBOUR + 0x10, false)] {
        let mut r = rig();
        let m = mirrored(&r);
        r.waiters.register(Waiter {
            space: SPACE,
            channel: ChannelId(3),
            address: S + 0x10,
            op: AcquireOp::Equal,
            payload: 7,
        });
        r.fences.record(command(1)).unwrap();
        let attached = r.fences.attach(m.action(four(address, 7))).unwrap();
        assert_eq!(
            matches!(attached, Attached::InFlight(_)),
            closed,
            "{address:#x}"
        );
        assert_eq!(r.fences.counters().closed.waiter, u64::from(closed));
    }
}

/// A pending release whose copy writes a byte of the range is settled; a range right after the
/// copy (and away from the release) has nothing pending.
#[test]
fn settle_waits_for_a_release_whose_copy_writes_the_range() {
    let mut r = rig();
    let m = mirrored(&r);
    r.fences.record(command(1)).unwrap();
    assert_eq!(
        r.fences.attach(m.action(four(SECOND + 0x10, 9))),
        Ok(Attached::Open)
    );
    assert_eq!(
        r.fences.settle(SPACE, S + 0x14..S + 0x18, PATIENCE),
        Ok(Settled::Nothing)
    );
    assert_eq!(r.fences.open_commands(), 1);
    let fence = r.fences.close(CloseReason::Present).unwrap().unwrap();
    r.backend.finish(fence);
    assert_eq!(
        r.fences.settle(SPACE, S + 0x10..S + 0x14, PATIENCE),
        Ok(Settled::Performed(released(1)))
    );
    assert_eq!(m.word(S + 0x10), 9);
}

/// The list is read when the release is performed: a registration removed after the release
/// was attached stops the copy, and one added after it starts it.
#[test]
fn the_list_is_read_when_the_release_is_performed() {
    let mut r = rig();
    let m = mirrored(&r);
    r.fences.record(command(1)).unwrap();
    r.fences.attach(m.action(four(SECOND + 0x10, 1))).unwrap();
    let fence = r.fences.close(CloseReason::Present).unwrap().unwrap();
    m.vm.remove_semaphore_registration(registration());
    r.backend.finish(fence);
    assert_eq!(r.fences.complete(), released(1));
    assert!(m.untouched(S + 0x10, 4));

    r.fences.record(command(2)).unwrap();
    r.fences.attach(m.action(four(SECOND + 0x20, 2))).unwrap();
    let fence = r.fences.close(CloseReason::Present).unwrap().unwrap();
    m.vm.add_semaphore_registration(registration());
    r.backend.finish(fence);
    assert_eq!(r.fences.complete(), released(1));
    assert_eq!(m.word(S + 0x20), 2);
}

/// One call to a memory, as [`Logged`] keeps it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Call {
    Read(u64),
    Write(u64),
    Mirror(u64),
}

/// The production memory, with each call logged in order. It answers as the memory it wraps.
struct Logged {
    inner: Arc<VmMemory<FlatMemory>>,
    calls: Mutex<Vec<Call>>,
}

impl GpuMemory for Logged {
    fn read(&self, va: u64, buf: &mut [u8]) -> Result<(), GpuFault> {
        self.calls.lock().unwrap().push(Call::Read(va));
        self.inner.read(va, buf)
    }
    fn write(&self, va: u64, data: &[u8]) -> Result<(), GpuFault> {
        self.calls.lock().unwrap().push(Call::Write(va));
        self.inner.write(va, data)
    }
    fn generation(&self) -> u64 {
        self.inner.generation()
    }
    fn semaphore_mirror(&self, va: u64) -> Option<u64> {
        self.calls.lock().unwrap().push(Call::Mirror(va));
        self.inner.semaphore_mirror(va)
    }
}

/// The release, its read-back, then the question and the copy with its read-back, as NX2's
/// puller queries the release before the mirror.
#[test]
fn the_copy_follows_the_releases_read_back() {
    let mut r = rig();
    let m = mirrored(&r);
    let logged = Arc::new(Logged {
        inner: Arc::clone(&m.mem),
        calls: Mutex::new(Vec::new()),
    });
    let release = SECOND + 0x10;
    let action = FencedAction::Release {
        memory: Arc::<Logged>::clone(&logged),
        space: SPACE,
        release: four(release, 5),
        failure: FailureLog::Warn,
    };
    r.fences.attach(action).unwrap();
    assert_eq!(
        *logged.calls.lock().unwrap(),
        [
            Call::Write(release),
            Call::Read(release),
            Call::Mirror(release),
            Call::Write(S + 0x10),
            Call::Read(S + 0x10),
        ]
    );
}
