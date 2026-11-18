//! The address space's recording point (module docs, "The Map recording point"): each mutation
//! the kernel asks the mapper for, and this address space makes, is one `EventKind::Map`, with
//! the guest's raw permission word.

use std::sync::Weak;
use std::sync::mpsc;
use std::time::Duration;

use starquake_kernel::memory::{
    AddressSpace as KernelSpace, MemoryArrangement, MemoryPermission, ProcessMemory, RegionOffsets,
};

use super::*;

/// What a test's sink was sent, in order.
#[derive(Default)]
struct Seen(Mutex<Vec<EventKind>>);

impl Seen {
    fn take(&self) -> Vec<EventKind> {
        std::mem::take(&mut *self.0.lock().unwrap())
    }
}

/// A sink that keeps what it is sent in `seen`.
fn keeping(seen: &Arc<Seen>) -> MapSink {
    let seen = Arc::clone(seen);
    MapSink::new(move |kind| seen.0.lock().unwrap().push(kind))
}

fn event(va: u64, size: u64, perms: u32, kind: MapKind) -> EventKind {
    EventKind::Map {
        va,
        size,
        perms,
        kind,
    }
}

/// The guest's raw permission word of each translation, as `Svc::MemoryPermission` writes it
/// (`eden/src/core/hle/kernel/svc_types.h` @ cf654052: `Read = (1 << 0)`, `Write = (1 << 1)`,
/// `Execute = (1 << 2)`).
#[test]
fn the_permission_word_is_the_svc_bits() {
    for (perms, word) in [
        (Perms::None, 0),
        (Perms::Read, 1),
        (Perms::ReadWrite, 3),
        (Perms::Execute, 4),
        (Perms::ReadExecute, 5),
        (Perms::ReadWriteExecute, 7),
    ] {
        assert_eq!(permission_word(perms), word, "{perms:?}");
    }
}

#[test]
fn the_permission_word_is_the_kernels_user_bits_for_every_translation() {
    // Every subset of the user's three bits: the kernel's translation of it, if it has one,
    // gives back the same bits.
    let bits = [
        MemoryPermission::USER_READ,
        MemoryPermission::USER_WRITE,
        MemoryPermission::USER_EXECUTE,
    ];
    let mut translated = 0;
    for subset in 0..8_u8 {
        let permission = bits
            .iter()
            .enumerate()
            .filter(|(index, _)| subset & (1 << index) != 0)
            .fold(MemoryPermission::NONE, |all, (_, bit)| all | *bit);
        if let Some(perms) = permission.guest() {
            assert_eq!(permission_word(perms), permission.svc(), "{subset:#b}");
            translated += 1;
        }
    }
    // Write without read is the two subsets with no translation.
    assert_eq!(translated, 6);
}

#[test]
fn each_mutation_the_kernel_asks_for_is_one_map_event() {
    let space = Arc::new(space());
    let seen = Arc::new(Seen::default());
    space.record_maps(keeping(&seen));
    let mut mapper = SpaceMapper(Arc::clone(&space));
    mapper.map(LOW, 2 * PAGE_SIZE, Perms::ReadWrite).unwrap();
    mapper.protect(LOW, PAGE_SIZE, Perms::Read).unwrap();
    let pages = mapper.allocate_shared(2 * PAGE_SIZE).unwrap();
    mapper
        .map_shared(FAR, pages.backing(), Perms::Read)
        .unwrap();
    mapper
        .map_alias(FAR + 4 * PAGE_SIZE, LOW, PAGE_SIZE, Perms::ReadWrite)
        .unwrap();
    mapper.unmap(FAR + 4 * PAGE_SIZE, PAGE_SIZE).unwrap();
    mapper.unmap(FAR, 2 * PAGE_SIZE).unwrap();
    mapper.release_shared(pages.backing()).unwrap();
    assert_eq!(
        seen.take(),
        [
            event(LOW, 2 * PAGE_SIZE, 3, MapKind::Map),
            event(LOW, PAGE_SIZE, 1, MapKind::Protect),
            event(FAR, 2 * PAGE_SIZE, 1, MapKind::Map),
            event(FAR + 4 * PAGE_SIZE, PAGE_SIZE, 3, MapKind::Map),
            event(FAR + 4 * PAGE_SIZE, PAGE_SIZE, 0, MapKind::Unmap),
            event(FAR, 2 * PAGE_SIZE, 0, MapKind::Unmap),
        ]
    );
    // Handing out and giving back pages maps nothing: neither is recorded.
}

/// What the mapper answers a call.
type Answer = Result<(), MapperError>;

#[test]
fn a_refused_mutation_records_nothing() {
    let space = Arc::new(space());
    let seen = Arc::new(Seen::default());
    space.record_maps(keeping(&seen));
    let mut mapper = SpaceMapper(Arc::clone(&space));
    mapper.map(LOW, PAGE_SIZE, Perms::ReadWrite).unwrap();
    assert_eq!(seen.take().len(), 1);
    let not_shared = BackingRange::new(RAM_BASE, PAGE_SIZE).unwrap();
    // Each refusal is the address space's own, for the reason the call is refused, in the
    // words the kernel hears ([`refused`]).
    let refusal = |error: SpaceError| -> Answer { Err(MapperError::Refused(error.to_string())) };
    let refusals: [(&str, Answer, Answer); 5] = [
        (
            "map",
            mapper.map(LOW, PAGE_SIZE, Perms::Read),
            refusal(MapError::AlreadyMapped { va: LOW }.into()),
        ),
        (
            "unmap",
            mapper.unmap(FAR, PAGE_SIZE),
            refusal(MapError::NotMapped { va: FAR }.into()),
        ),
        (
            "protect",
            mapper.protect(FAR, PAGE_SIZE, Perms::Read),
            refusal(MapError::NotMapped { va: FAR }.into()),
        ),
        (
            "map_shared",
            mapper.map_shared(FAR, not_shared, Perms::Read),
            refusal(SpaceError::NotShared {
                base: RAM_BASE,
                size: PAGE_SIZE,
            }),
        ),
        (
            "map_alias",
            mapper.map_alias(FAR, FAR + PAGE_SIZE, PAGE_SIZE, Perms::ReadWrite),
            refusal(
                MapError::NotMapped {
                    va: FAR + PAGE_SIZE,
                }
                .into(),
            ),
        ),
    ];
    for (name, result, expected) in refusals {
        assert_eq!(result, expected, "{name}");
    }
    assert_eq!(seen.take(), [], "a refusal recorded an event");
}

#[test]
fn nothing_is_recorded_before_a_sink_is_attached_and_a_new_sink_replaces_the_old() {
    let space = Arc::new(space());
    let mut mapper = SpaceMapper(Arc::clone(&space));
    mapper.map(LOW, PAGE_SIZE, Perms::Read).unwrap();
    let (first, second) = (Arc::new(Seen::default()), Arc::new(Seen::default()));
    space.record_maps(keeping(&first));
    mapper.map(LOW + PAGE_SIZE, PAGE_SIZE, Perms::Read).unwrap();
    space.record_maps(keeping(&second));
    mapper.unmap(LOW, PAGE_SIZE).unwrap();
    assert_eq!(
        first.take(),
        [event(LOW + PAGE_SIZE, PAGE_SIZE, 1, MapKind::Map)]
    );
    assert_eq!(second.take(), [event(LOW, PAGE_SIZE, 0, MapKind::Unmap)]);
}

#[test]
fn the_sink_is_called_with_no_lock_of_the_address_space_held() {
    // The sink reads the address space, which takes its shadow's lock and then the sink's: a
    // recording point that held either while calling the sink would never return. The test
    // waits a bounded time for it on another thread.
    let space = Arc::new(space());
    let weak: Weak<AddressSpace<FlatMemory>> = Arc::downgrade(&space);
    let (sent, read) = mpsc::channel();
    space.record_maps(MapSink::new(move |kind| {
        if let Some(space) = weak.upgrade() {
            let mut byte = [0xFF];
            let answer = space.read(LOW, &mut byte).map(|()| byte[0]);
            space.record_maps(MapSink::new(|_| {}));
            sent.send((kind, answer)).unwrap();
        }
    }));
    let worker = Arc::clone(&space);
    std::thread::spawn(move || {
        SpaceMapper(worker)
            .map(LOW, PAGE_SIZE, Perms::ReadWrite)
            .unwrap();
    });
    assert_eq!(
        read.recv_timeout(Duration::from_secs(10)),
        Ok((event(LOW, PAGE_SIZE, 3, MapKind::Map), Ok(0)))
    );
}

#[test]
fn the_kernels_heap_growth_and_shrink_are_recorded_through_the_mapper() {
    let space = Arc::new(space());
    let seen = Arc::new(Seen::default());
    // The kernel fixture's program: seven pages of code at `LOW`, the fixture's code address
    // (`crates/kernel/src/memory/fixture.rs`).
    let memory = ProcessMemory {
        code_address: LOW,
        code_size: 7 * PAGE_SIZE,
        arrangement: MemoryArrangement::FourGb,
        alias_region_extra: false,
        offsets: RegionOffsets::default(),
    };
    let mut kernel = KernelSpace::new(&memory, Box::new(SpaceMapper(Arc::clone(&space)))).unwrap();
    // The stack sets the largest heap (`KProcess::Run`).
    kernel.create_main_thread_stack(PAGE_SIZE).unwrap();
    space.record_maps(keeping(&seen));
    let heap = kernel.set_heap_size(2 * PAGE_SIZE).unwrap();
    kernel.set_heap_size(PAGE_SIZE).unwrap();
    assert_eq!(
        seen.take(),
        [
            event(heap, 2 * PAGE_SIZE, 3, MapKind::Map),
            event(heap + PAGE_SIZE, PAGE_SIZE, 0, MapKind::Unmap),
        ]
    );
}
