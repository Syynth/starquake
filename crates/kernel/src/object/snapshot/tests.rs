//! The kernel object snapshot: a live kernel with every object kind saved, restored, used and
//! saved again to the same bytes; the layout pinned by golden bytes; one fault case per check
//! with its exact error, the accepted case beside each edge; pairs of faults for the documented
//! order; and the errors in plain words.

use proptest::prelude::*;
use proptest::test_runner::{Config, RngAlgorithm, RngSeed};
use starquake_memory::{FlatMemory, PAGE_SIZE};

use super::*;
use crate::dispatch::Tables;
use crate::handle::{Closed, ObjectRefs, RefCounts};
use crate::memory::fixture::{self, SHARED};
use crate::memory::{GUARD_PAGES, LendFault, MemoryAttribute};
use crate::process::{self, Console, MainThread, ProcessDescription};
use crate::resource_limit::{LimitableResource, THREAD_COUNT_LIMIT};
use crate::snapshot::FormatError;
use crate::svc::resource_limit::{
    LimitTables, get_resource_limit_current_value, get_resource_limit_limit_value,
};
use crate::svc::shared_memory::free_released;
use crate::svc::transfer_memory::{create_transfer_memory, unlock_released};

/// The test servers' service numbers: distinct, so a lookup that ignores the number fails.
const SERVICE: ServiceId = ServiceId(0x31);
const OTHER_SERVICE: ServiceId = ServiceId(0x47);

/// Where the main thread starts.
const ENTRY: u64 = 0x8000_0040;

/// Fixed seed so a run is reproducible; failures are not persisted to disk.
const PROPTEST_SEED: u64 = 0x5351_4F42_534E_4150;

/// A live kernel with every object kind, and what a test needs of it.
struct Kernel {
    tables: Tables,
    main: MainThread,
    hooked: Vec<HookedEvent>,
    /// The transfer memory over heap pages 1-2, which its owner keeps read access to.
    transfer: (Handle, ObjectId),
    /// The session's client end.
    session: (Handle, ObjectId),
    /// An event the guest made: its writable and readable ends, each with a handle.
    event: (ObjectId, ObjectId),
    /// An event whose readable end is gone: its writable end.
    lone: ObjectId,
    /// The two hooked events' writable and readable ends (the first server's, then the
    /// second's).
    hooks: [(ObjectId, ObjectId); 2],
    /// The two shared memories: the first mapped at [`SHARED`].
    shared: [ObjectId; 2],
    /// The heap's start.
    heap: u64,
    /// The guest's handle to its process's resource limit, and the limit's object.
    limit: (Handle, ObjectId),
}

impl Kernel {
    fn own(&self) -> ProcessId {
        self.tables.processes.own()
    }

    fn objects(&self) -> &Objects {
        self.tables.scheduler.objects()
    }

    /// The object table and handle table saved as [`Objects::save`] saves them.
    fn save(&self) -> Result<Vec<u8>, ObjectSnapshotError> {
        let own = self.own();
        self.objects().save(
            &[(own, &self.tables.handles)],
            &self.hooked,
            &self.tables.scheduler,
            &[(own, &self.tables.address_space)],
        )
    }

    /// `bytes` restored against this kernel's scheduler and address space.
    fn restore(&self, bytes: &[u8]) -> Result<RestoredObjects, ObjectSnapshotError> {
        Objects::restore(bytes, &self.tables.scheduler)?
            .finish(&[(self.own(), &self.tables.address_space)])
    }
}

/// A handle to `object` in `tables`' handle table, with its own reference.
fn add_handle(tables: &mut Tables, object: ObjectId) -> Handle {
    tables
        .handles
        .add(tables.scheduler.objects_mut(), object)
        .unwrap()
}

/// Two servers in `tables`, each with a hooked event whose writable end the guest has a
/// handle to, and the first with a session the guest has a handle to: the hooked events'
/// ends, the session and the guest's handle to it.
fn servers(tables: &mut Tables) -> ([(ObjectId, ObjectId); 2], ObjectId, Handle) {
    let mut hooks = Vec::new();
    let mut session = None;
    for (name, service) in [("first", SERVICE), ("second", OTHER_SERVICE)] {
        let (objects, ipc) = tables.scheduler.ipc_mut();
        let server = ipc.add_server(name, service).unwrap();
        if session.is_none() {
            session = Some(ipc.open_session(objects, server, true).unwrap());
        }
        let (writable, readable) = tables.scheduler.create_event().unwrap();
        let (_, ipc) = tables.scheduler.ipc_mut();
        ipc.add_server_event(server, writable).unwrap();
        ipc.add_server_event(server, readable).unwrap();
        add_handle(tables, writable);
        hooks.push((writable, readable));
    }
    let session = session.unwrap();
    let handle = add_handle(tables, session);
    tables.scheduler.close_object(session).unwrap();
    (hooks.try_into().unwrap(), session, handle)
}

/// The fixture's process, started, with: a second handle to its main thread and one to its
/// process; a transfer memory over heap pages 1-2; an event the guest made, signalled, with a
/// handle to each end; an event whose readable end is gone; two servers, each with a hooked
/// event whose writable end the guest has a handle to, and the first with a session the guest
/// has a handle to; and two shared memories, the first mapped.
fn kernel() -> Kernel {
    let (space, _log, stack_top) = fixture::loaded();
    let mut tables =
        process::create(&ProcessDescription::homebrew(Console::Switch2), space).unwrap();
    let tls = tables.address_space.regions().kernel_map.start + GUARD_PAGES * PAGE_SIZE;
    let memory = FlatMemory::new(tls, usize::try_from(PAGE_SIZE).unwrap());
    let main = process::run(&mut tables, &memory, ENTRY, stack_top).unwrap();
    let own = tables.processes.own();
    let thread_object = tables.handles.get(main.handle).unwrap();
    add_handle(&mut tables, thread_object);
    let process_object = tables.scheduler.process(own).unwrap().object();
    add_handle(&mut tables, process_object);
    // The process's resource limit, as `GetInfo`'s `ResourceLimit` gives it out.
    let limit_object = tables.scheduler.process(own).unwrap().resource_limit();
    let limit = (add_handle(&mut tables, limit_object), limit_object);
    // The transfer memory.
    let heap = tables.address_space.set_heap_size(4 * PAGE_SIZE).unwrap();
    let Tables {
        handles,
        address_space,
        scheduler,
        ..
    } = &mut tables;
    let read = u32::from(SvcPermission::Read.bits());
    let lent = create_transfer_memory(
        handles,
        address_space,
        scheduler,
        own,
        heap + PAGE_SIZE,
        2 * PAGE_SIZE,
        read,
    )
    .unwrap();
    let transfer = (lent, handles.get(lent).unwrap());
    // The guest's event, signalled.
    let (writable, readable) = tables.scheduler.create_event().unwrap();
    for end in [writable, readable] {
        add_handle(&mut tables, end);
        tables.scheduler.close_object(end).unwrap();
    }
    tables.scheduler.signal_object(writable).unwrap();
    // An event whose readable end went with its creator's reference.
    let (lone, gone) = tables.scheduler.create_event().unwrap();
    add_handle(&mut tables, lone);
    tables.scheduler.close_object(lone).unwrap();
    tables.scheduler.close_object(gone).unwrap();
    let (hooks, session, session_handle) = servers(&mut tables);
    // The shared memories, the first mapped.
    let mut shared = [lone; 2];
    for (slot, pages) in shared.iter_mut().zip([2, 1]) {
        let Tables {
            address_space,
            scheduler,
            ..
        } = &mut tables;
        let (object, _) = fixture::shared_memory(
            address_space,
            scheduler.objects_mut(),
            pages,
            SvcPermission::ReadWrite,
        );
        *slot = object;
    }
    let first = match tables.scheduler.objects().get(shared[0]) {
        Ok(Object::SharedMemory(memory)) => memory,
        other => unreachable!("{other:?}"),
    };
    tables
        .address_space
        .map_shared(SHARED, shared[0], first, SvcPermission::ReadWrite)
        .unwrap();
    let hooked = hooks
        .iter()
        .zip([SERVICE, OTHER_SERVICE])
        .map(|(&(event, _), service)| HookedEvent { event, service })
        .collect();
    Kernel {
        tables,
        main,
        hooked,
        transfer,
        session: (session_handle, session),
        event: (writable, readable),
        lone,
        hooks,
        shared,
        heap,
        limit,
    }
}

// ----- Round trip ------------------------------------------------------------------------------

#[test]
fn a_live_kernel_saves_restores_and_saves_the_same_bytes() {
    let kernel = kernel();
    let bytes = kernel.save().unwrap();
    let restored = kernel.restore(&bytes).unwrap();
    assert_eq!(&restored.objects, kernel.objects());
    assert_eq!(
        restored.handles,
        [(kernel.own(), kernel.tables.handles.clone())]
    );
    assert_eq!(restored.hooked, kernel.hooked);
    // Every kind is there, with its references.
    let kinds: BTreeSet<&str> = restored
        .objects
        .rows
        .iter()
        .map(|(_, row)| match row.object {
            Object::Thread(_) => "thread",
            Object::Process(_) => "process",
            Object::Session => "session",
            Object::Event(_) => "event",
            Object::ReadableEvent(_) => "readable event",
            Object::SharedMemory(_) => "shared memory",
            Object::TransferMemory(_) => "transfer memory",
            Object::ResourceLimit(_) => "resource limit",
        })
        .collect();
    assert_eq!(kinds.len(), 8, "{kinds:?}");
    // Saved again, the bytes are the same.
    let own = kernel.own();
    let again = restored
        .objects
        .save(
            &[(own, &restored.handles[0].1)],
            &restored.hooked,
            &kernel.tables.scheduler,
            &[(own, &kernel.tables.address_space)],
        )
        .unwrap();
    assert_eq!(again, bytes);
}

#[test]
fn the_restored_tables_work_as_the_saved_ones_did() {
    let mut kernel = kernel();
    let bytes = kernel.save().unwrap();
    let restored = kernel.restore(&bytes).unwrap();
    // What the saved tables do next, on copies.
    let mut objects = kernel.objects().clone();
    let mut handles = kernel.tables.handles.clone();
    let made = objects.create(Object::Session).unwrap();
    let handle = handles.add(&mut objects, made).unwrap();
    // The restored tables take their place and do the same.
    *kernel.tables.scheduler.objects_mut() = restored.objects;
    kernel.tables.handles = restored.handles.into_iter().next().unwrap().1;
    let scheduler = kernel.tables.scheduler.objects_mut();
    assert_eq!(scheduler.create(Object::Session), Ok(made));
    assert_eq!(kernel.tables.handles.add(scheduler, made), Ok(handle));
    // The guest's event is still signalled, and clears through its writable end.
    let (writable, readable) = kernel.event;
    let signalled = |kernel: &Kernel| match kernel.objects().get(readable) {
        Ok(Object::ReadableEvent(state)) => state.is_signalled(),
        other => unreachable!("{other:?}"),
    };
    assert!(signalled(&kernel));
    kernel.tables.scheduler.clear_object(writable).unwrap();
    assert!(!signalled(&kernel));
    // Closing the transfer memory's handle releases it, and its pages are unlocked.
    let (lent, object) = kernel.transfer;
    let removed = kernel
        .tables
        .handles
        .remove(kernel.tables.scheduler.objects_mut(), lent)
        .unwrap();
    assert_eq!((removed.object, removed.closed), (object, Closed::Last));
    let own = kernel.own();
    unlock_released(
        &mut kernel.tables.scheduler,
        &mut kernel.tables.address_space,
        own,
    )
    .unwrap();
    let block = kernel
        .tables
        .address_space
        .query_info(kernel.heap + PAGE_SIZE)
        .unwrap();
    assert!(!block.attribute().contains(MemoryAttribute::LOCKED));
    // The session closes through its handle, and its server is told.
    let (session, session_object) = kernel.session;
    let removed = kernel
        .tables
        .handles
        .remove(kernel.tables.scheduler.objects_mut(), session)
        .unwrap();
    assert_eq!(removed.closed, Closed::Last);
    kernel
        .tables
        .scheduler
        .tear_down(session_object, Object::Session)
        .unwrap();
    assert!(!kernel.tables.scheduler.ipc().is_session(session_object));
    // The resource limit answers through the guest's handle, from the restored table.
    let (limit, limit_object) = kernel.limit;
    assert_eq!(kernel.tables.handles.get(limit), Ok(limit_object));
    let tables = LimitTables {
        handles: &kernel.tables.handles,
        space: &kernel.tables.address_space,
        scheduler: &kernel.tables.scheduler,
        processes: &kernel.tables.processes,
    };
    let threads = LimitableResource::ThreadCount.number();
    assert_eq!(
        get_resource_limit_limit_value(tables, limit, threads),
        Ok(THREAD_COUNT_LIMIT)
    );
    assert_eq!(
        get_resource_limit_current_value(tables, limit, threads),
        Ok(1)
    );
}

#[test]
fn a_space_held_by_a_lend_and_handles_of_every_kind_round_trip_with_a_second_process() {
    // A second process, its own handle table and address space with a lend of its own.
    let mut kernel = kernel();
    let second = second_process(&mut kernel);
    let bytes = second.save(&kernel).unwrap();
    let restored = second.restore(&kernel, &bytes).unwrap();
    assert_eq!(restored.handles.len(), 2);
    assert_eq!(
        restored.handles[1],
        (second.process, second.handles.clone())
    );
    assert_eq!(&restored.objects, kernel.objects());
}

/// Both processes' handle tables, in process order.
type Handles<'a> = [(ProcessId, &'a HandleTable); 2];
/// Both processes' address spaces, in process order.
type Spaces<'a> = [(ProcessId, &'a AddressSpace); 2];

/// A second process beside the kernel's own, with a handle table holding a handle to its own
/// transfer memory over heap pages 0-1 of its own address space.
struct Second {
    process: ProcessId,
    handles: HandleTable,
    space: AddressSpace,
    transfer: ObjectId,
}

impl Second {
    fn tables<'a>(&'a self, kernel: &'a Kernel) -> (Handles<'a>, Spaces<'a>) {
        let own = kernel.own();
        (
            [(own, &kernel.tables.handles), (self.process, &self.handles)],
            [
                (own, &kernel.tables.address_space),
                (self.process, &self.space),
            ],
        )
    }

    fn save(&self, kernel: &Kernel) -> Result<Vec<u8>, ObjectSnapshotError> {
        let (handles, spaces) = self.tables(kernel);
        kernel
            .objects()
            .save(&handles, &kernel.hooked, &kernel.tables.scheduler, &spaces)
    }

    fn restore(
        &self,
        kernel: &Kernel,
        bytes: &[u8],
    ) -> Result<RestoredObjects, ObjectSnapshotError> {
        let (_, spaces) = self.tables(kernel);
        Objects::restore(bytes, &kernel.tables.scheduler)?.finish(&spaces)
    }
}

fn second_process(kernel: &mut Kernel) -> Second {
    let own = kernel.own();
    let process = ProcessId::from_parts(own.index() + 1, own.generation());
    kernel.tables.scheduler.add_process(process).unwrap();
    let (mut space, _, _) = fixture::loaded();
    let heap = space.set_heap_size(2 * PAGE_SIZE).unwrap();
    space
        .lock_for_transfer_memory(heap, 2 * PAGE_SIZE, SvcPermission::None)
        .unwrap();
    let lent = TransferMemory::new(process, heap, 2 * PAGE_SIZE, SvcPermission::None);
    let objects = kernel.tables.scheduler.objects_mut();
    let transfer = objects
        .create(Object::TransferMemory(lent.with_pages_locked()))
        .unwrap();
    let mut handles = HandleTable::new(8).unwrap();
    handles.add(objects, transfer).unwrap();
    objects.close(transfer).unwrap();
    Second {
        process,
        handles,
        space,
        transfer,
    }
}

// ----- Layout ----------------------------------------------------------------------------------

/// A body with every object kind, each field a distinct non-zero value: slot 0 a thread, 1 a
/// process, 2 a session, 3 vacant (freed once, so generation 2), 4 an event naming slot 5, 5 a
/// signalled readable end, 6 a shared memory, 7 a transfer memory, 8 a resource limit; one
/// handle table; one hooked event.
fn golden_body() -> Body {
    let mut objects = Slab::with_max_slots(10);
    let rows = [
        SavedObject::Thread(ThreadId::from_parts(3, 4)),
        SavedObject::Process(ProcessId::from_parts(5, 6)),
        SavedObject::Session,
        SavedObject::Session,
        SavedObject::Event {
            readable: ObjectId::from_parts(5, 1),
        },
        SavedObject::ReadableEvent { signalled: true },
        SavedObject::SharedMemory {
            base: 0x7000,
            size: 0x2000,
            user_permission: SvcPermission::Read.bits(),
        },
        SavedObject::TransferMemory {
            owner: ProcessId::from_parts(5, 6),
            address: 0x9000,
            size: 0x3000,
            owner_permission: SvcPermission::ReadWrite.bits(),
            locked: true,
        },
        SavedObject::ResourceLimit {
            owner: ProcessId::from_parts(8, 9),
        },
    ];
    let mut ids = Vec::new();
    for (object, refs) in rows.into_iter().zip(10_u32..) {
        ids.push(objects.insert(SavedRow { object, refs }).unwrap());
    }
    objects.remove(ids[3]).unwrap();
    let mut refs = RefCounts::new();
    let mut table = HandleTable::new(3).unwrap();
    let first = table.add(&mut refs, ids[0]).unwrap();
    table.add(&mut refs, ids[7]).unwrap();
    table.remove(&mut refs, first).unwrap();
    Body {
        objects,
        handles: vec![SavedHandles {
            process: ProcessId::from_parts(5, 6),
            table: table.data(),
        }],
        hooked: vec![HookedEvent {
            event: ids[4],
            service: ServiceId(0x17),
        }],
    }
}

/// [`golden_body`]'s bytes, written out by hand from the module docs ("Format").
#[rustfmt::skip]
const GOLDEN: &[u8] = &[
    // Header: the magic, then the version.
    b'S', b'Q', b'O', b'B', 2,
    // The object table: nine slots.
    9,
    // Slot 0: occupied (tag 1), generation 1, a thread (tag 0) 3/4, 10 references.
    1, 1, 0, 3, 4, 10,
    // Slot 1: occupied, generation 1, a process (tag 1) 5/6, 11 references.
    1, 1, 1, 5, 6, 11,
    // Slot 2: occupied, generation 1, a session (tag 2), 12 references.
    1, 1, 2, 12,
    // Slot 3: vacant (tag 0), generation 2.
    0, 2,
    // Slot 4: occupied, generation 1, an event (tag 3) naming 5/1, 14 references.
    1, 1, 3, 5, 1, 14,
    // Slot 5: occupied, generation 1, a readable end (tag 4), signalled, 15 references.
    1, 1, 4, 1, 15,
    // Slot 6: occupied, generation 1, a shared memory (tag 5): base 0x7000, size 0x2000,
    // read (1), 16 references.
    1, 1, 5, 0x80, 0xe0, 0x01, 0x80, 0x40, 1, 16,
    // Slot 7: occupied, generation 1, a transfer memory (tag 6): owner 5/6, address 0x9000,
    // size 0x3000, read-write (3), locked, 17 references.
    1, 1, 6, 5, 6, 0x80, 0xa0, 0x02, 0x80, 0x60, 3, 1, 17,
    // Slot 8: occupied, generation 1, a resource limit (tag 7) of process 8/9, 18 references.
    1, 1, 7, 8, 9, 18,
    // The free stack: one entry, slot 3.
    1, 3,
    // The slot limit (10) and the live count (8).
    10, 8,
    // One handle table: process 5/6.
    1, 5, 6,
    // Its three entries: index 0 free, index 1 the transfer memory under linear id 2, index 2
    // free (some 1 / none 0; linear id; object 7/1).
    3, 0, 1, 2, 7, 1, 0,
    // Its free stack (0, never used, then 2, freed last: the first add took the highest index),
    // its next linear id (3) and high-water mark (2).
    2, 0, 2, 3, 2,
    // One hooked event: the event 4/1, service 0x17.
    1, 4, 1, 0x17,
];

#[test]
fn the_body_layout_is_pinned() {
    let body = golden_body();
    assert_eq!(encode(FORMAT, &body).unwrap(), GOLDEN);
    // The hand-written bytes read back as the body.
    assert_eq!(decode::<Body>(FORMAT, GOLDEN), Ok(body));
}

#[test]
fn the_magic_and_version_are_pinned_as_literals() {
    assert_eq!(OBJECTS_MAGIC, [0x53, 0x51, 0x4f, 0x42]);
    assert_eq!(OBJECTS_VERSION, 2);
}

fn config(cases: u32) -> ProptestConfig {
    ProptestConfig {
        cases,
        rng_algorithm: RngAlgorithm::ChaCha,
        rng_seed: RngSeed::Fixed(PROPTEST_SEED),
        failure_persistence: None,
        ..Config::default()
    }
}

fn any_saved_object() -> impl Strategy<Value = SavedObject> {
    let id = || (any::<u32>(), any::<u32>());
    prop_oneof![
        id().prop_map(|(i, g)| SavedObject::Thread(ThreadId::from_parts(i, g))),
        id().prop_map(|(i, g)| SavedObject::Process(ProcessId::from_parts(i, g))),
        Just(SavedObject::Session),
        id().prop_map(|(i, g)| SavedObject::Event {
            readable: ObjectId::from_parts(i, g)
        }),
        any::<bool>().prop_map(|signalled| SavedObject::ReadableEvent { signalled }),
        (any::<u64>(), any::<u64>(), any::<u8>()).prop_map(|(base, size, user_permission)| {
            SavedObject::SharedMemory {
                base,
                size,
                user_permission,
            }
        }),
        (id(), any::<u64>(), any::<u64>(), any::<u8>(), any::<bool>()).prop_map(
            |((i, g), address, size, owner_permission, locked)| SavedObject::TransferMemory {
                owner: ProcessId::from_parts(i, g),
                address,
                size,
                owner_permission,
                locked,
            }
        ),
        id().prop_map(|(i, g)| SavedObject::ResourceLimit {
            owner: ProcessId::from_parts(i, g)
        }),
    ]
}

proptest! {
    #![proptest_config(config(256))]

    /// Any rows and hooked events, built and read back, are what was built: every field is read
    /// where it was written.
    #[test]
    fn any_body_reads_back_as_built(
        rows in prop::collection::vec((any_saved_object(), any::<u32>()), 0..8),
        hooked in prop::collection::vec((any::<u32>(), any::<u32>(), any::<u32>()), 0..4),
    ) {
        let mut objects = Slab::new();
        for (object, refs) in rows {
            objects.insert(SavedRow { object, refs }).unwrap();
        }
        let body = Body {
            objects,
            handles: Vec::new(),
            hooked: hooked
                .into_iter()
                .map(|(i, g, service)| HookedEvent {
                    event: ObjectId::from_parts(i, g),
                    service: ServiceId(service),
                })
                .collect(),
        };
        let bytes = encode(FORMAT, &body).unwrap();
        prop_assert_eq!(decode::<Body>(FORMAT, &bytes), Ok(body));
    }
}

// ----- One fault per check -----------------------------------------------------------------

/// A saved kernel's body to damage, restored against the same kernel.
struct Case<'k> {
    kernel: &'k Kernel,
    body: Body,
}

impl<'k> Case<'k> {
    fn new(kernel: &'k Kernel) -> Self {
        let body = decode(FORMAT, &kernel.save().unwrap()).unwrap();
        Self { kernel, body }
    }

    fn restore(&self) -> Result<RestoredObjects, ObjectSnapshotError> {
        self.kernel.restore(&encode(FORMAT, &self.body).unwrap())
    }

    fn error(&self) -> ObjectSnapshotError {
        self.restore().unwrap_err()
    }

    /// The saved row of `id`.
    fn row(&mut self, id: ObjectId) -> &mut SavedRow {
        self.body.objects.get_mut(id).unwrap()
    }
}

fn object_error(object: ObjectId, fault: ObjectFault) -> ObjectSnapshotError {
    ObjectSnapshotError::Object { object, fault }
}

#[test]
fn the_header_is_checked_first_and_the_version_before_the_body() {
    let kernel = kernel();
    let good = kernel.save().unwrap();
    let mut magic = good.clone();
    magic[0] = b'X';
    let name = FORMAT.name;
    assert_eq!(
        kernel.restore(&magic).unwrap_err(),
        ObjectSnapshotError::Format(FormatError::BadMagic { name })
    );
    // A wrong version and a body cut short: the version is reported.
    let mut version = good[..good.len() - 1].to_vec();
    version[4] = u8::try_from(OBJECTS_VERSION + 1).unwrap();
    assert_eq!(
        kernel.restore(&version).unwrap_err(),
        ObjectSnapshotError::Format(FormatError::UnsupportedVersion {
            name,
            found: OBJECTS_VERSION + 1,
            supported: OBJECTS_VERSION
        })
    );
    // Both wrong: the magic.
    version[0] = b'X';
    assert_eq!(
        kernel.restore(&version).unwrap_err(),
        ObjectSnapshotError::Format(FormatError::BadMagic { name })
    );
    // A trailing byte.
    let mut trailing = good;
    trailing.push(0);
    assert_eq!(
        kernel.restore(&trailing).unwrap_err(),
        ObjectSnapshotError::Format(FormatError::TrailingBytes { name, count: 1 })
    );
}

#[test]
fn a_damaged_slot_layout_does_not_decode() {
    // The free stack of the golden body names an occupied slot (slot 4 for slot 3).
    let mut bytes = GOLDEN.to_vec();
    let free = bytes.len() - 23;
    assert_eq!(bytes[free..free + 2], [1, 3]);
    bytes[free + 1] = 4;
    assert!(matches!(
        decode::<Body>(FORMAT, &bytes),
        Err(FormatError::Body {
            name: "kernel object",
            ..
        })
    ));
}

#[test]
fn a_save_with_transfer_memories_still_to_unlock_is_refused() {
    let mut kernel = kernel();
    let (lent, _) = kernel.transfer;
    kernel
        .tables
        .handles
        .remove(kernel.tables.scheduler.objects_mut(), lent)
        .unwrap();
    assert_eq!(
        kernel.save(),
        Err(ObjectSnapshotError::PendingUnlocks { count: 1 })
    );
    // Once the dispatcher has unlocked them, the save goes ahead.
    let own = kernel.own();
    unlock_released(
        &mut kernel.tables.scheduler,
        &mut kernel.tables.address_space,
        own,
    )
    .unwrap();
    assert!(kernel.save().is_ok());
}

#[test]
fn a_save_with_shared_memories_still_to_give_back_is_refused_after_the_unlocks() {
    let mut kernel = kernel();
    // The second shared memory is unmapped and only its creator holds it: its last reference.
    let unmapped = kernel.shared[1];
    kernel.tables.scheduler.close_object(unmapped).unwrap();
    assert_eq!(
        kernel.save(),
        Err(ObjectSnapshotError::PendingFrees { count: 1 })
    );
    // Both kinds pending: the unlocks are reported first.
    let (lent, _) = kernel.transfer;
    kernel
        .tables
        .handles
        .remove(kernel.tables.scheduler.objects_mut(), lent)
        .unwrap();
    assert_eq!(
        kernel.save(),
        Err(ObjectSnapshotError::PendingUnlocks { count: 1 })
    );
    let own = kernel.own();
    unlock_released(
        &mut kernel.tables.scheduler,
        &mut kernel.tables.address_space,
        own,
    )
    .unwrap();
    assert_eq!(
        kernel.save(),
        Err(ObjectSnapshotError::PendingFrees { count: 1 })
    );
    // Once the dispatcher has given the pages back, the save goes ahead.
    free_released(
        &mut kernel.tables.scheduler,
        &mut kernel.tables.address_space,
    )
    .unwrap();
    assert!(kernel.save().is_ok());
}

#[test]
fn handle_tables_are_in_process_order_of_processes_the_kernel_has_with_sound_layouts() {
    let mut kernel = kernel();
    let second = second_process(&mut kernel);
    let own = kernel.own();
    let saved = |process: ProcessId, table: &HandleTable| SavedHandles {
        process,
        table: table.data(),
    };
    let bytes = second.save(&kernel).unwrap();
    let body: Body = decode(FORMAT, &bytes).unwrap();
    let restore = |body: &Body| second.restore(&kernel, &encode(FORMAT, body).unwrap());
    // Swapped, and the same process twice.
    let mut swapped = body.clone();
    swapped.handles.reverse();
    assert_eq!(
        restore(&swapped).unwrap_err(),
        ObjectSnapshotError::HandleOrder { process: own }
    );
    let mut twice = body.clone();
    twice.handles[1] = saved(own, &kernel.tables.handles);
    assert_eq!(
        restore(&twice).unwrap_err(),
        ObjectSnapshotError::HandleOrder { process: own }
    );
    // A process the kernel does not have, after the own one.
    let unknown = ProcessId::from_parts(second.process.index() + 1, 1);
    let mut stranger = body.clone();
    stranger.handles[1] = saved(unknown, &second.handles);
    assert_eq!(
        restore(&stranger).unwrap_err(),
        ObjectSnapshotError::HandleProcess { process: unknown }
    );
    // A next linear id of zero, which is never issued.
    let mut damaged = body;
    damaged.handles[1].table.next_linear_id = 0;
    assert_eq!(
        restore(&damaged).unwrap_err(),
        ObjectSnapshotError::HandleLayout {
            process: second.process,
            error: HandleLayoutError::NextLinearId(0)
        }
    );
}

#[test]
fn every_object_has_a_reference() {
    let kernel = kernel();
    let mut case = Case::new(&kernel);
    let shared = kernel.shared[1];
    case.row(shared).refs = 0;
    assert_eq!(
        case.error(),
        object_error(shared, ObjectFault::NoReferences)
    );
    case.row(shared).refs = 1;
    assert!(case.restore().is_ok());
}

#[test]
fn a_thread_or_process_object_is_the_one_its_row_names() {
    let kernel = kernel();
    let thread_object = kernel.tables.handles.get(kernel.main.handle).unwrap();
    let process_object = kernel
        .tables
        .scheduler
        .process(kernel.own())
        .unwrap()
        .object();
    // The process object made a second object of the main thread, and a thread that does not
    // exist.
    let mut case = Case::new(&kernel);
    case.row(process_object).object = SavedObject::Thread(kernel.main.thread);
    assert_eq!(
        case.error(),
        object_error(process_object, ObjectFault::Thread)
    );
    let mut case = Case::new(&kernel);
    case.row(thread_object).object = SavedObject::Thread(ThreadId::from_parts(
        kernel.main.thread.index(),
        kernel.main.thread.generation() + 1,
    ));
    assert_eq!(
        case.error(),
        object_error(thread_object, ObjectFault::Thread)
    );
    // The thread object made a second object of the process, and a process that does not
    // exist.
    let own = kernel.own();
    let mut case = Case::new(&kernel);
    case.row(thread_object).object = SavedObject::Process(own);
    assert_eq!(
        case.error(),
        object_error(thread_object, ObjectFault::Process)
    );
    let mut case = Case::new(&kernel);
    case.row(process_object).object =
        SavedObject::Process(ProcessId::from_parts(own.index(), own.generation() + 1));
    assert_eq!(
        case.error(),
        object_error(process_object, ObjectFault::Process)
    );
}

#[test]
fn a_resource_limit_is_the_one_its_process_s_row_names() {
    let mut kernel = kernel();
    let own = kernel.own();
    let (_, limit) = kernel.limit;
    let process_object = kernel.tables.scheduler.process(own).unwrap().object();
    // A limit of a process the kernel does not have, and of one whose generation differs only.
    for owner in [
        ProcessId::from_parts(own.index() + 7, own.generation()),
        ProcessId::from_parts(own.index(), own.generation() + 1),
    ] {
        let mut case = Case::new(&kernel);
        case.row(limit).object = SavedObject::ResourceLimit { owner };
        assert_eq!(
            case.error(),
            object_error(limit, ObjectFault::ResourceLimit)
        );
    }
    // A second limit of the calling process, where its process object was: the row names the
    // other one.
    let mut case = Case::new(&kernel);
    case.row(process_object).object = SavedObject::ResourceLimit { owner: own };
    assert_eq!(
        case.error(),
        object_error(process_object, ObjectFault::ResourceLimit)
    );
    // The limit made a process object.
    let mut case = Case::new(&kernel);
    case.row(limit).object = SavedObject::Process(own);
    assert_eq!(case.error(), object_error(limit, ObjectFault::Process));
    // The limit itself moved to the next generation of its slot: the row names the old one.
    let mut case = Case::new(&kernel);
    let saved = case.body.objects.remove(limit).unwrap();
    let moved = case.body.objects.insert(saved).unwrap();
    assert_eq!(
        (moved.index(), moved.generation()),
        (limit.index(), limit.generation() + 1)
    );
    assert_eq!(
        case.error(),
        object_error(moved, ObjectFault::ResourceLimit)
    );
    // A second process's limit is accepted as its own, and refused as the first's.
    let second = kernel
        .tables
        .processes
        .create(&ProcessDescription::homebrew(Console::Switch2))
        .unwrap();
    kernel.tables.scheduler.add_process(second).unwrap();
    let second_limit = kernel
        .tables
        .scheduler
        .process(second)
        .unwrap()
        .resource_limit();
    assert!(kernel.restore(&kernel.save().unwrap()).is_ok());
    let mut case = Case::new(&kernel);
    case.row(second_limit).object = SavedObject::ResourceLimit { owner: own };
    assert_eq!(
        case.error(),
        object_error(second_limit, ObjectFault::ResourceLimit)
    );
}

#[test]
fn a_session_is_one_the_ipc_table_holds() {
    let kernel = kernel();
    let mut case = Case::new(&kernel);
    let (_, readable) = kernel.event;
    case.row(readable).object = SavedObject::Session;
    assert_eq!(case.error(), object_error(readable, ObjectFault::Session));
}

#[test]
fn an_events_readable_end_is_a_readable_end_or_gone_and_its_own() {
    let kernel = kernel();
    let (writable, readable) = kernel.event;
    let process_object = kernel
        .tables
        .scheduler
        .process(kernel.own())
        .unwrap()
        .object();
    let mut case = Case::new(&kernel);
    case.row(writable).object = SavedObject::Event {
        readable: process_object,
    };
    assert_eq!(case.error(), object_error(writable, ObjectFault::Readable));
    // The lone event's readable end is gone: accepted. Naming the guest event's readable end
    // instead, it shares it with the guest's event, which comes before it.
    assert!(writable < kernel.lone);
    let mut case = Case::new(&kernel);
    case.row(kernel.lone).object = SavedObject::Event { readable };
    assert_eq!(
        case.error(),
        object_error(kernel.lone, ObjectFault::SharedReadable)
    );
    // The live readable end's slot under the next generation names an id the table has not
    // issued: the next readable end made there would be signalled through this event.
    let mut case = Case::new(&kernel);
    case.row(writable).object = SavedObject::Event {
        readable: ObjectId::from_parts(readable.index(), readable.generation() + 1),
    };
    assert_eq!(
        case.error(),
        object_error(writable, ObjectFault::ReadableNotIssued)
    );
}

#[test]
fn an_events_gone_readable_end_is_an_id_the_table_has_issued() {
    let mut kernel = kernel();
    // An event whose readable end is gone, made last, so its readable end's slot is the first
    // a new object takes and the slot after its writable end's is past the table's end.
    let (event, gone) = kernel.tables.scheduler.create_event().unwrap();
    add_handle(&mut kernel.tables, event);
    kernel.tables.scheduler.close_object(event).unwrap();
    kernel.tables.scheduler.close_object(gone).unwrap();
    // The ids the restored table hands out next, from the saved table itself: the gone readable
    // end's slot under its next generation, then a fresh slot past the end.
    let mut next = Case::new(&kernel).body.objects;
    let row = *next.get(event).unwrap();
    let vacant = next.insert(row).unwrap();
    let fresh = next.insert(row).unwrap();
    assert_eq!(
        (vacant.index(), vacant.generation()),
        (gone.index(), gone.generation() + 1)
    );
    let saved = Case::new(&kernel).body.objects;
    assert!(saved.iter().all(|(id, _)| id.index() < fresh.index()));
    assert_eq!(fresh.index(), event.index().max(gone.index()) + 1);
    // Accepted: the gone readable end itself, one generation below the vacant slot's.
    let mut case = Case::new(&kernel);
    case.row(event).object = SavedObject::Event { readable: gone };
    assert!(case.restore().is_ok());
    // Refused: the vacant slot's own generation and the one after it, and the fresh slot past
    // the end under its first generation and the one after it.
    for unissued in [
        vacant,
        ObjectId::from_parts(vacant.index(), vacant.generation() + 1),
        fresh,
        ObjectId::from_parts(fresh.index(), fresh.generation() + 1),
    ] {
        let mut case = Case::new(&kernel);
        case.row(event).object = SavedObject::Event { readable: unissued };
        assert_eq!(
            case.error(),
            object_error(event, ObjectFault::ReadableNotIssued),
            "{unissued:?}"
        );
    }
}

#[test]
fn a_shared_memory_has_pages_and_a_shared_memory_permission() {
    let kernel = kernel();
    let object = kernel.shared[1];
    let (base, size) = match kernel.objects().get(object) {
        Ok(Object::SharedMemory(memory)) => (memory.backing().base(), memory.size()),
        other => unreachable!("{other:?}"),
    };
    let saved = |base: u64, size: u64, permission: u8| SavedObject::SharedMemory {
        base,
        size,
        user_permission: permission,
    };
    let read_write = SvcPermission::ReadWrite.bits();
    for (bad, why) in [
        (saved(base + 1, size, read_write), "unaligned"),
        (saved(base, 0, read_write), "empty"),
        (saved(base, size + 1, read_write), "ragged"),
        (
            saved(u64::MAX - (PAGE_SIZE - 1), 2 * PAGE_SIZE, read_write),
            "wraps",
        ),
        (
            saved(base, size, SvcPermission::None.bits()),
            "no permission",
        ),
        (
            saved(base, size, SvcPermission::ReadExecute.bits()),
            "execute",
        ),
        (saved(base, size, 0xff), "no permission at all"),
    ] {
        let mut case = Case::new(&kernel);
        case.row(object).object = bad;
        assert_eq!(
            case.error(),
            object_error(object, ObjectFault::SharedMemory),
            "{why}"
        );
    }
    let mut case = Case::new(&kernel);
    case.row(object).object = saved(base, size, SvcPermission::Read.bits());
    assert!(case.restore().is_ok());
}

#[test]
fn shared_memories_share_no_page() {
    let kernel = kernel();
    let [first, second] = kernel.shared;
    let (base, size) = match kernel.objects().get(first) {
        Ok(Object::SharedMemory(memory)) => (memory.backing().base(), memory.size()),
        other => unreachable!("{other:?}"),
    };
    assert!(first < second);
    let saved = |base: u64| SavedObject::SharedMemory {
        base,
        size: PAGE_SIZE,
        user_permission: SvcPermission::ReadWrite.bits(),
    };
    // The second on the first's last page, and on its first: overlapping.
    for overlapping in [base + size - PAGE_SIZE, base] {
        let mut case = Case::new(&kernel);
        case.row(second).object = saved(overlapping);
        assert_eq!(
            case.error(),
            object_error(second, ObjectFault::SharedOverlap)
        );
    }
    // Starting exactly where the first ends, or ending exactly where it starts: accepted.
    for beside in [base + size, base - PAGE_SIZE] {
        let mut case = Case::new(&kernel);
        case.row(second).object = saved(beside);
        assert!(case.restore().is_ok(), "{beside:#x}");
    }
}

#[test]
fn a_transfer_memory_keeps_a_lend_permission_is_locked_and_its_owner_has_a_handle_table() {
    let kernel = kernel();
    let (_, object) = kernel.transfer;
    let own = kernel.own();
    let lent_at = kernel.heap + PAGE_SIZE;
    let saved = move |owner: ProcessId, permission: u8, locked: bool| SavedObject::TransferMemory {
        owner,
        address: lent_at,
        size: 2 * PAGE_SIZE,
        owner_permission: permission,
        locked,
    };
    let read = SvcPermission::Read.bits();
    for bad in [SvcPermission::ReadExecute.bits(), 0xff] {
        let mut case = Case::new(&kernel);
        case.row(object).object = saved(own, bad, true);
        assert_eq!(
            case.error(),
            object_error(object, ObjectFault::TransferPermission)
        );
    }
    let mut case = Case::new(&kernel);
    case.row(object).object = saved(own, read, false);
    assert_eq!(case.error(), object_error(object, ObjectFault::Unlocked));
    // A process the kernel has, but whose handle table the snapshot does not hold.
    let mut kernel = kernel;
    let other = ProcessId::from_parts(own.index() + 1, own.generation());
    kernel.tables.scheduler.add_process(other).unwrap();
    let mut case = Case::new(&kernel);
    case.row(object).object = saved(other, read, true);
    assert_eq!(case.error(), object_error(object, ObjectFault::Owner));
    // Another permission a lend leaves passes these checks and reaches check 6.
    let mut case = Case::new(&kernel);
    let keeps = SvcPermission::ReadWrite;
    case.row(object).object = saved(own, keeps.bits(), true);
    assert_eq!(
        case.error(),
        ObjectSnapshotError::Lends {
            process: own,
            error: LendError::Lend {
                address: kernel.heap + PAGE_SIZE,
                size: 2 * PAGE_SIZE,
                fault: LendFault::Permission
            }
        }
    );
}

#[test]
fn a_handle_names_a_live_object() {
    let kernel = kernel();
    let (handle, session) = kernel.session;
    let mut case = Case::new(&kernel);
    case.body.objects.remove(session).unwrap();
    assert_eq!(
        case.error(),
        ObjectSnapshotError::Handle {
            process: kernel.own(),
            handle,
            fault: HandleFault::Stale
        }
    );
}

#[test]
fn an_object_has_a_reference_for_every_handle_to_it() {
    // The main thread's object: two handles, and its own reference while it runs.
    let kernel = kernel();
    let thread = kernel.tables.handles.get(kernel.main.handle).unwrap();
    assert_eq!(kernel.objects().refs(thread), 3);
    let to_thread: Vec<Handle> = kernel
        .tables
        .handles
        .entries()
        .filter(|(_, object)| *object == thread)
        .map(|(handle, _)| handle)
        .collect();
    assert_eq!(to_thread.len(), 2);
    // As many references as handles: accepted.
    let mut case = Case::new(&kernel);
    case.row(thread).refs = 2;
    assert!(case.restore().is_ok());
    // One fewer: the second handle, lowest index first, is one too many.
    case.row(thread).refs = 1;
    assert_eq!(
        case.error(),
        ObjectSnapshotError::Handle {
            process: kernel.own(),
            handle: to_thread[1],
            fault: HandleFault::Overused
        }
    );
}

#[test]
fn a_second_limit_made_through_the_public_object_table_is_refused_by_save() {
    // `ResourceLimit::of` and `Objects::create` are public, so code outside `add_process` can
    // make a second limit for a process; its row names the first, so save refuses it.
    let mut kernel = kernel();
    let own = kernel.own();
    let stray = kernel
        .tables
        .scheduler
        .objects_mut()
        .create(Object::ResourceLimit(ResourceLimit::of(own)))
        .unwrap();
    assert_eq!(
        kernel.save().unwrap_err(),
        object_error(stray, ObjectFault::ResourceLimit)
    );
}

#[test]
fn a_resource_limit_has_a_reference_for_its_process_s_row_besides_its_handles() {
    let kernel = kernel();
    let (handle, limit) = kernel.limit;
    let to_limit: Vec<Handle> = kernel
        .tables
        .handles
        .entries()
        .filter(|(_, object)| *object == limit)
        .map(|(handle, _)| handle)
        .collect();
    assert_eq!(to_limit, [handle]);
    // The guest's handle and the process row's reference.
    assert_eq!(kernel.objects().refs(limit), 2);
    let mut case = Case::new(&kernel);
    case.row(limit).refs = 2;
    assert!(case.restore().is_ok());
    // As many references as handles: the handle takes the row's reference.
    case.row(limit).refs = 1;
    assert_eq!(
        case.error(),
        ObjectSnapshotError::Handle {
            process: kernel.own(),
            handle,
            fault: HandleFault::LimitRow
        }
    );
    assert_eq!(
        HandleFault::LimitRow.to_string(),
        "takes the reference its resource limit's process row holds"
    );
}

#[test]
fn a_hooked_event_is_a_live_writable_end_its_services_server_made() {
    let kernel = kernel();
    let [(first, first_readable), (second, _)] = kernel.hooks;
    let hook = |event, service| HookedEvent { event, service };
    let error = |hooked, fault| ObjectSnapshotError::Hook { hooked, fault };
    let with = |hooked: Vec<HookedEvent>| {
        let mut case = Case::new(&kernel);
        case.body.hooked = hooked;
        case.restore()
    };
    // Out of order, and twice.
    let swapped = vec![hook(second, OTHER_SERVICE), hook(first, SERVICE)];
    assert!(first < second);
    assert_eq!(
        with(swapped).unwrap_err(),
        error(hook(first, SERVICE), HookFault::Order)
    );
    let twice = vec![hook(first, SERVICE), hook(first, SERVICE)];
    assert_eq!(
        with(twice).unwrap_err(),
        error(hook(first, SERVICE), HookFault::Order)
    );
    // The readable end its server made, and an id that names nothing live.
    let gone = ObjectId::from_parts(first.index(), first.generation() + 1);
    for event in [first_readable, gone] {
        assert_eq!(
            with(vec![hook(event, SERVICE)]).unwrap_err(),
            error(hook(event, SERVICE), HookFault::NotAnEvent)
        );
    }
    // A service with no server.
    let nobody = ServiceId(OTHER_SERVICE.0 + 1);
    assert_eq!(
        with(vec![hook(first, nobody)]).unwrap_err(),
        error(hook(first, nobody), HookFault::UnknownService)
    );
    // The other server's event, and the guest's own.
    let (guests, _) = kernel.event;
    for (event, service) in [(second, SERVICE), (first, OTHER_SERVICE), (guests, SERVICE)] {
        assert_eq!(
            with(vec![hook(event, service)]).unwrap_err(),
            error(hook(event, service), HookFault::NotMadeBy)
        );
    }
    // Either one alone, with its own service: accepted.
    assert!(with(vec![hook(second, OTHER_SERVICE)]).is_ok());
    assert!(with(vec![hook(first, SERVICE)]).is_ok());
    assert!(with(Vec::new()).is_ok());
}

#[test]
fn the_address_spaces_are_one_per_handle_table_in_process_order() {
    let mut kernel = kernel();
    let second = second_process(&mut kernel);
    let own = kernel.own();
    let bytes = second.save(&kernel).unwrap();
    let finish = |spaces: &[(ProcessId, &AddressSpace)]| {
        Objects::restore(&bytes, &kernel.tables.scheduler)
            .unwrap()
            .finish(spaces)
    };
    let mine = (own, &kernel.tables.address_space);
    let theirs = (second.process, &second.space);
    assert!(finish(&[mine, theirs]).is_ok());
    assert_eq!(
        finish(&[theirs, mine]).unwrap_err(),
        ObjectSnapshotError::SpaceOrder { process: own }
    );
    assert_eq!(
        finish(&[mine, mine]).unwrap_err(),
        ObjectSnapshotError::SpaceOrder { process: own }
    );
    assert_eq!(
        finish(&[mine]).unwrap_err(),
        ObjectSnapshotError::MissingSpace {
            process: second.process
        }
    );
    assert_eq!(
        finish(&[theirs]).unwrap_err(),
        ObjectSnapshotError::MissingSpace { process: own }
    );
    let stranger = ProcessId::from_parts(second.process.index() + 1, 1);
    let extra = (stranger, &second.space);
    assert_eq!(
        finish(&[mine, theirs, extra]).unwrap_err(),
        ObjectSnapshotError::ExtraSpace { process: stranger }
    );
    // The first process that differs: one between the two is extra; one after the second,
    // in its place, leaves the second missing.
    let between = ProcessId::from_parts(own.index(), own.generation() + 1);
    assert!(own < between && between < second.process);
    assert_eq!(
        finish(&[mine, (between, &second.space), theirs]).unwrap_err(),
        ObjectSnapshotError::ExtraSpace { process: between }
    );
    assert_eq!(
        finish(&[mine, extra]).unwrap_err(),
        ObjectSnapshotError::MissingSpace {
            process: second.process
        }
    );
}

#[test]
fn each_space_is_checked_against_the_lends_of_its_own_process() {
    // The second process's space, given as the own process's: the own lend is not there, and
    // the second's locked pages are lent to nobody of the own process.
    let mut kernel = kernel();
    let second = second_process(&mut kernel);
    let own = kernel.own();
    let bytes = second.save(&kernel).unwrap();
    let pending = Objects::restore(&bytes, &kernel.tables.scheduler).unwrap();
    let heap = kernel.heap;
    // Both heaps start at the same address; the second's first two pages are locked, kept
    // by nobody, where the own lend keeps read.
    assert_eq!(second.space.regions().heap.start, heap);
    assert_eq!(
        pending
            .finish(&[(own, &second.space), (second.process, &second.space)])
            .unwrap_err(),
        ObjectSnapshotError::Lends {
            process: own,
            error: LendError::Lend {
                address: heap + PAGE_SIZE,
                size: 2 * PAGE_SIZE,
                fault: LendFault::Permission
            }
        }
    );
    // A space with no lends of its own, its locked pages nobody's: the second's lend gone.
    let mut objects = kernel.objects().clone();
    let mut handles = second.handles.clone();
    let lent = handles.handles()[0];
    handles.remove(&mut objects, lent).unwrap();
    assert_eq!(objects.take_unlocks().len(), 1);
    let (_, spaces) = second.tables(&kernel);
    let error = objects
        .save(
            &[(own, &kernel.tables.handles), (second.process, &handles)],
            &kernel.hooked,
            &kernel.tables.scheduler,
            &spaces,
        )
        .unwrap_err();
    let start = second.space.regions().heap.start;
    assert_eq!(
        error,
        ObjectSnapshotError::Lends {
            process: second.process,
            error: LendError::Unlent { address: start }
        }
    );
    assert!(objects.get(second.transfer).is_err());
}

// ----- The order of the checks ---------------------------------------------------------------

#[test]
fn the_handle_tables_come_before_the_objects() {
    let kernel = kernel();
    let own = kernel.own();
    let (writable, _) = kernel.event;
    // Handle tables (check 2) before objects (check 3): a damaged layout and an object with no
    // references.
    let mut case = Case::new(&kernel);
    case.body.handles[0].table.next_linear_id = 0;
    case.row(writable).refs = 0;
    assert_eq!(
        case.error(),
        ObjectSnapshotError::HandleLayout {
            process: own,
            error: HandleLayoutError::NextLinearId(0)
        }
    );
    // Within a handle table, the order before the process: a second table for an unknown
    // process that comes before the own one.
    let mut case = Case::new(&kernel);
    let unknown = ProcessId::from_parts(0, 0);
    assert!(unknown < own);
    let table = case.body.handles[0].table.clone();
    case.body.handles.push(SavedHandles {
        process: unknown,
        table,
    });
    assert_eq!(
        case.error(),
        ObjectSnapshotError::HandleOrder { process: unknown }
    );
}

#[test]
fn the_objects_are_checked_in_slot_order_and_each_check_in_its_order() {
    let kernel = kernel();
    let own = kernel.own();
    let (writable, readable) = kernel.event;
    let (_, transfer) = kernel.transfer;
    // The first faulty object in slot order, either way round.
    for (low, high) in [(writable, readable), (readable, writable)] {
        let mut case = Case::new(&kernel);
        case.row(low).refs = 0;
        case.row(high).object = SavedObject::Session;
        let first = low.min(high);
        let fault = if first == low {
            ObjectFault::NoReferences
        } else {
            ObjectFault::Session
        };
        assert_eq!(case.error(), object_error(first, fault));
    }
    // Within an object, the reference before the kind.
    let mut case = Case::new(&kernel);
    case.row(readable).refs = 0;
    case.row(readable).object = SavedObject::Session;
    assert_eq!(
        case.error(),
        object_error(readable, ObjectFault::NoReferences)
    );
    // A shared memory's pages before their overlap: a page-less one over the first.
    let [first_shared, second_shared] = kernel.shared;
    let mut case = Case::new(&kernel);
    let base = match kernel.objects().get(first_shared) {
        Ok(Object::SharedMemory(memory)) => memory.backing().base(),
        other => unreachable!("{other:?}"),
    };
    case.row(second_shared).object = SavedObject::SharedMemory {
        base,
        size: PAGE_SIZE,
        user_permission: SvcPermission::None.bits(),
    };
    assert_eq!(
        case.error(),
        object_error(second_shared, ObjectFault::SharedMemory)
    );
    // A transfer memory's permission before its lock, its lock before its owner.
    let unknown_owner = ProcessId::from_parts(own.index() + 7, 1);
    for (permission, fault) in [
        (SvcPermission::ReadExecute, ObjectFault::TransferPermission),
        (SvcPermission::Read, ObjectFault::Unlocked),
    ] {
        let mut case = Case::new(&kernel);
        case.row(transfer).object = SavedObject::TransferMemory {
            owner: unknown_owner,
            address: kernel.heap + PAGE_SIZE,
            size: 2 * PAGE_SIZE,
            owner_permission: permission.bits(),
            locked: false,
        };
        assert_eq!(case.error(), object_error(transfer, fault));
    }
}

#[test]
fn the_objects_come_before_the_handles_and_the_handles_before_the_hooks() {
    let kernel = kernel();
    let own = kernel.own();
    let (_, readable) = kernel.event;
    let (handle, session) = kernel.session;
    // Objects (check 3) before handles (check 4): a stale handle and a faulty object.
    let mut case = Case::new(&kernel);
    case.body.objects.remove(session).unwrap();
    case.row(readable).object = SavedObject::Session;
    assert_eq!(case.error(), object_error(readable, ObjectFault::Session));
    // Handles (check 4) before hooks (check 5).
    let mut case = Case::new(&kernel);
    case.body.objects.remove(session).unwrap();
    case.body.hooked.reverse();
    assert_eq!(
        case.error(),
        ObjectSnapshotError::Handle {
            process: own,
            handle,
            fault: HandleFault::Stale
        }
    );
    // The first stale handle, lowest index first: the session's and the guest event's
    // readable end's handles both stale.
    let readable_handle = kernel
        .tables
        .handles
        .entries()
        .find(|(_, object)| *object == readable)
        .map(|(handle, _)| handle)
        .unwrap();
    let mut case = Case::new(&kernel);
    case.body.objects.remove(session).unwrap();
    case.body.objects.remove(readable).unwrap();
    let first_stale = handle.index().min(readable_handle.index());
    let expected = if first_stale == handle.index() {
        handle
    } else {
        readable_handle
    };
    assert_eq!(
        case.error(),
        ObjectSnapshotError::Handle {
            process: own,
            handle: expected,
            fault: HandleFault::Stale
        }
    );
}

#[test]
fn the_hooks_are_checked_in_order_and_each_check_in_its_order() {
    let kernel = kernel();
    let (writable, _) = kernel.event;
    // Within a hook: the order before the event, the event before the service, the service
    // before who made it.
    let [(first, first_readable), (second, _)] = kernel.hooks;
    assert!(first_readable < second);
    let hook = |event, service| HookedEvent { event, service };
    let nobody = ServiceId(0x99);
    for (hooked, fault) in [
        (
            vec![hook(second, OTHER_SERVICE), hook(first_readable, nobody)],
            (hook(first_readable, nobody), HookFault::Order),
        ),
        (
            vec![hook(first_readable, nobody)],
            (hook(first_readable, nobody), HookFault::NotAnEvent),
        ),
        (
            vec![hook(writable, nobody)],
            (hook(writable, nobody), HookFault::UnknownService),
        ),
    ] {
        let mut case = Case::new(&kernel);
        case.body.hooked = hooked;
        assert_eq!(
            case.error(),
            ObjectSnapshotError::Hook {
                hooked: fault.0,
                fault: fault.1
            }
        );
    }
    // The first faulty hook in order.
    let mut case = Case::new(&kernel);
    case.body.hooked = vec![hook(first, OTHER_SERVICE), hook(second, SERVICE)];
    assert_eq!(
        case.error(),
        ObjectSnapshotError::Hook {
            hooked: hook(first, OTHER_SERVICE),
            fault: HookFault::NotMadeBy
        }
    );
}

#[test]
fn the_spaces_are_matched_before_their_lends_are_checked_and_in_process_order() {
    let mut kernel = kernel();
    let second = second_process(&mut kernel);
    let own = kernel.own();
    let bytes = second.save(&kernel).unwrap();
    let finish = |spaces: &[(ProcessId, &AddressSpace)]| {
        Objects::restore(&bytes, &kernel.tables.scheduler)
            .unwrap()
            .finish(spaces)
    };
    // A wrong space for the own process, and the second's missing.
    assert_eq!(
        finish(&[(own, &second.space)]).unwrap_err(),
        ObjectSnapshotError::MissingSpace {
            process: second.process
        }
    );
    // Both spaces swapped: each fails its lends; the own process's is reported.
    assert_eq!(
        finish(&[
            (own, &second.space),
            (second.process, &kernel.tables.address_space)
        ])
        .unwrap_err(),
        ObjectSnapshotError::Lends {
            process: own,
            error: LendError::Lend {
                address: kernel.heap + PAGE_SIZE,
                size: 2 * PAGE_SIZE,
                fault: LendFault::Permission
            }
        }
    );
    // The own space right, the second's the own one: the second's is reported.
    let start = second.space.regions().heap.start;
    assert_eq!(
        finish(&[
            (own, &kernel.tables.address_space),
            (second.process, &kernel.tables.address_space)
        ])
        .unwrap_err(),
        ObjectSnapshotError::Lends {
            process: second.process,
            error: LendError::Lend {
                address: start,
                size: 2 * PAGE_SIZE,
                fault: LendFault::NotLocked
            }
        }
    );
}

// ----- Words ---------------------------------------------------------------------------------

#[test]
fn the_errors_read_as_plain_words() {
    let object = ObjectId::from_parts(4, 2);
    let process = ProcessId::from_parts(1, 3);
    assert_eq!(
        object_error(object, ObjectFault::NoReferences).to_string(),
        "object 4 (generation 2) has no references"
    );
    assert_eq!(
        object_error(object, ObjectFault::ResourceLimit).to_string(),
        "object 4 (generation 2) is a resource limit whose process's row does not name it back"
    );
    assert_eq!(
        object_error(object, ObjectFault::ReadableNotIssued).to_string(),
        "object 4 (generation 2) is an event whose readable end names an id the object table \
         has not made yet"
    );
    assert_eq!(
        ObjectSnapshotError::PendingUnlocks { count: 1 }.to_string(),
        "released transfer memories whose pages are still locked: 1 (a snapshot is taken \
         between calls, when there are none)"
    );
    assert_eq!(
        ObjectSnapshotError::PendingFrees { count: 2 }.to_string(),
        "released shared memories whose pages are still to be given back: 2 (a snapshot is \
         taken between calls, when there are none)"
    );
    assert_eq!(
        ObjectSnapshotError::Handle {
            process,
            handle: Handle::from_raw(0x8001),
            fault: HandleFault::Overused
        }
        .to_string(),
        "handle 0x00008001 of process 1 (generation 3) is one handle more to its object than \
         the object has references"
    );
    assert_eq!(
        ObjectSnapshotError::Hook {
            hooked: HookedEvent {
                event: object,
                service: ServiceId(0x31)
            },
            fault: HookFault::NotMadeBy
        }
        .to_string(),
        "the hook of service 49 on object 4 (generation 2) names an event its service's \
         server did not make"
    );
    assert_eq!(
        ObjectSnapshotError::HookWithoutServer { event: object }.to_string(),
        "object 4 (generation 2) has a hook but no server made it"
    );
    assert_eq!(
        ObjectSnapshotError::HandleLayout {
            process,
            error: HandleLayoutError::NextLinearId(0)
        }
        .to_string(),
        "the handle table of process 1 (generation 3) is damaged: next linear id 0 is outside \
         the valid range"
    );
    assert_eq!(
        ObjectSnapshotError::MissingSpace { process }.to_string(),
        "no address space was given for process 1 (generation 3), whose handle table the \
         snapshot holds"
    );
    assert_eq!(
        ObjectSnapshotError::Lends {
            process,
            error: LendError::Unlent { address: 0x5000 }
        }
        .to_string(),
        "the memory of process 1 (generation 3) disagrees with its transfer memories: the \
         locked page at 0x5000 is lent to no transfer memory and is no alias's source"
    );
}

#[test]
fn the_first_stale_handle_is_found_in_process_order() {
    // A stale handle in each process's table: the own process's comes first, whichever slot
    // its object has.
    let mut kernel = kernel();
    let second = second_process(&mut kernel);
    let own = kernel.own();
    let bytes = second.save(&kernel).unwrap();
    let mut body: Body = decode(FORMAT, &bytes).unwrap();
    let (handle, session) = kernel.session;
    body.objects.remove(session).unwrap();
    body.objects.remove(second.transfer).unwrap();
    let restore = |body: &Body| second.restore(&kernel, &encode(FORMAT, body).unwrap());
    assert_eq!(
        restore(&body).unwrap_err(),
        ObjectSnapshotError::Handle {
            process: own,
            handle,
            fault: HandleFault::Stale
        }
    );
    // With the own one sound, the second's.
    let mut body: Body = decode(FORMAT, &bytes).unwrap();
    body.objects.remove(second.transfer).unwrap();
    assert_eq!(
        restore(&body).unwrap_err(),
        ObjectSnapshotError::Handle {
            process: second.process,
            handle: second.handles.handles()[0],
            fault: HandleFault::Stale
        }
    );
}

/// One live operation on a started process, its operands picks among what exists.
#[derive(Debug, Clone)]
enum LiveOp {
    /// `CreateEvent`: a handle to each end, the creator's references closed.
    Event,
    /// Closes the handle this picks, and unlocks a transfer memory that goes with it.
    Close(usize),
    /// `CreateTransferMemory` of `n` heap pages from page `at`, keeping the permission this
    /// picks.
    Lend(u64, u64, usize),
    /// Signals the object the handle this picks names, if it is an event.
    Signal(usize),
    /// A second handle to the object the handle this picks names.
    Duplicate(usize),
}

fn any_live_op() -> impl Strategy<Value = LiveOp> {
    prop_oneof![
        Just(LiveOp::Event),
        any::<usize>().prop_map(LiveOp::Close),
        (0_u64..8, 1_u64..4, 0_usize..3).prop_map(|(at, n, keep)| LiveOp::Lend(at, n, keep)),
        any::<usize>().prop_map(LiveOp::Signal),
        any::<usize>().prop_map(LiveOp::Duplicate),
    ]
}

/// The fixture's process, started, with eight pages of heap: its tables and the heap's start.
fn started() -> (Tables, u64) {
    let (space, _log, stack_top) = fixture::loaded();
    let mut tables =
        process::create(&ProcessDescription::homebrew(Console::Switch2), space).unwrap();
    let tls = tables.address_space.regions().kernel_map.start + GUARD_PAGES * PAGE_SIZE;
    let memory = FlatMemory::new(tls, usize::try_from(PAGE_SIZE).unwrap());
    process::run(&mut tables, &memory, ENTRY, stack_top).unwrap();
    let heap = tables.address_space.set_heap_size(8 * PAGE_SIZE).unwrap();
    (tables, heap)
}

/// Applies `op` to `tables`. Refusals are part of the run; the main thread's handle is never
/// closed (closing the last handle to a running thread is the thread table's business).
fn apply_live(tables: &mut Tables, heap: u64, op: &LiveOp) {
    let own = tables.processes.own();
    let handles: Vec<Handle> = tables.handles.handles();
    let pick = |n: usize| handles.get(n % handles.len().max(1)).copied();
    match *op {
        LiveOp::Event => {
            if let Ok((writable, readable)) = tables.scheduler.create_event() {
                for end in [writable, readable] {
                    let _added = tables.handles.add(tables.scheduler.objects_mut(), end);
                    tables.scheduler.close_object(end).unwrap();
                }
            }
        }
        LiveOp::Close(n) => {
            // The main thread's handle stays.
            let thread = |handle: &Handle| {
                let object = tables.handles.get(*handle);
                matches!(
                    object.map(|object| tables.scheduler.objects().get(object)),
                    Ok(Ok(Object::Thread(_)))
                )
            };
            if let Some(handle) = pick(n).filter(|handle| !thread(handle)) {
                tables
                    .handles
                    .remove(tables.scheduler.objects_mut(), handle)
                    .unwrap();
                unlock_released(&mut tables.scheduler, &mut tables.address_space, own).unwrap();
            }
        }
        LiveOp::Lend(at, n, keep) => {
            let keeps = [
                SvcPermission::None,
                SvcPermission::Read,
                SvcPermission::ReadWrite,
            ];
            let Tables {
                handles,
                address_space,
                scheduler,
                ..
            } = tables;
            let _lent = create_transfer_memory(
                handles,
                address_space,
                scheduler,
                own,
                heap + at * PAGE_SIZE,
                n * PAGE_SIZE,
                u32::from(keeps[keep % keeps.len()].bits()),
            );
        }
        LiveOp::Signal(n) => {
            if let Some(object) = pick(n).and_then(|handle| tables.handles.get(handle).ok()) {
                let _signalled = tables.scheduler.signal_object(object);
            }
        }
        LiveOp::Duplicate(n) => {
            if let Some(object) = pick(n).and_then(|handle| tables.handles.get(handle).ok()) {
                let _added = tables.handles.add(tables.scheduler.objects_mut(), object);
            }
        }
    }
}

proptest! {
    #![proptest_config(config(96))]

    /// Random live operations on a started process: after each, its tables save, restore as
    /// they were, and save again to the same bytes.
    #[test]
    fn tables_built_by_random_operations_save_and_restore_as_they_were(
        ops in prop::collection::vec(any_live_op(), 1..24)
    ) {
        let (mut tables, heap) = started();
        for op in &ops {
            apply_live(&mut tables, heap, op);
            let own = tables.processes.own();
            let objects = tables.scheduler.objects();
            let save = |objects: &Objects, handles: &HandleTable| {
                objects.save(
                    &[(own, handles)],
                    &[],
                    &tables.scheduler,
                    &[(own, &tables.address_space)],
                )
            };
            let bytes = save(objects, &tables.handles);
            prop_assert!(bytes.is_ok(), "save after {:?}: {:?}", op, bytes);
            let bytes = bytes.unwrap_or_default();
            let restored = Objects::restore(&bytes, &tables.scheduler)
                .and_then(|pending| pending.finish(&[(own, &tables.address_space)]));
            prop_assert!(restored.is_ok(), "restore after {:?}: {:?}", op, restored);
            if let Ok(restored) = restored {
                prop_assert_eq!(&restored.objects, objects);
                prop_assert_eq!(&restored.handles[0].1, &tables.handles);
                prop_assert_eq!(save(&restored.objects, &restored.handles[0].1), Ok(bytes));
            }
        }
    }
}

#[test]
fn the_random_operations_reach_lends_signals_and_shared_handles() {
    // A fixed sequence of the generated kind, so the property test is known to reach two lends
    // and one of them closed, a signalled event and an object with two handles. Picks count
    // the handles lowest index first, and the table hands out its highest index first: after
    // the event, the readable end, the writable end, the second lend, the first, the thread.
    let (mut tables, heap) = started();
    for op in [
        LiveOp::Lend(0, 2, 1),
        LiveOp::Lend(2, 1, 0),
        LiveOp::Event,
        LiveOp::Signal(0),
        LiveOp::Duplicate(1),
        LiveOp::Close(3),
    ] {
        apply_live(&mut tables, heap, &op);
    }
    let kinds: Vec<&str> = tables
        .handles
        .entries()
        .map(|(_, object)| match tables.scheduler.objects().get(object) {
            Ok(Object::Thread(_)) => "thread",
            Ok(Object::Event(_)) => "event",
            Ok(Object::ReadableEvent(state)) if state.is_signalled() => "signalled",
            Ok(Object::ReadableEvent(_)) => "readable",
            Ok(Object::TransferMemory(_)) => "lend",
            other => unreachable!("{other:?}"),
        })
        .collect();
    assert_eq!(kinds, ["event", "signalled", "event", "lend", "thread"]);
    // The second lend's page was given back; the first's are still lent.
    let locked = |page: u64| {
        let block = tables
            .address_space
            .query_info(heap + page * PAGE_SIZE)
            .unwrap();
        block.attribute().contains(MemoryAttribute::LOCKED)
    };
    assert_eq!((locked(0), locked(1), locked(2)), (true, true, false));
}

#[test]
fn a_restore_in_order_resolves_the_address_spaces_shared_memory_from_the_pending_table() {
    // The order a whole restore takes: the object table, then the address space with the
    // shared memories the pending table holds, then the lends against the new space.
    let kernel = kernel();
    let bytes = kernel.save().unwrap();
    let space_bytes = kernel.tables.address_space.save().unwrap();
    let pending = Objects::restore(&bytes, &kernel.tables.scheduler).unwrap();
    assert_eq!(pending.hooked(), kernel.hooked.as_slice());
    let [mapped, unmapped] = kernel.shared;
    for object in [mapped, unmapped] {
        let expected = match kernel.objects().get(object) {
            Ok(Object::SharedMemory(memory)) => memory,
            other => unreachable!("{other:?}"),
        };
        assert_eq!(pending.shared_memory(object), Some(expected));
    }
    let (_, transfer) = kernel.transfer;
    assert_eq!(pending.shared_memory(transfer), None);
    let (recording, _log) = crate::memory::testing::RecordingMapper::new();
    let space = AddressSpace::restore(&space_bytes, Box::new(recording), |object| {
        pending.shared_memory(object)
    })
    .unwrap();
    let restored = pending.finish(&[(kernel.own(), &space)]).unwrap();
    assert_eq!(&restored.objects, kernel.objects());
}
