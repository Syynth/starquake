//! A server's own reference to a lent transfer memory (`dispatch/hooks.rs`, "A server's own
//! reference"): kept during a request, it outlives the guest's handle; only the server's notes
//! say whose reference a release drops; the last one to go unlocks the owner's pages.

use super::*;

/// The other service number a test names, which has no server.
const NO_SERVER: ServiceId = ServiceId(0x53);

/// The guest lends the data's first page, keeping read access: the handle and the object.
fn lend(world: &World) -> (Handle, ObjectId) {
    let mut regs = GuestRegs::default();
    regs.x[1] = DATA;
    regs.x[2] = PAGE_SIZE;
    regs.x[3] = u64::from(crate::memory::SvcPermission::Read.bits());
    world.call(SvcId::CreateTransferMemory, &mut regs);
    assert_eq!(regs.x[0], 0);
    let handle = Handle::from_register(regs.x[1]);
    (handle, object_of(world, handle))
}

/// The result word of the guest's `CreateTransferMemory` of the data's first page again: a
/// page whose lend still locks it is refused, one whose lend is gone is taken.
fn lend_again(world: &World) -> u64 {
    let mut regs = GuestRegs::default();
    regs.x[1] = DATA;
    regs.x[2] = PAGE_SIZE;
    regs.x[3] = u64::from(crate::memory::SvcPermission::Read.bits());
    world.call(SvcId::CreateTransferMemory, &mut regs);
    regs.x[0]
}

/// The references the object table holds to `object`.
fn refs(world: &World, object: ObjectId) -> u32 {
    world
        .dispatcher
        .with::<(&Scheduler,), _>(|(scheduler,)| scheduler.objects().refs(object))
}

/// Has the server keep `times` references to `object` in one request; each keep's answer.
fn keep(world: &World, object: ObjectId, times: usize) -> Vec<SvcResult<(u64, u64)>> {
    let answers = Arc::new(Mutex::new(Vec::new()));
    let out = Arc::clone(&answers);
    let result = world.request(Box::new(move |cx, _| {
        for _ in 0..times {
            let kept = cx
                .keep_transfer_memory(object)
                .map(|transfer| (transfer.address(), transfer.size()));
            out.lock().unwrap().push(kept);
        }
    }));
    assert_eq!(result, 0);
    answers.lock().unwrap().clone()
}

fn result_code(error: KernelError) -> u64 {
    u64::from(error.code().raw())
}

#[test]
fn a_kept_transfer_memory_outlives_the_guest_s_handle_until_released() {
    let world = World::new();
    let (handle, lent) = lend(&world);
    assert_eq!(refs(&world, lent), 1);
    assert_eq!(keep(&world, lent, 1), [Ok((DATA, PAGE_SIZE))]);
    // The guest's handle and the server's own reference.
    assert_eq!(refs(&world, lent), 2);
    assert_eq!(world.dispatcher.kept_transfer_memory(SERVICE, lent), 1);
    // The guest closes its handle: the transfer memory stays, and so does the lock on its pages.
    assert_eq!(world.call_with_handle(SvcId::CloseHandle, handle), 0);
    assert_eq!(refs(&world, lent), 1);
    let found = world.dispatcher.transfer_memory(lent).unwrap();
    assert_eq!((found.address(), found.size()), (DATA, PAGE_SIZE));
    assert_eq!(
        lend_again(&world),
        result_code(KernelError::InvalidCurrentMemory)
    );
    // The server lets it go, outside any request: it is gone, and its pages are the owner's
    // again.
    world
        .dispatcher
        .release_kept_transfer_memory(SERVICE, lent)
        .unwrap();
    assert_eq!(refs(&world, lent), 0);
    assert_eq!(world.dispatcher.kept_transfer_memory(SERVICE, lent), 0);
    assert_eq!(
        world.dispatcher.transfer_memory(lent),
        Err(KernelError::InvalidHandle)
    );
    assert_eq!(lend_again(&world), 0);
}

#[test]
fn releasing_before_the_guest_closes_leaves_the_guest_s_reference() {
    let world = World::new();
    let (handle, lent) = lend(&world);
    keep(&world, lent, 1);
    world
        .dispatcher
        .release_kept_transfer_memory(SERVICE, lent)
        .unwrap();
    // The guest's handle still names it, locked.
    assert_eq!(refs(&world, lent), 1);
    assert!(world.dispatcher.transfer_memory(lent).is_ok());
    assert_eq!(
        lend_again(&world),
        result_code(KernelError::InvalidCurrentMemory)
    );
    // Closing the handle is what makes it go now.
    assert_eq!(world.call_with_handle(SvcId::CloseHandle, handle), 0);
    assert_eq!(refs(&world, lent), 0);
    assert_eq!(lend_again(&world), 0);
}

#[test]
fn only_a_kept_reference_is_released() {
    let world = World::new();
    let (handle, lent) = lend(&world);
    // Nothing kept: the guest's reference is not the server's to drop.
    assert_eq!(
        world.dispatcher.release_kept_transfer_memory(SERVICE, lent),
        Err(KernelError::InvalidHandle)
    );
    assert_eq!(refs(&world, lent), 1);
    keep(&world, lent, 1);
    // Another service, and one with no server, keep nothing.
    assert_eq!(
        world
            .dispatcher
            .release_kept_transfer_memory(NO_SERVER, lent),
        Err(KernelError::InvalidHandle)
    );
    assert_eq!(world.dispatcher.kept_transfer_memory(NO_SERVER, lent), 0);
    assert_eq!(refs(&world, lent), 2);
    // One kept, one released; the second release is refused and leaves the guest's.
    world
        .dispatcher
        .release_kept_transfer_memory(SERVICE, lent)
        .unwrap();
    assert_eq!(
        world.dispatcher.release_kept_transfer_memory(SERVICE, lent),
        Err(KernelError::InvalidHandle)
    );
    assert_eq!(refs(&world, lent), 1);
    assert_eq!(world.call_with_handle(SvcId::CloseHandle, handle), 0);
    assert_eq!(refs(&world, lent), 0);
}

#[test]
fn keeping_twice_keeps_two_references_each_released_once() {
    let world = World::new();
    let (handle, lent) = lend(&world);
    assert_eq!(
        keep(&world, lent, 2),
        [Ok((DATA, PAGE_SIZE)), Ok((DATA, PAGE_SIZE))]
    );
    assert_eq!(refs(&world, lent), 3);
    assert_eq!(world.dispatcher.kept_transfer_memory(SERVICE, lent), 2);
    assert_eq!(world.call_with_handle(SvcId::CloseHandle, handle), 0);
    world
        .dispatcher
        .release_kept_transfer_memory(SERVICE, lent)
        .unwrap();
    // One left: still there, still locked.
    assert_eq!(refs(&world, lent), 1);
    assert_eq!(world.dispatcher.kept_transfer_memory(SERVICE, lent), 1);
    assert!(world.dispatcher.transfer_memory(lent).is_ok());
    world
        .dispatcher
        .release_kept_transfer_memory(SERVICE, lent)
        .unwrap();
    assert_eq!(refs(&world, lent), 0);
    assert_eq!(lend_again(&world), 0);
}

#[test]
fn only_a_live_transfer_memory_is_kept() {
    let world = World::new();
    let (handle, lent) = lend(&world);
    let (_, writable, _) = world.hooked_event(Arc::new(Counting::default()));
    let never_made = ObjectId::from_parts(0x7FFF, 9);
    for object in [writable, never_made] {
        assert_eq!(keep(&world, object, 1), [Err(KernelError::InvalidHandle)]);
        assert_eq!(world.dispatcher.kept_transfer_memory(SERVICE, object), 0);
    }
    // The event's references are as they were: the server's two (its creator's, kept for as
    // long as it lives) and the guest's handle.
    assert_eq!(refs(&world, writable), 2);
    // Gone, it is not kept either.
    assert_eq!(world.call_with_handle(SvcId::CloseHandle, handle), 0);
    assert_eq!(keep(&world, lent, 1), [Err(KernelError::InvalidHandle)]);
    assert_eq!(world.dispatcher.kept_transfer_memory(SERVICE, lent), 0);
}

#[test]
fn a_request_releases_what_it_kept() {
    let world = World::new();
    let (handle, lent) = lend(&world);
    let answers = Arc::new(Mutex::new(Vec::new()));
    let out = Arc::clone(&answers);
    let result = world.request(Box::new(move |cx, _| {
        let mut answers = out.lock().unwrap();
        answers.push(cx.keep_transfer_memory(lent).map(|_| ()));
        answers.push(cx.release_kept_transfer_memory(lent));
        // A second release finds nothing kept.
        answers.push(cx.release_kept_transfer_memory(lent));
    }));
    assert_eq!(result, 0);
    assert_eq!(
        *answers.lock().unwrap(),
        [Ok(()), Ok(()), Err(KernelError::InvalidHandle)]
    );
    assert_eq!(refs(&world, lent), 1);
    assert_eq!(world.dispatcher.kept_transfer_memory(SERVICE, lent), 0);
    assert_eq!(world.call_with_handle(SvcId::CloseHandle, handle), 0);
    assert_eq!(refs(&world, lent), 0);
}

/// The guest connects to `hook:`; its session handle.
fn connect(world: &World) -> Handle {
    world.memory.write(NAMES, b"hook:\0").unwrap();
    let mut regs = GuestRegs::default();
    regs.x[1] = NAMES;
    world.call(SvcId::ConnectToNamedPort, &mut regs);
    assert_eq!(regs.x[0], 0);
    Handle::from_register(regs.x[1])
}

#[test]
fn open_sessions_counts_the_server_s_sessions_whose_client_has_not_gone() {
    let world = World::new();
    assert_eq!(world.dispatcher.open_sessions(SERVICE), 0);
    let first = connect(&world);
    assert_eq!(world.dispatcher.open_sessions(SERVICE), 1);
    let second = connect(&world);
    assert_eq!(world.dispatcher.open_sessions(SERVICE), 2);
    assert_eq!(world.dispatcher.open_sessions(NO_SERVER), 0);
    assert_eq!(world.call_with_handle(SvcId::CloseHandle, first), 0);
    assert_eq!(world.dispatcher.open_sessions(SERVICE), 1);
    assert_eq!(world.call_with_handle(SvcId::CloseHandle, second), 0);
    assert_eq!(world.dispatcher.open_sessions(SERVICE), 0);
}

#[test]
fn a_keep_for_a_server_the_table_does_not_know_drops_the_reference_it_opened() {
    // Through `IpcContext` the server is always one the table knows; driven directly, the note
    // fails after the reference was opened, and the reference is dropped again.
    let world = World::new();
    let (_handle, lent) = lend(&world);
    let unknown = ServerId::from_parts(0x7F, 3);
    let kept = world
        .dispatcher
        .with::<(&mut Scheduler,), _>(|(scheduler,)| {
            keep_transfer_memory_in(scheduler, unknown, lent).map(|_| ())
        });
    assert_eq!(
        kept,
        Err(SvcError::internal(&crate::ipc::IpcError::UnknownServer(
            unknown
        )))
    );
    // The guest's handle is the only reference, and the pages are still locked by it alone.
    assert_eq!(refs(&world, lent), 1);
    assert_eq!(world.dispatcher.kept_transfer_memory(SERVICE, lent), 0);
    assert_eq!(world.dispatcher.kept_transfer_memories(SERVICE), 0);
}

#[test]
fn kept_transfer_memories_counts_every_reference_the_server_keeps() {
    let world = World::new();
    let (_, lent) = lend(&world);
    assert_eq!(world.dispatcher.kept_transfer_memories(SERVICE), 0);
    keep(&world, lent, 1);
    assert_eq!(world.dispatcher.kept_transfer_memories(SERVICE), 1);
    // A second object, of a different count, and the same one again: every reference counts.
    let mut regs = GuestRegs::default();
    regs.x[1] = DATA + PAGE_SIZE;
    regs.x[2] = PAGE_SIZE;
    regs.x[3] = u64::from(crate::memory::SvcPermission::Read.bits());
    world.call(SvcId::CreateTransferMemory, &mut regs);
    assert_eq!(regs.x[0], 0);
    let second = object_of(&world, Handle::from_register(regs.x[1]));
    keep(&world, second, 2);
    keep(&world, lent, 1);
    assert_eq!(world.dispatcher.kept_transfer_memories(SERVICE), 4);
    assert_eq!(world.dispatcher.kept_transfer_memories(NO_SERVER), 0);
    world
        .dispatcher
        .release_kept_transfer_memory(SERVICE, second)
        .unwrap();
    assert_eq!(world.dispatcher.kept_transfer_memories(SERVICE), 3);
}
