//! The semaphore mirror page (tentative, decision log 2026-11-20): which registrations gsv lists
//! in each address space's table, and when (the `memory` module docs, "The semaphore mirror
//! page"). A table that lists a registration answers a release in its second page with the same
//! offset in its first.

use crate::completion::testing::{Locks, process};
use starquake_gpu::vm::{SEMAPHORE_MIRROR_OFFSET, SEMAPHORE_REGISTRATION_BYTES};

use super::*;

/// A two-page registration's start, away from the other tests' [`MEMORY`].
const SEMAPHORES: u64 = MEMORY + 0x10_0000;

/// A release 0x10 bytes into the second page of a registration at `start`.
fn release_in(start: u64) -> u64 {
    start + SEMAPHORE_MIRROR_OFFSET + 0x10
}

/// Whether `vm` copies a release in the second page of the registration at `start` into its
/// first page.
fn lists(vm: &GpuVm, start: u64) -> bool {
    match vm.semaphore_mirror(release_in(start)) {
        Some(copy) => {
            assert_eq!(copy, start + 0x10);
            true
        }
        None => false,
    }
}

impl FakeFront {
    fn table(&self, space: AsId) -> Arc<GpuVm> {
        Arc::clone(&self.tables.lock().unwrap()[&space])
    }
}

/// Through the gpu worker's 101: a two-page registration is listed at the request in every
/// address space, each of which answers the copy; the four-page registration of [`ready`] is
/// not.
#[test]
fn a_two_page_registration_is_listed_in_every_space() {
    let mut s = ready();
    let other = s.rig.address_space();
    s.rig
        .register(s.worker, SEMAPHORES, SEMAPHORE_REGISTRATION_BYTES, s.device);
    for space in [s.space, other] {
        let table = s.rig.front.table(AsId(space));
        assert!(lists(&table, SEMAPHORES), "{space}");
        assert!(!lists(&table, MEMORY), "{space}");
    }
}

/// NX2's `last - start + 1 == 0x2000`: one page fewer or more is not listed.
#[test]
fn only_two_page_registrations_are_listed() {
    let (mut gpu, front) = bare();
    let space = gpu.create_address_space(&front).unwrap();
    let starts = [SEMAPHORES, SEMAPHORES + 0x10_0000, SEMAPHORES + 0x20_0000];
    let sizes = [
        SEMAPHORE_REGISTRATION_BYTES - PAGE_SIZE,
        SEMAPHORE_REGISTRATION_BYTES,
        SEMAPHORE_REGISTRATION_BYTES + PAGE_SIZE,
    ];
    for (start, size) in starts.into_iter().zip(sizes) {
        register_bare(&mut gpu, page_range(start, size));
    }
    let table = front.table(space);
    assert_eq!(
        starts.map(|start| lists(&table, start)),
        [false, true, false]
    );
}

/// A table made after the registration (gpu 33) lists it too.
#[test]
fn a_new_space_lists_the_registrations_held() {
    let (mut gpu, front) = bare();
    register_bare(
        &mut gpu,
        page_range(SEMAPHORES, SEMAPHORE_REGISTRATION_BYTES),
    );
    register_bare(&mut gpu, page_range(MEMORY, MEMORY_SIZE));
    let space = gpu.create_address_space(&front).unwrap();
    let table = front.table(space);
    assert!(lists(&table, SEMAPHORES));
    assert!(!lists(&table, MEMORY));
}

/// Through the gpu worker's 102: the registration is unlisted in every space at the request,
/// before the GPU has applied the removal (the table still maps it then).
#[test]
fn a_destroy_unlists_at_the_request() {
    let mut s = ready();
    let other = s.rig.address_space();
    let handle = s
        .rig
        .register(s.worker, SEMAPHORES, SEMAPHORE_REGISTRATION_BYTES, s.device);
    let reply = s.rig.send_on(s.worker, &[0x76, 2, handle, 0]);
    assert_eq!(result_of(&reply), ResultCode::SUCCESS);
    for space in [s.space, other] {
        let table = s.rig.front.table(AsId(space));
        assert!(!lists(&table, SEMAPHORES), "{space}");
        assert!(
            s.rig.at(space, SEMAPHORES).is_some(),
            "still mapped: {space}"
        );
    }
}

/// A refused 102 (no such handle) unlists nothing.
#[test]
fn a_refused_destroy_keeps_the_listing() {
    let mut s = ready();
    let handle = s
        .rig
        .register(s.worker, SEMAPHORES, SEMAPHORE_REGISTRATION_BYTES, s.device);
    let reply = s.rig.send_on(s.worker, &[0x76, 2, handle + 1, 0]);
    assert_eq!(result_of(&reply), code(KernelError::NotFound));
    assert!(lists(&s.rig.front.table(AsId(s.space)), SEMAPHORES));
}

/// Two registrations of the same two pages: the copy stays until both are destroyed (counted,
/// where NX2's list would forget the start at the first destroy).
#[test]
fn two_registrations_of_one_range_are_listed_until_both_are_destroyed() {
    let (mut gpu, front) = bare();
    let space = gpu.create_address_space(&front).unwrap();
    let range = page_range(SEMAPHORES, SEMAPHORE_REGISTRATION_BYTES);
    let first = register_bare(&mut gpu, range);
    let second = register_bare(&mut gpu, range);
    let event = ObjectId::from_parts(4, 5);
    let table = front.table(space);
    gpu.destroy_memory(&front, first, event).unwrap();
    assert!(lists(&table, SEMAPHORES));
    gpu.destroy_memory(&front, second, event).unwrap();
    assert!(!lists(&table, SEMAPHORES));
}

/// The front keeps a destroyed space's table: its list is emptied, and the other space's kept.
#[test]
fn a_destroyed_space_lists_nothing() {
    let (mut gpu, front) = bare();
    let space = gpu.create_address_space(&front).unwrap();
    let other = gpu.create_address_space(&front).unwrap();
    register_bare(
        &mut gpu,
        page_range(SEMAPHORES, SEMAPHORE_REGISTRATION_BYTES),
    );
    let destroyed = front.table(space);
    gpu.destroy_address_space(&front, space).unwrap();
    assert!(!lists(&destroyed, SEMAPHORES));
    assert!(lists(&front.table(other), SEMAPHORES));
}

/// A snapshot holds the registration, not the list; the restored table lists it, and only once:
/// one destroy unlists it.
#[test]
fn a_restored_table_lists_its_two_page_registrations() {
    let (mut gpu, front) = bare();
    gpu.create_address_space(&front).unwrap();
    let range = page_range(SEMAPHORES, SEMAPHORE_REGISTRATION_BYTES);
    let handle = register_bare(&mut gpu, range);
    let bytes = gpu.save().unwrap();
    let kernel = Locks {
        process: process(),
        live: [((range.start(), range.size()), 1)].into_iter().collect(),
    };
    let restored_front = FakeFront::default();
    let mut restored = GpuMemory::restore(&bytes, &kernel, &restored_front).unwrap();
    let space = restored.address_spaces().next().unwrap();
    let table = restored_front.table(space);
    assert!(lists(&table, SEMAPHORES));
    restored
        .destroy_memory(&restored_front, handle, ObjectId::from_parts(4, 5))
        .unwrap();
    assert!(!lists(&table, SEMAPHORES));
}
