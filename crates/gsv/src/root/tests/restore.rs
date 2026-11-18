//! Restoring a root beside the kernel (module docs, "Restoring"): each of 204's events a
//! snapshot's [`RootState`] keeps is a live readable event the root's server made, checked
//! before the server numbers, and a root is restored only beside a kernel.

use std::sync::Arc;

use starquake_kernel::ipc::EventHook;
use starquake_services::framework::RestoredKernel;

use super::*;

/// The refusal for a first syncpoint event that is not the root's.
const FIRST: SnapshotError = SnapshotError::FamilyObject(
    "its root's first syncpoint event is not a live readable event its server made",
);
/// The refusal for a second syncpoint event that is not the root's.
const SECOND: SnapshotError = SnapshotError::FamilyObject(
    "its root's second syncpoint event is not a live readable event its server made",
);

/// Another server's number, whose events are not the root's.
const OTHER: ServiceId = ServiceId(ROOT.0 + 1);

/// A hook that does nothing, for a hooked event's writable end.
struct Quiet;

impl EventHook for Quiet {
    fn signalled(&self, _: ObjectId) {}
}

/// A root whose 204 made both its events, and the two.
fn with_events() -> (Rig, ObjectId, ObjectId) {
    let mut rig = Rig::new();
    let reply = rig.send(&[tipc_type(BIND_SYNCPOINT_MEMORY), 0]);
    assert_eq!(copied(&reply).len(), 2);
    let SyncpointEvents::Both(first, second) = rig.state().syncpoint_events() else {
        unreachable!("204 made both events");
    };
    (rig, first, second)
}

/// Ids that are not an event the root's server made, each made on `kernel` (whose current
/// service is the root's), by what each is.
fn not_the_roots(kernel: &mut FakeKernel) -> [(&'static str, ObjectId); 4] {
    let nothing = kernel.new_object();
    kernel.set_current_service(OTHER);
    let others = kernel.create_event().unwrap();
    kernel.set_current_service(ROOT);
    let (writable, _) = kernel.create_hooked_event(Arc::new(Quiet)).unwrap();
    let page = starquake_memory::PAGE_SIZE;
    let memory = kernel
        .create_shared_memory(page, SvcPermission::Read)
        .unwrap()
        .0;
    [
        ("an id that names nothing", nothing),
        ("another server's event", others),
        ("a writable end the root made", writable),
        ("a shared memory", memory),
    ]
}

/// The bytes a root over `state` saves.
fn saved_root(state: RootState) -> Vec<u8> {
    root_server_for(state, UnknownPolicy::Strict)
        .save()
        .unwrap()
}

/// The root state `server` holds.
fn state_of(server: &Server) -> RootState {
    *server.family::<RootFamily>().unwrap().state()
}

/// Each slot of 204's events is restored only as a live readable event the root's server
/// made: every other id, in the first slot of `One` and of `Both` and in the second of
/// `Both`, is refused with that slot's exact error, and the server keeps the state it had.
#[test]
fn each_syncpoint_event_restores_only_as_one_the_root_made() {
    let (mut rig, first, second) = with_events();
    let live = rig.state();
    for (what, bad) in not_the_roots(&mut rig.kernel) {
        let cases = [
            (SyncpointEvents::One(bad), FIRST),
            (SyncpointEvents::Both(bad, second), FIRST),
            (SyncpointEvents::Both(first, bad), SECOND),
        ];
        for (events, refused) in cases {
            let forged = RootState {
                syncpoint_events: events,
                ..live
            };
            let mut restored = root_server(UnknownPolicy::Strict).with_service_id(ROOT);
            assert_eq!(
                restored.restore_beside(&saved_root(forged), &rig.kernel),
                Err(refused),
                "{what}: {events:?}"
            );
            assert_eq!(state_of(&restored), RootState::new(), "{what}");
        }
    }
}

/// The first slot is checked before the second: with both wrong, the first is reported.
#[test]
fn the_first_syncpoint_event_is_checked_first() {
    let (mut rig, _, _) = with_events();
    let live = rig.state();
    let [(_, nothing), (_, others), ..] = not_the_roots(&mut rig.kernel);
    for events in [
        SyncpointEvents::Both(nothing, others),
        SyncpointEvents::Both(others, nothing),
    ] {
        let forged = RootState {
            syncpoint_events: events,
            ..live
        };
        let mut restored = root_server(UnknownPolicy::Strict).with_service_id(ROOT);
        assert_eq!(
            restored.restore_beside(&saved_root(forged), &rig.kernel),
            Err(FIRST),
            "{events:?}"
        );
    }
}

/// A root's own events restore, beside the kernel that made them, under the number the root's
/// server is known by: the restored root carries on, and 204 hands out the same two events.
/// A server known by another number refuses the same snapshot, and so does `restore`, which has
/// no kernel; each leaves the server as it was.
#[test]
fn a_root_s_own_events_restore_and_carry_on() {
    let (rig, first, second) = with_events();
    let saved = rig.server.save().unwrap();

    let mut elsewhere = root_server(UnknownPolicy::Strict).with_service_id(OTHER);
    assert_eq!(elsewhere.restore_beside(&saved, &rig.kernel), Err(FIRST));
    let mut unkerneled = root_server(UnknownPolicy::Strict).with_service_id(ROOT);
    assert_eq!(unkerneled.restore(&saved), Err(SnapshotError::NeedsKernel));
    assert_eq!(state_of(&unkerneled), RootState::new());

    let mut restored = root_server(UnknownPolicy::Strict).with_service_id(ROOT);
    assert_eq!(restored.restore_beside(&saved, &rig.kernel), Ok(()));
    assert_eq!(state_of(&restored), rig.state());
    let mut rig = rig;
    rig.server = restored;
    let again = copied(&rig.send(&[tipc_type(BIND_SYNCPOINT_MEMORY), 0]));
    let objects: Vec<ObjectId> = again
        .iter()
        .map(|handle| {
            rig.kernel
                .resolve_handle(Handle::from_raw(*handle))
                .unwrap()
        })
        .collect();
    assert_eq!(objects, [first, second]);

    // A lone first event the root made restores too, and so does a root with none.
    let lone = RootState {
        syncpoint_events: SyncpointEvents::One(first),
        ..rig.state()
    };
    for state in [lone, RootState::new()] {
        let mut restored = root_server(UnknownPolicy::Strict).with_service_id(ROOT);
        assert_eq!(
            restored.restore_beside(&saved_root(state), &rig.kernel),
            Ok(())
        );
        assert_eq!(state_of(&restored), state);
    }
}

/// A root checks its events before its server numbers (module docs, "Restoring"): a snapshot
/// with both wrong is refused for the event, and the same snapshot with the root's own events
/// for the number.
#[test]
fn the_events_are_checked_before_the_server_numbers() {
    let (mut rig, first, second) = with_events();
    let live = rig.state();
    let [(_, nothing), ..] = not_the_roots(&mut rig.kernel);
    let run = RootState::new().with_gpu_server(GPU);
    let mut wired = wired_with(run);
    let cases = [
        (SyncpointEvents::One(nothing), FIRST),
        (
            SyncpointEvents::Both(first, second),
            SnapshotError::FamilyWiring("its root names another gpu server"),
        ),
    ];
    for (events, refused) in cases {
        let forged = RootState {
            syncpoint_events: events,
            ..live
        };
        assert_eq!(forged.gpu_server(), None);
        assert_eq!(
            wired
                .server
                .restore_beside(&saved_root(forged), &rig.kernel),
            Err(refused),
            "{events:?}"
        );
        assert_eq!(wired.state(), run);
    }
}

/// A kernel, for tests of what a restore refuses before it asks the kernel, that answers every
/// id as a live readable event of every server's, and no id as a shared memory: so only the
/// checks before the kernel's can refuse.
pub(super) struct EveryEvent;

impl RestoredKernel for EveryEvent {
    fn is_service_readable_event(&self, _: ServiceId, _: ObjectId) -> bool {
        true
    }

    fn shared_memory(&self, _: ObjectId) -> Option<starquake_kernel::memory::SharedMemory> {
        None
    }
}
