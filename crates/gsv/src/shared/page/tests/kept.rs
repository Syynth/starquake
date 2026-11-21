//! The root's own references to the page's two transfer memories (module docs, "Binding (204)"
//! and "When the root closes"): 204 keeps one of each, a failing 204 gives back what it kept,
//! and the drain drops both once the root's last session has closed and none is open. Over the
//! services framework's fake kernel, whose count of the keeps is the kernel's rule
//! (`FakeKernel::kept_transfer_memory`); the same on the kernel's own dispatcher is in
//! `on_the_kernel.rs`.

use super::*;

/// The keeps the root holds on the rig's writable and read-only memories.
fn kept(rig: &Rig) -> [u32; 2] {
    [
        rig.kernel.kept_transfer_memory(ROOT, rig.writable),
        rig.kernel.kept_transfer_memory(ROOT, rig.read_only),
    ]
}

/// What the drain asks the host when the root has closed with the page bound and no root
/// session open: the count, then each memory's release, the writable one first.
fn let_go(rig: &Rig) -> [Act; 3] {
    [
        Act::RootSessions,
        Act::Release(rig.writable),
        Act::Release(rig.read_only),
    ]
}

#[test]
fn bind_keeps_one_reference_of_each_memory_for_the_root() {
    let mut rig = Rig::new();
    assert_eq!(kept(&rig), [0, 0]);
    let reply = rig.bind();
    assert_eq!(reply, success(BIND_SYNCPOINT_MEMORY, &copied(&reply), &[]));
    assert_eq!(kept(&rig), [1, 1]);
    // The root's, not gsv's recording service's.
    assert_eq!(rig.kernel.kept_transfer_memory(GSV, rig.writable), 0);
    assert_eq!(rig.kernel.kept_transfer_memory(GSV, rig.read_only), 0);
}

#[test]
fn a_refused_bind_keeps_nothing() {
    // A second 204 is refused before it keeps anything: the first's keeps stay one each.
    let mut rig = Rig::bound();
    assert_eq!(
        rig.bind(),
        result_only(BIND_SYNCPOINT_MEMORY, KernelError::InvalidState.code())
    );
    assert_eq!(kept(&rig), [1, 1]);
    // A refused event comes before the keeps.
    let mut rig = Rig::new();
    rig.kernel.refuse_objects(Some(KernelError::OutOfResource));
    rig.bind();
    assert_eq!(kept(&rig), [0, 0]);
}

#[test]
fn a_refused_keep_gives_back_what_the_bind_kept_and_the_next_bind_succeeds() {
    // The fake lets through two object calls (the request event, the answer event) and then
    // the first keep, or not.
    for (allowed, which) in [(2, "the writable memory's"), (3, "the read-only memory's")] {
        let mut rig = Rig::new();
        rig.kernel
            .refuse_objects_after(allowed, Some(KernelError::InvalidState));
        assert_eq!(
            rig.bind(),
            result_only(BIND_SYNCPOINT_MEMORY, KernelError::InvalidState.code()),
            "{which}"
        );
        assert_eq!(kept(&rig), [0, 0], "{which}");
        // Nothing is bound and neither memory was written; the events are kept for the next
        // 204.
        let page = rig.page();
        assert_eq!(page.binding(), None, "{which}");
        assert!(page.request_event().is_some(), "{which}");
        assert_eq!(rig.kernel.memory(WRITABLE, 1), Some(&[DIRT][..]), "{which}");
        assert_eq!(
            rig.kernel.memory(READ_ONLY, 1),
            Some(&[DIRT][..]),
            "{which}"
        );
        rig.kernel.refuse_objects(None);
        let reply = rig.bind();
        assert_eq!(
            reply,
            success(BIND_SYNCPOINT_MEMORY, &copied(&reply), &[]),
            "{which}"
        );
        assert_eq!(kept(&rig), [1, 1], "{which}");
    }
}

#[test]
fn a_refused_write_gives_back_both_references_and_binds_nothing() {
    // A read-only memory whose last byte is not mapped: clearing it is refused, after the
    // writable memory was cleared.
    const SHORT: u64 = 0x1_0020_0000;
    let mut rig = Rig::new();
    rig.kernel.map(SHORT, vec![DIRT; len(READ_ONLY_SIZE) - 1]);
    let short = rig
        .kernel
        .lend(SHORT, u64::from(READ_ONLY_SIZE), SvcPermission::Read);
    let handle = rig.kernel.insert_handle(short).unwrap().raw();
    let reply = rig.send(&bind_words(rig.handles[0], handle));
    assert_eq!(
        reply,
        result_only(
            BIND_SYNCPOINT_MEMORY,
            KernelError::InvalidCurrentMemory.code()
        )
    );
    assert_eq!(rig.kernel.kept_transfer_memory(ROOT, rig.writable), 0);
    assert_eq!(rig.kernel.kept_transfer_memory(ROOT, short), 0);
    assert_eq!(rig.page().binding(), None);
    // The bytes written before the refusal stay written.
    assert_eq!(rig.kernel.memory(WRITABLE, 1), Some(&[0][..]));
    assert_eq!(rig.kernel.memory(SHORT, 1), Some(&[DIRT][..]));
    // The title's good memories then bind.
    let reply = rig.bind();
    assert_eq!(reply, success(BIND_SYNCPOINT_MEMORY, &copied(&reply), &[]));
    assert_eq!(kept(&rig), [1, 1]);
}

#[test]
fn closing_the_root_s_last_session_lets_the_page_and_its_memories_go() {
    let mut rig = shown();
    // A held request, watching its syncpoint.
    rig.ask(&[SP]);
    rig.signal();
    rig.turn(T0);
    assert!(rig.page().held().is_some());
    assert_eq!(rig.watchers(SP), 1);
    let events = (rig.page().request_event(), rig.page().answer_event());
    // The title closes its handles, then its root session.
    assert!(rig.kernel.end_transfer_memory(rig.writable));
    assert!(rig.kernel.end_transfer_memory(rig.read_only));
    rig.server.close(rig.session).unwrap();
    assert!(!rig.shared.queue().is_empty());
    assert_eq!(rig.turn(T0), let_go(&rig));
    // Both references went, and with them the transfer memories, which the guest no longer held.
    assert_eq!(kept(&rig), [0, 0]);
    assert_eq!(
        Kernel::transfer_memory(&rig.kernel, rig.writable).map(|_| ()),
        Err(KernelError::InvalidHandle)
    );
    assert_eq!(
        Kernel::transfer_memory(&rig.kernel, rig.read_only).map(|_| ()),
        Err(KernelError::InvalidHandle)
    );
    // The page is unbound, as before 204, but for its events, which stay for the next 204.
    let page = rig.page();
    assert_eq!(page.binding(), None);
    assert_eq!(page.held(), None);
    assert!(!page.is_signalled());
    assert_eq!(page.shown(id(SP)), UNALLOCATED);
    assert_eq!((page.request_event(), page.answer_event()), events);
    assert_eq!(rig.watchers(SP), 0);
    assert_eq!(rig.shared.save_page().map(|_| ()), Ok(()));
    // The next turn has nothing to do.
    assert_eq!(rig.turn(T0), []);
}

#[test]
fn a_signal_waiting_when_the_root_closes_is_dropped() {
    let mut rig = Rig::bound();
    rig.ask(&[SP]);
    rig.signal();
    let request = rig.request_event();
    let mut host = PageHost::new(&mut rig.kernel);
    rig.shared.drain(&mut host);
    drop(host);
    assert!(rig.page().is_signalled());
    rig.server.close(rig.session).unwrap();
    // The page turn after the drain finds the page unbound and answers nothing.
    assert_eq!(rig.turn(T0), let_go(&rig));
    assert!(!rig.page().is_signalled());
    assert_eq!(rig.answers(), 0);
    // A later signal of the old request event is not a request.
    rig.shared
        .queue()
        .push(Completion::GuestSignalled { event: request });
    assert_eq!(rig.turn(T0), []);
    assert!(!rig.page().is_signalled());
}

#[test]
fn closing_a_root_session_while_another_is_open_queues_nothing() {
    let mut rig = Rig::bound();
    let second = rig.kernel.new_object();
    rig.server.accept(second).unwrap();
    rig.server.close(rig.session).unwrap();
    assert!(rig.shared.queue().is_empty());
    assert_eq!(rig.turn(T0), []);
    assert_eq!(kept(&rig), [1, 1]);
    // The last one's close is the root's.
    rig.server.close(second).unwrap();
    assert!(!rig.shared.queue().is_empty());
    assert_eq!(rig.turn(T0), let_go(&rig));
    assert_eq!(kept(&rig), [0, 0]);
}

#[test]
fn the_page_stays_bound_while_the_kernel_says_a_root_session_is_open() {
    // The root closed, and a new root session opened before the drain: the page is that
    // session's to use.
    let mut rig = Rig::bound();
    rig.server.close(rig.session).unwrap();
    rig.root_sessions = 1;
    assert_eq!(rig.turn(T0), [Act::RootSessions]);
    assert!(rig.page().binding().is_some());
    assert_eq!(kept(&rig), [1, 1]);
    // The close was acted on: a later turn asks nothing.
    rig.root_sessions = 0;
    assert_eq!(rig.turn(T0), []);
    assert!(rig.page().binding().is_some());
}

#[test]
fn a_host_that_cannot_count_root_sessions_leaves_the_page_bound() {
    let mut rig = Rig::bound();
    rig.server.close(rig.session).unwrap();
    let mut host = Bare(PageHost::new(&mut rig.kernel));
    rig.shared.drain(&mut host);
    assert_eq!(host.0.acts, []);
    assert_eq!(
        host.release_kept_transfer_memory(rig.writable),
        Err(KernelError::NotImplemented)
    );
    assert_eq!(host.root_sessions(), Err(KernelError::NotImplemented));
    drop(host);
    assert!(rig.page().binding().is_some());
    assert_eq!(kept(&rig), [1, 1]);
}

#[test]
fn an_unbound_page_ignores_the_root_s_close() {
    let mut rig = Rig::new();
    rig.server.close(rig.session).unwrap();
    assert!(!rig.shared.queue().is_empty());
    // Not even the count is asked.
    assert_eq!(rig.turn(T0), []);
    assert_eq!(rig.page(), Page::default());
}

#[test]
fn a_refused_release_is_logged_and_the_page_is_unbound_either_way() {
    let mut rig = Rig::bound();
    // The writable memory's keep is gone behind gsv's back: its release is refused, and the
    // read-only memory's still made.
    rig.kernel.release_kept(ROOT, rig.writable).unwrap();
    rig.server.close(rig.session).unwrap();
    assert_eq!(rig.turn(T0), let_go(&rig));
    assert_eq!(kept(&rig), [0, 0]);
    assert_eq!(rig.page().binding(), None);
}

#[test]
fn after_the_root_closes_a_new_session_binds_again_with_the_same_events() {
    let mut rig = Rig::bound();
    let events = (rig.page().request_event(), rig.page().answer_event());
    rig.server.close(rig.session).unwrap();
    rig.turn(T0);
    assert_eq!(kept(&rig), [0, 0]);
    // The title still holds its handles, so the memories are live, and a new root session's
    // 204 binds them again.
    let second = rig.kernel.new_object();
    rig.server.accept(second).unwrap();
    rig.session = second;
    let reply = rig.bind();
    assert_eq!(reply, success(BIND_SYNCPOINT_MEMORY, &copied(&reply), &[]));
    assert_eq!(kept(&rig), [1, 1]);
    assert_eq!(
        (rig.page().request_event(), rig.page().answer_event()),
        events
    );
}

#[test]
fn a_restored_page_must_name_memories_the_root_keeps_exactly_once() {
    let rig = Rig::bound();
    let page = rig.page();
    let binding = page.binding().unwrap();
    let bytes = snapshot_of(&page);
    for (memory, count, error) in [
        (binding.writable, 0, PageSnapshotError::WritableNotKept),
        (binding.writable, 2, PageSnapshotError::WritableNotKept),
        (binding.read_only, 0, PageSnapshotError::ReadOnlyNotKept),
        (binding.read_only, 2, PageSnapshotError::ReadOnlyNotKept),
    ] {
        let mut kernel = Holding::of(&page);
        kernel.kept.insert((ROOT, memory), count);
        let restored = fresh_shared();
        assert_eq!(
            restored.restore_page(&bytes, &kernel, ROOT),
            Err(error.clone()),
            "{memory:?} kept {count}"
        );
        assert_eq!(restored.lock().page(), &Page::default());
        assert!(kernel.attached_nothing());
    }
    // Kept by another service is not kept by the root.
    let mut kernel = Holding::of(&page);
    kernel.kept.remove(&(ROOT, binding.writable));
    kernel.kept.insert((GSV, binding.writable), 1);
    assert_eq!(
        fresh_shared().restore_page(&bytes, &kernel, ROOT),
        Err(PageSnapshotError::WritableNotKept)
    );
    // Both unkept: the writable memory's is reported.
    let mut kernel = Holding::of(&page);
    kernel.kept.clear();
    assert_eq!(
        fresh_shared().restore_page(&bytes, &kernel, ROOT),
        Err(PageSnapshotError::WritableNotKept)
    );
    // The writable memory unkept and the read-only one gone: the writable memory's first.
    let mut kernel = Holding::of(&page);
    kernel.kept.remove(&(ROOT, binding.writable));
    kernel.memories.remove(&binding.read_only);
    assert_eq!(
        fresh_shared().restore_page(&bytes, &kernel, ROOT),
        Err(PageSnapshotError::WritableNotKept)
    );
    // The read-only memory the wrong size and unkept: the size first.
    let mut kernel = Holding::of(&page);
    kernel.kept.remove(&(ROOT, binding.read_only));
    kernel
        .memories
        .insert(binding.read_only, u64::from(WRITABLE_SIZE));
    assert_eq!(
        fresh_shared().restore_page(&bytes, &kernel, ROOT),
        Err(PageSnapshotError::ReadOnlySize {
            size: u64::from(WRITABLE_SIZE)
        })
    );
    // As 204 left it, it is taken.
    fresh_shared()
        .restore_page(&bytes, &Holding::of(&page), ROOT)
        .unwrap();
}

#[test]
fn the_not_kept_refusals_read_as_plain_words() {
    assert_eq!(
        PageSnapshotError::WritableNotKept.to_string(),
        "the root does not keep exactly one reference to the syncpoint page's writable memory"
    );
    assert_eq!(
        PageSnapshotError::ReadOnlyNotKept.to_string(),
        "the root does not keep exactly one reference to the syncpoint page's read-only memory"
    );
}
