//! Lock arbitration and priority inheritance, driven through the scheduler with user memory in
//! a map: `ArbitrateLock`'s checks in the reference's order, the hand-over on unlock, waiter
//! order, inheritance along a chain and its undoing, cycles, termination, kernel waiters and
//! the split suspend flags.

use super::*;
use crate::sched::{SuspendType, ThreadState};
use crate::svc::SvcId;
use crate::sync::testing::{NoSignals, World, core, word};

/// The lock the tests use, and a second one.
const LOCK: u64 = 0x8000_1000;
const OTHER_LOCK: u64 = 0x8000_1004;
/// A condition variable for each lock: a condition-variable waiter holds no reference to the
/// owner a signal hands it to, so that owner can still be destroyed while it waits.
const CV: u64 = 0x8000_1010;
const OTHER_CV: u64 = 0x8000_1014;

fn refs(w: &World, index: usize) -> u32 {
    w.scheduler.objects().refs(w.objects[index])
}

#[test]
fn arbitrate_lock_returns_at_once_unless_the_word_names_the_owner_with_the_wait_bit() {
    let mut w = World::new(&[30, 30, 30]);
    let owner = w.thread_handles[0];
    // The owner's handle without the wait bit: the lock was handed on already.
    w.words.set(LOCK, w.tag(0));
    assert_eq!(w.lock_with(1, owner, LOCK), Ok(Arbitrated::Done));
    // Another thread's handle with the bit.
    w.words.set(LOCK, w.tag(2) | HANDLE_WAIT_MASK);
    assert_eq!(w.lock_with(1, owner, LOCK), Ok(Arbitrated::Done));
    // A free lock.
    w.words.set(LOCK, NO_OWNER);
    assert_eq!(w.lock_with(1, owner, LOCK), Ok(Arbitrated::Done));
    assert!(!w.waiting(1));
    assert_eq!(
        refs(&w, 0),
        2,
        "the creator's and the handle's, nothing more"
    );
    assert_eq!(w.scheduler.sync().locks().held(w.threads[0]), &[]);
}

#[test]
fn arbitrate_lock_checks_the_word_then_the_handle_and_changes_nothing_when_it_fails() {
    let mut w = World::new(&[30, 30]);
    let bogus = Handle::from_raw(0x0001_7fff);
    // Both faults: an unreadable word comes first.
    w.words.set(LOCK, bogus.raw() | HANDLE_WAIT_MASK);
    w.words.inaccessible.insert(LOCK);
    assert_eq!(
        w.lock_with(1, bogus, LOCK),
        Err(KernelError::InvalidCurrentMemory.into())
    );
    w.words.inaccessible.clear();
    assert_eq!(
        w.lock_with(1, bogus, LOCK),
        Err(KernelError::InvalidHandle.into())
    );
    // A pseudo-handle is no handle here (`GetObjectWithoutPseudoHandle`).
    w.words
        .set(LOCK, Handle::CURRENT_THREAD.raw() | HANDLE_WAIT_MASK);
    assert_eq!(
        w.lock_with(1, Handle::CURRENT_THREAD, LOCK),
        Err(KernelError::InvalidHandle.into())
    );
    // A handle to something other than a thread.
    let (_, readable) = w.scheduler.create_event().unwrap();
    let event = w.handles.add(w.scheduler.objects_mut(), readable).unwrap();
    w.words.set(LOCK, event.raw() | HANDLE_WAIT_MASK);
    assert_eq!(
        w.lock_with(1, event, LOCK),
        Err(KernelError::InvalidHandle.into())
    );
    assert!(!w.waiting(1));
    assert_eq!(w.scheduler.sync().locks().address_key(w.threads[1]), None);
    assert_eq!(refs(&w, 0), 2);
}

#[test]
fn unlock_hands_the_lock_to_the_most_urgent_waiter_with_the_others() {
    // 0 holds the lock; 1 (30), 2 (10) and 3 (25) wait on it.
    let mut w = World::new(&[20, 30, 10, 25]);
    let base_refs = refs(&w, 0);
    for waiter in [1, 2, 3] {
        assert_eq!(
            w.lock(waiter, 0, LOCK),
            Ok(Arbitrated::Waiting(w.threads[waiter]))
        );
    }
    let owner = w.threads[0];
    let held = w.scheduler.sync().locks().held(owner);
    assert_eq!(held.len(), 1);
    assert_eq!(held[0].key(), AddressKey::user(LOCK));
    assert_eq!(
        held[0].waiters(),
        &[w.threads[2], w.threads[3], w.threads[1]],
        "most urgent first"
    );
    assert_eq!(refs(&w, 0), base_refs + 3, "each wait holds the owner");
    assert_eq!(
        w.priority(0),
        10,
        "the owner inherits the most urgent waiter's"
    );
    assert_eq!(
        w.scheduler.thread(w.threads[1]).unwrap().state(),
        ThreadState::Waiting
    );

    assert_eq!(w.unlock(0, LOCK), Ok(()));
    assert_eq!(w.words.get(LOCK), w.tag(2) | HANDLE_WAIT_MASK);
    assert_eq!(
        w.scheduler.take_wakes(),
        [w.wake(2, SvcId::ArbitrateLock, ResultCode::SUCCESS)]
    );
    assert_eq!(w.x0(2), word(ResultCode::SUCCESS));
    assert_eq!(
        w.scheduler.thread(w.threads[2]).unwrap().state(),
        ThreadState::Runnable
    );
    assert_eq!(refs(&w, 0), base_refs + 2);
    assert_eq!(w.priority(0), 20, "back to its base");
    let locks = w.scheduler.sync().locks();
    assert_eq!(locks.held(owner), &[]);
    assert_eq!(locks.owner_of(w.threads[1]), Some(w.threads[2]));
    assert_eq!(locks.owner_of(w.threads[3]), Some(w.threads[2]));
    assert_eq!(locks.address_key(w.threads[2]), None, "its wait is over");

    // 2 lets go: 3 is next, and 1 still waits.
    assert_eq!(w.unlock(2, LOCK), Ok(()));
    assert_eq!(w.words.get(LOCK), w.tag(3) | HANDLE_WAIT_MASK);
    assert_eq!(w.priority(3), 25, "1 is less urgent: nothing to inherit");
    // 3 lets go: 1 is next, and nobody else waits.
    assert_eq!(w.unlock(3, LOCK), Ok(()));
    assert_eq!(w.words.get(LOCK), w.tag(1));
    assert!(!w.waiting(1));
    // 1 lets go of a lock nobody waits on: the word is cleared.
    assert_eq!(w.unlock(1, LOCK), Ok(()));
    assert_eq!(w.words.get(LOCK), NO_OWNER);
    assert_eq!(refs(&w, 0), base_refs);
    assert_eq!(w.scheduler.sync().locks(), &LockTable::default());
}

#[test]
fn waiters_of_one_priority_get_the_lock_in_the_order_they_came() {
    let mut w = World::new(&[40, 30, 30, 30]);
    for waiter in [2, 1, 3] {
        w.lock(waiter, 0, LOCK).unwrap();
    }
    let held = w.scheduler.sync().locks().held(w.threads[0]);
    assert_eq!(
        held[0].waiters(),
        &[w.threads[2], w.threads[1], w.threads[3]]
    );
}

#[test]
fn unlock_of_an_unwritable_word_fails_and_ends_the_next_owners_wait_with_that_result() {
    let mut w = World::new(&[30, 30]);
    w.lock(1, 0, LOCK).unwrap();
    w.words.read_only.insert(LOCK);
    assert_eq!(
        w.unlock(0, LOCK),
        Err(KernelError::InvalidCurrentMemory.into())
    );
    assert_eq!(
        w.scheduler.take_wakes(),
        [w.wake(
            1,
            SvcId::ArbitrateLock,
            KernelError::InvalidCurrentMemory.code()
        )]
    );
    // A lock nobody waits on: the same failure, and no wake.
    assert_eq!(
        w.unlock(0, LOCK),
        Err(KernelError::InvalidCurrentMemory.into())
    );
    assert_eq!(w.scheduler.take_wakes(), []);
}

#[test]
fn priority_inheritance_across_a_chain_of_three_threads_and_its_undoing() {
    // 0 holds the first lock; 1 holds the second and waits on the first; 2 waits on the second.
    let mut w = World::new(&[40, 30, 10]);
    w.lock(1, 0, LOCK).unwrap();
    assert_eq!([w.priority(0), w.priority(1), w.priority(2)], [30, 30, 10]);
    w.lock(2, 1, OTHER_LOCK).unwrap();
    assert_eq!([w.priority(0), w.priority(1), w.priority(2)], [10, 10, 10]);
    let base = |w: &World, index: usize| {
        w.scheduler
            .thread(w.threads[index])
            .unwrap()
            .base_priority()
            .get()
    };
    assert_eq!([base(&w, 0), base(&w, 1), base(&w, 2)], [40, 30, 10]);

    // The end of the chain becomes less urgent: both owners give the priority back.
    let (t2, caller) = (w.threads[2], core(2));
    let set = |w: &mut World, value: u8| {
        w.scheduler
            .set_base_priority(caller, t2, crate::sched::Priority::new(value).unwrap())
            .unwrap();
    };
    set(&mut w, 50);
    assert_eq!([w.priority(0), w.priority(1), w.priority(2)], [30, 30, 50]);
    // And more urgent again: both inherit.
    set(&mut w, 5);
    assert_eq!([w.priority(0), w.priority(1), w.priority(2)], [5, 5, 5]);

    // The end of the chain stops waiting: 1 and 0 fall back to what is left.
    w.scheduler.terminate_thread(t2).unwrap();
    assert_eq!([w.priority(0), w.priority(1)], [30, 30]);
    assert_eq!(w.scheduler.sync().locks().held(w.threads[1]), &[]);
    // 0 lets go of the first lock: it is back to its own priority, and 1 runs.
    w.unlock(0, LOCK).unwrap();
    assert_eq!([w.priority(0), w.priority(1)], [40, 30]);
    assert_eq!(w.scheduler.sync().locks(), &LockTable::default());
}

#[test]
fn an_owner_that_inherits_takes_its_core_from_a_less_urgent_thread() {
    // 0 (40) holds the lock and runs on core 0; 1 (10) will wait on it.
    let mut w = World::new(&[40, 10]);
    // A thread of priority 20 on core 0 is chosen over the owner.
    let middle = w
        .scheduler
        .create_thread(crate::sched::Priority::new(20).unwrap(), core(0))
        .unwrap();
    w.scheduler.start(middle).unwrap();
    w.update();
    assert_eq!(w.scheduler.highest(core(0)), Ok(Some(middle)));

    // Waiting on the owner lends it priority 10, and core 0 chooses the owner again.
    w.lock(1, 0, LOCK).unwrap();
    w.update();
    assert_eq!(w.scheduler.highest(core(0)), Ok(Some(w.threads[0])));
    assert_eq!(
        w.scheduler.highest(core(1)),
        Ok(None),
        "the waiter left core 1"
    );

    // Letting go gives the priority back: the middle thread is chosen again, and the woken
    // waiter has core 1.
    w.unlock(0, LOCK).unwrap();
    w.update();
    assert_eq!(w.scheduler.highest(core(0)), Ok(Some(middle)));
    assert_eq!(w.scheduler.highest(core(1)), Ok(Some(w.threads[1])));
    assert_eq!(
        w.scheduler.schedule(core(1), &mut NoSignals),
        Ok(Some(w.threads[1]))
    );
}

#[test]
fn a_cycle_of_threads_waiting_on_each_others_locks_settles() {
    let mut w = World::new(&[30, 20]);
    // 0 waits on 1's lock, then 1 waits on 0's: each owner inherits from the other.
    w.lock(0, 1, OTHER_LOCK).unwrap();
    assert_eq!([w.priority(0), w.priority(1)], [30, 20]);
    w.lock(1, 0, LOCK).unwrap();
    assert_eq!([w.priority(0), w.priority(1)], [20, 20]);
    let (t1, caller) = (w.threads[1], core(1));
    w.scheduler
        .set_base_priority(caller, t1, crate::sched::Priority::new(5).unwrap())
        .unwrap();
    assert_eq!([w.priority(0), w.priority(1)], [5, 5]);
    // A less urgent base priority changes nothing: each keeps the other's inherited 5.
    w.scheduler
        .set_base_priority(caller, t1, crate::sched::Priority::new(50).unwrap())
        .unwrap();
    assert_eq!([w.priority(0), w.priority(1)], [5, 5]);
}

#[test]
fn cancelling_a_lock_waiter_only_leaves_a_pending_cancel() {
    let mut w = World::new(&[30, 30]);
    w.lock(1, 0, LOCK).unwrap();
    w.scheduler.cancel_synchronization(w.threads[1]).unwrap();
    assert!(w.waiting(1), "a lock wait is not cancellable");
    assert!(w.scheduler.sync().is_cancel_pending(w.threads[1]));
    assert_eq!(w.scheduler.take_wakes(), []);
}

#[test]
fn a_terminated_lock_waiter_leaves_the_lock_and_releases_the_owner() {
    let mut w = World::new(&[30, 10, 20]);
    let base_refs = refs(&w, 0);
    w.lock(1, 0, LOCK).unwrap();
    w.lock(2, 0, LOCK).unwrap();
    assert_eq!(w.priority(0), 10);
    w.scheduler.terminate_thread(w.threads[1]).unwrap();
    assert_eq!(w.priority(0), 20, "the next waiter's");
    assert_eq!(refs(&w, 0), base_refs + 1);
    let held = w.scheduler.sync().locks().held(w.threads[0]);
    assert_eq!(held[0].waiters(), &[w.threads[2]]);
    assert_eq!(w.scheduler.take_wakes(), [], "termination reports nothing");
}

#[test]
fn kernel_waiters_are_counted_moved_with_the_lock_and_stop_a_suspend() {
    let mut w = World::new(&[30, 20, 25, 30]);
    let [owner, first, second, plain] = [0, 1, 2, 3].map(|index| w.threads[index]);
    let key = AddressKey {
        address: LOCK,
        kernel: true,
    };
    for waiter in [first, second] {
        w.scheduler
            .sync_table_mut()
            .locks
            .set_address_key(waiter, key, 0);
        w.scheduler.add_waiter(None, owner, waiter).unwrap();
    }
    assert_eq!(w.scheduler.sync().locks().kernel_waiters(owner), 2);

    // A thread with kernel waiters notes the request and keeps running.
    w.scheduler
        .request_suspend(owner, SuspendType::Thread)
        .unwrap();
    let row = w.scheduler.thread(owner).unwrap();
    assert!(row.is_suspend_requested(SuspendType::Thread));
    assert_eq!(row.suspend_requested(), SuspendType::Thread.bit());
    assert_eq!(row.suspend_flags(), 0);
    w.update();
    assert_eq!(w.scheduler.highest(core(0)), Ok(Some(owner)));
    // One without suspends at once.
    w.scheduler
        .request_suspend(plain, SuspendType::Thread)
        .unwrap();
    assert_eq!(
        w.scheduler.thread(plain).unwrap().suspend_flags(),
        SuspendType::Thread.bit()
    );
    w.update();
    assert_eq!(w.scheduler.highest(core(3)), Ok(None));

    // Resuming puts the requests (none) into the state either way.
    w.scheduler.resume(owner, SuspendType::Thread).unwrap();
    assert_eq!(w.scheduler.thread(owner).unwrap().suspend_requested(), 0);

    // Letting go moves the lock, and the kernel waiter left, to the next owner.
    let (next, has_waiters) = w.scheduler.remove_waiter_by_key(None, owner, key).unwrap();
    assert_eq!((next, has_waiters), (Some(first), true));
    let locks = w.scheduler.sync().locks();
    assert_eq!(locks.kernel_waiters(owner), 0);
    assert_eq!(locks.kernel_waiters(first), 1);
    assert_eq!(locks.owner_of(second), Some(first));
    w.scheduler.remove_waiter(None, first, second).unwrap();
    assert_eq!(w.scheduler.sync().locks().kernel_waiters(first), 0);

    // With no kernel waiters left, a request suspends.
    w.scheduler
        .request_suspend(first, SuspendType::Thread)
        .unwrap();
    assert_eq!(
        w.scheduler.thread(first).unwrap().suspend_flags(),
        SuspendType::Thread.bit()
    );
}

#[test]
fn a_user_waiter_is_no_kernel_waiter() {
    let mut w = World::new(&[30, 20]);
    w.lock(1, 0, LOCK).unwrap();
    assert_eq!(w.scheduler.sync().locks().kernel_waiters(w.threads[0]), 0);
    w.scheduler
        .request_suspend(w.threads[0], SuspendType::Thread)
        .unwrap();
    assert_eq!(
        w.scheduler.thread(w.threads[0]).unwrap().suspend_flags(),
        SuspendType::Thread.bit()
    );
}

/// Ends thread `index` and closes its object's last two references (its creator's and the
/// handle's), which destroys it (`KThread::Finalize`).
fn destroy(w: &mut World, index: usize) {
    w.scheduler.terminate_thread(w.threads[index]).unwrap();
    let object = w.objects[index];
    w.scheduler.close_object(object).unwrap();
    w.scheduler.close_object(object).unwrap();
}

#[test]
fn a_destroyed_owner_ends_the_waiters_of_its_most_recently_added_lock_first() {
    // 0 (10) and 1 (20) wait on condition variables; 2 (40) holds both locks; 3 signals.
    let mut w = World::new(&[10, 20, 40, 30]);
    w.cv_wait(0, LOCK, CV, crate::sync::Timeout::Never).unwrap();
    w.cv_wait(1, OTHER_LOCK, OTHER_CV, crate::sync::Timeout::Never)
        .unwrap();
    w.words.set(LOCK, w.tag(2));
    w.words.set(OTHER_LOCK, w.tag(2));
    // 0 reaches 2's LOCK first, then 1 its OTHER_LOCK.
    w.cv_signal(3, CV, 1).unwrap();
    w.cv_signal(3, OTHER_CV, 1).unwrap();
    let keys: Vec<_> = w
        .scheduler
        .sync()
        .locks()
        .held(w.threads[2])
        .iter()
        .map(HeldLock::key)
        .collect();
    assert_eq!(
        keys,
        [AddressKey::user(OTHER_LOCK), AddressKey::user(LOCK)],
        "most recently added first (`AddHeldLock`'s `push_front`)"
    );
    destroy(&mut w, 2);
    // The held list in order: 1 (OTHER_LOCK) before 0 (LOCK), though 0 is more urgent.
    let ended = KernelError::InvalidState.code();
    assert_eq!(
        w.scheduler.take_wakes(),
        [
            w.wake(1, SvcId::WaitProcessWideKeyAtomic, ended),
            w.wake(0, SvcId::WaitProcessWideKeyAtomic, ended),
        ]
    );
    assert_eq!(w.scheduler.sync().locks(), &LockTable::default());
}

#[test]
fn a_lock_handed_on_goes_to_the_front_of_the_new_owners_list() {
    // 0 (10) holds OTHER_LOCK, which 3 (20) waits on through a condition variable. 1 (40)
    // holds LOCK, which 0 and then 2 (30) wait on; 4 signals.
    let mut w = World::new(&[10, 40, 30, 20, 30]);
    w.cv_wait(3, OTHER_LOCK, OTHER_CV, crate::sync::Timeout::Never)
        .unwrap();
    w.words.set(OTHER_LOCK, w.tag(0));
    w.cv_signal(4, OTHER_CV, 1).unwrap();
    w.lock(0, 1, LOCK).unwrap();
    w.lock(2, 1, LOCK).unwrap();
    // 1 lets go: 0 gets LOCK, and 2 still waits on it, now on 0.
    w.unlock(1, LOCK).unwrap();
    assert_eq!(
        w.scheduler.take_wakes(),
        [w.wake(0, SvcId::ArbitrateLock, ResultCode::SUCCESS)]
    );
    let keys: Vec<_> = w
        .scheduler
        .sync()
        .locks()
        .held(w.threads[0])
        .iter()
        .map(HeldLock::key)
        .collect();
    assert_eq!(
        keys,
        [AddressKey::user(LOCK), AddressKey::user(OTHER_LOCK)],
        "the lock handed on is added at the front (`RemoveWaiterByKey`'s `AddHeldLock`)"
    );
    destroy(&mut w, 0);
    let ended = KernelError::InvalidState.code();
    assert_eq!(
        w.scheduler.take_wakes(),
        [
            w.wake(2, SvcId::ArbitrateLock, ended),
            w.wake(3, SvcId::WaitProcessWideKeyAtomic, ended),
        ]
    );
    assert_eq!(w.scheduler.sync().locks(), &LockTable::default());
}

#[test]
fn a_destroyed_owner_keeps_no_count_of_kernel_waiters() {
    // 1 waits on an address, and is made a kernel waiter on 0's lock by hand: nothing in
    // starquake takes a kernel light lock yet (the module docs), so only a test reaches this.
    // Eden asserts it never happens (`Finalize`'s `ASSERT(m_num_kernel_waiters == 0)`); the
    // count's row must not outlive the thread either way.
    let mut w = World::new(&[30, 20]);
    w.words.set(OTHER_LOCK, 0);
    w.wait_address(
        1,
        OTHER_LOCK,
        crate::sync::ArbitrationType::WaitIfEqual,
        0,
        crate::sync::Timeout::Never,
    )
    .unwrap();
    let (owner, waiter) = (w.threads[0], w.threads[1]);
    let key = AddressKey {
        address: LOCK,
        kernel: true,
    };
    w.scheduler
        .sync_table_mut()
        .locks
        .set_address_key(waiter, key, 0);
    w.scheduler.add_waiter(None, owner, waiter).unwrap();
    assert_eq!(w.scheduler.sync().locks().kernel_waiters(owner), 1);
    destroy(&mut w, 0);
    assert_eq!(
        w.scheduler.take_wakes(),
        [w.wake(1, SvcId::WaitForAddress, KernelError::InvalidState.code())]
    );
    assert_eq!(w.scheduler.sync().locks().kernel_waiters(owner), 0);
    assert_eq!(w.scheduler.sync().locks(), &LockTable::default());
}
