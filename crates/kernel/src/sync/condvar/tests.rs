//! Condition variables, driven through the scheduler with user memory in a map: letting go of
//! the lock and flagging the key, the order of the wait's steps, every outcome of taking the
//! lock on a signal, counts, timers in both halves of the wait, and an owner destroyed while a
//! waiter waits on it.

use starquake_trace::HostNanos;

use super::*;
use crate::sched::ThreadState;
use crate::svc::SvcId;
use crate::sync::LockTable;
use crate::sync::testing::{World, word};

const LOCK: u64 = 0x8000_3000;
const CV: u64 = 0x8000_3010;
const NOW: u64 = 1_000;
const AFTER: i64 = 500;

fn cv_waiters(w: &World, cv_key: u64) -> Vec<ThreadId> {
    let key = ProcessKey {
        process: None,
        address: cv_key,
    };
    w.scheduler.sync().condition_variables().waiters(key).to_vec()
}

#[test]
fn waiting_hands_the_lock_to_its_most_urgent_waiter_and_flags_the_key() {
    // 0 holds the lock and waits on the condition variable; 1 (20) and 2 (25) wait on the lock.
    let mut w = World::new(&[30, 20, 25]);
    w.lock(1, 0, LOCK).unwrap();
    w.lock(2, 0, LOCK).unwrap();
    assert_eq!(w.priority(0), 20);
    assert_eq!(
        w.cv_wait(0, LOCK, CV, Timeout::Never),
        Ok(Arbitrated::Waiting(w.threads[0]))
    );
    assert_eq!(w.words.get(LOCK), w.tag(1) | HANDLE_WAIT_MASK);
    assert_eq!(w.words.get(CV), HAS_WAITER_FLAG);
    assert_eq!(
        w.scheduler.take_wakes(),
        [w.wake(1, SvcId::ArbitrateLock, ResultCode::SUCCESS)]
    );
    assert_eq!(w.priority(0), 30, "the inherited priority went with the lock");
    assert_eq!(cv_waiters(&w, CV), [w.threads[0]]);
    assert_eq!(
        w.scheduler.thread(w.threads[0]).unwrap().state(),
        ThreadState::Waiting
    );
    let locks = w.scheduler.sync().locks();
    assert_eq!(
        locks.address_key(w.threads[0]),
        Some((AddressKey::user(LOCK), w.tag(0)))
    );
    assert_eq!(locks.owner_of(w.threads[2]), Some(w.threads[1]));
    assert_eq!(locks.owner_of(w.threads[0]), None, "in the tree, on no lock");
}

#[test]
fn waiting_on_a_lock_nobody_waits_for_clears_the_lock_word() {
    let mut w = World::new(&[30]);
    w.words.set(LOCK, w.tag(0));
    w.cv_wait(0, LOCK, CV, Timeout::Never).unwrap();
    assert_eq!(w.words.get(LOCK), NO_OWNER);
    assert_eq!(w.words.get(CV), HAS_WAITER_FLAG);
}

#[test]
fn a_zero_timeout_lets_go_of_the_lock_and_flags_the_key_before_it_times_out() {
    let mut w = World::new(&[30, 20]);
    w.lock(1, 0, LOCK).unwrap();
    assert_eq!(
        w.cv_wait(0, LOCK, CV, Timeout::Poll),
        Err(KernelError::TimedOut.into())
    );
    assert_eq!(w.words.get(LOCK), w.tag(1));
    assert_eq!(w.words.get(CV), HAS_WAITER_FLAG);
    assert!(!w.waiting(0));
    assert!(!w.waiting(1));
    assert_eq!(cv_waiters(&w, CV), []);
    assert_eq!(w.scheduler.sync().locks().address_key(w.threads[0]), None);
}

#[test]
fn an_unwritable_lock_word_fails_after_the_next_owner_is_woken_and_the_key_ignores_failure() {
    let mut w = World::new(&[30, 20]);
    w.lock(1, 0, LOCK).unwrap();
    w.words.read_only.insert(LOCK);
    // Both faults: the zero timeout is reported after the write.
    assert_eq!(
        w.cv_wait(0, LOCK, CV, Timeout::Poll),
        Err(KernelError::InvalidCurrentMemory.into())
    );
    assert_eq!(
        w.scheduler.take_wakes(),
        [w.wake(1, SvcId::ArbitrateLock, ResultCode::SUCCESS)]
    );
    assert_eq!(w.words.get(CV), HAS_WAITER_FLAG);
    assert!(!w.waiting(0));

    // A key the guest may not write is no failure.
    w.words.read_only.clear();
    w.words.inaccessible.insert(CV);
    assert_eq!(
        w.cv_wait(0, LOCK, CV, Timeout::Never),
        Ok(Arbitrated::Waiting(w.threads[0]))
    );
}

#[test]
fn a_signal_takes_a_free_lock_for_the_waiter_and_clears_the_key_when_nobody_is_left() {
    let mut w = World::new(&[30, 30]);
    w.cv_wait(0, LOCK, CV, Timeout::Never).unwrap();
    assert_eq!(w.cv_signal(1, CV, 1), Ok(()));
    assert_eq!(w.words.get(LOCK), w.tag(0));
    assert_eq!(w.words.get(CV), HAS_NO_WAITER_FLAG);
    assert_eq!(
        w.scheduler.take_wakes(),
        [w.wake(0, SvcId::WaitProcessWideKeyAtomic, ResultCode::SUCCESS)]
    );
    assert_eq!(w.x0(0), word(ResultCode::SUCCESS));
    assert_eq!(cv_waiters(&w, CV), []);
    assert_eq!(w.scheduler.sync().locks(), &LockTable::default());
}

#[test]
fn a_signal_with_the_lock_held_hands_the_waiter_to_the_owner_until_it_lets_go() {
    // 0 (10) waits on the condition variable; 1 (40) holds the lock; 2 signals.
    let mut w = World::new(&[10, 40, 30]);
    w.cv_wait(0, LOCK, CV, Timeout::Never).unwrap();
    w.words.set(LOCK, w.tag(1));
    w.cv_signal(2, CV, 1).unwrap();
    assert_eq!(w.words.get(LOCK), w.tag(1) | HANDLE_WAIT_MASK);
    assert_eq!(w.scheduler.take_wakes(), [], "still waiting, now for the lock");
    assert!(w.waiting(0));
    assert_eq!(w.scheduler.sync().locks().owner_of(w.threads[0]), Some(w.threads[1]));
    assert_eq!(w.priority(1), 10, "the owner inherits the waiter's priority");
    assert_eq!(cv_waiters(&w, CV), []);

    w.unlock(1, LOCK).unwrap();
    assert_eq!(w.words.get(LOCK), w.tag(0));
    assert_eq!(
        w.scheduler.take_wakes(),
        [w.wake(0, SvcId::WaitProcessWideKeyAtomic, ResultCode::SUCCESS)],
        "the waiter returns from the call it made"
    );
    assert_eq!(w.priority(1), 40);
}

#[test]
fn a_lock_word_naming_no_thread_ends_the_wait_with_invalid_state() {
    let mut w = World::new(&[30, 30]);
    let (_, readable) = w.scheduler.create_event().unwrap();
    let event = w.handles.add(w.scheduler.objects_mut(), readable).unwrap();
    for owner in [0x0001_7fff, Handle::CURRENT_THREAD.raw(), event.raw()] {
        w.cv_wait(0, LOCK, CV, Timeout::Never).unwrap();
        w.words.set(LOCK, owner);
        w.cv_signal(1, CV, 1).unwrap();
        assert_eq!(
            w.scheduler.take_wakes(),
            [w.wake(0, SvcId::WaitProcessWideKeyAtomic, KernelError::InvalidState.code())],
            "{owner:#x}"
        );
        assert_eq!(w.words.get(LOCK), owner | HANDLE_WAIT_MASK);
    }
}

#[test]
fn a_lock_word_the_guest_may_not_access_atomically_ends_the_wait_with_invalid_current_memory() {
    for read_only in [true, false] {
        let mut w = World::new(&[30, 30]);
        w.cv_wait(0, LOCK, CV, Timeout::Never).unwrap();
        if read_only {
            w.words.read_only.insert(LOCK);
        } else {
            w.words.inaccessible.insert(LOCK);
        }
        w.cv_signal(1, CV, 1).unwrap();
        assert_eq!(
            w.scheduler.take_wakes(),
            [w.wake(
                0,
                SvcId::WaitProcessWideKeyAtomic,
                KernelError::InvalidCurrentMemory.code()
            )]
        );
        assert_eq!(w.words.get(LOCK), NO_OWNER);
    }
}

#[test]
fn a_signal_takes_count_waiters_most_urgent_first_and_flags_the_key_only_when_none_are_left() {
    for (count, woken) in [(1, 1), (2, 2), (3, 3), (4, 3), (0, 3), (-1, 3)] {
        // 0 signals; 1 (30), 2 (10) and 3 (20) wait, each on its own lock.
        let mut w = World::new(&[40, 30, 10, 20]);
        for waiter in [1, 2, 3] {
            let lock = LOCK + 4 * u64::try_from(waiter).unwrap();
            w.cv_wait(waiter, lock, CV, Timeout::Never).unwrap();
        }
        assert_eq!(cv_waiters(&w, CV), [w.threads[2], w.threads[3], w.threads[1]]);
        w.cv_signal(0, CV, count).unwrap();
        let order = [2, 3, 1];
        let expected: Vec<_> = order[..woken]
            .iter()
            .map(|&index| w.wake(index, SvcId::WaitProcessWideKeyAtomic, ResultCode::SUCCESS))
            .collect();
        assert_eq!(w.scheduler.take_wakes(), expected, "count {count}");
        let flag = if woken == 3 {
            HAS_NO_WAITER_FLAG
        } else {
            HAS_WAITER_FLAG
        };
        assert_eq!(w.words.get(CV), flag, "count {count}");
    }
}

#[test]
fn a_signal_before_the_wait_clears_the_key_and_wakes_nothing() {
    let mut w = World::new(&[30, 30]);
    w.words.set(CV, HAS_WAITER_FLAG);
    w.cv_signal(1, CV, 0).unwrap();
    assert_eq!(w.words.get(CV), HAS_NO_WAITER_FLAG);
    assert_eq!(w.scheduler.take_wakes(), []);
    assert_eq!(
        w.cv_wait(0, LOCK, CV, Timeout::Never),
        Ok(Arbitrated::Waiting(w.threads[0]))
    );
    // A signal on another key leaves the waiter and flags only its own key.
    w.cv_signal(1, CV + 4, 0).unwrap();
    assert!(w.waiting(0));
    assert_eq!(w.words.get(CV), HAS_WAITER_FLAG);
}

#[test]
fn a_timer_ends_the_wait_in_the_tree_and_while_it_waits_on_the_owner() {
    let timeout = Timeout::from_nanos(HostNanos(NOW), AFTER);
    let Timeout::At(HostNanos(deadline)) = timeout else {
        panic!("a positive timeout has a deadline");
    };

    // In the tree.
    let mut w = World::new(&[10, 40, 30]);
    w.cv_wait(0, LOCK, CV, timeout).unwrap();
    assert_eq!(w.fire(deadline), 1);
    assert_eq!(
        w.scheduler.take_wakes(),
        [w.wake(0, SvcId::WaitProcessWideKeyAtomic, KernelError::TimedOut.code())]
    );
    assert_eq!(cv_waiters(&w, CV), []);

    // Handed to an owner: the timer still runs, and the owner gives the priority back.
    w.cv_wait(0, LOCK, CV, Timeout::from_nanos(HostNanos(deadline), AFTER))
        .unwrap();
    w.words.set(LOCK, w.tag(1));
    w.cv_signal(2, CV, 1).unwrap();
    assert_eq!(w.priority(1), 10);
    assert_eq!(w.fire(deadline * 2), 1);
    assert_eq!(
        w.scheduler.take_wakes(),
        [w.wake(0, SvcId::WaitProcessWideKeyAtomic, KernelError::TimedOut.code())]
    );
    assert_eq!(w.priority(1), 40);
    assert_eq!(w.scheduler.sync().locks(), &LockTable::default());
}

#[test]
fn destroying_an_owner_ends_its_waiters_waits_with_invalid_state() {
    let mut w = World::new(&[10, 40, 30]);
    w.cv_wait(0, LOCK, CV, Timeout::Never).unwrap();
    w.words.set(LOCK, w.tag(1));
    w.cv_signal(2, CV, 1).unwrap();
    assert_eq!(w.scheduler.sync().locks().owner_of(w.threads[0]), Some(w.threads[1]));
    // The owner ends, and its object's last two references (its creator's and the handle's)
    // go: it is destroyed, and the waiter's wait ends.
    w.scheduler.terminate_thread(w.threads[1]).unwrap();
    assert_eq!(w.scheduler.take_wakes(), [], "ending is not destroying");
    let object = w.objects[1];
    w.scheduler.close_object(object).unwrap();
    w.scheduler.close_object(object).unwrap();
    assert_eq!(
        w.scheduler.take_wakes(),
        [w.wake(0, SvcId::WaitProcessWideKeyAtomic, KernelError::InvalidState.code())]
    );
    assert_eq!(w.scheduler.sync().locks(), &LockTable::default());
}

#[test]
fn cancelling_a_condition_variable_waiter_only_leaves_a_pending_cancel() {
    let mut w = World::new(&[30]);
    w.cv_wait(0, LOCK, CV, Timeout::Never).unwrap();
    w.scheduler.cancel_synchronization(w.threads[0]).unwrap();
    assert!(w.waiting(0));
    assert!(w.scheduler.sync().is_cancel_pending(w.threads[0]));
}
