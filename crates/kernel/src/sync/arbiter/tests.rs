//! The address arbiter, driven through the scheduler with user memory in a map: every
//! arbitration type against every comparison, every signal type, the order of checks, counts,
//! timeouts, a signal before the wait and a waiter's place after its priority changes.

use starquake_trace::HostNanos;

use super::*;
use crate::sched::ThreadState;
use crate::svc::SvcId;
use crate::sync::TIMEOUT_SLACK_NS;
use crate::sync::testing::{World, core, word};

const AT: u64 = 0x8000_2000;
const VALUE: i32 = 7;
/// A timeout that never expires in these tests, and one that does.
const NEVER: Timeout = Timeout::Never;
const NOW: u64 = 1_000;
const AFTER: i64 = 500;

fn set(w: &World, value: i32) {
    w.words.set(AT, value.cast_unsigned());
}

fn get(w: &World) -> i32 {
    w.words.get(AT).cast_signed()
}

fn waiting_on(w: &World) -> Vec<ThreadId> {
    let key = ProcessKey {
        process: None,
        address: AT,
    };
    w.scheduler.sync().arbiter().waiters(key).to_vec()
}

#[test]
fn types_decode_from_their_values_and_nothing_else() {
    for kind in [
        ArbitrationType::WaitIfLessThan,
        ArbitrationType::DecrementAndWaitIfLessThan,
        ArbitrationType::WaitIfEqual,
    ] {
        assert_eq!(ArbitrationType::from_raw(kind.raw()), Some(kind));
    }
    for kind in [
        SignalType::Signal,
        SignalType::SignalAndIncrementIfEqual,
        SignalType::SignalAndModifyByWaitingCountIfEqual,
    ] {
        assert_eq!(SignalType::from_raw(kind.raw()), Some(kind));
    }
    // Eden's values (`svc_types.h`): 0, 1, 2; the first value past them is invalid.
    assert_eq!(ArbitrationType::WaitIfEqual.raw(), 2);
    assert_eq!(SignalType::SignalAndModifyByWaitingCountIfEqual.raw(), 2);
    for raw in [3, u32::MAX] {
        assert_eq!(ArbitrationType::from_raw(raw), None);
        assert_eq!(SignalType::from_raw(raw), None);
    }
}

#[test]
fn every_arbitration_type_waits_only_when_its_comparison_holds() {
    use ArbitrationType::{DecrementAndWaitIfLessThan, WaitIfEqual, WaitIfLessThan};
    // (type, word, waits, word afterwards)
    let cases = [
        (WaitIfLessThan, VALUE - 1, true, VALUE - 1),
        (WaitIfLessThan, VALUE, false, VALUE),
        (WaitIfLessThan, VALUE + 1, false, VALUE + 1),
        (DecrementAndWaitIfLessThan, VALUE - 1, true, VALUE - 2),
        (DecrementAndWaitIfLessThan, VALUE, false, VALUE),
        (DecrementAndWaitIfLessThan, VALUE + 1, false, VALUE + 1),
        (WaitIfEqual, VALUE - 1, false, VALUE - 1),
        (WaitIfEqual, VALUE, true, VALUE),
        (WaitIfEqual, VALUE + 1, false, VALUE + 1),
    ];
    for (kind, before, waits, after) in cases {
        let mut w = World::new(&[30]);
        set(&w, before);
        let result = w.wait_address(0, AT, kind, VALUE, NEVER);
        if waits {
            assert_eq!(result, Ok(Arbitrated::Waiting(w.threads[0])), "{kind:?} {before}");
            assert_eq!(waiting_on(&w), [w.threads[0]]);
            assert_eq!(
                w.scheduler.thread(w.threads[0]).unwrap().state(),
                ThreadState::Waiting
            );
        } else {
            assert_eq!(result, Err(KernelError::InvalidState.into()), "{kind:?} {before}");
            assert!(!w.waiting(0));
        }
        assert_eq!(get(&w), after, "{kind:?} {before}");
    }
}

#[test]
fn a_zero_timeout_times_out_after_the_comparison_and_keeps_the_decrement() {
    let mut w = World::new(&[30]);
    // Both faults: the comparison is reported first.
    set(&w, VALUE);
    assert_eq!(
        w.wait_address(0, AT, ArbitrationType::WaitIfLessThan, VALUE, Timeout::Poll),
        Err(KernelError::InvalidState.into())
    );
    set(&w, VALUE - 1);
    assert_eq!(
        w.wait_address(
            0,
            AT,
            ArbitrationType::DecrementAndWaitIfLessThan,
            VALUE,
            Timeout::Poll
        ),
        Err(KernelError::TimedOut.into())
    );
    assert_eq!(get(&w), VALUE - 2);
    assert!(!w.waiting(0));
}

#[test]
fn a_word_the_guest_may_not_access_is_invalid_current_memory_before_the_comparison() {
    let mut w = World::new(&[30]);
    // The comparison would fail too, and the timeout poll: memory comes first.
    set(&w, VALUE + 1);
    w.words.inaccessible.insert(AT);
    for kind in [
        ArbitrationType::WaitIfLessThan,
        ArbitrationType::DecrementAndWaitIfLessThan,
        ArbitrationType::WaitIfEqual,
    ] {
        assert_eq!(
            w.wait_address(0, AT, kind, VALUE, Timeout::Poll),
            Err(KernelError::InvalidCurrentMemory.into()),
            "{kind:?}"
        );
    }
    // A read-only word reads, but can't be decremented.
    w.words.inaccessible.clear();
    w.words.read_only.insert(AT);
    set(&w, VALUE - 1);
    assert_eq!(
        w.wait_address(
            0,
            AT,
            ArbitrationType::DecrementAndWaitIfLessThan,
            VALUE,
            NEVER
        ),
        Err(KernelError::InvalidCurrentMemory.into())
    );
    assert_eq!(get(&w), VALUE - 1);
    assert_eq!(
        w.wait_address(0, AT, ArbitrationType::WaitIfLessThan, VALUE, NEVER),
        Ok(Arbitrated::Waiting(w.threads[0]))
    );
}

#[test]
fn signal_wakes_the_first_count_waiters_most_urgent_first_and_all_for_count_up_to_zero() {
    for (count, woken) in [(1, 1), (2, 2), (3, 3), (4, 3), (0, 3), (-1, 3)] {
        // 0 signals; 1 (30), 2 (10) and 3 (20) wait.
        let mut w = World::new(&[40, 30, 10, 20]);
        set(&w, 0);
        for waiter in [1, 2, 3] {
            w.wait_address(waiter, AT, ArbitrationType::WaitIfEqual, 0, NEVER)
                .unwrap();
        }
        assert_eq!(waiting_on(&w), [w.threads[2], w.threads[3], w.threads[1]]);
        assert_eq!(
            w.signal_address(0, AT, SignalType::Signal, 0, count),
            Ok(())
        );
        let order = [2, 3, 1];
        let expected: Vec<_> = order[..woken]
            .iter()
            .map(|&index| w.wake(index, SvcId::WaitForAddress, ResultCode::SUCCESS))
            .collect();
        assert_eq!(w.scheduler.take_wakes(), expected, "count {count}");
        for &index in &order[..woken] {
            assert_eq!(w.x0(index), word(ResultCode::SUCCESS));
        }
        let left: Vec<_> = order[woken..].iter().map(|&index| w.threads[index]).collect();
        assert_eq!(waiting_on(&w), left, "count {count}");
    }
}

#[test]
fn a_signal_before_the_wait_wakes_nothing_and_the_wait_still_waits() {
    let mut w = World::new(&[30, 30]);
    set(&w, VALUE);
    assert_eq!(w.signal_address(1, AT, SignalType::Signal, 0, 0), Ok(()));
    assert_eq!(w.scheduler.take_wakes(), []);
    assert_eq!(
        w.wait_address(0, AT, ArbitrationType::WaitIfEqual, VALUE, NEVER),
        Ok(Arbitrated::Waiting(w.threads[0]))
    );
    // A signal on another address wakes nobody either.
    w.signal_address(1, AT + 4, SignalType::Signal, 0, 0).unwrap();
    assert!(w.waiting(0));
}

#[test]
fn signal_and_increment_if_equal_increments_then_wakes_or_fails_without_waking() {
    let mut w = World::new(&[30, 30]);
    set(&w, 0);
    w.wait_address(1, AT, ArbitrationType::WaitIfEqual, 0, NEVER)
        .unwrap();
    // A different word: nothing written, nobody woken.
    set(&w, VALUE + 1);
    assert_eq!(
        w.signal_address(0, AT, SignalType::SignalAndIncrementIfEqual, VALUE, 1),
        Err(KernelError::InvalidState.into())
    );
    assert_eq!(get(&w), VALUE + 1);
    assert!(w.waiting(1));
    // A read-only word: InvalidCurrentMemory, nobody woken.
    set(&w, VALUE);
    w.words.read_only.insert(AT);
    assert_eq!(
        w.signal_address(0, AT, SignalType::SignalAndIncrementIfEqual, VALUE, 1),
        Err(KernelError::InvalidCurrentMemory.into())
    );
    assert!(w.waiting(1));
    w.words.read_only.clear();
    assert_eq!(
        w.signal_address(0, AT, SignalType::SignalAndIncrementIfEqual, VALUE, 1),
        Ok(())
    );
    assert_eq!(get(&w), VALUE + 1);
    assert!(!w.waiting(1));
    // The increment wraps at the top of the range.
    set(&w, i32::MAX);
    w.signal_address(0, AT, SignalType::SignalAndIncrementIfEqual, i32::MAX, 0)
        .unwrap();
    assert_eq!(get(&w), i32::MIN);
}

/// Eden's `SignalAndModifyByWaitingCountIfEqual` computation, transcribed loop for loop
/// (`k_address_arbiter.cpp`), over a tree of `waiting` threads at the address.
fn eden_new_value(value: i32, waiting: usize, count: i32) -> i32 {
    if count <= 0 {
        if waiting > 0 {
            value.wrapping_sub(2)
        } else {
            value.wrapping_add(1)
        }
    } else if waiting > 0 {
        let mut tmp_num_waiters = 0_i32;
        // `while (++tmp_it != end && key == addr)`: the waiters after the first.
        for _ in 1..waiting {
            let before = tmp_num_waiters;
            tmp_num_waiters += 1;
            if before >= count {
                break;
            }
        }
        if tmp_num_waiters < count {
            value.wrapping_sub(1)
        } else {
            value
        }
    } else {
        value.wrapping_add(1)
    }
}

#[test]
fn the_modified_value_is_edens_for_every_waiter_count_and_signal_count() {
    for waiting in 0..8 {
        for count in -2..10 {
            for value in [VALUE, i32::MIN, i32::MAX] {
                assert_eq!(
                    modified_value(value, waiting, count),
                    eden_new_value(value, waiting, count),
                    "{waiting} waiting, count {count}, value {value}"
                );
            }
        }
    }
    // The edges: all woken at exactly `count` waiters, one left at `count + 1`.
    assert_eq!(modified_value(VALUE, 3, 3), VALUE - 1);
    assert_eq!(modified_value(VALUE, 4, 3), VALUE);
}

#[test]
fn signal_and_modify_by_waiting_count_writes_the_new_value_and_wakes() {
    // (waiters, count, word afterwards, woken)
    let cases = [
        (0, 1, VALUE + 1, 0),
        (2, 0, VALUE - 2, 2),
        (2, -1, VALUE - 2, 2),
        (2, 2, VALUE - 1, 2),
        (2, 3, VALUE - 1, 2),
        (3, 2, VALUE, 2),
    ];
    for (waiters, count, after, woken) in cases {
        let mut w = World::new(&[40, 30, 30, 30]);
        set(&w, 0);
        for waiter in 1..=waiters {
            w.wait_address(waiter, AT, ArbitrationType::WaitIfEqual, 0, NEVER)
                .unwrap();
        }
        set(&w, VALUE);
        assert_eq!(
            w.signal_address(
                0,
                AT,
                SignalType::SignalAndModifyByWaitingCountIfEqual,
                VALUE,
                count
            ),
            Ok(()),
            "{waiters} waiting, count {count}"
        );
        assert_eq!(get(&w), after, "{waiters} waiting, count {count}");
        assert_eq!(w.scheduler.take_wakes().len(), woken);
    }
}

#[test]
fn signal_and_modify_reads_without_writing_when_the_value_stays_and_fails_on_a_mismatch() {
    let mut w = World::new(&[40, 30, 30, 30]);
    set(&w, 0);
    for waiter in 1..=3 {
        w.wait_address(waiter, AT, ArbitrationType::WaitIfEqual, 0, NEVER)
            .unwrap();
    }
    // The new value is the old one (three wait, two are woken): a read-only word is fine.
    set(&w, VALUE);
    w.words.read_only.insert(AT);
    w.signal_address(
        0,
        AT,
        SignalType::SignalAndModifyByWaitingCountIfEqual,
        VALUE,
        2,
    )
    .unwrap();
    assert_eq!(w.scheduler.take_wakes().len(), 2);
    // One waits: the new value differs, so the write needs atomic access.
    assert_eq!(
        w.signal_address(
            0,
            AT,
            SignalType::SignalAndModifyByWaitingCountIfEqual,
            VALUE,
            2
        ),
        Err(KernelError::InvalidCurrentMemory.into())
    );
    w.words.read_only.clear();
    // A mismatch: InvalidState, nothing written, nobody woken.
    assert_eq!(
        w.signal_address(
            0,
            AT,
            SignalType::SignalAndModifyByWaitingCountIfEqual,
            VALUE + 1,
            2
        ),
        Err(KernelError::InvalidState.into())
    );
    assert_eq!(get(&w), VALUE);
    assert_eq!(w.scheduler.take_wakes(), []);
    // An inaccessible word.
    w.words.inaccessible.insert(AT);
    assert_eq!(
        w.signal_address(
            0,
            AT,
            SignalType::SignalAndModifyByWaitingCountIfEqual,
            VALUE,
            -1
        ),
        Err(KernelError::InvalidCurrentMemory.into())
    );
}

#[test]
fn a_timer_ends_an_address_wait_with_timed_out() {
    let mut w = World::new(&[30, 30]);
    set(&w, 0);
    let timeout = Timeout::from_nanos(HostNanos(NOW), AFTER);
    w.wait_address(0, AT, ArbitrationType::WaitIfEqual, 0, timeout)
        .unwrap();
    let deadline = NOW + AFTER.unsigned_abs() + TIMEOUT_SLACK_NS;
    assert_eq!(w.scheduler.sync().next_deadline(), Some(HostNanos(deadline)));
    assert_eq!(w.fire(deadline - 1), 0);
    assert_eq!(w.fire(deadline), 1);
    assert_eq!(
        w.scheduler.take_wakes(),
        [w.wake(0, SvcId::WaitForAddress, KernelError::TimedOut.code())]
    );
    assert_eq!(w.x0(0), word(KernelError::TimedOut.code()));
    assert_eq!(waiting_on(&w), []);
    // A signal later finds nobody, and a signal before the timer cancels it.
    w.wait_address(0, AT, ArbitrationType::WaitIfEqual, 0, timeout)
        .unwrap();
    w.signal_address(1, AT, SignalType::Signal, 0, 1).unwrap();
    assert_eq!(w.scheduler.sync().next_deadline(), None);
}

#[test]
fn a_waiter_whose_priority_changes_moves_to_its_new_place() {
    let mut w = World::new(&[40, 30, 20, 10]);
    set(&w, 0);
    for waiter in [1, 2, 3] {
        w.wait_address(waiter, AT, ArbitrationType::WaitIfEqual, 0, NEVER)
            .unwrap();
    }
    assert_eq!(waiting_on(&w), [w.threads[3], w.threads[2], w.threads[1]]);
    let (t1, caller) = (w.threads[1], core(0));
    w.scheduler
        .set_base_priority(caller, t1, crate::sched::Priority::new(5).unwrap())
        .unwrap();
    assert_eq!(waiting_on(&w), [w.threads[1], w.threads[3], w.threads[2]]);
    w.signal_address(0, AT, SignalType::Signal, 0, 1).unwrap();
    assert!(!w.waiting(1));
}

#[test]
fn cancelling_an_address_waiter_only_leaves_a_pending_cancel() {
    let mut w = World::new(&[30]);
    set(&w, 0);
    w.wait_address(0, AT, ArbitrationType::WaitIfEqual, 0, NEVER)
        .unwrap();
    w.scheduler.cancel_synchronization(w.threads[0]).unwrap();
    assert!(w.waiting(0));
    assert!(w.scheduler.sync().is_cancel_pending(w.threads[0]));
}

#[test]
fn a_terminated_address_waiter_leaves_the_tree() {
    let mut w = World::new(&[30]);
    set(&w, 0);
    let timeout = Timeout::from_nanos(HostNanos(NOW), AFTER);
    w.wait_address(0, AT, ArbitrationType::WaitIfEqual, 0, timeout)
        .unwrap();
    w.scheduler.terminate_thread(w.threads[0]).unwrap();
    assert_eq!(waiting_on(&w), []);
    assert_eq!(w.scheduler.sync().arbiter().key_count(), 0);
    assert_eq!(w.scheduler.sync().next_deadline(), None);
    assert_eq!(w.scheduler.take_wakes(), []);
}
