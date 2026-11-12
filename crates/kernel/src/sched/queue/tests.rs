//! Tests for the priority queues against `k_priority_queue.h`: which lists a push puts a thread
//! in, order within a level, the per-core level bitmap, walking across levels, and the
//! requeueing operations.

use starquake_cpu::GuestRegs;
use starquake_trace::GuestTicks;

use super::*;
use crate::sched::thread::ThreadState;

const CORES: u8 = 4;

fn core(index: u8) -> CoreId {
    CoreId::new(index).unwrap()
}

fn prio(value: u8) -> Priority {
    Priority::new(value).unwrap()
}

fn mask(cores: &[u8]) -> AffinityMask {
    AffinityMask::from_bits(cores.iter().fold(0, |bits, &c| bits | core(c).bit()))
}

/// A runnable row at `priority`, active on `active`, allowed on `allowed`.
fn add(threads: &mut Threads, priority: u8, active: Option<u8>, allowed: &[u8]) -> ThreadId {
    let active = active.map(core);
    threads
        .insert(Thread {
            state: ThreadState::Runnable,
            suspend_flags: 0,
            suspend_requested: 0,
            priority: prio(priority),
            base_priority: prio(priority),
            affinity: mask(allowed),
            ideal_core: active,
            active_core: active,
            current_core: core(0),
            last_scheduled_tick: GuestTicks(0),
            yield_schedule_count: None,
            links: vec![QueueLink::default(); usize::from(CORES)],
            context: GuestRegs::default(),
            process: None,
            object: None,
            holds_itself: false,
            guest_id: 0,
            tls_address: 0,
            virtual_ideal_core: active.map_or(-1, |core| i32::from(core.index())),
            virtual_affinity: mask(allowed).bits(),
        })
        .unwrap()
}

fn sched(pq: &PriorityQueue, t: &Threads, p: u8, c: u8) -> Vec<ThreadId> {
    pq.scheduled_list(t, prio(p), core(c)).unwrap()
}

fn sugg(pq: &PriorityQueue, t: &Threads, p: u8, c: u8) -> Vec<ThreadId> {
    pq.suggested_list(t, prio(p), core(c)).unwrap()
}

#[test]
fn push_back_schedules_on_the_active_core_and_suggests_to_the_rest() {
    let mut t = Threads::new();
    let mut pq = PriorityQueue::new(CORES);
    let a = add(&mut t, 10, Some(1), &[0, 1, 2]);
    pq.push_back(&mut t, a).unwrap();
    assert_eq!(sched(&pq, &t, 10, 1), vec![a]);
    assert_eq!(sugg(&pq, &t, 10, 0), vec![a]);
    assert_eq!(sugg(&pq, &t, 10, 2), vec![a]);
    assert_eq!(sugg(&pq, &t, 10, 1), vec![]);
    assert_eq!(sugg(&pq, &t, 10, 3), vec![]);
    assert!(pq.is_available(true, prio(10), core(1)));
    assert!(!pq.is_available(true, prio(10), core(0)));
    assert!(pq.is_available(false, prio(10), core(0)));
    assert_eq!(pq.scheduled_front(core(1)).unwrap(), Some(a));
    assert_eq!(pq.suggested_front(core(2)).unwrap(), Some(a));
}

#[test]
fn a_thread_with_no_active_core_is_only_suggested() {
    let mut t = Threads::new();
    let mut pq = PriorityQueue::new(CORES);
    let a = add(&mut t, 10, None, &[0, 3]);
    pq.push_back(&mut t, a).unwrap();
    for c in 0..CORES {
        assert_eq!(sched(&pq, &t, 10, c), vec![]);
    }
    assert_eq!(sugg(&pq, &t, 10, 0), vec![a]);
    assert_eq!(sugg(&pq, &t, 10, 3), vec![a]);
}

#[test]
fn levels_are_fifo_and_remove_relinks_neighbours() {
    let mut t = Threads::new();
    let mut pq = PriorityQueue::new(CORES);
    let ids: Vec<ThreadId> = (0..4).map(|_| add(&mut t, 20, Some(0), &[0])).collect();
    for &id in &ids {
        pq.push_back(&mut t, id).unwrap();
    }
    assert_eq!(sched(&pq, &t, 20, 0), ids);
    // Middle, then head, then tail.
    pq.remove(&mut t, ids[1]).unwrap();
    assert_eq!(sched(&pq, &t, 20, 0), vec![ids[0], ids[2], ids[3]]);
    pq.remove(&mut t, ids[0]).unwrap();
    assert_eq!(sched(&pq, &t, 20, 0), vec![ids[2], ids[3]]);
    pq.remove(&mut t, ids[3]).unwrap();
    assert_eq!(sched(&pq, &t, 20, 0), vec![ids[2]]);
    assert!(pq.is_available(true, prio(20), core(0)));
    pq.remove(&mut t, ids[2]).unwrap();
    assert_eq!(sched(&pq, &t, 20, 0), vec![]);
    assert!(!pq.is_available(true, prio(20), core(0)));
    assert_eq!(pq.scheduled_front(core(0)).unwrap(), None);
    // The removed rows' links are cleared.
    for id in ids {
        assert_eq!(t.get(id).unwrap().links[0], QueueLink::default());
    }
}

#[test]
fn front_is_the_most_urgent_level_and_next_walks_down_the_levels() {
    let mut t = Threads::new();
    let mut pq = PriorityQueue::new(CORES);
    let low = add(&mut t, Priority::LOWEST.get(), Some(2), &[2]);
    let mid_a = add(&mut t, 30, Some(2), &[2]);
    let mid_b = add(&mut t, 30, Some(2), &[2]);
    let high = add(&mut t, Priority::HIGHEST.get(), Some(2), &[2]);
    for id in [low, mid_a, mid_b, high] {
        pq.push_back(&mut t, id).unwrap();
    }
    assert_eq!(pq.scheduled_front(core(2)).unwrap(), Some(high));
    let mut walked = vec![high];
    let mut at = high;
    while let Some(next) = pq.scheduled_next(&t, core(2), at).unwrap() {
        walked.push(next);
        at = next;
    }
    assert_eq!(walked, vec![high, mid_a, mid_b, low]);
}

#[test]
fn move_to_scheduled_back_returns_the_new_head() {
    let mut t = Threads::new();
    let mut pq = PriorityQueue::new(CORES);
    let a = add(&mut t, 5, Some(0), &[0]);
    let b = add(&mut t, 5, Some(0), &[0]);
    pq.push_back(&mut t, a).unwrap();
    pq.push_back(&mut t, b).unwrap();
    assert_eq!(
        pq.move_to_scheduled_back(&mut t, a, core(0)).unwrap(),
        Some(b)
    );
    assert_eq!(sched(&pq, &t, 5, 0), vec![b, a]);
    // Alone at its level, a thread is its own new head.
    let c = add(&mut t, 6, Some(0), &[0]);
    pq.push_back(&mut t, c).unwrap();
    assert_eq!(
        pq.move_to_scheduled_back(&mut t, c, core(0)).unwrap(),
        Some(c)
    );
}

#[test]
fn change_priority_keeps_a_running_thread_in_front_but_suggests_it_at_the_back() {
    let mut t = Threads::new();
    let mut pq = PriorityQueue::new(CORES);
    let other = add(&mut t, 8, Some(0), &[0, 1]);
    let other_on_1 = add(&mut t, 8, Some(1), &[1]);
    let moving = add(&mut t, 12, Some(0), &[0, 1]);
    for id in [other, other_on_1, moving] {
        pq.push_back(&mut t, id).unwrap();
    }
    t.get_mut(moving).unwrap().priority = prio(8);
    pq.change_priority(&mut t, prio(12), true, moving).unwrap();
    assert_eq!(sched(&pq, &t, 12, 0), vec![]);
    assert_eq!(sugg(&pq, &t, 12, 1), vec![]);
    // Running: front of the scheduled list; the suggestion still goes at the back
    // ("Nintendo pushes onto the back of the suggested queue").
    assert_eq!(sched(&pq, &t, 8, 0), vec![moving, other]);
    assert_eq!(sugg(&pq, &t, 8, 1), vec![other, moving]);

    // Not running: back of the scheduled list.
    let already = add(&mut t, 9, Some(0), &[0]);
    pq.push_back(&mut t, already).unwrap();
    t.get_mut(moving).unwrap().priority = prio(9);
    pq.change_priority(&mut t, prio(8), false, moving).unwrap();
    assert_eq!(sched(&pq, &t, 9, 0), vec![already, moving]);
    assert_eq!(sched(&pq, &t, 8, 0), vec![other]);
    assert_eq!(sugg(&pq, &t, 9, 1), vec![moving]);
}

#[test]
fn change_core_swaps_scheduled_and_suggested_lists() {
    let mut t = Threads::new();
    let mut pq = PriorityQueue::new(CORES);
    let resident = add(&mut t, 7, Some(3), &[3]);
    let a = add(&mut t, 7, Some(1), &[1, 3]);
    pq.push_back(&mut t, resident).unwrap();
    pq.push_back(&mut t, a).unwrap();
    assert_eq!(sugg(&pq, &t, 7, 3), vec![a]);

    t.get_mut(a).unwrap().active_core = Some(core(3));
    pq.change_core(&mut t, Some(core(1)), a, true).unwrap();
    assert_eq!(sched(&pq, &t, 7, 3), vec![a, resident]);
    assert_eq!(sugg(&pq, &t, 7, 3), vec![]);
    assert_eq!(sched(&pq, &t, 7, 1), vec![]);
    assert_eq!(sugg(&pq, &t, 7, 1), vec![a]);

    // To no core: only suggested, everywhere it may run.
    t.get_mut(a).unwrap().active_core = None;
    pq.change_core(&mut t, Some(core(3)), a, false).unwrap();
    assert_eq!(sched(&pq, &t, 7, 3), vec![resident]);
    assert_eq!(sugg(&pq, &t, 7, 3), vec![a]);
    assert_eq!(sugg(&pq, &t, 7, 1), vec![a]);

    // Back onto core 1, at the back.
    let b = add(&mut t, 7, Some(1), &[1]);
    pq.push_back(&mut t, b).unwrap();
    t.get_mut(a).unwrap().active_core = Some(core(1));
    pq.change_core(&mut t, None, a, false).unwrap();
    assert_eq!(sched(&pq, &t, 7, 1), vec![b, a]);
    assert_eq!(sugg(&pq, &t, 7, 1), vec![]);
    assert_eq!(sugg(&pq, &t, 7, 3), vec![a]);

    // Same core: nothing moves.
    pq.change_core(&mut t, Some(core(1)), a, true).unwrap();
    assert_eq!(sched(&pq, &t, 7, 1), vec![b, a]);
}

#[test]
fn change_affinity_mask_leaves_old_lists_and_joins_new_ones() {
    let mut t = Threads::new();
    let mut pq = PriorityQueue::new(CORES);
    let a = add(&mut t, 40, Some(0), &[0, 1]);
    pq.push_back(&mut t, a).unwrap();
    let old = t.get(a).unwrap().affinity;
    {
        let row = t.get_mut(a).unwrap();
        row.affinity = mask(&[2, 3]);
        row.active_core = Some(core(3));
    }
    pq.change_affinity_mask(&mut t, Some(core(0)), old, a)
        .unwrap();
    assert_eq!(sched(&pq, &t, 40, 0), vec![]);
    assert_eq!(sugg(&pq, &t, 40, 1), vec![]);
    assert_eq!(sched(&pq, &t, 40, 3), vec![a]);
    assert_eq!(sugg(&pq, &t, 40, 2), vec![a]);
    assert!(!pq.is_available(true, prio(40), core(0)));
    assert!(!pq.is_available(false, prio(40), core(1)));
}

#[test]
fn walk_is_bounded_by_the_thread_table() {
    let mut t = Threads::new();
    add(&mut t, 1, Some(0), &[0]);
    add(&mut t, 1, Some(0), &[0]);
    let mut walk = Walk::new(&t);
    walk.step().unwrap();
    walk.step().unwrap();
    assert_eq!(walk.step(), Err(SchedError::Corrupt));
}

#[test]
fn a_cycle_in_the_links_is_corrupt_not_a_hang() {
    let mut t = Threads::new();
    let mut pq = PriorityQueue::new(CORES);
    let a = add(&mut t, 3, Some(0), &[0]);
    let b = add(&mut t, 3, Some(0), &[0]);
    pq.push_back(&mut t, a).unwrap();
    pq.push_back(&mut t, b).unwrap();
    t.get_mut(b).unwrap().links[0].next = Some(a);
    assert_eq!(
        pq.scheduled_list(&t, prio(3), core(0)),
        Err(SchedError::Corrupt)
    );
}

#[test]
fn a_stale_link_is_corrupt() {
    let mut t = Threads::new();
    let mut pq = PriorityQueue::new(CORES);
    let a = add(&mut t, 3, Some(0), &[0]);
    pq.push_back(&mut t, a).unwrap();
    t.remove(a).unwrap();
    assert_eq!(pq.scheduled_next(&t, core(0), a), Err(SchedError::Corrupt));
    assert_eq!(pq.remove(&mut t, a), Err(SchedError::Corrupt));
}
