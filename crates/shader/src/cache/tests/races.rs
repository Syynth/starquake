//! Races between workers finishing and the back thread looking up, writing and waiting, each on
//! a fixed schedule set by the scripted executor (module docs of the cache, "Where this differs
//! from the spec": the job inbox stands where the spec's generation counter would).

use super::*;

#[test]
fn a_result_posted_but_not_taken_leaves_the_entry_pending_until_the_next_lookup() {
    let mut rig = Rig::new();
    rig.place(A, GuestStage::Vertex, &program(GuestStage::Vertex, &[1]));
    let ticket = pending(rig.lookup(&[vertex(A)]));
    assert!(rig.executor.run_next());
    // The in-between state: posted, not yet taken.
    assert!(rig.cache.inbox.posted());
    assert!(
        rig.cache
            .shaders
            .values()
            .all(|entry| matches!(entry.state, super::super::ShaderState::Pending))
    );
    assert!(rig.cache.tickets.contains_key(&ticket));
    // The lookup takes it.
    ready(rig.lookup(&[vertex(A)]));
    assert!(!rig.cache.inbox.posted());
}

#[test]
fn a_job_finishing_between_lookup_and_wait_is_found_by_the_wait() {
    let mut rig = Rig::new();
    rig.place(A, GuestStage::Vertex, &program(GuestStage::Vertex, &[1]));
    let ticket = pending(rig.lookup(&[vertex(A)]));
    assert!(rig.executor.run_next());
    assert_eq!(ready(rig.cache.wait_timeout(ticket, WATCHDOG)).len(), 1);
}

#[test]
fn jobs_finishing_in_reverse_order_settle_their_own_entries() {
    let mut rig = Rig::new();
    let first = rig.place(A, GuestStage::Vertex, &program(GuestStage::Vertex, &[1]));
    let second = rig.place(B, GuestStage::Pixel, &program(GuestStage::Pixel, &[2]));
    let ticket = pending(rig.lookup(&[vertex(A), pixel(B)]));
    assert!(rig.executor.run_last());
    // Only the second is ready.
    assert_eq!(pending(rig.lookup(&[vertex(A), pixel(B)])), ticket);
    assert_eq!(ready(rig.lookup(&[pixel(B)]))[0].1.hash, second);
    assert!(rig.executor.run_next());
    let shaders = ready(rig.cache.wait_timeout(ticket, WATCHDOG));
    assert_eq!(shaders[0].1.hash, first);
    assert_eq!(shaders[1].1.hash, second);
}

#[test]
fn a_write_while_pending_gives_the_new_bytes_a_new_ticket_and_the_old_ticket_its_own_set() {
    let mut rig = Rig::new();
    let old = rig.place(A, GuestStage::Vertex, &program(GuestStage::Vertex, &[1]));
    let old_ticket = pending(rig.lookup(&[vertex(A)]));
    // The guest rewrites the program before the translation ran.
    let new = rig.place(A, GuestStage::Vertex, &program(GuestStage::Vertex, &[2]));
    rig.cache.invalidate(A..A + 1);
    let new_ticket = pending(rig.lookup(&[vertex(A)]));
    assert_ne!(old_ticket, new_ticket);
    assert_eq!(rig.executor.queued(), 2);
    // The old translation finishes: the address still waits for the new one.
    assert!(rig.executor.run_next());
    assert_eq!(pending(rig.lookup(&[vertex(A)])), new_ticket);
    // The old ticket names the programs as they were captured.
    assert_eq!(
        ready(rig.cache.wait_timeout(old_ticket, WATCHDOG))[0]
            .1
            .hash,
        old
    );
    assert!(rig.executor.run_next());
    assert_eq!(
        ready(rig.cache.wait_timeout(new_ticket, WATCHDOG))[0]
            .1
            .hash,
        new
    );
    assert_eq!(ready(rig.lookup(&[vertex(A)]))[0].1.hash, new);
}

#[test]
fn the_same_bytes_written_while_pending_keep_the_ticket_and_the_job() {
    let mut rig = Rig::new();
    let bytes = program(GuestStage::Vertex, &[1]);
    rig.place(A, GuestStage::Vertex, &bytes);
    let ticket = pending(rig.lookup(&[vertex(A)]));
    rig.place(A, GuestStage::Vertex, &bytes);
    rig.cache.invalidate(A..A + bytes.len() as u64);
    assert_eq!(pending(rig.lookup(&[vertex(A)])), ticket);
    assert_eq!(rig.executor.queued(), 1);
    assert_eq!(rig.counters().captures, 2);
    assert_eq!(rig.counters().translations, 1);
}

#[test]
fn a_ready_member_flushed_before_the_wait_answers_pending() {
    let mut rig = Rig::with_limits(Limits {
        shaders: 2,
        ..Limits::SHIPPING
    });
    rig.place(A, GuestStage::Vertex, &program(GuestStage::Vertex, &[1]));
    rig.place(B, GuestStage::Pixel, &program(GuestStage::Pixel, &[2]));
    rig.place(C, GuestStage::Vertex, &program(GuestStage::Vertex, &[3]));
    let ticket = pending(rig.lookup(&[vertex(A), pixel(B)]));
    rig.executor.run_all();
    // Another lookup finds the layer full and drops the settled entries, the set's included.
    pending(rig.lookup(&[vertex(C)]));
    assert_eq!(rig.counters().shader_flushes, 1);
    assert_eq!(pending(rig.cache.wait_timeout(ticket, WATCHDOG)), ticket);
    // The ticket is spent; once the other translation is done, a lookup queues the set again.
    assert!(!rig.cache.tickets.contains_key(&ticket));
    rig.executor.run_all();
    pending(rig.lookup(&[vertex(A), pixel(B)]));
    assert_eq!(rig.counters().translations, 5);
}

#[test]
fn a_late_result_for_a_settled_entry_changes_nothing() {
    let mut rig = Rig::new();
    let hash = rig.place(A, GuestStage::Vertex, &program(GuestStage::Vertex, &[1]));
    pending(rig.lookup(&[vertex(A)]));
    rig.executor.run_all();
    let first = ready(rig.lookup(&[vertex(A)]));
    // A second result for the same stamp, as a duplicate post would bring.
    let settle = super::super::jobs::Settle::new(Arc::clone(&rig.cache.inbox), first[0].0);
    settle.post(Err(no_exit()));
    let again = ready(rig.lookup(&[vertex(A)]));
    assert!(Arc::ptr_eq(&again[0].1, &first[0].1));
    assert_eq!(again[0].1.hash, hash);
    assert_eq!(rig.counters().rejections, 0);
}
