//! The caps of every layer, each at its edge on both sides, and what happens past it.

use super::*;

/// Limits that never bind, for changing one at a time.
const LOOSE: Limits = Limits {
    addresses: 1_000,
    programs: 1_000,
    program_bytes: 1 << 30,
    shaders: 1_000,
    shader_bytes: 1 << 30,
    tickets: 1_000,
};

/// The bytes of a vertex program with one filler tagged `tag`.
fn one(tag: u32) -> Vec<u8> {
    program(GuestStage::Vertex, &[tag])
}

/// Places a distinct one-filler vertex program at each of `count` addresses from [`A`], 0x200
/// bytes apart, and returns their requests.
fn place_many(rig: &mut Rig, count: u32) -> Vec<ProgramRequest> {
    (0..count)
        .map(|i| {
            let address = A + u64::from(i) * 0x200;
            rig.place(address, GuestStage::Vertex, &one(100 + i));
            vertex(address)
        })
        .collect()
}

#[test]
fn the_address_map_empties_when_a_capture_finds_it_full() {
    let mut rig = Rig::with_limits(Limits {
        addresses: 2,
        ..LOOSE
    });
    let requests = place_many(&mut rig, 3);
    rig.lookup(&requests[..1]);
    rig.lookup(&requests[1..2]);
    // One below the cap before the second capture: no flush.
    assert_eq!(rig.cache.addresses.len(), 2);
    assert_eq!(rig.counters().address_flushes, 0);
    // At the cap: the third capture empties it first.
    rig.lookup(&requests[2..3]);
    assert_eq!(rig.cache.addresses.len(), 1);
    assert_eq!(rig.counters().address_flushes, 1);
    // A flushed address is captured again; its program entry is still there.
    rig.lookup(&requests[..1]);
    assert_eq!(rig.counters().captures, 4);
    assert_eq!(rig.counters().programs, 3);
}

#[test]
fn the_program_entries_empty_at_their_count() {
    let mut rig = Rig::with_limits(Limits {
        programs: 2,
        ..LOOSE
    });
    let requests = place_many(&mut rig, 3);
    rig.lookup(&requests[..1]);
    rig.lookup(&requests[1..2]);
    assert_eq!(rig.cache.programs.len(), 2);
    assert_eq!(rig.counters().program_flushes, 0);
    rig.lookup(&requests[2..3]);
    assert_eq!(rig.cache.programs.len(), 1);
    assert_eq!(rig.counters().program_flushes, 1);
    assert_eq!(rig.cache.program_bytes, one(0).len() as u64);
    // The first program's address entry is still there, its program entry is not: it is
    // captured and made again.
    rig.lookup(&requests[..1]);
    assert_eq!(rig.counters().captures, 4);
    assert_eq!(rig.counters().programs, 4);
}

#[test]
fn the_program_entries_hold_exactly_their_byte_budget() {
    let len = one(0).len() as u64;
    for (budget, flushes) in [(2 * len, 0), (2 * len - 1, 1)] {
        let mut rig = Rig::with_limits(Limits {
            program_bytes: budget,
            ..LOOSE
        });
        let requests = place_many(&mut rig, 2);
        rig.lookup(&requests);
        assert_eq!(rig.counters().program_flushes, flushes, "budget {budget}");
        assert_eq!(
            rig.cache.programs.len(),
            2 - usize::try_from(flushes).expect("small")
        );
    }
}

#[test]
fn programs_of_different_sizes_are_charged_their_own_bytes() {
    let short = program(GuestStage::Vertex, &[1]);
    let long = program(GuestStage::Vertex, &[1, 2, 3, 4]);
    let budget = (short.len() + long.len()) as u64;
    let mut rig = Rig::with_limits(Limits {
        program_bytes: budget,
        ..LOOSE
    });
    rig.place(A, GuestStage::Vertex, &short);
    rig.place(B, GuestStage::Vertex, &long);
    // In one lookup, then the third flushes.
    rig.lookup(&[vertex(A), vertex(B)]);
    assert_eq!(rig.cache.program_bytes, budget);
    assert_eq!(rig.counters().program_flushes, 0);
    rig.place(C, GuestStage::Vertex, &program(GuestStage::Vertex, &[7]));
    rig.lookup(&[vertex(C)]);
    assert_eq!(rig.counters().program_flushes, 1);
    assert_eq!(rig.cache.program_bytes, short.len() as u64);
}

#[test]
fn the_shader_entries_drop_settled_ones_at_their_count() {
    let mut rig = Rig::with_limits(Limits {
        shaders: 2,
        ..LOOSE
    });
    let requests = place_many(&mut rig, 3);
    pending(rig.lookup(&requests[..1]));
    // One settled, one pending: the third drops the settled one only.
    rig.executor.run_all();
    ready(rig.lookup(&requests[..1]));
    pending(rig.lookup(&requests[1..2]));
    assert_eq!(rig.cache.shaders.len(), 2);
    assert_eq!(rig.counters().shader_flushes, 0);
    pending(rig.lookup(&requests[2..3]));
    assert_eq!(rig.cache.shaders.len(), 2);
    assert_eq!(rig.counters().shader_flushes, 1);
    assert!(
        rig.cache
            .shaders
            .values()
            .all(|entry| matches!(entry.state, super::super::ShaderState::Pending))
    );
}

#[test]
fn a_shader_layer_full_of_pending_translations_refuses_without_remembering() {
    let mut rig = Rig::with_limits(Limits {
        shaders: 2,
        ..LOOSE
    });
    let requests = place_many(&mut rig, 3);
    pending(rig.lookup(&requests[..1]));
    pending(rig.lookup(&requests[1..2]));
    let rejection = rejected(rig.lookup(&requests[2..3]));
    assert_eq!(rejection.request, requests[2]);
    assert_eq!(
        *rejection.reason,
        Rejection::Cache(CacheRefusal::Full { limit: 2 })
    );
    assert_eq!(rig.executor.queued(), 2);
    // Not a remembered rejection: once the others finish, it is queued.
    rig.executor.run_all();
    pending(rig.lookup(&requests[2..3]));
    assert_eq!(rig.counters().rejections, 0);
    assert_eq!(rig.counters().translations, 3);
}

#[test]
fn the_shader_entries_hold_exactly_their_byte_budget() {
    let len = one(0).len() as u64;
    for (budget, refused) in [(2 * len, false), (2 * len - 1, true)] {
        let mut rig = Rig::with_limits(Limits {
            shader_bytes: budget,
            ..LOOSE
        });
        let requests = place_many(&mut rig, 2);
        pending(rig.lookup(&requests[..1]));
        let second = rig.lookup(&requests[1..2]);
        assert_eq!(
            matches!(second, CacheLookup::Rejected(_)),
            refused,
            "budget {budget}"
        );
        assert_eq!(rig.cache.shader_bytes, if refused { len } else { 2 * len });
    }
}

#[test]
fn shaders_are_charged_their_programs_bytes() {
    let long = program(GuestStage::Vertex, &[1, 2, 3, 4, 5]);
    let mut rig = Rig::new();
    rig.place(A, GuestStage::Vertex, &long);
    rig.place(B, GuestStage::Vertex, &one(9));
    pending(rig.lookup(&[vertex(A), vertex(B)]));
    assert_eq!(rig.cache.shader_bytes, (long.len() + one(9).len()) as u64);
    // A flush recounts what stays.
    rig.executor.run_next();
    rig.cache.limits.shaders = 2;
    rig.place(C, GuestStage::Vertex, &one(10));
    pending(rig.lookup(&[vertex(C)]));
    assert_eq!(rig.counters().shader_flushes, 1);
    assert_eq!(rig.cache.shader_bytes, 2 * one(9).len() as u64);
}

#[test]
fn tickets_past_the_cap_drop_settled_sets_first_then_the_oldest() {
    let mut rig = Rig::with_limits(Limits {
        tickets: 2,
        ..LOOSE
    });
    let requests = place_many(&mut rig, 4);
    let first = pending(rig.lookup(&requests[..1]));
    let second = pending(rig.lookup(&requests[1..2]));
    // The first set settles without being looked up again: its ticket goes first.
    assert!(rig.executor.run_next());
    let third = pending(rig.lookup(&requests[2..3]));
    assert!(!rig.cache.tickets.contains_key(&first));
    assert!(rig.cache.tickets.contains_key(&second));
    assert_eq!(rig.counters().tickets_dropped, 0);
    // Nothing settled: the oldest pending ticket is dropped, and waiting on it answers pending.
    let fourth = pending(rig.lookup(&requests[3..4]));
    assert_eq!(
        rig.cache.tickets.keys().copied().collect::<Vec<_>>(),
        [third, fourth]
    );
    assert_eq!(rig.counters().tickets_dropped, 1);
    assert_eq!(pending(rig.cache.wait(second)), second);
    assert_eq!(rig.cache.set_tickets.len(), 2);
}

#[test]
fn a_hostile_stream_of_new_programs_stays_inside_every_cap() {
    let limits = Limits {
        addresses: 4,
        programs: 3,
        program_bytes: 1 << 20,
        shaders: 5,
        shader_bytes: 1 << 20,
        tickets: 2,
    };
    let mut rig = Rig::with_limits(limits);
    for round in 0..40_u32 {
        let address = A + u64::from(round % 8) * 0x200;
        rig.place(address, GuestStage::Vertex, &one(1_000 + round));
        rig.cache.invalidate(address..address + 1);
        rig.lookup(&[vertex(address)]);
        if round % 3 == 0 {
            rig.executor.run_all();
        }
        assert!(rig.cache.addresses.len() <= limits.addresses);
        assert!(rig.cache.programs.len() <= limits.programs);
        assert!(rig.cache.shaders.len() <= limits.shaders);
        assert!(rig.cache.tickets.len() <= limits.tickets);
        assert!(rig.cache.set_tickets.len() <= limits.tickets);
    }
}
