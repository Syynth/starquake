//! Rejections: where each is remembered, that it names the program, that it is counted once, the
//! order among several, and its words.

use super::*;
use crate::capture::CaptureError;
use crate::exchange::{ExchangeError, MAX_QUESTION_ROUNDS};

#[test]
fn an_unreadable_program_is_rejected_naming_it_without_a_hash() {
    let mut rig = Rig::new();
    // Nothing ends the program before the mapped run does.
    let fillers: Vec<u8> = (0..4).flat_map(|tag| filler(tag).to_le_bytes()).collect();
    let at = rig.memory.end() - fillers.len() as u64;
    rig.memory.write(at, &fillers);
    let rejection = rejected(rig.lookup(&[ProgramRequest {
        stage: GuestStage::Compute,
        address: at,
    }]));
    let reason = Rejection::Capture(CaptureError::InstructionUnreadable {
        index: 4,
        offset: 64,
    });
    assert_eq!(
        *rejection,
        ProgramRejection {
            request: ProgramRequest {
                stage: GuestStage::Compute,
                address: at,
            },
            hash: None,
            reason: Arc::new(reason),
        }
    );
    assert_eq!(
        rejection.to_string(),
        format!("the compute program at {at:#x}: instruction 4 cannot be read at byte 0x40")
    );
    assert_eq!(rig.counters().rejections, 1);
    assert_eq!(rig.fake.program_calls(), 0);
}

#[test]
fn a_rejected_program_names_it_with_its_hash_and_its_first_bad_instruction() {
    let mut rig = Rig::new();
    let hash = rig.place(A, GuestStage::Pixel, &program(GuestStage::Pixel, &[1, 2]));
    rig.fake
        .script()
        .program_rejections
        .insert(hash, decode_rejection(1));
    let rejection = rejected(rig.lookup(&[pixel(A)]));
    assert_eq!(rejection.request, pixel(A));
    assert_eq!(rejection.hash, Some(hash));
    assert_eq!(*rejection.reason, decode_rejection(1));
    assert_eq!(
        rejection.to_string(),
        format!(
            "the pixel program at {A:#x} (content hash {hash}): instruction 1: unknown opcode 0x000"
        )
    );
    // Nothing was queued.
    assert_eq!(rig.executor.queued(), 0);
}

#[test]
fn a_rejected_shader_is_rejected_after_its_translation() {
    let mut rig = Rig::new();
    let hash = rig.place(A, GuestStage::Vertex, &program(GuestStage::Vertex, &[1]));
    rig.fake
        .script()
        .shader_rejections
        .insert(hash, too_large());
    let ticket = pending(rig.lookup(&[vertex(A)]));
    rig.executor.run_all();
    let rejection = rejected(rig.cache.wait_timeout(ticket, WATCHDOG));
    assert_eq!(
        *rejection,
        ProgramRejection {
            request: vertex(A),
            hash: Some(hash),
            reason: Arc::new(too_large()),
        }
    );
    // Remembered: a later lookup is rejected without new work.
    let again = rejected(rig.lookup(&[vertex(A)]));
    assert_eq!(*again, *rejection);
    assert_eq!(rig.counters().translations, 1);
    assert_eq!(rig.counters().rejections, 1);
}

#[test]
fn a_shader_needing_a_feature_the_target_lacks_is_rejected() {
    let mut rig = Rig::new();
    let lacking = HostTarget {
        caps: Requirements::default(),
        ..TARGET
    };
    rig.cache = ShaderCache::with_limits(
        Arc::clone(&rig.fake),
        Arc::<Scripted>::clone(&rig.executor),
        lacking,
        Limits::SHIPPING,
    );
    let needing = rig.place(A, GuestStage::Vertex, &program(GuestStage::Vertex, &[1]));
    rig.place(B, GuestStage::Vertex, &program(GuestStage::Vertex, &[2]));
    rig.fake.script().needs.insert(
        needing,
        Requirements {
            subgroup: true,
            ..Requirements::default()
        },
    );
    pending(rig.lookup(&[vertex(A)]));
    pending(rig.lookup(&[vertex(B)]));
    rig.executor.run_all();
    let rejection = rejected(rig.lookup(&[vertex(A)]));
    assert_eq!(
        *rejection.reason,
        Rejection::Cache(CacheRefusal::Unsupported {
            missing: Requirements {
                subgroup: true,
                ..Requirements::default()
            }
        })
    );
    assert_eq!(
        rejection.reason.to_string(),
        "the shader needs subgroup operations, which the host lacks"
    );
    // A shader needing nothing is handed out.
    assert_eq!(ready(rig.lookup(&[vertex(B)])).len(), 1);
    assert_eq!(rig.counters().rejections, 1);
}

/// The narrower "needs subgroup size 32" flag is checked on its own (decided by the owner,
/// 2026-11-20): a target with subgroups but not of size 32 refuses a shader that needs both,
/// naming only the size, and hands out one that needs subgroups alone (a quad-only shader).
#[test]
fn a_target_without_subgroup_size_32_refuses_only_the_shaders_needing_it() {
    let mut rig = Rig::new();
    let lacking = HostTarget {
        caps: Requirements {
            subgroup: true,
            ..Requirements::default()
        },
        ..TARGET
    };
    rig.cache = ShaderCache::with_limits(
        Arc::clone(&rig.fake),
        Arc::<Scripted>::clone(&rig.executor),
        lacking,
        Limits::SHIPPING,
    );
    let sized = rig.place(A, GuestStage::Vertex, &program(GuestStage::Vertex, &[1]));
    let quad = rig.place(B, GuestStage::Vertex, &program(GuestStage::Vertex, &[2]));
    let subgroups = Requirements {
        subgroup: true,
        ..Requirements::default()
    };
    rig.fake.script().needs.insert(
        sized,
        Requirements {
            subgroup_size_32: true,
            ..subgroups
        },
    );
    rig.fake.script().needs.insert(quad, subgroups);
    pending(rig.lookup(&[vertex(A)]));
    pending(rig.lookup(&[vertex(B)]));
    rig.executor.run_all();
    let rejection = rejected(rig.lookup(&[vertex(A)]));
    assert_eq!(
        *rejection.reason,
        Rejection::Cache(CacheRefusal::Unsupported {
            missing: Requirements {
                subgroup_size_32: true,
                ..Requirements::default()
            }
        })
    );
    assert_eq!(ready(rig.lookup(&[vertex(B)])).len(), 1);
}

#[test]
fn a_shader_needing_a_feature_the_target_has_is_ready() {
    let mut rig = Rig::new();
    let hash = rig.place(A, GuestStage::Vertex, &program(GuestStage::Vertex, &[1]));
    rig.fake.script().needs.insert(
        hash,
        Requirements {
            subgroup: true,
            ..Requirements::default()
        },
    );
    pending(rig.lookup(&[vertex(A)]));
    rig.executor.run_all();
    let shaders = ready(rig.lookup(&[vertex(A)]));
    assert_eq!(
        shaders[0].1.needs,
        Requirements {
            subgroup: true,
            ..Requirements::default()
        }
    );
}

#[test]
fn a_job_dropped_unrun_settles_as_abandoned() {
    let mut rig = Rig::new();
    let hash = rig.place(A, GuestStage::Vertex, &program(GuestStage::Vertex, &[1]));
    let ticket = pending(rig.lookup(&[vertex(A)]));
    rig.executor.drop_all();
    let rejection = rejected(rig.cache.wait_timeout(ticket, WATCHDOG));
    assert_eq!(rejection.hash, Some(hash));
    assert_eq!(*rejection.reason, Rejection::Cache(CacheRefusal::Abandoned));
    assert_eq!(rig.fake.shader_calls(), 0);
}

#[test]
fn a_translator_that_panics_settles_as_abandoned() {
    let mut rig = Rig::new();
    let hash = rig.place(A, GuestStage::Vertex, &program(GuestStage::Vertex, &[1]));
    rig.fake.script().panics.insert(hash);
    let ticket = pending(rig.lookup(&[vertex(A)]));
    assert!(rig.executor.run_next());
    let rejection = rejected(rig.cache.wait_timeout(ticket, WATCHDOG));
    assert_eq!(*rejection.reason, Rejection::Cache(CacheRefusal::Abandoned));
    assert_eq!(rig.fake.shader_calls(), 1);
}

#[test]
fn questions_past_the_round_cap_are_rejected_each_lookup_and_counted_once() {
    let mut rig = Rig::new();
    let hash = rig.place(A, GuestStage::Vertex, &program(GuestStage::Vertex, &[1]));
    rig.fake.script().endless.insert(hash);
    let reason = Rejection::Questions(ExchangeError::TooManyRounds {
        rounds: MAX_QUESTION_ROUNDS,
    });
    for _ in 0..3 {
        let rejection = rejected(rig.lookup(&[vertex(A)]));
        assert_eq!(rejection.hash, Some(hash));
        assert_eq!(*rejection.reason, reason);
    }
    // Asked again each time (it depends on engine state), counted once, never queued.
    let asked = rig
        .state
        .asked
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .len();
    assert_eq!(asked, 3 * MAX_QUESTION_ROUNDS as usize);
    assert_eq!(rig.counters().rejections, 1);
    assert_eq!(rig.counters().programs, 1);
    assert_eq!(rig.executor.queued(), 0);
}

#[test]
fn a_rejection_is_counted_once_and_never_retried_without_a_write() {
    let mut rig = Rig::new();
    let hash = rig.place(A, GuestStage::Vertex, &program(GuestStage::Vertex, &[1]));
    rig.fake
        .script()
        .program_rejections
        .insert(hash, decode_rejection(0));
    for _ in 0..4 {
        rejected(rig.lookup(&[vertex(A)]));
    }
    assert_eq!(rig.fake.program_calls(), 1);
    assert_eq!(rig.counters().captures, 1);
    assert_eq!(rig.counters().rejections, 1);
}

#[test]
fn an_unreadable_program_is_not_read_again_without_a_write() {
    let mut rig = Rig::new();
    let request = vertex(rig.memory.end() - 16);
    for _ in 0..3 {
        rejected(rig.lookup(&[request]));
    }
    assert_eq!(rig.counters().captures, 1);
    assert_eq!(rig.counters().rejections, 1);
}

#[test]
fn a_write_clears_a_negative_capture_entry() {
    let mut rig = Rig::new();
    // A program with no end inside the mapped run: the capture reads to its end and fails.
    let fillers: Vec<u8> = (0..2).flat_map(|tag| filler(tag).to_le_bytes()).collect();
    let at = rig.memory.end() - fillers.len() as u64;
    rig.memory.write(at, &fillers);
    let request = ProgramRequest {
        stage: GuestStage::Compute,
        address: at,
    };
    rejected(rig.lookup(&[request]));
    // The guest writes an `EXIT` over the second filler; the back thread passes the write on.
    rig.memory.write(at + 16, &exit().to_le_bytes());
    rejected(rig.lookup(&[request]));
    assert_eq!(rig.counters().captures, 1);
    rig.cache.invalidate(at + 16..at + 32);
    pending(rig.lookup(&[request]));
    assert_eq!(rig.counters().captures, 2);
    assert_eq!(rig.counters().rejections, 1);
}

#[test]
fn a_write_of_other_bytes_clears_a_rejected_program_and_the_same_bytes_do_not() {
    let mut rig = Rig::new();
    let bytes = program(GuestStage::Vertex, &[1]);
    let hash = rig.place(A, GuestStage::Vertex, &bytes);
    rig.fake
        .script()
        .program_rejections
        .insert(hash, decode_rejection(0));
    rejected(rig.lookup(&[vertex(A)]));

    // The same bytes written back: captured and hashed again, not made again or recounted.
    rig.cache.invalidate(A..A + 1);
    rejected(rig.lookup(&[vertex(A)]));
    assert_eq!(rig.counters().captures, 2);
    assert_eq!(rig.fake.program_calls(), 1);
    assert_eq!(rig.counters().rejections, 1);

    // Other bytes: a new program entry, and this one translates.
    rig.place(A, GuestStage::Vertex, &program(GuestStage::Vertex, &[2]));
    rig.cache.invalidate(A..A + 1);
    pending(rig.lookup(&[vertex(A)]));
    assert_eq!(rig.counters().captures, 3);
    assert_eq!(rig.fake.program_calls(), 2);
}

#[test]
fn the_first_rejected_request_in_order_is_named_and_the_rest_not_looked_at() {
    let mut rig = Rig::new();
    let bad = rig.place(A, GuestStage::Vertex, &program(GuestStage::Vertex, &[1]));
    let also_bad = rig.place(B, GuestStage::Pixel, &program(GuestStage::Pixel, &[2]));
    {
        let mut script = rig.fake.script();
        script.program_rejections.insert(bad, decode_rejection(0));
        script
            .program_rejections
            .insert(also_bad, decode_rejection(1));
    }
    // Both rejected: the first is named, and the second is never captured.
    let rejection = rejected(rig.lookup(&[vertex(A), pixel(B)]));
    assert_eq!(rejection.request, vertex(A));
    assert_eq!(rig.counters().captures, 1);
    // The other order names the other.
    let rejection = rejected(rig.lookup(&[pixel(B), vertex(A)]));
    assert_eq!(rejection.request, pixel(B));
    assert_eq!(*rejection.reason, decode_rejection(1));
}

#[test]
fn a_rejection_after_a_pending_request_wins_and_the_pending_one_stays_queued() {
    let mut rig = Rig::new();
    rig.place(A, GuestStage::Vertex, &program(GuestStage::Vertex, &[1]));
    let bad = rig.place(B, GuestStage::Pixel, &program(GuestStage::Pixel, &[2]));
    rig.fake
        .script()
        .program_rejections
        .insert(bad, decode_rejection(0));
    let rejection = rejected(rig.lookup(&[vertex(A), pixel(B)]));
    assert_eq!(rejection.request, pixel(B));
    assert_eq!(rig.executor.queued(), 1);
    assert!(rig.cache.tickets.is_empty());
}

#[test]
fn a_wait_names_the_first_rejected_member_in_order() {
    let mut rig = Rig::new();
    let first = rig.place(A, GuestStage::Vertex, &program(GuestStage::Vertex, &[1]));
    let second = rig.place(B, GuestStage::Pixel, &program(GuestStage::Pixel, &[2]));
    {
        let mut script = rig.fake.script();
        script.shader_rejections.insert(first, decode_rejection(3));
        script.shader_rejections.insert(second, decode_rejection(4));
    }
    let ticket = pending(rig.lookup(&[vertex(A), pixel(B)]));
    // The second finishes first; the first is then rejected too, and is named.
    assert!(rig.executor.run_last());
    assert!(rig.executor.run_next());
    let rejection = rejected(rig.cache.wait_timeout(ticket, WATCHDOG));
    assert_eq!(rejection.request, vertex(A));
    assert_eq!(*rejection.reason, decode_rejection(3));
}

#[test]
fn a_wait_gives_a_rejection_while_an_earlier_member_is_still_pending() {
    let mut rig = Rig::new();
    rig.place(A, GuestStage::Vertex, &program(GuestStage::Vertex, &[1]));
    let bad = rig.place(B, GuestStage::Pixel, &program(GuestStage::Pixel, &[2]));
    rig.fake
        .script()
        .shader_rejections
        .insert(bad, decode_rejection(5));
    let ticket = pending(rig.lookup(&[vertex(A), pixel(B)]));
    // Only the pixel shader's job runs; the vertex one stays queued.
    assert!(rig.executor.run_last());
    let rejection = rejected(rig.cache.wait_timeout(ticket, WATCHDOG));
    assert_eq!(rejection.request, pixel(B));
    assert_eq!(rig.executor.queued(), 1);
}

#[test]
fn every_refusal_reads_in_plain_words() {
    assert_eq!(
        CacheRefusal::Abandoned.to_string(),
        "the translation stopped before it finished"
    );
    assert_eq!(
        CacheRefusal::Unsupported {
            missing: Requirements::default()
        }
        .to_string(),
        "the shader needs no device features, which the host lacks"
    );
    assert_eq!(
        CacheRefusal::Full { limit: 3 }.to_string(),
        "the shader cache is full: all 3 of its shader entries are translations in progress"
    );
    assert_eq!(
        CacheRefusal::OverBudget {
            charged: 0,
            needed: 5,
            limit: 4
        }
        .to_string(),
        "the shader cache is full: its translations in progress are charged 0 of its 4 bytes, \
         and this shader needs 5 more"
    );
    assert_eq!(
        CacheRefusal::Unsupported {
            missing: Requirements {
                subgroup: true,
                subgroup_size_32: true,
                f16: true,
                f64: true,
                i64: true,
                i64_atomics: true,
            }
        }
        .to_string(),
        "the shader needs subgroup operations, a subgroup size of 32, 16-bit floats, 64-bit \
         floats, 64-bit integers, 64-bit integer atomics, which the host lacks"
    );
    assert_eq!(
        CacheRefusal::Unsupported {
            missing: Requirements {
                subgroup_size_32: true,
                ..Requirements::default()
            }
        }
        .to_string(),
        "the shader needs a subgroup size of 32, which the host lacks"
    );
    assert_eq!(
        Rejection::Cache(CacheRefusal::Abandoned).to_string(),
        "the translation stopped before it finished"
    );
    assert_eq!(
        Rejection::Capture(CaptureError::NoEnd).to_string(),
        "no instruction within the first 65536 ends the program"
    );
}

/// Divergence (module docs, "The negative cache"): a capture refused because memory past the
/// mapped run was unreadable stays refused when that memory is mapped later with no write the
/// cache is told of; only an invalidation of the bytes it tried clears it.
#[test]
fn a_refused_capture_stays_refused_when_its_memory_is_mapped_without_a_write() {
    let mut rig = Rig::new();
    let fillers: Vec<u8> = (0..2).flat_map(|tag| filler(tag).to_le_bytes()).collect();
    let mapped_end = rig.memory.end();
    let at = mapped_end - fillers.len() as u64;
    rig.memory.write(at, &fillers);
    let request = ProgramRequest {
        stage: GuestStage::Compute,
        address: at,
    };
    let first = rejected(rig.lookup(&[request]));
    assert_eq!(
        *first.reason,
        Rejection::Capture(CaptureError::InstructionUnreadable {
            index: 2,
            offset: fillers.len() as u64,
        })
    );
    // The guest maps the next bytes, already holding an `EXIT`; no write reaches the cache.
    rig.memory.bytes.extend_from_slice(&exit().to_le_bytes());
    let again = rejected(rig.lookup(&[request]));
    assert_eq!(again.reason, first.reason);
    assert_eq!(rig.counters().captures, 1);
    // An invalidation of the first byte the capture tried clears it.
    rig.cache.invalidate(mapped_end..mapped_end + 1);
    pending(rig.lookup(&[request]));
    assert_eq!(rig.counters().captures, 2);
}

#[test]
fn each_missing_feature_is_found_alone_and_a_supported_one_is_not() {
    type Field = fn(&mut Requirements);
    let fields: [(&str, Field); 5] = [
        ("subgroup operations", |r| r.subgroup = true),
        ("16-bit floats", |r| r.f16 = true),
        ("64-bit floats", |r| r.f64 = true),
        ("64-bit integers", |r| r.i64 = true),
        ("64-bit integer atomics", |r| r.i64_atomics = true),
    ];
    for (name, set) in fields {
        let mut one = Requirements::default();
        set(&mut one);
        assert_eq!(super::super::feature_names(one), name);
        assert_eq!(
            super::super::missing_features(one, Requirements::default()),
            Some(one),
            "{name} lacked"
        );
        assert_eq!(
            super::super::missing_features(one, one),
            None,
            "{name} supported"
        );
    }
}
