//! Tests of the compute class's report semaphore through the executor (`back::report`,
//! "The compute class"): a release or report-only word is a fenced release of the channel's
//! address space, sharing the 3D class's path; an acquire or a trap writes nothing.

use super::*;
use crate::back::fence::tests::{REFUSED, TIMESTAMP};
use crate::back::fence::{FenceError, Waiter};
use crate::sema::AcquireOp;
use crate::sema::RELEASE_TIMESTAMP_OFFSET;

/// The payload every test releases: non-zero in each byte, so a short write shows.
const PAYLOAD: u32 = 0xa1b2_c3d4;

/// Semaphores away from [`SEM`] and [`SEM2`], which [`compute_qmd`]'s own releases write.
const CSEM: u64 = REPORT + 0x40;
const CSEM2: u64 = REPORT + 0x80;

const BIND: &str = "bind channel=3 subchannel=1 class=0xc7c0\n";

/// A compute `SET_REPORT_SEMAPHORE_D` word with these `OPERATION` and `STRUCTURE_SIZE` fields.
fn query(operation: u32, size: u32) -> u32 {
    fields(&[
        (clc7c0::SET_REPORT_SEMAPHORE_D_OPERATION, operation),
        (clc7c0::SET_REPORT_SEMAPHORE_D_STRUCTURE_SIZE, size),
    ])
}

fn release_word(size: u32) -> u32 {
    query(clc7c0::SET_REPORT_SEMAPHORE_D_OPERATION_RELEASE, size)
}

/// The ring text of a compute report on `channel`: the address, the payload, then the query
/// word.
fn report_on(channel: ChannelId, semaphore: u64, payload: u32, query: u32) -> String {
    let c = COMPUTE;
    let upper = u32::try_from(semaphore >> 32).unwrap();
    let lower = u32::try_from(semaphore & 0xffff_ffff).unwrap();
    write_on(channel, c, &clc7c0::SET_REPORT_SEMAPHORE_A, &[], upper)
        + &write_on(channel, c, &clc7c0::SET_REPORT_SEMAPHORE_B, &[], lower)
        + &write_on(channel, c, &clc7c0::SET_REPORT_SEMAPHORE_C, &[], payload)
        + &write_on(channel, c, &clc7c0::SET_REPORT_SEMAPHORE_D, &[], query)
}

fn one_word(semaphore: u64, payload: u32) -> String {
    let word = release_word(clc7c0::SET_REPORT_SEMAPHORE_D_STRUCTURE_SIZE_ONE_WORD);
    report_on(CHANNEL, semaphore, payload, word)
}

/// `text`, the ring text of a report, split before its last line: the address and the payload,
/// then the query word that triggers the release.
fn split_trigger(text: &str) -> (&str, &str) {
    text.split_at(text.trim_end().rfind('\n').unwrap() + 1)
}

#[test]
fn a_one_word_compute_report_writes_four_bytes() {
    let mut rig = Rig::new();
    // The words around the semaphore hold something else, so a write past 4 bytes shows.
    rig.write_words(CSEM - 4, &[0x1111_1111, 0x2222_2222, 0x3333_3333]);
    rig.run(&(String::from(BIND) + &one_word(CSEM, PAYLOAD)));
    assert_eq!(
        (rig.word(CSEM - 4), rig.word(CSEM), rig.word(CSEM + 4)),
        (0x1111_1111, PAYLOAD, 0x3333_3333)
    );
    assert_eq!(rig.exec.fences().counters().failed_releases, 0);
}

#[test]
fn a_four_word_compute_report_writes_the_payload_and_the_timestamp() {
    let mut rig = Rig::new();
    rig.write_words(CSEM - 4, &[0x1111_1111, 0, 0x2222_2222, 0, 0, 0x3333_3333]);
    let word = release_word(clc7c0::SET_REPORT_SEMAPHORE_D_STRUCTURE_SIZE_FOUR_WORDS);
    rig.run(&(String::from(BIND) + &report_on(CHANNEL, CSEM, PAYLOAD, word)));
    let timestamp = CSEM + RELEASE_TIMESTAMP_OFFSET;
    assert_eq!(
        [
            rig.word(CSEM - 4),
            rig.word(CSEM),
            rig.word(CSEM + 4),
            rig.word(timestamp),
            rig.word(timestamp + 4),
            rig.word(timestamp + 8),
        ],
        [
            0x1111_1111,
            PAYLOAD,
            // The payload is widened to 64 bits.
            0,
            u32::try_from(TIMESTAMP & 0xffff_ffff).unwrap(),
            u32::try_from(TIMESTAMP >> 32).unwrap(),
            0x3333_3333,
        ]
    );
}

#[test]
fn a_compute_report_waits_for_the_dispatch_before_it() {
    // Both forms are fenced: neither is written before the list holding the dispatch before
    // it completes, and a later report with no work before it rides the same list.
    for size in [
        clc7c0::SET_REPORT_SEMAPHORE_D_STRUCTURE_SIZE_ONE_WORD,
        clc7c0::SET_REPORT_SEMAPHORE_D_STRUCTURE_SIZE_FOUR_WORDS,
    ] {
        let mut rig = Rig::late();
        rig.write_words(QMD, &compute_qmd());
        let word = release_word(size);
        rig.run(&(compute_signal() + &report_on(CHANNEL, CSEM, PAYLOAD, word)));
        assert_eq!(rig.exec.counters().dispatches, 1);
        rig.exec.close(CloseReason::RingEmpty).unwrap();
        let late = rig.late.clone().unwrap();
        assert_eq!(late.submitted(), vec![FenceValue::new(1)]);
        rig.run(&report_on(CHANNEL, CSEM2, 7, word));
        rig.exec.complete();
        assert_eq!((rig.word(CSEM), rig.word(CSEM2)), (0, 0), "size {size}");
        late.finish(FenceValue::new(1));
        rig.exec.complete();
        assert_eq!(
            (rig.word(CSEM), rig.word(CSEM2)),
            (PAYLOAD, 7),
            "size {size}"
        );
    }
}

#[test]
fn each_write_of_the_compute_query_word_releases_again() {
    // The word is a trigger: the same word written twice releases twice, with the payload the
    // register holds at each write.
    let mut rig = Rig::new();
    let c = COMPUTE;
    let word = release_word(clc7c0::SET_REPORT_SEMAPHORE_D_STRUCTURE_SIZE_ONE_WORD);
    rig.run(&(String::from(BIND) + &one_word(CSEM, 1)));
    assert_eq!(rig.word(CSEM), 1);
    rig.run(
        &(write(c, &clc7c0::SET_REPORT_SEMAPHORE_C, &[], 2)
            + &write(c, &clc7c0::SET_REPORT_SEMAPHORE_D, &[], word)),
    );
    assert_eq!(rig.word(CSEM), 2);
    // The other three registers alone release nothing.
    rig.write_words(CSEM, &[0]);
    rig.run(
        &(address(
            c,
            &clc7c0::SET_REPORT_SEMAPHORE_A,
            &clc7c0::SET_REPORT_SEMAPHORE_B,
            &[],
            CSEM,
        ) + &write(c, &clc7c0::SET_REPORT_SEMAPHORE_C, &[], 3)),
    );
    assert_eq!(rig.word(CSEM), 0);
}

#[test]
fn a_compute_report_is_released_through_its_channels_address_space() {
    // Channel 5 lives in a second space whose memory maps CSEM elsewhere: the release must
    // land there, not in the first space's memory.
    let mut rig = Rig::new();
    let second = AsId(SPACE.0 + 1);
    let other_mem = memory();
    rig.exec
        .add_space(
            second,
            Arc::<Memory>::clone(&other_mem),
            Arc::new(WriteWatch::new()),
        )
        .unwrap();
    let channel = ChannelId(5);
    rig.exec.add_channel(channel, second).unwrap();
    let word = release_word(clc7c0::SET_REPORT_SEMAPHORE_D_STRUCTURE_SIZE_ONE_WORD);
    rig.run(
        &(String::from("bind channel=5 subchannel=1 class=0xc7c0\n")
            + &report_on(channel, CSEM, PAYLOAD, word)),
    );
    let mut bytes = [0; 4];
    other_mem.read(CSEM, &mut bytes).unwrap();
    assert_eq!(u32::from_le_bytes(bytes), PAYLOAD);
    assert_eq!(rig.word(CSEM), 0);
}

#[test]
fn a_release_or_report_only_releases_and_an_acquire_or_trap_writes_nothing() {
    for (operation, expected) in [
        (clc7c0::SET_REPORT_SEMAPHORE_D_OPERATION_RELEASE, PAYLOAD),
        (clc7c0::REPORT_SEMAPHORE_EXECUTE_OPERATION_ACQUIRE, 0),
        (
            clc7c0::REPORT_SEMAPHORE_EXECUTE_OPERATION_REPORT_ONLY,
            PAYLOAD,
        ),
        (clc7c0::SET_REPORT_SEMAPHORE_D_OPERATION_TRAP, 0),
    ] {
        let mut rig = Rig::new();
        let word = query(
            operation,
            clc7c0::SET_REPORT_SEMAPHORE_D_STRUCTURE_SIZE_ONE_WORD,
        );
        rig.run(&(String::from(BIND) + &report_on(CHANNEL, CSEM, PAYLOAD, word)));
        assert_eq!(rig.word(CSEM), expected, "operation {operation}");
    }
}

#[test]
fn a_compute_report_writes_the_payload_whatever_its_reduction_and_upper_bits() {
    // REDUCTION_ENABLE with RED_MIN, and bits 27:23 (the 3D class's `REPORT`) all set. The
    // payload is written over the old value, not reduced with it (Eden-NX2 would write the
    // smaller, 5), and no counter report is logged: the compute class has no counter.
    let mut rig = Rig::new();
    rig.write_words(CSEM, &[5]);
    let word = fields(&[
        (
            clc7c0::SET_REPORT_SEMAPHORE_D_OPERATION,
            clc7c0::SET_REPORT_SEMAPHORE_D_OPERATION_RELEASE,
        ),
        (
            clc7c0::SET_REPORT_SEMAPHORE_D_STRUCTURE_SIZE,
            clc7c0::SET_REPORT_SEMAPHORE_D_STRUCTURE_SIZE_ONE_WORD,
        ),
        (
            clc7c0::SET_REPORT_SEMAPHORE_D_REDUCTION_ENABLE,
            clc7c0::SET_REPORT_SEMAPHORE_D_REDUCTION_ENABLE_TRUE,
        ),
        (
            clc7c0::SET_REPORT_SEMAPHORE_D_REDUCTION_OP,
            clc7c0::SET_REPORT_SEMAPHORE_D_REDUCTION_OP_RED_MIN,
        ),
        (clc797::SET_REPORT_SEMAPHORE_D_REPORT, u32::MAX >> 27),
    ]);
    let ((), events) = captured(|| {
        rig.run(&(String::from(BIND) + &report_on(CHANNEL, CSEM, PAYLOAD, word)));
    });
    assert_eq!(rig.word(CSEM), PAYLOAD);
    assert!(
        !events
            .iter()
            .any(|(_, e)| e.contains("a counter report was asked for")),
        "{events:?}"
    );
}

#[test]
fn deferred_instances_are_drawn_before_a_compute_report_release() {
    // The channel's 3D instances, drawn after the address and payload words, wait; the compute
    // class's query word draws them first (every word of a class other than 3D does, exec
    // module docs, "Deferred instanced draws"), and its release waits for the list that holds
    // them.
    let mut rig = Rig::late();
    // Both classes are bound first: a bind is itself a boundary that draws the instances.
    rig.run(&(String::from(BIND) + &draw_state()));
    let report = one_word(CSEM, PAYLOAD);
    let (state, trigger) = split_trigger(&report);
    rig.run(state);
    rig.run(&instanced(clc797::BEGIN_OP_TRIANGLES));
    assert!(rig.exec.draws(CHANNEL).unwrap().has_deferred());
    rig.run(trigger);
    assert!(!rig.exec.draws(CHANNEL).unwrap().has_deferred());
    rig.exec.close(CloseReason::RingEmpty).unwrap();
    rig.exec.complete();
    assert_eq!(rig.word(CSEM), 0);
    let late = rig.late.clone().unwrap();
    let lists = late.inner.lists();
    let (_, list) = lists.last().unwrap();
    assert!(
        matches!(
            list.commands.last(),
            Some(Command::Draw(desc)) if matches!(
                desc.params,
                starquake_render::command::DrawParams::Direct {
                    instance_count: 2,
                    ..
                }
            )
        ),
        "{list:?}"
    );
    late.finish(FenceValue::new(1));
    rig.exec.complete();
    assert_eq!(rig.word(CSEM), PAYLOAD);
}

#[test]
fn a_failed_deferred_draw_before_a_compute_report_still_releases_and_returns_the_draws_error() {
    // The deferred instances have a topology the draw path refuses only when it draws them.
    // The query word still attaches its release, which is written, and returns the draw's
    // error (exec module docs, "Deferred instanced draws").
    let mut rig = Rig::new();
    rig.run(&(String::from(BIND) + &draw_state()));
    let report = one_word(CSEM, PAYLOAD);
    let (state, trigger) = split_trigger(&report);
    rig.run(state);
    rig.run(&instanced(UNDEFINED_OP));
    assert!(rig.exec.draws(CHANNEL).unwrap().has_deferred());
    let fenced = rig.exec.fences().counters().fenced;
    let trigger = parse_text(trigger).unwrap();
    assert_eq!(
        rig.exec.apply(&trigger[0]),
        Err(ExecError::Draw(DrawError::Topology {
            value: UNDEFINED_OP
        }))
    );
    assert!(!rig.exec.draws(CHANNEL).unwrap().has_deferred());
    assert_eq!(rig.exec.fences().counters().fenced - fenced, 1);
    rig.submit();
    rig.exec.complete();
    assert_eq!(rig.word(CSEM), PAYLOAD);
}

#[test]
fn a_compute_report_returns_the_submit_error_of_the_list_its_waiter_closes() {
    // The compute report's own error, with no deferred draw to supply one: a dispatch is in
    // the open list, another channel is parked on the semaphore, so attaching the release
    // closes the list for the waiter (`back::fence`, the close trigger), and the backend
    // refuses that submit. The query word returns the refusal; the refused list, with nothing
    // in flight, still performs its release.
    let mut rig = Rig::late();
    rig.write_words(QMD, &compute_qmd());
    rig.run(&compute_signal());
    assert_eq!(rig.exec.counters().dispatches, 1);
    rig.waiters.register(Waiter {
        channel: OTHER,
        space: SPACE,
        address: CSEM,
        op: AcquireOp::Equal,
        payload: PAYLOAD,
    });
    let report = one_word(CSEM, PAYLOAD);
    let (state, trigger) = split_trigger(&report);
    rig.run(state);
    rig.late.clone().unwrap().refuse_next();
    let trigger = parse_text(trigger).unwrap();
    assert_eq!(
        rig.exec.apply(&trigger[0]),
        Err(ExecError::Fence(FenceError::Submit(REFUSED)))
    );
    assert_eq!(rig.word(CSEM), PAYLOAD);
    assert!(rig.waiters.is_latched(OTHER));
}

#[test]
fn a_report_to_address_0_is_logged_once_per_channel_across_both_classes() {
    // Two compute reports to address 0 on channel 3, then a 3D one on the same channel, then
    // one compute report on channel 9: each fails and is counted; the first to address 0 on
    // each channel warns, whichever class wrote it, and the later ones are at debug level.
    let mut rig = Rig::new();
    let three_d = "bind channel=3 subchannel=0 class=0xc797\n".to_owned()
        + &write(THREE_D, &clc797::SET_REPORT_SEMAPHORE_A, &[], 0)
        + &write(THREE_D, &clc797::SET_REPORT_SEMAPHORE_B, &[], 0)
        + &write(THREE_D, &clc797::SET_REPORT_SEMAPHORE_C, &[], PAYLOAD)
        + &write(
            THREE_D,
            &clc797::SET_REPORT_SEMAPHORE_D,
            &[],
            fields(&[(
                clc797::SET_REPORT_SEMAPHORE_D_STRUCTURE_SIZE,
                clc797::SET_REPORT_SEMAPHORE_D_STRUCTURE_SIZE_ONE_WORD,
            )]),
        );
    let other = "bind channel=9 subchannel=1 class=0xc7c0\n".to_owned()
        + &one_word(0, PAYLOAD).replace("channel=3 ", "channel=9 ");
    let ((), events) = captured(|| {
        rig.run(&(String::from(BIND) + &one_word(0, PAYLOAD).repeat(2)));
        rig.run(&three_d);
        rig.run(&other);
    });
    assert_eq!(rig.exec.fences().counters().failed_releases, 4);
    let failed = |level| {
        at(&events, level)
            .iter()
            .filter(|e| e.contains("fenced release failed") && e.contains("address 0x0 "))
            .count()
    };
    assert_eq!(
        (failed(tracing::Level::WARN), failed(tracing::Level::DEBUG)),
        (2, 2),
        "{events:?}"
    );
}
