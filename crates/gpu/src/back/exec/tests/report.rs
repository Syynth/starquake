//! Tests of the 3D class's report semaphore through the executor (`back::report`): a payload
//! report is a fenced release of the channel's address space, and a counter report writes
//! nothing.

use super::*;
use crate::back::fence::tests::TIMESTAMP;
use crate::sema::RELEASE_TIMESTAMP_OFFSET;

/// The payload every test releases: non-zero in each byte, so a short write shows.
const PAYLOAD: u32 = 0xa1b2_c3d4;

/// A `SET_REPORT_SEMAPHORE_D` word: a release with these `REPORT` and `STRUCTURE_SIZE` fields.
fn query(report: u32, size: u32) -> u32 {
    fields(&[
        (
            clc797::SET_REPORT_SEMAPHORE_D_OPERATION,
            clc797::SET_REPORT_SEMAPHORE_D_OPERATION_RELEASE,
        ),
        (clc797::SET_REPORT_SEMAPHORE_D_REPORT, report),
        (clc797::SET_REPORT_SEMAPHORE_D_STRUCTURE_SIZE, size),
    ])
}

/// The ring text of a report on `channel`: the address, the payload, then the query word.
fn report_on(channel: ChannelId, semaphore: u64, payload: u32, query: u32) -> String {
    let c = THREE_D;
    let upper = u32::try_from(semaphore >> 32).unwrap();
    let lower = u32::try_from(semaphore & 0xffff_ffff).unwrap();
    write_on(channel, c, &clc797::SET_REPORT_SEMAPHORE_A, &[], upper)
        + &write_on(channel, c, &clc797::SET_REPORT_SEMAPHORE_B, &[], lower)
        + &write_on(channel, c, &clc797::SET_REPORT_SEMAPHORE_C, &[], payload)
        + &write_on(channel, c, &clc797::SET_REPORT_SEMAPHORE_D, &[], query)
}

fn one_word(semaphore: u64, payload: u32) -> String {
    let word = query(
        clc797::SET_REPORT_SEMAPHORE_D_REPORT_NONE,
        clc797::SET_REPORT_SEMAPHORE_D_STRUCTURE_SIZE_ONE_WORD,
    );
    report_on(CHANNEL, semaphore, payload, word)
}

const BIND: &str = "bind channel=3 subchannel=0 class=0xc797\n";

#[test]
fn a_one_word_payload_report_writes_four_bytes() {
    let mut rig = Rig::new();
    // The words around the semaphore hold something else, so a write past 4 bytes shows.
    rig.write_words(SEM - 4, &[0x1111_1111, 0x2222_2222, 0x3333_3333]);
    rig.run(&(String::from(BIND) + &one_word(SEM, PAYLOAD)));
    assert_eq!(
        (rig.word(SEM - 4), rig.word(SEM), rig.word(SEM + 4)),
        (0x1111_1111, PAYLOAD, 0x3333_3333)
    );
    assert_eq!(rig.exec.fences().counters().failed_releases, 0);
}

#[test]
fn a_four_word_payload_report_writes_the_payload_and_the_timestamp() {
    let mut rig = Rig::new();
    rig.write_words(SEM - 4, &[0x1111_1111, 0, 0x2222_2222, 0, 0, 0x3333_3333]);
    let word = query(
        clc797::SET_REPORT_SEMAPHORE_D_REPORT_NONE,
        clc797::SET_REPORT_SEMAPHORE_D_STRUCTURE_SIZE_FOUR_WORDS,
    );
    rig.run(&(String::from(BIND) + &report_on(CHANNEL, SEM, PAYLOAD, word)));
    let timestamp = SEM + RELEASE_TIMESTAMP_OFFSET;
    assert_eq!(
        [
            rig.word(SEM - 4),
            rig.word(SEM),
            rig.word(SEM + 4),
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
fn a_report_semaphore_release_waits_for_the_work_before_it() {
    // Both forms are fenced: neither is written before the list holding the draw before it
    // completes, and a later report with no work before it rides the same list.
    for size in [
        clc797::SET_REPORT_SEMAPHORE_D_STRUCTURE_SIZE_ONE_WORD,
        clc797::SET_REPORT_SEMAPHORE_D_STRUCTURE_SIZE_FOUR_WORDS,
    ] {
        let mut rig = Rig::late();
        let word = query(clc797::SET_REPORT_SEMAPHORE_D_REPORT_NONE, size);
        rig.run(&(draw_state() + &draw() + &report_on(CHANNEL, SEM, PAYLOAD, word)));
        rig.exec.close(CloseReason::RingEmpty).unwrap();
        let late = rig.late.clone().unwrap();
        assert_eq!(late.submitted(), vec![FenceValue::new(1)]);
        rig.run(&report_on(CHANNEL, SEM2, 7, word));
        rig.exec.complete();
        assert_eq!((rig.word(SEM), rig.word(SEM2)), (0, 0), "size {size}");
        late.finish(FenceValue::new(1));
        rig.exec.complete();
        assert_eq!((rig.word(SEM), rig.word(SEM2)), (PAYLOAD, 7), "size {size}");
    }
}

#[test]
fn each_write_of_the_query_word_releases_again() {
    // The word is a trigger: the same word written twice releases twice, with the payload the
    // register holds at each write.
    let mut rig = Rig::new();
    let c = THREE_D;
    let word = query(
        clc797::SET_REPORT_SEMAPHORE_D_REPORT_NONE,
        clc797::SET_REPORT_SEMAPHORE_D_STRUCTURE_SIZE_ONE_WORD,
    );
    rig.run(&(String::from(BIND) + &one_word(SEM, 1)));
    assert_eq!(rig.word(SEM), 1);
    rig.run(
        &(write(c, &clc797::SET_REPORT_SEMAPHORE_C, &[], 2)
            + &write(c, &clc797::SET_REPORT_SEMAPHORE_D, &[], word)),
    );
    assert_eq!(rig.word(SEM), 2);
    // The other three registers alone release nothing.
    rig.write_words(SEM, &[0]);
    rig.run(
        &(address(
            c,
            &clc797::SET_REPORT_SEMAPHORE_A,
            &clc797::SET_REPORT_SEMAPHORE_B,
            &[],
            SEM,
        ) + &write(c, &clc797::SET_REPORT_SEMAPHORE_C, &[], 3)),
    );
    assert_eq!(rig.word(SEM), 0);
}

#[test]
fn a_report_is_released_through_its_channels_address_space() {
    // Channel 5 lives in a second space whose memory maps SEM elsewhere: the release must land
    // there, not in the first space's memory.
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
    let word = query(
        clc797::SET_REPORT_SEMAPHORE_D_REPORT_NONE,
        clc797::SET_REPORT_SEMAPHORE_D_STRUCTURE_SIZE_ONE_WORD,
    );
    rig.run(
        &(String::from("bind channel=5 subchannel=0 class=0xc797\n")
            + &report_on(channel, SEM, PAYLOAD, word)),
    );
    let mut bytes = [0; 4];
    other_mem.read(SEM, &mut bytes).unwrap();
    assert_eq!(u32::from_le_bytes(bytes), PAYLOAD);
    assert_eq!(rig.word(SEM), 0);
}

#[test]
fn a_counter_report_writes_nothing_and_is_logged_once() {
    let mut rig = Rig::new();
    rig.run(BIND);
    let word = query(
        clc797::SET_REPORT_SEMAPHORE_D_REPORT_ZPASS_PIXEL_CNT64,
        clc797::SET_REPORT_SEMAPHORE_D_STRUCTURE_SIZE_ONE_WORD,
    );
    let ((), events) = captured(|| {
        rig.run(&report_on(CHANNEL, SEM, PAYLOAD, word));
        rig.run(&report_on(CHANNEL, SEM2, PAYLOAD, word));
    });
    assert_eq!((rig.word(SEM), rig.word(SEM2)), (0, 0));
    let warnings: Vec<String> = at(&events, tracing::Level::WARN)
        .into_iter()
        .filter(|e| e.contains("a counter report was asked for and is not written"))
        .collect();
    assert_eq!(warnings.len(), 1, "{events:?}");
    // A payload report after it is still released.
    rig.run(&one_word(SEM, PAYLOAD));
    assert_eq!(rig.word(SEM), PAYLOAD);
}

#[test]
fn an_acquire_or_a_trap_writes_nothing() {
    for operation in [
        clc797::SET_REPORT_SEMAPHORE_D_OPERATION_ACQUIRE,
        clc797::SET_REPORT_SEMAPHORE_D_OPERATION_TRAP,
    ] {
        let mut rig = Rig::new();
        let word = fields(&[
            (clc797::SET_REPORT_SEMAPHORE_D_OPERATION, operation),
            (
                clc797::SET_REPORT_SEMAPHORE_D_STRUCTURE_SIZE,
                clc797::SET_REPORT_SEMAPHORE_D_STRUCTURE_SIZE_ONE_WORD,
            ),
        ]);
        rig.run(&(String::from(BIND) + &report_on(CHANNEL, SEM, PAYLOAD, word)));
        assert_eq!(rig.word(SEM), 0, "operation {operation}");
    }
}

#[test]
fn a_report_to_an_unmapped_address_is_a_failed_release_and_the_stream_goes_on() {
    // The native SDK writes a payload report to address 0. The entry is accepted, the release
    // fails when it is performed, and the next report is written.
    let mut rig = Rig::new();
    let ((), events) = captured(|| {
        rig.run(&(String::from(BIND) + &one_word(0, PAYLOAD) + &one_word(SEM, PAYLOAD)));
    });
    assert_eq!(rig.exec.fences().counters().failed_releases, 1);
    assert_eq!(rig.word(SEM), PAYLOAD);
    assert!(
        at(&events, tracing::Level::WARN)
            .iter()
            .any(|e| e.contains("fenced release failed")),
        "{events:?}"
    );
}

#[test]
fn deferred_instances_are_drawn_before_a_report_semaphore_release() {
    let mut rig = Rig::late();
    let c = THREE_D;
    // BEGIN FIRST ... END, then BEGIN SUBSEQUENT ... END: the instances wait.
    let instanced = write(c, &clc797::SET_VERTEX_ARRAY_START, &[], 0)
        + &write(c, &clc797::DRAW_VERTEX_ARRAY, &[], 3)
        + &write(c, &clc797::BEGIN, &[], clc797::BEGIN_OP_TRIANGLES)
        + &write(c, &clc797::END, &[], 0)
        + &write(
            c,
            &clc797::BEGIN,
            &[],
            fields(&[
                (clc797::BEGIN_OP, clc797::BEGIN_OP_TRIANGLES),
                (
                    clc797::BEGIN_INSTANCE_ID,
                    clc797::BEGIN_INSTANCE_ID_SUBSEQUENT,
                ),
            ]),
        )
        + &write(c, &clc797::END, &[], 0);
    rig.run(&(draw_state() + &instanced));
    assert!(rig.exec.draws(CHANNEL).unwrap().has_deferred());
    // The address and the payload leave the instances waiting; the query word draws them.
    let report = one_word(SEM, PAYLOAD);
    let (state, trigger) = report.split_at(report.trim_end().rfind('\n').unwrap() + 1);
    rig.run(state);
    assert!(rig.exec.draws(CHANNEL).unwrap().has_deferred());
    rig.run(trigger);
    assert!(!rig.exec.draws(CHANNEL).unwrap().has_deferred());
    // The release is behind the draw: it waits for the list that holds it.
    rig.exec.close(CloseReason::RingEmpty).unwrap();
    rig.exec.complete();
    assert_eq!(rig.word(SEM), 0);
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
    assert_eq!(rig.word(SEM), PAYLOAD);
}
