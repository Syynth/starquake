//! What a channel waits on: acquires (the waiter table, latching, the poll), submission fences,
//! `SYNC_WAIT` entries and macro catch-ups; and the increments a submission appends.

use std::sync::atomic::AtomicU32;

use super::*;
use crate::interface::Fence;
use crate::mme::testing::{finish, inst0, mov_reg, with_imm, with_out};
use crate::mme::{AluOp, Instruction, LOAD_BARRIER, OutOp, Reg, encode_program};
use crate::syncpoint::SIGN_BIT;

/// A stream that binds 3D, acquires `SEM` with `op` and `payload`, then writes `value` to
/// scratch `index`.
fn acquire_then_write(op: AcquireOp, payload: u32, index: u32, value: u32) -> Vec<u32> {
    let mut pb = PushBuffer::new();
    bind_3d(&mut pb);
    acquire(&mut pb, SEM, op, payload);
    pb.methods(MethodMode::Incrementing, 0, scratch(index), &[value])
        .unwrap();
    pb.into_words()
}

/// Turns the front until a channel has parked (at most a few turns).
fn turn_until_parked(r: &mut Rig) {
    for _ in 0..8 {
        if r.front.counters().parks > 0 {
            return;
        }
        r.turn();
    }
    assert!(r.front.counters().parks > 0, "no channel parked");
}

// ---------------------------------------------------------------------------------------------
// Acquires
// ---------------------------------------------------------------------------------------------

#[test]
fn a_parked_channel_registers_its_waiter_with_its_own_address_space() {
    let mut r = rig();
    let a = r.channel(5);
    r.submit(
        a,
        vec![r.entry(CMD_A, &acquire_then_write(AcquireOp::Equal, 1, 0, 0xa1))],
    );
    r.settle();
    assert_eq!(r.front.counters().parks, 1);
    assert!(r.waiters.has_waiter(SPACE, SEM));
    assert!(!r.waiters.has_waiter(OTHER_SPACE, SEM));
    assert!(!r.waiters.has_waiter(SPACE, SEM2));
    assert_eq!(r.waiters.len(), 1);
    // A release of the acquire's value in the channel's space latches it: the channel resumes
    // and its waiter is removed.
    assert!(r.waiters.on_write(SPACE, SEM, 1));
    r.write(SEM, &[1]);
    crate::back::fence::FrontWake::wake(
        &*r.shared,
        &crate::back::fence::Report {
            latched: true,
            ..crate::back::fence::Report::default()
        },
    );
    r.settle();
    assert!(has(&r.take(), &scratch_text(a, 0, 0xa1)));
    assert!(r.waiters.is_empty());
}

#[test]
fn an_equal_acquire_whose_value_moves_on_is_still_latched() {
    // Row 70: channel A waits for SEM == 5. Channel B releases 5 and then 6; by the time the
    // front looks, the word reads 6, but the first release latched A's waiter, so A resumes.
    let mut r = rig();
    let a = r.channel(5);
    let b = r.channel(6);
    r.submit(
        a,
        vec![r.entry(CMD_A, &acquire_then_write(AcquireOp::Equal, 5, 0, 0xa1))],
    );
    r.settle();
    assert_eq!(r.front.counters().parks, 1);
    let mut pb = PushBuffer::new();
    release(&mut pb, SEM, 5);
    release(&mut pb, SEM, 6);
    r.submit(b, vec![r.entry(CMD_B, pb.words())]);
    r.settle();
    assert_eq!(r.read_word(SEM), 6);
    assert!(has(&r.take(), &scratch_text(a, 0, 0xa1)));
    assert!(r.waiters.is_empty());
}

#[test]
fn a_value_that_moves_on_with_no_release_is_missed() {
    // The contrast: the CPU writes 5 and then 6 between two looks (no release, nothing
    // latched); the poll reads 6, and the channel stays parked.
    let mut r = rig();
    let a = r.channel(5);
    r.submit(
        a,
        vec![r.entry(CMD_A, &acquire_then_write(AcquireOp::Equal, 5, 0, 0xa1))],
    );
    r.settle();
    r.write(SEM, &[5]);
    r.write(SEM, &[6]);
    let polls = r.front.counters().polls;
    r.settle();
    assert!(r.front.counters().polls > polls);
    assert!(!has(&r.take(), &scratch_text(a, 0, 0xa1)));
    assert!(r.waiters.has_waiter(SPACE, SEM));
}

/// What [`Hooked`] runs before the second read of `SEM`.
type Hook = Box<dyn FnOnce(&Memory) + Send>;

/// A memory that, on the second read of `SEM`, first runs a hook: a release performed between
/// the front's two reads.
struct Hooked {
    inner: Arc<Memory>,
    reads: AtomicU32,
    hook: Mutex<Option<Hook>>,
}

impl GpuMemory for Hooked {
    fn read(&self, va: u64, buf: &mut [u8]) -> Result<(), crate::memory::GpuFault> {
        if va == SEM && self.reads.fetch_add(1, Ordering::SeqCst) == 1 {
            let hook = self.hook.lock().unwrap().take();
            if let Some(hook) = hook {
                hook(&self.inner);
            }
        }
        self.inner.read(va, buf)
    }

    fn write(&self, va: u64, data: &[u8]) -> Result<(), crate::memory::GpuFault> {
        self.inner.write(va, data)
    }

    fn generation(&self) -> u64 {
        self.inner.generation()
    }
}

#[test]
fn a_release_between_registering_and_the_second_read_is_not_missed() {
    // Spec, "Semaphore acquires", step 1: the waiter is registered before the second read. A
    // release of 5 that lands just before that read latches it, and the word has already moved
    // on to 6 when it is read; the channel resumes in the same turn, without parking.
    let slot: Arc<Mutex<Option<Arc<Hooked>>>> = Arc::default();
    let seen = Arc::clone(&slot);
    let mut r = rig_with(RING_WORDS_FOR_TESTS, move |mem| {
        let hooked = Arc::new(Hooked {
            inner: mem,
            reads: AtomicU32::new(0),
            hook: Mutex::new(None),
        });
        *seen.lock().unwrap() = Some(Arc::clone(&hooked));
        hooked
    });
    let hooked = slot.lock().unwrap().clone().unwrap();
    let waiters = Arc::clone(&r.waiters);
    *hooked.hook.lock().unwrap() = Some(Box::new(move |mem: &Memory| {
        waiters.on_write(SPACE, SEM, 5);
        mem.write(SEM, &6u32.to_le_bytes()).unwrap();
    }));
    let a = r.channel(5);
    r.submit(
        a,
        vec![r.entry(CMD_A, &acquire_then_write(AcquireOp::Equal, 5, 0, 0xa1))],
    );
    r.turn();
    assert_eq!(hooked.reads.load(Ordering::SeqCst), 2);
    assert_eq!(r.front.counters().parks, 0);
    r.settle();
    assert!(has(&r.take(), &scratch_text(a, 0, 0xa1)));
    assert!(r.waiters.is_empty());
}

#[test]
fn an_acquire_met_at_the_second_read_does_not_park() {
    // The word changes between the pusher's read and the front's second read: met then.
    let slot: Arc<Mutex<Option<Arc<Hooked>>>> = Arc::default();
    let seen = Arc::clone(&slot);
    let mut r = rig_with(RING_WORDS_FOR_TESTS, move |mem| {
        let hooked = Arc::new(Hooked {
            inner: mem,
            reads: AtomicU32::new(0),
            hook: Mutex::new(Some(Box::new(|mem: &Memory| {
                mem.write(SEM, &3u32.to_le_bytes()).unwrap();
            }))),
        });
        *seen.lock().unwrap() = Some(Arc::clone(&hooked));
        hooked
    });
    let a = r.channel(5);
    r.submit(
        a,
        vec![r.entry(
            CMD_A,
            &acquire_then_write(AcquireOp::GreaterOrEqual, 3, 0, 0xa1),
        )],
    );
    r.settle();
    assert_eq!(r.front.counters().parks, 0);
    assert!(has(&r.take(), &scratch_text(a, 0, 0xa1)));
    assert!(r.waiters.is_empty());
}

#[test]
fn the_poll_steps_through_its_schedule_on_the_clock() {
    let mut r = rig();
    let a = r.channel(5);
    r.submit(
        a,
        vec![r.entry(CMD_A, &acquire_then_write(AcquireOp::Equal, 1, 0, 0xa1))],
    );
    turn_until_parked(&mut r);
    // The spins: every turn polls and may sleep for nothing.
    for spin in 0..POLL_SPINS {
        let polls = r.front.counters().polls;
        let turn = r.turn();
        assert_eq!(r.front.counters().polls, polls + 1, "spin {spin}");
        let expected = if spin + 1 == POLL_SPINS {
            POLL_FIRST_SLEEP
        } else {
            Duration::ZERO
        };
        assert_eq!(turn, Turn::Idle { park: expected }, "spin {spin}");
    }
    // The first sleep: one nanosecond short of it, no poll; at it, a poll, and the next step
    // doubles.
    let polls = r.front.counters().polls;
    r.clock.advance(
        POLL_FIRST_SLEEP
            .checked_sub(Duration::from_nanos(1))
            .unwrap(),
    );
    assert_eq!(
        r.turn(),
        Turn::Idle {
            park: Duration::from_nanos(1)
        }
    );
    assert_eq!(r.front.counters().polls, polls);
    r.clock.advance(Duration::from_nanos(1));
    assert_eq!(
        r.turn(),
        Turn::Idle {
            park: POLL_FIRST_SLEEP * 2
        }
    );
    assert_eq!(r.front.counters().polls, polls + 1);
    // A CPU write is seen at the next poll, not before it.
    r.write(SEM, &[1]);
    r.clock.advance(POLL_FIRST_SLEEP);
    r.turn();
    assert!(!has(&r.take(), &scratch_text(a, 0, 0xa1)));
    r.clock.advance(POLL_FIRST_SLEEP);
    r.settle();
    assert!(has(&r.take(), &scratch_text(a, 0, 0xa1)));
    // Nothing waits any more: the front parks for the safety net.
    assert_eq!(r.front.turn(), Ok(Turn::Idle { park: IDLE_PARK }));
}

#[test]
fn the_poll_never_sleeps_longer_than_its_ceiling() {
    let mut r = rig();
    let a = r.channel(5);
    r.submit(
        a,
        vec![r.entry(CMD_A, &acquire_then_write(AcquireOp::Equal, 1, 0, 0xa1))],
    );
    turn_until_parked(&mut r);
    let mut longest = Duration::ZERO;
    for _ in 0..POLL_SPINS + 20 {
        if let Turn::Idle { park } = r.turn() {
            longest = longest.max(park);
            r.clock.advance(park);
        }
    }
    assert_eq!(longest, POLL_LONGEST_SLEEP);
}

#[test]
fn the_poll_fires_while_other_channels_keep_the_front_busy() {
    // A waits on the CPU, past the poll's spins, so a poll needs the clock. B then has words
    // for four quanta, so every turn works. Once the clock passes the poll's time, the next
    // turn polls and A resumes while B is still busy.
    let mut r = rig();
    let a = r.channel(5);
    let b = r.channel(6);
    r.submit(
        a,
        vec![r.entry(CMD_A, &acquire_then_write(AcquireOp::Equal, 1, 0, 0xa1))],
    );
    turn_until_parked(&mut r);
    for _ in 0..POLL_SPINS {
        r.turn();
    }
    let nops = vec![0u32; usize::try_from(QUANTUM).unwrap() * 4];
    // B's submission rechecks A (still unmet) and B starts.
    r.submit(b, vec![r.entry(CMD_B, &nops)]);
    assert_eq!(r.turn(), Turn::Worked);
    // The CPU releases A's semaphore. Busy, but the poll is not due: A still waits.
    r.write(SEM, &[1]);
    assert_eq!(r.turn(), Turn::Worked);
    assert!(r.front.lanes[&a].wait.is_some());
    let polls = r.front.counters().polls;
    r.clock.advance(POLL_FIRST_SLEEP);
    assert_eq!(r.turn(), Turn::Worked);
    assert_eq!(r.front.counters().polls, polls + 1);
    assert!(r.front.lanes[&a].wait.is_none());
    assert!(r.front.lanes[&b].is_runnable(), "B is still busy");
    r.settle();
    assert!(has(&r.take(), &scratch_text(a, 0, 0xa1)));
}

#[test]
fn a_submission_on_any_channel_rechecks_a_parked_one() {
    // The CPU writes the word with no poll due (past the spins, the clock stopped): a
    // submission on another channel makes the front look again.
    let mut r = rig();
    let a = r.channel(5);
    let b = r.channel(6);
    r.submit(
        a,
        vec![r.entry(CMD_A, &acquire_then_write(AcquireOp::Equal, 1, 0, 0xa1))],
    );
    turn_until_parked(&mut r);
    for _ in 0..POLL_SPINS {
        r.turn();
    }
    r.write(SEM, &[1]);
    assert!(matches!(r.turn(), Turn::Idle { park } if !park.is_zero()));
    assert!(!has(&r.take(), &scratch_text(a, 0, 0xa1)));
    r.submit(b, vec![r.entry(CMD_B, &write_scratch(1, 0xb1))]);
    r.turn();
    r.settle();
    assert!(has(&r.take(), &scratch_text(a, 0, 0xa1)));
}

// ---------------------------------------------------------------------------------------------
// Submission fences and increments
// ---------------------------------------------------------------------------------------------

#[test]
fn a_fence_wait_holds_the_submission_until_an_increment_reaches_it() {
    let mut r = rig();
    let a = r.channel(5);
    let b = r.channel(9);
    let waits = Submission {
        wait: Some(Fence {
            id: syncpoint(9),
            value: 2,
        }),
        ..plain(vec![r.entry(CMD_A, &write_scratch(0, 0xa1))])
    };
    r.shared.submit(a, waits);
    r.settle();
    assert_eq!(r.front.counters().fence_waits, 1);
    assert_eq!(r.front.counters().words, 0);
    // One increment is not enough.
    let one = Submission {
        increments: 1,
        ..plain(vec![])
    };
    r.shared.submit(b, one.clone());
    r.settle();
    assert_eq!(r.syncpoints.read(syncpoint(9)), 1);
    assert!(!has(&r.take(), &scratch_text(a, 0, 0xa1)));
    // The second reaches it. Past the poll's spins with the clock stopped, and with B's
    // submission looked at before its increment is performed, only the back's report of the
    // increment can make the front look again.
    for _ in 0..POLL_SPINS {
        r.turn();
    }
    let polls = r.front.counters().polls;
    r.shared.submit(b, one);
    r.settle();
    assert_eq!(r.front.counters().polls, polls);
    assert_eq!(r.syncpoints.read(syncpoint(9)), 2);
    assert!(has(&r.take(), &scratch_text(a, 0, 0xa1)));
}

#[test]
fn a_fence_already_reached_does_not_wait() {
    let mut r = rig();
    let a = r.channel(5);
    r.syncpoints.increment(syncpoint(9));
    let reached = Submission {
        wait: Some(Fence {
            id: syncpoint(9),
            value: 1,
        }),
        ..plain(vec![r.entry(CMD_A, &write_scratch(0, 0xa1))])
    };
    r.shared.submit(a, reached);
    r.settle();
    assert_eq!(r.front.counters().fence_waits, 0);
    assert!(has(&r.take(), &scratch_text(a, 0, 0xa1)));
}

#[test]
fn one_counted_entry_meets_a_fence_it_passes_up_to_half_the_range() {
    // A fence one past the counter, and another channel's run of increments: the front looks
    // at the final value only (`back::fence`, "Waiters and a run of increments"). A run of 2^31
    // meets the fence; a run one longer leaves it 2^31 behind, and the channel stays parked.
    for (count, met) in [(SIGN_BIT, true), (SIGN_BIT + 1, false)] {
        let mut r = rig();
        let a = r.channel(5);
        let b = r.channel(9);
        let waits = Submission {
            wait: Some(Fence {
                id: syncpoint(9),
                value: 1,
            }),
            ..plain(vec![r.entry(CMD_A, &write_scratch(0, 0xa1))])
        };
        r.shared.submit(a, waits);
        r.settle();
        assert_eq!(r.front.counters().fence_waits, 1, "{count:#x}");
        r.shared.submit(
            b,
            Submission {
                increments: count,
                ..plain(vec![])
            },
        );
        r.settle();
        assert_eq!(r.syncpoints.read(syncpoint(9)), count, "{count:#x}");
        let texts = r.take();
        assert!(
            has(&texts, &format!("syncpoint id=9 count={count}")),
            "{count:#x}"
        );
        assert_eq!(
            has(&texts, &scratch_text(a, 0, 0xa1)),
            met,
            "{count:#x}: {texts:?}"
        );
    }
}

#[test]
fn a_fence_another_thread_increments_is_seen_by_the_poll() {
    // gsv's video worker increments syncpoints without the back: the poll sees it.
    let mut r = rig();
    let a = r.channel(5);
    let waits = Submission {
        wait: Some(Fence {
            id: syncpoint(9),
            value: 1,
        }),
        ..plain(vec![r.entry(CMD_A, &write_scratch(0, 0xa1))])
    };
    r.shared.submit(a, waits);
    r.settle();
    assert_eq!(r.front.counters().fence_waits, 1);
    r.syncpoints.increment(syncpoint(9));
    let polls = r.front.counters().polls;
    r.settle();
    assert!(r.front.counters().polls > polls);
    assert!(has(&r.take(), &scratch_text(a, 0, 0xa1)));
}

#[test]
fn wait_for_idle_comes_before_the_increments_and_only_with_them() {
    let mut r = rig();
    let a = r.channel(5);
    let entry = r.entry(CMD_A, &write_scratch(0, 0xa1));
    r.shared.submit(
        a,
        Submission {
            increments: 3,
            wfi: true,
            ..plain(vec![entry])
        },
    );
    r.settle();
    let texts = r.take();
    let write = position(&texts, &scratch_text(a, 0, 0xa1));
    let tail: Vec<&str> = texts[write + 1..].iter().map(String::as_str).collect();
    let wfi = format!("wait-for-idle channel={}", a.0);
    // One entry carries the three increments.
    assert_eq!(tail, [wfi.as_str(), "syncpoint id=5 count=3"]);
    assert_eq!(r.syncpoints.read(syncpoint(5)), 3);
    assert_eq!(r.front.counters().increments, 3);
    assert_eq!(r.front.counters().submissions, 1);
    // A wait for idle with no increments sends nothing; increments without one send no wait.
    r.shared.submit(
        a,
        Submission {
            wfi: true,
            ..plain(vec![])
        },
    );
    r.shared.submit(
        a,
        Submission {
            increments: 1,
            ..plain(vec![])
        },
    );
    r.settle();
    assert_eq!(r.take(), ["syncpoint id=5 count=1"]);
    assert_eq!(r.front.counters().submissions, 3);
    assert_eq!(r.front.counters().increments, 4);
    assert_eq!(r.syncpoints.read(syncpoint(5)), 4);
}

#[test]
fn u32_max_increments_go_out_as_one_entry_in_one_turn() {
    // One entry with the count, so the front's cost does not grow with it: the submission and
    // another channel's both finish in one turn, and the counter wraps to `u32::MAX` from 0.
    let mut r = rig();
    let a = r.channel(5);
    let b = r.channel(6);
    r.shared.submit(
        a,
        Submission {
            increments: u32::MAX,
            ..plain(vec![])
        },
    );
    r.submit(b, vec![r.entry(CMD_B, &write_scratch(1, 0xb1))]);
    r.front.turn().unwrap();
    assert_eq!(r.front.counters().submissions, 2);
    assert_eq!(r.front.counters().increments, u64::from(u32::MAX));
    // A turn that worked publishes nothing yet; the next, idle one does.
    r.settle();
    let texts = r.take();
    let syncpoints: Vec<&String> = texts
        .iter()
        .filter(|t| t.starts_with("syncpoint"))
        .collect();
    assert_eq!(syncpoints, ["syncpoint id=5 count=4294967295"]);
    assert!(has(&texts, &scratch_text(b, 1, 0xb1)));
    assert_eq!(r.syncpoints.read(syncpoint(5)), u32::MAX);
}

#[test]
fn a_submission_that_uses_its_quantum_sends_its_increments_next_turn() {
    // Exactly a quantum of words: nothing is left for the increments entry this turn.
    let quantum = usize::try_from(QUANTUM).unwrap();
    let mut r = rig();
    let a = r.channel(5);
    let nops = vec![0u32; quantum];
    r.shared.submit(
        a,
        Submission {
            increments: 7,
            ..plain(vec![r.entry(CMD_A, &nops)])
        },
    );
    r.front.turn().unwrap();
    assert_eq!(r.front.counters().words, u64::from(QUANTUM));
    assert_eq!(r.front.counters().increments, 0);
    assert_eq!(r.front.counters().submissions, 0);
    r.front.turn().unwrap();
    assert_eq!(r.front.counters().increments, 7);
    assert_eq!(r.front.counters().submissions, 1);
    r.settle();
    assert_eq!(r.take(), ["syncpoint id=5 count=7"]);
    assert_eq!(r.syncpoints.read(syncpoint(5)), 7);
}

#[test]
fn the_increments_entry_costs_one_unit_of_the_quantum() {
    // A quantum less one of words, then the entry: it takes the last unit, so the channel's
    // next submission, one word long, waits for the next turn.
    let quantum = usize::try_from(QUANTUM).unwrap();
    let mut r = rig();
    let a = r.channel(5);
    let nops = vec![0u32; quantum - 1];
    r.shared.submit(
        a,
        Submission {
            increments: 7,
            ..plain(vec![r.entry(CMD_A, &nops)])
        },
    );
    r.submit(a, vec![r.entry(CMD_FAR, &[0])]);
    r.front.turn().unwrap();
    assert_eq!(r.front.counters().words, u64::from(QUANTUM) - 1);
    assert_eq!(r.front.counters().increments, 7);
    assert_eq!(r.front.counters().submissions, 1);
    r.front.turn().unwrap();
    assert_eq!(r.front.counters().words, u64::from(QUANTUM));
    assert_eq!(r.front.counters().submissions, 2);
    r.settle();
    assert_eq!(r.take(), ["syncpoint id=5 count=7"]);
}

// ---------------------------------------------------------------------------------------------
// SYNC_WAIT
// ---------------------------------------------------------------------------------------------

#[test]
fn sync_wait_holds_the_entry_until_the_back_has_performed_earlier_work() {
    let mut r = rig();
    let a = r.channel(5);
    r.submit(a, vec![r.entry(CMD_A, &write_scratch(0, 0xa1))]);
    let mut pb = PushBuffer::new();
    pb.methods(MethodMode::Incrementing, 0, scratch(1), &[0xa2])
        .unwrap();
    r.write(CMD_B, pb.words());
    r.submit(a, vec![raw_entry(CMD_B, pb.words().len(), Sync::Wait)]);
    // The back has not drained: the entry waits, unread.
    r.front.turn().unwrap();
    assert_eq!(r.front.counters().sync_waits, 1);
    let words = r.front.counters().words;
    r.front.turn().unwrap();
    assert_eq!(r.front.counters().words, words);
    // The back performs the work and reports the marker: the entry runs.
    r.settle();
    let texts = r.take();
    assert!(
        position(&texts, &scratch_text(a, 0, 0xa1)) < position(&texts, &scratch_text(a, 1, 0xa2))
    );
    assert_eq!(r.front.counters().sync_waits, 1);
}

#[test]
fn sync_wait_with_nothing_unperformed_starts_at_once() {
    let mut r = rig();
    let a = r.channel(5);
    let words = write_scratch(0, 0xa1);
    r.write(CMD_A, &words);
    r.submit(a, vec![raw_entry(CMD_A, words.len(), Sync::Wait)]);
    r.settle();
    assert_eq!(r.front.counters().sync_waits, 0);
    assert!(has(&r.take(), &scratch_text(a, 0, 0xa1)));
}

// ---------------------------------------------------------------------------------------------
// Macro catch-ups
// ---------------------------------------------------------------------------------------------

/// `SET_MME_MEM_ADDRESS_A`, `_B` and `MME_DMA_READ_FIFOED`.
fn dma_methods() -> [u16; 3] {
    [
        m(clc797::SET_MME_MEM_ADDRESS_A.method()),
        m(clc797::SET_MME_MEM_ADDRESS_B.method()),
        m(clc797::MME_DMA_READ_FIFOED.method()),
    ]
}

/// Emits the next parameter to `method`.
fn emit_param(method: u16) -> Instruction {
    with_out(
        with_imm(Instruction::default(), 0, method),
        0,
        OutOp::Imm1,
        OutOp::Load0,
    )
}

/// A macro that reads one word by MME DMA (parameters: upper address, lower address, length)
/// and writes it to scratch 0.
fn dma_macro() -> Vec<Instruction> {
    let [upper, lower, read] = dma_methods();
    finish(vec![
        emit_param(upper),
        emit_param(lower),
        emit_param(read),
        with_imm(
            inst0(AluOp::Extended, Reg::Zero, Reg::Imm, Reg::Zero),
            u16::try_from(LOAD_BARRIER).unwrap(),
            0,
        ),
        mov_reg(0, Reg::Load0),
        with_out(
            with_imm(
                inst0(AluOp::Add, Reg::Zero, Reg::Gpr(0), Reg::Zero),
                0,
                scratch(0),
            ),
            0,
            OutOp::Imm1,
            OutOp::Alu0,
        ),
    ])
}

#[test]
fn a_macro_dma_read_with_unperformed_work_waits_for_the_back_and_runs_again() {
    let source = GPU_BASE + 50 * PAGE_SIZE + 0x20;
    let mut r = rig();
    let a = r.channel(5);
    // Upload the macro at instruction 0, bind it to entry 0, and call it with the source.
    let mut pb = PushBuffer::new();
    bind_3d(&mut pb);
    let pointer = m(clc797::LOAD_MME_INSTRUCTION_RAM_POINTER.method());
    let code = m(clc797::LOAD_MME_INSTRUCTION_RAM.method());
    let start_pointer = m(clc797::LOAD_MME_START_ADDRESS_RAM_POINTER.method());
    let start = m(clc797::LOAD_MME_START_ADDRESS_RAM.method());
    pb.methods(MethodMode::Incrementing, 0, pointer, &[0])
        .unwrap();
    pb.methods(
        MethodMode::NonIncrementing,
        0,
        code,
        &encode_program(&dma_macro()).unwrap(),
    )
    .unwrap();
    pb.methods(MethodMode::Incrementing, 0, start_pointer, &[0])
        .unwrap();
    pb.methods(MethodMode::Incrementing, 0, start, &[0])
        .unwrap();
    let call = m(clc797::CALL_MME_MACRO.method());
    let upper = u32::try_from(source >> 32).unwrap();
    let lower = u32::try_from(source & 0xffff_ffff).unwrap();
    pb.methods(MethodMode::IncrementOnce, 0, call, &[upper, lower, 1])
        .unwrap();
    pb.methods(MethodMode::Incrementing, 0, scratch(1), &[0xa2])
        .unwrap();
    r.write(source, &[0x0bad]);
    r.submit(a, vec![r.entry(CMD_A, pb.words())]);
    // The upload's words are pushed and unperformed: the call waits, and the words after it
    // are not run.
    r.front.turn().unwrap();
    assert_eq!(r.front.counters().catch_ups, 1);
    let words = r.front.counters().words;
    r.front.turn().unwrap();
    assert_eq!(r.front.counters().words, words);
    // The back's work writes the word the macro reads; then it reports the marker.
    r.write(source, &[0x1234]);
    r.settle();
    let texts = r.take();
    let result = position(&texts, &scratch_text(a, 0, 0x1234));
    assert!(result < position(&texts, &scratch_text(a, 1, 0xa2)));
    assert!(!has(&texts, &scratch_text(a, 0, 0x0bad)));
    assert_eq!(r.front.counters().catch_ups, 1);
    assert_eq!(r.front.counters().macro_failures, 0);
}

#[test]
fn a_failing_macro_call_is_counted_and_the_stream_goes_on() {
    // A macro that loads two parameters, called with one: the run fails on the missing
    // parameter, commits nothing, and the words after the call still run.
    let mut r = rig();
    let a = r.channel(5);
    let mut pb = PushBuffer::new();
    bind_3d(&mut pb);
    let call = m(clc797::CALL_MME_MACRO.method());
    let [upper, ..] = dma_methods();
    let program = finish(vec![emit_param(upper), emit_param(upper)]);
    pb.methods(
        MethodMode::Incrementing,
        0,
        m(clc797::LOAD_MME_INSTRUCTION_RAM_POINTER.method()),
        &[0],
    )
    .unwrap();
    pb.methods(
        MethodMode::NonIncrementing,
        0,
        m(clc797::LOAD_MME_INSTRUCTION_RAM.method()),
        &encode_program(&program).unwrap(),
    )
    .unwrap();
    // Two parameters wanted, one given.
    pb.methods(MethodMode::IncrementOnce, 0, call, &[7])
        .unwrap();
    pb.methods(MethodMode::Incrementing, 0, scratch(1), &[0xa2])
        .unwrap();
    r.submit(a, vec![r.entry(CMD_A, pb.words())]);
    r.settle();
    assert_eq!(r.front.counters().macro_failures, 1);
    assert!(has(&r.take(), &scratch_text(a, 1, 0xa2)));
}

// ---------------------------------------------------------------------------------------------
// What restarts a recheck and the poll's schedule
// ---------------------------------------------------------------------------------------------

/// A stream that acquires `address` equal to 1.
fn acquire_only(address: u64) -> Vec<u32> {
    let mut pb = PushBuffer::new();
    acquire(&mut pb, address, AcquireOp::Equal, 1);
    pb.into_words()
}

/// Turns the front until `parks` channels have parked (at most a few turns).
fn turn_until_parks(r: &mut Rig, parks: u64) {
    for _ in 0..8 {
        if r.front.counters().parks >= parks {
            return;
        }
        r.turn();
    }
    assert!(
        r.front.counters().parks >= parks,
        "fewer than {parks} parks"
    );
}

/// Turns the front until a turn is idle (at most a few turns); how long it may park.
fn next_idle_park(r: &mut Rig) -> Duration {
    let mut idle = None;
    for _ in 0..8 {
        if let Turn::Idle { park } = r.turn() {
            idle = Some(park);
            break;
        }
    }
    assert!(idle.is_some(), "no idle turn");
    idle.unwrap()
}

#[test]
fn a_release_the_back_writes_rechecks_a_parked_channel_at_once() {
    // A waits past the poll's spins with the clock stopped, so no poll comes. B's release,
    // written by the back, is reported, and that report alone makes the front look again.
    let mut r = rig();
    let a = r.channel(5);
    let b = r.channel(6);
    r.submit(
        a,
        vec![r.entry(CMD_A, &acquire_then_write(AcquireOp::Equal, 1, 0, 0xa1))],
    );
    turn_until_parked(&mut r);
    for _ in 0..POLL_SPINS + 4 {
        r.turn();
    }
    let polls = r.front.counters().polls;
    let mut pb = PushBuffer::new();
    release(&mut pb, SEM, 1);
    r.submit(b, vec![r.entry(CMD_B, pb.words())]);
    for _ in 0..6 {
        r.turn();
    }
    assert!(has(&r.take(), &scratch_text(a, 0, 0xa1)));
    assert_eq!(r.front.counters().polls, polls);
}

#[test]
fn a_channel_that_newly_parks_restarts_the_polls_schedule() {
    let mut r = rig();
    let a = r.channel(5);
    let b = r.channel(6);
    r.submit(a, vec![r.entry(CMD_A, &acquire_only(SEM))]);
    turn_until_parks(&mut r, 1);
    for _ in 0..POLL_SPINS {
        r.turn();
    }
    // A's schedule is in its sleeps.
    assert!(!next_idle_park(&mut r).is_zero());
    // B newly parks: the schedule is back in its spins.
    r.submit(b, vec![r.entry(CMD_B, &acquire_only(SEM2))]);
    turn_until_parks(&mut r, 2);
    assert_eq!(next_idle_park(&mut r), Duration::ZERO);
}

#[test]
fn a_channel_that_resumes_restarts_the_polls_schedule_for_the_others() {
    let mut r = rig();
    let a = r.channel(5);
    let b = r.channel(6);
    r.submit(
        a,
        vec![r.entry(CMD_A, &acquire_then_write(AcquireOp::Equal, 1, 0, 0xa1))],
    );
    r.submit(b, vec![r.entry(CMD_B, &acquire_only(SEM2))]);
    turn_until_parks(&mut r, 2);
    for _ in 0..POLL_SPINS {
        r.turn();
    }
    let park = next_idle_park(&mut r);
    assert!(!park.is_zero());
    // The CPU releases A's semaphore; the poll that sees it resumes A, and B's schedule is
    // back in its spins.
    r.write(SEM, &[1]);
    r.clock.advance(park);
    assert_eq!(r.turn(), Turn::Worked);
    assert!(r.front.lanes[&a].wait.is_none());
    assert!(r.front.lanes[&b].wait.is_some());
    assert_eq!(next_idle_park(&mut r), Duration::ZERO);
    assert!(has(&r.take(), &scratch_text(a, 0, 0xa1)));
}
