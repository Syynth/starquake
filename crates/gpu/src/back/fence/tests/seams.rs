//! The fence path's seams for the executor (slice G19): the submitted fences and applied updates
//! it drains, recording a whole batch of commands at once, the catch-up wait, and latching with
//! the words read back after a release.

use super::*;

/// A memory whose writes land in `inner` and whose reads fault: a release to it is written, and
/// reading it back fails.
struct WriteOnly(Arc<Memory>);

impl GpuMemory for WriteOnly {
    fn read(&self, va: u64, _buf: &mut [u8]) -> Result<(), GpuFault> {
        Err(GpuFault::Unmapped { va })
    }
    fn write(&self, va: u64, data: &[u8]) -> Result<(), GpuFault> {
        self.0.write(va, data)
    }
    fn generation(&self) -> u64 {
        self.0.generation()
    }
}

#[test]
fn a_release_whose_read_back_faults_latches_with_the_bytes_written() {
    let mut r = rig();
    r.waiters.register(Waiter {
        space: SPACE,
        channel: ChannelId(6),
        address: SEM,
        op: AcquireOp::Equal,
        payload: 0x77,
    });
    let action = FencedAction::Release {
        memory: Arc::new(WriteOnly(Arc::clone(&r.mem))),
        space: SPACE,
        release: four(SEM, 0x77),
    };
    // Nothing in flight: performed at once.
    let (attached, events) = captured(|| r.fences.attach(action));
    assert!(
        matches!(&attached, Ok(Attached::Performed(report)) if report.latched && report.released == 1),
        "{attached:?}"
    );
    assert_eq!(word(&r.mem, SEM), 0x77);
    let warnings = at(&events, tracing::Level::WARN);
    assert!(
        warnings
            .iter()
            .any(|e| e.contains("could not be read back")),
        "{events:?}"
    );
}

#[test]
fn newest_submitted_follows_each_submission() {
    let mut r = rig();
    assert_eq!(r.fences.newest_submitted(), FenceValue::default());
    r.fences.record(command(1)).unwrap();
    r.fences.close(CloseReason::Present).unwrap();
    assert_eq!(r.fences.newest_submitted(), FenceValue::new(1));
    r.fences.record(command(2)).unwrap();
    r.fences.close(CloseReason::Present).unwrap();
    assert_eq!(r.fences.newest_submitted(), FenceValue::new(2));
    // A list with no commands submits nothing.
    r.fences.close(CloseReason::Present).unwrap();
    assert_eq!(r.fences.newest_submitted(), FenceValue::new(2));
}

#[test]
fn newest_submitted_keeps_the_highest_when_fences_come_out_of_order() {
    let mut r = rig();
    r.backend.script(&[5, 3]);
    for submit in [1, 2] {
        r.fences.record(command(submit)).unwrap();
        r.fences.close(CloseReason::Present).unwrap();
    }
    assert_eq!(r.fences.newest_submitted(), FenceValue::new(5));
}

#[test]
fn a_refused_list_is_not_reported_submitted() {
    let mut r = rig();
    r.backend.refuse_next();
    r.fences.record(command(1)).unwrap();
    assert_eq!(
        r.fences.close(CloseReason::Present),
        Err(FenceError::Submit(REFUSED))
    );
    assert_eq!(r.fences.newest_submitted(), FenceValue::default());
}

#[test]
fn applied_updates_are_handed_over_in_order_with_their_space() {
    let mut r = rig();
    let vm = Arc::new(GpuVm::new());
    let first = VmUpdate::Unmap {
        range: PageRange::new(GPU_BASE, PAGE_SIZE).unwrap(),
    };
    let second = VmUpdate::DestroyMemory {
        range: PageRange::new(GPU_BASE + PAGE_SIZE, PAGE_SIZE).unwrap(),
    };
    let other = AsId(SPACE.0 + 1);
    for (token, space, update) in [(1, SPACE, first), (2, other, second)] {
        let marker = FencedAction::Marker {
            token: MarkerToken(token),
            ordered: Some(OrderedUpdate {
                change: Some(VmChange {
                    space,
                    vm: Arc::clone(&vm),
                    update,
                }),
                ticket: None,
            }),
        };
        r.fences.attach(marker).unwrap();
    }
    assert_eq!(
        r.fences.take_applied(),
        vec![(SPACE, first), (other, second)]
    );
    assert_eq!(r.fences.take_applied(), vec![]);
}

#[test]
fn an_update_is_not_handed_over_before_its_marker_is_performed() {
    let mut r = rig();
    let vm = Arc::new(GpuVm::new());
    let update = VmUpdate::Unmap {
        range: PageRange::new(GPU_BASE, PAGE_SIZE).unwrap(),
    };
    r.fences.record(command(1)).unwrap();
    let marker = FencedAction::Marker {
        token: MarkerToken(1),
        ordered: Some(OrderedUpdate {
            change: Some(VmChange {
                space: SPACE,
                vm,
                update,
            }),
            ticket: None,
        }),
    };
    assert_eq!(r.fences.attach(marker), Ok(Attached::Open));
    let fence = r.fences.close(CloseReason::Present).unwrap().unwrap();
    assert_eq!(r.fences.take_applied(), vec![]);
    r.backend.finish(fence);
    r.fences.complete();
    assert_eq!(r.fences.take_applied(), vec![(SPACE, update)]);
}

#[test]
fn a_recorded_list_stays_in_one_list_and_closes_once_after_it() {
    let mut r = rig();
    r.fences.record(command(1)).unwrap();
    // Past the close interval: a command-by-command record would close after the first of
    // the batch.
    r.clock.set(CLOSE_AFTER);
    let mut batch = CommandList::new();
    batch.push(command(2));
    batch.push(command(3));
    r.fences.record_list(&mut batch).unwrap();
    assert_eq!(batch.commands, vec![]);
    assert_eq!(r.fences.open_commands(), 0);
    assert_eq!(r.backend.submitted(), vec![FenceValue::new(1)]);
    assert_eq!(r.fences.counters().closed.elapsed, 1);
    let lists = r.backend.inner.lists();
    assert_eq!(
        lists.last().map(|(_, list)| list.commands.clone()),
        Some(vec![command(1), command(2), command(3)])
    );
}

#[test]
fn a_recorded_list_below_the_interval_stays_open() {
    let mut r = rig();
    r.fences.record(command(1)).unwrap();
    r.clock
        .set(CLOSE_AFTER.checked_sub(Duration::from_nanos(1)).unwrap());
    let mut batch = CommandList::new();
    batch.push(command(2));
    r.fences.record_list(&mut batch).unwrap();
    assert_eq!(r.fences.open_commands(), 2);
    assert_eq!(r.backend.submitted(), vec![]);
}

#[test]
fn an_empty_recorded_list_starts_no_interval() {
    let mut r = rig();
    r.clock.set(Duration::from_secs(1));
    r.fences.record_list(&mut CommandList::new()).unwrap();
    // Had the empty batch started the interval at 1 s, this command at 1 s + CLOSE_AFTER would
    // close the list.
    r.clock.set(Duration::from_secs(1) + CLOSE_AFTER);
    r.fences.record(command(1)).unwrap();
    assert_eq!(r.fences.open_commands(), 1);
}

#[test]
fn settle_with_nothing_pending_does_nothing() {
    let mut r = rig();
    r.fences.record(command(1)).unwrap();
    // A release to the word just past the range, and one in another space.
    let after = r.action(four(SEM + 4, 1));
    r.fences.attach(after).unwrap();
    let elsewhere = FencedAction::Release {
        memory: Arc::<Memory>::clone(&r.mem),
        space: AsId(SPACE.0 + 1),
        release: four(SEM, 1),
    };
    r.fences.attach(elsewhere).unwrap();
    assert_eq!(
        r.fences.settle(SPACE, SEM..SEM + 4, PATIENCE),
        Ok(Settled::Nothing)
    );
    assert_eq!(r.fences.open_commands(), 1);
    assert_eq!(r.backend.submitted(), vec![]);
}

#[test]
fn settle_closes_the_open_list_and_waits_for_it() {
    let backend = Arc::new(LateBackend::new(CompletionWake::none()));
    let mut r = rig_on(Arc::clone(&backend), Mode::Fenced);
    r.fences.record(command(1)).unwrap();
    // A 16-byte release whose last word is the range's first.
    let release = Release {
        address: SEM - 12,
        payload: 9,
        form: ReleaseForm {
            size: PayloadSize::Eight,
            timestamp: true,
        },
        op: ReleaseOp::Write,
    };
    let action = r.action(release);
    r.fences.attach(action).unwrap();
    let finisher = Arc::clone(&backend);
    let done = thread::spawn(move || {
        // Wait until the list is submitted, then complete it.
        for _ in 0..1000 {
            if let Some(&fence) = finisher.submitted().first() {
                finisher.finish(fence);
                return;
            }
            thread::sleep(Duration::from_millis(1));
        }
    });
    let settled = within(move || {
        let outcome = r.fences.settle(SPACE, SEM..SEM + 4, PATIENCE);
        (outcome, r.fences.counters().closed.catch_up, r)
    });
    done.join().unwrap();
    let (outcome, catch_ups, r) = settled;
    assert_eq!(outcome, Ok(Settled::Performed(released(1))));
    assert_eq!(catch_ups, 1);
    assert_eq!(read(&r.mem, SEM - 12, 1), vec![9]);
}

#[test]
fn settle_waits_for_the_newest_list_holding_a_release() {
    let mut r = rig();
    let first = r.list_with_release(SEM, 1);
    let second = r.list_with_release(SEM, 2);
    // A third list with a release elsewhere: not waited for.
    let third = r.list_with_release(SEM2, 3);
    assert!(first < second && second < third);
    r.backend.finish(second);
    assert_eq!(
        r.fences.settle(SPACE, SEM..SEM + 1, PATIENCE),
        Ok(Settled::Performed(released(2)))
    );
    assert_eq!(word(&r.mem, SEM), 2);
    assert_eq!(r.fences.lists_in_flight(), 1);
}

#[test]
fn settle_waits_for_the_newest_list_holding_a_release_not_the_oldest() {
    // Two lists hold a release of the range; only the older completes. Waiting for the older
    // would report the catch-up done while the newer release is still pending.
    let mut r = rig();
    let first = r.list_with_release(SEM, 1);
    let second = r.list_with_release(SEM, 2);
    assert!(first < second);
    r.backend.finish(first);
    assert_eq!(
        r.fences
            .settle(SPACE, SEM..SEM + 1, Duration::from_millis(1)),
        Err(FenceError::CatchUpTimedOut {
            fence: second.get()
        })
    );
    // The older list's release was performed on the way; the newer one waits.
    assert_eq!(word(&r.mem, SEM), 1);
    assert_eq!(r.fences.lists_in_flight(), 1);
}

#[test]
fn a_settle_that_times_out_is_the_exact_error_and_performs_what_completed() {
    let mut r = rig();
    let first = r.list_with_release(SEM2, 1);
    let second = r.list_with_release(SEM, 2);
    r.backend.finish(first);
    let outcome = r
        .fences
        .settle(SPACE, SEM..SEM + 4, Duration::from_millis(1));
    assert_eq!(
        outcome,
        Err(FenceError::CatchUpTimedOut {
            fence: second.get()
        })
    );
    assert_eq!(
        outcome.unwrap_err().to_string(),
        format!(
            "the host GPU did not finish command list {} in time for a catch-up",
            second.get()
        )
    );
    // The first list's release was performed on the way.
    assert_eq!(word(&r.mem, SEM2), 1);
    assert_eq!(r.fences.lists_in_flight(), 1);
}

#[test]
fn settle_sees_a_range_at_both_edges_of_a_release() {
    // A 4-byte release at SEM: ranges ending at SEM or starting at SEM + 4 miss it; ranges one
    // byte further in hit it.
    for (range, hit) in [
        (SEM - 4..SEM, false),
        (SEM - 4..SEM + 1, true),
        (SEM + 3..SEM + 8, true),
        (SEM + 4..SEM + 8, false),
    ] {
        let mut r = rig();
        let fence = r.list_with_release(SEM, 1);
        r.backend.finish(fence);
        let expected = if hit {
            Settled::Performed(released(1))
        } else {
            Settled::Nothing
        };
        assert_eq!(
            r.fences.settle(SPACE, range.clone(), PATIENCE),
            Ok(expected),
            "{range:x?}"
        );
    }
}

#[test]
fn catch_up_closes_the_open_list_and_waits_for_the_newest_submitted() {
    let mut r = rig();
    let before = word(&r.mem, SEM);
    r.fences.record(command(1)).unwrap();
    let action = r.action(four(SEM, 5));
    r.fences.attach(action).unwrap();
    // The host never finishes: the exact error, with the list closed and still in flight.
    assert_eq!(
        r.fences.catch_up(Duration::from_millis(1)),
        Err(FenceError::CatchUpTimedOut { fence: 1 })
    );
    assert_eq!(r.fences.counters().closed.catch_up, 1);
    assert_eq!(r.fences.lists_in_flight(), 1);
    assert_eq!(word(&r.mem, SEM), before);
    // Once it finishes, a catch-up with nothing open waits for the newest list submitted and
    // performs its release.
    r.backend.finish(FenceValue::new(1));
    assert_eq!(r.fences.catch_up(PATIENCE), Ok(released(1)));
    assert_eq!(r.fences.counters().closed.catch_up, 1);
    assert_eq!(word(&r.mem, SEM), 5);
    assert_eq!(r.fences.lists_in_flight(), 0);
}
