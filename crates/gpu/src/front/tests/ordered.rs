//! Ordered address-table updates and barriers: recorded positions, the marker, and the report.

use super::*;
use crate::interface::VmUpdate;

fn unmap_all() -> VmUpdate {
    VmUpdate::Unmap {
        range: PageRange::new(GPU_BASE, PAGES * PAGE_SIZE).unwrap(),
    }
}

fn ordered(ticket: u64) -> GpuCompletion {
    GpuCompletion::GpuOrdered {
        ticket: Ticket(ticket),
    }
}

/// A stream that acquires `SEM == 1` and then writes `value` to scratch 0.
fn waits_then_writes(value: u32) -> Vec<u32> {
    let mut pb = PushBuffer::new();
    bind_3d(&mut pb);
    acquire(&mut pb, SEM, AcquireOp::Equal, 1);
    pb.methods(MethodMode::Incrementing, 0, scratch(0), &[value])
        .unwrap();
    pb.into_words()
}

#[test]
fn an_unmap_is_not_reported_while_an_earlier_submission_on_another_channel_is_unread() {
    // A parks on an acquire; B's submission runs; then the unmap. B is done, but A's
    // submission (queued before the unmap) is not read: no marker, no report, the table as it
    // was.
    let mut r = rig();
    let a = r.channel(5);
    let b = r.channel(6);
    r.submit(a, vec![r.entry(CMD_A, &waits_then_writes(0xa1))]);
    r.submit(b, vec![r.entry(CMD_B, &write_scratch(1, 0xb1))]);
    r.shared.update_vm(SPACE, unmap_all(), Some(Ticket(0x31)));
    r.settle();
    assert!(has(&r.take(), &scratch_text(b, 1, 0xb1)));
    assert_eq!(r.front.counters().ordered, 0);
    assert!(r.ordered.is_empty());
    assert_eq!(r.pushed.all(), []);
    assert_eq!(r.vm.mappings().len(), 1);
    // A resumes and finishes: the marker goes out after A's write, and the back applies the
    // unmap and reports it.
    r.write(SEM, &[1]);
    r.settle();
    let texts = r.take();
    let marker = texts.iter().position(|t| t.starts_with("marker ")).unwrap();
    assert!(position(&texts, &scratch_text(a, 0, 0xa1)) < marker);
    assert_eq!(r.front.counters().ordered, 1);
    assert!(r.ordered.is_empty());
    assert_eq!(r.pushed.all(), [ordered(0x31)]);
    assert_eq!(r.vm.mappings(), []);
}

#[test]
fn a_parked_channel_holds_its_spaces_update_until_it_resumes() {
    // Spec verification item 12 (a), the spec's behaviour: the update waits for the parked
    // channel, however long.
    let mut r = rig();
    let a = r.channel(5);
    r.submit(a, vec![r.entry(CMD_A, &waits_then_writes(0xa1))]);
    r.shared.update_vm(SPACE, unmap_all(), Some(Ticket(0x32)));
    for _ in 0..3 {
        r.clock.advance(IDLE_PARK);
        r.settle();
    }
    assert_eq!(r.pushed.all(), []);
    r.write(SEM, &[1]);
    r.clock.advance(IDLE_PARK);
    r.settle();
    assert_eq!(r.pushed.all(), [ordered(0x32)]);
}

#[test]
fn work_queued_after_an_update_runs_before_it_is_applied() {
    // Spec verification item 12 (b), the spec's behaviour: B's submission after the update runs
    // while the update waits for A.
    let mut r = rig();
    let a = r.channel(5);
    let b = r.channel(6);
    r.submit(a, vec![r.entry(CMD_A, &waits_then_writes(0xa1))]);
    r.shared.update_vm(SPACE, unmap_all(), Some(Ticket(0x33)));
    r.submit(b, vec![r.entry(CMD_B, &write_scratch(1, 0xb1))]);
    r.settle();
    assert!(has(&r.take(), &scratch_text(b, 1, 0xb1)));
    assert_eq!(r.pushed.all(), []);
    assert_eq!(r.vm.mappings().len(), 1);
}

#[test]
fn a_channel_of_another_space_does_not_hold_an_update() {
    let mut r = rig();
    let other_vm = Arc::new(GpuVm::new());
    r.shared.create_address_space(
        OTHER_SPACE,
        Arc::clone(&other_vm),
        Arc::<Memory>::clone(&r.mem),
    );
    let c = r
        .shared
        .create_channel(ChannelConfig {
            space: OTHER_SPACE,
            syncpoint: syncpoint(7),
        })
        .unwrap();
    r.submit(c, vec![r.entry(CMD_C, &waits_then_writes(0xc1))]);
    r.settle();
    assert_eq!(r.front.counters().parks, 1);
    r.shared.update_vm(SPACE, unmap_all(), Some(Ticket(0x34)));
    r.settle();
    assert_eq!(r.pushed.all(), [ordered(0x34)]);
    assert_eq!(r.vm.mappings(), []);
}

#[test]
fn a_barrier_waits_for_every_channel_of_every_space() {
    let mut r = rig();
    r.shared.create_address_space(
        OTHER_SPACE,
        Arc::new(GpuVm::new()),
        Arc::<Memory>::clone(&r.mem),
    );
    let c = r
        .shared
        .create_channel(ChannelConfig {
            space: OTHER_SPACE,
            syncpoint: syncpoint(7),
        })
        .unwrap();
    r.submit(c, vec![r.entry(CMD_C, &waits_then_writes(0xc1))]);
    r.shared.barrier(Ticket(0x41));
    r.settle();
    assert_eq!(r.pushed.all(), []);
    r.write(SEM, &[1]);
    r.settle();
    assert_eq!(r.pushed.all(), [ordered(0x41)]);
    // A barrier carries no table change.
    assert_eq!(r.vm.mappings().len(), 1);
}

#[test]
fn an_update_after_a_barrier_is_not_held_by_another_spaces_channel() {
    // The barrier waits for C (another space); the update of SPACE after it covers only SPACE's
    // channels, which are done, so it is reported first.
    let mut r = rig();
    r.shared.create_address_space(
        OTHER_SPACE,
        Arc::new(GpuVm::new()),
        Arc::<Memory>::clone(&r.mem),
    );
    let a = r.channel(5);
    let c = r
        .shared
        .create_channel(ChannelConfig {
            space: OTHER_SPACE,
            syncpoint: syncpoint(7),
        })
        .unwrap();
    r.submit(a, vec![r.entry(CMD_A, &write_scratch(0, 0xa1))]);
    r.submit(c, vec![r.entry(CMD_C, &waits_then_writes(0xc1))]);
    r.shared.barrier(Ticket(0x42));
    r.shared.update_vm(SPACE, unmap_all(), Some(Ticket(0x43)));
    r.settle();
    assert_eq!(r.pushed.all(), [ordered(0x43)]);
    r.write(SEM, &[1]);
    r.settle();
    assert_eq!(r.pushed.all(), [ordered(0x43), ordered(0x42)]);
}

#[test]
fn updates_of_one_space_are_reported_in_call_order() {
    let mut r = rig();
    let a = r.channel(5);
    r.submit(a, vec![r.entry(CMD_A, &waits_then_writes(0xa1))]);
    r.shared.update_vm(SPACE, unmap_all(), Some(Ticket(0x51)));
    r.submit(a, vec![r.entry(CMD_B, &write_scratch(1, 0xa2))]);
    r.shared.barrier(Ticket(0x52));
    r.shared.update_vm(SPACE, unmap_all(), Some(Ticket(0x53)));
    r.settle();
    assert_eq!(r.pushed.all(), []);
    r.write(SEM, &[1]);
    r.settle();
    assert_eq!(
        r.pushed.all(),
        [ordered(0x51), ordered(0x52), ordered(0x53)]
    );
}

#[test]
fn an_update_of_an_unknown_space_is_reported_without_a_change() {
    let mut r = rig();
    let a = r.channel(5);
    r.submit(a, vec![r.entry(CMD_A, &waits_then_writes(0xa1))]);
    r.shared
        .update_vm(AsId(0x77), unmap_all(), Some(Ticket(0x61)));
    r.settle();
    // Not held by SPACE's parked channel, and nothing applied anywhere.
    assert_eq!(r.pushed.all(), [ordered(0x61)]);
    assert_eq!(r.vm.mappings().len(), 1);
}

#[test]
fn an_update_with_no_ticket_is_applied_and_not_reported() {
    let mut r = rig();
    r.shared.update_vm(SPACE, unmap_all(), None);
    r.settle();
    assert_eq!(r.front.counters().ordered, 1);
    assert_eq!(r.pushed.all(), []);
    assert_eq!(r.vm.mappings(), []);
}

#[test]
fn the_ordered_table_files_and_hands_out_by_token() {
    let table = OrderedTable::new();
    assert!(table.is_empty());
    table.insert(
        crate::ring::MarkerToken(3),
        crate::back::fence::OrderedUpdate {
            change: None,
            ticket: Some(Ticket(9)),
        },
    );
    assert_eq!(table.len(), 1);
    assert!(table.take(crate::ring::MarkerToken(4)).is_none());
    let taken = table.take(crate::ring::MarkerToken(3)).unwrap();
    assert_eq!(taken.ticket, Some(Ticket(9)));
    assert!(taken.change.is_none());
    assert!(table.is_empty());
    assert!(table.take(crate::ring::MarkerToken(3)).is_none());
}

/// A present of a frame at `CMD_C` (its contents are never read here).
fn present_request() -> crate::interface::PresentRequest {
    use crate::interface::{DisplayId, ImageDesc, LayerId, PresentRequest, Rect};
    PresentRequest {
        image: ImageDesc {
            address: CMD_C,
            width: 8,
            height: 8,
            pitch: 1,
            block_height_log2: 0,
            format: crate::back::exec::PRESENT_FORMAT_RGBA8,
        },
        shown: Rect {
            x: 0,
            y: 0,
            width: 8,
            height: 8,
        },
        layer: LayerId(2),
        display: DisplayId(0),
    }
}

#[test]
fn a_present_waits_for_the_submissions_queued_before_it() {
    // A parks; the present waits for it as a barrier does; B's work queued after the present
    // runs before it is pushed, and A's write comes before the present.
    let mut r = rig();
    let a = r.channel(5);
    let b = r.channel(6);
    r.submit(a, vec![r.entry(CMD_A, &waits_then_writes(0xa1))]);
    r.shared.present(present_request());
    r.submit(b, vec![r.entry(CMD_B, &write_scratch(1, 0xb1))]);
    r.settle();
    let present = RingEntry::Present(present_request()).to_string();
    let texts = r.take();
    assert!(has(&texts, &scratch_text(b, 1, 0xb1)));
    assert!(!has(&texts, &present));
    assert_eq!(r.front.counters().presents, 0);
    r.write(SEM, &[1]);
    r.settle();
    let texts = r.take();
    assert!(position(&texts, &scratch_text(a, 0, 0xa1)) < position(&texts, &present));
    assert_eq!(r.front.counters().presents, 1);
    // A present files nothing for the back to take and reports nothing.
    assert!(r.ordered.is_empty());
    assert_eq!(r.pushed.all(), []);
}

#[test]
fn a_present_with_no_channel_waiting_is_pushed_at_once() {
    let mut r = rig();
    r.shared.present(present_request());
    r.settle();
    assert_eq!(
        r.take(),
        [RingEntry::Present(present_request()).to_string()]
    );
}

/// Every channel the front reported dropped, with the entries pushed before it.
#[derive(Default)]
struct Retired(Mutex<Vec<(ChannelId, u64)>>);

impl ChannelRetired for Retired {
    fn retired(&self, channel: ChannelId, after: u64) {
        self.0.lock().unwrap().push((channel, after));
    }
}

#[test]
fn a_destroyed_channel_is_reported_with_the_entries_pushed_before_it() {
    // Three entries (the bind, the write and the barrier's marker, which counts too) before the
    // destroy; B's after it, which the count leaves out.
    let mut r = rig();
    let retired = Arc::new(Retired::default());
    let unused = Front::new(FrontDeps {
        shared: Arc::new(FrontShared::new()),
        ring: ring(1 << 4).unwrap().0,
        syncpoints: Arc::clone(&r.syncpoints),
        completions: Arc::<Pushed>::clone(&r.pushed),
        waiters: Arc::clone(&r.waiters),
        ordered: Arc::clone(&r.ordered),
        clock: Arc::<Clock>::clone(&r.clock),
    });
    let front = std::mem::replace(&mut r.front, unused);
    r.front = front.with_retired(Arc::<Retired>::clone(&retired));
    let a = r.channel(5);
    let b = r.channel(6);
    r.submit(a, vec![r.entry(CMD_A, &write_scratch(0, 0xa1))]);
    r.shared.barrier(Ticket(0x51));
    r.settle();
    let before = r.front.sent();
    assert_eq!(before, 3);
    r.shared.destroy_channel(a);
    r.submit(b, vec![r.entry(CMD_B, &write_scratch(1, 0xb1))]);
    r.settle();
    assert_eq!(retired.0.lock().unwrap().as_slice(), [(a, before)]);
    assert!(r.front.sent() > before);
}

#[test]
fn the_front_is_quiet_only_once_a_turn_ends_idle_after_it_took_its_requests() {
    let mut r = rig();
    // The rig made its address space: a request not yet taken.
    assert!(!r.shared.is_quiet());
    let a = r.channel(5);
    r.submit(a, vec![r.entry(CMD_A, &write_scratch(0, 0xa1))]);
    assert!(!r.shared.is_quiet());
    // Taken and worked on: still busy until a turn ends idle and marks it.
    assert_eq!(r.front.turn().unwrap(), Turn::Worked);
    assert!(!r.shared.is_quiet());
    r.shared.mark_idle();
    assert!(r.shared.is_quiet());
    // A back report makes it busy until the front takes it and ends a turn idle.
    crate::back::fence::FrontWake::wake(
        &*r.shared,
        &crate::back::fence::Report {
            released: 1,
            ..crate::back::fence::Report::default()
        },
    );
    assert!(!r.shared.is_quiet());
    r.shared.mark_idle();
    assert!(!r.shared.is_quiet(), "a report not yet taken keeps it busy");
    r.settle();
    r.shared.mark_idle();
    assert!(r.shared.is_quiet());
    r.shared.stop();
    assert!(!r.shared.is_quiet(), "a stopping front is not quiet");
}

#[test]
fn a_taken_back_report_keeps_the_front_busy_until_it_ends_idle() {
    // A report taken, with no request: the front resumes channels for it and has pushed nothing
    // yet, so it is busy until it ends a turn idle.
    let shared = FrontShared::new();
    assert!(shared.is_quiet());
    crate::back::fence::FrontWake::wake(
        &shared,
        &crate::back::fence::Report {
            latched: true,
            ..crate::back::fence::Report::default()
        },
    );
    let taken = shared.take();
    assert!(taken.requests.is_empty());
    assert!(taken.reports.released);
    assert!(!shared.is_quiet(), "the report is taken, not yet acted on");
    shared.mark_idle();
    assert!(shared.is_quiet());
}

#[test]
fn a_turn_that_takes_nothing_is_busy_until_it_ends_idle() {
    // A turn the acquire poll starts takes nothing and can still resume channels.
    let shared = FrontShared::new();
    let before = shared.quiet_turn().unwrap();
    let taken = shared.take();
    assert!(taken.requests.is_empty());
    assert_eq!(taken.reports, BackReports::default());
    assert_eq!(shared.quiet_turn(), None);
    shared.mark_idle();
    // Quiet again, one turn on.
    assert_eq!(shared.quiet_turn(), Some(before + 1));
}

#[test]
fn waiting_for_quiet_ends_when_the_front_ends_a_turn_idle() {
    let shared = Arc::new(FrontShared::new());
    // Quiet already: no wait.
    assert!(shared.wait_quiet(Duration::ZERO));
    let _taken = shared.take();
    // Busy, and nothing ends the turn: the wait runs out.
    assert!(!shared.wait_quiet(Duration::from_millis(1)));
    let marking = Arc::clone(&shared);
    let marker = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(5));
        marking.mark_idle();
    });
    assert!(shared.wait_quiet(Duration::from_secs(20)));
    marker.join().unwrap();
}
