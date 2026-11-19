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
