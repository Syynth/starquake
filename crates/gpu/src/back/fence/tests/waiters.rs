//! The acquire waiter table: register, replace, remove, latch, and the close-trigger query.

use super::super::{Waiter, WaiterTable};
use crate::interface::ChannelId;
use crate::sema::AcquireOp;

const ADDRESS: u64 = 0x1_2345_6780;

fn waiter(channel: u32, op: AcquireOp, payload: u32) -> Waiter {
    Waiter {
        channel: ChannelId(channel),
        address: ADDRESS,
        op,
        payload,
    }
}

#[test]
fn a_fresh_table_is_empty_and_answers_no() {
    let table = WaiterTable::new();
    assert!(table.is_empty());
    assert_eq!(table.len(), 0);
    assert!(!table.has_waiter(ADDRESS));
    assert!(!table.is_latched(ChannelId(0)));
    assert!(!table.on_write(ADDRESS, 0));
    assert!(!table.remove(ChannelId(0)));
}

#[test]
fn a_write_that_meets_the_condition_latches_and_a_later_write_does_not_unlatch() {
    let table = WaiterTable::new();
    table.register(waiter(3, AcquireOp::Equal, 0x5293));
    assert!(table.has_waiter(ADDRESS));
    // Not met: nothing latches.
    assert!(!table.on_write(ADDRESS, 0x5292));
    assert!(!table.is_latched(ChannelId(3)));
    // Met: latched, and no longer counted as waiting for a release.
    assert!(table.on_write(ADDRESS, 0x5293));
    assert!(table.is_latched(ChannelId(3)));
    assert!(!table.has_waiter(ADDRESS));
    // Row 70: the counter moves on past the value; the waiter stays latched, and a second
    // latch of it is not reported.
    assert!(!table.on_write(ADDRESS, 0x5310));
    assert!(table.is_latched(ChannelId(3)));
    // The value comes round again: the waiter is already latched, so nothing new is.
    assert!(!table.on_write(ADDRESS, 0x5293));
    assert!(table.is_latched(ChannelId(3)));
}

#[test]
fn a_write_elsewhere_latches_nothing() {
    let table = WaiterTable::new();
    table.register(waiter(1, AcquireOp::Equal, 7));
    assert!(!table.on_write(ADDRESS + 4, 7));
    assert!(!table.on_write(ADDRESS - 4, 7));
    assert!(!table.is_latched(ChannelId(1)));
    assert!(!table.has_waiter(ADDRESS + 4));
}

#[test]
fn every_acquire_condition_latches() {
    // Divergence from Eden-NX2, whose waiters match only equal, greater-or-equal and and.
    // Each case: the condition, its payload, a value that meets it and one that does not.
    let cases = [
        (AcquireOp::Equal, 0x100, 0x100, 0x101),
        (AcquireOp::GreaterOrEqual, 0x100, 0x100, 0xff),
        (
            AcquireOp::CircularGreaterOrEqual,
            0xffff_fff0,
            0x10,
            0xffff_ffe0,
        ),
        (AcquireOp::And, 0x0f0, 0x010, 0x100),
        (AcquireOp::Nor, 0x0f0, 0x100, 0x010),
    ];
    for (op, payload, meets, misses) in cases {
        let table = WaiterTable::new();
        table.register(waiter(9, op, payload));
        assert!(!table.on_write(ADDRESS, misses), "{op:?} {misses:#x}");
        assert!(table.on_write(ADDRESS, meets), "{op:?} {meets:#x}");
        assert!(table.is_latched(ChannelId(9)), "{op:?}");
    }
}

#[test]
fn registering_again_replaces_the_channels_waiter_unlatched() {
    let table = WaiterTable::new();
    table.register(waiter(2, AcquireOp::Equal, 1));
    assert!(table.on_write(ADDRESS, 1));
    table.register(waiter(2, AcquireOp::Equal, 5));
    assert_eq!(table.len(), 1);
    assert!(!table.is_latched(ChannelId(2)));
    assert!(!table.on_write(ADDRESS, 1));
    assert!(table.on_write(ADDRESS, 5));
}

#[test]
fn removing_a_waiter_leaves_the_others() {
    let table = WaiterTable::new();
    table.register(waiter(4, AcquireOp::Equal, 1));
    table.register(waiter(5, AcquireOp::Equal, 1));
    assert_eq!(table.len(), 2);
    assert!(table.remove(ChannelId(4)));
    assert!(!table.remove(ChannelId(4)));
    assert_eq!(table.len(), 1);
    assert!(!table.is_empty());
    // Only channel 5 latches; channel 4 is gone.
    assert!(table.on_write(ADDRESS, 1));
    assert!(table.is_latched(ChannelId(5)));
    assert!(!table.is_latched(ChannelId(4)));
}

#[test]
fn one_write_latches_every_waiter_it_meets_and_only_those() {
    let table = WaiterTable::new();
    table.register(waiter(6, AcquireOp::GreaterOrEqual, 3));
    table.register(waiter(7, AcquireOp::GreaterOrEqual, 4));
    table.register(waiter(8, AcquireOp::GreaterOrEqual, 5));
    assert!(table.on_write(ADDRESS, 4));
    assert!(table.is_latched(ChannelId(6)));
    assert!(table.is_latched(ChannelId(7)));
    assert!(!table.is_latched(ChannelId(8)));
    // Channel 8 still waits for a release there.
    assert!(table.has_waiter(ADDRESS));
}

#[test]
fn waiters_meet_by_gpu_address_alone() {
    // Open (module docs): two channels whose acquires read the same GPU address are latched by
    // one release there, whichever address space they belong to.
    let table = WaiterTable::new();
    table.register(waiter(10, AcquireOp::Equal, 2));
    table.register(waiter(11, AcquireOp::Equal, 2));
    assert!(table.on_write(ADDRESS, 2));
    assert!(table.is_latched(ChannelId(10)));
    assert!(table.is_latched(ChannelId(11)));
}
