//! Check 11 (`snapshot::transfer`, [`AddressSpace::check_lends`]): the locked memory against
//! the transfer memories that lent it. The rich space's lend passes and is needed; one fault
//! case per check with its exact error and the accepted case beside each edge; the block rules
//! against a model on every state, attribute set and permission; and random live operations,
//! lends and take-backs among them, leave a space that passes before and after a round trip.

use super::super::transfer::block_fault;
use super::*;
use crate::id::ProcessId;
use crate::memory::shared::TransferMemory;

/// The states with `FlagCanTransfer`, read from `KMemoryState` in
/// `eden/src/core/hle/kernel/k_memory_block.h` @ cf654052: `FlagsData` gives it, and only
/// `CodeData`, `Normal` and `AliasCodeData` take `FlagsData`. Each has `FlagReferenceCounted`
/// too (`FlagsData` again). Listed by hand, not derived from the flag constants, so that a wrong
/// state word fails here.
const LENDABLE: [MemoryState; 3] = [
    MemoryState::CodeData,
    MemoryState::Normal,
    MemoryState::AliasCodeData,
];

/// The owner of every lend here. Its id does not matter to the check, which is handed one
/// owner's lends.
fn owner() -> ProcessId {
    ProcessId::from_parts(2, 3)
}

/// A locked transfer memory of `pages` pages at `address`, of which the owner keeps
/// `permission`.
fn lend(address: u64, pages: u64, permission: SvcPermission) -> TransferMemory {
    TransferMemory::new(owner(), address, pages * PAGE_SIZE, permission).with_pages_locked()
}

/// The rich space's lend ([`rich`]): heap pages 12-13, the owner keeping read.
fn rich_lend(rich: &Rich) -> TransferMemory {
    lend(page(rich.heap, 12), 2, SvcPermission::Read)
}

/// The fault `bad` gives beside the rich space's own lend.
fn fault_of(rich: &Rich, bad: TransferMemory) -> Result<(), LendError> {
    rich.space.check_lends(&[rich_lend(rich), bad])
}

/// The error for `bad` failing with `fault`.
fn lend_error(bad: TransferMemory, fault: LendFault) -> Result<(), LendError> {
    Err(LendError::Lend {
        address: bad.address(),
        size: bad.size(),
        fault,
    })
}

#[test]
fn the_rich_spaces_lend_passes_live_and_restored_and_is_needed() {
    let rich = rich();
    assert_eq!(rich.space.check_lends(&[rich_lend(&rich)]), Ok(()));
    let bytes = rich.space.save().unwrap();
    let restored = restore_with(&bytes, &rich.objects).0.unwrap();
    assert_eq!(restored.check_lends(&[rich_lend(&rich)]), Ok(()));
    // Without it, its first page is locked memory nothing lent; the aliases' sources (heap
    // pages 6 and 10-11, locked too and below it) need no lend.
    let unlent = Err(LendError::Unlent {
        address: page(rich.heap, 12),
    });
    assert_eq!(rich.space.check_lends(&[]), unlent);
    assert_eq!(restored.check_lends(&[]), unlent);
}

#[test]
fn a_lend_must_be_whole_pages_inside_the_address_space() {
    let rich = rich();
    let space = rich.space.regions().address_space;
    let at = page(rich.heap, 14);
    let shape = |bad| lend_error(bad, LendFault::Shape);
    // Not page aligned at its start, or in its size, or empty.
    let unaligned = TransferMemory::new(owner(), at + 1, PAGE_SIZE, SvcPermission::None);
    assert_eq!(fault_of(&rich, unaligned), shape(unaligned));
    let ragged = TransferMemory::new(owner(), at, PAGE_SIZE + 1, SvcPermission::None);
    assert_eq!(fault_of(&rich, ragged), shape(ragged));
    let empty = TransferMemory::new(owner(), at, 0, SvcPermission::None);
    assert_eq!(fault_of(&rich, empty), shape(empty));
    // The last page of the address space has the shape (and is not locked, the next check);
    // one page more runs past the end.
    let last = lend(space.end - PAGE_SIZE, 1, SvcPermission::None);
    assert_eq!(
        fault_of(&rich, last),
        lend_error(last, LendFault::NotLocked)
    );
    let past = lend(space.end - PAGE_SIZE, 2, SvcPermission::None);
    assert_eq!(fault_of(&rich, past), shape(past));
    // The first page of the address space has the shape (the space starts at zero, so there
    // is no page below it).
    assert_eq!(space.start, 0);
    let first = lend(space.start, 1, SvcPermission::None);
    assert_eq!(
        rich.space.check_lends(&[first]),
        lend_error(first, LendFault::NotLocked)
    );
    // A range whose end wraps the 64-bit space.
    let wraps = lend(u64::MAX - (PAGE_SIZE - 1), 2, SvcPermission::None);
    assert_eq!(rich.space.check_lends(&[wraps]), shape(wraps));
}

#[test]
fn lends_may_touch_but_not_share_a_page_and_are_taken_in_address_order() {
    let mut rich = rich();
    let first = rich_lend(&rich);
    // A second live lend that starts exactly where the first ends.
    rich.space
        .lock_for_transfer_memory(page(rich.heap, 14), 2 * PAGE_SIZE, SvcPermission::None)
        .unwrap();
    let second = lend(page(rich.heap, 14), 2, SvcPermission::None);
    assert_eq!(rich.space.check_lends(&[first, second]), Ok(()));
    // Handed over in either order: they are sorted by address first.
    assert_eq!(rich.space.check_lends(&[second, first]), Ok(()));
    // One that starts a page before the first ends overlaps it, and so does the same lend
    // twice.
    let early = lend(page(rich.heap, 13), 3, SvcPermission::None);
    assert_eq!(
        rich.space.check_lends(&[first, early]),
        lend_error(early, LendFault::Overlap)
    );
    assert_eq!(
        rich.space.check_lends(&[second, first, second]),
        lend_error(second, LendFault::Overlap)
    );
}

#[test]
fn a_lend_shares_no_page_with_an_aliases_source() {
    let rich = rich();
    // The sources are heap page 6 and heap pages 10-11, locked and kernel-read only, as a lend
    // the owner keeps nothing of leaves its pages.
    for bad in [
        lend(page(rich.heap, 6), 1, SvcPermission::None),
        lend(page(rich.heap, 11), 1, SvcPermission::None),
        lend(page(rich.heap, 5), 2, SvcPermission::None),
        // Starts exactly where the first source ends and reaches the second: the walk must
        // move past the first source to see the overlap with heap page 10.
        lend(page(rich.heap, 7), 4, SvcPermission::None),
    ] {
        assert_eq!(
            fault_of(&rich, bad),
            lend_error(bad, LendFault::AliasSource),
            "{bad:?}"
        );
    }
    // The pages right beside a source are no source's: they reach the next check.
    for beside in [
        lend(page(rich.heap, 7), 1, SvcPermission::None),
        lend(page(rich.heap, 9), 1, SvcPermission::None),
    ] {
        assert_eq!(
            fault_of(&rich, beside),
            lend_error(beside, LendFault::NotLocked),
            "{beside:?}"
        );
    }
}

#[test]
fn every_block_of_a_lend_has_exactly_the_locked_attribute() {
    let rich = rich();
    // One page more than the live lend: heap page 14 is not locked.
    let longer = lend(page(rich.heap, 12), 3, SvcPermission::Read);
    assert_eq!(
        rich.space.check_lends(&[longer]),
        lend_error(longer, LendFault::NotLocked)
    );
    // The lend's blocks also uncached: an attribute no lend leaves.
    let mut case = Case::new(&rich);
    let index = at(&case.body, page(rich.heap, 12));
    case.body.blocks[index].attribute |= MemoryAttribute::UNCACHED.bits();
    let forged = case.restore().0.unwrap();
    assert_eq!(
        forged.check_lends(&[rich_lend(&rich)]),
        lend_error(rich_lend(&rich), LendFault::NotLocked)
    );
    // Device shared too, with the counts of one device-share lock exactly over it: a counted
    // attribute, but live the lend refuses a device-shared range and a device-share lock
    // refuses a lent one, so no lend's block has it.
    let mut case = Case::new(&rich);
    device_share_alone(&mut case.body, page(rich.heap, 12));
    insert_lock(&mut case.body, page(rich.heap, 12), 2 * PAGE_SIZE);
    let forged = case.restore().0.unwrap();
    assert_eq!(
        forged.check_lends(&[rich_lend(&rich)]),
        lend_error(rich_lend(&rich), LendFault::NotLocked)
    );
    let mut rich = rich;
    let heap = rich.heap;
    assert_eq!(
        rich.space
            .lock_for_transfer_memory(page(heap, 2), PAGE_SIZE, SvcPermission::Read),
        Err(KernelError::InvalidCurrentMemory.into())
    );
    assert_eq!(
        rich.space.lock_device_share(page(heap, 12), PAGE_SIZE),
        Err(KernelError::InvalidCurrentMemory.into())
    );
}

/// Where the main thread's stack starts: the first `Stack` block, at the stack region's bottom
/// (the aliases are near its top).
fn main_stack(space: &AddressSpace) -> u64 {
    space
        .blocks()
        .find(|block| block.state() == MemoryState::Stack)
        .unwrap()
        .address()
}

#[test]
fn a_locked_block_of_a_state_that_cannot_be_lent_is_refused() {
    // The main thread's stack, forged locked and kernel-read only: the restore takes `Locked`
    // on any block (module docs, "Not checked here"), and `Stack` has no `FlagCanTransfer`.
    let rich = rich();
    let mut case = Case::new(&rich);
    let stack = main_stack(&rich.space);
    let index = at(&case.body, stack);
    let block = &mut case.body.blocks[index];
    assert_eq!(block.state, MemoryState::Stack.word());
    block.attribute |= MemoryAttribute::LOCKED.bits();
    block.permission = MemoryPermission::from_svc(SvcPermission::None).bits();
    let pages = block.pages;
    let forged = case.restore().0.unwrap();
    let bad = lend(stack, pages, SvcPermission::None);
    assert_eq!(
        forged.check_lends(&[rich_lend(&rich), bad]),
        lend_error(bad, LendFault::State)
    );
    // Without the lend, its pages are locked memory nothing lent.
    assert_eq!(
        forged.check_lends(&[rich_lend(&rich)]),
        Err(LendError::Unlent { address: stack })
    );
}

#[test]
fn a_lends_blocks_have_the_permission_its_owner_keeps() {
    let rich = rich();
    for permission in SvcPermission::ALL {
        let candidate = lend(page(rich.heap, 12), 2, permission);
        let expected = if permission == SvcPermission::Read {
            Ok(())
        } else {
            lend_error(candidate, LendFault::Permission)
        };
        assert_eq!(
            rich.space.check_lends(&[candidate]),
            expected,
            "{permission:?}"
        );
    }
}

#[test]
fn every_locked_page_is_a_lends_or_an_aliases_source() {
    // Heap pages 14-15 forged locked, kernel-read only.
    let rich = rich();
    let mut case = Case::new(&rich);
    let index = at(&case.body, page(rich.heap, 14));
    let block = &mut case.body.blocks[index];
    assert_eq!(block.pages, 2);
    block.attribute |= MemoryAttribute::LOCKED.bits();
    block.permission = MemoryPermission::from_svc(SvcPermission::None).bits();
    let forged = case.restore().0.unwrap();
    let own = rich_lend(&rich);
    assert_eq!(
        forged.check_lends(&[own]),
        Err(LendError::Unlent {
            address: page(rich.heap, 14)
        })
    );
    // A lend of the first page leaves the second.
    let one = lend(page(rich.heap, 14), 1, SvcPermission::None);
    assert_eq!(
        forged.check_lends(&[own, one]),
        Err(LendError::Unlent {
            address: page(rich.heap, 15)
        })
    );
    // Two lends of a page each cover the block between them.
    let two = lend(page(rich.heap, 15), 1, SvcPermission::None);
    assert_eq!(forged.check_lends(&[two, own, one]), Ok(()));
}

#[test]
fn the_lend_checks_come_in_their_documented_order() {
    let rich = rich();
    // Shape before overlap: a ragged lend that also overlaps the one before.
    let ragged = TransferMemory::new(
        owner(),
        page(rich.heap, 13),
        PAGE_SIZE + 1,
        SvcPermission::Read,
    );
    assert_eq!(
        fault_of(&rich, ragged),
        lend_error(ragged, LendFault::Shape)
    );
    // Overlap before the alias source: overlapping the lend before it and an alias's source
    // (heap page 10).
    let mut grown = rich;
    grown
        .space
        .lock_for_transfer_memory(page(grown.heap, 7), PAGE_SIZE, SvcPermission::None)
        .unwrap();
    let low = lend(page(grown.heap, 7), 1, SvcPermission::None);
    let both = lend(page(grown.heap, 7), 4, SvcPermission::None);
    assert_eq!(
        grown.space.check_lends(&[low, both, rich_lend(&grown)]),
        lend_error(both, LendFault::Overlap)
    );
    let rich = grown;
    // The alias source before the block checks: the source's block is locked, but its
    // permission is not what a read lend keeps.
    let source = lend(page(rich.heap, 6), 1, SvcPermission::Read);
    assert_eq!(
        fault_of(&rich, source),
        lend_error(source, LendFault::AliasSource)
    );
    // Not locked before the state: the stack is neither locked nor lendable.
    let stack = lend(main_stack(&rich.space), 1, SvcPermission::None);
    assert_eq!(
        fault_of(&rich, stack),
        lend_error(stack, LendFault::NotLocked)
    );
    // The lends before the pages nothing lent: a faulty lend, and heap page 7, lent above but
    // not handed over.
    let longer = lend(page(rich.heap, 12), 3, SvcPermission::Read);
    assert_eq!(
        rich.space.check_lends(&[longer]),
        lend_error(longer, LendFault::NotLocked)
    );
    // The first faulty lend by address, though handed over last.
    let late = lend(page(rich.heap, 14), 1, SvcPermission::None);
    let early = lend(page(rich.heap, 9), 1, SvcPermission::None);
    assert_eq!(
        rich.space.check_lends(&[rich_lend(&rich), late, early]),
        lend_error(early, LendFault::NotLocked)
    );
}

/// What [`block_fault`] should find, written from the rules in the module docs of
/// `snapshot::transfer`, with the lendable states listed by hand ([`LENDABLE`]).
fn model(
    attribute: MemoryAttribute,
    state: MemoryState,
    first: MemoryState,
    permission: MemoryPermission,
    keeps: MemoryPermission,
) -> Option<LendFault> {
    if attribute != MemoryAttribute::LOCKED {
        Some(LendFault::NotLocked)
    } else if !LENDABLE.contains(&state) || state != first {
        Some(LendFault::State)
    } else if permission != keeps {
        Some(LendFault::Permission)
    } else {
        None
    }
}

#[test]
fn the_block_rules_agree_with_the_model_on_every_state_attribute_set_and_permission() {
    let keeps = MemoryPermission::from_svc(SvcPermission::Read);
    let named = [
        MemoryAttribute::LOCKED,
        MemoryAttribute::IPC_LOCKED,
        MemoryAttribute::DEVICE_SHARED,
        MemoryAttribute::UNCACHED,
        MemoryAttribute::PERMISSION_LOCKED,
        MemoryAttribute::GPU_ACCESSIBLE,
    ];
    let mut seen = BTreeSet::new();
    for &state in MemoryState::ALL {
        for &first in &[state, MemoryState::Normal] {
            for subset in 0..1_u32 << named.len() {
                let attribute = named
                    .iter()
                    .enumerate()
                    .filter(|(bit, _)| subset & (1 << bit) != 0)
                    .fold(MemoryAttribute::NONE, |set, (_, &one)| set | one);
                for svc in SvcPermission::ALL {
                    let permission = MemoryPermission::from_svc(svc);
                    let block =
                        MemoryBlock::new(0x1000_0000, 1, state, permission, attribute).unwrap();
                    let head =
                        MemoryBlock::new(0x0fff_f000, 1, first, keeps, MemoryAttribute::LOCKED)
                            .unwrap();
                    let found = block_fault(&block, &head, keeps);
                    assert_eq!(
                        found,
                        model(attribute, state, first, permission, keeps),
                        "{state:?} after {first:?}, {attribute:?}, {svc:?}"
                    );
                    seen.insert(format!("{found:?}"));
                }
            }
        }
    }
    // Every outcome is reached.
    assert_eq!(seen.len(), 4, "{seen:?}");
}

/// One operation on a live address space for the generated test below: an operation of the
/// address space's own generated test, a lend the model records, or the take-back of a lend
/// the model holds, of exactly its range, as `KTransferMemory::Finalize` does.
#[derive(Debug, Clone)]
enum LendOp {
    Other(Op),
    Lend(u64, u64, usize),
    TakeBack(usize),
}

fn any_lend_op() -> impl Strategy<Value = LendOp> {
    prop_oneof![
        3 => any_op().prop_map(LendOp::Other),
        2 => (0_u64..12, 1_u64..5, 0_usize..3).prop_map(|(at, n, keep)| LendOp::Lend(at, n, keep)),
        1 => any::<usize>().prop_map(LendOp::TakeBack),
    ]
}

/// Applies `op` to `space` and to `lends`, the model of its live transfer memories.
fn apply_lend(space: &mut AddressSpace, lends: &mut Vec<TransferMemory>, op: &LendOp) {
    let heap = space.regions().heap.start;
    match *op {
        // The address space's own lend and take-back keep no model; the take-back unlocks any
        // range, which the kernel never does.
        LendOp::Other(Op::Lend(..) | Op::TakeBack(..)) => {}
        LendOp::Other(ref other) => {
            apply(space, other);
        }
        LendOp::Lend(at, n, keep) => {
            let keeps = [
                SvcPermission::None,
                SvcPermission::Read,
                SvcPermission::ReadWrite,
            ];
            let permission = keeps[keep % keeps.len()];
            let made = lend(page(heap, at), n, permission);
            if space
                .lock_for_transfer_memory(made.address(), made.size(), permission)
                .is_ok()
            {
                lends.push(made);
            }
        }
        LendOp::TakeBack(pick) => {
            if lends.is_empty() {
                return;
            }
            let gone = lends.remove(pick % lends.len());
            space
                .unlock_for_transfer_memory(gone.address(), gone.size())
                .unwrap();
        }
    }
}

proptest! {
    #![proptest_config(config(96))]

    /// Random operations on the fixture's loaded address space, lends and exact take-backs
    /// among them: after each, the space passes check 11 with the live lends, before and after
    /// a round trip, and fails it with any one of them left out.
    #[test]
    fn a_space_built_by_random_operations_passes_with_its_live_lends(
        ops in prop::collection::vec(any_lend_op(), 1..24)
    ) {
        let (mut space, _log, _) = loaded();
        let mut lends = Vec::new();
        for op in &ops {
            apply_lend(&mut space, &mut lends, op);
            prop_assert_eq!(space.check_lends(&lends), Ok(()), "after {:?}", op);
            let bytes = space.save();
            prop_assert!(bytes.is_ok(), "save after {:?}: {:?}", op, bytes);
            let (restored, _) = restore_with(&bytes.unwrap_or_default(), &Resolver::new());
            prop_assert!(restored.is_ok(), "restore after {:?}", op);
            if let Ok(restored) = restored {
                prop_assert_eq!(restored.check_lends(&lends), Ok(()), "restored after {:?}", op);
            }
            for left_out in 0..lends.len() {
                let mut fewer = lends.clone();
                let gone = fewer.remove(left_out);
                prop_assert_eq!(
                    space.check_lends(&fewer),
                    Err(LendError::Unlent { address: gone.address() })
                );
            }
        }
    }
}

#[test]
fn the_random_operations_reach_neighbouring_lends_and_a_lend_beside_an_alias() {
    // A fixed sequence of the generated kind, so the property test is known to reach lends
    // side by side, a lend beside an alias's source, and a take-back between others.
    let (mut space, _log, _) = loaded();
    let mut lends = Vec::new();
    for op in [
        LendOp::Other(Op::Heap(12)),
        LendOp::Other(Op::Alias {
            dst: 0,
            src: 4,
            n: 1,
        }),
        LendOp::Lend(2, 2, 0),
        LendOp::Lend(5, 1, 1),
        LendOp::Lend(6, 2, 2),
        LendOp::TakeBack(1),
        LendOp::Lend(5, 1, 2),
    ] {
        apply_lend(&mut space, &mut lends, &op);
    }
    let ranges: Vec<(u64, u64)> = lends
        .iter()
        .map(|lend| (lend.address(), lend.size()))
        .collect();
    let heap = space.regions().heap.start;
    assert_eq!(
        ranges,
        [
            (page(heap, 2), 2 * PAGE_SIZE),
            (page(heap, 6), 2 * PAGE_SIZE),
            (page(heap, 5), PAGE_SIZE),
        ]
    );
    assert_eq!(space.check_lends(&lends), Ok(()));
}
