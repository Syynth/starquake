//! Check 4's attribute rules (decision log, 2026-11-17, "Address-space restore checks attributes
//! against state flags, not a permission table"): `Uncached` and `GpuAccessible` only on a state
//! with `CanChangeAttribute`, `PermissionLocked` only on a state with `CanPermissionLock`. Every
//! state on both sides of each rule, every combination of the three attributes on every state
//! against a model (so a rule that tests for exactly its attributes, not any one of them, fails),
//! the order against the checks around them, and a live space taken through every attribute
//! change the kernel makes, saved, restored and used.

use super::*;
use crate::memory::fixture::TEXT_PAGES;

/// The states with `FlagCanChangeAttribute`, read from `KMemoryState` in
/// `eden/src/core/hle/kernel/k_memory_block.h` @ cf654052 (`FlagsData` gives it to `CodeData`,
/// `Normal` and `AliasCodeData`; `Transferred` and `Insecure` name it themselves). Listed by
/// hand, not derived from [`flag::CAN_CHANGE_ATTRIBUTE`], so that a wrong state word fails here.
const CAN_CHANGE_ATTRIBUTE: [MemoryState; 5] = [
    MemoryState::CodeData,
    MemoryState::Normal,
    MemoryState::AliasCodeData,
    MemoryState::Transferred,
    MemoryState::Insecure,
];

/// The states with `FlagCanPermissionLock` (`CodeData` and `AliasCodeData`, same header).
const CAN_PERMISSION_LOCK: [MemoryState; 2] = [MemoryState::CodeData, MemoryState::AliasCodeData];

/// Each attribute check 4 tests against a state flag, the fault it gives on a state without
/// the flag, and the states with it.
fn rules() -> [(
    &'static str,
    MemoryAttribute,
    BlockFault,
    &'static [MemoryState],
); 3] {
    [
        (
            "uncached",
            MemoryAttribute::UNCACHED,
            BlockFault::ChangedAttribute,
            &CAN_CHANGE_ATTRIBUTE,
        ),
        (
            "GPU accessible",
            MemoryAttribute::GPU_ACCESSIBLE,
            BlockFault::ChangedAttribute,
            &CAN_CHANGE_ATTRIBUTE,
        ),
        (
            "permission locked",
            MemoryAttribute::PERMISSION_LOCKED,
            BlockFault::PermissionLock,
            &CAN_PERMISSION_LOCK,
        ),
    ]
}

#[test]
fn each_attribute_is_refused_exactly_on_the_states_without_its_flag() {
    // The thread-local page made each state that is not free, read-only (a permission the
    // kernel gives every state, code too), with and without the attribute. On a state with the
    // flag the attribute changes nothing: the restore goes on to the same outcome as without it
    // (the next check in the chain). On a state without it, the attribute's own fault.
    let rich = rich();
    let tls = thread_local(&rich);
    let read = MemoryPermission::from_svc(SvcPermission::Read).bits();
    for (name, attribute, fault, allowed) in rules() {
        let mut refused = 0;
        for &state in MemoryState::ALL {
            if state == MemoryState::Free {
                continue;
            }
            let mut case = Case::new(&rich);
            let index = at(&case.body, tls);
            let block = &mut case.body.blocks[index];
            block.state = state.word();
            block.permission = read;
            let without = case.restore().0.err();
            case.body.blocks[index].attribute |= attribute.bits();
            let with = case.restore().0.err();
            let own = Some(SnapshotError::Block { index, fault });
            if allowed.contains(&state) {
                assert_ne!(with, own, "{name} {state:?}");
                assert_eq!(with, without, "{name} {state:?}");
            } else {
                assert_eq!(with, own, "{name} {state:?}");
                refused += 1;
            }
        }
        // Every state but the free one and those allowed.
        assert_eq!(
            refused,
            MemoryState::ALL.len() - 1 - allowed.len(),
            "{name}"
        );
    }
}

/// The three attributes check 4 tests against a state flag, for building their combinations.
const RULED: [MemoryAttribute; 3] = [
    MemoryAttribute::UNCACHED,
    MemoryAttribute::GPU_ACCESSIBLE,
    MemoryAttribute::PERMISSION_LOCKED,
];

/// Every non-empty combination of [`RULED`]: the seven subsets of three attributes.
fn ruled_combinations() -> Vec<MemoryAttribute> {
    (1_u8..1 << RULED.len())
        .map(|subset| {
            RULED
                .iter()
                .enumerate()
                .filter(|(bit, _)| subset & (1 << bit) != 0)
                .fold(MemoryAttribute::NONE, |all, (_, &one)| all | one)
        })
        .collect()
}

/// The fault check 4's model gives `attribute` on `state`, from the hand-listed states: any of
/// `Uncached` and `GpuAccessible` on a state without `CanChangeAttribute` first, then
/// `PermissionLocked` on a state without `CanPermissionLock`; `None` when both rules pass.
fn model_fault(state: MemoryState, attribute: MemoryAttribute) -> Option<BlockFault> {
    let changed = attribute.contains(MemoryAttribute::UNCACHED)
        || attribute.contains(MemoryAttribute::GPU_ACCESSIBLE);
    let locked = attribute.contains(MemoryAttribute::PERMISSION_LOCKED);
    if changed && !CAN_CHANGE_ATTRIBUTE.contains(&state) {
        Some(BlockFault::ChangedAttribute)
    } else if locked && !CAN_PERMISSION_LOCK.contains(&state) {
        Some(BlockFault::PermissionLock)
    } else {
        None
    }
}

#[test]
fn every_combination_of_the_attributes_on_every_state_matches_the_model() {
    // Each of the seven combinations on the thread-local page made each state that is not
    // free, read-only. Combinations matter on the states with `CanChangeAttribute` and not
    // `CanPermissionLock` (`Normal`, `Transferred`, `Insecure`): there `Uncached` alone passes,
    // and `Uncached | PermissionLocked` must still be `PermissionLock`'s fault, so a rule that
    // tests for exactly its attributes rather than any one of them fails here.
    let rich = rich();
    let tls = thread_local(&rich);
    let read = MemoryPermission::from_svc(SvcPermission::Read).bits();
    let combinations = ruled_combinations();
    assert_eq!(combinations.len(), 7);
    let mut checked = 0;
    for &state in MemoryState::ALL {
        if state == MemoryState::Free {
            continue;
        }
        let mut case = Case::new(&rich);
        let index = at(&case.body, tls);
        let block = &mut case.body.blocks[index];
        block.state = state.word();
        block.permission = read;
        let without = case.restore().0.err();
        let saved = case.body.blocks[index].attribute;
        for &attribute in &combinations {
            case.body.blocks[index].attribute = saved | attribute.bits();
            let with = case.restore().0.err();
            let expected = model_fault(state, attribute).map_or_else(
                || without.clone(),
                |fault| Some(SnapshotError::Block { index, fault }),
            );
            assert_eq!(with, expected, "{attribute:?} on {state:?}");
            checked += 1;
        }
    }
    assert_eq!(checked, (MemoryState::ALL.len() - 1) * combinations.len());
}

#[test]
fn heap_memory_with_another_attribute_and_permission_locked_is_the_lock_s_fault() {
    // `Normal` has `CanChangeAttribute` but not `CanPermissionLock`. Heap blocks of the rich
    // space that are, or are made, uncached, GPU accessible or device shared restore as they
    // are; each of them permission locked too is refused with `PermissionLock`.
    let rich = rich();
    let heap = rich.heap;
    let uncached = MemoryAttribute::UNCACHED;
    let gpu = MemoryAttribute::GPU_ACCESSIBLE;
    let locked = MemoryAttribute::PERMISSION_LOCKED;
    for (name, address, extra) in [
        // The heap's last block (pages 14-15), made uncached.
        ("uncached", page(heap, 14), uncached),
        // The same block made GPU accessible.
        ("GPU accessible", page(heap, 14), gpu),
        // Page 9: already GPU accessible and uncached.
        (
            "GPU accessible and uncached",
            page(heap, 9),
            MemoryAttribute::NONE,
        ),
        // Page 3: device shared, with the use count of its three locks.
        ("device shared", page(heap, 3), MemoryAttribute::NONE),
    ] {
        let mut case = Case::new(&rich);
        let index = at(&case.body, address);
        case.body.blocks[index].attribute |= extra.bits();
        let attribute = MemoryAttribute::from_bits(case.body.blocks[index].attribute);
        assert_ne!(attribute, MemoryAttribute::NONE, "{name}");
        let restored = case.restore().0.unwrap();
        assert_eq!(
            restored.query_info(address).unwrap().attribute(),
            attribute,
            "{name}"
        );
        case.body.blocks[index].attribute |= locked.bits();
        assert_eq!(
            case.error(),
            SnapshotError::Block {
                index,
                fault: BlockFault::PermissionLock
            },
            "{name} and permission locked"
        );
    }
}

#[test]
fn each_attribute_restores_on_a_live_state_with_its_flag_and_is_refused_on_one_without() {
    // Pairs that differ only in the state: heap memory (`Normal`, `CanChangeAttribute`) uncached
    // or GPU accessible restores and keeps it, and the same block as `Code` (neither flag) is
    // refused; module data (`CodeData`, both flags) permission locked restores, and as `Code`
    // is refused. Each block is read-only so that `Code` takes its permission.
    let heap = rich().heap;
    let read = MemoryPermission::from_svc(SvcPermission::Read);
    // The heap's last page (no lock, alias or lend on it) and the data segment's last page.
    let heap_page = page(heap, 15);
    let data_page = DATA + 2 * PAGE_SIZE;
    for (name, address, attribute, fault) in [
        (
            "uncached heap",
            heap_page,
            MemoryAttribute::UNCACHED,
            BlockFault::ChangedAttribute,
        ),
        (
            "GPU-accessible heap",
            heap_page,
            MemoryAttribute::GPU_ACCESSIBLE,
            BlockFault::ChangedAttribute,
        ),
        (
            "permission-locked module data",
            data_page,
            MemoryAttribute::PERMISSION_LOCKED,
            BlockFault::PermissionLock,
        ),
    ] {
        // Reprotecting the page makes it a block of its own.
        let mut rich = rich();
        rich.space
            .set_memory_permission(address, PAGE_SIZE, SvcPermission::Read)
            .unwrap();
        rich.space
            .set_memory_attribute(address, PAGE_SIZE, attribute, attribute)
            .unwrap();
        let mut case = Case::new(&rich);
        let restored = case.restore().0.unwrap();
        let block = restored.query_info(address).unwrap();
        assert_eq!(block.attribute(), attribute, "{name}");
        assert_eq!(block.permission(), read, "{name}");
        let index = at(&case.body, address);
        case.body.blocks[index].state = MemoryState::Code.word();
        assert_eq!(
            case.error(),
            SnapshotError::Block { index, fault },
            "{name} as code"
        );
    }
}

#[test]
fn of_two_faults_in_one_block_the_attribute_rules_sit_between_the_permission_and_the_region() {
    // The thread-local page (`ThreadLocal`: neither flag) with two faults at once: the
    // permission check before the attribute rules, `CanChangeAttribute`'s rule before
    // `CanPermissionLock`'s, and both before the region check.
    let rich = rich();
    let tls = thread_local(&rich);
    let pairs: Vec<(&str, BlockChange, BlockFault)> = vec![
        (
            "a permission the state never has and uncached",
            Box::new(|b| {
                b.permission = READ_WRITE_EXECUTE;
                b.attribute |= MemoryAttribute::UNCACHED.bits();
            }),
            BlockFault::Permission,
        ),
        (
            "GPU accessible and permission locked",
            Box::new(|b| {
                b.attribute |=
                    (MemoryAttribute::GPU_ACCESSIBLE | MemoryAttribute::PERMISSION_LOCKED).bits();
            }),
            BlockFault::ChangedAttribute,
        ),
        (
            "uncached and the wrong region",
            Box::new(|b| {
                b.attribute |= MemoryAttribute::UNCACHED.bits();
                b.state = MemoryState::Stack.word();
            }),
            BlockFault::ChangedAttribute,
        ),
        (
            "permission locked and the wrong region",
            Box::new(|b| {
                b.attribute |= MemoryAttribute::PERMISSION_LOCKED.bits();
                b.state = MemoryState::Stack.word();
            }),
            BlockFault::PermissionLock,
        ),
    ];
    for (name, change, fault) in pairs {
        let mut case = Case::new(&rich);
        let index = at(&case.body, tls);
        change(&mut case.body.blocks[index]);
        assert_eq!(
            case.error(),
            SnapshotError::Block { index, fault },
            "{name}"
        );
    }
}

#[test]
fn an_attribute_fault_comes_after_the_counts_and_before_the_ranges() {
    // Two faults at once, against the checks on either side of check 4: a permission-locked
    // heap block (check 4) with a reservation one page short (check 3) is the reservation's
    // fault; with a heap size one page long instead (check 5's heap range, the counts made to
    // add up), it is the block's.
    let rich = rich();
    // The heap's last block: pages 14 and 15, above the lent pages.
    let heap_page = page(rich.heap, 14);
    let locked = MemoryAttribute::PERMISSION_LOCKED.bits();
    let mut case = Case::new(&rich);
    let index = at(&case.body, heap_page);
    case.body.blocks[index].attribute |= locked;
    let reserved = case.body.reserved - PAGE_SIZE;
    let used = case.body.reserved;
    case.body.reserved = reserved;
    assert_eq!(case.error(), SnapshotError::Reservation { reserved, used });
    // The heap one page longer than its blocks, with the largest heap and the reservation
    // made to fit it.
    let mut case = Case::new(&rich);
    case.body.blocks[index].attribute |= locked;
    let mut alone = Case::new(&rich);
    for body in [&mut case.body, &mut alone.body] {
        body.heap_size += PAGE_SIZE;
        body.reserved += PAGE_SIZE;
    }
    assert_eq!(alone.error(), SnapshotError::HeapRange);
    assert_eq!(
        case.error(),
        SnapshotError::Block {
            index,
            fault: BlockFault::PermissionLock
        }
    );
}

#[test]
fn a_live_space_taken_through_every_attribute_change_saves_restores_and_keeps_its_rules() {
    // The fixture's loaded space with heap pages uncached and GPU accessible (one of them
    // device shared too), the text made writable (`Code` to `CodeData`) and then uncached, GPU
    // accessible and permission locked, and the data segment permission locked: each restores
    // as it was, and the restored space enforces the attributes as the live one does.
    let (mut space, _log, _) = loaded();
    let heap = space.set_heap_size(8 * PAGE_SIZE).unwrap();
    let uncached = MemoryAttribute::UNCACHED;
    let gpu = MemoryAttribute::GPU_ACCESSIBLE;
    let locked = MemoryAttribute::PERMISSION_LOCKED;
    let text = TEXT_PAGES * PAGE_SIZE;
    space
        .set_memory_attribute(page(heap, 1), 2 * PAGE_SIZE, uncached | gpu, uncached | gpu)
        .unwrap();
    space
        .set_memory_attribute(page(heap, 4), PAGE_SIZE, gpu, gpu)
        .unwrap();
    space.lock_device_share(page(heap, 1), PAGE_SIZE).unwrap();
    space
        .set_process_memory_permission(CODE, text, SvcPermission::ReadWrite)
        .unwrap();
    assert_eq!(
        space.query_info(CODE).unwrap().state(),
        MemoryState::CodeData
    );
    let all = uncached | gpu | locked;
    space.set_memory_attribute(CODE, text, all, all).unwrap();
    space
        .set_memory_attribute(DATA, PAGE_SIZE, locked, locked)
        .unwrap();
    let bytes = space.save().unwrap();
    let mut restored = restore_with(&bytes, &Resolver::new()).0.unwrap();
    assert_eq!(Body::of(&restored), Body::of(&space));
    assert_eq!(
        restored.query_info(page(heap, 1)).unwrap().attribute(),
        uncached | gpu | MemoryAttribute::DEVICE_SHARED
    );
    assert_eq!(restored.query_info(CODE).unwrap().attribute(), all);
    // Used: on both spaces, a permission-locked block cannot be reprotected, and its uncached
    // bit can be cleared; heap memory still cannot be permission locked.
    for space in [&mut space, &mut restored] {
        assert_eq!(
            space.set_process_memory_permission(CODE, text, SvcPermission::Read),
            Err(SvcError::Kernel(KernelError::InvalidCurrentMemory))
        );
        assert_eq!(
            space.set_memory_attribute(page(heap, 6), PAGE_SIZE, locked, locked),
            Err(SvcError::Kernel(KernelError::InvalidCurrentMemory))
        );
        space
            .set_memory_attribute(CODE, text, uncached, MemoryAttribute::NONE)
            .unwrap();
        assert_eq!(space.query_info(CODE).unwrap().attribute(), gpu | locked);
    }
    assert_eq!(restored.save().unwrap(), space.save().unwrap());
}
