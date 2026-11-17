//! The `MapMemory` aliases in the address space snapshot: the version they came in with, round
//! trips of the shapes live operations leave (side by side, two runs in one block, part of a
//! block, held by a device), the restored aliases in use, the translation's order and undo,
//! and each refusal of check 8 at its edges, with its exact error.

use super::*;
use crate::memory::space::alias::LOCKED_SOURCE;

/// The fixture's loaded address space with 16 pages of heap, the log emptied, and where the
/// heap and the free stack pages start ([`stack_top`]).
fn heaped() -> (AddressSpace, MapperLog, u64, u64) {
    let (mut space, log, _) = loaded();
    let heap = space.set_heap_size(16 * PAGE_SIZE).unwrap();
    let stack = stack_top(&space, 0);
    drop(log.take());
    (space, log, heap, stack)
}

/// `body` encoded and restored over a new recording mapper, with no shared memory.
fn restore_body(body: &Body) -> (Result<AddressSpace, SnapshotError>, Vec<MapperCall>) {
    let bytes = encode(FORMAT, body).unwrap();
    let (restored, log) = restore_with(&bytes, &Resolver::new());
    (restored, log.take())
}

/// The error `body` restores with.
fn error_of(body: &Body) -> SnapshotError {
    restore_body(body).0.unwrap_err()
}

const fn alias(index: usize, fault: AliasFault) -> SnapshotError {
    SnapshotError::Alias { index, fault }
}

// ----- The version --------------------------------------------------------------------------------

#[test]
fn a_version_1_snapshot_is_refused_with_the_version_error() {
    let (space, ..) = heaped();
    let bytes = space.save().unwrap();
    // This build writes its own version after the magic.
    let mut header = ADDRESS_SPACE_MAGIC.to_vec();
    header.extend(varint(u64::from(ADDRESS_SPACE_VERSION)));
    assert_eq!(bytes.get(..header.len()), Some(&header[..]));
    // Version 1's body is this one without the aliases, which a space with none ends with as
    // their count, one byte of zero.
    let body = &bytes[header.len()..];
    assert_eq!(body.last(), Some(&0));
    let version_1 = 1;
    let mut old = ADDRESS_SPACE_MAGIC.to_vec();
    old.extend(varint(version_1));
    old.extend(&body[..body.len() - 1]);
    assert_eq!(
        restore_with(&old, &Resolver::new()).0.unwrap_err(),
        SnapshotError::Format(FormatError::UnsupportedVersion {
            name: "address space",
            found: 1,
            supported: ADDRESS_SPACE_VERSION
        })
    );
    // The same bytes under this build's version are a body cut short: the aliases are new.
    let mut cut = header;
    cut.extend(&body[..body.len() - 1]);
    assert!(matches!(
        restore_with(&cut, &Resolver::new()).0.unwrap_err(),
        SnapshotError::Format(FormatError::Body { .. })
    ));
}

// ----- Round trips of what live operations leave ----------------------------------------------------

#[test]
fn two_aliases_side_by_side_restore_and_map_their_sources_pages_last() {
    let (mut space, _log, heap, stack) = heaped();
    space
        .map_memory(page(stack, 0), page(heap, 4), 2 * PAGE_SIZE)
        .unwrap();
    space
        .map_memory(page(stack, 2), page(heap, 6), PAGE_SIZE)
        .unwrap();
    let bytes = space.save().unwrap();
    let (restored, log) = restore_with(&bytes, &Resolver::new());
    let mut restored = restored.unwrap();
    let body = Body::of(&space);
    assert_eq!(Body::of(&restored), body);
    assert_eq!(
        restored.aliases.runs(),
        [
            (page(stack, 0), page(heap, 4), 2 * PAGE_SIZE),
            (page(stack, 2), page(heap, 6), PAGE_SIZE)
        ]
    );
    let calls = log.take();
    assert_eq!(calls, expected_calls(&body, &Resolver::new()));
    // No fresh mapping over the destinations; the two aliases last, in address order.
    let tail = [
        MapperCall::MapAlias {
            address: page(stack, 0),
            source: page(heap, 4),
            size: 2 * PAGE_SIZE,
            perms: Perms::ReadWrite,
        },
        MapperCall::MapAlias {
            address: page(stack, 2),
            source: page(heap, 6),
            size: PAGE_SIZE,
            perms: Perms::ReadWrite,
        },
    ];
    assert_eq!(calls.get(calls.len() - 2..), Some(&tail[..]));
    assert!(!calls.iter().any(|call| matches!(
        call,
        MapperCall::Map { address, .. } if *address >= page(stack, 0) && *address < page(stack, 3)
    )));
    // The sources were mapped as they are, with no access for the guest.
    assert!(calls.contains(&MapperCall::Map {
        address: page(heap, 4),
        size: 2 * PAGE_SIZE,
        perms: LOCKED_SOURCE.guest().unwrap(),
    }));
    // The restored space unmaps both, as the original would.
    assert_eq!(
        restored.unmap_memory(page(stack, 0), page(heap, 4), 2 * PAGE_SIZE),
        Ok(())
    );
    assert_eq!(
        restored.unmap_memory(page(stack, 2), page(heap, 6), PAGE_SIZE),
        Ok(())
    );
    assert_eq!(restored.aliases.runs(), []);
    let source = restored.query_info(page(heap, 4)).unwrap();
    assert_eq!(source.permission(), MemoryPermission::USER_READ_WRITE);
    assert_eq!(source.attribute(), MemoryAttribute::NONE);
}

#[test]
fn a_block_holding_two_aliases_and_a_block_partly_aliased_restore() {
    let (mut space, _log, heap, stack) = heaped();
    // Two runs in one block: three pages aliased, the first unmapped, and the hole aliased
    // from another source. The new block has no barrier above it to keep it from the rest.
    space
        .map_memory(page(stack, 0), page(heap, 0), 3 * PAGE_SIZE)
        .unwrap();
    space
        .unmap_memory(page(stack, 0), page(heap, 0), PAGE_SIZE)
        .unwrap();
    space
        .map_memory(page(stack, 0), page(heap, 8), PAGE_SIZE)
        .unwrap();
    let block = space.query_info(page(stack, 0)).unwrap();
    assert_eq!((block.address(), block.pages()), (page(stack, 0), 3));
    // Part of a block: two pages aliased, the first unmapped and mapped again as plain stack
    // memory (`map_pages`, as the kernel maps a stack), which joins the alias's block.
    space
        .map_memory(page(stack, 10), page(heap, 4), 2 * PAGE_SIZE)
        .unwrap();
    space
        .unmap_memory(page(stack, 10), page(heap, 4), PAGE_SIZE)
        .unwrap();
    space
        .map_pages(
            page(stack, 10),
            1,
            MemoryState::Stack,
            MemoryPermission::USER_READ_WRITE,
        )
        .unwrap();
    let block = space.query_info(page(stack, 10)).unwrap();
    assert_eq!((block.address(), block.pages()), (page(stack, 10), 2));
    let bytes = space.save().unwrap();
    let (restored, log) = restore_with(&bytes, &Resolver::new());
    let mut restored = restored.unwrap();
    let body = Body::of(&space);
    assert_eq!(Body::of(&restored), body);
    let calls = log.take();
    assert_eq!(calls, expected_calls(&body, &Resolver::new()));
    // Of the stack pages, only the plain one is mapped as fresh memory.
    let stack_end = page(stack, 64);
    let fresh: Vec<&MapperCall> = calls
        .iter()
        .filter(|call| {
            matches!(call, MapperCall::Map { address, .. }
                if (page(stack, 0)..stack_end).contains(address))
        })
        .collect();
    assert_eq!(
        fresh,
        [&MapperCall::Map {
            address: page(stack, 10),
            size: PAGE_SIZE,
            perms: Perms::ReadWrite
        }]
    );
    // Each run is an alias of its own source on the restored space.
    for (dst, src, size) in [
        (page(stack, 1), page(heap, 1), 2 * PAGE_SIZE),
        (page(stack, 0), page(heap, 8), PAGE_SIZE),
        (page(stack, 11), page(heap, 5), PAGE_SIZE),
    ] {
        assert_eq!(restored.unmap_memory(dst, src, size), Ok(()), "{dst:#x}");
    }
    assert_eq!(restored.aliases.runs(), []);
    assert_eq!(
        restored.query_info(page(stack, 10)).unwrap().state(),
        MemoryState::Stack
    );
}

#[test]
fn an_alias_a_device_holds_restores_held_and_unmaps_once_let_go() {
    let (mut space, _log, heap, stack) = heaped();
    let (dst, src) = (page(stack, 0), page(heap, 0));
    space.map_memory(dst, src, 2 * PAGE_SIZE).unwrap();
    space.lock_device_share(dst, PAGE_SIZE).unwrap();
    let bytes = space.save().unwrap();
    let mut restored = restore_with(&bytes, &Resolver::new()).0.unwrap();
    assert_eq!(Body::of(&restored), Body::of(&space));
    assert_eq!(restored.device_locks(dst, PAGE_SIZE), 1);
    assert_eq!(
        restored.query_info(dst).unwrap().attribute(),
        MemoryAttribute::DEVICE_SHARED
    );
    let invalid = Err(SvcError::Kernel(KernelError::InvalidCurrentMemory));
    assert_eq!(restored.unmap_memory(dst, src, 2 * PAGE_SIZE), invalid);
    // The page the device does not hold, which starts where the held one ends, unmaps.
    assert_eq!(
        restored.unmap_memory(page(dst, 1), page(src, 1), PAGE_SIZE),
        Ok(())
    );
    assert_eq!(restored.unlock_device_share(dst, PAGE_SIZE), Ok(()));
    assert_eq!(restored.unmap_memory(dst, src, PAGE_SIZE), Ok(()));
    assert_eq!(restored.aliases.runs(), []);
}

#[test]
fn the_random_operations_reach_split_and_merged_aliases() {
    // A fixed sequence of the generated kind, so the property test's operations are known to
    // reach two runs in one block, an alias a device holds and a run cut by an unmap.
    let (mut space, _log, _) = loaded();
    for op in [
        Op::Heap(12),
        Op::Alias {
            dst: 0,
            src: 0,
            n: 3,
        },
        Op::Unalias {
            pick: 0,
            skip: 0,
            n: 1,
        },
        Op::Alias {
            dst: 0,
            src: 8,
            n: 1,
        },
        Op::Alias {
            dst: 4,
            src: 4,
            n: 3,
        },
        Op::Unalias {
            pick: 2,
            skip: 1,
            n: 1,
        },
        Op::LockStack(0, 1),
    ] {
        assert!(apply(&mut space, &op), "{op:?}");
    }
    let stack = stack_top(&space, 0);
    let heap = space.regions().heap.start;
    assert_eq!(
        space.aliases.runs(),
        [
            (page(stack, 0), page(heap, 8), PAGE_SIZE),
            (page(stack, 1), page(heap, 1), 2 * PAGE_SIZE),
            (page(stack, 4), page(heap, 4), PAGE_SIZE),
            (page(stack, 6), page(heap, 6), PAGE_SIZE),
        ]
    );
    let bytes = space.save().unwrap();
    let restored = restore_with(&bytes, &Resolver::new()).0.unwrap();
    assert_eq!(Body::of(&restored), Body::of(&space));
}

// ----- The translation's order and undo -------------------------------------------------------------

/// A space with two aliases, and the calls its restore makes.
fn two_aliases() -> (Body, Vec<MapperCall>) {
    let (mut space, _log, heap, stack) = heaped();
    space
        .map_memory(page(stack, 0), page(heap, 0), PAGE_SIZE)
        .unwrap();
    space
        .map_memory(page(stack, 4), page(heap, 4), 2 * PAGE_SIZE)
        .unwrap();
    let body = Body::of(&space);
    let calls = expected_calls(&body, &Resolver::new());
    (body, calls)
}

#[test]
fn a_refused_alias_unmaps_the_aliases_made_then_the_rest() {
    let (body, made) = two_aliases();
    let second = made.len() - 1;
    let (mapper, log) = RecordingMapper::new();
    log.refuse_calls(&[second]);
    let bytes = encode(FORMAT, &body).unwrap();
    let restored = AddressSpace::restore(&bytes, Box::new(mapper), |_| None);
    let alias = body.aliases[1];
    assert_eq!(
        restored.unwrap_err(),
        SnapshotError::Translation {
            address: alias.address,
            size: alias.size,
            error: MapperError("the test told the mapper to refuse".into()),
            stuck: 0,
        }
    );
    // Every mapping, the refused alias aside; the first alias unmapped, then the rest in
    // address order.
    let mut expected: Vec<MapperCall> = made[..second].to_vec();
    expected.push(undo(made[second - 1]));
    expected.extend(made[..second - 1].iter().copied().map(undo));
    assert_eq!(log.take(), expected);
}

#[test]
fn an_alias_whose_unmap_is_refused_in_the_undo_is_stuck() {
    let (body, made) = two_aliases();
    let second = made.len() - 1;
    let (mapper, log) = RecordingMapper::new();
    // The second alias, then the undo of the first.
    log.refuse_calls(&[second, second + 1]);
    let bytes = encode(FORMAT, &body).unwrap();
    let restored = AddressSpace::restore(&bytes, Box::new(mapper), |_| None);
    let Err(SnapshotError::Translation { stuck, .. }) = restored else {
        unreachable!("not a translation refusal");
    };
    assert_eq!(stuck, body.aliases[0].size);
    let mut expected: Vec<MapperCall> = made[..second].to_vec();
    expected.extend(made[..second - 1].iter().copied().map(undo));
    assert_eq!(log.take(), expected);
}

// ----- Check 8 at its edges -------------------------------------------------------------------------

/// A space with two aliases side by side: heap pages 0-1 at stack pages 0-1, and heap pages
/// 2-3 at stack pages 2-3. Its body, with where the heap and the stack pages start.
fn side_by_side() -> (Body, u64, u64) {
    let (mut space, _log, heap, stack) = heaped();
    space
        .map_memory(page(stack, 0), page(heap, 0), 2 * PAGE_SIZE)
        .unwrap();
    space
        .map_memory(page(stack, 2), page(heap, 2), 2 * PAGE_SIZE)
        .unwrap();
    (Body::of(&space), heap, stack)
}

#[test]
fn side_by_side_aliases_restore() {
    let (body, ..) = side_by_side();
    assert!(restore_body(&body).0.is_ok());
}

#[test]
fn an_alias_must_be_whole_pages_inside_the_address_space() {
    let (body, ..) = side_by_side();
    let end = body.regions.address_space.1;
    let cases: Vec<(&str, fn(&mut SavedAlias, u64), AliasFault)> = vec![
        ("no pages", |a, _| a.size = 0, AliasFault::Shape),
        (
            "a size not whole pages",
            |a, _| a.size += 1,
            AliasFault::Shape,
        ),
        (
            "an unaligned destination",
            |a, _| a.address += 1,
            AliasFault::Shape,
        ),
        (
            "an unaligned source",
            |a, _| a.source += 1,
            AliasFault::Shape,
        ),
        // A source that ends exactly at the address space's end passes, and fails on its
        // blocks; a page further does not.
        (
            "a source ending at the end",
            |a, end| a.source = end - a.size,
            AliasFault::Source,
        ),
        (
            "a source a page past the end",
            |a, end| a.source = end - a.size + PAGE_SIZE,
            AliasFault::Shape,
        ),
        (
            "a destination a page past the end",
            |a, end| a.address = end - a.size + PAGE_SIZE,
            AliasFault::Shape,
        ),
        (
            "a source that wraps",
            |a, _| a.source = u64::MAX - PAGE_SIZE + 1,
            AliasFault::Shape,
        ),
        (
            "a size that wraps the destination",
            |a, _| a.size = (u64::MAX - a.address + 1).next_multiple_of(PAGE_SIZE),
            AliasFault::Shape,
        ),
    ];
    for (what, change, fault) in cases {
        let mut case = body.clone();
        change(&mut case.aliases[1], end);
        assert_eq!(error_of(&case), alias(1, fault), "{what}");
    }
}

#[test]
fn an_alias_may_start_where_the_one_before_ends_and_not_a_page_before() {
    let (body, _, stack) = side_by_side();
    // One page below the first's end, and the same start: out of order.
    for address in [page(stack, 1), page(stack, 0)] {
        let mut case = body.clone();
        case.aliases[1].address = address;
        assert_eq!(error_of(&case), alias(1, AliasFault::Order), "{address:#x}");
    }
    // One page above it: in order, over a free page.
    let mut case = body;
    case.aliases[1].address = page(stack, 3);
    assert_eq!(error_of(&case), alias(1, AliasFault::Destination));
}

#[test]
fn a_destination_may_end_at_the_stack_regions_end_and_not_a_page_past_it() {
    let (mut space, _log, heap, _) = heaped();
    let top = space.regions().stack.end;
    assert!(top < space.regions().address_space.end);
    space
        .map_memory(top - PAGE_SIZE, page(heap, 0), PAGE_SIZE)
        .unwrap();
    let body = Body::of(&space);
    assert!(restore_body(&body).0.is_ok());
    let mut case = body;
    case.aliases[0].address = top;
    assert_eq!(error_of(&case), alias(0, AliasFault::Region));
}

#[test]
fn a_source_may_touch_its_destination_on_either_side_and_not_overlap_it() {
    let (body, _, stack) = side_by_side();
    // The second alias is at stack pages 2-3. A source that ends where it starts, or starts
    // where it ends, is not an overlap (and fails on its blocks); one a page closer is.
    for (source, fault) in [
        (page(stack, 0), AliasFault::Source),
        (page(stack, 1), AliasFault::Overlap),
        (page(stack, 4), AliasFault::Source),
        (page(stack, 3), AliasFault::Overlap),
    ] {
        let mut case = body.clone();
        case.aliases[1].source = source;
        assert_eq!(error_of(&case), alias(1, fault), "{source:#x}");
    }
}

/// Applies `change` to the saved block that starts at `address`.
fn change_block(body: &mut Body, address: u64, change: impl Fn(&mut SavedBlock)) {
    let index = at(body, address);
    change(&mut body.blocks[index]);
}

#[test]
fn each_part_of_the_source_check_refuses_on_its_own() {
    let (body, heap, _) = side_by_side();
    let read = MemoryPermission::from_svc(SvcPermission::Read).bits();
    let uncached = MemoryAttribute::UNCACHED.bits();
    let source = page(heap, 2);
    let cases: Vec<(&str, Box<dyn Fn(&mut Body)>)> = vec![
        (
            "a source the guest can read",
            Box::new(move |b| change_block(b, source, |s| s.permission = read)),
        ),
        (
            "a source locked and uncached",
            Box::new(move |b| change_block(b, source, |s| s.attribute |= uncached)),
        ),
        (
            "a source a page past its locked pages",
            Box::new(move |b| b.aliases[1].source = page(heap, 3)),
        ),
        (
            "a source a device holds",
            Box::new(move |b| {
                device_share_alone(b, source);
                insert_lock(b, source, 2 * PAGE_SIZE);
            }),
        ),
    ];
    for (what, change) in cases {
        let mut case = body.clone();
        change(&mut case);
        assert_eq!(error_of(&case), alias(1, AliasFault::Source), "{what}");
    }
}

#[test]
fn a_sources_blocks_must_be_alike() {
    // A source of module data's last page and the page above the code, both locked: refused
    // when the second is `AliasCodeData`, taken when it is `CodeData` like the first.
    let (mut space, _log, _, stack) = heaped();
    let last = DATA + 2 * PAGE_SIZE;
    space.map_memory(page(stack, 0), last, PAGE_SIZE).unwrap();
    space
        .map_pages(
            page(stack, 1),
            1,
            MemoryState::Stack,
            MemoryPermission::USER_READ_WRITE,
        )
        .unwrap();
    let mut body = Body::of(&space);
    body.aliases[0].size = 2 * PAGE_SIZE;
    // The free block above the code, its first page made a locked source.
    let above = last + PAGE_SIZE;
    let index = at(&body, above);
    let free = body.blocks[index];
    let mut carved = body.blocks[at(&body, last)];
    carved.address = above;
    carved.pages = 1;
    carved.disable_merge = DisableMerge::LOCKED.bits();
    let mut rest = free;
    rest.address = above + PAGE_SIZE;
    rest.pages -= 1;
    body.blocks.splice(index..=index, [carved, rest]);
    assert!(restore_body(&body).0.is_ok());
    body.blocks[index].state = MemoryState::AliasCodeData.word();
    assert_eq!(error_of(&body), alias(0, AliasFault::Source));
}

#[test]
fn each_part_of_the_destination_check_refuses_on_its_own() {
    let (body, _, stack) = side_by_side();
    let read = MemoryPermission::from_svc(SvcPermission::Read).bits();
    let uncached = MemoryAttribute::UNCACHED.bits();
    let destination = page(stack, 2);
    let cases: Vec<(&str, Box<dyn Fn(&mut Body)>)> = vec![
        (
            "a read-only destination",
            Box::new(move |b| change_block(b, destination, |d| d.permission = read)),
        ),
        (
            "an uncached destination",
            Box::new(move |b| change_block(b, destination, |d| d.attribute |= uncached)),
        ),
        (
            "a destination a page past the stack memory",
            Box::new(move |b| b.aliases[1].address = page(stack, 3)),
        ),
    ];
    for (what, change) in cases {
        let mut case = body.clone();
        change(&mut case);
        assert_eq!(error_of(&case), alias(1, AliasFault::Destination), "{what}");
    }
}

#[test]
fn a_source_may_touch_an_earlier_aliass_source_on_either_side_and_not_overlap_it() {
    // The first alias's source is heap pages 0-1, the second's 2-3.
    let (body, heap, _) = side_by_side();
    let mut case = body.clone();
    case.aliases[1].source = page(heap, 1);
    assert_eq!(
        error_of(&case),
        alias(1, AliasFault::SourceTaken),
        "over the end of one below"
    );
    let mut case = body.clone();
    case.aliases[1].source = page(heap, 0);
    assert_eq!(
        error_of(&case),
        alias(1, AliasFault::SourceTaken),
        "at the same start"
    );
    // Swapped: the first alias's source is heap pages 2-3, and the second's starts below it.
    let mut swapped = body;
    swapped.aliases[0].source = page(heap, 2);
    swapped.aliases[1].source = page(heap, 0);
    assert!(restore_body(&swapped).0.is_ok());
    swapped.aliases[1].source = page(heap, 1);
    assert_eq!(
        error_of(&swapped),
        alias(1, AliasFault::SourceTaken),
        "over the start of one above"
    );
}

#[test]
fn of_two_faulty_aliases_the_first_is_reported_whatever_its_check() {
    let (body, _, stack) = side_by_side();
    // A late check's fault in the first alias, an early one's in the second.
    let mut case = body.clone();
    case.aliases[0].address = page(stack, 4);
    case.aliases[1].size = 0;
    assert_eq!(error_of(&case), alias(0, AliasFault::Destination));
    // And the reverse.
    let mut case = body;
    case.aliases[0].size = 0;
    case.aliases[1].address = page(stack, 8);
    assert_eq!(error_of(&case), alias(0, AliasFault::Shape));
}

#[test]
fn a_refusal_in_check_8_asks_the_new_mapper_nothing() {
    let (mut body, ..) = side_by_side();
    body.aliases[1].size = 0;
    let (restored, calls) = restore_body(&body);
    assert_eq!(restored.unwrap_err(), alias(1, AliasFault::Shape));
    assert_eq!(calls, []);
}

#[test]
fn saving_a_space_whose_alias_record_disagrees_with_its_blocks_is_refused() {
    // A record of an alias whose source is plain heap: a kernel bug, which save reports.
    let (mut space, _log, heap, stack) = heaped();
    space
        .aliases
        .insert(page(stack, 0), page(heap, 0), PAGE_SIZE);
    assert_eq!(space.save(), Err(alias(0, AliasFault::Source)));
}
