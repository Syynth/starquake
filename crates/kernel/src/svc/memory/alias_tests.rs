//! `MapMemory` and `UnmapMemory`'s argument checks (`MapUnmapMemorySanityChecks`), each at its
//! edge on both sides, and their order.
//!
//! "Eden" below is `eden/src/core/hle/kernel/svc/svc_memory.cpp` @ cf654052. What the page
//! table checks after these is in `memory/space/alias_tests.rs`.

use super::*;
use crate::memory::fixture::loaded;
use crate::memory::testing::MapperLog;
use crate::result::SvcError;

fn kernel<T>(error: KernelError) -> CallResult<T> {
    Err(SvcError::Kernel(error))
}

type Call = fn(&mut AddressSpace, u64, u64, u64) -> CallResult<()>;

/// Both calls, by name.
const CALLS: [(&str, Call); 2] = [("MapMemory", map_memory), ("UnmapMemory", unmap_memory)];

/// The fixture's address space with a heap, its log emptied, the heap's start, and a free
/// page-aligned address near the top of the stack region.
fn rig() -> (AddressSpace, MapperLog, u64, u64) {
    let (mut space, log, _) = loaded();
    let heap = space.set_heap_size(HEAP_SIZE_ALIGNMENT).unwrap();
    drop(log.take());
    let stack = space.regions().stack.end - 64 * PAGE_SIZE;
    (space, log, heap, stack)
}

/// What a passing check leads to: for `MapMemory` an alias, for `UnmapMemory` (nothing is
/// aliased) the page table's refusal of a plain heap source.
fn passed(name: &str) -> CallResult<()> {
    if name == "MapMemory" {
        Ok(())
    } else {
        kernel(KernelError::InvalidCurrentMemory)
    }
}

#[test]
fn the_addresses_must_be_page_aligned() {
    for (name, call) in CALLS {
        for (dst_off, src_off, expected) in [
            (1, 0, kernel(KernelError::InvalidAddress)),
            (PAGE_SIZE - 1, 0, kernel(KernelError::InvalidAddress)),
            (0, 1, kernel(KernelError::InvalidAddress)),
            (0, PAGE_SIZE - 1, kernel(KernelError::InvalidAddress)),
            (0, 0, passed(name)),
        ] {
            let (mut space, _, heap, stack) = rig();
            assert_eq!(
                call(&mut space, stack + dst_off, heap + src_off, PAGE_SIZE),
                expected,
                "{name} {dst_off:#x} {src_off:#x}"
            );
        }
    }
}

#[test]
fn a_source_that_is_not_page_aligned_is_invalid_address_not_eden_s_invalid_size() {
    // Eden: `R_THROW(ResultInvalidSize)` for an unaligned source; Horizon:
    // `svc::ResultInvalidAddress` (module docs, "Divergences").
    for (name, call) in CALLS {
        let (mut space, _, heap, stack) = rig();
        assert_eq!(
            call(&mut space, stack, heap + 1, PAGE_SIZE),
            kernel(KernelError::InvalidAddress),
            "{name}"
        );
    }
}

#[test]
fn the_size_must_be_whole_pages_and_not_zero() {
    for (name, call) in CALLS {
        for (size, expected) in [
            (0, kernel(KernelError::InvalidSize)),
            (1, kernel(KernelError::InvalidSize)),
            (PAGE_SIZE - 1, kernel(KernelError::InvalidSize)),
            (PAGE_SIZE + 1, kernel(KernelError::InvalidSize)),
            (PAGE_SIZE, passed(name)),
        ] {
            let (mut space, _, heap, stack) = rig();
            assert_eq!(
                call(&mut space, stack, heap, size),
                expected,
                "{name} {size:#x}"
            );
        }
    }
}

#[test]
fn neither_range_may_wrap() {
    // `IsValidAddressRange`: `address + size > address`. The last page of the 64-bit space
    // wraps to exactly zero; the page below it does not, and goes on to the next check.
    let last_page = u64::MAX - (PAGE_SIZE - 1);
    let below = last_page - PAGE_SIZE;
    for (name, call) in CALLS {
        let (mut space, log, heap, stack) = rig();
        // The destination: wrapping, then not (the source is fine; the destination is
        // outside the stack region).
        assert_eq!(
            call(&mut space, last_page, heap, PAGE_SIZE),
            kernel(KernelError::InvalidCurrentMemory),
            "{name}"
        );
        assert_eq!(
            call(&mut space, below, heap, PAGE_SIZE),
            kernel(KernelError::InvalidMemoryRegion),
            "{name}"
        );
        // The source: wrapping, then not (it is outside the address space).
        assert_eq!(
            call(&mut space, stack, last_page, PAGE_SIZE),
            kernel(KernelError::InvalidCurrentMemory),
            "{name}"
        );
        assert_eq!(
            call(&mut space, stack, below, PAGE_SIZE),
            kernel(KernelError::InvalidCurrentMemory),
            "{name}"
        );
        // A size that wraps both from a low address: refused as a wrap, before the regions.
        assert_eq!(
            call(&mut space, stack, heap, last_page),
            kernel(KernelError::InvalidCurrentMemory),
            "{name}"
        );
        assert_eq!(log.take(), [], "{name}");
    }
}

#[test]
fn the_source_must_lie_inside_the_address_space() {
    for (name, call) in CALLS {
        let (mut space, _, _, stack) = rig();
        let end = space.regions().address_space.end;
        let start = space.regions().address_space.start;
        // The destination is a free page of the heap region, which the stack region's check
        // (the next one) refuses with `InvalidMemoryRegion`: a source that passes this check
        // gets that result, and one that fails it `InvalidCurrentMemory`. Ending exactly at
        // the address space's end passes; a page further does not.
        let free_heap = space.regions().heap.start + HEAP_SIZE_ALIGNMENT;
        assert_eq!(
            call(&mut space, free_heap, end - PAGE_SIZE, PAGE_SIZE),
            kernel(KernelError::InvalidMemoryRegion),
            "{name}"
        );
        assert_eq!(
            call(&mut space, free_heap, end - PAGE_SIZE, 2 * PAGE_SIZE),
            kernel(KernelError::InvalidCurrentMemory),
            "{name}"
        );
        // Starting exactly at the address space's start passes; a page below it does not.
        assert_eq!(
            call(&mut space, free_heap, start, PAGE_SIZE),
            kernel(KernelError::InvalidMemoryRegion),
            "{name}"
        );
        if let Some(below) = start.checked_sub(PAGE_SIZE) {
            assert_eq!(
                call(&mut space, free_heap, below, PAGE_SIZE),
                kernel(KernelError::InvalidCurrentMemory),
                "{name}"
            );
        }
        // With a destination the stack region holds, a source a page past the end is still
        // this check's result.
        assert_eq!(
            call(&mut space, stack, end - PAGE_SIZE, 2 * PAGE_SIZE),
            kernel(KernelError::InvalidCurrentMemory),
            "{name}"
        );
        // Outside the address space and with a destination outside the stack region: the
        // source's check comes first. (A source inside, with the same destination, is the
        // stack region's result.)
        assert_eq!(
            call(&mut space, end, end, PAGE_SIZE),
            kernel(KernelError::InvalidCurrentMemory),
            "{name}"
        );
        let heap = space.regions().heap.start;
        assert_eq!(
            call(&mut space, end, heap, PAGE_SIZE),
            kernel(KernelError::InvalidMemoryRegion),
            "{name}"
        );
    }
}

#[test]
fn the_destination_must_be_one_the_stack_region_can_hold() {
    // `CanContain(dst_address, size, KMemoryState::Stack)`, Horizon's check (not in Eden).
    for (name, call) in CALLS {
        let (mut space, _, heap, _) = rig();
        let stack = space.regions().stack;
        // Ending exactly at the region's end passes; a page past it, or starting a page below
        // its start, does not.
        assert_eq!(
            call(&mut space, stack.end - PAGE_SIZE, heap, PAGE_SIZE),
            passed(name),
            "{name}"
        );
        let (mut space, _, heap, _) = rig();
        assert_eq!(
            call(&mut space, stack.end - PAGE_SIZE, heap, 2 * PAGE_SIZE),
            kernel(KernelError::InvalidMemoryRegion),
            "{name}"
        );
        assert_eq!(
            call(&mut space, stack.start - PAGE_SIZE, heap, PAGE_SIZE),
            kernel(KernelError::InvalidMemoryRegion),
            "{name}"
        );
        // A free page of the heap region is no place for stack memory either.
        let free_heap = space.regions().heap.start + HEAP_SIZE_ALIGNMENT;
        assert_eq!(
            call(&mut space, free_heap, heap, PAGE_SIZE),
            kernel(KernelError::InvalidMemoryRegion),
            "{name}"
        );
    }
}

#[test]
fn the_checks_run_in_order_when_two_faults_are_present_at_once() {
    let last_page = u64::MAX - (PAGE_SIZE - 1);
    for (name, call) in CALLS {
        let (mut space, log, heap, stack) = rig();
        let end = space.regions().address_space.end;
        for (dst, src, size, expected, what) in [
            // The destination's alignment before the source's: both `InvalidAddress`, so the
            // pair below pins the source's alignment before the size.
            (
                stack,
                heap + 1,
                0,
                KernelError::InvalidAddress,
                "source alignment before size",
            ),
            (
                stack + 1,
                heap,
                PAGE_SIZE + 1,
                KernelError::InvalidAddress,
                "destination alignment before size",
            ),
            (
                stack,
                last_page,
                PAGE_SIZE + 1,
                KernelError::InvalidSize,
                "size before the wrap",
            ),
            (
                last_page,
                end,
                PAGE_SIZE,
                KernelError::InvalidCurrentMemory,
                "wrap before the regions",
            ),
            (
                end,
                end,
                PAGE_SIZE,
                KernelError::InvalidCurrentMemory,
                "source region before destination region",
            ),
        ] {
            assert_eq!(
                call(&mut space, dst, src, size),
                kernel(expected),
                "{name}: {what}"
            );
        }
        assert_eq!(log.take(), [], "{name}");
    }
}

#[test]
fn a_mapped_alias_is_unmapped_through_the_system_calls() {
    let (mut space, _, heap, stack) = rig();
    assert_eq!(map_memory(&mut space, stack, heap, PAGE_SIZE), Ok(()));
    assert_eq!(space.query_info(stack).unwrap().state(), MemoryState::Stack);
    assert_eq!(unmap_memory(&mut space, stack, heap, PAGE_SIZE), Ok(()));
    assert_eq!(space.query_info(stack).unwrap().state(), MemoryState::Free);
}
