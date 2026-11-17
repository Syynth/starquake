//! The `MapMemory` and `UnmapMemory` arms: the registers they read (destination in x0, source
//! in x1, size in x2, `svc_wrappers64.txt`) and write (the result in x0 alone), and what they
//! record.

use super::*;
use crate::memory::MemoryAttribute;
use crate::svc::memory::HEAP_SIZE_ALIGNMENT;

const PAGE: u64 = starquake_memory::PAGE_SIZE;

/// Dispatches `id` with `dst`, `src` and `size` in x0 to x2, checks that only x0 changed and
/// what was recorded, and returns x0.
fn call(dispatcher: &Dispatcher, id: SvcId, dst: u64, src: u64, size: u64) -> u64 {
    let mut regs = regs();
    regs.x[0] = dst;
    regs.x[1] = src;
    regs.x[2] = size;
    let before = regs.clone();
    let (record, events) = dispatch_on(dispatcher, id.number(), &mut regs, 0);
    assert_eq!(record.outcome, Outcome::Completed, "{}", id.name());
    assert_untouched_from(&regs, &before, 1);
    let mut args = [0; SVC_ARG_COUNT];
    args[..3].copy_from_slice(&[dst, src, size]);
    assert_eq!(
        kinds(&events),
        [EventKind::Svc {
            id: u32::from(id.number()),
            args,
            result: starquake_trace::ResultCode(u32::try_from(regs.x[0]).unwrap()),
        }]
    );
    regs.x[0]
}

#[test]
fn map_memory_and_unmap_memory_take_dst_src_and_size_in_that_order() {
    let dispatcher = Dispatcher::new(tables());
    let (heap, stack) = dispatcher.with::<(&mut AddressSpace,), _>(|(space,)| {
        let heap = space.set_heap_size(HEAP_SIZE_ALIGNMENT).unwrap();
        (heap, space.regions().stack.end - 16 * PAGE)
    });
    // Swapped source and destination: the heap is no place for stack memory.
    assert_eq!(
        call(&dispatcher, SvcId::MapMemory, heap, stack, PAGE),
        word(KernelError::InvalidMemoryRegion)
    );
    assert_eq!(
        call(&dispatcher, SvcId::MapMemory, stack, heap, 2 * PAGE),
        0
    );
    dispatcher.with::<(&AddressSpace,), _>(|(space,)| {
        let alias = space.query_info(stack).unwrap();
        assert_eq!(
            (alias.state(), alias.size()),
            (MemoryState::Stack, 2 * PAGE)
        );
        assert_eq!(
            space.query_info(heap).unwrap().attribute(),
            MemoryAttribute::LOCKED
        );
    });
    // The size is read from x2: one page of the two is a partial unmap, which Eden allows.
    assert_eq!(call(&dispatcher, SvcId::UnmapMemory, stack, heap, PAGE), 0);
    assert_eq!(
        call(
            &dispatcher,
            SvcId::UnmapMemory,
            stack + PAGE,
            heap + PAGE,
            PAGE
        ),
        0
    );
    dispatcher.with::<(&AddressSpace,), _>(|(space,)| {
        assert_eq!(space.query_info(stack).unwrap().state(), MemoryState::Free);
        assert_eq!(
            space.query_info(heap).unwrap().attribute(),
            MemoryAttribute::NONE
        );
    });
    // Nothing is aliased any more: the source is plain heap again.
    assert_eq!(
        call(&dispatcher, SvcId::UnmapMemory, stack, heap, PAGE),
        word(KernelError::InvalidCurrentMemory)
    );
}
