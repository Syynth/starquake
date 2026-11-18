//! An address space made with a system resource ([`AddressSpace::with_system_resource`]): the
//! charge in Eden's order (`KProcess::Initialize`: the code's reservation, then the resource,
//! then the code's mapping), its edges, and the memory sizes `GetInfo` reads, which count it as
//! Eden's `KProcess` getters do.

use starquake_memory::Perms;

use super::*;
use crate::memory::fixture::{CODE, CODE_SIZE, STACK_SIZE, created, memory};
use crate::memory::pool::Pool;
use crate::memory::system_resource::SECURE_ALIGNMENT;
use crate::memory::testing::{MapperCall, RecordingMapper};

/// The 4 GiB console's application pool, which every address space here counts against.
const POOL: u64 = MemoryArrangement::FourGb.application_pool_size();

/// A resource of `size` bytes in `pool`; `size` must fit a `u32`.
fn resource(size: u64, pool: Pool) -> SystemResource {
    SystemResource::new(u32::try_from(size).unwrap(), pool)
}

/// `code_size` bytes of code at the fixture's address, on a 4 GiB console.
fn code_of(code_size: u64) -> ProcessMemory {
    ProcessMemory {
        code_size,
        ..memory(MemoryArrangement::FourGb)
    }
}

/// An address space for `code_size` bytes of code and `resource`, and what its mapper was asked.
fn made(
    code_size: u64,
    resource: SystemResource,
) -> (Result<AddressSpace, CreateError>, Vec<MapperCall>) {
    let (mapper, log) = RecordingMapper::new();
    let space = AddressSpace::with_system_resource(&code_of(code_size), resource, Box::new(mapper));
    (space, log.take())
}

#[test]
fn a_space_made_without_a_resource_has_none_and_charges_nothing() {
    // `AddressSpace::new` is a process with the shared default resource
    // (`system_resource_num_pages == 0`).
    let (space, _log) = created();
    assert_eq!(space.system_resource(), SystemResource::NONE);
    assert_eq!(space.secure_memory_size(), 0);
    assert_eq!(space.limit().reserved(), CODE_SIZE);
    assert_eq!(
        space.used_user_physical_memory_size(),
        space.used_non_system_user_physical_memory_size()
    );
}

#[test]
fn a_resource_that_fits_exactly_beside_the_code_is_charged_and_one_page_more_is_limit_reached() {
    // The code is one secure alignment, so the pool's rest is a resource of whole alignments.
    let code = SECURE_ALIGNMENT;
    let exact = resource(POOL - code, Pool::Application);
    let (space, calls) = made(code, exact);
    let space = space.unwrap();
    assert_eq!(space.system_resource(), exact);
    assert_eq!((space.limit().reserved(), space.limit().free()), (POOL, 0));
    assert_eq!(space.secure_memory_size(), POOL - code);
    // The code is mapped after the resource is set aside.
    assert_eq!(
        calls,
        [MapperCall::Map {
            address: CODE,
            size: code,
            perms: Perms::None
        }]
    );
    // A page more: the reservation fails before the size's alignment is looked at, and the
    // code is never mapped.
    for size in [POOL - code + PAGE_SIZE, POOL - code + SECURE_ALIGNMENT] {
        let (space, calls) = made(code, resource(size, Pool::Application));
        assert_eq!(
            space.unwrap_err(),
            CreateError::SystemResource(KernelError::LimitReached),
            "{size:#x}"
        );
        assert_eq!(calls, []);
    }
    // And the same resource with a page more of code.
    let (space, calls) = made(code + PAGE_SIZE, exact);
    assert_eq!(
        space.unwrap_err(),
        CreateError::SystemResource(KernelError::LimitReached)
    );
    assert_eq!(calls, []);
}

#[test]
fn the_code_is_reserved_before_the_resource() {
    // Code that alone passes the pool, and a resource that is not aligned either: the code's
    // reservation fails first.
    let (space, calls) = made(POOL + PAGE_SIZE, resource(PAGE_SIZE, Pool::Application));
    assert_eq!(
        space.unwrap_err(),
        CreateError::Code(SvcError::Kernel(KernelError::LimitReached))
    );
    assert_eq!(calls, []);
    // The code that fits, with that resource: the resource's own check refuses it.
    let (space, calls) = made(CODE_SIZE, resource(PAGE_SIZE, Pool::Application));
    assert_eq!(
        space.unwrap_err(),
        CreateError::SystemResource(KernelError::InvalidSize)
    );
    assert_eq!(calls, []);
}

#[test]
fn an_applet_resource_is_kept_but_charges_nothing() {
    let applet = resource(2 * SECURE_ALIGNMENT, Pool::Applet);
    let (space, _calls) = made(CODE_SIZE, applet);
    let space = space.unwrap();
    assert_eq!(space.system_resource(), applet);
    assert_eq!(space.secure_memory_size(), 0);
    assert_eq!(space.limit().reserved(), CODE_SIZE);
}

#[test]
fn the_memory_sizes_count_the_resource_as_edens_getters_do() {
    // `GetUsedUserPhysicalMemorySize` adds the secure size to the non-system used size, and
    // `GetTotalUserPhysicalMemorySize` compares that with the free memory left after the
    // charge. On the 4 GiB console the pool is below the largest process memory, so the total
    // is the used size and the free memory.
    let secure = 5 * SECURE_ALIGNMENT;
    let (space, _calls) = made(CODE_SIZE, resource(secure, Pool::Application));
    let mut space = space.unwrap();
    space.create_main_thread_stack(STACK_SIZE).unwrap();
    let non_system = CODE_SIZE + STACK_SIZE;
    assert_eq!(
        space.used_non_system_user_physical_memory_size(),
        non_system
    );
    assert_eq!(space.used_user_physical_memory_size(), non_system + secure);
    assert_eq!(space.limit().reserved(), non_system + secure);
    assert_eq!(space.limit().free(), POOL - non_system - secure);
    assert!(POOL < space.max_process_memory());
    assert_eq!(space.total_user_physical_memory_size(), POOL);
}
