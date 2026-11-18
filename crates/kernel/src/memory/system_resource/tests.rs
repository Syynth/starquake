//! The system resource: its size in whole pages, what each pool charges and aligns to, and
//! `KSecureSystemResource::Initialize`'s checks in Eden's order, each at its exact edge.

use super::*;

/// The smallest application resource: one [`SECURE_ALIGNMENT`].
const SMALLEST: u64 = SECURE_ALIGNMENT;

/// A resource of `size` bytes in `pool`; `size` must fit a `u32`.
fn resource(size: u64, pool: Pool) -> SystemResource {
    SystemResource::new(u32::try_from(size).unwrap(), pool)
}

#[test]
fn the_constants_are_the_references() {
    // `SecureAlignment = 128_KiB`, and `RefCount` is a `u16`.
    assert_eq!(SECURE_ALIGNMENT, 0x2_0000);
    assert_eq!(REFERENCE_COUNT_SIZE, 2);
}

#[test]
fn a_recorded_size_counts_in_whole_pages() {
    let max = u64::from(u32::MAX);
    for (recorded, size) in [
        (0, 0),
        (1, 0),
        (PAGE_SIZE - 1, 0),
        (PAGE_SIZE, PAGE_SIZE),
        (PAGE_SIZE + 1, PAGE_SIZE),
        (2 * PAGE_SIZE - 1, PAGE_SIZE),
        (max, max - (PAGE_SIZE - 1)),
    ] {
        for pool in Pool::ALL {
            let made = resource(recorded, pool);
            assert_eq!(made.size(), size, "{recorded:#x}");
            assert_eq!(made.pool(), pool);
            assert_eq!(made.is_secure(), size != 0, "{recorded:#x}");
        }
    }
    assert_eq!(SystemResource::NONE, resource(0, Pool::Application));
    assert!(!SystemResource::NONE.is_secure());
}

#[test]
fn only_the_applet_pool_charges_nothing_and_only_the_system_pool_aligns_to_a_page() {
    let size = 3 * SMALLEST;
    for (pool, charged, alignment) in [
        (Pool::Application, size, SECURE_ALIGNMENT),
        (Pool::Applet, 0, SECURE_ALIGNMENT),
        (Pool::System, size, PAGE_SIZE),
        (Pool::SystemNonSecure, size, SECURE_ALIGNMENT),
    ] {
        let made = resource(size, pool);
        assert_eq!(made.required_secure_memory_size(), charged, "{pool:?}");
        assert_eq!(made.alignment(), alignment, "{pool:?}");
    }
}

#[test]
fn the_reference_counts_take_two_bytes_a_page_in_whole_pages() {
    // One page of counts holds 2048 pages' counts.
    let per_page = PAGE_SIZE / REFERENCE_COUNT_SIZE;
    for (pages, expected) in [
        (1, PAGE_SIZE),
        (per_page, PAGE_SIZE),
        (per_page + 1, 2 * PAGE_SIZE),
    ] {
        let made = resource(pages * PAGE_SIZE, Pool::System);
        assert_eq!(made.reference_count_size(), expected, "{pages} pages");
    }
}

#[test]
fn the_size_must_be_aligned_for_its_pool() {
    // `R_UNLESS(Common::IsAligned(size, alignment), ResultInvalidSize)`, at the edge on both
    // sides of one alignment and of two.
    for pool in [Pool::Application, Pool::Applet, Pool::SystemNonSecure] {
        let check = |size| resource(size, pool).check_allocation();
        assert_eq!(check(SMALLEST), Ok(()), "{pool:?}");
        assert_eq!(check(2 * SMALLEST), Ok(()), "{pool:?}");
        for size in [
            PAGE_SIZE,
            SMALLEST - PAGE_SIZE,
            SMALLEST + PAGE_SIZE,
            2 * SMALLEST - PAGE_SIZE,
        ] {
            assert_eq!(
                check(size),
                Err(KernelError::InvalidSize),
                "{pool:?} {size:#x}"
            );
        }
    }
    // In the system pool any whole pages are aligned.
    let system = |size| resource(size, Pool::System).check_allocation();
    assert_eq!(system(SMALLEST + PAGE_SIZE), Ok(()));
    assert_eq!(system(3 * PAGE_SIZE), Ok(()));
}

#[test]
fn the_size_must_be_more_than_its_reference_counts() {
    // `R_UNLESS(m_resource_size > rc_size, ResultOutOfMemory)`: one page holds its own counts
    // with no room left; two pages leave a page. Only the system pool's alignment lets a size
    // that small through.
    let system = |pages: u64| resource(pages * PAGE_SIZE, Pool::System);
    assert_eq!(system(1).size(), system(1).reference_count_size());
    assert_eq!(system(1).check_allocation(), Err(KernelError::OutOfMemory));
    assert_eq!(system(2).check_allocation(), Ok(()));
}

#[test]
fn no_secure_resource_passes_every_check_and_charges_nothing() {
    for pool in Pool::ALL {
        let none = resource(PAGE_SIZE - 1, pool);
        assert_eq!(none.check_allocation(), Ok(()));
        let mut full = MemoryLimit::new(PAGE_SIZE);
        full.reserve(PAGE_SIZE).unwrap();
        assert_eq!(none.initialize(&mut full), Ok(()));
        assert_eq!(full.reserved(), PAGE_SIZE);
    }
}

#[test]
fn a_resource_that_fits_exactly_is_charged_and_one_page_more_is_limit_reached() {
    let size = 4 * SMALLEST;
    // Exactly the free memory: charged in full, nothing left.
    let mut exact = MemoryLimit::new(size + PAGE_SIZE);
    exact.reserve(PAGE_SIZE).unwrap();
    assert_eq!(
        resource(size, Pool::Application).initialize(&mut exact),
        Ok(())
    );
    assert_eq!((exact.reserved(), exact.free()), (size + PAGE_SIZE, 0));
    // One page less free memory: the reservation fails first, and nothing is charged.
    let mut short = MemoryLimit::new(size);
    short.reserve(PAGE_SIZE).unwrap();
    let before = short;
    assert_eq!(
        resource(size, Pool::Application).initialize(&mut short),
        Err(KernelError::LimitReached)
    );
    assert_eq!(short, before);
    // And the resource a page larger than the free memory, which is not aligned either: the
    // reservation is checked first, so it is `LimitReached`, not `InvalidSize`.
    let mut limit = MemoryLimit::new(size);
    assert_eq!(
        resource(size + PAGE_SIZE, Pool::Application).initialize(&mut limit),
        Err(KernelError::LimitReached)
    );
    assert_eq!(limit.reserved(), 0);
    // The same unaligned size with room for it: `InvalidSize`, and the reservation it made is
    // given back.
    let mut roomy = MemoryLimit::new(2 * size);
    assert_eq!(
        resource(size + PAGE_SIZE, Pool::Application).initialize(&mut roomy),
        Err(KernelError::InvalidSize)
    );
    assert_eq!(roomy.reserved(), 0);
    // After a refusal, a resource that fits is charged.
    assert_eq!(
        resource(size, Pool::Application).initialize(&mut roomy),
        Ok(())
    );
    assert_eq!(roomy.reserved(), size);
}

#[test]
fn an_out_of_memory_resource_gives_its_reservation_back() {
    let mut limit = MemoryLimit::new(SMALLEST);
    assert_eq!(
        resource(PAGE_SIZE, Pool::System).initialize(&mut limit),
        Err(KernelError::OutOfMemory)
    );
    assert_eq!(limit.reserved(), 0);
}

#[test]
fn an_applet_resource_charges_nothing_and_is_checked_like_an_applications() {
    // Divergence: Eden asserts before the alignment check (`UNIMPLEMENTED_IF`); here the check
    // is made as for an application.
    let mut full = MemoryLimit::new(SMALLEST);
    full.reserve(SMALLEST).unwrap();
    assert_eq!(
        resource(SMALLEST, Pool::Applet).initialize(&mut full),
        Ok(())
    );
    assert_eq!(full.reserved(), SMALLEST);
    assert_eq!(
        resource(SMALLEST + PAGE_SIZE, Pool::Applet).initialize(&mut full),
        Err(KernelError::InvalidSize)
    );
    assert_eq!(full.reserved(), SMALLEST);
}
