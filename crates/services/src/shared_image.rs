//! Building the bytes a service lays a shared memory out with, before it writes them into the
//! memory in one write, and checking a restored id of such a memory against the kernel: shared
//! by [`crate::hid`] and [`crate::time`] and their `shared_memory` layouts.

use starquake_kernel::id::ObjectId;
use starquake_kernel::memory::{SvcPermission, shared_memory_size};

use crate::framework::RestoredKernel;

/// What a family's restore check answers for each way a shared memory id it keeps can fail
/// [`check_restored`], in the order the checks are made.
pub(crate) struct Refusals {
    /// The id names nothing live, or an object that is not a shared memory.
    pub(crate) not_shared_memory: &'static str,
    /// The memory is not the size the service made it.
    pub(crate) size: &'static str,
    /// The memory is not mapped with the permission the service made it with.
    pub(crate) permission: &'static str,
}

/// Checks `object`, a shared memory id a restored family keeps, against `kernel`: it names a
/// live shared memory, of the size the kernel gives one made for `size` bytes (whole pages,
/// [`shared_memory_size`]), that processes map with `permission`. The first that fails is
/// answered from `refusals`, in that order. Which server made it is not checked: the kernel does
/// not note it ([`RestoredKernel::shared_memory`]).
pub(crate) fn check_restored(
    kernel: &dyn RestoredKernel,
    object: ObjectId,
    size: u64,
    permission: SvcPermission,
    refusals: &Refusals,
) -> Result<(), &'static str> {
    let memory = kernel
        .shared_memory(object)
        .ok_or(refusals.not_shared_memory)?;
    if shared_memory_size(size) != Some(memory.size()) {
        return Err(refusals.size);
    }
    if memory.user_permission() != permission {
        return Err(refusals.permission);
    }
    Ok(())
}

/// A zeroed image of `size` bytes. Every caller's size is a constant of a few pages, which fits
/// any host's `usize`.
pub(crate) fn blank(size: u64) -> Vec<u8> {
    vec![0; usize::try_from(size).unwrap_or(usize::MAX)]
}

/// Copies `bytes` into `image` at `offset`. Every caller writes constant fields of a constant
/// layout that its golden-bytes test checks, so a write that did not lie inside the image would
/// be a mistake in that layout; it is skipped and logged, naming `memory`, rather than allowed
/// to panic.
pub(crate) fn put(image: &mut [u8], offset: u64, bytes: &[u8], memory: &str) {
    let target = usize::try_from(offset).ok().and_then(|start| {
        let end = start.checked_add(bytes.len())?;
        image.get_mut(start..end)
    });
    if let Some(target) = target {
        target.copy_from_slice(bytes);
    } else {
        tracing::error!(offset, memory, "a shared memory field lies past its end");
    }
}

#[cfg(test)]
mod tests {
    use starquake_kernel::memory::SharedMemory;
    use starquake_kernel::slab::SlabId;
    use starquake_memory::{BackingRange, PAGE_SIZE};
    use starquake_trace::ServiceId;

    use super::*;

    /// A kernel that holds at most one shared memory, which every id names.
    struct Holds(Option<SharedMemory>);

    impl RestoredKernel for Holds {
        fn is_service_readable_event(&self, _: ServiceId, _: ObjectId) -> bool {
            false
        }

        fn shared_memory(&self, _: ObjectId) -> Option<SharedMemory> {
            self.0
        }
    }

    /// A kernel holding a shared memory of `pages` pages mapped with `permission`.
    fn holding(pages: u64, permission: SvcPermission) -> Holds {
        let backing = BackingRange::new(0x10_0000, pages * PAGE_SIZE).unwrap();
        Holds(Some(SharedMemory::new(backing, permission).unwrap()))
    }

    const REFUSALS: Refusals = Refusals {
        not_shared_memory: "not one",
        size: "size",
        permission: "permission",
    };

    /// Each check of [`check_restored`] at its edge on both sides, in its order: no memory,
    /// then the size (rounded up to whole pages, as the kernel makes it; a page under, at and
    /// over), then the permission (compared with the one asked, either one), with both faults
    /// at once reported as the size.
    #[test]
    fn a_restored_memory_is_checked_for_kind_then_size_then_permission() {
        let id = ObjectId::from_parts(3, 1);
        let read = SvcPermission::Read;
        let write = SvcPermission::ReadWrite;
        let three_pages = 3 * PAGE_SIZE;
        let cases = [
            (Holds(None), three_pages, read, Err("not one")),
            (holding(2, read), three_pages, read, Err("size")),
            (holding(3, read), three_pages, read, Ok(())),
            (holding(4, read), three_pages, read, Err("size")),
            // A size that is not whole pages is the size the kernel rounds it to.
            (holding(3, read), three_pages - 1, read, Ok(())),
            (holding(3, read), 2 * PAGE_SIZE + 1, read, Ok(())),
            (holding(3, read), 2 * PAGE_SIZE, read, Err("size")),
            // No shared memory is made for zero bytes, so none has that size.
            (holding(1, read), 0, read, Err("size")),
            (holding(3, write), three_pages, read, Err("permission")),
            (holding(3, write), three_pages, write, Ok(())),
            (holding(3, read), three_pages, write, Err("permission")),
            (holding(4, write), three_pages, read, Err("size")),
        ];
        for (kernel, size, permission, expected) in cases {
            assert_eq!(
                check_restored(&kernel, id, size, permission, &REFUSALS),
                expected,
                "{size:#x} {permission:?}"
            );
        }
    }

    #[test]
    fn a_field_is_written_only_where_it_fits_whole() {
        let mut image = blank(4);
        put(&mut image, 2, &[7, 8], "test");
        assert_eq!(image, [0, 0, 7, 8]);
        // One byte past the end, and an offset past the end: nothing is written.
        put(&mut image, 3, &[9, 9], "test");
        put(&mut image, 5, &[9], "test");
        put(&mut image, u64::MAX, &[9], "test");
        assert_eq!(image, [0, 0, 7, 8]);
    }
}
