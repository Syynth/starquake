//! Shared and transfer memory as kernel objects: what [`Object::SharedMemory`] and
//! [`Object::TransferMemory`] hold, and [`SharedPages`], the pages of a shared memory as host
//! code reaches them.
//!
//! Ported from Eden (GPL-3.0-or-later), paths relative to the reference folder:
//!
//! - `eden/src/core/hle/kernel/k_shared_memory.h` and `.cpp` @ cf654052: `KSharedMemory`. Its
//!   size is rounded up to whole pages (`m_size = Common::AlignUp(size, PageSize)`), its pages
//!   are zeroed when it is made, and a mapping must name its whole size
//!   (`R_UNLESS(m_size == map_size, ResultInvalidSize)`) and, for a process that is not its
//!   owner, exactly its user permission (`R_UNLESS(map_perm == test_perm,
//!   ResultInvalidNewMemoryPermission)`).
//! - `eden/src/core/hle/kernel/k_transfer_memory.h` and `.cpp` @ cf654052: `KTransferMemory`,
//!   whose `Initialize` locks the owner's pages (`LockForTransferMemory`) and whose `Finalize`
//!   unlocks them (`UnlockForTransferMemory`) unless the memory is mapped by its receiver.
//!   `m_is_initialized` is [`TransferMemory::is_locked`] here.
//!
//! # Where the pages come from
//!
//! Eden takes a shared memory's pages from its page heap and services write them through
//! `KSharedMemory::GetPointer`. Starquake's kernel counts memory but has no pages of its own
//! (kernel spec, "Not planned": page heaps are "replaced by pool accounting"), so the pages come
//! from the [`Mapper`](super::Mapper), which owns the backing:
//! [`Mapper::allocate_shared`](super::Mapper::allocate_shared) returns them as [`SharedPages`]
//! (a range of guest-physical memory and the memory it lies in), and
//! [`Mapper::map_shared`](super::Mapper::map_shared) maps that range for guest code. The
//! object keeps only the range ([`SharedMemory::backing`]); the service that made it keeps the
//! [`SharedPages`] and reads and writes them by offset ([`GuestMemory`]).
//!
//! # Divergences from Eden
//!
//! - **Only a server makes shared memory**, through
//!   [`IpcContext::create_shared_memory`](crate::dispatch::IpcContext::create_shared_memory),
//!   as Eden's HLE services do (`hid_core/resources/shared_memory_holder.cpp`: no owner process,
//!   owner permission `None`). So no process is ever a shared memory's owner, and a mapping is
//!   always checked against the user permission. `CreateSharedMemory` (0x50), which would give
//!   one an owner, is deferred with the server-side calls. Test:
//!   `svc::shared_memory::tests::the_permission_must_be_the_user_permission`.
//! - **The user permission is read or read-write.** Eden accepts `DontCare` too, and then
//!   asserts that the mapping asks for read or write; no HLE service uses it. Test:
//!   `a_shared_memory_permission_is_read_or_read_write`.
//! - **No resource limit is charged.** Eden reserves the size from the system's resource limit
//!   (`ResultLimitReached` when it has no room). Starquake has no system resource limit yet
//!   (the later resource-limits slice), so making shared memory never fails that way.
//! - **Pages are not given back** when the last reference to a shared memory goes: the
//!   [`Mapper`](super::Mapper) has no call for it, and its only implementation
//!   (`starquake-core`) never reuses backing anyway (PROVISIONAL there).

use core::fmt;
use std::sync::Arc;

use starquake_memory::{BackingRange, Fault, GuestMemory, PAGE_SIZE};

use super::state::SvcPermission;
use crate::id::ProcessId;
use crate::result::{KernelError, SvcResult};

/// The size a shared memory of `size` bytes has: rounded up to whole pages
/// (`Common::AlignUp(size, PageSize)`). `None` for zero bytes, and for a size that rounds past
/// the end of the 64-bit space.
///
/// Eden passes a size of zero on to its page allocator, which has no pages to give; what
/// Horizon does with it is not in the reference. Refusing it here is a guess (see
/// [`IpcContext::create_shared_memory`](crate::dispatch::IpcContext::create_shared_memory)).
#[must_use]
pub fn shared_memory_size(size: u64) -> Option<u64> {
    size.checked_next_multiple_of(PAGE_SIZE)
        .filter(|size| *size != 0)
}

/// Whether `permission` may be a shared memory's permission, or a mapping's
/// (`IsValidSharedMemoryPermission` in `eden/src/core/hle/kernel/svc/svc_shared_memory.cpp`):
/// read, or read and write.
#[must_use]
pub const fn is_shared_memory_permission(permission: SvcPermission) -> bool {
    matches!(permission, SvcPermission::Read | SvcPermission::ReadWrite)
}

/// Whether `permission` may be what a transfer memory's owner keeps of its pages
/// (`IsValidTransferMemoryPermission` in
/// `eden/src/core/hle/kernel/svc/svc_transfer_memory.cpp`): none, read, or read and write.
#[must_use]
pub const fn is_transfer_memory_permission(permission: SvcPermission) -> bool {
    matches!(
        permission,
        SvcPermission::None | SvcPermission::Read | SvcPermission::ReadWrite
    )
}

/// The pages of one shared memory as host code reaches them: a range of guest-physical memory,
/// and the memory it lies in. Addresses given to its [`GuestMemory`] methods are offsets from
/// the first page; an access that does not lie wholly inside the pages is a [`Fault`] at the
/// first offset outside them and touches nothing.
#[derive(Clone)]
pub struct SharedPages {
    backing: BackingRange,
    memory: Arc<dyn GuestMemory>,
}

impl fmt::Debug for SharedPages {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SharedPages")
            .field("backing", &self.backing)
            .finish_non_exhaustive()
    }
}

impl SharedPages {
    /// The pages `backing` names, in `memory`, which is addressed by guest-physical address.
    #[must_use]
    pub fn new(backing: BackingRange, memory: Arc<dyn GuestMemory>) -> Self {
        Self { backing, memory }
    }

    /// The guest-physical pages.
    #[must_use]
    pub const fn backing(&self) -> BackingRange {
        self.backing
    }

    /// The size in bytes: whole pages, never zero ([`BackingRange`] guarantees both).
    #[must_use]
    pub const fn size(&self) -> u64 {
        self.backing.size()
    }

    /// The guest-physical address of `len` bytes at `offset`, if they lie wholly inside the
    /// pages; otherwise a fault at the first offset outside them.
    fn physical(&self, offset: u64, len: usize) -> Result<u64, Fault> {
        let size = self.size();
        let outside = Fault {
            address: offset.max(size),
        };
        let end = u64::try_from(len)
            .ok()
            .and_then(|len| offset.checked_add(len))
            .ok_or(outside)?;
        if end > size {
            return Err(outside);
        }
        // `offset < size` here unless `len` is zero, and `base + size` does not overflow.
        self.backing.base().checked_add(offset).ok_or(outside)
    }
}

impl GuestMemory for SharedPages {
    fn read(&self, va: u64, buf: &mut [u8]) -> Result<(), Fault> {
        let at = self.physical(va, buf.len())?;
        self.memory.read(at, buf).map_err(|_| Fault { address: va })
    }

    fn write(&self, va: u64, data: &[u8]) -> Result<(), Fault> {
        let at = self.physical(va, data.len())?;
        self.memory.write(at, data).map_err(|_| Fault { address: va })
    }
}

/// A shared memory (`KSharedMemory`): its pages, and the permission a process that maps it
/// must ask for. See the module docs for who makes one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SharedMemory {
    backing: BackingRange,
    user_permission: SvcPermission,
}

impl SharedMemory {
    /// A shared memory over `backing` that processes map with `user_permission`.
    /// [`KernelError::InvalidNewMemoryPermission`] unless the permission is read or read-write
    /// ([`is_shared_memory_permission`]).
    pub fn new(backing: BackingRange, user_permission: SvcPermission) -> SvcResult<Self> {
        if !is_shared_memory_permission(user_permission) {
            return Err(KernelError::InvalidNewMemoryPermission);
        }
        Ok(Self {
            backing,
            user_permission,
        })
    }

    /// The guest-physical pages.
    #[must_use]
    pub const fn backing(self) -> BackingRange {
        self.backing
    }

    /// The size in bytes (`m_size`): whole pages, never zero.
    #[must_use]
    pub const fn size(self) -> u64 {
        self.backing.size()
    }

    /// The permission a process that is not its owner must map it with (`m_user_permission`).
    #[must_use]
    pub const fn user_permission(self) -> SvcPermission {
        self.user_permission
    }
}

/// A transfer memory (`KTransferMemory`): a range of its owner's memory, lent while the object
/// lives. While it is lent the owner's pages carry [`MemoryAttribute::LOCKED`] and the owner
/// keeps only [`TransferMemory::owner_permission`] of them.
///
/// [`MemoryAttribute::LOCKED`]: super::MemoryAttribute::LOCKED
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TransferMemory {
    owner: ProcessId,
    address: u64,
    size: u64,
    owner_permission: SvcPermission,
    locked: bool,
}

impl TransferMemory {
    /// A transfer memory of `size` bytes of `owner`'s memory at `address`, whose pages are not
    /// locked yet (an object between `KTransferMemory::Create` and `Initialize`).
    #[must_use]
    pub const fn new(
        owner: ProcessId,
        address: u64,
        size: u64,
        owner_permission: SvcPermission,
    ) -> Self {
        Self {
            owner,
            address,
            size,
            owner_permission,
            locked: false,
        }
    }

    /// The same transfer memory, with its owner's pages locked (`m_is_initialized`).
    #[must_use]
    pub const fn with_pages_locked(self) -> Self {
        Self {
            locked: true,
            ..self
        }
    }

    /// The process whose memory it is (`m_owner`).
    #[must_use]
    pub const fn owner(self) -> ProcessId {
        self.owner
    }

    /// Where the memory starts in its owner's address space (`GetSourceAddress`).
    #[must_use]
    pub const fn address(self) -> u64 {
        self.address
    }

    /// Its size in bytes: whole pages.
    #[must_use]
    pub const fn size(self) -> u64 {
        self.size
    }

    /// What its owner may still do with the pages while they are lent (`m_owner_perm`).
    #[must_use]
    pub const fn owner_permission(self) -> SvcPermission {
        self.owner_permission
    }

    /// Whether the owner's pages are locked, so that destroying the object must unlock them
    /// (`m_is_initialized`, which `Finalize` relies on).
    #[must_use]
    pub const fn is_locked(self) -> bool {
        self.locked
    }
}

#[cfg(test)]
mod tests {
    use starquake_memory::FlatMemory;

    use super::*;
    use crate::id::SlabId;

    const BASE: u64 = 0x4000_0000;

    fn pages(count: u64) -> (SharedPages, Arc<FlatMemory>) {
        let size = count * PAGE_SIZE;
        let memory = Arc::new(FlatMemory::new(BASE, usize::try_from(size).unwrap()));
        let backing = BackingRange::new(BASE, size).unwrap();
        (SharedPages::new(backing, memory.clone()), memory)
    }

    #[test]
    fn a_shared_memory_is_whole_pages_and_never_empty() {
        assert_eq!(shared_memory_size(0), None);
        assert_eq!(shared_memory_size(1), Some(PAGE_SIZE));
        assert_eq!(shared_memory_size(PAGE_SIZE), Some(PAGE_SIZE));
        assert_eq!(shared_memory_size(PAGE_SIZE + 1), Some(2 * PAGE_SIZE));
        // The last size that rounds up inside the 64-bit space, and the first that does not.
        let last_page = u64::MAX - (PAGE_SIZE - 1);
        assert_eq!(shared_memory_size(last_page), Some(last_page));
        assert_eq!(shared_memory_size(last_page + 1), None);
    }

    #[test]
    fn a_shared_memory_permission_is_read_or_read_write() {
        let backing = BackingRange::new(BASE, PAGE_SIZE).unwrap();
        for permission in SvcPermission::ALL {
            let made = SharedMemory::new(backing, permission);
            match permission {
                SvcPermission::Read | SvcPermission::ReadWrite => {
                    assert_eq!(made.map(SharedMemory::user_permission), Ok(permission));
                }
                SvcPermission::None | SvcPermission::ReadExecute => {
                    assert_eq!(made, Err(KernelError::InvalidNewMemoryPermission));
                }
            }
        }
    }

    #[test]
    fn a_transfer_memory_permission_is_anything_but_execute() {
        for permission in SvcPermission::ALL {
            assert_eq!(
                is_transfer_memory_permission(permission),
                permission != SvcPermission::ReadExecute
            );
        }
    }

    #[test]
    fn shared_pages_are_reached_by_offset_from_their_first_page() {
        let (shared, memory) = pages(2);
        shared.write(PAGE_SIZE + 4, &[1, 2, 3]).unwrap();
        let mut seen = [0; 3];
        memory.read(BASE + PAGE_SIZE + 4, &mut seen).unwrap();
        assert_eq!(seen, [1, 2, 3]);
        memory.write(BASE, &[9, 8]).unwrap();
        let mut back = [0; 2];
        shared.read(0, &mut back).unwrap();
        assert_eq!(back, [9, 8]);
    }

    #[test]
    fn an_access_past_the_last_page_is_a_fault_where_the_pages_end_and_touches_nothing() {
        let (shared, memory) = pages(1);
        let size = shared.size();
        // Exactly the last byte is inside; one more is not.
        shared.write(size - 1, &[7]).unwrap();
        assert_eq!(
            shared.write(size - 1, &[1, 2]),
            Err(Fault { address: size })
        );
        let mut byte = [0];
        memory.read(BASE + size - 1, &mut byte).unwrap();
        assert_eq!(byte, [7]);
        // Starting past the end, the fault is where the access starts.
        assert_eq!(
            shared.read(size + 8, &mut byte),
            Err(Fault { address: size + 8 })
        );
        // An end that wraps the 64-bit space.
        assert_eq!(
            shared.read(u64::MAX, &mut [0; 2]),
            Err(Fault { address: u64::MAX })
        );
        // No bytes at the very end is fine.
        assert_eq!(shared.read(size, &mut []), Ok(()));
    }

    #[test]
    fn a_transfer_memory_starts_unlocked_and_locking_keeps_the_rest() {
        let owner = ProcessId::from_parts(3, 5);
        let made = TransferMemory::new(owner, 0x1000, 0x2000, SvcPermission::Read);
        assert!(!made.is_locked());
        let locked = made.with_pages_locked();
        assert!(locked.is_locked());
        assert_eq!(
            (
                locked.owner(),
                locked.address(),
                locked.size(),
                locked.owner_permission()
            ),
            (owner, 0x1000, 0x2000, SvcPermission::Read)
        );
    }
}
