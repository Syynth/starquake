//! `/dev/nvmap`: memory handles, the ioctls that create, allocate, duplicate, query and free them,
//! and the table that holds them (`docs/specs/graphics-server.md`, "Channels and submission:
//! nvsrv", the device table's `/dev/nvmap` row, "Eden, unchanged"; port slice 9).
//!
//! Ported from Eden (GPL-3.0-or-later) @ cf654052:
//! `eden/src/core/hle/service/nvdrv/devices/nvmap.cpp` and `nvmap.h` (the ioctls and their
//! argument structs) and `eden/src/core/hle/service/nvdrv/core/nvmap.cpp` and `nvmap.h` (the
//! handle table: `CreateHandle`, `GetHandle`, `Handle::Alloc`, `Handle::Duplicate`,
//! `FreeHandle`). Neither Switch 2 reference changes them: NX2 @ 50fa86bb has the same files
//! byte for byte, and PR @ e5fb3b97 adds only a diagnostic counter to `core/nvmap.cpp`. libnx's
//! client (`libnx/nx/source/nvidia/ioctl/nvmap.c` @ b156b477) lays the argument structs out the
//! same way.
//!
//! # The ioctls
//!
//! Every ioctl is group 1 ([`command::GROUP`]) on `Ioctl` (1); Eden's `nvmap::Ioctl2` and
//! `Ioctl3` know none. Each argument is a fixed struct ([`Fixed`]), so the caller gets back the
//! whole struct, with the fields the ioctl did not change as it sent them, whatever the result.
//!
//! | Cmd | Ioctl | Argument (offset: field) | Checks, in order, and their results |
//! |---|---|---|---|
//! | 1 | `IocCreate` | 0: size (in), 4: handle (out) | the size rounded up to a page is 0 as a `u32` → `BadValue`; then (starquake's, guesses) the table full or the ids used up → `InsufficientMemory` |
//! | 3 | `IocFromId` | 0: id (in), 4: handle (out) | id 0, an id with no handle, or an unallocated handle → `BadValue`; then (a guess) the duplicate count at its limit → `BadValue` |
//! | 4 | `IocAlloc` | 0: handle, 4: heap mask, 8: flags, 12: alignment, 16: kind (`u8`), 24: address (`u64`) | handle 0 → `BadValue`; an alignment that is not a power of two (0 passes) → `BadValue`; the alignment is then raised to a page; no such handle → `BadValue`; already allocated → `InsufficientMemory`; then (starquake's) the range past the end of the address space → `BadValue` |
//! | 5 | `IocFree` | 0: handle (in), 8: address (`u64`, out), 16: size (out), 20: flags (out) | handle 0, or no such handle → `Success` and nothing written |
//! | 9 | `IocParam` | 0: handle, 4: parameter (in), 8: result (out) | handle 0, no such handle, or an unknown parameter → `BadValue` |
//! | 14 | `IocGetId` | 0: id (out), 4: handle (in) | handle 0 → `BadValue`; no such handle → `AccessDenied` |
//!
//! A handle's id is its handle: the first is 4 and each next one 4 more
//! (`NvMap::HandleIdIncrement`), never reused. `IocCreate` makes a handle with one reference,
//! `IocFromId` adds one, and `IocFree` drops one and removes the handle when none is left.
//! `IocParam` answers, by parameter ([`param`]): the size as created, the alignment (0 until
//! allocated), [`BASE_ANSWER`], [`HEAP_ANSWER`] once allocated and 0 before, the kind (0 until
//! allocated), and 0 for "shared memory mapped" (only `MapSharedMem`, command 5, which is not
//! served, sets it in Eden).
//!
//! # Where this differs from Eden
//!
//! Each is pinned by the test named, in `nvmap/tests.rs`:
//!
//! | Case | Eden | Here |
//! |---|---|---|
//! | `IocAlloc` and `IocFree`'s device-share lock | `IocAlloc` locks the range for the device (`LockForMapDeviceAddressSpace`, its failure only asserted) and `IocFree` unlocks it once the last reference goes | neither touches the kernel: no device reads an nvmap handle yet (nvhost-as-gpu maps them, slice 10), and the lock's interface is changing to return a release token (owner's ruling of 2026-11-17). Taking it is the follow-up of whichever lands second (`alloc_and_free_take_no_device_lock`) |
//! | `IocAlloc` whose address plus page-rounded size is past 2^64 | stored; the lock fails and Eden asserts, then succeeds | `BadValue` (a guess), nothing stored (`an_allocation_past_the_address_space_is_refused`) |
//! | The table full, or handle ids used up | neither is bounded (the next id wraps after 2^30 handles) | at most [`MAX_HANDLES`] handles, ids up to `0xFFFF_FFFC`, then `InsufficientMemory` (a guess); the table is checked before the ids (`create_refuses_a_full_table_then_spent_ids`) |
//! | A handle duplicated [`MAX_DUPES`] times | the `s32` count overflows | `BadValue` (a guess), the count unchanged (`from_id_stops_at_the_duplicate_limit`) |
//! | Flags kept for an allocation | the whole word, bit 2 cleared when an address is given | only bit 0 ([`MAP_UNCACHED`]), the one bit anything reads back (`IocFree`) |
//!
//! Eden's handles also carry SMMU pins, emulator-internal duplicates and a session; nothing in this
//! slice pins a handle or duplicates one internally, so they are left out.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use starquake_memory::PAGE_SIZE;

use super::NvResult;
use super::ioctl::{Fixed, Ioctl};

#[cfg(test)]
mod tests;

/// nvmap's ioctl group and commands (`nvmap::Ioctl1`: `case 0x1:` for the group, then
/// `case 0x1` .. `case 0xe` for the commands).
pub mod command {
    /// nvmap's group.
    pub const GROUP: u32 = 0x1;
    /// `IocCreate`.
    pub const CREATE: u32 = 0x1;
    /// `IocFromId`.
    pub const FROM_ID: u32 = 0x3;
    /// `IocAlloc`.
    pub const ALLOC: u32 = 0x4;
    /// `IocFree`.
    pub const FREE: u32 = 0x5;
    /// `IocParam`.
    pub const PARAM: u32 = 0x9;
    /// `IocGetId`.
    pub const GET_ID: u32 = 0xE;
}

/// Where each field of each argument struct is, and each struct's size (`nvmap.h`: the
/// `Ioc*Params` structs and their `static_assert`ed sizes; little-endian `u32`s unless noted).
pub mod layout {
    /// `IocCreateParams`: 8 bytes.
    pub const CREATE_BYTES: usize = 8;
    /// `IocCreateParams::size`, in.
    pub const CREATE_SIZE: usize = 0;
    /// `IocCreateParams::handle`, out.
    pub const CREATE_HANDLE: usize = 4;

    /// `IocFromIdParams`: 8 bytes.
    pub const FROM_ID_BYTES: usize = 8;
    /// `IocFromIdParams::id`, in.
    pub const FROM_ID_ID: usize = 0;
    /// `IocFromIdParams::handle`, out.
    pub const FROM_ID_HANDLE: usize = 4;

    /// `IocAllocParams`: 32 bytes.
    pub const ALLOC_BYTES: usize = 32;
    /// `IocAllocParams::handle`.
    pub const ALLOC_HANDLE: usize = 0;
    /// `IocAllocParams::heap_mask`, which nothing reads.
    pub const ALLOC_HEAP_MASK: usize = 4;
    /// `IocAllocParams::flags`.
    pub const ALLOC_FLAGS: usize = 8;
    /// `IocAllocParams::align`, raised to a page when below one.
    pub const ALLOC_ALIGN: usize = 12;
    /// `IocAllocParams::kind`, one byte, then seven of padding.
    pub const ALLOC_KIND: usize = 16;
    /// `IocAllocParams::address`, a `u64`.
    pub const ALLOC_ADDRESS: usize = 24;

    /// `IocFreeParams`: 24 bytes.
    pub const FREE_BYTES: usize = 24;
    /// `IocFreeParams::handle`, in, then four bytes of padding.
    pub const FREE_HANDLE: usize = 0;
    /// `IocFreeParams::address`, a `u64`, out.
    pub const FREE_ADDRESS: usize = 8;
    /// `IocFreeParams::size`, out.
    pub const FREE_SIZE: usize = 16;
    /// `IocFreeParams::flags`, out.
    pub const FREE_FLAGS: usize = 20;

    /// `IocParamParams`: 12 bytes.
    pub const PARAM_BYTES: usize = 12;
    /// `IocParamParams::handle`, in.
    pub const PARAM_HANDLE: usize = 0;
    /// `IocParamParams::param`, in.
    pub const PARAM_PARAM: usize = 4;
    /// `IocParamParams::result`, out.
    pub const PARAM_RESULT: usize = 8;

    /// `IocGetIdParams`: 8 bytes.
    pub const GET_ID_BYTES: usize = 8;
    /// `IocGetIdParams::id`, out.
    pub const GET_ID_ID: usize = 0;
    /// `IocGetIdParams::handle`, in.
    pub const GET_ID_HANDLE: usize = 4;
}

/// `IocParam`'s parameters (`nvmap.h`: `enum class HandleParameterType`).
pub mod param {
    /// The size the handle was created with.
    pub const SIZE: u32 = 1;
    /// The alignment it was allocated with.
    pub const ALIGNMENT: u32 = 2;
    /// Answered with [`super::BASE_ANSWER`].
    pub const BASE: u32 = 3;
    /// Answered with [`super::HEAP_ANSWER`] once allocated.
    pub const HEAP: u32 = 4;
    /// The kind it was allocated with.
    pub const KIND: u32 = 5;
    /// Whether it was mapped through `MapSharedMem`.
    pub const IS_SHARED_MEM_MAPPED: u32 = 6;
}

use layout::{
    ALLOC_ADDRESS, ALLOC_ALIGN, ALLOC_BYTES, ALLOC_FLAGS, ALLOC_HANDLE, ALLOC_KIND, CREATE_BYTES,
    CREATE_HANDLE, CREATE_SIZE, FREE_ADDRESS, FREE_BYTES, FREE_FLAGS, FREE_HANDLE, FREE_SIZE,
    FROM_ID_BYTES, FROM_ID_HANDLE, FROM_ID_ID, GET_ID_BYTES, GET_ID_HANDLE, GET_ID_ID, PARAM_BYTES,
    PARAM_HANDLE, PARAM_PARAM, PARAM_RESULT,
};

/// What `IocParam` answers for [`param::BASE`]: `static_cast<u32_le>(-22)`, "posix EINVAL".
pub const BASE_ANSWER: u32 = (-22_i32).cast_unsigned();

/// What `IocParam` answers for [`param::HEAP`] once the handle is allocated (`0x40000000`).
pub const HEAP_ANSWER: u32 = 0x4000_0000;

/// The flags bit that asks for an uncached mapping, which `IocFree` reports back
/// (`core/nvmap.h`: `Handle::Flags`, `BitField<0, 1, u32> map_uncached`).
pub const MAP_UNCACHED: u32 = 1 << 0;

/// How far apart handle ids are, and the first id (`core/nvmap.h`: `HandleIdIncrement{4}`,
/// `next_handle_id{HandleIdIncrement}`).
pub const HANDLE_ID_INCREMENT: u32 = 4;

/// One past the last handle id: ids are `u32`s.
const HANDLE_ID_END: u64 = 1 << u32::BITS;

/// The largest page-rounded size a handle can have: the size is a `u32` in Eden, rounded up as a
/// `u32` (`Common::AlignUp(params.size, YUZU_PAGESIZE)`), so anything rounding past it wraps to 0
/// and is refused.
pub const MAX_HANDLE_SIZE: u64 = HANDLE_ID_END - PAGE_SIZE;

/// The most handles the table holds at once. A guess: Eden has no limit. It bounds what a
/// title can make the table hold.
pub const MAX_HANDLES: usize = 0x1_0000;

/// The most references one handle can hold: Eden counts them in an `s32` (`Handle::dupes`).
pub const MAX_DUPES: u32 = i32::MAX.unsigned_abs();

/// How a handle was allocated (`IocAlloc`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Allocation {
    uncached: bool,
    align: u32,
    kind: u8,
    address: u64,
}

impl Allocation {
    /// Whether the caller asked for an uncached mapping ([`MAP_UNCACHED`]).
    #[must_use]
    pub const fn uncached(&self) -> bool {
        self.uncached
    }

    /// The alignment: a power of two, at least a page.
    #[must_use]
    pub const fn align(&self) -> u32 {
        self.align
    }

    /// The kind (the GPU's memory kind, for compression).
    #[must_use]
    pub const fn kind(&self) -> u8 {
        self.kind
    }

    /// The caller's address of the memory; 0 if the caller gave none.
    #[must_use]
    pub const fn address(&self) -> u64 {
        self.address
    }
}

/// One memory handle (`core/nvmap.h`: `NvMap::Handle`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct NvMapHandle {
    size: u32,
    dupes: u32,
    allocation: Option<Allocation>,
}

impl NvMapHandle {
    /// The size it was created with (`orig_size`).
    #[must_use]
    pub const fn created_size(&self) -> u32 {
        self.size
    }

    /// The size rounded up to a page (`size`).
    #[must_use]
    pub fn size(&self) -> u64 {
        page_size(self.size)
    }

    /// How many references it holds (`dupes`).
    #[must_use]
    pub const fn dupes(&self) -> u32 {
        self.dupes
    }

    /// How it was allocated, once it is.
    #[must_use]
    pub const fn allocation(&self) -> Option<&Allocation> {
        self.allocation.as_ref()
    }
}

/// `size` rounded up to a page.
fn page_size(size: u32) -> u64 {
    u64::from(size).next_multiple_of(PAGE_SIZE)
}

/// The handle table (`core/nvmap.h`: `NvMap`), shared by every session of the nvsrv server and
/// saved with it. Deserialising goes through `SavedNvMap`, which refuses a table its ioctls
/// cannot reach ([`NvMapStateError`]).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "SavedNvMap")]
pub struct NvMap {
    handles: BTreeMap<u32, NvMapHandle>,
    next_id: u64,
}

/// The serialised form of [`NvMap`], field for field, and what deserialising validates.
#[derive(Deserialize)]
struct SavedNvMap {
    handles: BTreeMap<u32, NvMapHandle>,
    next_id: u64,
}

/// Why a saved [`NvMap`] could not be read back: it holds something no sequence of ioctls leads
/// to. The checks run in this order, the table's own first and then each handle's, by id, and
/// the first that fails is reported.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum NvMapStateError {
    /// The next id is not one the ids can reach: below the first, not a multiple of the
    /// increment, or past the last `u32`.
    #[error("the next nvmap handle id {next:#x} is not one handle ids can reach")]
    NextId {
        /// The next id in the data.
        next: u64,
    },
    /// More handles than the table holds.
    #[error("the nvmap table holds {count} handles, more than it can")]
    TooMany {
        /// How many there are.
        count: usize,
    },
    /// An id that is not a handle id: zero, or not a multiple of the increment.
    #[error("nvmap handle id {id:#x} is not a handle id")]
    NotAnId {
        /// The id.
        id: u32,
    },
    /// An id at or past the next id, so never handed out.
    #[error("nvmap handle id {id:#x} was never handed out")]
    NotIssued {
        /// The id.
        id: u32,
    },
    /// A size `IocCreate` refuses: 0 once rounded to a page.
    #[error("nvmap handle {id:#x} has size {size:#x}, which no handle can have")]
    Size {
        /// The handle.
        id: u32,
        /// Its size.
        size: u32,
    },
    /// No references, or more than a handle can hold.
    #[error("nvmap handle {id:#x} holds {dupes} references")]
    Dupes {
        /// The handle.
        id: u32,
        /// Its reference count.
        dupes: u32,
    },
    /// An unallocated handle with more than one reference: only an allocated one can be
    /// duplicated.
    #[error("unallocated nvmap handle {id:#x} holds {dupes} references")]
    UnallocatedDuplicated {
        /// The handle.
        id: u32,
        /// Its reference count.
        dupes: u32,
    },
    /// An alignment that is not a power of two of at least a page.
    #[error("nvmap handle {id:#x} is aligned to {align:#x}, which no allocation can be")]
    Align {
        /// The handle.
        id: u32,
        /// Its alignment.
        align: u32,
    },
    /// Memory that runs past the end of the address space.
    #[error("nvmap handle {id:#x} at {address:#x} runs past the end of the address space")]
    PastTheEnd {
        /// The handle.
        id: u32,
        /// Its address.
        address: u64,
    },
}

impl TryFrom<SavedNvMap> for NvMap {
    type Error = NvMapStateError;

    fn try_from(saved: SavedNvMap) -> Result<Self, Self::Error> {
        let SavedNvMap { handles, next_id } = saved;
        let increment = u64::from(HANDLE_ID_INCREMENT);
        if next_id < increment || !next_id.is_multiple_of(increment) || next_id > HANDLE_ID_END {
            return Err(NvMapStateError::NextId { next: next_id });
        }
        if handles.len() > MAX_HANDLES {
            return Err(NvMapStateError::TooMany {
                count: handles.len(),
            });
        }
        for (&id, handle) in &handles {
            check_handle(id, handle, next_id)?;
        }
        Ok(Self { handles, next_id })
    }
}

/// The checks of one saved handle, in [`NvMapStateError`]'s order.
fn check_handle(id: u32, handle: &NvMapHandle, next_id: u64) -> Result<(), NvMapStateError> {
    if id == 0 || !id.is_multiple_of(HANDLE_ID_INCREMENT) {
        return Err(NvMapStateError::NotAnId { id });
    }
    if u64::from(id) >= next_id {
        return Err(NvMapStateError::NotIssued { id });
    }
    if checked_page_size(handle.size).is_none() {
        return Err(NvMapStateError::Size {
            id,
            size: handle.size,
        });
    }
    let dupes = handle.dupes;
    if dupes == 0 || dupes > MAX_DUPES {
        return Err(NvMapStateError::Dupes { id, dupes });
    }
    let Some(allocation) = handle.allocation else {
        if dupes != 1 {
            return Err(NvMapStateError::UnallocatedDuplicated { id, dupes });
        }
        return Ok(());
    };
    let align = allocation.align;
    if !align.is_power_of_two() || u64::from(align) < PAGE_SIZE {
        return Err(NvMapStateError::Align { id, align });
    }
    if allocation.address.checked_add(handle.size()).is_none() {
        return Err(NvMapStateError::PastTheEnd {
            id,
            address: allocation.address,
        });
    }
    Ok(())
}

/// `size` rounded up to a page, or `None` where Eden's `u32` rounding gives 0 (`IocCreate`:
/// `Common::AlignUp(params.size, YUZU_PAGESIZE)`, then `CreateHandle`: `if (!size)
/// [[unlikely]] return NvResult::BadValue`): a size of 0, or one that rounds past the last
/// `u32`.
fn checked_page_size(size: u32) -> Option<u64> {
    Some(page_size(size)).filter(|&rounded| rounded != 0 && rounded <= MAX_HANDLE_SIZE)
}

impl Default for NvMap {
    fn default() -> Self {
        Self::new()
    }
}

impl NvMap {
    /// A table with no handles; the first id is [`HANDLE_ID_INCREMENT`].
    #[must_use]
    pub fn new() -> Self {
        Self {
            handles: BTreeMap::new(),
            next_id: u64::from(HANDLE_ID_INCREMENT),
        }
    }

    /// The handle with id `id`, if there is one.
    #[must_use]
    pub fn handle(&self, id: u32) -> Option<&NvMapHandle> {
        self.handles.get(&id)
    }

    /// How many handles there are.
    #[must_use]
    pub fn len(&self) -> usize {
        self.handles.len()
    }

    /// Whether there are none.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.handles.is_empty()
    }

    /// Runs an `Ioctl` (1) on `/dev/nvmap`: its result and the argument struct to give back, or
    /// `None` if nvmap has no such ioctl (`nvmap::Ioctl1`'s `UNIMPLEMENTED_MSG` path).
    pub(super) fn ioctl(&mut self, ioctl: Ioctl, input: &[u8]) -> Option<(NvResult, Vec<u8>)> {
        if ioctl.group() != command::GROUP {
            return None;
        }
        Some(match ioctl.command() {
            command::CREATE => run::<CREATE_BYTES>(input, |args| self.create(args)),
            command::FROM_ID => run::<FROM_ID_BYTES>(input, |args| self.duplicate(args)),
            command::ALLOC => run::<ALLOC_BYTES>(input, |args| self.alloc(args)),
            command::FREE => run::<FREE_BYTES>(input, |args| self.free(args)),
            command::PARAM => run::<PARAM_BYTES>(input, |args| self.param(args)),
            command::GET_ID => run::<GET_ID_BYTES>(input, |args| self.get_id(args)),
            _ => return None,
        })
    }

    /// `IocCreate`.
    fn create(&mut self, args: &mut Fixed<CREATE_BYTES>) -> NvResult {
        let size = args.u32_at::<CREATE_SIZE>();
        if checked_page_size(size).is_none() {
            tracing::warn!(size, "nvmap: create with a size that rounds to nothing");
            return NvResult::BadValue;
        }
        if self.handles.len() >= MAX_HANDLES {
            tracing::warn!("nvmap: the handle table is full");
            return NvResult::InsufficientMemory;
        }
        let Ok(id) = u32::try_from(self.next_id) else {
            tracing::warn!("nvmap: handle ids are used up");
            return NvResult::InsufficientMemory;
        };
        self.next_id += u64::from(HANDLE_ID_INCREMENT);
        self.handles.insert(
            id,
            NvMapHandle {
                size,
                dupes: 1,
                allocation: None,
            },
        );
        args.set_u32::<CREATE_HANDLE>(id);
        NvResult::Success
    }

    /// `IocFromId`: one more reference to an allocated handle.
    fn duplicate(&mut self, args: &mut Fixed<FROM_ID_BYTES>) -> NvResult {
        let id = args.u32_at::<FROM_ID_ID>();
        if id == 0 {
            return NvResult::BadValue;
        }
        let Some(handle) = self.handles.get_mut(&id) else {
            return NvResult::BadValue;
        };
        // `Handle::Duplicate`: "Unallocated handles cannot be duplicated".
        if handle.allocation.is_none() {
            return NvResult::BadValue;
        }
        if handle.dupes >= MAX_DUPES {
            tracing::warn!(id, "nvmap: a handle at its duplicate limit");
            return NvResult::BadValue;
        }
        handle.dupes += 1;
        args.set_u32::<FROM_ID_HANDLE>(id);
        NvResult::Success
    }

    /// `IocAlloc`.
    fn alloc(&mut self, args: &mut Fixed<ALLOC_BYTES>) -> NvResult {
        let id = args.u32_at::<ALLOC_HANDLE>();
        if id == 0 {
            return NvResult::BadValue;
        }
        let align = args.u32_at::<ALLOC_ALIGN>();
        // `if ((params.align - 1) & params.align)`, in `u32`: 0 passes.
        if align.wrapping_sub(1) & align != 0 {
            return NvResult::BadValue;
        }
        // "Force page size alignment at a minimum", written back to the caller's struct. The
        // larger of a `u32` and a page always fits a `u32`.
        let align = u32::try_from(u64::from(align).max(PAGE_SIZE)).unwrap_or(align);
        args.set_u32::<ALLOC_ALIGN>(align);
        let Some(handle) = self.handles.get_mut(&id) else {
            return NvResult::BadValue;
        };
        if handle.allocation.is_some() {
            return NvResult::InsufficientMemory;
        }
        let address = args.u64_at::<ALLOC_ADDRESS>();
        if address.checked_add(handle.size()).is_none() {
            tracing::warn!(id, address, "nvmap: an allocation past the address space");
            return NvResult::BadValue;
        }
        if address == 0 {
            // `Handle::Alloc`: "Mapping nvmap handles without a CPU side address is
            // unimplemented!", and the handle is allocated anyway.
            tracing::warn!(id, "nvmap: an allocation with no address");
        }
        handle.allocation = Some(Allocation {
            uncached: args.u32_at::<ALLOC_FLAGS>() & MAP_UNCACHED != 0,
            align,
            kind: args.u8_at::<ALLOC_KIND>(),
            address,
        });
        NvResult::Success
    }

    /// `IocFree`: one reference fewer; the handle goes with the last.
    fn free(&mut self, args: &mut Fixed<FREE_BYTES>) -> NvResult {
        let id = args.u32_at::<FREE_HANDLE>();
        // "Handle null freed?", and a handle `FreeHandle` cannot find: success, nothing written.
        let Some(handle) = self.handles.get_mut(&id) else {
            return NvResult::Success;
        };
        // A handle in the table holds at least one reference.
        handle.dupes = handle.dupes.saturating_sub(1);
        let handle = *handle;
        if handle.dupes == 0 {
            self.handles.remove(&id);
        }
        let allocation = handle.allocation;
        let uncached = allocation.is_some_and(|allocation| allocation.uncached);
        args.set_u64::<FREE_ADDRESS>(allocation.map_or(0, |allocation| allocation.address));
        // At most `MAX_HANDLE_SIZE`, which fits a `u32`.
        args.set_u32::<FREE_SIZE>(u32::try_from(handle.size()).unwrap_or(u32::MAX));
        args.set_u32::<FREE_FLAGS>(if uncached { MAP_UNCACHED } else { 0 });
        NvResult::Success
    }

    /// `IocParam`.
    fn param(&self, args: &mut Fixed<PARAM_BYTES>) -> NvResult {
        let id = args.u32_at::<PARAM_HANDLE>();
        if id == 0 {
            return NvResult::BadValue;
        }
        let Some(handle) = self.handles.get(&id) else {
            return NvResult::BadValue;
        };
        let allocation = handle.allocation;
        let answer = match args.u32_at::<PARAM_PARAM>() {
            param::SIZE => handle.size,
            param::ALIGNMENT => allocation.map_or(0, |allocation| allocation.align),
            param::BASE => BASE_ANSWER,
            param::HEAP => allocation.map_or(0, |_| HEAP_ANSWER),
            param::KIND => allocation.map_or(0, |allocation| u32::from(allocation.kind)),
            param::IS_SHARED_MEM_MAPPED => 0,
            _ => return NvResult::BadValue,
        };
        args.set_u32::<PARAM_RESULT>(answer);
        NvResult::Success
    }

    /// `IocGetId`.
    fn get_id(&self, args: &mut Fixed<GET_ID_BYTES>) -> NvResult {
        let id = args.u32_at::<GET_ID_HANDLE>();
        if id == 0 {
            return NvResult::BadValue;
        }
        // "This will always return EPERM irrespective of if the handle exists or not".
        if !self.handles.contains_key(&id) {
            return NvResult::AccessDenied;
        }
        args.set_u32::<GET_ID_ID>(id);
        NvResult::Success
    }
}

/// Runs `ioctl` on an `N`-byte argument read from `input` (`WrapFixed`): its result and the
/// argument as it leaves it.
fn run<const N: usize>(
    input: &[u8],
    ioctl: impl FnOnce(&mut Fixed<N>) -> NvResult,
) -> (NvResult, Vec<u8>) {
    let mut args = Fixed::from_input(input);
    let result = ioctl(&mut args);
    (result, args.into_output())
}
