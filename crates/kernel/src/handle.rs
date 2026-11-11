//! Per-process handle tables: Horizon's handle encoding, generations and the pseudo-handles.
//!
//! Ported from Eden `eden/src/core/hle/kernel/k_handle_table.h` and `k_handle_table.cpp`, and
//! `eden/src/core/hle/kernel/svc_common.h` (pseudo-handles), at the commit pinned in
//! `docs/references.md` (GPL-3.0-or-later). Result values come from `svc_results.h` and the
//! `Result` bit layout in `eden/src/core/hle/result.h`.
//!
//! A [`Handle`] is a 32-bit value: bits 0-14 the table index, bits 15-29 a "linear id" (what the
//! task calls the generation), bits 30-31 reserved and zero. The linear id is **not** a
//! per-slot generation: the table keeps one counter that runs `1..=0x7FFF` and wraps, and
//! `Add` stamps the next value on whichever slot it takes. A stale handle is therefore rejected
//! until the counter has wrapped all the way round, after exactly [`MAX_LINEAR_ID`] further adds
//! (pinned by a test); that is Horizon's behaviour, kept so handle values match what a guest
//! would see. (The generic [`Slab`](crate::slab::Slab) does not have this limit; it is for
//! kernel-internal ids, which guests never see.)
//!
//! The free list is a stack that starts so that the first `Add` takes the **highest** index
//! (`Initialize` links entry `i` to `i - 1` and leaves the head on the last one), and a removed
//! slot is the next to be reused. [`HandleTable`] keeps that order.
//!
//! What this port does not carry over, so a later slice can add it where a system call needs it:
//! `Reserve`/`Unreserve`/`Register` (two-step insertion used when a call must fail atomically),
//! `GetObjectByIndex`, `GetMultipleObjects`, `GetObjectForIpc`, and `Finalize` (close every
//! entry when a process dies). Objects are held as [`ObjectId`]s and reference counting goes
//! through [`ObjectRefs`], see [`refs`].

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::id::ObjectId;
use crate::slab::{FreeStackError, check_free_stack};

mod refs;

pub use refs::{Closed, ObjectRefs, RefCounts, RefError};

/// Bits of a handle holding the table index (`HandlePack::index`, `k_handle_table.h`).
pub const INDEX_BITS: u32 = 15;
/// Bits of a handle holding the linear id (`HandlePack::linear_id`).
pub const LINEAR_ID_BITS: u32 = 15;
/// Bits of a handle reserved and required to be zero (`HandlePack::reserved`).
pub const RESERVED_BITS: u32 = 2;

const INDEX_MASK: u32 = (1 << INDEX_BITS) - 1;
const LINEAR_ID_MASK: u32 = (1 << LINEAR_ID_BITS) - 1;

/// The largest number of handles a table can hold (`KHandleTable::MaxTableSize`).
pub const MAX_TABLE_SIZE: u16 = 1024;
/// The first linear id issued (`MinLinearId`); zero is never issued.
pub const MIN_LINEAR_ID: u16 = 1;
/// The last linear id issued before the counter wraps back to [`MIN_LINEAR_ID`] (`MaxLinearId`).
pub const MAX_LINEAR_ID: u16 = (1_u16 << LINEAR_ID_BITS) - 1;

/// Where a result's module sits in the raw value: `Result::Module` is `BitField<0, 9>`, and
/// `Result::Description` is `BitField<9, 13>` (`result.h`).
const RESULT_DESCRIPTION_SHIFT: u32 = 9;
/// `ErrorModule::Kernel` (`result.h`).
const KERNEL_MODULE: u32 = 1;

/// Builds the raw value of a kernel-module result from its description number.
const fn kernel_result(description: u32) -> u32 {
    KERNEL_MODULE | (description << RESULT_DESCRIPTION_SHIFT)
}

/// `ResultOutOfMemory{ErrorModule::Kernel, 104}` (`svc_results.h`): a table size above
/// [`MAX_TABLE_SIZE`].
pub const RESULT_OUT_OF_MEMORY: u32 = kernel_result(104);
/// `ResultOutOfHandles{ErrorModule::Kernel, 105}` (`svc_results.h`): the table is full.
pub const RESULT_OUT_OF_HANDLES: u32 = kernel_result(105);
/// `ResultInvalidHandle{ErrorModule::Kernel, 114}` (`svc_results.h`).
pub const RESULT_INVALID_HANDLE: u32 = kernel_result(114);

/// A guest-visible handle: a table index and a linear id packed in 32 bits. See the module docs.
///
/// Any `u32` is a `Handle`; whether it names anything is decided by the table.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Handle(u32);

impl Handle {
    /// `Svc::InvalidHandle` (`svc_common.h`): never names an object.
    pub const INVALID: Self = Self(0);
    /// `Svc::PseudoHandle::CurrentThread` (`svc_common.h`): the calling thread.
    pub const CURRENT_THREAD: Self = Self(0xFFFF_8000);
    /// `Svc::PseudoHandle::CurrentProcess` (`svc_common.h`): the calling process.
    pub const CURRENT_PROCESS: Self = Self(0xFFFF_8001);

    /// Wraps a raw handle value, as read from a guest register or message.
    #[must_use]
    pub const fn from_raw(raw: u32) -> Self {
        Self(raw)
    }

    /// The raw handle value.
    #[must_use]
    pub const fn raw(self) -> u32 {
        self.0
    }

    /// Packs an index and a linear id (`EncodeHandle`), with the reserved bits zero. `None` if
    /// either does not fit its field.
    #[must_use]
    pub fn from_parts(index: u16, linear_id: u16) -> Option<Self> {
        let (index, linear_id) = (u32::from(index), u32::from(linear_id));
        if index > INDEX_MASK || linear_id > LINEAR_ID_MASK {
            return None;
        }
        Some(Self(index | (linear_id << INDEX_BITS)))
    }

    /// The table index field.
    #[must_use]
    pub const fn index(self) -> u32 {
        self.0 & INDEX_MASK
    }

    /// The linear id field.
    #[must_use]
    pub const fn linear_id(self) -> u32 {
        (self.0 >> INDEX_BITS) & LINEAR_ID_MASK
    }

    /// The reserved field; zero in every handle a table issues.
    #[must_use]
    pub const fn reserved(self) -> u32 {
        self.0 >> (INDEX_BITS + LINEAR_ID_BITS)
    }

    /// Whether this is one of the two pseudo-handles (`Svc::IsPseudoHandle`).
    #[must_use]
    pub fn is_pseudo(self) -> bool {
        self == Self::CURRENT_THREAD || self == Self::CURRENT_PROCESS
    }
}

/// Why a handle table operation failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum HandleError {
    /// A table was requested larger than [`MAX_TABLE_SIZE`]. Horizon's `ResultOutOfMemory`.
    #[error("handle table size {requested} exceeds the maximum of {MAX_TABLE_SIZE}")]
    TableTooLarge {
        /// The requested size.
        requested: i32,
    },
    /// The table has no free entry. Horizon's `ResultOutOfHandles`.
    #[error("the handle table is full")]
    OutOfHandles,
    /// The handle names nothing in this table. Horizon's `ResultInvalidHandle`.
    #[error("handle {raw:#010x} does not name an object")]
    InvalidHandle {
        /// The raw handle value.
        raw: u32,
    },
    /// Opening or closing the object's reference failed; the table was not changed.
    #[error("reference count: {0}")]
    Refs(#[from] RefError),
    /// The table's bookkeeping is inconsistent. Unreachable through the public API; it exists so
    /// that a table built from damaged data fails instead of panicking.
    #[error("handle table bookkeeping is inconsistent")]
    Corrupt,
}

impl HandleError {
    /// The raw Horizon result this error corresponds to, if it has one: `OutOfMemory`,
    /// `OutOfHandles` or `InvalidHandle`. Reference-count and corruption errors are starquake's
    /// own and have no guest-visible result. (A later slice joins this to the kernel's result
    /// type; until then the raw values are the contract.)
    #[must_use]
    pub const fn result(&self) -> Option<u32> {
        match self {
            Self::TableTooLarge { .. } => Some(RESULT_OUT_OF_MEMORY),
            Self::OutOfHandles => Some(RESULT_OUT_OF_HANDLES),
            Self::InvalidHandle { .. } => Some(RESULT_INVALID_HANDLE),
            Self::Refs(_) | Self::Corrupt => None,
        }
    }
}

/// What a handle names once pseudo-handles are taken into account.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Resolved {
    /// [`Handle::CURRENT_THREAD`]: the dispatcher substitutes the calling thread.
    CurrentThread,
    /// [`Handle::CURRENT_PROCESS`]: the dispatcher substitutes the calling process.
    CurrentProcess,
    /// An entry in the table.
    Object(ObjectId),
}

/// What [`HandleTable::remove`] released.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Removed {
    /// The object the handle named.
    pub object: ObjectId,
    /// What closing the handle's reference left: [`Closed::Last`] means closing this handle
    /// destroyed the object's last reference.
    pub closed: Closed,
}

/// One occupied table entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
struct Entry {
    linear_id: u16,
    object: ObjectId,
}

/// A process's handle table. See the module docs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "HandleTableData")]
pub struct HandleTable {
    /// One slot per possible handle; the length is the table size.
    entries: Vec<Option<Entry>>,
    /// Indices of free slots, next to be used last.
    free: Vec<u16>,
    /// The linear id the next `add` stamps (`m_next_linear_id`).
    next_linear_id: u16,
    /// The most entries ever in use at once (`m_max_count`).
    max_count: u16,
}

/// The serialised form of a [`HandleTable`], and what deserialisation validates.
#[derive(Deserialize)]
struct HandleTableData {
    entries: Vec<Option<Entry>>,
    free: Vec<u16>,
    next_linear_id: u16,
    max_count: u16,
}

/// Why serialised handle table data was rejected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum HandleLayoutError {
    /// The table has no entries or more than [`MAX_TABLE_SIZE`].
    #[error("table size {size} is outside 1..={MAX_TABLE_SIZE}")]
    Size {
        /// Entries present.
        size: usize,
    },
    /// An entry's linear id is outside `MIN_LINEAR_ID..=MAX_LINEAR_ID`.
    #[error("entry {index} has linear id {linear_id} outside the valid range")]
    LinearId {
        /// The entry.
        index: usize,
        /// Its linear id.
        linear_id: u16,
    },
    /// The next linear id is outside `MIN_LINEAR_ID..=MAX_LINEAR_ID`.
    #[error("next linear id {0} is outside the valid range")]
    NextLinearId(u16),
    /// The free stack is damaged.
    #[error("free stack is damaged: {0}")]
    FreeStack(#[from] FreeStackError),
    /// Some free entry is missing from the free stack.
    #[error("free entries and the free stack disagree")]
    FreeCount,
    /// The recorded high-water mark is below the entries in use, or above the table size.
    #[error("high-water mark {0} is inconsistent")]
    MaxCount(u16),
}

fn linear_id_in_range(linear_id: u16) -> bool {
    (MIN_LINEAR_ID..=MAX_LINEAR_ID).contains(&linear_id)
}

impl TryFrom<HandleTableData> for HandleTable {
    type Error = HandleLayoutError;

    fn try_from(data: HandleTableData) -> Result<Self, Self::Error> {
        let HandleTableData {
            entries,
            free,
            next_linear_id,
            max_count,
        } = data;
        if !(1..=usize::from(MAX_TABLE_SIZE)).contains(&entries.len()) {
            return Err(HandleLayoutError::Size {
                size: entries.len(),
            });
        }
        if !linear_id_in_range(next_linear_id) {
            return Err(HandleLayoutError::NextLinearId(next_linear_id));
        }
        let mut used = 0_usize;
        for (index, entry) in entries.iter().enumerate() {
            if let Some(entry) = entry {
                if !linear_id_in_range(entry.linear_id) {
                    return Err(HandleLayoutError::LinearId {
                        index,
                        linear_id: entry.linear_id,
                    });
                }
                used += 1;
            }
        }
        check_free_stack(&free, entries.len(), |at| {
            matches!(entries.get(at), Some(None))
        })?;
        if free.len() != entries.len() - used {
            return Err(HandleLayoutError::FreeCount);
        }
        if usize::from(max_count) < used || usize::from(max_count) > entries.len() {
            return Err(HandleLayoutError::MaxCount(max_count));
        }
        Ok(Self {
            entries,
            free,
            next_linear_id,
            max_count,
        })
    }
}

impl HandleTable {
    /// A table for `size` handles (`KHandleTable::Initialize`).
    ///
    /// A `size` of zero or less means the maximum, [`MAX_TABLE_SIZE`], as in Horizon; a larger
    /// size is [`HandleError::TableTooLarge`]. The NPDM's handle table size capability is the
    /// usual source of `size`.
    pub fn new(size: i32) -> Result<Self, HandleError> {
        if size > i32::from(MAX_TABLE_SIZE) {
            return Err(HandleError::TableTooLarge { requested: size });
        }
        let size = u16::try_from(size)
            .ok()
            .filter(|&s| s != 0)
            .unwrap_or(MAX_TABLE_SIZE);
        Ok(Self {
            entries: vec![None; usize::from(size)],
            // Popping takes `size - 1` first, then `size - 2`, and so on, as Horizon's chain does.
            free: (0..size).collect(),
            next_linear_id: MIN_LINEAR_ID,
            max_count: 0,
        })
    }

    /// The number of handles the table can hold (`GetTableSize`).
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.entries.len()
    }

    /// The number of handles in use (`GetCount`).
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len() - self.free.len()
    }

    /// Whether no handle is in use.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The most handles ever in use at once (`GetMaxCount`).
    #[must_use]
    pub fn max_len(&self) -> usize {
        usize::from(self.max_count)
    }

    /// The entry `handle` names, with its index. Mirrors `IsValidHandle` plus the reserved-bits
    /// check of `GetObjectImpl`: the handle must have zero reserved bits and a non-zero linear id
    /// (together these exclude `0` and both pseudo-handles), an index inside the table, an object
    /// there, and a matching linear id.
    fn locate(&self, handle: Handle) -> Result<(u16, &Entry), HandleError> {
        let invalid = HandleError::InvalidHandle { raw: handle.raw() };
        if handle.reserved() != 0 || handle.linear_id() == 0 {
            return Err(invalid);
        }
        let index = u16::try_from(handle.index()).map_err(|_| invalid)?;
        match self.entries.get(usize::from(index)) {
            Some(Some(entry)) if u32::from(entry.linear_id) == handle.linear_id() => {
                Ok((index, entry))
            }
            _ => Err(invalid),
        }
    }

    /// Inserts `object`, taking one reference to it through `refs`, and returns its handle
    /// (`KHandleTable::Add`). [`HandleError::OutOfHandles`] if the table is full. On any error
    /// the table and `refs` are unchanged.
    pub fn add<R: ObjectRefs>(
        &mut self,
        refs: &mut R,
        object: ObjectId,
    ) -> Result<Handle, HandleError> {
        let Some(&index) = self.free.last() else {
            return Err(HandleError::OutOfHandles);
        };
        let linear_id = self.next_linear_id;
        let handle = Handle::from_parts(index, linear_id).ok_or(HandleError::Corrupt)?;
        let Some(slot) = self
            .entries
            .get_mut(usize::from(index))
            .filter(|slot| slot.is_none())
        else {
            return Err(HandleError::Corrupt);
        };
        refs.open(object)?;
        *slot = Some(Entry { linear_id, object });
        self.free.pop();
        // `AllocateLinearId`: hand out the current value, then step, wrapping past the maximum.
        self.next_linear_id = if linear_id >= MAX_LINEAR_ID {
            MIN_LINEAR_ID
        } else {
            linear_id + 1
        };
        self.max_count = self
            .max_count
            .max(u16::try_from(self.len()).unwrap_or(u16::MAX));
        Ok(handle)
    }

    /// The object `handle` names, treating pseudo-handles as invalid
    /// (`GetObjectWithoutPseudoHandle`).
    pub fn get(&self, handle: Handle) -> Result<ObjectId, HandleError> {
        self.locate(handle).map(|(_, entry)| entry.object)
    }

    /// What `handle` names, answering the two pseudo-handles without consulting the table
    /// (the pseudo-handle part of `GetObject`). Anything else is looked up as by [`Self::get`].
    pub fn resolve(&self, handle: Handle) -> Result<Resolved, HandleError> {
        if handle == Handle::CURRENT_THREAD {
            Ok(Resolved::CurrentThread)
        } else if handle == Handle::CURRENT_PROCESS {
            Ok(Resolved::CurrentProcess)
        } else {
            self.get(handle).map(Resolved::Object)
        }
    }

    /// Closes `handle`, dropping its reference to the object through `refs` (`KHandleTable::Remove`).
    ///
    /// The result says whether that was the object's last reference. Pseudo-handles can't be
    /// removed. A stale or invalid handle is [`HandleError::InvalidHandle`]; on any error the
    /// table and `refs` are unchanged.
    pub fn remove<R: ObjectRefs>(
        &mut self,
        refs: &mut R,
        handle: Handle,
    ) -> Result<Removed, HandleError> {
        if handle.is_pseudo() {
            return Err(HandleError::InvalidHandle { raw: handle.raw() });
        }
        let (index, entry) = self.locate(handle)?;
        let object = entry.object;
        let Some(slot) = self.entries.get_mut(usize::from(index)) else {
            return Err(HandleError::Corrupt);
        };
        let closed = refs.close(object)?;
        *slot = None;
        self.free.push(index);
        Ok(Removed { object, closed })
    }
}

#[cfg(test)]
mod tests;
