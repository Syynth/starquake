//! Worker 216 `ExecuteSparseMapRequests` on the bookkeeping: a batch of entries that map parts of
//! registrations into sparse regions, or put a region's own GPU-only pages back
//! ([`GpuMemory::check_sparse_map`], then [`SparseAdmission::apply`]).
//!
//! Ported from Eden PR #4588,
//! `eden-pr4588/src/core/hle/service/nvdrv/devices/nvhost_as_gpu.cpp` @ e5fb3b97
//! (`nvhost_as_gpu::NativeRemap`) and `eden-pr4588/src/core/hle/service/am/service/gpu_session.cpp`
//! @ e5fb3b97 (`IGpuWorker::ExecuteSparseMapRequests`'s loop over the entries),
//! GPL-3.0-or-later, Copyright 2026 Eden Emulator Project.
//!
//! # The checks, in order
//!
//! A batch is checked whole before anything changes; the first failing check answers:
//!
//! 1. no such address space, `NotFound` (PR's `OwnsAddressSpace`, which 216's handler checks
//!    before this; here again, so the bookkeeping never trusts its caller);
//! 2. entry by entry, in order (PR's handler loop): the entry does not decode
//!    ([`super::super::sparse_map::decode`]), `NotImplemented`; its handle is not 0 and names no
//!    registration, `NotFound`;
//! 3. entry by entry, in order (`NativeRemap`'s loop, whose every refusal 216 answers
//!    `InvalidCurrentMemory`): the range is not inside one sparse region (the region with the
//!    greatest start at or below the range's: `allocation_map.upper_bound`, then
//!    `range->address - allocation->first > reservation.size || range->size > reservation.size -
//!    (range->address - allocation->first)`); for an entry with a handle, the range does not fit
//!    in the registration from the source offset (`range->source_offset > handle->orig_size ||
//!    range->size > handle->orig_size - range->source_offset`) or [`kind::select`] refuses its
//!    kinds;
//! 4. the address space would hold more than [`MAX_PIECES`] pieces once the whole batch is
//!    applied, `InvalidCurrentMemory` (PR: `updated.Ranges().size() > 65536` is
//!    `InsufficientMemory`, which 216 answers `InvalidCurrentMemory`; PR counts only the native
//!    sparse ranges, here every piece counts, as for 211);
//! 5. the batch takes something out and there is no room for another job, `Busy`
//!    ([`MAX_PENDING_JOBS`](super::MAX_PENDING_JOBS); PR completes at once and has no such check).
//!
//! PR's `NativeRemap` also refuses a space whose VM is not set up in the native ABI
//! (`!m_extended_abi || !vm.initialised`) and a handle that is not allocated or belongs to
//! another process; here an address space has no such state, and a registration is always
//! backed and belongs to the process whose bookkeeping holds it.
//!
//! # Applying a batch
//!
//! The entries apply in order, each over what the ones before it left, as PR's
//! `NativeSparseMappingTable::Replace` does:
//!
//! - **An entry with a handle** maps the registration's pages from its source offset at its
//!   range, with the backend kind [`kind::select`] gives; the 211 mappings it overlaps are cut to
//!   the parts outside it, as a 211 cuts them (`crate::gpu`, "Conflicts"). Its page-table record
//!   is a 211 mapping's: a 213 can unmap it, a 212 change its kind, a 102 take it out.
//! - **An entry with no handle** cuts the mappings it overlaps likewise and maps nothing: the
//!   region's GPU-only pages show again (PR: `gmmu->MapSparse`).
//!
//! **Maps at once, removals in order**, as everywhere in the bookkeeping (`super`'s module docs,
//! "What the table shows"). The parts of mappings an entry with no handle takes out are sent to
//! the GPU as `VmUpdate::Unmap`s where they are the top layer (where the syncpoint shim is not),
//! then a barrier, and those pages are held until the barrier's ticket is drained; the job's
//! event is signalled then ([`SparseCompletion::Ordered`]). A part taken out and mapped again by
//! a later entry of the same batch is unmapped all the same, and its new mapping shows at the
//! drain. A batch that takes nothing out (only entries with handles, or entries with no handle
//! over pages already the region's) completes at once ([`SparseCompletion::AtOnce`]). A part
//! an entry with a handle replaces is not a removal: it is published at once, as a 211 over a
//! live mapping is.
//!
//! **The selected kinds.** One byte per entry, in order (PR writes `entries[i].fallback_kind`
//! after `NativeRemap`): for an entry with a handle the selected kind [`kind::select`] gives; for
//! one with no handle the fallback byte as the entry sent it (PR's `NativeRemap` leaves that
//! entry's `fallback_kind` unchanged).
//!
//! # Cost
//!
//! The check copies the 211 mappings the batch's ranges overlap (each once, by range in their
//! map, over the union of the ranges), applies the entries to the copy, and keeps the result for
//! [`SparseAdmission::apply`] to put in place: every mapping copied is cut or replaced by the
//! batch. So what is copied is the batch's entries and the mappings it changes, never the
//! address space's other mappings. Beyond that, the piece count walks the address space's
//! mappings once ([`GpuMemory::pieces`] is a sum over their runs), as a 211 and a 212 do.

use std::collections::BTreeMap;

use starquake_gpu::interface::{Kind, Ticket, VmUpdate};
use starquake_kernel::id::ObjectId;
use starquake_kernel::result::KernelError;

use super::super::kind;
use super::super::sparse_map::{SparseMap, SparseMapRequest, decode};
use super::{
    AsId, GpuMemory, GpuVmFront, Layer, MAX_PIECES, Record, Space, in_range, page_range, trimmed,
    union,
};

/// How a 216 completes ([`SparseAdmission::apply`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SparseCompletion {
    /// Nothing was taken out: the caller signals the job's event now.
    AtOnce,
    /// Removals were sent to the GPU: the drain signals the event when this ticket comes back.
    Ordered(Ticket),
}

/// A 216 batch that passed [`GpuMemory::check_sparse_map`]. It holds the bookkeeping, so nothing
/// changes it before [`apply`](SparseAdmission::apply), which therefore cannot fail; dropping it
/// changes nothing.
#[derive(Debug)]
pub struct SparseAdmission<'m> {
    memory: &'m mut GpuMemory,
    space: AsId,
    /// The starts of the mappings the batch changes, to take out.
    replaced: Vec<u64>,
    /// The mappings in their place, once the batch is applied.
    records: BTreeMap<u64, Record>,
    /// The removals to send, in address order.
    updates: Vec<(AsId, VmUpdate)>,
    /// The batch's ranges, merged: published once it is applied.
    ranges: Vec<(u64, u64)>,
    /// One selected kind per entry.
    selected: Vec<u8>,
}

impl SparseAdmission<'_> {
    /// The selected kinds, one byte per entry, in order (module docs, "The selected kinds"):
    /// what 216 writes to its out buffer.
    #[must_use]
    pub fn selected(&self) -> &[u8] {
        &self.selected
    }

    /// Whether the batch takes something out, and so completes in order.
    #[must_use]
    pub fn is_ordered(&self) -> bool {
        !self.updates.is_empty()
    }

    /// Applies the batch: puts its mappings in place, sends the GPU its removals and a barrier
    /// if it has any (the job, with `event`, waiting for the barrier's ticket), then publishes
    /// its ranges but for the pages a waiting job holds.
    pub fn apply(self, front: &dyn GpuVmFront, event: ObjectId) -> SparseCompletion {
        let Self {
            memory,
            space,
            replaced,
            records,
            updates,
            ranges,
            selected: _,
        } = self;
        if let Some(held) = memory.spaces.get_mut(&space) {
            for start in replaced {
                held.records.remove(&start);
            }
            held.records.extend(records);
        }
        let completion = if updates.is_empty() {
            SparseCompletion::AtOnce
        } else {
            SparseCompletion::Ordered(memory.order(front, &updates, event, Vec::new()))
        };
        memory.publish(space, &ranges);
        completion
    }
}

/// An entry that passed the per-entry checks: the decoded entry, for one with a handle its
/// backend kind, and the byte the out buffer gets for it.
struct Checked {
    entry: SparseMap,
    backend: Option<Kind>,
    selected: u8,
}

/// The batch applied to a copy of the mappings its ranges overlap: the copy afterwards, and the
/// parts of mappings its entries with no handle took out.
struct Applied {
    records: BTreeMap<u64, Record>,
    removed: Vec<(u64, u64)>,
}

/// Applies `checked`, in order, to `records` (the mappings the batch's ranges overlap, copied):
/// each entry cuts the mappings it overlaps, and one with a handle maps in their place.
fn apply_entries(mut records: BTreeMap<u64, Record>, checked: &[Checked]) -> Applied {
    let mut removed = Vec::new();
    for Checked { entry, backend, .. } in checked {
        let (start, end) = (entry.range.start(), entry.range.end());
        let (cut, kept) = trimmed(&records, start, end);
        for at in cut {
            if let Some(record) = records.remove(&at)
                && backend.is_none()
            {
                removed.push((at.max(start), record.end.min(end)));
            }
        }
        records.extend(kept);
        if let Some(kind) = *backend {
            let record = Record {
                end,
                handle: entry.handle,
                offset: entry.source_offset,
                spans: vec![(start, end, kind)],
            };
            records.insert(start, record);
        }
    }
    Applied { records, removed }
}

impl GpuMemory {
    /// 216's checks on a batch in `space`, in the module docs' order; the admitted batch, whose
    /// [`apply`](SparseAdmission::apply) cannot fail.
    pub fn check_sparse_map(
        &mut self,
        space: AsId,
        requests: &[SparseMapRequest],
    ) -> Result<SparseAdmission<'_>, KernelError> {
        let held = self.spaces.get(&space).ok_or(KernelError::NotFound)?;
        let mut entries = Vec::with_capacity(requests.len());
        for request in requests {
            let entry = decode(request).ok_or(KernelError::NotImplemented)?;
            if entry.handle != 0 && !self.registrations.contains_key(&entry.handle) {
                return Err(KernelError::NotFound);
            }
            entries.push(entry);
        }
        let refused = KernelError::InvalidCurrentMemory;
        let checked = entries
            .into_iter()
            .map(|entry| self.place_entry(held, entry).ok_or(refused))
            .collect::<Result<Vec<Checked>, KernelError>>()?;
        let ranges = union(
            checked
                .iter()
                .map(|c| (c.entry.range.start(), c.entry.range.end()))
                .collect(),
        );
        let mut copied: BTreeMap<u64, Record> = BTreeMap::new();
        for &(low, high) in &ranges {
            for (&start, record) in in_range(&held.records, |r| r.end, low, high) {
                copied.entry(start).or_insert_with(|| record.clone());
            }
        }
        let replaced: Vec<u64> = copied.keys().copied().collect();
        let spans_before: usize = copied.values().map(|r| r.spans.len()).sum();
        let Applied { records, removed } = apply_entries(copied, &checked);
        let spans_after: usize = records.values().map(|r| r.spans.len()).sum();
        let after = (self.pieces_of(held) + spans_after).saturating_sub(spans_before);
        if after > MAX_PIECES {
            return Err(refused);
        }
        let mut updates = Vec::new();
        for (low, high) in union(removed) {
            for (from, to) in self.shown(held, Layer::Mapping, low, high) {
                if let Some(range) = page_range(from, to) {
                    updates.push((space, VmUpdate::Unmap { range }));
                }
            }
        }
        if !updates.is_empty() {
            self.next_ticket()?;
        }
        let selected = checked.iter().map(|c| c.selected).collect();
        Ok(SparseAdmission {
            memory: self,
            space,
            replaced,
            records,
            updates,
            ranges,
            selected,
        })
    }

    /// `NativeRemap`'s checks of one entry in `held` (module docs, check 3): inside one sparse
    /// region and, with a handle, inside its registration from its source with kinds
    /// [`kind::select`] accepts; `None` if any fails.
    fn place_entry(&self, held: &Space, entry: SparseMap) -> Option<Checked> {
        let (start, end) = (entry.range.start(), entry.range.end());
        let (_, region) = held.regions.range(..=start).next_back()?;
        if end > region.end {
            return None;
        }
        if entry.handle == 0 {
            return Some(Checked {
                entry,
                backend: None,
                selected: entry.fallback,
            });
        }
        let size = self.registrations.get(&entry.handle)?.range.size();
        if entry.source_offset > size || entry.range.size() > size - entry.source_offset {
            return None;
        }
        let kinds = kind::select(entry.kind, entry.fallback)?;
        Some(Checked {
            entry,
            backend: Some(kinds.backend),
            selected: kinds.selected,
        })
    }
}
