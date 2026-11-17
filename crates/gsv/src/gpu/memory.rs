//! The GPU memory bookkeeping of one process: notification events, graphics devices, address
//! spaces with their translation tables, memory registrations (101), mappings (211), sparse
//! regions (214), and the ordered jobs (102, 212, 213, 215) the GPU has not yet reported done
//! (`docs/specs/graphics-server.md`, "The model" and "How mappings reach the GPU"; port slice 7).
//!
//! The request-time checks are ported from Eden PR #4588,
//! `eden-pr4588/src/core/hle/service/am/service/gpu_session.cpp` and `gpu_memory_registration.h`
//! @ e5fb3b97 (`IGpuWorker`, `RegisterMemory`, `MapMemory`, `DecodeGpuSparseRegion`,
//! `IsValidGpuSparseRegionRange`), and `eden-pr4588/src/core/hle/service/nvdrv/devices/nvhost_as_gpu.cpp`
//! @ e5fb3b97 (`MapNative4K`, `ReserveNativeSparse4K`, `ModifyKind`); the mapping behaviour from
//! Eden-NX2, `eden-nx2/src/core/hle/service/nx2_gsv.h` @ 50fa86bb (`INx2GpuSession`). Both
//! GPL-3.0-or-later, Copyright 2026 Eden Emulator Project. The module docs of [`crate::gpu`] list
//! every check, its source, and where this differs from the references.
//!
//! # What the table shows: layers
//!
//! The translation table of an address space ([`GpuVm`]) is never edited piece by piece here.
//! Every change publishes a range again from the bookkeeping ([`GpuMemory`]'s `republish`), in
//! four layers, each mapped over the one before (a map replaces what it overlaps):
//!
//! 1. sparse regions, as [`Target::GpuOnly`] at their own offsets, kind [`SPARSE_KIND`];
//! 2. registrations, each at its own address (decision 5: GPU address == CPU address, row 30),
//!    kind [`SELF_MAP_KIND`], in every address space (row 30: "identity-maps every range into
//!    each GMMU");
//! 3. 211 mappings, with the kinds their requests and later 212s gave them;
//! 4. the syncpoint shim, if one is placed.
//!
//! So a range always shows the top layer the bookkeeping holds there, and removing a layer
//! (a 213 inside a sparse region, say) shows what is under it once the removal is published.
//!
//! **Maps are published at once, removals in order.** A registration, a 211 and a 214 are
//! written into the table during the request. A 102, 212, 213 and 215 change the bookkeeping
//! during the request, and send the GPU ordered updates and then one barrier
//! ([`GpuVmFront`]); when the barrier's ticket comes back, the drain publishes every range the
//! job removed again ([`GpuMemory::ordered_done`]), which shows the layers under it, and only
//! then releases the job's device-share lock and signals its event. Between the GPU applying a
//! removal and that publish, the removed range is a hole in the table even where a lower layer
//! (or a map made after the removal was asked) belongs: that is a **guess** that no GPU work
//! queued after the request and before the job's event reads those pages. No reference has
//! the window (NX2 and PR change their tables at once).

use std::collections::BTreeMap;
use std::sync::Arc;

use starquake_gpu::interface::{Kind, RegionId, Target, Ticket, VmUpdate};
use starquake_gpu::memory::Translate;
use starquake_gpu::vm::{GpuVm, Mapping};
use starquake_kernel::id::ObjectId;
use starquake_kernel::memory::DeviceShareToken;
use starquake_kernel::result::KernelError;
use starquake_memory::{PAGE_SIZE, PageRange};

use super::kind::{self, GENERIC_16BX2};
use super::{AsId, GpuVmFront};

/// How many notification events a gpu session has: PR's `GpuDeviceRegistry::events`
/// (`std::array<Kernel::KEvent*, 64>`). gpu 11 answers indices 0 to one less (graphics-server,
/// gpu 11: **conflict**, PR's 63 chosen over NX2's 31).
pub const NOTIFICATION_COUNT: u32 = 64;

/// How many address spaces gpu 33 makes: PR's `GpuDeviceRegistry::address_spaces`
/// (`std::array<std::optional<s32>, 8>`); the ninth is `OutOfResource`.
pub const MAX_ADDRESS_SPACES: usize = 8;

/// The size of PR's graphics device table (`GpuDeviceRegistry::devices`, 256 entries); gpu 41
/// hands out ids 1 to one less (`for (u32 candidate = 1; ...)`), and a device byte names one.
pub const DEVICE_SLOTS: u32 = 256;

/// The most pieces an address space holds (module docs of [`crate::gpu`], "Bounds"): PR's bound
/// on an address space's sparse mappings (`nvhost_as_gpu::NativeRemap`:
/// `updated.Ranges().size() > 65536` is `InsufficientMemory`), applied to every piece here.
pub const MAX_PIECES: usize = 65_536;

/// The most ordered jobs waiting for the GPU at once; one more is `Busy`. A **guess**: no
/// reference bounds them (NX2's `RunInOrder` queues without limit, PR completes at once).
pub const MAX_PENDING_JOBS: usize = 4096;

/// The first GPU address of the window NVN reserves: row 25 of `eden-nx2/NX2_NOTES.md`
/// (`AllocAsEx(... [0x8000000, 0xC000000000) ...)`, observed) and PR's
/// `GetGpuAddressSpaceLayout(true).start`
/// (`eden-pr4588/src/core/hle/service/nvdrv/devices/gpu_address_space_layout.h` @ e5fb3b97).
pub const VA_WINDOW_START: u64 = 0x800_0000;

/// One past the last GPU address of that window: row 25's `0xC000000000`, PR's
/// `GetGpuAddressSpaceLayout(true).split` (`3ULL << 38`).
pub const VA_WINDOW_END: u64 = 0xC0_0000_0000;

/// The kind a registration's mapping at its own address gets. A **guess**: no reference gives
/// one (PR maps only on a 211, which names its kind; NX2 keeps no kinds); generic memory, PR's
/// backend for an uncompressed generic request ([`kind::select`] of 6 and 6).
pub const SELF_MAP_KIND: Kind = GENERIC_16BX2;

/// The kind a sparse region's GPU-only pages get. A **guess**, as for [`SELF_MAP_KIND`]: PR's
/// `gmmu->MapSparse` takes none.
pub const SPARSE_KIND: Kind = GENERIC_16BX2;

/// The page shift of a 4 KiB sparse region (PR `CreateSparseRegion`: `request->page_shift ==
/// 12`).
pub const SMALL_PAGE_SHIFT: u8 = 12;

/// The page shift of a 64 KiB sparse region, which PR decodes and refuses as not implemented
/// (`IsValidGpuSparseRegionRange`: `page_shift != 12 && page_shift != 16`).
pub const BIG_PAGE_SHIFT: u8 = 16;

/// A graphics device gpu 41 made (PR `GpuDeviceRegistry::Device`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Device {
    /// The name word.
    pub name: u64,
    /// The debug label word.
    pub label: u64,
}

/// A memory registration (101): a CPU range locked for the GPU, mapped at its own address in
/// every address space.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Registration {
    /// The CPU range.
    pub range: PageRange,
    /// The graphics device the request named.
    pub device: u8,
    /// The device-share lock taken on the range, released when the destroy is done.
    pub token: DeviceShareToken,
}

/// A run of pages of one 211 mapping with one kind: `(start, end, kind)`.
pub(crate) type Span = (u64, u64, Kind);

/// A 211 mapping as the bookkeeping holds it, keyed by its GPU start.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Record {
    /// One past its last GPU address.
    pub(crate) end: u64,
    /// The registration it maps.
    pub(crate) handle: u32,
    /// Where its first page is in the registration.
    pub(crate) offset: u64,
    /// Its pages' kinds, in order, covering exactly its range.
    pub(crate) spans: Vec<Span>,
}

/// A sparse region (214), keyed by its GPU start.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Region {
    /// One past its last GPU address.
    pub(crate) end: u64,
    /// Its id, which its GPU-only pages name.
    pub(crate) id: RegionId,
}

/// An address space (gpu 33): its table and what gsv placed in it.
#[derive(Debug)]
pub(crate) struct Space {
    /// The table the GPU reads, shared with it.
    pub(crate) vm: Arc<GpuVm>,
    /// The 211 mappings, by GPU start; they never overlap.
    pub(crate) records: BTreeMap<u64, Record>,
    /// The sparse regions, by GPU start; they never overlap.
    pub(crate) regions: BTreeMap<u64, Region>,
    /// The syncpoint shim's page, if placed.
    pub(crate) shim: Option<u64>,
}

impl PartialEq for Space {
    /// The bookkeeping, and the table's mappings and version: two tables are equal when they
    /// translate alike, whichever `Arc` holds them.
    fn eq(&self, other: &Self) -> bool {
        self.records == other.records
            && self.regions == other.regions
            && self.shim == other.shim
            && self.vm.mappings() == other.vm.mappings()
            && self.vm.generation() == other.vm.generation()
    }
}

impl Eq for Space {}

/// An ordered job (102, 212, 213, 215) the GPU has not yet reported done.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Job {
    /// The notification event its request named.
    pub(crate) event: ObjectId,
    /// The device-share locks to release once it is done (a 102's one).
    pub(crate) releases: Vec<DeviceShareToken>,
    /// The ranges to publish again once it is done.
    pub(crate) reveal: Vec<(AsId, PageRange)>,
}

/// What the drain does once an ordered job is done ([`GpuMemory::ordered_done`]): release these
/// locks, then signal this event.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Finished {
    /// The device-share locks to release, in order.
    pub releases: Vec<DeviceShareToken>,
    /// The job's notification event.
    pub event: ObjectId,
}

/// The ids handed out next.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Counters {
    /// The next address space id.
    pub(crate) space: u32,
    /// The next memory handle.
    pub(crate) handle: u32,
    /// The next sparse region id.
    pub(crate) region: u32,
    /// The next ticket.
    pub(crate) ticket: u64,
}

impl Counters {
    /// The first of each: 1, so that 0 never names anything (a guess at PR's handle numbering,
    /// whose nvmap ids start at 1; NX2 answers address space 1).
    const FIRST: Self = Self {
        space: 1,
        handle: 1,
        region: 1,
        ticket: 1,
    };
}

/// 101 `CreateGpuAccessibleMemory`, decoded (PR `GpuMemoryRegistration`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RegisterRequest {
    /// The first CPU address.
    pub first: u64,
    /// The last CPU address (inclusive).
    pub last: u64,
    /// The flags byte.
    pub flags: u8,
    /// The graphics device byte.
    pub device: u8,
}

/// 211 `NormalMapGpuMemory4K`, decoded (PR `MapMemory`), without its slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MapRequest {
    /// The address space.
    pub space: AsId,
    /// The first GPU address.
    pub first: u64,
    /// The last GPU address (inclusive).
    pub last: u64,
    /// The memory handle 101 returned.
    pub handle: u32,
    /// Where the mapping starts in the registration.
    pub offset: u64,
    /// The permission byte.
    pub permission: u8,
    /// The cache byte.
    pub cache: u8,
    /// The wanted kind.
    pub kind: u8,
    /// The fallback kind.
    pub fallback: u8,
    /// The mode byte.
    pub mode: u8,
    /// The graphics device byte.
    pub device: u8,
    /// The sync byte.
    pub sync: u8,
}

/// 212 `NormalModifyMemoryKind4K`, decoded, without its sync byte and slot (PR checks those
/// first).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ModifyRequest {
    /// The address space.
    pub space: AsId,
    /// The first GPU address.
    pub first: u64,
    /// The last GPU address (inclusive).
    pub last: u64,
    /// The wanted kind.
    pub kind: u8,
    /// The fallback kind.
    pub fallback: u8,
}

/// 213 `NormalUnmapGpuMemory4K` or 215 `DestroySparseRegion`, decoded, without its sync byte
/// and slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RangeRequest {
    /// The address space.
    pub space: AsId,
    /// The first GPU address.
    pub first: u64,
    /// The last GPU address (inclusive).
    pub last: u64,
}

/// 214 `CreateSparseRegion`, decoded (PR `GpuSparseRegionRequest`), without its slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SparseRequest {
    /// The address space.
    pub space: AsId,
    /// The first GPU address.
    pub first: u64,
    /// The last GPU address (inclusive).
    pub last: u64,
    /// The page shift.
    pub page_shift: u8,
    /// The allocation mode byte.
    pub mode: u8,
    /// The graphics device byte.
    pub device: u8,
    /// The sync byte.
    pub sync: u8,
}

/// `first..=last` as a start and a size, if `last` is not below `first` and is not the last
/// address (PR: `end >= begin && end != ~u64{0}`).
pub(crate) fn inclusive(first: u64, last: u64) -> Option<(u64, u64)> {
    if last < first || last == u64::MAX {
        return None;
    }
    // `last - first` is below `u64::MAX`, so one more fits.
    Some((first, last - first + 1))
}

/// Whether `size` bytes at `start` lie in NVN's window (PR `MapNative4K` and
/// `ReserveNativeSparse4K`: `address < va_range_start || address >= va_range_split || size >
/// va_range_split - address` is `BadParameter`).
pub(crate) fn in_window(start: u64, size: u64) -> bool {
    (VA_WINDOW_START..VA_WINDOW_END).contains(&start) && size <= VA_WINDOW_END - start
}

/// The part of `range` inside `start..end`, if any.
fn intersect(range: PageRange, start: u64, end: u64) -> Option<PageRange> {
    let low = range.start().max(start);
    let high = range.end().min(end);
    (low < high)
        .then(|| PageRange::new(low, high - low).ok())
        .flatten()
}

/// The spans of `spans` cut to `low..high`.
fn clip(spans: &[Span], low: u64, high: u64) -> Vec<Span> {
    spans
        .iter()
        .filter_map(|&(start, end, kind)| {
            let start = start.max(low);
            let end = end.min(high);
            (start < end).then_some((start, end, kind))
        })
        .collect()
}

/// The pieces of the records of `records` that overlap `start..end`, cut outside it: the
/// starts of the records to take out, and the pieces to put back (NX2's `TrimMappings`, on the
/// bookkeeping).
fn trimmed(
    records: &BTreeMap<u64, Record>,
    start: u64,
    end: u64,
) -> (Vec<u64>, Vec<(u64, Record)>) {
    let before = records
        .range(..start)
        .next_back()
        .filter(|(_, record)| record.end > start);
    let mut removed = Vec::new();
    let mut kept = Vec::new();
    for (&record_start, record) in before.into_iter().chain(records.range(start..end)) {
        removed.push(record_start);
        if record_start < start {
            kept.push((
                record_start,
                Record {
                    end: start,
                    handle: record.handle,
                    offset: record.offset,
                    spans: clip(&record.spans, record_start, start),
                },
            ));
        }
        if record.end > end {
            kept.push((
                end,
                Record {
                    end: record.end,
                    handle: record.handle,
                    // `end - record_start` is below the record's size, which fit with the offset.
                    offset: record.offset.saturating_add(end - record_start),
                    spans: clip(&record.spans, end, record.end),
                },
            ));
        }
    }
    (removed, kept)
}

/// Whether anything of `map` lies in `start..end`.
fn overlaps<T>(map: &BTreeMap<u64, T>, end_of: impl Fn(&T) -> u64, start: u64, end: u64) -> bool {
    map.range(..end)
        .next_back()
        .is_some_and(|(_, value)| end_of(value) > start)
}

/// The entries of `map` (non-overlapping, keyed by start) that may overlap `low..high`: the one
/// starting before `low`, then those starting inside.
fn in_range<T>(
    map: &BTreeMap<u64, T>,
    end_of: impl Fn(&T) -> u64,
    low: u64,
    high: u64,
) -> impl Iterator<Item = (&u64, &T)> {
    let before = map
        .range(..low)
        .next_back()
        .filter(|(_, value)| end_of(value) > low);
    before.into_iter().chain(map.range(low..high))
}

/// One process's GPU memory bookkeeping (module docs). Lives in gsv's shared state, under its
/// one lock ([`crate::shared::Locked::gpu`]).
#[derive(Debug, PartialEq, Eq)]
pub struct GpuMemory {
    pub(crate) events: BTreeMap<u32, ObjectId>,
    pub(crate) worker: bool,
    pub(crate) devices: BTreeMap<u8, Device>,
    pub(crate) spaces: BTreeMap<AsId, Space>,
    pub(crate) registrations: BTreeMap<u32, Registration>,
    pub(crate) pending: BTreeMap<Ticket, Job>,
    pub(crate) next: Counters,
}

impl Default for GpuMemory {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuMemory {
    /// No events, devices, address spaces or registrations.
    #[must_use]
    pub fn new() -> Self {
        Self {
            events: BTreeMap::new(),
            worker: false,
            devices: BTreeMap::new(),
            spaces: BTreeMap::new(),
            registrations: BTreeMap::new(),
            pending: BTreeMap::new(),
            next: Counters::FIRST,
        }
    }

    // ----- The gpu session's state -------------------------------------------------------------

    /// The notification event at `index`, if gpu 12 made it.
    #[must_use]
    pub fn notification_event(&self, index: u32) -> Option<ObjectId> {
        self.events.get(&index).copied()
    }

    /// Keeps `event` as the notification event at `index` (gpu 12 made it).
    pub(crate) fn set_notification_event(&mut self, index: u32, event: ObjectId) {
        self.events.insert(index, event);
    }

    /// Whether gpu 21 has made the worker.
    #[must_use]
    pub fn has_worker(&self) -> bool {
        self.worker
    }

    /// Marks the worker made (gpu 21).
    pub(crate) fn make_worker(&mut self) {
        self.worker = true;
    }

    /// The graphics device `id`, if gpu 41 made it.
    #[must_use]
    pub fn device(&self, id: u8) -> Option<Device> {
        self.devices.get(&id).copied()
    }

    /// gpu 41: the first free device id from 1 (PR `CreateGraphicsDevice`), or `OutOfResource`.
    pub fn create_device(&mut self, device: Device) -> Result<u8, KernelError> {
        let id = (1..DEVICE_SLOTS)
            .filter_map(|candidate| u8::try_from(candidate).ok())
            .find(|candidate| !self.devices.contains_key(candidate))
            .ok_or(KernelError::OutOfResource)?;
        self.devices.insert(id, device);
        Ok(id)
    }

    /// gpu 42: forgets device `id` (PR `DestroyGraphicsDevice`), or `NotFound`.
    pub fn destroy_device(&mut self, id: u8) -> Result<(), KernelError> {
        self.devices
            .remove(&id)
            .map(|_| ())
            .ok_or(KernelError::NotFound)
    }

    /// gpu 33: a new address space with an empty table, given to `front`, with every
    /// registration mapped at its own address. `OutOfResource` past [`MAX_ADDRESS_SPACES`] or
    /// when ids run out.
    pub fn create_address_space(&mut self, front: &dyn GpuVmFront) -> Result<AsId, KernelError> {
        if self.spaces.len() >= MAX_ADDRESS_SPACES {
            return Err(KernelError::OutOfResource);
        }
        let id = AsId(self.next.space);
        let next = self
            .next
            .space
            .checked_add(1)
            .ok_or(KernelError::OutOfResource)?;
        let vm = Arc::new(GpuVm::new());
        front.create_address_space(id, Arc::clone(&vm));
        self.spaces.insert(
            id,
            Space {
                vm,
                records: BTreeMap::new(),
                regions: BTreeMap::new(),
                shim: None,
            },
        );
        self.next.space = next;
        self.publish_all(id);
        Ok(id)
    }

    /// The address spaces, in id order.
    pub fn address_spaces(&self) -> impl Iterator<Item = AsId> + '_ {
        self.spaces.keys().copied()
    }

    /// The table of address space `space`.
    #[must_use]
    pub fn table(&self, space: AsId) -> Option<&Arc<GpuVm>> {
        self.spaces.get(&space).map(|space| &space.vm)
    }

    /// The registration `handle` names.
    #[must_use]
    pub fn registration(&self, handle: u32) -> Option<&Registration> {
        self.registrations.get(&handle)
    }

    /// How many ordered jobs wait for the GPU.
    #[must_use]
    pub fn pending_jobs(&self) -> usize {
        self.pending.len()
    }

    /// How many pieces address space `space` holds (module docs of [`crate::gpu`], "Bounds"):
    /// every registration, every sparse region of it, and every run of one kind of each of its
    /// 211 mappings.
    #[must_use]
    pub fn pieces(&self, space: AsId) -> usize {
        self.spaces
            .get(&space)
            .map_or(0, |space| self.pieces_of(space))
    }

    fn pieces_of(&self, space: &Space) -> usize {
        let spans: usize = space.records.values().map(|r| r.spans.len()).sum();
        self.registrations.len() + space.regions.len() + spans
    }

    // ----- The worker's jobs -------------------------------------------------------------------

    /// The event of job slot `slot`: PR's header check (`slot < events.size() &&
    /// events[slot]`), `InvalidCombination` when gpu 12 made none there.
    pub fn job_event(&self, slot: u32) -> Result<ObjectId, KernelError> {
        self.notification_event(slot)
            .ok_or(KernelError::InvalidCombination)
    }

    /// 101's checks, in PR's order (`RegisterMemory`), before the kernel is asked anything:
    /// the flags (`NotImplemented`), the device (`NotFound`), the range (`InvalidSize`), its
    /// start (`InvalidAddress`) and size (`InvalidSize`) alignment, and room for one more piece
    /// in every address space (`OutOfResource`, where PR's `nvmap.CreateHandle` fails). Returns
    /// the range.
    pub fn check_registration(&self, request: &RegisterRequest) -> Result<PageRange, KernelError> {
        if request.flags != 0 {
            return Err(KernelError::NotImplemented);
        }
        if !self.devices.contains_key(&request.device) {
            return Err(KernelError::NotFound);
        }
        let (start, size) =
            inclusive(request.first, request.last).ok_or(KernelError::InvalidSize)?;
        if !start.is_multiple_of(PAGE_SIZE) {
            return Err(KernelError::InvalidAddress);
        }
        let range = PageRange::new(start, size).map_err(|_| KernelError::InvalidSize)?;
        let full = self.registrations.len() >= MAX_PIECES
            || self
                .spaces
                .values()
                .any(|space| self.pieces_of(space) >= MAX_PIECES);
        if full || self.next.handle.checked_add(1).is_none() {
            return Err(KernelError::OutOfResource);
        }
        Ok(range)
    }

    /// 101 after the kernel checks: records the registration with its lock's token and maps it
    /// at its own address in every address space. Returns its handle. Call
    /// [`check_registration`](Self::check_registration) first; this repeats its room check, and
    /// on that error records nothing (the caller then releases the lock).
    pub fn register(
        &mut self,
        range: PageRange,
        device: u8,
        token: DeviceShareToken,
    ) -> Result<u32, KernelError> {
        let handle = self.next.handle;
        let next = handle.checked_add(1).ok_or(KernelError::OutOfResource)?;
        if self.registrations.len() >= MAX_PIECES
            || self
                .spaces
                .values()
                .any(|s| self.pieces_of(s) >= MAX_PIECES)
        {
            return Err(KernelError::OutOfResource);
        }
        self.registrations.insert(
            handle,
            Registration {
                range,
                device,
                token,
            },
        );
        self.next.handle = next;
        let spaces: Vec<AsId> = self.spaces.keys().copied().collect();
        for space in spaces {
            self.republish(space, range);
        }
        Ok(handle)
    }

    /// The checks every ordered job makes before it changes anything: room for one more
    /// pending job (`Busy`) and a ticket (`OutOfResource`).
    fn next_ticket(&self) -> Result<Ticket, KernelError> {
        if self.pending.len() >= MAX_PENDING_JOBS {
            return Err(KernelError::Busy);
        }
        self.next
            .ticket
            .checked_add(1)
            .map(|_| Ticket(self.next.ticket))
            .ok_or(KernelError::OutOfResource)
    }

    /// Sends `updates` to the GPU in order, then a barrier with the next ticket, and keeps
    /// `job` until the ticket comes back. [`next_ticket`](Self::next_ticket) must have passed.
    fn order(&mut self, front: &dyn GpuVmFront, updates: &[(AsId, VmUpdate)], job: Job) -> Ticket {
        let ticket = Ticket(self.next.ticket);
        self.next.ticket = self.next.ticket.saturating_add(1);
        for &(space, update) in updates {
            front.update_vm(space, update, None);
        }
        front.barrier(ticket);
        self.pending.insert(ticket, job);
        ticket
    }

    /// 102 `DestroyGpuAccessibleMemory`: takes registration `handle` out, with every 211
    /// mapping of it, and sends the GPU their removals (`NotFound` if no registration has the
    /// handle). Its lock is released, and `event` signalled, once the GPU is done.
    pub fn destroy_memory(
        &mut self,
        front: &dyn GpuVmFront,
        handle: u32,
        event: ObjectId,
    ) -> Result<Ticket, KernelError> {
        let registration = *self
            .registrations
            .get(&handle)
            .ok_or(KernelError::NotFound)?;
        self.next_ticket()?;
        self.registrations.remove(&handle);
        let mut updates = Vec::new();
        let mut reveal = Vec::new();
        for (&id, space) in &mut self.spaces {
            updates.push((
                id,
                VmUpdate::DestroyMemory {
                    range: registration.range,
                },
            ));
            reveal.push((id, registration.range));
            let mapped: Vec<u64> = space
                .records
                .iter()
                .filter(|(_, record)| record.handle == handle)
                .map(|(&start, _)| start)
                .collect();
            for start in mapped {
                let Some(record) = space.records.remove(&start) else {
                    continue;
                };
                if let Ok(range) = PageRange::new(start, record.end - start) {
                    updates.push((id, VmUpdate::Unmap { range }));
                    reveal.push((id, range));
                }
            }
        }
        let job = Job {
            event,
            releases: vec![registration.token],
            reveal,
        };
        Ok(self.order(front, &updates, job))
    }

    /// 211 `NormalMapGpuMemory4K`: maps part of a registration at a GPU address, at once, and
    /// returns the selected kind. Checks in PR's order (`MapMemory`, then `MapNative4K`): see
    /// the module docs of [`crate::gpu`].
    pub fn map(&mut self, request: &MapRequest) -> Result<u8, KernelError> {
        let space = self
            .spaces
            .get(&request.space)
            .ok_or(KernelError::NotFound)?;
        let registration = self
            .registrations
            .get(&request.handle)
            .ok_or(KernelError::NotFound)?;
        if !self.devices.contains_key(&request.device) {
            return Err(KernelError::NotFound);
        }
        if request.permission != 0 || request.cache > 1 || request.mode != 0 || request.sync > 1 {
            return Err(KernelError::NotImplemented);
        }
        let selected =
            kind::select(request.kind, request.fallback).ok_or(KernelError::NotImplemented)?;
        let (start, size) =
            inclusive(request.first, request.last).ok_or(KernelError::InvalidSize)?;
        // `MapNative4K`'s `BadParameter`, which `MapMemory` answers as `InvalidCurrentMemory`.
        let refused = KernelError::InvalidCurrentMemory;
        if !start.is_multiple_of(PAGE_SIZE)
            || !size.is_multiple_of(PAGE_SIZE)
            || !request.offset.is_multiple_of(PAGE_SIZE)
        {
            return Err(refused);
        }
        let held = registration.range.size();
        if request.offset > held || size > held - request.offset {
            return Err(refused);
        }
        if !in_window(start, size) {
            return Err(refused);
        }
        let range = PageRange::new(start, size).map_err(|_| refused)?;
        // conflict (decision 8, "the one a title ran on"): NX2's `MapAt` replaces what the range
        // overlaps, PR's `MapNative4K` refuses it; the older mappings are cut (module docs of
        // `crate::gpu`, "Conflicts").
        let (removed, kept) = trimmed(&space.records, range.start(), range.end());
        let removed_spans: usize = removed
            .iter()
            .filter_map(|start| space.records.get(start))
            .map(|r| r.spans.len())
            .sum();
        let kept_spans: usize = kept.iter().map(|(_, r)| r.spans.len()).sum();
        let after = (self.pieces_of(space) + kept_spans + 1).saturating_sub(removed_spans);
        if after > MAX_PIECES {
            return Err(KernelError::OutOfResource);
        }
        let record = Record {
            end: range.end(),
            handle: request.handle,
            offset: request.offset,
            spans: vec![(range.start(), range.end(), selected.backend)],
        };
        let Some(space) = self.spaces.get_mut(&request.space) else {
            return Err(KernelError::NotFound);
        };
        for start in removed {
            space.records.remove(&start);
        }
        space.records.extend(kept);
        space.records.insert(range.start(), record);
        self.republish(request.space, range);
        Ok(selected.selected)
    }

    /// 212 `NormalModifyMemoryKind4K`: gives the pages of a range inside one 211 mapping a new
    /// kind and returns the selected kind; the GPU applies it in order. Every refusal is
    /// `NotFound`, as PR answers them (`NormalModifyGpuMemory4K`); room for the new pieces
    /// (`OutOfResource`) and a ticket ([`next_ticket`](Self::next_ticket)) come after.
    pub fn modify_kind(
        &mut self,
        front: &dyn GpuVmFront,
        request: &ModifyRequest,
        event: ObjectId,
    ) -> Result<u8, KernelError> {
        let selected = kind::select(request.kind, request.fallback).ok_or(KernelError::NotFound)?;
        let space = self
            .spaces
            .get(&request.space)
            .ok_or(KernelError::NotFound)?;
        let (owner_start, owner) = space
            .records
            .range(..=request.first)
            .next_back()
            .filter(|&(&start, record)| {
                request.last >= request.first
                    && request.last != u64::MAX
                    && request.last < record.end
                    && request.first < record.end
                    && request.first >= start
            })
            .ok_or(KernelError::NotFound)?;
        let (start, size) = inclusive(request.first, request.last).ok_or(KernelError::NotFound)?;
        // `ModifyKind`'s alignment checks (`BadParameter`, which PR answers as `NotFound`).
        let range = PageRange::new(start, size).map_err(|_| KernelError::NotFound)?;
        let mut spans = clip(&owner.spans, *owner_start, range.start());
        spans.push((range.start(), range.end(), selected.backend));
        spans.extend(clip(&owner.spans, range.end(), owner.end));
        let after = (self.pieces_of(space) + spans.len()).saturating_sub(owner.spans.len());
        if after > MAX_PIECES {
            return Err(KernelError::OutOfResource);
        }
        self.next_ticket()?;
        let owner_start = *owner_start;
        if let Some(record) = self
            .spaces
            .get_mut(&request.space)
            .and_then(|space| space.records.get_mut(&owner_start))
        {
            record.spans = spans;
        }
        let update = VmUpdate::ModifyKind {
            range,
            kind: selected.backend,
        };
        let job = Job {
            event,
            releases: Vec::new(),
            reveal: vec![(request.space, range)],
        };
        self.order(front, &[(request.space, update)], job);
        Ok(selected.selected)
    }

    /// 213 `NormalUnmapGpuMemory4K`: takes out the 211 mapping whose range is exactly the
    /// request's (`NotFound` otherwise, PR's exact match) and sends the GPU its removal.
    /// Returns its size in bytes, PR's reply.
    pub fn unmap(
        &mut self,
        front: &dyn GpuVmFront,
        request: &RangeRequest,
        event: ObjectId,
    ) -> Result<u64, KernelError> {
        let range = self.exact(request, |space, start| {
            space.records.get(&start).map(|record| record.end)
        })?;
        self.next_ticket()?;
        if let Some(space) = self.spaces.get_mut(&request.space) {
            space.records.remove(&range.start());
        }
        let job = Job {
            event,
            releases: Vec::new(),
            reveal: vec![(request.space, range)],
        };
        self.order(front, &[(request.space, VmUpdate::Unmap { range })], job);
        Ok(range.size())
    }

    /// The range of a 213 or 215 request, if `end_at` finds an entry starting exactly there
    /// and ending exactly after its last address (PR: `mapping.address == begin && end >= begin
    /// && end - begin + 1 == mapping.size`); `NotFound` otherwise, a missing address space
    /// included.
    fn exact(
        &self,
        request: &RangeRequest,
        end_at: impl Fn(&Space, u64) -> Option<u64>,
    ) -> Result<PageRange, KernelError> {
        let space = self
            .spaces
            .get(&request.space)
            .ok_or(KernelError::NotFound)?;
        let (start, size) = inclusive(request.first, request.last).ok_or(KernelError::NotFound)?;
        match end_at(space, start) {
            Some(end) if end.checked_sub(start) == Some(size) => {
                PageRange::new(start, size).map_err(|_| KernelError::NotFound)
            }
            _ => Err(KernelError::NotFound),
        }
    }

    /// 214 `CreateSparseRegion`: reserves a GPU-only region, mapped at once. Checks in PR's
    /// order (`CreateSparseRegion`, then `ReserveNativeSparse4K`): see the module docs of
    /// [`crate::gpu`].
    pub fn create_sparse_region(
        &mut self,
        request: &SparseRequest,
    ) -> Result<RegionId, KernelError> {
        let space = self
            .spaces
            .get(&request.space)
            .ok_or(KernelError::NotFound)?;
        if !self.devices.contains_key(&request.device) {
            return Err(KernelError::NotFound);
        }
        let shift = request.page_shift;
        if shift != SMALL_PAGE_SHIFT && shift != BIG_PAGE_SHIFT {
            return Err(KernelError::InvalidSize);
        }
        let (start, size) =
            inclusive(request.first, request.last).ok_or(KernelError::InvalidSize)?;
        let mask = (1u64 << shift) - 1;
        if start & mask != 0 || size & mask != 0 {
            return Err(KernelError::InvalidSize);
        }
        if shift != SMALL_PAGE_SHIFT || request.mode > 1 || request.sync > 1 {
            return Err(KernelError::NotImplemented);
        }
        // `ReserveNativeSparse4K`'s `BadParameter`, which PR answers as `InvalidCurrentMemory`.
        let refused = KernelError::InvalidCurrentMemory;
        if u32::try_from(size / PAGE_SIZE).is_err() || !in_window(start, size) {
            return Err(refused);
        }
        let range = PageRange::new(start, size).map_err(|_| refused)?;
        let (end, end_of_region, end_of_record) =
            (range.end(), |r: &Region| r.end, |r: &Record| r.end);
        if overlaps(&space.regions, end_of_region, start, end)
            || overlaps(&space.records, end_of_record, start, end)
        {
            return Err(refused);
        }
        if self.pieces_of(space) >= MAX_PIECES {
            return Err(KernelError::OutOfResource);
        }
        let id = RegionId(self.next.region);
        let next = self
            .next
            .region
            .checked_add(1)
            .ok_or(KernelError::OutOfResource)?;
        if let Some(space) = self.spaces.get_mut(&request.space) {
            space.regions.insert(start, Region { end, id });
        }
        self.next.region = next;
        self.republish(request.space, range);
        Ok(id)
    }

    /// 215 `DestroySparseRegion`: takes out the sparse region whose range is exactly the
    /// request's (`NotFound` otherwise) and sends the GPU its destroy. 211 mappings inside it
    /// stay.
    pub fn destroy_sparse_region(
        &mut self,
        front: &dyn GpuVmFront,
        request: &RangeRequest,
        event: ObjectId,
    ) -> Result<RegionId, KernelError> {
        let range = self.exact(request, |space, start| {
            space.regions.get(&start).map(|region| region.end)
        })?;
        self.next_ticket()?;
        let region = self
            .spaces
            .get_mut(&request.space)
            .and_then(|space| space.regions.remove(&range.start()))
            .ok_or(KernelError::NotFound)?;
        let update = VmUpdate::DestroySparseRegion {
            range,
            region: region.id,
        };
        let job = Job {
            event,
            releases: Vec::new(),
            reveal: vec![(request.space, range)],
        };
        self.order(front, &[(request.space, update)], job);
        Ok(region.id)
    }

    /// Places address space `space`'s syncpoint shim at the page at `address`, at once: what
    /// nvhost-as-gpu ioctl 0xB will report (row 46; port slice 10 calls it). One per address
    /// space: a second is `InvalidState`. `NotFound` for no such space, `InvalidAddress` for an
    /// address that is not page aligned or whose page ends past the 64-bit space.
    pub fn map_syncpoint_shim(&mut self, space: AsId, address: u64) -> Result<(), KernelError> {
        let held = self.spaces.get_mut(&space).ok_or(KernelError::NotFound)?;
        if held.shim.is_some() {
            return Err(KernelError::InvalidState);
        }
        let range = PageRange::new(address, PAGE_SIZE).map_err(|_| KernelError::InvalidAddress)?;
        held.shim = Some(range.start());
        self.republish(space, range);
        Ok(())
    }

    /// The GPU reported `ticket` done: the job's removed ranges are published again (module
    /// docs, "What the table shows"), and the drain is told what to release and signal. `None`
    /// for a ticket no job waits on.
    pub fn ordered_done(&mut self, ticket: Ticket) -> Option<Finished> {
        let job = self.pending.remove(&ticket)?;
        for &(space, range) in &job.reveal {
            self.republish(space, range);
        }
        Some(Finished {
            releases: job.releases,
            event: job.event,
        })
    }

    // ----- Publishing ----------------------------------------------------------------------------

    /// Writes what the bookkeeping shows in `range` of address space `space` into its table, a
    /// layer at a time (module docs, "What the table shows"). Pages nothing covers are left as
    /// they are.
    pub(crate) fn republish(&self, space: AsId, range: PageRange) {
        self.publish(space, range.start(), range.end());
    }

    /// Writes everything the bookkeeping holds for address space `space` into its table, a layer
    /// at a time: for a table that is new (gpu 33, a restore).
    pub(crate) fn publish_all(&self, space: AsId) {
        self.publish(space, 0, u64::MAX);
    }

    /// Writes what the bookkeeping shows in `low..high` of `space` into its table. Sparse
    /// regions and 211 mappings are found by range in their maps (they do not overlap), so the
    /// time is the pieces in the range and the registrations, which may overlap each other and
    /// are walked whole (at most [`MAX_PIECES`]).
    fn publish(&self, space: AsId, low: u64, high: u64) {
        let Some(held) = self.spaces.get(&space) else {
            return;
        };
        let part_of = |start: u64, end: u64| {
            let from = start.max(low);
            let to = end.min(high);
            (from < to)
                .then(|| PageRange::new(from, to - from).ok())
                .flatten()
        };
        let mut maps: Vec<(PageRange, Target, Kind)> = Vec::new();
        for (&start, region) in in_range(&held.regions, |r| r.end, low, high) {
            if let Some(part) = part_of(start, region.end) {
                let target = Target::GpuOnly {
                    region: region.id,
                    offset: part.start() - start,
                };
                maps.push((part, target, SPARSE_KIND));
            }
        }
        for registration in self.registrations.values() {
            let own = registration.range;
            if let Some(part) = part_of(own.start(), own.end()) {
                let target = Target::Guest {
                    cpu_va: part.start(),
                };
                maps.push((part, target, SELF_MAP_KIND));
            }
        }
        for (&start, record) in in_range(&held.records, |r| r.end, low, high) {
            let Some(registration) = self.registrations.get(&record.handle) else {
                continue;
            };
            let base = registration.range.start().saturating_add(record.offset);
            for &(span_start, span_end, kind) in &record.spans {
                if let Some(part) = part_of(span_start, span_end) {
                    let cpu_va = base.saturating_add(part.start() - start);
                    maps.push((part, Target::Guest { cpu_va }, kind));
                }
            }
        }
        if let Some(shim) = held.shim
            && let Some(part) = part_of(shim, shim.saturating_add(PAGE_SIZE))
        {
            maps.push((part, Target::SyncpointShim, SELF_MAP_KIND));
        }
        for (part, target, kind) in maps {
            if let Err(error) = held.vm.map(part, target, kind) {
                // Every target is aligned and ends in range (the requests are checked), so this
                // is gsv's own mistake, not the guest's.
                tracing::error!(%error, ?target, start = part.start(), "gsv: a GPU map failed");
            }
        }
    }

    /// Every mapping of address space `space`'s table (for tests and logs).
    #[must_use]
    pub fn mappings(&self, space: AsId) -> Vec<Mapping> {
        self.spaces
            .get(&space)
            .map_or_else(Vec::new, |space| space.vm.mappings())
    }
}
