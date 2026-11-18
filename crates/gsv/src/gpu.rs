//! The gpu session (gsv 497) and its worker: notification events, graphics devices, address
//! spaces, memory registrations, mappings and sparse regions, with unmaps and destroys completed
//! in order behind the GPU's work (`docs/specs/graphics-server.md`, "GPU memory: the gpu session
//! and its worker"; port slice 7).
//!
//! * [`memory`]: the bookkeeping ([`GpuMemory`]), its checks, and what the table shows.
//! * [`kind`]: which page-table kind a request gets.
//! * [`saved`]: the bookkeeping's snapshot.
//!
//! Ported from Eden PR #4588, `eden-pr4588/src/core/hle/service/am/service/gpu_session.cpp`,
//! `gpu_session.h` and `gpu_memory_registration.h` @ e5fb3b97 (`IGpuSession`, `IGpuWorker`), and
//! Eden-NX2, `eden-nx2/src/core/hle/service/nx2_gsv.h` @ 50fa86bb (`INx2GpuSession`), both
//! GPL-3.0-or-later, Copyright 2026 Eden Emulator Project; "PR" and "NX2" below. "Row N" is a
//! row of `eden-nx2/NX2_NOTES.md` @ 50fa86bb.
//!
//! # The server
//!
//! [`gpu_server`] is the gpu family's server (decision 6). Its sessions are [`GpuSession`]s
//! (the session 497 hands out, through [`crate::root`]'s `Reply::copy_service`) and
//! [`GpuWorker`]s (22 hands them out). Every command is TIPC, as in both references
//! (`RegisterHandlersTipc`); a CMIF request, or a TIPC command neither table lists, goes to the
//! server's unknown-command policy. Each reply is TIPC: the result, then the data below.
//!
//! The state is gsv's shared state ([`crate::shared::Locked::gpu`]), one per process, which the
//! timer thread's drain also reaches; the server's family ([`GpuFamily`]) holds only the wiring:
//! the shared state and the GPU's front ([`GpuVmFront`]). Every handler takes gsv's lock, then
//! calls the kernel (events, the GPU-accessible attribute, the device-share lock) and the front
//! with it held: the lock order is gsv's, then the kernel's (`crate::shared`, "Lock order and
//! the drain"), and the front never calls back into gsv (graphics-server, "What gsv assumes":
//! no callbacks but `Completions::push`).
//!
//! # The gpu session
//!
//! | Id | Command | In | Out | Source |
//! |---|---|---|---|---|
//! | 11 | `GetNotificationRange` | nothing | `u32` 0, `u32` [`NOTIFICATION_COUNT`] - 1 | row 20; PR, **conflict** with NX2's 31: 63 chosen (spec) |
//! | 12 | `GetNotificationEvent` | `u32` index | copy handle: the index's event, made the first time | row 20; PR `GetNotificationEvent` |
//! | 21 | `RequestWorker` | `u32` index | nothing; signals the index's event | row 20; PR `RequestWorker` |
//! | 22 | `AcquireWorker` | nothing | copy handle: a worker session | row 20; PR `AcquireWorker` |
//! | 33 | `CreateAddressSpace` | `u32` 0 | `u32` address space id | PR `CreateAddressSpace` (read); NX2 answers 1 |
//! | 41 | `CreateGraphicsDevice` | `u64` name, `u64` flags, `u64` label | `u32` device id | PR `CreateGraphicsDevice` (read) |
//! | 42 | `DestroyGraphicsDevice` | `u32` id (its low byte) | nothing | PR `DestroyGraphicsDevice` (read) |
//! | 201 | user partition | `u32` (logged) | `u64` [`memory::VA_WINDOW_START`], `u64` the window's size | NX2 `GetUserPartition`; **conflict** with PR's numbering (spec: NX2's, row 44) |
//! | 202 | gsv partition | `u32` (logged) | four zero words | NX2 `GetGsvPartition` |
//! | 203 | VA range | `u32` (logged) | `u64` 1, `u64` the window's start, `u64` its size | NX2 `GetVaRange` |
//!
//! The checks are PR's: 12 and 21 refuse an index at or past [`NOTIFICATION_COUNT`] with
//! `InvalidCombination`; 21 refuses an index whose event 12 has not made with `InvalidState`,
//! and 22 refuses before any 21 with `InvalidState`; 33 refuses a non-zero word with
//! `InvalidCombination` and a ninth address space ([`memory::MAX_ADDRESS_SPACES`]) with
//! `OutOfResource`; 41 refuses non-zero flags with `NotImplemented` and a full table
//! ([`memory::DEVICE_SLOTS`]) with `OutOfResource`; 42 refuses an id with no device with
//! `NotFound`.
//!
//! # The worker
//!
//! Words are counted from the start of the message, as the spec's table and both references
//! count them: the two header words, then the data. A one-byte field is the low byte of its word
//! (PR: "single-byte TIPC fields; SDK padding is not zeroed").
//!
//! | Id | Command | Words | Out | Completes |
//! |---|---|---|---|---|
//! | 101 | `CreateGpuAccessibleMemory` | first 2–3, last 4–5, flags 6, device 7 | `u32` handle | no event |
//! | 102 | `DestroyGpuAccessibleMemory` | handle 2, slot 3 | nothing | in order, after the GPU |
//! | 211 | `NormalMapGpuMemory4K` | space 2, first 3–4, last 5–6, handle 7, offset 8–9, permission 10, cache 11, kind 12, fallback 13, mode 14, device 15, sync 16, slot 17 | `u32` selected kind | at once |
//! | 212 | `NormalModifyMemoryKind4K` | space 2, first 3–4, last 5–6, kind 7, fallback 8, sync 9, slot 10 | `u32` selected kind | in order |
//! | 213 | `NormalUnmapGpuMemory4K` | space 2, first 3–4, last 5–6, sync 7, slot 8 | `u64` bytes unmapped | in order |
//! | 214 | `CreateSparseRegion` | space 2, first 3–4, last 5–6, page shift 7, mode 8, device 9, sync 10, slot 11 | nothing | at once |
//! | 215 | `DestroySparseRegion` | as 213 | nothing | in order |
//!
//! Sources: rows 27, 40, 42 (101, 211), 68 (102, 213), 57, 80, 85 (214, 215), 149 (212), all
//! observed; the fields are PR's (`DecodeGpuMemoryRegistration`, `MapMemory`,
//! `NormalModifyGpuMemory4K`, `NormalUnmapGpuMemory4K`, `DecodeGpuSparseRegion`,
//! `DestroySparseRegion`'s comment). The reply shapes of 211 and 213 are a **conflict**: PR's
//! (the selected kind, the bytes unmapped) are used, NX2 replies with the result only (spec,
//! "verify"). 216 (`ExecuteSparseMapRequests`) is not served yet: it goes to the server's
//! unknown-command policy (see "Not built").
//!
//! **Checks, in order.** Each list is PR's order; the first failing check answers.
//!
//! - 101: flags not 0, `NotImplemented`; no such device, `NotFound`; last below first or the
//!   last address, `InvalidSize`; first not page aligned, `InvalidAddress`; size not page
//!   aligned, `InvalidSize`; no room ([`memory::MAX_PIECES`]), `OutOfResource`; a page without
//!   the GPU-accessible attribute (`svc 0x80`) or outside the address space,
//!   `InvalidCurrentMemory`; then the kernel's device-share lock, whose error passes through.
//!   (PR checks the attribute before its `nvmap.CreateHandle`; here the room check is first, so
//!   both faults at once answer `OutOfResource` and take no lock:
//!   `registration_room_is_checked_before_the_attribute`.) The lock is the last step that can
//!   fail: once it is taken, the registration is recorded ([`memory::Admission`]).
//! - 102: the slot (no event, `InvalidCombination`); no such handle, `NotFound`; no room for
//!   another job ([`memory::MAX_PENDING_JOBS`]), `Busy` (both at once answer `NotFound`:
//!   `pending_jobs_are_bounded_at_the_limit`). No event is signalled on a refusal.
//! - 211: the slot, `InvalidCombination`, nothing signalled; then, the event signalled whatever
//!   the result: no such address space, handle or device, `NotFound` (in that order);
//!   permission not 0, cache over 1, mode not 0 or sync over 1, `NotImplemented`; a kind pair
//!   [`kind::select`] refuses, `NotImplemented`; last below first or the last address,
//!   `InvalidSize`; then `InvalidCurrentMemory` for: the GPU address, size or offset not page
//!   aligned, the range not inside the registration from the offset, the range not inside NVN's
//!   window ([`memory::VA_WINDOW_START`] to [`memory::VA_WINDOW_END`], row 25); no room,
//!   `OutOfResource`.
//! - 212: sync over 1 or the slot, `InvalidCombination`, nothing signalled; then, signalled
//!   whatever the result, `NotFound` for: a kind pair refused, no such address space, a range
//!   not inside one 211 mapping, a range not page aligned; no room, `OutOfResource`; no room for
//!   a job, `Busy`.
//! - 213 and 215: sync over 1 or the slot, `InvalidCombination`, nothing signalled; then,
//!   signalled at once on a refusal: no such address space, or no 211 mapping (213) or sparse
//!   region (215) whose range is exactly the request's, `NotFound`; no room for a job, `Busy`.
//! - 214: the slot, `InvalidCombination`, nothing signalled; then, signalled whatever the
//!   result: no such address space or device, `NotFound`; a page shift not 12 or 16, last below
//!   first or the last address, or an end not aligned to the page shift, `InvalidSize`; a page
//!   shift of 16, a mode over 1 or sync over 1, `NotImplemented`; more than `u32::MAX` pages, a
//!   range outside NVN's window, or one overlapping another sparse region or a 211 mapping of
//!   the address space, `InvalidCurrentMemory`; no room, `OutOfResource`.
//!
//! **Completion.** A job that completes at once signals its slot's event during the request. An
//! ordered job (102, 212, 213, 215) changes the bookkeeping during the request, sends the GPU
//! its updates and a barrier ([`GpuVmFront::update_vm`], [`GpuVmFront::barrier`]), and replies;
//! its event is signalled only when the barrier's ticket comes back through the completion
//! queue and the timer thread's drain takes it (`crate::shared::Shared::drain`). A 102's
//! device-share lock is released then, by the drain, before the event is signalled, so a title
//! that reuses the memory once the event fires finds it unlocked (the owner's ruling of
//! 2026-11-17: the release goes through the completion queue's drain, in its recorded order).
//! Row 68: NX2 crashed when 102 and 213 completed at once.
//!
//! # The model
//!
//! See [`memory`], "What the table shows". In short: each registration is mapped at its own
//! address in every address space (decision 5; rows 30 and 68: "every address space"), a 211
//! maps part of one at a GPU address, a 214 maps GPU-only pages, and a range shows the top layer
//! the bookkeeping holds there. A 102 takes its registration out with every 211 mapping of it.
//!
//! # Conflicts
//!
//! Each is picked by the house rule of decision 8 and pinned by the test named, in
//! `gpu/tests.rs`. The spec's verification item 1 does not list the first two yet (`scopeNotes`
//! of this slice asks for it); until it does, the open question stated with each is the check
//! that settles it.
//!
//! - **A 211 over a live mapping.** PR's `MapNative4K` refuses a range that overlaps any
//!   allocation; NX2's `MapAt` replaces what it overlaps. NX2's is used, by the leg "the one a
//!   title ran on" (as `starquake_gpu::vm` picks for the table): the older 211 mappings are cut
//!   to the parts outside the new one. Open question: does the system module refuse a 211 over a
//!   live mapping, and with what result? `a_map_over_a_mapping_cuts_it`.
//! - **What a 102 does to the 211 mappings of its registration.** PR completes the destroy only
//!   once they are gone (`ReleaseState::Retained`); NX2 drops only the registration's own
//!   mapping. Here they are unmapped with it, in the same ordered job, by the leg "the larger"
//!   (the request succeeds wherever PR's eventually does). Open question: does a title ever
//!   destroy memory that still has a 211 mapping? `a_destroy_unmaps_the_registration_s_maps`.
//! - **Exact-match 212, 213 and 215** are PR's (`NormalModifyGpuMemory4K`'s "inside one mapping",
//!   `NormalUnmapGpuMemory4K`'s and `DestroySparseRegion`'s `mapping.address == begin && end -
//!   begin + 1 == mapping.size`); NX2 trims any range. Picked by the leg "readings that cite the
//!   system module" over a stub that answered everything (the spec lists these in verification
//!   1). `unmap_matches_a_mapping_exactly`, `modify_kind_stays_inside_one_mapping`.
//!
//! # Where this differs from the references
//!
//! Each row is pinned by the test named, in `gpu/tests.rs`:
//!
//! | Case | PR | NX2 | Here |
//! |---|---|---|---|
//! | The state of two gpu sessions | each 497 makes its own events, devices, address spaces and worker | one object per 497, its worker the same object | one per process, shared by every session (`two_sessions_share_the_state`) |
//! | 22 called twice | the same worker session | the same object | a new worker session each time, over the same state (`two_sessions_share_the_state`) |
//! | Handles from 12 and 22 | moved | copied | copied: the spec's **conflict**, NX2's (`notification_events_are_made_once_and_copied`) |
//! | An address space before root 434 registers it | `NotFound` from every worker job | no check | usable at once: root 434 is a stub (`a_map_is_published_and_signalled_at_once`, which sends no 434) |
//! | A request whose header words are not PR's exact ones | `InvalidCombination` | read as it comes | too few data words is `INVALID_CMIF_REQUEST`; more are read (`a_short_request_is_refused`) |
//! | A refused request's reply | the result and the reply's data words, zero | as on success | the result only, as the service framework answers every refusal (`a_refused_map_signals_and_answers_the_result_only`) |
//! | A 213 on a sparse region, a 215 on a 211 mapping | allowed: both search one list | no checks | `NotFound`: each searches its own (`unmap_matches_a_mapping_exactly`) |
//! | A 102 of a handle whose destroy is pending | `Busy` | success | `NotFound`: the handle is gone at the first 102 (`a_second_destroy_is_not_found`) |
//! | 102's completion | signals at once (`FinishRelease`) | after the GPU thread's queued work (`SignalAfterGpu`) | after the GPU's barrier, the lock released first (`a_destroy_releases_the_lock_only_when_the_gpu_is_done`) |
//! | 212, 213 and 215's completion | at once | 213 and 215 after the GPU, 212 at once | all three after the GPU's barrier (`an_unmap_s_event_fires_only_after_the_barrier`) |
//! | A 214 overlapping a registration's own mapping | no such mapping | replaces it | allowed: the region is under it (`a_sparse_region_lies_under_a_registration`) |
//! | A 215 of a region with 211 mappings inside | none lie there: `MapNative4K` refuses a 211 over a region's allocation | trims them: `SubDestroySparseRegion` calls `Nx2GpuMemoryRegistry::Unmap(first, last - first + 1)`, whose `TrimMappings` drops every mapping in the range (`eden-nx2/src/core/hle/service/nvdrv/devices/nx2_stub_device.h`) | they stay in the bookkeeping, and show again once the destroy is done (`a_sparse_region_is_destroyed_in_order`) |
//! | 216 | served | not served | not served yet (`sparse_map_requests_go_to_the_policy`) |
//! | 201 to 203 | 202 and 203 check the address space (PR's numbering) | no checks | NX2's replies, no checks (`partition_replies_are_nx2_s`) |
//!
//! # Bounds
//!
//! What a hostile title can make gsv hold is bounded by [`memory::MAX_PIECES`] per address space
//! (registrations, sparse regions and runs of one kind of 211 mappings, counted together; a
//! request that would pass it is `OutOfResource`, pinned at the edge by
//! `pieces_are_bounded_at_the_limit`), [`memory::MAX_ADDRESS_SPACES`], [`memory::DEVICE_SLOTS`],
//! [`NOTIFICATION_COUNT`] and [`memory::MAX_PENDING_JOBS`] (`Busy`). Each translation table holds
//! at most three entries per piece (a map splits at most two older entries).
//!
//! The time a publish takes is one sweep of the pieces in the span it writes, not one per range:
//! a drain publishes a job's ranges once per address space (`memory::GpuMemory`'s `publish`
//! states the cost), so a 102 that took out many 211 mappings costs the drain about the address
//! space's pieces, not their product with the mappings.
//!
//! # What is recorded and what is snapshotted
//!
//! Every request is one IPC event, recorded by the service framework. The GPU's reports come
//! back as `GpuOrderedDone` notes, recorded by the drain before it acts on them, so the table,
//! the events and the releases are a function of the recorded requests and notes in their order.
//! The bookkeeping saves and restores through [`saved`]; composing it, and the translation
//! tables, into gsv's shared snapshot is the wiring slice's (16).
//!
//! # Not built
//!
//! - 216 `ExecuteSparseMapRequests` (PR only, read): a batch of 40-byte entries in a buffer that
//!   map memory into sparse regions. The model it needs is here (a 211 inside a sparse region);
//!   its decoding is not.
//! - Root 434 (`RegisterGpuAddressSpace`), which PR requires before a worker job may name an
//!   address space.

pub mod kind;
pub mod memory;
pub mod saved;

#[cfg(test)]
mod tests;

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use starquake_gpu::interface::{Ticket, VmUpdate};
use starquake_gpu::vm::GpuVm;
use starquake_kernel::id::ObjectId;
use starquake_kernel::result::KernelError;
use starquake_services::commands;
use starquake_services::framework::{
    Context, Family, Fault, Interfaces, Outcome, Reply, Request, Server, SnapshotError,
    UnknownPolicy,
};

use crate::shared::Shared;
pub use memory::GpuMemory;
use memory::{
    Device, MapRequest, ModifyRequest, RangeRequest, RegisterRequest, SparseRequest, VA_WINDOW_END,
    VA_WINDOW_START,
};

/// The name of the gpu family's server, for logs. Starquake's own: the server is reached only
/// through root 497, never by name.
pub const GPU_NAME: &str = "gsv:gpu";

/// How many notification events a gpu session has ([`memory::NOTIFICATION_COUNT`]).
pub const NOTIFICATION_COUNT: u32 = memory::NOTIFICATION_COUNT;

/// The gpu session's and the worker's command ids (spec, "GPU memory: the gpu session and its
/// worker"; PR's `IGpuSession` and `IGpuWorker` handler tables).
pub mod command {
    /// `GetNotificationRange`.
    pub const GET_NOTIFICATION_RANGE: u32 = 11;
    /// `GetNotificationEvent`.
    pub const GET_NOTIFICATION_EVENT: u32 = 12;
    /// `RequestWorker`.
    pub const REQUEST_WORKER: u32 = 21;
    /// `AcquireWorker`.
    pub const ACQUIRE_WORKER: u32 = 22;
    /// `CreateAddressSpace`.
    pub const CREATE_ADDRESS_SPACE: u32 = 33;
    /// `CreateGraphicsDevice`.
    pub const CREATE_GRAPHICS_DEVICE: u32 = 41;
    /// `DestroyGraphicsDevice`.
    pub const DESTROY_GRAPHICS_DEVICE: u32 = 42;
    /// The user partition (NX2's reading of 201).
    pub const GET_USER_PARTITION: u32 = 201;
    /// The gsv partition (NX2's reading of 202).
    pub const GET_GSV_PARTITION: u32 = 202;
    /// The VA range (NX2's reading of 203).
    pub const GET_VA_RANGE: u32 = 203;
    /// `CreateGpuAccessibleMemory`.
    pub const CREATE_GPU_ACCESSIBLE_MEMORY: u32 = 101;
    /// `DestroyGpuAccessibleMemory`.
    pub const DESTROY_GPU_ACCESSIBLE_MEMORY: u32 = 102;
    /// `NormalMapGpuMemory4K`.
    pub const NORMAL_MAP_GPU_MEMORY_4K: u32 = 211;
    /// `NormalModifyMemoryKind4K`.
    pub const NORMAL_MODIFY_MEMORY_KIND_4K: u32 = 212;
    /// `NormalUnmapGpuMemory4K`.
    pub const NORMAL_UNMAP_GPU_MEMORY_4K: u32 = 213;
    /// `CreateSparseRegion`.
    pub const CREATE_SPARSE_REGION: u32 = 214;
    /// `DestroySparseRegion`.
    pub const DESTROY_SPARSE_REGION: u32 = 215;
    /// `ExecuteSparseMapRequests` (not served yet).
    pub const EXECUTE_SPARSE_MAP_REQUESTS: u32 = 216;
}

use command::{
    ACQUIRE_WORKER, CREATE_ADDRESS_SPACE, CREATE_GPU_ACCESSIBLE_MEMORY, CREATE_GRAPHICS_DEVICE,
    CREATE_SPARSE_REGION, DESTROY_GPU_ACCESSIBLE_MEMORY, DESTROY_GRAPHICS_DEVICE,
    DESTROY_SPARSE_REGION, GET_GSV_PARTITION, GET_NOTIFICATION_EVENT, GET_NOTIFICATION_RANGE,
    GET_USER_PARTITION, GET_VA_RANGE, NORMAL_MAP_GPU_MEMORY_4K, NORMAL_MODIFY_MEMORY_KIND_4K,
    NORMAL_UNMAP_GPU_MEMORY_4K, REQUEST_WORKER,
};

/// NX2's `GetVaRange` kind word (`constexpr u64 kind = 1`).
pub const VA_RANGE_KIND: u64 = 1;

/// The most a sync byte may be (PR: `sync <= 1` in every job's checks).
pub const MAX_SYNC: u8 = 1;

/// A GPU address space, as gsv numbers it (gpu 33). Graphics-server's `AsId`: a shared type of
/// the spec's interface block, kept here until `crates/gpu` defines it with the rest of
/// `GpuFront` (gpu-protocol's front slice).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct AsId(pub u32);

/// What gsv asks of the GPU's front for its address tables: the part of the spec's `GpuFront`
/// this slice uses, with the spec's names and signatures (graphics-server, "What this spec
/// offers and assumes of the GPU front/back"). `crates/gpu` does not define `GpuFront` yet; when
/// it does, it implements this, or this becomes it.
///
/// Called with gsv's shared lock held: an implementation must not block on the GPU or call into
/// gsv (it reports only through `Completions::push`).
pub trait GpuVmFront: Send + Sync {
    /// A new address space with its table.
    fn create_address_space(&self, space: AsId, vm: Arc<GpuVm>);

    /// An ordered update of `space`'s table, applied once no work queued before it can use the
    /// old mapping; `done` is reported back when it is applied, if given.
    fn update_vm(&self, space: AsId, update: VmUpdate, done: Option<Ticket>);

    /// Reports `done` once all work and updates queued before it are finished.
    fn barrier(&self, done: Ticket);
}

/// The gpu server's family: its wiring, the state it reaches and the GPU's front. It holds no
/// guest state of its own (module docs, "The server"), so its snapshot is empty.
pub struct GpuFamily {
    shared: Arc<Shared>,
    front: Arc<dyn GpuVmFront>,
}

impl Family for GpuFamily {
    /// Nothing: the state is gsv's shared state, saved with it ([`saved`]).
    fn save_state(&self) -> Result<Vec<u8>, SnapshotError> {
        Ok(Vec::new())
    }

    /// Accepts only what [`save_state`](Family::save_state) wrote.
    fn load_state(&mut self, bytes: &[u8]) -> Result<(), SnapshotError> {
        if bytes.is_empty() {
            Ok(())
        } else {
            Err(SnapshotError::FamilyWiring(
                "the gpu server's family holds no state of its own",
            ))
        }
    }
}

/// Takes gsv's lock and runs `f` on the bookkeeping, with the request's context and the front.
fn with_gpu<R>(
    cx: &mut Context<'_>,
    f: impl FnOnce(&mut Context<'_>, &mut GpuMemory, &dyn GpuVmFront) -> Result<R, Fault>,
) -> Result<R, Fault> {
    let family = cx.family::<GpuFamily>()?;
    let shared = Arc::clone(&family.shared);
    let front = Arc::clone(&family.front);
    let mut locked = shared.lock();
    f(cx, locked.gpu(), &*front)
}

/// The low byte of a word: a one-byte field (module docs, "The worker").
pub(crate) const fn low_byte(word: u32) -> u8 {
    word.to_le_bytes()[0]
}

/// A notification index, checked against [`NOTIFICATION_COUNT`] (PR: `packet[2] <
/// m_devices->events.size()`, else `InvalidCombination`).
fn notification_index(index: u32) -> Result<u32, Fault> {
    if index < NOTIFICATION_COUNT {
        Ok(index)
    } else {
        Err(KernelError::InvalidCombination.into())
    }
}

/// The session 497 hands out.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct GpuSession;

/// The worker session 22 hands out.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct GpuWorker;

commands! {
    impl GpuSession {}
    tipc {
        GET_NOTIFICATION_RANGE => get_notification_range,
        GET_NOTIFICATION_EVENT => get_notification_event,
        REQUEST_WORKER => request_worker,
        ACQUIRE_WORKER => acquire_worker,
        CREATE_ADDRESS_SPACE => create_address_space,
        CREATE_GRAPHICS_DEVICE => create_graphics_device,
        DESTROY_GRAPHICS_DEVICE => destroy_graphics_device,
        GET_USER_PARTITION => get_user_partition,
        GET_GSV_PARTITION => get_gsv_partition,
        GET_VA_RANGE => get_va_range,
    }
}

commands! {
    impl GpuWorker {}
    tipc {
        CREATE_GPU_ACCESSIBLE_MEMORY => create_gpu_accessible_memory,
        DESTROY_GPU_ACCESSIBLE_MEMORY => destroy_gpu_accessible_memory,
        NORMAL_MAP_GPU_MEMORY_4K => normal_map_gpu_memory_4k,
        NORMAL_MODIFY_MEMORY_KIND_4K => normal_modify_memory_kind_4k,
        NORMAL_UNMAP_GPU_MEMORY_4K => normal_unmap_gpu_memory_4k,
        CREATE_SPARSE_REGION => create_sparse_region,
        DESTROY_SPARSE_REGION => destroy_sparse_region,
    }
}

impl GpuSession {
    /// Command 11: out: `u32` 0, `u32` the last index.
    fn get_notification_range(
        _: &mut Self,
        _cx: &mut Context<'_>,
        _req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        reply
            .payload()
            .write_u32(0)
            .write_u32(NOTIFICATION_COUNT - 1);
        reply.ok()
    }

    /// Command 12: in: `u32` index. out: copy handle, the index's event, made the first time.
    fn get_notification_event(
        _: &mut Self,
        cx: &mut Context<'_>,
        req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        let index = notification_index(req.payload().read_u32()?)?;
        let event = with_gpu(cx, |cx, gpu, _| {
            if let Some(event) = gpu.notification_event(index) {
                return Ok(event);
            }
            let event = cx.create_event()?;
            gpu.set_notification_event(index, event);
            Ok(event)
        })?;
        reply.copy_handle(event);
        reply.ok()
    }

    /// Command 21: in: `u32` index. Makes the worker and signals the index's event.
    fn request_worker(
        _: &mut Self,
        cx: &mut Context<'_>,
        req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        let index = notification_index(req.payload().read_u32()?)?;
        with_gpu(cx, |cx, gpu, _| {
            let event = gpu
                .notification_event(index)
                .ok_or(KernelError::InvalidState)?;
            gpu.make_worker();
            cx.signal_event(event)
        })?;
        reply.ok()
    }

    /// Command 22: out: copy handle, a worker session.
    fn acquire_worker(
        _: &mut Self,
        cx: &mut Context<'_>,
        _req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        let ready = with_gpu(cx, |_, gpu, _| Ok(gpu.has_worker()))?;
        if !ready {
            return Err(KernelError::InvalidState.into());
        }
        reply.copy_session(Box::new(GpuWorker));
        reply.ok()
    }

    /// Command 33: in: `u32` 0. out: `u32` address space id.
    fn create_address_space(
        _: &mut Self,
        cx: &mut Context<'_>,
        req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        if req.payload().read_u32()? != 0 {
            return Err(KernelError::InvalidCombination.into());
        }
        let id = with_gpu(cx, |_, gpu, front| Ok(gpu.create_address_space(front)?))?;
        reply.payload().write_u32(id.0);
        reply.ok()
    }

    /// Command 41: in: `u64` name, `u64` flags, `u64` label. out: `u32` device id.
    fn create_graphics_device(
        _: &mut Self,
        cx: &mut Context<'_>,
        req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        let mut words = req.payload();
        let name = words.read_u64()?;
        let flags = words.read_u64()?;
        let label = words.read_u64()?;
        if flags != 0 {
            return Err(KernelError::NotImplemented.into());
        }
        let id = with_gpu(cx, |_, gpu, _| {
            Ok(gpu.create_device(Device { name, label })?)
        })?;
        reply.payload().write_u32(u32::from(id));
        reply.ok()
    }

    /// Command 42: in: `u32` id, of which the low byte names the device (PR: `static_cast<u8>(packet[2])`).
    fn destroy_graphics_device(
        _: &mut Self,
        cx: &mut Context<'_>,
        req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        let id = low_byte(req.payload().read_u32()?);
        with_gpu(cx, |_, gpu, _| Ok(gpu.destroy_device(id)?))?;
        reply.ok()
    }

    /// 201 (NX2 `GetUserPartition`). in: `u32`, logged. out: `u64` start, `u64` size of NVN's
    /// window.
    fn get_user_partition(
        _: &mut Self,
        _cx: &mut Context<'_>,
        req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        let word = req.payload().read_u32()?;
        tracing::debug!(word, "gsv:gpu user partition");
        reply
            .payload()
            .write_u64(VA_WINDOW_START)
            .write_u64(VA_WINDOW_END - VA_WINDOW_START);
        reply.ok()
    }

    /// 202 (NX2 `GetGsvPartition`). in: `u32`, logged. out: four zero words, an empty partition.
    fn get_gsv_partition(
        _: &mut Self,
        _cx: &mut Context<'_>,
        req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        let word = req.payload().read_u32()?;
        tracing::debug!(word, "gsv:gpu gsv partition");
        reply.payload().write_u64(0).write_u64(0);
        reply.ok()
    }

    /// 203 (NX2 `GetVaRange`). in: `u32`, logged. out: `u64` kind 1, `u64` start, `u64` size.
    fn get_va_range(
        _: &mut Self,
        _cx: &mut Context<'_>,
        req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        let word = req.payload().read_u32()?;
        tracing::debug!(word, "gsv:gpu VA range");
        reply
            .payload()
            .write_u64(VA_RANGE_KIND)
            .write_u64(VA_WINDOW_START)
            .write_u64(VA_WINDOW_END - VA_WINDOW_START);
        reply.ok()
    }
}

/// Signals `event` after a job refused once its slot was checked (PR signals then), and passes
/// the refusal on.
fn refuse_and_signal<T>(
    cx: &mut Context<'_>,
    event: ObjectId,
    error: KernelError,
) -> Result<T, Fault> {
    cx.signal_event(event)?;
    Err(error.into())
}

/// A range request's fields after the two header words: space, first, last, then the sync word
/// and the slot word (213, 215).
fn read_range(req: &Request<'_>) -> Result<(RangeRequest, u8, u32), Fault> {
    let mut words = req.payload();
    let request = RangeRequest {
        space: AsId(words.read_u32()?),
        first: words.read_u64()?,
        last: words.read_u64()?,
    };
    let sync = low_byte(words.read_u32()?);
    let slot = words.read_u32()?;
    Ok((request, sync, slot))
}

/// PR's first check of 212, 213 and 215: the sync byte and the slot, both `InvalidCombination`.
fn ordered_job_event(gpu: &GpuMemory, sync: u8, slot: u32) -> Result<ObjectId, Fault> {
    if sync > MAX_SYNC {
        return Err(KernelError::InvalidCombination.into());
    }
    Ok(gpu.job_event(slot)?)
}

impl GpuWorker {
    /// Command 101: Words: first 2–3, last 4–5, flags 6, device 7. out: `u32` handle.
    fn create_gpu_accessible_memory(
        _: &mut Self,
        cx: &mut Context<'_>,
        req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        let mut words = req.payload();
        let request = RegisterRequest {
            first: words.read_u64()?,
            last: words.read_u64()?,
            flags: low_byte(words.read_u32()?),
            device: low_byte(words.read_u32()?),
        };
        let handle = with_gpu(cx, |cx, gpu, _| {
            let admitted = gpu.check_registration(&request)?;
            let range = admitted.range();
            if !cx.is_gpu_accessible(range.start(), range.size())? {
                return Err(KernelError::InvalidCurrentMemory.into());
            }
            // The last fallible step: once the lock is taken, the registration is recorded.
            let token = cx.lock_device_share(range.start(), range.size())?;
            Ok(admitted.register(token))
        })?;
        reply.payload().write_u32(handle);
        reply.ok()
    }

    /// Command 102: Words: handle 2, slot 3. Completes in order.
    fn destroy_gpu_accessible_memory(
        _: &mut Self,
        cx: &mut Context<'_>,
        req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        let mut words = req.payload();
        let handle = words.read_u32()?;
        let slot = words.read_u32()?;
        with_gpu(cx, |_, gpu, front| {
            let event = gpu.job_event(slot)?;
            Ok(gpu.destroy_memory(front, handle, event)?)
        })?;
        reply.ok()
    }

    /// Command 211: Words: see the module docs. out: `u32` selected kind. Completes at once.
    fn normal_map_gpu_memory_4k(
        _: &mut Self,
        cx: &mut Context<'_>,
        req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        let mut words = req.payload();
        let space = AsId(words.read_u32()?);
        let first = words.read_u64()?;
        let last = words.read_u64()?;
        let handle = words.read_u32()?;
        let offset = words.read_u64()?;
        let mut byte = || words.read_u32().map(low_byte);
        let request = MapRequest {
            space,
            first,
            last,
            handle,
            offset,
            permission: byte()?,
            cache: byte()?,
            kind: byte()?,
            fallback: byte()?,
            mode: byte()?,
            device: byte()?,
            sync: byte()?,
        };
        let slot = words.read_u32()?;
        let selected = with_gpu(cx, |cx, gpu, _| {
            let event = gpu.job_event(slot)?;
            let mapped = gpu.map(&request);
            cx.signal_event(event)?;
            Ok(mapped?)
        })?;
        reply.payload().write_u32(u32::from(selected));
        reply.ok()
    }

    /// Command 212: Words: space 2, first 3–4, last 5–6, kind 7, fallback 8, sync 9, slot 10. out:
    /// `u32` selected kind. Completes in order.
    fn normal_modify_memory_kind_4k(
        _: &mut Self,
        cx: &mut Context<'_>,
        req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        let mut words = req.payload();
        let request = ModifyRequest {
            space: AsId(words.read_u32()?),
            first: words.read_u64()?,
            last: words.read_u64()?,
            kind: low_byte(words.read_u32()?),
            fallback: low_byte(words.read_u32()?),
        };
        let sync = low_byte(words.read_u32()?);
        let slot = words.read_u32()?;
        let selected = with_gpu(cx, |cx, gpu, front| {
            let event = ordered_job_event(gpu, sync, slot)?;
            match gpu.modify_kind(front, &request, event) {
                Ok(selected) => Ok(selected),
                Err(error) => refuse_and_signal(cx, event, error),
            }
        })?;
        reply.payload().write_u32(u32::from(selected));
        reply.ok()
    }

    /// Command 213: Words: space 2, first 3–4, last 5–6, sync 7, slot 8. out: `u64` bytes unmapped.
    /// Completes in order.
    fn normal_unmap_gpu_memory_4k(
        _: &mut Self,
        cx: &mut Context<'_>,
        req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        let (request, sync, slot) = read_range(req)?;
        let bytes = with_gpu(cx, |cx, gpu, front| {
            let event = ordered_job_event(gpu, sync, slot)?;
            match gpu.unmap(front, &request, event) {
                Ok(bytes) => Ok(bytes),
                Err(error) => refuse_and_signal(cx, event, error),
            }
        })?;
        reply.payload().write_u64(bytes);
        reply.ok()
    }

    /// Command 214: Words: space 2, first 3–4, last 5–6, page shift 7, mode 8, device 9, sync 10, slot
    /// 11; completes at once.
    fn create_sparse_region(
        _: &mut Self,
        cx: &mut Context<'_>,
        req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        let mut words = req.payload();
        let space = AsId(words.read_u32()?);
        let first = words.read_u64()?;
        let last = words.read_u64()?;
        let mut byte = || words.read_u32().map(low_byte);
        let request = SparseRequest {
            space,
            first,
            last,
            page_shift: byte()?,
            mode: byte()?,
            device: byte()?,
            sync: byte()?,
        };
        let slot = words.read_u32()?;
        with_gpu(cx, |cx, gpu, _| {
            let event = gpu.job_event(slot)?;
            let created = gpu.create_sparse_region(&request);
            cx.signal_event(event)?;
            created?;
            Ok(())
        })?;
        reply.ok()
    }

    /// Command 215: Words: as 213. Completes in order.
    fn destroy_sparse_region(
        _: &mut Self,
        cx: &mut Context<'_>,
        req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        let (request, sync, slot) = read_range(req)?;
        with_gpu(cx, |cx, gpu, front| {
            let event = ordered_job_event(gpu, sync, slot)?;
            match gpu.destroy_sparse_region(front, &request, event) {
                Ok(_) => Ok(()),
                Err(error) => refuse_and_signal(cx, event, error),
            }
        })?;
        reply.ok()
    }
}

/// The interfaces a gpu server's sessions can be.
#[must_use]
pub fn gpu_interfaces() -> Interfaces {
    Interfaces::new().with::<GpuSession>().with::<GpuWorker>()
}

/// The gpu family's server: every session a [`GpuSession`] (22 hands out [`GpuWorker`]s on
/// it), over `shared`'s bookkeeping and `front`.
#[must_use]
pub fn gpu_server(
    shared: Arc<Shared>,
    front: Arc<dyn GpuVmFront>,
    policy: UnknownPolicy,
) -> Server {
    Server::new(GPU_NAME, policy)
        .with_root(|| Box::new(GpuSession))
        .with_interfaces(gpu_interfaces())
        .with_family(GpuFamily { shared, front })
}
