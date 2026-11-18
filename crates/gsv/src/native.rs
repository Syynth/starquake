//! Native memory and native images: the root commands that make, bind and destroy them, size
//! them, and mark window alias image pools (`docs/specs/graphics-server.md`, "Native memory and
//! images"; port slice 8).
//!
//! * [`layout`]: the size of a native image (435, 1510, and 429's check).
//! * [`saved`]: the tables' snapshot.
//!
//! Ported from Eden PR #4588,
//! `eden-pr4588/src/core/hle/service/am/service/graphics_supervisor.cpp` @ e5fb3b97
//! (`IGraphicsSupervisor::CreateNativeMemory`, `BindNativeMemoryBacking`, `BindNativeMemory`,
//! `ReleaseNativeMemory`, `DestroyNativeMemory`, `BindNativeGpuMemory`, `CreateNativeImage`,
//! `DestroyNativeImage`, `ConfigureNativeWindowImagePool`, `CalculateImageRequirements`, and
//! `NativeImageRegistry`), GPL-3.0-or-later, Copyright 2026 Eden Emulator Project; "PR" below.
//! Eden-NX2's handlers (`INx2GraphicsSystemSession` in
//! `eden-nx2/src/core/hle/service/nx2_gsv.h` @ 50fa86bb, "NX2") check nothing and are read for
//! the divergences below. "Row N" is a row of `eden-nx2/NX2_NOTES.md` @ 50fa86bb.
//!
//! # Where the tables live
//!
//! The tables ([`NativeObjects`]) are in gsv's shared state ([`crate::shared::Locked::native`]),
//! one per process, because vi2's presents and the video pools read them too (spec: "small
//! tables in gsv's root, shared by vi2 (presents) and video (pools)"; "Concurrency inside gsv").
//! The root answers these commands from them when it is built with gsv's shared state
//! (`crate::root::root_server_with_page`, as a run builds it); a root built without answers
//! them with its logged stubs, as before this slice. Each handler takes gsv's lock, then calls
//! the kernel with it held (the lock order of `crate::shared`).
//!
//! # The commands
//!
//! Every command is TIPC. Words are counted from the start of the message, as PR counts them:
//! the two header words, then the data. A one-byte field is the low byte of its word.
//!
//! | Id | Command | Words | Out | Source |
//! |---|---|---|---|---|
//! | 111 | `CreateNativeMemory` | mode 2 (low byte), size 3–4 | `u32` id | row 47; PR (`0x7f, 3`); NX2 "u8 mode at word 2, u64 size" |
//! | 112 | `DestroyNativeMemory` | id 2 | nothing | row 47; PR (`0x80, 1`) |
//! | 114 | `BindNativeMemory` | id 2, address 3–4, size 5–6, flags 7–8, offset 9–10 | nothing | row 47; PR (`0x82, 9`); NX2 "u32 id, u64 address, u64 size, u64 flags, u64 offset" |
//! | 423 | `BindNativeGpuMemory` | id 2, address 3–4, size 5–6, parent 7, offset 8–9, two `u64` zeros 10–13 | nothing | rows 47, 48; PR (`0x1b7, 12`) |
//! | 429 | `CreateNativeImage` | preference 2–17, count 18, memory 19 | `u32` id | rows 33, 47; PR (`0x1bd, 18`) |
//! | 430 | `DestroyNativeImage` | id 2 | nothing | rows 33, 47; PR (`0x1be, 1`) |
//! | 314 | `DestroyNativeWindowAliasImagePool` | image 2 | nothing | rows 33, 47; PR (`0x14a, 1`) |
//! | 315 | `CreateNativeWindowAliasImagePool` | image 2, count 3 | nothing | rows 33, 47; PR (`0x14b, 2`) |
//! | 435 | `CalculateNativeImageSize` | preference 2–17, count 18 | `u64` size | rows 33, 47; PR (`0x1c3, 17`) |
//! | 1510 | `CalculateImageRequirements` | preference 2–17, count 18 | `u64` size, `u64` alignment | rows 33, 47; PR (`0x5f6, 17`) |
//!
//! 115 (`ConfigureNativeMemory`) is the root's own and keeps the mode it records
//! (`crate::root`).
//!
//! **Checks, in order.** Each list is PR's order; the first failing check answers. Every word a
//! command reads is read first, so a short request (`INVALID_CMIF_REQUEST`) changes nothing.
//!
//! - 111: a mode over [`MAX_MODE`], `InvalidCombination`; a size of 0 or over
//!   [`MAX_MEMORY_SIZE`], [`GSV_INVALID_SIZE`]; [`MAX_MEMORIES`] memories, or no id left,
//!   `OutOfResource`.
//! - 112: no such memory, `NotFound`. A memory an image uses is retired: it answers success, is
//!   gone for every command, and keeps its lock until its last image is destroyed (PR
//!   `retired_memories`, "Keep the backing locked until display/image references have
//!   retired"). Otherwise it is removed and its lock released; a release the kernel refuses
//!   answers the kernel's error, the memory gone all the same (PR: `m_native_memories.erase(it)`
//!   before `R_TRY(... UnlockForDeviceAddressSpace ...)`).
//! - 114 (PR `BindNativeMemoryBacking`): no such memory, `NotFound`; flags not
//!   [`BIND_PROCESS_MEMORY`] or [`BIND_GPU_ACCESSIBLE`], or an offset not 0, `NotImplemented`;
//!   bound already, `InvalidState`; a size of 0, a size not the memory's, an address of 0 or a
//!   range that wraps, `InvalidSize`; an address not page aligned, `InvalidAddress`; with
//!   [`BIND_GPU_ACCESSIBLE`], a page without the GPU-accessible attribute or outside the address
//!   space, `InvalidCurrentMemory`; then the kernel's device-share lock, whose error passes
//!   through (`InvalidCurrentMemory` for a range outside the address space, so PR's `Contains`
//!   check answers through it). The lock is the last step that can fail: once it is taken, the
//!   binding is recorded ([`BindAdmission`]).
//! - 423: either `u64` at words 10–13 not zero, `InvalidCombination`; then `NotFound` unless the
//!   parent is a gpu worker registration (101) and the range is inside it at the offset: offset
//!   at most its size, size at most its size less the offset, and address its start plus the
//!   offset; then 114's checks with [`BIND_GPU_ACCESSIBLE`] and offset 0.
//! - 429: the preference's kind not [`layout::KIND_NATIVE`], `InvalidCombination`; a preference
//!   or count [`layout::requirements`] refuses, `NotImplemented`; no such memory, or one not
//!   bound, `NotFound`; a memory smaller than the size, `InvalidSize`; [`MAX_IMAGES`] images, or
//!   no id left, `OutOfResource`.
//! - 430: no such image, `NotFound`; an image in a window pool, `Busy`. Destroying the last image
//!   of a retired memory removes the memory and releases its lock (PR `DrainRetiredMemories`),
//!   with 112's answer to a refused release.
//! - 314 and 315: no such image, `NotFound`; 315 with a count not [`WINDOW_POOL_COUNT`],
//!   `NotImplemented`; an image already in (315) or not in (314) a window pool, `InvalidState`.
//! - 435 and 1510: the preference's kind not [`layout::KIND_NATIVE`] (435) or
//!   [`layout::KIND_BLOCK`] (1510), `InvalidCombination`; a count of 0 with a preference
//!   [`layout::requirements`] accepts, [`GSV_INVALID_SIZE`]; any other refusal of it,
//!   `NotImplemented`.
//!
//! # Where this differs from the references
//!
//! NX2 checks nothing: every command succeeds, an unknown id included. Where PR checks, the
//! table says what is done here; each row is pinned by the test named, in `native/tests.rs`:
//!
//! | Case | PR | Here |
//! |---|---|---|
//! | Any command before the bootstrap is complete, the image formats are set up (1502) or the GPU context is ready (402) | `InvalidState` | accepted and logged, as the root's other commands (decision 7): `commands_before_the_bootstrap_are_accepted`; one warning naming the command before, none once ready, in `crates/gsv/tests/native_unready_log.rs` |
//! | A request whose header words are not PR's exact ones | `InvalidCombination` | too few data words is `INVALID_CMIF_REQUEST`; more are read, as the root's other commands (`a_short_request_changes_nothing`) |
//! | A refused request's reply | the result, then the reply's data words, zero | the result only, as the service framework answers every refusal (`each_create_check_refuses_on_its_own`) |
//! | Native memory ids | nvmap handle ids, a step of 4 from 4 (`eden-pr4588/src/core/hle/service/nvdrv/core/nvmap.h`, `HandleIdIncrement`) | **conflict**: NX2's, 1, 2, 3 and on (`next_memory{1}`), by decision 8's "the one a title ran on" (`ids_count_from_one`) |
//! | No id left | memories: nvmap's counter wraps; images: `m_next_native_image_id != 0`, `OutOfResource` | both as PR's images: the last id is `u32::MAX`, then `OutOfResource` (`ids_run_out_after_the_last`) |
//! | 423's parent | any nvmap handle of the process that is bound: a gpu worker registration, a native memory, or an nvsrv nvmap allocation | looked up among gpu worker registration handles only (spec: "bind to GPU-accessible memory"). Native memory ids and registration handles are separate counters that both start at [`FIRST_ID`], so the number names whatever registration has that handle: a native memory id that no registration has is `NotFound`, and one equal to a live registration's handle names that registration (`a_gpu_bind_s_parent_is_a_registration`) |
//! | A 102 of a registration a 423 bound part of | held (`ReleaseState::Retained`) until the native memory is released | goes ahead; the native memory holds its own device-share lock on its range, released by its 112 (`a_parent_destroy_leaves_the_bind_s_own_lock`) |
//! | 114 on I/O memory (`is_io`) | `NotImplemented` | not ported: the kernel's lock refuses a range whose state cannot be device mapped |
//! | 114's nvmap allocation failing | `OutOfResource` | not ported: there is no nvmap allocation here |
//! | 112 of a memory whose nvmap handle has other duplicates (`handle->dupes == 1`) | `Busy` | not ported: there are no nvmap handles to duplicate here; only an image's use retires a memory (`a_memory_in_use_is_retired_until_its_last_image_goes`) |
//! | 429's duplicate of the memory's nvmap handle failing (`memory->Duplicate(true)`) | `OutOfResource` | not ported: there is no nvmap handle to duplicate; only the image count and the ids refuse with `OutOfResource` (`images_are_bounded_at_the_limit`, `ids_run_out_after_the_last`) |
//! | The root object going (its session closed) | `~IGraphicsSupervisor` releases every native memory (`ReleaseNativeMemory`, or the nvmap free and unlock when that fails) and `~NativeImageRegistry` releases its images' memories and drains the retired ones, unlocking each range | the tables live in gsv's shared state for the whole run: closing the root session releases nothing, every memory, image and lock stays (`closing_the_root_session_releases_nothing`) |
//!
//! # Bounds
//!
//! What a hostile title can make gsv hold: [`MAX_MEMORIES`] live memories and [`MAX_IMAGES`]
//! images (PR's limits, pinned at the edge by `memories_are_bounded_at_the_limit` and
//! `images_are_bounded_at_the_limit`). A retired memory is not counted among the live ones, as
//! in PR, but each has at least one image, so there are at most [`MAX_IMAGES`] of them. A request
//! walks at most these tables once (counting the live memories, or looking for an image of a
//! memory), so its work is bounded by the limits, not by anything a title grows past them.
//!
//! # What is recorded and what is snapshotted
//!
//! Every request is one IPC event, recorded by the service framework; the spec's "What is
//! recorded" has no note for these commands, and nothing but a request changes the tables. The
//! tables save and restore through [`saved`]; composing that into gsv's whole snapshot is the
//! wiring slice's (16), as for the GPU memory.

pub mod layout;
pub mod pools;
pub mod saved;

#[cfg(test)]
mod tests;

use std::collections::BTreeMap;

use starquake_kernel::memory::DeviceShareToken;
use starquake_kernel::result::{KernelError, MODULE_BITS, ResultCode};
use starquake_memory::PAGE_SIZE;
use starquake_services::framework::{Context, Fault, Outcome, Reader, Reply, Request};

use crate::gpu::low_byte;
use crate::gpu::memory::Registration;
use crate::shared::Shared;
use layout::{KIND_BLOCK, KIND_NATIVE, PREFERENCE_WORDS, Preference, PreferenceError, WORD_KIND};

/// The most live native memories (PR `CreateNativeMemory`: `m_native_memories.size() < 1024`).
pub const MAX_MEMORIES: usize = 1024;
/// The most native images (PR `CreateNativeImage`: `m_image_registry->images.size() < 1024`).
pub const MAX_IMAGES: usize = 1024;
/// The largest mode byte 111 takes (PR: `(packet[2] & 0xff) <= 3`; "flags (low byte) 0..3").
pub const MAX_MODE: u8 = 3;
/// The largest native memory (PR `CreateNativeMemory`: `size <= (1ULL << 48)`).
pub const MAX_MEMORY_SIZE: u64 = 1 << 48;
/// The first id of each table, counted up from it (NX2 `next_memory{1}`, `next_image{1}`; PR
/// `m_next_native_image_id{1}`).
pub const FIRST_ID: u32 = 1;

/// 114's flags: bind any memory of the process (PR `BindNativeMemoryBacking`: `flags == 0`).
pub const BIND_PROCESS_MEMORY: u64 = 0;
/// 114's flags: bind memory every page of which has the GPU-accessible attribute (PR: `flags ==
/// 2`, then the `KMemoryAttribute::GpuAccessible` walk); 423 binds this way
/// (`BindNativeMemoryBacking(packet[2], address, size, 2, 0)`).
pub const BIND_GPU_ACCESSIBLE: u64 = 2;

/// The count 315 requires (PR `ConfigureNativeWindowImagePool`: `!create || packet[3] == 1`).
pub const WINDOW_POOL_COUNT: u32 = 1;

/// The graphics server's result module (PR: `static_cast<ErrorModule>(0xf3)`).
pub const GSV_MODULE: u32 = 0xF3;
/// The description of [`GSV_INVALID_SIZE`] (PR: `Result{static_cast<ErrorModule>(0xf3), 0x80}`).
pub const GSV_INVALID_SIZE_DESCRIPTION: u32 = 0x80;
/// gsv result `0x100f3`: 111 with a size of 0 ("size 0 is gsv result 0x100f3") or over
/// [`MAX_MEMORY_SIZE`], and 435 or 1510 with a count of 0 ("a valid layout with count 0 is gsv
/// result 0x100f3"). PR reads both from the system module (`gsv1 0x328d0/0x32ca8`, `gsv1
/// 0x4cf44`).
pub const GSV_INVALID_SIZE: ResultCode =
    ResultCode::from_raw(GSV_MODULE | (GSV_INVALID_SIZE_DESCRIPTION << MODULE_BITS));

/// The command ids of this module, with the root's names (`crate::protocol::command`).
pub use crate::protocol::command::{
    BIND_NATIVE_GPU_MEMORY, BIND_NATIVE_MEMORY, CALCULATE_IMAGE_REQUIREMENTS,
    CALCULATE_NATIVE_IMAGE_SIZE, CREATE_NATIVE_IMAGE, CREATE_NATIVE_MEMORY,
    CREATE_NATIVE_WINDOW_ALIAS_IMAGE_POOL, DESTROY_NATIVE_IMAGE, DESTROY_NATIVE_MEMORY,
    DESTROY_NATIVE_WINDOW_ALIAS_IMAGE_POOL,
};

/// What a native memory is bound to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    /// Process memory (114), with or without the GPU-accessible check.
    Process {
        /// Whether 114's flags were [`BIND_GPU_ACCESSIBLE`].
        gpu_accessible: bool,
    },
    /// Part of a gpu worker registration (423).
    Gpu {
        /// The registration's handle.
        parent: u32,
        /// Where the range starts in it.
        offset: u64,
    },
}

/// Where a native memory is bound, and the device-share lock it holds on the range.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Binding {
    /// The first CPU address.
    pub address: u64,
    /// What it was bound by.
    pub source: Source,
    /// The lock on the memory's range, released when the memory goes.
    pub token: DeviceShareToken,
}

/// A native memory (111).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NativeMemory {
    mode: u8,
    size: u64,
    binding: Option<Binding>,
    retired: bool,
}

impl NativeMemory {
    /// The mode byte 111 named.
    #[must_use]
    pub const fn mode(&self) -> u8 {
        self.mode
    }

    /// Its size in bytes.
    #[must_use]
    pub const fn size(&self) -> u64 {
        self.size
    }

    /// Where it is bound, if it is.
    #[must_use]
    pub const fn binding(&self) -> Option<&Binding> {
        self.binding.as_ref()
    }

    /// Whether 112 destroyed it while an image used it (module docs, "Checks, in order").
    #[must_use]
    pub const fn retired(&self) -> bool {
        self.retired
    }
}

/// A native image (429).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NativeImage {
    memory: u32,
    preference: Preference,
    count: u32,
    size: u64,
    window_pool: bool,
}

impl NativeImage {
    /// The native memory it lies in.
    #[must_use]
    pub const fn memory(&self) -> u32 {
        self.memory
    }

    /// Its preference (what vi2 reads its extent, pitch, block height and format from).
    #[must_use]
    pub const fn preference(&self) -> &Preference {
        &self.preference
    }

    /// The count 429 named.
    #[must_use]
    pub const fn count(&self) -> u32 {
        self.count
    }

    /// Its size: [`layout::requirements`] of its preference and count.
    #[must_use]
    pub const fn size(&self) -> u64 {
        self.size
    }

    /// Whether 315 put it in a window alias image pool.
    #[must_use]
    pub const fn window_pool(&self) -> bool {
        self.window_pool
    }
}

/// The native memories and images of a process (module docs, "Where the tables live"), and the
/// video image pools that lie in its memories ([`pools`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NativeObjects {
    memories: BTreeMap<u32, NativeMemory>,
    images: BTreeMap<u32, NativeImage>,
    /// The next memory id; 0 once [`u32::MAX`] has been handed out.
    next_memory: u32,
    /// The next image id, shared by native images and pool images; 0 once [`u32::MAX`] has
    /// been handed out.
    next_image: u32,
    /// The video image pools, by first image id ([`pools`]).
    pools: BTreeMap<u32, pools::VideoPool>,
    /// The GPU mappings of pool images, by address space and image ([`pools`]).
    mappings: BTreeMap<(u32, u32), pools::GpuMapping>,
}

impl Default for NativeObjects {
    fn default() -> Self {
        Self::new()
    }
}

/// A 114 or 423 that passed the checks of [`NativeObjects::check_bind`]. It holds the tables, so
/// nothing changes them before [`bind`](Self::bind) records the binding, which therefore cannot
/// fail; dropping it records nothing.
#[derive(Debug)]
pub struct BindAdmission<'t> {
    memory: &'t mut NativeMemory,
    address: u64,
    source: Source,
}

impl BindAdmission<'_> {
    /// The CPU range to lock: its address and size.
    #[must_use]
    pub const fn range(&self) -> (u64, u64) {
        (self.address, self.memory.size)
    }

    /// Whether the range must be GPU accessible ([`BIND_GPU_ACCESSIBLE`]).
    #[must_use]
    pub const fn needs_gpu_accessible(&self) -> bool {
        !matches!(
            self.source,
            Source::Process {
                gpu_accessible: false
            }
        )
    }

    /// Records the binding with its lock's token.
    pub fn bind(self, token: DeviceShareToken) {
        self.memory.binding = Some(Binding {
            address: self.address,
            source: self.source,
            token,
        });
    }
}

/// A 114 or 423, decoded (PR `BindNativeMemoryBacking`'s arguments).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BindRequest {
    /// The native memory.
    pub id: u32,
    /// The first CPU address.
    pub address: u64,
    /// The size.
    pub size: u64,
    /// The flags: [`BIND_PROCESS_MEMORY`] or [`BIND_GPU_ACCESSIBLE`].
    pub flags: u64,
    /// The offset (PR requires 0).
    pub offset: u64,
}

/// 423's parent, as the request names it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ParentRange {
    /// The registration's handle.
    pub handle: u32,
    /// Where the range starts in it.
    pub offset: u64,
}

/// Hands out `*next` and moves it on; `None` when the ids have run out (0). After
/// [`u32::MAX`] the next is 0.
fn take_id(next: &mut u32) -> Option<u32> {
    let id = *next;
    if id == 0 {
        return None;
    }
    *next = id.wrapping_add(1);
    Some(id)
}

impl NativeObjects {
    /// No memories and no images.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            memories: BTreeMap::new(),
            images: BTreeMap::new(),
            next_memory: FIRST_ID,
            next_image: FIRST_ID,
            pools: BTreeMap::new(),
            mappings: BTreeMap::new(),
        }
    }

    /// A live native memory (not retired).
    #[must_use]
    pub fn memory(&self, id: u32) -> Option<&NativeMemory> {
        self.memories.get(&id).filter(|memory| !memory.retired)
    }

    /// Every native memory, retired ones included, by id.
    pub fn memories(&self) -> impl Iterator<Item = (u32, &NativeMemory)> {
        self.memories.iter().map(|(&id, memory)| (id, memory))
    }

    /// A native image.
    #[must_use]
    pub fn image(&self, id: u32) -> Option<&NativeImage> {
        self.images.get(&id)
    }

    /// Every native image, by id.
    pub fn images(&self) -> impl Iterator<Item = (u32, &NativeImage)> {
        self.images.iter().map(|(&id, image)| (id, image))
    }

    /// The live memories (not retired).
    fn live_memories(&self) -> usize {
        self.memories
            .values()
            .filter(|memory| !memory.retired)
            .count()
    }

    /// Whether an image or a video image pool ([`pools`]) lies in memory `id`.
    fn in_use(&self, id: u32) -> bool {
        self.images.values().any(|image| image.memory == id) || self.pool_in(id)
    }

    /// After the last image or pool in memory `id` may have gone: if the memory is retired and
    /// nothing uses it now, removes it and returns its lock to release, if it was bound (PR
    /// `DrainRetiredMemories`, which 430 and 912 both run).
    fn release_if_retired(&mut self, id: u32) -> Option<DeviceShareToken> {
        let retired = self.memories.get(&id).is_some_and(|backing| backing.retired);
        if !retired || self.in_use(id) {
            return None;
        }
        self.memories
            .remove(&id)
            .and_then(|backing| backing.binding)
            .map(|binding| binding.token)
    }

    /// Sets the next ids, for tests at the end of the id space.
    #[cfg(test)]
    pub(crate) fn set_next_ids(&mut self, memory: u32, image: u32) {
        self.next_memory = memory;
        self.next_image = image;
    }

    /// 111 `CreateNativeMemory` (module docs, "Checks, in order").
    pub fn create_memory(&mut self, mode: u8, size: u64) -> Result<u32, ResultCode> {
        if mode > MAX_MODE {
            return Err(KernelError::InvalidCombination.code());
        }
        if size == 0 || size > MAX_MEMORY_SIZE {
            return Err(GSV_INVALID_SIZE);
        }
        if self.live_memories() >= MAX_MEMORIES {
            return Err(KernelError::OutOfResource.code());
        }
        let id = take_id(&mut self.next_memory).ok_or(KernelError::OutOfResource.code())?;
        self.memories.insert(
            id,
            NativeMemory {
                mode,
                size,
                binding: None,
                retired: false,
            },
        );
        Ok(id)
    }

    /// 114's checks before the kernel's (PR `BindNativeMemoryBacking`, to the alignment
    /// check), for a bind by `source`.
    fn admit(
        &mut self,
        request: &BindRequest,
        source: Source,
    ) -> Result<BindAdmission<'_>, ResultCode> {
        let memory = self
            .memories
            .get_mut(&request.id)
            .filter(|memory| !memory.retired)
            .ok_or(KernelError::NotFound.code())?;
        let flags_known =
            request.flags == BIND_PROCESS_MEMORY || request.flags == BIND_GPU_ACCESSIBLE;
        if !flags_known || request.offset != 0 {
            return Err(KernelError::NotImplemented.code());
        }
        if memory.binding.is_some() {
            return Err(KernelError::InvalidState.code());
        }
        let wraps = request.address.checked_add(request.size).is_none();
        if request.size == 0 || request.size != memory.size || request.address == 0 || wraps {
            return Err(KernelError::InvalidSize.code());
        }
        if !request.address.is_multiple_of(PAGE_SIZE) {
            return Err(KernelError::InvalidAddress.code());
        }
        Ok(BindAdmission {
            memory,
            address: request.address,
            source,
        })
    }

    /// 114 `BindNativeMemory`'s checks before the kernel's (module docs, "Checks, in order").
    pub fn check_bind(&mut self, request: &BindRequest) -> Result<BindAdmission<'_>, ResultCode> {
        let source = Source::Process {
            gpu_accessible: request.flags == BIND_GPU_ACCESSIBLE,
        };
        self.admit(request, source)
    }

    /// 423 `BindNativeGpuMemory`'s checks before the kernel's: `parent` is the registration
    /// the request's parent handle names, if any (module docs, "Checks, in order").
    pub fn check_gpu_bind(
        &mut self,
        parent: Option<&Registration>,
        range: ParentRange,
        request: &BindRequest,
    ) -> Result<BindAdmission<'_>, ResultCode> {
        let parent = parent.ok_or(KernelError::NotFound.code())?;
        let (start, size) = (parent.range.start(), parent.range.size());
        // PR: `offset <= parent->size && size <= parent->size - offset && parent->address +
        // offset >= parent->address && address == parent->address + offset`.
        let inside = range.offset <= size
            && request.size <= size - range.offset
            && start
                .checked_add(range.offset)
                .is_some_and(|first| first == request.address);
        if !inside {
            return Err(KernelError::NotFound.code());
        }
        let source = Source::Gpu {
            parent: range.handle,
            offset: range.offset,
        };
        self.admit(request, source)
    }

    /// 112 `DestroyNativeMemory`: the lock to release now, if the memory goes now and was
    /// bound (module docs, "Checks, in order").
    pub fn destroy_memory(&mut self, id: u32) -> Result<Option<DeviceShareToken>, ResultCode> {
        if self.memory(id).is_none() {
            return Err(KernelError::NotFound.code());
        }
        if self.in_use(id) {
            if let Some(memory) = self.memories.get_mut(&id) {
                memory.retired = true;
            }
            return Ok(None);
        }
        Ok(self
            .memories
            .remove(&id)
            .and_then(|memory| memory.binding)
            .map(|binding| binding.token))
    }

    /// 429 `CreateNativeImage` (module docs, "Checks, in order").
    pub fn create_image(
        &mut self,
        preference: &Preference,
        count: u32,
        memory: u32,
    ) -> Result<u32, ResultCode> {
        if preference[WORD_KIND] != KIND_NATIVE {
            return Err(KernelError::InvalidCombination.code());
        }
        let size = layout::requirements(preference, count)
            .map_err(|_| KernelError::NotImplemented.code())?
            .size;
        let backing = self
            .memory(memory)
            .filter(|backing| backing.binding.is_some())
            .ok_or(KernelError::NotFound.code())?;
        if backing.size < size {
            return Err(KernelError::InvalidSize.code());
        }
        if self.images.len() >= MAX_IMAGES {
            return Err(KernelError::OutOfResource.code());
        }
        let id = take_id(&mut self.next_image).ok_or(KernelError::OutOfResource.code())?;
        self.images.insert(
            id,
            NativeImage {
                memory,
                preference: *preference,
                count,
                size,
                window_pool: false,
            },
        );
        Ok(id)
    }

    /// 430 `DestroyNativeImage`: the lock to release now, if the image was the last of a
    /// retired memory that was bound (module docs, "Checks, in order").
    pub fn destroy_image(&mut self, id: u32) -> Result<Option<DeviceShareToken>, ResultCode> {
        let image = self.images.get(&id).ok_or(KernelError::NotFound.code())?;
        if image.window_pool {
            return Err(KernelError::Busy.code());
        }
        let memory = image.memory;
        self.images.remove(&id);
        Ok(self.release_if_retired(memory))
    }

    /// 315 (`create`, with its count) and 314: puts image `id` in a window alias image pool or
    /// takes it out (module docs, "Checks, in order").
    pub fn set_window_pool(&mut self, id: u32, create: bool, count: u32) -> Result<(), ResultCode> {
        let image = self
            .images
            .get_mut(&id)
            .ok_or(KernelError::NotFound.code())?;
        if create && count != WINDOW_POOL_COUNT {
            return Err(KernelError::NotImplemented.code());
        }
        if image.window_pool == create {
            return Err(KernelError::InvalidState.code());
        }
        image.window_pool = create;
        Ok(())
    }
}

/// 435's and 1510's answer for a preference of `kind` (module docs, "Checks, in order").
pub fn calculate(
    kind: u32,
    preference: &Preference,
    count: u32,
) -> Result<layout::Requirements, ResultCode> {
    if preference[WORD_KIND] != kind {
        return Err(KernelError::InvalidCombination.code());
    }
    layout::requirements(preference, count).map_err(|error| match error {
        PreferenceError::ZeroCount => GSV_INVALID_SIZE,
        _ => KernelError::NotImplemented.code(),
    })
}

/// Reads a preference: 16 words.
fn read_preference(words: &mut Reader<'_>) -> Result<Preference, Fault> {
    let mut preference = [0; PREFERENCE_WORDS];
    for word in &mut preference {
        *word = words.read_u32()?;
    }
    Ok(preference)
}

/// Releases `token`, a lock a memory held, during a request; a refusal is the request's answer
/// (module docs, "Checks, in order": the memory is gone all the same).
fn release(cx: &mut Context<'_>, token: Option<DeviceShareToken>) -> Result<(), Fault> {
    if let Some(token) = token {
        cx.unlock_device_share(token).inspect_err(|error| {
            tracing::error!(%error, "gsv: the kernel refused a native memory's release");
        })?;
    }
    Ok(())
}

/// Takes the kernel's steps of a bind `admission` passed and records it: the GPU-accessible
/// check if the bind asks for it, then the lock, the last step that can fail.
fn finish_bind(cx: &mut Context<'_>, admission: BindAdmission<'_>) -> Result<(), Fault> {
    let (address, size) = admission.range();
    if admission.needs_gpu_accessible() && !cx.is_gpu_accessible(address, size)? {
        return Err(KernelError::InvalidCurrentMemory.into());
    }
    let token = cx.lock_device_share(address, size)?;
    admission.bind(token);
    Ok(())
}

/// 111 `CreateNativeMemory`. Words: mode 2 (low byte), size 3–4. out: `u32` id.
pub fn create_memory(shared: &Shared, req: &Request<'_>, reply: &mut Reply) -> Outcome {
    let mut words = req.payload();
    let mode = low_byte(words.read_u32()?);
    let size = words.read_u64()?;
    let id = shared.lock().native().create_memory(mode, size)?;
    reply.payload().write_u32(id);
    reply.ok()
}

/// 112 `DestroyNativeMemory`. Words: id 2.
pub fn destroy_memory(
    shared: &Shared,
    cx: &mut Context<'_>,
    req: &Request<'_>,
    reply: &mut Reply,
) -> Outcome {
    let id = req.payload().read_u32()?;
    let mut locked = shared.lock();
    let token = locked.native().destroy_memory(id)?;
    release(cx, token)?;
    reply.ok()
}

/// 114 `BindNativeMemory`. Words: id 2, address 3–4, size 5–6, flags 7–8, offset 9–10.
pub fn bind_memory(
    shared: &Shared,
    cx: &mut Context<'_>,
    req: &Request<'_>,
    reply: &mut Reply,
) -> Outcome {
    let mut words = req.payload();
    let request = BindRequest {
        id: words.read_u32()?,
        address: words.read_u64()?,
        size: words.read_u64()?,
        flags: words.read_u64()?,
        offset: words.read_u64()?,
    };
    let mut locked = shared.lock();
    let admission = locked.native().check_bind(&request)?;
    finish_bind(cx, admission)?;
    reply.ok()
}

/// 423 `BindNativeGpuMemory`. Words: id 2, address 3–4, size 5–6, parent 7, offset 8–9, two
/// zero `u64`s 10–13.
pub fn bind_gpu_memory(
    shared: &Shared,
    cx: &mut Context<'_>,
    req: &Request<'_>,
    reply: &mut Reply,
) -> Outcome {
    let mut words = req.payload();
    let id = words.read_u32()?;
    let address = words.read_u64()?;
    let size = words.read_u64()?;
    let range = ParentRange {
        handle: words.read_u32()?,
        offset: words.read_u64()?,
    };
    let (first, second) = (words.read_u64()?, words.read_u64()?);
    // PR: `pair(10) == 0 && pair(12) == 0`, with the header checks.
    if first != 0 || second != 0 {
        return Err(KernelError::InvalidCombination.into());
    }
    let request = BindRequest {
        id,
        address,
        size,
        flags: BIND_GPU_ACCESSIBLE,
        offset: 0,
    };
    let mut locked = shared.lock();
    let parent = locked.gpu_memory().registration(range.handle).copied();
    let admission = locked
        .native()
        .check_gpu_bind(parent.as_ref(), range, &request)?;
    finish_bind(cx, admission)?;
    reply.ok()
}

/// 429 `CreateNativeImage`. Words: preference 2–17, count 18, memory 19. out: `u32` id.
pub fn create_image(shared: &Shared, req: &Request<'_>, reply: &mut Reply) -> Outcome {
    let mut words = req.payload();
    let preference = read_preference(&mut words)?;
    let count = words.read_u32()?;
    let memory = words.read_u32()?;
    let id = shared
        .lock()
        .native()
        .create_image(&preference, count, memory)?;
    reply.payload().write_u32(id);
    reply.ok()
}

/// 430 `DestroyNativeImage`. Words: id 2.
pub fn destroy_image(
    shared: &Shared,
    cx: &mut Context<'_>,
    req: &Request<'_>,
    reply: &mut Reply,
) -> Outcome {
    let id = req.payload().read_u32()?;
    let mut locked = shared.lock();
    let token = locked.native().destroy_image(id)?;
    release(cx, token)?;
    reply.ok()
}

/// 315 `CreateNativeWindowAliasImagePool` (image 2, count 3) and 314
/// `DestroyNativeWindowAliasImagePool` (image 2).
pub fn window_pool(shared: &Shared, req: &Request<'_>, reply: &mut Reply) -> Outcome {
    let create = req.command_id() == CREATE_NATIVE_WINDOW_ALIAS_IMAGE_POOL;
    let mut words = req.payload();
    let id = words.read_u32()?;
    let count = if create { words.read_u32()? } else { 0 };
    shared.lock().native().set_window_pool(id, create, count)?;
    reply.ok()
}

/// 435 `CalculateNativeImageSize` (out: `u64` size) and 1510 `CalculateImageRequirements` (out:
/// `u64` size, `u64` alignment). Words: preference 2–17, count 18.
pub fn calculate_requirements(req: &Request<'_>, reply: &mut Reply) -> Outcome {
    let gpu_image = req.command_id() == CALCULATE_NATIVE_IMAGE_SIZE;
    let mut words = req.payload();
    let preference = read_preference(&mut words)?;
    let count = words.read_u32()?;
    let kind = if gpu_image { KIND_NATIVE } else { KIND_BLOCK };
    let requirements = calculate(kind, &preference, count)?;
    let out = reply.payload();
    out.write_u64(requirements.size);
    if !gpu_image {
        out.write_u64(requirements.alignment);
    }
    reply.ok()
}
