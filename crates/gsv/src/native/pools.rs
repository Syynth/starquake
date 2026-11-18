//! Video image pools and native GPU images: the root commands that size, make, destroy and
//! describe a pool of decoder output images, and map them for the GPU
//! (`docs/specs/graphics-server.md`, "Video codec sessions" and "The root"; port slice 15).
//!
//! Ported from Eden PR #4588,
//! `eden-pr4588/src/core/hle/service/am/service/graphics_supervisor.cpp` @ e5fb3b97
//! (`IGraphicsSupervisor::CalculateVideoImageRequirements`, `CreateVideoImagePool`,
//! `DestroyVideoImagePool`, `QueryVideoImage`, `MapVideoGpuImage`, `UnmapVideoGpuImage`,
//! `QueryVideoGpuImage`) and `native_video_decoder.cpp` @ e5fb3b97
//! (`NativeVideoService::CreateImagePool`, `DestroyImagePool`, `QueryVideoImage`,
//! `QueryGpuImage`, `MapGpuImage`, `UnmapGpuImage`), "PR"; and Eden-NX2,
//! `eden-nx2/src/core/hle/service/nx2_gsv.h` @ 50fa86bb (`CalculateDecoderImagePoolBuffer`,
//! `CreateDecoderImage`, `GetCpuImageDescriptor`), "NX2"; both GPL-3.0-or-later, Copyright 2026
//! Eden Emulator Project. "Row N" is a row of `eden-nx2/NX2_NOTES.md` @ 50fa86bb.
//!
//! # Where the pools live
//!
//! In the native tables ([`NativeObjects`]), beside the native memories and images, because a
//! pool lies in a native memory and its images take their ids from the same counter as native
//! images (PR: `first_image = m_next_native_image_id; m_next_native_image_id += count`; NX2:
//! `const u32 first = m_objects->next_image; m_objects->next_image += count`). Image `i` of a
//! pool made with first id `f` is `f + i` (NX2: "the client names image i of the pool id + i").
//! The decoder service ([`crate::video`]) looks its output images up here.
//!
//! A pool keeps its memory in use, as a native image does: 112 of the memory retires it, and the
//! pool's 912 releases it with its last user (PR: the pool pins the memory's nvmap handle, so
//! `ReleaseNativeMemory` retires it, and `DestroyVideoImagePool` drains the retired memories).
//!
//! # The commands
//!
//! Every command is TIPC; words are counted from the start of the message, the two header words
//! first, as PR counts them.
//!
//! | Id | Command | Words | Out | Source |
//! |---|---|---|---|---|
//! | 991 | `CalculateVideoImageRequirements` | preference 2–17, count 18 | `u64` size, `u64` alignment | row 96; PR (`0x3ef, 17`); NX2 `CalculateDecoderImagePoolBuffer` |
//! | 911 | `CreateVideoImagePool` | preference 2–17, count 18, memory 19 | `u32` first image id | row 148; PR (`0x39f, 18`); NX2 `CreateDecoderImage` |
//! | 912 | `DestroyVideoImagePool` | first image id 2 | nothing | PR (`0x3a0, 1`) |
//! | 961 | `QueryVideoImage` | image 2 | sixteen `u64` descriptor words | row 148; PR (`0x3d1, 1`); NX2 `GetCpuImageDescriptor` |
//! | 428 | `QueryNativeGpuImage` | image 2 | `u64` 0, `u64` 0, sixteen `u64` descriptor words, `u32` 0 | PR (`0x1bc, 1`) |
//! | 426 | `MapNativeGpuImage` | image 2, 1 at 3, address space 4, mode 5 (low byte 1 or 3) | `u64` address, `u64` last address, sixteen `u64` descriptor words, `u32` 0 | PR (`0x1ba, 4`) |
//! | 427 | `UnmapNativeGpuImage` | image 2, address space 3, address 4–5, last address 6–7 | nothing | PR (`0x1bb, 6`) |
//!
//! **Checks, in order** (PR's; the first failing check answers; every word is read first):
//!
//! - 991: a preference and count [`layout`] refuses, `NotImplemented`.
//! - 911: no live memory with the id, `NotFound`; the image ids run out (the next id is 0, or
//!   `count` ids from it pass `u32::MAX`) or a count of 0, `OutOfResource`; a preference
//!   [`layout`] refuses, or a count over [`MAX_POOL_IMAGES`], `NotImplemented`;
//!   [`MAX_POOLS`] pools, `OutOfResource`; a memory not bound, `NotFound`; a memory smaller than
//!   the pool, `InvalidSize`.
//! - 912: no pool with that first id, `NotFound`. The pool goes, with its images' GPU mappings;
//!   if its memory is retired and nothing else uses it, the memory goes and its lock is released,
//!   as 430 does (a refused release answers the kernel's error, the memory gone all the same).
//! - 961: no pool image with the id, `NotFound`; a block-linear pool (code 5),
//!   `NotImplemented` (PR: `R_UNLESS(pool->layout.nv12, ...)`, here also taking NX2's pitch-linear
//!   RGBA and BGRA); the image's offset plus its bytes over `u32::MAX`, `InvalidSize`.
//! - 428: no pool image with the id, `NotFound`; its offset over `u32::MAX`, `InvalidSize`.
//! - 426: word 3 not 1 or the mode byte neither 1 nor 3, `InvalidCombination`; no address space
//!   with the id in gsv's GPU memory, `NotFound`; 428's checks; the image mapped in the space
//!   already: its count at `u32::MAX`, `OutOfResource`, else counted once more and the same
//!   address answered; otherwise [`MAX_MAPPINGS`] mappings, `OutOfResource`.
//! - 427: no mapping of the image in the space at that address whose last address is the one
//!   given, `NotFound`. Its count drops by one; at 0 it goes.
//!
//! # The descriptors
//!
//! **961** (PR `QueryVideoImage`, "Layout of the NV12 record returned by gsv (run in emulation
//! over a size grid)"), as `u64` words, the rest zero:
//!
//! - NV12: word 0 `extent << 32 | 0xa`; word 1 `luma bytes << 32 | offset`; word 2 the pitch;
//!   word 3 `chroma bytes << 32 | (offset + chroma offset)`; word 4 the pitch.
//! - RGBA or BGRA, pitch-linear (NX2's codes; NX2 `GetCpuImageDescriptor`: "u8 kind at +0; ...
//!   the plane's offset into the pool memory (u32 +0x8) and row pitch (u64 +0x10)"): word 0 the
//!   kind, [`DESCRIPTOR_RGBA`] or [`DESCRIPTOR_BGRA`]; word 1 the offset; word 2 the pitch.
//!
//! **428** (PR `QueryGpuImage`): word 0 [`GPU_DESCRIPTOR_KIND`], word 1 the extent word,
//! word 4 [`GPU_DESCRIPTOR_WORD_4`], word 5 `image bytes << 32 | offset`; 426 answers the same
//! with word 5's low half cleared (PR: `descriptor[5] &= 0xffffffff00000000ULL`). What words 0
//! and 4 mean is a **guess**: PR writes the constants without saying.
//!
//! # Where this differs from the references
//!
//! Each row is pinned by the test named, in `native/pools/tests.rs`:
//!
//! | Case | PR | NX2 | Here |
//! |---|---|---|---|
//! | 991, 911, 961 before 901 | `InvalidState` | accepted | accepted, as the root's other commands (decision 7): no state is kept for 901 (`pool_commands_need_no_901`) |
//! | 961's NV12 descriptor | the module's record, with the extent and the plane sizes | kind, offsets and pitches only (`u32` words 0, 2, 4, 6, 8) | **conflict**: PR's, which holds NX2's in the same words and more (decision 8, the larger) (`the_nv12_descriptor_is_pr_s_record`) |
//! | 961 of a pitch-linear RGBA or BGRA pool | (no such pool) | kinds 8 and 9 | NX2's (`a_pitch_linear_rgba_descriptor_is_nx2_s`) |
//! | 961 of a block-linear pool (code 5) | `NotImplemented` | (no such pool: NX2 lays code 5 out as BGRA) | PR's: follows from code 5's layout, a departure from decision 8 and an owner question ([`crate::video::layout`], "Where the references disagree") (`a_block_linear_pool_has_no_cpu_descriptor`) |
//! | An offset past 32 bits in 961 | `InvalidSize` | cut to 32 bits | PR's, for every layout; an image ending exactly at 2^32 is refused (`an_offset_past_32_bits_is_refused`, `an_image_ending_at_two_to_the_32_is_refused`) |
//! | 911's pool limits | 1024 pools of at most 1024 images | none | PR's (`pools_are_bounded_at_the_limit`) |
//! | 426's address space | one 434 registered on a gpu session (434 is a stub here, so PR would answer `NotFound` to every 426) | (no 426) | any address space of gsv's GPU memory (`a_mapping_needs_an_address_space`) |
//! | 426's address | a GPU address the address space allocates, with the pool's memory mapped there | (no 426) | the image's CPU address (the memory's binding plus the image's offset): GPU addresses equal CPU addresses (spec, "GPU memory"), and nothing is published in the translation table. A memory bound by 423 is part of a 101 registration, which is mapped at its own address; one bound by 114 is not mapped for the GPU at all. A **guess**, until a title maps a pool image and a trace shows the address it reads (`a_mapping_answers_the_image_s_cpu_address`) |
//! | 426's mappings | unbounded | (no 426) | at most [`MAX_MAPPINGS`], then `OutOfResource` (`mappings_are_bounded_at_the_limit`) |
//! | 912 of a pool whose images are mapped, or registered with a decoder | the pool's memory stays pinned until the mappings and decoders let go | (no 912) | the mappings go with the pool, and its memory is released if it is retired: nothing here writes an image (the null decoder), so nothing can write the memory after (`destroying_a_pool_drops_its_mappings`) |
//! | A refused request's reply | the result, then the data words, zero | | the result only, as the service framework answers every refusal |

pub use crate::video::layout::{self, Planes, Tiling, VideoLayout};

use starquake_kernel::memory::DeviceShareToken;
use starquake_kernel::result::{KernelError, ResultCode};
use starquake_services::framework::{Context, Fault, Outcome, Reader, Reply, Request};

use super::{NativeObjects, Preference, read_preference, release};
use crate::gpu::{AsId, low_byte};
use crate::shared::Shared;

#[cfg(test)]
mod tests;

/// The most pools (PR `CreateImagePool`: `R_UNLESS(state->pools.size() < 1024, ...)`).
pub const MAX_POOLS: usize = 1024;
/// The most images in a pool (PR `CreateImagePool`: `R_UNLESS(layout && count <= 1024, ...)`).
pub const MAX_POOL_IMAGES: u32 = 1024;
/// The most GPU mappings of pool images at once: starquake's own bound (PR has none), the
/// native image limit ([`super::MAX_IMAGES`]).
pub const MAX_MAPPINGS: usize = super::MAX_IMAGES;

/// 961's kind for an NV12 image (PR `descriptor[0] = ... | 0xa`; NX2 `Nx2PictureFormat::Nv12 ?
/// 0xa`).
pub const DESCRIPTOR_NV12: u64 = 0xA;
/// 961's kind for a pitch-linear RGBA image (NX2 `: 0x8`).
pub const DESCRIPTOR_RGBA: u64 = 0x8;
/// 961's kind for a pitch-linear BGRA image (NX2 `Nx2PictureFormat::Bgra8 ? 0x9`).
pub const DESCRIPTOR_BGRA: u64 = 0x9;
/// 428's word 0 (PR `descriptor[0] = 1`). Meaning a **guess**.
pub const GPU_DESCRIPTOR_KIND: u64 = 1;
/// 428's word 4 (PR `descriptor[4] = 0x101000001ULL`). Meaning a **guess**.
pub const GPU_DESCRIPTOR_WORD_4: u64 = 0x1_0100_0001;
/// What 426's word 3 must be (PR `packet[3] == 1`). Meaning a **guess**.
pub const MAP_WORD_3: u32 = 1;
/// The mode bytes 426's word 5 may hold (PR `(packet[5] & 0xff) == 1 || ... == 3`). Meaning a
/// **guess**.
pub const MAP_MODES: [u8; 2] = [1, 3];
/// How many `u64` words a descriptor holds (PR `std::array<u64, 16>`).
pub const DESCRIPTOR_WORDS: usize = 16;
/// The high half of a descriptor word (PR `descriptor[5] &= 0xffffffff00000000ULL`).
const HIGH_HALF: u64 = 0xFFFF_FFFF_0000_0000;

/// A descriptor: sixteen `u64` words.
pub type Descriptor = [u64; DESCRIPTOR_WORDS];

/// A video image pool (911).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VideoPool {
    pub(super) memory: u32,
    pub(super) preference: Preference,
    pub(super) count: u32,
    pub(super) layout: VideoLayout,
}

impl VideoPool {
    /// The native memory it lies in.
    #[must_use]
    pub const fn memory(&self) -> u32 {
        self.memory
    }

    /// The preference 911 named.
    #[must_use]
    pub const fn preference(&self) -> &Preference {
        &self.preference
    }

    /// How many images it holds.
    #[must_use]
    pub const fn count(&self) -> u32 {
        self.count
    }

    /// Its layout: [`layout::layout`] of its preference and count.
    #[must_use]
    pub const fn layout(&self) -> &VideoLayout {
        &self.layout
    }
}

/// A pool image mapped for the GPU (426), counted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuMapping {
    pub(super) address: u64,
    pub(super) size: u64,
    pub(super) references: u32,
}

impl GpuMapping {
    /// Its GPU address.
    #[must_use]
    pub const fn address(&self) -> u64 {
        self.address
    }

    /// Its bytes.
    #[must_use]
    pub const fn size(&self) -> u64 {
        self.size
    }

    /// How many 426s it has had that no 427 undid.
    #[must_use]
    pub const fn references(&self) -> u32 {
        self.references
    }
}

/// A pool image, found: its pool's first id, the pool and its index in it.
#[derive(Clone, Copy, Debug)]
pub struct PoolImage<'t> {
    /// The pool's first image id.
    pub first: u32,
    /// The pool.
    pub pool: &'t VideoPool,
    /// The image's index in the pool.
    pub index: u32,
}

impl PoolImage<'_> {
    /// Where the image starts in its pool's memory, if it fits in 32 bits.
    fn offset(&self) -> Result<u32, ResultCode> {
        self.pool
            .layout
            .offset(self.index)
            .and_then(|offset| u32::try_from(offset).ok())
            .ok_or(KernelError::InvalidSize.code())
    }
}

impl NativeObjects {
    /// Every pool, by first image id.
    pub fn pools(&self) -> impl Iterator<Item = (u32, &VideoPool)> {
        self.pools.iter().map(|(&first, pool)| (first, pool))
    }

    /// Every GPU mapping of a pool image, by address space and image.
    pub fn gpu_mappings(&self) -> impl Iterator<Item = ((AsId, u32), &GpuMapping)> {
        self.mappings
            .iter()
            .map(|(&(space, image), mapping)| ((AsId(space), image), mapping))
    }

    /// The pool image with id `id`, if a pool holds it.
    #[must_use]
    pub fn pool_image(&self, id: u32) -> Option<PoolImage<'_>> {
        let (&first, pool) = self.pools.range(..=id).next_back()?;
        let index = id - first;
        (index < pool.count).then_some(PoolImage { first, pool, index })
    }

    /// Whether a pool lies in memory `id`.
    pub(super) fn pool_in(&self, id: u32) -> bool {
        self.pools.values().any(|pool| pool.memory == id)
    }

    /// Sets a pool image's mapping count, for tests at its limit.
    #[cfg(test)]
    pub(crate) fn set_mapping_references(&mut self, space: AsId, image: u32, references: u32) {
        if let Some(mapping) = self.mappings.get_mut(&(space.0, image)) {
            mapping.references = references;
        }
    }

    /// 911 `CreateVideoImagePool`: the first image id (module docs, "Checks, in order").
    pub fn create_pool(
        &mut self,
        preference: &Preference,
        count: u32,
        memory: u32,
    ) -> Result<u32, ResultCode> {
        if self.memory(memory).is_none() {
            return Err(KernelError::NotFound.code());
        }
        let first = self.next_image;
        // PR: `m_next_native_image_id && count && u64{m_next_native_image_id} + count <= (u64{1}
        // << 32)`. The next id is then `first + count`, 0 when that is 2^32.
        let end = u64::from(first) + u64::from(count);
        if first == 0 || count == 0 || end > 1 << 32 {
            return Err(KernelError::OutOfResource.code());
        }
        let layout = layout::layout(preference, count)
            .ok()
            .filter(|_| count <= MAX_POOL_IMAGES)
            .ok_or(KernelError::NotImplemented.code())?;
        if self.pools.len() >= MAX_POOLS {
            return Err(KernelError::OutOfResource.code());
        }
        let backing = self
            .memory(memory)
            .filter(|backing| backing.binding.is_some())
            .ok_or(KernelError::NotFound.code())?;
        if backing.size < layout.size {
            return Err(KernelError::InvalidSize.code());
        }
        self.next_image = u32::try_from(end).unwrap_or(0);
        self.pools.insert(
            first,
            VideoPool {
                memory,
                preference: *preference,
                count,
                layout,
            },
        );
        Ok(first)
    }

    /// 912 `DestroyVideoImagePool`: the lock to release now, if the pool was the last user of
    /// a retired memory that was bound (module docs, "Checks, in order").
    pub fn destroy_pool(&mut self, first: u32) -> Result<Option<DeviceShareToken>, ResultCode> {
        let pool = self
            .pools
            .remove(&first)
            .ok_or(KernelError::NotFound.code())?;
        let end = u64::from(first) + u64::from(pool.count);
        self.mappings
            .retain(|&(_, image), _| !(u64::from(first)..end).contains(&u64::from(image)));
        Ok(self.release_if_retired(pool.memory))
    }

    /// 961 `QueryVideoImage` (module docs, "The descriptors").
    pub fn video_descriptor(&self, id: u32) -> Result<Descriptor, ResultCode> {
        let image = self.pool_image(id).ok_or(KernelError::NotFound.code())?;
        let layout = &image.pool.layout;
        if layout.tiling == Tiling::BlockLinear {
            return Err(KernelError::NotImplemented.code());
        }
        let offset = image.offset()?;
        // PR: `R_UNLESS(offset + pool->layout.image_bytes <= numeric_limits<u32>::max(), ...)`.
        let end = u64::from(offset) + u64::from(layout.image_bytes);
        if end > u64::from(u32::MAX) {
            return Err(KernelError::InvalidSize.code());
        }
        let offset = u64::from(offset);
        let pitch = u64::from(layout.pitch);
        let mut descriptor = [0; DESCRIPTOR_WORDS];
        match layout.planes {
            Planes::Nv12 => {
                descriptor[0] = (u64::from(layout.extent) << 32) | DESCRIPTOR_NV12;
                descriptor[1] = (u64::from(layout.luma_bytes) << 32) | offset;
                descriptor[2] = pitch;
                // Under `end`, so under 2^32: the chroma plane lies inside the image.
                descriptor[3] = (u64::from(layout.chroma_bytes) << 32)
                    | (offset + u64::from(layout.chroma_offset));
                descriptor[4] = pitch;
            }
            Planes::Rgba | Planes::Bgra => {
                descriptor[0] = if layout.planes == Planes::Rgba {
                    DESCRIPTOR_RGBA
                } else {
                    DESCRIPTOR_BGRA
                };
                descriptor[1] = offset;
                descriptor[2] = pitch;
            }
        }
        Ok(descriptor)
    }

    /// 428 `QueryNativeGpuImage` (module docs, "The descriptors").
    pub fn gpu_descriptor(&self, id: u32) -> Result<Descriptor, ResultCode> {
        let image = self.pool_image(id).ok_or(KernelError::NotFound.code())?;
        let offset = image.offset()?;
        let layout = &image.pool.layout;
        let mut descriptor = [0; DESCRIPTOR_WORDS];
        descriptor[0] = GPU_DESCRIPTOR_KIND;
        descriptor[1] = u64::from(layout.extent);
        descriptor[4] = GPU_DESCRIPTOR_WORD_4;
        descriptor[5] = u64::from(offset) | (u64::from(layout.image_bytes) << 32);
        Ok(descriptor)
    }

    /// 426 `MapNativeGpuImage`, after its word checks, for an address space that exists:
    /// the mapping and the descriptor (module docs, "Checks, in order").
    pub fn map_gpu_image(
        &mut self,
        space: AsId,
        id: u32,
    ) -> Result<(GpuMapping, Descriptor), ResultCode> {
        let mut descriptor = self.gpu_descriptor(id)?;
        descriptor[5] &= HIGH_HALF;
        if let Some(mapping) = self.mappings.get_mut(&(space.0, id)) {
            mapping.references = mapping
                .references
                .checked_add(1)
                .ok_or(KernelError::OutOfResource.code())?;
            return Ok((*mapping, descriptor));
        }
        if self.mappings.len() >= MAX_MAPPINGS {
            return Err(KernelError::OutOfResource.code());
        }
        let image = self.pool_image(id).ok_or(KernelError::NotFound.code())?;
        let offset = image.offset()?;
        let size = u64::from(image.pool.layout.image_bytes);
        // A pool's memory is bound when the pool is made and stays bound: a bound memory is
        // only ever retired, never unbound, while a pool uses it.
        let address = self
            .memories
            .get(&image.pool.memory)
            .and_then(|memory| memory.binding)
            .and_then(|binding| binding.address.checked_add(u64::from(offset)))
            .ok_or(KernelError::NotFound.code())?;
        let mapping = GpuMapping {
            address,
            size,
            references: 1,
        };
        self.mappings.insert((space.0, id), mapping);
        Ok((mapping, descriptor))
    }

    /// 427 `UnmapNativeGpuImage` (module docs, "Checks, in order").
    pub fn unmap_gpu_image(
        &mut self,
        space: AsId,
        id: u32,
        address: u64,
        last: u64,
    ) -> Result<(), ResultCode> {
        let key = (space.0, id);
        let mapping = self
            .mappings
            .get_mut(&key)
            // PR: `entry->address == address && end >= address && end - address == entry->size
            // - 1`.
            .filter(|mapping| {
                mapping.address == address
                    && last
                        .checked_sub(address)
                        .is_some_and(|span| span == mapping.size.saturating_sub(1))
            })
            .ok_or(KernelError::NotFound.code())?;
        // At least 1: a mapping with no references is removed.
        mapping.references = mapping.references.saturating_sub(1);
        if mapping.references == 0 {
            self.mappings.remove(&key);
        }
        Ok(())
    }
}

/// Reads a preference and a count (words 2–18), and the reader after them.
fn read_request<'r>(req: &Request<'r>) -> Result<(Preference, u32, Reader<'r>), Fault> {
    let mut words = req.payload();
    let preference = read_preference(&mut words)?;
    let count = words.read_u32()?;
    Ok((preference, count, words))
}

/// 991 `CalculateVideoImageRequirements`. Words: preference 2–17, count 18. out: `u64` size,
/// `u64` alignment.
pub fn calculate_requirements(req: &Request<'_>, reply: &mut Reply) -> Outcome {
    let (preference, count, _) = read_request(req)?;
    let layout =
        layout::layout(&preference, count).map_err(|_| KernelError::NotImplemented.code())?;
    let out = reply.payload();
    out.write_u64(layout.size);
    out.write_u64(layout.alignment);
    reply.ok()
}

/// 911 `CreateVideoImagePool`. Words: preference 2–17, count 18, memory 19. out: `u32` first
/// image id.
pub fn create_pool(shared: &Shared, req: &Request<'_>, reply: &mut Reply) -> Outcome {
    let (preference, count, mut words) = read_request(req)?;
    let memory = words.read_u32()?;
    let first = shared
        .lock()
        .native()
        .create_pool(&preference, count, memory)?;
    reply.payload().write_u32(first);
    reply.ok()
}

/// 912 `DestroyVideoImagePool`. Words: first image id 2.
pub fn destroy_pool(
    shared: &Shared,
    cx: &mut Context<'_>,
    req: &Request<'_>,
    reply: &mut Reply,
) -> Outcome {
    let first = req.payload().read_u32()?;
    let mut locked = shared.lock();
    let token = locked.native().destroy_pool(first)?;
    release(cx, token)?;
    reply.ok()
}

/// Writes a descriptor's words.
fn write_descriptor(reply: &mut Reply, descriptor: &Descriptor) {
    let out = reply.payload();
    for word in descriptor {
        out.write_u64(*word);
    }
}

/// 961 `QueryVideoImage`. Words: image 2. out: sixteen `u64` descriptor words.
pub fn query_video_image(shared: &Shared, req: &Request<'_>, reply: &mut Reply) -> Outcome {
    let id = req.payload().read_u32()?;
    let descriptor = shared.lock().native_objects().video_descriptor(id)?;
    write_descriptor(reply, &descriptor);
    reply.ok()
}

/// 428 `QueryNativeGpuImage`. Words: image 2. out: `u64` 0, `u64` 0, sixteen `u64` descriptor
/// words, `u32` 0.
pub fn query_gpu_image(shared: &Shared, req: &Request<'_>, reply: &mut Reply) -> Outcome {
    let id = req.payload().read_u32()?;
    let descriptor = shared.lock().native_objects().gpu_descriptor(id)?;
    reply.payload().write_u64(0).write_u64(0);
    write_descriptor(reply, &descriptor);
    reply.payload().write_u32(0);
    reply.ok()
}

/// 426 `MapNativeGpuImage`. Words: image 2, 1 at 3, address space 4, mode 5. out: `u64`
/// address, `u64` last address, sixteen `u64` descriptor words, `u32` 0.
pub fn map_gpu_image(shared: &Shared, req: &Request<'_>, reply: &mut Reply) -> Outcome {
    let mut words = req.payload();
    let id = words.read_u32()?;
    let word_3 = words.read_u32()?;
    let space = AsId(words.read_u32()?);
    let mode = low_byte(words.read_u32()?);
    if word_3 != MAP_WORD_3 || !MAP_MODES.contains(&mode) {
        return Err(KernelError::InvalidCombination.into());
    }
    let mut locked = shared.lock();
    if locked.gpu_memory().table(space).is_none() {
        return Err(KernelError::NotFound.into());
    }
    let (mapping, descriptor) = locked.native().map_gpu_image(space, id)?;
    drop(locked);
    // A mapping's size is an image's bytes, at least 4096 for PR's codes and one pitch-row
    // block for NX2's, never 0; its last byte does not wrap: the image lies in bound memory.
    let last = mapping
        .address
        .saturating_add(mapping.size.saturating_sub(1));
    reply.payload().write_u64(mapping.address).write_u64(last);
    write_descriptor(reply, &descriptor);
    reply.payload().write_u32(0);
    reply.ok()
}

/// 427 `UnmapNativeGpuImage`. Words: image 2, address space 3, address 4–5, last address 6–7.
pub fn unmap_gpu_image(shared: &Shared, req: &Request<'_>, reply: &mut Reply) -> Outcome {
    let mut words = req.payload();
    let id = words.read_u32()?;
    let space = AsId(words.read_u32()?);
    let address = words.read_u64()?;
    let last = words.read_u64()?;
    shared
        .lock()
        .native()
        .unmap_gpu_image(space, id, address, last)?;
    reply.ok()
}
