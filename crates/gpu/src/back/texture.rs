//! The back thread's texture cache (gpu-protocol spec, "Caches", "Learning of CPU writes", "GPU
//! writes and the CPU", "GPU-only memory" and "Block-linear layout"; slice G17; Eden-NX2 notes
//! rows 69, 87 and 103).
//!
//! [`TextureCache`] maps images in guest memory to host textures made through the render backend
//! ([`Backend::create_texture`]), filled with [`Command::Upload`]s of rows the layout crate has
//! detiled (decision 5), and written back through the layout crate only when something reads
//! the image as plain memory (decision 4). The cache lives here and the host objects in the
//! backend, which sees only [`TextureId`]s and [`ViewId`]s (decision log 2026-11-12, "Draw
//! commands cross to the backend as plain data; caches are split by what they are keyed on").
//! It follows the buffer cache's shapes ([`super::buffer`]) for watches, streaming, destruction
//! and eviction, so the two caches agree; where it differs, the difference is listed below.
//!
//! One cache serves one GPU address space, and is handed that space's process's
//! [`WriteWatch`]. The executor (slice G19) owns it and calls it on the back thread.
//!
//! # Sources
//!
//! Paths are relative to the reference folder (`tools/refs-path.sh`), at the commits pinned in
//! `docs/references.md`. Eden's texture cache (`eden/src/video_core/texture_cache/` @ cf654052,
//! GPL-3.0-or-later) is a reference only, as the spec has it: no code is ported from it here
//! except the format table ([`format`]). The header layout is Nvidia's ([`header`]).
//!
//! # Images
//!
//! An image is an [`ImageInfo`]: its GPU address and everything its layout and host texture
//! depend on (format, dimension, extent, layers, levels, samples, block-linear or pitch). Two
//! uses with equal `ImageInfo`s are one image; anything else is another image, even at the same
//! address (`tests::the_same_header_twice_is_one_image`,
//! `tests::a_different_layout_at_the_same_address_is_another_image`). Its view, the levels a
//! shader reads, its dimension and its swizzle, is not part of it ([`ViewInfo`]): each distinct
//! view is made once per image, up to [`MAX_VIEWS_PER_IMAGE`].
//!
//! The image's layout is the layout crate's ([`starquake_layout::Image`]) for the description,
//! and its host texture is 1D, 2D or 3D (a cube is a 2D texture of six layers per cube, viewed as
//! a cube) of the image's format, with [`TextureUsage::COPY_SRC`] and
//! [`TextureUsage::COPY_DST`], [`TextureUsage::SAMPLED`] when the device samples the format, and
//! [`TextureUsage::RENDER_TARGET`] when it renders to it ([`texture_usage`]).
//!
//! **Unmapped addresses (row 103).** The cache is keyed by GPU address, so an image is found
//! again by its address whether or not guest memory is mapped there: the GPU address is the
//! stand-in identity, fixed by the address. Eden keys images by CPU address, so for an image
//! with no CPU backing Eden-NX2 makes a stand-in CPU address from the GPU one
//! (`ImageCpuAddress` in `eden-nx2/src/video_core/texture_cache/texture_cache.h` @ 50fa86bb);
//! keying by GPU address needs none. A render target whose first upload finds its memory
//! unmapped ([`GpuFault::Unmapped`]) is **host-only**: it is never uploaded, watched or written
//! back, as the spec's "GPU-only memory" has render targets in sparse regions (slice G22 replaces
//! the unmapped-fault test with the GPU-only store's answer). A sampled texture whose memory is
//! unmapped when it must be read is an error ([`TextureError::Fault`]). Pinned by
//! `tests::an_unmapped_render_target_is_found_again_by_its_address` and
//! `tests::a_sampled_texture_at_an_unmapped_address_is_an_error`.
//!
//! **Images whose bytes cannot be converted.** Two kinds of image are made without contents and
//! never uploaded or written back, with the reason in [`TextureEntry::unconvertible`]:
//! multisampled images (Eden's `RefreshContents` skips them when the runtime cannot upload
//! multisampled images, `CanUploadMSAA`, and wgpu cannot copy into one) and block-linear images
//! of a GOB the layout crate cannot copy (the guessed `FermiZs` depth GOB, [`header`]'s docs and
//! the spec's verification item 9). Pinned by
//! `tests::a_multisampled_image_is_made_without_contents` and
//! `tests::a_depth_image_of_the_guessed_gob_is_made_without_contents`.
//!
//! # Uses and uploads
//!
//! [`TextureCache::texture`] (a sampled texture from a decoded header) and
//! [`TextureCache::render_target`] find or make the image, upload it when its host copy may be
//! older than guest memory, and return its view. An upload is the whole image: every level of
//! every layer, read from guest memory, detiled into packed rows and uploaded in pieces of whole
//! rows of at most `Limits::max_staging_size` bytes. A use uploads when:
//!
//! - the image is new, or guest memory changed under it since its last upload (a CPU write from
//!   the dirty log, a back-thread write, [`TextureCache::invalidate`]): it is *stale*;
//! - its watch has not been seen armed, or the use is the first to see it armed (the next
//!   section), unless the image is GPU-modified (below);
//! - it is streamed (below).
//!
//! A stale image that is GPU-modified is uploaded all the same, and is no longer GPU-modified:
//! the guest wrote its memory last, as Eden's `RefreshContents` lets a CPU modification win.
//! Pinned by `tests::a_cpu_write_to_a_gpu_modified_image_wins`.
//!
//! # Learning of CPU writes
//!
//! A new image's pages are watched ([`WriteWatch::request`]), as the buffer cache watches its
//! entries. Until [`WriteWatch::armed`] says the request took effect the image is *untracked*
//! and every use uploads it, and **the first use after arming uploads it once more**: a write
//! between the last untracked read and the arming is in no dirty log. This is the unvetted
//! proposal "G16/G17 caches re-read a range on first use after arming" (`docs/proposals.md`,
//! review notes on #197), which the buffer cache follows too; it is an **owner question**
//! raised in the pull request. For a texture it costs a second whole-image upload per new image.
//! A GPU-modified image is not uploaded for being untracked: its host copy is newer than guest
//! memory, and only a write the dirty log reports (or a back-thread write) makes guest memory
//! win. Pinned by `tests::the_first_use_after_arming_uploads_the_image_again` and
//! `tests::an_untracked_gpu_modified_image_is_not_uploaded`.
//!
//! The dirty log has one reader, so the texture cache does not drain it: the executor drains it
//! once (the buffer cache's [`super::buffer::BufferCache::sync_cpu_writes`] hands every range to
//! its sink) and passes one drain's ranges to [`TextureCache::cpu_writes`]. For each image
//! overlapping a range, the image turns stale and, unless streamed, counts one fault per call
//! however many of its pages were written, and either asks for the written pages to be watched
//! again or, at the buffer cache's [`STREAMED_AFTER_FAULTS`], turns streamed. A use that finds
//! no write since the previous use sets the count back to zero. A streamed image asks for no more
//! watches and uploads at every use until it is dropped. Pinned by
//! `tests::a_cpu_write_invalidates_the_image` and
//! `tests::an_image_turns_streamed_at_the_fault_threshold_not_before`.
//!
//! As in the buffer cache, the watch speaks in CPU addresses and the cache in GPU addresses, and
//! the cache uses one as the other (row 30: memory registered with the graphics session is at
//! GPU address = CPU address). The buffer cache's module docs give the gap that leaves until
//! slice G22.
//!
//! # GPU writes, write-back and unmaps
//!
//! An image is *GPU-modified* when the host GPU may have written its texture since its last
//! upload or write-back: [`TextureCache::render_target`] marks it (the open list draws into it),
//! and so does [`TextureCache::gpu_written`] (for writers the draw path knows of, such as storage
//! images). Its host copy is then newer than guest memory, and it is written back only on demand
//! (decision 4): when a DMA copy or inline write reads it as plain memory, when an MME DMA read
//! touches it, or on request in tests. [`TextureCache::needs_write_back`] tells a caller whether a
//! range holds such an image. A write-back has two halves:
//!
//! 1. [`TextureCache::begin_write_back`] appends a [`Command::Readback`] of every level of every
//!    layer of each GPU-modified image overlapping the range to the open list, with ids from the
//!    back thread's [`ReadbackIds`], and returns a [`WriteBack`];
//! 2. once that list has completed, [`TextureCache::finish_write_back`] takes the readbacks,
//!    tiles them through the layout crate into the image's guest bytes (read first, so padding
//!    keeps its bytes) and writes them to guest memory. Each image it writes is no longer
//!    GPU-modified, and every other image overlapping a written range turns stale. It returns
//!    the written ranges, which the caller hands to the buffer cache's `invalidate`.
//!
//! An image drawn into again between the halves (its GPU generation moved) or dropped meanwhile
//! is left out of the second half: its readback no longer holds its newest contents. Pinned by
//! `tests::an_image_drawn_into_between_the_halves_is_not_written_back`. Host-only and
//! unconvertible images are never written back; [`WriteBack::skipped`] lists them with the
//! reason.
//!
//! [`TextureCache::invalidate`] takes a range the back thread itself wrote into guest memory (a
//! DMA copy, an inline upload or a 2D blit on the CPU path): every image overlapping it turns
//! stale. The spec's host copy for a DMA copy or 2D blit whose side the cache holds is the copy
//! engine's, and is not built yet. [`TextureCache::apply_update`] takes an applied ordered
//! address-table update: every image overlapping its range is dropped whole, whatever the
//! update's kind and whether or not it is GPU-modified (its newest contents are lost, as the
//! guest has unmapped the memory they would go to). An image ending exactly at the range's start
//! or starting exactly at its end is kept.
//!
//! # Destruction
//!
//! As in the buffer cache: a dropped or evicted image's texture and views are destroyed after the
//! fence of the last list that used it, learned from [`TextureCache::submitted`]; an image used
//! since the last report waits for that report. Dropping the cache destroys every texture and
//! view after the last fence it was told of.
//!
//! # Eviction
//!
//! [`TextureCache::end_frame`] runs the collector with the buffer cache's limits and ages
//! ([`eviction_limits`] from [`Backend::budget`], [`collection`]), over the cache's own bytes:
//! images not used for the collection's frames are destroyed, least recently used first, up to
//! its count. A GPU-modified image is never collected: its contents exist only on the host until
//! it is written back. A new image counts against a hard cap as the buffer cache's new entries do:
//! when the cache's live bytes (its images', and those of textures the open list still reads)
//! with the new image's would pass the budget's `allowed`, idle images (not used by the open
//! list, not GPU-modified) are evicted, least recently used first, until it fits, and only once
//! nothing can fail; if it still does not fit, the use is refused ([`TextureError::OverBudget`]).
//!
//! An image's bytes are its guest layout's size ([`starquake_layout::Image::size_b`]), which is
//! at least the packed host size of an uncompressed single-sampled image, but less than a host
//! texture a backend decodes (ASTC to RGBA8 is four to eight times larger, row 103).
//!
//! # Swizzle
//!
//! A header's component swizzle goes into the view when the backend's [`Caps::view_swizzle`] is
//! true; otherwise the view has the identity swizzle and [`TextureView::swizzle`] carries the
//! header's for slice G18 to copy into the shader variant key (decision log 2026-11-12, "The
//! render backend starts on wgpu": texture channel remapping becomes shader variants; review
//! guide M13). Pinned by `tests::a_swizzled_texture_under_a_no_swizzle_caps_gets_an_identity_view`
//! and `tests::a_swizzled_texture_under_a_swizzle_caps_keeps_it_in_the_view`.
//!
//! # Divergences and guesses
//!
//! - **No overlap resolution.** Eden joins overlapping images, copies between aliases and
//!   matches views of compatible formats (`JoinImages`, `SynchronizeAliases`, `FindImage`'s
//!   relaxed matching in `texture_cache.h`). Here images of different `ImageInfo`s are
//!   independent even where they overlap: a GPU-modified image's contents do not reach another
//!   image over the same memory until it is written back and the other is used again. This
//!   narrows nothing a later slice cannot add (aliases are a lookup over the same map).
//! - **Ages and counts.** Eden's texture collector uses its own ages and counts (50, 25 or 10
//!   frames; 10, 20 or 40 images; `RunGarbageCollector` in `texture_cache.h`) and downloads a
//!   GPU-modified image before collecting it under memory pressure. This cache uses the buffer
//!   cache's, so the two collect alike, and never collects a GPU-modified image. Pinned by
//!   `tests::an_idle_image_is_collected_after_the_buffer_cache_s_age_not_before` and
//!   `tests::a_gpu_modified_image_is_never_collected`.
//! - **Huge images (row 87).** Eden-NX2 makes an image over 2^28 texels sparse with no memory and
//!   never uploads it. Here an image larger than the budget allows, or than the device's limits,
//!   is an error ([`TextureError::OverBudget`] or [`TextureError::Render`]); a header whose layout
//!   overflows is [`TextureError::Layout`]. Pinned by
//!   `tests::hostile_headers_are_errors_not_panics`.
//! - **Views per image (guess).** [`MAX_VIEWS_PER_IMAGE`] bounds what a guest can make the host
//!   allocate per image; no reference has such a cap.
//! - **Format fallbacks.** The texture is made in the guest's format; choosing a host stand-in
//!   ([`starquake_render::host_format`]) is not done here.

mod format;
mod header;
mod transfer;

#[cfg(test)]
mod tests;

use core::ops::Range;
use std::collections::BTreeMap;
use std::sync::Arc;

use starquake_layout::{GobType, ImageDesc, ImageDim, LayoutError, SampleLayout};
use starquake_memory::{PageRange, WatchTicket, WriteWatch};
use starquake_render::command::{CopyRegion, TextureRegion};
use starquake_render::resource::{Extent3d, Origin3d, TextureDimension};
use starquake_render::resource::{Swizzle, ViewDimension};
use starquake_render::{
    Backend, Caps, Command, CommandList, FenceValue, FormatFeatures, PixelFormat, Readback,
    ReadbackId, RenderError, ResourceId, TextureDesc, TextureId, TextureUsage, ViewDesc, ViewId,
};
use thiserror::Error;

pub use format::{FORMAT_TABLE, FormatRow, pixel_format};
pub use header::{
    CUBE_FACES, HEADER_WORDS, HeaderError, ImageInfo, MemoryLayout, TextureHeader, ViewInfo,
    gob_for,
};

use super::buffer::{STREAMED_AFTER_FAULTS, collection, eviction_limits};
use super::readback::ReadbackIds;
use crate::interface::VmUpdate;
use crate::memory::{GpuFault, GpuMemory};
use transfer::LevelShape;

/// The most views one image keeps (module docs, "Divergences and guesses": a guess).
pub const MAX_VIEWS_PER_IMAGE: usize = 32;

/// The usage of an image's texture on a device with `caps`: copies always, sampling and render
/// targets when the device can do them with `format` (module docs, "Images").
pub fn texture_usage(caps: &Caps, format: PixelFormat) -> TextureUsage {
    let features = caps.formats.get(format);
    let mut usage = TextureUsage::COPY_SRC.union(TextureUsage::COPY_DST);
    if features.contains(FormatFeatures::SAMPLE) {
        usage = usage.union(TextureUsage::SAMPLED);
    }
    if features.contains(FormatFeatures::RENDER) {
        usage = usage.union(TextureUsage::RENDER_TARGET);
    }
    usage
}

/// Why an image's guest bytes cannot be moved to or from its texture (module docs, "Images").
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Unconvertible {
    /// A multisampled image: the backend cannot upload into one.
    Multisampled,
    /// A block-linear image of a GOB the layout crate cannot copy.
    Gob(GobType),
}

/// Why [`TextureCache::texture`] or [`TextureCache::render_target`] refused.
///
/// For a new image the checks run in this order: `Layout` (the layout crate lays the image out),
/// `OutOfRange`, `NotRenderable` (render targets), `OverBudget`, then the backend makes the
/// texture (`Render`) and the upload is prepared level by level and layer by layer, each piece
/// asking for its staging slice (`StagingTooSmall` when one row does not fit, else `Render`)
/// after reading its level's bytes (`Fault`). For an image already held, `NotRenderable`, then
/// the upload if one is due; then the view (`TooManyViews`, `Render`). Pinned by
/// `tests::checks_run_in_their_documented_order`.
///
/// Nothing changes on any of them: no image, watch request, live texture or view, page state or
/// command is left behind (a texture made for the use is destroyed at once, being in no list),
/// and no idle image is evicted.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum TextureError {
    /// The layout crate cannot lay the image out.
    #[error("the image cannot be laid out: {0}")]
    Layout(LayoutError),
    /// The image's last byte is past the 64-bit address space.
    #[error("an image of {size:#x} bytes at {address:#x} runs past the end of the address space")]
    OutOfRange {
        /// Where it starts.
        address: u64,
        /// Its size in bytes.
        size: u64,
    },
    /// A render target of a format the device cannot render to.
    #[error("the device cannot render to {format}")]
    NotRenderable {
        /// The format.
        format: PixelFormat,
    },
    /// The new image would take the cache's live textures past what the device allows, even with
    /// every idle image evicted.
    #[error("the texture cache would hold {bytes:#x} bytes, more than the {allowed:#x} allowed")]
    OverBudget {
        /// The bytes the cache's live textures would take with the new image and every idle image
        /// evicted.
        bytes: u64,
        /// The budget's `allowed`.
        allowed: u64,
    },
    /// One packed row of a level is larger than the largest staging slice.
    #[error("a texture row of {row:#x} bytes is larger than the largest staging slice, {max:#x}")]
    StagingTooSmall {
        /// The row's bytes.
        row: u64,
        /// `Limits::max_staging_size`.
        max: u64,
    },
    /// The image already has [`MAX_VIEWS_PER_IMAGE`] views and the use asks for another.
    #[error("an image already has the most views it can have, {max}")]
    TooManyViews {
        /// [`MAX_VIEWS_PER_IMAGE`].
        max: usize,
    },
    /// The backend refused a texture, a view or a staging slice.
    #[error("the render backend refused: {0}")]
    Render(RenderError),
    /// Reading the guest bytes to upload faulted.
    #[error("reading guest memory for a texture upload: {0}")]
    Fault(GpuFault),
}

/// Why [`TextureCache::finish_write_back`] stopped.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum WriteBackError {
    /// A readback is not there: its list has not completed (or its bytes were already taken).
    #[error("readback {index} of texture {texture:?} is not ready")]
    NotReady {
        /// The image's texture.
        texture: TextureId,
        /// The readback's index in the write-back's list for that image.
        index: usize,
    },
    /// The backend cannot know a readback's bytes (the recording backend, after a draw).
    #[error("the bytes of texture {texture:?} are not known")]
    Unknown {
        /// The image's texture.
        texture: TextureId,
    },
    /// A readback holds a different number of bytes than the level it read.
    #[error("a readback of texture {texture:?} holds {len} bytes, its level {expected}")]
    WrongSize {
        /// The image's texture.
        texture: TextureId,
        /// Bytes it holds.
        len: usize,
        /// Bytes of the packed level.
        expected: u64,
    },
    /// The layout crate refused to tile a level.
    #[error("tiling a written-back level: {0}")]
    Layout(LayoutError),
    /// Reading the guest bytes around a level, or writing them back, faulted.
    #[error("writing an image back to guest memory: {fault}")]
    Fault {
        /// The fault.
        fault: GpuFault,
        /// The guest ranges written before it, in the order written (none when reading
        /// faulted).
        written: Vec<Range<u64>>,
    },
}

/// What [`TextureCache::texture`] returns.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TextureView {
    /// The image's texture.
    pub texture: TextureId,
    /// The view of it.
    pub view: ViewId,
    /// The swizzle the shader applies: the header's when the view could not take it, else the
    /// identity (module docs, "Swizzle").
    pub swizzle: Swizzle,
}

/// What [`TextureCache::render_target`] returns.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RenderTarget {
    /// The image's texture.
    pub texture: TextureId,
    /// A view of its first level and every layer, with the identity swizzle.
    pub view: ViewId,
}

/// What the cache holds for one image, for tests, logs and counters.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TextureEntry {
    /// The image.
    pub info: ImageInfo,
    /// Its texture.
    pub texture: TextureId,
    /// Its guest layout's size in bytes.
    pub bytes: u64,
    /// Views made of it.
    pub views: usize,
    /// Where its contents come from.
    pub source: Source,
    /// How the cache learns of CPU writes to it.
    pub watch: Watch,
    /// Whether the next use uploads it because guest memory changed.
    pub stale: bool,
    /// Calls to [`TextureCache::cpu_writes`] that found it written since its last clean use.
    pub faults: u32,
    /// Whether the host GPU may have written it since its last upload or write-back.
    pub gpu_modified: bool,
}

/// Where an image's contents come from (module docs, "Images").
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Source {
    /// Guest memory: uploaded from it, written back to it.
    Guest,
    /// Nothing: a render target with no guest memory behind it.
    HostOnly,
    /// Nothing: guest memory is behind it, but its bytes cannot be converted.
    Unconvertible(Unconvertible),
}

/// How the cache learns of CPU writes to an image (module docs, "Learning of CPU writes").
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Watch {
    /// Its watch has not been seen armed: every use uploads it, and so does the first use that
    /// sees it armed.
    Unarmed,
    /// Its watch is armed: CPU writes reach the dirty log.
    Armed,
    /// It uploads at every use instead of being watched.
    Streamed,
    /// It is not watched: its contents do not come from guest memory.
    Unwatched,
}

/// A write-back's first half: the readbacks queued, to be finished once their list completes.
#[derive(Debug)]
#[must_use]
pub struct WriteBack {
    images: Vec<PendingImage>,
    skipped: Vec<(TextureId, Skip)>,
}

/// Why [`TextureCache::begin_write_back`] left a GPU-modified image out.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Skip {
    /// It has no guest memory behind it.
    HostOnly,
    /// Its bytes cannot be converted.
    Unconvertible(Unconvertible),
}

impl WriteBack {
    /// The textures of the images whose readbacks were queued, in address order.
    pub fn textures(&self) -> impl Iterator<Item = TextureId> + '_ {
        self.images.iter().map(|i| i.texture)
    }

    /// The GPU-modified images left out, with the reason, in address order.
    pub fn skipped(&self) -> &[(TextureId, Skip)] {
        &self.skipped
    }

    /// Whether nothing was queued.
    pub fn is_empty(&self) -> bool {
        self.images.is_empty()
    }
}

/// One image's readbacks.
#[derive(Debug)]
struct PendingImage {
    address: u64,
    texture: TextureId,
    generation: u64,
    /// One per level per layer, levels outer.
    reads: Vec<ReadbackId>,
}

/// One cached image.
#[derive(Debug)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "four independent facts the buffer cache's entries also keep: stale pages, a write \
              since the last use, GPU modification and use by the open list"
)]
struct Image {
    info: ImageInfo,
    layout: starquake_layout::Image,
    /// The pages its bytes touch.
    pages: PageRange,
    texture: TextureId,
    desc: TextureDesc,
    views: Vec<(ViewDesc, ViewId)>,
    source: Source,
    watch: Watch,
    /// The newest watch request on its pages.
    pending: Option<WatchTicket>,
    stale: bool,
    faults: u32,
    written_since_use: bool,
    gpu_modified: bool,
    /// Moved every time it is marked GPU-modified.
    gpu_generation: u64,
    last_fence: FenceValue,
    in_open: bool,
    stamp: u64,
    frame: u64,
}

impl Image {
    fn bytes(&self) -> u64 {
        self.layout.size_b()
    }

    /// Whether a use now uploads it.
    fn needs_upload(&self, armed_now: bool) -> bool {
        if self.source != Source::Guest {
            return false;
        }
        self.stale
            || self.watch == Watch::Streamed
            || (!self.gpu_modified && (self.watch == Watch::Unarmed || armed_now))
    }

    fn overlaps(&self, range: &Range<u64>) -> bool {
        self.info.address < range.end && range.start < self.pages_end()
    }

    /// The end of the image's own bytes (not rounded to a page).
    fn pages_end(&self) -> u64 {
        self.info.address.saturating_add(self.bytes())
    }

    fn entry(&self) -> TextureEntry {
        TextureEntry {
            info: self.info,
            texture: self.texture,
            bytes: self.bytes(),
            views: self.views.len(),
            source: self.source,
            watch: self.watch,
            stale: self.stale,
            faults: self.faults,
            gpu_modified: self.gpu_modified,
        }
    }
}

/// Where an image sits: its address and its index among the images there.
type Slot = (u64, usize);

/// The guest writes of one image's write-back: each level's address and tiled bytes.
type LevelWrites = Vec<(u64, Vec<u8>)>;

/// A checked new image, before anything changes.
struct NewImage {
    info: ImageInfo,
    layout: starquake_layout::Image,
    pages: PageRange,
    desc: TextureDesc,
    unconvertible: Option<Unconvertible>,
    evict: Vec<TextureId>,
}

/// The texture cache of one GPU address space (see the module docs).
pub struct TextureCache {
    backend: Arc<dyn Backend>,
    watch: Arc<WriteWatch>,
    /// Images by address; at one address, in the order they were made.
    images: BTreeMap<u64, Vec<Image>>,
    /// The sum of the images' bytes.
    bytes: u64,
    /// Textures of images the open list uses.
    open: Vec<TextureId>,
    /// Textures and views of images the open list used, dropped before it was submitted.
    doomed: Vec<ResourceId>,
    /// The bytes of the doomed textures: still live, and counted by the hard cap.
    doomed_bytes: u64,
    last_fence: FenceValue,
    stamp: u64,
    frame: u64,
    generation: u64,
}

impl core::fmt::Debug for TextureCache {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("TextureCache")
            .field("images", &self.images)
            .field("bytes", &self.bytes)
            .field("frame", &self.frame)
            .finish_non_exhaustive()
    }
}

/// Whether a fault means the memory is unmapped.
fn is_unmapped(fault: &GpuFault) -> bool {
    matches!(fault, GpuFault::Unmapped { .. })
}

impl TextureCache {
    /// An empty cache making textures through `backend` and watching through `watch`.
    pub fn new(backend: Arc<dyn Backend>, watch: Arc<WriteWatch>) -> Self {
        Self {
            backend,
            watch,
            images: BTreeMap::new(),
            bytes: 0,
            open: Vec::new(),
            doomed: Vec::new(),
            doomed_bytes: 0,
            last_fence: FenceValue::ZERO,
            stamp: 0,
            frame: 0,
            generation: 0,
        }
    }

    /// The sum of the cached images' bytes.
    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    /// How many images the cache holds.
    pub fn len(&self) -> usize {
        self.images.values().map(Vec::len).sum()
    }

    /// Whether the cache holds no image.
    pub fn is_empty(&self) -> bool {
        self.images.is_empty()
    }

    /// Every image, in address order and, at one address, in the order they were made.
    pub fn entries(&self) -> impl Iterator<Item = TextureEntry> + '_ {
        self.images.values().flatten().map(Image::entry)
    }

    /// The image whose texture is `texture`, if the cache holds it.
    pub fn entry(&self, texture: TextureId) -> Option<TextureEntry> {
        self.slot_of(texture)
            .and_then(|s| self.get(s))
            .map(Image::entry)
    }

    fn get(&self, (address, index): Slot) -> Option<&Image> {
        self.images.get(&address).and_then(|v| v.get(index))
    }

    fn get_mut(&mut self, (address, index): Slot) -> Option<&mut Image> {
        self.images.get_mut(&address).and_then(|v| v.get_mut(index))
    }

    fn slot_of(&self, texture: TextureId) -> Option<Slot> {
        self.images.iter().find_map(|(&address, v)| {
            v.iter()
                .position(|i| i.texture == texture)
                .map(|index| (address, index))
        })
    }

    fn find(&self, info: &ImageInfo) -> Option<Slot> {
        let v = self.images.get(&info.address)?;
        v.iter()
            .position(|i| i.info == *info)
            .map(|index| (info.address, index))
    }

    /// The sampled texture `header` describes, with the uploads it needs appended to `list` (the
    /// open list, submitted later with [`TextureCache::submitted`] reported).
    ///
    /// On an error nothing changes ([`TextureError`] gives the order of checks).
    pub fn texture<M: GpuMemory + ?Sized>(
        &mut self,
        mem: &M,
        header: &TextureHeader,
        list: &mut CommandList,
    ) -> Result<TextureView, TextureError> {
        let in_view = self.backend.caps().view_swizzle;
        let (view_swizzle, swizzle) = if in_view {
            (header.view.swizzle, Swizzle::IDENTITY)
        } else {
            (Swizzle::IDENTITY, header.view.swizzle)
        };
        let view = ViewDesc {
            format: header.image.format,
            dimension: header.view.dimension,
            base_mip: header.view.base_mip,
            mip_count: header.view.mip_count,
            base_layer: 0,
            layer_count: view_layers(&header.image),
            swizzle: view_swizzle,
        };
        let (texture, view) = self.use_image(mem, &header.image, view, false, list)?;
        Ok(TextureView {
            texture,
            view,
            swizzle,
        })
    }

    /// The render target `info` describes: its first level and every layer, uploaded first if
    /// it must be (module docs, "Uses and uploads"), and marked GPU-modified, since the open list
    /// draws into it. A new render target whose memory is unmapped is host-only.
    ///
    /// On an error nothing changes ([`TextureError`] gives the order of checks).
    pub fn render_target<M: GpuMemory + ?Sized>(
        &mut self,
        mem: &M,
        info: &ImageInfo,
        list: &mut CommandList,
    ) -> Result<RenderTarget, TextureError> {
        let layers = view_layers(info);
        let dimension = match info.dim {
            ImageDim::D1 if layers > 1 => ViewDimension::D1Array,
            ImageDim::D1 => ViewDimension::D1,
            ImageDim::D2 if layers > 1 => ViewDimension::D2Array,
            ImageDim::D2 => ViewDimension::D2,
            ImageDim::D3 => ViewDimension::D3,
        };
        let view = ViewDesc {
            format: info.format,
            dimension,
            base_mip: 0,
            mip_count: 1,
            base_layer: 0,
            layer_count: layers,
            swizzle: Swizzle::IDENTITY,
        };
        let (texture, view) = self.use_image(mem, info, view, true, list)?;
        self.gpu_written(texture);
        Ok(RenderTarget { texture, view })
    }

    /// Finds or makes the image, uploads it if due, makes the view, and records the use. Returns
    /// the texture and view.
    fn use_image<M: GpuMemory + ?Sized>(
        &mut self,
        mem: &M,
        info: &ImageInfo,
        view: ViewDesc,
        render: bool,
        list: &mut CommandList,
    ) -> Result<(TextureId, ViewId), TextureError> {
        if let Some(slot) = self.find(info) {
            return self.use_existing(mem, slot, view, render, list);
        }
        self.make(mem, info, view, render, list)
    }

    fn use_existing<M: GpuMemory + ?Sized>(
        &mut self,
        mem: &M,
        slot: Slot,
        view: ViewDesc,
        render: bool,
        list: &mut CommandList,
    ) -> Result<(TextureId, ViewId), TextureError> {
        let Some(image) = self.get(slot) else {
            return Err(TextureError::Render(RenderError::Invalid(
                "the texture cache lost an image",
            )));
        };
        if render && !image.desc.usage.contains(TextureUsage::RENDER_TARGET) {
            return Err(TextureError::NotRenderable {
                format: image.info.format,
            });
        }
        let texture = image.texture;
        // Asked before the guest bytes are read, so a read after `true` comes after the
        // protection took effect.
        let armed_now = image.pending.is_some_and(|t| self.watch.armed(t));
        let upload = image.needs_upload(armed_now);
        let mut commands = Vec::new();
        if upload {
            self.stage(mem, &image.info, &image.layout, texture, &mut commands)?;
        }
        let (view_id, made) = self.view_for(image, view)?;

        // Nothing below can fail.
        let stamp = self.next_stamp();
        let frame = self.frame;
        if let Some(image) = self.images.get_mut(&slot.0).and_then(|v| v.get_mut(slot.1)) {
            if upload {
                // Guest memory is what the texture holds now.
                image.stale = false;
                image.gpu_modified = false;
            } else if image.source != Source::Guest {
                // Never uploaded: a write to its memory changes nothing it holds.
                image.stale = false;
            }
            if armed_now {
                image.watch = Watch::Armed;
                image.pending = None;
            }
            if made {
                image.views.push((view, view_id));
            }
            Self::record_use(image, stamp, frame, &mut self.open);
        }
        list.commands.append(&mut commands);
        Ok((texture, view_id))
    }

    /// The view `desc` of `image`: an existing one, or a new one (`true`) not yet recorded.
    fn view_for(&self, image: &Image, desc: ViewDesc) -> Result<(ViewId, bool), TextureError> {
        if let Some(&(_, id)) = image.views.iter().find(|(d, _)| *d == desc) {
            return Ok((id, false));
        }
        if image.views.len() >= MAX_VIEWS_PER_IMAGE {
            return Err(TextureError::TooManyViews {
                max: MAX_VIEWS_PER_IMAGE,
            });
        }
        let id = self
            .backend
            .create_view(image.texture, &desc)
            .map_err(TextureError::Render)?;
        Ok((id, true))
    }

    /// Checks a new image (module docs, [`TextureError`]'s order) and works out what it needs.
    /// Changes nothing.
    fn plan(&self, info: &ImageInfo, render: bool) -> Result<NewImage, TextureError> {
        let format = info.format;
        let layout_format = starquake_layout::Format::new(
            format.bytes_per_block(),
            format.block_width(),
            format.block_height(),
            1,
        )
        .map_err(TextureError::Layout)?;
        let (tiling, pitch_b) = info.layout.tiling().map_err(TextureError::Layout)?;
        let layout = starquake_layout::Image::new(&ImageDesc {
            dim: info.dim,
            format: layout_format,
            extent_px: starquake_layout::Extent4D::new(
                info.width,
                info.height,
                info.depth,
                info.layers,
            ),
            levels: info.levels,
            sample_layout: info.samples,
            tiling,
            pitch_b,
        })
        .map_err(TextureError::Layout)?;
        let size = layout.size_b();
        let pages = PageRange::covering(info.address, size.max(1)).map_err(|_| {
            TextureError::OutOfRange {
                address: info.address,
                size,
            }
        })?;
        let caps = self.backend.caps();
        let usage = texture_usage(caps, format);
        if render && !usage.contains(TextureUsage::RENDER_TARGET) {
            return Err(TextureError::NotRenderable { format });
        }
        let evict = self.make_room(size)?;
        let dimension = match info.dim {
            ImageDim::D1 => TextureDimension::D1,
            ImageDim::D2 => TextureDimension::D2,
            ImageDim::D3 => TextureDimension::D3,
        };
        let desc = TextureDesc {
            dimension,
            format,
            size: Extent3d {
                width: info.width,
                height: info.height,
                depth_or_layers: if info.dim == ImageDim::D3 {
                    info.depth
                } else {
                    info.layers
                },
            },
            mip_levels: info.levels,
            sample_count: info.samples.samples(),
            usage,
        };
        let unconvertible = if info.samples != SampleLayout::S1x1 {
            Some(Unconvertible::Multisampled)
        } else if tiling.is_tiled() && tiling.gob_type().byte_offset(0, 0).is_err() {
            Some(Unconvertible::Gob(tiling.gob_type()))
        } else {
            None
        };
        Ok(NewImage {
            info: *info,
            layout,
            pages,
            desc,
            unconvertible,
            evict,
        })
    }

    /// The idle images to evict so that a new image of `size` bytes fits the budget's `allowed`
    /// (module docs, "Eviction"): least recently used first, then lowest address. Changes
    /// nothing.
    fn make_room(&self, size: u64) -> Result<Vec<TextureId>, TextureError> {
        let allowed = self.backend.budget().allowed;
        let mut bytes = self
            .bytes
            .saturating_add(self.doomed_bytes)
            .saturating_add(size);
        if bytes <= allowed {
            return Ok(Vec::new());
        }
        let mut idle: Vec<(u64, u64, TextureId, u64)> = self
            .images
            .iter()
            .flat_map(|(&a, v)| v.iter().map(move |i| (a, i)))
            .filter(|(_, i)| !i.in_open && !i.gpu_modified)
            .map(|(a, i)| (i.stamp, a, i.texture, i.bytes()))
            .collect();
        idle.sort_unstable();
        let mut evict = Vec::new();
        for (_, _, texture, image_bytes) in idle {
            if bytes <= allowed {
                break;
            }
            bytes = bytes.saturating_sub(image_bytes);
            evict.push(texture);
        }
        if bytes > allowed {
            return Err(TextureError::OverBudget { bytes, allowed });
        }
        Ok(evict)
    }

    /// Makes the image `info`, uploads it, makes the view and records the use.
    fn make<M: GpuMemory + ?Sized>(
        &mut self,
        mem: &M,
        info: &ImageInfo,
        view: ViewDesc,
        render: bool,
        list: &mut CommandList,
    ) -> Result<(TextureId, ViewId), TextureError> {
        let plan = self.plan(info, render)?;
        let texture = self
            .backend
            .create_texture(&plan.desc)
            .map_err(TextureError::Render)?;
        let destroy_new = |cache: &Self| {
            cache
                .backend
                .destroy(ResourceId::Texture(texture), FenceValue::ZERO);
        };
        let mut commands = Vec::new();
        let mut host_only = false;
        if plan.unconvertible.is_none() {
            match self.stage(mem, &plan.info, &plan.layout, texture, &mut commands) {
                Ok(()) => {}
                Err(TextureError::Fault(fault)) if render && is_unmapped(&fault) => {
                    commands.clear();
                    host_only = true;
                }
                Err(error) => {
                    destroy_new(self);
                    return Err(error);
                }
            }
        }
        let view_id = match self.backend.create_view(texture, &view) {
            Ok(id) => id,
            Err(error) => {
                destroy_new(self);
                return Err(TextureError::Render(error));
            }
        };

        // Nothing below can fail.
        if let Some(reason) = plan.unconvertible {
            tracing::debug!(
                address = info.address,
                ?reason,
                "texture cache: image made without contents"
            );
        }
        for evicted in &plan.evict {
            if let Some(slot) = self.slot_of(*evicted) {
                self.retire(slot);
            }
        }
        let source = match plan.unconvertible {
            Some(reason) => Source::Unconvertible(reason),
            None if host_only => Source::HostOnly,
            None => Source::Guest,
        };
        let watched = source == Source::Guest;
        let pending = watched.then(|| self.watch.request(plan.pages));
        let stamp = self.next_stamp();
        let mut image = Image {
            info: plan.info,
            layout: plan.layout,
            pages: plan.pages,
            texture,
            desc: plan.desc,
            views: vec![(view, view_id)],
            source,
            watch: if watched {
                Watch::Unarmed
            } else {
                Watch::Unwatched
            },
            pending,
            stale: false,
            faults: 0,
            written_since_use: false,
            gpu_modified: false,
            gpu_generation: 0,
            last_fence: self.last_fence,
            in_open: false,
            stamp,
            frame: self.frame,
        };
        Self::record_use(&mut image, stamp, self.frame, &mut self.open);
        self.bytes = self.bytes.saturating_add(image.bytes());
        self.images.entry(info.address).or_default().push(image);
        list.commands.append(&mut commands);
        Ok((texture, view_id))
    }

    /// Reads every level of every layer of `layout` from guest memory, detiles it, and adds the
    /// uploads into `texture` to `commands`, levels outer, then layers, then slices, then rows.
    fn stage<M: GpuMemory + ?Sized>(
        &self,
        mem: &M,
        info: &ImageInfo,
        layout: &starquake_layout::Image,
        texture: TextureId,
        commands: &mut Vec<Command>,
    ) -> Result<(), TextureError> {
        let max = self.backend.caps().limits.max_staging_size;
        let block_height = info.format.block_height();
        for (level, layer, shape) in level_layers(layout) {
            let shape = shape.map_err(TextureError::Layout)?;
            let row = shape.row_b();
            let rows_per_piece = max.checked_div(row).unwrap_or(0);
            if rows_per_piece == 0 {
                return Err(TextureError::StagingTooSmall { row, max });
            }
            let guest = read_level(mem, info.address, layout, level, layer, &shape).map_err(
                |e| match e {
                    LevelError::Layout(e) => TextureError::Layout(e),
                    LevelError::Fault(f) => TextureError::Fault(f),
                },
            )?;
            let rows_total = shape.extent_b.height;
            // At most `rows_total`, which is a `u32`.
            let per_piece = u32::try_from(rows_per_piece).unwrap_or(u32::MAX);
            for z in 0..shape.extent_b.depth {
                let mut y = 0;
                while y < rows_total {
                    let rows = per_piece.min(rows_total - y);
                    // `rows * row <= max`, a `u64`.
                    let size = u64::from(rows) * row;
                    let mut slice = self.backend.staging(size).map_err(TextureError::Render)?;
                    shape
                        .read_rows(&guest, z, y, rows, slice.bytes_mut())
                        .map_err(TextureError::Layout)?;
                    let y_px = y.saturating_mul(block_height);
                    let height = rows
                        .saturating_mul(block_height)
                        .min(shape.extent_px.height.saturating_sub(y_px));
                    commands.push(Command::Upload {
                        from: slice,
                        to: CopyRegion::Texture(TextureRegion {
                            texture,
                            mip: level,
                            // A 3D image has one layer and a layered one one slice.
                            origin: Origin3d {
                                x: 0,
                                y: y_px,
                                z: layer.saturating_add(z),
                            },
                            extent: Extent3d {
                                width: shape.extent_px.width,
                                height,
                                depth_or_layers: 1,
                            },
                        }),
                    });
                    y += rows;
                }
            }
        }
        Ok(())
    }

    fn next_stamp(&mut self) -> u64 {
        // One per use: 2^64 uses cannot happen, and saturating keeps every earlier order.
        self.stamp = self.stamp.saturating_add(1);
        self.stamp
    }

    /// Records a use of `image` in the open list.
    fn record_use(image: &mut Image, stamp: u64, frame: u64, open: &mut Vec<TextureId>) {
        if !image.written_since_use {
            image.faults = 0;
        }
        image.written_since_use = false;
        image.stamp = stamp;
        image.frame = frame;
        if !image.in_open {
            image.in_open = true;
            open.push(image.texture);
        }
    }

    /// Removes the image at `slot` and destroys its texture and views once no list can use them.
    fn retire(&mut self, (address, index): Slot) {
        let Some(v) = self.images.get_mut(&address) else {
            return;
        };
        if index >= v.len() {
            return;
        }
        let image = v.remove(index);
        if v.is_empty() {
            self.images.remove(&address);
        }
        self.bytes = self.bytes.saturating_sub(image.bytes());
        let resources = image
            .views
            .iter()
            .map(|&(_, id)| ResourceId::View(id))
            .chain(core::iter::once(ResourceId::Texture(image.texture)));
        if image.in_open {
            self.doomed.extend(resources);
            self.doomed_bytes = self.doomed_bytes.saturating_add(image.bytes());
        } else {
            for resource in resources {
                self.backend.destroy(resource, image.last_fence);
            }
        }
    }

    /// Slots of the images overlapping `range`, in address order.
    fn overlapping(&self, range: &Range<u64>) -> Vec<Slot> {
        self.images
            .range(..range.end)
            .flat_map(|(&a, v)| {
                v.iter()
                    .enumerate()
                    .filter(|(_, i)| i.overlaps(range))
                    .map(move |(index, _)| (a, index))
            })
            .collect()
    }

    /// One drain of the write watch's dirty log (module docs, "Learning of CPU writes"): every
    /// image overlapping a range turns stale, and counts one fault for the whole call.
    pub fn cpu_writes(&mut self, ranges: &[PageRange]) {
        let mut counted: Vec<TextureId> = Vec::new();
        for range in ranges {
            let span = range.start()..range.end();
            for slot in self.overlapping(&span) {
                let watch = Arc::clone(&self.watch);
                let Some(image) = self.get_mut(slot) else {
                    continue;
                };
                image.stale = true;
                if image.watch == Watch::Streamed || image.source != Source::Guest {
                    continue;
                }
                if !counted.contains(&image.texture) {
                    counted.push(image.texture);
                    image.faults = image.faults.saturating_add(1);
                    image.written_since_use = true;
                }
                if image.faults >= STREAMED_AFTER_FAULTS {
                    image.watch = Watch::Streamed;
                    image.pending = None;
                } else {
                    let lo = range.start().max(image.pages.start());
                    let hi = range.end().min(image.pages.end());
                    if let Ok(pages) = PageRange::new(lo, hi.saturating_sub(lo)) {
                        image.pending = Some(watch.request(pages));
                        image.watch = Watch::Unarmed;
                    }
                }
            }
        }
    }

    /// The back thread itself wrote `range` of guest memory (module docs, "GPU writes,
    /// write-back and unmaps"): every image overlapping it turns stale. An empty range changes
    /// nothing.
    pub fn invalidate(&mut self, range: Range<u64>) {
        if range.is_empty() {
            return;
        }
        for slot in self.overlapping(&range) {
            if let Some(image) = self.get_mut(slot) {
                image.stale = true;
            }
        }
    }

    /// An ordered address-table update has been applied: every image overlapping its range is
    /// dropped, whatever its kind.
    pub fn apply_update(&mut self, update: &VmUpdate) {
        let range = update.range();
        let mut slots = self.overlapping(&(range.start()..range.end()));
        // Highest index first, so earlier indices at one address stay valid.
        slots.reverse();
        for slot in slots {
            self.retire(slot);
        }
    }

    /// The host GPU wrote (or will write, in the open list) the image whose texture is
    /// `texture`: it is GPU-modified from now on. Returns whether the cache holds it.
    pub fn gpu_written(&mut self, texture: TextureId) -> bool {
        let Some(slot) = self.slot_of(texture) else {
            return false;
        };
        self.generation = self.generation.saturating_add(1);
        let generation = self.generation;
        if let Some(image) = self.get_mut(slot) {
            image.gpu_modified = true;
            image.gpu_generation = generation;
        }
        true
    }

    /// Whether a GPU-modified image that can be written back overlaps `range`.
    pub fn needs_write_back(&self, range: Range<u64>) -> bool {
        self.overlapping(&range).into_iter().any(|s| {
            self.get(s)
                .is_some_and(|i| i.gpu_modified && i.source == Source::Guest)
        })
    }

    /// The first half of a write-back (module docs, "GPU writes, write-back and unmaps"): queues
    /// a readback of every level of every layer of each GPU-modified image overlapping `range`
    /// in `list`, the open list.
    pub fn begin_write_back(
        &mut self,
        range: Range<u64>,
        ids: &mut ReadbackIds,
        list: &mut CommandList,
    ) -> WriteBack {
        let mut out = WriteBack {
            images: Vec::new(),
            skipped: Vec::new(),
        };
        let stamp = self.next_stamp();
        let frame = self.frame;
        for slot in self.overlapping(&range) {
            let Some(image) = self.images.get_mut(&slot.0).and_then(|v| v.get_mut(slot.1)) else {
                continue;
            };
            if !image.gpu_modified {
                continue;
            }
            match image.source {
                Source::Guest => {}
                Source::HostOnly => {
                    out.skipped.push((image.texture, Skip::HostOnly));
                    continue;
                }
                Source::Unconvertible(reason) => {
                    out.skipped
                        .push((image.texture, Skip::Unconvertible(reason)));
                    continue;
                }
            }
            let mut reads = Vec::new();
            for level in 0..image.desc.mip_levels {
                let Some(extent) = image.desc.mip_extent(level) else {
                    continue;
                };
                let (slices, layers) = if image.desc.dimension == TextureDimension::D3 {
                    (extent.depth_or_layers, 1)
                } else {
                    (1, extent.depth_or_layers)
                };
                for layer in 0..layers {
                    let id = ids.allocate();
                    list.push(Command::Readback {
                        from: CopyRegion::Texture(TextureRegion {
                            texture: image.texture,
                            mip: level,
                            origin: Origin3d {
                                x: 0,
                                y: 0,
                                z: layer,
                            },
                            extent: Extent3d {
                                width: extent.width,
                                height: extent.height,
                                depth_or_layers: slices,
                            },
                        }),
                        id,
                    });
                    reads.push(id);
                }
            }
            out.images.push(PendingImage {
                address: image.info.address,
                texture: image.texture,
                generation: image.gpu_generation,
                reads,
            });
            Self::record_use(image, stamp, frame, &mut self.open);
        }
        out
    }

    /// The second half of a write-back, once the list holding its readbacks has completed:
    /// tiles each image's readbacks into its guest bytes and writes them, in address order.
    /// Returns the guest ranges written.
    ///
    /// Every readback is taken and every level tiled before the first write, so `NotReady`,
    /// `Unknown`, `WrongSize`, `Layout` and a `Fault` while reading the bytes around a level
    /// leave guest memory and every image as they were (the write-back is abandoned, and its
    /// images stay GPU-modified for a new one). A `Fault` while writing stops there: the images
    /// written before it are written back and no longer GPU-modified; the one that faulted keeps
    /// the levels written before the fault in guest memory and stays GPU-modified, as do those
    /// after it. The error carries the ranges written. Pinned by
    /// `tests::a_write_back_that_faults_while_writing_keeps_what_it_wrote`.
    pub fn finish_write_back<M: GpuMemory + ?Sized>(
        &mut self,
        mem: &M,
        write_back: WriteBack,
    ) -> Result<Vec<Range<u64>>, WriteBackError> {
        // Every fallible step but the writes: take the readbacks, read and tile each level.
        let WriteBack { images, .. } = write_back;
        let mut prepared: Vec<(TextureId, LevelWrites)> = Vec::new();
        for pending in images {
            let mut bytes = Vec::with_capacity(pending.reads.len());
            for (index, id) in pending.reads.iter().enumerate() {
                match self.backend.take_readback(*id) {
                    Some(Readback::Bytes(b)) => bytes.push(Some(b)),
                    Some(Readback::Unknown) => bytes.push(None),
                    None => {
                        return Err(WriteBackError::NotReady {
                            texture: pending.texture,
                            index,
                        });
                    }
                }
            }
            let current = self
                .slot_of(pending.texture)
                .and_then(|s| self.get(s))
                .filter(|i| {
                    i.gpu_modified
                        && i.gpu_generation == pending.generation
                        && i.info.address == pending.address
                });
            let Some(image) = current else {
                // Dropped, or drawn into again since the readbacks were queued.
                continue;
            };
            let mut writes = Vec::with_capacity(bytes.len());
            for ((level, layer, shape), packed) in level_layers(&image.layout).zip(bytes) {
                let shape = shape.map_err(WriteBackError::Layout)?;
                let packed = packed.ok_or(WriteBackError::Unknown {
                    texture: image.texture,
                })?;
                let expected = shape.packed_len().map_err(WriteBackError::Layout)?;
                if u64::try_from(packed.len()).ok() != Some(expected) {
                    return Err(WriteBackError::WrongSize {
                        texture: image.texture,
                        len: packed.len(),
                        expected,
                    });
                }
                let va = level_address(image.info.address, &image.layout, level, layer)
                    .map_err(WriteBackError::Layout)?;
                let mut guest =
                    read_level(mem, image.info.address, &image.layout, level, layer, &shape)
                        .map_err(|e| match e {
                            LevelError::Layout(e) => WriteBackError::Layout(e),
                            LevelError::Fault(fault) => WriteBackError::Fault {
                                fault,
                                written: Vec::new(),
                            },
                        })?;
                shape
                    .write_level(&packed, &mut guest)
                    .map_err(WriteBackError::Layout)?;
                writes.push((va, guest));
            }
            prepared.push((image.texture, writes));
        }

        // The writes, in address order; a fault stops them.
        let mut written: Vec<Range<u64>> = Vec::new();
        let mut fault = None;
        let mut done: Vec<TextureId> = Vec::new();
        'images: for (texture, writes) in &prepared {
            for (va, bytes) in writes {
                if let Err(f) = mem.write(*va, bytes) {
                    fault = Some(f);
                    break 'images;
                }
                // `read_level` read the same extent, so it does not overflow.
                let end = va.saturating_add(u64::try_from(bytes.len()).unwrap_or(u64::MAX));
                written.push(*va..end);
            }
            done.push(*texture);
        }
        for texture in &done {
            if let Some(slot) = self.slot_of(*texture)
                && let Some(image) = self.get_mut(slot)
            {
                image.gpu_modified = false;
                image.stale = false;
            }
        }
        for range in &written {
            for slot in self.overlapping(range) {
                if let Some(image) = self.get_mut(slot)
                    && !done.contains(&image.texture)
                {
                    image.stale = true;
                }
            }
        }
        match fault {
            Some(fault) => Err(WriteBackError::Fault { fault, written }),
            None => Ok(written),
        }
    }

    /// The open list was submitted and signals `fence`: every image it used now lasts until
    /// `fence`, and the textures and views of those dropped meanwhile are destroyed after it.
    pub fn submitted(&mut self, fence: FenceValue) {
        self.last_fence = self.last_fence.max(fence);
        for texture in core::mem::take(&mut self.open) {
            if let Some(slot) = self.slot_of(texture)
                && let Some(image) = self.get_mut(slot)
                && image.in_open
            {
                image.in_open = false;
                image.last_fence = fence;
            }
        }
        for resource in core::mem::take(&mut self.doomed) {
            self.backend.destroy(resource, fence);
        }
        self.doomed_bytes = 0;
    }

    /// Ends a frame (module docs, "Eviction"): runs the collector if the cache's bytes reach the
    /// limits from the backend's budget, then counts the frame. Returns how many images it
    /// evicted.
    pub fn end_frame(&mut self) -> usize {
        let limits = eviction_limits(self.backend.budget().allowed);
        let mut evicted = 0;
        if let Some(run) = collection(self.bytes, limits) {
            let mut old: Vec<(u64, u64, TextureId)> = self
                .images
                .iter()
                .flat_map(|(&a, v)| v.iter().map(move |i| (a, i)))
                .filter(|(_, i)| {
                    !i.in_open
                        && !i.gpu_modified
                        && self
                            .frame
                            .checked_sub(i.frame)
                            .is_some_and(|age| age >= run.after_frames)
                })
                .map(|(a, i)| (i.stamp, a, i.texture))
                .collect();
            old.sort_unstable();
            for (_, _, texture) in old.into_iter().take(run.at_most) {
                if let Some(slot) = self.slot_of(texture) {
                    self.retire(slot);
                    evicted += 1;
                }
            }
        }
        self.frame = self.frame.saturating_add(1);
        evicted
    }
}

/// The layers a view of the whole image covers: the image's layers, or 1 for a 3D image.
fn view_layers(info: &ImageInfo) -> u32 {
    if info.dim == ImageDim::D3 {
        1
    } else {
        info.layers
    }
}

/// Every level of every layer of `layout`, levels outer, with its shape.
fn level_layers(
    layout: &starquake_layout::Image,
) -> impl Iterator<Item = (u32, u32, Result<LevelShape, LayoutError>)> + '_ {
    let levels = u32::try_from(layout.levels().len()).unwrap_or(u32::MAX);
    let layers = layout.extent_px().array_len;
    (0..levels).flat_map(move |level| {
        let shape = LevelShape::of(layout, level);
        (0..layers).map(move |layer| (level, layer, shape))
    })
}

/// Why a level's guest bytes could not be read.
enum LevelError {
    Layout(LayoutError),
    Fault(GpuFault),
}

/// Where level `level` of layer `layer` of the image at `address` starts in guest memory.
fn level_address(
    address: u64,
    layout: &starquake_layout::Image,
    level: u32,
    layer: u32,
) -> Result<u64, LayoutError> {
    let offset = layout.level_layer_offset_b(level, layer)?;
    address.checked_add(offset).ok_or(LayoutError::Overflow)
}

/// The guest bytes of level `level` of layer `layer` of the image at `address`.
fn read_level<M: GpuMemory + ?Sized>(
    mem: &M,
    address: u64,
    layout: &starquake_layout::Image,
    level: u32,
    layer: u32,
    shape: &LevelShape,
) -> Result<Vec<u8>, LevelError> {
    let va = level_address(address, layout, level, layer).map_err(LevelError::Layout)?;
    let len =
        usize::try_from(shape.guest_len).map_err(|_| LevelError::Layout(LayoutError::Overflow))?;
    let mut guest = vec![0; len];
    mem.read(va, &mut guest).map_err(LevelError::Fault)?;
    Ok(guest)
}

impl Drop for TextureCache {
    fn drop(&mut self) {
        let after = self.last_fence;
        for image in core::mem::take(&mut self.images).into_values().flatten() {
            for (_, view) in &image.views {
                self.backend.destroy(ResourceId::View(*view), after);
            }
            self.backend
                .destroy(ResourceId::Texture(image.texture), after);
        }
        for resource in core::mem::take(&mut self.doomed) {
            self.backend.destroy(resource, after);
        }
    }
}
