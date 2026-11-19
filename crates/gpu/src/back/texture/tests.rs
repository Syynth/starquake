//! Tests of the texture cache, with the test GPU memory, the real `WriteWatch` driven by its fake
//! kernel, and the recording backend.

mod header;

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use proptest::prelude::*;
use proptest::test_runner::{Config, RngSeed};
use starquake_layout::{
    Extent4D, ImageDesc, ImageDim, LinearLayout, Offset4D, SampleLayout, tiled_to_linear,
};
use starquake_memory::{
    FakeWatchKernel, FlatMemory, GuestMemory, PAGE_SIZE, PageRange, WriteWatch,
};
use starquake_render::caps::{DynamicState, StageSupport};
use starquake_render::command::{CopyRegion, TextureRegion};
use starquake_render::resource::{Extent3d, Origin3d, SwizzleSource};
use starquake_render::{
    Backend, Caps, Command, CommandList, Features, FenceValue, FormatFeatures, FormatTable, Limits,
    PixelFormat, Readback, ReadbackId, RecordingBackend, SampleCounts, TextureId, TextureUsage,
};

use super::*;
use crate::back::buffer::{EVICT_AFTER_FRAMES, STREAMED_AFTER_FAULTS};
use crate::back::readback::ReadbackIds;
use crate::classes::MwField;
use crate::classes::clc797tex as tex;
use crate::interface::{Kind, RegionId, VmUpdate};
use crate::memory::{GpuFault, GpuMemory, TestGpuMemory};

/// Where the test memory starts, at the same GPU and CPU address (row 30).
const BASE: u64 = 0x8000_0000;
/// Pages of test memory, each mapped on its own so a test can unmap one.
const PAGES: u64 = 64;
/// The budget most rigs report: large enough that nothing is collected or refused.
const LARGE_BUDGET: u64 = 1 << 40;
/// The test image: 64 by 32 texels of `A8B8G8R8_UNORM` in one-GOB blocks.
const WIDTH: u32 = 64;
const HEIGHT: u32 = 32;
/// Its guest bytes: 256-byte rows are four GOBs across, 32 rows four GOBs down, and the image is
/// padded to the layout crate's `MIN_IMAGE_ALIGN_B`.
const IMAGE_PAGES: u64 = 2;

type Memory = TestGpuMemory<FlatMemory>;

fn page(n: u64) -> u64 {
    BASE + n * PAGE_SIZE
}

fn pages(first: u64, count: u64) -> PageRange {
    PageRange::new(page(first), count * PAGE_SIZE).unwrap()
}

// --- Header building ---

/// A texture header under construction, set field by field through the header's bit ranges.
#[derive(Clone, Copy, Debug)]
struct Build([u32; HEADER_WORDS]);

impl Build {
    fn new() -> Self {
        Self([0; HEADER_WORDS])
    }

    fn set(mut self, field: MwField, value: u32) -> Self {
        field.insert(&mut self.0, value).unwrap();
        self
    }

    /// The four data types, three bits each from bit `low` (as every form lays them out).
    fn types(self, low: u32, value: u32) -> Self {
        (0..4).fold(self, |b, i| {
            b.set(MwField::new(low + 3 * i + 2, low + 3 * i), value)
        })
    }

    /// The four swizzle sources, from bit `low`, each from its own channel.
    fn identity(self, low: u32) -> Self {
        let sources = [
            tex::TEXHEAD_BL_X_SOURCE_IN_R,
            tex::TEXHEAD_BL_X_SOURCE_IN_G,
            tex::TEXHEAD_BL_X_SOURCE_IN_B,
            tex::TEXHEAD_BL_X_SOURCE_IN_A,
        ];
        (0..4).fold(self, |b, i| {
            b.set(
                MwField::new(low + 3 * i + 2, low + 3 * i),
                sources[usize::try_from(i).unwrap()],
            )
        })
    }

    fn words(self) -> [u32; HEADER_WORDS] {
        self.0
    }
}

/// A block-linear header of the test image at `address`: 2D, one level, one-GOB blocks,
/// identity swizzle.
fn bl_header(address: u64) -> Build {
    use tex::*;
    Build::new()
        .set(
            TEXHEAD_BL_HEADER_VERSION,
            TEXHEAD_BL_HEADER_VERSION_SELECT_BLOCKLINEAR,
        )
        .set(TEXHEAD_BL_COMPONENTS, TEXHEAD_BL_COMPONENTS_SIZES_A8B8G8R8)
        .types(TEXHEAD_BL_R_DATA_TYPE.low(), super::format::UNORM)
        .identity(TEXHEAD_BL_X_SOURCE.low())
        .set(
            TEXHEAD_BL_ADDRESS_BITS31TO9,
            u32::try_from((address >> 9) & 0x7F_FFFF).unwrap(),
        )
        .set(
            TEXHEAD_BL_ADDRESS_BITS48TO32,
            u32::try_from(address >> 32).unwrap(),
        )
        .set(TEXHEAD_BL_WIDTH_MINUS_ONE, WIDTH - 1)
        .set(TEXHEAD_BL_HEIGHT_MINUS_ONE, HEIGHT - 1)
        .set(TEXHEAD_BL_TEXTURE_TYPE, TEXHEAD_BL_TEXTURE_TYPE_TWO_D)
}

fn header(address: u64) -> TextureHeader {
    TextureHeader::decode(&bl_header(address).words()).unwrap()
}

fn decode(build: Build) -> TextureHeader {
    TextureHeader::decode(&build.words()).unwrap()
}

// --- The rig ---

fn caps(max_staging: u64, view_swizzle: bool, formats: FormatTable) -> Caps {
    Caps {
        limits: Limits {
            max_buffer_size: 1 << 30,
            max_staging_size: max_staging,
            max_texture_dimension_1d: 4096,
            max_texture_dimension_2d: 4096,
            max_texture_dimension_3d: 256,
            max_texture_array_layers: 64,
            max_sampler_anisotropy: 16,
            max_vertex_attributes: 16,
            max_vertex_buffers: 8,
            max_vertex_buffer_stride: 2048,
            max_color_attachments: 8,
            max_viewports: 16,
            max_bind_groups: 4,
            max_bindings_per_bind_group: 16,
            max_compute_workgroups_per_dimension: 65_535,
            max_uniform_buffers_per_stage: 12,
            max_storage_buffers_per_stage: 8,
            max_sampled_textures_per_stage: 16,
            max_storage_textures_per_stage: 4,
            max_samplers_per_stage: 16,
        },
        features: Features::ALL,
        stages: StageSupport::ALL,
        dynamic_state: DynamicState::ALL,
        sample_counts: SampleCounts::ALL,
        view_swizzle,
        formats,
    }
}

/// Every byte of the test memory distinct from its neighbours, across pages too.
fn memory() -> Memory {
    let size = usize::try_from(PAGES * PAGE_SIZE).unwrap();
    let mem = TestGpuMemory::new(FlatMemory::new(BASE, size));
    for n in 0..PAGES {
        mem.map(pages(n, 1), page(n)).unwrap();
    }
    let bytes: Vec<u8> = (0..PAGES * PAGE_SIZE)
        .map(|o| (o.wrapping_mul(7).wrapping_add(o / 251) & 0xff) as u8)
        .collect();
    mem.guest().write(BASE, &bytes).unwrap();
    mem
}

struct Rig {
    backend: Arc<RecordingBackend>,
    watch: Arc<WriteWatch>,
    kernel: FakeWatchKernel,
    mem: Memory,
    cache: TextureCache,
    list: CommandList,
    ids: ReadbackIds,
}

impl Rig {
    fn new() -> Self {
        Self::with(
            caps(1 << 20, false, FormatTable::uniform(FormatFeatures::ALL)),
            LARGE_BUDGET,
        )
    }

    fn with(caps: Caps, allowed: u64) -> Self {
        let backend = Arc::new(RecordingBackend::new(caps, allowed));
        let watch = Arc::new(WriteWatch::new());
        let cache = TextureCache::new(Arc::<RecordingBackend>::clone(&backend), Arc::clone(&watch));
        Self {
            kernel: FakeWatchKernel::new(Arc::clone(&watch)),
            backend,
            watch,
            mem: memory(),
            cache,
            list: CommandList::new(),
            // Readback ids far from the ones `host_bytes` uses.
            ids: ReadbackIds::after(1 << 40),
        }
    }

    fn texture(&mut self, header: &TextureHeader) -> Result<TextureView, TextureError> {
        self.cache.texture(&self.mem, header, &mut self.list)
    }

    fn target(&mut self, info: &ImageInfo) -> Result<RenderTarget, TextureError> {
        self.cache.render_target(&self.mem, info, &mut self.list)
    }

    /// Uses `header` and returns the uploads that use added.
    fn uploads_of(&mut self, header: &TextureHeader) -> Vec<Upload> {
        let before = self.list.commands.len();
        self.texture(header).unwrap();
        uploads(&self.list.commands[before..])
    }

    fn submit(&mut self) -> FenceValue {
        let list = core::mem::take(&mut self.list);
        let fence = self.backend.submit(list).unwrap();
        self.cache.submitted(fence);
        fence
    }

    /// The texels of `region`, read back through a list of its own (after submitting the open
    /// list, so its uploads land first).
    fn host_bytes(&mut self, region: TextureRegion) -> Vec<u8> {
        self.submit();
        let id = self.ids.allocate();
        let mut list = CommandList::new();
        list.push(Command::Readback {
            from: CopyRegion::Texture(region),
            id,
        });
        self.backend.submit(list).unwrap();
        match self.backend.take_readback(id) {
            Some(Readback::Bytes(bytes)) => bytes,
            other => unreachable!("readback {other:?}"),
        }
    }

    fn guest_bytes(&self, va: u64, len: u64) -> Vec<u8> {
        let mut out = vec![0; usize::try_from(len).unwrap()];
        self.mem.guest().read(va, &mut out).unwrap();
        out
    }

    /// A guest CPU store: the bytes change, and each watched page it touches faults once.
    fn cpu_write(&self, va: u64, data: &[u8]) {
        self.mem.guest().write(va, data).unwrap();
        self.kernel
            .cpu_write(va, u64::try_from(data.len()).unwrap())
            .unwrap();
    }

    /// One drain of the dirty log, handed to the cache as the executor would.
    fn drain(&mut self) -> Vec<PageRange> {
        let mut dirty = Vec::new();
        self.watch.take_dirty(&mut dirty);
        self.cache.cpu_writes(&dirty);
        dirty
    }

    fn enter(&mut self) -> usize {
        self.kernel.enter().unwrap()
    }

    fn entry(&self, texture: TextureId) -> TextureEntry {
        self.cache.entry(texture).unwrap()
    }

    fn log(&self) -> Vec<String> {
        self.backend.snapshot().lines().map(str::to_owned).collect()
    }

    /// Uploads `bytes` (packed texels) into `region` as the host GPU writing it would.
    fn host_write(&mut self, region: TextureRegion, bytes: &[u8]) {
        let mut slice = self
            .backend
            .staging(u64::try_from(bytes.len()).unwrap())
            .unwrap();
        slice.bytes_mut().copy_from_slice(bytes);
        self.list.push(Command::Upload {
            from: slice,
            to: CopyRegion::Texture(region),
        });
    }

    fn write_back(&mut self, range: Range<u64>) -> Result<Vec<Range<u64>>, WriteBackError> {
        let wb = self
            .cache
            .begin_write_back(range, &mut self.ids, &mut self.list);
        self.submit();
        self.cache.finish_write_back(&self.mem, wb)
    }
}

/// One upload: its texture, level, origin, extent and staging length.
type Upload = (TextureId, u32, (u32, u32, u32), (u32, u32, u32), usize);

fn uploads(commands: &[Command]) -> Vec<Upload> {
    commands
        .iter()
        .filter_map(|c| match c {
            Command::Upload {
                from,
                to: CopyRegion::Texture(r),
            } => Some((
                r.texture,
                r.mip,
                (r.origin.x, r.origin.y, r.origin.z),
                (r.extent.width, r.extent.height, r.extent.depth_or_layers),
                from.len(),
            )),
            _ => None,
        })
        .collect()
}

/// The whole of level `mip`, layer `layer` of `texture` with `extent`.
fn region(texture: TextureId, mip: u32, layer: u32, extent: (u32, u32, u32)) -> TextureRegion {
    TextureRegion {
        texture,
        mip,
        origin: Origin3d {
            x: 0,
            y: 0,
            z: layer,
        },
        extent: Extent3d {
            width: extent.0,
            height: extent.1,
            depth_or_layers: extent.2,
        },
    }
}

/// G5's own answer: level `level` of layer `layer` of `info`, untiled from guest memory by the
/// layout crate, packed. The oracle the cache's uploads and write-backs are checked against.
fn g5_untile(mem: &Memory, info: &ImageInfo, level: u32, layer: u32) -> Vec<u8> {
    let format = starquake_layout::Format::new(
        info.format.bytes_per_block(),
        info.format.block_width(),
        info.format.block_height(),
        1,
    )
    .unwrap();
    let (tiling, pitch_b) = info.layout.tiling().unwrap();
    let image = starquake_layout::Image::new(&ImageDesc {
        dim: info.dim,
        format,
        extent_px: Extent4D::new(info.width, info.height, info.depth, info.layers),
        levels: info.levels,
        sample_layout: info.samples,
        tiling,
        pitch_b,
    })
    .unwrap();
    let mut extent = image.level_extent_b(level).unwrap();
    extent.array_len = 1;
    let level_tiling = image.levels()[usize::try_from(level).unwrap()].tiling;
    let mut tiled = vec![0; usize::try_from(image.level_layer_size_b(level).unwrap()).unwrap()];
    mem.guest()
        .read(
            info.address + image.level_layer_offset_b(level, layer).unwrap(),
            &mut tiled,
        )
        .unwrap();
    let row = usize::try_from(extent.width).unwrap();
    let rows = usize::try_from(extent.height).unwrap();
    let mut linear = vec![0; row * rows * usize::try_from(extent.depth).unwrap()];
    tiled_to_linear(
        &mut linear,
        LinearLayout {
            row_stride_b: row,
            plane_stride_b: row * rows,
        },
        &tiled,
        extent,
        Offset4D::new(0, 0, 0, 0),
        extent,
        &level_tiling,
    )
    .unwrap();
    linear
}

/// Distinct bytes from `seed`, `len` of them.
fn pattern(seed: u8, len: usize) -> Vec<u8> {
    (0..len)
        .map(|i| {
            let low = u8::try_from(i & 0xff).unwrap();
            let high = u8::try_from((i >> 8) & 0xff).unwrap();
            low.wrapping_mul(31).wrapping_add(seed).wrapping_add(high)
        })
        .collect()
}

/// Every byte of the test image, packed.
const IMAGE_BYTES: usize = (WIDTH * HEIGHT * 4) as usize;

// --- Identity ---

#[test]
fn the_same_header_twice_is_one_image() {
    let mut rig = Rig::new();
    let a = rig.texture(&header(page(0))).unwrap();
    let b = rig.texture(&header(page(0))).unwrap();
    assert_eq!(a, b);
    assert_eq!(rig.cache.len(), 1);
    let log = rig.log();
    assert_eq!(log.iter().filter(|l| l.starts_with("create t")).count(), 1);
    assert_eq!(log.iter().filter(|l| l.starts_with("create v")).count(), 1);
    assert_eq!(rig.entry(a.texture).bytes, IMAGE_PAGES * PAGE_SIZE);
}

#[test]
fn a_different_layout_at_the_same_address_is_another_image() {
    let mut rig = Rig::new();
    let a = rig.texture(&header(page(0))).unwrap();
    let taller = decode(bl_header(page(0)).set(tex::TEXHEAD_BL_GOBS_PER_BLOCK_HEIGHT, 1));
    let b = rig.texture(&taller).unwrap();
    assert_ne!(a.texture, b.texture);
    assert_eq!(rig.cache.len(), 2);
    // The same layout at the next page is another image too.
    let c = rig.texture(&header(page(1))).unwrap();
    assert_ne!(c.texture, a.texture);
    assert_eq!(rig.cache.len(), 3);
}

#[test]
fn an_unmapped_render_target_is_found_again_by_its_address() {
    let mut rig = Rig::new();
    for n in 40..40 + IMAGE_PAGES {
        rig.mem.unmap(pages(n, 1)).unwrap();
    }
    let info = header(page(40)).image;
    let first = rig.target(&info).unwrap();
    let second = rig.target(&info).unwrap();
    assert_eq!(first, second);
    assert_eq!(rig.cache.len(), 1);
    assert_eq!(uploads(&rig.list.commands), Vec::<Upload>::new());
    let entry = rig.entry(first.texture);
    assert_eq!(entry.source, Source::HostOnly);
    assert_eq!(entry.watch, Watch::Unwatched);
    assert!(!rig.watch.has_requests());
    assert!(entry.gpu_modified);
    // A sampled use of the same image finds it, and uploads nothing.
    let sampled = rig.texture(&header(page(40))).unwrap();
    assert_eq!(sampled.texture, first.texture);
    assert_eq!(uploads(&rig.list.commands), Vec::<Upload>::new());
    // It is never written back.
    assert!(!rig.cache.needs_write_back(page(40)..page(42)));
    let wb = rig
        .cache
        .begin_write_back(page(40)..page(42), &mut rig.ids, &mut rig.list);
    assert!(wb.is_empty());
    assert_eq!(wb.skipped(), [(first.texture, Skip::HostOnly)]);
}

#[test]
fn a_sampled_texture_at_an_unmapped_address_is_an_error() {
    let mut rig = Rig::new();
    rig.mem.unmap(pages(41, 1)).unwrap();
    assert_eq!(
        rig.texture(&header(page(40))),
        Err(TextureError::Fault(GpuFault::Unmapped { va: page(41) }))
    );
    assert!(rig.cache.is_empty());
    assert_eq!(rig.list.commands, Vec::<Command>::new());
    assert!(!rig.watch.has_requests());
    // The texture made for the use is gone.
    assert!(
        rig.log()
            .iter()
            .any(|l| l == "destroy t0 after fence 0: now")
    );
    // A render target there with only part of its memory unmapped is host-only too.
    let rt = rig.target(&header(page(40)).image).unwrap();
    assert_eq!(rig.entry(rt.texture).source, Source::HostOnly);
}

// --- Uploads ---

#[test]
fn uploads_are_detiled_through_the_layout_crate() {
    let mut rig = Rig::new();
    let h = header(page(0));
    let view = rig.texture(&h).unwrap();
    assert_eq!(
        uploads(&rig.list.commands),
        [(view.texture, 0, (0, 0, 0), (WIDTH, HEIGHT, 1), IMAGE_BYTES)]
    );
    let host = rig.host_bytes(region(view.texture, 0, 0, (WIDTH, HEIGHT, 1)));
    assert_eq!(host, g5_untile(&rig.mem, &h.image, 0, 0));
    // The test memory is not the identity under the layout: the check could fail.
    assert_ne!(
        host,
        rig.guest_bytes(page(0), u64::try_from(IMAGE_BYTES).unwrap())
    );
}

#[test]
fn every_level_of_every_layer_is_uploaded_in_order() {
    let mut rig = Rig::new();
    // A 2D array of 3 layers with 3 levels, in two-GOB blocks.
    let h = decode(
        bl_header(page(0))
            .set(
                tex::TEXHEAD_BL_TEXTURE_TYPE,
                tex::TEXHEAD_BL_TEXTURE_TYPE_TWO_D_ARRAY,
            )
            .set(tex::TEXHEAD_BL_DEPTH_MINUS_ONE, 2)
            .set(tex::TEXHEAD_BL_MAX_MIP_LEVEL, 2)
            .set(tex::TEXHEAD_BL_RES_VIEW_MAX_MIP_LEVEL, 2)
            .set(tex::TEXHEAD_BL_GOBS_PER_BLOCK_HEIGHT, 1),
    );
    let view = rig.texture(&h).unwrap();
    let t = view.texture;
    let mut expected = Vec::new();
    for level in 0..3 {
        let (w, hh) = (WIDTH >> level, HEIGHT >> level);
        for layer in 0..3 {
            expected.push((
                t,
                level,
                (0, 0, layer),
                (w, hh, 1),
                usize::try_from(w * hh * 4).unwrap(),
            ));
        }
    }
    assert_eq!(uploads(&rig.list.commands), expected);
    for level in 0..3 {
        for layer in 0..3 {
            let host = rig.host_bytes(region(
                t,
                level,
                layer,
                (WIDTH >> level, HEIGHT >> level, 1),
            ));
            assert_eq!(
                host,
                g5_untile(&rig.mem, &h.image, level, layer),
                "{level} {layer}"
            );
        }
    }
}

#[test]
fn a_3d_image_uploads_slice_by_slice() {
    let mut rig = Rig::new();
    let h = decode(
        bl_header(page(0))
            .set(
                tex::TEXHEAD_BL_TEXTURE_TYPE,
                tex::TEXHEAD_BL_TEXTURE_TYPE_THREE_D,
            )
            .set(tex::TEXHEAD_BL_WIDTH_MINUS_ONE, 15)
            .set(tex::TEXHEAD_BL_HEIGHT_MINUS_ONE, 15)
            .set(tex::TEXHEAD_BL_DEPTH_MINUS_ONE, 3)
            .set(tex::TEXHEAD_BL_GOBS_PER_BLOCK_DEPTH, 1),
    );
    let view = rig.texture(&h).unwrap();
    let t = view.texture;
    let expected: Vec<Upload> = (0..4)
        .map(|z| (t, 0, (0, 0, z), (16, 16, 1), 16 * 16 * 4))
        .collect();
    assert_eq!(uploads(&rig.list.commands), expected);
    let host = rig.host_bytes(region(t, 0, 0, (16, 16, 4)));
    assert_eq!(host, g5_untile(&rig.mem, &h.image, 0, 0));
}

/// A pitch header of `A8B8G8R8_UNORM` at page 0: 100 by 10 texels, rows `pitch` bytes apart.
fn pitch_header(pitch: u32) -> TextureHeader {
    decode(
        Build::new()
            .set(
                tex::TEXHEAD_PITCH_HEADER_VERSION,
                tex::TEXHEAD_PITCH_HEADER_VERSION_SELECT_PITCH,
            )
            .set(
                tex::TEXHEAD_PITCH_COMPONENTS,
                tex::TEXHEAD_PITCH_COMPONENTS_SIZES_A8B8G8R8,
            )
            .types(tex::TEXHEAD_PITCH_R_DATA_TYPE.low(), super::format::UNORM)
            .identity(tex::TEXHEAD_PITCH_X_SOURCE.low())
            .set(
                tex::TEXHEAD_PITCH_TEXTURE_TYPE,
                tex::TEXHEAD_PITCH_TEXTURE_TYPE_TWO_D,
            )
            .set(
                tex::TEXHEAD_PITCH_ADDRESS_BITS31TO5,
                u32::try_from((page(0) >> 5) & 0x7FF_FFFF).unwrap(),
            )
            .set(tex::TEXHEAD_PITCH_PITCH_BITS20TO5, pitch >> 5)
            .set(tex::TEXHEAD_PITCH_WIDTH_MINUS_ONE, 99)
            .set(tex::TEXHEAD_PITCH_HEIGHT_MINUS_ONE, 9),
    )
}

#[test]
fn a_pitch_image_uploads_its_rows() {
    // Three rows a piece, so the pieces after the first start at a row other than 0.
    let mut rig = Rig::with(
        caps(3 * 400, false, FormatTable::uniform(FormatFeatures::ALL)),
        LARGE_BUDGET,
    );
    let pitch = 512;
    let h = pitch_header(pitch);
    let view = rig.texture(&h).unwrap();
    let pieces: Vec<(u32, u32)> = uploads(&rig.list.commands)
        .iter()
        .map(|u| (u.2.1, u.3.1))
        .collect();
    assert_eq!(pieces, [(0, 3), (3, 3), (6, 3), (9, 1)]);
    let host = rig.host_bytes(region(view.texture, 0, 0, (100, 10, 1)));
    let mut expected = Vec::new();
    for y in 0..10u64 {
        expected.extend(rig.guest_bytes(page(0) + y * u64::from(pitch), 400));
    }
    assert_eq!(host, expected);
}

#[test]
fn an_upload_splits_into_whole_rows_at_the_staging_limit() {
    let row = u64::from(WIDTH) * 4;
    // Four rows a piece.
    let mut rig = Rig::with(
        caps(
            4 * row + row - 1,
            false,
            FormatTable::uniform(FormatFeatures::ALL),
        ),
        LARGE_BUDGET,
    );
    let h = header(page(0));
    let t = rig.texture(&h).unwrap().texture;
    let expected: Vec<Upload> = (0..HEIGHT / 4)
        .map(|i| {
            (
                t,
                0,
                (0, 4 * i, 0),
                (WIDTH, 4, 1),
                usize::try_from(4 * row).unwrap(),
            )
        })
        .collect();
    assert_eq!(uploads(&rig.list.commands), expected);
    let host = rig.host_bytes(region(t, 0, 0, (WIDTH, HEIGHT, 1)));
    assert_eq!(host, g5_untile(&rig.mem, &h.image, 0, 0));
}

/// A block-linear BC1 header at `address`: 16 by 10 texels, so its last row of 4-by-4 blocks is
/// half outside the image.
fn bc1_header(address: u64) -> TextureHeader {
    decode(
        bl_header(address)
            .set(
                tex::TEXHEAD_BL_COMPONENTS,
                tex::TEXHEAD_BL_COMPONENTS_SIZES_DXT1,
            )
            .set(tex::TEXHEAD_BL_WIDTH_MINUS_ONE, 15)
            .set(tex::TEXHEAD_BL_HEIGHT_MINUS_ONE, 9),
    )
}

#[test]
fn a_compressed_upload_clamps_its_last_piece_to_the_image() {
    let h = bc1_header(page(0));
    assert_eq!(h.image.format, PixelFormat::Bc1RgbaUnorm);
    let (block_w, block_h) = (h.image.format.block_width(), h.image.format.block_height());
    // One packed row of blocks a piece.
    let row = u64::from(16 / block_w) * u64::from(h.image.format.bytes_per_block());
    let mut rig = Rig::with(
        caps(row, false, FormatTable::uniform(FormatFeatures::ALL)),
        LARGE_BUDGET,
    );
    let t = rig.texture(&h).unwrap().texture;
    let pieces: Vec<(u32, u32)> = uploads(&rig.list.commands)
        .iter()
        .map(|u| (u.2.1, u.3.1))
        .collect();
    assert_eq!(
        pieces,
        [
            (0, block_h),
            (block_h, block_h),
            (2 * block_h, 10 - 2 * block_h)
        ]
    );
    // The upload and a write-back round-trip through the layout crate.
    let host = rig.host_bytes(region(t, 0, 0, (16, 10, 1)));
    assert_eq!(host, g5_untile(&rig.mem, &h.image, 0, 0));
    let drawn = pattern(11, host.len());
    // Drawn a row of blocks at a time, as the staging limit allows.
    let row_len = usize::try_from(row).unwrap();
    for (&(y, height), bytes) in pieces.iter().zip(drawn.chunks(row_len)) {
        let mut to = region(t, 0, 0, (16, height, 1));
        to.origin.y = y;
        rig.host_write(to, bytes);
    }
    rig.cache.gpu_written(t);
    rig.write_back(page(0)..page(1)).unwrap();
    assert_eq!(g5_untile(&rig.mem, &h.image, 0, 0), drawn);
}

#[test]
fn a_row_larger_than_the_staging_limit_is_refused_at_the_exact_edge() {
    let row = u64::from(WIDTH) * 4;
    let table = || FormatTable::uniform(FormatFeatures::ALL);
    let mut rig = Rig::with(caps(row, false, table()), LARGE_BUDGET);
    let t = rig.texture(&header(page(0))).unwrap().texture;
    assert_eq!(
        uploads(&rig.list.commands).len(),
        usize::try_from(HEIGHT).unwrap()
    );
    assert_eq!(rig.entry(t).bytes, IMAGE_PAGES * PAGE_SIZE);
    let mut rig = Rig::with(caps(row - 1, false, table()), LARGE_BUDGET);
    assert_eq!(
        rig.texture(&header(page(0))),
        Err(TextureError::StagingTooSmall { row, max: row - 1 })
    );
    assert!(rig.cache.is_empty());
}

// --- Learning of CPU writes ---

/// Makes the image at `address` and uses it until a use uploads nothing.
fn settle(rig: &mut Rig, address: u64) -> TextureId {
    let t = rig.texture(&header(address)).unwrap().texture;
    rig.enter();
    rig.texture(&header(address)).unwrap();
    assert_eq!(rig.uploads_of(&header(address)), Vec::<Upload>::new());
    t
}

#[test]
fn the_first_use_after_arming_uploads_the_image_again() {
    let mut rig = Rig::new();
    let h = header(page(0));
    assert_eq!(rig.uploads_of(&h).len(), 1, "new");
    let t = rig.cache.entries().next().unwrap().texture;
    assert_eq!(rig.entry(t).watch, Watch::Unarmed);
    assert_eq!(rig.uploads_of(&h).len(), 1, "untracked");
    assert_eq!(rig.enter(), 1);
    assert_eq!(
        rig.uploads_of(&h).len(),
        1,
        "the first use that sees it armed"
    );
    assert_eq!(rig.entry(t).watch, Watch::Armed);
    assert!(rig.uploads_of(&h).is_empty(), "tracked");
}

#[test]
fn a_cpu_write_invalidates_the_image() {
    let mut rig = Rig::new();
    let t = settle(&mut rig, page(0));
    let neighbour = settle(&mut rig, page(IMAGE_PAGES));
    rig.cpu_write(page(1) + 8, &[0xA5; 64]);
    assert_eq!(rig.drain(), [pages(1, 1)]);
    let entry = rig.entry(t);
    assert!(entry.stale);
    assert_eq!((entry.faults, entry.watch), (1, Watch::Unarmed));
    // The image starting exactly at the written image's end is untouched.
    assert!(!rig.entry(neighbour).stale);
    assert_eq!(rig.entry(neighbour).faults, 0);
    rig.enter();
    let h = header(page(0));
    assert_eq!(rig.uploads_of(&h).len(), 1);
    assert!(!rig.entry(t).stale);
    assert_eq!(rig.uploads_of(&h), Vec::<Upload>::new());
    let host = rig.host_bytes(region(t, 0, 0, (WIDTH, HEIGHT, 1)));
    assert_eq!(host, g5_untile(&rig.mem, &h.image, 0, 0));
    assert_eq!(
        rig.uploads_of(&header(page(IMAGE_PAGES))),
        Vec::<Upload>::new()
    );
}

#[test]
fn an_image_turns_streamed_at_the_fault_threshold_not_before() {
    let mut rig = Rig::new();
    let t = settle(&mut rig, page(0));
    for n in 1..STREAMED_AFTER_FAULTS {
        // Two ranges of one image in one drain count one fault.
        rig.cache.cpu_writes(&[pages(0, 1), pages(1, 1)]);
        assert_eq!(rig.entry(t).faults, n);
        assert_eq!(rig.entry(t).watch, Watch::Unarmed, "{n}");
        rig.enter();
    }
    rig.cpu_write(page(0), &[1]);
    rig.drain();
    assert_eq!(rig.entry(t).watch, Watch::Streamed);
    assert!(!rig.watch.has_requests());
    // Streamed: every use uploads.
    for _ in 0..3 {
        assert_eq!(rig.uploads_of(&header(page(0))).len(), 1);
    }
}

#[test]
fn a_clean_use_resets_the_fault_count() {
    let mut rig = Rig::new();
    let t = settle(&mut rig, page(0));
    rig.cpu_write(page(0), &[1]);
    rig.drain();
    rig.enter();
    // The use that finds the write keeps the count; the next clean use clears it.
    rig.texture(&header(page(0))).unwrap();
    assert_eq!(rig.entry(t).faults, 1);
    rig.texture(&header(page(0))).unwrap();
    assert_eq!(rig.entry(t).faults, 0);
}

#[test]
fn a_back_thread_write_makes_overlapping_images_stale() {
    let mut rig = Rig::new();
    let a = settle(&mut rig, page(0));
    let b = settle(&mut rig, page(IMAGE_PAGES));
    // A range ending exactly at the second image's start.
    rig.cache.invalidate(page(0)..page(IMAGE_PAGES));
    assert!(rig.entry(a).stale);
    assert!(!rig.entry(b).stale, "starts exactly at the range's end");
    rig.texture(&header(page(0))).unwrap();
    assert!(!rig.entry(a).stale);
    rig.cache
        .invalidate(page(IMAGE_PAGES)..page(IMAGE_PAGES) + 1);
    assert!(!rig.entry(a).stale, "ends exactly at the range's start");
    assert!(rig.entry(b).stale);
    // A back-thread write faults nothing: no new watch.
    assert_eq!(rig.entry(b).watch, Watch::Armed);
    rig.cache.invalidate(page(0)..page(0));
    assert!(!rig.entry(a).stale, "an empty range");
    // The last byte of the first image.
    rig.cache
        .invalidate(page(IMAGE_PAGES) - 1..page(IMAGE_PAGES));
    assert!(rig.entry(a).stale);
}

// --- GPU writes ---

#[test]
fn an_untracked_gpu_modified_image_is_not_uploaded() {
    let mut rig = Rig::new();
    let info = header(page(0)).image;
    let rt = rig.target(&info).unwrap();
    assert_eq!(
        uploads(&rig.list.commands).len(),
        1,
        "a new target loads its memory"
    );
    assert!(rig.entry(rt.texture).gpu_modified);
    let before = rig.list.commands.len();
    rig.target(&info).unwrap();
    assert!(
        uploads(&rig.list.commands[before..]).is_empty(),
        "untracked, but GPU-modified"
    );
    // Nor at the first use after arming.
    rig.enter();
    let before = rig.list.commands.len();
    rig.texture(&header(page(0))).unwrap();
    assert_eq!(uploads(&rig.list.commands[before..]), Vec::<Upload>::new());
    assert_eq!(rig.entry(rt.texture).watch, Watch::Armed);
}

#[test]
fn a_cpu_write_to_a_gpu_modified_image_wins() {
    let mut rig = Rig::new();
    let info = header(page(0)).image;
    let rt = rig.target(&info).unwrap();
    rig.enter();
    rig.cpu_write(page(0), &[9; 16]);
    rig.drain();
    let entry = rig.entry(rt.texture);
    assert!(entry.stale && entry.gpu_modified);
    let before = rig.list.commands.len();
    rig.texture(&header(page(0))).unwrap();
    assert_eq!(uploads(&rig.list.commands[before..]).len(), 1);
    let entry = rig.entry(rt.texture);
    assert!(!entry.stale && !entry.gpu_modified);
}

#[test]
fn gpu_written_marks_only_a_held_image() {
    let mut rig = Rig::new();
    let t = rig.texture(&header(page(0))).unwrap().texture;
    assert!(!rig.entry(t).gpu_modified);
    assert!(rig.cache.gpu_written(t));
    assert!(rig.entry(t).gpu_modified);
    assert!(!rig.cache.gpu_written(TextureId::new(77, 0)));
    // An id that differs only in its generation is not the image.
    assert!(
        !rig.cache
            .gpu_written(TextureId::new(t.index(), t.generation() + 1))
    );
}

#[test]
fn write_back_matches_g5_untile() {
    let mut rig = Rig::new();
    let h = header(page(0));
    let t = rig.texture(&h).unwrap().texture;
    let drawn = pattern(3, IMAGE_BYTES);
    rig.host_write(region(t, 0, 0, (WIDTH, HEIGHT, 1)), &drawn);
    assert!(!rig.cache.needs_write_back(page(0)..page(1)));
    rig.cache.gpu_written(t);
    assert!(rig.cache.needs_write_back(page(1)..page(1) + 1));
    assert!(
        !rig.cache
            .needs_write_back(page(IMAGE_PAGES)..page(IMAGE_PAGES + 1))
    );
    let after = rig.guest_bytes(page(IMAGE_PAGES), PAGE_SIZE);
    let written = rig.write_back(page(0)..page(0) + 1).unwrap();
    let whole = page(0)..page(IMAGE_PAGES);
    assert_eq!(written, vec![whole]);
    assert_eq!(g5_untile(&rig.mem, &h.image, 0, 0), drawn);
    assert!(!rig.entry(t).gpu_modified);
    assert_eq!(rig.guest_bytes(page(IMAGE_PAGES), PAGE_SIZE), after);
}

#[test]
fn every_level_of_every_layer_is_written_back() {
    let mut rig = Rig::new();
    let h = decode(
        bl_header(page(0))
            .set(
                tex::TEXHEAD_BL_TEXTURE_TYPE,
                tex::TEXHEAD_BL_TEXTURE_TYPE_TWO_D_ARRAY,
            )
            .set(tex::TEXHEAD_BL_DEPTH_MINUS_ONE, 1)
            .set(tex::TEXHEAD_BL_MAX_MIP_LEVEL, 1)
            .set(tex::TEXHEAD_BL_RES_VIEW_MAX_MIP_LEVEL, 1),
    );
    let t = rig.texture(&h).unwrap().texture;
    let mut drawn = Vec::new();
    let mut seed = 0;
    for level in 0..2 {
        for layer in 0..2 {
            let extent = (WIDTH >> level, HEIGHT >> level, 1);
            let bytes = pattern(seed, usize::try_from(extent.0 * extent.1 * 4).unwrap());
            seed += 40;
            rig.host_write(region(t, level, layer, extent), &bytes);
            drawn.push((level, layer, bytes));
        }
    }
    rig.cache.gpu_written(t);
    rig.write_back(page(0)..page(1)).unwrap();
    for (level, layer, bytes) in drawn {
        assert_eq!(
            g5_untile(&rig.mem, &h.image, level, layer),
            bytes,
            "{level} {layer}"
        );
    }
}

#[test]
fn a_write_back_makes_other_images_over_the_same_memory_stale() {
    let mut rig = Rig::new();
    let a = rig.texture(&header(page(0))).unwrap().texture;
    let other = rig
        .texture(&decode(
            bl_header(page(0)).set(tex::TEXHEAD_BL_GOBS_PER_BLOCK_HEIGHT, 1),
        ))
        .unwrap()
        .texture;
    let neighbour = rig.texture(&header(page(IMAGE_PAGES))).unwrap().texture;
    rig.submit();
    for t in [a, other, neighbour] {
        assert!(!rig.entry(t).stale);
    }
    rig.cache.gpu_written(a);
    rig.write_back(page(0)..page(1)).unwrap();
    assert!(!rig.entry(a).stale);
    assert!(rig.entry(other).stale);
    assert!(!rig.entry(neighbour).stale);
}

#[test]
fn an_image_drawn_into_between_the_halves_is_not_written_back() {
    let mut rig = Rig::new();
    let t = rig.texture(&header(page(0))).unwrap().texture;
    rig.cache.gpu_written(t);
    let before = rig.guest_bytes(page(0), IMAGE_PAGES * PAGE_SIZE);
    let wb = rig
        .cache
        .begin_write_back(page(0)..page(1), &mut rig.ids, &mut rig.list);
    assert_eq!(wb.textures().collect::<Vec<_>>(), [t]);
    // Drawn into again, in the list after the readback.
    rig.host_write(
        region(t, 0, 0, (WIDTH, HEIGHT, 1)),
        &pattern(1, IMAGE_BYTES),
    );
    rig.cache.gpu_written(t);
    rig.submit();
    assert_eq!(rig.cache.finish_write_back(&rig.mem, wb), Ok(Vec::new()));
    assert_eq!(rig.guest_bytes(page(0), IMAGE_PAGES * PAGE_SIZE), before);
    assert!(rig.entry(t).gpu_modified);
    // Unchanged between the halves, it is written back.
    assert_eq!(rig.write_back(page(0)..page(1)).unwrap().len(), 1);
    assert!(!rig.entry(t).gpu_modified);
}

#[test]
fn an_image_dropped_between_the_halves_is_not_written_back() {
    let mut rig = Rig::new();
    let t = rig.texture(&header(page(0))).unwrap().texture;
    rig.cache.gpu_written(t);
    let before = rig.guest_bytes(page(0), IMAGE_PAGES * PAGE_SIZE);
    let wb = rig
        .cache
        .begin_write_back(page(0)..page(1), &mut rig.ids, &mut rig.list);
    rig.cache.apply_update(&VmUpdate::Unmap {
        range: pages(0, IMAGE_PAGES),
    });
    rig.submit();
    assert_eq!(rig.cache.finish_write_back(&rig.mem, wb), Ok(Vec::new()));
    assert_eq!(rig.guest_bytes(page(0), IMAGE_PAGES * PAGE_SIZE), before);
}

#[test]
fn a_write_back_whose_list_has_not_completed_is_not_ready() {
    let mut rig = Rig::new();
    let t = rig.texture(&header(page(0))).unwrap().texture;
    rig.cache.gpu_written(t);
    let wb = rig
        .cache
        .begin_write_back(page(0)..page(1), &mut rig.ids, &mut rig.list);
    // Not submitted: the readback is not there.
    assert_eq!(
        rig.cache.finish_write_back(&rig.mem, wb),
        Err(WriteBackError::NotReady {
            texture: t,
            index: 0
        })
    );
    assert!(rig.entry(t).gpu_modified);
}

/// A memory whose writes fail from the `fail_from`th on (counting from 0).
struct FailingWrites<'a> {
    inner: &'a Memory,
    fail_from: usize,
    writes: AtomicUsize,
}

impl GpuMemory for FailingWrites<'_> {
    fn read(&self, va: u64, buf: &mut [u8]) -> Result<(), GpuFault> {
        self.inner.read(va, buf)
    }

    fn write(&self, va: u64, data: &[u8]) -> Result<(), GpuFault> {
        if self.writes.fetch_add(1, Ordering::Relaxed) >= self.fail_from {
            return Err(GpuFault::Unmapped { va });
        }
        self.inner.write(va, data)
    }

    fn generation(&self) -> u64 {
        self.inner.generation()
    }
}

#[test]
fn a_write_back_that_faults_while_writing_keeps_what_it_wrote() {
    let mut rig = Rig::new();
    let images: Vec<TextureId> = [0, 4, 8]
        .into_iter()
        .map(|n| rig.texture(&header(page(n))).unwrap().texture)
        .collect();
    for (i, &t) in images.iter().enumerate() {
        rig.host_write(
            region(t, 0, 0, (WIDTH, HEIGHT, 1)),
            &pattern(u8::try_from(i).unwrap() * 50, IMAGE_BYTES),
        );
        rig.cache.gpu_written(t);
    }
    let before: Vec<Vec<u8>> = [0, 4, 8]
        .into_iter()
        .map(|n| rig.guest_bytes(page(n), IMAGE_PAGES * PAGE_SIZE))
        .collect();
    let wb = rig
        .cache
        .begin_write_back(page(0)..page(12), &mut rig.ids, &mut rig.list);
    rig.submit();
    let first = page(0)..page(IMAGE_PAGES);
    let failing = FailingWrites {
        inner: &rig.mem,
        fail_from: 1,
        writes: AtomicUsize::new(0),
    };
    assert_eq!(
        rig.cache.finish_write_back(&failing, wb),
        Err(WriteBackError::Fault {
            fault: GpuFault::Unmapped { va: page(4) },
            written: vec![first],
        })
    );
    assert!(!rig.entry(images[0]).gpu_modified);
    assert!(rig.entry(images[1]).gpu_modified);
    assert!(rig.entry(images[2]).gpu_modified);
    assert_ne!(rig.guest_bytes(page(0), IMAGE_PAGES * PAGE_SIZE), before[0]);
    assert_eq!(rig.guest_bytes(page(4), IMAGE_PAGES * PAGE_SIZE), before[1]);
    assert_eq!(rig.guest_bytes(page(8), IMAGE_PAGES * PAGE_SIZE), before[2]);
}

#[test]
fn a_write_back_that_faults_while_reading_writes_nothing() {
    let mut rig = Rig::new();
    let a = rig.texture(&header(page(0))).unwrap().texture;
    let b = rig.texture(&header(page(4))).unwrap().texture;
    rig.cache.gpu_written(a);
    rig.cache.gpu_written(b);
    let before = rig.guest_bytes(page(0), IMAGE_PAGES * PAGE_SIZE);
    let wb = rig
        .cache
        .begin_write_back(page(0)..page(8), &mut rig.ids, &mut rig.list);
    rig.submit();
    rig.mem.unmap(pages(5, 1)).unwrap();
    assert_eq!(
        rig.cache.finish_write_back(&rig.mem, wb),
        Err(WriteBackError::Fault {
            fault: GpuFault::Unmapped { va: page(5) },
            written: Vec::new(),
        })
    );
    assert_eq!(rig.guest_bytes(page(0), IMAGE_PAGES * PAGE_SIZE), before);
    assert!(rig.entry(a).gpu_modified && rig.entry(b).gpu_modified);
}

#[test]
fn a_gpu_modified_image_stale_before_the_write_back_is_left_out() {
    let mut rig = Rig::new();
    let t = settle(&mut rig, page(0));
    rig.cache.gpu_written(t);
    // A newer CPU write: guest memory wins over the host copy.
    rig.cpu_write(page(0) + 8, &[0xA5; 64]);
    rig.drain();
    assert!(rig.entry(t).stale && rig.entry(t).gpu_modified);
    assert!(!rig.cache.needs_write_back(page(0)..page(1)));
    let before = rig.list.commands.len();
    let wb = rig
        .cache
        .begin_write_back(page(0)..page(1), &mut rig.ids, &mut rig.list);
    assert!(wb.is_empty());
    assert_eq!(wb.skipped(), [(t, Skip::Stale)]);
    assert_eq!(rig.list.commands.len(), before, "no readback queued");
    rig.submit();
    let guest = rig.guest_bytes(page(0), IMAGE_PAGES * PAGE_SIZE);
    assert_eq!(rig.cache.finish_write_back(&rig.mem, wb), Ok(Vec::new()));
    assert_eq!(rig.guest_bytes(page(0), IMAGE_PAGES * PAGE_SIZE), guest);
    assert_eq!(rig.guest_bytes(page(0) + 8, 64), [0xA5; 64]);
    assert!(rig.entry(t).stale);
}

#[test]
fn a_gpu_modified_image_turned_stale_between_the_halves_is_not_written_back() {
    let mut rig = Rig::new();
    let t = settle(&mut rig, page(0));
    rig.host_write(
        region(t, 0, 0, (WIDTH, HEIGHT, 1)),
        &pattern(3, IMAGE_BYTES),
    );
    rig.cache.gpu_written(t);
    assert!(rig.cache.needs_write_back(page(0)..page(1)));
    let wb = rig
        .cache
        .begin_write_back(page(0)..page(1), &mut rig.ids, &mut rig.list);
    assert_eq!(wb.textures().collect::<Vec<_>>(), [t]);
    rig.submit();
    // The CPU writes the image after its readback was queued.
    rig.cpu_write(page(0), &[0x5A; 64]);
    rig.drain();
    let guest = rig.guest_bytes(page(0), IMAGE_PAGES * PAGE_SIZE);
    assert_eq!(rig.cache.finish_write_back(&rig.mem, wb), Ok(Vec::new()));
    assert_eq!(rig.guest_bytes(page(0), IMAGE_PAGES * PAGE_SIZE), guest);
    let entry = rig.entry(t);
    assert!(entry.stale && entry.gpu_modified);
    // The next use uploads the CPU's bytes, which ends the GPU modification.
    assert_eq!(rig.uploads_of(&header(page(0))).len(), 1);
    let entry = rig.entry(t);
    assert!(!entry.stale && !entry.gpu_modified);
}

#[test]
fn an_image_a_later_image_overwrites_in_the_same_write_back_turns_stale() {
    let mut rig = Rig::new();
    let first = header(page(0));
    let later = decode(bl_header(page(0)).set(tex::TEXHEAD_BL_GOBS_PER_BLOCK_HEIGHT, 1));
    let a = rig.texture(&first).unwrap().texture;
    let b = rig.texture(&later).unwrap().texture;
    // A third image after both, over memory of its own.
    let c = rig.texture(&header(page(4))).unwrap().texture;
    let drawn = [pattern(10, IMAGE_BYTES), pattern(90, IMAGE_BYTES)];
    rig.host_write(region(a, 0, 0, (WIDTH, HEIGHT, 1)), &drawn[0]);
    rig.host_write(region(b, 0, 0, (WIDTH, HEIGHT, 1)), &drawn[1]);
    for t in [a, b, c] {
        rig.cache.gpu_written(t);
    }
    let wb = rig
        .cache
        .begin_write_back(page(0)..page(8), &mut rig.ids, &mut rig.list);
    assert_eq!(wb.textures().collect::<Vec<_>>(), [a, b, c]);
    rig.submit();
    let written = rig.cache.finish_write_back(&rig.mem, wb).unwrap();
    assert_eq!(
        written.iter().map(|r| r.start).collect::<Vec<_>>(),
        [page(0), page(0), page(4)]
    );
    // The later image's write is what guest memory holds.
    assert_eq!(g5_untile(&rig.mem, &later.image, 0, 0), drawn[1]);
    assert_ne!(g5_untile(&rig.mem, &first.image, 0, 0), drawn[0]);
    let entry = rig.entry(a);
    assert!(entry.stale && !entry.gpu_modified, "overwritten later");
    for t in [b, c] {
        let entry = rig.entry(t);
        assert!(!entry.stale && !entry.gpu_modified, "{t:?}");
    }
    // Its next use uploads what guest memory holds.
    assert_eq!(rig.uploads_of(&first).len(), 1);
}

#[test]
fn a_pitch_image_writes_back_row_by_row_and_keeps_its_padding() {
    let mut rig = Rig::new();
    let pitch = 512;
    let h = pitch_header(pitch);
    let t = rig.texture(&h).unwrap().texture;
    let row = 100 * 4;
    let drawn = pattern(7, row * 10);
    rig.host_write(region(t, 0, 0, (100, 10, 1)), &drawn);
    rig.cache.gpu_written(t);
    let span = 10 * u64::from(pitch);
    let before = rig.guest_bytes(page(0), span);
    rig.write_back(page(0)..page(1)).unwrap();
    let after = rig.guest_bytes(page(0), span);
    let pitch = usize::try_from(pitch).unwrap();
    for y in 0..10 {
        let at = y * pitch;
        assert_eq!(
            after[at..at + row],
            drawn[y * row..(y + 1) * row],
            "row {y}"
        );
        assert_eq!(
            after[at + row..at + pitch],
            before[at + row..at + pitch],
            "padding {y}"
        );
    }
}

#[test]
fn a_3d_image_writes_back_every_slice_of_every_level() {
    let mut rig = Rig::new();
    let h = decode(
        bl_header(page(0))
            .set(
                tex::TEXHEAD_BL_TEXTURE_TYPE,
                tex::TEXHEAD_BL_TEXTURE_TYPE_THREE_D,
            )
            .set(tex::TEXHEAD_BL_WIDTH_MINUS_ONE, 15)
            .set(tex::TEXHEAD_BL_HEIGHT_MINUS_ONE, 15)
            .set(tex::TEXHEAD_BL_DEPTH_MINUS_ONE, 3)
            .set(tex::TEXHEAD_BL_GOBS_PER_BLOCK_DEPTH, 1)
            .set(tex::TEXHEAD_BL_MAX_MIP_LEVEL, 1)
            .set(tex::TEXHEAD_BL_RES_VIEW_MAX_MIP_LEVEL, 1),
    );
    let t = rig.texture(&h).unwrap().texture;
    let mut drawn = Vec::new();
    for level in 0..2 {
        let extent = (16 >> level, 16 >> level, 4 >> level);
        let bytes = pattern(
            u8::try_from(level).unwrap() * 70 + 5,
            usize::try_from(extent.0 * extent.1 * extent.2 * 4).unwrap(),
        );
        rig.host_write(region(t, level, 0, extent), &bytes);
        drawn.push(bytes);
    }
    rig.cache.gpu_written(t);
    rig.write_back(page(0)..page(1)).unwrap();
    for (level, bytes) in (0..).zip(drawn) {
        assert_eq!(g5_untile(&rig.mem, &h.image, level, 0), bytes, "{level}");
    }
}

#[test]
fn readback_ids_are_distinct_and_carry_into_the_generation() {
    let mut ids = ReadbackIds::new();
    assert_eq!(ids.allocate(), ReadbackId::new(0, 0));
    assert_eq!(ids.allocate(), ReadbackId::new(1, 0));
    let mut ids = ReadbackIds::after(u64::from(u32::MAX));
    assert_eq!(ids.allocate(), ReadbackId::new(u32::MAX, 0));
    assert_eq!(ids.allocate(), ReadbackId::new(0, 1));
    let mut ids = ReadbackIds::after(u64::MAX - 1);
    assert_eq!(ids.allocate(), ReadbackId::new(u32::MAX - 1, u32::MAX));
    assert_eq!(ids.allocate(), ReadbackId::new(u32::MAX, u32::MAX));
    assert_eq!(ids.allocate(), ReadbackId::new(u32::MAX, u32::MAX));
}

// --- Images whose bytes cannot be converted ---

#[test]
fn a_multisampled_image_is_made_without_contents() {
    let mut rig = Rig::new();
    let h = decode(bl_header(page(0)).set(
        tex::TEXHEAD_BL_MULTI_SAMPLE_COUNT,
        tex::TEXHEAD_BL_MULTI_SAMPLE_COUNT_MODE_2X2,
    ));
    assert_eq!(h.image.samples, SampleLayout::S2x2);
    let rt = rig.target(&h.image).unwrap();
    assert_eq!(uploads(&rig.list.commands), Vec::<Upload>::new());
    let entry = rig.entry(rt.texture);
    assert_eq!(
        entry.source,
        Source::Unconvertible(Unconvertible::Multisampled)
    );
    assert_eq!(entry.watch, Watch::Unwatched);
    assert!(rig.log().iter().any(|l| l.contains(", 1 mips, 4 samples,")));
    let wb = rig
        .cache
        .begin_write_back(page(0)..page(1), &mut rig.ids, &mut rig.list);
    assert_eq!(
        wb.skipped(),
        [(rt.texture, Skip::Unconvertible(Unconvertible::Multisampled))]
    );
}

#[test]
fn a_depth_image_of_the_guessed_gob_is_made_without_contents() {
    let mut rig = Rig::new();
    let h = decode(
        bl_header(page(0))
            .set(
                tex::TEXHEAD_BL_COMPONENTS,
                tex::TEXHEAD_BL_COMPONENTS_SIZES_ZF32,
            )
            .types(tex::TEXHEAD_BL_R_DATA_TYPE.low(), super::format::FLOAT),
    );
    let view = rig.texture(&h).unwrap();
    assert_eq!(uploads(&rig.list.commands), Vec::<Upload>::new());
    assert_eq!(
        rig.entry(view.texture).source,
        Source::Unconvertible(Unconvertible::Gob(starquake_layout::GobType::FermiZs))
    );
    // It is not watched, and a write to its memory changes nothing it holds.
    assert!(!rig.watch.has_requests());
    rig.cache.cpu_writes(&[pages(0, 1)]);
    assert!(rig.entry(view.texture).stale);
    rig.texture(&h).unwrap();
    assert!(!rig.entry(view.texture).stale);
    assert_eq!(uploads(&rig.list.commands), Vec::<Upload>::new());
}

// --- Swizzle ---

fn swizzled() -> TextureHeader {
    decode(
        bl_header(page(0))
            .set(tex::TEXHEAD_BL_X_SOURCE, tex::TEXHEAD_BL_X_SOURCE_IN_B)
            .set(tex::TEXHEAD_BL_Z_SOURCE, tex::TEXHEAD_BL_Z_SOURCE_IN_R)
            .set(
                tex::TEXHEAD_BL_W_SOURCE,
                tex::TEXHEAD_BL_W_SOURCE_IN_ONE_FLOAT,
            ),
    )
}

#[test]
fn a_swizzled_texture_under_a_no_swizzle_caps_gets_an_identity_view() {
    let mut rig = Rig::new();
    let h = swizzled();
    let view = rig.texture(&h).unwrap();
    assert_eq!(
        view.swizzle,
        Swizzle {
            r: SwizzleSource::B,
            g: SwizzleSource::G,
            b: SwizzleSource::R,
            a: SwizzleSource::One,
        }
    );
    assert!(
        rig.log()
            .iter()
            .any(|l| l.starts_with("create v0: ") && l.ends_with("swizzle RGBA")),
        "{:?}",
        rig.log()
    );
}

#[test]
fn a_swizzled_texture_under_a_swizzle_caps_keeps_it_in_the_view() {
    let mut rig = Rig::with(
        caps(1 << 20, true, FormatTable::uniform(FormatFeatures::ALL)),
        LARGE_BUDGET,
    );
    let view = rig.texture(&swizzled()).unwrap();
    assert_eq!(view.swizzle, Swizzle::IDENTITY);
    assert!(
        rig.log()
            .iter()
            .any(|l| l.starts_with("create v0: ") && l.ends_with("swizzle BGR1")),
        "{:?}",
        rig.log()
    );
    // The identity header gets another view of the same image.
    let plain = rig.texture(&header(page(0))).unwrap();
    assert_eq!(plain.texture, view.texture);
    assert_ne!(plain.view, view.view);
}

// --- Views ---

#[test]
fn each_distinct_view_is_made_once_up_to_the_cap() {
    let mut rig = Rig::with(
        caps(1 << 20, true, FormatTable::uniform(FormatFeatures::ALL)),
        LARGE_BUDGET,
    );
    let sources = [
        tex::TEXHEAD_BL_X_SOURCE_IN_ZERO,
        tex::TEXHEAD_BL_X_SOURCE_IN_R,
        tex::TEXHEAD_BL_X_SOURCE_IN_G,
        tex::TEXHEAD_BL_X_SOURCE_IN_B,
        tex::TEXHEAD_BL_X_SOURCE_IN_A,
        tex::TEXHEAD_BL_X_SOURCE_IN_ONE_INT,
    ];
    let with = |i: usize| {
        decode(
            bl_header(page(0))
                .set(tex::TEXHEAD_BL_X_SOURCE, sources[i % 6])
                .set(tex::TEXHEAD_BL_Y_SOURCE, sources[i / 6 % 6]),
        )
    };
    // The headers' swizzles are distinct for i below 36, except the two ones.
    let mut views = Vec::new();
    // Bounded by the headers there are, so a broken cache fails rather than hangs.
    let mut i = 0;
    while views.len() < MAX_VIEWS_PER_IMAGE && i < 36 {
        let v = rig.texture(&with(i)).unwrap();
        if !views.contains(&v.view) {
            views.push(v.view);
        }
        i += 1;
    }
    assert_eq!(views.len(), MAX_VIEWS_PER_IMAGE);
    let t = rig.cache.entries().next().unwrap().texture;
    assert_eq!(rig.entry(t).views, MAX_VIEWS_PER_IMAGE);
    // A view it has is found again; one more is refused, and nothing changes.
    assert_eq!(rig.texture(&with(0)).unwrap().view, views[0]);
    let list = rig.list.commands.len();
    let mut refused = None;
    for j in i..36 {
        if let Err(e) = rig.texture(&with(j)) {
            refused = Some(e);
            break;
        }
    }
    assert_eq!(
        refused,
        Some(TextureError::TooManyViews {
            max: MAX_VIEWS_PER_IMAGE
        })
    );
    assert_eq!(rig.entry(t).views, MAX_VIEWS_PER_IMAGE);
    assert_eq!(rig.list.commands.len(), list);
}

// --- Unmaps ---

#[test]
fn an_ordered_unmap_drops_overlapping_images_and_keeps_neighbours() {
    let updates = [
        VmUpdate::Unmap {
            range: pages(IMAGE_PAGES, IMAGE_PAGES),
        },
        VmUpdate::ModifyKind {
            range: pages(IMAGE_PAGES, IMAGE_PAGES),
            kind: Kind(0xfe),
        },
        VmUpdate::DestroyMemory {
            range: pages(IMAGE_PAGES, IMAGE_PAGES),
        },
        VmUpdate::DestroySparseRegion {
            region: RegionId(1),
            range: pages(IMAGE_PAGES, IMAGE_PAGES),
        },
    ];
    for update in updates {
        let mut rig = Rig::new();
        let before = rig.texture(&header(page(0))).unwrap().texture;
        let dropped = rig
            .target(&header(page(IMAGE_PAGES)).image)
            .unwrap()
            .texture;
        // A second image at the dropped one's address, of another layout.
        let also = rig
            .texture(&decode(
                bl_header(page(IMAGE_PAGES)).set(tex::TEXHEAD_BL_GOBS_PER_BLOCK_HEIGHT, 1),
            ))
            .unwrap()
            .texture;
        let after = rig.texture(&header(page(2 * IMAGE_PAGES))).unwrap().texture;
        rig.cache.apply_update(&update);
        let left: Vec<TextureId> = rig.cache.entries().map(|e| e.texture).collect();
        assert_eq!(left, [before, after], "{update:?}");
        assert!(rig.cache.entry(dropped).is_none());
        assert!(rig.cache.entry(also).is_none());
    }
}

// --- Destruction ---

fn destroy_line(resource: &str, after: u64) -> String {
    format!("destroy {resource} after fence {after}: now")
}

fn line_index(log: &[String], line: &str) -> usize {
    log.iter()
        .position(|l| l == line)
        .unwrap_or_else(|| unreachable!("no line {line:?} in {log:?}"))
}

#[test]
fn an_image_dropped_while_its_list_is_open_is_destroyed_after_that_list() {
    let mut rig = Rig::new();
    rig.texture(&header(page(0))).unwrap();
    rig.submit();
    let used = rig.texture(&header(page(8))).unwrap();
    assert_eq!(
        (used.texture, used.view),
        (TextureId::new(1, 0), ViewId::new(1, 0))
    );
    rig.cache.apply_update(&VmUpdate::Unmap {
        range: pages(8, IMAGE_PAGES),
    });
    assert!(!rig.log().iter().any(|l| l.starts_with("destroy")));
    let fence = rig.submit();
    let log = rig.log();
    let submit = line_index(&log, &format!("submit fence {}:", fence.get()));
    assert!(submit < line_index(&log, &destroy_line("v1", fence.get())));
    assert!(submit < line_index(&log, &destroy_line("t1", fence.get())));
    // An image no open list uses goes after the last list that did.
    rig.cache.apply_update(&VmUpdate::Unmap {
        range: pages(0, IMAGE_PAGES),
    });
    line_index(&rig.log(), &destroy_line("t0", 1));
    assert_eq!(rig.backend.refused_destroys(), 0);
}

#[test]
fn dropping_the_cache_destroys_everything_after_the_last_fence() {
    let mut rig = Rig::new();
    rig.texture(&header(page(0))).unwrap();
    rig.texture(&header(page(4))).unwrap();
    let fence = rig.submit();
    let backend = Arc::clone(&rig.backend);
    drop(rig);
    let log: Vec<String> = backend.snapshot().lines().map(str::to_owned).collect();
    for resource in ["t0", "v0", "t1", "v1"] {
        line_index(&log, &destroy_line(resource, fence.get()));
    }
    assert_eq!(backend.budget().used, 0);
    assert_eq!(backend.refused_destroys(), 0);
}

// --- Eviction ---

const IMAGE: u64 = IMAGE_PAGES * PAGE_SIZE;

#[test]
fn an_idle_image_is_collected_after_the_buffer_cache_s_age_not_before() {
    // Twice the image: routine collection from one image's bytes, aggressive from two.
    let mut rig = Rig::with(
        caps(1 << 20, false, FormatTable::uniform(FormatFeatures::ALL)),
        2 * IMAGE,
    );
    assert_eq!(eviction_limits(2 * IMAGE).routine, IMAGE);
    let t = rig.texture(&header(page(0))).unwrap().texture;
    rig.submit();
    for _ in 0..EVICT_AFTER_FRAMES {
        assert_eq!(rig.cache.end_frame(), 0);
    }
    assert!(rig.cache.entry(t).is_some());
    assert_eq!(rig.cache.end_frame(), 1);
    assert!(rig.cache.is_empty());
}

#[test]
fn a_gpu_modified_image_is_never_collected() {
    let mut rig = Rig::with(
        caps(1 << 20, false, FormatTable::uniform(FormatFeatures::ALL)),
        2 * IMAGE,
    );
    let rt = rig.target(&header(page(0)).image).unwrap();
    rig.submit();
    for _ in 0..=2 * EVICT_AFTER_FRAMES {
        assert_eq!(rig.cache.end_frame(), 0);
    }
    assert!(rig.cache.entry(rt.texture).is_some());
}

#[test]
fn an_image_the_open_list_uses_is_not_collected() {
    let mut rig = Rig::with(
        caps(1 << 20, false, FormatTable::uniform(FormatFeatures::ALL)),
        2 * IMAGE,
    );
    let t = rig.texture(&header(page(0))).unwrap().texture;
    for _ in 0..=2 * EVICT_AFTER_FRAMES {
        assert_eq!(rig.cache.end_frame(), 0);
    }
    assert!(rig.cache.entry(t).is_some());
}

#[test]
fn a_new_image_over_the_budget_evicts_the_least_recently_used_idle_image_first() {
    let mut rig = Rig::with(
        caps(1 << 20, false, FormatTable::uniform(FormatFeatures::ALL)),
        3 * IMAGE,
    );
    let a = rig.texture(&header(page(0))).unwrap().texture;
    let b = rig.texture(&header(page(4))).unwrap().texture;
    let c = rig.texture(&header(page(8))).unwrap().texture;
    rig.submit();
    rig.texture(&header(page(0))).unwrap();
    rig.submit();
    let d = rig.texture(&header(page(12))).unwrap().texture;
    let left: Vec<TextureId> = rig.cache.entries().map(|e| e.texture).collect();
    assert_eq!(left, [a, c, d]);
    assert!(rig.cache.entry(b).is_none());
}

#[test]
fn a_new_image_past_the_budget_is_refused_at_the_exact_edge() {
    let mut rig = Rig::with(
        caps(1 << 20, false, FormatTable::uniform(FormatFeatures::ALL)),
        2 * IMAGE,
    );
    rig.texture(&header(page(0))).unwrap();
    // Exactly the budget: accepted.
    rig.texture(&header(page(4))).unwrap();
    let list = rig.list.commands.len();
    assert_eq!(
        rig.texture(&header(page(8))),
        Err(TextureError::OverBudget {
            bytes: 3 * IMAGE,
            allowed: 2 * IMAGE
        })
    );
    assert_eq!(rig.cache.len(), 2);
    assert_eq!(rig.list.commands.len(), list);
    // A GPU-modified idle image is not evicted to make room either.
    let first = rig.cache.entries().next().unwrap().texture;
    rig.cache.gpu_written(first);
    rig.submit();
    let d = rig.texture(&header(page(8))).unwrap().texture;
    let left: Vec<TextureId> = rig.cache.entries().map(|e| e.texture).collect();
    assert_eq!(left.len(), 2);
    assert_eq!(left[1], d);
    assert!(rig.cache.entries().next().unwrap().gpu_modified);
}

// --- Errors ---

#[test]
fn hostile_headers_are_errors_not_panics() {
    use tex::*;
    let mut rig = Rig::with(
        caps(1 << 20, false, FormatTable::uniform(FormatFeatures::ALL)),
        1 << 30,
    );
    let size_of = |h: &TextureHeader| {
        let info = h.image;
        let (tiling, pitch_b) = info.layout.tiling().unwrap();
        starquake_layout::Image::new(&ImageDesc {
            dim: info.dim,
            format: starquake_layout::Format::new(
                info.format.bytes_per_block(),
                info.format.block_width(),
                info.format.block_height(),
                1,
            )
            .unwrap(),
            extent_px: Extent4D::new(info.width, info.height, info.depth, info.layers),
            levels: info.levels,
            sample_layout: info.samples,
            tiling,
            pitch_b,
        })
        .unwrap()
        .size_b()
    };
    // The largest extents a header can state.
    let huge = decode(
        bl_header(page(0))
            .set(TEXHEAD_BL_TEXTURE_TYPE, TEXHEAD_BL_TEXTURE_TYPE_TWO_D_ARRAY)
            .set(TEXHEAD_BL_WIDTH_MINUS_ONE, 0x1_FFFF)
            .set(TEXHEAD_BL_HEIGHT_MINUS_ONE, 0xFFFF)
            .set(TEXHEAD_BL_HEIGHT_MINUS_ONE_BIT16, 1)
            .set(TEXHEAD_BL_DEPTH_MINUS_ONE, 0x3FFF)
            .set(TEXHEAD_BL_DEPTH_MINUS_ONE_BIT14, 1),
    );
    assert_eq!(
        rig.texture(&huge),
        Err(TextureError::OverBudget {
            bytes: size_of(&huge),
            allowed: 1 << 30
        })
    );
    // Row 87: an 8192 by 8192 by 68-layer ASTC 8x8 texture.
    let row_87 = decode(
        bl_header(page(0))
            .set(
                TEXHEAD_BL_COMPONENTS,
                TEXHEAD_BL_COMPONENTS_SIZES_ASTC_2D_8X8,
            )
            .set(TEXHEAD_BL_TEXTURE_TYPE, TEXHEAD_BL_TEXTURE_TYPE_TWO_D_ARRAY)
            .set(TEXHEAD_BL_WIDTH_MINUS_ONE, 8191)
            .set(TEXHEAD_BL_HEIGHT_MINUS_ONE, 8191)
            .set(TEXHEAD_BL_DEPTH_MINUS_ONE, 67),
    );
    assert_eq!(
        rig.texture(&row_87),
        Err(TextureError::OverBudget {
            bytes: size_of(&row_87),
            allowed: 1 << 30
        })
    );
    assert!(rig.cache.is_empty());
    assert_eq!(rig.list.commands, Vec::<Command>::new());
}

#[test]
fn headers_past_the_device_or_the_layout_are_errors() {
    use tex::*;
    let mut rig = Rig::with(
        caps(1 << 20, false, FormatTable::uniform(FormatFeatures::ALL)),
        1 << 30,
    );
    // Within the budget, past the device's largest 2D texture: the backend refuses it.
    let wide = decode(
        bl_header(page(0))
            .set(TEXHEAD_BL_WIDTH_MINUS_ONE, 4096)
            .set(TEXHEAD_BL_HEIGHT_MINUS_ONE, 0),
    );
    let desc = TextureDesc {
        dimension: starquake_render::resource::TextureDimension::D2,
        format: PixelFormat::A8B8G8R8Unorm,
        size: Extent3d {
            width: 4097,
            height: 1,
            depth_or_layers: 1,
        },
        mip_levels: 1,
        sample_count: 1,
        usage: texture_usage(rig.backend.caps(), PixelFormat::A8B8G8R8Unorm),
    };
    let expected =
        starquake_render::validate::validate_texture(rig.backend.caps(), &desc).unwrap_err();
    assert_eq!(rig.texture(&wide), Err(TextureError::Render(expected)));
    // A pitch header the layout crate refuses (a pitch below one row).
    let pitch = decode(
        Build::new()
            .set(
                TEXHEAD_PITCH_HEADER_VERSION,
                TEXHEAD_PITCH_HEADER_VERSION_SELECT_PITCH,
            )
            .set(
                TEXHEAD_PITCH_COMPONENTS,
                TEXHEAD_PITCH_COMPONENTS_SIZES_A8B8G8R8,
            )
            .types(TEXHEAD_PITCH_R_DATA_TYPE.low(), super::format::UNORM)
            .identity(TEXHEAD_PITCH_X_SOURCE.low())
            .set(TEXHEAD_PITCH_TEXTURE_TYPE, TEXHEAD_PITCH_TEXTURE_TYPE_TWO_D)
            .set(TEXHEAD_PITCH_PITCH_BITS20TO5, 1)
            .set(TEXHEAD_PITCH_WIDTH_MINUS_ONE, 99),
    );
    assert_eq!(
        rig.texture(&pitch),
        Err(TextureError::Layout(
            starquake_layout::LayoutError::PitchTooSmall {
                pitch: 32,
                row: 400
            }
        ))
    );
    assert!(rig.cache.is_empty());
    assert_eq!(rig.list.commands, Vec::<Command>::new());
}

#[test]
fn an_image_past_the_end_of_the_address_space_is_out_of_range() {
    let mut rig = Rig::new();
    let mut info = header(page(0)).image;
    info.address = 0u64.wrapping_sub(PAGE_SIZE);
    assert_eq!(
        rig.target(&info),
        Err(TextureError::OutOfRange {
            address: info.address,
            size: IMAGE
        })
    );
    // The image whose last page is the address space's last page also is: its end, 2^64, is
    // not representable.
    info.address = 0u64.wrapping_sub(IMAGE);
    assert_eq!(
        rig.target(&info),
        Err(TextureError::OutOfRange {
            address: info.address,
            size: IMAGE
        })
    );
    // One page lower it reaches memory (unmapped there: a host-only target).
    info.address = 0u64.wrapping_sub(IMAGE + PAGE_SIZE);
    let rt = rig.target(&info).unwrap();
    assert_eq!(rig.entry(rt.texture).source, Source::HostOnly);
}

#[test]
fn a_render_target_the_device_cannot_render_is_refused() {
    let mut formats = FormatTable::uniform(FormatFeatures::ALL);
    formats.set(
        PixelFormat::A8B8G8R8Unorm,
        FormatFeatures::SAMPLE.union(FormatFeatures::COPY),
    );
    let mut rig = Rig::with(caps(1 << 20, false, formats), LARGE_BUDGET);
    let info = header(page(0)).image;
    assert_eq!(
        rig.target(&info),
        Err(TextureError::NotRenderable {
            format: PixelFormat::A8B8G8R8Unorm
        })
    );
    assert!(rig.cache.is_empty());
    // An image made for sampling is refused as a target too.
    rig.texture(&header(page(0))).unwrap();
    let list = rig.list.commands.len();
    assert_eq!(
        rig.target(&info),
        Err(TextureError::NotRenderable {
            format: PixelFormat::A8B8G8R8Unorm
        })
    );
    assert!(!rig.cache.entries().next().unwrap().gpu_modified);
    assert_eq!(rig.list.commands.len(), list);
}

#[test]
fn a_render_target_view_is_an_array_exactly_when_the_image_is_layered() {
    let mut rig = Rig::new();
    let base = header(page(0)).image;
    let format = base.format;
    let cases = [
        (ImageDim::D2, 1, 3, "D2Array", page(0)),
        (ImageDim::D2, HEIGHT, 1, "D2", page(8)),
        (ImageDim::D1, 1, 2, "D1Array", page(16)),
        (ImageDim::D1, 1, 1, "D1", page(24)),
    ];
    for (n, (dim, height, layers, view, address)) in (0u32..).zip(cases) {
        let info = ImageInfo {
            dim,
            height: if dim == ImageDim::D2 { HEIGHT } else { height },
            layers,
            address,
            ..base
        };
        rig.target(&info).unwrap();
        let line = format!(
            "create v{n}: view of t{n} as {format} {view}, mips 0+1, layers 0+{layers}, swizzle RGBA"
        );
        assert!(rig.log().contains(&line), "{line} in {:?}", rig.log());
    }
}

#[test]
fn gpu_modified_images_are_kept_until_the_budget_refuses_new_ones() {
    // The divergence in the module docs: nothing collects or replaces a GPU-modified image.
    let mut rig = Rig::with(
        caps(1 << 20, false, FormatTable::uniform(FormatFeatures::ALL)),
        2 * IMAGE,
    );
    let first = rig.target(&header(page(0)).image).unwrap().texture;
    // A render target of another layout over the same memory does not replace it.
    let taller = decode(bl_header(page(0)).set(tex::TEXHEAD_BL_GOBS_PER_BLOCK_HEIGHT, 1)).image;
    let second = rig.target(&taller).unwrap().texture;
    rig.submit();
    for _ in 0..=2 * EVICT_AFTER_FRAMES {
        assert_eq!(rig.cache.end_frame(), 0);
    }
    assert_eq!(
        rig.target(&header(page(4)).image),
        Err(TextureError::OverBudget {
            bytes: 3 * IMAGE,
            allowed: 2 * IMAGE
        })
    );
    let left: Vec<TextureId> = rig.cache.entries().map(|e| e.texture).collect();
    assert_eq!(left, [first, second]);
}

#[test]
fn texture_usage_follows_the_format_table() {
    let mut formats = FormatTable::uniform(FormatFeatures::from_bits(0).unwrap());
    formats.set(PixelFormat::R8Unorm, FormatFeatures::SAMPLE);
    formats.set(PixelFormat::R8Snorm, FormatFeatures::RENDER);
    let caps = caps(1 << 20, false, formats);
    let copies = TextureUsage::COPY_SRC.union(TextureUsage::COPY_DST);
    assert_eq!(
        texture_usage(&caps, PixelFormat::R8Unorm),
        copies.union(TextureUsage::SAMPLED)
    );
    assert_eq!(
        texture_usage(&caps, PixelFormat::R8Snorm),
        copies.union(TextureUsage::RENDER_TARGET)
    );
    assert_eq!(texture_usage(&caps, PixelFormat::R8Uint), copies);
}

#[test]
fn checks_run_in_their_documented_order() {
    let mut formats = FormatTable::uniform(FormatFeatures::ALL);
    formats.set(
        PixelFormat::A8B8G8R8Unorm,
        FormatFeatures::SAMPLE.union(FormatFeatures::COPY),
    );
    let small = 2 * IMAGE;
    let mut rig = Rig::with(caps(1 << 20, false, formats), small);
    let base = header(page(0)).image;
    let end = 0u64.wrapping_sub(PAGE_SIZE);
    // Layout before OutOfRange.
    let info = ImageInfo {
        width: 0,
        address: end,
        ..base
    };
    assert_eq!(
        rig.target(&info),
        Err(TextureError::Layout(
            starquake_layout::LayoutError::ZeroExtent
        ))
    );
    // OutOfRange before NotRenderable.
    let info = ImageInfo {
        address: end,
        ..base
    };
    assert!(matches!(
        rig.target(&info),
        Err(TextureError::OutOfRange { .. })
    ));
    // NotRenderable before OverBudget.
    let info = ImageInfo {
        layers: 4,
        dim: ImageDim::D2,
        ..base
    };
    assert!(matches!(
        rig.target(&info),
        Err(TextureError::NotRenderable { .. })
    ));
    // OverBudget before the backend's refusal.
    let info = ImageInfo {
        width: 4097,
        height: 512,
        ..base
    };
    assert!(matches!(
        rig.texture(&TextureHeader {
            image: info,
            view: header(page(0)).view
        }),
        Err(TextureError::OverBudget { .. })
    ));
    // The backend's refusal before the upload's fault.
    let mut large = Rig::new();
    large.mem.unmap(pages(0, 1)).unwrap();
    let info = ImageInfo {
        width: 4097,
        height: 1,
        ..base
    };
    assert!(matches!(
        large.texture(&TextureHeader {
            image: info,
            view: header(page(0)).view
        }),
        Err(TextureError::Render(_))
    ));
    assert!(large.cache.is_empty());
    // The staging check before the read's fault.
    let mut tight = Rig::with(
        caps(255, false, FormatTable::uniform(FormatFeatures::ALL)),
        LARGE_BUDGET,
    );
    tight.mem.unmap(pages(0, 1)).unwrap();
    assert!(matches!(
        tight.texture(&header(page(0))),
        Err(TextureError::StagingTooSmall { .. })
    ));
    assert!(rig.cache.is_empty() && tight.cache.is_empty());
}

#[test]
fn a_held_image_runs_its_checks_in_their_documented_order() {
    let mut formats = FormatTable::uniform(FormatFeatures::ALL);
    formats.set(
        PixelFormat::A8B8G8R8Unorm,
        FormatFeatures::SAMPLE.union(FormatFeatures::COPY),
    );
    let mut rig = Rig::with(caps(1 << 20, true, formats), LARGE_BUDGET);
    let t = rig.texture(&header(page(0))).unwrap().texture;
    // Fill the image's views up to the cap, each with its own swizzle.
    let sources = [
        tex::TEXHEAD_BL_X_SOURCE_IN_ZERO,
        tex::TEXHEAD_BL_X_SOURCE_IN_R,
        tex::TEXHEAD_BL_X_SOURCE_IN_G,
        tex::TEXHEAD_BL_X_SOURCE_IN_B,
        tex::TEXHEAD_BL_X_SOURCE_IN_A,
        tex::TEXHEAD_BL_X_SOURCE_IN_ONE_INT,
    ];
    let with = |i: usize| {
        decode(
            bl_header(page(0))
                .set(tex::TEXHEAD_BL_X_SOURCE, sources[i % 6])
                .set(tex::TEXHEAD_BL_Y_SOURCE, sources[i / 6 % 6]),
        )
    };
    let mut i = 0;
    while rig.entry(t).views < MAX_VIEWS_PER_IMAGE && i < 36 {
        rig.texture(&with(i)).unwrap();
        i += 1;
    }
    assert_eq!(rig.entry(t).views, MAX_VIEWS_PER_IMAGE);
    let new_view = (i..36)
        .map(with)
        .find(|h| h.view.swizzle != with(0).view.swizzle && rig.texture(h).is_err())
        .unwrap();
    // Stale, its memory unmapped, at the view cap, and not renderable, all at once.
    rig.cache.invalidate(page(0)..page(1));
    rig.mem.unmap(pages(0, 1)).unwrap();
    let list = rig.list.commands.len();
    assert_eq!(
        rig.target(&header(page(0)).image),
        Err(TextureError::NotRenderable {
            format: PixelFormat::A8B8G8R8Unorm
        })
    );
    // The upload's fault before the view cap.
    assert_eq!(
        rig.texture(&new_view),
        Err(TextureError::Fault(GpuFault::Unmapped { va: page(0) }))
    );
    // With the memory back, the view cap.
    rig.mem.map(pages(0, 1), page(0)).unwrap();
    assert_eq!(
        rig.texture(&new_view),
        Err(TextureError::TooManyViews {
            max: MAX_VIEWS_PER_IMAGE
        })
    );
    // Nothing changed on any of them.
    assert_eq!(rig.list.commands.len(), list);
    assert!(rig.entry(t).stale);
    assert_eq!(rig.entry(t).views, MAX_VIEWS_PER_IMAGE);
}

#[test]
fn error_messages_are_plain_words() {
    assert_eq!(
        TextureError::OverBudget {
            bytes: 0x3000,
            allowed: 0x2000
        }
        .to_string(),
        "the texture cache would hold 0x3000 bytes, more than the 0x2000 allowed"
    );
    assert_eq!(
        TextureError::NotRenderable {
            format: PixelFormat::A8B8G8R8Unorm
        }
        .to_string(),
        "the device cannot render to A8B8G8R8_UNORM"
    );
    assert_eq!(
        TextureError::StagingTooSmall { row: 0x100, max: 0 }.to_string(),
        "a texture row of 0x100 bytes is larger than the largest staging slice, 0x0"
    );
    assert_eq!(
        TextureError::TooManyViews { max: 32 }.to_string(),
        "an image already has the most views it can have, 32"
    );
    assert_eq!(
        TextureError::Fault(GpuFault::Unmapped { va: 0x1000 }).to_string(),
        "reading guest memory for a texture upload: GPU address 0x1000 is not mapped"
    );
}

#[test]
fn write_back_error_messages_are_plain_words() {
    let texture = TextureId::new(3, 1);
    assert_eq!(
        WriteBackError::NotReady { texture, index: 0 }.to_string(),
        "readback 0 of texture 3 (generation 1) is not ready"
    );
    assert_eq!(
        WriteBackError::Unknown { texture }.to_string(),
        "the bytes of texture 3 (generation 1) are not known"
    );
    assert_eq!(
        WriteBackError::WrongSize {
            texture,
            len: 0,
            expected: 16
        }
        .to_string(),
        "a readback of texture 3 (generation 1) holds 0 bytes, where its level has 16"
    );
    assert_eq!(
        WriteBackError::Layout(starquake_layout::LayoutError::ZeroExtent).to_string(),
        "tiling a written-back level: an image extent has a zero dimension"
    );
    assert_eq!(
        WriteBackError::Fault {
            fault: GpuFault::Unmapped { va: 0x1000 },
            written: Vec::new(),
        }
        .to_string(),
        "writing an image back to guest memory: GPU address 0x1000 is not mapped"
    );
}

// --- Generated hostile headers ---

proptest! {
    #![proptest_config(Config {
        cases: 128,
        rng_seed: RngSeed::Fixed(0x0328_7e17),
        ..Config::default()
    })]

    /// Any header the decode accepts is made or refused by the cache; nothing panics, and a
    /// refusal leaves the cache and the list as they were.
    #[test]
    fn any_decoded_header_is_made_or_refused(words in proptest::array::uniform8(any::<u32>())) {
        let Ok(header) = TextureHeader::decode(&words) else {
            return Ok(());
        };
        let mut rig = Rig::with(
            caps(1 << 16, true, FormatTable::uniform(FormatFeatures::ALL)),
            1 << 24,
        );
        // The memory is at a fixed place; the header's own address mostly misses it.
        let header = TextureHeader {
            image: ImageInfo { address: page(0), ..header.image },
            ..header
        };
        if rig.texture(&header).is_ok() {
            prop_assert_eq!(rig.cache.len(), 1);
        } else {
            prop_assert!(rig.cache.is_empty());
            prop_assert!(rig.list.commands.is_empty());
        }
    }
}
