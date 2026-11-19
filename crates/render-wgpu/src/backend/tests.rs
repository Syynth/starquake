//! The backend on a real GPU. Each test skips itself, with a notice, in CI and where the host
//! has no adapter (crate docs), and runs on its own thread under a time bound, so a deadlock
//! fails it instead of hanging the gate.

use super::*;
use crate::plan::{BLITS_NOT_YET, DRAWS_NOT_YET, NO_BUFFER};
use starquake_render::command::{
    BlitDesc, BufferSlice, ClearDesc, ClearValue, Command, CopyDesc, CopyRegion, DispatchDesc,
    DispatchParams, Marker, PresentDesc, PresentTransform, Rect, TextureRegion,
};
use starquake_render::resource::{
    Extent3d, FilterMode, Origin3d, Swizzle, SwizzleSource, TextureDimension, ViewDimension,
};
use starquake_render::{
    BufferUsage, Frame, FramePixels, PixelFormat, RecordingBackend, TextureUsage, WakeCounter,
    check_completion_wake,
};
use std::sync::{OnceLock, Weak};

/// How long one GPU test may take.
const BOUND: Duration = Duration::from_secs(120);

/// How long a test waits for one list.
const LIST_WAIT: Duration = Duration::from_secs(10);

/// Runs `f` on another thread and fails if it takes longer than [`BOUND`].
pub(crate) fn bounded<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> T {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || tx.send(f()));
    rx.recv_timeout(BOUND).expect("the test finished in time")
}

/// A backend, or `None` (with a notice) in CI or on a host without an adapter.
fn open(test: &str, config: WgpuConfig, wake: CompletionWake) -> Option<WgpuBackend> {
    open_with(test, || WgpuBackend::new(config, wake))
}

/// The backend `make` opens, or `None` (with a notice) in CI or on a host without an adapter.
pub(crate) fn open_with(
    test: &str,
    make: impl FnOnce() -> Result<WgpuBackend, NewError>,
) -> Option<WgpuBackend> {
    if std::env::var_os("CI").is_some() {
        eprintln!("skipping {test}: GPU tests are local-only until verification 3 (decision 9)");
        return None;
    }
    match make() {
        Err(NewError::NoAdapter(reason)) => {
            eprintln!("skipping {test}: no GPU adapter on this host ({reason})");
            None
        }
        other => Some(other.expect("the adapter opens a device")),
    }
}

fn backend(test: &str) -> Option<WgpuBackend> {
    open(test, WgpuConfig::DEFAULT, CompletionWake::none())
}

fn finish(backend: &dyn Backend, fence: FenceValue) {
    assert_eq!(backend.wait(fence, LIST_WAIT), Ok(WaitOutcome::Reached));
}

const COPY: BufferUsage = BufferUsage::COPY_SRC.union(BufferUsage::COPY_DST);
const TARGET: TextureUsage = TextureUsage::SAMPLED
    .union(TextureUsage::RENDER_TARGET)
    .union(TextureUsage::COPY_SRC)
    .union(TextureUsage::COPY_DST);
const COPIES: TextureUsage = TextureUsage::COPY_SRC.union(TextureUsage::COPY_DST);

fn texture_desc(
    dimension: TextureDimension,
    format: PixelFormat,
    size: (u32, u32, u32),
    mip_levels: u32,
    usage: TextureUsage,
) -> TextureDesc {
    TextureDesc {
        dimension,
        format,
        size: Extent3d {
            width: size.0,
            height: size.1,
            depth_or_layers: size.2,
        },
        mip_levels,
        sample_count: 1,
        usage,
    }
}

fn whole_view(format: PixelFormat, dimension: ViewDimension, layers: (u32, u32)) -> ViewDesc {
    ViewDesc {
        format,
        dimension,
        base_mip: 0,
        mip_count: 1,
        base_layer: layers.0,
        layer_count: layers.1,
        swizzle: Swizzle::IDENTITY,
    }
}

fn slice(buffer: BufferId, offset: u64, size: u64) -> CopyRegion {
    CopyRegion::Buffer(BufferSlice {
        buffer,
        offset,
        size,
    })
}

fn region(
    texture: TextureId,
    mip: u32,
    origin: (u32, u32, u32),
    extent: (u32, u32, u32),
) -> CopyRegion {
    CopyRegion::Texture(TextureRegion {
        texture,
        mip,
        origin: Origin3d {
            x: origin.0,
            y: origin.1,
            z: origin.2,
        },
        extent: Extent3d {
            width: extent.0,
            height: extent.1,
            depth_or_layers: extent.2,
        },
    })
}

fn upload(bytes: Vec<u8>, to: CopyRegion) -> Command {
    Command::Upload {
        from: StagingSlice::from_bytes(bytes),
        to,
    }
}

/// Distinct, non-repeating-looking bytes from `seed`.
fn pattern(len: usize, seed: u8) -> Vec<u8> {
    (0..len)
        .map(|i| {
            u8::try_from(i % 251)
                .expect("small")
                .wrapping_mul(7)
                .wrapping_add(seed)
        })
        .collect()
}

/// The recording backend and the wgpu backend, fed the same calls.
struct Pair {
    rec: RecordingBackend,
    gpu: WgpuBackend,
    /// Readbacks the recording backend knew and the GPU matched.
    matched: usize,
    next_readback: u32,
}

impl Pair {
    fn new(gpu: WgpuBackend) -> Self {
        Self {
            rec: RecordingBackend::new(gpu.caps().clone(), WgpuConfig::DEFAULT.memory_limit),
            gpu,
            matched: 0,
            next_readback: 1,
        }
    }

    fn buffer(&self, size: u64) -> BufferId {
        let desc = BufferDesc { size, usage: COPY };
        let id = self.gpu.create_buffer(&desc).expect("buffer");
        assert_eq!(self.rec.create_buffer(&desc), Ok(id));
        id
    }

    fn texture(&self, desc: &TextureDesc) -> TextureId {
        let id = self.gpu.create_texture(desc).expect("texture");
        assert_eq!(self.rec.create_texture(desc), Ok(id));
        id
    }

    fn view(&self, texture: TextureId, desc: &ViewDesc) -> ViewId {
        let id = self.gpu.create_view(texture, desc).expect("view");
        assert_eq!(self.rec.create_view(texture, desc), Ok(id));
        id
    }

    fn readback(&mut self, from: CopyRegion) -> Command {
        let id = ReadbackId::new(self.next_readback, 0);
        self.next_readback += 1;
        Command::Readback { from, id }
    }

    /// Submits `commands` to both; both must accept or refuse alike. Every readback the
    /// recording backend knows must come back identical from the GPU.
    fn run(&mut self, commands: Vec<Command>) -> Result<(), RenderError> {
        let ids: Vec<ReadbackId> = commands
            .iter()
            .filter_map(|c| match c {
                Command::Readback { id, .. } => Some(*id),
                _ => None,
            })
            .collect();
        let list = CommandList { commands };
        let rec = self.rec.submit(list.clone());
        let gpu = self.gpu.submit(list);
        assert_eq!(gpu, rec, "both backends accept or refuse alike");
        let fence = gpu?;
        finish(&self.gpu, fence);
        for id in ids {
            let known = self.rec.take_readback(id).expect("recorded");
            let got = self.gpu.take_readback(id).expect("read back");
            if let Readback::Bytes(bytes) = known {
                let first = match &got {
                    Readback::Bytes(g) => g.iter().zip(&bytes).position(|(a, b)| a != b),
                    Readback::Unknown => Some(0),
                };
                assert_eq!(first, None, "readback {id:?} differs first at that byte");
                assert_eq!(got, Readback::Bytes(bytes), "readback {id:?}");
                self.matched += 1;
            }
        }
        assert_eq!(self.gpu.host_errors(), 0);
        Ok(())
    }
}

#[test]
fn adapter_report() {
    bounded(|| {
        let Some(gpu) = backend("adapter_report") else {
            return;
        };
        // Printed for the record (verification 1 asks for it in the PR).
        eprintln!("{}", gpu.adapter_report());
        assert!(!gpu.caps().view_swizzle);
        assert!(gpu.caps().sample_counts.supports(1));
        assert_eq!(
            gpu.caps().limits.max_staging_size,
            WgpuConfig::DEFAULT.max_staging_size
        );
        assert_eq!(
            gpu.caps().limits.max_sampler_anisotropy,
            crate::caps::WGPU_MAX_ANISOTROPY
        );
    });
}

#[test]
fn buffers_upload_copy_and_read_back_as_the_recording_backend_does() {
    bounded(|| {
        let Some(gpu) = backend("buffers") else {
            return;
        };
        let mut p = Pair::new(gpu);
        let a = p.buffer(256);
        let b = p.buffer(256);
        let reads = vec![
            p.readback(slice(a, 0, 256)),
            p.readback(slice(a, 1, 3)),
            p.readback(slice(a, 5, 250)),
            p.readback(slice(b, 64, 64)),
        ];
        let mut list = vec![
            upload(pattern(256, 1), slice(a, 0, 256)),
            Command::Copy(CopyDesc {
                src: slice(a, 16, 64),
                dst: slice(b, 64, 64),
            }),
        ];
        list.extend(reads);
        p.run(list).expect("accepted");
        assert_eq!(p.matched, 4);
    });
}

#[test]
fn textures_upload_copy_and_read_back_as_the_recording_backend_does() {
    bounded(|| {
        let Some(gpu) = backend("textures") else {
            return;
        };
        let mut p = Pair::new(gpu);
        let rgba = p.texture(&texture_desc(
            TextureDimension::D2,
            PixelFormat::A8B8G8R8Unorm,
            (16, 8, 1),
            2,
            TARGET,
        ));
        let srgb = p.texture(&texture_desc(
            TextureDimension::D2,
            PixelFormat::A8B8G8R8Srgb,
            (16, 8, 1),
            1,
            TARGET,
        ));
        // Rows of 7 bytes: the one-copy-per-row path.
        let r8 = p.texture(&texture_desc(
            TextureDimension::D2,
            PixelFormat::R8Unorm,
            (7, 5, 1),
            1,
            COPIES,
        ));
        // A 6x6 BC1 texture: 2x2 blocks, the last reaching the edge.
        let bc = p.texture(&texture_desc(
            TextureDimension::D2,
            PixelFormat::Bc1RgbaUnorm,
            (6, 6, 1),
            1,
            COPIES,
        ));
        let array = p.texture(&texture_desc(
            TextureDimension::D2,
            PixelFormat::R32Float,
            (4, 4, 3),
            1,
            COPIES,
        ));
        let deep = p.texture(&texture_desc(
            TextureDimension::D3,
            PixelFormat::R16G16Uint,
            (4, 2, 3),
            1,
            COPIES,
        ));
        let buf = p.buffer(4096);
        let mut list = vec![
            upload(
                pattern(16 * 8 * 4, 3),
                region(rgba, 0, (0, 0, 0), (16, 8, 1)),
            ),
            upload(pattern(8 * 4 * 4, 5), region(rgba, 1, (0, 0, 0), (8, 4, 1))),
            upload(pattern(7 * 5, 7), region(r8, 0, (0, 0, 0), (7, 5, 1))),
            upload(pattern(32, 9), region(bc, 0, (0, 0, 0), (6, 6, 1))),
            upload(
                pattern(4 * 4 * 3 * 4, 11),
                region(array, 0, (0, 0, 0), (4, 4, 3)),
            ),
            upload(
                pattern(4 * 2 * 3 * 4, 13),
                region(deep, 0, (0, 0, 0), (4, 2, 3)),
            ),
            Command::Copy(CopyDesc {
                src: region(rgba, 0, (4, 2, 0), (8, 4, 1)),
                dst: region(srgb, 0, (8, 4, 0), (8, 4, 1)),
            }),
            Command::Copy(CopyDesc {
                src: region(r8, 0, (1, 1, 0), (5, 3, 1)),
                dst: slice(buf, 3, 15),
            }),
            Command::Copy(CopyDesc {
                src: slice(buf, 100, 3 * 2 * 4),
                dst: region(rgba, 0, (13, 6, 0), (3, 2, 1)),
            }),
        ];
        let reads = [
            region(rgba, 0, (0, 0, 0), (16, 8, 1)),
            region(rgba, 1, (2, 1, 0), (5, 3, 1)),
            region(srgb, 0, (8, 4, 0), (8, 4, 1)),
            region(r8, 0, (0, 0, 0), (7, 5, 1)),
            region(r8, 0, (6, 4, 0), (1, 1, 1)),
            region(bc, 0, (4, 4, 0), (2, 2, 1)),
            region(bc, 0, (0, 0, 0), (6, 6, 1)),
            region(array, 0, (0, 0, 1), (4, 4, 2)),
            region(deep, 0, (1, 0, 1), (2, 2, 2)),
        ];
        // The buffer: fill it first, in an earlier list, so its bytes are known.
        p.run(vec![upload(pattern(4096, 17), slice(buf, 0, 4096))])
            .expect("accepted");
        for r in reads {
            list.push(p.readback(r));
        }
        list.push(p.readback(slice(buf, 0, 64)));
        p.run(list).expect("accepted");
        assert_eq!(p.matched, 10);
    });
}

#[test]
fn clears_write_what_the_recording_backend_knows() {
    bounded(|| {
        let Some(gpu) = backend("clears") else { return };
        let mut p = Pair::new(gpu);
        let cases = [
            (
                PixelFormat::A8B8G8R8Unorm,
                ClearValue::Float([1.0, 0.0, 1.0, 0.0]),
            ),
            (
                PixelFormat::A8B8G8R8Srgb,
                ClearValue::Float([0.0, 1.0, 1.0, 1.0]),
            ),
            (
                PixelFormat::A8B8G8R8Uint,
                ClearValue::Uint([255, 0, 7, 128]),
            ),
            (PixelFormat::R32Uint, ClearValue::Uint([u32::MAX, 0, 0, 0])),
            (PixelFormat::R32Sint, ClearValue::Sint([i32::MIN, 0, 0, 0])),
            (
                PixelFormat::R32Float,
                ClearValue::Float([0.1, 0.0, 0.0, 0.0]),
            ),
            (
                PixelFormat::R16Float,
                ClearValue::Float([0.5, 0.0, 0.0, 0.0]),
            ),
            (
                PixelFormat::R16G16B16A16Float,
                ClearValue::Float([-2.0, 0.25, 65504.0, 1.0]),
            ),
        ];
        for (format, value) in cases {
            let tex = p.texture(&texture_desc(
                TextureDimension::D2,
                format,
                (8, 4, 1),
                1,
                TARGET,
            ));
            let view = p.view(tex, &whole_view(format, ViewDimension::D2, (0, 1)));
            let bytes = usize::try_from(
                validate::packed_size(
                    format,
                    Extent3d {
                        width: 8,
                        height: 4,
                        depth_or_layers: 1,
                    },
                )
                .expect("small"),
            )
            .expect("small");
            let partial = Rect {
                x: 2,
                y: 1,
                width: 5,
                height: 2,
            };
            let mut list = vec![
                upload(pattern(bytes, 21), region(tex, 0, (0, 0, 0), (8, 4, 1))),
                Command::Clear(ClearDesc {
                    target: view,
                    value,
                    rect: Some(partial),
                }),
            ];
            list.push(p.readback(region(tex, 0, (0, 0, 0), (8, 4, 1))));
            list.push(Command::Clear(ClearDesc {
                target: view,
                value,
                rect: None,
            }));
            list.push(p.readback(region(tex, 0, (0, 0, 0), (8, 4, 1))));
            assert_eq!(p.run(list), Ok(()), "{format}");
        }
        assert_eq!(p.matched, 2 * 8);
    });
}

#[test]
fn clears_write_the_layers_and_slices_of_their_view() {
    bounded(|| {
        let Some(gpu) = backend("layered_clears") else {
            return;
        };
        let mut p = Pair::new(gpu);
        // Layers 1 and 2 of a five-layer texture, and every slice of a 3D one.
        let array = p.texture(&texture_desc(
            TextureDimension::D2,
            PixelFormat::R32Uint,
            (4, 4, 5),
            1,
            TARGET,
        ));
        let layers = p.view(
            array,
            &whole_view(PixelFormat::R32Uint, ViewDimension::D2Array, (1, 2)),
        );
        let deep = p.texture(&texture_desc(
            TextureDimension::D3,
            PixelFormat::R32Uint,
            (4, 4, 3),
            1,
            TARGET,
        ));
        let slices = p.view(
            deep,
            &whole_view(PixelFormat::R32Uint, ViewDimension::D3, (0, 1)),
        );
        let value = ClearValue::Uint([0xdead_beef, 0, 0, 0]);
        let mut list = vec![
            upload(
                pattern(4 * 4 * 5 * 4, 23),
                region(array, 0, (0, 0, 0), (4, 4, 5)),
            ),
            upload(
                pattern(4 * 4 * 3 * 4, 29),
                region(deep, 0, (0, 0, 0), (4, 4, 3)),
            ),
            Command::Clear(ClearDesc {
                target: layers,
                value,
                rect: Some(Rect {
                    x: 1,
                    y: 1,
                    width: 2,
                    height: 3,
                }),
            }),
            Command::Clear(ClearDesc {
                target: slices,
                value,
                rect: None,
            }),
        ];
        list.push(p.readback(region(array, 0, (0, 0, 0), (4, 4, 5))));
        list.push(p.readback(region(deep, 0, (0, 0, 0), (4, 4, 3))));
        p.run(list).expect("accepted");
        assert_eq!(p.matched, 2);
    });
}

#[test]
fn an_upload_lands_after_the_commands_before_it() {
    bounded(|| {
        let Some(gpu) = backend("upload_order") else {
            return;
        };
        let mut p = Pair::new(gpu);
        let tex = p.texture(&texture_desc(
            TextureDimension::D2,
            PixelFormat::A8B8G8R8Unorm,
            (4, 4, 1),
            1,
            TARGET,
        ));
        let view = p.view(
            tex,
            &whole_view(PixelFormat::A8B8G8R8Unorm, ViewDimension::D2, (0, 1)),
        );
        let a = p.buffer(64);
        let b = p.buffer(64);
        let mut list = vec![
            // A clear, then an upload into part of what it cleared: the upload wins there.
            Command::Clear(ClearDesc {
                target: view,
                value: ClearValue::Float([0.0; 4]),
                rect: None,
            }),
            upload(pattern(2 * 2 * 4, 31), region(tex, 0, (1, 1, 0), (2, 2, 1))),
            // An upload, a copy of it, then a second upload: the copy holds the first.
            upload(pattern(64, 37), slice(a, 0, 64)),
            Command::Copy(CopyDesc {
                src: slice(a, 0, 64),
                dst: slice(b, 0, 64),
            }),
            upload(pattern(64, 41), slice(a, 0, 64)),
        ];
        list.push(p.readback(region(tex, 0, (0, 0, 0), (4, 4, 1))));
        list.push(p.readback(slice(b, 0, 64)));
        list.push(p.readback(slice(a, 0, 64)));
        p.run(list).expect("accepted");
        assert_eq!(p.matched, 3);
    });
}

#[test]
fn random_lists_give_the_same_readbacks() {
    bounded(|| {
        let Some(gpu) = backend("random_lists") else {
            return;
        };
        let mut p = Pair::new(gpu);
        let mut rng = Rng(0x5eed_0382);
        let bufs = [p.buffer(1024), p.buffer(1024)];
        let formats = [
            PixelFormat::A8B8G8R8Unorm,
            PixelFormat::R8Uint,
            PixelFormat::R32Float,
        ];
        // Each format twice: wgpu copies between two textures only (`rules::SAME_LEVEL`).
        let textures: Vec<(TextureId, ViewId, PixelFormat)> = formats
            .iter()
            .chain(&formats)
            .map(|&f| {
                let t = p.texture(&texture_desc(TextureDimension::D2, f, (9, 6, 1), 1, TARGET));
                let v = p.view(t, &whole_view(f, ViewDimension::D2, (0, 1)));
                (t, v, f)
            })
            .collect();
        // Make every byte known first.
        let mut first = Vec::new();
        for &b in &bufs {
            first.push(upload(pattern(1024, 43), slice(b, 0, 1024)));
        }
        for &(t, _, f) in &textures {
            let n = 9 * 6 * usize::try_from(f.bytes_per_block()).expect("small");
            first.push(upload(pattern(n, 47), region(t, 0, (0, 0, 0), (9, 6, 1))));
        }
        p.run(first).expect("accepted");
        let mut refused = 0;
        for _ in 0..40 {
            let mut list = Vec::new();
            for _ in 0..=rng.below(6) {
                list.push(random_command(&mut rng, &bufs, &textures));
            }
            for &b in &bufs {
                list.push(p.readback(slice(b, 0, 1024)));
            }
            for &(t, _, _) in &textures {
                list.push(p.readback(region(t, 0, (0, 0, 0), (9, 6, 1))));
            }
            if p.run(list).is_err() {
                refused += 1;
            }
        }
        // Every list is accepted, and every readback compared: each byte was made known first.
        assert_eq!(refused, 0);
        assert_eq!(p.matched, 40 * (2 + textures.len()));
    });
}

/// A fixed-seed xorshift generator, so the random lists are the same every run.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }

    fn below32(&mut self, n: u32) -> u32 {
        u32::try_from(self.below(n.into())).expect("below n")
    }
}

/// One random command both backends can carry out (or both refuse).
fn random_command(
    rng: &mut Rng,
    bufs: &[BufferId; 2],
    textures: &[(TextureId, ViewId, PixelFormat)],
) -> Command {
    let index = |rng: &mut Rng| usize::try_from(rng.below(3)).expect("small");
    let pick = |rng: &mut Rng| textures[index(rng)];
    let area = |rng: &mut Rng| {
        let (left, top) = (rng.below32(9), rng.below32(6));
        Rect {
            x: left,
            y: top,
            width: rng.below32(9 - left) + 1,
            height: rng.below32(6 - top) + 1,
        }
    };
    let part = |texture, r: Rect| region(texture, 0, (r.x, r.y, 0), (r.width, r.height, 1));
    let seed = u8::try_from(rng.below(256)).expect("byte");
    match rng.below(6) {
        0 => {
            let offset = rng.below(256) * 4;
            let size = (rng.below((1024 - offset) / 4) + 1) * 4;
            let bytes = pattern(usize::try_from(size).expect("small"), seed);
            upload(bytes, slice(bufs[0], offset, size))
        }
        1 => {
            let (texture, _, format) = pick(rng);
            let r = area(rng);
            let len = r.width * r.height * format.bytes_per_block();
            upload(
                pattern(usize::try_from(len).expect("small"), seed),
                part(texture, r),
            )
        }
        2 => {
            let size = (rng.below(64) + 1) * 4;
            Command::Copy(CopyDesc {
                src: slice(bufs[0], rng.below(128) * 4, size),
                dst: slice(bufs[1], rng.below(128) * 4, size),
            })
        }
        3 => {
            let (texture, _, format) = pick(rng);
            let r = area(rng);
            let block = u64::from(format.bytes_per_block());
            let size = u64::from(r.width * r.height) * block;
            let buffer = slice(bufs[1], rng.below(64) * block, size);
            if rng.below(2) == 0 {
                Command::Copy(CopyDesc {
                    src: part(texture, r),
                    dst: buffer,
                })
            } else {
                Command::Copy(CopyDesc {
                    src: buffer,
                    dst: part(texture, r),
                })
            }
        }
        4 => {
            // From a texture to its twin of the same format, in either direction.
            let i = index(rng);
            let (from, to) = if rng.below(2) == 0 {
                (i, i + 3)
            } else {
                (i + 3, i)
            };
            let src = area(rng);
            let dst = Rect {
                x: rng.below32(9 - src.width + 1),
                y: rng.below32(6 - src.height + 1),
                ..src
            };
            Command::Copy(CopyDesc {
                src: part(textures[from].0, src),
                dst: part(textures[to].0, dst),
            })
        }
        _ => {
            let (_, target, format) = pick(rng);
            let value = match format {
                PixelFormat::A8B8G8R8Unorm => ClearValue::Float([1.0, 0.0, 1.0, 1.0]),
                PixelFormat::R8Uint => ClearValue::Uint([u32::from(seed), 0, 0, 0]),
                _ => ClearValue::Float([f32::from(seed) / 8.0, 0.0, 0.0, 0.0]),
            };
            let r = area(rng);
            let rect = (rng.below(2) == 0).then_some(r);
            Command::Clear(ClearDesc {
                target,
                value,
                rect,
            })
        }
    }
}

#[test]
fn presents_are_counted_and_dumped_frames_read_back() {
    #[derive(Debug, Default)]
    struct Sink(Mutex<Vec<Frame>>);
    impl FrameSink for Sink {
        fn frame(&self, frame: Frame) {
            self.0.lock().expect("unpoisoned").push(frame);
        }
    }
    bounded(|| {
        let Some(gpu) = backend("presents") else {
            return;
        };
        let p = Pair::new(gpu);
        let tex = p.texture(&texture_desc(
            TextureDimension::D2,
            PixelFormat::B8G8R8A8Unorm,
            (4, 4, 1),
            1,
            TARGET,
        ));
        let view = p.view(
            tex,
            &whole_view(PixelFormat::B8G8R8A8Unorm, ViewDimension::D2, (0, 1)),
        );
        let present = |frame, dump| {
            Command::Present(PresentDesc {
                view,
                src_rect: Rect {
                    x: 1,
                    y: 0,
                    width: 3,
                    height: 2,
                },
                transform: PresentTransform::FlipVertical,
                frame,
                dump,
            })
        };
        assert_eq!(p.gpu.set_present_target(PresentTarget::Offscreen), Ok(()));
        let list = CommandList {
            commands: vec![
                upload(pattern(64, 53), region(tex, 0, (0, 0, 0), (4, 4, 1))),
                present(1, false),
                present(2, true),
            ],
        };
        assert_eq!(p.rec.submit(list.clone()), Ok(FenceValue::new(1)));
        finish(&p.gpu, p.gpu.submit(list).expect("accepted"));
        assert_eq!(p.gpu.presents(), 2);
        let frames: Vec<u64> = p
            .gpu
            .take_present_feedback()
            .iter()
            .map(|f| f.frame)
            .collect();
        assert_eq!(frames, vec![1, 2]);
        assert_eq!(p.gpu.take_frame_dumps(), p.rec.take_frame_dumps());
        // With a sink, the frame goes to it, as RGBA8.
        let sink = Arc::new(Sink::default());
        let shared: Arc<dyn FrameSink> = Arc::<Sink>::clone(&sink);
        p.gpu.set_frame_sink(Some(shared));
        let fence = p
            .gpu
            .submit(CommandList {
                commands: vec![present(3, true)],
            })
            .expect("accepted");
        finish(&p.gpu, fence);
        let got = std::mem::take(&mut *sink.0.lock().expect("unpoisoned"));
        assert_eq!(got.len(), 1);
        let mut expected = Vec::new();
        let source = pattern(64, 53);
        for row in 0..2 {
            for col in 1..4 {
                let t = (row * 4 + col) * 4;
                expected.extend_from_slice(&[
                    source[t + 2],
                    source[t + 1],
                    source[t],
                    source[t + 3],
                ]);
            }
        }
        assert_eq!(got[0].pixels, FramePixels::Rgba8(expected));
        assert_eq!((got[0].frame, got[0].width, got[0].height), (3, 3, 2));
        assert!(
            p.gpu.take_frame_dumps().is_empty(),
            "nothing kept when a sink takes it"
        );
        assert_eq!(p.gpu.presents(), 3);
    });
}

#[test]
fn the_completion_wake_contract_holds() {
    bounded(|| {
        let counter = Arc::new(WakeCounter::new());
        let Some(gpu) = open(
            "wake_contract",
            WgpuConfig::DEFAULT,
            counter.completion_wake(),
        ) else {
            return;
        };
        assert_eq!(check_completion_wake(&gpu, &counter, 16, LIST_WAIT), Ok(()));
    });
}

#[test]
fn each_completion_wakes_after_the_new_value_is_stored() {
    let seen = bounded(|| {
        let slot: Arc<OnceLock<Weak<WgpuBackend>>> = Arc::new(OnceLock::new());
        let seen = Arc::new(Mutex::new(Vec::new()));
        let wake = {
            let (slot, seen) = (Arc::clone(&slot), Arc::clone(&seen));
            CompletionWake::new(move || {
                if let Some(backend) = slot.get().and_then(Weak::upgrade) {
                    // Takes the backend's lock: a wake made under it would deadlock here.
                    seen.lock().expect("unpoisoned").push(backend.completed());
                }
            })
        };
        let Some(gpu) = open("wake_after_store", WgpuConfig::DEFAULT, wake) else {
            return Vec::new();
        };
        let gpu = Arc::new(gpu);
        slot.set(Arc::downgrade(&gpu)).expect("set once");
        for n in 1..=8 {
            let fence = gpu.submit(CommandList::new()).expect("empty list");
            assert_eq!(fence, FenceValue::new(n));
            finish(gpu.as_ref(), fence);
        }
        // The last wake may run just after `wait` returns: give it its moment.
        let deadline = Instant::now() + LIST_WAIT;
        while seen.lock().expect("unpoisoned").last() != Some(&FenceValue::new(8)) {
            assert!(Instant::now() < deadline, "the last wake came");
            std::thread::yield_now();
        }
        seen.lock().expect("unpoisoned").clone()
    });
    if seen.is_empty() {
        return;
    }
    // Each wake saw a value its own completion stored: never zero, rising, ending at the last.
    assert!(seen.len() <= 8, "{seen:?}");
    assert!(seen[0] >= FenceValue::new(1), "{seen:?}");
    assert!(seen.windows(2).all(|w| w[0] < w[1]), "{seen:?}");
}

#[test]
fn the_queue_holds_at_most_its_depth() {
    bounded(|| {
        let config = WgpuConfig {
            queue_depth: std::num::NonZeroU64::MIN,
            ..WgpuConfig::DEFAULT
        };
        let Some(gpu) = open("queue_depth", config, CompletionWake::none()) else {
            return;
        };
        let tex = gpu
            .create_texture(&texture_desc(
                TextureDimension::D2,
                PixelFormat::A8B8G8R8Unorm,
                (1024, 1024, 1),
                1,
                TARGET,
            ))
            .expect("texture");
        let view = gpu
            .create_view(
                tex,
                &whole_view(PixelFormat::A8B8G8R8Unorm, ViewDimension::D2, (0, 1)),
            )
            .expect("view");
        let mut last = FenceValue::ZERO;
        for n in 0..12 {
            let clear = Command::Clear(ClearDesc {
                target: view,
                value: ClearValue::Float([
                    f32::from(u8::try_from(n).expect("small")) / 16.0,
                    0.0,
                    0.0,
                    1.0,
                ]),
                rect: Some(Rect {
                    x: 1,
                    y: 1,
                    width: 1000,
                    height: 1000,
                }),
            });
            last = gpu
                .submit(CommandList {
                    commands: vec![clear],
                })
                .expect("accepted");
            let unfinished = last.get() - gpu.completed().get();
            assert!(
                unfinished <= config.queue_depth.get(),
                "{unfinished} unfinished"
            );
        }
        finish(&gpu, last);
        assert_eq!(gpu.host_errors(), 0);
    });
}

#[test]
fn destroys_wait_for_their_fence() {
    bounded(|| {
        let Some(gpu) = backend("destroys") else {
            return;
        };
        let buf = gpu
            .create_buffer(&BufferDesc {
                size: 64,
                usage: COPY,
            })
            .expect("buffer");
        let read = |id, b| CommandList {
            commands: vec![Command::Readback {
                from: slice(b, 0, 4),
                id: ReadbackId::new(id, 0),
            }],
        };
        assert_eq!(gpu.budget().used, 64);
        // Waiting for a fence far ahead: the id stays usable, and a second destroy is refused.
        gpu.destroy(ResourceId::Buffer(buf), FenceValue::new(3));
        gpu.destroy(ResourceId::Buffer(buf), FenceValue::new(3));
        assert_eq!(gpu.refused_destroys(), 1);
        let f1 = gpu.submit(read(1, buf)).expect("still live");
        let f2 = gpu.submit(read(2, buf)).expect("still live");
        assert_eq!((f1.get(), f2.get()), (1, 2));
        finish(&gpu, f2);
        assert_eq!(
            gpu.submit(read(3, buf)).map(FenceValue::get),
            Ok(3),
            "still live at 2"
        );
        finish(&gpu, FenceValue::new(3));
        assert_eq!(
            gpu.submit(read(4, buf)),
            Err(RenderError::Invalid(NO_BUFFER)),
            "freed at 3"
        );
        assert_eq!(gpu.budget().used, 0);
        for id in 1..=3 {
            assert_eq!(
                gpu.take_readback(ReadbackId::new(id, 0)),
                Some(Readback::Bytes(vec![0; 4]))
            );
        }
        // A destroy whose fence has passed frees at once; a list queued before it still runs.
        let other = gpu
            .create_buffer(&BufferDesc {
                size: 8,
                usage: COPY,
            })
            .expect("buffer");
        let fence = gpu
            .submit(CommandList {
                commands: vec![
                    upload(vec![9; 8], slice(other, 0, 8)),
                    Command::Readback {
                        from: slice(other, 0, 8),
                        id: ReadbackId::new(5, 0),
                    },
                ],
            })
            .expect("accepted");
        gpu.destroy(ResourceId::Buffer(other), FenceValue::ZERO);
        assert_eq!(
            gpu.submit(read(6, other)),
            Err(RenderError::Invalid(NO_BUFFER))
        );
        finish(&gpu, fence);
        assert_eq!(
            gpu.take_readback(ReadbackId::new(5, 0)),
            Some(Readback::Bytes(vec![9; 8]))
        );
        // A fence equal to the completed value has passed: freed at once.
        let third = gpu
            .create_buffer(&BufferDesc {
                size: 8,
                usage: COPY,
            })
            .expect("buffer");
        gpu.destroy(ResourceId::Buffer(third), gpu.completed());
        assert_eq!(
            gpu.submit(read(7, third)),
            Err(RenderError::Invalid(NO_BUFFER))
        );
        // Ids that name nothing live are refused and counted.
        gpu.destroy(ResourceId::Buffer(other), FenceValue::ZERO);
        gpu.destroy(ResourceId::Shader(ShaderId::new(0, 0)), FenceValue::ZERO);
        assert_eq!(gpu.refused_destroys(), 3);
        assert_eq!(gpu.host_errors(), 0);
    });
}

#[test]
fn what_the_backend_refuses() {
    bounded(|| {
        let Some(gpu) = backend("refusals") else {
            return;
        };
        let buf = gpu
            .create_buffer(&BufferDesc {
                size: 64,
                usage: COPY,
            })
            .expect("buffer");
        let tex = gpu
            .create_texture(&texture_desc(
                TextureDimension::D2,
                PixelFormat::A8B8G8R8Unorm,
                (4, 4, 1),
                1,
                TARGET,
            ))
            .expect("texture");
        let view = gpu
            .create_view(
                tex,
                &whole_view(PixelFormat::A8B8G8R8Unorm, ViewDimension::D2, (0, 1)),
            )
            .expect("view");
        let one = |c| CommandList { commands: vec![c] };
        let blit = Command::Blit(BlitDesc {
            src: view,
            src_rect: Rect {
                x: 0,
                y: 0,
                width: 1,
                height: 1,
            },
            dst: view,
            dst_rect: Rect {
                x: 1,
                y: 1,
                width: 1,
                height: 1,
            },
            filter: FilterMode::Nearest,
        });
        assert_eq!(
            gpu.submit(one(blit)),
            Err(RenderError::Unsupported(BLITS_NOT_YET))
        );
        let dispatch = Command::Dispatch(DispatchDesc {
            pipeline: PipelineId::new(0, 0),
            bindings: Vec::new(),
            params: DispatchParams::Direct { x: 1, y: 1, z: 1 },
        });
        assert_eq!(
            gpu.submit(one(dispatch)),
            Err(RenderError::Unsupported(DRAWS_NOT_YET))
        );
        assert_eq!(
            gpu.submit(one(upload(vec![0; 6], slice(buf, 2, 6)))),
            Err(RenderError::Unsupported(crate::rules::BUFFER_ALIGNMENT))
        );
        let stale = BufferId::new(buf.index(), buf.generation() + 1);
        assert_eq!(
            gpu.submit(one(upload(vec![0; 4], slice(stale, 0, 4)))),
            Err(RenderError::Invalid(NO_BUFFER))
        );
        // Refused lists took no fence value.
        let marker = Command::Marker(Marker { submit: 1, draw: 1 });
        assert_eq!(gpu.submit(one(marker)), Ok(FenceValue::new(1)));
        let swizzled = ViewDesc {
            swizzle: Swizzle {
                r: SwizzleSource::B,
                ..Swizzle::IDENTITY
            },
            ..whole_view(PixelFormat::A8B8G8R8Unorm, ViewDimension::D2, (0, 1))
        };
        assert_eq!(
            gpu.create_view(tex, &swizzled),
            Err(RenderError::MissingFeature("view swizzles"))
        );
        assert!(matches!(
            gpu.create_shader(crate::fixtures::shader_desc(
                starquake_render::ShaderStage::Vertex,
                "main",
                starquake_render::ShaderSource::SpirV(Vec::new()),
                starquake_render::ShaderInterface::default(),
            )),
            Err(RenderError::Unsupported(crate::pipeline::NAGA_ONLY))
        ));
        assert_eq!(
            gpu.pipeline_status(PipelineId::new(0, 0)),
            PipelineStatus::Failed
        );
        assert_eq!(gpu.host_errors(), 0);
    });
}

#[test]
fn caps_limits_hold_at_their_edges() {
    bounded(|| {
        let config = WgpuConfig {
            memory_limit: 1 << 20,
            max_staging_size: 64 << 10,
            ..WgpuConfig::DEFAULT
        };
        let Some(gpu) = open("limits", config, CompletionWake::none()) else {
            return;
        };
        let l = gpu.caps().limits;
        let buffer = |size| {
            gpu.create_buffer(&BufferDesc { size, usage: COPY })
                .map(|_| ())
        };
        assert_eq!(l.max_buffer_size, 1 << 20, "capped at the memory limit");
        assert_eq!(buffer(l.max_buffer_size), Ok(()));
        assert!(matches!(
            buffer(l.max_buffer_size + 1),
            Err(RenderError::OutOfRange { .. })
        ));
        assert_eq!(l.max_staging_size, 64 << 10);
        assert!(gpu.staging(l.max_staging_size).is_ok());
        assert!(matches!(
            gpu.staging(l.max_staging_size + 1),
            Err(RenderError::OutOfRange { .. })
        ));
        let tex = |dimension, size| {
            gpu.create_texture(&texture_desc(
                dimension,
                PixelFormat::R8Unorm,
                size,
                1,
                TextureUsage::SAMPLED,
            ))
            .map(|_| ())
        };
        let d1 = l.max_texture_dimension_1d;
        assert_eq!(tex(TextureDimension::D1, (d1, 1, 1)), Ok(()));
        assert!(matches!(
            tex(TextureDimension::D1, (d1 + 1, 1, 1)),
            Err(RenderError::OutOfRange { .. })
        ));
        let d2 = l.max_texture_dimension_2d;
        assert_eq!(tex(TextureDimension::D2, (d2, 1, 1)), Ok(()));
        assert_eq!(tex(TextureDimension::D2, (1, d2, 1)), Ok(()));
        assert!(matches!(
            tex(TextureDimension::D2, (d2 + 1, 1, 1)),
            Err(RenderError::OutOfRange { .. })
        ));
        let d3 = l.max_texture_dimension_3d;
        assert_eq!(tex(TextureDimension::D3, (1, 1, d3)), Ok(()));
        assert!(matches!(
            tex(TextureDimension::D3, (1, 1, d3 + 1)),
            Err(RenderError::OutOfRange { .. })
        ));
        let layers = l.max_texture_array_layers;
        assert_eq!(tex(TextureDimension::D2, (1, 1, layers)), Ok(()));
        assert!(matches!(
            tex(TextureDimension::D2, (1, 1, layers + 1)),
            Err(RenderError::OutOfRange { .. })
        ));
        let mut sampler = crate::fixtures::sampler();
        sampler.max_anisotropy = l.max_sampler_anisotropy;
        assert!(gpu.create_sampler(&sampler).is_ok());
        sampler.max_anisotropy += 1;
        assert!(matches!(
            gpu.create_sampler(&sampler),
            Err(RenderError::OutOfRange { .. })
        ));
        for count in [1, 2, 4, 8, 16] {
            let desc = TextureDesc {
                sample_count: count,
                ..texture_desc(
                    TextureDimension::D2,
                    PixelFormat::A8B8G8R8Unorm,
                    (4, 4, 1),
                    1,
                    TextureUsage::RENDER_TARGET,
                )
            };
            let made = gpu.create_texture(&desc);
            assert_eq!(
                made.is_ok(),
                gpu.caps().sample_counts.supports(count),
                "{count}: {made:?}"
            );
        }
        // Every object above reaches wgpu without an error it would report later.
        finish(&gpu, gpu.submit(CommandList::new()).expect("empty list"));
        assert_eq!(gpu.host_errors(), 0);
    });
}

#[test]
fn a_lost_device_ends_every_call() {
    bounded(|| {
        let counter = Arc::new(WakeCounter::new());
        let Some(gpu) = open("lost", WgpuConfig::DEFAULT, counter.completion_wake()) else {
            return;
        };
        let seen = counter.count();
        gpu.lose_device_for_test();
        assert!(
            counter.wait_beyond(seen, LIST_WAIT),
            "the loss wakes the back thread"
        );
        assert_eq!(gpu.submit(CommandList::new()), Err(RenderError::DeviceLost));
        assert_eq!(
            gpu.wait(FenceValue::new(1), LIST_WAIT),
            Err(RenderError::DeviceLost)
        );
        assert_eq!(
            gpu.create_buffer(&BufferDesc {
                size: 4,
                usage: COPY
            }),
            Err(RenderError::DeviceLost)
        );
        assert_eq!(gpu.completed(), FenceValue::ZERO);
        assert_eq!(
            gpu.wait(FenceValue::ZERO, LIST_WAIT),
            Ok(WaitOutcome::Reached)
        );
    });
}

#[test]
fn dropping_the_backend_with_lists_in_flight_returns() {
    bounded(|| {
        let Some(gpu) = backend("drop") else { return };
        for _ in 0..4 {
            gpu.submit(CommandList::new()).expect("empty list");
        }
        drop(gpu);
    });
}

#[test]
fn readback_ids_are_reserved_until_taken() {
    bounded(|| {
        let Some(gpu) = backend("readback_ids") else {
            return;
        };
        let buf = gpu
            .create_buffer(&BufferDesc {
                size: 16,
                usage: COPY,
            })
            .expect("buffer");
        let read = |id| Command::Readback {
            from: slice(buf, 4, 8),
            id: ReadbackId::new(id, 0),
        };
        let one = |c| CommandList { commands: vec![c] };
        let taken = Err(RenderError::Invalid(crate::plan::READBACK_TAKEN));
        // A refused list reserves nothing: its readback id stays free.
        let blit = Command::Blit(BlitDesc {
            src: ViewId::new(0, 0),
            src_rect: Rect::default(),
            dst: ViewId::new(0, 0),
            dst_rect: Rect::default(),
            filter: FilterMode::Nearest,
        });
        let refused = CommandList {
            commands: vec![read(1), blit],
        };
        assert!(gpu.submit(refused).is_err());
        // In flight, then waiting to be taken: refused both times.
        let fence = gpu.submit(one(read(1))).expect("free");
        assert_eq!(gpu.submit(one(read(1))), taken);
        finish(&gpu, fence);
        assert_eq!(gpu.submit(one(read(1))), taken);
        assert_eq!(
            gpu.take_readback(ReadbackId::new(1, 0)),
            Some(Readback::Bytes(vec![0; 8]))
        );
        assert_eq!(gpu.take_readback(ReadbackId::new(1, 0)), None, "taken once");
        // Taken: free again.
        let fence = gpu.submit(one(read(1))).expect("free again");
        assert_eq!(
            fence,
            FenceValue::new(2),
            "the refused lists took no fence value"
        );
        finish(&gpu, fence);
        // A value past every list never arrives.
        assert_eq!(
            gpu.wait(FenceValue::new(3), Duration::from_millis(20)),
            Ok(WaitOutcome::TimedOut)
        );
        assert_eq!(gpu.host_errors(), 0);
    });
}

#[test]
fn aligned_rows_copy_in_one_go_as_the_recording_backend_does() {
    bounded(|| {
        let Some(gpu) = backend("aligned_rows") else {
            return;
        };
        let mut p = Pair::new(gpu);
        // 64-wide RGBA8: rows of exactly 256 bytes, several rows, several layers or slices.
        let array = p.texture(&texture_desc(
            TextureDimension::D2,
            PixelFormat::A8B8G8R8Unorm,
            (64, 4, 3),
            1,
            COPIES,
        ));
        let deep = p.texture(&texture_desc(
            TextureDimension::D3,
            PixelFormat::A8B8G8R8Unorm,
            (64, 4, 3),
            1,
            COPIES,
        ));
        // 128-wide BC1: 32 blocks of 8 bytes, block rows of 256 bytes, two block rows.
        let bc = p.texture(&texture_desc(
            TextureDimension::D2,
            PixelFormat::Bc1RgbaUnorm,
            (128, 8, 1),
            1,
            COPIES,
        ));
        let layer = 64 * 4 * 4;
        let buf = p.buffer(16384);
        p.run(vec![upload(pattern(16384, 29), slice(buf, 0, 16384))])
            .expect("accepted");
        let mut list = vec![
            upload(
                pattern(3 * layer, 31),
                region(array, 0, (0, 0, 0), (64, 4, 3)),
            ),
            upload(
                pattern(3 * layer, 37),
                region(deep, 0, (0, 0, 0), (64, 4, 3)),
            ),
            upload(pattern(512, 41), region(bc, 0, (0, 0, 0), (128, 8, 1))),
            // Buffer to texture at nonzero offsets on block boundaries.
            Command::Copy(CopyDesc {
                src: slice(buf, 256, 2 * 1024),
                dst: region(array, 0, (0, 0, 1), (64, 4, 2)),
            }),
            Command::Copy(CopyDesc {
                src: slice(buf, 4100, 2 * 1024),
                dst: region(deep, 0, (0, 0, 1), (64, 4, 2)),
            }),
            Command::Copy(CopyDesc {
                src: slice(buf, 8200, 512),
                dst: region(bc, 0, (0, 0, 0), (128, 8, 1)),
            }),
            // Texture to buffer, the same way.
            Command::Copy(CopyDesc {
                src: region(array, 0, (0, 0, 0), (64, 4, 3)),
                dst: slice(buf, 12288, 3 * 1024),
            }),
            Command::Copy(CopyDesc {
                src: region(deep, 0, (0, 1, 1), (64, 2, 2)),
                dst: slice(buf, 9000, 2 * 512),
            }),
        ];
        let reads = [
            region(array, 0, (0, 0, 0), (64, 4, 3)),
            region(array, 0, (0, 1, 1), (64, 2, 2)),
            region(deep, 0, (0, 0, 0), (64, 4, 3)),
            region(bc, 0, (0, 0, 0), (128, 8, 1)),
        ];
        for r in reads {
            list.push(p.readback(r));
        }
        list.push(p.readback(slice(buf, 0, 16384)));
        p.run(list).expect("accepted");
        assert_eq!(p.matched, 5);
    });
}

#[test]
fn depth_and_stencil_texels_copy_as_wgpu_allows() {
    bounded(|| {
        let Some(gpu) = backend("depth_copies") else {
            return;
        };
        if !gpu.depth_copies {
            eprintln!("skipping depth_copies: this device cannot copy depth texels to buffers");
            return;
        }
        let mut p = Pair::new(gpu);
        // Rows of 256 bytes, so guest buffer copies of several rows are one wgpu copy each.
        let d16 = p.texture(&texture_desc(
            TextureDimension::D2,
            PixelFormat::D16Unorm,
            (128, 4, 2),
            1,
            COPIES,
        ));
        let s8 = p.texture(&texture_desc(
            TextureDimension::D2,
            PixelFormat::S8Uint,
            (256, 2, 1),
            1,
            COPIES,
        ));
        let d32 = p.texture(&texture_desc(
            TextureDimension::D2,
            PixelFormat::D32Float,
            (64, 4, 1),
            1,
            TARGET,
        ));
        // One row of texels in each of two layers, rows of 16 bytes: one copy per layer.
        let line = p.texture(&texture_desc(
            TextureDimension::D2,
            PixelFormat::D16Unorm,
            (8, 1, 2),
            1,
            COPIES,
        ));
        let d32_view = p.view(
            d32,
            &whole_view(PixelFormat::D32Float, ViewDimension::D2, (0, 1)),
        );
        let buf = p.buffer(8192);
        p.run(vec![upload(pattern(8192, 43), slice(buf, 0, 8192))])
            .expect("accepted");
        let mut list = vec![
            upload(pattern(2048, 47), region(d16, 0, (0, 0, 0), (128, 4, 2))),
            upload(pattern(512, 53), region(s8, 0, (0, 0, 0), (256, 2, 1))),
            upload(pattern(32, 57), region(line, 0, (0, 0, 0), (8, 1, 2))),
            Command::Clear(ClearDesc {
                target: d32_view,
                value: ClearValue::Depth(0.25),
                rect: None,
            }),
            // Out of each into the buffer at offsets on 4 bytes, then into D16, S8 and the
            // one-row texture from it.
            Command::Copy(CopyDesc {
                src: region(d16, 0, (0, 0, 1), (128, 4, 1)),
                dst: slice(buf, 4, 1024),
            }),
            Command::Copy(CopyDesc {
                src: region(s8, 0, (0, 0, 0), (256, 2, 1)),
                dst: slice(buf, 1100, 512),
            }),
            Command::Copy(CopyDesc {
                src: region(d32, 0, (0, 0, 0), (64, 4, 1)),
                dst: slice(buf, 2000, 1024),
            }),
            Command::Copy(CopyDesc {
                src: region(line, 0, (0, 0, 0), (8, 1, 2)),
                dst: slice(buf, 3100, 32),
            }),
            Command::Copy(CopyDesc {
                src: slice(buf, 4000, 1024),
                dst: region(d16, 0, (0, 0, 0), (128, 4, 1)),
            }),
            Command::Copy(CopyDesc {
                src: slice(buf, 5200, 512),
                dst: region(s8, 0, (0, 0, 0), (256, 2, 1)),
            }),
            Command::Copy(CopyDesc {
                src: slice(buf, 6000, 16),
                dst: region(line, 0, (0, 0, 1), (8, 1, 1)),
            }),
        ];
        for r in [
            region(d16, 0, (0, 0, 0), (128, 4, 2)),
            region(d16, 0, (0, 0, 1), (128, 4, 1)),
            region(s8, 0, (0, 0, 0), (256, 2, 1)),
            region(d32, 0, (0, 0, 0), (64, 4, 1)),
            region(line, 0, (0, 0, 0), (8, 1, 2)),
        ] {
            list.push(p.readback(r));
        }
        list.push(p.readback(slice(buf, 0, 8192)));
        p.run(list).expect("accepted");
        assert_eq!(p.matched, 6);
        assert_eq!(p.gpu.host_errors(), 0);
    });
}

#[test]
fn depth_buffer_copies_wgpu_cannot_make_are_refused() {
    bounded(|| {
        let Some(gpu) = backend("depth_refusals") else {
            return;
        };
        if !gpu.depth_copies {
            eprintln!("skipping depth_refusals: this device cannot copy depth texels to buffers");
            return;
        }
        let mut p = Pair::new(gpu);
        let d16 = p.texture(&texture_desc(
            TextureDimension::D2,
            PixelFormat::D16Unorm,
            (128, 4, 1),
            1,
            COPIES,
        ));
        let d32 = p.texture(&texture_desc(
            TextureDimension::D2,
            PixelFormat::D32Float,
            (64, 4, 1),
            1,
            COPIES,
        ));
        let buf = p.buffer(2048);
        // What wgpu refuses and the recording backend accepts.
        let small = p.texture(&texture_desc(
            TextureDimension::D2,
            PixelFormat::D16Unorm,
            (8, 4, 1),
            1,
            COPIES,
        ));
        let one = |c| CommandList { commands: vec![c] };
        let refused = |m| Err(RenderError::Unsupported(m));
        assert_eq!(
            p.gpu.submit(one(upload(
                vec![0; 1024],
                region(d32, 0, (0, 0, 0), (64, 4, 1))
            ))),
            refused(crate::rules::DEPTH32_DESTINATION)
        );
        assert_eq!(
            p.gpu.submit(one(Command::Copy(CopyDesc {
                src: region(d16, 0, (0, 0, 0), (128, 4, 1)),
                dst: slice(buf, 2, 1024),
            }))),
            refused(crate::rules::DEPTH_OFFSET)
        );
        assert_eq!(
            p.gpu.submit(one(Command::Readback {
                from: region(d16, 0, (0, 0, 0), (127, 4, 1)),
                id: ReadbackId::new(900, 0),
            })),
            refused(crate::rules::WHOLE_COPY)
        );
        assert_eq!(
            p.gpu.submit(one(Command::Copy(CopyDesc {
                src: region(small, 0, (0, 0, 0), (8, 4, 1)),
                dst: slice(buf, 0, 64),
            }))),
            refused(crate::rules::DEPTH_ROWS)
        );
        // An upload and a readback of the same texture have no row limit.
        let list = vec![
            upload(pattern(64, 59), region(small, 0, (0, 0, 0), (8, 4, 1))),
            p.readback(region(small, 0, (0, 0, 0), (8, 4, 1))),
        ];
        p.run(list).expect("accepted");
        assert_eq!(p.matched, 1);
        assert_eq!(p.gpu.host_errors(), 0);
    });
}

#[test]
fn a_lists_rectangle_clears_share_one_scratch_per_format() {
    bounded(|| {
        let Some(gpu) = backend("scratch") else {
            return;
        };
        let mut p = Pair::new(gpu);
        let wide = p.texture(&texture_desc(
            TextureDimension::D2,
            PixelFormat::A8B8G8R8Uint,
            (32, 16, 2),
            1,
            TARGET,
        ));
        let wide_view = p.view(
            wide,
            &whole_view(PixelFormat::A8B8G8R8Uint, ViewDimension::D2Array, (0, 2)),
        );
        let small = p.texture(&texture_desc(
            TextureDimension::D2,
            PixelFormat::R32Uint,
            (8, 8, 1),
            1,
            TARGET,
        ));
        let small_view = p.view(
            small,
            &whole_view(PixelFormat::R32Uint, ViewDimension::D2, (0, 1)),
        );
        let rect = |target, n: u32, x, y, width, height| {
            Command::Clear(ClearDesc {
                target,
                value: ClearValue::Uint([n, n + 1, n + 2, n + 3]),
                rect: Some(Rect {
                    x,
                    y,
                    width,
                    height,
                }),
            })
        };
        let mut list = vec![
            upload(
                pattern(32 * 16 * 4 * 2, 59),
                region(wide, 0, (0, 0, 0), (32, 16, 2)),
            ),
            upload(
                pattern(8 * 8 * 4, 61),
                region(small, 0, (0, 0, 0), (8, 8, 1)),
            ),
            rect(wide_view, 10, 1, 1, 30, 2),
            rect(wide_view, 20, 2, 3, 3, 12),
            rect(small_view, 30, 1, 2, 5, 5),
            rect(wide_view, 40, 5, 5, 9, 9),
            rect(small_view, 50, 0, 0, 2, 7),
        ];
        list.push(p.readback(region(wide, 0, (0, 0, 0), (32, 16, 2))));
        list.push(p.readback(region(small, 0, (0, 0, 0), (8, 8, 1))));
        p.run(list).expect("accepted");
        assert_eq!(p.matched, 2);
        assert_eq!(p.gpu.scratch_textures(), 2, "one per format in the list");
        // A list of its own makes its own.
        let list = vec![
            rect(small_view, 60, 3, 3, 1, 1),
            p.readback(region(small, 0, (0, 0, 0), (8, 8, 1))),
        ];
        p.run(list).expect("accepted");
        assert_eq!(p.gpu.scratch_textures(), 3);
    });
}

#[test]
fn a_dumped_frame_reads_the_views_base_level_and_layer() {
    bounded(|| {
        let Some(gpu) = backend("dump_base") else {
            return;
        };
        let p = Pair::new(gpu);
        let tex = p.texture(&texture_desc(
            TextureDimension::D2,
            PixelFormat::A8B8G8R8Unorm,
            (8, 8, 3),
            2,
            TARGET,
        ));
        let view = p.view(
            tex,
            &ViewDesc {
                base_mip: 1,
                base_layer: 1,
                ..whole_view(PixelFormat::A8B8G8R8Unorm, ViewDimension::D2, (1, 1))
            },
        );
        let list = CommandList {
            commands: vec![
                upload(
                    pattern(8 * 8 * 4 * 3, 67),
                    region(tex, 0, (0, 0, 0), (8, 8, 3)),
                ),
                upload(
                    pattern(4 * 4 * 4 * 3, 71),
                    region(tex, 1, (0, 0, 0), (4, 4, 3)),
                ),
                Command::Present(PresentDesc {
                    view,
                    src_rect: Rect {
                        x: 1,
                        y: 1,
                        width: 2,
                        height: 3,
                    },
                    transform: PresentTransform::Identity,
                    frame: 1,
                    dump: true,
                }),
            ],
        };
        assert_eq!(p.rec.submit(list.clone()), Ok(FenceValue::new(1)));
        finish(&p.gpu, p.gpu.submit(list).expect("accepted"));
        let expected = p.rec.take_frame_dumps();
        assert_eq!(expected.len(), 1);
        assert!(matches!(expected[0].pixels, Readback::Bytes(_)), "known");
        assert_eq!(p.gpu.take_frame_dumps(), expected);
        assert_eq!(p.gpu.host_errors(), 0);
    });
}

#[test]
fn the_budget_counts_every_level_and_every_sample() {
    bounded(|| {
        let Some(gpu) = backend("budget") else {
            return;
        };
        let format = PixelFormat::A8B8G8R8Unorm;
        if !gpu.formats.samples(format).supports(4) {
            eprintln!("skipping budget: this device has no 4x {format}");
            return;
        }
        let p = Pair::new(gpu);
        let multisampled = TextureDesc {
            sample_count: 4,
            ..texture_desc(
                TextureDimension::D2,
                format,
                (16, 8, 1),
                1,
                TextureUsage::RENDER_TARGET,
            )
        };
        let mipped = texture_desc(TextureDimension::D2, format, (16, 8, 1), 3, COPIES);
        p.texture(&multisampled);
        p.texture(&mipped);
        // Derived from the descriptions: texel bytes times texels, per level, per sample.
        let texel = u64::from(format.bytes_per_block());
        let ms = 16 * 8 * texel * u64::from(multisampled.sample_count);
        let levels = (16 * 8 + 8 * 4 + 4 * 2) * texel;
        assert_eq!(p.gpu.budget().used, ms + levels);
        assert_eq!(p.gpu.budget().used, p.rec.budget().used);
    });
}

#[test]
fn the_last_handle_dropped_on_the_render_thread_does_not_join_it() {
    bounded(|| {
        let holder: Arc<Mutex<Option<Arc<WgpuBackend>>>> = Arc::new(Mutex::new(None));
        let (tx, rx) = std::sync::mpsc::channel::<()>();
        let rx = Arc::new(Mutex::new(rx));
        let wake = {
            let (holder, rx) = (Arc::clone(&holder), Arc::clone(&rx));
            CompletionWake::new(move || {
                let taken = holder.lock().expect("unpoisoned").take();
                if let Some(last) = taken {
                    // Wait until the test has dropped its own handle, so this one is the last.
                    let _ = rx.lock().expect("unpoisoned").recv_timeout(LIST_WAIT);
                    drop(last);
                }
            })
        };
        let Some(gpu) = open("drop_on_render_thread", WgpuConfig::DEFAULT, wake) else {
            return;
        };
        let gpu = Arc::new(gpu);
        let shared = Arc::clone(&gpu.shared);
        *holder.lock().expect("unpoisoned") = Some(Arc::clone(&gpu));
        gpu.submit(CommandList::new()).expect("empty list");
        drop(gpu);
        tx.send(()).expect("the wake waits");
        // The render thread ends by itself: every other handle to the shared state goes.
        let deadline = Instant::now() + LIST_WAIT;
        while Arc::strong_count(&shared) > 1 {
            assert!(Instant::now() < deadline, "the render thread ended");
            std::thread::yield_now();
        }
        let state = shared.state();
        assert!(state.shutdown, "dropped");
        assert!(!state.lost, "the render thread did not panic");
    });
}

#[test]
fn a_backend_joins_its_render_thread_only_from_another_thread() {
    let here = std::thread::current().id();
    let there = std::thread::spawn(|| std::thread::current().id())
        .join()
        .expect("joined");
    assert!(!should_join(here, here));
    assert!(should_join(there, here));
}

thread_local! {
    /// Comparisons made by [`Counted`] on this thread.
    static COMPARES: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// A key that counts its comparisons, equality ones too.
#[derive(Clone, Copy, Debug)]
struct Counted(u64);

impl PartialEq for Counted {
    fn eq(&self, other: &Self) -> bool {
        COMPARES.with(|c| c.set(c.get() + 1));
        self.0 == other.0
    }
}

impl Eq for Counted {}

impl Ord for Counted {
    fn cmp(&self, other: &Self) -> CmpOrdering {
        COMPARES.with(|c| c.set(c.get() + 1));
        self.0.cmp(&other.0)
    }
}

impl PartialOrd for Counted {
    fn partial_cmp(&self, other: &Self) -> Option<CmpOrdering> {
        Some(self.cmp(other))
    }
}

/// Comparisons `f` makes.
fn compares(f: impl FnOnce()) -> u64 {
    let before = COMPARES.with(std::cell::Cell::get);
    f();
    COMPARES.with(std::cell::Cell::get) - before
}

#[test]
fn a_destroy_costs_the_same_however_many_wait() {
    let per_call = |n: u64| {
        let mut pending = Pending::default();
        for i in 0..n {
            assert!(pending.push(Counted(i), FenceValue::new(1000 + i)));
        }
        let lookup = compares(|| assert!(pending.contains(&Counted(n / 2))));
        let refused = compares(|| assert!(!pending.push(Counted(n - 1), FenceValue::new(1))));
        let settle = compares(|| assert_eq!(pending.take_due(FenceValue::new(999)), Vec::new()));
        assert_eq!(pending.len(), usize::try_from(n).expect("small"));
        (lookup, refused, settle)
    };
    let (small, big) = (per_call(16), per_call(1 << 16));
    // A scan of the waiting ids would make about 2^15 comparisons for the big set; the index
    // makes a few per tree level, and a settle with nothing due compares no id.
    assert!(big.0 <= 128 && big.1 <= 128, "{small:?} {big:?}");
    assert_eq!((small.2, big.2), (0, 0));
}

#[test]
fn due_destroys_come_out_in_fence_then_call_order() {
    let mut pending = Pending::default();
    for (id, after) in [(1, 3), (2, 1), (3, 3), (4, 2), (5, 4), (6, 1)] {
        assert!(pending.push(Counted(id), FenceValue::new(after)));
    }
    assert!(
        !pending.push(Counted(4), FenceValue::new(9)),
        "waiting already"
    );
    assert_eq!(pending.take_due(FenceValue::ZERO), Vec::new());
    assert_eq!(
        pending.take_due(FenceValue::new(1)),
        vec![Counted(2), Counted(6)],
        "equal to the completed value is due"
    );
    assert_eq!(
        pending.take_due(FenceValue::new(3)),
        vec![Counted(4), Counted(1), Counted(3)]
    );
    assert!(
        !pending.contains(&Counted(1)),
        "taken ids are no longer waiting"
    );
    assert!(pending.contains(&Counted(5)), "one above stays");
    assert!(
        pending.push(Counted(1), FenceValue::new(9)),
        "a taken id may wait again"
    );
    assert_eq!(
        pending.take_due(FenceValue::new(u64::MAX)),
        vec![Counted(5), Counted(1)]
    );
    assert_eq!(pending.len(), 0);
}

#[test]
fn pending_ids_of_different_kinds_or_generations_differ() {
    let mut pending = Pending::default();
    let buffer = PendingId(ResourceId::Buffer(BufferId::new(1, 2)));
    assert!(pending.push(buffer, FenceValue::new(5)));
    for other in [
        ResourceId::Texture(TextureId::new(1, 2)),
        ResourceId::View(ViewId::new(1, 2)),
        ResourceId::Sampler(SamplerId::new(1, 2)),
        ResourceId::Buffer(BufferId::new(1, 3)),
        ResourceId::Buffer(BufferId::new(2, 2)),
    ] {
        assert!(!pending.contains(&PendingId(other)), "{other:?}");
    }
    assert!(pending.contains(&PendingId(ResourceId::Buffer(BufferId::new(1, 2)))));
}
