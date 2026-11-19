//! The render thread: takes the backend's queue in order, creates host objects, encodes lists,
//! submits them to wgpu's one queue and settles them when wgpu reports them finished
//! (docs/specs/render-backend.md, "Submit"; decision 3). For the shape only, it follows Eden's
//! Vulkan scheduler, which records work on one thread and hands it to a worker that submits in
//! order (`eden/src/video_core/renderer_vulkan/vk_scheduler.h` @ cf654052, `WorkerThread`), and
//! its staging pool, which keeps upload and readback memory apart from the resources
//! (`vk_staging_buffer_pool.h`); the code is new.
//!
//! # Encoding a list
//!
//! Steps are encoded in order into one command encoder, with two exceptions:
//!
//! - an upload uses `Queue::write_buffer` or `Queue::write_texture` (render-backend, "Buffers and
//!   uploads"), whose bytes wgpu copies in before the command buffers of the next
//!   `Queue::submit`. So when an upload follows an encoded command, the encoder so far is
//!   submitted first, and the upload lands after it, as the list orders it;
//! - a readback reads into a buffer of its own that can be mapped. After the list's last submit
//!   each such buffer is mapped (`Buffer::map_async`), and wgpu calls the mapping back once the
//!   submission has finished.
//!
//! A copy between a buffer and a texture is one wgpu copy when its rows are a multiple of 256
//! bytes (`COPY_BYTES_PER_ROW_ALIGNMENT`) or it is one row, and otherwise one copy per row, since
//! the backend keeps buffer data tightly packed and wgpu needs 256-byte row strides for copies of
//! several rows. The cost is bounded by the rows of the texture the copy names.
//!
//! A clear of a whole level clears each layer (or slice) in a render pass of its own; a clear of
//! a rectangle clears a scratch texture of the rectangle's size once and copies it into each
//! layer, since wgpu clears only whole attachments (the shared checks keep that path to colour,
//! single-sampled views, [`crate::rules::check_clear_area`]).
//!
//! # Settling
//!
//! wgpu reports each list's last submission finished through `Queue::on_submitted_work_done`,
//! whose callback runs after every mapping callback of that submission. Lists settle in order:
//! the thread copies out their readbacks and frame dumps, then stores them, the present
//! feedback and the new completed value together under the lock, then calls the completion wake,
//! then hands frames to their sinks (`crate::backend`, "Fences and the completion wake").
//!
//! # Waiting
//!
//! With nothing queued and nothing in flight the thread sleeps on the backend's condition
//! variable. With lists in flight it waits in wgpu for the oldest one, in slices of
//! [`POLL_SLICE`], so new work waits at most that long to be taken.

use crate::backend::Shared;
use crate::convert::TextureSpec;
use crate::plan::{ClearOps, ClearStep, Layers, ReadFrom, Rows, Step, TexelBox};
use starquake_render::command::PresentDesc;
use starquake_render::{
    BufferId, FenceValue, Frame, FrameDump, FrameSink, PixelFormat, PresentFeedback,
    PresentOutcome, Readback, ReadbackId, ResourceId, SamplerId, TextureId, ViewId,
};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex, PoisonError};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// The longest the render thread waits in wgpu for a list before it looks for new work.
pub const POLL_SLICE: Duration = Duration::from_millis(1);

/// How long a dropped backend waits for its lists in flight before it lets wgpu go.
pub const SHUTDOWN_WAIT: Duration = Duration::from_secs(5);

/// A host object to create.
#[derive(Debug)]
pub enum Create {
    /// A buffer of `size` bytes.
    Buffer {
        /// Its id.
        id: BufferId,
        /// Host size (`crate::rules::host_buffer_size`).
        size: u64,
        /// Usages.
        usage: wgpu::BufferUsages,
    },
    /// A texture.
    Texture {
        /// Its id.
        id: TextureId,
        /// What wgpu makes.
        spec: TextureSpec,
    },
    /// A view.
    View {
        /// Its id.
        id: ViewId,
        /// The texture viewed.
        texture: TextureId,
        /// What wgpu makes.
        view: wgpu::TextureViewDescriptor<'static>,
    },
    /// A sampler.
    Sampler {
        /// Its id.
        id: SamplerId,
        /// What wgpu makes.
        sampler: wgpu::SamplerDescriptor<'static>,
    },
}

/// A checked list.
#[derive(Debug)]
pub struct ListWork {
    /// Its fence value.
    pub fence: FenceValue,
    /// Its steps.
    pub steps: Vec<Step>,
    /// Where its dumped frames go, or `None` to keep them.
    pub sink: Option<Arc<dyn FrameSink>>,
}

/// Work for the render thread, in the order it was queued.
#[derive(Debug)]
pub enum Message {
    /// Create a host object.
    Create(Create),
    /// Release a host object.
    Free(ResourceId),
    /// Encode and submit a list.
    List(ListWork),
    /// Destroy the device, as a driver failure would lose it (tests only).
    #[cfg(test)]
    LoseDevice,
}

/// Host objects by slot, each with the generation of the id it was made for.
#[derive(Debug)]
struct HostTable<T>(Vec<Option<(u32, T)>>);

impl<T> HostTable<T> {
    const fn new() -> Self {
        Self(Vec::new())
    }

    fn insert(&mut self, index: u32, generation: u32, value: T) {
        let Ok(i) = usize::try_from(index) else {
            return;
        };
        if self.0.len() <= i {
            self.0.resize_with(i.saturating_add(1), || None);
        }
        if let Some(slot) = self.0.get_mut(i) {
            *slot = Some((generation, value));
        }
    }

    fn get(&self, index: u32, generation: u32) -> Option<&T> {
        match self.0.get(usize::try_from(index).ok()?)? {
            Some((g, value)) if *g == generation => Some(value),
            _ => None,
        }
    }

    fn remove(&mut self, index: u32, generation: u32) -> Option<T> {
        let slot = self.0.get_mut(usize::try_from(index).ok()?)?;
        if slot.as_ref().is_some_and(|(g, _)| *g == generation) {
            slot.take().map(|(_, value)| value)
        } else {
            None
        }
    }
}

/// What a readback buffer is for.
#[derive(Clone, Copy, Debug)]
enum ReadTarget {
    /// A `Readback` command.
    Readback(ReadbackId),
    /// A dumped frame.
    Dump,
}

/// A buffer being read back.
#[derive(Debug)]
struct Read {
    target: ReadTarget,
    /// The buffer, or `None` when the source could not be found (a host error, already counted).
    buffer: Option<wgpu::Buffer>,
    /// Where the bytes start in the buffer.
    skip: u64,
    /// How many bytes.
    len: u64,
    /// Set by the mapping callback: whether the map succeeded.
    mapped: Arc<Mutex<Option<bool>>>,
}

/// A present waiting for its list to finish.
#[derive(Debug)]
struct PresentRec {
    desc: PresentDesc,
    format: PixelFormat,
    /// Index in the list's reads of a dumped frame's pixels.
    read: Option<usize>,
}

/// A list submitted to wgpu and not yet settled.
#[derive(Debug)]
struct InFlight {
    fence: FenceValue,
    submission: wgpu::SubmissionIndex,
    reads: Vec<Read>,
    presents: Vec<PresentRec>,
    sink: Option<Arc<dyn FrameSink>>,
    done: bool,
}

/// Stops the backend if the thread panics, so no caller waits for it forever.
struct PanicGuard(Arc<Shared>);

impl Drop for PanicGuard {
    fn drop(&mut self) {
        if std::thread::panicking() {
            tracing::error!("the render thread panicked; the device counts as lost");
            self.0.lose();
        }
    }
}

/// The render thread's own state.
struct Render {
    device: wgpu::Device,
    queue: wgpu::Queue,
    shared: Arc<Shared>,
    buffers: HostTable<wgpu::Buffer>,
    textures: HostTable<wgpu::Texture>,
    views: HostTable<wgpu::TextureView>,
    samplers: HostTable<wgpu::Sampler>,
    in_flight: VecDeque<InFlight>,
    /// Fence values wgpu has reported finished, filled by its callbacks.
    finished: Arc<Mutex<Vec<FenceValue>>>,
    /// The origin of present feedback's host time.
    origin: Instant,
}

/// What the thread does next.
enum Next {
    Stop,
    Message(Message),
    Poll,
}

/// Installs the device's error and loss handlers and starts the render thread.
pub fn start(
    device: wgpu::Device,
    queue: wgpu::Queue,
    shared: Arc<Shared>,
) -> std::io::Result<JoinHandle<()>> {
    let errors = Arc::clone(&shared);
    device.on_uncaptured_error(Arc::new(move |error: wgpu::Error| {
        errors.host_error(&error);
    }));
    let lost = Arc::clone(&shared);
    device.set_device_lost_callback(move |reason, message| {
        tracing::error!("the host device was lost ({reason:?}): {message}");
        lost.lose();
    });
    let render = Render {
        device,
        queue,
        shared,
        buffers: HostTable::new(),
        textures: HostTable::new(),
        views: HostTable::new(),
        samplers: HostTable::new(),
        in_flight: VecDeque::new(),
        finished: Arc::default(),
        origin: Instant::now(),
    };
    std::thread::Builder::new()
        .name("starquake-render".into())
        .spawn(move || render.run())
}

impl Render {
    fn run(mut self) {
        let _guard = PanicGuard(Arc::clone(&self.shared));
        loop {
            match self.next() {
                Next::Stop => break,
                Next::Message(message) => self.handle(message),
                Next::Poll => self.poll(wgpu::PollType::Wait {
                    submission_index: self.in_flight.front().map(|f| f.submission.clone()),
                    timeout: Some(POLL_SLICE),
                }),
            }
            if !self.in_flight.is_empty() {
                self.poll(wgpu::PollType::Poll);
            }
            self.settle();
        }
        if !self.in_flight.is_empty() {
            self.poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(SHUTDOWN_WAIT),
            });
        }
    }

    /// The next thing to do, waiting while there is nothing.
    fn next(&mut self) -> Next {
        let mut state = self.shared.state();
        loop {
            if state.shutdown {
                return Next::Stop;
            }
            if state.lost {
                // Nothing completes on a lost device; drop the work and wait for the end.
                state.queue.clear();
                self.in_flight.clear();
            } else if let Some(message) = state.queue.pop_front() {
                return Next::Message(message);
            } else if !self.in_flight.is_empty() {
                return Next::Poll;
            }
            state = self
                .shared
                .work
                .wait(state)
                .unwrap_or_else(PoisonError::into_inner);
        }
    }

    fn poll(&self, poll: wgpu::PollType) {
        match self.device.poll(poll) {
            Ok(_) | Err(wgpu::PollError::Timeout) => {}
            Err(error) => tracing::warn!("polling the host device failed: {error}"),
        }
    }

    fn handle(&mut self, message: Message) {
        match message {
            Message::Create(create) => self.create(create),
            Message::Free(id) => self.free(id),
            Message::List(work) => self.encode(work),
            #[cfg(test)]
            Message::LoseDevice => {
                self.device.destroy();
                self.poll(wgpu::PollType::Poll);
            }
        }
    }

    fn create(&mut self, create: Create) {
        match create {
            Create::Buffer { id, size, usage } => {
                let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: None,
                    size,
                    usage,
                    mapped_at_creation: false,
                });
                self.buffers.insert(id.index(), id.generation(), buffer);
            }
            Create::Texture { id, spec } => {
                let texture = self.device.create_texture(&wgpu::TextureDescriptor {
                    label: None,
                    size: wgpu::Extent3d {
                        width: spec.host.size.width,
                        height: spec.host.size.height,
                        depth_or_array_layers: spec.host.size.depth_or_layers,
                    },
                    mip_level_count: spec.mip_levels,
                    sample_count: spec.sample_count,
                    dimension: spec.host.dimension,
                    format: spec.format,
                    usage: spec.usage,
                    view_formats: &spec.view_formats,
                });
                self.textures.insert(id.index(), id.generation(), texture);
            }
            Create::View { id, texture, view } => {
                match self.textures.get(texture.index(), texture.generation()) {
                    Some(t) => {
                        let view = t.create_view(&view);
                        self.views.insert(id.index(), id.generation(), view);
                    }
                    None => self.shared.host_error(&"a view's texture was not there"),
                }
            }
            Create::Sampler { id, sampler } => {
                let sampler = self.device.create_sampler(&sampler);
                self.samplers.insert(id.index(), id.generation(), sampler);
            }
        }
    }

    /// Releases a host object; wgpu keeps it until the GPU work that uses it is done.
    fn free(&mut self, id: ResourceId) {
        let released = match id {
            ResourceId::Buffer(i) => self.buffers.remove(i.index(), i.generation()).is_some(),
            ResourceId::Texture(i) => self.textures.remove(i.index(), i.generation()).is_some(),
            ResourceId::View(i) => self.views.remove(i.index(), i.generation()).is_some(),
            ResourceId::Sampler(i) => self.samplers.remove(i.index(), i.generation()).is_some(),
            ResourceId::Shader(_) | ResourceId::Pipeline(_) => false,
        };
        if !released {
            self.shared
                .host_error(&"a freed resource had no host object");
        }
    }

    fn buffer(&self, id: BufferId) -> Option<&wgpu::Buffer> {
        let found = self.buffers.get(id.index(), id.generation());
        if found.is_none() {
            self.shared
                .host_error(&"a checked buffer had no host object");
        }
        found
    }

    fn texture(&self, id: TextureId) -> Option<&wgpu::Texture> {
        let found = self.textures.get(id.index(), id.generation());
        if found.is_none() {
            self.shared
                .host_error(&"a checked texture had no host object");
        }
        found
    }

    fn encoder(&self) -> wgpu::CommandEncoder {
        self.device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None })
    }

    /// A buffer `len` bytes can be copied into and mapped from (at least one alignment unit).
    fn read_buffer(&self, len: u64) -> wgpu::Buffer {
        let size = len
            .checked_next_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT)
            .unwrap_or(u64::MAX)
            .max(wgpu::COPY_BUFFER_ALIGNMENT);
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Encodes a read of `from` into a buffer of its own.
    fn read(&self, encoder: &mut wgpu::CommandEncoder, target: ReadTarget, from: ReadFrom) -> Read {
        let mapped = Arc::default();
        let missing = |len| Read {
            target,
            buffer: None,
            skip: 0,
            len,
            mapped: Arc::clone(&mapped),
        };
        match from {
            ReadFrom::Buffer {
                buffer,
                offset,
                size,
            } => {
                let Some(src) = self.buffer(buffer) else {
                    return missing(size);
                };
                let align = wgpu::COPY_BUFFER_ALIGNMENT;
                let start = offset - offset % align;
                let end = offset
                    .saturating_add(size)
                    .checked_next_multiple_of(align)
                    .unwrap_or(u64::MAX);
                let len = end.saturating_sub(start);
                let dst = self.read_buffer(len);
                encoder.copy_buffer_to_buffer(src, start, &dst, 0, len);
                Read {
                    target,
                    buffer: Some(dst),
                    skip: offset - start,
                    len: size,
                    mapped,
                }
            }
            ReadFrom::Texture { at, rows } => {
                let Some(texture) = self.texture(at.texture) else {
                    return missing(rows.total());
                };
                let dst = self.read_buffer(rows.total());
                copy_rows(encoder, texture, at, &dst, 0, rows, Direction::ToBuffer);
                Read {
                    target,
                    buffer: Some(dst),
                    skip: 0,
                    len: rows.total(),
                    mapped,
                }
            }
        }
    }

    fn encode(&mut self, work: ListWork) {
        let mut encoder = self.encoder();
        // Whether the encoder holds a command an upload must not overtake.
        let mut recorded = false;
        let mut reads = Vec::new();
        let mut presents = Vec::new();
        for step in work.steps {
            let upload = matches!(step, Step::WriteBuffer { .. } | Step::WriteTexture { .. });
            if upload && recorded {
                let finished = std::mem::replace(&mut encoder, self.encoder());
                self.queue.submit([finished.finish()]);
            }
            // An upload leaves the encoder empty of commands: none before it, or just submitted.
            recorded = !upload;
            self.step(&mut encoder, step, &mut reads, &mut presents);
        }
        let submission = self.queue.submit([encoder.finish()]);
        for read in &reads {
            if let Some(buffer) = &read.buffer {
                let slot = Arc::clone(&read.mapped);
                buffer.map_async(wgpu::MapMode::Read, .., move |result| {
                    *slot.lock().unwrap_or_else(PoisonError::into_inner) = Some(result.is_ok());
                });
            }
        }
        let finished = Arc::clone(&self.finished);
        let fence = work.fence;
        self.queue.on_submitted_work_done(move || {
            finished
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .push(fence);
        });
        self.in_flight.push_back(InFlight {
            fence,
            submission,
            reads,
            presents,
            sink: work.sink,
            done: false,
        });
    }

    /// Encodes one step (an upload goes to the queue instead).
    fn step(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        step: Step,
        reads: &mut Vec<Read>,
        presents: &mut Vec<PresentRec>,
    ) {
        match step {
            Step::WriteBuffer {
                buffer,
                offset,
                bytes,
            } => {
                if let Some(b) = self.buffer(buffer) {
                    self.queue.write_buffer(b, offset, &bytes);
                }
            }
            Step::WriteTexture { at, rows, bytes } => {
                if let Some(t) = self.texture(at.texture) {
                    let layout = wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(rows.row_bytes),
                        rows_per_image: Some(rows.rows),
                    };
                    self.queue
                        .write_texture(texel_copy(t, at), &bytes, layout, extent(at));
                }
            }
            Step::CopyBuffer {
                src,
                src_offset,
                dst,
                dst_offset,
                size,
            } => {
                if let (Some(s), Some(d)) = (self.buffer(src), self.buffer(dst)) {
                    encoder.copy_buffer_to_buffer(s, src_offset, d, dst_offset, size);
                }
            }
            Step::BufferToTexture {
                buffer,
                offset,
                rows,
                at,
            } => {
                if let (Some(b), Some(t)) = (self.buffer(buffer), self.texture(at.texture)) {
                    copy_rows(encoder, t, at, b, offset, rows, Direction::ToTexture);
                }
            }
            Step::TextureToBuffer {
                at,
                buffer,
                offset,
                rows,
            } => {
                if let (Some(b), Some(t)) = (self.buffer(buffer), self.texture(at.texture)) {
                    copy_rows(encoder, t, at, b, offset, rows, Direction::ToBuffer);
                }
            }
            Step::TextureToTexture { src, dst } => {
                if let (Some(s), Some(d)) = (self.texture(src.texture), self.texture(dst.texture)) {
                    encoder.copy_texture_to_texture(
                        texel_copy(s, src),
                        texel_copy(d, dst),
                        extent(src),
                    );
                }
            }
            Step::Clear(clear) => self.clear(encoder, &clear),
            Step::Readback { id, from } => {
                reads.push(self.read(encoder, ReadTarget::Readback(id), from));
            }
            Step::Present { desc, format, read } => {
                let read = read.map(|(at, rows)| {
                    let from = ReadFrom::Texture { at, rows };
                    reads.push(self.read(encoder, ReadTarget::Dump, from));
                    reads.len() - 1
                });
                presents.push(PresentRec { desc, format, read });
            }
        }
    }

    fn clear(&self, encoder: &mut wgpu::CommandEncoder, step: &ClearStep) {
        let Some(texture) = self.texture(step.texture) else {
            return;
        };
        let (dimension, first, count, sliced) = match step.layers {
            Layers::Array { first, count } => (wgpu::TextureViewDimension::D2, first, count, false),
            Layers::Slices { count } => (wgpu::TextureViewDimension::D3, 0, count, true),
        };
        let view = |texture: &wgpu::Texture, layer: u32, mip: u32| {
            texture.create_view(&wgpu::TextureViewDescriptor {
                label: None,
                format: Some(step.format),
                dimension: Some(dimension),
                usage: Some(wgpu::TextureUsages::RENDER_ATTACHMENT),
                aspect: wgpu::TextureAspect::All,
                base_mip_level: mip,
                mip_level_count: Some(1),
                base_array_layer: layer,
                array_layer_count: Some(1),
            })
        };
        let end = first.saturating_add(count);
        let Some(rect) = step.rect else {
            for z in first..end {
                let (layer, slice) = if sliced { (0, Some(z)) } else { (z, None) };
                clear_pass(
                    encoder,
                    &view(texture, layer, step.mip),
                    slice,
                    step.ops,
                    step.format,
                );
            }
            return;
        };
        let size = wgpu::Extent3d {
            width: rect.width,
            height: rect.height,
            depth_or_array_layers: 1,
        };
        let scratch = self.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: step.format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let scratch_view = scratch.create_view(&wgpu::TextureViewDescriptor::default());
        clear_pass(encoder, &scratch_view, None, step.ops, step.format);
        for z in first..end {
            encoder.copy_texture_to_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &scratch,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyTextureInfo {
                    texture,
                    mip_level: step.mip,
                    origin: wgpu::Origin3d {
                        x: rect.x,
                        y: rect.y,
                        z,
                    },
                    aspect: wgpu::TextureAspect::All,
                },
                size,
            );
        }
    }

    /// Settles the lists wgpu has finished, oldest first (module docs).
    fn settle(&mut self) {
        let finished =
            std::mem::take(&mut *self.finished.lock().unwrap_or_else(PoisonError::into_inner));
        for fence in finished {
            if let Some(entry) = self.in_flight.iter_mut().find(|f| f.fence == fence) {
                entry.done = true;
            }
        }
        let mut readbacks = Vec::new();
        let mut feedback = Vec::new();
        let mut kept = Vec::new();
        let mut delivered = Vec::new();
        let mut last = None;
        while self.in_flight.front().is_some_and(|f| f.done) {
            let Some(entry) = self.in_flight.pop_front() else {
                break;
            };
            let bytes: Vec<Option<Vec<u8>>> =
                entry.reads.iter().map(|r| self.read_out(r)).collect();
            for (read, bytes) in entry.reads.iter().zip(&bytes) {
                if let ReadTarget::Readback(id) = read.target {
                    readbacks.push((id, bytes.clone().map_or(Readback::Unknown, Readback::Bytes)));
                }
            }
            let host_time = self.origin.elapsed();
            for present in entry.presents {
                feedback.push(PresentFeedback {
                    frame: present.desc.frame,
                    outcome: PresentOutcome::Shown,
                    host_time,
                });
                if !present.desc.dump {
                    continue;
                }
                let pixels = present
                    .read
                    .and_then(|i| bytes.get(i).cloned().flatten())
                    .map_or(Readback::Unknown, Readback::Bytes);
                let dump = FrameDump {
                    frame: present.desc.frame,
                    view: present.desc.view,
                    format: present.format,
                    width: present.desc.src_rect.width,
                    height: present.desc.src_rect.height,
                    transform: present.desc.transform,
                    pixels,
                };
                match &entry.sink {
                    Some(sink) => delivered.push((Arc::clone(sink), dump)),
                    None => kept.push(dump),
                }
            }
            last = Some(entry.fence);
        }
        let Some(last) = last else {
            return;
        };
        let mut state = self.shared.state();
        let presents = u64::try_from(feedback.len()).unwrap_or(u64::MAX);
        state.presents = state.presents.saturating_add(presents);
        state.readbacks.extend(readbacks);
        state.feedback.extend(feedback);
        state.dumps.extend(kept);
        state.completed = last;
        state.free_passed();
        drop(state);
        self.shared.moved.notify_all();
        // After the store, with no lock held (render-backend, "The completion wake").
        self.shared.wake.wake();
        for (sink, dump) in delivered {
            sink.frame(Frame::from(dump));
        }
    }

    /// The bytes a finished read copied out, or `None` (counted) when it has none.
    fn read_out(&self, read: &Read) -> Option<Vec<u8>> {
        let buffer = read.buffer.as_ref()?;
        let mapped = *read.mapped.lock().unwrap_or_else(PoisonError::into_inner);
        if mapped != Some(true) {
            self.shared.host_error(&"a readback buffer did not map");
            return None;
        }
        let bytes = match buffer.get_mapped_range(..) {
            Ok(view) => {
                let start = usize::try_from(read.skip).ok()?;
                let end = start.checked_add(usize::try_from(read.len).ok()?)?;
                view.get(start..end).map(<[u8]>::to_vec)
            }
            Err(error) => {
                self.shared.host_error(&error);
                None
            }
        };
        buffer.unmap();
        bytes
    }
}

/// Encodes a clear of one attachment.
fn clear_pass(
    encoder: &mut wgpu::CommandEncoder,
    view: &wgpu::TextureView,
    depth_slice: Option<u32>,
    ops: ClearOps,
    format: wgpu::TextureFormat,
) {
    let store = wgpu::StoreOp::Store;
    let (color, depth_stencil) = match ops {
        ClearOps::Color(color) => (
            Some(wgpu::RenderPassColorAttachment {
                view,
                depth_slice,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(color),
                    store,
                },
            }),
            None,
        ),
        ClearOps::DepthStencil { depth, stencil } => (
            None,
            Some(wgpu::RenderPassDepthStencilAttachment {
                view,
                depth_ops: format.has_depth_aspect().then_some(wgpu::Operations {
                    load: depth.map_or(wgpu::LoadOp::Load, wgpu::LoadOp::Clear),
                    store,
                }),
                stencil_ops: format.has_stencil_aspect().then_some(wgpu::Operations {
                    load: stencil.map_or(wgpu::LoadOp::Load, wgpu::LoadOp::Clear),
                    store,
                }),
            }),
        ),
    };
    let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: None,
        color_attachments: &[color],
        depth_stencil_attachment: depth_stencil,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    drop(pass);
}

/// wgpu's description of one side of a texture copy.
fn texel_copy(texture: &wgpu::Texture, at: TexelBox) -> wgpu::TexelCopyTextureInfo<'_> {
    wgpu::TexelCopyTextureInfo {
        texture,
        mip_level: at.mip,
        origin: wgpu::Origin3d {
            x: at.origin.x,
            y: at.origin.y,
            z: at.origin.z,
        },
        aspect: wgpu::TextureAspect::All,
    }
}

fn extent(at: TexelBox) -> wgpu::Extent3d {
    wgpu::Extent3d {
        width: at.extent.width,
        height: at.extent.height,
        depth_or_array_layers: at.extent.depth_or_layers,
    }
}

/// Which way [`copy_rows`] copies.
#[derive(Clone, Copy)]
enum Direction {
    ToTexture,
    ToBuffer,
}

/// Copies `at` between `texture` and tightly packed `rows` at `offset` in `buffer` (module docs:
/// one copy when wgpu allows it, else one per row).
fn copy_rows(
    encoder: &mut wgpu::CommandEncoder,
    texture: &wgpu::Texture,
    at: TexelBox,
    buffer: &wgpu::Buffer,
    offset: u64,
    rows: Rows,
    direction: Direction,
) {
    let mut copy = |layout: wgpu::TexelCopyBufferLayout, at: TexelBox| {
        let buffer = wgpu::TexelCopyBufferInfo { buffer, layout };
        match direction {
            Direction::ToTexture => {
                encoder.copy_buffer_to_texture(buffer, texel_copy(texture, at), extent(at));
            }
            Direction::ToBuffer => {
                encoder.copy_texture_to_buffer(texel_copy(texture, at), buffer, extent(at));
            }
        }
    };
    for (layout, part) in row_copies(at, offset, rows, texture.format().block_dimensions().1) {
        copy(layout, part);
    }
}

/// The copies [`copy_rows`] makes, as buffer layouts and texture boxes (module docs).
///
/// Every layout names its row stride. wgpu-hal 30.0.2's Metal encoder passes a missing
/// `bytes_per_row` to Metal as 0 (`src/metal/command.rs`, `copy_texture_to_buffer` and
/// `copy_buffer_to_texture`: `bytes_per_row.unwrap_or(0)`), which copies the wrong bytes; so a
/// one-row copy names the row's length rounded up to `COPY_BYTES_PER_ROW_ALIGNMENT`, which wgpu
/// accepts for a single row (it reads only the row's own bytes, `get_buffer_texture_copy_info`).
pub fn row_copies(
    at: TexelBox,
    offset: u64,
    rows: Rows,
    block_height: u32,
) -> Vec<(wgpu::TexelCopyBufferLayout, TexelBox)> {
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    if rows.row_bytes.is_multiple_of(align) {
        let layout = wgpu::TexelCopyBufferLayout {
            offset,
            bytes_per_row: Some(rows.row_bytes),
            rows_per_image: Some(rows.rows),
        };
        return vec![(layout, at)];
    }
    let stride = rows.row_bytes.checked_next_multiple_of(align);
    let mut out = Vec::new();
    let mut place = offset;
    for image in 0..rows.images {
        for row in 0..rows.rows {
            let part = TexelBox {
                origin: starquake_render::resource::Origin3d {
                    x: at.origin.x,
                    y: at.origin.y.saturating_add(row.saturating_mul(block_height)),
                    z: at.origin.z.saturating_add(image),
                },
                extent: starquake_render::resource::Extent3d {
                    width: at.extent.width,
                    height: block_height,
                    depth_or_layers: 1,
                },
                ..at
            };
            out.push((
                wgpu::TexelCopyBufferLayout {
                    offset: place,
                    bytes_per_row: stride,
                    rows_per_image: Some(1),
                },
                part,
            ));
            place = place.saturating_add(rows.row_bytes.into());
        }
    }
    out
}

#[cfg(test)]
mod tests;
