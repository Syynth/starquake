//! Render capture v2: golden bytes for every record kind, every reading check and its order,
//! every bound at its edge, a generated round trip, the capturing backend and replay.

mod bounds;
mod golden;
mod read;
mod replay;
mod roundtrip;
mod writer;
mod wrap;

use super::{
    CaptureBuffer, CaptureError, CaptureLimits, CaptureReader, CaptureSink, CaptureWriter, Record,
    Start, WaitSetting,
};
use crate::backend::{
    Backend, MemoryBudget, PipelineStatus, PresentFeedback, PresentTarget, Readback, WaitOutcome,
};
use crate::caps::{
    Caps, DynamicState, Features, FormatFeatures, FormatTable, Limits, SampleCounts, StageSupport,
};
use crate::command::{CommandList, StagingSlice};
use crate::error::RenderError;
use crate::id::{
    BufferId, FenceValue, PipelineId, ReadbackId, ResourceId, SamplerId, ShaderId, TextureId,
    ViewId,
};
use crate::pipeline::{ComputePipelineDesc, GraphicsPipelineDesc};
use crate::recording::RecordingBackend;
use crate::format::PixelFormat;
use crate::resource::{
    AddressMode, BorderColor, BufferDesc, Extent3d, FilterMode, ReductionMode, SamplerDesc,
    Swizzle, TextureDesc, TextureDimension, TextureUsage, ViewDesc, ViewDimension,
};
use crate::shader::{ShaderDesc, ShaderInterface, ShaderOrigin, ShaderSource, ShaderStage};
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// A device with room for what the tests make.
pub(super) fn caps() -> Caps {
    Caps {
        limits: Limits {
            max_buffer_size: 1 << 16,
            max_staging_size: 1 << 12,
            max_texture_dimension_1d: 256,
            max_texture_dimension_2d: 256,
            max_texture_dimension_3d: 64,
            max_texture_array_layers: 8,
            max_sampler_anisotropy: 16,
            max_vertex_attributes: 16,
            max_vertex_buffers: 8,
            max_vertex_buffer_stride: 2048,
            max_color_attachments: 8,
            max_viewports: 4,
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
        sample_counts: SampleCounts::X1,
        view_swizzle: true,
        formats: FormatTable::uniform(FormatFeatures::ALL),
    }
}

/// A `Start` for the tests' device.
pub(super) fn start() -> Start {
    Start {
        build: "test".to_owned(),
        backend: "recording".to_owned(),
        caps: caps(),
        wait: WaitSetting::Wait,
        memory_limit: 1 << 20,
    }
}

/// A writer with `limits` into a fresh buffer, and the buffer.
pub(super) fn writer_with(limits: CaptureLimits) -> (CaptureWriter, Arc<CaptureBuffer>) {
    let buffer = Arc::new(CaptureBuffer::new());
    let sink: Arc<dyn CaptureSink> = Arc::<CaptureBuffer>::clone(&buffer);
    let writer =
        CaptureWriter::with_limits(sink, start(), u64::MAX, limits).expect("the start fits");
    (writer, buffer)
}

/// A writer with the shipped bounds.
pub(super) fn writer() -> (CaptureWriter, Arc<CaptureBuffer>) {
    writer_with(CaptureLimits::SHIPPED)
}

/// The header and `Start` the tests' writer writes.
pub(super) fn prefix() -> Vec<u8> {
    writer().1.bytes()
}

/// Every record of `bytes`, read with `limits`, or the first error.
pub(super) fn read_with(bytes: &[u8], limits: CaptureLimits) -> Result<Vec<Record>, CaptureError> {
    let len = u64::try_from(bytes.len()).expect("small");
    let mut reader = CaptureReader::with_limits(bytes, len, limits)?;
    let mut records = Vec::new();
    while let Some(record) = reader.next_record()? {
        records.push(record);
    }
    Ok(records)
}

/// Every record of `bytes`, with the shipped bounds.
pub(super) fn read(bytes: &[u8]) -> Result<Vec<Record>, CaptureError> {
    read_with(bytes, CaptureLimits::SHIPPED)
}

/// A record framed by hand: its tag, a one-byte length (below 128), and its body.
pub(super) fn framed(tag: u8, body: &[u8]) -> Vec<u8> {
    let len = u8::try_from(body.len()).expect("short body");
    assert!(len < 0x80, "a one-byte length");
    let mut out = vec![tag, len];
    out.extend_from_slice(body);
    out
}

/// A record of any length: its tag, its length as postcard writes a `u32`, and its body.
pub(super) fn framed_any(tag: u8, body: &[u8]) -> Vec<u8> {
    let mut out = vec![tag];
    out.extend(length(u64::try_from(body.len()).expect("small")));
    out.extend_from_slice(body);
    out
}

/// A length as a record's varint.
pub(super) fn length(len: u64) -> Vec<u8> {
    postcard::to_allocvec(&u32::try_from(len).expect("a u32 length")).expect("encode")
}

/// The `End` record.
pub(super) const END: [u8; 2] = [0, 0];

/// `prefix()`, then `records`, then `End`.
pub(super) fn file(records: &[&[u8]]) -> Vec<u8> {
    let mut out = prefix();
    for record in records {
        out.extend_from_slice(record);
    }
    out.extend_from_slice(&END);
    out
}

/// What a [`Spy`] does differently from the recording backend it wraps.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct Knobs {
    /// Every `create_buffer` returns the same id.
    pub(super) repeat_buffer: bool,
    /// Every pipeline reports `Compiling`.
    pub(super) compiling: bool,
    /// Every `create_buffer` reports a lost device.
    pub(super) lost: bool,
    /// Every `wait` times out.
    pub(super) time_out: bool,
}

/// A recording backend with [`Knobs`], which keeps every `destroy` call it is given. Where a
/// knob is off it behaves as the recording backend does, call for call.
#[derive(Debug)]
pub(super) struct Spy {
    pub(super) inner: RecordingBackend,
    knobs: Knobs,
    destroys: Mutex<Vec<(ResourceId, FenceValue)>>,
}

impl Spy {
    /// A spy over a recording backend for [`caps`] with 1 MiB of memory.
    pub(super) fn new(knobs: Knobs) -> Self {
        Self {
            inner: RecordingBackend::new(caps(), 1 << 20),
            knobs,
            destroys: Mutex::default(),
        }
    }

    /// The `destroy` calls so far.
    pub(super) fn destroys(&self) -> Vec<(ResourceId, FenceValue)> {
        self.destroys.lock().expect("unpoisoned").clone()
    }
}

impl Backend for Spy {
    fn caps(&self) -> &Caps {
        self.inner.caps()
    }
    fn budget(&self) -> MemoryBudget {
        self.inner.budget()
    }
    fn create_buffer(&self, desc: &BufferDesc) -> Result<BufferId, RenderError> {
        if self.knobs.lost {
            return Err(RenderError::DeviceLost);
        }
        if self.knobs.repeat_buffer {
            return Ok(BufferId::new(1, 1));
        }
        self.inner.create_buffer(desc)
    }
    fn create_texture(&self, desc: &TextureDesc) -> Result<TextureId, RenderError> {
        self.inner.create_texture(desc)
    }
    fn create_view(&self, texture: TextureId, desc: &ViewDesc) -> Result<ViewId, RenderError> {
        self.inner.create_view(texture, desc)
    }
    fn create_sampler(&self, desc: &SamplerDesc) -> Result<SamplerId, RenderError> {
        self.inner.create_sampler(desc)
    }
    fn create_shader(&self, desc: ShaderDesc) -> Result<ShaderId, RenderError> {
        self.inner.create_shader(desc)
    }
    fn destroy(&self, resource: ResourceId, after: FenceValue) {
        self.destroys
            .lock()
            .expect("unpoisoned")
            .push((resource, after));
        self.inner.destroy(resource, after);
    }
    fn create_graphics_pipeline(
        &self,
        desc: GraphicsPipelineDesc,
    ) -> Result<PipelineId, RenderError> {
        self.inner.create_graphics_pipeline(desc)
    }
    fn create_compute_pipeline(
        &self,
        desc: ComputePipelineDesc,
    ) -> Result<PipelineId, RenderError> {
        self.inner.create_compute_pipeline(desc)
    }
    fn pipeline_status(&self, id: PipelineId) -> PipelineStatus {
        if self.knobs.compiling {
            return PipelineStatus::Compiling;
        }
        self.inner.pipeline_status(id)
    }
    fn staging(&self, size: u64) -> Result<StagingSlice, RenderError> {
        self.inner.staging(size)
    }
    fn submit(&self, list: CommandList) -> Result<FenceValue, RenderError> {
        self.inner.submit(list)
    }
    fn completed(&self) -> FenceValue {
        self.inner.completed()
    }
    fn wait(&self, value: FenceValue, timeout: Duration) -> Result<WaitOutcome, RenderError> {
        if self.knobs.time_out {
            return Ok(WaitOutcome::TimedOut);
        }
        self.inner.wait(value, timeout)
    }
    fn take_readback(&self, id: ReadbackId) -> Option<Readback> {
        self.inner.take_readback(id)
    }
    fn set_present_target(&self, target: PresentTarget) -> Result<(), RenderError> {
        self.inner.set_present_target(target)
    }
    fn take_present_feedback(&self) -> Vec<PresentFeedback> {
        self.inner.take_present_feedback()
    }
}

/// A 2 by 2 RGBA8 texture with every usage.
pub(super) fn texture_desc() -> TextureDesc {
    TextureDesc {
        dimension: TextureDimension::D2,
        format: PixelFormat::A8B8G8R8Unorm,
        size: Extent3d {
            width: 2,
            height: 2,
            depth_or_layers: 1,
        },
        mip_levels: 1,
        sample_count: 1,
        usage: TextureUsage::ALL,
    }
}

/// A view of all of [`texture_desc`].
pub(super) fn view_desc() -> ViewDesc {
    ViewDesc {
        format: PixelFormat::A8B8G8R8Unorm,
        dimension: ViewDimension::D2,
        base_mip: 0,
        mip_count: 1,
        base_layer: 0,
        layer_count: 1,
        swizzle: Swizzle::IDENTITY,
    }
}

/// A plain sampler.
pub(super) fn sampler_desc() -> SamplerDesc {
    SamplerDesc {
        mag_filter: FilterMode::Linear,
        min_filter: FilterMode::Linear,
        mip_filter: FilterMode::Nearest,
        address: [AddressMode::Repeat; 3],
        border_color: BorderColor::TransparentBlack,
        lod_bias: 0.0,
        min_lod: 0.0,
        max_lod: 4.0,
        max_anisotropy: 1,
        compare: None,
        reduction: ReductionMode::Average,
    }
}

/// A shader with entry `main` and no interface.
pub(super) fn shader(
    stage: ShaderStage,
    source: ShaderSource,
    origin: Option<ShaderOrigin>,
) -> ShaderDesc {
    ShaderDesc {
        stage,
        entry: "main".to_owned(),
        source,
        interface: ShaderInterface::default(),
        origin,
    }
}

/// An empty, validated naga module.
pub(super) fn naga() -> ShaderSource {
    let module = naga::Module::default();
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&module)
    .expect("an empty module validates");
    ShaderSource::Naga {
        module: Box::new(module),
        info: Box::new(info),
    }
}
