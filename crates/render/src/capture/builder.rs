//! `CaptureBuilder`: captures written through the real writer with no backend, for other
//! crates' tests (docs/specs/render-backend.md, "Render capture v2", "The format": "A builder in
//! test support"). Behind the `test-support` feature, as other crates do it
//! (`docs/specs/testing.md`, "Shared test support"), and in this crate's own tests.
//!
//! Its methods mirror the trait calls. It hands out ids itself, every one from one counter
//! (slot 1, 2, ... in generation 0), and fence values 1, 2, ... as a backend would. A refused
//! call, a result or anything else the methods do not cover is written with
//! [`CaptureBuilder::record`].
#![cfg(any(test, feature = "test-support"))]

use super::CaptureError;
use super::record::{
    CapturedTarget, Create, CreateShader, CreateView, Created, Destroy, FrameResult,
    PipelineOutcome, PipelineResult, ReadbackResult, Record, Start, TargetRecord, WaitSetting,
};
use super::write::{CaptureBuffer, CaptureSink, CaptureWriter};
use crate::backend::{PresentTarget, Readback};
use crate::caps::Caps;
use crate::command::CommandList;
use crate::frame::Frame;
use crate::id::{
    BufferId, FenceValue, PipelineId, ReadbackId, ResourceId, SamplerId, ShaderId, TextureId,
    ViewId,
};
use crate::pipeline::{ComputePipelineDesc, GraphicsPipelineDesc};
use crate::resource::{BufferDesc, SamplerDesc, TextureDesc, ViewDesc};
use crate::shader::ShaderDesc;
use std::sync::Arc;

/// Builds a capture call by call (module docs).
#[derive(Debug)]
pub struct CaptureBuilder {
    sink: Arc<CaptureBuffer>,
    writer: CaptureWriter,
    /// The slot the next id gets.
    slot: u32,
    /// The last fence value handed out.
    fence: FenceValue,
}

impl CaptureBuilder {
    /// A capture of a device with `caps`, from build `"test"` of backend `"builder"`, made with
    /// draws waiting for their pipelines, with no memory or size limit.
    pub fn new(caps: Caps) -> Result<Self, CaptureError> {
        Self::with_start(Start {
            build: "test".to_owned(),
            backend: "builder".to_owned(),
            caps,
            wait: WaitSetting::Wait,
            memory_limit: u64::MAX,
        })
    }

    /// A capture starting with `start`, with no size limit.
    pub fn with_start(start: Start) -> Result<Self, CaptureError> {
        let sink = Arc::new(CaptureBuffer::new());
        let shared: Arc<dyn CaptureSink> = Arc::<CaptureBuffer>::clone(&sink);
        let writer = CaptureWriter::new(shared, start, u64::MAX)?;
        Ok(Self {
            sink,
            writer,
            slot: 0,
            fence: FenceValue::ZERO,
        })
    }

    /// Writes any record as it is (refusals, results, hand-made faults the writer accepts).
    pub fn record(&mut self, record: &Record) -> Result<(), CaptureError> {
        self.writer.write(record)
    }

    /// The next slot.
    fn slot(&mut self) -> u32 {
        self.slot = self.slot.saturating_add(1);
        self.slot
    }

    /// `create_buffer`, accepted.
    pub fn create_buffer(&mut self, desc: BufferDesc) -> Result<BufferId, CaptureError> {
        let id = BufferId::new(self.slot(), 0);
        self.record(&Record::CreateBuffer(Create {
            desc,
            result: Created::Made(id),
        }))?;
        Ok(id)
    }

    /// `create_texture`, accepted.
    pub fn create_texture(&mut self, desc: TextureDesc) -> Result<TextureId, CaptureError> {
        let id = TextureId::new(self.slot(), 0);
        self.record(&Record::CreateTexture(Create {
            desc,
            result: Created::Made(id),
        }))?;
        Ok(id)
    }

    /// `create_view`, accepted.
    pub fn create_view(
        &mut self,
        texture: TextureId,
        desc: ViewDesc,
    ) -> Result<ViewId, CaptureError> {
        let id = ViewId::new(self.slot(), 0);
        self.record(&Record::CreateView(CreateView {
            texture,
            desc,
            result: Created::Made(id),
        }))?;
        Ok(id)
    }

    /// `create_sampler`, accepted.
    pub fn create_sampler(&mut self, desc: SamplerDesc) -> Result<SamplerId, CaptureError> {
        let id = SamplerId::new(self.slot(), 0);
        self.record(&Record::CreateSampler(Create {
            desc,
            result: Created::Made(id),
        }))?;
        Ok(id)
    }

    /// `create_shader`, accepted: its code recorded as the capturing backend records it.
    pub fn create_shader(&mut self, desc: &ShaderDesc) -> Result<ShaderId, CaptureError> {
        let code = self.writer.shader_code(desc)?;
        let id = ShaderId::new(self.slot(), 0);
        self.record(&Record::CreateShader(CreateShader {
            stage: desc.stage,
            entry: desc.entry.clone(),
            interface: desc.interface.clone(),
            code,
            result: Created::Made(id),
        }))?;
        Ok(id)
    }

    /// `create_graphics_pipeline`, accepted.
    pub fn create_graphics_pipeline(
        &mut self,
        desc: GraphicsPipelineDesc,
    ) -> Result<PipelineId, CaptureError> {
        let id = PipelineId::new(self.slot(), 0);
        self.record(&Record::CreateGraphicsPipeline(Create {
            desc,
            result: Created::Made(id),
        }))?;
        Ok(id)
    }

    /// `create_compute_pipeline`, accepted.
    pub fn create_compute_pipeline(
        &mut self,
        desc: ComputePipelineDesc,
    ) -> Result<PipelineId, CaptureError> {
        let id = PipelineId::new(self.slot(), 0);
        self.record(&Record::CreateComputePipeline(Create {
            desc,
            result: Created::Made(id),
        }))?;
        Ok(id)
    }

    /// `destroy`.
    pub fn destroy(&mut self, id: ResourceId, after: FenceValue) -> Result<(), CaptureError> {
        self.record(&Record::Destroy(Destroy { id, after }))
    }

    /// `submit`, accepted with the next fence value.
    pub fn submit(&mut self, list: &CommandList) -> Result<FenceValue, CaptureError> {
        // At the counter's end this repeats the last value, which the writer refuses.
        let fence = FenceValue::new(self.fence.get().saturating_add(1));
        let encoded = self.writer.encode_list(list)?;
        self.writer.write_submit(encoded, &Created::Made(fence))?;
        self.fence = fence;
        Ok(fence)
    }

    /// `set_present_target`, accepted.
    pub fn set_present_target(&mut self, target: &PresentTarget) -> Result<(), CaptureError> {
        self.record(&Record::PresentTarget(TargetRecord {
            target: CapturedTarget::of(target),
            result: Created::Made(()),
        }))
    }

    /// What a `take_readback` returned.
    pub fn readback_result(
        &mut self,
        id: ReadbackId,
        value: Option<Readback>,
    ) -> Result<(), CaptureError> {
        self.record(&Record::ReadbackResult(ReadbackResult { id, value }))
    }

    /// A pipeline's settled status.
    pub fn pipeline_outcome(
        &mut self,
        pipeline: PipelineId,
        outcome: PipelineResult,
    ) -> Result<(), CaptureError> {
        self.record(&Record::PipelineOutcome(PipelineOutcome {
            pipeline,
            outcome,
        }))
    }

    /// A dumped frame.
    pub fn frame_result(&mut self, frame: &Frame) -> Result<(), CaptureError> {
        self.record(&Record::FrameResult(FrameResult::of(frame)))
    }

    /// Ends the capture with `End` and returns its bytes.
    pub fn finish(mut self) -> Result<Vec<u8>, CaptureError> {
        self.writer.finish()?;
        Ok(self.sink.bytes())
    }
}
