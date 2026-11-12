//! The `Backend` trait used as the GPU crate will use it: as a shared `Arc<dyn Backend>`, by
//! several threads, with descriptions refused through `validate` before any handle exists.
//!
//! The backend here is a minimal stand-in (the recording backend is slice R1); it shows the
//! trait is object-safe, `Send + Sync`, and carries `RenderError` and the command list through.

use starquake_render::caps::StageSupport;
use starquake_render::caps::{DynamicState, Features, FormatFeatures, FormatTable, Limits};
use starquake_render::command::{BufferSlice, CopyRegion};
use starquake_render::validate;
use starquake_render::{
    Backend, BufferDesc, BufferId, BufferUsage, Caps, Command, CommandList, ComputePipelineDesc,
    FenceValue, GraphicsPipelineDesc, MemoryBudget, PipelineId, PipelineStatus, PresentFeedback,
    PresentTarget, Readback, ReadbackId, RenderError, ResourceId, SampleCounts, SamplerDesc,
    SamplerId, ShaderDesc, ShaderId, ShaderInterface, ShaderSource, ShaderStage, StagingSlice,
    TextureDesc, TextureId, ViewDesc, ViewId, WaitOutcome,
};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

/// Hands out ids after validating descriptions; completes every list on submit.
struct Minimal {
    caps: Caps,
    state: Mutex<State>,
}

#[derive(Default)]
struct State {
    buffers: Vec<BufferDesc>,
    next: u32,
    fence: u64,
    lists: Vec<CommandList>,
}

impl Minimal {
    fn new(caps: Caps) -> Self {
        Self {
            caps,
            state: Mutex::new(State::default()),
        }
    }

    fn state(&self) -> std::sync::MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn next_index(&self) -> u32 {
        let mut s = self.state();
        s.next += 1;
        s.next
    }
}

impl Backend for Minimal {
    fn caps(&self) -> &Caps {
        &self.caps
    }
    fn budget(&self) -> MemoryBudget {
        MemoryBudget {
            allowed: self.caps.limits.max_buffer_size,
            used: self.state().buffers.iter().map(|b| b.size).sum(),
        }
    }
    fn create_buffer(&self, desc: &BufferDesc) -> Result<BufferId, RenderError> {
        validate::validate_buffer(&self.caps, desc)?;
        let mut s = self.state();
        s.buffers.push(*desc);
        let index = u32::try_from(s.buffers.len()).map_err(|_| RenderError::Overflow {
            what: "buffer count",
        })?;
        Ok(BufferId::new(index, 0))
    }
    fn create_texture(&self, desc: &TextureDesc) -> Result<TextureId, RenderError> {
        validate::validate_texture(&self.caps, desc)?;
        Ok(TextureId::new(self.next_index(), 0))
    }
    fn create_view(&self, _texture: TextureId, _desc: &ViewDesc) -> Result<ViewId, RenderError> {
        Ok(ViewId::new(self.next_index(), 0))
    }
    fn create_sampler(&self, desc: &SamplerDesc) -> Result<SamplerId, RenderError> {
        validate::validate_sampler(&self.caps, desc)?;
        Ok(SamplerId::new(self.next_index(), 0))
    }
    fn create_shader(&self, desc: ShaderDesc) -> Result<ShaderId, RenderError> {
        validate::validate_shader(&self.caps, &desc)?;
        Ok(ShaderId::new(self.next_index(), 0))
    }
    fn destroy(&self, _resource: ResourceId, _after: FenceValue) {}
    fn create_graphics_pipeline(
        &self,
        desc: GraphicsPipelineDesc,
    ) -> Result<PipelineId, RenderError> {
        validate::validate_graphics_pipeline(&self.caps, &desc)?;
        Ok(PipelineId::new(self.next_index(), 0))
    }
    fn create_compute_pipeline(
        &self,
        _desc: ComputePipelineDesc,
    ) -> Result<PipelineId, RenderError> {
        Ok(PipelineId::new(self.next_index(), 0))
    }
    fn pipeline_status(&self, _id: PipelineId) -> PipelineStatus {
        PipelineStatus::Ready
    }
    fn staging(&self, size: u64) -> Result<StagingSlice, RenderError> {
        validate::validate_staging_size(&self.caps, size)?;
        let len = usize::try_from(size).map_err(|_| RenderError::Overflow {
            what: "staging size",
        })?;
        Ok(StagingSlice::from_bytes(vec![0; len]))
    }
    fn submit(&self, list: CommandList) -> Result<FenceValue, RenderError> {
        let mut s = self.state();
        s.fence += 1;
        s.lists.push(list);
        Ok(FenceValue::new(s.fence))
    }
    fn completed(&self) -> FenceValue {
        FenceValue::new(self.state().fence)
    }
    fn wait(&self, value: FenceValue, _timeout: Duration) -> Result<WaitOutcome, RenderError> {
        Ok(if value <= self.completed() {
            WaitOutcome::Reached
        } else {
            WaitOutcome::TimedOut
        })
    }
    fn take_readback(&self, _id: ReadbackId) -> Option<Readback> {
        Some(Readback::Unknown)
    }
    fn set_present_target(&self, _target: PresentTarget) -> Result<(), RenderError> {
        Ok(())
    }
    fn take_present_feedback(&self) -> Vec<PresentFeedback> {
        Vec::new()
    }
}

fn caps() -> Caps {
    Caps {
        limits: Limits {
            max_buffer_size: 1 << 16,
            max_staging_size: 1 << 12,
            max_texture_dimension_1d: 1024,
            max_texture_dimension_2d: 1024,
            max_texture_dimension_3d: 256,
            max_texture_array_layers: 64,
            max_sampler_anisotropy: 16,
            max_vertex_attributes: 16,
            max_vertex_buffers: 8,
            max_vertex_buffer_stride: 2048,
            max_color_attachments: 8,
            max_viewports: 1,
            max_bind_groups: 4,
            max_bindings_per_bind_group: 16,
            max_compute_workgroups_per_dimension: 65_535,
        },
        features: Features::EMPTY,
        stages: StageSupport::EMPTY,
        dynamic_state: DynamicState::VIEWPORT | DynamicState::SCISSOR,
        sample_counts: SampleCounts::X1,
        view_swizzle: false,
        formats: FormatTable::uniform(FormatFeatures::ALL),
    }
}

#[test]
fn shared_backend_across_threads() {
    let backend: Arc<dyn Backend> = Arc::new(Minimal::new(caps()));
    let max = backend.caps().limits.max_buffer_size;
    std::thread::scope(|scope| {
        for _ in 0..4 {
            let backend = Arc::clone(&backend);
            scope.spawn(move || {
                for _ in 0..8 {
                    let desc = BufferDesc {
                        size: 16,
                        usage: BufferUsage::UNIFORM,
                    };
                    backend.create_buffer(&desc).expect("valid buffer");
                }
            });
        }
    });
    assert_eq!(
        backend.budget(),
        MemoryBudget {
            allowed: max,
            used: 4 * 8 * 16
        }
    );
    let too_big = BufferDesc {
        size: max + 1,
        usage: BufferUsage::UNIFORM,
    };
    assert_eq!(
        backend.create_buffer(&too_big).map_err(|e| e.to_string()),
        Err(format!("buffer size {} is outside 1..={max}", max + 1))
    );
}

#[test]
fn refused_stage_and_staging() {
    let backend: Arc<dyn Backend> = Arc::new(Minimal::new(caps()));
    let geometry = ShaderDesc {
        stage: ShaderStage::Geometry,
        entry: "main".to_owned(),
        source: ShaderSource::SpirV(Vec::new()),
        interface: ShaderInterface::default(),
    };
    assert_eq!(
        backend.create_shader(geometry).map_err(|e| e.to_string()),
        Err("the device cannot run geometry shaders".to_owned())
    );
    let max = backend.caps().limits.max_staging_size;
    assert_eq!(
        backend.staging(max).map(|s| s.len()),
        Ok(usize::try_from(max).expect("small"))
    );
    assert_eq!(
        backend.staging(max + 1).map(|s| s.len()),
        Err(RenderError::OutOfRange {
            what: "staging size",
            value: max + 1,
            min: 1,
            max
        })
    );
}

#[test]
fn submit_upload_and_wait() {
    let backend: Arc<dyn Backend> = Arc::new(Minimal::new(caps()));
    let desc = BufferDesc {
        size: 64,
        usage: BufferUsage::COPY_DST | BufferUsage::VERTEX,
    };
    let buffer = backend.create_buffer(&desc).expect("valid");
    let mut staged = backend.staging(16).expect("valid");
    staged.bytes_mut().copy_from_slice(&[7; 16]);
    let slice = BufferSlice {
        buffer,
        offset: 48,
        size: 16,
    };
    validate::validate_upload(
        u64::try_from(staged.len()).expect("small"),
        validate::Resolved::Buffer {
            desc: &desc,
            slice: &slice,
        },
    )
    .expect("fits");
    let mut list = CommandList::new();
    list.push(Command::Upload {
        from: staged,
        to: CopyRegion::Buffer(slice),
    });
    let fence = backend.submit(list).expect("submitted");
    assert_eq!(fence, FenceValue::new(1));
    assert_eq!(backend.completed(), fence);
    assert_eq!(
        backend.wait(fence, Duration::from_millis(1)),
        Ok(WaitOutcome::Reached)
    );
    assert_eq!(
        backend.wait(fence.next().expect("next"), Duration::from_millis(1)),
        Ok(WaitOutcome::TimedOut)
    );
    assert_eq!(
        backend.take_readback(ReadbackId::new(1, 0)),
        Some(Readback::Unknown)
    );
    assert_eq!(backend.set_present_target(PresentTarget::Offscreen), Ok(()));
    assert_eq!(backend.take_present_feedback(), Vec::new());
    assert_eq!(
        backend.pipeline_status(PipelineId::new(1, 0)),
        PipelineStatus::Ready
    );
    backend.destroy(ResourceId::Buffer(buffer), fence);
}
