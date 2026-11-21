//! Compute dispatch: each dispatch the QMD scheduler decides on becomes a `Command::Dispatch`
//! (gpu-protocol spec, "Compute: QMD 3.0", step 3, and "What the back hands the render
//! backend"), and each QMD release a fenced release (step 4).
//!
//! [`ComputeSink`] is the executor's [`QmdSink`]. For a [`Dispatch`] it:
//!
//! 1. asks the shader cache for the program ([`ShaderLookup::lookup`] with one
//!    [`ProgramRequest`] of [`GuestStage::Compute`] at the QMD's program address), with a
//!    [`VariantKey`] of the QMD's block as the workgroup size and its shared memory size, every
//!    other field its default (a compute program reads no vertex attributes or colour targets,
//!    and no emulation flag applies to compute: [`starquake_render::emulation`] derives them from
//!    draw state only). A `Pending` answer follows the wait-while-compiling setting as a draw's
//!    does; `Rejected`, or a ticket still pending, skips the dispatch;
//! 2. makes the set's one shader a compute shader ([`HostShaders::get_as`]): a ready answer
//!    for a compute request is taken as the compute shader whatever its host stage says. The
//!    shader crate's production translator now translates compute programs (its driver takes
//!    vertex, pixel and compute, `only_the_stages_the_lowering_takes_translate` in
//!    `starquake_shader`, and the lowering makes a [`starquake_shader::host::HostStage`] of
//!    `Compute`), so a production lookup for compute is ready when the program lowers;
//! 3. binds each resource the shader uses: a constant buffer bank below
//!    [`crate::back::qmd::CONSTANT_BUFFERS`] from the QMD's binding of it, through the buffer
//!    cache (`DispatchNativeCompute` binds the QMD's eight constant buffers,
//!    `eden-nx2/src/video_core/engines/kepler_compute.cpp` @ 50fa86bb). A bank the QMD does not
//!    bind valid with a size above zero, a texture, a sampler or the driver constants is
//!    [`ComputeError`] (the compute class's texture pools are not read here);
//! 4. finds the compute pipeline ([`PipelineCache::lookup_compute`]); one that failed skips the
//!    dispatch;
//! 5. records the `Command::Dispatch` with the grid as its workgroup counts, after any uploads
//!    the bindings needed, in one batch ([`Fences::record_list`]).
//!
//! A grid past the host's `max_compute_workgroups_per_dimension` is refused
//! ([`ComputeError::Grid`]) before anything is resolved: the backend would refuse it at submit and
//! lose the whole list. A divergence: Eden-NX2 hands any grid to the host API.
//!
//! A skipped or refused dispatch answers [`Program::Rejected`], so the QMD still completes and
//! releases (row 61), and is counted in [`ExecCounters::dispatches_skipped`]. Its releases are
//! attached as fenced releases of the signalling channel's address space as they come, after the
//! dispatch, so they wait for it (the ruling of 2026-11-18: the QMD's releases carry the issuing
//! channel; the QMD table is shared, and each signal is read and released through the memory of
//! the channel that sent it).

use std::ops::Range;
use std::sync::Arc;

use starquake_render::command::{Binding, BindingResource, DispatchDesc, DispatchParams};
use starquake_render::pipeline::ComputePipelineDesc;
use starquake_render::shader::ShaderStage;
use starquake_render::{Backend, Command, CommandList, FenceValue};
use starquake_shader::exchange::TextureKind;
use starquake_shader::lookup::{EngineState, GuestStage, ProgramRequest, VariantKey};
use thiserror::Error;

use super::sinks::pages;
use super::{ExecCounters, ShaderCache};
use crate::back::buffer::{BufferCache, BufferError};
use crate::back::draw::shaders::{
    GuestPrograms, HostShaders, ResourceSource, ResourceUse, SetLookup,
};
use crate::back::draw::{DrawError, WaitWhileCompiling};
use crate::back::fence::{FailureLog, FenceError, FencedAction, Fences};
use crate::back::pipeline::{Lookup as PipelineLookup, PipelineCache, PipelineCacheError};
use crate::back::qmd::{ConstantBuffer, Dispatch, Outcome, Program, QmdSink};
use crate::back::texture::TextureCache;
use crate::interface::AsId;
use crate::memory::GpuMemory;
use crate::ring::Release;

/// Why a dispatch could not be built. The QMD still completes and releases.
#[derive(Debug, Error, PartialEq)]
pub enum ComputeError {
    /// The grid has more workgroups along an axis than the host allows.
    #[error("the dispatch's grid {grid:?} passes the host's {limit} workgroups per dimension")]
    Grid {
        /// The grid.
        grid: [u32; 3],
        /// The host's limit.
        limit: u32,
    },
    /// The ready set does not hold exactly one shader with its information.
    #[error("the compute shader set holds {shaders} shaders and {infos} stage records, not one")]
    BadSet {
        /// Shaders in the set.
        shaders: usize,
        /// Stage records beside them.
        infos: usize,
    },
    /// The shader reads a constant buffer bank the QMD does not bind.
    #[error("the compute shader reads constant buffer {bank}, which the QMD does not bind")]
    UnboundConstantBuffer {
        /// The bank.
        bank: u8,
    },
    /// The shader binds a resource the compute path does not supply.
    #[error("the compute shader binds {0}, which compute dispatches do not supply yet")]
    Resource(&'static str),
    /// Making the host shader failed.
    #[error("compute shader: {0}")]
    Shader(DrawError),
    /// The buffer cache refused.
    #[error("buffer cache: {0}")]
    Buffer(#[from] BufferError),
    /// The pipeline-key cache refused.
    #[error("pipeline cache: {0}")]
    Pipeline(#[from] PipelineCacheError),
    /// The fence path failed to record the dispatch or attach a release.
    #[error("fence path: {0}")]
    Fence(#[from] FenceError),
}

/// What [`EngineState`] answers for a compute program: constant buffer words from the QMD's
/// bindings, read from guest memory; no texture kinds (the compute class's texture pools are not
/// read here, so every handle answers `None`).
pub(super) struct ComputeAnswers<'a> {
    pub(super) memory: &'a dyn GpuMemory,
    pub(super) constant_buffers: &'a [ConstantBuffer],
}

impl EngineState for ComputeAnswers<'_> {
    fn cbuf_word(&self, _stage: GuestStage, bank: u8, offset: u32) -> Option<u32> {
        let binding = self.constant_buffers.get(usize::from(bank))?;
        if !binding.valid {
            return None;
        }
        // All four bytes inside the bound size, as the draw path's answers require.
        let end = offset.checked_add(4)?;
        if end > binding.size {
            return None;
        }
        let address = binding.address.checked_add(u64::from(offset))?;
        let mut bytes = [0; 4];
        self.memory.read(address, &mut bytes).ok()?;
        Some(u32::from_le_bytes(bytes))
    }

    fn texture_kind(&self, _stage: GuestStage, _handle: u32) -> Option<TextureKind> {
        None
    }
}

/// The executor's [`QmdSink`] for one signal (module docs).
pub(super) struct ComputeSink<'a> {
    pub(super) space: AsId,
    pub(super) memory: &'a Arc<dyn GpuMemory>,
    pub(super) buffers: &'a mut BufferCache,
    pub(super) textures: &'a mut TextureCache,
    /// The newest fence the space's caches were told of.
    pub(super) notified: &'a mut FenceValue,
    pub(super) shaders: &'a mut dyn ShaderCache,
    pub(super) host: &'a mut HostShaders,
    pub(super) pipelines: &'a mut PipelineCache,
    pub(super) fences: &'a mut Fences,
    pub(super) backend: &'a dyn Backend,
    pub(super) setting: WaitWhileCompiling,
    pub(super) counters: &'a mut ExecCounters,
    pub(super) scratch: &'a mut CommandList,
    /// The inline QMD's bytes, written before the first callback: dropped from the caches at
    /// the first callback, or by the executor after the scheduler returns, whatever it returned
    /// (`tests::a_refused_inline_qmd_is_still_invalidated_in_the_caches`).
    pub(super) inline: Option<Range<u64>>,
    /// The first fence-path failure, reported when the signal is done.
    pub(super) error: Option<FenceError>,
}

impl ComputeSink<'_> {
    /// Drops the inline QMD's bytes from the caches, once.
    pub(super) fn written(&mut self) {
        if let Some(range) = self.inline.take() {
            self.buffers.invalidate(range.clone());
            self.textures.invalidate(range.clone());
            if let Some(pages) = pages(&range) {
                self.shaders.invalidate(pages);
            }
        }
    }

    /// Keeps the first fence-path failure.
    fn failed(&mut self, error: FenceError) {
        if self.error.is_none() {
            self.error = Some(error);
        }
    }

    /// Builds and records `dispatch` (module docs); `Ok(false)` when it is skipped.
    fn build(&mut self, dispatch: &Dispatch) -> Result<bool, ComputeError> {
        let limit = self
            .backend
            .caps()
            .limits
            .max_compute_workgroups_per_dimension;
        if dispatch.grid.iter().any(|&n| n > limit) {
            return Err(ComputeError::Grid {
                grid: dispatch.grid,
                limit,
            });
        }
        // A release earlier in the signal may have closed the list for a waiter: the caches
        // learn of it before this dispatch uses them (the executor's module docs).
        super::catch_up(
            self.buffers,
            self.textures,
            self.notified,
            self.fences.newest_submitted(),
        );
        let requests = [ProgramRequest {
            stage: GuestStage::Compute,
            address: dispatch.program,
        }];
        let variant = VariantKey {
            workgroup_size: dispatch.block,
            shared_memory_size: dispatch.shared_memory,
            ..VariantKey::default()
        };
        let answers = ComputeAnswers {
            memory: &**self.memory,
            constant_buffers: &dispatch.constant_buffers,
        };
        let programs = GuestPrograms(&**self.memory);
        let mut lookup = self
            .shaders
            .lookup(&requests, &variant, &programs, &answers);
        if let SetLookup::Pending(ticket) = lookup {
            lookup = match self.setting {
                WaitWhileCompiling::Wait => self.shaders.wait(ticket),
                WaitWhileCompiling::Skip => return Ok(false),
            };
        }
        let set = match lookup {
            SetLookup::Ready(set) => set,
            SetLookup::Pending(_) | SetLookup::Rejected(_) => return Ok(false),
        };
        let (Some(shader), Some(info), 1, 1) = (
            set.shaders.stages.first(),
            set.stages.first(),
            set.shaders.stages.len(),
            set.stages.len(),
        ) else {
            return Err(ComputeError::BadSet {
                shaders: set.shaders.stages.len(),
                infos: set.stages.len(),
            });
        };
        let shader_id = self
            .host
            .get_as(shader, info, ShaderStage::Compute)
            .map_err(ComputeError::Shader)?;
        let bindings = self.bindings(dispatch, &info.resources)?;
        let pipeline = match self
            .pipelines
            .lookup_compute(ComputePipelineDesc { shader: shader_id }, info.stamp)?
        {
            PipelineLookup::Ready(id) | PipelineLookup::Compiling(id) => id,
            PipelineLookup::Failed(_) => return Ok(false),
        };
        let [x, y, z] = dispatch.grid;
        self.scratch.push(Command::Dispatch(DispatchDesc {
            pipeline,
            bindings,
            params: DispatchParams::Direct { x, y, z },
        }));
        self.fences.record_list(self.scratch)?;
        Ok(true)
    }

    /// The bindings of `resources` for `dispatch` (module docs, step 3).
    fn bindings(
        &mut self,
        dispatch: &Dispatch,
        resources: &[ResourceUse],
    ) -> Result<Vec<Binding>, ComputeError> {
        let mut bindings = Vec::with_capacity(resources.len());
        for resource in resources {
            let bound = match resource.source {
                ResourceSource::ConstantBuffer { bank, size } => {
                    let binding = dispatch
                        .constant_buffers
                        .get(usize::from(bank))
                        .filter(|b| b.valid && b.size > 0)
                        .ok_or(ComputeError::UnboundConstantBuffer { bank })?;
                    let len = if size == 0 {
                        binding.size
                    } else {
                        size.min(binding.size)
                    };
                    BindingResource::Buffer(self.buffers.bind(
                        &**self.memory,
                        binding.address,
                        u64::from(len),
                        self.scratch,
                    )?)
                }
                ResourceSource::Texture(_) => return Err(ComputeError::Resource("a texture")),
                ResourceSource::Sampler(_) => return Err(ComputeError::Resource("a sampler")),
                ResourceSource::DriverConstants => {
                    return Err(ComputeError::Resource("the driver constants"));
                }
            };
            bindings.push(Binding {
                group: resource.slot.group,
                binding: resource.slot.binding,
                resource: bound,
            });
        }
        Ok(bindings)
    }
}

impl QmdSink for ComputeSink<'_> {
    fn dispatch(&mut self, dispatch: &Dispatch) -> Program {
        self.written();
        let built = self.build(dispatch);
        // Uploads made before a failure are the caches' own consistent work: recorded.
        if let Err(error) = self.fences.record_list(self.scratch) {
            self.failed(error);
        }
        match built {
            Ok(true) => {
                self.counters.dispatches = self.counters.dispatches.saturating_add(1);
                Program::Accepted
            }
            Ok(false) => {
                self.counters.dispatches_skipped =
                    self.counters.dispatches_skipped.saturating_add(1);
                tracing::debug!(
                    qmd = dispatch.qmd,
                    program = dispatch.program,
                    "a dispatch was skipped: its program is rejected or pending, or its pipeline failed"
                );
                Program::Rejected
            }
            Err(error) => {
                self.counters.dispatches_skipped =
                    self.counters.dispatches_skipped.saturating_add(1);
                tracing::warn!(qmd = dispatch.qmd, %error, "a dispatch could not be built");
                if let ComputeError::Fence(error) = error {
                    self.failed(error);
                }
                Program::Rejected
            }
        }
    }

    fn release(&mut self, release: Release) {
        self.written();
        let action = FencedAction::Release {
            memory: Arc::clone(self.memory),
            space: self.space,
            release,
            failure: FailureLog::Warn,
        };
        if let Err(error) = self.fences.attach(action) {
            self.failed(error);
        }
    }

    fn completed(&mut self, _qmd: u64, _outcome: Outcome) {
        self.written();
    }
}
