//! Shaders and pipelines on the wgpu backend (docs/specs/render-backend.md, slice R6:
//! "Pipelines", "Shader emission", "Compiling without stalling a frame").
//!
//! # Shaders
//!
//! `create_shader` takes a [`ShaderSource::Naga`](starquake_render::ShaderSource::Naga) module
//! only (SPIR-V and MSL serve a later direct backend, [`NAGA_ONLY`]); it checks the description
//! against `Caps` (`validate::validate_shader`, which refuses a geometry or tessellation shader
//! with `RenderError::StageUnsupported`), checks that the module has an entry point of the
//! shader's name for its stage ([`NO_ENTRY_POINT`]), reads what the layouts and the requirements
//! need from the module ([`scan`]), and keeps the module. It does not touch wgpu: the GPU back
//! thread calls it, and the back thread never touches the host API (render-backend, "Where the
//! backend sits"). The first pipeline build that needs the shader hands the module to wgpu
//! (`wgpu::ShaderSource::Naga`, feature `naga-ir`), which validates it and writes the final code
//! itself; this crate runs no naga writer ("Shader emission"). The wgpu module is then shared by
//! every pipeline that uses the shader, and a module wgpu refused fails each of them with the
//! same reason.
//!
//! # Pipelines
//!
//! `create_graphics_pipeline` and `create_compute_pipeline` check the description, in this order,
//! and refuse it with the first problem found:
//!
//! 1. a compile pool exists ([`NO_COMPILE_POOL`]);
//! 2. the shared checks against `Caps` (`validate::validate_graphics_pipeline`: a geometry or
//!    tessellation stage is `RenderError::StageUnsupported` here, before any id is looked at);
//! 3. the device is not lost;
//! 4. each stage's shader is live and of its stage (`validate::validate_pipeline_stages`,
//!    `validate::validate_compute_stage`);
//! 5. the shaders' requirements ([`SUBGROUP_SIZE`], below);
//! 6. the layouts ([`layout`]);
//! 7. for a graphics pipeline, the fixed-function state ([`state`]).
//!
//! Then the id is allocated, the pipeline's status is `Compiling`, and one job goes to the
//! compile pool ([`CompilePool`]) before the call returns. wgpu's pipeline creation is
//! synchronous, so the job calls it on a pool worker: it creates the shaders' wgpu modules (once
//! per shader), the bind group layouts, the pipeline layout and the pipeline, inside wgpu error
//! scopes, and the status turns `Ready` when wgpu accepted them all and `Failed` when it reported
//! an error. A job that panics, or that the pool drops without running it, fails its pipeline
//! too, so a draw waiting for it is always released ([`CompilePool`]'s contract). A failed
//! pipeline is cached as failed: nothing builds it again (rows 90 and 112: cache rejections
//! negatively), and `pipeline_status` says `Failed` from then on.
//!
//! **Requirements.** A shader that uses subgroup operations or built-ins ([`scan`]) needs every
//! subgroup to be 32 invocations wide, the guest's warp (shader-pipeline, "What the render
//! backend (#14) gets": "It honours `requirements` (for example, a subgroup size of 32) or
//! refuses the pipeline"). wgpu cannot ask for a subgroup size, so a pipeline with such a shader
//! is refused unless the device has subgroups and reports a subgroup size range of exactly 32 to
//! 32 ([`subgroups_honoured`]). wgpu's Metal backend reports 4 to 64 on every Mac (wgpu-hal
//! 30.0.2, `src/metal/mod.rs`), so on the Mac every such pipeline is refused (an owner
//! question). The other requirements (f16, i64 and the binding-array features) are wgpu's to
//! check: it validates each module against the device's features, and a module that needs one
//! the device lacks fails its pipeline.
//!
//! # Draws: wait or skip
//!
//! The render thread meets a draw whose pipeline may still be compiling, and asks
//! [`crate::WgpuBackend::draw_pipeline`], which applies the GPU crate's wait-while-compiling
//! setting ([`WhileCompiling`], passed at construction; [`on_draw`] is the decision): under
//! `Wait` it blocks until the build settles, so pipeline waits block the render thread (holding
//! no lock of the backend's while it waits: `tests::under_wait_a_draw_waits_for_a_delayed_build`
//! asks `pipeline_status` meanwhile); under
//! `Skip` a compiling pipeline's draw is skipped and counted, each draw once
//! ([`PipelineCounts::skipped_while_compiling`]). A failed pipeline's draws are skipped under
//! either setting, and counted once per pipeline, by the first draw that meets it
//! ([`PipelineCounts::failed_pipelines`]), not per draw. Draws themselves arrive with slice R7,
//! whose encoder calls it; until then the list check refuses them.
//!
//! # Not here
//!
//! - **The driver cache blob.** wgpu's `PIPELINE_CACHE` is Vulkan only and unimplemented on
//!   Metal (decision 1's table), so the Mac has none; and wgpu 30's `Device::create_pipeline_cache`
//!   is an `unsafe fn`, which this crate's denial of `unsafe` excludes (an owner question). No
//!   pipeline is given a cache.
//! - **Destroying a shader or pipeline** frees its id (`destroy`, deferred by fence value like
//!   every resource); a build already queued keeps what it needs and finishes, and its result is
//!   dropped with the last handle to it.
//! - **A cap on pipelines.** The GPU crate's pipeline-key cache caps what it creates; the backend
//!   adds none (the cap is an open question in `docs/proposals.md`).

pub mod layout;
mod scan;
pub mod state;

use crate::backend::block_on;
use layout::{LayoutPlan, Stage};
use scan::ModuleFacts;
use starquake_render::shader::ShaderInterface;
use starquake_render::{
    CompileJob, CompilePool, PipelineStatus, RenderError, ShaderDesc, ShaderSource, ShaderStage,
    Slots, WhileCompiling,
};
use state::GraphicsPlan;
use std::borrow::Cow;
use std::collections::BTreeSet;
use std::fmt;
use std::panic::AssertUnwindSafe;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock, PoisonError};
use wgpu::naga;

/// A shader that is not a naga module.
pub const NAGA_ONLY: &str = "the wgpu backend takes shaders as naga modules only";
/// A module without the shader's entry point for its stage.
pub const NO_ENTRY_POINT: &str = "the shader's module has no entry point of its name for its stage";
/// A shader that needs subgroups of 32 on a device that cannot promise them.
pub const SUBGROUP_SIZE: &str =
    "the device cannot promise the subgroup size of 32 a shader of this pipeline needs";
/// A pipeline asked of a backend built without a compile pool.
pub const NO_COMPILE_POOL: &str =
    "the wgpu backend was built without a compile pool and cannot build pipelines";
/// A pool dropped a build without running it.
pub const DROPPED: &str = "the compile pool dropped the pipeline's build without running it";
/// A build panicked.
pub const PANICKED: &str = "the pipeline's build panicked";

/// The guest's warp width, which a shader using subgroups relies on (shader-pipeline, "What
/// naga 30.0.1 can express": subgroup operations need "a subgroup size of 32").
pub const GUEST_SUBGROUP_SIZE: u32 = 32;

/// Whether a device with `features` and a subgroup size range of `min` to `max` runs every
/// subgroup at [`GUEST_SUBGROUP_SIZE`] (module docs, "Requirements").
pub const fn subgroups_honoured(features: wgpu::Features, min: u32, max: u32) -> bool {
    features.contains(wgpu::Features::SUBGROUP)
        && min == GUEST_SUBGROUP_SIZE
        && max == GUEST_SUBGROUP_SIZE
}

/// Why a draw was not drawn.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SkipReason {
    /// The pipeline is compiling and the setting is [`WhileCompiling::Skip`].
    Compiling,
    /// The pipeline failed to build.
    Failed,
}

/// What a draw does with its pipeline's status under a setting (module docs, "Draws").
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DrawAction {
    /// Draw with it.
    Draw,
    /// Wait for its build to settle, then decide again.
    Wait,
    /// Skip the draw.
    Skip(SkipReason),
}

/// The decision for a draw whose pipeline has `status`, under `setting`.
pub const fn on_draw(status: PipelineStatus, setting: WhileCompiling) -> DrawAction {
    match (status, setting) {
        (PipelineStatus::Ready, _) => DrawAction::Draw,
        (PipelineStatus::Failed, _) => DrawAction::Skip(SkipReason::Failed),
        (PipelineStatus::Compiling, WhileCompiling::Wait) => DrawAction::Wait,
        (PipelineStatus::Compiling, WhileCompiling::Skip) => {
            DrawAction::Skip(SkipReason::Compiling)
        }
    }
}

/// A built wgpu pipeline.
#[derive(Clone, Debug)]
pub(crate) enum HostPipeline {
    /// A render pipeline.
    Render(wgpu::RenderPipeline),
    /// A compute pipeline.
    Compute(wgpu::ComputePipeline),
}

/// A pipeline a draw can use, from [`crate::WgpuBackend::draw_pipeline`], for the render
/// thread's encoder (slice R7) to bind.
#[derive(Clone, Debug)]
pub struct ReadyPipeline(pub(crate) HostPipeline);

impl ReadyPipeline {
    /// The render pipeline, for a graphics pipeline.
    pub fn render(&self) -> Option<&wgpu::RenderPipeline> {
        match &self.0 {
            HostPipeline::Render(p) => Some(p),
            HostPipeline::Compute(_) => None,
        }
    }

    /// The compute pipeline, for a compute pipeline.
    pub fn compute(&self) -> Option<&wgpu::ComputePipeline> {
        match &self.0 {
            HostPipeline::Compute(p) => Some(p),
            HostPipeline::Render(_) => None,
        }
    }
}

/// What a draw gets for its pipeline.
#[derive(Clone, Debug)]
pub enum DrawPipeline {
    /// The pipeline, ready.
    Ready(ReadyPipeline),
    /// The draw is skipped (and counted, [`PipelineCounts`]).
    Skipped(SkipReason),
}

/// The draws the backend skipped for their pipelines.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct PipelineCounts {
    /// Draws skipped because their pipeline was compiling under [`WhileCompiling::Skip`], each
    /// draw once.
    pub skipped_while_compiling: u64,
    /// Failed pipelines a draw met, each pipeline once however many draws meet it.
    pub failed_pipelines: u64,
}

/// The counters behind [`PipelineCounts`].
#[derive(Debug, Default)]
pub(crate) struct Counters {
    skipped: AtomicU64,
    failed: AtomicU64,
}

impl Counters {
    pub(crate) fn get(&self) -> PipelineCounts {
        PipelineCounts {
            skipped_while_compiling: self.skipped.load(Ordering::Relaxed),
            failed_pipelines: self.failed.load(Ordering::Relaxed),
        }
    }
}

/// A shader's module once wgpu has it, and the samplers its entry point must not filter
/// through ([`layout`], "Filtering samplers").
#[derive(Debug)]
struct HostModule {
    module: wgpu::ShaderModule,
    non_filtering: BTreeSet<(u32, u32)>,
}

/// A live shader (module docs, "Shaders").
#[derive(Debug)]
pub(crate) struct ShaderRec {
    /// The stage it runs as.
    pub(crate) stage: ShaderStage,
    entry: String,
    /// The index of its entry point in the module.
    entry_index: usize,
    interface: ShaderInterface,
    facts: ModuleFacts,
    /// The naga module, until the first build hands it to wgpu.
    source: Mutex<Option<Box<naga::Module>>>,
    /// The wgpu module, or why it could not be made; made once.
    host: OnceLock<Result<HostModule, String>>,
}

impl ShaderRec {
    /// Checks a shader description's code and keeps it (module docs, "Shaders"). The shared
    /// checks against `Caps` have run.
    pub(crate) fn new(desc: ShaderDesc) -> Result<Self, RenderError> {
        let ShaderSource::Naga { module, .. } = desc.source else {
            return Err(RenderError::Unsupported(NAGA_ONLY));
        };
        let entry_index = scan::naga_stage(desc.stage)
            .and_then(|stage| scan::entry_point(&module, &desc.entry, stage))
            .ok_or(RenderError::Invalid(NO_ENTRY_POINT))?;
        Ok(Self {
            stage: desc.stage,
            entry: desc.entry,
            entry_index,
            facts: scan::scan(&module),
            interface: desc.interface,
            source: Mutex::new(Some(module)),
            host: OnceLock::new(),
        })
    }

    /// Whether the module uses subgroups.
    pub(crate) fn uses_subgroups(&self) -> bool {
        self.facts.subgroups
    }

    /// This shader's part of a pipeline layout.
    fn stage(&self) -> Stage<'_> {
        Stage {
            visibility: match self.stage {
                ShaderStage::Vertex => wgpu::ShaderStages::VERTEX,
                ShaderStage::Fragment => wgpu::ShaderStages::FRAGMENT,
                ShaderStage::Compute => wgpu::ShaderStages::COMPUTE,
                // Never made: the shared checks refuse these stages on a wgpu device.
                ShaderStage::TessControl | ShaderStage::TessEval | ShaderStage::Geometry => {
                    wgpu::ShaderStages::NONE
                }
            },
            interface: &self.interface,
            facts: &self.facts,
        }
    }

    /// The wgpu module, made on the first call (module docs, "Shaders").
    fn host(&self, device: &wgpu::Device) -> Result<&HostModule, String> {
        self.host
            .get_or_init(|| {
                let module = self
                    .source
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .take()
                    .ok_or_else(|| "the shader's module was already handed over".to_owned())?;
                let non_filtering = if layout::may_need_non_filtering(&self.interface) {
                    layout::non_filtering_samplers(&module, self.entry_index, &self.interface)?
                } else {
                    BTreeSet::new()
                };
                let module = scoped(device, || {
                    device.create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: None,
                        source: wgpu::ShaderSource::Naga(Cow::Owned(*module)),
                    })
                })?;
                Ok(HostModule {
                    module,
                    non_filtering,
                })
            })
            .as_ref()
            .map_err(Clone::clone)
    }
}

/// Runs `f` inside wgpu error scopes for validation, out-of-memory and internal errors, and
/// returns the first error any of them caught, innermost first.
fn scoped<T>(device: &wgpu::Device, f: impl FnOnce() -> T) -> Result<T, String> {
    let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let memory = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    let internal = device.push_error_scope(wgpu::ErrorFilter::Internal);
    let value = f();
    // Popped innermost first, as wgpu requires; each pop takes effect at once.
    let pops = [internal.pop(), memory.pop(), validation.pop()];
    for pop in pops {
        match block_on(pop) {
            Ok(None) => {}
            Ok(Some(error)) => return Err(error.to_string()),
            Err(error) => return Err(error.to_string()),
        }
    }
    Ok(value)
}

/// Where a build stands.
#[derive(Clone, Debug)]
enum Built {
    Compiling,
    Ready(HostPipeline),
    Failed,
}

/// One pipeline's status, shared by the backend's table, its build and the draws that wait.
#[derive(Debug)]
pub(crate) struct PipelineCell {
    built: Mutex<Built>,
    /// Signalled when the build settles.
    settled: Condvar,
    /// Whether a draw has counted this pipeline's failure.
    counted: AtomicBool,
}

impl PipelineCell {
    /// A pipeline being built.
    pub(crate) fn new() -> Self {
        Self {
            built: Mutex::new(Built::Compiling),
            settled: Condvar::new(),
            counted: AtomicBool::new(false),
        }
    }

    fn built(&self) -> MutexGuard<'_, Built> {
        self.built.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Compiling, ready or failed.
    pub(crate) fn status(&self) -> PipelineStatus {
        match *self.built() {
            Built::Compiling => PipelineStatus::Compiling,
            Built::Ready(_) => PipelineStatus::Ready,
            Built::Failed => PipelineStatus::Failed,
        }
    }

    /// Stores the build's outcome and releases every waiting draw. The first outcome stands.
    fn settle(&self, outcome: Result<HostPipeline, String>) {
        let mut built = self.built();
        if !matches!(*built, Built::Compiling) {
            return;
        }
        *built = match outcome {
            Ok(pipeline) => Built::Ready(pipeline),
            Err(reason) => {
                tracing::warn!("a pipeline failed to build: {reason}");
                Built::Failed
            }
        };
        drop(built);
        self.settled.notify_all();
    }

    /// Blocks until the build settles.
    fn wait(&self) -> Built {
        let built = self
            .settled
            .wait_while(self.built(), |b| matches!(b, Built::Compiling))
            .unwrap_or_else(PoisonError::into_inner);
        built.clone()
    }
}

/// Settles its cell as failed when dropped unsettled: a dropped or panicking job (module docs).
struct Settle {
    cell: Arc<PipelineCell>,
    done: bool,
}

impl Settle {
    fn finish(mut self, outcome: Result<HostPipeline, String>) {
        self.cell.settle(outcome);
        self.done = true;
    }
}

impl Drop for Settle {
    fn drop(&mut self) {
        if !self.done {
            self.cell.settle(Err(DROPPED.to_owned()));
        }
    }
}

/// The pool job that runs `build` and settles `cell` with its outcome, or as failed when it
/// panics or the job is dropped unrun.
pub(crate) fn job(
    cell: Arc<PipelineCell>,
    build: impl FnOnce() -> Result<HostPipeline, String> + Send + 'static,
) -> CompileJob {
    let settle = Settle { cell, done: false };
    Box::new(move || {
        let outcome = std::panic::catch_unwind(AssertUnwindSafe(build))
            .unwrap_or_else(|_| Err(PANICKED.to_owned()));
        settle.finish(outcome);
    })
}

/// What a draw gets for the pipeline in `cell` under `setting` (module docs, "Draws"), counting
/// skips in `counters`.
pub(crate) fn acquire(
    cell: &PipelineCell,
    setting: WhileCompiling,
    counters: &Counters,
) -> DrawPipeline {
    let mut built = cell.built().clone();
    loop {
        let status = match built {
            Built::Compiling => PipelineStatus::Compiling,
            Built::Ready(_) => PipelineStatus::Ready,
            Built::Failed => PipelineStatus::Failed,
        };
        match on_draw(status, setting) {
            DrawAction::Draw => {
                // `on_draw` answers `Draw` only for a ready pipeline, so the `else` is
                // unreachable.
                let Built::Ready(pipeline) = built else {
                    return DrawPipeline::Skipped(SkipReason::Failed);
                };
                return DrawPipeline::Ready(ReadyPipeline(pipeline));
            }
            DrawAction::Wait => built = cell.wait(),
            DrawAction::Skip(SkipReason::Compiling) => {
                counters.skipped.fetch_add(1, Ordering::Relaxed);
                return DrawPipeline::Skipped(SkipReason::Compiling);
            }
            DrawAction::Skip(SkipReason::Failed) => {
                if !cell.counted.swap(true, Ordering::Relaxed) {
                    counters.failed.fetch_add(1, Ordering::Relaxed);
                }
                return DrawPipeline::Skipped(SkipReason::Failed);
            }
        }
    }
}

/// The live shaders and pipelines, under the backend's state lock.
#[derive(Debug, Default)]
pub(crate) struct Registry {
    pub(crate) shaders: Slots<Arc<ShaderRec>>,
    pub(crate) pipelines: Slots<Arc<PipelineCell>>,
}

/// The compile pool and the setting the backend was built with.
#[derive(Clone)]
pub(crate) struct Compile {
    pub(crate) pool: Arc<dyn CompilePool>,
    pub(crate) setting: WhileCompiling,
}

impl fmt::Debug for Compile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Compile")
            .field("setting", &self.setting)
            .finish_non_exhaustive()
    }
}

/// Refuses `shaders` that need subgroups of 32 when the device cannot promise them.
pub(crate) fn check_requirements<'a>(
    shaders: impl IntoIterator<Item = &'a ShaderRec>,
    subgroups: bool,
) -> Result<(), RenderError> {
    if subgroups || !shaders.into_iter().any(ShaderRec::uses_subgroups) {
        Ok(())
    } else {
        Err(RenderError::Unsupported(SUBGROUP_SIZE))
    }
}

/// Merges the shaders' interfaces into the pipeline's layouts.
pub(crate) fn plan_layout<'a>(
    shaders: impl IntoIterator<Item = &'a ShaderRec>,
) -> Result<LayoutPlan, RenderError> {
    let stages: Vec<Stage<'_>> = shaders.into_iter().map(ShaderRec::stage).collect();
    layout::plan(&stages)
}

/// Makes the bind group layouts and the pipeline layout for `plan`, turning the samplers in
/// `non_filtering` non-filtering. The caller runs it inside [`scoped`].
fn pipeline_layout(
    device: &wgpu::Device,
    plan: LayoutPlan,
    non_filtering: &BTreeSet<(u32, u32)>,
) -> wgpu::PipelineLayout {
    let plan = layout::with_non_filtering(plan, non_filtering);
    let groups: Vec<Option<wgpu::BindGroupLayout>> = plan
        .groups
        .iter()
        .map(|entries| {
            (!entries.is_empty()).then(|| {
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: None,
                    entries,
                })
            })
        })
        .collect();
    let refs: Vec<Option<&wgpu::BindGroupLayout>> = groups.iter().map(Option::as_ref).collect();
    device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &refs,
        immediate_size: 0,
    })
}

/// Everything a graphics build needs, owned, for the pool worker.
#[derive(Debug)]
pub(crate) struct GraphicsBuild {
    pub(crate) device: wgpu::Device,
    pub(crate) vertex: Arc<ShaderRec>,
    pub(crate) fragment: Option<Arc<ShaderRec>>,
    pub(crate) layout: LayoutPlan,
    pub(crate) plan: GraphicsPlan,
}

impl GraphicsBuild {
    /// Builds the pipeline (module docs, "Pipelines").
    pub(crate) fn run(self) -> Result<HostPipeline, String> {
        let device = &self.device;
        let vertex = self.vertex.host(device)?;
        let fragment = match &self.fragment {
            Some(f) => Some((f, f.host(device)?)),
            None => None,
        };
        let mut non_filtering = vertex.non_filtering.clone();
        if let Some((_, f)) = &fragment {
            non_filtering.extend(f.non_filtering.iter().copied());
        }
        let plan = &self.plan;
        scoped(device, || {
            let layout = pipeline_layout(device, self.layout, &non_filtering);
            let buffers: Vec<Option<wgpu::VertexBufferLayout<'_>>> = plan
                .buffers
                .iter()
                .map(|b| {
                    Some(wgpu::VertexBufferLayout {
                        array_stride: b.stride,
                        step_mode: b.step,
                        attributes: &b.attributes,
                    })
                })
                .collect();
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: None,
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &vertex.module,
                    entry_point: Some(&self.vertex.entry),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &buffers,
                },
                primitive: plan.primitive,
                depth_stencil: plan.depth_stencil.clone(),
                multisample: plan.multisample,
                fragment: fragment.map(|(shader, host)| wgpu::FragmentState {
                    module: &host.module,
                    entry_point: Some(&shader.entry),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    targets: &plan.targets,
                }),
                multiview_mask: None,
                cache: None,
            })
        })
        .map(HostPipeline::Render)
    }
}

/// Everything a compute build needs, owned, for the pool worker.
#[derive(Debug)]
pub(crate) struct ComputeBuild {
    pub(crate) device: wgpu::Device,
    pub(crate) shader: Arc<ShaderRec>,
    pub(crate) layout: LayoutPlan,
}

impl ComputeBuild {
    /// Builds the pipeline (module docs, "Pipelines").
    pub(crate) fn run(self) -> Result<HostPipeline, String> {
        let device = &self.device;
        let host = self.shader.host(device)?;
        scoped(device, || {
            let layout = pipeline_layout(device, self.layout, &host.non_filtering);
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: Some(&layout),
                module: &host.module,
                entry_point: Some(&self.shader.entry),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            })
        })
        .map(HostPipeline::Compute)
    }
}

#[cfg(test)]
mod tests;
