//! The compile pool seam and the wait-while-compiling setting (docs/specs/render-backend.md,
//! "Where the backend sits" and "Compiling without stalling a frame").
//!
//! One compile pool, created by `core`, builds pipelines for the backend and translates guest
//! shaders for the shader cache; each crate takes it through its own small executor trait, which
//! `core` implements (review guide, M16). [`CompilePool`] is the render crate's. A backend hands
//! it one [`CompileJob`] per pipeline from `create_*_pipeline` and returns the pipeline's id at
//! once; the job runs on a pool worker, where a backend whose host API builds pipelines
//! synchronously (wgpu) calls that API, and marks the pipeline ready or failed when it returns.
//!
//! [`WhileCompiling`] is the setting `core` passes each backend at construction beside the pool:
//! what the render thread does with a draw whose pipeline is still compiling. The GPU crate owns
//! the one setting, which covers shader translation and pipeline builds alike (decision 5,
//! decided by the owner on 2026-11-12: "wait by default, as one setting owned by `gpu`, with
//! skipping as an opt-in setting after the persistent caches exist"). The render crate depends on
//! no workspace crate, so it cannot name the GPU crate's type; `core` converts the GPU crate's
//! setting into this one, variant for variant, and is the one place that conversion happens.
//!
//! Neither is a trait method: like the completion wake ([`crate::wake`]), they are what a backend
//! is built with.

/// One unit of work for the compile pool: a pipeline build. It runs at most once.
pub type CompileJob = Box<dyn FnOnce() + Send + 'static>;

/// The render crate's view of the shared compile pool (module docs). `core` implements it.
///
/// The contract a backend relies on:
///
/// - [`CompilePool::spawn`] returns at once, without running the job on the calling thread
///   first (the caller may be the GPU back thread, which must not wait for a build);
/// - the pool runs each job at most once, on any of its workers, in any order;
/// - a pool that drops a job without running it (because it is shutting down) drops the job's
///   captured values, and a backend settles the pipeline as failed from their `Drop`, so a
///   draw waiting for it is released; a pool never keeps a job it will not run.
pub trait CompilePool: Send + Sync {
    /// Queues `job` to run once on a worker and returns at once.
    fn spawn(&self, job: CompileJob);
}

/// What a draw does while its pipeline is compiling: the GPU crate's one wait-while-compiling
/// setting, as `core` passes it to a backend (module docs).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum WhileCompiling {
    /// Wait for the build: correct frames and deterministic frame dumps. The default
    /// (decision 5). Pipeline waits block the render thread.
    #[default]
    Wait,
    /// Skip the draw and count it: smooth, with visible gaps (Eden's asynchronous shaders,
    /// `use_asynchronous_shaders` in `eden/src/video_core/renderer_vulkan/vk_pipeline_cache.cpp`
    /// @ cf654052). Opt-in, once the persistent caches exist.
    Skip,
}

#[cfg(test)]
mod tests {
    use super::WhileCompiling;

    #[test]
    fn waiting_is_the_default() {
        // Decision 5: "wait by default".
        assert_eq!(WhileCompiling::default(), WhileCompiling::Wait);
    }
}
