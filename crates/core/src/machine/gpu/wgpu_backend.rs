//! Opening the wgpu backend for a run (docs/specs/render-backend.md, slice R12; the parent
//! module's "The render backend").
//!
//! [`WgpuOpener::open`] builds the backend the way render-backend's "The trait" asks: "`core`
//! constructs each backend with the shared compile pool, the GPU crate's wait-while-compiling
//! setting and the completion wake", in that call (`WgpuBackend::with_compile_pool`):
//!
//! - **the compile pool**: a fresh [`CompileWorkers`] sized for the host
//!   (`crate::compile_pool`), which the backend holds and which shuts down when the backend is
//!   dropped at the end of the run;
//! - **the wait setting**: the GPU crate's, from the run's `GpuConfig` ([`while_compiling`], the
//!   one place the GPU crate's `WaitWhileCompiling` becomes the render crate's
//!   `WhileCompiling`: the render crate depends on no workspace crate, so it cannot name the
//!   GPU crate's type);
//! - **the completion wake**: the GPU ring's, from its consumer side (`GpuRing::completion_wake`;
//!   decision 14).
//!
//! The backend's settings are `WgpuConfig::DEFAULT` (its memory limit, staging size and queue
//! depth are guesses, listed there).
//!
//! Then it installs the run's frame sink ([`RunSink`]) and, for a run with a window, makes the
//! window its present target on the frontend's main thread ([`present_on_main`]).
//!
//! # Present feedback
//!
//! The wgpu backend keeps a `PresentFeedback` for every present until someone takes it
//! (`Backend::take_present_feedback`), and nothing in a run reads it: the GPU crate does not, and
//! the run's report has no field for it. So that what the backend holds does not grow with the
//! number of presents, the run's frame sink takes and drops it each time the backend hands it a
//! frame, which it does after storing that batch's feedback, with no lock held (render-wgpu's
//! `thread` module, `settle`). Every present of a run is dumped (`FrameDumps::Every`), so every
//! batch of feedback is followed by a frame. The sink holds the backend weakly, so the backend
//! and its sink do not keep each other alive; the render thread may then drop the backend's last
//! handle, which the backend allows for (it does not join its own thread).
//!
//! Without the `wgpu` feature, [`WgpuOpener::open`] always fails with `WGPU_NOT_BUILT`:
//! reachable only through `RenderSetup::headless` with the wgpu choice, since
//! `crate::renderer::choose_backend` refuses it on such a build.

use std::sync::Arc;
use std::sync::mpsc;
use std::time::Duration;

use starquake_gpu::back::draw::WaitWhileCompiling;
use starquake_render::{Backend, CompletionWake, RenderError, WhileCompiling};

use super::RunFrames;
use crate::renderer::{MainThread, WindowTarget};

/// Why a run's wgpu backend could not be opened on a build without it.
#[cfg(not(feature = "wgpu"))]
pub(crate) const WGPU_NOT_BUILT: &str = "this build of starquake leaves the wgpu backend out";

/// The words a run ends with when its window cannot be presented into, before the reason.
pub(crate) const WINDOW_FAILED: &str = "the window could not be shown";

/// The GPU crate's wait-while-compiling setting as the render crate takes it (module docs): the
/// one place it is converted, variant for variant.
#[must_use]
#[cfg_attr(
    not(any(test, feature = "wgpu")),
    expect(dead_code, reason = "only the wgpu backend takes the setting")
)]
pub(crate) const fn while_compiling(setting: WaitWhileCompiling) -> WhileCompiling {
    match setting {
        WaitWhileCompiling::Wait => WhileCompiling::Wait,
        WaitWhileCompiling::Skip => WhileCompiling::Skip,
    }
}

/// Runs `set` on the frontend's main thread `main` and waits at most `wait` for its answer
/// (module docs; `crate::renderer`, "The window"). The error is the words the run ends with:
/// [`WINDOW_FAILED`], then why.
#[cfg_attr(
    not(any(test, feature = "wgpu")),
    expect(dead_code, reason = "only the wgpu backend presents into a window")
)]
pub(crate) fn present_on_main(
    main: &dyn MainThread,
    wait: Duration,
    set: impl FnOnce() -> Result<(), RenderError> + Send + 'static,
) -> Result<(), String> {
    let (answer, answered) = mpsc::sync_channel(1);
    main.post(Box::new(move || {
        // The run may have stopped waiting; then nobody reads the answer.
        let _ = answer.send(set());
    }))
    .map_err(|gone| format!("{WINDOW_FAILED}: {gone}"))?;
    match answered.recv_timeout(wait) {
        Ok(Ok(())) => Ok(()),
        Ok(Err(error)) => Err(format!("{WINDOW_FAILED}: {error}")),
        Err(mpsc::RecvTimeoutError::Timeout) => Err(format!(
            "{WINDOW_FAILED}: the frontend's main thread did not answer within {} ms",
            wait.as_millis()
        )),
        Err(mpsc::RecvTimeoutError::Disconnected) => Err(format!(
            "{WINDOW_FAILED}: the frontend's main thread dropped the job without running it"
        )),
    }
}

#[cfg(feature = "wgpu")]
pub(crate) use built::Open;
#[cfg(all(test, feature = "wgpu"))]
use built::RunSink;

#[cfg(feature = "wgpu")]
mod built {
    use std::sync::{Arc, Weak};

    use starquake_render::{
        Backend, CompilePool, CompletionWake, Frame, FrameSink, WhileCompiling,
    };
    use starquake_render_wgpu::{NewError, WgpuBackend, WgpuConfig};

    use super::super::RunFrames;

    /// How a run opens the wgpu backend: `WgpuBackend::with_compile_pool` in a run, a stand-in
    /// that fails in the tests of what a run does when it cannot.
    pub(crate) type Open = dyn Fn(
            WgpuConfig,
            CompletionWake,
            Arc<dyn CompilePool>,
            WhileCompiling,
        ) -> Result<WgpuBackend, NewError>
        + Send
        + Sync;

    /// The run's frame sink on the wgpu backend: the run's frames, which count each frame
    /// handed over (the parent module's "At the end"), and the drain of the backend's present
    /// feedback (module docs, "Present feedback").
    #[derive(Debug)]
    pub(crate) struct RunSink {
        pub(crate) keeper: Arc<RunFrames>,
        pub(crate) backend: Weak<WgpuBackend>,
    }

    impl FrameSink for RunSink {
        fn frame(&self, frame: Frame) {
            self.keeper.frame(frame);
            if let Some(backend) = self.backend.upgrade() {
                drop(backend.take_present_feedback());
            }
        }
    }

    /// Opens the backend with `open` (module docs).
    pub(super) fn open(
        open: &Open,
        wake: CompletionWake,
        setting: WhileCompiling,
        keeper: &Arc<RunFrames>,
    ) -> Result<Arc<WgpuBackend>, String> {
        let pool = crate::compile_pool::CompileWorkers::for_host()
            .map_err(|error| format!("the compile pool could not be started: {error}"))?;
        let backend = open(WgpuConfig::DEFAULT, wake, Arc::new(pool), setting)
            .map(Arc::new)
            .map_err(|error| error.to_string())?;
        backend.set_frame_sink(Some(Arc::new(RunSink {
            keeper: Arc::clone(keeper),
            backend: Arc::downgrade(&backend),
        })));
        Ok(backend)
    }
}

/// How a run opens the wgpu backend (module docs).
pub(crate) struct WgpuOpener {
    #[cfg(feature = "wgpu")]
    open: Box<Open>,
    /// Why it cannot open the backend: `WGPU_NOT_BUILT`.
    #[cfg(not(feature = "wgpu"))]
    missing: &'static str,
}

impl WgpuOpener {
    /// The host's: `WgpuBackend::with_compile_pool`, or, on a build without the backend, an
    /// opener that always fails (`WGPU_NOT_BUILT`).
    pub(crate) fn host() -> Self {
        Self {
            #[cfg(feature = "wgpu")]
            open: Box::new(starquake_render_wgpu::WgpuBackend::with_compile_pool),
            #[cfg(not(feature = "wgpu"))]
            missing: WGPU_NOT_BUILT,
        }
    }

    /// An opener that opens with `open`, for the tests.
    #[cfg(all(test, feature = "wgpu"))]
    pub(crate) fn with(open: Box<Open>) -> Self {
        Self { open }
    }

    /// Opens the wgpu backend for a run (module docs): built with `wake` and the GPU crate's
    /// `setting`, dumping into `keeper`, and presenting into `window` if there is one, whose
    /// surface is made on the frontend's main thread within `present_wait`.
    #[cfg(feature = "wgpu")]
    pub(crate) fn open(
        &self,
        wake: CompletionWake,
        setting: WaitWhileCompiling,
        keeper: &Arc<RunFrames>,
        window: Option<&WindowTarget>,
        present_wait: Duration,
    ) -> Result<Arc<dyn Backend>, Unopened> {
        let backend = built::open(&*self.open, wake, while_compiling(setting), keeper)
            .map_err(Unopened::Backend)?;
        if let Some(window) = window {
            let target = Arc::clone(&backend);
            let handles = window.handles().clone();
            present_on_main(window.main().as_ref(), present_wait, move || {
                target.set_present_target(starquake_render::PresentTarget::Window(handles))
            })
            .map_err(Unopened::Window)?;
        }
        Ok(backend)
    }

    /// On a build without the wgpu backend: always `WGPU_NOT_BUILT` (module docs).
    #[cfg(not(feature = "wgpu"))]
    pub(crate) fn open(
        &self,
        _wake: CompletionWake,
        _setting: WaitWhileCompiling,
        _keeper: &Arc<RunFrames>,
        _window: Option<&WindowTarget>,
        _present_wait: Duration,
    ) -> Result<Arc<dyn Backend>, Unopened> {
        Err(Unopened::Backend(self.missing.to_owned()))
    }
}

/// Why [`WgpuOpener::open`] gave no backend.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Unopened {
    /// The backend could not be opened: why, in words. What the run does then is its choice's
    /// (`crate::renderer::IfUnavailable`).
    Backend(String),
    /// The backend opened, but the window could not be made its present target: the words the
    /// run ends with ([`WINDOW_FAILED`], then why). A run with a window always ends then.
    #[cfg_attr(
        not(feature = "wgpu"),
        expect(dead_code, reason = "only the wgpu backend presents into a window")
    )]
    Window(String),
}

#[cfg(test)]
mod tests;
