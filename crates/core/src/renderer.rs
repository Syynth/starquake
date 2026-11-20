//! Which render backend a run draws with, and the window it shows frames in
//! (docs/specs/render-backend.md, "Where the backend sits", decision 11 and slice R12).
//!
//! # The choice
//!
//! `core` picks a backend "by `cfg` and Cargo features" (render-backend, "Where the backend
//! sits"). The wgpu backend is built into `core` by its `wgpu` feature ([`WGPU_BUILT`]), which
//! the command-line app turns on by default. [`choose_backend`] is the choice, a pure function
//! of whether the wgpu backend is built and of the command line's two flags, `--backend` and
//! `--window`, tested for every combination:
//!
//! | wgpu built | `--backend` | `--window` | Choice |
//! |---|---|---|---|
//! | yes | none | no | [`Choice::Wgpu`], and the recording backend if wgpu cannot be opened |
//! | yes | none | yes | [`Choice::Wgpu`], and the run ends if it cannot be opened |
//! | yes | `wgpu` | either | [`Choice::Wgpu`], and the run ends if it cannot be opened |
//! | either | `recording` | no | [`Choice::Recording`] |
//! | either | `recording` | yes | refused: [`ChoiceError::RecordingWindow`] |
//! | no | none | no | [`Choice::Recording`] |
//! | no | `wgpu` | either | refused: [`ChoiceError::WgpuNotBuilt`] |
//! | no | none | yes | refused: [`ChoiceError::WindowNotBuilt`] |
//!
//! The refusals are checked in that order: the command line's own conflict first
//! (`--backend recording` with `--window`), then `--backend wgpu` on a build without it, then
//! `--window` on a build without it; each pair of faults present at once is tested.
//!
//! The wgpu backend is the default where it is built (slice R12). Where it was not asked for by
//! name (no `--backend` and no `--window`) and cannot be opened (a host with no GPU adapter, as
//! the hosted CI runners are), the run renders with the recording backend instead and says so
//! at warning level in its `log.txt` ([`IfUnavailable::UseRecording`]); a run that asked for it
//! ends as an emulator error instead ([`IfUnavailable::EndRun`]).
//!
//! The recording backend is the gate's: the library's [`crate::System::run`] renders with it
//! ([`RenderSetup::recording`]), and the app's own tests ask for it with `--backend recording`,
//! so the gate's runs do not depend on the host's GPU. It is also the backend for runs that need
//! no pixels drawn (render-backend, "The recording backend": "the `null` backend for runs that
//! need no pixels"): it still dumps every present of an image the guest wrote.
//!
//! **`--backend` on `run` is starquake's own flag** (an owner question). The observability
//! spec's headless contract gives `run` no backend flag; it names `--backend recording|wgpu`
//! for `render-replay` only (`docs/specs/observability.md`, "Headless runs"). `run` takes the
//! same spelling so that the gate, and a run that needs no GPU, can have the recording backend
//! where the wgpu backend is the default.
//!
//! # The window
//!
//! `--window` shows the frames in a plain window while the run goes on (decision 11); it
//! changes nothing the guest sees: the guest is paced by the emulated clock, and the frame
//! dumps are the same with and without it. The frontend's main thread owns the window and its
//! event loop and hands the window to the run as a [`WindowTarget`]: the window's handles and
//! size ([`WindowHandles`], slice R8) and a way to run a job on the main thread
//! ([`MainThread`]). On macOS the wgpu backend must make the window's surface on the main
//! thread (render-backend, "Present targets": "The surface is made in `set_present_target`,
//! which runs on the main thread on macOS"), so the run, on its own thread, posts that call to
//! the main thread and waits for its answer, at most [`PRESENT_TARGET_WAIT`]. The main thread
//! must therefore keep taking jobs while the run starts.

use std::sync::Arc;
use std::time::Duration;

pub use starquake_render::window::{WindowHandles, WindowSize, WindowSource, raw_window_handle};

/// Whether this build of `core` has the wgpu backend (its `wgpu` feature).
pub const WGPU_BUILT: bool = cfg!(feature = "wgpu");

/// The longest a run waits for the frontend's main thread to make the window's surface (module
/// docs, "The window"): starquake's own number, a guess. A run that waits longer ends as an
/// emulator error.
pub const PRESENT_TARGET_WAIT: Duration = Duration::from_secs(10);

/// The backend `--backend` names.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BackendFlag {
    /// `--backend recording`.
    Recording,
    /// `--backend wgpu`.
    Wgpu,
}

impl BackendFlag {
    /// The flag's value as the command line spells it.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Recording => "recording",
            Self::Wgpu => "wgpu",
        }
    }

    /// The flag whose value is `value` (exactly as [`BackendFlag::name`] spells it).
    #[must_use]
    pub fn from_name(value: &str) -> Option<Self> {
        [Self::Recording, Self::Wgpu]
            .into_iter()
            .find(|flag| flag.name() == value)
    }
}

/// What a run does when the wgpu backend it chose cannot be opened (module docs, "The choice").
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum IfUnavailable {
    /// Render with the recording backend, saying so at warning level: the default, when the
    /// wgpu backend was not asked for by name.
    UseRecording,
    /// End the run as an emulator error: the wgpu backend was asked for by name, or a window
    /// needs it.
    EndRun,
}

/// The backend a run draws with ([`choose_backend`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Choice {
    /// The recording backend (render-backend, "The recording backend").
    Recording,
    /// The wgpu backend, and what to do if it cannot be opened.
    Wgpu(IfUnavailable),
}

/// Why a command line's backend flags cannot be followed (module docs, "The choice").
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ChoiceError {
    /// `--window` with `--backend recording`.
    #[error(
        "`--window` shows frames the recording backend does not draw: drop `--backend recording`"
    )]
    RecordingWindow,
    /// `--backend wgpu` on a build without the wgpu backend.
    #[error(
        "`--backend wgpu` is not available: this build of starquake leaves the wgpu backend out"
    )]
    WgpuNotBuilt,
    /// `--window` on a build without the wgpu backend.
    #[error(
        "`--window` is not available: it needs the wgpu backend, which this build of starquake \
         leaves out"
    )]
    WindowNotBuilt,
}

/// The backend a run draws with, from whether the wgpu backend is built (`wgpu_built`, which a
/// frontend passes as [`WGPU_BUILT`]), `--backend` (`flag`) and `--window` (`window`): the
/// table in the module docs, refusals in its order.
pub fn choose_backend(
    wgpu_built: bool,
    flag: Option<BackendFlag>,
    window: bool,
) -> Result<Choice, ChoiceError> {
    match (flag, window) {
        (Some(BackendFlag::Recording), true) => Err(ChoiceError::RecordingWindow),
        (Some(BackendFlag::Recording), false) => Ok(Choice::Recording),
        (Some(BackendFlag::Wgpu), _) if !wgpu_built => Err(ChoiceError::WgpuNotBuilt),
        (None, true) if !wgpu_built => Err(ChoiceError::WindowNotBuilt),
        (None, false) if !wgpu_built => Ok(Choice::Recording),
        (None, false) => Ok(Choice::Wgpu(IfUnavailable::UseRecording)),
        (Some(BackendFlag::Wgpu), _) | (None, true) => Ok(Choice::Wgpu(IfUnavailable::EndRun)),
    }
}

/// A job for the frontend's main thread.
pub type MainJob = Box<dyn FnOnce() + Send + 'static>;

/// The frontend's main thread no longer takes jobs (its event loop has ended); the job was
/// dropped unrun.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error("the frontend's main thread no longer takes jobs")]
pub struct MainThreadGone;

/// The frontend's main thread, as a run asks it to do what only that thread may (module docs,
/// "The window"). The frontend implements it over its event loop.
pub trait MainThread: Send + Sync {
    /// Queues `job` to run once on the main thread and returns at once, without running it on
    /// the calling thread. [`MainThreadGone`] when the main thread no longer takes jobs, after
    /// dropping `job` unrun.
    fn post(&self, job: MainJob) -> Result<(), MainThreadGone>;
}

/// A window a run shows its frames in (module docs, "The window").
#[derive(Clone)]
pub struct WindowTarget {
    handles: WindowHandles,
    main: Arc<dyn MainThread>,
}

impl WindowTarget {
    /// The window `handles`, whose surface is made through `main`.
    pub fn new(handles: WindowHandles, main: Arc<dyn MainThread>) -> Self {
        Self { handles, main }
    }

    /// The window's handles and size.
    pub fn handles(&self) -> &WindowHandles {
        &self.handles
    }

    /// The frontend's main thread.
    pub fn main(&self) -> &Arc<dyn MainThread> {
        &self.main
    }
}

impl core::fmt::Debug for WindowTarget {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("WindowTarget")
            .field("handles", &self.handles)
            .finish_non_exhaustive()
    }
}

/// What a run renders with: the backend it chose, and the window it shows frames in, if any.
/// A window always goes with the wgpu backend, ending the run if it cannot be opened.
#[derive(Clone, Debug)]
pub struct RenderSetup {
    choice: Choice,
    window: Option<WindowTarget>,
}

impl Default for RenderSetup {
    /// [`RenderSetup::recording`].
    fn default() -> Self {
        Self::recording()
    }
}

impl RenderSetup {
    /// The recording backend and no window: the gate's (module docs, "The choice").
    #[must_use]
    pub const fn recording() -> Self {
        Self {
            choice: Choice::Recording,
            window: None,
        }
    }

    /// `choice`, with no window.
    #[must_use]
    pub const fn headless(choice: Choice) -> Self {
        Self {
            choice,
            window: None,
        }
    }

    /// The wgpu backend, showing its frames in `window`; the run ends if the backend cannot be
    /// opened ([`IfUnavailable::EndRun`]).
    #[must_use]
    pub fn windowed(window: WindowTarget) -> Self {
        Self {
            choice: Choice::Wgpu(IfUnavailable::EndRun),
            window: Some(window),
        }
    }

    /// The backend.
    #[must_use]
    pub const fn choice(&self) -> Choice {
        self.choice
    }

    /// The window, if any.
    #[must_use]
    pub const fn window(&self) -> Option<&WindowTarget> {
        self.window.as_ref()
    }
}

#[cfg(test)]
pub(crate) mod tests;
