//! The backend choice, for every combination of build, `--backend` and `--window`, and what a
//! run's render setup holds.

use super::raw_window_handle::{
    DisplayHandle, HandleError, HasDisplayHandle, HasWindowHandle, WindowHandle,
};
use super::*;

/// A window with no handles: enough for a [`WindowTarget`] whose surface is never made.
pub(crate) struct NoWindow;

impl HasWindowHandle for NoWindow {
    fn window_handle(&self) -> Result<WindowHandle<'_>, HandleError> {
        Err(HandleError::Unavailable)
    }
}

impl HasDisplayHandle for NoWindow {
    fn display_handle(&self) -> Result<DisplayHandle<'_>, HandleError> {
        Err(HandleError::Unavailable)
    }
}

/// A main thread that takes no jobs.
pub(crate) struct Gone;

impl MainThread for Gone {
    fn post(&self, _job: MainJob) -> Result<(), MainThreadGone> {
        Err(MainThreadGone)
    }
}

/// A window target over [`NoWindow`], `width` by `height`, posting to `main`.
pub(crate) fn no_window(width: u32, height: u32, main: Arc<dyn MainThread>) -> WindowTarget {
    WindowTarget::new(
        WindowHandles::new(Arc::new(NoWindow), WindowSize::new(width, height)),
        main,
    )
}

#[test]
fn every_combination_of_build_backend_flag_and_window_has_its_choice() {
    use BackendFlag::{Recording as R, Wgpu as W};
    use IfUnavailable::{EndRun, UseRecording};
    // The module docs' table, row by row, written out.
    let cases = [
        (true, None, false, Ok(Choice::Wgpu(UseRecording))),
        (true, None, true, Ok(Choice::Wgpu(EndRun))),
        (true, Some(W), false, Ok(Choice::Wgpu(EndRun))),
        (true, Some(W), true, Ok(Choice::Wgpu(EndRun))),
        (true, Some(R), false, Ok(Choice::Recording)),
        (true, Some(R), true, Err(ChoiceError::RecordingWindow)),
        (false, None, false, Ok(Choice::Recording)),
        (false, None, true, Err(ChoiceError::WindowNotBuilt)),
        (false, Some(W), false, Err(ChoiceError::WgpuNotBuilt)),
        // Two faults: `--backend wgpu` is reported before `--window`.
        (false, Some(W), true, Err(ChoiceError::WgpuNotBuilt)),
        (false, Some(R), false, Ok(Choice::Recording)),
        // Two faults: the conflict is reported before what the build lacks.
        (false, Some(R), true, Err(ChoiceError::RecordingWindow)),
    ];
    for (built, flag, window, expected) in cases {
        assert_eq!(
            choose_backend(built, flag, window),
            expected,
            "built {built}, flag {flag:?}, window {window}"
        );
    }
    assert_eq!(cases.len(), 2 * 3 * 2, "every combination once");
}

#[test]
fn the_flag_s_values_are_spelled_as_the_contract_spells_them() {
    // docs/specs/observability.md, "Headless runs": `[--backend recording|wgpu]`.
    assert_eq!(BackendFlag::Recording.name(), "recording");
    assert_eq!(BackendFlag::Wgpu.name(), "wgpu");
    assert_eq!(
        BackendFlag::from_name("recording"),
        Some(BackendFlag::Recording)
    );
    assert_eq!(BackendFlag::from_name("wgpu"), Some(BackendFlag::Wgpu));
    for other in ["", "Wgpu", "WGPU", "wgpu ", "null", "metal", "recordin"] {
        assert_eq!(BackendFlag::from_name(other), None, "{other:?}");
    }
}

#[test]
fn each_refusal_says_what_to_do_in_words() {
    assert_eq!(
        ChoiceError::RecordingWindow.to_string(),
        "`--window` shows frames the recording backend does not draw: drop `--backend recording`"
    );
    assert_eq!(
        ChoiceError::WgpuNotBuilt.to_string(),
        "`--backend wgpu` is not available: this build of starquake leaves the wgpu backend out"
    );
    assert_eq!(
        ChoiceError::WindowNotBuilt.to_string(),
        "`--window` is not available: it needs the wgpu backend, which this build of starquake \
         leaves out"
    );
    assert_eq!(
        MainThreadGone.to_string(),
        "the frontend's main thread no longer takes jobs"
    );
}

#[test]
fn this_build_says_whether_it_has_the_wgpu_backend() {
    // The gate builds the workspace with the app's default features, which turn it on.
    assert_eq!(WGPU_BUILT, cfg!(feature = "wgpu"));
}

#[test]
fn a_run_s_setup_holds_its_choice_and_window() {
    let default = RenderSetup::default();
    assert_eq!(default.choice(), Choice::Recording);
    assert!(default.window().is_none());
    let recording = RenderSetup::recording();
    assert_eq!(recording.choice(), Choice::Recording);
    assert!(recording.window().is_none());

    let wgpu = RenderSetup::headless(Choice::Wgpu(IfUnavailable::UseRecording));
    assert_eq!(wgpu.choice(), Choice::Wgpu(IfUnavailable::UseRecording));
    assert!(wgpu.window().is_none());

    let windowed = RenderSetup::windowed(no_window(640, 480, Arc::new(Gone)));
    assert_eq!(windowed.choice(), Choice::Wgpu(IfUnavailable::EndRun));
    let window = windowed.window().unwrap();
    assert_eq!(window.handles().size().get(), (640, 480));
    assert_eq!(
        window.main().post(Box::new(|| unreachable!())),
        Err(MainThreadGone)
    );
    assert_eq!(
        format!("{window:?}"),
        "WindowTarget { handles: WindowHandles { size: WindowSize(640x480), .. }, .. }"
    );
}
