use super::raw_window_handle::{
    DisplayHandle, HandleError, HasDisplayHandle, HasWindowHandle, WindowHandle,
};
use super::{HEIGHT_MASK, WIDTH_SHIFT, WindowHandles, WindowSize, WindowSource};
use std::sync::Arc;

/// A window whose handles are never available: which error it gives tells two apart.
struct NoWindow(bool);

impl NoWindow {
    fn error(&self) -> HandleError {
        if self.0 {
            HandleError::Unavailable
        } else {
            HandleError::NotSupported
        }
    }
}

impl HasWindowHandle for NoWindow {
    fn window_handle(&self) -> Result<WindowHandle<'_>, HandleError> {
        Err(self.error())
    }
}

impl HasDisplayHandle for NoWindow {
    fn display_handle(&self) -> Result<DisplayHandle<'_>, HandleError> {
        Err(self.error())
    }
}

#[test]
fn the_layout_constants() {
    // Written out: the width in the high half of the word, the height in the low half.
    assert_eq!(WIDTH_SHIFT, 32);
    assert_eq!(HEIGHT_MASK, 0x0000_0000_ffff_ffff);
}

#[test]
fn a_size_reads_back_as_set() {
    let size = WindowSize::new(1280, 720);
    assert_eq!(size.get(), (1280, 720));
    // Each half alone at its extremes, and values with the top bit set, so a swapped or
    // narrowed half shows.
    for (width, height) in [
        (0, 0),
        (1, 0),
        (0, 1),
        (u32::MAX, 0),
        (0, u32::MAX),
        (u32::MAX, u32::MAX),
        (1 << 31, 3),
        (5, 1 << 31),
        (0x8765_4321, 0x1234_5678),
    ] {
        size.set(width, height);
        assert_eq!(size.get(), (width, height), "{width} x {height}");
    }
}

#[test]
fn clones_share_one_size() {
    let frontend = WindowSize::new(640, 480);
    let backend = frontend.clone();
    frontend.set(800, 600);
    assert_eq!(backend.get(), (800, 600));
    backend.set(3, 4);
    assert_eq!(frontend.get(), (3, 4));
}

#[test]
fn handles_keep_their_window_and_size() {
    let size = WindowSize::new(16, 9);
    let window: Arc<dyn WindowSource> = Arc::new(NoWindow(true));
    let handles = WindowHandles::new(window, size.clone());
    // The window kept is the one given: its own error comes back.
    assert!(matches!(
        handles.window().window_handle(),
        Err(HandleError::Unavailable)
    ));
    let other = WindowHandles::new(Arc::new(NoWindow(false)), WindowSize::new(1, 1));
    assert!(matches!(
        other.window().display_handle(),
        Err(HandleError::NotSupported)
    ));
    // The size kept is the frontend's, shared.
    size.set(32, 18);
    assert_eq!(handles.size().get(), (32, 18));
    assert_eq!(handles.clone().size().get(), (32, 18));
}

#[test]
fn debug_shows_the_size_only() {
    let handles = WindowHandles::new(Arc::new(NoWindow(true)), WindowSize::new(1920, 1080));
    assert_eq!(
        format!("{handles:?}"),
        "WindowHandles { size: WindowSize(1920x1080), .. }"
    );
}
