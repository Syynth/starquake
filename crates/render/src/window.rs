//! What a window present target carries: the frontend's window, as `raw-window-handle` 0.6.2
//! display and window handles, and its size (docs/specs/render-backend.md, "Present targets";
//! slice R8).
//!
//! The frontend owns the window and its event loop; a backend owns the surface it makes from the
//! window's handles. Two things decide this module's shape:
//!
//! - **An owned window, not bare handles.** [`WindowHandles`] holds the window itself, shared
//!   (`Arc<dyn WindowSource>`): any `Send + Sync` value with both handle traits, such as a
//!   windowing library's window. A backend that makes its surface from an owned window keeps the
//!   window alive as long as the surface, so it needs no promise that a bare handle stays valid:
//!   wgpu makes a surface from such a window through its safe constructor
//!   (`wgpu-30.0.2/src/api/instance.rs`, `Instance::create_surface`), and the wgpu backend opts
//!   into no `unsafe` code for it (verification 11). Bare `RawWindowHandle`s would need wgpu's
//!   unsafe constructor, and they are not `Send`, so this crate's `PresentTarget` could not
//!   cross to a backend's render thread without `unsafe` here, which this crate denies.
//! - **The size travels beside the handles.** The handle traits say nothing of a window's size,
//!   and a backend's render thread must not ask the windowing library (on macOS that call is
//!   answered on the main thread, which may be waiting on the render thread). So the frontend
//!   writes the window's drawable size, in physical pixels, into a [`WindowSize`] on each resize,
//!   and the backend reads it before each present ([`WindowSize::get`] never blocks).
//!
//! **The size departs from the approved spec and awaits the owner.** The spec
//! (docs/specs/render-backend.md, "Present targets") defines the variant as
//! `Window(WindowHandles)` with the doc sentence "A window the frontend created;
//! `raw-window-handle` 0.6.2 display and window handles", which names the handles only.
//! [`WindowHandles`] also carries the [`WindowSize`], for the reason above. Whether that
//! addition stands is an open owner question (raised in the pull request that added this
//! module); until it is answered, the size is a departure, not the spec's design.
//!
//! Re-exported: the `raw_window_handle` crate itself, so a frontend (through `core`) names the
//! same version of the traits this crate does.

pub use raw_window_handle;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

/// A window a backend can make a surface from: anything with display and window handles that
/// can be shared between threads. Implemented for every such type.
pub trait WindowSource: HasWindowHandle + HasDisplayHandle + Send + Sync {}

impl<T: HasWindowHandle + HasDisplayHandle + Send + Sync + ?Sized> WindowSource for T {}

/// Where the width sits in [`WindowSize`]'s one 64-bit word: the high half. The height is the
/// low half ([`HEIGHT_MASK`]).
pub const WIDTH_SHIFT: u32 = 32;

/// The height's bits in [`WindowSize`]'s word.
pub const HEIGHT_MASK: u64 = 0xffff_ffff;

/// A window's drawable size in physical pixels, shared between the frontend, which sets it when
/// the window is resized, and a backend, which reads it before each present. Clones share one
/// value. Width and height are stored in one word, so a reader never sees the width of one
/// resize with the height of another. Zero in either means nothing can be shown (a minimised
/// window); a backend then drops the frames presented meanwhile.
#[derive(Clone)]
pub struct WindowSize(Arc<AtomicU64>);

impl WindowSize {
    /// A size of `width` by `height` physical pixels.
    pub fn new(width: u32, height: u32) -> Self {
        Self(Arc::new(AtomicU64::new(pack(width, height))))
    }

    /// Sets the size, for every clone.
    pub fn set(&self, width: u32, height: u32) {
        self.0.store(pack(width, height), Ordering::Relaxed);
    }

    /// The size last set: width, then height.
    pub fn get(&self) -> (u32, u32) {
        unpack(self.0.load(Ordering::Relaxed))
    }
}

fn pack(width: u32, height: u32) -> u64 {
    (u64::from(width) << WIDTH_SHIFT) | u64::from(height)
}

fn unpack(word: u64) -> (u32, u32) {
    let width = u32::try_from(word >> WIDTH_SHIFT).unwrap_or(u32::MAX);
    let height = u32::try_from(word & HEIGHT_MASK).unwrap_or(u32::MAX);
    (width, height)
}

impl fmt::Debug for WindowSize {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (width, height) = self.get();
        write!(f, "WindowSize({width}x{height})")
    }
}

/// A window to present into (`PresentTarget::Window`): the frontend's window and its size
/// (module docs).
#[derive(Clone)]
pub struct WindowHandles {
    window: Arc<dyn WindowSource>,
    size: WindowSize,
}

impl WindowHandles {
    /// The window `window`, whose drawable size the frontend keeps in `size`.
    pub fn new(window: Arc<dyn WindowSource>, size: WindowSize) -> Self {
        Self { window, size }
    }

    /// The window.
    pub fn window(&self) -> &Arc<dyn WindowSource> {
        &self.window
    }

    /// Its size, as the frontend last set it.
    pub fn size(&self) -> &WindowSize {
        &self.size
    }
}

impl fmt::Debug for WindowHandles {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WindowHandles")
            .field("size", &self.size)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests;
