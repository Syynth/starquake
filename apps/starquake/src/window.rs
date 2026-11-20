//! `starquake run --window`: a plain window that shows the frames while the program runs
//! (docs/specs/render-backend.md, decision 11: "a `--window` flag, as recommended"; slice R12).
//! Throwaway once the GPUI frontend arrives; it touches only `apps/`.
//!
//! The process's main thread owns the window and its event loop (`winit` 0.30). The run goes to
//! a thread of its own ([`RUN_THREAD`], with [`RUN_STACK`] bytes of stack), drawing with the wgpu
//! backend (`starquake_core::renderer::RenderSetup::windowed`), and hands the window's handles to
//! the backend: the backend makes the window's surface on the main thread, which on macOS is the
//! only thread allowed to (render-backend, "Present targets"), through a job posted to the event
//! loop (`starquake_core::renderer::MainThread`). When the run ends, its thread wakes the event loop, which closes the window
//! and returns the run's report.
//!
//! What the window does, all of it outside what the guest sees (the guest is paced by the
//! emulated clock, and the frame dumps are the same with and without a window):
//!
//! - it opens at [`FIRST_SIZE`] logical pixels, titled [`TITLE`];
//! - each resize sets the size the backend reads before each present, in physical pixels
//!   (`WindowSize`, slice R8's note on issue #389);
//! - closing it hides it; the run goes on to its stop condition, as a headless run would, since
//!   nothing stops a run from outside yet.
//!
//! # When no window can be made
//!
//! The run ends as an emulator error before it starts, with nothing run, when:
//!
//! - this is not the process's main thread ([`NOT_MAIN_THREAD`]): winit makes event loops on
//!   the main thread only (on macOS `AppKit` requires it, and elsewhere winit refuses by default).
//!   On macOS the check is `AppKit`'s own (`objc2::MainThreadMarker`); elsewhere it is the thread's
//!   name, which Rust gives the main thread as `main` and no other thread unless a program names
//!   one so (a **guess** that holds for this program, which names its other threads otherwise);
//! - the event loop or the window cannot be made ([`NO_WINDOW`], then winit's reason): a Linux
//!   host with neither an X11 nor a Wayland display, for one.
//!
//! The command line always runs on the main thread, so the first case is reached only when
//! [`crate::execute`] is called from another thread, as a test does
//! (`a_window_off_the_main_thread_is_refused_before_the_run`, in `tests/window.rs`). The second
//! is tested where it can be made to happen: a child process on Linux with no display variable
//! (`a_host_with_no_display_ends_the_run_before_it_starts`, same file).
//!
//! **The window path is checked by hand, not by a test**: a window cannot be opened in the gate
//! (render-backend, slice R8: "the window path is checked by hand on a Mac and a Linux
//! desktop"). The pure decisions it takes are tested.

use std::time::Instant;

#[cfg(feature = "wgpu")]
use starquake_core::renderer::{MainThread, RenderSetup, WindowTarget};
use starquake_core::{RunOptions, RunReport, System};

/// Why `--window` refused to make a window: not on the main thread (module docs).
pub const NOT_MAIN_THREAD: &str = "`--window` needs the process's main thread, which owns the \
                                   window and its events; this run was started on another thread";

/// The words before winit's reason when no window could be made (module docs).
pub const NO_WINDOW: &str = "`--window` could not make a window";

/// The window's title.
pub const TITLE: &str = "starquake";

/// The window's size when it opens, in logical pixels: starquake's own choice, the Switch 2's
/// handheld resolution being unknown here, so a 16:9 size that fits a laptop's screen.
pub const FIRST_SIZE: (u32, u32) = (1280, 720);

/// The name of the thread the run goes to.
pub const RUN_THREAD: &str = "run";

/// The bytes of stack the run's thread gets: a **guess**, the macOS main thread's 8 MiB, which
/// a run without a window runs on.
pub const RUN_STACK: usize = 8 << 20;

/// Whether `--window` may make its window on this thread, from whether it is the process's main
/// thread (module docs, "When no window can be made").
pub fn may_open(on_main_thread: bool) -> Result<(), &'static str> {
    if on_main_thread {
        Ok(())
    } else {
        Err(NOT_MAIN_THREAD)
    }
}

/// Whether the calling thread is the process's main thread: `AppKit`'s answer.
#[cfg(all(feature = "wgpu", target_os = "macos"))]
fn on_main_thread() -> bool {
    objc2::MainThreadMarker::new().is_some()
}

/// Whether the calling thread is the process's main thread: its name (module docs).
#[cfg(not(all(feature = "wgpu", target_os = "macos")))]
fn on_main_thread() -> bool {
    std::thread::current().name() == Some("main")
}

/// Runs `system` with `options`, showing its frames in a window (module docs). `started` is
/// when the command began, for the report of a run that never starts.
#[must_use]
pub fn run(system: System, options: RunOptions, started: Instant) -> RunReport {
    if let Err(refused) = may_open(on_main_thread()) {
        return RunReport::emulator_error(refused.to_owned(), started.elapsed());
    }
    shown::run(system, options, started)
}

#[cfg(not(feature = "wgpu"))]
mod shown {
    use std::time::Instant;

    use starquake_core::renderer::ChoiceError;
    use starquake_core::{RunOptions, RunReport, System};

    /// A build without the wgpu backend shows no window (`--window` is refused when the command
    /// line is parsed, so this is not reached from `crate::execute`).
    pub(super) fn run(_system: System, _options: RunOptions, started: Instant) -> RunReport {
        RunReport::emulator_error(ChoiceError::WindowNotBuilt.to_string(), started.elapsed())
    }
}

#[cfg(feature = "wgpu")]
mod shown {
    use std::sync::{Arc, Mutex, PoisonError};
    use std::thread::JoinHandle;
    use std::time::Instant;

    use starquake_core::renderer::{
        MainJob, MainThreadGone, WindowHandles, WindowSize, WindowSource,
    };
    use starquake_core::{RunOptions, RunReport, System};
    use winit::application::ApplicationHandler;
    use winit::dpi::LogicalSize;
    use winit::event::WindowEvent;
    use winit::event_loop::{ActiveEventLoop, EventLoop, EventLoopProxy};
    use winit::window::{Window, WindowId};

    use super::{
        FIRST_SIZE, MainThread, NO_WINDOW, RUN_STACK, RUN_THREAD, RenderSetup, TITLE, WindowTarget,
    };

    /// What the event loop is woken with.
    enum Wake {
        /// A job for the main thread (`MainThread::post`).
        Job(MainJob),
        /// The run has ended.
        Ended,
    }

    /// The main thread, as the run posts jobs to it: the event loop's proxy.
    struct Proxy(Mutex<EventLoopProxy<Wake>>);

    impl MainThread for Proxy {
        fn post(&self, job: MainJob) -> Result<(), MainThreadGone> {
            self.0
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .send_event(Wake::Job(job))
                .map_err(|_| MainThreadGone)
        }
    }

    /// The event loop's state (module docs).
    struct App {
        /// The run, until the window is made and it starts.
        pending: Option<(System, RunOptions)>,
        proxy: EventLoopProxy<Wake>,
        window: Option<Arc<Window>>,
        size: Option<WindowSize>,
        run: Option<JoinHandle<RunReport>>,
        /// Why the run never started.
        failure: Option<String>,
    }

    impl App {
        /// Makes the window and starts the run on its own thread.
        fn start(&mut self, event_loop: &ActiveEventLoop) -> Result<(), String> {
            let Some((system, options)) = self.pending.take() else {
                return Ok(());
            };
            let attributes = Window::default_attributes()
                .with_title(TITLE)
                .with_inner_size(LogicalSize::new(FIRST_SIZE.0, FIRST_SIZE.1));
            let window = Arc::new(
                event_loop
                    .create_window(attributes)
                    .map_err(|error| format!("{NO_WINDOW}: {error}"))?,
            );
            let physical = window.inner_size();
            let size = WindowSize::new(physical.width, physical.height);
            let source: Arc<dyn WindowSource> = Arc::<Window>::clone(&window);
            let target = WindowTarget::new(
                WindowHandles::new(source, size.clone()),
                Arc::new(Proxy(Mutex::new(self.proxy.clone()))),
            );
            let proxy = self.proxy.clone();
            let run = std::thread::Builder::new()
                .name(RUN_THREAD.to_owned())
                .stack_size(RUN_STACK)
                .spawn(move || {
                    let report = system.run_rendering(&options, RenderSetup::windowed(target));
                    // The event loop may already be gone (the application was quit); the report
                    // is returned through the thread's handle either way.
                    let _ = proxy.send_event(Wake::Ended);
                    report
                })
                .map_err(|error| format!("the run's thread could not be started: {error}"))?;
            self.window = Some(window);
            self.size = Some(size);
            self.run = Some(run);
            Ok(())
        }
    }

    impl ApplicationHandler<Wake> for App {
        fn resumed(&mut self, event_loop: &ActiveEventLoop) {
            if let Err(failure) = self.start(event_loop) {
                self.failure = Some(failure);
                event_loop.exit();
            }
        }

        fn user_event(&mut self, event_loop: &ActiveEventLoop, event: Wake) {
            match event {
                Wake::Job(job) => job(),
                Wake::Ended => event_loop.exit(),
            }
        }

        fn window_event(
            &mut self,
            _event_loop: &ActiveEventLoop,
            _window: WindowId,
            event: WindowEvent,
        ) {
            match event {
                WindowEvent::Resized(physical) => {
                    if let Some(size) = &self.size {
                        size.set(physical.width, physical.height);
                    }
                }
                WindowEvent::CloseRequested => {
                    if let Some(window) = &self.window {
                        window.set_visible(false);
                    }
                }
                _ => {}
            }
        }
    }

    /// Makes the event loop and the window, runs the program on its own thread, and returns its
    /// report once it has ended (module docs).
    pub(super) fn run(system: System, options: RunOptions, started: Instant) -> RunReport {
        let event_loop = match EventLoop::<Wake>::with_user_event().build() {
            Ok(event_loop) => event_loop,
            Err(error) => {
                return RunReport::emulator_error(
                    format!("{NO_WINDOW}: {error}"),
                    started.elapsed(),
                );
            }
        };
        let mut app = App {
            pending: Some((system, options)),
            proxy: event_loop.create_proxy(),
            window: None,
            size: None,
            run: None,
            failure: None,
        };
        let looped = event_loop.run_app(&mut app);
        if let Some(run) = app.run.take() {
            // The run started: its report is what the command ends with, however the event loop
            // ended. A run whose loop ended early (the application was quit) goes on without
            // its window to its stop condition; a job it posts after that is refused, which
            // ends it if it is still making the window's surface.
            drop(looped);
            drop(app.window.take());
            return run.join().unwrap_or_else(|_| {
                RunReport::emulator_error(
                    "the run's thread ended by a panic".to_owned(),
                    started.elapsed(),
                )
            });
        }
        let failure = match (app.failure, looped) {
            (Some(failure), _) => failure,
            (None, Err(error)) => format!("{NO_WINDOW}: {error}"),
            (None, Ok(())) => format!("{NO_WINDOW}: the event loop ended before it was resumed"),
        };
        RunReport::emulator_error(failure, started.elapsed())
    }
}
