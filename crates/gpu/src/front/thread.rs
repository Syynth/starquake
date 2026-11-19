//! The front thread itself: [`spawn`] runs [`Front::run`] on a thread of its own, and
//! [`FrontThread::stop`] ends it and reports how it ended, a panic included.

use std::io;
use std::sync::Arc;
use std::thread::{self, JoinHandle};

use super::{Front, FrontError, FrontShared};

/// A running front thread.
pub struct FrontThread {
    shared: Arc<FrontShared>,
    handle: JoinHandle<Result<(), FrontError>>,
}

/// Starts the front thread, named `gpu-front`, running `front`.
///
/// # Errors
///
/// The host's error if the thread cannot be made.
pub fn spawn(front: Front) -> io::Result<FrontThread> {
    let shared = Arc::clone(front.shared());
    let handle = thread::Builder::new()
        .name("gpu-front".to_owned())
        .spawn(move || front.run())?;
    Ok(FrontThread { shared, handle })
}

impl FrontThread {
    /// What the callers share with the thread.
    pub fn shared(&self) -> &Arc<FrontShared> {
        &self.shared
    }

    /// Whether the thread has ended (stopped, or failed).
    pub fn is_finished(&self) -> bool {
        self.handle.is_finished()
    }

    /// Asks the thread to stop and waits for it.
    ///
    /// # Errors
    ///
    /// The [`FrontError`] that ended the thread before it was asked to stop, or
    /// [`FrontError::Panicked`] if it panicked.
    pub fn stop(self) -> Result<(), FrontError> {
        self.shared.stop();
        self.handle.join().unwrap_or(Err(FrontError::Panicked))
    }
}
