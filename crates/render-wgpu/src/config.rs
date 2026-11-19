//! What `core` configures the wgpu backend with, beside the completion wake.
//!
//! The spec has `core` pass each backend the compile pool, the wait-while-compiling setting and
//! the completion wake (docs/specs/render-backend.md, "The trait"); the first two arrive with
//! pipelines (slice R6). The memory limit stands in for the device's budget until verification 8
//! says what wgpu exposes ("Memory budget and eviction"). None of these is a hidden input: a
//! backend's results depend only on the trait calls, its `Caps` and this configuration
//! ("Render capture v2", "What R5 and R6 must do now").

use std::num::NonZeroU64;

/// The wgpu backend's settings.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WgpuConfig {
    /// Host memory the backend may use: what `budget()` reports as allowed, and the largest
    /// buffer it accepts (`crate::caps`).
    pub memory_limit: u64,
    /// The largest staging slice one `staging` call may ask for (`Caps::limits`'
    /// `max_staging_size`, capped at the largest buffer).
    pub max_staging_size: u64,
    /// How many submitted lists may be unfinished before `submit` waits for the oldest: the
    /// bounded queue between the back thread and the host GPU (render-backend, "Command lists
    /// are data").
    pub queue_depth: NonZeroU64,
}

impl WgpuConfig {
    /// Settings for a run: 4 GiB of memory, 256 MiB staging slices and three lists in flight.
    /// All three are guesses, to be measured: the spec says only "a few lists" for the queue,
    /// and the memory figure waits on verification 8.
    pub const DEFAULT: Self = Self {
        memory_limit: 4 << 30,
        max_staging_size: 256 << 20,
        queue_depth: match NonZeroU64::new(3) {
            Some(depth) => depth,
            None => NonZeroU64::MIN,
        },
    };
}

/// Whether `submit` must wait: `depth` lists have been handed out (up to `submitted`) and not
/// finished (up to `completed`). Saturating, so a `completed` past `submitted` (which the backend
/// never stores) reads as empty.
pub const fn queue_full(submitted: u64, completed: u64, depth: NonZeroU64) -> bool {
    submitted.saturating_sub(completed) >= depth.get()
}

#[cfg(test)]
mod tests {
    use super::{WgpuConfig, queue_full};
    use std::num::NonZeroU64;

    #[test]
    fn the_queue_is_full_at_its_depth() {
        let depth = WgpuConfig::DEFAULT.queue_depth;
        let d = depth.get();
        assert!(!queue_full(d - 1, 0, depth), "one below");
        assert!(queue_full(d, 0, depth), "at the depth");
        assert!(queue_full(d + 1, 0, depth), "one above");
        assert!(!queue_full(10 + d - 1, 10, depth), "measured from completed");
        assert!(queue_full(10 + d, 10, depth));
        assert!(!queue_full(5, 9, depth), "completed past submitted reads as empty");
        assert!(!queue_full(0, 0, NonZeroU64::MIN));
        assert!(queue_full(1, 0, NonZeroU64::MIN));
        assert!(queue_full(u64::MAX, 0, NonZeroU64::MAX));
        assert!(!queue_full(u64::MAX - 1, 0, NonZeroU64::MAX));
    }

    #[test]
    fn the_defaults() {
        // Written out: the values the module docs give as guesses.
        assert_eq!(WgpuConfig::DEFAULT.memory_limit, 4 * 1024 * 1024 * 1024);
        assert_eq!(WgpuConfig::DEFAULT.max_staging_size, 256 * 1024 * 1024);
        assert_eq!(WgpuConfig::DEFAULT.queue_depth.get(), 3);
    }
}
