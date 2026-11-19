//! Readback ids for the back thread's write-backs (gpu-protocol spec, "GPU writes and the CPU").
//!
//! A `Command::Readback` names the [`ReadbackId`] its bytes are collected by
//! (`Backend::take_readback`), and the render backend does not hand ids out: the GPU crate picks
//! them. Every cache that writes GPU-written data back to guest memory takes its ids from one
//! [`ReadbackIds`] per back thread, so two caches' readbacks in one list never share an id. The
//! texture cache (slice G17) is the first user; the buffer cache's write-back of GPU-written
//! buffers (slice G16's "GPU-written buffers") takes its ids from the same allocator when it
//! arrives.

use starquake_render::ReadbackId;

/// Hands out distinct [`ReadbackId`]s: the index counts up, and the generation counts each time
/// the index wraps, so `2^64` ids are distinct.
#[derive(Debug, Default)]
pub struct ReadbackIds {
    next: u64,
}

impl ReadbackIds {
    /// An allocator whose first id is index 0, generation 0.
    pub const fn new() -> Self {
        Self { next: 0 }
    }

    /// An allocator that has already handed out `n` ids, for tests of the wrap.
    #[cfg(test)]
    pub(crate) const fn after(n: u64) -> Self {
        Self { next: n }
    }

    /// The next id: index `n mod 2^32`, generation `n / 2^32`, for the `n`th call (from 0).
    /// After `2^64` calls it keeps returning the last id, which no back thread lives to reach.
    pub fn allocate(&mut self) -> ReadbackId {
        let n = self.next;
        self.next = self.next.saturating_add(1);
        let index = u32::try_from(n & u64::from(u32::MAX)).unwrap_or(u32::MAX);
        let generation = u32::try_from(n >> u32::BITS).unwrap_or(u32::MAX);
        ReadbackId::new(index, generation)
    }
}
