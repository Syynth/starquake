//! Where the copy engines' side effects and the CPU-write drain go: [`CacheSink`], the copy
//! engines' [`CopySink`] over one address space's caches, and [`Drained`], the drain's
//! collector.

use core::ops::Range;
use std::time::Duration;

use starquake_memory::{PAGE_SIZE, PageRange};
use starquake_render::CommandList;
use thiserror::Error;

use super::{ExecCounters, ShaderCache};
use crate::back::buffer::{BufferCache, InvalidationSink};
use crate::back::copy::CopySink;
use crate::back::fence::{FenceError, Fences};
use crate::back::readback::ReadbackIds;
use crate::back::texture::{TextureCache, WriteBackError};
use crate::memory::{GpuFault, GpuMemory};
use crate::ring::Release;

/// Why a write-back before a copy failed; logged, and the copy reads guest memory as it is.
#[derive(Debug, Error)]
pub(super) enum WriteBackFailed {
    /// Recording, closing or waiting for the list holding the readbacks failed
    /// ([`Fences::catch_up`]).
    #[error(transparent)]
    Fence(#[from] FenceError),
    /// Writing an image back faulted; the ranges written before it were dropped from the caches.
    #[error("writing an image back to guest memory faulted: {0}")]
    Fault(GpuFault),
    /// The texture cache could not write the images back.
    #[error(transparent)]
    WriteBack(WriteBackError),
}

/// The ranges one CPU-write drain reported, in order.
pub(super) struct Drained(pub(super) Vec<PageRange>);

impl InvalidationSink for Drained {
    fn invalidate(&mut self, range: PageRange) {
        self.0.push(range);
    }
}

/// The whole pages covering `range`, as the shader cache's `invalidate` takes them; `None` for
/// an empty range or one whose last page would pass the end of the address space.
pub(super) fn pages(range: &Range<u64>) -> Option<PageRange> {
    if range.is_empty() {
        return None;
    }
    let start = range.start - range.start % PAGE_SIZE;
    let end = range.end.checked_next_multiple_of(PAGE_SIZE)?;
    PageRange::new(start, end - start).ok()
}

/// The copy engines' sink over one address space (the executor's module docs, "Copies of images
/// the texture cache holds"):
///
/// - [`CopySink::make_current`] writes back every GPU-modified image of the range first;
/// - [`CopySink::invalidate`] drops the range from the buffer and texture caches and sends its
///   pages to the shader cache;
/// - [`CopySink::release`] is kept, in order, for the executor to attach once the engine is done.
pub(super) struct CacheSink<'a> {
    pub(super) buffers: &'a mut BufferCache,
    pub(super) textures: &'a mut TextureCache,
    pub(super) memory: &'a dyn GpuMemory,
    pub(super) shaders: &'a mut dyn ShaderCache,
    pub(super) fences: &'a mut Fences,
    pub(super) readbacks: &'a mut ReadbackIds,
    pub(super) catch_up: Duration,
    pub(super) counters: &'a mut ExecCounters,
    pub(super) scratch: &'a mut CommandList,
    pub(super) releases: Vec<Release>,
}

impl CacheSink<'_> {
    /// The back thread wrote `range` of guest memory: every cache drops it.
    fn written(&mut self, range: Range<u64>) {
        self.buffers.invalidate(range.clone());
        self.textures.invalidate(range.clone());
        if let Some(pages) = pages(&range) {
            self.shaders.invalidate(pages);
        }
    }

    /// The write-back of `range` (module docs), or the reason it failed.
    fn write_back(&mut self, range: Range<u64>) -> Result<(), WriteBackFailed> {
        let write_back = self
            .textures
            .begin_write_back(range, self.readbacks, self.scratch);
        if write_back.is_empty() {
            self.scratch.commands.clear();
            return Ok(());
        }
        self.fences.record_list(self.scratch)?;
        self.fences.catch_up(self.catch_up)?;
        let finished = self.textures.finish_write_back(self.memory, write_back);
        let (written, failure) = match finished {
            Ok(written) => (written, None),
            Err(WriteBackError::Fault { fault, written }) => {
                (written, Some(WriteBackFailed::Fault(fault)))
            }
            Err(error) => (Vec::new(), Some(WriteBackFailed::WriteBack(error))),
        };
        for range in written {
            // The texture cache has marked its own images; the others drop the range.
            self.buffers.invalidate(range.clone());
            if let Some(pages) = pages(&range) {
                self.shaders.invalidate(pages);
            }
        }
        failure.map_or(Ok(()), Err)
    }
}

impl CopySink for CacheSink<'_> {
    fn make_current(&mut self, range: Range<u64>) {
        if !self.textures.needs_write_back(range.clone()) {
            return;
        }
        self.counters.write_backs = self.counters.write_backs.saturating_add(1);
        if let Err(reason) = self.write_back(range.clone()) {
            self.counters.write_back_failures = self.counters.write_back_failures.saturating_add(1);
            tracing::warn!(
                start = range.start,
                end = range.end,
                %reason,
                "a write-back before a copy failed; the copy reads guest memory as it is"
            );
        }
    }

    fn invalidate(&mut self, range: Range<u64>) {
        self.written(range);
    }

    fn release(&mut self, release: Release) {
        self.releases.push(release);
    }
}
