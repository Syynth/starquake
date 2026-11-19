//! Whether a record fits the records before it: the checks the writer and the reader share
//! (module docs of [`crate::capture`], "What reading checks, in order").
//!
//! [`Order::check`] decides and changes nothing; [`Order::apply`] records the record and cannot
//! fail, so a caller can run every other fallible step between the two.

use super::record::{Facts, RecordKind};
use super::{CaptureError, CaptureLimits};
use crate::id::{FenceValue, IdKind, ResourceId};
use std::collections::{BTreeMap, HashSet};

/// What the records so far have established.
#[derive(Debug, Default)]
pub(crate) struct Order {
    /// The position the next record has.
    next: u64,
    started: bool,
    limited: bool,
    /// The last accepted list's fence value, zero before any.
    last_fence: FenceValue,
    /// How many shader sources have been given.
    sources: u32,
    /// What the sources given so far are charged ([`super::source_charge`]).
    source_charge: u64,
    /// Ids created and not destroyed.
    live: HashSet<ResourceId>,
    /// Ids destroyed after a fence past every accepted list, by that fence: how many of each
    /// kind ([`IdKind::index`]). Each is still counted in `counts` until a list reaches it.
    held: BTreeMap<FenceValue, [u64; IdKind::COUNT]>,
    /// How many ids of each kind are held ([`IdKind::index`]): those in `live`, and those
    /// counted in `held` (module docs of [`crate::capture`], "What reading checks, in order").
    counts: [u64; IdKind::COUNT],
}

impl Order {
    /// The position the next record has.
    pub(crate) const fn next(&self) -> u64 {
        self.next
    }

    /// The last accepted list's fence value.
    pub(crate) const fn last_fence(&self) -> FenceValue {
        self.last_fence
    }

    /// Whether `id` is live in the capture.
    pub(crate) fn is_live(&self, id: ResourceId) -> bool {
        self.live.contains(&id)
    }

    /// Whether a record of `kind` with `facts` may come next, checked in the documented order:
    /// `Start` first, `Start` once, nothing but `End` after `Limit`, then the kind's own check
    /// (for a `ShaderSource`, its index before its charge; for a `Create*`, the id not live
    /// before the held count).
    pub(crate) fn check(
        &self,
        kind: RecordKind,
        facts: Facts,
        limits: CaptureLimits,
    ) -> Result<(), CaptureError> {
        let record = self.next;
        if !self.started && kind != RecordKind::Start {
            return Err(CaptureError::MissingStart { record, kind });
        }
        if self.started && kind == RecordKind::Start {
            return Err(CaptureError::DuplicateStart { record });
        }
        if self.limited && kind != RecordKind::End {
            return Err(CaptureError::AfterLimit { record, kind });
        }
        if let Some(fence) = facts.fence
            && fence <= self.last_fence
        {
            return Err(CaptureError::FenceOrder {
                record,
                fence: fence.get(),
            });
        }
        if let Some(source) = facts.source {
            if source.index != self.sources {
                return Err(CaptureError::SourceIndex {
                    record,
                    index: source.index,
                    expected: self.sources,
                });
            }
            let needed = self.source_charge.checked_add(source.charge);
            if needed.is_none_or(|needed| needed > limits.source_bytes) {
                return Err(CaptureError::SourcesTooLarge {
                    record,
                    needed: needed.unwrap_or(u64::MAX),
                    max: limits.source_bytes,
                });
            }
        }
        if let Some(index) = facts.uses_source
            && index >= self.sources
        {
            return Err(CaptureError::UnknownSource { record, index });
        }
        if let Some(id) = facts.created {
            if self.live.contains(&id) {
                return Err(CaptureError::IdLive { record, id });
            }
            if self.counts[id.kind().index()] >= limits.live_resources {
                return Err(CaptureError::TooManyLive {
                    record,
                    kind: id.kind(),
                    max: limits.live_resources,
                });
            }
        }
        Ok(())
    }

    /// Records a record that [`Order::check`] accepted.
    pub(crate) fn apply(&mut self, kind: RecordKind, facts: Facts) {
        self.next = self.next.saturating_add(1);
        match kind {
            RecordKind::Start => self.started = true,
            RecordKind::Limit => self.limited = true,
            _ => {}
        }
        if let Some(fence) = facts.fence {
            self.last_fence = fence;
            self.release_held(fence);
        }
        if let Some(source) = facts.source {
            self.sources = self.sources.saturating_add(1);
            self.source_charge = self.source_charge.saturating_add(source.charge);
        }
        if let Some(id) = facts.created
            && self.live.insert(id)
        {
            let count = &mut self.counts[id.kind().index()];
            *count = count.saturating_add(1);
        }
        if let Some((id, after)) = facts.destroyed
            && self.live.remove(&id)
        {
            if after > self.last_fence {
                // Replay holds the id and its resource until a list reaches `after`.
                let held = self.held.entry(after).or_default();
                held[id.kind().index()] = held[id.kind().index()].saturating_add(1);
            } else {
                let count = &mut self.counts[id.kind().index()];
                *count = count.saturating_sub(1);
            }
        }
    }

    /// Stops counting the ids held for fences at or below `fence`, which a list has now reached.
    fn release_held(&mut self, fence: FenceValue) {
        while let Some(entry) = self.held.first_entry() {
            if *entry.key() > fence {
                return;
            }
            let released = entry.remove();
            for (count, held) in self.counts.iter_mut().zip(released) {
                *count = count.saturating_sub(held);
            }
        }
    }
}
