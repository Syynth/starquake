//! Whether a record fits the records before it: the checks the writer and the reader share
//! (module docs of [`crate::capture`], "What reading checks, in order").
//!
//! [`Order::check`] decides and changes nothing; [`Order::apply`] records the record and cannot
//! fail, so a caller can run every other fallible step between the two.

use super::CaptureError;
use super::record::{Facts, RecordKind};
use crate::id::{FenceValue, ResourceId};
use std::collections::HashSet;

/// Kinds of id, for the live counts.
const ID_KINDS: usize = 6;

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
    /// Ids created and not destroyed.
    live: HashSet<ResourceId>,
    /// How many of `live` are of each kind ([`kind_index`]).
    counts: [u64; ID_KINDS],
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
    /// `Start` first, `Start` once, nothing but `End` after `Limit`, then the kind's own check.
    pub(crate) fn check(
        &self,
        kind: RecordKind,
        facts: Facts,
        max_live: u64,
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
        if let Some(index) = facts.source
            && index != self.sources
        {
            return Err(CaptureError::SourceIndex {
                record,
                index,
                expected: self.sources,
            });
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
            if self.counts[kind_index(id)] >= max_live {
                return Err(CaptureError::TooManyLive {
                    record,
                    kind: id.kind_name(),
                    max: max_live,
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
        }
        if facts.source.is_some() {
            self.sources = self.sources.saturating_add(1);
        }
        if let Some(id) = facts.created
            && self.live.insert(id)
        {
            let count = &mut self.counts[kind_index(id)];
            *count = count.saturating_add(1);
        }
        if let Some(id) = facts.destroyed
            && self.live.remove(&id)
        {
            let count = &mut self.counts[kind_index(id)];
            *count = count.saturating_sub(1);
        }
    }
}

/// Where `id`'s kind is counted.
const fn kind_index(id: ResourceId) -> usize {
    match id {
        ResourceId::Buffer(_) => 0,
        ResourceId::Texture(_) => 1,
        ResourceId::View(_) => 2,
        ResourceId::Sampler(_) => 3,
        ResourceId::Shader(_) => 4,
        ResourceId::Pipeline(_) => 5,
    }
}
