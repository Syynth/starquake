//! A set of guest pages kept as disjoint, coalesced ranges: the watched set and the dirty log of
//! [`super::WriteWatch`].

use std::collections::BTreeMap;

use super::PageRange;
use crate::translation::PAGE_SIZE;

/// Pages as ranges, keyed by start, each mapping to its end (exclusive).
///
/// Invariant: no two ranges overlap or touch (a range never starts where another ends), and
/// every range is a valid [`PageRange`] (page aligned, non-empty, end representable).
#[derive(Debug, Default)]
pub(super) struct PageSet {
    ranges: BTreeMap<u64, u64>,
}

impl PageSet {
    /// Adds every page of `range`, merging it with ranges it overlaps or touches.
    pub(super) fn insert(&mut self, range: PageRange) {
        let mut start = range.start();
        let mut end = range.end();
        // A range starting at or below `start` that reaches it.
        if let Some((&below_start, &below_end)) = self.ranges.range(..=start).next_back()
            && below_end >= start
        {
            start = below_start;
            end = end.max(below_end);
            self.ranges.remove(&below_start);
        }
        // Ranges starting inside the new one or exactly at its end. Each pass removes one, so
        // the loop runs at most once per range already in the set.
        while let Some((&next_start, &next_end)) = self.ranges.range(start..=end).next() {
            end = end.max(next_end);
            self.ranges.remove(&next_start);
        }
        self.ranges.insert(start, end);
    }

    /// Removes the page starting at `page` (page aligned), splitting the range that holds it.
    /// Returns whether the page was in the set.
    pub(super) fn remove_page(&mut self, page: u64) -> bool {
        let Some((&start, &end)) = self.ranges.range(..=page).next_back() else {
            return false;
        };
        if end <= page {
            return false;
        }
        self.ranges.remove(&start);
        if start < page {
            self.ranges.insert(start, page);
        }
        // `page` is below `end`, which is representable, so `page + PAGE_SIZE <= end` fits.
        if let Some(after) = page.checked_add(PAGE_SIZE)
            && after < end
        {
            self.ranges.insert(after, end);
        }
        true
    }

    /// Adds every page of `range` that is not in `except`.
    pub(super) fn insert_except(&mut self, range: PageRange, except: &PageSet) {
        let mut from = range.start();
        for (&start, &end) in except.ranges.range(..range.end()) {
            if end <= from {
                continue;
            }
            // `from < start < range.end()`, so the gap is a valid, non-empty range.
            if start > from {
                self.insert(PageRange {
                    start: from,
                    size: start - from,
                });
            }
            from = end;
        }
        if from < range.end() {
            self.insert(PageRange {
                start: from,
                size: range.end() - from,
            });
        }
    }

    /// Whether the page holding `va` is in the set.
    pub(super) fn contains(&self, va: u64) -> bool {
        self.ranges
            .range(..=va)
            .next_back()
            .is_some_and(|(_, &end)| va < end)
    }

    /// Moves every range into `out` in ascending order and empties the set.
    pub(super) fn drain_into(&mut self, out: &mut Vec<PageRange>) {
        let ranges = std::mem::take(&mut self.ranges);
        out.extend(ranges.into_iter().map(|(start, end)| PageRange {
            start,
            size: end - start,
        }));
    }
}
