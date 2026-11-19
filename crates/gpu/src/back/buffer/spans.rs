//! [`Spans`]: an exact set of GPU addresses kept as disjoint, coalesced half-open ranges, for the
//! buffer cache's per-entry page state (which pages are stale, which wait for their watch).
//!
//! The copy engines' [`crate::back::copy::PendingWrites`] also merges ranges, but it is a capped
//! queue of invalidations that over-approximates once it holds too many; the cache's page state
//! must stay exact (a page wrongly marked clean is never uploaded again), and needs removal,
//! which that queue does not have. The memory crate's page set behind `WriteWatch` is private to
//! that crate.

use core::ops::Range;
use std::collections::BTreeMap;

/// A set of addresses as disjoint ranges that neither overlap nor touch, by start.
///
/// Ranges are separated by at least one address not in the set, so a set inside an extent of `n`
/// addresses holds at most `n / 2 + 1` ranges: its size is bounded by the extent it lives in (one
/// cache entry), not by how often it is changed.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(super) struct Spans {
    /// End by start.
    ranges: BTreeMap<u64, u64>,
}

impl Spans {
    /// Adds every address of `range`. An empty range changes nothing.
    pub(super) fn insert(&mut self, range: Range<u64>) {
        if range.is_empty() {
            return;
        }
        let mut start = range.start;
        let mut end = range.end;
        // The range starting at or before `start` may reach it (overlap or touch).
        if let Some((&s, &e)) = self.ranges.range(..=start).next_back()
            && e >= start
        {
            start = s;
            end = end.max(e);
        }
        // Every range starting inside `start..=end` overlaps or touches.
        let inside: Vec<(u64, u64)> = self
            .ranges
            .range(start..=end)
            .map(|(&s, &e)| (s, e))
            .collect();
        for (s, e) in inside {
            self.ranges.remove(&s);
            end = end.max(e);
        }
        self.ranges.insert(start, end);
    }

    /// Removes every address of `range`. An empty range changes nothing.
    pub(super) fn remove(&mut self, range: Range<u64>) {
        if range.is_empty() {
            return;
        }
        // A range starting before `range.start` may reach into it: keep its head, and its tail
        // past `range.end` if it spans the whole of `range`.
        if let Some((&s, &e)) = self.ranges.range(..range.start).next_back()
            && e > range.start
        {
            self.ranges.insert(s, range.start);
            if e > range.end {
                self.ranges.insert(range.end, e);
            }
        }
        let inside: Vec<(u64, u64)> = self
            .ranges
            .range(range.start..range.end)
            .map(|(&s, &e)| (s, e))
            .collect();
        for (s, e) in inside {
            self.ranges.remove(&s);
            if e > range.end {
                self.ranges.insert(range.end, e);
            }
        }
    }

    /// Adds every address of `other`.
    pub(super) fn insert_all(&mut self, other: &Self) {
        for range in other.iter() {
            self.insert(range);
        }
    }

    /// Empties the set.
    pub(super) fn clear(&mut self) {
        self.ranges.clear();
    }

    /// The ranges, in increasing address order.
    pub(super) fn iter(&self) -> impl Iterator<Item = Range<u64>> + '_ {
        self.ranges.iter().map(|(&s, &e)| s..e)
    }

    /// How many addresses the set holds. Saturates, which no set inside one cache entry reaches.
    pub(super) fn len(&self) -> u64 {
        self.iter()
            .map(|r| r.end - r.start)
            .fold(0, u64::saturating_add)
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use proptest::test_runner::{Config, RngSeed};

    use super::Spans;

    /// The universe the model tests work in: addresses `0..UNIVERSE`.
    const UNIVERSE: u64 = 64;

    fn idx(a: u64) -> usize {
        usize::try_from(a).unwrap()
    }

    /// The ranges as pairs, for comparing.
    fn pairs(s: &Spans) -> Vec<(u64, u64)> {
        s.iter().map(|r| (r.start, r.end)).collect()
    }

    fn model_of(spans: &Spans) -> Vec<bool> {
        let mut bits = vec![false; idx(UNIVERSE)];
        for r in spans.iter() {
            for a in r {
                bits[idx(a)] = true;
            }
        }
        bits
    }

    /// The set is in canonical form: ranges non-empty, sorted, neither overlapping nor touching.
    fn canonical(spans: &Spans) -> bool {
        let ranges: Vec<_> = spans.iter().collect();
        ranges.iter().all(|r| r.start < r.end) && ranges.windows(2).all(|w| w[0].end < w[1].start)
    }

    #[test]
    fn insert_merges_overlapping_and_touching_ranges() {
        let mut s = Spans::default();
        s.insert(10..20);
        s.insert(30..40);
        s.insert(20..25); // touches the first
        assert_eq!(pairs(&s), [(10, 25), (30, 40)]);
        s.insert(24..31); // overlaps both
        assert_eq!(pairs(&s), [(10, 40)]);
        s.insert(5..5);
        assert_eq!(pairs(&s), [(10, 40)]);
        assert_eq!(s.len(), 30);
    }

    #[test]
    fn remove_splits_and_keeps_the_neighbours() {
        let mut s = Spans::default();
        s.insert(0..10);
        s.insert(20..30);
        s.remove(5..25);
        assert_eq!(pairs(&s), [(0, 5), (25, 30)]);
        s.remove(1..2);
        assert_eq!(pairs(&s), [(0, 1), (2, 5), (25, 30)]);
        // A removal ending exactly where a range starts leaves it whole.
        s.remove(20..25);
        assert_eq!(pairs(&s), [(0, 1), (2, 5), (25, 30)]);
        s.remove(0..0);
        assert_eq!(s.len(), 9);
        s.clear();
        assert_eq!(s.iter().count(), 0);
    }

    proptest! {
        #![proptest_config(Config {
            cases: 512,
            rng_seed: RngSeed::Fixed(0x5350_414e),
            ..Config::default()
        })]

        /// Any sequence of inserts and removes agrees with a bit-per-address model and leaves the
        /// set canonical.
        #[test]
        fn matches_a_bit_model(ops in prop::collection::vec(
            (any::<bool>(), 0..UNIVERSE, 0..UNIVERSE), 0..24)
        ) {
            let mut spans = Spans::default();
            let mut model = vec![false; idx(UNIVERSE)];
            for (add, a, b) in ops {
                let r = a.min(b)..a.max(b);
                for x in r.clone() {
                    model[idx(x)] = add;
                }
                if add { spans.insert(r) } else { spans.remove(r) }
                prop_assert!(canonical(&spans));
                prop_assert_eq!(&model_of(&spans), &model);
            }
            let mut copy = Spans::default();
            copy.insert_all(&spans);
            prop_assert_eq!(copy, spans.clone());
            let set = model.iter().filter(|&&b| b).count();
            prop_assert_eq!(spans.len(), u64::try_from(set).unwrap());
        }
    }
}
