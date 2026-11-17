//! Counted cover of GPU address ranges ([`Cover`]): how many of a set of ranges, which may
//! overlap, cover each address, and the runs of a range that some of them cover or none does.
//! The bookkeeping keeps two: the registrations' ranges (the second layer of the table is their
//! union, since every registration maps at its own address with one kind) and, per address
//! space, the ranges waiting ordered jobs hold (`super`'s module docs, "Pages a waiting job
//! holds").
//!
//! The cost of a change is the logarithm of the steps held plus the steps inside the changed
//! range; of [`Cover::runs`], the logarithm plus the steps inside the asked range. Each range
//! added adds at most two steps.

use std::collections::BTreeMap;
use std::ops::Bound::Excluded;

/// How many ranges cover each address: a step function. Each key starts a segment, and its value
/// is the count from there to the next key; nothing covers the addresses before the first key.
/// No key holds the count of the segment before it, so the steps are as few as the function
/// allows.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Cover {
    steps: BTreeMap<u64, u32>,
}

impl Cover {
    /// Counts `start..end` once more. An empty range changes nothing.
    pub(crate) fn add(&mut self, start: u64, end: u64) {
        self.change(start, end, true);
    }

    /// Counts `start..end` once less: undoes an [`add`](Self::add) of the same range. A count
    /// already 0 stays 0 (a range never added is gsv's own mistake, not the guest's).
    pub(crate) fn remove(&mut self, start: u64, end: u64) {
        self.change(start, end, false);
    }

    /// Whether nothing is covered.
    pub(crate) fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// The longest runs inside `low..high` that some range covers (`covered`) or that none does
    /// (`!covered`), in address order; runs that touch are one.
    pub(crate) fn runs(&self, low: u64, high: u64, covered: bool) -> Vec<(u64, u64)> {
        let mut runs: Vec<(u64, u64)> = Vec::new();
        if low >= high {
            return runs;
        }
        let mut push = |from: u64, to: u64, count: u32| {
            if from >= to || (count > 0) != covered {
                return;
            }
            match runs.last_mut() {
                Some(last) if last.1 == from => last.1 = to,
                _ => runs.push((from, to)),
            }
        };
        let mut at = low;
        let mut count = self.count_at(low);
        for (&key, &next) in self.steps.range((Excluded(low), Excluded(high))) {
            push(at, key, count);
            at = key;
            count = next;
        }
        push(at, high, count);
        runs
    }

    /// The count of the segment holding `at`.
    fn count_at(&self, at: u64) -> u32 {
        self.steps
            .range(..=at)
            .next_back()
            .map_or(0, |(_, &count)| count)
    }

    fn change(&mut self, start: u64, end: u64, up: bool) {
        if start >= end {
            return;
        }
        self.split(start);
        self.split(end);
        for (_, count) in self.steps.range_mut(start..end) {
            *count = if up {
                count.saturating_add(1)
            } else {
                count.saturating_sub(1)
            };
        }
        // Only the two ends can now hold the count before them: inside, every count moved alike.
        self.tidy(start);
        self.tidy(end);
    }

    /// Makes `at` a key, holding the count already there.
    fn split(&mut self, at: u64) {
        if !self.steps.contains_key(&at) {
            let count = self.count_at(at);
            self.steps.insert(at, count);
        }
    }

    /// Drops the key `at` if it holds the count of the segment before it.
    fn tidy(&mut self, at: u64) {
        let Some(&count) = self.steps.get(&at) else {
            return;
        };
        let before = self.steps.range(..at).next_back().map_or(0, |(_, &c)| c);
        if count == before {
            self.steps.remove(&at);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The addresses the model test plays over: small, so every pair of bounds can be asked.
    const SPACE: u64 = 12;

    /// A fixed-seed linear congruential generator: the model test's operations, the same on
    /// every run.
    struct Lcg(u64);

    impl Lcg {
        fn next(&mut self, below: u64) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (self.0 >> 33) % below
        }
    }

    /// The model's runs: per address, whether its count is non-zero, grouped.
    fn model_runs(counts: &[u32], low: u64, high: u64, covered: bool) -> Vec<(u64, u64)> {
        let mut runs: Vec<(u64, u64)> = Vec::new();
        for at in low..high {
            if (counts[usize::try_from(at).unwrap()] > 0) != covered {
                continue;
            }
            match runs.last_mut() {
                Some(last) if last.1 == at => last.1 = at + 1,
                _ => runs.push((at, at + 1)),
            }
        }
        runs
    }

    /// Adds and removes, interleaved and overlapping, against a per-address count: every run
    /// query over every pair of bounds agrees, and no step is redundant.
    #[test]
    fn the_cover_matches_a_per_address_model() {
        let mut cover = Cover::default();
        let mut counts = vec![0u32; usize::try_from(SPACE).unwrap()];
        let mut added: Vec<(u64, u64)> = Vec::new();
        let mut rng = Lcg(0x5EED);
        for _ in 0..400 {
            let remove = !added.is_empty() && rng.next(3) == 0;
            let (start, end) = if remove {
                added.swap_remove(
                    usize::try_from(rng.next(u64::try_from(added.len()).unwrap())).unwrap(),
                )
            } else {
                let start = rng.next(SPACE);
                let end = start + rng.next(SPACE - start + 1);
                added.push((start, end));
                (start, end)
            };
            for at in start..end {
                let count = &mut counts[usize::try_from(at).unwrap()];
                *count = if remove { *count - 1 } else { *count + 1 };
            }
            if remove {
                cover.remove(start, end);
            } else {
                cover.add(start, end);
            }
            for low in 0..=SPACE {
                for high in low..=SPACE {
                    for covered in [false, true] {
                        assert_eq!(
                            cover.runs(low, high, covered),
                            model_runs(&counts, low, high, covered),
                            "{low}..{high} covered {covered}"
                        );
                    }
                }
            }
            let mut before = 0;
            for (&at, &count) in &cover.steps {
                assert_ne!(count, before, "a redundant step at {at}");
                let model = counts.get(usize::try_from(at).unwrap()).copied();
                assert_eq!(count, model.unwrap_or(0), "the count at {at}");
                before = count;
            }
        }
        for (start, end) in added {
            cover.remove(start, end);
        }
        assert!(cover.is_empty());
    }

    /// Two ranges that touch are one run, and the runs of a range that ends exactly where a
    /// cover starts do not include it.
    #[test]
    fn touching_ranges_are_one_run_and_ends_are_exclusive() {
        let mut cover = Cover::default();
        cover.add(2, 4);
        cover.add(4, 6);
        assert_eq!(cover.runs(0, 10, true), [(2, 6)]);
        assert_eq!(cover.runs(0, 2, true), []);
        assert_eq!(cover.runs(0, 2, false), [(0, 2)]);
        assert_eq!(cover.runs(6, 10, true), []);
        cover.remove(2, 4);
        assert_eq!(cover.runs(0, 10, true), [(4, 6)]);
        assert_eq!(cover.runs(0, 10, false), [(0, 4), (6, 10)]);
        // A range never added leaves the counts at 0.
        cover.remove(0, 2);
        assert_eq!(cover.runs(0, 10, true), [(4, 6)]);
        cover.add(3, 3);
        assert_eq!(cover.runs(0, 10, true), [(4, 6)]);
    }
}
