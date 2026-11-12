//! Turns the ranges [`Stage1Tables`](starquake_memory::stage1::Stage1Tables) wants invalidated
//! into the operations the EL1 shim's flush entry points execute (`super::shim`, "TLB flush").
//!
//! Every operation uses a broadcast form (`TLBI ...IS`), which reaches every vCPU of the VM
//! (`tools/hvprobe/README.md`, "TLB invalidation across vCPUs", measured with `VMALLE1IS`; the
//! by-address forms used here are checked across vCPUs by this crate's hardware tests).
//!
//! The operand layouts are from the Arm Architecture Reference Manual (Arm DDI 0487), the
//! descriptions of `TLBI VAE1IS`, `TLBI VALE1IS` and `TLBI ASIDE1IS`: the address-space id in
//! bits 63:48, and for the by-address forms the page number (`VA[55:12]`) in bits 43:0, with the
//! translation-level hint `TTL` (bits 47:44) zero, meaning "any level".

use starquake_memory::PAGE_SIZE;
use starquake_memory::stage1::{FlushRange, VA_SPACE_END};

use crate::Error;

/// The address-space id sits at this bit of a `TLBI` operand (and of `TTBR0_EL1`).
pub(super) const ASID_SHIFT: u32 = 48;

/// The most pages one flush ([`HvfVcpu::flush_tlb`](super::HvfVcpu::flush_tlb)) invalidates
/// page by page. A flush that covers more invalidates everything cached under the
/// address-space id instead, which costs later TLB misses but no long loop at EL1. Starquake's
/// own number, a guess to be replaced by a measurement of the flush entry point's cost
/// (`docs/specs/execution-model.md`, "To verify", item 3).
pub const MAX_PAGE_FLUSHES: u64 = 512;

/// One run of a flush entry point.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum FlushOp {
    /// Invalidate `count` pages one by one, starting with the page `operand` names.
    Pages {
        /// The `TLBI` operand of the first page; the next page's is one more.
        operand: u64,
        /// How many pages; never zero.
        count: u64,
        /// Last-level entries only (`TLBI VALE1IS`) or every level of the walk
        /// (`TLBI VAE1IS`).
        leaf_only: bool,
    },
    /// Invalidate everything cached under the address-space id in `operand`
    /// (`TLBI ASIDE1IS`).
    Asid {
        /// The `TLBI` operand: the id in bits 63:48.
        operand: u64,
    },
}

/// The operations that invalidate `ranges` in the address space `asid`.
///
/// Each range must be non-empty, page aligned at both ends, and inside the 39-bit space guest
/// tables translate; otherwise the whole call is [`Error::BadFlushRange`] for the first such
/// range and nothing is planned. If the ranges cover more than [`MAX_PAGE_FLUSHES`] pages in
/// total, the plan is a single address-space-wide invalidation.
pub(super) fn plan(asid: u16, ranges: &[FlushRange]) -> Result<Vec<FlushOp>, Error> {
    let asid = u64::from(asid) << ASID_SHIFT;
    let mut ops = Vec::with_capacity(ranges.len());
    let mut total = 0u64;
    for range in ranges {
        let bad = Error::BadFlushRange {
            va: range.va,
            size: range.size,
        };
        let inside = range
            .va
            .checked_add(range.size)
            .is_some_and(|end| end <= VA_SPACE_END);
        if range.size == 0
            || !range.va.is_multiple_of(PAGE_SIZE)
            || !range.size.is_multiple_of(PAGE_SIZE)
            || !inside
        {
            return Err(bad);
        }
        let count = range.size / PAGE_SIZE;
        total = total.saturating_add(count);
        ops.push(FlushOp::Pages {
            // The page number is below 2^27 (checked above), so it stays clear of the id.
            operand: asid | (range.va / PAGE_SIZE),
            count,
            leaf_only: range.leaf_only,
        });
    }
    if total > MAX_PAGE_FLUSHES {
        return Ok(vec![FlushOp::Asid { operand: asid }]);
    }
    Ok(ops)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn range(va: u64, pages: u64, leaf_only: bool) -> FlushRange {
        FlushRange {
            va,
            size: pages * PAGE_SIZE,
            leaf_only,
        }
    }

    #[test]
    fn each_range_becomes_one_run_of_pages_with_the_asid_in_the_top_bits() {
        // Operands written out from the layout: id at bit 48, page number in the low bits.
        assert_eq!(
            plan(
                0xbeef,
                &[range(0x8000_1000, 3, true), range(0x1000, 1, false)]
            ),
            Ok(vec![
                FlushOp::Pages {
                    operand: 0xbeef_0000_0008_0001,
                    count: 3,
                    leaf_only: true
                },
                FlushOp::Pages {
                    operand: 0xbeef_0000_0000_0001,
                    count: 1,
                    leaf_only: false
                },
            ])
        );
        assert_eq!(plan(7, &[]), Ok(vec![]));
    }

    #[test]
    fn the_last_page_of_the_address_space_keeps_clear_of_the_asid() {
        let last = VA_SPACE_END - PAGE_SIZE;
        assert_eq!(
            plan(0xffff, &[range(last, 1, true)]),
            Ok(vec![FlushOp::Pages {
                operand: 0xffff_0000_07ff_ffff,
                count: 1,
                leaf_only: true
            }])
        );
        // With no id the operand is the page number alone.
        assert_eq!(
            plan(0, &[range(last, 1, true)]),
            Ok(vec![FlushOp::Pages {
                operand: last / PAGE_SIZE,
                count: 1,
                leaf_only: true
            }])
        );
    }

    #[test]
    fn more_pages_than_the_limit_become_one_address_space_flush() {
        let asid = 0x12;
        let whole = FlushOp::Asid {
            operand: 0x0012_0000_0000_0000,
        };
        // Exactly at the limit: still page by page, in one range or split across two.
        assert_eq!(
            plan(asid, &[range(0, MAX_PAGE_FLUSHES, false)]),
            Ok(vec![FlushOp::Pages {
                operand: 0x0012_0000_0000_0000,
                count: MAX_PAGE_FLUSHES,
                leaf_only: false
            }])
        );
        let split = [
            range(0, MAX_PAGE_FLUSHES - 1, true),
            range(0x4000_0000, 1, true),
        ];
        assert_eq!(plan(asid, &split).map(|ops| ops.len()), Ok(2));
        // One page more, either way.
        assert_eq!(
            plan(asid, &[range(0, MAX_PAGE_FLUSHES + 1, true)]),
            Ok(vec![whole])
        );
        let split = [
            range(0, MAX_PAGE_FLUSHES, true),
            range(0x4000_0000, 1, true),
        ];
        assert_eq!(plan(asid, &split), Ok(vec![whole]));
    }

    #[test]
    fn ranges_are_checked_over_their_whole_extent() {
        let bad = |va, size| {
            let flush = FlushRange {
                va,
                size,
                leaf_only: true,
            };
            // A good range before the bad one changes nothing: the call plans nothing.
            assert_eq!(
                plan(1, &[range(0, 1, true), flush]),
                Err(Error::BadFlushRange { va, size }),
                "{va:#x} {size:#x}"
            );
        };
        bad(0x1000, 0);
        bad(0x1001, PAGE_SIZE);
        bad(0x1000, PAGE_SIZE + 1);
        // Ends one page past the address space; starts at its end; overflows.
        bad(VA_SPACE_END - PAGE_SIZE, 2 * PAGE_SIZE);
        bad(VA_SPACE_END, PAGE_SIZE);
        bad(u64::MAX - (PAGE_SIZE - 1), PAGE_SIZE);
        // Ending exactly at the end of the address space is fine.
        assert!(plan(1, &[range(VA_SPACE_END - 2 * PAGE_SIZE, 2, true)]).is_ok());
    }

    #[test]
    fn the_error_names_the_range() {
        assert_eq!(
            Error::BadFlushRange {
                va: 0x1001,
                size: 0x1000
            }
            .to_string(),
            "cannot flush 0x1001+0x1000: not a non-empty run of pages in the address space"
        );
    }
}
