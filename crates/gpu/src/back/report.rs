//! The 3D class's report semaphore: what a write to `SET_REPORT_SEMAPHORE_D` asks the back to
//! do (gpu-protocol spec, "Releases and completion", and "Deferred, not ruled out", "Queries":
//! "report semaphores (`SET_REPORT_SEMAPHORE_A`–`D`) first write the values the front can know
//! (payloads, a timestamp from guest time); host queries plug into the fenced completion path
//! later").
//!
//! This is the first half of that sentence, and only it: a report of **no counter**
//! (`REPORT` = `NONE`) releases the payload in `SET_REPORT_SEMAPHORE_C` at the address in
//! `SET_REPORT_SEMAPHORE_A/B`, as a fenced release like every other ([`FencedAction::Release`],
//! `back::fence`). The native SDK ends a queue's work with one and waits for the word to
//! change, so without it a title stops after its first submissions. Counter reports (occlusion
//! counts, primitive counts, timestamps) are the second half and are still deferred. The
//! compute class's report semaphore (`clc7c0.h`'s `SET_REPORT_SEMAPHORE_A`–`D`) is not handled
//! here: it is issue #458.
//!
//! Method numbers and fields are `clc797.h`'s (`mesa-nouveau/src/nouveau/headers/nvidia/classes/`
//! @ 52238762, MIT), read through the generated [`clc797`]. The behaviour is that of Eden-NX2's
//! `Maxwell3D::ProcessQueryGet` (`eden-nx2/src/video_core/engines/maxwell_3d.cpp` @ 50fa86bb,
//! GPL-3.0-or-later) and the `QueryType::Payload` case of `QueryCacheBase::CounterReport`
//! (`eden-nx2/src/video_core/query_cache/query_cache.h` @ 50fa86bb). Paths are relative to the
//! reference folder (`tools/refs-path.sh`).
//!
//! # What a write to `SET_REPORT_SEMAPHORE_D` does
//!
//! [`decide`] reads the four registers from the 3D engine's mirror, after the mirror stored the
//! word, and answers with a [`ReportAction`]:
//!
//! | `OPERATION` (1:0) | `REPORT` (27:23) | Action |
//! |---|---|---|
//! | `RELEASE` (0) or `REPORT_ONLY` (2) | `NONE` (0) | [`ReportAction::Release`] of the payload |
//! | `RELEASE` (0) or `REPORT_ONLY` (2) | any other | [`ReportAction::Counter`]: nothing is written |
//! | `ACQUIRE` (1) or `TRAP` (3) | any | [`ReportAction::Nothing`] |
//!
//! `ProcessQueryGet` calls `rasterizer->Query` for `Operation::Release` and
//! `Operation::ReportOnly` and does nothing for the other two (`UNIMPLEMENTED_MSG`), and its
//! `report` value 0 is `QueryType::Payload`. The two calls differ in one flag: only a
//! `Release` with `short_query` set is marked `IsAFence` (below, "Divergences").
//!
//! The release:
//!
//! - **address**: `SET_REPORT_SEMAPHORE_A`'s `OFFSET_UPPER` (7:0) as bits 39:32 over
//!   `SET_REPORT_SEMAPHORE_B`'s `OFFSET_LOWER` (a 40-bit address; Eden's
//!   `report_semaphore.Address()` keeps more, below, "Divergences");
//! - **payload**: `SET_REPORT_SEMAPHORE_C`'s `PAYLOAD`, 32 bits;
//! - **form**: with `STRUCTURE_SIZE` (28) `ONE_WORD` (Eden's `short_query`), the 4 payload
//!   bytes; with `FOUR_WORDS`, the 16-byte structure, the payload widened to 64 bits and then
//!   the timestamp ([`crate::sema::ReleaseForm`]; row 52). `CounterReport` writes the same two
//!   shapes (`HasTimeout` is set when `short_query` is 0). The timestamp is the fence path's
//!   ([`crate::back::fence::FenceClock::gpu_timestamp`]; below, "Divergences").
//!
//! The executor (`back::exec`) attaches it for the channel's address space, after drawing the
//! channel's deferred instances, as it does for a `Release` entry.
//!
//! # Divergences from Eden-NX2
//!
//! - **Every payload release is fenced** (owner's decision 2, "Releases happen after the work
//!   they follow"). Eden-NX2 fences only a one-word `RELEASE`: `ProcessQueryGet` sets
//!   `IsAFence` for `Operation::Release` when `short_query` is set, and never for
//!   `Operation::ReportOnly`, and `CounterReport` then calls `SignalFence`. A four-word
//!   `RELEASE` and every `REPORT_ONLY`, unless its GPU accuracy is High, are written at the
//!   write itself (`!Settings::IsGPULevelHigh() && counter_type == QueryType::Payload`). Here
//!   every form of both operations waits for the host work before it
//!   (`back::exec::tests::a_report_semaphore_release_waits_for_the_work_before_it`), and is
//!   written at once when there is none
//!   (`back::exec::tests::a_one_word_payload_report_writes_four_bytes`).
//! - **The address keeps 8 bits of `OFFSET_UPPER`.** `clc797.h` gives the field bits 7:0, so a
//!   value above them is dropped and the address has 40 bits. Eden-NX2 reads `address_high` as
//!   a whole `u32` and shifts all 32 bits up (`Address()`,
//!   `eden-nx2/src/video_core/engines/maxwell_3d.h` @ 50fa86bb).
//!   `tests::the_address_takes_eight_upper_bits_and_all_lower_ones`.
//! - **The four-word form's timestamp is the host's monotonic clock**, in nanoseconds, as for
//!   every 16-byte release (the production [`crate::back::fence::FenceClock`], `crate::wiring`).
//!   The owner's ruling of 2026-11-19 (`docs/decision-log.md`, ruling 29) makes it guest time
//!   from the kernel's clock; issue #435 changes it for every release, this one included.
//!   Eden-NX2 writes `gpu.GetTicks()`. Not pinned here: the executor's tests use the fence
//!   path's fake clock (`back::exec::tests::a_four_word_payload_report_writes_the_payload_and_the_timestamp`).
//! - **A counter report writes nothing.** Eden-NX2 answers the kinds its backend counts from
//!   host queries, and every other kind as a payload of 1 (`streamer == nullptr`:
//!   `counter_type = QueryType::Payload; payload = 1U`). Which of those this project wants is
//!   the deferred half of the spec's sentence; until then such a write is logged once per run
//!   by the executor and nothing is written, so a title that waits on one still stops
//!   (`tests::a_counter_report_is_not_a_release`).
//! - **A release whose address is not mapped is a failed release**, found when it is performed
//!   and logged and counted there (`back::fence`, "A failed release is reported and skipped").
//!   Eden-NX2 drops it when it is asked for (`GpuToCpuAddress` gives nothing), without a word.
//!   The native SDK does write one to address 0.
//! - **The deferred instances are drawn first**, as before a `Release` entry (`back::exec`,
//!   "Deferred instanced draws"); Eden-NX2 has drawn them by the end of the macro.
//!
//! # Guesses and open items
//!
//! - `REDUCTION_ENABLE`, `REDUCTION_OP` and `REDUCTION_FORMAT` are ignored, as in Eden-NX2,
//!   which reads none of them: the payload is written. A **guess** that no title sets
//!   `REDUCTION_ENABLE` on a payload report; a trace of one that does would say whether the
//!   release should be a reduction ([`crate::sema::Reduction`]).
//! - `RELEASE` (4), `ACQUIRE` (8), `PIPELINE_LOCATION`, `COMPARISON`, `AWAKEN_ENABLE`,
//!   `SUB_REPORT`, `REPORT_DWORD_NUMBER`, `FLUSH_DISABLE` and `CONDITIONAL_TRAP` (19) are
//!   ignored, as in Eden-NX2 for a payload (`sub_report` only selects among counters; Eden-NX2's
//!   `query` union has no field at bit 19).
//! - `REPORT_SEMAPHORE_EXECUTE` and the 64-bit `SET_REPORT_SEMAPHORE_PAYLOAD_LOWER/UPPER` and
//!   `_ADDRESS_LOWER/UPPER` registers `clc797.h` adds are not acted on: neither reference acts
//!   on them, and no title has been seen to write them.

use crate::back::state::{Mirror, StateError};
use crate::classes::clc797;
use crate::ring::{Release, ReleaseOp};
use crate::sema::{PayloadSize, ReleaseForm};

/// Bit position of `SET_REPORT_SEMAPHORE_A`'s `OFFSET_UPPER` in the semaphore's GPU address.
const ADDRESS_UPPER_SHIFT: u32 = 32;

/// What a write to `SET_REPORT_SEMAPHORE_D` asks for (module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReportAction {
    /// Release the payload: a fenced release of the channel's address space.
    Release(Release),
    /// A counter report, which is not written yet (module docs, "Divergences").
    Counter {
        /// The `REPORT` field.
        report: u32,
    },
    /// An `ACQUIRE` or a `TRAP`: nothing, as in Eden-NX2.
    Nothing,
}

/// What the report semaphore registers of `mirror`, a 3D engine's, ask for now that
/// `SET_REPORT_SEMAPHORE_D` was written (module docs).
///
/// # Errors
///
/// A [`StateError`] if `mirror`'s register file does not hold the four registers. Every class's
/// file does, so none is reachable through the executor, which asks only for a 3D engine.
pub fn decide(mirror: &Mirror) -> Result<ReportAction, StateError> {
    let query = mirror.reg(&clc797::SET_REPORT_SEMAPHORE_D, &[])?;
    match query.field(clc797::SET_REPORT_SEMAPHORE_D_OPERATION)? {
        clc797::SET_REPORT_SEMAPHORE_D_OPERATION_RELEASE
        | clc797::SET_REPORT_SEMAPHORE_D_OPERATION_REPORT_ONLY => {}
        _ => return Ok(ReportAction::Nothing),
    }
    let report = query.field(clc797::SET_REPORT_SEMAPHORE_D_REPORT)?;
    if report != clc797::SET_REPORT_SEMAPHORE_D_REPORT_NONE {
        return Ok(ReportAction::Counter { report });
    }
    let one_word = query.field(clc797::SET_REPORT_SEMAPHORE_D_STRUCTURE_SIZE)?
        == clc797::SET_REPORT_SEMAPHORE_D_STRUCTURE_SIZE_ONE_WORD;
    let upper = mirror
        .reg(&clc797::SET_REPORT_SEMAPHORE_A, &[])?
        .field(clc797::SET_REPORT_SEMAPHORE_A_OFFSET_UPPER)?;
    let lower = mirror
        .reg(&clc797::SET_REPORT_SEMAPHORE_B, &[])?
        .field(clc797::SET_REPORT_SEMAPHORE_B_OFFSET_LOWER)?;
    let payload = mirror
        .reg(&clc797::SET_REPORT_SEMAPHORE_C, &[])?
        .field(clc797::SET_REPORT_SEMAPHORE_C_PAYLOAD)?;
    Ok(ReportAction::Release(Release {
        address: (u64::from(upper) << ADDRESS_UPPER_SHIFT) | u64::from(lower),
        payload: u64::from(payload),
        form: ReleaseForm {
            size: PayloadSize::Four,
            timestamp: !one_word,
        },
        op: ReleaseOp::Write,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::classes::clc797;
    use crate::field::Field;
    use crate::interface::ChannelId;
    use crate::ring::ClassId;

    /// Distinct values for every register, the upper address with bits above its field set.
    const UPPER: u32 = 0xffff_ff29;
    const LOWER: u32 = 0x0622_a004;
    const PAYLOAD: u32 = 0x8000_0031;
    const ADDRESS: u64 = 0x29_0622_a004;

    fn mirror(query: u32) -> Mirror {
        let mut mirror = Mirror::new(ChannelId(3), ClassId(clc797::CLASS));
        for (method, value) in [
            (&clc797::SET_REPORT_SEMAPHORE_A, UPPER),
            (&clc797::SET_REPORT_SEMAPHORE_B, LOWER),
            (&clc797::SET_REPORT_SEMAPHORE_C, PAYLOAD),
            (&clc797::SET_REPORT_SEMAPHORE_D, query),
        ] {
            mirror.write(method.method(), value).unwrap();
        }
        mirror
    }

    fn query(operation: u32, report: u32, size: u32) -> u32 {
        [
            (clc797::SET_REPORT_SEMAPHORE_D_OPERATION, operation),
            (clc797::SET_REPORT_SEMAPHORE_D_REPORT, report),
            (clc797::SET_REPORT_SEMAPHORE_D_STRUCTURE_SIZE, size),
        ]
        .iter()
        .fold(0, |word, &(field, value)| {
            field.insert(word, value).unwrap()
        })
    }

    fn release(timestamp: bool) -> ReportAction {
        ReportAction::Release(Release {
            address: ADDRESS,
            payload: u64::from(PAYLOAD),
            form: ReleaseForm {
                size: PayloadSize::Four,
                timestamp,
            },
            op: ReleaseOp::Write,
        })
    }

    /// Every bit of `SET_REPORT_SEMAPHORE_D` outside the three fields [`decide`] reads.
    fn ignored_bits() -> u32 {
        !query(
            clc797::SET_REPORT_SEMAPHORE_D_OPERATION.get(u32::MAX),
            clc797::SET_REPORT_SEMAPHORE_D_REPORT.get(u32::MAX),
            clc797::SET_REPORT_SEMAPHORE_D_STRUCTURE_SIZE.get(u32::MAX),
        )
    }

    #[test]
    fn every_operation_report_and_size_gives_the_tables_action() {
        // The whole input space of the three fields read: 4 operations, 32 reports, 2 sizes,
        // each with every other bit of the word clear and then set.
        let operations = [
            (clc797::SET_REPORT_SEMAPHORE_D_OPERATION_RELEASE, true),
            (clc797::SET_REPORT_SEMAPHORE_D_OPERATION_ACQUIRE, false),
            (clc797::SET_REPORT_SEMAPHORE_D_OPERATION_REPORT_ONLY, true),
            (clc797::SET_REPORT_SEMAPHORE_D_OPERATION_TRAP, false),
        ];
        let sizes = [
            (
                clc797::SET_REPORT_SEMAPHORE_D_STRUCTURE_SIZE_FOUR_WORDS,
                true,
            ),
            (
                clc797::SET_REPORT_SEMAPHORE_D_STRUCTURE_SIZE_ONE_WORD,
                false,
            ),
        ];
        let reports = clc797::SET_REPORT_SEMAPHORE_D_REPORT.get(u32::MAX);
        assert_eq!(reports, 31);
        let mut seen = 0;
        for (operation, reports_something) in operations {
            for report in 0..=reports {
                for (size, timestamp) in sizes {
                    for other in [0, ignored_bits()] {
                        let word = query(operation, report, size) | other;
                        let expected = if !reports_something {
                            ReportAction::Nothing
                        } else if report == clc797::SET_REPORT_SEMAPHORE_D_REPORT_NONE {
                            release(timestamp)
                        } else {
                            ReportAction::Counter { report }
                        };
                        assert_eq!(decide(&mirror(word)), Ok(expected), "{word:#x}");
                        seen += 1;
                    }
                }
            }
        }
        assert_eq!(seen, 4 * 32 * 2 * 2);
    }

    #[test]
    fn the_word_the_native_sdk_writes_is_a_one_word_release() {
        // Hand-built, not through `query`: RELEASE, RELEASE after writes (bit 4), pipeline
        // location ALL (15:12), report NONE, ONE_WORD (bit 28).
        assert_eq!(decide(&mirror(0x1000_f010)), Ok(release(false)));
        // The same with FOUR_WORDS: the 16-byte structure.
        assert_eq!(decide(&mirror(0x0000_f010)), Ok(release(true)));
    }

    #[test]
    fn a_counter_report_is_not_a_release() {
        // The first and last counter the header names, and the edge next to NONE.
        for report in [
            1,
            clc797::SET_REPORT_SEMAPHORE_D_REPORT_ZPASS_PIXEL_CNT,
            clc797::SET_REPORT_SEMAPHORE_D_REPORT_TS_PRIMITIVES_GENERATED,
        ] {
            let word = query(
                clc797::SET_REPORT_SEMAPHORE_D_OPERATION_RELEASE,
                report,
                clc797::SET_REPORT_SEMAPHORE_D_STRUCTURE_SIZE_ONE_WORD,
            );
            assert_eq!(decide(&mirror(word)), Ok(ReportAction::Counter { report }));
        }
    }

    #[test]
    fn the_address_takes_eight_upper_bits_and_all_lower_ones() {
        // Each address field alone, at its top value: neither spills into the other.
        let word = 0x1000_f010;
        let address_of = |upper: u32, lower: u32| {
            let mut mirror = mirror(word);
            mirror
                .write(clc797::SET_REPORT_SEMAPHORE_A.method(), upper)
                .unwrap();
            mirror
                .write(clc797::SET_REPORT_SEMAPHORE_B.method(), lower)
                .unwrap();
            match decide(&mirror) {
                Ok(ReportAction::Release(release)) => Some(release.address),
                _ => None,
            }
        };
        assert_eq!(address_of(u32::MAX, 0), Some(0xff_0000_0000));
        assert_eq!(address_of(0, u32::MAX), Some(0xffff_ffff));
        assert_eq!(address_of(0x100, 0), Some(0));
        assert_eq!(
            Field::new(7, 0),
            clc797::SET_REPORT_SEMAPHORE_A_OFFSET_UPPER
        );
    }
}
