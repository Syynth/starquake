//! The 3D class's report semaphore: what a write to `SET_REPORT_SEMAPHORE_D` asks the back to
//! do (gpu-protocol spec, "Releases and completion", "Report semaphores", decided by the owner
//! on 2026-11-19).
//!
//! A report of **no counter** (`REPORT` = `NONE`) releases the payload in
//! `SET_REPORT_SEMAPHORE_C` at the address in `SET_REPORT_SEMAPHORE_A/B`, as a fenced release
//! like every other ([`FencedAction::Release`], `back::fence`). The native SDK ends a queue's
//! work with one and waits for the word to change, so without it a title stops after its first
//! submissions. A counter report (occlusion counts, primitive counts, timestamps) writes
//! nothing, as the spec has it until a title needs one (below, "Divergences"). The compute
//! class's report semaphore (`clc7c0.h`'s `SET_REPORT_SEMAPHORE_A`–`D`), which the spec's
//! paragraph also covers, is not handled here: it is issue #458.
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
//!   `counter_type = QueryType::Payload; payload = 1U`). The spec writes nothing until a title
//!   needs one, then Eden-NX2's stand-in of 1; until then such a write is logged once per run
//!   by the executor and nothing is written, so a title that waits on one still stops
//!   (`tests::a_counter_report_is_not_a_release`).
//! - **A release whose address is not mapped is a failed release**, found when it is performed
//!   and logged and counted there (`back::fence`, "A failed release is reported and skipped").
//!   Eden-NX2 drops it when it is asked for (`GpuToCpuAddress` gives nothing), without a word.
//!   The native SDK does write one to address 0, and may at every flush: a failure of a
//!   channel's second and later releases to address 0 is logged at debug level, not as a
//!   warning (`back::exec`, "Report semaphores").
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
use crate::classes::{Method, clc7c0, clc797};
use crate::field::Field;
use crate::ring::{Release, ReleaseOp};
use crate::sema::{PayloadSize, ReleaseForm};

/// Bit position of `SET_REPORT_SEMAPHORE_A`'s `OFFSET_UPPER` in the semaphore's GPU address.
const ADDRESS_UPPER_SHIFT: u32 = 32;

/// One class's report semaphore: its four registers and the fields and values [`decide_in`]
/// reads (module docs).
struct Registers {
    /// `SET_REPORT_SEMAPHORE_A` and its `OFFSET_UPPER`.
    upper: (&'static Method, Field),
    /// `SET_REPORT_SEMAPHORE_B` and its `OFFSET_LOWER`.
    lower: (&'static Method, Field),
    /// `SET_REPORT_SEMAPHORE_C` and its `PAYLOAD`.
    payload: (&'static Method, Field),
    /// `SET_REPORT_SEMAPHORE_D`, the query word.
    query: &'static Method,
    /// The query word's `OPERATION`.
    operation: Field,
    /// The two `OPERATION` values that release: `RELEASE` and `REPORT_ONLY`.
    releasing: [u32; 2],
    /// The query word's `REPORT` and its `NONE` value, for a class that has one.
    report: Option<(Field, u32)>,
    /// The query word's `STRUCTURE_SIZE` and its `ONE_WORD` value.
    structure_size: (Field, u32),
}

/// The 3D class's (`clc797.h`).
const THREE_D: Registers = Registers {
    upper: (
        &clc797::SET_REPORT_SEMAPHORE_A,
        clc797::SET_REPORT_SEMAPHORE_A_OFFSET_UPPER,
    ),
    lower: (
        &clc797::SET_REPORT_SEMAPHORE_B,
        clc797::SET_REPORT_SEMAPHORE_B_OFFSET_LOWER,
    ),
    payload: (
        &clc797::SET_REPORT_SEMAPHORE_C,
        clc797::SET_REPORT_SEMAPHORE_C_PAYLOAD,
    ),
    query: &clc797::SET_REPORT_SEMAPHORE_D,
    operation: clc797::SET_REPORT_SEMAPHORE_D_OPERATION,
    releasing: [
        clc797::SET_REPORT_SEMAPHORE_D_OPERATION_RELEASE,
        clc797::SET_REPORT_SEMAPHORE_D_OPERATION_REPORT_ONLY,
    ],
    report: Some((
        clc797::SET_REPORT_SEMAPHORE_D_REPORT,
        clc797::SET_REPORT_SEMAPHORE_D_REPORT_NONE,
    )),
    structure_size: (
        clc797::SET_REPORT_SEMAPHORE_D_STRUCTURE_SIZE,
        clc797::SET_REPORT_SEMAPHORE_D_STRUCTURE_SIZE_ONE_WORD,
    ),
};

/// The compute class's (`clc7c0.h`; module docs, "The compute class"). Its query word has no
/// `REPORT`, and the header names no `REPORT_ONLY` for it: the value is
/// `REPORT_SEMAPHORE_EXECUTE`'s, which Eden-NX2 releases on (a **guess** that the two encodings
/// agree).
const COMPUTE: Registers = Registers {
    upper: (
        &clc7c0::SET_REPORT_SEMAPHORE_A,
        clc7c0::SET_REPORT_SEMAPHORE_A_OFFSET_UPPER,
    ),
    lower: (
        &clc7c0::SET_REPORT_SEMAPHORE_B,
        clc7c0::SET_REPORT_SEMAPHORE_B_OFFSET_LOWER,
    ),
    payload: (
        &clc7c0::SET_REPORT_SEMAPHORE_C,
        clc7c0::SET_REPORT_SEMAPHORE_C_PAYLOAD,
    ),
    query: &clc7c0::SET_REPORT_SEMAPHORE_D,
    operation: clc7c0::SET_REPORT_SEMAPHORE_D_OPERATION,
    releasing: [
        clc7c0::SET_REPORT_SEMAPHORE_D_OPERATION_RELEASE,
        clc7c0::REPORT_SEMAPHORE_EXECUTE_OPERATION_REPORT_ONLY,
    ],
    report: None,
    structure_size: (
        clc7c0::SET_REPORT_SEMAPHORE_D_STRUCTURE_SIZE,
        clc7c0::SET_REPORT_SEMAPHORE_D_STRUCTURE_SIZE_ONE_WORD,
    ),
};

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
    decide_in(mirror, &THREE_D)
}

/// What the report semaphore registers of `mirror`, a compute engine's, ask for now that
/// `SET_REPORT_SEMAPHORE_D` was written (module docs, "The compute class"). Never a
/// [`ReportAction::Counter`]: the compute class's query word has no `REPORT` field.
///
/// # Errors
///
/// A [`StateError`] if `mirror`'s register file does not hold the four registers. Every class's
/// file does, so none is reachable through the executor, which asks only for a compute engine.
pub fn decide_compute(mirror: &Mirror) -> Result<ReportAction, StateError> {
    decide_in(mirror, &COMPUTE)
}

/// The decision of [`decide`] and [`decide_compute`], over one class's `registers`.
fn decide_in(mirror: &Mirror, registers: &Registers) -> Result<ReportAction, StateError> {
    let query = mirror.reg(registers.query, &[])?;
    if !registers
        .releasing
        .contains(&query.field(registers.operation)?)
    {
        return Ok(ReportAction::Nothing);
    }
    if let Some((field, none)) = registers.report {
        let report = query.field(field)?;
        if report != none {
            return Ok(ReportAction::Counter { report });
        }
    }
    let (size, one) = registers.structure_size;
    let one_word = query.field(size)? == one;
    let read = |(method, field): (&'static Method, Field)| mirror.reg(method, &[])?.field(field);
    let upper = read(registers.upper)?;
    let lower = read(registers.lower)?;
    let payload = read(registers.payload)?;
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
    use crate::classes::{clc7c0, clc797};
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

    /// The compute class's mirror with the same register values as [`mirror`]'s.
    fn compute_mirror(query: u32) -> Mirror {
        let mut mirror = Mirror::new(ChannelId(3), ClassId(clc7c0::CLASS));
        for (method, value) in [
            (&clc7c0::SET_REPORT_SEMAPHORE_A, UPPER),
            (&clc7c0::SET_REPORT_SEMAPHORE_B, LOWER),
            (&clc7c0::SET_REPORT_SEMAPHORE_C, PAYLOAD),
            (&clc7c0::SET_REPORT_SEMAPHORE_D, query),
        ] {
            mirror.write(method.method(), value).unwrap();
        }
        mirror
    }

    /// A compute `SET_REPORT_SEMAPHORE_D` word with these `OPERATION` and `STRUCTURE_SIZE`.
    fn compute_query(operation: u32, size: u32) -> u32 {
        [
            (clc7c0::SET_REPORT_SEMAPHORE_D_OPERATION, operation),
            (clc7c0::SET_REPORT_SEMAPHORE_D_STRUCTURE_SIZE, size),
        ]
        .iter()
        .fold(0, |word, &(field, value)| {
            field.insert(word, value).unwrap()
        })
    }

    #[test]
    fn every_compute_operation_and_size_gives_the_tables_action() {
        // The whole input space of the two fields read: 4 operations, 2 sizes, each with every
        // other bit of the word clear and then set (the reduction fields and bits 27:23, where
        // the 3D class has `REPORT`, among them).
        let operations = [
            (clc7c0::SET_REPORT_SEMAPHORE_D_OPERATION_RELEASE, true),
            (clc7c0::REPORT_SEMAPHORE_EXECUTE_OPERATION_ACQUIRE, false),
            (clc7c0::REPORT_SEMAPHORE_EXECUTE_OPERATION_REPORT_ONLY, true),
            (clc7c0::SET_REPORT_SEMAPHORE_D_OPERATION_TRAP, false),
        ];
        let sizes = [
            (
                clc7c0::SET_REPORT_SEMAPHORE_D_STRUCTURE_SIZE_FOUR_WORDS,
                true,
            ),
            (
                clc7c0::SET_REPORT_SEMAPHORE_D_STRUCTURE_SIZE_ONE_WORD,
                false,
            ),
        ];
        let ignored = !compute_query(
            clc7c0::SET_REPORT_SEMAPHORE_D_OPERATION.get(u32::MAX),
            clc7c0::SET_REPORT_SEMAPHORE_D_STRUCTURE_SIZE.get(u32::MAX),
        );
        let mut seen = 0;
        for (operation, releases) in operations {
            for (size, timestamp) in sizes {
                for other in [0, ignored] {
                    let word = compute_query(operation, size) | other;
                    let expected = if releases {
                        release(timestamp)
                    } else {
                        ReportAction::Nothing
                    };
                    assert_eq!(
                        decide_compute(&compute_mirror(word)),
                        Ok(expected),
                        "{word:#x}"
                    );
                    seen += 1;
                }
            }
        }
        assert_eq!(seen, 4 * 2 * 2);
    }

    #[test]
    fn compute_words_built_by_hand_give_eden_nx2s_answer() {
        // Hand-built, not through `compute_query`, each against Eden-NX2's
        // `ProcessNativeReportSemaphore`: the operation in bits 1:0 (0 and 2 release), the
        // one-word structure in bit 28.
        let cases = [
            (0x1000_0000, release(false)),
            (0x0000_0000, release(true)),
            (0x1000_0002, release(false)),
            (0x0000_0002, release(true)),
            (0x1000_0001, ReportAction::Nothing),
            (0x1000_0003, ReportAction::Nothing),
            // Bits 27:23 (the 3D class's `REPORT`) all set: the compute class has no counter.
            (0x1f80_0000, release(false)),
            // FLUSH_DISABLE (2), CONDITIONAL_TRAP (19) and AWAKEN_ENABLE (20) set.
            (0x1018_0004, release(false)),
            // REDUCTION_ENABLE (3) with RED_MIN (11:9) and SIGNED_32 (18:17): the payload is
            // still written (module docs, "Divergences", the reduction).
            (0x1002_0208, release(false)),
        ];
        for (word, expected) in cases {
            assert_eq!(
                decide_compute(&compute_mirror(word)),
                Ok(expected),
                "{word:#x}"
            );
        }
    }

    #[test]
    fn the_compute_address_takes_eight_upper_bits_and_all_lower_ones() {
        let word = 0x1000_0000;
        let address_of = |upper: u32, lower: u32| {
            let mut mirror = compute_mirror(word);
            mirror
                .write(clc7c0::SET_REPORT_SEMAPHORE_A.method(), upper)
                .unwrap();
            mirror
                .write(clc7c0::SET_REPORT_SEMAPHORE_B.method(), lower)
                .unwrap();
            match decide_compute(&mirror) {
                Ok(ReportAction::Release(release)) => Some(release.address),
                _ => None,
            }
        };
        assert_eq!(address_of(u32::MAX, 0), Some(0xff_0000_0000));
        assert_eq!(address_of(0, u32::MAX), Some(0xffff_ffff));
        // Bit 8 of `SET_REPORT_SEMAPHORE_A` is dropped; Eden-NX2's 25-bit mask would keep it
        // as address bit 40 (module docs, "Divergences").
        assert_eq!(address_of(0x100, 0), Some(0));
        assert_eq!(address_of(0x1ff_ffff, 0), Some(0xff_0000_0000));
    }

    #[test]
    fn the_compute_registers_are_the_headers() {
        // Literals from `clc7c0.h` (mesa-nouveau @ 52238762), and the operation value 2 that
        // Eden-NX2's `ProcessNativeReportSemaphore` releases on (`operation != 0 && operation
        // != 2`), written out so a wrong generated constant fails here.
        assert_eq!(
            [
                clc7c0::SET_REPORT_SEMAPHORE_A.byte_address,
                clc7c0::SET_REPORT_SEMAPHORE_B.byte_address,
                clc7c0::SET_REPORT_SEMAPHORE_C.byte_address,
                clc7c0::SET_REPORT_SEMAPHORE_D.byte_address,
            ],
            [0x1b00, 0x1b04, 0x1b08, 0x1b0c]
        );
        assert_eq!(
            clc7c0::SET_REPORT_SEMAPHORE_A_OFFSET_UPPER,
            Field::new(7, 0)
        );
        assert_eq!(
            clc7c0::SET_REPORT_SEMAPHORE_B_OFFSET_LOWER,
            Field::new(31, 0)
        );
        assert_eq!(clc7c0::SET_REPORT_SEMAPHORE_C_PAYLOAD, Field::new(31, 0));
        assert_eq!(clc7c0::SET_REPORT_SEMAPHORE_D_OPERATION, Field::new(1, 0));
        assert_eq!(
            clc7c0::SET_REPORT_SEMAPHORE_D_STRUCTURE_SIZE,
            Field::new(28, 28)
        );
        assert_eq!(
            [
                clc7c0::SET_REPORT_SEMAPHORE_D_OPERATION_RELEASE,
                clc7c0::REPORT_SEMAPHORE_EXECUTE_OPERATION_ACQUIRE,
                clc7c0::REPORT_SEMAPHORE_EXECUTE_OPERATION_REPORT_ONLY,
                clc7c0::SET_REPORT_SEMAPHORE_D_OPERATION_TRAP,
                clc7c0::SET_REPORT_SEMAPHORE_D_STRUCTURE_SIZE_ONE_WORD,
            ],
            [0, 1, 2, 3, 1]
        );
    }
}
