//! Compute dispatch through QMD 3.0 (gpu-protocol spec, "Compute: QMD 3.0" and "Releases and
//! completion"; slice G12; Switch 2 scoping rows 53, 60, 61, 62 and 84).
//!
//! NVN2 dispatches compute work through queue metadata (QMD) structures of 64 words in GPU
//! memory, and schedules them with the compute class's PCAS methods (row 53). This module owns:
//!
//! - **Decoding** ([`Qmd::decode`]): the version check, the [`Dispatch`] description (program,
//!   grid, block, shared and local memory, the eight constant buffers), the three release slots
//!   as [`Release`]s, and the dependent QMD.
//! - **The scheduler** ([`QmdTable`]): one table for all channels (row 84), keyed by QMD
//!   address, with each QMD's loaded, scheduled and completed flags, its dependence counter and
//!   the adjustments deferred until it is loaded ([`QmdStatus`]). [`QmdTable::signal`] applies
//!   one [`PcasAction`] and then every dependent action it leads to.
//! - **The compute methods that drive it** ([`QmdTable::compute_method`]): the last
//!   `LOAD_INLINE_QMD_DATA` word writes the inline QMD to memory and may schedule it, and
//!   `SEND_SIGNALING_PCAS2_B` signals the QMD that `SEND_PCAS_A` names.
//!
//! What a run produces goes to a [`QmdSink`], in this order: the [`Dispatch`] (unless the grid
//! is empty), each enabled release slot in slot order (words 24, 28, 52), then
//! [`QmdSink::completed`]. Only then is the QMD's dependent action applied, so its dispatch comes
//! after. The sink's releases are to be performed as fenced releases, after the host GPU has
//! finished the dispatch (gpu-protocol decision 2; the fence path is slice G15).
//!
//! # Sources
//!
//! Paths are relative to the reference folder (`tools/refs-path.sh`), at the commits pinned in
//! `docs/references.md`.
//!
//! - Every QMD field is read through the generated constants of [`crate::classes::clc7c0qmd`],
//!   which carry the bit ranges of `mesa-nouveau/src/nouveau/headers/nvidia/classes/clc7c0qmd.h`
//!   @ 52238762 (Copyright (c) 2020, NVIDIA CORPORATION, MIT licence). The compute methods are
//!   those of [`crate::classes::clc7c0`] (`clc7c0.h`, same folder).
//! - The scheduler is a port of `NativeQmdScheduler::Signal` in
//!   `eden-nx2/src/video_core/native_qmd.h` @ 50fa86bb (Copyright 2026 Eden Emulator Project,
//!   GPL-3.0-or-later), and the method handling of `KeplerCompute::ProcessNativeMethod` and
//!   `DispatchNativeCompute` in `eden-nx2/src/video_core/engines/kepler_compute.cpp` @ 50fa86bb.
//!
//! The rules it mirrors, with the reference's exact check:
//!
//! - A QMD is version 3.0 and not a queue: `(qmd[18] & 0xff) != 0x30 || (qmd[4] & 0x100)` is
//!   refused (`native_qmd.h` lines 41 and 74), checked when the QMD is loaded and again when it
//!   runs. Here the version ([`QmdError::Version`]) is checked before the queue flag
//!   ([`QmdError::Queue`]); Eden-NX2 reports both as one error.
//! - PCAS actions 2 (`SCHEDULE`) and 3 (`INVALIDATE_COPY_SCHEDULE`) schedule; 3, or 2 on a QMD
//!   not yet loaded, (re)loads it from memory, setting its counter to the QMD's dependence
//!   counter (bits 157:142) plus the deferred adjustments. Actions 7 (`DECREMENT_DEPENDENCE`) and
//!   12 (`INCREMENT_DEPENDENCE`) move the counter by `OFFSET_MINUS_ONE + 1`, or defer the change
//!   while the QMD is not loaded or has completed. Actions 0 and 1 do nothing (`kepler_compute.cpp`
//!   returns before calling the scheduler); every other action is [`QmdError::PcasAction`].
//! - A QMD runs when it is loaded, scheduled, not completed and its counter is not above zero;
//!   a counter below zero at that point is [`QmdError::DependenceUnderflow`] (`native_qmd.h`
//!   line 69).
//! - The table holds at most [`MAX_QMDS`] (4,096) QMDs: a request for a QMD not in a full table
//!   is [`QmdError::TableFull`] (line 35). Within [`QmdTable::signal`] it is checked before
//!   anything else about the request. [`QmdTable::compute_method`] checks, in order: the
//!   register file ([`QmdError::Registers`]), then the `SEND_SIGNALING_PCAS2_B` action
//!   ([`QmdError::PcasAction`]), and only then signals, so the table limit comes after the
//!   action there; Eden-NX2 checks the limit (line 35) before the action (line 66). This is a
//!   listed divergence below.
//! - One [`QmdTable::signal`] handles at most [`MAX_CHAIN`] (256) requests, the first and its
//!   dependents: a run whose dependent would be request 257 is [`QmdError::ChainTooLong`]
//!   (line 108).
//! - A QMD that runs is read again from memory first, because NVN2 patches its release payloads
//!   while it waits (the comment above line 72). Its release slots are all checked before any
//!   is made visible (lines 75 to 86).
//! - A grid with a zero dimension (`!(qmd[12] && (qmd[13] & 0xffff) && (qmd[14] & 0xffff))`,
//!   line 89) completes and releases without a dispatch ([`Outcome::EmptyGrid`]). A block with a
//!   zero dimension is refused as `DispatchNativeCompute` refuses it ([`Outcome::EmptyBlock`]),
//!   without reaching the sink. A program the sink rejects ([`Program::Rejected`]) still
//!   completes and releases (row 61; the comment at lines 87–88, and line 89).
//! - The dependent QMD: enable bit 512 with action 1 (`QMD_SCHEDULE`), 3
//!   (`QMD_INVALIDATE_COPY_SCHEDULE`) or 4 (`QMD_DECREMENT_DEPENDENCE`) becomes PCAS action 2, 3
//!   or 7 with a count of one at `DEPENDENT_QMD0_POINTER << 8` (lines 105 to 110). Another
//!   action is [`QmdError::DependentAction`], the chain limit is checked next, and then a second
//!   dependent QMD (bit 517) is [`QmdError::SecondDependent`] (line 111). All three are reported
//!   after the QMD has completed and released, and the dependent is then not applied, as in
//!   Eden-NX2.
//! - The inline QMD: on method 0x107 (`LOAD_INLINE_QMD_DATA(63)`) the 64 words go to
//!   `((SET_INLINE_QMD_ADDRESS_A << 32) | SET_INLINE_QMD_ADDRESS_B) << 8` and, when scheduled,
//!   are signalled with action 3 (`kepler_compute.cpp`, the `method == 0x107` case). It is
//!   scheduled when `(control & (1U << 24)) || !(native_registers[0xcc] & (1U << 12))`
//!   (`kepler_compute.cpp`, the same case): `SET_QMD_VIRTUALIZATION_CONTROL`'s
//!   `SEND_PCAS_ENABLE` (bit 24, `clc7c0.h`) is set, or the inline QMD's
//!   `REQUIRE_SCHEDULING_PCAS` is clear. Method 0xCC is `LOAD_INLINE_QMD_DATA(4)`, the inline
//!   QMD's word 4, and its bit 12 is QMD bit 140, `NVC7C0_QMDV03_00_REQUIRE_SCHEDULING_PCAS`
//!   (`MW(140:140)`, `clc7c0qmd.h` line 504), read here through
//!   [`crate::classes::clc7c0qmd::QMDV03_00_REQUIRE_SCHEDULING_PCAS`]; a QMD that needs no
//!   scheduling action is scheduled once it is loaded (gpu-protocol, "Compute: QMD 3.0", step
//!   1; decided by the owner, 2026-11-18). Pinned by
//!   `an_inline_qmd_is_scheduled_when_send_pcas_is_set_or_require_scheduling_pcas_is_clear`.
//! - When a request fails, what the requests before it did stands: a QMD that ran keeps its
//!   dispatch, releases and completion, and a QMD loaded or scheduled by an earlier request stays
//!   so. What the failing request itself leaves depends on where it fails (gpu-protocol,
//!   "Compute: QMD 3.0", step 2; decided by the owner, 2026-11-18):
//!   - Failures before its entry is written change nothing and add no entry:
//!     [`QmdError::TableFull`]; [`QmdError::Read`], [`QmdError::Version`] or [`QmdError::Queue`]
//!     when the QMD is loaded; and [`QmdError::DependenceOverflow`]. Eden-NX2 adds an entry for
//!     the load-time read, version and queue failures (a listed divergence below) and has no
//!     overflow check.
//!   - Failures after its entry is written keep the request's own change (loaded, scheduled,
//!     the new counter), as Eden-NX2 does, since it updates the entry in place before these
//!     checks: [`QmdError::DependenceUnderflow`], and [`QmdError::Read`], [`QmdError::Version`],
//!     [`QmdError::Queue`] or [`QmdError::Release`] when the QMD is read again to run. The QMD is
//!     not marked completed and nothing reaches the sink; a QMD new to the table keeps its new
//!     entry. Pinned by `a_failed_run_keeps_the_requests_own_change`,
//!     `a_counter_below_zero_when_ready_is_an_error_and_zero_runs` and
//!     `a_qmd_that_goes_bad_while_waiting_fails_when_it_would_run`.
//!   - [`QmdError::DependentAction`], [`QmdError::ChainTooLong`] and
//!     [`QmdError::SecondDependent`] come after the QMD ran (above): it keeps its completion.
//!
//! # Divergences from Eden-NX2
//!
//! Each is pinned by the test named, in `qmd/tests.rs`. The first is the owner's
//! reference-conflict rule (gpu-protocol decision 10) applied as the issue asks: the header's
//! meaning, plus Eden-NX2's observed behaviour where the header does not explain it.
//!
//! - **Bit 19 of a release slot's second word is a max release.** The header calls the bit
//!   `RELEASE0_MEMBAR_TYPE` (`FE_SYSMEMBAR` when set); Eden-NX2 writes `max(old, payload)` for a
//!   release with it set and no reduction (row 62; `kepler_compute.cpp`, the `1U << 19` branch of
//!   the release callback). Here such a slot is a [`ReleaseOp::Reduce`] with
//!   [`MAX_RELEASE`]; a slot with reduction enabled uses its own reduction whatever bit 19 says,
//!   as in Eden-NX2. This is a **guess** until verification item 1 of the spec ("QMD release flag
//!   bit 19") is settled: if a title's max releases always carry reduction enable with op 2, the
//!   rule goes. Pinned by `divergence_bit_19_release_is_a_max_release`.
//! - **Every reduction the header defines.** Eden-NX2 refuses a release with reduction enabled
//!   and an op other than `RED_ADD`, or with any reduction format bit set (`native_qmd.h` lines
//!   80–81); here the op and format are decoded as the header defines them (`RED_ADD` to `RED_XOR`,
//!   unsigned or signed; [`crate::sema`]), and a format the header does not define is
//!   [`ReleaseRefusal::Reduction`]. Pinned by `divergence_every_header_reduction_is_released`.
//! - **The reduction format is ignored without reduction enable**, where Eden-NX2 refuses it
//!   (same check). Pinned by `divergence_format_bits_without_reduction_are_ignored`.
//! - **Constant buffer addresses are 49 bits.** `DispatchNativeCompute` refuses a program whose
//!   constant buffer upper address has bits 16:8 set (`high & 0x1ff00`); here the header's whole
//!   `CONSTANT_BUFFER_ADDR_UPPER` (17 bits) is decoded. Pinned by
//!   `divergence_constant_buffer_upper_bits_are_kept`.
//! - **An inline QMD address that does not fit in 64 bits is refused**
//!   ([`QmdError::InlineAddress`]), where Eden-NX2's shift drops the top eight bits of
//!   `SET_INLINE_QMD_ADDRESS_A`. Pinned by `divergence_inline_address_overflow_is_refused`.
//! - **A request that fails when its QMD is loaded adds no table entry**, as the spec's step 2
//!   says (decided by the owner, 2026-11-18). Eden-NX2's
//!   `states[request.address]` adds an empty entry before the QMD is read, so a QMD that is
//!   unmapped ([`QmdError::Read`]), not version 3.0 ([`QmdError::Version`]) or a queue
//!   ([`QmdError::Queue`]) when it is loaded still takes one of the 4,096 places. A request that
//!   fails after its entry is written (see above) keeps its entry, as in Eden-NX2. Pinned by
//!   `divergence_a_failed_request_adds_no_entry` (the load-time failures) and
//!   `a_failed_run_keeps_the_requests_own_change` (a run-time failure on a new QMD).
//! - **An unsupported PCAS action is reported before the table limit.**
//!   [`QmdTable::compute_method`] decodes the `SEND_SIGNALING_PCAS2_B` action before it signals,
//!   so an unsupported action for a QMD not in a full table is [`QmdError::PcasAction`]; Eden-NX2
//!   checks the limit first (line 35, then line 66) and reports "QMD cache limit". Pinned by
//!   `divergence_an_unsupported_action_is_reported_before_the_table_limit`.
//! - **A run that is not dispatched is not an error.** Eden-NX2's `Signal` sets its result to
//!   "native shader dispatch required" (`native_qmd.h` line 90) when its dispatch callback
//!   returns false: a program the shader path cannot run, a block with a zero dimension, or a
//!   constant buffer upper address with bits 16:8 set (the last is not refused here at all; see
//!   above). It still completes, releases and runs the rest of the chain, and returns that error
//!   at the end unless a later request fails first. Here [`QmdTable::signal`] returns `Ok(())` for [`Outcome::Rejected`] and
//!   [`Outcome::EmptyBlock`], and the outcome reaches only [`QmdSink::completed`]. Pinned by
//!   `a_rejected_program_still_completes_and_releases` and
//!   `an_empty_grid_or_block_completes_and_releases_without_reaching_the_sink`.
//! - **Releases are not checked against the address map here.** Eden-NX2 fails the signal with
//!   "unmapped QMD release" when a release address is unmapped, after the QMD completed; here a
//!   release is handed to the sink, and its write is the fence path's (G15). Pinned by
//!   `divergence_release_addresses_are_not_checked_here`.
//!
//! # Refused as Eden-NX2 refuses them
//!
//! A release slot whose `STRUCTURE_SIZE` is not `SEMAPHORE_ONE_WORD`, whose `PAYLOAD64B` is set
//! or whose `TRAP_TYPE` is not `TRAP_NONE` is [`QmdError::Release`], checked in that order
//! (Eden-NX2 checks them as one condition, lines 80–81). The header defines these forms, but not how
//! they combine with each other or with a reduction, so they stay refused until a trace shows one.
//!
//! # Guesses and what is not here
//!
//! - `SEND_SIGNALING_PCAS2_B` bits 23:14 are `OFFSET_MINUS_ONE` in the header; Eden-NX2 reads
//!   them as the count, minus one, by which actions 7 and 12 move the counter (line 32). The
//!   header does not say. A trace of a decrement by more than one would confirm it.
//! - The table is keyed by GPU address alone, as Eden-NX2's (one static map for every channel
//!   and process). Two GPU address spaces would share keys; whether NVN2 ever schedules QMDs
//!   from two is not known.
//! - Entries are never removed, as in Eden-NX2: a run that touches more than 4,096 distinct QMD
//!   addresses fails from then on.
//! - `SEND_PCAS_B` (0xAE) and `SEND_SIGNALING_PCAS_B` (0xAF) are not handled. Eden-NX2 ignores
//!   the first and treats the second, when its `SCHEDULE` bit is set, as a signal with the word
//!   itself as the action; the issue for this slice covers only `SEND_SIGNALING_PCAS2_B`.

#[cfg(test)]
pub(crate) mod tests;

use std::collections::BTreeMap;

use thiserror::Error;

use crate::classes::clc7c0::{
    LOAD_INLINE_QMD_DATA, SEND_PCAS_A, SEND_SIGNALING_PCAS2_B,
    SEND_SIGNALING_PCAS2_B_OFFSET_MINUS_ONE, SEND_SIGNALING_PCAS2_B_PCAS_ACTION,
    SEND_SIGNALING_PCAS2_B_PCAS_ACTION_DECREMENT_DEPENDENCE,
    SEND_SIGNALING_PCAS2_B_PCAS_ACTION_INCREMENT_DEPENDENCE,
    SEND_SIGNALING_PCAS2_B_PCAS_ACTION_INVALIDATE,
    SEND_SIGNALING_PCAS2_B_PCAS_ACTION_INVALIDATE_COPY_SCHEDULE,
    SEND_SIGNALING_PCAS2_B_PCAS_ACTION_NOP, SEND_SIGNALING_PCAS2_B_PCAS_ACTION_SCHEDULE,
    SET_INLINE_QMD_ADDRESS_A, SET_INLINE_QMD_ADDRESS_B, SET_QMD_VIRTUALIZATION_CONTROL,
    SET_QMD_VIRTUALIZATION_CONTROL_SEND_PCAS_ENABLE,
    SET_QMD_VIRTUALIZATION_CONTROL_SEND_PCAS_ENABLE_TRUE,
};
use crate::classes::clc7c0qmd as q;
use crate::classes::{MwArray, MwField};
use crate::memory::{GpuFault, GpuMemory};
use crate::ring::{Release, ReleaseOp};
use crate::sema::{
    self, PayloadSize, QMD_RELEASE, Reduction, ReductionFormat, ReductionOp, ReleaseForm,
};

/// The words of a QMD: `NativeQmdScheduler::Descriptor` is `std::array<Word, 64>`
/// (`native_qmd.h` line 18), and the header's last field, `DEBUG_ID_LOWER`, is `MW(2047:2016)`
/// (`clc7c0qmd.h` line 767), the top of word 63.
pub const QMD_WORDS: usize = QMD_WORDS_U32 as usize;

/// [`QMD_WORDS`] as a method count.
const QMD_WORDS_U32: u32 = 64;

/// The bytes of a QMD, as read from and written to GPU memory (little-endian words).
pub const QMD_BYTES: usize = QMD_WORDS * WORD_BYTES;

/// The bytes of one word.
const WORD_BYTES: usize = 4;

/// How far a QMD address is shifted in the methods and fields that hold one: `SEND_PCAS_A`'s
/// `QMD_ADDRESS_SHIFTED8`, `SET_INLINE_QMD_ADDRESS_A/B`'s `QMD_ADDRESS_SHIFTED8_UPPER/LOWER`
/// (`clc7c0.h` lines 442, 492, 495) and the dependent QMD pointer (`native_qmd.h` line 109,
/// `Address{qmd[15]} << 8`).
pub const QMD_ADDRESS_SHIFT: u32 = 8;

/// The most QMDs the table holds (`native_qmd.h` line 35, `states.size() >= 4096`).
pub const MAX_QMDS: usize = 4096;

/// The most requests one [`QmdTable::signal`] handles, the first included (`native_qmd.h`
/// line 30, `std::array<Pending, 256> pending`, and line 108, `count == pending.size()`).
pub const MAX_CHAIN: usize = 256;

/// The constant buffers a QMD binds: `NumConstBuffers = 8`
/// (`eden-nx2/src/video_core/engines/kepler_compute.h` line 84). In the header the
/// `CONSTANT_BUFFER_ADDR_LOWER(i)` array starts at bit 1024 with a stride of 64 and
/// `PROGRAM_ADDRESS_LOWER` follows at bit 1536 (`clc7c0qmd.h` lines 697 and 706), which leaves
/// room for eight.
pub const CONSTANT_BUFFERS: usize = 8;

/// `CONSTANT_BUFFER_SIZE_SHIFTED4`: the size is stored shifted right by four
/// (`clc7c0qmd.h` line 705; `DispatchNativeCompute`, `<< 4`).
pub const CONSTANT_BUFFER_SIZE_SHIFT: u32 = 4;

/// The QMD major version this module decodes: the header's `QMDV03_00` structure, and Eden-NX2's
/// `(qmd[18] & 0xff) != 0x30` (major 3 in bits 7:4, minor 0 in bits 3:0).
pub const QMD_MAJOR_VERSION: u32 = 3;

/// The QMD minor version this module decodes (see [`QMD_MAJOR_VERSION`]).
pub const QMD_MINOR_VERSION: u32 = 0;

/// The reduction a release slot with bit 19 set and no reduction gets: Eden-NX2's
/// `std::max(memory_manager.Read<u32>(address), payload)`, an unsigned 32-bit max. A listed
/// divergence (see the module docs).
pub const MAX_RELEASE: Reduction = Reduction {
    op: ReductionOp::Max,
    format: ReductionFormat::Unsigned,
};

/// The number of release slots.
pub const RELEASE_SLOTS: usize = 3;

/// The method of `LOAD_INLINE_QMD_DATA(63)`, the inline QMD's last word (0x107; Eden-NX2's
/// `method == 0x107`).
pub const LAST_INLINE_QMD_METHOD: u32 = LOAD_INLINE_QMD_DATA.method() + QMD_WORDS_U32 - 1;

/// The register file must reach the highest method this module reads.
const REGISTERS_NEEDED: usize = LAST_INLINE_QMD_METHOD as usize + 1;

/// The fields of one release slot, all from `clc7c0qmd.h`.
#[derive(Clone, Copy, Debug)]
struct ReleaseSlot {
    address_lower: MwField,
    address_upper: MwField,
    membar_type: MwField,
    reduction_op: MwField,
    enable: MwField,
    reduction_format: MwField,
    reduction_enable: MwField,
    trap_type: MwField,
    payload64b: MwField,
    structure_size: MwField,
    payload_lower: MwField,
}

/// The three release slots, at words 24, 28 and 52 (`native_qmd.h` line 118, `ReleaseSlots{24,
/// 28, 52}`): `RELEASE0_*` (`clc7c0qmd.h` lines 621 to 657), `RELEASE1_*` (lines 659 to 695)
/// and `RELEASE2_*` (lines 715 to 751). The value constants of `RELEASE0` (`_TRUE`,
/// `_FE_SYSMEMBAR`, `_TRAP_NONE`, `_SEMAPHORE_ONE_WORD`) are the same in all three.
const SLOTS: [ReleaseSlot; RELEASE_SLOTS] = [
    ReleaseSlot {
        address_lower: q::QMDV03_00_RELEASE0_ADDRESS_LOWER,
        address_upper: q::QMDV03_00_RELEASE0_ADDRESS_UPPER,
        membar_type: q::QMDV03_00_RELEASE0_MEMBAR_TYPE,
        reduction_op: q::QMDV03_00_RELEASE0_REDUCTION_OP,
        enable: q::QMDV03_00_RELEASE0_ENABLE,
        reduction_format: q::QMDV03_00_RELEASE0_REDUCTION_FORMAT,
        reduction_enable: q::QMDV03_00_RELEASE0_REDUCTION_ENABLE,
        trap_type: q::QMDV03_00_RELEASE0_TRAP_TYPE,
        payload64b: q::QMDV03_00_RELEASE0_PAYLOAD64B,
        structure_size: q::QMDV03_00_RELEASE0_STRUCTURE_SIZE,
        payload_lower: q::QMDV03_00_RELEASE0_PAYLOAD_LOWER,
    },
    ReleaseSlot {
        address_lower: q::QMDV03_00_RELEASE1_ADDRESS_LOWER,
        address_upper: q::QMDV03_00_RELEASE1_ADDRESS_UPPER,
        membar_type: q::QMDV03_00_RELEASE1_MEMBAR_TYPE,
        reduction_op: q::QMDV03_00_RELEASE1_REDUCTION_OP,
        enable: q::QMDV03_00_RELEASE1_ENABLE,
        reduction_format: q::QMDV03_00_RELEASE1_REDUCTION_FORMAT,
        reduction_enable: q::QMDV03_00_RELEASE1_REDUCTION_ENABLE,
        trap_type: q::QMDV03_00_RELEASE1_TRAP_TYPE,
        payload64b: q::QMDV03_00_RELEASE1_PAYLOAD64B,
        structure_size: q::QMDV03_00_RELEASE1_STRUCTURE_SIZE,
        payload_lower: q::QMDV03_00_RELEASE1_PAYLOAD_LOWER,
    },
    ReleaseSlot {
        address_lower: q::QMDV03_00_RELEASE2_ADDRESS_LOWER,
        address_upper: q::QMDV03_00_RELEASE2_ADDRESS_UPPER,
        membar_type: q::QMDV03_00_RELEASE2_MEMBAR_TYPE,
        reduction_op: q::QMDV03_00_RELEASE2_REDUCTION_OP,
        enable: q::QMDV03_00_RELEASE2_ENABLE,
        reduction_format: q::QMDV03_00_RELEASE2_REDUCTION_FORMAT,
        reduction_enable: q::QMDV03_00_RELEASE2_REDUCTION_ENABLE,
        trap_type: q::QMDV03_00_RELEASE2_TRAP_TYPE,
        payload64b: q::QMDV03_00_RELEASE2_PAYLOAD64B,
        structure_size: q::QMDV03_00_RELEASE2_STRUCTURE_SIZE,
        payload_lower: q::QMDV03_00_RELEASE2_PAYLOAD_LOWER,
    },
];

/// A field's value in a QMD's words. Every field this module reads lies inside the 64 words
/// (`tests::every_field_read_lies_inside_the_qmd`), so the `None` of [`MwField::get`] does not
/// arise; it reads as zero.
fn get(words: &[u32; QMD_WORDS], field: MwField) -> u32 {
    field.get(words).unwrap_or(0)
}

/// Element `index` of a field array's value (see [`get`]).
fn get_element(words: &[u32; QMD_WORDS], array: MwArray, index: u32) -> u32 {
    array.element(index).map_or(0, |field| get(words, field))
}

/// A 64-bit address from its upper and lower fields.
fn address(words: &[u32; QMD_WORDS], upper: MwField, lower: MwField) -> u64 {
    (u64::from(get(words, upper)) << u32::BITS) | u64::from(get(words, lower))
}

/// A QMD address from a field or method word that holds it shifted right by
/// [`QMD_ADDRESS_SHIFT`]; a 32-bit word always fits.
fn shifted_address(word: u32) -> u64 {
    u64::from(word) << QMD_ADDRESS_SHIFT
}

/// The address the inline QMD is written to: `SET_INLINE_QMD_ADDRESS_A/B` as one 64-bit value
/// shifted left by [`QMD_ADDRESS_SHIFT`] (module docs, the inline QMD); a register past the
/// end of `registers` reads as 0. The one derivation of it: [`QmdTable::compute_method`] writes
/// the inline QMD there, and the executor (slice G19) drops those bytes from its caches.
///
/// # Errors
///
/// [`QmdError::InlineAddress`] when the shifted address does not fit in 64 bits (a listed
/// divergence, `divergence_inline_address_overflow_is_refused`).
pub(crate) fn inline_qmd_address(registers: &[u32]) -> Result<u64, QmdError> {
    let register = |method: u32| {
        usize::try_from(method)
            .ok()
            .and_then(|index| registers.get(index))
            .copied()
            .unwrap_or(0)
    };
    let upper = register(SET_INLINE_QMD_ADDRESS_A.method());
    let lower = register(SET_INLINE_QMD_ADDRESS_B.method());
    let shifted = (u64::from(upper) << u32::BITS) | u64::from(lower);
    if shifted.leading_zeros() < QMD_ADDRESS_SHIFT {
        return Err(QmdError::InlineAddress { upper });
    }
    Ok(shifted << QMD_ADDRESS_SHIFT)
}

/// What a QMD asks the GPU to run, as `DispatchNativeCompute` reads it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Dispatch {
    /// The GPU address of the QMD.
    pub qmd: u64,
    /// The program's GPU address: `PROGRAM_ADDRESS_UPPER` (17 bits) and `_LOWER` (words 48–49,
    /// `clc7c0qmd.h` lines 706–707).
    pub program: u64,
    /// The grid in blocks: `CTA_RASTER_WIDTH` (32 bits), `_HEIGHT` and `_DEPTH` (16 bits each)
    /// (words 12–14, lines 567–569).
    pub grid: [u32; 3],
    /// The block in threads: `CTA_THREAD_DIMENSION0/1/2` (16 bits each; words 18–19, lines
    /// 605–607).
    pub block: [u32; 3],
    /// `SHARED_MEMORY_SIZE` in bytes (18 bits; word 17, line 597).
    pub shared_memory: u32,
    /// `SHADER_LOCAL_MEMORY_LOW_SIZE` (24 bits; word 23, line 619): Eden-NX2's `local_pos_alloc`.
    pub local_memory_low: u32,
    /// `SHADER_LOCAL_MEMORY_HIGH_SIZE` (24 bits; word 50, line 708): Eden-NX2's
    /// `local_crs_alloc`.
    pub local_memory_high: u32,
    /// The constant buffers (words 20 and 32–47).
    pub constant_buffers: [ConstantBuffer; CONSTANT_BUFFERS],
}

/// One constant buffer binding of a [`Dispatch`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct ConstantBuffer {
    /// `CONSTANT_BUFFER_VALID(i)` (bit 640 + i, `clc7c0qmd.h` line 608).
    pub valid: bool,
    /// `CONSTANT_BUFFER_ADDR_UPPER(i)` (17 bits) and `_LOWER(i)` (lines 697–698).
    pub address: u64,
    /// The size in bytes: `CONSTANT_BUFFER_SIZE_SHIFTED4(i)` (13 bits, line 705) shifted left by
    /// [`CONSTANT_BUFFER_SIZE_SHIFT`].
    pub size: u32,
}

/// A QMD's dependent QMD: the action to apply to another QMD once this one completes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Dependent {
    /// The dependent QMD's address: `DEPENDENT_QMD0_POINTER << 8` (word 15, `clc7c0qmd.h` line
    /// 570).
    pub address: u64,
    /// `DEPENDENT_QMD0_ACTION` as stored (3 bits, line 574); [`Dependent::pcas_action`] says
    /// which are supported.
    pub action: u32,
}

impl Dependent {
    /// The PCAS action this dependent applies (`native_qmd.h` lines 107 and 109):
    /// `QMD_SCHEDULE` is [`PcasAction::Schedule`], `QMD_INVALIDATE_COPY_SCHEDULE` is
    /// [`PcasAction::InvalidateCopySchedule`] and `QMD_DECREMENT_DEPENDENCE` is a
    /// [`PcasAction::DecrementDependence`] by one. Any other action is
    /// [`QmdError::DependentAction`] for the QMD at `qmd`.
    pub fn pcas_action(self, qmd: u64) -> Result<PcasAction, QmdError> {
        match self.action {
            q::QMDV03_00_DEPENDENT_QMD0_ACTION_QMD_SCHEDULE => Ok(PcasAction::Schedule),
            q::QMDV03_00_DEPENDENT_QMD0_ACTION_QMD_INVALIDATE_COPY_SCHEDULE => {
                Ok(PcasAction::InvalidateCopySchedule)
            }
            q::QMDV03_00_DEPENDENT_QMD0_ACTION_QMD_DECREMENT_DEPENDENCE => {
                Ok(PcasAction::DecrementDependence { count: 1 })
            }
            action => Err(QmdError::DependentAction {
                address: qmd,
                action,
            }),
        }
    }
}

/// A decoded QMD 3.0.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Qmd {
    /// `DEPENDENCE_COUNTER` (bits 157:142, `clc7c0qmd.h` line 508): how many decrements the
    /// QMD waits for once loaded.
    pub dependence_counter: u16,
    /// What it runs.
    pub dispatch: Dispatch,
    /// The release of each slot (words 24, 28 and 52), `None` where the slot is not enabled.
    pub releases: [Option<Release>; RELEASE_SLOTS],
    /// The dependent QMD, if `DEPENDENT_QMD0_ENABLE` (bit 512, line 571) is set.
    pub dependent: Option<Dependent>,
    /// Whether `DEPENDENT_QMD1_ENABLE` (bit 517, line 582) is set, which the scheduler refuses.
    pub second_dependent: bool,
}

impl Qmd {
    /// Decodes the QMD at GPU address `address` from its words: its version and queue flag
    /// ([`QmdError::Version`], then [`QmdError::Queue`]), then each release slot in slot order
    /// ([`QmdError::Release`]). The dependent action is decoded but not checked
    /// ([`Dependent::pcas_action`] does that).
    pub fn decode(address: u64, words: &[u32; QMD_WORDS]) -> Result<Self, QmdError> {
        let dependence_counter = check_header(address, words)?;
        let mut releases = [None; RELEASE_SLOTS];
        for (slot, (release, fields)) in releases.iter_mut().zip(&SLOTS).enumerate() {
            *release = decode_release(words, fields).map_err(|refusal| QmdError::Release {
                address,
                slot,
                refusal,
            })?;
        }
        let dependent = (get(words, q::QMDV03_00_DEPENDENT_QMD0_ENABLE)
            == q::QMDV03_00_DEPENDENT_QMD0_ENABLE_TRUE)
            .then(|| Dependent {
                address: shifted_address(get(words, q::QMDV03_00_DEPENDENT_QMD0_POINTER)),
                action: get(words, q::QMDV03_00_DEPENDENT_QMD0_ACTION),
            });
        Ok(Self {
            dependence_counter,
            dispatch: decode_dispatch(address, words),
            releases,
            dependent,
            second_dependent: get(words, q::QMDV03_00_DEPENDENT_QMD1_ENABLE)
                == q::QMDV03_00_DEPENDENT_QMD1_ENABLE_TRUE,
        })
    }
}

/// Checks a QMD's version and queue flag, and returns its dependence counter.
fn check_header(address: u64, words: &[u32; QMD_WORDS]) -> Result<u16, QmdError> {
    let major = get(words, q::QMDV03_00_QMD_MAJOR_VERSION);
    let minor = get(words, q::QMDV03_00_QMD_VERSION);
    if major != QMD_MAJOR_VERSION || minor != QMD_MINOR_VERSION {
        return Err(QmdError::Version {
            address,
            major,
            minor,
        });
    }
    if get(words, q::QMDV03_00_IS_QUEUE) != q::QMDV03_00_IS_QUEUE_FALSE {
        return Err(QmdError::Queue { address });
    }
    // The field is 16 bits wide, so this cannot fail; it reads as the largest counter if it did.
    Ok(u16::try_from(get(words, q::QMDV03_00_DEPENDENCE_COUNTER)).unwrap_or(u16::MAX))
}

/// Decodes one release slot: `None` when it is not enabled.
fn decode_release(
    words: &[u32; QMD_WORDS],
    slot: &ReleaseSlot,
) -> Result<Option<Release>, ReleaseRefusal> {
    if get(words, slot.enable) != q::QMDV03_00_RELEASE0_ENABLE_TRUE {
        return Ok(None);
    }
    let structure = get(words, slot.structure_size);
    if structure != q::QMDV03_00_RELEASE0_STRUCTURE_SIZE_SEMAPHORE_ONE_WORD {
        return Err(ReleaseRefusal::StructureSize(structure));
    }
    if get(words, slot.payload64b) != q::QMDV03_00_RELEASE0_PAYLOAD64B_FALSE {
        return Err(ReleaseRefusal::Payload64);
    }
    let trap = get(words, slot.trap_type);
    if trap != q::QMDV03_00_RELEASE0_TRAP_TYPE_TRAP_NONE {
        return Err(ReleaseRefusal::Trap(trap));
    }
    let op = if get(words, slot.reduction_enable) == q::QMDV03_00_RELEASE0_REDUCTION_ENABLE_TRUE {
        ReleaseOp::Reduce(
            sema::red_reduction(
                QMD_RELEASE,
                get(words, slot.reduction_op),
                get(words, slot.reduction_format),
            )
            .map_err(ReleaseRefusal::Reduction)?,
        )
    } else if get(words, slot.membar_type) == q::QMDV03_00_RELEASE0_MEMBAR_TYPE_FE_SYSMEMBAR {
        // Eden-NX2's bit-19 rule: a listed divergence (module docs).
        ReleaseOp::Reduce(MAX_RELEASE)
    } else {
        ReleaseOp::Write
    };
    Ok(Some(Release {
        address: address(words, slot.address_upper, slot.address_lower),
        payload: u64::from(get(words, slot.payload_lower)),
        form: ReleaseForm {
            size: PayloadSize::Four,
            timestamp: false,
        },
        op,
    }))
}

/// Decodes what a QMD runs.
fn decode_dispatch(qmd: u64, words: &[u32; QMD_WORDS]) -> Dispatch {
    let mut constant_buffers = [ConstantBuffer::default(); CONSTANT_BUFFERS];
    for (index, buffer) in (0u32..).zip(constant_buffers.iter_mut()) {
        let upper = get_element(words, q::QMDV03_00_CONSTANT_BUFFER_ADDR_UPPER, index);
        let lower = get_element(words, q::QMDV03_00_CONSTANT_BUFFER_ADDR_LOWER, index);
        *buffer = ConstantBuffer {
            valid: get_element(words, q::QMDV03_00_CONSTANT_BUFFER_VALID, index)
                == q::QMDV03_00_CONSTANT_BUFFER_VALID_TRUE,
            address: (u64::from(upper) << u32::BITS) | u64::from(lower),
            size: get_element(words, q::QMDV03_00_CONSTANT_BUFFER_SIZE_SHIFTED4, index)
                << CONSTANT_BUFFER_SIZE_SHIFT,
        };
    }
    Dispatch {
        qmd,
        program: address(
            words,
            q::QMDV03_00_PROGRAM_ADDRESS_UPPER,
            q::QMDV03_00_PROGRAM_ADDRESS_LOWER,
        ),
        grid: [
            get(words, q::QMDV03_00_CTA_RASTER_WIDTH),
            get(words, q::QMDV03_00_CTA_RASTER_HEIGHT),
            get(words, q::QMDV03_00_CTA_RASTER_DEPTH),
        ],
        block: [
            get(words, q::QMDV03_00_CTA_THREAD_DIMENSION0),
            get(words, q::QMDV03_00_CTA_THREAD_DIMENSION1),
            get(words, q::QMDV03_00_CTA_THREAD_DIMENSION2),
        ],
        shared_memory: get(words, q::QMDV03_00_SHARED_MEMORY_SIZE),
        local_memory_low: get(words, q::QMDV03_00_SHADER_LOCAL_MEMORY_LOW_SIZE),
        local_memory_high: get(words, q::QMDV03_00_SHADER_LOCAL_MEMORY_HIGH_SIZE),
        constant_buffers,
    }
}

/// What a PCAS signal asks of a QMD (`SEND_SIGNALING_PCAS2_B_PCAS_ACTION`, `clc7c0.h` lines
/// 457–469), for the actions the scheduler supports.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PcasAction {
    /// `SCHEDULE` (2): schedule the QMD, loading it first if it is not loaded.
    Schedule,
    /// `INVALIDATE_COPY_SCHEDULE` (3): load the QMD again, then schedule it.
    InvalidateCopySchedule,
    /// `DECREMENT_DEPENDENCE` (7): lower the dependence counter by `count`.
    DecrementDependence {
        /// By how much (`OFFSET_MINUS_ONE + 1`; see the module docs' guesses).
        count: u16,
    },
    /// `INCREMENT_DEPENDENCE` (12): raise the dependence counter by `count`.
    IncrementDependence {
        /// By how much (`OFFSET_MINUS_ONE + 1`).
        count: u16,
    },
}

impl PcasAction {
    /// Decodes a `SEND_SIGNALING_PCAS2_B` word: `None` for `NOP` and `INVALIDATE`, which do
    /// nothing here (Eden-NX2 returns before scheduling for them), and
    /// [`QmdError::PcasAction`] for an action the scheduler does not support.
    pub fn decode(word: u32) -> Result<Option<Self>, QmdError> {
        // OFFSET_MINUS_ONE is 10 bits wide, so the count is 1 to 1024.
        let count = u16::try_from(SEND_SIGNALING_PCAS2_B_OFFSET_MINUS_ONE.get(word))
            .unwrap_or(u16::MAX)
            .saturating_add(1);
        match SEND_SIGNALING_PCAS2_B_PCAS_ACTION.get(word) {
            SEND_SIGNALING_PCAS2_B_PCAS_ACTION_NOP
            | SEND_SIGNALING_PCAS2_B_PCAS_ACTION_INVALIDATE => Ok(None),
            SEND_SIGNALING_PCAS2_B_PCAS_ACTION_SCHEDULE => Ok(Some(Self::Schedule)),
            SEND_SIGNALING_PCAS2_B_PCAS_ACTION_INVALIDATE_COPY_SCHEDULE => {
                Ok(Some(Self::InvalidateCopySchedule))
            }
            SEND_SIGNALING_PCAS2_B_PCAS_ACTION_DECREMENT_DEPENDENCE => {
                Ok(Some(Self::DecrementDependence { count }))
            }
            SEND_SIGNALING_PCAS2_B_PCAS_ACTION_INCREMENT_DEPENDENCE => {
                Ok(Some(Self::IncrementDependence { count }))
            }
            action => Err(QmdError::PcasAction { action }),
        }
    }

    /// The `SEND_SIGNALING_PCAS2_B` word for this action (`OFFSET_MINUS_ONE` is zero for the
    /// schedule actions); `None` when a count is 0 or does not fit the field (above 1024).
    pub fn encode(self) -> Option<u32> {
        let (action, count) = match self {
            Self::Schedule => (SEND_SIGNALING_PCAS2_B_PCAS_ACTION_SCHEDULE, 1),
            Self::InvalidateCopySchedule => (
                SEND_SIGNALING_PCAS2_B_PCAS_ACTION_INVALIDATE_COPY_SCHEDULE,
                1,
            ),
            Self::DecrementDependence { count } => (
                SEND_SIGNALING_PCAS2_B_PCAS_ACTION_DECREMENT_DEPENDENCE,
                count,
            ),
            Self::IncrementDependence { count } => (
                SEND_SIGNALING_PCAS2_B_PCAS_ACTION_INCREMENT_DEPENDENCE,
                count,
            ),
        };
        let offset = u32::from(count.checked_sub(1)?);
        let word = SEND_SIGNALING_PCAS2_B_PCAS_ACTION.insert(0, action)?;
        SEND_SIGNALING_PCAS2_B_OFFSET_MINUS_ONE.insert(word, offset)
    }
}

/// Whether the sink could run a dispatch's program.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Program {
    /// The dispatch was recorded.
    Accepted,
    /// The program cannot run (the shader cache rejected it); the QMD still completes and
    /// releases (row 61).
    Rejected,
}

/// How a QMD that ran was dispatched.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Outcome {
    /// The sink accepted the dispatch.
    Dispatched,
    /// The sink rejected the program.
    Rejected,
    /// A grid dimension is zero: nothing to dispatch (`native_qmd.h` line 89).
    EmptyGrid,
    /// A block dimension is zero: refused before the sink, as `DispatchNativeCompute` refuses it.
    EmptyBlock,
}

/// Where a run's work goes. The back's executor (G19) implements it: it records the dispatch
/// and hands the releases to the fence path (G15).
pub trait QmdSink {
    /// A QMD's dispatch, with a non-empty grid and block. Returns whether its program can run.
    fn dispatch(&mut self, dispatch: &Dispatch) -> Program;
    /// One of a QMD's releases, in slot order, after its dispatch: to be performed once the
    /// work before it has finished on the host GPU.
    fn release(&mut self, release: Release);
    /// A QMD has completed, after its dispatch and releases.
    fn completed(&mut self, qmd: u64, outcome: Outcome);
}

/// One QMD's entry in the table: Eden-NX2's `NativeQmdScheduler::State` without the descriptor
/// (the QMD is read from memory whenever it is needed).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct QmdStatus {
    /// Whether the QMD has been read and its counter set.
    pub loaded: bool,
    /// Whether a schedule action has reached it.
    pub scheduled: bool,
    /// Whether it has run since it was last loaded.
    pub completed: bool,
    /// The dependence counter: it runs when this is zero (below zero is an error).
    pub dependencies: i32,
    /// Counter changes that arrived while it was not loaded or had completed, applied at the
    /// next load.
    pub deferred: i32,
}

/// One request of a signal's chain.
#[derive(Clone, Copy, Debug)]
struct Request {
    address: u64,
    action: PcasAction,
}

/// The QMD table shared by every channel (row 84; `native_qmd.h` line 131): the scheduler of
/// PCAS signals and dependent QMDs.
#[derive(Clone, Debug, Default)]
pub struct QmdTable {
    /// By QMD address. Ordered, so nothing depends on hash order.
    states: BTreeMap<u64, QmdStatus>,
}

impl QmdTable {
    /// An empty table.
    pub fn new() -> Self {
        Self::default()
    }

    /// The number of QMDs in the table.
    pub fn len(&self) -> usize {
        self.states.len()
    }

    /// Whether the table is empty.
    pub fn is_empty(&self) -> bool {
        self.states.is_empty()
    }

    /// The entry of the QMD at `address`, if it has one.
    pub fn status(&self, address: u64) -> Option<QmdStatus> {
        self.states.get(&address).copied()
    }

    /// Applies `action` to the QMD at `address`, then the dependent action of every QMD that
    /// runs as a result, in order, reading QMDs from `memory` and sending each run's work to
    /// `sink`. An error stops the chain; what ran before it stands (module docs).
    pub fn signal<M, S>(
        &mut self,
        address: u64,
        action: PcasAction,
        memory: &M,
        sink: &mut S,
    ) -> Result<(), QmdError>
    where
        M: GpuMemory + ?Sized,
        S: QmdSink + ?Sized,
    {
        let mut next = Some(Request { address, action });
        let mut requests = 1;
        while let Some(request) = next {
            next = self.step(request, &mut requests, memory, sink)?;
        }
        Ok(())
    }

    /// Handles one compute-class method, called after the channel's compute register mirror
    /// (`registers`, indexed by method number) has stored its value:
    ///
    /// - [`LAST_INLINE_QMD_METHOD`] writes the inline QMD (`LOAD_INLINE_QMD_DATA(0..64)`) to
    ///   the address in `SET_INLINE_QMD_ADDRESS_A/B` and, when `SET_QMD_VIRTUALIZATION_CONTROL`
    ///   has `SEND_PCAS_ENABLE` set or the inline QMD's `REQUIRE_SCHEDULING_PCAS` is clear (the
    ///   module docs give the rule and its source), signals it with
    ///   [`PcasAction::InvalidateCopySchedule`].
    /// - `SEND_SIGNALING_PCAS2_B` signals the QMD at `SEND_PCAS_A << 8` with its action.
    ///
    /// Every other method does nothing here. A register file shorter than the methods read is
    /// [`QmdError::Registers`].
    pub fn compute_method<M, S>(
        &mut self,
        method: u32,
        registers: &[u32],
        memory: &M,
        sink: &mut S,
    ) -> Result<(), QmdError>
    where
        M: GpuMemory + ?Sized,
        S: QmdSink + ?Sized,
    {
        let pcas2_b = SEND_SIGNALING_PCAS2_B.method();
        if method != pcas2_b && method != LAST_INLINE_QMD_METHOD {
            return Ok(());
        }
        if registers.len() < REGISTERS_NEEDED {
            return Err(QmdError::Registers {
                len: registers.len(),
                needed: REGISTERS_NEEDED,
            });
        }
        let register = |method: u32| {
            usize::try_from(method)
                .ok()
                .and_then(|index| registers.get(index))
                .copied()
                .unwrap_or(0)
        };
        if method == pcas2_b {
            let Some(action) = PcasAction::decode(register(pcas2_b))? else {
                return Ok(());
            };
            let address = shifted_address(register(SEND_PCAS_A.method()));
            return self.signal(address, action, memory, sink);
        }
        let address = inline_qmd_address(registers)?;
        let mut words = [0u32; QMD_WORDS];
        for (word, method) in words.iter_mut().zip(LOAD_INLINE_QMD_DATA.method()..) {
            *word = register(method);
        }
        let mut bytes = [0u8; QMD_BYTES];
        for (chunk, word) in bytes.as_chunks_mut::<WORD_BYTES>().0.iter_mut().zip(words) {
            chunk.copy_from_slice(&word.to_le_bytes());
        }
        memory
            .write(address, &bytes)
            .map_err(|fault| QmdError::InlineWrite { address, fault })?;
        let send_pcas = SET_QMD_VIRTUALIZATION_CONTROL_SEND_PCAS_ENABLE
            .get(register(SET_QMD_VIRTUALIZATION_CONTROL.method()))
            == SET_QMD_VIRTUALIZATION_CONTROL_SEND_PCAS_ENABLE_TRUE;
        // Eden-NX2's `!(native_registers[0xcc] & (1U << 12))`: method 0xCC is the inline QMD's
        // word 4, and bit 12 of it is `REQUIRE_SCHEDULING_PCAS`, QMD bit 140 (`clc7c0qmd.h`
        // line 504).
        let no_require_scheduling = get(&words, q::QMDV03_00_REQUIRE_SCHEDULING_PCAS)
            == q::QMDV03_00_REQUIRE_SCHEDULING_PCAS_FALSE;
        if send_pcas || no_require_scheduling {
            self.signal(address, PcasAction::InvalidateCopySchedule, memory, sink)?;
        }
        Ok(())
    }

    /// Handles one request of a chain and returns the dependent request it leads to.
    fn step<M, S>(
        &mut self,
        request: Request,
        requests: &mut usize,
        memory: &M,
        sink: &mut S,
    ) -> Result<Option<Request>, QmdError>
    where
        M: GpuMemory + ?Sized,
        S: QmdSink + ?Sized,
    {
        let address = request.address;
        let existing = self.states.get(&address).copied();
        if existing.is_none() && self.states.len() >= MAX_QMDS {
            return Err(QmdError::TableFull { address });
        }
        let old = existing.unwrap_or_default();
        let mut new = old;
        match request.action {
            PcasAction::Schedule | PcasAction::InvalidateCopySchedule => {
                if request.action == PcasAction::InvalidateCopySchedule || !old.loaded {
                    let words = read_qmd(memory, address)?;
                    let counter = check_header(address, &words)?;
                    new.dependencies = i32::from(counter)
                        .checked_add(old.deferred)
                        .ok_or(QmdError::DependenceOverflow { address })?;
                    new.deferred = 0;
                    new.loaded = true;
                    new.completed = false;
                }
                new.scheduled = true;
            }
            PcasAction::DecrementDependence { count }
            | PcasAction::IncrementDependence { count } => {
                let delta = if matches!(request.action, PcasAction::DecrementDependence { .. }) {
                    -i32::from(count)
                } else {
                    i32::from(count)
                };
                let overflow = QmdError::DependenceOverflow { address };
                if !old.loaded || old.completed {
                    new.deferred = old.deferred.checked_add(delta).ok_or(overflow)?;
                    self.states.insert(address, new);
                    return Ok(None);
                }
                new.dependencies = old.dependencies.checked_add(delta).ok_or(overflow)?;
            }
        }
        self.states.insert(address, new);
        if !new.loaded || !new.scheduled || new.completed || new.dependencies > 0 {
            return Ok(None);
        }
        if new.dependencies < 0 {
            return Err(QmdError::DependenceUnderflow {
                address,
                counter: new.dependencies,
            });
        }
        self.run(address, requests, memory, sink)
    }

    /// Runs the QMD at `address`, whose entry is ready, and returns its dependent request.
    fn run<M, S>(
        &mut self,
        address: u64,
        requests: &mut usize,
        memory: &M,
        sink: &mut S,
    ) -> Result<Option<Request>, QmdError>
    where
        M: GpuMemory + ?Sized,
        S: QmdSink + ?Sized,
    {
        // Read again: NVN2 patches release payloads while a QMD waits.
        let words = read_qmd(memory, address)?;
        let qmd = Qmd::decode(address, &words)?;
        let dispatch = qmd.dispatch;
        let outcome = if dispatch.grid.contains(&0) {
            Outcome::EmptyGrid
        } else if dispatch.block.contains(&0) {
            Outcome::EmptyBlock
        } else {
            match sink.dispatch(&dispatch) {
                Program::Accepted => Outcome::Dispatched,
                Program::Rejected => Outcome::Rejected,
            }
        };
        if let Some(status) = self.states.get_mut(&address) {
            status.completed = true;
        }
        for release in qmd.releases.into_iter().flatten() {
            sink.release(release);
        }
        sink.completed(address, outcome);
        let next = match qmd.dependent {
            Some(dependent) => {
                let action = dependent.pcas_action(address)?;
                if *requests >= MAX_CHAIN {
                    return Err(QmdError::ChainTooLong {
                        address: dependent.address,
                    });
                }
                *requests += 1;
                Some(Request {
                    address: dependent.address,
                    action,
                })
            }
            None => None,
        };
        if qmd.second_dependent {
            return Err(QmdError::SecondDependent { address });
        }
        Ok(next)
    }
}

/// Reads the QMD at `address`.
fn read_qmd<M: GpuMemory + ?Sized>(memory: &M, address: u64) -> Result<[u32; QMD_WORDS], QmdError> {
    let mut bytes = [0u8; QMD_BYTES];
    memory
        .read(address, &mut bytes)
        .map_err(|fault| QmdError::Read { address, fault })?;
    let mut words = [0u32; QMD_WORDS];
    for (word, chunk) in words.iter_mut().zip(bytes.as_chunks::<WORD_BYTES>().0) {
        *word = u32::from_le_bytes(*chunk);
    }
    Ok(words)
}

/// Why a release slot is refused.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum ReleaseRefusal {
    /// `STRUCTURE_SIZE` is not `SEMAPHORE_ONE_WORD` (1).
    #[error("its structure size {0} is not one word")]
    StructureSize(u32),
    /// `PAYLOAD64B` is set.
    #[error("it has a 64-bit payload")]
    Payload64,
    /// `TRAP_TYPE` is not `TRAP_NONE` (0).
    #[error("its trap type {0} is not none")]
    Trap(u32),
    /// The reduction's format is not one the header defines.
    #[error("{0}")]
    Reduction(sema::DecodeError),
}

/// Why a QMD request failed.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum QmdError {
    /// The table holds [`MAX_QMDS`] QMDs and this one is not among them.
    #[error("the QMD table is full ({MAX_QMDS} QMDs), so QMD {address:#x} cannot be added")]
    TableFull {
        /// The QMD that did not fit.
        address: u64,
    },
    /// A dependent QMD would be request number [`MAX_CHAIN`] + 1 of one signal.
    #[error(
        "dependent QMD {address:#x} would make one signal's chain longer than {MAX_CHAIN} QMDs"
    )]
    ChainTooLong {
        /// The dependent QMD that was not applied.
        address: u64,
    },
    /// The QMD cannot be read.
    #[error("QMD {address:#x} cannot be read: {fault}")]
    Read {
        /// The QMD.
        address: u64,
        /// Why.
        fault: GpuFault,
    },
    /// The QMD's version is not 3.0.
    #[error("QMD {address:#x} is version {major}.{minor}, not 3.0")]
    Version {
        /// The QMD.
        address: u64,
        /// `QMD_MAJOR_VERSION`.
        major: u32,
        /// `QMD_VERSION`.
        minor: u32,
    },
    /// The QMD's `IS_QUEUE` is set.
    #[error("QMD {address:#x} is a queue, which is not supported")]
    Queue {
        /// The QMD.
        address: u64,
    },
    /// A `SEND_SIGNALING_PCAS2_B` action the scheduler does not support.
    #[error("PCAS action {action} is not supported")]
    PcasAction {
        /// The action.
        action: u32,
    },
    /// The QMD is ready to run with its counter below zero.
    #[error("QMD {address:#x} is ready to run with its dependence counter at {counter}")]
    DependenceUnderflow {
        /// The QMD.
        address: u64,
        /// The counter.
        counter: i32,
    },
    /// A counter change would overflow the counter or the deferred change.
    #[error("QMD {address:#x}: the change to its dependence counter does not fit")]
    DependenceOverflow {
        /// The QMD.
        address: u64,
    },
    /// A release slot cannot be performed.
    #[error("QMD {address:#x}: release slot {slot}: {refusal}")]
    Release {
        /// The QMD.
        address: u64,
        /// The slot (0, 1 or 2: words 24, 28 or 52).
        slot: usize,
        /// Why.
        refusal: ReleaseRefusal,
    },
    /// The dependent QMD's action is not supported.
    #[error("QMD {address:#x}: dependent QMD action {action} is not supported")]
    DependentAction {
        /// The QMD whose dependent it is.
        address: u64,
        /// `DEPENDENT_QMD0_ACTION`.
        action: u32,
    },
    /// The QMD enables a second dependent QMD.
    #[error("QMD {address:#x} enables a second dependent QMD, which is not supported")]
    SecondDependent {
        /// The QMD.
        address: u64,
    },
    /// The inline QMD address, shifted left by eight, does not fit in 64 bits.
    #[error("the inline QMD address with upper word {upper:#x} does not fit in 64 bits")]
    InlineAddress {
        /// `SET_INLINE_QMD_ADDRESS_A`.
        upper: u32,
    },
    /// The inline QMD cannot be written.
    #[error("the inline QMD cannot be written at {address:#x}: {fault}")]
    InlineWrite {
        /// Where.
        address: u64,
        /// Why.
        fault: GpuFault,
    },
    /// The register file passed in does not reach every method this module reads.
    #[error("the compute register file has {len} registers, but the QMD methods need {needed}")]
    Registers {
        /// Its length.
        len: usize,
        /// The length needed.
        needed: usize,
    },
}
