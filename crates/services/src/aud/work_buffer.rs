//! The size of the work buffer an audio renderer needs: what `IAudioRendererManager`'s
//! `GetWorkBufferSize` answers, from the renderer parameter a title sends.
//!
//! Ported from Eden-NX2 @ 50fa86bb (GPL-3.0-or-later), paths relative to the reference folder:
//!
//! - `eden-nx2/src/audio_core/common/audio_renderer_parameter.h`
//!   (`AudioRendererParameterInternal`): the parameter's layout ([`RendererParameter`]);
//! - `eden-nx2/src/audio_core/audio_render_manager.cpp` (`Manager::GetWorkBufferSize`): the
//!   revision check before the size;
//! - `eden-nx2/src/audio_core/renderer/system.cpp` (`System::GetWorkBufferSize`): the size
//!   ([`work_buffer_size`]), with the helpers it calls:
//!   `renderer/nodes/node_states.h` (`NodeStates::GetWorkBufferSize`, `Stack::CalcBufferSize`),
//!   `renderer/nodes/edge_matrix.h` (`EdgeMatrix::GetWorkBufferSize`),
//!   `renderer/splitter/splitter_context.cpp` (`SplitterContext::CalcWorkBufferSize`),
//!   `renderer/performance/performance_manager.h`
//!   (`GetRequiredBufferSizeForPerformanceMetricsPerFrame`) and
//!   `renderer/command/command_generator.h` (`CalculateCommandBufferSize`);
//! - `eden-nx2/src/audio_core/common/feature_support.h` (`GetRevisionNum`,
//!   `CheckValidRevision`, `CheckFeatureSupported`'s table) and
//!   `renderer/behavior/behavior_info.cpp` (`SetUserLibRevision` and the `Is..Supported`
//!   queries): the revision and the features it turns on ([`revision_number`], [`Feature`]);
//! - `eden-nx2/src/audio_core/common/common.h`: the counts ([`MAX_EFFECTS`] and the rest).
//!
//! Every file above except `feature_support.h` and `behavior_info.*` is byte for byte the same in
//! Eden PR #4588 (`eden-pr4588/src/audio_core/` @ e5fb3b97); those two differ in how a revision
//! word is read ("The revision word").
//!
//! # The sizes are Eden's host sizes
//!
//! Eden sums `sizeof` of its own renderer classes, which are not the console's: they hold host
//! pointers, spans and vtables. The size here is Eden's, so a title is asked for the work buffer
//! Eden asks for, which row 38 of Eden-NX2's blocker log (`eden-nx2/NX2_NOTES.md`) saw a title
//! accept. What the console answers is not known; a trace of a title's `GetWorkBufferSize` reply
//! on a console would settle it. Each size constant below ([`sizes`]) was measured by compiling
//! the cited header with a C++20 compiler for a 64-bit host (Apple clang with libc++, arm64) and
//! printing `sizeof`; `work_buffer/tests.rs` pins each one as a literal
//! (`the_sizes_are_edens_measured_host_sizes`).
//!
//! One of them depends on the host's C++ library, so Eden itself answers differently on
//! different hosts: `BitArray` (`renderer/nodes/bit_array.h`) holds a `std::vector<bool>`, which
//! is 24 bytes in libc++ (making `BitArray` 32, the value used here) and 40 in libstdc++ (which
//! would make it 48; worked out from libstdc++'s layout, not measured). It only counts when the
//! splitter is supported (revision 2 on), once per sub-mix plus one. Using libc++'s is a guess
//! that any of Eden's answers is good enough; the same trace would settle it.
//!
//! # The revision word
//!
//! Conflict (graphics-server decision 8's rule, `docs/specs/graphics-server.md`): the references
//! read the parameter's revision word differently. Eden-NX2 (and Eden) read a word of `0x100` or
//! more as `(word - 'REV0') >> 24` with wrapping 32-bit arithmetic, whatever its low three bytes
//! hold, and accept a revision of at most [`CURRENT_REVISION`] (15). PR #4588 accepts only a word
//! whose low three bytes are `REV` or `REv` (the "native" tag, row 39), takes the top byte less
//! `'0'`, and accepts at most 15 for `REV` and 16 (`NativeCurrentRevision`) for `REv`. Neither
//! set contains the other, so the rule picks the reading a title ran on: Eden-NX2's, whose
//! `aud:u` row 38 records a title getting further with ([`revision_number`]). The two readings
//! agree on every word both accept: a word below `0x100` is its own number in both, and a `REv`
//! word's lowercase `v` adds `0x20 << 16`, which the shift by 24 drops. They differ on:
//!
//! - `REv@`, revision 16 tagged native: PR #4588 accepts it, here it is
//!   [`WorkBufferError::InvalidRevision`] (`a_native_revision_16_is_refused_as_eden_nx2_does`);
//! - a word of `0x100` or more whose low bytes are not `REV` or `REv` but whose wrapped
//!   difference is at most 15 after the shift: PR #4588 refuses it, here it is accepted as that
//!   number (`a_word_without_the_rev_tag_is_read_as_eden_nx2_reads_it`).
//!
//! A trace of a native title's `GetWorkBufferSize` (the revision word it sends, and whether a
//! console accepts `REv@`) would settle it.
//!
//! # Where this differs from Eden
//!
//! Eden computes in C++'s unsigned arithmetic: some terms are 32-bit (a sum or product of two
//! `u32` fields, `Stack::CalcBufferSize`'s `u32` result, `AlignUp` of a `u32`) and wrap at
//! 2^32, and the rest are 64-bit and wrap at 2^64. A title's parameter is guest input, so here
//! every term is computed exactly and a term that would wrap is refused:
//!
//! - **A term that would wrap** is [`WorkBufferError::Overflow`] where Eden answers the wrapped
//!   size. On every parameter where no term wraps, the size is Eden's exactly
//!   (`work_buffer/tests.rs`: `the_size_is_edens_on_every_golden_parameter`, whose values were
//!   taken from Eden-NX2's function compiled over its own headers). These 32-bit terms each have
//!   a test at their edge, answering Eden's size, and one past it, refused
//!   (`every_32_bit_term_is_refused_one_past_its_edge`): `sub_mixes * MAX_EFFECTS`,
//!   `(sinks + sub_mixes) * TARGET_SAMPLE_COUNT`, `mixes + MAX_CHANNELS`, the node states'
//!   `count * (count * 4)`, `voices * MAX_WAVE_BUFFERS`, `effects + voices * MAX_WAVE_BUFFERS`,
//!   the performance entry count, `perf_frames + 1` and the command buffer's node count.
//!   `sinks + sub_mixes` on its own is refused one past its edge
//!   (`a_sum_of_sinks_and_sub_mixes_past_32_bits_is_refused`); at its edge the product with
//!   `TARGET_SAMPLE_COUNT` already refuses. Some terms cannot be the first to wrap:
//!   `sub_mixes + 1` wraps only at `sub_mixes` = `u32::MAX`, which already wraps
//!   `sub_mixes * MAX_EFFECTS`; and with `sub_mixes + 1` at most 2^24, the node-state and
//!   edge-matrix terms other than `count * (count * 4)` cannot wrap before it does (`count * 4`,
//!   both `AlignUp`s, and `count * count`, which wraps only past `count` = 65535, where
//!   `count * (count * 4)` already has). They are checked all the same, so no path wraps. The
//!   64-bit terms are tested by parameters that wrap a product
//!   (`a_parameter_whose_64_bit_size_would_wrap_is_refused`), not at each term's exact edge.
//! - **A negative `splitter_destinations`** is [`WorkBufferError::NegativeSplitterDestinations`]
//!   when a term reads it (revision 2 on: the splitter's size, and from revision 5 the command
//!   buffer's). Eden converts the `s32` to an unsigned 64-bit value and wraps. Below revision 2
//!   no term reads it and any value is accepted, as in Eden
//!   (`a_negative_splitter_destination_count_is_refused_only_where_a_term_reads_it`).
//!
//! Both answer `IAudioRendererManager`'s [`INVALID_ARGUMENT`](super::INVALID_ARGUMENT), a guess
//! (Eden has no such check).

use crate::framework::Fault;
use crate::framework::payload::Reader;

#[cfg(test)]
mod tests;

/// `MaxEffects` (`common.h`: `constexpr u32 MaxEffects = 256`).
pub const MAX_EFFECTS: u32 = 256;
/// `TargetSampleCount` (`common.h`: `constexpr u32 TargetSampleCount = 240`).
pub const TARGET_SAMPLE_COUNT: u32 = 240;
/// `MaxChannels` (`common.h`: `constexpr u32 MaxChannels = 6`).
pub const MAX_CHANNELS: u32 = 6;
/// `MaxMixBuffers` (`common.h`: `constexpr u32 MaxMixBuffers = 24`).
pub const MAX_MIX_BUFFERS: u32 = 24;
/// `MaxWaveBuffers` (`common.h`: `constexpr u32 MaxWaveBuffers = 4`).
pub const MAX_WAVE_BUFFERS: u32 = 4;
/// `MaxBiquadFilters` (`common.h`: `constexpr u32 MaxBiquadFilters = 2`).
pub const MAX_BIQUAD_FILTERS: u32 = 2;
/// `PerformanceManager::MaxDetailEntries` (`renderer/performance/performance_manager.h`:
/// `static constexpr size_t MaxDetailEntries = 100`).
pub const MAX_DETAIL_ENTRIES: u64 = 100;
/// `CurrentRevision` (`feature_support.h`: `constexpr u32 CurrentRevision = 15`): the highest
/// revision a parameter may name.
pub const CURRENT_REVISION: u32 = 15;
/// What a revision word of `0x100` or more is read against: `Common::MakeMagic('R', 'E', 'V',
/// '0')` (`feature_support.h`, `GetRevisionNum`; `common/common_funcs.h`: `MakeMagic` puts the
/// first character in the low byte).
pub const REVISION_MAGIC: u32 = u32::from_le_bytes(*b"REV0");
/// The smallest word [`revision_number`] reads as a tag rather than a number
/// (`if (user_revision >= 0x100)`).
pub const REVISION_TAG_FROM: u32 = 0x100;
/// How far [`revision_number`] shifts a tag's difference (`user_revision >>= 24`).
pub const REVISION_TAG_SHIFT: u32 = 24;

/// The size of `AudioRendererParameterInternal`
/// (`static_assert(sizeof(AudioRendererParameterInternal) == 0x34, ..)`).
pub const PARAMETER_SIZE: usize = 0x34;

/// Eden's `sizeof` of each renderer class the size counts, on a 64-bit host (module docs, "The
/// sizes are Eden's host sizes"). Paths are relative to `eden-nx2/src/audio_core/renderer/`.
pub(crate) mod sizes {
    /// `sizeof(s32)` and `sizeof(u32)`.
    pub(crate) const WORD: u64 = 4;
    /// `sizeof(u64)`, which `NodeStates` and `EdgeMatrix` divide by.
    pub(crate) const U64: u64 = 8;
    /// `sizeof(MixInfo*)` and `sizeof(VoiceInfo*)`: a host pointer.
    pub(crate) const POINTER: u64 = 8;
    /// `MixInfo` (`mix/mix_info.h`).
    pub(crate) const MIX_INFO: u64 = 2368;
    /// `VoiceInfo` (`voice/voice_info.h`).
    pub(crate) const VOICE_INFO: u64 = 560;
    /// `VoiceChannelResource` (`voice/voice_channel_resource.h`).
    pub(crate) const VOICE_CHANNEL_RESOURCE: u64 = 200;
    /// `VoiceState` (`voice/voice_state.h`).
    pub(crate) const VOICE_STATE: u64 = 296;
    /// `BitArray` (`nodes/bit_array.h`), with libc++'s `std::vector<bool>` (module docs).
    pub(crate) const BIT_ARRAY: u64 = 32;
    /// `SplitterDestinationData` (`splitter/splitter_destinations_data.h`).
    pub(crate) const SPLITTER_DESTINATION_DATA: u64 = 264;
    /// `SplitterInfo` (`splitter/splitter_info.h`).
    pub(crate) const SPLITTER_INFO: u64 = 32;
    /// `MemoryPoolInfo` (`memory/memory_pool_info.h`).
    pub(crate) const MEMORY_POOL_INFO: u64 = 32;
    /// `EffectResultState` (`effect/effect_result_state.h`).
    pub(crate) const EFFECT_RESULT_STATE: u64 = 128;
    /// `UpsamplerInfo` (`upsampler/upsampler_info.h`).
    pub(crate) const UPSAMPLER_INFO: u64 = 616;
    /// `EffectInfoBase` (`effect/effect_info_base.h`).
    pub(crate) const EFFECT_INFO_BASE: u64 = 1592;
    /// `SinkInfoBase` (`sink/sink_info_base.h`).
    pub(crate) const SINK_INFO_BASE: u64 = 352;
    /// `PerformanceFrameHeaderVersion1` (`performance/performance_frame_header.h`).
    pub(crate) const PERFORMANCE_FRAME_HEADER_V1: u64 = 24;
    /// `PerformanceDetailVersion1` (`performance/performance_detail.h`).
    pub(crate) const PERFORMANCE_DETAIL_V1: u64 = 16;
    /// `PerformanceEntryVersion1` (`performance/performance_entry.h`).
    pub(crate) const PERFORMANCE_ENTRY_V1: u64 = 16;
    /// `PerformanceFrameHeaderVersion2`.
    pub(crate) const PERFORMANCE_FRAME_HEADER_V2: u64 = 48;
    /// `PerformanceDetailVersion2`.
    pub(crate) const PERFORMANCE_DETAIL_V2: u64 = 24;
    /// `PerformanceEntryVersion2`.
    pub(crate) const PERFORMANCE_ENTRY_V2: u64 = 24;
    /// `AdpcmDataSourceVersion1Command` (`command/data_source/adpcm.h`).
    pub(crate) const ADPCM_V1_COMMAND: u64 = 288;
    /// `PcmInt16DataSourceVersion1Command` (`command/data_source/pcm_int16.h`).
    pub(crate) const PCM_INT16_V1_COMMAND: u64 = 280;
    /// `PcmFloatDataSourceVersion1Command` (`command/data_source/pcm_float.h`).
    pub(crate) const PCM_FLOAT_V1_COMMAND: u64 = 280;
    /// `AdpcmDataSourceVersion2Command`.
    pub(crate) const ADPCM_V2_COMMAND: u64 = 296;
    /// `PcmInt16DataSourceVersion2Command`.
    pub(crate) const PCM_INT16_V2_COMMAND: u64 = 280;
    /// `PcmFloatDataSourceVersion2Command`.
    pub(crate) const PCM_FLOAT_V2_COMMAND: u64 = 280;
    /// `BiquadFilterCommand` (`command/effect/biquad_filter.h`).
    pub(crate) const BIQUAD_FILTER_COMMAND: u64 = 80;
    /// `VolumeRampCommand` (`command/mix/volume_ramp.h`).
    pub(crate) const VOLUME_RAMP_COMMAND: u64 = 40;
    /// `MixRampGroupedCommand` (`command/mix/mix_ramp_grouped.h`).
    pub(crate) const MIX_RAMP_GROUPED_COMMAND: u64 = 328;
    /// `DepopPrepareCommand` (`command/mix/depop_prepare.h`).
    pub(crate) const DEPOP_PREPARE_COMMAND: u64 = 96;
    /// `DepopForMixBuffersCommand` (`command/mix/depop_for_mix_buffers.h`).
    pub(crate) const DEPOP_FOR_MIX_BUFFERS_COMMAND: u64 = 48;
    /// `MixCommand` (`command/mix/mix.h`).
    pub(crate) const MIX_COMMAND: u64 = 40;
    /// `VolumeCommand` (`command/mix/volume.h`).
    pub(crate) const VOLUME_COMMAND: u64 = 40;
    /// `MixRampCommand` (`command/mix/mix_ramp.h`).
    pub(crate) const MIX_RAMP_COMMAND: u64 = 48;
    /// `DeviceSinkCommand` (`command/sink/device.h`).
    pub(crate) const DEVICE_SINK_COMMAND: u64 = 320;
    /// `CircularBufferSinkCommand` (`command/sink/circular_buffer.h`).
    pub(crate) const CIRCULAR_BUFFER_SINK_COMMAND: u64 = 56;
    /// `PerformanceCommand` (`command/performance/performance.h`).
    pub(crate) const PERFORMANCE_COMMAND: u64 = 64;
}

/// The fixed parts of `System::GetWorkBufferSize`, as Eden writes them.
mod fixed {
    /// The alignment of the mix-id and mix-buffer terms and of the first part's end (`0x40`).
    pub(super) const BUFFER_ALIGNMENT: u64 = 0x40;
    /// The alignment of the pointer arrays and the node-state term (`0x10`).
    pub(super) const ARRAY_ALIGNMENT: u64 = 0x10;
    /// What the first part adds at its end (`size += 0x50`).
    pub(super) const FIRST_PART_TAIL: u64 = 0x50;
    /// What the performance term adds before it aligns (`+ 0xC0`).
    pub(super) const PERFORMANCE_TAIL: u64 = 0xC0;
    /// The performance term's alignment (`0x100`).
    pub(super) const PERFORMANCE_ALIGNMENT: u64 = 0x100;
    /// The command buffer's size before revision 5 (`0x18000`).
    pub(super) const FIXED_COMMAND_BUFFER: u64 = 0x18000;
    /// What either command-buffer size adds (`(0x40 - 1) * 2`).
    pub(super) const COMMAND_BUFFER_SLACK: u64 = (0x40 - 1) * 2;
    /// The whole size's alignment (`0x1000`).
    pub(super) const WORK_BUFFER_ALIGNMENT: u64 = 0x1000;
}

/// A renderer parameter (`AudioRendererParameterInternal`), as a title sends it to
/// `GetWorkBufferSize` and `OpenAudioRenderer`: [`PARAMETER_SIZE`] bytes, little-endian, the
/// fields in this order at the offsets each names. Eden names `unk_21`; the others are its names.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RendererParameter {
    /// `0x00 u32 sample_rate`.
    pub sample_rate: u32,
    /// `0x04 u32 sample_count`.
    pub sample_count: u32,
    /// `0x08 u32 mixes`.
    pub mixes: u32,
    /// `0x0C u32 sub_mixes`.
    pub sub_mixes: u32,
    /// `0x10 u32 voices`.
    pub voices: u32,
    /// `0x14 u32 sinks`.
    pub sinks: u32,
    /// `0x18 u32 effects`.
    pub effects: u32,
    /// `0x1C u32 perf_frames`.
    pub perf_frames: u32,
    /// `0x20 u8 voice_drop_enabled`.
    pub voice_drop_enabled: u8,
    /// `0x21 u8 unk_21`.
    pub unknown_21: u8,
    /// `0x22 u8 rendering_device`.
    pub rendering_device: u8,
    /// `0x23 ExecutionMode execution_mode` (a `u8`: `Auto` 0, `Manual` 1).
    pub execution_mode: u8,
    /// `0x24 u32 splitter_infos`.
    pub splitter_infos: u32,
    /// `0x28 s32 splitter_destinations`.
    pub splitter_destinations: i32,
    /// `0x2C u32 external_context_size`.
    pub external_context_size: u32,
    /// `0x30 u32 revision`: a revision number, or a tag such as `REV?` ([`revision_number`]).
    pub revision: u32,
}

impl RendererParameter {
    /// Reads a parameter from `reader`, field by field in layout order; a payload that ends
    /// before the last field is the reader's [`Fault`].
    pub fn read(reader: &mut Reader<'_>) -> Result<Self, Fault> {
        Ok(Self {
            sample_rate: reader.read_u32()?,
            sample_count: reader.read_u32()?,
            mixes: reader.read_u32()?,
            sub_mixes: reader.read_u32()?,
            voices: reader.read_u32()?,
            sinks: reader.read_u32()?,
            effects: reader.read_u32()?,
            perf_frames: reader.read_u32()?,
            voice_drop_enabled: reader.read_u8()?,
            unknown_21: reader.read_u8()?,
            rendering_device: reader.read_u8()?,
            execution_mode: reader.read_u8()?,
            splitter_infos: reader.read_u32()?,
            splitter_destinations: reader.read_i32()?,
            external_context_size: reader.read_u32()?,
            revision: reader.read_u32()?,
        })
    }
}

/// Why [`work_buffer_size`] has no size for a parameter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum WorkBufferError {
    /// The revision is past [`CURRENT_REVISION`] (Eden's `CheckValidRevision`), checked before
    /// any size.
    #[error("revision word {word:#x} reads as revision {number}, past the newest, 15")]
    InvalidRevision {
        /// The parameter's revision word.
        word: u32,
        /// What [`revision_number`] reads it as.
        number: u32,
    },
    /// A term of the size would wrap where Eden's arithmetic wraps (module docs, "Where this
    /// differs from Eden").
    #[error("the work buffer's size does not fit Eden's arithmetic")]
    Overflow,
    /// A term reads a negative splitter destination count (module docs).
    #[error("the splitter destination count {0} is negative")]
    NegativeSplitterDestinations(i32),
}

/// A renderer feature a revision turns on: `CheckFeatureSupported`'s table
/// (`feature_support.h`), for the tags the size reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Feature {
    /// `SupportTags::Splitter`: the node states, edge matrix and splitter count.
    Splitter,
    /// `SupportTags::SplitterBugFix`: the splitter adds a word per destination.
    SplitterBugFix,
    /// `SupportTags::AudioRendererVariadicCommandBufferSize`: the command buffer is sized from
    /// the parameter.
    VariadicCommandBufferSize,
    /// `SupportTags::PerformanceMetricsDataFormatVersion2`: version 2 performance frames.
    PerformanceMetricsVersion2,
    /// `SupportTags::WaveBufferVer2`: version 2 data-source commands.
    WaveBufferVersion2,
    /// `SupportTags::EffectInfoVer2`: an effect result state per effect, twice.
    EffectInfoVersion2,
}

impl Feature {
    /// The first revision with the feature (`{SupportTags::Splitter, 2}` and the rest).
    #[must_use]
    pub const fn since(self) -> u32 {
        match self {
            Self::Splitter => 2,
            Self::SplitterBugFix
            | Self::VariadicCommandBufferSize
            | Self::PerformanceMetricsVersion2 => 5,
            Self::WaveBufferVersion2 => 8,
            Self::EffectInfoVersion2 => 9,
        }
    }

    /// Whether revision number `revision` has the feature (`(*feature).second <=
    /// user_revision`).
    #[must_use]
    pub const fn in_revision(self, revision: u32) -> bool {
        self.since() <= revision
    }
}

/// The revision number a revision word names, as Eden-NX2 reads it (`GetRevisionNum`; module
/// docs, "The revision word"): a word below [`REVISION_TAG_FROM`] is the number itself; any
/// other is `(word - REVISION_MAGIC) >> 24`, the subtraction wrapping at 32 bits.
#[must_use]
pub const fn revision_number(word: u32) -> u32 {
    if word >= REVISION_TAG_FROM {
        word.wrapping_sub(REVISION_MAGIC) >> REVISION_TAG_SHIFT
    } else {
        word
    }
}

/// `a + b`, or [`WorkBufferError::Overflow`] where Eden's 64-bit sum would wrap.
fn add(a: u64, b: u64) -> Result<u64, WorkBufferError> {
    a.checked_add(b).ok_or(WorkBufferError::Overflow)
}

/// `a * b`, or [`WorkBufferError::Overflow`] where Eden's 64-bit product would wrap.
fn mul(a: u64, b: u64) -> Result<u64, WorkBufferError> {
    a.checked_mul(b).ok_or(WorkBufferError::Overflow)
}

/// `Common::AlignUp(value, to)` on a 64-bit value, or [`WorkBufferError::Overflow`] where it
/// would wrap.
fn align_up(value: u64, to: u64) -> Result<u64, WorkBufferError> {
    value
        .checked_next_multiple_of(to)
        .ok_or(WorkBufferError::Overflow)
}

/// A value Eden holds in 32 bits: itself, or [`WorkBufferError::Overflow`] where Eden's would
/// have wrapped.
fn narrow(value: u64) -> Result<u64, WorkBufferError> {
    if value <= u64::from(u32::MAX) {
        Ok(value)
    } else {
        Err(WorkBufferError::Overflow)
    }
}

/// A splitter destination count, where a term reads it as Eden's `s32` converted to a size.
fn destinations(count: i32) -> Result<u64, WorkBufferError> {
    u64::try_from(count).map_err(|_| WorkBufferError::NegativeSplitterDestinations(count))
}

/// `NodeStates::GetWorkBufferSize(count)`: `(AlignUp(count, 0x40) / sizeof(u64)) * 2 + count *
/// sizeof(BitArray) + count * Stack::CalcBufferSize(count)`, where the `AlignUp`, `count * 4`
/// (`CalcBufferSize` returns `u32`) and the last product are 32-bit.
fn node_states_size(count: u64) -> Result<u64, WorkBufferError> {
    let bits = narrow(align_up(count, fixed::BUFFER_ALIGNMENT)?)? / sizes::U64 * 2;
    let stack = narrow(mul(count, sizes::WORD)?)?;
    let stacks = narrow(mul(count, stack)?)?;
    add(add(bits, mul(count, sizes::BIT_ARRAY)?)?, stacks)
}

/// `EdgeMatrix::GetWorkBufferSize(count)`: `AlignUp(count * count, 0x40) / sizeof(u64)`, the
/// product and the `AlignUp` 32-bit.
fn edge_matrix_size(count: u64) -> Result<u64, WorkBufferError> {
    let edges = narrow(mul(count, count)?)?;
    Ok(narrow(align_up(edges, fixed::BUFFER_ALIGNMENT)?)? / sizes::U64)
}

/// `SplitterContext::CalcWorkBufferSize`: nothing before the splitter; then a
/// `SplitterDestinationData` per destination and a `SplitterInfo` per splitter, and with the
/// splitter bug fix a word per destination, aligned to 0x10.
fn splitter_size(params: &RendererParameter, revision: u32) -> Result<u64, WorkBufferError> {
    if !Feature::Splitter.in_revision(revision) {
        return Ok(0);
    }
    let destinations = destinations(params.splitter_destinations)?;
    let mut size = add(
        mul(destinations, sizes::SPLITTER_DESTINATION_DATA)?,
        mul(u64::from(params.splitter_infos), sizes::SPLITTER_INFO)?,
    )?;
    if Feature::SplitterBugFix.in_revision(revision) {
        size = add(
            size,
            align_up(mul(destinations, sizes::WORD)?, fixed::ARRAY_ALIGNMENT)?,
        )?;
    }
    Ok(size)
}

/// `u32` values summed as Eden sums them in 32 bits, or [`WorkBufferError::Overflow`] where any
/// partial sum would wrap (with values that are never negative, that is where the whole sum
/// does).
fn sum32(values: &[u32]) -> Result<u64, WorkBufferError> {
    let mut sum = 0u64;
    for value in values {
        sum = add(sum, u64::from(*value))?;
    }
    narrow(sum)
}

/// `GetRequiredBufferSizeForPerformanceMetricsPerFrame`: a frame header, the detail entries and
/// an entry per voice, effect, sub-mix, sink and the final mix (that count summed in 32 bits),
/// in version 2's sizes from [`Feature::PerformanceMetricsVersion2`] on, else version 1's.
fn performance_frame_size(
    params: &RendererParameter,
    revision: u32,
) -> Result<u64, WorkBufferError> {
    let entries = sum32(&[
        params.voices,
        params.effects,
        params.sub_mixes,
        params.sinks,
        1,
    ])?;
    let (header, detail, entry) = if Feature::PerformanceMetricsVersion2.in_revision(revision) {
        (
            sizes::PERFORMANCE_FRAME_HEADER_V2,
            sizes::PERFORMANCE_DETAIL_V2,
            sizes::PERFORMANCE_ENTRY_V2,
        )
    } else {
        (
            sizes::PERFORMANCE_FRAME_HEADER_V1,
            sizes::PERFORMANCE_DETAIL_V1,
            sizes::PERFORMANCE_ENTRY_V1,
        )
    };
    add(
        add(header, mul(MAX_DETAIL_ENTRIES, detail)?)?,
        mul(entries, entry)?,
    )
}

/// `CommandGenerator::CalculateCommandBufferSize`: the largest command each part of a frame can
/// generate, per effect, voice, mix buffer, splitter destination and sink, and a performance
/// command per node and detail entry.
fn command_buffer_size(params: &RendererParameter, revision: u32) -> Result<u64, WorkBufferError> {
    let voices = u64::from(params.voices);
    let data_source = if Feature::WaveBufferVersion2.in_revision(revision) {
        sizes::ADPCM_V2_COMMAND
            .max(sizes::PCM_INT16_V2_COMMAND)
            .max(sizes::PCM_FLOAT_V2_COMMAND)
    } else {
        sizes::ADPCM_V1_COMMAND
            .max(sizes::PCM_INT16_V1_COMMAND)
            .max(sizes::PCM_FLOAT_V1_COMMAND)
    };
    let voice = data_source
        + sizes::BIQUAD_FILTER_COMMAND * u64::from(MAX_BIQUAD_FILTERS)
        + sizes::VOLUME_RAMP_COMMAND
        + sizes::MIX_RAMP_GROUPED_COMMAND;
    let mix_buffers = u64::from(MAX_MIX_BUFFERS);

    let mut size = mul(u64::from(params.effects), sizes::EFFECT_INFO_BASE)?;
    let per_voice = add(
        mul(
            u64::from(params.splitter_infos),
            sizes::DEPOP_PREPARE_COMMAND,
        )?,
        voice,
    )?;
    size = add(size, mul(voices, per_voice)?)?;
    size = add(
        size,
        sizes::DEPOP_FOR_MIX_BUFFERS_COMMAND + sizes::MIX_COMMAND * mix_buffers * mix_buffers,
    )?;
    size = add(
        size,
        sizes::DEPOP_FOR_MIX_BUFFERS_COMMAND + sizes::VOLUME_COMMAND * mix_buffers,
    )?;
    size = add(
        size,
        mul(
            mul(
                destinations(params.splitter_destinations)?,
                sizes::MIX_RAMP_COMMAND,
            )?,
            mix_buffers,
        )?,
    )?;
    size = add(
        size,
        mul(
            u64::from(params.sinks),
            sizes::DEVICE_SINK_COMMAND.max(sizes::CIRCULAR_BUFFER_SINK_COMMAND),
        )?,
    )?;
    let nodes = sum32(&[
        params.effects,
        params.voices,
        params.sinks,
        params.sub_mixes,
        1,
    ])?;
    add(
        size,
        mul(add(nodes, MAX_DETAIL_ENTRIES)?, sizes::PERFORMANCE_COMMAND)?,
    )
}

/// The work-buffer size a renderer with `params` needs: Eden's `Manager::GetWorkBufferSize`,
/// the revision check and then `System::GetWorkBufferSize` (module docs).
///
/// [`WorkBufferError::InvalidRevision`] comes first, before any term is computed; then
/// [`WorkBufferError::NegativeSplitterDestinations`] or [`WorkBufferError::Overflow`], whichever
/// term meets its fault first in Eden's order.
pub fn work_buffer_size(params: &RendererParameter) -> Result<u64, WorkBufferError> {
    let revision = revision_number(params.revision);
    if revision > CURRENT_REVISION {
        return Err(WorkBufferError::InvalidRevision {
            word: params.revision,
            number: revision,
        });
    }
    let mixes = u64::from(params.mixes);
    let sub_mixes = u64::from(params.sub_mixes);
    let voices = u64::from(params.voices);
    let sinks = u64::from(params.sinks);
    let effects = u64::from(params.effects);
    let effect_info_2 = Feature::EffectInfoVersion2.in_revision(revision);

    let mut size = align_up(mul(mixes, sizes::WORD)?, fixed::BUFFER_ALIGNMENT)?;
    let sub_mix_effects = narrow(mul(sub_mixes, u64::from(MAX_EFFECTS))?)?;
    size = add(size, mul(sub_mix_effects, sizes::WORD)?)?;
    let sub_mix_count = narrow(add(sub_mixes, 1)?)?;
    size = add(size, mul(sub_mix_count, sizes::MIX_INFO)?)?;
    let voice = sizes::VOICE_INFO + sizes::VOICE_CHANNEL_RESOURCE + sizes::VOICE_STATE;
    size = add(size, mul(voices, voice)?)?;
    size = add(
        size,
        align_up(mul(sub_mix_count, sizes::POINTER)?, fixed::ARRAY_ALIGNMENT)?,
    )?;
    size = add(
        size,
        align_up(mul(voices, sizes::POINTER)?, fixed::ARRAY_ALIGNMENT)?,
    )?;
    let sinks_and_sub_mixes = narrow(add(sinks, sub_mixes)?)?;
    let mix_samples = narrow(mul(sinks_and_sub_mixes, u64::from(TARGET_SAMPLE_COUNT))?)?;
    let per_buffer = add(
        mul(mix_samples, sizes::WORD)?,
        mul(u64::from(params.sample_count), sizes::WORD)?,
    )?;
    let buffers = narrow(add(mixes, u64::from(MAX_CHANNELS))?)?;
    size = add(
        size,
        align_up(mul(per_buffer, buffers)?, fixed::BUFFER_ALIGNMENT)?,
    )?;

    if Feature::Splitter.in_revision(revision) {
        let graph = add(
            node_states_size(sub_mix_count)?,
            edge_matrix_size(sub_mix_count)?,
        )?;
        size = add(size, align_up(graph, fixed::ARRAY_ALIGNMENT)?)?;
    }
    size = add(size, splitter_size(params, revision)?)?;
    let wave_buffers = narrow(mul(voices, u64::from(MAX_WAVE_BUFFERS))?)?;
    let pools = narrow(add(effects, wave_buffers)?)?;
    size = add(size, mul(pools, sizes::MEMORY_POOL_INFO)?)?;
    if effect_info_2 {
        size = add(size, mul(effects, sizes::EFFECT_RESULT_STATE)?)?;
    }
    size = add(size, fixed::FIRST_PART_TAIL)?;
    size = align_up(size, fixed::BUFFER_ALIGNMENT)?;

    size = add(size, mul(sinks_and_sub_mixes, sizes::UPSAMPLER_INFO)?)?;
    size = add(size, mul(effects, sizes::EFFECT_INFO_BASE)?)?;
    size = add(
        size,
        align_up(mul(voices, sizes::VOICE_STATE)?, fixed::BUFFER_ALIGNMENT)?,
    )?;
    size = add(size, mul(sinks, sizes::SINK_INFO_BASE)?)?;
    if effect_info_2 {
        size = add(size, mul(effects, sizes::EFFECT_RESULT_STATE)?)?;
    }

    if params.perf_frames > 0 {
        let frame = performance_frame_size(params, revision)?;
        let frames = narrow(add(u64::from(params.perf_frames), 1)?)?;
        let performance = add(mul(frame, frames)?, fixed::PERFORMANCE_TAIL)?;
        size = add(size, align_up(performance, fixed::PERFORMANCE_ALIGNMENT)?)?;
    }

    let commands = if Feature::VariadicCommandBufferSize.in_revision(revision) {
        command_buffer_size(params, revision)?
    } else {
        fixed::FIXED_COMMAND_BUFFER
    };
    size = add(size, add(commands, fixed::COMMAND_BUFFER_SLACK)?)?;
    align_up(size, fixed::WORK_BUFFER_ALIGNMENT)
}
