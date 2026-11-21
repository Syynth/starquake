//! Slice 13: the constant-buffer and texture pass (`docs/specs/shader-pipeline.md`, "Passes
//! needed for correctness", items 2 and 3).
//!
//! It runs on a function after the SSA rewrite, identity removal and constant propagation
//! (slices 8 and 8b; Eden's order, `frontend/maxwell/translate_program.cpp` @ cf654052, line
//! 389 onwards), on a structured function or a control-flow graph, and in two halves that
//! follow the question-and-answer exchange ([`crate::exchange`]):
//!
//! 1. [`scan`] reads the function and the program's texture descriptor buffer
//!    ([`DescriptorBuffer`], slice 12f) and traces every texture operation's handle to the
//!    constant-buffer words it is built from. It changes nothing, and every refusal of a
//!    handle happens here. The [`TextureScan`] it gives is a [`Questioner`]: it asks for each
//!    handle's words ([`Question::CbufWord`]), and once they are answered, what the handle
//!    points to ([`Question::TextureKind`], with [`TextureUsage::Image`] for a storage image
//!    and [`TextureUsage::Sampled`] for every other use).
//! 2. [`run`] takes the answers, builds the [`ResourceTable`] and checks every answer it needs
//!    before it changes the function; then it rewrites each texture operation (below) and
//!    bounds each indexed constant-buffer read ("Constant-buffer reads"). Every refusal from
//!    `run` except [`ResourceError::Verify`] leaves the function as it was (below, "When `run`
//!    refuses").
//!
//! # Sources
//!
//! Ported by behaviour from Eden's `eden/src/shader_recompiler/ir_opt/texture_pass.cpp` @
//! cf654052 (GPL-3.0-or-later over GPL-2.0-or-later yuzu code): `TexturePass`, `MakeInst`,
//! `Track`, `TryGetConstBuffer`, `TrackUniformIndex`, `DynamicDescriptorSizeShift`,
//! `DynamicDescriptorCount`, `IndexedInstruction`, `IsStorageImageOpcode`, `IsBindless`,
//! `GetTextureHandle` and the `Descriptors` class; the descriptor structs are
//! `shader_info.h`'s, `ConstBufferAddr` is `environment.h`'s. Compared line by line with PR
//! #4588's version (`eden-pr4588/src/shader_recompiler/ir_opt/texture_pass.cpp` @ e5fb3b97,
//! GPL-3.0-or-later); "Where the two references differ" lists every difference and which is
//! followed. The constant-buffer bound follows Eden's SPIR-V `GetCbuf`
//! (`backend/spirv/emit_spirv_context_get_set.cpp`), see "Constant-buffer reads".
//!
//! # Handle forms
//!
//! Eden's translators emit `BoundImage*` for a handle in the texture constant buffer and
//! `BindlessImage*` for a handle in a register, and the pass turns both into `Image*`
//! ([`classify`](opcodes::classify), Eden's `IndexedInstruction`). The texture translator
//! here (slice 12f) emits `Image*` itself, with an immediate byte offset for a constant-buffer
//! handle and a value for a register handle (`crate::translate::texture`, "Handles"). The pass
//! takes all three forms: `BoundImage*` must carry an immediate `u32`; `BindlessImage*` is
//! traced; `Image*` with an immediate `u32` is a bound handle and with a value is traced.
//!
//! **A listed divergence**: an `Image*` whose register handle became an immediate (a register
//! loaded with a constant, folded by the SSA rewrite and constant propagation, or `RZ`, which
//! reads as 0) is read here as a bound handle at that offset, where Eden's `BindlessImage*`
//! would be refused ("Failed to track bindless texture constant buffer"). The two cannot be
//! told apart after SSA; making the translator emit Eden's two forms removes it, and the pass
//! already takes them (pinned by `tests::forms::a_folded_register_handle_reads_as_bound`).
//!
//! A bound handle is read from the program's descriptor buffer
//! ([`DescriptorBuffer::index`], Eden's `TextureBoundBuffer`); a bound handle in a program
//! with none recorded is [`ResourceError::NoDescriptorBuffer`].
//!
//! # Tracing is exact
//!
//! Eden finds a bindless handle's source with a breadth-first search that returns the first
//! operation anywhere in the handle's operand tree that its `TryGetConstBuffer` matches, so a
//! handle computed from a constant-buffer word by an operation it does not know (an add, a
//! select, a phi of two words) binds the word as if it were the handle. Here the handle must be
//! exactly one of these forms, each operand followed through `Identity`:
//!
//! - `GetCbufU32` at an immediate bank below [`CBUF_BANKS`](crate::translate::float::CBUF_BANKS)
//!   and an immediate offset (aligned down to a word, as the read is); or at a register offset,
//!   an array of handles: `IAdd32` of an immediate base and the index, or the index alone
//!   (base 0). The array's spacing is [`dynamic_size_shift`](trace::dynamic_size_shift), its
//!   length [`dynamic_count`](trace::dynamic_count), and an index that is itself a
//!   constant-buffer word, possibly shifted, is kept as the [`IndexWord`] the GPU crate reads
//!   when drawing (Eden's `TrackUniformIndex`).
//! - `CompositeExtractU32x2` (element 0 or 1) of a `GetCbufU32x2` at an immediate bank and
//!   offset: the word that element is.
//! - `ShiftLeftLogical32` of a traced word by an immediate below 32; `BitwiseAnd32` of a
//!   traced word and an immediate mask; `BitwiseOr32` of two traced words, which Eden's rules
//!   order into a primary and a secondary word ([`HandleSource`]).
//!
//! Anything else is refused with the exact reason ([`Untraced`]), named by the operation where
//! the trace stopped. The trace descends at most [`MAX_TRACE_DEPTH`] operations, checking the
//! depth before it charges a step, and an OR below an OR is refused at once, so tracing one
//! handle takes at most `2 * MAX_TRACE_DEPTH + 1` steps, accepted or refused: an OR and two
//! chains down to the limit (pinned by
//! `tests::trace::the_longest_trace_takes_the_documented_steps`, and for a refused trace one
//! step past the limit by `tests::trace::a_refused_trace_takes_the_documented_steps`; the
//! limit's edge by `tests::trace::the_depth_limit_at_its_edge`, and through `Identity` by
//! `tests::trace::an_identity_chain_counts_towards_the_depth`).
//! Because the handle is computed exactly, the handle the descriptor's words give is the
//! handle the IR computes from them (`tests::trace::descriptor_handles_match_the_interpreter`).
//!
//! # What each operation becomes
//!
//! Each texture operation, visited in post order of the blocks (Eden's `post_order_blocks`,
//! here [`crate::opt`]'s block order reversed, every block included) and then sorted by the
//! primary word's bank and offset (stably: Eden's `std::ranges::sort` leaves equal keys in an
//! unspecified order), is turned into its `Image*` form with `descriptor_index` set to its
//! entry in the list of its kind:
//!
//! - `ImageRead`, `ImageWrite` and the image atomics take an [`ImageDescriptor`], or an
//!   [`ImageBufferDescriptor`] for a buffer, with the instruction's format and the answer's
//!   header format; a handle of two words is [`ResourceError::SeparateSampler`] (Eden:
//!   "Unexpected separate sampler").
//! - Every other operation takes a [`TextureDescriptor`], or a [`TextureBufferDescriptor`] for
//!   a buffer, and a sampled texture sets `is_integer` from the answer's component type.
//! - `ImageFetch` of a 2D, rect or 2D-array texture is multisampled when its sample argument
//!   is not [`ABSENT`]; of any other type its sample argument becomes [`ABSENT`] (Eden empties
//!   it); a 1D fetch whose answer is a buffer becomes a buffer fetch (Eden: "Fire Emblem: Three
//!   Houses").
//! - `ImageQueryDimensions` of a handle whose answer is a buffer becomes a buffer query.
//! - The first argument becomes the array index, `UMin32(ShiftRightLogical32(offset,
//!   size_shift), count - 1)`, inserted before the operation, for an array of more than one
//!   handle, and [`ABSENT`] otherwise. An array whose index is an [`IndexWord`] keeps one
//!   handle in a sampled texture's descriptor (Eden: "A uniform index is resolved when drawing,
//!   so a single descriptor is enough"); the other kinds drop the index word and index the
//!   array at run time, as Eden's descriptors of those kinds have no index fields.
//!
//! Descriptors merge as Eden's `Descriptors::Add` merges them ([`ResourceTable`]).
//!
//! # When `run` refuses
//!
//! [`run`] makes every check that can refuse before its first change: for each scanned
//! operation, that the function still holds it ([`ResourceError::Stale`], below), its flags,
//! its answers and its descriptor index; then that the arena has room for every instruction it
//! adds. After those, adding an instruction cannot fail (its block was checked, and the room
//! was), so a refusal of any of these kinds leaves the function as it was
//! (`tests::refusals::a_storage_image_of_two_words_is_refused_and_nothing_changes`,
//! `tests::refusals::a_changed_function_is_stale`). The exception is
//! [`ResourceError::Verify`], in test and debug builds only: it is reported after the rewrite,
//! which has been applied.
//!
//! The stale check compares what the scan recorded of each texture operation: that the
//! function holds an operation at its id with the scanned opcode and the scanned handle
//! argument, and that the scanned block lists that id
//! (`tests::refusals::a_changed_handle_is_stale`, `tests::refusals::a_moved_operation_is_stale`).
//! It does not look below the handle argument: a function whose handle's operand tree changed
//! below its first operation (the constant-buffer read rewritten in place, say) is not refused,
//! and the descriptor is built from the words the scan traced
//! (`tests::refusals::a_change_below_the_handle_is_not_seen`). A scan is meant for one `run`:
//! running a second time with the same scan is refused as stale where the first run changed a
//! texture operation's handle argument (every handle but a bound one at offset 0, which the
//! rewrite's [`ABSENT`] equals; `tests::refusals::a_second_run_with_one_scan_is_stale`), and
//! otherwise bounds each indexed constant-buffer read a second time.
//!
//! # Constant-buffer reads
//!
//! Every constant-buffer read keeps its opcode and its immediate bank, which names the host
//! binding: the host model has 32 slots, banks 0 to 31 (spec, item 2; slice 15's
//! [`crate::info`] lists the banks a program reads and their sizes, and refuses a bank past
//! the last). A read at a register offset reads a binding whose size is the whole bank,
//! [`INDEXED_CBUF_SIZE`](crate::info::INDEXED_CBUF_SIZE) (slice 15, Eden's `size = 0x10'000`), and is bounded by it: the
//! offset is compared with that size, the read happens at the offset where it is smaller and
//! at 0 elsewhere, and the result is zero where it is not. Because every read width (1, 2, 4
//! or 8 bytes) divides [`INDEXED_CBUF_SIZE`] and the read aligns its offset down to its width,
//! "offset below [`INDEXED_CBUF_SIZE`]" is exactly "the whole read inside the binding"
//! (`tests::cbuf::every_width_divides_the_bound`). Reads at an immediate offset are left as
//! they are; slice 15 sizes their binding to cover them.
//!
//! The zero past the bound is a **guess** at the hardware's answer: Eden selects zero the same
//! way only where the host's robust buffer access is broken (`GetCbuf`,
//! `profile.has_broken_robust`, comparing the element index with `0xFFFF`) and otherwise
//! leaves the read to the host's robustness; Eden's translation-time reads give zero past a
//! bank's bound size too (`video_core/shader_environment.cpp`, `ReadCbufValue`: the word is read
//! only `if (cbuf_offset < cbuf.size)`). The read itself is clamped as well, so no host
//! bounds-check policy is needed to keep it inside the binding (spec, "Untrusted input").
//! **A departure from the approved spec, and an open owner question.** `shader-pipeline.md`,
//! "Passes needed for correctness", item 2, says: "Reads with a register offset (indexed) are
//! bounds-checked against the bound size." The bound here is the host binding's size,
//! [`INDEXED_CBUF_SIZE`](crate::info::INDEXED_CBUF_SIZE), not the size the guest bound the bank
//! with, which only the draw knows. The two options: keep the host binding's size (what this
//! pass does: no per-draw state, and the read stays inside the binding), or bound by the
//! guest's size, which needs a driver constant per bank that the GPU crate fills when drawing.
//! Every indexed read takes this path (`tests::cbuf::reads_in_every_block_are_bounded`).
//!
//! # Where the two references differ
//!
//! Decision 8 of the graphics-server spec ("where one answer contains the other, the larger;
//! otherwise the one a title ran on") picks the winner of each difference. Eden mainline
//! accepts a superset of PR #4588's handles in each row marked "larger". Where neither answer
//! contains the other, the one a title ran on is PR #4588's: `shader-pipeline.md` (the slice
//! 12e paragraph) counts PR #4588, which runs titles through its lowering to Maxwell and then
//! its own `texture_pass.cpp`, as the one titles ran on. The rows marked **departure** do not
//! follow that winner; each says why and is an open **owner question**.
//!
//! | What | Eden @ cf654052 | PR #4588 @ e5fb3b97 | Here | Pinned by |
//! |---|---|---|---|---|
//! | An index that is a constant-buffer word (`TrackUniformIndex`) | kept, one descriptor | not tracked, the array indexed at run time | Eden's (larger) | `tests::table::a_uniform_index_keeps_one_descriptor` |
//! | The smallest array spacing (`DynamicDescriptorSizeShift`) | 4 bytes (shift 2: "Separate texture and sampler handles are 4 bytes apart") | 8 bytes (shift 3) | Eden's (larger) | `tests::trace::the_smallest_spacing_is_four_bytes` |
//! | A register offset that is not an `IAdd32` | an array from offset 0 | refused | Eden's (larger) | `tests::table::an_index_alone_starts_at_zero` |
//! | `BitwiseOr32` of an array and a single word | the array is the primary word | refused (`count > 1 \|\| count > 1`) | Eden's (larger) | `tests::table::an_array_or_a_single_word` |
//! | `BitwiseAnd32` with an immediate | the mask kept, applied after the shift | the shift set to the mask's trailing zero count, the mask dropped | **conflict, departure**: decision 8 picks PR #4588's (the one a title ran on); Eden's is kept because it computes the handle the IR computes and the PR's reading does not (with the mask `0xfff00` the IR computes `word & 0xfff00`, the PR's reading `word << 8`). **Owner question**: keep Eden's, or follow decision 8 and take the PR's | `tests::trace::a_mask_is_applied_after_the_shift` |
//! | A single word shifted left, with no second word | `(word << shift) & mask` | `(word << shift) \| word`: the PR ORs the primary word into itself when there is no secondary | **conflict, departure**: decision 8 picks PR #4588's (the one a title ran on); Eden's is kept because it computes the handle the IR computes and the PR's reading does not. **Owner question**: keep Eden's, or follow decision 8 and take the PR's | `tests::table::a_shifted_single_word` |
//! | Handles loaded from global or storage memory | not matched: the search goes on into the load's address, binding a constant-buffer word found there or failing ("Failed to track bindless texture constant buffer") | refused by name | refused, with the PR's reason ([`Untraced::GlobalMemory`]). This follows from exact tracing ("Tracing is exact", owner question 1 of the PR), which refuses every operation it does not follow, not from decision 8 | `tests::refusals::every_memory_load_is_refused_by_name` |
//! | Buffer-backed textures (PR's `buffer_texture.h`, `BufferTexture::Tracker`) | none | traced through storage buffers, host-dependent | not ported: refused as untraceable (decision 8) | `tests::refusals::every_memory_load_is_refused_by_name` |
//! | A handle from a vertex attribute (`IsAttributeSourcedHandle`) | accepted where the host sets `support_attribute_texture_handle` | none | **departure**: refused, [`Untraced::Attribute`]. Decision 8's "larger" picks Eden's, which accepts it on such hosts; it is refused because the host feature it depends on is not modelled here (the pass does not know the host, as with the SNORM patch below). **Owner question**: refuse, or accept it and leave the host check to the render backend | `tests::refusals::every_reason_reads_as_written` |
//!
//! # Where this differs from Eden
//!
//! | What | Eden | Here | Pinned by |
//! |---|---|---|---|
//! | Tracing | breadth-first, the first matching operation anywhere in the tree | exact forms ("Tracing is exact"); a refusal names the operation | `tests::refusals::an_add_after_the_read_is_refused` |
//! | A phi in the handle | `ResolveConditionalPhi` picks one side of a phi under a conditional header | refused, [`Untraced::Phi`] | `tests::refusals::a_phi_is_refused` |
//! | A shift of a shifted word | the second shift replaces the first | refused, [`Untraced::ShiftOfShifted`] | `tests::refusals::every_reason_reads_as_written` |
//! | A shift or mask of a two-word handle | applied to the primary word only | refused, [`Untraced::ShiftOfPair`], [`Untraced::MaskOfPair`] | same |
//! | An OR of an OR | refused, then the search goes on below it | refused, [`Untraced::MoreThanTwoWords`] | same |
//! | A shift by 32 or more | undefined (`u32 << 32` in C++) | refused, [`Untraced::ShiftTooWide`] | `tests::trace::shift_widths_at_the_edge` |
//! | A mask read from constant buffer 1 (`TryGetConstant`) | the word read as the mask | refused, [`Untraced::MaskNotImmediate`] | `tests::refusals::every_reason_reads_as_written` |
//! | A bank at or past 32 | accepted | refused, [`Untraced::BankPastLast`]; an index word there is not kept | `tests::trace::banks_at_the_edge` |
//! | Word offsets | as written (an unaligned offset reads four bytes from there) | aligned down to a word, as the IR's read is; a pair element from the eight-byte-aligned read | `tests::trace::offsets_are_aligned_as_the_read_is` |
//! | An element of a `GetCbufU32x2` | the read's offset, whichever element | the element's word | `tests::trace::offsets_are_aligned_as_the_read_is` |
//! | Trace depth | unbounded (a visited list) | [`MAX_TRACE_DEPTH`] | `tests::trace::the_depth_limit_at_its_edge` |
//! | The per-host caps on an array's length (`DynamicSampledTextureCap`, the host's descriptor limits) | applied | not applied: only [`DESCRIPTOR_MAX_COUNT`]; the render backend checks the host's limits | `tests::table::array_lengths_at_the_edge` |
//! | An image descriptor's source | bank and offset only | the whole [`HandleSource`] (shift and mask kept) | `tests::table::golden_tables` |
//! | A texel buffer descriptor's component type | none | the answer's | `tests::table::golden_tables` |
//! | `ImageQueryDimensions`' type | the header's type (`ReadTextureType`) | the instruction's, or buffer when the answer is a buffer: the answer has no other type ([`crate::exchange`], "What the questions do not cover") | `tests::rewrite::a_buffer_query_and_fetch` |
//! | A 2D sample of a rect texture (`PatchImageSampleImplicitLod`) | coordinates divided by the size | not patched: the answer does not say rect | same as the row above |
//! | A SNORM texel-buffer fetch (`PatchTexelFetch`) | patched where the host cannot render SNORM buffers | not patched: the host is not known here, and a sampled answer has no format ([`crate::exchange`]) | none: no input reaches it |
//! | An unanswered or missing word or header | a word past the bank's size reads 0 (`ReadCbufValue`), an unbound bank is an assertion, and a header is read from guest memory whatever it holds | refused: [`ResourceError::UnansweredWord`], [`ResourceError::NoCbufWord`], [`ResourceError::UnansweredKind`], [`ResourceError::NoTextureHeader`] (**owner question**: refuse, or read a missing word as 0 as Eden does) | `tests::answers::missing_answers_are_refused` |
//! | The answer's `depth` and `multisampled` | none | not read: the descriptor takes both from the instruction, as Eden does ("TODO: Read this from TIC") | none |
//!
//! # Bindless handles in uniform registers (the review of #201)
//!
//! A texture handle in a uniform register is that register's 32-bit value, odd or even: PR
//! #4588's `TEX`, `TLD` and `TLD4` take any `handle_register < 63`, refusing only `URZ`
//! (`eden-pr4588/src/shader_recompiler/frontend/ampere/ops_texture.cpp` @ e5fb3b97, the
//! check whose message is "native bindless texture handle"). Slice 12f makes that refusal
//! ([`crate::translate::texture::UNIFORM_HANDLE_LIMIT`]), and after SSA the register's value
//! is traced like any other. A bindless constant-buffer load (`LDC` `0x582`) stays refused by
//! slice 12e, as the PR refuses it, so no handle reaches this pass from one.
//!
//! # Cost
//!
//! [`scan`] takes the steps of [`crate::opt`]'s block order, one step per listed instruction
//! and one per trace step, at most `2 * MAX_TRACE_DEPTH + 1` per texture operation (pinned by
//! `tests::trace::the_depth_limit_at_its_edge` for one fixed input). Not charged: the
//! `Identity` links the trace follows below an operation it reached (the loop in the trace's
//! `resolve`, at most [`MAX_TRACE_DEPTH`] links per call), called at most twice per trace (the
//! vertex-attribute check) and at most four times per trace step (a constant-buffer read at a
//! register offset: its offset, the spacing and the index word, twice), so a trace follows at
//! most `(2 + 4 * (2 * MAX_TRACE_DEPTH + 1)) * MAX_TRACE_DEPTH` uncharged links. [`run`] takes one step per
//! texture operation, one per listed instruction, and one per instruction of each block it adds
//! to (`tests::cbuf::reads_in_every_block_are_bounded`); it adds at most two instructions per
//! texture operation and seven per indexed constant-buffer read, and merging descriptors is a
//! hash lookup, so time and memory grow linearly with the function.

mod cbuf;
pub(crate) mod opcodes;
mod table;
#[cfg(test)]
mod tests;
mod trace;

use std::collections::{BTreeMap, HashMap};
use std::fmt::{self, Display, Formatter};

use thiserror::Error;

use table::TableBuilder;
pub use table::{
    HandleSource, HandleWord, ImageBufferDescriptor, ImageDescriptor, IndexWord, ResourceTable,
    TextureBufferDescriptor, TextureDescriptor,
};
pub use trace::Untraced;

use crate::exchange::{Answers, Question, Questioner, SampledType, TextureKind, TextureUsage};
use crate::ir::{
    BlockId, BuildError, Flags, Function, Imm, Inst, Opcode, Operand, Operation, TextureInfo,
    TextureType, Value, VerifyError,
};
use crate::opt::{Work, arena_len, block_order};
use crate::translate::texture::{ABSENT, DescriptorBuffer};
use opcodes::Form;
use trace::{Traced, imm_u32};

/// The bytes between handles of an array whose spacing is not known (Eden's `DESCRIPTOR_SIZE`).
pub const DESCRIPTOR_SIZE: u32 = 8;
/// `log2(`[`DESCRIPTOR_SIZE`]`)` (Eden's `DESCRIPTOR_SIZE_SHIFT`).
pub const DESCRIPTOR_SIZE_SHIFT: u32 = DESCRIPTOR_SIZE.trailing_zeros();
/// The most handles an array indexed at run time holds (Eden's `DESCRIPTOR_MAX_COUNT`).
pub const DESCRIPTOR_MAX_COUNT: u32 = 1024;
/// The smallest spacing shift an index's shift is taken as (Eden's `size_shift >= 2`: "Separate
/// texture and sampler handles are 4 bytes apart").
pub const MIN_DYNAMIC_SIZE_SHIFT: u32 = 2;
/// The bytes of constant buffer an array may start within: Eden's `max_cbuf_bytes = 16 *
/// descriptor_limit`, with the limit [`DESCRIPTOR_MAX_COUNT`]. An array starting at or past it
/// holds one handle.
pub const DYNAMIC_ARRAY_BYTES: u32 = 16 * DESCRIPTOR_MAX_COUNT;
/// How many operations below the handle the trace descends. A **guess** (Eden has no limit):
/// the deepest form traced here, an OR of a shifted and masked word with another, is four
/// deep, and the limit only bounds a chain a hostile program builds.
pub const MAX_TRACE_DEPTH: u32 = 16;

/// Whether [`run`] verifies its output: in tests and debug builds, as [`crate::opt::run`]
/// does.
const VERIFY: bool = cfg!(any(test, debug_assertions));

/// A usage in words.
const fn usage_name(usage: TextureUsage) -> &'static str {
    match usage {
        TextureUsage::Sampled => "sampled",
        TextureUsage::Image => "image",
    }
}

/// Why the pass refused a function.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum ResourceError {
    /// A bindless handle could not be traced (decision 8).
    #[error("{at}: the texture handle cannot be traced: {why}")]
    Untraced {
        /// The texture operation.
        at: Value,
        /// Why.
        why: Untraced,
    },
    /// A bound handle in a program whose translation recorded no texture descriptor buffer.
    #[error("{0}: a bound texture handle, but the program has no texture descriptor buffer")]
    NoDescriptorBuffer(Value),
    /// A texture operation with its handle, or its texture information, missing or of the
    /// wrong kind.
    #[error("{0}: a texture operation with an argument or its flags missing or of the wrong kind")]
    Malformed(Value),
    /// The function no longer holds, at this id, an operation with the opcode and handle
    /// argument the scan found, listed in the block the scan found it in (module docs, "When
    /// `run` refuses": changes below the handle argument are not seen).
    #[error("{0}: the function changed since it was scanned")]
    Stale(Value),
    /// A handle's word was not asked about.
    #[error("constant buffer {bank} at {offset:#x} was not answered")]
    UnansweredWord {
        /// The bank.
        bank: u8,
        /// The byte offset.
        offset: u32,
    },
    /// Engine state had no word where a handle is read.
    #[error("constant buffer {bank} has no word at {offset:#x}")]
    NoCbufWord {
        /// The bank.
        bank: u8,
        /// The byte offset.
        offset: u32,
    },
    /// What a handle points to was not asked about.
    #[error("texture handle {handle:#x} ({usage}) was not answered", usage = usage_name(*usage))]
    UnansweredKind {
        /// The handle.
        handle: u32,
        /// The usage asked about.
        usage: TextureUsage,
    },
    /// A handle points to no valid texture header.
    #[error("texture handle {handle:#x} points to no texture header")]
    NoTextureHeader {
        /// The handle.
        handle: u32,
    },
    /// A storage image operation whose handle is built from two words (Eden: "Unexpected
    /// separate sampler").
    #[error("{0}: a storage image whose handle is built from two words")]
    SeparateSampler(Value),
    /// A descriptor list grew past what `descriptor_index` (16 bits) can name.
    #[error("{0}: more texture descriptors of one kind than 65536")]
    TooManyDescriptors(Value),
    /// An instruction could not be added.
    #[error(transparent)]
    Build(#[from] BuildError),
    /// The output does not verify.
    #[error("after the constant-buffer and texture pass: {0}")]
    Verify(VerifyError),
}

/// One texture operation the scan found.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Entry {
    /// The operation.
    at: Value,
    /// Its block.
    block: BlockId,
    /// Its opcode when scanned.
    opcode: Opcode,
    /// Its handle argument when scanned.
    handle: Operand,
    /// The `Image*` form it becomes.
    unbound: Opcode,
    /// Where its handle is found.
    traced: Traced,
    /// How it uses the texture.
    usage: TextureUsage,
}

/// What [`scan`] found: each texture operation with its handle's source, in Eden's order.
/// Asks for the answers [`run`] needs ([`Questioner`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TextureScan {
    entries: Vec<Entry>,
    work: Work,
}

impl TextureScan {
    /// The work the scan did.
    pub const fn work(&self) -> Work {
        self.work
    }

    /// Each texture operation found, with its handle's source, in the order [`run`] numbers
    /// descriptors.
    pub fn sources(&self) -> impl Iterator<Item = (Value, HandleSource)> + '_ {
        self.entries.iter().map(|e| (e.at, e.traced.source))
    }
}

impl Questioner for TextureScan {
    /// Every handle's words; then, for each handle whose words are all answered with a word,
    /// what it points to for its usage.
    fn questions(&self, answers: &Answers) -> Vec<Question> {
        let mut out = Vec::new();
        for entry in &self.entries {
            for word in entry.traced.source.words() {
                out.push(Question::CbufWord {
                    bank: word.bank(),
                    offset: word.offset(),
                });
            }
        }
        for entry in &self.entries {
            if let Some(handle) = entry
                .traced
                .source
                .handle(|bank, offset| answers.cbuf_word(bank, offset).flatten())
            {
                out.push(Question::TextureKind {
                    handle,
                    usage: entry.usage,
                });
            }
        }
        out
    }
}

/// Finds every texture operation of `function` and traces its handle; `buffer` is the
/// program's texture descriptor buffer, which bound handles are read from. Changes nothing.
pub fn scan(function: &Function, buffer: DescriptorBuffer) -> Result<TextureScan, ResourceError> {
    let mut work = Work::default();
    let mut order = block_order(function, &mut work);
    order.reverse();
    let mut entries = Vec::new();
    for block in order {
        let Some(listed) = function.block(block) else {
            continue;
        };
        for &at in &listed.insts {
            work.step();
            let Some(Inst::Op(op)) = function.inst(at) else {
                continue;
            };
            let Some((form, unbound)) = opcodes::classify(op.opcode) else {
                continue;
            };
            let handle = *op.args.first().ok_or(ResourceError::Malformed(at))?;
            let traced = match (form, imm_u32(handle)) {
                (Form::Bound | Form::Unbound, Some(offset)) => {
                    let bank = buffer
                        .index()
                        .ok_or(ResourceError::NoDescriptorBuffer(at))?;
                    Traced {
                        source: HandleSource::single(HandleWord::new(
                            bank,
                            offset - offset % crate::translate::float::CBUF_WORD_BYTES,
                        )),
                        dynamic: None,
                    }
                }
                (Form::Bound, None) => return Err(ResourceError::Malformed(at)),
                (Form::Bindless | Form::Unbound, _) => trace::trace(function, handle, &mut work)
                    .map_err(|why| ResourceError::Untraced { at, why })?,
            };
            let usage = if opcodes::is_storage_image(unbound) {
                TextureUsage::Image
            } else {
                TextureUsage::Sampled
            };
            entries.push(Entry {
                at,
                block,
                opcode: op.opcode,
                handle,
                unbound,
                traced,
                usage,
            });
        }
    }
    entries.sort_by_key(|e| {
        let primary = e.traced.source.primary();
        (primary.bank(), primary.offset())
    });
    Ok(TextureScan { entries, work })
}

/// What [`run`] does to one texture operation, decided before anything changes.
struct Rewrite {
    entry: Entry,
    flags: Flags,
    /// The sample argument emptied, for a fetch of a type that takes none.
    clear_sample: bool,
    /// The array length and spacing the index is clamped to, when more than one handle.
    index: Option<(Operand, u32, u32)>,
}

/// Inserts new instructions before existing ones: each is pushed to the end of its block, and
/// [`Inserter::finish`] moves it before its target in one pass per block.
#[derive(Default)]
struct Inserter {
    before: BTreeMap<Value, Vec<Value>>,
    original: BTreeMap<BlockId, usize>,
}

impl Inserter {
    /// Pushes `inst` into `block`, to go right before `target` (after any pushed earlier for
    /// the same target).
    fn push_before(
        &mut self,
        function: &mut Function,
        block: BlockId,
        target: Value,
        inst: Inst,
    ) -> Result<Value, BuildError> {
        let len = function
            .block(block)
            .map(|b| b.insts.len())
            .ok_or(BuildError::NoBlock(block))?;
        self.original.entry(block).or_insert(len);
        let value = function.push(block, inst)?;
        self.before.entry(target).or_default().push(value);
        Ok(value)
    }

    /// Moves every pushed instruction before its target.
    fn finish(mut self, function: &mut Function, work: &mut Work) {
        for (block, len) in self.original {
            let Some(listed) = function.block_mut(block) else {
                continue;
            };
            let old = std::mem::take(&mut listed.insts);
            let mut insts = Vec::with_capacity(old.len());
            for &value in old.iter().take(len) {
                work.step();
                if let Some(added) = self.before.remove(&value) {
                    insts.extend(added);
                }
                insts.push(value);
            }
            listed.insts = insts;
        }
    }
}

/// Builds the resource table from `scan` (taken of `function`) and `answers`, then rewrites
/// each texture operation and bounds each indexed constant-buffer read. Every check that can
/// refuse happens before the first change, so an error other than [`ResourceError::Verify`]
/// leaves `function` as it was (module docs, "When `run` refuses").
pub fn run(
    function: &mut Function,
    scan: &TextureScan,
    answers: &Answers,
) -> Result<(ResourceTable, Work), ResourceError> {
    let mut work = Work::default();
    // The block that first lists each scanned operation, found by the walk that finds the
    // reads to bound, so the stale check needs no scan of a block per operation.
    let mut homes: HashMap<Value, Option<BlockId>> =
        scan.entries.iter().map(|e| (e.at, None)).collect();
    let reads = cbuf::reads(function, &mut work, |block, at| {
        if let Some(home) = homes.get_mut(&at)
            && home.is_none()
        {
            *home = Some(block);
        }
    });
    let mut table = TableBuilder::default();
    let mut rewrites = Vec::with_capacity(scan.entries.len());
    for &entry in &scan.entries {
        work.step();
        let home = homes.get(&entry.at).copied().flatten();
        rewrites.push(plan(function, &mut table, entry, home, answers)?);
    }
    let added = rewrites
        .iter()
        .filter(|r| r.index.is_some())
        .count()
        .saturating_mul(2)
        .saturating_add(
            reads
                .iter()
                .map(|&(_, ty)| cbuf::added(ty))
                .fold(0_usize, usize::saturating_add),
        );
    let room = usize::try_from(u32::MAX).unwrap_or(usize::MAX);
    if arena_len(function).saturating_add(added) > room {
        return Err(ResourceError::Build(BuildError::Full));
    }

    let mut inserter = Inserter::default();
    for rewrite in &rewrites {
        apply(function, &mut inserter, rewrite)?;
    }
    for &(read, _) in &reads {
        cbuf::bound(function, &mut inserter, read)?;
    }
    inserter.finish(function, &mut work);
    if VERIFY {
        function.verify().map_err(ResourceError::Verify)?;
    }
    Ok((table.finish(), work))
}

/// The answer to the word at `bank` and `offset`.
fn word(answers: &Answers, bank: u8, offset: u32) -> Result<u32, ResourceError> {
    match answers.cbuf_word(bank, offset) {
        None => Err(ResourceError::UnansweredWord { bank, offset }),
        Some(None) => Err(ResourceError::NoCbufWord { bank, offset }),
        Some(Some(word)) => Ok(word),
    }
}

/// Decides what happens to `entry` and adds its descriptor to `table` (Eden's loop over
/// `to_replace`).
/// `home` is the block that lists `entry`'s operation, if any does.
fn plan(
    function: &Function,
    table: &mut TableBuilder,
    entry: Entry,
    home: Option<BlockId>,
    answers: &Answers,
) -> Result<Rewrite, ResourceError> {
    let at = entry.at;
    let op = match function.inst(at) {
        Some(Inst::Op(op))
            if op.opcode == entry.opcode
                && op.args.first() == Some(&entry.handle)
                && home == Some(entry.block) =>
        {
            op
        }
        _ => return Err(ResourceError::Stale(at)),
    };
    let Flags::Texture(mut info) = op.flags else {
        return Err(ResourceError::Malformed(at));
    };
    let source = entry.traced.source;
    let mut handle = 0;
    for part in source.words() {
        handle |= part.apply(word(answers, part.bank(), part.offset())?);
    }
    let kind = match answers.texture_kind(handle, entry.usage) {
        None => {
            return Err(ResourceError::UnansweredKind {
                handle,
                usage: entry.usage,
            });
        }
        Some(None) => return Err(ResourceError::NoTextureHeader { handle }),
        Some(Some(kind)) => kind,
    };

    let mut is_multisample = false;
    let mut clear_sample = false;
    match entry.unbound {
        Opcode::ImageQueryDimensions if kind.buffer => info.ty = TextureType::Buffer,
        Opcode::ImageFetch => {
            if matches!(
                info.ty,
                TextureType::Color2D | TextureType::Color2DRect | TextureType::ColorArray2D
            ) {
                is_multisample = op.args.get(4).is_some_and(|&sample| sample != ABSENT);
            } else {
                clear_sample = true;
            }
            if info.ty == TextureType::Color1D && kind.buffer {
                info.ty = TextureType::Buffer;
            }
        }
        _ => {}
    }

    let count = if opcodes::is_storage_image(entry.unbound)
        || info.ty == TextureType::Buffer
        || source.index().is_none()
    {
        source.count()
    } else {
        1
    };
    let index = describe(table, entry, &mut info, kind, is_multisample, count)?;
    info.descriptor_index =
        u16::try_from(index).map_err(|_| ResourceError::TooManyDescriptors(at))?;
    let index = if count > 1 {
        let dynamic = entry.traced.dynamic.ok_or(ResourceError::Malformed(at))?;
        Some((dynamic, source.size_shift(), count - 1))
    } else {
        None
    };
    Ok(Rewrite {
        entry,
        flags: Flags::Texture(info),
        clear_sample,
        index,
    })
}

/// Adds the descriptor of `entry` to `table` and gives its index; sets `info.is_integer` for a
/// sampled texture. `count` is the descriptor's array length.
fn describe(
    table: &mut TableBuilder,
    entry: Entry,
    info: &mut TextureInfo,
    kind: TextureKind,
    is_multisample: bool,
    count: u32,
) -> Result<usize, ResourceError> {
    let source = entry.traced.source;
    let component = kind.sampled;
    Ok(if opcodes::is_storage_image(entry.unbound) {
        if source.secondary().is_some() {
            return Err(ResourceError::SeparateSampler(entry.at));
        }
        let is_written = entry.unbound != Opcode::ImageRead;
        let is_read = entry.unbound != Opcode::ImageWrite;
        let source = source.with_count(count, None);
        if info.ty == TextureType::Buffer {
            table.add_image_buffer(ImageBufferDescriptor {
                source,
                image_format: info.image_format,
                header_format: kind.format,
                component,
                is_written,
                is_read,
            })
        } else {
            table.add_image(ImageDescriptor {
                source,
                ty: info.ty,
                image_format: info.image_format,
                header_format: kind.format,
                component,
                is_written,
                is_read,
            })
        }
    } else if info.ty == TextureType::Buffer {
        table.add_texture_buffer(TextureBufferDescriptor {
            source: source.with_count(count, None),
            component,
        })
    } else {
        info.is_integer = component != SampledType::Float;
        table.add_texture(TextureDescriptor {
            source: source.with_count(count, source.index()),
            ty: info.ty,
            is_depth: info.is_depth,
            is_multisample,
            component,
        })
    })
}

/// Applies `rewrite` to its operation.
fn apply(
    function: &mut Function,
    inserter: &mut Inserter,
    rewrite: &Rewrite,
) -> Result<(), ResourceError> {
    let Entry { at, block, .. } = rewrite.entry;
    let handle = match rewrite.index {
        Some((dynamic, size_shift, last)) => {
            let shifted = inserter.push_before(
                function,
                block,
                at,
                Inst::Op(Operation::new(
                    Opcode::ShiftRightLogical32,
                    vec![dynamic, Operand::Imm(Imm::U32(size_shift))],
                )),
            )?;
            let clamped = inserter.push_before(
                function,
                block,
                at,
                Inst::Op(Operation::new(
                    Opcode::UMin32,
                    vec![Operand::Value(shifted), Operand::Imm(Imm::U32(last))],
                )),
            )?;
            Operand::Value(clamped)
        }
        None => ABSENT,
    };
    let Some(Inst::Op(op)) = function.inst_mut(at) else {
        return Err(ResourceError::Stale(at));
    };
    op.opcode = rewrite.entry.unbound;
    op.flags = rewrite.flags;
    if let Some(first) = op.args.first_mut() {
        *first = handle;
    }
    if rewrite.clear_sample
        && let Some(sample) = op.args.get_mut(4)
    {
        *sample = ABSENT;
    }
    Ok(())
}

impl Display for TextureScan {
    /// One line per texture operation: its id and its handle's source.
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        for entry in &self.entries {
            writeln!(f, "{}: {}", entry.at, entry.traced.source)?;
        }
        Ok(())
    }
}
