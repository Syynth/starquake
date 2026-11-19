//! The front thread's register files: GPU protocol slice G9 (`docs/specs/gpu-protocol.md`,
//! "Engine methods: the register file", "Macros" and "Ordering guarantees").
//!
//! The front keeps, per channel and per class in use, a flat file of every register below the
//! class's macro range, the only authoritative copy of the engine's register values (owner's
//! decision 1). The back thread mirrors it from the ring and never changes a register itself.
//!
//! - `file` owns [`RegFile`] (one class on one channel), [`ShadowMode`], [`RegsError`] and
//!   [`EntrySink`], where the surviving writes go.
//! - `rules` owns [`ClassRules`]: per class, the file's size, the shadow RAM control, the
//!   plain-state methods and the side-effect registers ([`SIDE_EFFECTS_3D`],
//!   [`SIDE_EFFECTS_COMPUTE`]), for [`THREE_D`], [`COMPUTE`] and every [`OTHER`] class.
//! - `state_3d` owns [`STATE_3D`], the hand-written table of the 3D class's plain-state methods,
//!   keyed on the generated `clc797` constants (the generated tables do not say which methods
//!   are plain state).
//!
//! Ported from `eden-nx2/src/video_core/engines/maxwell_3d.cpp` (`ProcessShadowRam`,
//! `CallMethod`, `FrontWrite`, `FrontCallMultiMethod`) and
//! `eden-nx2/src/video_core/engines/kepler_compute.cpp` (`FrontApply`), Eden-NX2 @ 50fa86bb,
//! GPL-3.0-or-later; the root table offset from
//! `eden-pr4588/src/video_core/native_root_table.h` @ e5fb3b97, GPL-3.0-or-later; array lengths
//! from Mesa's `class_parser.py` (`mesa-nouveau/src/nouveau/headers/`) @ 52238762, MIT. Paths are
//! relative to the reference folder (`tools/refs-path.sh`).
//!
//! # A write
//!
//! [`RegFile::write`] and [`RegFile::write_run`] take each data word through the spec's three
//! steps, in this order:
//!
//! 1. **Shadow RAM** (3D only; the spec applies it "for 3D", and Eden-NX2's compute engine has
//!    none). The mode is the `MODE` field of the *shadow file's* copy of
//!    `SET_MME_SHADOW_RAM_CONTROL` (method 0x49), as row 144 found: in
//!    [`ShadowMode::Track`] and [`ShadowMode::TrackWithFilter`] the word is also stored in the
//!    shadow file; in [`ShadowMode::Replay`] the register takes the shadow file's value instead
//!    of the word, except for a data-load method (below); in [`ShadowMode::Passthrough`] the
//!    shadow file is left alone. A write to the control itself then stores the word sent in the
//!    shadow file, in every mode, so the next word sees the new mode (Eden-NX2's `FrontWrite`),
//!    while the register file holds the value step 1 resolved (in replay, the old shadow copy).
//!
//!    **Data loads in replay.** The class's data-load methods ([`ClassRules::data_loads`]: for
//!    3D, [`DATA_LOADS_3D`], the sixteen `LOAD_CONSTANT_BUFFER(i)` and `LOAD_INLINE_DATA`) take
//!    the words written even in replay mode, and the shadow file is left alone; every other
//!    method takes the shadow value. This is the spec's step 1 (decided by the owner,
//!    2026-11-18), after Eden-NX2's run path: `Maxwell3D::CallMultiMethod` hands a run of
//!    constant-buffer or inline data to the upload as written, without `ProcessShadowRam`. It
//!    is a **guess**, as the spec says, until a trace shows replay mode during an upload. In
//!    the track modes and passthrough a data load is treated as every other method. Pinned by
//!    `replay_data_loads_take_the_written_words_and_state_takes_the_shadow_value`,
//!    `data_loads_end_at_their_neighbours`, `track_and_passthrough_treat_data_loads_and_state_alike`
//!    and `data_load_methods_are_the_headers_constant_buffer_and_inline_data_methods`.
//! 2. **Side-effect registers.** A trigger listed in the class's side effects changes another
//!    register, once per word, after the trigger's own value is stored; the change is sent as an
//!    ordinary `Write` after the trigger's entry, and only if the register changed. The lists are
//!    [`SIDE_EFFECTS_3D`] and [`SIDE_EFFECTS_COMPUTE`]; no plain-state method has one. The macro
//!    upload pointer is not among them, as the spec's step 2 says (decided by the owner,
//!    2026-11-18): `LOAD_MME_INSTRUCTION_RAM_POINTER` (method 0x45) stays as written, and the
//!    macro host (G13, `crate::macros`) keeps the upload as a word offset, the pointer times 3,
//!    as both Switch 2 references do (`native_write_offset = pointer * 3`, `MacroEngine::AddCode`
//!    and `ClearCode` in `eden-nx2/src/video_core/macro.h`). Pinned by
//!    `instruction_upload_leaves_its_pointer_alone`.
//! 3. **Same-value dropping.** A write to a plain-state method whose resolved value equals the
//!    register's is dropped. Every other write updates the file and is sent. A method is plain
//!    state only if the class's state table lists it ([`STATE_3D`]; empty for compute and every
//!    other class); anything else, known or not, is a trigger and is always sent, so a wrong
//!    table costs speed, not correctness (decision 12). [`RegFile::set_drop_same_values`] turns
//!    dropping off.
//!
//! **Runs.** Words to one trigger method go out as one `Run` entry (or a `Write` for a single
//! word), split every [`crate::ring::MAX_RUN_WORDS`] words, each followed by the side-effect
//! registers' final values: a run of `n` constant-buffer words is one `Run` and one `Write` of
//! `LOAD_CONSTANT_BUFFER_OFFSET` advanced by `4n`. The back performs a run's words from the
//! side-effect registers' values before it, then applies the `Write`s. Words to a state method
//! are each a `Write` or dropped. Either way the register and shadow files end as if each word
//! had been written alone.
//!
//! **Order** (spec, "Ordering guarantees" 1): entries go to the sink in the order the words were
//! written, and a trigger's entry before its side-effect writes. Each trigger in
//! [`SIDE_EFFECTS_3D`] and [`SIDE_EFFECTS_COMPUTE`] has one rule today, so the order among one
//! trigger's side-effect writes (the order of the class's list) shows only once a trigger has
//! two, and no test pins it yet.
//!
//! # Divergences
//!
//! Each is pinned by the test named.
//!
//! - **The compute bind pointer is a register here.** The spec's "macro bind pointers" advance
//!   `LOAD_MME_START_ADDRESS_RAM_POINTER` (0x47) in both classes; Eden-NX2's compute engine keeps
//!   that pointer outside its registers (so a compute macro reading 0x47 sees the written value
//!   there) and stops advancing it at 0x80 entries (`FrontApply`). Its 3D front advances the
//!   register without a bound, as both classes do here. `compute_bind_pointer_advances`.
//! - **No same-value test of the shadow file.** Eden-NX2's `FrontWrite` keeps a same-value write
//!   that changed the shadow file (`shadow_kept`), because its back keeps its own shadow file.
//!   Here the shadow file is the front's alone, so only the register decides.
//!   `same_value_write_is_dropped_even_when_it_changes_the_shadow_file`.
//! - **The mode is the `MODE` field.** Eden compares the whole control word with the modes, so a
//!   word with bits above 1:0 set is passthrough there; here only the header's field (1:0)
//!   counts (decision 10: the headers for layouts). `mode_is_the_mode_field`.
//! - **A run is its words one by one.** Eden-NX2's multi-word paths skip the shadow RAM and the
//!   register store for constant-buffer and inline data (and for inline indices outside
//!   replay); here every word of a run goes through step 1 and the register keeps the last
//!   word. So in the track modes a data-load word is also stored in the shadow file, where
//!   Eden-NX2's run path stores nothing; in replay a data-load run sends its words as written,
//!   as Eden-NX2's does, and the register keeps the last of them. Inline indices are not data
//!   loads here: in replay each word takes the shadow value, as Eden-NX2's `CallMultiMethod`
//!   falls through to `CallMethod` for them in replay.
//!   `track_and_passthrough_treat_data_loads_and_state_alike`,
//!   `replay_triggers_that_are_not_data_loads_take_the_shadow_value` (the inline indices),
//!   `run_in_replay_sends_the_written_data_words`, `run_equals_single_writes`.
//! - **A single data-load word in replay takes the word written.** Eden-NX2's single-word path,
//!   `Maxwell3D::CallMethod`, runs `ProcessShadowRam` first and hands its result to
//!   `ProcessCBData(argument)` or `upload_state.ProcessData(argument, ..)`, so a lone word in
//!   replay uploads the shadow value while a run uploads the words written. Here a single word
//!   and a run alike take the words written, as the spec's step 1 says.
//!   `replay_data_loads_take_the_written_words_and_state_takes_the_shadow_value` (its single
//!   word).
//! - **The constant-buffer offset wraps within its field.** Eden adds to the whole register;
//!   here the offset advances within `LOAD_CONSTANT_BUFFER_OFFSET_V` (15:0) and the bits above
//!   are kept. `constant_buffer_offset_wraps_within_its_field`.
//! - **The root table offset lives in the selector register, and wraps.** PR #4588 keeps the
//!   offset in a private variable (`NativeRootTables::offset`,
//!   `eden-pr4588/src/video_core/native_root_table.h`) and leaves `SET_ROOT_TABLE_SELECTOR`
//!   (method 0x141) as written, since its `Select` only reads the word. Here the selector's
//!   `OFFSET` field (15:8) advances instead, so a macro `state` read of 0x141 after a load sees
//!   the advanced offset where PR #4588 shows the written one. PR #4588's variable reaches 256
//!   after the 64th word and refuses loads until the next select; here the 8-bit field wraps to
//!   0 after 252 (**guess**; a trace of a 65-word root table load would settle it). An offset
//!   that is not a multiple of 4 does not advance, as PR's `Load` refuses it (`offset % 4`).
//!   `root_table_offset_advances` (which asserts the selector register after each load).
//! - **Nothing is dropped for compute or the other classes**, as Eden-NX2's compute front sends
//!   every write; their state tables are empty. `compute_and_other_classes_drop_nothing`.
//!
//! # Owner question: which methods take the shadow value in replay
//!
//! The spec's step 1 says: "Data-load methods (constant-buffer loads and inline-to-memory
//! data) take the written words even in replay mode; only state methods take the shadow
//! value". Here every method that is not a data load takes the shadow value in replay,
//! triggers as well as plain-state methods (for example `LAUNCH_DMA` and the three
//! `DRAW_INLINE_INDEX*` methods), as Eden-NX2's `CallMethod` applies `ProcessShadowRam` to
//! every method before it looks at `execution_mask`. Read literally, "only state methods"
//! would let those triggers take the word written too. Keep this reading (the spec names only
//! the data loads as the exception), or extend the exception to every trigger? Pinned by
//! `replay_triggers_that_are_not_data_loads_take_the_shadow_value`.
//!
//! Not here: register defaults (Eden's `InitializeRegisterDefaults` sets some 3D registers to
//! non-zero values at boot; every file here starts from [`ClassRules::initial_values`], all
//! zero, which the back's mirror starts from too), the macro calls themselves (G13,
//! `crate::macros`), and counters of dropped writes.

mod file;
mod rules;
mod state_3d;

#[cfg(test)]
mod tests;

pub(crate) use file::Journal;
pub use file::{EntrySink, RegFile, RegsError, ShadowMode};
pub use rules::{
    COMPUTE, ClassRules, DATA_LOADS_3D, FALCON04_SCRATCH_VALUE, LOAD_CONSTANT_BUFFER_COUNT,
    MethodSpan, OTHER, SIDE_EFFECTS_3D, SIDE_EFFECTS_COMPUTE, SideEffect, SideEffectRule, THREE_D,
    WORD_BYTES,
};
pub use state_3d::STATE_3D;
