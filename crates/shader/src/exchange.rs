//! Translation inputs: the questions translation asks about engine state, their answers, and the
//! exchange loop between them (`docs/specs/shader-pipeline.md`, "Translation inputs: questions
//! and answers"; port slice 9; owner's decision 5, the non-blocking question-and-answer lookup).
//!
//! Eden's `Environment` lets translation read live engine state whenever it likes
//! (`eden/src/shader_recompiler/environment.h` @ cf654052). Starquake asks for engine state
//! through an explicit exchange instead, which is new (no reference has it), but the questions
//! of this slice do not cover everything `Environment` reads: the table under "What stands in
//! for each `Environment` member" gives the exact mapping, and what is not covered is listed
//! there as owner questions. The exchange:
//!
//! 1. The program's side, a [`Questioner`], says which [`Question`]s it needs answered, given
//!    the [`Answers`] so far.
//! 2. The GPU crate's back thread answers each new question from current engine state through
//!    [`EngineState`] (the sketch's trait, implemented by the GPU crate, slice G18).
//! 3. This repeats while the questioner asks something new: a question may depend on an earlier
//!    answer (a jump table found only once another table is known). [`exchange`] stops with the
//!    answers when nothing new is asked, and refuses a program that still asks something new
//!    after [`MAX_QUESTION_ROUNDS`] rounds ([`ExchangeError::TooManyRounds`]).
//!
//! The loop is pure: it reads nothing but what the questioner and the engine state return, and
//! it calls the engine state once per distinct question, in the order the questioner first
//! asked them. The [`Answers`] it returns are the record kept with each shader entry (spec,
//! "Identity, caching and invalidation", layer 3), and [`Answers::reanswer`] asks the same
//! questions again on a later draw.
//!
//! # Rounds
//!
//! A round is one call to the questioner that asks at least one question not answered yet; its
//! new questions are then answered. A call that asks nothing new ends the exchange and is not a
//! round, so a program with no questions takes zero rounds, and a program whose questions chain
//! [`MAX_QUESTION_ROUNDS`] deep is accepted. Questions asked again, in the same call or a later
//! one, are answered once. Pinned by `tests::the_round_cap_at_its_edge`,
//! `tests::no_questions_take_no_rounds` and `tests::repeated_questions_are_answered_once`.
//!
//! # What the loop does not bound
//!
//! The number of questions in a round is the questioner's: the loop stores each distinct
//! question once and does work in proportion to what the questioner returns. The translator
//! that implements [`Questioner`] (flow analysis and the texture pass, slices 10 and 13) bounds
//! it by the program it walks.
//!
//! # What stands in for each `Environment` member
//!
//! Every member of Eden's `Environment` (`eden/src/shader_recompiler/environment.h` @
//! cf654052), with what starquake has in its place: a [`Question`], a field of the program
//! request or the variant key ([`crate::lookup::ProgramRequest`], [`crate::variant::VariantKey`]),
//! a driver constant ([`crate::driver_constants`]), or **not covered**.
//!
//! | `Environment` member | Starquake | Covered |
//! |---|---|---|
//! | `ReadInstruction` | the program's bytes, read through [`crate::lookup::ProgramMemory`] from [`crate::lookup::ProgramRequest::address`] | yes |
//! | `ReadCbufValue` | [`Question::CbufWord`] | yes |
//! | `ReadTextureType` | only whether the type is `Buffer`: [`TextureKind::buffer`] | **partly**; see below |
//! | `ReadTexturePixelFormat` | [`TextureKind::format`], the header's format bits rather than Eden's converted `TexturePixelFormat` | yes |
//! | `IsTexturePixelFormatInteger` | [`TextureKind::sampled`] (which also tells signed from unsigned) | yes |
//! | `ReadViewportTransformState` | nothing | **no** |
//! | `TextureBoundBuffer` | nothing | **no** |
//! | `LocalMemorySize` | graphics: the program header, part of the program's bytes; compute: nothing | **graphics only** |
//! | `SharedMemorySize` | [`crate::variant::VariantKey::shared_memory_size`] | yes |
//! | `WorkgroupSize` | [`crate::variant::VariantKey::workgroup_size`] | yes |
//! | `HasHLEMacroState`, `GetReplaceConstBuffer` | nothing | **no** |
//! | `Dump` | nothing: a debug dump, not a translation input | not an input |
//! | `SPH` | the program header, part of the program's bytes | yes |
//! | `GpPassthroughMask` | nothing | **no** |
//! | `ShaderStage` | [`crate::lookup::ProgramRequest::stage`] | yes |
//! | `StartAddress` | [`crate::lookup::ProgramRequest::address`] (an absolute address, where Eden's is an offset from the program base) | yes |
//! | `IsProprietaryDriver` | nothing (Eden derives it from `TextureBoundBuffer`, `texture_bound == 2`) | **no** |
//! | `cbuf_word_cache`, `handle_cache`, `track_cache` | nothing: Eden's caches, not translation inputs | not an input |
//!
//! No driver constant stands in for an `Environment` member: the driver constants hold values
//! Eden takes from `RuntimeInfo` and the draw, not from `Environment`. [`TextureKind::depth`] and
//! [`TextureKind::multisampled`] have no `Environment` member either (Eden takes both from the
//! instruction); they are in the answer because the spec lists them.
//!
//! ## For the owner: what the questions do not cover
//!
//! Each of these is read by Eden from `Environment` and has no question, key field or driver
//! constant here. This slice adds no field for them; each needs the owner's call on whether it
//! becomes a question, a key field or a driver constant.
//!
//! - **The texture type: dimension, array and rect.** Eden's texture pass reads
//!   `ReadTextureType` to assign the type of `ImageQueryDimensions` (`TXQ`), and patches
//!   `ImageSampleImplicitLod` when the bound type is `Color2DRect`
//!   (`eden/src/shader_recompiler/ir_opt/texture_pass.cpp` @ cf654052). Only its third use, an
//!   `ImageFetch` on a 1D texture becoming `Buffer`, is covered, by [`TextureKind::buffer`].
//! - **`TextureBoundBuffer`**: the constant-buffer bank that bound texture handles are read from
//!   (`bindless_texture_const_buffer_slot` for graphics, `tex_cb_index` for compute;
//!   `eden/src/video_core/shader_environment.cpp`), used by the texture pass for the bank of
//!   bound textures. `IsProprietaryDriver`, which Eden derives from it, goes with it.
//! - **`ReadViewportTransformState`**: whether the viewport scale and offset are enabled
//!   (`viewport_scale_offset_enabled`). It gates Eden-NX2's `PositionPass`
//!   (`eden-nx2/src/shader_recompiler/ir_opt/position_pass.cpp` @ 50fa86bb), the row-65 path.
//! - **Compute `LocalMemorySize`**: Eden takes it from the QMD (`local_pos_alloc +
//!   local_crs_alloc`, `ComputeEnvironment`'s constructor in `shader_environment.cpp`), which is
//!   not in a compute program's bytes. A graphics program's comes from its header.
//! - **`HasHLEMacroState` and `GetReplaceConstBuffer`**: constant-buffer words that Eden's
//!   high-level macro emulation replaces, folded by its constant propagation pass
//!   (`eden/src/shader_recompiler/ir_opt/constant_propagation_pass.cpp`).
//! - **`GpPassthroughMask`**: the geometry passthrough mask, which Eden reads from
//!   `post_vtg_shader_attrib_skip_mask` (`shader_environment.cpp`) for a passthrough geometry
//!   program (`eden/src/shader_recompiler/frontend/maxwell/translate_program.cpp`).

use std::collections::{BTreeMap, BTreeSet};

use thiserror::Error;

use crate::translate::GuestStage;

/// The most rounds of answers a program may need (spec: "four is a guess; experiment E6
/// measures it"). A **guess**, to be set by experiment E6, which counts the rounds programs of
/// local title captures need ([`Exchanged::rounds`] reports it).
pub const MAX_QUESTION_ROUNDS: u32 = 4;

/// The bits of the texture header's first word that name a format: the component layout
/// (`NVC797_TEXHEAD_BL_COMPONENTS`, `MW(6:0)`) and the data type of each component
/// (`NVC797_TEXHEAD_BL_R_DATA_TYPE` to `_A_DATA_TYPE`, `MW(9:7)`, `MW(12:10)`, `MW(15:13)`,
/// `MW(18:16)`), from `mesa-nouveau/src/nouveau/headers/nvidia/classes/clc797tex.h` @
/// 52238762. Eden reads the same bits as `TICEntry::format` and `r_type` to `a_type`
/// (`eden/src/video_core/textures/texture.h` @ cf654052).
pub const TEXHEAD_FORMAT_MASK: u32 = (1 << 19) - 1;

/// A fact about engine state that a program's translation depends on (spec, "Translation
/// inputs: questions and answers").
///
/// Ordered so that [`Answers`] keeps its record in one deterministic order whatever order the
/// questions were asked in.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Question {
    /// A constant-buffer word the program's structure depends on, such as a `BRX` jump-table
    /// entry (rows 91, 92) or a texture handle loaded from a constant buffer.
    CbufWord {
        /// The constant-buffer bank (a 5-bit field in the instruction, so 0 to 31; spec,
        /// "Passes needed for correctness").
        bank: u8,
        /// The word's byte offset in the bank.
        offset: u32,
    },
    /// What a texture handle points to.
    TextureKind {
        /// The handle, as the program reads it.
        handle: u32,
    },
}

/// The component type a texture's samples have (spec: "sampled component type (float, signed,
/// unsigned)").
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SampledType {
    /// Floating point, including normalised formats.
    Float,
    /// Signed integer.
    SignedInt,
    /// Unsigned integer.
    UnsignedInt,
}

/// A texture's format as the guest's texture header gives it: the bits of its first word under
/// [`TEXHEAD_FORMAT_MASK`] (component layout and per-component data types).
///
/// Translation needs it for formatted surface loads and stores, whose host storage image must
/// declare its format (rows 83 and 86 list such programs as rejected by Eden-NX2's lowering).
/// Mapping it to a host format is the lowering's job (slices 13 and 17).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TextureFormat(u32);

impl TextureFormat {
    /// The format of a texture header whose first word is `word`; the bits outside
    /// [`TEXHEAD_FORMAT_MASK`] (the swizzle sources) are dropped.
    pub const fn from_texhead_word0(word: u32) -> Self {
        Self(word & TEXHEAD_FORMAT_MASK)
    }

    /// The format bits, in their places in the header's first word.
    pub const fn bits(self) -> u32 {
        self.0
    }
}

/// The answer to [`Question::TextureKind`]: what a texture handle points to (spec, "Translation
/// inputs": component type, depth, multisampled, buffer texture and storage-image format).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TextureKind {
    /// The component type its samples have.
    pub sampled: SampledType,
    /// Whether it has a depth format.
    pub depth: bool,
    /// Whether it is multisampled.
    pub multisampled: bool,
    /// Whether it is a buffer texture.
    pub buffer: bool,
    /// Its format, for a formatted surface load or store. The back thread fills it for every
    /// handle, because the question does not say how the program uses the handle; so two
    /// formats with the same kind give different answers, and a shader that only samples the
    /// texture is retranslated when the format changes.
    pub format: TextureFormat,
}

/// The answer to one [`Question`], as [`EngineState`] gave it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Answer {
    /// The answer to [`Question::CbufWord`]; `None` when engine state has no such word (the
    /// bank is not bound, or the offset is past its end).
    CbufWord(Option<u32>),
    /// The answer to [`Question::TextureKind`]; `None` when the handle points to no valid
    /// texture header.
    TextureKind(Option<TextureKind>),
}

/// Answers questions from current engine state, on the GPU back thread (spec, "What the GPU
/// front/back protocol and engine state (#12) gets"). Implemented by the GPU crate (slice
/// G18).
pub trait EngineState {
    /// The word at byte `offset` of constant-buffer `bank` bound to `stage`, or `None` if there
    /// is none.
    fn cbuf_word(&self, stage: GuestStage, bank: u8, offset: u32) -> Option<u32>;
    /// What `handle` points to for `stage`, or `None` if it points to no valid texture header.
    fn texture_kind(&self, stage: GuestStage, handle: u32) -> Option<TextureKind>;
}

/// The side of translation that asks: says which questions a program needs answered.
pub trait Questioner {
    /// Every question the program needs answered, given the answers so far. Questions already
    /// answered may be returned again; they are not asked twice.
    fn questions(&self, answers: &Answers) -> Vec<Question>;
}

/// The answers a program's translation used: the record kept with each shader entry, ordered
/// by [`Question`].
///
/// Filled only by [`exchange`] and [`Answers::reanswer`] ([`Answers::new`] is empty), so each
/// answer is of its question's kind, and the arms of [`Answers::cbuf_word`] and
/// [`Answers::texture_kind`] that meet an answer of the other kind cannot be reached.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Answers(BTreeMap<Question, Answer>);

impl Answers {
    /// No answers.
    pub const fn new() -> Self {
        Self(BTreeMap::new())
    }

    /// The answer to `question`, if it was asked.
    pub fn get(&self, question: Question) -> Option<Answer> {
        self.0.get(&question).copied()
    }

    /// The answer to [`Question::CbufWord`] for `bank` and `offset`: `None` if it was not asked,
    /// `Some(None)` if engine state had no such word.
    pub fn cbuf_word(&self, bank: u8, offset: u32) -> Option<Option<u32>> {
        match self.get(Question::CbufWord { bank, offset }) {
            Some(Answer::CbufWord(word)) => Some(word),
            Some(Answer::TextureKind(_)) | None => None,
        }
    }

    /// The answer to [`Question::TextureKind`] for `handle`: `None` if it was not asked,
    /// `Some(None)` if the handle points to no valid texture header.
    pub fn texture_kind(&self, handle: u32) -> Option<Option<TextureKind>> {
        match self.get(Question::TextureKind { handle }) {
            Some(Answer::TextureKind(kind)) => Some(kind),
            Some(Answer::CbufWord(_)) | None => None,
        }
    }

    /// Every question and its answer, ordered by question.
    pub fn iter(&self) -> impl Iterator<Item = (Question, Answer)> + '_ {
        self.0.iter().map(|(question, answer)| (*question, *answer))
    }

    /// How many questions were answered.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Whether no question was answered.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// The same questions answered again from `engine`, for a later draw (spec: "On a later draw
    /// the back thread re-answers the same list"). Asks in question order. A different result
    /// means the shader entry recorded with `self` does not fit the draw; the questions a new
    /// translation asks may then differ too, which a new [`exchange`] finds out.
    #[must_use]
    pub fn reanswer(&self, stage: GuestStage, engine: &dyn EngineState) -> Self {
        Self(
            self.0
                .keys()
                .map(|&question| (question, answer(stage, question, engine)))
                .collect(),
        )
    }
}

/// Answers one question from `engine`.
fn answer(stage: GuestStage, question: Question, engine: &dyn EngineState) -> Answer {
    match question {
        Question::CbufWord { bank, offset } => {
            Answer::CbufWord(engine.cbuf_word(stage, bank, offset))
        }
        Question::TextureKind { handle } => Answer::TextureKind(engine.texture_kind(stage, handle)),
    }
}

/// What a finished [`exchange`] gives.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Exchanged {
    /// Every answer, the record kept with the shader entry.
    pub answers: Answers,
    /// How many rounds it took, at most [`MAX_QUESTION_ROUNDS`].
    pub rounds: u32,
}

/// Why an exchange failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum ExchangeError {
    /// The program still asked a new question after the most rounds allowed.
    #[error("the program still asks new questions after {rounds} rounds of answers")]
    TooManyRounds {
        /// The rounds answered: [`MAX_QUESTION_ROUNDS`].
        rounds: u32,
    },
}

/// Runs the exchange for one program of `stage`: asks `questioner`, answers each new question
/// from `engine`, and repeats until nothing new is asked (see the module docs, "Rounds").
pub fn exchange(
    stage: GuestStage,
    questioner: &dyn Questioner,
    engine: &dyn EngineState,
) -> Result<Exchanged, ExchangeError> {
    let mut answers = Answers::new();
    let mut rounds = 0;
    loop {
        let mut seen = BTreeSet::new();
        let new: Vec<Question> = questioner
            .questions(&answers)
            .into_iter()
            .filter(|question| !answers.0.contains_key(question) && seen.insert(*question))
            .collect();
        if new.is_empty() {
            return Ok(Exchanged { answers, rounds });
        }
        if rounds >= MAX_QUESTION_ROUNDS {
            return Err(ExchangeError::TooManyRounds { rounds });
        }
        rounds += 1;
        for question in new {
            answers.0.insert(question, answer(stage, question, engine));
        }
    }
}

#[cfg(test)]
mod tests;
