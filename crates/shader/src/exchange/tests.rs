//! Exchange tests: the round cap at its edge, deduplication, call order, the stage passed
//! through, and re-answering.

use std::cell::RefCell;

use super::*;

/// Engine state that answers from fixed rules and records every call.
///
/// A constant-buffer word is `offset + STEP` plus the stage's index times `STAGE_SALT`, so a
/// chain of questions that follows the answers moves forward and a wrong stage changes the
/// value; bank [`UNBOUND_BANK`] and handle [`INVALID_HANDLE`] have no answer.
struct FakeState {
    calls: RefCell<Vec<(GuestStage, Call)>>,
    /// Added to every word, so two states give different answers to the same question.
    bias: u32,
}

const STEP: u32 = 0x10;
const STAGE_SALT: u32 = 0x1000;
const UNBOUND_BANK: u8 = 31;
const INVALID_HANDLE: u32 = 0xdead;
/// A stage that is not the first variant, so a hard-coded stage fails.
const STAGE: GuestStage = GuestStage::Pixel;
const CHAIN_BANK: u8 = 3;
const CHAIN_START: u32 = 0x40;

fn stage_index(stage: GuestStage) -> u32 {
    match stage {
        GuestStage::VertexCullBeforeFetch => 0,
        GuestStage::Vertex => 1,
        GuestStage::TessellationInit => 2,
        GuestStage::Tessellation => 3,
        GuestStage::Geometry => 4,
        GuestStage::Pixel => 5,
        GuestStage::Compute => 6,
    }
}

/// One call the engine state saw: the engine is asked about a handle, not about its usage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Call {
    Cbuf { bank: u8, offset: u32 },
    Texture(u32),
}

/// The calls `asked` make of the engine state.
fn asked(asked: Vec<(GuestStage, Question)>) -> Vec<(GuestStage, Call)> {
    asked
        .into_iter()
        .map(|(stage, question)| {
            let call = match question {
                Question::CbufWord { bank, offset } => Call::Cbuf { bank, offset },
                Question::TextureKind { handle, .. } => Call::Texture(handle),
            };
            (stage, call)
        })
        .collect()
}

fn kind_for(handle: u32) -> TextureKind {
    TextureKind {
        sampled: SampledType::SignedInt,
        depth: handle & 1 != 0,
        multisampled: handle & 2 != 0,
        buffer: handle & 4 != 0,
        format: TextureFormat::from_texhead_word0(handle),
    }
}

impl FakeState {
    fn new(bias: u32) -> Self {
        Self {
            calls: RefCell::new(Vec::new()),
            bias,
        }
    }

    fn calls(&self) -> Vec<(GuestStage, Call)> {
        self.calls.borrow().clone()
    }
}

impl EngineState for FakeState {
    fn cbuf_word(&self, stage: GuestStage, bank: u8, offset: u32) -> Option<u32> {
        self.calls
            .borrow_mut()
            .push((stage, Call::Cbuf { bank, offset }));
        (bank != UNBOUND_BANK).then(|| offset + STEP + stage_index(stage) * STAGE_SALT + self.bias)
    }

    fn texture_kind(&self, stage: GuestStage, handle: u32) -> Option<TextureKind> {
        self.calls.borrow_mut().push((stage, Call::Texture(handle)));
        (handle != INVALID_HANDLE).then(|| kind_for(handle))
    }
}

/// A program whose questions chain: it asks for the word at [`CHAIN_START`], then for the word
/// at the offset each answer gives, `depth` questions deep. Each question depends on the answer
/// before it, so each takes its own round.
struct Chain {
    depth: u32,
}

impl Questioner for Chain {
    fn questions(&self, answers: &Answers) -> Vec<Question> {
        let mut out = Vec::new();
        let mut offset = CHAIN_START;
        for _ in 0..self.depth {
            out.push(Question::CbufWord {
                bank: CHAIN_BANK,
                offset,
            });
            match answers.cbuf_word(CHAIN_BANK, offset) {
                Some(Some(next)) => offset = next,
                _ => break,
            }
        }
        out
    }
}

/// A program that asks a scripted list per call: call `n` returns `script[n]`, and every call
/// past the script returns the last entry.
struct Script(Vec<Vec<Question>>, RefCell<usize>);

impl Script {
    fn new(script: Vec<Vec<Question>>) -> Self {
        Self(script, RefCell::new(0))
    }
}

impl Questioner for Script {
    fn questions(&self, _answers: &Answers) -> Vec<Question> {
        let mut call = self.1.borrow_mut();
        let index = (*call).min(self.0.len().saturating_sub(1));
        *call += 1;
        self.0.get(index).cloned().unwrap_or_default()
    }
}

fn cbuf(bank: u8, offset: u32) -> Question {
    Question::CbufWord { bank, offset }
}

/// A question about `handle` used as an image, whose answer keeps the whole kind.
fn texture(handle: u32) -> Question {
    Question::TextureKind {
        handle,
        usage: TextureUsage::Image,
    }
}

/// A question about `handle` only sampled.
fn sampled(handle: u32) -> Question {
    Question::TextureKind {
        handle,
        usage: TextureUsage::Sampled,
    }
}

#[test]
fn the_round_cap_is_the_specs_guess() {
    // The spec: "four is a guess; experiment E6 measures it".
    assert_eq!(MAX_QUESTION_ROUNDS, 4);
}

#[test]
fn the_round_cap_at_its_edge() {
    for depth in [1, MAX_QUESTION_ROUNDS - 1, MAX_QUESTION_ROUNDS] {
        let state = FakeState::new(0);
        let done = exchange(STAGE, &Chain { depth }, &state).expect("within the cap");
        assert_eq!(done.rounds, depth);
        assert_eq!(done.answers.len(), usize::try_from(depth).expect("small"));
        assert_eq!(state.calls().len(), usize::try_from(depth).expect("small"));
    }
    let state = FakeState::new(0);
    let refused = exchange(
        STAGE,
        &Chain {
            depth: MAX_QUESTION_ROUNDS + 1,
        },
        &state,
    );
    assert_eq!(
        refused,
        Err(ExchangeError::TooManyRounds {
            rounds: MAX_QUESTION_ROUNDS
        })
    );
    assert_eq!(
        refused.map_err(|error| error.to_string()),
        Err(format!(
            "the program still asks new questions after {MAX_QUESTION_ROUNDS} rounds of answers"
        ))
    );
    // The rounds within the cap were answered; the question of the extra round was not.
    assert_eq!(
        state.calls().len(),
        usize::try_from(MAX_QUESTION_ROUNDS).expect("small")
    );
}

#[test]
fn each_question_follows_the_answer_before_it() {
    let state = FakeState::new(0);
    let done = exchange(STAGE, &Chain { depth: 3 }, &state).expect("within the cap");
    let salt = stage_index(STAGE) * STAGE_SALT;
    let second = CHAIN_START + STEP + salt;
    let third = second + STEP + salt;
    assert_eq!(
        state.calls(),
        asked(vec![
            (STAGE, cbuf(CHAIN_BANK, CHAIN_START)),
            (STAGE, cbuf(CHAIN_BANK, second)),
            (STAGE, cbuf(CHAIN_BANK, third)),
        ])
    );
    assert_eq!(
        done.answers.cbuf_word(CHAIN_BANK, third),
        Some(Some(third + STEP + salt))
    );
}

#[test]
fn no_questions_take_no_rounds() {
    let state = FakeState::new(0);
    let done = exchange(STAGE, &Script::new(vec![vec![]]), &state).expect("no questions");
    assert_eq!(
        done,
        Exchanged {
            answers: Answers::new(),
            rounds: 0
        }
    );
    assert!(done.answers.is_empty());
    assert_eq!(state.calls(), Vec::new());
}

#[test]
fn repeated_questions_are_answered_once() {
    let state = FakeState::new(0);
    let script = Script::new(vec![
        vec![cbuf(1, 8), texture(6), cbuf(1, 8)],
        vec![texture(6), cbuf(1, 8), cbuf(2, 0x20), cbuf(2, 0x20)],
        vec![cbuf(2, 0x20), texture(6), cbuf(1, 8)],
    ]);
    let done = exchange(STAGE, &script, &state).expect("within the cap");
    assert_eq!(done.rounds, 2);
    assert_eq!(
        state.calls(),
        asked(vec![
            (STAGE, cbuf(1, 8)),
            (STAGE, texture(6)),
            (STAGE, cbuf(2, 0x20)),
        ])
    );
}

#[test]
fn a_round_that_repeats_only_old_questions_ends_the_exchange() {
    // Every call asks something; only the first asks something new.
    let state = FakeState::new(0);
    let script = Script::new(vec![vec![cbuf(1, 8)], vec![cbuf(1, 8)]]);
    let done = exchange(STAGE, &script, &state).expect("within the cap");
    assert_eq!(done.rounds, 1);
    assert_eq!(state.calls(), asked(vec![(STAGE, cbuf(1, 8))]));
}

#[test]
fn questions_are_asked_in_first_ask_order_and_recorded_by_question() {
    let state = FakeState::new(0);
    let script = Script::new(vec![vec![texture(9), cbuf(2, 8), cbuf(1, 0x10)]]);
    let done = exchange(STAGE, &script, &state).expect("within the cap");
    assert_eq!(
        state.calls(),
        asked(vec![
            (STAGE, texture(9)),
            (STAGE, cbuf(2, 8)),
            (STAGE, cbuf(1, 0x10)),
        ])
    );
    let salt = stage_index(STAGE) * STAGE_SALT;
    assert_eq!(
        done.answers.iter().collect::<Vec<_>>(),
        vec![
            (cbuf(1, 0x10), Answer::CbufWord(Some(0x10 + STEP + salt))),
            (cbuf(2, 8), Answer::CbufWord(Some(8 + STEP + salt))),
            (texture(9), Answer::TextureKind(Some(kind_for(9)))),
        ]
    );
    assert_eq!(done.answers.len(), 3);
    assert!(!done.answers.is_empty());
}

#[test]
fn the_stage_reaches_the_engine_state() {
    for stage in [GuestStage::Vertex, GuestStage::Compute] {
        let state = FakeState::new(0);
        let script = Script::new(vec![vec![cbuf(1, 0), texture(1)]]);
        let done = exchange(stage, &script, &state).expect("within the cap");
        assert_eq!(
            state.calls(),
            asked(vec![(stage, cbuf(1, 0)), (stage, texture(1))])
        );
        assert_eq!(
            done.answers.cbuf_word(1, 0),
            Some(Some(STEP + stage_index(stage) * STAGE_SALT))
        );
    }
}

#[test]
fn missing_answers_are_recorded_as_missing() {
    let state = FakeState::new(0);
    let script = Script::new(vec![vec![cbuf(UNBOUND_BANK, 4), texture(INVALID_HANDLE)]]);
    let done = exchange(STAGE, &script, &state).expect("within the cap");
    assert_eq!(
        done.answers.get(cbuf(UNBOUND_BANK, 4)),
        Some(Answer::CbufWord(None))
    );
    assert_eq!(done.answers.cbuf_word(UNBOUND_BANK, 4), Some(None));
    assert_eq!(
        done.answers
            .texture_kind(INVALID_HANDLE, TextureUsage::Image),
        Some(None)
    );
    // Not asked: no answer at all.
    assert_eq!(done.answers.cbuf_word(UNBOUND_BANK, 8), None);
    assert_eq!(
        done.answers
            .texture_kind(INVALID_HANDLE + 1, TextureUsage::Image),
        None
    );
    assert_eq!(done.answers.get(texture(1)), None);
}

#[test]
fn accessors_read_their_own_kind() {
    // A constant-buffer question and a texture question with the same number do not answer
    // each other.
    let state = FakeState::new(0);
    let script = Script::new(vec![vec![cbuf(0, 7), texture(7)]]);
    let done = exchange(STAGE, &script, &state).expect("within the cap");
    assert_eq!(
        done.answers.texture_kind(7, TextureUsage::Image),
        Some(Some(kind_for(7)))
    );
    assert_eq!(
        done.answers.cbuf_word(0, 7),
        Some(Some(7 + STEP + stage_index(STAGE) * STAGE_SALT))
    );
}

#[test]
fn reanswer_asks_the_same_questions_in_question_order() {
    let first = FakeState::new(0);
    let script = Script::new(vec![vec![texture(5), cbuf(4, 0x30), cbuf(2, 0x10)]]);
    let done = exchange(STAGE, &script, &first).expect("within the cap");

    // Unchanged state: the same answers.
    let same = FakeState::new(0);
    assert_eq!(done.answers.reanswer(STAGE, &same), done.answers);

    // Changed state: new answers to the same questions, asked in question order, of the stage
    // given.
    let changed = FakeState::new(1);
    let again = done.answers.reanswer(GuestStage::Geometry, &changed);
    assert_eq!(
        changed.calls(),
        asked(vec![
            (GuestStage::Geometry, cbuf(2, 0x10)),
            (GuestStage::Geometry, cbuf(4, 0x30)),
            (GuestStage::Geometry, texture(5)),
        ])
    );
    assert_ne!(again, done.answers);
    assert_eq!(
        again.cbuf_word(2, 0x10),
        Some(Some(
            0x10 + STEP + stage_index(GuestStage::Geometry) * STAGE_SALT + 1
        ))
    );
    assert_eq!(again.len(), done.answers.len());
}

#[test]
fn the_texture_format_keeps_the_format_bits_only() {
    // MW(6:0) through MW(18:16) of clc797tex.h: bits 0 to 18.
    assert_eq!(TEXHEAD_FORMAT_MASK, 0x0007_ffff);
    assert_eq!(
        TextureFormat::from_texhead_word0(u32::MAX).bits(),
        TEXHEAD_FORMAT_MASK
    );
    // Bit 18 (the alpha data type's top bit) is kept; bit 19 (the first swizzle bit) is not.
    assert_eq!(TextureFormat::from_texhead_word0(1 << 18).bits(), 1 << 18);
    assert_eq!(TextureFormat::from_texhead_word0(1 << 19).bits(), 0);
    assert_eq!(
        TextureFormat::from_texhead_word0(0x1234_5678).bits(),
        0x1234_5678 & 0x0007_ffff
    );
}

/// Engine state whose only texture header, [`HANDLE`], has the kind `kind_for(HANDLE)` and the
/// format bits `format`: two of these differ only in the format.
struct OneHeader {
    format: u32,
}

const HANDLE: u32 = 6;

impl EngineState for OneHeader {
    fn cbuf_word(&self, _stage: GuestStage, _bank: u8, _offset: u32) -> Option<u32> {
        None
    }

    fn texture_kind(&self, _stage: GuestStage, handle: u32) -> Option<TextureKind> {
        (handle == HANDLE).then(|| TextureKind {
            format: TextureFormat::from_texhead_word0(self.format),
            ..kind_for(HANDLE)
        })
    }
}

/// Two format words with distinct non-zero bits in every format field (the components and the
/// four data types), so a field that is dropped still differs.
const FORMAT_A: u32 = 0x0004_6888;
const FORMAT_B: u32 = 0x0001_fa91;

#[test]
fn a_format_change_keeps_a_sampled_answer() {
    let script = || Script::new(vec![vec![sampled(HANDLE)]]);
    let a = exchange(STAGE, &script(), &OneHeader { format: FORMAT_A }).expect("one round");
    let b = exchange(STAGE, &script(), &OneHeader { format: FORMAT_B }).expect("one round");
    // The recorded answers, and so the shader entry's key, are equal.
    assert_eq!(a.answers, b.answers);
    assert_eq!(
        a.answers.texture_kind(HANDLE, TextureUsage::Sampled),
        Some(Some(TextureKind {
            format: TextureFormat::UNREAD,
            ..kind_for(HANDLE)
        }))
    );
    // Re-answered from the other format on a later draw: the same answers, so no
    // retranslation.
    assert_eq!(
        a.answers.reanswer(STAGE, &OneHeader { format: FORMAT_B }),
        a.answers
    );
    // The kind still counts: a header of another kind is a new answer.
    let other = TextureKind {
        depth: !kind_for(HANDLE).depth,
        ..kind_for(HANDLE)
    };
    struct OtherKind(TextureKind);
    impl EngineState for OtherKind {
        fn cbuf_word(&self, _: GuestStage, _: u8, _: u32) -> Option<u32> {
            None
        }
        fn texture_kind(&self, _: GuestStage, _: u32) -> Option<TextureKind> {
            Some(self.0)
        }
    }
    assert_ne!(a.answers.reanswer(STAGE, &OtherKind(other)), a.answers);
}

#[test]
fn a_format_change_changes_an_image_answer() {
    let script = || Script::new(vec![vec![texture(HANDLE)]]);
    let a = exchange(STAGE, &script(), &OneHeader { format: FORMAT_A }).expect("one round");
    let b = exchange(STAGE, &script(), &OneHeader { format: FORMAT_B }).expect("one round");
    assert_ne!(a.answers, b.answers);
    assert_eq!(
        a.answers.texture_kind(HANDLE, TextureUsage::Image),
        Some(Some(TextureKind {
            format: TextureFormat::from_texhead_word0(FORMAT_A),
            ..kind_for(HANDLE)
        }))
    );
    assert_ne!(
        a.answers.reanswer(STAGE, &OneHeader { format: FORMAT_B }),
        a.answers
    );
    assert_eq!(
        a.answers.reanswer(STAGE, &OneHeader { format: FORMAT_A }),
        a.answers
    );
}

#[test]
fn the_two_usages_of_one_handle_are_two_questions() {
    let state = FakeState::new(0);
    let script = Script::new(vec![vec![
        sampled(HANDLE),
        texture(HANDLE),
        sampled(HANDLE),
    ]]);
    let done = exchange(STAGE, &script, &state).expect("one round");
    // Each usage is answered once, from one engine call each.
    assert_eq!(
        state.calls(),
        asked(vec![(STAGE, sampled(HANDLE)), (STAGE, texture(HANDLE))])
    );
    // Recorded in question order: sampled before image (the usages' declaration order).
    assert_eq!(
        done.answers
            .iter()
            .map(|(question, _)| question)
            .collect::<Vec<_>>(),
        vec![sampled(HANDLE), texture(HANDLE)]
    );
    assert_eq!(
        done.answers.texture_kind(HANDLE, TextureUsage::Sampled),
        Some(Some(kind_for(HANDLE).for_usage(TextureUsage::Sampled)))
    );
    assert_eq!(
        done.answers.texture_kind(HANDLE, TextureUsage::Image),
        Some(Some(kind_for(HANDLE)))
    );
    // A handle asked about with one usage has no answer for the other.
    let only_sampled =
        exchange(STAGE, &Script::new(vec![vec![sampled(HANDLE)]]), &state).expect("one round");
    assert_eq!(
        only_sampled
            .answers
            .texture_kind(HANDLE, TextureUsage::Image),
        None
    );
}

#[test]
fn for_usage_drops_the_format_only_for_sampling() {
    let kind = TextureKind {
        format: TextureFormat::from_texhead_word0(FORMAT_A),
        ..kind_for(7)
    };
    assert_eq!(kind.for_usage(TextureUsage::Image), kind);
    assert_eq!(
        kind.for_usage(TextureUsage::Sampled),
        TextureKind {
            sampled: kind.sampled,
            depth: kind.depth,
            multisampled: kind.multisampled,
            buffer: kind.buffer,
            format: TextureFormat::UNREAD,
        }
    );
    assert_eq!(TextureFormat::UNREAD.bits(), 0);
}
