//! The questions the scan asks, and the refusals of missing answers.

use super::{BOUND_BANK, Engine, FLOAT, STAGE, answered, buffer, kind_of, parse, sample_of};
use crate::exchange::{
    Answers, Question, Questioner, SampledType, TextureFormat, TextureUsage, exchange,
};
use crate::opt::resources::{ResourceError, run, scan};

/// A bindless handle OR-ed from two words of bank 3.
fn pair() -> String {
    sample_of(
        "    %0: u32 = GetCbufU32 u32 3, u32 0x10\n    %1: u32 = GetCbufU32 u32 3, u32 0x14\n    \
         %2: u32 = ShiftLeftLogical32 %1, u32 0x10\n    %3: u32 = BitwiseOr32 %0, %2\n",
        "%3",
    )
}

#[test]
fn questions_take_two_rounds() {
    let function = parse(&pair());
    let found = scan(&function, buffer(BOUND_BANK)).expect("scans");
    // The words first; the handle only once both are answered.
    let none = Answers::new();
    assert_eq!(
        found.questions(&none),
        [
            Question::CbufWord {
                bank: 3,
                offset: 0x10
            },
            Question::CbufWord {
                bank: 3,
                offset: 0x14
            },
        ]
    );
    let engine = Engine::new()
        .word(3, 0x10, 0xabcd)
        .word(3, 0x14, 0x12)
        .kind(0x12_abcd, FLOAT);
    let exchanged = exchange(STAGE, &found, &engine).expect("exchanges");
    assert_eq!(exchanged.rounds, 2);
    assert_eq!(
        exchanged
            .answers
            .texture_kind(0x12_abcd, TextureUsage::Sampled),
        Some(Some(FLOAT))
    );
}

#[test]
fn a_storage_image_asks_with_image_usage() {
    let text = "block0:
    %0: u32 = GetCbufU32 u32 0x3, u32 0x10
    %1: u32x2 = CompositeConstructU32x2 u32 0x0, u32 0x0
    %2: u32x4 = ImageRead %0, %1 {type=2d, format=r32_uint}
    return
";
    let kind = kind_of(SampledType::UnsignedInt, false, 0x4_5678);
    let engine = Engine::new().word(3, 0x10, 0x77).kind(0x77, kind);
    let mut function = parse(text);
    let (found, answers) = answered(&function, buffer(BOUND_BANK), &engine);
    assert!(found.questions(&answers).contains(&Question::TextureKind {
        handle: 0x77,
        usage: TextureUsage::Image
    }));
    let (table, _) = run(&mut function, &found, &answers).expect("runs");
    let image = table.images().first().expect("an image");
    assert_eq!(
        image.header_format,
        TextureFormat::from_texhead_word0(0x4_5678)
    );
    assert_eq!(image.component, SampledType::UnsignedInt);
}

/// A questioner that asks only the words of another.
struct WordsOnly<'a>(&'a dyn Questioner);

impl Questioner for WordsOnly<'_> {
    fn questions(&self, answers: &Answers) -> Vec<Question> {
        self.0
            .questions(answers)
            .into_iter()
            .filter(|q| matches!(q, Question::CbufWord { .. }))
            .collect()
    }
}

#[test]
fn missing_answers_are_refused() {
    let text = pair();
    let function = parse(&text);
    let found = scan(&function, buffer(BOUND_BANK)).expect("scans");
    let refuse = |answers: &Answers| {
        let mut changed = function.clone();
        let error = run(&mut changed, &found, answers).expect_err("refused");
        assert_eq!(changed, function, "a refusal leaves the function");
        (error, error.to_string())
    };

    // Nothing answered: the first word is named.
    assert_eq!(
        refuse(&Answers::new()),
        (
            ResourceError::UnansweredWord {
                bank: 3,
                offset: 0x10
            },
            "constant buffer 3 at 0x10 was not answered".to_owned()
        )
    );
    // The second word missing from engine state.
    let engine = Engine::new().word(3, 0x10, 0xabcd);
    let answers = exchange(STAGE, &found, &engine).expect("exchanges").answers;
    assert_eq!(
        refuse(&answers),
        (
            ResourceError::NoCbufWord {
                bank: 3,
                offset: 0x14
            },
            "constant buffer 3 has no word at 0x14".to_owned()
        )
    );
    // Both words, but the handle not asked about.
    let engine = engine.word(3, 0x14, 0x12);
    let answers = exchange(STAGE, &WordsOnly(&found), &engine)
        .expect("exchanges")
        .answers;
    assert_eq!(
        refuse(&answers),
        (
            ResourceError::UnansweredKind {
                handle: 0x12_abcd,
                usage: TextureUsage::Sampled
            },
            "texture handle 0x12abcd (sampled) was not answered".to_owned()
        )
    );
    // Asked, but no texture header there.
    let answers = exchange(STAGE, &found, &engine).expect("exchanges").answers;
    assert_eq!(
        refuse(&answers),
        (
            ResourceError::NoTextureHeader { handle: 0x12_abcd },
            "texture handle 0x12abcd points to no texture header".to_owned()
        )
    );
    // The image usage reads as words too.
    assert_eq!(
        ResourceError::UnansweredKind {
            handle: 0,
            usage: TextureUsage::Image
        }
        .to_string(),
        "texture handle 0x0 (image) was not answered"
    );
    // With the header, it runs.
    let engine = engine.kind(0x12_abcd, FLOAT);
    let answers = exchange(STAGE, &found, &engine).expect("exchanges").answers;
    let mut changed = function.clone();
    assert!(run(&mut changed, &found, &answers).is_ok());
}
