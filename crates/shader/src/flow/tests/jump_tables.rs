//! Jump tables: the selector pattern on hand-built words (one change per condition), the
//! forward pass against a direct port of the reference's backward search, the questions, and
//! resolution with and without an extension through the exchange.

use std::collections::BTreeMap;

use proptest::prelude::*;

use super::super::classify::{FORM_IMM, LDC_OPCODE, with_form};
use super::super::jump_tables::{
    IMNMX_WORD2, LDC_WORD2, MAX_CLAMP, SHF_SCALE, SHF_WORD2, WORD1, WORD2, is_table_load,
    is_table_scale, table_clamp, tracked_brx,
};
use super::super::{
    FlowError, JumpTable, JumpTableQuestions, MAX_TABLE_ENTRIES, find_jump_tables, flow_graph,
    resolve_jump_tables,
};
use super::{Item, Program, exit, p, pure, r, size, to_start};
use crate::asm::assemble;
use crate::capture::{Capture, Extension, capture};
use crate::decode::integer::SHF_TYPE;
use crate::decode::{
    BRX_OPCODE, BrxSrc, CBUF_INDEX, CBUF_OFFSET, Field, Gpr, IMNMX_OPCODE, Instr, Instruction, Op,
    OpBrx, PredSrc, SHF_OPCODE, SRC2, UniformReg,
};
use crate::exchange::{Answers, EngineState, Question, TextureKind, exchange};
use crate::testing::config;
use crate::translate::GuestStage;

// The reference's literals (`program_capture.h`, `TrackCaptureIndirectBranches` @ e5fb3b97),
// written out to build words independently of the shipping constants.
const REF_LDC: u128 = 0xb82;
const REF_SHF: u128 = 0x819;
const REF_IMNMX: u128 = 0x817;
const REF_ALWAYS: u128 = 7 << 12;

/// `IMNMX.U32 dst, src, clamp, PT` as the reference matches it.
fn imnmx(dst: u8, src: u8, clamp: u32) -> Instruction {
    Instruction::from_bits(
        REF_IMNMX
            | REF_ALWAYS
            | u128::from(dst) << 16
            | u128::from(src) << 24
            | u128::from(clamp) << 32
            | 0x0380_0000 << 64,
    )
}

/// `SHF.L.U32 dst, src, 0x2, RZ`.
fn shf(dst: u8, src: u8) -> Instruction {
    Instruction::from_bits(
        REF_SHF
            | REF_ALWAYS
            | u128::from(dst) << 16
            | u128::from(src) << 24
            | 2 << 32
            | 0x6ff << 64,
    )
}

/// `LDC.32 dst, c[bank][src + offset]`.
fn ldc(dst: u8, src: u8, bank: u8, offset: u32) -> Instruction {
    Instruction::from_bits(
        REF_LDC
            | REF_ALWAYS
            | u128::from(dst) << 16
            | u128::from(src) << 24
            | (u128::from(bank) << 22 | u128::from(offset) << 6) << 32
            | 0x800 << 64,
    )
}

/// `word` with `bits` toggled.
fn toggle(word: Instruction, bits: u128) -> Instruction {
    Instruction::from_bits(word.bits() ^ bits)
}

/// `word` under guard `P0`.
fn guarded(word: Instruction) -> Instruction {
    toggle(word, 7 << 12)
}

/// The hand-built selector: clamp R3 to 5 into R4, scale into R5, load c[2][R5 + 0x40] into
/// R6, BRX R6, then the given tail.
fn selector(words: [Instruction; 3], brx: Item) -> Vec<Instruction> {
    let mut program = Program::new();
    for word in words {
        program.word(word);
    }
    program.item(brx).op(exit());
    program.build()
}

fn brx(reg: u8) -> Item {
    Item::Brx(BrxSrc::Reg(r(reg)))
}

fn standard() -> [Instruction; 3] {
    [imnmx(4, 3, 5), shf(5, 4), ldc(6, 5, 2, 0x40)]
}

#[test]
fn constants_match_the_reference_literals() {
    assert_eq!(LDC_OPCODE, 0xb82);
    assert_eq!(with_form(SHF_OPCODE, FORM_IMM), 0x819);
    assert_eq!(with_form(IMNMX_OPCODE, FORM_IMM), 0x817);
    assert_eq!(BRX_OPCODE, 0x949);
    assert_eq!(LDC_WORD2, 0x0000_0800);
    assert_eq!(SHF_SCALE, 2);
    assert_eq!(SHF_WORD2, 0x0000_06ff);
    assert_eq!(IMNMX_WORD2, 0x0380_0000);
    assert_eq!(MAX_CLAMP, 15);
    assert_eq!(u64::from(MAX_TABLE_ENTRIES), MAX_CLAMP + 1);
    assert_eq!(WORD1, Field::new(32, 32));
    assert_eq!(WORD2, Field::new(64, 32));
    // `(words[1] >> 22) & 0x1f` and `(words[1] >> 6) & 0xffff`.
    assert_eq!(CBUF_INDEX, Field::new(32 + 22, 5));
    assert_eq!(CBUF_OFFSET, Field::new(32 + 6, 16));
    // The `SHF` word: source 2 `RZ` and the type `U32` (3).
    let word = Instruction::from_bits(u128::from(SHF_WORD2) << 64);
    assert_eq!(word.field(SRC2), u64::from(Gpr::RZ.index()));
    assert_eq!(word.field(SHF_TYPE), 3);
}

#[test]
fn a_hand_built_selector_is_tracked() {
    let code = selector(standard(), brx(6));
    assert_eq!(
        find_jump_tables(&code),
        vec![JumpTable {
            branch: 3,
            bank: 2,
            offset: 0x40,
            entries: 6,
        }]
    );
}

#[test]
fn clamps_at_their_edges() {
    for (clamp, entries) in [(1, Some(2)), (15, Some(16)), (0, None), (16, None)] {
        let code = selector([imnmx(4, 3, clamp), shf(5, 4), ldc(6, 5, 2, 0x40)], brx(6));
        assert_eq!(
            find_jump_tables(&code).first().map(JumpTable::entries),
            entries,
            "clamp {clamp}"
        );
    }
}

#[test]
fn each_condition_of_the_pattern_is_needed() {
    let [clamp, scale, load] = standard();
    let cases: Vec<(&str, [Instruction; 3])> = vec![
        ("clamp guarded", [guarded(clamp), scale, load]),
        ("clamp signed", [toggle(clamp, 1 << 73), scale, load]),
        (
            "clamp register form",
            [toggle(clamp, 0x817 ^ 0x217), scale, load],
        ),
        (
            "clamp writes another register",
            [imnmx(7, 3, 5), scale, load],
        ),
        ("scale guarded", [clamp, guarded(scale), load]),
        ("scale by 3", [clamp, toggle(scale, 1 << 32), load]),
        ("scale signed type", [clamp, toggle(scale, 1 << 73), load]),
        ("scale reads another register", [clamp, shf(5, 7), load]),
        ("scale writes another register", [clamp, shf(7, 4), load]),
        ("load guarded", [clamp, scale, guarded(load)]),
        (
            "load of another type",
            [clamp, scale, toggle(load, 1 << 76)],
        ),
        (
            "load with another bit",
            [clamp, scale, toggle(load, 1 << 32)],
        ),
        ("load offset not a word", [clamp, scale, ldc(6, 5, 2, 0x41)]),
        (
            "load reads another register",
            [clamp, scale, ldc(6, 7, 2, 0x40)],
        ),
        (
            "load writes another register",
            [clamp, scale, ldc(7, 5, 2, 0x40)],
        ),
        ("clamp after the scale", [scale, clamp, load]),
        ("scale after the load", [clamp, load, scale]),
    ];
    for (what, words) in cases {
        assert_eq!(find_jump_tables(&selector(words, brx(6))), vec![], "{what}");
    }
    // The branch itself: guarded, with a predicate, not from the start, uniform.
    let branch = |guard: PredSrc, cond: PredSrc, extra: i64, src: BrxSrc| {
        assemble(&Instr {
            guard,
            op: Op::Brx(OpBrx {
                src,
                offset: crate::decode::BranchOffset::from_words(to_start(3).words() + extra),
                cond,
            }),
        })
        .expect("encodable")
    };
    let ur6 = BrxSrc::UniformReg(UniformReg::new(6).expect("ur6"));
    for word in [
        branch(p(0), PredSrc::TRUE, 0, BrxSrc::Reg(r(6))),
        branch(PredSrc::TRUE, p(0), 0, BrxSrc::Reg(r(6))),
        branch(PredSrc::TRUE, PredSrc::TRUE, 4, BrxSrc::Reg(r(6))),
        branch(PredSrc::TRUE, PredSrc::TRUE, 0, ur6),
    ] {
        let mut code = standard().to_vec();
        code.push(word);
        assert_eq!(find_jump_tables(&code), vec![]);
    }
    // The control case: the same builder with nothing changed is tracked.
    let mut code = standard().to_vec();
    code.push(branch(PredSrc::TRUE, PredSrc::TRUE, 0, BrxSrc::Reg(r(6))));
    assert_eq!(find_jump_tables(&code).len(), 1);
}

#[test]
fn undecodable_brx_is_not_tracked() {
    let mut code = selector(standard(), brx(6));
    // Bit 33 is part of the reference's displacement read (`words[1] & ~3U` drops it) but no
    // field of the decoder's `BRX`.
    code[3] = toggle(code[3], 1 << 33);
    assert!(crate::decode::decode(code[3]).is_err());
    assert_eq!(find_jump_tables(&code), vec![]);
}

#[test]
fn the_nearest_match_of_each_step_is_used() {
    // Two loads of R6: the nearer one's table.
    let code = selector(standard(), brx(6));
    let mut nearer = code[..3].to_vec();
    nearer.push(ldc(6, 5, 9, 0x80));
    let mut program = Program::new();
    for word in nearer {
        program.word(word);
    }
    let code = program.item(brx(6)).op(exit()).build();
    assert_eq!(find_jump_tables(&code)[0].bank(), 9);
    // Another write of R6 between the load and the branch does not stop the search.
    let mut program = Program::new();
    for word in standard() {
        program.word(word);
    }
    let code = program.op(pure(6)).item(brx(6)).op(exit()).build();
    assert_eq!(find_jump_tables(&code)[0].offset(), 0x40);
    // The nearest load's chain fails: the branch is not tracked, though an earlier load's
    // would succeed.
    let mut program = Program::new();
    for word in standard() {
        program.word(word);
    }
    let code = program
        .word(ldc(6, 7, 3, 0x80))
        .item(brx(6))
        .op(exit())
        .build();
    assert_eq!(find_jump_tables(&code), vec![]);
}

/// A direct port of `TrackCaptureIndirectBranches`' backward search, with this module's
/// pattern tests.
fn backward(code: &[Instruction]) -> Vec<JumpTable> {
    let reg = |word: Instruction, start: u32| {
        u8::try_from(word.field(Field::new(start, 8))).expect("eight bits")
    };
    let mut tables = Vec::new();
    for (branch, &word) in code.iter().enumerate() {
        let index = u32::try_from(branch).expect("small");
        let Some(source) = tracked_brx(word, index) else {
            continue;
        };
        let Some(load) = (0..branch)
            .rev()
            .find(|&i| is_table_load(code[i]) && usize::from(reg(code[i], 16)) == source)
        else {
            continue;
        };
        let selector = reg(code[load], 24);
        let Some(scale) = (0..load)
            .rev()
            .find(|&i| is_table_scale(code[i]) && reg(code[i], 16) == selector)
        else {
            continue;
        };
        let bounded = reg(code[scale], 24);
        let Some(clamp) = (0..scale).rev().find_map(|i| {
            (reg(code[i], 16) == bounded)
                .then(|| table_clamp(code[i]))
                .flatten()
        }) else {
            continue;
        };
        tables.push(JumpTable {
            branch: index,
            bank: u8::try_from(code[load].field(CBUF_INDEX)).expect("five bits"),
            offset: u32::try_from(code[load].field(CBUF_OFFSET)).expect("16 bits"),
            entries: u32::try_from(clamp).expect("small") + 1,
        });
    }
    tables
}

/// One item of a generated program: a pattern word over registers 0 to 3, or filler.
fn any_item() -> impl Strategy<Value = Item> {
    let reg = || 0_u8..4;
    prop_oneof![
        (reg(), reg(), 0_u32..17).prop_map(|(d, s, c)| Item::Word(imnmx(d, s, c))),
        (reg(), reg()).prop_map(|(d, s)| Item::Word(shf(d, s))),
        (reg(), reg(), 0_u8..32, 0_u32..64).prop_map(|(d, s, b, o)| Item::Word(ldc(d, s, b, o))),
        reg().prop_map(|s| Item::Brx(BrxSrc::Reg(r(s)))),
        reg().prop_map(|d| Item::Word(assemble(&pure(d)).expect("encodable"))),
    ]
}

proptest! {
    #![proptest_config(config())]

    #[test]
    fn forward_pass_matches_the_backward_search(
        items in proptest::collection::vec(any_item(), 1..48)
    ) {
        let mut program = Program::new();
        for item in items {
            program.item(item);
        }
        let code = program.build();
        prop_assert_eq!(find_jump_tables(&code), backward(&code));
    }
}

#[test]
fn a_table_asks_for_each_entry_and_reads_its_words() {
    let table = JumpTable {
        branch: 3,
        bank: 2,
        offset: 0x40,
        entries: 3,
    };
    let questions: Vec<Question> = table.questions().collect();
    assert_eq!(
        questions,
        [0x40, 0x44, 0x48].map(|offset| Question::CbufWord { bank: 2, offset })
    );
    // Unanswered: the first missing word is named.
    assert_eq!(
        table.words(&Answers::new()),
        Err(FlowError::MissingAnswer {
            bank: 2,
            offset: 0x40
        })
    );
    let engine = Words::new(&[((2, 0x40), 0x111), ((2, 0x48), 0x333)]);
    let answers = exchange(GuestStage::Compute, &Asks(questions), &engine)
        .expect("answered")
        .answers;
    // No word at 0x44: it reads as 0.
    assert_eq!(table.words(&answers), Ok(vec![0x111, 0, 0x333]));
}

/// A questioner that asks a fixed list.
struct Asks(Vec<Question>);

impl crate::exchange::Questioner for Asks {
    fn questions(&self, _: &Answers) -> Vec<Question> {
        self.0.clone()
    }
}

/// Engine state holding some constant-buffer words.
struct Words(BTreeMap<(u8, u32), u32>);

impl Words {
    fn new(words: &[((u8, u32), u32)]) -> Self {
        Self(words.iter().copied().collect())
    }
}

impl EngineState for Words {
    fn cbuf_word(&self, _: GuestStage, bank: u8, offset: u32) -> Option<u32> {
        self.0.get(&(bank, offset)).copied()
    }

    fn texture_kind(&self, _: GuestStage, _: u32) -> Option<TextureKind> {
        None
    }
}

/// The nested-table program (see `nested_tables_take_two_rounds`).
fn nested() -> Vec<u8> {
    let mut program = Program::new();
    program
        .word(imnmx(4, 3, 1))
        .word(shf(5, 4))
        .word(ldc(6, 5, 2, 0x10))
        .item(brx(6))
        .op(exit())
        .op(exit())
        .word(imnmx(4, 3, 1))
        .word(shf(5, 4))
        .word(ldc(7, 5, 3, 0x20))
        .item(brx(7))
        .op(exit())
        .op(exit());
    program
        .build()
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .collect()
}

/// The words of both tables of [`nested`]: the first names 5 and 6, the second 10 and 11.
fn nested_words() -> Words {
    Words::new(&[
        ((2, 0x10), 5 * size()),
        ((2, 0x14), 6 * size()),
        ((3, 0x20), 10 * size()),
        ((3, 0x24), 11 * size()),
    ])
}

#[test]
fn nested_tables_take_two_rounds() {
    // 0..3 the first selector and BRX R6; 4 EXIT, where the capture ends; 5 EXIT; 6..9 a second
    // selector and BRX R7; 10 EXIT; 11 EXIT. The first table names 5 and 6, past the capture's
    // end; the second, found only in the extended code, names 10 and 11.
    let bytes = nested();
    let mut capture = capture(&bytes[..], GuestStage::Compute).expect("captured");
    assert_eq!(capture.instruction_count(), 5);
    let questioner = JumpTableQuestions::new(&capture, &bytes[..]);
    let exchanged = exchange(GuestStage::Compute, &questioner, &nested_words()).expect("exchanged");
    assert_eq!(exchanged.rounds, 2);
    assert_eq!(exchanged.answers.len(), 4);
    let resolved =
        resolve_jump_tables(&mut capture, &bytes[..], &exchanged.answers).expect("resolved");
    assert_eq!(resolved.extension, Extension::Extended { from: 5 });
    assert_eq!(resolved.base, 0);
    // Extended through instruction 10, the first that ends the program at or past 6.
    assert_eq!(capture.instruction_count(), 11);
    let targets: Vec<(u32, Vec<u32>)> = resolved
        .tables
        .iter()
        .map(|table| (table.table.branch(), table.targets.clone()))
        .collect();
    // Instruction 11 lies past the extended capture: ignored.
    assert_eq!(targets, vec![(3, vec![5, 6]), (9, vec![10])]);
    assert_eq!(resolved.tables[1].words, vec![10 * size(), 11 * size()]);
    let graph =
        super::super::capture_flow_graph(&capture, &resolved.branch_tables()).expect("accepted");
    let code: Vec<Instruction> = capture.instructions().collect();
    assert_eq!(
        flow_graph(&code, &resolved.branch_tables()).as_ref(),
        Ok(&graph)
    );
    // The first BRX goes to 5 and 6, the second to 10.
    assert_eq!(graph.blocks().len(), 4);
    graph
        .build_ir()
        .expect("IR")
        .function
        .verify()
        .expect("verified");
}

#[test]
fn a_missing_answer_leaves_the_capture_unchanged() {
    let bytes = nested();
    let mut capture = capture(&bytes[..], GuestStage::Compute).expect("captured");
    let before = capture.clone();
    // Only the first table answered: the extension would ask the second.
    let first = Asks(vec![
        Question::CbufWord {
            bank: 2,
            offset: 0x10,
        },
        Question::CbufWord {
            bank: 2,
            offset: 0x14,
        },
    ]);
    let answers = exchange(GuestStage::Compute, &first, &nested_words())
        .expect("answered")
        .answers;
    assert_eq!(
        resolve_jump_tables(&mut capture, &bytes[..], &answers),
        Err(FlowError::MissingAnswer {
            bank: 3,
            offset: 0x20
        })
    );
    assert_eq!(capture, before);
    // The questioner asks for the second table once the first is answered.
    let asked = JumpTableQuestions::new(&capture, &bytes[..]).questions_for(&answers);
    assert_eq!(asked.len(), 4);
    // With nothing answered, it asks for the first table only.
    let asked = JumpTableQuestions::new(&capture, &bytes[..]).questions_for(&Answers::new());
    assert_eq!(asked.len(), 2);
}

impl<R: crate::capture::ProgramReader + ?Sized> JumpTableQuestions<'_, R> {
    fn questions_for(&self, answers: &Answers) -> Vec<Question> {
        crate::exchange::Questioner::questions(self, answers)
    }
}

#[test]
fn an_extension_that_cannot_be_read_is_abandoned() {
    let bytes = nested();
    // The program's bytes stop after instruction 7: the extension cannot reach an end.
    let short = &bytes[..8 * Instruction::SIZE];
    let mut capture = capture(short, GuestStage::Compute).expect("captured");
    let answers = exchange(
        GuestStage::Compute,
        &JumpTableQuestions::new(&capture, short),
        &nested_words(),
    )
    .expect("answered")
    .answers;
    let resolved = resolve_jump_tables(&mut capture, short, &answers).expect("resolved");
    assert!(matches!(resolved.extension, Extension::Abandoned(_)));
    assert_eq!(capture.instruction_count(), 5);
    // Both targets lie past the capture: the BRX has none.
    assert_eq!(resolved.tables[0].targets, Vec::<u32>::new());
    let code: Vec<Instruction> = capture.instructions().collect();
    assert_eq!(
        flow_graph(&code, &resolved.branch_tables()),
        Err(FlowError::NoTarget { index: 3 })
    );
}

#[test]
fn a_graphics_table_counts_from_the_header_unless_a_value_is_below_it() {
    let header = u32::try_from(crate::sph::HEADER_BYTES).expect("small");
    let mut program = Program::new();
    for word in standard() {
        program.word(word);
    }
    let code = program.item(brx(6)).op(exit()).op(exit()).build();
    let mut bytes = vec![0; crate::sph::HEADER_BYTES];
    bytes.extend(code.iter().flat_map(|word| word.to_le_bytes()));
    let resolve = |values: [u32; 6]| -> (u32, Vec<u32>) {
        let mut capture: Capture = capture(&bytes[..], GuestStage::Pixel).expect("captured");
        let engine = Words::new(
            &values
                .iter()
                .zip(0_u32..)
                .map(|(&value, entry)| ((2, 0x40 + 4 * entry), value))
                .collect::<Vec<_>>(),
        );
        let answers = exchange(
            GuestStage::Pixel,
            &JumpTableQuestions::new(&capture, &bytes[..]),
            &engine,
        )
        .expect("answered")
        .answers;
        let resolved = resolve_jump_tables(&mut capture, &bytes[..], &answers).expect("resolved");
        (resolved.base, resolved.tables[0].targets.clone())
    };
    // Values from the program's first byte: the header's size plus 16 per instruction.
    assert_eq!(
        resolve([header + 4 * size(), header + 5 * size(), 0, 0, 0, 0]),
        (header, vec![4, 5])
    );
    // A value below the header's size counts from instruction 0 for every value.
    assert_eq!(
        resolve([4 * size(), 5 * size(), 0, 0, 0, 0]),
        (0, vec![4, 5])
    );
}

#[test]
fn a_table_has_at_most_max_entries() {
    // The selector's clamp is the only way to make a table: the largest it accepts gives
    // MAX_TABLE_ENTRIES entries and as many questions; one more is not tracked.
    let largest = MAX_TABLE_ENTRIES - 1;
    let code = selector(
        [imnmx(4, 3, largest), shf(5, 4), ldc(6, 5, 2, 0x40)],
        brx(6),
    );
    let tables = find_jump_tables(&code);
    assert_eq!(tables.len(), 1);
    assert_eq!(tables[0].entries(), MAX_TABLE_ENTRIES);
    assert_eq!(
        tables[0].questions().count(),
        usize::try_from(MAX_TABLE_ENTRIES).expect("small")
    );
    let code = selector(
        [
            imnmx(4, 3, MAX_TABLE_ENTRIES),
            shf(5, 4),
            ldc(6, 5, 2, 0x40),
        ],
        brx(6),
    );
    assert_eq!(find_jump_tables(&code), vec![]);
}

#[test]
fn the_getters_read_the_tracked_fields() {
    let code = selector(standard(), brx(6));
    let table = find_jump_tables(&code)[0];
    assert_eq!(
        (
            table.branch(),
            table.bank(),
            table.offset(),
            table.entries()
        ),
        (3, 2, 0x40, 6)
    );
}

#[test]
fn a_graphics_extension_keeps_the_first_tables_base() {
    // The nested program behind a header. The first table's values count from the header, so
    // the base is the header's size. The second table, found only in the extended code, holds
    // a non-zero value below the header's size: read with the first base, as the reference
    // does, it names nothing and the other value names 10; a base recomputed from it would be
    // 0, and the values would name 4 and an instruction past the capture instead.
    let header = u32::try_from(crate::sph::HEADER_BYTES).expect("small");
    let mut bytes = vec![0; crate::sph::HEADER_BYTES];
    bytes.extend(nested());
    let mut capture = capture(&bytes[..], GuestStage::Pixel).expect("captured");
    assert_eq!(capture.instruction_count(), 5);
    let low = 4 * size();
    assert!(low != 0 && low < header);
    let engine = Words::new(&[
        ((2, 0x10), header + 5 * size()),
        ((2, 0x14), header + 6 * size()),
        ((3, 0x20), header + 10 * size()),
        ((3, 0x24), low),
    ]);
    let exchanged = exchange(
        GuestStage::Pixel,
        &JumpTableQuestions::new(&capture, &bytes[..]),
        &engine,
    )
    .expect("exchanged");
    assert_eq!(exchanged.rounds, 2);
    let resolved =
        resolve_jump_tables(&mut capture, &bytes[..], &exchanged.answers).expect("resolved");
    assert_eq!(resolved.extension, Extension::Extended { from: 5 });
    assert_eq!(resolved.base, header);
    assert_eq!(capture.instruction_count(), 11);
    let targets: Vec<(u32, Vec<u32>)> = resolved
        .tables
        .iter()
        .map(|table| (table.table.branch(), table.targets.clone()))
        .collect();
    assert_eq!(targets, vec![(3, vec![5, 6]), (9, vec![10])]);
}
