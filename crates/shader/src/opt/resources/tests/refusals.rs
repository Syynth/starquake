//! Every refusal with its exact reason, and that a refusal changes nothing.

use super::{
    BOUND_BANK, Engine, FLOAT, STAGE, answered, buffer, parse, sample_of, scan_error, value,
};
use std::fmt::Write as _;

use crate::exchange::exchange;
use crate::ir::{
    Builder, Flags, Function, Imm, Inst, Opcode, Operand, Operation, TextureInfo, TextureType,
};
use crate::opt::resources::{MAX_TRACE_DEPTH, ResourceError, Untraced, run, scan};
use crate::translate::texture::ABSENT;

/// The word every case reads, and a second one.
const WORD: &str = "    %0: u32 = GetCbufU32 u32 3, u32 0x10\n";
const WORD2: &str = "    %1: u32 = GetCbufU32 u32 3, u32 0x14\n";

/// `(lines, handle, reason, the error's text)` for each reason a sample's handle can have.
fn cases() -> Vec<(String, &'static str, Untraced, String)> {
    let r = |n: u32| format!("    %{n}: u32 = GetRegister r{n}\n");
    let e = |why: &str| format!("%60: the texture handle cannot be traced: {why}");
    vec![
        (
            "    %0: u32 = ShiftLeftLogical32 u32 0x5, u32 0x2\n".to_owned(),
            "%0",
            Untraced::Immediate,
            e("it is an immediate, which no constant buffer holds"),
        ),
        (
            format!("{WORD}    %1: u32 = IAdd32 %0, u32 0x4\n"),
            "%1",
            Untraced::Opcode {
                at: value(1),
                opcode: Opcode::IAdd32,
            },
            e("it comes from IAdd32 at %1"),
        ),
        (
            "    %0: u32 = LoadGlobal32 u64 0x1000\n".to_owned(),
            "%0",
            Untraced::GlobalMemory {
                at: value(0),
                opcode: Opcode::LoadGlobal32,
            },
            e("it is loaded from storage or global memory by LoadGlobal32 at %0"),
        ),
        (
            format!("{WORD}    %1: u32 = LoadStorage32 u32 0x0, %0\n"),
            "%1",
            Untraced::GlobalMemory {
                at: value(1),
                opcode: Opcode::LoadStorage32,
            },
            e("it is loaded from storage or global memory by LoadStorage32 at %1"),
        ),
        (
            "    %0: f32 = GetAttribute a0x80, u32 0x0\n    %1: u32 = BitCastU32F32 %0\n"
                .to_owned(),
            "%1",
            Untraced::Attribute,
            e("it comes from a vertex attribute"),
        ),
        (
            "    %0: u32 = GetAttributeU32 a0x80, u32 0x0\n".to_owned(),
            "%0",
            Untraced::Attribute,
            e("it comes from a vertex attribute"),
        ),
        (
            format!("{}    %1: u32 = GetCbufU32 %0, u32 0x10\n", r(0)),
            "%1",
            Untraced::VariableBank(value(1)),
            e("it is read at %1 from a constant buffer named by a value"),
        ),
        (
            "    %0: u32 = GetCbufU32 u32 0x20, u32 0x10\n".to_owned(),
            "%0",
            Untraced::BankPastLast(32),
            e("it is read from constant buffer 32; banks stop at 31"),
        ),
    ]
}

/// The shift and mask cases of [`cases`].
fn shift_and_mask_cases() -> Vec<(String, &'static str, Untraced, String)> {
    let r = |n: u32| format!("    %{n}: u32 = GetRegister r{n}\n");
    let e = |why: &str| format!("%60: the texture handle cannot be traced: {why}");
    vec![
        (
            format!("{WORD}{}    %2: u32 = ShiftLeftLogical32 %0, %1\n", r(1)),
            "%2",
            Untraced::ShiftNotImmediate(value(2)),
            e("%2 shifts it by a value"),
        ),
        (
            format!("{WORD}    %1: u32 = ShiftLeftLogical32 %0, u32 0x20\n"),
            "%1",
            Untraced::ShiftTooWide {
                at: value(1),
                shift: 32,
            },
            e("%1 shifts it by 32 bits"),
        ),
        (
            format!(
                "{WORD}    %1: u32 = ShiftLeftLogical32 %0, u32 0x2\n    %2: u32 = ShiftLeftLogical32 %1, u32 0x3\n"
            ),
            "%2",
            Untraced::ShiftOfShifted(value(2)),
            e("%2 shifts a word that is shifted already"),
        ),
        (
            format!(
                "{WORD}{WORD2}    %2: u32 = BitwiseOr32 %0, %1\n    %3: u32 = ShiftLeftLogical32 %2, u32 0x2\n"
            ),
            "%3",
            Untraced::ShiftOfPair(value(3)),
            e("%3 shifts a handle built from two words"),
        ),
        (
            format!("{WORD}{}    %2: u32 = BitwiseAnd32 %0, %1\n", r(1)),
            "%2",
            Untraced::MaskNotImmediate(value(2)),
            e("%2 masks it with a value"),
        ),
        (
            // Eden's mask from constant buffer 1, which is not followed here.
            format!(
                "{WORD}    %1: u32 = GetCbufU32 u32 1, u32 0x0\n    %2: u32 = BitwiseAnd32 %0, %1\n"
            ),
            "%2",
            Untraced::MaskNotImmediate(value(2)),
            e("%2 masks it with a value"),
        ),
        (
            "    %0: u32 = BitwiseAnd32 u32 0x5, u32 0xff\n".to_owned(),
            "%0",
            Untraced::Immediate,
            e("it is an immediate, which no constant buffer holds"),
        ),
        (
            format!(
                "{WORD}{WORD2}    %2: u32 = BitwiseOr32 %0, %1\n    %3: u32 = BitwiseAnd32 %2, u32 0xff\n"
            ),
            "%3",
            Untraced::MaskOfPair(value(3)),
            e("%3 masks a handle built from two words"),
        ),
    ]
}

/// The rest of [`cases`]: combined words, register offsets, pair elements and depth.
fn more_cases() -> Vec<(String, &'static str, Untraced, String)> {
    let r = |n: u32| format!("    %{n}: u32 = GetRegister r{n}\n");
    let e = |why: &str| format!("%60: the texture handle cannot be traced: {why}");
    vec![
        (
            format!(
                "{WORD}{WORD2}    %2: u32 = BitwiseOr32 %0, %1\n    %3: u32 = GetCbufU32 u32 3, u32 0x18\n    \
                 %4: u32 = BitwiseOr32 %2, %3\n"
            ),
            "%4",
            Untraced::MoreThanTwoWords(value(2)),
            e("%2 builds it from more than two words"),
        ),
        (
            format!(
                "{}{}    %2: u32 = GetCbufU32 u32 3, %0\n    %3: u32 = GetCbufU32 u32 4, %1\n    \
                 %4: u32 = BitwiseOr32 %2, %3\n",
                r(0),
                r(1)
            ),
            "%4",
            Untraced::TwoArrays(value(4)),
            e("%4 combines two handle arrays indexed at run time"),
        ),
        (
            "    %0: u32 = GetCbufU32 u32 5, u32 0x0\n    %1: u32 = IAdd32 %0, u32 0x4000\n    \
             %2: u32 = GetCbufU32 u32 3, %1\n    %3: u32 = GetCbufU32 u32 4, %1\n    \
             %4: u32 = BitwiseOr32 %2, %3\n"
                .to_owned(),
            "%4",
            Untraced::TwoIndexWords(value(4)),
            e("%4 combines two handle arrays indexed by constant-buffer words"),
        ),
        (
            format!(
                "{}{}    %2: u32 = IAdd32 %0, %1\n    %3: u32 = GetCbufU32 u32 3, %2\n",
                r(0),
                r(1)
            ),
            "%3",
            Untraced::DynamicOffset(value(3)),
            e("%3 reads it at a sum of two values"),
        ),
        (
            "    %0: u32x2 = GetCbufU32x2 u32 3, u32 0x10\n    %1: u32 = CompositeExtractU32x2 %0, u32 0x2\n"
                .to_owned(),
            "%1",
            Untraced::PairElement(value(1)),
            e("%1 takes it from a pair other than one constant-buffer read at an immediate offset"),
        ),
        (
            "    %0: u32x2 = CompositeConstructU32x2 u32 0x1, u32 0x2\n    %1: u32 = CompositeExtractU32x2 %0, u32 0x1\n"
                .to_owned(),
            "%1",
            Untraced::PairElement(value(1)),
            e("%1 takes it from a pair other than one constant-buffer read at an immediate offset"),
        ),
        (
            format!(
                "{}    %1: u32x2 = GetCbufU32x2 u32 3, %0\n    %2: u32 = CompositeExtractU32x2 %1, u32 0x1\n",
                r(0)
            ),
            "%2",
            Untraced::PairElement(value(2)),
            e("%2 takes it from a pair other than one constant-buffer read at an immediate offset"),
        ),
        (
            "    %0: u32x2 = GetCbufU32x2 u32 0x20, u32 0x10\n    %1: u32 = CompositeExtractU32x2 %0, u32 0x1\n"
                .to_owned(),
            "%1",
            Untraced::BankPastLast(32),
            e("it is read from constant buffer 32; banks stop at 31"),
        ),
        (
            {
                let mut lines = WORD.to_owned();
                for k in 1..=MAX_TRACE_DEPTH + 1 {
                    writeln!(
                        lines,
                        "    %{k}: u32 = BitwiseAnd32 %{}, u32 0xffffffff",
                        k - 1
                    )
                    .expect("writes");
                }
                lines
            },
            "%17",
            Untraced::TooDeep,
            e("it is more than 16 operations deep"),
        ),
    ]
}

#[test]
fn every_reason_reads_as_written() {
    assert_eq!(MAX_TRACE_DEPTH, 16, "the last case names its handle %17");
    for (lines, handle, why, text) in cases()
        .into_iter()
        .chain(shift_and_mask_cases())
        .chain(more_cases())
    {
        let error = scan_error(&sample_of(&lines, handle));
        assert_eq!(
            error,
            ResourceError::Untraced { at: value(60), why },
            "{lines}"
        );
        assert_eq!(error.to_string(), text);
    }
}

#[test]
fn a_phi_is_refused() {
    let text = "block0:
    %0: u32 = GetCbufU32 u32 0x3, u32 0x10
    %1: u32 = GetCbufU32 u32 0x3, u32 0x14
    %2: u1 = INotEqual %0, u32 0x0
    branch %2, block1, block2
block1:
    jump block2
block2:
    %3: u32 = Phi [block0: %0], [block1: %1]
    %4: f32x2 = CompositeConstructF32x2 f32 0x0, f32 0x0
    %5: f32x4 = ImageSampleImplicitLod %3, %4, u32 0x0, u32 0x0 {type=2d}
    return
";
    let error = scan(&parse(text), buffer(BOUND_BANK)).expect_err("refused");
    assert_eq!(
        error,
        ResourceError::Untraced {
            at: value(5),
            why: Untraced::Phi(value(3))
        }
    );
    assert_eq!(
        error.to_string(),
        "%5: the texture handle cannot be traced: it passes through the phi %3"
    );
}

/// Eden's search would find the read under the add and bind the word as the handle; here the
/// add is named.
#[test]
fn an_add_after_the_read_is_refused() {
    let text = sample_of(&format!("{WORD}    %1: u32 = IAdd32 u32 0x1, %0\n"), "%1");
    assert_eq!(
        scan_error(&text),
        ResourceError::Untraced {
            at: value(60),
            why: Untraced::Opcode {
                at: value(1),
                opcode: Opcode::IAdd32
            }
        }
    );
}

/// An argument of the wrong kind, which only an unverified function has.
#[test]
fn a_malformed_read_is_refused() {
    let mut function = Function::new();
    let entry = function.entry();
    let read = function
        .push(
            entry,
            Inst::Op(Operation::new(
                Opcode::GetCbufU32,
                vec![Operand::Imm(Imm::U8(3)), Operand::Imm(Imm::U32(0x10))],
            )),
        )
        .expect("pushes");
    let mut sample = Operation::new(
        Opcode::ImageSampleImplicitLod,
        vec![Operand::Value(read), ABSENT, ABSENT, ABSENT],
    );
    sample.flags = Flags::Texture(TextureInfo::default());
    let at = function.push(entry, Inst::Op(sample)).expect("pushes");
    let error = scan(&function, buffer(BOUND_BANK)).expect_err("refused");
    assert_eq!(
        error,
        ResourceError::Untraced {
            at,
            why: Untraced::Malformed(read)
        }
    );
    assert_eq!(
        error.to_string(),
        "%1: the texture handle cannot be traced: %0 has an argument missing or of the wrong kind"
    );
    // A texture operation with no handle at all.
    let mut empty = Function::new();
    let entry = empty.entry();
    let at = empty
        .push(
            entry,
            Inst::Op(Operation::new(Opcode::ImageSampleImplicitLod, Vec::new())),
        )
        .expect("pushes");
    assert_eq!(
        scan(&empty, buffer(BOUND_BANK)).expect_err("refused"),
        ResourceError::Malformed(at)
    );
}

#[test]
fn a_storage_image_of_two_words_is_refused_and_nothing_changes() {
    let text = format!(
        "block0:\n{WORD}{WORD2}    %2: u32 = BitwiseOr32 %0, %1\n    \
         %3: u32x2 = CompositeConstructU32x2 u32 0x0, u32 0x0\n    \
         %4: u32 = GetRegister r0\n    %5: u32 = GetCbufU32 u32 3, %4\n    \
         %6: u32x4 = ImageRead %2, %3 {{type=2d, format=r32_uint}}\n    return\n"
    );
    let engine = Engine::new()
        .word(3, 0x10, 0x1)
        .word(3, 0x14, 0x2)
        .kind(0x3, FLOAT);
    let function = parse(&text);
    let (found, answers) = answered(&function, buffer(BOUND_BANK), &engine);
    let mut changed = function.clone();
    let error = run(&mut changed, &found, &answers).expect_err("refused");
    assert_eq!(error, ResourceError::SeparateSampler(value(6)));
    assert_eq!(
        error.to_string(),
        "%6: a storage image whose handle is built from two words"
    );
    // The indexed read is not bounded either: the refusal comes before any change.
    assert_eq!(changed, function);
}

#[test]
fn a_changed_function_is_stale() {
    let text = sample_of(WORD, "%0");
    let engine = Engine::new().word(3, 0x10, 0x9).kind(0x9, FLOAT);
    let function = parse(&text);
    let (found, answers) = answered(&function, buffer(BOUND_BANK), &engine);
    let mut changed = function.clone();
    if let Some(slot) = changed.inst_mut(value(60)) {
        *slot = Inst::Op(Operation::new(Opcode::Void, Vec::new()));
    }
    let before = changed.clone();
    let error = run(&mut changed, &found, &answers).expect_err("refused");
    assert_eq!(error, ResourceError::Stale(value(60)));
    assert_eq!(
        error.to_string(),
        "%60: the function changed since it was scanned"
    );
    assert_eq!(changed, before);
}

/// One more descriptor of one kind than `descriptor_index` can name is refused; exactly as
/// many is accepted.
#[test]
fn descriptor_indices_at_their_edge() {
    let limit = u32::from(u16::MAX) + 1;
    for (count, refused) in [(limit, false), (limit + 1, true)] {
        let mut function = Function::new();
        let entry = function.entry();
        let mut b = Builder::new(&mut function, entry);
        let coords = b
            .op(
                Opcode::CompositeConstructF32x2,
                &[Operand::Imm(Imm::F32(0)), Operand::Imm(Imm::F32(0))],
            )
            .expect("builds");
        let info = Flags::Texture(TextureInfo {
            ty: TextureType::Color2D,
            ..TextureInfo::default()
        });
        let mut last = None;
        let mut engine = Engine::new().kind(1, FLOAT);
        for k in 0..count {
            let read = b
                .op(
                    Opcode::GetCbufU32,
                    &[Operand::Imm(Imm::U32(3)), Operand::Imm(Imm::U32(4 * k))],
                )
                .expect("builds");
            engine = engine.word(3, 4 * k, 1);
            last = Some(
                b.op_with(
                    Opcode::ImageSampleImplicitLod,
                    &[Operand::Value(read), Operand::Value(coords), ABSENT, ABSENT],
                    info,
                )
                .expect("builds"),
            );
        }
        b.terminate(crate::ir::Terminator::Return).expect("ends");
        let found = scan(&function, buffer(BOUND_BANK)).expect("scans");
        let answers = exchange(STAGE, &found, &engine).expect("exchanges").answers;
        let outcome = run(&mut function, &found, &answers);
        if refused {
            let at = last.expect("one");
            assert_eq!(outcome.err(), Some(ResourceError::TooManyDescriptors(at)));
            assert_eq!(
                ResourceError::TooManyDescriptors(at).to_string(),
                format!("{at}: more texture descriptors of one kind than 65536")
            );
        } else {
            assert_eq!(outcome.map(|(t, _)| t.textures().len()).ok(), Some(65536));
        }
    }
}

/// Each load from global or storage memory in PR #4588's list in `TryGetConstBuffer`
/// (`eden-pr4588/src/shader_recompiler/ir_opt/texture_pass.cpp` @ e5fb3b97), written out, is
/// refused by name, with its result type and its operands.
#[test]
fn every_memory_load_is_refused_by_name() {
    const LOADS: [(&str, &str, &str); 14] = [
        ("LoadGlobalU8", "u32", "u64 0x1000"),
        ("LoadGlobalS8", "u32", "u64 0x1000"),
        ("LoadGlobalU16", "u32", "u64 0x1000"),
        ("LoadGlobalS16", "u32", "u64 0x1000"),
        ("LoadGlobal32", "u32", "u64 0x1000"),
        ("LoadGlobal64", "u32x2", "u64 0x1000"),
        ("LoadGlobal128", "u32x4", "u64 0x1000"),
        ("LoadStorageU8", "u32", "u32 0x0, u32 0x10"),
        ("LoadStorageS8", "u32", "u32 0x0, u32 0x10"),
        ("LoadStorageU16", "u32", "u32 0x0, u32 0x10"),
        ("LoadStorageS16", "u32", "u32 0x0, u32 0x10"),
        ("LoadStorage32", "u32", "u32 0x0, u32 0x10"),
        ("LoadStorage64", "u32x2", "u32 0x0, u32 0x10"),
        ("LoadStorage128", "u32x4", "u32 0x0, u32 0x10"),
    ];
    for (name, ty, args) in LOADS {
        let text = sample_of(&format!("    %0: {ty} = {name} {args}\n"), "%0");
        let opcode = Opcode::from_name(name).expect("an opcode of that name");
        let error = scan_error(&text);
        assert_eq!(
            error,
            ResourceError::Untraced {
                at: value(60),
                why: Untraced::GlobalMemory {
                    at: value(0),
                    opcode
                }
            },
            "{name}"
        );
        assert_eq!(
            error.to_string(),
            format!(
                "%60: the texture handle cannot be traced: it is loaded from storage or global \
                 memory by {name} at %0"
            )
        );
    }
}

/// The scan, the answers and the function of a sample whose handle is the word at 0x10 of
/// bank 3, with a second word at 0x20 in `%1`.
fn scanned_sample() -> (Function, super::TextureScan, crate::exchange::Answers) {
    let text = sample_of(
        "    %0: u32 = GetCbufU32 u32 3, u32 0x10\n    %1: u32 = GetCbufU32 u32 3, u32 0x20\n",
        "%0",
    );
    let engine = Engine::new()
        .word(3, 0x10, 0x9)
        .kind(0x9, FLOAT)
        .word(3, 0x20, 0xa)
        .kind(0xa, FLOAT);
    let function = parse(&text);
    let (found, answers) = answered(&function, buffer(BOUND_BANK), &engine);
    (function, found, answers)
}

/// A texture operation whose handle argument changed after the scan, opcode unchanged, is
/// stale, and nothing changes.
#[test]
fn a_changed_handle_is_stale() {
    let (function, found, answers) = scanned_sample();
    let mut changed = function;
    if let Some(Inst::Op(op)) = changed.inst_mut(value(60))
        && let Some(first) = op.args.first_mut()
    {
        *first = Operand::Value(value(1));
    }
    let before = changed.clone();
    let error = run(&mut changed, &found, &answers).expect_err("refused");
    assert_eq!(error, ResourceError::Stale(value(60)));
    assert_eq!(changed, before);
}

/// A texture operation no longer listed by the block the scan found it in is stale, whether
/// no block lists it or another one does, and nothing changes.
#[test]
fn a_moved_operation_is_stale() {
    let text = "block0:
    %0: u32 = GetCbufU32 u32 3, u32 0x10
    %50: f32x2 = CompositeConstructF32x2 f32 0x0, f32 0x0
    %60: f32x4 = ImageSampleImplicitLod %0, %50, u32 0x0, u32 0x0 {type=2d}
    jump block1
block1:
    return
";
    let engine = Engine::new().word(3, 0x10, 0x9).kind(0x9, FLOAT);
    let function = parse(text);
    let (found, answers) = answered(&function, buffer(BOUND_BANK), &engine);
    let entry = function.entry();
    let unlist = |function: &mut Function| {
        if let Some(block) = function.block_mut(entry) {
            block.insts.retain(|&v| v != value(60));
        }
    };
    let mut unlisted = function.clone();
    unlist(&mut unlisted);
    let mut moved = unlisted.clone();
    let other = moved.blocks().nth(1).map(|(id, _)| id).expect("two blocks");
    if let Some(block) = moved.block_mut(other) {
        block.insts.insert(0, value(60));
    }
    for mut changed in [unlisted, moved] {
        let before = changed.clone();
        let error = run(&mut changed, &found, &answers).expect_err("refused");
        assert_eq!(error, ResourceError::Stale(value(60)));
        assert_eq!(changed, before);
    }
}

/// The documented limit of the stale check (module docs, "When `run` refuses"): a change below
/// the handle argument is not seen, and the descriptor is built from the word the scan traced.
#[test]
fn a_change_below_the_handle_is_not_seen() {
    let (function, found, answers) = scanned_sample();
    let mut changed = function;
    if let Some(slot) = changed.inst_mut(value(0)) {
        *slot = Inst::Op(Operation::new(
            Opcode::GetCbufU32,
            vec![Operand::Imm(Imm::U32(3)), Operand::Imm(Imm::U32(0x20))],
        ));
    }
    let (table, _) = run(&mut changed, &found, &answers).expect("runs");
    assert_eq!(
        table.to_string(),
        "texture 0: 2d, float, from cbuf 3 at 0x10\n"
    );
}

/// A scan is for one run: a second run with it is stale, as the first run set the handle
/// argument to `ABSENT`, and the second changes nothing.
#[test]
fn a_second_run_with_one_scan_is_stale() {
    let (mut function, found, answers) = scanned_sample();
    run(&mut function, &found, &answers).expect("runs");
    let once = function.clone();
    let error = run(&mut function, &found, &answers).expect_err("refused");
    assert_eq!(error, ResourceError::Stale(value(60)));
    assert_eq!(function, once);
}
