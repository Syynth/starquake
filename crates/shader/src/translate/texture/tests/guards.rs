//! Guards: a conditional block for a guard other than `PT`, nothing for `@!PT`, the
//! guard ignored once tested, and refusals under a guard that leave the function as it was.

use super::*;

const X: u32 = f(0.375);
const Y: u32 = f(-1.5);

fn sample_call() -> ScriptedImage {
    call(
        Opcode::ImageSampleImplicitLod,
        info(TextureType::Color2D),
        vec![Val::U32(0x40), Val::F32x2([X, Y]), NONE, NONE],
        Some(Val::F32x4(TEXEL)),
    )
}

/// `instr` translated with `guard_tested`, verified.
fn build(instr: Instr, guard_tested: bool) -> Function {
    let mut function = Function::new();
    let entry = function.entry();
    let mut builder = Builder::new(&mut function, entry);
    let mut buffer = DescriptorBuffer::new();
    translate_instr(&mut builder, instr, guard_tested, &mut buffer).expect("translates");
    builder.terminate(Terminator::Return).expect("terminates");
    assert_eq!(function.verify(), Ok(()));
    function
}

/// `@P1 TEX` tests P1, samples in a block of its own and continues in a third; it samples and
/// writes only where P1 holds. `@!P1` runs where P1 is false.
#[test]
fn a_guard_makes_a_conditional_block() {
    for negated in [false, true] {
        let guard = PredSrc { reg: p(1), negated };
        let instr = encodable(Instr {
            guard,
            op: Op::Tex(tex()),
        });
        let function = build(instr, false);
        let text = function.to_string();
        assert!(
            text.starts_with("block0:\n    %0: u1 = GetPred p1\n"),
            "{text}"
        );
        assert!(text.contains("block1:"), "{text}");
        assert!(
            text.ends_with("    jump block2\nblock2:\n    return\n"),
            "{text}"
        );
        for pred in [false, true] {
            let runs = pred != negated;
            let mut machine = machine_with(&[(10, X), (20, Y), (0, 9)]);
            machine.preds[1] = pred;
            let script = if runs { vec![sample_call()] } else { vec![] };
            run_script(&function, &mut machine, script);
            let expected = if runs { TEXEL[0] } else { 9 };
            assert_eq!(machine.regs[0], expected, "negated {negated}, p1 {pred}");
        }
    }
}

/// `@!PT` never runs and appends nothing, unchecked, whether or not the guard is tested: the
/// translation driver's rule (`crate::translate::guard_step`), as the PR's `Visitor::Run`
/// returns at `guard.Never()`. A form the group refuses is not refused there, and the
/// descriptor buffer is left as it was; an instruction of another group is still refused
/// (the group comes first).
#[test]
fn never_guarded_instructions_append_nothing() {
    let never = PredSrc {
        reg: Pred::PT,
        negated: true,
    };
    let valid = encodable(Instr {
        guard: never,
        op: Op::Tex(tex()),
    });
    let refused = encodable(Instr {
        guard: never,
        op: Op::Tex(OpTex { ndv: true, ..tex() }),
    });
    for guard_tested in [false, true] {
        for instr in [valid, refused] {
            let mut function = Function::new();
            let entry = function.entry();
            let mut builder = Builder::new(&mut function, entry);
            let mut buffer = DescriptorBuffer::new();
            assert_eq!(
                translate_instr(&mut builder, instr, guard_tested, &mut buffer),
                Ok(()),
                "{instr} {guard_tested}"
            );
            assert_eq!(builder.block(), entry);
            assert_eq!(
                builder.function(),
                &Function::new(),
                "{instr} {guard_tested}"
            );
            assert_eq!(buffer, DescriptorBuffer::new());
        }
        let other = Instr {
            guard: never,
            op: Op::Nop(crate::decode::OpNop {}),
        };
        let mut function = Function::new();
        let entry = function.entry();
        let mut builder = Builder::new(&mut function, entry);
        let mut buffer = DescriptorBuffer::new();
        assert_eq!(
            translate_instr(&mut builder, other, guard_tested, &mut buffer),
            Err(TextureError::Unsupported(TextureUnsupported::OtherGroup(
                "nop"
            )))
        );
    }
}

/// With the guard tested by flow analysis, the instruction is translated in place: no branch,
/// and it runs whatever the predicate.
#[test]
fn a_tested_guard_is_ignored() {
    let instr = encodable(Instr {
        guard: PredSrc {
            reg: p(1),
            negated: false,
        },
        op: Op::Tex(tex()),
    });
    let function = build(instr, true);
    assert_eq!(function.block_count(), 1);
    let mut machine = machine_with(&[(10, X), (20, Y)]);
    run_script(&function, &mut machine, vec![sample_call()]);
    assert_eq!(machine.regs[..4], TEXEL);
}

/// A refusal under a guard appends nothing: not even the guard's read or the new blocks.
#[test]
fn refusals_under_a_guard_append_nothing() {
    let instr = encodable(Instr {
        guard: PredSrc {
            reg: p(4),
            negated: false,
        },
        op: Op::Txq(OpTxq {
            query: TexQuery::TextureType,
            ..txq()
        }),
    });
    assert_refused_raw(
        instr,
        DescriptorBuffer::new(),
        TextureUnsupported::QueryKind(TexQuery::TextureType),
    );
}
