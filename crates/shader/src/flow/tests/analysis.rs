//! The walk: what it finds on accepted programs, and one exact refusal per check, each with
//! the accepted program one step away where there is one.

use super::super::analysis::{LINK_LOOKBACK, RETURN_COPY_LOOKBACK, RETURN_COPY_REGISTER_LIMIT};
use super::super::classify::WARP_WIDE_OPCODES;
use super::super::{Analysis, BarrierModel, BranchTables, FlowError, MAX_PROGRAM, analyze};
use super::{
    Item, NEVER, Program, SIZE, bsync, exit, exit_if, kill, offset, p, pure, r, refusal,
    side_effect, to_start,
};
use crate::asm::assemble;
use crate::decode::{
    BraCond, BranchOffset, BrxSrc, DecodeError, Field, GUARD, GUARD_NOT, Instr, Instruction,
    MovSrc, OPCODE, Op, OpBSSy, OpBSync, OpBra, OpBrx, OpCall, OpExit, OpKill, OpMov, OpRet,
    PredSrc, UniformPred, UniformPredSrc, decode,
};

fn tables() -> BranchTables {
    BranchTables::new(0)
}

/// The analysis under both models, which must agree.
fn accepted(code: &[Instruction]) -> Analysis {
    let registers = analyze(code, &tables(), BarrierModel::Registers).expect("accepted");
    let stack = analyze(code, &tables(), BarrierModel::Stack).expect("accepted");
    assert_eq!(
        registers.clone().without_visits(),
        stack.without_visits(),
        "the models disagree"
    );
    registers
}

fn word(instr: Instr) -> Instruction {
    assemble(&instr).expect("encodable")
}

/// A word with `opcode` under `guard`: flow analysis's view of an instruction it does not
/// decode.
fn raw(opcode: u16, guard: PredSrc) -> Instruction {
    Instruction::default()
        .with_field(OPCODE, u64::from(opcode))
        .with_field(GUARD, u64::from(guard.reg.index()))
        .with_flag(GUARD_NOT, guard.negated)
}

/// `BRA` by `offset` with `guard` and second predicate `cond`.
fn bra_by(guard: PredSrc, cond: PredSrc, offset: BranchOffset) -> Instruction {
    word(Instr {
        guard,
        op: Op::Bra(OpBra {
            cond: BraCond::Pred(cond),
            uniform: false,
            offset,
        }),
    })
}

/// `word` with bit `bit` set.
fn with_bit(word: Instruction, bit: u32) -> Instruction {
    word.with_flag(Field::bit(bit), true)
}

#[test]
fn a_straight_line_program_is_one_reachable_run() {
    let code = Program::new().op(pure(1)).op(pure(2)).op(exit()).build();
    let analysis = accepted(&code);
    assert!((0..3).all(|index| analysis.is_reachable(index)));
    assert!(analysis.is_leader(0));
    assert!(!analysis.is_leader(1) && !analysis.is_leader(2));
    assert_eq!(analysis.function_entries(), &[0]);
    assert_eq!(analysis.function_of(2), Some(0));
    assert_eq!(analysis.len(), 3);
    assert!(!analysis.is_empty());
}

#[test]
fn code_after_an_exit_is_unreachable() {
    let code = Program::new().op(exit()).op(pure(1)).op(exit()).build();
    let analysis = accepted(&code);
    assert!(analysis.is_reachable(0));
    assert!(!analysis.is_reachable(1) && !analysis.is_reachable(2));
    assert_eq!(analysis.function_of(1), None);
}

#[test]
fn plain_instructions_are_not_decoded() {
    // Opcode 0 is no instruction the decoder knows, and the word has a stray bit: the walk
    // passes over it (a listed divergence: the reference refuses it).
    let unknown = with_bit(raw(0, PredSrc::TRUE), 95);
    assert_eq!(
        decode(unknown),
        Err(DecodeError::UnknownOpcode { opcode: 0 })
    );
    accepted(&[unknown, word(exit())]);
    // A control instruction must decode: a stray bit on an `EXIT` is refused.
    let stray = with_bit(word(exit()), 95);
    let error = decode(stray).expect_err("a stray bit");
    assert_eq!(refusal(&[stray]), FlowError::Decode { index: 0, error });
}

#[test]
fn uniform_bra_is_refused() {
    let upt = UniformPredSrc {
        reg: UniformPred::UPT,
        negated: false,
    };
    for (cond, uniform) in [
        (BraCond::UniformPred(upt), true),
        (BraCond::UniformPred(upt), false),
        (BraCond::Pred(PredSrc::TRUE), true),
    ] {
        let bra = word(Instr::new(Op::Bra(OpBra {
            cond,
            uniform,
            offset: offset(0, 1),
        })));
        assert_eq!(
            refusal(&[bra, word(exit())]),
            FlowError::UniformBranch { index: 0 }
        );
    }
    // The same branch without `.U` is accepted.
    accepted(&[
        bra_by(PredSrc::TRUE, PredSrc::TRUE, offset(0, 1)),
        word(exit()),
    ]);
}

#[test]
fn target_errors_name_their_reason() {
    let exit = word(exit());
    // One word past the next instruction: byte 16 + 4.
    assert_eq!(
        refusal(&[bra_by(PredSrc::TRUE, PredSrc::TRUE, BranchOffset::from_words(1)), exit]),
        FlowError::MisalignedTarget { index: 0, byte: 20 }
    );
    // One instruction before instruction 0.
    assert_eq!(
        refusal(&[bra_by(PredSrc::TRUE, PredSrc::TRUE, to_start(0).with_extra(-1)), exit]),
        FlowError::TargetOutside {
            index: 0,
            target: -1
        }
    );
    // The instruction after the last: outside, while the last itself is accepted.
    assert_eq!(
        refusal(&[bra_by(PredSrc::TRUE, PredSrc::TRUE, offset(0, 2)), exit]),
        FlowError::TargetOutside {
            index: 0,
            target: 2
        }
    );
    accepted(&[bra_by(PredSrc::TRUE, PredSrc::TRUE, offset(0, 1)), exit]);
    // A misaligned target is reported before a uniform branch, as `BranchTarget` runs first.
    let both = word(Instr::new(Op::Bra(OpBra {
        cond: BraCond::Pred(PredSrc::TRUE),
        uniform: true,
        offset: BranchOffset::from_words(1),
    })));
    assert_eq!(
        refusal(&[both, exit]),
        FlowError::MisalignedTarget { index: 0, byte: 20 }
    );
}

/// An offset `instructions` further on.
trait Extra {
    fn with_extra(self, instructions: i64) -> BranchOffset;
}

impl Extra for BranchOffset {
    fn with_extra(self, instructions: i64) -> BranchOffset {
        BranchOffset::from_words(
            self.words() + instructions * i64::from(SIZE) / BranchOffset::WORD_BYTES,
        )
    }
}

#[test]
fn program_size_at_its_edges() {
    assert_eq!(
        analyze(&[], &tables(), BarrierModel::Registers),
        Err(FlowError::ProgramSize { len: 0 })
    );
    let filler = word(pure(1));
    let len = usize::try_from(MAX_PROGRAM).expect("small");
    let mut code = vec![filler; len];
    *code.last_mut().expect("not empty") = word(exit());
    for model in [BarrierModel::Registers, BarrierModel::Stack] {
        assert!(analyze(&code, &tables(), model).is_ok());
    }
    code.push(word(exit()));
    for model in [BarrierModel::Registers, BarrierModel::Stack] {
        assert_eq!(
            analyze(&code, &tables(), model),
            Err(FlowError::ProgramSize { len: len + 1 })
        );
    }
}

#[test]
fn a_path_running_off_the_end_is_refused() {
    assert_eq!(
        refusal(&[word(pure(1)), word(pure(2))]),
        FlowError::RunsPastEnd { last: 1 }
    );
}

#[test]
fn code_reached_from_two_functions_is_refused() {
    // The function at 3 branches back into the program's EXIT at 2.
    let code = [
        mov_imm(10, 2 * SIZE),
        call_word(1, 3, true, PredSrc::TRUE, PredSrc::TRUE),
        word(exit()),
        bra_by(PredSrc::TRUE, PredSrc::TRUE, offset(3, 2)),
    ];
    assert_eq!(refusal(&code), FlowError::SharedCode { index: 2 });
}

#[test]
fn warp_wide_instructions_may_not_be_guarded() {
    for opcode in WARP_WIDE_OPCODES {
        assert_eq!(
            refusal(&[raw(opcode, p(0)), word(exit())]),
            FlowError::PredicatedWarpWide { index: 0 },
            "{opcode:#x}"
        );
        // Unguarded, and statically disabled, are accepted.
        accepted(&[raw(opcode, PredSrc::TRUE), word(exit())]);
        accepted(&[raw(opcode, NEVER), word(exit())]);
    }
}

#[test]
fn a_kill_with_its_own_predicate_is_refused() {
    let own = word(Instr {
        guard: p(0),
        op: Op::Kill(OpKill { pred: p(1) }),
    });
    assert_eq!(
        refusal(&[own, word(exit())]),
        FlowError::KillPredicate { index: 0 }
    );
    let analysis = accepted(&[word(kill(p(0))), word(exit())]);
    assert!(analysis.is_leader(0) && analysis.is_leader(1));
}

#[test]
fn special_exits_are_refused() {
    for (keep_ref_count, no_at_exit) in [(true, false), (false, true), (true, true)] {
        let special = word(Instr::new(Op::Exit(OpExit {
            keep_ref_count,
            no_at_exit,
            pred: PredSrc::TRUE,
        })));
        assert_eq!(refusal(&[special]), FlowError::SpecialExit { index: 0 });
    }
}

#[test]
fn an_exit_with_two_predicates_needs_a_free_next_instruction() {
    let two = word(exit_if(p(0), p(1)));
    // The next instruction needs its own test: refused.
    assert_eq!(
        refusal(&[two, side_effect(p(2)), word(exit())]),
        FlowError::TwoPredicateExit { index: 0 }
    );
    // No next instruction: refused.
    assert_eq!(refusal(&[two]), FlowError::TwoPredicateExit { index: 0 });
    // A pure next instruction: accepted.
    let analysis = accepted(&[two, word(pure(1)), word(exit())]);
    assert!(analysis.is_leader(0) && analysis.is_leader(1));
    // One predicate only, either one: accepted before a guarded side effect.
    accepted(&[word(exit_if(p(0), PredSrc::TRUE)), side_effect(p(2)), word(exit())]);
    accepted(&[word(exit_if(PredSrc::TRUE, p(1))), side_effect(p(2)), word(exit())]);
    // A special exit is reported before two predicates.
    let both = word(Instr {
        guard: p(0),
        op: Op::Exit(OpExit {
            keep_ref_count: true,
            no_at_exit: false,
            pred: p(1),
        }),
    });
    assert_eq!(
        refusal(&[both, side_effect(p(2)), word(exit())]),
        FlowError::SpecialExit { index: 0 }
    );
}

/// A `CALL` at `index` of instruction `target` with `no_inc` and `cond`, under `guard`.
fn call_word(index: u32, target: u32, no_inc: bool, cond: PredSrc, guard: PredSrc) -> Instruction {
    word(Instr {
        guard,
        op: Op::Call(OpCall {
            offset: offset(index, target),
            no_inc,
            cond,
        }),
    })
}

/// `MOV dst, value`.
fn mov_imm(dst: u8, value: u32) -> Instruction {
    word(Instr::new(Op::Mov(OpMov {
        dst: r(dst),
        src: MovSrc::Imm(value),
        quad_lanes: OpMov::ALL_LANES,
    })))
}

/// `RET` at `index` through `reg`.
fn ret_word(index: u32, reg: u8) -> Instruction {
    word(Instr::new(Op::Ret(OpRet {
        src: r(reg),
        offset: to_start(index),
        no_dec: true,
        cond: PredSrc::TRUE,
    })))
}

#[test]
fn calls_need_their_mode() {
    // 0: MOV R10; 1: CALL 3; 2: EXIT; 3: RET R10.
    let program = |call: Instruction| [mov_imm(10, 2 * SIZE), call, word(exit()), ret_word(3, 10)];
    let analysis = accepted(&program(call_word(1, 3, true, PredSrc::TRUE, PredSrc::TRUE)));
    assert_eq!(analysis.function_entries(), &[0, 3]);
    assert_eq!(analysis.callee(1), Some(1));
    assert_eq!(analysis.function_of(3), Some(1));
    assert!(analysis.is_leader(2) && analysis.is_leader(3));
    for call in [
        call_word(1, 3, false, PredSrc::TRUE, PredSrc::TRUE),
        call_word(1, 3, true, p(1), PredSrc::TRUE),
    ] {
        assert_eq!(
            refusal(&program(call)),
            FlowError::CallMode { index: 1 }
        );
    }
    // The mode is checked before the linkage: no `MOV` and a bad mode.
    let code = [word(pure(1)), call_word(1, 3, false, PredSrc::TRUE, PredSrc::TRUE), word(exit()), ret_word(3, 10)];
    assert_eq!(refusal(&code), FlowError::CallMode { index: 1 });
}

#[test]
fn calls_need_a_return_address_and_a_target() {
    let call = call_word(1, 3, true, PredSrc::TRUE, PredSrc::TRUE);
    // No `MOV` of the return address.
    assert_eq!(
        refusal(&[word(pure(1)), call, word(exit()), ret_word(3, 10)]),
        FlowError::CallLinkage { index: 1 }
    );
    // A `MOV` of another address.
    assert_eq!(
        refusal(&[mov_imm(10, 3 * SIZE), call, word(exit()), ret_word(3, 10)]),
        FlowError::CallLinkage { index: 1 }
    );
    // A target past the end.
    assert_eq!(
        refusal(&[mov_imm(10, 2 * SIZE), call_word(1, 4, true, PredSrc::TRUE, PredSrc::TRUE), word(exit()), ret_word(3, 10)]),
        FlowError::CallLinkage { index: 1 }
    );
}

#[test]
fn the_return_address_is_looked_for_at_most_six_back() {
    let fill = usize::try_from(LINK_LOOKBACK).expect("small") - 1;
    // MOV, five fillers, CALL: the MOV is six back.
    let build = |fill: usize| {
        let call = u32::try_from(fill + 1).expect("small");
        let mut code = vec![mov_imm(10, (call + 1) * SIZE)];
        code.extend(std::iter::repeat_n(word(pure(1)), fill));
        code.push(call_word(call, call + 2, true, PredSrc::TRUE, PredSrc::TRUE));
        code.push(word(exit()));
        code.push(ret_word(call + 2, 10));
        (code, call)
    };
    let (code, _) = build(fill);
    accepted(&code);
    let (code, call) = build(fill + 1);
    assert_eq!(refusal(&code), FlowError::CallLinkage { index: call });
    // A control instruction in between stops the search.
    let code = [
        mov_imm(10, 3 * SIZE),
        word(super::brk(0, PredSrc::TRUE)),
        call_word(2, 4, true, PredSrc::TRUE, PredSrc::TRUE),
        word(exit()),
        ret_word(4, 10),
    ];
    // `BREAK` is a no-effect instruction to the walk: the search goes on past it.
    accepted(&code);
    let code = [
        mov_imm(10, 4 * SIZE),
        word(exit_if(p(0), PredSrc::TRUE)),
        word(pure(1)),
        call_word(3, 5, true, PredSrc::TRUE, PredSrc::TRUE),
        word(exit()),
        ret_word(5, 10),
    ];
    assert_eq!(refusal(&code), FlowError::CallLinkage { index: 3 });
}

#[test]
fn link_mov_must_decode() {
    let code = |link: Instruction| {
        [link, call_word(1, 3, true, PredSrc::TRUE, PredSrc::TRUE), word(exit()), ret_word(3, 10)]
    };
    let stray = with_bit(mov_imm(10, 2 * SIZE), 95);
    assert!(decode(stray).is_err());
    assert_eq!(refusal(&code(stray)), FlowError::CallLinkage { index: 1 });
    // A guarded MOV, or one of some lanes only, is not a return address either.
    let guarded = word(Instr {
        guard: p(0),
        op: Op::Mov(OpMov {
            dst: r(10),
            src: MovSrc::Imm(2 * SIZE),
            quad_lanes: OpMov::ALL_LANES,
        }),
    });
    let lanes = word(Instr::new(Op::Mov(OpMov {
        dst: r(10),
        src: MovSrc::Imm(2 * SIZE),
        quad_lanes: 0x7,
    })));
    for link in [guarded, lanes] {
        assert_eq!(refusal(&code(link)), FlowError::CallLinkage { index: 1 });
    }
}

#[test]
fn two_calls_of_a_function_use_one_link_register() {
    let program = |second: u8| {
        Program::new()
            .call(10, "f")
            .call(second, "f")
            .op(exit())
            .label("f")
            .ret(10)
            .build()
    };
    let analysis = accepted(&program(10));
    assert_eq!(analysis.callee(1), Some(1));
    assert_eq!(analysis.callee(3), Some(1));
    assert_eq!(
        refusal(&program(11)),
        FlowError::InconsistentLinkage { index: 3 }
    );
}

#[test]
fn a_function_calling_itself_is_refused() {
    let code = Program::new()
        .call(10, "f")
        .op(exit())
        .label("f")
        .call(10, "f")
        .ret(10)
        .build();
    assert_eq!(refusal(&code), FlowError::Recursion { index: 4 });
}

#[test]
fn returns_need_their_linkage() {
    // A `RET` in the program itself.
    assert_eq!(
        refusal(&[ret_word(0, 10)]),
        FlowError::ReturnLinkage { index: 0 }
    );
    let program = |ret: Instruction| [mov_imm(10, 2 * SIZE), call_word(1, 3, true, PredSrc::TRUE, PredSrc::TRUE), word(exit()), ret];
    accepted(&program(ret_word(3, 10)));
    let bad = |src, offset, no_dec, cond| {
        word(Instr::new(Op::Ret(OpRet {
            src,
            offset,
            no_dec,
            cond,
        })))
    };
    for ret in [
        bad(r(10), to_start(3), false, PredSrc::TRUE),
        bad(r(10), to_start(3), true, p(1)),
        bad(r(10), to_start(2), true, PredSrc::TRUE),
    ] {
        assert_eq!(
            refusal(&program(ret)),
            FlowError::ReturnLinkage { index: 3 }
        );
    }
}

#[test]
fn a_return_through_another_register_needs_a_copy_of_the_link() {
    // f: [fillers] MOV R11, R10; RET R11.
    let program = |copy: Option<Instruction>, gap: usize| {
        let mut code = vec![mov_imm(10, 2 * SIZE), call_word(1, 3, true, PredSrc::TRUE, PredSrc::TRUE), word(exit())];
        if let Some(copy) = copy {
            code.push(copy);
        }
        code.extend(std::iter::repeat_n(word(pure(1)), gap));
        let ret = u32::try_from(code.len()).expect("small");
        code.push(ret_word(ret, 11));
        (code, ret)
    };
    let copy = |src: u8, dst: u8| {
        word(Instr::new(Op::Mov(OpMov {
            dst: r(dst),
            src: MovSrc::Reg(r(src)),
            quad_lanes: OpMov::ALL_LANES,
        })))
    };
    let lookback = usize::try_from(RETURN_COPY_LOOKBACK).expect("small");
    // The copy right before, and exactly eight back: accepted.
    accepted(&program(Some(copy(10, 11)), 0).0);
    accepted(&program(Some(copy(10, 11)), lookback - 1).0);
    // Nine back: refused.
    let (code, ret) = program(Some(copy(10, 11)), lookback);
    assert_eq!(refusal(&code), FlowError::ReturnRegister { index: ret });
    // No copy, or a copy of another register: refused.
    let (code, ret) = program(None, 0);
    assert_eq!(refusal(&code), FlowError::ReturnRegister { index: ret });
    let (code, ret) = program(Some(copy(12, 11)), 0);
    assert_eq!(refusal(&code), FlowError::ReturnRegister { index: ret });
    // A later write of the register that is not the copy: refused.
    let mut code = vec![mov_imm(10, 2 * SIZE), call_word(1, 3, true, PredSrc::TRUE, PredSrc::TRUE), word(exit())];
    code.push(copy(10, 11));
    code.push(mov_imm(11, 0));
    code.push(ret_word(5, 11));
    assert_eq!(refusal(&code), FlowError::ReturnRegister { index: 5 });
}

#[test]
fn a_branch_into_the_copy_breaks_it() {
    // f at 3: @P0 BRA 5; MOV R11, R10; (5) RET R11: a branch lands between the copy and the
    // `RET`.
    let copy = word(Instr::new(Op::Mov(OpMov {
        dst: r(11),
        src: MovSrc::Reg(r(10)),
        quad_lanes: OpMov::ALL_LANES,
    })));
    let landing = [
        mov_imm(10, 2 * SIZE),
        call_word(1, 3, true, PredSrc::TRUE, PredSrc::TRUE),
        word(exit()),
        bra_by(p(0), PredSrc::TRUE, offset(3, 5)),
        copy,
        ret_word(5, 11),
    ];
    assert_eq!(refusal(&landing), FlowError::ReturnRegister { index: 5 });
    // The branch landing on the copy itself is outside the checked run: accepted.
    let before = [
        mov_imm(10, 2 * SIZE),
        call_word(1, 3, true, PredSrc::TRUE, PredSrc::TRUE),
        word(exit()),
        bra_by(p(0), PredSrc::TRUE, offset(3, 4)),
        copy,
        ret_word(5, 11),
    ];
    accepted(&before);
}

#[test]
fn return_registers_from_250_are_not_copies() {
    let limit = RETURN_COPY_REGISTER_LIMIT;
    let program = |link: u8, reg: u8| {
        let copy = word(Instr::new(Op::Mov(OpMov {
            dst: r(reg),
            src: MovSrc::Reg(r(link)),
            quad_lanes: OpMov::ALL_LANES,
        })));
        [mov_imm(link, 2 * SIZE), call_word(1, 3, true, PredSrc::TRUE, PredSrc::TRUE), word(exit()), copy, ret_word(4, reg)]
    };
    accepted(&program(limit - 1, limit - 2));
    accepted(&program(limit - 2, limit - 1));
    assert_eq!(
        refusal(&program(limit, limit - 1)),
        FlowError::ReturnRegister { index: 4 }
    );
    assert_eq!(
        refusal(&program(limit - 1, limit)),
        FlowError::ReturnRegister { index: 4 }
    );
}

#[test]
fn a_function_may_not_overwrite_its_link_register() {
    let code = Program::new()
        .call(10, "f")
        .op(exit())
        .label("f")
        .op(pure(10))
        .ret(10)
        .build();
    assert_eq!(refusal(&code), FlowError::LinkOverwritten { index: 3 });
    // The same write in the program is no overwrite.
    let code = Program::new()
        .op(pure(10))
        .call(10, "f")
        .op(exit())
        .label("f")
        .op(pure(11))
        .ret(10)
        .build();
    accepted(&code);
}

#[test]
fn function_checks_run_per_function_returns_first() {
    // f (function 1) overwrites its link and returns through another register; g (function 2)
    // returns through another register. f's return is reported first, then f's overwrite.
    let code = Program::new()
        .call(10, "f")
        .call(12, "g")
        .op(exit())
        .label("f")
        .op(pure(10))
        .ret(11)
        .label("g")
        .ret(13)
        .build();
    assert_eq!(refusal(&code), FlowError::ReturnRegister { index: 6 });
    let code = Program::new()
        .call(10, "f")
        .call(12, "g")
        .op(exit())
        .label("f")
        .op(pure(10))
        .ret(10)
        .label("g")
        .ret(13)
        .build();
    assert_eq!(refusal(&code), FlowError::LinkOverwritten { index: 5 });
}

#[test]
fn brx_needs_its_mode() {
    let brx = |guard: PredSrc, offset: BranchOffset, cond: PredSrc| {
        word(Instr {
            guard,
            op: Op::Brx(OpBrx {
                src: BrxSrc::Reg(r(2)),
                offset,
                cond,
            }),
        })
    };
    let mut tables = BranchTables::new(0);
    tables.insert(0, vec![1]);
    let ok = [brx(PredSrc::TRUE, to_start(0), PredSrc::TRUE), word(exit())];
    for model in [BarrierModel::Registers, BarrierModel::Stack] {
        let analysis = analyze(&ok, &tables, model).expect("accepted");
        assert_eq!(analysis.indirect_targets(0), Some(&[1][..]));
    }
    for bad in [
        brx(p(0), to_start(0), PredSrc::TRUE),
        brx(PredSrc::TRUE, to_start(0), p(1)),
        brx(PredSrc::TRUE, to_start(1), PredSrc::TRUE),
    ] {
        for model in [BarrierModel::Registers, BarrierModel::Stack] {
            assert_eq!(
                analyze(&[bad, word(exit())], &tables, model),
                Err(FlowError::BrxMode { index: 0 })
            );
        }
    }
}

#[test]
fn brx_without_targets_is_refused() {
    let brx = word(Instr::new(Op::Brx(OpBrx {
        src: BrxSrc::Reg(r(2)),
        offset: to_start(1),
        cond: PredSrc::TRUE,
    })));
    let code = [word(pure(2)), brx, word(exit()), word(exit())];
    // No table, and a table naming only instructions outside the program.
    for targets in [None, Some(vec![4, 5])] {
        let mut tables = BranchTables::new(0);
        if let Some(targets) = targets {
            tables.insert(1, targets);
        }
        for model in [BarrierModel::Registers, BarrierModel::Stack] {
            assert_eq!(
                analyze(&code, &tables, model),
                Err(FlowError::NoTarget { index: 1 })
            );
        }
    }
    // Entries outside the program are ignored when one is inside: the last instruction.
    let mut tables = BranchTables::new(0);
    tables.insert(1, vec![4, 3]);
    let analysis = analyze(&code, &tables, BarrierModel::Registers).expect("accepted");
    assert_eq!(analysis.indirect_targets(1), Some(&[3][..]));
    assert!(!analysis.is_reachable(2) && analysis.is_reachable(3));
}

#[test]
fn bssy_and_bsync_may_not_be_predicated() {
    let bssy = |guard: PredSrc, cond: PredSrc| {
        word(Instr {
            guard,
            op: Op::BSSy(OpBSSy {
                bar: super::b(0),
                offset: offset(0, 2),
                cond,
            }),
        })
    };
    let bsync = |guard: PredSrc, cond: PredSrc| {
        word(Instr {
            guard,
            op: Op::BSync(OpBSync {
                bar: super::b(0),
                cond,
            }),
        })
    };
    let ok = bssy(PredSrc::TRUE, PredSrc::TRUE);
    let sync = bsync(PredSrc::TRUE, PredSrc::TRUE);
    let analysis = accepted(&[ok, sync, word(exit())]);
    assert_eq!(analysis.sync_target(1), Some(2));
    for bad in [bssy(p(0), PredSrc::TRUE), bssy(PredSrc::TRUE, p(0))] {
        assert_eq!(
            refusal(&[bad, sync, word(exit())]),
            FlowError::PredicatedBssy { index: 0 }
        );
    }
    for bad in [bsync(p(0), PredSrc::TRUE), bsync(PredSrc::TRUE, p(0))] {
        assert_eq!(
            refusal(&[ok, bad, word(exit())]),
            FlowError::PredicatedBsync { index: 1 }
        );
    }
    // A misaligned target is reported before the predicate, as `BranchTarget` runs first.
    let both = word(Instr {
        guard: p(0),
        op: Op::BSSy(OpBSSy {
            bar: super::b(0),
            offset: BranchOffset::from_words(1),
            cond: PredSrc::TRUE,
        }),
    });
    assert_eq!(
        refusal(&[both, sync, word(exit())]),
        FlowError::MisalignedTarget { index: 0, byte: 20 }
    );
    // A target past the end.
    let outside = word(Instr::new(Op::BSSy(OpBSSy {
        bar: super::b(0),
        offset: offset(0, 3),
        cond: PredSrc::TRUE,
    })));
    assert_eq!(
        refusal(&[outside, sync, word(exit())]),
        FlowError::TargetOutside {
            index: 0,
            target: 3
        }
    );
}

#[test]
fn a_bsync_with_no_barrier_is_refused() {
    assert_eq!(
        refusal(&[word(bsync(3)), word(exit())]),
        FlowError::MissingBarrier { index: 0 }
    );
    // A disabled one is no refusal.
    let disabled = word(Instr {
        guard: NEVER,
        op: Op::BSync(OpBSync {
            bar: super::b(3),
            cond: PredSrc::TRUE,
        }),
    });
    accepted(&[disabled, word(exit())]);
}

#[test]
fn the_stack_model_refuses_a_barrier_set_twice() {
    // BSSY B0; BSSY B0; BSYNC B0; EXIT.
    let code = Program::new()
        .bssy(0, "join")
        .bssy(0, "join")
        .op(bsync(0))
        .label("join")
        .op(exit())
        .build();
    assert_eq!(
        analyze(&code, &tables(), BarrierModel::Stack),
        Err(FlowError::NestedBarrier { index: 1 })
    );
    let registers = analyze(&code, &tables(), BarrierModel::Registers).expect("accepted");
    assert_eq!(registers.sync_target(2), Some(3));
}

#[test]
fn the_stack_model_holds_sixteen_barriers() {
    let build = |count: u8| {
        let mut program = Program::new();
        for bar in 0..count {
            program.item(Item::Bssy {
                bar: bar % 16,
                label: "join",
            });
        }
        program.label("join").op(exit());
        program.build()
    };
    assert!(analyze(&build(16), &tables(), BarrierModel::Stack).is_ok());
    assert_eq!(
        analyze(&build(17), &tables(), BarrierModel::Stack),
        Err(FlowError::NestedBarrier { index: 16 })
    );
}

#[test]
fn paths_that_set_a_barrier_differently_are_refused() {
    // 0: @P0 BRA 3; 1: BSSY B0 -> 6; 2: BRA 4; 3: BSSY B0 -> 7; 4: BSYNC B0; 5: EXIT; 6: EXIT;
    // 7: EXIT.
    let code = Program::new()
        .bra(p(0), "other")
        .bssy(0, "six")
        .bra(PredSrc::TRUE, "sync")
        .label("other")
        .bssy(0, "seven")
        .label("sync")
        .op(bsync(0))
        .op(exit())
        .label("six")
        .op(exit())
        .label("seven")
        .op(exit())
        .build();
    assert_eq!(
        analyze(&code, &tables(), BarrierModel::Registers),
        Err(FlowError::TargetsDiffer { index: 4 })
    );
    assert_eq!(
        analyze(&code, &tables(), BarrierModel::Stack),
        Err(FlowError::StacksDiffer { index: 4 })
    );
}

#[test]
fn guarded_side_effects_split_and_pure_ones_do_not() {
    let analysis = accepted(&[word(pure(1)), side_effect(p(0)), word(pure(2)), word(exit())]);
    assert!(analysis.is_leader(1) && analysis.is_leader(2));
    let guarded_pure = word(Instr {
        guard: p(0),
        op: pure(1).op,
    });
    let analysis = accepted(&[word(pure(1)), guarded_pure, word(pure(2)), word(exit())]);
    assert!(!analysis.is_leader(1) && !analysis.is_leader(2));
    // A disabled side effect does not split either.
    let analysis = accepted(&[word(pure(1)), side_effect(NEVER), word(pure(2)), word(exit())]);
    assert!(!analysis.is_leader(1) && !analysis.is_leader(2));
}

#[test]
fn a_skip_call_is_a_branch() {
    // 0: @P0 CALL 2 (no return address); 1: BRA 3; 2: pure; 3: EXIT.
    let code = [
        call_word(0, 2, true, PredSrc::TRUE, p(0)),
        bra_by(PredSrc::TRUE, PredSrc::TRUE, offset(1, 3)),
        word(pure(1)),
        word(exit()),
    ];
    let analysis = accepted(&code);
    assert_eq!(analysis.function_entries(), &[0]);
    assert_eq!(analysis.callee(0), None);
    assert!((0..4).all(|index| analysis.is_reachable(index)));
    // With a conditional branch after it, it is a call, which needs its return address.
    let code = [
        call_word(0, 2, true, PredSrc::TRUE, p(0)),
        bra_by(p(1), PredSrc::TRUE, offset(1, 3)),
        word(pure(1)),
        word(exit()),
    ];
    assert_eq!(refusal(&code), FlowError::CallLinkage { index: 0 });
}

#[test]
fn nested_barriers_reconverge_at_their_own_points() {
    // BSSY B0 -> outer; BSSY B1 -> inner; @P0 BRA skip; pure; skip: BSYNC B1; inner: pure;
    // BSYNC B0; outer: EXIT.
    let program = Program::new()
        .bssy(0, "outer")
        .bssy(1, "inner")
        .bra(p(0), "skip")
        .op(pure(1))
        .label("skip")
        .op(bsync(1))
        .label("inner")
        .op(pure(2))
        .op(bsync(0))
        .label("outer")
        .op(exit())
        .clone();
    let analysis = accepted(&program.build());
    assert_eq!(analysis.sync_target(program.at("skip")), Some(program.at("inner")));
    assert_eq!(analysis.sync_target(6), Some(program.at("outer")));
    assert!(analysis.is_leader(program.at("inner")) && analysis.is_leader(program.at("outer")));
}

#[test]
fn an_early_exit_from_a_region_abandons_the_inner_barrier() {
    // BSSY B0 -> outer; BSSY B1 -> inner; @P0 BRA leave (to BSYNC B0, leaving B1 set); pure;
    // BSYNC B1; inner: pure; leave: BSYNC B0; outer: EXIT.
    let program = Program::new()
        .bssy(0, "outer")
        .bssy(1, "inner")
        .bra(p(0), "leave")
        .op(pure(1))
        .op(bsync(1))
        .label("inner")
        .op(pure(2))
        .label("leave")
        .op(bsync(0))
        .label("outer")
        .op(exit())
        .clone();
    let analysis = accepted(&program.build());
    assert_eq!(analysis.sync_target(program.at("leave")), Some(program.at("outer")));
    assert_eq!(analysis.sync_target(4), Some(program.at("inner")));
}

#[test]
fn a_loop_with_a_break_reaches_its_exit() {
    // BSSY B0 -> after; head: pure; @P1 BREAK B0; @P1 BRA out; @P0 BRA head; out: BSYNC B0;
    // after: EXIT.
    let program = Program::new()
        .bssy(0, "after")
        .label("head")
        .op(pure(1))
        .op(super::brk(0, p(1)))
        .bra(p(1), "out")
        .bra(p(0), "head")
        .label("out")
        .op(bsync(0))
        .label("after")
        .op(exit())
        .clone();
    let analysis = accepted(&program.build());
    assert!(analysis.is_leader(program.at("head")));
    assert_eq!(analysis.sync_target(program.at("out")), Some(program.at("after")));
}
