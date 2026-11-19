//! The two barrier models agree on generated structured programs (nested barriers, branches,
//! early exits, loops with breaks), and nothing panics on arbitrary programs.

use std::collections::BTreeMap;

use proptest::prelude::*;

use super::super::{BarrierModel, BranchTables, analyze, flow_graph};
use super::{bsync, exit, exit_if, offset, p, pure, side_effect, to_start};
use crate::asm::assemble;
use crate::decode::{
    BraCond, BrxSrc, Instr, Instruction, Op, OpBSSy, OpBra, OpBreak, OpBrx,
    OpCall, OpRet, PredSrc,
};
use crate::testing::{any_form_word, config};

/// A statement of a generated program.
#[derive(Clone, Debug)]
enum Stmt {
    /// A pure instruction.
    Pure,
    /// A side effect under a guard.
    Side,
    /// An exit under a guard.
    ExitIf,
    /// `if P0 { then } else { otherwise }`, reconverging at a barrier.
    If(Vec<Stmt>, Vec<Stmt>),
    /// `loop { body; if P1 break } while !P2`, reconverging at a barrier.
    Loop(Vec<Stmt>),
}

fn any_stmt() -> impl Strategy<Value = Stmt> {
    let leaf = prop_oneof![Just(Stmt::Pure), Just(Stmt::Side), Just(Stmt::ExitIf)];
    leaf.prop_recursive(4, 48, 4, |inner| {
        let body = || proptest::collection::vec(inner.clone(), 0..4);
        prop_oneof![
            (body(), body()).prop_map(|(then, otherwise)| Stmt::If(then, otherwise)),
            body().prop_map(Stmt::Loop),
        ]
    })
}

/// An instruction waiting for a label's index.
enum Pending {
    Done(Instr),
    Word(Instruction),
    Bra(PredSrc, usize),
    Bssy(u8, usize),
}

/// Emits statements, with labels as numbers.
#[derive(Default)]
struct Emitter {
    code: Vec<Pending>,
    labels: BTreeMap<usize, u32>,
    next_label: usize,
}

impl Emitter {
    fn label(&mut self) -> usize {
        self.next_label += 1;
        self.next_label
    }

    fn place(&mut self, label: usize) {
        let here = u32::try_from(self.code.len()).expect("small");
        self.labels.insert(label, here);
    }

    fn emit(&mut self, stmts: &[Stmt], depth: u8) {
        for stmt in stmts {
            match stmt {
                Stmt::Pure => self.code.push(Pending::Done(pure(1))),
                Stmt::Side => self.code.push(Pending::Word(side_effect(p(3)))),
                Stmt::ExitIf => self.code.push(Pending::Done(exit_if(p(4), PredSrc::TRUE))),
                Stmt::If(then, otherwise) => {
                    let (other, join) = (self.label(), self.label());
                    self.code.push(Pending::Bssy(depth, join));
                    self.code.push(Pending::Bra(p(0), other));
                    self.emit(then, depth + 1);
                    self.code.push(Pending::Done(bsync(depth)));
                    self.place(other);
                    self.emit(otherwise, depth + 1);
                    self.code.push(Pending::Done(bsync(depth)));
                    self.place(join);
                }
                Stmt::Loop(body) => {
                    let (head, out, after) = (self.label(), self.label(), self.label());
                    self.code.push(Pending::Bssy(depth, after));
                    self.place(head);
                    self.emit(body, depth + 1);
                    self.code.push(Pending::Done(Instr {
                        guard: p(1),
                        op: Op::Break(OpBreak {
                            bar: super::b(depth),
                            cond: PredSrc::TRUE,
                        }),
                    }));
                    self.code.push(Pending::Bra(p(1), out));
                    self.code.push(Pending::Bra(super::not_p(2), head));
                    self.place(out);
                    self.code.push(Pending::Done(bsync(depth)));
                    self.place(after);
                }
            }
        }
    }

    fn build(mut self, stmts: &[Stmt]) -> Vec<Instruction> {
        self.emit(stmts, 0);
        self.code.push(Pending::Done(exit()));
        let at = |label: &usize| self.labels[label];
        (0_u32..)
            .zip(&self.code)
            .map(|(index, pending)| {
                let instr = match pending {
                    Pending::Done(instr) => *instr,
                    Pending::Word(word) => return *word,
                    Pending::Bra(guard, label) => Instr {
                        guard: *guard,
                        op: Op::Bra(OpBra {
                            cond: BraCond::Pred(PredSrc::TRUE),
                            uniform: false,
                            offset: offset(index, at(label)),
                        }),
                    },
                    Pending::Bssy(bar, label) => Instr::new(Op::BSSy(OpBSSy {
                        bar: super::b(*bar),
                        offset: offset(index, at(label)),
                        cond: PredSrc::TRUE,
                    })),
                };
                assemble(&instr).expect("encodable")
            })
            .collect()
    }
}

proptest! {
    #![proptest_config(config())]

    #[test]
    fn the_models_agree_on_structured_programs(
        stmts in proptest::collection::vec(any_stmt(), 0..6)
    ) {
        let code = Emitter::default().build(&stmts);
        let tables = BranchTables::new(0);
        let registers = analyze(&code, &tables, BarrierModel::Registers);
        let stack = analyze(&code, &tables, BarrierModel::Stack);
        let (Ok(registers), Ok(stack)) = (registers, stack) else {
            return Err(TestCaseError::fail("a structured program was refused"));
        };
        prop_assert_eq!(registers.without_visits(), stack.without_visits());
        let graph = flow_graph(&code, &tables).expect("accepted");
        let ir = graph.build_ir().expect("IR");
        prop_assert_eq!(ir.function.verify(), Ok(()));
    }

    #[test]
    fn nothing_panics_on_arbitrary_programs(
        items in proptest::collection::vec(any_word(), 1..40),
        targets in proptest::collection::vec((0_u32..40, proptest::collection::vec(0_u32..48, 0..4)), 0..4),
        base in prop_oneof![Just(0_u32), Just(0x80)],
    ) {
        let code: Vec<Instruction> = (0_u32..)
            .zip(&items)
            .map(|(index, item)| item.word(index))
            .collect();
        let mut tables = BranchTables::new(base);
        for (branch, list) in targets {
            tables.insert(branch, list);
        }
        let _ = analyze(&code, &tables, BarrierModel::Stack);
        if let Ok(graph) = flow_graph(&code, &tables) {
            let ir = graph.build_ir().expect("IR");
            prop_assert_eq!(ir.function.verify(), Ok(()));
        }
        let _ = super::super::find_jump_tables(&code);
    }
}

/// A word of an arbitrary program: any decodable form, or a control instruction whose target
/// lies near it, so that paths go somewhere.
#[derive(Clone, Debug)]
enum Word {
    Any(Instruction),
    Bra(PredSrc, PredSrc, i8),
    Bssy(u8, i8),
    Call(PredSrc, i8),
    Ret(PredSrc, u8),
    Brx(u8),
    Fixed(Instr),
}

impl Word {
    fn word(&self, index: u32) -> Instruction {
        let near = |delta: i8| {
            let target = i64::from(index) + 1 + i64::from(delta);
            offset(index, u32::try_from(target.max(0)).unwrap_or(0))
        };
        let instr = match *self {
            Self::Any(word) => return word,
            Self::Fixed(instr) => instr,
            Self::Bra(guard, cond, delta) => Instr {
                guard,
                op: Op::Bra(OpBra {
                    cond: BraCond::Pred(cond),
                    uniform: false,
                    offset: near(delta),
                }),
            },
            Self::Bssy(bar, delta) => Instr::new(Op::BSSy(OpBSSy {
                bar: super::b(bar),
                offset: near(delta),
                cond: PredSrc::TRUE,
            })),
            Self::Call(guard, delta) => Instr {
                guard,
                op: Op::Call(OpCall {
                    offset: near(delta),
                    no_inc: true,
                    cond: PredSrc::TRUE,
                }),
            },
            Self::Ret(guard, reg) => Instr {
                guard,
                op: Op::Ret(OpRet {
                    src: super::r(reg),
                    offset: to_start(index),
                    no_dec: true,
                    cond: PredSrc::TRUE,
                }),
            },
            Self::Brx(reg) => Instr::new(Op::Brx(OpBrx {
                src: BrxSrc::Reg(super::r(reg)),
                offset: to_start(index),
                cond: PredSrc::TRUE,
            })),
        };
        assemble(&instr).unwrap_or_default()
    }
}

fn any_pred() -> impl Strategy<Value = PredSrc> {
    prop_oneof![
        3 => Just(PredSrc::TRUE),
        1 => (0_u8..8, any::<bool>()).prop_map(|(reg, negated)| PredSrc {
            reg: crate::decode::Pred::new(reg).expect("a predicate"),
            negated,
        }),
    ]
}

fn any_word() -> impl Strategy<Value = Word> {
    prop_oneof![
        any_form_word().prop_map(Word::Any),
        (any_pred(), any_pred(), -8_i8..8).prop_map(|(g, c, d)| Word::Bra(g, c, d)),
        (0_u8..3, 0_i8..8).prop_map(|(b, d)| Word::Bssy(b, d)),
        (0_u8..3).prop_map(|b| Word::Fixed(bsync(b))),
        (any_pred(), -8_i8..8).prop_map(|(g, d)| Word::Call(g, d)),
        (any_pred(), 8_u8..12).prop_map(|(g, r)| Word::Ret(g, r)),
        (0_u8..4).prop_map(Word::Brx),
        (8_u8..12, 0_u32..64).prop_map(|(d, v)| Word::Fixed(Instr::new(Op::Mov(crate::decode::OpMov {
            dst: super::r(d),
            src: crate::decode::MovSrc::Imm(v * 16),
            quad_lanes: crate::decode::OpMov::ALL_LANES,
        })))),
        any_pred().prop_map(|g| Word::Fixed(exit_if(g, PredSrc::TRUE))),
        Just(Word::Fixed(exit())),
        any_pred().prop_map(|g| Word::Any(side_effect(g))),
    ]
}
