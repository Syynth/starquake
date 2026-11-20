//! Interpreter equivalence: generated programs (assignments, comparisons, selects, uniform
//! registers, predicates and flags, nested `if`s and counted loops) give the same machine
//! before and after each pass of [`Pass::ORDER`], and every output verifies; as control-flow
//! graphs, and structured by the structurizer first (slice 8b, Eden's order), when no goto or
//! condition variable is left after the passes.
//!
//! Until dead code elimination the whole machine must match, guest registers included (the
//! set operations stay); after it, the guest registers, predicates, flags and goto variables
//! are no longer written (nothing reads them after the SSA rewrite), so everything else must
//! match. Each program ends by storing its data registers and predicates to output
//! attributes, so their values stay observable.

use proptest::collection::vec;
use proptest::prelude::*;
use proptest::test_runner::Config;

use crate::decode::{Gpr, Pred};
use crate::interp::{
    ConditionCodes, Machine, PRED_SLOTS, REG_SLOTS, UNIFORM_PRED_SLOTS, UNIFORM_REG_SLOTS,
};
use crate::ir::{
    Attribute, Builder, Function, Imm, Opcode, Operand, Terminator, UniformPred, UniformReg,
};
use crate::opt::{Pass, run};
use crate::structure::{MAX_STRUCTURE_WORK, structure};

/// The data registers a program computes in: r0 to r4.
const DATA_REGS: u8 = 5;
/// The first register a loop counts in (one per nesting level), above the data registers.
const COUNTER_BASE: u8 = 8;
/// The predicates a program writes: p0 to p2.
const PREDS: u8 = 3;
/// The uniform registers and predicates a program writes: ur0, ur1; up0, up1.
const UNIFORMS: u8 = 2;
/// How many constant buffer words a program reads.
const CBUF_WORDS: u32 = 8;
/// The first output attribute word: generic attribute 0 (`attribute.h`: `Generic0X = 32`).
const OUTPUT_BASE: u16 = 32;

/// A source operand.
#[derive(Clone, Copy, Debug)]
enum Src {
    Reg(u8),
    Imm(u32),
    Cbuf(u32),
    UniformReg(u8),
}

/// A statement of a generated program.
#[derive(Clone, Debug)]
enum Stmt {
    /// `rD = a op b`.
    Assign { dst: u8, op: Opcode, a: Src, b: Src },
    /// `pD = a cmp b`.
    Compare { dst: u8, op: Opcode, a: Src, b: Src },
    /// `pD = pA op pB` (`LogicalAnd` or `LogicalOr`), or `!pA` (`LogicalNot`).
    Logic { dst: u8, op: Opcode, a: u8, b: u8 },
    /// `rD = pP ? a : b`.
    Select { dst: u8, pred: u8, a: Src, b: Src },
    /// `urD = a`, `upD = pP`.
    Uniform { dst: u8, src: Src, pred: u8 },
    /// `Z = (a != 0)`, then `pD = Z`.
    Flag { src: Src, dst: u8 },
    /// `if pP { then } else { otherwise }`.
    If {
        pred: u8,
        then: Vec<Stmt>,
        otherwise: Vec<Stmt>,
    },
    /// The body `count` times.
    Loop { count: u32, body: Vec<Stmt> },
}

const BINARY: [Opcode; 9] = [
    Opcode::IAdd32,
    Opcode::ISub32,
    Opcode::IMul32,
    Opcode::BitwiseAnd32,
    Opcode::BitwiseOr32,
    Opcode::BitwiseXor32,
    Opcode::ShiftLeftLogical32,
    Opcode::ShiftRightArithmetic32,
    Opcode::ShiftRightLogical32,
];

const COMPARE: [Opcode; 6] = [
    Opcode::IEqual,
    Opcode::INotEqual,
    Opcode::ULessThan,
    Opcode::SLessThan,
    Opcode::SGreaterThanEqual,
    Opcode::UGreaterThan,
];

fn imm() -> BoxedStrategy<u32> {
    prop_oneof![
        Just(0),
        Just(1),
        Just(16),
        Just(31),
        Just(u32::MAX),
        any::<u32>()
    ]
    .boxed()
}

fn src() -> BoxedStrategy<Src> {
    prop_oneof![
        3 => (0..DATA_REGS).prop_map(Src::Reg),
        3 => imm().prop_map(Src::Imm),
        1 => (0..CBUF_WORDS).prop_map(Src::Cbuf),
        1 => (0..UNIFORMS).prop_map(Src::UniformReg),
    ]
    .boxed()
}

fn leaf() -> BoxedStrategy<Stmt> {
    prop_oneof![
        4 => (0..DATA_REGS, 0..BINARY.len(), src(), src())
            .prop_map(|(dst, op, a, b)| Stmt::Assign { dst, op: BINARY[op], a, b }),
        2 => (0..PREDS, 0..COMPARE.len(), src(), src())
            .prop_map(|(dst, op, a, b)| Stmt::Compare { dst, op: COMPARE[op], a, b }),
        1 => (0..PREDS, 0..3_usize, 0..PREDS, 0..PREDS).prop_map(|(dst, op, a, b)| Stmt::Logic {
            dst,
            op: [Opcode::LogicalAnd, Opcode::LogicalOr, Opcode::LogicalNot][op],
            a,
            b,
        }),
        1 => (0..DATA_REGS, 0..PREDS, src(), src())
            .prop_map(|(dst, pred, a, b)| Stmt::Select { dst, pred, a, b }),
        1 => (0..UNIFORMS, src(), 0..PREDS).prop_map(|(dst, src, pred)| Stmt::Uniform { dst, src, pred }),
        1 => (src(), 0..PREDS).prop_map(|(src, dst)| Stmt::Flag { src, dst }),
    ]
    .boxed()
}

fn stmt() -> BoxedStrategy<Stmt> {
    leaf().prop_recursive(3, 32, 4, |inner| {
        prop_oneof![
            3 => leaf(),
            1 => (0..PREDS, vec(inner.clone(), 0..4), vec(inner.clone(), 0..4))
                .prop_map(|(pred, then, otherwise)| Stmt::If { pred, then, otherwise }),
            1 => (1..=3_u32, vec(inner, 0..4)).prop_map(|(count, body)| Stmt::Loop { count, body }),
        ]
    })
    .boxed()
}

/// Builds a program as a control-flow graph.
struct Emit {
    function: Function,
    block: crate::ir::BlockId,
}

fn reg(index: u8) -> Operand {
    Operand::Imm(Imm::Reg(Gpr::new(index)))
}

fn pred(index: u8) -> Operand {
    Operand::Imm(Imm::Pred(Pred::new(index).expect("a predicate")))
}

fn uniform_reg(index: u8) -> Operand {
    Operand::Imm(Imm::UniformReg(
        UniformReg::new(index).expect("a uniform register"),
    ))
}

fn uniform_pred(index: u8) -> Operand {
    Operand::Imm(Imm::UniformPred(
        UniformPred::new(index).expect("a uniform predicate"),
    ))
}

fn u32_imm(value: u32) -> Operand {
    Operand::Imm(Imm::U32(value))
}

impl Emit {
    fn op(&mut self, opcode: Opcode, args: &[Operand]) -> Operand {
        let mut builder = Builder::new(&mut self.function, self.block);
        Operand::Value(builder.op(opcode, args).expect("well-formed"))
    }

    fn terminate(&mut self, terminator: Terminator) {
        Builder::new(&mut self.function, self.block)
            .terminate(terminator)
            .expect("well-formed");
    }

    fn new_block(&mut self) -> crate::ir::BlockId {
        self.function.add_block().expect("room")
    }

    fn src(&mut self, src: Src) -> Operand {
        match src {
            Src::Reg(index) => self.op(Opcode::GetRegister, &[reg(index)]),
            Src::Imm(value) => u32_imm(value),
            Src::Cbuf(word) => self.op(Opcode::GetCbufU32, &[u32_imm(0), u32_imm(word * 4)]),
            Src::UniformReg(index) => self.op(Opcode::GetUniformRegister, &[uniform_reg(index)]),
        }
    }

    fn stmts(&mut self, stmts: &[Stmt], depth: u8) {
        for stmt in stmts {
            self.stmt(stmt, depth);
        }
    }

    fn stmt(&mut self, stmt: &Stmt, depth: u8) {
        match *stmt {
            Stmt::Assign { dst, op, a, b } => {
                let (a, b) = (self.src(a), self.src(b));
                let value = self.op(op, &[a, b]);
                self.op(Opcode::SetRegister, &[reg(dst), value]);
            }
            Stmt::Compare { dst, op, a, b } => {
                let (a, b) = (self.src(a), self.src(b));
                let value = self.op(op, &[a, b]);
                self.op(Opcode::SetPred, &[pred(dst), value]);
            }
            Stmt::Logic { dst, op, a, b } => {
                let a = self.op(Opcode::GetPred, &[pred(a)]);
                let value = if op == Opcode::LogicalNot {
                    self.op(op, &[a])
                } else {
                    let b = self.op(Opcode::GetPred, &[pred(b)]);
                    self.op(op, &[a, b])
                };
                self.op(Opcode::SetPred, &[pred(dst), value]);
            }
            Stmt::Select { dst, pred: p, a, b } => {
                let cond = self.op(Opcode::GetPred, &[pred(p)]);
                let (a, b) = (self.src(a), self.src(b));
                let value = self.op(Opcode::SelectU32, &[cond, a, b]);
                self.op(Opcode::SetRegister, &[reg(dst), value]);
            }
            Stmt::Uniform { dst, src, pred: p } => {
                let value = self.src(src);
                self.op(Opcode::SetUniformRegister, &[uniform_reg(dst), value]);
                let bit = self.op(Opcode::GetPred, &[pred(p)]);
                self.op(Opcode::SetUniformPred, &[uniform_pred(dst), bit]);
            }
            Stmt::Flag { src, dst } => {
                let value = self.src(src);
                let bit = self.op(Opcode::INotEqual, &[value, u32_imm(0)]);
                self.op(Opcode::SetZFlag, &[bit]);
                let zero = self.op(Opcode::GetZFlag, &[]);
                self.op(Opcode::SetPred, &[pred(dst), zero]);
            }
            Stmt::If {
                pred: p,
                ref then,
                ref otherwise,
            } => {
                let cond = self.op(Opcode::GetPred, &[pred(p)]);
                let (then_block, else_block, join) =
                    (self.new_block(), self.new_block(), self.new_block());
                self.terminate(Terminator::Branch {
                    cond,
                    then: then_block,
                    otherwise: else_block,
                });
                self.block = then_block;
                self.stmts(then, depth);
                self.terminate(Terminator::Jump(join));
                self.block = else_block;
                self.stmts(otherwise, depth);
                self.terminate(Terminator::Jump(join));
                self.block = join;
            }
            Stmt::Loop { count, ref body } => {
                let counter = reg(COUNTER_BASE + depth);
                self.op(Opcode::SetRegister, &[counter, u32_imm(count)]);
                let header = self.new_block();
                self.terminate(Terminator::Jump(header));
                self.block = header;
                self.stmts(body, depth + 1);
                let left = self.op(Opcode::GetRegister, &[counter]);
                let next = self.op(Opcode::ISub32, &[left, u32_imm(1)]);
                self.op(Opcode::SetRegister, &[counter, next]);
                let again = self.op(Opcode::INotEqual, &[next, u32_imm(0)]);
                let exit = self.new_block();
                self.terminate(Terminator::Branch {
                    cond: again,
                    then: header,
                    otherwise: exit,
                });
                self.block = exit;
            }
        }
    }

    /// Stores every data register, predicate, uniform register and predicate to an output.
    fn finish(&mut self) {
        let mut attribute = OUTPUT_BASE;
        let mut store = |emit: &mut Self, value: Operand| {
            let bits = emit.op(Opcode::BitCastF32U32, &[value]);
            emit.op(
                Opcode::SetAttribute,
                &[
                    Operand::Imm(Imm::Attribute(Attribute(attribute))),
                    bits,
                    u32_imm(0),
                ],
            );
            attribute += 1;
        };
        for index in 0..DATA_REGS {
            let value = self.op(Opcode::GetRegister, &[reg(index)]);
            store(self, value);
        }
        for index in 0..PREDS {
            let bit = self.op(Opcode::GetPred, &[pred(index)]);
            let value = self.op(Opcode::SelectU32, &[bit, u32_imm(1), u32_imm(0)]);
            store(self, value);
        }
        for index in 0..UNIFORMS {
            let value = self.op(Opcode::GetUniformRegister, &[uniform_reg(index)]);
            store(self, value);
            let bit = self.op(Opcode::GetUniformPred, &[uniform_pred(index)]);
            let value = self.op(Opcode::SelectU32, &[bit, u32_imm(1), u32_imm(0)]);
            store(self, value);
        }
        self.terminate(Terminator::Return);
    }
}

/// The program `stmts` as a function.
fn build(stmts: &[Stmt]) -> Function {
    let mut emit = Emit {
        function: Function::new(),
        block: Function::new().entry(),
    };
    emit.stmts(stmts, 0);
    emit.finish();
    emit.function
}

/// The machine with the guest state dead code elimination may leave unwritten cleared.
fn observable(machine: &Machine) -> Machine {
    Machine {
        regs: [0; REG_SLOTS],
        preds: [false; PRED_SLOTS],
        uniform_regs: [0; UNIFORM_REG_SLOTS],
        uniform_preds: [false; UNIFORM_PRED_SLOTS],
        flags: ConditionCodes::default(),
        goto_variables: std::collections::BTreeMap::new(),
        ..machine.clone()
    }
}

/// Runs every pass on `function`, checking each output verifies, keeps the result and, when
/// structured, keeps each phi's incoming values in scope ([`super::check_phi_scope`]): so the
/// output of the SSA rewrite alone (the first pass) and of all of them (what
/// [`optimize`](crate::opt::optimize) runs) are both checked.
fn check_equivalence(function: &Function, start: &Machine) -> Result<Function, TestCaseError> {
    let reference = super::ran(function, start);
    let mut current = function.clone();
    for pass in Pass::ORDER {
        run(&mut current, pass).map_err(|e| TestCaseError::fail(format!("{pass}: {e}")))?;
        current
            .verify()
            .map_err(|e| TestCaseError::fail(format!("{pass}: {e}")))?;
        super::check_phi_scope(&current)
            .map_err(|e| TestCaseError::fail(format!("{pass}: {e}")))?;
        let mut machine = start.clone();
        machine
            .run(&current)
            .map_err(|e| TestCaseError::fail(format!("{pass}: {e}\n{current}")))?;
        if pass == Pass::DeadCodeElimination {
            prop_assert_eq!(
                observable(&machine),
                observable(&reference),
                "{}: {}",
                pass,
                current
            );
        } else {
            prop_assert_eq!(&machine, &reference, "{}: {}", pass, current);
        }
    }
    Ok(current)
}

/// [`check_equivalence`] on `function` structured first, which must run as `function` does;
/// the output of each pass keeps its phis' incoming values in scope, and the last names no goto
/// variable.
fn check_structured(function: &Function, start: &Machine) -> Result<(), TestCaseError> {
    let structured = structure(function.clone(), MAX_STRUCTURE_WORK)
        .map_err(|e| TestCaseError::fail(format!("structuring: {e}")))?
        .function;
    let reference = super::ran(function, start);
    let machine = super::ran(&structured, start);
    prop_assert_eq!(observable(&machine), observable(&reference));
    let optimized = check_equivalence(&structured, start)?;
    let text = optimized.to_string();
    prop_assert!(!text.contains("GotoVariable"), "{}", text);
    Ok(())
}

fn start(words: &[u32]) -> Machine {
    let mut machine = Machine::new();
    machine
        .cbufs
        .insert(0, words.iter().flat_map(|w| w.to_le_bytes()).collect());
    machine
}

proptest! {
    #![proptest_config(Config { cases: 256, ..crate::testing::config() })]

    #[test]
    fn generated_programs_keep_their_results(
        stmts in vec(stmt(), 1..8),
        words in vec(any::<u32>(), CBUF_WORDS as usize),
    ) {
        let function = build(&stmts);
        prop_assert!(function.verify().is_ok());
        check_equivalence(&function, &start(&words))?;
    }

    #[test]
    fn generated_programs_keep_their_results_structured(
        stmts in vec(stmt(), 1..8),
        words in vec(any::<u32>(), CBUF_WORDS as usize),
    ) {
        check_structured(&build(&stmts), &start(&words))?;
    }
}

#[test]
fn the_generator_makes_loops_ifs_and_phis() {
    // A hand-picked program of every statement kind, nested, keeps its result, and its SSA
    // form has phis (so the generated programs exercise them).
    let program = vec![
        Stmt::Assign {
            dst: 0,
            op: Opcode::IAdd32,
            a: Src::Cbuf(0),
            b: Src::Imm(1),
        },
        Stmt::Loop {
            count: 3,
            body: vec![
                Stmt::Assign {
                    dst: 0,
                    op: Opcode::IMul32,
                    a: Src::Reg(0),
                    b: Src::Imm(3),
                },
                Stmt::Compare {
                    dst: 1,
                    op: Opcode::ULessThan,
                    a: Src::Reg(0),
                    b: Src::Cbuf(1),
                },
                Stmt::If {
                    pred: 1,
                    then: vec![Stmt::Select {
                        dst: 2,
                        pred: 1,
                        a: Src::Reg(0),
                        b: Src::Imm(7),
                    }],
                    otherwise: vec![
                        Stmt::Flag {
                            src: Src::Reg(0),
                            dst: 2,
                        },
                        Stmt::Uniform {
                            dst: 1,
                            src: Src::Reg(2),
                            pred: 2,
                        },
                    ],
                },
                Stmt::Logic {
                    dst: 0,
                    op: Opcode::LogicalOr,
                    a: 1,
                    b: 2,
                },
            ],
        },
    ];
    let function = build(&program);
    for words in [[5, 100, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]] {
        check_equivalence(&function, &start(&words)).expect("equivalent");
        check_structured(&function, &start(&words)).expect("equivalent structured");
    }
    let mut ssa = function.clone();
    run(&mut ssa, Pass::SsaRewrite).expect("rewrites");
    assert!(ssa.to_string().contains("Phi"), "{ssa}");
    // Structured, the rewrite places phis too, and the structurizer's loop is there.
    let mut structured = structure(function, MAX_STRUCTURE_WORK)
        .expect("structures")
        .function;
    run(&mut structured, Pass::SsaRewrite).expect("rewrites");
    let text = structured.to_string();
    assert!(text.contains("Phi") && text.contains("loop {"), "{text}");
}

#[test]
fn a_join_of_two_copies_of_one_value_verifies() {
    // The generator's first failure, shrunk: both sides of an `if` copy r0 into ur0, so the
    // join's phi of ur0 is trivial, and must stand for r0's value, which dominates the join,
    // not for one side's copy (`ssa` module docs, "A trivial phi's value").
    let program = vec![Stmt::Loop {
        count: 1,
        body: vec![Stmt::If {
            pred: 0,
            then: vec![Stmt::Uniform {
                dst: 0,
                src: Src::Reg(0),
                pred: 2,
            }],
            otherwise: vec![Stmt::Uniform {
                dst: 0,
                src: Src::Reg(0),
                pred: 0,
            }],
        }],
    }];
    check_equivalence(&build(&program), &start(&[0; CBUF_WORDS as usize])).expect("equivalent");
    check_structured(&build(&program), &start(&[0; CBUF_WORDS as usize])).expect("equivalent");
}
