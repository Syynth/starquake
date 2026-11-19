//! Flow analysis tests: programs built with the test assembler and a label-resolving builder,
//! checked against expected walks, block graphs and IR.

mod analysis;
mod classify;
mod graph;
mod inline;
mod ir;
mod jump_tables;
mod linear;
mod models;

use std::collections::BTreeMap;
use std::ops::Range;

use super::{
    BarrierModel, BlockEnd, BlockIndex, BranchTables, FlowBlock, FlowError, FlowGraph,
    IndirectTarget, analyze, flow_graph,
};
use crate::asm::assemble;
use crate::decode::attribute::AST_OPCODE;
use crate::decode::{
    Barrier, BraCond, BranchOffset, BrxSrc, GUARD, GUARD_NOT, Gpr, Instr, Instruction, MovSrc,
    OPCODE, Op, OpBSSy, OpBSync, OpBra, OpBreak, OpBrx, OpCall, OpExit, OpKill, OpMov, OpRet, Pred,
    PredSrc,
};

/// Predicate `n`.
pub(super) fn p(n: u8) -> PredSrc {
    PredSrc {
        reg: Pred::new(n).expect("a predicate"),
        negated: false,
    }
}

/// Predicate `n`, negated.
pub(super) fn not_p(n: u8) -> PredSrc {
    PredSrc {
        negated: true,
        ..p(n)
    }
}

/// `!PT`: never true.
pub(super) const NEVER: PredSrc = PredSrc {
    reg: Pred::PT,
    negated: true,
};

/// Register `n`.
pub(super) fn r(n: u8) -> Gpr {
    Gpr::new(n)
}

/// Barrier `n`.
pub(super) fn b(n: u8) -> Barrier {
    Barrier::new(n).expect("a barrier")
}

/// [`Instruction::SIZE`] as a `u32`.
pub(super) fn size() -> u32 {
    u32::try_from(Instruction::SIZE).expect("16 bytes")
}

/// An instruction that is pure: a `MOV` of an immediate.
pub(super) fn pure(dst: u8) -> Instr {
    Instr::new(Op::Mov(OpMov {
        dst: r(dst),
        src: MovSrc::Imm(0x1234),
        quad_lanes: OpMov::ALL_LANES,
    }))
}

/// A word with a side effect (an `AST` opcode, which flow analysis does not decode) under
/// `guard`.
pub(super) fn side_effect(guard: PredSrc) -> Instruction {
    Instruction::default()
        .with_field(OPCODE, u64::from(AST_OPCODE))
        .with_field(GUARD, u64::from(guard.reg.index()))
        .with_flag(GUARD_NOT, guard.negated)
}

/// `EXIT` with `guard` and its own predicate `pred`.
pub(super) fn exit_if(guard: PredSrc, pred: PredSrc) -> Instr {
    Instr {
        guard,
        op: Op::Exit(OpExit {
            keep_ref_count: false,
            no_at_exit: false,
            pred,
        }),
    }
}

/// An unconditional `EXIT`.
pub(super) fn exit() -> Instr {
    exit_if(PredSrc::TRUE, PredSrc::TRUE)
}

/// `BSYNC bar`.
pub(super) fn bsync(bar: u8) -> Instr {
    Instr::new(Op::BSync(OpBSync {
        bar: b(bar),
        cond: PredSrc::TRUE,
    }))
}

/// `BREAK bar` under `guard`.
pub(super) fn brk(bar: u8, guard: PredSrc) -> Instr {
    Instr {
        guard,
        op: Op::Break(OpBreak {
            bar: b(bar),
            cond: PredSrc::TRUE,
        }),
    }
}

/// `KILL` under `guard`.
pub(super) fn kill(guard: PredSrc) -> Instr {
    Instr {
        guard,
        op: Op::Kill(OpKill {
            pred: PredSrc::TRUE,
        }),
    }
}

/// An item of a program being built.
#[derive(Clone, Debug)]
pub(super) enum Item {
    /// A finished word.
    Word(Instruction),
    /// A `BRA` to a label.
    Bra {
        guard: PredSrc,
        cond: PredSrc,
        label: &'static str,
    },
    /// A `BSSY` to a label.
    Bssy { bar: u8, label: &'static str },
    /// A `CALL` to a label.
    Call { guard: PredSrc, label: &'static str },
    /// The `MOV` of the return address of the `CALL` right after it.
    Link { dst: u8 },
    /// `RET` through `reg`, relative to the program's start.
    Ret { guard: PredSrc, reg: u8 },
    /// `BRX` through `src`, relative to the program's start.
    Brx(BrxSrc),
}

/// A program with labels, resolved when built.
#[derive(Clone, Debug, Default)]
pub(super) struct Program {
    items: Vec<Item>,
    labels: BTreeMap<&'static str, u32>,
}

/// The offset from instruction `from`'s end to instruction `to`.
pub(super) fn offset(from: u32, to: u32) -> BranchOffset {
    let words =
        (i64::from(to) - i64::from(from) - 1) * i64::from(size()) / BranchOffset::WORD_BYTES;
    BranchOffset::from_words(words)
}

/// The offset from instruction `from`'s end back to instruction 0.
pub(super) fn to_start(from: u32) -> BranchOffset {
    BranchOffset::from_words(-(i64::from(from) + 1) * i64::from(size()) / BranchOffset::WORD_BYTES)
}

impl Program {
    pub(super) fn new() -> Self {
        Self::default()
    }

    /// The index the next item gets.
    pub(super) fn here(&self) -> u32 {
        u32::try_from(self.items.len()).expect("a small program")
    }

    pub(super) fn item(&mut self, item: Item) -> &mut Self {
        self.items.push(item);
        self
    }

    pub(super) fn op(&mut self, instr: Instr) -> &mut Self {
        self.item(Item::Word(
            assemble(&instr).expect("an encodable instruction"),
        ))
    }

    pub(super) fn word(&mut self, word: Instruction) -> &mut Self {
        self.item(Item::Word(word))
    }

    pub(super) fn label(&mut self, name: &'static str) -> &mut Self {
        let here = self.here();
        assert!(
            self.labels.insert(name, here).is_none(),
            "label {name} twice"
        );
        self
    }

    pub(super) fn bra(&mut self, guard: PredSrc, label: &'static str) -> &mut Self {
        self.item(Item::Bra {
            guard,
            cond: PredSrc::TRUE,
            label,
        })
    }

    pub(super) fn bssy(&mut self, bar: u8, label: &'static str) -> &mut Self {
        self.item(Item::Bssy { bar, label })
    }

    /// The return-address `MOV` into `link` and an unconditional `CALL` of `label`.
    pub(super) fn call(&mut self, link: u8, label: &'static str) -> &mut Self {
        self.item(Item::Link { dst: link }).item(Item::Call {
            guard: PredSrc::TRUE,
            label,
        })
    }

    pub(super) fn ret(&mut self, reg: u8) -> &mut Self {
        self.item(Item::Ret {
            guard: PredSrc::TRUE,
            reg,
        })
    }

    /// The instruction index of `label`.
    pub(super) fn at(&self, label: &str) -> u32 {
        *self.labels.get(label).expect("a defined label")
    }

    pub(super) fn build(&self) -> Vec<Instruction> {
        (0_u32..)
            .zip(&self.items)
            .map(|(index, item)| self.word_of(index, item))
            .collect()
    }

    fn word_of(&self, index: u32, item: &Item) -> Instruction {
        let instr = match *item {
            Item::Word(word) => return word,
            Item::Bra { guard, cond, label } => Instr {
                guard,
                op: Op::Bra(OpBra {
                    cond: BraCond::Pred(cond),
                    uniform: false,
                    offset: offset(index, self.at(label)),
                }),
            },
            Item::Bssy { bar, label } => Instr::new(Op::BSSy(OpBSSy {
                bar: b(bar),
                offset: offset(index, self.at(label)),
                cond: PredSrc::TRUE,
            })),
            Item::Call { guard, label } => Instr {
                guard,
                op: Op::Call(OpCall {
                    offset: offset(index, self.at(label)),
                    no_inc: true,
                    cond: PredSrc::TRUE,
                }),
            },
            Item::Link { dst } => Instr::new(Op::Mov(OpMov {
                dst: r(dst),
                src: MovSrc::Imm((index + 2) * size()),
                quad_lanes: OpMov::ALL_LANES,
            })),
            Item::Ret { guard, reg } => Instr {
                guard,
                op: Op::Ret(OpRet {
                    src: r(reg),
                    offset: to_start(index),
                    no_dec: true,
                    cond: PredSrc::TRUE,
                }),
            },
            Item::Brx(src) => Instr::new(Op::Brx(OpBrx {
                src,
                offset: to_start(index),
                cond: PredSrc::TRUE,
            })),
        };
        assemble(&instr).expect("an encodable instruction")
    }
}

/// A block of an expected graph.
pub(super) fn block(code: Range<u32>, end: BlockEnd) -> FlowBlock {
    FlowBlock {
        code,
        guard_tested: false,
        end,
    }
}

/// A block of an expected graph whose guard the graph tests.
pub(super) fn tested(code: Range<u32>, end: BlockEnd) -> FlowBlock {
    FlowBlock {
        code,
        guard_tested: true,
        end,
    }
}

/// A jump to block `n`.
pub(super) fn jump(n: u32) -> BlockEnd {
    BlockEnd::Jump(BlockIndex(n))
}

/// A branch to `then` where `cond` holds, else `otherwise`.
pub(super) fn branch(cond: PredSrc, then: u32, otherwise: u32) -> BlockEnd {
    BlockEnd::Branch {
        cond,
        then: BlockIndex(then),
        otherwise: BlockIndex(otherwise),
    }
}

/// An indirect branch through `src` to (address, block) pairs.
pub(super) fn indirect(src: BrxSrc, targets: &[(u32, u32)]) -> BlockEnd {
    BlockEnd::Indirect {
        src,
        targets: targets
            .iter()
            .map(|&(address, block)| IndirectTarget {
                address,
                block: BlockIndex(block),
            })
            .collect(),
    }
}

/// A label made at run time (the builder takes `&'static str`); test-only, so leaking it is
/// harmless.
pub(super) fn name(text: String) -> &'static str {
    Box::leak(text.into_boxed_str())
}

/// The flow graph of `code` with no jump tables.
pub(super) fn graph(code: &[Instruction]) -> Result<FlowGraph, FlowError> {
    flow_graph(code, &BranchTables::new(0))
}

/// The error of analysing `code` under both models, which must agree on it.
pub(super) fn refusal(code: &[Instruction]) -> FlowError {
    let tables = BranchTables::new(0);
    let registers = analyze(code, &tables, BarrierModel::Registers).expect_err("a refusal");
    let stack = analyze(code, &tables, BarrierModel::Stack).expect_err("a refusal");
    assert_eq!(registers, stack, "the models refuse differently");
    registers
}
