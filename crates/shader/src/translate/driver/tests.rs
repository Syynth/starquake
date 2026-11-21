//! The translation driver's tests: whole programs, hand-assembled with the test assembler, with
//! branches, a loop, an indirect branch, calls and guarded instructions, run through the IR
//! interpreter (slice 7) as the control-flow graph the driver fills and as the structured
//! function the structurizer makes of it; the order of the checks with two faults at once; the
//! translation budget at its edge; the dispatch's refusals by name.

use std::collections::BTreeMap;

use super::*;
use crate::asm::assemble;
use crate::capture::MAX_INSTRUCTIONS;
use crate::decode::{
    AluSrc, BREAK_OPCODE, BraCond, BranchOffset, BrxSrc, DecodeError, GUARD, Gpr, IntCmpOp,
    MemAddr, MemEvictionPriority, MemSpace, MemType, MovSrc, OPCODE, OffsetStride, OpALd, OpASt,
    OpBra, OpBrx, OpCall, OpDepBar, OpExit, OpFAdd, OpHmma, OpIAdd3, OpISetP, OpKill, OpLd, OpMov,
    OpNop, OpRet, OpS2R, OpVote, OpWarpSync, Pred, PredSetOp, PredSrc, RoundMode, S2RDst,
    SpecialReg, UniformAddr, UniformPred, UniformReg, VoteDst, VoteOp,
};
use crate::interp::Machine;
use crate::ir::{Attribute, BuildError, Inst, Node, Opcode};
use crate::structure::MAX_STRUCTURE_WORK;
use crate::testing::{canonical_word, forms};
use crate::translate::control::{DEPBAR_REQUIRED_SCOREBOARD, FULL_WARP_MASK};
use crate::translate::{COLOR_TARGETS, ColorOutputs, attribute};

const VERTEX: Stage = Stage {
    stage: GuestStage::Vertex,
    color_outputs: ColorOutputs {
        targets: [0; COLOR_TARGETS],
    },
};

/// A pixel program writing component 0 of colour target 0.
const PIXEL: Stage = Stage {
    stage: GuestStage::Pixel,
    color_outputs: ColorOutputs {
        targets: [0b0001, 0, 0, 0, 0, 0, 0, 0],
    },
};

/// The `!PT` carry input that adds nothing.
const NOT_PT: PredSrc = PredSrc {
    reg: Pred::PT,
    negated: true,
};

/// `!PT`: never true.
const NEVER: PredSrc = NOT_PT;

fn r(index: u8) -> Gpr {
    Gpr::new(index)
}

fn p(index: u8) -> PredSrc {
    PredSrc {
        reg: Pred::new(index).expect("a predicate"),
        negated: false,
    }
}

fn not_p(index: u8) -> PredSrc {
    PredSrc {
        negated: true,
        ..p(index)
    }
}

fn ur(index: u8) -> UniformReg {
    UniformReg::new(index).expect("a uniform register")
}

/// [`Instruction::SIZE`] as a `u32`.
fn size() -> u32 {
    u32::try_from(Instruction::SIZE).expect("16 bytes")
}

fn under(guard: PredSrc, instr: Instr) -> Instr {
    Instr { guard, ..instr }
}

fn mov(dst: u8, bits: u32) -> Instr {
    Instr::new(Op::Mov(OpMov {
        dst: r(dst),
        src: MovSrc::Imm(bits),
        quad_lanes: OpMov::ALL_LANES,
    }))
}

/// `IADD3 dst, a, imm, RZ`.
fn add_imm(dst: u8, a: u8, imm: u32) -> Instr {
    Instr::new(Op::IAdd3(OpIAdd3 {
        dst: r(dst),
        overflow: [Pred::PT; 2],
        srcs: [AluSrc::reg(r(a)), AluSrc::Imm(imm), AluSrc::reg(Gpr::RZ)],
        carry: [NOT_PT; 2],
    }))
}

/// `ISETP.LT.U32.AND dst, PT, a, imm, PT`.
fn less_than(dst: u8, a: u8, imm: u32) -> Instr {
    Instr::new(Op::ISetP(OpISetP {
        dsts: [Pred::new(dst).expect("a predicate"), Pred::PT],
        srcs: [AluSrc::reg(r(a)), AluSrc::Imm(imm)],
        cmp_op: IntCmpOp::Lt,
        signed: false,
        set_op: PredSetOp::And,
        ex: false,
        accum: PredSrc::TRUE,
        low_cmp: PredSrc::TRUE,
    }))
}

/// `FADD dst, a, b`.
fn fadd(dst: u8, a: u8, b: u8) -> Instr {
    Instr::new(Op::FAdd(OpFAdd {
        dst: r(dst),
        srcs: [AluSrc::reg(r(a)), AluSrc::reg(r(b))],
        ignored: Gpr::RZ,
        saturate: false,
        rnd: RoundMode::NearestEven,
        ftz: false,
    }))
}

fn ast(data: u8, addr: u16) -> Instr {
    Instr::new(Op::ASt(OpASt {
        data: r(data),
        vtx: Gpr::RZ,
        offset: Gpr::RZ,
        addr,
        comps: 1,
        patch: false,
        phys: false,
    }))
}

fn ald(dst: u8, addr: u16) -> Instr {
    Instr::new(Op::ALd(OpALd {
        dst: r(dst),
        vtx: Gpr::RZ,
        offset: Gpr::RZ,
        addr,
        comps: 1,
        patch: false,
        phys: false,
        output: false,
    }))
}

fn exit_if(pred: PredSrc) -> Instr {
    Instr::new(Op::Exit(OpExit {
        keep_ref_count: false,
        no_at_exit: false,
        pred,
    }))
}

fn exit() -> Instr {
    exit_if(PredSrc::TRUE)
}

fn nop() -> Instr {
    Instr::new(Op::Nop(OpNop {}))
}

/// `VOTE.ANY ballot, vote, pred`.
fn vote_any(ballot: u8, vote: u8, pred: PredSrc) -> Instr {
    Instr::new(Op::Vote(OpVote {
        op: VoteOp::Any,
        dst: VoteDst::Reg {
            ballot: r(ballot),
            vote: Pred::new(vote).expect("a predicate"),
        },
        pred,
    }))
}

/// `VOTEU.ANY ballot, vote, pred`.
fn voteu_any(ballot: u8, vote: u8, pred: PredSrc) -> Instr {
    Instr::new(Op::Vote(OpVote {
        op: VoteOp::Any,
        dst: VoteDst::Uniform {
            ballot: ur(ballot),
            vote: UniformPred::new(vote).expect("a uniform predicate"),
        },
        pred,
    }))
}

/// How many times `function`'s text reads `what` (an opcode and its operand, such as
/// `GetPred p0`).
fn reads(function: &Function, what: &str) -> usize {
    function.to_string().matches(&format!("= {what}\n")).count()
}

/// An item of a program being built: a word, or a branch, call or return resolved when built.
#[derive(Clone, Copy, Debug)]
enum Item {
    Word(Instruction),
    Bra {
        guard: PredSrc,
        label: &'static str,
    },
    Call {
        label: &'static str,
    },
    /// The `MOV` of the return address of the `CALL` right after it.
    Link {
        dst: u8,
    },
    /// `RET` through `reg`, relative to the program's start.
    Ret {
        reg: u8,
    },
    /// `BRX` through `reg`, relative to the program's start.
    Brx {
        reg: u8,
    },
}

/// A program with labels.
#[derive(Clone, Debug, Default)]
struct Program {
    items: Vec<Item>,
    labels: BTreeMap<&'static str, u32>,
}

/// The offset from the end of instruction `from` to instruction `to`, in 32-bit words.
fn offset(from: u32, to: u32) -> BranchOffset {
    let bytes = (i64::from(to) - i64::from(from) - 1) * i64::from(size());
    BranchOffset::from_words(bytes / BranchOffset::WORD_BYTES)
}

impl Program {
    fn here(&self) -> u32 {
        u32::try_from(self.items.len()).expect("a small program")
    }

    fn op(&mut self, instr: Instr) -> &mut Self {
        self.items.push(Item::Word(
            assemble(&instr).expect("an encodable instruction"),
        ));
        self
    }

    fn item(&mut self, item: Item) -> &mut Self {
        self.items.push(item);
        self
    }

    fn label(&mut self, name: &'static str) -> &mut Self {
        let here = self.here();
        assert!(self.labels.insert(name, here).is_none(), "label {name}");
        self
    }

    fn at(&self, label: &str) -> u32 {
        *self.labels.get(label).expect("a label")
    }

    fn build(&self) -> Vec<Instruction> {
        (0_u32..)
            .zip(&self.items)
            .map(|(index, item)| {
                // Relative to the program's start: back to instruction 0's address.
                let to_start = offset(index, 0);
                let instr = match *item {
                    Item::Word(word) => return word,
                    Item::Bra { guard, label } => Instr {
                        guard,
                        op: Op::Bra(OpBra {
                            cond: BraCond::Pred(PredSrc::TRUE),
                            uniform: false,
                            offset: offset(index, self.at(label)),
                        }),
                    },
                    Item::Call { label } => Instr::new(Op::Call(OpCall {
                        offset: offset(index, self.at(label)),
                        no_inc: true,
                        cond: PredSrc::TRUE,
                    })),
                    Item::Link { dst } => mov(dst, (index + 2) * size()),
                    Item::Ret { reg } => Instr::new(Op::Ret(OpRet {
                        src: r(reg),
                        offset: to_start,
                        no_dec: true,
                        cond: PredSrc::TRUE,
                    })),
                    Item::Brx { reg } => Instr::new(Op::Brx(OpBrx {
                        src: BrxSrc::Reg(r(reg)),
                        offset: to_start,
                        cond: PredSrc::TRUE,
                    })),
                };
                assemble(&instr).expect("an encodable instruction")
            })
            .collect()
    }
}

/// `code` translated for `stage` with no jump tables; the function verifies.
fn translated(stage: Stage, code: &[Instruction]) -> IrBlocks {
    translated_with(stage, code, &BranchTables::new(0))
}

fn translated_with(stage: Stage, code: &[Instruction], tables: &BranchTables) -> IrBlocks {
    let blocks = translate_flow(stage, None, code, tables).expect("translates");
    assert_eq!(blocks.function.verify(), Ok(()));
    blocks
}

/// Runs `blocks` as the control-flow graph the driver filled and as the structured function the
/// structurizer makes of it, each on a machine `setup` prepares; both must end in the same
/// state, which is returned.
fn run_both(blocks: &IrBlocks, setup: impl Fn(&mut Machine)) -> Machine {
    let mut graph = Machine::default();
    setup(&mut graph);
    graph.run(&blocks.function).expect("the graph runs");
    let structured = blocks
        .clone()
        .structure(MAX_STRUCTURE_WORK)
        .expect("structures")
        .blocks
        .function;
    let mut tree = Machine::default();
    setup(&mut tree);
    tree.run(&structured).expect("the structured function runs");
    assert_eq!(graph.regs, tree.regs, "registers");
    assert_eq!(graph.outputs, tree.outputs, "attribute outputs");
    assert_eq!(graph.frag_colors, tree.frag_colors, "colours");
    assert_eq!(graph.uniform_regs, tree.uniform_regs, "uniform registers");
    assert_eq!(graph.demoted, tree.demoted, "demoted");
    graph
}

/// How many operations of `opcode` `function` holds.
fn count(function: &Function, opcode: Opcode) -> usize {
    function
        .listed()
        .filter(|(_, inst)| matches!(inst, Inst::Op(op) if op.opcode == opcode))
        .count()
}

/// A straight-line program fills block 0, which returns, and runs.
#[test]
fn a_straight_program_fills_one_block() {
    let code = Program::default()
        .op(mov(0, 3))
        .op(add_imm(1, 0, 4))
        .op(exit())
        .build();
    let blocks = translated(VERTEX, &code);
    assert_eq!(
        blocks.function.to_string(),
        "block0:\n\
         \x20   %0 = SetRegister r0, u32 0x3\n\
         \x20   %1: u32 = GetRegister r0\n\
         \x20   %2: u32 = IAdd32 %1, u32 0x4\n\
         \x20   %3: u32 = IAdd32 %2, u32 0x0\n\
         \x20   %4 = SetRegister r1, %3\n\
         \x20   return\n"
    );
    let machine = run_both(&blocks, |_| {});
    assert_eq!((machine.regs[0], machine.regs[1]), (3, 7));
}

/// A backward branch makes a loop: the body's instructions go in its block, before the test
/// block flow analysis puts the branch's test in, and the loop runs to its count.
#[test]
fn a_loop_runs_to_its_count() {
    let code = Program::default()
        .op(mov(0, 0))
        .label("loop")
        .op(add_imm(0, 0, 1))
        .op(less_than(0, 0, 5))
        .item(Item::Bra {
            guard: p(0),
            label: "loop",
        })
        .op(exit())
        .build();
    let blocks = translated(VERTEX, &code);
    let machine = run_both(&blocks, |_| {});
    assert_eq!(machine.regs[0], 5);
}

/// A guarded pure instruction (`MOV`, `classify.h`'s `pure`), which flow analysis leaves in its
/// block, becomes a conditional block (the owner's ruling of 2026-11-19, row 2; PR #4588 keeps a
/// select): the guard is read, the block branches to the instruction's block or past it, and
/// the next instruction goes in the block after. It runs exactly when the guard holds.
#[test]
fn a_guarded_pure_instruction_becomes_a_conditional_block() {
    for guard in [p(2), not_p(2)] {
        let code = Program::default()
            .op(under(guard, mov(0, 7)))
            .op(mov(1, 9))
            .op(exit())
            .build();
        let blocks = translated(VERTEX, &code);
        assert_eq!(count(&blocks.function, Opcode::SelectU32), 0);
        let read = if guard.negated {
            "\x20   %0: u1 = GetPred p2\n\x20   %1: u1 = LogicalNot %0\n\x20   branch %1, block1, block2\n"
        } else {
            "\x20   %0: u1 = GetPred p2\n\x20   branch %0, block1, block2\n"
        };
        let first = usize::from(guard.negated);
        assert_eq!(
            blocks.function.to_string(),
            format!(
                "block0:\n{read}block1:\n\
                 \x20   %{} = SetRegister r0, u32 0x7\n\
                 \x20   jump block2\n\
                 block2:\n\
                 \x20   %{} = SetRegister r1, u32 0x9\n\
                 \x20   return\n",
                first + 1,
                first + 2,
            )
        );
        for holds in [false, true] {
            let machine = run_both(&blocks, |machine| {
                machine.regs[0] = 3;
                machine.preds[2] = holds;
            });
            let runs = holds != guard.negated;
            assert_eq!(machine.regs[0], if runs { 7 } else { 3 }, "{guard} {holds}");
            assert_eq!(machine.regs[1], 9);
        }
    }
}

/// A guarded instruction that is not pure (`AST`) gets a block of its own from flow analysis,
/// which tests its guard: the driver does not test it again (one `GetPred`), and the attribute
/// is written exactly when the guard holds.
#[test]
fn a_guard_tested_block_is_not_tested_twice() {
    let code = Program::default()
        .op(mov(0, 1.5_f32.to_bits()))
        .op(under(p(2), ast(0, 0x80)))
        .op(exit())
        .build();
    let graph = flow_graph(&code, &BranchTables::new(0)).expect("a graph");
    assert!(graph.blocks().iter().any(|block| block.guard_tested));
    let blocks = translated(VERTEX, &code);
    assert_eq!(count(&blocks.function, Opcode::GetPred), 1);
    for holds in [false, true] {
        let machine = run_both(&blocks, |machine| machine.preds[2] = holds);
        assert_eq!(
            machine.outputs.get(&(Attribute(0x20), 0)).copied(),
            holds.then_some(1.5_f32.to_bits()),
            "{holds}"
        );
    }
}

/// The guard of a uniform datapath instruction (`S2UR`, `classify.h`'s `uniform`) reads the
/// uniform predicate of the same number; a vector one (`S2R`) reads the predicate.
#[test]
fn a_guard_on_a_uniform_instruction_reads_a_uniform_predicate() {
    let s2r = |dst| {
        Instr::new(Op::S2R(OpS2R {
            dst,
            sr: SpecialReg::LANE_ID,
        }))
    };
    let code = Program::default()
        .op(under(not_p(3), s2r(S2RDst::Uniform(ur(1)))))
        .op(under(p(3), s2r(S2RDst::Reg(r(4)))))
        .op(exit())
        .build();
    let blocks = translated(VERTEX, &code);
    assert_eq!(count(&blocks.function, Opcode::GetUniformPred), 1);
    assert_eq!(count(&blocks.function, Opcode::GetPred), 1);
    let text = blocks.function.to_string();
    assert!(
        text.starts_with("block0:\n    %0: u1 = GetUniformPred up3\n    %1: u1 = LogicalNot %0\n"),
        "{text}"
    );
    for holds in [false, true] {
        let machine = run_both(&blocks, |machine| {
            machine.uniform_preds[3] = holds;
            machine.preds[3] = holds;
            machine.uniform_regs[1] = 2;
            machine.regs[4] = 2;
            machine.system.lane_id = 6;
        });
        assert_eq!(
            machine.uniform_regs[1],
            if holds { 2 } else { 6 },
            "{holds}"
        );
        assert_eq!(machine.regs[4], if holds { 6 } else { 2 }, "{holds}");
    }
}

/// `@!PT` never runs: whatever the instruction, it becomes nothing, as PR #4588's
/// `Visitor::Run` returns at `guard.Never()`, and an `EXIT` under it does not end the program.
#[test]
fn never_guarded_instructions_are_skipped() {
    let code = Program::default()
        .op(under(NEVER, mov(0, 1)))
        .op(under(NEVER, ast(0, 0x80)))
        .op(under(NEVER, nop()))
        .op(under(NEVER, exit()))
        .op(exit())
        .build();
    let blocks = translated(VERTEX, &code);
    assert_eq!(blocks.function.to_string(), "block0:\n    return\n");
    // The same instructions under `PT` translate to something.
    let code = Program::default().op(mov(0, 1)).op(exit()).build();
    assert_ne!(
        translated(VERTEX, &code).function.to_string(),
        "block0:\n    return\n"
    );
}

/// A `BRX` goes to the target its register names, through its jump table: the addresses are the
/// table's base plus 16 bytes per instruction (slice 10).
#[test]
fn an_indirect_branch_goes_where_its_register_says() {
    let mut program = Program::default();
    program
        .item(Item::Brx { reg: 0 })
        .label("a")
        .op(mov(1, 10))
        .op(exit())
        .label("b")
        .op(mov(1, 20))
        .op(exit());
    let code = program.build();
    let base = 0x30;
    let mut tables = BranchTables::new(base);
    tables.insert(0, vec![program.at("a"), program.at("b")]);
    let blocks = translated_with(VERTEX, &code, &tables);
    for (label, wanted) in [("a", 10), ("b", 20)] {
        let address = base + program.at(label) * size();
        let machine = run_both(&blocks, |machine| machine.regs[0] = address);
        assert_eq!(machine.regs[1], wanted, "{label}");
    }
}

/// A called function is inlined at each call site and its instructions translated in each copy.
#[test]
fn calls_run_their_inlined_function() {
    let code = Program::default()
        .op(mov(0, 1))
        .item(Item::Link { dst: 20 })
        .item(Item::Call { label: "double" })
        .item(Item::Link { dst: 20 })
        .item(Item::Call { label: "double" })
        .op(exit())
        .label("double")
        .op(add_imm(0, 0, 0))
        .op(Instr::new(Op::IAdd3(OpIAdd3 {
            dst: r(0),
            overflow: [Pred::PT; 2],
            srcs: [AluSrc::reg(r(0)), AluSrc::reg(r(0)), AluSrc::reg(Gpr::RZ)],
            carry: [NOT_PT; 2],
        })))
        .item(Item::Ret { reg: 20 })
        .build();
    let blocks = translated(VERTEX, &code);
    let machine = run_both(&blocks, |_| {});
    assert_eq!(machine.regs[0], 4);
}

/// A conditional `EXIT` (a guard, or a predicate of its own) gets a block of its own whose test
/// flow analysis builds; the colour outputs are written where the program exits, with the
/// registers as they are there.
#[test]
fn a_conditional_exit_writes_its_outputs_only_where_it_exits() {
    let one = 1.0_f32.to_bits();
    let two = 2.0_f32.to_bits();
    for (guard, pred) in [(p(0), PredSrc::TRUE), (PredSrc::TRUE, p(0))] {
        let code = Program::default()
            .op(mov(0, one))
            .op(under(guard, exit_if(pred)))
            .op(mov(0, two))
            .op(exit())
            .build();
        let blocks = translated(PIXEL, &code);
        for holds in [false, true] {
            let machine = run_both(&blocks, |machine| machine.preds[0] = holds);
            assert_eq!(
                machine.frag_colors.get(&(0, 0)).copied(),
                Some(if holds { one } else { two }),
                "{guard} {pred} {holds}"
            );
        }
    }
}

/// A guarded `KILL` demotes exactly when its guard holds, and the program goes on.
#[test]
fn a_guarded_kill_demotes() {
    let code = Program::default()
        .op(under(
            p(1),
            Instr::new(Op::Kill(OpKill {
                pred: PredSrc::TRUE,
            })),
        ))
        .op(mov(0, 1.0_f32.to_bits()))
        .op(exit())
        .build();
    let blocks = translated(PIXEL, &code);
    for holds in [false, true] {
        let machine = run_both(&blocks, |machine| machine.preds[1] = holds);
        assert_eq!(machine.demoted, holds);
        assert_eq!(machine.regs[0], 1.0_f32.to_bits());
    }
}

/// Guarded instructions of the groups the interpreter can run: integer, float, attribute and
/// control, in branches, each run exactly when its guard holds; a guarded texture instruction
/// is translated into its conditional block. The control group's are a `VOTE`, which flow
/// analysis gives a block of its own behind a test of its guard (not pure), and a `VOTEU`,
/// which the driver splits on its guard read as a uniform predicate (pure and uniform), both
/// with destinations that tell whether they ran: a guard that is ignored, read twice or read as
/// the wrong kind of predicate changes them.
#[test]
fn guarded_instructions_of_every_group() {
    let code = Program::default()
        .op(ald(1, 0x80))
        .op(under(p(0), add_imm(2, 2, 5)))
        .op(under(p(1), fadd(3, 1, 1)))
        .op(under(p(2), ast(3, 0x90)))
        .op(under(p(0), vote_any(5, 4, p(1))))
        .op(under(p(3), voteu_any(6, 2, p(1))))
        .item(Item::Bra {
            guard: p(1),
            label: "skip",
        })
        .op(under(not_p(0), add_imm(4, 4, 1)))
        .label("skip")
        .op(exit())
        .build();
    let blocks = translated(VERTEX, &code);
    for bits in 0_u8..16 {
        let [p0, p1, p2, up3] = [1, 2, 4, 8].map(|bit| bits & bit != 0);
        let machine = run_both(&blocks, |machine| {
            machine.preds[0] = p0;
            machine.preds[1] = p1;
            machine.preds[2] = p2;
            // The `P3` guard of the `VOTEU` reads `UP3`; `P3` is its opposite, so reading the
            // wrong one fails.
            machine.preds[3] = !up3;
            machine.uniform_preds[3] = up3;
            machine.preds[4] = !p1;
            machine.uniform_preds[2] = !p1;
            machine.regs[3] = 7;
            machine.regs[5] = 0xdead;
            machine.uniform_regs[6] = 0xbeef;
            machine
                .attributes
                .insert((Attribute(0x20), 0), 1.25_f32.to_bits());
        });
        assert_eq!(machine.regs[2], if p0 { 5 } else { 0 }, "{bits}");
        let sum = if p1 { 2.5_f32.to_bits() } else { 7 };
        assert_eq!(machine.regs[3], sum, "{bits}");
        assert_eq!(
            machine.outputs.get(&(Attribute(0x24), 0)).copied(),
            p2.then_some(sum),
            "{bits}"
        );
        assert_eq!(machine.regs[4], u32::from(!p1 && !p0), "{bits}");
        // Lane 0's ballot is bit 0 where `P1` holds; the vote is `P1`.
        let ballot = u32::from(p1);
        assert_eq!(machine.regs[5], if p0 { ballot } else { 0xdead }, "{bits}");
        assert_eq!(machine.preds[4], if p0 { p1 } else { !p1 }, "{bits}");
        assert_eq!(
            machine.uniform_regs[6],
            if up3 { ballot } else { 0xbeef },
            "{bits}"
        );
        assert_eq!(
            machine.uniform_preds[2],
            if up3 { p1 } else { !p1 },
            "{bits}"
        );
    }
    let tex = crate::decode::OpTex {
        dsts: [r(8), r(10)],
        fault: Pred::PT,
        tex: crate::decode::TexRef::CBuf(crate::decode::TexCBufRef {
            index: 0,
            offset: 0x40,
        }),
        srcs: [r(1), r(20)],
        dim: crate::decode::TexDim::D2,
        lod_mode: crate::decode::TexLodMode::Auto,
        ndv: false,
        z_cmpr: false,
        offset: false,
        f16: false,
        eviction: MemEvictionPriority::Normal,
        nodep: false,
        channel_mask: crate::decode::ChannelMask::RGBA,
        scalar: true,
    };
    let code = Program::default()
        .op(under(p(5), Instr::new(Op::Tex(tex))))
        .op(exit())
        .build();
    let blocks = translated(PIXEL, &code);
    let text = blocks.function.to_string();
    assert!(text.contains("ImageSample"), "{text}");
    assert_eq!(count(&blocks.function, Opcode::GetPred), 1, "{text}");
}

/// Whether a structured tree holds a loop.
fn has_loop(nodes: &[Node]) -> bool {
    nodes.iter().any(|node| match node {
        Node::Loop { .. } => true,
        Node::If { body, .. } => has_loop(body),
        _ => false,
    })
}

/// The structurizer lays a conditional block out between its test and the code after it, so a
/// guarded instruction becomes an `if`, not a loop.
#[test]
fn a_guarded_instruction_structures_as_an_if() {
    let code = Program::default()
        .op(under(p(2), mov(0, 7)))
        .op(under(p(3), mov(1, 8)))
        .op(exit())
        .build();
    let structured = translated(VERTEX, &code)
        .structure(MAX_STRUCTURE_WORK)
        .expect("structures")
        .blocks
        .function;
    let tree = structured.structure().expect("a tree");
    assert!(!has_loop(tree), "{structured}");
    assert!(tree.iter().any(|node| matches!(node, Node::If { .. })));
}

/// A word with an unknown opcode or an unexplained bit is refused with the decoder's exact
/// error at its index; decoding comes before translation, so a later undecodable word is
/// reported before an earlier instruction the driver refuses; and of two undecodable words the
/// lower-numbered one is reported.
#[test]
fn the_lowest_undecodable_instruction_is_refused() {
    let unknown = Instruction::default();
    let mov_word = assemble(&mov(0, 1)).expect("encodable");
    let opcode = u16::try_from(mov_word.field(OPCODE)).expect("12 bits");
    let (_, known, _) = crate::testing::forms()
        .into_iter()
        .find(|(form, _, _)| *form == opcode)
        .expect("the form");
    let bit = (0..128)
        .find(|bit| known & 1 << bit == 0)
        .expect("an unknown bit");
    let stray = Instruction::from_bits(mov_word.bits() | 1 << bit);
    let stray_error = crate::decode::decode(stray).expect_err("a stray bit");
    assert!(matches!(stray_error, DecodeError::UnknownBit { opcode: o, .. } if o == opcode));
    let ld = Instr::new(Op::Ld(OpLd {
        dst: r(0),
        addr: MemAddr {
            reg: Gpr::RZ,
            reg_64: false,
            uniform: Some(UniformAddr {
                reg: UniformReg::URZ,
                wide: false,
                enabled: true,
            }),
            offset: 0,
        },
        mem_type: MemType::B32,
        space: MemSpace::Local(MemEvictionPriority::Normal),
        pred: PredSrc::TRUE,
        pred_dst: Pred::PT,
        zd: false,
        bit76: false,
    }));
    let ld_word = assemble(&ld).expect("encodable");
    let exit_word = assemble(&exit()).expect("encodable");
    let refusal = |code: &[Instruction]| {
        translate_flow(VERTEX, None, code, &BranchTables::new(0)).map(|blocks| blocks.function)
    };
    assert_eq!(
        refusal(&[mov_word, unknown, exit_word]),
        Err(TranslateError::Decode {
            index: 1,
            error: DecodeError::UnknownOpcode { opcode: 0 }
        })
    );
    assert_eq!(
        refusal(&[ld_word, stray, unknown, exit_word]),
        Err(TranslateError::Decode {
            index: 1,
            error: stray_error
        })
    );
    assert_eq!(
        refusal(&[ld_word, mov_word, unknown, stray, exit_word]),
        Err(TranslateError::Decode {
            index: 2,
            error: DecodeError::UnknownOpcode { opcode: 0 }
        })
    );
    // Under `!PT` too: the word is decoded before its guard is read.
    let never = unknown
        .with_field(GUARD, u64::from(Pred::PT.index()))
        .with_flag(crate::decode::GUARD_NOT, true);
    assert_eq!(
        refusal(&[never, exit_word]),
        Err(TranslateError::Decode {
            index: 0,
            error: DecodeError::UnknownOpcode { opcode: 0 }
        })
    );
    // With every word decodable, the memory translator refuses the `LDL` with its uniform
    // address on (`ops_memory.cpp`, `LDL`: `Require(!use_uniform)`).
    assert_eq!(
        refusal(&[ld_word, exit_word]),
        Err(TranslateError::Memory {
            index: 0,
            instr: ld,
            error: MemoryError::Unsupported(LDL_UNIFORM),
        })
    );
}

/// What slice 12e's translator refuses an `LDL` with its uniform address on as.
const LDL_UNIFORM: crate::translate::memory::MemoryUnsupported =
    crate::translate::memory::MemoryUnsupported::UniformAddress(
        crate::translate::memory::Space::Local,
    );

/// A memory instruction reaches slice 12e's translator through the driver: an `LDS` of a word
/// from a register plus an offset becomes the translator's `LoadSharedU32` at the address
/// `Visitor::SharedOffset` makes (exact golden IR), in a compute program.
#[test]
fn a_memory_instruction_reaches_the_memory_translator() {
    let lds = Instr::new(Op::Ld(OpLd {
        dst: r(2),
        addr: MemAddr {
            reg: r(1),
            reg_64: false,
            uniform: Some(UniformAddr {
                reg: UniformReg::new(0).expect("a uniform register"),
                wide: false,
                enabled: false,
            }),
            offset: 0x10,
        },
        mem_type: MemType::B32,
        space: MemSpace::Shared(OffsetStride::X1),
        pred: PredSrc::TRUE,
        pred_dst: Pred::PT,
        zd: false,
        bit76: false,
    }));
    let code = Program::default()
        .op(mov(1, 0x20))
        .op(lds)
        .op(exit())
        .build();
    let compute = Stage {
        stage: GuestStage::Compute,
        ..VERTEX
    };
    let blocks = translated(compute, &code);
    assert_eq!(
        blocks.function.to_string(),
        "block0:\n\
         \x20   %0 = SetRegister r1, u32 0x20\n\
         \x20   %1: u32 = GetRegister r1\n\
         \x20   %2: u32 = IAdd32 %1, u32 0x10\n\
         \x20   %3: u32 = LoadSharedU32 %2\n\
         \x20   %4 = SetRegister r2, %3\n\
         \x20   return\n"
    );
}

/// A memory form slice 12e's translator refuses is refused through the driver with its error,
/// the exact message naming the instruction.
#[test]
fn a_refused_memory_instruction_names_the_memory_translator_s_reason() {
    let mut program = vec![];
    let ld = OpLd {
        dst: r(0),
        addr: MemAddr {
            reg: Gpr::RZ,
            reg_64: false,
            uniform: Some(UniformAddr {
                reg: UniformReg::URZ,
                wide: false,
                enabled: true,
            }),
            offset: 0,
        },
        mem_type: MemType::B32,
        space: MemSpace::Local(MemEvictionPriority::Normal),
        pred: PredSrc::TRUE,
        pred_dst: Pred::PT,
        zd: false,
        bit76: false,
    };
    program.push(Instr::new(Op::Ld(ld)));
    program.push(exit());
    let error = translate_instrs(VERTEX, None, &program).expect_err("refused");
    assert_eq!(
        error,
        TranslateError::Memory {
            index: 0,
            instr: program[0],
            error: MemoryError::Unsupported(LDL_UNIFORM),
        }
    );
    assert_eq!(
        error.to_string(),
        "instruction 0 (ldl r0, [rz];): a uniform register in a local address is not supported"
    );
}

/// A memory instruction dispatched into a block that does not exist is the builder's error,
/// reported as [`TranslateError::Build`] as every group's is, not as
/// [`TranslateError::Memory`] (the dispatch's `MemoryError::Build` arm).
#[test]
fn a_memory_instruction_into_a_missing_block_is_the_builder_s_error() {
    let mut function = Function::new();
    let missing = {
        let mut other = Function::new();
        other.add_block().expect("a block")
    };
    let lds = Instr::new(Op::Ld(OpLd {
        dst: r(2),
        addr: MemAddr {
            reg: r(1),
            reg_64: false,
            uniform: Some(UniformAddr {
                reg: UniformReg::new(0).expect("a uniform register"),
                wide: false,
                enabled: false,
            }),
            offset: 0x10,
        },
        mem_type: MemType::B32,
        space: MemSpace::Shared(OffsetStride::X1),
        pred: PredSrc::TRUE,
        pred_dst: Pred::PT,
        zd: false,
        bit76: false,
    }));
    let mut driver = Driver::new(VERTEX, None);
    assert_eq!(
        driver.dispatch(&mut function, missing, 2, lds),
        Err(TranslateError::Build(BuildError::NoBlock(missing)))
    );
}

/// An operation no group takes (a tensor shell, which no word decodes to yet) is refused by its
/// mnemonic; the dispatch is reached directly, as no program can hold one.
#[test]
fn an_instruction_no_group_takes_is_refused_by_its_mnemonic() {
    let mut function = Function::new();
    let entry = function.entry();
    let instr = Instr::new(Op::Hmma(OpHmma {}));
    let mut driver = Driver::new(VERTEX, None);
    assert_eq!(
        driver.dispatch(&mut function, entry, 3, instr),
        Err(TranslateError::Unsupported {
            index: 3,
            instr,
            what: Unsupported::Instruction("hmma"),
        })
    );
}

/// A decoded operation of the half and double group reaches its translator (slice 12c): one of
/// each operation that [`decoded_samples`] reaches is translated or refused with the group's
/// own rejection, never by its mnemonic as an instruction no group takes.
#[test]
fn a_decoded_half_double_instruction_reaches_its_group() {
    let mut seen = std::collections::BTreeSet::new();
    for instr in decoded_samples() {
        if !half_double::in_group(&instr.op) || !seen.insert(instr.op.mnemonic()) {
            continue;
        }
        let mut function = Function::new();
        let entry = function.entry();
        let mut driver = Driver::new(VERTEX, None);
        let expected = {
            let mut scratch = Function::new();
            let at = scratch.entry();
            half_double::translate_instr(&mut Builder::new(&mut scratch, at), instr, true).map_err(
                |error| match error {
                    HalfDoubleError::Unsupported(what) => TranslateError::Unsupported {
                        index: 5,
                        instr,
                        what: Unsupported::HalfDouble(what),
                    },
                    HalfDoubleError::Build(error) => TranslateError::Build(error),
                },
            )
        };
        assert_eq!(
            driver.dispatch(&mut function, entry, 5, instr),
            expected,
            "{instr}"
        );
        assert_ne!(
            expected,
            Err(TranslateError::Unsupported {
                index: 5,
                instr,
                what: Unsupported::Instruction(instr.op.mnemonic()),
            }),
            "{instr}"
        );
    }
    assert_eq!(seen.len(), 10, "{seen:?}");
}

/// Every decoded operation of the uniform datapath (slice 6d) reaches slice 12d's translator:
/// each sample [`decoded_samples`] reaches translates or is refused with that translator's
/// reason ([`TranslateError::Uniform`]), never by the dispatch's fallback; between them every
/// decoded one of the group's 20 operations (all but the shell `UP2UR`) is reached. The samples
/// are canonical words and their one-bit changes, whose predicate fields are mostly `P0`, so
/// the forms PR #4588 requires `!PT` carries of translate in no sample (the uniform group's
/// `tests::every_decoded_form_translates_or_is_refused`); `UBMSK`'s refusal names it.
#[test]
fn a_decoded_uniform_datapath_instruction_reaches_its_group() {
    let mut seen = std::collections::BTreeSet::new();
    let mut translated = std::collections::BTreeSet::new();
    for instr in decoded_samples() {
        if !uniform::in_group(&instr.op) {
            continue;
        }
        seen.insert(instr.op.mnemonic());
        let mut function = Function::new();
        let entry = function.entry();
        let mut driver = Driver::new(VERTEX, None);
        let result = driver.dispatch(&mut function, entry, 5, instr);
        assert!(
            matches!(
                result,
                Ok(()) | Err(TranslateError::Uniform { index: 5, .. })
            ),
            "{instr}: {result:?}"
        );
        if result.is_ok() {
            translated.insert(instr.op.mnemonic());
        }
        if matches!(instr.op, Op::UBMsk(_)) {
            assert_eq!(
                result,
                Err(TranslateError::Uniform {
                    index: 5,
                    instr,
                    error: UniformError::Unsupported(uniform::UniformUnsupported::Instruction(
                        "ubmsk"
                    )),
                })
            );
        }
    }
    assert_eq!(seen.len(), 20, "{seen:?}");
    assert_eq!(translated.len(), 15, "{translated:?}");
    assert!(!translated.contains("ubmsk"));
}

/// A refusal of the uniform datapath's translator reaches the program's error with the
/// instruction's index and its reason in plain words.
#[test]
fn a_refused_uniform_instruction_names_its_reason() {
    use crate::decode::{OpR2UR, OpULdc, ULdcSrc};
    let ldcg = Instr::new(Op::ULdc(OpULdc {
        dst: ur(4),
        src: ULdcSrc::Global {
            addr: ur(2),
            offset: 0,
            pred: crate::decode::UniformPredSrc {
                reg: UniformPred::UPT,
                negated: true,
            },
        },
        mem_type: MemType::B128,
    }));
    let r2ur = Instr::new(Op::R2UR(OpR2UR {
        dst: UniformReg::URZ,
        src: r(1),
    }));
    for (instr, text) in [
        (
            ldcg,
            "instruction 1 (uldc.128 ur4, [ur2], !upt;): a 128-bit ldcg is not supported",
        ),
        (
            r2ur,
            "instruction 1 (r2ur urz, r1;): an r2ur into urz is not supported",
        ),
    ] {
        let code = [
            assemble(&mov(0, 1)).expect("encodable"),
            assemble(&instr).expect("encodable"),
            assemble(&exit()).expect("encodable"),
        ];
        let error =
            translate_flow(VERTEX, None, &code, &BranchTables::new(0)).expect_err("refused");
        assert!(
            matches!(error, TranslateError::Uniform { index: 1, .. }),
            "{error}"
        );
        assert_eq!(error.to_string(), text);
    }
}

/// The translation budget: a count of exactly the cap is accepted, one past it refused, whether
/// the count comes from one block or several; and the production cap is four per instruction
/// of a program at the capture's cap.
#[test]
fn the_translation_budget_at_its_edge() {
    let cap = 10;
    assert_eq!(charge([10].into_iter(), cap), Ok(10));
    assert_eq!(
        charge([11].into_iter(), cap),
        Err(TranslateError::TooManyInstructions)
    );
    assert_eq!(charge([4, 6, 0].into_iter(), cap), Ok(10));
    assert_eq!(
        charge([4, 6, 1].into_iter(), cap),
        Err(TranslateError::TooManyInstructions)
    );
    assert_eq!(
        charge([usize::MAX, 1].into_iter(), u32::MAX),
        Err(TranslateError::TooManyInstructions)
    );
    assert_eq!(charge(std::iter::empty(), 0), Ok(0));
    assert_eq!(MAX_INSTRUCTIONS, 65_536);
    assert_eq!(MAX_TRANSLATED_INSTRUCTIONS, 262_144);
}

/// A program whose calls, inlined, hold more instructions than the budget is refused before
/// anything is translated; the same function called fewer times translates.
#[test]
fn an_inlined_program_past_the_budget_is_refused() {
    const BODY: u32 = 600;
    let program = |calls: u32| {
        let mut program = Program::default();
        for _ in 0..calls {
            program
                .item(Item::Link { dst: 20 })
                .item(Item::Call { label: "f" });
        }
        program.op(exit()).label("f");
        for _ in 0..BODY {
            program.op(nop());
        }
        program.item(Item::Ret { reg: 20 }).build()
    };
    let budget = u64::from(MAX_TRANSLATED_INSTRUCTIONS);
    let over = u32::try_from(budget / u64::from(BODY) + 1).expect("fits");
    let code = program(over);
    let total: u64 = flow_graph(&code, &BranchTables::new(0))
        .expect("a graph")
        .blocks()
        .iter()
        .map(|block| u64::try_from(block.code.len()).expect("fits"))
        .sum();
    assert!(total > budget, "{total}");
    assert_eq!(
        translate_flow(VERTEX, None, &code, &BranchTables::new(0)).map(|_| ()),
        Err(TranslateError::TooManyInstructions)
    );
    assert!(translate_flow(VERTEX, None, &program(4), &BranchTables::new(0)).is_ok());
}

/// The stages the lowering takes (vertex, pixel and compute) translate; every other stage is
/// refused by name, each stage listed by hand.
#[test]
fn only_the_stages_the_lowering_takes_translate() {
    let code = Program::default().op(exit()).build();
    for stage in [
        GuestStage::VertexCullBeforeFetch,
        GuestStage::TessellationInit,
        GuestStage::Tessellation,
        GuestStage::Geometry,
    ] {
        assert!(!translates_stage(stage), "{stage}");
        let stage = Stage { stage, ..VERTEX };
        assert_eq!(
            translate_flow(stage, None, &code, &BranchTables::new(0)).map(|_| ()),
            Err(TranslateError::Stage(Unsupported::Stage(stage.stage)))
        );
    }
    let compute = Stage {
        stage: GuestStage::Compute,
        ..VERTEX
    };
    for stage in [VERTEX, PIXEL, compute] {
        assert!(translates_stage(stage.stage), "{}", stage.stage);
        assert!(translate_flow(stage, None, &code, &BranchTables::new(0)).is_ok());
    }
}

/// A program with control flow runs through the whole chain: the driver, the structurizer and
/// the lowering (`program::translate_program`).
#[test]
fn a_program_with_control_flow_lowers() {
    let code = Program::default()
        .op(under(p(0), mov(0, 7)))
        .op(exit())
        .build();
    let bytes: Vec<u8> = code.iter().flat_map(|word| word.to_le_bytes()).collect();
    assert!(crate::program::translate_program(VERTEX, &bytes).is_ok());
}

/// Which of the driver's groups take `op`, as names (`super`'s `GROUPS` for the attribute group,
/// tried on a scratch function).
fn groups_of(op: Op) -> Vec<&'static str> {
    let mut function = Function::new();
    let entry = function.entry();
    let mut t = Translator {
        b: Builder::new(&mut function, entry),
        stage: VERTEX,
        header: None,
    };
    [
        ("exit", matches!(op, Op::Exit(_))),
        ("integer", integer::is_integer(&op)),
        ("float", float::in_group(&op)),
        ("half and double", half_double::in_group(&op)),
        ("texture", texture::in_group(&op)),
        ("control", control::in_group(&op)),
        ("memory", memory::is_memory(&op)),
        ("uniform", uniform::in_group(&op)),
        ("attribute", attribute::translate(&mut t, op).is_some()),
    ]
    .into_iter()
    .filter_map(|(name, takes)| takes.then_some(name))
    .collect()
}

/// The shells: operations no word decodes to yet, each built by hand (a field added to one
/// stops this compiling), as PR #4588's tables name them: the uniform group's `UP2UR`, the
/// control group's `JMX` and `BPT`, and the tensor group.
fn shells() -> Vec<Op> {
    use crate::decode::{OpBpt, OpImma, OpJmx, OpLdsm, OpMovm, OpUP2UR};
    vec![
        Op::UP2UR(OpUP2UR {}),
        Op::Jmx(OpJmx {}),
        Op::Bpt(OpBpt {}),
        Op::Hmma(OpHmma {}),
        Op::Imma(OpImma {}),
        Op::Ldsm(OpLdsm {}),
        Op::Movm(OpMovm {}),
    ]
}

/// Decoded instructions that reach every operation a word decodes to, deterministically: each
/// form's canonical word ([`canonical_word`]) and each word that differs from it in one of the
/// form's known bits, where it decodes. Some operations are picked by a flag within a form
/// (`IADD3.X`, `LEA.X`, `OUT.FINAL`), which the one-bit changes reach;
/// `these_instructions_reach_every_decoded_operation` checks that every operation but the
/// [`shells`] is reached. Shared with the control group's tests.
pub(crate) fn decoded_samples() -> Vec<Instr> {
    let mut samples = vec![];
    for (opcode, known, _) in forms() {
        let canonical = canonical_word(opcode);
        let flips = (0..128_u32).filter(|bit| known >> bit & 1 != 0).map(Some);
        for bit in std::iter::once(None).chain(flips) {
            let word = bit.map_or(canonical, |bit| {
                Instruction::from_bits(canonical.bits() ^ 1 << bit)
            });
            if let Ok(instr) = crate::decode::decode(word) {
                samples.push(instr);
            }
        }
    }
    samples
}

/// [`decoded_samples`] reaches every operation but the [`shells`], which it does not reach:
/// between them they are every operation, each once.
#[test]
fn these_instructions_reach_every_decoded_operation() {
    let reached: std::collections::BTreeSet<&str> = decoded_samples()
        .iter()
        .map(|instr| instr.op.mnemonic())
        .collect();
    let shells: std::collections::BTreeSet<&str> = shells().iter().map(Op::mnemonic).collect();
    assert_eq!(shells.len(), 7);
    assert!(reached.is_disjoint(&shells), "{reached:?}");
    let all: std::collections::BTreeSet<&str> = crate::decode::MNEMONICS.iter().copied().collect();
    assert_eq!(all.len(), crate::decode::MNEMONICS.len());
    assert_eq!(
        reached
            .union(&shells)
            .copied()
            .collect::<std::collections::BTreeSet<_>>(),
        all
    );
}

/// The groups are disjoint, and every decoded operation (every one of [`decoded_samples`])
/// belongs to exactly one of them, so the dispatch order decides nothing; a shell belongs to
/// none but the control group's `JMX` and `BPT` and the uniform group's `UP2UR`, which they
/// refuse by name, so the fallback refusal is reached only by the tensor group's shells.
#[test]
fn the_groups_are_disjoint() {
    let mut half_double = 0;
    let mut uniform = 0;
    for instr in decoded_samples() {
        if half_double::in_group(&instr.op) {
            half_double += 1;
        }
        if uniform::in_group(&instr.op) {
            uniform += 1;
        }
        assert_eq!(groups_of(instr.op).len(), 1, "{instr}");
    }
    assert!(half_double > 0);
    assert!(uniform > 0);
    for op in shells() {
        let wanted: &[&str] = if matches!(op, Op::Jmx(_) | Op::Bpt(_)) {
            &["control"]
        } else if matches!(op, Op::UP2UR(_)) {
            &["uniform"]
        } else {
            &[]
        };
        assert_eq!(groups_of(op), wanted, "{op}");
    }
}

/// A guarded instruction that is not pure and that flow analysis leaves in its block
/// (`classify.h`'s `OpKind::Nop`: `NOP`, `DEPBAR`, `BREAK`, `WARPSYNC` with an immediate) is
/// translated as is: its guard is not read and no block is made, as PR #4588's `Visitor::Run`
/// reads a guard only for a pure instruction (`visitor.cpp`: `info.pure && !guard.Always()`).
#[test]
fn a_guarded_instruction_that_is_not_pure_reads_no_guard() {
    let depbar = Instr::new(Op::DepBar(OpDepBar {
        scoreboard: DEPBAR_REQUIRED_SCOREBOARD,
        le: true,
        count: 0,
    }));
    let warpsync = Instr::new(Op::WarpSync(OpWarpSync {
        mask: AluSrc::Imm(FULL_WARP_MASK),
        cond: PredSrc::TRUE,
    }));
    let words = [nop(), depbar, warpsync]
        .map(|instr| assemble(&instr).expect("encodable"))
        .into_iter()
        .chain([canonical_word(BREAK_OPCODE)]);
    for word in words {
        assert_eq!(classify(opcode_of(word)).kind, crate::flow::OpKind::Nop);
        for negated in [false, true] {
            let guarded = word
                .with_field(GUARD, 3)
                .with_flag(crate::decode::GUARD_NOT, negated);
            let code = [guarded, assemble(&exit()).expect("encodable")];
            let graph = flow_graph(&code, &BranchTables::new(0)).expect("a graph");
            assert!(
                graph.blocks().iter().all(|block| !block.guard_tested),
                "{word:?}"
            );
            let blocks = translated(VERTEX, &code);
            assert_eq!(
                blocks.function.to_string(),
                "block0:\n    return\n",
                "{negated} {word:?}"
            );
        }
    }
}

/// The driver's rule (`super::super::guard_step`) settles a guard on either of its two inputs:
/// a pure instruction (`MOV`) handed over in a block whose guard is tested is translated as if
/// unguarded, with no read and no split, as is one that is not pure (`AST`) in a block whose
/// guard is not tested; only a pure one in an untested block splits. Flow analysis gives a
/// tested block only to an instruction that is not pure, so a whole program cannot reach the
/// first case: this drives [`Driver::instruction`] directly.
#[test]
fn either_a_tested_block_or_an_impure_instruction_settles_the_guard() {
    let cases = [
        (under(p(2), mov(0, 7)), true, false),
        (under(p(2), ast(0, 0x80)), false, false),
        (under(p(2), mov(0, 7)), false, true),
    ];
    for (instr, guard_tested, splits) in cases {
        let info = classify(opcode_of(assemble(&instr).expect("encodable")));
        let mut function = Function::new();
        let entry = function.entry();
        let after = Driver::new(VERTEX, None)
            .instruction(&mut function, entry, 0, instr, info, guard_tested)
            .expect("translates");
        assert_eq!(after != entry, splits, "{instr} {guard_tested}");
        assert_eq!(
            count(&function, Opcode::GetPred),
            usize::from(splits),
            "{instr} {guard_tested}"
        );
    }
}

/// The guard of every decodable pure form (`classify.h`'s `pure`) is read before the split:
/// as the uniform predicate of the same number for a uniform one (of the decodable forms
/// `S2UR`, `0x9c3`, `VOTEU`, `0x886`, and slice 6d's `UMOV`, `ULDC`, `UPLOP3` and the twins
/// `classify.h` names pure: the list in the test), as the predicate for the others, which come
/// from the integer, float, half and double, attribute and memory groups, and `R2UR`, the
/// uniform group's, which reads its guard as data with no split (`classify.h`: "reads the guard
/// as data, so it is not split into a block"). The instruction itself may be refused after (a
/// canonical word is not always a form the group translates); the guard's read comes first.
#[test]
fn the_guard_of_every_pure_form_is_read_by_its_kind() {
    let mut uniform = vec![];
    let mut groups = std::collections::BTreeSet::new();
    for (opcode, _, _) in forms() {
        let info = classify(opcode);
        if !info.pure {
            continue;
        }
        let word = canonical_word(opcode)
            .with_field(GUARD, 3)
            .with_flag(crate::decode::GUARD_NOT, true);
        let instr = crate::decode::decode(word).expect("a canonical word decodes");
        let mut function = Function::new();
        let entry = function.entry();
        let outcome =
            Driver::new(VERTEX, None).instruction(&mut function, entry, 0, instr, info, false);
        assert!(
            !matches!(outcome, Err(TranslateError::Build(_))),
            "{instr}: {outcome:?}"
        );
        let read = if info.uniform {
            uniform.push(opcode);
            "GetUniformPred up3"
        } else {
            groups.extend(groups_of(instr.op));
            "GetPred p3"
        };
        let text = function.to_string();
        if matches!(instr.op, Op::R2UR(_)) {
            // `R2UR` reads its guard as data, the ballot's input, and makes no block.
            assert!(
                text.starts_with(
                    "block0:\n    %0: u1 = GetPred p3\n    %1: u1 = LogicalNot %0\n    \
                     %2: u32 = SubgroupBallot %1\n"
                ),
                "{instr}: {text}"
            );
            continue;
        }
        assert!(
            text.starts_with(&format!(
                "block0:\n    %0: u1 = {read}\n    %1: u1 = LogicalNot %0\n    branch %1, block1, block2\n"
            )),
            "{instr}: {text}"
        );
    }
    // `classify.h`'s uniform pure opcodes, as literals: `S2UR`, `VOTEU`, "UMOV imm", "UMOV UR",
    // "ULDC", "ULDC.64", "UPLOP3", and the twelve "Uniform ALU" bases in forms 1 (`0x200`) and
    // 4 (`0x800`), each of which slice 6d decodes.
    let mut expected = vec![0x9c3_u16, 0x886, 0x882, 0xc82, 0xab9, 0xabb, 0x89c];
    for base in [
        0x090_u16, 0x091, 0x092, 0x099, 0x0a4, 0x0a5, 0x0be, 0x09a, 0x0bd, 0x0bf, 0x08c, 0x087,
    ] {
        expected.extend([base | 0x200, base | 0x800]);
    }
    expected.sort_unstable();
    uniform.sort_unstable();
    assert_eq!(uniform, expected);
    assert_eq!(
        groups.into_iter().collect::<Vec<_>>(),
        [
            "attribute",
            "float",
            "half and double",
            "integer",
            "memory",
            "uniform"
        ]
    );
}

/// A guarded `VOTE` (not pure: flow analysis's own block, `guard_tested`) and a guarded `VOTEU`
/// (pure and uniform: the driver's split) each read their guard once, and run exactly where it
/// holds: the control group is handed both as already guarded.
#[test]
fn guarded_votes_read_their_guard_once() {
    let code = Program::default()
        .op(under(p(0), vote_any(5, 4, p(1))))
        .op(under(p(3), voteu_any(6, 2, p(1))))
        .op(exit())
        .build();
    let blocks = translated(VERTEX, &code);
    assert_eq!(
        reads(&blocks.function, "GetPred p0"),
        1,
        "{}",
        blocks.function
    );
    assert_eq!(
        reads(&blocks.function, "GetUniformPred up3"),
        1,
        "{}",
        blocks.function
    );
    assert_eq!(count(&blocks.function, Opcode::SelectU32), 0);
    for bits in 0_u8..4 {
        let [p0, up3] = [1, 2].map(|bit| bits & bit != 0);
        let machine = run_both(&blocks, |machine| {
            machine.preds[0] = p0;
            machine.uniform_preds[3] = up3;
            machine.preds[1] = true;
            machine.regs[5] = 0xdead;
            machine.uniform_regs[6] = 0xbeef;
        });
        assert_eq!(machine.regs[5], if p0 { 1 } else { 0xdead }, "{bits}");
        assert_eq!(machine.preds[4], p0, "{bits}");
        assert_eq!(
            machine.uniform_regs[6],
            if up3 { 1 } else { 0xbeef },
            "{bits}"
        );
        assert_eq!(machine.uniform_preds[2], up3, "{bits}");
    }
}

/// A guarded `FADD` is one conditional block: the driver reads the guard once and the float
/// group, handed it as tested, does not split again.
#[test]
fn a_guarded_float_instruction_is_one_conditional_block() {
    let code = Program::default()
        .op(under(p(1), fadd(3, 1, 1)))
        .op(exit())
        .build();
    let blocks = translated(VERTEX, &code);
    assert_eq!(count(&blocks.function, Opcode::GetPred), 1);
    assert_eq!(
        blocks.function.to_string(),
        "block0:\n\
         \x20   %0: u1 = GetPred p1\n\
         \x20   branch %0, block1, block2\n\
         block1:\n\
         \x20   %1: u32 = GetRegister r1\n\
         \x20   %2: f32 = BitCastF32U32 %1\n\
         \x20   %3: u32 = GetRegister r1\n\
         \x20   %4: f32 = BitCastF32U32 %3\n\
         \x20   %5: f32 = FPAdd32 %2, %4 {no_contraction, rn, fmz_none}\n\
         \x20   %6: u32 = BitCastU32F32 %5\n\
         \x20   %7 = SetRegister r3, %6\n\
         \x20   jump block2\n\
         block2:\n\
         \x20   return\n"
    );
}

/// The stage is checked before flow analysis: a geometry program with no instructions is
/// refused for its stage, though flow analysis would refuse its size.
#[test]
fn the_stage_is_checked_before_flow_analysis() {
    let geometry = Stage {
        stage: GuestStage::Geometry,
        ..VERTEX
    };
    assert_eq!(
        translate_flow(VERTEX, None, &[], &BranchTables::new(0)).map(|_| ()),
        Err(TranslateError::Flow(FlowError::ProgramSize { len: 0 }))
    );
    assert_eq!(
        translate_flow(geometry, None, &[], &BranchTables::new(0)).map(|_| ()),
        Err(TranslateError::Stage(Unsupported::Stage(
            GuestStage::Geometry
        )))
    );
}

/// Flow analysis comes before decoding: a program whose only instruction does not decode and
/// whose control leaves the program after it is refused by flow analysis; with an `EXIT` after
/// it, the same instruction is refused by the decoder.
#[test]
fn flow_analysis_is_checked_before_decoding() {
    let unknown = Instruction::default();
    let exit_word = assemble(&exit()).expect("encodable");
    let refusal = |code: &[Instruction]| {
        translate_flow(VERTEX, None, code, &BranchTables::new(0)).map(|_| ())
    };
    assert_eq!(
        refusal(&[unknown, exit_word]),
        Err(TranslateError::Decode {
            index: 0,
            error: DecodeError::UnknownOpcode { opcode: 0 }
        })
    );
    assert_eq!(
        refusal(&[unknown]),
        Err(TranslateError::Flow(FlowError::TargetOutside {
            index: 0,
            target: 1
        }))
    );
}

/// The budget is charged before anything is decoded: an inlined program past the budget whose
/// called function holds an instruction that does not decode is refused for its size; called
/// fewer times, the same program is refused for that instruction.
#[test]
fn the_budget_is_checked_before_decoding() {
    const BODY: u32 = 600;
    let unknown = Instruction::default();
    let program = |calls: u32| {
        let mut program = Program::default();
        for _ in 0..calls {
            program
                .item(Item::Link { dst: 20 })
                .item(Item::Call { label: "f" });
        }
        program.op(exit()).label("f");
        program.item(Item::Word(unknown));
        for _ in 1..BODY {
            program.op(nop());
        }
        let at = program.at("f");
        (program.item(Item::Ret { reg: 20 }).build(), at)
    };
    let over =
        u32::try_from(u64::from(MAX_TRANSLATED_INSTRUCTIONS) / u64::from(BODY) + 1).expect("fits");
    let (code, _) = program(over);
    assert_eq!(
        translate_flow(VERTEX, None, &code, &BranchTables::new(0)).map(|_| ()),
        Err(TranslateError::TooManyInstructions)
    );
    let (code, at) = program(4);
    assert_eq!(
        translate_flow(VERTEX, None, &code, &BranchTables::new(0)).map(|_| ()),
        Err(TranslateError::Decode {
            index: usize::try_from(at).expect("fits"),
            error: DecodeError::UnknownOpcode { opcode: 0 }
        })
    );
}

/// An instruction a flow block holds that was not decoded, or whose index is past the program,
/// is refused as a bug in the driver, with its own message.
#[test]
fn an_instruction_that_was_not_decoded_is_named() {
    let word = assemble(&mov(0, 1)).expect("encodable");
    let code = [word, word];
    let decoded = [Some(mov(0, 1)), None];
    assert_eq!(held(&code, &decoded, 0), Ok((0, word, mov(0, 1))));
    for index in [1, 2, u32::MAX] {
        assert_eq!(
            held(&code, &decoded, index),
            Err(TranslateError::NotDecoded { index })
        );
    }
    assert_eq!(
        TranslateError::NotDecoded { index: 1 }.to_string(),
        "instruction 1, which a flow block holds, was not decoded (a bug in the translation \
         driver)"
    );
}
