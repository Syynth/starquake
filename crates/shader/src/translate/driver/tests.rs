//! The translation driver's tests: whole programs, hand-assembled with the test assembler, with
//! branches, a loop, an indirect branch, calls and guarded instructions, run through the IR
//! interpreter (slice 7) as the control-flow graph the driver fills and as the structured
//! function the structurizer makes of it; the order of the checks with two faults at once; the
//! translation budget at its edge; the dispatch's refusals by name.

use std::collections::BTreeMap;

use proptest::prelude::*;

use super::*;
use crate::asm::assemble;
use crate::capture::MAX_INSTRUCTIONS;
use crate::decode::{
    AluSrc, BraCond, BranchOffset, BrxSrc, DecodeError, GUARD, Gpr, IntCmpOp, MemAddr,
    MemEvictionPriority, MemSpace, MemType, MovSrc, OPCODE, OpALd, OpASt, OpBra, OpBrx, OpCall,
    OpExit, OpFAdd, OpHmma, OpIAdd3, OpISetP, OpKill, OpLd, OpMov, OpNop, OpRet, OpS2R, Pred,
    PredSetOp, PredSrc, RoundMode, S2RDst, SpecialReg, UniformAddr, UniformReg,
};
use crate::interp::Machine;
use crate::ir::{Attribute, Inst, Node, Opcode};
use crate::structure::MAX_STRUCTURE_WORK;
use crate::testing::{any_instr, config};
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
/// is translated into its conditional block.
#[test]
fn guarded_instructions_of_every_group() {
    let code = Program::default()
        .op(ald(1, 0x80))
        .op(under(p(0), add_imm(2, 2, 5)))
        .op(under(p(1), fadd(3, 1, 1)))
        .op(under(p(2), ast(3, 0x90)))
        .op(under(p(0), nop()))
        .item(Item::Bra {
            guard: p(1),
            label: "skip",
        })
        .op(under(not_p(0), add_imm(4, 4, 1)))
        .label("skip")
        .op(exit())
        .build();
    let blocks = translated(VERTEX, &code);
    for bits in 0_u8..8 {
        let machine = run_both(&blocks, |machine| {
            machine.preds[0] = bits & 1 != 0;
            machine.preds[1] = bits & 2 != 0;
            machine.preds[2] = bits & 4 != 0;
            machine.regs[3] = 7;
            machine
                .attributes
                .insert((Attribute(0x20), 0), 1.25_f32.to_bits());
        });
        let [p0, p1, p2] = [bits & 1 != 0, bits & 2 != 0, bits & 4 != 0];
        assert_eq!(machine.regs[2], if p0 { 5 } else { 0 }, "{bits}");
        let sum = if p1 { 2.5_f32.to_bits() } else { 7 };
        assert_eq!(machine.regs[3], sum, "{bits}");
        assert_eq!(
            machine.outputs.get(&(Attribute(0x24), 0)).copied(),
            p2.then_some(sum),
            "{bits}"
        );
        assert_eq!(machine.regs[4], u32::from(!p1 && !p0), "{bits}");
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
    fn has_loop(nodes: &[Node]) -> bool {
        nodes.iter().any(|node| match node {
            Node::Loop { .. } => true,
            Node::If { body, .. } => has_loop(body),
            _ => false,
        })
    }
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
    // With every word decodable, the memory instruction is refused by name.
    assert_eq!(
        refusal(&[ld_word, exit_word]),
        Err(TranslateError::Unsupported {
            index: 0,
            instr: ld,
            what: Unsupported::MemoryGroup(ld.op.mnemonic()),
        })
    );
}

/// The memory group, whose translator is not joined yet, is refused by name with the exact
/// message.
#[test]
fn memory_instructions_are_refused_by_name() {
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
        TranslateError::Unsupported {
            index: 0,
            instr: program[0],
            what: Unsupported::MemoryGroup("ld"),
        }
    );
    assert_eq!(
        error.to_string(),
        "instruction 0 (ldl r0, [rz];): ld, of the memory group, which is not translated yet, is \
         not supported"
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

#[test]
fn only_vertex_and_pixel_stages_translate() {
    let code = Program::default().op(exit()).build();
    for stage in [
        GuestStage::VertexCullBeforeFetch,
        GuestStage::TessellationInit,
        GuestStage::Tessellation,
        GuestStage::Geometry,
        GuestStage::Compute,
    ] {
        let stage = Stage { stage, ..VERTEX };
        assert_eq!(
            translate_flow(stage, None, &code, &BranchTables::new(0)).map(|_| ()),
            Err(TranslateError::Stage(Unsupported::Stage(stage.stage)))
        );
    }
    for stage in [VERTEX, PIXEL] {
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
        ("texture", texture::in_group(&op)),
        ("control", control::in_group(&op)),
        ("memory", in_memory_group(&op)),
        ("attribute", attribute::translate(&mut t, op).is_some()),
    ]
    .into_iter()
    .filter_map(|(name, takes)| takes.then_some(name))
    .collect()
}

proptest! {
    #![proptest_config(config())]

    /// The groups are disjoint, and every decoded operation belongs to one of them, so the
    /// dispatch order decides nothing and the fallback refusal is reached only by operations no
    /// word decodes to yet.
    #[test]
    fn the_groups_are_disjoint(instr in any_instr()) {
        prop_assert_eq!(groups_of(instr.op).len(), 1, "{}", instr);
    }
}
