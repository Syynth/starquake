//! Generated control-flow graphs (proptest, fixed seed): every one structures, the verifier
//! accepts the result, and the graph and its structured form run the same on the same inputs.
//!
//! A generated graph has up to [`MAX_BLOCKS`] blocks. Each block counts its visits in r0, folds
//! its number into a hash of the path in r1, may demote, and ends in a `return`, an
//! `unreachable`, a jump forward, or a branch whose `then` goes to any block (itself, an
//! earlier one or the middle of a loop, so irreducible graphs come up) and whose `otherwise`
//! goes forward. A computed condition is a bit of the path hash while the visit count is below
//! [`FUEL`], so every run ends: once the fuel is spent, control only goes forward.

use proptest::prelude::*;

use super::same_run;
use crate::interp::Machine;
use crate::ir::{BlockId, Builder, Function, Imm, Opcode, Operand, Terminator};
use crate::structure::{MAX_STRUCTURE_WORK, structure};
use crate::testing::config;

/// The most blocks a generated graph has.
const MAX_BLOCKS: usize = 12;

/// The visits after which every computed condition is false.
const FUEL: u32 = 40;

/// The multiplier of the path hash.
const HASH: u32 = 0x9e37_79b1;

/// How a generated block's branch decides.
#[derive(Clone, Copy, Debug)]
enum Cond {
    /// Bit `n` (16 to 31) of the path hash, while fuel lasts.
    Computed(u32),
    /// The immediate `true`.
    True,
    /// The immediate `false`.
    False,
}

/// How a generated block ends; targets are picked from raw numbers when the graph is built.
#[derive(Clone, Copy, Debug)]
enum End {
    Return,
    Unreachable,
    /// A jump forward.
    Jump(usize),
    /// A branch to any block, else forward.
    Branch {
        cond: Cond,
        then: usize,
        otherwise: usize,
    },
}

/// A generated block.
#[derive(Clone, Copy, Debug)]
struct Spec {
    demote: bool,
    end: End,
}

fn any_cond() -> impl Strategy<Value = Cond> {
    prop_oneof![
        8 => (16_u32..32).prop_map(Cond::Computed),
        1 => Just(Cond::True),
        1 => Just(Cond::False),
    ]
}

fn any_end() -> impl Strategy<Value = End> {
    prop_oneof![
        1 => Just(End::Return),
        1 => Just(End::Unreachable),
        3 => any::<usize>().prop_map(End::Jump),
        8 => (any_cond(), any::<usize>(), any::<usize>())
            .prop_map(|(cond, then, otherwise)| End::Branch { cond, then, otherwise }),
    ]
}

fn any_graph() -> impl Strategy<Value = Vec<Spec>> {
    prop::collection::vec(
        (prop::bool::weighted(0.1), any_end()).prop_map(|(demote, end)| Spec { demote, end }),
        1..=MAX_BLOCKS,
    )
}

/// The block `index` of a graph of `count` blocks names, `raw` picking among those after it.
fn forward(index: usize, count: usize, raw: usize) -> u32 {
    let after = count - index - 1;
    u32::try_from(index + 1 + raw % after).expect("a small graph")
}

/// Builds the graph `specs` describes.
fn build(specs: &[Spec]) -> Function {
    let mut function = Function::new();
    let count = specs.len();
    let blocks: Vec<BlockId> = (0..count)
        .map(|index| {
            if index == 0 {
                function.entry()
            } else {
                function.add_block().expect("a block")
            }
        })
        .collect();
    let block = |index: u32| blocks[usize::try_from(index).expect("small")];
    for (index, spec) in specs.iter().enumerate() {
        let mut b = Builder::new(&mut function, blocks[index]);
        let imm = |value: u32| Operand::Imm(Imm::U32(value));
        let r = |n: u8| Operand::Imm(Imm::Reg(crate::decode::Gpr::new(n)));
        let visits = b.op(Opcode::GetRegister, &[r(0)]).expect("op");
        let visits = b
            .op(Opcode::IAdd32, &[Operand::Value(visits), imm(1)])
            .expect("op");
        b.op(Opcode::SetRegister, &[r(0), Operand::Value(visits)])
            .expect("op");
        let hash = b.op(Opcode::GetRegister, &[r(1)]).expect("op");
        let hash = b
            .op(Opcode::IMul32, &[Operand::Value(hash), imm(HASH)])
            .expect("op");
        let number = u32::try_from(index + 1).expect("small");
        let hash = b
            .op(Opcode::IAdd32, &[Operand::Value(hash), imm(number)])
            .expect("op");
        b.op(Opcode::SetRegister, &[r(1), Operand::Value(hash)])
            .expect("op");
        if spec.demote {
            b.op(Opcode::DemoteToHelperInvocation, &[]).expect("op");
        }
        // The last block cannot go forward: it returns unless it is unreachable.
        let end = match spec.end {
            End::Jump(_) | End::Branch { .. } if index + 1 == count => End::Return,
            end => end,
        };
        let terminator = match end {
            End::Return => Terminator::Return,
            End::Unreachable => Terminator::Unreachable,
            End::Jump(raw) => Terminator::Jump(block(forward(index, count, raw))),
            End::Branch {
                cond,
                then,
                otherwise,
            } => {
                let cond = match cond {
                    Cond::True => Operand::Imm(Imm::U1(true)),
                    Cond::False => Operand::Imm(Imm::U1(false)),
                    Cond::Computed(bit) => {
                        let fuel = b
                            .op(Opcode::ULessThan, &[Operand::Value(visits), imm(FUEL)])
                            .expect("op");
                        let bit = b
                            .op(
                                Opcode::BitFieldUExtract,
                                &[Operand::Value(hash), imm(bit), imm(1)],
                            )
                            .expect("op");
                        let set = b
                            .op(Opcode::INotEqual, &[Operand::Value(bit), imm(0)])
                            .expect("op");
                        Operand::Value(
                            b.op(
                                Opcode::LogicalAnd,
                                &[Operand::Value(fuel), Operand::Value(set)],
                            )
                            .expect("op"),
                        )
                    }
                };
                // An immediate `true` always takes `then`, which must go forward too.
                let then = if matches!(cond, Operand::Imm(Imm::U1(true))) {
                    forward(index, count, then)
                } else {
                    u32::try_from(then % count).expect("small")
                };
                Terminator::Branch {
                    cond,
                    then: block(then),
                    otherwise: block(forward(index, count, otherwise)),
                }
            }
        };
        b.terminate(terminator).expect("terminator");
    }
    function
}

proptest! {
    #![proptest_config(config())]

    #[test]
    fn generated_graphs_run_the_same_structured(
        specs in any_graph(),
        seeds in prop::collection::vec(any::<u32>(), 4),
    ) {
        let graph = build(&specs);
        graph.verify().expect("the generated graph verifies");
        let tree = structure(graph.clone(), MAX_STRUCTURE_WORK).expect("structures").function;
        prop_assert_eq!(tree.verify(), Ok(()));
        for seed in seeds {
            let mut machine = Machine::new();
            machine.regs[1] = seed;
            same_run(&graph, &tree, &machine);
        }
    }
}
