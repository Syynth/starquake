//! Inlining: recursion through other functions, the depth and size limits at their edges, a
//! kill in a function copied per call, and the internal-invariant errors.

use super::super::blocks::{RawBlock, RawEnd, RawGraph};
use super::super::{
    BarrierModel, BlockEnd, BlockIndex, BranchTables, FlowBlock, FlowError, FlowGraph, Invariant,
    MAX_FLOW_BLOCKS, MAX_INLINE_DEPTH, analyze, inline,
};
use super::{
    Program, block, branch, exit, graph, jump, kill, kill_then, name, p, pure, side_effect, tested,
};
use crate::decode::{Instruction, PredSrc};

#[test]
fn mutual_recursion_is_refused() {
    // The program calls a; a calls b; b calls a. The walk accepts it (the reference refuses only
    // a function calling itself); inlining refuses b's call, at index 6.
    let program = Program::new()
        .call(10, "a")
        .op(exit())
        .label("a")
        .call(11, "b")
        .ret(10)
        .label("b")
        .call(10, "a")
        .ret(11)
        .clone();
    let code = program.build();
    assert!(analyze(&code, &BranchTables::new(0), BarrierModel::Registers).is_ok());
    let call = program.at("b") + 1;
    assert_eq!(graph(&code), Err(FlowError::Recursion { index: call }));
}

/// The program calls f1, f1 calls f2, and so on to f`depth`; f`n` passes the return address
/// in R`10 + n - 1`. Returns the code and the index of each function's `CALL`.
fn chain(depth: u32) -> (Vec<Instruction>, Vec<u32>) {
    let link = |n: u32| u8::try_from(10 + n - 1).expect("a register");
    let mut program = Program::new();
    let mut calls = Vec::new();
    calls.push(program.here() + 1);
    program.call(link(1), "f1").op(exit());
    for n in 1..=depth {
        program.label(name(format!("f{n}")));
        if n < depth {
            calls.push(program.here() + 1);
            program.call(link(n + 1), name(format!("f{}", n + 1)));
        }
        program.ret(link(n));
    }
    (program.build(), calls)
}

#[test]
fn depth_at_its_edge() {
    let (code, _) = chain(MAX_INLINE_DEPTH);
    let graph = graph(&code).expect("accepted at the limit");
    // Every function once: the program's two blocks (the call, the exit) and, per function,
    // its call and its return block (the last has only its return).
    let functions = usize::try_from(MAX_INLINE_DEPTH).expect("small");
    assert_eq!(graph.blocks().len(), 2 + 2 * functions - 1);
    let (code, calls) = chain(MAX_INLINE_DEPTH + 1);
    // The call that would put f17 at depth 17 is f16's.
    let deepest = calls[usize::try_from(MAX_INLINE_DEPTH).expect("small")];
    assert_eq!(
        super::graph(&code),
        Err(FlowError::InlineDepth { index: deepest })
    );
}

/// The program calls f `calls` times, then runs `extra` blocks' worth of branches to the next
/// instruction and exits; f is a pure instruction, `guarded` guarded side effects and a
/// return. The inlined graph has `calls + 1 + extra` program blocks and `2 + 2 * guarded`
/// blocks per copy of f.
fn wide(calls: usize, extra: usize, guarded: usize) -> Vec<Instruction> {
    let mut program = Program::new();
    for _ in 0..calls {
        program.call(10, "f");
    }
    for n in 0..extra {
        let next = name(format!("next{n}"));
        program.bra(PredSrc::TRUE, next).label(next);
    }
    program.op(exit()).label("f").op(pure(1));
    for _ in 0..guarded {
        program.word(side_effect(p(0)));
    }
    program.ret(10);
    program.build()
}

#[test]
fn size_at_its_edge() {
    // 27 calls of a function of 9,708 blocks and a program of 28 blocks: 2^18 blocks.
    let calls = 27;
    assert_eq!((MAX_FLOW_BLOCKS - 1) % calls, 0);
    let per_copy = (MAX_FLOW_BLOCKS - 1) / calls - 1;
    assert_eq!(per_copy % 2, 0);
    let guarded = (per_copy - 2) / 2;
    let graph = graph(&wide(calls, 0, guarded)).expect("accepted at the limit");
    assert_eq!(graph.blocks().len(), MAX_FLOW_BLOCKS);
    // One more program block: the last call's copy goes past the limit.
    let code = wide(calls, 1, guarded);
    let last_call = u32::try_from(2 * calls - 1).expect("small");
    assert_eq!(
        super::graph(&code),
        Err(FlowError::InlineSize { index: last_call })
    );
}

#[test]
fn a_kill_in_a_function_continues_in_its_own_copy() {
    // 0 MOV R10; 1 CALL f; 2 MOV R10; 3 CALL f; 4 EXIT; f = 5: @P0 KILL; 6: RET R10. Each copy
    // of f has its test, its return and the KILL's block, whose end names the copy's return.
    let code = Program::new()
        .call(10, "f")
        .call(10, "f")
        .op(exit())
        .label("f")
        .op(kill(p(0)))
        .ret(10)
        .build();
    assert_eq!(
        graph(&code).expect("accepted").blocks(),
        &[
            block(0..1, jump(3)),
            block(2..3, jump(6)),
            block(4..5, BlockEnd::Exit),
            block(5..5, branch(p(0), 5, 4)),
            block(6..6, jump(1)),
            tested(5..6, kill_then(4)),
            block(5..5, branch(p(0), 8, 7)),
            block(6..6, jump(2)),
            tested(5..6, kill_then(7)),
        ]
    );
}

#[test]
fn a_block_id_that_names_no_block_is_an_internal_error() {
    // Inlining: a raw block jumping to raw block 5, which does not exist.
    let raw = RawGraph {
        blocks: vec![RawBlock {
            code: 0..0,
            guard_tested: false,
            end: RawEnd::Jump(5),
        }],
        functions: vec![vec![0]],
        position: vec![0],
    };
    assert_eq!(
        inline::inline(&raw),
        Err(FlowError::Internal(Invariant::NoBlock(5)))
    );
    // A kill's next block too.
    let raw = RawGraph {
        blocks: vec![RawBlock {
            code: 0..1,
            guard_tested: true,
            end: RawEnd::Kill(6),
        }],
        ..raw
    };
    assert_eq!(
        inline::inline(&raw),
        Err(FlowError::Internal(Invariant::NoBlock(6)))
    );
    // The IR's blocks: a flow block jumping to flow block 3 of a graph of one.
    let graph = FlowGraph {
        blocks: vec![FlowBlock {
            code: 0..0,
            guard_tested: false,
            end: BlockEnd::Jump(BlockIndex(3)),
        }],
    };
    assert_eq!(
        graph.build_ir(),
        Err(FlowError::Internal(Invariant::NoBlock(3)))
    );
}

#[test]
fn internal_and_unwalked_errors_read_as_plain_words() {
    let cases = [
        (
            FlowError::Internal(Invariant::BlockCount),
            "flow analysis broke an invariant of its own: more blocks than a block id can name",
        ),
        (
            FlowError::Internal(Invariant::Unfilled(4)),
            "flow analysis broke an invariant of its own: block 4 was never filled",
        ),
        (
            FlowError::Internal(Invariant::NoBlock(0)),
            "flow analysis broke an invariant of its own: block 0 does not exist",
        ),
        (
            FlowError::Unwalked {
                index: 2,
                target: 7,
            },
            "instruction 2 leads to instruction 7, which the walk did not reach",
        ),
    ];
    for (error, text) in cases {
        assert_eq!(error.to_string(), text);
    }
}
