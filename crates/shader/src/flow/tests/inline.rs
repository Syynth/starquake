//! Inlining: recursion through other functions, and the depth and size limits at their edges.

use super::super::{BarrierModel, BranchTables, FlowError, MAX_FLOW_BLOCKS, MAX_INLINE_DEPTH, analyze};
use super::{Program, exit, graph, name, p, pure, side_effect};
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
