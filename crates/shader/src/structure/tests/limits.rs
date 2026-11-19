//! The limits: the work budget and the nesting bound at their edges, with their exact errors.

use super::{parse, same_run};
use crate::interp::Machine;
use crate::ir::{Function, MAX_NESTING, Node};
use crate::structure::{MAX_STRUCTURE_WORK, StructureError, structure};

/// How deeply `nodes` nest, counted as the verifier counts: the body of a top-level `if` or
/// loop is level 1.
fn depth(nodes: &[Node]) -> usize {
    nodes
        .iter()
        .map(|node| match node {
            Node::If { body, .. } | Node::Loop { body, .. } => 1 + depth(body),
            _ => 0,
        })
        .max()
        .unwrap_or(0)
}

/// `count` blocks, each but the last branching on p0 to the last, else to the next. Each
/// branch becomes an `if` around the rest, so the tree nests `count - 2` levels deep (from two
/// blocks on).
fn skips(count: usize) -> Function {
    let mut text = String::new();
    for index in 0..count {
        text.push_str(&format!("block{index}:\n"));
        if index + 1 == count {
            text.push_str("    return\n");
        } else {
            text.push_str(&format!(
                "    %{index} = SetRegister r{reg}, u32 {index}\n    %{cond} = GetPred p0\n    \
                 branch %{cond}, block{last}, block{next}\n",
                reg = 1 + index % 8,
                cond = count + index,
                last = count - 1,
                next = index + 1
            ));
        }
    }
    parse(&text)
}

#[test]
fn the_work_limit_at_its_edge() {
    let graph = skips(6);
    let work = structure(graph.clone(), MAX_STRUCTURE_WORK)
        .expect("structures")
        .work;
    let at = structure(graph.clone(), work).expect("structures at the limit");
    assert_eq!(at.work, work);
    assert_eq!(
        structure(graph.clone(), work - 1),
        Err(StructureError::WorkLimit { limit: work - 1 })
    );
    assert_eq!(
        structure(graph, 0),
        Err(StructureError::WorkLimit { limit: 0 })
    );
}

#[test]
fn the_work_limit_reads_as_words() {
    assert_eq!(
        StructureError::WorkLimit { limit: 0 }.to_string(),
        "structuring stopped at its work limit of 0"
    );
    assert_eq!(
        StructureError::WorkLimit { limit: 1 }.to_string(),
        "structuring stopped at its work limit of 1"
    );
}

#[test]
fn a_large_graph_stops_at_a_small_limit() {
    // Thousands of blocks against a limit of a hundred: the run stops at the limit rather
    // than structuring the whole graph first.
    let limit = 100;
    assert_eq!(
        structure(skips(4096), limit),
        Err(StructureError::WorkLimit { limit })
    );
}

/// `count` diamonds one after another: each tests p0 and runs one of two arms, which join at
/// the next diamond's test; the last join returns.
fn diamonds(count: usize) -> Function {
    let mut text = String::new();
    for index in 0..count {
        let (test, then, otherwise, join) =
            (3 * index, 3 * index + 1, 3 * index + 2, 3 * index + 3);
        text.push_str(&format!(
            "block{test}:\n    %{test} = GetPred p0\n    branch %{test}, block{then}, block{otherwise}\n\
             block{then}:\n    %{then} = SetRegister r1, u32 {index}\n    jump block{join}\n\
             block{otherwise}:\n    %{otherwise} = SetRegister r2, u32 {index}\n    jump block{join}\n"
        ));
    }
    text.push_str(&format!("block{}:\n    return\n", 3 * count));
    parse(&text)
}

#[test]
fn work_grows_linearly_with_a_run_of_diamonds() {
    // Each diamond adds the same work, at 64 diamonds as at 1,024: the pass does not rescan
    // the statements it has already structured.
    let work = |count| {
        structure(diamonds(count), MAX_STRUCTURE_WORK)
            .expect("structures")
            .work
    };
    let step = work(65) - work(64);
    assert_eq!(work(1025) - work(1024), step);
}

#[test]
fn nesting_at_its_edge() {
    let deepest = skips(MAX_NESTING + 2);
    let structured = structure(deepest.clone(), MAX_STRUCTURE_WORK).expect("structures");
    assert_eq!(
        depth(structured.function.structure().expect("structured")),
        MAX_NESTING
    );
    let mut machine = Machine::new();
    same_run(&deepest, &structured.function, &machine);
    machine.preds[0] = true;
    same_run(&deepest, &structured.function, &machine);
    assert_eq!(
        structure(skips(MAX_NESTING + 3), MAX_STRUCTURE_WORK),
        Err(StructureError::TooDeep)
    );
    assert_eq!(
        StructureError::TooDeep.to_string(),
        format!("the structured tree would nest deeper than {MAX_NESTING} levels")
    );
}
