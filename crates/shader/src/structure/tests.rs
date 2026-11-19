//! The structurizer's tests: hand-built graphs checked against golden IR text and run in the
//! interpreter, the input checks with their exact errors, the limits at their edges, and
//! generated graphs whose structured form must run as the graph does.

mod checks;
mod generated;
mod golden;
mod limits;

use super::{MAX_STRUCTURE_WORK, Structured, structure};
use crate::interp::{InterpError, Machine};
use crate::ir::Function;

/// Parses `text`, which must be well-formed.
fn parse(text: &str) -> Function {
    Function::parse(text).expect("parses")
}

/// `text` structured with the production limit.
fn structured(text: &str) -> Structured {
    structure(parse(text), MAX_STRUCTURE_WORK).expect("structures")
}

/// How a run ended, for comparing a graph's run with its structured form's: the sites of an
/// error differ between the forms, so only its kind is kept.
#[derive(Debug, PartialEq, Eq)]
enum Ending {
    /// The program returned.
    Returned,
    /// Control reached an `unreachable`.
    Unreachable,
    /// Any other error, as words.
    Error(String),
}

/// Runs `function` on `machine` and says how it ended.
fn ending(function: &Function, machine: &mut Machine) -> Ending {
    match machine.run(function) {
        Ok(_) => Ending::Returned,
        Err(InterpError::ReachedUnreachable(_)) => Ending::Unreachable,
        Err(error) => Ending::Error(error.to_string()),
    }
}

/// Runs `graph` and `tree` from `machine`'s state and checks they end the same way and leave
/// the same state, except the goto variables, which only structuring adds. Returns the
/// graph's ending and final state.
fn same_run(graph: &Function, tree: &Function, machine: &Machine) -> (Ending, Machine) {
    let mut by_graph = machine.clone();
    let mut by_tree = machine.clone();
    let graph_ending = ending(graph, &mut by_graph);
    let tree_ending = ending(tree, &mut by_tree);
    assert_eq!(graph_ending, tree_ending, "the forms end differently");
    by_tree.goto_variables.clone_from(&by_graph.goto_variables);
    assert_eq!(by_graph, by_tree, "the forms leave different state");
    (graph_ending, by_graph)
}
