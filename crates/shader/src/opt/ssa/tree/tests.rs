//! The tree's scopes, checked directly: the rewrite only asks whether a body encloses another
//! across a merge block, so a scope check that ignored where a body closes would not change
//! its output (the blocks it adds always separate two sibling bodies).

use super::{Scopes, normalize, structured};
use crate::ir::{BlockId, Function};
use crate::opt::Work;

/// The structured function `text`, normalized, and its scopes.
fn scopes(text: &str) -> (Function, Scopes) {
    let mut function = Function::parse(text).expect("parses");
    let mut work = Work::default();
    normalize(&mut function, &mut work).expect("normalizes");
    let graph = structured(&function, &mut work);
    (function, graph.scopes)
}

/// Block `index` of `function`.
fn block(function: &Function, index: usize) -> BlockId {
    function
        .blocks()
        .nth(index)
        .map(|(id, _)| id)
        .expect("a block")
}

#[test]
fn a_body_encloses_its_own_and_nested_blocks_only() {
    // block1 is in the first `if`, block2 in the second, block3 in a loop inside the second.
    let (function, scopes) = scopes(
        "block0:
block1:
block2:
block3:
structure:
    block0
    if u1 true {
        block1
    }
    if u1 true {
        block2
        loop {
            block3
        } repeat u1 false
    }
    return
",
    );
    let [b0, b1, b2, b3] = [0, 1, 2, 3].map(|index| block(&function, index));
    for (outer, inner, encloses) in [
        (b0, b1, true),
        (b0, b3, true),
        (b2, b3, true),
        (b1, b1, true),
        // Siblings: the first `if` opens before the second but closes before it opens.
        (b1, b2, false),
        (b1, b3, false),
        (b2, b1, false),
        // Inner bodies do not enclose outer ones.
        (b3, b2, false),
        (b1, b0, false),
    ] {
        assert_eq!(scopes.encloses(outer, inner), encloses, "{outer} {inner}");
    }
}
