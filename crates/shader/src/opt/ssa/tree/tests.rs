//! The tree's scopes, checked directly: the rewrite only asks whether a body encloses another
//! across a merge block, so a scope check that ignored where a body closes would not change
//! its output (the blocks it adds always separate two sibling bodies). And the predecessors,
//! against the verifier's own derivation (`ir::verify`'s step 5).

use proptest::collection::vec;
use proptest::prelude::*;
use proptest::test_runner::Config;

use super::{Scopes, normalize, structured};
use crate::ir::{BlockId, Function, Imm, Inst, Node, Operand, Phi, Type, VerifyError};
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

/// A tree shape for [`the_verifier_takes_the_rewrites_predecessors`]; blocks are made as the
/// shape is built.
#[derive(Clone, Debug)]
enum Shape {
    Block,
    If(Vec<Shape>),
    Loop(Vec<Shape>),
    Break,
    Return,
    Unreachable,
}

/// Lists of shapes up to three levels deep.
fn shapes() -> impl Strategy<Value = Vec<Shape>> {
    let leaf = prop_oneof![
        4 => Just(Shape::Block),
        1 => Just(Shape::Break),
        1 => Just(Shape::Return),
        1 => Just(Shape::Unreachable),
    ];
    let node = leaf.prop_recursive(3, 32, 4, |inner| {
        prop_oneof![
            2 => Just(Shape::Block),
            1 => vec(inner.clone(), 0..4).prop_map(Shape::If),
            1 => vec(inner, 0..4).prop_map(Shape::Loop),
        ]
    });
    vec(node, 0..6)
}

/// The nodes of `shapes`, taking `spare` (the function's first block) for the first block and
/// adding a block to `function` for each other; a `break` outside every loop (which the
/// verifier refuses) is left out.
fn build(
    function: &mut Function,
    spare: &mut Option<BlockId>,
    shapes: &[Shape],
    in_loop: bool,
) -> Vec<Node> {
    let cond = Operand::Imm(Imm::U1(true));
    let mut nodes = Vec::new();
    for shape in shapes {
        match shape {
            Shape::Block => {
                let block = match spare.take() {
                    Some(block) => block,
                    None => function.add_block().expect("a block"),
                };
                nodes.push(Node::Block(block));
            }
            Shape::If(body) => nodes.push(Node::If {
                cond,
                body: build(function, spare, body, in_loop),
            }),
            Shape::Loop(body) => nodes.push(Node::Loop {
                body: build(function, spare, body, true),
                repeat: cond,
            }),
            Shape::Break if in_loop => nodes.push(Node::Break { cond }),
            Shape::Break => {}
            Shape::Return => nodes.push(Node::Return),
            Shape::Unreachable => nodes.push(Node::Unreachable),
        }
    }
    nodes
}

/// Whether the verifier accepts a phi in `block` of `function` whose one incoming block is
/// `from`; it must otherwise refuse it as not a predecessor.
fn verifier_accepts(function: &Function, block: BlockId, from: BlockId) -> bool {
    let mut probe = function.clone();
    let phi = Inst::Phi(Phi {
        ty: Type::U32,
        incoming: vec![(from, Operand::Imm(Imm::U32(0)))],
    });
    let at = probe.push(block, phi).expect("a phi");
    match probe.verify() {
        Ok(()) => true,
        Err(VerifyError::PhiNotPredecessor { at: refused, block }) => {
            assert_eq!((refused, block), (at, from), "{probe}");
            false
        }
        Err(other) => panic!("{other}\n{probe}"),
    }
}

proptest! {
    #![proptest_config(Config { cases: 128, ..crate::testing::config() })]

    /// The verifier's structured predecessors (`ir::verify`'s step 5, derived its own way, from
    /// points) are the ones this walk gives, block for block, on every normalized tree: each
    /// block of the function is tried as the one incoming block of a phi in each block.
    #[test]
    fn the_verifier_takes_the_rewrites_predecessors(shapes in shapes()) {
        let mut function = Function::new();
        let mut spare = Some(function.entry());
        let mut tree = build(&mut function, &mut spare, &shapes, false);
        // A tree with no block of its own still places the function's first block.
        if let Some(entry) = spare {
            tree.insert(0, Node::Block(entry));
        }
        function.set_structure(Some(tree));
        let mut work = Work::default();
        normalize(&mut function, &mut work).expect("normalizes");
        let graph = structured(&function, &mut work);
        let ids: Vec<BlockId> = function.blocks().map(|(id, _)| id).collect();
        for (&block, preds) in ids.iter().zip(&graph.preds) {
            for &from in &ids {
                prop_assert_eq!(
                    verifier_accepts(&function, block, from),
                    preds.contains(&from),
                    "{} from {}\n{}",
                    block,
                    from,
                    function
                );
            }
        }
    }
}

/// The probe itself: block1 takes from block0 and not from block1, so a verifier that
/// accepted every block, or none, would fail it.
#[test]
fn the_probe_tells_a_predecessor_from_another_block() {
    let function =
        Function::parse("block0:\nblock1:\nstructure:\n    block0\n    block1\n").expect("parses");
    let [b0, b1] = [0, 1].map(|index| block(&function, index));
    assert!(verifier_accepts(&function, b1, b0));
    assert!(!verifier_accepts(&function, b1, b1));
}

/// On a loop body that does not start with a block, which [`normalize`] never leaves, this
/// walk adds no back edge: block2 does not list itself, where the verifier (and the
/// interpreter, `ir::verify::phis::tests::a_loop_body_that_starts_with_an_if_takes_the_back_edge`)
/// let block2 follow itself. The rewrite normalizes first, so it never meets this tree.
#[test]
fn the_rewrites_walk_adds_no_back_edge_to_a_body_that_starts_with_an_if() {
    let function = Function::parse(
        "block0:
block1:
block2:
block3:
structure:
    block0
    loop {
        if u1 true {
            block1
        }
        block2
    } repeat u1 true
    block3
",
    )
    .expect("parses");
    let graph = structured(&function, &mut Work::default());
    let [b0, b1, b2] = [0, 1, 2].map(|index| block(&function, index));
    assert_eq!(graph.preds.get(2), Some(&vec![b0, b1]));
    assert!(verifier_accepts(&function, b2, b2));
}
