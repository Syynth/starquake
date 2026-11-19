//! The goto pass's transformations on hand-built trees: the ones no generated graph reaches
//! through `remove_gotos` (moving out of an `if`, a lift over a `break`), the sibling order,
//! and the internal invariants, each refused as an error rather than a panic.

use super::{Expr, Id, Kind, ROOT, Stmt, Tree};
use crate::ir::{BlockId, Function, Imm, Operand};
use crate::structure::{Invariant, MAX_STRUCTURE_WORK, StructureError};

/// A tree holding only its root.
fn tree() -> Tree {
    Tree {
        stmts: vec![Stmt {
            kind: Kind::Root,
            up: None,
            prev: None,
            next: None,
            first: None,
            last: None,
        }],
        exprs: Vec::new(),
        gotos: Vec::new(),
        work: 0,
        limit: MAX_STRUCTURE_WORK,
    }
}

/// Blocks `0..count` of a function.
fn blocks(count: usize) -> Vec<BlockId> {
    let mut function = Function::new();
    let mut ids = vec![function.entry()];
    for _ in 1..count {
        ids.push(function.add_block().expect("a block"));
    }
    ids
}

/// Makes a statement of `kind` and links it at the end of `parent`'s children.
fn add(tree: &mut Tree, parent: Id, kind: Kind) -> Id {
    let id = tree.make(kind).expect("made");
    tree.insert(parent, None, id).expect("linked");
    id
}

/// The tree under `parent` as text, one statement per line, indented by depth.
fn dump(tree: &Tree, parent: Id, depth: usize, out: &mut String) {
    let mut child = tree.stmt(parent).expect("a statement").first;
    while let Some(id) = child {
        let stmt = tree.stmt(id).expect("a statement");
        assert_eq!(stmt.up, Some(parent), "a child's parent link");
        let pad = "  ".repeat(depth);
        let line = match stmt.kind {
            Kind::Root => "root".to_owned(),
            Kind::Code(block) => format!("{block}"),
            Kind::Goto { cond, label } => {
                format!("goto L{} if {}", label_id(tree, label), expr(tree, cond))
            }
            Kind::Label(id) => format!("L{id}:"),
            Kind::If { cond } => format!("if {}", expr(tree, cond)),
            Kind::Loop { cond } => format!("loop while {}", expr(tree, cond)),
            Kind::Break { cond } => format!("break if {}", expr(tree, cond)),
            Kind::Return => "return".to_owned(),
            Kind::Unreachable => "unreachable".to_owned(),
            Kind::SetVariable { id, value } => format!("v{id} = {}", expr(tree, value)),
        };
        out.push_str(&pad);
        out.push_str(&line);
        out.push('\n');
        dump(tree, id, depth + 1, out);
        child = stmt.next;
    }
}

fn label_id(tree: &Tree, label: Id) -> u32 {
    tree.label_id(label).expect("a label")
}

fn expr(tree: &Tree, id: super::ExprId) -> String {
    match tree.expr_of(id).expect("a condition") {
        Expr::Operand(Operand::Imm(Imm::U1(value))) => value.to_string(),
        Expr::Operand(operand) => format!("{operand:?}"),
        Expr::Not(operand) => format!("!{}", expr(tree, operand)),
        Expr::Or(a, b) => format!("({} || {})", expr(tree, a), expr(tree, b)),
        Expr::Variable(id) => format!("v{id}"),
    }
}

fn text(tree: &Tree) -> String {
    let mut out = String::new();
    dump(tree, ROOT, 0, &mut out);
    out
}

fn always(tree: &mut Tree) -> super::ExprId {
    tree.expr(Expr::Operand(Operand::Imm(Imm::U1(true))))
        .expect("made")
}

#[test]
fn moving_out_of_an_if_keeps_every_statement() {
    // root: if true { goto L0; block0; block1 }; L0: block2
    let b = blocks(3);
    let mut tree = tree();
    let cond = always(&mut tree);
    let if_stmt = add(&mut tree, ROOT, Kind::If { cond });
    let label = add(&mut tree, ROOT, Kind::Label(0));
    add(&mut tree, ROOT, Kind::Code(b[2]));
    let goto_cond = always(&mut tree);
    let goto = add(
        &mut tree,
        if_stmt,
        Kind::Goto {
            cond: goto_cond,
            label,
        },
    );
    add(&mut tree, if_stmt, Kind::Code(b[0]));
    add(&mut tree, if_stmt, Kind::Code(b[1]));
    let moved = tree.move_outward(goto).expect("moved");
    // Eden's `pop_front` would drop block0 from the new `if`; both blocks stay.
    assert_eq!(
        text(&tree),
        "if true\n  v0 = true\n  if !v0\n    block0\n    block1\ngoto L0 if v0\nL0:\nblock2\n"
    );
    assert_eq!(tree.stmt(moved).expect("a statement").up, Some(ROOT));
}

#[test]
fn moving_out_of_a_loop_breaks() {
    // root: loop while true { block0; goto L0; block1 }; L0:
    let b = blocks(2);
    let mut tree = tree();
    let cond = always(&mut tree);
    let loop_stmt = add(&mut tree, ROOT, Kind::Loop { cond });
    let label = add(&mut tree, ROOT, Kind::Label(0));
    add(&mut tree, loop_stmt, Kind::Code(b[0]));
    let goto_cond = always(&mut tree);
    let goto = add(
        &mut tree,
        loop_stmt,
        Kind::Goto {
            cond: goto_cond,
            label,
        },
    );
    add(&mut tree, loop_stmt, Kind::Code(b[1]));
    tree.move_outward(goto).expect("moved");
    assert_eq!(
        text(&tree),
        "loop while true\n  block0\n  v0 = true\n  break if v0\n  block1\ngoto L0 if v0\nL0:\n"
    );
}

#[test]
fn a_lift_over_a_break_is_refused() {
    // In a loop: L0: block0; break if true; goto L0 (a backward goto at the label's level,
    // which is lifted over the break).
    let b = blocks(1);
    let mut tree = tree();
    let cond = always(&mut tree);
    let outer = add(&mut tree, ROOT, Kind::Loop { cond });
    let label = add(&mut tree, outer, Kind::Label(0));
    add(&mut tree, outer, Kind::Code(b[0]));
    let break_cond = always(&mut tree);
    add(&mut tree, outer, Kind::Break { cond: break_cond });
    let goto_cond = always(&mut tree);
    let goto = add(
        &mut tree,
        outer,
        Kind::Goto {
            cond: goto_cond,
            label,
        },
    );
    tree.gotos.push(goto);
    assert_eq!(tree.remove_gotos(), Err(StructureError::LiftOverBreak));
}

#[test]
fn a_lift_with_no_break_wraps_the_statements_in_a_loop() {
    // The same without the break: lifted, then the goto is next to its label and goes.
    let b = blocks(1);
    let mut tree = tree();
    let label = add(&mut tree, ROOT, Kind::Label(0));
    add(&mut tree, ROOT, Kind::Code(b[0]));
    let goto_cond = always(&mut tree);
    let goto = add(
        &mut tree,
        ROOT,
        Kind::Goto {
            cond: goto_cond,
            label,
        },
    );
    tree.gotos.push(goto);
    tree.remove_gotos().expect("removed");
    assert_eq!(text(&tree), "loop while v0\n  L0:\n  block0\n  v0 = true\n");
}

#[test]
fn siblings_are_ordered_as_eden_orders_them() {
    let b = blocks(3);
    let mut tree = tree();
    let first = add(&mut tree, ROOT, Kind::Code(b[0]));
    let middle = add(&mut tree, ROOT, Kind::Code(b[1]));
    let last = add(&mut tree, ROOT, Kind::Code(b[2]));
    for (left, right, ordered) in [
        (first, last, true),
        (last, first, false),
        (first, middle, true),
        (middle, last, true),
        (last, middle, false),
        (middle, middle, false),
    ] {
        assert_eq!(tree.are_ordered(left, right), Ok(ordered));
    }
}

#[test]
fn broken_invariants_are_errors() {
    let b = blocks(2);
    let mut tree = tree();
    let label = add(&mut tree, ROOT, Kind::Label(0));
    let code = add(&mut tree, ROOT, Kind::Code(b[0]));
    let goto_cond = always(&mut tree);
    let goto = add(
        &mut tree,
        ROOT,
        Kind::Goto {
            cond: goto_cond,
            label,
        },
    );
    // A goto in the root cannot move out of it.
    assert_eq!(
        tree.move_outward(goto),
        Err(StructureError::Internal(Invariant::Movement))
    );
    // A code statement is not a goto, nor a label.
    assert_eq!(
        tree.remove_goto(code),
        Err(StructureError::Internal(Invariant::Kind))
    );
    assert_eq!(
        tree.label_id(code),
        Err(StructureError::Internal(Invariant::Kind))
    );
    // A range whose end comes before its start is not a run of siblings.
    let cond = always(&mut tree);
    let if_stmt = tree.make(Kind::If { cond }).expect("made");
    assert_eq!(
        tree.splice(if_stmt, Some(goto), Some(label)),
        Err(StructureError::Internal(Invariant::Siblings))
    );
    // Linking before a statement of another parent is refused.
    let other = tree.make(Kind::Code(b[1])).expect("made");
    assert_eq!(
        tree.insert(if_stmt, Some(code), other),
        Err(StructureError::Internal(Invariant::Siblings))
    );
    // The root has no parent.
    assert_eq!(
        tree.parent(ROOT),
        Err(StructureError::Internal(Invariant::Parents))
    );
}
