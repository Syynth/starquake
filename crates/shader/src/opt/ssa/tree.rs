//! The graph the SSA rewrite reads predecessors from: a control-flow graph's terminators, or the
//! edges a structured function's tree implies (the parent module's docs, "Structured
//! functions"), with the scope each block's values are visible in.
//!
//! Eden's structurizer records each structured block's predecessors as it emits the tree
//! (`eden/src/shader_recompiler/frontend/maxwell/structured_control_flow.cpp` @ cf654052,
//! `TranslatePass::Visit`, every `AddBranch`), and its SSA rewrite reads them
//! (`ir_opt/ssa_rewrite_pass.cpp`, `ImmPredecessors`). Here a structured block has no
//! terminator, so [`structured`] derives the same edges from the tree, by the structured-phi
//! rule the interpreter uses (`crate::interp`): a block's predecessors are the blocks that can
//! have run last before it. [`normalize`] first adds the empty blocks Eden's structurizer makes
//! that the tree here may lack (a first block, a loop header, a merge block after an `if` or a
//! loop), so every point where paths meet is the start of a block.

use crate::ir::{BlockId, Function, Node, Operand};

use super::super::{PassError, Work, reachable, slot};

/// The scopes of a structured tree: the root and each `if` or loop body, numbered in tree
/// order, with the interval of a pre-order walk that encloses each, and the body each block is
/// placed in. A control-flow graph has one scope.
pub(super) struct Scopes {
    /// Each body's interval: the walk's count when it opened and when it closed.
    bodies: Vec<(u64, u64)>,
    /// Each block's body, by block index (the root for a block the tree does not place).
    blocks: Vec<usize>,
}

impl Scopes {
    /// One scope holding every block (a control-flow graph).
    fn flat(blocks: usize) -> Self {
        Self {
            bodies: vec![(0, u64::MAX)],
            blocks: vec![0; blocks],
        }
    }

    /// Whether this is [`Scopes::flat`]: every block shares one scope.
    pub(super) fn is_flat(&self) -> bool {
        self.bodies.len() == 1
    }

    /// Whether the body `outer` is placed in encloses (or is) the body `inner` is placed in:
    /// then the values of `outer`, once defined, are in scope in `inner` (the verifier's scope
    /// rule, `ir` module docs).
    pub(super) fn encloses(&self, outer: BlockId, inner: BlockId) -> bool {
        let body = |block: BlockId| {
            slot(block)
                .and_then(|index| self.blocks.get(index))
                .and_then(|&body| self.bodies.get(body))
                .copied()
        };
        match (body(outer), body(inner)) {
            (Some((open, close)), Some((inner_open, inner_close))) => {
                open <= inner_open && inner_close <= close
            }
            _ => false,
        }
    }
}

/// The edges the rewrite reads, each block's reachability and the scopes.
pub(super) struct Graph {
    /// Each block's predecessors, each once.
    pub(super) preds: Vec<Vec<BlockId>>,
    /// Each block's successors, each once.
    pub(super) succs: Vec<Vec<BlockId>>,
    /// Whether the entry reaches each block.
    pub(super) reachable: Vec<bool>,
    /// The scopes values are visible in.
    pub(super) scopes: Scopes,
}

impl Graph {
    /// A control-flow graph's edges: [`super::super::predecessors`], each block's successors in
    /// terminator order (a branch to one block twice counts once), and the blocks
    /// [`reachable`] from the entry.
    pub(super) fn of_graph(function: &Function, work: &mut Work) -> Self {
        let preds = super::super::predecessors(function);
        let succs = function
            .blocks()
            .map(|(_, block)| {
                let mut list: Vec<BlockId> = Vec::new();
                for to in block.terminator.successors() {
                    if list.last() != Some(&to) {
                        list.push(to);
                    }
                }
                list
            })
            .collect();
        let reachable = reachable(function, work);
        Self {
            preds,
            succs,
            reachable,
            scopes: Scopes::flat(function.block_count()),
        }
    }
}

/// What a level of [`normalize`]'s rebuilt tree is the body of.
enum Wrap {
    Root,
    If(Operand),
    Loop(Operand),
}

/// One level of [`normalize`]'s walk: the input nodes left and the output so far.
struct Level<'t> {
    nodes: std::slice::Iter<'t, Node>,
    out: Vec<Node>,
    wrap: Wrap,
}

/// Whether `node` is an `if` or a loop.
const fn is_construct(node: Option<&Node>) -> bool {
    matches!(node, Some(Node::If { .. } | Node::Loop { .. }))
}

/// Adds an empty block to `function` and returns its node.
fn new_block(function: &mut Function) -> Result<Node, PassError> {
    Ok(Node::Block(function.add_block()?))
}

/// Adds an empty block where the tree has none (the parent module's docs, "Structured
/// functions"): at the start of the root list and of each loop body when it does not start
/// with a block (Eden's first block and loop header), and after each `if` and loop that is
/// followed by neither a block, a `return` nor an `unreachable` (Eden's merge block). An empty
/// root gains nothing. The walk keeps an explicit stack, so a deep tree cannot overflow the
/// host's stack; it takes one step per node.
pub(super) fn normalize(function: &mut Function, work: &mut Work) -> Result<(), PassError> {
    let Some(tree) = function.structure().map(<[Node]>::to_vec) else {
        return Ok(());
    };
    let mut stack = vec![Level {
        nodes: tree.iter(),
        out: Vec::new(),
        wrap: Wrap::Root,
    }];
    while let Some(top) = stack.last_mut() {
        if let Some(node) = top.nodes.next() {
            work.step();
            let is_block = matches!(node, Node::Block(_));
            let opens = matches!(top.wrap, Wrap::Root | Wrap::Loop(_)) && top.out.is_empty();
            let merges = is_construct(top.out.last())
                && !matches!(node, Node::Block(_) | Node::Return | Node::Unreachable);
            if !is_block && (opens || merges) {
                let block = new_block(function)?;
                top.out.push(block);
            }
            match node {
                Node::If { cond, body } => stack.push(Level {
                    nodes: body.iter(),
                    out: Vec::new(),
                    wrap: Wrap::If(*cond),
                }),
                Node::Loop { body, repeat } => stack.push(Level {
                    nodes: body.iter(),
                    out: Vec::new(),
                    wrap: Wrap::Loop(*repeat),
                }),
                other => top.out.push(other.clone()),
            }
            continue;
        }
        let Some(Level { mut out, wrap, .. }) = stack.pop() else {
            break;
        };
        let empty_loop = matches!(wrap, Wrap::Loop(_)) && out.is_empty();
        if empty_loop || is_construct(out.last()) {
            let block = new_block(function)?;
            out.push(block);
        }
        let node = match wrap {
            Wrap::Root => {
                function.set_structure(Some(out));
                break;
            }
            Wrap::If(cond) => Node::If { cond, body: out },
            Wrap::Loop(repeat) => Node::Loop { body: out, repeat },
        };
        if let Some(parent) = stack.last_mut() {
            parent.out.push(node);
        }
    }
    Ok(())
}

/// What a frame of [`structured`]'s walk is the body of.
enum Kind {
    Root,
    /// An `if`: the blocks that can have run last before it, which are also last after it
    /// when its condition is false.
    If { before: Vec<BlockId> },
    /// A loop: its first block, which the end of the body jumps back to, and the blocks
    /// that can have run last at each `break`.
    Loop {
        header: Option<BlockId>,
        breaks: Vec<BlockId>,
    },
}

/// One frame of [`structured`]'s walk.
struct Frame<'t> {
    nodes: std::slice::Iter<'t, Node>,
    kind: Kind,
    /// The body's index in [`Scopes::bodies`].
    body: usize,
    /// Whether a node of the body has been walked.
    started: bool,
}

/// Sorts `list` and drops repeats.
fn sort_dedup(list: &mut Vec<BlockId>) {
    list.sort_unstable();
    list.dedup();
}

/// The edges the tree of `function` (normalized by [`normalize`]) implies (see the module
/// docs), each block's reachability from the tree's first block, and the scopes. Walking the
/// tree, the blocks that can have run last are carried along: a block's predecessors are
/// those, and it is then the only one; an `if` adds the ones before it to those at the end of
/// its body (its condition may be false); a loop's first block also follows the end of its
/// body (the repeat), and after the loop come the end of its body and every `break`'s blocks;
/// a `break` passes its blocks on too (its condition may be false); after a `return` or an
/// `unreachable` no block has run last. The tree's first block is the entry. A `break` outside
/// any loop adds no edge (the verifier refuses one). Each predecessor list is sorted by block
/// index, Eden's `OrderPhiArgs` order. The walk takes one step per node and one per edge.
pub(super) fn structured(function: &Function, work: &mut Work) -> Graph {
    let count = function.block_count();
    let mut preds: Vec<Vec<BlockId>> = vec![Vec::new(); count];
    let mut scopes = Scopes {
        bodies: vec![(0, 0)],
        blocks: vec![0; count],
    };
    let mut clock: u64 = 1;
    let mut entry = None;
    let mut last: Vec<BlockId> = Vec::new();
    let tree = function.structure().unwrap_or(&[]);
    let mut stack = vec![Frame {
        nodes: tree.iter(),
        kind: Kind::Root,
        body: 0,
        started: false,
    }];
    // The frames that are loops, innermost last.
    let mut loops: Vec<usize> = Vec::new();
    while let Some(top) = stack.last_mut() {
        let first = !top.started;
        top.started = true;
        let Some(node) = top.nodes.next() else {
            let Some(frame) = stack.pop() else { break };
            if let Some(interval) = scopes.bodies.get_mut(frame.body) {
                interval.1 = clock;
            }
            clock = clock.saturating_add(1);
            match frame.kind {
                Kind::Root => {}
                Kind::If { before } => {
                    last.extend(before);
                    sort_dedup(&mut last);
                }
                Kind::Loop { header, breaks } => {
                    loops.pop();
                    if let Some(list) = header.and_then(slot).and_then(|index| preds.get_mut(index))
                    {
                        for &from in &last {
                            work.step();
                            list.push(from);
                        }
                    }
                    last.extend(breaks);
                    sort_dedup(&mut last);
                }
            }
            continue;
        };
        work.step();
        let body = top.body;
        match node {
            Node::Block(block) => {
                if let Some(index) = slot(*block) {
                    if let Some(list) = preds.get_mut(index) {
                        for &from in &last {
                            work.step();
                            list.push(from);
                        }
                    }
                    if let Some(scope) = scopes.blocks.get_mut(index) {
                        *scope = body;
                    }
                }
                if first && let Kind::Loop { header, .. } = &mut top.kind {
                    *header = Some(*block);
                }
                if entry.is_none() && first && matches!(top.kind, Kind::Root) {
                    entry = Some(*block);
                }
                last.clear();
                last.push(*block);
            }
            Node::If { body: inner, .. } | Node::Loop { body: inner, .. } => {
                let kind = if matches!(node, Node::If { .. }) {
                    Kind::If {
                        before: last.clone(),
                    }
                } else {
                    loops.push(stack.len());
                    Kind::Loop {
                        header: None,
                        breaks: Vec::new(),
                    }
                };
                let index = scopes.bodies.len();
                scopes.bodies.push((clock, 0));
                clock = clock.saturating_add(1);
                stack.push(Frame {
                    nodes: inner.iter(),
                    kind,
                    body: index,
                    started: false,
                });
            }
            Node::Break { .. } => {
                if let Some(Frame {
                    kind: Kind::Loop { breaks, .. },
                    ..
                }) = loops.last().and_then(|&index| stack.get_mut(index))
                {
                    breaks.extend_from_slice(&last);
                }
            }
            Node::Return | Node::Unreachable => last.clear(),
        }
    }
    let ids: Vec<BlockId> = function.blocks().map(|(id, _)| id).collect();
    let mut succs: Vec<Vec<BlockId>> = vec![Vec::new(); count];
    for (list, &to) in preds.iter_mut().zip(&ids) {
        sort_dedup(list);
        for &from in list.iter() {
            work.step();
            if let Some(out) = slot(from).and_then(|from| succs.get_mut(from)) {
                out.push(to);
            }
        }
    }
    let reachable = reach(entry, &succs, count, work);
    Graph {
        preds,
        succs,
        reachable,
        scopes,
    }
}

/// The blocks `entry` reaches along `succs`.
fn reach(entry: Option<BlockId>, succs: &[Vec<BlockId>], count: usize, work: &mut Work) -> Vec<bool> {
    let mut seen = vec![false; count];
    let mut stack: Vec<BlockId> = Vec::new();
    if let Some(entry) = entry
        && let Some(flag) = slot(entry).and_then(|index| seen.get_mut(index))
    {
        *flag = true;
        stack.push(entry);
    }
    while let Some(block) = stack.pop() {
        for &to in slot(block).and_then(|index| succs.get(index)).map_or(&[][..], Vec::as_slice) {
            work.step();
            if let Some(flag) = slot(to).and_then(|index| seen.get_mut(index))
                && !*flag
            {
                *flag = true;
                stack.push(to);
            }
        }
    }
    seen
}
