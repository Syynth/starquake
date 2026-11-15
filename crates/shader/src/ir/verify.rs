//! The verifier: types, SSA dominance and structure (spec, "The intermediate representation";
//! Eden's `eden/src/shader_recompiler/ir_opt/verification_pass.cpp` @ cf654052,
//! GPL-2.0-or-later, whose checks are types, use counts, forward declarations and phi
//! placement; the parent module lists where this verifier differs).
//!
//! Checks run in this order, and the first failure is the error:
//!
//! 1. The function has a block.
//! 2. Block lists, blocks in arena order: every listed id names an instruction, and no id is
//!    listed twice (in one block or in two).
//! 3. Control. A control-flow graph: each block, in arena order, has a terminator whose targets
//!    exist. A structured function: the tree, in order, names blocks that exist, places none
//!    twice, breaks only inside a loop and nests at most [`MAX_NESTING`] levels; then every
//!    block, in arena order, is placed and has no terminator.
//! 4. Instructions and conditions, in execution order: a control-flow graph's blocks in arena
//!    order, each block's instructions and then its terminator's condition; a structured
//!    function's tree in order, an `if`'s condition before its body and a loop's repeat
//!    condition after its body. For an operation: (a) every value argument is defined where it
//!    is used, in argument order; (b) the argument count; (c) the flags' kind, then their
//!    ranges; (d) each argument, in order, has a result of the type its slot needs; (e) a
//!    register or predicate name is not one the operation cannot access. For a phi: (a) no
//!    operation comes before it in its block; (b) its type is a value type; (c) each incoming
//!    pair, in order: the block exists and is not repeated, in a control-flow graph the block is
//!    a predecessor, and the value is defined (dominating the predecessor in a control-flow
//!    graph, listed in some block in a structured function), has a result and has the phi's
//!    type; (d) in a control-flow graph, every predecessor has an incoming pair. A condition:
//!    defined, with a result, of type `u1`.
//!
//! "Defined where it is used" means, in a control-flow graph, defined earlier in the same block
//! or in a block that dominates the use's block (a use in a block the entry cannot reach is
//! accepted when its value is listed in some block, as LLVM's verifier does); in a structured
//! function, defined earlier in the tree and not inside an `if` or loop body that has already
//! ended.

use thiserror::Error;

use super::{
    Block, BlockId, Flags, Function, Imm, Inst, MAX_NESTING, Node, Opcode, Operand, Operation, Phi,
    Terminator, Type, Value,
};

/// Where a [`VerifyError`] was found.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Site {
    /// An instruction.
    Inst(Value),
    /// A block's terminator.
    Terminator(BlockId),
    /// The structured tree.
    Structure,
}

impl std::fmt::Display for Site {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Inst(value) => write!(f, "{value}"),
            Self::Terminator(block) => write!(f, "the terminator of {block}"),
            Self::Structure => f.write_str("the structure"),
        }
    }
}

/// Why [`Function::verify`] failed (or why [`Builder`](super::Builder) refused an instruction).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum VerifyError {
    /// The function has no blocks.
    #[error("the function has no blocks")]
    NoBlocks,
    /// A block lists an id with no instruction.
    #[error("{0} is listed in a block but does not exist")]
    Missing(Value),
    /// An id is listed twice.
    #[error("{0} is listed twice")]
    Repeated(Value),
    /// A block of a control-flow graph has no terminator.
    #[error("{0} has no terminator")]
    Unterminated(BlockId),
    /// A terminator, a phi or the tree names a block that does not exist.
    #[error("{at} names {block}, which does not exist")]
    NoSuchBlock {
        /// Where.
        at: Site,
        /// The block.
        block: BlockId,
    },
    /// A block of a structured function has a terminator.
    #[error("{0} has a terminator in a structured function")]
    TerminatorInStructure(BlockId),
    /// A block is not placed in the tree.
    #[error("{0} is not placed in the structure")]
    NotPlaced(BlockId),
    /// A block is placed in the tree twice.
    #[error("{0} is placed in the structure twice")]
    PlacedTwice(BlockId),
    /// A `break` outside every loop.
    #[error("the structure has a break outside every loop")]
    BreakOutsideLoop,
    /// The tree nests too deeply.
    #[error("the structure nests deeper than {max} levels", max = MAX_NESTING)]
    TooDeep,
    /// A value is used where it is not defined.
    #[error("{at} uses {operand} before it is defined")]
    Undefined {
        /// Where.
        at: Site,
        /// The value.
        operand: Value,
    },
    /// A value is used that has no result.
    #[error("{at} uses {operand}, which has no result")]
    NoResult {
        /// Where.
        at: Site,
        /// The value.
        operand: Value,
    },
    /// An operation has the wrong number of arguments.
    #[error("{at}: {opcode} takes {wanted} arguments, not {found}")]
    ArgCount {
        /// The instruction.
        at: Value,
        /// Its opcode.
        opcode: Opcode,
        /// How many it takes.
        wanted: usize,
        /// How many it has.
        found: usize,
    },
    /// An operation carries flags of the wrong kind.
    #[error("{at}: {opcode} does not carry these flags")]
    FlagsKind {
        /// The instruction.
        at: Value,
        /// Its opcode.
        opcode: Opcode,
    },
    /// Texture information with a 2-bit field out of range.
    #[error("{0}: a texture field is out of range")]
    FlagsRange(Value),
    /// An argument or condition has the wrong type.
    #[error("{at} needs argument {index} to be {wanted}, not {found}")]
    Type {
        /// Where.
        at: Site,
        /// The argument's index (a phi's incoming pair; 0 for a condition).
        index: usize,
        /// The type it needs.
        wanted: Type,
        /// The type it has.
        found: Type,
    },
    /// A get or set operation names `RZ`, `URZ`, `PT` or `UPT`.
    #[error("{at}: {opcode} cannot access {name}")]
    ReservedName {
        /// The instruction.
        at: Value,
        /// Its opcode.
        opcode: Opcode,
        /// The name.
        name: Imm,
    },
    /// A phi whose type no value has.
    #[error("{at} is a phi of type {ty}, which no value has")]
    PhiType {
        /// The phi.
        at: Value,
        /// Its type.
        ty: Type,
    },
    /// A phi after an operation in its block.
    #[error("{0} is a phi after an operation in its block")]
    PhiAfterOperation(Value),
    /// A phi names a block twice.
    #[error("{at} has two incoming values from {block}")]
    PhiRepeated {
        /// The phi.
        at: Value,
        /// The block.
        block: BlockId,
    },
    /// A phi names a block that is not a predecessor.
    #[error("{at} has an incoming value from {block}, which is not a predecessor")]
    PhiNotPredecessor {
        /// The phi.
        at: Value,
        /// The block.
        block: BlockId,
    },
    /// A phi has no incoming value from a predecessor.
    #[error("{at} has no incoming value from predecessor {block}")]
    PhiMissingPredecessor {
        /// The phi.
        at: Value,
        /// The predecessor.
        block: BlockId,
    },
}

impl Function {
    /// Checks the whole function (see the module docs of `verify` for the checks and their
    /// order).
    pub fn verify(&self) -> Result<(), VerifyError> {
        if self.blocks.is_empty() {
            return Err(VerifyError::NoBlocks);
        }
        let sites = self.def_sites()?;
        match self.structure() {
            None => {
                self.check_cfg()?;
                let graph = Graph::new(&self.blocks);
                Cfg {
                    function: self,
                    sites: &sites,
                    graph: &graph,
                }
                .check()
            }
            Some(nodes) => {
                let mut placed = vec![false; self.blocks.len()];
                place_nodes(nodes, 0, 0, &mut placed)?;
                for ((id, block), placed) in self.blocks().zip(&placed) {
                    if !placed {
                        return Err(VerifyError::NotPlaced(id));
                    }
                    if block.terminator != Terminator::None {
                        return Err(VerifyError::TerminatorInStructure(id));
                    }
                }
                Structured {
                    function: self,
                    sites: &sites,
                    visible: vec![false; self.insts.len()],
                    log: Vec::new(),
                }
                .nodes(nodes)
            }
        }
    }

    /// Step 2: where each instruction is listed, as (block index, position).
    fn def_sites(&self) -> Result<Vec<Option<(usize, usize)>>, VerifyError> {
        let mut sites = vec![None; self.insts.len()];
        for (block_index, block) in self.blocks.iter().enumerate() {
            for (position, &value) in block.insts.iter().enumerate() {
                let site = value
                    .slot()
                    .and_then(|slot| sites.get_mut(slot))
                    .ok_or(VerifyError::Missing(value))?;
                if site.is_some() {
                    return Err(VerifyError::Repeated(value));
                }
                *site = Some((block_index, position));
            }
        }
        Ok(sites)
    }

    /// Step 3 for a control-flow graph.
    fn check_cfg(&self) -> Result<(), VerifyError> {
        for (id, block) in self.blocks() {
            if block.terminator == Terminator::None {
                return Err(VerifyError::Unterminated(id));
            }
            for target in block.terminator.successors() {
                if self.block(target).is_none() {
                    return Err(VerifyError::NoSuchBlock {
                        at: Site::Terminator(id),
                        block: target,
                    });
                }
            }
        }
        Ok(())
    }
}

/// Step 3's walk of the tree: `depth` is the nesting level of `nodes`, `loops` how many
/// loops enclose them.
fn place_nodes(
    nodes: &[Node],
    depth: usize,
    loops: usize,
    placed: &mut [bool],
) -> Result<(), VerifyError> {
    for node in nodes {
        match node {
            Node::Block(block) => {
                let slot = block.slot().and_then(|slot| placed.get_mut(slot)).ok_or(
                    VerifyError::NoSuchBlock {
                        at: Site::Structure,
                        block: *block,
                    },
                )?;
                if *slot {
                    return Err(VerifyError::PlacedTwice(*block));
                }
                *slot = true;
            }
            Node::If { body, .. } | Node::Loop { body, .. } => {
                let inner = depth.checked_add(1).ok_or(VerifyError::TooDeep)?;
                if inner > MAX_NESTING {
                    return Err(VerifyError::TooDeep);
                }
                let loops = if matches!(node, Node::Loop { .. }) {
                    loops.saturating_add(1)
                } else {
                    loops
                };
                place_nodes(body, inner, loops, placed)?;
            }
            Node::Break { .. } if loops == 0 => return Err(VerifyError::BreakOutsideLoop),
            Node::Break { .. } | Node::Return | Node::Unreachable => {}
        }
    }
    Ok(())
}

/// Checks an operand against the type its slot needs: the value exists, has a result, and its
/// type fits (steps 4d and the condition checks; definedness is checked by the caller).
fn check_operand(
    function: &Function,
    at: Site,
    index: usize,
    operand: Operand,
    wanted: Type,
) -> Result<(), VerifyError> {
    if let Operand::Value(value) = operand {
        match function.inst(value) {
            None => return Err(VerifyError::Undefined { at, operand: value }),
            Some(inst) if inst.result() == Type::Void => {
                return Err(VerifyError::NoResult { at, operand: value });
            }
            Some(_) => {}
        }
    }
    let found = function.type_of(operand).ok_or(VerifyError::Type {
        at,
        index,
        wanted,
        found: Type::Opaque,
    })?;
    if wanted.accepts(found) {
        Ok(())
    } else {
        Err(VerifyError::Type {
            at,
            index,
            wanted,
            found,
        })
    }
}

/// Steps 4b to 4e for an operation: everything but definedness. [`Builder`](super::Builder)
/// runs the same checks as it appends.
pub(super) fn check_operation(
    function: &Function,
    at: Value,
    op: &Operation,
) -> Result<(), VerifyError> {
    let wanted = op.opcode.args();
    if wanted.len() != op.args.len() {
        return Err(VerifyError::ArgCount {
            at,
            opcode: op.opcode,
            wanted: wanted.len(),
            found: op.args.len(),
        });
    }
    if op.flags.kind() != op.opcode.flags_kind() {
        return Err(VerifyError::FlagsKind {
            at,
            opcode: op.opcode,
        });
    }
    if let Flags::Texture(info) = op.flags
        && !info.is_valid()
    {
        return Err(VerifyError::FlagsRange(at));
    }
    for (index, (&operand, &ty)) in op.args.iter().zip(wanted).enumerate() {
        check_operand(function, Site::Inst(at), index, operand, ty)?;
    }
    if let Some(name) = reserved_name(op) {
        return Err(VerifyError::ReservedName {
            at,
            opcode: op.opcode,
            name,
        });
    }
    Ok(())
}

/// The name a get or set operation must not access: `RZ` and `URZ` read zero and `PT` and `UPT`
/// read true, which the translators fold into immediates instead.
fn reserved_name(op: &Operation) -> Option<Imm> {
    let accessor = matches!(
        op.opcode,
        Opcode::GetRegister
            | Opcode::SetRegister
            | Opcode::GetPred
            | Opcode::SetPred
            | Opcode::GetUniformRegister
            | Opcode::SetUniformRegister
            | Opcode::GetUniformPred
            | Opcode::SetUniformPred
    );
    let name = op.args.first()?.imm()?;
    let reserved = match name {
        Imm::Reg(reg) => reg.is_zero(),
        Imm::Pred(pred) => pred.is_true(),
        Imm::UniformReg(reg) => reg.is_zero(),
        Imm::UniformPred(pred) => pred.is_true(),
        _ => false,
    };
    (accessor && reserved).then_some(name)
}

/// Steps 4b and the type part of 4c for a phi, for [`Builder`](super::Builder): its type is a
/// value type, and each incoming block exists and each incoming value has the phi's type.
pub(super) fn check_phi(function: &Function, at: Value, phi: &Phi) -> Result<(), VerifyError> {
    if !phi.ty.is_value() {
        return Err(VerifyError::PhiType { at, ty: phi.ty });
    }
    for (index, &(block, operand)) in phi.incoming.iter().enumerate() {
        if function.block(block).is_none() {
            return Err(VerifyError::NoSuchBlock {
                at: Site::Inst(at),
                block,
            });
        }
        check_operand(function, Site::Inst(at), index, operand, phi.ty)?;
    }
    Ok(())
}

/// Checks that a terminator's targets exist and its condition is a `u1`, for
/// [`Builder`](super::Builder); definedness is the verifier's.
pub(super) fn check_terminator(
    function: &Function,
    block: BlockId,
    terminator: Terminator,
) -> Result<(), VerifyError> {
    for target in terminator.successors() {
        if function.block(target).is_none() {
            return Err(VerifyError::NoSuchBlock {
                at: Site::Terminator(block),
                block: target,
            });
        }
    }
    if let Terminator::Branch { cond, .. } = terminator {
        check_operand(function, Site::Terminator(block), 0, cond, Type::U1)?;
    }
    Ok(())
}

/// The control-flow graph's edges and dominator tree (Cooper, Harvey and Kennedy's iterative
/// algorithm over the blocks the entry reaches).
struct Graph {
    /// Each block's predecessors, from every block (reachable or not), ascending, once each.
    preds: Vec<Vec<usize>>,
    /// Each reachable block's postorder number.
    postorder: Vec<Option<usize>>,
    /// Each reachable block's immediate dominator; the entry's is itself.
    idom: Vec<Option<usize>>,
}

impl Graph {
    fn successors(block: &Block, count: usize) -> impl Iterator<Item = usize> {
        block
            .terminator
            .successors()
            .filter_map(move |target| target.slot().filter(|&slot| slot < count))
    }

    fn new(blocks: &[Block]) -> Self {
        let count = blocks.len();
        let mut preds = vec![Vec::new(); count];
        for (index, block) in blocks.iter().enumerate() {
            for target in Self::successors(block, count) {
                if let Some(list) = preds.get_mut(target)
                    && !list.contains(&index)
                {
                    list.push(index);
                }
            }
        }
        for list in &mut preds {
            list.sort_unstable();
        }

        // Postorder from the entry, with an explicit stack; each block is pushed once.
        let mut postorder = vec![None; count];
        let mut order = Vec::new();
        let mut visited = vec![false; count];
        let mut stack: Vec<(usize, usize)> = Vec::new();
        if let Some(entry) = visited.first_mut() {
            *entry = true;
            stack.push((0, 0));
        }
        while let Some(&mut (block, ref mut next)) = stack.last_mut() {
            let successor = blocks
                .get(block)
                .and_then(|b| Self::successors(b, count).nth(*next));
            if let Some(target) = successor {
                *next = next.saturating_add(1);
                if let Some(seen) = visited.get_mut(target)
                    && !*seen
                {
                    *seen = true;
                    stack.push((target, 0));
                }
            } else {
                stack.pop();
                if let Some(number) = postorder.get_mut(block) {
                    *number = Some(order.len());
                }
                order.push(block);
            }
        }

        let mut graph = Self {
            preds,
            postorder,
            idom: vec![None; count],
        };
        if let Some(entry) = graph.idom.first_mut() {
            *entry = Some(0);
        }
        // Each round can only move an immediate dominator up the tree, so this settles within
        // one round per block; the bound only guards the loop.
        for _ in 0..=count {
            let mut changed = false;
            for &block in order.iter().rev().filter(|&&block| block != 0) {
                let mut new = None;
                let preds = graph.preds.get(block).cloned().unwrap_or_default();
                for pred in preds {
                    if graph.idom.get(pred).copied().flatten().is_none() {
                        continue;
                    }
                    new = Some(match new {
                        None => pred,
                        Some(other) => graph.intersect(pred, other),
                    });
                }
                if let Some(slot) = graph.idom.get_mut(block)
                    && *slot != new
                {
                    *slot = new;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        graph
    }

    fn number(&self, block: usize) -> usize {
        self.postorder.get(block).copied().flatten().unwrap_or(0)
    }

    fn parent(&self, block: usize) -> usize {
        self.idom.get(block).copied().flatten().unwrap_or(0)
    }

    /// The nearest common dominator of two processed blocks. Postorder numbers grow toward the
    /// entry along immediate dominators, so each step strictly climbs.
    fn intersect(&self, mut a: usize, mut b: usize) -> usize {
        while a != b {
            while self.number(a) < self.number(b) {
                let up = self.parent(a);
                if up == a {
                    return a;
                }
                a = up;
            }
            while self.number(b) < self.number(a) {
                let up = self.parent(b);
                if up == b {
                    return b;
                }
                b = up;
            }
        }
        a
    }

    fn reachable(&self, block: usize) -> bool {
        self.postorder.get(block).copied().flatten().is_some()
    }

    /// Whether `a` dominates `b`. A block the entry cannot reach is dominated by every block.
    fn dominates(&self, a: usize, b: usize) -> bool {
        if !self.reachable(b) {
            return true;
        }
        if !self.reachable(a) {
            return false;
        }
        let mut current = b;
        // The chain from `b` to the entry visits each block at most once.
        for _ in 0..=self.idom.len() {
            if current == a {
                return true;
            }
            let up = self.parent(current);
            if up == current {
                return false;
            }
            current = up;
        }
        false
    }
}

/// Step 4 for a control-flow graph.
struct Cfg<'a> {
    function: &'a Function,
    sites: &'a [Option<(usize, usize)>],
    graph: &'a Graph,
}

impl Cfg<'_> {
    fn check(&self) -> Result<(), VerifyError> {
        for (block_index, block) in self.function.blocks.iter().enumerate() {
            let mut seen_operation = false;
            for (position, &value) in block.insts.iter().enumerate() {
                match self.function.inst(value) {
                    Some(Inst::Op(op)) => {
                        seen_operation = true;
                        for &operand in &op.args {
                            self.defined(Site::Inst(value), operand, block_index, position)?;
                        }
                        check_operation(self.function, value, op)?;
                    }
                    Some(Inst::Phi(phi)) => {
                        if seen_operation {
                            return Err(VerifyError::PhiAfterOperation(value));
                        }
                        self.phi(value, phi, block_index)?;
                    }
                    None => return Err(VerifyError::Missing(value)),
                }
            }
            if let Terminator::Branch { cond, .. } = block.terminator {
                let id = BlockId(u32::try_from(block_index).unwrap_or(u32::MAX));
                let at = Site::Terminator(id);
                self.defined(at, cond, block_index, block.insts.len())?;
                check_operand(self.function, at, 0, cond, Type::U1)?;
            }
        }
        Ok(())
    }

    /// Whether `operand` is defined at position `position` of block `block`.
    fn defined(
        &self,
        at: Site,
        operand: Operand,
        block: usize,
        position: usize,
    ) -> Result<(), VerifyError> {
        let Operand::Value(value) = operand else {
            return Ok(());
        };
        let site = value.slot().and_then(|slot| self.sites.get(slot)).copied();
        let ok = match site.flatten() {
            None => false,
            Some((def_block, def_position)) if def_block == block => def_position < position,
            Some((def_block, _)) => self.graph.dominates(def_block, block),
        };
        if ok {
            Ok(())
        } else {
            Err(VerifyError::Undefined { at, operand: value })
        }
    }

    fn phi(&self, at: Value, phi: &Phi, block: usize) -> Result<(), VerifyError> {
        if !phi.ty.is_value() {
            return Err(VerifyError::PhiType { at, ty: phi.ty });
        }
        let preds = self.graph.preds.get(block).map_or(&[][..], Vec::as_slice);
        let mut seen = Vec::new();
        for (index, &(from, operand)) in phi.incoming.iter().enumerate() {
            let Some(slot) = from
                .slot()
                .filter(|&slot| slot < self.function.blocks.len())
            else {
                return Err(VerifyError::NoSuchBlock {
                    at: Site::Inst(at),
                    block: from,
                });
            };
            if seen.contains(&slot) {
                return Err(VerifyError::PhiRepeated { at, block: from });
            }
            seen.push(slot);
            if !preds.contains(&slot) {
                return Err(VerifyError::PhiNotPredecessor { at, block: from });
            }
            // The value must be available at the end of the predecessor.
            let end = self
                .function
                .blocks
                .get(slot)
                .map_or(0, |pred| pred.insts.len());
            self.defined(Site::Inst(at), operand, slot, end)?;
            check_operand(self.function, Site::Inst(at), index, operand, phi.ty)?;
        }
        if let Some(&missing) = preds.iter().find(|pred| !seen.contains(pred)) {
            return Err(VerifyError::PhiMissingPredecessor {
                at,
                block: BlockId(u32::try_from(missing).unwrap_or(u32::MAX)),
            });
        }
        Ok(())
    }
}

/// Step 4 for a structured function: a walk of the tree with the values in scope.
struct Structured<'a> {
    function: &'a Function,
    sites: &'a [Option<(usize, usize)>],
    /// Whether each value is in scope.
    visible: Vec<bool>,
    /// The values brought into scope, in order, so a body's can be removed when it ends.
    log: Vec<Value>,
}

impl Structured<'_> {
    /// Walks `nodes`; step 3 has bounded the nesting by [`MAX_NESTING`].
    fn nodes(&mut self, nodes: &[Node]) -> Result<(), VerifyError> {
        for node in nodes {
            match node {
                Node::Block(block) => self.block(*block)?,
                Node::If { cond, body } => {
                    self.condition(*cond)?;
                    self.scoped(body, None)?;
                }
                Node::Loop { body, repeat } => self.scoped(body, Some(*repeat))?,
                Node::Break { cond } => self.condition(*cond)?,
                Node::Return | Node::Unreachable => {}
            }
        }
        Ok(())
    }

    /// Walks a body, checks the repeat condition at its end, then takes its values out of
    /// scope.
    fn scoped(&mut self, body: &[Node], repeat: Option<Operand>) -> Result<(), VerifyError> {
        let mark = self.log.len();
        self.nodes(body)?;
        if let Some(repeat) = repeat {
            self.condition(repeat)?;
        }
        for value in self.log.drain(mark..) {
            if let Some(slot) = value.slot().and_then(|slot| self.visible.get_mut(slot)) {
                *slot = false;
            }
        }
        Ok(())
    }

    fn condition(&self, cond: Operand) -> Result<(), VerifyError> {
        self.defined(Site::Structure, cond)?;
        check_operand(self.function, Site::Structure, 0, cond, Type::U1)
    }

    fn defined(&self, at: Site, operand: Operand) -> Result<(), VerifyError> {
        match operand {
            Operand::Value(value)
                if !value
                    .slot()
                    .and_then(|slot| self.visible.get(slot))
                    .copied()
                    .unwrap_or(false) =>
            {
                Err(VerifyError::Undefined { at, operand: value })
            }
            _ => Ok(()),
        }
    }

    fn block(&mut self, block: BlockId) -> Result<(), VerifyError> {
        let insts = self
            .function
            .block(block)
            .map_or(&[][..], |b| b.insts.as_slice());
        let mut seen_operation = false;
        for &value in insts {
            match self.function.inst(value) {
                Some(Inst::Op(op)) => {
                    seen_operation = true;
                    for &operand in &op.args {
                        self.defined(Site::Inst(value), operand)?;
                    }
                    check_operation(self.function, value, op)?;
                }
                Some(Inst::Phi(phi)) => {
                    if seen_operation {
                        return Err(VerifyError::PhiAfterOperation(value));
                    }
                    self.phi(value, phi)?;
                }
                None => return Err(VerifyError::Missing(value)),
            }
            if let Some(slot) = value.slot().and_then(|slot| self.visible.get_mut(slot)) {
                *slot = true;
            }
            self.log.push(value);
        }
        Ok(())
    }

    /// A phi in a structured function: its incoming values need only be listed somewhere.
    fn phi(&self, at: Value, phi: &Phi) -> Result<(), VerifyError> {
        if !phi.ty.is_value() {
            return Err(VerifyError::PhiType { at, ty: phi.ty });
        }
        let mut seen = Vec::new();
        for (index, &(from, operand)) in phi.incoming.iter().enumerate() {
            if self.function.block(from).is_none() {
                return Err(VerifyError::NoSuchBlock {
                    at: Site::Inst(at),
                    block: from,
                });
            }
            if seen.contains(&from) {
                return Err(VerifyError::PhiRepeated { at, block: from });
            }
            seen.push(from);
            if let Operand::Value(value) = operand {
                let listed = value
                    .slot()
                    .and_then(|slot| self.sites.get(slot))
                    .is_some_and(Option::is_some);
                if !listed {
                    return Err(VerifyError::Undefined {
                        at: Site::Inst(at),
                        operand: value,
                    });
                }
            }
            check_operand(self.function, Site::Inst(at), index, operand, phi.ty)?;
        }
        Ok(())
    }
}
