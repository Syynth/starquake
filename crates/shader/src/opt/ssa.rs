//! The SSA rewrite: guest state read and written through get and set operations becomes SSA
//! values and phis, with the algorithm of Braun et al., "Simple and Efficient Construction of
//! Static Single Assignment Form" (CC 2013).
//!
//! Ported by behaviour from Eden's `eden/src/shader_recompiler/ir_opt/ssa_rewrite_pass.cpp` @
//! cf654052 (GPL-3.0-or-later, over GPL-2.0-or-later yuzu code). As there:
//!
//! - The variables are the per-thread registers (but `RZ`), the predicates (but `PT`), the
//!   four condition-code flags, the goto variables (by index) and the indirect branch variable;
//!   here also the uniform registers (but `URZ`) and the uniform predicates (but `UPT`), the
//!   uniform register class of the owner's decision 1 (Eden packs uniform registers into its
//!   per-thread file and has no such opcodes). A set operation records the value it writes; a
//!   get operation becomes an `Identity` of the value it reads (Eden's `ReplaceUsesWith`); the
//!   set operations stay for dead code elimination to remove.
//! - A read looks the variable up in its block, then in the single predecessor, else places
//!   an operandless phi to break cycles and fills it from each predecessor (Eden's
//!   `ReadVariable`, an explicit stack rather than recursion, so a long chain of blocks cannot
//!   overflow the host's stack). A phi in a block not yet sealed is incomplete and gets its
//!   operands when the block is sealed (`SealBlock`, `AddPhiOperands`).
//! - A phi whose operands are all one value or itself is trivial: it becomes an `Identity` of
//!   that value, or of a new `Undef` of the variable's type when it has none (the entry block,
//!   or no predecessors) (`TryRemoveTrivialPhi`). As in Eden, users of a removed phi are not
//!   rechecked ("TODO: Try to recursively remove all phi users").
//! - Phis are typed by the variable (`u32` for registers and the indirect branch variable,
//!   `u1` for the rest; Eden's `UndefOpcode`), and their incoming pairs are in block order
//!   (Eden's `OrderPhiArgs`); Eden's `GetConcreteType` fix-up for `Opaque` phis has no
//!   counterpart because no phi is created without a type.
//! - A read of `RZ`, `URZ`, `PT` or `UPT` is left as it is (constant propagation folds it,
//!   `constant` module), and a write to one is ignored. The verifier refuses both
//!   (`ir`'s `ReservedName`), so a verified function has none.
//!
//! Where this pass departs from Eden's, each pinned by a test in `tests`:
//!
//! - **Sealing.** Eden seals each block right after visiting it, in reverse post order. On a
//!   control-flow graph whose back edge leaves a block that writes the variable (a loop of one
//!   block, which flow analysis builds, slice 10), that loses the loop-carried value: the
//!   loop header's phi reads the back-edge block before the block is filled, and gets the
//!   value at its start. Eden's structurizer always ends a loop with a block of its own, which
//!   its translators never write a register in, so Eden does not meet this. Here a block is
//!   sealed once all its predecessors are filled, as Braun et al. require: before it is
//!   filled when its predecessors already are, else as soon as the last one is
//!   (`tests::ssa::a_self_loop_keeps_its_loop_carried_value`).
//! - **Unreachable blocks.** Eden removes them first; this IR keeps them, so they are filled
//!   after the reachable ones, in arena order, and a reachable block with an unreachable
//!   predecessor is sealed then (`tests::ssa::unreachable_blocks_are_rewritten`).
//! - **An entry with predecessors.** Eden's programs start with a block nothing jumps to; flow
//!   analysis can make block 0 a loop header. A phi there would have no value on entry (the
//!   interpreter's `PhiAtStart`), so when block 0 has predecessors its instructions and
//!   terminator move to a new last block and block 0 jumps to it; terminators and phis that
//!   named block 0 name the new block. An input phi in such a block 0 is refused
//!   ([`PassError::EntryPhi`]) before anything changes
//!   (`tests::ssa::an_entry_loop_moves_to_its_own_block`, `tests::ssa::entry_phis_are_refused`).
//! - **A trivial phi's value** is its operands' common value resolved through `Identity`
//!   chains; Eden keeps the first operand as written, which can be an `Identity` in one
//!   predecessor of a join, where it does not dominate the join's users. The resolved value
//!   does: it reaches the block along every edge. Eden's backend resolves identities, so the
//!   two mean the same; the verifier here checks dominance on the operand as written
//!   (`tests::ssa::a_trivial_join_takes_the_resolved_value`).
//! - **Placement.** Phis created here come first in their block, in creation order (Eden
//!   prepends each, so its order is the reverse); then the `Undef`s created for trivial phis;
//!   then the trivial phis, now `Identity` operations, in removal order, each with its
//!   argument resolved through any chain of removed phis to the value it stands for (Eden
//!   reinserts each after the phis, newest first, and keeps the chain). Resolving keeps every
//!   use after its definition in a block, which the verifier checks and Eden's order does not
//!   guarantee.
//! - **Structured functions** are refused ([`PassError::Structured`]). Eden runs the rewrite
//!   after structuring, on predecessors its structurizer records; in this IR a structured
//!   block has no terminator, so its predecessors would have to come from the tree, and the
//!   structured form's scope rule (`ir` module docs) makes the structurizer (slice 11)
//!   responsible for values that cross a body. Which order the pipeline uses is an open
//!   question for the owner; the rewrite runs on the control-flow graph that slice 10 builds
//!   (`tests::ssa::structured_functions_are_refused`).
//! - A goto variable named by a value rather than an immediate is
//!   [`PassError::GotoIndex`] (Eden's `Arg(0).U32()` throws on one)
//!   (`tests::ssa::goto_indices_must_be_immediates`).

use std::collections::BTreeMap;

use crate::ir::{BlockId, Function, Imm, Inst, Opcode, Operand, Phi, Terminator, Type, Value};

use super::{PassError, Work, arena_len, block_order, identity_of, predecessors, slot};

/// A variable the rewrite turns into SSA values.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Var {
    /// A per-thread register, by index.
    Reg(u8),
    /// A predicate, by index.
    Pred(u8),
    /// A uniform register, by index.
    UniformReg(u8),
    /// A uniform predicate, by index.
    UniformPred(u8),
    /// The zero flag.
    Zero,
    /// The sign flag.
    Sign,
    /// The carry flag.
    Carry,
    /// The overflow flag.
    Overflow,
    /// A goto variable, by index.
    Goto(u32),
    /// The indirect branch variable.
    IndirectBranch,
}

impl Var {
    /// The `Undef` opcode of the variable's type (Eden's `UndefOpcode`).
    const fn undef(self) -> Opcode {
        match self {
            Self::Reg(_) | Self::UniformReg(_) | Self::IndirectBranch => Opcode::UndefU32,
            Self::Pred(_)
            | Self::UniformPred(_)
            | Self::Zero
            | Self::Sign
            | Self::Carry
            | Self::Overflow
            | Self::Goto(_) => Opcode::UndefU1,
        }
    }

    /// The variable's type.
    const fn ty(self) -> Type {
        self.undef().result()
    }
}

/// What an instruction does to a variable.
enum Access {
    /// Reads it.
    Read(Var),
    /// Writes the operand to it.
    Write(Var, Operand),
}

/// The variable a get or set operation's name argument names; `None` for the reserved names
/// (`RZ`, `URZ`, `PT`, `UPT`) and for a malformed argument.
fn named(arg: Option<&Operand>) -> Option<Var> {
    match arg?.imm()? {
        Imm::Reg(reg) if !reg.is_zero() => Some(Var::Reg(reg.index())),
        Imm::Pred(pred) if !pred.is_true() => Some(Var::Pred(pred.index())),
        Imm::UniformReg(reg) if !reg.is_zero() => Some(Var::UniformReg(reg.index())),
        Imm::UniformPred(pred) if !pred.is_true() => Some(Var::UniformPred(pred.index())),
        _ => None,
    }
}

/// The goto variable an operation's first argument names.
fn goto(at: Value, args: &[Operand]) -> Result<Option<Var>, PassError> {
    match args.first() {
        Some(Operand::Imm(Imm::U32(index))) => Ok(Some(Var::Goto(*index))),
        Some(Operand::Value(_)) => Err(PassError::GotoIndex(at)),
        Some(Operand::Imm(_)) | None => Ok(None),
    }
}

/// What instruction `at` does to a variable (Eden's `VisitInst`), if anything.
fn access(at: Value, inst: &Inst) -> Result<Option<Access>, PassError> {
    let Inst::Op(op) = inst else {
        return Ok(None);
    };
    let args = op.args.as_slice();
    let second = || args.get(1).copied();
    let first = || args.first().copied();
    let write = |var: Option<Var>, value: Option<Operand>| match (var, value) {
        (Some(var), Some(value)) => Some(Access::Write(var, value)),
        _ => None,
    };
    let flag = |var: Var| write(Some(var), first());
    Ok(match op.opcode {
        Opcode::GetRegister
        | Opcode::GetPred
        | Opcode::GetUniformRegister
        | Opcode::GetUniformPred => named(args.first()).map(Access::Read),
        Opcode::SetRegister
        | Opcode::SetPred
        | Opcode::SetUniformRegister
        | Opcode::SetUniformPred => write(named(args.first()), second()),
        Opcode::GetGotoVariable => goto(at, args)?.map(Access::Read),
        Opcode::SetGotoVariable => write(goto(at, args)?, second()),
        Opcode::GetIndirectBranchVariable => Some(Access::Read(Var::IndirectBranch)),
        Opcode::SetIndirectBranchVariable => flag(Var::IndirectBranch),
        Opcode::GetZFlag => Some(Access::Read(Var::Zero)),
        Opcode::GetSFlag => Some(Access::Read(Var::Sign)),
        Opcode::GetCFlag => Some(Access::Read(Var::Carry)),
        Opcode::GetOFlag => Some(Access::Read(Var::Overflow)),
        Opcode::SetZFlag => flag(Var::Zero),
        Opcode::SetSFlag => flag(Var::Sign),
        Opcode::SetCFlag => flag(Var::Carry),
        Opcode::SetOFlag => flag(Var::Overflow),
        _ => None,
    })
}

/// Where a read's stack frame is (Eden's `Status`).
#[derive(Clone, Copy)]
enum Pc {
    /// Look the variable up.
    Start,
    /// Record the result as the block's definition and return it.
    SetValue,
    /// Add the predecessor's result to the phi, then go on with the next predecessor.
    PushPhiArgument,
}

/// One frame of a read (Eden's `ReadState`).
struct Frame {
    block: BlockId,
    /// The value found; set before it is read (the initial value is never used).
    result: Operand,
    phi: Option<Value>,
    /// The predecessor being read, an index into the block's predecessors.
    pred: usize,
    pc: Pc,
}

impl Frame {
    const fn new(block: BlockId) -> Self {
        Self {
            block,
            result: Operand::Imm(Imm::U1(false)),
            phi: None,
            pred: 0,
            pc: Pc::Start,
        }
    }
}

/// The rewrite's state over one function.
struct Ssa<'f> {
    function: &'f mut Function,
    preds: Vec<Vec<BlockId>>,
    /// Each block's current definition of each variable (Eden's `DefTable`).
    defs: Vec<BTreeMap<Var, Operand>>,
    /// Each block's incomplete phis (Eden's `incomplete_phis`).
    incomplete: Vec<BTreeMap<Var, Value>>,
    sealed: Vec<bool>,
    /// Each block's phis made here, in creation order.
    phis: Vec<Vec<Value>>,
    /// Each block's `Undef`s made for trivial phis.
    undefs: Vec<Vec<Value>>,
    /// Each block's trivial phis, now `Identity` operations, in removal order.
    removed: Vec<Vec<Value>>,
    work: Work,
}

/// Runs the rewrite (see the module docs).
pub(super) fn run(function: &mut Function) -> Result<Work, PassError> {
    if function.structure().is_some() {
        return Err(PassError::Structured);
    }
    let mut work = Work::default();
    split_entry(function, &mut work)?;
    let order = block_order(function, &mut work);
    let preds = predecessors(function);
    let count = function.block_count();
    // Take each block's list: instructions made below are appended to the emptied lists,
    // which the final placement replaces.
    let ids: Vec<BlockId> = function.blocks().map(|(id, _)| id).collect();
    let mut original = Vec::with_capacity(count);
    for &id in &ids {
        let list = function
            .block_mut(id)
            .map(|block| std::mem::take(&mut block.insts))
            .unwrap_or_default();
        original.push(list);
    }
    let mut ssa = Ssa {
        function,
        preds,
        defs: vec![BTreeMap::new(); count],
        incomplete: vec![BTreeMap::new(); count],
        sealed: vec![false; count],
        phis: vec![Vec::new(); count],
        undefs: vec![Vec::new(); count],
        removed: vec![Vec::new(); count],
        work,
    };
    let mut filled = vec![false; count];
    let mut waiting: Vec<usize> = ssa.preds.iter().map(Vec::len).collect();
    for &block in &order {
        let index = slot(block).unwrap_or(usize::MAX);
        if waiting.get(index) == Some(&0) {
            ssa.seal(block)?;
        }
        for &value in original.get(index).map_or(&[][..], Vec::as_slice) {
            ssa.visit(block, value)?;
        }
        if let Some(done) = filled.get_mut(index) {
            *done = true;
        }
        let successors: Vec<BlockId> = ssa
            .function
            .block(block)
            .map(|b| b.terminator.successors().collect())
            .unwrap_or_default();
        let mut previous = None;
        for to in successors {
            if previous == Some(to) {
                continue;
            }
            previous = Some(to);
            let Some(left) = slot(to).and_then(|i| waiting.get_mut(i)) else {
                continue;
            };
            *left = left.saturating_sub(1);
            if *left == 0 && slot(to).and_then(|i| filled.get(i)) == Some(&true) {
                ssa.seal(to)?;
            }
        }
    }
    for &block in &order {
        ssa.seal(block)?;
    }
    Ok(ssa.place(&ids, original))
}

impl Ssa<'_> {
    /// Visits instruction `value` of `block` (Eden's `VisitInst`).
    fn visit(&mut self, block: BlockId, value: Value) -> Result<(), PassError> {
        self.work.step();
        let inst = self.function.inst(value).ok_or(PassError::Missing(value))?;
        match access(value, inst)? {
            Some(Access::Write(var, operand)) => self.write(var, block, operand),
            Some(Access::Read(var)) => {
                let found = self.read(var, block)?;
                if let Some(slot) = self.function.inst_mut(value) {
                    *slot = identity_of(found);
                }
            }
            None => {}
        }
        Ok(())
    }

    fn write(&mut self, var: Var, block: BlockId, operand: Operand) {
        if let Some(defs) = slot(block).and_then(|index| self.defs.get_mut(index)) {
            defs.insert(var, operand);
        }
    }

    fn def(&self, var: Var, block: BlockId) -> Option<Operand> {
        slot(block)
            .and_then(|index| self.defs.get(index))
            .and_then(|defs| defs.get(&var).copied())
    }

    fn is_sealed(&self, block: BlockId) -> bool {
        slot(block)
            .and_then(|index| self.sealed.get(index))
            .copied()
            .unwrap_or(true)
    }

    fn preds_of(&self, block: BlockId) -> &[BlockId] {
        slot(block)
            .and_then(|index| self.preds.get(index))
            .map_or(&[], Vec::as_slice)
    }

    /// A new operandless phi of `var`'s type in `block`.
    fn new_phi(&mut self, var: Var, block: BlockId) -> Result<Value, PassError> {
        let phi = self.function.push(
            block,
            Inst::Phi(Phi {
                ty: var.ty(),
                incoming: Vec::new(),
            }),
        )?;
        if let Some(list) = slot(block).and_then(|index| self.phis.get_mut(index)) {
            list.push(phi);
        }
        Ok(phi)
    }

    /// The value of `var` at the end of `root` so far (Eden's `ReadVariable`).
    fn read(&mut self, var: Var, root: BlockId) -> Result<Operand, PassError> {
        // The bottom frame only collects the result.
        let mut stack = vec![Frame::new(root), Frame::new(root)];
        while stack.len() > 1 {
            self.work.step();
            let Some(top) = stack.last_mut() else { break };
            let block = top.block;
            match top.pc {
                Pc::Start => {
                    if let Some(def) = self.def(var, block) {
                        top.result = def;
                    } else if !self.is_sealed(block) {
                        // An incomplete control-flow graph.
                        let phi = self.new_phi(var, block)?;
                        if let Some(map) = slot(block).and_then(|i| self.incomplete.get_mut(i)) {
                            map.insert(var, phi);
                        }
                        if let Some(top) = stack.last_mut() {
                            top.result = Operand::Value(phi);
                        }
                    } else if let [pred] = *self.preds_of(block) {
                        // One predecessor: no phi needed.
                        top.pc = Pc::SetValue;
                        stack.push(Frame::new(pred));
                        continue;
                    } else {
                        // Break potential cycles with an operandless phi.
                        let phi = self.new_phi(var, block)?;
                        self.write(var, block, Operand::Value(phi));
                        if let Some(top) = stack.last_mut() {
                            top.phi = Some(phi);
                            top.pred = 0;
                        }
                        self.next_phi_operand(var, &mut stack)?;
                        continue;
                    }
                    self.set_value(var, &mut stack);
                }
                Pc::SetValue => self.set_value(var, &mut stack),
                Pc::PushPhiArgument => {
                    let result = top.result;
                    let from = self.preds_of(block).get(top.pred).copied();
                    let phi = top.phi;
                    top.pred = top.pred.saturating_add(1);
                    if let (Some(phi), Some(from)) = (phi, from)
                        && let Some(Inst::Phi(phi)) = self.function.inst_mut(phi)
                    {
                        phi.incoming.push((from, result));
                    }
                    self.next_phi_operand(var, &mut stack)?;
                }
            }
        }
        Ok(stack.first().map_or(Operand::Imm(Imm::U1(false)), |f| f.result))
    }

    /// Records the top frame's result for its block and returns it to the frame below.
    fn set_value(&mut self, var: Var, stack: &mut Vec<Frame>) {
        if let Some(top) = stack.pop() {
            self.write(var, top.block, top.result);
            if let Some(below) = stack.last_mut() {
                below.result = top.result;
            }
        }
    }

    /// Reads the top frame's next predecessor for its phi, or, when every predecessor is read,
    /// tries to remove the phi and returns the result (Eden's `prepare_phi_operand`).
    fn next_phi_operand(&mut self, var: Var, stack: &mut Vec<Frame>) -> Result<(), PassError> {
        let Some(top) = stack.last_mut() else {
            return Ok(());
        };
        let block = top.block;
        let next = slot(block)
            .and_then(|index| self.preds.get(index))
            .and_then(|preds| preds.get(top.pred))
            .copied();
        match (next, top.phi) {
            (Some(pred), _) => {
                top.pc = Pc::PushPhiArgument;
                stack.push(Frame::new(pred));
            }
            (None, Some(phi)) => {
                let result = self.try_remove_trivial_phi(phi, block, var)?;
                stack.pop();
                if let Some(below) = stack.last_mut() {
                    below.result = result;
                }
                self.write(var, block, result);
            }
            (None, None) => {
                stack.pop();
            }
        }
        Ok(())
    }

    /// Seals `block` (Eden's `SealBlock`): fills its incomplete phis from its predecessors.
    fn seal(&mut self, block: BlockId) -> Result<(), PassError> {
        let Some(index) = slot(block) else {
            return Ok(());
        };
        if self.sealed.get(index) != Some(&false) {
            return Ok(());
        }
        let pending = self
            .incomplete
            .get_mut(index)
            .map(std::mem::take)
            .unwrap_or_default();
        for (var, phi) in pending {
            self.add_phi_operands(var, phi, block)?;
        }
        if let Some(sealed) = self.sealed.get_mut(index) {
            *sealed = true;
        }
        Ok(())
    }

    /// Eden's `AddPhiOperands`.
    fn add_phi_operands(&mut self, var: Var, phi: Value, block: BlockId) -> Result<(), PassError> {
        let preds = self.preds_of(block).to_vec();
        let mut incoming = Vec::with_capacity(preds.len());
        for pred in preds {
            self.work.step();
            incoming.push((pred, self.read(var, pred)?));
        }
        if let Some(Inst::Phi(target)) = self.function.inst_mut(phi) {
            target.incoming.extend(incoming);
        }
        self.try_remove_trivial_phi(phi, block, var)?;
        Ok(())
    }

    /// Eden's `TryRemoveTrivialPhi`: the value the phi stands for.
    fn try_remove_trivial_phi(
        &mut self,
        phi: Value,
        block: BlockId,
        var: Var,
    ) -> Result<Operand, PassError> {
        let mut same: Option<Operand> = None;
        if let Some(Inst::Phi(node)) = self.function.inst(phi) {
            for &(_, operand) in &node.incoming {
                self.work.step();
                let resolved = resolve(self.function, operand);
                if same == Some(resolved) || operand == Operand::Value(phi) {
                    // A unique value or a self-reference.
                    continue;
                }
                if same.is_some() {
                    // The phi merges at least two values: not trivial.
                    return Ok(Operand::Value(phi));
                }
                // The resolved value, not the operand as written (see the module docs).
                same = Some(resolved);
            }
        } else {
            return Ok(Operand::Value(phi));
        }
        let index = slot(block).unwrap_or(usize::MAX);
        let same = match same {
            Some(same) => same,
            None => {
                // The phi is unreachable or in the entry block.
                let undef = self.function.push(
                    block,
                    Inst::Op(crate::ir::Operation::new(var.undef(), Vec::new())),
                )?;
                if let Some(list) = self.undefs.get_mut(index) {
                    list.push(undef);
                }
                Operand::Value(undef)
            }
        };
        if let Some(inst) = self.function.inst_mut(phi) {
            *inst = identity_of(same);
        }
        if let Some(list) = self.removed.get_mut(index) {
            list.push(phi);
        }
        Ok(same)
    }

    /// Writes each block's final list: its remaining phis, its `Undef`s, its removed phis
    /// (resolved), then its original instructions.
    fn place(mut self, ids: &[BlockId], original: Vec<Vec<Value>>) -> Work {
        for ((index, list), &id) in original.into_iter().enumerate().zip(ids) {
            let phis = self.phis.get(index).map_or(&[][..], Vec::as_slice);
            let undefs = self.undefs.get(index).map_or(&[][..], Vec::as_slice);
            let removed = self.removed.get(index).map_or(&[][..], Vec::as_slice);
            let mut insts =
                Vec::with_capacity(phis.len() + undefs.len() + removed.len() + list.len());
            insts.extend(
                phis.iter()
                    .copied()
                    .filter(|&phi| self.function.inst(phi).is_some_and(Inst::is_phi)),
            );
            // The block's own phis stay ahead of every operation.
            let leading = list
                .iter()
                .take_while(|&&value| self.function.inst(value).is_some_and(Inst::is_phi))
                .count();
            let (own_phis, list) = list.split_at(leading);
            insts.extend_from_slice(own_phis);
            insts.extend_from_slice(undefs);
            for &value in removed {
                self.work.step();
                let target = match self.function.inst(value) {
                    Some(Inst::Op(op)) => op.args.first().copied(),
                    _ => None,
                };
                if let Some(target) = target {
                    let resolved = resolve(self.function, target);
                    if let Some(inst) = self.function.inst_mut(value) {
                        *inst = identity_of(resolved);
                    }
                }
                insts.push(value);
            }
            insts.extend(list);
            if let Some(block) = self.function.block_mut(id) {
                block.insts = insts;
            }
        }
        self.work
    }
}

/// `operand` with any chain of `Identity` operations followed to its end (Eden's
/// `Value::Resolve`); a chain that loops or ends at a missing id stops where it is, at most
/// once per instruction in the arena.
pub(super) fn resolve(function: &Function, operand: Operand) -> Operand {
    let mut current = operand;
    for _ in 0..=arena_len(function) {
        let Operand::Value(value) = current else {
            return current;
        };
        match function.inst(value) {
            Some(Inst::Op(op)) if op.opcode == Opcode::Identity => match op.args.first() {
                Some(&next) => current = next,
                None => return current,
            },
            _ => return current,
        }
    }
    current
}

/// Moves block 0's instructions and terminator to a new last block when other blocks jump to
/// block 0 (see the module docs).
fn split_entry(function: &mut Function, work: &mut Work) -> Result<(), PassError> {
    let entry = function.entry();
    let preds = predecessors(function);
    if preds.first().is_none_or(Vec::is_empty) {
        return Ok(());
    }
    if let Some(block) = function.block(entry) {
        for &value in &block.insts {
            if function.inst(value).is_some_and(Inst::is_phi) {
                return Err(PassError::EntryPhi(value));
            }
        }
    }
    let new = function.add_block()?;
    let moved = function
        .block_mut(entry)
        .map(std::mem::take)
        .unwrap_or_default();
    if let Some(block) = function.block_mut(new) {
        *block = moved;
    }
    if let Some(block) = function.block_mut(entry) {
        block.terminator = Terminator::Jump(new);
    }
    let retarget = |to: BlockId| if to == entry { new } else { to };
    let ids: Vec<BlockId> = function.blocks().map(|(id, _)| id).collect();
    for id in ids {
        work.step();
        if id == entry {
            continue;
        }
        let Some(block) = function.block_mut(id) else {
            continue;
        };
        block.terminator = match block.terminator {
            Terminator::Jump(to) => Terminator::Jump(retarget(to)),
            Terminator::Branch {
                cond,
                then,
                otherwise,
            } => Terminator::Branch {
                cond,
                then: retarget(then),
                otherwise: retarget(otherwise),
            },
            other => other,
        };
    }
    let listed: Vec<Value> = function
        .listed()
        .filter(|(_, inst)| inst.is_phi())
        .map(|(value, _)| value)
        .collect();
    for value in listed {
        work.step();
        if let Some(Inst::Phi(phi)) = function.inst_mut(value) {
            for (from, _) in &mut phi.incoming {
                *from = retarget(*from);
            }
        }
    }
    Ok(())
}
