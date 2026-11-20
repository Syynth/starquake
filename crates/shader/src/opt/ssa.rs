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
//! # Structured functions
//!
//! The rewrite runs after the structurizer, on the structured function, in Eden's order (the
//! owner's ruling of 2026-11-19, decision log, "Owner rulings on the wave 44 and 45 questions",
//! row 3: `frontend/maxwell/translate_program.cpp` @ cf654052 calls `BuildASL` at line 337 and
//! `SsaRewritePass` at line 389). It also still takes a control-flow graph, as slice 8 built it
//! (the departures below that name a graph are about that form). On a structured function:
//!
//! - **Predecessors come from the tree** (the `tree` submodule). Eden's structurizer records
//!   them as it emits the tree (`frontend/maxwell/structured_control_flow.cpp`,
//!   `TranslatePass::Visit`, each `AddBranch`), and its rewrite reads them
//!   (`ImmPredecessors`). Here a structured block has no terminator, so the edges are derived
//!   by the structured-phi rule the interpreter and the structurizer use (`crate::interp`: a
//!   phi takes the value from the block run last): a block's predecessors are the blocks that
//!   can have run last before it. An `if`'s condition may be false and a `break`'s may be
//!   false, so both pass on the blocks before them; a loop's first block also follows the end
//!   of its body; after a loop come the end of its body and every `break`'s blocks; after a
//!   `return` or `unreachable` nothing has run. Immediate conditions are not looked at: an edge
//!   that cannot be taken only adds a phi operand. The tree's first block is the entry
//!   (`tests::structured::a_structured_if_merges_with_a_phi`,
//!   `tests::structured::a_loop_carries_its_value_in_a_header_phi_and_out_in_a_phi`,
//!   `tests::structured::a_break_and_the_repeat_both_reach_the_block_after_the_loop`,
//!   `tests::structured::blocks_after_a_return_read_undef`).
//! - **Blocks are added where paths meet** and the tree has none, where Eden's structurizer has
//!   a block of its own: an empty block at the start of the tree when it does not start with
//!   one (Eden's `PostOrder` throws "First node in abstract syntax list root is not a block"),
//!   at the start of each loop body that does not start with one (Eden's loop header block),
//!   and after each `if` and loop that is followed by anything but a block, a `return` or an
//!   `unreachable`, or by nothing (Eden's merge block). So each phi has a block to go in, the
//!   tree's first block is never a loop's, and predecessor lists stay short: without the merge
//!   blocks, the k-th of a run of `if`s would have k predecessors
//!   (`tests::structured::blocks_are_added_where_paths_meet`,
//!   `tests::structured::a_return_or_unreachable_after_an_if_adds_no_block`,
//!   `tests::structured::an_empty_loop_body_gets_a_header`,
//!   `tests::linear::a_tree_with_no_block_where_paths_meet_costs_its_size`). Departure: Eden
//!   adds these blocks always, its loop header goes ahead of the loop (its loop node does not
//!   rerun it), and it also adds a continue block before each repeat and a skip block after
//!   each `break`; here a block is added only where the tree lacks one, the header is the
//!   body's first block (the IR's loop runs its whole body again), and the structurizer's own
//!   condition blocks stand for the continue and skip blocks.
//! - **Scope.** The verifier's structured scope rule (a value is in scope to the end of the
//!   `if` or loop body that defines it, `ir` module docs) is stricter than the dominance Eden's
//!   rewrite relies on. So a block takes its single predecessor's value without a phi only
//!   when the predecessor's body encloses its own, and a trivial phi is removed only when its
//!   one value is in scope in the phi's block. Otherwise the phi stays and carries the value
//!   out: a phi of one operand after a loop
//!   (`tests::structured::a_loop_carries_its_value_in_a_header_phi_and_out_in_a_phi`), or of
//!   operands that are all one value (`tests::structured::a_break_and_the_repeat_both_reach_the_block_after_the_loop`).
//!   Every value a read finds is so in scope at the end of the block it is read for, which is
//!   where the lowering stores a phi's incoming value (`crate::lower`, "Structured control
//!   flow"). The structured verifier checks only that a phi's incoming value is listed
//!   somewhere, so a test-side checker (`tests::check_phi_scope`, itself pinned by
//!   `tests::structured::the_scope_checker_refuses_a_phi_value_out_of_scope`) checks this after
//!   the rewrite and after every later pass in `tests::equivalence`'s proptests, in
//!   `tests::structured`'s rewrites and in `flow`'s whole-program tests. Eden keeps neither
//!   rule (its verifier accepts a use past a body's end).
//! - **The shortcut to the entry.** Eden's lookup walks back block by block. The structurizer
//!   resets every goto variable at the start of the program and sets each again near its goto,
//!   so a read on a path that skipped the goto walked back to the start: on `tests::linear`'s
//!   run of n diamonds and loops, structured, the rewrite's steps grew as n² (88,216, 318,891
//!   and 1,207,741 for 50, 100 and 200, measured while building this slice). Here a read in a block other than the entry takes the
//!   entry's value at once when no block other than the entry writes the variable at a place
//!   in tree order up to the block's reach (the last place inside the outermost loop around the
//!   block, or the block's own place). That is exact: edges go forward in tree order except a
//!   loop's back edge, which stays inside the loop, so every path from the entry to the block
//!   passes only blocks up to its reach. The output differs from Eden's only by the trivial
//!   phis the walk would have made and removed, each an `Identity` of the same value
//!   (`tests::linear::every_pass_does_work_in_proportion_to_the_structured_program`,
//!   `tests::structured::the_rewrites_work_on_the_structured_diamond`,
//!   `tests::structured::a_write_later_in_a_loop_reaches_its_first_block`).
//! - **Phis are refused** ([`PassError::StructuredPhi`]): a phi names the blocks that can have
//!   run last before its block, and the blocks added above change them. Eden's rewrite never
//!   meets a phi before SSA, and the structurizer refuses one too (`StructureError::Phi`)
//!   (`tests::structured::structured_phis_are_refused_before_anything_changes`).
//!
//! The rest of slice 8's passes run after the rewrite on the structured form, as before
//! (`tests::equivalence::generated_programs_keep_their_results_structured`, which also checks
//! that no goto variable is left, and `tests::structured::the_structurizers_variables_are_gone_after_ssa`).
//!
//! Where this pass departs from Eden's otherwise, each pinned by a test in `tests`:
//!
//! - **Sealing.** Eden seals each block right after visiting it, in reverse post order. On a
//!   control-flow graph whose back edge leaves a block that writes the variable (a loop of one
//!   block, which flow analysis builds, slice 10), that loses the loop-carried value: the
//!   loop header's phi reads the back-edge block before the block is filled, and gets the
//!   value at its start. Eden's structurizer always ends a loop with a block of its own, which
//!   its translators never write a register in, so Eden does not meet this. Here a block is
//!   sealed once all its predecessors are filled, as Braun et al. require: before it is
//!   filled when its predecessors already are, else as soon as the last one is
//!   (`tests::ssa::a_self_loop_keeps_its_loop_carried_value`). A structured loop's first block
//!   is so sealed when the end of its body is filled.
//! - **Unreachable blocks.** Eden removes them first; this IR keeps them, so in a graph they
//!   are filled after the reachable ones, in arena order, and a reachable block with an
//!   unreachable predecessor is sealed then (`tests::ssa::unreachable_blocks_are_rewritten`);
//!   in a structured function every block is filled in tree order, the unreachable ones among
//!   the rest (`tests::structured::blocks_after_a_return_read_undef`). A read in a
//!   block the entry cannot reach looks only in that block: it takes the block's own earlier
//!   write, else a new `Undef` of the variable's type in the block, and never a phi or a
//!   predecessor's value (such a block never runs). Looking through its predecessors could
//!   give a phi whose operands are all the block's own later value: in a cycle of blocks the
//!   entry cannot reach, a phi took that value from every back edge, was trivial, and stood
//!   for it ahead of its definition, so the output did not verify
//!   (`tests::ssa::an_unreachable_cycle_of_blocks_reads_undef`,
//!   `tests::ssa::an_unreachable_self_loop_reads_undef`,
//!   `tests::ssa::an_unreachable_two_block_cycle_reads_undef`). A reachable block's phi
//!   takes such a predecessor's value the same way: its last write there, else an `Undef`
//!   there.
//! - **An entry with predecessors** (a graph only). Eden's programs start with a block nothing
//!   jumps to; flow analysis can make block 0 a loop header. A phi there would have no value on
//!   entry (the interpreter's `PhiAtStart`), so when block 0 has predecessors its instructions
//!   and terminator move to a new last block and block 0 jumps to it; terminators and phis that
//!   named block 0 name the new block. An input phi in such a block 0 is refused
//!   ([`PassError::EntryPhi`]) before anything changes
//!   (`tests::ssa::an_entry_loop_moves_to_its_own_block`, `tests::ssa::entry_phis_are_refused`).
//! - **A trivial phi's value** is its operands' common value resolved through `Identity`
//!   chains (by `identity`'s resolver, which follows each chain once and remembers its end,
//!   so many phis fed from one long chain cost its length once:
//!   `tests::linear::many_variables_fed_from_one_chain_cost_their_size`); Eden keeps the first
//!   operand as written, which can be an `Identity` in one
//!   predecessor of a join, where it does not dominate the join's users. The resolved value
//!   does: it reaches the block along every edge. Eden's backend resolves identities, so the
//!   two mean the same; the verifier here checks dominance on the operand as written
//!   (`tests::ssa::a_trivial_join_takes_the_resolved_value`).
//! - **Placement.** Phis created here come first in their block, in creation order (Eden
//!   prepends each, so its order is the reverse), then the block's own phis; then the
//!   `Undef`s created for trivial phis; then the trivial phis, now `Identity` operations, in
//!   removal order (Eden reinserts each right after the phis, so newest first). The order
//!   puts the `Undef`s ahead of the removed phis that name them
//!   (`tests::ssa::an_entry_phi_with_no_predecessors_is_left_alone`). A removed phi placed
//!   ahead of the block's own operations is only correct when its value is not one of them.
//!   That holds for the phis made here only because no phi is made in a block the entry cannot
//!   reach (above): there every operand can be the block's own later value. In a reachable
//!   block, one operand arrives along a path from the entry that does not pass through the
//!   block. This is an argument, not a proof: the verifier run after the pass in tests and
//!   debug builds checks it, and the equivalence proptests (`tests::equivalence`) exercise it
//!   on generated loops and joins, as graphs and structured.
//! - **Refusals come before any change.** Before anything is changed, one walk over the
//!   blocks in arena order, and each block's instructions in list order, refuses the first
//!   instruction that is missing ([`PassError::Missing`]), is a phi in a structured function
//!   ([`PassError::StructuredPhi`]), names a goto variable with a value rather than an
//!   immediate ([`PassError::GotoIndex`]; Eden's `Arg(0).U32()` throws on one), or is an
//!   `Identity` whose chain loops or ends without a value ([`PassError::Identity`]; the
//!   rewrite follows such chains for trivial phis, and Eden would loop forever), checked in
//!   that order for each instruction. An entry phi ([`PassError::EntryPhi`], a graph only) is
//!   checked after that walk (`tests::ssa::goto_indices_must_be_immediates`,
//!   `tests::ssa::a_missing_listed_id_is_refused_before_anything_changes`,
//!   `tests::ssa::a_broken_identity_chain_is_refused_before_anything_changes`,
//!   `tests::ssa::the_first_fault_in_block_order_is_refused`,
//!   `tests::structured::the_first_fault_in_block_order_is_refused_in_a_structured_function`).
//!   Two failures can still come after the rewrite has begun, and leave the function partly
//!   changed as [`run`](super::run) says: an instruction or block the arena cannot hold
//!   ([`PassError::Build`]; the blocks added to a structured tree are the first change), and
//!   [`PassError::Identity`] for a chain the rewrite itself made (a read becomes an
//!   `Identity` of the value found), which could loop on an input that uses a value ahead of
//!   its definition in one block (one that does not verify). Neither has a test.

mod tree;

use std::collections::BTreeMap;

use crate::ir::{BlockId, Function, Imm, Inst, Opcode, Operand, Phi, Terminator, Type, Value};

use super::identity::Resolver;
use super::{PassError, Work, block_order, identity_of, predecessors, slot};
use tree::{Graph, Places, Scopes};

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
    /// The scopes values are visible in (one, in a control-flow graph).
    scopes: Scopes,
    /// The block that lists each value, by value index, as far as the rewrite knows it.
    home: Vec<Option<BlockId>>,
    /// A structured function's places, for the shortcut to the entry.
    places: Option<Places>,
    /// Each variable's first place in tree order where a block other than the entry writes it
    /// (structured functions only).
    first_write: BTreeMap<Var, usize>,
    /// Each block's current definition of each variable (Eden's `DefTable`).
    defs: Vec<BTreeMap<Var, Operand>>,
    /// Each block's incomplete phis (Eden's `incomplete_phis`).
    incomplete: Vec<BTreeMap<Var, Value>>,
    sealed: Vec<bool>,
    /// Whether the entry reaches each block.
    reachable: Vec<bool>,
    /// Follows `Identity` chains for trivial phis, remembering each chain's end.
    resolver: Resolver,
    /// Each block's phis made here, in creation order.
    phis: Vec<Vec<Value>>,
    /// Each block's `Undef`s, made for trivial phis and for reads in blocks the entry cannot
    /// reach.
    undefs: Vec<Vec<Value>>,
    /// Each block's trivial phis, now `Identity` operations, in removal order.
    removed: Vec<Vec<Value>>,
    work: Work,
}

/// Runs the rewrite (see the module docs).
pub(super) fn run(function: &mut Function) -> Result<Work, PassError> {
    let structured = function.structure().is_some();
    let mut work = Work::default();
    let mut resolver = Resolver::new(function);
    check(function, structured, &mut resolver, &mut work)?;
    let Graph {
        preds,
        succs,
        reachable,
        scopes,
        places,
    } = if structured {
        tree::normalize(function, &mut work)?;
        tree::structured(function, &mut work)
    } else {
        split_entry(function, &mut work)?;
        Graph::of_graph(function, &mut work)
    };
    let order = block_order(function, &mut work);
    let first_write = places.as_ref().map_or_else(BTreeMap::new, |places| {
        first_writes(function, places, &mut work)
    });
    let count = function.block_count();
    // Take each block's list: instructions made below are appended to the emptied lists,
    // which the final placement replaces.
    let ids: Vec<BlockId> = function.blocks().map(|(id, _)| id).collect();
    let mut home = vec![None; super::arena_len(function)];
    let mut original = Vec::with_capacity(count);
    for &id in &ids {
        let list = function
            .block_mut(id)
            .map_or_default(|block| std::mem::take(&mut block.insts));
        for &value in &list {
            if let Some(slot) = usize::try_from(value.index())
                .ok()
                .and_then(|index| home.get_mut(index))
            {
                *slot = Some(id);
            }
        }
        original.push(list);
    }
    let mut ssa = Ssa {
        function,
        preds,
        scopes,
        home,
        places,
        first_write,
        defs: vec![BTreeMap::new(); count],
        incomplete: vec![BTreeMap::new(); count],
        sealed: vec![false; count],
        reachable,
        resolver,
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
        for &to in succs.get(index).map_or(&[][..], Vec::as_slice) {
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

    fn is_reachable(&self, block: BlockId) -> bool {
        slot(block)
            .and_then(|index| self.reachable.get(index))
            .copied()
            .unwrap_or(false)
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
        self.note_home(phi, block);
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
                    } else if !self.is_reachable(block) {
                        // The block never runs: its value is undefined (see the module docs).
                        let undef = self.new_undef(var, block)?;
                        if let Some(top) = stack.last_mut() {
                            top.result = Operand::Value(undef);
                        }
                    } else if let Some(entry) = self.unwritten_since_entry(var, block) {
                        // No block the entry's paths to here pass writes the variable (see
                        // the module docs): its value is the entry's.
                        top.pc = Pc::SetValue;
                        stack.push(Frame::new(entry));
                        continue;
                    } else if !self.is_sealed(block) {
                        // An incomplete control-flow graph.
                        let phi = self.new_phi(var, block)?;
                        if let Some(map) = slot(block).and_then(|i| self.incomplete.get_mut(i)) {
                            map.insert(var, phi);
                        }
                        if let Some(top) = stack.last_mut() {
                            top.result = Operand::Value(phi);
                        }
                    } else if let [pred] = *self.preds_of(block)
                        && self.scopes.encloses(pred, block)
                    {
                        // One predecessor whose values are in scope here: no phi needed.
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
        Ok(stack
            .first()
            .map_or(Operand::Imm(Imm::U1(false)), |f| f.result))
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
            .map_or_default(std::mem::take);
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
                let resolved = self
                    .resolver
                    .resolve(self.function, operand, &mut self.work)?;
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
        let same = if let Some(same) = same {
            if !self.visible(same, block) {
                // The one value is out of scope here (it is defined in an `if` or loop body
                // that has ended): the phi carries it out (see the module docs).
                return Ok(Operand::Value(phi));
            }
            same
        } else {
            // The phi has no operand but itself, as in the entry block, which has no
            // predecessors.
            Operand::Value(self.new_undef(var, block)?)
        };
        if let Some(inst) = self.function.inst_mut(phi) {
            *inst = identity_of(same);
        }
        if let Some(list) = slot(block).and_then(|index| self.removed.get_mut(index)) {
            list.push(phi);
        }
        Ok(same)
    }

    /// The entry, when `block` is not the entry and no block other than the entry writes `var`
    /// at a place up to `block`'s reach (see the module docs, "Structured functions"); `None`
    /// in a control-flow graph.
    fn unwritten_since_entry(&self, var: Var, block: BlockId) -> Option<BlockId> {
        let places = self.places.as_ref()?;
        if block == places.entry {
            return None;
        }
        let reach = places.place(block).and(places.reach(block))?;
        match self.first_write.get(&var) {
            Some(&first) if first <= reach => None,
            _ => Some(places.entry),
        }
    }

    /// Records that `value`, which the rewrite made, is listed in `block`.
    fn note_home(&mut self, value: Value, block: BlockId) {
        let Ok(index) = usize::try_from(value.index()) else {
            return;
        };
        if self.home.len() <= index {
            self.home.resize(index.saturating_add(1), None);
        }
        if let Some(slot) = self.home.get_mut(index) {
            *slot = Some(block);
        }
    }

    /// Whether `operand` is in scope in `block`: an immediate, or a value listed in a block
    /// whose body encloses `block`'s (always, in a control-flow graph).
    fn visible(&self, operand: Operand, block: BlockId) -> bool {
        let Operand::Value(value) = operand else {
            return true;
        };
        if self.scopes.is_flat() {
            return true;
        }
        usize::try_from(value.index())
            .ok()
            .and_then(|index| self.home.get(index))
            .copied()
            .flatten()
            .is_some_and(|home| self.scopes.encloses(home, block))
    }

    /// A new `Undef` of `var`'s type in `block`, placed ahead of the block's own operations.
    fn new_undef(&mut self, var: Var, block: BlockId) -> Result<Value, PassError> {
        let undef = self.function.push(
            block,
            Inst::Op(crate::ir::Operation::new(var.undef(), Vec::new())),
        )?;
        if let Some(list) = slot(block).and_then(|index| self.undefs.get_mut(index)) {
            list.push(undef);
        }
        self.note_home(undef, block);
        Ok(undef)
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
            insts.extend_from_slice(removed);
            self.work.step();
            insts.extend(list);
            if let Some(block) = self.function.block_mut(id) {
                block.insts = insts;
            }
        }
        self.work
    }
}

/// The checks that can refuse the function, all made before anything changes (see the module
/// docs): blocks in arena order, each block's instructions in list order, and for each
/// instruction, that it exists ([`PassError::Missing`]), that a goto variable it names is named
/// by an immediate ([`PassError::GotoIndex`]) and, for an `Identity`, that its chain ends at a
/// value ([`PassError::Identity`]). The first fault met is returned.
fn check(
    function: &Function,
    structured: bool,
    resolver: &mut Resolver,
    work: &mut Work,
) -> Result<(), PassError> {
    for (_, block) in function.blocks() {
        for &value in &block.insts {
            work.step();
            let inst = function.inst(value).ok_or(PassError::Missing(value))?;
            if structured && inst.is_phi() {
                return Err(PassError::StructuredPhi(value));
            }
            access(value, inst)?;
            if matches!(inst, Inst::Op(op) if op.opcode == Opcode::Identity) {
                resolver.resolve(function, Operand::Value(value), work)?;
            }
        }
    }
    Ok(())
}

/// Each variable's first place in tree order where a block other than the entry writes it: one
/// walk over the listed instructions, a step each.
fn first_writes(function: &Function, places: &Places, work: &mut Work) -> BTreeMap<Var, usize> {
    let mut first = BTreeMap::new();
    for (block, listed) in function.blocks() {
        if block == places.entry {
            continue;
        }
        let Some(place) = places.place(block) else {
            continue;
        };
        for &value in &listed.insts {
            work.step();
            let Some(inst) = function.inst(value) else {
                continue;
            };
            if let Ok(Some(Access::Write(var, _))) = access(value, inst) {
                first
                    .entry(var)
                    .and_modify(|at: &mut usize| *at = (*at).min(place))
                    .or_insert(place);
            }
        }
    }
    first
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
    let moved = function.block_mut(entry).map_or_default(std::mem::take);
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
