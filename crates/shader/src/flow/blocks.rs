//! The blocks of each function, from the walk's result (PR #4588's `Builder` in `flow.cpp` @
//! e5fb3b97, `Build` and `BuildFrom`, GPL-3.0-or-later, Copyright 2026 Eden Emulator Project).
//!
//! Every reachable leader gets a block, made before any is filled so that ends can name them;
//! filling a block walks from its leader to the next leader or to the control instruction that
//! ends it. A conditional instruction (a guarded side effect, a `KILL`, a guarded `CALL`, a
//! conditional `EXIT`) gets a test block at its leader and a block of its own behind it. The
//! blocks of each function are listed with its entry first, ready for inlining.

use std::ops::Range;

use super::analysis::{Analysis, guard_of, is_never, opcode_of, second_of};
use super::classify::{OpKind, classify};
use super::{BranchTables, FlowError};
use crate::decode::{BrxSrc, Instruction, Op, PredSrc, UniformReg, decode};

/// How control leaves a block before inlining: [`super::BlockEnd`] with calls and returns.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum RawEnd {
    /// Continue at a block.
    Jump(u32),
    /// Continue at `then` where `cond` holds, else at `otherwise`.
    Branch {
        cond: PredSrc,
        then: u32,
        otherwise: u32,
    },
    /// `BRX`: (table value, block) pairs.
    Indirect { src: BrxSrc, targets: Vec<(u32, u32)> },
    /// The thread exits.
    Exit,
    /// Run function `callee`, then continue at `next`. `call` is the `CALL`'s index.
    Call { call: u32, callee: u32, next: u32 },
    /// Return to the caller where `cond` holds (always when `None`), else continue at
    /// `otherwise`. `ret` is the `RET`'s index.
    Return {
        ret: u32,
        cond: Option<(PredSrc, u32)>,
    },
}

/// A block before inlining.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct RawBlock {
    pub code: Range<u32>,
    pub guard_tested: bool,
    pub end: RawEnd,
}

/// Every function's blocks.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct RawGraph {
    /// Every block; ends name blocks by their index here.
    pub blocks: Vec<RawBlock>,
    /// Per function, its blocks, the entry first.
    pub functions: Vec<Vec<u32>>,
    /// Per block, its position in its function's list.
    pub position: Vec<u32>,
}

/// The builder's state.
struct Builder<'a> {
    code: &'a [Instruction],
    analysis: &'a Analysis,
    tables: &'a BranchTables,
    /// The block control enters at each reachable leader.
    entry_block: Vec<Option<u32>>,
    blocks: Vec<Option<RawBlock>>,
    /// Each block's function.
    function: Vec<u32>,
}

/// Builds every function's blocks from `analysis`.
pub(super) fn build(
    code: &[Instruction],
    analysis: &Analysis,
    tables: &BranchTables,
) -> Result<RawGraph, FlowError> {
    let mut builder = Builder {
        code,
        analysis,
        tables,
        entry_block: vec![None; code.len()],
        blocks: Vec::new(),
        function: Vec::new(),
    };
    let mut leaders = Vec::new();
    for (index, slot) in (0_u32..).zip(builder.entry_block.iter_mut()) {
        if analysis.is_leader(index) && analysis.is_reachable(index) {
            let block = u32::try_from(builder.blocks.len())
                .map_err(|_| FlowError::RunsPastEnd { last: index })?;
            *slot = Some(block);
            builder.blocks.push(None);
            builder
                .function
                .push(analysis.function_of(index).unwrap_or(0));
            leaders.push((index, block));
        }
    }
    for (leader, block) in leaders {
        builder.build_from(leader, block)?;
    }
    builder.finish()
}

impl Builder<'_> {
    fn count(&self) -> u32 {
        self.analysis.len()
    }

    /// The block control enters at instruction `target`, reached from `index`.
    fn entry(&self, target: u32, index: u32) -> Result<u32, FlowError> {
        usize::try_from(target)
            .ok()
            .and_then(|slot| self.entry_block.get(slot))
            .copied()
            .flatten()
            .ok_or(FlowError::Unwalked { index, target })
    }

    /// Adds a block of `function`.
    fn add(&mut self, function: u32, block: RawBlock) -> Result<u32, FlowError> {
        let id = u32::try_from(self.blocks.len()).map_err(|_| FlowError::RunsPastEnd {
            last: self.count().saturating_sub(1),
        })?;
        self.blocks.push(Some(block));
        self.function.push(function);
        Ok(id)
    }

    /// Fills block `id`.
    fn fill(&mut self, id: u32, block: RawBlock) {
        if let Some(slot) = usize::try_from(id).ok().and_then(|slot| self.blocks.get_mut(slot)) {
            *slot = Some(block);
        }
    }

    /// Fills the block that starts at `leader` (`BuildFrom`).
    fn build_from(&mut self, leader: u32, block: u32) -> Result<(), FlowError> {
        let function = self.analysis.function_of(leader).unwrap_or(0);
        let count = self.count();
        let past_end = FlowError::RunsPastEnd {
            last: count.saturating_sub(1),
        };
        let mut index = leader;
        loop {
            if index >= count {
                return Err(past_end);
            }
            let next_index = index.saturating_add(1);
            if index != leader && self.analysis.is_leader(index) {
                // Fall through into the next block.
                let next = self.entry(index, index)?;
                self.fill(block, plain(leader..index, RawEnd::Jump(next)));
                return Ok(());
            }
            let word = *usize::try_from(index)
                .ok()
                .and_then(|slot| self.code.get(slot))
                .ok_or(past_end)?;
            let info = classify(opcode_of(word));
            // The walk decoded every reachable control instruction, so this cannot fail.
            let op = if info.kind == OpKind::Plain {
                None
            } else {
                Some(
                    decode(word)
                        .map_err(|error| FlowError::Decode { index, error })?
                        .op,
                )
            };
            let mut kind = info.kind;
            if kind == OpKind::Call && self.analysis.callee(index).is_none() {
                // A skip call: the walk took it as a branch (see `analysis::is_skip_call`).
                kind = OpKind::Bra;
            }
            let guard = guard_of(word);
            let second = second_of(word);
            if is_never(guard) {
                index = next_index;
                continue;
            }
            let conditional = match kind {
                OpKind::Plain => !info.pure && !guard.is_true(),
                OpKind::Kill => true,
                OpKind::Call => !guard.is_true(),
                OpKind::Exit => !(guard.is_true() && second.is_true()),
                _ => false,
            };
            if conditional {
                return self.conditional(block, function, index, kind, guard, second);
            }
            let end = match (kind, op) {
                (OpKind::Exit, _) => Some((leader..next_index, RawEnd::Exit)),
                (OpKind::Bra, Some(Op::Bra(bra))) => {
                    let target = super::analysis::branch_target(index, bra.offset)?;
                    Some(self.bra(function, leader, index, target, guard, second)?)
                }
                (OpKind::Bra, Some(Op::Call(call))) => {
                    let target = super::analysis::branch_target(index, call.offset)?;
                    Some(self.bra(function, leader, index, target, guard, second)?)
                }
                (OpKind::Bsync, _) => {
                    let target = self
                        .analysis
                        .sync_target(index)
                        .ok_or(FlowError::MissingBarrier { index })?;
                    Some((leader..index, RawEnd::Jump(self.entry(target, index)?)))
                }
                (OpKind::Call, _) => {
                    let callee = self
                        .analysis
                        .callee(index)
                        .ok_or(FlowError::CallLinkage { index })?;
                    let next = self.entry(next_index, index)?;
                    Some((
                        leader..index,
                        RawEnd::Call {
                            call: index,
                            callee,
                            next,
                        },
                    ))
                }
                (OpKind::Ret, _) => {
                    let cond = if guard.is_true() {
                        None
                    } else {
                        Some((guard, self.entry(next_index, index)?))
                    };
                    Some((leader..index, RawEnd::Return { ret: index, cond }))
                }
                (OpKind::Brx, Some(Op::Brx(brx))) => {
                    Some((leader..next_index, self.brx(index, brx.src)?))
                }
                _ => None,
            };
            if let Some((code, end)) = end {
                self.fill(block, plain(code, end));
                return Ok(());
            }
            index = next_index;
        }
    }

    /// The end of a block closed by a `BRA` at `index` to `target`: one test, or two in a row
    /// when both the guard and the second predicate are set.
    fn bra(
        &mut self,
        function: u32,
        leader: u32,
        index: u32,
        target: u32,
        guard: PredSrc,
        second: PredSrc,
    ) -> Result<(Range<u32>, RawEnd), FlowError> {
        let then = self.entry(target, index)?;
        let code = leader..index;
        if guard.is_true() && second.is_true() {
            return Ok((code, RawEnd::Jump(then)));
        }
        let next = self.entry(index.saturating_add(1), index)?;
        if second.is_true() {
            return Ok((code, branch(guard, then, next)));
        }
        if guard.is_true() {
            return Ok((code, branch(second, then, next)));
        }
        // Guard and second: test the guard first, then the second in a block of its own.
        let extra = self.add(function, plain(index..index, branch(second, then, next)))?;
        Ok((code, branch(guard, extra, next)))
    }

    /// The end of a block closed by a `BRX` at `index`.
    fn brx(&self, index: u32, src: BrxSrc) -> Result<RawEnd, FlowError> {
        if src == BrxSrc::UniformReg(UniformReg::URZ) {
            return Err(FlowError::BrxThroughUrz { index });
        }
        let table = self
            .analysis
            .indirect_targets(index)
            .ok_or(FlowError::NoTarget { index })?;
        let mut targets = Vec::with_capacity(table.len());
        for &target in table {
            let address = indirect_address(self.tables.base(), target)
                .ok_or(FlowError::TargetOutside {
                    index,
                    target: i64::from(target),
                })?;
            targets.push((address, self.entry(target, index)?));
        }
        Ok(RawEnd::Indirect { src, targets })
    }

    /// Fills `block` as the test in front of the conditional instruction at `index`, and adds
    /// the block that holds it.
    fn conditional(
        &mut self,
        block: u32,
        function: u32,
        index: u32,
        kind: OpKind,
        guard: PredSrc,
        second: PredSrc,
    ) -> Result<(), FlowError> {
        let next_index = index.saturating_add(1);
        if next_index >= self.count() {
            return Err(FlowError::RunsPastEnd {
                last: self.count().saturating_sub(1),
            });
        }
        let next = self.entry(next_index, index)?;
        let (code, end) = match kind {
            OpKind::Exit => (index..next_index, RawEnd::Exit),
            OpKind::Call => {
                let callee = self
                    .analysis
                    .callee(index)
                    .ok_or(FlowError::CallLinkage { index })?;
                (
                    index..index,
                    RawEnd::Call {
                        call: index,
                        callee,
                        next,
                    },
                )
            }
            _ => (index..next_index, RawEnd::Jump(next)),
        };
        let conditional = self.add(
            function,
            RawBlock {
                code,
                guard_tested: true,
                end,
            },
        )?;
        if kind == OpKind::Exit && !guard.is_true() && !second.is_true() {
            // Exit only when both hold: test the guard, then the second predicate in a block
            // of its own.
            let second_test = self.add(
                function,
                plain(next_index..next_index, branch(second, conditional, next)),
            )?;
            self.fill(block, plain(index..index, branch(guard, second_test, next)));
            return Ok(());
        }
        let cond = if kind == OpKind::Exit && guard.is_true() {
            second
        } else {
            guard
        };
        let end = if cond.is_true() {
            RawEnd::Jump(conditional)
        } else {
            branch(cond, conditional, next)
        };
        self.fill(block, plain(index..index, end));
        Ok(())
    }

    /// The graph, every block filled, with each function's blocks listed entry first.
    fn finish(self) -> Result<RawGraph, FlowError> {
        let mut blocks = Vec::with_capacity(self.blocks.len());
        for block in self.blocks {
            // Every leader block is filled by `build_from`, which returns an error otherwise.
            blocks.push(block.ok_or(FlowError::RunsPastEnd {
                last: self.analysis.len().saturating_sub(1),
            })?);
        }
        let entries = self.analysis.function_entries();
        let mut functions: Vec<Vec<u32>> = vec![Vec::new(); entries.len()];
        for (function, &entry) in functions.iter_mut().zip(entries) {
            let entry = usize::try_from(entry)
                .ok()
                .and_then(|slot| self.entry_block.get(slot))
                .copied()
                .flatten()
                .ok_or(FlowError::Unwalked {
                    index: entry,
                    target: entry,
                })?;
            function.push(entry);
        }
        let mut position = vec![0; blocks.len()];
        for (id, &owner) in (0_u32..).zip(&self.function) {
            let Some(list) = usize::try_from(owner).ok().and_then(|f| functions.get_mut(f))
            else {
                continue;
            };
            if list.first() != Some(&id) {
                list.push(id);
            }
        }
        for list in &functions {
            for (place, &id) in (0_u32..).zip(list) {
                if let Some(slot) = usize::try_from(id).ok().and_then(|id| position.get_mut(id)) {
                    *slot = place;
                }
            }
        }
        Ok(RawGraph {
            blocks,
            functions,
            position,
        })
    }
}

/// A block whose guard the graph has not tested.
const fn plain(code: Range<u32>, end: RawEnd) -> RawBlock {
    RawBlock {
        code,
        guard_tested: false,
        end,
    }
}

/// A two-way end.
const fn branch(cond: PredSrc, then: u32, otherwise: u32) -> RawEnd {
    RawEnd::Branch {
        cond,
        then,
        otherwise,
    }
}

/// The table value that names instruction `target`: `base` plus 16 bytes per instruction.
pub(super) fn indirect_address(base: u32, target: u32) -> Option<u32> {
    let size = u32::try_from(Instruction::SIZE).ok()?;
    target.checked_mul(size)?.checked_add(base)
}
