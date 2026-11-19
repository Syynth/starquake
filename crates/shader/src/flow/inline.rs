//! Inlining: each `CALL` replaced by a copy of its function's blocks, whose returns continue
//! after the call, so the [`FlowGraph`] is one function (Eden's structurizer does the same per
//! call site: `eden/src/shader_recompiler/frontend/maxwell/structured_control_flow.cpp` @
//! cf654052, `EndClass::Call`, `BuildTree` of the callee with the return block as its return
//! label).
//!
//! Copies are made breadth first from the program's function, each with its blocks in a
//! consecutive run of ids, the entry first, so the program's entry is block 0. A copy's ids are
//! its function's block positions plus the run's start, which needs no per-copy map. Each copy
//! knows its caller's copy, so a recursive call is found by walking at most
//! [`MAX_INLINE_DEPTH`] callers.

use super::blocks::{RawEnd, RawGraph};
use super::{
    BlockEnd, BlockIndex, FlowBlock, FlowError, FlowGraph, IndirectTarget, MAX_FLOW_BLOCKS,
    MAX_INLINE_DEPTH,
};

/// One copy of a function.
#[derive(Clone, Copy, Debug)]
struct Instance {
    /// The function copied.
    function: u32,
    /// The id of the copy's first block.
    start: u32,
    /// How many calls deep it is: 0 for the program.
    depth: u32,
    /// The copy that called it, and the block its returns continue at.
    caller: Option<(usize, u32)>,
}

/// Inlines every call of `raw` (see the module docs).
pub(super) fn inline(raw: &RawGraph) -> Result<FlowGraph, FlowError> {
    let mut copies = vec![Instance {
        function: 0,
        start: 0,
        depth: 0,
        caller: None,
    }];
    let mut blocks: Vec<FlowBlock> = Vec::new();
    let mut reserved = function_len(raw, 0);
    let mut current = 0;
    while let Some(&copy) = copies.get(current) {
        let list = function_blocks(raw, copy.function);
        for &id in list {
            let block = raw_block(raw, id)?;
            let end = match &block.end {
                RawEnd::Jump(target) => BlockEnd::Jump(map(raw, copy, *target)?),
                RawEnd::Branch {
                    cond,
                    then,
                    otherwise,
                } => BlockEnd::Branch {
                    cond: *cond,
                    then: map(raw, copy, *then)?,
                    otherwise: map(raw, copy, *otherwise)?,
                },
                RawEnd::Indirect { src, targets } => BlockEnd::Indirect {
                    src: *src,
                    targets: targets
                        .iter()
                        .map(|&(address, target)| {
                            Ok(IndirectTarget {
                                address,
                                block: map(raw, copy, target)?,
                            })
                        })
                        .collect::<Result<_, FlowError>>()?,
                },
                RawEnd::Exit => BlockEnd::Exit,
                RawEnd::Call { call, callee, next } => {
                    let depth = copy.depth.saturating_add(1);
                    if calls_itself(&copies, copy, *callee) {
                        return Err(FlowError::Recursion { index: *call });
                    }
                    if depth > MAX_INLINE_DEPTH {
                        return Err(FlowError::InlineDepth { index: *call });
                    }
                    let size = function_len(raw, *callee);
                    let start = reserved;
                    reserved = reserved
                        .checked_add(size)
                        .filter(|&total| total <= MAX_FLOW_BLOCKS)
                        .ok_or(FlowError::InlineSize { index: *call })?;
                    let return_to = map(raw, copy, *next)?;
                    copies.push(Instance {
                        function: *callee,
                        start: u32::try_from(start)
                            .map_err(|_| FlowError::InlineSize { index: *call })?,
                        depth,
                        caller: Some((current, return_to.0)),
                    });
                    BlockEnd::Jump(BlockIndex(
                        u32::try_from(start).map_err(|_| FlowError::InlineSize { index: *call })?,
                    ))
                }
                RawEnd::Return { ret, cond } => {
                    let Some((_, return_to)) = copy.caller else {
                        return Err(FlowError::ReturnLinkage { index: *ret });
                    };
                    let return_to = BlockIndex(return_to);
                    match cond {
                        None => BlockEnd::Jump(return_to),
                        Some((cond, otherwise)) => BlockEnd::Branch {
                            cond: *cond,
                            then: return_to,
                            otherwise: map(raw, copy, *otherwise)?,
                        },
                    }
                }
            };
            blocks.push(FlowBlock {
                code: block.code.clone(),
                guard_tested: block.guard_tested,
                end,
            });
        }
        current += 1;
    }
    Ok(FlowGraph { blocks })
}

/// Whether `callee` is `copy`'s function or the function of one of its callers.
fn calls_itself(copies: &[Instance], copy: Instance, callee: u32) -> bool {
    let mut current = Some(copy);
    // Bounded by the depth: each caller is one call shallower.
    while let Some(link) = current {
        if link.function == callee {
            return true;
        }
        current = link
            .caller
            .and_then(|(caller, _)| copies.get(caller).copied());
    }
    false
}

/// Function `function`'s blocks, the entry first.
fn function_blocks(raw: &RawGraph, function: u32) -> &[u32] {
    usize::try_from(function)
        .ok()
        .and_then(|slot| raw.functions.get(slot))
        .map_or(&[], Vec::as_slice)
}

/// How many blocks function `function` has.
fn function_len(raw: &RawGraph, function: u32) -> usize {
    function_blocks(raw, function).len()
}

/// Raw block `id`.
fn raw_block(raw: &RawGraph, id: u32) -> Result<&super::blocks::RawBlock, FlowError> {
    usize::try_from(id)
        .ok()
        .and_then(|slot| raw.blocks.get(slot))
        .ok_or(FlowError::Unwalked {
            index: id,
            target: id,
        })
}

/// The id, in `copy`, of raw block `id` of its function.
fn map(raw: &RawGraph, copy: Instance, id: u32) -> Result<BlockIndex, FlowError> {
    let position = usize::try_from(id)
        .ok()
        .and_then(|slot| raw.position.get(slot))
        .copied()
        .ok_or(FlowError::Unwalked {
            index: id,
            target: id,
        })?;
    copy.start
        .checked_add(position)
        .map(BlockIndex)
        .ok_or(FlowError::Unwalked {
            index: id,
            target: id,
        })
}
