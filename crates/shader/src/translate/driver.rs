//! The translation driver (slice 12i; spec, "Instruction translation", "The translation
//! driver"; decided by the owner, 2026-11-19): runs the group translators over a whole program.
//!
//! [`translate_flow`] rebuilds the program's control flow ([`flow_graph`], slice 10), builds the
//! IR's blocks from it ([`FlowGraph::build_ir`]) and fills each block from its flow block's
//! instructions, walking [`IrBlocks::sources`] in order and each flow block's `code` in order.
//! Each instruction's IR goes in front of its block's terminator: flow analysis has already
//! terminated every block, and a translator appends to a block's instructions, which come before
//! its terminator. The result is the filled control-flow graph, before structuring (slice 11) and
//! SSA (slice 8), in Eden's order (`frontend/maxwell/translate_program.cpp` @ cf654052: the
//! translators fill the blocks, then `BuildASL` structures them at line 337 and
//! `SsaRewritePass` runs at line 389).
//!
//! # Sources
//!
//! Ported by behaviour from PR #4588 (`eden-pr4588/src/shader_recompiler/frontend/ampere/` @
//! e5fb3b97, GPL-3.0-or-later, Copyright 2026 Eden Emulator Project): `translate.cpp`
//! (`Translate`: a block's instructions from its first to its last, `Visitor::Run` on each,
//! called for each block of the control-flow builder's graph) and `visitor.cpp` (`Visitor::Run`:
//! the `guard.Never()` return, the dispatch by operation); cross-checked with Eden's
//! `frontend/maxwell/translate/translate.cpp` @ cf654052 (`Translate`: the same walk over a
//! block's locations).
//!
//! # The order of the checks
//!
//! 1. The stage: vertex, pixel and compute programs, the stages the lowering takes
//!    ([`translates_stage`]); any other is [`TranslateError::Stage`], named, until the lowering
//!    takes it (slice 17 owns the graphics stages past vertex and pixel).
//! 2. Flow analysis ([`TranslateError::Flow`]), which decodes the control instructions it walks.
//! 3. The size: the instructions of every flow block, counted once per block, so once per
//!    inlined copy of a function, at most [`MAX_TRANSLATED_INSTRUCTIONS`]
//!    ([`TranslateError::TooManyInstructions`]).
//! 4. Decoding: every instruction a flow block holds, each once, in ascending instruction order,
//!    so the lowest-numbered instruction that does not decode is the one refused, with the
//!    decoder's exact error ([`TranslateError::Decode`]; an unknown opcode or a set bit outside
//!    its form's mask). An instruction under `!PT` is decoded too.
//! 5. Translation, block by block and instruction by instruction, stopping at the first
//!    refusal (there is no partial translation, spec "Instruction translation").
//!
//! Each pair of neighbouring checks is pinned with both faults at once:
//! `tests::the_stage_is_checked_before_flow_analysis` (1 before 2),
//! `tests::flow_analysis_is_checked_before_decoding` (2 before 4),
//! `tests::the_budget_is_checked_before_decoding` (3 before 4) and
//! `tests::the_lowest_undecodable_instruction_is_refused` (4 before 5). Flow analysis comes
//! before the size because the size is counted over its blocks.
//!
//! # Guards
//!
//! As `Visitor::Run` reads them (`visitor.cpp`: it reads a guard only when `info.pure &&
//! !guard.Always()`, after the `guard.Never()` return), with a conditional block where the PR
//! keeps a select:
//!
//! - An instruction under `!PT` never runs and becomes nothing, as `Visitor::Run` returns when
//!   `guard.Never()` (flow analysis leaves such an instruction in its block's range and skips it
//!   the same way, `flow::is_never`).
//! - An instruction under `PT`, and the one instruction of a block whose guard flow analysis has
//!   tested ([`crate::flow::FlowBlock::guard_tested`]: a guarded instruction that is not pure, a
//!   conditional `EXIT`, a `KILL`), is translated as if unguarded.
//! - A guarded pure instruction (`classify.h`'s `pure`), which flow analysis leaves in its
//!   block, becomes a conditional block (the owner's ruling of 2026-11-19, row 2, where PR #4588
//!   keeps a select, `guard_select`): the driver reads the guard (a uniform predicate of the same
//!   number for a uniform datapath instruction, `classify.h`'s `uniform`, as `Visitor::Run`
//!   reads it with `UPred`; a predicate for the others, `Pred`), splits the block with the shared
//!   guard helper ([`super::split_for_guard`]), translates the instruction into the new block as
//!   if unguarded, and carries on in the block after it.
//! - Any other guarded instruction flow analysis leaves in its block is one of `classify.h`'s
//!   `OpKind::Nop`: `NOP`, `DEPBAR`, `BREAK` and `WARPSYNC` with an immediate (flow analysis
//!   refuses a guarded `BSSY` and a guarded `BRX`, [`FlowError::PredicatedBssy`] and its
//!   indirect-branch checks, and ends a block at a `BRA`, `BSYNC`, `CALL` or `RET`). It is
//!   translated as is, its guard not read and no block made, as `Visitor::Run` reads no guard
//!   for an instruction that is not pure; none of the four writes IR (the control group's
//!   module docs). The spec says "Every guarded instruction becomes a conditional block, pure
//!   ones included" ("Instruction translation", "Guards"): for these four a block would hold
//!   nothing, so leaving it out changes no result. **Owner question**: whether a guarded
//!   instruction that writes no IR should still get its (empty) conditional block. Pinned by
//!   `tests::a_guarded_instruction_that_is_not_pure_reads_no_guard`.
//!
//! Every group's translator is so handed its instruction with the guard already handled.
//!
//! # Dispatch
//!
//! By the decoded operation, in this order: `EXIT` (`super`'s `Translator::exit`); the integer
//! ALU group ([`super::integer::is_integer`], `MOV` included); the float group
//! ([`super::float::in_group`]); the texture group ([`super::texture::in_group`], with one
//! [`DescriptorBuffer`] for the program); the control and warp group
//! ([`super::control::in_group`]); the memory group ([`super::memory::is_memory`]: `LD`, `ST`,
//! `LDC`, `ATOM`, `CCTL`, `MEMBAR`, `LDGSTS`, `LDGDEPBAR`, `ERRBAR`, `ATOMS.POPC`), with slice
//! 12e's translator ([`super::memory::translate_memory`], its guard already handled), whose
//! refusals are [`TranslateError::Memory`]; the attribute group (`super`'s `GROUPS`). An operation none of them takes (the half and double group, the uniform group,
//! the tensor group) is refused by its mnemonic ([`Unsupported::Instruction`]). The groups are
//! disjoint, so the order decides nothing but speed: `tests::the_groups_are_disjoint` checks
//! that every decoded operation (one instruction of each, from the deterministic
//! `tests::decoded_samples`, which `tests::these_instructions_reach_every_decoded_operation`
//! checks reach every operation but the shells no word decodes to) belongs to exactly one
//! group.
//!
//! # Bounds
//!
//! Inlining copies a function's blocks to every call site (slice 10), so a program's inlined
//! graph can hold far more instructions than the program: a function of 2^15 instructions
//! called from 2^15 sites would be 2^30 translations. [`MAX_TRANSLATED_INSTRUCTIONS`] bounds the
//! count before anything is decoded or translated (`tests::the_budget_is_checked_before_decoding`), and the IR the translators write grows with
//! it: each instruction's translator writes an amount of IR fixed by its form (a `REDUX`, for
//! one, a shuffle for each of the warp's 32 lanes), plus a guard's read and split. The decoded
//! instructions are kept once per instruction of the program, which flow analysis holds to
//! [`crate::flow::MAX_PROGRAM`].
//!
//! # Divergences from PR #4588
//!
//! | What | Here | Reference | Pinned by |
//! |---|---|---|---|
//! | A guarded pure instruction | A conditional block (the owner's ruling of 2026-11-19) | A select of each value it writes (`guard_select`) | `tests::a_guarded_pure_instruction_becomes_a_conditional_block` |
//! | The instructions translated | At most [`MAX_TRANSLATED_INSTRUCTIONS`], counted over the inlined graph (a **guess**, four per instruction of a program at the capture's cap, as [`crate::flow::MAX_FLOW_BLOCKS`] is four blocks per instruction) | No bound | `tests::the_translation_budget_at_its_edge` |
//! | Which undecodable instruction is refused | The lowest-numbered one any flow block holds, after flow analysis's own refusals and before any translation | The first its flow walk meets (`flow.cpp`: `LookupRow`, `FirstUnknownBit`, before `guard.Never()`), in walk order, among the walk's other refusals | `tests::the_lowest_undecodable_instruction_is_refused` |
//! | Stages | Vertex, pixel and compute, the stages the lowering takes; tessellation, geometry and the cull-before-fetch vertex stage are refused by name until slice 17's lowering takes them | No stage check in `Translate` | `tests::only_the_stages_the_lowering_takes_translate` |

use super::control::{self, Emitted};
use super::float::{self, FloatError};
use super::integer::{self, GuardMode, IntegerError};
use super::memory::{self, MemoryError};
use super::texture::{self, DescriptorBuffer, TextureError};
use super::{
    Failure, GuestStage, Stage, TranslateError, Translator, Unsupported, source, split_for_guard,
};
use crate::capture::MAX_INSTRUCTIONS;
use crate::decode::{Instr, Instruction, Op, decode};
use crate::flow::{
    BranchTables, FlowError, FlowGraph, Invariant, IrBlocks, OpInfo, classify, flow_graph,
    is_never, opcode_of,
};
use crate::ir::{BlockId, Builder, Function};
use crate::sph::PixelIo;

/// The most instructions [`translate_flow`] translates in one program, counted over the flow
/// graph with every call inlined (see the module docs, "Bounds"). A **guess**: four per
/// instruction of a program at the capture's cap, the same allowance
/// [`crate::flow::MAX_FLOW_BLOCKS`] makes for blocks; real programs would confirm or move it.
pub const MAX_TRANSLATED_INSTRUCTIONS: u32 = 4 * MAX_INSTRUCTIONS;

/// Whether [`translate_flow`] takes programs of `stage`: vertex, pixel and compute, the stages
/// the lowering takes ([`crate::lower::lower_with`]'s `LowerError::Stage` refuses the others).
/// Any other stage is refused by name ([`Unsupported::Stage`]) until the lowering takes it
/// (slice 17; module docs, "The order of the checks").
#[must_use]
pub const fn translates_stage(stage: GuestStage) -> bool {
    matches!(
        stage,
        GuestStage::Vertex | GuestStage::Pixel | GuestStage::Compute
    )
}

/// Translates `code`, a captured program's instructions, for `stage`: flow analysis with
/// `tables` giving each `BRX`'s targets, the IR's blocks, and each block filled by the group
/// translators (see the module docs for the order of the checks). `header` is a pixel program's
/// header, whose input map tells `IPA` how each generic attribute is interpolated (PR #4588's
/// `IPA` reads it, `ops_attr.cpp`: `sph.ps.GenericInputMap`); without it an `IPA` of a generic
/// attribute is refused ([`super::AttributeRejection::IpaNoHeader`]), and an `EXIT` writes no
/// sample mask or depth. The blocks keep their [`IrBlocks::sources`], so
/// [`IrBlocks::structure`] can lay them out next.
pub fn translate_flow(
    stage: Stage,
    header: Option<&PixelIo>,
    code: &[Instruction],
    tables: &BranchTables,
) -> Result<IrBlocks, TranslateError> {
    if !translates_stage(stage.stage) {
        return Err(TranslateError::Stage(Unsupported::Stage(stage.stage)));
    }
    let graph = flow_graph(code, tables)?;
    let mut blocks = graph.build_ir()?;
    charge(
        graph.blocks().iter().map(|block| block.code.len()),
        MAX_TRANSLATED_INSTRUCTIONS,
    )?;
    let decoded = decode_held(code, &graph)?;
    let mut driver = Driver::new(stage, header);
    let IrBlocks { function, sources } = &mut blocks;
    for source in sources.iter() {
        let flow = graph
            .block(source.flow)
            .ok_or(FlowError::Internal(Invariant::NoBlock(source.flow.index())))?;
        let mut at = source.block;
        for index in flow.code.clone() {
            let (slot, word, instr) = held(code, &decoded, index)?;
            let info = classify(opcode_of(word));
            at = driver.instruction(function, at, slot, instr, info, flow.guard_tested)?;
        }
    }
    Ok(blocks)
}

/// The instructions to translate, counted from each block's length in `lengths`, checked against
/// `cap`: the total, or [`TranslateError::TooManyInstructions`] once it passes `cap` (a total
/// of exactly `cap` is accepted). The count stops at the first block that passes it.
fn charge(lengths: impl Iterator<Item = usize>, cap: u32) -> Result<u64, TranslateError> {
    let mut total: u64 = 0;
    for length in lengths {
        total = u64::try_from(length)
            .ok()
            .and_then(|length| total.checked_add(length))
            .filter(|&sum| sum <= u64::from(cap))
            .ok_or(TranslateError::TooManyInstructions)?;
    }
    Ok(total)
}

/// Instruction `index` of a flow block: its position in `code`, its word and its decoded form
/// from `decoded` ([`decode_held`]). Flow analysis made every block from `code`, and
/// `decode_held` decoded every instruction a block holds, so this fails only on a bug in the
/// driver, with [`TranslateError::NotDecoded`].
fn held(
    code: &[Instruction],
    decoded: &[Option<Instr>],
    index: u32,
) -> Result<(usize, Instruction, Instr), TranslateError> {
    let not_decoded = TranslateError::NotDecoded { index };
    let slot = usize::try_from(index).map_err(|_| not_decoded)?;
    match (code.get(slot), decoded.get(slot)) {
        (Some(word), Some(Some(instr))) => Ok((slot, *word, *instr)),
        _ => Err(not_decoded),
    }
}

/// Every instruction a block of `graph` holds, decoded once, in ascending order: the first that
/// does not decode is refused. Instructions no block holds stay `None`.
fn decode_held(
    code: &[Instruction],
    graph: &FlowGraph,
) -> Result<Vec<Option<Instr>>, TranslateError> {
    let mut held = vec![false; code.len()];
    for block in graph.blocks() {
        for index in block.code.clone() {
            if let Some(slot) = usize::try_from(index)
                .ok()
                .and_then(|slot| held.get_mut(slot))
            {
                *slot = true;
            }
        }
    }
    let mut decoded = vec![None; code.len()];
    for (index, ((word, held), slot)) in code.iter().zip(&held).zip(&mut decoded).enumerate() {
        if *held {
            *slot = Some(decode(*word).map_err(|error| TranslateError::Decode { index, error })?);
        }
    }
    Ok(decoded)
}

/// The driver's state over one program.
struct Driver {
    stage: Stage,
    header: Option<PixelIo>,
    /// The program's texture descriptor buffer ([`texture::translate_instr`]).
    textures: DescriptorBuffer,
}

impl Driver {
    fn new(stage: Stage, header: Option<&PixelIo>) -> Self {
        Self {
            stage,
            header: header.copied(),
            textures: DescriptorBuffer::new(),
        }
    }

    /// Translates `instr`, instruction `index` of the program, into `function` at the end of
    /// block `at`, whose flow block's `guard_tested` is `guard_tested`; `info` is its opcode's
    /// class (`classify.h`'s `Classify`). See the module docs, "Guards". Returns the block where
    /// translation continues.
    fn instruction(
        &mut self,
        function: &mut Function,
        at: BlockId,
        index: usize,
        instr: Instr,
        info: OpInfo,
        guard_tested: bool,
    ) -> Result<BlockId, TranslateError> {
        if is_never(instr.guard) {
            return Ok(at);
        }
        if guard_tested || instr.guard.is_true() || !info.pure {
            self.dispatch(function, at, index, instr)?;
            return Ok(at);
        }
        self.conditional(function, at, index, instr, info.uniform)
    }

    /// Makes `instr`, instruction `index`, a conditional block at the end of block `at`: reads
    /// its guard (a uniform predicate when `uniform`, else a predicate), splits the block with
    /// [`split_for_guard`] and translates `instr` into the new block as if unguarded. Returns
    /// the block after it.
    fn conditional(
        &mut self,
        function: &mut Function,
        at: BlockId,
        index: usize,
        instr: Instr,
        uniform: bool,
    ) -> Result<BlockId, TranslateError> {
        let mut b = Builder::new(function, at);
        let cond = if uniform {
            source::uniform_guard(&mut b, instr.guard)?
        } else {
            source::pred(&mut b, instr.guard)?
        };
        let join = split_for_guard(&mut b, cond)?;
        let then = b.block();
        self.dispatch(function, then, index, instr)?;
        Ok(join)
    }

    /// Translates `instr`, instruction `index`, at the end of `block` as if unguarded, with the
    /// group that takes it (see the module docs, "Dispatch").
    fn dispatch(
        &mut self,
        function: &mut Function,
        block: BlockId,
        index: usize,
        instr: Instr,
    ) -> Result<(), TranslateError> {
        let unsupported = |what| TranslateError::Unsupported { index, instr, what };
        let op = instr.op;
        if integer::is_integer(&op) {
            return integer::translate_integer(
                &mut Builder::new(function, block),
                &instr,
                GuardMode::Tested,
            )
            .map_err(|error| match error {
                IntegerError::Build(error) => TranslateError::Build(error),
                error => TranslateError::Integer {
                    index,
                    instr,
                    error,
                },
            });
        }
        if float::in_group(&op) {
            return float::translate_instr(&mut Builder::new(function, block), instr, true)
                .map_err(|error| match error {
                    FloatError::Unsupported(what) => unsupported(Unsupported::Float(what)),
                    FloatError::Build(error) => TranslateError::Build(error),
                });
        }
        if texture::in_group(&op) {
            return texture::translate_instr(
                &mut Builder::new(function, block),
                instr,
                true,
                &mut self.textures,
            )
            .map_err(|error| match error {
                TextureError::Unsupported(what) => unsupported(Unsupported::Texture(what)),
                TextureError::Build(error) => TranslateError::Build(error),
            });
        }
        if control::in_group(&op) {
            // The subgroup requirement it reports is read from the IR by collect info
            // (`crate::info`'s requirements), so it is not kept here.
            let _subgroup: Emitted =
                control::translate_control(function, block, self.stage.stage, index, instr, true)?;
            return Ok(());
        }
        if memory::is_memory(&op) {
            return memory::translate_memory(
                &mut Builder::new(function, block),
                &instr,
                GuardMode::Tested,
            )
            .map_err(|error| match error {
                MemoryError::Build(error) => TranslateError::Build(error),
                error => TranslateError::Memory {
                    index,
                    instr,
                    error,
                },
            });
        }
        let mut t = Translator {
            b: Builder::new(function, block),
            stage: self.stage,
            header: self.header,
        };
        let result = match op {
            Op::Exit(exit) => t.exit(exit),
            other => t
                .group(other)
                .unwrap_or_else(|| Err(Unsupported::Instruction(other.mnemonic()).into())),
        };
        result.map_err(|failure| match failure {
            Failure::Unsupported(what) => unsupported(what),
            Failure::Build(error) => TranslateError::Build(error),
        })
    }
}

/// The test-support entry for the translators' tests that build their programs as
/// instructions: `program` assembled with the test assembler and translated by
/// [`translate_flow`] with no jump-table targets. A program with an instruction the assembler
/// cannot encode (a hand-built field value no word holds) goes through [`translate_decoded`]
/// instead.
#[cfg(test)]
pub(crate) fn translate_instrs(
    stage: Stage,
    header: Option<&PixelIo>,
    program: &[Instr],
) -> Result<Function, TranslateError> {
    let code: Result<Vec<Instruction>, _> = program.iter().map(crate::asm::assemble).collect();
    match code {
        Ok(code) => translate_flow(stage, header, &code, &BranchTables::new(0))
            .map(|blocks| blocks.function),
        Err(_) => translate_decoded(stage, header, program),
    }
}

/// `program`, decoded instructions of a straight-line program that ends at its first `EXIT`,
/// translated by the driver's per-instruction path ([`Driver::instruction`]) into one block that
/// returns, for the tests' hand-built instructions no word encodes: the blocks flow analysis
/// would build for such a program. A guarded instruction that is not pure and is a
/// straight-line one (`classify.h`'s `OpKind::Plain`), and a guarded one the assembler cannot
/// encode, gets the conditional block flow analysis would give it, behind a test of its guard
/// as a predicate, from [`Driver::conditional`]; every other instruction goes through
/// [`Driver::instruction`] with its class. Instructions after the first unguarded `EXIT` are
/// ignored, as flow analysis never reaches them; a guarded `EXIT` is not supported here.
#[cfg(test)]
pub(crate) fn translate_decoded(
    stage: Stage,
    header: Option<&PixelIo>,
    program: &[Instr],
) -> Result<Function, TranslateError> {
    use crate::flow::OpKind;
    use crate::ir::Terminator;
    if !translates_stage(stage.stage) {
        return Err(TranslateError::Stage(Unsupported::Stage(stage.stage)));
    }
    let mut function = Function::new();
    let mut at = function.entry();
    let mut driver = Driver::new(stage, header);
    for (index, instr) in program.iter().enumerate() {
        let info = crate::asm::assemble(instr)
            .ok()
            .map(|word| classify(opcode_of(word)));
        let exits = matches!(instr.op, Op::Exit(_));
        assert!(!exits || instr.guard.is_true(), "a guarded exit: {instr}");
        let guarded = !is_never(instr.guard) && !instr.guard.is_true();
        let own_block = guarded && info.is_none_or(|info| info.kind == OpKind::Plain && !info.pure);
        at = match info {
            _ if own_block => driver.conditional(&mut function, at, index, *instr, false)?,
            Some(info) => driver.instruction(&mut function, at, index, *instr, info, false)?,
            // Not guarded (a guarded one has its own block), so its class is not read.
            None if is_never(instr.guard) => at,
            None => {
                driver.dispatch(&mut function, at, index, *instr)?;
                at
            }
        };
        if exits {
            break;
        }
    }
    Builder::new(&mut function, at).terminate(Terminator::Return)?;
    Ok(function)
}

#[cfg(test)]
pub(crate) mod tests;
