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
//! e5fb3b97, GPL-3.0-or-later, Copyright 2026 Eden Emulator Project): `translate.cpp` (the
//! per-block walk over the control-flow builder's blocks) and `visitor.cpp` (`Visitor::Run`: the
//! `guard.Never()` return, the dispatch by operation); cross-checked with Eden's
//! `frontend/maxwell/translate/translate.cpp` @ cf654052 (`Translate`: each block's instructions
//! in address order).
//!
//! # The order of the checks
//!
//! 1. The stage: vertex and pixel programs only ([`TranslateError::Stage`]), as slice 1 had it.
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
//! # Guards
//!
//! An instruction under `!PT` never runs and becomes nothing, as `Visitor::Run` returns when
//! `guard.Never()` (flow analysis leaves such an instruction in its block's range and skips it
//! the same way, `flow::is_never`). An instruction under `PT`, and the one instruction of a block
//! whose guard flow analysis has tested ([`crate::flow::FlowBlock::guard_tested`]: a guarded
//! instruction that is not pure, a conditional `EXIT`, a `KILL`), is translated as if unguarded.
//! Any other guarded instruction is a pure one flow analysis left in its block, and becomes a
//! conditional block (the owner's ruling of 2026-11-19, row 2, where PR #4588 keeps a select,
//! `guard_select`): the driver reads the guard (a uniform predicate of the same number for a
//! uniform datapath instruction, `classify.h`'s `uniform`: `S2UR`, `VOTEU` and the uniform ALU
//! forms; a predicate for the others), splits the block with the shared guard helper
//! ([`super::split_for_guard`]), translates the instruction into the new block as if
//! unguarded, and carries on in the block after it. Every group's translator is so handed its
//! instruction with the guard already handled.
//!
//! # Dispatch
//!
//! By the decoded operation, in this order: `EXIT` (`super`'s `Translator::exit`); the integer
//! ALU group ([`super::integer::is_integer`], `MOV` included); the float group
//! ([`super::float::in_group`]); the texture group ([`super::texture::in_group`], with one
//! [`DescriptorBuffer`] for the program); the control and warp group
//! ([`super::control::in_group`]); the memory group, whose translator (slice 12e) is not joined
//! yet, refused by name ([`Unsupported::MemoryGroup`]: `LD`, `ST`, `LDC`, `ATOM`, `CCTL`,
//! `MEMBAR`, `LDGSTS`, `LDGDEPBAR`, `ERRBAR`, `ATOMS.POPC`); the attribute group (`super`'s
//! `GROUPS`). An operation none of them takes (the half and double group, the uniform group,
//! the tensor group) is refused by its mnemonic ([`Unsupported::Instruction`]). The groups are
//! disjoint, so the order decides nothing but speed
//! (`tests::the_groups_are_disjoint`).
//!
//! # Bounds
//!
//! Inlining copies a function's blocks to every call site (slice 10), so a program's inlined
//! graph can hold far more instructions than the program: a function of 2^15 instructions
//! called from 2^15 sites would be 2^30 translations. [`MAX_TRANSLATED_INSTRUCTIONS`] bounds the
//! count before anything is decoded or translated; the IR the translators write is bounded by
//! it times the most IR one instruction writes (a `REDUX`'s 32 shuffles are the largest). The
//! decoded instructions are kept once per instruction of the program, at most
//! [`crate::capture::MAX_INSTRUCTIONS`].
//!
//! # Divergences from PR #4588
//!
//! | What | Here | Reference | Pinned by |
//! |---|---|---|---|
//! | A guarded pure instruction | A conditional block (the owner's ruling of 2026-11-19) | A select of each value it writes (`guard_select`) | `tests::a_guarded_pure_instruction_becomes_a_conditional_block` |
//! | The instructions translated | At most [`MAX_TRANSLATED_INSTRUCTIONS`], counted over the inlined graph (a **guess**, four per instruction of a program at the capture's cap, as [`crate::flow::MAX_FLOW_BLOCKS`] is four blocks per instruction) | No bound | `tests::the_translation_budget_at_its_edge` |
//! | Which undecodable instruction is refused | The lowest-numbered one any flow block holds, after flow analysis | The first its flow walk meets (`LookupRow`, `FirstUnknownBit`), in walk order | `tests::the_lowest_undecodable_instruction_is_refused` |
//! | Stages | Vertex and pixel only | Every stage | `tests::only_vertex_and_pixel_stages_translate` |

use super::control::{self, Emitted};
use super::float::{self, FloatError};
use super::integer::{self, GuardMode, IntegerError};
use super::texture::{self, DescriptorBuffer, TextureError};
use super::{
    Failure, GuestStage, Stage, TranslateError, Translator, Unsupported, source, split_for_guard,
};
use crate::capture::MAX_INSTRUCTIONS;
use crate::decode::{Instr, Instruction, Op, decode};
use crate::flow::{
    BranchTables, FlowError, FlowGraph, Invariant, IrBlocks, classify, flow_graph, is_never,
    opcode_of,
};
use crate::ir::{BlockId, Builder, Function};
use crate::sph::PixelIo;

/// The most instructions [`translate_flow`] translates in one program, counted over the flow
/// graph with every call inlined (see the module docs, "Bounds"). A **guess**: four per
/// instruction of a program at the capture's cap, the same allowance
/// [`crate::flow::MAX_FLOW_BLOCKS`] makes for blocks; real programs would confirm or move it.
pub const MAX_TRANSLATED_INSTRUCTIONS: u32 = 4 * MAX_INSTRUCTIONS;

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
    if !matches!(stage.stage, GuestStage::Vertex | GuestStage::Pixel) {
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
            // Flow analysis made every block from `code`, and `decode_held` decoded every
            // instruction a block holds, so neither lookup fails.
            let slot = usize::try_from(index)
                .map_err(|_| FlowError::Internal(Invariant::NoBlock(source.flow.index())))?;
            let (Some(word), Some(Some(instr))) = (code.get(slot), decoded.get(slot)) else {
                return Err(FlowError::Internal(Invariant::NoBlock(source.flow.index())).into());
            };
            let uniform = classify(opcode_of(*word)).uniform;
            at = driver.instruction(function, at, slot, *instr, uniform, flow.guard_tested)?;
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
    /// block `at`, whose flow block's `guard_tested` is `guard_tested`; `uniform` says its guard
    /// is a uniform predicate (`classify.h`'s `uniform`). See the module docs, "Guards". Returns
    /// the block where translation continues.
    fn instruction(
        &mut self,
        function: &mut Function,
        at: BlockId,
        index: usize,
        instr: Instr,
        uniform: bool,
        guard_tested: bool,
    ) -> Result<BlockId, TranslateError> {
        if is_never(instr.guard) {
            return Ok(at);
        }
        if guard_tested || instr.guard.is_true() {
            self.dispatch(function, at, index, instr)?;
            return Ok(at);
        }
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
        if in_memory_group(&op) {
            return Err(unsupported(Unsupported::MemoryGroup(op.mnemonic())));
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

/// Whether `op` is of the memory group (slice 6e's operations), which the driver refuses by
/// name until slice 12e's translator is joined (see the module docs, "Dispatch").
const fn in_memory_group(op: &Op) -> bool {
    matches!(
        op,
        Op::Ld(_)
            | Op::St(_)
            | Op::Ldc(_)
            | Op::Atom(_)
            | Op::CCtl(_)
            | Op::MemBar(_)
            | Op::LdGsts(_)
            | Op::LdgDepBar(_)
            | Op::ErrBar(_)
            | Op::AtomsPopc(_)
    )
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
/// returns, for the tests' hand-built instructions no word encodes: the block flow analysis
/// would build for such a program, except that a guarded instruction that is not pure gets its
/// conditional block from the driver, not from flow analysis (the same blocks). A guard is read
/// as a uniform predicate when the assembler encodes the instruction and `classify.h` says so.
/// Instructions after the first unguarded `EXIT` are ignored, as flow analysis never reaches
/// them; a guarded `EXIT` is not supported here.
#[cfg(test)]
pub(crate) fn translate_decoded(
    stage: Stage,
    header: Option<&PixelIo>,
    program: &[Instr],
) -> Result<Function, TranslateError> {
    use crate::ir::Terminator;
    if !matches!(stage.stage, GuestStage::Vertex | GuestStage::Pixel) {
        return Err(TranslateError::Stage(Unsupported::Stage(stage.stage)));
    }
    let mut function = Function::new();
    let mut at = function.entry();
    let mut driver = Driver::new(stage, header);
    for (index, instr) in program.iter().enumerate() {
        let uniform =
            crate::asm::assemble(instr).is_ok_and(|word| classify(opcode_of(word)).uniform);
        let exits = matches!(instr.op, Op::Exit(_));
        assert!(!exits || instr.guard.is_true(), "a guarded exit: {instr}");
        at = driver.instruction(&mut function, at, index, *instr, uniform, false)?;
        if exits {
            break;
        }
    }
    Builder::new(&mut function, at).terminate(Terminator::Return)?;
    Ok(function)
}

#[cfg(test)]
pub(crate) mod tests;
