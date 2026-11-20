//! Flow analysis: a captured program's control flow rebuilt from SM86's convergence barriers,
//! calls and jump tables, and the IR's basic blocks built from it
//! (`docs/specs/shader-pipeline.md`, "Control-flow recovery"; port slice 10).
//!
//! # The steps
//!
//! 1. **Jump tables** ([`find_jump_tables`], [`JumpTableQuestions`], [`resolve_jump_tables`]):
//!    each `BRX` whose selector is clamped by `IMNMX`, scaled by `SHF` and loaded by `LDC` has
//!    its table read through slice 9's `CbufWord` questions ([`crate::exchange`]). The capture
//!    is extended through the furthest target (slice 3's [`Capture::resolve_jump_tables`]) and,
//!    when it was, the tables are found again in the longer code and read once more, keeping
//!    the first base ([`Capture::jump_targets`]). That second read can ask new questions, so
//!    the exchange can take two rounds; slice 9 caps the rounds at
//!    [`MAX_QUESTION_ROUNDS`](crate::exchange::MAX_QUESTION_ROUNDS) (a guess; verification E6).
//! 2. **The walk** ([`analyze`]): from instruction 0, every reachable instruction, the block
//!    leaders, the reconvergence point of each `BSYNC`, the functions `CALL` names and the
//!    targets of each `BRX`. Two barrier models: the strict `BSSY`/`BSYNC` stack and the
//!    barrier-register model ([`BarrierModel`]); [`flow_graph`] uses the register model (spec;
//!    see the divergence table for how PR #4588 picks between them). The tests check that the
//!    two agree on generated structured programs
//!    (`tests::models::the_models_agree_on_structured_programs`) and on the synthetic programs
//!    the walk's tests run under both (`tests::analysis`'s `accepted` and `refusal` helpers).
//! 3. **Blocks**: each reachable leader starts a block that runs to the next leader or to a
//!    control instruction; a guarded instruction that is not pure (`classify.h`'s `pure`), a
//!    `KILL`, a guarded `CALL` and a conditional `EXIT` get their own block behind a test of
//!    their guard (PR #4588's virtual blocks).
//! 4. **Inlining**: each `CALL` target is a function, inlined at every call site, so the
//!    [`FlowGraph`] is one function. Recursion is refused ([`FlowError::Recursion`]), calls nest
//!    at most [`MAX_INLINE_DEPTH`] deep and the inlined graph has at most [`MAX_FLOW_BLOCKS`]
//!    blocks.
//! 5. **The IR's blocks** ([`FlowGraph::build_ir`]): one IR block per flow block, which the
//!    translators fill from its instructions ([`IrBlocks::sources`]), and the tests that end
//!    the blocks. Building the structured IR the translators fill is slice 11's (owner's
//!    decision, 2026-11-18: "the IR build step in slices 10 (blocks) and 11"):
//!    [`IrBlocks::structure`] (`crate::structure`) keeps the sources.
//!
//! # Sources
//!
//! Ported from PR #4588 @ e5fb3b97 (GPL-3.0-or-later, Copyright 2026 Eden Emulator Project):
//! `eden-pr4588/src/shader_recompiler/frontend/ampere/flow.cpp` and `flow.h` (the walk, both
//! barrier models and the block builder), `classify.h` (the `classify` module) and
//! `program_capture.h` (`TrackCaptureIndirectBranches`, `ResolveCaptureIndirectBranches`'
//! second read). The inlining follows Eden's structurizer, which inlines a call's function at
//! each call site (`eden/src/shader_recompiler/frontend/maxwell/structured_control_flow.cpp` @
//! cf654052, `EndClass::Call`); the IR blocks' tests follow its `IndirectBranchCond` and its
//! conditions.
//!
//! # What a block holds
//!
//! A [`FlowBlock`] covers the instructions `code.start..code.end`, which the translators turn
//! into IR. Control instructions are in it as the reference places them: an `EXIT`, a `BRX`
//! and a conditional block's own instruction are inside; a `BRA`, `BSYNC`, `CALL` and `RET`
//! that end a block are not. `BSSY`, `BREAK`, `WARPSYNC`, `NOP`, `DEPBAR` and `BMOV` stay
//! inside the blocks they fall in and translate to nothing that affects the flow graph, as in
//! the reference (`classify.h`: `BREAK`, `WARPSYNC`, `NOP` and `DEPBAR` are `OpKind::Nop`;
//! `BMOV` is a plain instruction). An instruction whose guard is `!PT` never runs and is
//! skipped. When [`FlowBlock::guard_tested`] is set the block holds one instruction whose
//! guard the graph has already tested, so the translator ignores that guard; any other
//! guarded instruction in a block is a pure one (`classify.h`, `pure`). The translation driver
//! makes it a conditional block of its own (spec, "Instruction translation", "Guards"; the
//! owner's ruling of 2026-11-19, row 2), where PR #4588 keeps the guard as a select
//! ([`crate::translate::driver`]'s "Guards").
//!
//! A `KILL` gets a conditional block of its own that holds it and ends in [`BlockEnd::Kill`]:
//! the discard is that end, not the instruction, as in PR #4588, where the block's end class
//! is `EndClass::Kill` (`flow.cpp`, `BuildFrom`), the structurizer turns it into
//! `DemoteToHelperInvocation` (`structured_control_flow.cpp`, `StatementType::Kill`) and the
//! `KILL` instruction itself translates to nothing (`visitor.cpp`, `case 0x95b`). So the
//! translator turns the `KILL` in such a block into nothing, and [`FlowGraph::build_ir`]
//! emits the demote (pinned by `tests::graph::a_kill_gets_its_own_block` and
//! `tests::ir::a_kill_demotes_after_its_block_then_continues`).
//!
//! # Divergences from PR #4588
//!
//! | What | Here | Reference | Pinned by |
//! |---|---|---|---|
//! | Instructions that are not control instructions | Not decoded: the translators decode them | `LookupRow` and `FirstUnknownBit` for every reachable instruction, before its class | `tests::analysis::plain_instructions_are_not_decoded` |
//! | The uniform `BRA` (`0x547`) | Refused, [`FlowError::UniformBranch`] | No class row: a straight-line instruction whose branch is lost | `tests::analysis::uniform_bra_is_refused` |
//! | Mutual recursion (`A` calls `B` calls `A`) | Refused at inlining, [`FlowError::Recursion`] | Only a call of a function from itself is refused; Eden's structurizer would recurse without end | `tests::inline::mutual_recursion_is_refused` |
//! | Inlining depth and size | At most [`MAX_INLINE_DEPTH`] nested calls and [`MAX_FLOW_BLOCKS`] blocks, both guesses | No bound | `tests::inline::depth_at_its_edge`, `tests::inline::size_at_its_edge` |
//! | A `BRX` target's address | The table's own base ([`BranchTables::base`]) plus 16 bytes per instruction | The header's size plus 16 per instruction (`NativeSourceBase`), which does not match the table when its base is 0 | `tests::ir::indirect_addresses_use_the_table_base` |
//! | A `BRX` table's targets | Per `BRX`; one with no valid target is refused, [`FlowError::NoTarget`] | Falls back to the first table (`NativeIndirectTargets`) when a `BRX` has none | `tests::analysis::brx_without_targets_is_refused` |
//! | The `MOV`s `CALL` and `RET` linkage reads | Must decode as a `MOV` | Read from their bits alone | `tests::analysis::link_mov_must_decode` |
//! | Refused targets | [`FlowError::MisalignedTarget`] or [`FlowError::TargetOutside`] | One message for both | `tests::analysis::target_errors_name_their_reason` |
//! | A `BRX` the decoder refuses | Not tracked by [`find_jump_tables`] (flow analysis refuses it anyway) | Tracked by its bits | `tests::jump_tables::undecodable_brx_is_not_tracked` |
//! | Paths waiting to be walked from one instruction | Merged into one as they are queued, as the join would merge them | Each queued with its own barrier state | `tests::linear::paths_waiting_at_one_instruction_are_merged` (memory), `tests::analysis::stack_paths_waiting_at_one_instruction_keep_the_shorter_stack`, `tests::analysis::paths_waiting_at_one_instruction_merge_their_barriers` (register model), `tests::analysis::two_functions_waiting_at_one_instruction_are_refused`, `tests::analysis::stack_paths_waiting_with_unrelated_stacks_are_refused` |
//! | Which barrier model [`flow_graph`] walks with | The register model only | `CfgBuilder::Build` calls `AnalyzeFlow`, which runs the stack model first and retries with the register model only when the stack model's error text contains "reconvergence" | `tests::models::the_models_agree_on_structured_programs`, and the `accepted` and `refusal` helpers of `tests::analysis` |
//!
//! Merging waiting paths gives the same result for every program both accept: the join at the
//! instruction would merge them the same way, and the walk's result is the fixed point of
//! those merges. For a program with several faults, which one is reported first can differ,
//! since the reference's first fault depends on the order its stack of paths is walked in. A
//! merge of two waiting paths reports what the join would: [`FlowError::SharedCode`] for two
//! functions (`tests::analysis::two_functions_waiting_at_one_instruction_are_refused`), and,
//! in the stack model, [`FlowError::StacksDiffer`] for stacks neither of which is a prefix of
//! the other (`tests::analysis::stack_paths_waiting_with_unrelated_stacks_are_refused`).
//!
//! Walking with the register model alone gives the reference's result in these cases: where
//! the stack model accepts a program, the two models give the same analysis (the tests named in
//! the table check this on the programs they generate or build, not on every program); where it
//! refuses with a "reconvergence" error ([`FlowError::NestedBarrier`],
//! [`FlowError::MissingBarrier`], [`FlowError::TargetsDiffer`], [`FlowError::StacksDiffer`]),
//! the reference retries with the register model, which is what runs here. Where the stack model
//! refuses with any other error, the reference reports that error; the register model here
//! reports its own first fault, which can be another one when a program has several, for the
//! same reason as above.
//!
//! `JMX` and `BPT` have no encoding in any reference (slice 6h keeps them as shells), so they
//! are not recognised: their words are straight-line instructions to the walk. When `JMX` gets
//! an encoding, it resolves like `BRX`.
//!
//! # Linear time
//!
//! Row 121 of the scoping doc: a quadratic search cost 3.4 s on one shader. Every pass here
//! is linear in the program's size: the walk visits an instruction at most once per change of
//! its barrier state (at most 33 in the register model: 16 barriers added and 16 marked
//! conflicting; at most 17 in the stack model, whose stack only shortens), and the reference's
//! scans that were quadratic are done once per program here: the selector search of
//! `TrackCaptureIndirectBranches` (one forward pass that remembers the last match of each
//! pattern per register), the branch-target test of `ReturnRegisterCopiesLink` (one table of
//! targets) and the link-register check (one pass, the first overwrite of each function).
//! Inlining is linear in the inlined graph, which [`MAX_FLOW_BLOCKS`] bounds. Memory is linear
//! too: at most one path waits per instruction, with at most 16 barriers (the reference keeps
//! every path queued, which a chain of `BRX`s naming the same targets grows by up to 16 per `BRX`).
//! Pinned by `tests::linear::*`.

mod analysis;
mod blocks;
mod classify;
mod inline;
mod ir_build;
mod jump_tables;
#[cfg(test)]
mod tests;

use std::collections::BTreeMap;
use std::ops::Range;

use thiserror::Error;

pub use analysis::{Analysis, analyze};
pub(crate) use analysis::{is_never, opcode_of};
pub(crate) use classify::classify;
pub use ir_build::{BlockSource, IrBlocks};
pub use jump_tables::{
    JumpTable, JumpTableQuestions, MAX_TABLE_ENTRIES, ResolvedTable, ResolvedTables,
    find_jump_tables, resolve_jump_tables,
};

use crate::capture::{Capture, MAX_INSTRUCTIONS};
use crate::decode::{BrxSrc, DecodeError, Instruction, PredSrc};
use crate::ir::BuildError;

/// The most instructions flow analysis takes: the capture's cap.
pub const MAX_PROGRAM: u32 = MAX_INSTRUCTIONS;
/// How deeply calls may nest when they are inlined: a call from the program is depth 1. A
/// **guess**, to check against real programs (the reference has no bound).
pub const MAX_INLINE_DEPTH: u32 = 16;
/// The most blocks an inlined [`FlowGraph`] may have. A **guess**: four blocks per instruction
/// of a program at the cap, more than a program without calls can make (at most three per
/// instruction: a leader, a test and a conditional block).
pub const MAX_FLOW_BLOCKS: usize = 4 * MAX_INSTRUCTIONS as usize;

/// Which barrier model the walk uses (`AnalyzeFlowStack`, `AnalyzeFlowRegisters`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BarrierModel {
    /// `BSSY` pushes a barrier and `BSYNC` pops to it: barriers must nest, and paths that
    /// join must carry the same stack or one a prefix of the other.
    Stack,
    /// The sixteen barrier registers, each holding a reconvergence point; paths that join
    /// merge them, and a `BSYNC` needs only one point for its barrier.
    Registers,
}

/// The targets of each `BRX`, by the instruction index of the `BRX`, and the base their table
/// values count from.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BranchTables {
    base: u32,
    targets: BTreeMap<u32, Vec<u32>>,
}

impl BranchTables {
    /// No tables; table values count from `base` bytes before instruction 0.
    pub const fn new(base: u32) -> Self {
        Self {
            base,
            targets: BTreeMap::new(),
        }
    }

    /// Sets the targets of the `BRX` at `branch`, as instruction indices; they are kept
    /// sorted and distinct.
    pub fn insert(&mut self, branch: u32, mut targets: Vec<u32>) {
        targets.sort_unstable();
        targets.dedup();
        self.targets.insert(branch, targets);
    }

    /// The targets of the `BRX` at `branch`, or none.
    pub fn targets(&self, branch: u32) -> &[u32] {
        self.targets.get(&branch).map_or(&[], Vec::as_slice)
    }

    /// The byte offset, from the program's first byte, that a table value naming instruction 0
    /// holds ([`crate::capture::JumpTables::base`]).
    pub const fn base(&self) -> u32 {
        self.base
    }
}

/// Why a program's control flow cannot be rebuilt. Each names the instruction index where the
/// reference's `UnsupportedError` would, and its message's words where it has one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum FlowError {
    /// The program has no instructions or more than [`MAX_PROGRAM`] ("invalid native program
    /// size").
    #[error(
        "a program of {len} instructions cannot be analysed: it must have 1 to {max}",
        max = MAX_PROGRAM
    )]
    ProgramSize {
        /// The instructions given.
        len: usize,
    },
    /// A control instruction does not decode.
    #[error("instruction {index}: {error}")]
    Decode {
        /// The instruction.
        index: u32,
        /// Why it does not decode.
        error: DecodeError,
    },
    /// A path runs past the last instruction ("native program runs past its capture").
    #[error("control runs past instruction {last}, the last of the program")]
    RunsPastEnd {
        /// The last instruction's index.
        last: u32,
    },
    /// A relative branch lands on a byte that does not start an instruction ("native branch
    /// destination alignment").
    #[error(
        "instruction {index} branches to code byte {byte}, which does not start an instruction"
    )]
    MisalignedTarget {
        /// The branch.
        index: u32,
        /// The target, in bytes from instruction 0.
        byte: i64,
    },
    /// A branch, call, reconvergence point or the instruction after a conditional one lies
    /// outside the program ("native branch or call leaves captured program").
    #[error("instruction {index} leads to instruction {target}, outside the program")]
    TargetOutside {
        /// The instruction that leads there.
        index: u32,
        /// The instruction it leads to.
        target: i64,
    },
    /// Two functions reach the same instruction ("native code shared between subroutines").
    #[error("instruction {index} is reached from two functions")]
    SharedCode {
        /// The instruction.
        index: u32,
    },
    /// A barrier or warp-wide matrix instruction runs under a guard ("native predicated
    /// barrier or matrix operation").
    #[error("instruction {index} needs the whole warp but runs under a guard")]
    PredicatedWarpWide {
        /// The instruction.
        index: u32,
    },
    /// A `KILL` has a predicate of its own ("native KILL with extra predicate").
    #[error("the kill at instruction {index} has a predicate of its own")]
    KillPredicate {
        /// The `KILL`.
        index: u32,
    },
    /// An `EXIT` has bit 84 or 85 set ("native conditional/special EXIT").
    #[error("the exit at instruction {index} has a special mode")]
    SpecialExit {
        /// The `EXIT`.
        index: u32,
    },
    /// An `EXIT` with a guard and a predicate of its own, which needs a test in front of the
    /// next instruction that the next instruction needs too ("native EXIT with two
    /// predicates").
    #[error("the exit at instruction {index} has two predicates the flow graph cannot test")]
    TwoPredicateExit {
        /// The `EXIT`.
        index: u32,
    },
    /// A branch on a uniform predicate or with its `.U` bit ("native uniform branch").
    #[error("the branch at instruction {index} is a uniform branch")]
    UniformBranch {
        /// The branch.
        index: u32,
    },
    /// A `CALL` that is not an unconditional `.NOINC` call ("native call mode").
    #[error("the call at instruction {index} has an unsupported mode")]
    CallMode {
        /// The `CALL`.
        index: u32,
    },
    /// A `CALL` whose target is not an instruction of the program or which sets up no return
    /// address ("native call linkage").
    #[error("the call at instruction {index} has no return address set up or no target")]
    CallLinkage {
        /// The `CALL`.
        index: u32,
    },
    /// Two calls of one function set up its return address in different registers ("native
    /// inconsistent call linkage").
    #[error("the call at instruction {index} passes its return address in another register")]
    InconsistentLinkage {
        /// The `CALL`.
        index: u32,
    },
    /// A function calls itself, directly ("native recursive call") or through others.
    #[error("the call at instruction {index} is recursive")]
    Recursion {
        /// The `CALL`.
        index: u32,
    },
    /// A `RET` outside a function, guarded by a predicate of its own, of an unsupported mode
    /// or not relative to the program's start ("native return linkage").
    #[error("the return at instruction {index} has an unsupported linkage")]
    ReturnLinkage {
        /// The `RET`.
        index: u32,
    },
    /// A `BRX` that is guarded, has a predicate of its own or does not count from the
    /// program's start ("native computed branch mode").
    #[error("the indirect branch at instruction {index} has an unsupported mode")]
    BrxMode {
        /// The `BRX`.
        index: u32,
    },
    /// A `BRX` with no table target inside the program ("native computed branch without a
    /// resolved table").
    #[error("the indirect branch at instruction {index} has no target in the program")]
    NoTarget {
        /// The `BRX`.
        index: u32,
    },
    /// A `BRXU` through `URZ` ("native computed branch through URZ").
    #[error("the indirect branch at instruction {index} reads its target from urz")]
    BrxThroughUrz {
        /// The `BRX`.
        index: u32,
    },
    /// A `BSSY` that is guarded or has a predicate of its own ("native predicated BSSY").
    #[error("the bssy at instruction {index} is predicated")]
    PredicatedBssy {
        /// The `BSSY`.
        index: u32,
    },
    /// A `BSYNC` that is guarded or has a predicate of its own ("native predicated BSYNC").
    #[error("the bsync at instruction {index} is predicated")]
    PredicatedBsync {
        /// The `BSYNC`.
        index: u32,
    },
    /// Stack model: a `BSSY` of a barrier already on the stack ("native nested reconvergence
    /// token"). The reference also refuses a seventeenth barrier on the stack, which this
    /// check keeps; with sixteen barrier registers, a seventeenth always repeats one, so that
    /// part never decides.
    #[error("the bssy at instruction {index} sets up a barrier already in use")]
    NestedBarrier {
        /// The `BSSY`.
        index: u32,
    },
    /// A `BSYNC` whose barrier no path sets up ("native reconvergence token").
    #[error("the bsync at instruction {index} waits on a barrier no path set up")]
    MissingBarrier {
        /// The `BSYNC`.
        index: u32,
    },
    /// A `BSYNC` that paths give different reconvergence points ("native reconvergence target
    /// differs between paths").
    #[error("the bsync at instruction {index} reconverges at different points on different paths")]
    TargetsDiffer {
        /// The `BSYNC`.
        index: u32,
    },
    /// Stack model: paths join with stacks neither of which is a prefix of the other ("native
    /// reconvergence stacks differ at a join").
    #[error("paths join at instruction {index} with different barrier stacks")]
    StacksDiffer {
        /// The join.
        index: u32,
    },
    /// A `RET` returns through a register that is neither the link register nor a copy of it
    /// ("native return through a different register").
    #[error("the return at instruction {index} does not return through the link register")]
    ReturnRegister {
        /// The `RET`.
        index: u32,
    },
    /// A function writes its link register ("native subroutine overwrites its return
    /// address").
    #[error("instruction {index} overwrites its function's return address")]
    LinkOverwritten {
        /// The instruction.
        index: u32,
    },
    /// Inlining the call nests calls deeper than [`MAX_INLINE_DEPTH`].
    #[error(
        "the call at instruction {index} nests calls deeper than {max}",
        max = MAX_INLINE_DEPTH
    )]
    InlineDepth {
        /// The `CALL`.
        index: u32,
    },
    /// Inlining the call makes more than [`MAX_FLOW_BLOCKS`] blocks.
    #[error(
        "inlining the call at instruction {index} makes more than {max} blocks",
        max = MAX_FLOW_BLOCKS
    )]
    InlineSize {
        /// The `CALL`.
        index: u32,
    },
    /// A block leads to an instruction that has no block. No public function reaches it:
    /// blocks are built only from the walk's own result, which walks every instruction a
    /// branch, call, reconvergence point or table names. It is an error, not a panic, in case
    /// that ever stops holding.
    #[error("instruction {index} leads to instruction {target}, which the walk did not reach")]
    Unwalked {
        /// The instruction that leads there.
        index: u32,
        /// The instruction it leads to.
        target: u32,
    },
    /// A jump-table word was not asked through the exchange.
    #[error("constant buffer {bank} word {offset:#x} of a jump table has not been answered")]
    MissingAnswer {
        /// The constant-buffer bank.
        bank: u8,
        /// The word's byte offset.
        offset: u32,
    },
    /// The IR could not take another block or instruction.
    #[error("the IR cannot hold the flow graph: {0}")]
    Ir(#[from] BuildError),
    /// Flow analysis broke one of its own invariants: a fault in this module, not in the
    /// program. No public function reaches it; it is an error, not a panic, in case that ever
    /// stops holding.
    #[error("flow analysis broke an invariant of its own: {0}")]
    Internal(Invariant),
}

/// Which of flow analysis's own invariants broke ([`FlowError::Internal`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum Invariant {
    /// More blocks were made than a `u32` block id can name.
    #[error("more blocks than a block id can name")]
    BlockCount,
    /// A block was made but never filled.
    #[error("block {0} was never filled")]
    Unfilled(u32),
    /// A block id names no block.
    #[error("block {0} does not exist")]
    NoBlock(u32),
}

/// A block of the inlined [`FlowGraph`], by its index.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BlockIndex(u32);

impl BlockIndex {
    /// The block's position in [`FlowGraph::blocks`].
    pub const fn index(self) -> u32 {
        self.0
    }
}

/// Where an indirect branch goes when its register holds `address`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct IndirectTarget {
    /// The table value naming the target: [`BranchTables::base`] plus 16 bytes per
    /// instruction.
    pub address: u32,
    /// The block there.
    pub block: BlockIndex,
}

/// How control leaves a [`FlowBlock`].
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum BlockEnd {
    /// Continue at a block.
    Jump(BlockIndex),
    /// Continue at `then` where `cond` holds, else at `otherwise`. `cond` is never `PT` (that
    /// is a [`BlockEnd::Jump`]), and is `!PT` only when it is the own predicate of a `BRA` or an
    /// `EXIT` (an instruction whose guard is `!PT` never runs and is skipped).
    Branch {
        /// The predicate tested.
        cond: PredSrc,
        /// Where control goes when it holds.
        then: BlockIndex,
        /// Where control goes when it does not.
        otherwise: BlockIndex,
    },
    /// `BRX`: continue at the target whose address the register holds, in address order.
    Indirect {
        /// The register.
        src: BrxSrc,
        /// The targets, in ascending order of instruction.
        targets: Vec<IndirectTarget>,
    },
    /// The thread exits.
    Exit,
    /// The block's `KILL` discards: the thread becomes a helper invocation (the IR's
    /// `DemoteToHelperInvocation`), then control continues at the block (see the module docs,
    /// "What a block holds").
    Kill(BlockIndex),
}

/// A block of the flow graph: a run of instructions and how control leaves it.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct FlowBlock {
    /// The instructions it holds, by index (see the module docs, "What a block holds"); empty
    /// for a block that only tests a predicate.
    pub code: Range<u32>,
    /// Whether the graph has tested the guard of the block's one instruction.
    pub guard_tested: bool,
    /// How control leaves it.
    pub end: BlockEnd,
}

/// A program's control flow as one function, every call inlined: block 0 is the entry.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct FlowGraph {
    blocks: Vec<FlowBlock>,
}

impl FlowGraph {
    /// The blocks, the entry first.
    pub fn blocks(&self) -> &[FlowBlock] {
        &self.blocks
    }

    /// The block at `index`.
    pub fn block(&self, index: BlockIndex) -> Option<&FlowBlock> {
        usize::try_from(index.0)
            .ok()
            .and_then(|slot| self.blocks.get(slot))
    }
}

/// Rebuilds the control flow of `code`, the instructions of a captured program, with `tables`
/// giving each `BRX`'s targets: the register-model walk, the blocks and the inlining (see the
/// module docs).
pub fn flow_graph(code: &[Instruction], tables: &BranchTables) -> Result<FlowGraph, FlowError> {
    let analysis = analyze(code, tables, BarrierModel::Registers)?;
    let raw = blocks::build(code, &analysis, tables)?;
    inline::inline(&raw)
}

/// [`flow_graph`] over a capture's instructions.
pub fn capture_flow_graph(
    capture: &Capture,
    tables: &BranchTables,
) -> Result<FlowGraph, FlowError> {
    let code: Vec<Instruction> = capture.instructions().collect();
    flow_graph(&code, tables)
}
