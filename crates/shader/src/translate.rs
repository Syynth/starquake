//! Instruction translation: decoded instructions to IR, for straight-line programs.
//!
//! Ported by behaviour from PR #4588's native frontend
//! (`eden-pr4588/src/shader_recompiler/frontend/ampere/` @ e5fb3b97, GPL-3.0-or-later):
//! `ops_move.cpp` (`MOV`) and `ops_attr.cpp` (`EXIT`) here, and one module per instruction
//! group beside this one (spec, "Instruction translation"): `attribute` (slice 12g: `ALD`,
//! `AST`, `IPA`, `AL2P`, `S2R`, `CS2R`, `PIXLD`, and the deferred `OUT`, `ISBERD` and
//! `ISBEWR`), joined by `GROUPS`, and [`float`] (slice 12b: the 32-bit float and conversion
//! group, slice 1's `FFMA` included), whose refusals are [`Unsupported::Float`]. Each of the
//! PR's `Require` checks on these forms is kept with its constant and turns into an
//! [`Unsupported`] rejection naming the same thing, except the colour-register bound of `EXIT`,
//! which cannot fail here (see `Translator::exit`).
//!
//! **Guards.** An instruction of the float group ([`float::in_group`]) is handed to [`float`]
//! before its guard is read, and takes its own guard: a guarded one becomes a conditional
//! block, as the spec says, and one under `@!PT` is checked and put in a block that never runs
//! (that module's "Guards" and its divergence table). For every other instruction, in this
//! order: an instruction under `@!PT` never runs and translates to nothing, as PR #4588's
//! `Visitor::Run` returns when `guard.Never()` (`visitor.cpp`; `flow.h` defines it as
//! `index == 7 && negated`); one under `PT` is translated as is; a guarded `EXIT` is rejected
//! ([`Unsupported::Guard`], starquake's; see the list below). Any other guard is read
//! once, before the instruction's own IR, as `Visitor::Run` sets `guard_select` before it
//! dispatches, and the instruction is translated by its class in flow analysis's table
//! (`flow`'s `classify`, the PR's `classify.h`, `Classify`), looked up by the 12-bit opcode the
//! decoder reads the operation from (`opcode`):
//!
//! - a **pure** instruction (`OpInfo::pure`: of the ones translated here `MOV`, `ALD`, `IPA`,
//!   `S2R`, `S2UR` and `CS2R`) keeps its guard as a select: every register it writes
//!   gets `SelectU32` of the guard, the new value and the register's old value, and a uniform
//!   register the same with its old uniform value, as the PR's `guard_select` does
//!   (`visitor.cpp`, `Visitor::X` and `Visitor::UR`: `Select(*guard_select, value,
//!   GetReg(dest))`). None of these writes a predicate. The guard of a pure uniform datapath
//!   instruction (`OpInfo::uniform`: `S2UR` here) is read as the uniform predicate of the same
//!   number, `GetUniformPred` (the PR's `UPred(guard.index, guard.negated)`), the others' as a
//!   predicate through `flow`'s `read_pred`;
//! - **any other** instruction (`AST`, `AL2P` and `PIXLD` here, and an operation `opcode` does
//!   not name, which no translator takes) becomes a conditional block: the block it was in
//!   branches on the guard to a block of its own, which holds the instruction and jumps to a
//!   new block where translation continues. In the PR, flow analysis makes that block
//!   (`flow.cpp`: "Guarded side effect: becomes its own conditional block").
//!
//! The conditional block is built only by [`translate`] and [`translate_with_header`], the
//! straight-line entry, which has no flow graph and so builds the one block flow analysis
//! would have made; it starts a new block, so it needs a block that is still open. The flow
//! graph's fill path (slice 11) translates one instruction at a time into blocks
//! [`crate::flow::FlowGraph::build_ir`] has already terminated, where the contract of `flow`'s
//! "What a block holds" applies: in a block whose [`crate::flow::FlowBlock::guard_tested`] is
//! set the graph has tested the guard, so the translator ignores it, and every other guarded
//! instruction is pure, so only the select runs. An entry for that path takes `guard_tested`
//! as [`control::translate_control`] does and refuses a non-pure instruction under an untested
//! guard ([`control::ControlUnsupported::UntestedGuard`]); building it for this group is the
//! fill path's work, not done here.
//!
//! **Owner question:** the spec says, of instruction translation, "an instruction under a
//! predicate becomes a conditional block". A guarded pure instruction becomes a select instead,
//! as in the PR and as `flow`'s block contract and [`control`]'s `VOTEU` already do; the
//! registers come out the same, since a block that does not run leaves each register at its old
//! value, the select's other side. The float group follows the spec sentence instead (see
//! "Guards" above). Pinned by `tests::guarded_pure_instructions_become_selects`,
//! `tests::guarded_side_effects_become_conditional_blocks`,
//! `tests::a_guard_on_each_pure_uniform_op_reads_a_uniform_predicate`,
//! `tests::guards_classify_by_the_flow_table`, `tests::a_guarded_exit_is_rejected` and
//! `tests::never_guarded_instructions_are_skipped`.
//!
//! The instruction groups' translators (port slices 12a to 12h) each have a module here that
//! translates one instruction of a flow block (slice 10) at a time: [`control`] for the
//! control and warp group, whose instructions [`translate`] keeps refusing, and [`integer`] for
//! the integer ALU group (slice 12a). The attribute group (`attribute`) is joined to
//! [`translate`] through `GROUPS` instead.
//!
//! Slice 1 handles linear flow only (spec, port slice 1). Where starquake is stricter than the
//! PR, because the PR resolves the case in a part not built yet:
//!
//! - a guarded `EXIT` is rejected, and so is an `EXIT` with its own predicate other than `PT`
//!   (the PR's control-flow builder resolves both, `ops_attr.cpp`: "extra exit condition");
//! - the program must end with its first `EXIT`: no `EXIT` and instructions after it are
//!   rejected (the extent of a program is the capture's job, slice 3);
//! - `MOV` takes only register and immediate sources: a uniform-register or
//!   constant-buffer source, which the decoder reads since slice 4, is rejected (the PR reads
//!   both, `visitor.cpp`, `Visitor::Value`; constant buffers and uniform registers come with
//!   the passes of slices 12 and 13);
//! - every other instruction is rejected by its mnemonic.
//!
//! Pinned by `tests::exit_modes_and_predicates_are_rejected`,
//! `the_program_ends_at_its_only_exit`, `uniform_and_constant_buffer_sources_are_rejected` and
//! `other_instructions_are_rejected_by_mnemonic`. The group modules list their own divergences.

mod attribute;
pub mod control;
pub mod float;
pub mod integer;
#[cfg(test)]
mod tests;

use std::fmt::{self, Display, Formatter};

use thiserror::Error;

use crate::decode::{
    AluForm, EXIT_OPCODE, Gpr, Instr, MOV_OPCODE, MovSrc, Op, OpExit, OpMov, PredSrc, SpecialReg,
    UniformPred, UniformReg,
};
use crate::flow::{OpInfo, classify};
use crate::ir::{BlockId, BuildError, Builder, Function, Imm, Opcode, Operand, Terminator, Value};
use crate::sph::PixelIo;

pub use attribute::{AttributeRejection, Deferral};

/// The guest's pipeline stage of a program (spec, "What the GPU front/back protocol and engine
/// state (#12) gets": `GuestStage`; slot types from `SET_PIPELINE_SHADER`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GuestStage {
    /// Vertex, with a cull-before-fetch pass.
    VertexCullBeforeFetch,
    /// Vertex.
    Vertex,
    /// Tessellation control.
    TessellationInit,
    /// Tessellation evaluation.
    Tessellation,
    /// Geometry.
    Geometry,
    /// Pixel (fragment).
    Pixel,
    /// Compute.
    Compute,
}

impl GuestStage {
    /// The stage's name in plain words, as error messages print it.
    pub const fn name(self) -> &'static str {
        match self {
            Self::VertexCullBeforeFetch => "vertex (cull before fetch)",
            Self::Vertex => "vertex",
            Self::TessellationInit => "tessellation control",
            Self::Tessellation => "tessellation evaluation",
            Self::Geometry => "geometry",
            Self::Pixel => "pixel",
            Self::Compute => "compute",
        }
    }
}

impl Display for GuestStage {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// How many colour targets a pixel program can write (`ops_attr.cpp`, `EXIT`: `render_target <
/// 8`).
pub const COLOR_TARGETS: usize = 8;

/// The bits of a [`ColorOutputs`] target entry that name components; higher bits are ignored.
pub const COMPONENT_MASK: u8 = 0xf;

/// Which components of each colour target a pixel program writes: the output map of the
/// program's header (the SPH's `omap`), as [`crate::sph::PixelIo::color_outputs`] reads it from
/// a parsed header. Bit `c` of `targets[t]` means component `c` of target `t`; bits above
/// [`COMPONENT_MASK`] are ignored.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct ColorOutputs {
    /// Four bits per target.
    pub targets: [u8; COLOR_TARGETS],
}

/// What a program is translated for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Stage {
    /// The guest stage.
    pub stage: GuestStage,
    /// For a pixel program, which colour components it writes; ignored for other stages.
    pub color_outputs: ColorOutputs,
}

/// A form or field value the translator does not handle yet.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum Unsupported {
    /// The stage is not handled yet.
    #[error("{0} programs are not supported yet")]
    Stage(GuestStage),
    /// An instruction no translator handles yet: in [`translate`], one that is neither `MOV`
    /// and `EXIT`, nor in the [`float`] group, nor taken by a group of `GROUPS`; in [`control::translate_control`],
    /// `BMOV`, `NANOSLEEP`, `JMX` and `BPT`.
    #[error("{0}")]
    Instruction(&'static str),
    /// A control or warp form [`control::translate_control`] does not handle.
    #[error("{0}")]
    Control(#[from] control::ControlUnsupported),
    /// A uniform-register or constant-buffer source of `MOV` (PR #4588 reads both:
    /// `visitor.cpp`, `Visitor::Value`).
    #[error("a uniform-register or constant-buffer source")]
    Source,
    /// An `EXIT` under a guard other than `PT` and `!PT`: a conditional exit is control flow,
    /// which this straight-line translator does not build (every other guarded instruction
    /// becomes a select or a conditional block; see the module docs, "Guards").
    #[error("a guarded exit")]
    Guard,
    /// A `MOV` lane mask other than `0xf` (`ops_move.cpp`: "native MOV lane mask").
    #[error("a mov with a partial lane mask")]
    MovLaneMask,
    /// An attribute access with an offset register (`ops_misc.cpp`: "native indexed attribute
    /// access").
    #[error("an indexed attribute access")]
    IndexedAttribute,
    /// A patch or physical attribute access (`ops_misc.cpp`: "native patch or physical
    /// attribute access").
    #[error("a patch or physical attribute access")]
    PatchOrPhysical,
    /// `ALD` of an output (`ops_misc.cpp`: "native output attribute read").
    #[error("an ald of an output attribute")]
    OutputRead,
    /// An attribute address that is not word aligned or runs past `0x400` (`ops_misc.cpp`:
    /// "native attribute address", for `ALD` and `AST`; `IPA`'s one word is checked the same way,
    /// starquake's).
    #[error("attribute address {addr:#x} for a {comps}-word access")]
    AttributeAddress {
        /// The address.
        addr: u16,
        /// The word count.
        comps: u8,
    },
    /// Registers past R254 (`ops_misc.cpp`: "native attribute register range").
    #[error("attribute registers from {first} for {comps} words")]
    AttributeRegisters {
        /// The first register.
        first: Gpr,
        /// The word count.
        comps: u8,
    },
    /// A form of the attribute and system-value group (slice 12g) that is not handled.
    #[error(transparent)]
    Attribute(#[from] AttributeRejection),
    /// `EXIT` with a mode bit (`ops_attr.cpp`: "native conditional/special EXIT").
    #[error("an exit with a mode bit")]
    ExitMode,
    /// `EXIT` with a predicate.
    #[error("a conditional exit (flow analysis is not built yet)")]
    ConditionalExit,
    /// A special register other than the ones handled.
    #[error("special register {0}")]
    SpecialRegister(SpecialReg),
    /// A form of the float and conversion group that is not translated ([`float`]).
    #[error("{0}")]
    Float(float::FloatUnsupported),
}

/// Why a program was not translated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum TranslateError {
    /// An instruction is not handled.
    #[error("instruction {index} ({instr}): {what} is not supported")]
    Unsupported {
        /// The instruction's index in the program.
        index: usize,
        /// The instruction.
        instr: Instr,
        /// What is not supported.
        what: Unsupported,
    },
    /// An instruction follows the `EXIT` that ends the program.
    #[error("instruction {index} follows the exit at {exit}")]
    AfterExit {
        /// The first instruction after the `EXIT`.
        index: usize,
        /// The `EXIT`'s index.
        exit: usize,
    },
    /// The program has no `EXIT`.
    #[error("the program has no exit")]
    NoExit,
    /// The stage is not handled.
    #[error("{0}")]
    Stage(Unsupported),
    /// The IR could not be built: an arena is full, or a translator built a malformed
    /// instruction (a bug in this module).
    #[error(transparent)]
    Build(#[from] BuildError),
}

/// The bound PR #4588 puts on attribute addresses: an access must end at or before `0x400`
/// (`ops_misc.cpp`, `ALD`/`AST`: `address + components * 4 <= 0x400`).
pub const ATTRIBUTE_SPACE_END: u32 = 0x400;
/// The number of the first register an access may not reach: R255 is `RZ` (`ops_misc.cpp`:
/// `dest + components <= 255`).
pub const REGISTER_LIMIT: u32 = 255;

/// Translates a straight-line program that ends with its first `EXIT`, into blocks that end in
/// a return: one block, and two more (a then block and an after block) for each guarded
/// instruction that is not pure and for each instruction of the [`float`] group under a guard
/// other than `PT`, `@!PT` included (see the module docs, "Guards").
/// Without the program's header, an `IPA` of a generic attribute is rejected
/// ([`AttributeRejection::IpaNoHeader`]); [`translate_with_header`] takes it.
pub fn translate(stage: Stage, program: &[Instr]) -> Result<Function, TranslateError> {
    translate_inner(stage, None, program)
}

/// [`translate`], with the pixel program's header, whose input map tells `IPA` how each
/// generic attribute is interpolated (PR #4588's `IPA` reads it, `ops_attr.cpp`:
/// `sph.ps.GenericInputMap`).
pub fn translate_with_header(
    stage: Stage,
    header: &PixelIo,
    program: &[Instr],
) -> Result<Function, TranslateError> {
    translate_inner(stage, Some(header), program)
}

fn translate_inner(
    stage: Stage,
    header: Option<&PixelIo>,
    program: &[Instr],
) -> Result<Function, TranslateError> {
    if !matches!(stage.stage, GuestStage::Vertex | GuestStage::Pixel) {
        return Err(TranslateError::Stage(Unsupported::Stage(stage.stage)));
    }
    let mut function = Function::new();
    let entry = function.entry();
    let mut t = Translator {
        b: Builder::new(&mut function, entry),
        stage,
        header: header.copied(),
        select: None,
    };
    for (index, instr) in program.iter().enumerate() {
        let fail = |what| TranslateError::Unsupported {
            index,
            instr: *instr,
            what,
        };
        if float::in_group(&instr.op) {
            match float::translate_instr(&mut t.b, *instr, false) {
                Ok(()) => continue,
                Err(float::FloatError::Unsupported(what)) => {
                    return Err(fail(Unsupported::Float(what)));
                }
                Err(float::FloatError::Build(error)) => return Err(error.into()),
            }
        }
        let after = match guarding(*instr) {
            Guarding::Never => continue,
            Guarding::Always => None,
            Guarding::Select { .. } | Guarding::Block { .. } if matches!(instr.op, Op::Exit(_)) => {
                return Err(fail(Unsupported::Guard));
            }
            Guarding::Select { uniform } => {
                t.select = Some(t.read_guard(instr.guard, uniform)?);
                None
            }
            Guarding::Block { uniform } => Some(t.begin_guard(instr.guard, uniform)?),
        };
        let done = match instr.op {
            Op::Mov(op) => t.mov(op).map(|()| false),
            Op::Exit(op) => t.exit(op).map(|()| true),
            other => t.group(other).map(|()| false),
        };
        t.select = None;
        if let (Some(after), Ok(false)) = (after, &done) {
            t.end_guard(after)?;
        }
        match done {
            Ok(false) => {}
            Ok(true) => {
                if let Some(next) = index.checked_add(1)
                    && next < program.len()
                {
                    return Err(TranslateError::AfterExit {
                        index: next,
                        exit: index,
                    });
                }
                t.b.terminate(Terminator::Return)?;
                return Ok(function);
            }
            Err(Failure::Unsupported(what)) => return Err(fail(what)),
            Err(Failure::Build(error)) => return Err(error.into()),
        }
    }
    Err(TranslateError::NoExit)
}

/// An instruction translator's failure, before the instruction is attached.
enum Failure {
    Unsupported(Unsupported),
    Build(BuildError),
}

impl From<BuildError> for Failure {
    fn from(error: BuildError) -> Self {
        Self::Build(error)
    }
}

impl From<Unsupported> for Failure {
    fn from(what: Unsupported) -> Self {
        Self::Unsupported(what)
    }
}

fn require(ok: bool, what: Unsupported) -> Result<(), Failure> {
    if ok { Ok(()) } else { Err(what.into()) }
}

/// The register `count` places after `first`, if every one of them is below `RZ`.
fn register_run(first: Gpr, count: u32) -> Option<u32> {
    let end = u32::from(first.index()).checked_add(count)?;
    (end <= REGISTER_LIMIT).then_some(end)
}

/// A 32-bit immediate operand.
const fn imm32(bits: u32) -> Operand {
    Operand::Imm(Imm::U32(bits))
}

/// Whether an instruction under `guard` never runs: `@!PT` (PR #4588's `Guard::Never`,
/// `flow.h`: `index == 7 && negated`).
const fn never_runs(guard: PredSrc) -> bool {
    guard.reg.is_true() && guard.negated
}

/// How an instruction's guard is translated (see the module docs, "Guards").
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Guarding {
    /// `PT`: no guard.
    Always,
    /// `@!PT`: the instruction never runs.
    Never,
    /// A pure instruction: each register it writes selects the new value where the guard holds;
    /// `uniform` reads the guard as a uniform predicate.
    Select {
        /// Whether the guard is a uniform predicate (`OpInfo::uniform`).
        uniform: bool,
    },
    /// Any other instruction: a conditional block of its own.
    Block {
        /// Whether the guard is a uniform predicate (`OpInfo::uniform`).
        uniform: bool,
    },
}

/// How `instr`'s guard is translated: by its guard, then by its operation's class in flow
/// analysis's table ([`classify`] of [`opcode`]). An operation [`opcode`] does not name gets a
/// conditional block, which is right whatever the operation does.
fn guarding(instr: Instr) -> Guarding {
    if never_runs(instr.guard) {
        return Guarding::Never;
    }
    if instr.guard.is_true() {
        return Guarding::Always;
    }
    match opcode(instr.op).map(classify) {
        Some(OpInfo {
            pure: true,
            uniform,
            ..
        }) => Guarding::Select { uniform },
        Some(OpInfo { uniform, .. }) => Guarding::Block { uniform },
        None => Guarding::Block { uniform: false },
    }
}

/// The 12-bit opcode the decoder reads `op` from, for the operations [`translate`] reads the
/// guard of: `MOV` and `EXIT` here, the attribute group's from its module; `None` for any other
/// (the [`float`] group, slice 1's `FFMA` included, takes its own guard and never gets here).
/// `MOV` is given in its register form, since [`classify`] gives every operand form of it the
/// same class (its rows for it match the base operation; pinned by
/// `tests::guards_classify_by_the_flow_table`).
fn opcode(op: Op) -> Option<u16> {
    match op {
        Op::Mov(_) => Some(AluForm::Reg.opcode(MOV_OPCODE)),
        Op::Exit(_) => Some(EXIT_OPCODE),
        other => attribute::opcode(other),
    }
}

/// The value of guard `guard` read as a uniform predicate: `GetUniformPred`, then `LogicalNot`
/// when negated (PR #4588's `UPred`); `UPT` is the immediate.
fn read_uniform_guard(b: &mut Builder<'_>, guard: PredSrc) -> Result<Operand, TranslateError> {
    let value = match UniformPred::new(guard.reg.index()) {
        Some(pred) if !pred.is_true() => b.op(
            Opcode::GetUniformPred,
            &[Operand::Imm(Imm::UniformPred(pred))],
        )?,
        _ => return Ok(Operand::Imm(Imm::U1(!guard.negated))),
    };
    if !guard.negated {
        return Ok(value.into());
    }
    Ok(b.op(Opcode::LogicalNot, &[value.into()])?.into())
}

/// A group translator: translates `op` when it is in the group, else `None`.
type GroupTranslator = fn(&mut Translator<'_>, Op) -> Option<Result<(), Failure>>;

/// The instruction groups' translators, one module each (spec, "Instruction translation"),
/// tried in this order; an operation none of them takes is rejected by its mnemonic.
const GROUPS: &[GroupTranslator] = &[attribute::translate];

struct Translator<'f> {
    b: Builder<'f>,
    stage: Stage,
    /// The pixel program's header, when the caller gave it ([`translate_with_header`]).
    header: Option<PixelIo>,
    /// The guard of the pure instruction being translated, which every register write selects
    /// on (see the module docs, "Guards"); `None` outside one.
    select: Option<Operand>,
}

impl Translator<'_> {
    /// Translates `op` with the group that takes it.
    fn group(&mut self, op: Op) -> Result<(), Failure> {
        for translate in GROUPS {
            if let Some(result) = translate(self, op) {
                return result;
            }
        }
        Err(Unsupported::Instruction(op.mnemonic()).into())
    }

    /// Opens the conditional block of an instruction under `guard` (a uniform predicate when
    /// `uniform`): branches on it to a new block and continues there. Returns the block after
    /// the instruction, for [`Translator::end_guard`].
    fn begin_guard(&mut self, guard: PredSrc, uniform: bool) -> Result<BlockId, TranslateError> {
        let cond = self.read_guard(guard, uniform)?;
        let then = self.b.add_block()?;
        let after = self.b.add_block()?;
        self.b.terminate(Terminator::Branch {
            cond,
            then,
            otherwise: after,
        })?;
        self.b.switch_to(then)?;
        Ok(after)
    }

    /// The value of `guard`: a uniform predicate when `uniform`, else a predicate.
    fn read_guard(&mut self, guard: PredSrc, uniform: bool) -> Result<Operand, TranslateError> {
        if uniform {
            read_uniform_guard(&mut self.b, guard)
        } else {
            Ok(crate::flow::read_pred(&mut self.b, guard)?)
        }
    }

    /// `value`, or under a guard kept as a select ([`Translator::select`]), `SelectU32` of the
    /// guard, `value` and `old`, the destination's value before the instruction (`visitor.cpp`:
    /// `Select(*guard_select, value, GetReg(dest))`). `old` is read only under a guard.
    fn guarded(
        &mut self,
        value: Operand,
        old: impl FnOnce(&mut Self) -> Result<Operand, Failure>,
    ) -> Result<Operand, Failure> {
        let Some(cond) = self.select else {
            return Ok(value);
        };
        let old = old(self)?;
        Ok(self.op(Opcode::SelectU32, &[cond, value, old])?.into())
    }

    /// Closes a conditional block: jumps to `after` and continues there.
    fn end_guard(&mut self, after: BlockId) -> Result<(), TranslateError> {
        self.b.terminate(Terminator::Jump(after))?;
        self.b.switch_to(after)?;
        Ok(())
    }

    fn op(&mut self, opcode: Opcode, args: &[Operand]) -> Result<Value, Failure> {
        Ok(self.b.op(opcode, args)?)
    }

    /// Reads a register's raw bits; `RZ` reads zero.
    fn get(&mut self, reg: Gpr) -> Result<Operand, Failure> {
        if reg.is_zero() {
            Ok(imm32(0))
        } else {
            Ok(self
                .op(Opcode::GetRegister, &[Operand::Imm(Imm::Reg(reg))])?
                .into())
        }
    }

    /// Writes a register's raw bits, under a kept guard as a select ([`Translator::guarded`]);
    /// writes to `RZ` are dropped.
    fn set(&mut self, reg: Gpr, value: Operand) -> Result<(), Failure> {
        if !reg.is_zero() {
            let value = self.guarded(value, |t| t.get(reg))?;
            self.op(Opcode::SetRegister, &[Operand::Imm(Imm::Reg(reg)), value])?;
        }
        Ok(())
    }

    /// Writes a uniform register's raw bits, under a kept guard as a select
    /// ([`Translator::guarded`]); writes to `URZ` are dropped.
    fn set_uniform(&mut self, reg: UniformReg, value: Operand) -> Result<(), Failure> {
        if !reg.is_zero() {
            let value = self.guarded(value, |t| {
                Ok(t.op(
                    Opcode::GetUniformRegister,
                    &[Operand::Imm(Imm::UniformReg(reg))],
                )?
                .into())
            })?;
            self.op(
                Opcode::SetUniformRegister,
                &[Operand::Imm(Imm::UniformReg(reg)), value],
            )?;
        }
        Ok(())
    }

    fn get_f32(&mut self, reg: Gpr) -> Result<Value, Failure> {
        let bits = self.get(reg)?;
        self.op(Opcode::BitCastF32U32, &[bits])
    }

    fn set_f32(&mut self, reg: Gpr, value: Value) -> Result<(), Failure> {
        let bits = self.op(Opcode::BitCastU32F32, &[value.into()])?;
        self.set(reg, bits.into())
    }

    fn mov(&mut self, op: OpMov) -> Result<(), Failure> {
        require(op.quad_lanes == OpMov::ALL_LANES, Unsupported::MovLaneMask)?;
        let value = match op.src {
            MovSrc::Reg(reg) => self.get(reg)?,
            MovSrc::Imm(bits) => imm32(bits),
            MovSrc::UniformReg(_) | MovSrc::CBuf(_) => return Err(Unsupported::Source.into()),
        };
        self.set(op.dst, value)
    }

    /// `EXIT`: for a pixel program, the registers hold the colour outputs, four per target in
    /// target order over the targets with any component written, one register per component
    /// whether that component is written or not (`ops_attr.cpp`, `EXIT`). At most
    /// `COLOR_TARGETS * 4` = 32 registers are read, so the PR's `src_reg < 255` check cannot
    /// fail here and is not repeated. Sample mask and depth outputs come with the SPH parser
    /// (slice 2).
    fn exit(&mut self, op: OpExit) -> Result<(), Failure> {
        require(!op.keep_ref_count && !op.no_at_exit, Unsupported::ExitMode)?;
        require(op.pred.is_true(), Unsupported::ConditionalExit)?;
        if self.stage.stage != GuestStage::Pixel {
            return Ok(());
        }
        let mut reg: u8 = 0;
        let targets = self.stage.color_outputs.targets;
        for (target, mask) in (0_u8..).zip(targets) {
            if mask & COMPONENT_MASK == 0 {
                continue;
            }
            for component in 0_u8..4 {
                if mask & (1 << component) != 0 {
                    let value = self.get_f32(Gpr::new(reg))?;
                    self.op(
                        Opcode::SetFragColor,
                        &[
                            imm32(u32::from(target)),
                            imm32(u32::from(component)),
                            value.into(),
                        ],
                    )?;
                }
                reg = reg.saturating_add(1);
            }
        }
        Ok(())
    }
}
