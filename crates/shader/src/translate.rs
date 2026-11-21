//! Instruction translation: decoded instructions to IR, one flow block at a time.
//!
//! [`translate_flow`] is the entry point (slice 12i, the translation driver; [`driver`]'s module
//! docs): it rebuilds a program's control flow (slice 10, [`crate::flow`]), builds the IR's
//! blocks from it and fills each block from its instructions, dispatching each decoded
//! instruction to its group's translator, one module per group (spec, "Instruction
//! translation"):
//!
//! - [`integer`]: the integer ALU group (slice 12a), `MOV` included;
//! - [`float`]: the 32-bit float and conversion group (slice 12b);
//! - [`texture`]: the texture and surface group (slice 12f);
//! - [`memory`]: the memory group (slice 12e), which translates one instruction on its own but
//!   is not joined to the driver yet: the driver refuses its operations by name ([`driver`]'s
//!   module docs, "Dispatch");
//! - `attribute`: the attribute and system-value group (slice 12g: `ALD`, `AST`, `IPA`, `AL2P`,
//!   `S2R`, `CS2R`, `PIXLD`, and the deferred `OUT`, `ISBERD` and `ISBEWR`), joined through
//!   `GROUPS`;
//! - [`control`]: the control and warp group (slice 12h);
//! - `EXIT`, here: for a pixel program, the colour outputs, the sample mask and the depth
//!   (`Translator::exit`).
//!
//! Ported by behaviour from PR #4588's native frontend
//! (`eden-pr4588/src/shader_recompiler/frontend/ampere/` @ e5fb3b97, GPL-3.0-or-later):
//! `ops_attr.cpp` (`EXIT`) here, `visitor.cpp` (`Visitor::Run`) in [`driver`], and the group
//! modules' own sources. Each of the PR's `Require` checks on these forms is kept with its
//! constant and turns into a rejection naming the same thing, except the colour-register bound
//! of `EXIT`, which cannot fail here (see `Translator::exit`).
//!
//! **Guards.** A guarded instruction becomes a conditional block, pure ones included (spec,
//! "Instruction translation", "Guards"; decided by the owner, 2026-11-19): flow analysis gives a
//! guarded instruction that is not pure a block of its own and tests its guard there
//! ([`crate::flow::FlowBlock::guard_tested`]); the driver builds the conditional block of a
//! guarded pure instruction with [`split_for_guard`], the shared guard helper, and translates
//! the guarded instructions that write no IR (`NOP`, `DEPBAR`, `BREAK`, `WARPSYNC` with an
//! immediate) as they are, with no block ([`driver`]'s "Guards", which lists this as an open
//! owner question: the spec's "Every guarded instruction becomes a conditional block" read
//! literally would give each an empty block).
//!
//! There is one guard rule, the driver's, decided by [`guard_step`]: `!PT` becomes nothing,
//! unchecked; `PT` or a settled guard is ignored; any other guard makes a conditional block.
//! The group translators' own entry points that take a guard follow the same function, so
//! called on their own they give the IR the driver and flow analysis give together: the
//! integer and memory groups ([`integer::GuardMode`]) and the float and texture groups
//! (`guard_tested`, through [`conditional_block`]) split with [`split_for_guard`], reading the
//! guard as a predicate (of the decodable forms only `S2UR` and `VOTEU` are uniform datapath
//! instructions, `classify.h`'s `uniform`, whose guard the driver reads as a uniform
//! predicate, and neither is theirs: `driver::tests::the_guard_of_every_pure_form_is_read_by_its_kind`;
//! a uniform form the decoder learns later needs its group's read changed too); the control
//! group ([`control::translate_control`]), which does not build blocks, refuses an untested
//! guard on an instruction that writes IR ([`control::ControlUnsupported::UntestedGuard`]),
//! `VOTEU` included: its select is retired, as the owner's ruling of 2026-11-19 (row 2) leaves
//! selects to a later optimisation pass. The attribute group takes no guard: it is reached
//! only through the driver, which has handled the guard. Slice 1's straight-line `translate()`,
//! which kept a guarded pure instruction as a select, is retired (the owner's ruling of
//! 2026-11-19, row 1), and so is its `MOV` path, which took only register and immediate
//! sources: `MOV` is the integer group's, which reads every source form.
//!
//! **Sources.** The translators read their source operands through one shared reader (the
//! `source` module: registers, uniform registers, constant-buffer words, predicates and uniform
//! predicates), rather than each keeping a copy.

mod attribute;
pub mod control;
pub mod driver;
pub mod float;
pub mod integer;
pub mod memory;
pub(crate) mod source;
#[cfg(test)]
mod tests;
pub mod texture;

use std::fmt::{self, Display, Formatter};

use thiserror::Error;

use crate::decode::{DecodeError, Gpr, Instr, OpExit, PredSrc, SpecialReg, UniformReg};
use crate::flow::{FlowError, is_never};
use crate::ir::{BlockId, BuildError, Builder, Imm, Opcode, Operand, Terminator, Value};
use crate::sph::PixelIo;

pub use attribute::{AttributeRejection, Deferral};
pub use driver::{MAX_TRANSLATED_INSTRUCTIONS, translate_flow};

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
    /// An instruction no translator handles: in [`translate_flow`], one no group of the
    /// driver's dispatch takes ([`driver`]'s module docs, "Dispatch"); in the attribute group,
    /// `LDTRAM`; in [`control::translate_control`], `BMOV`, `NANOSLEEP`, `JMX` and `BPT`.
    #[error("{0}")]
    Instruction(&'static str),
    /// An instruction of the memory group, whose translator (slice 12e) is not joined to the
    /// driver yet ([`driver`]'s module docs, "Dispatch").
    #[error("{0}, of the memory group, which is not translated yet,")]
    MemoryGroup(&'static str),
    /// A control or warp form [`control::translate_control`] does not handle.
    #[error("{0}")]
    Control(#[from] control::ControlUnsupported),
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
    /// A special register other than the ones handled.
    #[error("special register {0}")]
    SpecialRegister(SpecialReg),
    /// A form of the float and conversion group that is not translated ([`float`]).
    #[error("{0}")]
    Float(float::FloatUnsupported),
    /// A form of the texture and surface group that is not translated ([`texture`]).
    #[error("{0}")]
    Texture(texture::TextureUnsupported),
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
    /// An instruction of the integer ALU group is not translated ([`integer::IntegerError`]:
    /// a form or field value it does not handle, or one of its refusals a decoded instruction
    /// in a flow block cannot reach).
    #[error("instruction {index} ({instr}): {error}")]
    Integer {
        /// The instruction's index in the program.
        index: usize,
        /// The instruction.
        instr: Instr,
        /// Why.
        error: integer::IntegerError,
    },
    /// An instruction a flow block holds does not decode (flow analysis decodes only control
    /// instructions; the driver decodes the rest).
    #[error("instruction {index}: {error}")]
    Decode {
        /// The instruction's index in the program.
        index: usize,
        /// Why it does not decode.
        error: DecodeError,
    },
    /// The program's control flow cannot be rebuilt (slice 10).
    #[error(transparent)]
    Flow(#[from] FlowError),
    /// An instruction a flow block holds is not among the instructions the driver decoded: its
    /// index is past the program or names one that was not decoded (a bug in the translation
    /// driver, which decodes every instruction a flow block holds first).
    #[error(
        "instruction {index}, which a flow block holds, was not decoded (a bug in the translation \
         driver)"
    )]
    NotDecoded {
        /// The instruction's index in the program, as the flow block holds it.
        index: u32,
    },
    /// The program, its calls inlined, holds more instructions than the driver translates
    /// ([`MAX_TRANSLATED_INSTRUCTIONS`]).
    #[error(
        "the program, its calls inlined, has more than {max} instructions to translate",
        max = MAX_TRANSLATED_INSTRUCTIONS
    )]
    TooManyInstructions,
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

/// The shared guard helper: makes the conditional block of an instruction under a guard whose
/// value is `cond` (a `u1`). The block `builder` is in ends in a branch on `cond` to a new
/// block (the instruction's) or to a second new block after it; the instruction's block jumps to
/// the block after; and the block after takes over the first block's terminator, so the split
/// works in a block flow analysis has already terminated as in one still open (whose
/// terminator, [`Terminator::None`], moves the same way). `builder` is left in the instruction's
/// block; the block after is returned, where translation continues once the instruction's IR is
/// in. The two new blocks are made in that order, the instruction's first.
///
/// A phi in a successor of the split block that names it would then name a block that no longer
/// jumps there; the driver splits only before SSA, where there are no phis, and the integer
/// group checks for them first (its module docs, "Guards").
pub(crate) fn split_for_guard(
    builder: &mut Builder<'_>,
    cond: Operand,
) -> Result<BlockId, BuildError> {
    let here = builder.block();
    let old = builder
        .function()
        .block(here)
        .map(|block| block.terminator)
        .ok_or(BuildError::NoBlock(here))?;
    let then = builder.add_block()?;
    let join = builder.add_block()?;
    builder.terminate(Terminator::Branch {
        cond,
        then,
        otherwise: join,
    })?;
    builder.switch_to(join)?;
    builder.terminate(old)?;
    builder.switch_to(then)?;
    builder.terminate(Terminator::Jump(join))?;
    Ok(join)
}

/// What a translator does with an instruction's guard: the translation driver's rule
/// ([`driver`]'s module docs, "Guards"), made by [`guard_step`], which the driver and every
/// group translator that takes a guard itself ([`integer`], [`float`], [`texture`],
/// [`memory`], [`control`]) follow, so that there is one rule.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum GuardStep {
    /// The guard is `!PT`: the instruction never runs and becomes nothing, its own checks not
    /// run, as PR #4588's `Visitor::Run` returns at `guard.Never()` (`visitor.cpp`).
    Skip,
    /// The guard is `PT`, or is settled (see [`guard_step`]): the instruction is translated as
    /// if unguarded.
    Unguarded,
    /// Any other guard: the instruction becomes a conditional block ([`split_for_guard`];
    /// spec, "Instruction translation", "Guards").
    Conditional,
}

/// The translation driver's guard rule ([`GuardStep`]), in this order: `!PT` is
/// [`GuardStep::Skip`], whatever `settled` says; `PT`, or a guard that is `settled`, is
/// [`GuardStep::Unguarded`]; any other guard is [`GuardStep::Conditional`]. A guard is settled
/// when flow analysis has tested it (the block's [`crate::flow::FlowBlock::guard_tested`]) and,
/// in the driver, when the instruction is not pure (`classify.h`'s `pure`): flow analysis tests
/// the guard of every such instruction it does not leave in its block, and the ones it leaves
/// there write no IR (the driver's "Guards"). Pinned for every input by
/// `tests::guard_step_for_every_guard`.
#[must_use]
pub(crate) const fn guard_step(guard: PredSrc, settled: bool) -> GuardStep {
    if is_never(guard) {
        GuardStep::Skip
    } else if settled || guard.is_true() {
        GuardStep::Unguarded
    } else {
        GuardStep::Conditional
    }
}

/// Translates one instruction under `guard` at the end of `builder`'s block by [`guard_step`]
/// (with `guard_tested` as the guard's being settled), for the groups whose translators take a
/// guard themselves and check an instruction whole before writing any of it ([`float`],
/// [`texture`]). For [`GuardStep::Skip`] nothing runs, `check` included. Otherwise `check`
/// runs first, and its refusal appends nothing; then `body` writes the instruction's IR from
/// what `check` returned: in `builder`'s block for [`GuardStep::Unguarded`], or, for
/// [`GuardStep::Conditional`], in the instruction's block of the split [`split_for_guard`]
/// makes after the guard's read ([`source`]'s `pred`), `builder` then left in the block after,
/// where translation continues.
pub(crate) fn conditional_block<T, E: From<BuildError>>(
    builder: &mut Builder<'_>,
    guard: PredSrc,
    guard_tested: bool,
    check: impl FnOnce() -> Result<T, E>,
    body: impl FnOnce(&mut Builder<'_>, T) -> Result<(), E>,
) -> Result<(), E> {
    match guard_step(guard, guard_tested) {
        GuardStep::Skip => Ok(()),
        GuardStep::Unguarded => body(builder, check()?),
        GuardStep::Conditional => {
            let checked = check()?;
            let cond = source::pred(builder, guard)?;
            let join = split_for_guard(builder, cond)?;
            body(builder, checked)?;
            builder.switch_to(join)?;
            Ok(())
        }
    }
}

/// A group translator: translates `op` when it is in the group, else `None`.
type GroupTranslator = fn(&mut Translator<'_>, crate::decode::Op) -> Option<Result<(), Failure>>;

/// The instruction groups' translators that run on a [`Translator`] (spec, "Instruction
/// translation"), tried in this order by [`Translator::group`].
const GROUPS: &[GroupTranslator] = &[attribute::translate];

/// The translator state of the groups that run on it: the attribute group, the control and
/// warp group ([`control`]) and `EXIT`.
struct Translator<'f> {
    b: Builder<'f>,
    stage: Stage,
    /// The pixel program's header, when the caller gave it.
    header: Option<PixelIo>,
}

impl source::Emit for Translator<'_> {
    fn emit_op(&mut self, opcode: Opcode, args: &[Operand]) -> Result<Operand, BuildError> {
        Ok(self.b.op(opcode, args)?.into())
    }
}

impl Translator<'_> {
    /// Translates `op` with the group of `GROUPS` that takes it; `None` when none does.
    fn group(&mut self, op: crate::decode::Op) -> Option<Result<(), Failure>> {
        GROUPS.iter().find_map(|translate| translate(self, op))
    }

    fn op(&mut self, opcode: Opcode, args: &[Operand]) -> Result<Value, Failure> {
        Ok(self.b.op(opcode, args)?)
    }

    /// Reads a register's raw bits; `RZ` reads zero (the shared reader).
    fn get(&mut self, reg: Gpr) -> Result<Operand, Failure> {
        Ok(source::reg(self, reg)?)
    }

    /// Writes a register's raw bits; writes to `RZ` are dropped.
    fn set(&mut self, reg: Gpr, value: Operand) -> Result<(), Failure> {
        if !reg.is_zero() {
            self.op(Opcode::SetRegister, &[Operand::Imm(Imm::Reg(reg)), value])?;
        }
        Ok(())
    }

    /// Writes a uniform register's raw bits; writes to `URZ` are dropped.
    fn set_uniform(&mut self, reg: UniformReg, value: Operand) -> Result<(), Failure> {
        if !reg.is_zero() {
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

    /// `EXIT`: for a pixel program, the registers hold the colour outputs, four per target in
    /// target order over the targets with any component written, one register per component
    /// whether that component is written or not (`ops_attr.cpp`, `EXIT`). At most
    /// `COLOR_TARGETS * 4` = 32 registers are read, so the PR's `src_reg < 255` check cannot
    /// fail here and is not repeated. With the pixel program's header, the sample mask
    /// (`OMAP_SAMPLE_MASK`) is the register after the colours and the depth (`OMAP_DEPTH`) the
    /// one after that, whether the sample mask is written or not, as both references read them
    /// (`ops_attr.cpp`, `EXIT`: `X(src_reg)` and `F(src_reg + 1)`; Eden's `exit_program.cpp`,
    /// `ExitFragment`, the same); at most register 33, so the PR's two register checks cannot
    /// fail either (slice 15).
    ///
    /// The `EXIT`'s own predicate and its guard are flow analysis's: a conditional `EXIT` gets a
    /// block of its own whose test reads them ([`crate::flow::FlowBlock::guard_tested`]), so the
    /// outputs are written only where the program exits, as the PR's control-flow builder makes
    /// it (`ops_attr.cpp`: "extra exit condition"). A mode bit is refused first (flow analysis
    /// refuses it before, [`FlowError::SpecialExit`]; the check is kept for the PR's order).
    fn exit(&mut self, op: OpExit) -> Result<(), Failure> {
        require(!op.keep_ref_count && !op.no_at_exit, Unsupported::ExitMode)?;
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
        let Some(header) = self.header else {
            return Ok(());
        };
        if header.writes_sample_mask {
            let mask = self.get(Gpr::new(reg))?;
            self.op(Opcode::SetSampleMask, &[mask])?;
        }
        if header.writes_depth {
            let depth = self.get_f32(Gpr::new(reg.saturating_add(1)))?;
            self.op(Opcode::SetFragDepth, &[depth.into()])?;
        }
        Ok(())
    }
}
