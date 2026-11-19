//! Instruction translation: decoded instructions to IR, for straight-line programs.
//!
//! Ported by behaviour from PR #4588's native frontend
//! (`eden-pr4588/src/shader_recompiler/frontend/ampere/` @ e5fb3b97, GPL-3.0-or-later):
//! `ops_move.cpp` (`MOV`), `ops_alu.cpp` (`FFMA`) and `ops_attr.cpp` (`EXIT`) here, and one
//! module per instruction group beside this one (spec, "Instruction translation"), joined by
//! [`GROUPS`]: `attribute` (slice 12g: `ALD`, `AST`, `IPA`, `AL2P`, `S2R`, `CS2R`, `PIXLD`, and
//! the deferred `OUT`, `ISBERD` and `ISBEWR`). Each of the PR's `Require` checks on these forms
//! is kept with its constant and turns into an [`Unsupported`] rejection naming the same thing,
//! except the colour-register bound of `EXIT`, which cannot fail here (see `Translator::exit`).
//!
//! **Guards.** An instruction under a guard predicate other than `PT` becomes a conditional
//! block (spec, "Instruction translation"): the block it was in reads the predicate (through
//! `flow`'s `read_pred`, as the flow graph's tests do) and branches to a block of its own,
//! which holds the instruction and jumps to a new block where translation continues. An
//! instruction under `@!PT` never runs and translates to nothing, as in the PR: `Visitor::Run`
//! returns when `guard.Never()`, which `flow.h` defines as `index == 7 && negated`. The PR
//! turns a guard on a pure instruction (`classify.h`, `Classify`) into a select of the result
//! instead (`visitor.cpp`, `Visitor::Run`: `guard_select`) and makes only a guarded side effect
//! its own conditional block (`flow.cpp`: "Guarded side effect: becomes its own conditional
//! block"); the two give the same registers, since the block that does not run leaves them as
//! the select's other side does. The guard of a uniform datapath instruction names a uniform
//! predicate (the PR's `classify.h`, `OpInfo::uniform`, which `flow`'s `classify` mirrors by
//! opcode); of the operations translated so far that is `S2UR` alone, whose guard reads
//! `GetUniformPred`. Pinned by `tests::guarded_instructions_become_conditional_blocks`,
//! `tests::a_guarded_s2ur_reads_a_uniform_predicate`, `tests::a_guarded_exit_is_rejected` and
//! `tests::never_guarded_instructions_are_skipped`.
//!
//! Slice 1 handles linear flow only (spec, port slice 1). Where starquake is stricter than the
//! PR, because the PR resolves the case in a part not built yet:
//!
//! - a guarded `EXIT` is rejected, and so is an `EXIT` with its own predicate other than `PT`
//!   (the PR's control-flow builder resolves both, `ops_attr.cpp`: "extra exit condition");
//! - the program must end with its first `EXIT`: no `EXIT` and instructions after it are
//!   rejected (the extent of a program is the capture's job, slice 3);
//! - `MOV` and `FFMA` take only register and immediate sources: a uniform-register or
//!   constant-buffer source, which the decoder reads since slice 4, is rejected (the PR reads
//!   both, `visitor.cpp`, `Visitor::Value`; constant buffers and uniform registers come with
//!   the passes of slices 12 and 13);
//! - every other instruction is rejected by its mnemonic.
//!
//! Pinned by `tests::exit_modes_and_predicates_are_rejected`,
//! `the_program_ends_at_its_only_exit`, `uniform_and_constant_buffer_sources_are_rejected` and
//! `other_instructions_are_rejected_by_mnemonic`. The group modules list their own divergences.

mod attribute;
#[cfg(test)]
mod tests;

use std::fmt::{self, Display, Formatter};

use thiserror::Error;

use crate::decode::{
    AluSrc, Gpr, Instr, MovSrc, Op, OpExit, OpFFma, OpMov, OpS2R, PredSrc, RoundMode, S2RDst,
    SpecialReg, UniformPred,
};
use crate::ir::{
    BlockId, BuildError, Builder, Flags, FmzMode, FpControl, FpRounding, Function, Imm, Opcode,
    Operand, Terminator, Value,
};
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
    /// An instruction no translator handles yet: neither `MOV`, `FFMA` and `EXIT` here nor a
    /// group of [`GROUPS`].
    #[error("{0}")]
    Instruction(&'static str),
    /// A uniform-register or constant-buffer source of `MOV` or `FFMA` (PR #4588 reads both:
    /// `visitor.cpp`, `Visitor::Value`).
    #[error("a uniform-register or constant-buffer source")]
    Source,
    /// An `EXIT` under a guard other than `PT` and `!PT`: a conditional exit is control flow,
    /// which this straight-line translator does not build (every other guarded instruction
    /// becomes a conditional block; see the module docs).
    #[error("a guarded exit")]
    Guard,
    /// A `MOV` lane mask other than `0xf` (`ops_move.cpp`: "native MOV lane mask").
    #[error("a mov with a partial lane mask")]
    MovLaneMask,
    /// `FFMA`'s multiply-by-zero mode (`ops_alu.cpp`: "native FFMA multiply-zero mode").
    #[error("ffma's multiply-by-zero mode")]
    FfmaMultiplyZero,
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
    /// "native attribute address").
    #[error("attribute address {addr:#x} with {comps} words")]
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
/// a return: one block, and two more for each guarded instruction (see the module docs).
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
    };
    for (index, instr) in program.iter().enumerate() {
        let fail = |what| TranslateError::Unsupported {
            index,
            instr: *instr,
            what,
        };
        if never_runs(instr.guard) {
            continue;
        }
        let guarded = !instr.guard.is_true();
        if guarded && matches!(instr.op, Op::Exit(_)) {
            return Err(fail(Unsupported::Guard));
        }
        let after = if guarded {
            Some(t.begin_guard(instr.guard, uniform_guard(instr.op))?)
        } else {
            None
        };
        let done = match instr.op {
            Op::Mov(op) => t.mov(op).map(|()| false),
            Op::FFma(op) => t.ffma(op).map(|()| false),
            Op::Exit(op) => t.exit(op).map(|()| true),
            other => t.group(other).map(|()| false),
        };
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

/// PR #4588's `RoundingOf` (`ops_alu.cpp`): the 2-bit rounding field as an IR rounding mode.
const fn rounding_of(mode: RoundMode) -> FpRounding {
    match mode {
        RoundMode::NearestEven => FpRounding::Rn,
        RoundMode::NegInf => FpRounding::Rm,
        RoundMode::PosInf => FpRounding::Rp,
        RoundMode::Zero => FpRounding::Rz,
    }
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

/// Whether `op` is a uniform datapath instruction, whose guard names a uniform predicate (PR
/// #4588's `classify.h`: "Uniform datapath instruction: its guard predicate is a uniform
/// predicate"). Of the operations translated so far, only `S2UR` is one.
const fn uniform_guard(op: Op) -> bool {
    matches!(
        op,
        Op::S2R(OpS2R {
            dst: S2RDst::Uniform(_),
            ..
        })
    )
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
        let cond = if uniform {
            read_uniform_guard(&mut self.b, guard)?
        } else {
            crate::flow::read_pred::<TranslateError>(&mut self.b, guard)?
        };
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

    /// Writes a register's raw bits; writes to `RZ` are dropped.
    fn set(&mut self, reg: Gpr, value: Operand) -> Result<(), Failure> {
        if !reg.is_zero() {
            self.op(Opcode::SetRegister, &[Operand::Imm(Imm::Reg(reg)), value])?;
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

    fn float_src(&mut self, src: AluSrc) -> Result<Value, Failure> {
        match src {
            AluSrc::Imm(bits) => self.op(Opcode::BitCastF32U32, &[imm32(bits)]),
            AluSrc::Reg { reg, abs, neg } => {
                let mut value = self.get_f32(reg)?;
                if abs {
                    value = self.op(Opcode::FPAbs32, &[value.into()])?;
                }
                if neg {
                    value = self.op(Opcode::FPNeg32, &[value.into()])?;
                }
                Ok(value)
            }
            AluSrc::UniformReg { .. } | AluSrc::CBuf { .. } => Err(Unsupported::Source.into()),
        }
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

    /// `FFMA` with PR #4588's controls (`ops_alu.cpp`, `FFMA`): no contraction, the rounding
    /// field through `RoundingOf`, and `FTZ` or no flushing.
    fn ffma(&mut self, op: OpFFma) -> Result<(), Failure> {
        require(!op.dnz, Unsupported::FfmaMultiplyZero)?;
        let [a, b, c] = op.srcs;
        let a = self.float_src(a)?;
        let b = self.float_src(b)?;
        let c = self.float_src(c)?;
        let control = FpControl {
            no_contraction: true,
            rounding: rounding_of(op.rnd),
            fmz_mode: if op.ftz { FmzMode::Ftz } else { FmzMode::None },
        };
        let mut value = self.b.op_with(
            Opcode::FPFma32,
            &[a.into(), b.into(), c.into()],
            Flags::Fp(control),
        )?;
        if op.saturate {
            value = self.op(Opcode::FPSaturate32, &[value.into()])?;
        }
        self.set_f32(op.dst, value)
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
