//! Instruction translation: decoded instructions to IR, for straight-line programs.
//!
//! Ported by behaviour from PR #4588's native frontend
//! (`eden-pr4588/src/shader_recompiler/frontend/ampere/` @ e5fb3b97, GPL-3.0-or-later):
//! `ops_move.cpp` (`MOV`), `ops_alu.cpp` (`FFMA`), `ops_misc.cpp` (`ALD`, `AST`, `S2R`) and
//! `ops_attr.cpp` (`EXIT`). Each of the PR's `Require` checks on these forms is kept with its
//! constant and turns into an [`Unsupported`] rejection naming the same thing, except the
//! colour-register bound of `EXIT`, which cannot fail here (see `Translator::exit`).
//!
//! Slice 1 handles linear flow only (spec, port slice 1). Where starquake is stricter than the
//! PR, because the PR resolves the case in a part not built yet:
//!
//! - an instruction under a guard predicate other than `PT` and `!PT` is rejected (flow analysis
//!   is slice 10). The PR turns a guard on a pure instruction (`MOV`, `FFMA`, `ALD` and `S2R`
//!   here; `classify.h`, `Classify`) into a select of the result (`visitor.cpp`,
//!   `Visitor::Run`: `guard_select`), and makes a guarded side effect (`AST` here) its own
//!   conditional block (`flow.cpp`: "Guarded side effect: becomes its own conditional block");
//! - an instruction under `@!PT` is rejected, where the PR skips it: `Visitor::Run` returns when
//!   `guard.Never()`, which `flow.h` defines as `index == 7 && negated`;
//! - an `EXIT` with its own predicate other than `PT` is rejected (the PR's control-flow builder
//!   resolves it, `ops_attr.cpp`: "extra exit condition");
//! - the program must end with its first `EXIT`: no `EXIT` and instructions after it are
//!   rejected (the extent of a program is the capture's job, slice 3);
//! - `ALD` and `AST` take only `RZ` as the vertex register (the PR passes the register to the
//!   IR; the vertex and pixel stages handled here have one vertex);
//! - `S2R` reads only `SR_LANEID` (the PR also reads the thread and block ids, the W scale
//!   factors and the lane masks).
//!
//! Pinned by `tests::guarded_instructions_are_rejected`, `never_guarded_instructions_are_rejected`,
//! `exit_modes_and_predicates_are_rejected`, `the_program_ends_at_its_only_exit`,
//! `attribute_access_checks` and `s2r_lane_id`.

#[cfg(test)]
mod tests;

use std::fmt::{self, Display, Formatter};

use thiserror::Error;

use crate::decode::{
    FloatSrc, Gpr, Instr, MovSrc, Op, OpALd, OpASt, OpExit, OpFFma, OpMov, OpS2R, RoundMode,
    SpecialReg,
};
use crate::ir::{
    Attribute, BuildError, Builder, Flags, FmzMode, FpControl, FpRounding, Function, Imm, Opcode,
    Operand, Terminator, Value,
};

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

/// Which components of each colour target a pixel program writes: a stand-in for the output
/// map of the program's header (the SPH's `omap`), which slice 2 parses. Bit `c` of
/// `targets[t]` means component `c` of target `t`; bits above [`COMPONENT_MASK`] are ignored.
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
    /// A guard other than `PT` needs flow analysis.
    #[error("a guarded instruction (flow analysis is not built yet)")]
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
    /// An attribute access with a vertex register.
    #[error("an attribute access with a vertex register")]
    AttributeVertex,
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

/// Translates a straight-line program that ends with its first `EXIT`, into one block that ends
/// in a return.
pub fn translate(stage: Stage, program: &[Instr]) -> Result<Function, TranslateError> {
    if !matches!(stage.stage, GuestStage::Vertex | GuestStage::Pixel) {
        return Err(TranslateError::Stage(Unsupported::Stage(stage.stage)));
    }
    let mut function = Function::new();
    let entry = function.entry();
    let mut t = Translator {
        b: Builder::new(&mut function, entry),
        stage,
    };
    for (index, instr) in program.iter().enumerate() {
        let fail = |what| TranslateError::Unsupported {
            index,
            instr: *instr,
            what,
        };
        if !instr.guard.is_true() {
            return Err(fail(Unsupported::Guard));
        }
        let done = match instr.op {
            Op::Mov(op) => t.mov(op).map(|()| false),
            Op::FFma(op) => t.ffma(op).map(|()| false),
            Op::ALd(op) => t.ald(op).map(|()| false),
            Op::ASt(op) => t.ast(op).map(|()| false),
            Op::S2R(op) => t.s2r(op).map(|()| false),
            Op::Exit(op) => t.exit(op).map(|()| true),
        };
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

/// The vertex index of an attribute access: always zero here (`ALD` and `AST` take only `RZ` as
/// the vertex register), as Eden's emitter passes `Imm32(0)` for a stage with one vertex
/// (`ir_emitter.cpp`, `GetAttribute(attribute)`).
const VERTEX_ZERO: Operand = imm32(0);

struct Translator<'f> {
    b: Builder<'f>,
    stage: Stage,
}

impl Translator<'_> {
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

    fn float_src(&mut self, src: FloatSrc) -> Result<Value, Failure> {
        match src {
            FloatSrc::Imm(bits) => self.op(Opcode::BitCastF32U32, &[imm32(bits)]),
            FloatSrc::Reg { reg, abs, neg } => {
                let mut value = self.get_f32(reg)?;
                if abs {
                    value = self.op(Opcode::FPAbs32, &[value.into()])?;
                }
                if neg {
                    value = self.op(Opcode::FPNeg32, &[value.into()])?;
                }
                Ok(value)
            }
        }
    }

    fn mov(&mut self, op: OpMov) -> Result<(), Failure> {
        require(op.quad_lanes == OpMov::ALL_LANES, Unsupported::MovLaneMask)?;
        let value = match op.src {
            MovSrc::Reg(reg) => self.get(reg)?,
            MovSrc::Imm(bits) => imm32(bits),
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

    /// The address check `ALD` and `AST` share; returns the first attribute.
    fn attribute(addr: u16, comps: u8) -> Result<u16, Failure> {
        let end = u32::from(comps)
            .checked_mul(4)
            .and_then(|size| u32::from(addr).checked_add(size));
        require(
            addr.is_multiple_of(4) && end.is_some_and(|end| end <= ATTRIBUTE_SPACE_END),
            Unsupported::AttributeAddress { addr, comps },
        )?;
        Ok(addr / 4)
    }

    /// `ALD`, with the PR's checks in its order: offset register, patch or physical, output,
    /// address, registers; then starquake's vertex register check.
    fn ald(&mut self, op: OpALd) -> Result<(), Failure> {
        require(op.offset.is_zero(), Unsupported::IndexedAttribute)?;
        require(!op.patch && !op.phys, Unsupported::PatchOrPhysical)?;
        require(!op.output, Unsupported::OutputRead)?;
        let first = Self::attribute(op.addr, op.comps)?;
        require(
            register_run(op.dst, u32::from(op.comps)).is_some(),
            Unsupported::AttributeRegisters {
                first: op.dst,
                comps: op.comps,
            },
        )?;
        require(op.vtx.is_zero(), Unsupported::AttributeVertex)?;
        for element in 0..op.comps {
            let attribute = Attribute(first.saturating_add(u16::from(element)));
            let value = self.op(
                Opcode::GetAttribute,
                &[Operand::Imm(Imm::Attribute(attribute)), VERTEX_ZERO],
            )?;
            let reg = Gpr::new(op.dst.index().saturating_add(element));
            self.set_f32(reg, value)?;
        }
        Ok(())
    }

    /// `AST`, with the PR's checks in its order: offset register, patch or physical, address,
    /// registers; then starquake's vertex register check.
    fn ast(&mut self, op: OpASt) -> Result<(), Failure> {
        require(op.offset.is_zero(), Unsupported::IndexedAttribute)?;
        require(!op.patch && !op.phys, Unsupported::PatchOrPhysical)?;
        let first = Self::attribute(op.addr, op.comps)?;
        // `RZ` as the data source stores zero for every component (`ops_misc.cpp`, `AST`).
        require(
            op.data.is_zero() || register_run(op.data, u32::from(op.comps)).is_some(),
            Unsupported::AttributeRegisters {
                first: op.data,
                comps: op.comps,
            },
        )?;
        require(op.vtx.is_zero(), Unsupported::AttributeVertex)?;
        for element in 0..op.comps {
            let reg = if op.data.is_zero() {
                Gpr::RZ
            } else {
                Gpr::new(op.data.index().saturating_add(element))
            };
            let value = self.get_f32(reg)?;
            let attribute = Attribute(first.saturating_add(u16::from(element)));
            self.op(
                Opcode::SetAttribute,
                &[
                    Operand::Imm(Imm::Attribute(attribute)),
                    value.into(),
                    VERTEX_ZERO,
                ],
            )?;
        }
        Ok(())
    }

    fn s2r(&mut self, op: OpS2R) -> Result<(), Failure> {
        require(
            op.sr == SpecialReg::LANE_ID,
            Unsupported::SpecialRegister(op.sr),
        )?;
        // A write to `RZ` is dropped, so the lane id is not read and needs no capability.
        if op.dst.is_zero() {
            return Ok(());
        }
        let value = self.op(Opcode::LaneId, &[])?;
        self.set(op.dst, value.into())
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
