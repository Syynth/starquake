//! Lowering the IR to a validated naga module and its interface tables.
//!
//! This module (with its `builder`) is the only code that builds naga IR (spec, "The backend
//! through naga"), so a naga upgrade touches one place. Slice 1 lowers straight-line vertex and
//! pixel functions:
//!
//! - Each guest register the function uses becomes a `u32` local variable, zero at entry;
//!   `GetReg` and `SetReg` load and store it.
//! - A vertex function's input attributes become vertex inputs: generic attribute `n` is a
//!   `vec4<f32>` at location `n`. Its outputs are one struct: the position built-in, then
//!   generic `n` at location `n`, with perspective interpolation at the pixel centre (a guess
//!   until the SPH's interpolation modes are read, slice 15). Components the program does not
//!   write are zero. Generics from [`HOST_LOCATIONS`] up (generic 31) are refused, read or
//!   written, because wgpu on Metal binds no location past 30.
//! - A pixel function's colour target `n` is a `vec4<f32>` output at location `n`; unwritten
//!   components are zero. `SR_LANEID` is the subgroup invocation id built-in, which needs the
//!   subgroup capability; naga 30.0.1 does not allow it in a vertex shader
//!   (`naga/src/valid/interface.rs`: `SubgroupInvocationId` is an input of compute, fragment,
//!   task, mesh and ray stages, not vertex).
//! - Rounding modes and flush-to-zero are not expressed (decision 10, a listed divergence):
//!   `FPFma32` becomes naga's fused `fma` whatever its `FpControl`.
//! - The function must be a control-flow graph of one block that returns
//!   ([`LowerError::ControlFlow`]), with only the operations the skeleton's translators emit
//!   (`GetRegister`, `SetRegister`, `BitCastF32U32`, `BitCastU32F32`, `FPAbs32`, `FPNeg32`,
//!   `FPFma32`, `FPSaturate32`, `GetAttribute` and `SetAttribute` with vertex index zero,
//!   `LaneId`, `SetFragColor` with immediate target and component); any other operation or a
//!   phi is refused until slice 16. A 32-bit immediate operand becomes a literal at its use,
//!   where slice 1's IR had an `Imm32` instruction, so the expressions keep their order
//!   (`tests::skeleton_modules_are_unchanged`).
//!
//! Generic attributes are attribute words 32 to 159, `n = (word - 32) / 4`, as Eden numbers
//! them (`eden/src/shader_recompiler/frontend/ir/attribute.h` @ cf654052: `Generic0X = 32`,
//! `Generic31W = 159`, `GenericAttributeIndex`). Divergence: Eden's `IsGeneric` stops at
//! `Generic31X` (156), so its words 157 to 159 are not generics there; here they are generic
//! 31's Y, Z and W, and so refused with the rest of generic 31 as
//! [`LowerError::GenericLocation`] rather than as an attribute the stage cannot use. Pinned by
//! `tests::generic_31_w_is_a_generic`.
//!
//! Validation uses every naga flag except the uniformity analysis (spec, "The backend through
//! naga"). Experiment E1 found that wgpu 30.0.1 validates with every flag and that naga 30.0.1
//! enforces uniformity only for cooperative-matrix operations, so the modules this crate makes
//! pass both; the tests check both.

pub(crate) mod builder;
#[cfg(test)]
mod e1;
#[cfg(test)]
mod tests;

use std::collections::{BTreeMap, BTreeSet};

use naga::valid::{Capabilities, ValidationFlags, Validator};
use naga::{
    Binding, Block, BuiltIn, Expression, FunctionArgument, FunctionResult, Handle, Interpolation,
    Literal, LocalVariable, MathFunction, Module, Sampling, Scalar, ScalarKind, ShaderStage,
    Statement, StructMember, TypeInner, UnaryOperator, VectorSize,
};
use thiserror::Error;

use crate::host::{
    GuestSlot, HostBuiltIn, HostCode, HostShader, HostStage, InterfaceSlot, Requirements,
    SlotBinding, SlotType,
};
use crate::ir::{
    Attribute, Function, Imm, Inst, Opcode, Operand, Operation, Terminator, Value, VerifyError,
};
use crate::translate::{COLOR_TARGETS, GuestStage};
use builder::{FunctionBuilder, named_ty, push, ty};

/// The entry point's name in every module.
pub const ENTRY_POINT: &str = "main";

/// The attribute word of the position's X (`attribute.h`: `PositionX = 28`).
pub const POSITION_X: u16 = 28;
/// The attribute word of generic 0's X (`attribute.h`: `Generic0X = 32`).
pub const GENERIC_0_X: u16 = 32;
/// How many generic attributes there are (`attribute.h`: `NUM_GENERICS = 32`).
pub const GENERICS: u16 = 32;
/// Words per attribute (X, Y, Z, W).
const COMPONENTS: u16 = 4;
/// How many locations a host stage binds: generic `n` uses location `n`, so generics from this
/// one up are refused. wgpu 30.0.1 on Metal binds vertex attributes below 31 (`wgpu-hal`
/// `metal/adapter.rs`: `max_vertex_attributes: 31`) and vertex outputs at locations up to
/// `max_inter_stage_shader_variables - 1` (`wgpu-core` `validation.rs`), which is 31 on Apple
/// GPUs (`metal/adapter.rs`: "min(124, 124 / 4)"); measured on an Apple M2. Devices with lower
/// limits (other Mac GPUs, Vulkan drivers) are the GPU crate's to check against the device. To
/// support generic 31 later, remap locations: the interface tables already keep the guest slot
/// apart from the host binding.
pub const HOST_LOCATIONS: u8 = 31;
/// The size in bytes of a `vec4<f32>`, the stride of the output structs.
const VEC4_BYTES: u32 = 16;

/// What an attribute word is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AttributeKind {
    Position { component: u32 },
    Generic { index: u8, component: u32 },
}

fn classify(attribute: Attribute) -> Option<AttributeKind> {
    let word = attribute.0;
    let component = u32::from(word % COMPONENTS);
    if (POSITION_X..POSITION_X + COMPONENTS).contains(&word) {
        return Some(AttributeKind::Position { component });
    }
    let generic = word.checked_sub(GENERIC_0_X)? / COMPONENTS;
    (generic < GENERICS).then(|| AttributeKind::Generic {
        index: u8::try_from(generic).unwrap_or(u8::MAX),
        component,
    })
}

/// Why a function was not lowered.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum LowerError {
    /// The function is malformed.
    #[error("the IR is malformed: {0}")]
    Verify(#[from] VerifyError),
    /// The stage is not handled yet.
    #[error("{0} programs are not lowered yet")]
    Stage(GuestStage),
    /// Control flow other than one block that returns, which is not lowered yet (slice 16).
    #[error("control flow other than one block that returns is not lowered yet")]
    ControlFlow,
    /// An operation the lowering does not handle yet.
    #[error("{at}: {opcode} is not lowered yet")]
    Operation {
        /// The instruction.
        at: Value,
        /// Its opcode.
        opcode: Opcode,
    },
    /// A phi, which the lowering does not handle yet.
    #[error("{0}: phis are not lowered yet")]
    Phi(Value),
    /// An argument the lowering needs as an immediate of a particular kind is something else.
    #[error("{at}: argument {index} is not an immediate the lowering handles")]
    Operand {
        /// The instruction.
        at: Value,
        /// The argument's index.
        index: usize,
    },
    /// An attribute the stage cannot read.
    #[error("a {stage} program cannot read attribute {attribute}")]
    Input {
        /// The stage.
        stage: GuestStage,
        /// The attribute.
        attribute: Attribute,
    },
    /// An attribute the stage cannot write.
    #[error("a {stage} program cannot write attribute {attribute}")]
    Output {
        /// The stage.
        stage: GuestStage,
        /// The attribute.
        attribute: Attribute,
    },
    /// A generic attribute with no host location (see [`HOST_LOCATIONS`]).
    #[error("generic {generic} has no host location: locations stop at {last}", last = HOST_LOCATIONS - 1)]
    GenericLocation {
        /// The generic attribute's index.
        generic: u8,
    },
    /// The lane id in a vertex program, which naga cannot express.
    #[error("the lane id cannot be read in a vertex program")]
    LaneIdInVertex,
    /// A colour output outside the targets and components that exist.
    #[error("colour output {target}.{component} does not exist")]
    ColorOutput {
        /// The target.
        target: u32,
        /// The component.
        component: u32,
    },
    /// A colour output in a stage that has none.
    #[error("a {0} program cannot write colour outputs")]
    ColorInStage(GuestStage),
    /// naga rejected the module: a bug in this module, reported rather than handed out.
    #[error("naga rejected the module: {0}")]
    Validation(String),
    /// The lowering's own bookkeeping disagrees with the function (a bug in this module).
    #[error("internal lowering error: {0}")]
    Internal(String),
}

/// The validation flags this crate uses: every flag but the uniformity analysis.
pub fn validation_flags() -> ValidationFlags {
    ValidationFlags::all() - ValidationFlags::CONTROL_FLOW_UNIFORMITY
}

/// The naga capabilities a shader with `requirements` is validated with.
pub fn capabilities(requirements: Requirements) -> Capabilities {
    let mut caps = Capabilities::empty();
    caps.set(Capabilities::SUBGROUP, requirements.subgroup);
    caps
}

/// The location of generic `generic`, if a host stage can bind it.
fn host_location(generic: u8) -> Result<u8, LowerError> {
    if generic < HOST_LOCATIONS {
        Ok(generic)
    } else {
        Err(LowerError::GenericLocation { generic })
    }
}

/// What a function reads and writes, collected before the module is built.
#[derive(Default)]
struct Interface {
    registers: BTreeSet<u8>,
    inputs: BTreeSet<u8>,
    generic_outputs: BTreeSet<u8>,
    targets: BTreeSet<u8>,
    lane_id: bool,
}

impl Interface {
    fn collect(stage: GuestStage, function: &Function) -> Result<Self, LowerError> {
        let vertex = stage == GuestStage::Vertex;
        let mut found = Self::default();
        for (at, inst) in function.listed() {
            let Inst::Op(op) = inst else {
                return Err(LowerError::Phi(at));
            };
            match op.opcode {
                Opcode::GetRegister | Opcode::SetRegister => {
                    let Some(Imm::Reg(reg)) = imm_arg(op, 0) else {
                        return Err(LowerError::Operand { at, index: 0 });
                    };
                    found.registers.insert(reg.index());
                }
                Opcode::GetAttribute => match classify(attribute_arg(at, op, 1)?) {
                    Some(AttributeKind::Generic { index, .. }) if vertex => {
                        found.inputs.insert(host_location(index)?);
                    }
                    _ => {
                        let attribute = attribute_arg(at, op, 1)?;
                        return Err(LowerError::Input { stage, attribute });
                    }
                },
                Opcode::SetAttribute => match classify(attribute_arg(at, op, 2)?) {
                    Some(AttributeKind::Generic { index, .. }) if vertex => {
                        found.generic_outputs.insert(host_location(index)?);
                    }
                    Some(AttributeKind::Position { .. }) if vertex => {}
                    _ => {
                        let attribute = attribute_arg(at, op, 2)?;
                        return Err(LowerError::Output { stage, attribute });
                    }
                },
                Opcode::LaneId if vertex => return Err(LowerError::LaneIdInVertex),
                Opcode::LaneId => found.lane_id = true,
                Opcode::SetFragColor => {
                    let Some(Imm::U32(target)) = imm_arg(op, 0) else {
                        return Err(LowerError::Operand { at, index: 0 });
                    };
                    let Some(Imm::U32(component)) = imm_arg(op, 1) else {
                        return Err(LowerError::Operand { at, index: 1 });
                    };
                    if vertex {
                        return Err(LowerError::ColorInStage(stage));
                    }
                    let target = u8::try_from(target)
                        .ok()
                        .filter(|&t| usize::from(t) < COLOR_TARGETS)
                        .ok_or(LowerError::ColorOutput { target, component })?;
                    if component >= u32::from(COMPONENTS) {
                        return Err(LowerError::ColorOutput {
                            target: u32::from(target),
                            component,
                        });
                    }
                    found.targets.insert(target);
                }
                Opcode::BitCastF32U32
                | Opcode::BitCastU32F32
                | Opcode::FPAbs32
                | Opcode::FPNeg32
                | Opcode::FPFma32
                | Opcode::FPSaturate32 => {}
                opcode => return Err(LowerError::Operation { at, opcode }),
            }
        }
        Ok(found)
    }
}

/// A location binding; varyings get perspective interpolation at the pixel centre.
fn location(location: u8, varying: bool) -> Binding {
    Binding::Location {
        location: u32::from(location),
        interpolation: varying.then_some(Interpolation::Perspective),
        sampling: varying.then_some(Sampling::Center),
        blend_src: None,
        per_primitive: false,
    }
}

/// The member of the output struct a write lands in.
#[derive(Clone, Copy)]
struct OutputMember {
    index: u32,
}

/// Lowers a straight-line function of `stage` (a control-flow graph of one block that returns)
/// to a validated naga module.
pub fn lower(stage: GuestStage, function: &Function) -> Result<HostShader, LowerError> {
    function.verify()?;
    let (host_stage, shader_stage) = match stage {
        GuestStage::Vertex => (HostStage::Vertex, ShaderStage::Vertex),
        GuestStage::Pixel => (HostStage::Fragment, ShaderStage::Fragment),
        other => return Err(LowerError::Stage(other)),
    };
    let straight = function.structure().is_none()
        && function.block_count() == 1
        && function
            .block(function.entry())
            .is_some_and(|entry| entry.terminator == Terminator::Return);
    if !straight {
        return Err(LowerError::ControlFlow);
    }
    let interface = Interface::collect(stage, function)?;
    let requirements = Requirements {
        subgroup: interface.lane_id,
    };

    let mut module = Module::default();
    let types = Types {
        u32: ty(&mut module, TypeInner::Scalar(Scalar::U32)),
        vec4: ty(
            &mut module,
            TypeInner::Vector {
                size: VectorSize::Quad,
                scalar: Scalar::F32,
            },
        ),
    };
    let mut lowering = Lowering::new(FunctionBuilder::new(ENTRY_POINT));
    let inputs = lowering.declare_inputs(&interface, types);
    let outputs = lowering.declare_outputs(&mut module, stage, &interface, types);
    for &reg in &interface.registers {
        let local = lowering.fb.local(&format!("r{reg}"), types.u32);
        lowering.registers.insert(reg, local);
    }

    let mut body = Block::new();
    for (at, inst) in function.listed() {
        if let Inst::Op(op) = inst {
            lowering.inst(&mut body, at, op)?;
        }
    }
    let result = match lowering.output {
        Some(output) => {
            let pointer = lowering
                .fb
                .expr(&mut body, Expression::LocalVariable(output));
            Some(lowering.fb.expr(&mut body, Expression::Load { pointer }))
        }
        None => None,
    };
    push(&mut body, Statement::Return { value: result });
    module.entry_points.push(naga::EntryPoint {
        name: ENTRY_POINT.into(),
        stage: shader_stage,
        early_depth_test: None,
        workgroup_size: [0, 0, 0],
        workgroup_size_overrides: None,
        function: lowering.fb.finish(body),
        mesh_info: None,
        task_payload: None,
        incoming_ray_payload: None,
    });

    let info = Validator::new(validation_flags(), capabilities(requirements))
        .validate(&module)
        .map_err(|error| LowerError::Validation(error.to_string()))?;
    Ok(HostShader {
        stage: host_stage,
        code: HostCode::Naga {
            module: Box::new(module),
            info: Box::new(info),
        },
        entry: ENTRY_POINT.into(),
        inputs,
        outputs,
        requirements,
    })
}

/// The types every module uses.
#[derive(Clone, Copy)]
struct Types {
    u32: Handle<naga::Type>,
    vec4: Handle<naga::Type>,
}

fn arg_index(fb: &FunctionBuilder) -> u32 {
    u32::try_from(fb.function.arguments.len()).unwrap_or(u32::MAX)
}

/// The state of lowering one function.
struct Lowering {
    fb: FunctionBuilder,
    values: BTreeMap<Value, Handle<Expression>>,
    registers: BTreeMap<u8, Handle<LocalVariable>>,
    input_args: BTreeMap<u8, u32>,
    lane_arg: Option<u32>,
    output: Option<Handle<LocalVariable>>,
    members: Vec<StructMember>,
    position: Option<OutputMember>,
    generic_members: BTreeMap<u8, OutputMember>,
    target_members: BTreeMap<u8, OutputMember>,
}

impl Lowering {
    fn new(fb: FunctionBuilder) -> Self {
        Self {
            fb,
            values: BTreeMap::new(),
            registers: BTreeMap::new(),
            input_args: BTreeMap::new(),
            lane_arg: None,
            output: None,
            members: Vec::new(),
            position: None,
            generic_members: BTreeMap::new(),
            target_members: BTreeMap::new(),
        }
    }

    /// Declares the entry point's arguments: generic inputs by location, then the lane id.
    fn declare_inputs(&mut self, interface: &Interface, types: Types) -> Vec<InterfaceSlot> {
        let mut inputs = Vec::new();
        for &generic in &interface.inputs {
            self.input_args.insert(generic, arg_index(&self.fb));
            self.fb.function.arguments.push(FunctionArgument {
                name: Some(format!("in{generic}")),
                ty: types.vec4,
                binding: Some(location(generic, false)),
            });
            inputs.push(InterfaceSlot {
                binding: SlotBinding::Location(u32::from(generic)),
                ty: SlotType::F32x4,
                guest: GuestSlot::Generic(generic),
            });
        }
        if interface.lane_id {
            self.lane_arg = Some(arg_index(&self.fb));
            self.fb.function.arguments.push(FunctionArgument {
                name: Some("lane_id".into()),
                ty: types.u32,
                binding: Some(Binding::BuiltIn(BuiltIn::SubgroupInvocationId)),
            });
            inputs.push(InterfaceSlot {
                binding: SlotBinding::BuiltIn(HostBuiltIn::SubgroupInvocationId),
                ty: SlotType::U32,
                guest: GuestSlot::LaneId,
            });
        }
        inputs
    }

    /// Adds a `vec4<f32>` member to the output struct.
    fn member(&mut self, name: String, binding: Binding, types: Types) -> OutputMember {
        let index = u32::try_from(self.members.len()).unwrap_or(u32::MAX);
        self.members.push(StructMember {
            name: Some(name),
            ty: types.vec4,
            binding: Some(binding),
            offset: index.saturating_mul(VEC4_BYTES),
        });
        OutputMember { index }
    }

    /// Declares the output struct (built-ins first, then locations ascending) and the local
    /// variable the body writes it through.
    fn declare_outputs(
        &mut self,
        module: &mut Module,
        stage: GuestStage,
        interface: &Interface,
        types: Types,
    ) -> Vec<InterfaceSlot> {
        let mut outputs = Vec::new();
        if stage == GuestStage::Vertex {
            let position = Binding::BuiltIn(BuiltIn::Position { invariant: false });
            self.position = Some(self.member("position".into(), position, types));
            outputs.push(InterfaceSlot {
                binding: SlotBinding::BuiltIn(HostBuiltIn::Position),
                ty: SlotType::F32x4,
                guest: GuestSlot::Position,
            });
        }
        for &generic in &interface.generic_outputs {
            let member = self.member(format!("out{generic}"), location(generic, true), types);
            self.generic_members.insert(generic, member);
            outputs.push(InterfaceSlot {
                binding: SlotBinding::Location(u32::from(generic)),
                ty: SlotType::F32x4,
                guest: GuestSlot::Generic(generic),
            });
        }
        for &target in &interface.targets {
            let member = self.member(format!("color{target}"), location(target, false), types);
            self.target_members.insert(target, member);
            outputs.push(InterfaceSlot {
                binding: SlotBinding::Location(u32::from(target)),
                ty: SlotType::F32x4,
                guest: GuestSlot::RenderTarget(target),
            });
        }
        if !self.members.is_empty() {
            let members = std::mem::take(&mut self.members);
            let span = u32::try_from(members.len())
                .unwrap_or(u32::MAX)
                .saturating_mul(VEC4_BYTES);
            let struct_ty = named_ty(module, "Output", TypeInner::Struct { members, span });
            self.fb.function.result = Some(FunctionResult {
                ty: struct_ty,
                binding: None,
            });
            self.output = Some(self.fb.local("out", struct_ty));
        }
        outputs
    }

    /// The expression of an operand; the verifier has checked it is defined earlier.
    fn value(&self, value: Value) -> Result<Handle<Expression>, LowerError> {
        self.values
            .get(&value)
            .copied()
            .ok_or(LowerError::Verify(VerifyError::Missing(value)))
    }

    /// The expression of an operand of instruction `at`: a value's, or a literal made here for
    /// a 32-bit immediate (where slice 1's IR had an `Imm32` instruction, so the module's
    /// expressions keep their order).
    fn operand(
        &mut self,
        body: &mut Block,
        at: Value,
        operand: Operand,
    ) -> Result<Handle<Expression>, LowerError> {
        let literal = match operand {
            Operand::Value(value) => return self.value(value),
            Operand::Imm(Imm::U32(bits)) => Literal::U32(bits),
            Operand::Imm(Imm::F32(bits)) => Literal::F32(f32::from_bits(bits)),
            Operand::Imm(_) => {
                return Err(LowerError::Internal(format!(
                    "an immediate operand of {at}"
                )));
            }
        };
        Ok(self.fb.expr(body, Expression::Literal(literal)))
    }

    fn register(&mut self, body: &mut Block, reg: u8) -> Result<Handle<Expression>, LowerError> {
        let local = *self
            .registers
            .get(&reg)
            .ok_or(LowerError::Internal(format!("no variable for r{reg}")))?;
        Ok(self.fb.expr(body, Expression::LocalVariable(local)))
    }

    /// Stores `value` to component `component` of output member `member`.
    fn store_output(
        &mut self,
        body: &mut Block,
        at: Value,
        member: Option<OutputMember>,
        component: u32,
        value: Operand,
    ) -> Result<(), LowerError> {
        let value = self.operand(body, at, value)?;
        let (Some(output), Some(member)) = (self.output, member) else {
            return Err(LowerError::Internal("no output member".into()));
        };
        let base = self.fb.expr(body, Expression::LocalVariable(output));
        let member = self.fb.expr(
            body,
            Expression::AccessIndex {
                base,
                index: member.index,
            },
        );
        let pointer = self.fb.expr(
            body,
            Expression::AccessIndex {
                base: member,
                index: component,
            },
        );
        push(body, Statement::Store { pointer, value });
        Ok(())
    }

    /// Reinterprets `value`'s bits as `kind`.
    fn bitcast(
        &mut self,
        body: &mut Block,
        at: Value,
        value: Operand,
        kind: ScalarKind,
    ) -> Result<Handle<Expression>, LowerError> {
        let expr = self.operand(body, at, value)?;
        Ok(self.fb.expr(
            body,
            Expression::As {
                expr,
                kind,
                convert: None,
            },
        ))
    }

    fn math(
        &mut self,
        body: &mut Block,
        fun: MathFunction,
        args: [Option<Handle<Expression>>; 3],
        arg: Handle<Expression>,
    ) -> Handle<Expression> {
        let [first, second, third] = args;
        self.fb.expr(
            body,
            Expression::Math {
                fun,
                arg,
                arg1: first,
                arg2: second,
                arg3: third,
            },
        )
    }

    /// Argument `index` of `op`.
    fn arg(at: Value, op: &Operation, index: usize) -> Result<Operand, LowerError> {
        op.args
            .get(index)
            .copied()
            .ok_or(LowerError::Operand { at, index })
    }

    fn inst(&mut self, body: &mut Block, at: Value, op: &Operation) -> Result<(), LowerError> {
        let arg = |index| Self::arg(at, op, index);
        let expr = match op.opcode {
            Opcode::GetRegister => {
                let Some(Imm::Reg(reg)) = imm_arg(op, 0) else {
                    return Err(LowerError::Operand { at, index: 0 });
                };
                let pointer = self.register(body, reg.index())?;
                Some(self.fb.expr(body, Expression::Load { pointer }))
            }
            Opcode::SetRegister => {
                let Some(Imm::Reg(reg)) = imm_arg(op, 0) else {
                    return Err(LowerError::Operand { at, index: 0 });
                };
                let value = self.operand(body, at, arg(1)?)?;
                let pointer = self.register(body, reg.index())?;
                push(body, Statement::Store { pointer, value });
                None
            }
            Opcode::BitCastF32U32 => Some(self.bitcast(body, at, arg(0)?, ScalarKind::Float)?),
            Opcode::BitCastU32F32 => Some(self.bitcast(body, at, arg(0)?, ScalarKind::Uint)?),
            Opcode::FPAbs32 => {
                let value = self.operand(body, at, arg(0)?)?;
                Some(self.math(body, MathFunction::Abs, [None; 3], value))
            }
            Opcode::FPNeg32 => {
                let expr = self.operand(body, at, arg(0)?)?;
                Some(self.fb.expr(
                    body,
                    Expression::Unary {
                        op: UnaryOperator::Negate,
                        expr,
                    },
                ))
            }
            // The controls are not expressed (decision 10, in the module docs).
            Opcode::FPFma32 => {
                let factor = self.operand(body, at, arg(0)?)?;
                let multiplier = self.operand(body, at, arg(1)?)?;
                let addend = self.operand(body, at, arg(2)?)?;
                let args = [Some(multiplier), Some(addend), None];
                Some(self.math(body, MathFunction::Fma, args, factor))
            }
            Opcode::FPSaturate32 => {
                let value = self.operand(body, at, arg(0)?)?;
                Some(self.math(body, MathFunction::Saturate, [None; 3], value))
            }
            Opcode::GetAttribute => {
                let attribute = attribute_arg(at, op, 1)?;
                let Some(AttributeKind::Generic { index, component }) = classify(attribute) else {
                    return Err(LowerError::Internal(format!("input {attribute}")));
                };
                let input = *self
                    .input_args
                    .get(&index)
                    .ok_or(LowerError::Internal(format!("input {attribute}")))?;
                let base = self.fb.expr(body, Expression::FunctionArgument(input));
                Some(self.fb.expr(
                    body,
                    Expression::AccessIndex {
                        base,
                        index: component,
                    },
                ))
            }
            Opcode::SetAttribute => {
                let attribute = attribute_arg(at, op, 2)?;
                let (member, component) = match classify(attribute) {
                    Some(AttributeKind::Position { component }) => (self.position, component),
                    Some(AttributeKind::Generic { index, component }) => {
                        (self.generic_members.get(&index).copied(), component)
                    }
                    None => (None, 0),
                };
                self.store_output(body, at, member, component, arg(1)?)?;
                None
            }
            Opcode::LaneId => {
                let lane = self
                    .lane_arg
                    .ok_or(LowerError::Internal("no lane id argument".into()))?;
                Some(self.fb.expr(body, Expression::FunctionArgument(lane)))
            }
            Opcode::SetFragColor => {
                let (Some(Imm::U32(target)), Some(Imm::U32(component))) =
                    (imm_arg(op, 0), imm_arg(op, 1))
                else {
                    return Err(LowerError::Internal(format!("colour output at {at}")));
                };
                let member = u8::try_from(target)
                    .ok()
                    .and_then(|target| self.target_members.get(&target).copied());
                self.store_output(body, at, member, component, arg(2)?)?;
                None
            }
            opcode => return Err(LowerError::Operation { at, opcode }),
        };
        if let Some(expr) = expr {
            self.values.insert(at, expr);
        }
        Ok(())
    }
}

/// Immediate argument `index` of `op`, if it is one.
fn imm_arg(op: &Operation, index: usize) -> Option<Imm> {
    op.args.get(index)?.imm()
}

/// The attribute an attribute access names (argument 0), after checking that its vertex index
/// (argument `vertex`) is the immediate zero: the stages lowered here have one vertex.
fn attribute_arg(at: Value, op: &Operation, vertex: usize) -> Result<Attribute, LowerError> {
    if imm_arg(op, vertex) != Some(Imm::U32(0)) {
        return Err(LowerError::Operand { at, index: vertex });
    }
    match imm_arg(op, 0) {
        Some(Imm::Attribute(attribute)) => Ok(attribute),
        _ => Err(LowerError::Operand { at, index: 0 }),
    }
}
