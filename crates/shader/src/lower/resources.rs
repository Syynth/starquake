//! Resources: the binding convention, the constant-buffer banks and the driver constants
//! (slice 17, `docs/specs/shader-pipeline.md`, "What the render backend (#14) gets, through the
//! GPU crate": "The binding convention (which group and binding each guest resource kind gets)
//! is defined once, in the shader crate: it is written into the module and listed in
//! `ResourceSlot`; the render backend does not guess it").
//!
//! # The binding convention
//!
//! | Resource | Group | Binding |
//! |---|---|---|
//! | the driver constants ([`crate::driver_constants`]) | [`resource_group`] of the stage | [`DRIVER_CONSTANTS_BINDING`] (0) |
//! | constant-buffer bank `b` (0 to 31) | [`resource_group`] of the stage | [`constant_buffer_binding`]`(b)`, `1 + b` |
//!
//! Each host stage has its own group ([`resource_group`]: vertex 0, fragment 1, compute 0),
//! because a pipeline's stages share one resource at a group and binding: under wgpu a
//! pipeline layout has one entry per group and binding, made visible to every stage that uses
//! it, and the render backend merges two stages' rows at one slot into one entry when they agree
//! on its kind (render-wgpu `pipeline::layout`, `STAGE_CONFLICT`). The guest's vertex and pixel
//! programs read different constant buffers under the same bank number (each stage has its own
//! bind group of banks: Eden-NX2's `stage_index`, the GPU crate's `bind_group_of`), so sharing a
//! group would bind one bank's buffer for both stages without an error. The convention is
//! starquake's own (no reference assigns host bindings this way; Eden's SPIR-V backend numbers
//! bindings in declaration order per stage, `spirv_emit_context.cpp`, `DefineConstantBuffers`),
//! and a **guess** at what serves the later kinds: textures, samplers and storage buffers are
//! not lowered yet, and take bindings past the banks' when they are.
//!
//! The driver constants are one buffer for every stage of a draw; each stage that reads them
//! lists them in its own group, and the GPU crate binds the same buffer in each.
//!
//! # Constant-buffer reads
//!
//! Bank `b` is declared, when the function reads it, as a uniform `array<vec4<u32>, size / 16>`
//! (the uniform address space's array stride is 16 bytes), where `size` is the bank's entry in
//! [`LowerInputs::constant_buffers`]: slice 15's collect info gives it
//! ([`constant_buffer_sizes`], [`crate::info::ShaderInfo::constant_buffers`]), a multiple of
//! [`CBUF_ALIGNMENT`] that covers every read at an immediate offset, or [`INDEXED_CBUF_SIZE`]
//! for a read at a register offset. Each of `GetCbufU8`, `GetCbufS8`, `GetCbufU16`,
//! `GetCbufS16`, `GetCbufU32`, `GetCbufF32` and `GetCbufU32x2` reads at its byte offset aligned
//! down to its width, as the interpreter does ([`crate::interp`]); the 8- and 16-bit reads
//! extract the field at bit `8 * (offset % 4)` of the word, zero- or sign-extended.
//!
//! **Bounds.** A read whose offset is at or past the declared size gives zero, and reads the
//! buffer at offset 0 instead, so the host never sees an index past the array. Because every
//! width divides [`CBUF_ALIGNMENT`] and the size is a multiple of it, "offset below the size"
//! is exactly "the whole read inside the declared bytes". This is the same rule slice 13 applies
//! to indexed reads ([`crate::opt::resources`], "Constant-buffer reads"; its zero is a guess at
//! the hardware's answer), applied here to every read, immediate offsets included, so the
//! module is in bounds whatever pass ran before it.
//!
//! # Checks
//!
//! In [`super::lower_with`]'s order: the verifier (step 1) refuses a bank or offset that is
//! not a `u32` ([`LowerError::Verify`]); each bank's size in [`LowerInputs::constant_buffers`]
//! is 0 (not declared) or a multiple of [`CBUF_ALIGNMENT`] no larger than
//! [`INDEXED_CBUF_SIZE`], the most a bank holds ([`LowerError::ConstantBufferSize`], step 3,
//! the lowest bank first, read or not); then, per read, in the walk of step 5: the bank is an
//! immediate, not a value ([`LowerError::ConstantBufferRegisterBank`]), below [`CBUF_BANKS`]
//! ([`LowerError::ConstantBufferBank`]), and its size is not 0
//! ([`LowerError::ConstantBufferUnsized`]). The offset may be an immediate or a value.
//!
//! # Where this differs from Eden
//!
//! | What | Here | Eden @ cf654052 | Pinned by |
//! |---|---|---|---|
//! | A bank that is a register (`LDC` with a register bank) | Refused, [`LowerError::ConstantBufferRegisterBank`] | A switch over every bank the program reads (`emit_spirv_context_get_set.cpp`, `GetCbuf`'s non-immediate binding arm) | `tests::resources::a_register_bank_is_refused` |
//! | A read past the declared size | Zero, read at offset 0 | Left to the host's robust buffer access, or zero past element `0xFFFF` where the host's robustness is broken (`GetCbuf`, `has_broken_robust`) | `tests::resources::reads_match_the_interpreter_on_the_gpu` |
//! | The buffer's declared type | `array<vec4<u32>, size / 16>`, one declaration per bank | One declaration per element type the program reads (`DefineConstantBuffers`, `used_constant_buffer_types`), each `array<T, 0x10000 / sizeof(T)>` | `tests::resources::resource_tables_are_golden` |

use std::collections::BTreeMap;
use std::num::NonZeroU32;

use naga::{
    AddressSpace, ArraySize, BinaryOperator, Block, Expression, GlobalVariable, Handle, Literal,
    MathFunction, MemoryDecorations, ResourceBinding, Scalar, ScalarKind, Span, StructMember,
    TypeInner, VectorSize,
};

use super::builder::{named_ty, ty};
use super::{Interface, LowerError, LowerInputs, Lowering, imm_arg};
use crate::driver_constants::{ConstantType, DRIVER_CONSTANTS_SIZE, DriverConstant};
use crate::host::{HostStage, ResourceKind, ResourceSlot, ResourceSource};
use crate::info::{CBUF_ALIGNMENT, INDEXED_CBUF_SIZE, ShaderInfo};
use crate::ir::{Imm, Opcode, Operand, Operation, Value};
pub use crate::translate::float::CBUF_BANKS;

/// The binding of the driver constants in each stage's group (module docs, "The binding
/// convention").
pub const DRIVER_CONSTANTS_BINDING: u32 = 0;

/// The binding of constant-buffer bank 0; bank `b` is at this plus `b`.
pub const CONSTANT_BUFFER_BINDING_BASE: u32 = 1;

/// How many constant-buffer banks [`LowerInputs::constant_buffers`] sizes.
pub const CONSTANT_BUFFER_SLOTS: usize = CBUF_BANKS as usize;

/// The bytes of one element of a declared constant buffer, a `vec4<u32>`.
const CBUF_ELEMENT_BYTES: u32 = 16;

/// The shift from a byte offset to its element (`offset >> 4`).
const BYTE_TO_ELEMENT_SHIFT: u32 = 4;

/// The shift from a byte offset to its word (`offset >> 2`).
const BYTE_TO_WORD_SHIFT: u32 = 2;

/// The words of an element, less one: the mask of a word's index in its element.
const WORD_IN_ELEMENT_MASK: u32 = 3;

/// The shift from a byte's index in its word to its bit (`* 8`).
const BYTE_TO_BIT_SHIFT: u32 = 3;

/// The bind group of a host stage's resources (module docs, "The binding convention").
pub const fn resource_group(stage: HostStage) -> u32 {
    match stage {
        HostStage::Vertex | HostStage::Compute => 0,
        HostStage::Fragment => 1,
    }
}

/// The binding of constant-buffer bank `bank` (module docs, "The binding convention").
pub const fn constant_buffer_binding(bank: u8) -> u32 {
    CONSTANT_BUFFER_BINDING_BASE + bank as u32
}

/// Each bank's size as [`LowerInputs::constant_buffers`] takes it, from what collect info found
/// (0 for a bank the program does not read).
pub fn constant_buffer_sizes(info: &ShaderInfo) -> [u32; CONSTANT_BUFFER_SLOTS] {
    let mut sizes = [0; CONSTANT_BUFFER_SLOTS];
    for used in info.constant_buffers() {
        if let Some(slot) = sizes.get_mut(usize::from(used.bank)) {
            *slot = used.size;
        }
    }
    sizes
}

/// Checks each bank's size (module docs, "Checks"), the lowest bank first.
pub(super) fn check_sizes(sizes: &[u32; CONSTANT_BUFFER_SLOTS]) -> Result<(), LowerError> {
    for (bank, &size) in (0_u8..).zip(sizes) {
        if size % CBUF_ALIGNMENT != 0 || size > INDEXED_CBUF_SIZE {
            return Err(LowerError::ConstantBufferSize { bank, size });
        }
    }
    Ok(())
}

/// What a constant-buffer read reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Read {
    /// An 8- or 16-bit field of a word, zero- or sign-extended.
    Field { bits: u32, signed: bool },
    /// One word, as a `u32` or reinterpreted as an `f32`.
    Word { float: bool },
    /// Two words, as a `vec2<u32>`.
    Pair,
}

impl Read {
    /// The read an opcode makes, if it is a constant-buffer read.
    const fn of(opcode: Opcode) -> Option<Self> {
        Some(match opcode {
            Opcode::GetCbufU8 => Self::Field {
                bits: 8,
                signed: false,
            },
            Opcode::GetCbufS8 => Self::Field {
                bits: 8,
                signed: true,
            },
            Opcode::GetCbufU16 => Self::Field {
                bits: 16,
                signed: false,
            },
            Opcode::GetCbufS16 => Self::Field {
                bits: 16,
                signed: true,
            },
            Opcode::GetCbufU32 => Self::Word { float: false },
            Opcode::GetCbufF32 => Self::Word { float: true },
            Opcode::GetCbufU32x2 => Self::Pair,
            _ => return None,
        })
    }

    /// The read's width in bytes, which its offset is aligned down to.
    const fn bytes(self) -> u32 {
        match self {
            Self::Field { bits, .. } => bits / 8,
            Self::Word { .. } => 4,
            Self::Pair => 8,
        }
    }
}

/// Whether `opcode` reads a constant buffer.
pub(super) const fn is_cbuf_read(opcode: Opcode) -> bool {
    Read::of(opcode).is_some()
}

/// The bank a constant-buffer read names, after the checks of the module docs ("Checks"). The
/// verifier has checked that both arguments are `u32`s (an immediate `u32` or a `u32` value),
/// so the [`LowerError::Operand`] arm is not reached after it.
pub(super) fn read_bank(at: Value, op: &Operation, inputs: &LowerInputs) -> Result<u8, LowerError> {
    let bank = match op.args.first() {
        Some(Operand::Imm(Imm::U32(bank))) => *bank,
        Some(Operand::Value(_)) => return Err(LowerError::ConstantBufferRegisterBank { at }),
        _ => return Err(LowerError::Operand { at, index: 0 }),
    };
    let bank = u8::try_from(bank)
        .ok()
        .filter(|&bank| bank < CBUF_BANKS)
        .ok_or(LowerError::ConstantBufferBank { at, bank })?;
    let size = inputs
        .constant_buffers
        .get(usize::from(bank))
        .copied()
        .unwrap_or(0);
    if size == 0 {
        return Err(LowerError::ConstantBufferUnsized { at, bank });
    }
    Ok(bank)
}

/// One declared bank: its global and its size in bytes.
#[derive(Clone, Copy)]
struct Bank {
    global: Handle<GlobalVariable>,
    size: u32,
}

/// The resources a function binds, once declared.
#[derive(Default)]
pub(super) struct Resources {
    banks: BTreeMap<u8, Bank>,
    driver_constants: Option<Handle<GlobalVariable>>,
}

impl Lowering {
    /// Declares the resources `interface` names, in binding order, and returns the table
    /// ([`crate::host::HostShader::resources`]).
    pub(super) fn declare_resources(
        &mut self,
        stage: HostStage,
        interface: &Interface,
        inputs: &LowerInputs,
    ) -> Result<Vec<ResourceSlot>, LowerError> {
        let group = resource_group(stage);
        let mut table = Vec::new();
        if interface.driver_constants {
            let global = self.declare_driver_constants(group);
            self.resources.driver_constants = Some(global);
            table.push(ResourceSlot {
                group,
                binding: DRIVER_CONSTANTS_BINDING,
                kind: ResourceKind::UniformBuffer {
                    size: driver_constants_bytes(),
                },
                source: ResourceSource::DriverConstants,
            });
        }
        for &bank in &interface.constant_buffers {
            let size = inputs
                .constant_buffers
                .get(usize::from(bank))
                .copied()
                .unwrap_or(0);
            let elements = NonZeroU32::new(size / CBUF_ELEMENT_BYTES).ok_or_else(|| {
                LowerError::Internal(format!("bank {bank} is read but declared with no bytes"))
            })?;
            let vec4 = ty(
                &mut self.module,
                TypeInner::Vector {
                    size: VectorSize::Quad,
                    scalar: Scalar::U32,
                },
            );
            let array = ty(
                &mut self.module,
                TypeInner::Array {
                    base: vec4,
                    size: ArraySize::Constant(elements),
                    stride: CBUF_ELEMENT_BYTES,
                },
            );
            let binding = constant_buffer_binding(bank);
            let global = self.uniform(&format!("cbuf{bank}"), array, group, binding);
            self.resources.banks.insert(bank, Bank { global, size });
            table.push(ResourceSlot {
                group,
                binding,
                kind: ResourceKind::UniformBuffer { size },
                source: ResourceSource::ConstantBuffer { bank },
            });
        }
        Ok(table)
    }

    /// A uniform global of type `ty` at `group` and `binding`.
    fn uniform(
        &mut self,
        name: &str,
        ty: Handle<naga::Type>,
        group: u32,
        binding: u32,
    ) -> Handle<GlobalVariable> {
        self.module.global_variables.append(
            GlobalVariable {
                name: Some(name.into()),
                space: AddressSpace::Uniform,
                binding: Some(ResourceBinding { group, binding }),
                ty,
                init: None,
                memory_decorations: MemoryDecorations::empty(),
            },
            Span::UNDEFINED,
        )
    }

    /// Declares the driver constants: a struct with one member per [`DriverConstant`], in
    /// [`DriverConstant::ALL`] order at its offset, [`DRIVER_CONSTANTS_SIZE`] bytes long.
    fn declare_driver_constants(&mut self, group: u32) -> Handle<GlobalVariable> {
        let f32_ty = ty(&mut self.module, TypeInner::Scalar(Scalar::F32));
        let members = DriverConstant::ALL
            .iter()
            .map(|&constant| StructMember {
                name: Some(constant_name(constant).into()),
                ty: match constant.ty() {
                    ConstantType::F32 => f32_ty,
                    ConstantType::U32 => self.types.u32,
                },
                binding: None,
                offset: u32::try_from(constant.offset()).unwrap_or(u32::MAX),
            })
            .collect();
        let block = named_ty(
            &mut self.module,
            "DriverConstants",
            TypeInner::Struct {
                members,
                span: driver_constants_bytes(),
            },
        );
        self.uniform("driver_constants", block, group, DRIVER_CONSTANTS_BINDING)
    }

    /// The value of driver constant `constant`, loaded from the block.
    pub(super) fn driver_constant(
        &mut self,
        body: &mut Block,
        constant: DriverConstant,
    ) -> Result<Handle<Expression>, LowerError> {
        let global = self
            .resources
            .driver_constants
            .ok_or_else(|| LowerError::Internal("the driver constants are not declared".into()))?;
        let index = DriverConstant::ALL
            .iter()
            .position(|&c| c == constant)
            .and_then(|index| u32::try_from(index).ok())
            .ok_or_else(|| LowerError::Internal(format!("no member for {constant:?}")))?;
        let base = self.expr(body, Expression::GlobalVariable(global));
        let pointer = self.expr(body, Expression::AccessIndex { base, index });
        Ok(self.expr(body, Expression::Load { pointer }))
    }

    /// Lowers constant-buffer read `op` (module docs, "Constant-buffer reads").
    pub(super) fn cbuf_read(
        &mut self,
        body: &mut Block,
        at: Value,
        op: &Operation,
    ) -> Result<Handle<Expression>, LowerError> {
        let read = Read::of(op.opcode).ok_or_else(|| {
            LowerError::Internal(format!("{} is not a constant-buffer read", op.opcode))
        })?;
        let bank = match imm_arg(op, 0) {
            Some(Imm::U32(bank)) => u8::try_from(bank).ok(),
            _ => None,
        }
        .and_then(|bank| self.resources.banks.get(&bank).copied())
        .ok_or_else(|| LowerError::Internal(format!("{at}: its bank is not declared")))?;
        let offset = self.arg(body, at, op, 1)?;
        // The offset aligned down to the read's width.
        let align = self.lit(body, Literal::U32(!(read.bytes() - 1)));
        let aligned = self.binary(body, BinaryOperator::And, offset, align);
        let size = self.lit(body, Literal::U32(bank.size));
        let inside = self.binary(body, BinaryOperator::Less, offset, size);
        let zero = self.lit(body, Literal::U32(0));
        let safe = self.select(body, inside, aligned, zero);
        let element_shift = self.lit(body, Literal::U32(BYTE_TO_ELEMENT_SHIFT));
        let element = self.binary(body, BinaryOperator::ShiftRight, safe, element_shift);
        let word_shift = self.lit(body, Literal::U32(BYTE_TO_WORD_SHIFT));
        let words = self.binary(body, BinaryOperator::ShiftRight, safe, word_shift);
        let mask = self.lit(body, Literal::U32(WORD_IN_ELEMENT_MASK));
        let word = self.binary(body, BinaryOperator::And, words, mask);
        let base = self.expr(body, Expression::GlobalVariable(bank.global));
        let vector = self.expr(
            body,
            Expression::Access {
                base,
                index: element,
            },
        );
        let load_word = |this: &mut Self, body: &mut Block, index| {
            let pointer = this.expr(
                body,
                Expression::Access {
                    base: vector,
                    index,
                },
            );
            this.expr(body, Expression::Load { pointer })
        };
        let first = load_word(self, body, word);
        let (value, zero) = match read {
            Read::Word { float: false } => (first, zero),
            Read::Word { float: true } => {
                let value = self.cast(body, first, ScalarKind::Float, None);
                let zero = self.lit(body, Literal::F32(0.0));
                (value, zero)
            }
            Read::Field { bits, signed } => {
                let byte_mask = self.lit(body, Literal::U32(WORD_IN_ELEMENT_MASK));
                let byte = self.binary(body, BinaryOperator::And, safe, byte_mask);
                let bit_shift = self.lit(body, Literal::U32(BYTE_TO_BIT_SHIFT));
                let bit = self.binary(body, BinaryOperator::ShiftLeft, byte, bit_shift);
                let count = self.lit(body, Literal::U32(bits));
                let source = if signed {
                    self.cast(body, first, ScalarKind::Sint, None)
                } else {
                    first
                };
                let field = self.math(body, MathFunction::ExtractBits, source, &[bit, count]);
                let value = if signed {
                    self.cast(body, field, ScalarKind::Uint, None)
                } else {
                    field
                };
                (value, zero)
            }
            Read::Pair => {
                let one = self.lit(body, Literal::U32(1));
                let next = self.binary(body, BinaryOperator::Add, word, one);
                let second = load_word(self, body, next);
                let u32x2 = ty(
                    &mut self.module,
                    TypeInner::Vector {
                        size: VectorSize::Bi,
                        scalar: Scalar::U32,
                    },
                );
                let value = self.expr(
                    body,
                    Expression::Compose {
                        ty: u32x2,
                        components: vec![first, second],
                    },
                );
                let zero = self.expr(body, Expression::ZeroValue(u32x2));
                (value, zero)
            }
        };
        Ok(self.select(body, inside, value, zero))
    }
}

/// The member name of a driver constant in the module.
const fn constant_name(constant: DriverConstant) -> &'static str {
    match constant {
        DriverConstant::AlphaReference => "alpha_reference",
        DriverConstant::ViewportFlip => "viewport_flip",
        DriverConstant::RenderAreaWidth => "render_area_width",
        DriverConstant::RenderAreaHeight => "render_area_height",
        DriverConstant::BaseVertex => "base_vertex",
        DriverConstant::BaseInstance => "base_instance",
        DriverConstant::PositionYScale => "position_y_scale",
        DriverConstant::PositionZScale => "position_z_scale",
        DriverConstant::PositionZOffset => "position_z_offset",
    }
}

/// The driver constants' size in bytes, [`DRIVER_CONSTANTS_SIZE`], as a `u32`.
fn driver_constants_bytes() -> u32 {
    u32::try_from(DRIVER_CONSTANTS_SIZE).unwrap_or(u32::MAX)
}
