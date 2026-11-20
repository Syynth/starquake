//! Running a lowered compute shader on the host's GPU and reading its registers back, to
//! compare the lowering's semantics with the interpreter's.
//!
//! The lowered module has no outputs, so the test patches a copy: a storage buffer of `u32`s
//! at group 0, binding 0, and, in front of every `return` of the entry point, a store of each
//! register asked for into the buffer, in the order asked. One invocation runs (the shader is
//! lowered with a workgroup of one), and the buffer is read back. Every wait is bounded by
//! [`WAIT`]. Skipped, with a notice, where the host has no GPU adapter or lacks a feature the
//! shader needs.

use std::num::NonZeroU32;
use std::sync::mpsc;
use std::time::Duration;

use naga::{
    AddressSpace, ArraySize, Block, Expression, GlobalVariable, Handle, MemoryDecorations,
    ResourceBinding, Span, Statement, StorageAccess, TypeInner,
};

use super::super::builder::ty;
use super::*;
use crate::interp::Machine;
use crate::ir::Builder;

/// `function` (one block) with each `u32`, `f32` and `u1` immediate that is not a component or
/// goto index replaced by a value computed from the local invocation id (0 for the one
/// invocation run), so the host compiler cannot fold the operations away and the GPU computes
/// them at run time.
pub(super) fn opaque(function: &Function) -> Function {
    let mut out = Function::new();
    let entry = out.entry();
    let mut builder = Builder::new(&mut out, entry);
    let index = |value: u32| Operand::Imm(Imm::U32(value));
    let id = builder
        .op(Opcode::LocalInvocationId, &[])
        .expect("the invocation id");
    let zero: Operand = builder
        .op(Opcode::CompositeExtractU32x3, &[id.into(), index(0)])
        .expect("its x")
        .into();
    let mut map = std::collections::BTreeMap::new();
    for (at, inst) in function.listed() {
        let op = match inst {
            Inst::Op(op) => Some(op),
            Inst::Phi(_) => None,
        }
        .expect("no phi");
        let mut args = Vec::new();
        for (slot, &arg) in op.args.iter().enumerate() {
            let kept = matches!(
                (op.opcode, slot),
                (Opcode::GetGotoVariable | Opcode::SetGotoVariable, 0)
                    | (
                        Opcode::CompositeExtractU32x2
                            | Opcode::CompositeExtractU32x3
                            | Opcode::CompositeExtractU32x4
                            | Opcode::CompositeExtractF16x2
                            | Opcode::CompositeExtractF16x3
                            | Opcode::CompositeExtractF16x4
                            | Opcode::CompositeExtractF32x2
                            | Opcode::CompositeExtractF32x3
                            | Opcode::CompositeExtractF32x4,
                        1
                    )
                    | (
                        Opcode::CompositeInsertU32x2
                            | Opcode::CompositeInsertU32x3
                            | Opcode::CompositeInsertU32x4
                            | Opcode::CompositeInsertF16x2
                            | Opcode::CompositeInsertF16x3
                            | Opcode::CompositeInsertF16x4
                            | Opcode::CompositeInsertF32x2
                            | Opcode::CompositeInsertF32x3
                            | Opcode::CompositeInsertF32x4,
                        2
                    )
            );
            let mut emit = |opcode, args: &[Operand]| -> Operand {
                builder.op(opcode, args).expect("a wrapper").into()
            };
            args.push(match arg {
                Operand::Value(value) => map.get(&value).copied().expect("defined earlier"),
                Operand::Imm(Imm::U32(value)) if !kept => {
                    emit(Opcode::IAdd32, &[index(value), zero])
                }
                Operand::Imm(Imm::F32(bits)) if !kept => {
                    let sum = emit(Opcode::IAdd32, &[index(bits), zero]);
                    emit(Opcode::BitCastF32U32, &[sum])
                }
                Operand::Imm(Imm::U1(value)) => {
                    let compare = if value {
                        Opcode::IEqual
                    } else {
                        Opcode::INotEqual
                    };
                    emit(compare, &[zero, index(0)])
                }
                other @ Operand::Imm(_) => other,
            });
        }
        let made = builder
            .op_with(op.opcode, &args, op.flags)
            .expect("the instruction builds again");
        map.insert(at, Operand::Value(made));
    }
    builder
        .terminate(Terminator::Return)
        .expect("the block returns");
    out
}

/// How long a test waits for the GPU.
const WAIT: Duration = Duration::from_secs(30);

/// The wgpu features a shader with `requirements` needs.
fn features(requirements: Requirements) -> wgpu::Features {
    let mut features = wgpu::Features::empty();
    features.set(wgpu::Features::SUBGROUP, requirements.subgroup);
    features.set(wgpu::Features::SHADER_F16, requirements.f16);
    features.set(wgpu::Features::SHADER_INT64, requirements.i64);
    features
}

/// Inserts `stores` in front of every `Return` in `block`, recursively.
fn before_returns(block: &mut Block, stores: &[Statement]) {
    let old = std::mem::take(block);
    for (statement, span) in old.span_into_iter() {
        let statement = match statement {
            Statement::Return { value } => {
                for store in stores {
                    block.push(store.clone(), Span::UNDEFINED);
                }
                Statement::Return { value }
            }
            Statement::If {
                condition,
                mut accept,
                mut reject,
            } => {
                before_returns(&mut accept, stores);
                before_returns(&mut reject, stores);
                Statement::If {
                    condition,
                    accept,
                    reject,
                }
            }
            Statement::Loop {
                mut body,
                mut continuing,
                break_if,
            } => {
                before_returns(&mut body, stores);
                before_returns(&mut continuing, stores);
                Statement::Loop {
                    body,
                    continuing,
                    break_if,
                }
            }
            Statement::Block(mut inner) => {
                before_returns(&mut inner, stores);
                Statement::Block(inner)
            }
            other => other,
        };
        block.push(statement, span);
    }
}

/// A copy of `shader`'s module that writes registers `regs` to a storage buffer at every
/// return; registers the shader never names read as 0.
fn patched(shader: &HostShader, regs: &[u8]) -> naga::Module {
    let mut module = naga_module(shader).clone();
    let count = u32::try_from(regs.len()).expect("few registers");
    let u32_ty = ty(&mut module, TypeInner::Scalar(naga::Scalar::U32));
    let array = ty(
        &mut module,
        TypeInner::Array {
            base: u32_ty,
            size: ArraySize::Constant(NonZeroU32::new(count).expect("a register")),
            stride: MEMORY_WORD_BYTES,
        },
    );
    let out = module.global_variables.append(
        GlobalVariable {
            name: Some("registers_out".into()),
            space: AddressSpace::Storage {
                access: StorageAccess::LOAD | StorageAccess::STORE,
            },
            binding: Some(ResourceBinding {
                group: 0,
                binding: 0,
            }),
            ty: array,
            init: None,
            memory_decorations: MemoryDecorations::empty(),
        },
        Span::UNDEFINED,
    );
    let function = &mut module.entry_points[0].function;
    let mut stores = Vec::new();
    let append = |function: &mut naga::Function, expression| -> Handle<Expression> {
        function.expressions.append(expression, Span::UNDEFINED)
    };
    for (slot, &reg) in (0_u32..).zip(regs) {
        let name = format!("r{reg}");
        let local = function
            .local_variables
            .iter()
            .find(|(_, local)| local.name.as_deref() == Some(name.as_str()))
            .map(|(handle, _)| handle);
        let value = match local {
            Some(local) => {
                let pointer = append(function, Expression::LocalVariable(local));
                let load = append(function, Expression::Load { pointer });
                stores.push(Statement::Emit(naga::Range::new_from_bounds(load, load)));
                load
            }
            None => append(function, Expression::Literal(naga::Literal::U32(0))),
        };
        let base = append(function, Expression::GlobalVariable(out));
        let element = append(function, Expression::AccessIndex { base, index: slot });
        stores.push(Statement::Emit(naga::Range::new_from_bounds(
            element, element,
        )));
        stores.push(Statement::Store {
            pointer: element,
            value,
        });
    }
    before_returns(&mut function.body, &stores);
    module
}

/// Runs `shader` (a compute shader with a workgroup of one) once on the GPU and returns
/// registers `regs`; `None` when the host cannot run it (a notice says why).
pub(super) fn gpu_registers(test: &str, shader: &HostShader, regs: &[u8]) -> Option<Vec<u32>> {
    let wanted = features(shader.requirements);
    let gpu = TestDevice::open(test, wanted)?;
    if !gpu.has(wanted) {
        eprintln!(
            "skipping the GPU half of {test}: {} lacks {wanted:?}",
            gpu.adapter.name
        );
        return None;
    }
    let module = patched(shader, regs);
    let size = u64::try_from(regs.len()).expect("few") * u64::from(MEMORY_WORD_BYTES);
    let result = gpu.scoped(|device| {
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(test),
            source: wgpu::ShaderSource::Naga(std::borrow::Cow::Owned(module)),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(test),
            layout: None,
            module: &shader_module,
            entry_point: Some(&shader.entry),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let storage = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("registers"),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: storage.as_entire_binding(),
            }],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&storage, 0, &readback, 0, size);
        gpu.queue.submit([encoder.finish()]);
        readback
    });
    let readback = result.expect("wgpu takes the module and the work");
    let (sender, receiver) = mpsc::channel();
    readback
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |mapped| {
            let _sent = sender.send(mapped);
        });
    gpu.device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(WAIT),
        })
        .expect("the GPU finishes");
    receiver
        .recv_timeout(WAIT)
        .expect("the map callback runs")
        .expect("the readback buffer maps");
    let bytes = readback
        .slice(..)
        .get_mapped_range()
        .expect("the mapped range")
        .to_vec();
    Some(
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| u32::from_le_bytes(*word))
            .collect(),
    )
}

/// Registers `regs` after the interpreter runs `function` once.
pub(super) fn interp_registers(function: &Function, regs: &[u8]) -> Vec<u32> {
    let mut machine = Machine::new();
    machine.run(function).expect("the interpreter runs it");
    regs.iter()
        .map(|&reg| machine.regs[usize::from(reg)])
        .collect()
}
