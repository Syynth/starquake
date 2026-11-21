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
            // Kept as immediates too: the warp operations' operands whose being an immediate
            // picks the lowering (`warp`: a quad direction or index, a shuffle's lane and
            // segment mask).
            let warp_kept = matches!(
                (op.opcode, slot),
                (Opcode::QuadBroadcast | Opcode::QuadSwap, 1) | (Opcode::ShuffleIndex, 1 | 3)
            );
            // A constant-buffer read's bank names its binding and must stay an immediate
            // (`resources`); its offset is made opaque like any other operand.
            let cbuf_bank = slot == 0 && super::super::resources::is_cbuf_read(op.opcode);
            let kept = warp_kept
                || cbuf_bank
                || matches!(
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
/// return; registers the shader never names read as 0. With more than one invocation
/// (`lanes`), invocation `i` writes its registers from word `i * regs.len()` on, by its local
/// invocation index, an argument the patch adds.
fn patched(shader: &HostShader, regs: &[u8], lanes: u32) -> naga::Module {
    let mut module = naga_module(shader).clone();
    let count = u32::try_from(regs.len()).expect("few registers");
    let u32_ty = ty(&mut module, TypeInner::Scalar(naga::Scalar::U32));
    let array = ty(
        &mut module,
        TypeInner::Array {
            base: u32_ty,
            size: ArraySize::Constant(
                NonZeroU32::new(count * lanes).expect("a register and a lane"),
            ),
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
    // Invocation `i`'s first word, `i * count`, when there are several.
    let first_word = (lanes > 1).then(|| {
        let index = u32::try_from(function.arguments.len()).expect("few arguments");
        function.arguments.push(naga::FunctionArgument {
            name: Some("invocation_index".into()),
            ty: u32_ty,
            binding: Some(naga::Binding::BuiltIn(naga::BuiltIn::LocalInvocationIndex)),
        });
        let invocation = append(function, Expression::FunctionArgument(index));
        let stride = append(function, Expression::Literal(naga::Literal::U32(count)));
        let first = append(
            function,
            Expression::Binary {
                op: naga::BinaryOperator::Multiply,
                left: invocation,
                right: stride,
            },
        );
        stores.push(Statement::Emit(naga::Range::new_from_bounds(first, first)));
        first
    });
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
        let element = match first_word {
            None => append(function, Expression::AccessIndex { base, index: slot }),
            Some(first) => {
                let offset = append(function, Expression::Literal(naga::Literal::U32(slot)));
                let word = append(
                    function,
                    Expression::Binary {
                        op: naga::BinaryOperator::Add,
                        left: first,
                        right: offset,
                    },
                );
                stores.push(Statement::Emit(naga::Range::new_from_bounds(word, word)));
                append(function, Expression::Access { base, index: word })
            }
        };
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
    gpu_run(test, shader, regs, 1, &[])
}

/// [`gpu_registers`], with each `(binding, bytes)` of `uniforms` bound as a uniform buffer at
/// that binding of group 0 (a compute shader's constant buffers, `resources`).
pub(super) fn gpu_registers_with(
    test: &str,
    shader: &HostShader,
    regs: &[u8],
    uniforms: &[(u32, &[u8])],
) -> Option<Vec<u32>> {
    gpu_run(test, shader, regs, 1, uniforms)
}

/// Runs one workgroup of `shader` (a compute shader whose workgroup has `lanes` invocations
/// along x) on the GPU and returns each invocation's registers `regs`, by local invocation
/// index; `None` when the host cannot run it (a notice says why).
pub(super) fn gpu_lanes(
    test: &str,
    shader: &HostShader,
    regs: &[u8],
    lanes: u32,
) -> Option<Vec<Vec<u32>>> {
    assert_eq!(shader.workgroup_size, Some([lanes, 1, 1]));
    let words = gpu_run(test, shader, regs, lanes, &[])?;
    Some(words.chunks(regs.len()).map(<[u32]>::to_vec).collect())
}

/// [`gpu_lanes`], as one list of words.
fn gpu_run(
    test: &str,
    shader: &HostShader,
    regs: &[u8],
    lanes: u32,
    uniforms: &[(u32, &[u8])],
) -> Option<Vec<u32>> {
    let wanted = features(shader.requirements);
    let gpu = TestDevice::open(test, wanted)?;
    if !gpu.has(wanted) {
        eprintln!(
            "skipping the GPU half of {test}: {} lacks {wanted:?}",
            gpu.adapter.name
        );
        return None;
    }
    let module = patched(shader, regs, lanes);
    let size =
        u64::try_from(regs.len()).expect("few") * u64::from(lanes) * u64::from(MEMORY_WORD_BYTES);
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
        let buffers = uniform_buffers(device, &gpu.queue, uniforms);
        let mut entries = vec![wgpu::BindGroupEntry {
            binding: 0,
            resource: storage.as_entire_binding(),
        }];
        entries.extend(
            buffers
                .iter()
                .map(|(binding, buffer)| wgpu::BindGroupEntry {
                    binding: *binding,
                    resource: buffer.as_entire_binding(),
                }),
        );
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &entries,
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

/// One uniform buffer per `(binding, bytes)` of `uniforms`, holding the bytes, with its binding.
fn uniform_buffers(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    uniforms: &[(u32, &[u8])],
) -> Vec<(u32, wgpu::Buffer)> {
    uniforms
        .iter()
        .map(|&(binding, bytes)| {
            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("uniform"),
                size: u64::try_from(bytes.len()).expect("small"),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&buffer, 0, bytes);
            (binding, buffer)
        })
        .collect()
}

/// Registers `regs` after the interpreter runs `function` once.
pub(super) fn interp_registers(function: &Function, regs: &[u8]) -> Vec<u32> {
    let mut machine = Machine::new();
    machine.run(function).expect("the interpreter runs it");
    regs.iter()
        .map(|&reg| machine.regs[usize::from(reg)])
        .collect()
}
