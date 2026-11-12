//! Experiment E1: what naga 30.0.1 and wgpu 30.0.1 accept (`docs/specs/shader-pipeline.md`,
//! "What to verify before building", E1).
//!
//! Each case builds a small naga module by hand for one feature the translator will need, runs
//! it through naga's validator with the flags this crate uses (everything but the uniformity
//! check) and with the flags wgpu uses (everything: `wgpu-core` 30.0.1,
//! `device/resource.rs`, `create_shader_module` passes `naga::valid::ValidationFlags::all()` to
//! `wgpu_naga_bridge::create_validator`), then creates it on a wgpu device through
//! `ShaderSource::Naga` and builds a pipeline from it (wgpu's Metal backend writes and compiles
//! the MSL only then). The assertions are the observed results, so a naga or wgpu upgrade that
//! changes one fails here; the spec's E1 table records them. Run with
//! `cargo test -p starquake-shader e1_`.
//!
//! The GPU half of each case skips itself with a notice where the host has no adapter, and the
//! case's feature half where the adapter lacks the wgpu feature the case needs.

use std::num::NonZeroU32;

use naga::valid::{Capabilities, ValidationFlags, Validator};
use naga::{
    AddressSpace, ArraySize, AtomicFunction, BinaryOperator, Binding, Block, BuiltIn,
    DerivativeAxis, DerivativeControl, EntryPoint, Expression, Function, FunctionArgument,
    FunctionResult, GatherMode, GlobalVariable, Handle, ImageClass, ImageDimension, Interpolation,
    Literal, MeshOutputTopology, MeshStageInfo, Module, ResourceBinding, Sampling, Scalar,
    ScalarKind, ShaderStage, Span, Statement, StorageAccess, StructMember, Type, TypeInner,
    VectorSize,
};

use super::builder::{FunctionBuilder, named_ty, push, ty};
use crate::gpu_harness::TestDevice;

/// The flags this crate validates with: everything except the uniformity analysis, which guest
/// shaders fail (they take derivatives inside divergent control flow).
fn our_flags() -> ValidationFlags {
    ValidationFlags::all() - ValidationFlags::CONTROL_FLOW_UNIFORMITY
}

fn validate(module: &Module, flags: ValidationFlags, caps: Capabilities) -> Result<(), String> {
    Validator::new(flags, caps)
        .validate(module)
        .map(|_| ())
        .map_err(|error| format!("{error:?}"))
}

fn vec4f(module: &mut Module) -> Handle<Type> {
    ty(
        module,
        TypeInner::Vector {
            size: VectorSize::Quad,
            scalar: Scalar::F32,
        },
    )
}

fn scalar(module: &mut Module, scalar: Scalar) -> Handle<Type> {
    ty(module, TypeInner::Scalar(scalar))
}

/// A varying binding at `location` with the interpolation both stages of a pipeline use.
fn varying(location: u32) -> Binding {
    Binding::Location {
        location,
        interpolation: Some(Interpolation::Perspective),
        sampling: Some(Sampling::Center),
        blend_src: None,
        per_primitive: false,
    }
}

fn entry(name: &str, stage: ShaderStage, function: Function) -> EntryPoint {
    EntryPoint {
        name: name.to_owned(),
        stage,
        early_depth_test: None,
        workgroup_size: if stage == ShaderStage::Compute || stage == ShaderStage::Mesh {
            [1, 1, 1]
        } else {
            [0, 0, 0]
        },
        workgroup_size_overrides: None,
        function,
        mesh_info: None,
        task_payload: None,
        incoming_ray_payload: None,
    }
}

const VS_ENTRY: &str = "vs_main";
const FS_ENTRY: &str = "fs_main";
const CS_ENTRY: &str = "cs_main";

/// A vertex module that writes a fixed position and the varying the fragment cases read.
fn passthrough_vertex() -> Module {
    let mut module = Module::default();
    let vec4 = vec4f(&mut module);
    let out = named_ty(
        &mut module,
        "VsOut",
        TypeInner::Struct {
            members: vec![
                StructMember {
                    name: Some("position".into()),
                    ty: vec4,
                    binding: Some(Binding::BuiltIn(BuiltIn::Position { invariant: false })),
                    offset: 0,
                },
                StructMember {
                    name: Some("v".into()),
                    ty: vec4,
                    binding: Some(varying(0)),
                    offset: 16,
                },
            ],
            span: 32,
        },
    );
    let mut fb = FunctionBuilder::new(VS_ENTRY);
    fb.function.result = Some(FunctionResult {
        ty: out,
        binding: None,
    });
    let mut body = Block::new();
    let half = fb.expr(&mut body, Expression::Literal(Literal::F32(0.5)));
    let one = fb.expr(&mut body, Expression::Literal(Literal::F32(1.0)));
    let position = fb.expr(
        &mut body,
        Expression::Compose {
            ty: vec4,
            components: vec![half, half, half, one],
        },
    );
    let value = fb.expr(
        &mut body,
        Expression::Compose {
            ty: out,
            components: vec![position, position],
        },
    );
    push(&mut body, Statement::Return { value: Some(value) });
    module
        .entry_points
        .push(entry(VS_ENTRY, ShaderStage::Vertex, fb.finish(body)));
    module
}

/// A fragment module whose entry point takes the varying `v` and returns the colour `build`
/// computes from it.
fn fragment(
    build: impl FnOnce(
        &mut Module,
        &mut FunctionBuilder,
        &mut Block,
        Handle<Expression>,
    ) -> Handle<Expression>,
) -> Module {
    let mut module = Module::default();
    let vec4 = vec4f(&mut module);
    let mut fb = FunctionBuilder::new(FS_ENTRY);
    fb.function.arguments.push(FunctionArgument {
        name: Some("v".into()),
        ty: vec4,
        binding: Some(varying(0)),
    });
    fb.function.result = Some(FunctionResult {
        ty: vec4,
        binding: Some(Binding::Location {
            location: 0,
            interpolation: None,
            sampling: None,
            blend_src: None,
            per_primitive: false,
        }),
    });
    let mut body = Block::new();
    let v = fb.expr(&mut body, Expression::FunctionArgument(0));
    let colour = build(&mut module, &mut fb, &mut body, v);
    push(
        &mut body,
        Statement::Return {
            value: Some(colour),
        },
    );
    module
        .entry_points
        .push(entry(FS_ENTRY, ShaderStage::Fragment, fb.finish(body)));
    module
}

fn component(
    fb: &mut FunctionBuilder,
    block: &mut Block,
    vector: Handle<Expression>,
    index: u32,
) -> Handle<Expression> {
    fb.expr(
        block,
        Expression::AccessIndex {
            base: vector,
            index,
        },
    )
}

fn binary(
    fb: &mut FunctionBuilder,
    block: &mut Block,
    op: BinaryOperator,
    left: Handle<Expression>,
    right: Handle<Expression>,
) -> Handle<Expression> {
    fb.expr(block, Expression::Binary { op, left, right })
}

fn literal(fb: &mut FunctionBuilder, block: &mut Block, value: Literal) -> Handle<Expression> {
    fb.expr(block, Expression::Literal(value))
}

fn convert(
    fb: &mut FunctionBuilder,
    block: &mut Block,
    expr: Handle<Expression>,
    kind: ScalarKind,
    width: u8,
) -> Handle<Expression> {
    fb.expr(
        block,
        Expression::As {
            expr,
            kind,
            convert: Some(width),
        },
    )
}

fn splat(
    fb: &mut FunctionBuilder,
    block: &mut Block,
    value: Handle<Expression>,
) -> Handle<Expression> {
    fb.expr(
        block,
        Expression::Splat {
            size: VectorSize::Quad,
            value,
        },
    )
}

/// `Kill` inside two nested loops.
fn kill_in_nested_loops() -> Module {
    fragment(|module, fb, body, v| {
        let u32_ty = scalar(module, Scalar::U32);
        let counter = fb.local("i", u32_ty);
        let counter = fb.expr(body, Expression::LocalVariable(counter));

        let mut outer = Block::new();
        let count = fb.expr(&mut outer, Expression::Load { pointer: counter });
        let four = literal(fb, &mut outer, Literal::U32(4));
        let done = binary(fb, &mut outer, BinaryOperator::GreaterEqual, count, four);
        let mut leave = Block::new();
        push(&mut leave, Statement::Break);
        push(
            &mut outer,
            Statement::If {
                condition: done,
                accept: leave,
                reject: Block::new(),
            },
        );

        let mut inner = Block::new();
        let x = component(fb, &mut inner, v, 0);
        let half = literal(fb, &mut inner, Literal::F32(0.5));
        let discard = binary(fb, &mut inner, BinaryOperator::Greater, x, half);
        let mut kill = Block::new();
        push(&mut kill, Statement::Kill);
        push(
            &mut inner,
            Statement::If {
                condition: discard,
                accept: kill,
                reject: Block::new(),
            },
        );
        push(&mut inner, Statement::Break);
        push(
            &mut outer,
            Statement::Loop {
                body: inner,
                continuing: Block::new(),
                break_if: None,
            },
        );

        let count = fb.expr(&mut outer, Expression::Load { pointer: counter });
        let one = literal(fb, &mut outer, Literal::U32(1));
        let next = binary(fb, &mut outer, BinaryOperator::Add, count, one);
        push(
            &mut outer,
            Statement::Store {
                pointer: counter,
                value: next,
            },
        );
        push(
            body,
            Statement::Loop {
                body: outer,
                continuing: Block::new(),
                break_if: None,
            },
        );
        v
    })
}

/// f16 and i64 arithmetic.
fn f16_and_i64() -> Module {
    fragment(|_module, fb, body, v| {
        let x = component(fb, body, v, 0);
        let half_x = convert(fb, body, x, ScalarKind::Float, 2);
        let two = literal(fb, body, Literal::F32(2.0));
        let half_two = convert(fb, body, two, ScalarKind::Float, 2);
        let product = binary(fb, body, BinaryOperator::Multiply, half_x, half_two);
        let r = convert(fb, body, product, ScalarKind::Float, 4);

        let y = component(fb, body, v, 1);
        let int_y = convert(fb, body, y, ScalarKind::Sint, 4);
        let long_y = convert(fb, body, int_y, ScalarKind::Sint, 8);
        let three = literal(fb, body, Literal::I64(3));
        let long_product = binary(fb, body, BinaryOperator::Multiply, long_y, three);
        let g = convert(fb, body, long_product, ScalarKind::Float, 4);
        let sum = binary(fb, body, BinaryOperator::Add, r, g);
        splat(fb, body, sum)
    })
}

/// A 64-bit atomic maximum on a storage buffer, in a compute shader.
fn atomic_u64_max() -> Module {
    let mut module = Module::default();
    let atomic = ty(&mut module, TypeInner::Atomic(Scalar::U64));
    let buffer = named_ty(
        &mut module,
        "Buffer",
        TypeInner::Struct {
            members: vec![StructMember {
                name: Some("value".into()),
                ty: atomic,
                binding: None,
                offset: 0,
            }],
            span: 8,
        },
    );
    let global = module.global_variables.append(
        GlobalVariable {
            name: Some("buffer".into()),
            space: AddressSpace::Storage {
                access: StorageAccess::LOAD | StorageAccess::STORE,
            },
            binding: Some(ResourceBinding {
                group: 0,
                binding: 0,
            }),
            ty: buffer,
            init: None,
            memory_decorations: naga::MemoryDecorations::empty(),
        },
        Span::UNDEFINED,
    );
    let mut fb = FunctionBuilder::new(CS_ENTRY);
    let mut body = Block::new();
    let buffer = fb.expr(&mut body, Expression::GlobalVariable(global));
    let value = component(&mut fb, &mut body, buffer, 0);
    let five = literal(&mut fb, &mut body, Literal::U64(5));
    push(
        &mut body,
        Statement::Atomic {
            pointer: value,
            fun: AtomicFunction::Max,
            value: five,
            result: None,
        },
    );
    module
        .entry_points
        .push(entry(CS_ENTRY, ShaderStage::Compute, fb.finish(body)));
    module
}

/// Subgroup ballot and shuffle in a pixel shader.
fn subgroup_ballot_and_shuffle() -> Module {
    fragment(|module, fb, body, v| {
        let f32_ty = scalar(module, Scalar::F32);
        let x = component(fb, body, v, 0);
        let half = literal(fb, body, Literal::F32(0.5));
        let predicate = binary(fb, body, BinaryOperator::Greater, x, half);
        let ballot = fb.result(Expression::SubgroupBallotResult);
        push(
            body,
            Statement::SubgroupBallot {
                result: ballot,
                predicate: Some(predicate),
            },
        );
        let lane = literal(fb, body, Literal::U32(3));
        let shuffled = fb.result(Expression::SubgroupOperationResult { ty: f32_ty });
        push(
            body,
            Statement::SubgroupGather {
                mode: GatherMode::Shuffle(lane),
                argument: x,
                result: shuffled,
            },
        );
        let mask = component(fb, body, ballot, 0);
        let mask = convert(fb, body, mask, ScalarKind::Float, 4);
        let sum = binary(fb, body, BinaryOperator::Add, mask, shuffled);
        splat(fb, body, sum)
    })
}

/// How many textures the binding-array case binds.
const BINDING_ARRAY_LENGTH: u32 = 4;

/// A binding array of textures indexed with a non-uniform value.
fn binding_array_non_uniform() -> Module {
    fragment(|module, fb, body, v| {
        let texture = ty(
            module,
            TypeInner::Image {
                dim: ImageDimension::D2,
                arrayed: false,
                class: ImageClass::Sampled {
                    kind: ScalarKind::Float,
                    multi: false,
                },
            },
        );
        let array = ty(
            module,
            TypeInner::BindingArray {
                base: texture,
                size: ArraySize::Constant(
                    NonZeroU32::new(BINDING_ARRAY_LENGTH).expect("the length is not zero"),
                ),
            },
        );
        let global = module.global_variables.append(
            GlobalVariable {
                name: Some("textures".into()),
                space: AddressSpace::Handle,
                binding: Some(ResourceBinding {
                    group: 0,
                    binding: 0,
                }),
                ty: array,
                init: None,
                memory_decorations: naga::MemoryDecorations::empty(),
            },
            Span::UNDEFINED,
        );
        let vec2i = ty(
            module,
            TypeInner::Vector {
                size: VectorSize::Bi,
                scalar: Scalar::I32,
            },
        );
        let x = component(fb, body, v, 0);
        let three = literal(fb, body, Literal::F32(3.0));
        let scaled = binary(fb, body, BinaryOperator::Multiply, x, three);
        let index = convert(fb, body, scaled, ScalarKind::Uint, 4);
        let textures = fb.expr(body, Expression::GlobalVariable(global));
        let image = fb.expr(
            body,
            Expression::Access {
                base: textures,
                index,
            },
        );
        let coordinate = fb.expr(body, Expression::ZeroValue(vec2i));
        let level = literal(fb, body, Literal::I32(0));
        fb.expr(
            body,
            Expression::ImageLoad {
                image,
                coordinate,
                array_index: None,
                sample: None,
                level: Some(level),
            },
        )
    })
}

/// A compute shader with a workgroup barrier inside an `if` on the invocation id (divergent
/// control flow).
fn barrier_in_divergent_flow() -> Module {
    let mut module = Module::default();
    let vec3u = ty(
        &mut module,
        TypeInner::Vector {
            size: VectorSize::Tri,
            scalar: Scalar::U32,
        },
    );
    let mut fb = FunctionBuilder::new(CS_ENTRY);
    fb.function.arguments.push(FunctionArgument {
        name: Some("id".into()),
        ty: vec3u,
        binding: Some(Binding::BuiltIn(BuiltIn::LocalInvocationId)),
    });
    let mut body = Block::new();
    let id = fb.expr(&mut body, Expression::FunctionArgument(0));
    let x = component(&mut fb, &mut body, id, 0);
    let zero = literal(&mut fb, &mut body, Literal::U32(0));
    let first = binary(&mut fb, &mut body, BinaryOperator::Equal, x, zero);
    let mut accept = Block::new();
    push(
        &mut accept,
        Statement::ControlBarrier(naga::Barrier::WORK_GROUP),
    );
    push(
        &mut body,
        Statement::If {
            condition: first,
            accept,
            reject: Block::new(),
        },
    );
    let mut ep = entry(CS_ENTRY, ShaderStage::Compute, fb.finish(body));
    ep.workgroup_size = [2, 1, 1];
    module.entry_points.push(ep);
    module
}

/// A derivative inside an `if` on a varying (divergent control flow).
fn derivative_in_divergent_flow() -> Module {
    fragment(|module, fb, body, v| {
        let f32_ty = scalar(module, Scalar::F32);
        let result = fb.local("r", f32_ty);
        let result = fb.expr(body, Expression::LocalVariable(result));
        let x = component(fb, body, v, 0);
        let half = literal(fb, body, Literal::F32(0.5));
        let divergent = binary(fb, body, BinaryOperator::Greater, x, half);
        let mut accept = Block::new();
        let y = component(fb, &mut accept, v, 1);
        let derivative = fb.expr(
            &mut accept,
            Expression::Derivative {
                axis: DerivativeAxis::X,
                ctrl: DerivativeControl::None,
                expr: y,
            },
        );
        push(
            &mut accept,
            Statement::Store {
                pointer: result,
                value: derivative,
            },
        );
        push(
            body,
            Statement::If {
                condition: divergent,
                accept,
                reject: Block::new(),
            },
        );
        let value = fb.expr(body, Expression::Load { pointer: result });
        splat(fb, body, value)
    })
}

/// A mesh entry point that writes one triangle.
fn mesh_triangle() -> Module {
    let mut module = Module::default();
    let vec4 = vec4f(&mut module);
    let u32_ty = scalar(&mut module, Scalar::U32);
    let vec3u = ty(
        &mut module,
        TypeInner::Vector {
            size: VectorSize::Tri,
            scalar: Scalar::U32,
        },
    );
    let vertex = named_ty(
        &mut module,
        "MeshVertex",
        TypeInner::Struct {
            members: vec![StructMember {
                name: Some("position".into()),
                ty: vec4,
                binding: Some(Binding::BuiltIn(BuiltIn::Position { invariant: false })),
                offset: 0,
            }],
            span: 16,
        },
    );
    let primitive = named_ty(
        &mut module,
        "MeshPrimitive",
        TypeInner::Struct {
            members: vec![StructMember {
                name: Some("indices".into()),
                ty: vec3u,
                binding: Some(Binding::BuiltIn(BuiltIn::TriangleIndices)),
                offset: 0,
            }],
            span: 16,
        },
    );
    let vertices = ty(
        &mut module,
        TypeInner::Array {
            base: vertex,
            size: ArraySize::Constant(NonZeroU32::MIN.saturating_add(2)),
            stride: 16,
        },
    );
    let primitives = ty(
        &mut module,
        TypeInner::Array {
            base: primitive,
            size: ArraySize::Constant(NonZeroU32::MIN),
            stride: 16,
        },
    );
    let output = named_ty(
        &mut module,
        "MeshOutput",
        TypeInner::Struct {
            members: vec![
                StructMember {
                    name: Some("vertex_count".into()),
                    ty: u32_ty,
                    binding: Some(Binding::BuiltIn(BuiltIn::VertexCount)),
                    offset: 0,
                },
                StructMember {
                    name: Some("primitive_count".into()),
                    ty: u32_ty,
                    binding: Some(Binding::BuiltIn(BuiltIn::PrimitiveCount)),
                    offset: 4,
                },
                StructMember {
                    name: Some("vertices".into()),
                    ty: vertices,
                    binding: Some(Binding::BuiltIn(BuiltIn::Vertices)),
                    offset: 16,
                },
                StructMember {
                    name: Some("primitives".into()),
                    ty: primitives,
                    binding: Some(Binding::BuiltIn(BuiltIn::Primitives)),
                    offset: 64,
                },
            ],
            span: 80,
        },
    );
    let global = module.global_variables.append(
        GlobalVariable {
            name: Some("mesh_output".into()),
            space: AddressSpace::WorkGroup,
            binding: None,
            ty: output,
            init: None,
            memory_decorations: naga::MemoryDecorations::empty(),
        },
        Span::UNDEFINED,
    );
    let mut fb = FunctionBuilder::new("ms_main");
    let mut body = Block::new();
    let out = fb.expr(&mut body, Expression::GlobalVariable(global));
    let vertex_count = component(&mut fb, &mut body, out, 0);
    let three = literal(&mut fb, &mut body, Literal::U32(3));
    push(
        &mut body,
        Statement::Store {
            pointer: vertex_count,
            value: three,
        },
    );
    let primitive_count = component(&mut fb, &mut body, out, 1);
    let one = literal(&mut fb, &mut body, Literal::U32(1));
    push(
        &mut body,
        Statement::Store {
            pointer: primitive_count,
            value: one,
        },
    );
    let mut ep = entry("ms_main", ShaderStage::Mesh, fb.finish(body));
    ep.mesh_info = Some(MeshStageInfo {
        topology: MeshOutputTopology::Triangles,
        max_vertices: 3,
        max_vertices_override: None,
        max_primitives: 1,
        max_primitives_override: None,
        vertex_output_type: vertex,
        primitive_output_type: primitive,
        output_variable: global,
    });
    module.entry_points.push(ep);
    module
}

/// Creates `module` (a fragment entry point) on `gpu` and a render pipeline from it.
fn on_gpu_fragment(gpu: &TestDevice, module: Module) -> Result<(), String> {
    on_gpu_fragment_with(gpu, module, None)
}

/// Like [`on_gpu_fragment`], with an explicit pipeline layout.
fn on_gpu_fragment_with(
    gpu: &TestDevice,
    module: Module,
    layout: Option<&wgpu::PipelineLayout>,
) -> Result<(), String> {
    let vertex = gpu.module(passthrough_vertex())?;
    let fragment = gpu.module(module)?;
    gpu.render_pipeline(
        layout,
        (&vertex, VS_ENTRY),
        (&fragment, FS_ENTRY),
        &[],
        &[Some(wgpu::ColorTargetState {
            format: wgpu::TextureFormat::Rgba8Unorm,
            blend: None,
            write_mask: wgpu::ColorWrites::ALL,
        })],
    )
}

/// Asserts the naga half of a case: accepted with this crate's flags and with wgpu's.
fn assert_valid_both_ways(module: &Module, caps: Capabilities) {
    assert_eq!(validate(module, our_flags(), caps), Ok(()));
    assert_eq!(validate(module, ValidationFlags::all(), caps), Ok(()));
}

/// Runs the GPU half of a case where the adapter has `features`; returns `None` when skipped.
fn on_gpu<T>(
    test: &str,
    features: wgpu::Features,
    run: impl FnOnce(&TestDevice) -> T,
) -> Option<T> {
    let gpu = TestDevice::open(test, features)?;
    if !gpu.has(features) {
        eprintln!(
            "skipping the wgpu half of {test}: {} lacks {features:?}",
            gpu.adapter.name
        );
        return None;
    }
    Some(run(&gpu))
}

#[test]
fn e1_kill_in_nested_loops() {
    let module = kill_in_nested_loops();
    assert_valid_both_ways(&module, Capabilities::empty());
    if let Some(result) = on_gpu("e1_kill_in_nested_loops", wgpu::Features::empty(), |gpu| {
        on_gpu_fragment(gpu, module)
    }) {
        assert_eq!(result, Ok(()));
    }
}

#[test]
fn e1_f16_and_i64() {
    let module = f16_and_i64();
    let caps = Capabilities::SHADER_FLOAT16 | Capabilities::SHADER_INT64;
    assert_valid_both_ways(&module, caps);
    let features = wgpu::Features::SHADER_F16 | wgpu::Features::SHADER_INT64;
    if let Some(result) = on_gpu("e1_f16_and_i64", features, |gpu| {
        on_gpu_fragment(gpu, module)
    }) {
        assert_eq!(result, Ok(()));
    }
}

#[test]
fn e1_atomic_u64_max() {
    let module = atomic_u64_max();
    let caps = Capabilities::SHADER_INT64 | Capabilities::SHADER_INT64_ATOMIC_MIN_MAX;
    assert_valid_both_ways(&module, caps);
    let features = wgpu::Features::SHADER_INT64 | wgpu::Features::SHADER_INT64_ATOMIC_MIN_MAX;
    if let Some(result) = on_gpu("e1_atomic_u64_max", features, |gpu| {
        let shader = gpu.module(module)?;
        gpu.compute_pipeline(&shader, CS_ENTRY)
    }) {
        assert_eq!(result, Ok(()));
    }
}

#[test]
fn e1_subgroup_ballot_and_shuffle_in_a_pixel_shader() {
    let module = subgroup_ballot_and_shuffle();
    assert_valid_both_ways(&module, Capabilities::SUBGROUP);
    if let Some(result) = on_gpu("e1_subgroup", wgpu::Features::SUBGROUP, |gpu| {
        on_gpu_fragment(gpu, module)
    }) {
        assert_eq!(result, Ok(()));
    }
}

#[test]
fn e1_binding_array_with_non_uniform_index() {
    let module = binding_array_non_uniform();
    let caps = Capabilities::TEXTURE_AND_SAMPLER_BINDING_ARRAY
        | Capabilities::TEXTURE_AND_SAMPLER_BINDING_ARRAY_NON_UNIFORM_INDEXING;
    assert_valid_both_ways(&module, caps);
    let features = wgpu::Features::TEXTURE_BINDING_ARRAY
        | wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING;
    if let Some((derived, explicit)) = on_gpu("e1_binding_array", features, |gpu| {
        // A layout derived from the shader does not work: wgpu's Metal backend finds no slot
        // for the array.
        let derived = on_gpu_fragment(gpu, module.clone());
        let layout = gpu
            .layout(&[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: NonZeroU32::new(BINDING_ARRAY_LENGTH),
            }])
            .expect("the layout is valid");
        (derived, on_gpu_fragment_with(gpu, module, Some(&layout)))
    }) {
        assert!(
            derived
                .as_ref()
                .is_err_and(|error| error.contains("mapping of ResourceBinding")),
            "{derived:?}"
        );
        assert_eq!(explicit, Ok(()));
    }
}

/// naga 30.0.1 has no built-in for the layer (or viewport index) a vertex shader writes, so the
/// module cannot be built. This match lists every `BuiltIn` variant with no wildcard: it stops
/// compiling when a naga upgrade adds one, which is the moment to check for `Layer`.
#[test]
fn e1_no_layer_built_in_for_a_vertex_shader() {
    fn vertex_output_for_layer(built_in: BuiltIn) -> bool {
        match built_in {
            BuiltIn::PrimitiveIndex
            | BuiltIn::Position { .. }
            | BuiltIn::ViewIndex
            | BuiltIn::BaseInstance
            | BuiltIn::BaseVertex
            | BuiltIn::ClipDistances
            | BuiltIn::CullDistance
            | BuiltIn::InstanceIndex
            | BuiltIn::PointSize
            | BuiltIn::VertexIndex
            | BuiltIn::DrawIndex
            | BuiltIn::FragDepth
            | BuiltIn::PointCoord
            | BuiltIn::FrontFacing
            | BuiltIn::Barycentric { .. }
            | BuiltIn::SampleIndex
            | BuiltIn::SampleMask
            | BuiltIn::GlobalInvocationId
            | BuiltIn::LocalInvocationId
            | BuiltIn::LocalInvocationIndex
            | BuiltIn::WorkGroupId
            | BuiltIn::WorkGroupSize
            | BuiltIn::NumWorkGroups
            | BuiltIn::NumSubgroups
            | BuiltIn::SubgroupId
            | BuiltIn::SubgroupSize
            | BuiltIn::SubgroupInvocationId
            | BuiltIn::MeshTaskSize
            | BuiltIn::CullPrimitive
            | BuiltIn::PointIndex
            | BuiltIn::LineIndices
            | BuiltIn::TriangleIndices
            | BuiltIn::VertexCount
            | BuiltIn::Vertices
            | BuiltIn::PrimitiveCount
            | BuiltIn::Primitives
            | BuiltIn::RayInvocationId
            | BuiltIn::NumRayInvocations
            | BuiltIn::InstanceCustomData
            | BuiltIn::GeometryIndex
            | BuiltIn::WorldRayOrigin
            | BuiltIn::WorldRayDirection
            | BuiltIn::ObjectRayOrigin
            | BuiltIn::ObjectRayDirection
            | BuiltIn::RayTmin
            | BuiltIn::RayTCurrentMax
            | BuiltIn::ObjectToWorld
            | BuiltIn::WorldToObject
            | BuiltIn::HitKind => false,
        }
    }
    // `ViewIndex` is the multiview index, an input, not the layer a vertex shader selects.
    assert!(!vertex_output_for_layer(BuiltIn::ViewIndex));
}

/// The open question of the spec's "The backend through naga": wgpu validates a module it is
/// given with every flag, uniformity included. It still accepts a derivative in divergent
/// control flow, because naga 30.0.1 gives derivatives no uniformity requirement at all:
/// `naga/src/valid/analyzer.rs` sets `DISABLE_UNIFORMITY_REQ_FOR_FRAGMENT_STAGE = true`, which
/// makes `UniformityRequirements::DERIVATIVE` and `IMPLICIT_LEVEL` zero. No filter is needed.
#[test]
fn e1_derivative_in_divergent_control_flow() {
    let module = derivative_in_divergent_flow();
    assert_valid_both_ways(&module, Capabilities::empty());
    if let Some(result) = on_gpu(
        "e1_derivative_in_divergent_control_flow",
        wgpu::Features::empty(),
        |gpu| on_gpu_fragment(gpu, module),
    ) {
        assert_eq!(result, Ok(()));
    }
}

/// A workgroup barrier in divergent control flow (a guest compute shader may have one where
/// naga cannot prove the condition uniform) is accepted too: naga 30.0.1 records
/// `WORK_GROUP_BARRIER` as a requirement of the statement but reports `NonUniformControlFlow`
/// only for expressions in an `Emit` (`process_block`, the `S::Emit` arm), which leaves
/// cooperative-matrix operations (`COOP_OPS`) as the only thing the check rejects.
#[test]
fn e1_barrier_in_divergent_control_flow() {
    let module = barrier_in_divergent_flow();
    assert_valid_both_ways(&module, Capabilities::empty());
    if let Some(result) = on_gpu(
        "e1_barrier_in_divergent_control_flow",
        wgpu::Features::empty(),
        |gpu| {
            let shader = gpu.module(module)?;
            gpu.compute_pipeline(&shader, CS_ENTRY)
        },
    ) {
        assert_eq!(result, Ok(()));
    }
}

/// The disputed fact of review guide M12: naga writes a mesh entry point as MSL.
#[test]
fn e1_mesh_entry_point_written_as_msl() {
    let module = mesh_triangle();
    let caps = Capabilities::MESH_SHADER;
    let info = Validator::new(our_flags(), caps)
        .validate(&module)
        .expect("the mesh module validates");
    assert_eq!(validate(&module, ValidationFlags::all(), caps), Ok(()));
    let options = naga::back::msl::Options {
        lang_version: (3, 0),
        ..naga::back::msl::Options::default()
    };
    let (msl, _) = naga::back::msl::write_string(
        &module,
        &info,
        &options,
        &naga::back::msl::PipelineOptions::default(),
    )
    .expect("naga writes the mesh entry point as MSL 3.0");
    assert!(msl.contains("[[mesh]]"), "{msl}");

    // MSL before 3.0 has no mesh shaders, and naga says so.
    let old = naga::back::msl::Options {
        lang_version: (2, 4),
        ..naga::back::msl::Options::default()
    };
    assert!(matches!(
        naga::back::msl::write_string(
            &module,
            &info,
            &old,
            &naga::back::msl::PipelineOptions::default()
        ),
        Err(naga::back::msl::Error::UnsupportedMeshShader)
    ));
}
