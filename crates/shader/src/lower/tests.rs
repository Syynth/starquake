//! Lowering tests: the synthetic vertex/pixel pair end to end (golden tables, validation with
//! this crate's flags and wgpu's, acceptance on a wgpu device), and each lowering error.

use naga::valid::{ValidationFlags, Validator};

use super::*;
use crate::asm::assemble_program;
use crate::decode::{
    FloatSrc, Gpr, Instr, MovSrc, Op, OpALd, OpASt, OpExit, OpFFma, OpMov, OpS2R, PredSrc,
    RoundMode, SpecialReg,
};
use crate::gpu_harness::TestDevice;
use crate::ir::FpControl;
use crate::program::{Rejection, translate_program};
use crate::translate::{ColorOutputs, Stage};

fn r(index: u8) -> Gpr {
    Gpr::new(index)
}

const VERTEX: Stage = Stage {
    stage: GuestStage::Vertex,
    color_outputs: ColorOutputs {
        targets: [0; COLOR_TARGETS],
    },
};

/// The pixel program writes all four components of colour target 0.
fn pixel_stage() -> Stage {
    let mut targets = [0; COLOR_TARGETS];
    targets[0] = 0xf;
    Stage {
        stage: GuestStage::Pixel,
        color_outputs: ColorOutputs { targets },
    }
}

fn exit() -> Instr {
    Instr::new(Op::Exit(OpExit {
        keep_ref_count: false,
        no_at_exit: false,
        pred: PredSrc::TRUE,
    }))
}

fn ald(dst: Gpr, addr: u16, comps: u8) -> Instr {
    Instr::new(Op::ALd(OpALd {
        dst,
        vtx: Gpr::RZ,
        offset: Gpr::RZ,
        addr,
        comps,
        patch: false,
        phys: false,
        output: false,
    }))
}

fn ast(data: Gpr, addr: u16, comps: u8) -> Instr {
    Instr::new(Op::ASt(OpASt {
        data,
        vtx: Gpr::RZ,
        offset: Gpr::RZ,
        addr,
        comps,
        patch: false,
        phys: false,
    }))
}

fn mov(dst: Gpr, src: MovSrc) -> Instr {
    Instr::new(Op::Mov(OpMov {
        dst,
        src,
        quad_lanes: OpMov::ALL_LANES,
    }))
}

fn ffma(dst: Gpr, srcs: [FloatSrc; 3]) -> Instr {
    Instr::new(Op::FFma(OpFFma {
        dst,
        srcs,
        saturate: false,
        rnd: RoundMode::NearestEven,
        ftz: false,
        dnz: false,
    }))
}

fn s2r_lane(dst: Gpr) -> Instr {
    Instr::new(Op::S2R(OpS2R {
        dst,
        sr: SpecialReg::LANE_ID,
    }))
}

const ONE: u32 = 0x3f80_0000;

/// The synthetic vertex program: reads generic 0 (four words) and generic 1 (two), writes the
/// position and generic 0's X.
fn vertex_program() -> Vec<u8> {
    assemble_program(&[
        ald(r(0), 0x80, 4),
        ald(r(4), 0x90, 2),
        ffma(
            r(0),
            [
                FloatSrc::reg(r(0)),
                FloatSrc::Imm(0.5_f32.to_bits()),
                FloatSrc::reg(r(4)),
            ],
        ),
        mov(r(6), MovSrc::Imm(ONE)),
        ast(r(0), 0x70, 4),
        ast(r(6), 0x80, 1),
        exit(),
    ])
    .expect("encodable")
}

/// The synthetic pixel program: colour target 0 from the lane id and constants.
fn pixel_program() -> Vec<u8> {
    assemble_program(&[
        s2r_lane(r(4)),
        mov(r(0), MovSrc::Imm(ONE)),
        ffma(
            r(1),
            [
                FloatSrc::reg(r(4)),
                FloatSrc::Imm(2.0_f32.to_bits()),
                FloatSrc::reg(r(0)),
            ],
        ),
        mov(r(2), MovSrc::Reg(Gpr::RZ)),
        mov(r(3), MovSrc::Reg(r(0))),
        exit(),
    ])
    .expect("encodable")
}

fn pair() -> (HostShader, HostShader) {
    let vertex = translate_program(VERTEX, &vertex_program()).expect("the vertex program");
    let pixel = translate_program(pixel_stage(), &pixel_program()).expect("the pixel program");
    (vertex, pixel)
}

fn naga_module(shader: &HostShader) -> &naga::Module {
    let HostCode::Naga { module, .. } = &shader.code;
    module
}

fn slot(binding: SlotBinding, ty: SlotType, guest: GuestSlot) -> InterfaceSlot {
    InterfaceSlot { binding, ty, guest }
}

#[test]
fn pair_golden_tables() {
    let (vertex, pixel) = pair();
    assert_eq!(vertex.stage, HostStage::Vertex);
    assert_eq!(vertex.entry, ENTRY_POINT);
    assert_eq!(
        vertex.inputs,
        [
            slot(
                SlotBinding::Location(0),
                SlotType::F32x4,
                GuestSlot::Generic(0)
            ),
            slot(
                SlotBinding::Location(1),
                SlotType::F32x4,
                GuestSlot::Generic(1)
            ),
        ]
    );
    assert_eq!(
        vertex.outputs,
        [
            slot(
                SlotBinding::BuiltIn(HostBuiltIn::Position),
                SlotType::F32x4,
                GuestSlot::Position
            ),
            slot(
                SlotBinding::Location(0),
                SlotType::F32x4,
                GuestSlot::Generic(0)
            ),
        ]
    );
    assert_eq!(vertex.requirements, Requirements::default());

    assert_eq!(pixel.stage, HostStage::Fragment);
    assert_eq!(pixel.entry, ENTRY_POINT);
    assert_eq!(
        pixel.inputs,
        [slot(
            SlotBinding::BuiltIn(HostBuiltIn::SubgroupInvocationId),
            SlotType::U32,
            GuestSlot::LaneId
        )]
    );
    assert_eq!(
        pixel.outputs,
        [slot(
            SlotBinding::Location(0),
            SlotType::F32x4,
            GuestSlot::RenderTarget(0)
        )]
    );
    assert_eq!(pixel.requirements, Requirements { subgroup: true });
}

/// The tables describe the module: each argument and output member has the binding its slot
/// names, in the same order.
#[test]
fn tables_match_the_module() {
    let (vertex, pixel) = pair();
    for shader in [&vertex, &pixel] {
        let module = naga_module(shader);
        assert_eq!(module.entry_points.len(), 1);
        let entry = &module.entry_points[0];
        assert_eq!(entry.name, shader.entry);
        let bindings = |slots: &[InterfaceSlot]| -> Vec<Binding> {
            slots
                .iter()
                .map(|slot| match slot.binding {
                    SlotBinding::Location(n) => {
                        let varying =
                            shader.stage == HostStage::Vertex && slots == shader.outputs.as_slice();
                        location(u8::try_from(n).expect("a small location"), varying)
                    }
                    SlotBinding::BuiltIn(HostBuiltIn::Position) => {
                        Binding::BuiltIn(BuiltIn::Position { invariant: false })
                    }
                    SlotBinding::BuiltIn(HostBuiltIn::SubgroupInvocationId) => {
                        Binding::BuiltIn(BuiltIn::SubgroupInvocationId)
                    }
                })
                .collect()
        };
        let arguments: Vec<Binding> = entry
            .function
            .arguments
            .iter()
            .filter_map(|arg| arg.binding.clone())
            .collect();
        assert_eq!(arguments, bindings(&shader.inputs));
        let result = entry.function.result.as_ref().expect("a result");
        let members = match &module.types[result.ty].inner {
            TypeInner::Struct { members, .. } => Some(members),
            _ => None,
        }
        .expect("the result is a struct");
        let members: Vec<Binding> = members.iter().filter_map(|m| m.binding.clone()).collect();
        assert_eq!(members, bindings(&shader.outputs));
    }
}

/// Every module validates with this crate's flags (when lowered) and with wgpu's (every flag,
/// experiment E1), with exactly the capabilities its requirements name.
#[test]
fn pair_validates_with_wgpus_flags() {
    let (vertex, pixel) = pair();
    for shader in [&vertex, &pixel] {
        let caps = capabilities(shader.requirements);
        Validator::new(ValidationFlags::all(), caps)
            .validate(naga_module(shader))
            .expect("valid with every flag");
    }
    // The lane id needs the subgroup capability: without it, naga refuses the pixel module.
    assert!(
        Validator::new(validation_flags(), Capabilities::empty())
            .validate(naga_module(&pixel))
            .is_err()
    );
}

/// The pair is accepted by wgpu (locally on a Mac; skipped with a notice where no GPU adapter
/// exists): both modules through `ShaderSource::Naga`, then a render pipeline whose vertex
/// buffer and colour targets come from the tables.
#[test]
fn pair_is_accepted_by_wgpu() {
    let (vertex, pixel) = pair();
    let Some(gpu) = TestDevice::open("pair_is_accepted_by_wgpu", wgpu::Features::SUBGROUP) else {
        return;
    };
    if !gpu.has(wgpu::Features::SUBGROUP) {
        eprintln!(
            "skipping pair_is_accepted_by_wgpu: {} lacks subgroups, which the pixel shader \
             requires",
            gpu.adapter.name
        );
        return;
    }
    let attributes: Vec<wgpu::VertexAttribute> = (0_u64..)
        .zip(&vertex.inputs)
        .filter_map(|(index, slot)| match slot.binding {
            SlotBinding::Location(location) => Some(wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x4,
                offset: index * u64::from(VEC4_BYTES),
                shader_location: location,
            }),
            SlotBinding::BuiltIn(_) => None,
        })
        .collect();
    let stride = u64::try_from(attributes.len()).expect("small") * u64::from(VEC4_BYTES);
    let buffers = [Some(wgpu::VertexBufferLayout {
        array_stride: stride,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &attributes,
    })];
    let mut targets = Vec::new();
    for slot in &pixel.outputs {
        if let SlotBinding::Location(location) = slot.binding {
            let index = usize::try_from(location).expect("small");
            targets.resize(targets.len().max(index + 1), None);
            targets[index] = Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba8Unorm,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            });
        }
    }
    let module = |shader: &HostShader| gpu.module(naga_module(shader).clone());
    let vs = module(&vertex).expect("wgpu takes the vertex module");
    let fs = module(&pixel).expect("wgpu takes the pixel module");
    assert_eq!(
        gpu.render_pipeline(
            None,
            (&vs, &vertex.entry),
            (&fs, &pixel.entry),
            &buffers,
            &targets
        ),
        Ok(())
    );
}

/// Attribute words 156 to 159 are generic 31's X to W here, all refused because generic 31 has
/// no host location; Eden's `IsGeneric` stops at 156 (the divergence in this module's docs), so
/// 157 to 159 would not be generics there. Word 160 is past the generics. Read and write alike.
#[test]
fn generic_31_w_is_a_generic() {
    let last_generic = u8::try_from(GENERICS - 1).expect("small");
    assert_eq!(last_generic, HOST_LOCATIONS);
    let generic_31_x = GENERIC_0_X + (GENERICS - 1) * COMPONENTS;
    let last = GENERIC_0_X + GENERICS * COMPONENTS - 1;
    let refused = Some(Rejection::Lower(LowerError::GenericLocation {
        generic: last_generic,
    }));
    for word in generic_31_x..=last {
        let read = assemble_program(&[ald(r(0), word * 4, 1), exit()]).expect("encodable");
        assert_eq!(translate_program(VERTEX, &read).err(), refused, "{word}");
        let write = assemble_program(&[ast(r(0), word * 4, 1), exit()]).expect("encodable");
        assert_eq!(translate_program(VERTEX, &write).err(), refused, "{word}");
    }
    let past = (last + 1) * 4;
    let program = assemble_program(&[ald(r(0), past, 1), exit()]).expect("encodable");
    assert_eq!(
        translate_program(VERTEX, &program).err(),
        Some(Rejection::Lower(LowerError::Input {
            stage: GuestStage::Vertex,
            attribute: Attribute(last + 1)
        }))
    );
}

/// The vertex program that reads and writes the last generic with a host location, W only.
fn last_location_program() -> Vec<u8> {
    let last = GENERIC_0_X + u16::from(HOST_LOCATIONS) * COMPONENTS - 1;
    assemble_program(&[ald(r(0), last * 4, 1), ast(r(0), last * 4, 1), exit()]).expect("encodable")
}

/// The last generic below [`HOST_LOCATIONS`] lowers to that location in and out, and wgpu builds
/// a pipeline from it (locally on a Mac; skipped with a notice where no GPU adapter exists).
#[test]
fn the_last_host_location_builds_a_pipeline() {
    let vertex = translate_program(VERTEX, &last_location_program()).expect("generic 30");
    let generic = HOST_LOCATIONS - 1;
    let binding = SlotBinding::Location(u32::from(generic));
    assert_eq!(
        vertex.inputs,
        [slot(binding, SlotType::F32x4, GuestSlot::Generic(generic))]
    );
    assert_eq!(
        vertex.outputs.last(),
        Some(&slot(binding, SlotType::F32x4, GuestSlot::Generic(generic)))
    );
    // A pixel program that writes colour target 0's X from an immediate (no subgroups needed).
    let mut targets = [0; COLOR_TARGETS];
    targets[0] = 0x1;
    let pixel = translate_program(
        Stage {
            stage: GuestStage::Pixel,
            color_outputs: ColorOutputs { targets },
        },
        &assemble_program(&[mov(r(0), MovSrc::Imm(ONE)), exit()]).expect("encodable"),
    )
    .expect("the pixel program");
    let Some(gpu) = TestDevice::open(
        "the_last_host_location_builds_a_pipeline",
        wgpu::Features::empty(),
    ) else {
        return;
    };
    let attributes = [wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x4,
        offset: 0,
        shader_location: u32::from(generic),
    }];
    let buffers = [Some(wgpu::VertexBufferLayout {
        array_stride: u64::from(VEC4_BYTES),
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &attributes,
    })];
    let targets = [Some(wgpu::ColorTargetState {
        format: wgpu::TextureFormat::Rgba8Unorm,
        blend: None,
        write_mask: wgpu::ColorWrites::ALL,
    })];
    let vs = gpu
        .module(naga_module(&vertex).clone())
        .expect("wgpu takes the vertex module");
    let fs = gpu
        .module(naga_module(&pixel).clone())
        .expect("wgpu takes the pixel module");
    assert_eq!(
        gpu.render_pipeline(
            None,
            (&vs, &vertex.entry),
            (&fs, &pixel.entry),
            &buffers,
            &targets
        ),
        Ok(())
    );
}

/// The position is words 28 to 31; word 27 (the point size) and the layer (word 25,
/// experiment E1: naga has no layer output) are not outputs a vertex program can write yet.
#[test]
fn vertex_outputs_are_the_position_and_generics() {
    for word in [POSITION_X, POSITION_X + COMPONENTS - 1] {
        let program = assemble_program(&[ast(Gpr::RZ, word * 4, 1), exit()]).expect("encodable");
        let shader = translate_program(VERTEX, &program).expect("the position");
        assert_eq!(shader.outputs.len(), 1);
    }
    for word in [POSITION_X - 1, 25] {
        let program = assemble_program(&[ast(Gpr::RZ, word * 4, 1), exit()]).expect("encodable");
        assert_eq!(
            translate_program(VERTEX, &program).err(),
            Some(Rejection::Lower(LowerError::Output {
                stage: GuestStage::Vertex,
                attribute: Attribute(word)
            }))
        );
    }
    // A vertex program cannot read the position.
    let program = assemble_program(&[ald(r(0), POSITION_X * 4, 1), exit()]).expect("encodable");
    assert_eq!(
        translate_program(VERTEX, &program).err(),
        Some(Rejection::Lower(LowerError::Input {
            stage: GuestStage::Vertex,
            attribute: Attribute(POSITION_X)
        }))
    );
}

#[test]
fn pixel_programs_have_no_attribute_access_yet() {
    let generic = GENERIC_0_X * 4;
    let read = assemble_program(&[ald(r(0), generic, 1), exit()]).expect("encodable");
    assert_eq!(
        translate_program(pixel_stage(), &read).err(),
        Some(Rejection::Lower(LowerError::Input {
            stage: GuestStage::Pixel,
            attribute: Attribute(GENERIC_0_X)
        }))
    );
    let write = assemble_program(&[ast(r(0), generic, 1), exit()]).expect("encodable");
    assert_eq!(
        translate_program(pixel_stage(), &write).err(),
        Some(Rejection::Lower(LowerError::Output {
            stage: GuestStage::Pixel,
            attribute: Attribute(GENERIC_0_X)
        }))
    );
}

#[test]
fn the_lane_id_cannot_be_read_in_a_vertex_program() {
    let program = assemble_program(&[s2r_lane(r(0)), exit()]).expect("encodable");
    let rejection = translate_program(VERTEX, &program).err();
    assert_eq!(
        rejection,
        Some(Rejection::Lower(LowerError::LaneIdInVertex))
    );
    assert_eq!(
        rejection.map(|error| error.to_string()),
        Some("the lane id cannot be read in a vertex program".into())
    );
}

/// A pixel program that writes no colour has no result.
#[test]
fn a_pixel_program_with_no_outputs_returns_nothing() {
    let shader = translate_program(
        Stage {
            stage: GuestStage::Pixel,
            color_outputs: ColorOutputs::default(),
        },
        &assemble_program(&[exit()]).expect("encodable"),
    )
    .expect("lowers");
    assert_eq!(shader.outputs, []);
    assert!(
        naga_module(&shader).entry_points[0]
            .function
            .result
            .is_none()
    );
}

/// Errors only hand-built IR can reach.
#[test]
fn malformed_or_misplaced_ir_is_refused() {
    let colour = |target, component| {
        let mut f = Function::new();
        let b = f.entry();
        let value = f
            .push(b, Inst::GetAttribute(Attribute(GENERIC_0_X)))
            .expect("room");
        f.push(
            b,
            Inst::SetFragColor {
                target,
                component,
                value,
            },
        )
        .expect("room");
        f
    };
    // `GetAttribute` is refused in a pixel program before the colour output is looked at, so
    // build the colour cases from an immediate.
    let colour_from_imm = |target, component| {
        let mut f = Function::new();
        let b = f.entry();
        let bits = f.push(b, Inst::Imm32(0)).expect("room");
        let value = f.push(b, Inst::BitCastF32(bits)).expect("room");
        f.push(
            b,
            Inst::SetFragColor {
                target,
                component,
                value,
            },
        )
        .expect("room");
        f
    };
    let last_target = u8::try_from(COLOR_TARGETS - 1).expect("small");
    assert!(lower(GuestStage::Pixel, &colour_from_imm(last_target, 3)).is_ok());
    assert_eq!(
        lower(GuestStage::Pixel, &colour_from_imm(last_target + 1, 0)).err(),
        Some(LowerError::ColorOutput {
            target: last_target + 1,
            component: 0
        })
    );
    assert_eq!(
        lower(GuestStage::Pixel, &colour_from_imm(0, 4)).err(),
        Some(LowerError::ColorOutput {
            target: 0,
            component: 4
        })
    );
    assert_eq!(
        lower(GuestStage::Vertex, &colour(0, 0)).err(),
        Some(LowerError::ColorInStage(GuestStage::Vertex))
    );
    assert_eq!(
        lower(GuestStage::Geometry, &Function::new()).err(),
        Some(LowerError::Stage(GuestStage::Geometry))
    );
    let mut f = Function::new();
    let b = f.entry();
    let lane = f.push(b, Inst::LaneId).expect("room");
    f.push(
        b,
        Inst::FFma(
            lane,
            lane,
            lane,
            FpControl {
                rounding: RoundMode::NearestEven,
                ftz: false,
                fmz: false,
            },
        ),
    )
    .expect("room");
    assert!(matches!(
        lower(GuestStage::Pixel, &f),
        Err(LowerError::Verify(VerifyError::Type { .. }))
    ));
}

/// Prints expression `handle` of `function` as a compact tree: locals and arguments by name, a
/// load as the variable it reads, `f32(..)`/`u32(..)` for bit casts, `-x` for negation, naga's
/// math function names, `base[index]` for an access by constant index. Anything else prints as
/// "unexpected ...", which no golden line contains.
fn print_expr(function: &naga::Function, handle: Handle<Expression>) -> String {
    let print = |handle| print_expr(function, handle);
    match function.expressions[handle] {
        Expression::Literal(Literal::U32(bits)) => format!("{bits:#x}"),
        Expression::LocalVariable(local) => {
            function.local_variables[local].name.clone().expect("named")
        }
        Expression::FunctionArgument(index) => function.arguments
            [usize::try_from(index).expect("small")]
        .name
        .clone()
        .expect("named"),
        Expression::Load { pointer } => print(pointer),
        Expression::As { expr, kind, .. } => match kind {
            ScalarKind::Float => format!("f32({})", print(expr)),
            ScalarKind::Uint => format!("u32({})", print(expr)),
            other => format!("unexpected cast to {other:?}"),
        },
        Expression::Unary {
            op: UnaryOperator::Negate,
            expr,
        } => format!("-{}", print(expr)),
        Expression::Math {
            fun,
            arg,
            arg1,
            arg2,
            arg3,
        } => {
            let args: Vec<String> = [Some(arg), arg1, arg2, arg3]
                .into_iter()
                .flatten()
                .map(print)
                .collect();
            format!("{fun:?}({})", args.join(", "))
        }
        Expression::AccessIndex { base, index } => format!("{}[{index}]", print(base)),
        ref other => format!("unexpected expression {other:?}"),
    }
}

/// The entry point's stores and return, one line each, in order.
fn stores(shader: &HostShader) -> Vec<String> {
    let function = &naga_module(shader).entry_points[0].function;
    function
        .body
        .iter()
        .filter_map(|statement| match *statement {
            Statement::Emit(_) => None,
            Statement::Store { pointer, value } => Some(format!(
                "{} = {}",
                print_expr(function, pointer),
                print_expr(function, value)
            )),
            Statement::Return { value } => Some(match value {
                Some(value) => format!("return {}", print_expr(function, value)),
                None => "return".into(),
            }),
            ref other => Some(format!("unexpected statement {other:?}")),
        })
        .collect()
}

/// The pair's bodies: which input component each load reads, the order of `fma`'s operands
/// (`a * b + c` is naga's `fma(a, b, c)`), and which output component each store hits. The
/// output struct's member 0 is the position in the vertex program.
#[test]
fn pair_golden_bodies() {
    let (vertex, pixel) = pair();
    let half = 0.5_f32.to_bits();
    let two = 2.0_f32.to_bits();
    assert_eq!(
        stores(&vertex),
        [
            "r0 = u32(in0[0])".to_string(),
            "r1 = u32(in0[1])".into(),
            "r2 = u32(in0[2])".into(),
            "r3 = u32(in0[3])".into(),
            "r4 = u32(in1[0])".into(),
            "r5 = u32(in1[1])".into(),
            format!("r0 = u32(Fma(f32(r0), f32({half:#x}), f32(r4)))"),
            format!("r6 = {ONE:#x}"),
            "out[0][0] = f32(r0)".into(),
            "out[0][1] = f32(r1)".into(),
            "out[0][2] = f32(r2)".into(),
            "out[0][3] = f32(r3)".into(),
            "out[1][0] = f32(r6)".into(),
            "return out".into(),
        ]
    );
    assert_eq!(
        stores(&pixel),
        [
            "r4 = lane_id".to_string(),
            format!("r0 = {ONE:#x}"),
            format!("r1 = u32(Fma(f32(r4), f32({two:#x}), f32(r0)))"),
            "r2 = 0x0".into(),
            "r3 = r0".into(),
            "out[0][0] = f32(r0)".into(),
            "out[0][1] = f32(r1)".into(),
            "out[0][2] = f32(r2)".into(),
            "out[0][3] = f32(r3)".into(),
            "return out".into(),
        ]
    );
}

/// Source modifiers and saturation: `|x|` is naga's `abs`, `-x` a negation applied after it,
/// and a saturating `FFMA` clamps the `fma` with `saturate`. Every component of a generic output
/// lands where its register says.
#[test]
fn modifiers_and_saturation_lower_to_their_naga_forms() {
    let modified = |reg, abs, neg| FloatSrc::Reg { reg, abs, neg };
    let program = assemble_program(&[
        ald(r(0), GENERIC_0_X * 4, 3),
        Instr::new(Op::FFma(OpFFma {
            dst: r(3),
            srcs: [
                modified(r(0), false, true),
                modified(r(1), true, false),
                modified(r(2), true, true),
            ],
            saturate: true,
            rnd: RoundMode::NearestEven,
            ftz: false,
            dnz: false,
        })),
        ast(r(0), (GENERIC_0_X + COMPONENTS) * 4, 4),
        exit(),
    ])
    .expect("encodable");
    let shader = translate_program(VERTEX, &program).expect("lowers");
    assert_eq!(
        stores(&shader),
        [
            "r0 = u32(in0[0])",
            "r1 = u32(in0[1])",
            "r2 = u32(in0[2])",
            "r3 = u32(Saturate(Fma(-f32(r0), Abs(f32(r1)), -Abs(f32(r2)))))",
            "out[1][0] = f32(r0)",
            "out[1][1] = f32(r1)",
            "out[1][2] = f32(r2)",
            "out[1][3] = f32(r3)",
            "return out",
        ]
    );
}

/// Every lowering error reads as plain words.
#[test]
fn lower_errors_read_as_words() {
    let mut malformed = Function::new();
    let b = malformed.entry();
    let zero = malformed.push(b, Inst::Imm32(0)).expect("room");
    malformed
        .push(b, Inst::SetReg(Gpr::RZ, zero))
        .expect("room");
    let verify = lower(GuestStage::Vertex, &malformed).expect_err("a write to rz is malformed");
    let attribute = Attribute(GENERIC_0_X);
    let cases = [
        (verify, "the IR is malformed: %1 reads or writes rz"),
        (
            LowerError::Stage(GuestStage::Geometry),
            "geometry programs are not lowered yet",
        ),
        (
            LowerError::Input {
                stage: GuestStage::Pixel,
                attribute,
            },
            "a pixel program cannot read attribute a0x80",
        ),
        (
            LowerError::Output {
                stage: GuestStage::Vertex,
                attribute,
            },
            "a vertex program cannot write attribute a0x80",
        ),
        (
            LowerError::GenericLocation {
                generic: HOST_LOCATIONS,
            },
            "generic 31 has no host location: locations stop at 30",
        ),
        (
            LowerError::LaneIdInVertex,
            "the lane id cannot be read in a vertex program",
        ),
        (
            LowerError::ColorOutput {
                target: 8,
                component: 0,
            },
            "colour output 8.0 does not exist",
        ),
        (
            LowerError::ColorInStage(GuestStage::Vertex),
            "a vertex program cannot write colour outputs",
        ),
        (
            LowerError::Validation("why".into()),
            "naga rejected the module: why",
        ),
        (
            LowerError::Internal("what".into()),
            "internal lowering error: what",
        ),
    ];
    for (error, text) in cases {
        assert_eq!(error.to_string(), text);
    }
}
