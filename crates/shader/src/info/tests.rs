//! Collect info tests: golden interface tables per stage from synthetic programs, each
//! interpolation mode and colour-target mask, the header fill, the constant-buffer sizes, the
//! requirement lists against Eden's, and each refusal with its exact error and order.

use super::attribute::{
    CLIP_DISTANCE_0, COLOR_BACK_SPECULAR_A, COLOR_FRONT_DIFFUSE_R, FIXED_FNC_TEXTURE_0_S,
    FIXED_FNC_TEXTURE_9_Q, FRONT_FACE, GENERIC_31_X, POINT_SPRITE_S, TESSELLATION_POINT_U,
    VIEWPORT_MASK,
};
use super::*;
use crate::asm::assemble_program;
use crate::decode::{
    Gpr, Instr, InterpFreq, InterpLoc, Op, OpALd, OpASt, OpExit, OpIpa, Pred, PredSrc,
};
use crate::host::Requirements;
use crate::ir::{ATTRIBUTE_WORD_BYTES, POSITION_W, Terminator, Type};
use crate::lower::capabilities;
use crate::program::{Rejection, graphics_program_ir};
use crate::sph::encode::encode_bytes;
use crate::sph::{
    Common, ComponentMask, HeaderError, PixelIo, SystemValuesAb, SystemValuesC, VtgIo, VtgStage,
};
use crate::translate::{
    COLOR_TARGETS, COMPONENT_MASK, TranslateError, Unsupported, translate_with_header,
};

fn r(index: u8) -> Gpr {
    Gpr::new(index)
}

/// The byte address of attribute word `word`.
fn addr(word: u16) -> u16 {
    word * u16::try_from(ATTRIBUTE_WORD_BYTES).expect("four")
}

fn exit() -> Instr {
    Instr::new(Op::Exit(OpExit {
        keep_ref_count: false,
        no_at_exit: false,
        pred: PredSrc::TRUE,
    }))
}

fn ald(dst: Gpr, word: u16, comps: u8) -> Instr {
    Instr::new(Op::ALd(OpALd {
        dst,
        vtx: Gpr::RZ,
        offset: Gpr::RZ,
        addr: addr(word),
        comps,
        patch: false,
        phys: false,
        output: false,
    }))
}

fn ast(data: Gpr, word: u16, comps: u8) -> Instr {
    Instr::new(Op::ASt(OpASt {
        data,
        vtx: Gpr::RZ,
        offset: Gpr::RZ,
        addr: addr(word),
        comps,
        patch: false,
        phys: false,
    }))
}

fn ipa(dst: Gpr, word: u16) -> Instr {
    Instr::new(Op::Ipa(OpIpa {
        dst,
        addr: addr(word),
        freq: InterpFreq::Pass,
        loc: InterpLoc::Default,
        offset: Gpr::RZ,
        pred: Pred::PT,
    }))
}

fn mask([x, y, z, w]: [bool; 4]) -> ComponentMask {
    ComponentMask { x, y, z, w }
}

fn all() -> ComponentMask {
    mask([true; 4])
}

fn vtg_header(stage: VtgStage, input: VtgMap, output: VtgMap) -> ProgramHeader {
    ProgramHeader {
        common: Common::default(),
        io: Io::Vtg(Box::new(VtgIo {
            stage,
            input,
            output,
            vmap: VtgMap::default(),
        })),
    }
}

fn pixel_header(io: &PixelIo) -> ProgramHeader {
    ProgramHeader {
        common: Common::default(),
        io: Io::Pixel(Box::new(*io)),
    }
}

/// The program's bytes: the header, then the instructions.
fn program(header: &ProgramHeader, instrs: &[Instr]) -> Vec<u8> {
    let mut bytes = encode_bytes(header).expect("encodable").to_vec();
    bytes.extend(assemble_program(instrs).expect("assembles"));
    bytes
}

fn entry(slot: Slot, components: u8, interpolation: Option<Interpolation>) -> InterfaceEntry {
    InterfaceEntry {
        slot,
        components,
        interpolation,
    }
}

fn imm(bits: u32) -> Operand {
    Operand::Imm(Imm::U32(bits))
}

/// The id of instruction `index` (ids have no public constructor; a function's arena makes
/// them).
fn value(index: u32) -> Value {
    let mut scratch = Function::new();
    let entry = scratch.entry();
    let mut last = None;
    for _ in 0..=index {
        last = scratch
            .push(entry, Inst::Op(Operation::new(Opcode::Void, vec![])))
            .ok();
    }
    last.expect("an id")
}

fn attr(word: u16) -> Operand {
    Operand::Imm(Imm::Attribute(Attribute(word)))
}

/// A one-block function of these operations, ending in a return.
fn function(ops: &[(Opcode, Vec<Operand>)]) -> Function {
    let mut function = Function::new();
    let entry = function.entry();
    for (opcode, args) in ops {
        function
            .push(entry, Inst::Op(Operation::new(*opcode, args.clone())))
            .expect("pushes");
    }
    if let Some(block) = function.block_mut(entry) {
        block.terminator = Terminator::Return;
    }
    function
}

fn compute(ops: &[(Opcode, Vec<Operand>)]) -> Result<ShaderInfo, InfoError> {
    collect(GuestStage::Compute, None, &function(ops))
}

fn words(mask: AttributeMask) -> Vec<u16> {
    (0..=u16::from(u8::MAX))
        .filter(|&word| has_word(mask, word))
        .collect()
}

// Golden interface tables, one per stage.

#[test]
fn vertex_interface_table_from_a_synthetic_program() {
    // Reads generic 0 (xyzw) and generic 3 (y only), writes the position from generic 0 and
    // generic 1's x and w from generic 3's y.
    let generic = |n: u16, component: u16| GENERIC_0_X + n * ATTRIBUTE_COMPONENTS + component;
    let instrs = [
        ald(r(0), generic(0, 0), 4),
        ald(r(4), generic(3, 1), 1),
        ast(r(0), POSITION_X, 4),
        ast(r(4), generic(1, 0), 1),
        ast(r(4), generic(1, 3), 1),
        exit(),
    ];
    let header = vtg_header(VtgStage::Vertex, VtgMap::default(), VtgMap::default());
    let ir =
        graphics_program_ir(GuestStage::Vertex, &program(&header, &instrs)).expect("translates");
    let info = ir.info;
    assert_eq!(info.stage(), GuestStage::Vertex);
    assert_eq!(
        info.inputs(),
        [
            entry(Slot::Generic(0), 0b1111, None),
            entry(Slot::Generic(3), 0b0010, None),
        ]
    );
    assert_eq!(
        info.outputs(),
        [
            entry(Slot::Position, 0b1111, None),
            entry(Slot::Generic(1), 0b1001, None),
        ]
    );
    assert_eq!(words(info.loads()), [32, 33, 34, 35, 45]);
    assert_eq!(words(info.stores()), [28, 29, 30, 31, 36, 39]);
    assert_eq!(info.color_outputs(), ColorOutputs::default());
    assert_eq!(info.requirements(), Requirements::default());
    assert_eq!(info.system_values(), SystemValues::EMPTY);
    assert_eq!(info.constant_buffers().count(), 0);
    assert_eq!(ir.header, header);
}

/// A pixel header whose generics 0, 1 and 2 are perspective, flat and screen-linear over x and
/// y, and whose targets and extra outputs are as given.
fn pixel_io(targets: [u8; COLOR_TARGETS], sample_mask: bool, depth: bool) -> PixelIo {
    let mut io = PixelIo::default();
    for (generic, mode) in [
        (0, PixelImap::Perspective),
        (1, PixelImap::Constant),
        (2, PixelImap::ScreenLinear),
    ] {
        io.generic[generic] = Components {
            x: mode,
            y: mode,
            z: PixelImap::Unused,
            w: PixelImap::Unused,
        };
    }
    io.targets =
        targets.map(|bits| mask([bits & 1 != 0, bits & 2 != 0, bits & 4 != 0, bits & 8 != 0]));
    io.writes_sample_mask = sample_mask;
    io.writes_depth = depth;
    io
}

#[test]
fn pixel_interface_table_from_a_synthetic_program() {
    // Target 0 writes red and alpha (r0, r3), target 2 green (r5); the sample mask is r8 and
    // the depth r9. The three generics feed them.
    let mut targets = [0; COLOR_TARGETS];
    targets[0] = 0b1001;
    targets[2] = 0b0010;
    let io = pixel_io(targets, true, true);
    let generic = |n: u16, component: u16| GENERIC_0_X + n * ATTRIBUTE_COMPONENTS + component;
    let instrs = [
        ipa(r(0), generic(0, 0)),
        ipa(r(3), generic(1, 1)),
        ipa(r(5), generic(2, 0)),
        ipa(r(8), generic(2, 1)),
        ipa(r(9), generic(1, 0)),
        exit(),
    ];
    let header = pixel_header(&io);
    let info = graphics_program_ir(GuestStage::Pixel, &program(&header, &instrs))
        .expect("translates")
        .info;
    // Generic 0 is perspective, so IPA also reads the position's W.
    assert_eq!(
        info.inputs(),
        [
            entry(Slot::Position, 0b1000, None),
            entry(Slot::Generic(0), 0b0001, Some(Interpolation::Smooth)),
            entry(Slot::Generic(1), 0b0011, Some(Interpolation::Flat)),
            entry(Slot::Generic(2), 0b0011, Some(Interpolation::NoPerspective)),
        ]
    );
    assert_eq!(
        info.outputs(),
        [
            entry(Slot::ColorTarget(0), 0b1001, None),
            entry(Slot::ColorTarget(2), 0b0010, None),
            entry(Slot::SampleMask, 1, None),
            entry(Slot::FragDepth, 1, None),
        ]
    );
    assert_eq!(info.color_outputs(), ColorOutputs { targets });
    assert!(info.stores_sample_mask());
    assert!(info.stores_frag_depth());
    assert_eq!(info.stores(), AttributeMask::EMPTY);
    assert_eq!(info.interpolation(0), Interpolation::Smooth);
    assert_eq!(info.interpolation(1), Interpolation::Flat);
    assert_eq!(info.interpolation(2), Interpolation::NoPerspective);
    assert_eq!(info.interpolation(3), Interpolation::Smooth);
}

#[test]
fn compute_interface_table_from_a_synthetic_function() {
    let info = compute(&[
        (Opcode::LocalInvocationId, vec![]),
        (Opcode::WorkgroupId, vec![]),
        (Opcode::GetCbufU32, vec![imm(0), imm(0x24)]),
        (Opcode::GetCbufU32x2, vec![imm(3), imm(0x100)]),
    ])
    .expect("collects");
    assert_eq!(info.stage(), GuestStage::Compute);
    assert_eq!(info.inputs(), []);
    assert_eq!(info.outputs(), []);
    assert_eq!(
        info.system_values().iter().collect::<Vec<_>>(),
        [SystemValue::WorkgroupId, SystemValue::LocalInvocationId]
    );
    assert_eq!(
        info.constant_buffers().collect::<Vec<_>>(),
        [
            ConstantBufferUse {
                bank: 0,
                size: 0x30
            },
            ConstantBufferUse {
                bank: 3,
                size: 0x110
            },
        ]
    );
    assert!(!info.indexed_constant_buffer_bank());
}

// The header against the slot.

#[test]
fn the_header_must_name_the_slots_stage() {
    let headers = [
        None,
        Some(vtg_header(
            VtgStage::Vertex,
            VtgMap::default(),
            VtgMap::default(),
        )),
        Some(vtg_header(
            VtgStage::TessellationInit,
            VtgMap::default(),
            VtgMap::default(),
        )),
        Some(vtg_header(
            VtgStage::Tessellation,
            VtgMap::default(),
            VtgMap::default(),
        )),
        Some(vtg_header(
            VtgStage::Geometry,
            VtgMap::default(),
            VtgMap::default(),
        )),
        Some(pixel_header(&PixelIo::default())),
    ];
    let slots = [
        GuestStage::VertexCullBeforeFetch,
        GuestStage::Vertex,
        GuestStage::TessellationInit,
        GuestStage::Tessellation,
        GuestStage::Geometry,
        GuestStage::Pixel,
        GuestStage::Compute,
    ];
    // The model: compute takes no header; both vertex slots take a vertex header; every other
    // slot takes a header of its own stage.
    let wanted = |slot| match slot {
        GuestStage::Compute => None,
        GuestStage::VertexCullBeforeFetch => Some(GuestStage::Vertex),
        other => Some(other),
    };
    for slot in slots {
        assert_eq!(header_stage(slot), wanted(slot), "{slot}");
        for header in &headers {
            let expected = match (wanted(slot), header) {
                (None, None) => Ok(()),
                (None, Some(_)) => Err(InfoError::ComputeHeader),
                (Some(_), None) => Err(InfoError::NoHeader(slot)),
                (Some(stage), Some(h)) if h.stage() == stage => Ok(()),
                (Some(_), Some(h)) => Err(InfoError::HeaderStage {
                    slot,
                    header: h.stage(),
                }),
            };
            assert_eq!(check_header(slot, header.as_ref()), expected, "{slot}");
            if let Err(error) = expected {
                let empty = Function::new();
                assert_eq!(collect(slot, header.as_ref(), &empty), Err(error));
            }
        }
    }
    // Spelled out for the cases a swapped arm would hide.
    let vertex = vtg_header(VtgStage::Vertex, VtgMap::default(), VtgMap::default());
    assert_eq!(
        check_header(GuestStage::VertexCullBeforeFetch, Some(&vertex)),
        Ok(())
    );
    assert_eq!(
        check_header(GuestStage::Pixel, Some(&vertex)),
        Err(InfoError::HeaderStage {
            slot: GuestStage::Pixel,
            header: GuestStage::Vertex
        })
    );
}

#[test]
fn translation_stage_takes_the_colour_outputs_from_a_pixel_header() {
    let mut targets = [0; COLOR_TARGETS];
    targets[1] = 0b0101;
    targets[7] = 0b1000;
    let pixel = pixel_header(&pixel_io(targets, false, false));
    assert_eq!(
        translation_stage(GuestStage::Pixel, &pixel),
        Stage {
            stage: GuestStage::Pixel,
            color_outputs: ColorOutputs { targets }
        }
    );
    let vertex = vtg_header(VtgStage::Vertex, VtgMap::default(), VtgMap::default());
    assert_eq!(
        translation_stage(GuestStage::VertexCullBeforeFetch, &vertex),
        Stage {
            stage: GuestStage::VertexCullBeforeFetch,
            color_outputs: ColorOutputs::default()
        }
    );
}

#[test]
fn graphics_program_ir_reports_each_step_in_order() {
    // Too short for a header.
    assert_eq!(
        graphics_program_ir(GuestStage::Vertex, &[0; 4]).err(),
        Some(Rejection::Header(HeaderError::Short {
            len: 4,
            needed: crate::sph::HEADER_BYTES
        }))
    );
    // A pixel header in a vertex slot, with code that would not translate: the slot check
    // comes first.
    let pixel = pixel_header(&PixelIo::default());
    let mut bad_code = program(&pixel, &[]);
    bad_code.push(0);
    assert_eq!(
        graphics_program_ir(GuestStage::Vertex, &bad_code).err(),
        Some(Rejection::Info(InfoError::HeaderStage {
            slot: GuestStage::Vertex,
            header: GuestStage::Pixel
        }))
    );
    // A compute slot is refused: compute programs have no header.
    assert_eq!(
        graphics_program_ir(GuestStage::Compute, &program(&pixel, &[exit()])).err(),
        Some(Rejection::Info(InfoError::ComputeHeader))
    );
    // A geometry program passes the header checks and is refused by the translator.
    let geometry = vtg_header(VtgStage::Geometry, VtgMap::default(), VtgMap::default());
    assert_eq!(
        graphics_program_ir(GuestStage::Geometry, &program(&geometry, &[exit()])).err(),
        Some(Rejection::Translate(TranslateError::Stage(
            Unsupported::Stage(GuestStage::Geometry)
        )))
    );
    // Collect info runs after the passes: a pixel input map mixing modes is refused there.
    let mut mixed = PixelIo::default();
    mixed.generic[4].x = PixelImap::Perspective;
    mixed.generic[4].y = PixelImap::Constant;
    assert_eq!(
        graphics_program_ir(
            GuestStage::Pixel,
            &program(&pixel_header(&mixed), &[exit()])
        )
        .err(),
        Some(Rejection::Info(InfoError::MixedInterpolation {
            generic: 4
        }))
    );
    assert_eq!(
        Rejection::Info(InfoError::MixedInterpolation { generic: 4 }).to_string(),
        "generic attribute 4 is interpolated differently per component"
    );
}

#[test]
fn dead_loads_are_removed_before_collect_info() {
    // r1 is loaded and never used: dead code elimination (slice 8) removes the load before
    // collect info runs, as in Eden's order.
    let generic0 = GENERIC_0_X;
    let instrs = [
        ald(r(0), generic0, 1),
        ald(r(1), generic0 + ATTRIBUTE_COMPONENTS, 1),
        ast(r(0), POSITION_X, 1),
        exit(),
    ];
    let header = vtg_header(VtgStage::Vertex, VtgMap::default(), VtgMap::default());
    let info = graphics_program_ir(GuestStage::Vertex, &program(&header, &instrs))
        .expect("translates")
        .info;
    assert_eq!(words(info.loads()), [generic0]);
}

// Interpolation.

#[test]
fn each_input_map_mode_gives_edens_interpolation() {
    assert_eq!(Interpolation::of(PixelImap::Unused), Interpolation::Smooth);
    assert_eq!(
        Interpolation::of(PixelImap::Perspective),
        Interpolation::Smooth
    );
    assert_eq!(Interpolation::of(PixelImap::Constant), Interpolation::Flat);
    assert_eq!(
        Interpolation::of(PixelImap::ScreenLinear),
        Interpolation::NoPerspective
    );
    assert_eq!(Interpolation::default(), Interpolation::Smooth);
}

/// The info of a pixel program reading every component of generic `generic` with input map
/// `map`.
fn pixel_reading(generic: u16, map: Components<PixelImap>) -> Result<ShaderInfo, InfoError> {
    let mut io = PixelIo::default();
    io.generic[usize::from(generic)] = map;
    let base = GENERIC_0_X + generic * ATTRIBUTE_COMPONENTS;
    let ops: Vec<_> = (base..base + ATTRIBUTE_COMPONENTS)
        .map(|word| (Opcode::GetAttribute, vec![attr(word), imm(0)]))
        .collect();
    collect(GuestStage::Pixel, Some(&pixel_header(&io)), &function(&ops))
}

#[test]
fn each_interpolation_mode_reaches_the_input_table() {
    let unused = PixelImap::Unused;
    for (mode, expected) in [
        (PixelImap::Perspective, Interpolation::Smooth),
        (PixelImap::Constant, Interpolation::Flat),
        (PixelImap::ScreenLinear, Interpolation::NoPerspective),
    ] {
        // The mode on one component only, the others unused, at the first and last generic
        // and on the first and last component.
        for (generic, map) in [
            (
                0,
                Components {
                    x: mode,
                    y: unused,
                    z: unused,
                    w: unused,
                },
            ),
            (
                31,
                Components {
                    x: unused,
                    y: unused,
                    z: unused,
                    w: mode,
                },
            ),
        ] {
            let info = pixel_reading(generic, map).expect("collects");
            let slot = Slot::Generic(u8::try_from(generic).expect("small"));
            assert_eq!(
                info.inputs(),
                [entry(slot, 0b1111, Some(expected))],
                "{mode:?}"
            );
            assert_eq!(
                info.interpolation(u8::try_from(generic).expect("small")),
                expected
            );
        }
    }
    // Every component unused: Eden's default.
    let info = pixel_reading(5, Components::default()).expect("collects");
    assert_eq!(
        info.inputs(),
        [entry(Slot::Generic(5), 0b1111, Some(Interpolation::Smooth))]
    );
}

#[test]
fn mixed_interpolation_is_refused_and_agreeing_components_are_not() {
    let p = PixelImap::Perspective;
    let c = PixelImap::Constant;
    let u = PixelImap::Unused;
    assert_eq!(
        pixel_reading(
            7,
            Components {
                x: p,
                y: u,
                z: u,
                w: c
            }
        )
        .err(),
        Some(InfoError::MixedInterpolation { generic: 7 })
    );
    assert_eq!(
        pixel_reading(
            7,
            Components {
                x: c,
                y: c,
                z: u,
                w: c
            }
        )
        .expect("agreeing")
        .interpolation(7),
        Interpolation::Flat
    );
    // Every generic is checked, read or not, and the lowest is reported first.
    let mut io = PixelIo::default();
    io.generic[9] = Components {
        x: p,
        y: c,
        z: u,
        w: u,
    };
    io.generic[30] = Components {
        x: c,
        y: p,
        z: u,
        w: u,
    };
    assert_eq!(
        collect(
            GuestStage::Pixel,
            Some(&pixel_header(&io)),
            &Function::new()
        ),
        Err(InfoError::MixedInterpolation { generic: 9 })
    );
}

#[test]
fn only_pixel_generic_inputs_carry_an_interpolation() {
    let mut io = PixelIo::default();
    io.generic[0].x = PixelImap::Constant;
    let ops = [
        (Opcode::GetAttribute, vec![attr(POSITION_W), imm(0)]),
        (Opcode::GetAttribute, vec![attr(GENERIC_0_X), imm(0)]),
        (Opcode::GetAttribute, vec![attr(FRONT_FACE), imm(0)]),
    ];
    let info =
        collect(GuestStage::Pixel, Some(&pixel_header(&io)), &function(&ops)).expect("collects");
    assert_eq!(
        info.inputs(),
        [
            entry(Slot::Position, 0b1000, None),
            entry(Slot::Generic(0), 0b0001, Some(Interpolation::Flat)),
            entry(Slot::FrontFace, 1, None),
        ]
    );
    // A vertex program's generic input has none.
    let vertex = vtg_header(VtgStage::Vertex, VtgMap::default(), VtgMap::default());
    let info = collect(GuestStage::Vertex, Some(&vertex), &function(&ops[1..2])).expect("collects");
    assert_eq!(info.inputs(), [entry(Slot::Generic(0), 0b0001, None)]);
}

// Pixel outputs.

#[test]
fn each_colour_target_mask_reaches_the_output_table() {
    for target in 0..COLOR_TARGETS {
        for bits in 1..=COMPONENT_MASK {
            let mut targets = [0; COLOR_TARGETS];
            targets[target] = bits;
            let header = pixel_header(&pixel_io(targets, false, false));
            // Each of the target's four registers holds a constant-free value: IPA of
            // generic 0's x, so the colours survive the passes.
            let first = u8::try_from(0).expect("zero");
            let instrs: Vec<_> = (first..4)
                .map(|reg| ipa(r(reg), GENERIC_0_X))
                .chain([exit()])
                .collect();
            let info = graphics_program_ir(GuestStage::Pixel, &program(&header, &instrs))
                .expect("translates")
                .info;
            let index = u8::try_from(target).expect("small");
            assert_eq!(info.color_outputs(), ColorOutputs { targets });
            assert_eq!(
                info.outputs(),
                [entry(Slot::ColorTarget(index), bits, None)],
                "target {target} mask {bits:#06b}"
            );
        }
    }
}

#[test]
fn a_pixel_program_writing_a_subset_of_its_targets() {
    // Targets 1, 4 and 6 of eight, with partial masks; registers follow the targets with any
    // component written, four each.
    let mut targets = [0; COLOR_TARGETS];
    targets[1] = 0b0011;
    targets[4] = 0b1000;
    targets[6] = 0b0101;
    let header = pixel_header(&pixel_io(targets, false, false));
    let instrs: Vec<_> = (0..12)
        .map(|reg| ipa(r(reg), GENERIC_0_X))
        .chain([exit()])
        .collect();
    let ir =
        graphics_program_ir(GuestStage::Pixel, &program(&header, &instrs)).expect("translates");
    assert_eq!(
        ir.info.outputs(),
        [
            entry(Slot::ColorTarget(1), 0b0011, None),
            entry(Slot::ColorTarget(4), 0b1000, None),
            entry(Slot::ColorTarget(6), 0b0101, None),
        ]
    );
    assert!(!ir.info.stores_sample_mask());
    assert!(!ir.info.stores_frag_depth());
}

#[test]
fn pixel_outputs_record_each_component() {
    let info = collect(
        GuestStage::Pixel,
        Some(&pixel_header(&PixelIo::default())),
        &function(&[
            (
                Opcode::SetFragColor,
                vec![imm(7), imm(3), Operand::Imm(Imm::F32(0))],
            ),
            (
                Opcode::SetFragColor,
                vec![imm(7), imm(0), Operand::Imm(Imm::F32(0))],
            ),
            (
                Opcode::SetFragColor,
                vec![imm(0), imm(2), Operand::Imm(Imm::F32(0))],
            ),
        ]),
    )
    .expect("collects");
    let mut targets = [0; COLOR_TARGETS];
    targets[0] = 0b0100;
    targets[7] = 0b1001;
    assert_eq!(info.color_outputs(), ColorOutputs { targets });
}

/// The IR of a pixel program with `targets`, the sample mask and depth as given, and an exit.
fn exit_ir(targets: [u8; COLOR_TARGETS], sample_mask: bool, depth: bool) -> String {
    let io = pixel_io(targets, sample_mask, depth);
    let stage = Stage {
        stage: GuestStage::Pixel,
        color_outputs: io.color_outputs(),
    };
    translate_with_header(stage, &io, &[exit()])
        .expect("translates")
        .to_string()
}

#[test]
fn exit_reads_the_sample_mask_and_depth_after_the_colours() {
    // Two targets with any component written: registers r0 to r7. The sample mask is r8, the
    // depth r9, whether the sample mask is written or not.
    let mut targets = [0; COLOR_TARGETS];
    targets[0] = 0b0001;
    targets[3] = 0b0001;
    let colours = "block0:\n\
         \x20   %0: u32 = GetRegister r0\n\
         \x20   %1: f32 = BitCastF32U32 %0\n\
         \x20   %2 = SetFragColor u32 0x0, u32 0x0, %1\n\
         \x20   %3: u32 = GetRegister r4\n\
         \x20   %4: f32 = BitCastF32U32 %3\n\
         \x20   %5 = SetFragColor u32 0x3, u32 0x0, %4\n";
    assert_eq!(
        exit_ir(targets, false, false),
        format!("{colours}    return\n")
    );
    assert_eq!(
        exit_ir(targets, true, false),
        format!("{colours}    %6: u32 = GetRegister r8\n    %7 = SetSampleMask %6\n    return\n")
    );
    assert_eq!(
        exit_ir(targets, false, true),
        format!(
            "{colours}    %6: u32 = GetRegister r9\n    %7: f32 = BitCastF32U32 %6\n    \
             %8 = SetFragDepth %7\n    return\n"
        )
    );
    assert_eq!(
        exit_ir(targets, true, true),
        format!(
            "{colours}    %6: u32 = GetRegister r8\n    %7 = SetSampleMask %6\n    \
             %8: u32 = GetRegister r9\n    %9: f32 = BitCastF32U32 %8\n    \
             %10 = SetFragDepth %9\n    return\n"
        )
    );
    // With no colour target, the sample mask is r0 and the depth r1.
    assert_eq!(
        exit_ir([0; COLOR_TARGETS], true, true),
        "block0:\n    %0: u32 = GetRegister r0\n    %1 = SetSampleMask %0\n    \
         %2: u32 = GetRegister r1\n    %3: f32 = BitCastF32U32 %2\n    \
         %4 = SetFragDepth %3\n    return\n"
    );
    // Every target written: the depth is register 33, the highest any header reaches.
    let full = exit_ir([COMPONENT_MASK; COLOR_TARGETS], true, true);
    assert!(full.contains("GetRegister r32\n"), "{full}");
    assert!(full.contains("GetRegister r33\n"), "{full}");
}

// Attribute words.

#[test]
fn attribute_words_match_edens_list() {
    // Literal rows from `attribute.h`.
    let rows: [(u16, Slot, u8); 27] = [
        (24, Slot::PrimitiveId, 0),
        (25, Slot::Layer, 0),
        (26, Slot::ViewportIndex, 0),
        (27, Slot::PointSize, 0),
        (28, Slot::Position, 0),
        (31, Slot::Position, 3),
        (32, Slot::Generic(0), 0),
        (37, Slot::Generic(1), 1),
        (156, Slot::Generic(31), 0),
        (159, Slot::Generic(31), 3),
        (160, Slot::LegacyColor(0), 0),
        (167, Slot::LegacyColor(1), 3),
        (175, Slot::LegacyColor(3), 3),
        (176, Slot::ClipDistances, 0),
        (183, Slot::ClipDistances, 7),
        (184, Slot::PointSprite, 0),
        (185, Slot::PointSprite, 1),
        (186, Slot::FogCoordinate, 0),
        (188, Slot::TessellationPoint, 0),
        (189, Slot::TessellationPoint, 1),
        (190, Slot::InstanceId, 0),
        (191, Slot::VertexId, 0),
        (192, Slot::FixedFncTexture(0), 0),
        (197, Slot::FixedFncTexture(1), 1),
        (231, Slot::FixedFncTexture(9), 3),
        (232, Slot::ViewportMask, 0),
        (255, Slot::FrontFace, 0),
    ];
    for (word, slot, component) in rows {
        assert_eq!(slot_of(Attribute(word)), Some((slot, component)), "{word}");
    }
    for word in [0, 23, 187, 233, 254, 256, u16::MAX] {
        assert_eq!(slot_of(Attribute(word)), None, "{word}");
    }
    // Every named word, counted: 24 to 186, 188 to 232, and 255.
    let named = (0..=u16::MAX)
        .filter(|&w| slot_of(Attribute(w)).is_some())
        .count();
    assert_eq!(named, (186 - 24 + 1) + (232 - 188 + 1) + 1);
    // The named constants are Eden's numbers.
    assert_eq!(
        [
            GENERIC_31_X,
            COLOR_FRONT_DIFFUSE_R,
            COLOR_BACK_SPECULAR_A,
            CLIP_DISTANCE_0,
            POINT_SPRITE_S,
            TESSELLATION_POINT_U,
            FIXED_FNC_TEXTURE_0_S,
            FIXED_FNC_TEXTURE_9_Q,
            VIEWPORT_MASK,
            FRONT_FACE,
        ],
        [156, 160, 175, 176, 184, 188, 192, 231, 232, 255]
    );
}

#[test]
fn unknown_attribute_words_are_refused() {
    for opcode in [
        Opcode::GetAttribute,
        Opcode::GetAttributeU32,
        Opcode::InterpolateAtOffset,
        Opcode::SetAttribute,
    ] {
        let arg_for = |word| match opcode {
            Opcode::SetAttribute => vec![attr(word), Operand::Imm(Imm::F32(0)), imm(0)],
            _ => vec![attr(word), imm(0)],
        };
        // Each edge pair: the refused word and its accepted neighbour.
        for (refused, accepted) in [(23, 24), (187, 186), (233, 232), (254, 255), (256, 255)] {
            assert_eq!(
                compute(&[(opcode, arg_for(refused))]),
                Err(InfoError::UnknownAttribute {
                    at: value(0),
                    attribute: Attribute(refused)
                }),
                "{opcode} {refused}"
            );
            assert!(
                compute(&[(opcode, arg_for(accepted))]).is_ok(),
                "{opcode} {accepted}"
            );
        }
        // A non-attribute first argument.
        assert_eq!(
            compute(&[(opcode, vec![imm(GENERIC_0_X.into()), imm(0)])]),
            Err(InfoError::Operand {
                at: value(0),
                index: 0
            })
        );
    }
    assert_eq!(
        InfoError::UnknownAttribute {
            at: value(2),
            attribute: Attribute(23)
        }
        .to_string(),
        "%2: attribute a0x5c is not one the guest has"
    );
}

#[test]
fn loads_and_stores_record_their_words() {
    let info = compute(&[
        (
            Opcode::GetAttribute,
            vec![attr(CLIP_DISTANCE_0 + 7), imm(0)],
        ),
        (Opcode::GetAttributeU32, vec![attr(FRONT_FACE), imm(0)]),
        (
            Opcode::InterpolateAtOffset,
            vec![attr(COLOR_FRONT_DIFFUSE_R + 5), imm(0)],
        ),
        (
            Opcode::SetAttribute,
            vec![
                attr(FIXED_FNC_TEXTURE_9_Q),
                Operand::Imm(Imm::F32(0)),
                imm(0),
            ],
        ),
        (
            Opcode::SetAttribute,
            vec![attr(POINT_SPRITE_S + 1), Operand::Imm(Imm::F32(0)), imm(0)],
        ),
    ])
    .expect("collects");
    assert_eq!(words(info.loads()), [165, 183, 255]);
    assert_eq!(words(info.stores()), [185, 231]);
    assert_eq!(
        info.inputs(),
        [
            entry(Slot::LegacyColor(1), 0b0010, None),
            entry(Slot::ClipDistances, 0b1000_0000, None),
            entry(Slot::FrontFace, 1, None),
        ]
    );
    assert_eq!(
        info.outputs(),
        [
            entry(Slot::PointSprite, 0b10, None),
            entry(Slot::FixedFncTexture(9), 0b1000, None),
        ]
    );
    assert!(!info.loads_indexed());
    assert!(!info.stores_indexed());
}

#[test]
fn unreachable_blocks_are_collected() {
    let mut function = function(&[]);
    let orphan = function.add_block().expect("a block");
    function
        .push(orphan, Inst::Op(Operation::new(Opcode::SampleId, vec![])))
        .expect("pushes");
    function
        .push(
            orphan,
            Inst::Op(Operation::new(
                Opcode::GetAttribute,
                vec![attr(GENERIC_0_X), imm(0)],
            )),
        )
        .expect("pushes");
    assert_ne!(orphan, function.entry());
    let info = collect(GuestStage::Compute, None, &function).expect("collects");
    assert!(info.system_values().contains(SystemValue::SampleId));
    assert_eq!(words(info.loads()), [GENERIC_0_X]);
}

// The header fill.

#[test]
fn a_pixel_header_fills_loads_only_with_indexed_loads() {
    let u = PixelImap::Unused;
    let p = PixelImap::Perspective;
    let mut io = PixelIo::default();
    // Generic 2: y and w used. Generic 0's x is loaded directly but unused in the map.
    io.generic[2] = Components {
        x: u,
        y: p,
        z: u,
        w: p,
    };
    let header = pixel_header(&io);
    let direct = (Opcode::GetAttribute, vec![attr(GENERIC_0_X), imm(0)]);
    let indexed = (Opcode::GetAttributeIndexed, vec![imm(0), imm(0)]);
    // Without indexed loads the masks are the walk's.
    let info = collect(
        GuestStage::Pixel,
        Some(&header),
        &function(std::slice::from_ref(&direct)),
    )
    .expect("collects");
    assert_eq!(words(info.loads()), [GENERIC_0_X]);
    // With them the header decides every generic word, clearing the direct load (Eden assigns).
    let info = collect(
        GuestStage::Pixel,
        Some(&header),
        &function(&[direct, indexed]),
    )
    .expect("collects");
    assert!(info.loads_indexed());
    let generic2 = GENERIC_0_X + 2 * ATTRIBUTE_COMPONENTS;
    assert_eq!(words(info.loads()), [generic2 + 1, generic2 + 3]);
}

/// A vertex map with each filled field set to a distinct value.
fn distinct_map(seed: bool) -> VtgMap {
    let mut map = VtgMap::default();
    map.generic[if seed { 1 } else { 30 }] = mask([seed, !seed, true, false]);
    map.system_values_ab = SystemValuesAb {
        primitive_id: seed,
        rt_array_index: !seed,
        viewport_index: seed,
        point_size: !seed,
        position: mask([!seed, seed, !seed, seed]),
        ..SystemValuesAb::default()
    };
    map.system_values_c = SystemValuesC {
        clip_distances: if seed { 0b1000_0001 } else { 0b0110_0000 },
        point_sprite_s: seed,
        point_sprite_t: !seed,
        fog_coordinate: seed,
        tessellation_point_u: !seed,
        tessellation_point_v: seed,
        instance_id: !seed,
        vertex_id: seed,
    };
    // Not filled from the header (Eden: "TODO: Legacy varyings").
    map.color[0] = all();
    map.texture[0] = all();
    map
}

#[test]
fn a_vertex_header_fills_loads_and_stores_from_their_own_maps() {
    let header = vtg_header(VtgStage::Vertex, distinct_map(true), distinct_map(false));
    let loads_indexed = (Opcode::GetAttributeIndexed, vec![imm(0), imm(0)]);
    let stores_indexed = (
        Opcode::SetAttributeIndexed,
        vec![imm(0), Operand::Imm(Imm::F32(0)), imm(0)],
    );
    // A direct load of the layer (cleared by the fill: the input map leaves it unused) and a
    // direct store of generic 0's x (cleared: the output map leaves generic 0 unused).
    let direct = [
        (Opcode::GetAttribute, vec![attr(LAYER), imm(0)]),
        (
            Opcode::SetAttribute,
            vec![attr(GENERIC_0_X), Operand::Imm(Imm::F32(0)), imm(0)],
        ),
    ];
    let mut ops = direct.to_vec();
    ops.push(loads_indexed.clone());
    ops.push(stores_indexed.clone());
    let info = collect(GuestStage::Vertex, Some(&header), &function(&ops)).expect("collects");
    let generic1 = GENERIC_0_X + ATTRIBUTE_COMPONENTS;
    let generic30 = GENERIC_0_X + 30 * ATTRIBUTE_COMPONENTS;
    // Input map (seed true): primitive id, viewport index, position y and w, generic 1's x and
    // z, clip distances 0 and 7, point sprite s, fog, tessellation v, vertex id.
    assert_eq!(
        words(info.loads()),
        [
            PRIMITIVE_ID,
            VIEWPORT_INDEX,
            POSITION_X + 1,
            POSITION_X + 3,
            generic1,
            generic1 + 2,
            CLIP_DISTANCE_0,
            CLIP_DISTANCE_0 + 7,
            POINT_SPRITE_S,
            FOG_COORDINATE,
            TESSELLATION_POINT_U + 1,
            VERTEX_ID,
        ]
    );
    // Output map (seed false): layer, point size, position x and z, generic 30's y and z, clip
    // distances 5 and 6, point sprite t, tessellation u, instance id.
    assert_eq!(
        words(info.stores()),
        [
            LAYER,
            POINT_SIZE,
            POSITION_X,
            POSITION_X + 2,
            generic30 + 1,
            generic30 + 2,
            CLIP_DISTANCE_0 + 5,
            CLIP_DISTANCE_0 + 6,
            POINT_SPRITE_S + 1,
            TESSELLATION_POINT_U,
            INSTANCE_ID,
        ]
    );
    // Only the loads are filled when only the loads are indexed, and the other way round.
    let mut ops = direct.to_vec();
    ops.push(loads_indexed);
    let info = collect(GuestStage::Vertex, Some(&header), &function(&ops)).expect("collects");
    assert_eq!(words(info.stores()), [GENERIC_0_X]);
    assert!(!info.loads().contains(u8::try_from(LAYER).expect("small")));
    let mut ops = direct.to_vec();
    ops.push(stores_indexed);
    let info = collect(GuestStage::Vertex, Some(&header), &function(&ops)).expect("collects");
    assert_eq!(words(info.loads()), [LAYER]);
    assert!(info.stores_indexed() && !info.loads_indexed());
}

// System values.

#[test]
fn each_system_value_opcode_is_recorded() {
    let rows = [
        (Opcode::WorkgroupId, SystemValue::WorkgroupId),
        (Opcode::LocalInvocationId, SystemValue::LocalInvocationId),
        (Opcode::InvocationId, SystemValue::InvocationId),
        (Opcode::InvocationInfo, SystemValue::InvocationInfo),
        (Opcode::SampleId, SystemValue::SampleId),
        (Opcode::IsHelperInvocation, SystemValue::IsHelperInvocation),
        (Opcode::LaneId, SystemValue::SubgroupInvocationId),
        (Opcode::SubgroupEqMask, SystemValue::SubgroupMasks),
        (Opcode::SubgroupLtMask, SystemValue::SubgroupMasks),
        (Opcode::SubgroupLeMask, SystemValue::SubgroupMasks),
        (Opcode::SubgroupGtMask, SystemValue::SubgroupMasks),
        (Opcode::SubgroupGeMask, SystemValue::SubgroupMasks),
    ];
    for (opcode, value) in rows {
        let info = compute(&[(opcode, vec![])]).expect("collects");
        assert_eq!(
            info.system_values().iter().collect::<Vec<_>>(),
            [value],
            "{opcode}"
        );
    }
    let named: Vec<_> = rows.iter().map(|&(opcode, _)| opcode).collect();
    for &opcode in Opcode::ALL {
        if !named.contains(&opcode) {
            assert_eq!(SystemValue::of(opcode), None, "{opcode}");
        }
    }
    // The set: each value alone, and every value at once.
    for value in SystemValue::ALL {
        let set = SystemValues::EMPTY.with(value);
        assert_eq!(set.iter().collect::<Vec<_>>(), [value]);
        assert_eq!(set.with(value), set);
    }
    let every = SystemValue::ALL
        .into_iter()
        .fold(SystemValues::EMPTY, SystemValues::with);
    assert_eq!(every.iter().collect::<Vec<_>>(), SystemValue::ALL);
}

// Constant buffers.

#[test]
fn constant_buffer_sizes_follow_eden() {
    // Each element size, read at offset 0x20: the end rounded up to 16.
    for (opcode, element) in [
        (Opcode::GetCbufU8, 1),
        (Opcode::GetCbufS8, 1),
        (Opcode::GetCbufU16, 2),
        (Opcode::GetCbufS16, 2),
        (Opcode::GetCbufU32, 4),
        (Opcode::GetCbufF32, 4),
        (Opcode::GetCbufU32x2, 8),
    ] {
        assert_eq!(cbuf_element_size(opcode), Some(element), "{opcode}");
        let info = compute(&[(opcode, vec![imm(5), imm(0x2c)])]).expect("collects");
        let expected = (0x2c + element).next_multiple_of(CBUF_ALIGNMENT);
        assert_eq!(
            info.constant_buffers().collect::<Vec<_>>(),
            [ConstantBufferUse {
                bank: 5,
                size: expected
            }],
            "{opcode}"
        );
    }
    // The end exactly on a multiple of 16 stays; one past it rounds up.
    let size = |offset| {
        compute(&[(Opcode::GetCbufU32, vec![imm(0), imm(offset)])])
            .expect("collects")
            .constant_buffers()
            .next()
            .map(|c| c.size)
    };
    assert_eq!(size(0x1c), Some(0x20));
    assert_eq!(size(0x1d), Some(0x30));
    // Several reads of a bank: the largest end, whatever the order.
    for order in [[0x40, 0x10], [0x10, 0x40]] {
        let ops: Vec<_> = order
            .iter()
            .map(|&offset| (Opcode::GetCbufU32, vec![imm(2), imm(offset)]))
            .collect();
        let info = compute(&ops).expect("collects");
        assert_eq!(
            info.constant_buffers().collect::<Vec<_>>(),
            [ConstantBufferUse {
                bank: 2,
                size: 0x50
            }]
        );
    }
    // A register offset: the whole bank.
    let register = Operand::Value(value(0));
    let info = compute(&[
        (Opcode::LaneId, vec![]),
        (Opcode::GetCbufU32, vec![imm(9), register]),
        (Opcode::GetCbufU32, vec![imm(10), imm(0)]),
    ])
    .expect("collects");
    assert_eq!(
        info.constant_buffers().collect::<Vec<_>>(),
        [
            ConstantBufferUse {
                bank: 9,
                size: INDEXED_CBUF_SIZE
            },
            ConstantBufferUse {
                bank: 10,
                size: 0x10
            },
        ]
    );
    assert_eq!(cbuf_element_size(Opcode::GetAttribute), None);
}

#[test]
fn a_register_offset_keeps_a_larger_immediate_end() {
    // A read ending past 64 KiB, then a register-offset read of the same bank: the larger size
    // stays (Eden assigns 0x10000, shrinking it; a listed divergence), in either order.
    let register = Operand::Value(value(0));
    let past = (Opcode::GetCbufU32x2, vec![imm(1), imm(0xfffc)]);
    let indexed = (Opcode::GetCbufU32, vec![imm(1), register]);
    for ops in [
        vec![(Opcode::LaneId, vec![]), past.clone(), indexed.clone()],
        vec![(Opcode::LaneId, vec![]), indexed, past],
    ] {
        let info = compute(&ops).expect("collects");
        assert_eq!(
            info.constant_buffers().collect::<Vec<_>>(),
            [ConstantBufferUse {
                bank: 1,
                size: 0x1_0010
            }]
        );
    }
}

#[test]
fn an_indexed_bank_reads_every_bank() {
    let register = Operand::Value(value(0));
    let info = compute(&[
        (Opcode::LaneId, vec![]),
        (Opcode::GetCbufU32, vec![imm(4), imm(0x1_fff0)]),
        (Opcode::GetCbufU32, vec![register, imm(0)]),
    ])
    .expect("collects");
    assert!(info.indexed_constant_buffer_bank());
    let uses: Vec<_> = info.constant_buffers().collect();
    assert_eq!(uses.len(), usize::from(CBUF_BANKS));
    for (bank, used) in (0_u8..).zip(&uses) {
        // Bank 4's earlier read ends past 64 KiB, so it keeps its size.
        let size = if bank == 4 {
            0x2_0000
        } else {
            INDEXED_CBUF_SIZE
        };
        assert_eq!(*used, ConstantBufferUse { bank, size });
    }
}

#[test]
fn constant_buffer_banks_stop_at_the_last() {
    let last = u32::from(CBUF_BANKS) - 1;
    let info = compute(&[(Opcode::GetCbufU32, vec![imm(last), imm(0)])]).expect("bank 31");
    assert_eq!(
        info.constant_buffers().collect::<Vec<_>>(),
        [ConstantBufferUse {
            bank: CBUF_BANKS - 1,
            size: CBUF_ALIGNMENT
        }]
    );
    let past = u32::from(CBUF_BANKS);
    for bank in [past, u32::MAX] {
        assert_eq!(
            compute(&[(Opcode::GetCbufU32, vec![imm(bank), imm(0)])]),
            Err(InfoError::ConstantBufferBank { at: value(0), bank })
        );
    }
    assert_eq!(
        InfoError::ConstantBufferBank {
            at: value(0),
            bank: past
        }
        .to_string(),
        "%0: constant buffer bank 32 does not exist; banks stop at 31"
    );
}

#[test]
fn constant_buffer_ends_overflow_is_refused() {
    let read = |offset| compute(&[(Opcode::GetCbufU32, vec![imm(0), imm(offset)])]);
    // The last offset whose end rounds up within 32 bits, and the first whose does not.
    let last = u32::MAX - (CBUF_ALIGNMENT - 1) - 4;
    assert_eq!(
        read(last).expect("fits").constant_buffers().next(),
        Some(ConstantBufferUse {
            bank: 0,
            size: u32::MAX - (CBUF_ALIGNMENT - 1)
        })
    );
    for offset in [last + 1, u32::MAX - 3, u32::MAX] {
        assert_eq!(
            read(offset),
            Err(InfoError::ConstantBufferOffset {
                at: value(0),
                offset
            })
        );
    }
    assert_eq!(
        InfoError::ConstantBufferOffset {
            at: value(0),
            offset: u32::MAX
        }
        .to_string(),
        "%0: a constant buffer read at offset 0xffffffff ends past 4 GiB"
    );
    // A missing offset argument.
    assert_eq!(
        compute(&[(Opcode::GetCbufU32, vec![imm(0)])]),
        Err(InfoError::Operand {
            at: value(0),
            index: 1
        })
    );
    assert_eq!(
        compute(&[(Opcode::GetCbufU32, vec![])]),
        Err(InfoError::Operand {
            at: value(0),
            index: 0
        })
    );
}

// Colour outputs.

#[test]
fn colour_outputs_stop_at_the_last_target_and_component() {
    let colour = |target, component| {
        compute(&[(
            Opcode::SetFragColor,
            vec![imm(target), imm(component), Operand::Imm(Imm::F32(0))],
        )])
    };
    let last_target = u32::try_from(COLOR_TARGETS).expect("small") - 1;
    let last_component = u32::from(ATTRIBUTE_COMPONENTS) - 1;
    assert!(colour(last_target, last_component).is_ok());
    for (target, component) in [
        (last_target + 1, 0),
        (0, last_component + 1),
        (u32::MAX, 0),
        (0, u32::MAX),
    ] {
        assert_eq!(
            colour(target, component),
            Err(InfoError::ColorOutput {
                at: value(0),
                target,
                component
            })
        );
    }
    assert_eq!(
        compute(&[(
            Opcode::SetFragColor,
            vec![Operand::Value(value(9)), imm(0), imm(0)]
        )]),
        Err(InfoError::Operand {
            at: value(0),
            index: 0
        })
    );
    assert_eq!(
        compute(&[(
            Opcode::SetFragColor,
            vec![imm(0), Operand::Value(value(9)), imm(0)]
        )]),
        Err(InfoError::Operand {
            at: value(0),
            index: 1
        })
    );
}

// Requirements.

#[test]
fn requirements_for_f16_f64_and_subgroup_operations() {
    let info = compute(&[
        (Opcode::FPAdd16, vec![]),
        (Opcode::FPMul64, vec![]),
        (Opcode::VoteAll, vec![]),
    ])
    .expect("collects");
    assert_eq!(
        info.requirements(),
        Requirements {
            subgroup: true,
            f16: true,
            f64: true,
            i64: false,
            i64_atomics: false,
        }
    );
    // Each alone, and the 64-bit integer flags.
    let alone = |opcode| {
        compute(&[(opcode, vec![])])
            .expect("collects")
            .requirements()
    };
    let only = |f: fn(&mut Requirements)| {
        let mut r = Requirements::default();
        f(&mut r);
        r
    };
    assert_eq!(alone(Opcode::ConvertF32F16), only(|r| r.f16 = true));
    assert_eq!(alone(Opcode::PackDouble2x32), only(|r| r.f64 = true));
    assert_eq!(alone(Opcode::QuadSwap), only(|r| r.subgroup = true));
    assert_eq!(alone(Opcode::IAdd64), only(|r| r.i64 = true));
    assert_eq!(
        alone(Opcode::StorageAtomicXor64),
        only(|r| {
            r.i64 = true;
            r.i64_atomics = true;
        })
    );
    // `BitCastU64F64` is on both of Eden's lists.
    assert_eq!(
        alone(Opcode::BitCastU64F64),
        only(|r| {
            r.f64 = true;
            r.i64 = true;
        })
    );
    assert_eq!(alone(Opcode::FPAdd32), Requirements::default());
}

/// Whether `ty` is a 16-bit float type, a 64-bit float type, or a 64-bit integer.
fn has(opcode: Opcode, types: &[Type]) -> bool {
    types.contains(&opcode.result()) || opcode.args().iter().any(|t| types.contains(t))
}

#[test]
fn requirement_lists_agree_with_the_opcode_types() {
    use Opcode as O;
    // An independent source: the opcode table's types. Every opcode that takes or gives an f16
    // or f64 type is on Eden's list for it, and the reverse, except the listed exceptions.
    let f16_types = [Type::F16, Type::F16x2, Type::F16x3, Type::F16x4];
    let f64_types = [Type::F64, Type::F64x2, Type::F64x3, Type::F64x4];
    let mut f16_diff = Vec::new();
    let mut f64_diff = Vec::new();
    for &opcode in Opcode::ALL {
        let r = opcode_requirements(opcode);
        if r.f16 != has(opcode, &f16_types) {
            f16_diff.push(opcode);
        }
        if r.f64 != has(opcode, &f64_types) {
            f64_diff.push(opcode);
        }
    }
    assert_eq!(f16_diff, Vec::<Opcode>::new());
    assert_eq!(f64_diff, Vec::<Opcode>::new());
    // 64-bit integers: Eden's list against the `U64` type. Eden counts every global-memory
    // access (64-bit addresses, which this table types as opaque or `u32x2`); nothing else
    // differs, since the four `u64` forms Eden's list misses are added here (the module docs'
    // divergence table).
    let i64_diff: Vec<_> = Opcode::ALL
        .iter()
        .copied()
        .filter(|&opcode| opcode_requirements(opcode).i64 != has(opcode, &[Type::U64]))
        .collect();
    assert_eq!(
        i64_diff,
        [
            O::LoadGlobalU8,
            O::LoadGlobalS8,
            O::LoadGlobalU16,
            O::LoadGlobalS16,
            O::LoadGlobal32,
            O::LoadGlobal64,
            O::LoadGlobal128,
            O::WriteGlobalU8,
            O::WriteGlobalS8,
            O::WriteGlobalU16,
            O::WriteGlobalS16,
            O::WriteGlobal32,
            O::WriteGlobal64,
            O::WriteGlobal128,
            O::GlobalAtomicIAdd32x2,
            O::GlobalAtomicSMin32x2,
            O::GlobalAtomicUMin32x2,
            O::GlobalAtomicSMax32x2,
            O::GlobalAtomicUMax32x2,
            O::GlobalAtomicAnd32x2,
            O::GlobalAtomicOr32x2,
            O::GlobalAtomicXor32x2,
            O::GlobalAtomicExchange32x2,
        ]
    );
    for added in [
        O::IAbs64,
        O::ConvertF16S64,
        O::ConvertF32S64,
        O::ConvertF64S64,
    ] {
        assert!(opcode_requirements(added).i64, "{added}");
    }
}

#[test]
fn capabilities_follow_each_requirement() {
    use naga::valid::Capabilities;
    let only = |f: fn(&mut Requirements)| {
        let mut r = Requirements::default();
        f(&mut r);
        capabilities(r)
    };
    assert_eq!(capabilities(Requirements::default()), Capabilities::empty());
    assert_eq!(only(|r| r.subgroup = true), Capabilities::SUBGROUP);
    assert_eq!(only(|r| r.f16 = true), Capabilities::SHADER_FLOAT16);
    assert_eq!(only(|r| r.f64 = true), Capabilities::FLOAT64);
    assert_eq!(only(|r| r.i64 = true), Capabilities::SHADER_INT64);
    assert_eq!(
        only(|r| r.i64_atomics = true),
        Capabilities::SHADER_INT64_ATOMIC_ALL_OPS
    );
}

// Order of checks.

#[test]
fn checks_run_in_the_documented_order() {
    let bad_op = (Opcode::GetCbufU32, vec![imm(u32::from(CBUF_BANKS)), imm(0)]);
    let ops = function(std::slice::from_ref(&bad_op));
    // The header against the stage, before the interpolation and the instructions.
    let mut mixed = PixelIo::default();
    mixed.generic[0].x = PixelImap::Constant;
    mixed.generic[0].y = PixelImap::Perspective;
    let mixed = pixel_header(&mixed);
    assert_eq!(
        collect(GuestStage::Vertex, Some(&mixed), &ops),
        Err(InfoError::HeaderStage {
            slot: GuestStage::Vertex,
            header: GuestStage::Pixel
        })
    );
    // The interpolation before the instructions.
    assert_eq!(
        collect(GuestStage::Pixel, Some(&mixed), &ops),
        Err(InfoError::MixedInterpolation { generic: 0 })
    );
    // Instructions in listed order: the first fault wins, whichever kind it is.
    let unknown = (Opcode::GetAttribute, vec![attr(0), imm(0)]);
    assert_eq!(
        compute(&[unknown.clone(), bad_op.clone()]),
        Err(InfoError::UnknownAttribute {
            at: value(0),
            attribute: Attribute(0)
        })
    );
    assert_eq!(
        compute(&[bad_op, unknown]),
        Err(InfoError::ConstantBufferBank {
            at: value(0),
            bank: u32::from(CBUF_BANKS)
        })
    );
}

#[test]
fn errors_read_as_words() {
    assert_eq!(
        InfoError::NoHeader(GuestStage::Pixel).to_string(),
        "a pixel program needs its shader program header"
    );
    assert_eq!(
        InfoError::ComputeHeader.to_string(),
        "a compute program has no shader program header, but one was given"
    );
    assert_eq!(
        InfoError::HeaderStage {
            slot: GuestStage::VertexCullBeforeFetch,
            header: GuestStage::Geometry
        }
        .to_string(),
        "a program bound as a vertex (cull before fetch) program has a geometry header"
    );
    assert_eq!(
        InfoError::Operand {
            at: value(3),
            index: 1
        }
        .to_string(),
        "%3: argument 1 is not the immediate collect info reads"
    );
    assert_eq!(
        InfoError::ColorOutput {
            at: value(0),
            target: 8,
            component: 0
        }
        .to_string(),
        "%0: colour output 8.0 does not exist"
    );
}
