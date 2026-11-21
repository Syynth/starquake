//! The epilogue's emulation (the `emulation` submodule): each flag's effect on the module, the
//! stages and conditions that leave it out, and the wgpu acceptance of a vertex and pixel pair
//! with constant buffers and driver constants, built from program bytes, with an explicit
//! pipeline layout from the resource tables.

use std::fmt::Write as _;

use super::*;
use crate::asm::assemble;
use crate::driver_constants::DRIVER_CONSTANTS_SIZE;
use crate::host::{ResourceKind, ResourceSlot, ResourceSource};
use crate::info::CBUF_ALIGNMENT;
use crate::testing::ldc;
use crate::variant::{CompareFunction, EmulationKey};

/// The position's first attribute byte (`0x70`, attribute word 28).
const POSITION_BYTE: u32 = POSITION_X as u32 * ATTRIBUTE_WORD_BYTES;

/// The four position writes, from four distinct constants, as IR lines `%0` to `%3`.
fn position_writes() -> String {
    let mut text = String::new();
    for component in 0..4_u32 {
        let value = f64::from(component + 2);
        let byte = POSITION_BYTE + component * ATTRIBUTE_WORD_BYTES;
        writeln!(
            text,
            "    %{component} = SetAttribute a{byte:#x}, f32 {value:?}, u32 0"
        )
        .expect("writes to a string");
    }
    text
}

/// A vertex function of one block that writes the position and ends in `end` (`return` or
/// `unreachable`).
fn vertex_ending(end: &str) -> Function {
    ir(&format!("block0:\n{}    {end}\n", position_writes()))
}

/// A vertex function of one block that writes the position and returns.
fn vertex() -> Function {
    vertex_ending("return")
}

/// A pixel function that writes colour target `target` from four distinct constants and
/// returns.
fn pixel(target: u32) -> Function {
    let mut text = String::from("block0:\n");
    for component in 0..4_u32 {
        let value = f64::from(component + 2);
        writeln!(
            text,
            "    %{component} = SetFragColor u32 {target}, u32 {component}, f32 {value:?}"
        )
        .expect("writes to a string");
    }
    text.push_str("    return\n");
    ir(&text)
}

/// The lowering inputs with `emulation`.
fn emulating(emulation: EmulationKey) -> LowerInputs {
    LowerInputs {
        emulation,
        ..LowerInputs::DEFAULT
    }
}

/// The driver constants' row in `group`.
fn driver_row(group: u32) -> ResourceSlot {
    ResourceSlot {
        group,
        binding: 0,
        kind: ResourceKind::UniformBuffer {
            size: u32::try_from(DRIVER_CONSTANTS_SIZE).expect("small"),
        },
        source: ResourceSource::DriverConstants,
    }
}

/// The comparison operators of `shader`'s entry point, in expression order.
fn comparisons(shader: &HostShader) -> Vec<naga::BinaryOperator> {
    use naga::BinaryOperator as B;
    naga_module(shader).entry_points[0]
        .function
        .expressions
        .iter()
        .filter_map(|(_, expression)| match *expression {
            Expression::Binary { op, .. }
                if matches!(
                    op,
                    B::Less | B::LessEqual | B::Greater | B::GreaterEqual | B::Equal | B::NotEqual
                ) =>
            {
                Some(op)
            }
            _ => None,
        })
        .collect()
}

/// Whether `shader`'s entry point discards: a `Kill` anywhere in its body.
fn kills(shader: &HostShader) -> bool {
    fn block_kills(block: &Block) -> bool {
        block.iter().any(|statement| match statement {
            Statement::Kill => true,
            Statement::If { accept, reject, .. } => block_kills(accept) || block_kills(reject),
            Statement::Block(inner) => block_kills(inner),
            Statement::Loop {
                body, continuing, ..
            } => block_kills(body) || block_kills(continuing),
            _ => false,
        })
    }
    block_kills(&naga_module(shader).entry_points[0].function.body)
}

/// The depth-range rewrite: z becomes `(z + w) * 0.5`, read from the stored position, with no
/// driver constant; the flipped viewport reads the three position words and lists the driver
/// constants in the vertex group; with both, the rewrite comes first (its `0.5` is written
/// before the flip reads its first constant). The MSL of both is pinned whole.
#[test]
fn the_flipped_viewport_reads_three_constants() {
    let base = lowered(GuestStage::Vertex, &vertex(), &LowerInputs::DEFAULT);
    let base_msl = msl(&base);
    assert!(!base_msl.contains("driver_constants") && !base_msl.contains("0.5"));
    assert_eq!(base.resources, []);

    let rewrite_key = EmulationKey {
        depth_range_rewrite: true,
        ..EmulationKey::default()
    };
    let rewrite = lowered(GuestStage::Vertex, &vertex(), &emulating(rewrite_key));
    let rewrite_msl = msl(&rewrite);
    assert!(rewrite_msl.contains(" * 0.5"), "{rewrite_msl}");
    assert!(!rewrite_msl.contains("driver_constants"), "{rewrite_msl}");
    assert_eq!(rewrite.resources, []);

    let flip_key = EmulationKey {
        flipped_viewport: true,
        ..EmulationKey::default()
    };
    let flip = lowered(GuestStage::Vertex, &vertex(), &emulating(flip_key));
    let flip_msl = msl(&flip);
    for member in ["position_y_scale", "position_z_scale", "position_z_offset"] {
        assert!(flip_msl.contains(member), "{member}: {flip_msl}");
    }
    assert!(!flip_msl.contains(" * 0.5"));
    assert_eq!(flip.resources, [driver_row(0)]);

    let both_key = EmulationKey {
        depth_range_rewrite: true,
        flipped_viewport: true,
        ..EmulationKey::default()
    };
    let both = lowered(GuestStage::Vertex, &vertex(), &emulating(both_key));
    let both_msl = msl(&both);
    let half = both_msl.find(" * 0.5").expect("the rewrite");
    let first_constant = both_msl
        .find("driver_constants.position_y_scale")
        .expect("the flip");
    assert!(half < first_constant, "{both_msl}");
    insta::assert_snapshot!("epilogue_vertex", both_msl);
}

/// The alpha test, for each function: `Always` leaves the module as it is with no flag;
/// `Never` discards with no comparison and no driver constants; each other function discards
/// when the comparison of target 0's alpha with the reference (ordered, through the NaN-safe
/// helper) fails, reading the reference from the driver constants in the fragment group.
#[test]
fn the_alpha_test_reads_the_reference() {
    use naga::BinaryOperator as B;
    let base = lowered(GuestStage::Pixel, &pixel(0), &LowerInputs::DEFAULT);
    assert!(!kills(&base));
    assert_eq!(comparisons(&base), []);
    let with = |function| {
        lowered(
            GuestStage::Pixel,
            &pixel(0),
            &emulating(EmulationKey {
                alpha_test: Some(function),
                ..EmulationKey::default()
            }),
        )
    };
    let always = with(CompareFunction::Always);
    assert_eq!(msl(&always), msl(&base));
    assert_eq!(always.resources, []);
    let never = with(CompareFunction::Never);
    assert!(kills(&never));
    assert_eq!(comparisons(&never), []);
    assert_eq!(never.resources, []);
    assert!(!msl(&never).contains("alpha_reference"));
    for (function, op) in [
        (CompareFunction::Less, B::Less),
        (CompareFunction::Equal, B::Equal),
        (CompareFunction::LessThanEqual, B::LessEqual),
        (CompareFunction::Greater, B::Greater),
        (CompareFunction::NotEqual, B::NotEqual),
        (CompareFunction::GreaterThanEqual, B::GreaterEqual),
    ] {
        let shader = with(function);
        assert!(kills(&shader), "{function:?}");
        assert_eq!(comparisons(&shader), [op], "{function:?}");
        assert_eq!(shader.resources, [driver_row(1)], "{function:?}");
        let text = msl(&shader);
        assert!(text.contains("alpha_reference"), "{function:?}");
        // Ordered: NaN in either operand fails the test (`isnan` of both, joined with or and
        // negated before the and).
        assert_eq!(
            text.matches("metal::isnan(").count(),
            2,
            "{function:?}: {text}"
        );
    }
    insta::assert_snapshot!("epilogue_alpha_test", msl(&with(CompareFunction::Greater)));
}

/// The alpha test reads colour target 0's alpha (component 3), not another component or
/// target.
#[test]
fn the_alpha_test_reads_target_0_s_alpha() {
    let shader = lowered(
        GuestStage::Pixel,
        &pixel(0),
        &emulating(EmulationKey {
            alpha_test: Some(CompareFunction::Less),
            ..EmulationKey::default()
        }),
    );
    let function = &naga_module(&shader).entry_points[0].function;
    let reads: Vec<String> = function
        .expressions
        .iter()
        .filter_map(|(handle, expression)| match *expression {
            Expression::Load { .. } => Some(print_expr(function, handle)),
            _ => None,
        })
        .collect();
    assert!(reads.contains(&"out[0][3]".to_string()), "{reads:?}");
}

/// Flags of another stage, a function that ends in `unreachable`, and an alpha test on a pixel
/// program that does not write target 0 each leave the module as it is with no flag.
#[test]
fn the_epilogue_is_left_out_where_it_does_not_apply() {
    let every_flag = emulating(EmulationKey {
        alpha_test: Some(CompareFunction::Less),
        depth_range_rewrite: true,
        primitive_restart_rewrite: true,
        attribute_fetch: true,
        typed_outputs: true,
        flipped_viewport: true,
    });
    let pixel_flags = emulating(EmulationKey {
        alpha_test: Some(CompareFunction::Less),
        ..EmulationKey::default()
    });
    let vertex_flags = emulating(EmulationKey {
        depth_range_rewrite: true,
        flipped_viewport: true,
        ..EmulationKey::default()
    });
    let same = |stage, function: &Function, inputs: &LowerInputs| {
        let plain = lowered(stage, function, &LowerInputs::DEFAULT);
        let flagged = lowered(stage, function, inputs);
        assert_eq!(msl(&flagged), msl(&plain), "{stage}");
        assert_eq!(flagged.resources, plain.resources, "{stage}");
    };
    // The vertex flags on a pixel program, the pixel flag on a vertex program.
    same(GuestStage::Pixel, &pixel(0), &vertex_flags);
    same(GuestStage::Vertex, &vertex(), &pixel_flags);
    // `unreachable`: nothing, whatever the flags.
    same(
        GuestStage::Vertex,
        &vertex_ending("unreachable"),
        &every_flag,
    );
    // Target 1 written, not target 0: no alpha test.
    same(GuestStage::Pixel, &pixel(1), &pixel_flags);
    // The flags the epilogue does not lower yet change nothing on either stage.
    let unlowered = emulating(EmulationKey {
        primitive_restart_rewrite: true,
        attribute_fetch: true,
        typed_outputs: true,
        ..EmulationKey::default()
    });
    same(GuestStage::Vertex, &vertex(), &unlowered);
    same(GuestStage::Pixel, &pixel(0), &unlowered);
}

/// Each return does the steps: a structured function with an early return inside an `if` and
/// a closing return gives two rewrites, one inside the `if` before its return; the same
/// function with a closing `unreachable` gives one (module docs).
#[test]
fn each_return_does_the_steps() {
    let structured = |end: &str| {
        ir(&format!(
            "block0:\n{}\
             \x20   %4: u1 = ULessThan u32 1, u32 2\n\
             \x20   %5: u1 = ConditionRef %4\n\
             block1:\n\
             \x20   %6 = Epilogue\n\
             block2:\n\
             structure:\n\
             \x20   block0\n\
             \x20   if %5 {{\n\
             \x20       block1\n\
             \x20       return\n\
             \x20   }}\n\
             \x20   block2\n\
             \x20   {end}\n",
            position_writes()
        ))
    };
    let key = emulating(EmulationKey {
        depth_range_rewrite: true,
        ..EmulationKey::default()
    });
    let both = lowered(GuestStage::Vertex, &structured("return"), &key);
    assert_eq!(msl(&both).matches(" * 0.5").count(), 2, "{}", msl(&both));
    let body = &naga_module(&both).entry_points[0].function.body;
    let early = body
        .iter()
        .find_map(|statement| match statement {
            Statement::If { accept, .. } => Some(accept),
            _ => None,
        })
        .expect("the if");
    assert!(
        early
            .iter()
            .any(|statement| matches!(statement, Statement::Store { .. })),
        "the early return's rewrite is inside the if"
    );
    let one = lowered(GuestStage::Vertex, &structured("unreachable"), &key);
    assert_eq!(msl(&one).matches(" * 0.5").count(), 1, "{}", msl(&one));
}

/// A program of one block from program bytes has no `Epilogue` (it is not structured) and
/// still does the steps at its return.
#[test]
fn a_program_of_one_block_does_the_steps() {
    let (vertex, pixel) = acceptance_pair();
    let vertex_msl = msl(&vertex);
    assert!(vertex_msl.contains(" * 0.5"), "{vertex_msl}");
    assert!(vertex_msl.contains("position_y_scale"), "{vertex_msl}");
    assert!(kills(&pixel));
    assert!(msl(&pixel).contains("alpha_reference"));
}

/// A bind group layout entry for a uniform buffer at `binding`, visible to `stage`.
fn uniform_entry(binding: u32, stage: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: stage,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// The layout entries of `shader`'s resource table.
fn entries(shader: &HostShader, stage: wgpu::ShaderStages) -> Vec<wgpu::BindGroupLayoutEntry> {
    shader
        .resources
        .iter()
        .map(|slot| match slot.kind {
            ResourceKind::UniformBuffer { .. } => uniform_entry(slot.binding, stage),
        })
        .collect()
}

/// The bank both programs of the acceptance pair read.
const BANK: u8 = 2;

/// The acceptance pair from program bytes: a vertex program that loads a word of bank
/// [`BANK`] into the position's x, and a pixel program that loads one into target 0's red,
/// each lowered with every emulation its stage has (the vertex program with the flip and the
/// rewrite, the pixel program with an alpha test) and the bank sized by the alignment.
fn acceptance_pair() -> (HostShader, HostShader) {
    let program = |instructions: Vec<crate::decode::Instruction>| -> Vec<u8> {
        instructions
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect()
    };
    let vertex_code = program(vec![
        ldc(0, Gpr::RZ.index(), BANK, 0x4),
        assemble(&mov(r(1), MovSrc::Imm(ONE))).expect("encodable"),
        assemble(&mov(r(2), MovSrc::Reg(Gpr::RZ))).expect("encodable"),
        assemble(&mov(r(3), MovSrc::Imm(ONE))).expect("encodable"),
        assemble(&ast(r(0), 0x70, 4)).expect("encodable"),
        assemble(&exit()).expect("encodable"),
    ]);
    let pixel_code = program(vec![
        ldc(0, Gpr::RZ.index(), BANK, 0x8),
        assemble(&mov(r(1), MovSrc::Reg(Gpr::RZ))).expect("encodable"),
        assemble(&mov(r(2), MovSrc::Reg(Gpr::RZ))).expect("encodable"),
        assemble(&mov(r(3), MovSrc::Imm(ONE))).expect("encodable"),
        assemble(&exit()).expect("encodable"),
    ]);
    let mut constant_buffers = [0; CONSTANT_BUFFER_SLOTS];
    constant_buffers[usize::from(BANK)] = CBUF_ALIGNMENT;
    let inputs = |emulation| LowerInputs {
        constant_buffers,
        emulation,
        ..LowerInputs::DEFAULT
    };
    let vertex_ir = crate::program::program_ir(VERTEX, &vertex_code).expect("translates");
    let pixel_ir = crate::program::program_ir(pixel_stage(), &pixel_code).expect("translates");
    for function in [&vertex_ir, &pixel_ir] {
        assert!(
            function.structure().is_none()
                && function.listed().all(|(_, inst)| {
                    !matches!(inst, Inst::Op(op) if op.opcode == Opcode::Epilogue)
                }),
            "a program of one block is not structured and has no Epilogue"
        );
    }
    let vertex = lowered(
        GuestStage::Vertex,
        &vertex_ir,
        &inputs(EmulationKey {
            depth_range_rewrite: true,
            flipped_viewport: true,
            ..EmulationKey::default()
        }),
    );
    let pixel = lowered(
        GuestStage::Pixel,
        &pixel_ir,
        &inputs(EmulationKey {
            alpha_test: Some(CompareFunction::GreaterThanEqual),
            ..EmulationKey::default()
        }),
    );
    (vertex, pixel)
}

/// The pair's resource tables: the driver constants and bank [`BANK`] in each stage's own
/// group, as the convention says.
#[test]
fn acceptance_pair_tables_are_golden() {
    let (vertex, pixel) = acceptance_pair();
    let bank = |group| ResourceSlot {
        group,
        binding: 1 + u32::from(BANK),
        kind: ResourceKind::UniformBuffer {
            size: CBUF_ALIGNMENT,
        },
        source: ResourceSource::ConstantBuffer { bank: BANK },
    };
    assert_eq!(vertex.resources, [driver_row(0), bank(0)]);
    assert_eq!(pixel.resources, [driver_row(1), bank(1)]);
}

/// wgpu acceptance (spec, "The offline test harness"): the pair's modules are created as
/// shader modules and a render pipeline with an explicit layout made from the resource tables
/// alone (group 0 the vertex stage's, group 1 the fragment stage's); a layout with one group
/// for both stages is refused, so the layout is checked against the modules. Skipped, with a
/// notice, where the host has no GPU adapter.
#[test]
fn acceptance_pair_is_accepted_by_wgpu() {
    let (vertex, pixel) = acceptance_pair();
    let Some(gpu) = TestDevice::open(
        "acceptance_pair_is_accepted_by_wgpu",
        wgpu::Features::empty(),
    ) else {
        return;
    };
    let vertex_module = gpu
        .module(naga_module(&vertex).clone())
        .expect("wgpu takes the vertex module");
    let pixel_module = gpu
        .module(naga_module(&pixel).clone())
        .expect("wgpu takes the pixel module");
    let vertex_entries = entries(&vertex, wgpu::ShaderStages::VERTEX);
    let pixel_entries = entries(&pixel, wgpu::ShaderStages::FRAGMENT);
    let target = Some(wgpu::ColorTargetState {
        format: wgpu::TextureFormat::Rgba8Unorm,
        blend: None,
        write_mask: wgpu::ColorWrites::ALL,
    });
    let build = |groups: &[&[wgpu::BindGroupLayoutEntry]]| {
        let layout = gpu.layout_groups(groups).expect("the layout");
        gpu.render_pipeline(
            Some(&layout),
            (&vertex_module, ENTRY_POINT),
            (&pixel_module, ENTRY_POINT),
            &[],
            std::slice::from_ref(&target),
        )
    };
    build(&[&vertex_entries, &pixel_entries]).expect("wgpu builds the pipeline");
    // One group for both stages, as a convention that shared a group would have: the pixel
    // module binds group 1, which the layout lacks, so wgpu refuses it.
    let both = wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT;
    let shared = entries(&vertex, both);
    assert!(
        build(&[&shared]).is_err(),
        "a layout with one group for both stages is refused"
    );
}

/// A structured tree that does not end in `return` or `unreachable` gets the entry point's
/// closing return, which does the steps too.
#[test]
fn the_closing_return_does_the_steps() {
    let function = ir(&format!(
        "block0:\n{}structure:\n    block0\n",
        position_writes()
    ));
    let key = emulating(EmulationKey {
        depth_range_rewrite: true,
        ..EmulationKey::default()
    });
    let shader = lowered(GuestStage::Vertex, &function, &key);
    assert_eq!(
        msl(&shader).matches(" * 0.5").count(),
        1,
        "{}",
        msl(&shader)
    );
}
