//! The index rewrite. With no GPU: the shader parses and validates with naga and matches the
//! names and layout of `rewrite.rs`; the parameter block's bytes; which draws are rewritten;
//! how many indices each writes. On a GPU (skipped with a notice in CI and where the host has
//! no adapter): the shader's output for every mode and source equals an independent model.

use super::*;
use wgpu::naga::valid::{Capabilities, ValidationFlags, Validator};

fn module() -> naga::Module {
    naga::front::wgsl::parse_str(SOURCE).expect("the rewrite shader parses")
}

/// The value of the module's `u32` constant `name`.
fn constant(module: &naga::Module, name: &str) -> u32 {
    let (_, c) = module
        .constants
        .iter()
        .find(|(_, c)| c.name.as_deref() == Some(name))
        .unwrap_or_else(|| unreachable!("the shader declares {name}"));
    match module.global_expressions[c.init] {
        naga::Expression::Literal(naga::Literal::U32(v)) => v,
        ref other => unreachable!("{name} is {other:?}"),
    }
}

#[test]
fn the_shader_validates_and_matches_its_names() {
    let m = module();
    Validator::new(ValidationFlags::all(), Capabilities::empty())
        .validate(&m)
        .expect("the rewrite shader validates with no optional capability");
    assert_eq!(m.entry_points.len(), 1);
    let entry = &m.entry_points[0];
    assert_eq!(
        (entry.name.as_str(), entry.stage),
        (ENTRY, naga::ShaderStage::Compute)
    );
    assert_eq!(entry.workgroup_size, [WORKGROUP_SIZE, 1, 1]);
    let space = |binding: u32| {
        m.global_variables
            .iter()
            .find(|(_, g)| {
                g.binding
                    == Some(naga::ResourceBinding {
                        group: GROUP,
                        binding,
                    })
            })
            .map(|(_, g)| g.space)
    };
    assert_eq!(space(PARAMS_BINDING), Some(naga::AddressSpace::Uniform));
    assert_eq!(
        space(INPUT_BINDING),
        Some(naga::AddressSpace::Storage {
            access: naga::StorageAccess::LOAD
        })
    );
    assert_eq!(
        space(OUTPUT_BINDING),
        Some(naga::AddressSpace::Storage {
            access: naga::StorageAccess::LOAD | naga::StorageAccess::STORE
        })
    );
    assert_eq!(m.global_variables.len(), 3, "nothing else is bound");
    for (name, value) in [
        ("MODE_LIST", Mode::List as u32),
        ("MODE_QUADS", Mode::Quads as u32),
        ("MODE_QUAD_STRIP", Mode::QuadStrip as u32),
        ("MODE_FAN", Mode::Fan as u32),
        ("MODE_LOOP", Mode::Loop as u32),
        ("SOURCE_SEQUENTIAL", Source::Sequential as u32),
        ("SOURCE_U8", Source::U8 as u32),
        ("SOURCE_U16", Source::U16 as u32),
        ("SOURCE_U32", Source::U32 as u32),
        ("RESTART", RESTART),
        ("WORKGROUP_SIZE", WORKGROUP_SIZE),
    ] {
        assert_eq!(constant(&m, name), value, "{name}");
    }
}

#[test]
fn the_parameter_block_matches_the_shaders_struct() {
    let m = module();
    let (_, params) = m
        .types
        .iter()
        .find(|(_, t)| t.name.as_deref() == Some("Params"))
        .expect("the shader declares Params");
    let naga::TypeInner::Struct { members, span } = &params.inner else {
        unreachable!("Params is a struct");
    };
    assert!(u64::from(*span) <= PARAMS_SIZE, "the block holds the struct");
    let offsets: Vec<(&str, usize)> = members
        .iter()
        .map(|m| {
            (
                m.name.as_deref().unwrap_or(""),
                usize::try_from(m.offset).expect("small"),
            )
        })
        .collect();
    assert_eq!(
        offsets,
        [
            ("count", COUNT_OFFSET),
            ("mode", MODE_OFFSET),
            ("source", SOURCE_OFFSET),
            ("first", FIRST_OFFSET),
            ("input_count", INPUT_COUNT_OFFSET),
            ("restart", RESTART_OFFSET),
        ]
    );
}

/// The block's bytes, each field distinct and non-zero, for every mode and source.
#[test]
fn parameter_blocks_have_these_bytes() {
    let block = params(
        Rewrite {
            mode: Mode::QuadStrip,
            source: Source::U16,
            restart: true,
        },
        0x0102_0304,
        0x1112_1314,
        0x2122_2324,
    );
    assert_eq!(
        block,
        [
            0x04, 0x03, 0x02, 0x01, // count
            0x02, 0x00, 0x00, 0x00, // mode: quad strip
            0x02, 0x00, 0x00, 0x00, // source: 16-bit
            0x14, 0x13, 0x12, 0x11, // first
            0x24, 0x23, 0x22, 0x21, // input count
            0x01, 0x00, 0x00, 0x00, // restart
            0, 0, 0, 0, 0, 0, 0, 0, // padding
        ]
    );
    // The values the shader's constants have, written out.
    let modes = [
        (Mode::List, 0),
        (Mode::Quads, 1),
        (Mode::QuadStrip, 2),
        (Mode::Fan, 3),
        (Mode::Loop, 4),
    ];
    let sources = [
        (Source::Sequential, 0),
        (Source::U8, 1),
        (Source::U16, 2),
        (Source::U32, 3),
    ];
    for (mode, m) in modes {
        for (source, s) in sources {
            for restart in [false, true] {
                let block = params(
                    Rewrite {
                        mode,
                        source,
                        restart,
                    },
                    1,
                    2,
                    3,
                );
                assert_eq!(block[MODE_OFFSET..MODE_OFFSET + 4], [m, 0, 0, 0]);
                assert_eq!(block[SOURCE_OFFSET..SOURCE_OFFSET + 4], [s, 0, 0, 0]);
                assert_eq!(
                    block[RESTART_OFFSET..RESTART_OFFSET + 4],
                    [u8::from(restart), 0, 0, 0]
                );
            }
        }
    }
}

/// Which draws are rewritten, and how, for every topology the pipeline state accepts, every
/// source and both restart settings, written out by hand from the rules on [`plan`].
#[test]
fn the_rewrite_plan_for_every_topology_and_source() {
    use IndexFormat::{U8, U16, U32};
    let rw = |mode, source, restart| {
        Some(Rewrite {
            mode,
            source,
            restart,
        })
    };
    let topologies = [
        Topology::Points,
        Topology::Lines,
        Topology::LineLoop,
        Topology::LineStrip,
        Topology::Triangles,
        Topology::TriangleStrip,
        Topology::TriangleFan,
        Topology::Quads,
        Topology::QuadStrip,
        Topology::Polygon,
    ];
    for topology in topologies {
        for restart in [false, true] {
            for index in [None, Some(U8), Some(U16), Some(U32)] {
                let source = Source::of(index);
                let expected = match (topology, index) {
                    (Topology::Quads, _) => rw(Mode::Quads, source, false),
                    (Topology::QuadStrip, _) => rw(Mode::QuadStrip, source, false),
                    (Topology::TriangleFan | Topology::Polygon, _) => {
                        rw(Mode::Fan, source, false)
                    }
                    (Topology::LineLoop, _) => rw(Mode::Loop, source, false),
                    (Topology::LineStrip | Topology::TriangleStrip, Some(U8 | U16)) => {
                        rw(Mode::List, source, restart)
                    }
                    (Topology::Points | Topology::Lines | Topology::Triangles, Some(U8)) => {
                        rw(Mode::List, source, false)
                    }
                    _ => None,
                };
                assert_eq!(
                    plan(topology, restart, index),
                    expected,
                    "{topology:?} restart {restart} {index:?}"
                );
            }
        }
    }
    // The cases the divergences name, written out: restart off is not honoured for 32-bit
    // strip indices, nor inside the rewritten topologies.
    assert_eq!(plan(Topology::TriangleStrip, false, Some(U32)), None);
    assert_eq!(
        plan(Topology::TriangleFan, true, Some(U16)),
        rw(Mode::Fan, Source::U16, false)
    );
}

/// The output counts at each mode's edges, and where they stop fitting a `u32`.
#[test]
fn output_counts_at_their_edges() {
    let cases = [
        (Mode::List, 0, Some(0)),
        (Mode::List, 5, Some(5)),
        (Mode::List, u32::MAX, Some(u32::MAX)),
        (Mode::Quads, 3, Some(0)),
        (Mode::Quads, 4, Some(6)),
        (Mode::Quads, 7, Some(6)),
        (Mode::Quads, 8, Some(12)),
        (Mode::QuadStrip, 3, Some(0)),
        (Mode::QuadStrip, 4, Some(6)),
        (Mode::QuadStrip, 5, Some(6)),
        (Mode::QuadStrip, 6, Some(12)),
        (Mode::Fan, 2, Some(0)),
        (Mode::Fan, 3, Some(3)),
        (Mode::Fan, 4, Some(6)),
        (Mode::Loop, 1, Some(0)),
        (Mode::Loop, 2, Some(3)),
        (Mode::Loop, 4, Some(5)),
        (Mode::Loop, u32::MAX - 1, Some(u32::MAX)),
        (Mode::Loop, u32::MAX, None),
        (Mode::Fan, u32::MAX, None),
        (Mode::Quads, u32::MAX, None),
        (Mode::QuadStrip, u32::MAX, None),
    ];
    for (mode, n, count) in cases {
        assert_eq!(output_count(mode, n), count, "{mode:?} {n}");
    }
    // The largest quad and fan inputs that still fit.
    let quads = u32::MAX / 6 * 4;
    assert_eq!(output_count(Mode::Quads, quads), Some(quads / 4 * 6));
    assert_eq!(output_count(Mode::Quads, quads + 4), None);
    let fan = u32::MAX / 3 + 2;
    assert_eq!(output_count(Mode::Fan, fan), Some((fan - 2) * 3));
    assert_eq!(output_count(Mode::Fan, fan + 1), None);
}

#[test]
fn workgroups_cover_the_count_up_to_the_limit() {
    assert_eq!(workgroups(1, 100), 1);
    assert_eq!(workgroups(WORKGROUP_SIZE, 100), 1);
    assert_eq!(workgroups(WORKGROUP_SIZE + 1, 100), 2);
    assert_eq!(workgroups(WORKGROUP_SIZE * 100, 100), 100);
    assert_eq!(workgroups(WORKGROUP_SIZE * 100 + 1, 100), 100, "at the limit");
    assert_eq!(workgroups(0, 100), 1, "never an empty dispatch");
}

// ---------------------------------------------------------------------------------------------
// On a GPU.

/// The rewrite of `n` inputs from `first` in `input`, by the rules of the module docs and
/// Eden's orders, written independently of the shader.
fn model(rw: Rewrite, n: u32, first: u32, input: &[u8]) -> Vec<u32> {
    let count = output_count(rw.mode, n).expect("fits");
    let position = |k: u32| match rw.mode {
        Mode::List => k,
        Mode::Quads => [0, 1, 2, 0, 2, 3][(k % 6) as usize] + 4 * (k / 6),
        Mode::QuadStrip => [0, 3, 1, 0, 2, 3][(k % 6) as usize] + 2 * (k / 6),
        Mode::Fan => {
            let t = k / 3;
            [t + 1, t + 2, 0][(k % 3) as usize]
        }
        Mode::Loop => {
            if k == n {
                0
            } else {
                k
            }
        }
    };
    let value = |p: u32| {
        let size = rw.source.index_bytes();
        if size == 0 {
            return first + p;
        }
        let at = (first + p * size) as usize;
        let mut word = [0; 4];
        word[..size as usize].copy_from_slice(&input[at..at + size as usize]);
        let v = u32::from_le_bytes(word);
        let max = u32::MAX >> (32 - 8 * size);
        if rw.restart && v == max { RESTART } else { v }
    };
    (0..count).map(|k| value(position(k))).collect()
}

/// A device and its queue, or `None` (with a notice) in CI or on a host without an adapter.
fn device(test: &str) -> Option<(wgpu::Device, wgpu::Queue)> {
    if std::env::var_os("CI").is_some() {
        eprintln!("skipping {test}: GPU tests are local-only until verification 3 (decision 9)");
        return None;
    }
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let Ok(Ok(adapter)) = crate::backend::block_on(
        instance.request_adapter(&wgpu::RequestAdapterOptions::default()),
    ) else {
        eprintln!("skipping {test}: no GPU adapter on this host");
        return None;
    };
    let opened = crate::backend::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
        .expect("the adapter answers")
        .expect("the adapter opens a device");
    Some(opened)
}

/// Runs the shader on `input` and returns its output.
fn run(
    (device, queue): &(wgpu::Device, wgpu::Queue),
    rewriter: &IndexRewrite,
    rw: Rewrite,
    n: u32,
    first: u32,
    input: &[u8],
    max_groups: u32,
) -> Vec<u32> {
    use wgpu::util::DeviceExt;
    let count = output_count(rw.mode, n).expect("fits");
    // Each block and the output at an offset of 256, the alignment every device allows.
    let mut block = vec![0; 256];
    block.extend_from_slice(&params(rw, count, first, n));
    let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: &block,
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let mut padded = input.to_vec();
    padded.resize(input.len().next_multiple_of(4).max(4), 0);
    let input_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: &padded,
        usage: wgpu::BufferUsages::STORAGE,
    });
    let bytes = u64::from(count) * 4;
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 256 + bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let read = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    let input_size = u64::try_from(padded.len()).expect("small");
    rewriter.encode(
        device,
        &mut encoder,
        Run {
            params: (&params_buffer, 256),
            input: (rw.source != Source::Sequential).then_some((&input_buffer, 0, input_size)),
            output: (&output, 256, count),
            max_groups,
        },
    );
    encoder.copy_buffer_to_buffer(&output, 256, &read, 0, bytes);
    queue.submit([encoder.finish()]);
    read.map_async(wgpu::MapMode::Read, .., |r| r.expect("maps"));
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(std::time::Duration::from_secs(10)),
        })
        .expect("the rewrite finishes");
    let view = read.get_mapped_range(..).expect("mapped");
    view.chunks(4)
        .map(|w| u32::from_le_bytes(w.try_into().expect("a word")))
        .collect()
}

/// Every mode from every source, with restart on and off, the first index off a word boundary,
/// inputs holding each format's restart value, and one workgroup for more indices than it has
/// invocations, so the shader's loop runs.
#[test]
fn the_shader_writes_what_the_model_says() {
    crate::backend::tests::bounded(|| {
        let Some(gpu) = device("index_rewrite") else {
            return;
        };
        let rewriter = IndexRewrite::new(&gpu.0).expect("the rewrite's pipeline is made");
        let modes = [
            (Mode::List, 150),
            (Mode::Quads, 103),
            (Mode::QuadStrip, 105),
            (Mode::Fan, 70),
            (Mode::Loop, 130),
        ];
        let sources = [
            (Source::Sequential, 7),
            (Source::U8, 3),
            (Source::U16, 2),
            (Source::U32, 4),
        ];
        for (mode, n) in modes {
            for (source, first) in sources {
                for restart in [false, true] {
                    let rw = Rewrite {
                        mode,
                        source,
                        restart,
                    };
                    let size = source.index_bytes().max(1);
                    let len = usize::try_from(first + n * size).expect("small");
                    // Distinct values, with every fourth index all ones.
                    let mut input: Vec<u8> = (0..len)
                        .map(|i| u8::try_from((i * 37 + 11) % 251).expect("byte"))
                        .collect();
                    for p in (0..n).step_by(4) {
                        let at = usize::try_from(first + p * size).expect("small");
                        let width = usize::try_from(size).expect("small");
                        input[at..at + width].fill(0xFF);
                    }
                    let got = run(&gpu, &rewriter, rw, n, first, &input, 1);
                    assert_eq!(got, model(rw, n, first, &input), "{rw:?}");
                }
            }
        }
    });
}
