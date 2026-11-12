//! Recording backend tests that look inside: memory held only for what was written, the order
//! of deferred destroys, and the text of every command in the snapshot.

use super::snapshot;
use super::{RecordingBackend, State};
use crate::backend::Backend;
use crate::caps::{
    Caps, DynamicState, Features, FormatFeatures, FormatTable, Limits, SampleCounts, StageSupport,
};
use crate::command::{
    BlitDesc, BufferSlice, ClearDesc, ClearValue, Command, CommandList, CopyDesc, CopyRegion,
    DispatchDesc, DispatchParams, IndexBufferBinding, IndexFormat, PresentDesc, PresentTransform,
    Rect, RenderTargets, TextureRegion,
};
use crate::format::PixelFormat;
use crate::id::{BufferId, FenceValue, PipelineId, ReadbackId, ResourceId, TextureId, ViewId};
use crate::resource::{
    BufferDesc, BufferUsage, Extent3d, FilterMode, Origin3d, Swizzle, SwizzleSource, TextureDesc,
    TextureDimension, TextureUsage, ViewDesc, ViewDimension,
};
use crate::shader::{ShaderDesc, ShaderInterface, ShaderSource, ShaderStage};

fn caps() -> Caps {
    Caps {
        limits: Limits {
            max_buffer_size: 1 << 16,
            max_staging_size: 1 << 12,
            max_texture_dimension_1d: 256,
            max_texture_dimension_2d: 256,
            max_texture_dimension_3d: 64,
            max_texture_array_layers: 8,
            max_sampler_anisotropy: 16,
            max_vertex_attributes: 16,
            max_vertex_buffers: 8,
            max_vertex_buffer_stride: 2048,
            max_color_attachments: 8,
            max_viewports: 1,
            max_bind_groups: 4,
            max_bindings_per_bind_group: 16,
            max_compute_workgroups_per_dimension: 65_535,
            max_uniform_buffers_per_stage: 12,
            max_storage_buffers_per_stage: 8,
            max_sampled_textures_per_stage: 16,
            max_storage_textures_per_stage: 4,
            max_samplers_per_stage: 16,
        },
        features: Features::ALL,
        stages: StageSupport::ALL,
        dynamic_state: DynamicState::ALL,
        sample_counts: SampleCounts::X1,
        view_swizzle: true,
        formats: FormatTable::uniform(FormatFeatures::ALL),
    }
}

fn texture_desc(mips: u32) -> TextureDesc {
    TextureDesc {
        dimension: TextureDimension::D2,
        format: PixelFormat::R8Unorm,
        size: Extent3d {
            width: 4,
            height: 4,
            depth_or_layers: 1,
        },
        mip_levels: mips,
        sample_count: 1,
        usage: TextureUsage::ALL,
    }
}

fn allocated(state: &State, t: TextureId) -> Vec<bool> {
    state
        .texture(t)
        .expect("live")
        .levels
        .iter()
        .map(super::store::Store::is_allocated)
        .collect()
}

#[test]
fn only_written_levels_hold_memory() {
    let b = RecordingBackend::new(caps(), 0);
    let t = b.create_texture(&texture_desc(3)).expect("valid");
    let v = b
        .create_view(
            t,
            &ViewDesc {
                format: PixelFormat::R8Unorm,
                dimension: ViewDimension::D2,
                base_mip: 0,
                mip_count: 3,
                base_layer: 0,
                layer_count: 1,
                swizzle: Swizzle::IDENTITY,
            },
        )
        .expect("valid");
    let vs = b
        .create_shader(ShaderDesc {
            stage: ShaderStage::Compute,
            entry: "main".to_owned(),
            source: ShaderSource::Msl(String::new()),
            interface: ShaderInterface::default(),
        })
        .expect("valid");
    let p = b
        .create_compute_pipeline(crate::pipeline::ComputePipelineDesc { shader: vs })
        .expect("valid");
    // Forgetting what was never written takes no memory.
    b.submit(CommandList {
        commands: vec![Command::Dispatch(DispatchDesc {
            pipeline: p,
            bindings: vec![crate::command::Binding {
                group: 0,
                binding: 0,
                resource: crate::command::BindingResource::View(v),
            }],
            params: DispatchParams::Direct { x: 1, y: 1, z: 1 },
        })],
    })
    .expect("valid");
    assert_eq!(allocated(&b.state(), t), vec![false; 3]);
    b.submit(CommandList {
        commands: vec![Command::Upload {
            from: crate::command::StagingSlice::from_bytes(vec![1; 4]),
            to: CopyRegion::Texture(TextureRegion {
                texture: t,
                mip: 1,
                origin: Origin3d::default(),
                extent: Extent3d {
                    width: 2,
                    height: 2,
                    depth_or_layers: 1,
                },
            }),
        }],
    })
    .expect("valid");
    assert_eq!(allocated(&b.state(), t), vec![false, true, false]);
}

/// The op `plan::check` makes of `command` on `b`'s current state.
fn planned(b: &RecordingBackend, command: Command) -> String {
    let list = CommandList {
        commands: vec![command],
    };
    let state = b.state();
    let ops = super::plan::check(&state, &caps(), &list).expect("valid");
    format!("{ops:?}")
}

/// Each guard on buffer copies of combined depth-stencil and `X8_D24` textures, seen in the
/// planned op itself: through the backend no byte of such a texture is ever known, so the
/// readback and copy-out guards could not be told apart from the upload and copy-in ones.
#[test]
fn buffer_copies_of_depth_stencil_textures_are_planned_unknown() {
    let b = RecordingBackend::new(caps(), 0);
    let buf = b
        .create_buffer(&BufferDesc {
            size: 64,
            usage: BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
        })
        .expect("valid");
    let texture = |format| {
        b.create_texture(&TextureDesc {
            format,
            mip_levels: 1,
            size: Extent3d {
                width: 2,
                height: 1,
                depth_or_layers: 1,
            },
            ..texture_desc(1)
        })
        .expect("valid")
    };
    let region = |t| {
        CopyRegion::Texture(TextureRegion {
            texture: t,
            mip: 0,
            origin: Origin3d::default(),
            extent: Extent3d {
                width: 2,
                height: 1,
                depth_or_layers: 1,
            },
        })
    };
    for (format, exact) in [
        (PixelFormat::D24UnormS8Uint, false),
        (PixelFormat::S8UintD24Unorm, false),
        (PixelFormat::D32FloatS8Uint, false),
        (PixelFormat::X8D24Unorm, false),
        (PixelFormat::D32Float, true),
        (PixelFormat::S8Uint, true),
    ] {
        let (t, u) = (texture(format), texture(format));
        let size = 2 * u64::from(format.bytes_per_block());
        let range = CopyRegion::Buffer(BufferSlice {
            buffer: buf,
            offset: 0,
            size,
        });
        let len = usize::try_from(size).expect("small");
        let upload = planned(
            &b,
            Command::Upload {
                from: crate::command::StagingSlice::from_bytes(vec![1; len]),
                to: region(t),
            },
        );
        let readback = planned(
            &b,
            Command::Readback {
                from: region(t),
                id: ReadbackId::new(0, 0),
            },
        );
        let copy = |src, dst| planned(&b, Command::Copy(CopyDesc { src, dst }));
        let out = copy(region(t), range);
        let into = copy(range, region(t));
        let between = copy(region(t), region(u));
        assert_eq!(upload.starts_with("[Upload"), exact, "{format}: {upload}");
        assert_eq!(upload.starts_with("[Forget"), !exact, "{format}: {upload}");
        assert_eq!(
            readback.starts_with("[Readback { from: None"),
            !exact,
            "{format}: {readback}"
        );
        assert_eq!(
            readback.starts_with("[Readback { from: Some"),
            exact,
            "{format}: {readback}"
        );
        for (what, op) in [("out", &out), ("into", &into)] {
            assert_eq!(op.starts_with("[Copy"), exact, "{format} {what}: {op}");
            assert_eq!(op.starts_with("[Forget"), !exact, "{format} {what}: {op}");
        }
        // Between two textures of one format, bytes and known flags move as they are.
        assert!(between.starts_with("[Copy"), "{format}: {between}");
    }
}

#[test]
fn pending_destroys_are_freed_in_call_order_once_their_value_passes() {
    let b = RecordingBackend::new(caps(), 0);
    let desc = BufferDesc {
        size: 4,
        usage: BufferUsage::COPY_DST,
    };
    let ids: Vec<BufferId> = (0..3)
        .map(|_| b.create_buffer(&desc).expect("valid"))
        .collect();
    b.destroy(ResourceId::Buffer(ids[2]), FenceValue::new(1));
    b.destroy(ResourceId::Buffer(ids[0]), FenceValue::new(1));
    b.destroy(ResourceId::Buffer(ids[1]), FenceValue::new(2));
    b.submit(CommandList::new()).expect("valid");
    let text = b.snapshot();
    let freed: Vec<&str> = text.lines().filter(|l| l.starts_with("freed")).collect();
    assert_eq!(freed, vec!["freed b2", "freed b0"]);
    assert_eq!(b.state().pending.len(), 1);
    b.submit(CommandList::new()).expect("valid");
    assert_eq!(b.state().pending, Vec::new());
    assert!(b.snapshot().ends_with("submit fence 2:\nfreed b1\n"));
    // A reused slot prints its generation.
    let again = b.create_buffer(&desc).expect("valid");
    assert_eq!((again.index(), again.generation()), (1, 1));
    assert!(
        b.snapshot()
            .ends_with("create b1@1: buffer 4 bytes, usage COPY_DST\n")
    );
}

#[test]
fn an_empty_log_is_an_empty_snapshot() {
    assert_eq!(RecordingBackend::new(caps(), 0).snapshot(), "");
}

#[test]
fn shader_lines_name_their_code() {
    let module = naga::Module::default();
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&module)
    .expect("an empty module is valid");
    let desc = |source| ShaderDesc {
        stage: ShaderStage::Fragment,
        entry: "fs".to_owned(),
        source,
        interface: ShaderInterface::default(),
    };
    assert_eq!(
        snapshot::shader(&desc(ShaderSource::Naga {
            module: Box::new(module),
            info: Box::new(info),
        })),
        "shader fragment \"fs\", naga module (0 functions, 0 entry points), resources [], \
         inputs [], outputs []"
    );
    assert_eq!(
        snapshot::shader(&desc(ShaderSource::Msl("abc".to_owned()))),
        "shader fragment \"fs\", MSL 3 bytes, resources [], inputs [], outputs []"
    );
    assert_eq!(
        snapshot::shader(&desc(ShaderSource::SpirV(vec![1, 2]))),
        "shader fragment \"fs\", SPIR-V 2 words, resources [], inputs [], outputs []"
    );
}

#[test]
fn every_command_has_a_line() {
    let buf = BufferId::new(1, 0);
    let tex = TextureId::new(2, 3);
    let view = ViewId::new(4, 0);
    let rect = Rect {
        x: 1,
        y: 2,
        width: 3,
        height: 4,
    };
    let slice = BufferSlice {
        buffer: buf,
        offset: 8,
        size: 4,
    };
    let region = CopyRegion::Texture(TextureRegion {
        texture: tex,
        mip: 1,
        origin: Origin3d { x: 1, y: 2, z: 3 },
        extent: Extent3d {
            width: 4,
            height: 5,
            depth_or_layers: 6,
        },
    });
    let mut targets = RenderTargets::default();
    targets.colors[2] = Some(view);
    targets.depth_stencil = Some(ViewId::new(5, 1));
    for (command, line) in [
        (
            Command::Copy(CopyDesc {
                src: region,
                dst: CopyRegion::Buffer(slice),
            }),
            "copy t2@3 mip 1 at (1,2,3) size 4x5x6 to b1[8..12]".to_owned(),
        ),
        (
            Command::Blit(BlitDesc {
                src: view,
                src_rect: rect,
                dst: view,
                dst_rect: rect,
                filter: FilterMode::Linear,
            }),
            "blit v4 (1,2 3x4) to v4 (1,2 3x4) Linear".to_owned(),
        ),
        (
            Command::SetRenderTargets(targets),
            "set render targets colour [2=v4] depth/stencil v5@1".to_owned(),
        ),
        (
            Command::Dispatch(DispatchDesc {
                pipeline: PipelineId::new(6, 0),
                bindings: vec![crate::command::Binding {
                    group: 1,
                    binding: 2,
                    resource: crate::command::BindingResource::Buffer(slice),
                }],
                params: DispatchParams::Indirect {
                    buffer: buf,
                    offset: 4,
                },
            }),
            "dispatch p6 indirect from b1 at 4, bindings [1:2=b1[8..12]]".to_owned(),
        ),
        (
            Command::Readback {
                from: CopyRegion::Buffer(slice),
                id: ReadbackId::new(9, 2),
            },
            "readback b1[8..12] as r9@2".to_owned(),
        ),
        (
            Command::Present(PresentDesc {
                view,
                src_rect: rect,
                transform: PresentTransform::Identity,
                frame: 3,
                dump: false,
            }),
            "present v4 (1,2 3x4) Identity frame 3".to_owned(),
        ),
    ] {
        assert_eq!(snapshot::command(&command), line);
    }
}

#[test]
fn clear_lines() {
    let view = ViewId::new(4, 0);
    let clear = |value| {
        Command::Clear(ClearDesc {
            target: view,
            value,
            rect: None,
        })
    };
    for (command, line) in [
        (
            clear(ClearValue::Sint([-1, 0, 1, 2])),
            "clear v4 to sint [-1, 0, 1, 2] in the whole view".to_owned(),
        ),
        (
            clear(ClearValue::Uint([1, 2, 3, 4])),
            "clear v4 to uint [1, 2, 3, 4] in the whole view".to_owned(),
        ),
        (
            clear(ClearValue::Depth(0.5)),
            "clear v4 to depth 0.5 in the whole view".to_owned(),
        ),
        (
            clear(ClearValue::Stencil(7)),
            "clear v4 to stencil 7 in the whole view".to_owned(),
        ),
        (
            clear(ClearValue::DepthStencil {
                depth: 1.0,
                stencil: 2,
            }),
            "clear v4 to depth 1.0 stencil 2 in the whole view".to_owned(),
        ),
    ] {
        assert_eq!(snapshot::command(&command), line);
    }
}

#[test]
fn draw_lines() {
    let buf = BufferId::new(1, 0);
    let slice = BufferSlice {
        buffer: buf,
        offset: 8,
        size: 4,
    };
    let mut draw = crate::command::DrawDesc {
        pipeline: PipelineId::new(6, 0),
        bindings: Vec::new(),
        vertex_buffers: Vec::new(),
        index_buffer: Some(IndexBufferBinding {
            slice,
            format: IndexFormat::U32,
        }),
        dynamic: crate::command::DynamicValues {
            viewports: Vec::new(),
            scissors: Vec::new(),
            blend_constant: [0.0; 4],
            stencil_reference: 0,
            stencil_read_mask: 0,
            stencil_write_mask: 0,
            depth_bias: crate::pipeline::DepthBias::default(),
            line_width: 1.0,
        },
        params: crate::command::DrawParams::Indirect {
            buffer: buf,
            offset: 16,
            draw_count: 2,
            stride: 20,
            indexed: true,
        },
    };
    let dynamic = format!("{:?}", draw.dynamic);
    assert_eq!(
        snapshot::command(&Command::Draw(draw.clone())),
        format!(
            "draw p6 indirect from b1 at 16, 2 draws, stride 20, indexed true, bindings [], \
             vertex buffers [], index b1[8..12] U32, dynamic {dynamic}"
        )
    );
    draw.index_buffer = None;
    draw.params = crate::command::DrawParams::Direct {
        vertex_count: 1,
        instance_count: 2,
        first_vertex: 3,
        first_instance: 4,
    };
    assert_eq!(
        snapshot::command(&Command::Draw(draw)),
        format!(
            "draw p6 Direct {{ vertex_count: 1, instance_count: 2, first_vertex: 3, \
             first_instance: 4 }}, bindings [], vertex buffers [], index none, dynamic {dynamic}"
        )
    );
}

#[test]
fn view_lines_spell_the_swizzle() {
    let swizzle = Swizzle {
        r: SwizzleSource::B,
        g: SwizzleSource::Zero,
        b: SwizzleSource::One,
        a: SwizzleSource::R,
    };
    let desc = ViewDesc {
        format: PixelFormat::R8Unorm,
        dimension: ViewDimension::D2Array,
        base_mip: 1,
        mip_count: 2,
        base_layer: 3,
        layer_count: 4,
        swizzle,
    };
    assert_eq!(
        snapshot::view(TextureId::new(0, 0), &desc),
        "view of t0 as R8_UNORM D2Array, mips 1+2, layers 3+4, swizzle B01R"
    );
    let ga = Swizzle {
        r: SwizzleSource::G,
        g: SwizzleSource::A,
        ..Swizzle::IDENTITY
    };
    assert_eq!(
        snapshot::view(
            TextureId::new(0, 0),
            &ViewDesc {
                swizzle: ga,
                ..desc
            }
        ),
        "view of t0 as R8_UNORM D2Array, mips 1+2, layers 3+4, swizzle GABA"
    );
}

#[test]
fn the_last_fence_value_refuses_another_list() {
    let b = RecordingBackend::new(caps(), 0);
    b.state().completed = FenceValue::new(u64::MAX - 1);
    assert_eq!(b.submit(CommandList::new()), Ok(FenceValue::new(u64::MAX)));
    assert_eq!(
        b.submit(CommandList::new()),
        Err(crate::error::RenderError::Overflow {
            what: "fence value"
        })
    );
    assert_eq!(b.completed(), FenceValue::new(u64::MAX));
}

#[test]
fn a_texture_too_large_to_address_is_an_error_not_an_allocation() {
    let mut huge = caps();
    huge.limits.max_texture_dimension_2d = u32::MAX;
    huge.limits.max_texture_array_layers = u32::MAX;
    let b = RecordingBackend::new(huge, 0);
    let desc = TextureDesc {
        format: PixelFormat::R32G32B32A32Float,
        size: Extent3d {
            width: u32::MAX,
            height: u32::MAX,
            depth_or_layers: u32::MAX,
        },
        ..texture_desc(1)
    };
    assert_eq!(
        b.create_texture(&desc),
        Err(crate::error::RenderError::Overflow {
            what: "recorded contents offset"
        })
    );
    // The largest that fits in 64 bits is accepted, and holds no memory until written.
    let big = TextureDesc {
        size: Extent3d {
            width: 1 << 16,
            height: 1 << 16,
            depth_or_layers: 1 << 16,
        },
        ..desc
    };
    let t = b.create_texture(&big).expect("valid");
    assert_eq!(allocated(&b.state(), t), vec![false]);
    // 2^48 texels of 16 bytes.
    assert_eq!(b.budget().used, 1 << 52);
}
