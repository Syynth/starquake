//! The submit-side check, with no GPU: each command's step, and each refusal in order.

use super::*;
use crate::fixtures::{formats, test_caps, texture, view};
use crate::rules;
use starquake_render::command::{
    BlitDesc, BufferSlice, ClearDesc, CopyDesc, DispatchDesc, DispatchParams, Marker,
    PresentTransform,
};
use starquake_render::resource::{FilterMode, ViewDimension};
use starquake_render::{BufferUsage, PipelineId, StagingSlice};

/// [`super::check`] with no pipelines, on the test device: these tests make no draws (their
/// check is `crate::draw::tests`').
fn check(
    tables: &Tables,
    caps: &Caps,
    depth_copies: bool,
    taken: &BTreeSet<ReadbackId>,
    list: CommandList,
) -> Result<Checked, RenderError> {
    let draws = crate::fixtures::draw_limits(caps);
    let rules = Rules {
        caps,
        depth_copies,
        draws: &draws,
    };
    super::check(tables, &Registry::default(), &rules, taken, list)
}

/// Tables with a buffer, an RGBA8 render target, a BC1 texture, a 3D texture, a D32 texture,
/// and a view of each texture.
struct World {
    tables: Tables,
    buffer: BufferId,
    other: BufferId,
    rgba: TextureId,
    srgb: TextureId,
    rgba_view: ViewId,
    srgb_view: ViewId,
    bc: TextureId,
    deep_view: ViewId,
    depth_view: ViewId,
    array_view: ViewId,
}

fn add_texture(tables: &mut Tables, desc: TextureDesc) -> TextureId {
    let host =
        rules::check_texture(&test_caps(), &formats(), wgpu::Features::all(), &desc).expect("ok");
    let (i, g) = tables
        .textures
        .insert(TextureRec {
            desc,
            host,
            bytes: 0,
        })
        .expect("room");
    TextureId::new(i, g)
}

fn add_view(tables: &mut Tables, texture: TextureId, desc: ViewDesc) -> ViewId {
    let (i, g) = tables
        .views
        .insert(ViewRec { texture, desc })
        .expect("room");
    ViewId::new(i, g)
}

fn world() -> World {
    let mut tables = Tables::default();
    let copy = BufferUsage::COPY_SRC.union(BufferUsage::COPY_DST);
    let (i, g) = tables
        .buffers
        .insert(BufferDesc {
            size: 4096,
            usage: copy,
        })
        .expect("room");
    let buffer = BufferId::new(i, g);
    let (i, g) = tables
        .buffers
        .insert(BufferDesc {
            size: 4096,
            usage: copy,
        })
        .expect("room");
    let other = BufferId::new(i, g);
    let all = TextureUsage::ALL.difference(TextureUsage::STORAGE);
    let rgba = add_texture(&mut tables, texture(PixelFormat::A8B8G8R8Unorm, 8, 4, all));
    let rgba_view = add_view(
        &mut tables,
        rgba,
        view(PixelFormat::A8B8G8R8Unorm, ViewDimension::D2),
    );
    let srgb_view = add_view(
        &mut tables,
        rgba,
        view(PixelFormat::A8B8G8R8Srgb, ViewDimension::D2),
    );
    let srgb = add_texture(&mut tables, texture(PixelFormat::A8B8G8R8Srgb, 8, 4, all));
    let copies = TextureUsage::COPY_SRC.union(TextureUsage::COPY_DST);
    let bc = add_texture(
        &mut tables,
        texture(PixelFormat::Bc1RgbaUnorm, 6, 6, copies),
    );
    let deep = add_texture(
        &mut tables,
        TextureDesc {
            dimension: TextureDimension::D3,
            size: Extent3d {
                width: 4,
                height: 4,
                depth_or_layers: 3,
            },
            ..texture(PixelFormat::A8B8G8R8Unorm, 4, 4, all)
        },
    );
    let deep_view = add_view(
        &mut tables,
        deep,
        view(PixelFormat::A8B8G8R8Unorm, ViewDimension::D3),
    );
    let depth = add_texture(&mut tables, texture(PixelFormat::D32Float, 8, 4, all));
    let depth_view = add_view(
        &mut tables,
        depth,
        view(PixelFormat::D32Float, ViewDimension::D2),
    );
    let array = add_texture(
        &mut tables,
        TextureDesc {
            size: Extent3d {
                width: 4,
                height: 4,
                depth_or_layers: 5,
            },
            ..texture(PixelFormat::A8B8G8R8Unorm, 4, 4, all)
        },
    );
    let array_view = add_view(
        &mut tables,
        array,
        ViewDesc {
            base_layer: 1,
            layer_count: 3,
            ..view(PixelFormat::A8B8G8R8Unorm, ViewDimension::D2Array)
        },
    );
    World {
        tables,
        buffer,
        other,
        rgba,
        srgb,
        rgba_view,
        srgb_view,
        bc,
        deep_view,
        depth_view,
        array_view,
    }
}

/// The clear a step is.
fn as_clear(step: &Step) -> ClearStep {
    match step {
        Step::Clear(clear) => Some(*clear),
        _ => None,
    }
    .expect("a clear")
}

/// The format and dump read of a present step.
fn as_present(step: &Step) -> (PixelFormat, Option<(TexelBox, Rows)>) {
    match step {
        Step::Present { format, read, .. } => Some((*format, *read)),
        _ => None,
    }
    .expect("a present")
}

fn run(w: &World, commands: Vec<Command>) -> Result<Checked, RenderError> {
    check(
        &w.tables,
        &test_caps(),
        true,
        &BTreeSet::new(),
        CommandList { commands },
    )
}

fn steps(w: &World, commands: Vec<Command>) -> Vec<Step> {
    run(w, commands).expect("accepted").steps
}

fn slice(buffer: BufferId, offset: u64, size: u64) -> CopyRegion {
    CopyRegion::Buffer(BufferSlice {
        buffer,
        offset,
        size,
    })
}

fn region(texture: TextureId, x: u32, y: u32, w: u32, h: u32) -> CopyRegion {
    CopyRegion::Texture(TextureRegion {
        texture,
        mip: 0,
        origin: Origin3d { x, y, z: 0 },
        extent: Extent3d {
            width: w,
            height: h,
            depth_or_layers: 1,
        },
    })
}

fn upload(bytes: usize, to: CopyRegion) -> Command {
    Command::Upload {
        from: StagingSlice::from_bytes(
            (0..bytes)
                .map(|i| u8::try_from(i % 251).expect("small"))
                .collect(),
        ),
        to,
    }
}

#[test]
fn uploads_become_queue_writes() {
    let w = world();
    let s = steps(&w, vec![upload(8, slice(w.buffer, 16, 8))]);
    assert_eq!(
        s[0],
        Step::WriteBuffer {
            buffer: w.buffer,
            offset: 16,
            bytes: (0..8).collect(),
        }
    );
    // A BC1 region that reaches the 6x6 level's edge: one whole block from (4, 4).
    let s = steps(&w, vec![upload(8, region(w.bc, 4, 4, 2, 2))]);
    assert_eq!(
        s[0],
        Step::WriteTexture {
            at: TexelBox {
                texture: w.bc,
                mip: 0,
                origin: Origin3d { x: 4, y: 4, z: 0 },
                extent: Extent3d {
                    width: 4,
                    height: 4,
                    depth_or_layers: 1
                },
            },
            rows: Rows {
                row_bytes: 8,
                rows: 1,
                images: 1
            },
            bytes: (0..8).collect(),
        }
    );
}

#[test]
fn uploads_check_shared_rules_before_wgpu_rules() {
    let w = world();
    assert_eq!(
        run(&w, vec![upload(6, slice(w.buffer, 2, 6))]).err(),
        Some(RenderError::Unsupported(rules::BUFFER_ALIGNMENT))
    );
    // The size is wrong too: the shared check is reported first.
    assert_eq!(
        run(&w, vec![upload(5, slice(w.buffer, 2, 6))]).err(),
        Some(RenderError::SizeMismatch {
            what: "staging slice",
            expected: 6,
            actual: 5
        })
    );
    let stale = BufferId::new(w.buffer.index(), w.buffer.generation() + 1);
    assert_eq!(
        run(&w, vec![upload(8, slice(stale, 0, 8))]).err(),
        Some(RenderError::Invalid(NO_BUFFER))
    );
}

#[test]
fn buffer_copies_become_copy_steps() {
    let w = world();
    let copy = |src, dst| Command::Copy(CopyDesc { src, dst });
    let s = steps(
        &w,
        vec![copy(slice(w.buffer, 0, 64), slice(w.other, 64, 64))],
    );
    assert_eq!(
        s[0],
        Step::CopyBuffer {
            src: w.buffer,
            src_offset: 0,
            dst: w.other,
            dst_offset: 64,
            size: 64
        }
    );
    assert_eq!(
        run(
            &w,
            vec![copy(slice(w.buffer, 0, 64), slice(w.buffer, 64, 64))]
        )
        .err(),
        Some(RenderError::Unsupported(rules::SAME_BUFFER))
    );
}

#[test]
fn texture_copies_become_their_kind_of_step() {
    let w = world();
    let copy = |src, dst| Command::Copy(CopyDesc { src, dst });
    let rows = Rows {
        row_bytes: 16,
        rows: 2,
        images: 1,
    };
    let at = TexelBox {
        texture: w.rgba,
        mip: 0,
        origin: Origin3d { x: 4, y: 2, z: 0 },
        extent: Extent3d {
            width: 4,
            height: 2,
            depth_or_layers: 1,
        },
    };
    let s = steps(
        &w,
        vec![copy(slice(w.buffer, 128, 32), region(w.rgba, 4, 2, 4, 2))],
    );
    assert_eq!(
        s[0],
        Step::BufferToTexture {
            buffer: w.buffer,
            offset: 128,
            rows,
            at
        }
    );
    let s = steps(
        &w,
        vec![copy(region(w.rgba, 4, 2, 4, 2), slice(w.buffer, 128, 32))],
    );
    assert_eq!(
        s[0],
        Step::TextureToBuffer {
            at,
            buffer: w.buffer,
            offset: 128,
            rows
        }
    );
    let s = steps(
        &w,
        vec![copy(region(w.srgb, 0, 0, 4, 2), region(w.rgba, 4, 2, 4, 2))],
    );
    assert_eq!(
        s[0],
        Step::TextureToTexture {
            src: TexelBox {
                texture: w.srgb,
                origin: Origin3d::default(),
                ..at
            },
            dst: at
        }
    );
    assert_eq!(
        run(
            &w,
            vec![copy(region(w.rgba, 0, 0, 4, 2), region(w.rgba, 4, 2, 4, 2))]
        )
        .err(),
        Some(RenderError::Unsupported(rules::SAME_LEVEL))
    );
    assert_eq!(
        run(
            &w,
            vec![copy(slice(w.buffer, 130, 32), region(w.rgba, 4, 2, 4, 2))]
        )
        .err(),
        Some(RenderError::Unsupported(rules::BLOCK_OFFSET))
    );
    assert_eq!(
        run(&w, vec![copy(slice(w.buffer, 0, 6), slice(w.other, 64, 6))]).err(),
        Some(RenderError::Unsupported(rules::BUFFER_ALIGNMENT))
    );
    assert_eq!(
        run(&w, vec![copy(slice(w.buffer, 0, 8), slice(w.other, 66, 8))]).err(),
        Some(RenderError::Unsupported(rules::BUFFER_ALIGNMENT)),
        "the destination side is checked too"
    );
    assert_eq!(
        run(
            &w,
            vec![copy(slice(w.buffer, 0, 6), slice(w.buffer, 64, 6))]
        )
        .err(),
        Some(RenderError::Unsupported(rules::BUFFER_ALIGNMENT)),
        "alignment before the same-buffer rule"
    );
}

#[test]
fn blits_wait_for_their_slice_and_dispatches_need_a_pipeline() {
    let w = world();
    let blit = Command::Blit(BlitDesc {
        src: w.rgba_view,
        src_rect: Rect::default(),
        dst: w.rgba_view,
        dst_rect: Rect::default(),
        filter: FilterMode::Nearest,
    });
    assert_eq!(
        run(&w, vec![blit]).err(),
        Some(RenderError::Unsupported(BLITS_NOT_YET))
    );
    let dispatch = Command::Dispatch(DispatchDesc {
        pipeline: PipelineId::new(0, 0),
        bindings: Vec::new(),
        params: DispatchParams::Direct { x: 1, y: 1, z: 1 },
    });
    assert_eq!(
        run(&w, vec![dispatch]).err(),
        Some(RenderError::Invalid(validate::NO_PIPELINE))
    );
    // A good command before a refused one changes nothing: the whole list is refused.
    let marker = Command::Marker(Marker { submit: 1, draw: 2 });
    let blit = Command::Blit(BlitDesc {
        src: w.rgba_view,
        src_rect: Rect::default(),
        dst: w.rgba_view,
        dst_rect: Rect::default(),
        filter: FilterMode::Nearest,
    });
    assert!(run(&w, vec![marker, upload(8, slice(w.buffer, 0, 8)), blit]).is_err());
}

#[test]
fn clears_name_their_layers_area_and_value() {
    let w = world();
    let clear = |target, rect, value| Command::Clear(ClearDesc::unmasked(target, value, rect));
    let red = ClearValue::Float([1.0, 0.0, 0.0, 1.0]);
    let full = Rect {
        x: 0,
        y: 0,
        width: 8,
        height: 4,
    };
    let part = Rect { width: 7, ..full };
    let s = steps(
        &w,
        vec![
            clear(w.srgb_view, Some(full), red),
            clear(w.rgba_view, Some(part), red),
            clear(w.deep_view, None, red),
            clear(w.array_view, None, red),
        ],
    );
    let whole = as_clear(&s[0]);
    assert_eq!(
        whole.rect, None,
        "a rectangle covering the level is a whole clear"
    );
    assert_eq!(
        whole.format,
        wgpu::TextureFormat::Rgba8UnormSrgb,
        "the view's format"
    );
    assert_eq!(whole.layers, Layers::Array { first: 0, count: 1 });
    assert_eq!(whole.texture, w.rgba);
    let partial = as_clear(&s[1]);
    assert_eq!(partial.rect, Some(part));
    let deep = as_clear(&s[2]);
    assert_eq!(deep.layers, Layers::Slices { count: 3 });
    let array = as_clear(&s[3]);
    assert_eq!(array.layers, Layers::Array { first: 1, count: 3 });
    // How each is done (`crate::clear::clear_how`): a pass, the scratch, passes again.
    let hows: Vec<_> = s.iter().map(|step| as_clear(step).how).collect();
    assert_eq!(
        hows,
        [
            ClearHow::Pass,
            ClearHow::Scratch,
            ClearHow::Pass,
            ClearHow::Pass
        ]
    );
    assert_eq!(partial.size, (8, 4));
    // A partial clear of a depth view is accepted and drawn (owner's ruling of 2026-11-19).
    let depth_part = Rect { width: 7, ..full };
    let drawn = as_clear(
        &steps(
            &w,
            vec![clear(
                w.depth_view,
                Some(depth_part),
                ClearValue::Depth(0.5),
            )],
        )[0],
    );
    assert_eq!(drawn.rect, Some(depth_part));
    assert!(matches!(drawn.how, ClearHow::Draw(_)), "{:?}", drawn.how);
    let depth = as_clear(&steps(&w, vec![clear(w.depth_view, None, ClearValue::Depth(0.5))])[0]);
    assert_eq!(depth.how, ClearHow::Pass);
    assert_eq!(
        depth.ops,
        ClearOps::DepthStencil {
            depth: Some(0.5),
            stencil: None
        }
    );
}

#[test]
fn clear_values_reach_wgpu_exactly() {
    let color = |r, g, b, a| ClearOps::Color(wgpu::Color { r, g, b, a });
    assert_eq!(
        clear_ops(ClearValue::Uint([u32::MAX, 0, 1, 7])),
        color(4_294_967_295.0, 0.0, 1.0, 7.0)
    );
    assert_eq!(
        clear_ops(ClearValue::Sint([i32::MIN, i32::MAX, -1, 0])),
        color(-2_147_483_648.0, 2_147_483_647.0, -1.0, 0.0)
    );
    assert_eq!(
        clear_ops(ClearValue::Float([0.25, 0.5, 0.75, 1.0])),
        color(0.25, 0.5, 0.75, 1.0)
    );
    assert_eq!(
        clear_ops(ClearValue::Stencil(9)),
        ClearOps::DepthStencil {
            depth: None,
            stencil: Some(9)
        }
    );
    assert_eq!(
        clear_ops(ClearValue::DepthStencil {
            depth: 0.25,
            stencil: 3
        }),
        ClearOps::DepthStencil {
            depth: Some(0.25),
            stencil: Some(3)
        }
    );
}

#[test]
fn readback_ids_are_used_once() {
    let w = world();
    let read = |n| Command::Readback {
        from: slice(w.buffer, 1, 3),
        id: ReadbackId::new(n, 0),
    };
    let checked = run(&w, vec![read(1), read(2)]).expect("two ids");
    assert_eq!(
        checked.readbacks,
        vec![ReadbackId::new(1, 0), ReadbackId::new(2, 0)]
    );
    assert_eq!(
        checked.steps[0],
        Step::Readback {
            id: ReadbackId::new(1, 0),
            from: ReadFrom::Buffer {
                buffer: w.buffer,
                offset: 1,
                size: 3
            }
        },
        "a buffer readback needs no alignment"
    );
    assert_eq!(
        run(&w, vec![read(1), read(1)]).err(),
        Some(RenderError::Invalid(READBACK_TAKEN))
    );
    let taken = BTreeSet::from([ReadbackId::new(2, 0)]);
    let list = CommandList {
        commands: vec![read(1), read(2)],
    };
    assert_eq!(
        check(&w.tables, &test_caps(), true, &taken, list).err(),
        Some(RenderError::Invalid(READBACK_TAKEN))
    );
    let mut w = world();
    let two_aspects = add_texture(
        &mut w.tables,
        texture(PixelFormat::D32FloatS8Uint, 4, 4, TextureUsage::COPY_SRC),
    );
    let read_depth = Command::Readback {
        from: region(two_aspects, 0, 0, 4, 4),
        id: ReadbackId::new(9, 0),
    };
    assert_eq!(
        run(&w, vec![read_depth.clone()]).err(),
        Some(RenderError::Unsupported(rules::DEPTH_STENCIL_BUFFER_COPY))
    );
    // Both faults: the wgpu rule (3) comes before the readback id rule (4).
    let taken = BTreeSet::from([ReadbackId::new(9, 0)]);
    let list = CommandList {
        commands: vec![read_depth],
    };
    assert_eq!(
        check(&w.tables, &test_caps(), true, &taken, list).err(),
        Some(RenderError::Unsupported(rules::DEPTH_STENCIL_BUFFER_COPY))
    );
    // A single-aspect depth texture reads back where the device can copy depth texels.
    let d16 = add_texture(
        &mut w.tables,
        texture(PixelFormat::D16Unorm, 4, 4, TextureUsage::COPY_SRC),
    );
    let read_d16 = || CommandList {
        commands: vec![Command::Readback {
            from: region(d16, 0, 0, 4, 4),
            id: ReadbackId::new(10, 0),
        }],
    };
    let none = BTreeSet::new();
    assert!(check(&w.tables, &test_caps(), true, &none, read_d16()).is_ok());
    assert_eq!(
        check(&w.tables, &test_caps(), false, &none, read_d16()).err(),
        Some(RenderError::Unsupported(rules::DEPTH_COPIES))
    );
}

#[test]
fn texture_readbacks_fit_the_largest_buffer_with_padded_rows() {
    let w = world();
    // The 8x4 RGBA8 texture: rows of 32 bytes pad to 256, so the read takes 4 * 256 bytes.
    let (stride, size) = rules::padded_read(32, 4, 1).expect("small");
    assert_eq!(stride, wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
    assert_eq!(size, 4 * u64::from(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT));
    let with_max = |max| {
        let mut caps = test_caps();
        caps.limits.max_buffer_size = max;
        caps
    };
    let read = |id| Command::Readback {
        from: region(w.rgba, 0, 0, 8, 4),
        id: ReadbackId::new(id, 0),
    };
    let none = BTreeSet::new();
    let list = |commands| CommandList { commands };
    assert!(check(&w.tables, &with_max(size), true, &none, list(vec![read(1)])).is_ok());
    assert_eq!(
        check(
            &w.tables,
            &with_max(size - 1),
            true,
            &none,
            list(vec![read(1)])
        )
        .err(),
        Some(RenderError::Unsupported(rules::READ_TOO_LARGE))
    );
    // The packed size (128 bytes) is not what is measured.
    assert_eq!(
        check(&w.tables, &with_max(128), true, &none, list(vec![read(1)])).err(),
        Some(RenderError::Unsupported(rules::READ_TOO_LARGE))
    );
    // Both faults: the size (rule 3) before a taken readback id (rule 4).
    let taken = BTreeSet::from([ReadbackId::new(1, 0)]);
    assert_eq!(
        check(
            &w.tables,
            &with_max(size - 1),
            true,
            &taken,
            list(vec![read(1)])
        )
        .err(),
        Some(RenderError::Unsupported(rules::READ_TOO_LARGE))
    );
    // A dumped frame that would not fit reads nothing (its pixels are unknown), and is not
    // refused; one exactly at the limit reads.
    let present = Command::Present(PresentDesc {
        view: w.rgba_view,
        src_rect: Rect {
            x: 0,
            y: 0,
            width: 8,
            height: 4,
        },
        transform: PresentTransform::Identity,
        frame: 1,
        dump: true,
    });
    let at_limit = check(
        &w.tables,
        &with_max(size),
        true,
        &none,
        list(vec![present.clone()]),
    )
    .expect("accepted");
    assert!(as_present(&at_limit.steps[0]).1.is_some());
    let over = check(
        &w.tables,
        &with_max(size - 1),
        true,
        &none,
        list(vec![present]),
    )
    .expect("accepted");
    assert_eq!(as_present(&over.steps[0]).1, None);
}

#[test]
fn padded_reads_overflow_to_none() {
    assert_eq!(rules::padded_read(u32::MAX, 1, 1), None, "the stride");
    let stride = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    assert_eq!(
        rules::padded_read(stride, u32::MAX, 1),
        Some((stride, u64::from(stride) * u64::from(u32::MAX)))
    );
    assert_eq!(
        rules::padded_read(stride, u32::MAX, u32::MAX),
        None,
        "the size"
    );
    assert_eq!(
        rules::check_read_size(u32::MAX, 1, 1, u64::MAX),
        Err(RenderError::Unsupported(rules::READ_TOO_LARGE))
    );
}

#[test]
fn presents_read_dumped_frames_from_the_base_layer() {
    let w = world();
    let present = |view, dump| {
        Command::Present(PresentDesc {
            view,
            src_rect: Rect {
                x: 1,
                y: 1,
                width: 2,
                height: 2,
            },
            transform: PresentTransform::Identity,
            frame: 7,
            dump,
        })
    };
    let s = steps(
        &w,
        vec![
            present(w.rgba_view, false),
            present(w.array_view, true),
            present(w.deep_view, true),
        ],
    );
    assert_eq!(as_present(&s[0]), (PixelFormat::A8B8G8R8Unorm, None));
    let (at, rows) = as_present(&s[1]).1.expect("dumped");
    assert_eq!(
        at.origin,
        Origin3d { x: 1, y: 1, z: 1 },
        "the view's base layer"
    );
    assert_eq!(
        rows,
        Rows {
            row_bytes: 8,
            rows: 2,
            images: 1
        }
    );
    let read = as_present(&s[2]).1.expect("dumped");
    assert_eq!(read.0.origin.z, 0, "slice 0 of a 3D texture");
}

#[test]
fn render_targets_are_checked_and_resolved() {
    use crate::draw::{Attachment, Targets};
    let w = world();
    let mut targets = RenderTargets::default();
    targets.colors[0] = Some(w.rgba_view);
    targets.depth_stencil = Some(w.depth_view);
    let depth = w.tables.view(w.depth_view).expect("live").0.texture;
    let attachment = |texture, format| Attachment {
        texture,
        format,
        mip: 0,
        layer: 0,
        three_d: false,
        size: (8, 4),
        samples: 1,
    };
    let resolved = Targets {
        colors: vec![Some(attachment(w.rgba, wgpu::TextureFormat::Rgba8Unorm))],
        depth_stencil: Some(attachment(depth, wgpu::TextureFormat::Depth32Float)),
    };
    assert_eq!(
        steps(&w, vec![Command::SetRenderTargets(targets)]),
        [Step::Targets(resolved.clone())]
    );
    // The same targets again change nothing; others do.
    let mut others = targets;
    others.colors[0] = Some(w.srgb_view);
    assert_eq!(
        steps(
            &w,
            vec![
                Command::SetRenderTargets(targets),
                Command::SetRenderTargets(targets),
                Command::SetRenderTargets(others),
            ]
        )
        .len(),
        2
    );
    let mut wrong = RenderTargets::default();
    wrong.colors[0] = Some(w.depth_view);
    assert_eq!(
        run(&w, vec![Command::SetRenderTargets(wrong)]).err(),
        Some(RenderError::Invalid(COLOR_TARGET))
    );
    let wrong = RenderTargets {
        depth_stencil: Some(w.rgba_view),
        ..RenderTargets::default()
    };
    assert_eq!(
        run(&w, vec![Command::SetRenderTargets(wrong)]).err(),
        Some(RenderError::Invalid(DEPTH_TARGET))
    );
    let mut w = world();
    let sampled = add_texture(
        &mut w.tables,
        texture(PixelFormat::A8B8G8R8Unorm, 4, 4, TextureUsage::SAMPLED),
    );
    let sampled_view = add_view(
        &mut w.tables,
        sampled,
        view(PixelFormat::A8B8G8R8Unorm, ViewDimension::D2),
    );
    let mut wrong = RenderTargets::default();
    wrong.colors[1] = Some(sampled_view);
    assert_eq!(
        run(&w, vec![Command::SetRenderTargets(wrong)]).err(),
        Some(RenderError::MissingUsage {
            resource: "render target texture",
            usage: "render target"
        })
    );
    assert_eq!(
        steps(&w, vec![Command::Marker(Marker { submit: 1, draw: 1 })]),
        []
    );
}

#[test]
fn a_view_whose_texture_is_gone_is_stale() {
    let mut w = world();
    let rgba = w.rgba;
    w.tables
        .textures
        .remove(rgba.index(), rgba.generation())
        .expect("live");
    let clear = Command::Clear(ClearDesc::unmasked(
        w.rgba_view,
        ClearValue::Float([0.0; 4]),
        None,
    ));
    assert_eq!(
        run(&w, vec![clear]).err(),
        Some(RenderError::Invalid(NO_TEXTURE))
    );
    let stale = ViewId::new(w.rgba_view.index(), w.rgba_view.generation() + 1);
    let clear = Command::Clear(ClearDesc::unmasked(
        stale,
        ClearValue::Float([0.0; 4]),
        None,
    ));
    assert_eq!(
        run(&w, vec![clear]).err(),
        Some(RenderError::Invalid(NO_VIEW))
    );
}
