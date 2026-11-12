//! Draws, dispatches, presents and readbacks; every command of the list in one list; refused
//! lists and the rules only the backend can check.

use super::fixtures::{
    backend, buffer, clear, compute, counting, draw, graphics, rb, read, rect, region, slice,
    submit, texture, upload, view,
};
use starquake_render::command::{
    Binding, BindingResource, BlitDesc, BufferSlice, ClearValue, CopyDesc, DispatchDesc,
    DispatchParams, DrawParams, IndexBufferBinding, IndexFormat, Marker, PresentDesc,
    PresentTransform, RenderTargets,
};
use starquake_render::resource::FilterMode;
use starquake_render::{
    Backend, BufferUsage, Command, CommandList, FenceValue, FrameDump, PixelFormat,
    PresentFeedback, PresentOutcome, Readback, RenderError, TextureUsage,
};
use std::time::Duration;

const COPY: BufferUsage = BufferUsage::COPY_SRC.union(BufferUsage::COPY_DST);

fn targets(color: starquake_render::ViewId, depth: Option<starquake_render::ViewId>) -> Command {
    let mut t = RenderTargets::default();
    t.colors[0] = Some(color);
    t.depth_stencil = depth;
    Command::SetRenderTargets(t)
}

#[test]
fn a_draw_makes_its_targets_unknown_and_nothing_else() {
    let b = backend();
    let p = graphics(&b);
    let c = PixelFormat::A8B8G8R8Unorm;
    let d = PixelFormat::D32Float;
    let color = texture(&b, c, (2, 2), 2, 2);
    let depth = texture(&b, d, (2, 2), 1, 1);
    let cv = view(&b, color, c, 0, (0, 1));
    let dv = view(&b, depth, d, 0, (0, 1));
    let fill = |b: &starquake_render::RecordingBackend| {
        submit(
            b,
            vec![
                upload(&[1; 32], region(color, 0, (0, 0, 0), (2, 2, 2))),
                upload(&[2; 4], region(color, 1, (0, 0, 0), (1, 1, 1))),
                upload(&[3; 16], region(depth, 0, (0, 0, 0), (2, 2, 1))),
            ],
        );
    };
    fill(&b);
    submit(&b, vec![targets(cv, Some(dv)), Command::Draw(draw(p))]);
    assert_eq!(
        read(&b, region(color, 0, (0, 0, 0), (2, 2, 1))),
        Readback::Unknown
    );
    assert_eq!(
        read(&b, region(depth, 0, (0, 0, 0), (2, 2, 1))),
        Readback::Unknown
    );
    // Layer 1 and mip 1 are outside the view.
    assert_eq!(
        read(&b, region(color, 0, (0, 0, 1), (2, 2, 1))),
        Readback::Bytes(vec![1; 16])
    );
    assert_eq!(
        read(&b, region(color, 1, (0, 0, 0), (1, 1, 1))),
        Readback::Bytes(vec![2; 4])
    );

    // Render targets last for one list: a draw in the next list writes no target.
    fill(&b);
    submit(&b, vec![targets(cv, None)]);
    submit(&b, vec![Command::Draw(draw(p))]);
    assert_eq!(
        read(&b, region(color, 0, (0, 0, 0), (2, 2, 1))),
        Readback::Bytes(vec![1; 16])
    );
    // A draw before the targets are set in a list writes none either.
    submit(&b, vec![Command::Draw(draw(p)), targets(cv, None)]);
    assert_eq!(
        read(&b, region(color, 0, (0, 0, 0), (2, 2, 1))),
        Readback::Bytes(vec![1; 16])
    );
}

#[test]
fn draws_and_dispatches_forget_storage_bindings_only() {
    let b = backend();
    let g = graphics(&b);
    let cp = compute(&b);
    let storage = buffer(&b, 16, COPY | BufferUsage::STORAGE);
    let uniform = buffer(&b, 16, COPY | BufferUsage::UNIFORM);
    let f = PixelFormat::R32Float;
    let image = texture(&b, f, (2, 2), 1, 1);
    let sampled_desc = starquake_render::TextureDesc {
        usage: TextureUsage::SAMPLED | TextureUsage::COPY_DST | TextureUsage::COPY_SRC,
        ..super::fixtures::texture_desc(
            f,
            starquake_render::resource::TextureDimension::D2,
            (2, 2, 1),
            1,
        )
    };
    let sampled = b.create_texture(&sampled_desc).expect("valid");
    let iv = view(&b, image, f, 0, (0, 1));
    let sv = view(&b, sampled, f, 0, (0, 1));
    let bind = |group, binding, resource| Binding {
        group,
        binding,
        resource,
    };
    let bindings = vec![
        bind(
            0,
            0,
            BindingResource::Buffer(BufferSlice {
                buffer: storage,
                offset: 4,
                size: 8,
            }),
        ),
        bind(
            0,
            1,
            BindingResource::Buffer(BufferSlice {
                buffer: uniform,
                offset: 0,
                size: 16,
            }),
        ),
        bind(0, 2, BindingResource::View(iv)),
        bind(0, 3, BindingResource::View(sv)),
    ];
    for op in [
        Command::Dispatch(DispatchDesc {
            pipeline: cp,
            bindings: bindings.clone(),
            params: DispatchParams::Direct { x: 1, y: 1, z: 1 },
        }),
        Command::Draw(starquake_render::command::DrawDesc {
            bindings: bindings.clone(),
            ..draw(g)
        }),
    ] {
        submit(
            &b,
            vec![
                upload(&[4; 16], slice(storage, 0, 16)),
                upload(&[5; 16], slice(uniform, 0, 16)),
                upload(&[6; 16], region(image, 0, (0, 0, 0), (2, 2, 1))),
                upload(&[7; 16], region(sampled, 0, (0, 0, 0), (2, 2, 1))),
                op,
            ],
        );
        // Only the bound range of the storage buffer.
        assert_eq!(read(&b, slice(storage, 4, 8)), Readback::Unknown);
        assert_eq!(read(&b, slice(storage, 0, 4)), Readback::Bytes(vec![4; 4]));
        assert_eq!(read(&b, slice(storage, 12, 4)), Readback::Bytes(vec![4; 4]));
        assert_eq!(
            read(&b, slice(uniform, 0, 16)),
            Readback::Bytes(vec![5; 16])
        );
        assert_eq!(
            read(&b, region(image, 0, (0, 0, 0), (2, 2, 1))),
            Readback::Unknown
        );
        assert_eq!(
            read(&b, region(sampled, 0, (0, 0, 0), (2, 2, 1))),
            Readback::Bytes(vec![7; 16])
        );
    }
}

#[test]
fn presents_give_feedback_and_dumps() {
    let b = backend();
    let f = PixelFormat::A8B8G8R8Unorm;
    let t = texture(&b, f, (2, 2), 1, 1);
    let v = view(&b, t, f, 0, (0, 1));
    let present = |frame, dump| {
        Command::Present(PresentDesc {
            view: v,
            src_rect: rect(1, 0, 1, 2),
            transform: PresentTransform::Rotate90,
            frame,
            dump,
        })
    };
    submit(
        &b,
        vec![
            present(1, true),
            upload(&counting(0, 16), region(t, 0, (0, 0, 0), (2, 2, 1))),
            present(2, true),
            present(3, false),
        ],
    );
    let shown = |frame| PresentFeedback {
        frame,
        outcome: PresentOutcome::Shown,
        host_time: Duration::ZERO,
    };
    assert_eq!(
        b.take_present_feedback(),
        vec![shown(1), shown(2), shown(3)]
    );
    assert_eq!(b.take_present_feedback(), Vec::new(), "taken");
    let dump = |frame, pixels| FrameDump {
        frame,
        view: v,
        format: f,
        width: 1,
        height: 2,
        transform: PresentTransform::Rotate90,
        pixels,
    };
    // Frame 1 was presented before the upload; the dump is the right column, untransformed.
    assert_eq!(
        b.take_frame_dumps(),
        vec![
            dump(1, Readback::Unknown),
            dump(2, Readback::Bytes(vec![4, 5, 6, 7, 12, 13, 14, 15])),
        ]
    );
    assert_eq!(b.take_frame_dumps(), Vec::new(), "taken");
}

#[test]
fn a_compressed_present_dumps_unknown() {
    let b = backend();
    let f = PixelFormat::Bc1RgbaUnorm;
    let t = texture(&b, f, (4, 4), 1, 1);
    let v = view(&b, t, f, 0, (0, 1));
    submit(
        &b,
        vec![
            upload(&[1; 8], region(t, 0, (0, 0, 0), (4, 4, 1))),
            Command::Present(PresentDesc {
                view: v,
                src_rect: rect(0, 0, 4, 4),
                transform: PresentTransform::Identity,
                frame: 9,
                dump: true,
            }),
        ],
    );
    let dumps = b.take_frame_dumps();
    assert_eq!(dumps.len(), 1);
    assert_eq!(dumps[0].pixels, Readback::Unknown);
}

/// One list holding every kind of command, each checked by its effect.
#[test]
fn every_command_runs() {
    let b = backend();
    let g = graphics(&b);
    let cp = compute(&b);
    let f = PixelFormat::A8B8G8R8Unorm;
    let src = texture(&b, f, (2, 2), 1, 1);
    let dst = texture(&b, f, (2, 2), 1, 1);
    let rt = texture(&b, f, (2, 2), 1, 1);
    let sv = view(&b, src, f, 0, (0, 1));
    let dv = view(&b, dst, f, 0, (0, 1));
    let rv = view(&b, rt, f, 0, (0, 1));
    let buf = buffer(&b, 16, COPY);
    let list = vec![
        Command::Marker(Marker { submit: 1, draw: 0 }),
        upload(&counting(0, 16), region(src, 0, (0, 0, 0), (2, 2, 1))),
        Command::Copy(CopyDesc {
            src: region(src, 0, (0, 0, 0), (2, 2, 1)),
            dst: slice(buf, 0, 16),
        }),
        Command::Blit(BlitDesc {
            src: sv,
            src_rect: rect(0, 0, 2, 2),
            dst: dv,
            dst_rect: rect(0, 0, 2, 2),
            filter: FilterMode::Nearest,
        }),
        clear(rv, ClearValue::Float([0.0, 1.0, 0.0, 1.0]), None),
        Command::Readback {
            from: region(rt, 0, (0, 0, 0), (1, 1, 1)),
            id: rb(1),
        },
        targets(rv, None),
        Command::Draw(draw(g)),
        Command::Dispatch(DispatchDesc {
            pipeline: cp,
            bindings: Vec::new(),
            params: DispatchParams::Direct { x: 1, y: 1, z: 1 },
        }),
        Command::Readback {
            from: region(rt, 0, (0, 0, 0), (1, 1, 1)),
            id: rb(2),
        },
        Command::Present(PresentDesc {
            view: dv,
            src_rect: rect(0, 0, 2, 2),
            transform: PresentTransform::Identity,
            frame: 1,
            dump: true,
        }),
    ];
    let fence = submit(&b, list.clone());
    assert_eq!(fence, FenceValue::new(1));
    assert_eq!(b.completed(), fence);
    assert_eq!(
        read(&b, slice(buf, 0, 16)),
        Readback::Bytes(counting(0, 16))
    );
    assert_eq!(
        b.take_readback(rb(1)),
        Some(Readback::Bytes(vec![0, 0xff, 0, 0xff]))
    );
    assert_eq!(
        b.take_readback(rb(2)),
        Some(Readback::Unknown),
        "after the draw"
    );
    assert_eq!(b.take_readback(rb(1)), None, "taken");
    let dumps = b.take_frame_dumps();
    assert_eq!(dumps.len(), 1);
    assert_eq!(dumps[0].pixels, Readback::Bytes(counting(0, 16)), "blitted");
    assert_eq!(b.take_present_feedback().len(), 1);
    let lists = b.lists();
    assert_eq!(
        lists.first(),
        Some(&(fence, CommandList { commands: list }))
    );
}

#[test]
fn a_refused_list_changes_nothing() {
    let b = backend();
    let buf = buffer(&b, 16, COPY);
    let gone = buffer(&b, 16, COPY);
    b.destroy(starquake_render::ResourceId::Buffer(gone), FenceValue::ZERO);
    let before = b.lists().len();
    let result = b.submit(CommandList {
        commands: vec![
            upload(&[1; 16], slice(buf, 0, 16)),
            Command::Readback {
                from: slice(buf, 0, 16),
                id: rb(1),
            },
            Command::Copy(CopyDesc {
                src: slice(buf, 0, 4),
                dst: slice(gone, 0, 4),
            }),
        ],
    });
    assert_eq!(
        result.map_err(|e| e.to_string()),
        Err("no live buffer has this id".to_owned())
    );
    assert_eq!(b.completed(), FenceValue::ZERO);
    assert_eq!(b.take_readback(rb(1)), None);
    assert_eq!(b.lists().len(), before);
    assert_eq!(
        read(&b, slice(buf, 0, 16)),
        Readback::Unknown,
        "the upload did not happen"
    );
    assert_eq!(
        b.completed(),
        FenceValue::new(1),
        "the next list takes fence 1"
    );
}

fn refused(b: &starquake_render::RecordingBackend, commands: Vec<Command>) -> String {
    match b.submit(CommandList { commands }) {
        Ok(fence) => unreachable!("the list was accepted as fence {fence:?}"),
        Err(error) => error.to_string(),
    }
}

#[test]
fn overlapping_copies_are_refused_and_touching_ones_run() {
    let b = backend();
    let buf = buffer(&b, 16, COPY);
    assert_eq!(
        refused(
            &b,
            vec![Command::Copy(CopyDesc {
                src: slice(buf, 0, 8),
                dst: slice(buf, 7, 8),
            })]
        ),
        "a copy's source and destination overlap"
    );
    submit(
        &b,
        vec![
            upload(&counting(0, 8), slice(buf, 0, 8)),
            Command::Copy(CopyDesc {
                src: slice(buf, 0, 8),
                dst: slice(buf, 8, 8),
            }),
        ],
    );
    assert_eq!(read(&b, slice(buf, 8, 8)), Readback::Bytes(counting(0, 8)));

    let f = PixelFormat::R8Uint;
    let t = texture(&b, f, (4, 4), 2, 1);
    assert_eq!(
        refused(
            &b,
            vec![Command::Copy(CopyDesc {
                src: region(t, 0, (0, 0, 0), (2, 2, 1)),
                dst: region(t, 0, (1, 1, 0), (2, 2, 1)),
            })]
        ),
        "a copy's source and destination overlap"
    );
    // The same box in another layer is no overlap.
    submit(
        &b,
        vec![
            upload(&[3; 4], region(t, 0, (0, 0, 0), (2, 2, 1))),
            Command::Copy(CopyDesc {
                src: region(t, 0, (0, 0, 0), (2, 2, 1)),
                dst: region(t, 0, (0, 0, 1), (2, 2, 1)),
            }),
        ],
    );
    assert_eq!(
        read(&b, region(t, 0, (0, 0, 1), (2, 2, 1))),
        Readback::Bytes(vec![3; 4])
    );
    let v = view(&b, t, f, 0, (0, 1));
    assert_eq!(
        refused(
            &b,
            vec![Command::Blit(BlitDesc {
                src: v,
                src_rect: rect(0, 0, 2, 2),
                dst: v,
                dst_rect: rect(1, 0, 2, 2),
                filter: FilterMode::Nearest,
            })]
        ),
        "a blit's source and destination overlap"
    );
    // Touching blit rectangles run.
    submit(
        &b,
        vec![Command::Blit(BlitDesc {
            src: v,
            src_rect: rect(0, 0, 2, 2),
            dst: v,
            dst_rect: rect(2, 0, 2, 2),
            filter: FilterMode::Nearest,
        })],
    );
    assert_eq!(
        read(&b, region(t, 0, (2, 0, 0), (2, 2, 1))),
        Readback::Bytes(vec![3; 4])
    );
}

#[test]
fn readback_ids_must_be_free() {
    let b = backend();
    let buf = buffer(&b, 4, COPY);
    let rd = |id| Command::Readback {
        from: slice(buf, 0, 4),
        id: rb(id),
    };
    assert_eq!(
        refused(&b, vec![rd(1), rd(1)]),
        "a readback id is already waiting to be taken"
    );
    submit(&b, vec![rd(1)]);
    assert_eq!(
        refused(&b, vec![rd(1)]),
        "a readback id is already waiting to be taken"
    );
    assert_eq!(b.take_readback(rb(1)), Some(Readback::Unknown));
    submit(&b, vec![rd(1)]);
    assert_eq!(
        b.take_readback(rb(1)),
        Some(Readback::Unknown),
        "free again once taken"
    );
}

#[test]
fn render_targets_are_checked() {
    let b = backend();
    let c = PixelFormat::A8B8G8R8Unorm;
    let d = PixelFormat::D32Float;
    let color = texture(&b, c, (2, 2), 1, 1);
    let depth = texture(&b, d, (2, 2), 1, 1);
    let cv = view(&b, color, c, 0, (0, 1));
    let dv = view(&b, depth, d, 0, (0, 1));
    assert_eq!(
        refused(&b, vec![targets(dv, None)]),
        "a colour target view must store colour"
    );
    assert_eq!(
        refused(&b, vec![targets(cv, Some(cv))]),
        "a depth/stencil target view must store depth or stencil"
    );
    let plain = b
        .create_texture(&starquake_render::TextureDesc {
            usage: TextureUsage::SAMPLED,
            ..super::fixtures::texture_desc(
                c,
                starquake_render::resource::TextureDimension::D2,
                (2, 2, 1),
                1,
            )
        })
        .expect("valid");
    let pv = view(&b, plain, c, 0, (0, 1));
    assert_eq!(
        refused(&b, vec![targets(pv, None)]),
        "the render target texture lacks the render target usage"
    );
    submit(&b, vec![targets(cv, Some(dv))]);
}

#[test]
fn pipeline_kinds_and_bindings_are_checked() {
    let b = backend();
    let g = graphics(&b);
    let cp = compute(&b);
    let plain = buffer(&b, 16, COPY);
    let bind = |resource| Binding {
        group: 0,
        binding: 0,
        resource,
    };
    assert_eq!(
        refused(&b, vec![Command::Draw(draw(cp))]),
        "a draw needs a graphics pipeline"
    );
    let dispatch = |pipeline, bindings, params| {
        Command::Dispatch(DispatchDesc {
            pipeline,
            bindings,
            params,
        })
    };
    let direct = DispatchParams::Direct { x: 1, y: 1, z: 1 };
    assert_eq!(
        refused(&b, vec![dispatch(g, Vec::new(), direct)]),
        "a dispatch needs a compute pipeline"
    );
    let whole = |buffer| BufferSlice {
        buffer,
        offset: 0,
        size: 16,
    };
    assert_eq!(
        refused(
            &b,
            vec![dispatch(
                cp,
                vec![bind(BindingResource::Buffer(whole(plain)))],
                direct
            )]
        ),
        "the bound buffer lacks the uniform or storage usage"
    );
    assert_eq!(
        refused(
            &b,
            vec![dispatch(
                cp,
                Vec::new(),
                DispatchParams::Indirect {
                    buffer: plain,
                    offset: 0
                }
            )]
        ),
        "the indirect buffer lacks the indirect usage"
    );
    let sampled_only = texture(&b, PixelFormat::R8Unorm, (1, 1), 1, 1);
    let copy_only = b
        .create_texture(&starquake_render::TextureDesc {
            usage: TextureUsage::COPY_SRC,
            ..super::fixtures::texture_desc(
                PixelFormat::R8Unorm,
                starquake_render::resource::TextureDimension::D2,
                (1, 1, 1),
                1,
            )
        })
        .expect("valid");
    let cv = view(&b, copy_only, PixelFormat::R8Unorm, 0, (0, 1));
    assert_eq!(
        refused(
            &b,
            vec![dispatch(cp, vec![bind(BindingResource::View(cv))], direct)]
        ),
        "the bound texture lacks the sampled or storage usage"
    );
    let ok = view(&b, sampled_only, PixelFormat::R8Unorm, 0, (0, 1));
    submit(
        &b,
        vec![dispatch(cp, vec![bind(BindingResource::View(ok))], direct)],
    );
    assert_eq!(
        refused(
            &b,
            vec![dispatch(
                cp,
                vec![bind(BindingResource::Sampler(
                    starquake_render::SamplerId::new(0, 0)
                ))],
                direct
            )]
        ),
        "no live sampler has this id"
    );
}

#[test]
fn draw_buffers_are_checked() {
    let b = backend();
    let g = graphics(&b);
    let plain = buffer(&b, 16, COPY);
    let vertex = buffer(&b, 16, BufferUsage::VERTEX);
    let whole = |buffer| BufferSlice {
        buffer,
        offset: 0,
        size: 16,
    };
    let with = |f: &dyn Fn(&mut starquake_render::command::DrawDesc)| {
        let mut d = draw(g);
        f(&mut d);
        Command::Draw(d)
    };
    assert_eq!(
        refused(&b, vec![with(&|d| d.vertex_buffers = vec![whole(plain)])]),
        "the vertex buffer lacks the vertex usage"
    );
    assert_eq!(
        refused(
            &b,
            vec![with(&|d| {
                d.index_buffer = Some(IndexBufferBinding {
                    slice: whole(vertex),
                    format: IndexFormat::U16,
                });
                d.params = DrawParams::Indexed {
                    index_count: 3,
                    instance_count: 1,
                    first_index: 0,
                    base_vertex: 0,
                    first_instance: 0,
                };
            })]
        ),
        "the index buffer lacks the index usage"
    );
    assert_eq!(
        refused(
            &b,
            vec![with(&|d| {
                d.params = DrawParams::Indirect {
                    buffer: vertex,
                    offset: 0,
                    draw_count: 1,
                    stride: 16,
                    indexed: false,
                };
            })]
        ),
        "the indirect buffer lacks the indirect usage"
    );
    // A vertex range past the buffer's end.
    assert_eq!(
        refused(
            &b,
            vec![with(&|d| {
                d.vertex_buffers = vec![BufferSlice {
                    buffer: vertex,
                    offset: 8,
                    size: 9,
                }];
            })]
        ),
        "buffer range ends at 17, past the end at 16"
    );
}

#[test]
fn validation_errors_pass_through() {
    let b = backend();
    let buf = buffer(&b, 8, BufferUsage::COPY_DST);
    // An upload of the wrong size; a readback from a buffer without the copy-source usage.
    assert_eq!(
        refused(&b, vec![upload(&[0; 3], slice(buf, 0, 4))]),
        "staging slice is 3 bytes, but 4 are needed"
    );
    assert_eq!(
        refused(
            &b,
            vec![Command::Readback {
                from: slice(buf, 0, 4),
                id: rb(0),
            }]
        ),
        "the copy source buffer lacks the copy source usage"
    );
    let f = PixelFormat::A8B8G8R8Unorm;
    let t = texture(&b, f, (2, 2), 1, 1);
    let v = view(&b, t, f, 0, (0, 1));
    assert_eq!(
        refused(&b, vec![clear(v, ClearValue::Uint([0; 4]), None)]),
        "the clear value does not suit the view's format"
    );
    assert_eq!(
        refused(
            &b,
            vec![Command::Present(PresentDesc {
                view: v,
                src_rect: rect(0, 0, 3, 1),
                transform: PresentTransform::Identity,
                frame: 0,
                dump: false,
            })]
        ),
        "rectangle width ends at 3, past the end at 2"
    );
    // The error is the same whatever the error type's other variants: pinned as a value too.
    assert_eq!(
        b.submit(CommandList {
            commands: vec![upload(&[0; 3], slice(buf, 0, 4))]
        }),
        Err(RenderError::SizeMismatch {
            what: "staging slice",
            expected: 4,
            actual: 3
        })
    );
}
