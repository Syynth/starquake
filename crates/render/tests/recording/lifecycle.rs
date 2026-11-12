//! Resource lifetimes, pipelines and their scripts, fences and waiting, the budget, and the
//! snapshot.

use super::fixtures::{
    MEMORY, backend, buffer, clear, compute, draw, graphics, graphics_desc, rb, read, rect, region,
    shader, slice, submit, texture, upload, view,
};
use starquake_render::command::{ClearValue, PresentDesc, PresentTransform, RenderTargets};
use starquake_render::resource::{Swizzle, ViewDimension};
use starquake_render::{
    Backend, BufferId, BufferUsage, Command, CommandList, ComputePipelineDesc, FenceValue,
    MemoryBudget, PipelineId, PipelineStatus, PixelFormat, PresentTarget, Readback, ResourceId,
    ShaderStage, TextureId, ViewDesc, WaitOutcome,
};
use std::sync::Arc;
use std::time::Duration;

const COPY: BufferUsage = BufferUsage::COPY_SRC.union(BufferUsage::COPY_DST);

fn refused(b: &starquake_render::RecordingBackend, commands: Vec<Command>) -> String {
    match b.submit(CommandList { commands }) {
        Ok(fence) => unreachable!("the list was accepted as fence {fence:?}"),
        Err(error) => error.to_string(),
    }
}

#[test]
fn destroy_waits_for_its_fence_value() {
    let b = backend();
    let buf = buffer(&b, 4, COPY);
    let id = ResourceId::Buffer(buf);
    // Fence 2 has not been reached: the buffer stays usable through fence 1's list.
    b.destroy(id, FenceValue::new(2));
    submit(&b, vec![upload(&[1; 4], slice(buf, 0, 4))]);
    assert_eq!(read(&b, slice(buf, 0, 4)), Readback::Bytes(vec![1; 4]));
    // The read's list was fence 2, so the buffer is now gone.
    assert_eq!(b.completed(), FenceValue::new(2));
    assert_eq!(
        refused(&b, vec![upload(&[1; 4], slice(buf, 0, 4))]),
        "no live buffer has this id"
    );
    assert_eq!(b.refused_destroys(), 0);
    // Its slot is reused with the next generation; the old id still names nothing.
    let again = buffer(&b, 4, COPY);
    assert_eq!((again.index(), again.generation()), (buf.index(), 1));
    assert_eq!(
        refused(&b, vec![upload(&[1; 4], slice(buf, 0, 4))]),
        "no live buffer has this id"
    );
}

#[test]
fn destroy_at_a_passed_value_is_immediate_and_bad_destroys_are_counted() {
    let b = backend();
    let t = texture(&b, PixelFormat::R8Unorm, (1, 1), 1, 1);
    submit(&b, Vec::new());
    b.destroy(ResourceId::Texture(t), FenceValue::new(1));
    assert_eq!(
        refused(&b, vec![upload(&[1], region(t, 0, (0, 0, 0), (1, 1, 1)))]),
        "no live texture has this id"
    );
    assert_eq!(b.refused_destroys(), 0);
    b.destroy(ResourceId::Texture(t), FenceValue::new(1));
    assert_eq!(b.refused_destroys(), 1, "already destroyed");
    let buf = buffer(&b, 4, COPY);
    b.destroy(ResourceId::Buffer(buf), FenceValue::new(10));
    b.destroy(ResourceId::Buffer(buf), FenceValue::new(10));
    assert_eq!(b.refused_destroys(), 2, "already waiting");
    b.destroy(ResourceId::Buffer(BufferId::new(99, 0)), FenceValue::ZERO);
    assert_eq!(b.refused_destroys(), 3, "never existed");
}

#[test]
fn a_view_of_a_destroyed_texture_names_nothing() {
    let b = backend();
    let f = PixelFormat::R8Uint;
    let t = texture(&b, f, (1, 1), 1, 1);
    let v = view(&b, t, f, 0, (0, 1));
    b.destroy(ResourceId::Texture(t), FenceValue::ZERO);
    assert_eq!(
        refused(&b, vec![clear(v, ClearValue::Uint([0; 4]), None)]),
        "no live texture has this id"
    );
    let desc = ViewDesc {
        format: f,
        dimension: ViewDimension::D2Array,
        base_mip: 0,
        mip_count: 1,
        base_layer: 0,
        layer_count: 1,
        swizzle: Swizzle::IDENTITY,
    };
    assert_eq!(
        b.create_view(t, &desc).map_err(|e| e.to_string()),
        Err("no live texture has this id".to_owned())
    );
    b.destroy(ResourceId::View(v), FenceValue::ZERO);
    assert_eq!(
        refused(
            &b,
            vec![Command::Present(PresentDesc {
                view: v,
                src_rect: rect(0, 0, 1, 1),
                transform: PresentTransform::Identity,
                frame: 0,
                dump: false,
            })]
        ),
        "no live view has this id"
    );
    let other = TextureId::new(t.index(), t.generation() + 1);
    assert_eq!(
        b.create_view(other, &desc).map_err(|e| e.to_string()),
        Err("no live texture has this id".to_owned()),
        "a generation that was never handed out"
    );
}

#[test]
fn pipelines_check_their_shaders() {
    let b = backend();
    let vs = shader(&b, ShaderStage::Vertex);
    let fs = shader(&b, ShaderStage::Fragment);
    let cs = shader(&b, ShaderStage::Compute);
    assert_eq!(
        b.create_graphics_pipeline(graphics_desc(fs, Some(fs)))
            .map_err(|e| e.to_string()),
        Err("a pipeline's vertex shader is not a vertex shader".to_owned())
    );
    assert_eq!(
        b.create_graphics_pipeline(graphics_desc(vs, Some(vs)))
            .map_err(|e| e.to_string()),
        Err("a pipeline's fragment shader is not a fragment shader".to_owned())
    );
    assert!(b.create_graphics_pipeline(graphics_desc(vs, None)).is_ok());
    assert_eq!(
        b.create_compute_pipeline(ComputePipelineDesc { shader: vs })
            .map_err(|e| e.to_string()),
        Err("a compute pipeline's shader is not a compute shader".to_owned())
    );
    assert!(
        b.create_compute_pipeline(ComputePipelineDesc { shader: cs })
            .is_ok()
    );
    b.destroy(ResourceId::Shader(fs), FenceValue::ZERO);
    assert_eq!(
        b.create_graphics_pipeline(graphics_desc(vs, Some(fs)))
            .map_err(|e| e.to_string()),
        Err("no live shader has this id".to_owned())
    );
    assert_eq!(
        b.create_compute_pipeline(ComputePipelineDesc { shader: fs })
            .map_err(|e| e.to_string()),
        Err("no live shader has this id".to_owned())
    );
}

#[test]
fn pipeline_status_follows_its_script() {
    let b = backend();
    let p = graphics(&b);
    assert_eq!(b.pipeline_status(p), PipelineStatus::Ready, "ready at once");
    b.script_pipeline(
        p,
        vec![
            PipelineStatus::Compiling,
            PipelineStatus::Compiling,
            PipelineStatus::Failed,
        ],
    )
    .expect("live pipeline");
    assert_eq!(b.pipeline_status(p), PipelineStatus::Compiling);
    assert_eq!(b.pipeline_status(p), PipelineStatus::Compiling);
    assert_eq!(b.pipeline_status(p), PipelineStatus::Failed);
    assert_eq!(
        b.pipeline_status(p),
        PipelineStatus::Failed,
        "the last stays"
    );
    assert_eq!(
        b.script_pipeline(p, Vec::new()).map_err(|e| e.to_string()),
        Err("a pipeline script needs at least one status".to_owned())
    );
    let c = compute(&b);
    b.destroy(ResourceId::Pipeline(c), FenceValue::ZERO);
    assert_eq!(
        b.pipeline_status(c),
        PipelineStatus::Failed,
        "no live pipeline"
    );
    assert_eq!(
        b.script_pipeline(c, vec![PipelineStatus::Ready])
            .map_err(|e| e.to_string()),
        Err("no live pipeline has this id".to_owned())
    );
    assert_eq!(
        b.pipeline_status(PipelineId::new(77, 0)),
        PipelineStatus::Failed
    );
}

#[test]
fn fences_and_waiting() {
    let b = backend();
    assert_eq!(b.completed(), FenceValue::ZERO);
    assert_eq!(
        b.wait(FenceValue::ZERO, Duration::ZERO),
        Ok(WaitOutcome::Reached)
    );
    // An unreached value waits the whole timeout before giving up.
    let timeout = Duration::from_millis(20);
    let start = std::time::Instant::now();
    assert_eq!(
        b.wait(FenceValue::new(1), timeout),
        Ok(WaitOutcome::TimedOut)
    );
    assert!(start.elapsed() >= timeout, "returned early");
    assert_eq!(submit(&b, Vec::new()), FenceValue::new(1));
    assert_eq!(submit(&b, Vec::new()), FenceValue::new(2));
    assert_eq!(
        b.wait(FenceValue::new(1), Duration::ZERO),
        Ok(WaitOutcome::Reached)
    );
}

#[test]
fn a_wait_is_woken_by_another_threads_submit() {
    let b = Arc::new(backend());
    let waiter = {
        let b = Arc::clone(&b);
        // Bounded: if the submit never wakes it, the wait times out and the test fails.
        std::thread::spawn(move || b.wait(FenceValue::new(1), Duration::from_secs(30)))
    };
    // The waiter may start before or after the submit; either way it must see fence 1.
    submit(&b, Vec::new());
    assert_eq!(
        waiter.join().expect("waiter thread"),
        Ok(WaitOutcome::Reached)
    );
}

#[test]
fn budget_counts_every_live_resource() {
    let b = backend();
    assert_eq!(
        b.budget(),
        MemoryBudget {
            allowed: MEMORY,
            used: 0
        }
    );
    let buf = buffer(&b, 100, COPY);
    // 4x4 RGBA8 with 3 mips: 64 + 16 + 4 bytes, times 2 layers.
    let t = texture(&b, PixelFormat::A8B8G8R8Unorm, (4, 4), 2, 3);
    assert_eq!(b.budget().used, 100 + (64 + 16 + 4) * 2);
    // A 4-sample texture counts each sample.
    let ms = b
        .create_texture(&starquake_render::TextureDesc {
            sample_count: 4,
            ..super::fixtures::texture_desc(
                PixelFormat::R8Unorm,
                starquake_render::resource::TextureDimension::D2,
                (2, 2, 1),
                1,
            )
        })
        .expect("valid");
    assert_eq!(b.budget().used, 100 + (64 + 16 + 4) * 2 + 4 * 4);
    b.destroy(ResourceId::Texture(t), FenceValue::ZERO);
    b.destroy(ResourceId::Texture(ms), FenceValue::ZERO);
    assert_eq!(b.budget().used, 100);
    b.destroy(ResourceId::Buffer(buf), FenceValue::ZERO);
    assert_eq!(b.budget().used, 0);
}

#[test]
fn creation_refuses_bad_descriptions() {
    let b = backend();
    let max = b.caps().limits.max_buffer_size;
    assert_eq!(
        b.create_buffer(&starquake_render::BufferDesc {
            size: max + 1,
            usage: COPY
        })
        .map_err(|e| e.to_string()),
        Err(format!("buffer size {} is outside 1..={max}", max + 1))
    );
    let staging = b.caps().limits.max_staging_size;
    assert_eq!(
        b.staging(staging).map(|s| s.len()),
        Ok(usize::try_from(staging).expect("small"))
    );
    assert!(
        b.staging(staging)
            .expect("valid")
            .bytes()
            .iter()
            .all(|v| *v == 0)
    );
    assert_eq!(
        b.staging(staging + 1)
            .map(|s| s.len())
            .map_err(|e| e.to_string()),
        Err(format!(
            "staging size {} is outside 1..={staging}",
            staging + 1
        ))
    );
    let f = PixelFormat::R8Unorm;
    let t = texture(&b, f, (1, 1), 1, 1);
    let swizzled = ViewDesc {
        format: f,
        dimension: ViewDimension::D2,
        base_mip: 0,
        mip_count: 1,
        base_layer: 0,
        layer_count: 1,
        swizzle: Swizzle {
            r: starquake_render::resource::SwizzleSource::G,
            ..Swizzle::IDENTITY
        },
    };
    assert_eq!(
        b.create_view(t, &swizzled).map_err(|e| e.to_string()),
        Err("the device lacks view swizzles".to_owned())
    );
    let bad_sampler = starquake_render::SamplerDesc {
        mag_filter: starquake_render::resource::FilterMode::Linear,
        min_filter: starquake_render::resource::FilterMode::Linear,
        mip_filter: starquake_render::resource::FilterMode::Linear,
        address: [starquake_render::resource::AddressMode::Repeat; 3],
        border_color: starquake_render::resource::BorderColor::OpaqueBlack,
        lod_bias: 0.0,
        min_lod: 1.0,
        max_lod: 0.0,
        max_anisotropy: 1,
        compare: None,
        reduction: starquake_render::resource::ReductionMode::Average,
    };
    assert_eq!(
        b.create_sampler(&bad_sampler).map_err(|e| e.to_string()),
        Err("a sampler's minimum level of detail is above its maximum".to_owned())
    );
}

#[test]
fn present_target_is_offscreen() {
    let b = backend();
    assert_eq!(b.set_present_target(PresentTarget::Offscreen), Ok(()));
}

/// The snapshot of one small run, pinned with `insta`. A change to it must be explained in the
/// PR that makes it (decision log, 2026-11-11: snapshot changes are read).
#[test]
fn snapshot_of_a_small_run() {
    let backend = backend();
    let b = &backend;
    b.set_present_target(PresentTarget::Offscreen)
        .expect("offscreen");
    let buf = buffer(b, 16, COPY | BufferUsage::VERTEX);
    let format = PixelFormat::A8B8G8R8Unorm;
    let tex = texture(b, format, (2, 2), 1, 1);
    let tview = view(b, tex, format, 0, (0, 1));
    let sampler = b
        .create_sampler(&starquake_render::SamplerDesc {
            mag_filter: starquake_render::resource::FilterMode::Nearest,
            min_filter: starquake_render::resource::FilterMode::Nearest,
            mip_filter: starquake_render::resource::FilterMode::Nearest,
            address: [starquake_render::resource::AddressMode::ClampToEdge; 3],
            border_color: starquake_render::resource::BorderColor::TransparentBlack,
            lod_bias: 0.0,
            min_lod: 0.0,
            max_lod: 0.0,
            max_anisotropy: 1,
            compare: None,
            reduction: starquake_render::resource::ReductionMode::Average,
        })
        .expect("valid");
    let pipe = graphics(b);
    let comp = compute(b);
    let mut targets = RenderTargets::default();
    targets.colors[0] = Some(tview);
    let mut draw_desc = draw(pipe);
    draw_desc.vertex_buffers = vec![starquake_render::command::BufferSlice {
        buffer: buf,
        offset: 0,
        size: 16,
    }];
    draw_desc.bindings = vec![starquake_render::command::Binding {
        group: 0,
        binding: 1,
        resource: starquake_render::command::BindingResource::Sampler(sampler),
    }];
    submit(
        b,
        vec![
            Command::Marker(starquake_render::command::Marker { submit: 7, draw: 0 }),
            upload(&[1, 2, 3, 4], slice(buf, 0, 4)),
            clear(
                tview,
                ClearValue::Float([0.0, 0.0, 0.0, 1.0]),
                Some(rect(0, 0, 1, 1)),
            ),
            Command::SetRenderTargets(targets),
            Command::Draw(draw_desc),
            Command::Dispatch(starquake_render::command::DispatchDesc {
                pipeline: comp,
                bindings: Vec::new(),
                params: starquake_render::command::DispatchParams::Direct { x: 2, y: 1, z: 1 },
            }),
            Command::Readback {
                from: region(tex, 0, (0, 0, 0), (2, 2, 1)),
                id: rb(3),
            },
            Command::Present(PresentDesc {
                view: tview,
                src_rect: rect(0, 0, 2, 2),
                transform: PresentTransform::FlipVertical,
                frame: 1,
                dump: true,
            }),
        ],
    );
    b.destroy(ResourceId::Texture(tex), FenceValue::new(2));
    b.destroy(ResourceId::Buffer(buf), FenceValue::ZERO);
    b.destroy(ResourceId::Buffer(buf), FenceValue::ZERO);
    submit(b, Vec::new());
    insta::assert_snapshot!(b.snapshot());
}
