//! Draws and dispatches with no GPU: the wgpu rules mirrored in [`rules`], and the list check of
//! render targets, draws and dispatches ([`check`]) against made-up resources and pipelines.

use super::rules::{
    BindingLimits, BufferKind, Subresources, Uses, VertexStep, check_buffer_binding,
    check_indirect, check_viewport, clamp_scissor, filtering, indirect_draw_size,
    sample_type_fits, vertex_limits, within,
};
use super::*;
use crate::fixtures::{draw_limits, formats, sampler, test_caps, texture, view};
use crate::pipeline::{PipelineEntry, Registry};
use crate::plan::{self, Checked, Rules, Step, TextureRec, ViewRec};
use starquake_render::command::{
    Binding, BindingResource, Command, CommandList, DispatchDesc, DispatchParams, DrawDesc,
    DrawParams, DynamicValues, IndexBufferBinding, IndexFormat, Marker, RenderTargets,
};
use starquake_render::pipeline::DepthBias;
use starquake_render::resource::{CompareFunction, FilterMode, ViewDimension};
use starquake_render::{
    BufferDesc, BufferUsage, PixelFormat, RenderError, TextureUsage, ViewDesc, validate,
};
use std::collections::BTreeSet;

// ---------------------------------------------------------------------------------------------
// The mirrored rules.

#[test]
fn vertex_limits_follow_wgpu_cores_formula() {
    let step = |stride, last_stride, instance| VertexStep {
        stride,
        last_stride,
        instance,
    };
    // (size - last) / stride + 1: 64 bytes of 8-byte elements whose last needs 8 bytes.
    assert_eq!(vertex_limits(&[step(8, 8, false)], &[Some(64)]), (8, u64::MAX));
    assert_eq!(vertex_limits(&[step(8, 8, false)], &[Some(63)]), (7, u64::MAX));
    // A last element wider than the stride needs its whole width.
    assert_eq!(vertex_limits(&[step(8, 12, false)], &[Some(64)]), (7, u64::MAX));
    // Too short for one element: none.
    assert_eq!(vertex_limits(&[step(8, 12, false)], &[Some(11)]), (0, u64::MAX));
    assert_eq!(vertex_limits(&[step(8, 12, false)], &[Some(12)]), (1, u64::MAX));
    // A stride of 0 limits nothing once one element fits.
    assert_eq!(vertex_limits(&[step(0, 4, false)], &[Some(4)]), (u64::MAX, u64::MAX));
    assert_eq!(vertex_limits(&[step(0, 4, false)], &[Some(3)]), (0, u64::MAX));
    // Instance buffers limit instances; the smallest of each kind wins.
    let steps = [step(8, 8, false), step(4, 4, true), step(16, 8, false), step(4, 4, true)];
    let sizes = [Some(64), Some(40), Some(72), Some(12)];
    assert_eq!(vertex_limits(&steps, &sizes), (5, 3));
    // A slot left unbound: both limits 0, as wgpu's.
    assert_eq!(vertex_limits(&steps, &sizes[..3]), (0, 0));
    assert_eq!(
        vertex_limits(&steps, &[Some(64), None, Some(72), Some(12)]),
        (0, 0)
    );
    assert!(within(3, 5, 8) && !within(3, 6, 8));
    assert!(within(u32::MAX, u32::MAX, u64::MAX), "no overflow");
}

#[test]
fn viewports_are_checked_as_wgpu_sets_them() {
    let max = 16_u32;
    let v = |x, y, width, height, min_depth, max_depth| Viewport {
        x,
        y,
        width,
        height,
        min_depth,
        max_depth,
    };
    let ok = [
        v(0.0, 0.0, 16.0, 16.0, 0.0, 1.0),
        v(0.0, 0.0, 0.0, 0.0, 0.5, 0.5),
        v(-32.0, -32.0, 16.0, 16.0, 0.0, 1.0),
        v(15.0, 15.0, 16.0, 16.0, 0.0, 1.0),
    ];
    for vp in ok {
        assert_eq!(check_viewport(&vp, max), Ok(()), "{vp:?}");
    }
    let refused = [
        v(0.0, 0.0, -1.0, 16.0, 0.0, 1.0),
        v(0.0, 0.0, 16.0, -1.0, 0.0, 1.0),
        v(0.0, 0.0, 17.0, 16.0, 0.0, 1.0),
        v(0.0, 0.0, 16.0, 17.0, 0.0, 1.0),
        v(-32.5, 0.0, 1.0, 1.0, 0.0, 1.0),
        v(0.0, -32.5, 1.0, 1.0, 0.0, 1.0),
        v(16.0, 0.0, 16.0, 16.0, 0.0, 1.0),
        v(0.0, 16.0, 16.0, 16.0, 0.0, 1.0),
        v(0.0, 0.0, 1.0, 1.0, -0.5, 1.0),
        v(0.0, 0.0, 1.0, 1.0, 0.0, 1.5),
        v(0.0, 0.0, 1.0, 1.0, 0.75, 0.25),
    ];
    for vp in refused {
        assert_eq!(
            check_viewport(&vp, max),
            Err(RenderError::Unsupported(VIEWPORT)),
            "{vp:?}"
        );
    }
}

#[test]
fn scissors_are_cut_to_the_target() {
    let r = |x, y, width, height| Rect {
        x,
        y,
        width,
        height,
    };
    assert_eq!(clamp_scissor(None, 8, 4), r(0, 0, 8, 4));
    assert_eq!(clamp_scissor(Some(&r(1, 1, 2, 2)), 8, 4), r(1, 1, 2, 2));
    assert_eq!(clamp_scissor(Some(&r(0, 0, 8, 4)), 8, 4), r(0, 0, 8, 4));
    assert_eq!(clamp_scissor(Some(&r(6, 3, 100, 100)), 8, 4), r(6, 3, 2, 1));
    assert_eq!(clamp_scissor(Some(&r(9, 5, 1, 1)), 8, 4), r(8, 4, 0, 0));
    assert_eq!(
        clamp_scissor(Some(&r(0, 0, u32::MAX, u32::MAX)), 8, 4),
        r(0, 0, 8, 4)
    );
}

#[test]
fn indirect_parameters_stay_in_their_buffer() {
    let draw = indirect_draw_size(false);
    let indexed = indirect_draw_size(true);
    assert_eq!((draw, indexed), (16, 20));
    let range = Err(RenderError::Invalid(INDIRECT_RANGE));
    let stride = Err(RenderError::Invalid(INDIRECT_STRIDE));
    assert_eq!(check_indirect(0, 0, 1, draw, 16), Ok(()));
    assert_eq!(check_indirect(0, 0, 1, draw, 15), range);
    assert_eq!(check_indirect(4, 0, 1, draw, 20), Ok(()));
    assert_eq!(check_indirect(4, 0, 1, draw, 19), range);
    assert_eq!(check_indirect(2, 0, 1, draw, 64), range, "off a word");
    // Two draws 16 bytes apart end at 32.
    assert_eq!(check_indirect(0, 16, 2, draw, 32), Ok(()));
    assert_eq!(check_indirect(0, 16, 2, draw, 31), range);
    assert_eq!(check_indirect(0, 12, 2, draw, 64), stride, "below the size");
    assert_eq!(check_indirect(0, 18, 2, draw, 64), stride, "not whole words");
    assert_eq!(check_indirect(0, 20, 2, indexed, 40), Ok(()));
    assert_eq!(check_indirect(0, 16, 2, indexed, 64), stride);
    // One draw's stride is not read.
    assert_eq!(check_indirect(0, 3, 1, draw, 16), Ok(()));
    // Both faults: the offset is reported first.
    assert_eq!(check_indirect(2, 3, 2, draw, 64), range);
    let words = u64::MAX - 3;
    assert_eq!(check_indirect(0, words, 3, draw, u64::MAX), range, "overflow");
}

#[test]
fn bound_buffers_are_checked_in_wgpus_order() {
    let limits = BindingLimits {
        uniform_alignment: 256,
        storage_alignment: 64,
        max_uniform: 1024,
        max_storage: 2048,
    };
    let check = |kind, offset, size, min| check_buffer_binding(kind, offset, size, min, &limits);
    use BufferKind::{Storage, Uniform};
    assert_eq!(check(Uniform, 256, 1024, 16), Ok(()));
    assert_eq!(check(Uniform, 128, 16, 16), Err(RenderError::Invalid(BUFFER_OFFSET)));
    assert_eq!(check(Storage, 128, 16, 16), Ok(()), "storage alignment");
    assert_eq!(check(Storage, 32, 16, 16), Err(RenderError::Invalid(BUFFER_OFFSET)));
    assert_eq!(check(Uniform, 0, 6, 0), Ok(()), "a uniform range of any size");
    assert_eq!(check(Storage, 0, 6, 0), Err(RenderError::Invalid(STORAGE_SIZE)));
    assert_eq!(check(Uniform, 0, 1025, 0), Err(RenderError::Unsupported(BUFFER_RANGE)));
    assert_eq!(check(Storage, 0, 2048, 0), Ok(()));
    assert_eq!(check(Storage, 0, 2052, 0), Err(RenderError::Unsupported(BUFFER_RANGE)));
    assert_eq!(check(Uniform, 0, 15, 16), Err(RenderError::Invalid(BINDING_TOO_SMALL)));
    assert_eq!(check(Uniform, 0, 16, 16), Ok(()));
    // Faults together: offset, then the storage size, then the limit, then the minimum.
    assert_eq!(check(Storage, 1, 6, 0), Err(RenderError::Invalid(BUFFER_OFFSET)));
    assert_eq!(check(Storage, 0, 2050, 0), Err(RenderError::Invalid(STORAGE_SIZE)));
    assert_eq!(check(Uniform, 0, 1025, 2000), Err(RenderError::Unsupported(BUFFER_RANGE)));
}

#[test]
fn sample_types_fit_as_wgpu_matches_them() {
    use wgpu::TextureSampleType as T;
    let types = [
        T::Float { filterable: true },
        T::Float { filterable: false },
        T::Depth,
        T::Sint,
        T::Uint,
    ];
    // wgpu-core's match, written out: (entry, view) pairs that fit.
    let fits = [
        (T::Float { filterable: true }, T::Float { filterable: true }),
        (T::Float { filterable: false }, T::Float { filterable: true }),
        (T::Float { filterable: false }, T::Float { filterable: false }),
        (T::Float { filterable: false }, T::Depth),
        (T::Depth, T::Depth),
        (T::Sint, T::Sint),
        (T::Uint, T::Uint),
    ];
    for entry in types {
        for view in types {
            assert_eq!(
                sample_type_fits(entry, view),
                fits.contains(&(entry, view)),
                "{entry:?} {view:?}"
            );
        }
    }
}

#[test]
fn a_sampler_filters_when_any_filter_is_linear() {
    let (n, l) = (FilterMode::Nearest, FilterMode::Linear);
    assert!(!filtering(n, n, n));
    assert!(filtering(l, n, n));
    assert!(filtering(n, l, n));
    assert!(filtering(n, n, l));
}

#[test]
fn uses_follow_wgpus_one_exclusive_or_any_inclusive_rule() {
    let b = BufferId::new(1, 0);
    let other = BufferId::new(2, 0);
    let mut u = Uses::new();
    u.buffer(b, wgpu::BufferUses::VERTEX);
    u.buffer(b, wgpu::BufferUses::UNIFORM);
    u.buffer(other, wgpu::BufferUses::STORAGE_READ_WRITE);
    u.buffer(other, wgpu::BufferUses::STORAGE_READ_WRITE);
    assert!(u.is_valid(), "reads together, and one write twice");
    u.buffer(other, wgpu::BufferUses::INDEX);
    assert!(!u.is_valid(), "a write and a read");

    let t = TextureId::new(3, 0);
    let part = |m0, m1, l0, l1| Subresources {
        mips: m0..m1,
        layers: l0..l1,
    };
    let mut u = Uses::new();
    u.texture(t, &part(0, 1, 0, 1), wgpu::TextureUses::COLOR_TARGET);
    u.texture(t, &part(1, 3, 0, 1), wgpu::TextureUses::RESOURCE);
    u.texture(t, &part(0, 1, 1, 2), wgpu::TextureUses::RESOURCE);
    assert!(u.is_valid(), "another mip, another layer");
    let mut clash = u.clone();
    clash.texture(t, &part(0, 2, 0, 1), wgpu::TextureUses::RESOURCE);
    assert!(!clash.is_valid(), "mips 0..2 meet the target's mip 0");

    // `fits` judges without adding; `merge` adds.
    let mut pass = Uses::new();
    pass.buffer(b, wgpu::BufferUses::STORAGE_READ_WRITE);
    let mut draw = Uses::new();
    draw.buffer(b, wgpu::BufferUses::VERTEX);
    assert!(!pass.fits(&draw));
    assert!(pass.is_valid(), "fits changes nothing");
    let mut reads = Uses::new();
    reads.buffer(other, wgpu::BufferUses::VERTEX);
    assert!(pass.fits(&reads));
    pass.merge(&reads);
    assert!(pass.is_valid());
    assert!(!pass.fits(&{
        let mut w = Uses::new();
        w.buffer(other, wgpu::BufferUses::STORAGE_READ_WRITE);
        w
    }));
    let mut tex_pass = Uses::new();
    tex_pass.texture(t, &part(0, 1, 0, 1), wgpu::TextureUses::COLOR_TARGET);
    let mut sample_other_mip = Uses::new();
    sample_other_mip.texture(t, &part(1, 2, 0, 1), wgpu::TextureUses::RESOURCE);
    assert!(tex_pass.fits(&sample_other_mip));
    let mut sample_same = Uses::new();
    sample_same.texture(t, &part(0, 1, 0, 1), wgpu::TextureUses::RESOURCE);
    assert!(!tex_pass.fits(&sample_same));
}

#[test]
fn index_buffers_also_get_the_storage_usage() {
    assert_eq!(
        extra_buffer_usages(BufferUsage::INDEX),
        wgpu::BufferUsages::STORAGE
    );
    assert_eq!(
        extra_buffer_usages(BufferUsage::INDEX.union(BufferUsage::VERTEX)),
        wgpu::BufferUsages::STORAGE
    );
    for other in [
        BufferUsage::VERTEX,
        BufferUsage::UNIFORM,
        BufferUsage::STORAGE,
        BufferUsage::INDIRECT,
        BufferUsage::COPY_SRC,
        BufferUsage::COPY_DST,
        BufferUsage::EMPTY,
    ] {
        assert_eq!(extra_buffer_usages(other), wgpu::BufferUsages::empty());
    }
}

// ---------------------------------------------------------------------------------------------
// The list check.

/// Resources and pipelines for the list check: an 8 by 8 RGBA target and a 4 by 4 one, a
/// sampled texture, buffers of each usage, two samplers, and pipelines whose facts are written
/// by hand.
struct World {
    tables: plan::Tables,
    pipelines: Registry,
    target: ViewId,
    small: ViewId,
    picture: ViewId,
    picture_array: ViewId,
    /// A sampled `R32_UINT` texture, which samples as unsigned integers.
    integers: ViewId,
    /// A sampled `D32_FLOAT_S8_UINT` texture.
    depth_stencil: ViewId,
    uniform: BufferId,
    storage: BufferId,
    vertices: BufferId,
    indices: BufferId,
    args: BufferId,
    plain: BufferId,
    linear: SamplerId,
    shadow: SamplerId,
    /// Triangles into one RGBA target; a uniform at (0, 0) of 16 bytes, a filterable 2D texture
    /// at (0, 1), a filtering sampler at (0, 2); one vertex buffer of 8-byte elements.
    graphics: PipelineId,
    /// [`World::graphics`] drawing quads, with no vertex buffer.
    quads: PipelineId,
    /// [`World::graphics`] drawing triangle strips.
    strips: PipelineId,
    /// [`World::graphics`] whose layout has a writable storage buffer at (0, 3) as well.
    writer: PipelineId,
    /// A compute pipeline with a writable storage buffer at (0, 0).
    compute: PipelineId,
    /// [`World::graphics`] whose layout has a binding array of textures at (0, 4) as well.
    arrays: PipelineId,
}

const RGBA: PixelFormat = PixelFormat::A8B8G8R8Unorm;

fn add_buffer(tables: &mut plan::Tables, size: u64, usage: BufferUsage) -> BufferId {
    let (i, g) = tables
        .buffers
        .insert(BufferDesc { size, usage })
        .expect("room");
    BufferId::new(i, g)
}

fn add_view(tables: &mut plan::Tables, desc: starquake_render::TextureDesc, v: ViewDesc) -> ViewId {
    let host = crate::rules::check_texture(&test_caps(), &formats(), wgpu::Features::all(), &desc)
        .expect("a texture the test device makes");
    let (i, g) = tables
        .textures
        .insert(TextureRec {
            desc,
            host,
            bytes: 0,
        })
        .expect("room");
    let texture = TextureId::new(i, g);
    let (i, g) = tables
        .views
        .insert(ViewRec { texture, desc: v })
        .expect("room");
    ViewId::new(i, g)
}

fn entry(binding: u32, ty: wgpu::BindingType, min_size: u64) -> EntryInfo {
    EntryInfo {
        binding,
        ty,
        array: false,
        min_size,
    }
}

const UNIFORM_ENTRY: wgpu::BindingType = wgpu::BindingType::Buffer {
    ty: wgpu::BufferBindingType::Uniform,
    has_dynamic_offset: false,
    min_binding_size: None,
};
const STORAGE_ENTRY: wgpu::BindingType = wgpu::BindingType::Buffer {
    ty: wgpu::BufferBindingType::Storage { read_only: false },
    has_dynamic_offset: false,
    min_binding_size: None,
};
const TEXTURE_ENTRY: wgpu::BindingType = wgpu::BindingType::Texture {
    sample_type: wgpu::TextureSampleType::Float { filterable: true },
    view_dimension: wgpu::TextureViewDimension::D2,
    multisampled: false,
};
const SAMPLER_ENTRY: wgpu::BindingType =
    wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering);

fn graphics_info(topology: Topology, vertex: Vec<VertexStep>, extra: &[EntryInfo]) -> PipelineInfo {
    let mut group = vec![
        entry(0, UNIFORM_ENTRY, 16),
        entry(1, TEXTURE_ENTRY, 0),
        entry(2, SAMPLER_ENTRY, 0),
    ];
    group.extend_from_slice(extra);
    PipelineInfo::Graphics(Box::new(GraphicsInfo {
        topology,
        primitive_restart: false,
        colors: vec![Some(wgpu::TextureFormat::Rgba8Unorm)],
        depth_stencil: None,
        samples: 1,
        vertex,
        layout: LayoutInfo {
            groups: vec![group],
        },
    }))
}

fn add_pipeline(pipelines: &mut Registry, info: PipelineInfo) -> PipelineId {
    let (i, g) = pipelines
        .pipelines
        .insert(PipelineEntry {
            cell: Arc::new(PipelineCell::new()),
            info: Arc::new(info),
        })
        .expect("room");
    PipelineId::new(i, g)
}

fn world() -> World {
    let mut tables = plan::Tables::default();
    let rt = TextureUsage::RENDER_TARGET.union(TextureUsage::SAMPLED);
    let target = add_view(&mut tables, texture(RGBA, 8, 8, rt), view(RGBA, ViewDimension::D2));
    let small = add_view(&mut tables, texture(RGBA, 4, 4, rt), view(RGBA, ViewDimension::D2));
    let picture = add_view(
        &mut tables,
        texture(RGBA, 4, 4, TextureUsage::SAMPLED),
        view(RGBA, ViewDimension::D2),
    );
    let picture_array = add_view(
        &mut tables,
        texture(RGBA, 4, 4, TextureUsage::SAMPLED),
        view(RGBA, ViewDimension::D2Array),
    );
    let uniform = add_buffer(&mut tables, 1024, BufferUsage::UNIFORM);
    let storage = add_buffer(
        &mut tables,
        1024,
        BufferUsage::STORAGE.union(BufferUsage::VERTEX),
    );
    let vertices = add_buffer(&mut tables, 64, BufferUsage::VERTEX);
    let indices = add_buffer(&mut tables, 64, BufferUsage::INDEX);
    let args = add_buffer(&mut tables, 64, BufferUsage::INDIRECT);
    let plain = add_buffer(&mut tables, 1024, BufferUsage::COPY_SRC);
    let (i, g) = tables.samplers.insert(sampler()).expect("room");
    let linear = SamplerId::new(i, g);
    let (i, g) = tables
        .samplers
        .insert(starquake_render::SamplerDesc {
            compare: Some(CompareFunction::Less),
            ..sampler()
        })
        .expect("room");
    let shadow = SamplerId::new(i, g);
    let mut pipelines = Registry::default();
    let eight = vec![VertexStep {
        stride: 8,
        last_stride: 8,
        instance: false,
    }];
    let graphics = add_pipeline(
        &mut pipelines,
        graphics_info(Topology::Triangles, eight.clone(), &[]),
    );
    let quads = add_pipeline(&mut pipelines, graphics_info(Topology::Quads, Vec::new(), &[]));
    let strips = add_pipeline(
        &mut pipelines,
        graphics_info(Topology::TriangleStrip, eight.clone(), &[]),
    );
    let writer = add_pipeline(
        &mut pipelines,
        graphics_info(Topology::Triangles, eight.clone(), &[entry(3, STORAGE_ENTRY, 4)]),
    );
    let compute = add_pipeline(
        &mut pipelines,
        PipelineInfo::Compute(LayoutInfo {
            groups: vec![vec![entry(0, STORAGE_ENTRY, 4)]],
        }),
    );
    let array = EntryInfo {
        array: true,
        ..entry(4, TEXTURE_ENTRY, 0)
    };
    let arrays = add_pipeline(
        &mut pipelines,
        graphics_info(Topology::Triangles, eight, &[array]),
    );
    let integers = add_view(
        &mut tables,
        texture(PixelFormat::R32Uint, 4, 4, TextureUsage::SAMPLED),
        view(PixelFormat::R32Uint, ViewDimension::D2),
    );
    let depth_stencil = add_view(
        &mut tables,
        texture(PixelFormat::D32FloatS8Uint, 4, 4, TextureUsage::SAMPLED),
        view(PixelFormat::D32FloatS8Uint, ViewDimension::D2),
    );
    World {
        tables,
        pipelines,
        target,
        small,
        picture,
        picture_array,
        integers,
        depth_stencil,
        uniform,
        storage,
        vertices,
        indices,
        args,
        plain,
        linear,
        shadow,
        graphics,
        quads,
        strips,
        writer,
        compute,
        arrays,
    }
}

fn binding(binding: u32, resource: BindingResource) -> Binding {
    Binding {
        group: 0,
        binding,
        resource,
    }
}

fn buffer(buffer: BufferId, offset: u64, size: u64) -> BufferSlice {
    BufferSlice {
        buffer,
        offset,
        size,
    }
}

fn dynamic() -> DynamicValues {
    DynamicValues {
        viewports: vec![Viewport {
            x: 0.0,
            y: 0.0,
            width: 8.0,
            height: 8.0,
            min_depth: 0.0,
            max_depth: 1.0,
        }],
        scissors: Vec::new(),
        blend_constant: [0.25, 0.5, 0.75, 1.0],
        stencil_reference: 3,
        stencil_read_mask: !0,
        stencil_write_mask: !0,
        depth_bias: DepthBias::default(),
        line_width: 1.0,
    }
}

impl World {
    /// A draw the check accepts: three vertices with the graphics pipeline.
    fn draw(&self) -> DrawDesc {
        DrawDesc {
            pipeline: self.graphics,
            bindings: vec![
                binding(0, BindingResource::Buffer(buffer(self.uniform, 256, 16))),
                binding(1, BindingResource::View(self.picture)),
                binding(2, BindingResource::Sampler(self.linear)),
            ],
            vertex_buffers: vec![buffer(self.vertices, 0, 64)],
            index_buffer: None,
            dynamic: dynamic(),
            params: DrawParams::Direct {
                vertex_count: 3,
                instance_count: 1,
                first_vertex: 0,
                first_instance: 0,
            },
        }
    }

    fn targets(&self, view: ViewId) -> Command {
        let mut t = RenderTargets::default();
        t.colors[0] = Some(view);
        Command::SetRenderTargets(t)
    }

    fn check_with(&self, limits: &DrawLimits, commands: Vec<Command>) -> Result<Checked, RenderError> {
        let caps = test_caps();
        let rules = Rules {
            caps: &caps,
            depth_copies: true,
            draws: limits,
        };
        plan::check(
            &self.tables,
            &self.pipelines,
            &rules,
            &BTreeSet::new(),
            CommandList { commands },
        )
    }

    fn check(&self, commands: Vec<Command>) -> Result<Checked, RenderError> {
        self.check_with(&draw_limits(&test_caps()), commands)
    }

    /// The error of the list `[targets, draw]` after `fault` changes them.
    fn refusal(&self, fault: &dyn Fn(&Self, &mut Vec<Command>, &mut DrawDesc)) -> Option<RenderError> {
        let mut before = vec![self.targets(self.target)];
        let mut draw = self.draw();
        fault(self, &mut before, &mut draw);
        before.push(Command::Draw(draw));
        self.check(before).err()
    }

    fn draw_steps(&self, commands: Vec<Command>) -> Vec<DrawStep> {
        self.check(commands)
            .expect("accepted")
            .steps
            .into_iter()
            .filter_map(|s| match s {
                Step::Draw(d) => Some(*d),
                _ => None,
            })
            .collect()
    }
}

/// A fault in a draw or the list before it.
type Fault = fn(&World, &mut Vec<Command>, &mut DrawDesc);

/// One fault per step of the order the `draw` module docs give, each with its exact error.
fn faults() -> Vec<(&'static str, Fault, RenderError)> {
    vec![
        (
            "1: the shared checks",
            |_, _, d| d.dynamic.scissors = vec![Rect::default(); 2],
            RenderError::Invalid("a draw needs one scissor per viewport, or none"),
        ),
        (
            "2: a compute pipeline",
            |w, _, d| d.pipeline = w.compute,
            RenderError::Invalid(NEEDS_GRAPHICS),
        ),
        (
            "3: a bound buffer without the usage",
            |w, _, d| d.bindings[0].resource = BindingResource::Buffer(buffer(w.plain, 256, 16)),
            RenderError::MissingUsage {
                resource: "bound buffer",
                usage: "uniform or storage",
            },
        ),
        (
            "4: no targets",
            |_, before, _| before.clear(),
            RenderError::Invalid(NO_TARGETS),
        ),
        (
            "5: a comparison sampler",
            |w, _, d| d.bindings[2].resource = BindingResource::Sampler(w.shadow),
            RenderError::Invalid(SAMPLER_KIND),
        ),
        (
            "6: no vertex buffer",
            |_, _, d| d.vertex_buffers.clear(),
            RenderError::Invalid(MISSING_VERTEX_BUFFER),
        ),
        (
            "7: a flipped viewport",
            |_, _, d| d.dynamic.viewports[0].height = -8.0,
            RenderError::Unsupported(VIEWPORT),
        ),
        (
            "8: a vertex past the buffer",
            |_, _, d| {
                d.params = DrawParams::Direct {
                    vertex_count: 9,
                    instance_count: 1,
                    first_vertex: 0,
                    first_instance: 0,
                }
            },
            RenderError::Invalid(VERTEX_LIMIT),
        ),
        (
            "9: the target sampled",
            |w, _, d| d.bindings[1].resource = BindingResource::View(w.target),
            RenderError::Invalid(USAGE_CONFLICT),
        ),
    ]
}

/// Each step's fault alone gives its error, and with any later step's fault as well, still its
/// own: every pair of the documented order.
#[test]
fn checks_refuse_in_order() {
    let w = world();
    assert!(w.check(vec![w.targets(w.target), Command::Draw(w.draw())]).is_ok());
    let faults = faults();
    for (name, fault, error) in &faults {
        assert_eq!(w.refusal(&|w, b, d| fault(w, b, d)).as_ref(), Some(error), "{name}");
    }
    for (i, (first, early, error)) in faults.iter().enumerate() {
        for (second, late, _) in &faults[i + 1..] {
            let both = |w: &World, b: &mut Vec<Command>, d: &mut DrawDesc| {
                late(w, b, d);
                early(w, b, d);
            };
            assert_eq!(w.refusal(&both).as_ref(), Some(error), "{first} with {second}");
        }
    }
}

/// The refusals of steps 2 to 9 not in [`faults`], each alone with its exact error.
#[test]
fn every_draw_refusal_names_its_rule() {
    let w = world();
    let cases: Vec<(&str, Fault, RenderError)> = vec![
        (
            "a stale pipeline",
            |w, _, d| d.pipeline = PipelineId::new(w.graphics.index(), 9),
            RenderError::Invalid(validate::NO_PIPELINE),
        ),
        (
            "a stale sampler",
            |w, _, d| {
                d.bindings[2].resource = BindingResource::Sampler(SamplerId::new(w.linear.index(), 9))
            },
            RenderError::Invalid(plan::NO_SAMPLER),
        ),
        (
            "a vertex buffer without the usage",
            |w, _, d| d.vertex_buffers[0] = buffer(w.plain, 0, 64),
            RenderError::MissingUsage {
                resource: "vertex buffer",
                usage: "vertex",
            },
        ),
        (
            "targets of two sizes",
            |w, b, _| {
                let mut t = RenderTargets::default();
                t.colors[0] = Some(w.target);
                t.colors[1] = Some(w.small);
                *b = vec![Command::SetRenderTargets(t)];
            },
            RenderError::Unsupported(TARGETS_DIFFER),
        ),
        (
            "targets the pipeline was not made for",
            |w, b, _| {
                let mut t = RenderTargets::default();
                t.colors[1] = Some(w.target);
                *b = vec![Command::SetRenderTargets(t)];
            },
            RenderError::Invalid(TARGET_MISMATCH),
        ),
        (
            "a binding the layout lacks",
            |w, _, d| d.bindings.push(binding(7, BindingResource::Sampler(w.linear))),
            RenderError::Invalid(EXTRA_BINDING),
        ),
        (
            "a binding left out",
            |_, _, d| {
                d.bindings.pop();
            },
            RenderError::Invalid(MISSING_BINDING),
        ),
        (
            "a sampler where the uniform goes",
            |w, _, d| d.bindings[0].resource = BindingResource::Sampler(w.linear),
            RenderError::Invalid(BINDING_KIND),
        ),
        (
            "a uniform off the alignment",
            |w, _, d| d.bindings[0].resource = BindingResource::Buffer(buffer(w.uniform, 16, 16)),
            RenderError::Invalid(BUFFER_OFFSET),
        ),
        (
            "a uniform smaller than its declaration",
            |w, _, d| d.bindings[0].resource = BindingResource::Buffer(buffer(w.uniform, 0, 15)),
            RenderError::Invalid(BINDING_TOO_SMALL),
        ),
        (
            "an array view for a 2D binding",
            |w, _, d| d.bindings[1].resource = BindingResource::View(w.picture_array),
            RenderError::Invalid(VIEW_KIND),
        ),
        (
            "a vertex buffer off a word",
            |w, _, d| d.vertex_buffers[0] = buffer(w.storage, 2, 64),
            RenderError::Invalid(VERTEX_OFFSET),
        ),
        (
            "indices past the index buffer",
            |w, _, d| {
                d.index_buffer = Some(IndexBufferBinding {
                    slice: buffer(w.indices, 0, 64),
                    format: IndexFormat::U16,
                });
                d.params = DrawParams::Indexed {
                    index_count: 30,
                    instance_count: 1,
                    first_index: 3,
                    base_vertex: 0,
                    first_instance: 0,
                };
            },
            RenderError::Invalid(INDEX_LIMIT),
        ),
        (
            "16-bit indices off their size",
            |w, _, d| {
                d.index_buffer = Some(IndexBufferBinding {
                    slice: buffer(w.indices, 1, 32),
                    format: IndexFormat::U16,
                });
                d.params = DrawParams::Indexed {
                    index_count: 3,
                    instance_count: 1,
                    first_index: 0,
                    base_vertex: 0,
                    first_instance: 0,
                };
            },
            RenderError::Invalid(INDEX_OFFSET),
        ),
        (
            "an indirect quad draw",
            |w, _, d| {
                d.pipeline = w.quads;
                d.vertex_buffers.clear();
                d.params = DrawParams::Indirect {
                    buffer: w.args,
                    offset: 0,
                    draw_count: 1,
                    stride: 16,
                    indexed: false,
                };
            },
            RenderError::Unsupported(INDIRECT_REWRITE),
        ),
        (
            "more indirect draws than the cap",
            |w, _, d| {
                d.params = DrawParams::Indirect {
                    buffer: w.args,
                    offset: 0,
                    draw_count: MAX_INDIRECT_DRAWS + 1,
                    stride: 16,
                    indexed: false,
                };
            },
            RenderError::Unsupported(INDIRECT_DRAWS),
        ),
        (
            "indirect parameters past the buffer",
            |w, _, d| {
                d.params = DrawParams::Indirect {
                    buffer: w.args,
                    offset: 52,
                    draw_count: 1,
                    stride: 16,
                    indexed: false,
                };
            },
            RenderError::Invalid(INDIRECT_RANGE),
        ),
        (
            "a binding array in the layout",
            |w, _, d| d.pipeline = w.arrays,
            RenderError::Unsupported(BINDING_ARRAY),
        ),
        (
            "unsigned integers for a float binding",
            |w, _, d| d.bindings[1].resource = BindingResource::View(w.integers),
            RenderError::Invalid(VIEW_KIND),
        ),
        (
            "a view of depth and stencil",
            |w, _, d| d.bindings[1].resource = BindingResource::View(w.depth_stencil),
            RenderError::Unsupported(DEPTH_STENCIL_VIEW),
        ),
        (
            "a buffer written as storage and read as vertices by one draw",
            |w, _, d| {
                d.pipeline = w.writer;
                d.bindings
                    .push(binding(3, BindingResource::Buffer(buffer(w.storage, 0, 64))));
                d.vertex_buffers[0] = buffer(w.storage, 0, 64);
            },
            RenderError::Invalid(USAGE_CONFLICT),
        ),
    ];
    for (name, fault, error) in cases {
        assert_eq!(
            w.refusal(&|w, b, d| fault(w, b, d)),
            Some(error),
            "{name}"
        );
    }
}

/// The edges of the counts: the last vertex and index that fit, and one past each.
#[test]
fn counts_are_checked_at_their_edges() {
    let w = world();
    let direct = |vertex_count, first_vertex| {
        let mut d = w.draw();
        d.params = DrawParams::Direct {
            vertex_count,
            instance_count: 1,
            first_vertex,
            first_instance: 0,
        };
        w.check(vec![w.targets(w.target), Command::Draw(d)]).err()
    };
    // 64 bytes of 8-byte vertices: 8.
    assert_eq!(direct(8, 0), None);
    assert_eq!(direct(9, 0), Some(RenderError::Invalid(VERTEX_LIMIT)));
    assert_eq!(direct(1, 7), None);
    assert_eq!(direct(1, 8), Some(RenderError::Invalid(VERTEX_LIMIT)));
    assert_eq!(direct(u32::MAX, 1), Some(RenderError::Invalid(VERTEX_LIMIT)));
    let indexed = |count, first| {
        let mut d = w.draw();
        d.index_buffer = Some(IndexBufferBinding {
            slice: buffer(w.indices, 0, 64),
            format: IndexFormat::U32,
        });
        d.params = DrawParams::Indexed {
            index_count: count,
            instance_count: 1,
            first_index: first,
            base_vertex: 0,
            first_instance: 0,
        };
        w.check(vec![w.targets(w.target), Command::Draw(d)]).err()
    };
    // 64 bytes of 32-bit indices: 16.
    assert_eq!(indexed(16, 0), None);
    assert_eq!(indexed(17, 0), Some(RenderError::Invalid(INDEX_LIMIT)));
    assert_eq!(indexed(2, 14), None);
    assert_eq!(indexed(2, 15), Some(RenderError::Invalid(INDEX_LIMIT)));
    let indirect = |draw_count| {
        let mut d = w.draw();
        d.params = DrawParams::Indirect {
            buffer: w.args,
            offset: 0,
            draw_count,
            stride: 0,
            indexed: false,
        };
        w.check(vec![w.targets(w.target), Command::Draw(d)]).err()
    };
    assert_eq!(indirect(1), None);
    assert_eq!(indirect(MAX_INDIRECT_DRAWS + 1), Some(RenderError::Unsupported(INDIRECT_DRAWS)));
    // At the cap, the stride is the next check.
    assert_eq!(
        indirect(MAX_INDIRECT_DRAWS),
        Some(RenderError::Invalid(INDIRECT_STRIDE))
    );
}

/// The pass the render thread will have open: a draw starts one after new targets, after any
/// command but a marker or the same targets, after a rewrite, and when it conflicts with an
/// earlier draw of the pass.
#[test]
fn draws_start_a_pass_only_when_they_must() {
    let w = world();
    let draw = || Command::Draw(w.draw());
    let marker = Command::Marker(Marker { submit: 1, draw: 2 });
    let clear = Command::Clear(starquake_render::command::ClearDesc {
        target: w.small,
        value: starquake_render::command::ClearValue::Float([0.0; 4]),
        rect: None,
    });
    let fresh = |commands| -> Vec<bool> {
        w.draw_steps(commands).iter().map(|d| d.fresh_pass).collect()
    };
    assert_eq!(
        fresh(vec![
            w.targets(w.target),
            draw(),
            draw(),
            marker,
            draw(),
            w.targets(w.target),
            draw(),
        ]),
        [true, false, false, false]
    );
    assert_eq!(
        fresh(vec![w.targets(w.target), draw(), clear, draw()]),
        [true, true]
    );
    assert_eq!(
        fresh(vec![
            w.targets(w.target),
            draw(),
            w.targets(w.small),
            w.targets(w.target),
            draw()
        ]),
        [true, true]
    );
    // A quad draw is rewritten first; the triangles after it share its pass.
    let mut quad = w.draw();
    quad.pipeline = w.quads;
    quad.vertex_buffers.clear();
    quad.params = DrawParams::Direct {
        vertex_count: 4,
        instance_count: 1,
        first_vertex: 0,
        first_instance: 0,
    };
    assert_eq!(
        fresh(vec![w.targets(w.target), draw(), Command::Draw(quad), draw()]),
        [true, true, false]
    );
    // The writer writes the storage buffer; a later draw reads it as vertices.
    let mut writes = w.draw();
    writes.pipeline = w.writer;
    writes
        .bindings
        .push(binding(3, BindingResource::Buffer(buffer(w.storage, 0, 64))));
    let mut reads = w.draw();
    reads.vertex_buffers = vec![buffer(w.storage, 0, 64)];
    assert_eq!(
        fresh(vec![
            w.targets(w.target),
            Command::Draw(writes.clone()),
            Command::Draw(writes),
            Command::Draw(reads.clone()),
            Command::Draw(reads),
        ]),
        [true, false, true, false]
    );
}

/// A draw's step: its groups in binding order, the scissor cut to the target, the values the
/// render thread sets.
#[test]
fn a_draw_becomes_this_step() {
    let w = world();
    let mut d = w.draw();
    d.dynamic.scissors = vec![Rect {
        x: 6,
        y: 1,
        width: 10,
        height: 2,
    }];
    // Bindings in another order than the layout's.
    d.bindings.reverse();
    let steps = w.draw_steps(vec![w.targets(w.target), Command::Draw(d.clone())]);
    let expected = DrawStep {
        pipeline: w.graphics,
        cell: Arc::new(PipelineCell::new()),
        fresh_pass: true,
        groups: vec![BoundGroup {
            group: 0,
            entries: vec![
                (0, Bound::Buffer(buffer(w.uniform, 256, 16))),
                (1, Bound::View(w.picture)),
                (2, Bound::Sampler(w.linear, true)),
            ],
        }],
        vertex_buffers: vec![buffer(w.vertices, 0, 64)],
        index: None,
        rewrite: None,
        viewport: d.dynamic.viewports[0],
        scissor: Rect {
            x: 6,
            y: 1,
            width: 2,
            height: 2,
        },
        blend_constant: wgpu::Color {
            r: 0.25,
            g: 0.5,
            b: 0.75,
            a: 1.0,
        },
        stencil_reference: 3,
        call: DrawCall::Vertices {
            vertices: 0..3,
            instances: 0..1,
        },
    };
    assert_eq!(steps, [expected]);
}

/// Rewrites are placed in the list's scratch at the device's alignments, the parameter blocks in
/// one buffer and the outputs in another; a strip draw from 16-bit indices is widened.
#[test]
fn rewrites_are_placed_in_the_lists_scratch() {
    let w = world();
    let limits = draw_limits(&test_caps());
    let uniform = u64::from(limits.binding.uniform_alignment);
    let storage = u64::from(limits.binding.storage_alignment);
    let mut quad = w.draw();
    quad.pipeline = w.quads;
    quad.vertex_buffers.clear();
    quad.params = DrawParams::Direct {
        vertex_count: 8,
        instance_count: 1,
        first_vertex: 5,
        first_instance: 0,
    };
    let mut strip = w.draw();
    strip.pipeline = w.strips;
    strip.index_buffer = Some(IndexBufferBinding {
        slice: buffer(w.indices, 6, 40),
        format: IndexFormat::U16,
    });
    strip.params = DrawParams::Indexed {
        index_count: 5,
        instance_count: 2,
        first_index: 3,
        base_vertex: -1,
        first_instance: 0,
    };
    let checked = w
        .check(vec![
            w.targets(w.target),
            Command::Draw(quad),
            Command::Draw(strip),
        ])
        .expect("accepted");
    let rewrites: Vec<(RewriteStep, DrawCall)> = checked
        .steps
        .iter()
        .filter_map(|s| match s {
            Step::Draw(d) => d.rewrite.map(|r| (r, d.call.clone())),
            _ => None,
        })
        .collect();
    // The strip's first index is at byte 6 + 3 * 2 = 12, bound from 0 with 12 bytes to skip.
    let read: u64 = 12 + 5 * 2;
    assert_eq!(
        rewrites,
        [
            (
                RewriteStep {
                    params_offset: 0,
                    input: None,
                    output_offset: 0,
                    count: 12,
                },
                DrawCall::Indices {
                    indices: 0..12,
                    base_vertex: 0,
                    instances: 0..1,
                }
            ),
            (
                RewriteStep {
                    params_offset: uniform,
                    input: Some((w.indices, 0, read.next_multiple_of(4))),
                    output_offset: 48_u64.next_multiple_of(storage),
                    count: 5,
                },
                DrawCall::Indices {
                    indices: 0..5,
                    base_vertex: -1,
                    instances: 0..2,
                }
            ),
        ]
    );
    assert_eq!(
        checked.rewrites.output,
        48_u64.next_multiple_of(storage) + 20
    );
    let mut params = vec![0; usize::try_from(uniform).expect("small") + rewrite::PARAMS_LEN];
    params[..rewrite::PARAMS_LEN].copy_from_slice(&rewrite::params(
        rewrite::Rewrite {
            mode: rewrite::Mode::Quads,
            source: rewrite::Source::Sequential,
            restart: false,
        },
        12,
        5,
        8,
    ));
    let at = usize::try_from(uniform).expect("small");
    params[at..].copy_from_slice(&rewrite::params(
        rewrite::Rewrite {
            mode: rewrite::Mode::List,
            source: rewrite::Source::U16,
            restart: false,
        },
        5,
        12,
        5,
    ));
    assert_eq!(checked.rewrites.params, params);
}

/// The scratch bound: a list whose rewrite outputs end exactly at the largest buffer is
/// accepted, and one a byte further is refused; so is a rewrite input larger than a storage
/// binding.
#[test]
fn rewrites_are_bounded_by_the_device() {
    let w = world();
    let quad = |n| {
        let mut d = w.draw();
        d.pipeline = w.quads;
        d.vertex_buffers.clear();
        d.params = DrawParams::Direct {
            vertex_count: n,
            instance_count: 1,
            first_vertex: 0,
            first_instance: 0,
        };
        Command::Draw(d)
    };
    let mut limits = draw_limits(&test_caps());
    // Four quads: 24 indices, 96 bytes.
    limits.max_buffer_size = 96;
    assert!(w.check_with(&limits, vec![w.targets(w.target), quad(16)]).is_ok());
    limits.max_buffer_size = 95;
    assert_eq!(
        w.check_with(&limits, vec![w.targets(w.target), quad(16)]).err(),
        Some(RenderError::Unsupported(SCRATCH))
    );
    let mut limits = draw_limits(&test_caps());
    limits.binding.max_storage = 92;
    assert_eq!(
        w.check_with(&limits, vec![w.targets(w.target), quad(16)]).err(),
        Some(RenderError::Unsupported(REWRITE_SIZE))
    );
    limits.binding.max_storage = 96;
    assert!(w.check_with(&limits, vec![w.targets(w.target), quad(16)]).is_ok());
    // A quad draw of three vertices writes no index and draws nothing.
    assert_eq!(
        w.check(vec![w.targets(w.target), quad(3)]).expect("accepted").steps.len(),
        1,
        "only the targets"
    );
}

/// A dispatch: its pipeline must be a compute one, its bindings fit its layout, its indirect
/// parameters lie in their buffer, and it closes the open pass.
#[test]
fn dispatches_are_checked_and_close_the_pass() {
    let w = world();
    let dispatch = |pipeline, params| {
        Command::Dispatch(DispatchDesc {
            pipeline,
            bindings: vec![binding(0, BindingResource::Buffer(buffer(w.storage, 0, 64)))],
            params,
        })
    };
    let direct = DispatchParams::Direct { x: 1, y: 2, z: 3 };
    let checked = w
        .check(vec![
            w.targets(w.target),
            Command::Draw(w.draw()),
            dispatch(w.compute, direct),
            Command::Draw(w.draw()),
        ])
        .expect("accepted");
    let fresh: Vec<bool> = checked
        .steps
        .iter()
        .filter_map(|s| match s {
            Step::Draw(d) => Some(d.fresh_pass),
            _ => None,
        })
        .collect();
    assert_eq!(fresh, [true, true]);
    assert_eq!(
        w.check(vec![dispatch(w.graphics, direct)]).err(),
        Some(RenderError::Invalid(NEEDS_COMPUTE))
    );
    let indirect = |offset| DispatchParams::Indirect {
        buffer: w.args,
        offset,
    };
    assert!(w.check(vec![dispatch(w.compute, indirect(52))]).is_ok());
    assert_eq!(
        w.check(vec![dispatch(w.compute, indirect(56))]).err(),
        Some(RenderError::Invalid(INDIRECT_RANGE))
    );
    // The storage buffer bound as the indirect buffer too: not an indirect buffer.
    let both = Command::Dispatch(DispatchDesc {
        pipeline: w.compute,
        bindings: vec![binding(0, BindingResource::Buffer(buffer(w.storage, 0, 64)))],
        params: DispatchParams::Indirect {
            buffer: w.storage,
            offset: 0,
        },
    });
    assert_eq!(
        w.check(vec![both]).err(),
        Some(RenderError::MissingUsage {
            resource: "indirect buffer",
            usage: "indirect"
        })
    );
}
