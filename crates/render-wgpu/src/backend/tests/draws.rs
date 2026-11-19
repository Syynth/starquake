//! Draws and dispatches on a real GPU (slice R7, `crate::draw`). Every scene has pixels that can
//! be computed: axis-aligned rectangles whose edges lie on pixel edges, in flat colours, drawn
//! into a cleared 8 by 8 target and read back exactly (decision 9). A pixel is covered when its
//! centre is inside the rectangle, so a rectangle from (x0, y0) to (x1, y1) covers exactly the
//! pixels x0..x1 by y0..y1. Topologies the host lacks are compared with the host topology the
//! rewrite turns them into, drawn the same way.

use super::*;
use crate::draw::{
    INDIRECT_REWRITE, NO_TARGETS, TARGET_MISMATCH, USAGE_CONFLICT, VIEWPORT,
};
use crate::fixtures::{graphics_desc, shader_desc, wgsl};
use starquake_render::command::{
    Binding, BindingResource, DrawDesc, DrawParams, DynamicValues, IndexBufferBinding,
    IndexFormat, RenderTargets, Viewport,
};
use starquake_render::pipeline::{
    DepthBias, Topology, VertexAttribute, VertexBufferLayout, VertexComponents, VertexKind,
    VertexLayout, VertexStepMode,
};
use starquake_render::shader::{ResourceBinding, ResourceKind, SampleKind, ShaderInterface};
use starquake_render::{
    CompileJob, CompilePool, ComputePipelineDesc, GraphicsPipelineDesc, ShaderStage,
};

/// The target's side, in pixels.
const SIDE: u32 = 8;
/// The target's format: four bytes a pixel, red first.
const RGBA: PixelFormat = PixelFormat::A8B8G8R8Unorm;
/// What the target is cleared to.
const BLACK: [u8; 4] = [0, 0, 0, 255];
const MAGENTA: [f32; 4] = [1.0, 0.0, 1.0, 1.0];
const GREEN: [f32; 4] = [0.0, 1.0, 0.0, 1.0];

/// Runs each compile job at once on a thread of its own.
pub(super) struct ThreadPool;

impl CompilePool for ThreadPool {
    fn spawn(&self, job: CompileJob) {
        std::thread::spawn(job);
    }
}

/// Maps pixel coordinates (from the top-left corner) to clip space for an 8 by 8 target.
const VERTEX: &str = "
    @vertex fn main(@location(0) p: vec2<f32>) -> @builtin(position) vec4<f32> {
        return vec4(p.x / 4.0 - 1.0, 1.0 - p.y / 4.0, 0.0, 1.0);
    }";

/// One flat colour, from a uniform buffer.
const FLAT: &str = "
    @group(0) @binding(0) var<uniform> colour: vec4<f32>;
    @fragment fn main() -> @location(0) vec4<f32> { return colour; }";

/// The colour of a pixel in a byte for each channel.
fn bytes(colour: [f32; 4]) -> [u8; 4] {
    colour.map(|c| if c > 0.5 { 255 } else { 0 })
}

/// A rectangle's corners in pixels: left, top, right, bottom.
type Rect4 = (f32, f32, f32, f32);

/// The image of rectangles drawn in order over black, each with its colour.
fn image(rects: &[(Rect4, [f32; 4])]) -> Vec<u8> {
    let mut out = Vec::new();
    for y in 0..SIDE {
        for x in 0..SIDE {
            let (cx, cy) = (x as f32 + 0.5, y as f32 + 0.5);
            let mut pixel = BLACK;
            for &((x0, y0, x1, y1), colour) in rects {
                if cx > x0 && cx < x1 && cy > y0 && cy < y1 {
                    pixel = bytes(colour);
                }
            }
            out.extend_from_slice(&pixel);
        }
    }
    out
}

/// Little-endian bytes of `f32`s.
fn floats(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// Little-endian bytes of `u32`s.
fn words(values: &[u32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// The four corners of a rectangle, in the order a quad, a fan and a loop go round it.
fn corners((x0, y0, x1, y1): Rect4) -> [f32; 8] {
    [x0, y0, x1, y0, x1, y1, x0, y1]
}

/// A backend with a compile pool, a target and its view, and the buffers the scenes use.
struct Scene {
    gpu: WgpuBackend,
    target: TextureId,
    view: ViewId,
    vs: ShaderId,
    fs: ShaderId,
    next_readback: u32,
}

const VERTICES: BufferUsage = BufferUsage::VERTEX.union(BufferUsage::COPY_DST);
const INDICES: BufferUsage = BufferUsage::INDEX.union(BufferUsage::COPY_DST);
const UNIFORMS: BufferUsage = BufferUsage::UNIFORM.union(BufferUsage::COPY_DST);

fn scene(test: &str) -> Option<Scene> {
    scene_with(test, WhileCompiling::Wait)
}

fn scene_with(test: &str, setting: WhileCompiling) -> Option<Scene> {
    let gpu = crate::backend::tests::open_with(test, || {
        WgpuBackend::with_compile_pool(
            WgpuConfig::DEFAULT,
            CompletionWake::none(),
            Arc::new(ThreadPool),
            setting,
        )
    })?;
    Some(scene_on(gpu))
}

/// The scene's target, view and shaders, made on `gpu`.
fn scene_on(gpu: WgpuBackend) -> Scene {
    let target = gpu
        .create_texture(&texture_desc(
            TextureDimension::D2,
            RGBA,
            (SIDE, SIDE, 1),
            1,
            TARGET,
        ))
        .expect("target");
    let view = gpu
        .create_view(target, &whole_view(RGBA, ViewDimension::D2, (0, 1)))
        .expect("view");
    let vs = gpu
        .create_shader(shader_desc(
            ShaderStage::Vertex,
            "main",
            wgsl(VERTEX),
            ShaderInterface::default(),
        ))
        .expect("vertex shader");
    let fs = gpu
        .create_shader(shader_desc(
            ShaderStage::Fragment,
            "main",
            wgsl(FLAT),
            uniform_at_zero(),
        ))
        .expect("fragment shader");
    Scene {
        gpu,
        target,
        view,
        vs,
        fs,
        next_readback: 1,
    }
}

/// An interface with a uniform buffer at group 0, binding 0.
fn uniform_at_zero() -> ShaderInterface {
    ShaderInterface {
        resources: vec![ResourceBinding {
            group: 0,
            binding: 0,
            kind: ResourceKind::UniformBuffer,
        }],
        ..ShaderInterface::default()
    }
}

/// A pipeline description reading `vec2<f32>` positions from buffer 0.
fn positions(vs: ShaderId, fs: ShaderId, topology: Topology) -> GraphicsPipelineDesc {
    let mut desc = graphics_desc(vs, Some(fs));
    desc.vertex_input = VertexLayout {
        buffers: vec![VertexBufferLayout {
            stride: 8,
            step: VertexStepMode::Vertex,
        }],
        attributes: vec![VertexAttribute {
            location: 0,
            buffer: 0,
            offset: 0,
            components: VertexComponents::R32G32,
            kind: VertexKind::Float,
        }],
    };
    desc.topology = topology;
    desc
}

/// Per-draw state that draws into the whole 8 by 8 target.
pub(super) fn dynamic() -> DynamicValues {
    DynamicValues {
        viewports: vec![Viewport {
            x: 0.0,
            y: 0.0,
            width: SIDE as f32,
            height: SIDE as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        }],
        scissors: Vec::new(),
        blend_constant: [0.0; 4],
        stencil_reference: 0,
        stencil_read_mask: !0,
        stencil_write_mask: !0,
        depth_bias: DepthBias::default(),
        line_width: 1.0,
    }
}

/// Indices of `format` as bytes.
fn index_bytes(format: IndexFormat, indices: &[u32]) -> Vec<u8> {
    indices
        .iter()
        .flat_map(|&i| match format {
            IndexFormat::U8 => vec![u8::try_from(i).expect("fits")],
            IndexFormat::U16 => u16::try_from(i).expect("fits").to_le_bytes().to_vec(),
            IndexFormat::U32 => i.to_le_bytes().to_vec(),
        })
        .collect()
}

/// How a test draws its vertices.
#[derive(Clone, Debug)]
enum Feed {
    /// Non-indexed.
    Direct,
    /// Indexed with these indices.
    Indexed(IndexFormat, Vec<u32>),
    /// Non-indexed, the parameters read from a buffer.
    Indirect,
}

impl Scene {
    fn buffer(&self, usage: BufferUsage, bytes: &[u8]) -> (BufferId, Command) {
        let size = u64::try_from(bytes.len()).expect("small").next_multiple_of(4);
        let id = self
            .gpu
            .create_buffer(&BufferDesc { size, usage })
            .expect("buffer");
        let mut padded = bytes.to_vec();
        padded.resize(usize::try_from(size).expect("small"), 0);
        (id, upload(padded, slice(id, 0, size)))
    }

    fn pipeline(&self, desc: GraphicsPipelineDesc) -> PipelineId {
        let id = self.gpu.create_graphics_pipeline(desc).expect("accepted");
        assert!(matches!(
            self.gpu.draw_pipeline(id),
            Ok(crate::DrawPipeline::Ready(_))
        ));
        id
    }

    fn targets(&self) -> Command {
        let mut targets = RenderTargets::default();
        targets.colors[0] = Some(self.view);
        Command::SetRenderTargets(targets)
    }

    fn clear(&self) -> Command {
        Command::Clear(ClearDesc {
            target: self.view,
            value: ClearValue::Float([0.0, 0.0, 0.0, 1.0]),
            rect: None,
        })
    }

    fn readback(&mut self) -> (Command, ReadbackId) {
        let id = ReadbackId::new(self.next_readback, 0);
        self.next_readback += 1;
        let from = region(self.target, 0, (0, 0, 0), (SIDE, SIDE, 1));
        (Command::Readback { from, id }, id)
    }

    fn take(&self, id: ReadbackId) -> Vec<u8> {
        match self.gpu.take_readback(id) {
            Some(Readback::Bytes(bytes)) => bytes,
            other => unreachable!("the readback came back as {other:?}"),
        }
    }

    fn run(&self, commands: Vec<Command>) {
        let fence = self.gpu.submit(CommandList { commands }).expect("accepted");
        finish(&self.gpu, fence);
    }

    /// The commands that draw `vertices` (x, y pairs) with `topology` in `colour`, fed as
    /// `feed`, and the buffers' uploads before them.
    fn draw(
        &self,
        topology: Topology,
        restart: bool,
        vertices: &[f32],
        feed: &Feed,
        colour: [f32; 4],
    ) -> Vec<Command> {
        let mut desc = positions(self.vs, self.fs, topology);
        desc.primitive_restart = restart;
        let pipeline = self.pipeline(desc);
        self.draw_with(pipeline, vertices, feed, colour)
    }

    /// [`Scene::draw`] with a pipeline the caller made.
    fn draw_with(
        &self,
        pipeline: PipelineId,
        vertices: &[f32],
        feed: &Feed,
        colour: [f32; 4],
    ) -> Vec<Command> {
        let (vb, up_vertices) = self.buffer(VERTICES, &floats(vertices));
        let (colour_buffer, up_colour) = self.buffer(UNIFORMS, &floats(&colour));
        let count = u32::try_from(vertices.len() / 2).expect("small");
        let mut list = vec![up_vertices, up_colour];
        let (params, index_buffer) = match feed {
            Feed::Direct => (
                DrawParams::Direct {
                    vertex_count: count,
                    instance_count: 1,
                    first_vertex: 0,
                    first_instance: 0,
                },
                None,
            ),
            Feed::Indexed(format, indices) => {
                let bytes = index_bytes(*format, indices);
                let (ib, up) = self.buffer(INDICES, &bytes);
                list.push(up);
                let size = u64::try_from(bytes.len()).expect("small");
                (
                    DrawParams::Indexed {
                        index_count: u32::try_from(indices.len()).expect("small"),
                        instance_count: 1,
                        first_index: 0,
                        base_vertex: 0,
                        first_instance: 0,
                    },
                    Some(IndexBufferBinding {
                        slice: BufferSlice {
                            buffer: ib,
                            offset: 0,
                            size,
                        },
                        format: *format,
                    }),
                )
            }
            Feed::Indirect => {
                let args = words(&[count, 1, 0, 0]);
                let (args_buffer, up) = self.buffer(
                    BufferUsage::INDIRECT.union(BufferUsage::COPY_DST),
                    &args,
                );
                list.push(up);
                (
                    DrawParams::Indirect {
                        buffer: args_buffer,
                        offset: 0,
                        draw_count: 1,
                        stride: 16,
                        indexed: false,
                    },
                    None,
                )
            }
        };
        let size = u64::try_from(vertices.len() * 4).expect("small");
        list.push(Command::Draw(DrawDesc {
            pipeline,
            bindings: vec![Binding {
                group: 0,
                binding: 0,
                resource: BindingResource::Buffer(BufferSlice {
                    buffer: colour_buffer,
                    offset: 0,
                    size: 16,
                }),
            }],
            vertex_buffers: vec![BufferSlice {
                buffer: vb,
                offset: 0,
                size,
            }],
            index_buffer,
            dynamic: dynamic(),
            params,
        }));
        list
    }

    /// Clears the target, runs `draws`, and reads the target back.
    fn pixels(&mut self, draws: Vec<Command>) -> Vec<u8> {
        let (read, id) = self.readback();
        let mut list = vec![self.clear(), self.targets()];
        list.extend(draws);
        list.push(read);
        self.run(list);
        self.take(id)
    }
}

/// Two triangles covering the rectangle, as a triangle list.
fn triangles((x0, y0, x1, y1): Rect4) -> Vec<f32> {
    vec![x0, y0, x1, y0, x1, y1, x0, y0, x1, y1, x0, y1]
}

const RECT: Rect4 = (2.0, 1.0, 6.0, 5.0);
const OTHER: Rect4 = (0.0, 6.0, 3.0, 8.0);

#[test]
fn rectangles_draw_their_exact_pixels() {
    bounded(|| {
        let Some(mut s) = scene("rectangles") else {
            return;
        };
        let mut draws = s.draw(
            Topology::Triangles,
            false,
            &triangles(RECT),
            &Feed::Direct,
            MAGENTA,
        );
        draws.extend(s.draw(
            Topology::Triangles,
            false,
            &triangles(OTHER),
            &Feed::Direct,
            GREEN,
        ));
        assert_eq!(
            s.pixels(draws),
            image(&[(RECT, MAGENTA), (OTHER, GREEN)])
        );
        assert_eq!(s.gpu.host_errors(), 0);
    });
}

/// Quads, quad strips, fans and polygons draw the pixels their triangle-list equivalents do,
/// non-indexed and from 8-bit, 16-bit and 32-bit indices.
#[test]
fn rewritten_topologies_draw_as_their_triangle_equivalents() {
    bounded(|| {
        let Some(mut s) = scene("rewritten_topologies") else {
            return;
        };
        let expected = image(&[(RECT, MAGENTA)]);
        let reference = s.draw(
            Topology::Triangles,
            false,
            &triangles(RECT),
            &Feed::Direct,
            MAGENTA,
        );
        assert_eq!(s.pixels(reference), expected, "the reference");
        let (x0, y0, x1, y1) = RECT;
        let around = corners(RECT).to_vec();
        // A quad strip of two quads: pairs down the left and right edges, top, middle, bottom.
        let ym = (y0 + y1) / 2.0;
        let strip = vec![x0, y0, x1, y0, x0, ym, x1, ym, x0, y1, x1, y1];
        let cases: [(Topology, &Vec<f32>, usize); 4] = [
            (Topology::Quads, &around, 4),
            (Topology::QuadStrip, &strip, 6),
            (Topology::TriangleFan, &around, 4),
            (Topology::Polygon, &around, 4),
        ];
        for (topology, vertices, n) in cases {
            let feeds = [
                Feed::Direct,
                Feed::Indexed(IndexFormat::U8, (0..n).map(|i| i as u32).collect()),
                Feed::Indexed(IndexFormat::U16, (0..n).map(|i| i as u32).collect()),
                Feed::Indexed(IndexFormat::U32, (0..n).map(|i| i as u32).collect()),
            ];
            for feed in feeds {
                let draws = s.draw(topology, false, vertices, &feed, MAGENTA);
                assert_eq!(s.pixels(draws), expected, "{topology:?} {feed:?}");
            }
        }
        assert_eq!(s.gpu.host_errors(), 0);
    });
}

/// A line loop draws the pixels of the line strip that returns to its first vertex, and not
/// those of the open strip.
#[test]
fn a_line_loop_draws_as_its_closed_strip() {
    bounded(|| {
        let Some(mut s) = scene("line_loop") else {
            return;
        };
        // Through pixel centres, so each line lights the pixels it passes through.
        let around = corners((1.5, 1.5, 6.5, 6.5));
        let mut closed = around.to_vec();
        closed.extend_from_slice(&around[..2]);
        let strip = s.draw(Topology::LineStrip, false, &closed, &Feed::Direct, GREEN);
        let closed_pixels = s.pixels(strip);
        let open = s.draw(Topology::LineStrip, false, &around, &Feed::Direct, GREEN);
        let open_pixels = s.pixels(open);
        assert_ne!(closed_pixels, open_pixels, "the closing line lights pixels");
        for feed in [
            Feed::Direct,
            Feed::Indexed(IndexFormat::U8, vec![0, 1, 2, 3]),
            Feed::Indexed(IndexFormat::U16, vec![0, 1, 2, 3]),
            Feed::Indexed(IndexFormat::U32, vec![0, 1, 2, 3]),
        ] {
            let draws = s.draw(Topology::LineLoop, false, &around, &feed, GREEN);
            assert_eq!(s.pixels(draws), closed_pixels, "{feed:?}");
        }
        assert_eq!(s.gpu.host_errors(), 0);
    });
}

/// 8-bit indices draw as 16-bit ones do, a strip as well as a list.
#[test]
fn eight_bit_indices_draw_as_wider_ones() {
    bounded(|| {
        let Some(mut s) = scene("eight_bit_indices") else {
            return;
        };
        let around = corners(RECT);
        // A strip: corners 0, 1, 3, 2 make the two triangles of the rectangle.
        for (topology, indices) in [
            (Topology::Triangles, vec![0, 1, 2, 0, 2, 3]),
            (Topology::TriangleStrip, vec![0, 1, 3, 2]),
        ] {
            for restart in [false, true] {
                let wide = s.draw(
                    topology,
                    restart,
                    &around,
                    &Feed::Indexed(IndexFormat::U16, indices.clone()),
                    MAGENTA,
                );
                let narrow = s.draw(
                    topology,
                    restart,
                    &around,
                    &Feed::Indexed(IndexFormat::U8, indices.clone()),
                    MAGENTA,
                );
                let wide = s.pixels(wide);
                assert_eq!(wide, image(&[(RECT, MAGENTA)]), "{topology:?}");
                assert_eq!(s.pixels(narrow), wide, "{topology:?} restart {restart}");
            }
        }
        assert_eq!(s.gpu.host_errors(), 0);
    });
}

/// With primitive restart off, the 16-bit index `0xFFFF` of a strip is vertex 65535: the
/// rewrite widens the indices, so the host does not restart there.
#[test]
fn restart_off_draws_the_vertex_at_the_restart_value() {
    bounded(|| {
        let Some(mut s) = scene("restart_off") else {
            return;
        };
        let last = usize::from(u16::MAX);
        let mut vertices = vec![0.0; (last + 1) * 2];
        // Triangle (0, 1, 2) covers the top half of RECT; (1, 2, 0xFFFF) the bottom half.
        let (x0, y0, x1, y1) = RECT;
        vertices[..6].copy_from_slice(&[x0, y0, x1, y0, x0, y1]);
        vertices[last * 2..].copy_from_slice(&[x1, y1]);
        let draws = s.draw(
            Topology::TriangleStrip,
            false,
            &vertices,
            &Feed::Indexed(IndexFormat::U16, vec![0, 1, 2, u32::from(u16::MAX)]),
            MAGENTA,
        );
        assert_eq!(s.pixels(draws), image(&[(RECT, MAGENTA)]));
        assert_eq!(s.gpu.host_errors(), 0);
    });
}

/// Indirect draws draw what the direct draw of the same parameters draws, one draw and several
/// `stride` bytes apart.
#[test]
fn indirect_draws_match_direct_ones() {
    bounded(|| {
        let Some(mut s) = scene("indirect") else {
            return;
        };
        let direct = s.draw(
            Topology::Triangles,
            false,
            &triangles(RECT),
            &Feed::Direct,
            MAGENTA,
        );
        let direct = s.pixels(direct);
        let indirect = s.draw(
            Topology::Triangles,
            false,
            &triangles(RECT),
            &Feed::Indirect,
            MAGENTA,
        );
        assert_eq!(s.pixels(indirect), direct);

        // Two indexed draws, 32 bytes apart: the first draws RECT's triangles, the second
        // OTHER's, from one vertex buffer.
        let mut both = triangles(RECT);
        both.extend(triangles(OTHER));
        let mut draws = s.draw(
            Topology::Triangles,
            false,
            &both,
            &Feed::Indexed(IndexFormat::U16, (0..12).collect()),
            MAGENTA,
        );
        let mut params = words(&[6, 1, 0, 0, 0]);
        params.resize(32, 0);
        params.extend(words(&[6, 1, 6, 0, 0]));
        let (args, up) = s.buffer(BufferUsage::INDIRECT.union(BufferUsage::COPY_DST), &params);
        let Some(Command::Draw(draw)) = draws.last_mut() else {
            unreachable!("the last command is the draw");
        };
        draw.params = DrawParams::Indirect {
            buffer: args,
            offset: 0,
            draw_count: 2,
            stride: 32,
            indexed: true,
        };
        draws.insert(0, up);
        assert_eq!(
            s.pixels(draws),
            image(&[(RECT, MAGENTA), (OTHER, MAGENTA)])
        );
        assert_eq!(s.gpu.host_errors(), 0);
    });
}

/// A readback of the target between two draws closes the render pass: it reads the first
/// draw's pixels, and a second readback reads both.
#[test]
fn a_readback_of_the_target_sees_the_draws_before_it() {
    bounded(|| {
        let Some(mut s) = scene("readback_closes_pass") else {
            return;
        };
        let (first_read, first) = s.readback();
        let (second_read, second) = s.readback();
        let mut list = vec![s.clear(), s.targets()];
        list.extend(s.draw(
            Topology::Triangles,
            false,
            &triangles(RECT),
            &Feed::Direct,
            MAGENTA,
        ));
        list.push(first_read);
        list.extend(s.draw(
            Topology::Triangles,
            false,
            &triangles(OTHER),
            &Feed::Direct,
            GREEN,
        ));
        list.push(second_read);
        s.run(list);
        assert_eq!(s.take(first), image(&[(RECT, MAGENTA)]));
        assert_eq!(s.take(second), image(&[(RECT, MAGENTA), (OTHER, GREEN)]));
        assert_eq!(s.gpu.host_errors(), 0);
    });
}

/// wgpu needs a fragment output typed for an integer target: a `u32` output draws exact values
/// into `R32_UINT`, and a float output fails the pipeline, so the typed-output emulation is
/// needed (`crate::draw`, "Typed outputs").
#[test]
fn an_integer_target_needs_a_typed_output() {
    bounded(|| {
        let Some(s) = scene("typed_outputs") else {
            return;
        };
        let target = s
            .gpu
            .create_texture(&texture_desc(
                TextureDimension::D2,
                PixelFormat::R32Uint,
                (SIDE, SIDE, 1),
                1,
                TARGET,
            ))
            .expect("target");
        let view = s
            .gpu
            .create_view(
                target,
                &whole_view(PixelFormat::R32Uint, ViewDimension::D2, (0, 1)),
            )
            .expect("view");
        let typed = s
            .gpu
            .create_shader(shader_desc(
                ShaderStage::Fragment,
                "main",
                wgsl("@fragment fn main() -> @location(0) u32 { return 0xdeadbeefu; }"),
                ShaderInterface::default(),
            ))
            .expect("typed");
        let untyped = s
            .gpu
            .create_shader(shader_desc(
                ShaderStage::Fragment,
                "main",
                wgsl("@fragment fn main() -> @location(0) vec4<f32> { return vec4(1.0); }"),
                ShaderInterface::default(),
            ))
            .expect("untyped");
        let desc = |fs| {
            let mut d = positions(s.vs, fs, Topology::Triangles);
            d.targets.colors[0] = Some(PixelFormat::R32Uint);
            d
        };
        let failed = s.gpu.create_graphics_pipeline(desc(untyped)).expect("accepted");
        assert!(matches!(
            s.gpu.draw_pipeline(failed),
            Ok(crate::DrawPipeline::Skipped(crate::SkipReason::Failed))
        ));
        let pipeline = s.pipeline(desc(typed));
        let (vb, up) = s.buffer(VERTICES, &floats(&triangles(RECT)));
        let mut targets = RenderTargets::default();
        targets.colors[0] = Some(view);
        let id = ReadbackId::new(99, 0);
        s.run(vec![
            up,
            Command::Clear(ClearDesc {
                target: view,
                value: ClearValue::Uint([7, 0, 0, 0]),
                rect: None,
            }),
            Command::SetRenderTargets(targets),
            Command::Draw(DrawDesc {
                pipeline,
                bindings: Vec::new(),
                vertex_buffers: vec![BufferSlice {
                    buffer: vb,
                    offset: 0,
                    size: 48,
                }],
                index_buffer: None,
                dynamic: dynamic(),
                params: DrawParams::Direct {
                    vertex_count: 6,
                    instance_count: 1,
                    first_vertex: 0,
                    first_instance: 0,
                },
            }),
            Command::Readback {
                from: region(target, 0, (0, 0, 0), (SIDE, SIDE, 1)),
                id,
            },
        ]);
        let expected: Vec<u8> = image(&[(RECT, MAGENTA)])
            .chunks(4)
            .flat_map(|p| {
                let value: u32 = if p == BLACK { 7 } else { 0xdead_beef };
                value.to_le_bytes()
            })
            .collect();
        assert_eq!(s.take(id), expected);
        assert_eq!(s.gpu.host_errors(), 0);
    });
}

/// A dispatch writes a storage buffer the next command reads back.
#[test]
fn a_dispatch_writes_its_storage_buffer() {
    bounded(|| {
        let Some(s) = scene("dispatch") else {
            return;
        };
        let cs = s
            .gpu
            .create_shader(shader_desc(
                ShaderStage::Compute,
                "main",
                wgsl(
                    "@group(0) @binding(0) var<storage, read_write> out: array<u32>;
                     @compute @workgroup_size(4) fn main(@builtin(global_invocation_id) i: vec3<u32>) {
                         out[i.x] = i.x * 3u + 1u;
                     }",
                ),
                ShaderInterface {
                    resources: vec![ResourceBinding {
                        group: 0,
                        binding: 0,
                        kind: ResourceKind::StorageBuffer { writable: true },
                    }],
                    ..ShaderInterface::default()
                },
            ))
            .expect("compute shader");
        let pipeline = s
            .gpu
            .create_compute_pipeline(ComputePipelineDesc { shader: cs })
            .expect("accepted");
        let out = s
            .gpu
            .create_buffer(&BufferDesc {
                size: 32,
                usage: BufferUsage::STORAGE.union(COPY),
            })
            .expect("buffer");
        let id = ReadbackId::new(1, 0);
        let binding = |size| Binding {
            group: 0,
            binding: 0,
            resource: BindingResource::Buffer(BufferSlice {
                buffer: out,
                offset: 0,
                size,
            }),
        };
        s.run(vec![
            upload(vec![0xAA; 32], slice(out, 0, 32)),
            Command::Dispatch(DispatchDesc {
                pipeline,
                bindings: vec![binding(32)],
                params: DispatchParams::Direct { x: 2, y: 1, z: 1 },
            }),
            Command::Readback {
                from: slice(out, 0, 32),
                id,
            },
        ]);
        assert_eq!(s.take(id), words(&[1, 4, 7, 10, 13, 16, 19, 22]));
        assert_eq!(s.gpu.host_errors(), 0);
    });
}

/// Holds compile jobs until the test releases them: a scripted compile delay.
#[derive(Default)]
struct HeldPool {
    jobs: std::sync::Mutex<Vec<CompileJob>>,
}

impl HeldPool {
    /// Runs every held job, in order, on one new thread.
    fn release(&self) {
        let jobs = std::mem::take(&mut *self.jobs.lock().expect("not poisoned"));
        std::thread::spawn(move || {
            for job in jobs {
                job();
            }
        });
    }
}

impl CompilePool for HeldPool {
    fn spawn(&self, job: CompileJob) {
        self.jobs.lock().expect("not poisoned").push(job);
    }
}

/// The render thread asks for each draw's pipeline under the backend's setting: under `Skip` a
/// draw whose pipeline is compiling is skipped, counted and draws nothing; under `Wait` the
/// render thread waits for the build and draws. The build is held until the test releases it.
#[test]
fn a_draw_of_a_compiling_pipeline_waits_or_is_skipped() {
    for setting in [WhileCompiling::Skip, WhileCompiling::Wait] {
        bounded(move || {
            let pool = Arc::new(HeldPool::default());
            let same: Arc<HeldPool> = Arc::clone(&pool);
            let held: Arc<dyn CompilePool> = same;
            let Some(gpu) = crate::backend::tests::open_with("compiling_draw", || {
                WgpuBackend::with_compile_pool(
                    WgpuConfig::DEFAULT,
                    CompletionWake::none(),
                    held,
                    setting,
                )
            }) else {
                return;
            };
            let mut s = scene_on(gpu);
            let pipeline = s
                .gpu
                .create_graphics_pipeline(positions(s.vs, s.fs, Topology::Triangles))
                .expect("accepted");
            if setting == WhileCompiling::Skip {
                let draws = s.draw_with(pipeline, &triangles(RECT), &Feed::Direct, MAGENTA);
                assert_eq!(s.pixels(draws), image(&[]), "skipped");
                assert_eq!(s.gpu.pipeline_counts().skipped_while_compiling, 1);
                pool.release();
                let start = std::time::Instant::now();
                while s.gpu.pipeline_status(pipeline) == PipelineStatus::Compiling {
                    assert!(start.elapsed() < LIST_WAIT, "the build settles");
                    std::thread::sleep(Duration::from_millis(1));
                }
            } else {
                // The render thread waits in the draw until the build is released.
                let releaser = Arc::clone(&pool);
                std::thread::spawn(move || {
                    std::thread::sleep(Duration::from_millis(50));
                    releaser.release();
                });
            }
            let draws = s.draw_with(pipeline, &triangles(RECT), &Feed::Direct, MAGENTA);
            assert_eq!(s.pixels(draws), image(&[(RECT, MAGENTA)]), "{setting:?}");
            let skipped = u64::from(setting == WhileCompiling::Skip);
            assert_eq!(s.gpu.pipeline_counts().skipped_while_compiling, skipped);
            assert_eq!(s.gpu.host_errors(), 0);
        });
    }
}

/// A filtering sampler bound where the pipeline's build made the binding non-filtering (the
/// shader samples an unfilterable texture through it) skips the draw and counts it.
#[test]
fn a_filtering_sampler_on_an_unfilterable_texture_skips_the_draw() {
    bounded(|| {
        let Some(mut s) = scene("filtering_skip") else {
            return;
        };
        let fs = s
            .gpu
            .create_shader(shader_desc(
                ShaderStage::Fragment,
                "main",
                wgsl(
                    "@group(0) @binding(0) var t: texture_2d<f32>;
                     @group(0) @binding(1) var s: sampler;
                     @fragment fn main() -> @location(0) vec4<f32> {
                         return textureSampleLevel(t, s, vec2(0.5), 0.0);
                     }",
                ),
                ShaderInterface {
                    resources: vec![
                        ResourceBinding {
                            group: 0,
                            binding: 0,
                            kind: ResourceKind::SampledTexture {
                                dimension: ViewDimension::D2,
                                sample: SampleKind::Float { filterable: false },
                                multisampled: false,
                            },
                        },
                        ResourceBinding {
                            group: 0,
                            binding: 1,
                            kind: ResourceKind::Sampler { comparison: false },
                        },
                    ],
                    ..ShaderInterface::default()
                },
            ))
            .expect("fragment shader");
        let pipeline = s.pipeline(positions(s.vs, fs, Topology::Triangles));
        let image_tex = s
            .gpu
            .create_texture(&texture_desc(
                TextureDimension::D2,
                RGBA,
                (4, 4, 1),
                1,
                TARGET,
            ))
            .expect("texture");
        let image_view = s
            .gpu
            .create_view(image_tex, &whole_view(RGBA, ViewDimension::D2, (0, 1)))
            .expect("view");
        let linear = s
            .gpu
            .create_sampler(&crate::fixtures::sampler())
            .expect("sampler");
        let (vb, up) = s.buffer(VERTICES, &floats(&triangles(RECT)));
        let draw = Command::Draw(DrawDesc {
            pipeline,
            bindings: vec![
                Binding {
                    group: 0,
                    binding: 0,
                    resource: BindingResource::View(image_view),
                },
                Binding {
                    group: 0,
                    binding: 1,
                    resource: BindingResource::Sampler(linear),
                },
            ],
            vertex_buffers: vec![BufferSlice {
                buffer: vb,
                offset: 0,
                size: 48,
            }],
            index_buffer: None,
            dynamic: dynamic(),
            params: DrawParams::Direct {
                vertex_count: 6,
                instance_count: 1,
                first_vertex: 0,
                first_instance: 0,
            },
        });
        assert_eq!(s.pixels(vec![up, draw]), image(&[]));
        assert_eq!(s.gpu.encode_skips(), 1);
        assert_eq!(s.gpu.host_errors(), 0);
    });
}

/// A buffer one draw binds as writable storage and the next draws vertices from cannot share a
/// render pass in wgpu; the second draw starts a new one, and both draw.
#[test]
fn a_conflicting_use_starts_a_new_pass() {
    bounded(|| {
        let Some(mut s) = scene("conflict_new_pass") else {
            return;
        };
        let fs = s
            .gpu
            .create_shader(shader_desc(
                ShaderStage::Fragment,
                "main",
                wgsl(
                    "@group(0) @binding(0) var<uniform> colour: vec4<f32>;
                     @group(0) @binding(1) var<storage, read_write> scratch: array<u32>;
                     @fragment fn main() -> @location(0) vec4<f32> { return colour; }",
                ),
                ShaderInterface {
                    resources: vec![
                        ResourceBinding {
                            group: 0,
                            binding: 0,
                            kind: ResourceKind::UniformBuffer,
                        },
                        ResourceBinding {
                            group: 0,
                            binding: 1,
                            kind: ResourceKind::StorageBuffer { writable: true },
                        },
                    ],
                    ..ShaderInterface::default()
                },
            ))
            .expect("fragment shader");
        let writer = s.pipeline(positions(s.vs, fs, Topology::Triangles));
        let shared = s
            .gpu
            .create_buffer(&BufferDesc {
                size: 48,
                usage: VERTICES.union(BufferUsage::STORAGE),
            })
            .expect("buffer");
        let (colour, up_colour) = s.buffer(UNIFORMS, &floats(&GREEN));
        let (vb, up_vertices) = s.buffer(VERTICES, &floats(&triangles(OTHER)));
        let mut draws = vec![
            up_colour,
            up_vertices,
            upload(floats(&triangles(RECT)), slice(shared, 0, 48)),
        ];
        let at = |buffer| BufferSlice {
            buffer,
            offset: 0,
            size: 48,
        };
        draws.push(Command::Draw(DrawDesc {
            pipeline: writer,
            bindings: vec![
                Binding {
                    group: 0,
                    binding: 0,
                    resource: BindingResource::Buffer(BufferSlice {
                        buffer: colour,
                        offset: 0,
                        size: 16,
                    }),
                },
                Binding {
                    group: 0,
                    binding: 1,
                    resource: BindingResource::Buffer(at(shared)),
                },
            ],
            vertex_buffers: vec![at(vb)],
            index_buffer: None,
            dynamic: dynamic(),
            params: DrawParams::Direct {
                vertex_count: 6,
                instance_count: 1,
                first_vertex: 0,
                first_instance: 0,
            },
        }));
        let mut second = s.draw(
            Topology::Triangles,
            false,
            &triangles(RECT),
            &Feed::Direct,
            MAGENTA,
        );
        let Some(Command::Draw(draw)) = second.last_mut() else {
            unreachable!("the last command is the draw");
        };
        draw.vertex_buffers = vec![at(shared)];
        draws.extend(second);
        assert_eq!(
            s.pixels(draws),
            image(&[(OTHER, GREEN), (RECT, MAGENTA)])
        );
        assert_eq!(s.gpu.host_errors(), 0);
    });
}

/// A few refusals end to end, each the exact error of `submit` (the full order is
/// `draw::tests::checks_refuse_in_order`).
#[test]
fn what_draws_refuse() {
    bounded(|| {
        let Some(s) = scene("draw_refusals") else {
            return;
        };
        let draws = s.draw(
            Topology::Triangles,
            false,
            &triangles(RECT),
            &Feed::Direct,
            MAGENTA,
        );
        let mut uploads = draws.clone();
        let Some(Command::Draw(draw)) = uploads.pop() else {
            unreachable!("the last command is the draw");
        };
        s.run(uploads);
        let submit = |commands: Vec<Command>| s.gpu.submit(CommandList { commands });
        assert_eq!(
            submit(vec![Command::Draw(draw.clone())]),
            Err(RenderError::Invalid(NO_TARGETS))
        );
        let mut flipped = draw.clone();
        flipped.dynamic.viewports[0].height = -(SIDE as f32);
        assert_eq!(
            submit(vec![s.targets(), Command::Draw(flipped)]),
            Err(RenderError::Unsupported(VIEWPORT))
        );
        let mut quads = positions(s.vs, s.fs, Topology::Quads);
        quads.primitive_restart = false;
        let quads = s.pipeline(quads);
        let args = s
            .gpu
            .create_buffer(&BufferDesc {
                size: 16,
                usage: BufferUsage::INDIRECT,
            })
            .expect("buffer");
        let mut indirect = draw.clone();
        indirect.pipeline = quads;
        indirect.params = DrawParams::Indirect {
            buffer: args,
            offset: 0,
            draw_count: 1,
            stride: 16,
            indexed: false,
        };
        assert_eq!(
            submit(vec![s.targets(), Command::Draw(indirect)]),
            Err(RenderError::Unsupported(INDIRECT_REWRITE))
        );
        let mut two = positions(s.vs, s.fs, Topology::Triangles);
        two.targets.colors[1] = Some(RGBA);
        let two = s.pipeline(two);
        let mut mismatched = draw.clone();
        mismatched.pipeline = two;
        assert_eq!(
            submit(vec![s.targets(), Command::Draw(mismatched)]),
            Err(RenderError::Invalid(TARGET_MISMATCH))
        );
        // The target sampled by the draw that renders into it.
        let sampler = s
            .gpu
            .create_sampler(&crate::fixtures::sampler())
            .expect("sampler");
        let fs = s
            .gpu
            .create_shader(shader_desc(
                ShaderStage::Fragment,
                "main",
                wgsl(
                    "@group(0) @binding(0) var t: texture_2d<f32>;
                     @group(0) @binding(1) var s: sampler;
                     @fragment fn main() -> @location(0) vec4<f32> {
                         return textureSampleLevel(t, s, vec2(0.5), 0.0);
                     }",
                ),
                ShaderInterface {
                    resources: vec![
                        ResourceBinding {
                            group: 0,
                            binding: 0,
                            kind: ResourceKind::SampledTexture {
                                dimension: ViewDimension::D2,
                                sample: SampleKind::Float { filterable: true },
                                multisampled: false,
                            },
                        },
                        ResourceBinding {
                            group: 0,
                            binding: 1,
                            kind: ResourceKind::Sampler { comparison: false },
                        },
                    ],
                    ..ShaderInterface::default()
                },
            ))
            .expect("fragment shader");
        let sampling = s.pipeline(positions(s.vs, fs, Topology::Triangles));
        let mut feedback = draw;
        feedback.pipeline = sampling;
        feedback.bindings = vec![
            Binding {
                group: 0,
                binding: 0,
                resource: BindingResource::View(s.view),
            },
            Binding {
                group: 0,
                binding: 1,
                resource: BindingResource::Sampler(sampler),
            },
        ];
        assert_eq!(
            submit(vec![s.targets(), Command::Draw(feedback)]),
            Err(RenderError::Invalid(USAGE_CONFLICT))
        );
        assert_eq!(s.gpu.host_errors(), 0);
    });
}
