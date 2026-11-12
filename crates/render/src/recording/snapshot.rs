//! The recording backend's text snapshot: one line per call or command, in call order.
//!
//! Ids print as a kind letter and the slot index (`b0` buffer, `t` texture, `v` view, `s`
//! sampler, `sh` shader, `p` pipeline, `r` readback), with `@` and the generation when it is not
//! zero. Flag sets print their flag names. Uploaded bytes print as their length and FNV-1a hash
//! (64-bit; offset basis and prime from the FNV reference, `draft-eastlake-fnv`), so a snapshot
//! stays short and still changes when the bytes do. Nested state (samplers, pipelines, draw
//! state) prints as Rust's derived `Debug` form: complete, and stable for a given build.

use super::{DestroyOutcome, Event};
use crate::backend::PresentTarget;
use crate::command::{
    Binding, BindingResource, BufferSlice, ClearValue, Command, CopyRegion, DispatchParams,
    DrawParams, Rect,
};
use crate::id::{
    BufferId, PipelineId, ReadbackId, ResourceId, SamplerId, ShaderId, TextureId, ViewId,
};
use crate::pipeline::{ColorWrites, ComputePipelineDesc, EmulationFlags, GraphicsPipelineDesc};
use crate::resource::{
    BufferDesc, BufferUsage, SamplerDesc, Swizzle, SwizzleSource, TextureDesc, TextureUsage,
    ViewDesc,
};
use crate::shader::{ShaderDesc, ShaderSource};

/// FNV-1a 64-bit offset basis.
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
/// FNV-1a 64-bit prime.
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// FNV-1a over `bytes`.
pub(super) fn fnv1a(bytes: &[u8]) -> u64 {
    bytes.iter().fold(FNV_OFFSET, |h, b| {
        (h ^ u64::from(*b)).wrapping_mul(FNV_PRIME)
    })
}

fn id(prefix: &str, index: u32, generation: u32) -> String {
    if generation == 0 {
        format!("{prefix}{index}")
    } else {
        format!("{prefix}{index}@{generation}")
    }
}

fn buffer_id(i: BufferId) -> String {
    id("b", i.index(), i.generation())
}
fn texture_id(i: TextureId) -> String {
    id("t", i.index(), i.generation())
}
fn view_id(i: ViewId) -> String {
    id("v", i.index(), i.generation())
}
fn sampler_id(i: SamplerId) -> String {
    id("s", i.index(), i.generation())
}
fn shader_id(i: ShaderId) -> String {
    id("sh", i.index(), i.generation())
}
fn pipeline_id(i: PipelineId) -> String {
    id("p", i.index(), i.generation())
}
fn readback_id(i: ReadbackId) -> String {
    id("r", i.index(), i.generation())
}

fn resource_id(r: ResourceId) -> String {
    match r {
        ResourceId::Buffer(i) => buffer_id(i),
        ResourceId::Texture(i) => texture_id(i),
        ResourceId::View(i) => view_id(i),
        ResourceId::Sampler(i) => sampler_id(i),
        ResourceId::Shader(i) => shader_id(i),
        ResourceId::Pipeline(i) => pipeline_id(i),
    }
}

/// Flag names joined by `|`, or `none`.
fn names<T: Copy + PartialEq>(named: &[(&str, T)], contains: impl Fn(T) -> bool) -> String {
    let set: Vec<&str> = named
        .iter()
        .filter(|(_, f)| contains(*f))
        .map(|(n, _)| *n)
        .collect();
    if set.is_empty() {
        "none".to_owned()
    } else {
        set.join("|")
    }
}

/// A buffer description.
pub(super) fn buffer(d: &BufferDesc) -> String {
    format!(
        "buffer {} bytes, usage {}",
        d.size,
        names(BufferUsage::NAMED, |f| d.usage.contains(f))
    )
}

/// A texture description.
pub(super) fn texture(d: &TextureDesc) -> String {
    format!(
        "texture {:?} {} {}x{}x{}, {} mips, {} samples, usage {}",
        d.dimension,
        d.format,
        d.size.width,
        d.size.height,
        d.size.depth_or_layers,
        d.mip_levels,
        d.sample_count,
        names(TextureUsage::NAMED, |f| d.usage.contains(f))
    )
}

/// A swizzle as four letters: the source of red, green, blue and alpha (`0` and `1` for the
/// constants).
fn swizzle(s: Swizzle) -> String {
    [s.r, s.g, s.b, s.a]
        .iter()
        .map(|c| match c {
            SwizzleSource::Zero => '0',
            SwizzleSource::One => '1',
            SwizzleSource::R => 'R',
            SwizzleSource::G => 'G',
            SwizzleSource::B => 'B',
            SwizzleSource::A => 'A',
        })
        .collect()
}

/// A view description.
pub(super) fn view(texture: TextureId, d: &ViewDesc) -> String {
    format!(
        "view of {} as {} {:?}, mips {}+{}, layers {}+{}, swizzle {}",
        texture_id(texture),
        d.format,
        d.dimension,
        d.base_mip,
        d.mip_count,
        d.base_layer,
        d.layer_count,
        swizzle(d.swizzle)
    )
}

/// A sampler description.
pub(super) fn sampler(d: &SamplerDesc) -> String {
    format!("sampler {d:?}")
}

/// A shader description: stage, entry point, code kind and size, and interface counts.
pub(super) fn shader(d: &ShaderDesc) -> String {
    let code = match &d.source {
        ShaderSource::Naga { module, .. } => format!(
            "naga module ({} functions, {} entry points)",
            module.functions.len(),
            module.entry_points.len()
        ),
        ShaderSource::SpirV(words) => format!("SPIR-V {} words", words.len()),
        ShaderSource::Msl(text) => format!("MSL {} bytes", text.len()),
    };
    format!(
        "shader {} {:?}, {code}, resources {:?}, inputs {:?}, outputs {:?}",
        d.stage.name(),
        d.entry,
        d.interface.resources,
        d.interface.inputs,
        d.interface.outputs
    )
}

/// A graphics pipeline description.
pub(super) fn graphics_pipeline(d: &GraphicsPipelineDesc) -> String {
    let s = &d.stages;
    let stages: Vec<String> = [
        ("vertex", Some(s.vertex)),
        ("tess control", s.tess_control),
        ("tess eval", s.tess_eval),
        ("geometry", s.geometry),
        ("fragment", s.fragment),
    ]
    .into_iter()
    .filter_map(|(name, id)| id.map(|id| format!("{name} {}", shader_id(id))))
    .collect();
    // Only the colour targets the pipeline has, each with its blend state.
    let colors: Vec<String> = d
        .targets
        .colors
        .iter()
        .zip(&d.blend)
        .enumerate()
        .filter_map(|(i, (format, blend))| {
            format.map(|f| {
                let writes = names(ColorWrites::NAMED, |w| blend.write_mask.contains(w));
                if blend.enabled {
                    format!(
                        "{i}={f} blend {:?} {:?} writes {writes}",
                        blend.color, blend.alpha
                    )
                } else {
                    format!("{i}={f} writes {writes}")
                }
            })
        })
        .collect();
    format!(
        "graphics pipeline {}, {:?}, restart {}, vertex input {:?}, raster {:?}, depth/stencil \
         {:?}, colour [{}], depth/stencil target {}, {} samples, logic op {:?}, emulation {}",
        stages.join(", "),
        d.topology,
        d.primitive_restart,
        d.vertex_input,
        d.raster,
        d.depth_stencil,
        colors.join(", "),
        d.targets
            .depth_stencil
            .map_or_else(|| "none".to_owned(), |f| f.to_string()),
        d.targets.sample_count,
        d.logic_op,
        names(EmulationFlags::NAMED, |f| d.emulation.contains(f))
    )
}

/// A compute pipeline description.
pub(super) fn compute_pipeline(d: ComputePipelineDesc) -> String {
    format!("compute pipeline of {}", shader_id(d.shader))
}

fn slice(s: &BufferSlice) -> String {
    let end = s.offset.saturating_add(s.size);
    format!("{}[{}..{}]", buffer_id(s.buffer), s.offset, end)
}

fn region(r: &CopyRegion) -> String {
    match r {
        CopyRegion::Buffer(s) => slice(s),
        CopyRegion::Texture(t) => format!(
            "{} mip {} at ({},{},{}) size {}x{}x{}",
            texture_id(t.texture),
            t.mip,
            t.origin.x,
            t.origin.y,
            t.origin.z,
            t.extent.width,
            t.extent.height,
            t.extent.depth_or_layers
        ),
    }
}

fn rect(r: &Rect) -> String {
    format!("({},{} {}x{})", r.x, r.y, r.width, r.height)
}

fn binding(b: &Binding) -> String {
    let what = match &b.resource {
        BindingResource::Buffer(s) => slice(s),
        BindingResource::View(v) => view_id(*v),
        BindingResource::Sampler(s) => sampler_id(*s),
    };
    format!("{}:{}={what}", b.group, b.binding)
}

fn list<T>(items: &[T], f: impl Fn(&T) -> String) -> String {
    let parts: Vec<String> = items.iter().map(f).collect();
    format!("[{}]", parts.join(", "))
}

fn clear_value(v: &ClearValue) -> String {
    match v {
        ClearValue::Float(c) => format!("float {c:?}"),
        ClearValue::Sint(c) => format!("sint {c:?}"),
        ClearValue::Uint(c) => format!("uint {c:?}"),
        ClearValue::Depth(d) => format!("depth {d:?}"),
        ClearValue::Stencil(s) => format!("stencil {s}"),
        ClearValue::DepthStencil { depth, stencil } => {
            format!("depth {depth:?} stencil {stencil}")
        }
    }
}

/// One command.
pub(super) fn command(c: &Command) -> String {
    match c {
        Command::Upload { from, to } => format!(
            "upload {} bytes fnv1a {:#018x} to {}",
            from.len(),
            fnv1a(from.bytes()),
            region(to)
        ),
        Command::Copy(d) => format!("copy {} to {}", region(&d.src), region(&d.dst)),
        Command::Blit(b) => format!(
            "blit {} {} to {} {} {:?}",
            view_id(b.src),
            rect(&b.src_rect),
            view_id(b.dst),
            rect(&b.dst_rect),
            b.filter
        ),
        Command::Clear(d) => format!(
            "clear {} to {} in {}",
            view_id(d.target),
            clear_value(&d.value),
            d.rect
                .as_ref()
                .map_or_else(|| "the whole view".to_owned(), rect)
        ),
        Command::SetRenderTargets(t) => {
            let colors: Vec<String> = t
                .colors
                .iter()
                .enumerate()
                .filter_map(|(i, v)| v.map(|v| format!("{i}={}", view_id(v))))
                .collect();
            format!(
                "set render targets colour [{}] depth/stencil {}",
                colors.join(", "),
                t.depth_stencil.map_or_else(|| "none".to_owned(), view_id)
            )
        }
        Command::Draw(d) => {
            let params = match d.params {
                DrawParams::Indirect {
                    buffer,
                    offset,
                    draw_count,
                    stride,
                    indexed,
                } => format!(
                    "indirect from {} at {offset}, {draw_count} draws, stride {stride}, indexed {indexed}",
                    buffer_id(buffer)
                ),
                other => format!("{other:?}"),
            };
            format!(
                "draw {} {params}, bindings {}, vertex buffers {}, index {}, dynamic {:?}",
                pipeline_id(d.pipeline),
                list(&d.bindings, binding),
                list(&d.vertex_buffers, slice),
                d.index_buffer.as_ref().map_or_else(
                    || "none".to_owned(),
                    |i| format!("{} {:?}", slice(&i.slice), i.format)
                ),
                d.dynamic
            )
        }
        Command::Dispatch(d) => {
            let params = match d.params {
                DispatchParams::Direct { x, y, z } => format!("{x}x{y}x{z} groups"),
                DispatchParams::Indirect { buffer, offset } => {
                    format!("indirect from {} at {offset}", buffer_id(buffer))
                }
            };
            format!(
                "dispatch {} {params}, bindings {}",
                pipeline_id(d.pipeline),
                list(&d.bindings, binding)
            )
        }
        Command::Readback { from, id } => {
            format!("readback {} as {}", region(from), readback_id(*id))
        }
        Command::Present(p) => format!(
            "present {} {} {:?} frame {}{}",
            view_id(p.view),
            rect(&p.src_rect),
            p.transform,
            p.frame,
            if p.dump { " dump" } else { "" }
        ),
        Command::Marker(m) => format!("marker submit {} draw {}", m.submit, m.draw),
    }
}

/// The whole log.
pub(super) fn render(log: &[Event]) -> String {
    let mut lines = Vec::new();
    for event in log {
        match event {
            Event::Created(id, line) => lines.push(format!("create {}: {line}", resource_id(*id))),
            Event::Destroy { id, after, outcome } => {
                let what = match outcome {
                    DestroyOutcome::Now => "now",
                    DestroyOutcome::Pending => "pending",
                    DestroyOutcome::Refused => "refused",
                };
                lines.push(format!(
                    "destroy {} after fence {}: {what}",
                    resource_id(*id),
                    after.get()
                ));
            }
            Event::Freed(id) => lines.push(format!("freed {}", resource_id(*id))),
            Event::Submitted(fence, l) => {
                lines.push(format!("submit fence {}:", fence.get()));
                lines.extend(l.commands.iter().map(|c| format!("  {}", command(c))));
            }
            Event::Target(PresentTarget::Offscreen) => {
                lines.push("present target offscreen".to_owned());
            }
        }
    }
    let mut out = String::new();
    for line in lines {
        out.push_str(&line);
        out.push('\n');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::fnv1a;

    #[test]
    fn fnv1a_reference_values() {
        // FNV-1a 64 of the empty string is the offset basis; of "a" it is 0xaf63dc4c8601ec8c
        // (the FNV reference's test vectors).
        assert_eq!(fnv1a(b""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(fnv1a(b"a"), 0xaf63_dc4c_8601_ec8c);
        assert_eq!(fnv1a(b"foobar"), 0x8594_4171_f739_67e8);
    }
}
