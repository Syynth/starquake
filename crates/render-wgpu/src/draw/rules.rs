//! The rules wgpu applies to a draw or dispatch as it is encoded, mirrored as pure functions so
//! the check pass refuses what wgpu would refuse later (`crate::plan`, rule 3). Each cites the
//! wgpu-core 30.0.2 check it mirrors; where this is stricter, the `draw` module docs list it.

use super::{
    BUFFER_OFFSET, BUFFER_RANGE, BINDING_TOO_SMALL, INDIRECT_RANGE, INDIRECT_STRIDE, STORAGE_SIZE,
    VIEWPORT,
};
use starquake_render::command::{Rect, Viewport};
use starquake_render::{BufferId, RenderError, TextureId};
use std::collections::BTreeMap;
use std::ops::Range;

/// One vertex buffer of a pipeline, as wgpu-core's `VertexStep` has it (`src/pipeline.rs`):
/// the stride, the bytes the last element needs, and whether it steps per instance.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VertexStep {
    /// Bytes between elements.
    pub stride: u64,
    /// The largest attribute end, offset plus format size (`create_render_pipeline`'s
    /// `last_stride` in `src/device/resource.rs`).
    pub last_stride: u64,
    /// Per instance rather than per vertex.
    pub instance: bool,
}

/// How many vertices and instances the bound vertex buffers hold: wgpu-core's
/// `VertexLimits::new` (`src/command/render.rs`), with `sizes[i]` the bound size of slot `i`,
/// `None` for a slot left unbound. A missing slot makes both limits 0, as wgpu's does; a slot
/// too short for one element makes its limit 0; a stride of 0 limits nothing.
pub fn vertex_limits(steps: &[VertexStep], sizes: &[Option<u64>]) -> (u64, u64) {
    let mut vertex = u64::MAX;
    let mut instance = u64::MAX;
    for (i, step) in steps.iter().enumerate() {
        let Some(size) = sizes.get(i).copied().flatten() else {
            return (0, 0);
        };
        let limit = if size < step.last_stride {
            0
        } else {
            match (size - step.last_stride).checked_div(step.stride) {
                Some(whole) => whole + 1,
                // A stride of 0 repeats one element, which fits.
                None => continue,
            }
        };
        let slot = if step.instance {
            &mut instance
        } else {
            &mut vertex
        };
        *slot = (*slot).min(limit);
    }
    (vertex, instance)
}

/// Whether `first + count` stays within `limit` (wgpu-core's `validate_vertex_limit`,
/// `validate_instance_limit` and the index check of `draw_indexed`, in `u64`).
pub fn within(first: u32, count: u32, limit: u64) -> bool {
    u64::from(first) + u64::from(count) <= limit
}

/// Checks a viewport as wgpu-core's `set_viewport` does (`src/command/render.rs`): a size from
/// 0 to `max_dimension` (the device's `max_texture_dimension_2d`), a position within twice that
/// on either side, and depths within 0..=1 with the near one not above the far one. The same
/// `f32` arithmetic, so the edges agree.
#[expect(
    clippy::cast_precision_loss,
    reason = "wgpu-core converts the limit to f32 the same way, and the check must agree with it"
)]
pub fn check_viewport(v: &Viewport, max_dimension: u32) -> Result<(), RenderError> {
    let max = max_dimension as f32;
    let range = max * 2.0;
    let size_ok = v.width >= 0.0 && v.height >= 0.0 && v.width <= max && v.height <= max;
    let position_ok = v.x >= -range
        && v.y >= -range
        && v.x + v.width <= range - 1.0
        && v.y + v.height <= range - 1.0;
    let depth_ok = (0.0..=1.0).contains(&v.min_depth)
        && (0.0..=1.0).contains(&v.max_depth)
        && v.min_depth <= v.max_depth;
    if size_ok && position_ok && depth_ok {
        Ok(())
    } else {
        Err(RenderError::Unsupported(VIEWPORT))
    }
}

/// A scissor rectangle within a target of `width` by `height`: `rect` cut to the target, or the
/// whole target for none. wgpu-core's `set_scissor` refuses a rectangle that reaches past the
/// target; the part past it covers no pixel, so cutting it off draws the same.
pub fn clamp_scissor(rect: Option<&Rect>, width: u32, height: u32) -> Rect {
    let Some(r) = rect else {
        return Rect {
            x: 0,
            y: 0,
            width,
            height,
        };
    };
    let x = r.x.min(width);
    let y = r.y.min(height);
    Rect {
        x,
        y,
        width: r.width.min(width - x),
        height: r.height.min(height - y),
    }
}

/// Bytes of one indirect draw's parameters: wgpu's `DrawIndirectArgs` (16) or
/// `DrawIndexedIndirectArgs` (20), the layout of Vulkan's commands.
pub const fn indirect_draw_size(indexed: bool) -> u64 {
    if indexed { 20 } else { 16 }
}

/// Bytes of an indirect dispatch's parameters: wgpu's `DispatchIndirectArgs`, three `u32`s.
pub const INDIRECT_DISPATCH_SIZE: u64 = 12;

/// Checks the parameters of `count` indirect draws or dispatches of `args` bytes each, `stride`
/// bytes apart from `offset`, in a buffer of `size` bytes: the offset on a 4-byte boundary
/// (wgpu-core's `UnalignedIndirectBufferOffset`), the stride at least `args` and a multiple of 4
/// when there are several (Vulkan's rule for `vkCmdDrawIndirect`; wgpu has no stride, so each
/// draw is its own call), and the last parameters inside the buffer (wgpu-core's
/// `IndirectBufferOverrun`, checked per call).
pub fn check_indirect(
    offset: u64,
    stride: u64,
    count: u32,
    args: u64,
    size: u64,
) -> Result<(), RenderError> {
    if !offset.is_multiple_of(4) {
        return Err(RenderError::Invalid(INDIRECT_RANGE));
    }
    if count > 1 && (stride < args || !stride.is_multiple_of(4)) {
        return Err(RenderError::Invalid(INDIRECT_STRIDE));
    }
    let last = stride
        .checked_mul(u64::from(count.saturating_sub(1)))
        .and_then(|s| s.checked_add(offset))
        .and_then(|s| s.checked_add(args));
    match last {
        Some(end) if end <= size => Ok(()),
        _ => Err(RenderError::Invalid(INDIRECT_RANGE)),
    }
}

/// What kind of buffer binding a layout entry takes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BufferKind {
    /// A uniform buffer.
    Uniform,
    /// A storage buffer.
    Storage,
}

/// The device limits a bound buffer range must keep.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BindingLimits {
    /// `min_uniform_buffer_offset_alignment`.
    pub uniform_alignment: u32,
    /// `min_storage_buffer_offset_alignment`.
    pub storage_alignment: u32,
    /// `max_uniform_buffer_binding_size`.
    pub max_uniform: u64,
    /// `max_storage_buffer_binding_size`.
    pub max_storage: u64,
}

/// Checks a buffer range bound as `kind`, in wgpu-core's order (`create_bind_group`'s buffer
/// arm and `create_buffer_binding`, `src/device/resource.rs`): the offset alignment, the
/// storage size alignment (`STORAGE_BINDING_SIZE_ALIGNMENT`, 4), the size limit; then the late
/// size check wgpu makes at the draw (`LateMinBufferBindingSizeMismatch`): at least `min`, the
/// size of the shader's declaration.
pub fn check_buffer_binding(
    kind: BufferKind,
    offset: u64,
    size: u64,
    min: u64,
    limits: &BindingLimits,
) -> Result<(), RenderError> {
    let (alignment, max) = match kind {
        BufferKind::Uniform => (limits.uniform_alignment, limits.max_uniform),
        BufferKind::Storage => (limits.storage_alignment, limits.max_storage),
    };
    if alignment == 0 || !offset.is_multiple_of(alignment.into()) {
        return Err(RenderError::Invalid(BUFFER_OFFSET));
    }
    if kind == BufferKind::Storage && !size.is_multiple_of(4) {
        return Err(RenderError::Invalid(STORAGE_SIZE));
    }
    if size > max {
        return Err(RenderError::Unsupported(BUFFER_RANGE));
    }
    if size < min {
        return Err(RenderError::Invalid(BINDING_TOO_SMALL));
    }
    Ok(())
}

/// Whether a view whose format samples as `view` fits a layout entry that samples as `entry`:
/// wgpu-core's match in `create_texture_binding` (`src/device/resource.rs`), less its last
/// clause (a float format filterable only by an adapter-specific feature), which this backend
/// does not mirror and so refuses (`draw` module docs, "Divergences").
pub const fn sample_type_fits(
    entry: wgpu::TextureSampleType,
    view: wgpu::TextureSampleType,
) -> bool {
    use wgpu::TextureSampleType as T;
    matches!(
        (entry, view),
        (T::Uint, T::Uint)
            | (T::Sint, T::Sint)
            | (T::Depth, T::Depth)
            | (T::Float { filterable: false }, T::Float { .. } | T::Depth)
            | (T::Float { filterable: true }, T::Float { filterable: true })
    )
}

/// Whether a sampler with these filters filters, as wgpu-core decides it (`create_sampler`:
/// any of the three is linear).
pub const fn filtering(
    mag: starquake_render::resource::FilterMode,
    min: starquake_render::resource::FilterMode,
    mip: starquake_render::resource::FilterMode,
) -> bool {
    use starquake_render::resource::FilterMode as F;
    matches!(mag, F::Linear) || matches!(min, F::Linear) || matches!(mip, F::Linear)
}


/// Part of a texture: mip levels and layers (a 3D texture's slices are all layer 0, as wgpu
/// tracks them).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Subresources {
    /// The mip levels.
    pub mips: Range<u32>,
    /// The layers.
    pub layers: Range<u32>,
}

impl Subresources {
    fn meets(&self, other: &Self) -> bool {
        let meet = |a: &Range<u32>, b: &Range<u32>| a.start < b.end && b.start < a.end;
        meet(&self.mips, &other.mips) && meet(&self.layers, &other.layers)
    }
}

/// Whether a resource in all of `state`'s uses at once breaks wgpu-core's usage scope rule,
/// "any inclusive uses, or one exclusive use" (`invalid_resource_state`, `src/track/mod.rs`).
fn invalid_buffer(state: wgpu::BufferUses) -> bool {
    state.intersects(wgpu::BufferUses::EXCLUSIVE) && !state.bits().is_power_of_two()
}

/// [`invalid_buffer`] for a texture's uses.
fn invalid_texture(state: wgpu::TextureUses) -> bool {
    state.intersects(wgpu::TextureUses::EXCLUSIVE) && !state.bits().is_power_of_two()
}

/// The resources one usage scope uses (a render pass, or one dispatch), with wgpu's usage bits,
/// so that a conflict wgpu-core would refuse is found first. Buffers are tracked whole and
/// textures by mip level and layer, as wgpu-core's trackers do (`src/track/`).
///
/// Adding a use never fails: a conflict is remembered ([`Uses::is_valid`]), so a draw's whole
/// set is built first and judged once.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Uses {
    buffers: BTreeMap<BufferId, wgpu::BufferUses>,
    /// Each texture's uses, one entry per distinct part with the union of its uses, so a
    /// texture many draws sample the same way stays one entry.
    textures: BTreeMap<TextureId, Parts>,
    conflict: bool,
}

/// A texture's parts, keyed by mip and layer bounds, each with the union of its uses.
type Parts = BTreeMap<(u32, u32, u32, u32), wgpu::TextureUses>;

impl Uses {
    /// No uses.
    pub fn new() -> Self {
        Self::default()
    }

    /// Whether no two uses added conflict.
    pub const fn is_valid(&self) -> bool {
        !self.conflict
    }

    /// Adds a use of `buffer`.
    pub fn buffer(&mut self, buffer: BufferId, usage: wgpu::BufferUses) {
        let state = self
            .buffers
            .entry(buffer)
            .or_insert(wgpu::BufferUses::empty());
        *state |= usage;
        self.conflict |= invalid_buffer(*state);
    }

    /// Adds a use of `part` of `texture`. The rule holds for a set of uses exactly when it
    /// holds for each pair that shares a subresource (a breaking set has an exclusive use and a
    /// different one), so each new use is checked against each part it meets.
    pub fn texture(&mut self, texture: TextureId, part: &Subresources, usage: wgpu::TextureUses) {
        let parts = self.textures.entry(texture).or_default();
        let clash = parts
            .iter()
            .any(|(&key, &u)| key_part(key).meets(part) && invalid_texture(u | usage));
        self.conflict |= clash;
        *parts
            .entry(part_key(part))
            .or_insert(wgpu::TextureUses::empty()) |= usage;
    }

    /// Whether every use of `other` could join these without a conflict.
    pub fn fits(&self, other: &Self) -> bool {
        let buffers = other.buffers.iter().all(|(id, &usage)| {
            self.buffers
                .get(id)
                .is_none_or(|&state| !invalid_buffer(state | usage))
        });
        let textures = other.textures.iter().all(|(id, theirs)| {
            self.textures.get(id).is_none_or(|ours| {
                theirs.iter().all(|(&k, &usage)| {
                    let part = key_part(k);
                    ours.iter().all(|(&o, &u)| {
                        !key_part(o).meets(&part) || !invalid_texture(u | usage)
                    })
                })
            })
        });
        buffers && textures
    }

    /// Adds every use of `other`.
    pub fn merge(&mut self, other: &Self) {
        for (&id, &usage) in &other.buffers {
            self.buffer(id, usage);
        }
        for (&id, parts) in &other.textures {
            for (&key, &usage) in parts {
                self.texture(id, &key_part(key), usage);
            }
        }
        self.conflict |= other.conflict;
    }
}

/// The map key of a part.
const fn part_key(part: &Subresources) -> (u32, u32, u32, u32) {
    (
        part.mips.start,
        part.mips.end,
        part.layers.start,
        part.layers.end,
    )
}

/// The part a map key names.
const fn key_part((m0, m1, l0, l1): (u32, u32, u32, u32)) -> Subresources {
    Subresources {
        mips: m0..m1,
        layers: l0..l1,
    }
}
