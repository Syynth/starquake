//! Validation of descriptions against [`Caps`], before a host object exists.
//!
//! Every description comes from guest state, so every one is checked: sizes, mip and layer
//! counts, sample counts, format support, binding counts, and copy extents (start, end and the
//! whole region, in 64-bit arithmetic that cannot overflow for `u32` inputs, or checked where it
//! can) (docs/specs/render-backend.md, "Errors and untrusted input"). Each function returns the
//! first problem it finds as a [`RenderError`]; none panics on any input.
//!
//! Where both the device and the guest GPU bound a count (colour targets, vertex attributes and
//! buffers, viewports), the limit is the smaller of [`crate::caps::Limits`] and the guest
//! constant in [`crate::caps`].
//!
//! Resource ids in commands are not checked here: a backend looks up the descriptions an id
//! names and passes them to [`validate_copy`], [`validate_upload`], [`validate_readback`] and
//! [`validate_rect`] as [`Resolved`] values.
//!
//! Rules that are this crate's own choice rather than a device limit, marked as such: a sampled
//! texture's view format must keep the texture's block size and aspect (guess: Vulkan's
//! "size-compatible" view rule, which wgpu narrows further; the wgpu backend may refuse more);
//! texture data outside a texture is tightly packed ([`packed_size`]).

use crate::caps::{
    Caps, Features, FormatFeatures, GUEST_COLOR_TARGETS, GUEST_VERTEX_ATTRIBUTES,
    GUEST_VERTEX_BUFFERS, GUEST_VIEWPORTS, StageSupport,
};
use crate::command::{
    Binding, BufferSlice, DispatchDesc, DispatchParams, DrawDesc, DrawParams, Rect, RenderTargets,
    TextureRegion,
};
use crate::error::RenderError;
use crate::format::{FormatAspect, PixelFormat};
use crate::pipeline::{GraphicsPipelineDesc, PolygonMode, RasterState, Topology, VertexLayout};
use crate::resource::{
    AddressMode, BorderColor, BufferDesc, Extent3d, SamplerDesc, TextureDesc, TextureDimension,
    TextureUsage, ViewDesc, ViewDimension,
};
use crate::shader::{InterfaceVariable, ResourceKind, ShaderDesc, ShaderStage};
use crate::{BufferUsage, ResourceBinding};

/// Faces of a cube, and so layers per cube.
pub const CUBE_FACES: u32 = 6;

/// Most components of a stage input or output.
pub const MAX_INTERFACE_COMPONENTS: u32 = 4;

/// `value` within `min..=max`, or an [`RenderError::OutOfRange`] naming `what`.
fn in_range(what: &'static str, value: u64, min: u64, max: u64) -> Result<(), RenderError> {
    if (min..=max).contains(&value) {
        Ok(())
    } else {
        Err(RenderError::OutOfRange {
            what,
            value,
            min,
            max,
        })
    }
}

/// `index` below `count`, or an [`RenderError::NotBelow`] naming `what`.
fn below(what: &'static str, index: u64, count: u64) -> Result<(), RenderError> {
    if index < count {
        Ok(())
    } else {
        Err(RenderError::NotBelow { what, index, count })
    }
}

/// `end` at most `limit`, or an [`RenderError::PastEnd`] naming `what`.
fn within(what: &'static str, end: u64, limit: u64) -> Result<(), RenderError> {
    if end <= limit {
        Ok(())
    } else {
        Err(RenderError::PastEnd { what, end, limit })
    }
}

fn require(caps: &Caps, feature: Features, name: &'static str) -> Result<(), RenderError> {
    if caps.features.contains(feature) {
        Ok(())
    } else {
        Err(RenderError::MissingFeature(name))
    }
}

fn format_supports(
    caps: &Caps,
    format: PixelFormat,
    needed: FormatFeatures,
    usage: &'static str,
) -> Result<(), RenderError> {
    if caps.formats.get(format).contains(needed) {
        Ok(())
    } else {
        Err(RenderError::FormatUnsupported { format, usage })
    }
}

/// Colour targets usable at once: the device's limit, or the guest's 8 if smaller.
pub fn color_target_limit(caps: &Caps) -> u32 {
    caps.limits.max_color_attachments.min(guest_color_targets())
}

/// Vertex attributes usable at once: the device's limit, or the guest's 32 if smaller.
pub fn vertex_attribute_limit(caps: &Caps) -> u32 {
    caps.limits
        .max_vertex_attributes
        .min(GUEST_VERTEX_ATTRIBUTES)
}

/// Vertex buffers usable at once: the device's limit, or the guest's 32 if smaller.
pub fn vertex_buffer_limit(caps: &Caps) -> u32 {
    caps.limits.max_vertex_buffers.min(GUEST_VERTEX_BUFFERS)
}

/// Viewports usable at once: the device's limit, or the guest's 16 if smaller.
pub fn viewport_limit(caps: &Caps) -> u32 {
    caps.limits.max_viewports.min(GUEST_VIEWPORTS)
}

fn guest_color_targets() -> u32 {
    // GUEST_COLOR_TARGETS is 8; the conversion cannot fail, and saturates if it ever could.
    u32::try_from(GUEST_COLOR_TARGETS).unwrap_or(u32::MAX)
}

fn count(len: usize) -> u64 {
    u64::try_from(len).unwrap_or(u64::MAX)
}

/// Checks a buffer description.
pub fn validate_buffer(caps: &Caps, desc: &BufferDesc) -> Result<(), RenderError> {
    if desc.usage.is_empty() {
        return Err(RenderError::Invalid("a buffer needs at least one usage"));
    }
    in_range("buffer size", desc.size, 1, caps.limits.max_buffer_size)
}

/// Checks the size of a staging request, which bounds what one call can make the host allocate.
pub fn validate_staging_size(caps: &Caps, size: u64) -> Result<(), RenderError> {
    in_range("staging size", size, 1, caps.limits.max_staging_size)
}

/// The number of mip levels of a full chain for a texture whose largest side is `largest`:
/// one per halving down to 1.
pub const fn full_mip_chain(largest: u32) -> u32 {
    largest.bit_width()
}

/// Checks a texture description.
pub fn validate_texture(caps: &Caps, desc: &TextureDesc) -> Result<(), RenderError> {
    if desc.usage.is_empty() {
        return Err(RenderError::Invalid("a texture needs at least one usage"));
    }
    let l = &caps.limits;
    let Extent3d {
        width,
        height,
        depth_or_layers,
    } = desc.size;
    let (w, h, d) = (
        u64::from(width),
        u64::from(height),
        u64::from(depth_or_layers),
    );
    let layers = u64::from(l.max_texture_array_layers);
    let largest = match desc.dimension {
        TextureDimension::D1 => {
            in_range("texture width", w, 1, l.max_texture_dimension_1d.into())?;
            in_range("1D texture height", h, 1, 1)?;
            in_range("texture layer count", d, 1, layers)?;
            width
        }
        TextureDimension::D2 => {
            in_range("texture width", w, 1, l.max_texture_dimension_2d.into())?;
            in_range("texture height", h, 1, l.max_texture_dimension_2d.into())?;
            in_range("texture layer count", d, 1, layers)?;
            width.max(height)
        }
        TextureDimension::Cube => {
            in_range("texture width", w, 1, l.max_texture_dimension_2d.into())?;
            if height != width {
                return Err(RenderError::Invalid("a cube texture must be square"));
            }
            in_range("texture layer count", d, 1, layers)?;
            if !depth_or_layers.is_multiple_of(CUBE_FACES) {
                return Err(RenderError::Invalid(
                    "a cube texture's layer count must be a multiple of 6",
                ));
            }
            width
        }
        TextureDimension::D3 => {
            let max = l.max_texture_dimension_3d.into();
            in_range("texture width", w, 1, max)?;
            in_range("texture height", h, 1, max)?;
            in_range("texture depth", d, 1, max)?;
            width.max(height).max(depth_or_layers)
        }
    };
    in_range(
        "mip level count",
        desc.mip_levels.into(),
        1,
        full_mip_chain(largest).into(),
    )?;
    if !caps.sample_counts.supports(desc.sample_count) {
        return Err(RenderError::SampleCount(desc.sample_count));
    }
    if desc.sample_count > 1 {
        if desc.dimension != TextureDimension::D2 {
            return Err(RenderError::Invalid("a multisampled texture must be 2D"));
        }
        if desc.mip_levels != 1 {
            return Err(RenderError::Invalid(
                "a multisampled texture must have one mip level",
            ));
        }
    }
    for (usage, feature, name) in [
        (TextureUsage::SAMPLED, FormatFeatures::SAMPLE, "sampling"),
        (
            TextureUsage::STORAGE,
            FormatFeatures::STORAGE,
            "storage images",
        ),
        (
            TextureUsage::RENDER_TARGET,
            FormatFeatures::RENDER,
            "render targets",
        ),
        (TextureUsage::COPY_SRC, FormatFeatures::COPY, "copies"),
        (TextureUsage::COPY_DST, FormatFeatures::COPY, "copies"),
    ] {
        if desc.usage.contains(usage) {
            format_supports(caps, desc.format, feature, name)?;
        }
    }
    Ok(())
}

/// Whether two formats can view the same texels: the same block size and bits per block.
fn same_block(a: PixelFormat, b: PixelFormat) -> bool {
    a.block_width() == b.block_width()
        && a.block_height() == b.block_height()
        && a.bits_per_block() == b.bits_per_block()
}

/// Checks a view of `texture`.
pub fn validate_view(
    caps: &Caps,
    texture: &TextureDesc,
    view: &ViewDesc,
) -> Result<(), RenderError> {
    if !view.swizzle.is_identity() && !caps.view_swizzle {
        return Err(RenderError::MissingFeature("view swizzles"));
    }
    if !same_block(view.format, texture.format) {
        return Err(RenderError::Invalid(
            "a view's format must have the same block size as its texture's",
        ));
    }
    if view.format.aspect() != texture.format.aspect() {
        return Err(RenderError::Invalid(
            "a view's format must store the same aspect as its texture's",
        ));
    }
    let mips = u64::from(texture.mip_levels);
    in_range("view mip count", view.mip_count.into(), 1, mips)?;
    within(
        "view mip range",
        u64::from(view.base_mip) + u64::from(view.mip_count),
        mips,
    )?;
    let layers = u64::from(texture.layer_count());
    in_range("view layer count", view.layer_count.into(), 1, layers)?;
    within(
        "view layer range",
        u64::from(view.base_layer) + u64::from(view.layer_count),
        layers,
    )?;
    let square = texture.size.width == texture.size.height;
    let fits = match (texture.dimension, view.dimension) {
        (TextureDimension::D1, ViewDimension::D1)
        | (TextureDimension::D2 | TextureDimension::Cube, ViewDimension::D2) => {
            if view.layer_count != 1 {
                return Err(RenderError::Invalid(
                    "a single-layer view must have one layer",
                ));
            }
            true
        }
        (TextureDimension::D2 | TextureDimension::Cube, ViewDimension::Cube) => {
            if view.layer_count != CUBE_FACES {
                return Err(RenderError::Invalid("a cube view must have 6 layers"));
            }
            square
        }
        (TextureDimension::D2 | TextureDimension::Cube, ViewDimension::CubeArray) => {
            if !view.layer_count.is_multiple_of(CUBE_FACES) {
                return Err(RenderError::Invalid(
                    "a cube array view's layer count must be a multiple of 6",
                ));
            }
            square
        }
        (TextureDimension::D1, ViewDimension::D1Array)
        | (TextureDimension::D2 | TextureDimension::Cube, ViewDimension::D2Array)
        | (TextureDimension::D3, ViewDimension::D3) => true,
        _ => {
            return Err(RenderError::Invalid(
                "the view dimension does not fit the texture's",
            ));
        }
    };
    if fits {
        Ok(())
    } else {
        Err(RenderError::Invalid("a cube view needs a square texture"))
    }
}

/// Checks a sampler description.
pub fn validate_sampler(caps: &Caps, desc: &SamplerDesc) -> Result<(), RenderError> {
    in_range(
        "sampler anisotropy",
        desc.max_anisotropy.into(),
        1,
        caps.limits.max_sampler_anisotropy.into(),
    )?;
    if !(desc.lod_bias.is_finite() && desc.min_lod.is_finite() && desc.max_lod.is_finite()) {
        return Err(RenderError::Invalid(
            "sampler level-of-detail values must be finite",
        ));
    }
    if desc.min_lod > desc.max_lod {
        return Err(RenderError::Invalid(
            "a sampler's minimum level of detail is above its maximum",
        ));
    }
    for mode in desc.address {
        match mode {
            AddressMode::ClampToBorder => require(
                caps,
                Features::CLAMP_TO_BORDER,
                "the clamp-to-border address mode",
            )?,
            AddressMode::MirrorClampToEdge => require(
                caps,
                Features::MIRROR_CLAMP_TO_EDGE,
                "the mirror-clamp-to-edge address mode",
            )?,
            AddressMode::Repeat | AddressMode::MirrorRepeat | AddressMode::ClampToEdge => {}
        }
    }
    if let BorderColor::Custom(c) = desc.border_color
        && !c.iter().all(|v| v.is_finite())
    {
        return Err(RenderError::Invalid("a border colour must be finite"));
    }
    Ok(())
}

fn stage_supported(caps: &Caps, stage: ShaderStage) -> Result<(), RenderError> {
    let needed = match stage {
        ShaderStage::Vertex | ShaderStage::Fragment | ShaderStage::Compute => StageSupport::EMPTY,
        ShaderStage::TessControl | ShaderStage::TessEval => StageSupport::TESSELLATION,
        ShaderStage::Geometry => StageSupport::GEOMETRY,
    };
    if caps.stages.contains(needed) {
        Ok(())
    } else {
        Err(RenderError::StageUnsupported(stage))
    }
}

/// Group and binding numbers below the device's limits, each pair used once.
fn check_bind_slots(
    caps: &Caps,
    slots: impl Iterator<Item = (u32, u32)>,
    duplicate: &'static str,
) -> Result<(), RenderError> {
    let mut seen = Vec::new();
    for (group, binding) in slots {
        below(
            "bind group",
            group.into(),
            caps.limits.max_bind_groups.into(),
        )?;
        below(
            "binding",
            binding.into(),
            caps.limits.max_bindings_per_bind_group.into(),
        )?;
        seen.push((group, binding));
    }
    seen.sort_unstable();
    if seen.windows(2).any(|w| w[0] == w[1]) {
        return Err(RenderError::Invalid(duplicate));
    }
    Ok(())
}

fn check_interface(
    vars: &[InterfaceVariable],
    location_limit: Option<(&'static str, u32)>,
    duplicate: &'static str,
) -> Result<(), RenderError> {
    let mut locations = Vec::with_capacity(vars.len());
    for v in vars {
        in_range(
            "interface components",
            v.components.into(),
            1,
            MAX_INTERFACE_COMPONENTS.into(),
        )?;
        if let Some((what, limit)) = location_limit {
            below(what, v.location.into(), limit.into())?;
        }
        locations.push(v.location);
    }
    locations.sort_unstable();
    if locations.windows(2).any(|w| w[0] == w[1]) {
        return Err(RenderError::Invalid(duplicate));
    }
    Ok(())
}

/// Checks a shader description: its stage against the device, and its interface tables.
pub fn validate_shader(caps: &Caps, desc: &ShaderDesc) -> Result<(), RenderError> {
    stage_supported(caps, desc.stage)?;
    if desc.entry.is_empty() {
        return Err(RenderError::Invalid("a shader needs an entry point name"));
    }
    let resources = &desc.interface.resources;
    check_bind_slots(
        caps,
        resources.iter().map(|r| (r.group, r.binding)),
        "a shader binds one group and binding twice",
    )?;
    for ResourceBinding { kind, .. } in resources {
        if let ResourceKind::StorageTexture { format, .. } = *kind {
            format_supports(caps, format, FormatFeatures::STORAGE, "storage images")?;
        }
    }
    let input_limit = (desc.stage == ShaderStage::Vertex)
        .then(|| ("vertex input location", vertex_attribute_limit(caps)));
    check_interface(
        &desc.interface.inputs,
        input_limit,
        "a shader uses one input location twice",
    )?;
    let output_limit = (desc.stage == ShaderStage::Fragment)
        .then(|| ("fragment output location", color_target_limit(caps)));
    check_interface(
        &desc.interface.outputs,
        output_limit,
        "a shader uses one output location twice",
    )
}

/// The number of colour targets a fixed array uses: one past the highest one present.
fn used_targets<T>(targets: &[Option<T>]) -> u64 {
    targets
        .iter()
        .rposition(Option::is_some)
        .map_or(0, |i| count(i).saturating_add(1))
}

/// Checks a graphics pipeline description.
pub fn validate_graphics_pipeline(
    caps: &Caps,
    desc: &GraphicsPipelineDesc,
) -> Result<(), RenderError> {
    check_stages(caps, desc)?;
    check_vertex_input(caps, &desc.vertex_input)?;
    check_raster(caps, &desc.raster)?;
    check_targets(caps, desc)?;
    if desc.logic_op.is_some() {
        require(caps, Features::LOGIC_OP, "logic operations")?;
    }
    Ok(())
}

/// Tessellation in pairs and with patches; optional stages supported.
fn check_stages(caps: &Caps, desc: &GraphicsPipelineDesc) -> Result<(), RenderError> {
    let stages = &desc.stages;
    let tessellated = match (stages.tess_control, stages.tess_eval) {
        (Some(_), Some(_)) => true,
        (None, None) => false,
        _ => {
            return Err(RenderError::Invalid(
                "tessellation needs both a control and an evaluation shader",
            ));
        }
    };
    if tessellated {
        stage_supported(caps, ShaderStage::TessControl)?;
    }
    if stages.geometry.is_some() {
        stage_supported(caps, ShaderStage::Geometry)?;
    }
    match (desc.topology == Topology::Patches, tessellated) {
        (true, false) => Err(RenderError::Invalid("patches need tessellation shaders")),
        (false, true) => Err(RenderError::Invalid(
            "tessellation shaders need the patch topology",
        )),
        _ => Ok(()),
    }
}

/// Buffer and attribute counts, strides, and attribute buffers and locations.
fn check_vertex_input(caps: &Caps, input: &VertexLayout) -> Result<(), RenderError> {
    let attribute_limit = vertex_attribute_limit(caps);
    in_range(
        "vertex buffer count",
        count(input.buffers.len()),
        0,
        vertex_buffer_limit(caps).into(),
    )?;
    in_range(
        "vertex attribute count",
        count(input.attributes.len()),
        0,
        attribute_limit.into(),
    )?;
    for b in &input.buffers {
        in_range(
            "vertex buffer stride",
            b.stride.into(),
            0,
            caps.limits.max_vertex_buffer_stride.into(),
        )?;
    }
    let mut locations = Vec::with_capacity(input.attributes.len());
    for a in &input.attributes {
        below(
            "vertex attribute buffer",
            a.buffer.into(),
            count(input.buffers.len()),
        )?;
        below(
            "vertex attribute location",
            a.location.into(),
            attribute_limit.into(),
        )?;
        locations.push(a.location);
    }
    locations.sort_unstable();
    if locations.windows(2).any(|w| w[0] == w[1]) {
        return Err(RenderError::Invalid(
            "a pipeline uses one vertex attribute location twice",
        ));
    }
    Ok(())
}

/// Optional raster features, and finite values.
fn check_raster(caps: &Caps, raster: &RasterState) -> Result<(), RenderError> {
    match raster.polygon_mode {
        PolygonMode::Fill => {}
        PolygonMode::Line => require(caps, Features::POLYGON_MODE_LINE, "line polygon mode")?,
        PolygonMode::Point => require(caps, Features::POLYGON_MODE_POINT, "point polygon mode")?,
    }
    if raster.depth_clamp {
        require(caps, Features::DEPTH_CLAMP, "depth clamping")?;
    }
    let bias = raster.depth_bias;
    if [
        bias.constant,
        bias.slope_scale,
        bias.clamp,
        raster.line_width,
    ]
    .iter()
    .all(|v| v.is_finite())
    {
        Ok(())
    } else {
        Err(RenderError::Invalid("rasteriser values must be finite"))
    }
}

/// Sample count, colour target count, target formats and blending.
fn check_targets(caps: &Caps, desc: &GraphicsPipelineDesc) -> Result<(), RenderError> {
    let targets = &desc.targets;
    if !caps.sample_counts.supports(targets.sample_count) {
        return Err(RenderError::SampleCount(targets.sample_count));
    }
    in_range(
        "colour target count",
        used_targets(&targets.colors),
        0,
        color_target_limit(caps).into(),
    )?;
    for (format, blend) in targets.colors.iter().zip(&desc.blend) {
        let Some(format) = *format else { continue };
        if format.aspect() != FormatAspect::Color {
            return Err(RenderError::Invalid(
                "a colour target needs a colour format",
            ));
        }
        format_supports(caps, format, FormatFeatures::RENDER, "render targets")?;
        if blend.enabled {
            format_supports(caps, format, FormatFeatures::BLEND, "blending")?;
            let factors = [
                blend.color.src,
                blend.color.dst,
                blend.alpha.src,
                blend.alpha.dst,
            ];
            if factors.iter().any(|f| f.is_dual_source()) {
                require(caps, Features::DUAL_SOURCE_BLEND, "dual-source blending")?;
            }
        }
    }
    if let Some(format) = targets.depth_stencil {
        if format.aspect() == FormatAspect::Color {
            return Err(RenderError::Invalid(
                "a depth/stencil target needs a depth or stencil format",
            ));
        }
        format_supports(caps, format, FormatFeatures::RENDER, "render targets")?;
    }
    Ok(())
}

/// Checks the render targets a `SetRenderTargets` command binds.
pub fn validate_render_targets(caps: &Caps, targets: &RenderTargets) -> Result<(), RenderError> {
    in_range(
        "colour target count",
        used_targets(&targets.colors),
        0,
        color_target_limit(caps).into(),
    )
}

fn check_bindings(
    caps: &Caps,
    bindings: &[Binding],
    duplicate: &'static str,
) -> Result<(), RenderError> {
    check_bind_slots(
        caps,
        bindings.iter().map(|b| (b.group, b.binding)),
        duplicate,
    )
}

/// Checks a draw's counts, dynamic state and bindings against the device.
pub fn validate_draw(caps: &Caps, draw: &DrawDesc) -> Result<(), RenderError> {
    let d = &draw.dynamic;
    in_range(
        "viewport count",
        count(d.viewports.len()),
        1,
        viewport_limit(caps).into(),
    )?;
    if !d.scissors.is_empty() && d.scissors.len() != d.viewports.len() {
        return Err(RenderError::Invalid(
            "a draw needs one scissor per viewport, or none",
        ));
    }
    let viewport_values = d
        .viewports
        .iter()
        .flat_map(|v| [v.x, v.y, v.width, v.height, v.min_depth, v.max_depth]);
    let other_values = d.blend_constant.into_iter().chain([
        d.depth_bias.constant,
        d.depth_bias.slope_scale,
        d.depth_bias.clamp,
        d.line_width,
    ]);
    if !viewport_values.chain(other_values).all(f32::is_finite) {
        return Err(RenderError::Invalid("draw state values must be finite"));
    }
    in_range(
        "vertex buffer count",
        count(draw.vertex_buffers.len()),
        0,
        vertex_buffer_limit(caps).into(),
    )?;
    check_bindings(
        caps,
        &draw.bindings,
        "a draw binds one group and binding twice",
    )?;
    let indexed = match draw.params {
        DrawParams::Direct { .. } => false,
        DrawParams::Indexed { .. } => true,
        DrawParams::Indirect {
            draw_count,
            indexed,
            ..
        } => {
            if draw_count == 0 {
                return Err(RenderError::OutOfRange {
                    what: "indirect draw count",
                    value: 0,
                    min: 1,
                    max: u32::MAX.into(),
                });
            }
            if draw_count > 1 {
                require(caps, Features::MULTI_DRAW_INDIRECT, "multi-draw indirect")?;
            }
            indexed
        }
    };
    if indexed && draw.index_buffer.is_none() {
        return Err(RenderError::Invalid(
            "an indexed draw needs an index buffer",
        ));
    }
    Ok(())
}

/// Checks a dispatch's group counts and bindings against the device.
pub fn validate_dispatch(caps: &Caps, dispatch: &DispatchDesc) -> Result<(), RenderError> {
    check_bindings(
        caps,
        &dispatch.bindings,
        "a dispatch binds one group and binding twice",
    )?;
    if let DispatchParams::Direct { x, y, z } = dispatch.params {
        let max = caps.limits.max_compute_workgroups_per_dimension.into();
        for v in [x, y, z] {
            in_range("workgroup count", v.into(), 0, max)?;
        }
    }
    Ok(())
}

/// Bytes of `extent` texels of `format`, tightly packed: whole blocks per row, rows per slice,
/// slices (or layers) one after another.
pub fn packed_size(format: PixelFormat, extent: Extent3d) -> Result<u64, RenderError> {
    let overflow = RenderError::Overflow {
        what: "packed texture size",
    };
    let blocks_wide = u64::from(extent.width.div_ceil(format.block_width()));
    let blocks_high = u64::from(extent.height.div_ceil(format.block_height()));
    blocks_wide
        .checked_mul(blocks_high)
        .and_then(|v| v.checked_mul(extent.depth_or_layers.into()))
        .and_then(|v| v.checked_mul(format.bytes_per_block().into()))
        .ok_or(overflow)
}

/// Checks that `slice` is non-empty and lies inside the buffer `desc` describes.
pub fn validate_buffer_slice(desc: &BufferDesc, slice: &BufferSlice) -> Result<(), RenderError> {
    in_range("buffer range size", slice.size, 1, desc.size)?;
    let end = slice
        .offset
        .checked_add(slice.size)
        .ok_or(RenderError::Overflow {
            what: "buffer range end",
        })?;
    within("buffer range", end, desc.size)
}

/// Checks that `region` is non-empty, lies inside its mip level of the texture `desc`
/// describes, and, for a compressed format, starts on a block boundary and covers whole blocks
/// or reaches the mip level's edge.
pub fn validate_texture_region(
    desc: &TextureDesc,
    region: &TextureRegion,
) -> Result<(), RenderError> {
    let Some(mip) = desc.mip_extent(region.mip) else {
        return Err(RenderError::NotBelow {
            what: "mip level",
            index: region.mip.into(),
            count: desc.mip_levels.into(),
        });
    };
    let e = region.extent;
    if e.width == 0 || e.height == 0 || e.depth_or_layers == 0 {
        return Err(RenderError::Invalid("a texture region must not be empty"));
    }
    let o = region.origin;
    let end = |start: u32, len: u32| u64::from(start) + u64::from(len);
    within("texture region width", end(o.x, e.width), mip.width.into())?;
    within(
        "texture region height",
        end(o.y, e.height),
        mip.height.into(),
    )?;
    within(
        "texture region depth or layers",
        end(o.z, e.depth_or_layers),
        mip.depth_or_layers.into(),
    )?;
    let (bw, bh) = (desc.format.block_width(), desc.format.block_height());
    if !o.x.is_multiple_of(bw) || !o.y.is_multiple_of(bh) {
        return Err(RenderError::Invalid(
            "a compressed texture region must start on a block boundary",
        ));
    }
    let whole_or_edge = |len: u32, block: u32, end: u64, edge: u32| {
        len.is_multiple_of(block) || end == u64::from(edge)
    };
    if !whole_or_edge(e.width, bw, end(o.x, e.width), mip.width)
        || !whole_or_edge(e.height, bh, end(o.y, e.height), mip.height)
    {
        return Err(RenderError::Invalid(
            "a compressed texture region must cover whole blocks or reach the mip edge",
        ));
    }
    Ok(())
}

/// One side of a copy, with the description its id names.
#[derive(Clone, Copy, Debug)]
pub enum Resolved<'a> {
    /// A buffer range.
    Buffer {
        /// The buffer's description.
        desc: &'a BufferDesc,
        /// The range.
        slice: &'a BufferSlice,
    },
    /// A texture region.
    Texture {
        /// The texture's description.
        desc: &'a TextureDesc,
        /// The region.
        region: &'a TextureRegion,
    },
}

/// Which way a resolved side is used.
#[derive(Clone, Copy)]
enum Side {
    Source,
    Destination,
}

/// Checks one side's bounds and usage, and returns its byte size.
fn check_side(side: Side, resolved: Resolved<'_>) -> Result<u64, RenderError> {
    match resolved {
        Resolved::Buffer { desc, slice } => {
            validate_buffer_slice(desc, slice)?;
            let (needed, resource, usage) = match side {
                Side::Source => (BufferUsage::COPY_SRC, "copy source buffer", "copy source"),
                Side::Destination => (
                    BufferUsage::COPY_DST,
                    "copy destination buffer",
                    "copy destination",
                ),
            };
            if !desc.usage.contains(needed) {
                return Err(RenderError::MissingUsage { resource, usage });
            }
            Ok(slice.size)
        }
        Resolved::Texture { desc, region } => {
            validate_texture_region(desc, region)?;
            let (needed, resource, usage) = match side {
                Side::Source => (TextureUsage::COPY_SRC, "copy source texture", "copy source"),
                Side::Destination => (
                    TextureUsage::COPY_DST,
                    "copy destination texture",
                    "copy destination",
                ),
            };
            if !desc.usage.contains(needed) {
                return Err(RenderError::MissingUsage { resource, usage });
            }
            packed_size(desc.format, region.extent)
        }
    }
}

/// Checks a copy: each side in bounds and created for copying, and the two sides compatible.
pub fn validate_copy(src: Resolved<'_>, dst: Resolved<'_>) -> Result<(), RenderError> {
    let src_bytes = check_side(Side::Source, src)?;
    let dst_bytes = check_side(Side::Destination, dst)?;
    match (src, dst) {
        (
            Resolved::Texture {
                desc: s,
                region: sr,
            },
            Resolved::Texture {
                desc: d,
                region: dr,
            },
        ) => {
            if !same_block(s.format, d.format) {
                return Err(RenderError::Invalid(
                    "a texture copy needs formats with the same block size",
                ));
            }
            if sr.extent != dr.extent {
                return Err(RenderError::Invalid("a texture copy needs equal extents"));
            }
            Ok(())
        }
        (Resolved::Buffer { .. }, Resolved::Texture { .. }) => {
            if src_bytes == dst_bytes {
                Ok(())
            } else {
                Err(RenderError::SizeMismatch {
                    what: "copy source buffer range",
                    expected: dst_bytes,
                    actual: src_bytes,
                })
            }
        }
        (Resolved::Texture { .. } | Resolved::Buffer { .. }, Resolved::Buffer { .. }) => {
            if src_bytes == dst_bytes {
                Ok(())
            } else {
                Err(RenderError::SizeMismatch {
                    what: "copy destination buffer range",
                    expected: src_bytes,
                    actual: dst_bytes,
                })
            }
        }
    }
}

/// Checks an upload of `staging_len` bytes into `dst`: the destination in bounds and created as
/// a copy destination, and the slice exactly the bytes it needs.
pub fn validate_upload(staging_len: u64, dst: Resolved<'_>) -> Result<(), RenderError> {
    let needed = check_side(Side::Destination, dst)?;
    if staging_len == needed {
        Ok(())
    } else {
        Err(RenderError::SizeMismatch {
            what: "staging slice",
            expected: needed,
            actual: staging_len,
        })
    }
}

/// Checks a readback source: in bounds and created as a copy source.
pub fn validate_readback(src: Resolved<'_>) -> Result<(), RenderError> {
    check_side(Side::Source, src).map(|_| ())
}

/// Checks that `rect` lies inside a `width` by `height` image (a blit, clear or present
/// rectangle against its view's mip level).
pub fn validate_rect(width: u32, height: u32, rect: &Rect) -> Result<(), RenderError> {
    within(
        "rectangle width",
        u64::from(rect.x) + u64::from(rect.width),
        width.into(),
    )?;
    within(
        "rectangle height",
        u64::from(rect.y) + u64::from(rect.height),
        height.into(),
    )
}

#[cfg(test)]
mod tests;
