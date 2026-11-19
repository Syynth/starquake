//! Descriptions in wgpu's terms: what the render thread creates for each checked description.
//!
//! Pure conversions, made on the calling thread when a handle is allocated and carried to the
//! render thread, which creates the host object from them in order (docs/specs/render-backend.md,
//! "The trait": "The host object is created on the render thread, in order, before the first
//! command list that uses it").
//!
//! Where the backend asks wgpu for more than the description says:
//!
//! - a buffer is made [`crate::rules::host_buffer_size`] bytes, the size rounded up to the copy
//!   alignment, so a readback can read the aligned range around any range of it;
//! - a texture also gets the copy-source and copy-destination usages its format allows: the
//!   backend reads a presented view back for a frame dump and clears part of a view by copying
//!   into it, whatever usages the guest asked for; and it lists its format's sRGB counterpart
//!   as a view format, so that views may reinterpret it as the shared checks allow
//!   (`crate::rules::check_view`).
//!
//! Where it gives wgpu less, because wgpu 30's sampler has no such field or rule (to be settled
//! with draws, slice R7, when sampling is first observable):
//!
//! - the LOD bias is not passed on (`wgpu::SamplerDescriptor` has none);
//! - LOD clamps below zero are raised to zero (wgpu-core refuses a negative `lod_min_clamp`, and
//!   no level below zero exists);
//! - anisotropy is passed on only when all three filters are linear, and is 1 otherwise
//!   (wgpu-core refuses anisotropy with a nearest filter, `Device::create_sampler`).

use crate::format::texture_format;
use crate::rules::{HostTexture, border_color};
use starquake_render::resource::{
    AddressMode, CompareFunction, FilterMode, ViewDimension,
};
use starquake_render::{BufferUsage, SamplerDesc, TextureDesc, TextureUsage, ViewDesc};

/// wgpu's usages for a buffer.
pub fn buffer_usages(usage: BufferUsage) -> wgpu::BufferUsages {
    let mut out = wgpu::BufferUsages::empty();
    for (ours, theirs) in [
        (BufferUsage::VERTEX, wgpu::BufferUsages::VERTEX),
        (BufferUsage::INDEX, wgpu::BufferUsages::INDEX),
        (BufferUsage::UNIFORM, wgpu::BufferUsages::UNIFORM),
        (BufferUsage::STORAGE, wgpu::BufferUsages::STORAGE),
        (BufferUsage::INDIRECT, wgpu::BufferUsages::INDIRECT),
        (BufferUsage::COPY_SRC, wgpu::BufferUsages::COPY_SRC),
        (BufferUsage::COPY_DST, wgpu::BufferUsages::COPY_DST),
    ] {
        if usage.contains(ours) {
            out |= theirs;
        }
    }
    out
}

/// wgpu's usages for a texture: the description's, plus the copy usages `allowed` (what the
/// device allows for the format) has (module docs).
pub fn texture_usages(usage: TextureUsage, allowed: wgpu::TextureUsages) -> wgpu::TextureUsages {
    let mut out = wgpu::TextureUsages::empty();
    for (ours, theirs) in [
        (TextureUsage::SAMPLED, wgpu::TextureUsages::TEXTURE_BINDING),
        (TextureUsage::STORAGE, wgpu::TextureUsages::STORAGE_BINDING),
        (TextureUsage::RENDER_TARGET, wgpu::TextureUsages::RENDER_ATTACHMENT),
        (TextureUsage::COPY_SRC, wgpu::TextureUsages::COPY_SRC),
        (TextureUsage::COPY_DST, wgpu::TextureUsages::COPY_DST),
    ] {
        if usage.contains(ours) {
            out |= theirs;
        }
    }
    out | (allowed & (wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST))
}

/// A texture as wgpu will be asked to make it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TextureSpec {
    /// The format.
    pub format: wgpu::TextureFormat,
    /// The shape.
    pub host: HostTexture,
    /// Mip levels.
    pub mip_levels: u32,
    /// Samples.
    pub sample_count: u32,
    /// Usages.
    pub usage: wgpu::TextureUsages,
    /// Other formats views may use.
    pub view_formats: Vec<wgpu::TextureFormat>,
}

impl TextureSpec {
    /// The spec for a checked texture, with `allowed` the usages the device allows for its
    /// format and `reinterpret` whether the device can view in another format. `None` when the
    /// format has no wgpu form, which a checked texture always has.
    pub fn new(
        desc: &TextureDesc,
        host: HostTexture,
        allowed: wgpu::TextureUsages,
        reinterpret: bool,
    ) -> Option<Self> {
        let format = texture_format(desc.format)?;
        let pair = if format.is_srgb() {
            format.remove_srgb_suffix()
        } else {
            format.add_srgb_suffix()
        };
        let view_formats = if reinterpret && pair != format {
            vec![pair]
        } else {
            Vec::new()
        };
        Some(Self {
            format,
            host,
            mip_levels: desc.mip_levels,
            sample_count: desc.sample_count,
            usage: texture_usages(desc.usage, allowed),
            view_formats,
        })
    }
}

/// wgpu's view description for a checked view; `None` when the view's format has no wgpu form
/// or the dimension is a 1D array, which `crate::rules::check_view` refuses first.
pub fn view_descriptor(view: &ViewDesc) -> Option<wgpu::TextureViewDescriptor<'static>> {
    let dimension = match view.dimension {
        ViewDimension::D1 => wgpu::TextureViewDimension::D1,
        ViewDimension::D1Array => return None,
        ViewDimension::D2 => wgpu::TextureViewDimension::D2,
        ViewDimension::D2Array => wgpu::TextureViewDimension::D2Array,
        ViewDimension::Cube => wgpu::TextureViewDimension::Cube,
        ViewDimension::CubeArray => wgpu::TextureViewDimension::CubeArray,
        ViewDimension::D3 => wgpu::TextureViewDimension::D3,
    };
    Some(wgpu::TextureViewDescriptor {
        label: None,
        format: Some(texture_format(view.format)?),
        dimension: Some(dimension),
        usage: None,
        aspect: wgpu::TextureAspect::All,
        base_mip_level: view.base_mip,
        mip_level_count: Some(view.mip_count),
        base_array_layer: view.base_layer,
        array_layer_count: Some(view.layer_count),
    })
}

const fn filter(mode: FilterMode) -> wgpu::FilterMode {
    match mode {
        FilterMode::Nearest => wgpu::FilterMode::Nearest,
        FilterMode::Linear => wgpu::FilterMode::Linear,
    }
}

const fn address(mode: AddressMode) -> wgpu::AddressMode {
    match mode {
        AddressMode::Repeat => wgpu::AddressMode::Repeat,
        AddressMode::MirrorRepeat => wgpu::AddressMode::MirrorRepeat,
        // The device never reports mirror-clamp-to-edge (`crate::caps`), so the shared checks
        // refuse it before this; clamping to the edge is the nearest wgpu has.
        AddressMode::ClampToEdge | AddressMode::MirrorClampToEdge => {
            wgpu::AddressMode::ClampToEdge
        }
        AddressMode::ClampToBorder => wgpu::AddressMode::ClampToBorder,
    }
}

const fn compare(function: CompareFunction) -> wgpu::CompareFunction {
    match function {
        CompareFunction::Never => wgpu::CompareFunction::Never,
        CompareFunction::Less => wgpu::CompareFunction::Less,
        CompareFunction::Equal => wgpu::CompareFunction::Equal,
        CompareFunction::LessEqual => wgpu::CompareFunction::LessEqual,
        CompareFunction::Greater => wgpu::CompareFunction::Greater,
        CompareFunction::NotEqual => wgpu::CompareFunction::NotEqual,
        CompareFunction::GreaterEqual => wgpu::CompareFunction::GreaterEqual,
        CompareFunction::Always => wgpu::CompareFunction::Always,
    }
}

/// wgpu's sampler description for a checked sampler (module docs for what it leaves out).
pub fn sampler_descriptor(desc: &SamplerDesc) -> wgpu::SamplerDescriptor<'static> {
    let linear = matches!(
        (desc.mag_filter, desc.min_filter, desc.mip_filter),
        (FilterMode::Linear, FilterMode::Linear, FilterMode::Linear)
    );
    let anisotropy = if linear {
        u16::try_from(desc.max_anisotropy).unwrap_or(u16::MAX)
    } else {
        1
    };
    let clamps = desc.address.contains(&AddressMode::ClampToBorder);
    wgpu::SamplerDescriptor {
        label: None,
        address_mode_u: address(desc.address[0]),
        address_mode_v: address(desc.address[1]),
        address_mode_w: address(desc.address[2]),
        mag_filter: filter(desc.mag_filter),
        min_filter: filter(desc.min_filter),
        mipmap_filter: match desc.mip_filter {
            FilterMode::Nearest => wgpu::MipmapFilterMode::Nearest,
            FilterMode::Linear => wgpu::MipmapFilterMode::Linear,
        },
        lod_min_clamp: desc.min_lod.max(0.0),
        lod_max_clamp: desc.max_lod.max(0.0),
        compare: desc.compare.map(compare),
        anisotropy_clamp: anisotropy,
        border_color: if clamps {
            border_color(desc.border_color)
        } else {
            None
        },
    }
}

#[cfg(test)]
mod tests;
