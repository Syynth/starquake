//! Descriptions of buffers, textures, views and samplers.
//!
//! Plain values, derived by the GPU crate from guest state and checked by [`crate::validate`]
//! against [`crate::Caps`] before a host object exists (docs/specs/render-backend.md, "Textures,
//! views, samplers, render targets" and "Errors and untrusted input").

use crate::flags::flag_set;
use crate::format::PixelFormat;
use serde::{Deserialize, Serialize};

flag_set! {
    /// What a buffer is used for.
    BufferUsage {
        /// Vertex data.
        VERTEX = 0,
        /// Index data.
        INDEX = 1,
        /// Uniform (constant) data.
        UNIFORM = 2,
        /// Storage data a shader reads or writes.
        STORAGE = 3,
        /// Indirect draw or dispatch parameters.
        INDIRECT = 4,
        /// The source of a copy or readback.
        COPY_SRC = 5,
        /// The destination of a copy or upload.
        COPY_DST = 6,
    }
}

flag_set! {
    /// What a texture is used for.
    TextureUsage {
        /// Sampled in a shader.
        SAMPLED = 0,
        /// A storage image.
        STORAGE = 1,
        /// A colour or depth/stencil render target.
        RENDER_TARGET = 2,
        /// The source of a copy, blit, readback or present.
        COPY_SRC = 3,
        /// The destination of a copy, blit or upload.
        COPY_DST = 4,
    }
}

/// A host buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BufferDesc {
    /// Size in bytes.
    pub size: u64,
    /// What it is used for.
    pub usage: BufferUsage,
}

/// The shape of a texture.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TextureDimension {
    /// One-dimensional, optionally with layers.
    D1,
    /// Two-dimensional, optionally with layers.
    D2,
    /// Three-dimensional.
    D3,
    /// Cube maps: six square layers per cube, optionally several cubes.
    Cube,
}

/// A size in texels: width, height, and depth (3D) or layer count (everything else).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Extent3d {
    /// Width in texels.
    pub width: u32,
    /// Height in texels (1 for 1D).
    pub height: u32,
    /// Depth of a 3D texture, or the layer count of any other.
    pub depth_or_layers: u32,
}

/// A position in texels: x, y, and the z slice (3D) or first layer (everything else).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Origin3d {
    /// Column.
    pub x: u32,
    /// Row.
    pub y: u32,
    /// Slice of a 3D texture, or layer of any other.
    pub z: u32,
}

/// A host texture. Render targets are textures with [`TextureUsage::RENDER_TARGET`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TextureDesc {
    /// The shape.
    pub dimension: TextureDimension,
    /// The pixel format.
    pub format: PixelFormat,
    /// Size of mip 0, and the depth or layer count.
    pub size: Extent3d,
    /// Mip levels, at least 1.
    pub mip_levels: u32,
    /// Samples per texel, 1 for a texture that is not multisampled.
    pub sample_count: u32,
    /// What it is used for.
    pub usage: TextureUsage,
}

impl TextureDesc {
    /// The size of mip level `mip`: width and height halve per level down to 1, as does a 3D
    /// texture's depth; a layer count stays. `None` when `mip` is not below
    /// [`TextureDesc::mip_levels`].
    pub const fn mip_extent(&self, mip: u32) -> Option<Extent3d> {
        if mip >= self.mip_levels {
            return None;
        }
        let depth_or_layers = match self.dimension {
            TextureDimension::D3 => mip_size(self.size.depth_or_layers, mip),
            TextureDimension::D1 | TextureDimension::D2 | TextureDimension::Cube => {
                self.size.depth_or_layers
            }
        };
        Some(Extent3d {
            width: mip_size(self.size.width, mip),
            height: mip_size(self.size.height, mip),
            depth_or_layers,
        })
    }

    /// Layers a view of this texture can choose from: the layer count, or 1 for a 3D texture.
    pub const fn layer_count(&self) -> u32 {
        match self.dimension {
            TextureDimension::D3 => 1,
            TextureDimension::D1 | TextureDimension::D2 | TextureDimension::Cube => {
                self.size.depth_or_layers
            }
        }
    }
}

/// `size` halved `mip` times, but never below 1.
const fn mip_size(size: u32, mip: u32) -> u32 {
    match size.checked_shr(mip) {
        Some(s) if s > 0 => s,
        _ => 1,
    }
}

/// How a view presents its texture to a shader.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ViewDimension {
    /// One 1D layer.
    D1,
    /// Several 1D layers.
    D1Array,
    /// One 2D layer.
    D2,
    /// Several 2D layers.
    D2Array,
    /// Six square layers as a cube.
    Cube,
    /// Several cubes, six layers each.
    CubeArray,
    /// A 3D texture.
    D3,
}

/// Where one component of a view's result comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SwizzleSource {
    /// Constant 0.
    Zero,
    /// Constant 1.
    One,
    /// The red channel.
    R,
    /// The green channel.
    G,
    /// The blue channel.
    B,
    /// The alpha channel.
    A,
}

/// A view's component remapping (the guest texture header's per-texture swizzle).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Swizzle {
    /// Source of the result's red.
    pub r: SwizzleSource,
    /// Source of the result's green.
    pub g: SwizzleSource,
    /// Source of the result's blue.
    pub b: SwizzleSource,
    /// Source of the result's alpha.
    pub a: SwizzleSource,
}

impl Swizzle {
    /// Each component from its own channel.
    pub const IDENTITY: Self = Self {
        r: SwizzleSource::R,
        g: SwizzleSource::G,
        b: SwizzleSource::B,
        a: SwizzleSource::A,
    };

    /// Whether this is [`Swizzle::IDENTITY`].
    pub const fn is_identity(self) -> bool {
        matches!(
            self,
            Self {
                r: SwizzleSource::R,
                g: SwizzleSource::G,
                b: SwizzleSource::B,
                a: SwizzleSource::A,
            }
        )
    }
}

/// A view of a texture.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ViewDesc {
    /// The format to read the texture as: same block size and aspect as the texture's.
    pub format: PixelFormat,
    /// How the view presents the texture.
    pub dimension: ViewDimension,
    /// First mip level.
    pub base_mip: u32,
    /// Mip levels, at least 1.
    pub mip_count: u32,
    /// First layer.
    pub base_layer: u32,
    /// Layers, at least 1.
    pub layer_count: u32,
    /// Component remapping; anything but the identity needs [`crate::Caps::view_swizzle`].
    pub swizzle: Swizzle,
}

/// Texel filtering.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FilterMode {
    /// The nearest texel.
    Nearest,
    /// Linear interpolation.
    Linear,
}

/// What a sampler does with coordinates outside 0..1.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AddressMode {
    /// Wrap around.
    Repeat,
    /// Wrap around, mirroring every other repeat.
    MirrorRepeat,
    /// Clamp to the edge texel.
    ClampToEdge,
    /// Return the border colour; needs [`crate::caps::Features::CLAMP_TO_BORDER`].
    ClampToBorder,
    /// Mirror once, then clamp to the edge; needs
    /// [`crate::caps::Features::MIRROR_CLAMP_TO_EDGE`].
    MirrorClampToEdge,
}

/// The colour [`AddressMode::ClampToBorder`] returns.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum BorderColor {
    /// (0, 0, 0, 0).
    TransparentBlack,
    /// (0, 0, 0, 1).
    OpaqueBlack,
    /// (1, 1, 1, 1).
    OpaqueWhite,
    /// Any colour, as the guest's sampler states it (finite values only).
    Custom([f32; 4]),
}

/// A depth or stencil comparison.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CompareFunction {
    /// Never passes.
    Never,
    /// Passes when the new value is less.
    Less,
    /// Passes when equal.
    Equal,
    /// Passes when less or equal.
    LessEqual,
    /// Passes when greater.
    Greater,
    /// Passes when not equal.
    NotEqual,
    /// Passes when greater or equal.
    GreaterEqual,
    /// Always passes.
    Always,
}

/// How filtered texels combine.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ReductionMode {
    /// The weighted average.
    Average,
    /// The minimum.
    Min,
    /// The maximum.
    Max,
}

/// A sampler.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SamplerDesc {
    /// Magnification filter.
    pub mag_filter: FilterMode,
    /// Minification filter.
    pub min_filter: FilterMode,
    /// Filter between mip levels.
    pub mip_filter: FilterMode,
    /// Address modes for u, v and w.
    pub address: [AddressMode; 3],
    /// Border colour for [`AddressMode::ClampToBorder`].
    pub border_color: BorderColor,
    /// Added to the computed level of detail.
    pub lod_bias: f32,
    /// Lowest level of detail used.
    pub min_lod: f32,
    /// Highest level of detail used.
    pub max_lod: f32,
    /// Anisotropy, 1 for none.
    pub max_anisotropy: u32,
    /// Depth comparison, for shadow samplers.
    pub compare: Option<CompareFunction>,
    /// How filtered texels combine.
    pub reduction: ReductionMode,
}

#[cfg(test)]
mod tests {
    use super::{Extent3d, Swizzle, SwizzleSource, TextureDesc, TextureDimension, TextureUsage};
    use crate::format::PixelFormat;

    fn desc(dimension: TextureDimension, size: (u32, u32, u32), mips: u32) -> TextureDesc {
        TextureDesc {
            dimension,
            format: PixelFormat::A8B8G8R8Unorm,
            size: Extent3d {
                width: size.0,
                height: size.1,
                depth_or_layers: size.2,
            },
            mip_levels: mips,
            sample_count: 1,
            usage: TextureUsage::SAMPLED,
        }
    }

    #[test]
    fn mip_extent_halves_down_to_one() {
        let t = desc(TextureDimension::D3, (16, 4, 8), 5);
        let at = |mip| {
            t.mip_extent(mip)
                .map(|e| (e.width, e.height, e.depth_or_layers))
        };
        assert_eq!(at(0), Some((16, 4, 8)));
        assert_eq!(at(1), Some((8, 2, 4)));
        assert_eq!(at(2), Some((4, 1, 2)));
        assert_eq!(at(4), Some((1, 1, 1)));
        assert_eq!(at(5), None, "mip_levels is the first missing level");
        assert_eq!(t.layer_count(), 1);
    }

    #[test]
    fn mip_extent_keeps_layers() {
        let t = desc(TextureDimension::D2, (16, 16, 6), u32::MAX);
        let e = t.mip_extent(3).expect("present");
        assert_eq!((e.width, e.height, e.depth_or_layers), (2, 2, 6));
        // A level past 31 shifts everything out; the size stays 1, never 0, and nothing panics.
        let e = t.mip_extent(40).expect("present");
        assert_eq!((e.width, e.height, e.depth_or_layers), (1, 1, 6));
        assert_eq!(t.layer_count(), 6);
    }

    #[test]
    fn swizzle_identity() {
        assert!(Swizzle::IDENTITY.is_identity());
        for (i, s) in [
            Swizzle {
                r: SwizzleSource::G,
                ..Swizzle::IDENTITY
            },
            Swizzle {
                g: SwizzleSource::Zero,
                ..Swizzle::IDENTITY
            },
            Swizzle {
                b: SwizzleSource::One,
                ..Swizzle::IDENTITY
            },
            Swizzle {
                a: SwizzleSource::R,
                ..Swizzle::IDENTITY
            },
        ]
        .into_iter()
        .enumerate()
        {
            assert!(!s.is_identity(), "component {i}");
        }
    }
}
