//! The backend's own helper shaders: WGSL sources and what a backend needs to use them
//! (docs/specs/render-backend.md, "Shader emission" and slice R4).
//!
//! These are not guest shaders. A backend draws with them to do what its host API has no
//! command for: the present blit ([`HostShader::Present`]) and clears a render pass's load
//! action cannot express ([`HostShader::Clear`]). They are written in WGSL (the spec's
//! decision 12): wgpu takes the source as it is, and a later direct backend translates the same
//! source with naga at startup. This crate only holds the sources; it runs no naga front end
//! outside its tests.
//!
//! # What is here
//!
//! - [`HostShader`]: every helper shader, with its source, its entry points and the resources
//!   it binds, as the same [`ResourceBinding`] rows a guest shader's interface has, so a
//!   backend builds bind group layouts for both the same way.
//! - [`present`] and [`clear`]: each shader's entry point names, binding numbers and the byte
//!   layout of its parameter block, named once here for the backend that fills it.
//!
//! # What the tests check, and what they cannot
//!
//! With no GPU, `host_shaders::tests` parses every source with naga's WGSL front end and
//! validates it with every validation flag on and no optional capability, at the naga version
//! the workspace pins (the one wgpu uses), then compares the parsed module with this module's
//! tables: entry point names and stages, bindings and their kinds, the parameter blocks' sizes
//! and member offsets, and each fragment entry point's output type. A `.wgsl` file under `src/`
//! that is not listed in [`HostShader::ALL`] fails `every_wgsl_file_is_a_listed_host_shader`, so
//! a shader cannot be added without being validated.
//!
//! No test here runs a shader: what the shaders draw is checked on a GPU, by the wgpu backend's
//! slices (R5 for clears, R8 for the present blit).
//!
//! # Sources
//!
//! Ported from Eden's host shaders, `eden/src/video_core/host_shaders/` @ cf654052
//! (GPL-2.0-or-later, Copyright 2019, 2020 and 2023 yuzu Emulator Project): `vulkan_present.vert`
//! and `vulkan_present.frag` for the present blit, and `vulkan_color_clear.vert` (the vertex
//! positions of `full_screen_triangle.vert`), `vulkan_color_clear.frag` and
//! `vulkan_depthstencil_clear.frag` for clears. Each `.wgsl` file's header names its sources and
//! lists where it differs from them.

use crate::resource::ViewDimension;
use crate::shader::{ResourceBinding, ResourceKind, SampleKind, ShaderStage};

/// One entry point of a helper shader.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct EntryPoint {
    /// The function's name in the source.
    pub name: &'static str,
    /// The stage it runs as.
    pub stage: ShaderStage,
}

/// Defines [`HostShader`], its list and its sources from one set of rows, so that a shader
/// cannot be a variant without being in [`HostShader::ALL`].
macro_rules! host_shaders {
    ($( $(#[$meta:meta])* $variant:ident = $file:literal ),* $(,)?) => {
        /// One of the backend's own helper shaders (module docs).
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        pub enum HostShader {
            $( $(#[$meta])* $variant, )*
        }

        impl HostShader {
            /// Every helper shader.
            pub const ALL: &'static [Self] = &[$(Self::$variant),*];

            /// The source's file name in this crate's `src/host_shaders/` folder.
            pub const fn file_name(self) -> &'static str {
                match self {
                    $(Self::$variant => $file,)*
                }
            }

            /// The WGSL source.
            pub const fn source(self) -> &'static str {
                match self {
                    $(Self::$variant => include_str!(concat!("host_shaders/", $file)),)*
                }
            }
        }
    };
}

host_shaders! {
    /// The present blit: draws a presented view into the surface texture, scaled, cropped and
    /// turned by four vertices the backend supplies ([`present`]).
    Present = "present.wgsl",
    /// Clears drawn as a triangle that covers the whole target, for the clears a render pass's
    /// load action cannot express, such as a clear within a rectangle ([`clear`]).
    Clear = "clear.wgsl",
}

impl HostShader {
    /// The shader's entry points, in the order the source defines them.
    pub const fn entry_points(self) -> &'static [EntryPoint] {
        match self {
            Self::Present => present::ENTRY_POINTS,
            Self::Clear => clear::ENTRY_POINTS,
        }
    }

    /// The resources the shader binds, in the order the source declares them.
    pub const fn resources(self) -> &'static [ResourceBinding] {
        match self {
            Self::Present => present::RESOURCES,
            Self::Clear => clear::RESOURCES,
        }
    }
}

/// The present blit's interface (`present.wgsl`).
///
/// Drawn as a triangle strip of [`VERTEX_COUNT`](present::VERTEX_COUNT) vertices with no vertex
/// buffer: vertex `i` is `vertices[i]` of the parameter block, its position multiplied by the
/// block's matrix. The fragment stage samples the presented view's base mip level at the
/// interpolated texture coordinate.
pub mod present {
    use super::{
        EntryPoint, ResourceBinding, ResourceKind, SampleKind, ShaderStage, ViewDimension,
    };

    /// The vertex entry point.
    pub const VERTEX: &str = "vs_main";

    /// The fragment entry point. Its one output, at location 0, is four floats.
    pub const FRAGMENT: &str = "fs_main";

    /// The entry points, in source order.
    pub const ENTRY_POINTS: &[EntryPoint] = &[
        EntryPoint {
            name: VERTEX,
            stage: ShaderStage::Vertex,
        },
        EntryPoint {
            name: FRAGMENT,
            stage: ShaderStage::Fragment,
        },
    ];

    /// The bind group every resource is in.
    pub const GROUP: u32 = 0;

    /// Binding of the parameter block, a uniform buffer of [`PARAMS_SIZE`] bytes.
    pub const PARAMS_BINDING: u32 = 0;

    /// Binding of the presented view, a two-dimensional float texture.
    pub const TEXTURE_BINDING: u32 = 1;

    /// Binding of the sampler the view is read with.
    pub const SAMPLER_BINDING: u32 = 2;

    /// The resources, in source order.
    pub const RESOURCES: &[ResourceBinding] = &[
        ResourceBinding {
            group: GROUP,
            binding: PARAMS_BINDING,
            kind: ResourceKind::UniformBuffer,
        },
        ResourceBinding {
            group: GROUP,
            binding: TEXTURE_BINDING,
            kind: ResourceKind::SampledTexture {
                dimension: ViewDimension::D2,
                sample: SampleKind::Float { filterable: true },
                multisampled: false,
            },
        },
        ResourceBinding {
            group: GROUP,
            binding: SAMPLER_BINDING,
            kind: ResourceKind::Sampler { comparison: false },
        },
    ];

    /// Vertices of the strip: the draw's vertex count, and the length of the block's vertex
    /// array.
    pub const VERTEX_COUNT: u32 = 4;

    /// Size of the parameter block, in bytes.
    pub const PARAMS_SIZE: u32 = 128;

    /// Offset of the matrix in the block: sixteen `f32`s, column after column.
    pub const MATRIX_OFFSET: u32 = 0;

    /// Offset of the first vertex in the block.
    pub const VERTICES_OFFSET: u32 = 64;

    /// Bytes from one vertex to the next.
    pub const VERTEX_STRIDE: u32 = 16;

    /// Offset of a vertex's position within the vertex: two `f32`s, x then y.
    pub const VERTEX_POSITION_OFFSET: u32 = 0;

    /// Offset of a vertex's texture coordinate within the vertex: two `f32`s, u then v.
    pub const VERTEX_TEX_COORD_OFFSET: u32 = 8;
}

/// The clear shader's interface (`clear.wgsl`).
///
/// Drawn as a triangle list of [`VERTEX_COUNT`](clear::VERTEX_COUNT) vertices with no vertex
/// buffer; the triangle covers the whole viewport, so the draw's scissor rectangle is what
/// bounds the clear. The parameter block holds the value as four 32-bit patterns, which each
/// fragment entry point reinterprets as its own type without converting them.
pub mod clear {
    use super::{EntryPoint, ResourceBinding, ResourceKind, ShaderStage};

    /// The vertex entry point.
    pub const VERTEX: &str = "vs_main";

    /// The fragment entry point for a float or normalised colour target: writes the four
    /// patterns as `f32`s to location 0.
    pub const FRAGMENT_COLOR_FLOAT: &str = "fs_color_float";

    /// The fragment entry point for a signed integer colour target: writes the four patterns as
    /// `i32`s to location 0.
    pub const FRAGMENT_COLOR_SINT: &str = "fs_color_sint";

    /// The fragment entry point for an unsigned integer colour target: writes the four patterns
    /// as `u32`s to location 0.
    pub const FRAGMENT_COLOR_UINT: &str = "fs_color_uint";

    /// The fragment entry point for a depth target: writes the first pattern as an `f32` to the
    /// fragment's depth. Nothing in the shader is specific to stencil: as in Eden, a stencil
    /// clear is the pipeline's stencil operation writing the reference value.
    pub const FRAGMENT_DEPTH: &str = "fs_depth";

    /// The entry points, in source order.
    pub const ENTRY_POINTS: &[EntryPoint] = &[
        EntryPoint {
            name: VERTEX,
            stage: ShaderStage::Vertex,
        },
        EntryPoint {
            name: FRAGMENT_COLOR_FLOAT,
            stage: ShaderStage::Fragment,
        },
        EntryPoint {
            name: FRAGMENT_COLOR_SINT,
            stage: ShaderStage::Fragment,
        },
        EntryPoint {
            name: FRAGMENT_COLOR_UINT,
            stage: ShaderStage::Fragment,
        },
        EntryPoint {
            name: FRAGMENT_DEPTH,
            stage: ShaderStage::Fragment,
        },
    ];

    /// The bind group the parameter block is in.
    pub const GROUP: u32 = 0;

    /// Binding of the parameter block, a uniform buffer of [`PARAMS_SIZE`] bytes.
    pub const PARAMS_BINDING: u32 = 0;

    /// The resources, in source order.
    pub const RESOURCES: &[ResourceBinding] = &[ResourceBinding {
        group: GROUP,
        binding: PARAMS_BINDING,
        kind: ResourceKind::UniformBuffer,
    }];

    /// Vertices of the covering triangle: the draw's vertex count.
    pub const VERTEX_COUNT: u32 = 3;

    /// Size of the parameter block, in bytes.
    pub const PARAMS_SIZE: u32 = 16;

    /// Offset of the value in the block: four 32-bit patterns, in the order red, green, blue,
    /// alpha for a colour, and the depth first for a depth clear.
    pub const VALUE_OFFSET: u32 = 0;
}

#[cfg(test)]
mod tests;
