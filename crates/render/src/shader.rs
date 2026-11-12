//! Shaders as the backend receives them: a [`ShaderDesc`] with its code and [`ShaderInterface`].
//!
//! The shader crate hands over a validated naga module with tables of its resources, inputs and
//! outputs; the GPU crate carries it here as [`ShaderSource::Naga`] and converts the tables into
//! a [`ShaderInterface`] (docs/specs/render-backend.md, "Shader emission"; owner's decision of
//! 2026-11-12). Group and binding numbers are the shader crate's, already written into the
//! module; this crate never assigns them. SPIR-V and MSL serve a later direct backend, captures
//! and experiments.

use crate::format::PixelFormat;
use crate::resource::ViewDimension;
use serde::{Deserialize, Serialize};

/// A shader stage.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ShaderStage {
    /// Vertex.
    Vertex,
    /// Tessellation control (hull).
    TessControl,
    /// Tessellation evaluation (domain).
    TessEval,
    /// Geometry.
    Geometry,
    /// Fragment (pixel).
    Fragment,
    /// Compute.
    Compute,
}

impl ShaderStage {
    /// The stage's name in plain words, as errors print it.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Vertex => "vertex",
            Self::TessControl => "tessellation control",
            Self::TessEval => "tessellation evaluation",
            Self::Geometry => "geometry",
            Self::Fragment => "fragment",
            Self::Compute => "compute",
        }
    }
}

/// A shader's code.
#[derive(Debug)]
pub enum ShaderSource {
    /// A validated naga module with its validation info: what the wgpu backend takes.
    Naga {
        /// The module.
        module: Box<naga::Module>,
        /// What naga's validator reported about it.
        info: Box<naga::valid::ModuleInfo>,
    },
    /// SPIR-V words, for a later direct Vulkan backend.
    SpirV(Vec<u32>),
    /// Metal shading language source, for a later direct Metal backend.
    Msl(String),
}

/// The scalar type of an interface variable.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ScalarKind {
    /// 32-bit float.
    Float,
    /// 32-bit signed integer.
    Sint,
    /// 32-bit unsigned integer.
    Uint,
}

/// How a stage output is interpolated into the next stage's input.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Interpolation {
    /// Perspective-correct.
    Perspective,
    /// Linear in screen space.
    Linear,
    /// Not interpolated: the provoking vertex's value.
    Flat,
}

/// One input or output of a stage, at a location.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct InterfaceVariable {
    /// The location number.
    pub location: u32,
    /// Scalar type.
    pub scalar: ScalarKind,
    /// Components, 1 to 4.
    pub components: u32,
    /// Interpolation.
    pub interpolation: Interpolation,
}

/// What a sampled texture returns to the shader.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SampleKind {
    /// Floats; `filterable` when the shader samples it with a filtering sampler.
    Float {
        /// Whether a filtering sampler is used with it.
        filterable: bool,
    },
    /// Depth values, for comparison sampling.
    Depth,
    /// Signed integers.
    Sint,
    /// Unsigned integers.
    Uint,
}

/// The kind of resource at a binding.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ResourceKind {
    /// A uniform (constant) buffer.
    UniformBuffer,
    /// A storage buffer.
    StorageBuffer {
        /// Whether the shader writes it.
        writable: bool,
    },
    /// A sampled texture.
    SampledTexture {
        /// The view dimension the shader declares.
        dimension: ViewDimension,
        /// What sampling returns.
        sample: SampleKind,
        /// Whether it is multisampled.
        multisampled: bool,
    },
    /// A storage image.
    StorageTexture {
        /// The view dimension the shader declares.
        dimension: ViewDimension,
        /// The image's format.
        format: PixelFormat,
        /// Whether the shader writes it.
        writable: bool,
    },
    /// A sampler.
    Sampler {
        /// Whether it is a comparison (shadow) sampler.
        comparison: bool,
    },
}

/// A resource the shader uses, at the group and binding the shader crate assigned.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ResourceBinding {
    /// Bind group (descriptor set) number.
    pub group: u32,
    /// Binding number within the group.
    pub binding: u32,
    /// What is bound there.
    pub kind: ResourceKind,
}

/// The backend-neutral tables of a shader's resources, inputs and outputs, from which a backend
/// builds bind group and pipeline layouts (render-backend, "Shader emission").
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ShaderInterface {
    /// Resources, each at a distinct group and binding.
    pub resources: Vec<ResourceBinding>,
    /// Stage inputs, each at a distinct location.
    pub inputs: Vec<InterfaceVariable>,
    /// Stage outputs, each at a distinct location.
    pub outputs: Vec<InterfaceVariable>,
}

/// A shader for `Backend::create_shader`.
#[derive(Debug)]
pub struct ShaderDesc {
    /// The stage it runs as.
    pub stage: ShaderStage,
    /// The entry point's name.
    pub entry: String,
    /// The code.
    pub source: ShaderSource,
    /// Its resources, inputs and outputs.
    pub interface: ShaderInterface,
}
