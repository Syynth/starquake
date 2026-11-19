//! Shaders as the backend receives them: a [`ShaderDesc`] with its code and [`ShaderInterface`].
//!
//! The shader crate hands over a validated naga module with tables of its resources, inputs and
//! outputs; the GPU crate carries it here as [`ShaderSource::Naga`] and converts the tables into
//! a [`ShaderInterface`] (docs/specs/render-backend.md, "Shader emission"; owner's decision of
//! 2026-11-12). Group and binding numbers are the shader crate's, already written into the
//! module; this crate never assigns them. SPIR-V and MSL serve a later direct backend, captures
//! and experiments.
//!
//! The interface also carries what the backend would otherwise read from the module: the
//! shader's [`ShaderRequirements`] and each binding's array count ([`ResourceBinding::count`]),
//! so a backend builds its layouts and checks its requirements from the interface alone
//! (render-backend, "Shader emission"; shader-pipeline, "What the render backend (#14) gets";
//! decided by the owner, 2026-11-19, ruling 20).
//!
//! A shader may carry its [`ShaderOrigin`]: the inputs the shader crate translated it from, which
//! a render capture stores so that a replay can translate it again (render-backend, "Render
//! capture v2", "Shaders"; the owner's decision 15 of 2026-11-18). Backends ignore it.

use crate::format::PixelFormat;
use crate::resource::ViewDimension;
use serde::{Deserialize, Serialize};
use std::num::NonZeroU32;
use std::sync::Arc;

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
    /// How many resources of `kind` the binding holds: `None` for one, `Some(n)` for a binding
    /// array of `n` (a layout entry's count; render-backend, "Shader emission"). A binding array
    /// whose size is not a constant has no count, so an interface cannot describe one.
    pub count: Option<NonZeroU32>,
}

/// Device features a shader needs beyond the baseline: the render crate's form of the shader
/// crate's `Requirements`, which the GPU crate copies field by field (shader-pipeline, "What the
/// render backend (#14) gets": `requirements`). A backend honours each one or refuses the
/// pipeline.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ShaderRequirements {
    /// The shader uses subgroup operations or built-ins (naga's `Capabilities::SUBGROUP`), and
    /// so needs subgroups 32 invocations wide, the guest's warp (shader-pipeline, "Compute
    /// shaders": every stage that uses subgroup operations "records 'needs subgroup size 32'
    /// in its requirements").
    pub subgroup: bool,
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
    /// What the device must support to run the shader.
    pub requirements: ShaderRequirements,
}

/// Where a shader came from: the shader crate's encoded capture record of the shader entry
/// (shader-pipeline slice 20's format, opaque here, since this crate cannot name the shader
/// crate's types) and the stage's index in that entry's shader set (render-backend, "Render
/// capture v2", "Shaders").
///
/// The GPU crate fills it only when the run captures; a render capture stores each distinct
/// `record` once ([`crate::capture`]) and a replay hands it back to a
/// [`crate::capture::ShaderResolver`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShaderOrigin {
    /// The encoded record, as the shader crate wrote it.
    pub record: Arc<[u8]>,
    /// The stage's index in the record's shader set.
    pub stage_index: u32,
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
    /// What the shader was translated from, when the run captures; `None` otherwise. Backends
    /// ignore it.
    pub origin: Option<ShaderOrigin>,
}
