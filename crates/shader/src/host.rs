//! What the crate hands out: a validated naga module and the tables describing its interface
//! (spec, "What the render backend (#14) gets, through the GPU crate": `HostShader`).
//!
//! Slice 1 fills the parts a vertex/pixel pair of the six slice-1 instructions needs: stage,
//! code, entry point, inputs, outputs and requirements. Slice 16 adds compute shaders (the
//! [`HostStage::Compute`] stage and [`HostShader::workgroup_size`]), the 16-bit float and
//! 64-bit integer requirements, and the host target's features the lowering honours
//! ([`HostCaps`]). The resource table joins with slice 13; the driver-constants layout, which
//! slice 9 defines ([`crate::driver_constants::DriverConstantsLayout`]), joins with slice 17,
//! which lowers reads of it; and the SPIR-V and MSL forms of `HostCode` with slice 22.

/// The host pipeline stage of a shader.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum HostStage {
    /// Vertex.
    Vertex,
    /// Fragment (the guest's pixel stage).
    Fragment,
    /// Compute.
    Compute,
}

/// The shader's code in the form the host target asked for.
#[derive(Clone, Debug)]
pub enum HostCode {
    /// A validated naga module and its validation info: what the wgpu backend takes (owner's
    /// decision of 2026-11-12).
    Naga {
        /// The module.
        module: Box<naga::Module>,
        /// What naga's validator found, which naga's writers need.
        info: Box<naga::valid::ModuleInfo>,
    },
}

/// A host built-in an interface slot binds to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum HostBuiltIn {
    /// The clip-space position a vertex shader writes.
    Position,
    /// The invocation's index in its subgroup (the guest's lane id).
    SubgroupInvocationId,
}

/// Where an interface slot is bound in the module.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SlotBinding {
    /// A numbered location: a vertex attribute, a varying or a colour target.
    Location(u32),
    /// A built-in.
    BuiltIn(HostBuiltIn),
}

/// The type of an interface slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SlotType {
    /// Four 32-bit floats.
    F32x4,
    /// One 32-bit unsigned integer.
    U32,
}

/// What a slot is to the guest.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GuestSlot {
    /// Generic attribute `n` (vertex input or varying), attribute words `32 + 4n` to `35 + 4n`.
    Generic(u8),
    /// The position, attribute words 28 to 31.
    Position,
    /// Colour target `n`.
    RenderTarget(u8),
    /// `SR_LANEID`.
    LaneId,
}

/// One input or output of a shader.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct InterfaceSlot {
    /// Where the module binds it.
    pub binding: SlotBinding,
    /// Its type.
    pub ty: SlotType,
    /// What it is to the guest.
    pub guest: GuestSlot,
}

/// Device features a shader needs beyond the baseline.
///
/// Slice 1 fills `subgroup` from what it lowers; slice 15's collect info
/// ([`crate::info::ShaderInfo::requirements`]) fills every field from the IR, before the host
/// lowering passes of slice 18 remove what the host lacks.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "one flag per host feature the spec's Requirements names"
)]
pub struct Requirements {
    /// Subgroup operations or built-ins (naga's `Capabilities::SUBGROUP`, wgpu's
    /// `Features::SUBGROUP`). Guest warp code assumes 32 lanes, so this also means "needs a
    /// subgroup size of 32" (spec, "Compute shaders"), which the render backend checks.
    pub subgroup: bool,
    /// 16-bit float arithmetic (naga's `Capabilities::SHADER_FLOAT16`, wgpu's
    /// `Features::SHADER_F16`).
    pub f16: bool,
    /// 64-bit floats (naga's `Capabilities::FLOAT64`).
    pub f64: bool,
    /// 64-bit integer arithmetic (naga's `Capabilities::SHADER_INT64`, wgpu's
    /// `Features::SHADER_INT64`).
    pub i64: bool,
    /// 64-bit integer atomics (naga's `Capabilities::SHADER_INT64_ATOMIC_ALL_OPS`).
    pub i64_atomics: bool,
}

/// A device feature a [`Requirements`] can name, in the order [`Requirements::missing`] checks
/// them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Feature {
    /// 16-bit floats ([`Requirements::f16`]).
    Float16,
    /// 64-bit integers ([`Requirements::i64`]).
    Int64,
    /// Subgroup operations ([`Requirements::subgroup`]).
    Subgroup,
}

impl Feature {
    /// The feature in plain words, as error messages print it.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Float16 => "16-bit floats",
            Self::Int64 => "64-bit integers",
            Self::Subgroup => "subgroup operations",
        }
    }
}

impl std::fmt::Display for Feature {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// The features a host target supports, from the render backend's `Caps` (spec, "What the
/// render backend (#14) gets": `HostTarget { form, caps: HostCaps }`; only the features the
/// lowering can need are here so far). The cache never hands out a module that needs a feature
/// the target lacks: such a shader is rejected with that reason.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct HostCaps {
    /// 16-bit float arithmetic.
    pub f16: bool,
    /// 64-bit integer arithmetic.
    pub i64: bool,
    /// Subgroup operations with a subgroup size of 32.
    pub subgroup: bool,
}

impl HostCaps {
    /// A target with every feature.
    pub const ALL: Self = Self {
        f16: true,
        i64: true,
        subgroup: true,
    };
    /// A target with none.
    pub const NONE: Self = Self {
        f16: false,
        i64: false,
        subgroup: false,
    };
}

impl Requirements {
    /// The first feature, in [`Feature`] order, that this needs and `caps` lacks.
    pub const fn missing(self, caps: HostCaps) -> Option<Feature> {
        if self.f16 && !caps.f16 {
            Some(Feature::Float16)
        } else if self.i64 && !caps.i64 {
            Some(Feature::Int64)
        } else if self.subgroup && !caps.subgroup {
            Some(Feature::Subgroup)
        } else {
            None
        }
    }
}

/// One translated shader.
#[derive(Clone, Debug)]
pub struct HostShader {
    /// The stage.
    pub stage: HostStage,
    /// The code.
    pub code: HostCode,
    /// The entry point's name in the module.
    pub entry: String,
    /// The inputs, in binding order (locations ascending, then built-ins).
    pub inputs: Vec<InterfaceSlot>,
    /// The outputs, in binding order (built-ins, then locations ascending).
    pub outputs: Vec<InterfaceSlot>,
    /// What the device must support.
    pub requirements: Requirements,
    /// A compute shader's workgroup size, x, y and z (from the variant key); `None` for the
    /// other stages.
    pub workgroup_size: Option<[u32; 3]>,
}
