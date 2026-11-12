//! What the crate hands out: a validated naga module and the tables describing its interface
//! (spec, "What the render backend (#14) gets, through the GPU crate": `HostShader`).
//!
//! Slice 1 fills the parts a vertex/pixel pair of the six slice-1 instructions needs: stage,
//! code, entry point, inputs, outputs and requirements. The resource table, the driver-constants
//! layout and the workgroup size join with the slices that produce them (13, 9 and 16), and
//! the SPIR-V and MSL forms of `HostCode` with slice 22.

/// The host pipeline stage of a shader.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum HostStage {
    /// Vertex.
    Vertex,
    /// Fragment (the guest's pixel stage).
    Fragment,
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
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Requirements {
    /// Subgroup operations or built-ins (naga's `Capabilities::SUBGROUP`, wgpu's
    /// `Features::SUBGROUP`).
    pub subgroup: bool,
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
}
