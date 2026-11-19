//! Slice 15: inputs and outputs, and collect info (`docs/specs/shader-pipeline.md`, "Passes
//! needed for correctness", items 5 and 7).
//!
//! [`collect`] walks a translated function with the program's header (slice 2, [`crate::sph`])
//! and gathers what the lowering and the render backend need to know about it: which attribute
//! words it reads and writes ([`ShaderInfo::loads`], [`ShaderInfo::stores`]; the variant key's
//! interface-matching field, slice 9, takes them), its interface as tables of guest slots
//! ([`ShaderInfo::inputs`], [`ShaderInfo::outputs`], with each pixel input's interpolation),
//! the colour components, sample mask and depth a pixel program writes, the system values it
//! reads, the constant buffers it reads and how much of each ([`ShaderInfo::constant_buffers`],
//! the part of the resource table this pass owns), and the host features it needs
//! ([`ShaderInfo::requirements`]).
//!
//! [`check_header`] and [`translation_stage`] are the header's wiring: a graphics program's
//! header must name the stage of the pipeline slot it is bound to, and a pixel program's colour
//! outputs come from its header. [`crate::program::graphics_program_ir`] runs the chain on a
//! program's bytes: header, translation, the slice 8 passes, then this pass.
//!
//! # Sources
//!
//! Ported by behaviour from Eden (`eden/src/shader_recompiler/` @ cf654052, GPL-3.0-or-later
//! over GPL-2.0-or-later yuzu code): `ir_opt/collect_shader_info_pass.cpp` (`VisitUsages`, the
//! constant-buffer part of `Visit`, `GatherInfoFromHeader`), `frontend/maxwell/translate_program.cpp`
//! (`CollectInterpolationInfo`), and the attribute list of `frontend/ir/attribute.h`
//! ([`attribute`]). Pixel outputs follow `frontend/maxwell/translate/impl/exit_program.cpp`
//! and PR #4588's `EXIT` (`eden-pr4588/src/shader_recompiler/frontend/ampere/ops_attr.cpp` @
//! e5fb3b97), which the translator ports.
//!
//! # What is collected
//!
//! | Operation | Collected | Eden |
//! |---|---|---|
//! | `GetAttribute`, `GetAttributeU32`, `InterpolateAtOffset` | the word in [`ShaderInfo::loads`] | `info.loads.mask[attr] = true` (`InterpolateAtOffset` is starquake's opcode, a load) |
//! | `SetAttribute` | the word in [`ShaderInfo::stores`] | `info.stores.mask[attr] = true` |
//! | `GetAttributeIndexed`, `SetAttributeIndexed` | [`ShaderInfo::loads_indexed`], [`ShaderInfo::stores_indexed`], then the header fills the masks (below) | `loads_indexed_attributes`, `stores_indexed_attributes` |
//! | `SetFragColor` | the component in [`ShaderInfo::color_outputs`] | `stores_frag_color[target]` (per target) |
//! | `SetSampleMask`, `SetFragDepth` | [`ShaderInfo::stores_sample_mask`], [`ShaderInfo::stores_frag_depth`] | `stores_sample_mask`, `stores_frag_depth` |
//! | `WorkgroupId`, `LocalInvocationId`, `InvocationId`, `InvocationInfo`, `SampleId`, `IsHelperInvocation`, `LaneId`, the five subgroup masks | a [`SystemValue`] | `uses_workgroup_id` and the rest |
//! | `GetCbuf*` | the bank and the size read ([`ConstantBufferUse`]) | `AddConstantBufferDescriptor`, `constant_buffer_used_sizes` |
//! | each opcode of Eden's lists | a [`Requirements`] flag ([`opcode_requirements`]) | `uses_fp16`, `uses_fp64`, `uses_int64`, `uses_int64_bit_atomics`, the subgroup flags |
//!
//! Constant-buffer sizes are Eden's: a read at an immediate offset grows the bank's size to the
//! end of the read rounded up to [`CBUF_ALIGNMENT`]; a read at a register offset makes it
//! [`INDEXED_CBUF_SIZE`]; a read whose bank is a register marks every bank read at that size
//! ([`ShaderInfo::indexed_constant_buffer_bank`]).
//!
//! **The header fill** is Eden's `GatherInfoFromHeader`: for a pixel program with indexed
//! loads, each generic word of [`ShaderInfo::loads`] is set exactly when the header's input map
//! gives its component a mode other than unused (set or cleared, as Eden assigns it); for other
//! graphics programs with indexed loads, the generic words, the clip distances, the primitive
//! id, layer, viewport index, point size, position, point sprite, fog coordinate, tessellation
//! point, instance id and vertex id words of the loads are set or cleared from the input map,
//! and with indexed stores the same words of the stores from the output map. Legacy colours
//! and fixed-function texture coordinates are not filled (Eden: "TODO: Legacy varyings").
//! Compute programs have no header and nothing is filled.
//!
//! **Interpolation** is Eden's `CollectInterpolationInfo`, for pixel programs: a generic
//! attribute whose components the input map gives modes other than unused takes that mode
//! ([`Interpolation::of`]), one that gives none is [`Interpolation::Smooth`], and one whose used
//! components disagree is refused ([`InfoError::MixedInterpolation`]; Eden throws "Per component
//! interpolation"). Every generic is checked, read or not, as Eden checks every one.
//!
//! # Checks, in order
//!
//! [`collect`] checks the header against the stage first ([`check_header`]: the four
//! [`InfoError`] header variants), then the pixel header's interpolation
//! ([`InfoError::MixedInterpolation`], the lowest generic first), then the instructions in
//! [`Function::listed`] order, reporting the first that fails: an argument that is missing or
//! not of the kind read ([`InfoError::Operand`]: an attribute immediate, a 32-bit immediate, or
//! for a constant-buffer bank and offset a 32-bit immediate or a register), an attribute word
//! no slot names ([`InfoError::UnknownAttribute`]), a colour output past the targets
//! ([`InfoError::ColorOutput`]), a constant-buffer bank past [`CBUF_BANKS`]
//! ([`InfoError::ConstantBufferBank`]) or a read whose end overflows
//! ([`InfoError::ConstantBufferOffset`]). Pinned by `tests::checks_run_in_the_documented_order`.
//! Within one constant-buffer read the bank's kind is checked, then the offset's, then the
//! bank's range, then the end (`tests::constant_buffer_arguments_of_the_wrong_immediate_kind_are_refused`).
//!
//! # Where this differs from Eden
//!
//! | What | Here | Eden | Pinned by |
//! |---|---|---|---|
//! | Which blocks are walked | Every block [`Function::listed`] lists, reachable or not | `post_order_blocks`, after `RemoveUnreachableBlocks` | `tests::unreachable_blocks_are_collected` |
//! | An attribute word no slot names (0 to 23, 187, 233 to 254, 256 and up) | Refused, [`InfoError::UnknownAttribute`] | Set in the mask; the backend throws when it emits the access | `tests::unknown_attribute_words_are_refused` |
//! | A colour output's component | Recorded per component | Per target (`stores_frag_color`) | `tests::pixel_outputs_record_each_component` |
//! | A constant-buffer bank past the last | Refused, [`InfoError::ConstantBufferBank`] | Indexes its arrays past their end | `tests::constant_buffer_banks_stop_at_the_last` |
//! | A register-indexed bank | Every one of the [`CBUF_BANKS`] banks the spec's host model has ("Passes needed for correctness", item 2) | `MAX_INDIRECT_CBUFS` = 14 banks | `tests::an_indexed_bank_reads_every_bank` |
//! | A bank's size when an immediate read ends past [`INDEXED_CBUF_SIZE`] and a register-offset or register-bank read also reads it | The larger of the two, in either order | Assigns `0x10000` at the register read, shrinking it when that read comes later | `tests::a_register_offset_keeps_a_larger_immediate_end` |
//! | A constant-buffer read whose end overflows `u32` | Refused, [`InfoError::ConstantBufferOffset`] | Wraps | `tests::constant_buffer_ends_overflow_is_refused` |
//! | The header against the stage | Checked ([`check_header`]) | Not checked: Eden reads the header layout its stage names | `tests::the_header_must_name_the_slots_stage` |
//! | `IAbs64`, `ConvertF16S64`, `ConvertF32S64`, `ConvertF64S64` | Need [`Requirements::i64`]: each takes or gives a `u64` | Missing from `uses_int64`'s list | `tests::requirement_lists_agree_with_the_opcode_types` |
//! | When the interpolation is checked | Before the instructions are walked, so a pixel program with mixed interpolation and a faulty instruction reports [`InfoError::MixedInterpolation`] | `CollectInterpolationInfo` runs after `CollectShaderInfoPass` (`frontend/maxwell/translate_program.cpp` @ cf654052, lines 407 and 411), so the instruction's fault would be met first. Only the error reported for a program with both faults differs | `tests::checks_run_in_the_documented_order` |
//! | A constant-buffer bank or offset that is an immediate of another kind than `u32`, or a missing offset | Refused, [`InfoError::Operand`] | `Value::U32` has only `DEBUG_ASSERT(type == Type::U32)` (`frontend/ir/value.h`), so a release build reads the immediate's bits; Eden's opcode table fixes the argument count | `tests::constant_buffer_arguments_of_the_wrong_immediate_kind_are_refused` |
//! | `GetPatch`, `SetPatch` | Not collected, and not refused: the walk ignores both opcodes, whatever the patch word. The translator emits neither yet | `GetPatch` and `SetPatch` set `uses_patches`, and `SetPatch` of a tessellation level sets `stores_tess_level_outer` or `stores_tess_level_inner`; Eden throws on a `GetPatch` of a non-generic patch and on a `SetPatch` of any other patch | `tests::patch_operations_are_not_collected` |
//! | Every other `Info` field `collect_shader_info_pass.cpp` writes: `uses_int8`, `uses_int16`; the float denormal flags (`uses_fp16_denorms_flush`, `uses_fp16_denorms_preserve`, `uses_fp32_denorms_flush`, `uses_fp32_denorms_preserve`, from `VisitFpModifiers`); the atomic-kind flags (`uses_atomic_s32_min`, `uses_atomic_s32_max`, `uses_atomic_f32_add`, `uses_atomic_f16x2_add`, `uses_atomic_f32x2_add`, `uses_atomic_f16x2_min`, `uses_atomic_f32x2_min`, `uses_atomic_f16x2_max`, `uses_atomic_f32x2_max`, `uses_shared_increment`, `uses_shared_decrement`, `uses_global_increment`, `uses_global_decrement`); the texture and image flags (`uses_sampled_1d`, `uses_sparse_residency`, `uses_shadow_lod`, `uses_typeless_image_reads`, `uses_typeless_image_writes`, `uses_image_buffers`, `uses_image_1d`, `uses_atomic_image_u32`); `uses_global_memory`, `stores_global_memory`, `used_storage_buffer_types`, `uses_local_memory`; `uses_demote_to_helper_invocation`; `uses_derivatives`; `uses_rescaling_uniform` (the spec deliberately does not port Eden's rescaling pass: resolution scaling is a later feature); `used_constant_buffer_types`, `used_indirect_cbuf_types`, `constant_buffer_descriptors` and `constant_buffer_mask` (the banks and sizes of [`ShaderInfo::constant_buffers`] stand in for the last two); `used_clip_distances` (`GatherInfoFromHeader`); the NVN storage buffers (`nvn_buffer_base`, `nvn_buffer_used`); and the `uses_is_helper_invocation` Eden also sets for `GetAttributeIndexed` (here only `IsHelperInvocation` records [`SystemValue::IsHelperInvocation`]) | Not collected. 8- and 16-bit integers are not host requirements here: a guess that the lowering (slices 16 and 17) reads them through 32-bit values, since naga has no 8-bit integer type. The rest serve Eden's own SPIR-V capabilities, or slices 13, 14, 16 to 18 | Collected | — |
//!
//! # Open, for the owner
//!
//! wgpu refuses a pipeline whose vertex output and fragment input interpolation differ
//! (`wgpu-core` 30.0.2, `validation.rs`: `InputError::InterpolationMismatch`), and only the
//! pixel program's header gives the mode. A vertex program's outputs carry no interpolation
//! here ([`InterfaceEntry::interpolation`] is `None` outside pixel inputs), so the lowering of
//! slice 17 needs the pixel program's modes for the vertex outputs: from the next stage (which
//! the 2026-11-19 ruling on interface matching does not specialise on) or by another route.
//! Slice 1's guess of perspective interpolation for vertex outputs stays in `lower` until then.

pub mod attribute;
#[cfg(test)]
mod tests;

use thiserror::Error;

use crate::host::Requirements;
use crate::ir::{
    ATTRIBUTE_COMPONENTS, Attribute, Function, GENERIC_0_X, Imm, Inst, Opcode, Operand, Operation,
    POSITION_X, Value,
};
use crate::sph::{Components, Io, PixelImap, PixelIo, ProgramHeader, VtgMap};
use crate::translate::float::CBUF_BANKS;
use crate::translate::{ColorOutputs, GuestStage, Stage};
use crate::variant::{ATTRIBUTE_WORDS, AttributeMask};
use attribute::{
    CLIP_DISTANCE_0, CLIP_DISTANCES, FOG_COORDINATE, GENERICS, INSTANCE_ID, LAYER, POINT_SIZE,
    POINT_SPRITE_S, PRIMITIVE_ID, Slot, TESSELLATION_POINT_U, VERTEX_ID, VIEWPORT_INDEX, slot_of,
};

/// The alignment of a constant buffer's used size (Eden: `Common::AlignUp(.., 16u)`).
pub const CBUF_ALIGNMENT: u32 = 16;
/// The size of a constant buffer read at a register offset, the whole 64 KiB a bank can hold
/// (Eden: `size = 0x10'000`).
pub const INDEXED_CBUF_SIZE: u32 = 0x1_0000;

/// How a pixel program's input is interpolated (Eden's `Interpolation`, `shader_info.h`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Interpolation {
    /// Perspective-correct.
    #[default]
    Smooth,
    /// One value for the whole primitive.
    Flat,
    /// Linear in screen space.
    NoPerspective,
}

impl Interpolation {
    /// The interpolation an input map mode gives (`CollectInterpolationInfo`: `Unused` and
    /// `Perspective` are `Smooth`, `Constant` is `Flat`, `ScreenLinear` is `NoPerspective`).
    pub const fn of(imap: PixelImap) -> Self {
        match imap {
            PixelImap::Unused | PixelImap::Perspective => Self::Smooth,
            PixelImap::Constant => Self::Flat,
            PixelImap::ScreenLinear => Self::NoPerspective,
        }
    }
}

/// A value the program reads that is neither an attribute nor a register.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum SystemValue {
    /// `WorkgroupId` (`uses_workgroup_id`).
    WorkgroupId,
    /// `LocalInvocationId` (`uses_local_invocation_id`).
    LocalInvocationId,
    /// `InvocationId` (`uses_invocation_id`).
    InvocationId,
    /// `InvocationInfo` (`uses_invocation_info`).
    InvocationInfo,
    /// `SampleId` (`uses_sample_id`).
    SampleId,
    /// `IsHelperInvocation` (`uses_is_helper_invocation`).
    IsHelperInvocation,
    /// `LaneId` (`uses_subgroup_invocation_id`).
    SubgroupInvocationId,
    /// `SubgroupEqMask` to `SubgroupGeMask` (`uses_subgroup_mask`).
    SubgroupMasks,
}

impl SystemValue {
    /// Every system value, in declaration order.
    pub const ALL: [Self; 8] = [
        Self::WorkgroupId,
        Self::LocalInvocationId,
        Self::InvocationId,
        Self::InvocationInfo,
        Self::SampleId,
        Self::IsHelperInvocation,
        Self::SubgroupInvocationId,
        Self::SubgroupMasks,
    ];

    /// The system value `opcode` reads, if it reads one (`VisitUsages`).
    pub const fn of(opcode: Opcode) -> Option<Self> {
        Some(match opcode {
            Opcode::WorkgroupId => Self::WorkgroupId,
            Opcode::LocalInvocationId => Self::LocalInvocationId,
            Opcode::InvocationId => Self::InvocationId,
            Opcode::InvocationInfo => Self::InvocationInfo,
            Opcode::SampleId => Self::SampleId,
            Opcode::IsHelperInvocation => Self::IsHelperInvocation,
            Opcode::LaneId => Self::SubgroupInvocationId,
            Opcode::SubgroupEqMask
            | Opcode::SubgroupLtMask
            | Opcode::SubgroupLeMask
            | Opcode::SubgroupGtMask
            | Opcode::SubgroupGeMask => Self::SubgroupMasks,
            _ => return None,
        })
    }

    const fn bit(self) -> u16 {
        1 << self as u16
    }
}

/// A set of [`SystemValue`]s.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct SystemValues(u16);

impl SystemValues {
    /// No values.
    pub const EMPTY: Self = Self(0);

    /// The set with `value` added.
    #[must_use]
    pub const fn with(self, value: SystemValue) -> Self {
        Self(self.0 | value.bit())
    }

    /// Whether `value` is in the set.
    pub const fn contains(self, value: SystemValue) -> bool {
        self.0 & value.bit() != 0
    }

    /// The values in the set, in declaration order.
    pub fn iter(self) -> impl Iterator<Item = SystemValue> {
        SystemValue::ALL
            .into_iter()
            .filter(move |&v| self.contains(v))
    }
}

/// One entry of an interface table.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct InterfaceEntry {
    /// The slot.
    pub slot: Slot,
    /// Which of the slot's components are used: bit `c` for component `c` (an attribute slot's
    /// words in order, a colour target's red to alpha; bit 0 for a slot with one component).
    pub components: u8,
    /// For a pixel program's generic input, how it is interpolated; `None` for every other
    /// entry (see the module docs, "Open, for the owner").
    pub interpolation: Option<Interpolation>,
}

/// One constant-buffer bank the program reads, and how many bytes of it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ConstantBufferUse {
    /// The bank, below [`CBUF_BANKS`].
    pub bank: u8,
    /// The bytes read from the bank's start: a multiple of [`CBUF_ALIGNMENT`], or
    /// [`INDEXED_CBUF_SIZE`] for a read at a register offset.
    pub size: u32,
}

/// Why a function's info was not collected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum InfoError {
    /// A graphics stage was given no header.
    #[error("a {0} program needs its shader program header")]
    NoHeader(GuestStage),
    /// A compute program was given a header (compute programs have none).
    #[error("a compute program has no shader program header, but one was given")]
    ComputeHeader,
    /// The header names another stage than the pipeline slot the program is bound to.
    #[error("a program bound as a {slot} program has a {header} header")]
    HeaderStage {
        /// The pipeline slot's stage.
        slot: GuestStage,
        /// The stage the header names.
        header: GuestStage,
    },
    /// A pixel program's input map gives the used components of one generic attribute different
    /// modes (Eden: "Per component interpolation").
    #[error("generic attribute {generic} is interpolated differently per component")]
    MixedInterpolation {
        /// The generic attribute.
        generic: u8,
    },
    /// An argument is missing, or is not of the kind collect info reads: an attribute
    /// immediate for an attribute access, a 32-bit immediate for a colour output's target and
    /// component, and a 32-bit immediate or a register for a constant-buffer bank and offset.
    #[error("{at}: argument {index} is missing or not of the kind collect info reads")]
    Operand {
        /// The instruction.
        at: Value,
        /// The argument's index.
        index: usize,
    },
    /// An attribute word that no slot names.
    #[error("{at}: attribute {attribute} is not one the guest has")]
    UnknownAttribute {
        /// The instruction.
        at: Value,
        /// The attribute word.
        attribute: Attribute,
    },
    /// A colour output past the targets or components that exist.
    #[error("{at}: colour output {target}.{component} does not exist")]
    ColorOutput {
        /// The instruction.
        at: Value,
        /// The target.
        target: u32,
        /// The component.
        component: u32,
    },
    /// A constant-buffer bank at or past [`CBUF_BANKS`].
    #[error("{at}: constant buffer bank {bank} does not exist; banks stop at {last}", last = CBUF_BANKS - 1)]
    ConstantBufferBank {
        /// The instruction.
        at: Value,
        /// The bank.
        bank: u32,
    },
    /// A constant-buffer read whose end, rounded up to [`CBUF_ALIGNMENT`], does not fit 32 bits.
    #[error("{at}: a constant buffer read at offset {offset:#x} ends past 4 GiB")]
    ConstantBufferOffset {
        /// The instruction.
        at: Value,
        /// The offset.
        offset: u32,
    },
}

/// What collect info found in one program.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShaderInfo {
    stage: GuestStage,
    interpolation: [Interpolation; GENERICS as usize],
    inputs: Vec<InterfaceEntry>,
    outputs: Vec<InterfaceEntry>,
    walk: Walk,
}

impl ShaderInfo {
    /// The stage the info was collected for.
    pub const fn stage(&self) -> GuestStage {
        self.stage
    }

    /// The attribute words the program reads, after the header fill.
    pub const fn loads(&self) -> AttributeMask {
        self.walk.loads
    }

    /// The attribute words the program writes, after the header fill.
    pub const fn stores(&self) -> AttributeMask {
        self.walk.stores
    }

    /// Whether the program reads attributes at a register-computed word (Eden's
    /// `loads_indexed_attributes`).
    pub const fn loads_indexed(&self) -> bool {
        self.walk.indexed.loads
    }

    /// Whether the program writes attributes at a register-computed word (Eden's
    /// `stores_indexed_attributes`).
    pub const fn stores_indexed(&self) -> bool {
        self.walk.indexed.stores
    }

    /// How a pixel program interpolates generic attribute `generic`; [`Interpolation::Smooth`]
    /// for every generic of other stages and past the last generic (Eden's default).
    pub fn interpolation(&self, generic: u8) -> Interpolation {
        self.interpolation
            .get(usize::from(generic))
            .copied()
            .unwrap_or_default()
    }

    /// The inputs: one entry per slot with a word in [`Self::loads`], in word order.
    pub fn inputs(&self) -> &[InterfaceEntry] {
        &self.inputs
    }

    /// The outputs: one entry per slot with a word in [`Self::stores`], in word order, then for
    /// a pixel program each colour target written, the sample mask and the depth.
    pub fn outputs(&self) -> &[InterfaceEntry] {
        &self.outputs
    }

    /// The colour components the program writes (`SetFragColor`).
    pub const fn color_outputs(&self) -> ColorOutputs {
        self.walk.pixel.color
    }

    /// Whether the program writes the sample mask.
    pub const fn stores_sample_mask(&self) -> bool {
        self.walk.pixel.sample_mask
    }

    /// Whether the program writes depth.
    pub const fn stores_frag_depth(&self) -> bool {
        self.walk.pixel.depth
    }

    /// The system values the program reads.
    pub const fn system_values(&self) -> SystemValues {
        self.walk.system_values
    }

    /// The constant-buffer banks the program reads, ascending, with their sizes.
    pub fn constant_buffers(&self) -> impl Iterator<Item = ConstantBufferUse> + '_ {
        (0_u8..)
            .zip(self.walk.constant_buffers.sizes)
            .filter_map(|(bank, size)| size.map(|size| ConstantBufferUse { bank, size }))
    }

    /// Whether the program reads a constant buffer whose bank is a register (Eden's
    /// `uses_cbuf_indirect`).
    pub const fn indexed_constant_buffer_bank(&self) -> bool {
        self.walk.constant_buffers.indexed_bank
    }

    /// The host features the program needs, before the host lowering passes of slice 18.
    pub const fn requirements(&self) -> Requirements {
        self.walk.requirements
    }
}

/// The stage a header must name for a program bound to `slot`; `None` for compute, which has
/// no header. Both vertex slots take a vertex header: Eden's `VertexA` and `VertexB` read the
/// same layout (`program_header.h`, `vtg`), and the header has one vertex shader type
/// (`SHADER_TYPE_VERTEX`).
pub const fn header_stage(slot: GuestStage) -> Option<GuestStage> {
    match slot {
        GuestStage::VertexCullBeforeFetch | GuestStage::Vertex => Some(GuestStage::Vertex),
        GuestStage::TessellationInit => Some(GuestStage::TessellationInit),
        GuestStage::Tessellation => Some(GuestStage::Tessellation),
        GuestStage::Geometry => Some(GuestStage::Geometry),
        GuestStage::Pixel => Some(GuestStage::Pixel),
        GuestStage::Compute => None,
    }
}

/// Checks that a program bound to `slot` has the header its stage needs: none for compute,
/// else one naming [`header_stage`] of `slot`. Starquake's check: Eden reads the layout the
/// slot names whatever the header says.
pub fn check_header(slot: GuestStage, header: Option<&ProgramHeader>) -> Result<(), InfoError> {
    match (header_stage(slot), header) {
        (None, None) => Ok(()),
        (None, Some(_)) => Err(InfoError::ComputeHeader),
        (Some(_), None) => Err(InfoError::NoHeader(slot)),
        (Some(expected), Some(header)) if header.stage() == expected => Ok(()),
        (Some(_), Some(header)) => Err(InfoError::HeaderStage {
            slot,
            header: header.stage(),
        }),
    }
}

/// What the translator is given for a graphics program bound to `slot` with `header`: the
/// slot's stage, and for a pixel header its colour outputs ([`PixelIo::color_outputs`]); no
/// colour outputs for other headers.
pub fn translation_stage(slot: GuestStage, header: &ProgramHeader) -> Stage {
    let color_outputs = match &header.io {
        Io::Pixel(io) => io.color_outputs(),
        Io::Vtg(_) => ColorOutputs::default(),
    };
    Stage {
        stage: slot,
        color_outputs,
    }
}

/// The [`Requirements`] one operation of `opcode` needs: Eden's lists in `VisitUsages`, each
/// written out as Eden has it, plus four `u64` forms its `uses_int64` list misses (marked; the
/// module docs' divergence table) (`uses_fp16`, `uses_fp64`, `uses_int64` and the global-memory
/// case that sets it, `uses_int64_bit_atomics`, and the subgroup flags: `uses_subgroup_invocation_id`,
/// `uses_subgroup_mask`, `uses_subgroup_vote`, `uses_subgroup_shuffles`, `uses_quad_shuffles`,
/// `uses_fswzadd`, all of which [`Requirements::subgroup`] covers).
pub const fn opcode_requirements(opcode: Opcode) -> Requirements {
    Requirements {
        subgroup: uses_subgroup(opcode),
        f16: uses_f16(opcode),
        f64: uses_f64(opcode),
        i64: uses_i64(opcode) || is_global_store_or_atomic(opcode),
        i64_atomics: uses_i64_atomics(opcode),
    }
}

/// Eden's `uses_fp16` list.
const fn uses_f16(opcode: Opcode) -> bool {
    use Opcode as O;
    matches!(
        opcode,
        O::CompositeConstructF16x2
            | O::CompositeConstructF16x3
            | O::CompositeConstructF16x4
            | O::CompositeExtractF16x2
            | O::CompositeExtractF16x3
            | O::CompositeExtractF16x4
            | O::CompositeInsertF16x2
            | O::CompositeInsertF16x3
            | O::CompositeInsertF16x4
            | O::SelectF16
            | O::BitCastU16F16
            | O::BitCastF16U16
            | O::PackFloat2x16
            | O::UnpackFloat2x16
            | O::ConvertS16F16
            | O::ConvertS32F16
            | O::ConvertS64F16
            | O::ConvertU16F16
            | O::ConvertU32F16
            | O::ConvertU64F16
            | O::ConvertF16S8
            | O::ConvertF16S16
            | O::ConvertF16S32
            | O::ConvertF16S64
            | O::ConvertF16U8
            | O::ConvertF16U16
            | O::ConvertF16U32
            | O::ConvertF16U64
            | O::ConvertF16F32
            | O::ConvertF32F16
            | O::FPAbs16
            | O::FPAdd16
            | O::FPCeil16
            | O::FPFloor16
            | O::FPFma16
            | O::FPMul16
            | O::FPNeg16
            | O::FPRoundEven16
            | O::FPSaturate16
            | O::FPClamp16
            | O::FPTrunc16
            | O::FPOrdEqual16
            | O::FPUnordEqual16
            | O::FPOrdNotEqual16
            | O::FPUnordNotEqual16
            | O::FPOrdLessThan16
            | O::FPUnordLessThan16
            | O::FPOrdGreaterThan16
            | O::FPUnordGreaterThan16
            | O::FPOrdLessThanEqual16
            | O::FPUnordLessThanEqual16
            | O::FPOrdGreaterThanEqual16
            | O::FPUnordGreaterThanEqual16
            | O::FPIsNan16
            | O::GlobalAtomicAddF16x2
            | O::GlobalAtomicMinF16x2
            | O::GlobalAtomicMaxF16x2
            | O::StorageAtomicAddF16x2
            | O::StorageAtomicMinF16x2
            | O::StorageAtomicMaxF16x2
    )
}

/// Eden's `uses_fp64` list.
const fn uses_f64(opcode: Opcode) -> bool {
    use Opcode as O;
    matches!(
        opcode,
        O::CompositeConstructF64x2
            | O::CompositeConstructF64x3
            | O::CompositeConstructF64x4
            | O::CompositeExtractF64x2
            | O::CompositeExtractF64x3
            | O::CompositeExtractF64x4
            | O::CompositeInsertF64x2
            | O::CompositeInsertF64x3
            | O::CompositeInsertF64x4
            | O::SelectF64
            | O::BitCastU64F64
            | O::BitCastF64U64
            | O::PackDouble2x32
            | O::UnpackDouble2x32
            | O::FPAbs64
            | O::FPAdd64
            | O::FPCeil64
            | O::FPFloor64
            | O::FPFma64
            | O::FPMax64
            | O::FPMin64
            | O::FPMul64
            | O::FPNeg64
            | O::FPRecip64
            | O::FPRecipSqrt64
            | O::FPRoundEven64
            | O::FPSaturate64
            | O::FPClamp64
            | O::FPTrunc64
            | O::FPOrdEqual64
            | O::FPUnordEqual64
            | O::FPOrdNotEqual64
            | O::FPUnordNotEqual64
            | O::FPOrdLessThan64
            | O::FPUnordLessThan64
            | O::FPOrdGreaterThan64
            | O::FPUnordGreaterThan64
            | O::FPOrdLessThanEqual64
            | O::FPUnordLessThanEqual64
            | O::FPOrdGreaterThanEqual64
            | O::FPUnordGreaterThanEqual64
            | O::FPIsNan64
            | O::ConvertS16F64
            | O::ConvertS32F64
            | O::ConvertS64F64
            | O::ConvertU16F64
            | O::ConvertU32F64
            | O::ConvertU64F64
            | O::ConvertF32F64
            | O::ConvertF64F32
            | O::ConvertF64S8
            | O::ConvertF64S16
            | O::ConvertF64S32
            | O::ConvertF64S64
            | O::ConvertF64U8
            | O::ConvertF64U16
            | O::ConvertF64U32
            | O::ConvertF64U64
    )
}

/// Eden's `uses_int64_bit_atomics` list.
const fn uses_i64_atomics(opcode: Opcode) -> bool {
    use Opcode as O;
    matches!(
        opcode,
        O::SharedAtomicExchange64
            | O::GlobalAtomicIAdd64
            | O::GlobalAtomicSMin64
            | O::GlobalAtomicUMin64
            | O::GlobalAtomicSMax64
            | O::GlobalAtomicUMax64
            | O::GlobalAtomicAnd64
            | O::GlobalAtomicOr64
            | O::GlobalAtomicXor64
            | O::GlobalAtomicExchange64
            | O::StorageAtomicIAdd64
            | O::StorageAtomicSMin64
            | O::StorageAtomicUMin64
            | O::StorageAtomicSMax64
            | O::StorageAtomicUMax64
            | O::StorageAtomicAnd64
            | O::StorageAtomicOr64
            | O::StorageAtomicXor64
    )
}

/// Eden's `uses_int64` list, with the four `u64` forms it misses (marked), and the global
/// loads its global-memory case also counts.
const fn uses_i64(opcode: Opcode) -> bool {
    use Opcode as O;
    matches!(
        opcode,
        O::UndefU64
        | O::LoadGlobalU8
        | O::LoadGlobalS8
        | O::LoadGlobalU16
        | O::LoadGlobalS16
        | O::LoadGlobal32
        | O::LoadGlobal64
        | O::LoadGlobal128
        | O::WriteGlobalU8
        | O::WriteGlobalS8
        | O::WriteGlobalU16
        | O::WriteGlobalS16
        | O::WriteGlobal32
        | O::WriteGlobal64
        | O::WriteGlobal128
        | O::SelectU64
        | O::BitCastU64F64
        | O::BitCastF64U64
        | O::PackUint2x32
        | O::UnpackUint2x32
        | O::IAdd64
        | O::ISub64
        | O::INeg64
        // Starquake's additions: `u64` forms Eden's list misses (module docs).
        | O::IAbs64
        | O::ConvertF16S64
        | O::ConvertF32S64
        | O::ConvertF64S64
        | O::ShiftLeftLogical64
        | O::ShiftRightLogical64
        | O::ShiftRightArithmetic64
        | O::ConvertS64F16
        | O::ConvertS64F32
        | O::ConvertS64F64
        | O::ConvertU64F16
        | O::ConvertU64F32
        | O::ConvertU64F64
        | O::ConvertU64U32
        | O::ConvertU32U64
        | O::ConvertF16U64
        | O::ConvertF32U64
        | O::ConvertF64U64
        | O::SharedAtomicExchange64
        | O::GlobalAtomicIAdd64
        | O::GlobalAtomicSMin64
        | O::GlobalAtomicUMin64
        | O::GlobalAtomicSMax64
        | O::GlobalAtomicUMax64
        | O::GlobalAtomicAnd64
        | O::GlobalAtomicOr64
        | O::GlobalAtomicXor64
        | O::GlobalAtomicExchange64
        | O::StorageAtomicIAdd64
        | O::StorageAtomicSMin64
        | O::StorageAtomicUMin64
        | O::StorageAtomicSMax64
        | O::StorageAtomicUMax64
        | O::StorageAtomicAnd64
        | O::StorageAtomicOr64
        | O::StorageAtomicXor64
        | O::StorageAtomicExchange64
    )
}

/// The global stores and atomics of Eden's global-memory case, which fall through to the
/// loads and set `uses_int64`.
const fn is_global_store_or_atomic(opcode: Opcode) -> bool {
    use Opcode as O;
    matches!(
        opcode,
        O::GlobalAtomicIAdd32
            | O::GlobalAtomicSMin32
            | O::GlobalAtomicUMin32
            | O::GlobalAtomicSMax32
            | O::GlobalAtomicUMax32
            | O::GlobalAtomicInc32
            | O::GlobalAtomicDec32
            | O::GlobalAtomicAnd32
            | O::GlobalAtomicOr32
            | O::GlobalAtomicXor32
            | O::GlobalAtomicExchange32
            | O::GlobalAtomicIAdd32x2
            | O::GlobalAtomicSMin32x2
            | O::GlobalAtomicUMin32x2
            | O::GlobalAtomicSMax32x2
            | O::GlobalAtomicUMax32x2
            | O::GlobalAtomicAnd32x2
            | O::GlobalAtomicOr32x2
            | O::GlobalAtomicXor32x2
            | O::GlobalAtomicExchange32x2
            | O::GlobalAtomicAddF32
            | O::GlobalAtomicAddF16x2
            | O::GlobalAtomicAddF32x2
            | O::GlobalAtomicMinF16x2
            | O::GlobalAtomicMinF32x2
            | O::GlobalAtomicMaxF16x2
            | O::GlobalAtomicMaxF32x2
    )
}

/// Eden's subgroup flags' lists: `uses_subgroup_invocation_id`, `uses_subgroup_mask`,
/// `uses_subgroup_vote`, `uses_subgroup_shuffles`, `uses_quad_shuffles`, `uses_fswzadd`.
const fn uses_subgroup(opcode: Opcode) -> bool {
    use Opcode as O;
    matches!(
        opcode,
        O::LaneId
            | O::SubgroupEqMask
            | O::SubgroupLtMask
            | O::SubgroupLeMask
            | O::SubgroupGtMask
            | O::SubgroupGeMask
            | O::VoteAll
            | O::VoteAny
            | O::VoteEqual
            | O::SubgroupBallot
            | O::ShuffleIndex
            | O::ShuffleUp
            | O::ShuffleDown
            | O::ShuffleButterfly
            | O::QuadBroadcast
            | O::QuadSwap
            | O::FSwizzleAdd
    )
}

/// `a` with every flag of `b` added.
const fn union(a: Requirements, b: Requirements) -> Requirements {
    Requirements {
        subgroup: a.subgroup || b.subgroup,
        f16: a.f16 || b.f16,
        f64: a.f64 || b.f64,
        i64: a.i64 || b.i64,
        i64_atomics: a.i64_atomics || b.i64_atomics,
    }
}

/// The size of one constant-buffer read of `opcode`, or `None` for other opcodes (Eden's
/// `GetElementSize`).
const fn cbuf_element_size(opcode: Opcode) -> Option<u32> {
    Some(match opcode {
        Opcode::GetCbufU8 | Opcode::GetCbufS8 => 1,
        Opcode::GetCbufU16 | Opcode::GetCbufS16 => 2,
        Opcode::GetCbufU32 | Opcode::GetCbufF32 => 4,
        Opcode::GetCbufU32x2 => 8,
        _ => return None,
    })
}

/// Grows a bank's used size to at least `size` (the larger of the two: see the module docs'
/// divergence table for where Eden assigns instead).
fn grow(slot: &mut Option<u32>, size: u32) {
    *slot = Some(slot.map_or(size, |old| old.max(size)));
}

/// `mask` with `word` set to `value` (Eden's `VaryingState::Set`).
fn set_word(mask: AttributeMask, word: u16, value: bool) -> AttributeMask {
    let mut lanes = mask.lanes();
    let lane = usize::from(word) / 64;
    let bit = u64::from(word) % 64;
    if let Some(slot) = lanes.get_mut(lane) {
        if value {
            *slot |= 1 << bit;
        } else {
            *slot &= !(1 << bit);
        }
    }
    AttributeMask::from_lanes(lanes)
}

/// Whether `word` is in `mask`.
fn has_word(mask: AttributeMask, word: u16) -> bool {
    u8::try_from(word).is_ok_and(|word| mask.contains(word))
}

/// The four flags of a component mask, in component order.
const fn flags(mask: Components<bool>) -> [bool; 4] {
    [mask.x, mask.y, mask.z, mask.w]
}

/// Each generic's interpolation (`CollectInterpolationInfo`).
fn interpolation(io: &PixelIo) -> Result<[Interpolation; GENERICS as usize], InfoError> {
    let mut out = [Interpolation::Smooth; GENERICS as usize];
    for ((generic, map), slot) in (0_u8..).zip(&io.generic).zip(&mut out) {
        let mut mode = None;
        for imap in [map.x, map.y, map.z, map.w] {
            if imap == PixelImap::Unused {
                continue;
            }
            if mode.is_some_and(|mode| mode != imap) {
                return Err(InfoError::MixedInterpolation { generic });
            }
            mode = Some(imap);
        }
        *slot = mode.map_or(Interpolation::Smooth, Interpolation::of);
    }
    Ok(out)
}

/// Whether the program reads and writes attributes at register-computed words.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Indexed {
    loads: bool,
    stores: bool,
}

/// What a pixel program writes besides attributes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct PixelOutputs {
    color: ColorOutputs,
    sample_mask: bool,
    depth: bool,
}

/// Each bank's used size, and whether a bank was a register.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ConstantBuffers {
    sizes: [Option<u32>; CBUF_BANKS as usize],
    indexed_bank: bool,
}

/// The walk's state, which [`ShaderInfo`] keeps.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Walk {
    loads: AttributeMask,
    stores: AttributeMask,
    indexed: Indexed,
    pixel: PixelOutputs,
    system_values: SystemValues,
    constant_buffers: ConstantBuffers,
    requirements: Requirements,
}

impl Walk {
    fn new() -> Self {
        Self {
            loads: AttributeMask::EMPTY,
            stores: AttributeMask::EMPTY,
            indexed: Indexed::default(),
            pixel: PixelOutputs::default(),
            system_values: SystemValues::EMPTY,
            constant_buffers: ConstantBuffers {
                sizes: [None; CBUF_BANKS as usize],
                indexed_bank: false,
            },
            requirements: Requirements::default(),
        }
    }

    fn visit(&mut self, at: Value, op: &Operation) -> Result<(), InfoError> {
        self.requirements = union(self.requirements, opcode_requirements(op.opcode));
        if let Some(value) = SystemValue::of(op.opcode) {
            self.system_values = self.system_values.with(value);
        }
        match op.opcode {
            Opcode::GetAttribute | Opcode::GetAttributeU32 | Opcode::InterpolateAtOffset => {
                let word = attribute_arg(at, op)?;
                self.loads = set_word(self.loads, word, true);
            }
            Opcode::SetAttribute => {
                let word = attribute_arg(at, op)?;
                self.stores = set_word(self.stores, word, true);
            }
            Opcode::GetAttributeIndexed => self.indexed.loads = true,
            Opcode::SetAttributeIndexed => self.indexed.stores = true,
            Opcode::SetFragColor => self.frag_color(at, op)?,
            Opcode::SetSampleMask => self.pixel.sample_mask = true,
            Opcode::SetFragDepth => self.pixel.depth = true,
            opcode => {
                if let Some(size) = cbuf_element_size(opcode) {
                    self.cbuf(at, op, size)?;
                }
            }
        }
        Ok(())
    }

    fn frag_color(&mut self, at: Value, op: &Operation) -> Result<(), InfoError> {
        let target = u32_arg(at, op, 0)?;
        let component = u32_arg(at, op, 1)?;
        let entry = usize::try_from(target)
            .ok()
            .filter(|_| component < u32::from(ATTRIBUTE_COMPONENTS))
            .and_then(|t| self.pixel.color.targets.get_mut(t))
            .ok_or(InfoError::ColorOutput {
                at,
                target,
                component,
            })?;
        *entry |= 1 << component;
        Ok(())
    }

    fn cbuf(&mut self, at: Value, op: &Operation, element: u32) -> Result<(), InfoError> {
        // Both arguments' kinds first (bank, then offset), then the bank's range, then the
        // read's end.
        let bank = cbuf_arg(at, op, 0)?;
        let offset = cbuf_arg(at, op, 1)?;
        let Some(bank) = bank else {
            // A register bank: every bank, whole (Eden's `AddRegisterIndexedLdc`, over the
            // spec's 32 banks).
            self.constant_buffers.indexed_bank = true;
            for slot in &mut self.constant_buffers.sizes {
                grow(slot, INDEXED_CBUF_SIZE);
            }
            return Ok(());
        };
        let slot = usize::try_from(bank)
            .ok()
            .and_then(|b| self.constant_buffers.sizes.get_mut(b))
            .ok_or(InfoError::ConstantBufferBank { at, bank })?;
        let size = match offset {
            Some(offset) => offset
                .checked_add(element)
                .and_then(|end| end.checked_next_multiple_of(CBUF_ALIGNMENT))
                .ok_or(InfoError::ConstantBufferOffset { at, offset })?,
            None => INDEXED_CBUF_SIZE,
        };
        grow(slot, size);
        Ok(())
    }

    /// Eden's `GatherInfoFromHeader`.
    fn fill_from_header(&mut self, header: Option<&ProgramHeader>) {
        match header.map(|h| &h.io) {
            Some(Io::Pixel(io)) if self.indexed.loads => {
                for (index, map) in (0_u16..).zip(&io.generic) {
                    let base = GENERIC_0_X + index * ATTRIBUTE_COMPONENTS;
                    for (word, imap) in (base..).zip([map.x, map.y, map.z, map.w]) {
                        self.loads = set_word(self.loads, word, imap != PixelImap::Unused);
                    }
                }
            }
            Some(Io::Vtg(io)) => {
                if self.indexed.loads {
                    self.loads = fill_vtg(self.loads, &io.input);
                }
                if self.indexed.stores {
                    self.stores = fill_vtg(self.stores, &io.output);
                }
            }
            Some(Io::Pixel(_)) | None => {}
        }
    }
}

/// Sets or clears the words Eden's `GatherInfoFromHeader` fills from one vertex map.
fn fill_vtg(mut mask: AttributeMask, map: &VtgMap) -> AttributeMask {
    for (index, generic) in (0_u16..).zip(&map.generic) {
        let base = GENERIC_0_X + index * ATTRIBUTE_COMPONENTS;
        for (word, used) in (base..).zip(flags(*generic)) {
            mask = set_word(mask, word, used);
        }
    }
    let clip = map.system_values_c.clip_distances;
    for index in 0..CLIP_DISTANCES {
        mask = set_word(mask, CLIP_DISTANCE_0 + index, clip >> index & 1 != 0);
    }
    let ab = &map.system_values_ab;
    let c = &map.system_values_c;
    let singles = [
        (PRIMITIVE_ID, ab.primitive_id),
        (LAYER, ab.rt_array_index),
        (VIEWPORT_INDEX, ab.viewport_index),
        (POINT_SIZE, ab.point_size),
        (FOG_COORDINATE, c.fog_coordinate),
        (INSTANCE_ID, c.instance_id),
        (VERTEX_ID, c.vertex_id),
    ];
    for (word, used) in singles {
        mask = set_word(mask, word, used);
    }
    for (word, used) in (POSITION_X..).zip(flags(ab.position)) {
        mask = set_word(mask, word, used);
    }
    for (word, used) in (POINT_SPRITE_S..).zip([c.point_sprite_s, c.point_sprite_t]) {
        mask = set_word(mask, word, used);
    }
    for (word, used) in
        (TESSELLATION_POINT_U..).zip([c.tessellation_point_u, c.tessellation_point_v])
    {
        mask = set_word(mask, word, used);
    }
    mask
}

/// Argument 0 as an attribute word a slot names.
fn attribute_arg(at: Value, op: &Operation) -> Result<u16, InfoError> {
    let Some(Operand::Imm(Imm::Attribute(attribute))) = op.args.first() else {
        return Err(InfoError::Operand { at, index: 0 });
    };
    if slot_of(*attribute).is_none() {
        return Err(InfoError::UnknownAttribute {
            at,
            attribute: *attribute,
        });
    }
    Ok(attribute.0)
}

/// Argument `index` of a constant-buffer read, a bank or an offset: `Some` for a 32-bit
/// immediate, `None` for a register (an instruction's value). A missing argument or an
/// immediate of another kind is [`InfoError::Operand`].
fn cbuf_arg(at: Value, op: &Operation, index: usize) -> Result<Option<u32>, InfoError> {
    match op.args.get(index) {
        Some(Operand::Imm(Imm::U32(value))) => Ok(Some(*value)),
        Some(Operand::Value(_)) => Ok(None),
        Some(Operand::Imm(_)) | None => Err(InfoError::Operand { at, index }),
    }
}

/// Argument `index` as a 32-bit immediate.
fn u32_arg(at: Value, op: &Operation, index: usize) -> Result<u32, InfoError> {
    match op.args.get(index) {
        Some(Operand::Imm(Imm::U32(value))) => Ok(*value),
        _ => Err(InfoError::Operand { at, index }),
    }
}

/// The interface table of `mask`: one entry per slot with a word in it, in word order;
/// `pixel_inputs` gives each generic its interpolation.
fn table(
    mask: AttributeMask,
    pixel_inputs: Option<&[Interpolation; GENERICS as usize]>,
) -> Vec<InterfaceEntry> {
    let mut entries: Vec<InterfaceEntry> = Vec::new();
    let words = u16::try_from(ATTRIBUTE_WORDS).unwrap_or(u16::MAX);
    for word in (0..words).filter(|&word| has_word(mask, word)) {
        // Every word in a collected mask has a slot: the walk refuses the others and the header
        // fill sets only named words.
        let Some((slot, component)) = slot_of(Attribute(word)) else {
            continue;
        };
        let bit = 1_u8 << component;
        match entries.last_mut() {
            Some(last) if last.slot == slot => last.components |= bit,
            _ => {
                let interpolation = match (slot, pixel_inputs) {
                    (Slot::Generic(index), Some(modes)) => modes.get(usize::from(index)).copied(),
                    _ => None,
                };
                entries.push(InterfaceEntry {
                    slot,
                    components: bit,
                    interpolation,
                });
            }
        }
    }
    entries
}

/// Collects the info of `function`, a program of `stage` with `header` (none for compute), in
/// the order of checks the module docs give.
pub fn collect(
    stage: GuestStage,
    header: Option<&ProgramHeader>,
    function: &Function,
) -> Result<ShaderInfo, InfoError> {
    check_header(stage, header)?;
    let modes = match header.map(|h| &h.io) {
        Some(Io::Pixel(io)) => Some(interpolation(io)?),
        _ => None,
    };
    let mut walk = Walk::new();
    for (at, inst) in function.listed() {
        if let Inst::Op(op) = inst {
            walk.visit(at, op)?;
        }
    }
    walk.fill_from_header(header);
    let inputs = table(walk.loads, modes.as_ref());
    let mut outputs = table(walk.stores, None);
    for (target, mask) in (0_u8..).zip(walk.pixel.color.targets) {
        if mask != 0 {
            outputs.push(InterfaceEntry {
                slot: Slot::ColorTarget(target),
                components: mask,
                interpolation: None,
            });
        }
    }
    for (stored, slot) in [
        (walk.pixel.sample_mask, Slot::SampleMask),
        (walk.pixel.depth, Slot::FragDepth),
    ] {
        if stored {
            outputs.push(InterfaceEntry {
                slot,
                components: 1,
                interpolation: None,
            });
        }
    }
    Ok(ShaderInfo {
        stage,
        interpolation: modes.unwrap_or([Interpolation::Smooth; GENERICS as usize]),
        inputs,
        outputs,
        walk,
    })
}
