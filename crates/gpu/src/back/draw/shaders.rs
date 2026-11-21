//! Asking for shaders (gpu-protocol spec, "Shader pipeline (#13)"; shader-pipeline, "What the GPU
//! front/back protocol and engine state (#12) gets"): the seam to the shader cache, the two
//! answering traits the back implements, the variant key, and turning a translated shader into
//! the render crate's [`ShaderDesc`].
//!
//! # The seam
//!
//! The shader cache (shader-pipeline slice 19) is not built. The draw path asks for shaders
//! through [`ShaderLookup`], with the spec's `lookup` signature, and a fake implements it in the
//! tests, returning fixed sets, pending tickets and rejections. Two things the draw path needs
//! are not in slice 9's types yet, so the seam's answer, [`SetLookup`], carries them beside the
//! [`ShaderSet`]:
//!
//! - **each shader's resources** ([`ResourceUse`]): the spec gives `HostShader` a resource table
//!   ("constant buffers by bank and range, textures, samplers, images and storage buffers by
//!   binding or bindless source, with group and binding numbers"), which slices 13 and 16 add;
//!   the seam's table has the kinds this slice binds (constant buffers, sampled textures,
//!   samplers, the driver constants), with the render crate's [`ResourceKind`] at each slot;
//! - **each shader's stamp** ([`ShaderStamp`]): the pipeline-key cache records shaders by stamp,
//!   a digest of the shader-entry key, which slice 21 defines.
//!
//! [`ShaderLookup::wait`] is also new: the spec says a pending ticket under the wait setting
//! blocks the back thread ("Translation waits block the back thread") but names no call to wait
//! with. When slice 19 lands, its `Lookup` either grows these or the seam adapts it.
//!
//! # Answering
//!
//! [`GuestPrograms`] implements [`ProgramMemory`] over any [`GpuMemory`] (the production one is
//! [`crate::memory::VmMemory`], over the address table `GpuVm`, slice G22). [`ProgramMemory`]
//! has one error, [`Unmapped`]; a [`GpuFault::Unmapped`] keeps its address, and the other faults
//! are reported as unmapped at the access's start ([`GpuFault::OutOfRange`]) or at the faulting
//! piece ([`GpuFault::Guest`]), so translation sees the program end there.
//! [`GpuFault::ShimWrite`] is a write fault and does not reach a read; it would be reported at
//! its address too.
//!
//! [`StateAnswers`] implements [`EngineState`] from the 3D mirror, the root tables and guest
//! memory, following Eden-NX2's `GraphicsEnvironment`
//! (`eden-nx2/src/video_core/shader_environment.cpp` @ 50fa86bb, GPL-3.0-or-later):
//!
//! - a stage reads the bind group of Eden's `stage_index` (vertex, both forms, 0; tessellation
//!   init 1; tessellation 2; geometry 3; pixel 4); `SET_PIPELINE_BINDING`'s group is not read,
//!   as Eden reads none (decision 10, "the one a title ran on";
//!   `tests::a_stage_reads_the_bind_group_of_its_slot_not_the_pipeline_binding`);
//! - banks 24 to 31 come from the root tables (PR #4588's `ReadCbufValue`,
//!   `eden-pr4588/src/video_core/shader_environment.cpp`);
//! - a texture handle is a header index in bits 19:0 and a sampler index in 31:20, or the whole
//!   word for both when `SET_SAMPLER_BINDING` is `VIA_HEADER_BINDING` (Eden's `TexturePair`,
//!   `eden/src/video_core/textures/texture.h`), and the header is the pool entry at that index;
//! - a texture kind carries the header's format bits whatever the question's usage: the shader
//!   crate's exchange drops them from the answer to a sampled-only question
//!   (`TextureKind::for_usage`; owner's ruling 7 of 2026-11-19), so a format change that keeps
//!   the kind does not change a sampled answer
//!   (`tests::a_format_change_keeps_a_sampled_answer_through_engine_state`).
//!
//! Where it differs from Eden, each pinned by the test named:
//!
//! - **A word is read only when all four bytes lie inside the bound size.** Eden's
//!   `ReadCbufValue` checks `offset < size` and reads four bytes; here `offset + 4 <= size`.
//!   Past the size the answer is `None`, where Eden returns 0 (the spec's answer has no zero
//!   stand-in). `tests::a_constant_buffer_word_must_fit_inside_the_bound_size`.
//! - **An unbound bank, an unmapped word or a header index past the pool's maximum answers
//!   `None`**: Eden asserts on the first and the third, and reads whatever the second gives.
//!   `tests::unanswerable_questions_answer_none`.
//!
//! # The variant key
//!
//! [`variant_key`] builds the [`VariantKey`] from engine state: each vertex attribute's type,
//! each colour output's type (Eden's `color_output_types`: integer formats signed or unsigned,
//! the rest float), dual-source blending, and the [`EmulationKey`] copied field by field from the
//! backend's [`EmulationFlags`], with the alpha function present exactly when `ALPHA_TEST` is
//! set (`tests::every_emulation_flag_reaches_the_key_field_it_mirrors`, and through a whole
//! draw `tests::every_emulation_flag_reaches_the_key_through_a_draw`). The compute fields are
//! zero for a draw.
//!
//! # The previous stage's stores
//!
//! Each stage that follows another is specialised on what the stage before it stores
//! ([`VariantKey::previous_stage_stores`]; shader-pipeline, "Pipeline-state specialisation";
//! decided by the owner, 2026-11-19, ruling 8), as Eden's `MakeRuntimeInfo` sets
//! `previous_stage_stores = previous_program->info.stores`
//! (`eden/src/video_core/renderer_vulkan/vk_pipeline_cache.cpp` @ cf654052). What a stage
//! stores is a fact of its program's translation, not of a register, so the draw path learns it
//! from a ready set: [`stored_words`] reads a translated shader's outputs, and
//! [`previous_stage_stores`] gives each following stage of the requests the stores of the
//! request before it. Like the swizzles below, the draw path looks a set up first with the
//! stores the previous answer for the same requests gave (none for a set never seen), and once
//! more when the ready set's differ (`tests::the_previous_stages_stores_reach_the_key`).
//!
//! Where this differs from Eden, each pinned by the test named:
//!
//! - **Whole slots, not components.** A shader's outputs name a whole slot (the position, a
//!   generic), so every word of a stored slot is in the mask, where Eden's `info.stores` has the
//!   components the program writes. Eden's SPIR-V backend reads the mask a slot at a time
//!   (`previous_stage_stores.Generic(index)`,
//!   `eden/src/shader_recompiler/backend/spirv/spirv_emit_context.cpp` @ cf654052), so the key
//!   loses nothing it uses; its GLSL backend reads it a component at a time
//!   (`Generic(index, element)`, `backend/glsl/emit_glsl_context_get_set.cpp`).
//!   `tests::stored_words_are_whole_slots`.
//! - **A previous stage with no shader in the set stores nothing.** The set's shaders are a
//!   vertex and a pixel shader ([`starquake_shader::host::HostStage`] has no tessellation or
//!   geometry stage yet), so a following stage whose previous request is a tessellation or
//!   geometry stage gets the empty mask. A **guess** until those stages are translated.
//!   `tests::previous_stage_stores_follow_the_request_order`.
//! - **Learned from a first lookup's ready set only**, as the swizzles are: a set first given
//!   by a wait is drawn with the stores its key carried (none for a set never seen; Eden's
//!   SPIR-V backend declares no generic input the previous stage does not store), and the next
//!   draw looks it up with the learned ones.
//!   `tests::stores_are_not_learned_from_a_waited_set`.
//!
//! # The vertex outputs' interpolation
//!
//! The vertex stage's outputs take the paired pixel program's interpolation modes
//! ([`VariantKey::output_interpolation`]; shader-pipeline, "Pipeline-state specialisation", the
//! named exception; decided by the owner, 2026-11-20, ruling 2): wgpu refuses a pipeline whose
//! vertex outputs and fragment inputs differ in interpolation, and only the pixel program's
//! header gives the mode. Unlike the stores above, the mode is a fact of the pixel program's
//! header, not of its translation, so the draw path reads it before the first lookup: each draw
//! reads the 128-byte header ([`starquake_shader::sph::HEADER_BYTES`]) at the pixel request's address and parses it
//! with the shader crate's parser, reduced per generic attribute by
//! [`OutputInterpolation::from_pixel_io`] ([`pixel_interpolation`]). [`PixelModes`] remembers the
//! last header and its modes, so a draw whose header bytes are unchanged parses nothing; the
//! bytes are read and compared every draw, so a program rewritten in place is seen at the next
//! draw (`tests::the_pixel_programs_interpolation_reaches_the_key`). Only the last header is
//! remembered: a header that differs from it is parsed again, whether new or seen before, and
//! a draw with no pixel request or an unmapped header leaves the memo as it was
//! (`tests::pixel_modes_remember_only_the_last_header`). The memo saves parsing only, so no test
//! sees it: dropping it (never storing, or a fresh [`PixelModes`] per draw) gives the same keys.
//!
//! The key holds [`OutputInterpolation::SMOOTH`] (every generic smooth, Eden's value-initialised
//! `Info::interpolation`) when:
//!
//! - the requests have no pixel program (rasterisation off, step 3 of the draw path);
//! - the header is not mapped;
//! - the bytes are not a pixel header the shader crate reads (a `HeaderError`, or a type 1
//!   header in the pixel slot);
//! - the header gives a generic two modes ([`starquake_shader::variant::KeyError::MixedInterpolation`]).
//!
//! The draw still looks the set up rather than failing: the shader crate refuses the same pixel
//! programs when it translates them (an unmapped or unparsable header in
//! `program::translate_program`, which parses the header first; a type 1 header in the pixel
//! slot or a generic with two modes in `info::collect`, `HeaderStage` and `MixedInterpolation`,
//! once production translation runs collect info), and the draw path skips a rejected set once
//! per set, as for any rejection. This keeps one place, the shader cache, that refuses
//! programs. `tests::a_pixel_header_the_key_cannot_use_leaves_every_generic_smooth`.
//!
//! The flipped viewport ([`viewport_flip`], [`constants_flip`]): the backend's `emulation()`
//! reads how the first viewport handed to it flips, and sets `FLIPPED_VIEWPORT` when the host
//! cannot set that viewport; the driver constants then carry the same flip for the vertex
//! shader, and carry none without the flag, when the shader does not read them and the host
//! sets the viewport as it is (render-backend, "Fixed-function state the host lacks"; owner's
//! ruling 28 of 2026-11-19). `tests::each_viewport_flip_reaches_the_flag_and_the_constants`.
//!
//! The texture swizzles go in only when `Caps::view_swizzle` is false, keyed by binding order,
//! which only a set's translation knows (slice 9's owner question). This slice therefore looks
//! a set up first with the swizzles of the textures the previous answer for the same requests
//! bound (none for a set never seen), and, if the set it gets binds textures whose swizzles
//! differ, once more with those
//! (`tests::swizzles_reach_the_key_in_binding_order_only_without_view_swizzle`). A
//! **provisional** answer to that question, raised with the owner. The binding order is each
//! shader's texture uses in set order, then in resource-table order.

use std::collections::HashMap;
use std::num::NonZeroU32;
use std::sync::Arc;

use starquake_render::caps::GUEST_COLOR_TARGETS;
use starquake_render::command::Viewport;
use starquake_render::emulation::ViewportFlip;
use starquake_render::format::NumericType;
use starquake_render::pipeline::EmulationFlags;
use starquake_render::shader::{
    InterfaceVariable, Interpolation, ResourceBinding, ResourceKind, ScalarKind, ShaderDesc,
    ShaderInterface, ShaderRequirements, ShaderSource, ShaderStage,
};
use starquake_render::{Backend, FenceValue, PixelFormat, ResourceId, ShaderId};
use starquake_shader::exchange::{SampledType, TextureFormat, TextureKind};
use starquake_shader::host::{
    GuestSlot, HostCode, HostShader, HostStage, InterfaceSlot, Requirements, SlotBinding, SlotType,
};
use starquake_shader::ir::{ATTRIBUTE_COMPONENTS, GENERIC_0_X, POSITION_X};
use starquake_shader::lookup::{
    EngineState, GuestStage, ProgramMemory, ProgramRequest, Rejection, ShaderSet, Ticket, Unmapped,
    VariantKey,
};
use starquake_shader::lower::GENERICS;
use starquake_shader::sph::{HEADER_BYTES as SPH_BYTES, Io, ProgramHeader};
use starquake_shader::translate::COLOR_TARGETS as SHADER_COLOR_TARGETS;
use starquake_shader::variant::{
    AttributeMask, AttributeType, ColorType, CompareFunction as KeyCompare, EmulationKey,
    FollowingStage, OutputInterpolation, PreviousStores, Swizzle, SwizzleSource, TextureSwizzles,
    VERTEX_ATTRIBUTES as KEY_ATTRIBUTES,
};

use super::DrawError;
use super::root::{FIRST_ROOT_BANK, RootTables};
use crate::back::pipeline::ShaderStamp;
use crate::back::state::Mirror;
use crate::back::texture::{HEADER_WORDS, TextureHeader, pixel_format};
use crate::classes::{clc797, clc797tex as tex};
use crate::memory::{GpuFault, GpuMemory};

/// Where a resource is bound in a shader's module.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Slot {
    /// Bind group.
    pub group: u32,
    /// Binding in the group.
    pub binding: u32,
}

/// Where a texture or sampler handle is read from: a word of a constant buffer (bound textures
/// read it at an offset of the bound texture bank, bindless ones wherever the program loaded
/// it; spec, "Passes needed for correctness" item 3).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct HandleSource {
    /// The constant buffer bank.
    pub bank: u8,
    /// The handle word's byte offset in the bank.
    pub offset: u32,
}

/// What a resource binding of a shader reads (module docs, "The seam").
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ResourceSource {
    /// Bytes `0..size` of constant buffer `bank`.
    ConstantBuffer {
        /// The bank, 0 to 31.
        bank: u8,
        /// The bytes the shader reads.
        size: u32,
    },
    /// The texture a handle names.
    Texture(HandleSource),
    /// The sampler a handle names.
    Sampler(HandleSource),
    /// The driver constants ([`starquake_shader::driver_constants`]).
    DriverConstants,
}

/// One resource a shader binds (module docs, "The seam").
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ResourceUse {
    /// Where the module binds it.
    pub slot: Slot,
    /// What the module declares there.
    pub kind: ResourceKind,
    /// The module's binding-array size there: `None` for one resource, `Some(n)` for a binding
    /// array of `n` ([`ResourceBinding::count`], which [`shader_desc`] copies it to; decided by
    /// the owner, 2026-11-19, ruling 20). The draw and compute paths bind one resource per row
    /// and no lookup answers a row with a count yet: binding arrays are bound with the slices
    /// that add them to the shader crate's resource table (13 and 16).
    pub count: Option<NonZeroU32>,
    /// What fills it.
    pub source: ResourceSource,
}

/// What the seam knows of one shader of a set beside the [`HostShader`] (module docs).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StageInfo {
    /// The shader's stamp.
    pub stamp: ShaderStamp,
    /// Its resources.
    pub resources: Vec<ResourceUse>,
}

/// A ready set: the shaders and, for each in the same order, what [`StageInfo`] adds.
#[derive(Clone, Debug)]
pub struct ReadySet {
    /// The shaders.
    pub shaders: Arc<ShaderSet>,
    /// One per shader of `shaders`, in its order.
    pub stages: Vec<StageInfo>,
}

/// What a lookup gives (slice 9's `Lookup`, with [`ReadySet`] for its ready arm).
#[derive(Clone, Debug)]
pub enum SetLookup {
    /// The shaders are ready.
    Ready(ReadySet),
    /// The shaders are being translated.
    Pending(Ticket),
    /// A program was rejected.
    Rejected(Arc<Rejection>),
}

/// The shader cache as the draw path sees it (module docs, "The seam").
pub trait ShaderLookup {
    /// The shaders of `requests` specialised on `variant`; never blocks and never translates
    /// (the spec's `ShaderCache::lookup`).
    fn lookup(
        &mut self,
        requests: &[ProgramRequest],
        variant: &VariantKey,
        memory: &dyn ProgramMemory,
        state: &dyn EngineState,
    ) -> SetLookup;

    /// Blocks until `ticket`'s translation has finished, and gives its result. A `Pending`
    /// answer is taken as a translation that never finished: the draw is skipped.
    fn wait(&mut self, ticket: Ticket) -> SetLookup;
}

/// Guest memory as translation reads it (module docs, "Answering").
pub struct GuestPrograms<'a, M: GpuMemory + ?Sized>(pub &'a M);

impl<M: GpuMemory + ?Sized> ProgramMemory for GuestPrograms<'_, M> {
    fn read(&self, address: u64, out: &mut [u8]) -> Result<(), Unmapped> {
        self.0.read(address, out).map_err(|fault| Unmapped {
            address: match fault {
                GpuFault::Unmapped { va }
                | GpuFault::OutOfRange { va, .. }
                | GpuFault::Guest { va, .. }
                | GpuFault::ShimWrite { va } => va,
            },
        })
    }
}

/// The bind group a stage reads (Eden-NX2's `stage_index`); `None` for compute, which has no
/// bind groups.
pub const fn bind_group_of(stage: GuestStage) -> Option<u32> {
    match stage {
        GuestStage::VertexCullBeforeFetch | GuestStage::Vertex => Some(0),
        GuestStage::TessellationInit => Some(1),
        GuestStage::Tessellation => Some(2),
        GuestStage::Geometry => Some(3),
        GuestStage::Pixel => Some(4),
        GuestStage::Compute => None,
    }
}

/// Engine state as translation asks about it (module docs, "Answering").
pub struct StateAnswers<'a, M: GpuMemory + ?Sized> {
    /// The 3D mirror.
    pub mirror: &'a Mirror,
    /// The root tables.
    pub roots: &'a RootTables,
    /// Guest memory.
    pub memory: &'a M,
}

impl<M: GpuMemory + ?Sized> StateAnswers<'_, M> {
    /// The bound constant buffer `bank` of `stage`'s bind group: its address and size, or `None`
    /// when not bound.
    pub fn constant_buffer(&self, stage: GuestStage, bank: u8) -> Option<(u64, u32)> {
        let group = bind_group_of(stage)?;
        let binding = self
            .mirror
            .three_d()
            .ok()?
            .bind_group(group)
            .ok()?
            .constant_buffer(u32::from(bank))
            .ok()?;
        binding.valid.then_some((binding.address, binding.size))
    }

    /// The texture header index and sampler index a raw handle names (Eden's `TexturePair`).
    pub fn handle_pair(&self, raw: u32) -> (u32, u32) {
        let via_header = self
            .mirror
            .reg(&clc797::SET_SAMPLER_BINDING, &[])
            .and_then(|reg| reg.field(clc797::SET_SAMPLER_BINDING_V))
            .is_ok_and(|value| value == clc797::SET_SAMPLER_BINDING_V_VIA_HEADER_BINDING);
        if via_header {
            (raw, raw)
        } else {
            (raw & HEADER_INDEX_MASK, raw >> SAMPLER_INDEX_SHIFT)
        }
    }

    /// The words of texture header `index`, or `None` past the pool's maximum or unmapped.
    pub fn header_words(&self, index: u32) -> Option<[u32; HEADER_WORDS]> {
        let pools = self.mirror.three_d().ok()?.tex_pools();
        if index > pools.header_maximum_index().ok()? {
            return None;
        }
        let address = pools
            .header_address()
            .ok()?
            .checked_add(u64::from(index).checked_mul(HEADER_BYTES)?)?;
        read_words::<HEADER_WORDS, M>(self.memory, address).ok()
    }
}

/// Bits 19:0 of a handle: the texture header index (Eden's `tic_id`).
pub const HEADER_INDEX_MASK: u32 = (1 << 20) - 1;

/// Bit 20 of a handle: where the sampler index starts (Eden's `tsc_id`, 31:20).
pub const SAMPLER_INDEX_SHIFT: u32 = 20;

/// Bytes in one texture header.
pub const HEADER_BYTES: u64 = HEADER_WORDS as u64 * 4;

/// Reads `N` little-endian words at `address`.
pub(super) fn read_words<const N: usize, M: GpuMemory + ?Sized>(
    memory: &M,
    address: u64,
) -> Result<[u32; N], GpuFault> {
    let mut bytes = [0u8; 4];
    let mut words = [0u32; N];
    for (i, word) in words.iter_mut().enumerate() {
        let offset = u64::try_from(i * 4).unwrap_or(u64::MAX);
        let at = address.checked_add(offset).ok_or(GpuFault::OutOfRange {
            va: address,
            len: offset,
        })?;
        memory.read(at, &mut bytes)?;
        *word = u32::from_le_bytes(bytes);
    }
    Ok(words)
}

impl<M: GpuMemory + ?Sized> EngineState for StateAnswers<'_, M> {
    fn cbuf_word(&self, stage: GuestStage, bank: u8, offset: u32) -> Option<u32> {
        if bank >= FIRST_ROOT_BANK {
            return self.roots.word(bank, offset);
        }
        let (address, size) = self.constant_buffer(stage, bank)?;
        let end = offset.checked_add(4)?;
        if end > size {
            return None;
        }
        let [word] =
            read_words::<1, M>(self.memory, address.checked_add(u64::from(offset))?).ok()?;
        Some(word)
    }

    fn texture_kind(&self, _stage: GuestStage, handle: u32) -> Option<TextureKind> {
        let (index, _) = self.handle_pair(handle);
        texture_kind(&self.header_words(index)?)
    }
}

/// What a texture header points to, or `None` for a header whose format is unknown or whose
/// layout does not decode.
pub fn texture_kind(words: &[u32; HEADER_WORDS]) -> Option<TextureKind> {
    let get = |field: crate::classes::MwField| field.get(words);
    let buffer =
        get(tex::TEXHEAD_BL_HEADER_VERSION)? == tex::TEXHEAD_BL_HEADER_VERSION_SELECT_ONE_D_BUFFER;
    let types = [
        get(tex::TEXHEAD_BL_R_DATA_TYPE)?,
        get(tex::TEXHEAD_BL_G_DATA_TYPE)?,
        get(tex::TEXHEAD_BL_B_DATA_TYPE)?,
        get(tex::TEXHEAD_BL_A_DATA_TYPE)?,
    ];
    // A buffer header has no sRGB bit (`TEXHEAD_1D_*` defines none).
    let srgb = !buffer && get(tex::TEXHEAD_BL_S_R_G_B_CONVERSION)? != 0;
    let format = pixel_format(get(tex::TEXHEAD_BL_COMPONENTS)?, types, srgb)?;
    let multisampled = if buffer {
        false
    } else {
        TextureHeader::decode(words).ok()?.image.samples != starquake_layout::SampleLayout::S1x1
    };
    Some(TextureKind {
        sampled: match format.numeric() {
            NumericType::Uint => SampledType::UnsignedInt,
            NumericType::Sint => SampledType::SignedInt,
            NumericType::Unorm | NumericType::Snorm | NumericType::Float => SampledType::Float,
        },
        depth: format.has_depth(),
        multisampled,
        buffer,
        format: TextureFormat::from_texhead_word0(words[0]),
    })
}

/// A header's swizzle as the variant key's sources: `X_SOURCE` to `W_SOURCE`, whose codes are
/// the key's ([`SwizzleSource::from_code`]); `None` for an undefined code.
pub fn header_swizzle(words: &[u32; HEADER_WORDS]) -> Option<Swizzle> {
    let source = |field: crate::classes::MwField| {
        SwizzleSource::from_code(u8::try_from(field.get(words)?).ok()?)
    };
    Some(Swizzle {
        x: source(tex::TEXHEAD_BL_X_SOURCE)?,
        y: source(tex::TEXHEAD_BL_Y_SOURCE)?,
        z: source(tex::TEXHEAD_BL_Z_SOURCE)?,
        w: source(tex::TEXHEAD_BL_W_SOURCE)?,
    })
}

/// The key's comparison for a render-crate comparison (both are Eden's eight, in one order).
pub const fn key_compare(compare: starquake_render::resource::CompareFunction) -> KeyCompare {
    use starquake_render::resource::CompareFunction as C;
    match compare {
        C::Never => KeyCompare::Never,
        C::Less => KeyCompare::Less,
        C::Equal => KeyCompare::Equal,
        C::LessEqual => KeyCompare::LessThanEqual,
        C::Greater => KeyCompare::Greater,
        C::NotEqual => KeyCompare::NotEqual,
        C::GreaterEqual => KeyCompare::GreaterThanEqual,
        C::Always => KeyCompare::Always,
    }
}

/// The key's mirror of the backend's emulation flags; `alpha` is the alpha test's function
/// (`None` while the test is off), copied in only when `ALPHA_TEST` is set. The backend's
/// `emulation()` sets `ALPHA_TEST` only for `Some`, so the key's function is present exactly
/// when the flag is.
pub fn emulation_key(
    flags: EmulationFlags,
    alpha: Option<starquake_render::resource::CompareFunction>,
) -> EmulationKey {
    EmulationKey {
        alpha_test: alpha
            .filter(|_| flags.contains(EmulationFlags::ALPHA_TEST))
            .map(key_compare),
        depth_range_rewrite: flags.contains(EmulationFlags::DEPTH_RANGE_REWRITE),
        primitive_restart_rewrite: flags.contains(EmulationFlags::PRIMITIVE_RESTART_REWRITE),
        attribute_fetch: flags.contains(EmulationFlags::ATTRIBUTE_FETCH),
        typed_outputs: flags.contains(EmulationFlags::TYPED_OUTPUTS),
        flipped_viewport: flags.contains(EmulationFlags::FLIPPED_VIEWPORT),
    }
}

/// How the first of `viewports` (those handed to the backend) flips: what `emulation()` reads.
/// None for no viewport, which the backend refuses anyway.
pub fn viewport_flip(viewports: &[Viewport]) -> ViewportFlip {
    viewports
        .first()
        .map_or(ViewportFlip::NONE, ViewportFlip::of)
}

/// The flip the driver constants carry: `flip` when `flags` has `FLIPPED_VIEWPORT`, else none.
pub fn constants_flip(flags: EmulationFlags, flip: ViewportFlip) -> ViewportFlip {
    if flags.contains(EmulationFlags::FLIPPED_VIEWPORT) {
        flip
    } else {
        ViewportFlip::NONE
    }
}

/// A colour output's type from its target's format (Eden's `color_output_types`).
pub fn color_type(format: Option<PixelFormat>) -> ColorType {
    match format {
        Some(format) if format.is_signed_integer() => ColorType::SignedInt,
        Some(format) if format.is_integer() => ColorType::UnsignedInt,
        _ => ColorType::Float,
    }
}

/// The attribute words a translated shader stores, from its outputs (module docs, "The previous
/// stage's stores"): every word of the position (`POSITION_X` to `POSITION_W`) and of each
/// generic `n` below [`GENERICS`] (`GENERIC_0_X + 4n` to `+ 3`). Colour targets and the lane id
/// are not attribute words; a generic at or past [`GENERICS`] names none.
pub fn stored_words(outputs: &[InterfaceSlot]) -> AttributeMask {
    outputs.iter().fold(AttributeMask::EMPTY, |mask, slot| {
        let first = match slot.guest {
            GuestSlot::Position => Some(POSITION_X),
            GuestSlot::Generic(n) if u16::from(n) < GENERICS => {
                GENERIC_0_X.checked_add(u16::from(n).saturating_mul(ATTRIBUTE_COMPONENTS))
            }
            GuestSlot::Generic(_) | GuestSlot::RenderTarget(_) | GuestSlot::LaneId => None,
        };
        let Some(first) = first else {
            return mask;
        };
        (0..ATTRIBUTE_COMPONENTS)
            .filter_map(|component| u8::try_from(first.checked_add(component)?).ok())
            .fold(mask, AttributeMask::with)
    })
}

/// The previous stages' stores of `requests` (module docs, "The previous stage's stores"): each
/// following stage gets `stores_of` the request before it, in request order (pipeline order),
/// or the empty mask when that stage's stores are unknown. A stage not among the requests keeps
/// the empty mask.
pub fn previous_stage_stores(
    requests: &[ProgramRequest],
    stores_of: impl Fn(GuestStage) -> Option<AttributeMask>,
) -> PreviousStores {
    requests.iter().zip(requests.iter().skip(1)).fold(
        PreviousStores::EMPTY,
        |stores, (previous, request)| match FollowingStage::of(request.stage) {
            Some(stage) => stores.with(
                stage,
                stores_of(previous.stage).unwrap_or(AttributeMask::EMPTY),
            ),
            None => stores,
        },
    )
}

/// The vertex outputs' interpolation that a pixel program's header bytes give (module docs,
/// "The vertex outputs' interpolation"): the header's modes, or
/// [`OutputInterpolation::SMOOTH`] when the bytes are not a pixel header the shader crate reads
/// or give one generic two modes.
pub fn pixel_interpolation(header: &[u8; SPH_BYTES]) -> OutputInterpolation {
    match ProgramHeader::from_program(header) {
        Ok((
            ProgramHeader {
                io: Io::Pixel(io), ..
            },
            _,
        )) => OutputInterpolation::from_pixel_io(&io).unwrap_or(OutputInterpolation::SMOOTH),
        _ => OutputInterpolation::SMOOTH,
    }
}

/// Reads the paired pixel program's header for each draw and remembers the last header's
/// modes, so a draw whose pixel header is unchanged parses nothing (module docs, "The vertex
/// outputs' interpolation").
#[derive(Debug, Default)]
pub struct PixelModes {
    /// The last header parsed and the modes it gave.
    last: Option<([u8; SPH_BYTES], OutputInterpolation)>,
}

impl PixelModes {
    /// The vertex outputs' interpolation for `requests`: [`pixel_interpolation`] of the pixel
    /// request's header as `programs` holds it now, or [`OutputInterpolation::SMOOTH`] when
    /// there is no pixel request or its header is not mapped.
    pub fn read<P: ProgramMemory + ?Sized>(
        &mut self,
        requests: &[ProgramRequest],
        programs: &P,
    ) -> OutputInterpolation {
        let Some(pixel) = requests.iter().find(|r| r.stage == GuestStage::Pixel) else {
            return OutputInterpolation::SMOOTH;
        };
        let mut header = [0; SPH_BYTES];
        if programs.read(pixel.address, &mut header).is_err() {
            return OutputInterpolation::SMOOTH;
        }
        if let Some((last, modes)) = &self.last
            && *last == header
        {
            return *modes;
        }
        let modes = pixel_interpolation(&header);
        self.last = Some((header, modes));
        modes
    }
}

/// The variant key of a draw (module docs, "The variant key").
pub fn variant_key(
    attributes: &[AttributeType; KEY_ATTRIBUTES],
    targets: &[Option<PixelFormat>; GUEST_COLOR_TARGETS],
    dual_source_blend: bool,
    emulation: EmulationKey,
    previous_stage_stores: PreviousStores,
    output_interpolation: OutputInterpolation,
    texture_swizzles: TextureSwizzles,
) -> VariantKey {
    let mut color_targets = [ColorType::Float; SHADER_COLOR_TARGETS];
    for (slot, format) in color_targets.iter_mut().zip(targets) {
        *slot = color_type(*format);
    }
    VariantKey {
        vertex_attributes: *attributes,
        color_targets,
        dual_source_blend,
        previous_stage_stores,
        workgroup_size: [0; 3],
        shared_memory_size: 0,
        emulation,
        output_interpolation,
        texture_swizzles,
    }
}

/// The render crate's stage of a host shader.
pub const fn host_stage(stage: HostStage) -> ShaderStage {
    match stage {
        HostStage::Vertex => ShaderStage::Vertex,
        HostStage::Fragment => ShaderStage::Fragment,
        HostStage::Compute => ShaderStage::Compute,
    }
}

/// The interface variable of one host slot bound at a location; `None` for a built-in (the
/// render crate's interface lists locations only). Integer slots are flat, float slots
/// perspective: the shader crate's slot has no interpolation yet, so this is a **guess** until
/// slice 13 adds one (wgpu requires flat for integers).
fn variable(binding: SlotBinding, ty: SlotType) -> Option<InterfaceVariable> {
    let SlotBinding::Location(location) = binding else {
        return None;
    };
    let (scalar, components, interpolation) = match ty {
        SlotType::F32x4 => (ScalarKind::Float, 4, Interpolation::Perspective),
        SlotType::U32 => (ScalarKind::Uint, 1, Interpolation::Flat),
    };
    Some(InterfaceVariable {
        location,
        scalar,
        components,
        interpolation,
    })
}

/// The render crate's requirements for a translated shader's, field by field (shader-pipeline,
/// "What the render backend (#14) gets": the interface "also carries `requirements`"; decided
/// by the owner, 2026-11-19, ruling 20).
pub const fn requirements(requirements: Requirements) -> ShaderRequirements {
    let Requirements {
        subgroup,
        // Not passed on yet: #473 adds the field to the render crate. A shader that sets it
        // also sets `subgroup`, which the render backend checks the size against meanwhile.
        subgroup_size_32: _,
        f16,
        f64,
        i64,
        i64_atomics,
    } = requirements;
    ShaderRequirements {
        subgroup,
        f16,
        f64,
        i64,
        i64_atomics,
    }
}

/// The one conversion from a translated shader to the render crate's description (render-backend
/// spec, "Interfaces": "that conversion is one function (G18), which the replay's
/// `ShaderResolver` also calls"): the naga module as [`ShaderSource::Naga`], the slots at
/// locations as inputs and outputs, `resources` as the resource table with each row's array
/// count, and the shader's requirements ([`requirements`]).
pub fn shader_desc(shader: &HostShader, resources: &[ResourceUse]) -> ShaderDesc {
    let HostCode::Naga { module, info } = &shader.code;
    let slots = |list: &[starquake_shader::host::InterfaceSlot]| {
        list.iter()
            .filter_map(|slot| variable(slot.binding, slot.ty))
            .collect()
    };
    ShaderDesc {
        stage: host_stage(shader.stage),
        entry: shader.entry.clone(),
        source: ShaderSource::Naga {
            module: module.clone(),
            info: info.clone(),
        },
        interface: ShaderInterface {
            resources: resources
                .iter()
                .map(|r| ResourceBinding {
                    group: r.slot.group,
                    binding: r.slot.binding,
                    kind: r.kind,
                    count: r.count,
                })
                .collect(),
            inputs: slots(&shader.inputs),
            outputs: slots(&shader.outputs),
            requirements: requirements(shader.requirements),
        },
        origin: None,
    }
}

/// The most host shaders [`HostShaders`] makes (a **guess** bounding what a guest can make the
/// host hold; the pipeline-key cache's cap, `MAX_PIPELINES`, is the same).
pub const MAX_HOST_SHADERS: usize = crate::back::pipeline::MAX_PIPELINES;

/// One host shader per stamp, made through the backend on first use and kept as long as the
/// table (as the pipeline-key cache keeps its pipelines).
pub struct HostShaders {
    backend: Arc<dyn Backend>,
    by_stamp: HashMap<ShaderStamp, ShaderId>,
    limit: usize,
}

impl core::fmt::Debug for HostShaders {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("HostShaders")
            .field("len", &self.by_stamp.len())
            .finish_non_exhaustive()
    }
}

impl HostShaders {
    /// An empty table making shaders through `backend`.
    pub fn new(backend: Arc<dyn Backend>) -> Self {
        Self::with_limit(backend, MAX_HOST_SHADERS)
    }

    pub(super) fn with_limit(backend: Arc<dyn Backend>, limit: usize) -> Self {
        Self {
            backend,
            by_stamp: HashMap::new(),
            limit,
        }
    }

    /// How many host shaders the table holds.
    pub fn len(&self) -> usize {
        self.by_stamp.len()
    }

    /// Whether it holds none.
    pub fn is_empty(&self) -> bool {
        self.by_stamp.is_empty()
    }

    /// The host shader for `shader`, stamped `info.stamp`, made on first use.
    pub fn get(&mut self, shader: &HostShader, info: &StageInfo) -> Result<ShaderId, DrawError> {
        self.get_as(shader, info, host_stage(shader.stage))
    }

    /// [`HostShaders::get`], made as a shader of `stage` whatever `shader.stage` says: the
    /// executor's compute path (slice G19) makes compute shaders this way, since
    /// [`HostStage`] has no compute stage yet.
    pub fn get_as(
        &mut self,
        shader: &HostShader,
        info: &StageInfo,
        stage: ShaderStage,
    ) -> Result<ShaderId, DrawError> {
        if let Some(&id) = self.by_stamp.get(&info.stamp) {
            return Ok(id);
        }
        if self.by_stamp.len() >= self.limit {
            return Err(DrawError::ShadersFull { limit: self.limit });
        }
        let mut desc = shader_desc(shader, &info.resources);
        desc.stage = stage;
        let id = self
            .backend
            .create_shader(desc)
            .map_err(DrawError::Render)?;
        self.by_stamp.insert(info.stamp, id);
        Ok(id)
    }

    /// Destroys every host shader the table made, after `fence`, in id order, and empties
    /// it (the draw path's teardown; `Draws`' `Drop`).
    pub fn destroy_all(&mut self, fence: FenceValue) {
        let mut ids: Vec<_> = self.by_stamp.drain().map(|(_, id)| id).collect();
        ids.sort_unstable();
        for id in ids {
            self.backend.destroy(ResourceId::Shader(id), fence);
        }
    }
}
