//! The driver constants: the small per-draw uniform block a translated shader reads numbers
//! from (`docs/specs/shader-pipeline.md`, "Pipeline-state specialisation": "Values go in driver
//! constants, not in the key"; owner's decision 16; port slice 9).
//!
//! The shader crate defines and versions the layout; the GPU crate fills it each draw (slice
//! G18) with [`DriverConstants::encode`], and the lowering (slice 17) reads each value at its
//! [`DriverConstant::offset`]. Layout version [`DRIVER_CONSTANTS_VERSION`]:
//!
//! | Offset | Value | Type | Reference |
//! |---|---|---|---|
//! | 0 | alpha-test reference | `f32` | Eden `RuntimeInfo::alpha_test_reference`, the register's bits read as a float (`eden/src/video_core/renderer_vulkan/vk_pipeline_cache.cpp` @ cf654052: `std::bit_cast<float>(key.state.alpha_test_ref)`) |
//! | 4 | viewport flip: -1.0 when y is negated, else 1.0 | `f32` | Eden `EmitYDirection`, `y_negate ? -1.0f : 1.0f` (`eden/src/shader_recompiler/backend/spirv/emit_spirv_context_get_set.cpp`) |
//! | 8 | render-area width | `f32` | Eden-NX2 `RenderAreaLayout`, floats (`eden-nx2/src/shader_recompiler/backend/spirv/emit_spirv.h` @ 50fa86bb), read by its bank-24 blit pass (row 65; `ir_opt/position_pass.cpp`, `BlitConstantsPass`) |
//! | 12 | render-area height | `f32` | same |
//! | 16 | base vertex | `u32` | Eden reads `BaseVertex` as an unsigned word (`emit_spirv_context_get_set.cpp`) |
//! | 20 | base instance | `u32` | Eden reads `BaseInstance` as an unsigned word (same file) |
//! | 24 | position y scale: -1.0 when the guest viewport flips y, else 1.0 | `f32` | starquake's own (below) |
//! | 28 | position z scale: -1.0 when the guest viewport's depth range is reversed, else 1.0 | `f32` | starquake's own (below) |
//! | 32 | position z offset, in units of w: 1.0 when the depth range is reversed and clip-space z covers 0..w, else 0.0 | `f32` | starquake's own (below) |
//! | 36 | padding, zero | | |
//!
//! Every value is a 32-bit little-endian word at a 4-byte-aligned offset, and the block is
//! [`DRIVER_CONSTANTS_SIZE`] bytes. Version 2 added the three position words. That the block is padded to a multiple of 16 bytes, so it
//! can be bound as a uniform buffer on every host, is a **guess**, settled when slice 17 lowers
//! reads of it and wgpu accepts the module. The order of the values is starquake's own; changing
//! it, or any value's type, means bumping [`DRIVER_CONSTANTS_VERSION`]. Pinned by
//! `tests::golden_bytes` and `tests::offsets_are_the_tables`.
//!
//! # Where this differs from Eden
//!
//! Eden is cited above for two of the values, but Eden does not read either per draw: it bakes
//! both into the compiled shader as constants from `RuntimeInfo`, so each distinct value is
//! another pipeline. Starquake reads both from this block each draw, under owner's decision 16
//! (numbers go in driver constants, not in the key).
//!
//! - **Alpha-test reference**: Eden compiles `RuntimeInfo::alpha_test_reference` in as a
//!   constant (`eden/src/shader_recompiler/backend/spirv/emit_spirv_special.cpp` @ cf654052:
//!   `ctx.Const(ctx.runtime_info.alpha_test_reference)`), set from the pipeline key's
//!   `alpha_test_ref` (`vk_pipeline_cache.cpp`). Here it is [`DriverConstant::AlphaReference`],
//!   at its offset in the block. Pinned by `tests::golden_bytes`.
//! - **Viewport flip**: Eden's `EmitYDirection` returns
//!   `ctx.Const(ctx.runtime_info.y_negate ? -1.0f : 1.0f)`, a constant compiled in from
//!   `RuntimeInfo::y_negate`, set from the pipeline key's `y_negate` (`vk_pipeline_cache.cpp`).
//!   Here it is [`DriverConstant::ViewportFlip`], at its offset in the block. Pinned by
//!   `tests::golden_bytes`.
//!
//! # The flipped viewport
//!
//! wgpu cannot set a viewport with a negative height or a reversed depth range, so under the
//! render backend's `FLIPPED_VIEWPORT` emulation flag (mirrored as
//! [`crate::variant::EmulationKey::flipped_viewport`]) the backend sets the host viewport
//! positive and the vertex shader flips instead (render-backend, "Fixed-function state the host
//! lacks"; owner's ruling 28 of 2026-11-19; the render crate's `emulation` module docs give the
//! arithmetic). Slice 17 lowers it: on the last stage before rasterisation, with the flag set,
//! the written position `(x, y, z, w)` becomes
//! `(x, y * PositionYScale, z * PositionZScale + w * PositionZOffset, w)`, after the -1..1
//! depth-range rewrite where that flag is set too. The GPU crate fills the three words from the
//! draw's first viewport and the clip range: [`DriverConstants::flip_y`],
//! [`DriverConstants::reverse_depth`] and [`DriverConstants::clip_depth`].
//!
//! The depth mirror keeps the guest's depths only for the range clip-space z covers when the
//! shader is done with it: `w - z` (scale -1.0, offset 1.0) when that is 0..w (the guest's 0..1
//! depth mode, or its -1..1 mode after the rewrite), and `-z` (scale -1.0, offset 0.0) when it
//! is -w..w (the guest's -1..1 mode on a host that takes it natively, so no rewrite is asked
//! for). `w - z` there would clip `z = -w` and shift every depth. Pinned by
//! `tests::the_position_words_flip_and_mirror` and, for the clip range each device and depth
//! mode leaves, the render crate's
//! `emulation::tests::a_native_minus_one_to_one_range_is_mirrored_about_zero`.
//!
//! A guest depth range outside 0..1 is not remapped here, although the approved spec's row
//! says the vertex shader "remaps depth" for "a reversed or out-of-range depth range": the GPU
//! crate clamps both depths to 0..1 as Eden does, which gives different depths than a remap.
//! That is an **open owner question** (keep Eden's clamp, or remap and clip in the shader),
//! stated with both options in the render crate's `emulation` module docs ("A depth outside
//! 0..=1: an open owner question"). The words are a scale and an offset rather than two switches
//! so such a remap would need no layout change; that is a guess at what it needs. No reference
//! does any of this: Eden hands the flipped viewport to Vulkan as it is.
//!
//! Where the GPU crate takes each value from (the render area from the surface clip or colour
//! target 0, as Eden-NX2's `vk_graphics_pipeline.cpp` does, and the base vertex and instance from
//! the draw) is G18's to decide and cite; this module only says where each goes.

/// The version of the layout below.
pub const DRIVER_CONSTANTS_VERSION: u32 = 2;

/// The size of the block in bytes: nine words, padded to a multiple of 16 (a guess; see the
/// module docs).
pub const DRIVER_CONSTANTS_SIZE: usize = 48;

/// The size of one value.
const WORD: usize = 4;

/// One value of the driver constants.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DriverConstant {
    /// The alpha-test reference.
    AlphaReference,
    /// The viewport flip, -1.0 or 1.0.
    ViewportFlip,
    /// The render area's width.
    RenderAreaWidth,
    /// The render area's height.
    RenderAreaHeight,
    /// The base vertex.
    BaseVertex,
    /// The base instance.
    BaseInstance,
    /// The scale applied to the position's y: -1.0 or 1.0.
    PositionYScale,
    /// The scale applied to the position's z: -1.0 or 1.0.
    PositionZScale,
    /// The offset added to the position's z, in units of its w: 1.0 or 0.0.
    PositionZOffset,
}

/// The type a driver constant is read as.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ConstantType {
    /// A 32-bit float.
    F32,
    /// A 32-bit unsigned integer.
    U32,
}

impl DriverConstant {
    /// Every value, in layout order.
    pub const ALL: [Self; 9] = [
        Self::AlphaReference,
        Self::ViewportFlip,
        Self::RenderAreaWidth,
        Self::RenderAreaHeight,
        Self::BaseVertex,
        Self::BaseInstance,
        Self::PositionYScale,
        Self::PositionZScale,
        Self::PositionZOffset,
    ];

    /// The value's byte offset in the block.
    pub const fn offset(self) -> usize {
        let index = match self {
            Self::AlphaReference => 0,
            Self::ViewportFlip => 1,
            Self::RenderAreaWidth => 2,
            Self::RenderAreaHeight => 3,
            Self::BaseVertex => 4,
            Self::BaseInstance => 5,
            Self::PositionYScale => 6,
            Self::PositionZScale => 7,
            Self::PositionZOffset => 8,
        };
        index * WORD
    }

    /// The type the value is read as.
    pub const fn ty(self) -> ConstantType {
        match self {
            Self::AlphaReference
            | Self::ViewportFlip
            | Self::RenderAreaWidth
            | Self::RenderAreaHeight
            | Self::PositionYScale
            | Self::PositionZScale
            | Self::PositionZOffset => ConstantType::F32,
            Self::BaseVertex | Self::BaseInstance => ConstantType::U32,
        }
    }
}

/// The layout a shader was translated against: what [`crate::host::HostShader`] will carry
/// once slice 17 lowers reads of the driver constants (spec, `HostShader::driver_constants`).
/// Only [`DriverConstantsLayout::CURRENT`] exists; a capture or cache entry made against
/// another version is refused by whoever reads it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DriverConstantsLayout {
    version: u32,
}

impl DriverConstantsLayout {
    /// The layout this build writes and reads.
    pub const CURRENT: Self = Self {
        version: DRIVER_CONSTANTS_VERSION,
    };

    /// The layout's version.
    pub const fn version(self) -> u32 {
        self.version
    }

    /// The block's size in bytes.
    pub const fn size(self) -> usize {
        DRIVER_CONSTANTS_SIZE
    }
}

/// The values of one draw's driver constants.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DriverConstants {
    /// The alpha-test reference value.
    pub alpha_reference: f32,
    /// Whether y is negated (Eden's `y_negate`); written as -1.0, else 1.0.
    pub y_negate: bool,
    /// The render area's width in pixels.
    pub render_area_width: f32,
    /// The render area's height in pixels.
    pub render_area_height: f32,
    /// The base vertex, the register's 32 bits (a negative offset is its two's complement).
    pub base_vertex: u32,
    /// The base instance.
    pub base_instance: u32,
    /// Whether the vertex shader negates the position's y (the guest viewport's height is
    /// negative); written as a y scale of -1.0, else 1.0 (module docs, "The flipped viewport").
    pub flip_y: bool,
    /// Whether the vertex shader mirrors the position's z (the guest viewport's depth range is
    /// reversed); written as a z scale of -1.0, else 1.0, with the offset that
    /// [`DriverConstants::clip_depth`] picks; without a reversal the offset is 0.0.
    pub reverse_depth: bool,
    /// The range clip-space z covers once the vertex shader is done with it, which picks the
    /// reversal's z offset (module docs, "The flipped viewport").
    pub clip_depth: ClipDepth,
}

/// The range clip-space z covers once the vertex shader is done with it: what the depth
/// reversal mirrors within (module docs, "The flipped viewport").
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum ClipDepth {
    /// 0..w: the guest's 0..1 depth mode, or its -1..1 mode after the `DEPTH_RANGE_REWRITE`
    /// rewrite. The reversal is `w - z`, a z offset of 1.0.
    #[default]
    ZeroToW,
    /// -w..w: the guest's -1..1 depth mode without the rewrite (the host takes it natively). The
    /// reversal is `-z`, a z offset of 0.0.
    MinusWToW,
}

impl DriverConstants {
    /// The 32 bits of `constant`.
    pub fn word(&self, constant: DriverConstant) -> u32 {
        match constant {
            DriverConstant::AlphaReference => self.alpha_reference.to_bits(),
            DriverConstant::ViewportFlip => {
                if self.y_negate { -1.0_f32 } else { 1.0_f32 }.to_bits()
            }
            DriverConstant::RenderAreaWidth => self.render_area_width.to_bits(),
            DriverConstant::RenderAreaHeight => self.render_area_height.to_bits(),
            DriverConstant::BaseVertex => self.base_vertex,
            DriverConstant::BaseInstance => self.base_instance,
            DriverConstant::PositionYScale => sign(self.flip_y).to_bits(),
            DriverConstant::PositionZScale => sign(self.reverse_depth).to_bits(),
            DriverConstant::PositionZOffset => {
                if self.reverse_depth && self.clip_depth == ClipDepth::ZeroToW {
                    1.0_f32
                } else {
                    0.0_f32
                }
                .to_bits()
            }
        }
    }

    /// The block's bytes, each value at its offset and the padding zero.
    pub fn encode(&self) -> [u8; DRIVER_CONSTANTS_SIZE] {
        let mut out = [0; DRIVER_CONSTANTS_SIZE];
        for constant in DriverConstant::ALL {
            let offset = constant.offset();
            if let Some(slot) = out.get_mut(offset..offset + WORD) {
                slot.copy_from_slice(&self.word(constant).to_le_bytes());
            }
        }
        out
    }
}

/// -1.0 when `negate`, else 1.0.
const fn sign(negate: bool) -> f32 {
    if negate { -1.0 } else { 1.0 }
}

/// Every value lies inside the block.
const _: () = {
    let mut index = 0;
    while index < DriverConstant::ALL.len() {
        assert!(DriverConstant::ALL[index].offset() + WORD <= DRIVER_CONSTANTS_SIZE);
        index += 1;
    }
};

#[cfg(test)]
mod tests;
