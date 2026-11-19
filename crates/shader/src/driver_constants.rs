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
//! | 24 | padding, zero | | |
//!
//! Every value is a 32-bit little-endian word at a 4-byte-aligned offset, and the block is
//! [`DRIVER_CONSTANTS_SIZE`] bytes. That the block is padded to a multiple of 16 bytes, so it
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
//! Where the GPU crate takes each value from (the render area from the surface clip or colour
//! target 0, as Eden-NX2's `vk_graphics_pipeline.cpp` does, and the base vertex and instance from
//! the draw) is G18's to decide and cite; this module only says where each goes.

/// The version of the layout below.
pub const DRIVER_CONSTANTS_VERSION: u32 = 1;

/// The size of the block in bytes: six words, padded to a multiple of 16 (a guess; see the
/// module docs).
pub const DRIVER_CONSTANTS_SIZE: usize = 32;

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
    pub const ALL: [Self; 6] = [
        Self::AlphaReference,
        Self::ViewportFlip,
        Self::RenderAreaWidth,
        Self::RenderAreaHeight,
        Self::BaseVertex,
        Self::BaseInstance,
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
        };
        index * WORD
    }

    /// The type the value is read as.
    pub const fn ty(self) -> ConstantType {
        match self {
            Self::AlphaReference
            | Self::ViewportFlip
            | Self::RenderAreaWidth
            | Self::RenderAreaHeight => ConstantType::F32,
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
