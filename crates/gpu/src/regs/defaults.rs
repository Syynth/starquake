//! The engine classes' register reset values ([`RegisterDefault`]): what a register holds when a
//! class's file is made, before the guest writes it. [`super::ClassRules::initial_values`] applies
//! the class's list, and is the one starting state the front's register file, its shadow file
//! and the back's mirror all begin from.
//!
//! Values follow Eden-NX2 (gpu-protocol decision 10: Nvidia's headers settle layouts, so each
//! value is placed by the generated `clc797` and `cl902d` methods and fields, and Eden-NX2's
//! engines settle the values). Eden mainline (`eden`) and PR #4588 (`eden-pr4588`) have the same
//! `Maxwell3D::InitializeRegisterDefaults` and `Fermi2D` constructor, word for word, at the
//! commits pinned in `docs/references.md`; no reference disagrees on a value.
//!
//! Sources, relative to the reference folder (`tools/refs-path.sh`):
//!
//! - 3D ([`DEFAULTS_3D`]): `Maxwell3D::InitializeRegisterDefaults` in
//!   `eden-nx2/src/video_core/engines/maxwell_3d.cpp` @ 50fa86bb, with the byte offsets of its
//!   fields from the `ASSERT_REG_POSITION` lines and struct layouts of
//!   `eden-nx2/src/video_core/engines/maxwell_3d.h`, each matched to the header method at that
//!   byte address (`clc797.h`).
//! - 2D ([`DEFAULTS_2D`]): the `Fermi2D` constructor in
//!   `eden-nx2/src/video_core/engines/fermi_2d.cpp` @ 50fa86bb.
//! - Compute, DMA copy and inline-to-memory set nothing: the `KeplerCompute`, `MaxwellDMA` and
//!   `KeplerMemory` constructors (`eden-nx2/src/video_core/engines/kepler_compute.cpp`,
//!   `maxwell_dma.cpp`, `kepler_memory.cpp`) write no register, and each engine's `regs{}`
//!   (`kepler_compute.h`, `maxwell_dma.h`, `kepler_memory.h`) value-initialises every register
//!   to zero. Their lists are empty.
//!
//! # How Eden-NX2's 3D fields land on the header's methods
//!
//! Every field `InitializeRegisterDefaults` sets, after its `memset` to zero, by Eden's byte
//! offset and the header method there:
//!
//! - `viewports[i].depth_range_far` (0x0C00 + 0x10·i + 0xC) is `SET_VIEWPORT_CLIP_MAX_Z(i)`:
//!   1.0. `depth_range_near` (`SET_VIEWPORT_CLIP_MIN_Z(i)`) is set to 0.0, the zero it already
//!   holds, so it has no entry.
//! - `viewport_transform[i].swizzle` (0x0A00 + 0x20·i + 0x18) is
//!   `SET_VIEWPORT_COORDINATE_SWIZZLE(i)`: `POS_X`, `POS_Y`, `POS_Z`, `POS_W` (Eden's
//!   `ViewportSwizzle` 0, 2, 4, 6 at bits 0, 4, 8, 12, the header's `X`, `Y`, `Z`, `W` fields).
//! - `blend` (0x133C) is `SET_BLEND_SEPARATE_FOR_ALPHA` onwards: the colour and alpha ops are
//!   `D3D_ADD` (Eden's `Add_D3D`, 1), the source coefficients `D3D_ONE` (`One_D3D`, 2) and the
//!   destination coefficients `D3D_ZERO` (`Zero_D3D`, 1). Eden's struct puts
//!   `enable_global_color_key` (`SET_GLOBAL_COLOR_KEY`, left 0) between `alpha_source` and
//!   `alpha_dest`, as the header does.
//! - `blend_per_target[i]` (0x1E00 + 0x20·i) is `SET_BLEND_PER_TARGET_*(i)`: the same six
//!   values.
//! - `stencil_front_op` (0x1384) is `SET_STENCIL_OP_FAIL`, `_ZFAIL`, `_ZPASS` (`D3D_KEEP`, 1)
//!   and `SET_STENCIL_FUNC` (`OGL_ALWAYS`, Eden's `Always_GL`, 0x207); `stencil_back_op`
//!   (0x1598) is the `SET_BACK_STENCIL_*` four, alike.
//! - `stencil_front_func_mask` (0x1398), `stencil_front_mask` (0x139C), `stencil_back_mask`
//!   (0x0F58) and `stencil_back_func_mask` (0x0F5C) are `SET_STENCIL_FUNC_MASK`,
//!   `SET_STENCIL_MASK`, `SET_BACK_STENCIL_MASK` and `SET_BACK_STENCIL_FUNC_MASK`: 0xFFFFFFFF,
//!   the whole word as Eden writes it, though the header's `V` field is only bits 7:0 (a
//!   register holds the word written, and a reader takes the field).
//! - `stencil_two_side_enable` (0x1594) is `SET_TWO_SIDED_STENCIL_TEST`: `TRUE`.
//! - `depth_test_func` (0x130C) is `SET_DEPTH_FUNC`: `OGL_ALWAYS` (0x207).
//! - `gl_front_face` (0x191C) is `OGL_SET_FRONT_FACE`: Eden sets `CounterClockWise` (0x901)
//!   and later in the same function `ClockWise` (0x900), so the default is `CW`.
//! - `gl_cull_face` (0x1920) is `OGL_SET_CULL_FACE`: `BACK` (0x405).
//! - `point_size` (0x1518) is `SET_POINT_SIZE`: 1.0.
//! - `color_mask[i]` (0x1A00 + 4·i) is `SET_CT_WRITE(i)`: the `R`, `G`, `B` and `A` enables
//!   (bits 0, 4, 8, 12) all `TRUE`.
//! - `vertex_attrib_format[i]` (0x1160 + 4·i, 32 of them) is `SET_VERTEX_ATTRIBUTE_A(i)` for
//!   the first 16 and `SET_VERTEX_ATTRIBUTE_B(i - 16)` for the rest (the header puts `_B` at
//!   0x11A0, with the same fields): Eden's `constant` bit 6 is the header's `SOURCE` field,
//!   `INACTIVE` (1).
//! - `rasterize_enable` (0x037C) is `SET_RASTER_ENABLE`, `color_target_mrt_enable` (0x0FAC)
//!   `SET_CT_MRT_ENABLE` and `framebuffer_srgb` (0x15B8) `SET_SRGB_WRITE`: each `TRUE`.
//! - `line_width_smooth` (0x13B0) and `line_width_aliased` (0x13B4) are
//!   `SET_LINE_WIDTH_FLOAT` and `SET_ALIASED_LINE_WIDTH_FLOAT`: 1.0.
//! - `polygon_mode_front` (0x0DAC) and `polygon_mode_back` (0x0DB0) are
//!   `SET_FRONT_POLYGON_MODE` and `SET_BACK_POLYGON_MODE`: `FILL` (0x1B02).
//!
//! A float is its IEEE 754 single-precision bits (1.0 is 0x3F800000), as Eden's `f32` and
//! `float` fields store it. The array lengths are Mesa's (`METHOD_ARRAY_SIZES` in
//! `mesa-nouveau/src/nouveau/headers/class_parser.py`), and Eden's agree: 16 viewports
//! (`NumViewports`), 8 blend targets and colour masks (`NumRenderTargets`), and 16 vertex
//! attributes in each of the header's two halves (Eden's `NumVertexAttributes`, 32, covers both).
//!
//! The literal values are pinned by
//! `regs::tests::defaults::reset_values_are_the_references_written_out`.

use crate::back::state::{BLEND_TARGETS, VERTEX_ATTRIBUTES_PER_HALF, VIEWPORTS};
use crate::classes::{Method, cl902d, clc797 as c};

use super::rules::MethodSpan;

/// One register's reset value, or the same value for the first elements of an array method.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RegisterDefault {
    /// The registers set.
    pub span: MethodSpan,
    /// The value each of them holds when the file is made.
    pub value: u32,
}

impl RegisterDefault {
    /// `method` (element 0, for an array method) holds `value`.
    pub const fn one(method: &'static Method, value: u32) -> Self {
        Self {
            span: MethodSpan::one(method),
            value,
        }
    }

    /// Elements `0..count` of the array method `method` each hold `value`.
    pub const fn array(method: &'static Method, count: u32, value: u32) -> Self {
        Self {
            span: MethodSpan::array(method, count),
            value,
        }
    }
}

/// 1.0 as an IEEE 754 single (Eden's `1.0f` in an `f32` or `float` register).
const ONE_FLOAT: u32 = 1.0_f32.to_bits();

/// Every viewport's identity swizzle: `POS_X`, `POS_Y`, `POS_Z`, `POS_W` (Eden's
/// `ViewportSwizzle::PositiveX` to `PositiveW`).
const IDENTITY_SWIZZLE: u32 = {
    let word = c::SET_VIEWPORT_COORDINATE_SWIZZLE_X
        .insert_truncated(0, c::SET_VIEWPORT_COORDINATE_SWIZZLE_X_POS_X);
    let word = c::SET_VIEWPORT_COORDINATE_SWIZZLE_Y
        .insert_truncated(word, c::SET_VIEWPORT_COORDINATE_SWIZZLE_Y_POS_Y);
    let word = c::SET_VIEWPORT_COORDINATE_SWIZZLE_Z
        .insert_truncated(word, c::SET_VIEWPORT_COORDINATE_SWIZZLE_Z_POS_Z);
    c::SET_VIEWPORT_COORDINATE_SWIZZLE_W
        .insert_truncated(word, c::SET_VIEWPORT_COORDINATE_SWIZZLE_W_POS_W)
};

/// Every colour target's write mask: all four components enabled (Eden's `color_mask` `R`,
/// `G`, `B`, `A` all 1).
const ALL_COMPONENTS: u32 = {
    let word = c::SET_CT_WRITE_R_ENABLE.insert_truncated(0, c::SET_CT_WRITE_R_ENABLE_TRUE);
    let word = c::SET_CT_WRITE_G_ENABLE.insert_truncated(word, c::SET_CT_WRITE_G_ENABLE_TRUE);
    let word = c::SET_CT_WRITE_B_ENABLE.insert_truncated(word, c::SET_CT_WRITE_B_ENABLE_TRUE);
    c::SET_CT_WRITE_A_ENABLE.insert_truncated(word, c::SET_CT_WRITE_A_ENABLE_TRUE)
};

/// Eden writes every stencil mask's whole word (`0xFFFFFFFF`); the header's field is 7:0.
const ALL_STENCIL_BITS: u32 = u32::MAX;

/// The 3D class's (`AMPERE_B`, `0xC797`) reset values, from Eden-NX2's
/// `Maxwell3D::InitializeRegisterDefaults`, in the function's order (the module docs map each
/// field to its method). Every register not listed starts at zero.
pub const DEFAULTS_3D: &[RegisterDefault] = &[
    RegisterDefault::array(&c::SET_VIEWPORT_CLIP_MAX_Z, VIEWPORTS, ONE_FLOAT),
    RegisterDefault::array(
        &c::SET_VIEWPORT_COORDINATE_SWIZZLE,
        VIEWPORTS,
        IDENTITY_SWIZZLE,
    ),
    RegisterDefault::one(&c::SET_BLEND_COLOR_OP, c::SET_BLEND_COLOR_OP_V_D3D_ADD),
    RegisterDefault::one(
        &c::SET_BLEND_COLOR_SOURCE_COEFF,
        c::SET_BLEND_COLOR_SOURCE_COEFF_V_D3D_ONE,
    ),
    RegisterDefault::one(
        &c::SET_BLEND_COLOR_DEST_COEFF,
        c::SET_BLEND_COLOR_DEST_COEFF_V_D3D_ZERO,
    ),
    RegisterDefault::one(&c::SET_BLEND_ALPHA_OP, c::SET_BLEND_ALPHA_OP_V_D3D_ADD),
    RegisterDefault::one(
        &c::SET_BLEND_ALPHA_SOURCE_COEFF,
        c::SET_BLEND_ALPHA_SOURCE_COEFF_V_D3D_ONE,
    ),
    RegisterDefault::one(
        &c::SET_BLEND_ALPHA_DEST_COEFF,
        c::SET_BLEND_ALPHA_DEST_COEFF_V_D3D_ZERO,
    ),
    RegisterDefault::array(
        &c::SET_BLEND_PER_TARGET_COLOR_OP,
        BLEND_TARGETS,
        c::SET_BLEND_PER_TARGET_COLOR_OP_V_D3D_ADD,
    ),
    RegisterDefault::array(
        &c::SET_BLEND_PER_TARGET_COLOR_SOURCE_COEFF,
        BLEND_TARGETS,
        c::SET_BLEND_PER_TARGET_COLOR_SOURCE_COEFF_V_D3D_ONE,
    ),
    RegisterDefault::array(
        &c::SET_BLEND_PER_TARGET_COLOR_DEST_COEFF,
        BLEND_TARGETS,
        c::SET_BLEND_PER_TARGET_COLOR_DEST_COEFF_V_D3D_ZERO,
    ),
    RegisterDefault::array(
        &c::SET_BLEND_PER_TARGET_ALPHA_OP,
        BLEND_TARGETS,
        c::SET_BLEND_PER_TARGET_ALPHA_OP_V_D3D_ADD,
    ),
    RegisterDefault::array(
        &c::SET_BLEND_PER_TARGET_ALPHA_SOURCE_COEFF,
        BLEND_TARGETS,
        c::SET_BLEND_PER_TARGET_ALPHA_SOURCE_COEFF_V_D3D_ONE,
    ),
    RegisterDefault::array(
        &c::SET_BLEND_PER_TARGET_ALPHA_DEST_COEFF,
        BLEND_TARGETS,
        c::SET_BLEND_PER_TARGET_ALPHA_DEST_COEFF_V_D3D_ZERO,
    ),
    RegisterDefault::one(&c::SET_STENCIL_OP_FAIL, c::SET_STENCIL_OP_FAIL_V_D3D_KEEP),
    RegisterDefault::one(&c::SET_STENCIL_OP_ZFAIL, c::SET_STENCIL_OP_ZFAIL_V_D3D_KEEP),
    RegisterDefault::one(&c::SET_STENCIL_OP_ZPASS, c::SET_STENCIL_OP_ZPASS_V_D3D_KEEP),
    RegisterDefault::one(&c::SET_STENCIL_FUNC, c::SET_STENCIL_FUNC_V_OGL_ALWAYS),
    RegisterDefault::one(&c::SET_STENCIL_FUNC_MASK, ALL_STENCIL_BITS),
    RegisterDefault::one(&c::SET_STENCIL_MASK, ALL_STENCIL_BITS),
    RegisterDefault::one(
        &c::SET_TWO_SIDED_STENCIL_TEST,
        c::SET_TWO_SIDED_STENCIL_TEST_ENABLE_TRUE,
    ),
    RegisterDefault::one(
        &c::SET_BACK_STENCIL_OP_FAIL,
        c::SET_BACK_STENCIL_OP_FAIL_V_D3D_KEEP,
    ),
    RegisterDefault::one(
        &c::SET_BACK_STENCIL_OP_ZFAIL,
        c::SET_BACK_STENCIL_OP_ZFAIL_V_D3D_KEEP,
    ),
    RegisterDefault::one(
        &c::SET_BACK_STENCIL_OP_ZPASS,
        c::SET_BACK_STENCIL_OP_ZPASS_V_D3D_KEEP,
    ),
    RegisterDefault::one(
        &c::SET_BACK_STENCIL_FUNC,
        c::SET_BACK_STENCIL_FUNC_V_OGL_ALWAYS,
    ),
    RegisterDefault::one(&c::SET_BACK_STENCIL_FUNC_MASK, ALL_STENCIL_BITS),
    RegisterDefault::one(&c::SET_BACK_STENCIL_MASK, ALL_STENCIL_BITS),
    RegisterDefault::one(&c::SET_DEPTH_FUNC, c::SET_DEPTH_FUNC_V_OGL_ALWAYS),
    RegisterDefault::one(&c::OGL_SET_CULL_FACE, c::OGL_SET_CULL_FACE_V_BACK),
    RegisterDefault::one(&c::SET_POINT_SIZE, ONE_FLOAT),
    RegisterDefault::array(&c::SET_CT_WRITE, BLEND_TARGETS, ALL_COMPONENTS),
    RegisterDefault::array(
        &c::SET_VERTEX_ATTRIBUTE_A,
        VERTEX_ATTRIBUTES_PER_HALF,
        c::SET_VERTEX_ATTRIBUTE_A_SOURCE
            .insert_truncated(0, c::SET_VERTEX_ATTRIBUTE_A_SOURCE_INACTIVE),
    ),
    RegisterDefault::array(
        &c::SET_VERTEX_ATTRIBUTE_B,
        VERTEX_ATTRIBUTES_PER_HALF,
        c::SET_VERTEX_ATTRIBUTE_B_SOURCE
            .insert_truncated(0, c::SET_VERTEX_ATTRIBUTE_B_SOURCE_INACTIVE),
    ),
    RegisterDefault::one(&c::SET_RASTER_ENABLE, c::SET_RASTER_ENABLE_V_TRUE),
    RegisterDefault::one(&c::SET_CT_MRT_ENABLE, c::SET_CT_MRT_ENABLE_V_TRUE),
    RegisterDefault::one(&c::SET_SRGB_WRITE, c::SET_SRGB_WRITE_ENABLE_TRUE),
    RegisterDefault::one(&c::SET_ALIASED_LINE_WIDTH_FLOAT, ONE_FLOAT),
    RegisterDefault::one(&c::SET_LINE_WIDTH_FLOAT, ONE_FLOAT),
    // Eden sets `CounterClockWise` first and `ClockWise` here; the last write stands.
    RegisterDefault::one(&c::OGL_SET_FRONT_FACE, c::OGL_SET_FRONT_FACE_V_CW),
    RegisterDefault::one(&c::SET_BACK_POLYGON_MODE, c::SET_BACK_POLYGON_MODE_V_FILL),
    RegisterDefault::one(&c::SET_FRONT_POLYGON_MODE, c::SET_FRONT_POLYGON_MODE_V_FILL),
];

/// The 2D class's (`FERMI_TWOD_A`, `0x902D`) reset values: `SET_SRC_DEPTH` and `SET_DST_DEPTH`
/// are 1, as Eden-NX2's `Fermi2D` constructor sets them (`regs.src.depth = 1; regs.dst.depth =
/// 1;`, "Nvidia's OpenGL driver seems to assume these values"). Every other register starts at
/// zero. The 2D blit reads the depth as the mirror stores it, so without these a title that
/// never writes the depth has every block-linear blit refused (`crate::back::copy::twod`).
pub const DEFAULTS_2D: &[RegisterDefault] = &[
    RegisterDefault::one(&cl902d::SET_SRC_DEPTH, 1),
    RegisterDefault::one(&cl902d::SET_DST_DEPTH, 1),
];
