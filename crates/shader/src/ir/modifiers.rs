//! Operation modifiers: floating-point controls and texture information.
//!
//! Ported from Eden's `eden/src/shader_recompiler/frontend/ir/modifiers.h` and the two enums
//! `TextureType` and `ImageFormat` of `eden/src/shader_recompiler/shader_info.h` @ cf654052
//! (GPL-3.0-or-later). Eden packs these into an instruction's untyped 32-bit flags word; here
//! they are the typed [`Flags`], and the opcode table says which kind each opcode carries
//! ([`Opcode::flags_kind`](super::Opcode::flags_kind)). Controls are carried even where the
//! backend cannot honour them (spec, "The intermediate representation"; decision 10).

/// A rounding mode (Eden's `FpRounding`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum FpRounding {
    /// Not specified for this instruction.
    #[default]
    DontCare,
    /// Round to nearest even.
    Rn,
    /// Round towards negative infinity.
    Rm,
    /// Round towards positive infinity.
    Rp,
    /// Round towards zero.
    Rz,
}

impl FpRounding {
    /// The modes that print, in declaration order.
    pub const NAMED: [Self; 4] = [Self::Rn, Self::Rm, Self::Rp, Self::Rz];

    /// The mode's flag in the text form; `None` for [`FpRounding::DontCare`], which prints
    /// nothing.
    pub const fn name(self) -> Option<&'static str> {
        match self {
            Self::DontCare => None,
            Self::Rn => Some("rn"),
            Self::Rm => Some("rm"),
            Self::Rp => Some("rp"),
            Self::Rz => Some("rz"),
        }
    }
}

/// How denormals and multiplication by zero behave (Eden's `FmzMode`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum FmzMode {
    /// Not specified for this instruction.
    #[default]
    DontCare,
    /// Flush denormals to zero; NaN is propagated.
    Ftz,
    /// Flush denormals to zero; `x * 0 == 0`.
    Fmz,
    /// Denormals are not flushed; NaN is propagated.
    None,
}

impl FmzMode {
    /// The modes that print, in declaration order.
    pub const NAMED: [Self; 3] = [Self::Ftz, Self::Fmz, Self::None];

    /// The mode's flag in the text form; `None` for [`FmzMode::DontCare`], which prints
    /// nothing.
    pub const fn name(self) -> Option<&'static str> {
        match self {
            Self::DontCare => None,
            Self::Ftz => Some("ftz"),
            Self::Fmz => Some("fmz"),
            Self::None => Some("fmz_none"),
        }
    }
}

/// Floating-point controls (Eden's `FpControl`). The default is Eden's: contraction allowed,
/// rounding and denormal handling unspecified.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct FpControl {
    /// The operation must not be fused with another (Eden's `no_contraction`).
    pub no_contraction: bool,
    /// The rounding mode.
    pub rounding: FpRounding,
    /// Denormal and multiply-by-zero handling.
    pub fmz_mode: FmzMode,
}

/// The flag [`FpControl::no_contraction`] prints as.
pub const NO_CONTRACTION: &str = "no_contraction";

/// A texture's dimension (Eden's `TextureType`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum TextureType {
    /// One-dimensional.
    #[default]
    Color1D,
    /// A one-dimensional array.
    ColorArray1D,
    /// Two-dimensional.
    Color2D,
    /// A two-dimensional array.
    ColorArray2D,
    /// Three-dimensional.
    Color3D,
    /// A cube.
    ColorCube,
    /// A cube array.
    ColorArrayCube,
    /// A texel buffer.
    Buffer,
    /// A two-dimensional rectangle (unnormalised coordinates).
    Color2DRect,
}

impl TextureType {
    /// Every type, in declaration order.
    pub const ALL: [Self; 9] = [
        Self::Color1D,
        Self::ColorArray1D,
        Self::Color2D,
        Self::ColorArray2D,
        Self::Color3D,
        Self::ColorCube,
        Self::ColorArrayCube,
        Self::Buffer,
        Self::Color2DRect,
    ];

    /// The type's name in the text form.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Color1D => "1d",
            Self::ColorArray1D => "array_1d",
            Self::Color2D => "2d",
            Self::ColorArray2D => "array_2d",
            Self::Color3D => "3d",
            Self::ColorCube => "cube",
            Self::ColorArrayCube => "array_cube",
            Self::Buffer => "buffer",
            Self::Color2DRect => "2d_rect",
        }
    }
}

/// A storage image's format (Eden's `ImageFormat`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum ImageFormat {
    /// Not known from the instruction.
    #[default]
    Typeless,
    /// `R8_UINT`.
    R8Uint,
    /// `R8_SINT`.
    R8Sint,
    /// `R16_UINT`.
    R16Uint,
    /// `R16_SINT`.
    R16Sint,
    /// `R32_UINT`.
    R32Uint,
    /// `R32G32_UINT`.
    R32G32Uint,
    /// `R32G32B32A32_UINT`.
    R32G32B32A32Uint,
}

impl ImageFormat {
    /// Every format, in declaration order.
    pub const ALL: [Self; 8] = [
        Self::Typeless,
        Self::R8Uint,
        Self::R8Sint,
        Self::R16Uint,
        Self::R16Sint,
        Self::R32Uint,
        Self::R32G32Uint,
        Self::R32G32B32A32Uint,
    ];

    /// The format's name in the text form.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Typeless => "typeless",
            Self::R8Uint => "r8_uint",
            Self::R8Sint => "r8_sint",
            Self::R16Uint => "r16_uint",
            Self::R16Sint => "r16_sint",
            Self::R32Uint => "r32_uint",
            Self::R32G32Uint => "r32g32_uint",
            Self::R32G32B32A32Uint => "r32g32b32a32_uint",
        }
    }
}

/// The largest value of [`TextureInfo::gather_component`] and
/// [`TextureInfo::num_derivatives`]: both are 2-bit fields in Eden's `TextureInstInfo`
/// (`BitField<23, 2>` and `BitField<25, 2>`).
pub const TEXTURE_TWO_BIT_MAX: u8 = 3;

/// What a texture or image operation knows about its resource (Eden's `TextureInstInfo`). The
/// default is Eden's zero word.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "mirrors the one-bit fields of Eden's TextureInstInfo, one for one"
)]
pub struct TextureInfo {
    /// The index of the resource's descriptor, set by the texture pass.
    pub descriptor_index: u16,
    /// The dimension.
    pub ty: TextureType,
    /// A depth comparison.
    pub is_depth: bool,
    /// A level-of-detail bias is given.
    pub has_bias: bool,
    /// A level-of-detail clamp is given.
    pub has_lod_clamp: bool,
    /// Reduced precision is allowed.
    pub relaxed_precision: bool,
    /// The component a gather reads, 0 to [`TEXTURE_TWO_BIT_MAX`].
    pub gather_component: u8,
    /// How many derivatives are given, 0 to [`TEXTURE_TWO_BIT_MAX`].
    pub num_derivatives: u8,
    /// A storage image's format.
    pub image_format: ImageFormat,
    /// The `.NDV` modifier is set.
    pub ndv_is_active: bool,
    /// The sampled texture holds integers (sampled-texture operations only).
    pub is_integer: bool,
}

impl TextureInfo {
    /// Whether the 2-bit fields are in range.
    pub const fn is_valid(&self) -> bool {
        self.gather_component <= TEXTURE_TWO_BIT_MAX && self.num_derivatives <= TEXTURE_TWO_BIT_MAX
    }
}

/// An instruction's modifiers. Which variant an opcode carries is fixed by
/// [`Opcode::flags_kind`](super::Opcode::flags_kind); the verifier checks it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Flags {
    /// No modifiers.
    #[default]
    None,
    /// Floating-point controls.
    Fp(FpControl),
    /// Texture information.
    Texture(TextureInfo),
}

impl Flags {
    /// The kind of these flags.
    pub const fn kind(&self) -> super::FlagsKind {
        match self {
            Self::None => super::FlagsKind::None,
            Self::Fp(_) => super::FlagsKind::Fp,
            Self::Texture(_) => super::FlagsKind::Texture,
        }
    }

    /// The default flags of `kind`: what an instruction carries when none are given.
    pub fn default_of(kind: super::FlagsKind) -> Self {
        match kind {
            super::FlagsKind::None => Self::None,
            super::FlagsKind::Fp => Self::Fp(FpControl::default()),
            super::FlagsKind::Texture => Self::Texture(TextureInfo::default()),
        }
    }
}
