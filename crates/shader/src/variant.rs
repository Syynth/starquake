//! The variant key: the part of 3D engine state a translated shader is specialised on, and its
//! versioned byte encoding (`docs/specs/shader-pipeline.md`, "Pipeline-state specialisation";
//! port slice 9).
//!
//! The GPU crate builds the key (slice G18) and the shader crate specialises on it. Owner's
//! decision 16: the key holds types and structure only; numbers go in the driver constants
//! ([`crate::driver_constants`]). The fields, with where each comes from:
//!
//! | Field | What it is | Source |
//! |---|---|---|
//! | [`VariantKey::vertex_attributes`] | the component type of each vertex attribute | Eden `RuntimeInfo::generic_input_types` (`eden/src/shader_recompiler/runtime_info.h` @ cf654052), 32 entries (`Maxwell::NumVertexAttributes`, `eden/src/video_core/engines/maxwell_3d.h`) |
//! | [`VariantKey::color_targets`] | the component type of each colour target | Eden `RuntimeInfo::color_output_types`, filled with `Float`, `SignedInt` or `UnsignedInt` only (`eden/src/video_core/renderer_vulkan/vk_pipeline_cache.cpp`) |
//! | [`VariantKey::dual_source_blend`] | whether dual-source blending is on | Eden `RuntimeInfo::dual_source_blend` |
//! | [`VariantKey::next_stage_reads`] | which attribute words the next stage reads | spec, "interface matching"; see below |
//! | [`VariantKey::workgroup_size`], [`VariantKey::shared_memory_size`] | a compute program's workgroup and shared-memory size | spec, "Compute shaders"; Eden `Environment::WorkgroupSize`, `SharedMemorySize` (`environment.h`) |
//! | [`VariantKey::emulation`] | the render backend's `EmulationFlags`, field by field | spec, "Pipeline-state specialisation"; review guide, M13 |
//! | [`VariantKey::texture_swizzles`] | each texture binding's swizzle, when the host has no view swizzle | spec, the exception to decision 16; review guide, M11 |
//!
//! # Where this differs from Eden
//!
//! - Eden specialises a stage on what the stage before it stores
//!   (`RuntimeInfo::previous_stage_stores`, a mask over `IR::Attribute`). The spec specialises a
//!   stage on what the stage after it reads ("which outputs the next stage reads"), so the key
//!   holds [`VariantKey::next_stage_reads`]. Both are masks over the guest's attribute words;
//!   this one covers the 256 words of the attribute space below
//!   [`crate::translate::ATTRIBUTE_SPACE_END`], which hold every guest attribute
//!   (`IR::Attribute::FrontFace` is word 255 in `eden/src/shader_recompiler/frontend/ir/attribute.h`);
//!   Eden's 512-bit mask also has room for host-only values past them.
//! - Eden keeps the alpha-test function in `RuntimeInfo::alpha_test_func` and its reference
//!   value beside it. The key holds the function as [`EmulationKey::alpha_test`], present
//!   exactly when the render backend's `ALPHA_TEST` flag is set; the reference value is a
//!   driver constant.
//! - Eden's Vulkan rasterizer gathers at most 192 textures per pipeline
//!   (`eden/src/video_core/renderer_vulkan/vk_rasterizer.h`, `MAX_TEXTURES`). The key holds at
//!   most [`MAX_TEXTURE_SWIZZLES`], that same number, and refuses more
//!   ([`KeyError::TooManySwizzles`]) where Eden's fixed-capacity vector has no room. Pinned by
//!   `tests::the_swizzle_cap_at_its_edge`.
//!
//! # Encoding
//!
//! [`VariantKey::encode`] writes the key as bytes for the persistent cache and shader captures
//! (spec, decision 7; slices 20 and 21). The format is starquake's own, little-endian, and
//! changing it means bumping [`VARIANT_KEY_VERSION`]:
//!
//! | Bytes | Field |
//! |---|---|
//! | 1 | version, [`VARIANT_KEY_VERSION`] |
//! | 32 | one [`AttributeType`] code per vertex attribute |
//! | 8 | one [`ColorType`] code per colour target |
//! | 1 | dual-source blending, 0 or 1 |
//! | 32 | next stage's reads: four 64-bit lanes, lane 0 (attribute words 0 to 63) first |
//! | 12 | workgroup size x, y, z, 32 bits each |
//! | 4 | shared-memory size, 32 bits |
//! | 1 | alpha test: 0 for none, else 1 plus the [`CompareFunction`] code |
//! | 4 | depth-range rewrite, primitive-restart rewrite, attribute fetch, typed outputs: 0 or 1 each |
//! | 1 | texture swizzle count, at most [`MAX_TEXTURE_SWIZZLES`] |
//! | 4 per swizzle | the [`SwizzleSource`] codes of x, y, z, w |
//!
//! [`VariantKey::decode`] checks, in this order, and reports the first failure: the version
//! (before anything else, so a key of another version is never read as this one), then each
//! field in the order of the table (a missing byte is [`KeyError::Truncated`], a code outside
//! the field's set is [`KeyError::InvalidValue`], a swizzle count past the cap is
//! [`KeyError::TooManySwizzles`] before any swizzle is read), then that no byte follows
//! ([`KeyError::TrailingBytes`]). Pinned by `tests::the_version_is_checked_first`,
//! `tests::the_swizzle_count_is_checked_before_the_swizzles`,
//! `tests::the_first_invalid_field_is_reported` and `tests::trailing_bytes_are_refused`.

use std::fmt::{self, Display, Formatter};

use thiserror::Error;

use crate::translate::{ATTRIBUTE_SPACE_END, COLOR_TARGETS};

/// The version of the byte encoding [`VariantKey::encode`] writes.
pub const VARIANT_KEY_VERSION: u8 = 1;

/// How many vertex attributes the guest has (`Maxwell::NumVertexAttributes` = 32,
/// `eden/src/video_core/engines/maxwell_3d.h` @ cf654052; `RuntimeInfo::generic_input_types`
/// has as many entries).
pub const VERTEX_ATTRIBUTES: usize = 32;

/// The most texture swizzles a key holds: Eden's Vulkan rasterizer's cap on the textures of one
/// pipeline (`MAX_TEXTURES` = 192, `eden/src/video_core/renderer_vulkan/vk_rasterizer.h` @
/// cf654052). A key is built per set of stages, so the cap counts the bindings of every stage.
pub const MAX_TEXTURE_SWIZZLES: usize = 192;

/// How many attribute words the guest's attribute space holds: [`ATTRIBUTE_SPACE_END`] bytes of
/// 32-bit words.
pub const ATTRIBUTE_WORDS: usize = (ATTRIBUTE_SPACE_END / 4) as usize;

/// How many 64-bit lanes an [`AttributeMask`] has.
pub const ATTRIBUTE_MASK_LANES: usize = ATTRIBUTE_WORDS / u64::BITS as usize;

/// An attribute word is named by a `u8`, which is exact because the space has 256 words.
const _: () = assert!(ATTRIBUTE_WORDS == 1 << u8::BITS);
/// The swizzle count is encoded in one byte.
const _: () = assert!(MAX_TEXTURE_SWIZZLES <= u8::MAX as usize);

/// The size of an encoded swizzle: one code per component.
pub const SWIZZLE_SIZE: usize = 4;

/// The size of an encoded key with no swizzles (see the module docs' table).
pub const VARIANT_KEY_FIXED_SIZE: usize = 1 // version
    + VERTEX_ATTRIBUTES
    + COLOR_TARGETS
    + 1 // dual-source blending
    + ATTRIBUTE_MASK_LANES * 8
    + 3 * 4 // workgroup size
    + 4 // shared-memory size
    + 1 // alpha test
    + 4 // the four emulation switches
    + 1; // swizzle count

/// The size of the longest encoded key.
pub const VARIANT_KEY_MAX_SIZE: usize =
    VARIANT_KEY_FIXED_SIZE + MAX_TEXTURE_SWIZZLES * SWIZZLE_SIZE;

/// Defines a fieldless enum with a byte code per variant, and its `code` and `from_code`.
macro_rules! coded {
    (
        $(#[$meta:meta])*
        $name:ident {
            $( $(#[$variant_meta:meta])* $variant:ident = $code:literal, )*
        }
    ) => {
        $(#[$meta])*
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        pub enum $name {
            $( $(#[$variant_meta])* $variant, )*
        }

        impl $name {
            /// Every variant, in code order.
            pub const ALL: &'static [Self] = &[$(Self::$variant),*];

            /// The variant's code in the key's encoding.
            pub const fn code(self) -> u8 {
                match self {
                    $(Self::$variant => $code,)*
                }
            }

            /// The variant with `code`, if there is one.
            pub const fn from_code(code: u8) -> Option<Self> {
                match code {
                    $($code => Some(Self::$variant),)*
                    _ => None,
                }
            }
        }
    };
}

coded! {
    /// The component type of a vertex attribute (Eden's `AttributeType`, in its order, which
    /// gives the codes; `eden/src/shader_recompiler/runtime_info.h` @ cf654052).
    #[derive(Default)]
    AttributeType {
        /// Floating point, including normalised formats; Eden's value-initialised default.
        #[default]
        Float = 0,
        /// Signed integer.
        SignedInt = 1,
        /// Unsigned integer.
        UnsignedInt = 2,
        /// Signed integer converted to float without normalising.
        SignedScaled = 3,
        /// Unsigned integer converted to float without normalising.
        UnsignedScaled = 4,
        /// Not fetched.
        Disabled = 5,
    }
}

coded! {
    /// The component type of a colour target: the three of Eden's `AttributeType` values that
    /// `vk_pipeline_cache.cpp` writes to `color_output_types`, with the same codes.
    #[derive(Default)]
    ColorType {
        /// Floating point, including normalised formats; Eden's value-initialised default.
        #[default]
        Float = 0,
        /// Signed integer.
        SignedInt = 1,
        /// Unsigned integer.
        UnsignedInt = 2,
    }
}

coded! {
    /// A comparison (Eden's `Shader::CompareFunction`, in its order, which gives the codes;
    /// `runtime_info.h` @ cf654052).
    CompareFunction {
        /// Never passes.
        Never = 0,
        /// Passes when less.
        Less = 1,
        /// Passes when equal.
        Equal = 2,
        /// Passes when less or equal.
        LessThanEqual = 3,
        /// Passes when greater.
        Greater = 4,
        /// Passes when not equal.
        NotEqual = 5,
        /// Passes when greater or equal.
        GreaterThanEqual = 6,
        /// Always passes.
        Always = 7,
    }
}

coded! {
    /// Where a component of a swizzled texture result comes from. The codes are the texture
    /// header's (`NVC797_TEXHEAD_BL_X_SOURCE_IN_*`,
    /// `mesa-nouveau/src/nouveau/headers/nvidia/classes/clc797tex.h` @ 52238762; Eden's
    /// `SwizzleSource`, `eden/src/video_core/textures/texture.h`); code 1 is none.
    SwizzleSource {
        /// Zero.
        Zero = 0,
        /// The red component.
        R = 2,
        /// The green component.
        G = 3,
        /// The blue component.
        B = 4,
        /// The alpha component.
        A = 5,
        /// Integer one.
        OneInt = 6,
        /// Floating-point one.
        OneFloat = 7,
    }
}

/// The swizzle of one texture binding: where each component of the result comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Swizzle {
    /// The first component's source.
    pub x: SwizzleSource,
    /// The second component's source.
    pub y: SwizzleSource,
    /// The third component's source.
    pub z: SwizzleSource,
    /// The fourth component's source.
    pub w: SwizzleSource,
}

impl Swizzle {
    /// Each component from itself.
    pub const IDENTITY: Self = Self {
        x: SwizzleSource::R,
        y: SwizzleSource::G,
        z: SwizzleSource::B,
        w: SwizzleSource::A,
    };

    /// The sources of x, y, z and w, in that order.
    pub const fn sources(self) -> [SwizzleSource; 4] {
        [self.x, self.y, self.z, self.w]
    }
}

/// The swizzle of each texture binding, in binding order, at most [`MAX_TEXTURE_SWIZZLES`].
/// Empty when the render backend swizzles in the view (`Caps::view_swizzle`).
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct TextureSwizzles(Vec<Swizzle>);

impl TextureSwizzles {
    /// The swizzles `swizzles`, or [`KeyError::TooManySwizzles`] past the cap.
    pub fn new(swizzles: Vec<Swizzle>) -> Result<Self, KeyError> {
        if swizzles.len() > MAX_TEXTURE_SWIZZLES {
            return Err(KeyError::TooManySwizzles {
                count: swizzles.len(),
            });
        }
        Ok(Self(swizzles))
    }

    /// The swizzles, in binding order.
    pub fn as_slice(&self) -> &[Swizzle] {
        &self.0
    }
}

/// A set of attribute words, one bit per word of the guest's attribute space.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct AttributeMask([u64; ATTRIBUTE_MASK_LANES]);

impl AttributeMask {
    /// No words.
    pub const EMPTY: Self = Self([0; ATTRIBUTE_MASK_LANES]);

    /// The mask with these lanes; lane `i` bit `b` is attribute word `64 * i + b`.
    pub const fn from_lanes(lanes: [u64; ATTRIBUTE_MASK_LANES]) -> Self {
        Self(lanes)
    }

    /// The lanes; lane `i` bit `b` is attribute word `64 * i + b`.
    pub const fn lanes(self) -> [u64; ATTRIBUTE_MASK_LANES] {
        self.0
    }

    /// The lane index and bit of `word`.
    const fn place(word: u8) -> (usize, u32) {
        (word as usize / 64, word as u32 % 64)
    }

    /// The mask with `word` added.
    #[must_use]
    pub const fn with(self, word: u8) -> Self {
        let (lane, bit) = Self::place(word);
        let mut lanes = self.0;
        lanes[lane] |= 1 << bit;
        Self(lanes)
    }

    /// Whether `word` is in the mask.
    pub const fn contains(self, word: u8) -> bool {
        let (lane, bit) = Self::place(word);
        self.0[lane] & (1 << bit) != 0
    }
}

/// The render backend's `EmulationFlags` (render-backend, "Fixed-function state the host
/// lacks"), mirrored field by field because the shader crate cannot see the render crate's
/// types. The GPU crate copies each flag in (slice G18), and [`EmulationKey::MIRRORED`] names
/// the flag each field mirrors, so a test there can check that every flag is copied.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "mirrors the render crate's EmulationFlags field by field, as the spec requires"
)]
pub struct EmulationKey {
    /// `ALPHA_TEST`: the comparison the fragment shader's discard makes, present exactly when
    /// the flag is set (spec: "alpha test function ... the reference value is not in the key").
    pub alpha_test: Option<CompareFunction>,
    /// `DEPTH_RANGE_REWRITE`: the vertex shader rewrites z from the -1..1 depth range.
    pub depth_range_rewrite: bool,
    /// `PRIMITIVE_RESTART_REWRITE`: restart indices are handled because the host cannot turn
    /// restart off.
    pub primitive_restart_rewrite: bool,
    /// `ATTRIBUTE_FETCH`: attributes past the host's limit are fetched from a storage buffer.
    pub attribute_fetch: bool,
    /// `TYPED_OUTPUTS`: the fragment shader's outputs are typed to match integer targets.
    pub typed_outputs: bool,
}

/// Reads whether one mirrored flag is set in a key.
pub type FlagReader = fn(&EmulationKey) -> bool;

impl EmulationKey {
    /// Each field with the name of the render crate's `EmulationFlags` constant it mirrors, in
    /// field order, and how to read whether it is set.
    pub const MIRRORED: &'static [(&'static str, FlagReader)] = &[
        ("ALPHA_TEST", |key| key.alpha_test.is_some()),
        ("DEPTH_RANGE_REWRITE", |key| key.depth_range_rewrite),
        ("PRIMITIVE_RESTART_REWRITE", |key| {
            key.primitive_restart_rewrite
        }),
        ("ATTRIBUTE_FETCH", |key| key.attribute_fetch),
        ("TYPED_OUTPUTS", |key| key.typed_outputs),
    ];
}

/// The part of 3D engine state a translated shader is specialised on (see the module docs).
///
/// `Default` gives Eden's value-initialised `RuntimeInfo` for the fields it has (every type
/// `Float`, no dual-source blending, no alpha test) and zero or empty for the rest.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct VariantKey {
    /// The component type of each vertex attribute, by attribute index.
    pub vertex_attributes: [AttributeType; VERTEX_ATTRIBUTES],
    /// The component type of each colour target, by target index.
    pub color_targets: [ColorType; COLOR_TARGETS],
    /// Whether dual-source blending is on.
    pub dual_source_blend: bool,
    /// The attribute words the next stage reads.
    pub next_stage_reads: AttributeMask,
    /// A compute program's workgroup size, x, y and z.
    pub workgroup_size: [u32; 3],
    /// A compute program's shared-memory size in bytes.
    pub shared_memory_size: u32,
    /// The render backend's emulation flags.
    pub emulation: EmulationKey,
    /// Each texture binding's swizzle, when the host has no view swizzle.
    pub texture_swizzles: TextureSwizzles,
}

/// A field of the encoded key, as errors name it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KeyField {
    /// The version byte.
    Version,
    /// The type of a vertex attribute.
    VertexAttribute(u8),
    /// The type of a colour target.
    ColorTarget(u8),
    /// The dual-source blending switch.
    DualSourceBlend,
    /// The mask of attribute words the next stage reads.
    NextStageReads,
    /// The workgroup size along an axis (0 is x).
    WorkgroupSize(u8),
    /// The shared-memory size.
    SharedMemorySize,
    /// The alpha test.
    AlphaTest,
    /// The depth-range rewrite switch.
    DepthRangeRewrite,
    /// The primitive-restart rewrite switch.
    PrimitiveRestartRewrite,
    /// The attribute fetch switch.
    AttributeFetch,
    /// The typed outputs switch.
    TypedOutputs,
    /// The texture swizzle count.
    SwizzleCount,
    /// A component of a texture swizzle (0 is x).
    Swizzle {
        /// The swizzle's index.
        binding: u8,
        /// The component.
        component: u8,
    },
}

impl Display for KeyField {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Version => f.write_str("the version"),
            Self::VertexAttribute(index) => write!(f, "the type of vertex attribute {index}"),
            Self::ColorTarget(index) => write!(f, "the type of colour target {index}"),
            Self::DualSourceBlend => f.write_str("the dual-source blending switch"),
            Self::NextStageReads => f.write_str("the attributes the next stage reads"),
            Self::WorkgroupSize(axis) => write!(f, "the workgroup size along axis {axis}"),
            Self::SharedMemorySize => f.write_str("the shared-memory size"),
            Self::AlphaTest => f.write_str("the alpha test"),
            Self::DepthRangeRewrite => f.write_str("the depth-range rewrite switch"),
            Self::PrimitiveRestartRewrite => f.write_str("the primitive-restart rewrite switch"),
            Self::AttributeFetch => f.write_str("the attribute fetch switch"),
            Self::TypedOutputs => f.write_str("the typed outputs switch"),
            Self::SwizzleCount => f.write_str("the texture swizzle count"),
            Self::Swizzle { binding, component } => {
                write!(f, "component {component} of texture swizzle {binding}")
            }
        }
    }
}

/// Why a variant key could not be built or decoded.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum KeyError {
    /// The bytes end inside a field.
    #[error("the variant key ends after {len} bytes, inside {field}")]
    Truncated {
        /// The field the bytes end in.
        field: KeyField,
        /// How many bytes there are.
        len: usize,
    },
    /// The version is not the one this build reads.
    #[error(
        "variant key version {found} is not supported; this build reads version {expected}",
        expected = VARIANT_KEY_VERSION
    )]
    Version {
        /// The version found.
        found: u8,
    },
    /// A field holds a code outside its set.
    #[error("{field} has the invalid value {value}")]
    InvalidValue {
        /// The field.
        field: KeyField,
        /// The value.
        value: u8,
    },
    /// More texture swizzles than [`MAX_TEXTURE_SWIZZLES`].
    #[error(
        "{count} texture swizzles are more than the {max} a variant key holds",
        max = MAX_TEXTURE_SWIZZLES
    )]
    TooManySwizzles {
        /// How many there are.
        count: usize,
    },
    /// Bytes follow the end of the key.
    #[error("the variant key ends at byte {end}, but {len} bytes were given")]
    TrailingBytes {
        /// Where the key ends.
        end: usize,
        /// How many bytes were given.
        len: usize,
    },
}

/// Reads an encoded key field by field.
struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl Reader<'_> {
    fn take<const N: usize>(&mut self, field: KeyField) -> Result<[u8; N], KeyError> {
        let truncated = KeyError::Truncated {
            field,
            len: self.bytes.len(),
        };
        let end = self.pos.checked_add(N).ok_or(truncated)?;
        let taken = self
            .bytes
            .get(self.pos..end)
            .and_then(|slice| <[u8; N]>::try_from(slice).ok())
            .ok_or(truncated)?;
        self.pos = end;
        Ok(taken)
    }

    fn byte(&mut self, field: KeyField) -> Result<u8, KeyError> {
        let [byte] = self.take::<1>(field)?;
        Ok(byte)
    }

    fn coded<T>(&mut self, field: KeyField, from_code: fn(u8) -> Option<T>) -> Result<T, KeyError> {
        let value = self.byte(field)?;
        from_code(value).ok_or(KeyError::InvalidValue { field, value })
    }

    fn switch(&mut self, field: KeyField) -> Result<bool, KeyError> {
        self.coded(field, |code| match code {
            0 => Some(false),
            1 => Some(true),
            _ => None,
        })
    }

    fn u32(&mut self, field: KeyField) -> Result<u32, KeyError> {
        Ok(u32::from_le_bytes(self.take(field)?))
    }

    fn u64(&mut self, field: KeyField) -> Result<u64, KeyError> {
        Ok(u64::from_le_bytes(self.take(field)?))
    }
}

/// The index `index` of an array of at most 256 entries, as errors name it.
fn index_u8(index: usize) -> u8 {
    // Every array indexed here (attributes, targets, axes, swizzles, components) has at most
    // 256 entries, so this never saturates.
    u8::try_from(index).unwrap_or(u8::MAX)
}

impl VariantKey {
    /// The key as bytes (see the module docs' table).
    pub fn encode(&self) -> Vec<u8> {
        let swizzles = self.texture_swizzles.as_slice();
        let mut out = Vec::with_capacity(VARIANT_KEY_FIXED_SIZE + swizzles.len() * SWIZZLE_SIZE);
        out.push(VARIANT_KEY_VERSION);
        out.extend(self.vertex_attributes.iter().map(|ty| ty.code()));
        out.extend(self.color_targets.iter().map(|ty| ty.code()));
        out.push(u8::from(self.dual_source_blend));
        for lane in self.next_stage_reads.lanes() {
            out.extend_from_slice(&lane.to_le_bytes());
        }
        for size in self.workgroup_size {
            out.extend_from_slice(&size.to_le_bytes());
        }
        out.extend_from_slice(&self.shared_memory_size.to_le_bytes());
        let emulation = &self.emulation;
        out.push(
            emulation
                .alpha_test
                .map_or(0, |function| function.code() + 1),
        );
        out.push(u8::from(emulation.depth_range_rewrite));
        out.push(u8::from(emulation.primitive_restart_rewrite));
        out.push(u8::from(emulation.attribute_fetch));
        out.push(u8::from(emulation.typed_outputs));
        // `TextureSwizzles` holds at most `MAX_TEXTURE_SWIZZLES`, which fits a byte (asserted
        // above), so this never saturates.
        out.push(u8::try_from(swizzles.len()).unwrap_or(u8::MAX));
        for swizzle in swizzles {
            out.extend(swizzle.sources().map(SwizzleSource::code));
        }
        out
    }

    /// The key `bytes` encode, checked in the order the module docs give.
    pub fn decode(bytes: &[u8]) -> Result<Self, KeyError> {
        let mut reader = Reader { bytes, pos: 0 };
        let found = reader.byte(KeyField::Version)?;
        if found != VARIANT_KEY_VERSION {
            return Err(KeyError::Version { found });
        }
        let mut key = Self::default();
        for (index, ty) in key.vertex_attributes.iter_mut().enumerate() {
            *ty = reader.coded(
                KeyField::VertexAttribute(index_u8(index)),
                AttributeType::from_code,
            )?;
        }
        for (index, ty) in key.color_targets.iter_mut().enumerate() {
            *ty = reader.coded(KeyField::ColorTarget(index_u8(index)), ColorType::from_code)?;
        }
        key.dual_source_blend = reader.switch(KeyField::DualSourceBlend)?;
        let mut lanes = [0; ATTRIBUTE_MASK_LANES];
        for lane in &mut lanes {
            *lane = reader.u64(KeyField::NextStageReads)?;
        }
        key.next_stage_reads = AttributeMask::from_lanes(lanes);
        for (axis, size) in key.workgroup_size.iter_mut().enumerate() {
            *size = reader.u32(KeyField::WorkgroupSize(index_u8(axis)))?;
        }
        key.shared_memory_size = reader.u32(KeyField::SharedMemorySize)?;
        key.emulation.alpha_test = reader.coded(KeyField::AlphaTest, |code| match code {
            0 => Some(None),
            code => CompareFunction::from_code(code - 1).map(Some),
        })?;
        key.emulation.depth_range_rewrite = reader.switch(KeyField::DepthRangeRewrite)?;
        key.emulation.primitive_restart_rewrite =
            reader.switch(KeyField::PrimitiveRestartRewrite)?;
        key.emulation.attribute_fetch = reader.switch(KeyField::AttributeFetch)?;
        key.emulation.typed_outputs = reader.switch(KeyField::TypedOutputs)?;
        let count = usize::from(reader.byte(KeyField::SwizzleCount)?);
        if count > MAX_TEXTURE_SWIZZLES {
            return Err(KeyError::TooManySwizzles { count });
        }
        let mut swizzles = Vec::with_capacity(count);
        for binding in 0..count {
            let mut sources = [SwizzleSource::Zero; 4];
            for (component, source) in sources.iter_mut().enumerate() {
                *source = reader.coded(
                    KeyField::Swizzle {
                        binding: index_u8(binding),
                        component: index_u8(component),
                    },
                    SwizzleSource::from_code,
                )?;
            }
            let [x, y, z, w] = sources;
            swizzles.push(Swizzle { x, y, z, w });
        }
        key.texture_swizzles = TextureSwizzles::new(swizzles)?;
        if reader.pos != bytes.len() {
            return Err(KeyError::TrailingBytes {
                end: reader.pos,
                len: bytes.len(),
            });
        }
        Ok(key)
    }
}

#[cfg(test)]
mod tests;
