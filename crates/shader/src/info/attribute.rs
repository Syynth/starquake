//! The guest's attribute words, named: a port of Eden's attribute list
//! (`eden/src/shader_recompiler/frontend/ir/attribute.h` @ cf654052, GPL-3.0-or-later), which
//! PR #4588 uses unchanged (the same file in `eden-pr4588` @ e5fb3b97).
//!
//! An attribute word is the byte address an attribute instruction reads divided by four
//! ([`crate::ir::Attribute`]). [`slot_of`] says which interface slot a word belongs to and which
//! of the slot's components it is; the words Eden's list does not name (0 to 23, 187, 233 to
//! 254) belong to no slot, and neither does any word at or past the attribute space's end
//! ([`crate::translate::ATTRIBUTE_SPACE_END`] bytes, word 256: Eden's `BaseInstance = 256` and
//! after are host values its own passes add, never guest attributes).
//!
//! Eden's `IsGeneric` ends at `Generic31X` (156), so its `GenericAttributeIndex` refuses the Y, Z
//! and W of generic 31; its list still names them (`Generic31Y = 157` to `Generic31W = 159`).
//! Here they are generic 31's components, as the list names them (the attribute translator
//! keeps Eden's `IsGeneric` for `IPA`'s rule; `lower` documents the same choice).

use crate::ir::{ATTRIBUTE_COMPONENTS, Attribute, GENERIC_0_X, POSITION_X};

/// `PrimitiveId = 24`.
pub const PRIMITIVE_ID: u16 = 24;
/// `Layer = 25`.
pub const LAYER: u16 = 25;
/// `ViewportIndex = 26`.
pub const VIEWPORT_INDEX: u16 = 26;
/// `PointSize = 27`.
pub const POINT_SIZE: u16 = 27;
/// How many generic attributes there are (`NUM_GENERICS = 32`).
pub const GENERICS: u16 = 32;
/// `Generic31X = 156`, the last word Eden's `IsGeneric` counts.
pub const GENERIC_31_X: u16 = GENERIC_0_X + (GENERICS - 1) * ATTRIBUTE_COMPONENTS;
/// `ColorFrontDiffuseR = 160`, the first legacy colour word.
pub const COLOR_FRONT_DIFFUSE_R: u16 = GENERIC_0_X + GENERICS * ATTRIBUTE_COMPONENTS;
/// The legacy colours: front diffuse, front specular, back diffuse, back specular
/// (`ColorFrontDiffuseR = 160` to `ColorBackSpecularA = 175`).
pub const LEGACY_COLORS: u16 = 4;
/// `ColorBackSpecularA = 175`, the last legacy colour word.
pub const COLOR_BACK_SPECULAR_A: u16 =
    COLOR_FRONT_DIFFUSE_R + LEGACY_COLORS * ATTRIBUTE_COMPONENTS - 1;
/// `ClipDistance0 = 176`.
pub const CLIP_DISTANCE_0: u16 = 176;
/// How many clip distances there are (`ClipDistance0` to `ClipDistance7`).
pub const CLIP_DISTANCES: u16 = 8;
/// `PointSpriteS = 184`; `PointSpriteT` follows it.
pub const POINT_SPRITE_S: u16 = 184;
/// `FogCoordinate = 186`.
pub const FOG_COORDINATE: u16 = 186;
/// `TessellationEvaluationPointU = 188`; `TessellationEvaluationPointV` follows it.
pub const TESSELLATION_POINT_U: u16 = 188;
/// `InstanceId = 190`.
pub const INSTANCE_ID: u16 = 190;
/// `VertexId = 191`.
pub const VERTEX_ID: u16 = 191;
/// `FixedFncTexture0S = 192`, the first fixed-function texture coordinate word.
pub const FIXED_FNC_TEXTURE_0_S: u16 = 192;
/// How many fixed-function texture coordinates there are (`NUM_FIXEDFNCTEXTURE = 10`).
pub const FIXED_FNC_TEXTURES: u16 = 10;
/// `FixedFncTexture9Q = 231`, the last fixed-function texture coordinate word.
pub const FIXED_FNC_TEXTURE_9_Q: u16 =
    FIXED_FNC_TEXTURE_0_S + FIXED_FNC_TEXTURES * ATTRIBUTE_COMPONENTS - 1;
/// `ViewportMask = 232`.
pub const VIEWPORT_MASK: u16 = 232;
/// `FrontFace = 255`.
pub const FRONT_FACE: u16 = 255;

/// Words in a pair slot (the point sprite's S and T, the tessellation point's U and V).
const PAIR: u16 = 2;

/// One input or output of a shader's interface, as the guest names it: an attribute slot (a
/// run of attribute words) or one of a pixel program's outputs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Slot {
    /// `PrimitiveId`.
    PrimitiveId,
    /// `Layer`.
    Layer,
    /// `ViewportIndex`.
    ViewportIndex,
    /// `PointSize`.
    PointSize,
    /// `PositionX` to `PositionW`.
    Position,
    /// Generic attribute `n`, `Generic{n}X` to `Generic{n}W`.
    Generic(u8),
    /// Legacy colour `n`: 0 front diffuse, 1 front specular, 2 back diffuse, 3 back specular,
    /// each R, G, B, A.
    LegacyColor(u8),
    /// `ClipDistance0` to `ClipDistance7`, distance `i` as component `i`.
    ClipDistances,
    /// `PointSpriteS` and `PointSpriteT`.
    PointSprite,
    /// `FogCoordinate`.
    FogCoordinate,
    /// `TessellationEvaluationPointU` and `TessellationEvaluationPointV`.
    TessellationPoint,
    /// `InstanceId`.
    InstanceId,
    /// `VertexId`.
    VertexId,
    /// Fixed-function texture coordinate `n`, S, T, R, Q.
    FixedFncTexture(u8),
    /// `ViewportMask`.
    ViewportMask,
    /// `FrontFace`.
    FrontFace,
    /// Colour target `n` of a pixel program (not an attribute word: `SetFragColor`).
    ColorTarget(u8),
    /// A pixel program's sample mask (`SetSampleMask`).
    SampleMask,
    /// A pixel program's depth (`SetFragDepth`).
    FragDepth,
}

/// Narrows a quotient or remainder that is known to fit in a byte.
fn small(value: u16) -> u8 {
    // Cannot fail: every caller passes a value below 64.
    u8::try_from(value).unwrap_or(u8::MAX)
}

/// The run of four-word slots starting at `first`, `count` long, that `word` is in: the slot's
/// index and the word's component.
fn quad(word: u16, first: u16, count: u16) -> Option<(u8, u8)> {
    let offset = word.checked_sub(first)?;
    let index = offset / ATTRIBUTE_COMPONENTS;
    (index < count).then(|| (small(index), small(offset % ATTRIBUTE_COMPONENTS)))
}

/// The slot attribute word `attribute` belongs to, and which of the slot's components it is;
/// `None` for a word Eden's list does not name, or one past the attribute space.
pub fn slot_of(attribute: Attribute) -> Option<(Slot, u8)> {
    let word = attribute.0;
    let single = |slot| Some((slot, 0));
    match word {
        PRIMITIVE_ID => single(Slot::PrimitiveId),
        LAYER => single(Slot::Layer),
        VIEWPORT_INDEX => single(Slot::ViewportIndex),
        POINT_SIZE => single(Slot::PointSize),
        FOG_COORDINATE => single(Slot::FogCoordinate),
        INSTANCE_ID => single(Slot::InstanceId),
        VERTEX_ID => single(Slot::VertexId),
        VIEWPORT_MASK => single(Slot::ViewportMask),
        FRONT_FACE => single(Slot::FrontFace),
        _ => {
            if let Some((_, component)) = quad(word, POSITION_X, 1) {
                return Some((Slot::Position, component));
            }
            if let Some((index, component)) = quad(word, GENERIC_0_X, GENERICS) {
                return Some((Slot::Generic(index), component));
            }
            if let Some((index, component)) = quad(word, COLOR_FRONT_DIFFUSE_R, LEGACY_COLORS) {
                return Some((Slot::LegacyColor(index), component));
            }
            if let Some((index, component)) = quad(word, FIXED_FNC_TEXTURE_0_S, FIXED_FNC_TEXTURES)
            {
                return Some((Slot::FixedFncTexture(index), component));
            }
            let run = |first: u16, len: u16, slot| {
                let offset = word.checked_sub(first)?;
                (offset < len).then(|| (slot, small(offset)))
            };
            run(CLIP_DISTANCE_0, CLIP_DISTANCES, Slot::ClipDistances)
                .or_else(|| run(POINT_SPRITE_S, PAIR, Slot::PointSprite))
                .or_else(|| run(TESSELLATION_POINT_U, PAIR, Slot::TessellationPoint))
        }
    }
}
