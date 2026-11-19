//! Samplers: decoding a sampler pool entry (`TEXSAMP0` to `TEXSAMP7` of
//! `classes/clc797tex.h`) into a [`SamplerDesc`], and one host sampler per distinct entry
//! (render-backend spec, "Resources and who owns the caches": "Sampler pool decoding →
//! `SamplerDesc`, dedup by value").
//!
//! # Sources
//!
//! Paths are relative to the reference folder (`tools/refs-path.sh`), at the commits pinned in
//! `docs/references.md`; Eden is GPL-3.0-or-later. Fields are read through
//! [`crate::classes::clc797tex`]; what they mean follows Eden's `Sampler::Sampler`
//! (`eden/src/video_core/renderer_vulkan/vk_texture_cache.cpp` @ cf654052), its
//! `MaxwellToVK::Sampler` functions (`maxwell_to_vk.cpp`) and `TSCEntry`
//! (`eden/src/video_core/textures/texture.h`):
//!
//! - filters: point is nearest, linear is linear; mip `NONE` filters as nearest and clamps the
//!   level of detail to 0 to 0.25, as Eden does;
//! - address modes: `CLAMP_OGL` is clamp to edge under a point magnification filter and clamp to
//!   border under a linear one; the three mirror-once modes are mirror clamp to edge;
//! - levels of detail: the clamps over 256 (fixed point 4.8), the bias a signed 5.8 value over
//!   256;
//! - the border colour: the four float words, as custom;
//! - depth compare: the comparison when `DEPTH_COMPARE` is set.
//!
//! # Divergences from Eden
//!
//! - **Anisotropy follows the header's ratios** (`ANISO_1_TO_1` to `ANISO_16_TO_1`: 1, 2, 4, 6,
//!   8, 10, 12, 16), clamped to `Limits::max_sampler_anisotropy`; Eden takes `1 << field`
//!   (so 6 to 1 is 8) and adds its own settings. `tests::anisotropy_follows_the_header_ratios`.
//! - **No sRGB border colour**: Eden converts the sRGB border bytes when `S_R_G_B_CONVERSION` is
//!   set; the float border words are used here.
//! - **Undefined values are refused** (a magnification filter of 0 or a VCAA mode, a minification
//!   or mip filter of 0): Eden asserts. The anisotropic minification filter is linear, a
//!   **guess** from its name. `tests::undefined_sampler_fields_are_refused`.
//!
//! # Dedup
//!
//! [`Samplers`] keeps one host sampler per distinct eight words, up to [`MAX_SAMPLERS`] (a
//! **guess** bounding what a guest can make the host hold; neither reference caps it). A new
//! entry past the cap is refused ([`DrawError::SamplersFull`]). Samplers live as long as the
//! table.

use std::collections::HashMap;
use std::sync::Arc;

use starquake_render::resource::{
    AddressMode, BorderColor, CompareFunction, FilterMode, ReductionMode, SamplerDesc,
};
use starquake_render::{Backend, FenceValue, ResourceId, SamplerId};

use super::DrawError;
use crate::classes::clc797tex as tex;

/// Words in a sampler pool entry (`TEXSAMP0` to `TEXSAMP7`).
pub const SAMPLER_WORDS: usize = 8;

/// Bytes in a sampler pool entry.
pub const SAMPLER_BYTES: u64 = SAMPLER_WORDS as u64 * 4;

/// The most distinct samplers [`Samplers`] holds (a guess; module docs).
pub const MAX_SAMPLERS: usize = 4096;

/// Fixed-point scale of the level-of-detail fields (Eden's `/ 256.0f`).
const LOD_SCALE: f32 = 256.0;

/// Mip `NONE`'s largest level of detail (Eden's `max_lod = 0.25f`).
const NO_MIP_MAX_LOD: f32 = 0.25;

/// A filter from the point and linear values of a filter field.
fn filter(value: u32, point: u32, linear: u32, extra_linear: Option<u32>) -> Option<FilterMode> {
    if value == point {
        Some(FilterMode::Nearest)
    } else if value == linear || Some(value) == extra_linear {
        Some(FilterMode::Linear)
    } else {
        None
    }
}

/// An address mode from an `ADDRESS_*` field (the three share their values).
fn address(value: u32, mag_linear: bool) -> AddressMode {
    match value {
        tex::TEXSAMP0_ADDRESS_U_WRAP => AddressMode::Repeat,
        tex::TEXSAMP0_ADDRESS_U_MIRROR => AddressMode::MirrorRepeat,
        tex::TEXSAMP0_ADDRESS_U_CLAMP_OGL if mag_linear => AddressMode::ClampToBorder,
        tex::TEXSAMP0_ADDRESS_U_CLAMP_TO_EDGE | tex::TEXSAMP0_ADDRESS_U_CLAMP_OGL => {
            AddressMode::ClampToEdge
        }
        tex::TEXSAMP0_ADDRESS_U_BORDER => AddressMode::ClampToBorder,
        // The three mirror-once modes; the field is three bits, so nothing else reaches here.
        _ => AddressMode::MirrorClampToEdge,
    }
}

/// A depth comparison from `DEPTH_COMPARE_FUNC` (three bits, every value named).
fn compare(value: u32) -> CompareFunction {
    match value {
        tex::TEXSAMP0_DEPTH_COMPARE_FUNC_ZC_NEVER => CompareFunction::Never,
        tex::TEXSAMP0_DEPTH_COMPARE_FUNC_ZC_LESS => CompareFunction::Less,
        tex::TEXSAMP0_DEPTH_COMPARE_FUNC_ZC_EQUAL => CompareFunction::Equal,
        tex::TEXSAMP0_DEPTH_COMPARE_FUNC_ZC_LEQUAL => CompareFunction::LessEqual,
        tex::TEXSAMP0_DEPTH_COMPARE_FUNC_ZC_GREATER => CompareFunction::Greater,
        tex::TEXSAMP0_DEPTH_COMPARE_FUNC_ZC_NOTEQUAL => CompareFunction::NotEqual,
        tex::TEXSAMP0_DEPTH_COMPARE_FUNC_ZC_GEQUAL => CompareFunction::GreaterEqual,
        _ => CompareFunction::Always,
    }
}

/// The anisotropy ratio of `MAX_ANISOTROPY` (three bits, every value named).
fn anisotropy(value: u32) -> u32 {
    match value {
        tex::TEXSAMP0_MAX_ANISOTROPY_ANISO_1_TO_1 => 1,
        tex::TEXSAMP0_MAX_ANISOTROPY_ANISO_2_TO_1 => 2,
        tex::TEXSAMP0_MAX_ANISOTROPY_ANISO_4_TO_1 => 4,
        tex::TEXSAMP0_MAX_ANISOTROPY_ANISO_6_TO_1 => 6,
        tex::TEXSAMP0_MAX_ANISOTROPY_ANISO_8_TO_1 => 8,
        tex::TEXSAMP0_MAX_ANISOTROPY_ANISO_10_TO_1 => 10,
        tex::TEXSAMP0_MAX_ANISOTROPY_ANISO_12_TO_1 => 12,
        _ => 16,
    }
}

/// A level-of-detail clamp as a float.
#[expect(
    clippy::cast_precision_loss,
    reason = "the fields are 12 bits, exact in an f32"
)]
fn lod(value: u32) -> f32 {
    value as f32 / LOD_SCALE
}

/// `MIP_LOD_BIAS`, a signed 13-bit value over 256 (Eden's `LodBias`).
#[expect(
    clippy::cast_precision_loss,
    reason = "a 13-bit value is exact in an f32"
)]
fn lod_bias(field: u32) -> f32 {
    let width = tex::TEXSAMP1_MIP_LOD_BIAS.width();
    let sign = 1u32 << (width - 1);
    let signed = i64::from(field ^ sign) - i64::from(sign);
    signed as f32 / LOD_SCALE
}

/// Decodes a sampler pool entry (module docs); `max_anisotropy` is the device's limit.
pub fn decode(words: &[u32; SAMPLER_WORDS], max_anisotropy: u32) -> Result<SamplerDesc, DrawError> {
    let [w0, w1, w2, _w3, w4, w5, w6, w7] = *words;
    let mag_raw = tex::TEXSAMP1_MAG_FILTER.get(w1);
    let mag_filter = filter(
        mag_raw,
        tex::TEXSAMP1_MAG_FILTER_MAG_POINT,
        tex::TEXSAMP1_MAG_FILTER_MAG_LINEAR,
        None,
    )
    .ok_or(DrawError::Sampler {
        field: "MAG_FILTER",
        value: mag_raw,
    })?;
    let minify_raw = tex::TEXSAMP1_MIN_FILTER.get(w1);
    let minify = filter(
        minify_raw,
        tex::TEXSAMP1_MIN_FILTER_MIN_POINT,
        tex::TEXSAMP1_MIN_FILTER_MIN_LINEAR,
        Some(tex::TEXSAMP1_MIN_FILTER_MIN_ANISO),
    )
    .ok_or(DrawError::Sampler {
        field: "MIN_FILTER",
        value: minify_raw,
    })?;
    let mipmap_raw = tex::TEXSAMP1_MIP_FILTER.get(w1);
    let no_mip = mipmap_raw == tex::TEXSAMP1_MIP_FILTER_MIP_NONE;
    let mipmap = if no_mip {
        FilterMode::Nearest
    } else {
        filter(
            mipmap_raw,
            tex::TEXSAMP1_MIP_FILTER_MIP_POINT,
            tex::TEXSAMP1_MIP_FILTER_MIP_LINEAR,
            None,
        )
        .ok_or(DrawError::Sampler {
            field: "MIP_FILTER",
            value: mipmap_raw,
        })?
    };
    let mag_linear = mag_filter == FilterMode::Linear;
    let (min_lod, max_lod) = if no_mip {
        (0.0, NO_MIP_MAX_LOD)
    } else {
        (
            lod(tex::TEXSAMP2_MIN_LOD_CLAMP.get(w2)),
            lod(tex::TEXSAMP2_MAX_LOD_CLAMP.get(w2)),
        )
    };
    let reduction = match tex::TEXSAMP1_REDUCTION_FILTER.get(w1) {
        tex::TEXSAMP1_REDUCTION_FILTER_RED_MINIMUM => ReductionMode::Min,
        tex::TEXSAMP1_REDUCTION_FILTER_RED_MAXIMUM => ReductionMode::Max,
        tex::TEXSAMP1_REDUCTION_FILTER_RED_NONE => ReductionMode::Average,
        other => {
            return Err(DrawError::Sampler {
                field: "REDUCTION_FILTER",
                value: other,
            });
        }
    };
    Ok(SamplerDesc {
        mag_filter,
        min_filter: minify,
        mip_filter: mipmap,
        address: [
            address(tex::TEXSAMP0_ADDRESS_U.get(w0), mag_linear),
            address(tex::TEXSAMP0_ADDRESS_V.get(w0), mag_linear),
            address(tex::TEXSAMP0_ADDRESS_P.get(w0), mag_linear),
        ],
        border_color: BorderColor::Custom([
            f32::from_bits(w4),
            f32::from_bits(w5),
            f32::from_bits(w6),
            f32::from_bits(w7),
        ]),
        lod_bias: lod_bias(tex::TEXSAMP1_MIP_LOD_BIAS.get(w1)),
        min_lod,
        max_lod,
        max_anisotropy: anisotropy(tex::TEXSAMP0_MAX_ANISOTROPY.get(w0)).min(max_anisotropy.max(1)),
        compare: tex::TEXSAMP0_DEPTH_COMPARE
            .is_set(w0)
            .then(|| compare(tex::TEXSAMP0_DEPTH_COMPARE_FUNC.get(w0))),
        reduction,
    })
}

/// One host sampler per distinct pool entry (module docs, "Dedup").
pub struct Samplers {
    backend: Arc<dyn Backend>,
    by_words: HashMap<[u32; SAMPLER_WORDS], SamplerId>,
    limit: usize,
}

impl core::fmt::Debug for Samplers {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Samplers")
            .field("len", &self.by_words.len())
            .finish_non_exhaustive()
    }
}

impl Samplers {
    /// An empty table making samplers through `backend`.
    pub fn new(backend: Arc<dyn Backend>) -> Self {
        Self::with_limit(backend, MAX_SAMPLERS)
    }

    pub(super) fn with_limit(backend: Arc<dyn Backend>, limit: usize) -> Self {
        Self {
            backend,
            by_words: HashMap::new(),
            limit,
        }
    }

    /// How many host samplers the table holds.
    pub fn len(&self) -> usize {
        self.by_words.len()
    }

    /// Whether the table holds none.
    pub fn is_empty(&self) -> bool {
        self.by_words.is_empty()
    }

    /// The host sampler for `words`, made on first use. Errors leave the table as it was.
    pub fn get(&mut self, words: &[u32; SAMPLER_WORDS]) -> Result<SamplerId, DrawError> {
        if let Some(&id) = self.by_words.get(words) {
            return Ok(id);
        }
        let desc = decode(words, self.backend.caps().limits.max_sampler_anisotropy)?;
        if self.by_words.len() >= self.limit {
            return Err(DrawError::SamplersFull { limit: self.limit });
        }
        let id = self
            .backend
            .create_sampler(&desc)
            .map_err(DrawError::Render)?;
        self.by_words.insert(*words, id);
        Ok(id)
    }

    /// Destroys every host sampler the table made, after `fence`, in id order, and empties
    /// it (the draw path's teardown; `Draws`' `Drop`).
    pub fn destroy_all(&mut self, fence: FenceValue) {
        let mut ids: Vec<_> = self.by_words.drain().map(|(_, id)| id).collect();
        ids.sort_unstable();
        for id in ids {
            self.backend.destroy(ResourceId::Sampler(id), fence);
        }
    }
}
