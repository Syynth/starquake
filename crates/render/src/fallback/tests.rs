//! `host_format` against the spec's table of known conversions and Eden's list of alternatives,
//! with synthetic capability tables.
//!
//! The expected stand-ins are written out here, format by format, from the table in the parent
//! module's docs (the spec's rows and Eden's lists); nothing here asks the shipping code what
//! it would choose.

use super::{Conversion, HostFormat, features_for_usage, host_format};
use crate::caps::{FormatFeatures, FormatTable};
use crate::error::RenderError;
use crate::format::{Compression, FormatAspect, PixelFormat};
use crate::resource::TextureUsage;
use PixelFormat as F;

/// Every guest format that has stand-ins, with them in order and their conversion.
const STAND_INS: [(PixelFormat, &[PixelFormat], Conversion); 37] = [
    // The spec's first row: ASTC, all block sizes, to RGBA8 of the same encoding.
    (
        F::Astc2d4x4Unorm,
        &[F::A8B8G8R8Unorm],
        Conversion::AstcDecode,
    ),
    (
        F::Astc2d5x4Unorm,
        &[F::A8B8G8R8Unorm],
        Conversion::AstcDecode,
    ),
    (
        F::Astc2d5x5Unorm,
        &[F::A8B8G8R8Unorm],
        Conversion::AstcDecode,
    ),
    (
        F::Astc2d6x5Unorm,
        &[F::A8B8G8R8Unorm],
        Conversion::AstcDecode,
    ),
    (
        F::Astc2d6x6Unorm,
        &[F::A8B8G8R8Unorm],
        Conversion::AstcDecode,
    ),
    (
        F::Astc2d8x5Unorm,
        &[F::A8B8G8R8Unorm],
        Conversion::AstcDecode,
    ),
    (
        F::Astc2d8x6Unorm,
        &[F::A8B8G8R8Unorm],
        Conversion::AstcDecode,
    ),
    (
        F::Astc2d8x8Unorm,
        &[F::A8B8G8R8Unorm],
        Conversion::AstcDecode,
    ),
    (
        F::Astc2d10x5Unorm,
        &[F::A8B8G8R8Unorm],
        Conversion::AstcDecode,
    ),
    (
        F::Astc2d10x6Unorm,
        &[F::A8B8G8R8Unorm],
        Conversion::AstcDecode,
    ),
    (
        F::Astc2d10x8Unorm,
        &[F::A8B8G8R8Unorm],
        Conversion::AstcDecode,
    ),
    (
        F::Astc2d10x10Unorm,
        &[F::A8B8G8R8Unorm],
        Conversion::AstcDecode,
    ),
    (
        F::Astc2d12x10Unorm,
        &[F::A8B8G8R8Unorm],
        Conversion::AstcDecode,
    ),
    (
        F::Astc2d12x12Unorm,
        &[F::A8B8G8R8Unorm],
        Conversion::AstcDecode,
    ),
    (F::Astc2d4x4Srgb, &[F::A8B8G8R8Srgb], Conversion::AstcDecode),
    (F::Astc2d5x4Srgb, &[F::A8B8G8R8Srgb], Conversion::AstcDecode),
    (F::Astc2d5x5Srgb, &[F::A8B8G8R8Srgb], Conversion::AstcDecode),
    (F::Astc2d6x5Srgb, &[F::A8B8G8R8Srgb], Conversion::AstcDecode),
    (F::Astc2d6x6Srgb, &[F::A8B8G8R8Srgb], Conversion::AstcDecode),
    (F::Astc2d8x5Srgb, &[F::A8B8G8R8Srgb], Conversion::AstcDecode),
    (F::Astc2d8x6Srgb, &[F::A8B8G8R8Srgb], Conversion::AstcDecode),
    (F::Astc2d8x8Srgb, &[F::A8B8G8R8Srgb], Conversion::AstcDecode),
    (
        F::Astc2d10x5Srgb,
        &[F::A8B8G8R8Srgb],
        Conversion::AstcDecode,
    ),
    (
        F::Astc2d10x6Srgb,
        &[F::A8B8G8R8Srgb],
        Conversion::AstcDecode,
    ),
    (
        F::Astc2d10x8Srgb,
        &[F::A8B8G8R8Srgb],
        Conversion::AstcDecode,
    ),
    (
        F::Astc2d10x10Srgb,
        &[F::A8B8G8R8Srgb],
        Conversion::AstcDecode,
    ),
    (
        F::Astc2d12x10Srgb,
        &[F::A8B8G8R8Srgb],
        Conversion::AstcDecode,
    ),
    (
        F::Astc2d12x12Srgb,
        &[F::A8B8G8R8Srgb],
        Conversion::AstcDecode,
    ),
    // The spec's second row.
    (F::D24UnormS8Uint, &[F::D32FloatS8Uint], Conversion::Texel),
    // The spec's third row, made concrete by Eden's alternatives.
    (
        F::S8UintD24Unorm,
        &[F::D24UnormS8Uint, F::D32FloatS8Uint],
        Conversion::Texel,
    ),
    (
        F::S8Uint,
        &[F::D24UnormS8Uint, F::D32FloatS8Uint],
        Conversion::Texel,
    ),
    (F::X8D24Unorm, &[F::D32Float], Conversion::Texel),
    (F::B5G6R5Unorm, &[F::R5G6B5Unorm], Conversion::Texel),
    (
        F::R32G32B32Float,
        &[F::R32G32B32A32Float],
        Conversion::Texel,
    ),
    (
        F::B10G11R11Float,
        &[F::R16G16B16A16Float],
        Conversion::Texel,
    ),
    // The two formats Eden's `SurfaceFormat` gives another format on every device.
    (F::A1B5G5R5Unorm, &[F::A1R5G5B5Unorm], Conversion::Texel),
    (
        F::R16G16B16X16Float,
        &[F::R16G16B16A16Float],
        Conversion::Texel,
    ),
];

/// Each format feature with what an error calls it, in the order errors pick the first.
const IN_WORDS: [(FormatFeatures, &str); 6] = [
    (FormatFeatures::SAMPLE, "sampling"),
    (FormatFeatures::FILTER, "filtering"),
    (FormatFeatures::RENDER, "render targets"),
    (FormatFeatures::BLEND, "blending"),
    (FormatFeatures::STORAGE, "storage images"),
    (FormatFeatures::COPY, "copies"),
];

fn stand_ins(guest: PixelFormat) -> (&'static [PixelFormat], Conversion) {
    STAND_INS
        .iter()
        .find(|row| row.0 == guest)
        .map_or((&[], Conversion::None), |row| (row.1, row.2))
}

fn native(format: PixelFormat) -> HostFormat {
    HostFormat {
        format,
        conversion: Conversion::None,
    }
}

fn stood_in(format: PixelFormat, conversion: Conversion) -> HostFormat {
    HostFormat { format, conversion }
}

fn unsupported(format: PixelFormat, usage: &'static str) -> RenderError {
    RenderError::FormatUnsupported { format, usage }
}

/// The table above is the module docs' table: 28 ASTC formats (every one Eden lists, and every
/// ASTC entry of `PixelFormat`) and 9 others, each once, and no stand-in is its own guest.
#[test]
fn the_expected_table_is_well_formed() {
    let astc: Vec<_> = STAND_INS
        .iter()
        .filter(|row| row.2 == Conversion::AstcDecode)
        .collect();
    assert_eq!(astc.len(), 28);
    assert_eq!(STAND_INS.len() - astc.len(), 9);
    for (index, (guest, alternatives, conversion)) in STAND_INS.iter().enumerate() {
        assert!(
            STAND_INS[..index].iter().all(|row| row.0 != *guest),
            "{guest} is listed twice"
        );
        assert!(!alternatives.is_empty(), "{guest}");
        assert!(!alternatives.contains(guest), "{guest}");
        assert_ne!(*conversion, Conversion::None, "{guest}");
    }
    // Every format whose name says ASTC is in the ASTC rows, and nothing else is.
    for format in PixelFormat::ALL {
        let named_astc = format.name().starts_with("ASTC_");
        let listed = astc.iter().any(|row| row.0 == format);
        assert_eq!(named_astc, listed, "{format}");
        if listed {
            let srgb = format.name().ends_with("_SRGB");
            let expected = if srgb {
                F::A8B8G8R8Srgb
            } else {
                F::A8B8G8R8Unorm
            };
            assert_eq!(stand_ins(format).0, [expected], "{format}");
        }
    }
}

#[test]
fn a_format_the_device_has_is_used_as_it_is() {
    let everything = FormatTable::uniform(FormatFeatures::ALL);
    for guest in PixelFormat::ALL {
        for (feature, _) in IN_WORDS {
            assert_eq!(host_format(&everything, guest, feature), Ok(native(guest)));
        }
        assert_eq!(
            host_format(&everything, guest, FormatFeatures::ALL),
            Ok(native(guest))
        );
    }
}

/// A texture that needs nothing of its format gets the guest format, whatever the device has.
#[test]
fn needing_nothing_is_always_native() {
    let nothing = FormatTable::uniform(FormatFeatures::EMPTY);
    for guest in PixelFormat::ALL {
        assert_eq!(
            host_format(&nothing, guest, FormatFeatures::EMPTY),
            Ok(native(guest))
        );
    }
}

/// Every guest format, with every set of needed features, on every device that has or lacks
/// each format of its chain (the guest format, then its stand-ins).
///
/// A format the device "has" reports exactly the needed features and no more; one it "lacks"
/// reports every feature but one of the needed ones, for each of them in turn. The formats
/// outside the chain report everything in one pass and nothing in the other, so a choice
/// that wandered outside the chain would show.
#[test]
fn every_combination() {
    let mut checked = 0_u32;
    for guest in PixelFormat::ALL {
        let (alternatives, conversion) = stand_ins(guest);
        let mut chain = vec![guest];
        chain.extend_from_slice(alternatives);
        for needed_bits in 1..=FormatFeatures::ALL.bits() {
            let needed = FormatFeatures::from_bits(needed_bits).expect("defined bits");
            for (missing, missing_name) in IN_WORDS {
                if !needed.contains(missing) {
                    continue;
                }
                let lacking = FormatFeatures::ALL.difference(missing);
                for has in 0..(1_u32 << chain.len()) {
                    for others in [FormatFeatures::EMPTY, FormatFeatures::ALL] {
                        let mut table = FormatTable::uniform(others);
                        for (position, format) in chain.iter().enumerate() {
                            let present = has & (1 << position) != 0;
                            table.set(*format, if present { needed } else { lacking });
                        }
                        let first = (0..chain.len()).find(|position| has & (1 << position) != 0);
                        let expected = match first {
                            Some(0) => Ok(native(guest)),
                            Some(position) => Ok(stood_in(chain[position], conversion)),
                            None => Err(unsupported(guest, missing_name)),
                        };
                        assert_eq!(
                            host_format(&table, guest, needed),
                            expected,
                            "{guest} needing {needed:?}, chain {chain:?} has {has:#b}, \
                             lacking {missing_name}, others {others:?}"
                        );
                        checked += 1;
                    }
                }
            }
        }
    }
    // 63 needed sets hold 6 x 32 (set, member) pairs; a chain of n formats has 2^n devices,
    // each in two passes. 75 formats have a chain of 1, 35 of 2 and 2 of 3.
    assert_eq!(checked, 6 * 32 * 2 * (75 * 2 + 35 * 4 + 2 * 8));
}

/// What the module docs promise of every stand-in ("no narrower or lossy stand-in"), checked
/// on the shipping lists, for every format. A stand-in is uncompressed, keeps the guest's sRGB
/// encoding, has every aspect the guest has and is colour exactly when the guest is, and has
/// at least the guest's bits per texel. A colour stand-in keeps the channels' numeric type; a
/// depth stand-in may hold normalised depth as float (the spec's "D32 float + S8" row).
#[test]
fn a_stand_in_can_hold_its_guest() {
    let mut pairs = 0_u32;
    for guest in PixelFormat::ALL {
        let (alternatives, _) = super::stand_ins(guest);
        for stand_in in alternatives {
            let pair = format!("{stand_in} for {guest}");
            assert!(!stand_in.is_compressed(), "{pair}");
            assert_eq!(stand_in.is_srgb(), guest.is_srgb(), "{pair}");
            assert!(stand_in.has_depth() || !guest.has_depth(), "{pair}");
            assert!(stand_in.has_stencil() || !guest.has_stencil(), "{pair}");
            let colour = guest.aspect() == FormatAspect::Color;
            assert_eq!(stand_in.aspect() == FormatAspect::Color, colour, "{pair}");
            if colour {
                assert_eq!(stand_in.numeric(), guest.numeric(), "{pair}");
            }
            if !guest.is_compressed() {
                assert!(
                    stand_in.bits_per_block() >= guest.bits_per_block(),
                    "{pair}"
                );
            }
            pairs += 1;
        }
    }
    // 28 ASTC formats and 7 others with one stand-in each, and 2 with two.
    assert_eq!(pairs, 28 + 7 + 2 * 2);
}

/// The error names the first needed feature the device lacks for the guest format, in the
/// order sampling, filtering, render targets, blending, storage images, copies.
#[test]
fn the_error_names_the_first_missing_feature() {
    let guest = F::R8Unorm;
    let nothing = FormatTable::uniform(FormatFeatures::EMPTY);
    for (index, (feature, name)) in IN_WORDS.iter().enumerate() {
        // Needing this feature and every later one: this one is named.
        let needed = IN_WORDS[index..]
            .iter()
            .fold(FormatFeatures::EMPTY, |set, (f, _)| set | *f);
        assert_eq!(
            host_format(&nothing, guest, needed),
            Err(unsupported(guest, name))
        );
        // With this one present and the later ones missing, the next is named.
        let mut table = FormatTable::uniform(FormatFeatures::EMPTY);
        table.set(guest, *feature);
        if let Some((_, next)) = IN_WORDS.get(index + 1) {
            assert_eq!(
                host_format(&table, guest, needed),
                Err(unsupported(guest, next))
            );
        } else {
            assert_eq!(host_format(&table, guest, needed), Ok(native(guest)));
        }
    }
    assert_eq!(
        host_format(
            &nothing,
            guest,
            FormatFeatures::RENDER | FormatFeatures::COPY
        )
        .map_err(|e| e.to_string()),
        Err("R8_UNORM cannot be used for render targets on this device".to_owned())
    );
}

/// The error is about the guest format even when stand-ins were tried: it is the guest format
/// the caller asked for, and its first missing feature that is named.
#[test]
fn the_error_is_about_the_guest_format() {
    let mut table = FormatTable::uniform(FormatFeatures::ALL);
    table.set(F::S8Uint, FormatFeatures::SAMPLE);
    table.set(F::D24UnormS8Uint, FormatFeatures::RENDER);
    table.set(F::D32FloatS8Uint, FormatFeatures::EMPTY);
    assert_eq!(
        host_format(
            &table,
            F::S8Uint,
            FormatFeatures::SAMPLE | FormatFeatures::RENDER | FormatFeatures::COPY
        ),
        Err(unsupported(F::S8Uint, "render targets"))
    );
}

/// A stand-in must have every needed feature, not only the ones the guest format lacks: here
/// D32 float + S8 has the render-target feature D24 + S8 is missing, but cannot be sampled.
#[test]
fn a_stand_in_needs_every_needed_feature() {
    let needed = FormatFeatures::SAMPLE | FormatFeatures::RENDER;
    let mut table = FormatTable::uniform(FormatFeatures::ALL);
    table.set(F::D24UnormS8Uint, FormatFeatures::SAMPLE);
    table.set(F::D32FloatS8Uint, FormatFeatures::RENDER);
    assert_eq!(
        host_format(&table, F::D24UnormS8Uint, needed),
        Err(unsupported(F::D24UnormS8Uint, "render targets"))
    );
    table.set(F::D32FloatS8Uint, needed);
    assert_eq!(
        host_format(&table, F::D24UnormS8Uint, needed),
        Ok(stood_in(F::D32FloatS8Uint, Conversion::Texel))
    );
}

/// A device shaped like wgpu's on an Apple Silicon Mac.
///
/// From the spec's decision 1 table and its "Formats" table: ASTC is native (sampled, filtered
/// and copied, as compressed formats are), D24 + S8 is missing (the spec's guess for Apple
/// GPUs) and D32 float + S8 is there. From the `TextureFormat` list of wgpu-types 30.0.2: it
/// has no three-channel float, no RGBX float and no packed format of 16 bits or fewer, so the
/// nine such guest formats below report nothing. How the wgpu backend reports them is slice
/// R5's to decide; nothing is the plain reading of that list.
#[test]
fn a_device_shaped_like_wgpu_on_a_mac() {
    let compressed = FormatFeatures::SAMPLE | FormatFeatures::FILTER | FormatFeatures::COPY;
    let mut mac = FormatTable::uniform(FormatFeatures::ALL);
    for format in PixelFormat::ALL {
        if format.name().starts_with("ASTC_") {
            mac.set(format, compressed);
        }
    }
    mac.set(F::D24UnormS8Uint, FormatFeatures::EMPTY);
    mac.set(F::S8UintD24Unorm, FormatFeatures::EMPTY);
    for format in [
        F::R32G32B32Float,
        F::R16G16B16X16Float,
        F::R5G6B5Unorm,
        F::B5G6R5Unorm,
        F::A1R5G5B5Unorm,
        F::A1B5G5R5Unorm,
        F::A5B5G5R1Unorm,
        F::A4B4G4R4Unorm,
        F::G4R4Unorm,
    ] {
        mac.set(format, FormatFeatures::EMPTY);
    }

    // The two float formats get their four-channel form, converted.
    let target = features_for_usage(TextureUsage::RENDER_TARGET | TextureUsage::SAMPLED);
    assert_eq!(
        host_format(&mac, F::R32G32B32Float, target),
        Ok(stood_in(F::R32G32B32A32Float, Conversion::Texel))
    );
    assert_eq!(
        host_format(&mac, F::R16G16B16X16Float, target),
        Ok(stood_in(F::R16G16B16A16Float, Conversion::Texel))
    );
    // The packed formats have nothing: where one has a stand-in, it is another packed format.
    for guest in [
        F::R5G6B5Unorm,
        F::B5G6R5Unorm,
        F::A1R5G5B5Unorm,
        F::A1B5G5R5Unorm,
        F::A5B5G5R1Unorm,
        F::A4B4G4R4Unorm,
        F::G4R4Unorm,
    ] {
        assert_eq!(
            host_format(&mac, guest, FormatFeatures::SAMPLE),
            Err(unsupported(guest, "sampling"))
        );
    }

    // ASTC, the console's most used format, is uploaded as it is.
    for guest in [F::Astc2d8x8Unorm, F::Astc2d8x8Srgb, F::Astc2d4x4Unorm] {
        assert_eq!(host_format(&mac, guest, compressed), Ok(native(guest)));
    }
    // D24 + S8, in either byte order, becomes D32 float + S8, converted.
    let depth_target = features_for_usage(
        TextureUsage::RENDER_TARGET | TextureUsage::SAMPLED | TextureUsage::COPY_SRC,
    );
    for guest in [F::D24UnormS8Uint, F::S8UintD24Unorm] {
        assert_eq!(
            host_format(&mac, guest, depth_target),
            Ok(stood_in(F::D32FloatS8Uint, Conversion::Texel))
        );
    }
    // What the device has stays native.
    for guest in [
        F::D32FloatS8Uint,
        F::D32Float,
        F::A8B8G8R8Unorm,
        F::Bc7Unorm,
    ] {
        assert_eq!(
            host_format(&mac, guest, FormatFeatures::SAMPLE),
            Ok(native(guest))
        );
    }
}

/// A device with no ASTC (the spec's platform matrix: AMD has none): every ASTC format is
/// decoded to RGBA8 of its own encoding.
#[test]
fn a_device_without_astc_decodes_every_block_size() {
    let mut desktop = FormatTable::uniform(FormatFeatures::ALL);
    let mut astc = 0;
    for format in PixelFormat::ALL {
        if format.name().starts_with("ASTC_") {
            desktop.set(format, FormatFeatures::EMPTY);
            astc += 1;
        }
    }
    assert_eq!(astc, 28);
    let sampled =
        features_for_usage(TextureUsage::SAMPLED | TextureUsage::COPY_DST) | FormatFeatures::FILTER;
    for format in PixelFormat::ALL {
        if !format.name().starts_with("ASTC_") {
            continue;
        }
        let rgba8 = if format.name().ends_with("_SRGB") {
            F::A8B8G8R8Srgb
        } else {
            F::A8B8G8R8Unorm
        };
        assert_eq!(
            host_format(&desktop, format, sampled),
            Ok(stood_in(rgba8, Conversion::AstcDecode)),
            "{format}"
        );
    }
    // An sRGB block is never decoded into linear RGBA8, nor a linear one into sRGB.
    desktop.set(F::A8B8G8R8Srgb, FormatFeatures::EMPTY);
    assert_eq!(
        host_format(&desktop, F::Astc2d8x8Srgb, sampled),
        Err(unsupported(F::Astc2d8x8Srgb, "sampling"))
    );
    assert_eq!(
        host_format(&desktop, F::Astc2d8x8Unorm, sampled),
        Ok(stood_in(F::A8B8G8R8Unorm, Conversion::AstcDecode))
    );
}

/// Eden decodes every ASTC format unless the device has all of them; here each format is
/// decided on its own report.
#[test]
fn astc_is_decided_per_format() {
    let mut table = FormatTable::uniform(FormatFeatures::ALL);
    table.set(F::Astc2d12x12Unorm, FormatFeatures::EMPTY);
    assert_eq!(
        host_format(&table, F::Astc2d8x8Unorm, FormatFeatures::SAMPLE),
        Ok(native(F::Astc2d8x8Unorm))
    );
    assert_eq!(
        host_format(&table, F::Astc2d12x12Unorm, FormatFeatures::SAMPLE),
        Ok(stood_in(F::A8B8G8R8Unorm, Conversion::AstcDecode))
    );
}

/// The divergences from Eden's alternatives that the module docs list.
#[test]
fn where_this_differs_from_eden() {
    let lacking = |guest: PixelFormat| {
        let mut table = FormatTable::uniform(FormatFeatures::ALL);
        table.set(guest, FormatFeatures::EMPTY);
        table
    };
    // Eden ends X8_D24_UNORM's list with D16_UNORM: narrower, so not offered. With D32 float
    // missing too, D16 being there does not help.
    let mut table = lacking(F::X8D24Unorm);
    table.set(F::D32Float, FormatFeatures::EMPTY);
    assert_eq!(
        host_format(&table, F::X8D24Unorm, FormatFeatures::RENDER),
        Err(unsupported(F::X8D24Unorm, "render targets"))
    );
    // Eden offers A8B8G8R8_SRGB for B10G11R11_FLOAT after R16G16B16A16_FLOAT.
    let mut table = lacking(F::B10G11R11Float);
    table.set(F::R16G16B16A16Float, FormatFeatures::EMPTY);
    assert_eq!(
        host_format(&table, F::B10G11R11Float, FormatFeatures::RENDER),
        Err(unsupported(F::B10G11R11Float, "render targets"))
    );
    // Eden offers R8_UNORM for G4R4_UNORM, and R4G4B4A4 (which `PixelFormat` lacks) for
    // A4B4G4R4_UNORM, and gives A5B5G5R1_UNORM Vulkan's R5G5B5A1 (which it lacks too).
    for guest in [F::G4R4Unorm, F::A4B4G4R4Unorm, F::A5B5G5R1Unorm] {
        assert_eq!(
            host_format(&lacking(guest), guest, FormatFeatures::SAMPLE),
            Err(unsupported(guest, "sampling"))
        );
    }
    // A BC, ETC2 or EAC format the device lacks is decoded by Eden and is an error here: all
    // 14 BC and 10 ETC2 and EAC entries of `PixelFormat`.
    let mut decoded_by_eden = 0;
    for guest in PixelFormat::ALL {
        if matches!(guest.compression(), Compression::Bc | Compression::Etc2) {
            assert_eq!(
                host_format(&lacking(guest), guest, FormatFeatures::SAMPLE),
                Err(unsupported(guest, "sampling"))
            );
            decoded_by_eden += 1;
        }
    }
    assert_eq!(decoded_by_eden, 14 + 10);
    // Stand-ins are one step: D24_UNORM_S8_UINT is S8_UINT's first stand-in, and a device
    // lacking both does not go on to D24's own stand-ins beyond S8_UINT's list. (Here the two
    // lists agree on D32_FLOAT_S8_UINT; R5G6B5_UNORM has no stand-in of its own, so a device
    // lacking it and B5G6R5_UNORM has nothing.)
    let mut table = lacking(F::B5G6R5Unorm);
    table.set(F::R5G6B5Unorm, FormatFeatures::EMPTY);
    assert_eq!(
        host_format(&table, F::B5G6R5Unorm, FormatFeatures::SAMPLE),
        Err(unsupported(F::B5G6R5Unorm, "sampling"))
    );
}

/// Each usage's feature, from `validate`'s documented checks: sampling, storage images, render
/// targets, and copies for either copy usage. Filtering and blending are never implied.
#[test]
fn features_for_each_usage() {
    let rows = [
        (TextureUsage::SAMPLED, FormatFeatures::SAMPLE),
        (TextureUsage::STORAGE, FormatFeatures::STORAGE),
        (TextureUsage::RENDER_TARGET, FormatFeatures::RENDER),
        (TextureUsage::COPY_SRC, FormatFeatures::COPY),
        (TextureUsage::COPY_DST, FormatFeatures::COPY),
    ];
    assert_eq!(
        features_for_usage(TextureUsage::EMPTY),
        FormatFeatures::EMPTY
    );
    for subset in 0..(1_u32 << rows.len()) {
        let mut usage = TextureUsage::EMPTY;
        let mut features = FormatFeatures::EMPTY;
        for (index, (flag, feature)) in rows.iter().enumerate() {
            if subset & (1 << index) != 0 {
                usage = usage | *flag;
                features = features | *feature;
            }
        }
        assert_eq!(features_for_usage(usage), features, "{usage:?}");
    }
    assert_eq!(
        features_for_usage(TextureUsage::ALL),
        FormatFeatures::ALL
            .difference(FormatFeatures::FILTER)
            .difference(FormatFeatures::BLEND)
    );
}

/// What errors call each feature, and that a set of several has no name of its own.
#[test]
fn features_in_words() {
    for (feature, name) in IN_WORDS {
        assert_eq!(feature.in_words(), name);
    }
    assert_eq!(IN_WORDS.len(), FormatFeatures::NAMED.len());
    assert_eq!(FormatFeatures::EMPTY.in_words(), "several uses");
    assert_eq!(
        (FormatFeatures::SAMPLE | FormatFeatures::COPY).in_words(),
        "several uses"
    );
}
