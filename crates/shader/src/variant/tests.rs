//! Variant-key tests: golden bytes, the round trip, every refusal with its exact error, the
//! documented check order, the swizzle cap at its edge, and the code tables written out.

use proptest::prelude::*;

use super::*;
use crate::testing::config;

const GOLDEN_SWIZZLES: [Swizzle; 2] = [
    Swizzle {
        x: SwizzleSource::B,
        y: SwizzleSource::OneFloat,
        z: SwizzleSource::R,
        w: SwizzleSource::OneInt,
    },
    Swizzle {
        x: SwizzleSource::A,
        y: SwizzleSource::G,
        z: SwizzleSource::OneFloat,
        w: SwizzleSource::B,
    },
];

/// A key with distinct non-zero values in every field (as distinct as each field's codes
/// allow: neighbouring entries of an array always differ).
fn golden_key() -> VariantKey {
    let mut key = VariantKey::default();
    for (index, ty) in key.vertex_attributes.iter_mut().enumerate() {
        *ty = AttributeType::ALL[1 + index % 5];
    }
    for (index, ty) in key.color_targets.iter_mut().enumerate() {
        *ty = [ColorType::SignedInt, ColorType::UnsignedInt][index % 2];
    }
    key.dual_source_blend = true;
    key.next_stage_reads = AttributeMask::from_lanes([
        0x0102_0304_0506_0708,
        0x1112_1314_1516_1718,
        0x2122_2324_2526_2728,
        0x3132_3334_3536_3738,
    ]);
    key.workgroup_size = [0x4142_4344, 0x5152_5354, 0x6162_6364];
    key.shared_memory_size = 0x7172_7374;
    key.emulation = EmulationKey {
        alpha_test: Some(CompareFunction::GreaterThanEqual),
        depth_range_rewrite: true,
        primitive_restart_rewrite: true,
        attribute_fetch: true,
        typed_outputs: true,
    };
    key.texture_swizzles = TextureSwizzles::new(GOLDEN_SWIZZLES.to_vec()).expect("two");
    key
}

#[rustfmt::skip]
const GOLDEN_BYTES: [u8; 104] = [
    0x01, // version
    // vertex attribute types
    1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1,
    2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2,
    // colour target types
    1, 2, 1, 2, 1, 2, 1, 2,
    0x01, // dual-source blending
    // next stage's reads, lanes 0 to 3
    0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01,
    0x18, 0x17, 0x16, 0x15, 0x14, 0x13, 0x12, 0x11,
    0x28, 0x27, 0x26, 0x25, 0x24, 0x23, 0x22, 0x21,
    0x38, 0x37, 0x36, 0x35, 0x34, 0x33, 0x32, 0x31,
    // workgroup size x, y, z
    0x44, 0x43, 0x42, 0x41, 0x54, 0x53, 0x52, 0x51, 0x64, 0x63, 0x62, 0x61,
    0x74, 0x73, 0x72, 0x71, // shared-memory size
    0x07, // alpha test: greater or equal (6) plus one
    0x01, 0x01, 0x01, 0x01, // depth range, primitive restart, attribute fetch, typed outputs
    0x02, // swizzle count
    4, 7, 2, 6, // swizzle 0: B, one (float), R, one (int)
    5, 3, 7, 4, // swizzle 1: A, G, one (float), B
];

#[test]
fn layout_constants() {
    assert_eq!(VARIANT_KEY_VERSION, 1);
    assert_eq!(VERTEX_ATTRIBUTES, 32);
    assert_eq!(MAX_TEXTURE_SWIZZLES, 192);
    assert_eq!(ATTRIBUTE_WORDS, 256);
    assert_eq!(ATTRIBUTE_MASK_LANES, 4);
    assert_eq!(SWIZZLE_SIZE, 4);
    assert_eq!(VARIANT_KEY_FIXED_SIZE, 96);
    assert_eq!(VARIANT_KEY_MAX_SIZE, 96 + 192 * 4);
}

#[test]
fn golden_bytes() {
    assert_eq!(golden_key().encode(), GOLDEN_BYTES);
    assert_eq!(VariantKey::decode(&GOLDEN_BYTES), Ok(golden_key()));
}

#[test]
fn the_default_key_decodes_from_zeros() {
    // Hand-built: the version, then every field zero and no swizzles.
    let mut bytes = vec![0; VARIANT_KEY_FIXED_SIZE];
    bytes[0] = 1;
    assert_eq!(VariantKey::decode(&bytes), Ok(VariantKey::default()));
    assert_eq!(VariantKey::default().encode(), bytes);
}

#[test]
fn defaults() {
    // Eden's value-initialised `RuntimeInfo`: every attribute and output type `Float` (the
    // first enumerator), no dual-source blending, no alpha-test function.
    let fresh = VariantKey::default();
    assert_eq!(fresh.vertex_attributes, [AttributeType::Float; 32]);
    assert_eq!(fresh.color_targets, [ColorType::Float; 8]);
    assert!(!fresh.dual_source_blend);
    assert_eq!(fresh.next_stage_reads, AttributeMask::EMPTY);
    assert_eq!(fresh.workgroup_size, [0, 0, 0]);
    assert_eq!(fresh.shared_memory_size, 0);
    assert_eq!(
        fresh.emulation,
        EmulationKey {
            alpha_test: None,
            depth_range_rewrite: false,
            primitive_restart_rewrite: false,
            attribute_fetch: false,
            typed_outputs: false,
        }
    );
    assert_eq!(fresh.texture_swizzles.as_slice().len(), 0);
}

/// Reads one switch of a key.
type Read = fn(&VariantKey) -> bool;

fn any_coded<T: Clone + core::fmt::Debug + 'static>(all: &'static [T]) -> impl Strategy<Value = T> {
    prop::sample::select(all)
}

fn any_swizzle() -> impl Strategy<Value = Swizzle> {
    prop::array::uniform4(any_coded(SwizzleSource::ALL)).prop_map(|[x, y, z, w]| Swizzle {
        x,
        y,
        z,
        w,
    })
}

fn any_key() -> impl Strategy<Value = VariantKey> {
    let emulation = (
        prop::option::of(any_coded(CompareFunction::ALL)),
        prop::array::uniform4(any::<bool>()),
    )
        .prop_map(
            |(alpha_test, [depth, restart, fetch, typed])| EmulationKey {
                alpha_test,
                depth_range_rewrite: depth,
                primitive_restart_rewrite: restart,
                attribute_fetch: fetch,
                typed_outputs: typed,
            },
        );
    (
        prop::array::uniform32(any_coded(AttributeType::ALL)),
        prop::array::uniform8(any_coded(ColorType::ALL)),
        any::<bool>(),
        prop::array::uniform4(any::<u64>()),
        prop::array::uniform3(any::<u32>()),
        any::<u32>(),
        emulation,
        prop::collection::vec(any_swizzle(), 0..=MAX_TEXTURE_SWIZZLES),
    )
        .prop_map(
            |(attributes, colors, dual, lanes, workgroup, shared, emulation, swizzles)| {
                VariantKey {
                    vertex_attributes: attributes,
                    color_targets: colors,
                    dual_source_blend: dual,
                    next_stage_reads: AttributeMask::from_lanes(lanes),
                    workgroup_size: workgroup,
                    shared_memory_size: shared,
                    emulation,
                    texture_swizzles: TextureSwizzles::new(swizzles).expect("within the cap"),
                }
            },
        )
}

proptest! {
    #![proptest_config(config())]

    /// Build, encode, decode: any two reads swapped give a different key (or an error), since
    /// the generated fields take independent values.
    #[test]
    fn keys_round_trip(key in any_key()) {
        let bytes = key.encode();
        prop_assert_eq!(
            bytes.len(),
            VARIANT_KEY_FIXED_SIZE + key.texture_swizzles.as_slice().len() * SWIZZLE_SIZE
        );
        prop_assert_eq!(VariantKey::decode(&bytes), Ok(key));
    }

    /// Arbitrary bytes after a valid version never panic, and anything accepted re-encodes to
    /// the same bytes.
    #[test]
    fn arbitrary_bytes_never_panic(tail in prop::collection::vec(any::<u8>(), 0..VARIANT_KEY_MAX_SIZE)) {
        let mut bytes = vec![VARIANT_KEY_VERSION];
        bytes.extend_from_slice(&tail);
        if let Ok(key) = VariantKey::decode(&bytes) {
            prop_assert_eq!(key.encode(), bytes);
        }
    }
}

#[test]
fn the_version_is_checked_first() {
    assert_eq!(
        VariantKey::decode(&[]),
        Err(KeyError::Truncated {
            field: KeyField::Version,
            len: 0
        })
    );
    // Another version with a truncated body: the version is reported.
    for found in [0, 2, u8::MAX] {
        assert_eq!(
            VariantKey::decode(&[found]),
            Err(KeyError::Version { found })
        );
    }
    // Another version in front of an otherwise valid key.
    let mut other = GOLDEN_BYTES;
    other[0] = 2;
    assert_eq!(
        VariantKey::decode(&other),
        Err(KeyError::Version { found: 2 })
    );
    assert_eq!(
        KeyError::Version { found: 2 }.to_string(),
        format!(
            "variant key version 2 is not supported; this build reads version {VARIANT_KEY_VERSION}"
        )
    );
    // The right version with nothing after it.
    assert_eq!(
        VariantKey::decode(&[VARIANT_KEY_VERSION]),
        Err(KeyError::Truncated {
            field: KeyField::VertexAttribute(0),
            len: 1
        })
    );
}

#[test]
fn every_prefix_is_truncated_in_the_right_field() {
    for len in 0..GOLDEN_BYTES.len() {
        let result = VariantKey::decode(&GOLDEN_BYTES[..len]);
        assert!(
            matches!(result, Err(KeyError::Truncated { len: reported, .. }) if reported == len),
            "prefix of {len} bytes: {result:?}"
        );
    }
    // The field each length ends in, from the module docs' table.
    let cases = [
        (0, KeyField::Version),
        (1, KeyField::VertexAttribute(0)),
        (32, KeyField::VertexAttribute(31)),
        (33, KeyField::ColorTarget(0)),
        (40, KeyField::ColorTarget(7)),
        (41, KeyField::DualSourceBlend),
        (42, KeyField::NextStageReads),
        (73, KeyField::NextStageReads),
        (74, KeyField::WorkgroupSize(0)),
        (77, KeyField::WorkgroupSize(0)),
        (78, KeyField::WorkgroupSize(1)),
        (82, KeyField::WorkgroupSize(2)),
        (85, KeyField::WorkgroupSize(2)),
        (86, KeyField::SharedMemorySize),
        (89, KeyField::SharedMemorySize),
        (90, KeyField::AlphaTest),
        (91, KeyField::DepthRangeRewrite),
        (92, KeyField::PrimitiveRestartRewrite),
        (93, KeyField::AttributeFetch),
        (94, KeyField::TypedOutputs),
        (95, KeyField::SwizzleCount),
        (
            96,
            KeyField::Swizzle {
                binding: 0,
                component: 0,
            },
        ),
        (
            99,
            KeyField::Swizzle {
                binding: 0,
                component: 3,
            },
        ),
        (
            100,
            KeyField::Swizzle {
                binding: 1,
                component: 0,
            },
        ),
        (
            103,
            KeyField::Swizzle {
                binding: 1,
                component: 3,
            },
        ),
    ];
    for (len, field) in cases {
        assert_eq!(
            VariantKey::decode(&GOLDEN_BYTES[..len]),
            Err(KeyError::Truncated { field, len }),
            "prefix of {len} bytes"
        );
    }
    assert_eq!(
        KeyError::Truncated {
            field: KeyField::WorkgroupSize(1),
            len: 80
        }
        .to_string(),
        "the variant key ends after 80 bytes, inside the workgroup size along axis 1"
    );
    assert_eq!(
        KeyError::Truncated {
            field: KeyField::Version,
            len: 0
        }
        .to_string(),
        "the variant key ends after 0 bytes, inside the version"
    );
}

/// The golden bytes with `offset` set to `value`.
fn patched(offset: usize, value: u8) -> Vec<u8> {
    let mut bytes = GOLDEN_BYTES.to_vec();
    bytes[offset] = value;
    bytes
}

#[test]
fn each_coded_field_accepts_its_last_code_and_refuses_the_next() {
    // (offset, field, the highest valid code, the lowest invalid code above it)
    let cases = [
        (1, KeyField::VertexAttribute(0), 5, 6),
        (32, KeyField::VertexAttribute(31), 5, 6),
        (33, KeyField::ColorTarget(0), 2, 3),
        (40, KeyField::ColorTarget(7), 2, 3),
        (41, KeyField::DualSourceBlend, 1, 2),
        (90, KeyField::AlphaTest, 8, 9),
        (91, KeyField::DepthRangeRewrite, 1, 2),
        (92, KeyField::PrimitiveRestartRewrite, 1, 2),
        (93, KeyField::AttributeFetch, 1, 2),
        (94, KeyField::TypedOutputs, 1, 2),
        (
            96,
            KeyField::Swizzle {
                binding: 0,
                component: 0,
            },
            7,
            8,
        ),
        (
            103,
            KeyField::Swizzle {
                binding: 1,
                component: 3,
            },
            7,
            8,
        ),
    ];
    for (offset, field, valid, invalid) in cases {
        let accepted = VariantKey::decode(&patched(offset, valid));
        assert!(accepted.is_ok(), "{field}: {accepted:?}");
        assert_eq!(accepted.map(|key| key.encode()), Ok(patched(offset, valid)));
        assert_eq!(
            VariantKey::decode(&patched(offset, invalid)),
            Err(KeyError::InvalidValue {
                field,
                value: invalid
            }),
            "{field}"
        );
        assert_eq!(
            VariantKey::decode(&patched(offset, u8::MAX)),
            Err(KeyError::InvalidValue {
                field,
                value: u8::MAX
            }),
            "{field}"
        );
    }
}

#[test]
fn the_codes_inside_each_set() {
    // Zero is valid everywhere; swizzle code 1 is the header's gap.
    let zero = |offset| VariantKey::decode(&patched(offset, 0));
    assert_eq!(
        zero(1).map(|key| key.vertex_attributes[0]),
        Ok(AttributeType::Float)
    );
    assert_eq!(
        zero(33).map(|key| key.color_targets[0]),
        Ok(ColorType::Float)
    );
    assert_eq!(zero(41).map(|key| key.dual_source_blend), Ok(false));
    assert_eq!(zero(90).map(|key| key.emulation.alpha_test), Ok(None));
    assert_eq!(
        VariantKey::decode(&patched(90, 1)).map(|key| key.emulation.alpha_test),
        Ok(Some(CompareFunction::Never))
    );
    assert_eq!(
        VariantKey::decode(&patched(90, 8)).map(|key| key.emulation.alpha_test),
        Ok(Some(CompareFunction::Always))
    );
    let switches: [(usize, Read); 4] = [
        (91, |key| key.emulation.depth_range_rewrite),
        (92, |key| key.emulation.primitive_restart_rewrite),
        (93, |key| key.emulation.attribute_fetch),
        (94, |key| key.emulation.typed_outputs),
    ];
    for (offset, read) in switches {
        assert_eq!(
            zero(offset).map(|key| read(&key)),
            Ok(false),
            "offset {offset}"
        );
        assert_eq!(
            VariantKey::decode(&GOLDEN_BYTES).map(|key| read(&key)),
            Ok(true)
        );
    }
    assert_eq!(
        zero(96).map(|key| key.texture_swizzles.as_slice()[0].x),
        Ok(SwizzleSource::Zero)
    );
    let field = KeyField::Swizzle {
        binding: 0,
        component: 0,
    };
    assert_eq!(
        VariantKey::decode(&patched(96, 1)),
        Err(KeyError::InvalidValue { field, value: 1 })
    );
    assert_eq!(
        KeyError::InvalidValue { field, value: 1 }.to_string(),
        "component 0 of texture swizzle 0 has the invalid value 1"
    );
}

#[test]
fn the_first_invalid_field_is_reported() {
    let mut bytes = patched(33, 3);
    bytes[91] = 2;
    bytes[103] = 1;
    assert_eq!(
        VariantKey::decode(&bytes),
        Err(KeyError::InvalidValue {
            field: KeyField::ColorTarget(0),
            value: 3
        })
    );
    // An invalid field before a truncation: the invalid field.
    assert_eq!(
        VariantKey::decode(&patched(41, 2)[..60]),
        Err(KeyError::InvalidValue {
            field: KeyField::DualSourceBlend,
            value: 2
        })
    );
}

#[test]
fn the_swizzle_cap_at_its_edge() {
    let at_cap = vec![Swizzle::IDENTITY; MAX_TEXTURE_SWIZZLES];
    let swizzles = TextureSwizzles::new(at_cap).expect("at the cap");
    let key = VariantKey {
        texture_swizzles: swizzles,
        ..VariantKey::default()
    };
    let bytes = key.encode();
    assert_eq!(bytes.len(), VARIANT_KEY_MAX_SIZE);
    assert_eq!(VariantKey::decode(&bytes), Ok(key));

    let past = vec![Swizzle::IDENTITY; MAX_TEXTURE_SWIZZLES + 1];
    assert_eq!(
        TextureSwizzles::new(past),
        Err(KeyError::TooManySwizzles {
            count: MAX_TEXTURE_SWIZZLES + 1
        })
    );
    assert_eq!(
        KeyError::TooManySwizzles {
            count: MAX_TEXTURE_SWIZZLES + 1
        }
        .to_string(),
        format!(
            "{} texture swizzles are more than the {MAX_TEXTURE_SWIZZLES} a variant key holds",
            MAX_TEXTURE_SWIZZLES + 1
        )
    );
    // The same count in the bytes, with every swizzle present.
    let mut bytes = VariantKey::default().encode();
    let count = u8::try_from(MAX_TEXTURE_SWIZZLES + 1).expect("fits");
    *bytes.last_mut().expect("count byte") = count;
    for _ in 0..=MAX_TEXTURE_SWIZZLES {
        bytes.extend(Swizzle::IDENTITY.sources().map(SwizzleSource::code));
    }
    assert_eq!(
        VariantKey::decode(&bytes),
        Err(KeyError::TooManySwizzles {
            count: MAX_TEXTURE_SWIZZLES + 1
        })
    );
}

#[test]
fn the_swizzle_count_is_checked_before_the_swizzles() {
    // A count past the cap and too few bytes for it: the count is reported.
    let count = u8::try_from(MAX_TEXTURE_SWIZZLES + 1).expect("fits");
    let bytes = patched(95, count);
    assert_eq!(
        VariantKey::decode(&bytes),
        Err(KeyError::TooManySwizzles {
            count: MAX_TEXTURE_SWIZZLES + 1
        })
    );
    // A count within the cap and too few bytes: the swizzle it ends in.
    assert_eq!(
        VariantKey::decode(&patched(95, 3)),
        Err(KeyError::Truncated {
            field: KeyField::Swizzle {
                binding: 2,
                component: 0
            },
            len: GOLDEN_BYTES.len()
        })
    );
    // A count below what follows: the rest is trailing.
    assert_eq!(
        VariantKey::decode(&patched(95, 1)),
        Err(KeyError::TrailingBytes {
            end: GOLDEN_BYTES.len() - SWIZZLE_SIZE,
            len: GOLDEN_BYTES.len()
        })
    );
}

#[test]
fn trailing_bytes_are_refused() {
    let mut bytes = GOLDEN_BYTES.to_vec();
    bytes.push(0);
    assert_eq!(
        VariantKey::decode(&bytes),
        Err(KeyError::TrailingBytes {
            end: GOLDEN_BYTES.len(),
            len: GOLDEN_BYTES.len() + 1
        })
    );
    assert_eq!(
        KeyError::TrailingBytes { end: 104, len: 105 }.to_string(),
        "the variant key ends at byte 104, but 105 bytes were given"
    );
}

/// Checks a coded enum against its reference table, written out: each variant's code, and
/// that every other byte is refused.
fn check_codes<T: Copy + PartialEq + core::fmt::Debug>(
    all: &[T],
    code: fn(T) -> u8,
    from_code: fn(u8) -> Option<T>,
    table: &[(T, u8)],
) {
    assert_eq!(
        all.iter()
            .map(|&value| (value, code(value)))
            .collect::<Vec<_>>(),
        table
    );
    for byte in 0..=u8::MAX {
        let expected = table
            .iter()
            .find(|(_, listed)| *listed == byte)
            .map(|(value, _)| *value);
        assert_eq!(from_code(byte), expected, "code {byte}");
    }
}

#[test]
fn code_tables_match_the_references() {
    // Eden `AttributeType`, in declaration order (runtime_info.h).
    check_codes(
        AttributeType::ALL,
        AttributeType::code,
        AttributeType::from_code,
        &[
            (AttributeType::Float, 0),
            (AttributeType::SignedInt, 1),
            (AttributeType::UnsignedInt, 2),
            (AttributeType::SignedScaled, 3),
            (AttributeType::UnsignedScaled, 4),
            (AttributeType::Disabled, 5),
        ],
    );
    check_codes(
        ColorType::ALL,
        ColorType::code,
        ColorType::from_code,
        &[
            (ColorType::Float, 0),
            (ColorType::SignedInt, 1),
            (ColorType::UnsignedInt, 2),
        ],
    );
    // Eden `Shader::CompareFunction`, in declaration order (runtime_info.h).
    check_codes(
        CompareFunction::ALL,
        CompareFunction::code,
        CompareFunction::from_code,
        &[
            (CompareFunction::Never, 0),
            (CompareFunction::Less, 1),
            (CompareFunction::Equal, 2),
            (CompareFunction::LessThanEqual, 3),
            (CompareFunction::Greater, 4),
            (CompareFunction::NotEqual, 5),
            (CompareFunction::GreaterThanEqual, 6),
            (CompareFunction::Always, 7),
        ],
    );
    // `NVC797_TEXHEAD_BL_X_SOURCE_IN_*` (clc797tex.h).
    check_codes(
        SwizzleSource::ALL,
        SwizzleSource::code,
        SwizzleSource::from_code,
        &[
            (SwizzleSource::Zero, 0x0),
            (SwizzleSource::R, 0x2),
            (SwizzleSource::G, 0x3),
            (SwizzleSource::B, 0x4),
            (SwizzleSource::A, 0x5),
            (SwizzleSource::OneInt, 0x6),
            (SwizzleSource::OneFloat, 0x7),
        ],
    );
}

#[test]
fn attribute_masks_place_each_word() {
    assert!(!AttributeMask::EMPTY.contains(0));
    assert!(!AttributeMask::EMPTY.contains(u8::MAX));
    let cases: [(u8, [u64; 4]); 5] = [
        (0, [1, 0, 0, 0]),
        (1, [2, 0, 0, 0]),
        (63, [1 << 63, 0, 0, 0]),
        (64, [0, 1, 0, 0]),
        (255, [0, 0, 0, 1 << 63]),
    ];
    for (word, lanes) in cases {
        let mask = AttributeMask::EMPTY.with(word);
        assert_eq!(mask.lanes(), lanes, "word {word}");
        assert!(mask.contains(word));
        // The words on either side are not in it.
        if let Some(below) = word.checked_sub(1) {
            assert!(!mask.contains(below), "word {word}");
        }
        if let Some(above) = word.checked_add(1) {
            assert!(!mask.contains(above), "word {word}");
        }
        // Adding a word already there changes nothing.
        assert_eq!(mask.with(word), mask);
    }
    let both = AttributeMask::EMPTY.with(3).with(200);
    assert!(both.contains(3) && both.contains(200));
    assert_eq!(AttributeMask::from_lanes(both.lanes()), both);
}

#[test]
fn every_emulation_flag_has_a_key_field() {
    // The names of the render crate's `EmulationFlags` constants
    // (crates/render/src/pipeline.rs), written out; G18 checks the copy against
    // `EmulationFlags::NAMED`.
    let names: Vec<&str> = EmulationKey::MIRRORED
        .iter()
        .map(|(name, _)| *name)
        .collect();
    assert_eq!(
        names,
        [
            "ALPHA_TEST",
            "DEPTH_RANGE_REWRITE",
            "PRIMITIVE_RESTART_REWRITE",
            "ATTRIBUTE_FETCH",
            "TYPED_OUTPUTS",
        ]
    );
    // Each reader sees its own field and no other.
    let only = [
        EmulationKey {
            alpha_test: Some(CompareFunction::Less),
            ..EmulationKey::default()
        },
        EmulationKey {
            depth_range_rewrite: true,
            ..EmulationKey::default()
        },
        EmulationKey {
            primitive_restart_rewrite: true,
            ..EmulationKey::default()
        },
        EmulationKey {
            attribute_fetch: true,
            ..EmulationKey::default()
        },
        EmulationKey {
            typed_outputs: true,
            ..EmulationKey::default()
        },
    ];
    for (set, key) in only.iter().enumerate() {
        let seen: Vec<bool> = EmulationKey::MIRRORED
            .iter()
            .map(|(_, read)| read(key))
            .collect();
        let expected: Vec<bool> = (0..only.len()).map(|index| index == set).collect();
        assert_eq!(seen, expected, "field {set}");
    }
    let none: Vec<bool> = EmulationKey::MIRRORED
        .iter()
        .map(|(_, read)| read(&EmulationKey::default()))
        .collect();
    assert_eq!(none, [false; 5]);
}

#[test]
fn key_field_names_are_plain_words() {
    let cases = [
        (KeyField::Version, "the version"),
        (
            KeyField::VertexAttribute(31),
            "the type of vertex attribute 31",
        ),
        (KeyField::ColorTarget(7), "the type of colour target 7"),
        (KeyField::DualSourceBlend, "the dual-source blending switch"),
        (
            KeyField::NextStageReads,
            "the attributes the next stage reads",
        ),
        (
            KeyField::WorkgroupSize(2),
            "the workgroup size along axis 2",
        ),
        (KeyField::SharedMemorySize, "the shared-memory size"),
        (KeyField::AlphaTest, "the alpha test"),
        (
            KeyField::DepthRangeRewrite,
            "the depth-range rewrite switch",
        ),
        (
            KeyField::PrimitiveRestartRewrite,
            "the primitive-restart rewrite switch",
        ),
        (KeyField::AttributeFetch, "the attribute fetch switch"),
        (KeyField::TypedOutputs, "the typed outputs switch"),
        (KeyField::SwizzleCount, "the texture swizzle count"),
        (
            KeyField::Swizzle {
                binding: 5,
                component: 2,
            },
            "component 2 of texture swizzle 5",
        ),
    ];
    for (field, text) in cases {
        assert_eq!(field.to_string(), text);
    }
}
