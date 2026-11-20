//! Driver-constants tests: golden bytes and the layout table written out.

use super::*;

/// Distinct non-zero values in every field; the base vertex is at or above 2^31.
const VALUES: DriverConstants = DriverConstants {
    alpha_reference: 0.5,
    y_negate: true,
    render_area_width: 1280.0,
    render_area_height: 720.0,
    base_vertex: 0x8000_0003,
    base_instance: 0x0102_0304,
    flip_y: true,
    reverse_depth: true,
    clip_depth: ClipDepth::ZeroToW,
};

#[test]
fn golden_bytes() {
    assert_eq!(
        VALUES.encode(),
        [
            0x00, 0x00, 0x00, 0x3f, // alpha reference 0.5
            0x00, 0x00, 0x80, 0xbf, // viewport flip -1.0
            0x00, 0x00, 0xa0, 0x44, // render-area width 1280.0
            0x00, 0x00, 0x34, 0x44, // render-area height 720.0
            0x03, 0x00, 0x00, 0x80, // base vertex 0x8000_0003
            0x04, 0x03, 0x02, 0x01, // base instance 0x0102_0304
            0x00, 0x00, 0x80, 0xbf, // position y scale -1.0
            0x00, 0x00, 0x80, 0xbf, // position z scale -1.0
            0x00, 0x00, 0x80, 0x3f, // position z offset 1.0
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // padding
            0x00, 0x00, 0x00, 0x00,
        ]
    );
    let unflipped = DriverConstants {
        y_negate: false,
        ..VALUES
    };
    assert_eq!(unflipped.encode()[4..8], [0x00, 0x00, 0x80, 0x3f]);
    // Each viewport switch changes only its own words.
    let upright = DriverConstants {
        flip_y: false,
        ..VALUES
    };
    let mut expected = VALUES.encode();
    expected[24..28].copy_from_slice(&[0x00, 0x00, 0x80, 0x3f]);
    assert_eq!(upright.encode(), expected);
    let forward = DriverConstants {
        reverse_depth: false,
        ..VALUES
    };
    let mut expected = VALUES.encode();
    expected[28..36].copy_from_slice(&[0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x00, 0x00]);
    assert_eq!(forward.encode(), expected);
    // A -w..w clip range turns the reversal's offset to 0.0 and leaves every other word.
    let symmetric = DriverConstants {
        clip_depth: ClipDepth::MinusWToW,
        ..VALUES
    };
    let mut expected = VALUES.encode();
    expected[32..36].copy_from_slice(&[0x00, 0x00, 0x00, 0x00]);
    assert_eq!(symmetric.encode(), expected);
    // Without a reversal the clip range changes nothing.
    let forward_symmetric = DriverConstants {
        clip_depth: ClipDepth::MinusWToW,
        ..forward
    };
    assert_eq!(forward_symmetric.encode(), forward.encode());
}

#[test]
fn offsets_are_the_tables() {
    let layout: Vec<(DriverConstant, usize, ConstantType)> = DriverConstant::ALL
        .iter()
        .map(|&constant| (constant, constant.offset(), constant.ty()))
        .collect();
    assert_eq!(
        layout,
        vec![
            (DriverConstant::AlphaReference, 0, ConstantType::F32),
            (DriverConstant::ViewportFlip, 4, ConstantType::F32),
            (DriverConstant::RenderAreaWidth, 8, ConstantType::F32),
            (DriverConstant::RenderAreaHeight, 12, ConstantType::F32),
            (DriverConstant::BaseVertex, 16, ConstantType::U32),
            (DriverConstant::BaseInstance, 20, ConstantType::U32),
            (DriverConstant::PositionYScale, 24, ConstantType::F32),
            (DriverConstant::PositionZScale, 28, ConstantType::F32),
            (DriverConstant::PositionZOffset, 32, ConstantType::F32),
        ]
    );
    assert_eq!(DRIVER_CONSTANTS_SIZE, 48);
    assert_eq!(DRIVER_CONSTANTS_VERSION, 2);
    assert_eq!(DriverConstantsLayout::CURRENT.version(), 2);
    assert_eq!(DriverConstantsLayout::CURRENT.size(), 48);
}

#[test]
fn each_word_is_its_field() {
    let words: Vec<u32> = DriverConstant::ALL
        .iter()
        .map(|&constant| VALUES.word(constant))
        .collect();
    assert_eq!(
        words,
        vec![
            0.5_f32.to_bits(),
            (-1.0_f32).to_bits(),
            1280.0_f32.to_bits(),
            720.0_f32.to_bits(),
            0x8000_0003,
            0x0102_0304,
            (-1.0_f32).to_bits(),
            (-1.0_f32).to_bits(),
            1.0_f32.to_bits(),
        ]
    );
    // Each word sits at its offset in the encoding.
    let bytes = VALUES.encode();
    for constant in DriverConstant::ALL {
        let offset = constant.offset();
        assert_eq!(
            bytes[offset..offset + 4],
            VALUES.word(constant).to_le_bytes(),
            "{constant:?}"
        );
    }
}

#[test]
fn the_float_bits_are_kept() {
    // A NaN with a payload, as a guest register's bits read as a float can be.
    let nan = f32::from_bits(0x7fc0_1234);
    let constants = DriverConstants {
        alpha_reference: nan,
        ..VALUES
    };
    assert_eq!(constants.encode()[0..4], 0x7fc0_1234_u32.to_le_bytes());
}

#[test]
fn defaults() {
    let fresh = DriverConstants::default();
    assert_eq!(fresh.alpha_reference.to_bits(), 0);
    assert!(!fresh.y_negate);
    assert_eq!(fresh.word(DriverConstant::ViewportFlip), 1.0_f32.to_bits());
    assert_eq!(fresh.render_area_width.to_bits(), 0);
    assert_eq!(fresh.render_area_height.to_bits(), 0);
    assert_eq!(fresh.base_vertex, 0);
    assert_eq!(fresh.base_instance, 0);
    // No flip: the position passes through unchanged.
    assert!(!fresh.flip_y);
    assert!(!fresh.reverse_depth);
    assert_eq!(fresh.clip_depth, ClipDepth::ZeroToW);
    assert_eq!(
        fresh.word(DriverConstant::PositionYScale),
        1.0_f32.to_bits()
    );
    assert_eq!(
        fresh.word(DriverConstant::PositionZScale),
        1.0_f32.to_bits()
    );
    assert_eq!(
        fresh.word(DriverConstant::PositionZOffset),
        0.0_f32.to_bits()
    );
}

/// The position the lowering writes, `(x, y * ys, z * zs + w * zo, w)`, from the encoded words.
fn shaded(constants: &DriverConstants, [x, y, z, w]: [f32; 4]) -> [f32; 4] {
    let bytes = constants.encode();
    let read = |constant: DriverConstant| {
        let at = constant.offset();
        f32::from_le_bytes(bytes[at..at + 4].try_into().expect("a word"))
    };
    [
        x,
        y * read(DriverConstant::PositionYScale),
        z * read(DriverConstant::PositionZScale) + w * read(DriverConstant::PositionZOffset),
        w,
    ]
}

/// What each switch does to a position through the module docs' formula: y negated, z mirrored
/// within 0..w (`w - z`) or, with a -w..w clip range, about zero (`-z`), and nothing at all with
/// both flips off whatever the clip range.
#[test]
fn the_position_words_flip_and_mirror() {
    let position = [0.5, 0.25, 0.75, 2.0];
    let with = |flip_y, reverse_depth, symmetric: bool| DriverConstants {
        flip_y,
        reverse_depth,
        clip_depth: if symmetric {
            ClipDepth::MinusWToW
        } else {
            ClipDepth::ZeroToW
        },
        ..DriverConstants::default()
    };
    for symmetric in [false, true] {
        assert_eq!(shaded(&with(false, false, symmetric), position), position);
        assert_eq!(
            shaded(&with(true, false, symmetric), position),
            [0.5, -0.25, 0.75, 2.0]
        );
    }
    assert_eq!(
        shaded(&with(false, true, false), position),
        [0.5, 0.25, 1.25, 2.0]
    );
    assert_eq!(
        shaded(&with(true, true, false), position),
        [0.5, -0.25, 1.25, 2.0]
    );
    assert_eq!(
        shaded(&with(false, true, true), position),
        [0.5, 0.25, -0.75, 2.0]
    );
    assert_eq!(
        shaded(&with(true, true, true), position),
        [0.5, -0.25, -0.75, 2.0]
    );
    // The ends of 0..w swap.
    assert_eq!(
        shaded(&with(false, true, false), [0.0, 0.0, 0.0, 2.0])[2],
        2.0
    );
    assert_eq!(
        shaded(&with(false, true, false), [0.0, 0.0, 2.0, 2.0])[2],
        0.0
    );
    // The ends of -w..w swap.
    assert_eq!(
        shaded(&with(false, true, true), [0.0, 0.0, -2.0, 2.0])[2],
        2.0
    );
    assert_eq!(
        shaded(&with(false, true, true), [0.0, 0.0, 2.0, 2.0])[2],
        -2.0
    );
}
