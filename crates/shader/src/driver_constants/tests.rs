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
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // padding
        ]
    );
    let unflipped = DriverConstants {
        y_negate: false,
        ..VALUES
    };
    assert_eq!(unflipped.encode()[4..8], [0x00, 0x00, 0x80, 0x3f]);
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
        ]
    );
    assert_eq!(DRIVER_CONSTANTS_SIZE, 32);
    assert_eq!(DRIVER_CONSTANTS_VERSION, 1);
    assert_eq!(DriverConstantsLayout::CURRENT.version(), 1);
    assert_eq!(DriverConstantsLayout::CURRENT.size(), 32);
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
}
