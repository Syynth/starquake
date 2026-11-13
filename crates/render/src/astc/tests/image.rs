//! Whole images: the order of blocks, cropping at the edges, layers, and the size checks.

use super::{COLOR_AT, GRID_4X4, WEIGHTS_0_TO_3, byte_values, plain_weights};
use crate::astc::{BLOCK_BYTES, ERROR_COLOR, decode};
use crate::error::RenderError;
use crate::format::PixelFormat;
use crate::resource::Extent3d;
use crate::validate::packed_size;
use PixelFormat as F;

/// A void-extent block of one opaque grey (C.2.23): every texel is `level`.
fn flat(level: u8) -> [u8; BLOCK_BYTES] {
    let component = u128::from(level) << 8;
    let color = (component << 64) | (component << 80) | (component << 96) | (0xFFFF << 112);
    let block: u128 = 0x1FC | (0b11 << 10) | (((1 << 52) - 1) << 12) | color;
    block.to_le_bytes()
}

/// A 4 by 4 block whose texel (x, y) has luminance `RAMP[x]` on even rows and `RAMP[3 - x]`
/// on odd ones, so a block shows which way round it was placed.
fn ramp_block() -> [u8; BLOCK_BYTES] {
    let weights = [0, 1, 2, 3, 3, 2, 1, 0, 0, 1, 2, 3, 3, 2, 1, 0];
    let block: u128 =
        GRID_4X4 | WEIGHTS_0_TO_3 | byte_values(&[0, 255], COLOR_AT) | plain_weights(&weights, 2);
    block.to_le_bytes()
}

const RAMP: [u8; 4] = [0, 84, 171, 255];

const fn extent(width: u32, height: u32, depth_or_layers: u32) -> Extent3d {
    Extent3d {
        width,
        height,
        depth_or_layers,
    }
}

/// The red channel of each texel, row by row; the tests' blocks are grey and opaque, which
/// this checks too.
fn levels(rgba: &[u8], width: usize) -> Vec<Vec<u8>> {
    let pixels: Vec<u8> = rgba
        .chunks(4)
        .map(|texel| {
            assert_eq!(texel, [texel[0], texel[0], texel[0], 0xFF]);
            texel[0]
        })
        .collect();
    pixels.chunks(width).map(<[u8]>::to_vec).collect()
}

/// Blocks are in raster order: along a row of blocks, then the next row.
#[test]
fn blocks_are_placed_in_raster_order() {
    let data: Vec<u8> = [flat(10), flat(20), flat(30), ramp_block()].concat();
    let mut out = vec![0x77; 8 * 8 * 4];
    assert_eq!(
        decode(F::Astc2d4x4Unorm, extent(8, 8, 1), &data, &mut out),
        Ok(0)
    );
    let [a, b, c, d] = RAMP;
    assert_eq!(
        levels(&out, 8),
        [
            [10, 10, 10, 10, 20, 20, 20, 20],
            [10, 10, 10, 10, 20, 20, 20, 20],
            [10, 10, 10, 10, 20, 20, 20, 20],
            [10, 10, 10, 10, 20, 20, 20, 20],
            [30, 30, 30, 30, a, b, c, d],
            [30, 30, 30, 30, d, c, b, a],
            [30, 30, 30, 30, a, b, c, d],
            [30, 30, 30, 30, d, c, b, a],
        ]
    );
}

/// An image that is not a whole number of blocks: the blocks at the right and bottom edges
/// are cropped, and a neighbour that starts exactly at the image's edge does not exist.
#[test]
fn edge_blocks_are_cropped() {
    // 6 by 5 texels of 4 by 4 blocks: 2 by 2 blocks, the right ones 2 wide, the bottom ones
    // 1 high.
    let data: Vec<u8> = [flat(10), ramp_block(), ramp_block(), flat(40)].concat();
    let mut out = vec![0x77; 6 * 5 * 4];
    assert_eq!(
        decode(F::Astc2d4x4Unorm, extent(6, 5, 1), &data, &mut out),
        Ok(0)
    );
    let [a, b, c, d] = RAMP;
    assert_eq!(
        levels(&out, 6),
        [
            [10, 10, 10, 10, a, b],
            [10, 10, 10, 10, d, c],
            [10, 10, 10, 10, a, b],
            [10, 10, 10, 10, d, c],
            [a, b, c, d, 40, 40],
        ]
    );
    // Exactly one block wide and high, then one texel more each way.
    let mut out = vec![0x77; 4 * 4 * 4];
    assert_eq!(
        decode(F::Astc2d4x4Unorm, extent(4, 4, 1), &ramp_block(), &mut out),
        Ok(0)
    );
    assert_eq!(levels(&out, 4)[1], [d, c, b, a]);
    let mut out = vec![0x77; 5 * 5 * 4];
    assert_eq!(
        decode(F::Astc2d4x4Unorm, extent(5, 5, 1), &data, &mut out),
        Ok(0)
    );
    assert_eq!(levels(&out, 5)[4], [a, b, c, d, 40]);
    // A footprint that is not square, and larger than the image.
    let mut out = vec![0x77; 3 * 2 * 4];
    assert_eq!(
        decode(F::Astc2d10x5Srgb, extent(3, 2, 1), &flat(99), &mut out),
        Ok(0)
    );
    assert_eq!(levels(&out, 3), [[99, 99, 99], [99, 99, 99]]);
}

/// Layers (or the slices of a 3D texture) follow one another, each a whole 2D image.
#[test]
fn layers_follow_one_another() {
    // 5 by 4 texels, three layers: two blocks a layer.
    let data: Vec<u8> = [flat(1), flat(2), flat(3), flat(4), flat(5), flat(6)].concat();
    let mut out = vec![0x77; 5 * 4 * 4 * 3];
    assert_eq!(
        decode(F::Astc2d4x4Unorm, extent(5, 4, 3), &data, &mut out),
        Ok(0)
    );
    let rows = levels(&out, 5);
    assert_eq!(rows.len(), 12);
    for (layer, (left, right)) in [(1, 2), (3, 4), (5, 6)].into_iter().enumerate() {
        for row in &rows[layer * 4..layer * 4 + 4] {
            assert_eq!(row, &[left, left, left, left, right]);
        }
    }
}

/// Illegal blocks are decoded to the error colour and counted.
#[test]
fn illegal_blocks_are_counted() {
    let illegal = [0; BLOCK_BYTES];
    let data: Vec<u8> = [illegal, flat(10), illegal, illegal].concat();
    let mut out = vec![0x77; 8 * 8 * 4];
    assert_eq!(
        decode(F::Astc2d4x4Srgb, extent(8, 8, 1), &data, &mut out),
        Ok(3)
    );
    for (index, texel) in out.chunks(4).enumerate() {
        let (x, y) = (index % 8, index / 8);
        if x >= 4 && y < 4 {
            assert_eq!(texel, [10, 10, 10, 0xFF]);
        } else {
            assert_eq!(texel, ERROR_COLOR);
        }
    }
    // A cropped illegal block still counts once.
    let mut out = vec![0x77; 4];
    assert_eq!(
        decode(F::Astc2d12x12Unorm, extent(1, 1, 1), &illegal, &mut out),
        Ok(1)
    );
    assert_eq!(out, ERROR_COLOR);
}

/// An empty image needs no data and writes nothing.
#[test]
fn an_empty_image_is_nothing() {
    for empty in [extent(0, 4, 1), extent(4, 0, 1), extent(4, 4, 0)] {
        assert_eq!(decode(F::Astc2d4x4Unorm, empty, &[], &mut []), Ok(0));
    }
}

/// The checks, in order: the format, the data's size, the output's size.
#[test]
fn sizes_are_checked_before_anything_is_read() {
    let not_astc = RenderError::Invalid("the format is not ASTC");
    let data = ramp_block();
    let mut out = vec![0x77; 4 * 4 * 4];
    for format in [F::A8B8G8R8Unorm, F::Bc7Unorm, F::Etc2RgbaUnorm, F::D32Float] {
        // Wrong sizes too: the format is reported first.
        assert_eq!(
            decode(format, extent(4, 4, 1), &data[..3], &mut out[..5]),
            Err(not_astc.clone())
        );
    }
    assert_eq!(not_astc.to_string(), "the format is not ASTC");

    // The data must be exactly the blocks the extent needs: one byte short, one over.
    let short = |actual: u64| RenderError::SizeMismatch {
        what: "ASTC data",
        expected: 16,
        actual,
    };
    assert_eq!(
        decode(F::Astc2d4x4Unorm, extent(4, 4, 1), &data[..15], &mut out),
        Err(short(15))
    );
    let mut long = data.to_vec();
    long.push(0);
    // With the output wrong as well, the data is reported first.
    assert_eq!(
        decode(F::Astc2d4x4Unorm, extent(4, 4, 1), &long, &mut out[..63]),
        Err(short(17))
    );
    assert_eq!(
        short(15).to_string(),
        "ASTC data is 15 bytes, but 16 are needed"
    );

    // The output must be exactly width * height * 4 bytes a layer.
    let wrong = |actual: u64| RenderError::SizeMismatch {
        what: "ASTC decode output",
        expected: 64,
        actual,
    };
    assert_eq!(
        decode(F::Astc2d4x4Unorm, extent(4, 4, 1), &data, &mut out[..63]),
        Err(wrong(63))
    );
    let mut long = vec![0x77; 65];
    assert_eq!(
        decode(F::Astc2d4x4Unorm, extent(4, 4, 1), &data, &mut long),
        Err(wrong(65))
    );
    // Nothing was written by a refused call.
    assert!(out.iter().all(|byte| *byte == 0x77));
    assert!(long.iter().all(|byte| *byte == 0x77));
    assert_eq!(
        decode(F::Astc2d4x4Unorm, extent(4, 4, 1), &data, &mut out),
        Ok(0)
    );
}

/// Sizes that do not fit in 64 bits are refused, not wrapped.
#[test]
fn sizes_that_overflow_are_refused() {
    let huge = extent(u32::MAX, u32::MAX, u32::MAX);
    // The data size: (2^32 / 4)^2 blocks a layer, times 2^32 layers, times 16 bytes.
    assert_eq!(
        packed_size(F::Astc2d4x4Unorm, huge),
        Err(RenderError::Overflow {
            what: "packed texture size"
        })
    );
    assert_eq!(
        decode(F::Astc2d4x4Unorm, huge, &[], &mut []),
        Err(RenderError::Overflow {
            what: "packed texture size"
        })
    );
    // One layer of 2^32 - 1 texels each way, in 12 by 12 blocks: the data size fits in 64
    // bits (about 2^64 / 9 bytes), and the output's, 4 bytes a texel, does not. The output
    // size is worked out, and refused, before either length is compared.
    let wide = extent(u32::MAX, u32::MAX, 1);
    let blocks = u64::from(u32::MAX.div_ceil(12));
    assert_eq!(
        packed_size(F::Astc2d12x12Unorm, wide),
        Ok(blocks * blocks * 16)
    );
    assert_eq!(
        decode(F::Astc2d12x12Unorm, wide, &[], &mut []),
        Err(RenderError::Overflow {
            what: "packed texture size"
        })
    );
}
