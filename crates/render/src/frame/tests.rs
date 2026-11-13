//! The RGBA8 form of presented formats, and the PNG encoding.

use super::{Frame, FramePixels, RGBA8_BYTES, encode_png, rgba8};
use crate::command::PresentTransform;
use crate::error::RenderError;
use crate::format::PixelFormat;

/// The formats with an exact RGBA8 form, and which way round their bytes are.
#[test]
fn rgba8_of_each_format() {
    let texels = vec![1, 2, 3, 4, 250, 251, 252, 253];
    for format in PixelFormat::ALL {
        let expected = match format {
            PixelFormat::A8B8G8R8Unorm | PixelFormat::A8B8G8R8Srgb => Some(texels.clone()),
            PixelFormat::B8G8R8A8Unorm | PixelFormat::B8G8R8A8Srgb => {
                Some(vec![3, 2, 1, 4, 252, 251, 250, 253])
            }
            _ => None,
        };
        assert_eq!(rgba8(format, texels.clone()), expected, "{format}");
    }
    assert_eq!(
        rgba8(PixelFormat::B8G8R8A8Unorm, Vec::new()),
        Some(Vec::new())
    );
    assert_eq!(RGBA8_BYTES, 4);
}

/// A PNG decoded with the `png` crate's decoder: (width, height, RGBA8 bytes).
fn decode(file: &[u8]) -> (u32, u32, Vec<u8>) {
    let decoder = png::Decoder::new(std::io::Cursor::new(file));
    let mut reader = decoder.read_info().expect("a PNG header");
    let mut pixels = vec![0; reader.output_buffer_size().expect("a size that fits")];
    let info = reader.next_frame(&mut pixels).expect("image data");
    assert_eq!(info.color_type, png::ColorType::Rgba);
    assert_eq!(info.bit_depth, png::BitDepth::Eight);
    pixels.truncate(info.buffer_size());
    (info.width, info.height, pixels)
}

/// `count` bytes that differ from their neighbours, the same every run.
fn noise(count: usize) -> Vec<u8> {
    let mut state: u32 = 0x9E37_79B9;
    (0..count)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state.to_le_bytes()[1]
        })
        .collect()
}

/// An encoded image decodes to the same pixels, at sizes that are not multiples of anything.
#[test]
fn a_png_decodes_to_the_pixels_encoded() {
    for (width, height) in [
        (1, 1),
        (2, 2),
        (3, 5),
        (17, 9),
        (64, 1),
        (1, 64),
        (640, 360),
    ] {
        let pixels = noise((width * height) as usize * RGBA8_BYTES);
        let file = encode_png(width, height, &pixels).expect("a valid image");
        assert_eq!(
            decode(&file),
            (width, height, pixels),
            "{width} by {height}"
        );
    }
    // A flat image, which compresses to almost nothing, round-trips too.
    let flat = vec![0x40; 256 * 256 * RGBA8_BYTES];
    let file = encode_png(256, 256, &flat).expect("a valid image");
    assert!(file.len() < flat.len() / 16, "{}", file.len());
    assert_eq!(decode(&file), (256, 256, flat));
}

/// The bytes of a PNG file that do not depend on the compressor, worked out from the PNG
/// specification: the signature; the `IHDR` chunk (length 13, the size, bit depth 8, colour
/// type 6 for RGBA, compression 0, filter 0, no interlace, then its CRC-32); and the `IEND`
/// chunk at the end. Between them is `IDAT` and nothing else.
#[test]
fn png_structure() {
    let file = encode_png(3, 2, &noise(3 * 2 * RGBA8_BYTES)).expect("a valid image");
    let signature = [0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
    #[rustfmt::skip]
    let ihdr = [
        0, 0, 0, 13, b'I', b'H', b'D', b'R',
        0, 0, 0, 3, // width
        0, 0, 0, 2, // height
        8, 6, 0, 0, 0,
        // CRC-32 of "IHDR" and the 13 bytes above, from Python's `zlib.crc32`.
        0x9D, 0x74, 0x66, 0x1A,
    ];
    let iend = [0, 0, 0, 0, b'I', b'E', b'N', b'D', 0xAE, 0x42, 0x60, 0x82];
    assert_eq!(file[..8], signature);
    assert_eq!(file[8..33], ihdr);
    assert_eq!(file[file.len() - 12..], iend);
    // One IDAT chunk fills the rest: its length field says so.
    let idat = &file[33..file.len() - 12];
    let length = u32::from_be_bytes([idat[0], idat[1], idat[2], idat[3]]) as usize;
    assert_eq!(&idat[4..8], b"IDAT");
    assert_eq!(idat.len(), 4 + 4 + length + 4);
}

/// The whole file for a tiny image: red, green, blue and translucent white, two by two.
/// The expected file was built by hand from the PNG and zlib specifications, not taken from
/// the encoder, which stores an image this small instead of compressing it: the image data is
/// one stored deflate block (each row behind a filter byte of 0, "none"), and the checksums
/// are from Python's `zlib`.
#[test]
fn golden_bytes() {
    #[rustfmt::skip]
    let pixels = [
        0xFF, 0x00, 0x00, 0xFF,   0x00, 0xFF, 0x00, 0xFF,
        0x00, 0x00, 0xFF, 0xFF,   0xFF, 0xFF, 0xFF, 0x80,
    ];
    let file = encode_png(2, 2, &pixels).expect("a valid image");
    assert_eq!(decode(&file), (2, 2, pixels.to_vec()));
    assert_eq!(file, GOLDEN, "{file:02X?}");
}

/// See `golden_bytes`.
#[rustfmt::skip]
const GOLDEN: [u8; 86] = [
    // The signature.
    0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A,
    // IHDR: 2 by 2, 8 bits, RGBA.
    0, 0, 0, 13, b'I', b'H', b'D', b'R',
    0, 0, 0, 2, 0, 0, 0, 2, 8, 6, 0, 0, 0,
    0x72, 0xB6, 0x0D, 0x24,
    // IDAT, 29 bytes: the zlib header, a final stored block of 18 bytes (its length and the
    // length's complement, little-endian), ...
    0, 0, 0, 29, b'I', b'D', b'A', b'T',
    0x78, 0x01,
    0x01, 0x12, 0x00, 0xED, 0xFF,
    // ... the two rows, each a filter byte then two pixels, ...
    0x00,   0xFF, 0x00, 0x00, 0xFF,   0x00, 0xFF, 0x00, 0xFF,
    0x00,   0x00, 0x00, 0xFF, 0xFF,   0xFF, 0xFF, 0xFF, 0x80,
    // ... the rows' Adler-32, and the chunk's CRC-32.
    0x49, 0x49, 0x09, 0x78,
    0x4B, 0xD9, 0xCE, 0x03,
    // IEND.
    0, 0, 0, 0, b'I', b'E', b'N', b'D', 0xAE, 0x42, 0x60, 0x82,
];

/// The file for an image that does compress, which the hand-built file above cannot show:
/// four rows of one colour. These bytes are the encoder's own (fast compression), checked by
/// decoding them with Python's `zlib` and a PNG unfilter written from the specification (the
/// rows use the Paeth and up filters); they move if the compression setting changes, and also
/// if the `png` crate or its deflate implementation changes what it writes (module docs), in
/// which case check that the new file still decodes to the image and replace them.
#[test]
fn golden_bytes_of_a_compressed_image() {
    let pixels = [0x11, 0x22, 0x33, 0x44].repeat(16 * 4);
    let file = encode_png(16, 4, &pixels).expect("a valid image");
    assert_eq!(decode(&file), (16, 4, pixels));
    assert_eq!(file, GOLDEN_COMPRESSED, "{file:02X?}");
}

/// See `golden_bytes_of_a_compressed_image`.
#[rustfmt::skip]
const GOLDEN_COMPRESSED: [u8; 134] = [
    0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D,
    0x49, 0x48, 0x44, 0x52, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x04,
    0x08, 0x06, 0x00, 0x00, 0x00, 0x87, 0xB4, 0xBF, 0xEC, 0x00, 0x00, 0x00,
    0x4D, 0x49, 0x44, 0x41, 0x54, 0x78, 0x01, 0xED, 0xC0, 0x03, 0xA0, 0x24,
    0x59, 0x96, 0xC6, 0xF1, 0xFF, 0x77, 0xEE, 0x8D, 0xC8, 0xCC, 0xA7, 0x72,
    0x4B, 0x63, 0xAE, 0x6D, 0xDB, 0xB6, 0x6D, 0xDB, 0xB6, 0x6D, 0xDB, 0xB6,
    0x6D, 0x69, 0x8C, 0x9E, 0x96, 0x4A, 0xAF, 0x9E, 0x32, 0x33, 0x22, 0xEE,
    0xF9, 0x76, 0xB7, 0x6A, 0x7A, 0xA6, 0x87, 0x3B, 0x6B, 0xD5, 0xAF, 0x9E,
    0x78, 0xE8, 0xCB, 0xBD, 0x3E, 0xFF, 0x76, 0x04, 0xFF, 0x3E, 0x04, 0xFF,
    0x3E, 0x04, 0xFF, 0x3E, 0xFC, 0x23, 0xB2, 0xCA, 0x00, 0xB5, 0x2E, 0xCD,
    0x1E, 0x95, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE, 0x42,
    0x60, 0x82,
];

#[test]
fn an_empty_image_is_refused() {
    let empty = RenderError::Invalid("a frame must not be empty");
    assert_eq!(encode_png(0, 4, &[]), Err(empty.clone()));
    assert_eq!(encode_png(4, 0, &[]), Err(empty.clone()));
    assert_eq!(encode_png(0, 0, &[]), Err(empty.clone()));
    // An empty image is reported before a wrong length.
    assert_eq!(encode_png(0, 4, &[1, 2, 3]), Err(empty.clone()));
    assert_eq!(empty.to_string(), "a frame must not be empty");
}

#[test]
fn the_pixels_must_be_exactly_the_image() {
    let wrong = |actual: u64| RenderError::SizeMismatch {
        what: "frame pixels",
        expected: 3 * 2 * 4,
        actual,
    };
    let pixels = noise(3 * 2 * RGBA8_BYTES + 1);
    assert_eq!(encode_png(3, 2, &pixels[..23]), Err(wrong(23)));
    assert_eq!(encode_png(3, 2, &pixels), Err(wrong(25)));
    assert!(encode_png(3, 2, &pixels[..24]).is_ok());
    assert_eq!(encode_png(3, 2, &[]), Err(wrong(0)));
    assert_eq!(
        wrong(23).to_string(),
        "frame pixels is 23 bytes, but 24 are needed"
    );
    // Width and height are not interchangeable: 3 by 2 pixels are not 2 by 3... in bytes they
    // are, so the size alone cannot tell; the header can.
    let file = encode_png(2, 3, &pixels[..24]).expect("a valid image");
    assert_eq!(decode(&file).0, 2);
    assert_eq!(decode(&file).1, 3);
}

/// A size whose byte count does not fit in 64 bits is refused, not wrapped.
#[test]
fn a_size_that_overflows_is_refused() {
    let overflow = Err(RenderError::Overflow { what: "frame size" });
    let mismatch = |expected| {
        Err(RenderError::SizeMismatch {
            what: "frame pixels",
            expected,
            actual: 0,
        })
    };
    assert_eq!(encode_png(u32::MAX, u32::MAX, &[]), overflow);
    // A size that fits is a size mismatch, as any other.
    assert_eq!(
        encode_png(u32::MAX, 1, &[]),
        mismatch(u64::from(u32::MAX) * 4)
    );
    // The edge, with either side the long one. Four bytes for each of (2^32 - 1) x 2^30
    // pixels is 2^64 - 2^32 bytes, the most two sides can ask for and still fit; one more
    // row or column does not fit.
    let edge = 1 << 30;
    let most = u64::MAX - u64::from(u32::MAX);
    assert_eq!(encode_png(u32::MAX, edge, &[]), mismatch(most));
    assert_eq!(encode_png(edge, u32::MAX, &[]), mismatch(most));
    assert_eq!(encode_png(u32::MAX, edge + 1, &[]), overflow);
    assert_eq!(encode_png(edge + 1, u32::MAX, &[]), overflow);
    // The pixel count alone fits where its bytes do not: 2^62 pixels are 2^64 bytes.
    assert_eq!(encode_png(1 << 31, 1 << 31, &[]), overflow);
    assert_eq!(
        encode_png((1 << 31) - 1, 1 << 31, &[]),
        mismatch(u64::MAX - (1 << 33) + 1)
    );
}

fn frame(pixels: FramePixels) -> Frame {
    Frame {
        frame: 7,
        width: 2,
        height: 1,
        transform: PresentTransform::Rotate90,
        pixels,
    }
}

#[test]
fn a_frames_png() {
    let pixels = vec![1, 2, 3, 4, 5, 6, 7, 8];
    let known = frame(FramePixels::Rgba8(pixels.clone()));
    let file = known.png().expect("a valid frame").expect("known pixels");
    // The transform is not applied: the file is the frame as stored.
    assert_eq!(decode(&file), (2, 1, pixels.clone()));
    assert_eq!(Ok(file), encode_png(2, 1, &pixels));
    // Unknown pixels have no PNG.
    assert_eq!(frame(FramePixels::Unknown).png(), Ok(None));
    // A frame whose pixels are not its size is an error, as for `encode_png`.
    assert_eq!(
        frame(FramePixels::Rgba8(vec![1, 2, 3, 4])).png(),
        Err(RenderError::SizeMismatch {
            what: "frame pixels",
            expected: 8,
            actual: 4
        })
    );
}
