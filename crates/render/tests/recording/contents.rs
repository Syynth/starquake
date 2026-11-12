//! The CPU model: uploads, copies, clears and blits across formats, with pixel results computed
//! by hand, and readbacks of what nothing (or a draw) wrote.

use super::fixtures::{
    backend, buffer, clear, counting, ok, read, rect, region, slice, submit, texture, upload, view,
};
use starquake_render::command::{BlitDesc, ClearValue, CopyDesc};
use starquake_render::resource::{FilterMode, TextureDimension};
use starquake_render::{Backend, BufferUsage, Command, PixelFormat, Readback};

fn bytes(v: Vec<u8>) -> Readback {
    Readback::Bytes(v)
}

const COPY: BufferUsage = BufferUsage::COPY_SRC.union(BufferUsage::COPY_DST);

#[test]
fn buffer_upload_is_known_and_the_rest_unknown() {
    let b = backend();
    let buf = buffer(&b, 16, COPY);
    submit(
        &b,
        vec![upload(&[1, 2, 3, 4, 5, 6, 7, 8], slice(buf, 4, 8))],
    );
    assert_eq!(
        read(&b, slice(buf, 4, 8)),
        bytes(vec![1, 2, 3, 4, 5, 6, 7, 8])
    );
    assert_eq!(read(&b, slice(buf, 6, 2)), bytes(vec![3, 4]));
    // One unknown byte at either end makes the whole readback unknown.
    assert_eq!(read(&b, slice(buf, 3, 9)), Readback::Unknown);
    assert_eq!(read(&b, slice(buf, 4, 9)), Readback::Unknown);
    // A buffer nothing wrote.
    let other = buffer(&b, 16, COPY);
    assert_eq!(read(&b, slice(other, 0, 16)), Readback::Unknown);
}

/// Uploads a whole `width` by `height` single-layer image of `format` with bytes counting from 1,
/// then reads back the region at `left`, `top` of `cut_w` by `cut_h` texels and checks it against
/// the same region cut by hand from the uploaded rows.
fn upload_and_cut(
    format: PixelFormat,
    (width, height): (u32, u32),
    (left, top, cut_w, cut_h): (u32, u32, u32, u32),
) {
    let backend = backend();
    let tex = texture(&backend, format, (width, height), 1, 1);
    let (bw, bh) = (format.block_width(), format.block_height());
    let to_usize = |v: u32| ok(usize::try_from(v));
    let block = to_usize(format.bytes_per_block());
    let row = to_usize(width.div_ceil(bw)) * block;
    let image = counting(1, row * to_usize(height.div_ceil(bh)));
    submit(
        &backend,
        vec![upload(
            &image,
            region(tex, 0, (0, 0, 0), (width, height, 1)),
        )],
    );
    let (first_col, first_row) = (to_usize(left / bw), to_usize(top / bh));
    let cols = to_usize(cut_w.div_ceil(bw));
    let rows = to_usize(cut_h.div_ceil(bh));
    let mut expected = Vec::new();
    for row_index in first_row..first_row + rows {
        let start = row_index * row + first_col * block;
        expected.extend_from_slice(&image[start..start + cols * block]);
    }
    assert_eq!(
        read(&backend, region(tex, 0, (left, top, 0), (cut_w, cut_h, 1))),
        bytes(expected),
        "{format}"
    );
}

#[test]
fn texture_upload_and_readback_across_formats() {
    // 8, 16, 32, 64 and 128 bits per texel, and two block-compressed formats.
    upload_and_cut(PixelFormat::R8Unorm, (5, 3), (1, 1, 3, 2));
    upload_and_cut(PixelFormat::R5G6B5Unorm, (4, 4), (2, 0, 2, 4));
    upload_and_cut(PixelFormat::A8B8G8R8Unorm, (4, 4), (1, 2, 2, 2));
    upload_and_cut(PixelFormat::R16G16B16A16Float, (3, 2), (2, 1, 1, 1));
    upload_and_cut(PixelFormat::R32G32B32A32Float, (2, 2), (0, 1, 2, 1));
    // BC1: 4x4 blocks of 8 bytes; an 8x8 image is 2x2 blocks.
    upload_and_cut(PixelFormat::Bc1RgbaUnorm, (8, 8), (4, 0, 4, 4));
    // ASTC 8x5 on a 10x7 image: 2x2 blocks, the right and bottom ones partial; a region that
    // reaches the edge may cover a partial block.
    upload_and_cut(PixelFormat::Astc2d8x5Unorm, (10, 7), (8, 5, 2, 2));
}

#[test]
fn hand_computed_rgba8_region() {
    // A 2x2 RGBA8 texture, texel (x, y) holding bytes [x, y, 7, 9]; the right column read back.
    let b = backend();
    let t = texture(&b, PixelFormat::A8B8G8R8Unorm, (2, 2), 1, 1);
    let image = [0, 0, 7, 9, 1, 0, 7, 9, 0, 1, 7, 9, 1, 1, 7, 9];
    submit(&b, vec![upload(&image, region(t, 0, (0, 0, 0), (2, 2, 1)))]);
    assert_eq!(
        read(&b, region(t, 0, (1, 0, 0), (1, 2, 1))),
        bytes(vec![1, 0, 7, 9, 1, 1, 7, 9])
    );
}

#[test]
fn layers_and_mips_are_separate() {
    let b = backend();
    let t = texture(&b, PixelFormat::R8G8Unorm, (4, 4), 3, 2);
    // Layer 1 of mip 1 (2x2), and nothing else.
    submit(
        &b,
        vec![upload(&counting(10, 8), region(t, 1, (0, 0, 1), (2, 2, 1)))],
    );
    assert_eq!(
        read(&b, region(t, 1, (0, 0, 1), (2, 2, 1))),
        bytes(counting(10, 8))
    );
    assert_eq!(
        read(&b, region(t, 1, (0, 0, 0), (2, 2, 1))),
        Readback::Unknown
    );
    assert_eq!(
        read(&b, region(t, 1, (0, 0, 2), (2, 2, 1))),
        Readback::Unknown
    );
    assert_eq!(
        read(&b, region(t, 0, (0, 0, 1), (2, 2, 1))),
        Readback::Unknown
    );
    // Two layers at once are packed layer after layer.
    submit(
        &b,
        vec![upload(&counting(50, 8), region(t, 1, (0, 0, 2), (2, 2, 1)))],
    );
    let mut both = counting(10, 8);
    both.extend(counting(50, 8));
    assert_eq!(read(&b, region(t, 1, (0, 0, 1), (2, 2, 2))), bytes(both));
}

#[test]
fn copies_between_buffers_and_textures() {
    let b = backend();
    let src = buffer(&b, 64, COPY);
    let dst = buffer(&b, 64, COPY);
    let t = texture(&b, PixelFormat::R16Uint, (4, 2), 1, 1);
    let u = texture(&b, PixelFormat::R16Uint, (4, 2), 2, 1);
    submit(
        &b,
        vec![
            upload(&counting(0, 16), slice(src, 0, 16)),
            // Buffer to texture: 16 tightly packed bytes are the whole 4x2 image.
            Command::Copy(CopyDesc {
                src: slice(src, 0, 16),
                dst: region(t, 0, (0, 0, 0), (4, 2, 1)),
            }),
            // Texture to texture: the right half into layer 1 of `u`, at the left.
            Command::Copy(CopyDesc {
                src: region(t, 0, (2, 0, 0), (2, 2, 1)),
                dst: region(u, 0, (0, 0, 1), (2, 2, 1)),
            }),
            // Texture to buffer: that layer's left half, packed, at offset 32.
            Command::Copy(CopyDesc {
                src: region(u, 0, (0, 0, 1), (2, 2, 1)),
                dst: slice(dst, 32, 8),
            }),
            // Buffer to buffer.
            Command::Copy(CopyDesc {
                src: slice(src, 2, 4),
                dst: slice(dst, 0, 4),
            }),
        ],
    );
    // Row 0 of `t` is bytes 0..8, row 1 bytes 8..16; its right half is bytes 4..8 and 12..16.
    assert_eq!(
        read(&b, slice(dst, 32, 8)),
        bytes(vec![4, 5, 6, 7, 12, 13, 14, 15])
    );
    assert_eq!(read(&b, slice(dst, 0, 4)), bytes(vec![2, 3, 4, 5]));
    assert_eq!(read(&b, slice(dst, 4, 1)), Readback::Unknown);
    assert_eq!(
        read(&b, region(u, 0, (0, 0, 0), (2, 2, 1))),
        Readback::Unknown
    );
}

#[test]
fn a_copy_carries_unknown_bytes_as_unknown() {
    let b = backend();
    let src = buffer(&b, 8, COPY);
    let dst = buffer(&b, 8, COPY);
    submit(
        &b,
        vec![
            upload(&[9; 8], slice(dst, 0, 8)),
            upload(&[1, 2], slice(src, 0, 2)),
            // src bytes 2..4 are unknown: copying them makes dst 2..4 unknown again.
            Command::Copy(CopyDesc {
                src: slice(src, 0, 4),
                dst: slice(dst, 0, 4),
            }),
        ],
    );
    assert_eq!(read(&b, slice(dst, 0, 2)), bytes(vec![1, 2]));
    assert_eq!(read(&b, slice(dst, 2, 1)), Readback::Unknown);
    assert_eq!(read(&b, slice(dst, 3, 1)), Readback::Unknown);
    assert_eq!(read(&b, slice(dst, 4, 4)), bytes(vec![9; 4]));
}

#[test]
fn clears_across_formats() {
    let b = backend();
    for (format, value, texel) in [
        (
            PixelFormat::A8B8G8R8Unorm,
            ClearValue::Float([1.0, 0.0, 1.0, 0.0]),
            vec![0xff, 0, 0xff, 0],
        ),
        (
            PixelFormat::B8G8R8A8Unorm,
            ClearValue::Float([1.0, 0.0, 0.0, 1.0]),
            vec![0, 0, 0xff, 0xff],
        ),
        (
            PixelFormat::R16G16Float,
            ClearValue::Float([0.5, -2.0, 0.0, 0.0]),
            vec![0x00, 0x38, 0x00, 0xc0],
        ),
        (
            PixelFormat::R32Uint,
            ClearValue::Uint([0xdead_beef, 0, 0, 0]),
            vec![0xef, 0xbe, 0xad, 0xde],
        ),
        (
            PixelFormat::R8G8Sint,
            ClearValue::Sint([-1, 5, 0, 0]),
            vec![0xff, 0x05],
        ),
        (
            PixelFormat::D16Unorm,
            ClearValue::Depth(1.0),
            vec![0xff, 0xff],
        ),
    ] {
        let t = texture(&b, format, (3, 2), 1, 1);
        let v = view(&b, t, format, 0, (0, 1));
        submit(&b, vec![clear(v, value, Some(rect(1, 0, 2, 2)))]);
        assert_eq!(
            read(&b, region(t, 0, (1, 0, 0), (2, 2, 1))),
            bytes(texel.repeat(4)),
            "{format}"
        );
        // Column 0 was outside the rectangle.
        assert_eq!(
            read(&b, region(t, 0, (0, 0, 0), (1, 1, 1))),
            Readback::Unknown,
            "{format}"
        );
    }
}

#[test]
fn an_inexact_clear_makes_its_area_unknown() {
    let b = backend();
    let f = PixelFormat::A8B8G8R8Unorm;
    let t = texture(&b, f, (2, 1), 1, 1);
    let v = view(&b, t, f, 0, (0, 1));
    submit(
        &b,
        vec![
            upload(&[1; 8], region(t, 0, (0, 0, 0), (2, 1, 1))),
            // 0.5 on 8 bits is 127.5: a device may write either neighbour.
            clear(
                v,
                ClearValue::Float([0.5, 0.0, 0.0, 1.0]),
                Some(rect(0, 0, 1, 1)),
            ),
        ],
    );
    assert_eq!(
        read(&b, region(t, 0, (0, 0, 0), (1, 1, 1))),
        Readback::Unknown
    );
    assert_eq!(
        read(&b, region(t, 0, (1, 0, 0), (1, 1, 1))),
        bytes(vec![1; 4])
    );
}

#[test]
fn a_clear_covers_the_view_layers_and_only_its_base_mip() {
    let b = backend();
    let f = PixelFormat::R8Uint;
    let t = texture(&b, f, (2, 2), 4, 2);
    let v = view(&b, t, f, 0, (1, 2));
    submit(&b, vec![clear(v, ClearValue::Uint([7, 0, 0, 0]), None)]);
    assert_eq!(
        read(&b, region(t, 0, (0, 0, 1), (2, 2, 2))),
        bytes(vec![7; 8])
    );
    for layer in [0, 3] {
        assert_eq!(
            read(&b, region(t, 0, (0, 0, layer), (2, 2, 1))),
            Readback::Unknown,
            "layer {layer}"
        );
    }
    assert_eq!(
        read(&b, region(t, 1, (0, 0, 1), (1, 1, 1))),
        Readback::Unknown
    );
}

#[test]
fn a_clear_of_a_3d_view_covers_every_slice_of_its_level() {
    let b = backend();
    let f = PixelFormat::R8Uint;
    let t = b
        .create_texture(&super::fixtures::texture_desc(
            f,
            TextureDimension::D3,
            (2, 2, 4),
            2,
        ))
        .expect("valid");
    let v = b
        .create_view(
            t,
            &starquake_render::ViewDesc {
                format: f,
                dimension: starquake_render::resource::ViewDimension::D3,
                base_mip: 1,
                mip_count: 1,
                base_layer: 0,
                layer_count: 1,
                swizzle: starquake_render::resource::Swizzle::IDENTITY,
            },
        )
        .expect("valid");
    submit(&b, vec![clear(v, ClearValue::Uint([3, 0, 0, 0]), None)]);
    // Mip 1 of a 2x2x4 texture is 1x1x2.
    assert_eq!(
        read(&b, region(t, 1, (0, 0, 0), (1, 1, 2))),
        bytes(vec![3, 3])
    );
    assert_eq!(
        read(&b, region(t, 0, (0, 0, 0), (1, 1, 1))),
        Readback::Unknown
    );
}

fn blit(
    src: starquake_render::ViewId,
    s: (u32, u32, u32, u32),
    dst: starquake_render::ViewId,
    d: (u32, u32, u32, u32),
) -> Command {
    Command::Blit(BlitDesc {
        src,
        src_rect: rect(s.0, s.1, s.2, s.3),
        dst,
        dst_rect: rect(d.0, d.1, d.2, d.3),
        filter: FilterMode::Linear,
    })
}

#[test]
fn an_unscaled_blit_of_one_format_copies_texels() {
    let b = backend();
    let f = PixelFormat::A8B8G8R8Unorm;
    let s = texture(&b, f, (4, 2), 2, 1);
    let d = texture(&b, f, (4, 4), 1, 1);
    // The blit reads the source view's base layer: layer 1.
    let sv = view(&b, s, f, 0, (1, 1));
    let dv = view(&b, d, f, 0, (0, 1));
    submit(
        &b,
        vec![
            upload(&counting(0, 32), region(s, 0, (0, 0, 1), (4, 2, 1))),
            blit(sv, (2, 0, 2, 2), dv, (0, 2, 2, 2)),
        ],
    );
    // Source row 0 texels 2..4 are bytes 8..16, row 1 bytes 24..32.
    let mut expected = counting(8, 8);
    expected.extend(counting(24, 8));
    assert_eq!(
        read(&b, region(d, 0, (0, 2, 0), (2, 2, 1))),
        bytes(expected)
    );
    assert_eq!(
        read(&b, region(d, 0, (0, 0, 0), (2, 2, 1))),
        Readback::Unknown
    );
}

#[test]
fn scaled_or_converting_blits_make_their_destination_unknown() {
    let b = backend();
    let f = PixelFormat::A8B8G8R8Unorm;
    let s = texture(&b, f, (4, 4), 1, 1);
    let d = texture(&b, f, (4, 4), 1, 1);
    let sv = view(&b, s, f, 0, (0, 1));
    let dv = view(&b, d, f, 0, (0, 1));
    // The same storage seen as sRGB: same block size, another format.
    let d_srgb = view(&b, d, PixelFormat::A8B8G8R8Srgb, 0, (0, 1));
    let known = |b: &starquake_render::RecordingBackend| {
        submit(
            b,
            vec![
                upload(&[5; 64], region(s, 0, (0, 0, 0), (4, 4, 1))),
                upload(&[6; 64], region(d, 0, (0, 0, 0), (4, 4, 1))),
            ],
        );
    };
    known(&b);
    submit(&b, vec![blit(sv, (0, 0, 2, 2), dv, (0, 0, 4, 2))]);
    assert_eq!(
        read(&b, region(d, 0, (0, 0, 0), (4, 2, 1))),
        Readback::Unknown
    );
    assert_eq!(
        read(&b, region(d, 0, (0, 2, 0), (4, 2, 1))),
        bytes(vec![6; 32])
    );
    known(&b);
    submit(&b, vec![blit(sv, (0, 0, 2, 2), d_srgb, (2, 2, 2, 2))]);
    assert_eq!(
        read(&b, region(d, 0, (2, 2, 0), (2, 2, 1))),
        Readback::Unknown
    );
    assert_eq!(
        read(&b, region(d, 0, (0, 0, 0), (4, 2, 1))),
        bytes(vec![6; 32])
    );
}
