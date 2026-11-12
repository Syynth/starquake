//! The CPU model: uploads, copies, clears and blits across formats, with pixel results computed
//! by hand, and readbacks of what nothing (or a draw) wrote.

use super::fixtures::{
    MEMORY, backend, buffer, caps, clear, counting, ok, read, rect, region, slice, submit, texture,
    upload, view,
};
use starquake_render::command::{BlitDesc, ClearValue, CopyDesc, PresentDesc, PresentTransform};
use starquake_render::resource::{
    FilterMode, Swizzle, SwizzleSource, TextureDimension, ViewDimension,
};
use starquake_render::{
    Backend, BufferUsage, Caps, Command, CommandList, PixelFormat, Readback, RecordingBackend,
    TextureId, ViewDesc, ViewId,
};

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
    src: ViewId,
    s: (u32, u32, u32, u32),
    dst: ViewId,
    d: (u32, u32, u32, u32),
    filter: FilterMode,
) -> Command {
    Command::Blit(BlitDesc {
        src,
        src_rect: rect(s.0, s.1, s.2, s.3),
        dst,
        dst_rect: rect(d.0, d.1, d.2, d.3),
        filter,
    })
}

#[test]
fn an_unscaled_integer_blit_copies_texels() {
    let b = backend();
    let f = PixelFormat::A8B8G8R8Uint;
    let s = texture(&b, f, (4, 2), 2, 1);
    let d = texture(&b, f, (4, 4), 1, 1);
    // The blit reads the source view's base layer: layer 1.
    let sv = view(&b, s, f, 0, (1, 1));
    let dv = view(&b, d, f, 0, (0, 1));
    submit(
        &b,
        vec![
            upload(&counting(0, 32), region(s, 0, (0, 0, 1), (4, 2, 1))),
            blit(sv, (2, 0, 2, 2), dv, (0, 2, 2, 2), FilterMode::Nearest),
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
    // Signed integers copy too.
    let sint = PixelFormat::R8Sint;
    let (si, di) = (
        texture(&b, sint, (2, 1), 1, 1),
        texture(&b, sint, (2, 1), 1, 1),
    );
    let (siv, div) = (view(&b, si, sint, 0, (0, 1)), view(&b, di, sint, 0, (0, 1)));
    submit(
        &b,
        vec![
            upload(&[0x80, 0x7f], region(si, 0, (0, 0, 0), (2, 1, 1))),
            blit(siv, (0, 0, 2, 1), div, (0, 0, 2, 1), FilterMode::Nearest),
        ],
    );
    assert_eq!(
        read(&b, region(di, 0, (0, 0, 0), (2, 1, 1))),
        bytes(vec![0x80, 0x7f])
    );
}

/// Uploads `src_bytes` to a `size` texture of `format`, `[9; ..]` to a second one, blits the
/// whole first into the whole second (unscaled), and returns what the second reads back.
fn unscaled_blit_of(format: PixelFormat, size: (u32, u32), src_bytes: &[u8]) -> Readback {
    let b = backend();
    let s = texture(&b, format, size, 1, 1);
    let d = texture(&b, format, size, 1, 1);
    let (sv, dv) = (
        view(&b, s, format, 0, (0, 1)),
        view(&b, d, format, 0, (0, 1)),
    );
    let whole = (0, 0, size.0, size.1);
    submit(
        &b,
        vec![
            upload(src_bytes, region(s, 0, (0, 0, 0), (size.0, size.1, 1))),
            upload(
                &vec![9; src_bytes.len()],
                region(d, 0, (0, 0, 0), (size.0, size.1, 1)),
            ),
            blit(sv, whole, dv, whole, FilterMode::Nearest),
        ],
    );
    read(&b, region(d, 0, (0, 0, 0), (size.0, size.1, 1)))
}

#[test]
fn blits_that_convert_through_float_make_their_destination_unknown() {
    // The review's probe: snorm 0x80 is -1.0 to a blit, written back as 0x81.
    assert_eq!(
        unscaled_blit_of(PixelFormat::R8Snorm, (1, 1), &[0x80]),
        Readback::Unknown
    );
    for (format, texel) in [
        (PixelFormat::R8Unorm, vec![0x80]),
        (PixelFormat::A8B8G8R8Srgb, vec![1, 2, 3, 4]),
        // A binary16 NaN with a payload, and a subnormal.
        (PixelFormat::R16Float, vec![0x01, 0x7e]),
        (PixelFormat::R16Float, vec![0x01, 0x00]),
        (PixelFormat::R32Float, vec![0x01, 0x00, 0xc0, 0x7f]),
        (PixelFormat::D32Float, vec![0, 0, 0x80, 0x3f]),
        (PixelFormat::D16Unorm, vec![0x34, 0x12]),
    ] {
        assert_eq!(
            unscaled_blit_of(format, (1, 1), &texel),
            Readback::Unknown,
            "{format}"
        );
    }
    // The same blit of an integer format copies.
    assert_eq!(
        unscaled_blit_of(PixelFormat::R8Uint, (1, 1), &[0x80]),
        bytes(vec![0x80])
    );
}

#[test]
fn a_blit_scaled_in_one_direction_makes_its_destination_unknown() {
    let b = backend();
    let f = PixelFormat::R8Uint;
    let s = texture(&b, f, (4, 4), 1, 1);
    let d = texture(&b, f, (4, 4), 1, 1);
    let (sv, dv) = (view(&b, s, f, 0, (0, 1)), view(&b, d, f, 0, (0, 1)));
    let known = |b: &RecordingBackend| {
        submit(
            b,
            vec![
                upload(&[5; 16], region(s, 0, (0, 0, 0), (4, 4, 1))),
                upload(&[6; 16], region(d, 0, (0, 0, 0), (4, 4, 1))),
            ],
        );
    };
    // 2x2 to 2x4: equal widths, different heights. Applying it as a copy would fail the size
    // check midway; it must run and leave the destination unknown.
    known(&b);
    assert_eq!(
        b.submit(CommandList {
            commands: vec![blit(
                sv,
                (0, 0, 2, 2),
                dv,
                (0, 0, 2, 4),
                FilterMode::Nearest
            )]
        })
        .map(|_| ()),
        Ok(())
    );
    assert_eq!(
        read(&b, region(d, 0, (0, 0, 0), (2, 4, 1))),
        Readback::Unknown
    );
    assert_eq!(
        read(&b, region(d, 0, (2, 0, 0), (2, 4, 1))),
        bytes(vec![6; 8])
    );
    // 2x2 to 4x2: equal heights, different widths.
    known(&b);
    submit(
        &b,
        vec![blit(
            sv,
            (0, 0, 2, 2),
            dv,
            (0, 0, 4, 2),
            FilterMode::Nearest,
        )],
    );
    assert_eq!(
        read(&b, region(d, 0, (0, 0, 0), (4, 2, 1))),
        Readback::Unknown
    );
    assert_eq!(
        read(&b, region(d, 0, (0, 2, 0), (4, 2, 1))),
        bytes(vec![6; 8])
    );
}

#[test]
fn compressed_blits() {
    let b = backend();
    let bc = PixelFormat::Bc1RgbaUnorm;
    let f = PixelFormat::A8B8G8R8Unorm;
    let s = texture(&b, bc, (4, 4), 1, 1);
    let d = texture(&b, f, (4, 4), 1, 1);
    let other_bc = texture(&b, bc, (4, 4), 1, 1);
    let sv = view(&b, s, bc, 0, (0, 1));
    let dv = view(&b, d, f, 0, (0, 1));
    let bv = view(&b, other_bc, bc, 0, (0, 1));
    // Unscaled and block-aligned, between known blocks: still refused into a compressed view.
    submit(
        &b,
        vec![
            upload(&[1; 8], region(s, 0, (0, 0, 0), (4, 4, 1))),
            upload(&[2; 8], region(other_bc, 0, (0, 0, 0), (4, 4, 1))),
            upload(&[3; 64], region(d, 0, (0, 0, 0), (4, 4, 1))),
        ],
    );
    assert_eq!(
        b.submit(CommandList {
            commands: vec![blit(
                sv,
                (0, 0, 4, 4),
                bv,
                (0, 0, 4, 4),
                FilterMode::Nearest
            )]
        }),
        Err(starquake_render::RenderError::Invalid(
            "a blit destination must not be compressed"
        ))
    );
    assert_eq!(
        read(&b, region(other_bc, 0, (0, 0, 0), (4, 4, 1))),
        bytes(vec![2; 8])
    );
    // A compressed source decodes through float: the destination is unknown.
    submit(
        &b,
        vec![blit(
            sv,
            (0, 0, 4, 4),
            dv,
            (0, 0, 4, 4),
            FilterMode::Nearest,
        )],
    );
    assert_eq!(
        read(&b, region(d, 0, (0, 0, 0), (4, 4, 1))),
        Readback::Unknown
    );
}

/// A backend that allows view swizzles.
fn swizzling_backend() -> RecordingBackend {
    RecordingBackend::new(
        Caps {
            view_swizzle: true,
            ..caps()
        },
        MEMORY,
    )
}

/// Red and blue swapped.
const SWAP_RB: Swizzle = Swizzle {
    r: SwizzleSource::B,
    g: SwizzleSource::G,
    b: SwizzleSource::R,
    a: SwizzleSource::A,
};

fn swizzled_view(
    b: &RecordingBackend,
    t: TextureId,
    format: PixelFormat,
    swizzle: Swizzle,
) -> ViewId {
    ok(b.create_view(
        t,
        &ViewDesc {
            format,
            dimension: ViewDimension::D2Array,
            base_mip: 0,
            mip_count: 1,
            base_layer: 0,
            layer_count: 1,
            swizzle,
        },
    ))
}

#[test]
fn blits_and_dumps_through_swizzled_views_are_unknown() {
    let b = swizzling_backend();
    let uint = PixelFormat::A8B8G8R8Uint;
    let s = texture(&b, uint, (1, 1), 1, 1);
    let d = texture(&b, uint, (1, 1), 1, 1);
    let (s_plain, s_swapped) = (
        swizzled_view(&b, s, uint, Swizzle::IDENTITY),
        swizzled_view(&b, s, uint, SWAP_RB),
    );
    let (d_plain, d_swapped) = (
        swizzled_view(&b, d, uint, Swizzle::IDENTITY),
        swizzled_view(&b, d, uint, SWAP_RB),
    );
    let one = (0, 0, 1, 1);
    for (src, dst, expected, what) in [
        (s_plain, d_plain, bytes(vec![1, 2, 3, 4]), "identity"),
        (s_swapped, d_plain, Readback::Unknown, "swizzled source"),
        (
            s_plain,
            d_swapped,
            Readback::Unknown,
            "swizzled destination",
        ),
    ] {
        submit(
            &b,
            vec![
                upload(&[1, 2, 3, 4], region(s, 0, (0, 0, 0), (1, 1, 1))),
                upload(&[9; 4], region(d, 0, (0, 0, 0), (1, 1, 1))),
                blit(src, one, dst, one, FilterMode::Nearest),
            ],
        );
        assert_eq!(
            read(&b, region(d, 0, (0, 0, 0), (1, 1, 1))),
            expected,
            "{what}"
        );
    }

    // A present samples its view: a swizzled view's dump is unknown.
    let unorm = PixelFormat::A8B8G8R8Unorm;
    let t = texture(&b, unorm, (1, 1), 1, 1);
    let present = |view, frame| {
        Command::Present(PresentDesc {
            view,
            src_rect: rect(0, 0, 1, 1),
            transform: PresentTransform::Identity,
            frame,
            dump: true,
        })
    };
    submit(
        &b,
        vec![
            upload(&[1, 2, 3, 4], region(t, 0, (0, 0, 0), (1, 1, 1))),
            present(swizzled_view(&b, t, unorm, Swizzle::IDENTITY), 1),
            present(swizzled_view(&b, t, unorm, SWAP_RB), 2),
        ],
    );
    let pixels: Vec<(u64, Readback)> = b
        .take_frame_dumps()
        .into_iter()
        .map(|d| (d.frame, d.pixels))
        .collect();
    assert_eq!(
        pixels,
        vec![(1, bytes(vec![1, 2, 3, 4])), (2, Readback::Unknown)]
    );
}

#[test]
fn combined_depth_stencil_and_x8_d24_bytes_are_never_known() {
    const COPY_BUF: BufferUsage = COPY;
    for format in [
        PixelFormat::D24UnormS8Uint,
        PixelFormat::S8UintD24Unorm,
        PixelFormat::D32FloatS8Uint,
        PixelFormat::X8D24Unorm,
    ] {
        let b = backend();
        let t = texture(&b, format, (2, 1), 1, 1);
        let u = texture(&b, format, (2, 1), 1, 1);
        let len = usize::try_from(2 * format.bytes_per_block()).expect("small");
        let size = u64::try_from(len).expect("small");
        let image = counting(1, len);
        // The review's probe: an upload read straight back.
        submit(&b, vec![upload(&image, region(t, 0, (0, 0, 0), (2, 1, 1)))]);
        assert_eq!(
            read(&b, region(t, 0, (0, 0, 0), (2, 1, 1))),
            Readback::Unknown,
            "{format} readback"
        );
        // A copy out to a buffer makes the buffer's range unknown, and only that range.
        let buf = buffer(&b, 2 * size, COPY_BUF);
        submit(
            &b,
            vec![
                upload(&vec![7; 2 * len], slice(buf, 0, 2 * size)),
                Command::Copy(CopyDesc {
                    src: region(t, 0, (0, 0, 0), (2, 1, 1)),
                    dst: slice(buf, 0, size),
                }),
                // A buffer copied in leaves the texture unknown.
                Command::Copy(CopyDesc {
                    src: slice(buf, size, size),
                    dst: region(u, 0, (0, 0, 0), (2, 1, 1)),
                }),
            ],
        );
        assert_eq!(read(&b, slice(buf, 0, size)), Readback::Unknown, "{format}");
        assert_eq!(
            read(&b, slice(buf, size, size)),
            bytes(vec![7; len]),
            "{format}"
        );
        assert_eq!(
            read(&b, region(u, 0, (0, 0, 0), (2, 1, 1))),
            Readback::Unknown,
            "{format} copied in"
        );
    }
    // One-aspect depth formats without padding move exactly.
    for format in [
        PixelFormat::D32Float,
        PixelFormat::D16Unorm,
        PixelFormat::S8Uint,
    ] {
        let b = backend();
        let t = texture(&b, format, (2, 1), 1, 1);
        let len = usize::try_from(2 * format.bytes_per_block()).expect("small");
        submit(
            &b,
            vec![upload(
                &counting(1, len),
                region(t, 0, (0, 0, 0), (2, 1, 1)),
            )],
        );
        assert_eq!(
            read(&b, region(t, 0, (0, 0, 0), (2, 1, 1))),
            bytes(counting(1, len)),
            "{format}"
        );
    }
}

#[test]
fn scaled_or_converting_blits_make_their_destination_unknown() {
    let b = backend();
    let f = PixelFormat::A8B8G8R8Uint;
    let s = texture(&b, f, (4, 4), 1, 1);
    let d = texture(&b, f, (4, 4), 1, 1);
    let sv = view(&b, s, f, 0, (0, 1));
    let dv = view(&b, d, f, 0, (0, 1));
    // The same storage seen as another integer format of the same kind and block size.
    let d_other = view(&b, d, PixelFormat::R32Uint, 0, (0, 1));
    let known = |b: &RecordingBackend| {
        submit(
            b,
            vec![
                upload(&[5; 64], region(s, 0, (0, 0, 0), (4, 4, 1))),
                upload(&[6; 64], region(d, 0, (0, 0, 0), (4, 4, 1))),
            ],
        );
    };
    known(&b);
    submit(
        &b,
        vec![blit(
            sv,
            (0, 0, 2, 2),
            dv,
            (0, 0, 4, 2),
            FilterMode::Nearest,
        )],
    );
    assert_eq!(
        read(&b, region(d, 0, (0, 0, 0), (4, 2, 1))),
        Readback::Unknown
    );
    assert_eq!(
        read(&b, region(d, 0, (0, 2, 0), (4, 2, 1))),
        bytes(vec![6; 32])
    );
    known(&b);
    submit(
        &b,
        vec![blit(
            sv,
            (0, 0, 2, 2),
            d_other,
            (2, 2, 2, 2),
            FilterMode::Nearest,
        )],
    );
    assert_eq!(
        read(&b, region(d, 0, (2, 2, 0), (2, 2, 1))),
        Readback::Unknown
    );
    assert_eq!(
        read(&b, region(d, 0, (0, 0, 0), (4, 2, 1))),
        bytes(vec![6; 32])
    );
}
