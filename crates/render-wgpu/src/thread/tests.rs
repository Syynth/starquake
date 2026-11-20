//! The render thread's pure parts: the copies a buffer-texture copy splits into, a texture
//! read's one copy and its unpadding, the scratch sizes of a list's clears, and the host table.

use super::{HostTable, read_copy_layout, row_copies, scratch_sizes, unpad};
use crate::clear::ClearHow;
use crate::plan::{ClearOps, ClearStep, Layers, Rows, Step, TexelBox};
use crate::rules::padded_read;
use starquake_render::command::Rect;
use starquake_render::resource::{Extent3d, Origin3d};
use starquake_render::{BufferId, TextureId};

/// The copies [`row_copies`] makes, collected.
fn collected(
    at: TexelBox,
    offset: u64,
    rows: Rows,
    block_height: u32,
) -> Vec<(wgpu::TexelCopyBufferLayout, TexelBox)> {
    row_copies(at, offset, rows, block_height).collect()
}

fn at(width: u32, height: u32, depth: u32) -> TexelBox {
    TexelBox {
        texture: TextureId::new(3, 1),
        mip: 2,
        origin: Origin3d { x: 4, y: 8, z: 1 },
        extent: Extent3d {
            width,
            height,
            depth_or_layers: depth,
        },
    }
}

#[test]
fn aligned_rows_are_one_copy() {
    let rows = Rows {
        row_bytes: wgpu::COPY_BYTES_PER_ROW_ALIGNMENT,
        rows: 3,
        images: 2,
    };
    let copies = collected(at(64, 3, 2), 12, rows, 1);
    assert_eq!(copies.len(), 1);
    let (layout, part) = copies[0];
    assert_eq!(layout.offset, 12);
    assert_eq!(
        layout.bytes_per_row,
        Some(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
    );
    assert_eq!(layout.rows_per_image, Some(3));
    assert_eq!(part, at(64, 3, 2));
}

#[test]
fn one_unaligned_row_is_one_copy_with_a_padded_stride() {
    let rows = Rows {
        row_bytes: 12,
        rows: 1,
        images: 1,
    };
    let copies = collected(at(3, 1, 1), 8, rows, 1);
    assert_eq!(copies.len(), 1);
    let (layout, part) = copies[0];
    // Never `None`: wgpu-hal's Metal encoder turns a missing stride into 0 (module docs).
    assert_eq!(
        (layout.offset, layout.bytes_per_row, layout.rows_per_image),
        (8, Some(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT), Some(1))
    );
    assert_eq!(part, at(3, 1, 1));
    // One byte past the alignment pads to the next multiple.
    let wide = Rows {
        row_bytes: wgpu::COPY_BYTES_PER_ROW_ALIGNMENT + 4,
        ..rows
    };
    let (layout, _) = collected(at(65, 1, 1), 0, wide, 1)[0];
    assert_eq!(
        layout.bytes_per_row,
        Some(2 * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
    );
}

#[test]
fn unaligned_rows_are_one_copy_per_row_in_packed_order() {
    // Two images of three rows of 4x4 blocks (block height 4), 24 bytes a row.
    let rows = Rows {
        row_bytes: 24,
        rows: 3,
        images: 2,
    };
    let copies = collected(at(12, 12, 2), 100, rows, 4);
    assert_eq!(copies.len(), 6);
    for (i, (layout, part)) in copies.iter().enumerate() {
        let (image, row) = (i / 3, i % 3);
        let i = u64::try_from(i).expect("small");
        assert_eq!(layout.offset, 100 + 24 * i, "copy {i}");
        assert_eq!(
            (layout.bytes_per_row, layout.rows_per_image),
            (Some(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT), Some(1))
        );
        let row = u32::try_from(row).expect("small");
        let image = u32::try_from(image).expect("small");
        assert_eq!(
            part.origin,
            Origin3d {
                x: 4,
                y: 8 + 4 * row,
                z: 1 + image
            },
            "copy {i}"
        );
        assert_eq!(
            part.extent,
            Extent3d {
                width: 12,
                height: 4,
                depth_or_layers: 1
            }
        );
        assert_eq!((part.texture, part.mip), (TextureId::new(3, 1), 2));
    }
    // One image of several rows splits too; several images of one row each as well.
    let one_image = Rows { images: 1, ..rows };
    assert_eq!(collected(at(12, 12, 1), 0, one_image, 4).len(), 3);
    let one_row = Rows {
        rows: 1,
        images: 2,
        ..rows
    };
    assert_eq!(collected(at(12, 4, 2), 0, one_row, 4).len(), 2);
}

#[test]
fn host_tables_match_index_and_generation() {
    let mut t = HostTable::new();
    t.insert(2, 5, 'a');
    assert_eq!(t.get(2, 5), Some(&'a'));
    assert_eq!(t.get(2, 4), None, "an older generation");
    assert_eq!(t.get(2, 6), None, "a newer generation");
    assert_eq!(t.get(1, 5), None, "an empty slot below");
    assert_eq!(t.get(3, 5), None, "past the end");
    assert_eq!(t.remove(2, 4), None, "a stale id releases nothing");
    assert_eq!(t.get(2, 5), Some(&'a'));
    t.insert(3, 0, 'b');
    assert_eq!(t.remove(2, 5), Some('a'));
    assert_eq!(t.remove(2, 5), None);
    assert_eq!(t.get(3, 0), Some(&'b'), "the neighbour is unchanged");
    t.insert(2, 6, 'c');
    assert_eq!(t.get(2, 6), Some(&'c'));
    assert_eq!(t.get(2, 5), None);
}

#[test]
fn row_copies_are_made_one_at_a_time() {
    // The shape the review named: a 1-wide R8 texture of 16384 rows by 2048 layers. Its copies
    // are not collected: the first and one deep inside come out without the rest being made.
    let rows = Rows {
        row_bytes: 1,
        rows: 16384,
        images: 2048,
    };
    let mut it = row_copies(at(1, 16384, 2048), 0, rows, 1);
    let (first, _) = it.next().expect("a first copy");
    assert_eq!(first.offset, 0);
    let (layout, part) = row_copies(at(1, 16384, 2048), 0, rows, 1)
        .nth(16384 + 5)
        .expect("a copy in the second layer");
    assert_eq!(layout.offset, 16384 + 5);
    assert_eq!(
        (part.origin.y, part.origin.z),
        (8 + 5, 2),
        "row 5 of the second layer"
    );
}

#[test]
fn a_texture_read_is_one_copy_with_padded_rows() {
    // Unaligned rows, several rows and images: still one copy, rows 256 bytes apart.
    let rows = Rows {
        row_bytes: 7,
        rows: 5,
        images: 3,
    };
    let (stride, size) = padded_read(rows.row_bytes, rows.rows, rows.images).expect("small");
    assert_eq!(stride, wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
    assert_eq!(size, u64::from(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) * 5 * 3);
    let layout = read_copy_layout(rows, stride);
    assert_eq!(
        (layout.offset, layout.bytes_per_row, layout.rows_per_image),
        (0, Some(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT), Some(5))
    );
    // Aligned rows keep their own stride.
    let aligned = Rows {
        row_bytes: 2 * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT,
        ..rows
    };
    assert_eq!(
        padded_read(aligned.row_bytes, 5, 3).map(|p| p.0),
        Some(aligned.row_bytes)
    );
    // Unpadding takes each row's own bytes back, in packed order.
    let stride_bytes = usize::try_from(stride).expect("small");
    let mut padded = vec![0xee; stride_bytes * 15];
    for i in 0..15 {
        for b in 0..7 {
            padded[i * stride_bytes + b] = u8::try_from(i * 7 + b).expect("small");
        }
    }
    let packed: Vec<u8> = (0..105).map(|n| u8::try_from(n).expect("small")).collect();
    assert_eq!(unpad(&padded, rows, stride), Some(packed));
    // One byte short of the last row is refused, not read past.
    let short = &padded[..stride_bytes * 14 + 6];
    assert_eq!(unpad(short, rows, stride), None);
    assert!(unpad(&padded[..stride_bytes * 14 + 7], rows, stride).is_some());
}

fn clear(format: wgpu::TextureFormat, rect: Option<(u32, u32)>) -> Step {
    let how = if rect.is_some() {
        ClearHow::Scratch
    } else {
        ClearHow::Pass
    };
    clear_by(format, rect, how)
}

/// A rectangle clear of `format` drawn with the clear shader.
fn drawn(format: wgpu::TextureFormat, rect: (u32, u32)) -> Step {
    let how = ClearHow::Draw(crate::clear::DrawnClear {
        key: crate::clear::PipelineKey {
            format,
            samples: 1,
            target: crate::clear::DrawTarget::Color {
                kind: crate::clear::ColorKind::Float,
                mask: wgpu::ColorWrites::RED,
            },
        },
        params: [0; 16],
        stencil_reference: 0,
    });
    clear_by(format, Some(rect), how)
}

fn clear_by(format: wgpu::TextureFormat, rect: Option<(u32, u32)>, how: ClearHow) -> Step {
    Step::Clear(ClearStep {
        texture: TextureId::new(0, 0),
        format,
        mip: 0,
        layers: Layers::Array { first: 0, count: 1 },
        rect: rect.map(|(width, height)| Rect {
            x: 1,
            y: 1,
            width,
            height,
        }),
        size: (64, 64),
        ops: ClearOps::Color(wgpu::Color::BLACK),
        how,
    })
}

#[test]
fn a_lists_rectangle_clears_share_one_scratch_per_format() {
    use wgpu::TextureFormat::{R8Unorm, Rgba8Unorm, Rgba16Float};
    let other = Step::CopyBuffer {
        src: BufferId::new(0, 0),
        src_offset: 0,
        dst: BufferId::new(1, 0),
        dst_offset: 0,
        size: 4,
    };
    let steps = [
        clear(Rgba16Float, Some((30, 2))),
        other,
        clear(Rgba16Float, Some((4, 50))),
        clear(Rgba8Unorm, Some((9, 9))),
        clear(Rgba16Float, Some((12, 12))),
        // Whole-level clears need no scratch, and neither do drawn ones.
        clear(R8Unorm, None),
        drawn(R8Unorm, (5, 5)),
        drawn(Rgba8Unorm, (40, 40)),
    ];
    let sizes = scratch_sizes(&steps);
    assert_eq!(sizes.len(), 2);
    assert_eq!(
        sizes.get(&Rgba16Float),
        Some(&(30, 50)),
        "widest and tallest"
    );
    assert_eq!(sizes.get(&Rgba8Unorm), Some(&(9, 9)));
    assert_eq!(sizes.get(&R8Unorm), None);
    assert!(scratch_sizes(&[]).is_empty());
}
