//! The render thread's pure parts: the copies a buffer-texture copy splits into, and the host
//! table.

use super::{HostTable, row_copies};
use crate::plan::{Rows, TexelBox};
use starquake_render::TextureId;
use starquake_render::resource::{Extent3d, Origin3d};

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
    let copies = row_copies(at(64, 3, 2), 12, rows, 1);
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
    let copies = row_copies(at(3, 1, 1), 8, rows, 1);
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
    let (layout, _) = row_copies(at(65, 1, 1), 0, wide, 1)[0];
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
    let copies = row_copies(at(12, 12, 2), 100, rows, 4);
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
    assert_eq!(row_copies(at(12, 12, 1), 0, one_image, 4).len(), 3);
    let one_row = Rows {
        rows: 1,
        images: 2,
        ..rows
    };
    assert_eq!(row_copies(at(12, 4, 2), 0, one_row, 4).len(), 2);
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
