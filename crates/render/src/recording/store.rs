//! The CPU copy of a buffer's or a texture mip level's contents, with what is known of it.
//!
//! Every byte has a known flag. A byte is known once a command with an exact answer wrote it
//! (an upload, a copy of known bytes, a modelled clear or blit); it is unknown before anything
//! wrote it and after a draw or dispatch may have written it. A store holds no memory until the
//! first write that could make something known, so a resource nothing writes costs nothing.
//!
//! Data moves through [`Rows`]: the byte ranges of a buffer range (one row) or of a box of
//! texture blocks (one row per block row, slice after slice), laid out as the tightly packed
//! form of `validate::packed_size`. Each `Rows` is checked against its store's length when it is
//! built, so walking it cannot leave the store.

use crate::error::RenderError;

/// The bytes of one buffer or mip level, and which of them are known.
#[derive(Debug)]
pub(super) struct Store {
    len: usize,
    data: Option<Data>,
}

#[derive(Debug)]
struct Data {
    bytes: Vec<u8>,
    known: Vec<bool>,
}

/// Bytes gathered from a store: either nothing known at all, or the bytes with a known flag
/// each.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum Packed {
    /// None of `len` bytes is known.
    Unknown {
        /// How many bytes.
        len: usize,
    },
    /// Bytes, each with whether it is known.
    Data {
        /// The bytes (zero where unknown).
        bytes: Vec<u8>,
        /// Whether each byte is known.
        known: Vec<bool>,
    },
}

impl Packed {
    /// Every byte known.
    pub(super) fn known(bytes: Vec<u8>) -> Self {
        let known = vec![true; bytes.len()];
        Self::Data { bytes, known }
    }

    /// The bytes, when every one is known.
    pub(super) fn into_known(self) -> Option<Vec<u8>> {
        match self {
            Self::Data { bytes, known } if known.iter().all(|k| *k) => Some(bytes),
            Self::Data { .. } | Self::Unknown { .. } => None,
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::Unknown { len } => *len,
            Self::Data { bytes, .. } => bytes.len(),
        }
    }
}

fn overflow() -> RenderError {
    RenderError::Overflow {
        what: "recorded contents offset",
    }
}

fn to_usize(v: u64) -> Result<usize, RenderError> {
    usize::try_from(v).map_err(|_| overflow())
}

/// A box of rows in a store: `slices` groups of `per_slice` rows of `row_bytes` bytes each.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Rows {
    first: usize,
    row_bytes: usize,
    per_slice: usize,
    row_stride: usize,
    slices: usize,
    slice_stride: usize,
}

/// A box of whole blocks in a mip level whose blocks are laid out `wide` by `high` by `deep`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct BlockBox {
    /// First block column.
    pub x: u32,
    /// First block row.
    pub y: u32,
    /// First slice or layer.
    pub z: u32,
    /// Blocks across.
    pub width: u32,
    /// Block rows.
    pub height: u32,
    /// Slices or layers.
    pub depth: u32,
}

/// The block layout of a whole mip level.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct LevelShape {
    /// Blocks per row.
    pub wide: u32,
    /// Block rows per slice.
    pub high: u32,
    /// Slices or layers.
    pub deep: u32,
    /// Bytes per block.
    pub block_bytes: u32,
}

impl LevelShape {
    /// Bytes of the whole level.
    pub(super) fn len(self) -> Result<usize, RenderError> {
        let v = u64::from(self.wide)
            .checked_mul(self.high.into())
            .and_then(|v| v.checked_mul(self.deep.into()))
            .and_then(|v| v.checked_mul(self.block_bytes.into()))
            .ok_or_else(overflow)?;
        to_usize(v)
    }
}

impl Rows {
    /// One row: `len` bytes from `offset`, which must end within `store_len`.
    pub(super) fn range(offset: u64, len: u64, store_len: usize) -> Result<Self, RenderError> {
        let rows = Self {
            first: to_usize(offset)?,
            row_bytes: to_usize(len)?,
            per_slice: 1,
            row_stride: 0,
            slices: 1,
            slice_stride: 0,
        };
        rows.check(store_len)?;
        Ok(rows)
    }

    /// The rows of `area` in a level shaped `shape`, which must lie within it.
    pub(super) fn blocks(shape: LevelShape, area: BlockBox) -> Result<Self, RenderError> {
        let bytes = u64::from(shape.block_bytes);
        let row_stride = u64::from(shape.wide)
            .checked_mul(bytes)
            .ok_or_else(overflow)?;
        let slice_stride = row_stride
            .checked_mul(shape.high.into())
            .ok_or_else(overflow)?;
        let first = u64::from(area.z)
            .checked_mul(slice_stride)
            .and_then(|v| v.checked_add(u64::from(area.y).checked_mul(row_stride)?))
            .and_then(|v| v.checked_add(u64::from(area.x).checked_mul(bytes)?))
            .ok_or_else(overflow)?;
        let inside = |start: u32, len: u32, total: u32| {
            u64::from(start) + u64::from(len) <= u64::from(total)
        };
        if !(inside(area.x, area.width, shape.wide)
            && inside(area.y, area.height, shape.high)
            && inside(area.z, area.depth, shape.deep))
        {
            return Err(RenderError::Invalid(
                "a recorded region lies outside its mip level",
            ));
        }
        let rows = Self {
            first: to_usize(first)?,
            row_bytes: to_usize(
                u64::from(area.width)
                    .checked_mul(bytes)
                    .ok_or_else(overflow)?,
            )?,
            per_slice: to_usize(area.height.into())?,
            row_stride: to_usize(row_stride)?,
            slices: to_usize(area.depth.into())?,
            slice_stride: to_usize(slice_stride)?,
        };
        rows.check(shape.len()?)?;
        Ok(rows)
    }

    /// Total bytes the rows cover.
    pub(super) fn total(&self) -> Result<usize, RenderError> {
        self.row_bytes
            .checked_mul(self.per_slice)
            .and_then(|v| v.checked_mul(self.slices))
            .ok_or_else(overflow)
    }

    /// Checks that the last byte of the last row lies within `store_len` (empty boxes always
    /// fit).
    fn check(&self, store_len: usize) -> Result<(), RenderError> {
        if self.row_bytes == 0 || self.per_slice == 0 || self.slices == 0 {
            return Ok(());
        }
        let end = (self.slices - 1)
            .checked_mul(self.slice_stride)
            .and_then(|v| v.checked_add((self.per_slice - 1).checked_mul(self.row_stride)?))
            .and_then(|v| v.checked_add(self.first))
            .and_then(|v| v.checked_add(self.row_bytes))
            .ok_or_else(overflow)?;
        if end <= store_len {
            Ok(())
        } else {
            Err(RenderError::Invalid(
                "a recorded region lies outside its store",
            ))
        }
    }

    /// The byte offset of each row, in packed order. Within the store by [`Rows::check`].
    fn starts(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.slices).flat_map(move |s| {
            (0..self.per_slice)
                .map(move |r| self.first + s * self.slice_stride + r * self.row_stride)
        })
    }
}

impl Store {
    /// An empty store of `len` bytes, all unknown.
    pub(super) const fn new(len: usize) -> Self {
        Self { len, data: None }
    }

    /// Length in bytes.
    pub(super) const fn len(&self) -> usize {
        self.len
    }

    fn data_mut(&mut self) -> &mut Data {
        let len = self.len;
        self.data.get_or_insert_with(|| Data {
            bytes: vec![0; len],
            known: vec![false; len],
        })
    }

    /// The bytes `rows` covers, packed.
    pub(super) fn gather(&self, rows: &Rows) -> Result<Packed, RenderError> {
        rows.check(self.len)?;
        let total = rows.total()?;
        let Some(data) = &self.data else {
            return Ok(Packed::Unknown { len: total });
        };
        let mut bytes = Vec::with_capacity(total);
        let mut known = Vec::with_capacity(total);
        for start in rows.starts() {
            let range = start..start + rows.row_bytes;
            let (Some(b), Some(k)) = (data.bytes.get(range.clone()), data.known.get(range)) else {
                return Err(RenderError::Invalid(
                    "a recorded region lies outside its store",
                ));
            };
            bytes.extend_from_slice(b);
            known.extend_from_slice(k);
        }
        Ok(Packed::Data { bytes, known })
    }

    /// Writes `packed` into the bytes `rows` covers; its length must be theirs.
    pub(super) fn scatter(&mut self, rows: &Rows, packed: &Packed) -> Result<(), RenderError> {
        rows.check(self.len)?;
        let total = rows.total()?;
        if packed.len() != total {
            return Err(RenderError::SizeMismatch {
                what: "recorded copy",
                expected: u64::try_from(total).unwrap_or(u64::MAX),
                actual: u64::try_from(packed.len()).unwrap_or(u64::MAX),
            });
        }
        match packed {
            Packed::Unknown { .. } => self.forget(rows),
            Packed::Data { bytes, known } => {
                let data = self.data_mut();
                for (i, start) in rows.starts().enumerate() {
                    let from = i * rows.row_bytes..(i + 1) * rows.row_bytes;
                    let to = start..start + rows.row_bytes;
                    let (Some(src_b), Some(src_k)) = (bytes.get(from.clone()), known.get(from))
                    else {
                        return Err(overflow());
                    };
                    let (Some(dst_b), Some(dst_k)) = (
                        data.bytes.get_mut(to.clone()),
                        data.known.get_mut(to.clone()),
                    ) else {
                        return Err(overflow());
                    };
                    dst_b.copy_from_slice(src_b);
                    dst_k.copy_from_slice(src_k);
                }
                Ok(())
            }
        }
    }

    /// Writes `texel` over and over into the bytes `rows` covers; each row must be whole texels.
    pub(super) fn fill(&mut self, rows: &Rows, texel: &[u8]) -> Result<(), RenderError> {
        rows.check(self.len)?;
        if texel.is_empty() || !rows.row_bytes.is_multiple_of(texel.len()) {
            return Err(RenderError::Invalid(
                "a recorded fill does not cover whole texels",
            ));
        }
        let data = self.data_mut();
        for start in rows.starts() {
            let to = start..start + rows.row_bytes;
            let (Some(dst_b), Some(dst_k)) = (
                data.bytes.get_mut(to.clone()),
                data.known.get_mut(to.clone()),
            ) else {
                return Err(overflow());
            };
            for chunk in dst_b.chunks_exact_mut(texel.len()) {
                chunk.copy_from_slice(texel);
            }
            dst_k.fill(true);
        }
        Ok(())
    }

    /// Marks the bytes `rows` covers unknown.
    pub(super) fn forget(&mut self, rows: &Rows) -> Result<(), RenderError> {
        rows.check(self.len)?;
        let Some(data) = &mut self.data else {
            return Ok(());
        };
        for start in rows.starts() {
            let Some(k) = data.known.get_mut(start..start + rows.row_bytes) else {
                return Err(overflow());
            };
            k.fill(false);
        }
        Ok(())
    }

    /// Whether any byte holds memory yet (for tests: a store nothing wrote stays empty).
    #[cfg(test)]
    pub(super) const fn is_allocated(&self) -> bool {
        self.data.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::{BlockBox, LevelShape, Packed, Rows, Store};
    use crate::error::RenderError;

    const SHAPE: LevelShape = LevelShape {
        wide: 4,
        high: 3,
        deep: 2,
        block_bytes: 2,
    };

    #[test]
    fn rows_of_a_block_box_in_packed_order() {
        let area = BlockBox {
            x: 1,
            y: 1,
            z: 1,
            width: 2,
            height: 2,
            depth: 1,
        };
        let rows = Rows::blocks(SHAPE, area).expect("inside");
        // Slice 1 starts at 4 * 3 * 2 = 24; row 1 at 24 + 8; column 1 at + 2.
        assert_eq!(rows.starts().collect::<Vec<_>>(), vec![34, 42]);
        assert_eq!(rows.total(), Ok(8));
    }

    #[test]
    fn a_box_at_the_edge_fits_and_one_past_does_not() {
        let edge = BlockBox {
            x: 2,
            y: 0,
            z: 1,
            width: 2,
            height: SHAPE.high,
            depth: 1,
        };
        assert!(Rows::blocks(SHAPE, edge).is_ok());
        let err = Err(RenderError::Invalid(
            "a recorded region lies outside its mip level",
        ));
        for past in [
            BlockBox { x: 3, ..edge },
            BlockBox { y: 1, ..edge },
            BlockBox { z: 2, ..edge },
            BlockBox {
                width: u32::MAX,
                ..edge
            },
        ] {
            assert_eq!(Rows::blocks(SHAPE, past).map(|_| ()), err, "{past:?}");
        }
    }

    #[test]
    fn a_range_must_end_within_the_store() {
        assert!(Rows::range(6, 4, 10).is_ok());
        assert_eq!(
            Rows::range(7, 4, 10).map(|_| ()),
            Err(RenderError::Invalid(
                "a recorded region lies outside its store"
            ))
        );
        // The supported hosts are 64-bit: the offset converts, and its end overflows.
        assert_eq!(
            Rows::range(u64::MAX, 1, 10).map(|_| ()),
            Err(RenderError::Overflow {
                what: "recorded contents offset"
            })
        );
    }

    #[test]
    fn untouched_store_gathers_unknown_without_memory() {
        let mut s = Store::new(16);
        let rows = Rows::range(0, 16, 16).expect("fits");
        assert_eq!(s.gather(&rows), Ok(Packed::Unknown { len: 16 }));
        s.forget(&rows).expect("fits");
        s.scatter(&rows, &Packed::Unknown { len: 16 })
            .expect("fits");
        assert!(!s.is_allocated(), "nothing known was written");
    }

    #[test]
    fn scatter_gather_and_forget() {
        let mut s = Store::new(8);
        let mid = Rows::range(2, 4, 8).expect("fits");
        s.scatter(&mid, &Packed::known(vec![1, 2, 3, 4]))
            .expect("fits");
        let all = Rows::range(0, 8, 8).expect("fits");
        assert_eq!(
            s.gather(&all),
            Ok(Packed::Data {
                bytes: vec![0, 0, 1, 2, 3, 4, 0, 0],
                known: vec![false, false, true, true, true, true, false, false],
            })
        );
        assert_eq!(
            s.gather(&mid).map(Packed::into_known),
            Ok(Some(vec![1, 2, 3, 4]))
        );
        s.forget(&Rows::range(3, 1, 8).expect("fits"))
            .expect("fits");
        assert_eq!(s.gather(&mid).map(Packed::into_known), Ok(None));
        // The bytes either side of the forgotten one are still known.
        assert_eq!(
            s.gather(&Rows::range(4, 2, 8).expect("fits"))
                .map(Packed::into_known),
            Ok(Some(vec![3, 4]))
        );
    }

    #[test]
    fn scatter_refuses_a_wrong_length() {
        let mut s = Store::new(8);
        let rows = Rows::range(0, 4, 8).expect("fits");
        assert_eq!(
            s.scatter(&rows, &Packed::known(vec![1; 3])),
            Err(RenderError::SizeMismatch {
                what: "recorded copy",
                expected: 4,
                actual: 3
            })
        );
    }

    #[test]
    fn fill_repeats_the_texel_and_needs_whole_texels() {
        let mut s = Store::new(SHAPE.len().expect("small"));
        let area = BlockBox {
            x: 1,
            y: 0,
            z: 0,
            width: 2,
            height: 1,
            depth: 1,
        };
        let rows = Rows::blocks(SHAPE, area).expect("inside");
        s.fill(&rows, &[0xab, 0xcd]).expect("whole texels");
        assert_eq!(
            s.gather(&Rows::range(0, 8, s.len()).expect("fits")),
            Ok(Packed::Data {
                bytes: vec![0, 0, 0xab, 0xcd, 0xab, 0xcd, 0, 0],
                known: vec![false, false, true, true, true, true, false, false],
            })
        );
        assert_eq!(
            s.fill(&rows, &[1, 2, 3]),
            Err(RenderError::Invalid(
                "a recorded fill does not cover whole texels"
            ))
        );
        assert_eq!(
            s.fill(&rows, &[]),
            Err(RenderError::Invalid(
                "a recorded fill does not cover whole texels"
            ))
        );
    }

    #[test]
    fn level_shape_length_overflows_as_an_error() {
        let huge = LevelShape {
            wide: u32::MAX,
            high: u32::MAX,
            deep: u32::MAX,
            block_bytes: 16,
        };
        assert_eq!(
            huge.len(),
            Err(RenderError::Overflow {
                what: "recorded contents offset"
            })
        );
    }
}
