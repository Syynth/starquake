// Parts of this crate are ported from Mesa's `nil` (`mesa-nouveau/src/nouveau/nil/`, at commit
// 52238762), which carries this licence:
//
// Copyright © 2024 Collabora, Ltd.
// Copyright © 2024 Valve Corp. and Collabora, Ltd. (copy.rs)
// SPDX-License-Identifier: MIT
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software
// and associated documentation files (the "Software"), to deal in the Software without
// restriction, including without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all copies or
// substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
// BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
// DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

//! Block-linear image layout and tiling for the console GPU.
//!
//! Pure arithmetic over an image description (format, extent, levels, layers, sample layout,
//! GOB kind, block size): where each level and layer starts, how each level is tiled, where the
//! mip tail is, how large the image is, and the conversion between linear and tiled bytes for a
//! region. It works on byte slices the caller has already copied out of guest memory, depends on
//! no other workspace crate, and has no `unsafe`. Every input may be guest-controlled, so
//! nothing here panics: a malformed description or region is a [`LayoutError`].
//!
//! Ported from Mesa's `nil` crate (`mesa-nouveau/src/nouveau/nil/{tiling,image,extent,format,
//! copy}.rs` @ 52238762, MIT; the licence is at the top of this file), as the GPU protocol spec
//! lays out (`docs/specs/gpu-protocol.md`, "Block-linear layout: `starquake-layout`", slice G5).
//! Each module lists where it differs from Mesa.
//!
//! - [`tiling`]: GOB kinds ([`GobType`]), tilings ([`Tiling`]) and the byte offset of a position
//!   in a block-linear level ([`block_linear_offset`]). Its docs give the derivation that the
//!   Switch 2 colour GOB is Mesa's Turing GOB.
//! - [`image`]: an image's layout ([`Image`], from an [`ImageDesc`]).
//! - [`copy`]: tiling and untiling a region ([`linear_to_tiled`], [`tiled_to_linear`]).
//! - [`extent`] and [`format`]: unit-tagged extents and offsets, and what the layout needs of a
//!   format.

pub mod copy;
pub mod error;
pub mod extent;
pub mod format;
pub mod image;
pub mod tiling;

pub use copy::{LinearLayout, linear_to_tiled, tiled_to_linear};
pub use error::{LayoutError, Result};
pub use extent::units::{Bytes, Pixels};
pub use extent::{Extent4D, Offset4D};
pub use format::Format;
pub use image::{
    Image, ImageDesc, ImageDim, ImageLevel, LINEAR_DEFAULT_ALIGN_B, LINEAR_PITCH_ALIGN_B,
    MAX_LEVELS, MIN_IMAGE_ALIGN_B, SampleLayout,
};
pub use tiling::{
    GOB_HEIGHT, GOB_LINE_WIDTH_B, GOB_SECTOR_B, GOB_SIZE_B, GOB_WIDTH_B, GobType, Tiling,
    block_linear_offset,
};
