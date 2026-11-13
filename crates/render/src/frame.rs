//! Frame dumps: presented frames handed to a [`FrameSink`] as RGBA8, and their PNG encoding
//! (docs/specs/render-backend.md, "Frame dumps and headless runs"; slice R9, first part).
//!
//! A present whose `dump` flag is set ([`crate::command::PresentDesc::dump`]) is read back at
//! the guest's own resolution, and the backend hands it to the sink the app installed. The
//! frame is the presented rectangle as it is stored, before the present's transform: the
//! transform travels with the frame for whoever wants the picture the right way up.
//!
//! This crate does no file I/O: [`Frame::png`] and [`encode_png`] return the file's bytes, and
//! the app writes them where the observability spec says (`frames/`); that wiring is its own
//! slice.
//!
//! # Pixels a backend does not know
//!
//! A sink is never given wrong pixels. Where a backend cannot say what a frame holds, the
//! frame's pixels are [`FramePixels::Unknown`], and it has no PNG. The recording backend knows
//! a frame when every byte of the presented rectangle is known (an upload or an exact clear
//! put it there; a draw makes its targets unknown) and the view's format has an exact RGBA8
//! form: the four 8-bit formats of [`rgba8`], which are what the console presents. Any other
//! format would have to be converted through the present blit's arithmetic, which the
//! recording backend does not model.
//!
//! # The PNG
//!
//! Eight bits a channel, RGBA, no interlacing and no chunks but `IHDR`, `IDAT` and `IEND`:
//! the bytes of an sRGB format and of a linear one are written alike, as stored. It is
//! compressed with the `png` crate's fast setting, so the exact bytes of a file depend on the
//! version of that crate and of its deflate implementation, which `Cargo.lock` pins;
//! `frame::tests::golden_bytes` shows when they move. Compare frames by decoding them.

use crate::command::PresentTransform;
use crate::error::RenderError;
use crate::format::PixelFormat;
use std::fmt::Debug;

/// Bytes in one RGBA8 pixel.
pub const RGBA8_BYTES: usize = 4;

/// What a presented frame holds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FramePixels {
    /// The frame's pixels, row after row from the top, four bytes each: red, green, blue,
    /// alpha.
    Rgba8(Vec<u8>),
    /// The backend cannot say what the frame holds (module docs).
    Unknown,
}

/// A presented frame, as a [`FrameSink`] receives it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Frame {
    /// The frame number from the present.
    pub frame: u64,
    /// Width of the presented rectangle, in pixels.
    pub width: u32,
    /// Height of the presented rectangle, in pixels.
    pub height: u32,
    /// The transform the present asked for; it is not applied to the pixels.
    pub transform: PresentTransform,
    /// The pixels.
    pub pixels: FramePixels,
}

impl Frame {
    /// The frame as a PNG file's bytes, or `None` when its pixels are unknown.
    ///
    /// Errors: as [`encode_png`], for a frame whose pixels are not its size.
    pub fn png(&self) -> Result<Option<Vec<u8>>, RenderError> {
        match &self.pixels {
            FramePixels::Rgba8(rgba) => encode_png(self.width, self.height, rgba).map(Some),
            FramePixels::Unknown => Ok(None),
        }
    }
}

/// Receives the frames presented with the `dump` flag set, in the order they were presented.
///
/// The app installs one on its backend. A backend calls it from the thread that presents (for
/// the recording backend, the thread that submits the list, after the list has completed and
/// with none of the backend's locks held).
pub trait FrameSink: Debug + Send + Sync {
    /// Takes one presented frame.
    fn frame(&self, frame: Frame);
}

/// The RGBA8 form of `bytes`, tightly packed texels of `format`, for the formats whose RGBA8
/// form is exact: `A8B8G8R8_UNORM` and `A8B8G8R8_SRGB` (already red, green, blue, alpha in
/// memory) and `B8G8R8A8_UNORM` and `B8G8R8A8_SRGB` (red and blue exchanged). `None` for every
/// other format.
pub fn rgba8(format: PixelFormat, mut bytes: Vec<u8>) -> Option<Vec<u8>> {
    match format {
        PixelFormat::A8B8G8R8Unorm | PixelFormat::A8B8G8R8Srgb => Some(bytes),
        PixelFormat::B8G8R8A8Unorm | PixelFormat::B8G8R8A8Srgb => {
            for pixel in bytes.as_chunks_mut::<RGBA8_BYTES>().0 {
                pixel.swap(BLUE_IN_BGRA, RED_IN_BGRA);
            }
            Some(bytes)
        }
        _ => None,
    }
}

/// Where blue is in a `B8G8R8A8` texel, which is where red is in RGBA8.
const BLUE_IN_BGRA: usize = 0;
/// Where red is in a `B8G8R8A8` texel, which is where blue is in RGBA8.
const RED_IN_BGRA: usize = 2;

/// Encodes `rgba`, `width` by `height` RGBA8 pixels row after row, as a PNG file's bytes
/// (module docs).
///
/// Errors, in this order: an empty image; a size whose byte count does not fit in 64 bits;
/// `rgba` not exactly `width * height * 4` bytes.
pub fn encode_png(width: u32, height: u32, rgba: &[u8]) -> Result<Vec<u8>, RenderError> {
    if width == 0 || height == 0 {
        return Err(RenderError::Invalid("a frame must not be empty"));
    }
    let expected = u64::from(width)
        .checked_mul(height.into())
        .and_then(|pixels| pixels.checked_mul(RGBA8_BYTES_U64))
        .ok_or(RenderError::Overflow { what: "frame size" })?;
    // A length that does not fit in 64 bits is no frame's size.
    let actual = u64::try_from(rgba.len()).unwrap_or(u64::MAX);
    if actual != expected {
        return Err(RenderError::SizeMismatch {
            what: "frame pixels",
            expected,
            actual,
        });
    }
    let mut file = Vec::new();
    let mut encoder = png::Encoder::new(&mut file, width, height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.set_compression(png::Compression::Fast);
    // Writing to a `Vec` cannot fail and the size was checked above, so no error is expected
    // here; whatever the encoder still refuses is reported, not unwrapped.
    encoder
        .write_header()
        .and_then(|mut writer| {
            writer.write_image_data(rgba)?;
            writer.finish()
        })
        .map_err(|_| RenderError::Invalid("the PNG encoder refused the frame"))?;
    Ok(file)
}

/// [`RGBA8_BYTES`] for 64-bit size arithmetic.
const RGBA8_BYTES_U64: u64 = RGBA8_BYTES as u64;

#[cfg(test)]
mod tests;
