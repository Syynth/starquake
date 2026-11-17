//! Positional reads: a source that several readers share without a mutable borrow.
//!
//! A [`ReadAt`] source reads at an offset through `&self`, so two open files of one container,
//! or a reader on each of two threads, can read it at once, each at its own offset
//! (decision log, 2026-11-16, "RomFS from a NUP: positional reads and a capped decompression
//! cache"). [`Positional`] gives such a source a cursor, so that code written for `Read + Seek`
//! (`Nup::open`, `RomFs::new`) runs on it too, while the source itself stays shareable.
//!
//! What implements it: byte vectors and slices (in memory), `std::fs::File` on Unix hosts
//! (`std::os::unix::fs::FileExt::read_at`, which does not move the file's own cursor), and
//! shared or borrowed sources (`Arc<T>`, `&T`).

use std::io::{self, Read, Seek, SeekFrom};
use std::sync::Arc;

/// A source read at explicit offsets through a shared reference.
pub trait ReadAt: Send + Sync {
    /// Reads up to `buf.len()` bytes from `offset` into `buf` and returns how many: zero only
    /// when `buf` is empty or `offset` is at or past the end. Fewer than `buf.len()` is not an
    /// error; [`read_exact_at`] loops until the buffer is full.
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> io::Result<usize>;

    /// The source's length in bytes.
    fn size(&self) -> io::Result<u64>;
}

/// Fills `buf` from `offset` of `source`. A source that ends first is an
/// [`io::ErrorKind::UnexpectedEof`] error; an interrupted read is tried again.
pub fn read_exact_at<S: ReadAt + ?Sized>(
    source: &S,
    mut offset: u64,
    mut buf: &mut [u8],
) -> io::Result<()> {
    while !buf.is_empty() {
        match source.read_at(offset, buf) {
            Ok(0) => {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "the source ended before the bytes asked for",
                ));
            }
            Ok(count) => {
                let advance = u64::try_from(count).map_err(|_| overflow())?;
                offset = offset.checked_add(advance).ok_or_else(overflow)?;
                buf = buf.get_mut(count..).unwrap_or_default();
            }
            Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
            Err(error) => return Err(error),
        }
    }
    Ok(())
}

/// The error for an offset past `u64::MAX`.
fn overflow() -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidInput,
        "the offset passes the largest a source can have",
    )
}

/// Reads from `bytes` at `offset`, as every in-memory source does.
fn read_slice(bytes: &[u8], offset: u64, buf: &mut [u8]) -> usize {
    let tail = usize::try_from(offset)
        .ok()
        .and_then(|start| bytes.get(start..))
        .unwrap_or_default();
    let count = tail.len().min(buf.len());
    if let (Some(to), Some(from)) = (buf.get_mut(..count), tail.get(..count)) {
        to.copy_from_slice(from);
    }
    count
}

/// The length of an in-memory source.
fn slice_size(bytes: &[u8]) -> u64 {
    u64::try_from(bytes.len()).unwrap_or(u64::MAX)
}

impl ReadAt for [u8] {
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> io::Result<usize> {
        Ok(read_slice(self, offset, buf))
    }

    fn size(&self) -> io::Result<u64> {
        Ok(slice_size(self))
    }
}

impl ReadAt for Vec<u8> {
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> io::Result<usize> {
        Ok(read_slice(self, offset, buf))
    }

    fn size(&self) -> io::Result<u64> {
        Ok(slice_size(self))
    }
}

impl<T: ReadAt + ?Sized> ReadAt for &T {
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> io::Result<usize> {
        (**self).read_at(offset, buf)
    }

    fn size(&self) -> io::Result<u64> {
        (**self).size()
    }
}

impl<T: ReadAt + ?Sized> ReadAt for Arc<T> {
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> io::Result<usize> {
        (**self).read_at(offset, buf)
    }

    fn size(&self) -> io::Result<u64> {
        (**self).size()
    }
}

#[cfg(unix)]
impl ReadAt for std::fs::File {
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> io::Result<usize> {
        std::os::unix::fs::FileExt::read_at(self, buf, offset)
    }

    fn size(&self) -> io::Result<u64> {
        Ok(self.metadata()?.len())
    }
}

/// A [`ReadAt`] source with a cursor of its own, so that it is also `Read + Seek`. Its
/// [`ReadAt`] reads ignore and keep the cursor.
#[derive(Debug, Clone)]
pub struct Positional<S> {
    source: S,
    position: u64,
}

impl<S> Positional<S> {
    /// `source`, with its cursor at the start.
    pub fn new(source: S) -> Self {
        Self {
            source,
            position: 0,
        }
    }

    /// The source.
    pub fn get_ref(&self) -> &S {
        &self.source
    }

    /// The source, without the cursor.
    pub fn into_inner(self) -> S {
        self.source
    }
}

impl<S: ReadAt> Read for Positional<S> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let count = self.source.read_at(self.position, buf)?;
        let advance = u64::try_from(count).map_err(|_| overflow())?;
        self.position = self.position.checked_add(advance).ok_or_else(overflow)?;
        Ok(count)
    }
}

impl<S: ReadAt> Seek for Positional<S> {
    fn seek(&mut self, to: SeekFrom) -> io::Result<u64> {
        let (base, delta) = match to {
            SeekFrom::Start(offset) => {
                self.position = offset;
                return Ok(offset);
            }
            SeekFrom::End(delta) => (self.source.size()?, delta),
            SeekFrom::Current(delta) => (self.position, delta),
        };
        let position = base.checked_add_signed(delta).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "a seek before the start or past the largest offset",
            )
        })?;
        self.position = position;
        Ok(position)
    }
}

impl<S: ReadAt> ReadAt for Positional<S> {
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> io::Result<usize> {
        self.source.read_at(offset, buf)
    }

    fn size(&self) -> io::Result<u64> {
        self.source.size()
    }
}

#[cfg(test)]
mod tests;
