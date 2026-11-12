//! Reading one file of a container: streaming, decompressing, and checking as it goes.
//!
//! A [`FileReader`] reads the file's extent through a [`Take`] of exactly `extent_size` bytes,
//! so it can never read another file's bytes, and delivers at most the file's `size` bytes. A
//! raw extent is the content (SPEC §4.1, codec 0). A zstd extent is one Zstandard frame that
//! must embed the content size (SPEC §4.1, codec 1); it is decoded by `ruzstd` with its default
//! window limit (`ruzstd::decoding::DEFAULT_MAX_WINDOW_SIZE`, 128 MiB), which bounds the
//! decoder's memory. The reference tool decompresses with `max_output_size=size` and then
//! compares the length (`read.py`, `NupReader.read_file`).
//!
//! When the last byte of the content has been delivered, the reader checks, in this order, that
//! the frame holds nothing more ([`NupError::ContentTooLong`]), that the extent holds nothing
//! after the frame ([`NupError::TrailingExtentBytes`]), and that the content matches the file's
//! SHA-256 ([`NupError::HashMismatch`]). A frame that ends cleanly before the file's size is
//! reached is [`NupError::ContentTruncated`]; a frame cut short inside a block, or bytes that are
//! not a frame, are whatever `ruzstd` reports, as [`NupError::Zstd`]. These errors reach a `Read` caller as an `io::Error` of kind
//! `InvalidData` that carries the [`NupError`]; [`NupError::from_io`] takes it back out.

use std::io::{self, Read, Seek, SeekFrom, Take};

use ruzstd::decoding::{FrameDecoder, StreamingDecoder};
use sha2::{Digest, Sha256};

use super::layout::SHA256_SIZE;
use super::{Codec, FileId, Nup, NupError};

/// Where a reader gets its stored bytes.
enum Body<'a, S: Read> {
    /// The extent is the content.
    Raw(Take<&'a mut S>),
    /// The extent is one zstd frame.
    Zstd(Box<StreamingDecoder<Take<&'a mut S>, FrameDecoder>>),
}

/// The content of one file, read from the container on demand. See the module docs.
pub struct FileReader<'a, S: Read> {
    file: usize,
    body: Body<'a, S>,
    remaining: u64,
    hasher: Sha256,
    expected: [u8; SHA256_SIZE],
    finished: bool,
}

impl<S: Read> std::fmt::Debug for FileReader<'_, S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FileReader")
            .field("file", &self.file)
            .field("remaining", &self.remaining)
            .field("finished", &self.finished)
            .finish_non_exhaustive()
    }
}

/// An `io::Error` carrying `error`.
fn invalid(error: NupError) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, error)
}

impl<S: Read + Seek> FileReader<'_, S> {
    /// The checks that run once the whole content has been delivered (module docs).
    fn finish(&mut self) -> Result<(), NupError> {
        self.finished = true;
        let file = self.file;
        let mut probe = [0u8; 1];
        let leftover = match &mut self.body {
            Body::Raw(take) => take.limit(),
            Body::Zstd(decoder) => {
                if decoder
                    .read(&mut probe)
                    .map_err(|error| zstd_error(file, &error))?
                    != 0
                {
                    return Err(NupError::ContentTooLong { file });
                }
                decoder.get_ref().limit()
            }
        };
        if leftover != 0 {
            return Err(NupError::TrailingExtentBytes { file });
        }
        let digest = std::mem::take(&mut self.hasher).finalize();
        if digest.as_slice() != self.expected.as_slice() {
            return Err(NupError::HashMismatch { file });
        }
        Ok(())
    }
}

/// A zstd decoding failure of `file` as a [`NupError`].
fn zstd_error(file: usize, error: &dyn std::fmt::Display) -> NupError {
    NupError::Zstd {
        file,
        message: error.to_string(),
    }
}

impl<S: Read + Seek> Read for FileReader<'_, S> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if self.finished || buf.is_empty() {
            return Ok(0);
        }
        if self.remaining == 0 {
            self.finish().map_err(invalid)?;
            return Ok(0);
        }
        let want = usize::try_from(self.remaining).map_or(buf.len(), |left| left.min(buf.len()));
        let window = buf.get_mut(..want).unwrap_or_default();
        let file = self.file;
        let count = match &mut self.body {
            Body::Raw(take) => take.read(window)?,
            Body::Zstd(decoder) => decoder
                .read(window)
                .map_err(|error| invalid(zstd_error(file, &error)))?,
        };
        if count == 0 {
            return Err(invalid(NupError::ContentTruncated { file }));
        }
        self.hasher.update(window.get(..count).unwrap_or_default());
        self.remaining = self.remaining.saturating_sub(super::to_u64(count));
        if self.remaining == 0 {
            self.finish().map_err(invalid)?;
        }
        Ok(count)
    }
}

impl<S: Read + Seek> Nup<S> {
    /// A reader for `file`'s content. See the module docs for what it checks.
    pub fn reader(&mut self, file: FileId) -> Result<FileReader<'_, S>, NupError> {
        let index = file.index();
        let entry = self.files.get(index).ok_or(NupError::NoSuchFile(index))?;
        let (size, codec, expected) = (entry.size, entry.codec, entry.sha256);
        let (offset, extent_size) = (entry.extent_offset, entry.extent_size);
        self.source.seek(SeekFrom::Start(offset))?;
        let take = (&mut self.source).take(extent_size);
        let body = match codec {
            Codec::Raw => Body::Raw(take),
            Codec::Zstd => {
                let decoder =
                    StreamingDecoder::new(take).map_err(|error| zstd_error(index, &error))?;
                let frame = decoder.decoder.content_size();
                if frame != size {
                    return Err(NupError::ZstdContentSize {
                        file: index,
                        size,
                        frame,
                    });
                }
                Body::Zstd(Box::new(decoder))
            }
        };
        Ok(FileReader {
            file: index,
            body,
            remaining: size,
            hasher: Sha256::new(),
            expected,
            finished: false,
        })
    }

    /// `file`'s whole content, checked, if it is at most `limit` bytes.
    pub fn read_to_vec(&mut self, file: FileId, limit: u64) -> Result<Vec<u8>, NupError> {
        let index = file.index();
        let size = self
            .files
            .get(index)
            .ok_or(NupError::NoSuchFile(index))?
            .size;
        if size > limit {
            return Err(NupError::FileTooLarge {
                file: index,
                size,
                limit,
            });
        }
        // `size <= limit`, which the caller chose; the allocation is bounded by it.
        let mut out = Vec::with_capacity(usize::try_from(size).unwrap_or(0));
        self.reader(file)?
            .read_to_end(&mut out)
            .map_err(NupError::from_io)?;
        Ok(out)
    }

    /// Reads `file` to its end, checking its size and SHA-256, and keeps nothing.
    pub fn verify_file(&mut self, file: FileId) -> Result<(), NupError> {
        // `io::copy` reads until a read returns 0, and the reader runs its end checks before it
        // first returns 0 (for an empty file, on the first read).
        io::copy(&mut self.reader(file)?, &mut io::sink()).map_err(NupError::from_io)?;
        Ok(())
    }
}
