//! Reading one file of a container: streaming, decompressing, and checking as it goes.
//!
//! A [`FileReader`] reads the file's extent through a [`Take`] of exactly `extent_size` bytes,
//! so it can never read another file's bytes, and delivers at most the file's `size` bytes. A
//! raw extent is the content (SPEC §4.1, codec 0). A zstd extent is one Zstandard frame that
//! must embed the content size (SPEC §4.1, codec 1); it is decoded by `ruzstd` with its default
//! window limit (`ruzstd::decoding::DEFAULT_MAX_WINDOW_SIZE`, 128 MiB), which bounds the
//! decoder's memory (a frame whose window is over it is [`NupError::Zstd`]; pinned by
//! `a_window_over_the_decoders_limit_is_a_decode_error_and_one_at_it_is_not`). The reference tool decompresses with `max_output_size=size` and then
//! compares the length (`read.py`, `NupReader.read_file`).
//!
//! When the last byte of the content has been delivered, the reader checks, in this order, that
//! the frame holds nothing more ([`NupError::ContentTooLong`]), that the extent holds nothing
//! after the frame ([`NupError::TrailingExtentBytes`]), and that the content matches the file's
//! SHA-256 ([`NupError::HashMismatch`]). A frame that ends cleanly before the file's size is
//! reached is [`NupError::ContentTruncated`]; a frame cut short inside a block, or bytes that are
//! not a frame, are whatever `ruzstd` reports, as [`NupError::Zstd`]. These errors reach a `Read`
//! caller as an `io::Error` of kind `InvalidData` that carries the [`NupError`];
//! [`NupError::from_io`] takes it back out. A failure is sticky: every later read of the same
//! reader returns the same error, never a clean end of file. (An `io::Error` from the source
//! itself is passed on as it comes, and a later read tries the source again.)
//!
//! A zstd frame's optional content checksum is read as part of the frame, so its four bytes must
//! be inside the extent, but `ruzstd` does not compare it with the content; the file's SHA-256
//! is the integrity check. The reference tool decompresses with the `zstandard` module, which
//! does compare it (a divergence, looser here; pinned by
//! `a_frame_with_a_content_checksum_ends_with_its_four_bytes`).

use std::io::{self, Read, Seek, SeekFrom, Take};

use crate::read_at::{Positional, ReadAt, read_exact_at};

use ruzstd::decoding::{FrameDecoder, StreamingDecoder};
use sha2::{Digest, Sha256};

use super::layout::SHA256_SIZE;
use super::{Codec, FileEntry, FileId, Nup, NupError};

/// A zstd decoding failure of `file` as a [`NupError`].
fn zstd_error(file: usize, error: &dyn std::fmt::Display) -> NupError {
    NupError::Zstd {
        file,
        message: error.to_string(),
    }
}

/// Where a reader gets its stored bytes: `R` is the container's source, placed at the extent.
enum Body<R: Read> {
    /// The extent is the content.
    Raw(Take<R>),
    /// The extent is one zstd frame.
    Zstd(Box<StreamingDecoder<Take<R>, FrameDecoder>>),
}

/// Why a reader stopped for good: a [`NupError`] that is kept so that every later read can
/// return it again (`NupError` holds an `io::Error` in one variant, so it is not `Clone`; these
/// are the variants a reader itself produces, none of which does).
#[derive(Debug, Clone, PartialEq, Eq)]
enum Failure {
    /// [`NupError::ContentTruncated`].
    ContentTruncated,
    /// [`NupError::ContentTooLong`].
    ContentTooLong,
    /// [`NupError::TrailingExtentBytes`].
    TrailingExtentBytes,
    /// [`NupError::HashMismatch`].
    HashMismatch,
    /// [`NupError::Zstd`], with the decoder's message.
    Zstd(String),
}

impl Failure {
    /// The error for `file`.
    fn error(&self, file: usize) -> NupError {
        match self {
            Self::ContentTruncated => NupError::ContentTruncated { file },
            Self::ContentTooLong => NupError::ContentTooLong { file },
            Self::TrailingExtentBytes => NupError::TrailingExtentBytes { file },
            Self::HashMismatch => NupError::HashMismatch { file },
            Self::Zstd(message) => zstd_error(file, message),
        }
    }

    /// A decoder failure.
    fn zstd(error: &dyn std::fmt::Display) -> Self {
        Self::Zstd(error.to_string())
    }
}

/// Where a reader is in its file.
#[derive(Debug, Clone, PartialEq, Eq)]
enum State {
    /// Content is still to come.
    Reading,
    /// All of it was delivered and the end checks passed.
    Done,
    /// A read failed; every later read fails the same way.
    Failed(Failure),
}

/// The content of one file, read from the container on demand. See the module docs. `R` is how
/// it reaches the container's source: a mutable borrow ([`Nup::reader`]) or a cursor of its own
/// over a shared source ([`Nup::reader_at`]).
pub struct FileReader<R: Read> {
    file: usize,
    body: Body<R>,
    remaining: u64,
    hasher: Sha256,
    expected: [u8; SHA256_SIZE],
    state: State,
}

impl<R: Read> std::fmt::Debug for FileReader<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FileReader")
            .field("file", &self.file)
            .field("remaining", &self.remaining)
            .field("state", &self.state)
            .finish_non_exhaustive()
    }
}

/// An `io::Error` carrying `error`.
fn invalid(error: NupError) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, error)
}

impl<R: Read> FileReader<R> {
    /// The checks that run once the whole content has been delivered (module docs).
    fn finish(&mut self) -> Result<(), Failure> {
        let mut probe = [0u8; 1];
        let leftover = match &mut self.body {
            Body::Raw(take) => take.limit(),
            Body::Zstd(decoder) => {
                if decoder
                    .read(&mut probe)
                    .map_err(|error| Failure::zstd(&error))?
                    != 0
                {
                    return Err(Failure::ContentTooLong);
                }
                decoder.get_ref().limit()
            }
        };
        if leftover != 0 {
            return Err(Failure::TrailingExtentBytes);
        }
        let digest = std::mem::take(&mut self.hasher).finalize();
        if digest.as_slice() != self.expected.as_slice() {
            return Err(Failure::HashMismatch);
        }
        Ok(())
    }

    /// Runs the end checks and records their outcome.
    fn conclude(&mut self) -> io::Result<()> {
        match self.finish() {
            Ok(()) => {
                self.state = State::Done;
                Ok(())
            }
            Err(failure) => Err(self.fail(failure)),
        }
    }

    /// Records `failure` for every later read and returns it as this read's error.
    fn fail(&mut self, failure: Failure) -> io::Error {
        let error = invalid(failure.error(self.file));
        self.state = State::Failed(failure);
        error
    }
}

impl<R: Read> Read for FileReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        match &self.state {
            State::Failed(failure) => return Err(invalid(failure.error(self.file))),
            State::Done => return Ok(0),
            State::Reading => {}
        }
        if buf.is_empty() {
            return Ok(0);
        }
        if self.remaining == 0 {
            // An empty file: the end checks run on its first read.
            self.conclude()?;
            return Ok(0);
        }
        let want = usize::try_from(self.remaining).map_or(buf.len(), |left| left.min(buf.len()));
        let window = buf.get_mut(..want).unwrap_or_default();
        let count = match &mut self.body {
            Body::Raw(take) => take.read(window)?,
            Body::Zstd(decoder) => match decoder.read(window) {
                Ok(count) => count,
                Err(error) => return Err(self.fail(Failure::zstd(&error))),
            },
        };
        if count == 0 {
            return Err(self.fail(Failure::ContentTruncated));
        }
        self.hasher.update(window.get(..count).unwrap_or_default());
        self.remaining = self.remaining.saturating_sub(super::to_u64(count));
        if self.remaining == 0 {
            self.conclude()?;
        }
        Ok(count)
    }
}

/// A reader for the file at `index` with `entry`'s record, whose extent `source` is placed at
/// the start of: the checks every reader makes before its first read.
fn start<R: Read>(index: usize, entry: &FileEntry, source: R) -> Result<FileReader<R>, NupError> {
    let take = source.take(entry.extent_size);
    let body = match entry.codec {
        Codec::Raw => Body::Raw(take),
        Codec::Zstd => {
            let decoder = StreamingDecoder::new(take).map_err(|error| zstd_error(index, &error))?;
            let frame = decoder.decoder.content_size();
            if frame != entry.size {
                return Err(NupError::ZstdContentSize {
                    file: index,
                    size: entry.size,
                    frame,
                });
            }
            Body::Zstd(Box::new(decoder))
        }
    };
    Ok(FileReader {
        file: index,
        body,
        remaining: entry.size,
        hasher: Sha256::new(),
        expected: entry.sha256,
        state: State::Reading,
    })
}

/// `reader`'s whole content, `size` bytes, checked: the allocation is `size`, which the caller
/// has bounded.
fn collect<R: Read>(mut reader: FileReader<R>, size: u64) -> Result<Vec<u8>, NupError> {
    let mut out = Vec::with_capacity(usize::try_from(size).unwrap_or(0));
    reader.read_to_end(&mut out).map_err(NupError::from_io)?;
    Ok(out)
}

impl<S> Nup<S> {
    /// `file`'s record and index, or [`NupError::NoSuchFile`].
    fn entry(&self, file: FileId) -> Result<(usize, &FileEntry), NupError> {
        let index = file.index();
        let entry = self.files.get(index).ok_or(NupError::NoSuchFile(index))?;
        Ok((index, entry))
    }

    /// `file`'s size, if it is at most `limit` bytes ([`NupError::FileTooLarge`] otherwise).
    fn size_within(&self, file: FileId, limit: u64) -> Result<u64, NupError> {
        let (index, entry) = self.entry(file)?;
        if entry.size > limit {
            return Err(NupError::FileTooLarge {
                file: index,
                size: entry.size,
                limit,
            });
        }
        Ok(entry.size)
    }
}

impl<S: Read + Seek> Nup<S> {
    /// A reader for `file`'s content. See the module docs for what it checks.
    pub fn reader(&mut self, file: FileId) -> Result<FileReader<&mut S>, NupError> {
        let index = file.index();
        let entry = self.files.get(index).ok_or(NupError::NoSuchFile(index))?;
        self.source.seek(SeekFrom::Start(entry.extent_offset))?;
        start(index, entry, &mut self.source)
    }

    /// `file`'s whole content, checked, if it is at most `limit` bytes.
    pub fn read_to_vec(&mut self, file: FileId, limit: u64) -> Result<Vec<u8>, NupError> {
        let size = self.size_within(file, limit)?;
        // `size <= limit`, which the caller chose; the allocation is bounded by it.
        collect(self.reader(file)?, size)
    }

    /// Reads `file` to its end, checking its size and SHA-256, and keeps nothing.
    pub fn verify_file(&mut self, file: FileId) -> Result<(), NupError> {
        // `io::copy` reads until a read returns 0, and the reader runs its end checks before it
        // first returns 0 (for an empty file, on the first read).
        io::copy(&mut self.reader(file)?, &mut io::sink()).map_err(NupError::from_io)?;
        Ok(())
    }
}

/// Positional reads, through `&self`: several of them may run at once on one container, each
/// at its own offset (`crate::read_at`).
impl<S: ReadAt> Nup<S> {
    /// A reader for `file`'s content with a cursor of its own over the shared source; what it
    /// checks is what [`Nup::reader`]'s does (module docs).
    pub fn reader_at(&self, file: FileId) -> Result<FileReader<Positional<&S>>, NupError> {
        let (index, entry) = self.entry(file)?;
        let mut cursor = Positional::new(&self.source);
        cursor.seek(SeekFrom::Start(entry.extent_offset))?;
        start(index, entry, cursor)
    }

    /// `file`'s whole content, checked as [`Nup::read_to_vec`] checks it, if it is at most
    /// `limit` bytes; read through [`Nup::reader_at`]. The size is compared with `limit` before
    /// anything is allocated or read.
    pub fn read_to_vec_at(&self, file: FileId, limit: u64) -> Result<Vec<u8>, NupError> {
        let size = self.size_within(file, limit)?;
        collect(self.reader_at(file)?, size)
    }

    /// Up to `buf.len()` bytes of the raw file `file`, from `offset` bytes into it, straight from
    /// its extent: the count read, fewer than `buf.len()` only at the end of the file and zero
    /// at or past it. A zstd file has no random access and is [`NupError::NotRaw`].
    ///
    /// The bytes are not checked against the file's SHA-256: a hash covers the whole content,
    /// and a read of part of it cannot check it. `Nup::verify` (and `starquake verify`) does. A
    /// source shorter than the extent [`Nup::open`] checked is [`NupError::Io`].
    pub fn read_raw_at(
        &self,
        file: FileId,
        offset: u64,
        buf: &mut [u8],
    ) -> Result<usize, NupError> {
        let (index, entry) = self.entry(file)?;
        if entry.codec != Codec::Raw {
            return Err(NupError::NotRaw { file: index });
        }
        // A raw extent is exactly the file's size (`Nup::open` checks it).
        let left = entry.size.saturating_sub(offset);
        let count = usize::try_from(left).map_or(buf.len(), |left| left.min(buf.len()));
        let Some(window) = buf.get_mut(..count).filter(|window| !window.is_empty()) else {
            return Ok(0);
        };
        // `offset < size` here, and the extent lies inside the file (`Nup::open`), so this does
        // not overflow; were it to, it is an error, not a read somewhere else.
        let at = entry.extent_offset.checked_add(offset).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "the offset passes the extent")
        })?;
        read_exact_at(&self.source, at, window)?;
        Ok(count)
    }
}
