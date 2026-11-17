//! [`FspHost`]: what a run's `fsp-srv` asks the host for, on the environment's host I/O channel
//! (`starquake_services::fsp::host`): the SD card, a [`HostFolder`], and the title's `RomFS`
//! ([`RomFs`], from the narrow title API) two ways, as the data storage (the image) and as the
//! data file system (the files in it), told apart by each request's mount.
//!
//! The data storage is one file, named by the empty path:
//!
//! - A probe answers that it is there when the run has a title `RomFS`, and
//!   [`HostFailure::NoDevice`] when it has none.
//! - A size request answers the image's size; a read the bytes from its offset on, at most the
//!   length asked for, fewer at the end of the image and none past it. The bytes held are at
//!   most what the image has from the offset on, however large the length is.
//! - A read the image fails (the title file could not be read, or a file in it is damaged or
//!   too large for the decompression cache) is [`HostFailure::Io`], logged in plain words.
//! - A request with a path is [`HostFailure::NotFound`]: the storage has no files of its own.
//!
//! The data file system is the files of the same image, found by their paths through the
//! loader's `RomFS` reader (`starquake_loader::romfs`, a port of Eden's `romfs.cpp`, which
//! `FSP_SRV::OpenDataFileSystemByCurrentProcess` extracts the same way): its tables are read
//! from the image at the first request for the data file system and kept by this host.
//!
//! - A probe answers that it is there when the run has a title `RomFS` whose tables read, and
//!   [`HostFailure::NoDevice`] when it has none. Tables that do not read (the image is damaged,
//!   or the title file failed) are [`HostFailure::Io`], logged in plain words once, and every
//!   later request for the data file system gets the same: the failure is kept.
//! - A size request answers the file's size; a read the file's bytes from its offset on, at most
//!   the length asked for, fewer at the end of the file and none past it.
//! - A path that names no file (nothing there, a directory, a path that is not UTF-8 or not
//!   normalised) is [`HostFailure::NotFound`]; an entry that the tables cannot follow, or whose
//!   data lies outside the image, or a read of the image that fails, is [`HostFailure::Io`].
//!
//! The reads of the title file happen here, behind the environment, as the SD card's reads of
//! host files do, so `fsp-srv` itself never touches the host and a recorded run keeps every
//! answer.

use std::io;

use starquake_loader::read_at::{Positional, ReadAt};
use starquake_loader::romfs::{FileEntry, RomFs as Tables, RomFsError};
use starquake_services::fsp::host::{HostFailure, HostRequest, Mount};
use starquake_trace::{HostIoHandler, HostIoReply, HostIoRequest};

use super::host_folder::{HostFolder, serve};
use crate::title::RomFs;

/// The title's `RomFS` image as a positional source, for the loader's `RomFS` reader.
#[derive(Debug)]
struct Image(RomFs);

impl ReadAt for Image {
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> io::Result<usize> {
        self.0.read_at(offset, buf)
    }

    fn size(&self) -> io::Result<u64> {
        Ok(self.0.size())
    }
}

/// The data file system: the image's tables, read by the loader.
type Files = Tables<Positional<Image>>;

/// The host behind `fsp-srv` in a run: an SD card and a `RomFS`, each optional.
#[derive(Default)]
pub struct FspHost {
    sd_card: Option<HostFolder>,
    romfs: Option<RomFs>,
    /// The data file system's tables, read at its first request; `Err` once they did not read
    /// (module docs).
    files: Option<Result<Files, ()>>,
}

impl std::fmt::Debug for FspHost {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FspHost")
            .field("sd_card", &self.sd_card)
            .field("romfs", &self.romfs)
            .finish_non_exhaustive()
    }
}

/// A copy serves the same SD card and `RomFS`, and reads the tables again when it needs them.
impl Clone for FspHost {
    fn clone(&self) -> Self {
        Self::new(self.sd_card.clone(), self.romfs.clone())
    }
}

/// Two hosts are equal when they serve the same SD card and the same `RomFS`.
impl PartialEq for FspHost {
    fn eq(&self, other: &Self) -> bool {
        self.sd_card == other.sd_card && self.romfs == other.romfs
    }
}

impl Eq for FspHost {}

impl FspHost {
    /// A host with the SD card `sd_card`, and the data storage and data file system `romfs`.
    #[must_use]
    pub fn new(sd_card: Option<HostFolder>, romfs: Option<RomFs>) -> Self {
        Self {
            sd_card,
            romfs,
            files: None,
        }
    }

    /// Whether it has neither: a run then has no host I/O handler at all.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.sd_card.is_none() && self.romfs.is_none()
    }

    /// The answer to `request`, by its mount.
    fn answer(&mut self, request: &HostRequest) -> Result<Vec<u8>, HostFailure> {
        match request.mount() {
            Mount::SdCard => self
                .sd_card
                .as_ref()
                .map_or(Err(HostFailure::NoDevice), |card| card.answer(request)),
            Mount::DataStorage => {
                let romfs = self.romfs.as_ref().ok_or(HostFailure::NoDevice)?;
                answer_storage(romfs, request)
            }
            Mount::DataFileSystem => answer_files(self.files()?, request),
        }
    }

    /// The data file system's tables, read from the image the first time (module docs).
    fn files(&mut self) -> Result<&Files, HostFailure> {
        let romfs = self.romfs.as_ref().ok_or(HostFailure::NoDevice)?;
        let files = self.files.get_or_insert_with(|| {
            Tables::new(Positional::new(Image(romfs.clone()))).map_err(|error| {
                tracing::error!(
                    %error,
                    "the title's RomFS tables could not be read: it has no data file system"
                );
            })
        });
        files.as_ref().map_err(|()| HostFailure::Io)
    }
}

/// The data file system's answer to `request` (module docs).
fn answer_files(files: &Files, request: &HostRequest) -> Result<Vec<u8>, HostFailure> {
    match request {
        HostRequest::Probe { .. } => Ok(Vec::new()),
        HostRequest::Size { path, .. } => Ok(file(files, path)?.size.to_le_bytes().to_vec()),
        HostRequest::Read {
            path, offset, len, ..
        } => {
            let entry = file(files, path)?;
            // The data must lie in the image before a byte is held for it.
            files
                .file_extent(&entry)
                .map_err(|error| io_failure(&error))?;
            let there = entry.size.saturating_sub(*offset).min(*len);
            let count = usize::try_from(there).map_err(|_| HostFailure::Io)?;
            let mut bytes = vec![0u8; count];
            let read = files
                .read_file_shared(&entry, *offset, &mut bytes)
                .map_err(|error| io_failure(&error))?;
            bytes.truncate(read);
            Ok(bytes)
        }
    }
}

/// The file at `path` in the data file system (module docs).
fn file(files: &Files, path: &[u8]) -> Result<FileEntry, HostFailure> {
    let path = std::str::from_utf8(path).map_err(|_| HostFailure::NotFound)?;
    match files.find_file(path) {
        Ok(Some(entry)) => Ok(entry),
        Ok(None) | Err(RomFsError::InvalidPath { .. } | RomFsError::PathTooLong) => {
            Err(HostFailure::NotFound)
        }
        Err(error) => Err(io_failure(&error)),
    }
}

/// A `RomFS` the data file system cannot follow, logged.
fn io_failure(error: &RomFsError) -> HostFailure {
    tracing::error!(%error, "a file of the title's RomFS could not be read");
    HostFailure::Io
}

/// The data storage's answer to `request` (module docs).
fn answer_storage(romfs: &RomFs, request: &HostRequest) -> Result<Vec<u8>, HostFailure> {
    match request {
        HostRequest::Probe { .. } => Ok(Vec::new()),
        HostRequest::Size { path, .. } if path.is_empty() => {
            Ok(romfs.size().to_le_bytes().to_vec())
        }
        HostRequest::Read {
            path, offset, len, ..
        } if path.is_empty() => read(romfs, *offset, *len),
        HostRequest::Size { .. } | HostRequest::Read { .. } => Err(HostFailure::NotFound),
    }
}

/// Up to `len` bytes of the image from `offset`.
fn read(romfs: &RomFs, offset: u64, len: u64) -> Result<Vec<u8>, HostFailure> {
    let there = romfs.size().saturating_sub(offset).min(len);
    let count = usize::try_from(there).map_err(|_| HostFailure::Io)?;
    let mut bytes = vec![0u8; count];
    let mut done = 0usize;
    while let Some(rest) = bytes.get_mut(done..).filter(|rest| !rest.is_empty()) {
        let at = u64::try_from(done)
            .ok()
            .and_then(|done| offset.checked_add(done))
            .ok_or(HostFailure::Io)?;
        match romfs.read_at(at, rest) {
            Ok(0) => break,
            Ok(read) => done = done.saturating_add(read),
            Err(error) => {
                tracing::error!(%error, offset = at, "the title's RomFS could not be read");
                return Err(HostFailure::Io);
            }
        }
    }
    bytes.truncate(done);
    Ok(bytes)
}

impl HostIoHandler for FspHost {
    fn handle(&mut self, request: &HostIoRequest) -> HostIoReply {
        serve(request, |request| self.answer(request))
    }
}

#[cfg(test)]
pub(crate) mod tests;
