//! [`HostFolder`]: a host folder as the SD card `fsp-srv` opens, answering its host file requests
//! (`starquake_services::fsp::host`) on the environment's host I/O channel.
//!
//! Read-only: the protocol has no request that writes. A path in a request is one the service
//! has normalised (`starquake_services::fsp::path`), but it comes from the guest, and a replayed
//! log could hold anything, so it is checked again here before it touches the host: it must be
//! UTF-8, start with `/`, and every component must be a plain name (not empty, not `.` or `..`,
//! no backslash). What it names, once symbolic links are followed, must still be inside the
//! folder. A path that fails any of these is [`HostFailure::NotFound`].

use std::fs::{self, File};
use std::io::{self, ErrorKind, Read, Seek, SeekFrom};
use std::path::PathBuf;

use starquake_services::fsp::host::{CHANNEL, HostFailure, HostRequest, Mount};
use starquake_services::fsp::path::{ALTERNATE_DIRECTORY_SEPARATOR, DIRECTORY_SEPARATOR};
use starquake_trace::{HostIoHandler, HostIoReply, HostIoRequest};

/// A host folder serving as the SD card.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostFolder {
    root: PathBuf,
}

impl HostFolder {
    /// The folder at `root`. It need not exist yet: a probe answers [`HostFailure::NoDevice`]
    /// while it does not.
    #[must_use]
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// The host path a request's path names inside the folder, if it is a plain one that stays
    /// inside it (see the module docs).
    fn resolve(&self, path: &[u8]) -> Result<PathBuf, HostFailure> {
        let text = std::str::from_utf8(path).map_err(|_| HostFailure::NotFound)?;
        let separator = char::from(DIRECTORY_SEPARATOR);
        let rest = text.strip_prefix(separator).ok_or(HostFailure::NotFound)?;
        let mut resolved = self.root.clone();
        if !rest.is_empty() {
            for component in rest.split(separator) {
                let plain = !component.is_empty()
                    && component != "."
                    && component != ".."
                    && !component.contains(char::from(ALTERNATE_DIRECTORY_SEPARATOR))
                    && !component.contains('\0');
                if !plain {
                    return Err(HostFailure::NotFound);
                }
                resolved.push(component);
            }
        }
        let root = fs::canonicalize(&self.root).map_err(|_| HostFailure::NoDevice)?;
        let real = fs::canonicalize(&resolved).map_err(|error| failure(&error))?;
        if !real.starts_with(&root) {
            return Err(HostFailure::NotFound);
        }
        Ok(real)
    }

    fn probe(&self) -> Result<Vec<u8>, HostFailure> {
        if self.root.is_dir() {
            Ok(Vec::new())
        } else {
            Err(HostFailure::NoDevice)
        }
    }

    /// The file at `path`, open, and its size.
    fn file(&self, path: &[u8]) -> Result<(File, u64), HostFailure> {
        let path = self.resolve(path)?;
        let metadata = fs::metadata(&path).map_err(|error| failure(&error))?;
        if !metadata.is_file() {
            return Err(HostFailure::NotAFile);
        }
        let file = File::open(&path).map_err(|error| failure(&error))?;
        Ok((file, metadata.len()))
    }

    fn size(&self, path: &[u8]) -> Result<Vec<u8>, HostFailure> {
        let (_, size) = self.file(path)?;
        Ok(size.to_le_bytes().to_vec())
    }

    /// Up to `len` bytes from `offset`. The bytes held are at most what the file has from the
    /// offset on, however large `len` is.
    fn read(&self, path: &[u8], offset: u64, len: u64) -> Result<Vec<u8>, HostFailure> {
        let (mut file, size) = self.file(path)?;
        if offset >= size {
            return Ok(Vec::new());
        }
        file.seek(SeekFrom::Start(offset))
            .map_err(|error| failure(&error))?;
        let mut bytes = Vec::new();
        file.take(len)
            .read_to_end(&mut bytes)
            .map_err(|error| failure(&error))?;
        Ok(bytes)
    }

    fn answer(&self, request: &HostRequest) -> Result<Vec<u8>, HostFailure> {
        match request {
            HostRequest::Probe {
                mount: Mount::SdCard,
            } => self.probe(),
            HostRequest::Size {
                mount: Mount::SdCard,
                path,
            } => self.size(path),
            HostRequest::Read {
                mount: Mount::SdCard,
                path,
                offset,
                len,
            } => self.read(path, *offset, *len),
        }
    }
}

/// The failure a host error stands for.
fn failure(error: &io::Error) -> HostFailure {
    match error.kind() {
        ErrorKind::NotFound | ErrorKind::NotADirectory => HostFailure::NotFound,
        ErrorKind::IsADirectory => HostFailure::NotAFile,
        _ => HostFailure::Io,
    }
}

impl HostIoHandler for HostFolder {
    fn handle(&mut self, request: &HostIoRequest) -> HostIoReply {
        if request.channel != CHANNEL {
            return HostIoReply::Unavailable;
        }
        let answer = HostRequest::decode(&request.payload)
            .map_err(|error| {
                tracing::error!(%error, "a host file request could not be decoded");
                HostFailure::BadRequest
            })
            .and_then(|request| self.answer(&request));
        match answer {
            Ok(bytes) => HostIoReply::Data(bytes),
            Err(failure) => failure.reply(),
        }
    }
}

#[cfg(test)]
mod tests;
