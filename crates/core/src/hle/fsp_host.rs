//! [`FspHost`]: what a run's `fsp-srv` asks the host for, on the environment's host I/O channel
//! (`starquake_services::fsp::host`): the SD card, a [`HostFolder`], and the title's data storage,
//! its `RomFS` served as an image ([`RomFs`], from the narrow title API), told apart by each
//! request's mount.
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
//! The reads of the title file happen here, behind the environment, as the SD card's reads of
//! host files do, so `fsp-srv` itself never touches the host and a recorded run keeps every
//! answer.

use starquake_services::fsp::host::{HostFailure, HostRequest, Mount};
use starquake_trace::{HostIoHandler, HostIoReply, HostIoRequest};

use super::host_folder::{HostFolder, serve};
use crate::title::RomFs;

/// The host behind `fsp-srv` in a run: an SD card and a data storage, each optional.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct FspHost {
    sd_card: Option<HostFolder>,
    romfs: Option<RomFs>,
}

impl FspHost {
    /// A host with the SD card `sd_card` and the data storage `romfs`.
    #[must_use]
    pub fn new(sd_card: Option<HostFolder>, romfs: Option<RomFs>) -> Self {
        Self { sd_card, romfs }
    }

    /// Whether it has neither: a run then has no host I/O handler at all.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.sd_card.is_none() && self.romfs.is_none()
    }

    /// The answer to `request`, by its mount.
    fn answer(&self, request: &HostRequest) -> Result<Vec<u8>, HostFailure> {
        match request.mount() {
            Mount::SdCard => self
                .sd_card
                .as_ref()
                .map_or(Err(HostFailure::NoDevice), |card| card.answer(request)),
            Mount::DataStorage => {
                let romfs = self.romfs.as_ref().ok_or(HostFailure::NoDevice)?;
                answer_storage(romfs, request)
            }
        }
    }
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
