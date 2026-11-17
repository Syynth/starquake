//! A NUP title as a program to run: the narrow title API that `starquake run <title.nup>` uses,
//! so a frontend opens a title through this crate instead of driving the loader itself.
//!
//! [`open`] opens a NUP container from a positional source (`starquake_loader::read_at::ReadAt`:
//! a host file, or bytes in memory), chooses the package the title runs, reads `main.npdm` and
//! the modules from its ExeFS, and loads them as [`System::from_exefs`] does: the process its
//! `main.npdm` describes, with the module set placed in the loader's
//! `starquake_loader::exefs::MODULE_NAMES` order (`rtld`, `main`, `subsdk0` to `subsdk9`,
//! `sdk`). It also makes that same package's RomFS the run's data storage ([`RomFs`], which a
//! run is given in `RunOptions::romfs`), served to `fsp-srv`'s `OpenDataStorageByCurrentProcess`
//! through the environment (`crate::hle::FspHost`). [`System::from_nup`] loads the program alone,
//! from any `Read + Seek` source. [`is_nup`] tells a frontend whether a file's first bytes are a
//! NUP container's.
//!
//! # Which package
//!
//! The loader's `Nup::newest_package` (decision log, 2026-11-16, "M2 runs a title's newest
//! package, whole"): the newest update when the container has one, the base otherwise, with that
//! one package's ExeFS and RomFS; base and update are never mixed. A newest update that lacks a
//! part its base has (an ExeFS, or a RomFS) is a patch that needs its base, and is
//! [`TitleError::Container`] with the loader's `NupError::IncompleteUpdate`, in plain words. The
//! loader's `nup::title` module docs list the guesses this rests on ("newest", and what makes an
//! update not complete). `starquake inspect` shows the loader's `Nup::title`, which agrees except
//! for an update with no ExeFS: inspect shows the base's, and a run refuses it.
//!
//! A package with no ExeFS is [`TitleError::NoExeFs`], as Eden's `AppLoader_NCA::Load` returns
//! `ErrorNoExeFS` when neither the program nor its update has one
//! (`eden/src/core/loader/nca.cpp` @ cf654052, path relative to the reference folder). A package
//! with no RomFS runs with no data storage: `OpenDataStorageByCurrentProcess` then fails as
//! Eden's does (`starquake_services::fsp`, "No data storage").
//!
//! # What is read
//!
//! Only the container's table of contents (`Nup::open`), `main.npdm` and the ExeFS files named
//! in `MODULE_NAMES` are read from the source while loading. Other ExeFS files and the RomFS are
//! not, so a title of tens of gigabytes is never loaded whole: the RomFS's image is laid out from
//! the table of contents, and its files are read when the guest reads them (the loader's
//! `nup::storage`, which keeps compressed files decompressed in a cache capped at
//! `starquake_loader::nup::storage::CACHE_CAP`). Each file is read whole and checked
//! against its size and SHA-256 by the loader's reader. The reads are bounded, so a corrupt
//! size cannot make a run allocate more: at most [`NPDM_READ_LIMIT`] bytes of `main.npdm`, and
//! at most [`EXEFS_READ_LIMIT`] bytes for `main.npdm` and the modules together. Both are
//! starquake limits, not the format's. A file past its limit is [`TitleError::Read`] with the
//! loader's `NupError::FileTooLarge`. For a module, that limit is what was left of the
//! [`EXEFS_READ_LIMIT`] budget after the files read before it; `main.npdm` is the exception,
//! read first with the smaller of [`NPDM_READ_LIMIT`] and the whole budget as its limit.
//!
//! # Order of checks
//!
//! The first failure is the error:
//!
//! 1. The container opens (`Nup::open`) and has a package to run (`Nup::newest_package`: a
//!    program package, no two updates sharing the newest version, and a newest update that
//!    needs nothing of its base): [`TitleError::Container`].
//! 2. The package has an ExeFS: [`TitleError::NoExeFs`].
//! 3. The ExeFS has a `main.npdm`: [`TitleError::ExeFs`] with `ExeFsError::MissingNpdm`, before
//!    any file is read.
//! 4. `main.npdm`, then each module present in `MODULE_NAMES` order, is read within its limit:
//!    [`TitleError::Read`].
//! 5. The loader parses the files (`ExeFs::parse`, whose docs give its own order: the metadata,
//!    `main` missing, then each module): [`TitleError::ExeFs`].
//! 6. The program loads as [`System::from_exefs`] loads it: [`TitleError::Load`].
//! 7. For [`open`], the package's RomFS, if it has one, is laid out as an image:
//!    [`TitleError::RomFs`].

use std::io::{self, Read, Seek};
use std::sync::Arc;

use starquake_loader::exefs::{ExeFs, ExeFsError, MODULE_NAMES, NPDM_FILE};
use starquake_loader::nup::layout::MAGIC;
use starquake_loader::nup::storage::{CACHE_CAP, RomFsStorage, StorageError};
use starquake_loader::nup::{DirId, FileId, Node, Nup, NupError, Whole};
use starquake_loader::read_at::{Positional, ReadAt};

use crate::system::{Error, System};

/// The most bytes of `main.npdm` a title's run reads: 1 MiB. A starquake limit; an NPDM is a few
/// kilobytes.
pub const NPDM_READ_LIMIT: u64 = 1 << 20;

/// The most bytes a title's run reads from its ExeFS, `main.npdm` and the modules together:
/// 1 GiB. A starquake limit, so a corrupt size cannot make a run allocate more.
pub const EXEFS_READ_LIMIT: u64 = 1 << 30;

/// How many leading bytes of a file [`is_nup`] needs: the length of the NUP magic.
pub const PREFIX_LEN: usize = MAGIC.len();

/// Whether `prefix`, a file's first bytes, starts with the NUP signature `NUP` (SPEC §2, rule 1,
/// the bytes before the format's major version in the loader's `nup::layout::MAGIC`). The
/// version byte is not compared, so a NUP file of a version the loader does not read is still
/// recognised, and opening it says which version it is. Reading [`PREFIX_LEN`] bytes is
/// enough; fewer than three bytes is never a NUP file.
pub fn is_nup(prefix: &[u8]) -> bool {
    let [n, u, p, _major] = MAGIC;
    prefix.starts_with(&[n, u, p])
}

/// Why a title could not be loaded as a program.
#[derive(Debug, thiserror::Error)]
pub enum TitleError {
    /// The container could not be opened, or holds no title (no program package, or two
    /// updates sharing the newest version).
    #[error("the title could not be opened: {0}")]
    Container(NupError),
    /// Neither the newest update nor the base has an ExeFS.
    #[error("the title has no ExeFS")]
    NoExeFs,
    /// An ExeFS file could not be read: the source failed, the file is damaged, or it is past
    /// its read limit.
    #[error("the ExeFS file {name} could not be read: {error}")]
    Read {
        /// The file's name.
        name: &'static str,
        /// Why.
        error: NupError,
    },
    /// The ExeFS's files are not a program the loader takes.
    #[error(transparent)]
    ExeFs(#[from] ExeFsError),
    /// The program could not be loaded.
    #[error(transparent)]
    Load(#[from] Error),
    /// The RomFS could not be laid out as the image the title reads.
    #[error("the title's RomFS could not be served: {0}")]
    RomFs(#[from] StorageError),
}

/// A title's RomFS, served as the `RomFS` image its file-system library reads: the run's data
/// storage. Cloning it shares the one image, its source and its cache; two are equal when they
/// are the same one.
#[derive(Clone)]
pub struct RomFs {
    image: Arc<dyn ReadAt>,
    size: u64,
}

impl RomFs {
    /// The `RomFS` that `storage` serves.
    pub fn new<S: ReadAt + 'static>(storage: RomFsStorage<S>) -> Self {
        let size = storage.size();
        Self {
            image: Arc::new(storage),
            size,
        }
    }

    /// The image's size in bytes.
    #[must_use]
    pub fn size(&self) -> u64 {
        self.size
    }

    /// Reads up to `buf.len()` bytes of the image from `offset`: fewer only at its end, none at
    /// or past it. A failure carries the loader's `StorageError` in plain words.
    pub fn read_at(&self, offset: u64, buf: &mut [u8]) -> io::Result<usize> {
        self.image.read_at(offset, buf)
    }
}

impl std::fmt::Debug for RomFs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RomFs")
            .field("size", &self.size)
            .finish_non_exhaustive()
    }
}

impl PartialEq for RomFs {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.image, &other.image)
    }
}

impl Eq for RomFs {}

/// A NUP title opened to run ([`open`]): its program, and its RomFS when it has one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Title {
    /// The program the title's package holds.
    pub system: System,
    /// The same package's RomFS: the run's data storage (`RunOptions::romfs`).
    pub romfs: Option<RomFs>,
}

/// Opens the NUP title in `source` to run: its program, loaded as [`System::from_nup`] loads it,
/// and its RomFS, served from `source` as the guest reads it. See the module docs for what is
/// read and in which order it is checked.
///
/// # Errors
///
/// A [`TitleError`] saying which check failed.
pub fn open<S: ReadAt + 'static>(source: S) -> Result<Title, TitleError> {
    open_with(source, READ_LIMITS, CACHE_CAP)
}

/// [`open`] with `limits` and a decompression cache of `cache_cap` bytes.
fn open_with<S: ReadAt + 'static>(
    source: S,
    limits: ReadLimits,
    cache_cap: u64,
) -> Result<Title, TitleError> {
    let (mut nup, whole) = container(Positional::new(source))?;
    let system = load_program(&mut nup, whole.exefs, limits)?;
    let romfs = match whole.romfs {
        Some(dir) => Some(RomFs::new(RomFsStorage::new(nup, dir, cache_cap)?)),
        None => None,
    };
    Ok(Title { system, romfs })
}

/// The read limits of a title's run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ReadLimits {
    /// The most bytes of `main.npdm`.
    npdm: u64,
    /// The most bytes of `main.npdm` and the modules together.
    exefs: u64,
}

/// The limits [`System::from_nup`] reads with.
const READ_LIMITS: ReadLimits = ReadLimits {
    npdm: NPDM_READ_LIMIT,
    exefs: EXEFS_READ_LIMIT,
};

impl System {
    /// Loads the program a NUP title holds, read from `source` (a file, or bytes in memory):
    /// the ExeFS of its newest update or its base, loaded as [`System::from_exefs`] loads one.
    /// The title module's docs say what is read and in which order it is checked. Works on
    /// every host.
    ///
    /// # Errors
    ///
    /// A [`TitleError`] saying which check failed.
    pub fn from_nup<S: Read + Seek>(source: S) -> Result<Self, TitleError> {
        load(source, READ_LIMITS)
    }
}

/// [`System::from_nup`] with `limits`.
fn load<S: Read + Seek>(source: S, limits: ReadLimits) -> Result<System, TitleError> {
    let (mut nup, whole) = container(source)?;
    load_program(&mut nup, whole.exefs, limits)
}

/// The container in `source`, opened, and the package it runs (module docs, check 1).
fn container<S: Read + Seek>(source: S) -> Result<(Nup<S>, Whole), TitleError> {
    let nup = Nup::open(source).map_err(TitleError::Container)?;
    let whole = nup.newest_package().map_err(TitleError::Container)?;
    Ok((nup, whole))
}

/// The file of the ExeFS `exefs` called `name`.
fn exefs_file<S>(nup: &Nup<S>, exefs: DirId, name: &str) -> Option<FileId> {
    match nup.lookup(exefs, name.as_bytes())? {
        Node::File(file) => Some(file),
        Node::Dir(_) => None,
    }
}

/// Loads the program in the ExeFS `exefs` of `nup` (module docs, checks 2 to 6).
fn load_program<S: Read + Seek>(
    nup: &mut Nup<S>,
    exefs: Option<DirId>,
    limits: ReadLimits,
) -> Result<System, TitleError> {
    let exefs = exefs.ok_or(TitleError::NoExeFs)?;
    let Some(npdm) = exefs_file(nup, exefs, NPDM_FILE) else {
        return Err(ExeFsError::MissingNpdm.into());
    };
    let mut wanted: Vec<(&'static str, FileId)> = vec![(NPDM_FILE, npdm)];
    wanted.extend(
        MODULE_NAMES
            .into_iter()
            .filter_map(|name| Some((name, exefs_file(nup, exefs, name)?))),
    );

    let mut budget = limits.exefs;
    let mut files = Vec::with_capacity(wanted.len());
    for (name, file) in wanted {
        let limit = if name == NPDM_FILE {
            budget.min(limits.npdm)
        } else {
            budget
        };
        let bytes = nup
            .read_to_vec(file, limit)
            .map_err(|error| TitleError::Read { name, error })?;
        // `read_to_vec` returned at most `limit <= budget` bytes.
        budget = budget.saturating_sub(u64::try_from(bytes.len()).unwrap_or(u64::MAX));
        files.push((name, bytes));
    }
    let borrowed: Vec<(&str, &[u8])> = files
        .iter()
        .map(|(name, bytes)| (*name, bytes.as_slice()))
        .collect();
    let exefs = ExeFs::parse(&borrowed)?;
    Ok(System::from_exefs(exefs)?)
}

#[cfg(test)]
mod tests;
