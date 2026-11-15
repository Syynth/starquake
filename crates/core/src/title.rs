//! A NUP title as a program to run: the narrow title API that `starquake run <title.nup>` uses,
//! so a frontend opens a title through this crate instead of driving the loader itself.
//!
//! [`System::from_nup`] opens a NUP container from any `Read + Seek` source, chooses the title's
//! ExeFS, reads `main.npdm` and the modules from it, and loads them as [`System::from_exefs`]
//! does: the process its `main.npdm` describes, with the module set placed in the loader's
//! `starquake_loader::exefs::MODULE_NAMES` order (`rtld`, `main`, `subsdk0` to `subsdk9`,
//! `sdk`). [`is_nup`] tells a frontend whether a file's first bytes are a NUP container's.
//!
//! # Which ExeFS
//!
//! The ExeFS is the one `starquake inspect` shows, because both take it from the loader's
//! `Nup::title`: the newest update's ExeFS as a whole when that update has one, the base's
//! otherwise (the loader's `nup::title` module docs, which cite Eden's
//! `PatchManager::PatchExeFS`, and list the guess in "newest"). A title with no ExeFS in either
//! is [`TitleError::NoExeFs`], as Eden's `AppLoader_NCA::Load` returns `ErrorNoExeFS` when
//! neither the program nor its update has one (`eden/src/core/loader/nca.cpp` @ cf654052, path
//! relative to the reference folder).
//!
//! # What is read
//!
//! Only the container's table of contents (`Nup::open`), `main.npdm` and the ExeFS files named
//! in `MODULE_NAMES` are read from the source. Other ExeFS files and the RomFS are not, so a
//! title of tens of gigabytes is never loaded whole. Each file is read whole and checked
//! against its size and SHA-256 by the loader's reader. The reads are bounded, so a corrupt
//! size cannot make a run allocate more: at most [`NPDM_READ_LIMIT`] bytes of `main.npdm`, and
//! at most [`EXEFS_READ_LIMIT`] bytes for `main.npdm` and the modules together. Both are
//! starquake limits, not the format's. A file past its limit is [`TitleError::Read`] with the
//! loader's `NupError::FileTooLarge`. For a module, that limit is what was left of the
//! [`EXEFS_READ_LIMIT`] budget after the files read before it; `main.npdm` is the exception,
//! read first with the smaller of [`NPDM_READ_LIMIT`] and the whole budget as its limit.
//!
//! The RomFS is not served yet: a title that opens its RomFS gets the file-system service's
//! current answer.
//!
//! # Order of checks
//!
//! The first failure is the error:
//!
//! 1. The container opens (`Nup::open`) and has a title (`Nup::title`: a program package, and
//!    no two updates sharing the newest version): [`TitleError::Container`].
//! 2. The title has an ExeFS: [`TitleError::NoExeFs`].
//! 3. The ExeFS has a `main.npdm`: [`TitleError::ExeFs`] with `ExeFsError::MissingNpdm`, before
//!    any file is read.
//! 4. `main.npdm`, then each module present in `MODULE_NAMES` order, is read within its limit:
//!    [`TitleError::Read`].
//! 5. The loader parses the files (`ExeFs::parse`, whose docs give its own order: the metadata,
//!    `main` missing, then each module): [`TitleError::ExeFs`].
//! 6. The program loads as [`System::from_exefs`] loads it: [`TitleError::Load`].

use std::io::{Read, Seek};

use starquake_loader::exefs::{ExeFs, ExeFsError, MODULE_NAMES, NPDM_FILE};
use starquake_loader::nup::layout::MAGIC;
use starquake_loader::nup::{FileId, Nup, NupError};

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
    let mut nup = Nup::open(source).map_err(TitleError::Container)?;
    let title = nup.title().map_err(TitleError::Container)?;
    if title.exefs.is_none() {
        return Err(TitleError::NoExeFs);
    }
    let Some(npdm) = title.exefs_file(&nup, NPDM_FILE) else {
        return Err(ExeFsError::MissingNpdm.into());
    };
    let mut wanted: Vec<(&'static str, FileId)> = vec![(NPDM_FILE, npdm)];
    wanted.extend(
        MODULE_NAMES
            .into_iter()
            .filter_map(|name| Some((name, title.exefs_file(&nup, name)?))),
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
