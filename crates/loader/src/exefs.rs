//! An ExeFS: a program's `main.npdm` and its modules, by file name. [`ExeFs::parse`] takes the
//! files (from a folder, a PFS0 or a title container; reading them is the caller's) and returns
//! the parsed metadata and each module present as a [`CodeSet`], in the order the modules are
//! placed in the process.
//!
//! Ported from `AppLoader_DeconstructedRomDirectory::Load`
//! (`eden/src/core/loader/deconstructed_rom_directory.cpp` @ cf654052, GPL-3.0-or-later, path
//! relative to the reference folder): the `main.npdm` file name (a missing one is
//! `ErrorMissingNPDM`), and the modules' names and order, `static_modules = {"rtld", "main",
//! "subsdk0", ..., "subsdk9", "sdk"}` ([`MODULE_NAMES`]). A module that is not there is skipped;
//! each one that is is placed directly after the one before.
//!
//! # Divergences
//!
//! - **`main` is required** ([`ExeFsError::MissingMain`]). Eden loads a set without one; Atmosphère's
//!   loader refuses it (`R_UNLESS_LOG(ctx.ali.has_main, ldr::ResultInvalidNso(), "[ldr] Missing
//!   main!\n")` in `CheckAutoLoad`, `atmosphere/stratosphere/loader/source/ldr_process_creation.cpp`
//!   @ 4e3f3703, fact only). Test: `a_set_without_main_is_refused`.
//! - **A file name given twice** is [`ExeFsError::DuplicateFile`]; a folder or a PFS0 cannot hold
//!   two, so neither reference has the case. Test: `a_file_given_twice_is_refused`.
//! - Atmosphère's other set rules (`sdk` needs `rtld`; the browser DLLs `wkc0`-`wkc9`, which
//!   only its list has) are not applied: Eden applies none of them, and no title run here has
//!   needed them.
//!
//! Files with other names (`control.nacp`, icons) are ignored, as Eden ignores them here.

use crate::codeset::{CodeSet, CodeSetError};
use crate::npdm::{Npdm, NpdmError};
use crate::nso::{Nso, NsoError};

/// The metadata's file name.
pub const NPDM_FILE: &str = "main.npdm";

/// The program's own module, which every set must have.
pub const MAIN_MODULE: &str = "main";

/// The modules an ExeFS may hold, in the order they are placed (Eden's `static_modules`).
pub const MODULE_NAMES: [&str; 13] = [
    "rtld",
    MAIN_MODULE,
    "subsdk0",
    "subsdk1",
    "subsdk2",
    "subsdk3",
    "subsdk4",
    "subsdk5",
    "subsdk6",
    "subsdk7",
    "subsdk8",
    "subsdk9",
    "sdk",
];

/// One module of the set.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExeFsModule {
    /// Its file name, one of [`MODULE_NAMES`].
    pub name: &'static str,
    /// What it puts in its pages.
    pub code: CodeSet,
}

/// A parsed ExeFS.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExeFs {
    /// The program's metadata.
    pub npdm: Npdm,
    /// The modules present, in [`MODULE_NAMES`] order.
    pub modules: Vec<ExeFsModule>,
}

/// Why a module of the set could not be loaded.
#[derive(Debug, thiserror::Error)]
pub enum ModuleError {
    /// The file is not a well-formed NSO.
    #[error(transparent)]
    Nso(#[from] NsoError),
    /// The NSO cannot be loaded as a module.
    #[error(transparent)]
    CodeSet(#[from] CodeSetError),
}

/// Why an ExeFS could not be parsed. (Not `Clone` or `PartialEq`: an NSO's LZ4 error is
/// neither.)
#[derive(Debug, thiserror::Error)]
pub enum ExeFsError {
    /// There is no `main.npdm`.
    #[error("the ExeFS has no {NPDM_FILE}")]
    MissingNpdm,
    /// `main.npdm` is malformed.
    #[error("the ExeFS's {NPDM_FILE} is malformed: {0}")]
    Npdm(#[from] NpdmError),
    /// There is no `main` module.
    #[error("the ExeFS has no {MAIN_MODULE} module")]
    MissingMain,
    /// A module is malformed.
    #[error("the ExeFS's {name} module is malformed: {error}")]
    Module {
        /// Its file name.
        name: &'static str,
        /// What is wrong with it.
        error: ModuleError,
    },
    /// The same file name was given more than once.
    #[error("the ExeFS file {0} was given more than once")]
    DuplicateFile(String),
}

impl ExeFs {
    /// Parses the files of an ExeFS, given as `(name, bytes)` (see the module docs). The checks
    /// come in this order: a name given twice, the metadata (missing, then malformed), `main`
    /// missing, then each module in [`MODULE_NAMES`] order.
    ///
    /// # Errors
    ///
    /// An [`ExeFsError`] saying which file and why.
    pub fn parse<'a>(files: &[(&'a str, &'a [u8])]) -> Result<Self, ExeFsError> {
        let mut seen = std::collections::BTreeSet::new();
        for (name, _) in files {
            if !seen.insert(*name) {
                return Err(ExeFsError::DuplicateFile((*name).to_owned()));
            }
        }
        let file = |wanted: &str| {
            files
                .iter()
                .find(|(name, _)| *name == wanted)
                .map(|(_, bytes)| *bytes)
        };
        let npdm = Npdm::parse(file(NPDM_FILE).ok_or(ExeFsError::MissingNpdm)?)?;
        if file(MAIN_MODULE).is_none() {
            return Err(ExeFsError::MissingMain);
        }
        let mut modules = Vec::new();
        for name in MODULE_NAMES {
            let Some(bytes) = file(name) else {
                continue;
            };
            let code = Nso::parse(bytes)
                .map_err(ModuleError::from)
                .and_then(|nso| Ok(CodeSet::from_nso(nso)?))
                .map_err(|error| ExeFsError::Module { name, error })?;
            modules.push(ExeFsModule { name, code });
        }
        Ok(Self { npdm, modules })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const NSO: &[u8] = include_bytes!("../../../tests/homebrew/hello/hello.nso");
    const NPDM: &[u8] = include_bytes!("../../../tests/homebrew/hello/main.npdm");

    fn names(exefs: &ExeFs) -> Vec<&'static str> {
        exefs.modules.iter().map(|module| module.name).collect()
    }

    #[test]
    fn modules_come_in_edens_order_whatever_order_the_files_are_given_in() {
        let files = [
            ("sdk", NSO),
            ("control.nacp", &b"ignored"[..]),
            (MAIN_MODULE, NSO),
            (NPDM_FILE, NPDM),
            ("subsdk9", NSO),
            ("rtld", NSO),
            ("subsdk0", NSO),
        ];
        let exefs = ExeFs::parse(&files).unwrap();
        assert_eq!(
            names(&exefs),
            ["rtld", MAIN_MODULE, "subsdk0", "subsdk9", "sdk"]
        );
        assert_eq!(exefs.npdm, Npdm::parse(NPDM).unwrap());
        let expected = CodeSet::from_nso(Nso::parse(NSO).unwrap()).unwrap();
        assert!(exefs.modules.iter().all(|module| module.code == expected));
    }

    #[test]
    fn the_module_list_is_edens() {
        // `static_modules` in `AppLoader_DeconstructedRomDirectory::Load`, golden.
        let mut expected = vec!["rtld".to_owned(), "main".to_owned()];
        expected.extend((0..10).map(|n| format!("subsdk{n}")));
        expected.push("sdk".to_owned());
        assert_eq!(MODULE_NAMES.to_vec(), expected);
        assert_eq!(NPDM_FILE, "main.npdm");
    }

    #[test]
    fn a_set_without_metadata_is_refused_before_anything_else() {
        assert!(matches!(
            ExeFs::parse(&[("rtld", NSO)]),
            Err(ExeFsError::MissingNpdm)
        ));
        assert!(matches!(
            ExeFs::parse(&[(MAIN_MODULE, &b"not an nso"[..]), (NPDM_FILE, b"short")]),
            Err(ExeFsError::Npdm(NpdmError::Truncated { len: 5 }))
        ));
        assert_eq!(
            ExeFsError::MissingNpdm.to_string(),
            "the ExeFS has no main.npdm"
        );
    }

    #[test]
    fn a_set_without_main_is_refused() {
        // Divergence from Eden, which loads such a set (see the module docs).
        let error = ExeFs::parse(&[(NPDM_FILE, NPDM), ("rtld", NSO), ("sdk", NSO)]).unwrap_err();
        assert!(matches!(error, ExeFsError::MissingMain));
        assert_eq!(error.to_string(), "the ExeFS has no main module");
        // With a `main` alone the set loads.
        assert_eq!(
            names(&ExeFs::parse(&[(NPDM_FILE, NPDM), (MAIN_MODULE, NSO)]).unwrap()),
            [MAIN_MODULE]
        );
    }

    #[test]
    fn a_malformed_module_is_named() {
        let error =
            ExeFs::parse(&[(NPDM_FILE, NPDM), (MAIN_MODULE, NSO), ("sdk", b"nope")]).unwrap_err();
        let expected = NsoError::Truncated {
            len: 4,
            needed: crate::nso::HEADER_SIZE,
        };
        assert!(matches!(
            &error,
            ExeFsError::Module {
                name: "sdk",
                error: ModuleError::Nso(NsoError::Truncated { len: 4, needed }),
            } if *needed == crate::nso::HEADER_SIZE
        ));
        assert_eq!(
            error.to_string(),
            format!("the ExeFS's sdk module is malformed: {expected}")
        );
    }

    #[test]
    fn a_module_that_parses_but_cannot_be_placed_is_named() {
        // The fixture's rodata starts on a page boundary; the NSO's header offset word for it
        // (segment header 1, memory offset at +4) moved one byte off gives a misaligned segment.
        let mut moved = NSO.to_vec();
        let at = crate::nso::SEGMENT_HEADERS_OFFSET + crate::nso::SEGMENT_HEADER_SIZE + 4;
        let offset = u32::from_le_bytes(moved[at..at + 4].try_into().unwrap());
        moved[at..at + 4].copy_from_slice(&(offset + 1).to_le_bytes());
        let error = ExeFs::parse(&[(NPDM_FILE, NPDM), (MAIN_MODULE, &moved)]).unwrap_err();
        assert!(matches!(
            error,
            ExeFsError::Module {
                name: MAIN_MODULE,
                error: ModuleError::CodeSet(CodeSetError::MisalignedSegment { offset: at_ }),
            } if at_ == offset + 1
        ));
    }

    #[test]
    fn a_file_given_twice_is_refused() {
        assert!(matches!(
            ExeFs::parse(&[(NPDM_FILE, NPDM), (MAIN_MODULE, NSO), (MAIN_MODULE, NSO)]),
            Err(ExeFsError::DuplicateFile(name)) if name == MAIN_MODULE
        ));
        // Checked before anything else: a duplicate with no metadata is still a duplicate.
        assert!(matches!(
            ExeFs::parse(&[("x", NSO), ("x", NSO)]),
            Err(ExeFsError::DuplicateFile(name)) if name == "x"
        ));
        assert_eq!(
            ExeFsError::DuplicateFile("sdk".into()).to_string(),
            "the ExeFS file sdk was given more than once"
        );
    }
}
