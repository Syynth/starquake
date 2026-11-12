//! The NUP reader against the owner's own dumps, on the owner's machine only.
//!
//! The test runs when the `STARQUAKE_TITLES` environment variable names a folder; every `.nup`
//! file in it must open, verify (every file's SHA-256), yield a title, and have an `ExeFS` whose
//! `main.npdm` parses and whose other files parse as NSOs. When the variable is unset (as in CI)
//! the test passes without doing anything, so CI never needs a title.
//!
//! Nothing derived from a title may be committed, printed in CI logs or put in a snapshot: the
//! messages here name a title by its position in the sorted folder listing ("title 0"), never by
//! its file name, and print loader errors (offsets, sizes, record indices) but no file names
//! from inside the title.

use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;

use starquake_loader::npdm::Npdm;
use starquake_loader::nso::Nso;
use starquake_loader::nup::{Node, Nup};

/// The variable naming the folder of dumps.
const TITLES_VAR: &str = "STARQUAKE_TITLES";
/// The largest `ExeFS` file read whole. A test-local bound: the title has been verified by then,
/// so sizes are the real ones, and `ExeFS` modules are tens of megabytes.
const EXEFS_READ_LIMIT: u64 = 1 << 30;

#[test]
fn every_local_title_verifies_and_its_exefs_parses() {
    let Some(folder) = std::env::var_os(TITLES_VAR) else {
        eprintln!("skipped: {TITLES_VAR} is not set");
        return;
    };
    let mut titles: Vec<PathBuf> = std::fs::read_dir(&folder)
        .map_err(|error| format!("{TITLES_VAR} cannot be listed: {error}"))
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "nup"))
        .collect();
    titles.sort();
    assert!(
        !titles.is_empty(),
        "{TITLES_VAR} names a folder with no .nup files"
    );

    for (index, path) in titles.iter().enumerate() {
        // Errors become strings before they are unwrapped, so that a failure prints the message
        // written here and never a `Debug` form that could hold a name from inside the title.
        let file = File::open(path)
            .map_err(|error| format!("title {index}: {error}"))
            .unwrap();
        let mut nup = Nup::open(BufReader::new(file))
            .map_err(|error| format!("title {index} does not open: {error}"))
            .unwrap();
        let summary = nup
            .verify()
            .map_err(|error| {
                format!(
                    "title {index}: package {} has a damaged file: {}",
                    error.package, error.source
                )
            })
            .unwrap();
        assert_eq!(
            summary.files,
            u64::try_from(nup.file_count()).unwrap(),
            "title {index}: verify skipped files"
        );

        let title = nup
            .title()
            .map_err(|error| format!("title {index}: {error}"))
            .unwrap();
        let npdm = title
            .exefs_file(&nup, "main.npdm")
            .ok_or_else(|| format!("title {index}: no main.npdm in its ExeFS"))
            .unwrap();
        let bytes = nup
            .read_to_vec(npdm, EXEFS_READ_LIMIT)
            .map_err(|error| format!("title {index}: main.npdm: {error}"))
            .unwrap();
        Npdm::parse(&bytes)
            .map_err(|error| format!("title {index}: main.npdm: {error}"))
            .unwrap();

        let mut modules = 0;
        for file in title.exefs_files(&nup) {
            if nup.name(Node::File(file)) == b"main.npdm" {
                continue;
            }
            modules += 1;
            let bytes = nup
                .read_to_vec(file, EXEFS_READ_LIMIT)
                .map_err(|error| format!("title {index}: ExeFS module {modules}: {error}"))
                .unwrap();
            Nso::parse(&bytes)
                .map_err(|error| format!("title {index}: ExeFS module {modules}: {error}"))
                .unwrap();
        }
        assert!(modules > 0, "title {index}: its ExeFS has no modules");
        eprintln!(
            "title {index}: verified {} files, {} distinct extents; main.npdm and {modules} \
             modules parse",
            summary.files, summary.extents
        );
    }
}
