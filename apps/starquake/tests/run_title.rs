//! `starquake run <title.nup>`: a NUP title's ExeFS run as a program, end to end.
//!
//! The titles are synthetic, built with the loader's builder around the hello fixture
//! (`tests/homebrew/hello/`): its `main.npdm` and its NSO as `main`.
//!
//! Tests that need a VM need a Mac with Apple Silicon and skip themselves only when the `CI`
//! environment variable is set; the rest run on every host.

use std::ffi::OsString;
use std::path::{Path, PathBuf};

use starquake_core::artifacts::{GUEST_LOG_FILE, SUMMARY_FILE};
use starquake_core::exit_code;
use starquake_core::title::PREFIX_LEN;
use starquake_loader::exefs::{MAIN_MODULE, NPDM_FILE};
use starquake_loader::nso::{HEADER_SIZE, NsoError};
use starquake_loader::nup::build::{
    self, Content, EXEFS_DIR_NAME, PackageSpec, ROMFS_DIR_NAME, Tree,
};
use starquake_loader::nup::layout::MAGIC;
use starquake_loader::nup::{NupError, PackageKind};

const NSO: &[u8] = include_bytes!("../../../tests/homebrew/hello/hello.nso");
const NRO: &[u8] = include_bytes!("../../../tests/homebrew/hello/hello.nro");
const NPDM: &[u8] = include_bytes!("../../../tests/homebrew/hello/main.npdm");

/// A synthetic title id.
const TITLE_ID: u64 = 0x0100_0000_0000_5000;

mod common;

/// Bytes that are not an NSO.
const NOT_AN_NSO: &[u8] = b"nope";

/// Helpers, in a `cfg(test)` module so they may unwrap like the tests do.
#[cfg(test)]
mod support {
    use super::*;

    /// What a command printed and returned.
    pub(crate) struct Ran {
        pub(crate) code: u8,
        pub(crate) stdout: String,
        pub(crate) stderr: String,
    }

    /// An empty scratch folder for the test called `name`.
    pub(crate) fn scratch(name: &str) -> PathBuf {
        let folder = Path::new(env!("CARGO_TARGET_TMPDIR"))
            .join("run-title")
            .join(name);
        if folder.exists() {
            std::fs::remove_dir_all(&folder).unwrap();
        }
        std::fs::create_dir_all(&folder).unwrap();
        folder
    }

    /// Writes `bytes` to `folder/name` and returns its path.
    pub(crate) fn write(folder: &Path, name: &str, bytes: &[u8]) -> PathBuf {
        let path = folder.join(name);
        std::fs::write(&path, bytes).unwrap();
        path
    }

    /// A package tree holding `files` in its ExeFS, stored raw.
    pub(crate) fn exefs(files: &[(&str, &[u8])]) -> Tree {
        files.iter().fold(Tree::new(), |tree, (name, bytes)| {
            tree.with_file(&format!("{EXEFS_DIR_NAME}/{name}"), Content::raw(*bytes))
                .unwrap()
        })
    }

    pub(crate) fn base(tree: Tree) -> PackageSpec {
        PackageSpec::new(PackageKind::Base, TITLE_ID, 0, tree)
    }

    pub(crate) fn update(version: u32, tree: Tree) -> PackageSpec {
        PackageSpec::new(PackageKind::Update, TITLE_ID, version, tree)
    }

    /// Writes a container of `packages` to `folder/title.nup` and returns its path.
    pub(crate) fn title(folder: &Path, packages: &[PackageSpec]) -> PathBuf {
        write(folder, "title.nup", &build::build(packages).unwrap().bytes)
    }

    /// `run <program> --artifacts <folder>/artifacts --until timeout=30s --date <the pinned
    /// date>`; returns what it printed and the artifacts folder.
    pub(crate) fn run(program: &Path) -> (Ran, PathBuf) {
        let artifacts = program.parent().unwrap().join("artifacts");
        let mut line: Vec<OsString> = vec![
            "starquake".into(),
            "run".into(),
            program.into(),
            "--artifacts".into(),
            artifacts.clone().into(),
            "--until".into(),
            "timeout=30s".into(),
        ];
        line.extend(common::pinned_date());
        let (mut stdout, mut stderr) = (Vec::new(), Vec::new());
        let code = starquake::execute(&line, &mut stdout, &mut stderr);
        let ran = Ran {
            code,
            stdout: String::from_utf8(stdout).unwrap(),
            stderr: String::from_utf8(stderr).unwrap(),
        };
        (ran, artifacts)
    }

    pub(crate) fn text(folder: &Path, name: &str) -> String {
        std::fs::read_to_string(folder.join(name)).unwrap()
    }

    pub(crate) fn summary(folder: &Path) -> serde_json::Value {
        serde_json::from_str(&text(folder, SUMMARY_FILE)).unwrap()
    }

    /// Runs `program`, which must fail to load with `message` (after its path): checks the exit
    /// code, the line printed, and the summary.
    pub(crate) fn refused(program: &Path, message: &str) {
        let (ran, artifacts) = run(program);
        let message = format!("{}: {message}", program.display());
        assert_eq!(ran.code, exit_code::EMULATOR_ERROR, "{}", ran.stdout);
        assert_eq!(ran.stderr, "");
        assert_eq!(
            ran.stdout,
            format!(
                "starquake: emulator error: {message} (exit code {}); artifacts in {}\n",
                exit_code::EMULATOR_ERROR,
                artifacts.display()
            )
        );
        let summary = summary(&artifacts);
        assert_eq!(summary["exit_reason"], "emulator_error");
        assert_eq!(summary["exit_code"], u64::from(exit_code::EMULATOR_ERROR));
        assert_eq!(summary["message"], message);
    }

    /// Whether this host can run guest code in a test: a Mac with Apple Silicon, off hosted CI.
    pub(crate) fn can_run_guests() -> bool {
        cfg!(all(target_os = "macos", target_arch = "aarch64")) && std::env::var_os("CI").is_none()
    }
}

use support::{
    base, can_run_guests, exefs, refused, run, scratch, summary, text, title, update, write,
};

/// The hello fixture as an ExeFS.
const HELLO: [(&str, &[u8]); 2] = [(NPDM_FILE, NPDM), (MAIN_MODULE, NSO)];

#[test]
fn a_title_holding_the_hello_program_runs_to_exit_as_the_fixture_does() {
    if !can_run_guests() {
        return;
    }
    let folder = scratch("hello");
    let program = title(&folder, &[base(exefs(&HELLO))]);
    let (ran, artifacts) = run(&program);
    assert_eq!(ran.code, exit_code::STOP_CONDITION, "{}", ran.stdout);
    assert_eq!(ran.stderr, "");
    assert_eq!(summary(&artifacts)["exit_reason"], "process_exited");

    // The fixture's own NRO, run the same way, prints the same lines.
    let twin_folder = scratch("hello-twin");
    let twin = write(&twin_folder, "hello.nro", NRO);
    let (twin_ran, twin_artifacts) = run(&twin);
    assert_eq!(
        twin_ran.code,
        exit_code::STOP_CONDITION,
        "{}",
        twin_ran.stdout
    );
    let log = text(&artifacts, GUEST_LOG_FILE);
    assert_eq!(log, text(&twin_artifacts, GUEST_LOG_FILE));
    assert_eq!(log.lines().count(), 3, "{log}");
}

#[test]
fn an_updates_exefs_is_run_instead_of_the_bases() {
    if !can_run_guests() {
        return;
    }
    // The base's `main` would not load; the update's is the hello program.
    let folder = scratch("update");
    let broken = exefs(&[(NPDM_FILE, NPDM), (MAIN_MODULE, NOT_AN_NSO)]);
    let program = title(&folder, &[base(broken), update(1, exefs(&HELLO))]);
    let (ran, artifacts) = run(&program);
    assert_eq!(ran.code, exit_code::STOP_CONDITION, "{}", ran.stdout);
    assert_eq!(text(&artifacts, GUEST_LOG_FILE).lines().count(), 3);
}

#[test]
fn a_title_with_no_exefs_is_an_emulator_error() {
    let folder = scratch("no-exefs");
    let romfs_only = Tree::new()
        .with_file(
            &format!("{ROMFS_DIR_NAME}/data"),
            Content::raw(b"data".to_vec()),
        )
        .unwrap();
    let program = title(&folder, &[base(romfs_only)]);
    refused(&program, "the title has no ExeFS");
}

#[test]
fn an_exefs_without_main_npdm_is_an_emulator_error() {
    let folder = scratch("no-npdm");
    let program = title(&folder, &[base(exefs(&[(MAIN_MODULE, NSO)]))]);
    refused(&program, "the ExeFS has no main.npdm");
}

#[test]
fn a_module_the_loader_refuses_is_an_emulator_error_that_names_it() {
    let folder = scratch("bad-module");
    let files = [(NPDM_FILE, NPDM), (MAIN_MODULE, NSO), ("sdk", NOT_AN_NSO)];
    let program = title(&folder, &[base(exefs(&files))]);
    let cause = NsoError::Truncated {
        len: NOT_AN_NSO.len(),
        needed: HEADER_SIZE,
    };
    refused(
        &program,
        &format!("the ExeFS's sdk module is malformed: {cause}"),
    );
}

#[test]
fn a_file_that_starts_like_a_title_is_opened_as_one() {
    // The NUP signature and nothing else: opened as a title, which is too short.
    let folder = scratch("short-title");
    let [n, u, p, _major] = MAGIC;
    let signature = [n, u, p];
    let program = write(&folder, "short.nro", &signature);
    let cause = NupError::SourceTooShort {
        len: u64::try_from(signature.len()).unwrap(),
    };
    refused(&program, &format!("the title could not be opened: {cause}"));
}

#[test]
fn a_file_shorter_than_the_titles_magic_is_read_as_an_nro() {
    let folder = scratch("tiny");
    let tiny = vec![0x4e; PREFIX_LEN - 1];
    let program = write(&folder, "tiny.nup", &tiny);
    let cause = starquake_core::System::new(&tiny).unwrap_err();
    refused(&program, &cause.to_string());
}
