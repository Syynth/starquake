//! `starquake inspect` and `starquake verify` end to end, on synthetic NUP files.
//!
//! The containers are built with the loader's test builder; their `ExeFS` holds the homebrew
//! fixture's `hello.nso` and `main.npdm` (`tests/homebrew/hello`), so the NSO and NPDM parsers
//! have real input. Expected values are read back through the loader's parsers, not restated.

use std::ffi::OsString;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use starquake::title::{NPDM_READ_LIMIT, exit};
use starquake_core::loader::npdm::Npdm;
use starquake_core::loader::nso::Nso;
// The builder is behind the loader's `test-support` feature, which this crate's dev-dependency on
// the loader turns on; it is the same crate `starquake_core::loader` re-exports.
use starquake_core::loader::nup::{Nup, PackageKind};
use starquake_loader::nup::build::{self, Built, Content, PackageSpec, Tree};

const BASE_ID: u64 = 0x0100_0000_0000_3000;
const UPDATE_ID: u64 = 0x0100_0000_0000_3800;
const ADDON_ID: u64 = 0x0100_0000_0000_4001;

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

    /// Runs `starquake <args>` in this process.
    pub(crate) fn starquake(args: &[&str]) -> Ran {
        let mut line = vec![OsString::from("starquake")];
        line.extend(args.iter().map(OsString::from));
        let (mut stdout, mut stderr) = (Vec::new(), Vec::new());
        let code = starquake::execute(&line, &mut stdout, &mut stderr);
        Ran {
            code,
            stdout: String::from_utf8(stdout).unwrap(),
            stderr: String::from_utf8(stderr).unwrap(),
        }
    }

    /// An empty scratch folder for the test called `name`.
    pub(crate) fn scratch(name: &str) -> PathBuf {
        let folder = Path::new(env!("CARGO_TARGET_TMPDIR"))
            .join("title")
            .join(name);
        if folder.exists() {
            std::fs::remove_dir_all(&folder).unwrap();
        }
        std::fs::create_dir_all(&folder).unwrap();
        folder
    }

    pub(crate) fn fixture(name: &str) -> Vec<u8> {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/homebrew/hello")
            .join(name);
        std::fs::read(path).unwrap()
    }

    /// A base whose `ExeFS` is the fixture, and an update with the same modules stored as zstd and
    /// a `RomFS` file.
    pub(crate) fn title() -> Built {
        let base = Tree::new()
            .with_file("exefs/main", Content::raw(fixture("hello.nso")))
            .unwrap()
            .with_file("exefs/main.npdm", Content::raw(fixture("main.npdm")))
            .unwrap()
            .with_file("romfs/hello.txt", Content::raw(b"hello".to_vec()))
            .unwrap();
        let update = Tree::new()
            .with_file("exefs/main", Content::zstd(fixture("hello.nso")))
            .unwrap()
            .with_file("exefs/main.npdm", Content::zstd(fixture("main.npdm")))
            .unwrap()
            .with_file("romfs/hello.txt", Content::raw(b"hello, again".to_vec()))
            .unwrap();
        build::build(&[
            PackageSpec::new(PackageKind::Base, BASE_ID, 0, base),
            PackageSpec::new(PackageKind::Update, UPDATE_ID, 0x1_0203, update),
        ])
        .unwrap()
    }

    pub(crate) fn write(folder: &Path, name: &str, bytes: &[u8]) -> PathBuf {
        let path = folder.join(name);
        std::fs::write(&path, bytes).unwrap();
        path
    }

    pub(crate) fn text(path: &Path) -> &str {
        path.to_str().unwrap()
    }
}

use support::{fixture, scratch, starquake, text, title, write};

#[test]
fn inspect_prints_the_packages_the_newest_exefs_and_the_npdm() {
    let folder = scratch("inspect");
    let built = title();
    let path = write(&folder, "title.nup", &built.bytes);
    let ran = starquake(&["inspect", text(&path)]);
    assert_eq!(ran.stderr, "");
    assert_eq!(ran.code, exit::OK);

    let nup = Nup::open(std::io::Cursor::new(&built.bytes)).unwrap();
    let npdm = Npdm::parse(&fixture("main.npdm")).unwrap();
    let nso = Nso::parse(&fixture("hello.nso")).unwrap();
    let build_id = nso.build_id.iter().fold(String::new(), |mut hex, byte| {
        let _ = write!(hex, "{byte:02x}");
        hex
    });
    let nso_size = fixture("hello.nso").len();
    let npdm_size = fixture("main.npdm").len();
    let stored = |path: &str| built.layout.extents[built.file_index(1, path).unwrap()].1;

    let expected_lines = [
        format!(
            "container: NUP version 1, 2 packages, {} directories, {} files",
            nup.dir_count(),
            nup.file_count()
        ),
        format!(
            "package 0: base, title {BASE_ID:016x}, version 0 (0.0.0), exefs 2 files ({} bytes), \
             romfs 1 files (5 bytes)",
            nso_size + npdm_size
        ),
        format!(
            "package 1: update, title {UPDATE_ID:016x}, version 66051 (1.2.3), exefs 2 files ({} \
             bytes), romfs 1 files (12 bytes)",
            nso_size + npdm_size
        ),
        format!("title: {BASE_ID:016x}, base in package 0, newest update in package 1"),
        "exefs (from package 1):".to_owned(),
        format!(
            "  main             {nso_size:>12} bytes  zstd, {} bytes stored",
            stored("exefs/main")
        ),
        format!(
            "  main.npdm        {npdm_size:>12} bytes  zstd, {} bytes stored",
            stored("exefs/main.npdm")
        ),
        "main.npdm:".to_owned(),
        format!("  name: {}", npdm.name),
        format!("  program id (ACI0): {:016x}", npdm.aci.title_id),
        "  ACID layout: Switch 1 (magic at 0x200)".to_owned(),
        "modules:".to_owned(),
        format!(
            "  main             build id {build_id}, text {:#x}, rodata {:#x}, data {:#x}, bss {:#x}",
            nso.text.data.len(),
            nso.rodata.data.len(),
            nso.data.data.len(),
            nso.bss_size
        ),
    ];
    for line in expected_lines {
        assert!(
            ran.stdout.lines().any(|l| l == line),
            "missing {line:?} in:\n{}",
            ran.stdout
        );
    }
    let svcs = npdm.aci.kernel_capabilities.svc_mask.iter().count();
    assert!(
        ran.stdout
            .contains(&format!("  allowed system calls ({svcs}): ")),
        "{}",
        ran.stdout
    );
    let opened = npdm.aci.services.iter().filter(|s| !s.server).count();
    assert!(
        ran.stdout
            .contains(&format!("  services it may open ({opened}): ")),
        "{}",
        ran.stdout
    );
}

#[test]
fn verify_reports_an_intact_title_with_its_counts() {
    let folder = scratch("verify-ok");
    let built = title();
    let path = write(&folder, "title.nup", &built.bytes);
    let ran = starquake(&["verify", text(&path)]);
    let summary = Nup::open(std::io::Cursor::new(&built.bytes))
        .unwrap()
        .verify()
        .unwrap();
    assert_eq!(
        (ran.code, ran.stderr.as_str()),
        (exit::OK, ""),
        "{}",
        ran.stdout
    );
    assert_eq!(
        ran.stdout,
        format!(
            "starquake: {} is intact: the table of contents and all {} files ({} distinct \
             extents, {} bytes) match their SHA-256\n",
            path.display(),
            summary.files,
            summary.extents,
            summary.bytes
        )
    );
}

#[test]
fn verify_names_the_first_damaged_file_and_exits_with_code_1() {
    let folder = scratch("verify-damaged");
    let mut built = title();
    let file = built.file_index(1, "romfs/hello.txt").unwrap();
    let (offset, _) = built.layout.extents[file];
    built.bytes[offset] ^= 1;
    let path = write(&folder, "title.nup", &built.bytes);
    let ran = starquake(&["verify", text(&path)]);
    assert_eq!(ran.code, exit::DAMAGED);
    assert_eq!(
        ran.stdout,
        format!(
            "starquake: {} is damaged: the first file that failed is romfs/hello.txt in package 1\n",
            path.display()
        )
    );
    assert_eq!(
        ran.stderr,
        format!(
            "starquake: {}: package 1, romfs/hello.txt: file {file}'s content does not match its \
             SHA-256\n",
            path.display()
        )
    );
}

#[test]
fn a_file_that_is_not_a_nup_or_is_cut_short_is_damage() {
    let folder = scratch("not-a-nup");
    let nro = write(&folder, "hello.nro", &fixture("hello.nro"));
    for command in ["inspect", "verify"] {
        let ran = starquake(&[command, text(&nro)]);
        assert_eq!(ran.code, exit::DAMAGED, "{command}");
        assert_eq!(ran.stdout, "", "{command}");
        assert!(
            ran.stderr.starts_with(&format!(
                "starquake: {}: not a NUP file: the magic is ",
                nro.display()
            )),
            "{command}: {}",
            ran.stderr
        );
    }

    let built = title();
    let cut = write(&folder, "cut.nup", &built.bytes[..built.bytes.len() - 1]);
    let ran = starquake(&["verify", text(&cut)]);
    assert_eq!(ran.code, exit::DAMAGED);
    assert_eq!(
        ran.stderr,
        format!(
            "starquake: {}: the superblock says the file is {} bytes, but it is {}\n",
            cut.display(),
            built.bytes.len(),
            built.bytes.len() - 1
        )
    );
}

#[test]
fn a_file_that_cannot_be_opened_exits_with_code_2() {
    let folder = scratch("missing");
    let missing = folder.join("missing.nup");
    for command in ["inspect", "verify"] {
        let ran = starquake(&[command, text(&missing)]);
        assert_eq!(ran.code, exit::UNREADABLE, "{command}");
        assert_eq!(ran.stdout, "");
        assert!(
            ran.stderr.starts_with(&format!(
                "starquake: could not open {}: ",
                missing.display()
            )),
            "{command}: {}",
            ran.stderr
        );
    }
}

#[test]
fn inspect_says_when_a_container_holds_no_program() {
    let folder = scratch("no-program");
    let addon = Tree::new()
        .with_file("romfs/dlc.bin", Content::raw(vec![7; 32]))
        .unwrap();
    let built = build::build(&[PackageSpec::new(PackageKind::AddOn, ADDON_ID, 0, addon)]).unwrap();
    let path = write(&folder, "dlc.nup", &built.bytes);
    let ran = starquake(&["inspect", text(&path)]);
    assert_eq!(ran.code, exit::DAMAGED);
    assert!(
        ran.stdout
            .contains(&format!("package 0: addon, title {ADDON_ID:016x}, version 0 (0.0.0), no exefs, romfs 1 files (32 bytes)")),
        "{}",
        ran.stdout
    );
    assert_eq!(
        ran.stderr,
        format!(
            "starquake: {}: no title: the container holds no program: no base and no update \
             package\n",
            path.display()
        )
    );
}

#[test]
fn inspect_reports_an_npdm_it_cannot_parse_or_will_not_read() {
    let folder = scratch("bad-npdm");
    let tree = |npdm: Vec<u8>| {
        Tree::new()
            .with_file("exefs/main", Content::raw(fixture("hello.nso")))
            .unwrap()
            .with_file("exefs/main.npdm", Content::raw(npdm))
            .unwrap()
    };
    let garbage = build::build(&[PackageSpec::new(
        PackageKind::Base,
        BASE_ID,
        0,
        tree(vec![0xEE; 0x80]),
    )])
    .unwrap();
    let path = write(&folder, "garbage.nup", &garbage.bytes);
    let ran = starquake(&["inspect", text(&path)]);
    assert_eq!(ran.code, exit::DAMAGED);
    assert_eq!(
        ran.stderr,
        format!(
            "starquake: {}: main.npdm: NPDM magic is not META\n",
            path.display()
        )
    );

    // One byte over the read limit is refused before it is read.
    let limit = usize::try_from(NPDM_READ_LIMIT).unwrap();
    let huge = build::build(&[PackageSpec::new(
        PackageKind::Base,
        BASE_ID,
        0,
        tree(vec![0; limit + 1]),
    )])
    .unwrap();
    let path = write(&folder, "huge.nup", &huge.bytes);
    let ran = starquake(&["inspect", text(&path)]);
    assert_eq!(ran.code, exit::DAMAGED);
    let npdm = huge.file_index(0, "exefs/main.npdm").unwrap();
    assert_eq!(
        ran.stderr,
        format!(
            "starquake: {}: main.npdm: file {npdm} is {} bytes, more than the {limit} allowed\n",
            path.display(),
            limit + 1
        )
    );
}

#[test]
fn the_process_exit_status_is_verifys_exit_code() {
    let folder = scratch("process");
    let mut built = title();
    let path = write(&folder, "good.nup", &built.bytes);
    let status = std::process::Command::new(env!("CARGO_BIN_EXE_starquake"))
        .args(["verify", text(&path)])
        .output()
        .unwrap();
    assert_eq!(status.status.code(), Some(i32::from(exit::OK)));

    let file = built.file_index(0, "exefs/main").unwrap();
    built.bytes[built.layout.extents[file].0] ^= 1;
    let path = write(&folder, "bad.nup", &built.bytes);
    let status = std::process::Command::new(env!("CARGO_BIN_EXE_starquake"))
        .args(["verify", text(&path)])
        .output()
        .unwrap();
    assert_eq!(status.status.code(), Some(i32::from(exit::DAMAGED)));
}
