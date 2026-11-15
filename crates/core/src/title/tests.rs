//! Loading a program from a NUP title: which ExeFS, what is read, and each error, on synthetic
//! containers built with the loader's builder around the hello fixture (`tests/homebrew/hello/`).

use std::io::{Cursor, Read, Seek, SeekFrom};

use starquake_loader::exefs::{ExeFs, ExeFsError, MAIN_MODULE, MODULE_NAMES, ModuleError};
use starquake_loader::npdm::{META_SIZE, Npdm};
use starquake_loader::nso::{HEADER_SIZE, NsoError};
use starquake_loader::nup::build::{
    self, Content, EXEFS_DIR_NAME, PackageSpec, ROMFS_DIR_NAME, Tree,
};
use starquake_loader::nup::{NupError, PackageKind};

use super::*;
use crate::image::LoadError;

const NSO: &[u8] = include_bytes!("../../../../tests/homebrew/hello/hello.nso");
const NPDM: &[u8] = include_bytes!("../../../../tests/homebrew/hello/main.npdm");

/// A synthetic title id.
const TITLE_ID: u64 = 0x0100_0000_0000_5000;

/// Bytes that are not an NSO.
const NOT_AN_NSO: &[u8] = b"nope";

/// A package tree holding `files`, each at `exefs/<name>` and stored raw.
fn exefs_tree(files: &[(&str, &[u8])]) -> Tree {
    files.iter().fold(Tree::new(), |tree, (name, bytes)| {
        tree.with_file(&format!("{EXEFS_DIR_NAME}/{name}"), Content::raw(*bytes))
            .unwrap()
    })
}

/// A container of `packages`, as a source.
fn container(packages: &[PackageSpec]) -> Cursor<Vec<u8>> {
    Cursor::new(build::build(packages).unwrap().bytes)
}

fn base(tree: Tree) -> PackageSpec {
    PackageSpec::new(PackageKind::Base, TITLE_ID, 0, tree)
}

fn update(version: u32, tree: Tree) -> PackageSpec {
    PackageSpec::new(PackageKind::Update, TITLE_ID, version, tree)
}

/// A container whose base's ExeFS is `files`.
fn base_only(files: &[(&str, &[u8])]) -> Cursor<Vec<u8>> {
    container(&[base(exefs_tree(files))])
}

/// The program [`System::from_exefs`] loads from `files`.
fn expected(files: &[(&str, &[u8])]) -> System {
    System::from_exefs(ExeFs::parse(files).unwrap()).unwrap()
}

/// The hello fixture as an ExeFS: its metadata and its one module, `main`.
const HELLO: [(&str, &[u8]); 2] = [(NPDM_FILE, NPDM), (MAIN_MODULE, NSO)];

/// The size of a fixture file, as the loader's reader counts it.
fn size(bytes: &[u8]) -> u64 {
    u64::try_from(bytes.len()).unwrap()
}

#[test]
fn is_nup_looks_at_the_three_signature_bytes_only() {
    let [n, u, p, major] = MAGIC;
    assert_eq!(PREFIX_LEN, MAGIC.len());
    // The magic itself, and the signature with any other version byte or none.
    assert!(is_nup(&MAGIC));
    assert!(is_nup(&[n, u, p]));
    assert!(is_nup(&[n, u, p, major.wrapping_add(1)]));
    assert!(is_nup(&[n, u, p, major, 0xff, 0xff]));
    // Fewer than three bytes, or any of the three different.
    assert!(!is_nup(&[]));
    assert!(!is_nup(&[n]));
    assert!(!is_nup(&[n, u]));
    assert!(!is_nup(&[n ^ 1, u, p, major]));
    assert!(!is_nup(&[n, u ^ 1, p, major]));
    assert!(!is_nup(&[n, u, p ^ 1, major]));
    // The signature further in: an NRO has its magic at 0x10, after other bytes.
    assert!(!is_nup(&[0, n, u, p]));
    assert!(!is_nup(&starquake_loader::nro::MAGIC));
}

#[test]
fn a_base_only_title_loads_as_its_exefs_does() {
    let system = System::from_nup(base_only(&HELLO)).unwrap();
    assert_eq!(system, expected(&HELLO));
}

#[test]
fn every_module_the_loader_places_is_read_and_placed() {
    let mut files = vec![(NPDM_FILE, NPDM)];
    files.extend(MODULE_NAMES.iter().map(|name| (*name, NSO)));
    let system = System::from_nup(base_only(&files)).unwrap();
    assert_eq!(system, expected(&files));
    // Not the same program as `main` alone.
    assert_ne!(system, expected(&HELLO));
}

#[test]
fn the_newest_updates_exefs_wins_over_the_base() {
    // The base's `main` would fail to load; the update's loads.
    let broken = exefs_tree(&[(NPDM_FILE, NPDM), (MAIN_MODULE, NOT_AN_NSO)]);
    let good = exefs_tree(&HELLO);
    let system =
        System::from_nup(container(&[base(broken.clone()), update(1, good.clone())])).unwrap();
    assert_eq!(system, expected(&HELLO));

    // The other way round, the update's broken `main` is the one loaded.
    let error =
        System::from_nup(container(&[base(good.clone()), update(1, broken.clone())])).unwrap_err();
    assert!(
        matches!(
            error,
            TitleError::ExeFs(ExeFsError::Module {
                name: MAIN_MODULE,
                ..
            })
        ),
        "{error:?}"
    );

    // Of two updates, the newer one's.
    let newest_good = [
        base(broken.clone()),
        update(2, good.clone()),
        update(1, broken.clone()),
    ];
    assert_eq!(
        System::from_nup(container(&newest_good)).unwrap(),
        expected(&HELLO)
    );
    let newest_broken = [
        base(good.clone()),
        update(1, good.clone()),
        update(2, broken),
    ];
    assert!(System::from_nup(container(&newest_broken)).is_err());

    // An update with no ExeFS leaves the base's.
    let romfs_only = Tree::new()
        .with_file(
            &format!("{ROMFS_DIR_NAME}/data"),
            Content::raw(b"data".to_vec()),
        )
        .unwrap();
    assert_eq!(
        System::from_nup(container(&[base(good), update(1, romfs_only)])).unwrap(),
        expected(&HELLO)
    );
}

#[test]
fn an_updates_exefs_replaces_the_bases_as_a_whole() {
    // The base has an `rtld` the update lacks: it is not part of the program.
    let with_rtld = exefs_tree(&[(NPDM_FILE, NPDM), ("rtld", NSO), (MAIN_MODULE, NSO)]);
    let system =
        System::from_nup(container(&[base(with_rtld), update(1, exefs_tree(&HELLO))])).unwrap();
    assert_eq!(system, expected(&HELLO));
}

#[test]
fn a_title_with_no_exefs_is_refused() {
    let romfs_only = Tree::new()
        .with_file(
            &format!("{ROMFS_DIR_NAME}/data"),
            Content::raw(b"data".to_vec()),
        )
        .unwrap();
    let error = System::from_nup(container(&[base(romfs_only)])).unwrap_err();
    assert!(matches!(error, TitleError::NoExeFs), "{error:?}");
    assert_eq!(error.to_string(), "the title has no ExeFS");
}

#[test]
fn an_exefs_without_main_npdm_is_refused_before_any_file_is_read() {
    // Both faults: no `main.npdm`, and a `main` past every read limit.
    let error = load(
        base_only(&[(MAIN_MODULE, NSO)]),
        ReadLimits { npdm: 0, exefs: 0 },
    )
    .unwrap_err();
    assert!(
        matches!(error, TitleError::ExeFs(ExeFsError::MissingNpdm)),
        "{error:?}"
    );
    assert_eq!(error.to_string(), "the ExeFS has no main.npdm");
}

#[test]
fn a_module_the_loader_refuses_is_named() {
    let error = System::from_nup(base_only(&[
        (NPDM_FILE, NPDM),
        (MAIN_MODULE, NSO),
        ("sdk", NOT_AN_NSO),
    ]))
    .unwrap_err();
    let cause = NsoError::Truncated {
        len: NOT_AN_NSO.len(),
        needed: HEADER_SIZE,
    };
    assert!(
        matches!(
            &error,
            TitleError::ExeFs(ExeFsError::Module {
                name: "sdk",
                error: ModuleError::Nso(NsoError::Truncated { len, needed }),
            }) if *len == NOT_AN_NSO.len() && *needed == HEADER_SIZE
        ),
        "{error:?}"
    );
    assert_eq!(
        error.to_string(),
        format!("the ExeFS's sdk module is malformed: {cause}")
    );
}

#[test]
fn a_set_without_main_is_the_loaders_error() {
    let error = System::from_nup(base_only(&[(NPDM_FILE, NPDM), ("rtld", NSO)])).unwrap_err();
    assert!(
        matches!(error, TitleError::ExeFs(ExeFsError::MissingMain)),
        "{error:?}"
    );
}

#[test]
fn files_the_loader_does_not_place_are_not_read() {
    // A file past the whole budget, and a damaged one: neither is read, so the title loads.
    let big = vec![0x5a; usize::try_from(size(NPDM) + size(NSO) + 1).unwrap()];
    let damaged = Content::raw(b"icon".to_vec()).with_stored(b"ICON".to_vec());
    let tree = exefs_tree(&HELLO)
        .with_file(&format!("{EXEFS_DIR_NAME}/control.nacp"), Content::raw(big))
        .unwrap()
        .with_file(&format!("{EXEFS_DIR_NAME}/icon"), damaged)
        .unwrap();
    let limits = ReadLimits {
        npdm: size(NPDM),
        exefs: size(NPDM) + size(NSO),
    };
    let system = load(container(&[base(tree)]), limits).unwrap();
    assert_eq!(system, expected(&HELLO));
}

#[test]
fn main_npdm_is_read_up_to_its_own_limit_exactly() {
    let at = ReadLimits {
        npdm: size(NPDM),
        exefs: EXEFS_READ_LIMIT,
    };
    assert_eq!(load(base_only(&HELLO), at).unwrap(), expected(&HELLO));
    let under = ReadLimits {
        npdm: size(NPDM) - 1,
        ..at
    };
    let error = load(base_only(&HELLO), under).unwrap_err();
    assert!(
        matches!(
            &error,
            TitleError::Read {
                name: NPDM_FILE,
                error: NupError::FileTooLarge { size: s, limit, .. },
            } if *s == size(NPDM) && *limit == size(NPDM) - 1
        ),
        "{error:?}"
    );
    assert!(
        error
            .to_string()
            .starts_with("the ExeFS file main.npdm could not be read: file "),
        "{error}"
    );
}

#[test]
fn the_exefs_budget_covers_main_npdm_and_the_modules_together() {
    // Exactly enough for both.
    let at = ReadLimits {
        npdm: NPDM_READ_LIMIT,
        exefs: size(NPDM) + size(NSO),
    };
    assert_eq!(load(base_only(&HELLO), at).unwrap(), expected(&HELLO));

    // One byte short: `main` is refused with what was left after `main.npdm`.
    let short = ReadLimits {
        exefs: at.exefs - 1,
        ..at
    };
    let error = load(base_only(&HELLO), short).unwrap_err();
    assert!(
        matches!(
            &error,
            TitleError::Read {
                name: MAIN_MODULE,
                error: NupError::FileTooLarge { size: s, limit, .. },
            } if *s == size(NSO) && *limit == size(NSO) - 1
        ),
        "{error:?}"
    );

    // A budget smaller than `main.npdm`'s own limit bounds `main.npdm` too.
    let tiny = ReadLimits {
        npdm: NPDM_READ_LIMIT,
        exefs: size(NPDM) - 1,
    };
    let error = load(base_only(&HELLO), tiny).unwrap_err();
    assert!(
        matches!(
            &error,
            TitleError::Read {
                name: NPDM_FILE,
                error: NupError::FileTooLarge { limit, .. },
            } if *limit == size(NPDM) - 1
        ),
        "{error:?}"
    );
}

#[test]
fn the_shipping_limits_are_the_ones_used() {
    assert_eq!(
        READ_LIMITS,
        ReadLimits {
            npdm: NPDM_READ_LIMIT,
            exefs: EXEFS_READ_LIMIT,
        }
    );
    assert!(NPDM_READ_LIMIT <= EXEFS_READ_LIMIT);
}

/// A container in memory whose reads fail from `fail_from` on, as a disk that goes away after
/// the table of contents has been read.
struct FailingTail {
    inner: Cursor<Vec<u8>>,
    fail_from: u64,
}

impl Read for FailingTail {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        if self.inner.position() >= self.fail_from {
            return Err(std::io::Error::other("disk gone"));
        }
        self.inner.read(buf)
    }
}

impl Seek for FailingTail {
    fn seek(&mut self, to: SeekFrom) -> std::io::Result<u64> {
        self.inner.seek(to)
    }
}

#[test]
fn a_file_that_cannot_be_read_is_named_and_comes_before_the_loaders_checks() {
    // The source fails at the blob region; `main` is also missing, which the loader would
    // report if `main.npdm` were read.
    let built = build::build(&[base(exefs_tree(&[(NPDM_FILE, NPDM), ("rtld", NSO)]))]).unwrap();
    let source = FailingTail {
        inner: Cursor::new(built.bytes),
        fail_from: u64::try_from(built.layout.blob_offset).unwrap(),
    };
    let error = System::from_nup(source).unwrap_err();
    assert!(
        matches!(&error, TitleError::Read { name: NPDM_FILE, error } if error.is_io()),
        "{error:?}"
    );
    assert_eq!(
        error.to_string(),
        "the ExeFS file main.npdm could not be read: reading the container failed: disk gone"
    );
}

#[test]
fn a_damaged_module_is_a_read_error() {
    let mut damaged = NSO.to_vec();
    if let Some(byte) = damaged.last_mut() {
        *byte ^= 1;
    }
    let tree = Tree::new()
        .with_file(&format!("{EXEFS_DIR_NAME}/{NPDM_FILE}"), Content::raw(NPDM))
        .unwrap()
        .with_file(
            &format!("{EXEFS_DIR_NAME}/{MAIN_MODULE}"),
            Content::raw(NSO).with_stored(damaged),
        )
        .unwrap();
    let error = System::from_nup(container(&[base(tree)])).unwrap_err();
    assert!(
        matches!(
            &error,
            TitleError::Read {
                name: MAIN_MODULE,
                error: NupError::HashMismatch { .. },
            }
        ),
        "{error:?}"
    );
}

#[test]
fn a_compressed_exefs_loads_as_a_raw_one() {
    let tree = HELLO.iter().fold(Tree::new(), |tree, (name, bytes)| {
        tree.with_file(&format!("{EXEFS_DIR_NAME}/{name}"), Content::zstd(*bytes))
            .unwrap()
    });
    assert_eq!(
        System::from_nup(container(&[base(tree)])).unwrap(),
        expected(&HELLO)
    );
}

#[test]
fn a_container_that_does_not_open_or_has_no_title_is_refused() {
    let error = System::from_nup(Cursor::new(vec![0; 0x200])).unwrap_err();
    assert!(
        matches!(error, TitleError::Container(NupError::BadMagic(_))),
        "{error:?}"
    );
    assert_eq!(
        error.to_string(),
        format!(
            "the title could not be opened: {}",
            NupError::BadMagic([0; 4])
        )
    );

    // Two updates sharing the newest version: the container opens, but has no one title.
    let good = exefs_tree(&HELLO);
    let packages = [base(good.clone()), update(3, good.clone()), update(3, good)];
    let error = System::from_nup(container(&packages)).unwrap_err();
    assert!(
        matches!(
            error,
            TitleError::Container(NupError::AmbiguousUpdate { version: 3, .. })
        ),
        "{error:?}"
    );
}

/// The fixture's metadata marked 32-bit: the one byte of the `META` header whose bit 0 the
/// parser reads as 64-bit, found by flipping bit 0 of each in turn.
fn npdm_32_bit() -> Vec<u8> {
    (0..META_SIZE)
        .find_map(|at| {
            let mut bytes = NPDM.to_vec();
            bytes[at] ^= 1;
            Npdm::parse(&bytes)
                .ok()
                .filter(|npdm| !npdm.is_64_bit)
                .map(|_| bytes)
        })
        .unwrap()
}

#[test]
fn a_program_that_cannot_be_loaded_is_the_load_error() {
    let npdm = npdm_32_bit();
    let error = System::from_nup(base_only(&[(NPDM_FILE, &npdm), (MAIN_MODULE, NSO)])).unwrap_err();
    assert!(
        matches!(error, TitleError::Load(Error::Load(LoadError::Not64Bit))),
        "{error:?}"
    );
    assert_eq!(error.to_string(), LoadError::Not64Bit.to_string());
}
