//! The unused-dependency check for the tables `cargo machete` does not read.
//!
//! `cargo machete` 0.9.2 analyses only a package's top-level `[dependencies]` table. This module
//! covers the rest: `[dev-dependencies]`, `[build-dependencies]` and the same three tables under
//! `[target.<cfg>.*]`. (Top-level `[dependencies]` stays machete's.)
//!
//! # How a dependency counts as used
//!
//! The check is textual, like machete's, and no more precise than it: a dependency is used when
//! its crate name (the key in the manifest, `-` written as `_`, as the code spells it) appears as
//! a whole word in the package's Rust sources.
//!
//! - A **build dependency** is looked for in the build script only: `build.rs`, the file
//!   `package.build` names, or nowhere when `build = false`.
//! - Every other dependency is looked for in every `.rs` file below the package folder, skipping
//!   folders named `target`. That includes `tests/`, `benches/` and `examples/`, and doc
//!   comments, where doctests use dev-dependencies.
//!
//! # Where it is imprecise
//!
//! - A word in a comment or string counts as a use, so a dependency that is only mentioned is
//!   not reported (the same bias as machete's).
//! - A crate whose library name differs from its package name (`name = ...` under `[lib]` of the
//!   dependency) is reported although it is used: list it in `xtask/unused-deps.toml`
//!   (`[[allow_tables]]`) with that reason.
//! - A dependency used by a macro expansion only, with no mention in the source, is reported:
//!   same remedy.
//!
//! Findings use the allow-list and its two-way rule (`[[allow_tables]]`, see
//! [`crate::unused_deps::TABLES_SCOPE`]).

use std::fs;
use std::path::{Path, PathBuf};

use crate::unused_deps::{Error, Unused, member_manifests, parse_manifest, tables_package};

/// The dependency tables of a manifest or of one `[target.*]` table the check reads.
const TABLES: [&str; 3] = ["dependencies", "dev-dependencies", "build-dependencies"];

/// One unused dependency in a table other than the top-level `[dependencies]`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct TableUnused {
    /// The pair as the allow-list names it: the package carries [`tables_package`].
    pub unused: Unused,
    /// Where it is declared: `dev-dependencies`, or `target.'cfg(unix)'.build-dependencies`.
    pub table: String,
}

/// A dependency declared in a table the check reads.
#[derive(Debug, PartialEq, Eq)]
struct Declared {
    /// `dev-dependencies`, `target.<cfg>.dependencies` and so on, as the manifest spells it.
    table: String,
    /// Whether this is a build dependency (looked for in the build script only).
    build: bool,
    /// The key in the table: the name the code uses, `-` and all.
    name: String,
}

/// The package's Rust code, as the check sees it.
#[derive(Debug, Default)]
pub struct Sources {
    /// The build script's text; empty when the package has none.
    pub build_script: String,
    /// The text of every other `.rs` file of the package, joined.
    pub all: String,
}

/// Whether `word` occurs in `text` bounded by non-identifier characters (or the text's ends).
fn contains_word(text: &str, word: &str) -> bool {
    let is_ident = |c: char| c.is_alphanumeric() || c == '_';
    text.match_indices(word).any(|(start, _)| {
        let before = text[..start].chars().next_back();
        let after = text[start + word.len()..].chars().next();
        !before.is_some_and(is_ident) && !after.is_some_and(is_ident)
    })
}

/// The dependencies the manifest declares in dev, build and target-specific tables, in manifest
/// order (tables sorted by name, as the TOML parser keeps them).
fn declared(manifest: &toml::Table) -> Vec<Declared> {
    let mut found = Vec::new();
    let mut read = |prefix: &str, table: &toml::Table| {
        for kind in TABLES {
            let Some(dependencies) = table.get(kind).and_then(toml::Value::as_table) else {
                continue;
            };
            // Top-level `[dependencies]` is `cargo machete`'s.
            if prefix.is_empty() && kind == "dependencies" {
                continue;
            }
            for name in dependencies.keys() {
                found.push(Declared {
                    table: format!("{prefix}{kind}"),
                    build: kind == "build-dependencies",
                    name: name.clone(),
                });
            }
        }
    };
    read("", manifest);
    if let Some(targets) = manifest.get("target").and_then(toml::Value::as_table) {
        for (cfg, table) in targets {
            if let Some(table) = table.as_table() {
                read(&format!("target.'{cfg}'."), table);
            }
        }
    }
    found
}

/// The dev, build and target-specific dependencies of one package that its sources never name.
///
/// `manifest` is the package's `Cargo.toml` text and `path` its name in error messages. A
/// manifest without `[package]` (a virtual workspace root) has none.
pub fn unused_in_package(
    path: &str,
    manifest: &str,
    sources: &Sources,
) -> Result<Vec<TableUnused>, Error> {
    let manifest = parse_manifest(path, manifest)?;
    let Some(package) = manifest
        .get("package")
        .and_then(|package| package.get("name"))
        .and_then(toml::Value::as_str)
    else {
        return Ok(Vec::new());
    };
    let mut unused: Vec<TableUnused> = declared(&manifest)
        .into_iter()
        .filter(|dependency| {
            let word = dependency.name.replace('-', "_");
            let text = if dependency.build {
                &sources.build_script
            } else {
                &sources.all
            };
            !contains_word(text, &word)
        })
        .map(|dependency| TableUnused {
            unused: Unused {
                package: tables_package(package),
                dependency: dependency.name,
            },
            table: dependency.table,
        })
        .collect();
    unused.sort();
    Ok(unused)
}

/// The build script of the package: `package.build` as a path, `build.rs` when the key is absent
/// and the file exists, and none when `build = false`.
fn build_script_path(manifest: &toml::Table, folder: &Path) -> Option<PathBuf> {
    let build = manifest
        .get("package")
        .and_then(|package| package.get("build"));
    match build {
        Some(toml::Value::String(path)) => Some(folder.join(path)),
        Some(_) => None,
        None => Some(folder.join("build.rs")).filter(|path| path.is_file()),
    }
}

fn io_error(path: &Path) -> impl FnOnce(std::io::Error) -> Error {
    let path = path.display().to_string();
    move |source| Error::Io { path, source }
}

/// Every `.rs` file below `folder`, skipping folders named `target`, in path order.
fn rust_files(folder: &Path, files: &mut Vec<PathBuf>) -> Result<(), Error> {
    let mut entries = Vec::new();
    for entry in fs::read_dir(folder).map_err(io_error(folder))? {
        entries.push(entry.map_err(io_error(folder))?.path());
    }
    entries.sort();
    for path in entries {
        if path.is_dir() {
            if path.file_name().is_some_and(|name| name != "target") {
                rust_files(&path, files)?;
            }
        } else if path.extension().is_some_and(|extension| extension == "rs") {
            files.push(path);
        }
    }
    Ok(())
}

/// Reads the sources of the package whose manifest is `manifest_path`.
fn read_sources(manifest_path: &Path, manifest: &str) -> Result<Sources, Error> {
    let folder = manifest_path.parent().unwrap_or_else(|| Path::new("."));
    let table = parse_manifest(&manifest_path.display().to_string(), manifest)?;
    let read = |path: &Path| fs::read_to_string(path).map_err(io_error(path));
    let build_script = match build_script_path(&table, folder) {
        Some(path) => read(&path)?,
        None => String::new(),
    };
    let mut files = Vec::new();
    rust_files(folder, &mut files)?;
    let mut all = String::new();
    for file in files {
        all.push_str(&read(&file)?);
        all.push('\n');
    }
    Ok(Sources { build_script, all })
}

/// Runs the check on every workspace member of the checkout at `root_dir`, in member order.
pub fn check_table_dependencies(root_dir: &Path) -> Result<Vec<TableUnused>, Error> {
    let root_path = root_dir.join("Cargo.toml");
    let root = fs::read_to_string(&root_path).map_err(io_error(&root_path))?;
    let mut found = Vec::new();
    for path in member_manifests(root_dir, &root)? {
        let manifest = fs::read_to_string(&path).map_err(io_error(&path))?;
        let sources = read_sources(&path, &manifest)?;
        found.extend(unused_in_package(
            &path.display().to_string(),
            &manifest,
            &sources,
        )?);
    }
    found.sort();
    Ok(found)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::unused_deps::{Allowlist, Scope, TABLES_SCOPE, Verdict, judge};

    fn sources(build_script: &str, all: &str) -> Sources {
        Sources {
            build_script: build_script.to_owned(),
            all: all.to_owned(),
        }
    }

    /// `(table, dependency)` of every finding for `manifest`.
    fn findings(manifest: &str, sources: &Sources) -> Vec<(String, String)> {
        unused_in_package("Cargo.toml", manifest, sources)
            .unwrap()
            .into_iter()
            .map(|found| (found.table, found.unused.dependency))
            .collect()
    }

    fn pair(table: &str, dependency: &str) -> (String, String) {
        (table.to_owned(), dependency.to_owned())
    }

    const MANIFEST: &str = r#"
[package]
name = "demo"

[dependencies]
top-level = "1"

[dev-dependencies]
proptest = "1"
insta = "1"

[build-dependencies]
cc = "1"

[target.'cfg(unix)'.dependencies]
libc = "0.2"

[target.'cfg(unix)'.dev-dependencies]
tempfile = "3"

[target.'cfg(windows)'.build-dependencies]
winres = "0.1"
"#;

    #[test]
    fn a_dependency_in_every_unread_table_is_reported_when_nothing_names_it() {
        // Findings are sorted by dependency name. Each of the unread tables has an entry, and the
        // top-level `[dependencies]` entry is left to machete.
        assert_eq!(
            findings(MANIFEST, &sources("", "")),
            [
                pair("build-dependencies", "cc"),
                pair("dev-dependencies", "insta"),
                pair("target.'cfg(unix)'.dependencies", "libc"),
                pair("dev-dependencies", "proptest"),
                pair("target.'cfg(unix)'.dev-dependencies", "tempfile"),
                pair("target.'cfg(windows)'.build-dependencies", "winres"),
            ]
        );
    }

    #[test]
    fn findings_carry_the_tables_scope_so_the_allowlist_can_tell_them_apart() {
        let found = unused_in_package("Cargo.toml", MANIFEST, &sources("", "")).unwrap();
        assert!(
            found
                .iter()
                .all(|found| found.unused.package == format!("{TABLES_SCOPE}demo"))
        );
        assert_eq!(found.len(), 6);
    }

    #[test]
    fn a_named_dependency_is_used() {
        let all = "use proptest::prelude::*;\n#[test]\nfn t() { insta::assert_snapshot!(1); }\n\
                   fn f() { libc::getpid(); let _ = tempfile::tempdir(); }";
        let build = "fn main() { cc::Build::new(); }";
        assert_eq!(
            findings(MANIFEST, &sources(build, all)),
            [pair("target.'cfg(windows)'.build-dependencies", "winres")]
        );
    }

    #[test]
    fn a_hyphen_in_the_manifest_is_an_underscore_in_the_code() {
        let manifest = "[package]\nname = \"p\"\n[dev-dependencies]\nstarquake-testkit = \"1\"\n";
        assert_eq!(
            findings(manifest, &sources("", "use starquake_testkit::Rig;")),
            []
        );
        // The hyphenated spelling is not how Rust names a crate.
        assert_eq!(
            findings(manifest, &sources("", "// starquake-testkit")),
            [pair("dev-dependencies", "starquake-testkit")]
        );
    }

    #[test]
    fn only_a_whole_word_counts() {
        let manifest = "[package]\nname = \"p\"\n[dev-dependencies]\nrand = \"1\"\n";
        for text in [
            "operand",
            "rand_core::X",
            "my_rand",
            "rand2",
            "brand::new()",
        ] {
            assert_eq!(
                findings(manifest, &sources("", text)),
                [pair("dev-dependencies", "rand")],
                "{text}"
            );
        }
        for text in ["rand::random()", "use rand;", "x(rand)", "(rand", "rand"] {
            assert_eq!(findings(manifest, &sources("", text)), [], "{text}");
        }
    }

    #[test]
    fn a_build_dependency_is_looked_for_in_the_build_script_only() {
        let manifest = "[package]\nname = \"p\"\n[build-dependencies]\ncc = \"1\"\n";
        assert_eq!(
            findings(manifest, &sources("", "cc::Build::new();")),
            [pair("build-dependencies", "cc")]
        );
        assert_eq!(findings(manifest, &sources("cc::Build::new();", "")), []);
    }

    #[test]
    fn a_dev_dependency_is_not_satisfied_by_the_build_script() {
        let manifest = "[package]\nname = \"p\"\n[dev-dependencies]\ncc = \"1\"\n";
        assert_eq!(
            findings(manifest, &sources("cc::Build::new();", "")),
            [pair("dev-dependencies", "cc")]
        );
    }

    #[test]
    fn a_virtual_manifest_has_no_package_and_no_findings() {
        let manifest = "[workspace]\nmembers = []\n[dev-dependencies]\nx = \"1\"\n";
        assert_eq!(findings(manifest, &sources("", "")), []);
    }

    #[test]
    fn a_manifest_that_is_not_toml_is_an_error_naming_the_file() {
        let error =
            unused_in_package("crates/x/Cargo.toml", "[package", &sources("", "")).unwrap_err();
        assert!(matches!(error, Error::Manifest { .. }));
        assert!(error.to_string().starts_with("crates/x/Cargo.toml: "));
    }

    #[test]
    fn the_build_script_is_build_rs_unless_the_manifest_says_otherwise() {
        let root = std::env::temp_dir().join(format!("table-deps-{}", std::process::id()));
        fs::create_dir_all(&root).unwrap();
        let parse = |text: &str| text.parse::<toml::Table>().unwrap();
        // No `build` key and no build.rs: no script.
        assert_eq!(
            build_script_path(&parse("[package]\nname = \"p\""), &root),
            None
        );
        // build.rs is picked up by its name once it exists.
        fs::write(root.join("build.rs"), "").unwrap();
        assert_eq!(
            build_script_path(&parse("[package]\nname = \"p\""), &root),
            Some(root.join("build.rs"))
        );
        // `build = "path"` names it, `build = false` switches it off.
        assert_eq!(
            build_script_path(&parse("[package]\nbuild = \"gen/b.rs\""), &root),
            Some(root.join("gen/b.rs"))
        );
        assert_eq!(
            build_script_path(&parse("[package]\nbuild = false"), &root),
            None
        );
        fs::remove_dir_all(&root).unwrap();
    }

    /// A scratch checkout laid out as `files` says, for the test called `name`.
    fn checkout(name: &str, files: &[(&str, &str)]) -> PathBuf {
        let root = std::env::temp_dir().join(format!("table-deps-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        for (path, text) in files {
            let path = root.join(path);
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(path, text).unwrap();
        }
        root
    }

    #[test]
    fn a_checkout_is_read_member_by_member_with_tests_and_target_folders_handled() {
        let root = checkout(
            "walk",
            &[
                ("Cargo.toml", "[workspace]\nmembers = [\"crates/*\"]\n"),
                (
                    "crates/a/Cargo.toml",
                    "[package]\nname = \"a\"\n[dev-dependencies]\nproptest = \"1\"\ninsta = \"1\"\n\
                     serde_json = \"1\"\n[build-dependencies]\ncc = \"1\"\n",
                ),
                ("crates/a/src/lib.rs", "pub fn f() {}"),
                ("crates/a/tests/it.rs", "use proptest::prelude::*;"),
                (
                    "crates/a/src/deep/er/mod_.rs",
                    "fn g() { serde_json::json!(1); }",
                ),
                // A file below a `target` folder is build output, not source.
                ("crates/a/target/gen.rs", "use insta::x;"),
                ("crates/a/build.rs", "fn main() { cc::Build::new(); }"),
                (
                    "crates/b/Cargo.toml",
                    "[package]\nname = \"b\"\n[dev-dependencies]\ninsta = \"1\"\n",
                ),
                ("crates/b/src/lib.rs", ""),
            ],
        );
        let found = check_table_dependencies(&root).unwrap();
        let got: Vec<(String, String)> = found
            .into_iter()
            .map(|found| (found.unused.package, found.unused.dependency))
            .collect();
        assert_eq!(
            got,
            [
                (format!("{TABLES_SCOPE}a"), "insta".to_owned()),
                (format!("{TABLES_SCOPE}b"), "insta".to_owned()),
            ]
        );
        fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn an_unreadable_member_is_an_error_not_a_pass() {
        let root = checkout(
            "missing",
            &[("Cargo.toml", "[workspace]\nmembers = [\"crates/gone\"]\n")],
        );
        let error = check_table_dependencies(&root).unwrap_err();
        assert!(matches!(error, Error::Io { .. }), "{error}");
        fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn the_allowlist_excuses_a_table_finding_and_calls_a_needless_entry_stale() {
        let found = unused_in_package(
            "Cargo.toml",
            "[package]\nname = \"p\"\n[dev-dependencies]\nlib-renamed = \"1\"\n",
            &sources("", ""),
        )
        .unwrap();
        let found: Vec<Unused> = found.into_iter().map(|found| found.unused).collect();
        let listed = "[[allow_tables]]\npackage = \"p\"\ndependencies = [\"lib-renamed\"]\n\
                      reason = \"its library name differs from its package name\"\n";
        let allowlist = Allowlist::parse(listed).unwrap();
        assert_eq!(
            judge(&found, &allowlist, Scope::Tables),
            Verdict {
                unlisted: Vec::new(),
                stale: Vec::new()
            }
        );
        // Once the dependency is used the same entry is stale.
        assert_eq!(
            judge(&[], &allowlist, Scope::Tables).stale,
            [Unused {
                package: tables_package("p"),
                dependency: "lib-renamed".into()
            }]
        );
        // And an unlisted finding is reported.
        let none = Allowlist::parse("").unwrap();
        assert_eq!(judge(&found, &none, Scope::Tables).unlisted, found);
    }

    #[test]
    fn an_entry_for_one_check_is_never_stale_for_another() {
        let listed = "[[allow]]\npackage = \"p\"\ndependencies = [\"a\"]\nreason = \"r\"\n\
                      [[allow_tables]]\npackage = \"p\"\ndependencies = [\"b\"]\nreason = \"r\"\n\
                      [[allow_workspace]]\ndependencies = [\"c\"]\nreason = \"r\"\n";
        let allowlist = Allowlist::parse(listed).unwrap();
        let names = |scope| -> Vec<String> {
            judge(&[], &allowlist, scope)
                .stale
                .into_iter()
                .map(|stale| stale.dependency)
                .collect()
        };
        assert_eq!(names(Scope::Crates), ["a"]);
        assert_eq!(names(Scope::Tables), ["b"]);
        assert_eq!(names(Scope::Workspace), ["c"]);
    }

    /// This repository, against the shipped allow-list: every dev, build and target dependency
    /// that no source names is excused with a reason, and every excuse is still needed.
    #[test]
    fn this_workspace_matches_its_allowlist() {
        let root_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
        let found: Vec<Unused> = check_table_dependencies(&root_dir)
            .unwrap()
            .into_iter()
            .map(|found| found.unused)
            .collect();
        let allowlist = Allowlist::parse(crate::unused_deps::ALLOWLIST).unwrap();
        assert_eq!(
            judge(&found, &allowlist, Scope::Tables),
            Verdict {
                unlisted: Vec::new(),
                stale: Vec::new()
            }
        );
    }
}
