//! The unused-dependency checks of the gate.
//!
//! Two checks live here, both about `Cargo.toml` files rather than compiled code:
//!
//! - **Crate dependencies.** `cargo machete` lists, per package, the dependencies whose names it
//!   cannot find in the package's sources (it is textual, fast and imprecise). A dependency
//!   declared ahead of the task that will use it is listed in `xtask/unused-deps.toml` with a
//!   reason. The list is exact in both directions: a finding that is not listed fails the gate,
//!   and so does a listed entry that `cargo machete` no longer reports (the dependency is now
//!   used, or gone), so the list cannot rot.
//! - **Workspace dependencies.** `cargo machete` does not look at the root
//!   `[workspace.dependencies]`, so a registry dependency declared there that no member uses
//!   (`name.workspace = true`, or `name = { workspace = true }`) is found here. Path
//!   dependencies on the workspace's own crates are not checked: scaffolded crates are declared
//!   ahead of their users, and `cargo machete` already covers the members that name them.
//!
//! # What is covered
//!
//! `cargo machete` 0.9.2 analyses only a package's top-level `[dependencies]` table (its source
//! has a `TODO extend to dev dependencies + build dependencies`). Checked by running it: an
//! unused dependency in `[dev-dependencies]`, in `[build-dependencies]` or in a
//! `[target.'cfg(..)'.dependencies]` table is not reported. The `table-deps` step
//! ([`crate::table_deps`]) covers those tables with its own textual search, and its findings use
//! the same allow-list under `[[allow_tables]]` ([`TABLES_SCOPE`]). The workspace check still
//! counts an inheritance from any of those tables as a use of the root entry, which is correct
//! for it: it asks whether the root entry is used by someone, and the `table-deps` step asks
//! whether that someone uses it.
//!
//! # No silent ignores
//!
//! `cargo machete` can be told to skip a dependency with `ignored = [...]` under
//! `[package.metadata.cargo-machete]` in a crate manifest or `[workspace.metadata.cargo-machete]`
//! in the root, and its advice text recommends that. Such a list has no reason and no entry in
//! `xtask/unused-deps.toml`, which would bypass the rule that an early dependency is listed with
//! its reason. The workspace check therefore fails when any manifest it reads carries one
//! (see [`Error::MacheteIgnore`]).
//!
//! # Workspace members
//!
//! The root manifest's `members` and `exclude` lists are read with a subset of cargo's rules:
//! literal paths, and a trailing `/*` in `members` standing for every subfolder that has a
//! `Cargo.toml`. A pattern with any other glob character (`*`, `?`, `[`) is an
//! [`Error::UnsupportedPattern`], never a guess. `exclude` entries are literal folders; a member
//! whose folder is, or is below, an excluded folder is left out, as in cargo. The root manifest's
//! own `[package]`, if it has one, counts as a user too.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use serde::Deserialize;

/// The allow-list, compiled in so that a checkout's `xtask` binary always carries its own copy.
pub const ALLOWLIST: &str = include_str!("../unused-deps.toml");

/// One dependency of one package that is declared but not used.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Unused {
    /// The package name, as `cargo machete` prints it.
    pub package: String,
    /// The dependency's name in the package's manifest.
    pub dependency: String,
}

/// Why the unused-dependency checks could not run to a verdict.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// `cargo machete` printed something this module does not understand.
    #[error("cannot read the output of `cargo machete`: {0}")]
    Machete(String),
    /// `cargo machete` ended with a status that is neither "nothing unused" (0) nor "findings"
    /// (1): it failed (2), or a signal ended it (no code).
    #[error("cargo machete failed (exit status {0:?})")]
    MacheteStatus(Option<i32>),
    /// A manifest silences `cargo machete` with its own `ignored` list.
    #[error(
        "{path}: [{table}.metadata.cargo-machete] `ignored` silences `cargo machete` without a \
         reason; remove it and, if the dependency is declared ahead of a planned task, list it \
         with the reason in xtask/unused-deps.toml"
    )]
    MacheteIgnore {
        /// The manifest that carries the list.
        path: String,
        /// `package` or `workspace`: the table the list is under.
        table: &'static str,
    },
    /// A `members` or `exclude` entry of the root manifest that this module cannot expand.
    #[error(
        "Cargo.toml: workspace {list} entry {pattern:?} is not supported by the workspace-deps \
         check (only literal paths, and a trailing `/*` in `members`)"
    )]
    UnsupportedPattern {
        /// `members` or `exclude`.
        list: &'static str,
        /// The entry as written.
        pattern: String,
    },
    /// The allow-list is not valid TOML of the expected shape.
    #[error("xtask/unused-deps.toml: {0}")]
    AllowlistSyntax(#[from] toml::de::Error),
    /// The allow-list has an entry that breaks one of its rules.
    #[error("xtask/unused-deps.toml: {0}")]
    AllowlistEntry(String),
    /// A manifest is not valid TOML.
    #[error("{path}: {source}")]
    Manifest {
        /// The manifest that failed to parse.
        path: String,
        /// The parser's complaint.
        source: toml::de::Error,
    },
    /// A manifest or folder could not be read.
    #[error("{path}: {source}")]
    Io {
        /// What was being read.
        path: String,
        /// The operating-system error.
        source: std::io::Error,
    },
}

/// Parses the report `cargo machete` prints when it exits with status 1.
///
/// The report is a header sentence, then for each package a line `name -- ./path/Cargo.toml:`
/// followed by one tab-indented line per unused dependency, then advice text. A tab-indented
/// line outside a package, or a report without any dependency line, is an error rather than "no
/// findings": a format change must not turn the check into a silent pass.
pub fn parse_machete(report: &str) -> Result<Vec<Unused>, Error> {
    let mut found = Vec::new();
    let mut package: Option<&str> = None;
    for line in report.lines() {
        if let Some(dependency) = line.strip_prefix('\t') {
            let package = package.ok_or_else(|| {
                Error::Machete(format!("dependency {dependency:?} before any package"))
            })?;
            found.push(Unused {
                package: package.to_owned(),
                dependency: dependency.trim().to_owned(),
            });
        } else if let Some(header) = line.strip_suffix(':')
            && let Some((name, manifest)) = header.split_once(" -- ")
            && manifest.ends_with("Cargo.toml")
        {
            package = Some(name);
        }
    }
    if found.is_empty() {
        return Err(Error::Machete(
            "no unused dependency in a failing report".into(),
        ));
    }
    found.sort();
    Ok(found)
}

/// Reads the outcome of a `cargo machete` run: its exit `code` (`None` when a signal ended it)
/// and its standard output.
///
/// Status 0 is "nothing unused" and is the only status that yields no findings; status 1 is a
/// report to parse; anything else (2 is machete's own failure) is an [`Error::MacheteStatus`],
/// so a crash cannot pass as a clean run.
pub fn machete_findings(code: Option<i32>, stdout: &str) -> Result<Vec<Unused>, Error> {
    match code {
        Some(0) => Ok(Vec::new()),
        Some(1) => parse_machete(stdout),
        status => Err(Error::MacheteStatus(status)),
    }
}

/// The `package` an allow-list pair carries when it is about the root `[workspace.dependencies]`
/// rather than one crate. It cannot clash with a package name.
pub const WORKSPACE_SCOPE: &str = "[workspace.dependencies]";

/// The prefix of the `package` of an allow-list pair about a crate's dev, build or target-specific
/// dependency tables (`[[allow_tables]]`) rather than its top-level `[dependencies]` (`[[allow]]`).
/// It cannot clash with a package name, and keeps the two lists apart: `cargo machete` never
/// reports the former, so its step must not call such an entry stale.
pub const TABLES_SCOPE: &str = "[dev, build and target dependencies] ";

/// The package an `[[allow_tables]]` or table-check pair carries for crate `package`.
pub fn tables_package(package: &str) -> String {
    format!("{TABLES_SCOPE}{package}")
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct AllowFile {
    #[serde(default)]
    allow: Vec<AllowEntry>,
    #[serde(default)]
    allow_workspace: Vec<WorkspaceEntry>,
    #[serde(default)]
    allow_tables: Vec<AllowEntry>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct AllowEntry {
    package: String,
    dependencies: Vec<String>,
    reason: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct WorkspaceEntry {
    dependencies: Vec<String>,
    reason: String,
}

/// The dependencies that are knowingly declared ahead of their first use, each with a reason.
#[derive(Debug, Default)]
pub struct Allowlist {
    entries: Vec<(Unused, String)>,
}

impl Allowlist {
    /// Parses `xtask/unused-deps.toml`: `[[allow]]` tables (a crate's dependencies) with a
    /// `package`, and `[[allow_workspace]]` tables (root `[workspace.dependencies]`), each with a
    /// non-empty list of `dependencies` and a non-blank `reason`. Duplicate pairs are an error.
    pub fn parse(text: &str) -> Result<Self, Error> {
        let file: AllowFile = toml::from_str(text)?;
        let workspace = file.allow_workspace.into_iter().map(|entry| AllowEntry {
            package: WORKSPACE_SCOPE.to_owned(),
            dependencies: entry.dependencies,
            reason: entry.reason,
        });
        let tables = file.allow_tables.into_iter().map(|entry| AllowEntry {
            package: tables_package(&entry.package),
            ..entry
        });
        let mut entries: Vec<(Unused, String)> = Vec::new();
        for entry in file.allow.into_iter().chain(workspace).chain(tables) {
            if entry.reason.trim().is_empty() {
                return Err(Error::AllowlistEntry(format!(
                    "{}: every entry needs a reason",
                    entry.package
                )));
            }
            if entry.dependencies.is_empty() {
                return Err(Error::AllowlistEntry(format!(
                    "{}: an entry must list at least one dependency",
                    entry.package
                )));
            }
            for dependency in entry.dependencies {
                let unused = Unused {
                    package: entry.package.clone(),
                    dependency,
                };
                if entries.iter().any(|(listed, _)| *listed == unused) {
                    return Err(Error::AllowlistEntry(format!(
                        "{} / {} is listed twice",
                        unused.package, unused.dependency
                    )));
                }
                entries.push((unused, entry.reason.clone()));
            }
        }
        Ok(Self { entries })
    }

    /// Every listed (package, dependency) pair, in file order.
    pub fn pairs(&self) -> impl Iterator<Item = &Unused> {
        self.entries.iter().map(|(unused, _)| unused)
    }

    /// The reason a pair is listed, if it is.
    pub fn reason(&self, unused: &Unused) -> Option<&str> {
        self.entries
            .iter()
            .find(|(listed, _)| listed == unused)
            .map(|(_, reason)| reason.as_str())
    }
}

/// Which findings an allow-list comparison is about.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scope {
    /// Crate dependencies, as `cargo machete` reports them.
    Crates,
    /// The root `[workspace.dependencies]`.
    Workspace,
    /// The dev, build and target-specific dependency tables of the crates.
    Tables,
}

impl Scope {
    /// The scope an allow-list pair belongs to, from its package.
    fn of(unused: &Unused) -> Self {
        if unused.package == WORKSPACE_SCOPE {
            Self::Workspace
        } else if unused.package.starts_with(TABLES_SCOPE) {
            Self::Tables
        } else {
            Self::Crates
        }
    }

    fn covers(self, unused: &Unused) -> bool {
        Self::of(unused) == self
    }
}

/// The outcome of comparing what a check found with the allow-list.
#[derive(Debug, PartialEq, Eq)]
pub struct Verdict {
    /// Found but not listed: remove the dependency, or list it with a reason.
    pub unlisted: Vec<Unused>,
    /// Listed but not found: delete the entry.
    pub stale: Vec<Unused>,
}

impl Verdict {
    /// Whether the gate passes.
    pub fn is_clean(&self) -> bool {
        self.unlisted.is_empty() && self.stale.is_empty()
    }
}

/// Compares the findings of one `scope` with the allow-list entries of that scope, in both
/// directions.
pub fn judge(found: &[Unused], allowlist: &Allowlist, scope: Scope) -> Verdict {
    let unlisted = found
        .iter()
        .filter(|unused| allowlist.reason(unused).is_none())
        .cloned()
        .collect();
    let stale = allowlist
        .pairs()
        .filter(|listed| scope.covers(listed) && !found.contains(listed))
        .cloned()
        .collect();
    Verdict { unlisted, stale }
}

/// The dependency tables of a manifest or of one `[target.*]` table.
const DEPENDENCY_TABLES: [&str; 3] = ["dependencies", "dev-dependencies", "build-dependencies"];

pub(crate) fn parse_manifest(path: &str, text: &str) -> Result<toml::Table, Error> {
    text.parse().map_err(|source| Error::Manifest {
        path: path.to_owned(),
        source,
    })
}

/// Names of the registry dependencies (those with a version) in the root `[workspace.dependencies]`.
fn registry_workspace_dependencies(root: &toml::Table) -> BTreeSet<String> {
    let Some(table) = root
        .get("workspace")
        .and_then(|workspace| workspace.get("dependencies"))
        .and_then(toml::Value::as_table)
    else {
        return BTreeSet::new();
    };
    table
        .iter()
        .filter(|(_, spec)| match spec {
            toml::Value::String(_) => true,
            toml::Value::Table(spec) => spec.contains_key("version"),
            _ => false,
        })
        .map(|(name, _)| name.clone())
        .collect()
}

/// Adds to `used` every dependency the manifest inherits from the workspace
/// (`name.workspace = true`).
fn collect_inherited(manifest: &toml::Table, used: &mut BTreeSet<String>) {
    let mut tables: Vec<&toml::Table> = vec![manifest];
    if let Some(targets) = manifest.get("target").and_then(toml::Value::as_table) {
        tables.extend(targets.values().filter_map(toml::Value::as_table));
    }
    for table in tables {
        for kind in DEPENDENCY_TABLES {
            let Some(dependencies) = table.get(kind).and_then(toml::Value::as_table) else {
                continue;
            };
            for (name, spec) in dependencies {
                let inherits = spec
                    .get("workspace")
                    .and_then(toml::Value::as_bool)
                    .unwrap_or(false);
                if inherits {
                    used.insert(name.clone());
                }
            }
        }
    }
}

/// Fails if the manifest silences `cargo machete` with an `ignored` list, in
/// `[package.metadata.cargo-machete]` or `[workspace.metadata.cargo-machete]`.
fn reject_machete_ignores(path: &str, manifest: &toml::Table) -> Result<(), Error> {
    for table in ["package", "workspace"] {
        let ignored = manifest
            .get(table)
            .and_then(|table| table.get("metadata"))
            .and_then(|metadata| metadata.get("cargo-machete"))
            .and_then(|machete| machete.get("ignored"));
        if ignored.is_some() {
            return Err(Error::MacheteIgnore {
                path: path.to_owned(),
                table,
            });
        }
    }
    Ok(())
}

/// The registry dependencies in the root `[workspace.dependencies]` that no member inherits,
/// sorted by name. `members` are the member manifests as `(path, text)`. The root manifest's own
/// dependency tables (if it has a `[package]`) count as a user too. A manifest with a
/// `cargo machete` `ignored` list is an error.
pub fn unused_workspace_dependencies(
    root: &str,
    members: &[(String, String)],
) -> Result<Vec<Unused>, Error> {
    let root = parse_manifest("Cargo.toml", root)?;
    reject_machete_ignores("Cargo.toml", &root)?;
    let declared = registry_workspace_dependencies(&root);
    let mut used = BTreeSet::new();
    collect_inherited(&root, &mut used);
    for (path, text) in members {
        let manifest = parse_manifest(path, text)?;
        reject_machete_ignores(path, &manifest)?;
        collect_inherited(&manifest, &mut used);
    }
    Ok(declared
        .difference(&used)
        .map(|dependency| Unused {
            package: WORKSPACE_SCOPE.to_owned(),
            dependency: dependency.clone(),
        })
        .collect())
}

/// Whether a `members` or `exclude` entry uses glob syntax this module does not expand.
fn has_glob(pattern: &str) -> bool {
    pattern.contains(['*', '?', '['])
}

/// The string entries of `[workspace] <list>` in the root manifest.
fn workspace_list<'a>(table: &'a toml::Table, list: &str) -> Vec<&'a str> {
    table
        .get("workspace")
        .and_then(|workspace| workspace.get(list))
        .and_then(toml::Value::as_array)
        .map(|entries| entries.iter().filter_map(toml::Value::as_str).collect())
        .unwrap_or_default()
}

/// The manifests of the workspace members named by the root manifest's `members` list, relative
/// to `root_dir`, minus the members under an `exclude` folder. See the module docs for the
/// supported subset of cargo's patterns; anything else is an [`Error::UnsupportedPattern`].
/// Unreadable folder entries are errors too, so the check cannot pass over a member it failed to
/// see.
pub fn member_manifests(root_dir: &Path, root: &str) -> Result<Vec<PathBuf>, Error> {
    let table = parse_manifest("Cargo.toml", root)?;
    let excluded = workspace_list(&table, "exclude");
    if let Some(pattern) = excluded.iter().find(|pattern| has_glob(pattern)) {
        return Err(Error::UnsupportedPattern {
            list: "exclude",
            pattern: (*pattern).to_owned(),
        });
    }
    let mut manifests = Vec::new();
    for pattern in workspace_list(&table, "members") {
        if let Some(parent) = pattern.strip_suffix("/*")
            && !has_glob(parent)
        {
            let folder = root_dir.join(parent);
            let io = |source| Error::Io {
                path: folder.display().to_string(),
                source,
            };
            let mut found = Vec::new();
            for entry in fs::read_dir(&folder).map_err(io)? {
                let manifest = entry.map_err(io)?.path().join("Cargo.toml");
                if manifest.is_file() {
                    found.push(manifest);
                }
            }
            found.sort();
            manifests.extend(found);
        } else if has_glob(pattern) {
            return Err(Error::UnsupportedPattern {
                list: "members",
                pattern: pattern.to_owned(),
            });
        } else {
            manifests.push(root_dir.join(pattern).join("Cargo.toml"));
        }
    }
    manifests.retain(|manifest| {
        !excluded
            .iter()
            .any(|folder| manifest.starts_with(root_dir.join(folder)))
    });
    Ok(manifests)
}

/// Runs the workspace-dependency check against the checkout at `root_dir`.
pub fn check_workspace_dependencies(root_dir: &Path) -> Result<Vec<Unused>, Error> {
    let read = |path: &Path| {
        fs::read_to_string(path).map_err(|source| Error::Io {
            path: path.display().to_string(),
            source,
        })
    };
    let root = read(&root_dir.join("Cargo.toml"))?;
    let members = member_manifests(root_dir, &root)?
        .into_iter()
        .map(|path| read(&path).map(|text| (path.display().to_string(), text)))
        .collect::<Result<Vec<_>, _>>()?;
    unused_workspace_dependencies(&root, &members)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unused(package: &str, dependency: &str) -> Unused {
        Unused {
            package: package.to_owned(),
            dependency: dependency.to_owned(),
        }
    }

    /// A report in the shape `cargo machete` 0.9 prints (header, blocks, advice).
    const REPORT: &str = "Analyzing dependencies of crates in this directory...\n\
        cargo-machete found the following unused dependencies in this directory:\n\
        starquake-gpu -- ./crates/gpu/Cargo.toml:\n\
        \tstarquake-memory\n\
        \ttracing\n\
        starquake -- ./apps/starquake/Cargo.toml:\n\
        \tstarquake-core\n\
        \n\
        If you believe cargo-machete has detected an unused dependency incorrectly,\n\
        you can add the dependency to the list of dependencies to ignore in the\n\
        `[package.metadata.cargo-machete]` section of the appropriate Cargo.toml.\n\
        For example:\n\
        \n\
        [package.metadata.cargo-machete]\n\
        ignored = [\"prost\"]\n\
        \n\
        Done!\n";

    #[test]
    fn machete_reports_are_read_per_package() {
        assert_eq!(
            parse_machete(REPORT).unwrap(),
            [
                unused("starquake", "starquake-core"),
                unused("starquake-gpu", "starquake-memory"),
                unused("starquake-gpu", "tracing"),
            ]
        );
    }

    #[test]
    fn a_failing_report_without_findings_is_an_error_not_a_pass() {
        for report in [
            "",
            "Done!\n",
            "cargo-machete found the following unused dependencies:\n",
        ] {
            let error = parse_machete(report).unwrap_err();
            assert_eq!(
                error.to_string(),
                "cannot read the output of `cargo machete`: no unused dependency in a failing report"
            );
        }
    }

    #[test]
    fn a_dependency_line_before_any_package_is_an_error() {
        let error = parse_machete("\ttracing\n").unwrap_err();
        assert_eq!(
            error.to_string(),
            "cannot read the output of `cargo machete`: dependency \"tracing\" before any package"
        );
    }

    #[test]
    fn advice_text_that_looks_like_a_header_does_not_start_a_package() {
        // `x -- y:` has the shape of a package header but its path is not a manifest.
        let report = "starquake -- ./apps/starquake/Cargo.toml:\n\
            \ttracing\n\
            see the notes -- below:\n\
            \tserde\n";
        assert_eq!(
            parse_machete(report).unwrap(),
            [unused("starquake", "serde"), unused("starquake", "tracing")]
        );
        // Before any real package it is an error, not a package named after the advice.
        let error = parse_machete("see the notes -- below:\n\tserde\n").unwrap_err();
        assert_eq!(
            error.to_string(),
            "cannot read the output of `cargo machete`: dependency \"serde\" before any package"
        );
    }

    #[test]
    fn only_exit_status_0_means_machete_found_nothing() {
        assert_eq!(machete_findings(Some(0), "").unwrap(), []);
        assert_eq!(
            machete_findings(Some(1), REPORT).unwrap(),
            parse_machete(REPORT).unwrap()
        );
        // Whatever it printed, a crash or a signal is a failure, not a clean run.
        for (code, stdout) in [
            (Some(2), ""),
            (Some(2), REPORT),
            (None, ""),
            (Some(101), ""),
        ] {
            let error = machete_findings(code, stdout).unwrap_err();
            assert!(matches!(error, Error::MacheteStatus(status) if status == code));
            assert_eq!(
                error.to_string(),
                format!("cargo machete failed (exit status {code:?})")
            );
        }
        // Status 1 with a report that has no findings is an error too.
        assert!(matches!(
            machete_findings(Some(1), "Done!\n"),
            Err(Error::Machete(_))
        ));
    }

    const ALLOWLIST_TEXT: &str = r#"
[[allow]]
package = "a"
dependencies = ["x", "y"]
reason = "declared ahead of the first slice"

[[allow]]
package = "b"
dependencies = ["x"]
reason = "same dependency, other package"

[[allow_workspace]]
dependencies = ["x", "w"]
reason = "declared in the root ahead of its first user"
"#;

    #[test]
    fn the_allowlist_keys_entries_by_package_and_dependency() {
        let list = Allowlist::parse(ALLOWLIST_TEXT).unwrap();
        assert_eq!(
            list.pairs().cloned().collect::<Vec<_>>(),
            [
                unused("a", "x"),
                unused("a", "y"),
                unused("b", "x"),
                unused(WORKSPACE_SCOPE, "x"),
                unused(WORKSPACE_SCOPE, "w"),
            ]
        );
        assert_eq!(
            list.reason(&unused("b", "x")),
            Some("same dependency, other package")
        );
        assert_eq!(list.reason(&unused("b", "y")), None);
    }

    #[test]
    fn allowlist_entries_must_carry_a_reason_and_a_dependency_and_be_unique() {
        let cases = [
            (
                "[[allow]]\npackage = \"a\"\ndependencies = [\"x\"]\nreason = \"  \"\n",
                "xtask/unused-deps.toml: a: every entry needs a reason",
            ),
            (
                "[[allow]]\npackage = \"a\"\ndependencies = []\nreason = \"why\"\n",
                "xtask/unused-deps.toml: a: an entry must list at least one dependency",
            ),
            (
                "[[allow]]\npackage = \"a\"\ndependencies = [\"x\", \"x\"]\nreason = \"why\"\n",
                "xtask/unused-deps.toml: a / x is listed twice",
            ),
            (
                "[[allow_workspace]]\ndependencies = [\"x\"]\nreason = \"\"\n",
                "xtask/unused-deps.toml: [workspace.dependencies]: every entry needs a reason",
            ),
            (
                "[[allow_workspace]]\ndependencies = [\"x\"]\nreason = \"a\"\n\
                 [[allow_workspace]]\ndependencies = [\"x\"]\nreason = \"b\"\n",
                "xtask/unused-deps.toml: [workspace.dependencies] / x is listed twice",
            ),
        ];
        for (text, message) in cases {
            assert_eq!(Allowlist::parse(text).unwrap_err().to_string(), message);
        }
    }

    #[test]
    fn allowlist_entries_reject_unknown_and_missing_fields() {
        for text in [
            "[[allow]]\npackage = \"a\"\ndependencies = [\"x\"]\n",
            "[[allow]]\npackage = \"a\"\ndependencies = [\"x\"]\nreason = \"why\"\nnote = 1\n",
            "[[allow]]\ndependencies = [\"x\"]\nreason = \"why\"\n",
            "[[allow_workspace]]\npackage = \"a\"\ndependencies = [\"x\"]\nreason = \"why\"\n",
        ] {
            assert!(
                matches!(Allowlist::parse(text), Err(Error::AllowlistSyntax(_))),
                "{text:?}"
            );
        }
    }

    #[test]
    fn the_shipped_allowlist_is_valid() {
        let list = Allowlist::parse(ALLOWLIST).unwrap();
        assert!(list.pairs().next().is_some());
    }

    #[test]
    fn findings_are_judged_in_both_directions() {
        let list = Allowlist::parse(ALLOWLIST_TEXT).unwrap();
        let found = [unused("a", "x"), unused("a", "z"), unused("b", "x")];
        assert_eq!(
            judge(&found, &list, Scope::Crates),
            Verdict {
                unlisted: vec![unused("a", "z")],
                stale: vec![unused("a", "y")],
            }
        );
        let all_listed = [unused("a", "x"), unused("a", "y"), unused("b", "x")];
        assert!(judge(&all_listed, &list, Scope::Crates).is_clean());
    }

    #[test]
    fn crate_and_workspace_entries_are_judged_separately() {
        let list = Allowlist::parse(ALLOWLIST_TEXT).unwrap();
        // The workspace entries are not stale for a crates run, and vice versa.
        let workspace = [unused(WORKSPACE_SCOPE, "x"), unused(WORKSPACE_SCOPE, "w")];
        assert!(judge(&workspace, &list, Scope::Workspace).is_clean());
        assert_eq!(
            judge(&[unused(WORKSPACE_SCOPE, "x")], &list, Scope::Workspace).stale,
            [unused(WORKSPACE_SCOPE, "w")]
        );
        assert_eq!(
            judge(&[unused(WORKSPACE_SCOPE, "q")], &list, Scope::Workspace).unlisted,
            [unused(WORKSPACE_SCOPE, "q")]
        );
        // A crate dependency that shares a name with a listed workspace one is still unlisted.
        assert_eq!(
            judge(&[unused("c", "x")], &list, Scope::Crates).unlisted,
            [unused("c", "x")]
        );
    }

    const ROOT: &str = r#"
[workspace]
members = ["crates/*"]

[workspace.dependencies]
starquake-a = { path = "crates/a" }
used = "1.0.0"
dev_only = "2.0.0"
via_target = { version = "3.0.0", default-features = false }
idle = { version = "4.0.0", features = ["f"] }
also_idle = "5.0.0"
build_only = "6.0.0"
via_target_build = "7.0.0"
"#;

    fn unused_dependency(name: &str) -> Unused {
        unused(WORKSPACE_SCOPE, name)
    }

    fn member(text: &str) -> (String, String) {
        ("member/Cargo.toml".to_owned(), text.to_owned())
    }

    #[test]
    fn workspace_dependencies_nobody_inherits_are_found() {
        let members = [
            member("[dependencies]\nused.workspace = true\n"),
            member("[dev-dependencies]\ndev_only = { workspace = true }\n"),
            member("[target.'cfg(unix)'.dependencies]\nvia_target.workspace = true\n"),
            member("[build-dependencies]\nbuild_only.workspace = true\n"),
            member(
                "[target.'cfg(unix)'.build-dependencies]\nvia_target_build = { workspace = true }\n",
            ),
        ];
        assert_eq!(
            unused_workspace_dependencies(ROOT, &members).unwrap(),
            [
                unused(WORKSPACE_SCOPE, "also_idle"),
                unused(WORKSPACE_SCOPE, "idle")
            ]
        );
    }

    #[test]
    fn a_root_package_that_inherits_counts_as_a_user() {
        let root =
            format!("[package]\nname = \"r\"\n[dependencies]\nidle.workspace = true\n{ROOT}");
        let found = unused_workspace_dependencies(&root, &[]).unwrap();
        assert!(!found.contains(&unused_dependency("idle")), "{found:?}");
        assert!(found.contains(&unused_dependency("also_idle")), "{found:?}");
    }

    #[test]
    fn a_cargo_machete_ignore_list_is_refused_with_the_manifest_and_table_named() {
        let message = |path: &str, table: &str| {
            format!(
                "{path}: [{table}.metadata.cargo-machete] `ignored` silences `cargo machete` \
                 without a reason; remove it and, if the dependency is declared ahead of a \
                 planned task, list it with the reason in xtask/unused-deps.toml"
            )
        };
        let in_member = member("[package.metadata.cargo-machete]\nignored = [\"lz4_flex\"]\n");
        let error = unused_workspace_dependencies(ROOT, &[in_member]).unwrap_err();
        assert!(matches!(error, Error::MacheteIgnore { .. }));
        assert_eq!(error.to_string(), message("member/Cargo.toml", "package"));

        let root = format!("{ROOT}\n[workspace.metadata.cargo-machete]\nignored = []\n");
        let error = unused_workspace_dependencies(&root, &[]).unwrap_err();
        assert_eq!(error.to_string(), message("Cargo.toml", "workspace"));

        // Other metadata, and a machete table without a list, silence nothing.
        let quiet = member("[package.metadata.docs]\nx = 1\n[package.metadata.cargo-machete]\n");
        assert!(unused_workspace_dependencies(ROOT, &[quiet]).is_ok());
    }

    #[test]
    fn a_member_that_pins_its_own_version_does_not_use_the_workspace_entry() {
        let members = [member("[dependencies]\nidle = \"4.0.0\"\n")];
        assert!(
            unused_workspace_dependencies(ROOT, &members)
                .unwrap()
                .contains(&unused(WORKSPACE_SCOPE, "idle"))
        );
    }

    #[test]
    fn path_dependencies_on_workspace_crates_are_not_checked() {
        let unused = unused_workspace_dependencies(ROOT, &[]).unwrap();
        assert!(
            !unused.contains(&unused_dependency("starquake-a")),
            "{unused:?}"
        );
    }

    #[test]
    fn a_broken_manifest_is_an_error_naming_the_file() {
        let error = unused_workspace_dependencies(ROOT, &[member("[dependencies")]).unwrap_err();
        assert!(matches!(&error, Error::Manifest { path, .. } if path == "member/Cargo.toml"));
    }

    fn repository() -> (PathBuf, String) {
        let root_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
        let root = fs::read_to_string(root_dir.join("Cargo.toml")).unwrap();
        (root_dir, root)
    }

    #[test]
    fn excluded_folders_are_not_members() {
        let (root_dir, _) = repository();
        let all = member_manifests(&root_dir, "[workspace]\nmembers = [\"crates/*\"]\n").unwrap();
        let first = all.first().expect("the repository has crates").clone();
        let folder = first.parent().unwrap().strip_prefix(&root_dir).unwrap();
        let text = format!(
            "[workspace]\nmembers = [\"crates/*\"]\nexclude = [{:?}]\n",
            folder.to_str().unwrap()
        );
        let rest = member_manifests(&root_dir, &text).unwrap();
        assert_eq!(rest, all[1..]);
    }

    #[test]
    fn a_literal_member_under_an_excluded_folder_is_left_out() {
        let (root_dir, _) = repository();
        let text = "[workspace]\nmembers = [\"xtask\", \"crates/ipc\"]\nexclude = [\"crates\"]\n";
        assert_eq!(
            member_manifests(&root_dir, text).unwrap(),
            [root_dir.join("xtask/Cargo.toml")]
        );
    }

    #[test]
    fn patterns_beyond_the_supported_subset_are_clear_errors() {
        let (root_dir, _) = repository();
        for (text, list, pattern) in [
            (
                "[workspace]\nmembers = [\"crates/i*\"]\n",
                "members",
                "crates/i*",
            ),
            ("[workspace]\nmembers = [\"*/ipc\"]\n", "members", "*/ipc"),
            (
                "[workspace]\nmembers = [\"cr?tes/*\"]\n",
                "members",
                "cr?tes/*",
            ),
            (
                "[workspace]\nmembers = [\"crates/[i]pc\"]\n",
                "members",
                "crates/[i]pc",
            ),
            (
                "[workspace]\nmembers = [\"xtask\"]\nexclude = [\"crates/*\"]\n",
                "exclude",
                "crates/*",
            ),
        ] {
            let error = member_manifests(&root_dir, text).unwrap_err();
            assert!(matches!(&error, Error::UnsupportedPattern { .. }), "{text}");
            assert_eq!(
                error.to_string(),
                format!(
                    "Cargo.toml: workspace {list} entry {pattern:?} is not supported by the \
                     workspace-deps check (only literal paths, and a trailing `/*` in `members`)"
                )
            );
        }
    }

    /// The repository this test runs in: members are found by their `/*` patterns, and the
    /// workspace entries nobody inherits are exactly the ones the shipped allow-list names.
    #[test]
    fn this_workspace_matches_its_allowlist() {
        let (root_dir, root) = repository();
        let manifests = member_manifests(&root_dir, &root).unwrap();
        for expected in ["xtask", "crates/ipc", "apps/starquake"] {
            assert!(
                manifests
                    .iter()
                    .any(|path| path.ends_with(Path::new(expected).join("Cargo.toml"))),
                "{expected} is not among {manifests:?}"
            );
        }
        let found = check_workspace_dependencies(&root_dir).unwrap();
        let allowlist = Allowlist::parse(ALLOWLIST).unwrap();
        assert_eq!(
            judge(&found, &allowlist, Scope::Workspace),
            Verdict {
                unlisted: Vec::new(),
                stale: Vec::new()
            }
        );
    }
}
