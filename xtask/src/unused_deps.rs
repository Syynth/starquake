//! The unused-dependency checks of the gate.
//!
//! Two checks live here, both about `Cargo.toml` files rather than compiled code:
//!
//! - **Crate dependencies.** `cargo machete` lists, per package, the dependencies whose names it
//!   cannot find in the package's sources (it is textual, fast and imprecise; it also looks at
//!   `[dev-dependencies]`). A dependency declared ahead of the task that will use it is listed
//!   in `xtask/unused-deps.toml` with a reason. The list is exact in both directions: a finding
//!   that is not listed fails the gate, and so does a listed entry that `cargo machete` no longer
//!   reports (the dependency is now used, or gone), so the list cannot rot.
//! - **Workspace dependencies.** `cargo machete` does not look at the root
//!   `[workspace.dependencies]`, so a registry dependency declared there that no member uses
//!   (`name.workspace = true`, or `name = { workspace = true }`) is found here. Path
//!   dependencies on the workspace's own crates are not checked: scaffolded crates are declared
//!   ahead of their users, and `cargo machete` already covers the members that name them.

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

/// The `package` an allow-list pair carries when it is about the root `[workspace.dependencies]`
/// rather than one crate. It cannot clash with a package name.
pub const WORKSPACE_SCOPE: &str = "[workspace.dependencies]";

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct AllowFile {
    #[serde(default)]
    allow: Vec<AllowEntry>,
    #[serde(default)]
    allow_workspace: Vec<WorkspaceEntry>,
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
        let mut entries: Vec<(Unused, String)> = Vec::new();
        for entry in file.allow.into_iter().chain(workspace) {
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
}

impl Scope {
    fn covers(self, unused: &Unused) -> bool {
        (unused.package == WORKSPACE_SCOPE) == (self == Self::Workspace)
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

fn parse_manifest(path: &str, text: &str) -> Result<toml::Table, Error> {
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

/// The registry dependencies in the root `[workspace.dependencies]` that no member inherits,
/// sorted by name. `members` are the member manifests as `(path, text)`.
pub fn unused_workspace_dependencies(
    root: &str,
    members: &[(String, String)],
) -> Result<Vec<Unused>, Error> {
    let declared = registry_workspace_dependencies(&parse_manifest("Cargo.toml", root)?);
    let mut used = BTreeSet::new();
    for (path, text) in members {
        collect_inherited(&parse_manifest(path, text)?, &mut used);
    }
    Ok(declared
        .difference(&used)
        .map(|dependency| Unused {
            package: WORKSPACE_SCOPE.to_owned(),
            dependency: dependency.clone(),
        })
        .collect())
}

/// The manifests of the workspace members named by the root manifest's `members` list, relative
/// to `root_dir`. A pattern ending in `/*` stands for every subfolder that has a `Cargo.toml`.
pub fn member_manifests(root_dir: &Path, root: &str) -> Result<Vec<PathBuf>, Error> {
    let table = parse_manifest("Cargo.toml", root)?;
    let patterns: Vec<&str> = table
        .get("workspace")
        .and_then(|workspace| workspace.get("members"))
        .and_then(toml::Value::as_array)
        .map(|members| members.iter().filter_map(toml::Value::as_str).collect())
        .unwrap_or_default();
    let mut manifests = Vec::new();
    for pattern in patterns {
        if let Some(parent) = pattern.strip_suffix("/*") {
            let folder = root_dir.join(parent);
            let entries = fs::read_dir(&folder).map_err(|source| Error::Io {
                path: folder.display().to_string(),
                source,
            })?;
            let mut found: Vec<PathBuf> = entries
                .filter_map(Result::ok)
                .map(|entry| entry.path().join("Cargo.toml"))
                .filter(|manifest| manifest.is_file())
                .collect();
            found.sort();
            manifests.extend(found);
        } else {
            manifests.push(root_dir.join(pattern).join("Cargo.toml"));
        }
    }
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

    /// The repository this test runs in: members are found by their `/*` patterns, and the
    /// workspace entries nobody inherits are exactly the ones the shipped allow-list names.
    #[test]
    fn this_workspace_matches_its_allowlist() {
        let root_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
        let root = fs::read_to_string(root_dir.join("Cargo.toml")).unwrap();
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
