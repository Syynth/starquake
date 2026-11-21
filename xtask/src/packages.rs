//! The gate's `packages` step: every workspace package checked on its own.
//!
//! The workspace-wide `clippy --workspace --all-targets` and `nextest run --workspace` build all
//! members in one cargo invocation, so cargo unifies their features: a dev-dependency that turns
//! on another crate's `test-support` feature turns it on for every member's build. A package
//! whose production code used an item gated behind `cfg(feature = "test-support")` therefore
//! passed the gate while `cargo check -p starquake` failed on `main` (PR #446). The owner's
//! ruling of 2026-11-19 (`docs/decision-log.md`, "Owner rulings on the wave 44 and 45 questions
//! and the first title bring-up", row 14) adds this step.
//!
//! # What runs
//!
//! For each workspace member, in member order, one `cargo clippy -p <package> --locked -- -D
//! warnings` ([`commands`]), each in its own cargo invocation so that no two packages share a
//! feature resolution. Cargo's resolver 3 (the root manifest's `resolver = "3"`, whose feature
//! resolution is resolver 2's) resolves features for the selected package only, so each
//! invocation builds the package as a user of it would: its default features, its default
//! targets (library and binaries, not tests, examples or benches) and no dev-dependencies.
//!
//! Clippy runs the whole of `cargo check` (it is the compiler with extra lints), so a separate
//! `cargo check -p` would compile every package twice for nothing. `-D warnings` also catches the
//! dead-code warnings an item used only by tests gives when the package is built without them.
//! All invocations share the checkout's target folder, so a dependency compiled with the same
//! features is reused.
//!
//! # Not covered
//!
//! - A package's test, example and bench targets on their own (`-p <package> --all-targets`):
//!   the workspace-wide clippy and nextest build those, with unified features.
//! - Feature sets other than the default: `--no-default-features` (the app without `wgpu`, say)
//!   and each optional feature alone.
//! - Other targets: the Linux-target clippy in CLAUDE.md is run by hand, outside the gate.
//!
//! # Members
//!
//! The members come from the root manifest's `members` and `exclude` lists, expanded by the
//! same rules as the dependency checks ([`crate::unused_deps::member_manifests`]), and each
//! member's name is its manifest's `[package] name`. A crate added under a `members` folder is
//! therefore picked up with no change here.

use std::path::Path;

use crate::unused_deps;
use crate::walk;

/// Why the package list could not be read.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The members could not be listed or a manifest could not be read or parsed.
    #[error(transparent)]
    Members(#[from] unused_deps::Error),
    /// The root manifest's member lists name no package, so there is nothing to check.
    #[error("Cargo.toml: the workspace has no members, so no package would be checked")]
    NoPackages,
    /// A member manifest has no `[package] name` string.
    #[error("{path}: a workspace member's manifest has no [package] name")]
    NoName {
        /// The member manifest.
        path: String,
    },
}

impl From<walk::Unreadable> for Error {
    fn from(unreadable: walk::Unreadable) -> Self {
        Self::Members(unreadable.into())
    }
}

/// The package names of the member manifests `members` (path and text), in the given order.
pub fn package_names(members: &[(String, String)]) -> Result<Vec<String>, Error> {
    members
        .iter()
        .map(|(path, text)| {
            let manifest =
                text.parse::<toml::Table>()
                    .map_err(|source| unused_deps::Error::Manifest {
                        path: path.clone(),
                        source,
                    })?;
            manifest
                .get("package")
                .and_then(|package| package.get("name"))
                .and_then(toml::Value::as_str)
                .map(str::to_owned)
                .ok_or_else(|| Error::NoName { path: path.clone() })
        })
        .collect()
}

/// The `cargo` arguments that check each of `packages` on its own: one command per package, in
/// the given order, each selecting exactly that package.
pub fn commands(packages: &[String]) -> Vec<Vec<String>> {
    packages
        .iter()
        .map(|package| {
            ["clippy", "-p", package, "--locked", "--", "-D", "warnings"]
                .map(str::to_owned)
                .to_vec()
        })
        .collect()
}

/// The workspace packages of the checkout at `root_dir`, in member order. An empty list is
/// [`Error::NoPackages`], so the step cannot pass without checking anything.
pub fn workspace_packages(root_dir: &Path) -> Result<Vec<String>, Error> {
    let root = walk::read_text(&root_dir.join("Cargo.toml"))?;
    let members = unused_deps::member_manifests(root_dir, &root)?
        .into_iter()
        .map(|path| walk::read_text(&path).map(|text| (path.display().to_string(), text)))
        .collect::<Result<Vec<_>, _>>()?;
    let packages = package_names(&members)?;
    if packages.is_empty() {
        return Err(Error::NoPackages);
    }
    Ok(packages)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::fs;
    use std::process::Command;

    use super::*;

    fn member(path: &str, text: &str) -> (String, String) {
        (path.to_owned(), text.to_owned())
    }

    fn strings(list: &[&str]) -> Vec<String> {
        list.iter().map(|item| (*item).to_owned()).collect()
    }

    #[test]
    fn each_member_is_named_by_its_package_name_in_member_order() {
        let members = [
            member("b/Cargo.toml", "[package]\nname = \"zed\"\n"),
            member(
                "a/Cargo.toml",
                "[package]\nname = \"alpha-one\"\nversion = \"1.0.0\"\n[dependencies]\nx = \"1\"\n",
            ),
            member("c/Cargo.toml", "[package]\nname = \"mid\"\n"),
        ];
        assert_eq!(
            package_names(&members).unwrap(),
            strings(&["zed", "alpha-one", "mid"])
        );
        assert_eq!(package_names(&[]).unwrap(), Vec::<String>::new());
    }

    #[test]
    fn a_member_without_a_package_name_is_an_error_naming_it() {
        for text in [
            "[workspace]\nmembers = []\n",
            "[package]\nversion = \"1.0.0\"\n",
            "[package]\nname = 3\n",
            "package = \"x\"\n",
        ] {
            let members = [
                member("ok/Cargo.toml", "[package]\nname = \"ok\"\n"),
                member("bad/Cargo.toml", text),
            ];
            let error = package_names(&members).unwrap_err();
            assert!(
                matches!(&error, Error::NoName { path } if path == "bad/Cargo.toml"),
                "{text:?}: {error:?}"
            );
            assert_eq!(
                error.to_string(),
                "bad/Cargo.toml: a workspace member's manifest has no [package] name"
            );
        }
    }

    #[test]
    fn a_member_that_is_not_toml_is_an_error_naming_it() {
        let members = [member("bad/Cargo.toml", "[package\n")];
        let error = package_names(&members).unwrap_err();
        assert!(
            matches!(
                &error,
                Error::Members(unused_deps::Error::Manifest { path, .. }) if path == "bad/Cargo.toml"
            ),
            "{error:?}"
        );
        assert!(error.to_string().starts_with("bad/Cargo.toml: "), "{error}");
    }

    #[test]
    fn every_package_gets_its_own_clippy_run_with_warnings_denied() {
        let packages = strings(&["starquake", "starquake-gpu", "xtask"]);
        let expected: Vec<Vec<String>> = packages
            .iter()
            .map(|package| strings(&["clippy", "-p", package, "--locked", "--", "-D", "warnings"]))
            .collect();
        assert_eq!(commands(&packages), expected);
        // One package per invocation: two `-p` in one command would unify their features again.
        for command in commands(&packages) {
            assert_eq!(command.iter().filter(|arg| *arg == "-p").count(), 1);
            // Default features and default targets only.
            assert!(
                !command
                    .iter()
                    .any(|arg| arg.starts_with("--all") || arg.starts_with("--features")),
                "{command:?}"
            );
        }
    }

    #[test]
    fn no_packages_means_no_commands() {
        assert_eq!(commands(&[]), Vec::<Vec<String>>::new());
    }

    #[test]
    fn a_package_added_under_a_members_folder_is_picked_up() {
        let root = walk::scratch_checkout(
            "packages",
            "added",
            &[
                (
                    "Cargo.toml",
                    "[workspace]\nmembers = [\"apps/*\", \"crates/*\", \"xtask\"]\n\
                     exclude = [\"crates/scripts\"]\n",
                ),
                ("apps/app/Cargo.toml", "[package]\nname = \"the-app\"\n"),
                ("crates/b/Cargo.toml", "[package]\nname = \"crate-b\"\n"),
                (
                    "crates/scripts/Cargo.toml",
                    "[package]\nname = \"skipped\"\n",
                ),
                ("crates/notes/README.md", "a folder without a manifest"),
                ("xtask/Cargo.toml", "[package]\nname = \"xtask\"\n"),
            ],
        );
        assert_eq!(
            workspace_packages(&root).unwrap(),
            strings(&["the-app", "crate-b", "xtask"])
        );
        fs::create_dir_all(root.join("crates/a")).unwrap();
        fs::write(
            root.join("crates/a/Cargo.toml"),
            "[package]\nname = \"crate-a\"\n",
        )
        .unwrap();
        assert_eq!(
            workspace_packages(&root).unwrap(),
            strings(&["the-app", "crate-a", "crate-b", "xtask"])
        );
        fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn a_listed_member_that_is_missing_is_an_error_not_a_skip() {
        let root = walk::scratch_checkout(
            "packages",
            "missing",
            &[("Cargo.toml", "[workspace]\nmembers = [\"crates/gone\"]\n")],
        );
        let error = workspace_packages(&root).unwrap_err();
        assert!(
            matches!(&error, Error::Members(unused_deps::Error::Io { path, .. })
                if path.ends_with("Cargo.toml") && path.contains("gone")),
            "{error:?}"
        );
        fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn a_member_pattern_the_listing_cannot_expand_is_an_error_not_a_guess() {
        let root = walk::scratch_checkout(
            "packages",
            "glob",
            &[
                ("Cargo.toml", "[workspace]\nmembers = [\"crates/a*\"]\n"),
                ("crates/ab/Cargo.toml", "[package]\nname = \"ab\"\n"),
            ],
        );
        let error = workspace_packages(&root).unwrap_err();
        assert!(
            matches!(
                &error,
                Error::Members(unused_deps::Error::UnsupportedPattern { list: "members", pattern })
                    if pattern == "crates/a*"
            ),
            "{error:?}"
        );
        assert_eq!(
            error.to_string(),
            "Cargo.toml: workspace members entry \"crates/a*\" is not supported by the gate's \
             member listing (only literal paths, and a trailing `/*` in `members`)"
        );
        fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn a_workspace_whose_member_lists_name_no_package_is_an_error_not_a_pass() {
        let crate_a = ("crates/a/Cargo.toml", "[package]\nname = \"a\"\n");
        for (name, manifest) in [
            ("none", "[workspace]\nmembers = []\n"),
            (
                "excluded",
                "[workspace]\nmembers = [\"crates/*\"]\nexclude = [\"crates/a\"]\n",
            ),
        ] {
            let root =
                walk::scratch_checkout("packages", name, &[("Cargo.toml", manifest), crate_a]);
            let error = workspace_packages(&root).unwrap_err();
            assert!(matches!(error, Error::NoPackages), "{name}: {error:?}");
            assert_eq!(
                error.to_string(),
                "Cargo.toml: the workspace has no members, so no package would be checked"
            );
            fs::remove_dir_all(&root).unwrap();
        }
        // One step off: the same layout without the exclusion has one package.
        let root = walk::scratch_checkout(
            "packages",
            "one",
            &[
                ("Cargo.toml", "[workspace]\nmembers = [\"crates/*\"]\n"),
                crate_a,
            ],
        );
        assert_eq!(workspace_packages(&root).unwrap(), strings(&["a"]));
        fs::remove_dir_all(&root).unwrap();
    }

    /// The workspace's members as cargo itself lists them, from `cargo metadata --no-deps`, which
    /// lists only the members.
    fn cargo_members(root: &Path) -> BTreeSet<String> {
        let output = Command::new(env!("CARGO"))
            .args([
                "metadata",
                "--no-deps",
                "--format-version",
                "1",
                "--offline",
            ])
            .current_dir(root)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let metadata: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
        metadata["packages"]
            .as_array()
            .unwrap()
            .iter()
            .map(|package| package["name"].as_str().unwrap().to_owned())
            .collect()
    }

    #[test]
    fn this_workspace_checks_exactly_the_members_cargo_knows() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
        let listed = workspace_packages(root).unwrap();
        let unique: BTreeSet<String> = listed.iter().cloned().collect();
        assert_eq!(unique.len(), listed.len(), "a package is listed twice");
        assert_eq!(unique, cargo_members(root));
        // The app, whose build PR #446 found broken, and this tool are among them.
        assert!(unique.contains("starquake"));
        assert!(unique.contains("xtask"));
    }
}
