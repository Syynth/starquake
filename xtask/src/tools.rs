//! The pinned versions of the cargo plugins the gate runs (`xtask/tool-versions.toml`).
//!
//! The file is compiled in, like the unused-dependency allow-list. Three things use it:
//!
//! - the gate compares each installed plugin's `--version` output with the pin ([`check_output`],
//!   [`passes`]): under CI a different version fails the gate, locally it is a notice;
//! - the tests below read `.github/workflows/*.yml` and `CLAUDE.md` and fail when either names a
//!   different version than the pin, so the three places cannot drift;
//! - the install hints the gate prints name the pinned version ([`install_argument`]).
//!
//! What this does not pin: the compiler (`rust-toolchain.toml` does) and the GitHub actions (each
//! is pinned to a commit in the workflow).

use std::collections::BTreeMap;

use serde::Deserialize;

/// The pins, compiled in so that a checkout's `xtask` binary carries its own copy.
pub const PINS: &str = include_str!("../tool-versions.toml");

/// The cargo plugins the gate can run, as (`cargo` subcommand, crate name). The crate names are
/// also the names `taiki-e/install-action` takes.
pub const PLUGINS: [(&str, &str); 3] = [
    ("nextest", "cargo-nextest"),
    ("deny", "cargo-deny"),
    ("machete", "cargo-machete"),
];

/// The crate names of [`PLUGINS`].
pub const TOOLS: [&str; 3] = [PLUGINS[0].1, PLUGINS[1].1, PLUGINS[2].1];

/// Why `xtask/tool-versions.toml` cannot be used.
#[derive(Debug, PartialEq, thiserror::Error)]
pub enum Error {
    /// Not valid TOML of the expected shape.
    #[error("xtask/tool-versions.toml: {0}")]
    Syntax(#[from] toml::de::Error),
    /// A pinned version that is not `major.minor.patch` (digits only): a range or a tag would
    /// not pin anything.
    #[error(
        "xtask/tool-versions.toml: {tool} = {version:?} is not an exact major.minor.patch version"
    )]
    NotExact {
        /// The tool.
        tool: String,
        /// What the file says.
        version: String,
    },
    /// A tool of [`TOOLS`] has no pin.
    #[error("xtask/tool-versions.toml: no version is pinned for {0}")]
    Missing(&'static str),
    /// A pin for something that is not in [`TOOLS`]; nothing would use it.
    #[error("xtask/tool-versions.toml: {0} is pinned but the gate does not use it")]
    Unknown(String),
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PinFile {
    tools: BTreeMap<String, String>,
}

/// The pinned version of every tool in [`TOOLS`].
#[derive(Debug)]
pub struct Pins {
    versions: BTreeMap<String, String>,
}

/// Whether `version` is `major.minor.patch` with all-digit parts.
fn is_exact(version: &str) -> bool {
    let mut parts = version.split('.');
    let digits = |part: Option<&str>| {
        part.is_some_and(|part| !part.is_empty() && part.bytes().all(|b| b.is_ascii_digit()))
    };
    digits(parts.next()) && digits(parts.next()) && digits(parts.next()) && parts.next().is_none()
}

impl Pins {
    /// Parses `xtask/tool-versions.toml`: a `[tools]` table with an exact version for every tool
    /// of [`TOOLS`] and for nothing else.
    pub fn parse(text: &str) -> Result<Self, Error> {
        let file: PinFile = toml::from_str(text)?;
        for (tool, version) in &file.tools {
            if !TOOLS.contains(&tool.as_str()) {
                return Err(Error::Unknown(tool.clone()));
            }
            if !is_exact(version) {
                return Err(Error::NotExact {
                    tool: tool.clone(),
                    version: version.clone(),
                });
            }
        }
        for tool in TOOLS {
            if !file.tools.contains_key(tool) {
                return Err(Error::Missing(tool));
            }
        }
        Ok(Self {
            versions: file.tools,
        })
    }

    /// The pinned version of `tool`, if it is one of [`TOOLS`].
    pub fn version(&self, tool: &str) -> Option<&str> {
        self.versions.get(tool).map(String::as_str)
    }
}

/// The argument that installs `tool` at `version`, for `cargo install` and
/// `taiki-e/install-action` alike: `cargo-deny@0.19.8`.
pub fn install_argument(tool: &str, version: &str) -> String {
    format!("{tool}@{version}")
}

/// What an installed plugin's `--version` output says, against the pin.
#[derive(Debug, PartialEq, Eq)]
pub enum Check {
    /// The installed version is the pinned one.
    Matches,
    /// A different version is installed.
    Differs {
        /// The version the output names.
        installed: String,
        /// The pinned version.
        pinned: String,
    },
    /// The output names no version, so nothing can be said about it.
    Unreadable,
}

/// The version a plugin's `--version` output names: the first word of its first line that starts
/// with a digit (`cargo-deny 0.19.8`, `cargo-nextest 0.9.140 (a9fef2964 2026-07-05)`, or just
/// `0.9.2`, which is all `cargo machete --version` prints).
fn reported_version(output: &str) -> Option<&str> {
    output
        .lines()
        .next()?
        .split_whitespace()
        .find(|word| word.starts_with(|c: char| c.is_ascii_digit()))
}

/// Compares `--version` output with `pinned`.
pub fn check_output(output: &str, pinned: &str) -> Check {
    match reported_version(output) {
        Some(installed) if installed == pinned => Check::Matches,
        Some(installed) => Check::Differs {
            installed: installed.to_owned(),
            pinned: pinned.to_owned(),
        },
        None => Check::Unreadable,
    }
}

/// Whether the gate may go on: a matching version always; anything else only locally (where the
/// caller prints a notice). Under CI a version the pin does not name, or one that cannot be
/// read, fails, so the checks run by CI are the checks the pin describes.
pub fn passes(check: &Check, in_ci: bool) -> bool {
    matches!(check, Check::Matches) || !in_ci
}

/// Whether the gate may go on given every installed plugin's check (tool name and check): only
/// if [`passes`] holds for each of them.
pub fn version_verdict(checks: &[(&str, Check)], in_ci: bool) -> bool {
    checks.iter().all(|(_, check)| passes(check, in_ci))
}

/// The notice the gate prints for a check that is not [`Check::Matches`]; `None` when it matches.
/// `tool` is the crate name (`cargo-deny`).
pub fn describe(tool: &str, check: &Check, pinned: &str) -> Option<String> {
    let install = format!("cargo install {} --locked", install_argument(tool, pinned));
    match check {
        Check::Matches => None,
        Check::Differs { installed, pinned } => Some(format!(
            "{tool} {installed} is installed but xtask/tool-versions.toml pins {pinned}; \
             `{install}` installs the pinned one"
        )),
        Check::Unreadable => Some(format!(
            "`--version` of {tool} names no version, so it cannot be compared with the pin \
             {pinned}; `{install}` installs the pinned one"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn pins() -> Pins {
        Pins::parse(PINS).unwrap()
    }

    fn repo_file(path: &str) -> String {
        std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join(path))
            .unwrap()
    }

    #[test]
    fn the_shipped_pins_parse_and_cover_every_tool() {
        let pins = pins();
        for tool in TOOLS {
            assert!(pins.version(tool).is_some(), "{tool} has no pin");
        }
        assert_eq!(pins.version("cargo-audit"), None);
    }

    #[test]
    fn every_plugin_subcommand_is_its_crate_name_without_the_cargo_prefix() {
        for (subcommand, crate_name) in PLUGINS {
            assert_eq!(crate_name.strip_prefix("cargo-"), Some(subcommand));
        }
        assert_eq!(TOOLS.len(), PLUGINS.len());
    }

    fn differs() -> Check {
        Check::Differs {
            installed: "1.0.0".to_owned(),
            pinned: "2.0.0".to_owned(),
        }
    }

    #[test]
    fn one_mismatch_in_ci_stops_the_gate_whichever_tool_it_is() {
        for bad in [differs(), Check::Unreadable] {
            for position in 0..TOOLS.len() {
                let checks: Vec<(&str, Check)> = TOOLS
                    .iter()
                    .enumerate()
                    .map(|(index, tool)| {
                        let check = if index == position {
                            match &bad {
                                Check::Unreadable => Check::Unreadable,
                                _ => differs(),
                            }
                        } else {
                            Check::Matches
                        };
                        (*tool, check)
                    })
                    .collect();
                assert!(!version_verdict(&checks, true), "{bad:?} at {position}");
                // The same findings are a notice locally.
                assert!(version_verdict(&checks, false), "{bad:?} at {position}");
            }
        }
    }

    #[test]
    fn matching_or_no_installed_plugins_pass_everywhere() {
        let all_match: Vec<(&str, Check)> =
            TOOLS.iter().map(|tool| (*tool, Check::Matches)).collect();
        for in_ci in [true, false] {
            assert!(version_verdict(&all_match, in_ci));
            assert!(version_verdict(&[], in_ci));
        }
    }

    #[test]
    fn a_pin_must_be_an_exact_version() {
        for bad in [
            "0.19",
            "^0.19.8",
            "0.19.8-rc.1",
            "latest",
            "",
            "0.19.8.1",
            "0..8",
        ] {
            let text = format!(
                "[tools]\ncargo-nextest = \"0.9.140\"\ncargo-deny = {bad:?}\ncargo-machete = \"0.9.2\"\n"
            );
            assert_eq!(
                Pins::parse(&text).unwrap_err(),
                Error::NotExact {
                    tool: "cargo-deny".into(),
                    version: bad.into()
                },
                "{bad:?}"
            );
        }
    }

    #[test]
    fn every_tool_needs_a_pin_and_nothing_else_may_have_one() {
        let two = "[tools]\ncargo-nextest = \"0.9.140\"\ncargo-deny = \"0.19.8\"\n";
        assert_eq!(
            Pins::parse(two).unwrap_err(),
            Error::Missing("cargo-machete")
        );
        let extra = format!("{two}cargo-machete = \"0.9.2\"\ncargo-audit = \"0.21.0\"\n");
        assert_eq!(
            Pins::parse(&extra).unwrap_err(),
            Error::Unknown("cargo-audit".into())
        );
        assert!(matches!(
            Pins::parse("[tools]\nunknown_key = true\n[other]\n"),
            Err(Error::Syntax(_))
        ));
    }

    #[test]
    fn install_arguments_join_the_tool_and_the_version() {
        assert_eq!(
            install_argument("cargo-deny", "0.19.8"),
            "cargo-deny@0.19.8"
        );
    }

    #[test]
    fn the_version_is_read_from_the_first_line_of_the_output() {
        // The shapes the three plugins print.
        let nextest = "cargo-nextest 0.9.140 (a9fef2964 2026-07-05)\nrelease: 0.9.140\n";
        assert_eq!(check_output(nextest, "0.9.140"), Check::Matches);
        assert_eq!(
            check_output("cargo-deny 0.19.8\n", "0.19.8"),
            Check::Matches
        );
        assert_eq!(
            check_output("cargo-machete 0.9.2\n", "0.9.2"),
            Check::Matches
        );
        // `cargo machete --version` prints the number alone.
        assert_eq!(check_output("0.9.2\n", "0.9.2"), Check::Matches);
        // A name with no number, or a number on a later line, names no version.
        assert_eq!(
            check_output("cargo-machete\n0.9.2\n", "0.9.2"),
            Check::Unreadable
        );
    }

    #[test]
    fn a_different_version_names_both() {
        assert_eq!(
            check_output("cargo-deny 0.20.2\n", "0.19.8"),
            Check::Differs {
                installed: "0.20.2".into(),
                pinned: "0.19.8".into()
            }
        );
        // A later line naming the pinned version does not count.
        assert_eq!(
            check_output("cargo-nextest 0.9.151\nrelease: 0.9.140\n", "0.9.140"),
            Check::Differs {
                installed: "0.9.151".into(),
                pinned: "0.9.140".into()
            }
        );
        // A longer version that starts with the pin is not the pin.
        assert!(matches!(
            check_output("cargo-deny 0.19.80\n", "0.19.8"),
            Check::Differs { .. }
        ));
    }

    #[test]
    fn output_without_a_version_is_unreadable() {
        for output in ["", "\n", "cargo-deny\n", "error: no such command\n"] {
            assert_eq!(
                check_output(output, "0.19.8"),
                Check::Unreadable,
                "{output:?}"
            );
        }
    }

    #[test]
    fn only_a_matching_version_passes_under_ci() {
        let differs = Check::Differs {
            installed: "1.0.0".into(),
            pinned: "0.19.8".into(),
        };
        assert!(passes(&Check::Matches, true));
        assert!(passes(&Check::Matches, false));
        assert!(!passes(&differs, true));
        assert!(passes(&differs, false));
        assert!(!passes(&Check::Unreadable, true));
        assert!(passes(&Check::Unreadable, false));
    }

    #[test]
    fn the_notice_names_the_tool_both_versions_and_the_install_command() {
        assert_eq!(describe("cargo-deny", &Check::Matches, "0.19.8"), None);
        let differs = Check::Differs {
            installed: "0.20.2".into(),
            pinned: "0.19.8".into(),
        };
        assert_eq!(
            describe("cargo-deny", &differs, "0.19.8").unwrap(),
            "cargo-deny 0.20.2 is installed but xtask/tool-versions.toml pins 0.19.8; \
             `cargo install cargo-deny@0.19.8 --locked` installs the pinned one"
        );
        assert_eq!(
            describe("cargo-machete", &Check::Unreadable, "0.9.2").unwrap(),
            "`--version` of cargo-machete names no version, so it cannot be compared with the pin \
             0.9.2; `cargo install cargo-machete@0.9.2 --locked` installs the pinned one"
        );
    }

    /// The `tool:` values of a workflow: every line `tool: a,b,c` as its entries.
    fn tool_lists(workflow: &str) -> Vec<Vec<String>> {
        workflow
            .lines()
            .filter_map(|line| line.trim().strip_prefix("tool:"))
            .map(|list| list.split(',').map(|tool| tool.trim().to_owned()).collect())
            .collect()
    }

    #[test]
    fn the_workflows_install_exactly_the_pinned_versions() {
        let pins = pins();
        let expected: Vec<String> = TOOLS
            .iter()
            .map(|tool| install_argument(tool, pins.version(tool).unwrap()))
            .collect();
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../.github/workflows");
        let mut lists = 0;
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            let text = std::fs::read_to_string(&path).unwrap();
            for list in tool_lists(&text) {
                lists += 1;
                for tool in TOOLS {
                    // A tool that is installed must be installed at its pin; an entry without
                    // `@version` would take the latest release.
                    let entry = list
                        .iter()
                        .find(|entry| entry.split('@').next() == Some(tool));
                    if let Some(entry) = entry {
                        let want = install_argument(tool, pins.version(tool).unwrap());
                        assert_eq!(*entry, want, "{}", path.display());
                    }
                }
            }
        }
        // ci.yml installs all three in each of its two jobs (linux, macos).
        let ci = repo_file(".github/workflows/ci.yml");
        let ci_lists = tool_lists(&ci);
        assert_eq!(ci_lists.len(), 2, "ci.yml jobs that install the gate tools");
        for list in ci_lists {
            assert_eq!(list, expected);
        }
        assert!(lists >= 2);
    }

    #[test]
    fn the_install_hint_in_claude_md_names_the_pinned_versions() {
        let pins = pins();
        let command = format!(
            "cargo install {} {} --locked",
            install_argument("cargo-deny", pins.version("cargo-deny").unwrap()),
            install_argument("cargo-machete", pins.version("cargo-machete").unwrap()),
        );
        assert!(
            repo_file("CLAUDE.md").contains(&command),
            "CLAUDE.md should say `{command}`"
        );
    }
}
