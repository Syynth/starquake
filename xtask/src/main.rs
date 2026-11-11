//! Repository automation. `cargo xtask ci` is the gate: CI runs exactly this, and so should
//! every agent before saying a change is done.

#![expect(
    clippy::print_stderr,
    reason = "a command-line tool: progress and errors go to stderr"
)]

use std::process::{Command, ExitCode};
use std::time::Instant;

fn main() -> ExitCode {
    if let Some("ci") = std::env::args().nth(1).as_deref() {
        ci()
    } else {
        eprintln!("usage: cargo xtask ci");
        ExitCode::FAILURE
    }
}

fn cargo() -> Command {
    Command::new(std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into()))
}

/// Free space below which the gate refuses to build. Rust build folders grow fast, and several
/// agents build at once; running out mid-build fails every one of them.
const DEFAULT_MIN_FREE_GIB: u64 = 15;

/// Free space in GiB on the volume holding the current directory, or `None` if `df` can't say.
fn free_gib() -> Option<u64> {
    let output = Command::new("df").args(["-Pk", "."]).output().ok()?;
    let text = String::from_utf8(output.stdout).ok()?;
    let available_kib: u64 = text
        .lines()
        .nth(1)?
        .split_whitespace()
        .nth(3)?
        .parse()
        .ok()?;
    Some(available_kib / (1024 * 1024))
}

/// Returns false (after explaining) if the disk is too full to build safely.
fn disk_ok() -> bool {
    // Hosted CI runners start with little free space and are thrown away afterwards.
    if std::env::var_os("CI").is_some() {
        return true;
    }
    let min = std::env::var("STARQUAKE_MIN_FREE_GIB")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(DEFAULT_MIN_FREE_GIB);
    match free_gib() {
        Some(free) if free < min => {
            eprintln!(
                "==> disk: only {free} GiB free (floor is {min} GiB). Free space first: \
                 `cargo clean`, remove merged worktrees (`git worktree prune`), or set \
                 STARQUAKE_MIN_FREE_GIB to override."
            );
            false
        }
        Some(free) => {
            eprintln!("==> disk: {free} GiB free (floor {min} GiB)");
            true
        }
        None => true,
    }
}

/// One gate step: a display name and the arguments passed to `cargo`.
#[derive(Debug, PartialEq, Eq)]
struct Step {
    name: &'static str,
    args: Vec<&'static str>,
}

/// What the gate does about `cargo deny`.
#[derive(Debug, PartialEq, Eq)]
enum Deny {
    /// Run the check.
    Run,
    /// The tool is not installed and this is a local run: skip, and say so.
    SkipNotInstalled,
    /// The tool is not installed under CI, where the check is mandatory: fail the gate.
    MissingInCi,
}

/// Decides whether `cargo deny` runs: whenever it is installed, and always under CI, where being
/// absent is a failure rather than a skip (so CI can never silently lose the check).
fn deny_decision(installed: bool, in_ci: bool) -> Deny {
    match (installed, in_ci) {
        (true, _) => Deny::Run,
        (false, false) => Deny::SkipNotInstalled,
        (false, true) => Deny::MissingInCi,
    }
}

/// The gate's steps in order. The checks that need no compilation (fmt, deny) come first.
fn steps(has_nextest: bool, run_deny: bool) -> Vec<Step> {
    let tests = if has_nextest {
        vec![
            "nextest",
            "run",
            "--workspace",
            "--locked",
            "--no-tests=pass",
        ]
    } else {
        vec!["test", "--workspace", "--locked"]
    };
    let mut steps = vec![Step {
        name: "fmt",
        args: vec!["fmt", "--all", "--check"],
    }];
    if run_deny {
        // Licences, bans, sources and advisories, configured in `deny.toml`. The advisory
        // database is fetched, so this step needs network access.
        steps.push(Step {
            name: "deny",
            args: vec!["deny", "--locked", "check"],
        });
    }
    steps.extend([
        Step {
            name: "clippy",
            args: vec![
                "clippy",
                "--workspace",
                "--all-targets",
                "--locked",
                "--",
                "-D",
                "warnings",
            ],
        },
        Step {
            name: "test",
            args: tests,
        },
        Step {
            name: "doctest",
            args: vec!["test", "--workspace", "--doc", "--locked"],
        },
    ]);
    steps
}

/// Whether `cargo <subcommand> --version` works, i.e. the cargo plugin is installed.
fn has_cargo_plugin(subcommand: &str) -> bool {
    cargo()
        .args([subcommand, "--version"])
        .output()
        .is_ok_and(|output| output.status.success())
}

fn ci() -> ExitCode {
    if !disk_ok() {
        return ExitCode::FAILURE;
    }
    let in_ci = std::env::var_os("CI").is_some();
    let run_deny = match deny_decision(has_cargo_plugin("deny"), in_ci) {
        Deny::Run => true,
        Deny::SkipNotInstalled => {
            eprintln!(
                "==> deny: SKIPPED, cargo-deny is not installed (CI always runs it; \
                 `cargo install cargo-deny --locked` to run it here)"
            );
            false
        }
        Deny::MissingInCi => {
            eprintln!("==> deny FAILED: cargo-deny must be installed when CI is set");
            return ExitCode::FAILURE;
        }
    };
    let total = Instant::now();
    for Step { name, args } in steps(has_cargo_plugin("nextest"), run_deny) {
        let start = Instant::now();
        eprintln!("==> {name}: cargo {}", args.join(" "));
        let ok = cargo()
            .args(&args)
            .status()
            .is_ok_and(|status| status.success());
        if !ok {
            eprintln!("==> {name} FAILED");
            return ExitCode::FAILURE;
        }
        eprintln!("==> {name} ok ({:.1}s)", start.elapsed().as_secs_f64());
    }
    eprintln!("==> gate passed in {:.1}s", total.elapsed().as_secs_f64());
    ExitCode::SUCCESS
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(steps: &[Step]) -> Vec<&'static str> {
        steps.iter().map(|step| step.name).collect()
    }

    fn args_of<'a>(steps: &'a [Step], name: &str) -> Option<&'a [&'static str]> {
        steps
            .iter()
            .find(|step| step.name == name)
            .map(|step| step.args.as_slice())
    }

    #[test]
    fn deny_runs_when_installed_locally_and_in_ci() {
        assert_eq!(deny_decision(true, false), Deny::Run);
        assert_eq!(deny_decision(true, true), Deny::Run);
    }

    #[test]
    fn deny_is_skipped_locally_but_mandatory_in_ci() {
        assert_eq!(deny_decision(false, false), Deny::SkipNotInstalled);
        assert_eq!(deny_decision(false, true), Deny::MissingInCi);
    }

    #[test]
    fn the_gate_includes_deny_only_when_asked() {
        assert_eq!(
            names(&steps(true, true)),
            ["fmt", "deny", "clippy", "test", "doctest"]
        );
        assert_eq!(
            names(&steps(true, false)),
            ["fmt", "clippy", "test", "doctest"]
        );
    }

    #[test]
    fn deny_checks_against_the_lockfile() {
        assert_eq!(
            args_of(&steps(false, true), "deny"),
            Some(&["deny", "--locked", "check"][..])
        );
    }

    #[test]
    fn tests_use_nextest_only_when_present() {
        assert_eq!(
            args_of(&steps(true, false), "test").and_then(|args| args.first()),
            Some(&"nextest")
        );
        assert_eq!(
            args_of(&steps(false, false), "test").and_then(|args| args.first()),
            Some(&"test")
        );
    }
}
