//! Repository automation. `cargo xtask ci` is the gate: CI runs exactly this, and so should
//! every agent before saying a change is done.

#![expect(
    clippy::print_stderr,
    reason = "a command-line tool: progress and errors go to stderr"
)]

mod unused_deps;

use std::path::Path;
use std::process::{Command, ExitCode, Stdio};
use std::time::Instant;

/// The environment variable that has the same effect as `--offline`.
const OFFLINE_VARIABLE: &str = "STARQUAKE_OFFLINE";

const USAGE: &str = "usage: cargo xtask ci [--offline]\n\
    \x20 --offline  skip the one network-dependent check (cargo-deny's advisory database); the same \
    as setting STARQUAKE_OFFLINE to anything but empty or 0. CI refuses it.";

/// A command line the tool understands.
#[derive(Debug, PartialEq, Eq)]
enum Task {
    /// Run the gate; `offline` skips the advisory check.
    Ci { offline: bool },
}

/// Why a command line was not understood.
#[derive(Debug, PartialEq, Eq, thiserror::Error)]
enum UsageError {
    /// No command, or one that does not exist.
    #[error("unknown or missing command")]
    Command,
    /// An argument the command does not take.
    #[error("unexpected argument {0:?}")]
    Argument(String),
}

/// Whether the environment value of [`OFFLINE_VARIABLE`] asks for offline mode: any value that is
/// neither empty nor `0`.
fn offline_from_env(value: Option<&str>) -> bool {
    value.is_some_and(|value| !value.is_empty() && value != "0")
}

/// Reads the command line (without the program name). `offline_env` is the offline switch as the
/// environment sets it; the flag and the variable are equivalent.
fn parse_args<I: IntoIterator<Item = String>>(
    args: I,
    offline_env: bool,
) -> Result<Task, UsageError> {
    let mut args = args.into_iter();
    if args.next().as_deref() != Some("ci") {
        return Err(UsageError::Command);
    }
    let mut offline = offline_env;
    for arg in args {
        if arg == "--offline" {
            offline = true;
        } else {
            return Err(UsageError::Argument(arg));
        }
    }
    Ok(Task::Ci { offline })
}

fn main() -> ExitCode {
    let offline_env = offline_from_env(std::env::var(OFFLINE_VARIABLE).ok().as_deref());
    match parse_args(std::env::args().skip(1), offline_env) {
        Ok(Task::Ci { offline }) => ci(offline),
        Err(error) => {
            eprintln!("{error}\n{USAGE}");
            ExitCode::FAILURE
        }
    }
}

/// A `cargo` command for the checkout's gate steps.
///
/// `cargo run` (which is what `cargo xtask` is) sets `CARGO_PKG_NAME=xtask` for the program it
/// runs. `cargo machete` reads that variable and, when it is set, stops recognising its own
/// arguments (it then treats `machete` and `--version` as folders to scan), so the variable must
/// not reach the cargo commands spawned from here. None of them is part of the xtask package.
fn cargo() -> Command {
    let mut command = Command::new(std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into()));
    command.env_remove("CARGO_PKG_NAME");
    command
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

/// What a gate step does.
#[derive(Debug, PartialEq, Eq)]
enum Action {
    /// Run `cargo` with these arguments.
    Cargo(Vec<&'static str>),
    /// Fail if a registry dependency in the root `[workspace.dependencies]` has no user.
    WorkspaceDependencies,
    /// Run `cargo machete` and compare its findings with `xtask/unused-deps.toml`.
    Machete,
}

/// One gate step: a display name and what it does.
#[derive(Debug, PartialEq, Eq)]
struct Step {
    name: &'static str,
    action: Action,
}

/// Whether an optional cargo plugin is used by this run.
#[derive(Debug, PartialEq, Eq)]
enum Availability {
    /// Run the check.
    Run,
    /// The tool is not installed and this is a local run: skip, and say so.
    SkipNotInstalled,
    /// The tool is not installed under CI, where the check is mandatory: fail the gate.
    MissingInCi,
}

/// Decides whether an optional tool runs: whenever it is installed, and always under CI, where
/// being absent is a failure rather than a skip (so CI can never silently lose the check).
fn availability(installed: bool, in_ci: bool) -> Availability {
    match (installed, in_ci) {
        (true, _) => Availability::Run,
        (false, false) => Availability::SkipNotInstalled,
        (false, true) => Availability::MissingInCi,
    }
}

/// Which `cargo deny` checks run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DenyChecks {
    /// All of them: licences, bans, sources and advisories.
    All,
    /// Offline: everything except the advisories, whose database is fetched from the network.
    WithoutAdvisories,
}

/// The `cargo deny` arguments for `checks`. Licences, bans, sources and advisories are
/// configured in `deny.toml`; with all checks the advisory database is fetched, so that run needs
/// network access.
fn deny_args(checks: DenyChecks) -> Vec<&'static str> {
    match checks {
        DenyChecks::All => vec!["deny", "--locked", "check"],
        DenyChecks::WithoutAdvisories => vec![
            "deny",
            "--locked",
            "--offline",
            "check",
            "bans",
            "licenses",
            "sources",
        ],
    }
}

/// What this run of the gate includes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Plan {
    /// Use `cargo nextest` for the tests.
    nextest: bool,
    /// Run `cargo deny`, and which of its checks; `None` when the tool is skipped.
    deny: Option<DenyChecks>,
    /// Run `cargo machete` against the allow-list.
    machete: bool,
}

/// The gate's steps in order. The checks that need no compilation (fmt, the dependency checks,
/// deny) come first.
fn steps(plan: Plan) -> Vec<Step> {
    let tests = if plan.nextest {
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
    let mut steps = vec![
        Step {
            name: "fmt",
            action: Action::Cargo(vec!["fmt", "--all", "--check"]),
        },
        Step {
            name: "workspace-deps",
            action: Action::WorkspaceDependencies,
        },
    ];
    if plan.machete {
        steps.push(Step {
            name: "machete",
            action: Action::Machete,
        });
    }
    if let Some(checks) = plan.deny {
        steps.push(Step {
            name: "deny",
            action: Action::Cargo(deny_args(checks)),
        });
    }
    steps.extend([
        Step {
            name: "clippy",
            action: Action::Cargo(vec![
                "clippy",
                "--workspace",
                "--all-targets",
                "--locked",
                "--",
                "-D",
                "warnings",
            ]),
        },
        Step {
            name: "test",
            action: Action::Cargo(tests),
        },
        Step {
            name: "doctest",
            action: Action::Cargo(vec!["test", "--workspace", "--doc", "--locked"]),
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

/// Decides one optional tool's place in the plan, saying why on stderr when it is skipped.
/// `None` means the gate must fail.
fn tool_runs(subcommand: &str, install: &str, in_ci: bool) -> Option<bool> {
    match availability(has_cargo_plugin(subcommand), in_ci) {
        Availability::Run => Some(true),
        Availability::SkipNotInstalled => {
            eprintln!(
                "==> {subcommand}: SKIPPED, cargo-{subcommand} is not installed (CI always runs \
                 it; `{install}` to run it here)"
            );
            Some(false)
        }
        Availability::MissingInCi => {
            eprintln!(
                "==> {subcommand} FAILED: cargo-{subcommand} must be installed when CI is set"
            );
            None
        }
    }
}

/// Runs `cargo machete` and judges its findings against the allow-list. Returns whether the
/// step passes.
fn run_machete() -> bool {
    let allowlist = match unused_deps::Allowlist::parse(unused_deps::ALLOWLIST) {
        Ok(allowlist) => allowlist,
        Err(error) => {
            eprintln!("{error}");
            return false;
        }
    };
    let output = match cargo()
        .args(["machete", "--skip-target-dir"])
        .stderr(Stdio::inherit())
        .output()
    {
        Ok(output) => output,
        Err(error) => {
            eprintln!("cannot run cargo machete: {error}");
            return false;
        }
    };
    // Exit status: 0 = nothing unused, 1 = findings, 2 (or a signal) = machete itself failed.
    let found = match output.status.code() {
        Some(0) => Vec::new(),
        Some(1) => match unused_deps::parse_machete(&String::from_utf8_lossy(&output.stdout)) {
            Ok(found) => found,
            Err(error) => {
                eprintln!("{error}");
                return false;
            }
        },
        status => {
            eprintln!("cargo machete failed (exit status {status:?})");
            return false;
        }
    };
    let verdict = unused_deps::judge(&found, &allowlist, unused_deps::Scope::Crates);
    for unused in &verdict.unlisted {
        eprintln!(
            "unused dependency: {} / {}: remove it from the crate's Cargo.toml, or, if it is \
             declared ahead of a planned task, list it with the reason in xtask/unused-deps.toml",
            unused.package, unused.dependency
        );
    }
    for unused in &verdict.stale {
        eprintln!(
            "stale entry in xtask/unused-deps.toml: {} / {} is no longer reported as unused; \
             delete the entry",
            unused.package, unused.dependency
        );
    }
    verdict.is_clean()
}

/// Checks the root `[workspace.dependencies]` for registry dependencies nobody uses, against
/// the workspace entries of the allow-list. Returns whether the step passes.
fn workspace_dependencies() -> bool {
    let allowlist = match unused_deps::Allowlist::parse(unused_deps::ALLOWLIST) {
        Ok(allowlist) => allowlist,
        Err(error) => {
            eprintln!("{error}");
            return false;
        }
    };
    let found = match unused_deps::check_workspace_dependencies(Path::new(".")) {
        Ok(found) => found,
        Err(error) => {
            eprintln!("{error}");
            return false;
        }
    };
    let verdict = unused_deps::judge(&found, &allowlist, unused_deps::Scope::Workspace);
    for unused in &verdict.unlisted {
        eprintln!(
            "unused workspace dependency: {0} is in [workspace.dependencies] but no member \
             inherits it (`{0}.workspace = true`); remove it from the root Cargo.toml, or, if it \
             is declared ahead of a planned task, list it with the reason under \
             [[allow_workspace]] in xtask/unused-deps.toml",
            unused.dependency
        );
    }
    for unused in &verdict.stale {
        eprintln!(
            "stale [[allow_workspace]] entry in xtask/unused-deps.toml: {} is now used or no \
             longer declared; delete the entry",
            unused.dependency
        );
    }
    verdict.is_clean()
}

fn ci(offline: bool) -> ExitCode {
    let in_ci = std::env::var_os("CI").is_some();
    if offline && in_ci {
        eprintln!(
            "==> offline FAILED: the advisory check is mandatory under CI (CI is set), so \
             --offline and {OFFLINE_VARIABLE} are refused"
        );
        return ExitCode::FAILURE;
    }
    if !disk_ok() {
        return ExitCode::FAILURE;
    }
    let Some(deny) = tool_runs("deny", "cargo install cargo-deny --locked", in_ci) else {
        return ExitCode::FAILURE;
    };
    let Some(machete) = tool_runs("machete", "cargo install cargo-machete --locked", in_ci) else {
        return ExitCode::FAILURE;
    };
    let deny = deny.then_some(if offline {
        DenyChecks::WithoutAdvisories
    } else {
        DenyChecks::All
    });
    if offline && deny.is_some() {
        eprintln!(
            "==> offline: deny runs WITHOUT the advisory check (needs the network); CI always \
             runs it"
        );
    }
    let total = Instant::now();
    let plan = Plan {
        nextest: has_cargo_plugin("nextest"),
        deny,
        machete,
    };
    for Step { name, action } in steps(plan) {
        let start = Instant::now();
        let ok = match action {
            Action::Cargo(args) => {
                eprintln!("==> {name}: cargo {}", args.join(" "));
                cargo()
                    .args(&args)
                    .status()
                    .is_ok_and(|status| status.success())
            }
            Action::WorkspaceDependencies => {
                eprintln!("==> {name}: [workspace.dependencies] against the members");
                workspace_dependencies()
            }
            Action::Machete => {
                eprintln!("==> {name}: cargo machete against xtask/unused-deps.toml");
                run_machete()
            }
        };
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

    const FULL: Plan = Plan {
        nextest: true,
        deny: Some(DenyChecks::All),
        machete: true,
    };

    fn names(steps: &[Step]) -> Vec<&'static str> {
        steps.iter().map(|step| step.name).collect()
    }

    fn cargo_args<'a>(steps: &'a [Step], name: &str) -> Option<&'a [&'static str]> {
        steps
            .iter()
            .find(|step| step.name == name)
            .and_then(|step| match &step.action {
                Action::Cargo(args) => Some(args.as_slice()),
                _ => None,
            })
    }

    fn parse(list: &[&str], env: bool) -> Result<Task, UsageError> {
        parse_args(list.iter().map(|arg| (*arg).to_owned()), env)
    }

    #[test]
    fn spawned_cargo_does_not_inherit_the_xtask_package_name() {
        // `cargo run` sets CARGO_PKG_NAME, which breaks `cargo machete`'s argument parsing.
        let removed: Vec<_> = cargo()
            .get_envs()
            .filter(|(_, value)| value.is_none())
            .map(|(key, _)| key.to_owned())
            .collect();
        assert_eq!(removed, ["CARGO_PKG_NAME"]);
    }

    #[test]
    fn tools_run_when_installed_locally_and_in_ci() {
        assert_eq!(availability(true, false), Availability::Run);
        assert_eq!(availability(true, true), Availability::Run);
    }

    #[test]
    fn tools_are_skipped_locally_but_mandatory_in_ci() {
        assert_eq!(availability(false, false), Availability::SkipNotInstalled);
        assert_eq!(availability(false, true), Availability::MissingInCi);
    }

    #[test]
    fn the_gate_runs_the_cheap_checks_before_compiling() {
        assert_eq!(
            names(&steps(FULL)),
            [
                "fmt",
                "workspace-deps",
                "machete",
                "deny",
                "clippy",
                "test",
                "doctest"
            ]
        );
    }

    #[test]
    fn deny_and_machete_are_in_the_gate_only_when_asked() {
        let plan = Plan {
            deny: None,
            machete: false,
            ..FULL
        };
        assert_eq!(
            names(&steps(plan)),
            ["fmt", "workspace-deps", "clippy", "test", "doctest"]
        );
        let plan = Plan {
            machete: false,
            ..FULL
        };
        assert!(!names(&steps(plan)).contains(&"machete"));
        let plan = Plan { deny: None, ..FULL };
        assert!(!names(&steps(plan)).contains(&"deny"));
    }

    #[test]
    fn the_steps_for_the_dependency_checks_carry_their_own_actions() {
        let all = steps(FULL);
        let action_of = |name| all.iter().find(|step| step.name == name).map(|s| &s.action);
        assert_eq!(
            action_of("workspace-deps"),
            Some(&Action::WorkspaceDependencies)
        );
        assert_eq!(action_of("machete"), Some(&Action::Machete));
    }

    #[test]
    fn deny_checks_against_the_lockfile_and_runs_every_check() {
        assert_eq!(
            cargo_args(&steps(FULL), "deny"),
            Some(&["deny", "--locked", "check"][..])
        );
    }

    #[test]
    fn offline_deny_skips_exactly_the_advisories() {
        let plan = Plan {
            deny: Some(DenyChecks::WithoutAdvisories),
            ..FULL
        };
        assert_eq!(
            cargo_args(&steps(plan), "deny"),
            Some(
                &[
                    "deny",
                    "--locked",
                    "--offline",
                    "check",
                    "bans",
                    "licenses",
                    "sources"
                ][..]
            )
        );
    }

    #[test]
    fn tests_use_nextest_only_when_present() {
        assert_eq!(
            cargo_args(&steps(FULL), "test").and_then(|args| args.first()),
            Some(&"nextest")
        );
        let plan = Plan {
            nextest: false,
            ..FULL
        };
        assert_eq!(
            cargo_args(&steps(plan), "test").and_then(|args| args.first()),
            Some(&"test")
        );
    }

    #[test]
    fn ci_takes_an_optional_offline_flag() {
        assert_eq!(parse(&["ci"], false), Ok(Task::Ci { offline: false }));
        assert_eq!(
            parse(&["ci", "--offline"], false),
            Ok(Task::Ci { offline: true })
        );
        assert_eq!(
            parse(&["ci", "--offline", "--offline"], false),
            Ok(Task::Ci { offline: true })
        );
    }

    #[test]
    fn the_environment_switch_is_equivalent_to_the_flag() {
        assert_eq!(parse(&["ci"], true), Ok(Task::Ci { offline: true }));
        assert_eq!(
            parse(&["ci", "--offline"], true),
            Ok(Task::Ci { offline: true })
        );
    }

    #[test]
    fn only_a_nonempty_value_other_than_zero_turns_the_environment_switch_on() {
        assert!(!offline_from_env(None));
        assert!(!offline_from_env(Some("")));
        assert!(!offline_from_env(Some("0")));
        assert!(offline_from_env(Some("1")));
        assert!(offline_from_env(Some("true")));
    }

    #[test]
    fn unknown_commands_and_arguments_are_usage_errors() {
        assert_eq!(parse(&[], false), Err(UsageError::Command));
        assert_eq!(parse(&["build"], false), Err(UsageError::Command));
        assert_eq!(parse(&["--offline"], false), Err(UsageError::Command));
        assert_eq!(
            parse(&["ci", "--online"], false),
            Err(UsageError::Argument("--online".to_owned()))
        );
        assert_eq!(
            parse(&["ci", "--offline", "extra"], false),
            Err(UsageError::Argument("extra".to_owned()))
        );
    }
}
