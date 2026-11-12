//! Repository automation. `cargo xtask ci` is the gate: CI runs exactly this, and so should
//! every agent before saying a change is done.

#![expect(
    clippy::print_stderr,
    reason = "a command-line tool: progress and errors go to stderr"
)]

mod host_access;
mod table_deps;
mod tools;
mod unused_deps;
mod walk;

use std::path::{Path, PathBuf};
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

/// Why the gate refuses to start.
#[derive(Debug, PartialEq, Eq, thiserror::Error)]
enum Refusal {
    /// `--offline` (or the variable) under CI, where the advisory check is mandatory.
    #[error(
        "==> offline FAILED: the advisory check is mandatory under CI (CI is set), so \
         --offline and {OFFLINE_VARIABLE} are refused"
    )]
    OfflineInCi,
    /// The folder above `xtask/` is not a cargo workspace root.
    #[error("==> workspace FAILED: the folder above {0} has no [workspace] manifest")]
    NotAWorkspaceRoot(String),
    /// The workspace root exists but the process cannot change into it.
    #[error("==> workspace FAILED: cannot enter {0}")]
    CannotEnter(String),
}

/// Decides whether the gate may run with this offline setting: the advisory check can be skipped
/// locally, never under CI.
fn check_offline(offline: bool, in_ci: bool) -> Result<(), Refusal> {
    if offline && in_ci {
        Err(Refusal::OfflineInCi)
    } else {
        Ok(())
    }
}

/// Checks the offline setting, then moves to the workspace root so that the gate behaves the same
/// from any folder of the checkout.
fn start(offline: bool, in_ci: bool) -> Result<(), Refusal> {
    check_offline(offline, in_ci)?;
    let root = workspace_root(Path::new(env!("CARGO_MANIFEST_DIR")))?;
    std::env::set_current_dir(&root).map_err(|_| Refusal::CannotEnter(root.display().to_string()))
}

/// The workspace root of the checkout this binary belongs to: the folder above `xtask/`, given
/// the compile-time `xtask` folder `manifest_dir`. The binary is built per checkout (see
/// `.cargo/config.toml`), so this is the checkout it runs in, however deep the shell's current
/// directory is. The folder must really hold a `[workspace]` manifest.
fn workspace_root(manifest_dir: &Path) -> Result<PathBuf, Refusal> {
    let not_a_root = || Refusal::NotAWorkspaceRoot(manifest_dir.display().to_string());
    let root = manifest_dir.parent().ok_or_else(not_a_root)?;
    let manifest = std::fs::read_to_string(root.join("Cargo.toml")).map_err(|_| not_a_root())?;
    let is_workspace = manifest
        .parse::<toml::Table>()
        .is_ok_and(|table| table.contains_key("workspace"));
    if is_workspace {
        Ok(root.to_owned())
    } else {
        Err(not_a_root())
    }
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
    /// Fail if a dev, build or target-specific dependency of a member is named by no source.
    TableDependencies,
    /// Fail if kernel or service code reaches the host clock, randomness, file system or
    /// network directly (`host_access`).
    HostAccess,
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
        Step {
            name: "table-deps",
            action: Action::TableDependencies,
        },
        Step {
            name: "host-access",
            action: Action::HostAccess,
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

/// The standard output of `cargo <subcommand> --version`, or `None` if the cargo plugin is not
/// installed (the command fails).
fn plugin_version_output(subcommand: &str) -> Option<String> {
    let output = cargo().args([subcommand, "--version"]).output().ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).into_owned())
}

/// Compares each installed plugin with its pin in `xtask/tool-versions.toml`. A mismatch is
/// printed; under CI it also fails the step, locally it is a notice. Returns whether the gate
/// may go on.
fn check_tool_versions(pins: &tools::Pins, in_ci: bool) -> bool {
    let mut checks = Vec::new();
    for (subcommand, tool) in tools::PLUGINS {
        let Some(output) = plugin_version_output(subcommand) else {
            continue;
        };
        let Some(pinned) = pins.version(tool) else {
            continue;
        };
        let check = tools::check_output(&output, pinned);
        if let Some(notice) = tools::describe(tool, &check, pinned) {
            eprintln!(
                "==> {tool}: {}{notice}",
                if tools::passes(&check, in_ci) {
                    "NOTICE, "
                } else {
                    "FAILED, "
                }
            );
        }
        checks.push((tool, check));
    }
    tools::version_verdict(&checks, in_ci)
}

/// Which optional cargo plugins are installed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Installed {
    nextest: bool,
    deny: bool,
    machete: bool,
}

/// A tool the gate cannot go without under CI is not installed.
#[derive(Debug, PartialEq, Eq, thiserror::Error)]
#[error("==> {0} FAILED: cargo-{0} must be installed when CI is set")]
struct MissingInCi(&'static str);

/// Builds the plan from what is installed, whether this is CI and whether the run is offline.
/// `deny` and `machete` are in the plan whenever they are installed; under CI one that is missing
/// fails (deny is checked first), so CI can never silently lose a check.
fn plan_for(installed: Installed, in_ci: bool, offline: bool) -> Result<Plan, MissingInCi> {
    let runs = |present, tool| match availability(present, in_ci) {
        Availability::Run => Ok(true),
        Availability::SkipNotInstalled => Ok(false),
        Availability::MissingInCi => Err(MissingInCi(tool)),
    };
    let deny = runs(installed.deny, "deny")?;
    let machete = runs(installed.machete, "machete")?;
    Ok(Plan {
        nextest: installed.nextest,
        deny: deny.then_some(if offline {
            DenyChecks::WithoutAdvisories
        } else {
            DenyChecks::All
        }),
        machete,
    })
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
    let found = match unused_deps::machete_findings(
        output.status.code(),
        &String::from_utf8_lossy(&output.stdout),
    ) {
        Ok(found) => found,
        Err(error) => {
            eprintln!("{error}");
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

/// Checks the dev, build and target-specific dependency tables (which `cargo machete` skips)
/// against the sources and the `[[allow_tables]]` entries of the allow-list. Returns whether the
/// step passes.
fn table_dependencies() -> bool {
    let allowlist = match unused_deps::Allowlist::parse(unused_deps::ALLOWLIST) {
        Ok(allowlist) => allowlist,
        Err(error) => {
            eprintln!("{error}");
            return false;
        }
    };
    let found = match table_deps::check_table_dependencies(Path::new(".")) {
        Ok(found) => found,
        Err(error) => {
            eprintln!("{error}");
            return false;
        }
    };
    let pairs: Vec<_> = found.iter().map(|found| found.unused.clone()).collect();
    let verdict = unused_deps::judge(&pairs, &allowlist, unused_deps::Scope::Tables);
    let plain = |package: &str| {
        package
            .strip_prefix(unused_deps::TABLES_SCOPE)
            .unwrap_or(package)
            .to_owned()
    };
    for unused in &verdict.unlisted {
        let table = found
            .iter()
            .find(|found| found.unused == *unused)
            .map_or("", |found| found.table.as_str());
        eprintln!(
            "unused dependency: {} / {} in [{table}] is named by none of the crate's sources: \
             remove it, or, if it is used in a way a text search cannot see (a library name that \
             differs from the package name, say) or is declared ahead of a planned task, list it \
             with the reason under [[allow_tables]] in xtask/unused-deps.toml",
            plain(&unused.package),
            unused.dependency
        );
    }
    for unused in &verdict.stale {
        eprintln!(
            "stale [[allow_tables]] entry in xtask/unused-deps.toml: {} / {} is now used or no \
             longer declared; delete the entry",
            plain(&unused.package),
            unused.dependency
        );
    }
    verdict.is_clean()
}

/// Checks that kernel and service code does not reach the host directly. Returns whether the
/// step passes.
fn host_access() -> bool {
    let violations = match host_access::check(Path::new(".")) {
        Ok(violations) => violations,
        Err(error) => {
            eprintln!("{error}");
            return false;
        }
    };
    for violation in &violations {
        eprintln!(
            "{}:{}: {} reaches {}; kernel and service code reads the clock, randomness and host \
             I/O only through the injectable Env (docs/specs/observability.md, \"Replaying the \
             kernel and services\"). Test code is exempt.",
            violation.file, violation.finding.line, violation.finding.path, violation.finding.what
        );
    }
    violations.is_empty()
}

fn ci(offline: bool) -> ExitCode {
    let in_ci = std::env::var_os("CI").is_some();
    if let Err(refusal) = start(offline, in_ci) {
        eprintln!("{refusal}");
        return ExitCode::FAILURE;
    }
    if !disk_ok() {
        return ExitCode::FAILURE;
    }
    let pins = match tools::Pins::parse(tools::PINS) {
        Ok(pins) => pins,
        Err(error) => {
            eprintln!("{error}");
            return ExitCode::FAILURE;
        }
    };
    let installed = Installed {
        nextest: plugin_version_output("nextest").is_some(),
        deny: plugin_version_output("deny").is_some(),
        machete: plugin_version_output("machete").is_some(),
    };
    let plan = match plan_for(installed, in_ci, offline) {
        Ok(plan) => plan,
        Err(missing) => {
            eprintln!("{missing}");
            return ExitCode::FAILURE;
        }
    };
    if !check_tool_versions(&pins, in_ci) {
        return ExitCode::FAILURE;
    }
    for (present, (_, crate_name)) in [
        (installed.deny, tools::PLUGINS[1]),
        (installed.machete, tools::PLUGINS[2]),
    ] {
        if !present {
            eprintln!(
                "==> {crate_name}: SKIPPED, not installed (CI always runs it; `cargo install {} \
                 --locked` to run it here)",
                tools::install_argument(crate_name, pins.version(crate_name).unwrap_or("*"))
            );
        }
    }
    if offline && plan.deny.is_some() {
        eprintln!(
            "==> offline: deny runs WITHOUT the advisory check (needs the network); CI always \
             runs it"
        );
    }
    let total = Instant::now();
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
            Action::TableDependencies => {
                eprintln!("==> {name}: dev, build and target dependencies against the sources");
                table_dependencies()
            }
            Action::HostAccess => {
                eprintln!(
                    "==> {name}: no host clock, randomness, file or network access in {}",
                    host_access::CRATES.join(", ")
                );
                host_access()
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

    const ALL_INSTALLED: Installed = Installed {
        nextest: true,
        deny: true,
        machete: true,
    };

    #[test]
    fn offline_is_refused_under_ci_and_only_then() {
        assert_eq!(check_offline(true, true), Err(Refusal::OfflineInCi));
        assert_eq!(check_offline(true, false), Ok(()));
        assert_eq!(check_offline(false, true), Ok(()));
        assert_eq!(check_offline(false, false), Ok(()));
    }

    #[test]
    fn the_offline_refusal_message_names_the_flag_and_the_variable() {
        assert_eq!(
            Refusal::OfflineInCi.to_string(),
            "==> offline FAILED: the advisory check is mandatory under CI (CI is set), so \
             --offline and STARQUAKE_OFFLINE are refused"
        );
    }

    #[test]
    fn the_workspace_root_is_the_folder_above_xtask() {
        let xtask = Path::new(env!("CARGO_MANIFEST_DIR"));
        let root = workspace_root(xtask).unwrap();
        assert_eq!(root, xtask.parent().unwrap());
        // The gate reads `Cargo.toml`, `xtask/unused-deps.toml` and so on relative to the root.
        assert!(root.join("Cargo.toml").is_file());
        assert!(root.join("xtask").join("Cargo.toml").is_file());
    }

    #[test]
    fn a_folder_that_is_not_below_a_workspace_root_is_refused() {
        // `xtask/src` has `xtask/` above it, whose manifest is a package, not a workspace.
        let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let error = workspace_root(&src).unwrap_err();
        assert_eq!(error, Refusal::NotAWorkspaceRoot(src.display().to_string()));
        assert_eq!(
            error.to_string(),
            format!(
                "==> workspace FAILED: the folder above {} has no [workspace] manifest",
                src.display()
            )
        );
        assert!(matches!(
            workspace_root(Path::new("/")),
            Err(Refusal::NotAWorkspaceRoot(_))
        ));
    }

    #[test]
    fn installed_tools_end_up_in_the_plan() {
        assert_eq!(
            plan_for(ALL_INSTALLED, false, false),
            Ok(Plan {
                nextest: true,
                deny: Some(DenyChecks::All),
                machete: true
            })
        );
        // The same under CI.
        assert_eq!(plan_for(ALL_INSTALLED, true, false), Ok(FULL));
        let no_nextest = Installed {
            nextest: false,
            ..ALL_INSTALLED
        };
        assert_eq!(
            plan_for(no_nextest, false, false),
            Ok(Plan {
                nextest: false,
                ..FULL
            })
        );
    }

    #[test]
    fn a_missing_tool_is_skipped_locally_and_fails_under_ci() {
        let no_machete = Installed {
            machete: false,
            ..ALL_INSTALLED
        };
        let no_deny = Installed {
            deny: false,
            ..ALL_INSTALLED
        };
        assert_eq!(
            plan_for(no_machete, false, false),
            Ok(Plan {
                machete: false,
                ..FULL
            })
        );
        assert_eq!(
            plan_for(no_deny, false, false),
            Ok(Plan { deny: None, ..FULL })
        );
        assert_eq!(
            plan_for(no_machete, true, false),
            Err(MissingInCi("machete"))
        );
        assert_eq!(plan_for(no_deny, true, false), Err(MissingInCi("deny")));
        let neither = Installed {
            deny: false,
            machete: false,
            ..ALL_INSTALLED
        };
        assert_eq!(plan_for(neither, true, false), Err(MissingInCi("deny")));
        assert_eq!(
            MissingInCi("machete").to_string(),
            "==> machete FAILED: cargo-machete must be installed when CI is set"
        );
    }

    #[test]
    fn offline_drops_only_the_advisory_check() {
        assert_eq!(
            plan_for(ALL_INSTALLED, false, true),
            Ok(Plan {
                deny: Some(DenyChecks::WithoutAdvisories),
                ..FULL
            })
        );
        let no_deny = Installed {
            deny: false,
            ..ALL_INSTALLED
        };
        assert_eq!(
            plan_for(no_deny, false, true),
            Ok(Plan { deny: None, ..FULL })
        );
    }

    #[test]
    fn the_gate_runs_the_cheap_checks_before_compiling() {
        assert_eq!(
            names(&steps(FULL)),
            [
                "fmt",
                "workspace-deps",
                "table-deps",
                "host-access",
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
            [
                "fmt",
                "workspace-deps",
                "table-deps",
                "host-access",
                "clippy",
                "test",
                "doctest"
            ]
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
        assert_eq!(action_of("table-deps"), Some(&Action::TableDependencies));
        assert_eq!(action_of("host-access"), Some(&Action::HostAccess));
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
