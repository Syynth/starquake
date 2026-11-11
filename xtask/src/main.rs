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

fn ci() -> ExitCode {
    if !disk_ok() {
        return ExitCode::FAILURE;
    }
    let has_nextest = cargo()
        .args(["nextest", "--version"])
        .output()
        .is_ok_and(|output| output.status.success());
    let tests: &[&str] = if has_nextest {
        &[
            "nextest",
            "run",
            "--workspace",
            "--locked",
            "--no-tests=pass",
        ]
    } else {
        &["test", "--workspace", "--locked"]
    };
    let steps: [(&str, &[&str]); 4] = [
        ("fmt", &["fmt", "--all", "--check"]),
        (
            "clippy",
            &[
                "clippy",
                "--workspace",
                "--all-targets",
                "--locked",
                "--",
                "-D",
                "warnings",
            ],
        ),
        ("test", tests),
        ("doctest", &["test", "--workspace", "--doc", "--locked"]),
    ];
    let total = Instant::now();
    for (name, args) in steps {
        let start = Instant::now();
        eprintln!("==> {name}: cargo {}", args.join(" "));
        let ok = cargo()
            .args(args)
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
