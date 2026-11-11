//! Repository automation. `cargo xtask ci` is the gate: CI runs exactly this, and so should
//! every agent before saying a change is done.

use std::process::{Command, ExitCode};
use std::time::Instant;

fn main() -> ExitCode {
    match std::env::args().nth(1).as_deref() {
        Some("ci") => ci(),
        _ => {
            eprintln!("usage: cargo xtask ci");
            ExitCode::FAILURE
        }
    }
}

fn cargo() -> Command {
    Command::new(std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into()))
}

fn ci() -> ExitCode {
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
