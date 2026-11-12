//! Under CI the advisory check is mandatory, so `--offline` and `STARQUAKE_OFFLINE` are refused.
//! The refusal comes before the disk check and before any cargo command is spawned, so these
//! tests run the real binary without running the gate.

/// Helpers live in a `cfg(test)` module so clippy's allow-unwrap-in-tests setting covers them.
#[cfg(test)]
mod support {
    use std::process::Command;

    /// What the refusal prints on stderr: the whole of it, with no usage text after it.
    pub const REFUSAL: &str = "==> offline FAILED: the advisory check is mandatory under CI (CI is \
        set), so --offline and STARQUAKE_OFFLINE are refused\n";

    /// Runs `xtask ci` with `args`, `CI` set to `ci` (or removed) and `STARQUAKE_OFFLINE` set to
    /// `offline` (or removed). Returns (exit code, stderr).
    pub fn run(args: &[&str], ci: Option<&str>, offline: Option<&str>) -> (i32, String) {
        let mut command = Command::new(env!("CARGO_BIN_EXE_xtask"));
        command
            .arg("ci")
            .args(args)
            .env_remove("CI")
            .env_remove("STARQUAKE_OFFLINE");
        if let Some(value) = ci {
            command.env("CI", value);
        }
        if let Some(value) = offline {
            command.env("STARQUAKE_OFFLINE", value);
        }
        let output = command.output().unwrap();
        (
            output.status.code().expect("xtask was killed by a signal"),
            String::from_utf8_lossy(&output.stderr).into_owned(),
        )
    }
}

use support::{REFUSAL, run};

/// Exit status of a refused or failed gate (`ExitCode::FAILURE`).
const EXIT_FAILURE: i32 = 1;

#[test]
fn the_flag_is_refused_under_ci() {
    assert_eq!(
        run(&["--offline"], Some("1"), None),
        (EXIT_FAILURE, REFUSAL.to_owned())
    );
}

#[test]
fn the_environment_variable_is_refused_under_ci() {
    assert_eq!(
        run(&[], Some("1"), Some("1")),
        (EXIT_FAILURE, REFUSAL.to_owned())
    );
    assert_eq!(
        run(&[], Some("true"), Some("yes")),
        (EXIT_FAILURE, REFUSAL.to_owned())
    );
}

#[test]
fn ci_is_set_by_being_present_not_by_its_value() {
    // The gate treats any value of CI (even empty) as CI, like `std::env::var_os("CI").is_some()`.
    assert_eq!(
        run(&["--offline"], Some(""), None),
        (EXIT_FAILURE, REFUSAL.to_owned())
    );
}

#[test]
fn a_usage_error_is_not_the_refusal() {
    let (code, stderr) = run(&["--online"], Some("1"), None);
    assert_eq!(code, EXIT_FAILURE);
    assert!(
        stderr.starts_with("unexpected argument \"--online\"\nusage: cargo xtask ci"),
        "{stderr}"
    );
}
