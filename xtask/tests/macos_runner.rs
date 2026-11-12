//! Drives `tools/macos-runner.sh`, the cargo runner that signs a binary with the hypervisor
//! entitlement before running it (`.cargo/config.toml`).
//!
//! The case that matters is `cargo run`: before every run cargo replaces
//! `target/<profile>/<name>` with a fresh copy of the unsigned binary it built earlier, keeping
//! that binary's old modification time. A runner that only asks "did I sign this path since it
//! last changed?" by time then runs the unsigned copy, and Hypervisor.framework refuses to
//! create a VM. These tests replace a signed binary the way cargo does and check that the runner
//! signs it again.

#![cfg(target_os = "macos")]

use std::path::{Path, PathBuf};
use std::process::Command;

/// The entitlement the runner must leave on every binary it runs
/// (`tools/hypervisor.entitlements`).
const ENTITLEMENT: &str = "com.apple.security.hypervisor";

/// Helpers live in a `cfg(test)` module so clippy's allow-unwrap-in-tests setting covers them.
#[cfg(test)]
mod support {
    use super::*;

    fn runner() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../tools/macos-runner.sh")
    }

    /// Runs `command` and returns its exit code with everything it printed.
    fn output(command: &mut Command) -> (i32, String) {
        let output = command.output().unwrap();
        let mut text = String::from_utf8_lossy(&output.stdout).into_owned();
        text.push_str(&String::from_utf8_lossy(&output.stderr));
        (output.status.code().expect("killed by a signal"), text)
    }

    /// An empty scratch folder for the test called `name`.
    pub(crate) fn scratch(name: &str) -> PathBuf {
        let folder = Path::new(env!("CARGO_TARGET_TMPDIR"))
            .join("macos-runner")
            .join(name);
        if folder.exists() {
            std::fs::remove_dir_all(&folder).unwrap();
        }
        std::fs::create_dir_all(&folder).unwrap();
        folder
    }

    /// What cargo's build leaves behind: a binary signed ad hoc with no entitlements, last
    /// modified long ago. (A copy of this test binary; run with `--list` it prints its tests and
    /// exits.)
    pub(crate) fn built(folder: &Path) -> PathBuf {
        let built = folder.join("built");
        std::fs::copy(std::env::current_exe().unwrap(), &built).unwrap();
        let signed = output(
            Command::new("codesign")
                .args(["--force", "--sign", "-"])
                .arg(&built),
        );
        assert_eq!(signed.0, 0, "{}", signed.1);
        let aged = output(
            Command::new("touch")
                .args(["-t", "202001010000"])
                .arg(&built),
        );
        assert_eq!(aged.0, 0, "{}", aged.1);
        assert!(
            !entitled(&built),
            "the built binary starts without the entitlement"
        );
        built
    }

    /// What cargo does before each `cargo run`: puts a copy of the built binary at `path`,
    /// replacing whatever is there and keeping the built binary's modification time.
    pub(crate) fn uplift(built: &Path, path: &Path) {
        if path.exists() {
            std::fs::remove_file(path).unwrap();
        }
        let copied = output(Command::new("cp").arg("-p").arg(built).arg(path));
        assert_eq!(copied.0, 0, "{}", copied.1);
    }

    /// Runs `binary --list` through the runner and returns its exit code.
    pub(crate) fn run(binary: &Path) -> i32 {
        let ran = output(Command::new("sh").arg(runner()).arg(binary).arg("--list"));
        assert!(
            ran.1.contains(": test"),
            "the binary ran and listed its tests: {}",
            ran.1
        );
        ran.0
    }

    /// Whether `binary`'s signature carries the hypervisor entitlement.
    pub(crate) fn entitled(binary: &Path) -> bool {
        let shown = output(
            Command::new("codesign")
                .args(["--display", "--entitlements", "-"])
                .arg(binary),
        );
        assert_eq!(shown.0, 0, "{}", shown.1);
        shown.1.contains(ENTITLEMENT)
    }
}

use support::{built, entitled, run, scratch, uplift};

#[test]
fn a_binary_is_signed_with_the_entitlement_before_it_runs() {
    let folder = scratch("first");
    let built = built(&folder);
    let binary = folder.join("binary");
    uplift(&built, &binary);
    assert_eq!(run(&binary), 0);
    assert!(entitled(&binary));
}

#[test]
fn a_binary_cargo_replaced_with_its_unsigned_build_is_signed_again() {
    let folder = scratch("replaced");
    let built = built(&folder);
    let binary = folder.join("binary");
    for round in 0..3 {
        // Each `cargo run`: the unsigned build comes back, as old as it ever was.
        uplift(&built, &binary);
        assert!(
            !entitled(&binary),
            "round {round}: replaced by the unsigned build"
        );
        assert_eq!(run(&binary), 0, "round {round}");
        assert!(entitled(&binary), "round {round}: signed before running");
    }
}

#[test]
fn a_binary_that_has_not_changed_is_not_signed_twice() {
    let folder = scratch("unchanged");
    let built = built(&folder);
    let binary = folder.join("binary");
    uplift(&built, &binary);
    assert_eq!(run(&binary), 0);
    let signed = std::fs::read(&binary).unwrap();
    let modified = std::fs::metadata(&binary).unwrap().modified().unwrap();
    assert_eq!(run(&binary), 0);
    // Signing again would rewrite the file.
    assert_eq!(
        std::fs::metadata(&binary).unwrap().modified().unwrap(),
        modified
    );
    assert_eq!(std::fs::read(&binary).unwrap(), signed);
    assert!(entitled(&binary));
}
