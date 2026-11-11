//! `cargo xtask` must run the `xtask` of the checkout it is invoked in, even when several
//! checkouts share one `CARGO_TARGET_DIR` (as the worktrees of a wave do), while the cargo
//! commands that `xtask` spawns keep using the shared folder.
//!
//! The test builds two scratch checkouts whose `xtask` sources differ, gives both the alias from
//! this repository's real `.cargo/config.toml`, points both at one shared target folder, and runs
//! them alternately. With the alias as it was before (`run --package xtask --quiet --`, no
//! `--target-dir`) the second build overwrites `target/debug/xtask`, and the first checkout then
//! runs the second checkout's binary.

/// Helpers live in a `cfg(test)` module so clippy's allow-unwrap-in-tests setting covers them.
#[cfg(test)]
mod support {
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::process::Command;

    /// What a scratch `xtask` prints: its marker, the shared folder it was handed, and where its
    /// own executable lives.
    #[derive(Debug)]
    pub struct Report {
        pub marker: String,
        pub target_dir_env: String,
        pub exe: PathBuf,
    }

    /// The real repository's `xtask` alias, read from its `.cargo/config.toml`.
    pub fn real_alias() -> String {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../.cargo/config.toml");
        let text = fs::read_to_string(path).unwrap();
        let config: toml::Table = text.parse().unwrap();
        config["alias"]["xtask"].as_str().unwrap().to_owned()
    }

    /// A scratch folder unique to this test process, so concurrent test runs that share a
    /// `CARGO_TARGET_TMPDIR` never touch each other's checkouts.
    pub fn scratch_root() -> PathBuf {
        let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
            .join(format!("per-checkout-{}", std::process::id()));
        drop(fs::remove_dir_all(&root));
        fs::create_dir_all(&root).unwrap();
        root
    }

    /// Writes a minimal workspace at `root/name` whose `xtask` prints `name`, the
    /// `CARGO_TARGET_DIR` it sees and its own executable path, and which carries `alias`.
    pub fn checkout(root: &Path, name: &str, alias: &str) -> PathBuf {
        let dir = root.join(name);
        fs::create_dir_all(dir.join("xtask/src")).unwrap();
        fs::create_dir_all(dir.join(".cargo")).unwrap();
        fs::write(
            dir.join("Cargo.toml"),
            "[workspace]\nresolver = \"3\"\nmembers = [\"xtask\"]\n",
        )
        .unwrap();
        fs::write(
            dir.join("xtask/Cargo.toml"),
            "[package]\nname = \"xtask\"\nversion = \"0.0.0\"\nedition = \"2024\"\n",
        )
        .unwrap();
        fs::write(
            dir.join("xtask/src/main.rs"),
            format!(
                "fn main() {{\n    println!(\"{name}\");\n    \
                 println!(\"{{}}\", std::env::var(\"CARGO_TARGET_DIR\").unwrap_or_default());\n    \
                 println!(\"{{}}\", std::env::current_exe().unwrap().display());\n}}\n"
            ),
        )
        .unwrap();
        fs::write(
            dir.join(".cargo/config.toml"),
            format!("[alias]\nxtask = {alias:?}\n"),
        )
        .unwrap();
        fs::copy(
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../rust-toolchain.toml"),
            dir.join("rust-toolchain.toml"),
        )
        .unwrap();
        dir
    }

    /// Runs `cargo xtask` in `checkout` with `shared` as the shared build folder.
    pub fn run_xtask(checkout: &Path, shared: &Path) -> Report {
        let output = Command::new(env!("CARGO"))
            .arg("xtask")
            .current_dir(checkout)
            .env("CARGO_TARGET_DIR", shared)
            .env("CARGO_INCREMENTAL", "0")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "cargo xtask failed in {}: {}",
            checkout.display(),
            String::from_utf8_lossy(&output.stderr)
        );
        let stdout = String::from_utf8(output.stdout).unwrap();
        let mut lines = stdout.lines();
        Report {
            marker: lines.next().unwrap().to_owned(),
            target_dir_env: lines.next().unwrap().to_owned(),
            exe: PathBuf::from(lines.next().unwrap()),
        }
    }
}

use std::path::Path;

use support::{checkout, real_alias, run_xtask, scratch_root};

#[test]
fn each_checkout_sharing_a_target_dir_runs_its_own_xtask() {
    let root = scratch_root();
    let shared = root.join("shared-target");
    let alias = real_alias();
    let alpha = checkout(&root, "alpha", &alias);
    let beta = checkout(&root, "beta", &alias);

    // Alternate so that whichever checkout built last cannot be what both of them run.
    for _round in 0..2 {
        assert_eq!(run_xtask(&alpha, &shared).marker, "alpha");
        assert_eq!(run_xtask(&beta, &shared).marker, "beta");
    }
}

#[test]
fn the_xtask_binary_stays_in_its_checkout_and_spawned_cargo_keeps_the_shared_folder() {
    let root = scratch_root();
    let shared = root.join("shared-target");
    let alpha = checkout(&root, "alpha", &real_alias());

    let report = run_xtask(&alpha, &shared);
    // /tmp is a symlink on macOS, so compare canonical paths.
    let alpha = alpha.canonicalize().unwrap();
    // Nothing builds in the shared folder when the alias works; create it so it can be resolved.
    std::fs::create_dir_all(&shared).unwrap();
    let shared = shared.canonicalize().unwrap();
    let exe = report.exe.canonicalize().unwrap();

    assert!(
        exe.starts_with(&alpha),
        "{} is not inside {}",
        exe.display(),
        alpha.display()
    );
    assert!(
        !exe.starts_with(&shared),
        "{} is inside the shared folder {}",
        exe.display(),
        shared.display()
    );
    // Commands that xtask spawns inherit this variable, so they keep building in the shared folder.
    assert_eq!(
        Path::new(&report.target_dir_env).canonicalize().unwrap(),
        shared
    );
}
