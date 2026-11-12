//! Drives `tools/fetch-refs.sh` with a stand-in `git`, so no network is touched, and checks it
//! against `docs/references.md` and `CLAUDE.md`.
//!
//! The script must never ask for a username or password (an agent or a CI job cannot answer, and
//! a hung prompt looks like a hang): every `git` it starts gets prompting switched off, and a
//! failed fetch ends the script with a message that names the source.

/// Helpers live in a `cfg(test)` module so clippy's allow-unwrap-in-tests setting covers them.
#[cfg(test)]
mod support {
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::process::{Command, Stdio};

    pub fn repo_file(path: &str) -> String {
        fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join(path)).unwrap()
    }

    /// An empty scratch folder for the test called `name`.
    fn scratch(name: &str) -> PathBuf {
        let folder = Path::new(env!("CARGO_TARGET_TMPDIR"))
            .join("fetch-refs")
            .join(name);
        let _ = fs::remove_dir_all(&folder);
        fs::create_dir_all(&folder).unwrap();
        folder
    }

    /// A stand-in for `git` that records every call (its arguments and the four variables that
    /// decide whether git may prompt) in `$FAKE_GIT_LOG`, creates the folder `init` and `clone`
    /// are asked for, answers `rev-parse HEAD` with the commit `$FAKE_GIT_HAVE` (`dir=commit ...`)
    /// gives that folder, and fails a `fetch` whose arguments end in `$FAKE_GIT_FAIL_COMMIT`.
    const FAKE_GIT: &str = r#"#!/bin/sh
echo "git $* | prompt=$GIT_TERMINAL_PROMPT askpass=$GIT_ASKPASS ssh_askpass=$SSH_ASKPASS gcm=$GCM_INTERACTIVE" >> "$FAKE_GIT_LOG"
case " $* " in
    *" rev-parse HEAD "*)
        for entry in $FAKE_GIT_HAVE; do
            case "$entry" in "$2="*) echo "${entry#*=}" ;; esac
        done
        exit 0 ;;
esac
case "$1" in
    init) mkdir -p "$3/.git" ;;
    clone) for last; do :; done; mkdir -p "$last/.git" ;;
esac
case " $* " in
    *" fetch "*) case "$*" in *"$FAKE_GIT_FAIL_COMMIT") exit 128 ;; esac ;;
esac
exit 0
"#;

    /// What a run of the script did.
    pub struct Run {
        pub code: i32,
        pub stdout: String,
        pub stderr: String,
        /// The stand-in git's log, one line per call.
        pub calls: Vec<String>,
        /// The folder the script was told to fill.
        pub refs: PathBuf,
    }

    /// Runs `tools/fetch-refs.sh` in a scratch folder named `name` with the stand-in git, extra
    /// environment `env`, no standard input, and `fail_commit` as the fetch to fail.
    pub fn run(name: &str, fail_commit: &str, env: &[(&str, &str)]) -> Run {
        run_with_checkouts(name, fail_commit, env, &[])
    }

    /// Like [`run`], with checkouts already in the refs folder: each `(folder, commit)` is a
    /// folder with a `.git` inside, whose `HEAD` the stand-in git reports as `commit`.
    pub fn run_with_checkouts(
        name: &str,
        fail_commit: &str,
        env: &[(&str, &str)],
        have: &[(&str, &str)],
    ) -> Run {
        let folder = scratch(name);
        let bin = folder.join("bin");
        fs::create_dir_all(&bin).unwrap();
        let git = bin.join("git");
        fs::write(&git, FAKE_GIT).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&git, fs::Permissions::from_mode(0o755)).unwrap();
        }
        let log = folder.join("git.log");
        fs::write(&log, "").unwrap();
        let refs = folder.join("refs");
        for (dir, _) in have {
            fs::create_dir_all(refs.join(dir).join(".git")).unwrap();
        }
        let have_list = have
            .iter()
            .map(|(dir, commit)| format!("{dir}={commit}"))
            .collect::<Vec<_>>()
            .join(" ");
        let path = format!(
            "{}:{}",
            bin.display(),
            std::env::var("PATH").unwrap_or_default()
        );
        let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("../tools/fetch-refs.sh");
        let mut command = Command::new("sh");
        command
            .arg(script)
            .env("PATH", path)
            .env("STARQUAKE_REFS", &refs)
            .env("FAKE_GIT_LOG", &log)
            .env("FAKE_GIT_FAIL_COMMIT", fail_commit)
            .env("FAKE_GIT_HAVE", have_list)
            .stdin(Stdio::null());
        for (key, value) in env {
            command.env(key, value);
        }
        let output = command.output().unwrap();
        Run {
            code: output
                .status
                .code()
                .expect("fetch-refs.sh was killed by a signal"),
            stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
            calls: fs::read_to_string(&log)
                .unwrap()
                .lines()
                .map(str::to_owned)
                .collect(),
            refs,
        }
    }

    /// The commit the script pins for the sparse Mesa checkout (`mesa_commit=`).
    pub fn mesa_commit() -> String {
        repo_file("tools/fetch-refs.sh")
            .lines()
            .find_map(|line| line.strip_prefix("mesa_commit="))
            .unwrap()
            .to_owned()
    }

    /// The sources the script fetches, from its `fetch_commit <dir> <url> <commit>` lines, plus
    /// the sparse Mesa checkout: (folder, url, commit).
    pub fn script_sources() -> Vec<(String, String, String)> {
        let script = repo_file("tools/fetch-refs.sh");
        let mut sources = Vec::new();
        for line in script.lines() {
            let line = line.trim().trim_start_matches("GIT_LFS_SKIP_SMUDGE=1 ");
            if let Some(rest) = line.strip_prefix("fetch_commit ") {
                let words: Vec<_> = rest.split_whitespace().collect();
                if let [dir, url, commit] = words[..] {
                    sources.push((dir.to_owned(), url.to_owned(), commit.to_owned()));
                }
            }
        }
        sources
    }

    /// The rows of the table in `docs/references.md`: (folder, commit prefix, licence).
    pub fn documented_sources() -> Vec<(String, String, String)> {
        repo_file("docs/references.md")
            .lines()
            .filter(|line| line.starts_with("| `"))
            .filter_map(|line| {
                let cells: Vec<_> = line.split('|').map(str::trim).collect();
                // "", folder, project, commit, licence, used for, "".
                let strip = |cell: &str| cell.trim_matches('`').to_owned();
                (cells.len() == 7).then(|| (strip(cells[1]), strip(cells[3]), cells[4].to_owned()))
            })
            .collect()
    }
}

use support::{
    documented_sources, mesa_commit, repo_file, run, run_with_checkouts, script_sources,
};

/// Exit status of a failed script (`exit 1`).
const EXIT_FAILURE: i32 = 1;

/// The fetch the stand-in git is told not to fail: no commit ends an argument list like this.
const NO_FAILURE: &str = "-";

/// What git is run with, every time: prompting off.
const NO_PROMPT: &str = "prompt=0 askpass=false ssh_askpass=false gcm=never";

#[test]
fn every_git_call_has_prompting_switched_off_whatever_the_caller_set() {
    // The caller's environment asks for the opposite; the script overrides it.
    let run = run(
        "no-prompt",
        NO_FAILURE,
        &[
            ("GIT_TERMINAL_PROMPT", "1"),
            ("GIT_ASKPASS", "/usr/bin/some-prompter"),
            ("SSH_ASKPASS", "/usr/bin/some-prompter"),
            ("GCM_INTERACTIVE", "always"),
        ],
    );
    assert_eq!(run.code, 0, "{}", run.stderr);
    // At least one call per source went through the stand-in.
    assert!(run.calls.len() >= script_sources().len());
    for call in &run.calls {
        assert!(call.ends_with(NO_PROMPT), "{call}");
    }
}

#[test]
fn a_clean_run_fetches_every_documented_source_at_its_documented_commit() {
    let run = run("clean", NO_FAILURE, &[]);
    assert_eq!(run.code, 0, "{}", run.stderr);
    assert_eq!(
        run.stdout,
        format!("references are in {}\n", run.refs.display())
    );
    let documented = documented_sources();
    let scripted = script_sources();
    // Mesa has its own sparse-checkout code in the script, so it is not a `fetch_commit` line.
    let mut folders: Vec<_> = scripted.iter().map(|(dir, _, _)| dir.as_str()).collect();
    folders.push("mesa-nouveau");
    folders.sort_unstable();
    let mut documented_folders: Vec<_> =
        documented.iter().map(|(dir, _, _)| dir.as_str()).collect();
    documented_folders.sort_unstable();
    assert_eq!(folders, documented_folders);
    for (dir, url, commit) in &scripted {
        let (_, prefix, _) = documented.iter().find(|(name, _, _)| name == dir).unwrap();
        assert!(
            commit.starts_with(prefix.as_str()),
            "{dir}: the script pins {commit}, docs/references.md {prefix}"
        );
        let remote = format!("git -C {dir} remote add origin {url} |");
        assert!(
            run.calls.iter().any(|call| call.starts_with(&remote)),
            "{dir} was not added with {url}"
        );
        let fetch = format!("git -C {dir} fetch -q --depth 1 origin {commit} |");
        assert!(
            run.calls.iter().any(|call| call.starts_with(&fetch)),
            "{dir} was not fetched at {commit}"
        );
    }
    // Mesa's commit is in its own fetch call.
    let (_, mesa_prefix, _) = documented
        .iter()
        .find(|(name, _, _)| name == "mesa-nouveau")
        .unwrap();
    assert!(
        run.calls
            .iter()
            .any(|call| call.contains(" fetch ") && call.contains(&format!(" {mesa_prefix}"))),
        "mesa-nouveau was not fetched at {mesa_prefix}"
    );
}

#[test]
fn a_failed_fetch_stops_the_script_and_names_the_source() {
    let sources = script_sources();
    let (dir, url, commit) = &sources[2];
    let run = run("fails", commit, &[]);
    assert_eq!(run.code, EXIT_FAILURE);
    assert_eq!(
        run.stderr,
        format!(
            "fetch-refs: could not fetch {dir} ({url}) at {commit}. This script never asks for a \
             username or password: check the network, the address and the commit, and that the \
             source is public.\n"
        )
    );
    assert_eq!(run.stdout, "");
    // The two before it were fetched, the ones after it were not tried.
    for (earlier, _, commit) in &sources[..2] {
        let fetch = format!("git -C {earlier} fetch -q --depth 1 origin {commit} |");
        assert!(
            run.calls.iter().any(|call| call.starts_with(&fetch)),
            "{earlier}"
        );
    }
    for (later, _, _) in &sources[3..] {
        let prefix = format!("git -C {later} ");
        assert!(
            !run.calls.iter().any(|call| call.starts_with(&prefix)),
            "{later}"
        );
    }
}

#[test]
fn a_failed_mesa_fetch_names_mesa() {
    let mesa = documented_sources()
        .into_iter()
        .find(|(name, _, _)| name == "mesa-nouveau")
        .unwrap();
    let commit = mesa_commit();
    let commit = commit.as_str();
    assert!(commit.starts_with(&mesa.1));
    let run = run("mesa-fails", commit, &[]);
    assert_eq!(run.code, EXIT_FAILURE);
    assert!(
        run.stderr.starts_with(&format!(
            "fetch-refs: could not fetch mesa-nouveau (https://gitlab.freedesktop.org/mesa/mesa.git) at {commit}."
        )),
        "{}",
        run.stderr
    );
}

#[test]
fn atmosphere_is_pinned_as_a_reference_for_facts_only() {
    let documented = documented_sources();
    let (_, _, licence) = documented
        .iter()
        .find(|(name, _, _)| name == "atmosphere")
        .expect("docs/references.md lists atmosphere");
    assert_eq!(licence, "GPL-2.0-only");
    let sources = script_sources();
    let (_, url, commit) = sources
        .iter()
        .find(|(dir, _, _)| dir == "atmosphere")
        .expect("fetch-refs.sh fetches atmosphere");
    assert_eq!(url, "https://github.com/Atmosphere-NX/Atmosphere.git");
    assert_eq!(commit.len(), 40);
    let references = repo_file("docs/references.md");
    assert!(references.contains("**Atmosphère is for facts only.**"));
    let claude = repo_file("CLAUDE.md");
    assert!(claude.contains("Atmosphère"));
    assert!(claude.contains("GPL-2.0-only"));
}

#[test]
fn libnx_is_listed_with_its_isc_licence_in_the_rules() {
    let licensing = repo_file("CLAUDE.md");
    let licensing = licensing.split("## Licensing").nth(1).unwrap();
    assert!(licensing.contains("libnx"), "{licensing}");
    assert!(licensing.contains("ISC"), "{licensing}");
    let documented = documented_sources();
    let (_, _, licence) = documented
        .iter()
        .find(|(name, _, _)| name == "libnx")
        .unwrap();
    assert_eq!(licence, "ISC");
}

#[test]
fn the_arm_reference_manual_is_listed_as_the_source_code_comments_cite() {
    let references = repo_file("docs/references.md");
    assert!(references.contains("DDI 0487"));
    assert!(references.contains("https://developer.arm.com/documentation/ddi0487/latest"));
}

#[test]
fn a_checkout_already_at_its_pinned_commit_needs_no_network() {
    // Every source is present at its pin and every fetch would fail (as one does when a source
    // has gone away): the script succeeds without starting a single fetch.
    let mut have: Vec<(String, String)> = script_sources()
        .into_iter()
        .map(|(dir, _, commit)| (dir, commit))
        .collect();
    have.push(("mesa-nouveau".to_owned(), mesa_commit()));
    let have: Vec<(&str, &str)> = have.iter().map(|(d, c)| (d.as_str(), c.as_str())).collect();
    let gone = script_sources().last().unwrap().2.clone();
    let run = run_with_checkouts("up-to-date", &gone, &[], &have);
    assert_eq!(run.code, 0, "{}", run.stderr);
    assert!(
        run.calls.iter().all(|call| !call.contains(" fetch ")),
        "{:?}",
        run.calls
    );
}

#[test]
fn a_checkout_at_another_commit_is_fetched_again() {
    let sources = script_sources();
    let (dir, _, commit) = &sources[0];
    let elsewhere = "0".repeat(40);
    let run = run_with_checkouts("moved", "-", &[], &[(dir.as_str(), elsewhere.as_str())]);
    assert_eq!(run.code, 0, "{}", run.stderr);
    let fetch = format!("git -C {dir} fetch -q --depth 1 origin {commit} |");
    assert!(run.calls.iter().any(|call| call.starts_with(&fetch)));
    // The existing folder is reused, not initialised again.
    assert!(
        run.calls
            .iter()
            .all(|call| !call.starts_with(&format!("git init -q {dir} ")))
    );
}
