//! Drives `.githooks/commit-msg`, the one Conventional Commits checker shared by the local hook
//! (run by git on every commit) and the PR-title check in `.github/workflows/pr-title.yml`
//! (run as `commit-msg --title FILE`).

use std::path::PathBuf;
use std::process::Command;

/// Conventional Commits types the hook accepts (the Angular set, which the spec's examples use).
const TYPES: [&str; 11] = [
    "feat", "fix", "docs", "style", "refactor", "perf", "test", "build", "ci", "chore", "revert",
];

/// Helpers live in a `cfg(test)` module so clippy's allow-unwrap-in-tests setting covers them.
#[cfg(test)]
mod support {
    use std::fs;
    use std::path::PathBuf;
    use std::process::Command;
    use std::sync::atomic::{AtomicUsize, Ordering};

    pub fn hook_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../.githooks/commit-msg")
    }

    /// Exit status of an accepted subject (`.githooks/commit-msg`: `exit 0`).
    pub const EXIT_ACCEPTED: i32 = 0;
    /// Exit status of a subject that is not in the Conventional Commits format (`exit 1`).
    pub const EXIT_REJECTED: i32 = 1;
    /// Exit status of a usage error or an unreadable message file (`exit 2`). A missing or
    /// unrunnable script exits 127 instead, which no assertion here accepts.
    pub const EXIT_USAGE: i32 = 2;

    /// Runs the hook on `message`; `title` selects the PR-title mode. Returns (exit code, stderr).
    pub fn run(message: &str, title: bool) -> (i32, String) {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let file = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!(
            "commit-msg-{}-{}.txt",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::write(&file, message).unwrap();
        let mut command = Command::new("sh");
        command.arg(hook_path());
        if title {
            command.arg("--title");
        }
        let output = command.arg(&file).output().unwrap();
        fs::remove_file(&file).unwrap();
        (
            output.status.code().expect("hook was killed by a signal"),
            String::from_utf8_lossy(&output.stderr).into_owned(),
        )
    }

    /// Runs the hook with `command`'s arguments and returns (exit code, stderr).
    pub fn run_raw(mut command: Command) -> (i32, String) {
        let output = command.output().unwrap();
        (
            output.status.code().expect("hook was killed by a signal"),
            String::from_utf8_lossy(&output.stderr).into_owned(),
        )
    }

    /// Whether `message` is accepted as a commit message (exit 0).
    pub fn accepted(message: &str) -> bool {
        run(message, false).0 == EXIT_ACCEPTED
    }

    /// Asserts the hook rejects `message` the way a format rejection does: exit exactly 1 with the
    /// explanation on stderr, so a missing or unrunnable script cannot pass for a rejection.
    pub fn assert_rejected(message: &str, title: bool) {
        let (code, stderr) = run(message, title);
        assert_eq!(
            code, EXIT_REJECTED,
            "{message:?} (title: {title}): {stderr}"
        );
        assert!(
            stderr.contains("Conventional Commits"),
            "{message:?} (title: {title}): stderr: {stderr}"
        );
    }
}

use support::{
    EXIT_ACCEPTED, EXIT_REJECTED, EXIT_USAGE, accepted, assert_rejected, hook_path, run, run_raw,
};

#[test]
fn accepts_every_listed_type() {
    for ty in TYPES {
        let message = format!("{ty}: do the thing\n");
        assert!(accepted(&message), "rejected {message:?}");
    }
}

#[test]
fn accepts_scopes_and_breaking_marker() {
    for message in [
        "feat(ipc): add the command table\n",
        "fix(cpu,memory): align the walk\n",
        "chore(xtask): x\n",
        "feat!: drop the old API\n",
        "feat(kernel)!: drop the old API\n",
        "docs(a/b.c_d-e): scoped\n",
    ] {
        assert!(accepted(message), "rejected {message:?}");
    }
}

#[test]
fn accepts_body_footer_and_missing_trailing_newline() {
    assert!(accepted(
        "fix(loader): bound the segment\n\nLonger text.\n\nCloses #1\n"
    ));
    assert!(accepted("fix: no newline at all"));
    assert!(accepted("fix: crlf line ending\r\n"));
}

#[test]
fn ignores_git_comment_and_leading_blank_lines() {
    assert!(accepted(
        "\n# Please enter the commit message\nfix: real subject\n# more\n"
    ));
}

#[test]
fn rejects_malformed_subjects() {
    for message in [
        "",
        "\n\n",
        "# only a comment\n",
        "Add a thing\n",
        "feat add a thing\n",
        "feat:no space\n",
        "feat:  two spaces\n",
        "feat: \n",
        "feat:\n",
        "Feat: capitalised type\n",
        "feature: unknown type\n",
        "feat(): empty scope\n",
        "feat(scope: unclosed\n",
        "feat (scope): space before scope\n",
        "feat(sc ope): space in scope\n",
        "feat!!: double bang\n",
        "feat(scope)!!: double bang\n",
        " feat: leading space\n",
        "wip: unknown type\n",
    ] {
        assert_rejected(message, false);
    }
}

#[test]
fn only_the_first_message_line_is_judged_as_the_subject() {
    assert_rejected("not conventional\n\nfix: in the body\n", false);
}

#[test]
fn rejection_explains_the_expected_format() {
    let (code, stderr) = run("Add a thing\n", false);
    assert_eq!(code, EXIT_REJECTED);
    assert!(stderr.contains("Conventional Commits"), "stderr: {stderr}");
    assert!(stderr.contains("Add a thing"), "stderr: {stderr}");
}

#[test]
fn git_generated_subjects_pass_only_as_commit_messages() {
    for message in [
        "Merge pull request #31 from owner/branch\n",
        "Merge branch 'topic' into main\n",
        "Merge branches 'a' and 'b'\n",
        "Merge remote-tracking branch 'origin/main' into topic\n",
        "Merge tag 'v1.0' into main\n",
        "Merge commit 'abc1234' into main\n",
        "fixup! feat: something\n",
        "squash! feat: something\n",
        "amend! feat: something\n",
    ] {
        assert!(accepted(message), "rejected {message:?}");
        assert_rejected(message, true);
    }
}

/// The titles GitHub generates for a revert PR: `Revert "<title of the reverted PR>"`, where the
/// reverted PR's own title is conventional. A revert of a revert nests the same wrapper.
const GITHUB_REVERT_TITLES: [&str; 5] = [
    "Revert \"feat: something\"",
    "Revert \"fix(cpu,memory): align the walk\"",
    "Revert \"feat(kernel)!: drop the old API\"",
    "Revert \"Revert \"feat: something\"\"",
    "Revert \"Revert \"Revert \"chore(xtask): x\"\"\"",
];

#[test]
fn github_revert_titles_pass_as_titles_and_as_commit_messages() {
    for title in GITHUB_REVERT_TITLES {
        let message = format!("{title}\n");
        assert!(accepted(&message), "rejected as a commit: {message:?}");
        let (code, stderr) = run(&message, true);
        assert_eq!(
            code, EXIT_ACCEPTED,
            "rejected as a title: {message:?}: {stderr}"
        );
    }
}

#[test]
fn a_revert_commit_message_may_carry_gits_body() {
    let message = "Revert \"feat: something\"\n\nThis reverts commit 0123abc.\n";
    assert!(accepted(message));
    assert_eq!(run(message, true).0, EXIT_ACCEPTED);
}

#[test]
fn revert_titles_must_wrap_a_conventional_title_exactly() {
    for message in [
        // The wrapped title is not conventional.
        "Revert \"Add a thing\"\n",
        "Revert \"Revert \"Add a thing\"\"\n",
        "Revert \"\"\n",
        "Revert \"Revert \"\"\n",
        // Not the generated form: unclosed, trailing text, lower case, no space, no quotes.
        "Revert \"feat: something\n",
        "Revert \"feat: something\" (#12)\n",
        "Revert \"feat: something\" and more\n",
        "revert \"feat: something\"\n",
        "Revert\"feat: something\"\n",
        "Revert feat: something\n",
        "Reapply \"feat: something\"\n",
        // Unbalanced nesting.
        "Revert \"Revert \"feat: something\"\n",
    ] {
        assert_rejected(message, true);
    }
}

#[test]
fn hand_written_merge_subjects_are_rejected() {
    for message in [
        "Merge sort: hand written\n",
        "Merge the two modules\n",
        "Merge\n",
        "Merge branch\n",
        "Merge pull request without a number\n",
        "Merged branch 'x'\n",
        "Merge pull request: hand written\n",
    ] {
        assert_rejected(message, false);
        assert_rejected(message, true);
    }
}

#[test]
fn title_mode_applies_the_same_rule() {
    let (code, _) = run("feat(ci): path-filtered macOS job", true);
    assert_eq!(code, EXIT_ACCEPTED);
    assert_rejected("Add a macOS job", true);
    assert_rejected("", true);
}

#[test]
fn missing_file_is_a_usage_error() {
    let mut command = Command::new("sh");
    command
        .arg(hook_path())
        .arg(PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("does-not-exist"));
    let (code, stderr) = run_raw(command);
    assert_eq!(code, EXIT_USAGE, "stderr: {stderr}");
    assert!(stderr.contains("cannot read"), "stderr: {stderr}");
}

#[test]
fn missing_argument_is_a_usage_error() {
    let mut command = Command::new("sh");
    command.arg(hook_path());
    let (code, stderr) = run_raw(command);
    assert_eq!(code, EXIT_USAGE, "stderr: {stderr}");
    assert!(stderr.contains("usage"), "stderr: {stderr}");
}

#[test]
fn title_flag_without_a_file_is_a_usage_error() {
    let mut command = Command::new("sh");
    command.arg(hook_path()).arg("--title");
    let (code, stderr) = run_raw(command);
    assert_eq!(code, EXIT_USAGE, "stderr: {stderr}");
    assert!(stderr.contains("usage"), "stderr: {stderr}");
}
