# starquake

A Switch 2 emulator in Rust, for macOS (Apple Silicon) and Linux. GPLv3. Most code is written by
agents, so the repo is built to make changes checkable.

## Before you say a change is done

Run `cargo xtask ci` (or `just ci`): fmt check, `cargo deny check` (licences, sources, advisories;
`deny.toml`), clippy with `-D warnings` and the workspace lint set, nextest, doctests. CI runs
exactly this. The deny step is skipped locally, with a notice, when `cargo-deny` isn't installed
(`cargo install cargo-deny --locked`); under CI a missing `cargo-deny` fails the gate. It fetches
the advisory database, so it needs network access.

Platform-specific code must still compile (or be `cfg`-gated) on Linux; check with
`cargo clippy --workspace --all-targets --target x86_64-unknown-linux-gnu -- -D warnings`.

## Where things are

- `docs/roadmap.md`: milestones, waves, and how work is scheduled. The backlog is GitHub issues.
- `docs/architecture.md`: how the pieces fit, threads, and dependency direction. Read it first.
- `docs/decision-log.md`: the owner's decisions and why. Follow them; propose changes, don't drift.
- `docs/port-map.md`: layer by layer, what to port from Eden (and its Switch 2 forks) and what to
  redesign. Check it before porting anything.
- `docs/references.md`: the pinned reference sources and how to fetch them
  (`tools/refs-path.sh` prints their folder).
- `docs/switch2-eden-diff-scoping.md`: what Switch 2 support actually requires, from Eden-NX2's
  149-row blocker log and PR #4588. Cite its rows when implementing protocol behaviour.
- `docs/oboromi-evaluation.md`: why this is a fresh repo.
- `tools/hvprobe`: measured Hypervisor.framework behaviour (`tools/hvprobe/README.md`).

## Architecture rules

- Dependencies point one way: `memory` → `cpu`, `ipc` → `kernel` → `services` → `gsv` → `gpu` →
  `shader`, `render`. `core` wires everything; frontends depend only on `core`. No cycles, and
  no reaching around a layer.
- Guest-controlled values are untrusted: checked arithmetic and bounds on anything derived from
  guest memory, file contents or IPC words. A malformed input is an `Err`, never a panic.
- Protocol facts (command ids, struct layouts) need a source in a comment: a scoping-doc row, a
  reference path and commit, or a trace. Mark guesses as guesses.

## Code style

- **Zero lint warnings.** Nothing counts as pre-existing; fix lints in files you touch. Run
  `cargo clippy --fix` before fixing by hand.
- **`#[expect(lint, reason = "...")]`, never `#[allow]`.** A suppression needs a real reason, and
  new ones should be rare enough to mention in the PR.
- **No `unwrap`, `expect`, `panic!`, `todo!`, `dbg!`, or printing in library code.** Tests may
  use them. Log with `tracing`.
- **Errors use `thiserror`.** No hand-written `Display` + `Error` impls, no `anyhow` in libraries.
- **`unsafe` is denied workspace-wide.** Only crates that must talk to the host or run guest code
  opt in, with a crate-level `#![expect(unsafe_code, reason = "...")]`, and every `unsafe` block
  carries a `// SAFETY:` comment.
- **Dependencies** are declared once in the root `[workspace.dependencies]` with a version, and
  used as `dep.workspace = true`. Check the latest published version before adding one, and say
  why it's needed in the PR. Prefer pure-Rust crates; no C/C++ build steps in the default build.
- Every public item gets a doc comment. `//!` docs at the top of each module say what it owns.
- Modules are `foo.rs` + `foo/`, not `foo/mod.rs`.

## Tests

- Write the test first and watch it fail before making it pass.
- Deterministic: explicit seeds, and no `HashMap` iteration where order affects output.
- Tests import the shipping constants; they don't restate literals.
- Unit tests in `#[cfg(test)] mod tests`; cross-module tests in `tests/`. `proptest` for parsers
  and encoders, `insta` for snapshots.
- Hypervisor tests need a real Mac. They skip themselves only when the `CI` environment variable
  is set (hosted runners can't create VMs); anywhere else, failing to create a VM is a failure.
- Never commit keys, firmware, game files or dumps derived from them. Fixtures are synthetic or
  built from homebrew.

## Git and GitHub

- Conventional Commits (`feat(ipc): ...`, `fix(loader): ...`, `docs: ...`, `chore: ...`).
  Commit at every green checkpoint; one logical change per commit. Run `just setup` once per clone:
  it points `core.hooksPath` at `.githooks`, whose `commit-msg` hook rejects non-conventional subjects. CI
  applies the same check to PR titles.
- One issue → one branch → one PR, with an honest `Closes #N`. Findings become issues; design
  questions become `needs-design` issues.
- Implementation PRs may merge on a green gate and review. Design and spec PRs wait for the owner.
- Never `git stash` (worktrees share one stash). Stage explicit paths.
- No local paths, usernames or unrelated personal-project details in code, docs, issues, PRs or
  commit messages.

## Licensing

Code may be ported from GPL projects (Eden, oboromi, rapid-arm-emu) with attribution in the file
header; MIT sources (Mesa) likewise. Don't use leaked or NDA material.
