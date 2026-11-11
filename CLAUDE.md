# starquake

A Switch 2 emulator in Rust, for macOS (Apple Silicon) and Linux. GPLv3. Most code is written by
agents, so the repo is built to make changes checkable.

## Working mode (until milestone 1)

The lead session runs autonomously until a program runs end to end (decision log, 2026-11-11):
it decides design questions, amends specs and vets `docs/proposals.md` itself, and records every
such choice for the owner to review afterwards. It stops only for things that need the owner
(hardware, passwords, anything irreversible outside the repo). The route is **skeleton first**:
get the thinnest end-to-end run working, then widen it, so the pieces are proven to fit.

- Judgement-heavy or single-threaded work (integration, foundations, bring-up) goes to one agent
  on the strongest model. Parallel waves are for well-specified ports.
- Every PR gets two independent reviews: one for fidelity to the spec and the reference, one
  that runs the code and tries to break its tests.

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
- `docs/specs/`: the approved designs (execution model, kernel, service framework, observability
  and replay, test strategy). Code follows them; changing one needs the owner's approval.
- `docs/proposals.md`: work and house rules that agents have proposed and the owner hasn't vetted
  yet. Nothing in it is a commitment.
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

- Dependencies point one way: `memory` → `cpu` → `kernel` → `services` → `gsv` → `gpu` →
  `shader`, `render`. `ipc` and `trace` are leaves (no workspace dependencies) that the layers
  above depend on; `testkit` is a dev-dependency only and depends on none of our crates. `core` wires everything; frontends depend only on `core`. No cycles, and no
  reaching around a layer.
- Guest-controlled values are untrusted: checked arithmetic and bounds on anything derived from
  guest memory, file contents or IPC words. A malformed input is an `Err`, never a panic.
  Check arithmetic for the whole extent written or read (start, end, trailing data), not just
  the start, and bound loops over untrusted structures by the structure actually walked.
- Compatibility comes first. A design may defer a feature (server-side IPC, real system modules,
  homebrew services), but must not rule it out, and must say how it could be added later.
- Protocol facts (command ids, struct layouts) need a source in a comment: a scoping-doc row, a
  reference path and commit, or a trace. Mark guesses as guesses, and say what would confirm them.
- When mirroring reference behaviour, cite the exact check and its constant. Where starquake is
  stricter or looser than the reference, list the divergence in the module docs and pin it with
  a test. No blanket claims ("every failure", "all replies"): state the exact set or condition.

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
- Don't duplicate logic across sibling modules; extract one `pub(crate)` helper and call it from
  both.

## Tests

- Write the test first and watch it fail before making it pass. A test added for a fix is run
  against the unfixed code first; say in the PR that it failed.
- Deterministic: explicit seeds, and no `HashMap` iteration where order affects output. Every
  `proptest!` block sets a fixed `rng_seed`.
- Tests import the shipping constants; they don't restate literals. Expectations are derived from
  the shipping bound, with boundary cases exactly at the limit.
- A parser or decoder that reads several ordered fields or lists gets a generated
  build-then-parse round-trip test that fails if any two reads are swapped.
- Negative tests assert the exact error (variant, exit code, message), so a missing or broken
  implementation can't pass them.
- Never hide a flaky test behind retries: reproduce it under plain `cargo test` as well as
  nextest, find the root cause, and document a platform quirk where it is one.
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
- One issue → one branch → one PR, with an honest `Closes #N`.
- **Don't file GitHub issues for work you propose.** Follow-ups, findings and ideas go into
  `docs/proposals.md`; the owner promotes the ones worth doing to issues. Work that belongs to
  an existing issue goes there as a comment.
- Implementation PRs may merge on a green gate and review. Design and spec PRs wait for the owner.
- Never merge before the PR's own CI checks have been reported and passed; a green local gate is
  not enough, and "no checks reported" means wait. Verify PR-body claims (test counts, how tests
  were run) before stating them.
- Never `git stash` (worktrees share one stash). Stage explicit paths.
- No local paths, usernames or unrelated personal-project details in code, docs, issues, PRs or
  commit messages.

## Licensing

Code may be ported from GPL projects (Eden, oboromi, rapid-arm-emu) with attribution in the file
header; MIT sources (Mesa) likewise. Don't use leaked or NDA material.
