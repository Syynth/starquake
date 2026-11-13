# starquake

A Switch 2 emulator in Rust, for macOS (Apple Silicon) and Linux. GPLv3. Most code is written by
agents, so the repo is built to make changes checkable.

## Working mode (toward milestone 2)

Milestone 1 is reached. The owner extended the lead session's delegation toward milestone 2 for
**implementation only** (decision log, 2026-11-12): the lead files implementation issues that
follow approved specs, runs them in parallel lanes, and merges their PRs on a green gate and two
reviews. Design questions, spec changes and design PRs wait for the owner. The lead stops for
those and for anything that needs the owner (hardware, passwords, anything irreversible outside
the repo), and records its own choices in `docs/proposals.md` and the decision log.

- Judgement-heavy or single-threaded work (integration, foundations, bring-up) goes to one agent,
  not a wave. Parallel waves are for well-specified ports.
- Build, fix and review agents run on Opus (decision log, 2026-11-12). Fable is used sparingly
  and only when the owner's allowance has room; where a Fable review was planned, two Opus
  reviews stand in.
- Every PR gets two independent reviews: one for fidelity to the spec and the reference, one
  that runs the code and tries to break its tests.

## Before you say a change is done

Run `cargo xtask ci` (or `just ci`) from the repository root: fmt check, an unused-dependency
check (the root `[workspace.dependencies]`, the dev, build and target tables, then `cargo machete`
per crate), the host-access check (below), `cargo deny check`
(licences, sources, advisories; `deny.toml`), clippy with `-D warnings` and the workspace lint
set, nextest, doctests. CI runs exactly this.

- The unused-dependency check covers every dependency table. `cargo machete` reads only each
  crate's top-level `[dependencies]` (0.9.2 does not look at `[dev-dependencies]`,
  `[build-dependencies]` or `[target.*]` tables); the gate's `table-deps` step covers those with
  a text search: a dependency counts as used when its crate name (`-` written as `_`) appears as
  a word in the crate's `.rs` files, or, for a build dependency, in its build script. Both
  are checked against `xtask/unused-deps.toml`. `ignored = [...]` under
  `[package.metadata.cargo-machete]` or `[workspace.metadata.cargo-machete]` fails the gate:
  the reasoned list is `xtask/unused-deps.toml`.
- `cargo-deny` and `cargo-machete` are skipped locally, with a notice, when not installed
  (`cargo install cargo-deny@0.19.8 cargo-machete@0.9.2 --locked`); under CI a missing one fails
  the gate.
- The gate's plugins (`cargo-nextest`, `cargo-deny`, `cargo-machete`) are pinned in
  `xtask/tool-versions.toml`, so a tool release cannot turn `main` red. CI installs exactly those
  versions (`.github/workflows/ci.yml`; a test fails when the workflow or the install command
  above differs from the file). The gate compares what is installed with the pins: a different
  version is a notice locally and a failure under CI. To move a pin, change the file, the
  workflow and this command in one commit.
- A dependency that is declared ahead of the task that uses it, or that the text search cannot
  see used (a library name that differs from the package name), goes in
  `xtask/unused-deps.toml` with a reason (`[[allow]]` for `[dependencies]`, `[[allow_tables]]`
  for the other tables). The list is exact: an entry that is no longer needed fails the gate too,
  so delete it in the same change that starts using the dependency.
- `cargo deny` fetches the advisory database, so the gate needs network access. Offline, run
  `cargo xtask ci --offline` (or set `STARQUAKE_OFFLINE=1`): only the advisory check is skipped,
  and the output says so. CI refuses the switch.
- The `host-access` step reads the non-test code of `crates/kernel`, `crates/services` and
  `crates/gsv` and fails on a direct use of the host clock (`Instant::now`, `SystemTime::now`),
  `std::fs`, `std::net`, thread sleeping or a random-number source (`rand`, `getrandom`,
  `fastrand`, a `RandomState`). Read these through the injectable `Env` instead, or replay breaks
  silently (`docs/specs/observability.md`). Test code is exempt, and only that: `#[cfg(test)]`,
  `#[cfg(all(test, ..))]`, `#[test]`-style functions and `#![cfg(test)]` (not `any(test, ..)`);
  there is no escape comment. The step is a text search, not a sandbox: the module docs of
  `xtask/src/host_access.rs` list what it cannot see.
- `cargo xtask ci` moves to the workspace root itself, so it also works from a subfolder (the
  alias then builds its binary in that folder's own `target/`, which is git-ignored).
- `cargo xtask` builds its own binary in `target/xtask` inside the checkout, so worktrees that
  share one `CARGO_TARGET_DIR` each run their own `xtask` (see `docs/roadmap.md`).

Platform-specific code must still compile (or be `cfg`-gated) on Linux; check with
`cargo clippy --workspace --all-targets --target x86_64-unknown-linux-gnu -- -D warnings`.

## Where things are

- `docs/roadmap.md`: milestones, waves, and how work is scheduled. The backlog is GitHub issues.
- `docs/specs/`: the approved designs (execution model, kernel, service framework, observability
  and replay, test strategy, and the graphics half: `graphics-server.md`, `gpu-protocol.md`,
  `shader-pipeline.md`, `render-backend.md`). Code follows them; changing one needs the owner's
  approval. `graphics-review-guide.md` records how the four graphics specs were reconciled.
- `docs/proposals.md`: work and house rules that agents have proposed and the owner hasn't vetted
  yet. Nothing in it is a commitment.
- `docs/architecture.md`: how the pieces fit, threads, and dependency direction. Read it first.
- `docs/decision-log.md`: the owner's decisions and why. Follow them; propose changes, don't drift.
- `docs/port-map.md`: layer by layer, what to port from Eden (and its Switch 2 forks) and what to
  redesign. Check it before porting anything.
- `docs/references.md`: the pinned reference sources and how to fetch them
  (`tools/refs-path.sh` prints their folder).
- `docs/reference-emulator.md`: the locally built reference emulator whose traces starquake is
  compared with, what it is trusted for, and how to run it (`tools/oracle/`).
- `docs/switch2-eden-diff-scoping.md`: what Switch 2 support actually requires, from Eden-NX2's
  149-row blocker log and PR #4588. Cite its rows when implementing protocol behaviour.
- `docs/oboromi-evaluation.md`: why this is a fresh repo.
- `tools/hvprobe`: measured Hypervisor.framework behaviour (`tools/hvprobe/README.md`).

## Architecture rules

- Dependencies point one way: `memory` → `cpu` → `kernel` → `services` → `gsv` → `gpu` →
  `shader`, `render`. `shader` and `render` do not depend on each other: `gpu` carries the shader
  crate's naga module to the render backend. `ipc`, `trace` and `layout` (block-linear tiling)
  are leaves (no workspace dependencies) that the layers above depend on; `testkit` is a
  dev-dependency only and depends on none of our crates. A render backend crate
  (`render-wgpu`, later direct backends) depends only on `render`, is the only crate that names
  its host API's types, and is chosen by `core`. `core` wires everything; frontends depend only
  on `core`. No cycles, and no reaching around a layer.
- Kernel and service code (`kernel`, `services`, `gsv`) is replayable: the clock, random numbers
  and host file and network I/O come only through the injectable `Env`, never from `std`
  directly. The gate's `host-access` step catches the direct calls it lists (`xtask/src/host_access.rs`)
  and nothing else: method calls such as `.elapsed()` and `Path::exists`, aliases and other
  clock or random crates are for review to catch.
- Guest-controlled values are untrusted: checked arithmetic and bounds on anything derived from
  guest memory, file contents or IPC words. A malformed input is an `Err`, never a panic.
  Check arithmetic for the whole extent written or read (start, end, trailing data), not just
  the start, and bound loops over untrusted structures by the structure actually walked.
- Compatibility comes first. A design may defer a feature (server-side IPC, real system modules,
  homebrew services), but must not rule it out, and must say how it could be added later.
- An id or pointer read from guest-writable or possibly corrupt data is validated against what
  was actually allocated, not just against a range.
- When a layer relies on a permission or an error the reference documents, check that the
  production backend honours it end to end, or enforce it in the layer.
- When an operation can fail after it has begun changing shared state, its required side effect
  (flush, cleanup, rollback) happens on the error path too. A freed resource is not reusable
  until every pending follow-up on it (a TLB flush, a notification) has run.
- Decisions in command-line and gate code are pure functions, tested for every input combination.
- User-facing errors and command output are plain words, not `Debug` forms.
- Whatever saves state passes the same validity checks that restoring enforces, and restoring
  validates every relation between fields.
- Bound memory and time by what a hostile input can force, not only by loop counts. A failed
  streaming read is sticky: every later read returns the same error, never a clean end of file.
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
- When two crates share a unit or convention (ticks or nanoseconds, who converts), give the value
  a distinct newtype or name the conversion's owner in one place.
- Never derive deserialisation on a validated newtype; deserialise through the validating
  constructor and test that an out-of-range value is an error.
- Name a decoded bit position, shift or width once, as a constant shared by the decoder and the
  encoder.
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
- Pin every serialised or on-disk layout with a golden-bytes test that covers every variant with
  distinct non-zero values; changing the layout means bumping its version. Decode and check a
  header's magic and version before the body.
- A test invariant mirrors the reference's exact condition; no escape clause that switches the
  assertion off in common generated cases.
- Test every range operation with a neighbour that starts exactly at the end bound, and every
  state-transition branch for both the changed and the unchanged outcome.
- Every boundary check gets a test at the exact edge on both sides. A claim that something is
  "pinned by a test" names the test.
- Tests that touch the file system use a per-test unique folder and clean up; run a new test many
  times under both plain `cargo test` and nextest before trusting it.
- Negative tests assert the exact error (variant, exit code, message), so a missing or broken
  implementation can't pass them.
- When docs promise an order of checks, a test has both faults present at once. Decode tests
  give each field a distinct value.
- A parser gets one hand-built accepting case per format feature, not only inputs made by the
  project's own builder or encoder.
- Mutation-check new tests: for each new branch, break it in a scratch copy and confirm a test
  fails. Never claim a test pins a property it cannot observe.
- Never hide a flaky test behind retries: reproduce it under plain `cargo test` as well as
  nextest, find the root cause, and document a platform quirk where it is one.
- Unit tests in `#[cfg(test)] mod tests`; cross-module tests in `tests/`. `proptest` for parsers
  and encoders, `insta` for snapshots.
- Hypervisor tests need a real Mac. They skip themselves only when the `CI` environment variable
  is set (hosted runners can't create VMs); anywhere else, failing to create a VM is a failure.
- GPU tests (the wgpu render backend's) are the reverse, once an experiment shows wgpu runs
  cleanly and deterministically on Mesa's software Vulkan driver (lavapipe) on the hosted Linux
  runners: required in Linux CI on lavapipe, and skipped locally, with a notice, where no GPU
  adapter exists. Until that experiment passes they are local-only, like the hypervisor tests
  (`docs/specs/render-backend.md`, decision 9). Golden images are compared exactly only for
  scenes whose pixels can be computed; others use a stated tolerance.
- Never commit keys, firmware, game files or dumps derived from them. Fixtures are synthetic or
  built from homebrew.

## Git and GitHub

- Conventional Commits (`feat(ipc): ...`, `fix(loader): ...`, `docs: ...`, `chore: ...`).
  Commit at every green checkpoint; one logical change per commit. Run `just setup` once per clone:
  it points `core.hooksPath` at `.githooks`, whose `commit-msg` hook rejects non-conventional
  subjects. CI applies the same check to PR titles (git's own subjects are fine as commit
  messages; as a PR title only GitHub's generated `Revert "<conventional title>"` is accepted
  besides a conventional one).
- One issue → one branch → one PR, with an honest `Closes #N`.
- Issues labelled `help wanted` are reserved for human contributors (`CONTRIBUTING.md`): don't
  put them in a wave or build them, and leave pull requests from human contributors for the
  owner to merge.
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

- **libnx is ISC** (permissive): code may be ported, and the notice in the reference folder's
  `libnx/LICENSE.md` (its copyright line and permission text) goes in the header of the file that
  holds the port. The source files themselves carry only `@copyright libnx Authors`.
- **Atmosphère is GPL-2.0-only, so it is a reference for facts only.** Its licence can't be
  combined with this project's GPL-3.0-or-later: never port, translate or closely paraphrase its
  code. Read it to settle a fact (a result code, a command id, a layout), write the code from the
  fact, and cite the file in a comment marked "fact only" (`docs/references.md`).
