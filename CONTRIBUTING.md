# Contributing to starquake

Most of starquake is written by AI agents that a lead session coordinates, and the repository is
built so that every change can be checked: one gate, two reviews per pull request, and approved
specs that code follows. Human contributors are welcome and work the same way.

## Setting up

1. Clone the repository and run `just setup` once. It points git at the repository's hooks, which
   reject commit messages that aren't [Conventional Commits](https://www.conventionalcommits.org/).
2. The Rust toolchain is pinned in `rust-toolchain.toml`; `rustup` installs it on first use.
3. Install the gate's cargo plugins at the pinned versions. The command is in `CLAUDE.md`
   ("Before you say a change is done"), and `xtask/tool-versions.toml` holds the versions.
4. Run the gate: `cargo xtask ci` (or `just ci`). CI runs exactly the same thing. Also check that
   your change builds for Linux:
   `cargo clippy --workspace --all-targets --target x86_64-unknown-linux-gnu -- -D warnings`.
5. Optional: `just refs` fetches the pinned reference sources (Eden, its Switch 2 forks, Mesa,
   libnx and others) into a sibling folder. Most porting work needs them; `docs/references.md`
   says what each one is for.

Linux and Intel Macs build and test everything except the hypervisor backend. Running guest code
natively, and the tests that do, needs an Apple Silicon Mac.

## Finding something to work on

- Issues labelled **`help wanted`** are reserved for human contributors: the agents don't take
  them. Start there. **`good first issue`** marks the smaller ones.
- Comment on the issue before you start, so nobody else picks it up.
- Read the spec the issue names before writing code. The specs in `docs/specs/` are approved
  designs; if one seems wrong, say so on the issue rather than working around it.
- `docs/architecture.md` explains how the pieces fit, and `docs/port-map.md` says what to port
  from Eden and what to redesign.

## Making a change

- **Rules:** `CLAUDE.md` is the project's rulebook, for people as much as agents: lint settings,
  error handling, testing, licensing and attribution, and what may never be committed.
- **Branches and pull requests:** one issue, one branch, one pull request, whose description
  says `Closes #N` only if it closes the issue completely. Never push to `main` directly: the
  repository can't enforce branch protection on its current plan, so this is on trust.
- **Tests first:** write the test, watch it fail, then make it pass. A bug fix comes with a test
  that failed before the fix.
- **Merging:** your pull request merges once its CI has passed and it has been reviewed. The
  owner reviews and merges human contributions; the lead session may add its own review.

## What never goes in the repository

Keys, firmware, game files, or anything derived from them, including traces, logs or screenshots
from a commercial game. Fixtures are synthetic or built from homebrew. Atmosphère may be read to
settle a fact but never ported or paraphrased (its licence is incompatible); `CLAUDE.md` explains
the licensing rules for each reference.
