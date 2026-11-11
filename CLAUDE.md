# starquake

A Switch 2 emulator in Rust, for macOS (Apple Silicon) and Linux. GPLv3. Most code is written by
agents, so the repo is built to make changes checkable.

## Before you say a change is done

Run `cargo xtask ci`: fmt check, clippy with `-D warnings` and the workspace lint set, nextest,
doctests. CI runs exactly this on macOS arm64 and Linux x86-64. Platform-specific code must still
compile (or be `cfg`-gated) on both; check with
`cargo clippy --workspace --all-targets --target x86_64-unknown-linux-gnu -- -D warnings`.

## Where things are

- `docs/architecture.md`: how the pieces fit, threads, and dependency direction. Read it first.
- `docs/decision-log.md`: the user's decisions and why. Follow them; propose changes, don't drift.
- `docs/switch2-eden-diff-scoping.md`: what Switch 2 support actually requires, from Eden-NX2's
  149-row blocker log and PR #4588. Cite its rows when implementing protocol behaviour.
- `docs/oboromi-evaluation.md`: why this is a fresh repo, and what is worth borrowing.
- `tools/hvprobe`: measured Hypervisor.framework behaviour (`tools/hvprobe/README.md`).

## Rules

- Dependencies point one way: `memory` → `cpu`, `ipc` → `kernel` → `services` → `gsv` → `gpu` →
  `shader`, `render`. `core` wires everything; frontends depend only on `core`. No cycles, and
  no reaching around a layer.
- Every `unsafe` block gets a `// SAFETY:` comment; every public item gets a doc comment.
- Guest-controlled values are untrusted: checked arithmetic and bounds on anything derived from
  guest memory or IPC words.
- Protocol facts (command ids, struct layouts) need a source in a comment: a scoping-doc row, a
  reference implementation, or a trace. Mark guesses as guesses.
- Never commit keys, firmware, game files or dumps derived from them. Tests use homebrew or
  synthetic data only.
- Licensing: code may be ported from GPL projects (Eden, oboromi, rapid-arm-emu) with attribution
  in the file header; MIT sources (Mesa NAK) likewise. Don't use leaked or NDA material.
