# Roadmap

How the work is planned and scheduled. The backlog itself lives in GitHub issues; this file
explains the shape. Work that agents propose waits in `docs/proposals.md` until the owner promotes
it to an issue. Detailed planning only goes as far as the next milestone.

## Three kinds of work

| Kind | Who | Parallel? | Output | Merge rule |
|---|---|---|---|---|
| **Design** | The lead session, with the owner's approval | No | A spec under `docs/specs/`, trait definitions, decision-log entries | Waits for owner review |
| **Port** | Build agents, one worktree each | Yes, in waves | One reference file group translated into one Rust module, with tests | Merges on a green gate plus adversarial review |
| **Bring-up** | One session on real hardware | No | A milestone actually running | Owner drives it |

A port task is ready for an agent only when it has a fixed interface to code against and a test
that proves it right without the rest of the emulator. Design work exists to produce those two
things. Issues without them carry the `needs-design` label and stay out of waves.

## Milestones

| | Milestone | Done when |
|---|---|---|
| M0 | Foundation | Workspace, gate, CI, references, backlog. **Done.** |
| M1 | Homebrew hello-world | A small homebrew program loads, runs natively under the hypervisor backend, makes system calls and IPC requests, and exits. No keys or firmware involved. |
| M2 | A Switch 2 title boots to graphics init | Loader, kernel and services get a real title as far as its first GPU submission (Eden-NX2 notes rows 4–29). Needs titles dumped from the owner's own console. |
| M3 | First present | The GPU memory model, gated command buffers, macros and a minimal shader path put a frame on screen (rows 30–57). |
| M4 | Correct frames in one title | Shader coverage, textures, queue synchronisation. |
| M5 | Performance | Front/back GPU split tuned, macro memo, JITs. |

## Route to milestone 1: skeleton first

1. **Walking skeleton.** A synthetic guest program, built with the testkit, is loaded, run under
   the hypervisor, makes system calls through the kernel dispatcher, and exits; the command-line
   app writes a trace and an exit code. Provisional parts (bring-up memory, one thread, no
   scheduler) are marked and get replaced below.
2. **Widen behind it,** in waves and single-agent tasks: finish the dispatcher's table
   borrows; stage-1 page tables; kernel slices 4–10; the service framework and `sm:`; loader
   wiring; recorder wiring.
3. **Milestone 1:** a real libnx homebrew program sends a request to `sm:` and exits, with a
   golden trace. Needs devkitPro installed by the owner.

Later: a desktop app frontend built with GPUI on top of `starquake-core`.

## Waves

- **Wave 0, design:** execution model (CPU, memory, kernel threading), test strategy. Specs are
  proposals until the owner approves them.
- **Wave 1, ports that depend only on external facts** (file formats, wire formats, instruction
  encodings), one crate or module each so they don't collide:
  hypervisor backend, IPC framing, loader formats, code patcher, Turing macro interpreter.
- **Wave 2:** kernel core and the first services, once the execution model is approved.
- **Side tracks** with offline tests, picked up whenever the main path is waiting: shader
  translator, macro memo, audio renderer, file system formats.

Waves run through the owner's `autonomous-pump` process: each issue is built in its own worktree,
reviewed adversarially, and landed one at a time through a merge queue behind `cargo xtask ci`.

## Disk and machine limits

- At most 3–4 build agents at once on an 8-core machine.
- One shared build folder for a wave (`CARGO_TARGET_DIR`), warmed before the wave starts, with
  incremental compilation off. Dev builds carry no debug info.
- `cargo xtask ci` refuses to build with less than 15 GiB free.
- The `xtask` binary is per checkout, not shared. The `cargo xtask` alias (`.cargo/config.toml`)
  passes `--target-dir target/xtask`, so the binary is built and run from a folder inside the
  checkout, while the cargo commands it spawns inherit `CARGO_TARGET_DIR` and keep building in the
  shared folder. Without this, worktrees sharing a build folder overwrite each other's
  `debug/xtask` and one runs another's gate. Run `cargo xtask` (or `just ci`) from the repository
  root: the path is relative to the directory cargo is started in.
  `xtask/tests/per_checkout_binary.rs` runs two checkouts with different `xtask` sources against
  one shared folder and checks that each runs its own.
- Worktrees are removed when their PR lands; reference sources are one shared checkout.
- CI: Linux on every PR; macOS on `main`, and on PRs only for a narrow fmt/clippy/build job when
  `crates/cpu`, `crates/memory` (a dependency of the hvf backend), `tools/hvprobe`,
  `tools/macos-runner.sh`, `.cargo`, the root `Cargo.toml` (workspace lints and dependencies),
  `Cargo.lock`, `rust-toolchain.toml` or `clippy.toml` change, because hosted macOS minutes cost
  ten times as much. The list lives in `.github/workflows/macos-pr.yml`; keep the two in step.
