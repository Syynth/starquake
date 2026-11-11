# Decision log

User decisions captured during development. Append new entries at the bottom.

Entry format:

```
## <short description>
- **WHEN:** <date, YYYY-MM-DD>
- **PROJECT:** <project/repo name>
- **SYSTEM:** <short tag>
- **SCOPE:** <minor/local | moderate | architectural>
- **STATUS:** tentative   (only if tentative)
- **WHAT:** <what was decided>
- **WHY:** <the rationale>
```

## Linux is a first-class target alongside macOS
- **WHEN:** 2026-11-11
- **PROJECT:** switch2-emu (Rust Switch 2 emulator)
- **SYSTEM:** cross-system
- **SCOPE:** architectural
- **WHAT:** Support Linux (including x86-64 handhelds like Steam Deck, best-effort) as well as macOS; the design must not be Mac-only. Consequence: the CPU layer needs pluggable backends (Hypervisor.framework on Apple Silicon, KVM on ARM64 Linux, a JIT for x86-64), and rendering needs Vulkan, not only Metal.
- **WHY:** The emulator should be usable on Linux devices, not locked to the Mac it is developed on. The Steam Deck may be too weak to run games at speed, but the architecture shouldn't rule it out.

## Build on a fork we own, not upstream contributions
- **WHEN:** 2026-11-11
- **PROJECT:** switch2-emu
- **SYSTEM:** cross-system
- **SCOPE:** architectural
- **WHAT:** Work in our own fork (most likely of oboromi, pending the investigation) so we control direction, rather than contributing to an upstream project or starting from zero. A fork of oboromi or Eden inherits GPLv3.
- **WHY:** Ownership: freedom to experiment and steer without upstream review or roadmap constraints.

## Development is primarily agent-driven
- **WHEN:** 2026-11-11
- **PROJECT:** switch2-emu
- **SYSTEM:** process
- **SCOPE:** architectural
- **WHAT:** Most code will be written by agents rather than by hand. The project should invest early in what makes that work: specs and protocol notes, trace/auto-stub tooling, deterministic offline test harnesses (recorded macro calls, shader dumps checked against nvdisasm), and a fast build/test gate.
- **WHY:** The work is huge and mostly mechanical against references (IPC protocols, shader ISA tables), so the user's time goes to direction and review rather than typing.

## Start a fresh repo instead of forking oboromi
- **WHEN:** 2026-11-11
- **PROJECT:** switch2-emu
- **SYSTEM:** cross-system
- **SCOPE:** architectural
- **WHAT:** Don't fork oboromi. Start a fresh GPLv3 Rust workspace that we own, using the crate layout from the scoping doc. We may vendor GPLv3 code with attribution, starting with rapid-arm-emu's `arm-instruction-decoder` (for the load-time patcher and a later x86-64 JIT) and later its `io-mmu`. This supersedes the "most likely of oboromi" part of "Build on a fork we own"; the ownership goal stands.
- **WHY:** The evaluation (`docs/oboromi-evaluation.md`) found oboromi is a scaffold. The CPU is Unicorn with no exits or page tables, the services are empty, 5 of 201 SM86 handlers work, and Vulkan only creates a device. A fork would mean deleting nearly all of it. Its C/C++ build (Unicorn, Qt, CMake) works against the fast build/test gate and doesn't build on this Mac. Its SM86 tables look like NVIDIA's internal machine description, which is a provenance risk. Staying GPLv3 keeps us free to copy from oboromi or Eden later.

## Project name: starquake
- **WHEN:** 2026-11-11
- **PROJECT:** starquake (previously referred to as switch2-emu)
- **SYSTEM:** cross-system
- **SCOPE:** moderate
- **WHAT:** The emulator is named "starquake". The repo folder is named `starquake`, and crates use a `starquake-` prefix or keep their role names in the workspace (e.g. `hos-cpu`). The name avoids Nintendo trademarks (switch, nx, nintendo).
- **WHY:** It continues the magnet theme the user liked in "magnetar" (a nod to the Switch 2's magnetic controllers): magnetars have starquakes. It's free on crates.io, no existing emulator uses it, and steering clear of trademarked terms lowers takedown risk.

## Backlog and coordination live in a private GitHub repo
- **WHEN:** 2026-11-11
- **PROJECT:** starquake
- **SYSTEM:** process
- **SCOPE:** moderate
- **STATUS:** tentative
- **WHAT:** The repo is hosted privately on GitHub. The backlog is GitHub issues (one issue → one branch → one PR), and waves of port work run through the autonomous-pump process. Logged without the usual approval prompt because the owner asked for it and then went offline; confirm or edit.
- **WHY:** A remote makes CI actually run on Linux, gives the pump its issue board, and allows cloud sessions that use no local disk. Private, because public console-emulator repositories have a history of takedowns.

## Adopt the owner's standing Rust conventions
- **WHEN:** 2026-11-11
- **PROJECT:** starquake
- **SYSTEM:** process
- **SCOPE:** moderate
- **STATUS:** tentative
- **WHAT:** Starquake uses the same lint profile and habits as the owner's other Rust workspaces: clippy `pedantic` and `cargo` groups, the restriction denies (`unwrap_used`, `expect_used`, `panic`, `todo`, `dbg_macro`, printing, `undocumented_unsafe_blocks`, ...), `#[expect]` with a reason instead of `#[allow]`, `thiserror`, workspace-inherited dependencies, tests first, Conventional Commits, and no debug info in dev builds. Three deliberate differences, each open to a ruling: `unsafe_code` stays denied but the CPU crate opts in (an emulator must run guest code); `missing_docs` is on; and the gate is `cargo xtask ci` (with a `justfile` forwarding to it) because it also checks free disk space. Logged without the usual approval prompt because the owner asked for their preferences to be found and applied, then went offline; confirm or edit.
- **WHY:** Consistency with the owner's other projects, and lint rules catch what review misses when most code is written by agents.

## Proposed work goes in a file first; GitHub issues are for vetted work
- **WHEN:** 2026-11-11
- **PROJECT:** starquake
- **SYSTEM:** process
- **SCOPE:** moderate
- **WHAT:** Work that agents propose (wave scope-review follow-ups, review findings, ideas) is recorded in a proposals file in the repo, `docs/proposals.md`, not filed as GitHub issues. An entry is promoted to a GitHub issue only after the owner has vetted it; dropped entries stay in the file with the reason. This replaces the earlier rule that findings become issues. Issues already filed stay as they are.
- **WHY:** Agents generate follow-ups faster than they can be judged, and most are hygiene. Keeping them in a file keeps the issue board a list of work the owner has actually chosen to do, and makes the proposals reviewable in one place and in PRs.

## Compatibility first: defer features, don't design them out
- **WHEN:** 2026-11-11
- **PROJECT:** starquake
- **SYSTEM:** kernel (cross-system)
- **SCOPE:** architectural
- **WHAT:** Very good compatibility is a project goal. The kernel doesn't have to implement server-side IPC, guest-hosted services or Nintendo's real system modules now, but its design must let them be added later without a redesign: sessions have a client end, a server end and a request queue; multiple guest processes are supported. More generally, a spec may defer a compatibility feature but must not rule it out, and must say how it could be added.
- **WHY:** The owner wants very good compatibility, including the option to fall back on real system modules and to run homebrew that provides its own services. Ruling these out to save work now would trade compatibility away permanently.

## Kernel state is tables behind a dispatcher; one lock to start
- **WHEN:** 2026-11-11
- **PROJECT:** starquake
- **SYSTEM:** kernel
- **SCOPE:** architectural
- **WHAT:** Kernel state lives in a few tables (scheduler, per-process address space, per-process handle table), with objects referenced by id. Each system call declares the tables it touches in its function signature, and only the dispatcher takes locks, in one fixed order. One lock guards all tables at first; it is split table by table later, when measurements justify it, without changing the system calls. The first three port slices (dispatcher, handle table, scheduler) are a trial of the design before the rest depend on it.
- **WHY:** The owner wanted the single-arena design to leave room for parallelism, the way Bevy runs systems in parallel when they don't need exclusive world access. Declaring table access per call gives that option cheaply, matches Horizon's own lock structure, and keeps the benefits of the arena: no unsafe, a deterministic kernel, replay and snapshots.
