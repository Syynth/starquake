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
- **WHAT:** The emulator is named "starquake". The repo folder is `~/code/rs/starquake`, and crates use a `starquake-` prefix or keep their role names in the workspace (e.g. `hos-cpu`). The name avoids Nintendo trademarks (switch, nx, nintendo).
- **WHY:** It continues the magnet theme the user liked in "magnetar" (a nod to the Switch 2's magnetic controllers): magnetars have starquakes. It's free on crates.io, no existing emulator uses it, and steering clear of trademarked terms lowers takedown risk.
