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

## Services: one request at a time per server; handler style decided by a trial
- **WHEN:** 2026-11-11
- **PROJECT:** starquake
- **SYSTEM:** services (and kernel IPC)
- **SCOPE:** architectural
- **WHAT:** (1) A service handles one request at a time per server by default, across all its sessions, as most real system modules do. A server that is multi-threaded on a console can opt in to several workers, with a stated reason. A deferred request doesn't hold the server. (2) The way service methods are written (a `commands!` table with hand-written unpacking) is a trial: `sm:`, `set:sys` and an `fsp-srv` subset are written that way, then judged before a fourth service is ported. If the methods are long or error-prone, the framework moves to a typed style where the method signature declares its inputs and outputs.
- **WHY:** Per-server queueing keeps thread-safety out of ordinary service code, which is where agent-written code is most likely to have subtle bugs, and it matches the console. The handler style will be repeated thousands of times and is expensive to change once many services exist, so it should be chosen from evidence, not up front.

## Test support is split by dependency; fixtures and snapshots are guarded
- **WHEN:** 2026-11-11
- **PROJECT:** starquake
- **SYSTEM:** testing
- **SCOPE:** moderate
- **WHAT:** (1) `starquake-testkit` holds only test helpers that depend on none of our crates (instruction encoders, a synthetic program builder). Each crate exports its own fakes behind a `test-support` feature. (2) Prebuilt homebrew test binaries are committed next to their source, with a source-hash manifest that the gate checks. (3) A PR that changes a trace snapshot must explain each change in its description, and review checks the explanation against the diff.
- **WHY:** Fakes in a shared crate break the defining crate's own unit tests (Rust builds two incompatible copies), and a large shared testkit is a merge hot spot and a shared blind spot. The hash check catches a binary drifting from its source without needing the toolchain in CI. Snapshots only protect anything if their changes are read, and agents can otherwise pass a test by accepting whatever the new output is.

## Five foundation specs approved
- **WHEN:** 2026-11-11
- **PROJECT:** starquake
- **SYSTEM:** cross-system
- **SCOPE:** architectural
- **WHAT:** The owner approved the execution model, observability and replay, kernel, service framework and test strategy specs (`docs/specs/`) as written on 2026-11-11, with each spec's recommended option standing where the owner didn't revise it. Notably: deferred replies instead of service threads; stage-1 page tables carry all fine-grained mapping; one event stream with an always-on flight recorder; all nondeterminism through an injectable environment; snapshots as a standing rule for kernel and service state; a deterministic interpreter backend for tests; a fixed memory layout by default; permission violations logged before they are enforced; auto-stub on for bring-up and strict in tests. Changing any of these means amending the spec with the owner's approval.
- **WHY:** The owner walked through each spec's decisions and tradeoffs, revised the ones they were concerned about (the kernel's server side and state model, service queueing and handler style, test support), and accepted the recommendations for the rest so that kernel work could start.

## First proposals vetted: six promoted, eight house rules adopted
- **WHEN:** 2026-11-11
- **PROJECT:** starquake
- **SYSTEM:** process
- **SCOPE:** moderate
- **WHAT:** The owner vetted the first 26 entries in `docs/proposals.md`. Six work items became issues (#63-#68), two were folded into existing work, one was dropped and one stays proposed (ExeFS/RomFS/NCA loading, until milestone 2 planning). Eight rules from the wave 1 and 2 reviews were adopted into `CLAUDE.md`: fixed proptest seeds; expectations from shipping constants at the exact limit; swap-detecting round-trip tests for parsers; guesses marked and divergences from the reference pinned by tests; whole-extent arithmetic checks on untrusted input; no retries for flaky tests and exact negative assertions; no duplicated logic across sibling modules; and the merge queue waits for a PR's own CI checks. Two candidates were dropped as house rules.
- **WHY:** The rules each came from a real review finding in agent-written code, and the owner wants recurring mistakes turned into standing rules. Hygiene items were promoted where they are small and prevent recurrence; larger or premature items wait.

## Autonomous mode until something runs; design delegated; stronger reviews
- **WHEN:** 2026-11-11
- **PROJECT:** starquake
- **SYSTEM:** process
- **SCOPE:** architectural
- **WHAT:** Until the first program runs end to end (milestone 1), the lead session works autonomously: it makes design decisions itself, amends specs, and records what it decided for the owner to review afterwards. It stops only for things that need the owner (hardware, passwords, purchases, anything irreversible outside the repo) or when it is genuinely blocked. Reviews get stronger than one adversarial reviewer per PR. Work that is single-threaded or needs taste and judgement (integration, foundations, bring-up) goes to a single agent on the strongest model instead of a parallel wave; waves stay for well-specified ports. Token cost of waves is accepted.
- **WHY:** Nothing runs end to end yet, and the owner's review of every decision was the bottleneck. The owner judges that code quality is better when judgement-heavy work is done by one strong agent with the whole picture, and wants reviews to catch more.

## A GPUI app frontend later; a trace is the acceptance evidence for now
- **WHEN:** 2026-11-11
- **PROJECT:** starquake
- **SYSTEM:** frontend
- **SCOPE:** moderate
- **WHAT:** The eventual user-facing frontend is a desktop app built with GPUI. Until then the command-line app is the only frontend, and "working" is shown by a trace and exit code from a headless run, provided the evidence is trustworthy (a known program with known expected output, checked independently). Frontends depend only on `starquake-core`, so the GPUI app is additive.
- **WHY:** The owner wants a real app eventually but doesn't want UI work to delay getting something running, and is satisfied by a trace as long as it really proves the program ran.

## In autonomous mode the lead logs decisions and vets proposals directly
- **WHEN:** 2026-11-11
- **PROJECT:** starquake
- **SYSTEM:** process
- **SCOPE:** moderate
- **WHAT:** While autonomous mode lasts, the lead writes decision-log entries without asking for approval of each one: entries for decisions the owner states, and entries for decisions the lead makes under delegation, marked "Decided by the lead under delegation". The lead also vets `docs/proposals.md` itself: it promotes what the milestone 1 path needs, rules on spec deviations, and records each choice in the file. Anything off that path stays proposed. The owner reviews both afterwards and can overturn any of them.
- **WHY:** Stopping for approval of every entry and every proposal was most of the ceremony in the process, and it blocked all remaining kernel and service work. The owner chose both options when asked.

## Design rulings made by the lead under delegation, 2026-11-12
- **WHEN:** 2026-11-12
- **PROJECT:** starquake
- **SYSTEM:** cross-system
- **SCOPE:** moderate
- **WHAT:** Decided by the lead under delegation, after the walking skeleton and wave 5: (1) the EL1 shim is mapped in the upper address half (TTBR1) with its own static tables, separate from guest tables; (2) the kernel handles the HIPC layer of a message (handles, process id, descriptors) and the service framework parses the CMIF or TIPC payload; (3) per-crate fakes live behind a feature named `test-support`, and the workspace allows the clippy lint that rejects that name; (4) Atmosphère is a pinned reference for facts only, because its GPL-2.0-only licence can't be combined with this project's; (5) a run's text trace and guest log are capped (100,000 events, 4 MiB), provisionally; (6) the kernel owns guest time, with a per-process counter rate and one zero point shared with patched counter reads.
- **WHY:** Each was a gap or conflict that building exposed: the stage-1 builder can't map privileged code; only a server knows whether its session is a domain; two tasks chose different feature names; six service result codes were guesses a facts-only reference settles; an uncapped run could exhaust host memory; and three crates each assumed another converted the clock. The owner delegated design decisions until milestone 1 and reviews them afterwards.

## The first real title is one the owner dumped; its files stay local and read-only
- **WHEN:** 2026-11-12
- **PROJECT:** starquake
- **SYSTEM:** titles
- **SCOPE:** moderate
- **WHAT:** The target title for milestone 2 is a Switch 2 game the owner dumped themselves, in the NUP container format (an open format for decrypted dumps, so no keys are involved). The dump lives outside the repository and is locked read-only. It is never copied, moved or modified, and never loaded whole into memory. Its path, its name, its title id and anything read from it stay out of code, tests, snapshots, commits, PRs and CI logs; reports about it say only what worked and what failed. Starquake reads it with `starquake inspect` and `starquake verify`, and later runs it.
- **WHY:** The owner would rather work towards a game they want to play than a smaller test title. A dump they made themselves is the only kind of game file this project works with, and keeping every trace of it local is what makes the repository safe to share later.

## The reference emulator is the trace oracle for homebrew
- **WHEN:** 2026-11-12
- **PROJECT:** starquake
- **SYSTEM:** testing
- **SCOPE:** moderate
- **WHAT:** Decided by the lead under delegation, after the owner asked whether another emulator could run the same fixture. A locally built Eden-NX2 runs the homebrew fixture headlessly with system-call and service-request logging, and its trace is the target for milestone 1: the address-space task (#97) and the sessions task (#98) are done when starquake's trace of the same file matches it call for call. The reference is trusted for the order of calls and their arguments; a result code it returns is checked against the other sources before it is copied, and known reference bugs are listed and not copied (`docs/reference-emulator.md`). Only normalised traces of homebrew programs are committed.
- **WHY:** "The program ran" was not checkable without something to compare against. An independent emulator running the identical file turns the distance to milestone 1 into an ordered list of calls, and it found starquake's first departure at the first system call.

## Each agent builds in its own checkout's build folder
- **WHEN:** 2026-11-12
- **PROJECT:** starquake
- **SYSTEM:** process
- **SCOPE:** minor/local
- **WHAT:** Decided by the lead under delegation. Agents in a wave no longer share one `CARGO_TARGET_DIR`. Each builds in the default `target/` of its own worktree, with incremental compilation off, and the worktree is removed when its PR lands.
- **WHY:** The shared folder served a sibling checkout's stale build several times: a stale `xtask` in wave 2, and in wave 8 a review that saw every test fail, a compile error from another checkout's sources, and a red gate on `main` that was only the cache. A wrong test result costs more than the disk: each worktree's folder is one to two gigabytes, and the machine now has about 98 GiB free.

## House rules adopted from waves 8 and 9
- **WHEN:** 2026-11-12
- **PROJECT:** starquake
- **SYSTEM:** process
- **SCOPE:** minor/local
- **WHAT:** Decided by the lead under delegation. Eight rules from review findings moved into `CLAUDE.md`: saving passes the checks restoring enforces; one named constant per decoded bit field; mutation-check new tests; a test with both faults present when an order of checks is promised; sticky streaming-read failures; bounds by what a hostile input can force; a hand-built accepting case per format feature; plain-word errors asserted exactly. The candidates that existing rules already cover were dropped (`docs/proposals.md`). Also ruled: adding a crash-report kind bumps the report's version.
- **WHY:** Each came from a defect a reviewer found in agent-written code in these waves, and the owner wants recurring mistakes turned into standing rules.

