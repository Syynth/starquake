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

## Fable is used sparingly; builds and fixes run on Opus
- **WHEN:** 2026-11-12
- **PROJECT:** starquake
- **SYSTEM:** process
- **SCOPE:** moderate
- **WHAT:** The owner's direction: keep using Fable, but much more sparingly, and not for build agents by default. The allocation that follows is the lead's reading of that. Build and fix agents run on Opus, and the routine steps (merging, lessons, scope review) on Sonnet or Opus. Fable is kept for work where one bounded pass of judgement matters most: the fidelity review of a foundation PR (the kernel, the execution model, anything later work builds on) and the lead session's own design decisions. A build goes to Fable only when the lead can say why Opus would not do, and the report says so. This amends "Autonomous mode until something runs", which sent judgement-heavy work to the strongest model.
- **WHY:** One day of Fable builds and reviews used the owner's whole weekly allowance. Builds are where the tokens go: a build agent runs for hours of tool calls, while a review is one read of a finished change. Spending Fable on the review keeps the strongest judgement on what merges, at a fraction of the cost.

## The graphics-half designs start now, before milestone 1 lands
- **WHEN:** 2026-11-12
- **PROJECT:** starquake
- **SYSTEM:** gpu
- **SCOPE:** architectural
- **WHAT:** The owner asked to start the designs that block parallel work. Four specs are drafted at once, one agent each: the graphics server session model (#11), the GPU front/back protocol and engine state model (#12), the shader pipeline (#13) and the render backend (#14). Each draft lists its decisions for the owner, its interfaces to the other three, and its implementation cut into pull-request-sized slices marked as parallel or not and as testable offline or not. The lead reconciles the interfaces between the four. The drafts are design PRs: they wait for the owner's review and are not merged or built on under delegation.
- **WHY:** The graphics half is the largest part of the project and nothing in it is built. It is blocked on these designs, not on code, and much of it can be built and tested offline by many agents at once as soon as the designs are approved. The kernel work that leads to milestone 1 is narrow and serial, so starting the designs now costs it nothing.

## Milestone 1 reached; the lead's delegation ends with it
- **WHEN:** 2026-11-12
- **PROJECT:** starquake
- **SYSTEM:** process
- **SCOPE:** moderate
- **WHAT:** Recorded by the lead. A real libnx homebrew program now runs from file to exit on `main`: 30 system calls and 5 service requests in the same order as the reference emulator, its three lines printed, exit code 0, nothing stubbed, checked by a test and re-run by the lead (#98, PR #106; issue #20 closed). "Autonomous mode until something runs" set this as its end, so from here the lead no longer decides design questions, amends specs or promotes proposals on its own: design and spec PRs wait for the owner, and proposals wait to be vetted. Implementation PRs for filed issues still merge on a green gate and two reviews, as before. Three house rules adopted while the last wave was in flight are marked as such in `docs/proposals.md`.
- **WHY:** The owner granted the delegation to get past the point where nothing ran end to end, and tied it to this milestone. Whether to extend it toward milestone 2 is the owner's call.

## Delegation extended toward milestone 2, for implementation only
- **WHEN:** 2026-11-12
- **PROJECT:** starquake
- **SYSTEM:** process
- **SCOPE:** moderate
- **WHAT:** After milestone 1 the owner extended the lead's delegation toward milestone 2, for implementation work only. The lead files implementation issues that follow approved specs (kernel slices, CPU, loader, services, test fixtures), runs them in parallel lanes on Opus, and merges their PRs on a green gate and two reviews. It does not decide design questions, amend specs or merge design PRs: those wait for the owner, including the four graphics drafts.
- **WHY:** Progress can continue in parallel while the owner reviews designs at their own pace. Implementation against an approved spec is checked by the gate, two reviewers and the reference traces; designs set direction, so they stay with the owner.

## Two Opus reviews where a Fable review was planned
- **WHEN:** 2026-11-12
- **PROJECT:** starquake
- **SYSTEM:** process
- **SCOPE:** minor/local
- **WHAT:** The owner's direction after the Fable allowance ran out: where a PR was to get a Fable fidelity review, it gets two Opus reviews instead (the fidelity review and the test-breaking review, both on Opus). Builds and fixes stay on Opus. This amends "Fable is used sparingly", which kept Fable for the fidelity review of foundation PRs.
- **WHY:** The week's Fable allowance is spent. Two independent Opus reviews keep the second, differently-aimed read of every change, which is what the review process relies on.

## The render backend starts on wgpu, behind our own trait
- **WHEN:** 2026-11-12
- **PROJECT:** starquake
- **SYSTEM:** render
- **SCOPE:** architectural
- **WHAT:** The first host graphics backend is wgpu, behind starquake's own backend trait and plain-data command lists; no wgpu type crosses into the GPU crate. A direct Vulkan backend is added when Linux work or a title needs a stage wgpu lacks (geometry, tessellation, transform feedback). A native Metal backend is written only if measurements at the performance milestone call for it. The draft's recommendation (Vulkan through ash, with MoltenVK on the Mac) was not taken.
- **WHY:** The owner knows wgpu and will read and debug this code. It runs natively on Metal on the development Mac with nothing to install and no translation layer, it is safe pure Rust, and it takes the translated shaders directly, so first frames come sooner. Its gaps (checked against wgpu 30.0.1) bite after first frames, except texture channel remapping, which becomes shader variants. Keeping the trait ours means the choice can be revisited.

## Translated shaders reach the backend as a naga module; naga is the only shader backend for now
- **WHEN:** 2026-11-12
- **PROJECT:** starquake
- **SYSTEM:** shader, render
- **SCOPE:** architectural
- **WHAT:** The shader crate's output is a validated naga module plus tables describing its resources, inputs and outputs. The wgpu backend takes the module directly. A later direct backend asks the shader crate for final Vulkan or Metal code instead, which the shader crate emits; the backend says which form it wants. The shader and render crates do not depend on each other: the GPU crate carries the module across, and all three use the naga version wgpu uses. naga is the only shader backend at first; the project's own intermediate form stays independent of naga so a direct code writer can be added later. Game memory addresses stop at the GPU crate, and wgpu types stop at the wgpu backend crate.
- **WHY:** wgpu is built on naga and accepts a module on its safe, supported path, so emitting final code now would add plumbing for a route wgpu does not need and would bypass its checks. naga cannot express geometry or tessellation stages, but wgpu cannot run them either, so nothing is lost by starting naga-only. Keeping emission in the shader crate for direct backends keeps naga's breaking changes in one place.

## The game sees GPU work as done only when the host GPU has finished it
- **WHEN:** 2026-11-12
- **PROJECT:** starquake
- **SYSTEM:** gpu
- **SCOPE:** architectural
- **WHAT:** A semaphore release or syncpoint increment becomes visible to the game once the host GPU has actually finished the work before it, not when the work is recorded. Signals with no host work behind them are written at once, and several are batched into one completion check. Signalling at record time is kept only as a debugging switch. The graphics server draft's assumption that increments happen at record time is dropped.
- **WHY:** An early signal lets the game read back an image or a query result that does not exist yet, which shows up as occasional wrong pixels or stale data and is very hard to trace. A late one only costs latency, which is easy to measure. The project aims for correct frames before fast ones, and wgpu reports when submitted work has completed.

## Presents pace the game through a buffer queue
- **WHEN:** 2026-11-12
- **PROJECT:** starquake
- **SYSTEM:** gsv
- **SCOPE:** architectural
- **WHAT:** A present returns at once with the list of buffers still in use. The graphics server's emulated 60 Hz clock shows one ready frame per tick and frees the buffer it replaces; the game waits only when it has no free buffer. The fence wait the other reference used (hold the reply until the previous frame is done and the next tick has passed) is kept as a per-run fallback setting, implemented as a deferred reply so it never blocks a core. The GPU and render drafts' assumption that a present waits on the frame's fence is dropped.
- **WHY:** It never blocks the emulator and lets drawing overlap GPU work. The fence-wait approach lost up to half of each frame to waiting, needed a 4-second safety timeout and deadlocked when tuned. The risk, that a game does not wait for free buffers properly, is unconfirmed either way and is covered by the fallback.

## Syncpoints and the GPU address table live in the GPU crate; the graphics server decides, the GPU applies
- **WHEN:** 2026-11-12
- **PROJECT:** starquake
- **SYSTEM:** gsv, gpu
- **SCOPE:** architectural
- **WHAT:** The syncpoint counters and the GPU address table are plain data types in the GPU crate. The graphics server allocates counters, computes each submission's fence, and writes mappings; the GPU performs increments after host completion and reads the table on every access. Counter changes go into a queue that the graphics server's timer thread drains, so game threads are woken in a recorded order. Unmaps, kind changes and destroys reach the GPU as ordered updates with a barrier, and the game is told only when no earlier GPU work, including the thread that reads command buffers, can still use the old mapping. The GPU draft's memory-access interface is kept, with its production implementation in the GPU crate over the table. The GPU draft's interface for the graphics server to own syncpoints is dropped.
- **WHY:** The graphics server already depends on the GPU crate, so no new crate is needed, and the GPU reads the table on its hottest path, where a call through an interface would cost on every access. Waking game threads from one recorded place keeps replay deterministic, as the observability spec requires. Ordered unmaps stop the GPU reading memory the game has already been told is free.

## Draw commands cross to the backend as plain data; caches are split by what they are keyed on
- **WHEN:** 2026-11-12
- **PROJECT:** starquake
- **SYSTEM:** gpu, render
- **SCOPE:** architectural
- **WHAT:** The GPU crate's back thread writes draw and dispatch commands as plain data (handles and numbers); the render backend's own thread turns them into host API calls, submits and presents. Caches keyed by game memory or register state (textures, buffers, pipeline keys, write tracking, eviction, detiling) live in the GPU crate; host objects live in the backend and are referred to by handle. Each backend implements the same trait and never sees a game memory address.
- **WHY:** Eden-NX2 measured the thread that issues host calls as its bottleneck, so the back thread should not wait on the graphics driver. Plain-data lists give a recording backend for GPU-free tests and host-level captures without extra work, and keep wgpu types out of the GPU crate so a direct backend can be added. Writing cache policy once, in the GPU crate, means it is tested without a GPU and not repeated per backend.

## Twenty-one graphics decisions accepted as recommended
- **WHEN:** 2026-11-12
- **PROJECT:** starquake
- **SYSTEM:** gsv, gpu, shader, render
- **SCOPE:** moderate
- **WHAT:** The owner accepted, as the drafts recommend: (1) a draw whose shader or pipeline is not ready waits, one setting owned by the GPU crate, with skipping as an opt-in setting after the persistent caches exist (gpu 9, shader 6, render 5); (2) one house rule when references disagree: Nvidia's headers and module-cited readings for layouts and meanings; where one answer contains the other, the larger; otherwise the one a title ran on; each case marked in code and pinned by a test (gsv 8, gpu 10); (3) new crates: a leaf `starquake-layout` crate for tiling, no separate shader-decoder crate for now, and `starquake-render` plus `starquake-render-wgpu` (gpu 5, shader 14, render 10); and eighteen local items: gsv 5, 6, 7, 11; gpu 6, 7, 11, 12, 13; shader 4, 10, 12, 15, 16; render 6, 11, 12, 13, each as its draft recommends. Item numbers refer to each draft's "Decisions for the owner".
- **WHY:** Each is the conservative or conventional choice, follows from the six foundational rulings, or keeps an option open for later; none sets direction beyond its own area. Accepting them together lets the drafts settle and the review focus on the coupled groups and the four contested items.

## Ten coupled graphics decisions accepted as recommended
- **WHEN:** 2026-11-12
- **PROJECT:** starquake
- **SYSTEM:** gsv, gpu, shader, render, observability
- **SCOPE:** moderate
- **WHAT:** (1) GPU submission queues are bounded at 65,536 entries; a full queue makes the submit wait as a deferred reply (gsv 12). (2) GPU-only memory is backed lazily on the host and not charged to the title's pools, revisited if a trace shows the console charges it to remote memory (gsv 4). (3) GPU-written data is copied back only when something reads it as plain memory, and shader memory accesses through untraceable addresses are rejected at first; both are revisited when a direct backend can share memory (gpu 4, shader 9). (4) The GPU notices a CPU semaphore release by polling with back-off, measured, before any write-trapping (gpu 3). (5) The shared syncpoint page is refreshed on demand, at most 1 ms stale (gsv 9). (6) GPU captures record every byte each GPU thread read, in order; the three replay commands (`starquake gpu-replay`, `starquake render-replay`, `cargo xtask shader-replay`) are approved as an addition to the observability spec's headless contract (gpu 8). (7) The persistent shader cache: key and format defined now, built after milestone 3, storing inputs not translated modules (shader 7). (8) Translation sees GPU state only through the non-blocking question-and-answer lookup (shader 5). (9) Texture handles are traced first; true bindless only after a survey of titles shows it is needed (shader 8). (10) Mesh and task shaders after milestone 3, with the decoder reading their instructions from the start (shader 11).
- **WHY:** Each follows from the six foundational rulings. They save host memory on unified machines, keep replays exact, avoid kernel work nobody owns yet, and defer what wgpu cannot do until a survey or a direct backend shows it is needed, without designing it out.

## The GPU front thread owns the register file
- **WHEN:** 2026-11-12
- **PROJECT:** starquake
- **SYSTEM:** gpu
- **SCOPE:** architectural
- **WHAT:** The GPU front thread keeps the only copy of every engine's register values, runs macros, and sends the back thread only the changes. Register writes with side effects (one register changing another, or triggering an action) are computed on the front thread, each rule covered by a per-engine fixture, and captures replayed against recorded frames catch rules that were missed. The back thread keeps a mirror with dirty flags and never changes a register itself (gpu 1).
- **WHY:** Macros are a large share of the GPU's work, so running them on the front thread keeps most of the gain from splitting work across two threads; Eden-NX2 found this arrangement fastest. Keeping one copy avoids two copies drifting apart, which forced Eden-NX2 to build and maintain a checker. The risk, a missed side-effect rule, is a correctness bug that offline fixtures and replay can find.

## The shader translator uses a new arena-based Rust IR
- **WHEN:** 2026-11-12
- **PROJECT:** starquake
- **SYSTEM:** shader
- **SCOPE:** architectural
- **WHAT:** The shader crate's intermediate representation is a new Rust design that keeps Eden's operation set and pass behaviour but stores instructions in arenas referred to by typed ids, not pointer-linked objects. It has its own register class for Switch 2's uniform registers, a text form that can be printed and parsed for tests, and a verifier run after every pass in tests and debug builds. Eden's passes are ported by behaviour, not line by line (shader 1).
- **WHY:** Eden's operations and passes are the proven part; its pointer-linked data layout fights Rust's ownership rules and tends toward `unsafe`, which the project bans outside host-facing crates. The kernel's move to tables and ids ported Eden's logic cleanly, and the same approach gives deterministic, testable IR. A native design also avoids Eden's Switch 1-era trick of packing uniform registers into the top of the regular register file.

## The wgpu backend's GPU tests run in Linux CI on lavapipe, if it runs cleanly
- **WHEN:** 2026-11-12
- **PROJECT:** starquake
- **SYSTEM:** render, testing
- **SCOPE:** moderate
- **WHAT:** If an experiment shows wgpu runs cleanly and deterministically on Mesa's software Vulkan driver (lavapipe) on GitHub's hosted Linux runners, the render backend's GPU tests are required in Linux CI and skipped locally, with a notice, where no GPU adapter exists. Golden images are compared exactly only for scenes whose pixels can be computed; others use a stated tolerance. If the experiment fails, the GPU tests are local-only, like the hypervisor tests (render 9).
- **WHY:** Most changes are made by agents, and an agent that breaks the render backend should learn it from the PR's checks, not from the owner noticing a broken frame later. The recording backend already tests most logic without a GPU, so these tests only need to cover what is specific to wgpu, and the cost is a package install and somewhat slower CI.

## No video decoding at first, then FFmpeg behind an off-by-default feature
- **WHEN:** 2026-11-12
- **PROJECT:** starquake
- **SYSTEM:** gsv
- **SCOPE:** moderate
- **WHAT:** The first graphics-server slices answer video codec sessions without decoding, so movies show a frozen picture and the game carries on. Later, FFmpeg is added behind a Cargo feature that is off by default, covering VP9, AV1 and H.264; each format moves to a pure-Rust decoder when a good enough one exists. The decoder sits behind the video session protocol, so it can be swapped without changing the sessions (gsv 10).
- **WHY:** Movies matter for the experience but not for getting frames on screen. FFmpeg covers every relevant format but is a C library, which the project keeps out of the default build, so it stays opt-in. A game that waits for a movie to finish may sit on a frozen frame; that is found when a real title hits it.

## Approved specs amended for the graphics designs
- **WHEN:** 2026-11-12
- **PROJECT:** starquake
- **SYSTEM:** cross-system
- **SCOPE:** architectural
- **WHAT:** The owner approved the changes the graphics drafts imply to already-approved specs: (1) execution model: write tracking for the GPU caches goes through a `WriteWatch` in the memory crate, requested by the GPU back thread, applied by the kernel at its next entry on any core, and drained as a list of dirtied pages, so the kernel never calls into the GPU; (2) kernel and services: a device-share lock over a memory range, a range query of the GPU-accessible attribute set by `svc 0x80`, a hook run after the kernel lock is released when a service-created event fires, and registry entries reachable only from other services; the services `Context` exposes these and transfer memory; GPU-only memory is never charged to a title's pools; (3) observability: a `gpu.txt` artifact, GPU, shader and render counters in `summary.json`, the replay commands and a `--window` flag in the headless contract; (4) architecture and `CLAUDE.md`: presents flow graphics server to GPU to render; new crates `starquake-layout`, `starquake-render` and `starquake-render-wgpu`; every platform starts on wgpu, with copies until a direct backend; the timer thread also drains the graphics server's completion queue; one compile pool shared by shader translation and pipeline builds; and the GPU-test rule for Linux CI.
- **WHY:** Each follows from the graphics rulings or is what the graphics server needs from the kernel, and keeping the approved specs in step with the designs stops agents building against stale text. The write-tracking change keeps the dependency rule that the kernel never reaches up into the GPU crate.

## Services keep the explicit handler style
- **WHEN:** 2026-11-12
- **PROJECT:** starquake
- **SYSTEM:** services
- **SCOPE:** moderate
- **WHAT:** After the handler-style trial on `sm:`, `set:sys` and the first `fsp-srv` subset, services keep the explicit style: a `commands!` table and explicit reads and writes in each method. Two additions close what a glance cannot check: a declared buffer kind per buffer index, and constant command ids accepted by `commands!`. The trial's departure is accepted: `fsp-srv`'s `OpenSdCardFileSystem`, `OpenFile` and `GetSize` wait on host metadata calls on the calling thread; only `Read` is deferred. The services spec is amended to say so.
- **WHY:** Across the three services no method spends more than about a third of its lines unpacking or packing, and review found no layout mistakes in the handlers; every finding was in the framework. Switching to typed signatures would rewrite three services to fix a problem the trial did not show. The blocking calls are fast metadata lookups, and deferring them would add complexity for little gain.

## The repository is public and `main` is protected
- **WHEN:** 2026-11-12
- **PROJECT:** starquake
- **SYSTEM:** process
- **SCOPE:** moderate
- **WHAT:** The owner made the repository public so that `main` could be protected, after inviting a human contributor. `main` now accepts changes only through pull requests whose `linux` and `title` checks have passed, with no force-pushes or deletion, and the rules apply to admins too. No approving review is required, because the agents merge under the owner's account and GitHub does not let an account approve its own pull request; the macOS check is not required because it is skipped on most pull requests. Before the change the history, issues and pull requests were audited: no local paths, dump details, keys or tokens. Commit metadata carries the owner's name and email, as it already does in the owner's other public repositories.
- **WHY:** Branch protection is not available for private repositories on the owner's plan, and with a second person able to push, "never push to main" needed to be enforced rather than trusted. Public repositories also get free Actions minutes, including macOS.


## Services Context gains clear_event and process termination
- **WHEN:** 2026-11-16
- **PROJECT:** starquake
- **SYSTEM:** services
- **SCOPE:** moderate (spec amendment)
- **WHAT:** Amend docs/specs/services.md: the kernel interface (Context) gains clear_event, and a call by which a service ends the calling process (ISelfController::Exit); termination runs after the reply is sent, through the kernel's normal teardown path.
- **WHY:** Eden clears the applet message event at SetFocusHandlingMode(0,0,1), and without clear_event libnx's main loop wakes on every poll; Exit needs a service-initiated termination the spec did not offer. Both are on the real-title path to M2.

## RomFS from a NUP: positional reads and a capped decompression cache
- **WHEN:** 2026-11-16
- **PROJECT:** starquake
- **SYSTEM:** loader / fsp-srv
- **SCOPE:** moderate
- **WHAT:** The file-system service reads a NUP's RomFS through positional reads (read_at on a shared source), so open files share one source. A compressed file (one zstd frame, no random access) is decompressed whole on first open into a size-capped in-memory cache, evicted least-recently-used and re-decompressed on a miss.
- **WHY:** The reader's mutable borrow stops open files sharing the source, and guests read at arbitrary offsets; an in-memory cache keeps it simple and free of host file I/O (which would have to go through Env), with the cap bounding memory.

## M2 runs a title's base package only
- **WHEN:** 2026-11-16
- **PROJECT:** starquake
- **SYSTEM:** loader / core
- **SCOPE:** minor/local
- **STATUS:** superseded the same day by "M2 runs a title's newest package, whole"
- **WHAT:** For M2, a NUP with several packages runs its base package and logs that updates are ignored. How an update combines with its base is ruled on when a title needs one.
- **WHY:** The update rules (replace or overlay the ExeFS, which version wins, how packages relate by title id) need checking against the reference, and nothing on the M2 path needs an update.

## Render capture v2 is designed in M3
- **WHEN:** 2026-11-16
- **PROJECT:** starquake
- **SYSTEM:** render-backend
- **SCOPE:** minor/local
- **WHAT:** The self-contained capture format (v2: creation calls and a serialisable shader description, replayable on a fresh backend) is designed in M3, with the first real backend slices, before R5.
- **WHY:** It is not on the M2 path, and it depends on the shader description, which is not settled yet.

## M2 runs a title's newest package, whole
- **WHEN:** 2026-11-16
- **PROJECT:** starquake
- **SYSTEM:** loader / core
- **SCOPE:** minor/local
- **WHAT:** For M2, a NUP title runs its newest package: that package's ExeFS, and the file-system service serves that same package's RomFS. Base and update are never mixed. A NUP whose newest update is not complete (a patch needing the base) is an error in plain words until overlay rules are decided. Supersedes "M2 runs a title's base package only" (2026-11-16), which was ruled on the wrong premise that nothing chose a package yet.
- **WHY:** `starquake run` already runs the newest update, and the owner's dump shows a NUP update carrying a complete ExeFS and RomFS, so running it whole needs no overlay rules and plays the newest version.

## The run's date comes from the host, recorded
- **WHEN:** 2026-11-17
- **PROJECT:** starquake
- **SYSTEM:** services / observability
- **SCOPE:** moderate (spec amendment)
- **WHAT:** Env gains a wall-clock reading, taken once at boot and recorded so a replay reproduces it; it is the RTC at boot unless a setting pins a date (tests always pin one). time:u owns turning the RTC and uptime into calendar time; the kernel keeps owning ticks.
- **WHY:** Games use the date for save timestamps and daily events, so a fixed 2023 date is wrong for play; reading it once and recording it keeps runs replayable.

## Event hooks are re-attached by their owner after a restore
- **WHEN:** 2026-11-17
- **PROJECT:** starquake
- **SYSTEM:** kernel / gsv
- **SCOPE:** moderate (spec amendment)
- **WHAT:** A snapshot saves which events are hooked and by which service; the owning service re-attaches its callbacks in its restore step, checked against the restored events, before the guest runs again.
- **WHY:** A hook is code and cannot be serialised; keeping re-attachment with the owner keeps service-specific callbacks out of the kernel, and doing it before the guest resumes means no signal is missed.

## Device-share unlocks go through gsv's completion queue
- **WHEN:** 2026-11-17
- **PROJECT:** starquake
- **SYSTEM:** gsv / kernel
- **SCOPE:** moderate (spec amendment)
- **WHAT:** Taking a device-share lock returns a token naming the process and the exact range. When the host GPU finishes, the GPU thread posts the token to gsv's completion queue, and the timer thread, draining it in its recorded order, unlocks through the kernel, which validates the token.
- **WHY:** The unlocking host thread is not a guest thread, so it cannot name 'the current process'; routing it through the recorded queue keeps unlock timing deterministic for replay.

## Env failures: replay failures stop the run, host I/O failures are guest results
- **WHEN:** 2026-11-17
- **PROJECT:** starquake
- **SYSTEM:** observability / cross-system
- **SCOPE:** moderate (spec amendment)
- **WHAT:** An Env call whose replay runs out or disagrees always stops the run as an emulator error, never a guest-visible result. A host I/O failure (a missing or unreadable file) becomes the Horizon result code the reference returns for that condition.
- **WHY:** A replay mismatch means the run is no longer the recorded one, so carrying on would hide it; I/O failures also happen on a real console, and games handle them.

## The syncpoint mode-1 conflict is settled by a trace
- **WHEN:** 2026-11-17
- **PROJECT:** starquake
- **SYSTEM:** gsv
- **SCOPE:** minor/local (spec amendment)
- **WHAT:** syncpt 201's mode 1, where PR #4588 compares as 64 bits and Eden-NX2 does not, joins the graphics-server spec's list of reference disagreements a title trace settles ("What to verify before building", item 1). The code keeps PR's reading, per decision 8, until then.
- **WHY:** It is a real disagreement between the references with no trace yet; listing it with the others means it is checked when traces are taken.

## Defer as-gpu ModifyKind and Remap, and ctrl-gpu QueryEvent 1 and 2, until a title sends them
- **WHEN:** 2026-11-17
- **PROJECT:** starquake
- **SYSTEM:** graphics-server
- **SCOPE:** moderate (spec amendment)
- **WHAT:** The spec's device table marks as-gpu ModifyKind (0xC) and Remap (0x14), and ctrl-gpu QueryEvent 1 and 2, as deferred. They go to the unknown-ioctl policy, which logs them, until a title sends one. The spec says how they would be added: an ordered GPU-memory job with no notification event, plus as-gpu AllocateSpace for Remap. #188 can close without them.
- **WHY:** Only PR #4588's code names them; no trace shows a title using them. Row 30 shows Switch 2 never mapping through as-gpu, and row 149 shows mapping changes arriving as gsv 212, which is already built. Building them would add machinery nothing else needs. Deferring doesn't rule them out, because the unknown-ioctl log names any title that sends one.

## Address-space restore checks attributes against state flags, not a permission table
- **WHEN:** 2026-11-17
- **PROJECT:** starquake
- **SYSTEM:** kernel
- **SCOPE:** moderate
- **WHAT:** Restoring an address space refuses an attribute that the block's memory state doesn't allow, using the reference's own state flags (CanChangeAttribute for Uncached and GpuAccessible, CanPermissionLock for PermissionLocked). Locked ranges are checked against the transfer memory that lent them once the object-table snapshot (#260) lands. There is no per-state permission table: restore keeps accepting any permission a system call can set, and the docs say so.
- **WHY:** The flag checks come straight from the reference, so restore can't refuse anything a live address space could hold. A permission table would be stricter than the live kernel, whose internal path can set read-execute on Normal memory; only the system call refuses that. The only threat here is a corrupt or hostile snapshot file.

## Keep PR's shim placement when a view's start equals its split
- **WHEN:** 2026-11-17
- **PROJECT:** starquake
- **SYSTEM:** graphics-server
- **SCOPE:** minor/local (spec amendment)
- **WHAT:** When a parentless AllocAsEx's split equals its start, as-gpu keeps PR #4588's rule and answers Success. The shim goes on the page below the start, which is GPU address 0 when the start is the first page. The case is added to the spec's verification item 1, so a trace can show whether NvRm ever sends it.
- **WHY:** Nobody knows whether NvRm sends this layout. Refusing it, or switching to NX2's rule for this one case, could break a title that runs under PR. A test already pins the behaviour, and the stakes are low.

## Keep PR's ctrl-gpu interface levels until a trace of NvRm's reads settles them
- **WHEN:** 2026-11-17
- **PROJECT:** starquake
- **SYSTEM:** graphics-server
- **SCOPE:** moderate (spec amendment)
- **WHAT:** The characteristics block keeps PR #4588's nine interface levels at +0x70. The conflict goes into the spec's verification item 1: row 24 asks for every level to be at least 6, NX2's comment asks only for [0x78] at least 6 plus feature thresholds, and NX2 ships 0x40 everywhere. When the title first reaches this ioctl, a reference-emulator trace finds which levels NvRm reads. If the title fails there, NX2's 0x40 values are the first experiment.
- **WHY:** The spec already takes PR's whole layout. PR's values meet NX2's narrower rule but not row 24's broad reading, and nothing says yet which levels NvRm reads. A trace settles it with evidence instead of a guess.

## Cap a channel's waiting submissions at 65,536 entries; past it, answer Busy
- **WHEN:** 2026-11-17
- **PROJECT:** starquake
- **SYSTEM:** graphics-server
- **SCOPE:** moderate
- **WHAT:** Each nvhost-gpu channel holds at most 65,536 entries of submissions waiting for GPU queue room, the same figure decision 12 uses for the queue. A SubmitGPFIFO that would pass the cap answers NvResult::Busy (0xE). It is refused before the syncpoint's maximum is raised, so no fence goes out for it.
- **WHY:** Waiting submissions can't be dropped once their fence has been given out. Without a cap, a title that abandons requests (closing its session, or giving an unwritable out buffer) grows the table without limit. Neither reference has a cap, because neither waits. A well-behaved title waits for its replies and never gets near the cap.

## nvhost-gpu answers event ids 1, 2, 3 and 7 only: an accepted departure from decision 8
- **WHEN:** 2026-11-17
- **PROJECT:** starquake
- **SYSTEM:** graphics-server
- **SCOPE:** minor/local
- **WHAT:** QueryEvent on an nvhost-gpu fd answers ids 1, 2 and 3 (Eden's) and 7 (PR #4588's; NVN asks for it, row 34). Any other id answers BadParameter. This departs from decision 8's take-the-larger rule, which would pick NX2's event-for-any-id. The departure is listed as a divergence, and verification item 1 asks whether NvRm asks for any id besides 7.
- **WHY:** NX2's answer makes a new kernel event for every id a title sends, without bound. PR's answer covers every id that has been observed in use.

## Keep the auto-stub policy, and make its failures diagnosable from one run
- **WHEN:** 2026-11-17
- **PROJECT:** starquake
- **SYSTEM:** services / core
- **SCOPE:** moderate
- **WHAT:** Auto-stub keeps answering success to unknown commands and unknown services. To make its failures easy to trace, the run artifacts name what was stubbed: summary.json and the crash report list the unknown service names and every auto-stubbed command with its interface, and the command-line tool writes the warning log to a run artifact. This is implementation under #207, not a spec change.
- **WHY:** Auto-stub lets one run discover a title's whole service list, which a stricter policy would lose. A stub session for an unknown service has no command table, so it can't tell which commands should return an object anyway. The real cost has been diagnosis: three times (appletOE 10, ReceiveMessage, parental control) a stubbed success sent the title wrong far from the cause, and each time finding the cause needed a scratch build.

## Milestone 2 is reached; file the GPU-front slices and continue into milestone 3
- **WHEN:** 2026-11-18
- **PROJECT:** starquake
- **SYSTEM:** roadmap
- **SCOPE:** moderate
- **WHAT:** Milestone 2 counts as reached by its exit: a title gets as far as its first GPU submission. Whether that submission is correct is checked in milestone 3, when the GPU front executes it. The lead files gpu-protocol slices G8 to G23 as implementation issues and runs them in waves toward milestone 3 (first present), under the implementation-only delegation.
- **WHY:** The title's next blocker is a GPU semaphore release that only the GPU back writes, so the GPU front is the shortest path forward. The slices follow an approved spec, so they fit the implementation delegation.

## Close milestone 2: #33 closed, #205 deferred to milestone 4, #21 closed
- **WHEN:** 2026-11-18
- **PROJECT:** starquake
- **SYSTEM:** roadmap
- **SCOPE:** moderate
- **WHAT:** #33 closes as delivered by PR #127. Its leftovers (dynamic NRO loads through `ldr:ro`, trampoline reach past 128 MiB for large module sets, the NPDM system-call mask check, what `RemoteMemorySize` sizes) go to `docs/proposals.md`. #205's open part (unpatched `cntfrq_el0` and `cntvct_el0` reads disagree with the guest's rate) moves to milestone 4 unchanged, to be decided there. #21 closes, and so does milestone 2.
- **WHY:** The title already loads and runs, so #33's leftovers wait until a title needs them. #205 is deferred because milestone 4 is where a title's frame timing first shows, so the cost of trapping every counter read can be weighed against a real symptom.

## The completion wake: backends get a wake at construction, and the back thread sleeps on one wake-up
- **WHEN:** 2026-11-18
- **PROJECT:** starquake
- **SYSTEM:** graphics (render-backend, gpu-protocol)
- **SCOPE:** architectural (spec amendment)
- **WHAT:** Core builds each render backend with a completion wake, as it passes the compile pool. The render thread stores each new `completed()` value and then calls the wake. The wake goes through the GPU ring's consumer wake-up, so the back thread sleeps on one thing and wakes for ring entries or for completion. Release writes stay on the back thread. A timer wake-up that finds `completed()` moved is logged and counted as a lost wake, and a contract test every backend must pass checks that a wake follows every advance. G15 builds against it; R5 implements it for wgpu.
- **WHY:** It keeps every write the guest sees on one thread, in ring order, and reacts at once when the GPU finishes. A lost wake is bounded by the ring's 100 ms recheck, then detected and caught by a test. Eden's fencing thread carries the same obligation inside `wait()` and splits release writes across two threads. Polling would add delay to every semaphore the guest waits on.

## gpu-protocol errata from wave 36
- **WHEN:** 2026-11-18
- **PROJECT:** starquake
- **SYSTEM:** gpu-protocol
- **SCOPE:** moderate (spec amendment)
- **WHAT:** (1) `SET_REFERENCE` stays a ring entry: G4's ring gains the kind, and the back closes the open list with no action attached, as Eden-NX2 does. (2) Opcode 0 or 2 with tertiary op 0 is an old-format incrementing or non-incrementing method, per `clc76f.h`; only tertiary ops 1–3 are subdevice-mask operations, which are skipped. (3) The macro upload pointer leaves the side-effect list: the macro host (G13) keeps the upload as a word offset (pointer times 3). (4) Method 0xCC bit 12 is the inline QMD's `REQUIRE_SCHEDULING_PCAS` (bit 140); a clear bit schedules the QMD at once, as a cited rule. (5) A failing QMD request leaves nothing if it fails before its table entry is written, and keeps its own change if it fails after. (6) In shadow-RAM replay mode, data-upload methods take the written words; only state methods take the shadow value (a guess until a trace shows replay during an upload).
- **WHY:** Nvidia's headers settle layouts (decision 10), and where the Switch 2 references agree and the spec is silent or wrong, the spec follows them. (2) and (6) avoid dropping real methods and uploading stale data, and (1) and (3) match what the code and both references already do.

## gpu-protocol errata from waves 37 and 38
- **WHEN:** 2026-11-18
- **PROJECT:** starquake
- **SYSTEM:** gpu-protocol
- **SCOPE:** moderate (spec amendment)
- **WHAT:** (1) A 3D pipeline's program address comes from its own `SET_PIPELINE_PROGRAM_ADDRESS_A/B(j)` (`clc797.h`); there is no `SET_PROGRAM_REGION`. (2) The facts table says `clc7c0.h` lists the compute macro calls (`CALL_MME_MACRO` at byte 0x3800). (3) Two macro code stores per process, one 3D and one compute; a channel's compute class uses the 3D store until it writes 0x45 or 0x47 (both references' `NativeMacroContext`), refining decision 11. (4) A start table holds 256 entries, the header's `CALL_MME_MACRO` count (Eden-NX2 wraps at 0x80), a guess until a trace shows a call past 0x80. (5) A catch-up rerun starts from the start recorded at the call and runs the code as it is at the rerun, a listed divergence from Eden-NX2 pinned in G13's tests. (6) Overlapping copies: a pitch-to-pitch DMA copy reads each line whole before writing it, as Eden-NX2's per-line `CopyBlock` does; a copy with a block-linear side, and every 2D blit, reads the whole source region first, at most `MAX_COPY_BYTES`. (7) Guessed bounds, each to confirm by trace: 2^18 method writes and 0x1000 DMA words per macro run; `MAX_COPY_BYTES` 256 MiB and `MAX_PENDING_RANGES` 1,024; a size for all 32 vertex streams (the header names 16, Eden-NX2 sizes 32; a decision-10 divergence); `STREAMED_AFTER_FAULTS = 4` and eviction limits at half the budget in the buffer cache. (8) In shadow-RAM replay mode every method except data loads takes the shadow value, triggers included, as Eden-NX2's `CallMethod` does. (9) The 3D and compute group tables and typed accessors are hand-written, keyed on the generated constants, with a test that each entry names a generated method: an accepted departure from decision 6. (10) The ring's `Release` entry, and the copy and QMD releases, carry the issuing channel, so the back knows which `GpuMemory` a release writes (G19, with G4's ring). (11) The buffer cache re-reads a range on its first use after its watch is armed, as built in G16 (PR #346).
- **WHY:** Nvidia's headers settle layouts and fields (decision 10). Where both Switch 2 references agree and the spec is silent or wrong, the spec follows them. Behaviour already built and reviewed is written into the spec rather than left as an unlisted departure, and guessed bounds are recorded so a trace can confirm them.

## Shader and render slice plan errata
- **WHEN:** 2026-11-18
- **PROJECT:** starquake
- **SYSTEM:** shader-pipeline, render-backend
- **SCOPE:** moderate (spec amendment)
- **WHAT:** Shader-pipeline: the slice table and critical-path sentence agree (13 on 8, 9, 12e and 12f; 10 on 3, 6h and 9; 15 on 2, 8 and 12g; every 12x on its 6x and 7; 8, 11 and 18 on 7; 6d on 4, 6a, 6b and 6e; 12h also on 10), and the cache (19) is on the critical path. Owners for unowned work: pass 6 (local and shared memory) in slice 16; the IR build step in slices 10 (blocks) and 11 (the structured IR the translators fill); `cargo xtask shader-nvdisasm` in slice 4 as a follow-up tool; the speed test in slice 17. One shader capture format: slice 20 defines it, `shader-replay` reads it, and the GPU capture (`SQGC`, G20) embeds it. PR #4588's test file is `src/tests/video_core/native_shader.cpp`. Render-backend: R9's second part runs on G21's recording-backend wiring in `core`, not R12, and decides the dump flag's path from `core` to the GPU crate; R12 depends on R9b, R5 and R8. The stale "Open with #12" line about the completion wake is replaced by a pointer to "The completion wake" (decision 14).
- **WHY:** The spec's tables must match what the slices actually need so waves can be scheduled from them; the drafts of the filed issues showed where they did not, including a cycle between R9b and R12 and work no slice owned.

## Render capture v2 no longer gates R5
- **WHEN:** 2026-11-18
- **PROJECT:** starquake
- **SYSTEM:** render-backend
- **SCOPE:** moderate (spec amendment)
- **WHAT:** R5 starts now. The lead drafts the render capture v2 design (creation calls and a serialisable shader description, replayable on a fresh backend) as a design PR for the owner, and R11's second part waits for it. This supersedes the "before R5" part of the 2026-11-16 entry "Render capture v2 is designed in M3"; the rest of that entry stands.
- **WHY:** R5 is the critical path to a first frame, and only R11's second part depends on the capture format, so the design proceeds in parallel instead of blocking R5.

## Render capture v2: the design in PR #395, as recommended
- **WHEN:** 2026-11-18
- **PROJECT:** starquake
- **SYSTEM:** render-backend, observability
- **SCOPE:** moderate (spec amendment)
- **WHAT:** The render capture v2 design is approved, with its five choices as recommended. (15) A capture stores each shader's inputs as slice 20's record and retranslates on replay; SPIR-V or MSL a backend received is stored as it is, and a module with no origin is recorded as missing. This reads the 2026-11-16 entry's "serialisable shader description" as the inputs, not a serialised naga module. (16) Replay maps the capture's ids and fence values to the fresh backend's, and refuses a list naming an id it does not hold. (17) `starquake run --capture render` writes `render.sqrc` into the artifacts folder for the whole run, with `--capture-limit` (default 4 GiB, a guess). (18) Version 1 files are refused with the exact version error. (19) A capture from another translator version is refused by default; `--retranslate` replays with the new translator and reports every shader whose entry or interface changed. R11's second part splits into R11b (the render crate) and R11c (the wiring, after R12, shader slice 20 and G18); R5 and R6 need no new code, only the constraints the design lists.
- **WHY:** Storing inputs keeps the one shader capture format, survives naga upgrades, feeds nothing from a file to naga or wgpu unchecked, and replays on a later direct backend; a serialised naga module cannot trust its `ModuleInfo` and can overflow the stack on a crafted file, WGSL text is lossy, and wgpu never hands back final code. Mapping ids avoids a shared allocator that does not exist. Refusing old or mismatched captures by default keeps a replay from silently comparing different things.

## Owner rulings on the wave 39 to 43 questions: the lead's recommendations, taken as a batch
- **WHEN:** 2026-11-19
- **PROJECT:** starquake
- **SYSTEM:** graphics (gpu-protocol, render-backend, shader-pipeline, observability)
- **SCOPE:** moderate to architectural (spec amendment)
- **WHAT:** The owner took the lead's recommendation on each of the 46 "awaiting the owner" rows of `docs/proposals.md` under "From wave 39" to "From wave 43". Rows with no recommendation of their own are accepted as built, provisional where the code marks a guess. The rulings, in row order:
  1. G14's guesses and departures (#325) are accepted and listed in gpu-protocol as choices.
  2. `Syncpoint` ring entries carry a count (#325).
  3. GPU-modified images stay until the budget refuses new ones; revisited with write-back eviction in M4 (#328).
  4. `ShaderStamp` is a digest of the shader cache's entry key, defined with shader slice 21; 128 bits stays a guess (#334).
  5. Viewport, scissor, blend constant and stencil reference are always dynamic (#334).
  6. `MAX_PIPELINES` = 65,536 as a guess; a key is recorded when the backend accepts the pipeline (#334).
  7. The `TextureKind` question gains a usage field (#348).
  8. A stage is specialised on what the previous stage stores (Eden's `MakeRuntimeInfo`), not on what the next reads (#348).
  9. New slice G24 (`CLEAR_SURFACE`) after G18 (#329).
  10. Slice 19's rejection names the failing program; "once per program" stays (#329).
  11. Slice 19 adapts to G18's `ShaderLookup`/`SetLookup` seam; the spec names the ticket wait (#329).
  12. Swizzles are bound by two lookups, the second only when they differ (#329).
  13. `GpuMemory::generation` comes from the VM's `Translate::generation` (#333).
  14. `MME_DMA_READ` (0x157) is performed at the write and the memo stores no call that writes it; the compute class's MME DMA methods are register writes only, until a trace says otherwise (#393).
  15. Eden's local-memory formula stays until verification E3 (#349).
  16. Sampler LOD bias moves into the shader (slice 17, for R7); guest 1D arrays become 2D textures of height 1; the stencil aspect of `D32_FLOAT_S8_UINT` comes with the D24S8 conversion; the anisotropy and negative-LOD-clamp rules stand (#382).
  17. render-backend errata from R5: wgpu pin 30.0.2; wgpu refuses depth clears outside 0..=1 and clears of compressed formats (#382).
  18. The wgpu backend treats the Apple GPU family as meeting "subgroup size 32" (#383).
  19. No driver pipeline cache under wgpu for now; revisited with a direct Vulkan backend (#383).
  20. `ShaderInterface` carries the shader's requirements and binding-array counts (#383).
  21. Deferred instances are drawn at G19's boundaries, with no macro-end ring entry, as built (#330).
  22. Presents read through the space `set_present_space` names, as built; the format mapping and the ignored pitch stay guesses (#330).
  23. New slice G25 (`LOAD_CONSTANT_BUFFER` data written to guest memory) after G19 (#330).
  24. R8's choices as built: `WindowSize` in `WindowHandles`; the rotation corners, `FRAME_LATENCY` = 2 and the letterbox aspect stay guesses (#385).
  25. render-backend errata from R8 and R11b: verification 11 settled with no `unsafe`; the surface is made in `set_present_target` on the main thread on macOS; "Render capture v2" is approved, not a draft (#385, #388).
  26. R11b's bounds and departures as built, guesses still marked (#388).
  27. The program content hash is seeded with the header length, 0 or 128 (#350).
  28. Guest y-flips and depth remaps: an `EmulationFlags` field and a driver constant handle them in the vertex shader (slice 17); the backend keeps the host viewport positive (#384).
  29. The release timestamp is guest time from the kernel's clock, through the `Gpu`'s config (#332).
  30. `is_idle` waits up to `IDLE_POLL_WAIT` for a poll, as built (#332).
  31. No `destroy_address_space`, and presents read through the first space made, as built and provisional (#332).
  32. A restored run uses the null front until a front snapshot exists, as built (#332).
  33. The decoder framework gains known bits (with their own divergence accounting) and a per-form canonical word, in a small slice before 6c and 6d (#351, #352).
  34. NAK's integer encodings stay until the `nvdisasm` check, E2 (#351).
  35. `OUT.FINAL` and `ISBEWR` as built (#357).
  36. render-backend errata from R7: typed outputs are required by wgpu; every non-draw command is outside a pass under wgpu; R7 sets the strip index format (#384).
  37. When the GPU answers busy, the parked core asks again when the GPU next reports quiet, up to a named retry count (PR #422 review).
  38. The macOS gate stays on pushes to `main`; the lead checks `main`'s last macOS run after every wave (PR #422, #423).
  39. PR #423's timer thread sleeps in pieces sized by measured lateness (no `unsafe`); the held-page test times the timer thread's recorded note; the vsync bound is not tightened now (#423).
  40. render-backend names the recording backend's no-call-log mode that runs use (#386).
  41. Frames as built: a fenced present counts as a frame by host timing, stated as such; the dump flag's path is `ExecConfig::dumps` in `GpuConfig::exec`; observability states which frames a run keeps (#386).
  42. G20's choices as built (#331).
  43. The surface forms with a constant-buffer handle (`SULD` 0xb97, `SUST` 0xb9b, `SUATOM` 0x593) are decoded from PR #4588, marked for the `nvdisasm` check (#356).
  44. The memory decoder's choices as built (#355).
  45. Flow analysis bounds and choices as built; the bounds stay guesses (#362).
  46. The interpreter's choices as built; guesses stay marked and `interp` stays public (#360).
- **WHY:** The owner's batch ruling ("Take all", 2026-11-19) on the lead's recommendations, for the reasons each row gives. In brief: a count bounds a guest-chosen increment run (2); write-back eviction belongs with M4 (3); wgpu, Vulkan and Metal all set the four states per draw (5); builds finish asynchronously, so acceptance is the only point a key can be recorded (6); without a usage field, a shader that only samples is retranslated on a format change (7); the GPU crate cannot build the next stage's reads from engine state (8); a first present needs clears and constant-buffer data in memory, and no slice owned them (9, 23); naming the program keeps the spec's per-program log (10); G18's seam already carries what the draw path needs (11); generation needs one source (13); performing 0x157 at the write matches Eden-NX2 (14); a trace or verification settles the guessed encodings and formulas (15, 34, 43); wgpu has no sampler LOD bias, no 1D arrays and no aspect on copies (16); Apple GPUs run 32-wide SIMD groups while wgpu reports 4..64 (18); wgpu's cache API is `unsafe` and Vulkan-only, so the Mac gains nothing (19); the backend should not dig layouts out of the naga module (20); seeding by header length keeps compute and graphics entries apart (27); wgpu refuses negative viewports and reversed or out-of-range depth ranges, and guest y-flips are likely common (28); host nanoseconds break replay (29); a busy answer must not hide a stall until the timeout (37); macOS minutes cost ten times Linux's, so a check after every wave replaces running it on PRs (38); one precise wait would need `unsafe` outside `cpu` (39); a run's recording backend must not grow with each present (40); building on a decoder framework that can represent both references keeps divergences counted (33). Behaviour already built and reviewed is written into the specs rather than left as an unlisted departure, and its guesses stay marked so a trace can confirm them (1, 4, 12, 17, 21, 22, 24 to 26, 30 to 32, 35, 36, 41, 42, 44 to 46).

## Owner rulings on the wave 44 and 45 questions and the first title bring-up
- **WHEN:** 2026-11-19
- **PROJECT:** starquake
- **SYSTEM:** graphics (shader-pipeline, gpu-protocol, render-backend, observability), execution model, services, CI
- **SCOPE:** moderate to architectural (spec amendment)
- **WHAT:** The owner took the lead's recommendation on each of the 14 "awaiting the owner" rows of `docs/proposals.md` under "From waves 44 and 45". Rows with no recommendation of their own are accepted as built, provisional where the code marks a guess. The rulings, in row order:
  1. The translation driver becomes shader slice 12i, after 12a to 12h: it walks `IrBlocks::sources`, decodes each flow block and dispatches to the group translators before each block's terminator, honouring `guard_tested`; it owns the shared guard helper and source reader and retires slice 1's straight-line `MOV` path; slices 13, 14, 15 and 17 depend on it (#364, #365, #370, #371).
  2. A guarded instruction becomes a conditional block, as the spec says, pure ones included; selects are a later optimisation pass, and slice 10's flow docs follow the spec (#364).
  3. Follow Eden's order: the structurizer runs first, then SSA. Slice 8 is reworked (slice 8b) to take predecessors from the structured tree and accept structured functions, and slices 13 to 17 depend on that rework (#361). This replaces the first form of this ruling, SSA before structuring, which the owner corrected on hearing the trade-offs.
  4. Slice 8 and 11 choices as built, guesses still marked: single-walk dead code elimination, `MAX_STRUCTURE_WORK` = 2^28, `ir::MAX_NESTING` = 256, conditions kept in variables (#361, #363).
  5. The translators' reference conflicts as built, marked as guesses where they are: PR #4588's native visitor followed, the interpreter's `INACTIVE_LANE_VALUE`, `F2I.FTZ` with a half operand not emulated, `IPA.OFFSET` carried in the IR, `AL2P` and `PIXLD` from Eden (#365, #370, #371).
  6. `ClearDesc` gains colour and stencil write masks and backends draw masked clears; the GPU crate stops parsing `clear.wgsl`; `SET_SURFACE_CLIP` stays a listed divergence; integer clear values await a trace (#428).
  7. A constant-buffer load past the buffer's size writes the in-bounds words and refuses the rest (#429).
  8. `Syncpoints::increment_by` pushes in steps of at most 2^31 - 1 (#431).
  9. The parked core's re-ask as built: `GPU_QUIET_RETRIES` = 8 is a guess, and the bound is per park (#436).
  10. Apple subgroups as built: the name-prefix identification is a guess, and Apple GPUs reached through Vulkan are not counted (#433).
  11. PR #451 merges; gpu-protocol gains the payload-only report (a 3D or compute `SET_REPORT_SEMAPHORE` release that reports no counter is an ordinary fenced release, 4 bytes or 16 with the timestamp); counter reports write nothing until a title needs one, then Eden-NX2's stand-in of 1, then host queries (PR #451).
  12. `IApplicationFunctions` 1 (`PopLaunchParameter`) answers Eden's "no data in channel" result for both kinds, as a listed divergence, until accounts are designed (title bring-up, blocker 3).
  13. Guest RAM is sized from the console's memory arrangement in the program's metadata (the application pool, less the system resource's charge) and backed lazily; this replaces `WORKING_RAM` (title bring-up, blocker 4).
  14. `cargo xtask ci` gains a step that checks the app and each crate on its own; CLAUDE.md's gate list changes when the step is built, not before (PR #446).
- **WHY:** The owner's batch ruling ("Take all", 2026-11-19) on the lead's recommendations, for the reasons each row gives. In brief: the per-group translators have no caller, and a first frame needs one (1); the spec already says conditional blocks, and a select is an optimisation that can come later (2); as built, neither order chains: the structurizer takes a function before SSA, refusing phis (`StructureError::Phi`) and any value that crosses blocks, and expects SSA afterwards to turn its goto and condition variables into values, while slice 8's SSA rewrite refuses structured functions (`PassError::Structured`); Eden's `translate_program.cpp` runs `BuildASL` (line 337) before `SsaRewritePass` (line 389), and following it is the smaller fix: slice 8 adapts, the structurizer stays as built, one SSA pass cleans up the structurizer's variables too, and slices 13 to 17 see a single SSA form. The lead's first recommendation misstated SSA-first as Eden's order, and the owner corrected the ruling (3); a write that stops at the size bound is closer to Eden-NX2, which writes the whole run, than refusing it all (7); gsv's completion queue compares by 32-bit wrapping and would drop a change 2^31 or more ahead (8); the GPU crate should not parse the render crate's shaders, and backends know which clears their host API can do (6); a title spins on the payload-only report at the end of its first flush, so it cannot run past it (11); the SDK dereferences the zeroed auto-stub reply, and Eden's answer gets the title further until accounts are designed (12); a fixed 16 MiB allowance ran out early in the first title's start-up, and lazy backing costs the host only what the title touches (13); a feature-unified workspace build hid a package that did not build on its own (14). Behaviour already built and reviewed is written into the specs rather than left as an unlisted departure, and its guesses stay marked so a trace can confirm them (4, 5, 9, 10).

## Owner rulings on the wave 46 questions
- **WHEN:** 2026-11-20
- **PROJECT:** starquake
- **SYSTEM:** graphics (shader-pipeline, render-backend, gpu-protocol)
- **SCOPE:** moderate (spec amendment)
- **WHAT:** The owner took the lead's recommendation on each of the 8 "awaiting the owner" rows of `docs/proposals.md` under "From wave 46". Rows with no recommendation of their own are accepted as built, provisional where the code marks a guess. The rulings, in row order:
  1. Warp operations and derivatives: new shader slice 16b after 16 lowers votes, ballots, lane masks, shuffles, quad operations, `FSwizzleAdd` and derivatives to naga's subgroup and derivative operations, with the subgroup-size-32 requirement where the reference needs it; slice 17, and every slice that lowers pixel programs, depends on it (#375).
  2. Vertex-output interpolation: the vertex stage's variant key carries the paired pixel program header's interpolation modes, which the GPU crate reads when it builds the draw; a named exception to ruling 8 of the 2026-11-19 entry "Owner rulings on the wave 39 to 43 questions" (#374).
  3. Metal fast math: every lowering slice routes float comparisons and float-to-integer conversions through the NaN-safe helpers in `lower/ops.rs`; render-backend records the measured behaviour (#375).
  4. Shared atomics and narrow shared stores (`WriteSharedU8`, `WriteSharedU16`): new shader slice 16c after 16 lowers memory atomics, shared and global (12e's), with an atomic shared array and a compare-and-swap loop where naga has no direct atomic, as Eden does (#375).
  5. 12e's choices as built: where PR #4588's lowering and its native translator disagree, the lowering is followed (graphics-server decision 8): bit 76 of the uniform `LDG`/`STG` is a memory descriptor, and `LDGSTS` takes any enable predicate; `.STRONG.SM` ordering is ignored, as the native translator does (#368).
  6. The release timestamp's unit is GPU ticks at 614.4 MHz (Eden's `GPUTickFreq`), a guess for Switch 2 that a hardware trace would settle; the clock goes through `GpuParts::fence_clock`, not `GpuConfig`; gpu-protocol names both (#435).
  7. R6 keeps the naga cross-check in production; the shader crate gains a narrower "needs subgroup size 32" flag, so a quad-only shader is not refused where size 32 is not met (#434).
  8. `DESCRIPTOR_BUFFERS` = 18 (PR #4588's bound) and the `TEX.B` operand layout (PR #4588's, where NAK disagrees) are accepted as built, as guesses (#369).
- **WHY:** The owner's batch ruling ("Take all", 2026-11-20) on the lead's recommendations, for the reasons each row gives. In brief: slice 16 refuses warp operations and derivatives, no slice owned them, and pixel programs use derivatives and quad operations often (1); wgpu refuses a pipeline whose vertex outputs and fragment inputs differ in interpolation, and only the pixel program's header gives the mode, so the vertex stage cannot learn it from what an earlier stage stores (2); wgpu-hal 30.0.2 leaves Metal's fast math on, under which a plain comparison of two NaNs is true, and naga's float-to-integer conversion saturates to 0x7fffff80 rather than the maximum, so one set of helpers keeps every slice's results the interpreter's (3); atomics and narrow stores both need the shared array to be atomic, one change best made by the slice that lowers the atomics (4); PR #4588 runs titles through its lowering by default, which makes the lowering's reading "the one a title ran on" (5); 614.4 MHz is the Switch 1 rate that both Switch 2 references keep, and `GpuConfig` is `Copy + Eq` settings that a shared clock cannot join (6); wgpu-core 30.0.2 does not compare a binding array's size with its layout, so without the cross-check a wrong count would reach the host driver, and a quad operation does not need a 32-wide subgroup (7). Behaviour already built and reviewed is written into the specs rather than left as an unlisted departure, and its guesses stay marked so a trace or `nvdisasm` check can confirm them (5, 6, 8).
