# Proposals

Work that agents have proposed and the owner has not yet vetted. **Nothing here is a commitment.**
GitHub issues are for work the owner has chosen to do; this file is where candidates wait.

How it works:

- Agents add entries here (wave scope reviews, review findings, ideas noticed in passing). They
  do not file GitHub issues for proposed work.
- The owner vets an entry. A promoted entry becomes a GitHub issue and its status changes to
  `promoted #N`. A dropped entry stays, with status `dropped` and the reason.
- Small items that belong to an existing issue are folded into that issue as a comment, and
  listed under "Folded" so the trail survives.
- Keep entries short: what, why, where it came from, and a suggested label and milestone.

Statuses: `proposed`, `promoted #N`, `folded into ...`, `accepted`/`ruled` (spec deviations), `adopted`, `dropped: <reason>`. Entries marked "(lead, under delegation)" were decided by the lead in autonomous mode and are open to the owner's review.

## Work items

### From wave 2 (ledger: #22)

| Status | Proposal | Why | Suggested |
|---|---|---|---|
| promoted #63 | **chore(xtask): unused-dependency check in the gate**; drop the unused workspace `bitflags` | #39 removed the IPC crate's unused dependencies by hand, but nothing stops it recurring. `bitflags` is now unused in the root `[workspace.dependencies]`. | infra, M1 |
| promoted #64 | **chore(ipc): de-duplicate the request and reply builders** and the test-side assembler; export the domain limit constants | Duplicated buffer-write logic, inconsistent overflow behaviour (saturate in one place, `FieldOutOfRange` in another), a test assembler the request builder supersedes, and inline literals for the domain header limits. | area:ipc, M1 |
| promoted #65 | **docs: pin libnx as a reference** and verify the request builder's raw-size and domain conventions against it | The builder's size conventions are marked as guesses. libnx is not a pinned reference, and Eden only parses requests. | area:ipc, M1 |
| promoted #66 (trimmed: offline switch and revert titles; the scheduled run and cache budget are dropped for now) | **ci: cargo-deny operability**: a scheduled advisory run, a documented triage and ignore path, an offline switch for `cargo xtask ci`, GitHub's `Revert "..."` PR titles, and the macOS PR job's cache budget | A new advisory can turn `main` and every PR red with no code change; the gate now needs network access; the title check rejects GitHub's own revert titles; per-PR macOS caches count toward the 10 GB limit. | infra, M1 |
| promoted #67 | **infra: shared build-folder safety across worktrees** | During wave 2 a stale or sibling `xtask` binary ran under `cargo xtask ci` in one worktree, and mutated scratch copies left stale artifacts. | infra, M1 |
| promoted #68 (blocked on the owner installing devkitPro) | **test: a homebrew-built NSO/NPDM fixture** | Several loader facts stay unconfirmed without a real file: NSO hash flags, module-name words, the wildcard form, the ACID access-control base, and the ACI0-against-ACID rules. Overlaps the test strategy's homebrew fixture decision (#8). | area:loader, M1 |
| dropped: the merge queue now waits for PR checks; the self-hosted runner question is recorded in the test strategy spec | **decision: required-checks policy** | Branch protection was unavailable while the repo was private (HTTP 403); since 2026-11-12 the repo is public and `main` is protected (decision log). The macOS PR job is path-filtered and may be absent, the merge queue must check the checks itself (wave 2 merged two PRs before their CI finished), and a move to squash merges would leave the PR-title check as the only subject enforcement. | owner, infra |
| folded into the `fsp-srv` trial slice (service framework spec, slice 6) | **feat(fs): port the path normaliser** (`fs_path_utility`) | RomFS `find_dir`/`find_file` reject non-normalised paths, so the file-system service needs normalisation that matches Eden's. #16 cited the source but did not port it. | area:loader, M2 |
| partly delivered by #95 (the NUP container reader); the rest is proposed again under "From waves 8 and 9" | **feat(loader): ExeFS title loading and RomFS mount**, including the NCA container and the `path_hash` signedness question | The PFS0 and RomFS parsers are reachable only from tests. Titles need a seekable source from an NCA container, and hashing of bytes 0x80 and above is unsettled. | area:loader, M2 |
| folded into #33 | **feat(loader): expose the Switch 2 `RemoteMemorySize` word** (META 0x10) | Not exposed by NPDM parsing; Switch 2 process setup will probably need it. | area:loader, M2 |

### Folded into existing issues (wave 2)

- Validation call sites and the refuse-or-warn policy → #33.
- The virtual-to-physical core map and core count (4 or 6) → #9.
- Wiring the request builder into synthetic guests and golden traces → #8.
- Whether `proptest-regressions/` is committed → #38.

### From wave 3 (ledger: #22)

| Status | Proposal | Why | Suggested |
|---|---|---|---|
| promoted #75 (lead, under delegation) | **feat(kernel): finish the dispatcher: pass declared tables to calls as disjoint borrows**, with a real compile-fail demonstration and the first calls that touch a table | #56 landed the dispatcher, but no implemented call takes a table yet: the lock is held and never read, and the two `compile_fail` doctests only show that a field is private. The table-parameter convention the kernel spec is built on isn't exercised until this exists. | area:kernel, M1; do before kernel slices 4–10 |
| folded into #75 (lead, under delegation) | **chore(kernel): one result-code encoding and one `Handle` type** | #57 and #56 were built side by side: `handle.rs` encodes Horizon results itself, duplicating `result.rs`, and the two may define `Handle` twice. | area:kernel, M1 |
| folded: the dispatcher part into #74, the rest into kernel slices 3–10 as they land (lead, under delegation) | **feat(trace): wire the recorder into the dispatcher, scheduler and address space; write `events.bin` on crash or timeout; one result word shared with the kernel** | The dispatcher's recording point is a no-op, and trace and kernel each have a `ResultCode`. Milestone 1's hello-world is meant to produce the first replay log. | M1 |
| promoted #78 (lead, under delegation) | **infra: lint banning direct host clock and file-system calls in the kernel and service crates** | Observability spec, order of work step 2. #55 left it out and nothing owns it. It is what keeps replay deterministic. | infra, M1 |
| folded into #75 (lead, under delegation) | **chore(kernel): a `Bool` kind and a word-size test in the system-call convention table** | `Kind::Word` accepts any type name and assumes 4 bytes; `CreateSession` and `CreatePort` take 1-byte bools, so rows beyond the retail calls would convert wrongly. Also commit the script that generated the table from Eden's wrappers. | area:kernel |
| folded into #74 (lead, under delegation) | **test(loader): NRO round trip against the testkit's builder; share the `MOD0` and homebrew constants** | Nothing checks the loader's and the testkit's NRO layouts against each other, and they duplicate constants. | area:loader, M1 |
| promoted #77 (lead, under delegation) | **feat(services): framework core, then control requests and domains** (service port slices 1–2) | The service spec is approved and these slices need only the IPC crate and a fake kernel, but no issue exists. Milestone 1 needs a request to reach `sm:`. | area:services, M1 |
| promoted (lead, under delegation): issues are filed when #75 lands | **feat(kernel): port slices 4–10**, filed after the scheduler trial (#59) reports | Milestone 1's critical path. Deliberately waiting on the trial's verdict on the tables-and-ids design. | area:kernel, M1 |

### Spec deviations from wave 3, awaiting the owner

Places where wave 3's code narrowed or departed from an approved spec. Each needs a ruling:
accept it and amend the spec, or change the code.

| Status | Deviation | Spec |
|---|---|---|
| accepted (lead, under delegation): a replay that runs out or disagrees stops the run with a replay error; it is never a guest-visible result. Kernel spec amended | `get_system_tick` is fallible: the environment's clock returns an error when a replay runs out or disagrees, and the dispatcher then writes no registers. The spec sketched an infallible call. | kernel, observability |
| ruled (lead, under delegation): the kernel owns it. The environment returns host nanoseconds; the kernel converts with the process's counter frequency (31.25 MHz for Switch 2, 19.2 MHz for Switch 1); `core` gives the same frequency pair to the loader's patcher, with a test that the two agree. Kernel spec amended; done in #75's follow-up slices | **Nothing owns the conversion from the host clock to guest ticks.** The kernel assumes its environment already counts guest ticks, and converts nanoseconds at 19.2 MHz (the Switch 1 rate) in one place; Switch 2 processes need 31.25 MHz. Kernel, core and trace each leave it to another crate. | execution model, kernel |
| accepted (lead, under delegation): the golden trace holds system calls, IPC requests, guest log lines and lifecycle events per thread; wake, map, fault and schedule events stay out because they depend on timing. Observability spec amended | The golden-trace text view omits `Schedule` events; whether `Wake`, `Map`, `Fault` and `Lifecycle` lines belong in it is open. | observability, testing |
| accepted (lead, under delegation) for now: the kernel records while holding its own lock, so the orders agree. Revisit if the dispatcher's lock-wait measurement shows contention | The recorder has its own mutex, because the kernel lock isn't in its crate. The spec has sequence numbers assigned under the kernel's lock. | observability, kernel |
| ruled (lead, under delegation): add a fourth status for planned calls whose slice hasn't landed; in #75 | The system-call table has three statuses; calls the spec lists as "built now" but whose slice hasn't landed are marked deferred. A fourth status or a slice field may be wanted. | kernel |
| accepted (lead, under delegation): all three are hardware-facing | `QueryIoMapping`, `ReadWriteRegister` and `QueryPhysicalAddress` are classified "not planned"; `SleepSystem` is left deferred. | kernel |
| accepted (lead, under delegation) as provisional: the dump is versioned; the deferred-reply completion event arrives with the kernel's session slice | Event field types and the flight-recorder dump layout are guesses; typed environment methods for input and host file and network I/O are deferred behind a generic extension point; no event yet records the completion order of deferred replies (observability rule 3). | observability |

### Folded into existing issues (wave 3)

- NPDM handle-table size, capability check-and-log, and clock-to-guest-ticks placement → #33.
- A real libnx-built NRO as a pinned fixture → #68.
- Unported handle-table parts and the typed current-process and current-thread lookups → kernel slices 4, 8 and 9.
- The deferred-reply completion event → kernel slice 9 and service slice 3.
- Lock-wait measurement in the dispatcher → kernel slice 6.
- Golden-trace line kinds → the milestone 1 bring-up (#20).

### Infrastructure observed in wave 3

| Status | Proposal | Why |
|---|---|---|
| mitigated (lead, under delegation): single-agent tasks use their own build folder; waves keep the shared one until #67 lands | **Shared build folder: stale cross-target artifacts** | One agent's Linux-target clippy run failed with spurious import errors from a sibling's build in the shared folder; an isolated folder passed. Wider than #67, which covers only the `xtask` binary. |

### From wave 4, the scheduler trial (ledger: #22)

The trial's verdict (#59, PR #72): the scheduler ported onto tables and ids without needing to
borrow two rows at once. Copying neighbour ids out and editing one row at a time was enough, and
queue walks are bounded loops over lookups by id. The awkward part was error handling for lookups
of ids the kernel itself stored. The verdict is partial: walking wait lists is only exercised by
slices 4 and 5.

| Status | Proposal | Why | Suggested |
|---|---|---|---|
| ruled (lead, under delegation): serialise the thread rows plus each queue's ordered list of ids, rebuild the links on load and validate them. Not on the milestone 1 path, so the work stays proposed | **decision: how the scheduler table is serialised** (validate every link on load, or rebuild the queues from the rows keeping per-level order), and amend the kernel spec | The spec says tables are serialisable from slice 2, but the scheduler's linked queues aren't yet. Snapshots depend on it. | owner, area:kernel |
| folded into #75 (lead, under delegation) | **chore(kernel): a slab helper for internal lookups**, with one fixed error for a stale id the kernel itself stored | Removes repeated error-mapping noise in the scheduler and every later slice. Recommended by the trial; do before slice 4. | area:kernel, M1 |
| proposed | **feat(kernel): pinned threads** (the pinned-thread lookup, core-migration disables, suspend-allowed flags) | No port slice owns it, and Eden's scheduler depends on it. Needs a row in the spec's port plan. | area:kernel, M1 |
| accepted (lead, under delegation): carried into those issues when they are filed | **Additions to the briefs for kernel slices 4–10** when they are filed: slice 5 splits requested and in-state suspend flags and adds kernel-waiter checks; slice 8 makes the yield count per process and adds virtual-to-physical core translation and the core-mask call's checks and waits; slice 10 takes queue rotation (time slicing) | Found while porting the scheduler; each belongs to a later slice. | area:kernel |

### From the walking skeleton (#74) and wave 5 (ledger: #22)

All vetted by the lead under delegation; open to the owner's review.

| Status | Proposal | Why |
|---|---|---|
| promoted #85 (lead, under delegation) | **The hypervisor backend runs guests on stage-1 tables**, with a hardware cross-check of the walker, atomic descriptor writes and an instruction-abort exit | The skeleton's memory is identity-mapped and read-write-execute. Nothing owned adopting the tables. |
| ruled (lead, under delegation): the shim lives in the upper address half (TTBR1) with its own static tables; in #85 | **Mapping the EL1 shim** | Every page the stage-1 builder writes is EL0-only and never privileged-executable, so the shim can't be mapped through it. |
| ruled (lead, under delegation): the kernel handles the HIPC layer (handles, process id, descriptors), the framework parses CMIF and TIPC; specs amended | **Which side parses messages** | Both specs said the kernel parses, but only the server knows whether a session is a domain. |
| promoted #86 (lead, under delegation) | **Byte-exact libnx message layouts** | The reply builder pads before domain object ids, and parsers need the expected size as libnx does. Milestone 1's program is built with libnx. |
| promoted #88 (lead, under delegation) | **Pin gate tools; list the Arm reference manual; add Atmosphère as a facts-only reference** | A tool release can turn `main` red; six result codes in the service framework are guesses Atmosphère would settle. Atmosphère is GPL-2.0-only, so it is a reference for facts, never for code. |
| ruled (lead, under delegation): the feature is `test-support`, as the owner decided; the workspace allows the lint that rejected the name; services renames its `fakes` feature in #87 | **Name of the per-crate fakes feature** | The skeleton and the service framework chose differently. |
| promoted #87 (lead, under delegation) | **Service slices 3–5 with a snapshot hook** (deferred replies, per-server queue, trace sink, `sm:`, `set:sys`) | The observability spec's snapshot rule needs a hook before the first real services are written. |
| promoted (lead, under delegation): split by what the homebrew fixture needs. Slice 6 is #97 and slice 9 is #98; slices 4, 5, 8 and 10 are filed after milestone 1 | **Kernel slices 4–10** | Unchanged from wave 3. |
| folded into #75 (lead, under delegation) | **Align the zero point of patched counter reads and `GetSystemTick`; per-process counter rate; name the environment's clock type for what it returns (host nanoseconds)** | The skeleton shares a rate between the two clocks but not a starting point. |
| not delivered: #85 closed without it; proposed again under "From waves 8 and 9" | **Unpatched counter reads answered at the host rate by the shim** | A read the patcher misses would disagree with patched ones. |
| promoted #206 (lead, under delegation) | **Crash report: a frame-pointer backtrace; a program counter for timed-out runs** | Reviewer follow-up; useful once real programs run. |
| proposed | **A stop handle for runs (Ctrl-C leaves no artifacts); print guest log lines as they happen; a default timeout** | Reviewer and builder follow-ups on the command-line app. |
| proposed | **Stage-1 tables: a failed write during `map` can leave a table page registered but unlinked; a table descriptor with other ignored bits set is accepted** | Found by #76's fix pass; neither affects correct guests. |
| proposed | **Pinned threads** | No kernel slice owns them (from wave 4). |
| accepted (lead, under delegation) as provisional | **Run limits: the trace keeps a run's first 100,000 events and the guest log its first 4 MiB**, with truncation recorded in the summary | The observability spec asks for a per-run cap without numbers. The newest events are always in the flight recorder. |
| accepted (lead, under delegation) | **`guest_log.txt` in the artifacts folder; usage and file errors exit with code 2; `summary.json` and `crash.json` layouts as the skeleton defines them** | Gaps in the observability spec that the skeleton filled; spec amended. |

### From waves 8 and 9 (ledger: #22)

Wave 8 landed the dispatcher's table borrows and guest time (#75), stage-1 tables on hardware
(#85), `sm:` and `set:sys` (#87) and the libnx fixture (#68). Wave 9 landed the NUP reader (#95).
All vetted by the lead under delegation; open to the owner's review.

| Status | Proposal | Why |
|---|---|---|
| promoted #97, #98 (lead, under delegation) | **The address space, then sessions and services, driven by the reference trace of the homebrew fixture** | The fixture stops at its first system call. The reference emulator's trace of the same file gives the ordered list of calls to implement (`docs/reference-emulator.md`). |
| promoted #100 (lead, under delegation) | **Decompress ZBIC-compressed NSO segments** | A Switch 2 title's main module is compressed this way and `Nso::parse` refuses it, so nothing of a real title can be mapped. Loader only, so it runs beside the kernel work. |
| folded into #97 | **`AddressSpace::pieces` loops forever on a zero-length piece; nextest has no slow-test timeout** | Found by a wave 8 reviewer; a hang in a test would stall the gate. |
| folded into #98 | **A golden-trace test for the homebrew fixture** | Milestone 1's "done when" needs one; the expected trace lived only in a README. |
| adopted (lead, under delegation) | **Each agent builds in its own checkout's build folder** | A shared `CARGO_TARGET_DIR` served a sibling's stale build several times across waves 2 and 8 (decision log). |
| promoted #205 (lead, under delegation) | **The shim answers unpatched `cntpct_el0` reads at the guest's rate** | A read the patcher misses returns the raw host counter while patched reads are rescaled. Needs the interrupt-recovery cases and a hardware test. |
| done: a program's console, from its NPDM, sets its counter rate (`crates/core/src/system.rs`, `clock.rs`) | **Something owns which counter rate a program gets** | The kernel has a per-process rate (#75) but every process is given the Switch 1 rate. Belongs with process creation (slice 8). |
| proposed | **A backend-neutral TLB flush on the `Vcpu` trait** | The flush is a method of the hypervisor backend only, and `run` does not refuse while a flush is owed. The kernel is generic over `Vcpu`, and KVM will share the tables. |
| proposed | **The loader reads libnx's `LNY0`/`LNY1` extension to MOD0** (GOT and RELRO ranges) | libnx protects its own RELRO page at start-up (the fixture's first system call), so nothing on the milestone 1 path needs it yet. |
| proposed | **A second homebrew fixture built with a firmware version set** | The first leaves it at zero, so the TIPC form of `sm:` and `set:sys` command 4 are never exercised. |
| proposed | **Service-framework hardening**: check interface registration at construction; unique service ids in the wiring; command ids from constants; per-service stub names; a build identity in the whole-run snapshot header; trace events for the command name, the protocol and a deferred reply that never completes (with a dump version bump) | Review findings on #87. None is on the milestone 1 path. |
| proposed | **Kernel follow-ups from #75**: an "error with outputs" return form (needed by `WaitSynchronization`, slice 4); where kernel objects sit in the lock order (before slices 6 and 9 share them); `serde` on the object and scheduler tables; a check that a dispatch arm's table set matches its call's signature; lock waits written to `summary.json`; the stubbed list in the summary carrying each call's status | Fix-pass notes. The first two are carried into the slices named. |
| proposed | **The host counter read moves to the CPU backend** | The kernel's clock reads it through `core` today; only the backend knows the host's rate. |
| proposed | **Loader conformance pass against the facts-only reference** (NSO segment size check, NPDM and ACID structure checks, raw capability words) | Review of #68 found checks the reference makes and starquake does not. Off the milestone 1 path. |
| proposed | **Test hygiene**: a per-process scratch folder in `crates/core/tests/artifacts.rs`; macOS CI runs clippy on `starquake-core --all-targets`; `clippy.toml` allows panics in tests; enforce `compile_fail` error codes | Small findings from reviews; hosted CI never compiles the macOS-only code in `core`. |
| folded into #36 | **Narrow #36** to `wfi`/`wfe`, the instruction-cache policy and printing the syndrome in crash reports | #85 delivered the syndrome in exits and the TLB flush entry point. |
| folded into #25 | **Measure** the ASID width (8 bits on this host), whether an unmap is visible to other vCPUs before the flush, and a bound for the number of flush ranges (512 is a guess) | #85's hardware tests and `tools/hvprobe` disagree on how long a stale translation survives; the spec's sentence stays until measured. |
| ruled (lead, under delegation): yes | **Does a new crash-report kind bump `CRASH_VERSION`?** | An older reader would meet a kind it does not know. Any change to a serialised layout bumps its version (`CLAUDE.md`); the constant's doc comment is corrected with the next crash-report change. |
| dropped (lead): no guest needs it either way | **Should `RamConfig` refuse a RAM base below 4 GiB?** | Raised in review of #85; no guest needs it either way yet. |
| done by #173 | **Run a title: map a title's ExeFS into a process, `starquake run <title.nup>`** | Nothing turns an opened title into a running process. Fold with #33 (wiring the parsers into process creation). After milestone 1. |
| ruled by the owner, 2026-11-16: positional reads on a shared source; a compressed file is decompressed whole on first open into a size-capped in-memory cache, evicted least-recently-used (decision log) | **Design: RomFS from a NUP file for the file-system service** (positional reads; seeking inside a compressed file) | The reader borrows its source mutably, and a compressed file is one zstd frame with no random access. Settle before `fsp-srv` builds on it. |
| partly done by #173; the rest stays proposed | **A narrow title API in `core`** in place of re-exporting the loader crate | `core` re-exports the whole loader and the app's tests depend on the loader directly; "frontends only see `core`" covers neither. |
| proposed | **NUP follow-ups**: read the meta block's TLVs (and raise the 0-based against 1-based numbering upstream); parse only the NSO header for `inspect`; cap the total path bytes of a directory walk (names that are prefixes of one pooled string make it quadratic); `verify` says "damaged" for an I/O error; the tree depth limit of 128 is a guess | Review findings on #99. |
| ruled by the owner for M2, 2026-11-16: run the base package only and log that updates are ignored; the rules are settled when a title needs an update (decision log) | **Three rules for combining a title's packages**: an update replaces the whole ExeFS or overlays it; the highest version wins; how a base and its update are related by title id | The reader reports packages and leaves the choice to its caller. The first needs a ruling against the reference before a title is mapped. |

### From waves 10 to 12 (ledger: #22)

Wave 10 landed the process address space (#97), wave 11 ZBIC decompression (#100) and wave 12
sessions, servers and named ports (#98), which completed milestone 1. Recorded by the lead; the
delegation that let the lead promote proposals ended with milestone 1, so everything here is
`proposed` and waits for the owner.

| Status | Proposal | Why |
|---|---|---|
| proposed | **The brief for kernel slice 10 (core loop and timers)** names: wiring `Dispatcher::deliver_replies` and `take_woken` into the loop; making the loop's service set injectable so its waiting arm can be tested; notices that arrive while a handler is out | These exist and are tested in the kernel but nothing in a real run reaches them, and the loop's waiting arm has no test. |
| proposed | **One request queue per HLE server; serialisable IPC tables** | The kernel and the service framework each keep a queue that must agree. `Ipc`, `Request` and `Translated` are not serialisable, which snapshots need. |
| promoted #207 (lead, under delegation) | **Trace: a protocol or control mark on IPC events; fill `service_commands` in `summary.json`** (a trace layout version bump) | The fixture's comparison infers which service a control request belongs to from the next request. A run's summary lists no service commands. Merge with the service-hardening proposal. |
| promoted #209 (lead, under delegation) | **IPC failure paths**: roll back handles that a failed handler put in the sender's table | The error-path rule; nothing does this yet. No program on the current path triggers it. |
| proposed | **Session limits on `ConnectToNamedPort`** | The reference can refuse with a limit error; starquake has no session limits. Listed as a divergence and pinned by a test. |
| proposed, for the kernel spec | **Two notes**: the kernel does not read buffer descriptors when a request is sent; moving a handle into a guest server's table is untracked | Neither is ruled out by the design; both need a sentence in the spec when guest servers are taken up. |
| proposed | **Record address-space changes in the event stream; a serialisable address-space table** | No kernel slice owns the recording point for map, unmap and protect. Snapshots must rebuild the table's mapper. |
| proposed | **ZBIC follow-ups**: a fallback when a rewritten block crosses a zstd limit (hold until a title trips it); a cap on the decoder's up-front window reservation from an untrusted size; a strict verification mode; whether NRO and NRR can carry ZBIC | The reservation is the only one a hostile file can reach today; it is documented and pinned by a test, not prevented. |
| proposed (the owner passed on it for now, 2026-11-12) | **`starquake ls` and `extract` for a title's files** | A first use of the RomFS reading path the file-system service needs. The `nup_tally` example in the loader covers looking around for now. |
| done: milestone 1 delivered these; the remaining items are filed (#33 and the M2 issues) | **After milestone 1: kernel slices 4 (events, waits), 5 (address arbiter, condition variables), 8 (process and threads) and 10 (core loop); CPU #35 and #36; title mapping #33; then services** | The widest set of work that can run side by side before the graphics designs are approved: at most two kernel lanes, one CPU lane, one loader lane, then one lane per service. |

### Folded or already recorded (waves 10 to 12)

- The scope review of wave 10 did not run (the network dropped); its one follow-up is the
  address-space recording row above.
- The wave 9 rows (run a title, RomFS from NUP, a narrow title API in `core`) stand as written.

### From waves 17 and 18 (ledger: #22)

Filed from these waves under the delegation: #177, #178, #179–#194 (graphics-server slices), #199,
#200 and #203. #33 is retargeted to M2. The rest:

| Status | Proposal | Why | Suggested |
|---|---|---|---|
| promoted #208 (lead, under delegation) | **Charge a process's system resource (META 0x14) to its pool**, as Eden's `KProcess::Initialize` does; check `svc` info process handles against `ProcessRecord::object` | `GetInfo` over-reports free memory, and #178's fit check documents the gap | area:kernel, M2 |
| proposed (fold into #161's build hint) | **`FakeKernel::resolve_handle` models handle translation** | So a services test can send libnx's exact `OpenApplicationProxy` with `CURRENT_PROCESS` | area:services, M2 |
| proposed | **Size the stage-1 table pool with guest RAM** (a fixed 4 MiB today) and settle the 16 MiB working-RAM guess | A larger title runs out of table pages before RAM; may need an owner decision on sizing to the whole application pool | area:cpu, M2 |
| proposed | **Export the NSO header offsets** from the loader; a trampolined-program RAM-sizing test once #203 lands | `crates/core/tests/guest_ram.rs` restates the `.bss` offset | area:loader |
| proposed | **The kernel half of `WriteWatch`**: arm every taken request, take and protect under the fault path's lock, range-wide invalidation, a cancel path | #197 documents the two promises the kernel must keep | area:kernel, M3 |
| proposed | **The `RawView` host-access type** in the memory crate | `GpuMemory::view` waits on it; needs `unsafe` and volatile or atomic accessors | area:memory, M3 |
| proposed | **G16/G17 caches re-read a range on first use after arming; G22 takes `GpuMemory::generation` from `Translate::generation`** | Review notes on #197 | area:gpu, M3 |
| proposed (fold into the decoder slice issues 6a–6i) | **Shader decoder follow-ups**: fp16 swizzle and source-2 modifier bits; uniform ALU forms; encodings for JMX, UP2UR and BPT; owners for IMAD.HI, ERRBAR, ATOMS.POPC and BCLEAR; the shared bit-91 list in the framework tests | Review notes on #201 | area:shader, M3 |
| proposed | **`docs/architecture.md`'s walking-skeleton paragraph** names `System::from_nup` and NUP titles | Skipped in #173 as outside its files | docs |

### From waves 19 and 20 (ledger: #22)

Filed from these waves under the delegation: #215, #216, #221, #223–#226. Review notes for
graphics-server slice 7 are on #185, and the owner's hook and unlock rulings are on #183 and
#185.

| Status | Proposal | Why | Suggested |
|---|---|---|---|
| ruled (lead, under delegation) | **`GetInfo` ids 29–33 follow PR #4588's values**, each marked as a guess with its citation and what would confirm it (#223) | The title stops at id 29. The kernel spec says the ids land citing the scoping doc; the only reference that answers them is PR #4588, and a guess marker keeps it honest | area:kernel, M2 |
| awaiting the owner | **Spec note: shared mappings write their stage-1 entries eagerly** | #218 backs a shared mapping's every page at map time; the execution model spec says pages are backed on first touch | execution model |
| awaiting the owner | **Graphics-server verification item 1 gains the 211-overlap, kind, exact-match and root 115 conflicts**, and a ruling on whether a 211 over a live mapping is refused or ordered | Found building GpuVm (#217) and the root (#219); decision 8's house rule applies until then | gsv |
| proposed | **Kernel tests pin `QueryMemory` reporting GeneratedCode as 0x14 and `SetProcessMemoryPermission` requiring FlagCode**; a trampoline and code-memory test once `ControlCodeMemory` lands | Review of #211 | area:kernel |
| proposed | **A fixed-length body for the Syncpoints snapshot** | Review of #212; low priority | area:gpu |
| proposed | **Service framework: a session-close notification** (a closed `IAppletResource` frees its row as Eden's destructor does) **and a ruling on short CMIF payloads** (a 0- or 4-byte `ToCalendarTimeWithMyRule` payload answers success because the reader reads padding) | Review of #213 | area:services |
| later (not M2) | **Cross-process shared memory** (system-wide guest RAM and shared-run records) and **`Break` terminating one process in a multi-process kernel** | #218 and #222 document both limits | area:kernel |

### From wave 21 (ledger: #22)

Filed from this wave under the delegation: #232 (every `GetInfo` id Eden answers; the title stops
at id 22), #233 (the device-share release token, which blocks #185) and #234 (a process end asked
during a timer-thread delivery). Notes for #182, #183, #185, #194 and #208 are comments on those
issues.

| Status | Proposal | Why | Suggested |
|---|---|---|---|
| awaiting the owner | **The Switch 1 caller gate for `GetInfo` ids 29–33** | #223 refuses them to a Switch 1 caller with `InvalidEnumValue`; PR #4588 has no gate. The two exact cases where the answers differ are listed in `crates/kernel/src/svc/info.rs` | area:kernel |
| awaiting the owner | **What "the stack" is for a crash backtrace walk** | #206 walks the user read-write block that holds sp, falling back to the one that holds x29 when sp has left the stack. That is a guess, marked as one | area:core |
| awaiting the owner | **Monotonicity of drained syncpoint values** | #181 documents that drained values can go backwards (push 5, drain, push 4, drain) rather than remember each id's last value, which would compare wrongly after 2^31 increments | gsv |
| awaiting the owner | **What a failed completion drain does**, and **the encoding of multi-field service notes and their coalescing rule** | Not settled by graphics-server.md; needed by #194 | gsv |
| awaiting the owner | **An end asked by a handler that then fails**: #199 honours it (pinned by `an_end_asked_by_a_handler_that_then_fails_is_honoured`) | The services spec says the end runs after the reply; it does not say what happens when the handler then returns an error | services |
| proposed | **A test-only `Processes` constructor whose own row is not first** | `Processes::own` is always the first row, so `GetInfo` and other callers cannot be tested with a non-default caller id, against the house rule on lookups | area:kernel, M2 |
| proposed | **Crash-report extras**: backtraces for every thread, captured at crash time; suspend flags and wait targets; symbols; each thread's last events; a test for `Scheduler::threads()` | Scope review of #206 | area:core, M3 |
| proposed | **Correct the record that PR #4588 is the only reference for ids 29–33** where a second source turns up | Scope review of #223 | docs |

### From wave 22 (ledger: #22)

Filed from this wave under the delegation: #240 (`MapPhysicalMemory`, where the title now stops),
#241 (an `fsp-srv` worker in a real run), #242 (a resource limit for each process) and #243
(ending a process releases its device-share locks and nvmap handles). Notes for #194 and #208 are
comments on those issues.

| Status | Proposal | Why | Suggested |
|---|---|---|---|
| awaiting the owner | **Replaying device-share releases**: a note in the observability spec on how a recorded release is replayed | Scope review of #233 | observability |
| awaiting the owner | **A one-line amendment to `services.md`** for the device-share release path, and **which gsv thread pushes `DeviceShareReleased`** | Scope review of #233; graphics-server.md does not name the thread | services, gsv |
| awaiting the owner | **Whether an nvmap free's unlock waits for GPU ordering** | Scope review of #187 | gsv |
| awaiting the owner | **The NX2 stub-device divergence**, and **the CMIF evidence for the spec's open question 9** | Scope review of #187 | gsv |
| proposed | **Reserve a process-local region in `process::run`** (`GetInfo` 20) and **keep per-thread and idle tick counts in the scheduler** (`GetInfo` 10 and 25) | #232 answers all three as zero, listed as divergences | area:kernel, M2 |
| proposed | **A `Context` API to mark a reply as auto-stubbed** (`IpcOutcome::Stubbed` with a sub-key), so nvsrv's unknown ioctls show up in the "commands still to implement" trace the services spec promises | Scope review of #187; #207 only marks which service an event is for | area:services, M2 |
| proposed | **`starquake inspect` shows the package `newest_package` picks** | Scope review of #200 | area:loader |
| proposed | **`Nup::open` refuses `/` in a node name**; **the cost of `romfs::build`'s sort comparator**; **the `RomFsStorage` cache mutex held through decompression** | Scope review of #200 | area:loader |
| proposed | **A 12 GiB test fixture for the `GetInfo` rig** | #232 builds one ad hoc for a single test | area:kernel |

### From wave 23 (ledger: #22)

Filed from this wave under the delegation:
- #249: appletOE's Switch 2 commands. The title now stops at appletOE command 10.
- #250: address-space snapshot and restore. #240 is closed, and its snapshot item moved here.
- #251: fsp-srv follow-ups.

Notes for #185 (command 216 is still to build) and #194 are comments on those issues.

| Status | Proposal | Why | Suggested |
|---|---|---|---|
| awaiting the owner | **The auto-stub policy for a command whose reply carries an object or handle** | The title's unknown appletOE command 10 was answered success with no object. The title then read through a null pointer, a crash far from its cause. Refusing such commands, or stopping the run with a plain-words error naming the command, would point at the cause. The services spec's stub policy doesn't separate the two | services |
| awaiting the owner | **Graphics-server verification item 1 gains the 211-over-a-live-mapping and 102-with-live-211-maps conflicts**, and **whether `GpuFront` needs an ordered map** | Scope review of #185 | gsv |
| awaiting the owner | **How replay reconciles 201's reached-at-arm push with the `SyncpointObserved` notes fed back** | Scope review of #182 | gsv, observability |
| awaiting the owner | **The result code for an unreadable data storage** (FS 5000 is a guess) | Scope review of #241 | services |
| proposed | **A session-close hook in the service framework, and a bound on gsv's syncpt waiter table**. A closed syncpt session leaves up to 63 waits armed. Extends the wave 19–20 entry "Service framework: a session-close notification" | Scope review of #182; both references clean up in their destructors | area:services, M2 |
| proposed | **Run-level `--record` and `--replay` for reads** | Scope review of #241. Check #226 for overlap | area:core |
| proposed | **`MapPhysicalMemoryUnsafe` and `UnmapPhysicalMemoryUnsafe`** (0x48, 0x49), which Eden lacks too | Scope review of #240 | area:kernel |
| proposed | **`FlushDataCache` (0x2B)**: the homebrew gap trace in `crates/core/tests/homebrew.rs` now stops there | Scope review of #240 | area:kernel |
| proposed | **Reads take turns under the `LiveEnv` lock** | Scope review of #241; a throughput note | area:core |

### From wave 24 (ledger: #22)

Filed from this wave under the delegation:
- #257: `MapMemory` and `UnmapMemory`. The title now gets past appletOE command 10 and the application proxy's commands, then stops: `MapMemory` isn't built, so it fails, and the title calls `svcBreak`.
- #258: appletOE application proxy commands 900 and 130 (auto-stubbed today), `IApplicationFunctions` 330, and the inputs of 320.
- #259: destroy a gsv address space when its last as-gpu fd closes. Without it the 8 slots run out.
- #260: snapshot and restore of the object table and handle tables.
- #261: snapshot of touched guest pages, and the address space's `Map` recording point.

#188 stays open for as-gpu `ModifyKind` and `Remap`, which wait for the first question below.

| Status | Proposal | Why | Suggested |
|---|---|---|---|
| ruled 2026-11-17 (decision log) | **Build or drop as-gpu `ModifyKind` (0xC) and `Remap` (0x14), and ctrl-gpu `QueryEvent` 1 and 2** | They come only from PR #4588, and row 30 shows Switch 2 never mapping through as-gpu. Building them needs an ordered kind change with no notification event, and as-gpu `AllocateSpace` allocations in `GpuMemory`. Today they fall to the unknown-ioctl policy, and `QueryEvent` 1 and 2 answer `BadParameter`, although the nvsrv docs promised them | gsv |
| ruled 2026-11-17 (decision log) | **Whether address-space restore should match live operations exactly**, using a table of the permissions each state allows | Scope review of #250. Restore refuses unknown bits and non-SVC permissions. It doesn't check per-state permission and attribute relations. For example, `Normal` or `CodeData` with RX is refused only by `SetMemoryPermission`, so a snapshot that has it restores as saved. Unknown `DisableMerge` bits also restore as saved | kernel |
| ruled 2026-11-17 (decision log) | **as-gpu's shim when a view's start equals its split** | Review of #188. starquake keeps PR's behaviour: the shim lands below the view's start, or at GPU address 0 for a one-page start, and the call succeeds. The module docs list the question | gsv |
| ruled 2026-11-17 (decision log) | **Which ctrl-gpu characteristics levels the SDK reads** | Review of #188. Three sources conflict: row 24 says every level at 0x70..0x80 is at least 6; PR has -1 at +0x74 and zeros from +0x7A; PR's channel level is 0x1D, but NX2's feature threshold is 0x1f. The module docs list all three | gsv |
| proposed | **Move `Dispatcher::transfer_memory` next to `IpcContext::transfer_memory` in `dispatch/ipc.rs`** | #183 put it in `dispatch/hooks.rs` to stay inside its own files | area:kernel |

## House-rule candidates

Rules distilled from review findings. A vetted rule moves into `CLAUDE.md`; until then it is
passed to build agents as a provisional rule for the next wave.

### From waves 1 and 2 (vetted by the owner, 2026-11-11)

| Status | Rule |
|---|---|
| adopted | Every proptest block sets an explicit fixed seed. |
| adopted | Expectations come from the shipping constants, with boundary cases exactly at the limit. (Importing constants was already a rule.) |
| adopted | A parser that reads ordered fields gets a round-trip test that fails if two reads are swapped. |
| adopted | Unsourced layouts are marked as guesses; claims about reference behaviour cite the exact check; divergences are listed and pinned by a test; no blanket claims. |
| adopted | Arithmetic on untrusted sizes is checked for the whole extent; loops over untrusted structures are bounded by the structure walked. |
| adopted | No retries to hide a flaky test; negative tests assert exact errors; a test for a fix is run against the unfixed code first; PR claims are verified. (Watching a new test fail was already a rule.) |
| adopted | Don't duplicate logic across sibling modules. |
| adopted | The merge queue waits for a PR's own CI checks before merging. |
| dropped: a fact about the CPU backend, not a house rule; folded into #36 | Host writes into guest-executable memory do instruction-cache maintenance. |
| dropped: a one-off fix, already applied | Widen CI path filters to cover every input that can break the job. |

### From wave 3

| Status | Rule |
|---|---|
| adopted (lead, under delegation) | When two crates share a unit or convention (ticks or nanoseconds, who converts), give the value a distinct newtype or name the conversion's owner in one place, so neither side can assume the other converts. |
| adopted (lead, under delegation) | Decode and validate a header's magic and version before decoding the body, so a layout change is reported as an unsupported version. |
| adopted (lead, under delegation) | Pin every serialised or on-disk layout with a golden-bytes test covering every variant with distinct non-zero values; a layout change requires a version bump. |
| dropped (lead, under delegation): covered by the no-duplicated-logic rule | Don't wrap a type in a second type that repeats its storage and logic; compose by wrapping the existing one. |
| adopted (lead, under delegation) as a merge-queue step | Before merging, test-merge current `main`, build, and re-run the full gate; re-verify numbers quoted in the PR body. |

### From wave 4

| Status | Rule |
|---|---|
| adopted (lead, under delegation) | Never derive deserialisation on a validated newtype over a primitive; deserialise through the validating constructor, and test that an out-of-range value is an error. |
| dropped (lead, under delegation): covered by the cite-the-exact-check rule | When a doc or PR states an error for every argument, probe each argument against the reference, including the reference's order of checks, and pin each case with a test. |
| adopted (lead, under delegation) | A test invariant mirrors the reference's exact condition; no broad escape clause that switches the assertion off in common generated cases. |
| dropped (lead, under delegation): too specific | Property tests that drive a clock keep time monotonic, including in their final step. |

### From wave 5

| Status | Rule |
|---|---|
| adopted (lead, under delegation) | Every boundary check gets a test at the exact edge on both sides. |
| adopted (lead, under delegation) | A docs or PR claim that something is "pinned by a test" names the test. |
| adopted (lead, under delegation) | Decisions in command-line or gate code are pure functions with every input combination tested. |
| adopted (lead, under delegation) | Tests that touch the file system use a per-test unique folder and clean up; new tests are run many times under both plain `cargo test` and nextest. |
| adopted (lead, under delegation) | When an operation can fail after it has begun changing shared state, the required side effect (flush, cleanup, rollback) happens on the error path too, pinned by a test double that fails mid-operation. |
| adopted (lead, under delegation) | A freed resource is not reusable until every pending follow-up on it has run. |
| adopted (lead, under delegation) | An id or pointer read from guest-writable or possibly corrupt data is validated against what was actually allocated, not just a range. |
| dropped (lead, under delegation): covered by "report spec conflicts, don't resolve them" | When a spec prescribes a mechanism, follow it or report the conflict. |

### From wave 8

| Status | Rule |
|---|---|
| adopted (lead, under delegation) | Whatever saves state passes the same validity checks that restoring enforces, and restoring validates every relation between fields; each check is tested at its exact edge with the exact error. |
| adopted (lead, under delegation) | Name a decoded bit position, shift or width once, as a constant shared by the decoder and the encoder; decode tests give each field a distinct value. |
| adopted (lead, under delegation) | Mutation-check new tests: for each new branch, break it in a scratch copy and confirm a test fails. Never claim a test pins a property it cannot observe. |
| dropped (lead, under delegation): covered by the cite-the-exact-check and mark-guesses rules | Trace the reference's actual code path (discarded results and soft asserts included) before stating what it does; state the exact condition under which a doc claim holds; keep the divergence list complete. |
| dropped (lead, under delegation): covered by the error-path rule | Every early return releases what it holds, with a test that the resource is reusable afterwards. |
| dropped (lead, under delegation): already a rule | Types with invariants deserialise through a validating constructor. |

### From wave 9

| Status | Rule |
|---|---|
| adopted (lead, under delegation) | When docs promise an order of checks, a test has both faults present at once. |
| adopted (lead, under delegation) | A failed streaming read is sticky: every later read returns the same error, never a clean end of file. |
| adopted (lead, under delegation) | Bound memory and time by what a hostile input can force, not only by loop counts. |
| adopted (lead, under delegation) | A parser gets one hand-built accepting case per format feature, not only inputs made by the project's own builder or encoder. |
| adopted (lead, under delegation) | User-facing errors and command output are plain words, not `Debug` forms, asserted as exact strings. |
| dropped (lead, under delegation): covered by the both-sides-of-the-edge and name-the-test rules | Edge tests use the real alignment step; each "looser than the reference" item names its test. |

### From waves 10 to 12

| Status | Rule |
|---|---|
| adopted (lead, under delegation, before milestone 1 landed) | Test every range operation (map, protect, unmap, query) with a neighbour that starts exactly at the end bound, and assert the neighbour is unchanged. |
| adopted (lead, under delegation, before milestone 1 landed) | Every state-transition branch gets a test of both the changed and the unchanged outcome. |
| adopted (lead, under delegation, before milestone 1 landed) | When a layer relies on a permission or an error the reference documents, check that the production backend honours it end to end, or enforce it in the layer. |
| proposed | A format limit that is a reference fact gets one golden test that states the literal with its citation. Every other test imports the constant. (Needs the owner: it qualifies the rule that tests never restate literals.) |
| proposed | Bound every allocation a wrapped third-party decoder makes from an untrusted header, and test it with the smallest hostile input that states the largest size. |
| proposed | A bound a test claims to pin must be observable: assert a measured peak or a pure helper, not only the final output. |
| proposed | A feature in an issue's scope with end-to-end behaviour (a command's output) gets an app-level test with the exact expected line. |
| proposed | Every commit that reaches `main` passed the gate: squash work-in-progress commits before opening the PR. |
| proposed | Never discard with `let _ =` a result that reports a freed resource; route it through the one teardown path, on error paths too. |
| proposed | When a shared resource is taken out and its lock released, the admission logic must know it is out; pin it with a threaded test that sends a second request meanwhile. |
| dropped: covered by existing rules | Mutation-check new bounds; back every stated invariant with an assertion; test documented orderings with two competing items; grep the code before stating a docs claim. |

### From wave 21 (ledger: #22)

| Status | Candidate |
|---|---|
| proposed | Give every test double the real implementation's contract, with a parity test that runs one scenario on both; list and pin each difference. |
| proposed | Pin every documented concurrency property (lock order, callbacks with no lock held, latched wakes) with a bounded-timeout threaded test, mutation-checked. |
| proposed | A queue or map that coalesces or reorders states its exact ordering and monotonicity guarantee, checked by a model test that interleaves entry kinds. |
| proposed | A type with a cross-field invariant has no public `Default` or `DerefMut` that bypasses its constructor, and a small enum where a free-form field would do. |

### From wave 22 (ledger: #22)

| Status | Candidate |
|---|---|
| proposed | A doc claim that a branch is unreachable names the configuration that does reach it, and a test runs on that configuration. |
| proposed | A restore that consumes a counted resource (locks, handles, tokens) counts per key against the live count; N is accepted and N+1 refused at the extra entry's index. |
| proposed | Don't call a constructor "test-only" or a value "unforgeable" unless the compiler enforces it. |
| proposed | A documented log level or other observable side effect is pinned by a test that captures it, or the claim goes. |
| proposed | When two algorithms would both satisfy a sort or order rule on the golden data, add a case where they differ. |

### From wave 23 (ledger: #22)

| Status | Candidate |
|---|---|
| proposed | Every documented check order, cap and restore validation has its own failing test, and both faults present at once for ordering. |
| proposed | Concurrent workers each get their own environment or handle, or the docs say they serialize; test the production arrangement, not only a double. |
| proposed | Per-session growth is bounded over a run's lifetime: closing a session cleans up what it armed or opened, with a test at the cap. |
| proposed | A deferred change shows its pending state to every observer until it is applied; test the in-between state. |
| proposed | Sibling slices that extend one shared struct, trait or drain add beside each other and never change each other's signatures. |

### From wave 24 (ledger: #22)

| Status | Candidate |
|---|---|
| proposed | Every multi-byte read of guest input is tested with values that differ only above the low byte, and with payloads one byte short at each width; a mutated read width fails a test. |
| proposed | Every comparison is tested at equal, one below and one above, with a duplicate as well as a swapped pair; swapping the operator fails a test. |
| proposed | Restore validation lists every field, id and cross-field relation that live operations enforce. Each one gets its own exact-error refusal, or is listed under "Not checked here". |
| proposed | Every fallible step runs before the first write to guest memory or shared state. The failure test asserts nothing was touched and that the next attempt succeeds. |
| proposed | A record, note or event the spec defines is emitted on every path that triggers it, including the case with no values, and a test expects it there. |
| proposed | Every id read from a snapshot is checked against what the kernel actually made, including its kind and size, or listed as "not checked here". |
