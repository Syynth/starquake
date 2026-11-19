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
| ruled 2026-11-17 (decision log: keep it, make failures diagnosable under #207) | **The auto-stub policy for a command whose reply carries an object or handle** | The title's unknown appletOE command 10 was answered success with no object. The title then read through a null pointer, a crash far from its cause. Refusing such commands, or stopping the run with a plain-words error naming the command, would point at the cause. The services spec's stub policy doesn't separate the two | services |
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

### From wave 25 (ledger: #22)

Filed from this wave under the delegation:
- #268: `ICommonStateGetter` `ReceiveMessage` and the applet's message queue. The title now gets past `MapMemory` and starts its second thread. It then calls `ReceiveMessage` over and over, because the command is auto-stubbed with success and its event stays signalled, until the timeout.
- #269: snapshot and restore of `MapMemory` aliases. Today a process with an alias can't be saved, and libnx maps thread stacks this way. The issue also covers how aliases and device-share locks interact.
- #270: one service-framework path for `Env` failures, applied to `time:u` and every service.
- #271: address-space restore checks attributes against state flags (the owner's ruling of 2026-11-17).

Notes for #189 (a bound channel holds its address space) and #207 (name the interface of a stubbed command in the run summary) are comments on those issues.

| Status | Proposal | Why | Suggested |
|---|---|---|---|
| awaiting the owner | **A `GpuFront` call that forgets an address space**, a spec amendment (for example `destroy_address_space`, ordered after the removals) | Scope review of #259. gsv now destroys a space on its last close, but the spec's GPU front has no call that forgets one. It keeps an empty table per destroyed id, and ids are never reused, so a looping title grows it without bound | graphics-server spec |
| awaiting the owner | **Two `MapMemory` checks where starquake follows Horizon, not Eden** | Review of #257. An unaligned source answers `InvalidAddress` where Eden answers `InvalidSize`. The stack-region check (`InvalidMemoryRegion`) is one Eden lacks. Both are pinned by named tests | kernel |
| awaiting the owner | **Whether reading the RomFS tables on the core thread fits services.md's "short host metadata lookups" exception** | Review of #251. The first `OpenDataFileSystemByCurrentProcess` reads the whole table set with the live environment's lock held | services |
| awaiting the owner | **Whether emulator-bug host answers stop the run**: `HostFailure::BadRequest`, unnamed host failure codes and malformed size answers, which answer `HOST_IO_FAILED` today | Review of #251. Extends the wave 23 question on the unreadable-data-storage result code | services |
| proposed | **Run rustdoc with warnings denied in the gate**, and fix the warnings already there (redundant or unresolved links in the services framework and fsp) | Scope review of #251. The gate doesn't run rustdoc today | area:xtask |
| proposed | **Check the restored message event's id against the kernel without changing it** | Review of #258. `GetEventHandle` copies a restored message-event id unchecked: the kernel's only event checks (clear and signal) change what the guest sees. This needs a restore step that can see the kernel | area:services |
| proposed | **`IApplicationFunctions::InitializeGamePlayRecording` (66)** | Review of #258. libnx sends it before setting the recording state; no title has needed it yet | area:services |
| proposed | **Back a `MapMemory` alias lazily** | Review of #257. core's `map_alias` backs the whole source up front; the execution model's backing pools should make it lazy | area:core |
| proposed | **Share one copy of the RomFS tables between fsp-srv's environments** (behind an `Arc`) | Review of #251. Each of the three environments keeps its own copy | area:services |

### From wave 26 (ledger: #22)

Filed from this wave under the delegation:
- #277: parental control. The title now gets past `ReceiveMessage`. It then asks for `pctl:a`, `pctl:s`, `pctl:r` and `pctl`, gets auto-stubs, and reads through the null object its `CreateServiceWithoutInitialize` "returned".
- #278: `lm`, `ectx:aw` and `aoc:u`, which the title also asks for and nothing provides.

#189's PR (#275) waits for its waiting-submission cap. The owner ruled on that cap and on its event ids on 2026-11-17 (decision log). #207 gains the auto-stub diagnostics the owner asked for the same day (a comment on #207).

| Status | Proposal | Why | Suggested |
|---|---|---|---|
| awaiting the owner | **Whether a services family-state change bumps the run snapshot version**, and **a dedicated `SnapshotError` variant for family validation** | Scope review of wave 26 | observability |
| awaiting the owner | **Whether `GpuFront::submit` hands the submission back on `Full`**, and **what happens to a deferred `SubmitGPFIFO` whose session closes** | Review of #189 | gpu-protocol, graphics-server |
| awaiting the owner | **Recording `Submitted::Full`**: which submissions wait depends on host timing, which isn't recorded, so replay may differ | Review of #189 | observability, graphics-server |
| proposed | **A framework call that closes a kernel event, and a serialisable `DeferredBuffer`** | Review of #189. Each closed channel fd leaves up to three events, and gpu 497 has the same leak. A restored deferred reply can't write its out buffer (nvhost-gpu, fsp-srv `Read`) | area:services |
| proposed | **Generalise `Stop::Job` and `Stop::Replay`** so the framework stops importing fsp's `ReplayError`, and share one test `Unexpected` halt | Review of #270 | area:services |
| proposed | **Extend the host-access gate and the `Env` scan** to `starquake_trace`'s `HostClock`, `SystemClock` and `HostIoHandler`, and to `crates/gsv` and `crates/core` | Review of #270: a scratch file passed both checks | area:xtask |
| proposed | **`unlock_for_transfer_memory` gets Eden's page-group or permission check**, and #260 adds the alias-source versus transfer-memory cross-check | Review of #269. Called directly, it unlocks an alias source and leaves a stale alias record | area:kernel |
| proposed | **am: the operation and performance-mode change message, `PushUnorderedMessage` sources, and a bound on the unordered and restored queues** | Review of #268 | area:services |
| proposed | **A deterministic per-call cost test** (a counting allocator in its own test binary) in place of #268's wall-clock one | Review of #268; the wall-clock test has about 8x headroom | area:services |
| proposed | **Wire the title's NACP through to `appletOE`** so `IApplicationFunctions::GetDesiredLanguage` reads its supported languages, as Eden does, and make the system language a setting | #282: without the NACP the command answers Eden's no-metadata answer, `en-US`, even for a title that does not support American English (`crates/services/src/am.rs`, "The title's later start-up commands") | area:services, area:loader |

### From wave 27 (ledger: #22)

Landed: #277 (parental control, PR #280), #189 (nvhost-gpu channels with the owner's cap, PR #275) and #207 (the auto-stub diagnostics, PR #281). #278 (`lm`, `ectx:aw`, `aoc:u`) was built with a green gate, but its builder returned before opening the PR. Wave 28 finishes it from its branch.

With pctl in place the title no longer crashes: it runs to the timeout. It then polls `WaitSynchronization` without end on a handle the kernel calls invalid, right after the auto-stubbed `IApplicationFunctions` command 210, which Eden answers with an event. #282, filed under the delegation, covers that command and the five other start-up commands the new `stubbed_commands` list names.

| Status | Proposal | Why | Suggested |
|---|---|---|---|
| awaiting the owner | **Whether stub records tell a guest-named stub apart from a real interface.** A stub asked for under sm's own interface name ("Sm") is counted with sm today. The collision is listed as a divergence and pinned by `sm::tests::a_stub_asked_for_by_sms_own_interface_name_is_counted_with_it` | Review of #207 (a comment on PR #281) | observability |
| proposed | **Wire the title's NACP through to services**, so `GetDesiredLanguage` can answer from the supported languages, as Eden does | #282's scope | area:services, area:core |

### From wave 28 (ledger: #22)

Landed: #278 (`lm`, `ectx:aw`, `aoc:u`, PR #285), #282 (the six start-up commands, PR #286), #271 (address-space restore checks attributes against state flags, PR #284) and #209 (IPC reply rollback, PR #287).

The title no longer spins: command 210's event now waits properly. It then reaches the graphics server and asks gsv root command 1402 for the nvsrv session. The root still hands out a placeholder session with no commands, so nvsrv's command 13 is auto-stubbed, and the title records a terminate result and calls `Break`. Wiring the real servers is #194 (graphics-server slice 16), which wave 29 builds with one agent.

| Status | Proposal | Why | Suggested |
|---|---|---|---|
| awaiting the owner | **lm's lines have no `GuestLog` event**, which departs from the observability spec. Options: give lm lines `GuestLog` events, send them to a file of their own, or amend the spec and `docs/architecture.md` | Review of #278 (PR #285) | observability |
| proposed | **aoc's purchased event is one per server**, where Eden makes and closes one per manager; the departure is listed in aoc's module docs | Review of #278 | area:services |

### From wave 29 (ledger: #22)

Landed: #194 (gsv's servers wired into core, PR #291), #260 (the kernel's object and handle tables in a snapshot, PR #290) and #205 (counter reads the patcher missed, PR #289).

The title now talks to the real nvsrv, syncpoint and gpu sessions. It then opens the display context and asks root command 305 for the vi2 session, which is still a placeholder (slice 13, #191). Its commands 3, 1 and 11 are auto-stubbed, and the title polls a handle that does not exist. Wave 30 builds the vsync clock (#190) that the vi2 session needs first, alongside #186 and #242. #191 follows.

| Status | Proposal | Why | Suggested |
|---|---|---|---|
| proposed | **Watch #194's timing test on loaded runners.** It compares a wall-clock wait with half a preemption interval (5 ms). If it ever flakes, raise the threshold; never retry | Review of #194 (PR #291) | area:core |
| proposed | **`cntvct_el0` and `cntfrq_el0` at the guest's rate.** #205 covers `cntpct_el0`; the virtual counter still reads at the host rate and `cntfrq_el0` is not yet the guest's | #205's PR body (two "done when" items not fully met) | area:cpu |

### From wave 30 (ledger: #22)

Landed: #190 (the vsync clock, PR #295), #186 (native memory and image layouts, PR #294) and #242 (a resource limit for each process, PR #293).

The title still stops at root 305's placeholder vi2 session, as expected: wave 31 builds it (#191), with #193 (video sessions) and #185's last command (216).

| Status | Proposal | Why | Suggested |
|---|---|---|---|
| awaiting the owner | **Whether the vsync clock keeps the spec's no-banking rule.** Both references' vsync events catch up on missed ticks (NX2's `m_vsync_timer` is a looping CoreTiming event; PR's `CoreTiming::Advance` is the same). Not banking lost time comes from NX2's present pacing (`PaceToVblank`, called from 229), which the spec applies to the clock. The code follows the approved spec | Review of #190 (PR #295) | graphics-server |
| proposed | **Restore's fed-back tick bound (`MAX_FED_BACK = 1`)** assumes a live turn delivers at most one tick; revisit if the clock ever banks ticks | #190's fix | area:gsv |

### From wave 31 (ledger: #22)

Landed: #191 (the vi2 session, PR #299), #193 (video sessions without decoding, PR #298) and #185's last command, 216 (PR #297).

The title now gets past root 305 into nvsrv. It stops on eight ctrl-gpu and nvhost-gpu ioctls nvsrv does not know, plus ctrl-gpu `QueryEvent` 2, which the owner deferred on 2026-11-17 until a title sent it. That work is filed as #300 (in wave 32). No frame is drawn yet.

| Status | Proposal | Why | Suggested |
|---|---|---|---|
| proposed | **A services-framework session-close hook that runs with a `Context`**, keeping the lock order (gsv's lock, then the kernel's), used by vi2, syncpt, video and nvsrv. Merges the entries from waves 19 and 20 and the #182 and #189 reviews. Closed vi2, syncpt and video sessions keep their events, syncpoints and listening group events; vsync's listener cap (64) can be used up by closed vi2 sessions | Scope review of wave 31; #191 and #193 | area:services |
| proposed | **Root 434 address-space registration, and GPU mapping of memory bound by 114.** 426 answers on the GPU VA == CPU VA rule and accepts any address space; 912 releases memory a decoder still has images registered in, which is safe only with the null decoder | #193's build | area:gsv |
| proposed | **Cap the total reveal ranges held across waiting jobs** (Busy past it), with the matching `gpu::saved` restore check and edge tests. #185's second note, not done | #185's build | area:gsv |
| proposed | **A layout version on `SavedRootState`**, whose golden bytes changed when the video server field was added | Scope review of wave 31 | area:gsv |
| proposed | **Rustdoc with warnings denied in the gate** (extends the earlier proposal): `cargo doc -p starquake-gsv` fails on public docs linking private items | #185's build | infra |
| proposed | **Move closed #190, #191 and #193 from M3 to M2** (waves 29 and 30 proposed the same) | Scope review | owner |
| awaiting the owner | **vi2 `NOTIFICATION_LIMIT` and the 1/3 commands.** The limit comes from NX2 while the bitmap bounds follow PR. Either answer 0, or widen 201, the client masks and the saved bitmap check to `(NOTIFICATION_LIMIT + 1) * 32`. Documented in `crates/gsv/src/vi2.rs` | Review of #191 (PR #299) | graphics-server |
| awaiting the owner | **Three new reference conflicts for verification item 1**, settled by decision 8: vi2 1/3 (PR 0, NX2 1), the 11/13 word (PR requires 0), the 301 flags; and 352's result, set to `VI_OUT_OF_RANGE` | #191 | graphics-server |
| awaiting the owner | **The vsync fallback's "after the first frame" grace.** The spec has it; the code ticks from the first tick, since the clock gives vi2 no per-tick turn | #191 | graphics-server |
| awaiting the owner | **Video layout codes that depart from decision 8.** Code 5 keeps PR's block-linear RGBA layout, so 961 refuses where NX2 answers a BGRA descriptor; codes neither reference names are refused where NX2 sizes them as RGBA; code 1's size follows PR (640x360 gives 0x65400, NX2's observed row gives 0x64000); 109 replies with PR's consumed size | Review of #193 (PR #298) | graphics-server |
| awaiting the owner | **216's completion and limit.** A 216 that takes something out completes after the GPU's barrier (PR completes at once), and 216 answers Busy at the job limit (PR has no such check) | #185 | graphics-server |
| proposed | **A guess to confirm with a trace:** a part a 216 batch takes out and maps again in a later entry is a hole until the drain | #185 | area:gsv |

### From wave 32 (ledger: #22)

Landed: #300 (ctrl-gpu and nvhost-gpu ioctls and ctrl-gpu events, PR #304), #192 (present and pacing, PR #305), #221 (the run's date from the host, recorded, PR #302) and #208 (a process's system resource charged to its pool, PR #303).

The title gets one ioctl further: nvsrv now logs only nvhost-gpu `ZCullBind` (`0xc010480b`), then the run waits until the time limit with no frame drawn. That is filed as #306 (in wave 33), which also covers the ioctls after it that have a reference handler.

| Status | Proposal | Why | Suggested |
|---|---|---|---|
| proposed | **A services-framework call that frees a kernel event**, alongside the session-close hook proposed in wave 31. ctrl-gpu's and nvhost-gpu's events are never freed, so open and close cycles grow the kernel's object count; a closed vi2 session without 202 leaves its clients in the present table (bounded by `MAX_PRESENT_CLIENTS = 128`, a guess). Fold #243's process-end release in as a comment | #300, #192 | area:services |
| proposed | **A run setting or command-line option that selects `Pacing::FenceWait`**; core always uses the buffer queue today | #192 | area:core |
| proposed | **End-to-end tests that run in CI:** booting a program with a system resource and checking GetInfo 7 and 21; a whole-run replay whose recording lacks the date reading (core has no whole-run replay yet); persisting the `Env` recording | #208, #221 | area:core |
| proposed | **Small hardening:** `EnvError`'s Display text names Rust variants ("the replay asked for WallClock"), not plain words; `GetProcessId` should share `GetInfo`'s `ProcessRecord::object` check; only the application pool's size is modelled; GetInfo 17 is still zero (no tracking of the secure resource's page use) | #221, #208 | area:kernel |
| proposed | **For the GPU front (#12):** a destroyed image's memory can be released while its frame is queued or shown (PR keeps the backing until the buffer retires); nothing reads `SetErrorNotifier` or `SetPreemptionModes` state yet | #192, #300 | comment on #12 |
| proposed | **Watch `machine::tests::graphics::a_held_page_request_is_answered_at_the_bound_by_the_timer_thread`**: it failed once while two nextest runs shared the machine, then passed 15 of 15 alone | #192's fix | area:core |
| awaiting the owner | **The graphics-server spec paragraph after the nvsrv device table is out of date.** It still lists ctrl-gpu `QueryEvent` 1 and 2 as deferred, and says nvhost-gpu's events are made at open, where the code makes them at first query. #304 made ctrl-gpu's at open, as the spec says | #300 | graphics-server |
| awaiting the owner | **`SetPreemptionModes` (`0xc008481f`)**: #304 ports PR #4588's refusals; NX2 sends it to its catch-all (echo and Success), which decision 8's "the larger" could be read as picking | #300 | graphics-server |
| awaiting the owner | **Whether the vsync clock runs while a present waits.** #305 keeps it running, which changes slice 12's stop rule and its `DeadlineWithoutListeners` restore check; the spec does not say | #192 | graphics-server |
| awaiting the owner | **The bound on a held 229 reply** under the fence-wait fallback | #192 | graphics-server |
| awaiting the owner | **`--date <seconds>`**: #302 adds it because #221 asks the command line to pin a date; the observability spec's headless contract has no date flag. Its name, format (whole seconds since the epoch) and range are the owner's call. Also, the one reading at boot is taken as the RTC at uptime zero, where Eden subtracts the uptime at the moment it reads | #221 | observability |
| awaiting the owner | **Horizon's "64-bit" loader check** is the bitwise test `flags & AddressSpace64Bit39`, which also passes 36-bit and 32-bit-no-alias spaces; #303 follows the code, not the comment. Also, Horizon takes application/applet from ACI0's program type while Eden's GetInfo 23 uses the ACID pool, and the two can disagree | #208 | area:kernel |

### From wave 33 (ledger: #22)

Landed: #306 (nvhost-gpu ZCullBind, PR #308), #224 (shared-memory lifetime, PR #309), #225 (checked ids in restored family state, PR #310) and #226 (entropy and x18 drawn at creation through `Env`, PR #311).

**A review gap.** The disk dropped below the test-breaking reviewer's 20 GiB floor, so the test-breaking reviews of #310 and #311 did not run. The fix agents skipped those "review not run" findings as naming no file, and both PRs merged with one review. The lead freed about 28 GiB of leftover review clones and ran both reviews after the merge; their results are on the PRs. From wave 34 the pump treats "review not run" as blocking, and reviewers delete their own clones.

**The title.** ZCullBind was not the blocker: the title sends the same requests whether it answers or not. nvsrv now logs no unknown ioctl, and the run ends polling one handle with a zero timeout until the time limit, with no frame drawn. A bring-up investigation (one agent) is finding which object that is and what should signal it.

| Status | Proposal | Why | Suggested |
|---|---|---|---|
| proposed | **The session-close hook and free-event call** (waves 31 and 32) should also check restored session-state kernel ids: vi2's event and notification event, syncpt's events, video's event. Fold #243 in as a comment | #225's scope | area:services |
| proposed | **Process exit closes its shared-mapping references** (Eden's `KProcess::Finalize` does; teardown does not unmap them today), plus an end-to-end give-back test through `Dispatcher`, `Locked::drop`, `AddressSpace::release_shared` and `SpaceMapper` | #224 | area:kernel |
| proposed | **A service's `SharedPages` are not invalidated when pages are given back**; safe only while no server lets go of a shared memory it made. Needs a generation check before any such API | #224 | area:kernel |
| proposed | **Restore-check hardening:** aoc's two event ids restore unchecked; gsv's `RestoreKernel` overlaps `framework::RestoredKernel`; time:u settings through `with_checked_family`; a version for nvsrv `Channel` snapshot state | #225, #306 | area:services |
| proposed | **Persist and replay the `Env` recording:** `--record`, `--seed`, `--replay`, `environment_dropped` in summary.json (a `SUMMARY_VERSION` bump), end-of-run checks for unconsumed records; fsp-srv workers' `Env` under replay; per-process pointer-authentication keys through `Env` | #226 | area:core |
| proposed | **Draw order versus Eden:** x18 is drawn before the thread reservation and `CreateThread`'s argument checks, so a refused call consumes a random number (not guest-visible) | #226 | area:kernel |
| awaiting the owner | **Who consumes `Channel::zcull`?** #308 stores the bind; the spec defers zcull and nothing reads it | #306 | graphics-server |
| awaiting the owner | **Should the kernel note which server made a shared memory?** Without it, hid's and time:u's restore checks cover kind, size and permission only | #225 | area:kernel |
| awaiting the owner | **Unconsumed replay records: reported, or an error?** And is the `RunOptions`/`RunReport` replay API shape, with its 16 MiB recording cap, what the observability spec should say? | #226 | observability |

### From wave 34 (ledger: #22)

Landed: #234 (a process end asked during a timer-thread delivery, PR #315), #243 (a process end releases its device-share locks and nvmap handles, PR #313) and #261 (touched-page snapshots and the address space's Map recording point, PR #314). Every PR had both reviews; all findings were applied.

**The title, and milestone 2.** The title now submits GPFIFO work through nvhost-gpu (`SubmitGpfifo`, `0xc018481b`), which is milestone 2's exit ("as far as its first GPU submission"). It then spins: thread 1 waits for a GPU semaphore release (Switch 2 scoping, row 35), and the handle it polls with a zero timeout is nvhost-gpu's error-notifier event, which neither reference ever signals. `NullFront` drops what a submission carries and performs only syncpoint increments, so the release is never written. The fix is the GPU front: gpu-protocol slices G8 (channel pusher), G14 (front loop), G15 (back fences and release writes), G19 (back executor) and G21 (`Gpu` as `GpuFront`), none filed yet. Not yet confirmed: whether the release is a host-method or an engine (QMD) release, and which page is polled; the two references disagree.

| Status | Proposal | Why | Suggested |
|---|---|---|---|
| proposed | **Find the cause of the timer-thread test's flakiness**, now that `a_held_page_request_is_answered_at_the_bound_by_the_timer_thread` has failed a second gate run (5.18 ms against its 5 ms bound) and 1 run in 30 alone. Replaces wave 32's "watch" row | #311's review | area:core |
| proposed | **A scripted-guest check that fails should fail the test**; today it hangs until nextest's timeout | #311's review | area:core |
| proposed | **Tests around #243's teardown:** the widened completion-queue `Wake` contract (a pusher may hold the kernel's lock) with the production `Bell`, the real `Dispatcher` and the timer thread; a real-kernel nvsrv rig where the process ends through its last thread's `ExitThread`; teardown's device-share step gated by a "process ended" pending flag, like `has_unlocks` and `has_frees` | #243 | area:gsv |
| proposed | **nvmap locks across a snapshot:** a restored table holds no tokens, so a later `IocFree` releases nothing and the lock stays until the process ends. Save a "locked" flag and reclaim the tokens on restore, which needs one per-range lock count shared by gsv's lock holders (the gap noted in `gpu/saved.rs`) | #243 | area:gsv |
| proposed | **A framework hook so a deferred reply can end the process**: `framework/server.rs` `deliver` builds no `Context`. Needed only when a title defers `ISelfController::Exit` | #234 | area:services |
| proposed | **Multi-process kernel (later milestone):** a handle table, guest process id and `CURRENT_PROCESS` per process; end reports for other processes; `timer::deliver` no longer stopping at the first ended server; `is_stalled` across processes; a must-use return for `Vec<Ended>`. #234 and #243 are marked provisional, one process per set of tables. Pairs with the cross-process shared-memory rows (#218, #222) | #234, #243 | area:kernel |
| proposed | **Snapshot completeness (observability "Order of work", step 6):** a run-level snapshot, which removes the `dead_code` expect on the snapshot module; a `SharedPages` constructor for service restore (`Physical` is private to `space.rs`); check restored shared runs against the restored shared memories; refuse a hostile entry count at the first entry past the budget (a `DeserializeSeed`), not after counting the whole list | #261 | area:core |
| proposed | **Record the loader's mappings**: `load_placed` runs before the run's recorder exists | #261 | area:core |
| awaiting the owner | **Is milestone 2 reached, and do the GPU-front slices (G8 to G23) get filed for milestone 3?** | the title | roadmap |
| awaiting the owner | **A stopgap until the GPU front exists:** let `NullFront` perform host-method semaphore releases so the title gets past its spin. Gpu-protocol decision 2 has the back perform releases, in ring order, after the host work before them; a front that writes them while drawing nothing departs from that, so it is the owner's call. The alternative is to wait for G8 to G21 | the title | gpu-protocol |
| awaiting the owner | **nvmap releases its device-share lock during `IocFree` and at session close**, as Eden does and #243 asks, where the kernel spec's sentence has the lock held until GPU ordering allows. The departure is quoted in `crates/gsv/src/nvsrv/nvmap.rs`; the open wave-31 row on whether a free waits for GPU ordering is the same question | #243 | kernel |
| awaiting the owner | **A lock asked for after the process ended reaches a server as `TerminationRequested`.** #313 chose it over `InvalidCurrentMemory`; neither reference has this case | #243 | kernel |
| awaiting the owner | **`Family::session_closed`**, a count-only services hook #313 adds so nvsrv learns its last session closed: keep it, or fold it into the wave-31 session-close hook that runs with a `Context` | #243 | area:services |
| awaiting the owner | **Map events are stamped core 0 and the main thread** with a copy of the run's clock, not the dispatch's caller and `Env` reading. Pass the caller to the `Mapper`, or have the dispatcher record queued Map events under its lock? | #261 | observability |

### From wave 35 (ledger: #22)

Landed: #226's two missing tests (PR #317): host-I/O payload bytes count toward the kept recording's cap, tested at the exact one-byte edge, and the console check at boot comes before any draw. With it, the open milestone 2 issues are #33 (needs design), #205 (waiting on the owner's counter-register question) and #21 (the owner's).

| Status | Proposal | Why | Suggested |
|---|---|---|---|
| proposed | **Whole-run replay across several cores:** calls from different cores and the timer thread reach the `Env` in host-timing order, and the timer thread reads an unrecorded copy of the clock, so a whole-run replay is not deterministic. Needs recorded ordering (observability rule 3) or the deterministic interpreter backend (#47). Later milestone | #226 | observability |

### From closing milestone 2 (2026-11-18)

#33 closed as delivered by PR #127 (decision log, 2026-11-18); its leftovers wait here until a title needs them. #33's folder argument for `starquake run` is not listed: `run` now loads a NUP title's ExeFS.

| Status | Proposal | Why | Suggested |
|---|---|---|---|
| proposed | **Dynamic NRO loads through an `ldr:ro` service**: mapping and patching on the dynamic-load path. `CodeSet::parse_nro` and core's placement are what the service would call | #33 | area:loader |
| proposed | **Trampoline reach for large module sets**: every module's trampolines sit in one run below the first module, so a counter read more than 128 MiB above it is refused (`BranchOutOfRange`). Lifting it means placing trampolines between modules, off Eden's addresses, and checking the guess that `rtld` walks the address space upwards for code | #33 | area:loader |
| proposed | **The NPDM system-call mask check-and-log** (kernel spec, "System calls"): check whether process creation applies it; when #33 closed only the ACI0-against-ACID check was called, and it warned without refusing | #33 | area:kernel |
| proposed | **`RemoteMemorySize` (META 0x10)** is parsed and exposed raw, and nothing uses it; no source yet says what it sizes | #33 | area:loader |

### From wave 36 (ledger: #22)

Landed, the first milestone 3 wave: G8 (the channel pusher, #319, PR #335), G9 (front register files and shadow RAM, #320, PR #338) and G12 (QMD 3.0 decoding, the PCAS scheduler and QMD releases, #323, PR #336). Wiring notes for the later slices are comments on #321, #324, #325 and #330.

| Status | Proposal | Why | Suggested |
|---|---|---|---|
| done (#392, PR #399) | **GPU register defaults:** Eden runs `InitializeRegisterDefaults` (non-zero 3D defaults: depth range, swizzles, blend, stencil and colour masks) and then copies them into the shadow file. The front file, the shadow file and the back mirror start zeroed here; they must start from the same values, or same-value dropping drops a write the back never saw. Nothing owns it; needed before G13 and G14 run real streams | #320 | area:gpu |
| proposed | **GPU state snapshots:** save and restore the register and shadow files and the QMD table, with restore validation (the 4,096-entry cap, counter and deferred bounds). The spec puts this state in GPU snapshots; no slice covers it | #320, #323 | area:gpu |
| proposed | **`SEND_SIGNALING_PCAS_B` (0xAF) and `SEND_PCAS_B` (0xAE):** Eden-NX2 treats 0xAF with SCHEDULE set as a scheduler signal, so a QMD scheduled that way would never run here. Settle with a trace | #323 | area:gpu |
| proposed | **GPU front counters** for the observability summary: writes dropped as same-value, and other dropped writes | #320 | area:gpu |
| proposed | **A macOS CI job that runs everything except the hypervisor tests.** A test that compared constants by address (`std::ptr::eq`) failed only in macOS's default incremental build, and PR CI passed because the macOS job is skipped | #320 | infra |
| proposed | **Release forms still refused, as Eden-NX2 refuses them:** `STRUCTURE_SIZE` other than one word, `PAYLOAD64B`, `TRAP_TYPE`. A trace settles how they combine | #323 | area:gpu |
| decided (decision log, 2026-11-18, "gpu-protocol errata from wave 36") | **`SET_REFERENCE` has no ring entry.** The spec's "Host methods" lists it as one; the ring sketch and G4's `RingEntry` have none. G8 hands it to `ChannelSink::set_reference`. A `RingEntry` variant, or a spec edit, before G14 and G19 wire it | #319 | gpu-protocol |
| decided (decision log, 2026-11-18, "gpu-protocol errata from wave 36") | **Old-format method headers:** the spec's "The method stream" calls opcodes 0 and 2 subdevice-mask operations to skip; `clc76f.h` and G1's decoder read opcode 0 or 2 with tertiary opcode 0 as an old-format method header. G8 follows the header (decision 10); the spec sentence and facts table would be amended to match. An all-zero NOP word does nothing either way | #319 | gpu-protocol |
| decided (decision log, 2026-11-18, "gpu-protocol errata from wave 36") | **The macro upload pointer is not a register side effect.** The spec lists it as one; both Switch 2 references leave `LOAD_MME_INSTRUCTION_RAM_POINTER` (0x45) as written and track a word offset (pointer times 3) in the macro engine, which no register can hold. #338 follows the references; G13 owns the offset | #320 | gpu-protocol |
| decided (decision log, 2026-11-18, "gpu-protocol errata from wave 36") | **QMD word 4 bit 12** (method 0xCC) is `REQUIRE_SCHEDULING_PCAS` in `clc7c0qmd.h`; the spec says no header explains the condition. Cite the field in the spec | #323 | gpu-protocol |
| decided (decision log, 2026-11-18, "gpu-protocol errata from wave 36") | **What a QMD request that fails at run time leaves behind.** #336 keeps the request's own change, as Eden-NX2 does, and documents the exact sets; the spec does not say | #323 | gpu-protocol |
| decided (decision log, 2026-11-18, "gpu-protocol errata from wave 36") | **Replay of data-upload runs:** runs to data-upload methods send shadow values, where Eden-NX2's fast paths upload the raw words. A possible compatibility difference | #320 | gpu-protocol |

### From wave 37 (ledger: #22)

Landed: G13 (the front macro host, #324, PR #340) and G11 (DMA copy, inline-to-memory and 2D on the CPU path, #322, PR #341). G10 (#321) was built and pushed but stopped before its checks; wave 38 resumes it from its branch. Wiring notes for G14 and G19 are comments on #325 and #330.

| Status | Proposal | Why | Suggested |
|---|---|---|---|
| done (#392, PR #399) | **Engine register reset values:** seed Eden's `InitializeRegisterDefaults` (and Fermi2D's depth of 1, already exported as `twod::BLIT_RESET_REGISTERS`) through G10's `ClassRules::initial_values`, which seeds the front file, the shadow file and the back mirror from one place. Wave 36's row; G14 and G19 need it before real streams | #321, #322 | area:gpu |
| done (#393, PR #400) | **`MME_DMA_READ` (0x157) into data RAM** at `SET_MME_DATA_RAM_ADDRESS`: Eden-NX2 performs it, G13 does not (row 58 names only the FIFO form). The memo must not store such calls, since they change data RAM without a barrier | #324 | area:gpu |
| proposed | **The 2D format converter:** port `eden-nx2/src/video_core/engines/sw_blitter/converter.cpp` (about 1,200 lines). Blits between formats (`BlitConversion`) and bilinear filtering of formats other than the two 8888 ones (`BlitFilter`) are refused today | #322 | area:gpu |
| proposed | **GPU counters for MME DMA reads, catch-ups and reruns**, folded into wave 36's front counters row; verification item 3 cannot be read from a run today | #324 | area:gpu |
| proposed | **Macro store tests:** the cross-channel rewrite counter and `forget_channel` for the compute store and start entries; the memo stores a call whose write left the register file (arena space only) | #324 | area:gpu |
| proposed | **Root tables for constant-buffer banks 24 to 31** have no owner yet (G18, with verification item 6) | #321 | area:gpu |
| decided (decision log, 2026-11-18, "gpu-protocol errata from waves 37 and 38") | **`SET_PROGRAM_REGION` does not exist in `clc797.h`.** The spec's "3D: from state to a draw" names it; Ampere pipelines take a 64-bit address each through `SET_PIPELINE_PROGRAM_ADDRESS_A` and `_B` | #321 | gpu-protocol |
| decided (decision log, 2026-11-18, "gpu-protocol errata from waves 37 and 38") | **`clc7c0.h` does list the compute macro calls** (`CALL_MME_MACRO` at byte 0x3800); the spec's facts table says it does not | #324 | gpu-protocol |
| decided (decision log, 2026-11-18, "gpu-protocol errata from waves 37 and 38") | **Two code stores per process:** both references hold a 3D store and a compute store, and a channel's compute class uses the 3D store until it writes 0x45 or 0x47. The spec (and decision 11's wording) says one code store; G13 follows the references | #324 | gpu-protocol |
| decided (decision log, 2026-11-18, "gpu-protocol errata from waves 37 and 38") | **The macro start table's size:** `clc797.h` lists 256 `CALL_MME_MACRO` elements; Eden-NX2 wraps at 0x80, and G13 follows it | #324 | gpu-protocol |
| decided (decision log, 2026-11-18, "gpu-protocol errata from waves 37 and 38") | **A catch-up rerun after another channel rewrites the code:** G13 records the start the abandoned run used and reruns from it, reading the code as it is at the resume; listed as a divergence | #324 | gpu-protocol |
| decided (decision log, 2026-11-18, "gpu-protocol errata from waves 37 and 38") | **Overlapping copies:** G11 reads each line whole before writing it for a pitch-to-pitch DMA copy, as Eden-NX2 does, and reads the whole source region first (at most `MAX_COPY_BYTES`) for a copy with a block-linear side and for every 2D blit. The spec does not say | #322 | gpu-protocol |
| decided (decision log, 2026-11-18, "gpu-protocol errata from waves 37 and 38") | **Guessed bounds with no reference behind them:** at most 2^18 method writes and 0x1000 DMA words per macro run; `MAX_COPY_BYTES` (256 MiB) and `MAX_PENDING_RANGES` (1,024) for copies; vertex stream sizes (16 `SET_VERTEX_STREAM_SIZE` elements for 32 streams). Record them in the spec as guesses to confirm by trace | #321, #322, #324 | gpu-protocol |

### From wave 38 (ledger: #22)

Landed: G10 (back-thread engine mirrors, dirty groups and typed accessors, #321, PR #344, resumed from wave 37's branch), #342 (the 2026-11-18 rulings in `gpu::regs` and `gpu::back::qmd`, PR #345), G15 (fenced releases, completion and the completion wake, #326, PR #347) and G16 (the buffer cache, #327, PR #346). Shader-pipeline slice 9, which G18 waits on, is filed as #348 under the approved spec. Wiring notes are comments on #325, #328 and #329.

| Status | Proposal | Why | Suggested |
|---|---|---|---|
| proposed | **GPU-written buffers and textures (milestone 4):** on-demand write-back (decision 4), keeping them out of routine eviction (row 139), a `ReadbackId` allocator shared by the buffer and texture caches, and a G15 hook so the guest write happens after the readback's list completes. Waits for the first host-GPU writer | #327 | area:gpu |
| proposed | **`gpu::regs` and `gpu::back::state` follow-ups:** the front's `STATE_3D` still gives `SET_VERTEX_STREAM_SIZE` 16 elements where G10 gives 32; a contract and test for ungrouped registers; pin the legacy index registers 0x17d0, 0x17d4 and 0x17e0 | #321 | area:gpu |
| proposed | **Test chores:** one shared `pub(crate)` tracing-capture test helper (the fence and macro tests each have a copy); whether the buffer cache's `Spans` should be the memory crate's `PageSet` | #326, #327 | area:gpu |
| proposed | **The release timestamp's source:** `FenceClock::gpu_timestamp` is injected and not recorded; G21 picks the source (the spec's deferred queries say "from guest time"), and replay must supply the same clock | #326 | area:gpu |
| decided (decision log, 2026-11-18, "gpu-protocol errata from waves 37 and 38") | **Ruling 6's wording:** the spec says "only state methods take the shadow value". #345 lets every method that is not a data load take it, triggers included, as Eden-NX2's `CallMethod` applies `ProcessShadowRam` to every method. Amend the sentence to "every method except data loads", or narrow the code | #342 | gpu-protocol |
| decided (decision log, 2026-11-18, "gpu-protocol errata from waves 37 and 38") | **The 3D group table and typed accessors are hand-written,** keyed on the generated constants; the spec says they are generated (decision 6). Mesa's headers carry no group information. Accept hand-written tables, or extend `cargo xtask gen-classes` | #321 | gpu-protocol |
| decided (decision log, 2026-11-18, "gpu-protocol errata from waves 37 and 38") | **Releases name no address space:** the ring's `Release` and the copy and QMD releases carry no channel, so the back cannot tell which `GpuMemory` a release writes. Add a channel to the ring's `Release` (with G19), or track the issuing channel | #326 | gpu-protocol |
| decided (decision log, 2026-11-18, "gpu-protocol errata from waves 37 and 38") | **The buffer cache re-reads a range on its first use after arming its watch** (from the review notes on #197); the spec-literal alternative is a one-line change in `Entry::upload_spans` | #327 | gpu-protocol |
| decided (decision log, 2026-11-18, "gpu-protocol errata from waves 37 and 38") | **Buffer cache guesses:** `STREAMED_AFTER_FAULTS = 4` and the budget halves in `eviction_limits`; record them in the spec as guesses to confirm by trace | #327 | gpu-protocol |

### From filing the shader and render slices (2026-11-18)

The remaining shader-pipeline slices (#349 to #381, #348 before them) and render-backend slices (#382 to #389) are filed under the approved specs, in milestone 3 except slice 21 (milestone 4) and slice 22 (unscheduled, for a direct backend). Drafting them turned up these spec gaps.

| Status | Proposal | Why | Suggested |
|---|---|---|---|
| decided (decision log, 2026-11-18, "Render capture v2 no longer gates R5") | **Render capture v2 is to be designed "before R5"** (decision log, 2026-11-16), and no design exists. R11b (#388) needs it; R5 (#382) is otherwise ready to start. Let R5 go ahead while the lead drafts the capture v2 design as a design PR, or hold R5 until it merges | #382, #388 | render-backend |
| decided (decision log, 2026-11-18, "Shader and render slice plan errata") | **Shader slice dependencies:** the spec's critical-path sentence disagrees with its own table (13 needs 12e, 12f, 8 and 9; 10 needs 3 and 9; 15 needs 2; every 12x needs 7) and leaves out the cache (19); slices 8, 11 and 18 test through the interpreter (7) without listing it; 6d builds on structs 6a, 6b and 6e make; 12h's branches become flow-analysis edges (10) | #348 to #381 | shader-pipeline |
| decided (decision log, 2026-11-18, "Shader and render slice plan errata") | **Work no shader slice owns,** placed by the drafts: the local and shared memory pass (pass 6) in slice 16; the IR build step between flow analysis and the translators in slices 10 and 11; `cargo xtask shader-nvdisasm`; the speed test in slice 17 | #362, #363, #375, #376 | shader-pipeline |
| decided (decision log, 2026-11-18, "Shader and render slice plan errata") | **One shader capture format:** the spec does not say whether `shader-replay` reads slice 20's format or the shader records inside the GPU capture (`SQGC`, G20 #331). The drafts make slice 20's format the one G20 embeds | #379, #331 | shader-pipeline |
| decided (decision log, 2026-11-18, "Shader and render slice plan errata") | **R9b and R12 are circular:** R9b needs a backend in core, and R12, which chooses it, depends on R9. R9b can take G21's recording-backend wiring instead; also, no spec says how core sets the GPU crate's dump flag | #386, #389 | render-backend |
| decided (decision log, 2026-11-18, "Shader and render slice plan errata") | **Small spec fixes:** PR #4588's test file is `src/tests/video_core/native_shader.cpp`; render-backend's "Open with #12: how the back thread wakes" line is stale since decision 14 | #348 to #389 | shader-pipeline, render-backend |
| proposed | **Instructions with no reference source:** no encoding source for `JMX`, `UP2UR` and `BPT`; no translator in either Switch 2 reference for `IDP4`, `UP2UR`, `DSETP`, `AL2P` and `PIXLD` (Eden's Maxwell translators cover the last three). The decoder and translator issues mark them as guesses | #351 to #371 | area:shader |
| proposed | **Render verification 3 (wgpu on lavapipe in Linux CI)** has no owner or issue; until it runs, every GPU test stays local-only | #382 | infra |

### From wave 39 (ledger: #22)

Landed: G14 (the front loop, #325, PR #394), G17 (the texture cache, #328, PR #396), G23 (the pipeline-key cache, #334, PR #391) and shader slice 9 (questions, answers, the variant key and driver constants, #348, PR #390). G23's fix agent squashed and force-pushed its own PR branch before landing, against the house rule; the result was reviewed and is correct, and from wave 40 the rules forbid any history rewrite explicitly. Also fixed here: gpu-protocol's "Render backend (#14)" list still called the completion wake open.

| Status | Proposal | Why | Suggested |
|---|---|---|---|
| proposed | **Pipeline lifetimes:** the pipeline-key cache never destroys a pipeline, and a failed pipeline keeps its id reserved in the backend; no slice owns this | #334 | area:gpu |
| proposed | **Texture cache follow-ups:** depth textures read from guest memory need verification item 9 or a FermiZS byte order in `starquake-layout`; host copies for a DMA copy or 2D blit over a cached image (G11 still copies through guest memory); render's `host_format` fallback is not consulted; no joins or aliases between images of different layouts over the same memory; 1D buffer textures and row-87 huge images are refused | #328 | area:gpu |
| proposed | **Environment members slice 9 does not cover:** the texture type's dimension, array and rect; `TextureBoundBuffer`; `ReadViewportTransformState` (the row-65 position pass); compute `LocalMemorySize` from the QMD; `HasHLEMacroState`, `GetReplaceConstBuffer`, `GpPassthroughMask`. The slice that first needs each adds its question or key field (13, 15 or 17) | #348 | area:shader |
| awaiting the owner | **G14's guesses and departures** (in `front.rs`): the "process" for ordered updates and the macro store is the address space; `SYNC_WAIT` waits before the entry, as `clc76f.h`'s names read (Eden-NX2 waits after, behind a setting); a submission to an unknown or destroyed channel is dropped and answered `Queued`; an update after a barrier can be reported before it when the barrier waits on another space's channel; `BackWork` is answered from reported markers, so the first macro DMA read after any pushed work costs a catch-up; an empty submission is charged one queue place; the Eden-NX2 path's `WaitForIdle` difference | #325 | gpu-protocol |
| awaiting the owner | **`Syncpoint` ring entries carry no count,** so a guest asking for `u32::MAX` increments costs time linear in the count (pushed a quantum at a time). Give the entry a count | #325 | gpu-protocol |
| awaiting the owner | **GPU-modified images stay until the budget refuses new ones** (no write-back eviction yet) | #328 | gpu-protocol |
| awaiting the owner | **Shader identity across boots:** the precompile list keys each stage by a caller-supplied 128-bit `ShaderStamp`; what it holds (a digest of the shader cache's entry key, defined with shader slice 21) and how `core` maps stamps back to shaders at start-up are undecided; 128 bits is a guess | #334 | gpu-protocol, shader-pipeline |
| awaiting the owner | **Four render states are always dynamic:** `GraphicsPipelineDesc` has no field for the viewport, scissor, blend constant or stencil reference (only `DynamicValues`), so "keys only on what is not dynamic" cannot hold for a device that cannot set them per draw. wgpu, Vulkan and Metal all set them per draw | #334 | render-backend |
| awaiting the owner | **Pipeline-key cache choices:** `MAX_PIPELINES` = 65,536 (a guess, stricter than both references, which have no cap); a key is recorded when the backend accepts the pipeline, not after its build succeeds, because builds finish asynchronously | #334 | gpu-protocol |
| awaiting the owner | **`TextureKind` has no usage field,** so its answer always carries the header's format bits and a shader that only samples is retranslated when the format changes but the kind does not. Add a usage field to the question | #348 | shader-pipeline |
| awaiting the owner | **Interface specialisation:** the spec specialises a stage on what the next stage reads; Eden specialises on what the previous stage stores (`MakeRuntimeInfo`), and the GPU crate cannot build the next stage's reads from engine state. Slice 9 follows the spec provisionally | #348 | shader-pipeline |

### From wave 40 (ledger: #22)

Landed: G18 (from 3D state to a draw, #329, PR #404), G22 (production GPU memory and the GPU-only store, #333, PR #401), engine register reset values (#392, PR #399), `MME_DMA_READ` and the 256-entry start table (#393, PR #400), R5 (the wgpu backend core, `starquake-render-wgpu`, #382, PR #403) and shader slice 2 (the SPH v4 header, #349, PR #402). G19's blockers are all cleared; wiring notes for G19 and G21 are comments on #330 and #332.

| Status | Proposal | Why | Suggested |
|---|---|---|---|
| awaiting the owner | **`CLEAR_SURFACE` has no slice.** The render backend has `Command::Clear`, but gpu-protocol's slice table gives no slice the 3D clear method (Eden's `DrawManager::Clear`: a render-pass clear, or a draw for masked or scissored clears). A first present needs it. Recommendation: add a slice G24 after G18 and file it | #329 | gpu-protocol |
| proposed | **Draw-path completeness:** the viewport-transform-enable path; the per-stage input filter (attributes the vertex program does not load are still in the layout; Eden leaves them out); `ATTRIBUTE_FETCH` (skipped, with no slice for the storage-buffer path); `DrawTexture`; inline-index, zero-index, auto and indirect draws (skipped and counted); storage buffers and images in the resource table; tessellation and geometry stages. The first two are likely needed by milestone 3, the rest when a title needs them | #329 | area:gpu |
| proposed | **G18's guesses:** per-slot interpolation (flat for integers, perspective for floats); the constants buffer's 256-byte alignment (`Caps` names no uniform offset alignment); the alpha and logic-op functions reset to 0, which the header does not name, so they are read only while their test is on | #329 | area:gpu |
| awaiting the owner | **Rejected shaders are logged once per set, not once per program** as the spec says, because the shader cache's rejection does not say which program failed. Either the rejection names the program or the spec says "per set" | #329 | gpu-protocol, shader-pipeline |
| awaiting the owner | **The slice-19 seam:** slice 9's `Lookup::Ready` carries only the shader set, but the draw path also needs each shader's resource table, a per-shader `ShaderStamp` and a `wait(ticket)` the spec implies but never names. G18's `ShaderLookup`/`SetLookup` seam carries all three; slice 19 adapts to it | #329 | shader-pipeline |
| awaiting the owner | **Swizzle binding order:** a lookup needs the bound textures' swizzles, which depend on the set. Provisionally two lookups: first with the swizzles the same requests bound last time, then again if they differ | #329 | gpu-protocol |
| proposed | **G22 follow-ups:** the buffer and texture caches still request write watches at GPU ranges (`GpuMemory` needs a call returning a range's CPU pages, and the watch test should flip); the GPU-only store has no host cap (bounded only by the sparse regions gsv accounts for); no model-based test of `GpuOnlyStore`; store counters are not in `summary.json` (graphics-server verification item 6 wants them) | #333 | area:gpu |
| awaiting the owner | **`GpuMemory::generation`** is taken from the VM's `Translate::generation`, the unvetted proposal from the review notes on #197 | #333 | gpu-protocol |
| proposed | **Release latching:** `back::fence` latches waiters with a release's payload words rather than the incremented counter value; read the words back through memory after the write (noted on #330) | #333 | area:gpu |
| proposed | **Region id reuse:** gsv must not reuse a destroyed sparse region's id before `GpuOrdered` for that destroy, or the late free also frees the new region's pages (noted on #332) | #333 | area:gsv |
| awaiting the owner | **`MME_DMA_READ` (0x157) in the spec:** gpu-protocol's "Macros" section names only the FIFO form (0x158). PR #400 performs 0x157 at the write, as Eden-NX2 does, and the memo stores no call (3D or compute) that writes it. The compute class's MME DMA methods are register writes only, following Eden-NX2's compute front; a trace of a compute macro would decide. Add a sentence for each | #393 | gpu-protocol |
| proposed | **Stencil mask reset values** are 0xFFFFFFFF, as Eden writes them, while `clc797.h`'s field is 7:0 (decision 10: values follow Eden-NX2). A trace of a fresh channel's register reads would settle it | #392 | area:gpu |
| awaiting the owner | **Local memory size:** Eden computes `low \| high << 24`; NAK stores bits 32..55 in the high field. PR #402 follows Eden. Verification E3 with a large-local-memory shader would settle it. The CRS size (which PR #4588 adds) is exposed raw for slice 16 to decide | #349 | shader-pipeline |
| proposed: no change | **A compute program's header:** the spec says a compute program has no header, but PR #4588's native compute path starts at `sizeof(sph)` (0x50). Wave 41 settled it: those offsets are into PR #4588's lowered code buffer, and its capture reads a compute program's instruction 0 at the program address, which agrees with the spec (slice 3's scope note, #350) | #349 | shader-pipeline |
| proposed | **Slice 15 wiring:** `program::translate_program` still takes the stage and colour outputs by hand; slice 15 should call `ProgramHeader::from_program`, check the header's stage against the pipeline slot, and build `ColorOutputs` from the header | #349 | area:shader |
| proposed | **Render backend gaps for R6 and R7:** vertex divisors; null view and sampler bindings; a scratch path for copies within one resource (wgpu refuses them even when regions do not overlap); depth and stencil through buffers, which the D24S8 to D32FS8 conversion needs; unaligned buffer ranges (wgpu needs 4-byte alignment); `max_viewports` below 16; non-finite dynamic state; a shared list resolver for the recording and wgpu backends, which both resolve ids and apply the readback and render-target rules | #382, #329 | area:render |
| awaiting the owner | **Render backend choices R5 made:** sampler LOD bias is dropped (it probably belongs in the shader, R7), anisotropy is 1 when any filter is nearest, negative LOD clamps are raised to 0; guest 1D arrays have no wgpu equivalent (2D textures of height 1 is a shader-side decision); the stencil aspect of `D32_FLOAT_S8_UINT` is not copied because copy commands carry no aspect | #382 | render-backend |
| awaiting the owner | **render-backend errata from R5:** the spec pins wgpu 30.0.1, the workspace uses 30.0.2; wgpu refuses depth clear values outside 0..=1, and compressed formats cannot be cleared, settling two guesses in `view_ops` | #382 | render-backend |
| proposed | **Report upstream:** wgpu-hal 30.0.2's Metal encoder turns a missing `bytes_per_row` into 0 | #382 | owner |
| proposed | **Chore:** `crates/gpu/src/macros/tests.rs` keeps its own tracing capture; switch it to `crate::test_log` | #329 | area:gpu |

### From wave 41 (ledger: #22)

Landed: G19 (the back-thread executor, #330, PR #408; #330 stays open for host copies and blits between cached textures), R6 (pipelines on the compile pool, PR #407; #383 stays open for the driver cache blob), R8 (window presentation, #385, PR #406), R11b (render capture v2 in the render crate, #388, PR #409) and shader slice 6h (control and warp instructions, #358, PR #405). Shader slice 3 (#350) was built and pushed but stopped before its checks; wave 42 resumes it from its branch. Wiring notes are comments on #332, #384, #389, #397 and #362.

| Status | Proposal | Why | Suggested |
|---|---|---|---|
| awaiting the owner | **Subgroups on the Mac:** wgpu-hal 30.0.2's Metal backend reports a subgroup size range of 4..64 on every Mac, and wgpu cannot request a size, so R6 refuses every pipeline that uses subgroup operations (the lane id included) on the Mac. The spec already says Apple GPUs run 32-wide SIMD groups (shader-pipeline, "Compute programs"). Recommendation: the wgpu backend treats the Apple GPU family as meeting "subgroup size 32" | #383 | render-backend |
| awaiting the owner | **The driver pipeline cache:** wgpu 30's `create_pipeline_cache` is `unsafe`, and the spec allows `unsafe` in render-wgpu only for surface creation. The cache is Vulkan-only, so the Mac gains nothing either way. Recommendation: no driver cache under wgpu for now; revisit with a direct Vulkan backend | #383 | render-backend |
| awaiting the owner | **`ShaderDesc` lacks `requirements` and binding-array counts:** the GPU crate's conversion drops `HostShader::requirements`, and `ResourceBinding` has no array count, so R6 reads both from the naga module, departing from "layouts from `ShaderInterface`". Recommendation: carry both in the interface (touches the GPU crate's conversion and R11b's capture records) | #383 | render-backend, shader-pipeline |
| awaiting the owner | **No macro-end entry in the ring:** Eden-NX2 draws deferred instances at each macro's end (`FinishFrontMacro`); G19 draws them at other boundaries | #330 | gpu-protocol |
| awaiting the owner | **Present requests:** `PresentRequest` names no address space (G19 uses `set_present_space`, which G21 sets); the format mapping (1 to A8B8G8R8, 4 to A2B10G10R10 unorm) and ignoring the pitch are guesses | #330 | gpu-protocol |
| proposed | **Compute dispatches are all rejected** until the shader crate has a compute `HostStage` (slice 16, #375, lowers compute entry points); compute texture and sampler pools and compute driver constants are not read yet. Check that #375 and a GPU-side follow-up cover all three | #330 | area:shader, area:gpu |
| awaiting the owner | **`LOAD_CONSTANT_BUFFER` data is not written to guest memory** (Eden's `ProcessCBMultiData`); no slice in gpu-protocol's table owns it. Recommendation: a small GPU slice after G19 | #330 | gpu-protocol |
| proposed | **Host copies and blits between cached textures:** a DMA copy or 2D blit over a cached surface still runs on the CPU after a write-back; row 137 makes the host path the default. The rest of #330 | #330 | area:gpu |
| proposed | **Report semaphores** (`SET_REPORT_SEMAPHORE_A`-`D`): the spec defers queries, but conditional render reads their results, and no slice writes even the values the front can know | #330 | area:gpu |
| proposed | **Invalidate drain timing:** the drain runs at `MEM_OP_D` for every operation; Eden-NX2 drains at the `MEM_OP_B` write. Listed as a divergence | #330 | area:gpu |
| proposed | **Nesting depth across crates:** R6 refuses naga modules nested deeper than 256 (measured on this Mac: Metal compiles 256 and refuses 257, and 1536 overflowed a 2 MiB thread). Whether the shader crate's output stays within it, and the compile pool's stack size, are unverified; no layer owns the bound end to end | #383 | area:shader, area:render |
| proposed | **One wait setting:** `render::compile::WhileCompiling` and `capture::WaitSetting` mirror the GPU crate's `WaitWhileCompiling`; `core` and the GPU crate convert in one place (noted on #389 and #397) | #383, #388 | area:render |
| awaiting the owner | **R8's choices:** `WindowSize` is added to the spec's handles-only `WindowHandles`; the rotation corner mapping (no reference rotates), `FRAME_LATENCY` = 2 and the letterbox aspect taken from the presented rectangle (Eden uses a 16:9 setting) are guesses | #385 | render-backend |
| awaiting the owner | **render-backend errata from R8 and R11b:** verification 11 is settled (no `unsafe`: an owned window goes through wgpu's safe `create_surface`), and the surface is made in `set_present_target` on the main thread on macOS; "Render capture v2" still says "a draft for the owner" though it was approved on 2026-11-18 | #385, #388 | render-backend |
| awaiting the owner | **R11b's bounds and departures:** per-kind body caps (stricter than one `MAX_RECORD_BYTES`); a 1 GiB shader-source budget, 2 GiB `MAX_LIST_DECODED_BYTES` and 4096 `MAX_COMMAND_BINDINGS` (guesses); a 2 MiB `MAX_SHADER_SOURCE_BYTES` until R11c pins it to slice 20's constant; `Limit` written for any writer refusal; "only `End` after `Limit`" as a reading check; a memory-limit refusal counted rather than stopping the replay; held destroys; frames from late lists can still evict each other in the shared sink | #388 | render-backend |
| awaiting the owner | **Slice 3's hash seed:** the content hash is seeded with the header length (0 or 128), so a compute and a graphics program with equal bytes never share a cache entry | #350 | shader-pipeline |
| proposed | **nvdisasm check (E2)** of BRX/BRXU, CALL, RET, DEPBAR and BAR 0xb1d, whose encodings come from PR #4588 alone; JMX and BPT have no encoding source; BAR 0x31d and some WARPSYNC/NANOSLEEP forms wait for an E4 census | #358 | area:shader (M4) |
| proposed | **Hand check of a real window** on a Mac and a Linux desktop (resize, the main-thread layer), with R12's `--window` (noted on #389) | #385 | area:render |

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

### From wave 25 (ledger: #22)

| Status | Candidate |
|---|---|
| proposed | Every documented check and claim gets its own mutation (operand, mask, walk order, width, initial value); a surviving mutation gets a new test or is listed as equivalent or unreachable. |
| proposed | A boundary check is pinned by two cases that differ only at the edge, chosen so that the accepted side reaches the next check and both give different, exact outcomes. |
| proposed | Every constructor default is asserted directly on a fresh object, citing the reference default. |
| proposed | A first-fault walk order is tested with two or more faulty items, so a reversed walk fails. |
| proposed | A claim that two behaviours agree with the reference, or that a state is unreachable, traces the reference's other callers first, and names the exact exception. |
| proposed | Restore tests accept the real state that live operations leave, not only the refusal cases. |
| proposed | A doc claim of an error set, "logged once" or "same in replay" names the exact paths, with a test that fails if one is dropped. |
| proposed | A restored id or mount the current run may not support is validated against the live run, or its answer is a listed divergence pinned by a test. |

### From wave 26 (ledger: #22)

| Status | Candidate |
|---|---|
| proposed | When docs call a text search or scan exact, list the call forms it can't see, and test each one, or widen the scan. |
| proposed | Don't claim replay reproduces an outcome that depends on unrecorded host timing or state; name the unrecorded input in the docs. |
| proposed | A departure from a decision-log rule or a spec is said so in the docs and PR body, with its reason, and raised as an owner question. |
| proposed | A per-request handler never clones a container a client or snapshot can grow; a test shows per-call work doesn't grow with it. |

### From wave 27 (ledger: #22)

- When a table maps a name to a number (or two views of one fact), a test cross-checks the two independent sources instead of looping over the same table on both sides.
- When a gate depends on a combination of conditions, a test varies each condition alone, on the production-like configuration.
- Error text is read aloud against the empty, zero and one cases, and the zero case's exact string is asserted.
- A classification branch that picks a name, protocol or interface is tested for every variant that reaches it; a fallback records "unknown" rather than borrowing another object's name.
- Observability records are not keyed by a guest-controlled string alone, or the ambiguity is a listed divergence pinned by a test.
- A build agent returns only when its work is pushed; no background job may still be editing tracked files.

### From wave 28 (ledger: #22)

- Restore validation checks that ids of the same role are pairwise distinct and that no two slots share one kernel object.
- A branch that only changes a log level or other side effect gets a captured-log test, or the docs say its mutation is equivalent.
- Command ids, opcodes and layout constants are pinned against the cited reference values written as literals in one test.
- Attribute, flag and permission checks are pinned with every non-empty subset of flags on each state, against a model.
- An object made per session or per manager has a bounded lifetime, and any departure from the reference's lifetime is listed.
- A wiring change gets one end-to-end test through the production constructor that asserts the output reaches the final report or the guest.
- Each rollback error path is tested for every taking path: placed handles released, reference counts restored, services-side state forgotten.

### From wave 29 (ledger: #22)

- A new timer, bell or wake path gets a test that fails if that exact path is removed, with the effect asserted well inside the fallback interval.
- Each new background turn gets at least one test through the production thread or constructor and the real clock.
- A method that keeps the first or last of several failures is tested with two distinguishable failures.
- Boundary tests are placed where the guarded check is the only thing that can refuse, not at the end of the backing memory.
- Periodic work skips an idle turn before it reads the clock, so recorded readings don't depend on host timing.
- Where references disagree, the doc says "conflict", names the winner under decision 8, and pins each case with a named test.

### From wave 30 (ledger: #22)

- A doc that describes a reference's behaviour traces the exact function that produces it and names file and commit.
- Each unchecked restore relation states its exact consequence for the rest of the run, or is validated through the live kernel, and is pinned by a test.
- Restored state is checked for wedged or never-draining outcomes, and any restored count that drives a loop is bounded, with edge tests.
- Ids, handles and counters that start equal in a fresh rig are made to differ in at least one success case.
- Save-side validation is pinned as well as restore-side.

### From wave 31 (ledger: #22)

- Every data-reading command gets a table-driven short-payload test that trims the last word and asserts the exact refusal with the session unchanged.
- A step that can fail after the commit point is made non-fatal and logged, and the docs' "last step that can fail" matches the code.
- A restored counter that is later incremented is bounded so the next step cannot overflow, tested at `u64::MAX`.
- A test a doc names as pinning a behaviour exercises that exact path, and its comments describe what it sends.
- A constant taken from one reference while related bounds come from the other has its dependent bounds derived from it, or the coupling is flagged as an owner question.

### From wave 32 (ledger: #22)

- Every documented check or error order gets a test with both faults present at once, mutation-checked by swapping the order.
- A relation check is tested with cases where the two similar inputs differ, so a mutation that substitutes one for the other fails.
- Every error path after a partial side effect is listed and tested; test doubles get fault-injection knobs such as "fail from the Nth call".
- Code reachable only through an environment the production wiring cannot vary has its decision extracted into a pure function, or is documented as unreachable.
- Replay-sensitive code records the same notes as live, and a test feeds a live run's notes into a replay and asserts identical outputs.
- A design departure that narrows compatibility or drops a spec-listed field is flagged as an owner question, not landed as the rule's outcome.
- User-facing numeric inputs are validated against the downstream bound, and end-to-end tests pin dates, seeds and other environment inputs.

### From wave 33 (ledger: #22)

- Protocol constants are pinned against reference literals written out in one test, never only through the shipping constant.
- A test double enforces the real component's refusals, and a shared scenario runs against both.
- After changing a behaviour, docs, invariant lists, architecture tables and test names are searched for the old claim and updated in the same change.
- A refusal routed through an existing error variant gets its Display text and docs reread and its exact-string test updated.
- A review that did not run blocks the merge; it is never skipped as a finding with no file.

### From wave 34 (ledger: #22)

- A test of a value taken from a request, sender or event (core, thread, process) uses a sender that differs from the rig's defaults and from the handler's own, and asserts the exact field.
- Each branch of a deliver, settle or end path (an error while ending, first versus last of several requests) has its own test asserting the exact reports and their order.
- A departure from an approved spec sentence quotes the sentence in the module docs and PR body, and every path that triggers it is covered, not just the main one.
- A function whose result must be acted on (tokens to release, steps to forward) is `pub(crate)` and `#[must_use]`.
- Untrusted snapshots decode through a budget on entries and in-memory size, not only a byte-length bound, with a test at the exact edge.
- A request that arrives after its process or session ended is refused with the ended error and logged at the same level on every path that can produce it.

### From wave 35 (ledger: #22)

- A refusing size-bound test sets the limit at the exact edge (two records at twice the size less one), so an undercount of one byte flips the outcome.
- Size accounting is tested with items of different sizes, pushed in one call and across several calls.

### From wave 36 (ledger: #22)

- Constants are compared by value, method number or name, never by address (`std::ptr::eq`): promoted constants get addresses that vary with codegen units and incremental builds.
- The full gate is also run on the host platform's default dev profile (incremental, macOS), because Linux CI can pass tests that fail there.
- Test counts in a PR body come from the tree (`nextest list`), with `proptest!` blocks counted apart from the properties in them.
- Structurally invalid input (an unaligned address where alignment is documented) is refused at the API boundary with a named error, tested one step off the valid value.
- Where a failure can come after a state change began, the docs list which failing paths leave state untouched and which keep their own change, and a test pins one of each.
- Docs that claim reference or spec behaviour list every divergence (error reporting, check order, masked bits, read-back values), each with a named test; one that contradicts an approved spec sentence quotes it as an owner question in the module docs.

### From wave 37 (ledger: #22)

- A change that adds a way a function stops, returns or clears state updates every doc passage that lists those outcomes.
- State shared across channels or engines is bound when the work starts, not at resume (or a generation is recorded and checked), with a test where a sibling rewrites it in between.
- A chunked copy checks whether source and destination can overlap, and tests an overlap longer than one chunk.
- A divergence that applies to sibling modules sharing one helper is listed, with a named test, in each module.
- Reset and default register values come from the reference constructor, cited, and are tested with those registers left unwritten.
- A function that promises a flush or cleanup on error has a failure-path test asserting the exact ordered events.
- Register and field widths and signedness come from the reference types, tested with a value at or above 2^31.

### From wave 38 (ledger: #22)

- Table-driven mappings are pinned with hand-written literal cases from the reference, not expectations derived from the same table, covering each half of a paired entry.
- Every typed getter has a test that each register it reads marks its view's group.
- Set and bitmask operations are tested with operands that are not subsets and with clearing something already clean.
- A worker that takes work in priority order has a bounded starvation test where the high-priority source never empties.
- Ordering and carry-over logic is tested with two or more items in flight; removing the sole element of a collection is tested.
- Every documented ordering between a signal and another side effect is pinned by a test double that observes the effect at the moment of the signal.
- Every pool holding live bytes counts toward a documented memory cap, tested with a hostile growing sequence.
- A doc that cites a reference name is checked against the reference tree; a doc that says code is generated or mirrors a spec sentence is checked literally.

### From wave 39 (ledger: #22)

- Anything a hostile guest can repeat costs a nonzero unit of every queue, cap and budget, with one charge function shared by the fit check and the release path.
- When a change makes an object stale, every later pass that writes back, uploads or clears flags re-checks it when it acts.
- Every counter or queue bound has a test that fails when the release on any exit path is removed.
- Every documented wake, recheck, reset or poll-schedule rule has a test that fails when that line is removed, with the fake clock held still.
- Per-item lookups on hot paths are indexed, never a scan of the whole container.
- PR bodies and docs make no "all", "every" or "row for row" claim that was not checked literally.
- Published history is never rewritten: no rebase, squash, amend or force-push of a pushed branch.

### From wave 40 (ledger: #22)

- When mirroring a reference state machine, every register the reference reads on that path is implemented or listed as not read, with a divergence test.
- Ids, banks and indices from guest state are bounded above as well as below, with a named error and a test on both sides of each edge.
- One hostile value (a non-finite float, an oversized count, a host-limit mismatch) fails one item, never a whole submission.
- Every resource a layer creates is destroyed on drop and counted in the budget; per-operation scratch is reused.
- No allocation or linear scan per draw or call on hot paths, with a test that per-call cost does not grow.
- A claim that a rule matches a host API or reference cites the exact check in its source.

### From wave 41 (ledger: #22)

- A failure in deferred work that an entry drains first is logged and counted; the entry's own action still happens.
- Before destroying an object that queued commands may use, flush its deferred work and fence the open list.
- Cleanup and invalidation happen on every outcome of a guest-memory write, error paths after bytes were written included.
- Every check that can drop an output runs before the resource it outputs to is acquired.
- Hostile-input bounds cover nesting depth, with a named constant tested at the limit and one past it.
- Items a reader holds back from a hostile file count toward the same bound as live items.
- A claim naming a set is checked against the code and the reference, with each refused case listed as a divergence.
