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
| dropped: the merge queue now waits for PR checks; the self-hosted runner question is recorded in the test strategy spec | **decision: required-checks policy** | Branch protection is unavailable on this private repo (HTTP 403). The macOS PR job is path-filtered and may be absent, the merge queue must check the checks itself (wave 2 merged two PRs before their CI finished), and a move to squash merges would leave the PR-title check as the only subject enforcement. | owner, infra |
| folded into the `fsp-srv` trial slice (service framework spec, slice 6) | **feat(fs): port the path normaliser** (`fs_path_utility`) | RomFS `find_dir`/`find_file` reject non-normalised paths, so the file-system service needs normalisation that matches Eden's. #16 cited the source but did not port it. | area:loader, M2 |
| proposed | **feat(loader): ExeFS title loading and RomFS mount**, including the NCA container and the `path_hash` signedness question | The PFS0 and RomFS parsers are reachable only from tests. Titles need a seekable source from an NCA container, and hashing of bytes 0x80 and above is unsettled. | area:loader, M2 |
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
