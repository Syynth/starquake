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

Statuses: `proposed`, `promoted #N`, `dropped: <reason>`.

## Work items

### From wave 2 (ledger: #22)

| Status | Proposal | Why | Suggested |
|---|---|---|---|
| proposed | **chore(xtask): unused-dependency check in the gate**; drop the unused workspace `bitflags` | #39 removed the IPC crate's unused dependencies by hand, but nothing stops it recurring. `bitflags` is now unused in the root `[workspace.dependencies]`. | infra, M1 |
| proposed | **chore(ipc): de-duplicate the request and reply builders** and the test-side assembler; export the domain limit constants | Duplicated buffer-write logic, inconsistent overflow behaviour (saturate in one place, `FieldOutOfRange` in another), a test assembler the request builder supersedes, and inline literals for the domain header limits. | area:ipc, M1 |
| proposed | **docs: pin libnx as a reference** and verify the request builder's raw-size and domain conventions against it | The builder's size conventions are marked as guesses. libnx is not a pinned reference, and Eden only parses requests. | area:ipc, M1 |
| proposed | **ci: cargo-deny operability**: a scheduled advisory run, a documented triage and ignore path, an offline switch for `cargo xtask ci`, GitHub's `Revert "..."` PR titles, and the macOS PR job's cache budget | A new advisory can turn `main` and every PR red with no code change; the gate now needs network access; the title check rejects GitHub's own revert titles; per-PR macOS caches count toward the 10 GB limit. | infra, M1 |
| proposed | **infra: shared build-folder safety across worktrees** | During wave 2 a stale or sibling `xtask` binary ran under `cargo xtask ci` in one worktree, and mutated scratch copies left stale artifacts. | infra, M1 |
| proposed | **test: a homebrew-built NSO/NPDM fixture** | Several loader facts stay unconfirmed without a real file: NSO hash flags, module-name words, the wildcard form, the ACID access-control base, and the ACI0-against-ACID rules. Overlaps the test strategy's homebrew fixture decision (#8). | area:loader, M1 |
| proposed | **decision: required-checks policy** | Branch protection is unavailable on this private repo (HTTP 403). The macOS PR job is path-filtered and may be absent, the merge queue must check the checks itself (wave 2 merged two PRs before their CI finished), and a move to squash merges would leave the PR-title check as the only subject enforcement. | owner, infra |
| proposed | **feat(fs): port the path normaliser** (`fs_path_utility`) | RomFS `find_dir`/`find_file` reject non-normalised paths, so the file-system service needs normalisation that matches Eden's. #16 cited the source but did not port it. | area:loader, M2 |
| proposed | **feat(loader): ExeFS title loading and RomFS mount**, including the NCA container and the `path_hash` signedness question | The PFS0 and RomFS parsers are reachable only from tests. Titles need a seekable source from an NCA container, and hashing of bytes 0x80 and above is unsettled. | area:loader, M2 |
| proposed | **feat(loader): expose the Switch 2 `RemoteMemorySize` word** (META 0x10) | Not exposed by NPDM parsing; Switch 2 process setup will probably need it. | area:loader, M2 |

### Folded into existing issues (wave 2)

- Validation call sites and the refuse-or-warn policy → #33.
- The virtual-to-physical core map and core count (4 or 6) → #9.
- Wiring the request builder into synthetic guests and golden traces → #8.
- Whether `proptest-regressions/` is committed → #38.

## House-rule candidates

Rules distilled from review findings. A vetted rule moves into `CLAUDE.md`; until then it is
passed to build agents as a provisional rule for the next wave.

### From wave 1 (used provisionally in wave 2)

| Status | Rule |
|---|---|
| proposed | Every proptest block sets an explicit fixed seed; never rely on proptest's default random seed. |
| proposed | Import shipping constants in tests and code; never restate them as literals or redefine them locally. Share one definition and reuse decode helpers. |
| proposed | When a parser reads several ordered fields or lists, add a generated build-then-parse round-trip test that fails if any two reads are swapped. |
| proposed | Mark every layout or behaviour assumption not backed by a cited reference as a guess; check doc claims about reference behaviour against the reference; list intentional divergences and pin them with a test. |
| proposed | No blanket claims in docs or comments ("every failure", "all replies"); state the exact set or condition. |
| proposed | Check arithmetic on guest- or file-controlled addresses and sizes for the whole extent, not just start points, with an overflow test for each new bound. |
| proposed | Never paper over a flaky test with retries; reproduce it under plain `cargo test` as well as nextest, find the root cause, and document platform quirks. Verify PR-body claims before stating them. |
| proposed | Host writes into guest-executable memory do instruction-cache maintenance, with a test that rewrites code in place. |
| proposed | When adding a test for a fix, run it against the unfixed code first and say in the PR that it failed. |

### From wave 2

| Status | Rule |
|---|---|
| proposed | Derive test expectations from the shipping bound or constant, and place boundary cases exactly at the limit; verify any size or arithmetic claimed in a test comment. |
| proposed | Don't duplicate logic across sibling modules; extract one `pub(crate)` helper and call it from both. |
| proposed | Assert exact exit codes and error messages in negative tests, so a missing or broken tool can't pass them; run the tests with the implementation removed and confirm they fail. |
| proposed | Bound loop and cycle detection by the structure actually walked (a visited set), not by input size; test a small loop inside a large input. |
| proposed | When mirroring reference behaviour, cite the exact check and its constant and don't conflate similar checks; a stricter or looser behaviour is a listed divergence pinned by a test. |
| proposed | Widen CI path filters to cover every input that can break the job (dependent crates, workspace manifest, lockfile, toolchain and lint config). |
| proposed | The merge queue waits for a PR's own CI checks before merging; a green local gate is not enough. |
