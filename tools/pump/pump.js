// starquake's tuned copy of the autonomous-pump workflow template. See tools/pump/README.md.
//
// Differences from the generic template: two independent reviewers per PR (a fidelity review and
// a test-breaking review in a scratch clone; the stricter verdict wins), Opus for builds, fixes
// and reviews (decision log, 2026-11-12), per-checkout build folders, waiting for the PR's own CI
// before merging, entries that resume a stopped build, and starquake's house rules in RULES.
// autonomous-pump — Workflow template.
//
// USE: COPY this file, fill the CONFIG block below with literals, and run
// Workflow({scriptPath: "<your copy>"}). Inline config is the ONLY supported
// path — the Workflow tool's `args` reaches scripts unparsed in this build,
// so an args-reading fallback just wastes a launch. Adapt prompts to the
// project, but keep the quality steps (reachability proof, conventions lint,
// adversarial review, serial merge, no-silent-drops reconciliation).
//
// Flow: PER-ITEM PIPELINE (v3 — no head-of-line blocking): each issue runs
// build -> review -> land INDEPENDENTLY; a finished build goes straight to ITS
// adversarial review while slower siblings keep building. The ONLY
// serialization is the TRAIN QUEUE: merge/fix agents share one persistent
// train worktree and each merge advances main, so landings run one-at-a-time
// in COMPLETION order. Lessons + scope reconciliation close the wave.

export const meta = {
  name: "autonomous-pump",
  description: "Per-item pipeline (build -> adversarial review -> serial train landing, no cross-item barriers) + lessons + scope reconciliation, over a batch of issues.",
  phases: [
    { title: "Build" },
    { title: "Review" },
    { title: "Merge train" },
    { title: "Fix loop" },
    { title: "Lessons" },
    { title: "Scope reconciliation" },
  ],
};

// ── CONFIG (fill per run — literals only) ────────────────────────────────────
const REPO = "Syynth/starquake";
// Gate: build shared deps -> test -> typecheck. CACHE is prepended to every
// gate invocation so all agents share one build cache (turbo/nx: unchanged
// packages become cache hits — this is the pump's single biggest speed lever).
const CACHE = "unset CARGO_TARGET_DIR; export CARGO_INCREMENTAL=0";
const GATE = "cargo xtask ci && cargo clippy --workspace --all-targets --target x86_64-unknown-linux-gnu --locked -- -D warnings";
const MILESTONE = ""; // fill per wave: the milestone name scope reconciliation checks against
const LEDGER = 22; // optional: standing wave-ledger issue number (durable-by-default rule)
const WAVE_ID = "wNN"; // fill per wave: the ledger comment and the train folder use it
// Match the harness's configured co-author line for this model/session.
const TRAILER = "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"; // match the session's attribution
const CONV = "Rust 2024 workspace. FIRST read CLAUDE.md at the repo root and obey it (lint rules, thiserror, no unwrap/expect/panic/print in library code, #[expect(lint, reason)] never #[allow], tests first, Conventional Commits, dependencies only via [workspace.dependencies]). Reference sources to port from are in the folder printed by tools/refs-path.sh (see docs/references.md); cite what you port in file headers as paths relative to that folder. PR bodies end with the line: Generated with Claude Code.";
// Seed RULES from the previous wave's `lessons` output — that's the learning loop.
const RULES = "- PRIVACY: never write absolute local paths, usernames or home directories into code, comments, docs, commit messages, PR bodies or issue comments; reference sources are cited as paths relative to the reference folder (eden/src/...).\n- Stay inside the files your issue names; other modules in the same crate belong to sibling agents working right now.\n- Do not weaken the gate: no lint level changes in Cargo.toml or clippy.toml, no #[allow], no deleting tests; a new #[expect] needs a reason and a mention in the PR body.\n- Guest- or file-controlled input must never panic: checked arithmetic and bounds everywhere, errors via thiserror.\n- Tests must be deterministic and must not restate shipping constants as literals where the constant is importable.\n- NEVER run cargo clean: the build folder is shared with sibling agents. NEVER run git stash.\n- Hypervisor tests skip themselves ONLY when the CI environment variable is set.\n- Every proptest block must set an explicit fixed rng_seed (ProptestConfig with RngSeed::Fixed); never rely on proptest's default random seed.\n- Import shipping constants in tests and code (flags, offsets, limits, masks, opcodes, page sizes); never restate them as literals or redefine them locally. Share one pub(super) or pub(crate) definition and reuse decode helpers.\n- When a parser or decoder reads several ordered fields or lists, add a generated build-then-parse round-trip test that fails if any two reads are swapped; verify by mutating the code in a scratch copy.\n- Mark every layout or behaviour assumption not backed by a cited reference source as a guess in the docs; check doc claims about reference behaviour against the reference source; list intentional divergences explicitly and pin them with a test.\n- No blanket claims in docs or comments (every failure, all replies); state the exact set or condition and verify it against the code and the reference.\n- Check arithmetic on guest- or file-controlled addresses and sizes for the whole extent (start, end, trailing data), not just start points, and add an overflow test for each new bound.\n- Never paper over a flaky test with retries; reproduce it under plain cargo test as well as nextest, find the root cause, and document platform quirks. Verify PR-body claims (test counts, how tests were run) before stating them.\n- Host writes into guest-executable memory must do instruction-cache maintenance, with a test that rewrites code in place and asserts the new code runs.\n- When you add a test for a fix, run it against the unfixed code first and say in the PR that you saw it fail.\n- Derive test expectations from the shipping bound or constant, and place boundary cases exactly at the limit; verify any size or arithmetic claimed in a test comment.\n- Do not duplicate logic across sibling modules; extract one pub(crate) helper and call it from both.\n- Assert exact exit codes and error messages in negative tests; run the tests with the implementation removed and confirm they fail.\n- Bound loop and cycle detection by the structure actually walked (a visited set), not by input size.\n- When mirroring reference behaviour, cite the exact check and its constant; a stricter or looser behaviour is a listed divergence pinned by a test.\n- Read the approved specs in docs/specs/ that cover your crate before designing anything; the code follows them, and a conflict with a spec is reported in scopeNotes, not resolved by you.\n- Do NOT file GitHub issues for work you propose; put follow-ups in scopeNotes and the one issue comment the steps ask for.\n- When two crates share a unit or convention (ticks or nanoseconds, who converts), give the value a distinct newtype or name the conversion owner in one place.\n- Pin every serialised layout with a golden-bytes test covering every variant with distinct non-zero values.\n- Do not wrap a type in a second type that repeats its storage and logic; compose by wrapping the existing one.\n- Before merging, test-merge current main, build, and re-run the full gate; re-verify numbers quoted in the PR body.\n- Do NOT file GitHub issues for proposed work; follow-ups go in scopeNotes (the orchestrator records them in docs/proposals.md).\n- When docs promise an order of checks (A is reported before B), add a test with both faults present at once.\n- A failed streaming read is sticky: every later read returns the same error, never a clean end of file; test by reading twice.\n- Bound memory and time by what a hostile input can force, not only by loop counts (no per-entry clone of a growing path or prefix).\n- For a parser, add one hand-built accepting case per format feature, not only inputs made by the project's own builder or encoder.\n- User-facing errors and command output are plain words, not Rust Debug forms, asserted as exact strings.\n- Name a decoded bit position, shift or width once as a constant shared by decoder and encoder; decode tests give each field a distinct value.\n- Mutation-check your own tests before opening the PR: for each new branch, break it in a scratch copy and confirm a test fails; never claim a test pins a property it cannot observe.\n- Test every range operation (map, protect, unmap, query) with a neighbour that starts exactly at the end bound and assert the neighbour is unchanged.\n- Every state-transition branch gets a test of both the changed and the unchanged outcome.\n- When a layer relies on a guest permission or an error the reference documents, check that the production caller's backend honours it end to end, or enforce it in the layer.\n- Never discard with `let _ =` a result that reports a freed resource; route it through the one teardown path, on error paths too.\n- When a shared resource is taken out of its slot and the lock released, the admission logic must know it is out; pin it with a threaded test.\n- A test you write must not be able to hang the gate: bound every wait in a test with a timeout, and never run a test command without one.\n- When a merge of main conflicts with a sibling's work in the same files, resolve it so both features stay whole, and run both features' tests after the merge.\n- A rollback after a partly-applied change restores every field it wrote, not only the queue links; pin it with a test that fails midway.";
// Batch entries: { n, hint, closes?: [n, ...], lane?: "light", gate?: "<override>", model?: "opus" }
//  - closes: all issues this ONE PR honestly closes (clustered same-hot-file work);
//    defaults to [n].
//  - lane "light": docs/comments/banners-level changes — lower agent effort and
//    (via gate override) a cheaper gate, e.g. typecheck-only. Default lane: full.
const BATCH = [
  // Fill per wave, for example:
  // { n: 123, model: "opus", hint: "crates/foo only (spec slice X). Siblings: #124 edits crates/bar." },
  // Local paths (the reference folder, a title folder) may appear in a hint in your filled copy,
  // which stays outside the repository; never commit a filled copy (tools/pump/README.md).
];
// ─────────────────────────────────────────────────────────────────────────────

const DISK = `PRE-FLIGHT: run "df -h /System/Volumes/Data" FIRST (macOS: bare "/" is the sealed system volume and lies). If free space is under 20GiB, STOP and return ok:false with reason "ENOSPC risk" instead of building. BUILD FOLDER: use the default target folder of your own checkout; do NOT set CARGO_TARGET_DIR (a shared folder has served stale artifacts from sibling checkouts). Prefix cargo commands with: ${CACHE}. The macOS cargo runner signs test binaries automatically; do not bypass it.
RESUME, DON'T REDO: on start, check "git log origin/main..HEAD" and "git status" for a prior attempt's work in this worktree or on the remote branch (git fetch origin; git branch -r) and continue it. COMMIT AND PUSH work-in-progress to your branch at every green sub-step, so a dropped connection loses nothing.
NEVER "git stash": all worktrees share ONE stash stack, so a sibling agent can pop YOUR work-in-progress (this has happened). To set WIP aside, commit it to your own branch and amend/reset later.`;

// ⚠ ONE train worktree PER WAVE: suffix the path uniquely for THIS wave (e.g.
// /tmp/pump-merge-train-t8). CONCURRENT waves sharing one path race the same
// `train-pr` branch and have SILENTLY DROPPED files from merge trees (caught
// twice, by luck, in gate diffs). The persistence win is per-wave node_modules
// reuse — it never required cross-wave sharing.
const TRAIN_WT = `/tmp/starquake-train-${WAVE_ID}`;
const TRAIN_SETUP = `TRAIN WORKTREE — MANDATORY STEP 0, before ANY other git command: your shell starts in the USER'S ACTIVE worktree; running checkout/reset/merge there DESTROYS their in-progress work (this has happened MORE THAN ONCE — cd alone has proven insufficient because shell state resets between Bash calls). RULES: (1) \`cd ${TRAIN_WT}\` first (if absent, create it with \`git worktree add ${TRAIN_WT} --detach origin/main\` from the primary checkout, then cd); (2) EVERY git command you run for the rest of this task MUST be spelled \`git -C ${TRAIN_WT} …\` — the -C flag is not optional, even right after a cd, even for status/log; a bare \`git checkout\` in the wrong cwd is how user work gets destroyed; (3) non-git commands (cargo, gh) must be prefixed with \`cd ${TRAIN_WT} && \` in the same Bash invocation. Start clean: \`git -C ${TRAIN_WT} fetch origin --quiet && git -C ${TRAIN_WT} checkout --detach origin/main && git -C ${TRAIN_WT} reset --hard && git -C ${TRAIN_WT} clean -fd\`. Cargo.lock conflicts: take main's copy, then let a plain \`cargo check --workspace\` (shared cache exports, no --locked) add the missing entries, and commit the result.
⚠ ${TRAIN_WT} is the ONLY worktree you may use. NEVER use, cd into, or run git against ANY path under .claude/worktrees/ or any other "already provisioned" worktree you discover — those are the USER'S live sessions (an agent that migrated to one hijacked a dev server mid-drive and left it detached). If ${TRAIN_WT} looks contended or its state looks wrong (unexpected branch tip, files missing after a merge), do NOT relocate: re-verify against the PR head SHA via gh, recreate ${TRAIN_WT} from scratch, or return merged:false with the reason. After ANY merge, verify the PR's changed files are all present in the tree (git -C ${TRAIN_WT} diff --stat origin/main) before gating — a raced checkout can drop files with no conflict marker.`;

const gateFor = (b) => `${CACHE} && ${b.gate ?? GATE}`;
const effortFor = (b) => (b.lane === "light" ? "medium" : "high");
// MODEL TIERS (credit control): subagents inherit the session model by default,
// which burns the top tier on mechanical work. Builds/train/fix run on sonnet
// — the light lane lowers EFFORT and the gate, never the model (haiku struck
// from the pipeline by ruling, 2026-09-02: a "light" item still edits code
// with proof obligations, and a below-sonnet build just moves the cost into
// the review + fix round). Adversarial REVIEW stays one tier up (opus) — the
// review is the pump's quality bar and has repeatedly caught what the gate
// missed. Per-entry override: { n, model: "opus" } for a known-hard build.
const buildModelFor = (b) => b.model ?? "sonnet";
const REVIEW_MODEL = "opus";  // fidelity reviewer: spec + reference + design (strongest model)
const BREAK_MODEL = "opus";   // test-breaking reviewer: runs the code in a scratch clone
const rank = { approve: 0, changes: 1, reject: 2 };
const mergeVerdicts = (a, c) => {
  const vs = [a, c].filter(Boolean);
  if (!vs.length) return null;
  const decision = vs.map((v) => v.decision).sort((x, y) => rank[y] - rank[x])[0];
  const tag = (v, name) => (v?.findings ?? []).map((f) => "[" + name + "] " + f);
  return {
    pr: vs[0].pr, decision,
    findings: [...tag(a, "fidelity"), ...tag(c, "test-breaking")],
    scopeGaps: [...(a?.scopeGaps ?? []), ...(c?.scopeGaps ?? [])],
  };
};
const TRAIN_MODEL = "sonnet";
const closesList = (b) => (b.closes ?? [b.n]);
const closesLine = (b) => closesList(b).map((x) => `Closes #${x}`).join(", ");

// (v3 note: verdicts ride the pipeline WITH their item — no string-matching of
// PR references between phases, which once mis-parked approved PRs.)

const BUILD = {
  type: "object", additionalProperties: false,
  required: ["ok", "issue", "pr", "gateGreen", "reachability", "summary"],
  properties: {
    ok: { type: "boolean" }, issue: { type: "number" }, pr: { type: "string" },
    gateGreen: { type: "boolean" }, gateOutput: { type: "string" },
    reachability: { type: "string", description: "the concrete user path that exercises this change" },
    summary: { type: "string" },
    scopeNotes: { type: "string", description: "work discovered BEYOND this issue (hidden coupling, missing prereqs, an under-sized issue, adjacent follow-ups) — kept out of the PR; empty if scope held" },
  },
};
const VERDICT = {
  type: "object", additionalProperties: false, required: ["pr", "decision", "findings"],
  properties: {
    pr: { type: "string" }, decision: { type: "string", enum: ["approve", "changes", "reject"] },
    findings: { type: "array", items: { type: "string" } },
    scopeGaps: { type: "array", items: { type: "string" }, description: "where the issue/milestone UNDER-captured the work (scope the plan missed)" },
  },
};
const SCOPE = {
  type: "object", additionalProperties: false,
  required: ["assessment", "recommendation", "proposal"],
  properties: {
    assessment: { type: "string", enum: ["scope-held", "under-captured", "over-captured"] },
    discoveredWork: { type: "array", items: { type: "object", additionalProperties: false, required: ["title", "why"], properties: {
      title: { type: "string" }, why: { type: "string" }, suggestedMilestone: { type: "string" } } } },
    recommendation: { type: "string", enum: ["scope-held", "file-issues", "expand-milestone", "add-milestone"] },
    proposal: { type: "string", description: "concrete proposal for the HUMAN to approve: which issues to file + which milestone to expand or add, with rationale" },
  },
};
const MERGE = {
  type: "object", additionalProperties: false, required: ["pr", "merged", "detail"],
  properties: { pr: { type: "string" }, merged: { type: "boolean" }, detail: { type: "string" } },
};
const LESSONS = {
  type: "object", additionalProperties: false, required: ["lessons"],
  properties: { lessons: { type: "array", items: { type: "string" }, description: "generalizable house-rule candidates, imperative voice, ready to paste into the next wave's RULES" } },
};

if (!BATCH.length) { log("CONFIG.BATCH is empty — fill the CONFIG block."); return { error: "empty batch" }; }

// ── Orchestration: PER-ITEM PIPELINE (v3 — no head-of-line blocking) ─────────
// Each issue flows build → review → land INDEPENDENTLY: a finished build goes
// straight to ITS adversarial review while slower siblings keep building (the
// old all-builds barrier once parked two green PRs for ~an hour behind one
// slow sibling's retries). The ONLY serialization is this TRAIN QUEUE: merge
// and fix agents share the single persistent train worktree and every merge
// advances main, so landings run one-at-a-time in COMPLETION order.
let trainChain = Promise.resolve();
const enqueueTrain = (job) => {
  const run = trainChain.then(job, job);
  // The chain itself must never carry a rejection (it would poison every
  // later landing); each caller gets ITS job's promise.
  trainChain = run.then(() => undefined, () => undefined);
  return run;
};
let landedCount = 0;

const results = (await pipeline(
  BATCH,
  // BUILD (parallel across items, worktree-isolated).
  (b) => b.pr
    ? Promise.resolve({ ok: true, issue: b.n, pr: b.pr, gateGreen: true, reachability: "see the PR body", summary: "Built in an earlier wave that was interrupted before review or landing finished; the PR body describes the work." })
    : agent(`You are an autonomous build agent in a FRESH GIT WORKTREE. Implement GitHub issue #${b.n} of ${REPO} (${b.hint}) and open ONE PR${closesList(b).length > 1 ? ` that honestly closes ALL of: ${closesList(b).map((x) => "#" + x).join(", ")} (clustered because they share files)` : ""}. ${CONV}${RULES ? `\nHOUSE RULES (learned from prior reviews — obey):\n${RULES}\n` : ""}
${DISK}
STEPS (Bash):
1. ${closesList(b).map((x) => `\`gh issue view ${x} --repo ${REPO}\``).join(" + ")} — read full scope + acceptance.
2. \`git fetch origin --quiet && git checkout -B auto/issue-${b.n} origin/main\` (unless resuming a prior attempt per RESUME). Read CLAUDE.md, docs/architecture.md and docs/port-map.md, then the reference files the issue names (\`tools/refs-path.sh\` prints their folder).
3. Implement TDD. SCOPE DISCIPLINE: your own new files + minimal wiring; avoid shared hot files unless the hint names them; if you must touch one, make a minimal additive change so sibling PRs merge cleanly.
4. PROVE REACHABILITY: this is a library port with no application wiring yet, so state in \`reachability\` the PUBLIC entry points you added (exported from the crate, documented) and which tests drive each one. Private or untested code paths, and public items nothing exercises, are a FAIL.
5. CONVENTIONS LINT: no invented protocol facts — every layout, command id, encoding or constant must be traceable to a reference file or a doc row, cited in a comment; honor CLAUDE.md and the house rules; grep your diff for absolute paths and usernames.
6. Run the GATE to GREEN: \`${gateFor(b)}\`. Capture counts.
7. VERIFY THE DIFF IS REAL: \`git diff origin/main --stat\` must be NON-EMPTY and match what you believe you built — an empty diff means your edits didn't land (wrong cwd/worktree); STOP and fix rather than opening a hollow PR (one shipped with a fully fabricated body and was caught by review — an instant reject and a wasted cycle). Then stage explicit paths (NOT git add -A), commit (end the message with "${TRAILER}"), \`git push -u origin auto/issue-${b.n}\`.
8. \`gh pr create --repo ${REPO} --base main --head auto/issue-${b.n} --title "<concise>" --body "${closesLine(b)}\\n\\n<what + how + gate counts + reachability>"\` — only an HONEST closes list (if you delivered N of M deliverables, drop the unearned Closes and say so).
9. SCOPE OVERFLOW: if the work revealed anything BEYOND ${closesList(b).map((x) => "#" + x).join("/")} (hidden coupling, a missing prerequisite, an under-sized issue, obvious adjacent follow-ups), record it in \`scopeNotes\` AND post it as ONE comment on issue #${b.n} (\`gh issue comment ${b.n} --repo ${REPO} --body "Scope discovered beyond this fence: …"\`) so the record survives the session (durable-by-default rule, decision 2026-07-18). Do NOT grow this PR to cover it.
Return ok, issue, pr (number/url), gateGreen (true only if step 6 fully passed), gateOutput, reachability, summary, scopeNotes.`,
      { label: `build#${b.n}`, phase: "Build", effort: effortFor(b), model: buildModelFor(b), isolation: "worktree", schema: BUILD }),
  // REVIEW — starts the moment THIS item's build returns; siblings unaffected.
  (build, b) => {
    if (!build || !build.ok || !build.pr) return { build: build ?? null, verdict: null };
    if (b.prereviewed) return { build, verdict: { pr: build.pr, decision: "changes", scopeGaps: [], findings: ["Both reviews of this PR were completed in an earlier, interrupted wave and are posted as comments on the PR (read them with gh pr view --comments): a fidelity review and a test-breaking review. Apply EVERY finding in both. An interrupted fix attempt pushed its work in progress as the PR branch's newest commit (ungated): see what it already covers, finish it, and make the gate green."] } };
    const fidelity = () => agent(`You are the FIDELITY reviewer (a second reviewer independently runs the code and attacks the tests). Check this PR line by line against the approved spec in docs/specs and against the reference sources it cites (folder printed by tools/refs-path.sh), and judge the design: is this the code a careful senior engineer would want to build the rest on? Adversarially review PR ${build.pr} (issue #${b.n}) of ${REPO} for an autonomous merge decision. Try to REFUTE it. ${CONV}${RULES ? `\nHOUSE RULES:\n${RULES}\n` : ""}
Build reported: gateGreen=${build.gateGreen}; reachability="${build.reachability}"; ${build.summary}
READ-ONLY MANDATE: you have NO worktree — your shell cwd is the USER'S LIVE session worktree; NEVER run \`gh pr checkout\`, \`git checkout\`, \`git reset\`, \`git stash\`, or ANY state-mutating git command anywhere (a reviewer that checked out a PR branch in the user's worktree hijacked their live session). Inspect via \`gh pr diff\`/\`gh api\`/\`gh pr view\` and Read files at their committed paths; if you must run code, clone/fetch into a fresh dir under /tmp.
DO: \`gh pr diff ${build.pr} --repo ${REPO}\` + read changed files in context. Judge: correct + COMPLETE for #${b.n}? Actually REACHABLE/wired (not dead code)? In scope (no unrelated churn)? Conventions + house rules honored (no invented tokens)? Bugs / missing tests / regressions? Gate genuinely green? Decide approve | changes (list SPECIFIC actionable fixes) | reject (fundamentally wrong). Default to "changes" if materially off.
DISCIPLINE: every \`findings\` entry must be a REAL, actionable fix to THIS PR's diff — never placeholders. If your only material notes are follow-up scope (work outside this PR's fence), the decision is "approve" and those notes go in \`scopeGaps\`; "changes" with an empty/junk findings list wastes a fix-agent run.
ALWAYS post your verdict as ONE comment on the PR (\`gh pr comment ${build.pr} --repo ${REPO} --body "…"\`) — decision, findings, and scope gaps, INCLUDING approvals (durable-by-default rule, decision 2026-07-18: an approval's scope gaps are exactly the context that evaporates otherwise). Commenting is allowed under the read-only mandate; git state mutation is not. ALSO: note in \`scopeGaps\` anything the issue (or its milestone) UNDER-captured — work that clearly belongs but no issue covers. Return {pr, decision, findings, scopeGaps}.`,
      { label: `review#${b.n}`, phase: "Review", effort: effortFor(b), model: (b.review || REVIEW_MODEL), schema: VERDICT });
    const breaker = () => agent(`You are the TEST-BREAKING reviewer for PR ${build.pr} (issue #${b.n}) of ${REPO}. Another reviewer checks fidelity to the spec; your job is to find out whether this PR's tests would catch real bugs. ${CONV}
WORK ONLY IN A SCRATCH CLONE: your shell cwd is the USER'S LIVE checkout, so never run a state-changing git command there. Clone the repository into a fresh directory under /tmp (gh repo clone ${REPO} <dir>), cd into that clone inside EVERY command, check out the PR head there (gh pr checkout), and build with the clone's own default build folder (do not set CARGO_TARGET_DIR). Run df -h /System/Volumes/Data first and stop with a finding if under 20GiB is free.
DO: 1. Run the gate in the clone (cargo xtask ci) and record the real result and test count. 2. Read the PR diff and choose at least FIVE places where a plausible bug could hide (off-by-one, swapped fields or operands, wrong constant, missing bounds check, wrong error variant, wrong order of checks, a branch never taken). For each: make that ONE mutation in the clone, run the relevant tests, record whether any test failed, and revert it before the next. 3. Write at least THREE hostile inputs or call sequences the tests do not cover as scratch tests in the clone and record what happened (a panic, hang or wrong result is a finding). 4. Check the PR body's claims (test counts, what was run) against what you observed.
DECIDE: approve only if the gate is green, every mutation was caught, no hostile input misbehaved and the PR body is truthful. Otherwise changes, with ONE finding per surviving mutation or misbehaviour that says exactly which behaviour is untested or broken and what test would pin it. reject only if the gate is red in a way the PR hid.
Never push from the clone. Delete the clone directory when you are done. Post ONE comment on the PR (gh pr comment ${build.pr} --repo ${REPO}) with a table of the mutations (place, mutation, caught or survived), the hostile inputs and your verdict. Return {pr, decision, findings, scopeGaps}.`,
      { label: `break#${b.n}`, phase: "Review", effort: "high", model: BREAK_MODEL, schema: VERDICT });
    return parallel([fidelity, breaker]).then(([a, c]) => ({ build, verdict: mergeVerdicts(a ?? null, c ?? null) }));
  },
  // LAND — merge approvals / fix "changes", ONE at a time via the train queue,
  // in completion order. "reject" and failed builds park (human review).
  (r, b) => {
    const base = { issue: b.n, build: r?.build ?? null, verdict: r?.verdict ?? null };
    if (!base.build || !base.build.ok || !base.build.pr || !base.verdict) return { ...base, land: null };
    const decision = base.verdict.decision;
    if (decision === "reject") return { ...base, land: null };
    const job = decision === "approve"
      ? () => agent(`Land PR ${base.build.pr} (${closesLine(b)}) onto main of ${REPO}. Built off main in parallel; main has since advanced. ${CONV}
${TRAIN_SETUP}
STEPS: 1. In the train worktree: \`git fetch origin --quiet && git checkout -B train-pr origin/$(gh pr view ${base.build.pr} --repo ${REPO} --json headRefName -q .headRefName)\`. 2. \`git merge origin/main\` — combine ADDITIVE conflicts (registry/index appends: keep ALL entries; keep both wirings). ⚠ diff3/union conflict styles can duplicate or drop closing braces and adjacent lines when combining — after resolving, recheck brace balance and that BOTH sides' entries survived; the gate is the arbiter. If untangleable, \`git merge --abort\` + merged:false. 3. Run the GATE to GREEN: \`${gateFor(b)}\`. Fix semantic conflicts. 4. \`gh pr diff ${base.build.pr} --repo ${REPO}\` sanity-check. 5. If green + MERGEABLE (and AFTER pushing, WAIT for the PR CI checks on the pushed head: poll gh pr checks until at least the linux check is reported and nothing is pending; never merge on no-checks-reported or pending; a failed check means fix it or park with merged:false): \`git push origin train-pr:$(gh pr view ${base.build.pr} --repo ${REPO} --json headRefName -q .headRefName)\`, \`git checkout --detach origin/main\`, then \`gh pr merge ${base.build.pr} --repo ${REPO} --merge --delete-branch\` (detached first so the local-branch deletion can't fail). Else leave open + comment + merged:false. NOTEWORTHY-ONLY comment rule (durable-by-default, decision 2026-07-18): if the landing involved anything a future bisect would want — conflict resolutions, semantic fixes, files verified after a raced checkout — post ONE PR comment describing it; routine clean merges stay SILENT (no noise).
Return {pr, merged, detail (conflicts + gate counts + result, or park reason)}.`,
          { label: `merge#${b.n}`, phase: "Merge train", effort: effortFor(b), model: "opus", schema: MERGE })
      : () => agent(`Fix the review-blocking issues in PR ${base.build.pr} (${closesLine(b)}) of ${REPO}, then land it. The PR is otherwise good; apply the reviewer's SPECIFIC fixes only. ${CONV}${RULES ? `\nHOUSE RULES:\n${RULES}\n` : ""}
${TRAIN_SETUP}
REVIEWER FINDINGS — these ARE the adversarial review's verdict. They ride this workflow and are NOT posted on GitHub, so an empty \`gh api .../reviews\` or \`.../comments\` result means NOTHING — never treat it as evidence the findings aren't real (this dismissal has let a real stack-corruption bug merge). Treat each finding as authoritative and apply it. Skip an INDIVIDUAL finding only if it references files/lines absent from this PR's diff (a misattributed finding), and list every skipped finding with its reason in your report:\n- ${(base.verdict.findings ?? []).join("\n- ")}
STEPS: in the train worktree, \`git checkout -B train-fix origin/$(gh pr view ${base.build.pr} --repo ${REPO} --json headRefName -q .headRefName)\`; merge origin/main (combine; recheck brace balance per the diff3 caveat); apply ONLY the findings' fixes + tests that guard them; GATE to GREEN: \`${gateFor(b)}\`; commit (end with "${TRAILER}") + push back to the PR branch; THEN post ONE disposition comment on the PR (\`gh pr comment\`) tying commits to findings — which findings you applied and which you skipped with the misattribution reason (durable-by-default rule); WAIT for the PR CI checks on the pushed head (poll gh pr checks until at least the linux check is reported and nothing is pending; never merge on no-checks-reported or pending; a failed check means fix it or park); then if MERGEABLE, detach then \`gh pr merge ${base.build.pr} --repo ${REPO} --merge --delete-branch\`. If a finding needs a human decision (scope split, design call), DON'T guess — leave the PR open, comment, and report merged:false with what's needed.
Return {pr, merged, detail}.`,
          { label: `fix#${b.n}`, phase: "Fix loop", effort: effortFor(b), model: "opus", schema: MERGE });
    return enqueueTrain(job).then((m) => {
      const land = { kind: decision === "approve" ? "merge" : "fix", ...(m ?? { merged: false, detail: "no result" }) };
      if (land.merged) landedCount += 1;
      log(`train: #${b.n} ${land.merged ? "LANDED" : "PARKED"} via ${land.kind} (${landedCount} landed so far)`);
      return { ...base, land };
    });
  },
)).filter(Boolean);

// Flat views for the wrap-up phases (shapes match the old barrier variables).
const built = results.map((r) => r.build).filter(Boolean);
const reviewed = results.map((r) => r.verdict).filter(Boolean);

phase("Lessons"); // close the learning loop mechanically: findings -> next wave's RULES
const allFindings = reviewed.flatMap((v) => (v?.findings ?? []).map((f) => `${v.pr} [${v.decision}]: ${f}`));
const lessons = allFindings.length
  ? await agent(`Distill HOUSE-RULE candidates from this wave's review findings for ${REPO}. A lesson qualifies only if it GENERALIZES (would prevent a class of mistake in future builds), not a one-off fix. Merge duplicates; imperative voice; one line each ("use only tokens from X", "wire features into the UI, not just a hook"). 0-6 lessons — an empty list is a valid answer.
FINDINGS:\n${allFindings.map((s) => "- " + s).join("\n")}
Current house rules (don't repeat):\n${RULES || "(none)"}
Return {lessons}.`,
      { label: "lessons", phase: "Lessons", effort: "medium", model: "sonnet", schema: LESSONS })
  : { lessons: [] };

phase("Scope reconciliation"); // did building reveal work the plan didn't capture? PROPOSE, don't auto-restructure.
const scopeSignals = [
  ...built.filter((b) => b && b.scopeNotes && b.scopeNotes.trim()).map((b) => `#${b.issue} (build): ${b.scopeNotes}`),
  ...reviewed.flatMap((v) => (v?.scopeGaps ?? []).map((g) => `${v.pr} (review): ${g}`)),
];
const scopeReconciliation = await agent(`You reconcile PLANNED scope against what this batch's building actually revealed, for ${REPO}${MILESTONE ? ` — milestone "${MILESTONE}"` : ""}. The pump SURFACES; the human decides milestone structure — so PROPOSE, do NOT create issues or milestones.
Discovered-scope signals from this batch:
${scopeSignals.length ? scopeSignals.map((s) => "- " + s).join("\n") : "(none reported)"}
DO: weigh those signals; \`gh issue list --repo ${REPO}${MILESTONE ? ` --milestone "${MILESTONE}" --state all` : ""}\` (and read the roadmap/plan doc if the repo has one) to see what's already captured. Judge whether ${MILESTONE ? `the "${MILESTONE}" milestone` : "this batch"}'s scope HELD, UNDER-captured, or OVER-captured. List concrete NEW work no existing issue covers (dedup against the board). Recommend exactly one: scope-held | file-issues | expand-milestone | add-milestone — with a crisp \`proposal\` the human can act on.${LEDGER ? `
FINALLY (durable-by-default rule): append ONE compact wave-ledger comment to issue #${LEDGER} (\`gh issue comment ${LEDGER} --repo ${REPO}\`): wave id ${WAVE_ID}; the batch; landed/parked; the lessons harvested this wave (the orchestrator passes them below if any); your scope assessment + proposal summary. Issues later filed from the proposal carry the pump:scope label; graduated lessons carry pump:lesson.` : ""}
Return {assessment, discoveredWork:[{title,why,suggestedMilestone}], recommendation, proposal}.`,
  { label: "scope-reconcile", phase: "Scope reconciliation", effort: "high", model: "sonnet", schema: SCOPE });

// Reconciliation: every built+ok PR must be accounted for — merged, or parked
// WITH a reason. No silent drops. (v3: verdict + landing ride each item, so
// this is a straight per-item read — no cross-phase matching.)
const landedList = results
  .filter((r) => r.land?.merged)
  .map((r) => ({ issue: r.issue, pr: r.land.pr ?? r.build?.pr, via: r.land.kind, detail: r.land.detail }));
const parked = results
  .filter((r) => r.build && r.build.ok && r.build.pr && !r.land?.merged)
  .map((r) => ({
    issue: r.issue, pr: r.build.pr, decision: r.verdict?.decision ?? "(no verdict)",
    reason: r.land?.detail
      ?? (r.verdict?.decision === "reject" ? `rejected: ${(r.verdict?.findings ?? []).join("; ")}` : "not approved"),
    findings: r.verdict?.findings,
  }));
const buildFailed = results
  .filter((r) => !r.build || !r.build.ok || !r.build.pr)
  .map((r) => ({ issue: r.issue, summary: r.build?.summary ?? "(no build result)" }));

return {
  landed: landedList,
  parked,        // open PRs needing attention (with reasons / findings) — review these
  buildFailed,   // never reached a PR
  lessons: lessons?.lessons ?? [],  // paste-ready house-rule candidates for the next wave's RULES
  scopeReconciliation, // PROPOSAL for the human: did scope hold? new issues / milestone to add or expand?
  counts: { batch: BATCH.length, landed: landedList.length, parked: parked.length, buildFailed: buildFailed.length, lessons: (lessons?.lessons ?? []).length, scopeDiscovered: scopeReconciliation?.discoveredWork?.length ?? 0 },
};
