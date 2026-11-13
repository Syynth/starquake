# The pump: how a wave of agent work is run

`pump.js` is the workflow script the lead session uses to build a batch of GitHub issues in
parallel and land them on `main` one at a time. It is starquake's tuned copy of the
`autonomous-pump` workflow template (a skill in the owner's own Claude setup, not in this
repository). This copy carries what the project learned running it.

## What one wave does

For each issue in the batch, independently:

1. **Build.** An agent implements the issue in a fresh git worktree, runs the gate, and opens one
   pull request.
2. **Review, twice.** A fidelity reviewer checks the change against the spec, the references and
   `CLAUDE.md`. A test-breaking reviewer clones the branch into a scratch folder, mutates the
   code to see which changes no test catches, and tries hostile inputs. The stricter verdict
   wins.
3. **Land.** One agent at a time, in a single merge worktree, merges current `main` into the
   branch, applies every review finding, reruns the gate, waits for the pull request's own CI to
   pass, and merges.

After the batch: one agent distils review findings into candidate house rules, and another
compares what was built with the milestone's scope and proposes follow-ups. Both go into
`docs/proposals.md` through the lead, not as issues.

## Running a wave

1. Copy `pump.js` to a scratch folder outside the repository. A filled copy may name local
   folders (the reference sources, a title folder) in its hints, so it is never committed.
2. Fill the configuration block: `WAVE_ID` (the next wave number, for example `w18`; the merge
   worktree's folder takes its name from it), `MILESTONE`, and `BATCH`. Each batch entry is one
   issue: `{ n, model, hint }`, plus `closes` when one pull request honestly closes several
   issues. The hint names the files or crates the issue may touch and what the other entries in
   the batch are touching, so siblings stay out of each other's files.
3. Add the previous wave's adopted house rules to `RULES` if they are not already in
   `CLAUDE.md`.
4. Dry-run it with stubbed agents: `node tools/pump/check.js <your-copy.js>`. It should print
   `ran ok` with one landed entry per batch item.
5. Run it with the `Workflow` tool, passing the copy as `scriptPath`.

To continue a stopped wave, edit the copy if needed and pass `resumeFromRunId`: agents whose
prompt and options did not change return their saved results.

## Limits and rules

- **Three to four builds at once** on an eight-core machine (`docs/roadmap.md`). Give each entry
  a different crate where possible.
- **Models:** builds, fixes and both reviews on Opus (decision log, 2026-11-12). Routine steps
  (lessons, scope) on Sonnet.
- **Build folders:** each agent builds in its own worktree's `target/`; a shared
  `CARGO_TARGET_DIR` served stale builds (decision log, 2026-11-12).
- **Merging:** never before the pull request's own CI has reported and passed.
- **Worktrees:** the merge worktree is per wave (`/tmp/starquake-train-<WAVE_ID>`). After a wave,
  the lead removes it and the build worktrees under `.claude/worktrees/`.
- **Privacy:** no local paths, usernames, or anything read from a game dump in code, commits,
  pull requests or comments. A title-related task reports its local check in plain words only.
