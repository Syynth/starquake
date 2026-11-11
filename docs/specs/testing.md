# Test strategy

Status: **proposal, awaiting owner review** (issue #8).

Most of starquake is written by agents and merged on a green gate, so the tests are the real
specification. This spec says how each layer is tested without the rest of the emulator, and
without keys, firmware or games.

## Principles

- **Every port task ships with a test that could fail.** Written first, seen red, then green.
- **Prefer an independent oracle** over a round trip through the code under test: bytes laid out
  by hand from the documented format, expected values taken from a reference implementation or a
  real assembler, traces recorded from a run. A builder and a parser written by the same agent
  agree with each other even when both are wrong.
- **Untrusted input never panics.** Every parser and decoder has a property test that feeds it
  arbitrary bytes.
- **Deterministic.** Explicit seeds, no dependence on hash-map order, wall-clock time or
  cross-thread timing.
- **Nothing proprietary in the repository.** Fixtures are synthetic or built from our own
  homebrew source.

## Tiers

| Tier | Runs | What | Where |
|---|---|---|---|
| 1. Unit and property | Every gate run, every host | Pure logic: formats, framing, tables, interpreters, kernel bookkeeping | `#[cfg(test)]` and `tests/` |
| 2. Hardware | Gate on a real Mac; skipped only when `CI` is set | Anything that creates a VM: vCPU exits, stage-1 tables against real translation, patched code actually running | `crates/cpu`, later `crates/kernel` |
| 3. Guest programs | Gate on a real Mac | Synthetic and homebrew programs run end to end, compared against golden traces | `tests/` in `crates/core` |
| 4. Titles | Never in CI; local only | Real titles from the owner's dumps, located by an environment variable | Bring-up sessions |

## Oracles by layer

| Layer | Oracle |
|---|---|
| IPC framing | Hand-assembled message bytes; Eden's layout structs as the cross-check |
| File formats | Bytes written at documented offsets; an independent reader where one exists (Eden-NX2's `npdm_info.py`) |
| Code patcher | Instruction words checked against a real assembler; a tiny interpreter for the trampoline; tier 2 run under the hypervisor |
| CPU backend | Hand-written guest programs with known register results |
| Page tables | Software walker against the builder everywhere; against a real vCPU in tier 2 |
| Kernel objects | Scenario tests ported from Horizon's documented behaviour and Eden's logic; model-based property tests for the scheduler and address arbiter |
| Services | Golden traces (below) |
| Macro processor | Mesa's simulator tests, which were verified on hardware |
| Shader translator | Per-instruction fixtures (encodings from Mesa NAK, behaviour from PR #4588's tests); every translated shader must pass `naga` validation |
| GPU command stream | Recorded method streams replayed into the front end, compared against expected engine state |

## Shared test support

A small `starquake-testkit` crate (dev-dependency only) so each crate doesn't reinvent it:

- **A64 snippets:** encoders for the dozen instructions guest test programs need (`movz`, `svc`,
  `b`, `ldr`/`str`, `mrs`, `ret`, …), each verified once against a real assembler.
- **Synthetic programs:** wrap a snippet as an NRO so the loader, kernel and CPU tests share one
  way of producing a runnable guest.
- **In-memory fakes:** a `GuestMemory`, a `Vcpu` that plays back scripted exits, and a host for
  the macro processor.

## Golden traces

A trace is a text log of what a guest program did at the system boundary: one line per system
call and IPC request, with decoded arguments and results, grouped per guest thread so that
cross-thread timing doesn't change the file. Tests run a program and compare the trace with a
committed snapshot (`insta`). A behaviour change shows up as a reviewable diff.

The same format is what auto-stubbed (unimplemented) calls are logged in, so a trace from a real
title doubles as a to-do list.

## Homebrew fixtures

Milestone 1 needs a real homebrew program, not only synthetic snippets. Proposal: keep the
program's source in `tests/homebrew/`, build it with devkitA64 by hand when it changes, and commit
the small binary next to its source with the toolchain version recorded. CI doesn't need the
toolchain.

## Fuzzing

`cargo fuzz` targets for the parsers and decoders (IPC, NSO/NRO/NPDM, macro and shader
decoding). Not part of the gate; run before releases and when a parser changes.

## Decisions asked of the owner

1. **Commit prebuilt homebrew binaries** (built from in-repo source) rather than building them in
   CI. Recommended: keeps the gate fast and toolchain-free.
2. **A `starquake-testkit` crate** as the one place for test helpers.
3. **Tier 2 tests fail, not skip, on a Mac that can't create a VM**, and skip only on hosted CI.
   That makes a broken entitlement or runner visible immediately.
4. **`insta` snapshots for traces**, reviewed like code.
