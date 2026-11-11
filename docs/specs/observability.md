# Observability, recording and replay

Status: **approved by the owner, 2026-11-11** (issue #46). Builds on the execution model and the
kernel spec, and adds requirements to each.

Most debugging on this project will be done by agents working from files, often on machines with
no hypervisor, no GPU and no games. This spec decides what starquake records while it runs, what
can be replayed afterwards, and what a run leaves behind.

## What can be replayed

| Layer | Exact replay? | How |
|---|---|---|
| Kernel and services | Yes | Re-run them from the event log. No game, no hypervisor, any host |
| GPU | Yes, later | Capture a frame's command stream and the memory it reads; replay it offline |
| Guest code on a native backend | No | Six real cores interleave differently every run. Use snapshots, the flight recorder and a live debugger |
| Guest code, exactly | Only on the deterministic interpreter (#47) | Slow; practical for test programs and homebrew, not full titles |

The first row is the valuable one, and it follows from two choices already proposed: the single
kernel lock gives every system call a place in one global order, and the arena makes the kernel a
state machine. The rest of this spec is mostly about not breaking that property.

## The event stream

Every crossing of a boundary we own produces one structured event:

```rust
struct Event { seq: u64, time: GuestTicks, core: u8, thread: ThreadId, kind: EventKind }

enum EventKind {
    Svc { id, args, result },                 // system call, with decoded arguments
    Ipc { session, service, command, sizes, result, outcome },   // stubbed / deferred / done
    Wake { thread, reason },  Schedule { core, from, to },
    Map { va, size, perms, kind },  Fault { address, write, action },
    ServiceNote { .. },  GuestLog { .. },  Lifecycle { .. },
}
```

- **One recording point per boundary.** The system-call dispatcher, the service framework's
  dispatch, the scheduler's switch and the address space's mutators each call the recorder once.
  Individual system calls and services never log their own entry and exit.
- **`seq` is assigned under the kernel lock**, so sorting by it reproduces the kernel's order.
- **Sinks** consume the same events:
  - the flight recorder (below), always on;
  - a full binary log, when recording for replay;
  - the golden-trace text format from the test strategy, which is a per-thread view of this
    stream with timing removed.
- **Budget:** well under 100 ns per event with only the flight recorder attached. A title makes
  about 1,400 system calls a frame, so this is noise.
- **GPU register writes are not events.** At several million a second they go to the GPU's own
  capture format, off by default.

### Flight recorder

A fixed-size ring per core holding the most recent events (a few thousand each). It costs no
allocation and no I/O while running, and is written out on a guest crash, an emulator error, a
timeout, or on request. It is the answer to "what happened just before this?" for every failure,
including ones that can't be reproduced.

## Replaying the kernel and services

To re-run the HLE side from a log and reach the same state, everything nondeterministic it
consumes must come from the log. That makes three design rules:

1. **All nondeterminism enters through one injectable environment.** The clock, random numbers,
   host file and network I/O, and input state are read through an `Env` trait. Live runs record
   what it returned; replays feed the recording back. Kernel and service code never reads the
   host clock or the file system directly. A lint rule bans the obvious calls in those crates.
2. **Guest memory the host reads is part of the log** in full-recording mode: IPC messages and
   the buffers a service reads. The flight recorder keeps only sizes and hashes.
3. **The order in which deferred replies complete is recorded**, since it comes from other
   threads (GPU, timers, I/O workers).

A replay is then: build a kernel, feed it the recorded calls in `seq` order, and compare each
result with the recorded one. A mismatch is itself a finding: it means something nondeterministic
slipped past the rules above. Any point in the replay can be inspected in an ordinary debugger,
and a log from a homebrew run is a regression test.

## Snapshots

A snapshot is everything needed to resume a run: guest memory (only pages that were touched),
every thread's saved context, the kernel arena, each service's state, and later GPU state.

- **Kernel:** the arena refers to objects by id, so serialising it is direct.
- **Services:** each session keeps its state in a plain serialisable struct. Shared state is
  serialised once and re-linked by id. Deferred-reply tokens are ids, not pointers.
- **Format:** versioned; a snapshot from another build is refused, not misread.
- **When:** on request, periodically while recording, and on crash where possible.
- **What they are for:** skipping minutes of boot to reach a crash; handing a failing state to an
  agent; bisecting when state went wrong. Resuming on a native backend is not deterministic: a
  snapshot gets you close to a bug, not to the same instruction.

Retrofitting snapshots into an emulator is notoriously painful. The rule that services keep
serialisable state is cheap only if it exists before the first service is written.

## Deterministic interpreter backend

Specified separately (#47). What this spec needs from it:

- Runs on any host, so CI and cloud agents can run guest programs.
- One host thread, time slices counted in instructions, a virtual clock: the same inputs always
  give the same run, so full record-and-replay and stepping backwards by re-running become
  possible.
- The hypervisor backend on a real Mac is its oracle: run one instruction or a short sequence on
  both and compare registers and memory. Every mismatch is a small, self-contained bug report.

## Headless runs

One command-line contract for people, agents and CI:

```text
starquake run <program>
    --script input.toml         scripted input, by frame or by time
    --clock fixed --seed N      fixed real-time clock and random seed
    --until frames=600 | trace="<pattern>" | timeout=60s
    --record ring | full | off
    --snapshot-every 30s
    --artifacts <dir>
```

Exit codes: 0 reached the stop condition, 1 the guest crashed, 2 emulator error, 3 timeout,
4 an unimplemented feature was hit under the strict policy.

The artifacts folder always contains:

| File | Contents |
|---|---|
| `summary.json` | Exit reason, duration, frames, counts per system call and service command, every stubbed command hit |
| `trace.txt` | The golden-trace view of the event stream |
| `events.bin` | Flight recorder dump, or the full log when recording |
| `crash.json`, `crash.txt` | When the guest crashed (below) |
| `frames/` | PNG frame dumps on request or at the end |
| `snapshots/` | When requested |

No window is needed for any of this. Configuration comes from flags and one file, not from
scattered environment variables.

## Crash reports

Written as data first and text second:

- Registers and the exception details (needs the richer exit information in #36).
- A backtrace by frame pointer, each frame as module plus offset, with symbol names where the
  module has them.
- The faulting address's memory block, and the full memory map.
- The last events from the flight recorder for the crashing thread and for all threads.
- Build id of each module, the emulator's version, and the run's command line.

## Later

- **GPU capture and replay**, and a render backend that records draw calls instead of drawing,
  so GPU-path tests run without a GPU. Specified with the GPU (#12).
- **A debugger stub** speaking the GDB remote protocol: breakpoints, single-step, register and
  memory access, and write watchpoints built on the stage-1 write-protection mechanism. It must be
  scriptable, so an agent can drive it without a terminal.

## Storage and what may be committed

- Logs, captures and snapshots from commercial titles contain their data. They are never
  committed and never uploaded; the default artifacts folder is outside the repository.
- Committed fixtures come from homebrew runs or are reduced by hand until they contain nothing
  from a title.
- Full logs and snapshots are large. Each run has a size cap, and old artifact folders are
  cleaned up the same way build folders are.

## What runs where

| | Linux CI, cloud agents | Mac with hypervisor | With titles (local only) |
|---|---|---|---|
| Kernel and service replay from homebrew logs | Yes | Yes | Yes, from local logs |
| Guest programs on the interpreter | Yes | Yes | No |
| Guest programs natively | No | Yes | Yes |
| Differential tests, interpreter against hardware | No | Yes | No |
| Title bring-up | No | No | Yes |

## Order of work

1. The event type, the recorder interface and the flight recorder (before the kernel's first
   port slice, so every slice records from the start).
2. The `Env` trait and the lint rule.
3. Artifacts folder, exit codes and `summary.json` in the command-line app.
4. Kernel replay from a log, as soon as milestone 1's hello-world produces one.
5. Crash reports.
6. Snapshots: kernel and memory first, services as they are written.
7. The interpreter backend (#47), in parallel from step 1.

## Decisions

Approved by the owner on 2026-11-11 as written. Where a decision names a recommendation, the
recommendation stands.

1. **One event stream with an always-on flight recorder**, and no ad hoc logging at boundaries.
2. **All nondeterminism through an injectable environment**, enforced by lint, so the kernel and
   services can be replayed exactly.
3. **Snapshots designed in now**: serialisable kernel and service state as a standing rule. This
   adds `serde` and a binary format as dependencies.
4. **Build the deterministic interpreter backend** (details in #47). It is the largest cost here
   and the only way guest-code tests run in CI.
5. **The headless contract above** (flags, exit codes, artifacts) as the one way runs are driven.
6. **Title-derived artifacts never leave the machine**; only homebrew or hand-reduced fixtures are
   committed.
